"""Map bus-bay clusters from one or more GTFS feeds and polygon layers.

Draws one PNG per cluster with numbered stops and a legend of each stop's
routes and directions. Stops are joined to
polygons by coordinates, never by cross-agency stop IDs or stop codes.
Identifiers remain text and are namespaced by feed_id. Nearby records from
different feeds are reported for review, not silently merged into one bay.

By default this is an all-service inventory, not a claim that every listed
route runs on the same day. Set SERVICE_DATE to apply calendar.txt plus
calendar_dates.txt independently to each feed. Direction labels default to
the literal GTFS direction_id; 0/1 are not inferred compass directions.

Inputs
------
- One or more GTFS feeds (folder or .zip): stops, routes, trips and
  stop_times, plus agency, calendar, calendar_dates and feed_info if present.
- One or more cluster polygon layers (.shp, a .zip holding a shapefile, or a
  GeoPackage) with a defined CRS and an optional name field.
- Optional local backdrop layer (streets, parcels); no online tiles are fetched.

Outputs
-------
Each run writes a new ``bus_bay_maps_<timestamp>`` folder in OUTPUT_DIR:
- maps/*.png: one map per cluster.
- cluster_stop_assignments.csv: one row per numbered stop in the map legends.
- route_direction_assignments.csv: stop, route and direction detail, with
  headsigns and trip counts.
- near_miss_stops.csv and cross_feed_nearby_stops.csv: spatial-review lists.
- cluster_index.csv, run.log, run_manifest.json and rerun_config.json.
- bus_bay_cluster_mapper_runlog.txt: run-log sidecar with the verbatim
  CONFIGURATION block, the settings used and input fingerprints.

Typical usage
-------------
Edit CONFIGURATION or run: python bus_bay_cluster_mapper.py --config settings.json
Relative JSON paths resolve against the JSON file, not the working directory.
A run's rerun_config.json can be passed back with --config to repeat it.
Dependencies: pandas, geopandas, shapely, pyproj, numpy, matplotlib.
Standalone: no repository checkout, internet connection or vehicle blocks needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import re
import sys
import textwrap
import uuid
import zipfile
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from pyproj import CRS
from shapely.geometry import box

matplotlib.use("Agg")
LOGGER = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION — EDIT HERE
# =============================================================================
# === BEGIN CONFIG ===
GTFS_SOURCES = [
    {"feed_id": "Connector", "path": r"Path\To\connector_gtfs.zip"},
    {"feed_id": "WMATA", "path": r"Path\To\wmata_gtfs.zip"},
]
# Each source can be a .shp, a ZIP containing one shapefile, or a GeoPackage.
# For multiple shapefiles inside one ZIP, give zip_member for the desired .shp.
# Repeated names within ONE layer are dissolved into one cluster. Names in
# separate sources stay distinct. Set name_field="" to use feature numbers.
CLUSTER_SOURCES = [
    {"source_id": "Bays", "path": r"Path\To\stop_clusters.shp", "name_field": "GEO_NAME"},
]
OUTPUT_DIR = r"Path\To\Output"
SERVICE_DATE = ""  # YYYYMMDD or YYYY-MM-DD; blank = all supplied service IDs.
CLUSTER_NAMES: list[str] = []  # Blank = every cluster; match names exactly.
ROUTE_TYPES: list[str] = []  # Blank = all modes; ["3"] restricts to standard bus.
INCLUDE_UNSERVED_STOPS = True
INCLUDE_NON_PASSENGER_VISITS = False  # False excludes pickup=1 AND drop_off=1.
NEAR_MISS_FEET = 130.0  # Report outside stops this close to a cluster; do not add them.
CROSS_FEED_REVIEW_FEET = 50.0  # Flag close records; never auto-merge agencies.
SHORT_BAY_LABELS = True  # "Bay C" on map where possible; full names in legend/CSV.
FIGURE_DPI = 160
BACKDROP_PATH = ""  # Optional local street/parcel layer; no online tiles are fetched.
# Optional exact direction display labels: "feed_id|route_id|direction_id": label.
# Example: {"Connector|101|0": "Southbound"}. Unlisted values retain 0/1/? labels.
DIRECTION_LABELS: dict[str, str] = {}
# Every output must be traceable: a failed run-log write aborts the script.
# Set to False only when writing to a genuinely read-only location.
REQUIRE_RUN_LOG = True
# === END CONFIG ===

METERS_PER_FOOT = 0.3048
PALETTE = ("#17689B", "#B44D22", "#347C50", "#8055A3", "#A27900", "#327B80")
MARKERS = ("o", "s", "^", "D", "P", "v")
DETAIL_COLUMNS = [
    "feed_id",
    "stop_id",
    "route_id",
    "route_name",
    "agency_id",
    "agency_name",
    "direction_id",
    "direction_label",
    "route_direction_label",
    "trip_headsigns",
    "trip_count",
    "visit_count",
    "pickup_types",
    "drop_off_types",
]


@dataclass
class Settings:
    """Inputs and plotting choices; paths are resolved before a run starts."""

    gtfs_sources: list[dict[str, str]]
    cluster_sources: list[dict[str, str]]
    output_dir: str
    service_date: str = ""
    cluster_names: list[str] = field(default_factory=list)
    route_types: list[str] = field(default_factory=list)
    include_unserved_stops: bool = True
    include_non_passenger_visits: bool = False
    near_miss_feet: float = 130.0
    cross_feed_review_feet: float = 50.0
    short_bay_labels: bool = True
    figure_dpi: int = 160
    backdrop_path: str = ""
    direction_labels: dict[str, str] = field(default_factory=dict)
    require_run_log: bool = True


def require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    """Fail with the table name when a required field is missing."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{context}: missing column(s): {sorted(missing)}")


def unique_ids(frame: pd.DataFrame, column: str, context: str) -> None:
    """Reject empty or duplicated primary identifiers without changing their text."""
    require_columns(frame, [column], context)
    if frame[column].str.strip().eq("").any() or frame[column].duplicated().any():
        raise ValueError(f"{context}: {column} must be populated and unique.")


def natural_key(value: object) -> tuple:
    """Sort numeric route and bay fragments numerically, retaining leading zeros."""
    return tuple(
        (0, int(p)) if p.isdigit() else (1, p.casefold()) for p in re.split(r"(\d+)", str(value))
    )


def iter_rows(frame: pd.DataFrame, index: bool = False) -> Iterator[Any]:
    """Expose dynamically named pandas row fields across pandas/geopandas stubs."""
    yield from frame.itertuples(index=index)


class GTFSReader:
    """Read a GTFS directory or ZIP with an optional wrapper directory."""

    def __init__(self, path: str) -> None:
        """Index table names without extracting the archive."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.members: dict[str, list[str]] = {}
        if self.path.is_file():
            if not zipfile.is_zipfile(self.path):
                raise ValueError(f"Not a GTFS directory or ZIP: {self.path}")
            with zipfile.ZipFile(self.path) as archive:
                for name in archive.namelist():
                    if not name.endswith("/") and "__MACOSX" not in Path(name).parts:
                        self.members.setdefault(Path(name).name, []).append(name)

    def has(self, name: str) -> bool:
        """Return whether a table exists, rejecting ambiguous duplicate members."""
        if self.path.is_dir():
            return (self.path / name).is_file()
        matches = self.members.get(name, [])
        if len(matches) > 1:
            raise ValueError(f"{self.path.name}: multiple {name} files: {matches}")
        return bool(matches)

    @contextmanager
    def open(self, name: str) -> Generator[IO[bytes], None, None]:
        """Keep an archive member open for the lifetime of a streaming read."""
        if not self.has(name):
            raise FileNotFoundError(f"{self.path}: {name}")
        if self.path.is_dir():
            with (self.path / name).open("rb") as handle:
                yield handle
        else:
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(self.members[name][0]) as member:
                    yield member

    def table(self, name: str, optional: bool = False) -> pd.DataFrame:
        """Read text fields without interpreting literal NA or leading zeros."""
        if optional and not self.has(name):
            return pd.DataFrame()
        with self.open(name) as handle:
            return pd.read_csv(handle, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    def stop_time_chunks(self, chunk_size: int = 200_000) -> Iterator[pd.DataFrame]:
        """Stream only stop-visit fields instead of loading an entire large feed."""
        columns = {"trip_id", "stop_id", "pickup_type", "drop_off_type"}
        with self.open("stop_times.txt") as handle:
            yield from pd.read_csv(
                handle,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
                usecols=lambda c: c in columns,
                chunksize=chunk_size,
            )


def active_service_ids(reader: GTFSReader, date: str) -> set[str] | None:
    """Apply weekly calendars and add/remove exceptions for one service date."""
    if not date:
        return None
    day = dt.datetime.strptime(date.replace("-", ""), "%Y%m%d").date()
    token = day.strftime("%Y%m%d")
    calendar = reader.table("calendar.txt", optional=True)
    exceptions = reader.table("calendar_dates.txt", optional=True)
    if calendar.empty and exceptions.empty:
        raise ValueError(f"{reader.path.name}: date filtering requires a service calendar.")
    active: set[str] = set()
    if not calendar.empty:
        weekday = day.strftime("%A").lower()
        require_columns(calendar, ["service_id", "start_date", "end_date", weekday], "calendar.txt")
        unique_ids(calendar, "service_id", "calendar.txt")
        active.update(
            calendar.loc[
                calendar.start_date.le(token)
                & calendar.end_date.ge(token)
                & calendar[weekday].eq("1"),
                "service_id",
            ]
        )
    if not exceptions.empty:
        require_columns(exceptions, ["service_id", "date", "exception_type"], "calendar_dates.txt")
        if exceptions.duplicated(["service_id", "date"]).any():
            raise ValueError(f"{reader.path.name}: duplicate service/date exceptions.")
        selected = exceptions[exceptions.date.eq(token)]
        if not selected.exception_type.isin(["1", "2"]).all():
            raise ValueError(f"{reader.path.name}: invalid calendar exception_type.")
        active.update(selected.loc[selected.exception_type.eq("1"), "service_id"])
        active.difference_update(selected.loc[selected.exception_type.eq("2"), "service_id"])
    return active


def service_span(reader: GTFSReader) -> tuple[str, str]:
    """Describe the feed's published span, falling back to calendar bounds."""
    info = reader.table("feed_info.txt", optional=True)
    if not info.empty and {"feed_start_date", "feed_end_date"}.issubset(info.columns):
        first = str(info.iloc[0].feed_start_date)
        last = str(info.iloc[0].feed_end_date)
        if first and last:
            return first, last
    calendar = reader.table("calendar.txt", optional=True)
    exceptions = reader.table("calendar_dates.txt", optional=True)
    starts = calendar.get("start_date", pd.Series(dtype=str)).tolist()
    ends = calendar.get("end_date", pd.Series(dtype=str)).tolist()
    if not exceptions.empty:
        adds = exceptions.loc[exceptions.exception_type.eq("1"), "date"].tolist()
        starts.extend(adds)
        ends.extend(adds)
    return min(starts, default=""), max(ends, default="")


def load_stops(source: dict[str, str]) -> gpd.GeoDataFrame:
    """Load physical stop/platform records and retain feed-local identifiers."""
    reader = GTFSReader(source["path"])
    stops = reader.table("stops.txt")
    require_columns(stops, ["stop_id", "stop_name", "stop_lat", "stop_lon"], "stops.txt")
    unique_ids(stops, "stop_id", f"{source['feed_id']} stops.txt")
    for name in ("stop_code", "location_type"):
        if name not in stops:
            stops[name] = ""
    stops = stops[stops.location_type.isin(["", "0"])].copy()
    lat = pd.to_numeric(stops.stop_lat, errors="coerce")
    lon = pd.to_numeric(stops.stop_lon, errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    if not valid.all():
        bad = stops.loc[~valid, "stop_id"].head(10).tolist()
        raise ValueError(f"{source['feed_id']}: invalid stop coordinates for {bad}")
    stops["stop_lat"], stops["stop_lon"] = lat, lon
    stops["feed_id"] = source["feed_id"]
    stops["stop_key"] = source["feed_id"] + "::" + stops.stop_id
    stops["stop_name"] = stops.stop_name.where(stops.stop_name.ne(""), stops.stop_id)
    return gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(lon, lat), crs=4326)


def vector_uri(path: str, zip_member: str = "") -> str:
    """Resolve a local vector layer, including a zipped shapefile."""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if candidate.suffix.lower() != ".zip":
        return str(candidate)
    with zipfile.ZipFile(candidate) as archive:
        shapes = [n for n in archive.namelist() if n.lower().endswith(".shp")]
    if zip_member:
        if zip_member not in shapes:
            raise ValueError(f"{candidate.name}: no shapefile member {zip_member!r}.")
        selected = zip_member
    elif len(shapes) == 1:
        selected = shapes[0]
    else:
        raise ValueError(f"{candidate.name}: set zip_member to one of {shapes}.")
    return f"zip://{candidate.resolve().as_posix()}!{selected}"


def load_clusters(sources: list[dict[str, str]], names: list[str]) -> gpd.GeoDataFrame:
    """Dissolve same-name features within each source, preserving separate layers."""
    pieces = []
    for source in sources:
        layer = gpd.read_file(vector_uri(source["path"], source.get("zip_member", "")))
        if layer.crs is None:
            raise ValueError(f"{source['source_id']}: cluster layer has no CRS.")
        if layer.empty or layer.geometry.is_empty.any() or layer.geometry.isna().any():
            raise ValueError(f"{source['source_id']}: empty cluster geometry/layer.")
        if not layer.geom_type.isin(["Polygon", "MultiPolygon"]).all() or not layer.is_valid.all():
            raise ValueError(
                f"{source['source_id']}: valid Polygon/MultiPolygon geometry is required."
            )
        name_field = source.get("name_field", "")
        if name_field and name_field not in layer.columns:
            raise ValueError(f"{source['source_id']}: missing name field {name_field!r}.")
        labels = (
            layer[name_field].fillna("").astype(str).str.strip()
            if name_field
            else pd.Series("", index=layer.index)
        )
        layer["cluster_name"] = [value or f"Cluster {i + 1}" for i, value in enumerate(labels)]
        dissolved = (
            layer[["cluster_name", "geometry"]]
            .dissolve(by="cluster_name", as_index=False)
            .to_crs(4326)
        )
        dissolved["source_id"] = source["source_id"]
        dissolved["cluster_id"] = source["source_id"] + "::" + dissolved.cluster_name
        pieces.append(dissolved)
    result = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=4326)
    if names:
        missing = set(names) - set(result.cluster_name)
        if missing:
            raise ValueError(f"Unknown cluster names: {sorted(missing)}")
        result = result[result.cluster_name.isin(names)].copy()
    return gpd.GeoDataFrame(
        result.sort_values(["source_id", "cluster_name"]).reset_index(drop=True),
        geometry="geometry",
        crs=4326,
    )


def route_direction_inventory(
    source: dict[str, str],
    stop_ids: set[str],
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream selected visits and aggregate feed-specific route/direction assignments."""
    reader = GTFSReader(source["path"])
    routes, trips = reader.table("routes.txt"), reader.table("trips.txt")
    unique_ids(routes, "route_id", "routes.txt")
    unique_ids(trips, "trip_id", "trips.txt")
    require_columns(trips, ["route_id", "service_id"], "trips.txt")
    if not trips.route_id.isin(routes.route_id).all():
        raise ValueError(f"{source['feed_id']}: trips reference missing routes.")
    for column in ("route_short_name", "route_long_name", "agency_id"):
        if column not in routes:
            routes[column] = ""
    for column in ("direction_id", "trip_headsign"):
        if column not in trips:
            trips[column] = ""
    if not trips.direction_id.isin(["", "0", "1"]).all():
        LOGGER.warning(
            "%s has direction IDs outside 0/1; retaining literal labels.", source["feed_id"]
        )
    agency = reader.table("agency.txt", optional=True)
    agencies: dict[str, str] = {}
    if not agency.empty:
        if "agency_id" not in agency:
            agency["agency_id"] = ""
        agencies = dict(zip(agency.agency_id, agency.agency_name))
        if len(agency) == 1:
            routes["agency_id"] = routes.agency_id.replace("", str(agency.iloc[0].agency_id))
        elif routes.agency_id.eq("").any():
            raise ValueError(f"{source['feed_id']}: multi-agency routes need agency_id.")
    routes["agency_name"] = routes.agency_id.map(agencies).fillna(source["feed_id"])
    routes["route_name"] = routes.route_short_name.where(
        routes.route_short_name.ne(""), routes.route_long_name
    )
    routes["route_name"] = routes.route_name.where(routes.route_name.ne(""), routes.route_id)
    routes["route_display"] = routes.route_name
    repeated_names = routes.duplicated(["agency_id", "route_name"], keep=False)
    routes.loc[repeated_names, "route_display"] = (
        routes.loc[repeated_names, "route_name"]
        + " ["
        + routes.loc[repeated_names, "route_id"]
        + "]"
    )
    if len(set(routes.agency_id)) > 1:
        routes["route_display"] = routes.agency_name + ": " + routes.route_display
    active = active_service_ids(reader, settings.service_date)
    selected = trips if active is None else trips[trips.service_id.isin(active)]
    if settings.route_types:
        require_columns(routes, ["route_type"], "routes.txt")
        allowed_routes = set(routes.loc[routes.route_type.isin(settings.route_types), "route_id"])
        selected = selected[selected.route_id.isin(allowed_routes)]
    selected = selected[["trip_id", "route_id", "direction_id", "trip_headsign"]]
    start, end = service_span(reader)
    metadata = {
        "feed_id": source["feed_id"],
        "path": source["path"],
        "feed_start": start,
        "feed_end": end,
        "total_trip_records": len(trips),
        "selected_trip_records": len(selected),
    }
    if selected.empty:
        LOGGER.warning("%s has no selected trips for this date/filter.", source["feed_id"])
    keep_trips, known_trips = set(selected.trip_id), set(trips.trip_id)
    chunks = []
    rows_read = 0
    for chunk in reader.stop_time_chunks():
        require_columns(chunk, ["trip_id", "stop_id"], "stop_times.txt")
        rows_read += len(chunk)
        matching = chunk[chunk.stop_id.isin(stop_ids)].copy()
        if not matching.trip_id.isin(known_trips).all():
            raise ValueError(f"{source['feed_id']}: cluster stop visits reference unknown trips.")
        matching = matching[matching.trip_id.isin(keep_trips)].copy()
        for column in ("pickup_type", "drop_off_type"):
            if column not in matching:
                matching[column] = "0"
            matching[column] = matching[column].replace("", "0")
        if not settings.include_non_passenger_visits:
            matching = matching[~(matching.pickup_type.eq("1") & matching.drop_off_type.eq("1"))]
        if not matching.empty:
            chunks.append(matching)
    LOGGER.info("%s: read %s stop-time rows in chunks.", source["feed_id"], f"{rows_read:,}")
    if not chunks:
        return pd.DataFrame(columns=DETAIL_COLUMNS), metadata
    visits = pd.concat(chunks, ignore_index=True).merge(
        selected, on="trip_id", validate="many_to_one"
    )
    visits = visits.merge(
        routes[["route_id", "route_name", "route_display", "agency_id", "agency_name"]],
        on="route_id",
        validate="many_to_one",
    )
    records = []
    keys = ["stop_id", "route_id", "direction_id"]
    for values, group in visits.groupby(keys, sort=False, dropna=False):
        assert isinstance(values, tuple)
        sid, rid, direction = values
        first = group.iloc[0]
        label = settings.direction_labels.get(
            f"{source['feed_id']}|{rid}|{direction}", direction or "?"
        )
        records.append(
            {
                "feed_id": source["feed_id"],
                "stop_id": sid,
                "route_id": rid,
                "route_name": first.route_name,
                "agency_id": first.agency_id,
                "agency_name": first.agency_name,
                "direction_id": direction,
                "direction_label": label,
                "route_direction_label": f"{first.route_display} ({label})",
                "trip_headsigns": " | ".join(sorted(set(group.trip_headsign) - {""})),
                "trip_count": group.trip_id.nunique(),
                "visit_count": len(group),
                "pickup_types": ",".join(sorted(set(group.pickup_type))),
                "drop_off_types": ",".join(sorted(set(group.drop_off_type))),
            }
        )
    metadata["selected_cluster_visits"] = len(visits)
    metadata["frequency_table_present"] = reader.has("frequencies.txt")
    return pd.DataFrame(records, columns=DETAIL_COLUMNS), metadata


def build_assignment_tables(
    memberships: gpd.GeoDataFrame,
    inventory: pd.DataFrame,
    settings: Settings,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Create one authoritative stop legend and a normalized assignment table."""
    summaries = []
    for values, group in inventory.groupby(["feed_id", "stop_id"], sort=False):
        assert isinstance(values, tuple)
        feed_id, stop_id = values
        labels = sorted(set(group.route_direction_label), key=natural_key)
        summaries.append(
            {
                "feed_id": feed_id,
                "stop_id": stop_id,
                "route_directions": "; ".join(labels),
                "agency_names": "; ".join(sorted(set(group.agency_name))),
            }
        )
    summary = pd.DataFrame(
        summaries, columns=["feed_id", "stop_id", "route_directions", "agency_names"]
    )
    stops = memberships.merge(
        summary, on=["feed_id", "stop_id"], how="left", validate="many_to_one"
    )
    if not settings.include_unserved_stops:
        stops = stops[stops.route_directions.notna()].copy()
    stops["route_directions"] = stops.route_directions.fillna("No qualifying trips")
    stops["agency_names"] = stops.agency_names.fillna("")
    order = sorted(
        stops.index,
        key=lambda i: (
            stops.loc[i, "cluster_id"],
            stops.loc[i, "feed_id"],
            natural_key(stops.loc[i, "stop_name"]),
            natural_key(stops.loc[i, "stop_id"]),
        ),
    )
    stops = stops.loc[order].reset_index(drop=True)
    stops["map_ref"] = (
        stops.groupby("cluster_id", sort=False).cumcount().add(1).map(lambda n: f"{n:02d}")
    )
    stops["service_date"] = settings.service_date or "ALL_SERVICES"
    stops["map_file"] = stops.cluster_id.map(map_filename)
    stops["map_label"] = [
        f"{row.map_ref} {short_stop_name(row.stop_name, settings.short_bay_labels)}"
        for row in iter_rows(stops)
    ]
    base = stops.drop(columns=["geometry", "route_directions", "agency_names"])
    detail = base.merge(inventory, on=["feed_id", "stop_id"], how="inner", validate="many_to_many")
    return gpd.GeoDataFrame(stops, geometry="geometry", crs=memberships.crs), detail


def map_filename(cluster_id: str) -> str:
    """Create a stable filename even when two source layers reuse cluster names."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", cluster_id).strip("_")[:100] or "cluster"
    digest = hashlib.sha256(cluster_id.encode()).hexdigest()[:8]
    return f"{slug}_{digest}.png"


def short_stop_name(name: str, short_bays: bool) -> str:
    """Use a recognizable bay label while preserving full names in the legend."""
    match = re.search(r"\bBAY[\s+_-]*([A-Za-z0-9]+)\b", name, flags=re.IGNORECASE)
    if short_bays and match:
        return "Bay " + match.group(1).upper()
    return "\n".join(textwrap.wrap(name, width=28, break_long_words=False, break_on_hyphens=False))


def local_crs(geometry: Any) -> CRS:
    """Use a local UTM projection; distances are measured in metres, reported in feet."""
    crs = gpd.GeoSeries([geometry], crs=4326).estimate_utm_crs()
    if crs is None:
        raise ValueError("Could not select a local projected CRS for this cluster.")
    return CRS.from_user_input(crs)


def spatial_review(
    clusters: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    memberships: gpd.GeoDataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag excluded nearby stops and possible shared locations without merging."""
    near_rows, shared_rows = [], []
    for cluster in iter_rows(clusters):
        crs = local_crs(cluster.geometry)
        polygon = gpd.GeoSeries([cluster.geometry], crs=4326).to_crs(crs).iloc[0]
        projected = stops.to_crs(crs)
        inside = memberships[memberships.cluster_id.eq(cluster.cluster_id)]
        if settings.near_miss_feet > 0:
            distances = projected.distance(polygon)
            near = projected[
                ~projected.stop_key.isin(inside.stop_key)
                & distances.le(settings.near_miss_feet * METERS_PER_FOOT)
            ]
            for point in near.itertuples():
                near_rows.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "cluster_name": cluster.cluster_name,
                        "feed_id": point.feed_id,
                        "stop_id": point.stop_id,
                        "stop_code": point.stop_code,
                        "stop_name": point.stop_name,
                        "distance_ft": round(
                            float(distances.loc[point.Index]) / METERS_PER_FOOT, 1
                        ),
                    }
                )
        members = projected[projected.stop_key.isin(inside.stop_key)]
        records = list(iter_rows(members))
        for i, first in enumerate(records):
            for second in records[i + 1 :]:
                if first.feed_id == second.feed_id:
                    continue
                distance = first.geometry.distance(second.geometry)
                if distance <= settings.cross_feed_review_feet * METERS_PER_FOOT:
                    shared_rows.append(
                        {
                            "cluster_id": cluster.cluster_id,
                            "cluster_name": cluster.cluster_name,
                            "feed_id_a": first.feed_id,
                            "stop_id_a": first.stop_id,
                            "stop_code_a": first.stop_code,
                            "stop_name_a": first.stop_name,
                            "feed_id_b": second.feed_id,
                            "stop_id_b": second.stop_id,
                            "stop_code_b": second.stop_code,
                            "stop_name_b": second.stop_name,
                            "distance_ft": round(float(distance) / METERS_PER_FOOT, 1),
                        }
                    )
    return (
        pd.DataFrame(
            near_rows,
            columns=[
                "cluster_id",
                "cluster_name",
                "feed_id",
                "stop_id",
                "stop_code",
                "stop_name",
                "distance_ft",
            ],
        ),
        pd.DataFrame(
            shared_rows,
            columns=[
                "cluster_id",
                "cluster_name",
                "feed_id_a",
                "stop_id_a",
                "stop_code_a",
                "stop_name_a",
                "feed_id_b",
                "stop_id_b",
                "stop_code_b",
                "stop_name_b",
                "distance_ft",
            ],
        ),
    )


def overlap_area(first: Bbox, second: Bbox) -> float:
    """Return rectangle intersection area in display pixels."""
    hit = Bbox.intersection(first, second)
    return 0.0 if hit is None else float(hit.width * hit.height)


def place_labels(ax: Axes, points: gpd.GeoDataFrame, colors: dict[str, str]) -> int:
    """Place short labels greedily in display space and connect them to true coordinates."""
    canvas = ax.figure.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    canvas.draw()
    renderer = canvas.get_renderer()
    occupied: list[Bbox] = []
    for point in iter_rows(points):
        x, y = ax.transData.transform((point.geometry.x, point.geometry.y))
        occupied.append(Bbox.from_extents(x - 6, y - 6, x + 6, y + 6))
    # Keep the scale and north arrow readable as well.
    for bounds in ((0.02, 0.01, 0.38, 0.12), (0.02, 0.82, 0.15, 0.99)):
        a = ax.transAxes.transform(bounds[:2])
        b = ax.transAxes.transform(bounds[2:])
        occupied.append(Bbox.from_extents(*a, *b))
    warnings = 0
    for point in iter_rows(points):
        x, y = point.geometry.x, point.geometry.y
        origin = ax.transData.transform((x, y))
        label = ax.text(
            x,
            y,
            point.map_label,
            fontsize=8.5,
            color=colors[point.feed_id],
            zorder=6,
            va="center",
            bbox={"facecolor": "white", "alpha": 0.94, "edgecolor": "none", "pad": 1.5},
        )
        best: tuple[float, float, float, str, Bbox] | None = None
        for radius in (13, 24, 38, 55, 78, 108, 145, 190, 240):
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)):
                alignment = "left" if dx >= 0 else "right"
                position = ax.transData.inverted().transform(origin + np.array([dx, dy]) * radius)
                label.set_position((float(position[0]), float(position[1])))
                label.set_horizontalalignment(alignment)
                bbox = label.get_window_extent(renderer).expanded(1.05, 1.20)
                collisions = sum(overlap_area(bbox, other) for other in occupied)
                outside = bbox.width * bbox.height - overlap_area(bbox, ax.bbox)
                penalty = 1_000_000 * (collisions + max(0.0, outside)) + radius * math.hypot(dx, dy)
                candidate = (penalty, float(position[0]), float(position[1]), alignment, bbox)
                if best is None or penalty < best[0]:
                    best = candidate
            if best is not None and best[0] < 1_000_000:
                break
        assert best is not None
        _, tx, ty, alignment, bbox = best
        label.set_position((tx, ty))
        label.set_horizontalalignment(alignment)
        occupied.append(bbox)
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(tx, ty),
            zorder=4,
            arrowprops={"arrowstyle": "-", "color": "#657783", "lw": 0.65},
        )
        warnings += int(best[0] >= 1_000_000)
    return warnings


def render_map(
    cluster: Any,
    points: gpd.GeoDataFrame,
    settings: Settings,
    path: Path,
    feed_metadata: list[dict[str, Any]],
    backdrop: gpd.GeoDataFrame | None,
) -> int:
    """Draw the cluster, numbered stop/name labels, and the CSV-backed route legend."""
    feeds = [source["feed_id"] for source in settings.gtfs_sources]
    colors = {feed: PALETTE[i % len(PALETTE)] for i, feed in enumerate(feeds)}
    markers = {feed: MARKERS[i % len(MARKERS)] for i, feed in enumerate(feeds)}
    entries = []
    total_lines = 0
    for stop in iter_rows(points):
        heading = (
            f"{stop.map_ref}   {stop.feed_id}   code {stop.stop_code or '—'}   ID {stop.stop_id}"
        )
        name = textwrap.wrap(stop.stop_name, 57, break_long_words=False, break_on_hyphens=False)
        routes = textwrap.wrap(
            stop.route_directions, 57, break_long_words=False, break_on_hyphens=False
        )
        headings = textwrap.wrap(heading, 52, break_long_words=True, break_on_hyphens=False)
        entries.append((stop.feed_id, headings, name, routes))
        total_lines += len(headings) + len(name) + len(routes) + 1
    height = max(8.4, (total_lines * 12.0 + 160) / 72)
    fig = Figure(figsize=(17, height), dpi=settings.figure_dpi, facecolor="white")
    FigureCanvasAgg(fig)
    fig.text(
        0.035, 1 - 0.38 / height, cluster.cluster_name, fontsize=18, weight="bold", color="#243746"
    )
    scope = (
        f"Service date {settings.service_date}"
        if settings.service_date
        else "All supplied service IDs — route inventory, not a same-day schedule"
    )
    fig.text(0.035, 1 - 0.70 / height, scope, fontsize=10, color="#536877")
    map_height = min(height - 2.0, 9.8)
    ax = fig.add_axes((0.035, 1 - (1.15 + map_height) / height, 0.575, map_height / height))
    ax.set_facecolor("#F4F7F8")
    crs = local_crs(cluster.geometry)
    polygon = gpd.GeoSeries([cluster.geometry], crs=4326).to_crs(crs)
    local = points.to_crs(crs)
    xmin, ymin, xmax, ymax = polygon.total_bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    width, span_y = max(xmax - xmin, 80), max(ymax - ymin, 80)
    limits = (cx - width * 0.72, cy - span_y * 0.72, cx + width * 0.72, cy + span_y * 0.72)
    if backdrop is not None:
        local_background = backdrop.to_crs(crs)
        clipped = local_background[local_background.intersects(box(*limits))].copy()
        if not clipped.empty:
            clipped.geometry = clipped.intersection(box(*limits))
            clipped.plot(ax=ax, color="#D4DCE0", edgecolor="#C5CDD2", linewidth=0.6, zorder=1)
    polygon.plot(
        ax=ax, facecolor="#E6EEF2", edgecolor="#8198A6", linewidth=1.1, alpha=0.75, zorder=2
    )
    for feed in feeds:
        group = local[local.feed_id.eq(feed)]
        if not group.empty:
            ax.scatter(
                group.geometry.x,
                group.geometry.y,
                s=53,
                marker=markers[feed],
                c=colors[feed],
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )
    ax.set_xlim(limits[0], limits[2])
    ax.set_ylim(limits[1], limits[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")  # GeoPandas adds Easting/Northing labels that overlap the footer.
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_color("#C7D2D8")
    ax.annotate(
        "N",
        xy=(0.075, 0.94),
        xytext=(0.075, 0.83),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        color="#243746",
        arrowprops={"arrowstyle": "-|>", "color": "#243746", "lw": 1.3},
    )
    target = (limits[2] - limits[0]) * 0.23 / METERS_PER_FOOT
    power = 10 ** math.floor(math.log10(target))
    scale_ft = max(value * power for value in (0.1, 0.2, 0.5, 1, 2, 5) if value * power <= target)
    scale = scale_ft * METERS_PER_FOOT
    sx, sy = limits[0] + (limits[2] - limits[0]) * 0.07, limits[1] + (limits[3] - limits[1]) * 0.055
    ax.plot([sx, sx + scale], [sy, sy], color="#243746", linewidth=2.2, zorder=7)
    ax.annotate(
        f"{scale_ft:g} ft",
        (sx + scale / 2, sy),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    warnings = place_labels(ax, local, colors)
    if points.empty:
        ax.text(
            0.5,
            0.5,
            "No stops match the selected inputs",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
        )
    legend_x, y = 0.645, 1 - 1.14 / height
    fig.text(
        legend_x, y, "Stops and routes (direction)", fontsize=12, weight="bold", color="#243746"
    )
    y -= 0.27 / height
    for feed, headings, names, route_lines in entries:
        for heading in headings:
            fig.text(
                legend_x, y, heading, fontsize=9.6, weight="bold", color=colors[feed], va="top"
            )
            y -= 0.19 / height
        for line in names:
            fig.text(legend_x, y, line, fontsize=9, color="#334753", va="top")
            y -= 0.17 / height
        for line in route_lines:
            fig.text(legend_x, y, line, fontsize=9, color="#182F3E", va="top")
            y -= 0.17 / height
        y -= 0.14 / height
    periods = []
    for item in feed_metadata:
        dates = [
            f"{value[:4]}-{value[4:6]}-{value[6:8]}" if len(value) == 8 else "unspecified"
            for value in (item["feed_start"], item["feed_end"])
        ]
        periods.append(f"{item['feed_id']}: {dates[0]} to {dates[1]}")
    fig.text(
        0.035, 0.61 / height, "Feed periods — " + "; ".join(periods), fontsize=9, color="#536877"
    )
    fig.text(
        0.035,
        0.39 / height,
        "0/1 are GTFS direction IDs; ? is unspecified. Full headsigns are in the detail CSV. "
        "Points remain at their source coordinates.",
        fontsize=9,
        color="#536877",
    )
    fig.text(
        0.035,
        0.17 / height,
        "Cluster polygon shown. Feed-specific records are not auto-merged. "
        "Map references match cluster_stop_assignments.csv.",
        fontsize=9,
        color="#536877",
    )
    fig.savefig(path, dpi=settings.figure_dpi, facecolor="white")
    fig.clear()
    return warnings


def input_fingerprints(sources: list[dict[str, str]]) -> dict[str, str]:
    """Fingerprint source archives, GTFS tables and shapefile companion files."""
    result = {}
    for source in sources:
        path = Path(source["path"])
        files = list(path.glob("*.txt")) if path.is_dir() else [path]
        if path.suffix.lower() == ".shp":
            files = [p for p in path.parent.glob(path.stem + ".*") if p.is_file()]
        for file in files:
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            result[str(file)] = digest.hexdigest()
    return result


# Canonical version lives in utils/run_log.py -- keep this copy in sync.
def extract_config_block(source_file: Path) -> str:
    r"""Return the text between the CONFIG markers in *source_file*.

    Reads ``source_file`` as UTF-8 text and slices out the lines strictly
    *between* the first occurrence of ``# === BEGIN CONFIG ===`` and the first
    subsequent occurrence of ``# === END CONFIG ===``.  The marker lines
    themselves are excluded; whitespace and inline comments inside the block
    are preserved verbatim.

    Args:
        source_file: Path to the Python source file to scan (typically
            ``Path(__file__)`` from the calling script).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    _BEGIN = "# === BEGIN CONFIG ==="
    _END = "# === END CONFIG ==="

    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == _BEGIN:
            begin_idx = i
        elif begin_idx is not None and stripped == _END:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. Expected '{_BEGIN}' and '{_END}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def write_run_log(output: Path, settings: Settings, fingerprints: dict[str, str]) -> bool:
    """Write ``bus_bay_cluster_mapper_runlog.txt`` into the run's output folder.

    The log captures the CONFIGURATION block verbatim, the settings actually
    used (which --config JSON or CLI flags may have changed), and input
    fingerprints.

    Args:
        output: The run's output folder.
        settings: The resolved settings for this run.
        fingerprints: Input path to SHA-256 digest, from :func:`input_fingerprints`.

    Returns:
        ``True`` if the log was written, ``False`` otherwise.
    """
    log_path = output / "bus_bay_cluster_mapper_runlog.txt"
    try:
        source_file: Path | None = Path(__file__).resolve()
    except NameError:  # Pasted into a notebook cell: no source file on disk.
        source_file = None
    if source_file is None:
        config_text = "(config block unavailable: interactive session, no __file__ on disk)"
    else:
        try:
            config_text = extract_config_block(source_file)
        except (OSError, ValueError) as exc:
            LOGGER.error("Could not extract config block for run log: %s", exc)
            return False
    lines = [
        "=" * 72,
        "BUS BAY CLUSTER MAPPER RUN LOG",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Output folder:    {output}",
        f"Source script:    {source_file or '<interactive>'}",
        "",
        "-" * 72,
        "EFFECTIVE SETTINGS (CONFIGURATION, --config JSON or CLI flags, as resolved)",
        "-" * 72,
        json.dumps(asdict(settings), indent=2),
        "",
        "-" * 72,
        "INPUT FINGERPRINTS (SHA-256)",
        "-" * 72,
        *(f"{path}\n  sha256: {digest}" for path, digest in fingerprints.items()),
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]
    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Error writing run log: %s", exc)
        return False
    LOGGER.info("Run log saved to %s", log_path)
    return True


def is_placeholder(path: str) -> bool:
    r"""Return whether *path* still holds the template's ``Path\To`` placeholder."""
    return bool(re.search(r"(^|[\\/])path[\\/]to([\\/]|$)", str(path), flags=re.IGNORECASE))


def validate_settings(settings: Settings) -> None:
    """Reject ambiguous source identifiers and invalid plotting/filter settings."""
    for sources, key in (
        (settings.gtfs_sources, "feed_id"),
        (settings.cluster_sources, "source_id"),
    ):
        identifiers = [s.get(key, "") for s in sources]
        if not sources or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Provide one or more sources with distinct {key} values.")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in identifiers):
            raise ValueError(f"{key} values must use letters, numbers, underscores or hyphens.")
        if any("path" not in source for source in sources):
            raise ValueError("Every source needs a path.")
    # Check inputs before the run folder is created, so a mistake leaves no empty output.
    problems = [
        f"{label} {source[key]!r}: {source['path']}"
        for label, sources, key in (
            ("GTFS_SOURCES", settings.gtfs_sources, "feed_id"),
            ("CLUSTER_SOURCES", settings.cluster_sources, "source_id"),
        )
        for source in sources
        if not Path(source["path"]).exists()
    ]
    if settings.backdrop_path and not Path(settings.backdrop_path).exists():
        problems.append(f"BACKDROP_PATH: {settings.backdrop_path}")
    if is_placeholder(settings.output_dir):
        problems.append(f"OUTPUT_DIR: {settings.output_dir}")
    if problems:
        raise ValueError(
            "Edit the CONFIGURATION section (or the --config JSON) before running. "
            "Not found or still a placeholder: " + "; ".join(problems)
        )
    if settings.service_date:
        settings.service_date = dt.datetime.strptime(
            settings.service_date.replace("-", ""), "%Y%m%d"
        ).strftime("%Y-%m-%d")
    if (
        settings.figure_dpi < 72
        or settings.near_miss_feet < 0
        or settings.cross_feed_review_feet < 0
    ):
        raise ValueError("DPI must be >=72 and review distances nonnegative.")


def run(settings: Settings) -> Path:
    """Export cluster maps, matching assignments and a reproducible run manifest."""
    validate_settings(settings)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = Path(settings.output_dir).resolve() / f"bus_bay_maps_{run_id}"
    maps = output / "maps"
    maps.mkdir(parents=True)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        clusters = load_clusters(settings.cluster_sources, settings.cluster_names)
        frames = [load_stops(source) for source in settings.gtfs_sources]
        stops = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
        memberships = gpd.sjoin(
            stops,
            clusters[["cluster_id", "cluster_name", "source_id", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")
        if memberships.duplicated(["cluster_id", "feed_id", "stop_id"]).any():
            raise ValueError("Duplicate membership after dissolving cluster polygons.")
        if memberships.stop_key.duplicated().any():
            LOGGER.warning(
                "Some stop records belong to overlapping clusters; retained separately per cluster."
            )
        LOGGER.info(
            "%d clusters; %d feed-specific stop memberships.", len(clusters), len(memberships)
        )
        inventories, feed_metadata = [], []
        for source in settings.gtfs_sources:
            selected = set(memberships.loc[memberships.feed_id.eq(source["feed_id"]), "stop_id"])
            inventory, metadata = route_direction_inventory(source, selected, settings)
            inventories.append(inventory)
            feed_metadata.append(metadata)
        inventory = pd.concat(inventories, ignore_index=True)
        legend, detail = build_assignment_tables(memberships, inventory, settings)
        valid_spans = [
            (f["feed_start"], f["feed_end"])
            for f in feed_metadata
            if f["feed_start"] and f["feed_end"]
        ]
        no_common_dates = len(valid_spans) > 1 and max(s for s, _ in valid_spans) > min(
            e for _, e in valid_spans
        )
        if no_common_dates:
            LOGGER.warning(
                "Feed periods have no shared date. All-service maps are an inventory "
                "across feed versions."
            )
        near, shared = spatial_review(clusters, stops, memberships, settings)
        near.to_csv(output / "near_miss_stops.csv", index=False)
        shared.to_csv(output / "cross_feed_nearby_stops.csv", index=False)
        if len(near):
            LOGGER.warning(
                "%d nearby stop/cluster pairs fall outside the polygons; see near_miss_stops.csv.",
                len(near),
            )
        if len(shared):
            LOGGER.warning(
                "%d cross-feed pairs are close enough for review; no records were merged.",
                len(shared),
            )
        columns = [
            "cluster_id",
            "cluster_name",
            "source_id",
            "map_file",
            "map_ref",
            "feed_id",
            "agency_names",
            "stop_id",
            "stop_code",
            "stop_name",
            "map_label",
            "stop_lat",
            "stop_lon",
            "route_directions",
            "service_date",
        ]
        legend[columns].to_csv(output / "cluster_stop_assignments.csv", index=False)
        detail.drop(columns=[c for c in detail if c not in columns + DETAIL_COLUMNS]).to_csv(
            output / "route_direction_assignments.csv", index=False
        )
        backdrop = None
        if settings.backdrop_path:
            backdrop = gpd.read_file(vector_uri(settings.backdrop_path))
            if backdrop.crs is None:
                raise ValueError("The backdrop layer has no CRS.")
        cluster_rows = []
        for i, cluster in enumerate(iter_rows(clusters), 1):
            selected = legend[legend.cluster_id.eq(cluster.cluster_id)]
            filename = map_filename(cluster.cluster_id)
            LOGGER.info(
                "Map %d/%d: %s (%d stops)", i, len(clusters), cluster.cluster_name, len(selected)
            )
            warnings = render_map(
                cluster, selected, settings, maps / filename, feed_metadata, backdrop
            )
            if warnings:
                LOGGER.warning(
                    "%s: %d label placements need visual review.", cluster.cluster_name, warnings
                )
            cluster_rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.cluster_name,
                    "map_file": filename,
                    "stops": len(selected),
                    "route_direction_assignments": int(
                        detail.cluster_id.eq(cluster.cluster_id).sum()
                    ),
                    "label_review_count": warnings,
                }
            )
        pd.DataFrame(cluster_rows).to_csv(output / "cluster_index.csv", index=False)
        sources = settings.gtfs_sources + settings.cluster_sources
        if settings.backdrop_path:
            sources = sources + [{"path": settings.backdrop_path}]
        fingerprints = input_fingerprints(sources)
        manifest = {
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "configuration": asdict(settings),
            "feeds": feed_metadata,
            "input_sha256": fingerprints,
            "feed_periods_have_no_common_date": no_common_dates,
            "clusters": cluster_rows,
            "notes": [
                "Stop IDs and stop codes are never matched across feeds.",
                "Counts are trip definitions, not frequency-expanded departures.",
                "All-service inventory may combine multiple service versions.",
                "Map and legend CSV use the same stop references and route strings.",
            ],
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "rerun_config.json").write_text(
            json.dumps(asdict(settings), indent=2), encoding="utf-8"
        )
        if not write_run_log(output, settings, fingerprints) and settings.require_run_log:
            raise RuntimeError(
                "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
                "error when a sidecar file is genuinely impossible."
            )
        LOGGER.info("Finished: %s", output)
        return output
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def parse_settings(argv: Sequence[str] | None = None) -> Settings:
    """Read the editable configuration or JSON, including notebook-safe arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="JSON settings; paths resolve relative to this file."
    )
    parser.add_argument("--output", help="Override the output parent directory.")
    parser.add_argument("--service-date", help="YYYY-MM-DD; an empty string selects all services.")
    parser.add_argument(
        "--cluster", action="append", help="Exact cluster name; repeat to select several."
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if argv is None and "ipykernel" in sys.modules and "-f" in args:
        index = args.index("-f")
        del args[index : index + 2]
    parsed = parser.parse_args(args)
    if parsed.config:
        base = parsed.config.resolve().parent
        data = json.loads(parsed.config.read_text(encoding="utf-8"))
        for key in ("gtfs_sources", "cluster_sources"):
            for source in data[key]:
                source["path"] = str((base / Path(source["path"])).resolve())
        for key in ("output_dir", "backdrop_path"):
            if data.get(key):
                data[key] = str((base / Path(data[key])).resolve())
        settings = Settings(**data)
    else:
        settings = Settings(
            gtfs_sources=GTFS_SOURCES,
            cluster_sources=CLUSTER_SOURCES,
            output_dir=OUTPUT_DIR,
            service_date=SERVICE_DATE,
            cluster_names=CLUSTER_NAMES,
            route_types=ROUTE_TYPES,
            include_unserved_stops=INCLUDE_UNSERVED_STOPS,
            include_non_passenger_visits=INCLUDE_NON_PASSENGER_VISITS,
            near_miss_feet=NEAR_MISS_FEET,
            cross_feed_review_feet=CROSS_FEED_REVIEW_FEET,
            short_bay_labels=SHORT_BAY_LABELS,
            figure_dpi=FIGURE_DPI,
            backdrop_path=BACKDROP_PATH,
            direction_labels=DIRECTION_LABELS,
            require_run_log=REQUIRE_RUN_LOG,
        )
    if parsed.output:
        settings.output_dir = parsed.output
    if parsed.service_date is not None:
        settings.service_date = parsed.service_date
    if parsed.cluster:
        settings.cluster_names = parsed.cluster
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mapper with concise configuration-error reporting."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        run(parse_settings(argv))
    except (ValueError, OSError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
