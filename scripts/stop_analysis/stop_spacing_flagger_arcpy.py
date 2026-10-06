"""GTFS to GIS pipeline for stop-spacing QA and segment analysis (ArcPy version).

This module converts a General Transit Feed Specification (GTFS) package
(directory or .zip) into projected ESRI Shapefiles suitable for spatial analysis
and provides quality assurance (QA) checks on stop spacing.

Outputs:
* Shapefiles for served stops, route polylines, and stop-to-stop segments
* Logs flagging consecutive served stops that are spaced too closely
* CSVs identifying potential “missed” stops located between long stop-to-stop gaps

The long-spacing check examines whether stops from other routes fall within a
specified buffer distance of unusually long segments and may merit further
review as possible missed service opportunities.

Spacing is measured along each distinct stopping pattern: the ordered stops
of a trip in ``stop_times.txt``, on the shape that trip uses. Express and
local trips sharing a shape are measured separately, and a stop visited twice
(a loop) keeps both visits. Each stop is placed along the shape from
``shape_dist_traveled`` when both files provide it; otherwise the stops are
matched to the shape in trip order.

Typical usage:
Update the paths in the CONFIGURATION section and run from ArcGIS Pro's Python
window or a shell whose environment provides ``arcpy`` (an ArcGIS Pro install).
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Sequence, Tuple

import arcpy
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# GTFS source – folder containing *.txt or a .zip GTFS package.
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"

# Output folder for shapefiles and QA logs (NOT a geodatabase).
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

# Route filtering
FILTER_OUT_LIST: list[str] = []
INCLUDE_ROUTE_IDS: list[str] = ["101", "202", "303"]  # empty list → all routes except filtered-out

# Route geometry options – union each route/direction's shapes in routes.shp
# only; spacing is always measured along each stopping pattern's own shape.
ROUTE_UNION: bool = False

# Projected CRS in any linear unit; distances are reported in feet.
# Example: 2248 = NAD83 / Maryland (ftUS)
PROJECTED_WKID: int = 2248

# A pattern's stop farther than this from its place on the shape is skipped;
# 100 m (328 ft) is the GTFS Best Practices stop-to-shape limit.
SERVED_STOP_MAX_OFFSET_FT: float = 328.0

# direction_id is optional in GTFS; trips without a 0/1 value are analyzed
# under this direction instead of being dropped.
UNKNOWN_DIRECTION_ID: int = -1

# Short-spacing QA – “too close” consecutive served stops along a route
MIN_SPACING_FT: float = 400.0
SPACING_LOG_FILE: str = "short_spacing_segments.txt"

# Long-spacing QA – “too long” gaps and potential missed stops
LONG_SPACING_FT: float = 1_500.0
NEAR_BUFFER_FT: float = 99.0
LONG_SPACING_LOG_FILE: str = "long_spacing_segments.txt"  # currently unused (CSV + summary)
LONG_SPACING_CSV_FILE: str = "long_spacing_segments.csv"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# HELPERS – I/O AND BASIC GTFS HANDLING
# =============================================================================


def _ensure_output_folder(folder: str | Path) -> Path:
    """Create (if necessary) and return the output folder as a Path."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_gtfs_tables(gtfs_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Load the five core GTFS tables into DataFrames.

    Args:
        gtfs_path: Path to either a directory containing *.txt files or a .zip GTFS.

    Every column is read as text and blanks stay empty strings, so IDs keep
    leading zeros and values such as "NA"; _prepare_tables converts the numeric
    fields afterwards.

    Returns:
        Mapping of table name to DataFrame with keys:
        "stops", "routes", "trips", "stop_times", "shapes".

    Raises:
        ValueError: If the path is neither a folder nor a .zip GTFS.
    """
    gtfs = Path(gtfs_path)
    filenames: Dict[str, str] = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
    }

    def _read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    if gtfs.is_dir():
        logging.info("Detected GTFS directory at %s", gtfs)
        return {k: _read(gtfs / v) for k, v in filenames.items()}

    if gtfs.is_file() and gtfs.suffix.lower() == ".zip":
        logging.info("Detected GTFS zip at %s – extracting to temporary directory …", gtfs)
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        tables = {k: _read(root / v) for k, v in filenames.items()}
        return tables

    raise ValueError("GTFS_PATH must be a folder or a .zip file.")


def _validate_columns(dfs: Dict[str, pd.DataFrame]) -> None:
    """Raise ValueError if any required GTFS column is missing."""
    required: Dict[str, set[str]] = {
        "stops": {"stop_id", "stop_lat", "stop_lon", "stop_name"},
        "routes": {"route_id", "route_short_name"},
        "trips": {"trip_id", "route_id", "shape_id"},
        "stop_times": {"trip_id", "stop_id", "stop_sequence"},
        "shapes": {
            "shape_id",
            "shape_pt_sequence",
            "shape_pt_lat",
            "shape_pt_lon",
        },
    }

    missing_msgs: list[str] = []
    for tbl, needed in required.items():
        present = set(dfs[tbl].columns)
        missing = needed - present
        if missing:
            missing_msgs.append(f"{tbl}.txt → missing {', '.join(sorted(missing))}")

    if missing_msgs:
        joined = "\n".join(" • " + msg for msg in missing_msgs)
        raise ValueError(f"GTFS validation failed – required columns not found:\n{joined}")


def _prepare_tables(
    dfs: Dict[str, pd.DataFrame],
    unknown_direction: int = UNKNOWN_DIRECTION_ID,
) -> None:
    """Convert the numeric GTFS fields in place; IDs stay text.

    Unparseable numbers become NaN. trips.direction_id becomes an int, with
    unknown_direction for trips whose value is blank, invalid or absent (the
    field is optional in GTFS).
    """
    numeric: Dict[str, List[str]] = {
        "stops": ["stop_lat", "stop_lon"],
        "stop_times": ["stop_sequence", "shape_dist_traveled"],
        "shapes": ["shape_pt_sequence", "shape_pt_lat", "shape_pt_lon", "shape_dist_traveled"],
    }
    for tbl, cols in numeric.items():
        for col in cols:
            if col in dfs[tbl].columns:
                dfs[tbl][col] = pd.to_numeric(dfs[tbl][col], errors="coerce")

    trips = dfs["trips"]
    raw = trips["direction_id"] if "direction_id" in trips.columns else pd.Series("", trips.index)
    direction = pd.to_numeric(raw, errors="coerce")
    unknown = ~direction.isin([0, 1])
    if unknown.any():
        logging.info(
            "%d of %d trips have no valid direction_id; analyzing them as direction %d.",
            unknown.sum(),
            len(trips),
            unknown_direction,
        )
    trips["direction_id"] = direction.where(~unknown, unknown_direction).astype(int)


def _filter_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    include_ids: Sequence[str | int],
    exclude_ids: Sequence[str | int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply include/exclude lists and return filtered routes and trips.

    Args:
        routes: routes.txt DataFrame.
        trips: trips.txt DataFrame.
        include_ids: Route IDs to include. Empty means include all.
        exclude_ids: Route IDs to drop.

    Returns:
        (routes_filtered, trips_filtered)
    """
    exclude = {str(rid) for rid in exclude_ids}
    include = {str(rid) for rid in include_ids}
    routes_ok = routes.loc[~routes["route_id"].isin(exclude)].copy()
    if include:
        routes_ok = routes_ok.loc[routes_ok["route_id"].isin(include)].copy()

    trips_ok = trips.loc[trips["route_id"].isin(routes_ok["route_id"])].copy()
    return routes_ok, trips_ok


def _get_projected_sr(wkid: int) -> arcpy.SpatialReference:
    """Return the projected spatial reference for the given WKID.

    Raises:
        ValueError: If the WKID is unknown or not a projected coordinate
            system (degrees are not a distance unit).
    """
    try:
        sr = arcpy.SpatialReference(wkid)
    except RuntimeError as err:
        raise ValueError(f"Spatial reference WKID {wkid} is not recognized.") from err
    if sr.name == "Unknown":
        raise ValueError(f"Spatial reference WKID {wkid} is not recognized.")
    if sr.type != "Projected":
        raise ValueError(
            f"Spatial reference {sr.name} (WKID {wkid}) is not projected; set "
            "PROJECTED_WKID to a projected coordinate system (e.g. 2248)."
        )
    return sr


def _feet_factor(sr: arcpy.SpatialReference) -> float:
    """Return factor to convert from SR linear units to feet.

    Read from the SR's unit, so any projected unit works: about 1.000002 for
    US survey feet (e.g. WKID 2248) and 3.28084 for metres (e.g. WKID 26918).
    """
    return sr.metersPerUnit / 0.3048


def _is_empty_polyline(geom: arcpy.Polyline | None) -> bool:
    """Return True if a Polyline is None, has no points, or has zero length."""
    if geom is None:
        return True
    if getattr(geom, "pointCount", 0) == 0:
        return True
    if getattr(geom, "length", 0.0) == 0.0:
        return True
    return False


# =============================================================================
# GEOMETRY BUILDERS (ARCPY)
# =============================================================================


def _build_stop_geometries(
    stops_df: pd.DataFrame,
    projected_sr: arcpy.SpatialReference,
) -> Dict[str, arcpy.PointGeometry]:
    """Build projected PointGeometry objects keyed by stop_id.

    Args:
        stops_df: stops.txt DataFrame (must contain stop_id, stop_lat, stop_lon).
        projected_sr: Target projected spatial reference.

    Returns:
        Mapping stop_id → PointGeometry (projected).
    """
    required = {"stop_id", "stop_lat", "stop_lon"}
    missing = required - set(stops_df.columns)
    if missing:
        raise ValueError(f"stops.txt missing columns: {', '.join(sorted(missing))}")

    wgs84 = arcpy.SpatialReference(4326)
    out: Dict[str, arcpy.PointGeometry] = {}

    for _, row in stops_df.iterrows():
        stop_id = str(row["stop_id"])
        try:
            lon = float(row["stop_lon"])
            lat = float(row["stop_lat"])
        except (TypeError, ValueError):
            lon = lat = float("nan")
        if not (np.isfinite(lon) and np.isfinite(lat)):
            logging.warning("Skipping stop %s due to invalid coords.", stop_id)
            continue

        pt = arcpy.Point(lon, lat)
        pt_geom = arcpy.PointGeometry(pt, wgs84).projectAs(projected_sr)
        out[stop_id] = pt_geom

    logging.info("Built %d stop point geometries.", len(out))
    return out


def _build_shape_geometries(
    shapes_df: pd.DataFrame,
    projected_sr: arcpy.SpatialReference,
) -> Tuple[Dict[str, arcpy.Polyline], Dict[str, np.ndarray]]:
    """Build projected Polyline geometries keyed by shape_id.

    Args:
        shapes_df: shapes.txt DataFrame.
        projected_sr: Target projected spatial reference.

    Returns:
        (lines, vertex_dists): mapping shape_id → Polyline in projected_sr,
        and shape_id → each vertex's shape_dist_traveled for shapes whose
        values are present and never decrease.
    """
    required = {
        "shape_id",
        "shape_pt_sequence",
        "shape_pt_lat",
        "shape_pt_lon",
    }
    missing = required - set(shapes_df.columns)
    if missing:
        raise ValueError(f"shapes.txt missing columns: {', '.join(sorted(missing))}")

    wgs84 = arcpy.SpatialReference(4326)
    out: Dict[str, arcpy.Polyline] = {}
    vertex_dists: Dict[str, np.ndarray] = {}
    has_dists = "shape_dist_traveled" in shapes_df.columns

    shapes = shapes_df.copy()
    shapes["shape_pt_sequence"] = pd.to_numeric(
        shapes["shape_pt_sequence"],
        errors="coerce",
    )

    shapes_sorted = shapes.sort_values(["shape_id", "shape_pt_sequence"])

    for shape_id, group in shapes_sorted.groupby("shape_id"):
        array = arcpy.Array()
        dists: List[float] = []
        for _, row in group.iterrows():
            try:
                lon = float(row["shape_pt_lon"])
                lat = float(row["shape_pt_lat"])
            except (TypeError, ValueError):
                lon = lat = float("nan")
            if not (np.isfinite(lon) and np.isfinite(lat)):
                logging.warning("Skipping bad shape point in shape_id=%s", shape_id)
                continue
            pt = arcpy.Point(lon, lat)
            array.add(pt)
            dists.append(float(row["shape_dist_traveled"]) if has_dists else float("nan"))

        if array.count < 2:
            logging.debug("Shape %s has fewer than 2 points; skipping.", shape_id)
            continue

        line_wgs = arcpy.Polyline(array, wgs84)
        line_proj = line_wgs.projectAs(projected_sr)
        out[str(shape_id)] = line_proj
        dist_arr = np.asarray(dists)
        if np.isfinite(dist_arr).all() and (np.diff(dist_arr) >= 0).all():
            vertex_dists[str(shape_id)] = dist_arr

    logging.info("Built %d shape polylines.", len(out))
    return out, vertex_dists


def _build_routes_from_shapes(
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    shape_geoms: Dict[str, arcpy.Polyline],
    union_shapes: bool,
) -> List[Dict[str, Any]]:
    """Build route polylines keyed by (route_id, direction_id), for routes.shp.

    Args:
        trips_df: Filtered trips.txt DataFrame.
        routes_df: Filtered routes.txt DataFrame.
        shape_geoms: Mapping shape_id → Polyline (projected).
        union_shapes: If True, union all shapes for (route_id, direction_id)
            into a single polyline; otherwise keep individual shapes.

    Returns:
        List of route records:
        {
            "route_id": str,
            "direction_id": int,
            "route_short": str | None,
            "geometry": arcpy.Polyline,
        }
    """
    trips = trips_df.loc[trips_df["shape_id"] != ""].copy()

    # NEW: collapse to unique combinations so we do not duplicate per trip.
    trips = trips.drop_duplicates(subset=["route_id", "direction_id", "shape_id"]).copy()
    logging.info(
        "Routes – using %d unique (route_id, direction_id, shape_id) combinations.",
        len(trips),
    )

    # Route short name lookup
    route_short_lookup = (
        routes_df[["route_id", "route_short_name"]]
        .assign(route_id_str=lambda df: df["route_id"].astype(str))
        .set_index("route_id_str")["route_short_name"]
        .to_dict()
    )

    records: List[Dict[str, Any]] = []

    if not union_shapes:
        for _, row in trips.iterrows():
            shape_id = str(row["shape_id"])
            line = shape_geoms.get(shape_id)
            if line is None:
                logging.debug("Missing geometry for shape_id=%s; skipping trip.", shape_id)
                continue

            rid = str(row["route_id"])
            try:
                drn = int(row["direction_id"])
            except (TypeError, ValueError):
                logging.warning("Bad direction_id for route_id=%s; skipping.", rid)
                continue

            rshort = route_short_lookup.get(rid)
            records.append(
                {
                    "route_id": rid,
                    "direction_id": drn,
                    "route_short": rshort,
                    "geometry": line,
                }
            )
        logging.info("Routes – built %d route-shape records.", len(records))
        return records

    # Union shapes per (route_id, direction_id)
    grouped = trips.groupby(["route_id", "direction_id"])["shape_id"].apply(
        lambda s: sorted(set(str(x) for x in s))
    )

    for (rid, drn_val), shape_ids in grouped.items():
        lines: List[arcpy.Polyline] = []
        for sid in shape_ids:
            line = shape_geoms.get(sid)
            if line is not None:
                lines.append(line)

        if not lines:
            logging.debug(
                "No geometries found for route_id=%s, direction_id=%s; skipping.", rid, drn_val
            )
            continue

        geom_union = lines[0]
        for line in lines[1:]:
            geom_union = geom_union.union(line)

        try:
            drn = int(drn_val)
        except (TypeError, ValueError):
            logging.warning("Bad direction_id=%s for route_id=%s; skipping union.", drn_val, rid)
            continue

        rshort = route_short_lookup.get(str(rid))
        records.append(
            {
                "route_id": str(rid),
                "direction_id": drn,
                "route_short": rshort,
                "geometry": geom_union,
            }
        )

    logging.info("Routes – built %d unioned route polylines.", len(records))
    return records


# =============================================================================
# STOP AGGREGATION (ROUTE/DIRECTION LISTS)
# =============================================================================


def _build_stop_aggregates(
    dfs: Dict[str, pd.DataFrame],
    trips_selected: pd.DataFrame,
    routes_selected: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return stop layers for all routes and for the filtered subset.

    Args:
        dfs: Dictionary of raw GTFS tables.
        trips_selected: Trips that survived the include/exclude filter.
        routes_selected: Routes that survived the include/exclude filter.

    Returns:
        (all_stops_df, selected_stops_df) where each DataFrame contains:
        stop_id, stop_name, stop_lat, stop_lon, route_id, direction_id,
        route_short_name, route_dirs. The three *_id/name columns are Python
        lists with normalized types (route_id → str, direction_id → int,
        route_short_name → str); route_dirs lists the actual (route_id,
        direction_id) pairs, which the separate lists cannot reconstruct.
    """
    stops = dfs["stops"]
    stop_times = dfs["stop_times"]

    def _agg_for(trips_df: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
        served = stop_times.loc[stop_times["trip_id"].isin(trips_df["trip_id"])]

        trip_attrs = trips_df[["trip_id", "route_id", "direction_id"]].merge(
            routes_df[["route_id", "route_short_name"]],
            on="route_id",
            how="left",
        )
        merged = served[["trip_id", "stop_id"]].merge(
            trip_attrs,
            on="trip_id",
            how="left",
        )

        merged["direction_id"] = pd.to_numeric(
            merged["direction_id"],
            errors="coerce",
        )
        merged = merged.drop_duplicates(["stop_id", "route_id", "direction_id"])
        merged["route_dirs"] = [
            (str(rid), int(drn)) for rid, drn in zip(merged["route_id"], merged["direction_id"])
        ]

        agg = (
            merged.groupby("stop_id")[
                ["route_id", "direction_id", "route_short_name", "route_dirs"]
            ]
            .agg(lambda s: sorted(set(s.dropna())))
            .reset_index()
        )

        base = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
        out_df = base.merge(agg, on="stop_id", how="inner")

        # Normalize types once here; downstream functions rely on this.
        out_df["route_id"] = out_df["route_id"].apply(lambda vals: [str(v) for v in vals])
        out_df["direction_id"] = out_df["direction_id"].apply(lambda vals: [int(v) for v in vals])
        out_df["route_short_name"] = out_df["route_short_name"].apply(
            lambda vals: [str(v) for v in vals]
        )
        return out_df

    all_trips = dfs["trips"]
    all_routes = dfs["routes"]

    all_stops_df = _agg_for(all_trips, all_routes)
    selected_stops_df = _agg_for(trips_selected, routes_selected)

    logging.info(
        "Stops – all served stops: %d; filtered served stops: %d",
        len(all_stops_df),
        len(selected_stops_df),
    )
    return all_stops_df, selected_stops_df


# =============================================================================
# STOPPING PATTERNS + ORDERED STOPS
# =============================================================================


def _vertices(line: arcpy.Polyline) -> np.ndarray:
    """Return a single-part polyline's vertices as an (n, 2) array."""
    return np.array([(pt.X, pt.Y) for pt in line.getPart(0)], dtype=float).reshape(-1, 2)


def _vertex_along(vertices: np.ndarray) -> np.ndarray:
    """Return each vertex's distance along the polyline (SR units)."""
    steps = np.diff(vertices, axis=0)
    return np.concatenate(([0.0], np.cumsum(np.hypot(steps[:, 0], steps[:, 1]))))


def _dists_along_line(
    line: arcpy.Polyline,
    vertex_dists: np.ndarray | None,
    stop_dists: Sequence[float],
) -> Tuple[float, ...]:
    """Convert stops' shape_dist_traveled to SR distances along line.

    Interpolates between the shape's vertices, whose own values are
    vertex_dists. Returns NaNs when either side lacks the values.
    """
    stop_along = np.asarray(stop_dists, dtype=float)
    vertices = _vertices(line)
    if (
        vertex_dists is None
        or len(vertex_dists) != len(vertices)
        or not np.isfinite(stop_along).all()
    ):
        return tuple(np.full(len(stop_along), np.nan))
    return tuple(np.interp(stop_along, vertex_dists, _vertex_along(vertices)))


def _build_pattern_records(
    stop_times_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    shape_geoms: Dict[str, arcpy.Polyline],
    shape_vertex_dists: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Return one record per distinct stopping pattern.

    A pattern is a (route_id, direction_id, shape_id) and the trip's stop IDs
    in stop_sequence order, repeats included.

    Returns:
        List of records:
        {
            "route_id": str,
            "direction_id": int,
            "route_short": str | None,
            "shape_id": str,
            "stop_ids": tuple of str,
            "stop_dists": tuple of float (SR units along the shape from
                shape_dist_traveled; NaN when the feed lacks it),
            "geometry": arcpy.Polyline,
        }
    """
    st = stop_times_df.loc[stop_times_df["trip_id"].isin(trips_df["trip_id"])]
    st = st.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])
    if "shape_dist_traveled" not in st.columns:
        st = st.assign(shape_dist_traveled=np.nan)

    per_trip = (
        st.groupby("trip_id", sort=False)
        .agg(stop_ids=("stop_id", tuple), stop_dists=("shape_dist_traveled", tuple))
        .reset_index()
        .merge(trips_df[["trip_id", "route_id", "direction_id", "shape_id"]], on="trip_id")
    )
    no_shape = ~per_trip["shape_id"].isin(list(shape_geoms))
    if no_shape.any():
        logging.warning(
            "Skipping %d of %d trips whose shape_id is blank or has no usable shape.",
            no_shape.sum(),
            len(per_trip),
        )
    patterns = (
        per_trip.loc[~no_shape]
        .drop_duplicates(["route_id", "direction_id", "shape_id", "stop_ids"])
        .sort_values(["route_id", "direction_id", "shape_id"], kind="stable")
    )

    route_short_lookup = dict(zip(routes_df["route_id"], routes_df["route_short_name"]))
    records: List[Dict[str, Any]] = []
    for row in patterns.itertuples(index=False):
        line = shape_geoms[row.shape_id]
        records.append(
            {
                "route_id": str(row.route_id),
                "direction_id": int(row.direction_id),
                "route_short": route_short_lookup.get(row.route_id),
                "shape_id": row.shape_id,
                "stop_ids": row.stop_ids,
                "stop_dists": _dists_along_line(
                    line, shape_vertex_dists.get(row.shape_id), row.stop_dists
                ),
                "geometry": line,
            }
        )

    logging.info("Patterns – %d stopping patterns from %d trips.", len(records), len(per_trip))
    return records


class RouteStop(NamedTuple):
    """Representation of a stop ordered along a route polyline."""

    stop_id: str
    stop_name: str | None
    measure: float


def _match_in_order(vertices: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Place points on a polyline in their given order.

    Each point goes to its nearest spot on one segment of the polyline. The
    segments are chosen so they never go backwards from one point to the
    next, and so that the total of each point's distance from the line, plus
    any backtrack between two points on the same segment, is as small as
    possible (a dynamic programme over segments). Unlike projecting each
    point on its own, a stop on a street the route uses twice is placed on
    the pass that fits the trip's order.

    Args:
        vertices: ``(k + 1, 2)`` polyline vertices.
        xy: ``(n, 2)`` points in order.

    Returns:
        ``(along, offset)``: each point's distance along the polyline and its
        distance from that spot.
    """
    start = vertices[:-1]
    vec = np.diff(vertices, axis=0)
    seg_len = np.hypot(vec[:, 0], vec[:, 1])
    seg_from = np.concatenate(([0.0], np.cumsum(seg_len)[:-1]))
    len_sq = np.where(seg_len > 0, seg_len**2, 1.0)

    def _on_segments(pt: np.ndarray, segs: np.ndarray | slice) -> Tuple[np.ndarray, np.ndarray]:
        rel = pt - start[segs]
        t = np.clip((rel * vec[segs]).sum(axis=-1) / len_sq[segs], 0.0, 1.0)
        gap = rel - t[..., None] * vec[segs]
        return seg_from[segs] + t * seg_len[segs], np.hypot(gap[..., 0], gap[..., 1])

    # cost[j]: least total so far with the latest point on segment j
    idx = np.arange(len(seg_len))
    prev_along, cost = _on_segments(xy[0], slice(None))
    back: List[np.ndarray] = []
    for pt in xy[1:]:
        along, offset = _on_segments(pt, slice(None))
        # best over strictly earlier segments, and which segment gave it
        best = np.minimum.accumulate(cost)
        improves = np.concatenate(([True], cost[1:] < best[:-1]))
        best_arg = np.maximum.accumulate(np.where(improves, idx, 0))
        earlier = np.concatenate(([np.inf], best[:-1]))
        earlier_arg = np.concatenate(([0], best_arg[:-1]))
        # or the same segment, paying for any step backwards along it; ties
        # go to the earlier segment, i.e. the first pass that fits
        same = cost + np.maximum(prev_along - along, 0.0)
        back.append(np.where(same < earlier, idx, earlier_arg))
        cost = offset + np.minimum(same, earlier)
        prev_along = along

    segs = [int(np.argmin(cost))]
    for prev_best in reversed(back):
        segs.append(int(prev_best[segs[-1]]))
    segs.reverse()
    return _on_segments(xy, np.asarray(segs))


def _ordered_pattern_stops(
    rec: Dict[str, Any],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    stop_names: Dict[str, str],
    max_offset: float,
) -> List[RouteStop]:
    """Return a pattern's stops in trip order, with measures along its shape.

    The shape_dist_traveled positions in rec["stop_dists"] are used when
    every stop has one, they never decrease, and each lies within max_offset
    of its stop. Otherwise the stops are matched to the shape in trip order
    (see _match_in_order), so a loop or a retraced street keeps the trip's
    sequence. Stops without a geometry or farther than max_offset (SR units)
    from their position are left out.
    """
    line: arcpy.Polyline = rec["geometry"]
    vertices = _vertices(line)
    stop_ids: List[str] = list(rec["stop_ids"])
    geoms = [stop_geoms.get(sid) for sid in stop_ids]
    xy = np.array(
        [(g.firstPoint.X, g.firstPoint.Y) if g is not None else (np.nan, np.nan) for g in geoms],
        dtype=float,
    ).reshape(-1, 2)

    measures = np.asarray(rec.get("stop_dists", [np.nan] * len(stop_ids)), dtype=float)
    use_hint = (
        len(measures) == len(stop_ids)
        and np.isfinite(measures).all()
        and (np.diff(measures) >= 0).all()
    )
    if use_hint:
        along = _vertex_along(vertices)
        at_x = np.interp(measures, along, vertices[:, 0])
        at_y = np.interp(measures, along, vertices[:, 1])
        use_hint = bool((np.hypot(xy[:, 0] - at_x, xy[:, 1] - at_y) <= max_offset).all())

    if not use_hint:
        measures = np.full(len(stop_ids), np.nan)
        usable = [
            i for i, g in enumerate(geoms) if g is not None and line.distanceTo(g) <= max_offset
        ]
        if usable:
            along, offset = _match_in_order(vertices, xy[usable])
            ok = offset <= max_offset
            measures[np.asarray(usable)[ok]] = np.maximum.accumulate(along[ok])

    return [
        RouteStop(stop_id=sid, stop_name=stop_names.get(sid), measure=float(m))
        for sid, m in zip(stop_ids, measures)
        if np.isfinite(m)
    ]


def _consecutive_stops(
    patterns: List[Dict[str, Any]],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    stop_names: Dict[str, str],
    max_offset: float,
) -> Iterator[Tuple[Dict[str, Any], RouteStop, RouteStop]]:
    """Yield (pattern, stop, next_stop) for consecutive stops of each pattern.

    A pair repeated by another pattern of the same route, direction and shape
    (e.g. a local and an express sharing a stretch) is yielded once.
    """
    seen: set = set()
    for rec in patterns:
        if _is_empty_polyline(rec["geometry"]):
            continue
        stops = _ordered_pattern_stops(rec, stop_geoms, stop_names, max_offset)
        for s0, s1 in zip(stops[:-1], stops[1:]):
            key = (rec["route_id"], rec["direction_id"], rec["shape_id"], s0, s1)
            if key not in seen:
                seen.add(key)
                yield rec, s0, s1


# =============================================================================
# EXPORT SHAPEFILES (STOPS, ROUTES, SEGMENTS)
# =============================================================================


def _export_stops_shapefile(
    stops_df: pd.DataFrame,
    stop_geoms: Dict[str, arcpy.PointGeometry],
    sr: arcpy.SpatialReference,
    out_folder: Path,
) -> None:
    """Write filtered served stops to stops.shp."""
    arcpy.env.overwriteOutput = True

    out_name = "stops"
    fc_path = os.path.join(out_folder.as_posix(), out_name + ".shp")

    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        out_folder.as_posix(),
        out_name,
        "POINT",
        spatial_reference=sr,
    )

    # Shapefile field names must be <= 10 characters.
    fields = [
        ("stop_id", "TEXT", 64),
        ("stop_name", "TEXT", 128),
        ("routes", "TEXT", 254),  # comma-separated route_ids
        ("dirs", "TEXT", 254),  # comma-separated direction_ids
        ("rshorts", "TEXT", 254),  # comma-separated route_short_names
    ]
    for name, ftype, length in fields:
        arcpy.management.AddField(
            fc_path,
            name,
            ftype,
            field_length=length if length is not None else None,
        )

    insert_fields = ["stop_id", "stop_name", "routes", "dirs", "rshorts", "SHAPE@"]

    rows_written = 0
    with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
        for row in stops_df.itertuples(index=False):
            sid = str(row.stop_id)
            geom = stop_geoms.get(sid)
            if geom is None:
                logging.debug("No geometry for stop_id=%s; skipping.", sid)
                continue

            routes_str = ",".join(str(r) for r in row.route_id)
            dirs_str = ",".join(str(d) for d in row.direction_id)
            shorts_str = ",".join(str(s) for s in row.route_short_name)

            cursor.insertRow(
                [
                    sid,
                    str(row.stop_name),
                    routes_str,
                    dirs_str,
                    shorts_str,
                    geom,
                ]
            )
            rows_written += 1

    logging.info("Wrote %s (%d features).", fc_path, rows_written)


def _export_routes_shapefile(
    routes: List[Dict[str, Any]],
    sr: arcpy.SpatialReference,
    out_folder: Path,
) -> None:
    """Write route polylines to routes.shp."""
    arcpy.env.overwriteOutput = True

    out_name = "routes"
    fc_path = os.path.join(out_folder.as_posix(), out_name + ".shp")

    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        out_folder.as_posix(),
        out_name,
        "POLYLINE",
        spatial_reference=sr,
    )

    # Field names <= 10 chars.
    fields = [
        ("route_id", "TEXT", 64),
        ("dir", "SHORT", None),
        ("rshort", "TEXT", 64),
    ]
    for name, ftype, length in fields:
        arcpy.management.AddField(
            fc_path,
            name,
            ftype,
            field_length=length if length is not None else None,
        )

    insert_fields = ["route_id", "dir", "rshort", "SHAPE@"]

    rows_written = 0
    with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
        for rec in routes:
            geom: arcpy.Polyline | None = rec.get("geometry")
            if _is_empty_polyline(geom):
                logging.debug(
                    "Skipping empty or null route geometry for route_id=%s, dir=%s",
                    rec.get("route_id"),
                    rec.get("direction_id"),
                )
                continue

            cursor.insertRow(
                [
                    rec["route_id"],
                    int(rec["direction_id"]),
                    rec.get("route_short"),
                    geom,
                ]
            )
            rows_written += 1

    logging.info("Wrote %s (%d features).", fc_path, rows_written)


def _export_segments_shapefile(
    patterns: List[Dict[str, Any]],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    stop_names: Dict[str, str],
    sr: arcpy.SpatialReference,
    out_folder: Path,
    max_offset_ft: float = SERVED_STOP_MAX_OFFSET_FT,
) -> None:
    """Split each pattern's shape between consecutive stops and write segments.shp.

    A pattern's stops farther than max_offset_ft from the shape are skipped,
    and a segment repeated by another pattern on the same shape is written
    once.
    """
    arcpy.env.overwriteOutput = True

    out_name = "segments"
    fc_path = os.path.join(out_folder.as_posix(), out_name + ".shp")

    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        out_folder.as_posix(),
        out_name,
        "POLYLINE",
        spatial_reference=sr,
    )

    # Field names <= 10 chars.
    fields = [
        ("route_id", "TEXT", 64),
        ("dir", "SHORT", None),
        ("rshort", "TEXT", 64),
        ("len_ft", "DOUBLE", None),
    ]
    for name, ftype, length in fields:
        arcpy.management.AddField(
            fc_path,
            name,
            ftype,
            field_length=length if length is not None else None,
        )

    insert_fields = ["route_id", "dir", "rshort", "len_ft", "SHAPE@"]

    ft_factor = _feet_factor(sr)
    max_offset = max_offset_ft / ft_factor
    rows_written = 0

    with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
        for rec, s0, s1 in _consecutive_stops(patterns, stop_geoms, stop_names, max_offset):
            start_m = s0.measure
            end_m = s1.measure
            if end_m <= start_m:
                continue

            line: arcpy.Polyline = rec["geometry"]
            seg_geom = line.segmentAlongLine(start_m, end_m, use_percentage=False)
            if _is_empty_polyline(seg_geom):
                continue

            length_ft = seg_geom.length * ft_factor
            cursor.insertRow(
                [
                    rec["route_id"],
                    int(rec["direction_id"]),
                    rec.get("route_short"),
                    float(length_ft),
                    seg_geom,
                ]
            )
            rows_written += 1

    logging.info("Wrote %s (%d features).", fc_path, rows_written)


# =============================================================================
# QA – SHORT AND LONG SPACING (ARCPY GEOMETRY)
# =============================================================================


def _flag_short_spacing(
    patterns: List[Dict[str, Any]],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    stop_names: Dict[str, str],
    sr: arcpy.SpatialReference,
    threshold_ft: float,
    log_path: Path,
    max_offset_ft: float = SERVED_STOP_MAX_OFFSET_FT,
) -> None:
    """Write a log of consecutive stops spaced closer than threshold_ft.

    Stops are evaluated in trip order along each stopping pattern's shape.
    A pattern's stops farther than max_offset_ft from the shape are skipped.
    """
    ft_factor = _feet_factor(sr)
    max_offset = max_offset_ft / ft_factor
    count = 0

    with log_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "route_id\tdirection_id\tbegin_stop_id\tbegin_stop_name\t"
            "end_stop_id\tend_stop_name\tspacing_ft\n"
        )

        for rec, s0, s1 in _consecutive_stops(patterns, stop_geoms, stop_names, max_offset):
            spacing_ft = (s1.measure - s0.measure) * ft_factor
            if spacing_ft < threshold_ft:
                fh.write(
                    f"{rec['route_id']}\t{int(rec['direction_id'])}\t"
                    f"{s0.stop_id}\t{s0.stop_name}\t"
                    f"{s1.stop_id}\t{s1.stop_name}\t"
                    f"{spacing_ft:.1f}\n"
                )
                count += 1

    logging.info(
        "Wrote short-spacing log → %s (%d flagged segments).",
        log_path.name,
        count,
    )


def _flag_long_spacing_csv(
    patterns: List[Dict[str, Any]],
    all_stops_df: pd.DataFrame,
    stop_geoms: Dict[str, arcpy.PointGeometry],
    stop_names: Dict[str, str],
    sr: arcpy.SpatialReference,
    threshold_ft: float,
    near_buffer_ft: float,
    csv_path: Path,
    summary: bool = True,
    max_offset_ft: float = SERVED_STOP_MAX_OFFSET_FT,
) -> None:
    """Export a CSV of “missed” stops that fill unusually long gaps.

    A long gap is any consecutive pair of stops in a stopping pattern whose
    spacing exceeds threshold_ft. For every stop not served by the pattern's
    (route_id, direction_id) that lies inside the gap and within
    near_buffer_ft of it, a row is written to the CSV. A pattern's stops
    farther than max_offset_ft from the shape are skipped. The CSV and
    summary are rewritten on every run, header only when nothing is
    flagged, so results from an earlier run never linger.
    """
    ft_factor = _feet_factor(sr)
    near_buffer = near_buffer_ft / ft_factor  # SR units
    max_offset = max_offset_ft / ft_factor
    records: List[Dict[str, Any]] = []

    # Stop coordinates aligned with all_stops_df rows, for a quick box filter.
    xs = np.full(len(all_stops_df), np.nan)
    ys = np.full(len(all_stops_df), np.nan)
    for pos, sid in enumerate(all_stops_df["stop_id"]):
        geom = stop_geoms.get(str(sid))
        if geom is not None and geom.firstPoint is not None:
            xs[pos], ys[pos] = geom.firstPoint.X, geom.firstPoint.Y

    for rec, s0, s1 in _consecutive_stops(patterns, stop_geoms, stop_names, max_offset):
        start_m = float(s0.measure)
        end_m = float(s1.measure)
        seg_len_ft = (end_m - start_m) * ft_factor
        if seg_len_ft <= threshold_ft:
            continue

        rid = rec["route_id"]
        drn = int(rec["direction_id"])
        # The gap itself, so a loop's other passes and curves are handled
        line: arcpy.Polyline = rec["geometry"]
        gap = line.segmentAlongLine(start_m, end_m, use_percentage=False)
        ext = gap.extent
        in_box = np.flatnonzero(
            (xs >= ext.XMin - near_buffer)
            & (xs <= ext.XMax + near_buffer)
            & (ys >= ext.YMin - near_buffer)
            & (ys <= ext.YMax + near_buffer)
        )

        for pos in in_box:
            st_row = all_stops_df.iloc[pos]
            # Skip stops served by this route/direction.
            if (rid, drn) in st_row.route_dirs:
                continue

            sid = str(st_row.stop_id)
            pt_geom = stop_geoms[sid]
            proj_m = gap.measureOnLine(pt_geom, use_percentage=False)
            if not (np.isfinite(proj_m) and 0 < proj_m < gap.length):
                continue

            dist_to_route_ft = gap.distanceTo(pt_geom) * ft_factor
            if dist_to_route_ft <= near_buffer_ft:
                records.append(
                    {
                        "route_id": rid,
                        "route_short": rec.get("route_short"),
                        "direction_id": drn,
                        "seg_len_ft": round(seg_len_ft, 1),
                        "start_stop_id": s0.stop_id,
                        "start_stop_name": s0.stop_name or "",
                        "end_stop_id": s1.stop_id,
                        "end_stop_name": s1.stop_name or "",
                        "flagged_stop_id": sid,
                        "flagged_stop_name": str(st_row.stop_name),
                        "dist_to_route_ft": round(dist_to_route_ft, 1),
                    }
                )

    fieldnames = [
        "route_id",
        "route_short",
        "direction_id",
        "seg_len_ft",
        "start_stop_id",
        "start_stop_name",
        "end_stop_id",
        "end_stop_name",
        "flagged_stop_id",
        "flagged_stop_name",
        "dist_to_route_ft",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    if records:
        logging.info("Wrote long-spacing CSV → %s (%d rows).", csv_path.name, len(records))
    else:
        logging.info("No long-spacing issues found; wrote empty %s.", csv_path.name)

    if summary:
        flagged_pairs = {(rec["route_id"], rec["direction_id"]) for rec in records}
        summ_path = csv_path.with_name(f"{csv_path.stem}_summary.txt")
        with summ_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("route_id\tdirection_id\n")
            for rid, drn in sorted(flagged_pairs):
                fh.write(f"{rid}\t{drn}\n")
        logging.info(
            "Wrote summary → %s (%d route/direction pairs).",
            summ_path.name,
            len(flagged_pairs),
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:  # noqa: D401
    """Run the entire GTFS-to-GIS pipeline with both spacing QA checks.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if GTFS_PATH == r"Path\To\Your\GTFS_Folder" or OUTPUT_FOLDER == r"Path\To\Your\Output_Folder":
        logging.warning(
            "GTFS_PATH and/or OUTPUT_FOLDER are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2
    arcpy.env.overwriteOutput = True

    try:
        sr = _get_projected_sr(PROJECTED_WKID)
    except ValueError as err:
        logging.error("%s", err)
        return 2
    feet_factor = _feet_factor(sr)
    logging.info("Using SR: %s (1 unit ≈ %.3f ft)", sr.name, feet_factor)

    out_dir = _ensure_output_folder(OUTPUT_FOLDER)
    logging.info("STEP 0  Reading GTFS tables …")
    dfs = _read_gtfs_tables(GTFS_PATH)

    try:
        _validate_columns(dfs)
    except ValueError as err:
        logging.error("ERROR – invalid GTFS feed:\n%s", err)
        return 1
    _prepare_tables(dfs)

    logging.info("STEP 0·1  Filtering routes and trips …")
    routes_df, trips_df = _filter_routes(
        dfs["routes"],
        dfs["trips"],
        INCLUDE_ROUTE_IDS,
        FILTER_OUT_LIST,
    )
    logging.info(
        "Routes kept: %d; Trips kept: %d (after include/exclude).",
        len(routes_df),
        len(trips_df),
    )
    if trips_df.empty:
        logging.error(
            "No trips left after applying INCLUDE_ROUTE_IDS / FILTER_OUT_LIST; "
            "check that the IDs match route_id values in routes.txt."
        )
        return 2

    logging.info("STEP 1  Building stop aggregates …")
    all_stops_df, sel_stops_df = _build_stop_aggregates(dfs, trips_df, routes_df)

    logging.info("STEP 2  Building geometries for shapes and stops …")
    shape_geoms, shape_vertex_dists = _build_shape_geometries(dfs["shapes"], sr)
    stop_geoms = _build_stop_geometries(dfs["stops"], sr)
    stop_names = dict(zip(dfs["stops"]["stop_id"], dfs["stops"]["stop_name"]))
    logging.info(
        "Built %d shape polylines and %d stop points.",
        len(shape_geoms),
        len(stop_geoms),
    )

    logging.info("STEP 3  Building route polylines and stopping patterns …")
    routes = _build_routes_from_shapes(trips_df, routes_df, shape_geoms, ROUTE_UNION)
    patterns = _build_pattern_records(
        dfs["stop_times"], trips_df, routes_df, shape_geoms, shape_vertex_dists
    )

    logging.info("STEP 4  Exporting stops and routes shapefiles …")
    _export_stops_shapefile(sel_stops_df, stop_geoms, sr, out_dir)
    _export_routes_shapefile(routes, sr, out_dir)

    logging.info("STEP 5  Building stop-to-stop segment shapefile …")
    _export_segments_shapefile(
        patterns,
        stop_geoms,
        stop_names,
        sr,
        out_dir,
        max_offset_ft=SERVED_STOP_MAX_OFFSET_FT,
    )

    logging.info("STEP 6  Short-spacing QA …")
    _flag_short_spacing(
        patterns,
        stop_geoms,
        stop_names,
        sr,
        MIN_SPACING_FT,
        out_dir / SPACING_LOG_FILE,
        max_offset_ft=SERVED_STOP_MAX_OFFSET_FT,
    )

    logging.info("STEP 7  Long-spacing QA …")
    _flag_long_spacing_csv(
        patterns,
        all_stops_df,
        stop_geoms,
        stop_names,
        sr,
        LONG_SPACING_FT,
        NEAR_BUFFER_FT,
        out_dir / LONG_SPACING_CSV_FILE,
        max_offset_ft=SERVED_STOP_MAX_OFFSET_FT,
    )

    logging.info("All done! Outputs in: %s", out_dir)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        logging.exception("UNEXPECTED ERROR")
        sys.exit(1)
