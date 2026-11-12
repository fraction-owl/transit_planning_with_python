"""Calculate each GTFS route's share of length in each district, such as supervisor districts.

GeoPandas counterpart of route_length_by_district_arcpy.py: the same method,
CSV tables, and QA checks, with a GeoPackage in place of the file geodatabase.
Edit only the configuration section unless changing the analysis method.

Inputs:
    GTFS_FOLDER: a GTFS feed folder or .zip with routes.txt, trips.txt, and shapes.txt.
    DISTRICTS_PATH: district polygons (shapefile, GeoPackage, or file geodatabase
        layer) with a name field and a defined coordinate system.

Method:
    * Group by route_id, using every shape referenced by trips.txt across all
      directions and services; no service-date filter or trip weighting is applied.
    * Union each route's shapes. Coincident linework (within XY_GRID_FEET, 1 mm by
      default) counts once, so opposite directions on one street count once only if
      their shapes trace the same path; branches stay separate. This is unique mapped
      route linework, not revenue miles or average trip length.
    * Project with pyproj, using the most accurate datum transformation installed for
      each input's extent (named in the log), and measure planar miles in
      ANALYSIS_EPSG (default NAD83 / Maryland, ftUS).
    * OUTSIDE_LENGTH picks the length/percentage bases: full route, district-only
      (outside length excluded from the total; the default), or both.
    * Split linework shared by several district polygons (shared boundaries or
      overlaps) equally among them, flagged separately. A route on a boundary road
      usually lies slightly to one side, so it is assigned to that side, not split.
    * Invalid district polygons are repaired with shapely.make_valid and logged.

Outputs (in a new timestamped subfolder beneath OUTPUT_DIR):
    route_district_percentages.csv: one row per route; length totals and
        percentage columns for every district (0-100, not 0-1) on the selected bases.
    route_district_detail.csv: one row per route/district; length and percentage
        columns, and raw intersected miles (no outside row if OUTSIDE_LENGTH is "exclude").
    route_length_qa.csv: coverage, shared length and length-conservation checks, both bases.
    route_district.gpkg: unique_routes, district_segments, districts_projected.
    run_settings.json and run.log: settings, field selection, transformations.

In district_segments, alloc_mi is the additive length field: a shared piece's rows
divide it by share_n, and outside pieces (share_n=0) keep their full length. Do not sum
geom_mi across shared rows. Routes with no trips get status NO_TRIPS; a route with no
length in any district gets blank district-only percentages; missing or invalid shapes
on routes with trips raise an error. CSV values round to CSV_DECIMALS (one by default),
so displayed percentages may sum to 99.9 or 100.1; QA differences use QA_CSV_DECIMALS.

Typical usage:
    Set the paths in the CONFIGURATION section and run with Python 3.12 or newer
    and the packages in requirements.txt, from a shell or a notebook. The log
    prints each route's miles and district shares as it goes.
"""

from __future__ import annotations

import csv
import datetime as datetime_module
import json
import logging
import math
import os
import re
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from collections.abc import Collection, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from pyproj import CRS
from pyproj.transformer import AreaOfInterest, Transformer, TransformerGroup
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_FOLDER = Path(r"Path\To\Your\GTFS_Folder")  # GTFS feed folder or .zip
# A shapefile may also be given as a .zip holding it (leave DISTRICTS_LAYER None).
DISTRICTS_PATH = Path(r"Path\To\Your\Districts.shp")
# Layer name inside a GeoPackage or file geodatabase; None for a shapefile.
DISTRICTS_LAYER: str | None = None
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

# None auto-detects a district-specific field and logs its values. Generic NAME
# is not auto-selected: district layers often use it for the supervisor's name.
# If detection fails, enter the actual district field printed in the log, e.g. "DIST_NAME".
DISTRICT_NAME_FIELD: str | None = None

# Empty sets process every routes.txt record. Filters use route_short_name,
# falling back to route_id when route_short_name is blank.
ROUTES_TO_INCLUDE: set[str] = set()
ROUTES_TO_EXCLUDE: set[str] = set()

# NAD83 / Maryland (ftUS), the same default as the arcpy version.
ANALYSIS_EPSG = 2248
OUTSIDE_LABEL = "Outside districts"

# Length outside every district polygon in the percentage and detail tables:
#   "exclude": district-only shares of the length inside districts (default).
#   "include": full-route shares, with outside length as its own column and row.
#   "both": both sets of columns side by side.
# route_length_qa.csv and the GIS layers always report both.
OUTSIDE_LENGTH = "exclude"

# Decimal places for miles and percentages in the CSV tables. Calculations, the
# length check, and GIS layer attributes keep full precision.
CSV_DECIMALS = 1
# The QA shared-length and length-check difference columns report amounts near the
# 0.1-foot check thresholds, so they keep more places (0.000001 mile is about 0.06 inch).
QA_CSV_DECIMALS = 6

# Optional explicit datum transformations, given as a transformation description
# from the log; None selects the most accurate one installed for the data's extent.
GTFS_TRANSFORMATION: str | None = None
DISTRICT_TRANSFORMATION: str | None = None

# Precision grid for every overlay, in feet. Linework within about this distance
# counts as coincident. 1 mm matches ArcGIS's default XY tolerance.
XY_GRID_FEET = 0.001 / 0.3048

# A failed length check stops the run after the log names the route; run.log and
# the GIS layers written so far remain for review (unique_routes is written last).
# The tolerance is the larger of 0.1 feet and one millionth of route length.
LENGTH_CHECK_ABS_MILES = 0.1 / 5280.0
LENGTH_CHECK_REL = 1e-6
LOG_LEVEL = logging.INFO

# === END CONFIG ===

LOGGER = logging.getLogger("route_length_by_district_gpd")
METERS_PER_MILE = 1609.344
QA_DETAIL_FIELDS = {"shared_boundary_or_overlap_miles", "length_check_delta_miles"}
RouteInfo = dict[str, str]
Piece = tuple[tuple[str, ...], BaseGeometry]


@contextmanager
def open_gtfs_file(
    gtfs_path: str | os.PathLike[str], file_name: str
) -> Generator[IO[bytes], None, None]:
    """Open one GTFS text file from a feed folder or a ``.zip`` of the feed.

    The counterpart of :func:`load_gtfs_data` for scripts that read a file with
    their own options (``pandas.read_csv`` with ``usecols`` or
    ``keep_default_na=False``, or the :mod:`csv` module), so they accept a
    zipped feed without extracting it. Zip members are found the way
    :func:`load_gtfs_data` finds them: at the archive root or inside a single
    wrapper folder.

    Args:
        gtfs_path: Folder containing the GTFS feed, or a ``.zip`` archive of it.
        file_name: GTFS file to open, e.g. ``"stops.txt"``.

    Yields:
        A binary file handle. ``pandas.read_csv`` reads it directly; wrap it in
        :class:`io.TextIOWrapper` for the :mod:`csv` module.

    Raises:
        FileNotFoundError: *gtfs_path* does not exist, or the feed has no
            *file_name*.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, or *file_name* is found in more than one place in the zip.
    """
    path = os.fspath(gtfs_path)
    if os.path.isdir(path):
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        with open(file_path, "rb") as handle:
            yield handle
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    if not (os.path.isfile(path) and path.lower().endswith(".zip")):
        raise ValueError(f"'{path}' is neither a directory nor a .zip file.")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc
    with archive:
        matches = [name for name in archive.namelist() if os.path.basename(name) == file_name]
        if not matches:
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous GTFS file in '{path}' (found in multiple locations): {file_name}"
            )
        with archive.open(matches[0]) as handle:
            yield handle


def read_gtfs(filename: str, required: set[str]) -> pd.DataFrame:
    """Read a GTFS table as text, without coercing identifiers or literal 'NA' values."""
    try:
        with warnings.catch_warnings(), open_gtfs_file(GTFS_FOLDER, filename) as handle:
            # pandas only warns when the first data row has extra fields; treat it as an error.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            table = pd.read_csv(
                handle, dtype=str, keep_default_na=False, index_col=False, encoding="utf-8-sig"
            )
    except (pd.errors.ParserError, pd.errors.ParserWarning, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read {filename}: {exc}") from exc
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{filename} is missing fields: {sorted(missing)}")
    for column in table.columns:
        table[column] = table[column].str.strip()
    return table


def route_label(route: RouteInfo) -> str:
    """Return the label used by route filters and sorting."""
    return route["route_short_name"] or route["route_id"]


def natural_key(value: str) -> list[tuple[int, Any]]:
    """Sort route numbers numerically while also supporting alphanumeric names."""
    return [
        (0, int(token)) if token.isdigit() else (1, token.casefold())
        for token in re.split(r"(\d+)", value)
    ]


def load_routes_and_trips() -> tuple[dict[str, RouteInfo], dict[str, set[str]], Counter[str]]:
    """Load selected routes and unique shape references, rejecting incomplete data."""
    LOGGER.info("Reading routes.txt and trips.txt from %s", GTFS_FOLDER)
    routes_table = read_gtfs("routes.txt", {"route_id"})
    for column in ("route_short_name", "route_long_name"):
        if column not in routes_table.columns:
            routes_table[column] = ""
    bad_ids = routes_table["route_id"].eq("") | routes_table["route_id"].duplicated()
    if bad_ids.any():
        route_id = routes_table.loc[bad_ids, "route_id"].iloc[0]
        raise ValueError(f"Blank or duplicate route_id in routes.txt: {route_id!r}")
    all_routes: dict[str, RouteInfo] = {
        str(route_id): {
            "route_id": str(route_id),
            "route_short_name": str(short_name),
            "route_long_name": str(long_name),
        }
        for route_id, short_name, long_name in zip(
            routes_table["route_id"],
            routes_table["route_short_name"],
            routes_table["route_long_name"],
            strict=True,
        )
    }
    available = {route_label(route) for route in all_routes.values()}
    unknown = (ROUTES_TO_INCLUDE | ROUTES_TO_EXCLUDE) - available
    if unknown:
        raise ValueError(f"Route filters contain unknown names: {sorted(unknown)}")
    routes = {
        key: route
        for key, route in all_routes.items()
        if (not ROUTES_TO_INCLUDE or route_label(route) in ROUTES_TO_INCLUDE)
        and route_label(route) not in ROUTES_TO_EXCLUDE
    }
    if not routes:
        raise ValueError("No routes selected.")

    trips = read_gtfs("trips.txt", {"route_id", "trip_id", "shape_id"})
    unknown_route = ~trips["route_id"].isin(set(all_routes))
    if unknown_route.any():
        trip = trips[unknown_route].iloc[0]
        raise ValueError(f"Trip {trip['trip_id']!r} references unknown route {trip['route_id']!r}.")
    trips = trips[trips["route_id"].isin(set(routes))]
    bad_trips = trips["trip_id"].eq("") | trips["trip_id"].duplicated()
    if bad_trips.any():
        trip_id = trips.loc[bad_trips, "trip_id"].iloc[0]
        raise ValueError(f"Blank or duplicate selected trip_id: {trip_id!r}")
    blank_shapes = trips["shape_id"].eq("")
    if blank_shapes.any():
        missing_shapes = trips.loc[blank_shapes, "route_id"].value_counts().to_dict()
        raise ValueError(f"Trips with blank shape_id, by route_id: {missing_shapes}")
    trip_counts: Counter[str] = Counter(trips["route_id"].tolist())
    route_shapes = {
        str(route_id): set(shape_ids)
        for route_id, shape_ids in trips.groupby("route_id")["shape_id"]
    }
    LOGGER.info("Selected %d routes and %d trips.", len(routes), len(trips))
    return routes, route_shapes, trip_counts


def choose_transformer(
    source: CRS,
    target: CRS,
    bounds: tuple[float, float, float, float],
    override: str | None,
) -> Transformer | None:
    """Choose a suitable datum transformation, never silently a ballpark one.

    Args:
        source: Coordinate system of the input data.
        target: Analysis coordinate system.
        bounds: Input extent as (west, south, east, north) in degrees.
        override: Description of a specific transformation to use, or None.

    Returns:
        The transformer, or None when source and target are the same system.
    """
    if source == target:
        return None
    with warnings.catch_warnings():
        # pyproj warns when the best transformation needs missing grids; logged below instead.
        warnings.simplefilter("ignore", UserWarning)
        group = TransformerGroup(
            source,
            target,
            always_xy=True,
            area_of_interest=AreaOfInterest(*bounds),
            allow_ballpark=False,
        )
    candidates = list(group.transformers)
    descriptions = [candidate.description for candidate in candidates]
    if override:
        for candidate in candidates:
            if candidate.description.casefold() == override.strip().casefold():
                LOGGER.info("Using configured transformation: %s", candidate.description)
                return candidate
        raise ValueError(
            f"Transformation {override!r} is not available for {source.name} -> "
            f"{target.name}. Available: {descriptions}"
        )
    if not candidates:
        needing_grids = [operation.name for operation in group.unavailable_operations]
        raise ValueError(
            f"No datum transformation found: {source.name} -> {target.name}. "
            f"Candidates needing PROJ grid files: {needing_grids}. Install the grids "
            "(pyproj's projsync) or enable PROJ network access, then rerun."
        )
    if not group.best_available:
        LOGGER.warning(
            "More accurate transformations need PROJ grid files that are not installed: %s",
            "; ".join(operation.name for operation in group.unavailable_operations),
        )
    LOGGER.info("Transformation %s -> %s: %s", source.name, target.name, descriptions[0])
    if len(descriptions) > 1:
        LOGGER.info("Other available transformations: %s", "; ".join(descriptions[1:]))
    return candidates[0]


def project_geometry(geometry: BaseGeometry, transformer: Transformer | None) -> BaseGeometry:
    """Project a geometry using the selected transformation when necessary."""
    if transformer is None:
        return geometry

    def transform_xy(coords: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    projected = shapely.transform(geometry, transform_xy)
    if not np.isfinite(shapely.get_coordinates(projected)).all():
        raise ValueError("Projection produced non-finite coordinates; check the input CRS.")
    return projected


def lines_only(geometry: BaseGeometry) -> MultiLineString:
    """Keep the linear parts of an overlay result, dropping touch points and empties."""
    parts = shapely.get_parts(shapely.get_parts(geometry))
    return MultiLineString(
        [part for part in parts if isinstance(part, LineString) and not part.is_empty]
    )


def load_shape_geometries(
    needed: set[str], target: CRS
) -> tuple[dict[str, BaseGeometry], str | None]:
    """Validate and project the shapes referenced by selected routes."""
    if not needed:
        return {}, None
    LOGGER.info("Reading shapes.txt for %d shapes.", len(needed))
    required = {"shape_id", "shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"}
    table = read_gtfs("shapes.txt", required)
    table = table[table["shape_id"].isin(needed)]
    lon = pd.to_numeric(table["shape_pt_lon"], errors="coerce")
    lat = pd.to_numeric(table["shape_pt_lat"], errors="coerce")
    valid = (
        table["shape_pt_sequence"].str.fullmatch(r"\d+")
        & lon.between(-180, 180)
        & lat.between(-90, 90)
    )
    if not valid.all():
        row = table[~valid].iloc[0].to_dict()
        raise ValueError(f"Invalid sequence/coordinates for shape {row['shape_id']!r}: {row}")
    missing = needed - set(table["shape_id"])
    if missing:
        raise ValueError(f"Referenced shape_ids absent from shapes.txt: {sorted(missing)}")

    points = pd.DataFrame(
        {
            "shape_id": table["shape_id"],
            "sequence": table["shape_pt_sequence"].astype("int64"),
            "lon": lon,
            "lat": lat,
        }
    )
    duplicated = points.duplicated(["shape_id", "sequence"])
    if duplicated.any():
        shape_id = points.loc[duplicated, "shape_id"].iloc[0]
        raise ValueError(f"Duplicate shape_pt_sequence for shape {shape_id!r}.")
    LOGGER.info("Read %d shape points; building and projecting shapes.", len(points))
    bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    transformer = choose_transformer(CRS.from_epsg(4326), target, bounds, GTFS_TRANSFORMATION)
    geometries: dict[str, BaseGeometry] = {}
    points = points.sort_values(["shape_id", "sequence"], kind="stable")
    for shape_id, group in points.groupby("shape_id", sort=True):
        coords = group[["lon", "lat"]].to_numpy()
        changed = np.ones(len(coords), dtype=bool)
        changed[1:] = (coords[1:] != coords[:-1]).any(axis=1)
        coords = coords[changed]
        if len(np.unique(coords, axis=0)) < 2:
            raise ValueError(f"Shape {shape_id!r} has fewer than two distinct coordinates.")
        geometry = project_geometry(LineString(coords), transformer)
        if geometry.is_empty or geometry.length <= 0:
            raise ValueError(f"Shape {shape_id!r} has no measurable projected length.")
        geometries[str(shape_id)] = geometry
    LOGGER.info("Built %d projected GTFS shapes.", len(geometries))
    return geometries, transformer.description if transformer else None


def find_district_field(districts: gpd.GeoDataFrame) -> str:
    """Resolve a district field without mistaking generic NAME for district names."""
    fields = [str(column) for column in districts.columns if column != districts.geometry.name]
    available = ", ".join(f"{field} ({districts[field].dtype})" for field in fields)
    LOGGER.info("Available district fields: %s", available)
    if DISTRICT_NAME_FIELD:
        matches = [field for field in fields if field.casefold() == DISTRICT_NAME_FIELD.casefold()]
        if matches:
            return matches[0]
        raise ValueError(
            f"District field {DISTRICT_NAME_FIELD!r} not found. Available fields: {available}"
        )
    candidates = [
        "DISTRICT_NAME",
        "DISTRICT_N",
        "DIST_NAME",
        "DISTNAME",
        "SUP_DIST_NAME",
        "SUP_DIST_N",
        "SUPERVISOR_DISTRICT_NAME",
        "SUPDISTNAME",
        "SUPERVISOR_DISTRICT",
        "DISTRICT",
        "SUP_DIST",
        "SUPDIST",
        "SUP_DISTRI",
        "DIST",
    ]

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    for candidate in candidates:
        matches = [field for field in fields if normalize(candidate) == normalize(field)]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(
        "Set DISTRICT_NAME_FIELD to the actual district-name field. Generic NAME is "
        "not auto-selected because it may contain supervisor names. "
        f"Available fields: {available}"
    )


def load_shapefile(path: str, member: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load a shapefile given a direct ``.shp`` path or a ``.zip`` archive.

    Args:
        path: Path to a ``.shp`` file, or to a ``.zip`` archive containing
            one. A zipped shapefile is opened directly via GDAL's ``zip://``
            virtual filesystem, so the archive is never extracted to disk.
        member: For a ``.zip`` containing more than one ``.shp``, the name
            of the one to load (matched case-insensitively against its
            base filename, e.g. ``"Hospitals.shp"``). Ignored when *path*
            is a direct ``.shp`` file. Required when the archive contains
            more than one shapefile. macOS metadata entries (anything under
            ``__MACOSX/`` and ``._*`` files) are not counted as shapefiles.

    Returns:
        The loaded :class:`geopandas.GeoDataFrame`.

    Raises:
        OSError: *path* does not exist.
        ValueError: *path* is neither a ``.shp`` nor a ``.zip`` file; the
            zip is unreadable/corrupt; the zip contains no shapefile; the
            zip contains more than one shapefile and *member* was not
            supplied; or *member* does not match any shapefile in the zip.
    """
    if not os.path.exists(path):
        raise OSError(f"The path '{path}' does not exist.")

    lower_path = path.lower()
    if lower_path.endswith(".shp"):
        logging.info("Reading %s", path)
        return gpd.read_file(path)

    if not lower_path.endswith(".zip"):
        raise ValueError(f"'{path}' is neither a .shp nor a .zip file.")

    try:
        with zipfile.ZipFile(path) as archive:
            # Zips made with macOS Finder add __MACOSX/._<name>.shp metadata
            # files next to the real ones; they are not shapefiles.
            shp_members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".shp")
                and not name.startswith("__MACOSX/")
                and not os.path.basename(name).startswith("._")
            ]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc

    if not shp_members:
        raise ValueError(f"No .shp file found inside '{path}'.")

    if member is not None:
        matches = [name for name in shp_members if os.path.basename(name).lower() == member.lower()]
        if not matches:
            raise ValueError(f"'{member}' not found inside '{path}'. Available: {shp_members}")
        chosen = matches[0]
    elif len(shp_members) == 1:
        chosen = shp_members[0]
    else:
        raise ValueError(
            f"'{path}' contains {len(shp_members)} shapefiles; pass `member=` to "
            f"choose one of: {shp_members}"
        )

    source = f"zip://{path}!{chosen}"
    logging.info("Reading %s", source)
    return gpd.read_file(source)


def load_districts(
    target: CRS, grid: float
) -> tuple[dict[str, BaseGeometry], str, str | None, list[str]]:
    """Project district polygons and union multiple records with the same label."""
    LOGGER.info("Reading districts from %s", DISTRICTS_PATH)
    if DISTRICTS_LAYER is None and str(DISTRICTS_PATH).lower().endswith(".zip"):
        districts = load_shapefile(str(DISTRICTS_PATH))
    else:
        districts = gpd.read_file(DISTRICTS_PATH, layer=DISTRICTS_LAYER)
    if districts.empty:
        raise ValueError("No district polygons found.")
    if districts.crs is None:
        raise ValueError("DISTRICTS_PATH has no coordinate system; define it before running.")
    if not set(districts.geom_type.dropna()) <= {"Polygon", "MultiPolygon"}:
        raise ValueError("DISTRICTS_PATH must contain polygons.")
    field = find_district_field(districts)
    lonlat = Transformer.from_crs(districts.crs, 4326, always_xy=True)
    bounds = lonlat.transform_bounds(*districts.total_bounds)
    transformer = choose_transformer(districts.crs, target, bounds, DISTRICT_TRANSFORMATION)
    grouped: dict[str, list[BaseGeometry]] = defaultdict(list)
    repaired: list[str] = []
    for value, geometry in zip(districts[field], districts.geometry, strict=True):
        label = "" if pd.isna(value) else str(value).strip()
        if not label or label.casefold() == OUTSIDE_LABEL.casefold():
            raise ValueError(f"Blank or reserved district label: {label!r}")
        if geometry is None or geometry.is_empty:
            raise ValueError(f"Null or empty district polygon: {label}")
        projected = project_geometry(geometry, transformer)
        # Repair before measuring: a self-intersecting ring can report zero area.
        if not projected.is_valid:
            projected = shapely.make_valid(projected, method="structure", keep_collapsed=False)
            repaired.append(label)
        if projected.is_empty or projected.area <= 0:
            raise ValueError(f"Zero-area district polygon after projection and repair: {label}")
        grouped[label].append(projected)
    if repaired:
        LOGGER.warning("Repaired invalid district geometry: %s", ", ".join(repaired))
    labels = sorted(grouped)
    # union_all skips the grid for a single polygon, so snap every district explicitly.
    unions = [
        shapely.set_precision(shapely.union_all(grouped[label], grid_size=grid), grid)
        for label in labels
    ]
    shapely.prepare(unions)
    LOGGER.info("District field: %s. Values: %s", field, ", ".join(labels))
    description = transformer.description if transformer else None
    return dict(zip(labels, unions, strict=True)), field, description, repaired


def length_miles(geometry: BaseGeometry, miles_per_unit: float) -> float:
    """Measure length in miles with the same planar method throughout."""
    return float(geometry.length) * miles_per_unit


def split_by_district(
    route_geometry: BaseGeometry, districts: dict[str, BaseGeometry], grid: float
) -> list[Piece]:
    """Partition route linework by its complete set of district memberships.

    Each piece appears exactly once in this partition, including shared boundaries
    and outside pieces. Intersection with a polygon keeps linework on its boundary
    and difference drops it, so the two always partition the line between them.
    """
    pieces: list[Piece] = [((), route_geometry)]
    for label, polygon in districts.items():
        updated: list[Piece] = []
        for members, line in pieces:
            # Skip only beyond the grid distance; overlays snap anything closer.
            if not shapely.dwithin(polygon, line, grid):
                updated.append((members, line))
                continue
            inside = lines_only(shapely.intersection(line, polygon, grid_size=grid))
            if inside.is_empty or inside.length <= 0:
                updated.append((members, line))
                continue
            updated.append((members + (label,), inside))
            outside = lines_only(shapely.difference(line, polygon, grid_size=grid))
            if not outside.is_empty and outside.length > 0:
                updated.append((members, outside))
        pieces = updated
    return pieces


def summarize_pieces(
    pieces: Sequence[Piece], labels: Sequence[str], total_miles: float, miles_per_unit: float
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Allocate shared pieces equally and verify conservation of route length."""
    allocated = dict.fromkeys([*labels, OUTSIDE_LABEL], 0.0)
    raw = dict.fromkeys([*labels, OUTSIDE_LABEL], 0.0)
    shared_miles = 0.0
    for members, geometry in pieces:
        miles = length_miles(geometry, miles_per_unit)
        recipients = members or (OUTSIDE_LABEL,)
        for label in recipients:
            allocated[label] += miles / len(recipients)
            raw[label] += miles
        if len(members) > 1:
            shared_miles += miles
    delta = math.fsum(allocated.values()) - total_miles
    tolerance = max(LENGTH_CHECK_ABS_MILES, total_miles * LENGTH_CHECK_REL)
    if not math.isfinite(total_miles) or total_miles <= 0:
        raise ValueError("Route has no positive finite length.")
    if not math.isfinite(delta) or abs(delta) > tolerance:
        raise ValueError(
            f"District allocation failed length check: difference {delta:.9f} miles; "
            f"allowed {tolerance:.9f}. Inspect the source geometries."
        )
    return allocated, raw, shared_miles, delta


def inside_percentage(label: str, allocated_miles: float, inside_miles: float) -> float | None:
    """Calculate a district-only share; outside length contributes zero."""
    if inside_miles <= 0:
        return None
    return 0.0 if label == OUTSIDE_LABEL else allocated_miles / inside_miles * 100


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[dict[str, Any]],
    omit: Collection[str] = (),
) -> None:
    """Write an Excel-readable UTF-8 CSV of fields not in omit, with numbers rounded."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[field for field in fields if field not in omit],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            formatted: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, float):
                    places = QA_CSV_DECIMALS if key in QA_DETAIL_FIELDS else CSV_DECIMALS
                    # Fixed notation avoids 1.9e-05; adding 0.0 writes -0.0 as 0.0.
                    value = f"{round(value, places) + 0.0:.{places}f}"
                formatted[key] = value
            writer.writerow(formatted)
    LOGGER.info("Wrote %s (%d rows).", path.name, len(rows))


def write_layer(
    gpkg: Path,
    layer: str,
    fields: Sequence[tuple[str, str]],
    rows: Sequence[Sequence[Any]],
    crs: CRS,
) -> None:
    """Write one GeoPackage layer; the last value in each row is its geometry."""
    dtypes = {"TEXT": "object", "DOUBLE": "float64", "LONG": "int64"}
    frame = pd.DataFrame(
        {name: [row[index] for row in rows] for index, (name, _) in enumerate(fields)}
    ).astype({name: dtypes[kind] for name, kind in fields})
    layer_gdf = gpd.GeoDataFrame(frame, geometry=[row[-1] for row in rows], crs=crs)
    layer_gdf.to_file(gpkg, layer=layer, driver="GPKG")
    LOGGER.info("Wrote %d features to %s.", len(rows), layer)


def run_analysis(run_dir: Path) -> None:
    """Build route linework, allocate district lengths, and write tables and GIS data."""
    target = CRS.from_epsg(ANALYSIS_EPSG)
    if not target.is_projected:
        raise ValueError("ANALYSIS_EPSG must be a suitable local projected coordinate system.")
    if XY_GRID_FEET <= 0:
        raise ValueError("XY_GRID_FEET must be greater than zero.")
    if OUTSIDE_LENGTH not in {"exclude", "include", "both"}:
        raise ValueError('OUTSIDE_LENGTH must be "exclude", "include", or "both".')
    meters_per_unit = target.axis_info[0].unit_conversion_factor
    miles_per_unit = meters_per_unit / METERS_PER_MILE
    grid = XY_GRID_FEET * 0.3048 / meters_per_unit
    LOGGER.info("Analysis coordinate system: %s (%s)", target.name, target.axis_info[0].unit_name)
    routes, route_shapes, trip_counts = load_routes_and_trips()
    districts, district_field, district_transform, repaired = load_districts(target, grid)
    labels = [*districts, OUTSIDE_LABEL]
    needed = set().union(*route_shapes.values()) if route_shapes else set()
    shapes, gtfs_transform = load_shape_geometries(needed, target)

    gpkg = run_dir / "route_district.gpkg"
    LOGGER.info("Creating %s", gpkg)
    route_fields = [("route_id", "TEXT"), ("route_name", "TEXT")]
    write_layer(
        gpkg, "districts_projected", [("district", "TEXT")], list(districts.items()), target
    )

    wide_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    ordered = sorted(routes, key=lambda key: (natural_key(route_label(routes[key])), key))
    segment_fields = [
        *route_fields,
        ("district", "TEXT"),
        ("share_n", "LONG"),
        ("geom_mi", "DOUBLE"),
        ("alloc_mi", "DOUBLE"),
        ("in_dist_mi", "DOUBLE"),
        ("pct_route", "DOUBLE"),
        ("pct_inside", "DOUBLE"),
    ]
    segment_rows: list[list[Any]] = []
    route_feature_rows: list[list[Any]] = []
    LOGGER.info("Allocating %d routes among %d districts.", len(ordered), len(districts))
    # Segments are written even if a route fails, so the pieces so far can be reviewed.
    try:
        for index, route_id in enumerate(ordered, start=1):
            info = routes[route_id]
            name = route_label(info)
            LOGGER.info("Route %s (%d/%d)", name, index, len(ordered))
            if not trip_counts[route_id]:
                LOGGER.warning("Route %s has no trips; percentages will be blank.", name)
                wide_rows.append({**info, "status": "NO_TRIPS"})
                qa_rows.append(
                    {
                        **info,
                        "status": "NO_TRIPS",
                        "trip_count": 0,
                        "shape_count": 0,
                    }
                )
                for label in labels:
                    detail_rows.append({**info, "status": "NO_TRIPS", "district": label})
                continue

            route_parts = [shapes[sid] for sid in sorted(route_shapes[route_id])]
            geometry = lines_only(shapely.union_all(route_parts, grid_size=grid))
            total = length_miles(geometry, miles_per_unit)
            pieces = split_by_district(geometry, districts, grid)
            allocated, raw, shared, delta = summarize_pieces(
                pieces, list(districts), total, miles_per_unit
            )
            inside = math.fsum(allocated[label] for label in districts)
            status = "REVIEW_SHARED_LENGTH" if shared > LENGTH_CHECK_ABS_MILES else "OK"
            if shared > LENGTH_CHECK_ABS_MILES:
                LOGGER.warning("Route %s: %.6f miles shared by multiple districts.", name, shared)
            shares = [f"{lb} {allocated[lb] / total:.1%}" for lb in labels if allocated[lb] > 0]
            LOGGER.info("Route %s: %.2f mi; %s", name, total, ", ".join(shares))
            wide = {
                **info,
                "status": status,
                "total_route_miles": total,
                "inside_districts_miles": inside,
                "outside_districts_miles": allocated[OUTSIDE_LABEL],
            }
            district_only = {
                label: inside_percentage(label, allocated[label], inside) for label in labels
            }
            for label in labels:
                wide[f"{label} (full route %)"] = allocated[label] / total * 100
                wide[f"{label} (district-only %)"] = district_only[label]
            wide["full_route_percent_sum"] = math.fsum(allocated.values()) / total * 100
            wide["district_only_percent_sum"] = (
                math.fsum(value for value in district_only.values() if value is not None)
                if inside > 0
                else None
            )
            wide_rows.append(wide)
            qa_rows.append(
                {
                    **info,
                    "status": status,
                    "trip_count": trip_counts[route_id],
                    "shape_count": len(route_shapes[route_id]),
                    "total_route_miles": total,
                    "inside_districts_miles": inside,
                    "outside_districts_miles": allocated[OUTSIDE_LABEL],
                    "shared_boundary_or_overlap_miles": shared,
                    "length_check_delta_miles": delta,
                    "full_route_percent_sum": wide["full_route_percent_sum"],
                    "district_only_percent_sum": wide["district_only_percent_sum"],
                }
            )
            for label in labels:
                detail_rows.append(
                    {
                        **info,
                        "status": status,
                        "district": label,
                        "total_route_miles": total,
                        "inside_districts_miles": inside,
                        "miles_including_outside": allocated[label],
                        "miles_excluding_outside": (
                            0.0 if label == OUTSIDE_LABEL else allocated[label]
                        ),
                        "raw_intersected_miles": raw[label],
                        "percent_of_full_route": allocated[label] / total * 100,
                        "percent_of_inside_length": inside_percentage(
                            label, allocated[label], inside
                        ),
                    }
                )
            route_feature_rows.append(
                [
                    route_id,
                    name,
                    total,
                    inside,
                    allocated[OUTSIDE_LABEL],
                    geometry,
                ]
            )
            for members, line in pieces:
                miles = length_miles(line, miles_per_unit)
                recipients = members or (OUTSIDE_LABEL,)
                for label in recipients:
                    allocated_miles = miles / len(recipients)
                    segment_rows.append(
                        [
                            route_id,
                            name,
                            label,
                            len(members),
                            miles,
                            allocated_miles,
                            0.0 if label == OUTSIDE_LABEL else allocated_miles,
                            allocated_miles / total * 100,
                            inside_percentage(label, allocated_miles, inside),
                            line,
                        ]
                    )
    finally:
        write_layer(gpkg, "district_segments", segment_fields, segment_rows, target)

    statuses = Counter(row["status"] for row in qa_rows)
    LOGGER.info("Processed %d routes. Status counts: %s", len(qa_rows), dict(statuses))
    write_layer(
        gpkg,
        "unique_routes",
        [
            *route_fields,
            ("total_mi", "DOUBLE"),
            ("inside_mi", "DOUBLE"),
            ("outside_mi", "DOUBLE"),
        ],
        route_feature_rows,
        target,
    )

    identity = ["route_id", "route_short_name", "route_long_name", "status"]
    # OUTSIDE_LENGTH leaves one basis out of the next two tables; the QA table keeps both.
    full_route_fields = {
        "total_route_miles",
        "outside_districts_miles",
        "full_route_percent_sum",
        "miles_including_outside",
        "percent_of_full_route",
        *(f"{label} (full route %)" for label in labels),
        f"{OUTSIDE_LABEL} (district-only %)",  # Always 0; shown only beside full-route shares.
    }
    district_only_fields = {
        "district_only_percent_sum",
        "miles_excluding_outside",
        "percent_of_inside_length",
        *(f"{label} (district-only %)" for label in labels),
    }
    omit: set[str] = set()
    if OUTSIDE_LENGTH == "exclude":
        omit = full_route_fields
        detail_rows = [row for row in detail_rows if row["district"] != OUTSIDE_LABEL]
    elif OUTSIDE_LENGTH == "include":
        omit = district_only_fields
    write_csv(
        run_dir / "route_district_percentages.csv",
        [
            *identity,
            "total_route_miles",
            "inside_districts_miles",
            "outside_districts_miles",
            *[
                f"{label} ({basis} %)"
                for label in labels
                for basis in ("full route", "district-only")
            ],
            "full_route_percent_sum",
            "district_only_percent_sum",
        ],
        wide_rows,
        omit,
    )
    write_csv(
        run_dir / "route_district_detail.csv",
        [
            *identity,
            "district",
            "total_route_miles",
            "inside_districts_miles",
            "miles_including_outside",
            "miles_excluding_outside",
            "raw_intersected_miles",
            "percent_of_full_route",
            "percent_of_inside_length",
        ],
        detail_rows,
        omit,
    )
    write_csv(
        run_dir / "route_length_qa.csv",
        [
            *identity,
            "trip_count",
            "shape_count",
            "total_route_miles",
            "inside_districts_miles",
            "outside_districts_miles",
            "shared_boundary_or_overlap_miles",
            "length_check_delta_miles",
            "full_route_percent_sum",
            "district_only_percent_sum",
        ],
        qa_rows,
    )
    settings = {
        "gtfs_folder": str(GTFS_FOLDER),
        "districts_path": str(DISTRICTS_PATH),
        "districts_layer": DISTRICTS_LAYER,
        "district_name_field": district_field,
        "district_labels": list(districts),
        "repaired_district_geometries": repaired,
        "analysis_epsg": ANALYSIS_EPSG,
        "analysis_crs": target.name,
        "gtfs_transformation": gtfs_transform,
        "district_transformation": district_transform,
        "routes_to_include": sorted(ROUTES_TO_INCLUDE),
        "routes_to_exclude": sorted(ROUTES_TO_EXCLUDE),
        "route_grouping": "route_id",
        "length_method": "PLANAR",
        "length_units": "MILES",
        "xy_grid_feet": XY_GRID_FEET,
        "route_length_basis": "Union of all shapes referenced by trips; coincident linework once.",
        "service_filter": "None: all trips, directions, and services in the feed.",
        "trip_frequency_weighted": False,
        "snap_nearby_shapes": False,
        "percentage_denominators": {
            "full_route": "Full route length, including outside all district polygons.",
            "district_only": "Length inside the union of district polygons; outside length zeroed.",
        },
        "shared_length_policy": "Equal split among all intersecting district labels.",
        "outside_label": OUTSIDE_LABEL,
        "outside_length": OUTSIDE_LENGTH,
        "csv_decimals": CSV_DECIMALS,
        "qa_csv_decimals": QA_CSV_DECIMALS,
        "output_directory": str(run_dir),
        "versions": {
            "geopandas": gpd.__version__,
            "shapely": shapely.__version__,
            "geos": shapely.geos_version_string,
            "pyproj": pyproj.__version__,
            "proj": pyproj.proj_version_str,
            "pandas": pd.__version__,
        },
        "length_check_abs_miles": LENGTH_CHECK_ABS_MILES,
        "length_check_relative": LENGTH_CHECK_REL,
    }
    (run_dir / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    LOGGER.info("Main table: %s", run_dir / "route_district_percentages.csv")


def main() -> int:
    """Run the configured analysis; return 0 on success, 1 on failure, 2 for missing inputs."""
    started = time.perf_counter()
    LOGGER.setLevel(LOG_LEVEL)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)
    run_dir: Path | None = None
    try:
        # Check inputs before creating an output folder, so placeholder paths do nothing.
        missing = []
        for name in ("routes.txt", "trips.txt", "shapes.txt"):
            try:
                with open_gtfs_file(GTFS_FOLDER, name):
                    pass
            except (FileNotFoundError, ValueError):
                missing.append(str(GTFS_FOLDER / name))
        if not DISTRICTS_PATH.exists():
            missing.append(str(DISTRICTS_PATH))
        if missing:
            LOGGER.error("Input not found; update the CONFIGURATION paths: %s", "; ".join(missing))
            return 2
        timestamp = datetime_module.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = OUTPUT_DIR / f"route_district_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        log_file = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        log_file.setFormatter(formatter)
        LOGGER.addHandler(log_file)
        LOGGER.info("Starting route length by district. Outputs: %s", run_dir)
        run_analysis(run_dir)
        LOGGER.info("Completed successfully in %.1f seconds.", time.perf_counter() - started)
        return 0
    except Exception:
        LOGGER.exception("Analysis failed. Completed outputs remain in %s", run_dir or OUTPUT_DIR)
        return 1
    finally:
        for handler in list(LOGGER.handlers):
            handler.flush()
            handler.close()
            LOGGER.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
