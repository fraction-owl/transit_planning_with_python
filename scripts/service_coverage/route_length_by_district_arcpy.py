"""Calculate each GTFS route's share of length in each district, such as supervisor districts.

Edit only the configuration section unless changing the analysis method. Uses
ArcPy and the Python standard library; pandas and additional GIS extensions are
not required.

Inputs:
    GTFS_FOLDER: an unzipped GTFS feed with routes.txt, trips.txt, and shapes.txt.
    DISTRICTS_FC: district polygons (shapefile or feature class) with a name field.

Method:
    * Group by route_id, retaining route_short_name and route_long_name.
    * Use every shape referenced by trips.txt, across all directions and services.
      No service-date filter or trip-frequency weighting is applied.
    * Union each route's shapes. Coincident linework counts once; branches and
      spatially separate inbound/outbound alignments remain separate. This is
      unique mapped route linework, not revenue miles or average trip length.
      Nearby but noncoincident shapes are not snapped together. Coincident means
      within the XY tolerance (1 mm by default), so opposite directions on one
      street count once only if their shapes trace the same path.
    * Measure planar miles in ANALYSIS_EPSG (default NAD83 / Virginia North, ftUS).
    * Report either or both length/percentage bases, set by OUTSIDE_LENGTH: full
      route (including outside areas), and district-only (outside length set to
      zero and excluded from the total; the default).
    * Split linework shared by multiple district polygons equally among them.
      This includes exact shared boundaries and polygon overlaps. Shared length
      is separately flagged, and the GIS output retains every membership. A route
      on a road that forms a district boundary usually lies slightly to one side
      of the boundary line, so it is assigned to that side, not split or flagged.

Outputs (in a new timestamped subfolder beneath OUTPUT_DIR):
    route_district_percentages.csv: one row per route; length totals and
        percentage columns for every district (0-100, not 0-1) on the selected bases.
    route_district_detail.csv: one row per route/district; length and percentage
        columns on the selected bases, and raw intersected miles. The outside row
        is left out when OUTSIDE_LENGTH is "exclude".
    route_length_qa.csv: coverage, shared length, and length-conservation checks,
        always on both bases.
    route_district.gdb: unique_routes, district_segments, districts_projected.
    run_settings.json and run.log: settings, field selection, transformations.

For district_segments, alloc_mi is the additive length field. A shared piece has
one row per district; its physical geometry is repeated, but alloc_mi is divided
by share_n. Outside pieces have share_n=0 and retain their full length. Do not
sum Shape_Length or geom_mi across shared rows.
in_dist_mi is alloc_mi with outside pieces set to zero. If a route has no length
inside any district, district-only percentages are undefined and left blank.
Routes with no trips have blank percentages and status NO_TRIPS. Missing or
invalid shapes for routes with trips cause an error rather than partial results.
total_route_miles is the unique linework length described above.
CSV miles and percentages are rounded to CSV_DECIMALS places (one by default),
so a row's displayed percentages may sum to 99.9 or 100.1; the percent-sum
columns are summed before rounding. The QA shared-length and length-check
difference columns use QA_CSV_DECIMALS (six by default).

Typical usage:
    Set the paths in the CONFIGURATION section and run with ArcGIS Pro's Python
    (3.9 or newer) from a shell, the Pro Python window, or a notebook. The log
    prints each route's miles and district shares as it goes.

ArcPy references:
    https://pro.arcgis.com/en/pro-app/3.3/arcpy/classes/polyline.htm
    https://pro.arcgis.com/en/pro-app/3.3/arcpy/functions/listtransformations.htm
"""

from __future__ import annotations

import csv
import datetime as datetime_module
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import arcpy

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_FOLDER = Path(r"Path\To\Your\GTFS_Folder")
DISTRICTS_FC = Path(r"Path\To\Your\Districts.shp")
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

# None auto-detects a district-specific field and logs its values. Generic NAME
# is not auto-selected: district layers often use it for the supervisor's name.
# If detection fails, enter the actual district field printed in the log, e.g. "DIST_NAME".
DISTRICT_NAME_FIELD: Optional[str] = None

# Empty sets process every routes.txt record. Filters use route_short_name,
# falling back to route_id when route_short_name is blank.
ROUTES_TO_INCLUDE: Set[str] = set()
ROUTES_TO_EXCLUDE: Set[str] = set()

# EPSG code of a local projected CRS (ArcGIS lists it as the WKID).
ANALYSIS_EPSG = 2283
OUTSIDE_LABEL = "Outside districts"

# Length outside every district polygon in the percentage and detail tables:
#   "exclude": district-only shares of the length inside districts (default).
#   "include": full-route shares, with outside length as its own column and row.
#   "both": both sets of columns side by side.
# route_length_qa.csv and the GIS feature classes always report both.
OUTSIDE_LENGTH = "exclude"

# Decimal places for miles and percentages in the CSV tables. Calculations, the
# length check, and GIS feature class attributes keep full precision.
CSV_DECIMALS = 1
# The QA shared-length and length-check difference columns report amounts near the
# 0.1-foot check thresholds, so they keep more places (0.000001 mile is about 0.06 inch).
QA_CSV_DECIMALS = 6

# Optional explicit geographic transformations; None selects the first suitable
# installed transformation returned by ArcPy for the input data's extent.
GTFS_TRANSFORMATION: Optional[str] = None
DISTRICT_TRANSFORMATION: Optional[str] = None

# A failed length check stops the run after the log names the route; run.log and
# the GIS output already written remain for review (unique_routes is written last).
# The tolerance is the larger of 0.1 feet and one millionth of route length.
LENGTH_CHECK_ABS_MILES = 0.1 / 5280.0
LENGTH_CHECK_REL = 1e-6
LOG_LEVEL = logging.INFO

# === END CONFIG ===

LOGGER = logging.getLogger("route_length_by_district_arcpy")
QA_DETAIL_FIELDS = {"shared_boundary_or_overlap_miles", "length_check_delta_miles"}
RouteInfo = Dict[str, str]
Piece = Tuple[Tuple[str, ...], Any]


def read_gtfs(filename: str, required: Set[str]) -> Iterator[Dict[str, str]]:
    """Read a GTFS table without coercing identifiers or literal 'NA' values."""
    path = GTFS_FOLDER / filename
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{filename} is missing fields: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{filename}, row {line_number}: too many CSV columns.")
            yield {key: (value or "").strip() for key, value in row.items()}


def route_label(route: RouteInfo) -> str:
    """Return the label used by route filters and sorting."""
    return route["route_short_name"] or route["route_id"]


def natural_key(value: str) -> List[Tuple[int, Any]]:
    """Sort route numbers numerically while also supporting alphanumeric names."""
    return [
        (0, int(token)) if token.isdigit() else (1, token.casefold())
        for token in re.split(r"(\d+)", value)
    ]


def load_routes_and_trips() -> Tuple[Dict[str, RouteInfo], Dict[str, Set[str]], Counter]:
    """Load selected routes and unique shape references, rejecting incomplete data."""
    LOGGER.info("Reading routes.txt and trips.txt from %s", GTFS_FOLDER)
    all_routes: Dict[str, RouteInfo] = {}
    for row in read_gtfs("routes.txt", {"route_id"}):
        route_id = row["route_id"]
        if not route_id or route_id in all_routes:
            raise ValueError(f"Blank or duplicate route_id in routes.txt: {route_id!r}")
        all_routes[route_id] = {
            "route_id": route_id,
            "route_short_name": row.get("route_short_name", ""),
            "route_long_name": row.get("route_long_name", ""),
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

    route_shapes: Dict[str, Set[str]] = defaultdict(set)
    trip_counts: Counter = Counter()
    missing_shapes: Counter = Counter()
    trip_ids: Set[str] = set()
    for row in read_gtfs("trips.txt", {"route_id", "trip_id", "shape_id"}):
        route_id, trip_id = row["route_id"], row["trip_id"]
        if route_id not in all_routes:
            raise ValueError(f"Trip {trip_id!r} references unknown route {route_id!r}.")
        if route_id not in routes:
            continue
        if not trip_id or trip_id in trip_ids:
            raise ValueError(f"Blank or duplicate selected trip_id: {trip_id!r}")
        trip_ids.add(trip_id)
        trip_counts[route_id] += 1
        if row["shape_id"]:
            route_shapes[route_id].add(row["shape_id"])
        else:
            missing_shapes[route_id] += 1
    if missing_shapes:
        raise ValueError(f"Trips with blank shape_id, by route_id: {dict(missing_shapes)}")
    LOGGER.info("Selected %d routes and %d trips.", len(routes), sum(trip_counts.values()))
    return routes, route_shapes, trip_counts


def choose_transformation(
    source: Any, target: Any, extent: Any, override: Optional[str]
) -> Optional[str]:
    """Choose a suitable datum transformation, never silently omit a needed one."""
    if not source or source.name == "Unknown":
        raise ValueError("Input spatial reference is unknown; define it before running.")
    if override:
        LOGGER.info("Using configured transformation: %s", override)
        return override
    if source.GCS.datumName == target.GCS.datumName:
        return None
    candidates = arcpy.ListTransformations(source, target, extent)
    if not candidates:
        raise ValueError(f"No datum transformation found: {source.name} -> {target.name}")
    LOGGER.info("Transformation %s -> %s: %s", source.name, target.name, candidates[0])
    return candidates[0]


def project_geometry(geometry: Any, target: Any, transformation: Optional[str]) -> Any:
    """Project a geometry using the selected transformation when necessary."""
    if transformation:
        return geometry.projectAs(target, transformation)
    return geometry.projectAs(target)


def union_geometries(geometries: Sequence[Any]) -> Any:
    """Union geometries in balanced passes to avoid repeatedly growing one object."""
    items = list(geometries)
    if not items:
        raise ValueError("Cannot union an empty geometry collection.")
    while len(items) > 1:
        items = [
            items[index].union(items[index + 1]) if index + 1 < len(items) else items[index]
            for index in range(0, len(items), 2)
        ]
    # Normalize coincident/retraced segments even for a route with only one shape.
    return items[0].union(items[0])


def load_shape_geometries(needed: Set[str], target: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Validate and project the shapes referenced by selected routes."""
    if not needed:
        return {}, None
    LOGGER.info("Reading shapes.txt for %d shapes.", len(needed))
    points: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    required = {"shape_id", "shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"}
    for row in read_gtfs("shapes.txt", required):
        shape_id = row["shape_id"]
        if shape_id not in needed:
            continue
        try:
            sequence = int(row["shape_pt_sequence"])
            lon, lat = float(row["shape_pt_lon"]), float(row["shape_pt_lat"])
        except ValueError as exc:
            raise ValueError(f"Invalid point values for shape {shape_id!r}: {row}") from exc
        if sequence < 0 or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"Invalid sequence/coordinates for shape {shape_id!r}: {row}")
        points[shape_id].append((sequence, lon, lat))
    missing = needed - set(points)
    if missing:
        raise ValueError(f"Referenced shape_ids absent from shapes.txt: {sorted(missing)}")

    source = arcpy.SpatialReference(4326)
    all_points = [point for values in points.values() for point in values]
    LOGGER.info("Read %d shape points; building and projecting shapes.", len(all_points))
    extent = arcpy.Extent(
        min(point[1] for point in all_points),
        min(point[2] for point in all_points),
        max(point[1] for point in all_points),
        max(point[2] for point in all_points),
    )
    transformation = choose_transformation(source, target, extent, GTFS_TRANSFORMATION)
    geometries: Dict[str, Any] = {}
    for shape_id, values in points.items():
        values.sort()
        if len({point[0] for point in values}) != len(values):
            raise ValueError(f"Duplicate shape_pt_sequence for shape {shape_id!r}.")
        coords: List[Tuple[float, float]] = []
        for _, lon, lat in values:
            if not coords or coords[-1] != (lon, lat):
                coords.append((lon, lat))
        if len(set(coords)) < 2:
            raise ValueError(f"Shape {shape_id!r} has fewer than two distinct coordinates.")
        geometry = arcpy.Polyline(arcpy.Array([arcpy.Point(*xy) for xy in coords]), source)
        geometry = project_geometry(geometry, target, transformation)
        if geometry.pointCount < 2 or geometry.length <= 0:
            raise ValueError(f"Shape {shape_id!r} has no measurable projected length.")
        geometries[shape_id] = geometry
    LOGGER.info("Built %d projected GTFS shapes.", len(geometries))
    return geometries, transformation


def find_district_field() -> str:
    """Resolve a district field without mistaking generic NAME for district names."""
    fields = [
        field
        for field in arcpy.ListFields(str(DISTRICTS_FC))
        if field.type not in {"OID", "Geometry", "Blob", "Raster"}
    ]
    available = ", ".join(f"{field.name} ({field.type})" for field in fields)
    LOGGER.info("Available district fields: %s", available)
    if DISTRICT_NAME_FIELD:
        matches = [f.name for f in fields if f.name.casefold() == DISTRICT_NAME_FIELD.casefold()]
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
        matches = [
            field.name
            for field in fields
            if normalize(candidate) in {normalize(field.name), normalize(field.aliasName or "")}
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(
        "Set DISTRICT_NAME_FIELD to the actual district-name field. Generic NAME is "
        "not auto-selected because it may contain supervisor names. "
        f"Available fields: {available}"
    )


def load_districts(target: Any) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """Project district polygons and union multiple records with the same label."""
    LOGGER.info("Reading districts from %s", DISTRICTS_FC)
    description = arcpy.Describe(str(DISTRICTS_FC))
    if description.shapeType != "Polygon":
        raise ValueError("DISTRICTS_FC must contain polygons.")
    field = find_district_field()
    transformation = choose_transformation(
        description.spatialReference, target, description.extent, DISTRICT_TRANSFORMATION
    )
    grouped: Dict[str, List[Any]] = defaultdict(list)
    with arcpy.da.SearchCursor(str(DISTRICTS_FC), [field, "SHAPE@"]) as cursor:
        for value, geometry in cursor:
            label = str(value).strip() if value is not None else ""
            if not label or label.casefold() == OUTSIDE_LABEL.casefold():
                raise ValueError(f"Blank or reserved district label: {label!r}")
            if geometry is None or geometry.pointCount == 0 or geometry.area <= 0:
                raise ValueError(f"Null, empty, or zero-area district polygon: {label}")
            projected = project_geometry(geometry, target, transformation)
            if projected.pointCount == 0 or projected.area <= 0:
                raise ValueError(f"District polygon became empty after projection: {label}")
            grouped[label].append(projected)
    if not grouped:
        raise ValueError("No district polygons found.")
    districts = {label: union_geometries(grouped[label]) for label in sorted(grouped)}
    LOGGER.info("District field: %s. Values: %s", field, ", ".join(districts))
    return districts, field, transformation


def length_miles(geometry: Any) -> float:
    """Measure length in miles with the same planar method throughout."""
    return float(geometry.getLength("PLANAR", "MILES"))


def split_by_district(route_geometry: Any, districts: Dict[str, Any]) -> List[Piece]:
    """Partition route linework by its complete set of district memberships.

    Each piece appears exactly once in this partition, including shared boundaries
    and outside pieces. Subtract the intersected LINE from the original line so
    boundary inclusion is consistent between intersection and difference.
    """
    pieces: List[Piece] = [((), route_geometry)]
    for label, polygon in districts.items():
        updated: List[Piece] = []
        for members, line in pieces:
            if line.disjoint(polygon):
                updated.append((members, line))
                continue
            inside = line.intersect(polygon, 2)
            if inside.pointCount == 0 or inside.length <= 0:
                updated.append((members, line))
                continue
            updated.append((members + (label,), inside))
            outside = line.difference(inside)
            if outside.pointCount > 0 and outside.length > 0:
                updated.append((members, outside))
        pieces = updated
    return pieces


def summarize_pieces(
    pieces: Sequence[Piece], labels: Sequence[str], total_miles: float
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """Allocate shared pieces equally and verify conservation of route length."""
    allocated = dict.fromkeys([*labels, OUTSIDE_LABEL], 0.0)
    raw = dict.fromkeys([*labels, OUTSIDE_LABEL], 0.0)
    shared_miles = 0.0
    for members, geometry in pieces:
        miles = length_miles(geometry)
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


def inside_percentage(label: str, allocated_miles: float, inside_miles: float) -> Optional[float]:
    """Calculate a district-only share; outside length contributes zero."""
    if inside_miles <= 0:
        return None
    return 0.0 if label == OUTSIDE_LABEL else allocated_miles / inside_miles * 100


def create_feature_class(
    gdb: Path,
    name: str,
    geometry_type: str,
    spatial_reference: Any,
    fields: Sequence[Tuple[str, str, Optional[int]]],
) -> str:
    """Create one output feature class with explicitly sized text fields."""
    path = str(gdb / name)
    arcpy.management.CreateFeatureclass(
        str(gdb),
        name,
        geometry_type,
        spatial_reference=spatial_reference,
        has_m="DISABLED",
        has_z="DISABLED",
    )
    for field, field_type, size in fields:
        if size is None:
            arcpy.management.AddField(path, field, field_type)
        else:
            arcpy.management.AddField(path, field, field_type, field_length=size)
    return path


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Dict[str, Any]],
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
            formatted: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, float):
                    places = QA_CSV_DECIMALS if key in QA_DETAIL_FIELDS else CSV_DECIMALS
                    # Fixed notation avoids 1.9e-05; adding 0.0 writes -0.0 as 0.0.
                    value = f"{round(value, places) + 0.0:.{places}f}"
                formatted[key] = value
            writer.writerow(formatted)
    LOGGER.info("Wrote %s (%d rows).", path.name, len(rows))


def insert_rows(feature_class: str, fields: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write one feature class and release its cursor before another writer opens."""
    with arcpy.da.InsertCursor(feature_class, list(fields)) as cursor:
        for row in rows:
            cursor.insertRow(row)
    LOGGER.info("Wrote %d features to %s.", len(rows), Path(feature_class).name)


def run_analysis(run_dir: Path) -> None:
    """Build route linework, allocate district lengths, and write tables and GIS data."""
    target = arcpy.SpatialReference(ANALYSIS_EPSG)
    if target.type != "Projected":
        raise ValueError("ANALYSIS_EPSG must be a suitable local projected coordinate system.")
    if OUTSIDE_LENGTH not in {"exclude", "include", "both"}:
        raise ValueError('OUTSIDE_LENGTH must be "exclude", "include", or "both".')
    LOGGER.info("Analysis coordinate system: %s", target.name)
    routes, route_shapes, trip_counts = load_routes_and_trips()
    districts, district_field, district_transform = load_districts(target)
    labels = [*districts, OUTSIDE_LABEL]
    needed = set().union(*route_shapes.values()) if route_shapes else set()
    shapes, gtfs_transform = load_shape_geometries(needed, target)

    gdb = run_dir / "route_district.gdb"
    LOGGER.info("Creating %s", gdb)
    arcpy.management.CreateFileGDB(str(run_dir), gdb.name)
    route_fields = [
        ("route_id", "TEXT", max(64, max(len(r) for r in routes))),
        ("route_name", "TEXT", max(64, max(len(route_label(r)) for r in routes.values()))),
    ]
    district_size = max(64, max(map(len, labels)))
    route_fc = create_feature_class(
        gdb,
        "unique_routes",
        "POLYLINE",
        target,
        [
            *route_fields,
            ("total_mi", "DOUBLE", None),
            ("inside_mi", "DOUBLE", None),
            ("outside_mi", "DOUBLE", None),
        ],
    )
    segment_fields = [
        *route_fields,
        ("district", "TEXT", district_size),
        ("share_n", "LONG", None),
        ("geom_mi", "DOUBLE", None),
        ("alloc_mi", "DOUBLE", None),
        ("in_dist_mi", "DOUBLE", None),
        ("pct_route", "DOUBLE", None),
        ("pct_inside", "DOUBLE", None),
    ]
    segment_fc = create_feature_class(gdb, "district_segments", "POLYLINE", target, segment_fields)
    district_fc = create_feature_class(
        gdb, "districts_projected", "POLYGON", target, [("district", "TEXT", district_size)]
    )
    insert_rows(district_fc, ["district", "SHAPE@"], list(districts.items()))

    wide_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    qa_rows: List[Dict[str, Any]] = []
    ordered = sorted(routes, key=lambda key: (natural_key(route_label(routes[key])), key))
    route_cursor_fields = [
        "route_id",
        "route_name",
        "total_mi",
        "inside_mi",
        "outside_mi",
        "SHAPE@",
    ]
    segment_cursor_fields = [field[0] for field in segment_fields] + ["SHAPE@"]
    route_feature_rows: List[List[Any]] = []
    LOGGER.info("Allocating %d routes among %d districts.", len(ordered), len(districts))
    # Keep only the segment writer open; route features are written after it closes.
    with arcpy.da.InsertCursor(segment_fc, segment_cursor_fields) as segment_cursor:
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

            geometry = union_geometries([shapes[sid] for sid in sorted(route_shapes[route_id])])
            total = length_miles(geometry)
            pieces = split_by_district(geometry, districts)
            allocated, raw, shared, delta = summarize_pieces(pieces, list(districts), total)
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
            for label in labels:
                wide[f"{label} (full route %)"] = allocated[label] / total * 100
                wide[f"{label} (district-only %)"] = inside_percentage(
                    label, allocated[label], inside
                )
            wide["full_route_percent_sum"] = math.fsum(allocated.values()) / total * 100
            wide["district_only_percent_sum"] = (
                math.fsum(wide[f"{label} (district-only %)"] for label in labels)
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
                miles = length_miles(line)
                recipients = members or (OUTSIDE_LABEL,)
                for label in recipients:
                    allocated_miles = miles / len(recipients)
                    segment_cursor.insertRow(
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

    statuses = Counter(row["status"] for row in qa_rows)
    LOGGER.info("Processed %d routes. Status counts: %s", len(qa_rows), dict(statuses))
    insert_rows(route_fc, route_cursor_fields, route_feature_rows)

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
    omit: Set[str] = set()
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
        "districts_fc": str(DISTRICTS_FC),
        "district_name_field": district_field,
        "district_labels": list(districts),
        "analysis_epsg": ANALYSIS_EPSG,
        "analysis_crs": target.name,
        "gtfs_transformation": gtfs_transform,
        "district_transformation": district_transform,
        "routes_to_include": sorted(ROUTES_TO_INCLUDE),
        "routes_to_exclude": sorted(ROUTES_TO_EXCLUDE),
        "route_grouping": "route_id",
        "length_method": "PLANAR",
        "length_units": "MILES",
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
        "arcgis_version": arcpy.GetInstallInfo().get("Version", "Unknown"),
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
    run_dir: Optional[Path] = None
    try:
        # Check inputs before creating an output folder, so placeholder paths do nothing.
        paths = [GTFS_FOLDER / name for name in ("routes.txt", "trips.txt", "shapes.txt")]
        missing = [str(path) for path in paths if not path.is_file()]
        if not arcpy.Exists(str(DISTRICTS_FC)):
            missing.append(str(DISTRICTS_FC))
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
        # Each run uses new output names, avoiding locks and stale temporary layers.
        # No map extent, feature-layer selection, or inherited output CRS is used.
        with arcpy.EnvManager(
            overwriteOutput=False, outputCoordinateSystem=arcpy.SpatialReference(ANALYSIS_EPSG)
        ):
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
