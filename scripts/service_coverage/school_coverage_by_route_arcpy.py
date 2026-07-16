"""Roll school counts and total enrollment up to GTFS routes using ArcPy.

ArcPy port of ``school_coverage_by_route_gpd.py`` for environments running
ArcGIS Pro's bundled Python (no geopandas/shapely). Like the geopandas
version, it is an intermediate ("prep") script: it turns a point layer of
schools into a single, route-keyed feature table that the modeling pipeline
(``scripts/modeling/prep_features_public.py`` → ``monthly_ridership_model.py``)
can join onto the ridership anchor by ``route_id``.

It buffers each route's stops (a simple fixed-radius catchment is intentional
for now) and, for every route, counts the schools whose point falls inside the
catchment and sums their enrollment. The result is one row per route_id with
``schools_served`` and ``enrollment_served`` — the exact columns the
orchestrator registry already describes for ``school_coverage_by_route.csv``.

Inputs
------
- A GTFS folder containing routes.txt, trips.txt, stop_times.txt, stops.txt
  (and shapes.txt only when ``USE_SHAPE_BUFFER`` is enabled).
- One or more school *point* layers carrying an enrollment column. These are
  the enrollment-joined points written by
  ``national_data_tools/schools_prep_join_gpd.py``
  (``va_md_dc_<type>_schools_enrollment.gpkg``); point SCHOOLS_PATH at a single
  file or a folder to combine public + private + postsec into one rollup.
  GeoPackages, shapefiles, and GeoJSON files are supported (GeoJSON is
  converted in-memory via ``arcpy.conversion.JSONToFeatures``).

Outputs
-------
- ``school_coverage_by_route.csv`` — columns ``route_id``, ``route_short_name``
  (when available), ``schools_served``, ``enrollment_served`` (grand total), and
  the grade-band breakout ``enrollment_1_8_served``, ``enrollment_9_12_served``,
  ``enrollment_postsec_served``.

Typical usage
-------------
Update the paths in the CONFIGURATION section and run from a shell, ArcGIS
Pro's Python window, or a Jupyter notebook using ArcGIS Pro's bundled Python.

Assumptions
-----------
- The analysis spatial reference is projected; the buffer distance is given in
  feet and converted to the spatial reference's linear unit.
- Schools with no matched enrollment (NaN) still count toward ``schools_served``
  but contribute 0 to ``enrollment_served``.

Requires
--------
ArcGIS Pro (arcpy) and pandas (bundled with Pro).
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import arcpy
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top-level directories
GTFS_DIR = Path(r"Path\To\Your\GTFS_Data")  # folder containing GTFS .txt files
SCHOOLS_PATH = Path(r"Path\To\Your\Schools_Data")  # school point file, or a folder of them
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")  # where the rollup CSV is written

# When SCHOOLS_PATH is a folder, every file matching these glob patterns is loaded
# and combined (so public + private + postsec roll up together). The patterns are
# matched recursively. Ignored when SCHOOLS_PATH points straight at a file.
SCHOOLS_GLOBS: Tuple[str, ...] = (
    "*schools_enrollment*.gpkg",
    "*schools_enrollment*.shp",
    "*schools_enrollment*.geojson",
)

# Column on the school layer holding total enrollment. This is the name
# schools_prep_join_gpd.py writes for every source (CCD / ELSI / IPEDS).
ENROLLMENT_COLUMN = "enroll_total"

# Enrollment is also broken into grade bands, summed from the per-source ``g_*``
# columns schools_prep_join_gpd.py emits, so the rollup ships separate
# grades-1-8, grades-9-12, and postsecondary totals alongside the grand total:
#   - ELSI (public/private): g_grades_1_8 -> 1-8, g_grades_9_12 -> 9-12
#   - CCD (public): g_grade_1..g_grade_8 -> 1-8, g_grade_9..g_grade_12 -> 9-12
#   - IPEDS (postsec, detected via g_undergrad/g_graduate): enroll_total -> postsec
# Pre-K / kindergarten / ungraded counts stay in the grand total but fall in no
# band. These canonical band names are intentionally not configurable.
ENROLL_TOTAL_COL = "enroll_total"
ENROLL_1_8_COL = "enroll_1_8"
ENROLL_9_12_COL = "enroll_9_12"
ENROLL_POSTSEC_COL = "enroll_postsec"

# Optional filter: only analyze these route_id values.
# Leave empty (`[]`) to process every route in routes.txt
ROUTE_FILTER: List[str] = []

# Analysis options
USE_SHAPE_BUFFER = False  # False → buffer stops (simple catchment); True → route geometry
BUFFER_DIST_FT = 1320.0  # ¼ mile in feet

# Output filename — matches the orchestrator registry's school_coverage_by_route.csv.
OUTPUT_CSV_NAME = "school_coverage_by_route.csv"

# Projected spatial reference (WKID) used for buffering and the point-in-polygon
# tests. EPSG:3857 (Web Mercator) works globally; its latitude-dependent scale
# distortion is corrected automatically when buffering (see
# _buffer_distance_in_sr_units). Swap for a local CRS (e.g. 2283 for northern
# Virginia in feet) when higher spatial accuracy is needed.
PROJECTED_CRS_WKID = 3857

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def _load_gtfs_tables(gtfs_dir: Path, need_shapes: bool) -> Mapping[str, pd.DataFrame]:
    """Load the GTFS text files needed to build route catchments.

    Args:
        gtfs_dir: Directory containing GTFS .txt files.
        need_shapes: When True, shapes.txt is also required (shape-buffer mode).

    Returns:
        Mapping keyed by table name (without .txt) to DataFrame.

    Raises:
        FileNotFoundError: If a required file is missing.
    """
    names = ["routes", "trips", "stop_times", "stops"]
    if need_shapes:
        names.append("shapes")

    tables: Dict[str, pd.DataFrame] = {}
    for name in names:
        path = gtfs_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[name] = pd.read_csv(path, dtype={"route_id": str})
        logging.debug("Loaded %s (%d rows)", name, len(tables[name]))
    return tables


def _buffer_distance_in_sr_units(
    target_sr: arcpy.SpatialReference,
    buffer_dist_ft: float,
    mean_latitude: Optional[float] = None,
) -> float:
    """Convert a buffer distance in feet to the linear units of *target_sr*.

    Web Mercator (WKID 3857 / 102100) inflates distances by roughly
    ``1/cos(latitude)``: at 39°N a true quarter mile spans ~1.29x as many map
    "meters", so a buffer drawn in raw map units under-covers the ground by
    the same factor. When *target_sr* is Web Mercator and *mean_latitude* is
    supplied, the distance is scaled up accordingly so it spans a true ground
    distance at that latitude.

    Args:
        target_sr: Projected spatial reference used for geometry operations.
        buffer_dist_ft: Buffer distance in feet.
        mean_latitude: Mean WGS 84 latitude of the analysis features, used to
            correct Web Mercator's scale distortion. Ignored for other CRSs.

    Returns:
        Buffer distance expressed in the units of *target_sr*.

    Raises:
        ValueError: If *target_sr* is not a projected spatial reference.
    """
    if target_sr.type != "Projected":
        raise ValueError(
            f"PROJECTED_CRS_WKID must reference a projected spatial reference; "
            f"got {target_sr.name!r} ({target_sr.type})."
        )
    meters_per_unit = target_sr.metersPerUnit or 1.0
    buffer_units = buffer_dist_ft * 0.3048 / meters_per_unit
    if (
        target_sr.factoryCode in (3857, 102100)
        and mean_latitude is not None
        and -89.0 < mean_latitude < 89.0
    ):
        scale = 1.0 / math.cos(math.radians(mean_latitude))
        buffer_units *= scale
        logging.info(
            "Web Mercator inflates distances by %.4f at latitude %.3f; scaling "
            "the buffer to preserve ground distance.",
            scale,
            mean_latitude,
        )
    logging.debug(
        "Buffer distance: %.2f ft -> %.2f %s",
        buffer_dist_ft,
        buffer_units,
        target_sr.linearUnitName,
    )
    return buffer_units


def _prepare_route_buffers(
    tables: Mapping[str, pd.DataFrame],
    use_shape_buffer: bool,
    buffer_dist_ft: float,
    target_sr: arcpy.SpatialReference,
    route_filter: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Return one buffered catchment geometry per route_id.

    Depending on *use_shape_buffer*, the buffer is built around the union of
    (a) the route's shape(s) or (b) all of its stops. The buffer distance is
    given in feet and converted to the linear units of *target_sr*.

    Args:
        tables: GTFS tables from :func:`_load_gtfs_tables`.
        use_shape_buffer: Buffer route geometry when True, else buffer stops.
        buffer_dist_ft: Catchment radius in feet.
        target_sr: Projected spatial reference used for buffering.
        route_filter: Optional list of route_id values to keep (empty = all).

    Returns:
        List of ``{"route_id": str, "geometry": arcpy polygon}`` records.

    Raises:
        ValueError: If shape-buffer mode is requested but trips.txt or
            shapes.txt is malformed.
    """
    wgs84_sr = arcpy.SpatialReference(4326)

    trips = tables["trips"].copy()
    trips["route_id"] = trips["route_id"].astype(str)

    # stop_id -> (lon, lat), and route_id -> unique stop_ids (the default
    # catchment buffers stops).
    stops = tables["stops"][["stop_id", "stop_lat", "stop_lon"]]

    lat_values = pd.to_numeric(stops["stop_lat"], errors="coerce").dropna()
    mean_latitude = float(lat_values.mean()) if not lat_values.empty else None
    buff_dist = _buffer_distance_in_sr_units(target_sr, buffer_dist_ft, mean_latitude)
    stop_coords = {
        row.stop_id: (row.stop_lon, row.stop_lat)
        for row in stops.itertuples(index=False)
        if pd.notna(row.stop_lat) and pd.notna(row.stop_lon)
    }
    route_stop_ids = (
        tables["stop_times"][["trip_id", "stop_id"]]
        .merge(trips[["trip_id", "route_id"]], on="trip_id", how="inner")
        .drop_duplicates(subset=["route_id", "stop_id"])
        .groupby("route_id")["stop_id"]
        .apply(list)
    )

    shape_lines: Dict[str, arcpy.Polyline] = {}
    route_shapes: Optional[pd.Series] = None
    if use_shape_buffer:
        if "shape_id" not in trips.columns:
            raise ValueError("trips.txt missing shape_id column (required for shape-buffer mode)")
        shapes_df = tables["shapes"]
        if {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.difference(
            shapes_df.columns
        ):
            raise ValueError("shapes.txt missing required columns")
        shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
        for shape_id, grp in shapes_df.groupby("shape_id"):
            pts = [
                arcpy.Point(lon, lat) for lon, lat in zip(grp["shape_pt_lon"], grp["shape_pt_lat"])
            ]
            if len(pts) < 2:
                continue
            line = arcpy.Polyline(arcpy.Array(pts), wgs84_sr)
            shape_lines[str(shape_id)] = line.projectAs(target_sr)
        route_shapes = (
            trips.dropna(subset=["shape_id"])
            .drop_duplicates(subset=["route_id", "shape_id"])
            .groupby("route_id")["shape_id"]
            .apply(list)
        )

    route_ids = route_shapes.index if route_shapes is not None else trips["route_id"].unique()

    buffers: List[Dict[str, object]] = []
    for route_id in route_ids:
        route_id = str(route_id)
        if route_filter and route_id not in route_filter:
            continue

        geom: Optional[arcpy.Geometry] = None
        if use_shape_buffer and route_shapes is not None:
            lines = [
                shape_lines[str(s)] for s in route_shapes.loc[route_id] if str(s) in shape_lines
            ]
            if lines:
                geom = lines[0]
                for line in lines[1:]:
                    geom = geom.union(line)
        else:
            coords = [
                stop_coords[sid] for sid in route_stop_ids.get(route_id, []) if sid in stop_coords
            ]
            if coords:
                multi = arcpy.Multipoint(
                    arcpy.Array([arcpy.Point(lon, lat) for lon, lat in coords]), wgs84_sr
                )
                geom = multi.projectAs(target_sr)

        if geom is None:
            logging.warning("No geometry for route %s – skipped", route_id)
            continue

        buffers.append({"route_id": route_id, "geometry": geom.buffer(buff_dist)})

    return buffers


def _band_of_grade_column(column: str) -> Optional[str]:
    """Map a ``schools_prep_join_gpd.py`` grade column to a band, or None.

    Handles both the ELSI banded form (``g_grades_1_8`` / ``g_grades_9_12``) and
    the CCD per-grade form (``g_grade_1`` … ``g_grade_12``). Pre-K, kindergarten,
    and ungraded columns return None (they belong to no requested band).

    Returns:
        ``"1_8"``, ``"9_12"``, or ``None``.
    """
    name = column.lower()
    if "grades_1_8" in name:
        return "1_8"
    if "grades_9_12" in name:
        return "9_12"
    match = re.fullmatch(r"g_grade_(\d+)", name)
    if match:
        grade = int(match.group(1))
        if 1 <= grade <= 8:
            return "1_8"
        if 9 <= grade <= 12:
            return "9_12"
    return None


def _normalize_enrollment(schools: pd.DataFrame, enrollment_column: str) -> pd.DataFrame:
    """Add the four canonical enrollment columns to one school table.

    Computes ``enroll_total`` plus the grade-band breakout (``enroll_1_8``,
    ``enroll_9_12``, ``enroll_postsec``) from whatever ``g_*`` columns the table
    carries. Postsecondary layers are detected by their ``g_undergrad`` /
    ``g_graduate`` columns and routed wholesale into ``enroll_postsec``; every
    other layer is treated as K-12 and binned by grade. NaN counts contribute 0.

    Args:
        schools: One school layer's attributes as read from disk.
        enrollment_column: Column holding total enrollment on this layer.

    Returns:
        The table with the four canonical numeric columns added.
    """
    out = schools.copy()
    if enrollment_column in out.columns:
        total = pd.to_numeric(out[enrollment_column], errors="coerce")
    else:
        total = pd.Series(0.0, index=out.index)

    cols_lower = {c.lower() for c in out.columns}
    is_postsec = "g_undergrad" in cols_lower or "g_graduate" in cols_lower

    band_1_8 = pd.Series(0.0, index=out.index)
    band_9_12 = pd.Series(0.0, index=out.index)
    band_postsec = pd.Series(0.0, index=out.index)

    if is_postsec:
        band_postsec = total.fillna(0.0)
    else:
        for col in out.columns:
            band = _band_of_grade_column(str(col))
            if band is None:
                continue
            values = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            if band == "1_8":
                band_1_8 = band_1_8 + values
            else:
                band_9_12 = band_9_12 + values

    out[ENROLL_TOTAL_COL] = total
    out[ENROLL_1_8_COL] = band_1_8
    out[ENROLL_9_12_COL] = band_9_12
    out[ENROLL_POSTSEC_COL] = band_postsec
    return out


# Canonical enrollment columns carried through the join, in output order.
_BAND_COLUMNS: Tuple[str, ...] = (
    ENROLL_TOTAL_COL,
    ENROLL_1_8_COL,
    ENROLL_9_12_COL,
    ENROLL_POSTSEC_COL,
)

# Maps each canonical enrollment column to its route-level output column name.
_OUTPUT_ENROLLMENT_COLUMNS: Dict[str, str] = {
    ENROLL_TOTAL_COL: "enrollment_served",
    ENROLL_1_8_COL: "enrollment_1_8_served",
    ENROLL_9_12_COL: "enrollment_9_12_served",
    ENROLL_POSTSEC_COL: "enrollment_postsec_served",
}


def _school_datasets_from_file(path: Path, memory_index: int) -> Tuple[List[str], List[str]]:
    """Resolve one school file into ArcPy-readable point dataset paths.

    Shapefiles are returned as-is; GeoPackages are expanded into their point
    feature classes; GeoJSON files are converted into ``memory`` feature
    classes (which the caller must delete).

    Args:
        path: A school vector file discovered under SCHOOLS_PATH.
        memory_index: Unique index used to name in-memory conversions.

    Returns:
        Tuple of (dataset paths to read, temporary dataset paths to delete).
    """
    suffix = path.suffix.lower()

    if suffix == ".gpkg":
        original_ws = arcpy.env.workspace
        try:
            arcpy.env.workspace = str(path)
            feature_classes = arcpy.ListFeatureClasses() or []
        finally:
            arcpy.env.workspace = original_ws
        datasets = []
        for name in feature_classes:
            dataset = os.path.join(str(path), name)
            shape_type = arcpy.Describe(dataset).shapeType
            if shape_type in ("Point", "Multipoint"):
                datasets.append(dataset)
            else:
                logging.debug("Skipping non-point layer %s in %s", name, path.name)
        if not datasets:
            logging.warning("No point layers found in %s", path.name)
        return datasets, []

    if suffix in (".geojson", ".json"):
        out_fc = f"memory\\schools_geojson_{memory_index}"
        arcpy.conversion.JSONToFeatures(str(path), out_fc, "POINT")
        return [out_fc], [out_fc]

    return [str(path)], []


def _read_school_points(
    dataset: str,
    target_sr: arcpy.SpatialReference,
    enrollment_column: str,
) -> pd.DataFrame:
    """Read one school point dataset into a normalized attribute table.

    Args:
        dataset: ArcPy-readable point dataset path.
        target_sr: Spatial reference the point coordinates are projected into.
        enrollment_column: Column holding total enrollment on this layer.

    Returns:
        DataFrame with ``x``/``y`` (in *target_sr*), an arcpy ``geometry``
        column, and the four canonical enrollment columns.
    """
    desc = arcpy.Describe(dataset)
    if not desc.spatialReference or desc.spatialReference.name == "Unknown":
        logging.warning(
            "Layer %s has an unknown spatial reference; coordinates are used as-is.",
            dataset,
        )

    field_names = [f.name for f in arcpy.ListFields(dataset) if f.type not in ("Geometry", "OID")]
    enroll_field = next((n for n in field_names if n.lower() == enrollment_column.lower()), None)
    if enroll_field is None:
        logging.warning(
            "Enrollment column %r missing in %s; counts only.", enrollment_column, dataset
        )
    grade_fields = [n for n in field_names if _band_of_grade_column(n) is not None]
    postsec_fields = [n for n in field_names if n.lower() in ("g_undergrad", "g_graduate")]

    enroll_fields = [enroll_field] if enroll_field else []
    attr_fields = list(dict.fromkeys(enroll_fields + grade_fields + postsec_fields))
    cursor_fields = ["SHAPE@XY"] + attr_fields

    rows: List[Dict[str, object]] = []
    with arcpy.da.SearchCursor(dataset, cursor_fields, spatial_reference=target_sr) as cursor:
        for row in cursor:
            xy = row[0]
            if xy is None:
                continue
            record: Dict[str, object] = {"x": xy[0], "y": xy[1]}
            record.update(zip(attr_fields, row[1:]))
            rows.append(record)

    schools = pd.DataFrame(rows, columns=["x", "y"] + attr_fields)
    schools = _normalize_enrollment(schools, enrollment_column)
    schools["geometry"] = [
        arcpy.PointGeometry(arcpy.Point(x, y), target_sr)
        for x, y in zip(schools["x"], schools["y"])
    ]
    return schools[["x", "y", "geometry", *(_BAND_COLUMNS)]]


def load_schools_points(
    schools_path: Path,
    target_sr: arcpy.SpatialReference,
    schools_globs: Sequence[str] = SCHOOLS_GLOBS,
    enrollment_column: str = ENROLLMENT_COLUMN,
) -> pd.DataFrame:
    """Load school point layer(s), normalize enrollment, combine, and reproject.

    Accepts either a single vector file or a folder. For a folder, every file
    matching *schools_globs* (recursively) is read and concatenated, so the
    public / private / postsec outputs of ``schools_prep_join_gpd.py`` roll up
    together. Each layer is normalized to the four canonical enrollment columns
    (total + grades-1-8 / grades-9-12 / postsecondary bands) so layers with
    different source schemas stack cleanly.

    Args:
        schools_path: A school vector file, or a folder containing such files.
        target_sr: Projected spatial reference for the combined point table.
        schools_globs: Glob patterns used when *schools_path* is a folder.
        enrollment_column: Column holding total enrollment on each layer.

    Returns:
        Point DataFrame in *target_sr* with ``x``/``y``, an arcpy ``geometry``
        column, and the four canonical enrollment columns.

    Raises:
        FileNotFoundError: If no school layer is found.
    """
    if schools_path.is_dir():
        paths = sorted({p for pattern in schools_globs for p in schools_path.rglob(pattern)})
        if not paths:
            raise FileNotFoundError(
                f"No school layers matching {list(schools_globs)} under {schools_path}"
            )
    elif schools_path.exists():
        paths = [schools_path]
    else:
        raise FileNotFoundError(schools_path)

    frames: List[pd.DataFrame] = []
    for index, path in enumerate(paths):
        datasets, temporary = _school_datasets_from_file(path, index)
        try:
            loaded = 0
            for dataset in datasets:
                schools = _read_school_points(dataset, target_sr, enrollment_column)
                frames.append(schools)
                loaded += len(schools)
            logging.info("Loaded %d schools from %s", loaded, path.name)
        finally:
            for temp_fc in temporary:
                arcpy.management.Delete(temp_fc)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["x", "y", "geometry", *(_BAND_COLUMNS)])
    logging.info("Combined %d school points from %d layer(s)", len(combined), len(paths))
    return combined


def summarize_schools_by_route(
    route_buffers: List[Dict[str, object]],
    schools: pd.DataFrame,
) -> pd.DataFrame:
    """Count schools and sum enrollment (overall + by band) per route catchment.

    Args:
        route_buffers: One buffered catchment per route_id (from
            :func:`_prepare_route_buffers`).
        schools: School points carrying the canonical enrollment columns
            (from :func:`load_schools_points`), in the same spatial reference
            as *route_buffers*.

    Returns:
        DataFrame with one row per route_id and columns ``schools_served``,
        ``enrollment_served``, ``enrollment_1_8_served``,
        ``enrollment_9_12_served``, and ``enrollment_postsec_served``. Routes
        with no schools nearby report zeros.
    """
    count_cols = list(_OUTPUT_ENROLLMENT_COLUMNS.values())
    xs = schools["x"].to_numpy(dtype=float) if not schools.empty else None
    ys = schools["y"].to_numpy(dtype=float) if not schools.empty else None

    records: List[Dict[str, object]] = []
    seen: set = set()
    for buffer_rec in route_buffers:
        route_id = str(buffer_rec["route_id"])
        if route_id in seen:
            continue
        seen.add(route_id)

        record: Dict[str, object] = {"route_id": route_id, "schools_served": 0}
        record.update({c: 0.0 for c in count_cols})

        if not schools.empty:
            geom = buffer_rec["geometry"]
            extent = geom.extent
            candidates = schools[
                (xs >= extent.XMin)
                & (xs <= extent.XMax)
                & (ys >= extent.YMin)
                & (ys <= extent.YMax)
            ]
            if not candidates.empty:
                inside = candidates[[not geom.disjoint(pt) for pt in candidates["geometry"]]]
                record["schools_served"] = len(inside)
                for band_col, out_col in _OUTPUT_ENROLLMENT_COLUMNS.items():
                    record[out_col] = float(inside[band_col].sum())

        records.append(record)

    summary = pd.DataFrame(records, columns=["route_id", "schools_served"] + count_cols)
    summary["schools_served"] = summary["schools_served"].fillna(0).astype(int)
    for out_col in count_cols:
        summary[out_col] = summary[out_col].fillna(0).round(0).astype(int)
    return summary


def _attach_route_short_name(summary: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable ``route_short_name`` column when routes.txt carries one."""
    if "route_short_name" not in routes_df.columns:
        return summary
    lookup = routes_df.assign(route_id=routes_df["route_id"].astype(str))[
        ["route_id", "route_short_name"]
    ].drop_duplicates(subset="route_id")
    merged = summary.merge(lookup, on="route_id", how="left")
    cols = [
        "route_id",
        "route_short_name",
        "schools_served",
        *_OUTPUT_ENROLLMENT_COLUMNS.values(),
    ]
    return merged[cols]


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Build the route-level school coverage rollup and write it to CSV.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if (
        GTFS_DIR == Path(r"Path\To\Your\GTFS_Data")
        or SCHOOLS_PATH == Path(r"Path\To\Your\Schools_Data")
        or OUTPUT_DIR == Path(r"Path\To\Your\Output_Folder")
    ):
        logging.warning(
            "GTFS_DIR, SCHOOLS_PATH, and/or OUTPUT_DIR are still set to their default "
            "placeholder paths. Update the CONFIGURATION section before running."
        )
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_sr = arcpy.SpatialReference(PROJECTED_CRS_WKID)

    logging.info("Loading GTFS from %s", GTFS_DIR)
    tables = _load_gtfs_tables(GTFS_DIR, need_shapes=USE_SHAPE_BUFFER)

    logging.info("Building route catchments (use_shape_buffer=%s)", USE_SHAPE_BUFFER)
    route_buffers = _prepare_route_buffers(
        tables,
        USE_SHAPE_BUFFER,
        BUFFER_DIST_FT,
        target_sr,
        route_filter=ROUTE_FILTER,
    )
    if not route_buffers:
        logging.error("No route catchments produced – nothing to do")
        return 1

    logging.info("Loading school points from %s", SCHOOLS_PATH)
    schools = load_schools_points(SCHOOLS_PATH, target_sr)

    logging.info("Rolling schools up to %d routes", len(route_buffers))
    summary = summarize_schools_by_route(route_buffers, schools)
    summary = _attach_route_short_name(summary, tables["routes"])

    out_path = OUTPUT_DIR / OUTPUT_CSV_NAME
    summary.to_csv(out_path, index=False)
    logging.info(
        "Wrote %s (%d routes, %d schools, %d total enrollment served)",
        out_path,
        len(summary),
        int(summary["schools_served"].sum()),
        int(summary["enrollment_served"].sum()),
    )
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
