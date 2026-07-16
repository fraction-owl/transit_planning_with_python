"""Joins ridership data to bus stop features, optionally with a spatial join to polygon areas.

Designed for ArcGIS Pro workflows, this script merges stop-level ridership data from
an Excel file with stop locations (from a shapefile or GTFS stops.txt), and optionally
joins to a polygon layer (e.g., Census Blocks) for geographic aggregation.

Outputs:
    Shapefiles land in OUTPUT_FOLDER/shapefiles/; CSVs and the run log stay at the
    OUTPUT_FOLDER root.

    - bus_stops_generated.shp: stops feature class built from GTFS stops.txt
      (when BUS_STOPS_INPUT is a GTFS feed rather than a shapefile).
    - BusStops_JoinedPolygon.shp: stops spatially joined to the polygon layer
      (when POLYGON_LAYER is set).
    - bus_stops_with_polygon.csv: stop attributes (plus polygon fields) exported
      from the joined feature class.
    - BusStops_Matched_JoinedPolygon.shp: matched stops carrying ridership fields
      (single-run mode), or one BusStops_<route>.shp per route when SPLIT_BY_ROUTE.
    - agg_ridership_per_stop.csv: network-wide ridership per stop (single-run mode).
    - agg_ridership_by_polygon.csv and polygon_with_ridership.shp: ridership
      aggregated to the polygon layer (when POLYGON_LAYER is set).
    - plots/route_<route>_<measure>.png: per-route boardings/alightings maps drawn
      from GTFS shapes (when DRAW_PLOTS is True).
    - stops_ridership_joiner_arcpy_runlog.txt: run-log sidecar capturing the
      verbatim CONFIGURATION block.

Typical usage:
    Configure paths and options at the top of the script, then run inside ArcGIS Pro
    or as a standalone Python script with access to the ArcPy environment.
"""

import csv
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import arcpy
import pandas as pd

# Path to this script's own source. Undefined in a Jupyter kernel (no __file__),
# so fall back to None; the run log handles a missing source gracefully.
try:
    SELF_PATH: Optional[Path] = Path(__file__).resolve()
except NameError:
    SELF_PATH = None

# Sentinel markers used by extract_config_block / write_run_log to identify
# the configuration block within this file's source. Each string must appear
# exactly once in this file as a stand-alone comment line (other than these
# constant definitions themselves). Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# INPUTS --------------------------------------------------------------------
# Bus stops can be a GTFS feed FOLDER, a GTFS stops.txt, or a .shp.
# For plotting (DRAW_PLOTS), point this at a GTFS feed FOLDER so that
# shapes.txt, trips.txt, routes.txt and stops.txt can all be read from it.
BUS_STOPS_INPUT = r"Your\File\Path\To\GTFS_folder"  # folder, stops.txt, or .shp

# Path to Excel with ridership data.
EXCEL_FILE = r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).XLSX"

# Optional: Filter your Excel data for certain routes. If empty, no filter.
# Example: ROUTE_FILTER_LIST = ["101", "202"]
ROUTE_FILTER_LIST: list[str] = []

# Set to False to create one shapefile for all stops,
# or True to create a separate shapefile per unique route.
SPLIT_BY_ROUTE = False

# OUTPUTS -------------------------------------------------------------------
OUTPUT_FOLDER = r"Your\Folder\Path\To\Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Subfolder names (under OUTPUT_FOLDER) for the two bulky output types.
# CSVs and the run log stay at the OUTPUT_FOLDER root.
SHAPEFILE_SUBDIR: str = "shapefiles"
PLOT_SUBDIR: str = "plots"

SHAPEFILE_DIR: str = os.path.join(OUTPUT_FOLDER, SHAPEFILE_SUBDIR)
os.makedirs(SHAPEFILE_DIR, exist_ok=True)

# Optional: Polygon features to join ridership data to.
# If empty, the spatial-join and aggregation steps will be skipped.
POLYGON_LAYER = r"Your\File\Path\To\census_blocks.shp"

# File paths for intermediate & final outputs (shapefiles → SHAPEFILE_DIR):
GTFS_STOPS_FC = os.path.join(SHAPEFILE_DIR, "bus_stops_generated.shp")
JOINED_FC = os.path.join(SHAPEFILE_DIR, "BusStops_JoinedPolygon.shp")
MATCHED_JOINED_FC = os.path.join(SHAPEFILE_DIR, "BusStops_Matched_JoinedPolygon.shp")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "bus_stops_with_polygon.csv")
POLYGON_WITH_RIDERSHIP_SHP = os.path.join(SHAPEFILE_DIR, "polygon_with_ridership.shp")

# FIELDS & JOIN KEYS -------------------------------------------------------
# 1. Key fields in the bus stops data:
GTFS_KEY_FIELD = "stop_code"  # GTFS "unique" stop identifier
SHAPE_KEY_FIELD = "StopId"  # Shapefile "unique" stop identifier

# 2. Additional useful fields in GTFS or shapefile:
GTFS_SECONDARY_ID_FIELD = "stop_id"  # For reference, e.g. "stop_id" in stops.txt
SHAPE_SECONDARY_ID_FIELD = "StopNum"  # For reference, e.g. "StopNum" in shapefile

# 3. Polygon fields to export (and optional join field):
POLYGON_JOIN_FIELD = "GEOID"  # e.g., Census GEOID
POLYGON_FIELDS_TO_KEEP = ["NAME", "GEOID", "GEOIDFQ"]  # Must include the join field

# PLOTTING ------------------------------------------------------------------
# Master switch for per-route boardings/alightings maps. Requires BUS_STOPS_INPUT
# to resolve to a GTFS feed folder (for shapes.txt / trips.txt / routes.txt).
DRAW_PLOTS: bool = False

# Ridership color bins: (lower_inclusive, upper_exclusive, color, legend_label).
# Half-open [lower, upper); use math.inf for the open-ended top bin. Edit freely.
# NOTE: as written green = low ridership and red = high — the inverse of the usual
# "red = alert" convention. Swap the color strings if that's not intended.
RIDERSHIP_BINS: list[tuple[float, float, str, str]] = [
    (0.0, 5.0, "green", "0–4.9"),
    (5.0, 25.0, "yellow", "5–24.9"),
    (25.0, math.inf, "red", "25+"),
]

# Draw every shape_id for a route (all patterns/directions) vs. only the single
# longest representative shape. Ridership ROUTE_NAME is matched to GTFS routes on
# route_short_name (uppercased + stripped); adjust normalize_route_name if needed.
PLOT_ALL_SHAPES_PER_ROUTE: bool = True

PLOT_DIR: str = os.path.join(OUTPUT_FOLDER, PLOT_SUBDIR)
PLOT_DPI: int = 200
PLOT_MARKER_SIZE: int = 25  # stop marker area (points^2)
PLOT_EXTENT_PAD_FRAC: float = 0.05  # pad centerline bbox by this fraction

# Legend placement. "outside" pins it to the right of the map so it never covers
# data (the map gets a right margin). "best" lets matplotlib pick the in-frame
# corner that overlaps the least data — lighter, but can still land on stops when
# the route fills the frame. Any valid matplotlib loc string also works.
PLOT_LEGEND_LOC: str = "outside"

# Optional roads/basemap shapefile drawn UNDER the route and stops for context.
# Leave empty to skip. Read via arcpy and reprojected to WGS84 on the fly, so a
# projected roads layer (e.g. State Plane) still aligns with the GTFS lon/lat.
# It is read once and bbox-filtered to each route's extent for speed.
# Roads are kept light and thin so they recede; the darker/thicker route reads on
# top. (Darkening roads makes them blend with the route — push them lighter.)
ROADS_SHAPEFILE: str = r""
ROADS_COLOR: str = "0.75"  # matplotlib grayscale: larger = lighter (route line is 0.4)
ROADS_LINEWIDTH: float = 0.5  # thinner than the route line (1.2)

# Route centerline styling. When USE_GTFS_ROUTE_COLOR is True and routes.txt
# supplies a route_color, that hex is used for the line; otherwise the neutral
# ROUTE_DEFAULT_COLOR is used. Default off: a colored line can clash with the
# green/yellow/red ridership encoding of the stops.
USE_GTFS_ROUTE_COLOR: bool = False
ROUTE_DEFAULT_COLOR: str = "0.4"
ROUTE_LINEWIDTH: float = 1.2

# ENVIRONMENT & FLAGS ------------------------------------------------------
# A folder or a .txt is treated as GTFS input; anything else (e.g. .shp) is not.
IS_GTFS_INPUT = os.path.isdir(BUS_STOPS_INPUT) or BUS_STOPS_INPUT.lower().endswith(".txt")
arcpy.env.overwriteOutput = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so the analyst is never
# left with outputs that lack a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# In a Jupyter kernel __file__ is undefined, so the run log cannot read this
# script's own source to capture the config block verbatim. Optionally point
# this at the .py on disk to restore verbatim capture. If left empty, the run
# log falls back to a snapshot of the live config values instead.
SOURCE_FILE_OVERRIDE: str = r""

# === END CONFIG ===

# =============================================================================
# FUNCTIONS
# =============================================================================


def resolve_gtfs_dir(bus_stops_input: str) -> Optional[str]:
    """Return the GTFS feed folder implied by *bus_stops_input*, or None.

    A directory is returned as-is; a stops.txt path returns its parent folder;
    a shapefile (or anything else) returns None.
    """
    if os.path.isdir(bus_stops_input):
        return bus_stops_input
    if bus_stops_input.lower().endswith(".txt"):
        return os.path.dirname(bus_stops_input)
    return None


def resolve_stops_table(bus_stops_input: str) -> str:
    """Return the GTFS stops table path used by XYTableToPoint.

    If *bus_stops_input* is a folder, its stops.txt is used; otherwise the
    input is returned unchanged.
    """
    if os.path.isdir(bus_stops_input):
        return os.path.join(bus_stops_input, "stops.txt")
    return bus_stops_input


def create_bus_stops_feature_class() -> Tuple[str, List[str]]:
    """Create or identify the bus stops feature class.

    If input is a GTFS stops.txt file, convert it to a point feature class.
    Otherwise, assume we have a shapefile. Returns:
      - bus_stops_fc: path to the resulting feature class
      - fields_to_export: list of fields to export (including the polygon fields)
    """
    if POLYGON_LAYER.strip():
        extra_fields = POLYGON_FIELDS_TO_KEEP
    else:
        extra_fields = []

    if IS_GTFS_INPUT:
        # Convert GTFS stops.txt to point feature class
        arcpy.management.XYTableToPoint(
            in_table=resolve_stops_table(BUS_STOPS_INPUT),
            out_feature_class=GTFS_STOPS_FC,
            x_field="stop_lon",
            y_field="stop_lat",
            coordinate_system=arcpy.SpatialReference(4326),  # WGS84
        )
        logging.info("GTFS stops feature class created at: %s", GTFS_STOPS_FC)
        bus_stops_fc = GTFS_STOPS_FC

        # Fields to export to CSV: the GTFS key field, secondary field, plus polygon fields.
        fields_to_export = [
            GTFS_KEY_FIELD,
            GTFS_SECONDARY_ID_FIELD,
            "stop_name",
        ] + extra_fields

    else:
        # Using an existing shapefile of bus stops directly
        logging.info("Using existing bus stops shapefile: %s", BUS_STOPS_INPUT)
        bus_stops_fc = BUS_STOPS_INPUT

        # Fields to export to CSV: the shapefile key field, secondary field, plus polygon fields.
        fields_to_export = [
            SHAPE_KEY_FIELD,
            SHAPE_SECONDARY_ID_FIELD,
        ] + extra_fields

    return bus_stops_fc, fields_to_export


def spatial_join_bus_stops_to_polygons(bus_stops_fc: str, fields_to_export: List[str]) -> str:
    """Perform a spatial join of bus stops to polygon features (if provided)."""
    polygon_layer_str = POLYGON_LAYER.strip()
    if polygon_layer_str:
        arcpy.SpatialJoin_analysis(
            target_features=bus_stops_fc,
            join_features=polygon_layer_str,
            out_feature_class=JOINED_FC,
            join_operation="JOIN_ONE_TO_ONE",
            join_type="KEEP_ALL",
            match_option="INTERSECT",
        )
        logging.info(
            "Spatial join completed. Joined feature class created at: %s",
            JOINED_FC,
        )

        # Export joined data to CSV
        with (
            arcpy.da.SearchCursor(JOINED_FC, fields_to_export) as cursor,
            open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile,
        ):
            writer = csv.writer(csvfile)
            writer.writerow(fields_to_export)
            for row in cursor:
                writer.writerow(row)

        logging.info("CSV export completed. CSV file created at: %s", OUTPUT_CSV)
        current_fc = JOINED_FC
    else:
        logging.info("POLYGON_LAYER is empty. Skipping spatial join.")
        # Export the bus stops feature class to CSV so that merge can still work.
        with (
            arcpy.da.SearchCursor(bus_stops_fc, fields_to_export) as cursor,
            open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile,
        ):
            writer = csv.writer(csvfile)
            writer.writerow(fields_to_export)
            for row in cursor:
                writer.writerow(row)
        logging.info("CSV export completed. CSV file created at: %s", OUTPUT_CSV)
        current_fc = bus_stops_fc

    return current_fc


def read_and_filter_ridership_data() -> pd.DataFrame:
    """Read ridership data from EXCEL_FILE and optionally filter by routes.

    Return a DataFrame with aggregated totals.
    """
    df_excel = pd.read_excel(EXCEL_FILE)

    # Optional route filter
    if ROUTE_FILTER_LIST:
        initial_count = len(df_excel)
        df_excel = df_excel[df_excel["ROUTE_NAME"].isin(ROUTE_FILTER_LIST)]
        logging.info("Filtered Excel data to routes in %s.", ROUTE_FILTER_LIST)
        logging.info(
            "Records reduced from %d to %d.",
            initial_count,
            len(df_excel),
        )
    else:
        logging.info("No route filter applied.")

    # Calculate TOTAL
    df_excel["TOTAL"] = df_excel["XBOARDINGS"] + df_excel["XALIGHTINGS"]
    return df_excel


def merge_ridership_and_csv(
    df_excel: pd.DataFrame, fields_to_export: List[str]
) -> Tuple[pd.DataFrame, str]:
    """Merge ridership data (df_excel) with the CSV from the spatial join.

    Raises error if no polygon layer was provided and CSV does not exist.

    Returns:
      - df_joined: merged DataFrame
      - key_field: which field was used as the merge key (GTFS_KEY_FIELD or SHAPE_KEY_FIELD)
    """
    # Read from the CSV we created in the spatial join (or direct bus stops).
    df_csv = pd.read_csv(OUTPUT_CSV)

    # Merge on appropriate key (GTFS vs. shapefile)
    if IS_GTFS_INPUT:
        df_excel["STOP_ID"] = df_excel["STOP_ID"].astype(str)
        df_csv[GTFS_KEY_FIELD] = df_csv[GTFS_KEY_FIELD].astype(str)
        df_joined = df_excel.merge(df_csv, left_on="STOP_ID", right_on=GTFS_KEY_FIELD, how="inner")
        key_field = GTFS_KEY_FIELD
    else:
        df_excel["STOP_ID"] = df_excel["STOP_ID"].astype(str)
        df_csv[SHAPE_KEY_FIELD] = df_csv[SHAPE_KEY_FIELD].astype(str)
        df_joined = df_excel.merge(df_csv, left_on="STOP_ID", right_on=SHAPE_KEY_FIELD, how="inner")
        key_field = SHAPE_KEY_FIELD

    logging.info(
        "Data merged successfully. Number of matched bus stops: %d",
        len(df_joined),
    )
    return df_joined, key_field


def filter_matched_bus_stops(current_fc: str, df_joined: pd.DataFrame, key_field: str) -> str:
    """Filter the joined feature class to include only matched bus stops.

    Returns the path to the filtered shapefile.
    """
    matched_keys = df_joined[key_field].dropna().unique().tolist()
    if not matched_keys:
        logging.error("No matched bus stops found in Excel data. Exiting script.")
        exit()

    arcpy.MakeFeatureLayer_management(current_fc, "joined_lyr")
    fields = arcpy.ListFields(current_fc, key_field)
    if not fields:
        logging.error("Field '%s' not found in '%s'. Exiting.", key_field, current_fc)
        exit()

    field_type = fields[0].type
    field_delimited = arcpy.AddFieldDelimiters(current_fc, key_field)

    # Prepare values for WHERE clause based on field type
    if field_type in ["String", "Guid", "Date"]:
        formatted_keys = []
        for k in matched_keys:
            escaped = k.replace("'", "''")
            formatted_keys.append(f"'{escaped}'")
    elif field_type in ["Integer", "SmallInteger", "Double", "Single", "OID"]:
        formatted_keys = [str(k) for k in matched_keys]
    else:
        logging.error(
            "Unsupported field type '%s' for field '%s'. Exiting.",
            field_type,
            key_field,
        )
        exit()

    # Due to potential large number of keys, split into chunks
    chunk_size = 999
    where_clauses = []
    for i in range(0, len(formatted_keys), chunk_size):
        chunk = formatted_keys[i : i + chunk_size]
        clause = f"{field_delimited} IN ({', '.join(chunk)})"
        where_clauses.append(clause)

    full_where_clause = " OR ".join(where_clauses)
    logging.debug(
        "Constructed WHERE clause (first 200 chars): %s",
        full_where_clause[:200],
    )

    try:
        arcpy.SelectLayerByAttribute_management("joined_lyr", "NEW_SELECTION", full_where_clause)
    except arcpy.ExecuteError:
        logging.error("Failed SelectLayerByAttribute. Check WHERE clause syntax.")
        logging.error("WHERE clause attempted: %s", full_where_clause)
        raise

    selected_count = int(arcpy.GetCount_management("joined_lyr").getOutput(0))
    if selected_count == 0:
        logging.error("No features matched the WHERE clause. Exiting script.")
        exit()
    else:
        logging.info("Number of features selected: %d", selected_count)

    arcpy.CopyFeatures_management("joined_lyr", MATCHED_JOINED_FC)
    logging.info("Filtered joined feature class created at: %s", MATCHED_JOINED_FC)

    return MATCHED_JOINED_FC


def _clear_readonly(fc: str) -> None:
    """Best-effort: clear the read-only attribute on a shapefile's sidecar files.

    Some network shares create shapefile components read-only, which makes
    AddField fail with ERROR 000499 ("table is not editable").
    """
    import stat

    base = os.path.splitext(fc)[0]
    for ext in (".dbf", ".shp", ".shx", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"):
        p = base + ext
        if os.path.isfile(p):
            try:
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass


def _retry_arcpy(
    func: Callable[..., object],
    *args: object,
    what: str,
    attempts: int = 3,
    delay: float = 1.5,
) -> object:
    """Run a flaky arcpy write op with bounded retries.

    Writes to a UNC share intermittently fail with ERROR 999999 (generic) or
    000499 (schema lock), often from a transient network hiccup or a lingering
    lock. Retry a few times, then fail loud — a live ArcGIS Pro map lock can't
    be retried away.

    Args:
        func: The arcpy callable (e.g. arcpy.CopyFeatures_management).
        *args: Positional args passed to ``func``.
        what: Short description for log messages (e.g. "CopyFeatures 671").
        attempts: Max attempts before re-raising.
        delay: Seconds to wait between attempts.
    """
    for attempt in range(1, attempts + 1):
        try:
            return func(*args)
        except arcpy.ExecuteError:
            if attempt == attempts:
                logging.error(
                    "%s failed after %d attempts (ERROR 999999/000499 = generic write "
                    "failure or lock). Close any layers from the output folder in an open "
                    "ArcGIS Pro map, confirm the target is writable on the share, then re-run.",
                    what,
                    attempts,
                )
                raise
            logging.warning("%s attempt %d failed; retrying…", what, attempt)
            time.sleep(delay)


def update_bus_stops_ridership(current_fc: str, df_joined: pd.DataFrame, key_field: str) -> None:
    """Add ridership fields to the bus stops shapefile and update them with data from df_joined."""
    ridership_fields = [
        ("XBOARD", "DOUBLE"),
        ("XALIGHT", "DOUBLE"),
        ("XTOTAL", "DOUBLE"),
    ]

    # ERROR 000499 ("table is not editable") on AddField over a UNC share is
    # usually a read-only sidecar file or a transient lock. Clear read-only,
    # then retry a few times before failing loud (a live Pro map lock can't be
    # retried away — the message says what to do).
    _clear_readonly(current_fc)
    existing_fields = [f.name for f in arcpy.ListFields(current_fc)]
    for f_name, f_type in ridership_fields:
        if f_name in existing_fields:
            continue
        for attempt in range(1, 4):
            try:
                arcpy.management.AddField(current_fc, f_name, f_type)
                break
            except arcpy.ExecuteError:
                if attempt == 3:
                    logging.error(
                        "AddField failed for '%s' on '%s' (ERROR 000499 = schema lock / "
                        "not editable). Remove this shapefile from any open ArcGIS Pro map "
                        "and confirm it is not read-only on the share, then re-run.",
                        f_name,
                        current_fc,
                    )
                    raise
                logging.warning("AddField '%s' attempt %d failed; retrying…", f_name, attempt)
                time.sleep(1.5)

    logging.info("Ridership fields added (if not existing).")

    # Build dictionary from the joined DataFrame
    stop_ridership_dict = {}
    for _, row in df_joined.iterrows():
        code = row[key_field] if not pd.isna(row[key_field]) else None
        if code is not None:
            stop_ridership_dict[str(code)] = {
                "XBOARD": row["XBOARDINGS"],
                "XALIGHT": row["XALIGHTINGS"],
                "XTOTAL": row["TOTAL"],
            }

    with arcpy.da.UpdateCursor(current_fc, [key_field, "XBOARD", "XALIGHT", "XTOTAL"]) as cursor:
        for r in cursor:
            code_val = str(r[0])
            if code_val in stop_ridership_dict:
                r[1] = stop_ridership_dict[code_val]["XBOARD"]
                r[2] = stop_ridership_dict[code_val]["XALIGHT"]
                r[3] = stop_ridership_dict[code_val]["XTOTAL"]
            else:
                # Should not occur if we've filtered for matched features
                r[1], r[2], r[3] = 0, 0, 0
            cursor.updateRow(r)

    logging.info("Bus stops shapefile updated with ridership data at: %s", current_fc)


def aggregate_ridership(df_joined: pd.DataFrame) -> None:
    """Aggregate ridership by the polygon join field and update the polygon layer shapefile.

    Also exports the aggregated data to CSV for verification.
    """
    if not POLYGON_LAYER.strip():
        logging.info("POLYGON_LAYER is empty, so aggregation steps have been skipped.")
        return

    # Group by the designated polygon join field, e.g. "GEOID"
    df_agg = df_joined.groupby(POLYGON_JOIN_FIELD, as_index=False).agg(
        {"XBOARDINGS": "sum", "XALIGHTINGS": "sum", "TOTAL": "sum"}
    )
    logging.info("Ridership data aggregated by %s.", POLYGON_JOIN_FIELD)

    # ─── Export aggregated ridership spreadsheet ───
    agg_polygon_csv = os.path.join(OUTPUT_FOLDER, "agg_ridership_by_polygon.csv")
    df_agg.to_csv(agg_polygon_csv, index=False)
    logging.info("Aggregated ridership by polygon exported to: %s", agg_polygon_csv)

    # Copy the source polygons so we can add fields without touching the original
    arcpy.management.CopyFeatures(POLYGON_LAYER, POLYGON_WITH_RIDERSHIP_SHP)

    agg_fields = [
        ("XBOARD_SUM", "DOUBLE"),
        ("XALITE_SUM", "DOUBLE"),
        ("TOTAL_SUM", "DOUBLE"),
    ]

    existing_fields_blocks = [f.name for f in arcpy.ListFields(POLYGON_WITH_RIDERSHIP_SHP)]
    for f_name, f_type in agg_fields:
        if f_name not in existing_fields_blocks:
            arcpy.management.AddField(POLYGON_WITH_RIDERSHIP_SHP, f_name, f_type)

    logging.info(
        "Aggregation fields added to polygon shapefile (if not existing).",
    )

    # Build lookup dictionary for fast updates
    agg_dict = {
        row[POLYGON_JOIN_FIELD]: {
            "XBOARD_SUM": row["XBOARDINGS"],
            "XALITE_SUM": row["XALIGHTINGS"],
            "TOTAL_SUM": row["TOTAL"],
        }
        for _, row in df_agg.iterrows()
    }

    with arcpy.da.UpdateCursor(
        POLYGON_WITH_RIDERSHIP_SHP,
        [POLYGON_JOIN_FIELD, "XBOARD_SUM", "XALITE_SUM", "TOTAL_SUM"],
    ) as cursor:
        for rec in cursor:
            geoid = rec[0]
            if geoid in agg_dict:
                rec[1] = agg_dict[geoid]["XBOARD_SUM"]
                rec[2] = agg_dict[geoid]["XALITE_SUM"]
                rec[3] = agg_dict[geoid]["TOTAL_SUM"]
            else:
                rec[1], rec[2], rec[3] = 0, 0, 0
            cursor.updateRow(rec)

    logging.info(
        "Polygon shapefile updated with aggregated ridership data at: %s",
        POLYGON_WITH_RIDERSHIP_SHP,
    )


def process_stops_for_single_run() -> None:
    """(Helper) Original single-run flow (no splitting by route).

    Creates one shapefile for the entire network of bus stops,
    and now also exports an intermediate aggregated ridership CSV.
    """
    # Step 1: Create or identify the bus-stops feature class
    bus_stops_fc, fields_to_export = create_bus_stops_feature_class()

    # Step 2: Spatial Join (optional) → also exports CSV of bus stops (+ polygons)
    current_fc = spatial_join_bus_stops_to_polygons(bus_stops_fc, fields_to_export)

    # Step 3: Read ridership data from Excel & optionally filter by routes
    df_excel = read_and_filter_ridership_data()

    # ─── AGGREGATE PER STOP (network-wide) ───
    # Collapse any multi-route rows down to one row per STOP_ID
    df_excel = df_excel.groupby("STOP_ID", as_index=False).agg(
        {"XBOARDINGS": "sum", "XALIGHTINGS": "sum", "TOTAL": "sum"}
    )

    # Export the intermediate aggregated ridership spreadsheet
    agg_per_stop_csv = os.path.join(OUTPUT_FOLDER, "agg_ridership_per_stop.csv")
    df_excel.to_csv(agg_per_stop_csv, index=False)
    logging.info("Aggregated ridership per stop exported to: %s", agg_per_stop_csv)

    # Step 4: Merge ridership data with CSV from spatial join
    df_joined, key_field = merge_ridership_and_csv(df_excel, fields_to_export)

    # Step 4a: Filter to matched bus stops
    filtered_fc = filter_matched_bus_stops(current_fc, df_joined, key_field)

    # Step 5: Update the bus-stops shapefile with ridership fields
    update_bus_stops_ridership(filtered_fc, df_joined, key_field)

    # Steps 6 & 7: Aggregate ridership (optional, by polygon)
    aggregate_ridership(df_joined)

    logging.info("Single-run process complete.")


# =============================================================================
# PLOTTING
# =============================================================================


def normalize_route_name(name: object) -> str:
    """Uppercase/strip a route name so ridership rows match GTFS route_short_name."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    return str(name).strip().upper()


def _read_gtfs_table(gtfs_dir: str, filename: str, required_cols: List[str]) -> pd.DataFrame:
    """Read a GTFS table as strings; fail loud if the file or a column is missing."""
    path = os.path.join(gtfs_dir, filename)
    if not os.path.isfile(path):
        logging.error("Required GTFS file not found: %s", path)
        sys.exit(1)
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error("GTFS file '%s' is missing columns: %s", path, missing)
        sys.exit(1)
    return df


def _polyline_length(pl: pd.DataFrame) -> float:
    """Cumulative length of a polyline in coordinate units (lon/lat degrees)."""
    d = (pl["lon"].diff() ** 2 + pl["lat"].diff() ** 2) ** 0.5
    return float(d.sum())


def build_route_shape_lookup(gtfs_dir: str) -> tuple:
    """Map normalized route_short_name -> (list of centerline polylines, color).

    Reads routes.txt, trips.txt and shapes.txt. Each polyline is a DataFrame with
    ordered 'lon'/'lat' columns. Honors PLOT_ALL_SHAPES_PER_ROUTE (all patterns vs.
    single longest representative). Routes with no usable shape are omitted.

    Returns:
        (name_to_polylines, name_to_color) where name_to_color maps each route
        name to a '#RRGGBB' string from route_color, or None when absent/invalid.
    """
    routes = _read_gtfs_table(gtfs_dir, "routes.txt", ["route_id", "route_short_name"])
    trips = _read_gtfs_table(gtfs_dir, "trips.txt", ["route_id", "shape_id"])
    shapes = _read_gtfs_table(
        gtfs_dir,
        "shapes.txt",
        ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    )

    has_color = "route_color" in routes.columns

    shapes = shapes.dropna(subset=["shape_id", "shape_pt_lat", "shape_pt_lon"]).copy()
    shapes["shape_pt_lat"] = shapes["shape_pt_lat"].astype(float)
    shapes["shape_pt_lon"] = shapes["shape_pt_lon"].astype(float)
    shapes["shape_pt_sequence"] = shapes["shape_pt_sequence"].astype(int)

    # shape_id -> ordered polyline
    shape_polylines: dict = {}
    for shape_id, grp in shapes.sort_values("shape_pt_sequence").groupby("shape_id"):
        shape_polylines[shape_id] = (
            grp[["shape_pt_lon", "shape_pt_lat"]]
            .rename(columns={"shape_pt_lon": "lon", "shape_pt_lat": "lat"})
            .reset_index(drop=True)
        )

    # route_id -> ordered unique shape_ids
    trips_valid = trips.dropna(subset=["shape_id"])
    rid_to_sids = (
        trips_valid.groupby("route_id")["shape_id"]
        .apply(lambda s: list(dict.fromkeys(s)))
        .to_dict()
    )

    name_to_polylines: dict = {}
    name_to_color: dict = {}
    for _, r in routes.iterrows():
        norm = normalize_route_name(r["route_short_name"])
        if not norm:
            continue
        pls = [
            shape_polylines[s] for s in rid_to_sids.get(r["route_id"], []) if s in shape_polylines
        ]
        if pls:
            name_to_polylines.setdefault(norm, []).extend(pls)
        # First valid route_color wins when several route_ids share a name.
        if has_color and name_to_color.get(norm) is None:
            hexcol = _normalize_hex_color(r["route_color"])
            if hexcol is not None:
                name_to_color[norm] = hexcol

    if not PLOT_ALL_SHAPES_PER_ROUTE:
        for norm, pls in name_to_polylines.items():
            name_to_polylines[norm] = [max(pls, key=_polyline_length)]

    return name_to_polylines, name_to_color


def load_stop_coords(gtfs_dir: str) -> pd.DataFrame:
    """Return GTFS stop coordinates keyed by stop_code (the stable public key)."""
    stops = _read_gtfs_table(gtfs_dir, "stops.txt", ["stop_code", "stop_lat", "stop_lon"])
    stops = stops.dropna(subset=["stop_code", "stop_lat", "stop_lon"]).copy()
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    stops["stop_code"] = stops["stop_code"].astype(str)
    return stops[["stop_code", "stop_lat", "stop_lon"]]


def _normalize_hex_color(value: object) -> Optional[str]:
    """Return a '#RRGGBB' string from a GTFS route_color value, or None if invalid.

    GTFS stores route_color as 6 hex digits without a leading '#'. Empty, missing,
    or malformed values return None so the caller can fall back to a default.
    """
    if value is None:
        return None
    s = str(value).strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        int(s, 16)
    except ValueError:
        return None
    return "#" + s


def load_road_polylines(roads_fc: str) -> list:
    """Read an optional roads shapefile into bbox-indexed lon/lat polylines.

    Uses an arcpy cursor with spatial_reference=WGS84 so a projected roads layer
    is reprojected on the fly and aligns with the GTFS lon/lat geometry. Returns a
    list of (xmin, xmax, ymin, ymax, DataFrame[lon, lat]) tuples; the bbox lets
    callers cheaply filter to a route's extent before drawing. Fails loud if the
    layer is missing.
    """
    if not os.path.isfile(roads_fc) and not arcpy.Exists(roads_fc):
        logging.error("ROADS_SHAPEFILE set but not found: %s", roads_fc)
        sys.exit(1)

    sr = arcpy.SpatialReference(4326)
    indexed: list = []
    with arcpy.da.SearchCursor(roads_fc, ["SHAPE@"], spatial_reference=sr) as cursor:
        for (geom,) in cursor:
            if geom is None:
                continue
            for part in geom:
                pts = [(pnt.X, pnt.Y) for pnt in part if pnt is not None]
                if len(pts) < 2:
                    continue
                df = pd.DataFrame(pts, columns=["lon", "lat"])
                indexed.append(
                    (
                        float(df["lon"].min()),
                        float(df["lon"].max()),
                        float(df["lat"].min()),
                        float(df["lat"].max()),
                        df,
                    )
                )
    logging.info("Loaded %d road segments from %s.", len(indexed), roads_fc)
    return indexed


def _roads_in_extent(
    roads_indexed: list, xmin: float, xmax: float, ymin: float, ymax: float
) -> list:
    """Return road polylines whose bbox overlaps the given extent."""
    return [
        df
        for (x0, x1, y0, y1, df) in roads_indexed
        if not (x1 < xmin or x0 > xmax or y1 < ymin or y0 > ymax)
    ]


def color_for_value(value: float) -> str:
    """Return the RIDERSHIP_BINS color for a ridership *value* (half-open bins)."""
    for lower, upper, color, _label in RIDERSHIP_BINS:
        if lower <= value < upper:
            return color
    return RIDERSHIP_BINS[-1][2]


def plot_route_ridership(
    route_label: str,
    stops_df: pd.DataFrame,
    polylines: List[pd.DataFrame],
    value_name: str,
    out_path: str,
    roads_indexed: Optional[list] = None,
    route_color: Optional[str] = None,
) -> None:
    """Render one route map colored by ridership bin and save it to *out_path*.

    Draws optional roads, then the centerline(s), then stops on top colored per
    RIDERSHIP_BINS, with a bin legend and a north arrow. The view is zoomed to the
    centerline extent.

    Args:
        route_label: Route name shown in the title.
        stops_df: DataFrame with 'lon', 'lat', 'value' columns.
        polylines: Ordered centerline polylines (each with 'lon'/'lat' columns).
        value_name: "Boardings" or "Alightings" (legend title / filename stem).
        out_path: Destination PNG path.
        roads_indexed: Optional bbox-indexed roads from load_road_polylines;
            filtered to the map extent and drawn underneath.
        route_color: Optional '#RRGGBB' centerline color; falls back to
            ROUTE_DEFAULT_COLOR when None.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)  # file output; leaves an active GUI backend alone
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute the zoom extent from the centerline first, so roads can be filtered
    # to it before drawing.
    if polylines:
        allpts = pd.concat(polylines, ignore_index=True)
        xmin, xmax = float(allpts["lon"].min()), float(allpts["lon"].max())
        ymin, ymax = float(allpts["lat"].min()), float(allpts["lat"].max())
        padx = (xmax - xmin) * PLOT_EXTENT_PAD_FRAC or 0.001
        pady = (ymax - ymin) * PLOT_EXTENT_PAD_FRAC or 0.001
        ext_xmin, ext_xmax = xmin - padx, xmax + padx
        ext_ymin, ext_ymax = ymin - pady, ymax + pady
        center_lat = (ymin + ymax) / 2.0
    else:
        ext_xmin = ext_xmax = ext_ymin = ext_ymax = None
        center_lat = float(stops_df["lat"].mean())

    # Roads underneath everything (zorder 0), filtered to the map extent.
    if roads_indexed and ext_xmin is not None:
        for road in _roads_in_extent(roads_indexed, ext_xmin, ext_xmax, ext_ymin, ext_ymax):
            ax.plot(
                road["lon"],
                road["lat"],
                color=ROADS_COLOR,
                linewidth=ROADS_LINEWIDTH,
                zorder=0,
            )

    # Centerline(s) above roads
    line_color = route_color if route_color else ROUTE_DEFAULT_COLOR
    for pl in polylines:
        ax.plot(pl["lon"], pl["lat"], color=line_color, linewidth=ROUTE_LINEWIDTH, zorder=1)

    if ext_xmin is not None:
        ax.set_xlim(ext_xmin, ext_xmax)
        ax.set_ylim(ext_ymin, ext_ymax)

    # Draw low-ridership stops first so higher-ridership stops (of more interest)
    # render on top rather than being hidden underneath. NaN values, if any, sink
    # to the bottom.
    stops_df = stops_df.sort_values("value", na_position="first", kind="stable")

    # Stops on top, colored by bin
    ax.scatter(
        stops_df["lon"],
        stops_df["lat"],
        c=stops_df["value"].map(color_for_value),
        s=PLOT_MARKER_SIZE,
        edgecolor="black",
        linewidth=0.3,
        zorder=3,
    )

    # Keep lon/lat visually square at this latitude so "up" reads as north
    ax.set_aspect(1.0 / math.cos(math.radians(center_lat)))

    # North indicator drawn as a single point-sized text glyph (up-arrow over N).
    # A text glyph scales with fontsize (points), not with the axes box, so it
    # keeps its shape on wide, short east–west route extents where the previous
    # axes-fraction arrow got vertically squashed.
    ax.annotate(
        "↑\nN",
        xy=(0.06, 0.88),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        linespacing=0.9,
        zorder=5,
    )

    # Legend from the bins
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        for _lo, _hi, color, label in RIDERSHIP_BINS
    ]
    if PLOT_LEGEND_LOC == "outside":
        ax.legend(
            handles=handles,
            title=value_name,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            framealpha=0.9,
        )
    else:
        ax.legend(handles=handles, title=value_name, loc=PLOT_LEGEND_LOC, framealpha=0.9)

    ax.set_title(f"Route {route_label} — {value_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    # bbox_inches="tight" ensures an outside legend is included, not clipped.
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out_path)


def generate_route_plots() -> None:
    """Render per-route boardings and alightings maps from GTFS shapes.

    Fails loud (sys.exit(1)) if BUS_STOPS_INPUT is not a GTFS folder/stops.txt or
    if a required GTFS file is missing. Routes with no matching shape or no matching
    stop coordinates are skipped and reported, not silently dropped.
    """
    gtfs_dir = resolve_gtfs_dir(BUS_STOPS_INPUT)
    if gtfs_dir is None:
        logging.error(
            "DRAW_PLOTS is True but BUS_STOPS_INPUT is not a GTFS folder or stops.txt. "
            "Point BUS_STOPS_INPUT at a GTFS feed folder to enable plotting."
        )
        sys.exit(1)

    os.makedirs(PLOT_DIR, exist_ok=True)

    name_to_polylines, name_to_color = build_route_shape_lookup(gtfs_dir)
    stop_coords = load_stop_coords(gtfs_dir)

    # Optional roads basemap, read once and reused (bbox-filtered per route).
    roads_indexed = load_road_polylines(ROADS_SHAPEFILE) if ROADS_SHAPEFILE.strip() else None

    # Route-level ridership (before the network-wide per-stop collapse used elsewhere)
    ridership = read_and_filter_ridership_data()
    ridership["STOP_ID"] = ridership["STOP_ID"].astype(str)
    ridership = ridership.groupby(["ROUTE_NAME", "STOP_ID"], as_index=False).agg(
        {"XBOARDINGS": "sum", "XALIGHTINGS": "sum"}
    )

    plotted = 0
    skipped: list[str] = []
    for route_name, grp in ridership.groupby("ROUTE_NAME"):
        norm = normalize_route_name(route_name)
        polylines = name_to_polylines.get(norm)
        if not polylines:
            skipped.append(str(route_name))
            continue

        route_color = name_to_color.get(norm) if USE_GTFS_ROUTE_COLOR else None

        merged = grp.merge(stop_coords, left_on="STOP_ID", right_on="stop_code", how="inner")
        if merged.empty:
            logging.warning("Route %s: no GTFS stop coordinates matched; skipping.", route_name)
            skipped.append(str(route_name))
            continue

        safe = str(route_name).strip().replace(os.sep, "_").replace(" ", "_")
        for value_col, value_name in (("XBOARDINGS", "Boardings"), ("XALIGHTINGS", "Alightings")):
            stops_df = pd.DataFrame(
                {
                    "lon": merged["stop_lon"].to_numpy(),
                    "lat": merged["stop_lat"].to_numpy(),
                    "value": merged[value_col].astype(float).to_numpy(),
                }
            )
            out_path = os.path.join(PLOT_DIR, f"route_{safe}_{value_name.lower()}.png")
            plot_route_ridership(
                str(route_name),
                stops_df,
                polylines,
                value_name,
                out_path,
                roads_indexed=roads_indexed,
                route_color=route_color,
            )
        plotted += 1
        logging.info("Plotted route %s (boardings + alightings).", route_name)

    logging.info("Route plots complete: %d plotted, %d skipped.", plotted, len(skipped))
    if skipped:
        logging.warning(
            "No GTFS shape/stop match for %d route(s): %s",
            len(skipped),
            ", ".join(sorted(set(skipped))),
        )


# =============================================================================
# RUN LOG
# =============================================================================


# Canonical version lives in utils/run_log.py — keep this copy in sync.
def extract_config_block(source_file: Path) -> str:
    """Return the text between the CONFIG markers in *source_file*.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def _resolve_source_path() -> Optional[Path]:
    """Return a readable path to this script's source, or None.

    Prefers SOURCE_FILE_OVERRIDE (for notebook runs), then SELF_PATH (defined
    when running as a .py file). Returns None when neither is available.
    """
    override = SOURCE_FILE_OVERRIDE.strip()
    if override:
        p = Path(override)
        if p.is_file():
            return p
        logging.warning("SOURCE_FILE_OVERRIDE set but not found: %s", p)
    return SELF_PATH


def _live_config_snapshot() -> str:
    """Best-effort config record built from live module globals.

    Used when the source file is unavailable (e.g. running in a Jupyter kernel),
    so the run log still carries a record of the configuration actually used.
    Captures UPPER_SNAKE_CASE globals holding simple scalar/sequence values.
    """
    g = globals()
    lines: list[str] = []
    for name in sorted(g):
        if name.startswith("_") or name != name.upper():
            continue
        val = g[name]
        if isinstance(val, (str, int, float, bool, list, tuple, type(None))):
            lines.append(f"{name} = {val!r}")
    return "\n".join(lines)


def write_run_log(output_folder: str) -> bool:
    """Write a run log of the configuration into *output_folder*.

    When the script's source is readable, the config block is captured verbatim.
    Otherwise (e.g. a Jupyter kernel with no __file__) it falls back to a live
    snapshot of the config globals so a run still produces a configuration record.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = Path(output_folder) / "stops_ridership_joiner_arcpy_runlog.txt"

    source_path = _resolve_source_path()
    if source_path is not None:
        try:
            config_text = extract_config_block(source_path)
            config_heading = "CONFIGURATION (verbatim from source)"
            source_label = str(source_path)
        except (OSError, ValueError) as exc:
            logging.warning(
                "Could not extract config block from '%s' (%s); falling back to a live snapshot.",
                source_path,
                exc,
            )
            config_text = _live_config_snapshot()
            config_heading = "CONFIGURATION (live snapshot — source unreadable)"
            source_label = f"{source_path} (unreadable)"
    else:
        config_text = _live_config_snapshot()
        config_heading = "CONFIGURATION (live snapshot — source file unavailable)"
        source_label = "unavailable (running without __file__, e.g. Jupyter)"

    lines: list[str] = [
        "=" * 72,
        "STOPS RIDERSHIP JOINER (ARCPY) RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output folder:    {output_folder}",
        f"Source script:    {source_label}",
        "",
        "-" * 72,
        config_heading,
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point for the script.

    Either processes all routes at once (creating a single shapefile) or splits
    by route (creating multiple shapefiles), depending on SPLIT_BY_ROUTE.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _DEFAULT_BUS_STOPS = r"Your\File\Path\To\GTFS_folder"
    _DEFAULT_EXCEL = r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).XLSX"
    if BUS_STOPS_INPUT == _DEFAULT_BUS_STOPS or EXCEL_FILE == _DEFAULT_EXCEL:
        logging.warning(
            "File paths are still set to their defaults. Update BUS_STOPS_INPUT and "
            "EXCEL_FILE in the CONFIGURATION section before running."
        )
        return

    # >>>>> NEW BRANCHING LOGIC <<<<<
    if not SPLIT_BY_ROUTE:
        # ---- Original single-run approach ----
        logging.info("SPLIT_BY_ROUTE = False. Running single shapefile process.")
        process_stops_for_single_run()

    else:
        # ---- Per-route approach (from the second script) ----
        logging.info("SPLIT_BY_ROUTE = True. Creating one shapefile per route.")

        # Step 1: Create or identify the bus stops feature class
        bus_stops_fc, fields_to_export = create_bus_stops_feature_class()

        # Step 2: Spatial Join (Optional) -> also exports CSV
        current_fc = spatial_join_bus_stops_to_polygons(bus_stops_fc, fields_to_export)

        # Step 3: Read ridership data from Excel & optionally filter by routes
        df_excel = read_and_filter_ridership_data()

        # Identify unique routes
        unique_routes = df_excel["ROUTE_NAME"].unique()
        logging.info("Found the following unique routes: %s", unique_routes)

        # For each route, merge, filter, and export a shapefile
        for route in unique_routes:
            logging.info("=== Processing route: %s ===", route)
            df_route = df_excel[df_excel["ROUTE_NAME"] == route].copy()
            if df_route.empty:
                logging.warning("No ridership data for route %s. Skipping.", route)
                continue

            # Merge data
            df_joined, key_field = merge_ridership_and_csv(df_route, fields_to_export)
            if df_joined.empty:
                logging.warning(
                    "No matched bus stops found for route %s. Skipping.",
                    route,
                )
                continue

            # Create a route-specific feature class path
            route_output_fc = os.path.join(SHAPEFILE_DIR, f"BusStops_{route}.shp")

            matched_keys = df_joined[key_field].dropna().unique().tolist()
            if not matched_keys:
                logging.warning("No matched bus stops found. Skipping route %s.", route)
                continue

            arcpy.MakeFeatureLayer_management(current_fc, "joined_lyr_route")
            field_delimited = arcpy.AddFieldDelimiters(current_fc, key_field)

            # We have to chunk keys to avoid 'IN' clause limit
            chunk_size = 999
            where_clauses = []
            for i in range(0, len(matched_keys), chunk_size):
                chunk = matched_keys[i : i + chunk_size]
                # Quote or not quote based on field type if needed.
                # Here we'll assume string type for simplicity:
                chunk_str = ", ".join(f"'{k}'" for k in chunk)
                where_clauses.append(f"{field_delimited} IN ({chunk_str})")

            route_where_clause = " OR ".join(where_clauses)
            arcpy.SelectLayerByAttribute_management(
                "joined_lyr_route", "NEW_SELECTION", route_where_clause
            )

            selected_count = int(arcpy.GetCount_management("joined_lyr_route").getOutput(0))
            if selected_count == 0:
                logging.warning(
                    "No bus stops found in FC for route %s. Skipping.",
                    route,
                )
                continue

            _retry_arcpy(
                arcpy.CopyFeatures_management,
                "joined_lyr_route",
                route_output_fc,
                what=f"CopyFeatures {route}",
            )
            logging.info("Route-specific shapefile created at: %s", route_output_fc)

            # Release the in-memory layer so it can't hold a lock on the next iteration.
            arcpy.management.Delete("joined_lyr_route")

            # Now update ridership fields
            update_bus_stops_ridership(route_output_fc, df_joined, key_field)

        # After all routes are processed, optionally aggregate if you want
        # aggregated polygon results across *all* stops (regardless of route).
        # If you only want polygons by route, you'd do a per-route polygon join.
        # For simplicity, we show it once for the entire dataset:
        aggregate_ridership(df_excel)

        logging.info("Per-route process complete.")

    # Per-route ridership maps (independent of SPLIT_BY_ROUTE).
    if DRAW_PLOTS:
        logging.info("DRAW_PLOTS = True. Generating per-route ridership maps.")
        generate_route_plots()

    if not write_run_log(OUTPUT_FOLDER) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        sys.exit(1)

    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
