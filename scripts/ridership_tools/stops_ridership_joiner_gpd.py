"""Join ridership data to bus stop point features (GeoPandas port).

This script merges stop-level ridership data from an Excel file with stop locations
(from a shapefile/GeoPackage/GeoJSON/etc. or GTFS stops.txt), and optionally performs
a spatial join to polygons (e.g., Census Blocks) for geographic aggregation.

Outputs:
- Stops with ridership attributes (one file, or split by route)
- CSV summaries (per-stop and optional per-polygon aggregation)
- Optional polygon layer with aggregated ridership
- Optional per-route boardings/alightings maps from GTFS shapes (see DRAW_PLOTS)

Typical usage:
    Update the paths in the CONFIGURATION section and run from a shell or a
    Jupyter notebook with the open-source geospatial stack (geopandas) installed.
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
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
# Bus stops can be a GTFS feed FOLDER, a GTFS stops.txt, or a vector file
# (.shp/.gpkg/.geojson/...). For plotting (DRAW_PLOTS), point this at a GTFS
# feed FOLDER so shapes.txt, trips.txt, routes.txt and stops.txt can be read.
BUS_STOPS_INPUT = Path(r"Your\File\Path\To\GTFS_folder")  # folder, stops.txt, or vector file
EXCEL_FILE = Path(r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).XLSX")

ROUTE_FILTER_LIST: list[str] = []
SPLIT_BY_ROUTE = False

OUTPUT_FOLDER = Path(r"Your\Folder\Path\To\Output")

# Subfolder names (under OUTPUT_FOLDER) for the two bulky output types.
# CSVs and the run log stay at the OUTPUT_FOLDER root.
VECTOR_SUBDIR: str = "vector"
PLOT_SUBDIR: str = "plots"

VECTOR_DIR = OUTPUT_FOLDER / VECTOR_SUBDIR
PLOT_DIR = OUTPUT_FOLDER / PLOT_SUBDIR

# Optional polygons (set to None to disable)
POLYGON_LAYER: Optional[Path] = Path(r"Your\File\Path\To\census_blocks.shp")

# OUTPUT FORMAT: "gpkg" strongly recommended; "shp" supported
OUT_FORMAT = "gpkg"  # "gpkg" | "shp"

# FIELDS & JOIN KEYS --------------------------------------------------------
GTFS_KEY_FIELD = "stop_code"
SHAPE_KEY_FIELD = "StopId"

GTFS_SECONDARY_ID_FIELD = "stop_id"
SHAPE_SECONDARY_ID_FIELD = "StopNum"

POLYGON_JOIN_FIELD = "GEOID"
POLYGON_FIELDS_TO_KEEP = ["NAME", "GEOID", "GEOIDFQ"]

GTFS_LON_FIELD = "stop_lon"
GTFS_LAT_FIELD = "stop_lat"

# Excel fields expected
EXCEL_STOP_ID_FIELD = "STOP_ID"
EXCEL_ROUTE_FIELD = "ROUTE_NAME"
EXCEL_BOARD_FIELD = "XBOARDINGS"
EXCEL_ALIGHT_FIELD = "XALIGHTINGS"

# Output ridership fields (short for shapefile compatibility)
OUT_BOARD = "XBOARD"
OUT_ALIGHT = "XALIGHT"
OUT_TOTAL = "XTOTAL"

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

PLOT_DPI: int = 200
PLOT_MARKER_SIZE: int = 25  # stop marker area (points^2)
PLOT_EXTENT_PAD_FRAC: float = 0.05  # pad centerline bbox by this fraction

# Legend placement. "outside" pins it to the right of the map so it never covers
# data (the map gets a right margin). "best" lets matplotlib pick the in-frame
# corner that overlaps the least data — lighter, but can still land on stops when
# the route fills the frame. Any valid matplotlib loc string also works.
PLOT_LEGEND_LOC: str = "outside"

# Optional roads/basemap vector file drawn UNDER the route and stops for context.
# Set to None to skip. Read via geopandas and reprojected to WGS84 on the fly, so a
# projected roads layer (e.g. State Plane) still aligns with the GTFS lon/lat.
# It is read once and bbox-filtered to each route's extent for speed.
# Roads are kept light and thin so they recede; the darker/thicker route reads on
# top. (Darkening roads makes them blend with the route — push them lighter.)
ROADS_SHAPEFILE: Optional[Path] = None
ROADS_COLOR: str = "0.75"  # matplotlib grayscale: larger = lighter (route line is 0.4)
ROADS_LINEWIDTH: float = 0.5  # thinner than the route line (1.2)

# Route centerline styling. When USE_GTFS_ROUTE_COLOR is True and routes.txt
# supplies a route_color, that hex is used for the line; otherwise the neutral
# ROUTE_DEFAULT_COLOR is used. Default off: a colored line can clash with the
# green/yellow/red ridership encoding of the stops.
USE_GTFS_ROUTE_COLOR: bool = False
ROUTE_DEFAULT_COLOR: str = "0.4"
ROUTE_LINEWIDTH: float = 1.2

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
# HELPERS
# =============================================================================


def is_gtfs_txt(path: Path) -> bool:
    """Return True if input should be treated as GTFS stops.txt."""
    return path.suffix.lower() == ".txt"


def is_gtfs_input(path: Path) -> bool:
    """Return True if input is GTFS (a feed folder or a stops.txt)."""
    return path.is_dir() or is_gtfs_txt(path)


def resolve_gtfs_dir(path: Path) -> Optional[Path]:
    """Return the GTFS feed folder implied by *path*, or None.

    A directory is returned as-is; a stops.txt path returns its parent folder;
    a vector file (or anything else) returns None.
    """
    if path.is_dir():
        return path
    if is_gtfs_txt(path):
        return path.parent
    return None


def resolve_stops_table(path: Path) -> Path:
    """Return the GTFS stops table path.

    If *path* is a folder, its stops.txt is used; otherwise the input is
    returned unchanged.
    """
    if path.is_dir():
        return path / "stops.txt"
    return path


def _safe_to_str(series: pd.Series) -> pd.Series:
    """Convert values to string, preserving NaNs."""
    return series.astype("string").astype(object)


def _require_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _to_common_crs(
    points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reproject points/polygons to a common CRS (prefers polygons CRS)."""
    if polygons.crs is None and points.crs is None:
        raise ValueError("Both points and polygons are missing CRS; cannot spatial-join safely.")
    if polygons.crs is None:
        raise ValueError("Polygon layer has no CRS; define it before running.")
    if points.crs is None:
        raise ValueError("Stop layer has no CRS; define it before running.")

    if points.crs != polygons.crs:
        points = points.to_crs(polygons.crs)

    return points, polygons


def output_path(base: str, route: Optional[str] = None) -> Path:
    """Build an output file path (under VECTOR_DIR) for the chosen output format."""
    suffix = ".gpkg" if OUT_FORMAT.lower() == "gpkg" else ".shp"
    name = f"{base}_{route}{suffix}" if route else f"{base}{suffix}"
    return VECTOR_DIR / name


def _drop_case_insensitive_duplicate_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop later columns whose names collide case-insensitively.

    OGR field names are case-insensitive, so GPKG/SHP cannot hold, for example,
    both GTFS ``stop_id`` and Excel ``STOP_ID`` (which duplicate the stop key
    after the inner join). Keeps the first occurrence, drops later collisions,
    and logs each drop. Only affects the vector write; CSV exports keep both.
    """
    seen: dict[str, str] = {}
    drop: list[str] = []
    geom_name = gdf.geometry.name
    for col in gdf.columns:
        if col == geom_name:
            continue
        key = col.lower()
        if key in seen:
            logging.warning(
                "Dropping column '%s' from vector output: it collides "
                "case-insensitively with '%s' (OGR field names are case-insensitive).",
                col,
                seen[key],
            )
            drop.append(col)
        else:
            seen[key] = col
    return gdf.drop(columns=drop) if drop else gdf


def write_vector(gdf: gpd.GeoDataFrame, path: Path, layer: Optional[str] = None) -> None:
    """Write a GeoDataFrame to disk as GPKG or SHP (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = _drop_case_insensitive_duplicate_fields(gdf)
    if path.suffix.lower() == ".gpkg":
        gdf.to_file(path, layer=layer or "data", driver="GPKG")
    elif path.suffix.lower() == ".shp":
        gdf.to_file(path, driver="ESRI Shapefile")
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


# =============================================================================
# CORE STEPS
# =============================================================================


def load_bus_stops() -> tuple[gpd.GeoDataFrame, str]:
    """Load bus stop points as a GeoDataFrame and return (gdf, key_field)."""
    if is_gtfs_input(BUS_STOPS_INPUT):
        stops_table = resolve_stops_table(BUS_STOPS_INPUT)
        df = pd.read_csv(stops_table)
        _require_columns(
            df,
            [GTFS_KEY_FIELD, GTFS_SECONDARY_ID_FIELD, "stop_name", GTFS_LON_FIELD, GTFS_LAT_FIELD],
            context=f"GTFS stops file {stops_table}",
        )

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[GTFS_LON_FIELD], df[GTFS_LAT_FIELD]),
            crs="EPSG:4326",
        )
        logging.info("Loaded GTFS stops.txt with %d records.", len(gdf))
        return gdf, GTFS_KEY_FIELD

    gdf = gpd.read_file(BUS_STOPS_INPUT)
    _require_columns(
        gdf,
        [SHAPE_KEY_FIELD, SHAPE_SECONDARY_ID_FIELD],
        context=f"stop layer {BUS_STOPS_INPUT}",
    )
    logging.info("Loaded stop layer with %d features: %s", len(gdf), BUS_STOPS_INPUT)
    return gdf, SHAPE_KEY_FIELD


def spatial_join_to_polygons(
    stops: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
    """Optionally spatial-join stops to polygons; returns (stops_joined, polygons_or_none)."""
    if not POLYGON_LAYER:
        logging.info("POLYGON_LAYER is None; skipping spatial join.")
        return stops, None

    polygons = gpd.read_file(POLYGON_LAYER)
    _require_columns(polygons, [POLYGON_JOIN_FIELD], context=f"polygon layer {POLYGON_LAYER}")

    keep = list(dict.fromkeys(POLYGON_FIELDS_TO_KEEP + [POLYGON_JOIN_FIELD]))
    keep = [c for c in keep if c in polygons.columns]
    polygons = polygons[keep + ["geometry"]].copy()

    stops, polygons = _to_common_crs(stops, polygons)

    # "within" = point must lie inside polygon. Use "intersects" if you want boundary hits.
    joined = gpd.sjoin(stops, polygons, how="left", predicate="within")
    joined = joined.drop(
        columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore"
    )

    logging.info("Spatial join complete. Stops rows: %d.", len(joined))
    return joined, polygons


def read_and_filter_excel() -> pd.DataFrame:
    """Read ridership data from Excel and optionally filter by routes; adds TOTAL."""
    df = pd.read_excel(EXCEL_FILE)

    _require_columns(
        df,
        [EXCEL_STOP_ID_FIELD, EXCEL_ROUTE_FIELD, EXCEL_BOARD_FIELD, EXCEL_ALIGHT_FIELD],
        context=f"Excel ridership file {EXCEL_FILE}",
    )

    if ROUTE_FILTER_LIST:
        before = len(df)
        df = df[df[EXCEL_ROUTE_FIELD].isin(ROUTE_FILTER_LIST)].copy()
        logging.info("Route filter applied. Records: %d -> %d", before, len(df))
    else:
        logging.info("No route filter applied.")

    df["TOTAL"] = df[EXCEL_BOARD_FIELD] + df[EXCEL_ALIGHT_FIELD]
    df[EXCEL_STOP_ID_FIELD] = _safe_to_str(df[EXCEL_STOP_ID_FIELD])

    return df


def aggregate_excel_per_stop(df_excel: pd.DataFrame) -> pd.DataFrame:
    """Collapse Excel ridership rows to one row per STOP_ID."""
    return df_excel.groupby(EXCEL_STOP_ID_FIELD, as_index=False).agg(
        {
            EXCEL_BOARD_FIELD: "sum",
            EXCEL_ALIGHT_FIELD: "sum",
            "TOTAL": "sum",
        }
    )


def merge_ridership(
    stops: gpd.GeoDataFrame,
    df_excel: pd.DataFrame,
    stops_key_field: str,
) -> gpd.GeoDataFrame:
    """Inner-join ridership to stops on STOP_ID vs the chosen stop key field."""
    if stops_key_field not in stops.columns:
        raise ValueError(f"Stop key field '{stops_key_field}' not found in stops layer.")

    stops_copy = stops.copy()
    stops_copy[stops_key_field] = _safe_to_str(stops_copy[stops_key_field])

    out = stops_copy.merge(
        df_excel,
        left_on=stops_key_field,
        right_on=EXCEL_STOP_ID_FIELD,
        how="inner",
        validate="one_to_one" if df_excel[EXCEL_STOP_ID_FIELD].is_unique else "many_to_one",
    )

    logging.info("Matched stops after join: %d", len(out))
    return gpd.GeoDataFrame(out, geometry="geometry", crs=stops.crs)


def add_output_ridership_fields(stops_joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create standardized output fields (XBOARD/XALIGHT/XTOTAL)."""
    out = stops_joined.copy()
    out[OUT_BOARD] = out[EXCEL_BOARD_FIELD].astype(float)
    out[OUT_ALIGHT] = out[EXCEL_ALIGHT_FIELD].astype(float)
    out[OUT_TOTAL] = out["TOTAL"].astype(float)
    return out


def aggregate_by_polygon(
    matched_stops: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Aggregate stop ridership by POLYGON_JOIN_FIELD and join to polygons."""
    if POLYGON_JOIN_FIELD not in matched_stops.columns:
        raise ValueError(
            f"Matched stops missing polygon join field '{POLYGON_JOIN_FIELD}'. "
            "Confirm the spatial join ran and that field was kept."
        )

    df_agg = matched_stops.groupby(POLYGON_JOIN_FIELD, as_index=False).agg(
        {EXCEL_BOARD_FIELD: "sum", EXCEL_ALIGHT_FIELD: "sum", "TOTAL": "sum"}
    )
    df_agg = df_agg.rename(
        columns={
            EXCEL_BOARD_FIELD: "XBOARD_SUM",
            EXCEL_ALIGHT_FIELD: "XALITE_SUM",
            "TOTAL": "TOTAL_SUM",
        }
    )

    polygons_out = polygons.merge(df_agg, on=POLYGON_JOIN_FIELD, how="left")
    for c in ["XBOARD_SUM", "XALITE_SUM", "TOTAL_SUM"]:
        polygons_out[c] = polygons_out[c].fillna(0.0)

    logging.info("Polygon aggregation complete. Polygons: %d", len(polygons_out))
    return gpd.GeoDataFrame(polygons_out, geometry="geometry", crs=polygons.crs)


# =============================================================================
# PIPELINES
# =============================================================================


def run_single() -> None:
    """Run the non-split pipeline (one output for all matched stops)."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    stops, stops_key_field = load_bus_stops()
    stops_joined, polygons = spatial_join_to_polygons(stops)

    df_excel = read_and_filter_excel()
    df_excel_stop = aggregate_excel_per_stop(df_excel)

    agg_per_stop_csv = OUTPUT_FOLDER / "agg_ridership_per_stop.csv"
    df_excel_stop.to_csv(agg_per_stop_csv, index=False)
    logging.info("Wrote %s", agg_per_stop_csv)

    matched = merge_ridership(stops_joined, df_excel_stop, stops_key_field)
    matched = add_output_ridership_fields(matched)

    stops_out = output_path("bus_stops_matched")
    layer = "bus_stops_matched" if stops_out.suffix.lower() == ".gpkg" else None
    write_vector(matched, stops_out, layer=layer)
    logging.info("Wrote %s", stops_out)

    matched_csv = OUTPUT_FOLDER / "bus_stops_with_polygon.csv"
    matched.drop(columns="geometry").to_csv(matched_csv, index=False)
    logging.info("Wrote %s", matched_csv)

    if polygons is not None:
        poly_out = aggregate_by_polygon(matched, polygons)

        poly_out_path = output_path("polygon_with_ridership")
        layer = "polygon_with_ridership" if poly_out_path.suffix.lower() == ".gpkg" else None
        write_vector(poly_out, poly_out_path, layer=layer)
        logging.info("Wrote %s", poly_out_path)

        poly_csv = OUTPUT_FOLDER / "agg_ridership_by_polygon.csv"
        poly_out.drop(columns="geometry").to_csv(poly_csv, index=False)
        logging.info("Wrote %s", poly_csv)


def run_split_by_route() -> None:
    """Run the split-by-route pipeline (one output per route)."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    stops, stops_key_field = load_bus_stops()
    stops_joined, polygons = spatial_join_to_polygons(stops)

    df_excel = read_and_filter_excel()
    unique_routes = sorted(pd.unique(df_excel[EXCEL_ROUTE_FIELD].dropna()))
    logging.info("Found %d routes.", len(unique_routes))

    for route in unique_routes:
        df_route = df_excel[df_excel[EXCEL_ROUTE_FIELD] == route].copy()
        if df_route.empty:
            continue

        df_route_stop = aggregate_excel_per_stop(df_route)

        matched = merge_ridership(stops_joined, df_route_stop, stops_key_field)
        if matched.empty:
            logging.warning("No matched stops for route %s; skipping.", route)
            continue

        matched = add_output_ridership_fields(matched)

        stops_out = output_path("bus_stops_matched", route=str(route))
        layer = f"bus_stops_matched_{route}" if stops_out.suffix.lower() == ".gpkg" else None
        write_vector(matched, stops_out, layer=layer)
        logging.info("Wrote %s", stops_out)

        matched_csv = OUTPUT_FOLDER / f"bus_stops_with_polygon_{route}.csv"
        matched.drop(columns="geometry").to_csv(matched_csv, index=False)
        logging.info("Wrote %s", matched_csv)

    # Optional: aggregate polygons across ALL filtered Excel records (not per-route)
    if polygons is not None:
        df_all_stop = aggregate_excel_per_stop(df_excel)
        matched_all = merge_ridership(stops_joined, df_all_stop, stops_key_field)
        matched_all = add_output_ridership_fields(matched_all)

        poly_out = aggregate_by_polygon(matched_all, polygons)
        poly_out_path = output_path("polygon_with_ridership")
        layer = "polygon_with_ridership" if poly_out_path.suffix.lower() == ".gpkg" else None
        write_vector(poly_out, poly_out_path, layer=layer)
        logging.info("Wrote %s", poly_out_path)

        poly_csv = OUTPUT_FOLDER / "agg_ridership_by_polygon.csv"
        poly_out.drop(columns="geometry").to_csv(poly_csv, index=False)
        logging.info("Wrote %s", poly_csv)


# =============================================================================
# PLOTTING
# =============================================================================


def normalize_route_name(name: object) -> str:
    """Uppercase/strip a route name so ridership rows match GTFS route_short_name."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    return str(name).strip().upper()


def _read_gtfs_table(gtfs_dir: Path, filename: str, required_cols: list[str]) -> pd.DataFrame:
    """Read a GTFS table as strings; fail loud if the file or a column is missing."""
    path = gtfs_dir / filename
    if not path.is_file():
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


def build_route_shape_lookup(gtfs_dir: Path) -> tuple:
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


def load_stop_coords(gtfs_dir: Path) -> pd.DataFrame:
    """Return GTFS stop coordinates keyed by GTFS_KEY_FIELD (the stable public key)."""
    stops = _read_gtfs_table(
        gtfs_dir, "stops.txt", [GTFS_KEY_FIELD, GTFS_LAT_FIELD, GTFS_LON_FIELD]
    )
    stops = stops.dropna(subset=[GTFS_KEY_FIELD, GTFS_LAT_FIELD, GTFS_LON_FIELD]).copy()
    stops["stop_lat"] = stops[GTFS_LAT_FIELD].astype(float)
    stops["stop_lon"] = stops[GTFS_LON_FIELD].astype(float)
    stops[GTFS_KEY_FIELD] = stops[GTFS_KEY_FIELD].astype(str)
    return stops[[GTFS_KEY_FIELD, "stop_lon", "stop_lat"]]


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


def load_road_polylines(roads_fc: Path) -> list:
    """Read an optional roads vector file into bbox-indexed lon/lat polylines.

    Read via geopandas and reprojected to WGS84 so a projected roads layer is
    aligned with the GTFS lon/lat geometry. Returns a list of
    (xmin, xmax, ymin, ymax, DataFrame[lon, lat]) tuples; the bbox lets callers
    cheaply filter to a route's extent before drawing. Fails loud if missing.
    """
    if not Path(roads_fc).exists():
        logging.error("ROADS_SHAPEFILE set but not found: %s", roads_fc)
        sys.exit(1)

    roads = gpd.read_file(roads_fc)
    if roads.crs is not None and roads.crs.to_epsg() != 4326:
        roads = roads.to_crs("EPSG:4326")

    indexed: list = []
    for geom in roads.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            continue
        for part in parts:
            xs, ys = part.xy
            if len(xs) < 2:
                continue
            df = pd.DataFrame({"lon": list(xs), "lat": list(ys)})
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
    polylines: list[pd.DataFrame],
    value_name: str,
    out_path: Path,
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
    # keeps its shape on wide, short east–west route extents where an
    # axes-fraction arrow would get vertically squashed.
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

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    name_to_polylines, name_to_color = build_route_shape_lookup(gtfs_dir)
    stop_coords = load_stop_coords(gtfs_dir)

    # Optional roads basemap, read once and reused (bbox-filtered per route).
    roads_indexed = load_road_polylines(ROADS_SHAPEFILE) if ROADS_SHAPEFILE else None

    # Route-level ridership (before the network-wide per-stop collapse used elsewhere)
    ridership = read_and_filter_excel()
    ridership[EXCEL_STOP_ID_FIELD] = ridership[EXCEL_STOP_ID_FIELD].astype(str)
    ridership = ridership.groupby([EXCEL_ROUTE_FIELD, EXCEL_STOP_ID_FIELD], as_index=False).agg(
        {EXCEL_BOARD_FIELD: "sum", EXCEL_ALIGHT_FIELD: "sum"}
    )

    plotted = 0
    skipped: list[str] = []
    for route_name, grp in ridership.groupby(EXCEL_ROUTE_FIELD):
        norm = normalize_route_name(route_name)
        polylines = name_to_polylines.get(norm)
        if not polylines:
            skipped.append(str(route_name))
            continue

        route_color = name_to_color.get(norm) if USE_GTFS_ROUTE_COLOR else None

        merged = grp.merge(
            stop_coords, left_on=EXCEL_STOP_ID_FIELD, right_on=GTFS_KEY_FIELD, how="inner"
        )
        if merged.empty:
            logging.warning("Route %s: no GTFS stop coordinates matched; skipping.", route_name)
            skipped.append(str(route_name))
            continue

        safe = str(route_name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
        for value_col, value_name in (
            (EXCEL_BOARD_FIELD, "Boardings"),
            (EXCEL_ALIGHT_FIELD, "Alightings"),
        ):
            stops_df = pd.DataFrame(
                {
                    "lon": merged["stop_lon"].to_numpy(),
                    "lat": merged["stop_lat"].to_numpy(),
                    "value": merged[value_col].astype(float).to_numpy(),
                }
            )
            out_path = PLOT_DIR / f"route_{safe}_{value_name.lower()}.png"
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
    Captures UPPER_SNAKE_CASE globals holding simple scalar/sequence/path values.
    """
    g = globals()
    lines: list[str] = []
    for name in sorted(g):
        if name.startswith("_") or name != name.upper():
            continue
        val = g[name]
        if isinstance(val, (str, int, float, bool, list, tuple, Path, type(None))):
            lines.append(f"{name} = {val!r}")
    return "\n".join(lines)


def write_run_log(output_folder: Path) -> bool:
    """Write a run log of the configuration into *output_folder*.

    When the script's source is readable, the config block is captured verbatim.
    Otherwise (e.g. a Jupyter kernel with no __file__) it falls back to a live
    snapshot of the config globals so a run still produces a configuration record.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_folder / "stops_ridership_joiner_gpd_runlog.txt"

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
        "STOPS RIDERSHIP JOINER (GPD) RUN LOG",
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


def main() -> int:
    """Main entry point.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _DEFAULT_BUS_STOPS = r"Your\File\Path\To\GTFS_folder"
    _DEFAULT_EXCEL = r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).XLSX"
    if str(BUS_STOPS_INPUT) == _DEFAULT_BUS_STOPS or str(EXCEL_FILE) == _DEFAULT_EXCEL:
        logging.warning(
            "File paths are still set to their defaults. Update BUS_STOPS_INPUT and "
            "EXCEL_FILE in the CONFIGURATION section before running."
        )
        return 2

    if not BUS_STOPS_INPUT.exists():
        raise FileNotFoundError(f"BUS_STOPS_INPUT not found: {BUS_STOPS_INPUT}")
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"EXCEL_FILE not found: {EXCEL_FILE}")
    if POLYGON_LAYER is not None and not POLYGON_LAYER.exists():
        raise FileNotFoundError(f"POLYGON_LAYER not found: {POLYGON_LAYER}")

    logging.info("Output folder: %s", OUTPUT_FOLDER)
    logging.info("Split by route: %s", SPLIT_BY_ROUTE)
    logging.info("Output format: %s", OUT_FORMAT)

    if SPLIT_BY_ROUTE:
        run_split_by_route()
    else:
        run_single()

    # Per-route ridership maps (independent of SPLIT_BY_ROUTE).
    if DRAW_PLOTS:
        logging.info("DRAW_PLOTS = True. Generating per-route ridership maps.")
        generate_route_plots()

    if not write_run_log(OUTPUT_FOLDER) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        return 1

    logging.info("Done. Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
