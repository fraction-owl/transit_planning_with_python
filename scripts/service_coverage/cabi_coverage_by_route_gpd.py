"""Roll Capital Bikeshare station counts and day-type ridership up to GTFS routes.

This is an intermediate ("prep") script in the same spirit as
``school_coverage_by_route_gpd.py`` and ``points_of_interest_coverage_gpd.py``:
it turns the bikeshare station point layer into a single, route-keyed feature
table that the modeling pipeline (``scripts/modeling/prep_features_public.py``
→ ``monthly_ridership_model.py``) can join onto the ridership anchor by
``route_id``.

It buffers each route's stops (a simple fixed-radius catchment is intentional
for now) and, for every route, counts the bikeshare stations whose point falls
inside the catchment and sums their average daily ridership by day type. The
result is one row per route_id with ``cabi_stations_served``,
``cabi_weekday_riders_served``, ``cabi_saturday_riders_served``, and
``cabi_sunday_riders_served`` — the columns the orchestrator registry
describes for ``cabi_coverage_by_route.csv``.

Inputs
------
- A GTFS folder containing routes.txt, trips.txt, stop_times.txt, stops.txt
  (and shapes.txt only when ``--shape-buffer`` is used).
- A station *point* layer carrying ``station_id`` — the
  ``gbfs_stations.geojson`` / ``gbfs_stations.shp`` written by
  ``gbfs_tools/gbfs_stations_exporter.py``; pass a single file or a folder to
  glob for one.
- ``station_daytype_ridership.csv`` written by
  ``gbfs_tools/bikeshare_ridership_trends.py`` — one row per station with
  ``avg_weekday_riders`` / ``avg_saturday_riders`` / ``avg_sunday_riders``.
  Its weekday average covers non-holiday weekdays only (observed U.S. federal
  holidays are classified as Sunday-equivalent there), so the route-level
  ``cabi_weekday_riders_served`` matches the non-holiday-weekday posture of
  the GTFS-side feature scripts.

Output
------
- ``cabi_coverage_by_route.csv`` — columns ``route_id``, ``route_short_name``
  (when available), ``cabi_stations_served``, ``cabi_weekday_riders_served``,
  ``cabi_saturday_riders_served``, and ``cabi_sunday_riders_served``.

Assumptions
-----------
- The projected CRS uses meters or feet; the buffer distance is given in feet
  and converted to meters when the CRS is metric. Under the default Web
  Mercator CRS the buffer is additionally scaled by ~1/cos(latitude) so it
  spans a true ground distance (raw Web Mercator "meters" shrink with
  latitude).
- Stations with no ridership row (e.g. new docks with no recorded trips) still
  count toward ``cabi_stations_served`` but contribute 0 riders.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List, Mapping, Sequence

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import LineString
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top-level directories
GTFS_DIR = Path(r"data/gtfs")  # folder containing GTFS .txt files
STATIONS_PATH = Path(r"data/gbfs")  # station point file, or a folder holding one
RIDERSHIP_CSV = Path(r"data/station_daytype_ridership.csv")  # day-type averages
OUTPUT_DIR = Path(r"output")  # where the rollup CSV is written

# When STATIONS_PATH is a folder, the first file matching these glob patterns
# (searched recursively, in pattern order) is loaded. Ignored when
# STATIONS_PATH points straight at a file.
STATIONS_GLOBS: tuple[str, ...] = (
    "gbfs_stations*.geojson",
    "gbfs_stations*.shp",
    "*stations*.geojson",
    "*stations*.shp",
)

# Column identifying the station on both the point layer and the ridership CSV.
STATION_ID_FIELD = "station_id"

# Day-type average columns expected on the ridership CSV, mapped to their
# route-level output column names.
RIDERSHIP_OUTPUT_COLUMNS: dict[str, str] = {
    "avg_weekday_riders": "cabi_weekday_riders_served",
    "avg_saturday_riders": "cabi_saturday_riders_served",
    "avg_sunday_riders": "cabi_sunday_riders_served",
}

# Optional filter: only analyze these route_id values.
# Leave empty (`[]`) to process every route in routes.txt
ROUTE_FILTER: list[str] = []

# Analysis options
USE_SHAPE_BUFFER = False  # False → buffer stops (simple catchment); True → route geometry
BUFFER_DIST_FT = 1320.0  # ¼ mile in feet

# Output filename — matches the orchestrator registry's cabi_coverage_by_route.csv.
OUTPUT_CSV_NAME = "cabi_coverage_by_route.csv"

# Projected CRS used for buffering and the spatial join.
# EPSG:3857 (Web Mercator) works globally; its latitude-dependent scale
# distortion is corrected automatically when buffering (see
# _web_mercator_ground_scale). Swap for a local CRS (e.g. "EPSG:2283" for
# northern Virginia in feet) when higher spatial accuracy is needed.
PROJECTED_CRS = "EPSG:3857"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def _load_gtfs_tables(gtfs_dir: Path, *, need_shapes: bool) -> Mapping[str, pd.DataFrame]:
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

    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        path = gtfs_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[name] = pd.read_csv(path, dtype={"route_id": str})
        logging.debug("Loaded %s (%d rows)", name, len(tables[name]))
    return tables


def _prepare_route_buffers(
    tables: Mapping[str, pd.DataFrame],
    use_shape_buffer: bool,
    buffer_dist_ft: float,
    route_filter: list[str] | None = None,
    projected_crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with one buffered catchment geometry per route_id.

    Depending on *use_shape_buffer*, the buffer is built around the union of
    (a) the route's shape(s) or (b) all of its stops. The buffer distance is
    given in feet and converted to meters when *projected_crs* is metric.

    Args:
        tables: GTFS tables from :func:`_load_gtfs_tables`.
        use_shape_buffer: Buffer route geometry when True, else buffer stops.
        buffer_dist_ft: Catchment radius in feet.
        route_filter: Optional list of route_id values to keep (empty = all).
        projected_crs: Projected CRS used for buffering.

    Raises:
        ValueError: If shape-buffer mode is requested but trips.txt lacks a
            ``shape_id`` column or shapes.txt is malformed.
    """
    # shape_id is optional in GTFS and only needed to buffer route geometry,
    # so stop-buffer mode must not require the column.
    trip_cols = ["route_id", "trip_id"]
    if use_shape_buffer:
        if "shape_id" not in tables["trips"].columns:
            raise ValueError("trips.txt missing shape_id column (required for shape-buffer mode)")
        trip_cols.append("shape_id")
    trips = tables["trips"][trip_cols].copy()
    trips["route_id"] = trips["route_id"].astype(str)

    # Stops GeoDataFrame (always built; the default catchment buffers stops).
    stops = tables["stops"][["stop_id", "stop_lat", "stop_lon"]].copy()
    stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    ).to_crs(projected_crs)

    shapes_gdf: gpd.GeoDataFrame | None = None
    route_shapes: pd.Series | None = None
    if use_shape_buffer:
        shapes_df = tables["shapes"]
        if {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.difference(
            shapes_df.columns
        ):
            raise ValueError("shapes.txt missing required columns")
        shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
        lines = (
            shapes_df.groupby("shape_id")
            .apply(
                lambda grp: LineString(grp[["shape_pt_lon", "shape_pt_lat"]].to_numpy(dtype=float))
            )
            .to_frame(name="geometry")
        )
        shapes_gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(
            projected_crs
        )
        route_shapes = (
            trips.drop_duplicates(subset=["route_id", "shape_id"])
            .groupby("route_id")["shape_id"]
            .apply(list)
        )

    # ¼ mile etc. is configured in feet; reproject-aware conversion to meters.
    buff_dist = buffer_dist_ft
    if _crs_is_metric(stops.crs):
        buff_dist = buffer_dist_ft * 0.3048
    buff_dist *= _web_mercator_ground_scale(stops.crs, tables["stops"]["stop_lat"])

    route_ids = route_shapes.index if route_shapes is not None else trips["route_id"].unique()

    buffers: List[dict[str, object]] = []
    for route_id in route_ids:
        route_id = str(route_id)
        if route_filter and route_id not in route_filter:
            continue

        if use_shape_buffer and shapes_gdf is not None and route_shapes is not None:
            shp_ids = [s for s in route_shapes.loc[route_id] if s in shapes_gdf.index]
            geoms = shapes_gdf.loc[shp_ids, "geometry"]
        else:
            trip_stops = (
                tables["stop_times"]
                .merge(trips[trips.route_id == route_id][["trip_id"]], on="trip_id", how="inner")[
                    "stop_id"
                ]
                .unique()
            )
            geoms = stops[stops.stop_id.isin(trip_stops)].geometry

        if geoms.empty:
            logging.warning("No geometry for route %s – skipped", route_id)
            continue

        buf = unary_union(list(geoms)).buffer(buff_dist)
        buffers.append({"route_id": route_id, "geometry": buf})

    return gpd.GeoDataFrame(buffers, geometry="geometry", crs=projected_crs)


def _crs_is_metric(crs: object) -> bool:
    """Return True when *crs* measures distance in meters (best-effort)."""
    try:
        unit = crs.axis_info[0].unit_name.lower()  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        return True  # default GTFS reprojection target (3857) is metric
    return "metre" in unit or "meter" in unit


def _web_mercator_ground_scale(crs: object, latitudes: pd.Series) -> float:
    """Return the Web Mercator map-per-ground distance factor, or 1.0 elsewhere.

    Web Mercator (EPSG:3857) inflates distances by roughly ``1/cos(latitude)``:
    at 39°N a true quarter mile spans ~1.29x as many map "meters", so a buffer
    drawn in raw map units under-covers the ground by the same factor. Buffer
    distances are multiplied by this factor so they span a true ground distance
    at the analysis area's mean latitude. Any other CRS returns 1.0 (projected
    local CRSs are treated as true-scale).

    Args:
        crs: The projected CRS in use (anything pyproj can parse).
        latitudes: WGS 84 latitudes of the analysis features; their mean sets
            the correction.

    Returns:
        The multiplier to apply to a ground distance before buffering.
    """
    try:
        epsg = CRS.from_user_input(crs).to_epsg()
    except CRSError:
        return 1.0
    if epsg != 3857:
        return 1.0
    lat = pd.to_numeric(latitudes, errors="coerce").dropna()
    if lat.empty:
        return 1.0
    mean_lat = float(lat.mean())
    if not -89.0 < mean_lat < 89.0:
        return 1.0
    scale = 1.0 / math.cos(math.radians(mean_lat))
    logging.info(
        "Web Mercator inflates distances by %.4f at latitude %.3f; scaling the "
        "buffer to preserve ground distance.",
        scale,
        mean_lat,
    )
    return scale


def _safe_to_str(value: object) -> str:
    """Return a trimmed string form of an identifier value.

    Args:
        value: Any identifier-like value.

    Returns:
        A stripped string, with trailing ``.0`` removed for float-like ids.
    """
    return str(value).strip().removesuffix(".0")


def load_stations_layer(
    stations_path: Path,
    stations_globs: Sequence[str] = STATIONS_GLOBS,
    station_id_field: str = STATION_ID_FIELD,
    projected_crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """Load the bikeshare station point layer and reproject it.

    Accepts either a single vector file or a folder; for a folder, the first
    file matching *stations_globs* (searched recursively, in pattern order) is
    used. Duplicate station ids keep their first occurrence.

    Args:
        stations_path: A station vector file, or a folder containing one.
        stations_globs: Glob patterns tried when *stations_path* is a folder.
        station_id_field: Column identifying the station.
        projected_crs: CRS to reproject the layer into.

    Returns:
        Point GeoDataFrame in *projected_crs* with a normalized string
        ``station_id`` column.

    Raises:
        FileNotFoundError: If no station layer is found.
        KeyError: If the layer is missing *station_id_field*.
    """
    if stations_path.is_dir():
        path = next(
            (p for pattern in stations_globs for p in sorted(stations_path.rglob(pattern))),
            None,
        )
        if path is None:
            raise FileNotFoundError(
                f"No station layers matching {list(stations_globs)} under {stations_path}"
            )
    elif stations_path.exists():
        path = stations_path
    else:
        raise FileNotFoundError(stations_path)

    stations = gpd.read_file(path)
    if station_id_field not in stations.columns:
        raise KeyError(
            f"Station layer is missing the join column {station_id_field!r}; "
            f"found columns: {list(stations.columns)}"
        )
    if stations.crs is None:
        stations = stations.set_crs(epsg=4326)
    stations = stations.to_crs(projected_crs)
    stations[station_id_field] = stations[station_id_field].map(_safe_to_str)
    stations = stations.drop_duplicates(subset=station_id_field)
    logging.info("Loaded %d stations from %s", len(stations), path.name)
    return stations[[station_id_field, "geometry"]].reset_index(drop=True)


def load_daytype_ridership(
    ridership_csv: Path, station_id_field: str = STATION_ID_FIELD
) -> pd.DataFrame:
    """Load the per-station day-type ridership averages.

    Args:
        ridership_csv: Path to ``station_daytype_ridership.csv`` (from
            ``bikeshare_ridership_trends.py``).
        station_id_field: Column identifying the station.

    Returns:
        DataFrame with one row per station and the three day-type average
        columns, id normalized to match the station layer.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        KeyError: If the id column or a day-type average column is missing.
    """
    if not ridership_csv.exists():
        raise FileNotFoundError(ridership_csv)
    ridership = pd.read_csv(ridership_csv)
    missing = [
        col for col in (station_id_field, *RIDERSHIP_OUTPUT_COLUMNS) if col not in ridership.columns
    ]
    if missing:
        raise KeyError(
            f"Ridership table {ridership_csv.name} is missing column(s) {missing}; "
            f"found columns: {list(ridership.columns)}"
        )
    ridership[station_id_field] = ridership[station_id_field].map(_safe_to_str)
    ridership = ridership.drop_duplicates(subset=station_id_field)
    return ridership[[station_id_field, *RIDERSHIP_OUTPUT_COLUMNS]].reset_index(drop=True)


def join_ridership_onto_stations(
    stations: gpd.GeoDataFrame,
    ridership: pd.DataFrame,
    station_id_field: str = STATION_ID_FIELD,
) -> gpd.GeoDataFrame:
    """Left-join day-type averages onto station points, zero-filling gaps.

    Every station geometry is kept: a station with no ridership row (e.g. a
    new dock) still counts toward ``cabi_stations_served`` downstream but
    contributes 0 riders. Ridership rows with no matching geometry are logged
    and dropped — they cannot be placed on a route.

    Args:
        stations: Station points (from :func:`load_stations_layer`).
        ridership: Day-type averages (from :func:`load_daytype_ridership`).
        station_id_field: Column identifying the station on both sides.

    Returns:
        Stations enriched with the three day-type average columns.
    """
    merged = stations.merge(ridership, on=station_id_field, how="left")
    unmatched = len(ridership) - merged[list(RIDERSHIP_OUTPUT_COLUMNS)[0]].notna().sum()
    if unmatched > 0:
        logging.warning(
            "%d ridership station(s) have no geometry in the station layer and are dropped.",
            unmatched,
        )
    for col in RIDERSHIP_OUTPUT_COLUMNS:
        merged[col] = merged[col].fillna(0.0)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=stations.crs)


def summarize_stations_by_route(
    route_buffers: gpd.GeoDataFrame,
    stations_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Count stations and sum day-type ridership per route catchment.

    Args:
        route_buffers: One buffered catchment per route_id (from
            :func:`_prepare_route_buffers`).
        stations_gdf: Station points carrying the day-type average columns
            (from :func:`join_ridership_onto_stations`), in the same CRS as
            *route_buffers*.

    Returns:
        DataFrame with one row per route_id and columns
        ``cabi_stations_served``, ``cabi_weekday_riders_served``,
        ``cabi_saturday_riders_served``, and ``cabi_sunday_riders_served``.
        Routes with no stations nearby report zeros.
    """
    routes = route_buffers[["route_id"]].drop_duplicates().reset_index(drop=True)
    rider_cols = list(RIDERSHIP_OUTPUT_COLUMNS.values())

    if not stations_gdf.empty:
        joined = gpd.sjoin(
            stations_gdf,
            route_buffers[["route_id", "geometry"]],
            predicate="intersects",
            how="inner",
        )
        agg = {"cabi_stations_served": (STATION_ID_FIELD, "size")}
        agg.update(
            {out_col: (avg_col, "sum") for avg_col, out_col in RIDERSHIP_OUTPUT_COLUMNS.items()}
        )
        grouped = joined.groupby("route_id").agg(**agg)
        summary = routes.merge(grouped, on="route_id", how="left")
    else:
        summary = routes.assign(cabi_stations_served=0, **{c: 0.0 for c in rider_cols})

    summary["cabi_stations_served"] = summary["cabi_stations_served"].fillna(0).astype(int)
    for out_col in rider_cols:
        summary[out_col] = summary[out_col].fillna(0.0).round(2)
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
        "cabi_stations_served",
        *RIDERSHIP_OUTPUT_COLUMNS.values(),
    ]
    return merged[cols]


# =============================================================================
# MAIN
# =============================================================================


def run(
    gtfs_dir: str | Path | None = None,
    stations_path: str | Path | None = None,
    ridership_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_shape_buffer: bool | None = None,
    buffer_dist_ft: float | None = None,
    route_filter: Sequence[str] | None = None,
    projected_crs: str | None = None,
    station_id_field: str | None = None,
    output_csv_name: str | None = None,
) -> pd.DataFrame:
    """Build the route-level bikeshare coverage rollup and write it to CSV.

    Unset args fall back to the CONFIGURATION block, so ``m.GTFS_DIR = ...;
    m.run()`` works after a plain import.

    Returns:
        The summary DataFrame that was written to disk.
    """
    gtfs_dir = Path(GTFS_DIR if gtfs_dir is None else gtfs_dir)
    stations_path = Path(STATIONS_PATH if stations_path is None else stations_path)
    ridership_csv = Path(RIDERSHIP_CSV if ridership_csv is None else ridership_csv)
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    use_shape_buffer = USE_SHAPE_BUFFER if use_shape_buffer is None else use_shape_buffer
    buffer_dist_ft = BUFFER_DIST_FT if buffer_dist_ft is None else buffer_dist_ft
    route_filter = list(ROUTE_FILTER if route_filter is None else route_filter)
    projected_crs = PROJECTED_CRS if projected_crs is None else projected_crs
    station_id_field = STATION_ID_FIELD if station_id_field is None else station_id_field
    output_csv_name = OUTPUT_CSV_NAME if output_csv_name is None else output_csv_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading GTFS from %s", gtfs_dir)
    tables = _load_gtfs_tables(gtfs_dir, need_shapes=use_shape_buffer)

    logging.info("Building route catchments (use_shape_buffer=%s)", use_shape_buffer)
    route_buffers = _prepare_route_buffers(
        tables,
        use_shape_buffer,
        buffer_dist_ft,
        route_filter=route_filter,
        projected_crs=projected_crs,
    )
    if route_buffers.empty:
        logging.error("No route catchments produced – nothing to do")
        return pd.DataFrame(columns=["route_id", "cabi_stations_served"])

    logging.info("Loading bikeshare stations from %s", stations_path)
    stations = load_stations_layer(
        stations_path,
        station_id_field=station_id_field,
        projected_crs=projected_crs,
    )

    logging.info("Loading day-type ridership from %s", ridership_csv)
    ridership = load_daytype_ridership(ridership_csv, station_id_field)
    stations_gdf = join_ridership_onto_stations(stations, ridership, station_id_field)

    logging.info("Rolling %d stations up to %d routes", len(stations_gdf), len(route_buffers))
    summary = summarize_stations_by_route(route_buffers, stations_gdf)
    summary = _attach_route_short_name(summary, tables["routes"])

    out_path = output_dir / output_csv_name
    summary.to_csv(out_path, index=False)
    logging.info(
        "Wrote %s (%d routes, %d station placements, %.1f weekday riders served)",
        out_path,
        len(summary),
        int(summary["cabi_stations_served"].sum()),
        float(summary["cabi_weekday_riders_served"].sum()),
    )
    logging.info("Script completed successfully.")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block."""
    parser = argparse.ArgumentParser(
        description=(
            "Roll bikeshare station counts and day-type ridership up to GTFS routes. "
            "Defaults come from the CONFIGURATION block at the top of this file."
        )
    )
    parser.add_argument(
        "--gtfs-dir", type=Path, default=GTFS_DIR, help="Folder containing GTFS .txt files."
    )
    parser.add_argument(
        "--stations-path",
        type=Path,
        default=STATIONS_PATH,
        help="Station point file (gbfs_stations.geojson/.shp), or a folder holding one.",
    )
    parser.add_argument(
        "--ridership-csv",
        type=Path,
        default=RIDERSHIP_CSV,
        help="station_daytype_ridership.csv from bikeshare_ridership_trends.py.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Where the rollup CSV is written."
    )
    parser.add_argument(
        "--buffer-ft", type=float, default=BUFFER_DIST_FT, help="Catchment radius in feet."
    )
    parser.add_argument(
        "--shape-buffer",
        dest="use_shape_buffer",
        action="store_true",
        default=USE_SHAPE_BUFFER,
        help="Buffer route geometry instead of stops (requires shapes.txt).",
    )
    parser.add_argument(
        "--routes",
        nargs="*",
        default=ROUTE_FILTER,
        metavar="ROUTE_ID",
        help="Only analyze these route_id values (default: all).",
    )
    parser.add_argument(
        "--station-id-field",
        default=STATION_ID_FIELD,
        help="Column identifying the station on the point layer and the ridership CSV.",
    )
    parser.add_argument(
        "--projected-crs", default=PROJECTED_CRS, help="Projected CRS for buffering/joins."
    )
    parser.add_argument(
        "--output-name", default=OUTPUT_CSV_NAME, help="Name of the output CSV file."
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(argv)


# Literal placeholder input paths shipped in the CONFIGURATION block, frozen
# here (do not edit) so main() can tell an unedited config from a real one. An
# input equal to its placeholder in BOTH the CONFIG constant and the CLI arg
# was customized nowhere. Comparing args against the live CONFIG constants
# instead would always match whenever a flag is omitted (argparse defaults to
# those constants), wrongly blocking the edit-CONFIG-then-run workflow.
_PLACEHOLDER_GTFS_DIR = Path(r"data/gtfs")
_PLACEHOLDER_STATIONS_PATH = Path(r"data/gbfs")
_PLACEHOLDER_RIDERSHIP_CSV = Path(r"data/station_daytype_ridership.csv")


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. Defaults fall back to the CONFIGURATION block."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    unset = [
        (name, flag)
        for name, flag, arg_value, config_value, placeholder in (
            ("GTFS_DIR", "--gtfs-dir", args.gtfs_dir, GTFS_DIR, _PLACEHOLDER_GTFS_DIR),
            (
                "STATIONS_PATH",
                "--stations-path",
                args.stations_path,
                STATIONS_PATH,
                _PLACEHOLDER_STATIONS_PATH,
            ),
            (
                "RIDERSHIP_CSV",
                "--ridership-csv",
                args.ridership_csv,
                RIDERSHIP_CSV,
                _PLACEHOLDER_RIDERSHIP_CSV,
            ),
        )
        if Path(arg_value) == placeholder and Path(config_value) == placeholder
    ]
    if unset:
        logging.warning(
            "%s still point(s) at the placeholder path(s) from the CONFIGURATION "
            "block. Update the CONFIGURATION section or pass %s before running.",
            " and ".join(name for name, _ in unset),
            " / ".join(flag for _, flag in unset),
        )
        return
    run(
        gtfs_dir=args.gtfs_dir,
        stations_path=args.stations_path,
        ridership_csv=args.ridership_csv,
        output_dir=args.output_dir,
        use_shape_buffer=args.use_shape_buffer,
        buffer_dist_ft=args.buffer_ft,
        route_filter=args.routes,
        projected_crs=args.projected_crs,
        station_id_field=args.station_id_field,
        output_csv_name=args.output_name,
    )


def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":
    # In a notebook (pasted cell or %run), use the CONFIGURATION block instead
    # of argparse, which would otherwise try to parse the kernel's own argv.
    if _in_ipython():
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        run()
    else:
        main()
