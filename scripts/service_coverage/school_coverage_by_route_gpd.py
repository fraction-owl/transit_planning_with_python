"""Roll school counts and total enrollment up to GTFS routes for modeling joins.

This is an intermediate ("prep") script in the same spirit as
``points_of_interest_coverage_gpd.py`` and the route-level output of
``gtfs_service_demographics_gpd.py``: it turns a point layer of schools into a
single, route-keyed feature table that the modeling pipeline
(``scripts/modeling/prep_features.py`` → ``fit_model.py``) can join onto the
ridership anchor by ``route_id``.

It buffers each route's stops (a simple fixed-radius catchment is intentional
for now) and, for every route, counts the schools whose point falls inside the
catchment and sums their enrollment. The result is one row per route_id with
``schools_served`` and ``enrollment_served`` — the exact columns the orchestrator
registry already describes for ``school_coverage_by_route.csv``.

Inputs
------
- A GTFS folder containing routes.txt, trips.txt, stop_times.txt, stops.txt
  (and shapes.txt only when ``--shape-buffer`` is used).
- One or more school *point* layers carrying an enrollment column. These are the
  enrollment-joined points written by
  ``national_data_tools/schools_prep_join_gpd.py``
  (``va_md_dc_<type>_schools_enrollment.gpkg``); pass a single file or a folder
  to combine public + private + postsec into one rollup.

Output
------
- ``school_coverage_by_route.csv`` — columns ``route_id``, ``route_short_name``
  (when available), ``schools_served``, ``enrollment_served``.

Assumptions
-----------
- The projected CRS uses meters or feet; the buffer distance is given in feet and
  converted to meters when the CRS is metric.
- Schools with no matched enrollment (NaN) still count toward ``schools_served``
  but contribute 0 to ``enrollment_served``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Mapping, Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top-level directories
GTFS_DIR = Path(r"data/gtfs")  # folder containing GTFS .txt files
SCHOOLS_PATH = Path(r"data/schools")  # school point file, or a folder of them
OUTPUT_DIR = Path(r"output")  # where the rollup CSV is written

# When SCHOOLS_PATH is a folder, every file matching these glob patterns is loaded
# and combined (so public + private + postsec roll up together). The patterns are
# matched recursively. Ignored when SCHOOLS_PATH points straight at a file.
SCHOOLS_GLOBS: tuple[str, ...] = (
    "*schools_enrollment*.gpkg",
    "*schools_enrollment*.shp",
    "*schools_enrollment*.geojson",
)

# Column on the school layer holding total enrollment. This is the name
# schools_prep_join_gpd.py writes for every source (CCD / ELSI / IPEDS).
ENROLLMENT_COLUMN = "enroll_total"

# Optional filter: only analyze these route_id values.
# Leave empty (`[]`) to process every route in routes.txt
ROUTE_FILTER: list[str] = []

# Analysis options
USE_SHAPE_BUFFER = False  # False → buffer stops (simple catchment); True → route geometry
BUFFER_DIST_FT = 1320.0  # ¼ mile in feet

# Output filename — matches the orchestrator registry's school_coverage_by_route.csv.
OUTPUT_CSV_NAME = "school_coverage_by_route.csv"

# Projected CRS used for buffering and the spatial join.
# EPSG:3857 (Web Mercator) works globally; swap for a local CRS (e.g. "EPSG:2283"
# for northern Virginia in feet) when higher spatial accuracy is needed.
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
        ValueError: If shape-buffer mode is requested but shapes.txt is malformed.
    """
    trips = tables["trips"][["route_id", "trip_id", "shape_id"]].copy()
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
                lambda grp: LineString(
                    grp[["shape_pt_lon", "shape_pt_lat"]].to_numpy(dtype=float)
                )
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


def load_schools_layer(
    schools_path: Path,
    schools_globs: Sequence[str] = SCHOOLS_GLOBS,
    enrollment_column: str = ENROLLMENT_COLUMN,
    projected_crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """Load school point layer(s), combine them, and reproject.

    Accepts either a single vector file or a folder. For a folder, every file
    matching *schools_globs* (recursively) is read and concatenated, so the
    public / private / postsec outputs of ``schools_prep_join_gpd.py`` roll up
    together. The enrollment column is coerced to numeric; a layer missing it is
    kept (its rows still count as schools) with enrollment treated as 0.

    Args:
        schools_path: A school vector file, or a folder containing such files.
        schools_globs: Glob patterns used when *schools_path* is a folder.
        enrollment_column: Column holding total enrollment on each layer.
        projected_crs: CRS to reproject the combined layer into.

    Returns:
        Point GeoDataFrame in *projected_crs* with a numeric enrollment column.

    Raises:
        FileNotFoundError: If no school layer is found.
    """
    if schools_path.is_dir():
        paths = sorted(
            {p for pattern in schools_globs for p in schools_path.rglob(pattern)}
        )
        if not paths:
            raise FileNotFoundError(
                f"No school layers matching {list(schools_globs)} under {schools_path}"
            )
    elif schools_path.exists():
        paths = [schools_path]
    else:
        raise FileNotFoundError(schools_path)

    frames: list[gpd.GeoDataFrame] = []
    for path in paths:
        gdf = gpd.read_file(path)
        if enrollment_column in gdf.columns:
            gdf[enrollment_column] = pd.to_numeric(gdf[enrollment_column], errors="coerce")
        else:
            logging.warning(
                "Enrollment column %r missing in %s; counts only.",
                enrollment_column,
                path.name,
            )
            gdf[enrollment_column] = 0.0
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        frames.append(gdf[[enrollment_column, "geometry"]].to_crs(projected_crs))
        logging.info("Loaded %d schools from %s", len(gdf), path.name)

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=projected_crs
    )
    logging.info("Combined %d school points from %d layer(s)", len(combined), len(paths))
    return combined


def summarize_schools_by_route(
    route_buffers: gpd.GeoDataFrame,
    schools_gdf: gpd.GeoDataFrame,
    enrollment_column: str = ENROLLMENT_COLUMN,
) -> pd.DataFrame:
    """Count schools and sum enrollment falling inside each route catchment.

    Args:
        route_buffers: One buffered catchment per route_id (from
            :func:`_prepare_route_buffers`).
        schools_gdf: School points with a numeric enrollment column, in the same
            CRS as *route_buffers*.
        enrollment_column: Column on *schools_gdf* holding total enrollment.

    Returns:
        DataFrame with one row per route_id and columns ``schools_served`` and
        ``enrollment_served``. Routes with no schools nearby report zeros.
    """
    routes = route_buffers[["route_id"]].drop_duplicates().reset_index(drop=True)

    if not schools_gdf.empty:
        joined = gpd.sjoin(
            schools_gdf,
            route_buffers[["route_id", "geometry"]],
            predicate="intersects",
            how="inner",
        )
        grouped = joined.groupby("route_id").agg(
            schools_served=(enrollment_column, "size"),
            enrollment_served=(enrollment_column, "sum"),
        )
        summary = routes.merge(grouped, on="route_id", how="left")
    else:
        summary = routes.assign(schools_served=0, enrollment_served=0.0)

    summary["schools_served"] = summary["schools_served"].fillna(0).astype(int)
    summary["enrollment_served"] = summary["enrollment_served"].fillna(0).round(0).astype(int)
    return summary


def _attach_route_short_name(
    summary: pd.DataFrame, routes_df: pd.DataFrame
) -> pd.DataFrame:
    """Add a readable ``route_short_name`` column when routes.txt carries one."""
    if "route_short_name" not in routes_df.columns:
        return summary
    lookup = routes_df.assign(route_id=routes_df["route_id"].astype(str))[
        ["route_id", "route_short_name"]
    ].drop_duplicates(subset="route_id")
    merged = summary.merge(lookup, on="route_id", how="left")
    cols = ["route_id", "route_short_name", "schools_served", "enrollment_served"]
    return merged[cols]


# =============================================================================
# MAIN
# =============================================================================


def run(
    gtfs_dir: str | Path | None = None,
    schools_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_shape_buffer: bool | None = None,
    buffer_dist_ft: float | None = None,
    route_filter: Sequence[str] | None = None,
    projected_crs: str | None = None,
    enrollment_column: str | None = None,
    output_csv_name: str | None = None,
) -> pd.DataFrame:
    """Build the route-level school coverage rollup and write it to CSV.

    Unset args fall back to the CONFIGURATION block, so ``m.GTFS_DIR = ...;
    m.run()`` works after a plain import.

    Returns:
        The summary DataFrame that was written to disk.
    """
    gtfs_dir = Path(GTFS_DIR if gtfs_dir is None else gtfs_dir)
    schools_path = Path(SCHOOLS_PATH if schools_path is None else schools_path)
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    use_shape_buffer = USE_SHAPE_BUFFER if use_shape_buffer is None else use_shape_buffer
    buffer_dist_ft = BUFFER_DIST_FT if buffer_dist_ft is None else buffer_dist_ft
    route_filter = list(ROUTE_FILTER if route_filter is None else route_filter)
    projected_crs = PROJECTED_CRS if projected_crs is None else projected_crs
    enrollment_column = ENROLLMENT_COLUMN if enrollment_column is None else enrollment_column
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
        return pd.DataFrame(columns=["route_id", "schools_served", "enrollment_served"])

    logging.info("Loading school points from %s", schools_path)
    schools_gdf = load_schools_layer(
        schools_path,
        enrollment_column=enrollment_column,
        projected_crs=projected_crs,
    )

    logging.info("Rolling schools up to %d routes", len(route_buffers))
    summary = summarize_schools_by_route(route_buffers, schools_gdf, enrollment_column)
    summary = _attach_route_short_name(summary, tables["routes"])

    out_path = output_dir / output_csv_name
    summary.to_csv(out_path, index=False)
    logging.info(
        "Wrote %s (%d routes, %d schools, %d total enrollment served)",
        out_path,
        len(summary),
        int(summary["schools_served"].sum()),
        int(summary["enrollment_served"].sum()),
    )
    logging.info("Script completed successfully.")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block."""
    parser = argparse.ArgumentParser(
        description=(
            "Roll school counts and total enrollment up to GTFS routes. Defaults "
            "come from the CONFIGURATION block at the top of this file."
        )
    )
    parser.add_argument(
        "--gtfs-dir", type=Path, default=GTFS_DIR, help="Folder containing GTFS .txt files."
    )
    parser.add_argument(
        "--schools-path",
        type=Path,
        default=SCHOOLS_PATH,
        help="School point file, or a folder of *schools_enrollment* layers to combine.",
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
        "--enrollment-column",
        default=ENROLLMENT_COLUMN,
        help="Column on the school layer holding total enrollment.",
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


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. Defaults fall back to the CONFIGURATION block."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run(
        gtfs_dir=args.gtfs_dir,
        schools_path=args.schools_path,
        output_dir=args.output_dir,
        use_shape_buffer=args.use_shape_buffer,
        buffer_dist_ft=args.buffer_ft,
        route_filter=args.routes,
        projected_crs=args.projected_crs,
        enrollment_column=args.enrollment_column,
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
