"""Analyze transit service coverage of strategic sites using GTFS and GIS data.

This script evaluates how well individual transit routes serve strategically important
locations—such as public housing developments, high schools, hospitals, parks, metro/rail
stations, and other community facilities—based on spatial proximity.

Intended Use
------------
This tool is designed to support transit planning by quantifying access to key destinations.
Results can inform decisions about route coverage, service prioritization, and equity evaluation.

Assumptions
-----------
- GTFS and GIS layers are projected in a CRS using feet or meters.
- Buffer distance is assumed to be in feet (auto-converted to meters if needed).
- Each shapefile includes a column with a readable feature name (e.g., school name).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top‑level directories
GTFS_DIR = Path(r"data/gtfs")  # folder containing GTFS .txt files
SHP_INPUT_DIR = Path(r"data/shapefiles")  # folder with .shp layers to test
OUTPUT_DIR = Path(r"output")  # where CSVs and PNGs are written

# List of `(filename, id_column)` describing each layer to test
# (filenames are relative to SHP_INPUT_DIR)
LAYER_SPECS: list[tuple[str, str]] = [
    ("Hospitals_and_Urgent_Care_Facilities.shp", "DESCRIPTIO"),
    ("School_Facilities.shp", "SCHOOL_NAM"),
    ("Libraries.shp", "DESCRIPTIO"),
    ("Metrorail_Stations.shp", "NAME"),
]

# Optional filter: only analyze these route_id values.
# Leave empty (`[]`) to process every route in routes.txt
ROUTE_FILTER: list[str] = []

# Analysis options
USE_SHAPE_BUFFER = True  # True → buffer route geometry; False → buffer stops
BUFFER_DIST_FT = 1320.0  # ¼ mile in feet
# Simplify each route's geometry (Douglas-Peucker, in projected meters) before
# buffering. Buffering full-resolution shapes (GTFS shapes can carry thousands
# of vertices per route) is the slowest step here; a tolerance this small is
# negligible against a 402 m (1/4 mile) buffer but cuts buffer time many-fold.
# Set to 0.0 to disable simplification.
SIMPLIFY_TOLERANCE_M = 10.0
# Per-route map PNGs are OFF by default: rendering one matplotlib figure per
# route is slow and stalls headless/orchestrator runs (the script appeared to
# "get stuck"). Flip MAKE_PLOTS to True (or pass --plots) for the maps; the CSV
# summaries are always written either way.
MAKE_PLOTS = False  # True -> also write a per-route buffer PNG
PLOT_FIG_DPI = 250  # resolution for PNG exports (only used when MAKE_PLOTS)

# Projected CRS used for buffering and spatial joins.
# EPSG:3857 (Web Mercator) works globally; swap for a local CRS (e.g. "EPSG:2283"
# for northern Virginia in feet) when higher spatial accuracy is needed.
PROJECTED_CRS = "EPSG:3857"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def _load_gtfs_tables(gtfs_dir: Path) -> Mapping[str, pd.DataFrame]:
    """Load GTFS text files into pandas DataFrames.

    Args:
        gtfs_dir: Directory containing GTFS .txt files.

    Returns:
        Mapping keyed by table name (without .txt) to DataFrame.
    """
    tables = {}
    for fn in ["routes", "trips", "stop_times", "stops", "shapes"]:
        path = gtfs_dir / f"{fn}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[fn] = pd.read_csv(path)
        logging.debug("Loaded %s (%d rows)", fn, len(tables[fn]))
    return tables


def _prepare_route_buffers(
    tables: Mapping[str, pd.DataFrame],
    use_shape_buffer: bool,
    buffer_dist_ft: float,
    route_filter: list[str] | None = None,
    projected_crs: str = "EPSG:3857",
    simplify_tolerance_m: float = SIMPLIFY_TOLERANCE_M,
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with one buffered geometry per route_id.

    Depending on *use_shape_buffer*, the buffer is built around the union of
    (a) the route's shape(s) or (b) all its stops.

    The returned GDF is in the CRS of the original GTFS shapes; if that CRS
    uses meters, the function converts *buffer_dist_ft* accordingly.

    Args:
        tables: GTFS tables keyed by name (needs ``shapes``, ``trips``,
            ``stops``, and ``stop_times`` for stop-buffer mode).
        use_shape_buffer: Buffer the route's shape geometry when True, else the
            union of its stops.
        buffer_dist_ft: Buffer distance in feet (converted to meters internally).
        route_filter: Only build buffers for these route_ids; empty/None means all.
        projected_crs: CRS used for buffering and the returned geometries.
        simplify_tolerance_m: Douglas-Peucker tolerance (in the projected CRS's
            units, meters for the default EPSG:3857) applied to each route's
            geometry before buffering. Buffering full-resolution shapes is the
            slowest step; simplifying first cuts that cost many-fold with no
            meaningful effect on a ¼-mile buffer. Pass 0.0 to disable.

    Raises:
        ValueError: If shapes.txt lacks an EPSG code in the header.
    """
    # Load shapes as GeoSeries
    shapes_df = tables["shapes"]
    if {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.difference(
        shapes_df.columns
    ):
        raise ValueError("shapes.txt missing required columns")

    # Convert shape points to LineStrings. This is a per-shape, Python-level
    # build over every row of shapes.txt, so on a large regional feed it is the
    # slowest step here — log before/after (and the row/shape counts) so a long
    # run is visibly progressing rather than appearing frozen.
    logging.info(
        "Building shape geometries from %d shape point(s) across %d shape_id(s)...",
        len(shapes_df),
        shapes_df["shape_id"].nunique(),
    )
    shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])

    geom_by_shape: dict[object, LineString] = {}
    skipped_degenerate = 0
    for shape_id, grp in shapes_df.groupby("shape_id", sort=False):
        coords = grp[["shape_pt_lon", "shape_pt_lat"]].to_numpy(dtype=float)
        # LineString needs >= 2 distinct points; a 1-point (or empty) shape
        # otherwise raises and would abort the whole run.
        if len(coords) < 2:
            skipped_degenerate += 1
            continue
        geom_by_shape[shape_id] = LineString(coords)

    if skipped_degenerate:
        logging.warning(
            "Skipped %d shape_id(s) with fewer than 2 points (cannot form a line).",
            skipped_degenerate,
        )

    shapes_gdf = gpd.GeoDataFrame(
        {"geometry": list(geom_by_shape.values())},
        index=pd.Index(list(geom_by_shape.keys()), name="shape_id"),
        geometry="geometry",
        crs="EPSG:4326",
    )
    logging.info("Built %d shape geometries; reprojecting to %s...", len(shapes_gdf), projected_crs)

    # Trips to route mapping
    trips = tables["trips"][["route_id", "trip_id", "shape_id"]]
    route_shapes = (
        trips.drop_duplicates(subset=["route_id", "shape_id"])
        .groupby("route_id")["shape_id"]
        .apply(list)
    )

    # Stops GeoDataFrame
    stops = tables["stops"][["stop_id", "stop_lat", "stop_lon"]].copy()
    stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    )

    # Choose projected CRS (meters) to allow buffering
    shapes_gdf = shapes_gdf.to_crs(projected_crs)
    stops = stops.to_crs(projected_crs)
    buff_dist_m = buffer_dist_ft * 0.3048  # convert to meters

    selected_routes = [rid for rid in route_shapes.index if not route_filter or rid in route_filter]
    logging.info(
        "Buffering %d route(s) (buffer=%.1f ft -> %.1f m, simplify=%.1f m)...",
        len(selected_routes),
        buffer_dist_ft,
        buff_dist_m,
        simplify_tolerance_m,
    )
    log_every = max(1, len(selected_routes) // 20)  # ~20 progress lines

    buffers: List[dict[str, object]] = []
    for processed, route_id in enumerate(selected_routes, start=1):
        shp_ids = route_shapes.loc[route_id]

        if use_shape_buffer:
            # Drop any shape_ids the route references but that produced no
            # geometry (missing from shapes.txt or skipped as degenerate).
            present = [s for s in shp_ids if s in shapes_gdf.index]
            geoms = shapes_gdf.loc[present, "geometry"]
        else:
            trip_stops = (
                tables["stop_times"]
                .merge(
                    trips[trips.route_id == route_id][["trip_id"]],
                    on="trip_id",
                    how="inner",
                )["stop_id"]
                .unique()
            )
            geoms = stops[stops.stop_id.isin(trip_stops)].geometry

        if geoms.empty:
            logging.warning("No geometry for route %s – skipped", route_id)
            continue

        geom = unary_union(list(geoms))
        # Thin the vertex count before buffering — full-resolution shapes make
        # buffer() the dominant cost. The tolerance is tiny next to the buffer.
        if simplify_tolerance_m > 0:
            geom = geom.simplify(simplify_tolerance_m)
        buf = geom.buffer(buff_dist_m)
        buffers.append({"route_id": route_id, "geometry": buf})

        if processed % log_every == 0 or processed == len(selected_routes):
            logging.info("  buffered %d/%d route(s)", processed, len(selected_routes))

    buffer_gdf = gpd.GeoDataFrame(buffers, geometry="geometry", crs=projected_crs)
    logging.info("Built %d route buffer(s).", len(buffer_gdf))
    return buffer_gdf


def _load_layers(
    layer_specs: Iterable[tuple[str, str]],
    shp_dir: Path,
    projected_crs: str = "EPSG:3857",
) -> dict[str, gpd.GeoDataFrame]:
    """Recursively load each designated shapefile (case‑insensitive search).

    The search now walks *all* subfolders under *shp_dir* using ``Path.rglob``.
    If multiple copies of the same filename are discovered, the first match in
    lexicographic order is used and a warning is logged.

    Args:
        layer_specs: Tuples of (filename, id_column).
        shp_dir: Root directory to search.
        projected_crs: CRS string used to reproject each loaded layer.

    Returns:
    -------
    dict[str, gpd.GeoDataFrame]
        Mapping of the *original* filename to its loaded GeoDataFrame.
    """
    layers: dict[str, gpd.GeoDataFrame] = {}

    for filename, id_col in layer_specs:
        # Case‑insensitive recursive search for the .shp
        matches = sorted(p for p in shp_dir.rglob("*.shp") if p.name.lower() == filename.lower())

        if not matches:
            logging.warning("Layer %s NOT FOUND anywhere under %s", filename, shp_dir)
            continue
        if len(matches) > 1:
            logging.warning("Multiple copies of %s found; using %s", filename, matches[0])

        path = matches[0]

        try:
            gdf = gpd.read_file(path)
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to read %s – %s", path, exc)
            continue

        if id_col not in gdf.columns:
            # List the layer's actual attribute columns so the configured
            # id_col can be corrected without opening the shapefile by hand.
            available = [c for c in gdf.columns if c != "geometry"]
            logging.warning(
                "Column '%s' missing in %s – skipped. Available columns: %s",
                id_col,
                path,
                available,
            )
            continue

        layers[filename] = gdf[[id_col, "geometry"]].to_crs(projected_crs)
        logging.info("Loaded %s (%d features)", path.relative_to(shp_dir), len(gdf))

    return layers


def _count_features(
    route_buffers: gpd.GeoDataFrame,
    layers: Mapping[str, gpd.GeoDataFrame],
    layer_specs: Iterable[tuple[str, str]],
    output_dir: Path,
    plot_fig_dpi: int = 250,
    make_plots: bool = False,
) -> pd.DataFrame:
    """For each route, count intersecting features and write a per-route CSV.

    A per-route map PNG is written too, but only when *make_plots* is True. The
    plot is opt-in because rendering one figure per route is slow enough to
    stall headless/orchestrator runs; the CSV summaries are always written.

    Returns:
        A summary DataFrame indexed by route_id with feature counts.
    """
    # Import pyplot lazily with a non-interactive backend so a headless run that
    # only wants the CSVs never touches a GUI backend (a common hang source).
    plt = None
    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    summary_records: list[dict[str, object]] = []

    for _, route_row in route_buffers.iterrows():
        route_id = route_row.route_id
        buf_geom = route_row.geometry

        per_route_counts: dict[str, object] = {"route_id": route_id}
        feature_name_lists: dict[str, List[str]] = {}

        for filename, id_col in layer_specs:
            if filename not in layers:
                continue
            layer_gdf = layers[filename]
            hits = layer_gdf[layer_gdf.intersects(buf_geom)]
            per_route_counts[filename] = len(hits)
            feature_name_lists[filename] = hits[id_col].astype(str).tolist()

        # Save per‑route CSV
        csv_rows = [
            {
                "layer": fname,
                "count": per_route_counts.get(fname, 0),
                "names": ", ".join(feature_name_lists.get(fname, [])),
            }
            for fname, _ in layer_specs
            if fname in layers
        ]
        pd.DataFrame(csv_rows).to_csv(output_dir / f"{route_id}_feature_summary.csv", index=False)

        # Plot quick map (opt-in; see make_plots above).
        if make_plots and plt is not None:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=plot_fig_dpi)
            gpd.GeoSeries([buf_geom]).plot(ax=ax, facecolor="none", edgecolor="black")
            for fname in feature_name_lists:
                layers[fname][layers[fname].intersects(buf_geom)].plot(
                    ax=ax, label=fname.split(".")[0]
                )
            ax.set_title(f"Route {route_id} buffer & intersecting features")
            ax.axis("off")
            ax.legend()
            fig_path = output_dir / f"{route_id}_buffer_plot.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)

        summary_records.append(per_route_counts)
        logging.info("Processed route %s - CSV written%s", route_id, " & PNG" if make_plots else "")

    summary_df = pd.DataFrame(summary_records).set_index("route_id").fillna(0).astype(int)
    return summary_df


# =============================================================================
# MAIN
# =============================================================================


def run(
    gtfs_dir: str | Path | None = None,
    shp_input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_shape_buffer: bool | None = None,
    buffer_dist_ft: float | None = None,
    route_filter: Sequence[str] | None = None,
    projected_crs: str | None = None,
    plot_fig_dpi: int | None = None,
    make_plots: bool | None = None,
    simplify_tolerance_m: float | None = None,
) -> None:
    """Run the GTFS feature-coverage analysis."""
    gtfs_dir = Path(GTFS_DIR if gtfs_dir is None else gtfs_dir)
    shp_input_dir = Path(SHP_INPUT_DIR if shp_input_dir is None else shp_input_dir)
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    use_shape_buffer = USE_SHAPE_BUFFER if use_shape_buffer is None else use_shape_buffer
    buffer_dist_ft = BUFFER_DIST_FT if buffer_dist_ft is None else buffer_dist_ft
    route_filter = list(ROUTE_FILTER if route_filter is None else route_filter)
    projected_crs = PROJECTED_CRS if projected_crs is None else projected_crs
    plot_fig_dpi = PLOT_FIG_DPI if plot_fig_dpi is None else plot_fig_dpi
    make_plots = MAKE_PLOTS if make_plots is None else make_plots
    simplify_tolerance_m = (
        SIMPLIFY_TOLERANCE_M if simplify_tolerance_m is None else simplify_tolerance_m
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading GTFS from %s", gtfs_dir)
    tables = _load_gtfs_tables(gtfs_dir)

    logging.info("Building route buffers (use_shape_buffer=%s)", use_shape_buffer)
    route_buffers = _prepare_route_buffers(
        tables,
        use_shape_buffer,
        buffer_dist_ft,
        route_filter=route_filter,
        projected_crs=projected_crs,
        simplify_tolerance_m=simplify_tolerance_m,
    )

    if route_buffers.empty:
        logging.error("No buffers produced – nothing to do")
        return

    logging.info("Loading designated shapefiles")
    layers = _load_layers(LAYER_SPECS, shp_input_dir, projected_crs=projected_crs)

    if not layers:
        logging.error("No valid layers loaded – nothing to analyze")
        return

    logging.info("Counting features per route")
    summary_df = _count_features(
        route_buffers,
        layers,
        LAYER_SPECS,
        output_dir,
        plot_fig_dpi=plot_fig_dpi,
        make_plots=make_plots,
    )

    # Save summary CSV
    summary_path = output_dir / "all_routes_feature_summary.csv"
    summary_df.to_csv(summary_path)
    logging.info("Summary written to %s", summary_path)
    logging.info("Done.")
    logging.info("Script completed successfully.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze transit route coverage of strategic sites. Defaults come from "
            "the CONFIGURATION block at the top of this file; the LAYER_SPECS list "
            "stays in the config block."
        )
    )
    parser.add_argument(
        "--gtfs-dir", type=Path, default=GTFS_DIR, help="Folder containing GTFS .txt files."
    )
    parser.add_argument(
        "--shp-input-dir",
        type=Path,
        default=SHP_INPUT_DIR,
        help="Folder with .shp layers to test.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Where CSVs and PNGs are written."
    )
    parser.add_argument(
        "--buffer-ft", type=float, default=BUFFER_DIST_FT, help="Buffer distance in feet."
    )
    parser.add_argument(
        "--buffer-stops",
        dest="use_shape_buffer",
        action="store_false",
        default=USE_SHAPE_BUFFER,
        help="Buffer stops instead of route geometry.",
    )
    parser.add_argument(
        "--routes",
        nargs="*",
        default=ROUTE_FILTER,
        metavar="ROUTE_ID",
        help="Only analyze these route_id values (default: all).",
    )
    parser.add_argument(
        "--projected-crs", default=PROJECTED_CRS, help="Projected CRS for buffering/joins."
    )
    parser.add_argument("--dpi", type=int, default=PLOT_FIG_DPI, help="Resolution for PNG exports.")
    parser.add_argument(
        "--simplify-m",
        dest="simplify_tolerance_m",
        type=float,
        default=SIMPLIFY_TOLERANCE_M,
        help="Simplify route geometry by this tolerance (projected units) before buffering; "
        "0 disables.",
    )
    parser.add_argument(
        "--plots",
        dest="make_plots",
        action="store_true",
        default=MAKE_PLOTS,
        help="Also render a per-route map PNG (slow; off by default).",
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
        shp_input_dir=args.shp_input_dir,
        output_dir=args.output_dir,
        use_shape_buffer=args.use_shape_buffer,
        buffer_dist_ft=args.buffer_ft,
        route_filter=args.routes,
        projected_crs=args.projected_crs,
        plot_fig_dpi=args.dpi,
        make_plots=args.make_plots,
        simplify_tolerance_m=args.simplify_tolerance_m,
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
