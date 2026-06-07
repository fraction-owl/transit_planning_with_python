"""Count community land-use POIs within walking distance of each transit stop.

For every stop in a GTFS feed, this script buffers the stop by a walk-access
distance (default ¼ mile) and counts how many point-of-interest (POI) features
of each category fall inside that buffer. The result is a stop-level table with
one count column per POI category — a feature set for direct-ridership /
demand modeling that captures the *non-employment, community* trip generators a
pop-plus-jobs model (e.g., TBEST's socioeconomic engine) systematically
underweights: houses of worship, schools, hospitals, civic facilities, etc.

This is the stop-level companion to ``route_site_coverage_gpd.py`` (which buffers
whole *routes*). Counting — rather than a binary present/absent flag — keeps the
signal proportional to how many generators a stop actually serves.

Intended Use
------------
Feeds the "Model A" (service-planning / stop-analysis) feature table. Start with
raw counts; a later revision can graduate any category from a count to an
ordinal intensity tier (small/medium/large) without reshaping the output.

Assumptions
-----------
- POI layers are point (or polygon) vector files readable by GeoPandas
  (Shapefile, GeoJSON, GeoPackage, ...). Polygons are tested by intersection.
- ``PROJECTED_CRS`` is a projected system in **US feet** so the buffer distance
  is applied directly in feet (default: EPSG:2248, NAD83 / Maryland ftUS).
- Each POI layer carries a readable name column (used only for validation /
  logging, not for the counts).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Sentinel markers used by extract_config_block / write_run_log to identify
# the configuration block when generating the run-log sidecar.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Top-level directories.
GTFS_DIR = Path(r"tests/fixtures/gtfs_basic")  # folder containing stops.txt
POI_INPUT_DIR = Path(r"tests/fixtures/poi_sample")  # folder of POI vector files
OUTPUT_DIR = Path(r"output")  # where the CSV + run log are written
OUTPUT_CSV_NAME = "stop_poi_counts.csv"

# Column in stops.txt that uniquely identifies a stop.
STOP_ID_COL = "stop_id"

# Each entry is ``(filename, id_column, category)``:
#   filename   – searched (case-insensitive, recursively) under POI_INPUT_DIR
#   id_column  – a readable name column used for validation/logging
#   category   – the output count-column name
# The first eight are the committed categories; the last three (marked
# RECOMMENDED) are high-value transit-dependent generators — delete any you
# don't have data for, the script just skips missing layers with a warning.
LAYER_SPECS: list[tuple[str, str, str]] = [
    ("houses_of_worship.geojson", "name", "houses_of_worship"),
    ("secondary_schools.geojson", "name", "secondary_schools"),
    ("universities.geojson", "name", "universities"),
    ("hospitals.geojson", "name", "hospitals"),
    ("stadiums.geojson", "name", "stadiums"),
    ("libraries.geojson", "name", "libraries"),
    ("rec_centers.geojson", "name", "rec_centers"),
    ("community_centers.geojson", "name", "community_centers"),
    ("grocery_stores.geojson", "name", "grocery_stores"),  # RECOMMENDED
    ("social_service_offices.geojson", "name", "social_service_offices"),  # RECOMMENDED
    ("community_colleges.geojson", "name", "community_colleges"),  # RECOMMENDED
]

# Walk-access buffer applied to each stop, in feet (1320 ft = ¼ mile).
BUFFER_DIST_FT: float = 1320.0

# Coordinate reference systems.
GTFS_CRS = "EPSG:4326"  # lat/lon as published in GTFS
PROJECTED_CRS = "EPSG:2248"  # NAD83 / Maryland (ftUS) — buffer is applied in feet

# A run-log sidecar is a required deliverable. Leave True unless writing to a
# genuinely read-only location.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_gtfs_stops(gtfs_dir: Path, stop_id_col: str, gtfs_crs: str) -> gpd.GeoDataFrame:
    """Load ``stops.txt`` into a GeoDataFrame of stop points.

    Args:
        gtfs_dir: Directory containing the GTFS ``stops.txt`` file.
        stop_id_col: Name of the unique stop-identifier column in ``stops.txt``.
        gtfs_crs: CRS of the published lat/lon coordinates (typically EPSG:4326).

    Returns:
        A GeoDataFrame with a normalized ``stop_id`` column, any ``stop_name``,
        the original ``stop_lat``/``stop_lon`` columns, and point geometry.

    Raises:
        FileNotFoundError: If ``stops.txt`` is not present in *gtfs_dir*.
        ValueError: If required columns are missing.
    """
    path = gtfs_dir / "stops.txt"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {stop_id_col, "stop_lat", "stop_lon"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"stops.txt missing required columns: {sorted(missing)}")

    keep = [stop_id_col, "stop_lat", "stop_lon"]
    if "stop_name" in df.columns:
        keep.insert(1, "stop_name")
    stops = df[keep].rename(columns={stop_id_col: "stop_id"})

    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs=gtfs_crs,
    )
    logging.info("Loaded %d stops from %s", len(gdf), path)
    return gdf


def build_stop_buffers(
    stops_gdf: gpd.GeoDataFrame,
    buffer_dist_ft: float,
    projected_crs: str,
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of one buffered polygon per stop.

    The stops are reprojected to *projected_crs* (assumed to use US feet) and
    each point is buffered by *buffer_dist_ft*, applied directly in feet.

    Args:
        stops_gdf: Stop points with a ``stop_id`` column and a defined CRS.
        buffer_dist_ft: Buffer radius in feet.
        projected_crs: A projected CRS in US feet used for buffering.

    Returns:
        A GeoDataFrame with columns ``stop_id`` and buffered ``geometry`` in
        *projected_crs*.
    """
    projected = stops_gdf.to_crs(projected_crs)
    buffers = projected[["stop_id"]].copy()
    buffers = gpd.GeoDataFrame(
        buffers,
        geometry=projected.geometry.buffer(buffer_dist_ft),
        crs=projected_crs,
    )
    return buffers


def load_poi_layers(
    layer_specs: Sequence[tuple[str, str, str]],
    poi_dir: Path,
    projected_crs: str,
) -> dict[str, gpd.GeoDataFrame]:
    """Load each POI layer found under *poi_dir*, keyed by category.

    Filenames are matched case-insensitively and searched recursively. Layers
    that are missing, unreadable, or lack their declared id column are skipped
    with a warning so a partial POI catalog still produces output.

    Args:
        layer_specs: Tuples of ``(filename, id_column, category)``.
        poi_dir: Root directory to search for POI vector files.
        projected_crs: CRS to reproject every layer into (matches the buffers).

    Returns:
        Mapping of category name to its loaded GeoDataFrame.
    """
    layers: dict[str, gpd.GeoDataFrame] = {}

    for filename, id_col, category in layer_specs:
        matches = sorted(p for p in poi_dir.rglob("*") if p.name.lower() == filename.lower())
        if not matches:
            logging.warning("Layer %s NOT FOUND under %s — skipped", filename, poi_dir)
            continue
        if len(matches) > 1:
            logging.warning("Multiple copies of %s found; using %s", filename, matches[0])

        path = matches[0]
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:  # noqa: BLE001 — report any driver error and skip
            logging.warning("Failed to read %s — %s", path, exc)
            continue

        if id_col not in gdf.columns:
            logging.warning("Column '%s' missing in %s — skipped", id_col, path)
            continue

        layers[category] = gdf[[id_col, "geometry"]].to_crs(projected_crs)
        logging.info("Loaded %s (%d features) as '%s'", path.name, len(gdf), category)

    return layers


def count_pois_within_buffers(
    stop_buffers: gpd.GeoDataFrame,
    layers: Mapping[str, gpd.GeoDataFrame],
    categories: Sequence[str],
) -> pd.DataFrame:
    """Count POIs of each category intersecting each stop buffer.

    Args:
        stop_buffers: One buffered polygon per stop, with a ``stop_id`` column.
        layers: Mapping of category name to its POI GeoDataFrame (same CRS as
            *stop_buffers*).
        categories: Ordered category names. A category with no loaded layer
            yields a column of zeros, so the output schema is stable.

    Returns:
        A DataFrame with ``stop_id``, one integer count column per category
        (in *categories* order), and a ``poi_total`` column.
    """
    result = stop_buffers[["stop_id"]].copy()

    for category in categories:
        layer = layers.get(category)
        if layer is None or layer.empty:
            result[category] = 0
            continue

        joined = gpd.sjoin(
            layer[["geometry"]],
            stop_buffers[["stop_id", "geometry"]],
            predicate="intersects",
            how="inner",
        )
        per_stop = joined.groupby("stop_id").size()
        result[category] = result["stop_id"].map(per_stop).fillna(0).astype(int)

    result["poi_total"] = result[list(categories)].sum(axis=1).astype(int)
    return result


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


def write_run_log(output_folder: Path) -> bool:
    """Write a run log of the configuration block into *output_folder*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_folder / "stop_poi_coverage_gpd_runlog.txt"

    try:
        config_text: str = extract_config_block(Path(__file__))
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: list[str] = [
        "=" * 72,
        "STOP POI COVERAGE (GPD) RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output folder:    {output_folder}",
        f"Source script:    {Path(__file__).resolve()}",
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
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
    """Run the stop-level POI coverage count."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stops_gdf = load_gtfs_stops(GTFS_DIR, STOP_ID_COL, GTFS_CRS)
    stop_buffers = build_stop_buffers(stops_gdf, BUFFER_DIST_FT, PROJECTED_CRS)

    layers = load_poi_layers(LAYER_SPECS, POI_INPUT_DIR, PROJECTED_CRS)
    if not layers:
        logging.error("No POI layers loaded from %s — nothing to count.", POI_INPUT_DIR)
        return

    categories = [category for _, _, category in LAYER_SPECS]
    counts = count_pois_within_buffers(stop_buffers, layers, categories)

    # Attach readable stop attributes for the output table.
    attr_cols = [c for c in ("stop_id", "stop_name", "stop_lat", "stop_lon") if c in stops_gdf]
    output = pd.DataFrame(stops_gdf[attr_cols]).merge(counts, on="stop_id", how="left")

    output_path = OUTPUT_DIR / OUTPUT_CSV_NAME
    output.to_csv(output_path, index=False)
    logging.info("Wrote stop-level POI counts for %d stops to %s", len(output), output_path)

    if not write_run_log(OUTPUT_DIR) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "allow output without a run log (not recommended)."
        )

    logging.info("Done.")
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
