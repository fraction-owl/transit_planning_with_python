"""Join Capital Bikeshare ridership totals onto station geometries for mapping.

This script bridges the two other tools in ``gbfs_tools``:

* :mod:`gbfs_stations_exporter` writes station point geometries
  (``gbfs_stations.geojson`` and ``gbfs_stations.shp``).
* :mod:`bikeshare_ridership_trends` writes ridership summaries, including
  ``monthly_station_ridership.csv`` (one row per month and station).

It aggregates the per-month ridership to per-station totals and joins them onto
the station geometries, writing *new* GeoJSON and/or Shapefile versions that
carry departures, arrivals, and total trip attributes. Those enriched layers
are ready to drop into a GIS for ridership maps (proportional symbols,
choropleths, and so on).

Both inputs and outputs use EPSG:4326 (WGS 84), inherited from the source
geometries. The join key defaults to ``station_id``.

Typical usage (edit the CONFIG block, then run):

    python gbfs_ridership_join.py

Every CONFIG value also has a matching command-line flag that overrides it, e.g.

    python gbfs_ridership_join.py \
        --ridership-input output/monthly_station_ridership.csv \
        --geojson-input output/gbfs_stations.geojson --output-dir output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd

# === BEGIN CONFIG ===
RIDERSHIP_INPUT: str = "output/monthly_station_ridership.csv"
GEOJSON_INPUT: Optional[str] = "output/gbfs_stations.geojson"
SHAPEFILE_INPUT: Optional[str] = "output/gbfs_stations.shp"
OUTPUT_DIR: str = "output"
STATION_ID_FIELD: str = "station_id"
# === END CONFIG ===

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Ridership measures summed when collapsing months to per-station totals.
#: These match the count columns of ``monthly_station_ridership.csv``.
RIDERSHIP_MEASURES: tuple[str, ...] = ("departures", "arrivals", "total")


def _safe_to_str(value: object) -> str:
    """Return a trimmed string form of an identifier value.

    Args:
        value: Any identifier-like value.

    Returns:
        A stripped string, with trailing ``.0`` removed for float-like ids.
    """
    return str(value).strip().removesuffix(".0")


def aggregate_station_totals(
    ridership: pd.DataFrame, id_field: str = STATION_ID_FIELD
) -> pd.DataFrame:
    """Collapse per-month station ridership to per-station totals.

    Rows are grouped by ``id_field`` and the ridership measures
    (:data:`RIDERSHIP_MEASURES`) are summed. A ``station_name`` column, if
    present, is carried through using its first value per station. Input that is
    already aggregated (no ``month`` column) passes through unchanged in shape.

    Args:
        ridership: Station ridership table, e.g. ``monthly_station_ridership``.
        id_field: Column identifying the station.

    Returns:
        A DataFrame with one row per station and summed ridership measures.
    """
    if id_field not in ridership.columns:
        raise KeyError(
            f"Ridership table is missing the join column {id_field!r}; "
            f"found columns: {list(ridership.columns)}"
        )
    measures = [c for c in RIDERSHIP_MEASURES if c in ridership.columns]
    aggregations: dict[str, str] = {measure: "sum" for measure in measures}
    if "station_name" in ridership.columns:
        aggregations["station_name"] = "first"
    totals = ridership.groupby(id_field, as_index=False).agg(aggregations)
    for measure in measures:
        totals[measure] = totals[measure].fillna(0).astype(int)
    return totals


def load_station_ridership(path: str | Path, id_field: str = STATION_ID_FIELD) -> pd.DataFrame:
    """Load a station ridership CSV and aggregate it to per-station totals.

    Args:
        path: Path to ``monthly_station_ridership.csv`` (or an already
            aggregated station ridership CSV).
        id_field: Column identifying the station.

    Returns:
        A DataFrame with one row per station, ready to join onto geometries.
    """
    ridership = pd.read_csv(path)
    return aggregate_station_totals(ridership, id_field)


def join_ridership(
    stations: gpd.GeoDataFrame,
    ridership: pd.DataFrame,
    id_field: str = STATION_ID_FIELD,
) -> gpd.GeoDataFrame:
    """Join per-station ridership totals onto station geometries.

    The join key is normalized on both sides with :func:`_safe_to_str` so that
    string ids from GBFS match numeric-looking ids parsed from CSV. The merge is
    a left join: every station geometry is kept, and stations with no recorded
    ridership get zero-filled measures.

    Args:
        stations: Station point geometries (from ``gbfs_stations_exporter``).
        ridership: Per-station ridership totals.
        id_field: Column/property identifying the station on both sides.

    Returns:
        A GeoDataFrame of stations enriched with ridership attributes.
    """
    if id_field not in stations.columns:
        raise KeyError(
            f"Station layer is missing the join column {id_field!r}; "
            f"found columns: {list(stations.columns)}"
        )
    stations = stations.copy()
    ridership = ridership.copy()
    stations["_join_key"] = stations[id_field].map(_safe_to_str)
    ridership["_join_key"] = ridership[id_field].map(_safe_to_str)
    # Drop the duplicate id column from the right side; keep the geometry's.
    ridership = ridership.drop(columns=[id_field])
    # Let ridership win for any other overlapping non-key column (e.g. name).
    overlap = [c for c in ridership.columns if c != "_join_key" and c in stations.columns]
    stations = stations.drop(columns=overlap)
    merged = stations.merge(ridership, on="_join_key", how="left")
    merged = merged.drop(columns=["_join_key"])
    measures = [c for c in RIDERSHIP_MEASURES if c in merged.columns]
    for measure in measures:
        merged[measure] = merged[measure].fillna(0).astype(int)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=stations.crs)


def export_layer(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write a GeoDataFrame, choosing the driver from the file extension.

    Args:
        gdf: The enriched station GeoDataFrame.
        output_path: Destination ``.geojson`` or ``.shp`` path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON", index=False)
    else:
        gdf.to_file(output_path, driver="ESRI Shapefile", index=False)


def _joined_output_path(input_path: str, output_dir: Path) -> Path:
    """Return the ``*_ridership`` output path for a given geometry input.

    Args:
        input_path: Path to a source geometry file.
        output_dir: Directory for the joined output.

    Returns:
        ``<output_dir>/<stem>_ridership<suffix>``.
    """
    source = Path(input_path)
    return output_dir / f"{source.stem}_ridership{source.suffix}"


def run(
    ridership_input: str | None = None,
    geojson_input: str | None = None,
    shapefile_input: str | None = None,
    output_dir: str | Path | None = None,
    station_id_field: str | None = None,
) -> None:
    """Run the ridership-to-geometry join end to end.

    Unset args fall back to the CONFIG block at the top of this file, so
    ``m.RIDERSHIP_INPUT = ...; m.run()`` works after a plain import. Pass an
    empty string for ``geojson_input`` or ``shapefile_input`` to skip that
    output.
    """
    ridership_input = RIDERSHIP_INPUT if ridership_input is None else ridership_input
    geojson_input = GEOJSON_INPUT if geojson_input is None else geojson_input
    shapefile_input = SHAPEFILE_INPUT if shapefile_input is None else shapefile_input
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    station_id_field = STATION_ID_FIELD if station_id_field is None else station_id_field

    if not geojson_input and not shapefile_input:
        raise ValueError("Set GEOJSON_INPUT and/or SHAPEFILE_INPUT in the CONFIG block.")
    ridership = load_station_ridership(ridership_input, station_id_field)
    output_dir = Path(output_dir)
    for geometry_input in (geojson_input, shapefile_input):
        if not geometry_input:
            continue
        stations = gpd.read_file(geometry_input)
        joined = join_ridership(stations, ridership, station_id_field)
        output_path = _joined_output_path(geometry_input, output_dir)
        export_layer(joined, output_path)
        logger.info("Joined ridership onto %d stations -> %s", len(joined), output_path)
    logger.info("Script completed successfully.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIG block values."""
    parser = argparse.ArgumentParser(
        description=(
            "Join Capital Bikeshare ridership totals onto station geometries. "
            "Defaults come from the CONFIG block at the top of this file."
        )
    )
    parser.add_argument(
        "--ridership-input", default=RIDERSHIP_INPUT, help="Per-station ridership CSV."
    )
    parser.add_argument(
        "--geojson-input",
        default=GEOJSON_INPUT,
        help="Station GeoJSON to enrich (empty string to skip).",
    )
    parser.add_argument(
        "--shapefile-input",
        default=SHAPEFILE_INPUT,
        help="Station Shapefile to enrich (empty string to skip).",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for joined outputs.")
    parser.add_argument(
        "--station-id-field", default=STATION_ID_FIELD, help="Join column on both sides."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. Defaults fall back to the CONFIG block."""
    args = parse_args(argv)
    run(
        ridership_input=args.ridership_input,
        geojson_input=args.geojson_input,
        shapefile_input=args.shapefile_input,
        output_dir=args.output_dir,
        station_id_field=args.station_id_field,
    )


def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":
    # In a notebook (pasted cell or %run), use the CONFIG block instead of
    # argparse, which would otherwise try to parse the kernel's own argv.
    if _in_ipython():
        run()
    else:
        main()
