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
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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


def main() -> None:
    """Run the ridership-to-geometry join end to end."""
    if not GEOJSON_INPUT and not SHAPEFILE_INPUT:
        raise ValueError("Set GEOJSON_INPUT and/or SHAPEFILE_INPUT in the CONFIG block.")
    ridership = load_station_ridership(RIDERSHIP_INPUT, STATION_ID_FIELD)
    output_dir = Path(OUTPUT_DIR)
    for geometry_input in (GEOJSON_INPUT, SHAPEFILE_INPUT):
        if not geometry_input:
            continue
        stations = gpd.read_file(geometry_input)
        joined = join_ridership(stations, ridership, STATION_ID_FIELD)
        output_path = _joined_output_path(geometry_input, output_dir)
        export_layer(joined, output_path)
        logger.info("Joined ridership onto %d stations -> %s", len(joined), output_path)


if __name__ == "__main__":
    main()
