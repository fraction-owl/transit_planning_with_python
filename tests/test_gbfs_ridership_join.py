"""Tests for gbfs_ridership_join."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from scripts.gbfs_tools import gbfs_ridership_join as mod


@pytest.fixture()
def stations() -> gpd.GeoDataFrame:
    """A tiny three-station layer (one station has no ridership)."""
    return gpd.GeoDataFrame(
        {"station_id": ["1", "2", "3"], "name": ["Dupont", "Eastern", "Quiet"]},
        geometry=[Point(-77.0, 38.9), Point(-76.99, 38.88), Point(-77.01, 38.91)],
        crs="EPSG:4326",
    )


@pytest.fixture()
def monthly_ridership() -> pd.DataFrame:
    """Two months of ridership for two of the three stations."""
    return pd.DataFrame(
        {
            "month": ["2024-05", "2024-06", "2024-05"],
            # ids parsed from CSV often arrive numeric; exercise the cast.
            "station_id": [1, 1, 2],
            "station_name": ["Dupont", "Dupont", "Eastern"],
            "departures": [10, 5, 4],
            "arrivals": [8, 7, 5],
            "total": [18, 12, 9],
        }
    )


def test_aggregate_sums_months_per_station(monthly_ridership: pd.DataFrame) -> None:
    """Per-month rows collapse to one summed row per station."""
    totals = mod.aggregate_station_totals(monthly_ridership)
    by_id = totals.set_index("station_id")
    assert len(totals) == 2
    assert by_id.loc[1, "departures"] == 15
    assert by_id.loc[1, "total"] == 30
    assert by_id.loc[2, "arrivals"] == 5


def test_aggregate_missing_id_raises() -> None:
    """A ridership table without the join column is rejected."""
    with pytest.raises(KeyError):
        mod.aggregate_station_totals(pd.DataFrame({"departures": [1]}))


def test_join_keeps_all_stations_and_zero_fills(
    stations: gpd.GeoDataFrame, monthly_ridership: pd.DataFrame
) -> None:
    """Every station is retained; unmatched stations get zero measures."""
    totals = mod.aggregate_station_totals(monthly_ridership)
    joined = mod.join_ridership(stations, totals)
    assert len(joined) == 3
    assert joined.crs == stations.crs
    by_id = joined.set_index("station_id")
    assert by_id.loc["1", "total"] == 30
    # Station 3 had no ridership and is zero-filled rather than dropped.
    assert by_id.loc["3", "total"] == 0
    assert by_id.loc["3", "departures"] == 0


def test_join_missing_id_raises(stations: gpd.GeoDataFrame) -> None:
    """A station layer without the join column is rejected."""
    no_id = stations.drop(columns=["station_id"])
    with pytest.raises(KeyError):
        mod.join_ridership(no_id, pd.DataFrame({"station_id": ["1"]}))


def test_load_station_ridership_from_csv(tmp_path: Path, monthly_ridership: pd.DataFrame) -> None:
    """Loading a CSV returns aggregated per-station totals."""
    csv_path = tmp_path / "monthly_station_ridership.csv"
    monthly_ridership.to_csv(csv_path, index=False)
    totals = mod.load_station_ridership(csv_path)
    assert len(totals) == 2
    assert totals.set_index("station_id").loc[1, "departures"] == 15


def test_export_layer_writes_geojson_and_shapefile(
    tmp_path: Path, stations: gpd.GeoDataFrame, monthly_ridership: pd.DataFrame
) -> None:
    """Both output formats are written and round-trip with ridership columns."""
    totals = mod.aggregate_station_totals(monthly_ridership)
    joined = mod.join_ridership(stations, totals)

    geojson_path = tmp_path / "gbfs_stations_ridership.geojson"
    shp_path = tmp_path / "gbfs_stations_ridership.shp"
    mod.export_layer(joined, geojson_path)
    mod.export_layer(joined, shp_path)

    assert geojson_path.exists()
    assert shp_path.exists()
    reloaded = gpd.read_file(geojson_path)
    assert "total" in reloaded.columns
    assert len(reloaded) == 3


def test_joined_output_path_naming() -> None:
    """Output paths append a ``_ridership`` suffix in the output dir."""
    out = mod._joined_output_path("output/gbfs_stations.geojson", Path("dest"))
    assert out == Path("dest/gbfs_stations_ridership.geojson")
