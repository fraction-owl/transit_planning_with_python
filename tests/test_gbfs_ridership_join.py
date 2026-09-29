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
    """A tiny three-station layer (one station has no ridership).

    Mirrors Capital Bikeshare GBFS: ``station_id`` is an opaque id and the trip
    history station number lives in ``short_name``.
    """
    return gpd.GeoDataFrame(
        {
            "station_id": ["08263b1f", "08253c42", "0825f1b3"],
            "short_name": ["1", "2", "3"],
            "name": ["Dupont", "Eastern", "Quiet"],
        },
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
    by_id = joined.set_index("short_name")
    assert by_id.loc["1", "total"] == 30
    # Station 3 had no ridership and is zero-filled rather than dropped.
    assert by_id.loc["3", "total"] == 0
    assert by_id.loc["3", "departures"] == 0


def test_join_missing_id_raises(stations: gpd.GeoDataFrame) -> None:
    """A station layer without the join column is rejected."""
    no_id = stations.drop(columns=["short_name"])
    with pytest.raises(KeyError):
        mod.join_ridership(no_id, pd.DataFrame({"station_id": ["1"]}))


def test_load_station_ridership_from_csv(tmp_path: Path, monthly_ridership: pd.DataFrame) -> None:
    """Loading a CSV returns aggregated per-station totals."""
    csv_path = tmp_path / "monthly_station_ridership.csv"
    monthly_ridership.to_csv(csv_path, index=False)
    totals = mod.load_station_ridership(csv_path)
    assert len(totals) == 2
    assert totals.set_index("station_id").loc["1", "departures"] == 15


def test_join_on_gbfs_station_id_is_rejected(
    stations: gpd.GeoDataFrame, monthly_ridership: pd.DataFrame
) -> None:
    """Joining trip station numbers to GBFS station_id matches nothing and fails loudly."""
    totals = mod.aggregate_station_totals(monthly_ridership)
    with pytest.raises(ValueError, match="short_name"):
        mod.join_ridership(stations, totals, geometry_id_field="station_id")


def test_join_logs_unmatched_ridership_ids(
    stations: gpd.GeoDataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Ridership ids with no station feature are reported, not silently dropped."""
    totals = pd.DataFrame({"station_id": ["1", "99"], "total": [5, 7]})
    with caplog.at_level("WARNING"):
        joined = mod.join_ridership(stations, totals)
    assert joined["total"].sum() == 5
    assert "99" in caplog.text
    assert "7 total trips" in caplog.text


def test_load_preserves_string_ids(tmp_path: Path) -> None:
    """Ids like "001" and "NA" survive loading and join to matching stations."""
    csv_path = tmp_path / "monthly_station_ridership.csv"
    csv_path.write_text(
        "month,station_id,station_name,departures,arrivals,total\n"
        "2024-05,001,Zero One,1,2,3\n"
        "2024-05,NA,North Ave,4,5,9\n"
        "2024-06,NA,North Ave,1,0,1\n",
        encoding="utf-8",
    )
    totals = mod.load_station_ridership(csv_path)
    assert sorted(totals["station_id"]) == ["001", "NA"]
    stations = gpd.GeoDataFrame(
        {"short_name": ["001", "NA"]},
        geometry=[Point(-77.0, 38.9), Point(-77.1, 38.8)],
        crs="EPSG:4326",
    )
    joined = mod.join_ridership(stations, totals).set_index("short_name")
    assert joined.loc["001", "total"] == 3
    assert joined.loc["NA", "total"] == 10


def test_join_keeps_distinct_string_ids_distinct(stations: gpd.GeoDataFrame) -> None:
    """Ids "1" and "1.0" are different and must not merge into one key."""
    totals = pd.DataFrame({"station_id": ["1", "1.0"], "total": [5, 7]})
    joined = mod.join_ridership(stations, totals)
    assert len(joined) == len(stations)
    assert joined.set_index("short_name").loc["1", "total"] == 5


def test_join_rejects_ids_that_collide_after_normalization(
    stations: gpd.GeoDataFrame,
) -> None:
    """Different raw ids normalizing to one key raise instead of duplicating features."""
    totals = pd.DataFrame({"station_id": ["1", " 1"], "total": [5, 7]})
    with pytest.raises(ValueError, match="same join key"):
        mod.join_ridership(stations, totals)


def test_join_rejects_duplicate_geometry_keys(stations: gpd.GeoDataFrame) -> None:
    """Two features sharing a key would double-count ridership on the map."""
    stations.loc[2, "short_name"] = "1"
    totals = pd.DataFrame({"station_id": ["1"], "total": [5]})
    with pytest.raises(ValueError, match="duplicate"):
        mod.join_ridership(stations, totals)


def test_aggregate_keeps_first_nonblank_name() -> None:
    """A blank name in an early month does not hide a later real name."""
    ridership = pd.DataFrame(
        {"station_id": ["1", "1"], "station_name": ["", "Dupont"], "total": [1, 2]}
    )
    totals = mod.aggregate_station_totals(ridership)
    assert totals.loc[0, "station_name"] == "Dupont"


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
