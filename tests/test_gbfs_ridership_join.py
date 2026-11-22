"""Tests for gbfs_ridership_join."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
from shapely.geometry import Point

matplotlib.use("Agg")  # headless: the pipeline test draws trend charts

from scripts.gbfs_tools import bikeshare_ridership_trends as trends
from scripts.gbfs_tools import gbfs_ridership_join as mod
from scripts.gbfs_tools import gbfs_stations_exporter as exporter


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


def test_joined_output_path_for_zipped_shapefile_is_shp() -> None:
    """A zipped shapefile input is written as a plain .shp, not a .zip."""
    out = mod._joined_output_path("input/Stations.ZIP", Path("dest"))
    assert out == Path("dest/Stations_ridership.shp")


def test_join_reports_activity_on_blank_ids(
    stations: gpd.GeoDataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Rows with a blank id are dropped, but their activity is logged."""
    totals = pd.DataFrame(
        {
            "station_id": ["1", ""],
            "departures": [3, 25],
            "arrivals": [2, 17],
            "total": [5, 42],
        }
    )
    with caplog.at_level("WARNING"):
        joined = mod.join_ridership(stations, totals)
    assert joined["total"].sum() == 5
    assert "1 ridership row(s) have a blank 'station_id'" in caplog.text
    assert "25 departures, 17 arrivals, 42 total trips" in caplog.text


def test_run_rejects_ridership_with_only_blank_ids(
    tmp_path: Path, stations: gpd.GeoDataFrame
) -> None:
    """A nonempty ridership table with no usable ids stops before writing a map."""
    geojson = tmp_path / "gbfs_stations.geojson"
    stations.to_file(geojson, driver="GeoJSON")
    csv_path = tmp_path / "monthly_station_ridership.csv"
    csv_path.write_text(
        "month,station_id,station_name,departures,arrivals,total\n2024-05,,Blank,25,17,42\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="blank 'station_id'"):
        mod.run(
            ridership_input=str(csv_path),
            geojson_input=str(geojson),
            shapefile_input="",
            output_dir=out_dir,
        )
    assert not (out_dir / "gbfs_stations_ridership.geojson").exists()


# ---------------------------------------------------------------------------
# End to end: GBFS export -> ridership trends -> join, checked on the saved map
# ---------------------------------------------------------------------------

_TRIP_HEADER = (
    "ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,"
    "end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual\n"
)


def test_pipeline_saved_maps_reconcile_with_trip_activity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Map activity plus activity reported as unmatched equals docked trip activity."""
    # Station layer, also zipped to exercise the zipped-shapefile input.
    gbfs_dir = tmp_path / "gbfs"
    gbfs_dir.mkdir()
    info = gbfs_dir / "station_information.json"
    stations = [
        {"station_id": "08263b1f", "short_name": "31000", "name": "Dupont"},
        {"station_id": "08253c42", "short_name": "31001", "name": "Eastern"},
        {"station_id": "0825f1b3", "short_name": "31002", "name": "Quiet"},
    ]
    for i, station in enumerate(stations):
        station.update(lat=38.90 + i / 100, lon=-77.0)
    info.write_text(json.dumps({"data": {"stations": stations}}), encoding="utf-8")
    exporter.gbfs_stations_to_files(str(info), gbfs_dir, "both")
    zipped = gbfs_dir / "gbfs_stations.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        for part in gbfs_dir.glob("gbfs_stations.*"):
            if part.suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                archive.write(part, part.name)

    # Trips as (month, start id, end id, rider, bike). Station 39999 has no
    # feature; a blank id is dockless; blank / docked_bike are "other".
    trips = [
        ("2026-01", "31000", "31001", "member", "classic_bike"),
        ("2026-01", "31001", "31000", "casual", "electric_bike"),
        ("2026-01", "31000", "39999", "member", "electric_bike"),
        ("2026-01", "", "31001", "casual", "electric_bike"),
        ("2026-02", "31001", "", "", "docked_bike"),
        ("2026-02", "31001", "31000", "member", "classic_bike"),
    ]
    trip_dir = tmp_path / "trips"
    trip_dir.mkdir()
    for month in sorted({t[0] for t in trips}):
        rows = [
            f"R{n},{bike},{month}-0{n + 1} 10:00:00,{month}-0{n + 1} 10:10:00,"
            f"S,{start},S,{end},38.9,-77.0,38.9,-77.0,{rider}\n"
            for n, (m, start, end, rider, bike) in enumerate(trips)
            if m == month
        ]
        path = trip_dir / f"{month.replace('-', '')}-capitalbikeshare-tripdata.csv"
        path.write_text(_TRIP_HEADER + "".join(rows), encoding="utf-8")
    trends_dir = tmp_path / "trends"
    trends.generate_and_write(
        input_path=str(trip_dir), output_dir=str(trends_dir), max_station_plots=1
    )

    system = pd.read_csv(trends_dir / "monthly_system_ridership.csv")
    assert system["total_trips"].sum() == len(trips)
    for split in (
        ["member_trips", "casual_trips", "other_rider_trips"],
        ["electric_trips", "classic_trips", "other_bike_trips"],
    ):
        assert system[split].sum(axis=1).equals(system["total_trips"])

    map_dir = tmp_path / "maps"
    with caplog.at_level("WARNING"):
        mod.run(
            ridership_input=str(trends_dir / "monthly_station_ridership.csv"),
            geojson_input=str(gbfs_dir / "gbfs_stations.geojson"),
            shapefile_input=str(zipped),
            output_dir=map_dir,
        )
    # Logged once per map: the one arrival at station 39999.
    unmatched = [int(n) for n in re.findall(r"\((\d+) total trips\) have no", caplog.text)]
    assert unmatched == [1, 1]

    docked_activity = sum(bool(t[1]) for t in trips) + sum(bool(t[2]) for t in trips)
    for map_name in ("gbfs_stations_ridership.geojson", "gbfs_stations_ridership.shp"):
        saved = gpd.read_file(map_dir / map_name)
        assert saved["total"].sum() + unmatched[0] == docked_activity
        assert (saved["departures"] + saved["arrivals"]).equals(saved["total"])
        totals = saved.set_index("short_name")["total"].to_dict()
        assert totals == {"31000": 4, "31001": 5, "31002": 0}
