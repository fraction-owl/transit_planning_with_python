from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

import scripts.service_coverage.cabi_coverage_by_route_gpd as cabi_mod
from scripts.service_coverage.cabi_coverage_by_route_gpd import (
    _prepare_route_buffers,
    _web_mercator_ground_scale,
    join_ridership_onto_stations,
    load_daytype_ridership,
    load_stations_layer,
    run,
    summarize_stations_by_route,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

GTFS_BASIC = Path("tests/fixtures/gtfs_basic")


def _write_stations(path: Path) -> None:
    """Write a 3-station point layer: two near gtfs_basic stops, one far away."""
    gdf = gpd.GeoDataFrame(
        {
            "station_id": ["31001", "31002", "31003"],
            "name": ["Near S1", "Far away", "Near S2 (no ridership)"],
        },
        geometry=[
            Point(-77.0801, 38.7401),  # ~10 m from stop S1
            Point(-76.9, 38.9),  # far outside every catchment
            Point(-77.0821, 38.7461),  # ~10 m from stop S2
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")


def _write_ridership(path: Path) -> None:
    """Write day-type averages for two stations; 31003 has no ridership row."""
    pd.DataFrame(
        {
            "station_id": ["31001", "31002"],
            "station_name": ["Near S1", "Far away"],
            "avg_weekday_riders": [10.5, 3.0],
            "avg_saturday_riders": [5.25, 1.0],
            "avg_sunday_riders": [2.0, 0.5],
            "weekday_days": [500, 500],
            "saturday_days": [104, 104],
            "sunday_days": [126, 126],
        }
    ).to_csv(path, index=False)


@pytest.fixture()
def inputs(tmp_path: Path) -> dict[str, Path]:
    stations = tmp_path / "gbfs_stations.geojson"
    ridership = tmp_path / "station_daytype_ridership.csv"
    _write_stations(stations)
    _write_ridership(ridership)
    return {"stations": stations, "ridership": ridership, "out": tmp_path / "out"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_load_stations_layer_from_folder_glob(inputs: dict[str, Path]) -> None:
    stations = load_stations_layer(inputs["stations"].parent)
    assert sorted(stations["station_id"]) == ["31001", "31002", "31003"]
    assert stations.crs is not None and stations.crs.to_epsg() == 3857


def test_load_stations_layer_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stations_layer(tmp_path / "nope")


def test_load_stations_layer_missing_id_column_raises(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame({"nope": ["1"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    path = tmp_path / "gbfs_stations.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(KeyError, match="station_id"):
        load_stations_layer(path)


def test_load_daytype_ridership_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "station_daytype_ridership.csv"
    pd.DataFrame({"station_id": ["1"], "avg_weekday_riders": [1.0]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="avg_saturday_riders"):
        load_daytype_ridership(path)


def test_load_daytype_ridership_normalizes_float_like_ids(tmp_path: Path) -> None:
    # A numeric-looking id parsed as float ("31001.0") must match the string id.
    path = tmp_path / "station_daytype_ridership.csv"
    pd.DataFrame(
        {
            "station_id": [31001.0],
            "avg_weekday_riders": [1.0],
            "avg_saturday_riders": [1.0],
            "avg_sunday_riders": [1.0],
        }
    ).to_csv(path, index=False)
    ridership = load_daytype_ridership(path)
    assert list(ridership["station_id"]) == ["31001"]


# ---------------------------------------------------------------------------
# Join + rollup
# ---------------------------------------------------------------------------


def test_join_zero_fills_stations_without_ridership(inputs: dict[str, Path]) -> None:
    stations = load_stations_layer(inputs["stations"])
    ridership = load_daytype_ridership(inputs["ridership"])
    joined = join_ridership_onto_stations(stations, ridership)
    row = joined[joined["station_id"] == "31003"].iloc[0]
    assert row["avg_weekday_riders"] == 0.0
    assert row["avg_saturday_riders"] == 0.0
    assert row["avg_sunday_riders"] == 0.0


def test_summarize_empty_stations_reports_zeros() -> None:
    tables = {
        "routes": pd.DataFrame({"route_id": ["R1"]}),
        "trips": pd.DataFrame({"route_id": ["R1"], "trip_id": ["T1"], "shape_id": ["SH1"]}),
        "stop_times": pd.DataFrame({"trip_id": ["T1"], "stop_id": ["S1"], "stop_sequence": [1]}),
        "stops": pd.DataFrame({"stop_id": ["S1"], "stop_lat": [0.0], "stop_lon": [0.0]}),
    }
    buffers = _prepare_route_buffers(tables, use_shape_buffer=False, buffer_dist_ft=1320.0)
    empty = gpd.GeoDataFrame(
        {"station_id": [], "avg_weekday_riders": []}, geometry=[], crs=buffers.crs
    )
    summary = summarize_stations_by_route(buffers, empty)
    assert list(summary["route_id"]) == ["R1"]
    assert summary["cabi_stations_served"].iloc[0] == 0
    assert summary["cabi_weekday_riders_served"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# _prepare_route_buffers
# ---------------------------------------------------------------------------


def test_prepare_route_buffers_stop_mode_without_shape_id_column() -> None:
    """shape_id is optional in GTFS; stop-buffer mode must not require it."""
    tables = {
        "routes": pd.DataFrame({"route_id": ["R1"]}),
        "trips": pd.DataFrame({"route_id": ["R1"], "trip_id": ["T1"]}),
        "stop_times": pd.DataFrame({"trip_id": ["T1"], "stop_id": ["S1"], "stop_sequence": [1]}),
        "stops": pd.DataFrame({"stop_id": ["S1"], "stop_lat": [0.0], "stop_lon": [0.0]}),
    }
    buffers = _prepare_route_buffers(tables, use_shape_buffer=False, buffer_dist_ft=1320.0)
    assert list(buffers["route_id"]) == ["R1"]
    assert not buffers.geometry.is_empty.any()


def test_prepare_route_buffers_shape_mode_missing_shape_id_raises() -> None:
    """Shape-buffer mode without a trips.txt shape_id column fails clearly."""
    tables = {
        "routes": pd.DataFrame({"route_id": ["R1"]}),
        "trips": pd.DataFrame({"route_id": ["R1"], "trip_id": ["T1"]}),
        "stop_times": pd.DataFrame({"trip_id": ["T1"], "stop_id": ["S1"], "stop_sequence": [1]}),
        "stops": pd.DataFrame({"stop_id": ["S1"], "stop_lat": [0.0], "stop_lon": [0.0]}),
    }
    with pytest.raises(ValueError, match="shape_id"):
        _prepare_route_buffers(tables, use_shape_buffer=True, buffer_dist_ft=1320.0)


def test_web_mercator_ground_scale_matches_inverse_cosine() -> None:
    """At 38.9°N the Web Mercator correction is 1/cos(lat); elsewhere it is 1."""
    scale = _web_mercator_ground_scale("EPSG:3857", pd.Series([38.9]))
    assert scale == pytest.approx(1.0 / math.cos(math.radians(38.9)))
    assert _web_mercator_ground_scale("EPSG:2283", pd.Series([38.9])) == 1.0
    assert _web_mercator_ground_scale("EPSG:3857", pd.Series([], dtype=float)) == 1.0


def test_stop_buffer_spans_true_ground_distance_at_dc_latitude() -> None:
    """A ¼-mile buffer at DC latitude reaches a point ~1200 ft of *ground* away.

    Web Mercator map "meters" shrink by cos(latitude) on the ground, so without
    the correction a 1320 ft buffer only spans ~1027 ft at 38.9°N and this point
    would be missed. A point beyond the radius stays outside (no over-buffering).
    """
    lat, lon = 38.9, -77.03
    meters_per_deg_lon_ground = 111_320.0 * math.cos(math.radians(lat))
    near_dlon = (1200 * 0.3048) / meters_per_deg_lon_ground  # ~1200 ft ground
    far_dlon = (1500 * 0.3048) / meters_per_deg_lon_ground  # ~1500 ft ground
    tables = {
        "routes": pd.DataFrame({"route_id": ["R1"]}),
        "trips": pd.DataFrame({"route_id": ["R1"], "trip_id": ["T1"]}),
        "stop_times": pd.DataFrame({"trip_id": ["T1"], "stop_id": ["S1"], "stop_sequence": [1]}),
        "stops": pd.DataFrame({"stop_id": ["S1"], "stop_lat": [lat], "stop_lon": [lon]}),
    }
    buffers = _prepare_route_buffers(tables, use_shape_buffer=False, buffer_dist_ft=1320.0)
    points = gpd.GeoSeries(
        [Point(lon + near_dlon, lat), Point(lon + far_dlon, lat)], crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    catchment = buffers.geometry.iloc[0]
    assert catchment.contains(points.iloc[0])
    assert not catchment.contains(points.iloc[1])


# ---------------------------------------------------------------------------
# main (placeholder guard)
# ---------------------------------------------------------------------------


def test_main_blocks_unedited_placeholder_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CONFIG untouched and no flags, main() warns and does not run."""
    calls: list[dict] = []
    monkeypatch.setattr(cabi_mod, "run", lambda **kw: calls.append(kw))
    assert cabi_mod.main([]) == 2
    assert calls == []


def test_main_runs_after_config_edit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The documented edit-CONFIG-then-run workflow must reach run()."""
    calls: list[dict] = []
    monkeypatch.setattr(cabi_mod, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cabi_mod, "GTFS_DIR", tmp_path / "gtfs")
    monkeypatch.setattr(cabi_mod, "STATIONS_PATH", tmp_path / "stations.geojson")
    monkeypatch.setattr(cabi_mod, "RIDERSHIP_CSV", tmp_path / "ridership.csv")
    assert cabi_mod.main([]) == 0
    assert len(calls) == 1


def test_main_runs_with_cli_flags_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Flags alone (CONFIG untouched) must also reach run()."""
    calls: list[dict] = []
    monkeypatch.setattr(cabi_mod, "run", lambda **kw: calls.append(kw))
    assert (
        cabi_mod.main(
            [
                "--gtfs-dir",
                str(tmp_path / "gtfs"),
                "--stations-path",
                str(tmp_path / "stations.geojson"),
                "--ridership-csv",
                str(tmp_path / "ridership.csv"),
            ]
        )
        == 0
    )
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# run (end to end)
# ---------------------------------------------------------------------------


def test_run_end_to_end_rolls_up_by_route(inputs: dict[str, Path]) -> None:
    summary = run(
        gtfs_dir=GTFS_BASIC,
        stations_path=inputs["stations"],
        ridership_csv=inputs["ridership"],
        output_dir=inputs["out"],
    )
    out_csv = inputs["out"] / "cabi_coverage_by_route.csv"
    assert out_csv.exists()
    assert list(summary.columns) == [
        "route_id",
        "route_short_name",
        "cabi_stations_served",
        "cabi_weekday_riders_served",
        "cabi_saturday_riders_served",
        "cabi_sunday_riders_served",
    ]

    by_route = summary.set_index("route_id")
    # 31001 (near S1) and 31003 (near S2) fall in R1's stop catchment; 31003
    # has no ridership row, so it adds to the station count but 0 riders.
    assert by_route.loc["R1", "cabi_stations_served"] == 2
    assert by_route.loc["R1", "cabi_weekday_riders_served"] == pytest.approx(10.5)
    assert by_route.loc["R1", "cabi_saturday_riders_served"] == pytest.approx(5.25)
    assert by_route.loc["R1", "cabi_sunday_riders_served"] == pytest.approx(2.0)
    # 31002 is far from every stop; routes not reaching a station report zeros.
    assert by_route.loc["R2", "cabi_stations_served"] == 0
    assert by_route.loc["R2", "cabi_weekday_riders_served"] == 0.0

    # The written CSV round-trips the same numbers.
    written = pd.read_csv(out_csv, dtype={"route_id": str})
    assert written["cabi_stations_served"].sum() == summary["cabi_stations_served"].sum()


def test_run_output_covers_every_route(inputs: dict[str, Path]) -> None:
    summary = run(
        gtfs_dir=GTFS_BASIC,
        stations_path=inputs["stations"],
        ridership_csv=inputs["ridership"],
        output_dir=inputs["out"],
    )
    routes = pd.read_csv(GTFS_BASIC / "routes.txt", dtype={"route_id": str})
    assert sorted(summary["route_id"]) == sorted(routes["route_id"].astype(str))
