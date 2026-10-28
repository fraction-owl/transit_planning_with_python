from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from scripts.gtfs_exports.gtfs_route_features_gpd import (
    _service_day_coverage,
    collapse_to_route_number,
    compute_route_supply_metrics,
    compute_serves_cbd,
    find_cbd_stop_ids,
    load_cbd_area,
    main,
)

FIXTURES = Path(__file__).parent / "fixtures"
GTFS_BASIC = FIXTURES / "gtfs_basic"
MOCK_GTFS_DC_ZIP = FIXTURES / "mock_gtfs_dc.zip"
MOCK_CBD_ZIP = FIXTURES / "mock_cbd_dc.zip"


def _two_stop_times(spec: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a stop_times frame: each (trip_id, first_time, last_time) -> two rows."""
    rows: list[tuple[str, str, str, str, str]] = []
    for trip_id, first, last in spec:
        rows.append((trip_id, first, first, "1", f"{trip_id}_a"))
        rows.append((trip_id, last, last, "2", f"{trip_id}_b"))
    return pd.DataFrame(
        rows, columns=["trip_id", "arrival_time", "departure_time", "stop_sequence", "stop_id"]
    )


# ---------------------------------------------------------------------------
# _service_day_coverage  (one-hour bins, divided by 24)
# ---------------------------------------------------------------------------


def test_coverage_one_bin_per_departure() -> None:
    # A single departure occupies exactly one one-hour bin (06:00 -> hour 6).
    trips = pd.DataFrame({"route_id": ["A"], "start_sec": [21600.0]})
    cov = _service_day_coverage(trips).set_index("route_id")["pct_day_with_service"]
    assert cov["A"] == pytest.approx(100.0 * 1 / 24)


def test_coverage_counts_only_the_start_bin_not_the_run() -> None:
    # A single long trip (06:00-08:00) credits ONLY its departure bin (hour 6), never the
    # bins it runs through: a rider at a stop sees one bus, not service all morning.
    # end_sec is supplied to prove it is ignored.
    trips = pd.DataFrame({"route_id": ["A"], "start_sec": [21600.0], "end_sec": [28800.0]})
    cov = _service_day_coverage(trips).set_index("route_id")["pct_day_with_service"]
    assert cov["A"] == pytest.approx(100.0 * 1 / 24)


def test_hourly_service_fills_each_operating_hour() -> None:
    # Hourly departures 06:00, 07:00, 08:00, 09:00 cover four whole hours -> 4/24.
    # One-hour bins credit a 60-minute-frequency route for every hour it runs (not half);
    # the 30- vs 60-minute distinction belongs to median_headway_min, not this span metric.
    trips = pd.DataFrame({"route_id": ["A"] * 4, "start_sec": [21600.0, 25200.0, 28800.0, 32400.0]})
    cov = _service_day_coverage(trips).set_index("route_id")["pct_day_with_service"]
    assert cov["A"] == pytest.approx(100.0 * 4 / 24)


def test_coverage_counts_distinct_departure_bins() -> None:
    # Departures at 06:00 (hour 6), 08:00 (hour 8), 09:00 (hour 9) -> three bins; the
    # empty hour 7 is a real gap and is not counted.
    trips = pd.DataFrame({"route_id": ["A"] * 3, "start_sec": [21600.0, 28800.0, 32400.0]})
    cov = _service_day_coverage(trips).set_index("route_id")["pct_day_with_service"]
    assert cov["A"] == pytest.approx(100.0 * 3 / 24)


def test_coverage_departures_in_same_bin_dedupe() -> None:
    # 06:00 and 06:30 both fall in the same one-hour bin (hour 6) -> one served bin.
    trips = pd.DataFrame({"route_id": ["A", "A"], "start_sec": [21600.0, 23400.0]})
    cov = _service_day_coverage(trips).set_index("route_id")["pct_day_with_service"]
    assert cov["A"] == pytest.approx(100.0 * 1 / 24)


def test_coverage_keeps_extended_clock_no_wrap() -> None:
    # 01:00 -> hour 1 and 25:00 -> hour 25 stay distinct; wrapping 25:00 to 01:00 would
    # wrongly collapse them into a single bin.
    trips = pd.DataFrame({"route_id": ["A", "A"], "start_sec": [3600.0, 90000.0]})
    cov = _service_day_coverage(trips).set_index("route_id")["pct_day_with_service"]
    assert cov["A"] == pytest.approx(100.0 * 2 / 24)


def test_coverage_drops_missing_start() -> None:
    # Rows without a start time are ignored; the route is summarized from the rest.
    trips = pd.DataFrame({"route_id": ["A", "A"], "start_sec": [21600.0, float("nan")]})
    cov = _service_day_coverage(trips).set_index("route_id")["pct_day_with_service"]
    assert cov["A"] == pytest.approx(100.0 * 1 / 24)


# ---------------------------------------------------------------------------
# compute_route_supply_metrics  (modal vs longest shape)
# ---------------------------------------------------------------------------


def test_modal_shape_length_distinct_from_longest() -> None:
    # S1 is the longest (10 mi) but is run by one trip; S2 (5 mi) is run by three,
    # so route_length_modal_mi must report the 5-mi variant, not the 10-mi one.
    trips = pd.DataFrame(
        {
            "route_id": ["A", "A", "A", "A"],
            "trip_id": ["t1", "t2", "t3", "t4"],
            "shape_id": ["S1", "S2", "S2", "S2"],
        }
    )
    stop_times = _two_stop_times(
        [
            ("t1", "06:00:00", "06:30:00"),
            ("t2", "07:00:00", "07:30:00"),
            ("t3", "08:00:00", "08:30:00"),
            ("t4", "09:00:00", "09:30:00"),
        ]
    )
    shape_len = pd.DataFrame({"shape_id": ["S1", "S2"], "shape_len_m": [16093.44, 8046.72]})

    supply = compute_route_supply_metrics(trips, stop_times, shape_len).set_index("route_id")
    assert supply.loc["A", "route_length_mi"] == pytest.approx(10.0)
    assert supply.loc["A", "route_length_modal_mi"] == pytest.approx(5.0)
    # Daily revenue miles still sum every trip's own shape: 10 + 3*5 = 25.
    assert supply.loc["A", "revenue_miles"] == pytest.approx(25.0)
    # Four trips departing at 06/07/08/09 -> four distinct one-hour bins.
    assert supply.loc["A", "pct_day_with_service"] == pytest.approx(100.0 * 4 / 24)


def test_modal_tie_breaks_toward_longer_shape() -> None:
    # Equal trip counts -> the longer shape wins, so the fuller variant is reported.
    trips = pd.DataFrame(
        {
            "route_id": ["A", "A"],
            "trip_id": ["t1", "t2"],
            "shape_id": ["S_short", "S_long"],
        }
    )
    stop_times = _two_stop_times([("t1", "06:00:00", "06:30:00"), ("t2", "07:00:00", "07:30:00")])
    shape_len = pd.DataFrame(
        {"shape_id": ["S_short", "S_long"], "shape_len_m": [8046.72, 16093.44]}
    )
    supply = compute_route_supply_metrics(trips, stop_times, shape_len).set_index("route_id")
    assert supply.loc["A", "route_length_modal_mi"] == pytest.approx(10.0)


def test_no_shapes_yields_nan_lengths() -> None:
    trips = pd.DataFrame({"route_id": ["A"], "trip_id": ["t1"], "shape_id": [""]})
    stop_times = _two_stop_times([("t1", "06:00:00", "06:30:00")])
    supply = compute_route_supply_metrics(
        trips, stop_times, pd.DataFrame(columns=["shape_id", "shape_len_m"])
    ).set_index("route_id")
    assert pd.isna(supply.loc["A", "route_length_mi"])
    assert pd.isna(supply.loc["A", "route_length_modal_mi"])
    # Coverage does not depend on shapes, so it is still populated.
    assert supply.loc["A", "pct_day_with_service"] == pytest.approx(100.0 * 1 / 24)


# ---------------------------------------------------------------------------
# compute_route_supply_metrics  (n_directions)
# ---------------------------------------------------------------------------


def test_n_directions_counts_unique_direction_ids() -> None:
    # A runs both directions; B is one-way (all trips direction 0).
    trips = pd.DataFrame(
        {
            "route_id": ["A", "A", "A", "B", "B"],
            "trip_id": ["t1", "t2", "t3", "t4", "t5"],
            "shape_id": ["S", "S", "S", "S", "S"],
            "direction_id": ["0", "1", "0", "0", "0"],
        }
    )
    stop_times = _two_stop_times(
        [
            ("t1", "06:00:00", "06:30:00"),
            ("t2", "07:00:00", "07:30:00"),
            ("t3", "08:00:00", "08:30:00"),
            ("t4", "06:00:00", "06:30:00"),
            ("t5", "07:00:00", "07:30:00"),
        ]
    )
    shape_len = pd.DataFrame({"shape_id": ["S"], "shape_len_m": [8046.72]})
    supply = compute_route_supply_metrics(trips, stop_times, shape_len).set_index("route_id")
    assert supply.loc["A", "n_directions"] == 2
    assert supply.loc["B", "n_directions"] == 1


def test_n_directions_ignores_blanks_and_handles_missing_column() -> None:
    # Blank direction_id rows carry no information; all-blank -> NaN, not 1.
    trips = pd.DataFrame(
        {
            "route_id": ["A", "A", "B"],
            "trip_id": ["t1", "t2", "t3"],
            "shape_id": ["S", "S", "S"],
            "direction_id": ["1", "", None],
        }
    )
    stop_times = _two_stop_times(
        [
            ("t1", "06:00:00", "06:30:00"),
            ("t2", "07:00:00", "07:30:00"),
            ("t3", "06:00:00", "06:30:00"),
        ]
    )
    shape_len = pd.DataFrame({"shape_id": ["S"], "shape_len_m": [8046.72]})
    supply = compute_route_supply_metrics(trips, stop_times, shape_len).set_index("route_id")
    assert supply.loc["A", "n_directions"] == 1
    assert pd.isna(supply.loc["B", "n_directions"])

    # Feed without a direction_id column at all -> NaN for every route.
    no_dir = compute_route_supply_metrics(
        trips.drop(columns="direction_id"), stop_times, shape_len
    ).set_index("route_id")
    assert no_dir["n_directions"].isna().all()


# ---------------------------------------------------------------------------
# collapse_to_route_number  (multiple GTFS route_ids -> one public number)
# ---------------------------------------------------------------------------


def _collision_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["100a", "100b"],
            "route_short_name": ["100", "100"],
            "trips_per_day": [10, 30],
            "revenue_hours": [5.0, 15.0],
            "revenue_miles": [20.0, 60.0],
            "route_length_mi": [9.0, 12.0],
            "route_length_modal_mi": [4.0, 8.0],
            "pct_day_with_service": [50.0, 90.0],
            "median_headway_min": [20.0, 10.0],
            "n_directions": [1, 2],
            "n_stops": [10, 20],
            "shared_stop_share": [0.5, 0.5],
            "n_competitor_routes": [2, 3],
            "competitor_trips_at_shared_stops": [4.0, 6.0],
            "competition_intensity": [0.4, 0.2],
            "min_start_sec": [21600.0, 20000.0],
            "max_end_sec": [80000.0, 79000.0],
        }
    )


def test_collision_uses_trips_weighted_means_for_new_cols() -> None:
    collapsed = collapse_to_route_number(_collision_metrics(), route_col="route_short_name")
    collapsed = collapsed.set_index("route_id")
    # (50*10 + 90*30) / 40 = 80 ; (4*10 + 8*30) / 40 = 7
    assert collapsed.loc["100", "pct_day_with_service"] == pytest.approx(80.0)
    assert collapsed.loc["100", "route_length_modal_mi"] == pytest.approx(7.0)
    # route_length_mi still takes the max extent across sub-routes.
    assert collapsed.loc["100", "route_length_mi"] == pytest.approx(12.0)
    # n_directions takes the max (two one-way variants stay conservative, not summed).
    assert collapsed.loc["100", "n_directions"] == 2


def test_no_collision_passes_new_columns_through() -> None:
    solo = _collision_metrics().iloc[[0]].copy()
    collapsed = collapse_to_route_number(solo, route_col="route_short_name").set_index("route_id")
    assert collapsed.loc["100", "pct_day_with_service"] == pytest.approx(50.0)
    assert collapsed.loc["100", "route_length_modal_mi"] == pytest.approx(4.0)
    assert collapsed.loc["100", "n_directions"] == 1


# ---------------------------------------------------------------------------
# main — full pipeline against gtfs_basic
# ---------------------------------------------------------------------------


def test_main_end_to_end_on_gtfs_basic(tmp_path: Path) -> None:
    assert main(["--gtfs-dir", str(GTFS_BASIC), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "gtfs_route_features_runlog.txt").exists()

    features = pd.read_csv(tmp_path / "gtfs_route_features.csv").set_index("route_id")
    assert features.index.tolist() == ["R1", "R2", "R3"]
    assert features["trips_per_day"].tolist() == [3, 3, 3]
    # Each route runs three trips of 25 / 20 / 15 minutes.
    assert features["revenue_hours"].tolist() == pytest.approx([1.25, 1.0, 0.75])
    # R1's first departure is 07:00 and its last arrival 17:25.
    assert features.loc["R1", "span_hours"] == pytest.approx(10 + 25 / 60, abs=1e-4)
    # Trip starts are 4+ hours apart, which the headway median ignores; each
    # route departs in three one-hour bins.
    assert features["median_headway_min"].isna().all()
    assert features["pct_day_with_service"].tolist() == [12.5, 12.5, 12.5]
    # SHP_R1 is 3,450 m end to end (its shape_dist_traveled).
    assert features.loc["R1", "route_length_mi"] == pytest.approx(3450 / 1609.344, rel=0.01)
    assert features["n_stops"].tolist() == [6, 5, 4]
    # R1 shares S2-S5 with R2 and R3. R2 spreads 3 trips over 5 stops and R3 3
    # over 4, putting 3.3 competitor trips at those stops: 1.1 per R1 trip.
    assert features.loc["R1", "shared_stop_share"] == pytest.approx(4 / 6, abs=1e-4)
    assert features["n_competitor_routes"].tolist() == [2, 2, 2]
    assert features.loc["R1", "competition_intensity"] == pytest.approx(1.1)
    # No CBD layer configured: the column is left out rather than filled with zeros.
    assert "serves_cbd" not in features.columns
    runlog = (tmp_path / "gtfs_route_features_runlog.txt").read_text(encoding="utf-8")
    assert "CBD layer:       (none)" in runlog


# ---------------------------------------------------------------------------
# serves_cbd  (optional CBD polygon layer)
# ---------------------------------------------------------------------------


def _write_layer(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    gdf.to_file(path)
    return path


def test_serves_cbd_on_the_mock_dc_feed_and_cbd_fixture(tmp_path: Path) -> None:
    # Routes 10, 20, and 30 stop in the downtown polygon only (none stop in the annex).
    # Route 40 has no stop in either polygon, although its line cuts across the annex.
    # Holiday-only route 50H has no weekday service, so it has no row at all.
    with zipfile.ZipFile(MOCK_GTFS_DC_ZIP) as zf:
        zf.extractall(tmp_path / "feed")
    out_dir = tmp_path / "out"
    argv = ["--gtfs-dir", str(tmp_path / "feed" / "dc"), "--output-dir", str(out_dir)]

    assert main([*argv, "--cbd-path", str(MOCK_CBD_ZIP)]) == 0

    features = pd.read_csv(out_dir / "gtfs_route_features.csv", dtype={"route_id": str})
    assert dict(zip(features["route_id"], features["serves_cbd"])) == {
        "10": 1,
        "20": 1,
        "30": 1,
        "40": 0,
    }
    runlog = (out_dir / "gtfs_route_features_runlog.txt").read_text(encoding="utf-8")
    assert f"CBD layer:       {MOCK_CBD_ZIP}  sha256=" in runlog


def test_serves_cbd_counts_a_stop_served_only_on_another_schedule(tmp_path: Path) -> None:
    # R2's only CBD stop (S11) is on a Saturday trip; the weekday supply features still
    # come from the Tuesday trips, but the flag counts every schedule in the feed.
    feed = tmp_path / "feed"
    shutil.copytree(GTFS_BASIC, feed)
    with (feed / "calendar.txt").open("a", encoding="utf-8") as fh:
        fh.write("SAT,0,0,0,0,0,1,0,20260101,20261231\n")
    with (feed / "stops.txt").open("a", encoding="utf-8") as fh:
        fh.write("S11,1011,Saturday Market,38.7600,-77.0700\n")
    with (feed / "trips.txt").open("a", encoding="utf-8") as fh:
        fh.write("R2,SAT,T10,0,B7,SHP_R2\n")
    with (feed / "stop_times.txt").open("a", encoding="utf-8") as fh:
        fh.write("T10,09:00:00,09:00:00,S7,1\nT10,09:10:00,09:10:00,S11,2\n")
    cbd = gpd.GeoDataFrame(
        {"NAME": ["CBD"]}, geometry=[Point(-77.0700, 38.7600).buffer(0.001)], crs="EPSG:4326"
    )
    cbd_path = _write_layer(cbd, tmp_path / "cbd.shp")

    argv = ["--gtfs-dir", str(feed), "--output-dir", str(tmp_path / "out")]
    assert main([*argv, "--cbd-path", str(cbd_path)]) == 0

    features = pd.read_csv(tmp_path / "out" / "gtfs_route_features.csv").set_index("route_id")
    assert features["serves_cbd"].to_dict() == {"R1": 0, "R2": 1, "R3": 0}
    assert features.loc["R2", "trips_per_day"] == 3


def test_stop_on_a_polygon_edge_counts_as_inside() -> None:
    stops = pd.DataFrame(
        {
            "stop_id": ["inside", "on_edge", "on_corner", "outside", "no_coords"],
            "stop_lat": ["38.745", "38.745", "38.750", "38.745", ""],
            "stop_lon": ["-77.085", "-77.080", "-77.080", "-77.0799", ""],
        }
    )
    area = gpd.GeoSeries([box(-77.09, 38.74, -77.08, 38.75)], crs="EPSG:4326")

    assert find_cbd_stop_ids(stops, area) == {"inside", "on_edge", "on_corner"}


def test_any_polygon_in_the_layer_counts(tmp_path: Path) -> None:
    layer = gpd.GeoDataFrame(
        {"NAME": ["Downtown", "Annex"]},
        geometry=[box(-77.09, 38.74, -77.08, 38.75), box(-77.06, 38.76, -77.05, 38.77)],
        crs="EPSG:4326",
    )
    area = load_cbd_area(_write_layer(layer, tmp_path / "cbd.shp"))
    stops = pd.DataFrame(
        {
            "stop_id": ["downtown", "annex", "between"],
            "stop_lat": ["38.745", "38.765", "38.755"],
            "stop_lon": ["-77.085", "-77.055", "-77.070"],
        }
    )

    assert find_cbd_stop_ids(stops, area) == {"downtown", "annex"}


def test_serves_cbd_collapses_route_ids_onto_the_public_number(tmp_path: Path) -> None:
    # Two route_ids publish as route 5; only the second stops in the CBD, which is
    # enough to flag route 5. Route 6 never stops there.
    stop_times = tmp_path / "stop_times.txt"
    stop_times.write_text("trip_id,stop_id\nT1,A\nT1,B\nT2,B\nT2,CBD\nT3,A\n", encoding="utf-8")
    all_trips = pd.DataFrame({"route_id": ["5a", "5b", "6"], "trip_id": ["T1", "T2", "T3"]})
    routes = pd.DataFrame({"route_id": ["5a", "5b", "6"], "route_short_name": ["5", "5", "6"]})

    flags = compute_serves_cbd(stop_times, all_trips, routes, {"CBD"}, chunksize=2)

    assert flags.set_index("route_id")["serves_cbd"].to_dict() == {"5": 1, "6": 0}


@pytest.mark.filterwarnings("ignore:'crs' was not provided")
def test_cbd_layer_must_be_polygons_with_a_coordinate_system(tmp_path: Path) -> None:
    points = gpd.GeoDataFrame({"NAME": ["x"]}, geometry=[Point(-77.0, 38.9)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="must hold polygons"):
        load_cbd_area(_write_layer(points, tmp_path / "points.shp"))

    no_crs = gpd.GeoDataFrame({"NAME": ["x"]}, geometry=[box(-77.1, 38.8, -77.0, 38.9)])
    with pytest.raises(ValueError, match="no coordinate system"):
        load_cbd_area(_write_layer(no_crs, tmp_path / "no_crs.shp"))
