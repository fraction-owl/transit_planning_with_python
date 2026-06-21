from __future__ import annotations

import pandas as pd
import pytest

from scripts.gtfs_exports.gtfs_route_features import (
    _service_day_coverage,
    collapse_to_route_number,
    compute_route_supply_metrics,
)


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


def test_no_collision_passes_new_columns_through() -> None:
    solo = _collision_metrics().iloc[[0]].copy()
    collapsed = collapse_to_route_number(solo, route_col="route_short_name").set_index("route_id")
    assert collapsed.loc["100", "pct_day_with_service"] == pytest.approx(50.0)
    assert collapsed.loc["100", "route_length_modal_mi"] == pytest.approx(4.0)
