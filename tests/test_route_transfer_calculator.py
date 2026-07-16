from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.gtfs_exports.route_transfer_calculator as rtc
from scripts.gtfs_exports.route_transfer_calculator import (
    build_output_tables,
    compute_transfers,
    distance_to_meters,
    has_timed_connection,
    parse_gtfs_time,
    service_ids_active_on_day,
)

# ---------------------------------------------------------------------------
# parse_gtfs_time
# ---------------------------------------------------------------------------


def test_parse_gtfs_time_basic() -> None:
    assert parse_gtfs_time("07:35:00") == 7 * 3600 + 35 * 60


def test_parse_gtfs_time_after_midnight_kept() -> None:
    # 25:10:00 must stay > 24h, not wrap to 01:10.
    assert parse_gtfs_time("25:10:00") == 25 * 3600 + 10 * 60


@pytest.mark.parametrize("bad", [None, "", "7:35", "abc", 730, float("nan")])
def test_parse_gtfs_time_invalid_returns_none(bad: object) -> None:
    assert parse_gtfs_time(bad) is None


# ---------------------------------------------------------------------------
# distance_to_meters
# ---------------------------------------------------------------------------


def test_distance_to_meters_units() -> None:
    assert distance_to_meters(1, "miles") == pytest.approx(1609.34)
    assert distance_to_meters(10, "feet") == pytest.approx(3.048)
    assert distance_to_meters(5, "meters") == 5.0


def test_distance_to_meters_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError):
        distance_to_meters(1, "furlongs")


# ---------------------------------------------------------------------------
# service_ids_active_on_day
# ---------------------------------------------------------------------------


def test_service_ids_active_on_day_reads_calendar_flags() -> None:
    calendar = pd.DataFrame(
        {
            "service_id": ["weekday", "weekend"],
            "monday": ["1", "0"],
            "saturday": ["0", "1"],
        }
    )
    assert service_ids_active_on_day(calendar, "monday") == {"weekday"}
    assert service_ids_active_on_day(calendar, "saturday") == {"weekend"}


def test_service_ids_active_on_day_weekday_keeps_any_monday_to_friday_service() -> None:
    calendar = pd.DataFrame(
        {
            "service_id": ["weekday", "friday_only", "weekend", "daily"],
            "monday": ["1", "0", "0", "1"],
            "tuesday": ["1", "0", "0", "1"],
            "wednesday": ["1", "0", "0", "1"],
            "thursday": ["1", "0", "0", "1"],
            "friday": ["1", "1", "0", "1"],
            "saturday": ["0", "0", "1", "1"],
            "sunday": ["0", "0", "1", "1"],
        }
    )
    # Daily service runs on weekdays too; weekend-only service is excluded.
    assert service_ids_active_on_day(calendar, "weekday") == {"weekday", "friday_only", "daily"}


def test_service_ids_active_on_day_none_calendar_returns_none() -> None:
    assert service_ids_active_on_day(None, "monday") is None
    assert service_ids_active_on_day(None, "weekday") is None


def test_service_ids_active_on_day_invalid_day_raises() -> None:
    with pytest.raises(ValueError):
        service_ids_active_on_day(None, "someday")


# ---------------------------------------------------------------------------
# has_timed_connection
# ---------------------------------------------------------------------------


def test_has_timed_connection_feasible_returns_shortest_wait() -> None:
    arrivals = np.array([28800.0])  # 08:00
    departures = np.array([29100.0, 30000.0])  # 08:05, 08:20
    feasible, wait = has_timed_connection(arrivals, departures, 60.0, 1800.0)
    assert feasible
    assert wait == pytest.approx(300.0)  # catches the 08:05 bus


def test_has_timed_connection_too_soon_is_infeasible() -> None:
    # Connector leaves only 30s after arrival but the rider needs 120s to walk.
    arrivals = np.array([28800.0])
    departures = np.array([28830.0])
    feasible, wait = has_timed_connection(arrivals, departures, 120.0, 1800.0)
    assert not feasible
    assert wait is None


def test_has_timed_connection_too_late_is_infeasible() -> None:
    arrivals = np.array([28800.0])
    departures = np.array([28800.0 + 3600.0])  # 1h later, beyond 30-min wait
    feasible, _ = has_timed_connection(arrivals, departures, 60.0, 1800.0)
    assert not feasible


def test_has_timed_connection_empty_arrays() -> None:
    assert has_timed_connection(np.array([]), np.array([1.0]), 0.0, 60.0) == (False, None)
    assert has_timed_connection(np.array([1.0]), np.array([]), 0.0, 60.0) == (False, None)


# ---------------------------------------------------------------------------
# compute_transfers + build_output_tables (small synthetic network)
# ---------------------------------------------------------------------------


def _routes_table() -> pd.DataFrame:
    routes = pd.DataFrame(
        {
            "groute_id": ["f::A", "f::B", "f::C"],
            "feed": ["f", "f", "f"],
            "route_id": ["A", "B", "C"],
            "route_short_name": ["A", "B", "C"],
            "route_long_name": ["", "", ""],
            "route_label": ["A", "B", "C"],
            "is_target": [True, False, False],
        }
    ).set_index("groute_id")
    return routes


def _stops_gdf() -> pd.DataFrame:
    # S1 (route A) and S2 (route B) are 100 m apart; S3 (route C) is 2 km away.
    return pd.DataFrame(
        {
            "gstop_id": ["f::S1", "f::S2", "f::S3"],
            "x": [0.0, 100.0, 2000.0],
            "y": [0.0, 0.0, 0.0],
        }
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gstop_id": ["f::S1", "f::S2", "f::S3"],
            "groute_id": ["f::A", "f::B", "f::C"],
            "feed": ["f", "f", "f"],
            "arrival_sec": [28800.0, 29100.0, 29100.0],
            "departure_sec": [28800.0, 29100.0, 29100.0],
        }
    )


def test_compute_transfers_finds_timed_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 30.0)
    monkeypatch.setattr(rtc, "WALK_SPEED_MPH", 3.0)

    results = compute_transfers(_stops_gdf(), _events(), _routes_table(), radius_m=400.0)

    # A can transfer to B (100 m, 5 min later) but not C (2 km away).
    assert set(results.keys()) == {"f::A"}
    assert set(results["f::A"].keys()) == {"f::B"}
    assert results["f::A"]["f::B"]["min_wait_seconds"] == pytest.approx(300.0)


def test_compute_transfers_respects_max_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    # Only a 2-minute wait window: the 5-minute-later B departure is now too late.
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 2.0)
    monkeypatch.setattr(rtc, "WALK_SPEED_MPH", 3.0)

    results = compute_transfers(_stops_gdf(), _events(), _routes_table(), radius_m=400.0)
    assert results == {}


def test_compute_transfers_spatial_only_ignores_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", False)
    results = compute_transfers(_stops_gdf(), _events(), _routes_table(), radius_m=400.0)
    assert set(results["f::A"].keys()) == {"f::B"}


def test_build_output_tables_includes_zero_transfer_targets() -> None:
    routes = _routes_table()
    results = {
        "f::A": {"f::B": {"stop_pairs": 1.0, "min_distance_m": 100.0, "min_wait_seconds": 300.0}}
    }
    summary, detail = build_output_tables(results, routes)

    assert list(summary["route_id"]) == ["A"]
    row = summary.iloc[0]
    assert row["transfer_route_count"] == 1
    assert row["transfer_routes"] == "B"
    assert len(detail) == 1
    assert detail.iloc[0]["min_transfer_wait_min"] == pytest.approx(5.0)
    assert not bool(detail.iloc[0]["cross_feed"])


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_weekday_filter() -> None:
    args = rtc.parse_args([])
    assert args.day == "weekday"


def test_parse_args_overrides() -> None:
    args = rtc.parse_args(
        ["--gtfs-dirs", "feed_a", "feed_b", "--output-dir", "out", "--day", "all"]
    )
    assert args.gtfs_dirs == ["feed_a", "feed_b"]
    assert str(args.output_dir) == "out"
    assert args.day == "all"


def test_parse_args_rejects_unknown_tokens() -> None:
    # Strict parsing: a stray flag (e.g. a mismatched orchestrator cmd template)
    # fails loudly with argparse's exit code 2 instead of being silently ignored.
    with pytest.raises(SystemExit) as excinfo:
        rtc.parse_args(["--input-dir", "somewhere", "--gtfs-dirs", "feed_a"])
    assert excinfo.value.code == 2
