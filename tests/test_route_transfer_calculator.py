from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.gtfs_exports.route_transfer_calculator as rtc
from scripts.gtfs_exports.route_transfer_calculator import (
    build_output_tables,
    compatible_service_ids,
    compute_transfers,
    distance_to_meters,
    has_timed_connection,
    parse_gtfs_time,
    weekdays_for_filter,
)

GTFS_BASIC = Path(__file__).parent / "fixtures" / "gtfs_basic"

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
# weekdays_for_filter / compatible_service_ids
# ---------------------------------------------------------------------------

MON_JAN_5 = dt.date(2026, 1, 5)
FRI_JAN_9 = dt.date(2026, 1, 9)
SAT_JAN_10 = dt.date(2026, 1, 10)


def test_weekdays_for_filter() -> None:
    assert weekdays_for_filter("weekday") == {0, 1, 2, 3, 4}
    assert weekdays_for_filter("Friday") == {4}
    assert weekdays_for_filter(None) == set(range(7))
    with pytest.raises(ValueError):
        weekdays_for_filter("someday")


def test_compatible_service_ids_needs_a_common_date() -> None:
    active = {
        "MON": {MON_JAN_5},
        "FRI": {FRI_JAN_9},
        "DAILY": {MON_JAN_5, FRI_JAN_9, SAT_JAN_10},
        "SAT": {SAT_JAN_10},
    }
    weekday = compatible_service_ids(active, weekdays_for_filter("weekday"))
    # Monday-only and Friday-only service never run on the same date.
    assert weekday["MON"] == {"MON", "DAILY"}
    assert weekday["FRI"] == {"FRI", "DAILY"}
    # Saturday's common date with DAILY is not a weekday.
    assert "SAT" not in weekday
    assert compatible_service_ids(active, weekdays_for_filter(None))["SAT"] == {"SAT", "DAILY"}


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


def _routes_table(agencies: tuple[str, str, str] = ("1", "1", "1")) -> pd.DataFrame:
    routes = pd.DataFrame(
        {
            "route_id": ["A", "B", "C"],
            "agency_id": list(agencies),
            "route_short_name": ["A", "B", "C"],
            "route_long_name": ["", "", ""],
            "route_label": ["A", "B", "C"],
            "is_target": [True, False, False],
        }
    ).set_index("route_id", drop=False)
    return routes


def _stops_gdf() -> pd.DataFrame:
    # S1 (route A) and S2 (route B) are 100 m apart; S3 (route C) is 2 km away.
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "x": [0.0, 100.0, 2000.0],
            "y": [0.0, 0.0, 0.0],
        }
    )


def _events(services: tuple[str, str, str] = ("WK", "WK", "WK")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "route_id": ["A", "B", "C"],
            "service_id": list(services),
            "arrival_sec": [28800.0, 29100.0, 29100.0],
            "departure_sec": [28800.0, 29100.0, 29100.0],
        }
    )


WK_ONLY = {"WK": {"WK"}}


def test_compute_transfers_finds_timed_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 30.0)
    monkeypatch.setattr(rtc, "WALK_SPEED_MPH", 3.0)

    results = compute_transfers(
        _stops_gdf(), _events(), _routes_table(), radius_m=400.0, compatible=WK_ONLY
    )

    # A can transfer to B (100 m, 5 min later) but not C (2 km away).
    assert set(results.keys()) == {"A"}
    assert set(results["A"].keys()) == {"B"}
    assert results["A"]["B"]["min_wait_seconds"] == pytest.approx(300.0)


def test_compute_transfers_respects_max_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    # Only a 2-minute wait window: the 5-minute-later B departure is now too late.
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 2.0)
    monkeypatch.setattr(rtc, "WALK_SPEED_MPH", 3.0)

    results = compute_transfers(
        _stops_gdf(), _events(), _routes_table(), radius_m=400.0, compatible=WK_ONLY
    )
    assert results == {}


def test_compute_transfers_spatial_only_ignores_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", False)
    results = compute_transfers(
        _stops_gdf(), _events(), _routes_table(), radius_m=400.0, compatible=WK_ONLY
    )
    assert set(results["A"].keys()) == {"B"}


@pytest.mark.parametrize("time_check", [True, False])
def test_compute_transfers_needs_services_on_a_common_date(
    monkeypatch: pytest.MonkeyPatch, time_check: bool
) -> None:
    # A arrives under Monday-only service and B departs under Friday-only service.
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", time_check)
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 30.0)
    compatible = {"MON": {"MON"}, "FRI": {"FRI"}}
    events = _events(services=("MON", "FRI", "FRI"))

    results = compute_transfers(
        _stops_gdf(), events, _routes_table(), radius_m=400.0, compatible=compatible
    )
    assert results == {}


def test_build_output_tables_includes_zero_transfer_targets() -> None:
    routes = _routes_table()
    results = {"A": {"B": {"stop_pairs": 1.0, "min_distance_m": 100.0, "min_wait_seconds": 300.0}}}
    summary, detail = build_output_tables(results, routes)

    assert list(summary["route_id"]) == ["A"]
    row = summary.iloc[0]
    assert row["transfer_route_count"] == 1
    assert row["transfer_routes"] == "B"
    assert len(detail) == 1
    assert detail.iloc[0]["min_transfer_wait_min"] == pytest.approx(5.0)
    assert not bool(detail.iloc[0]["cross_agency"])


def test_build_output_tables_prefixes_other_agency_connectors() -> None:
    # In a merged multi-agency feed, B belongs to another agency than target A.
    routes = _routes_table(agencies=("FFX", "WMATA", "FFX"))
    results = {"A": {"B": {"stop_pairs": 2.0, "min_distance_m": 0.0, "min_wait_seconds": 60.0}}}
    summary, detail = build_output_tables(results, routes)

    assert summary.iloc[0]["agency_id"] == "FFX"
    assert summary.iloc[0]["transfer_routes"] == "WMATA:B"
    row = detail.iloc[0]
    assert (row["target_agency_id"], row["connector_agency_id"]) == ("FFX", "WMATA")
    assert bool(row["cross_agency"])


def test_build_route_table_filters_targets_by_agency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "TARGET_AGENCY_IDS", {"FFX"})
    routes = pd.DataFrame(
        {"route_id": ["1", "2"], "agency_id": ["FFX", "WMATA"], "route_short_name": ["1", "2"]}
    )
    table = rtc._build_route_table(routes)
    assert table["is_target"].tolist() == [True, False]
    assert table.loc["1", "route_label"] == "1"


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_weekday_filter() -> None:
    args = rtc.parse_args([])
    assert args.day == "weekday"


def test_parse_args_overrides() -> None:
    args = rtc.parse_args(["--gtfs-dir", "merged_feed", "--output-dir", "out", "--day", "all"])
    assert args.gtfs_dir == "merged_feed"
    assert str(args.output_dir) == "out"
    assert args.day == "all"


def test_parse_args_rejects_unknown_tokens() -> None:
    # Strict parsing: a stray flag (e.g. a mismatched orchestrator cmd template)
    # fails loudly with argparse's exit code 2 instead of being silently ignored.
    with pytest.raises(SystemExit) as excinfo:
        rtc.parse_args(["--input-dir", "somewhere", "--gtfs-dir", "feed_a"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# main — full pipeline against gtfs_basic
# ---------------------------------------------------------------------------


def test_main_end_to_end_on_gtfs_basic(tmp_path: Path) -> None:
    assert rtc.main(["--gtfs-dir", str(GTFS_BASIC), "--output-dir", str(tmp_path)]) == 0

    # Only shared stops fall inside the 0.25-mile walk. At them, R2 leaves
    # S3-S5 25 minutes after R1 arrives; every other pairing waits 35+ minutes.
    summary = pd.read_csv(tmp_path / "route_transfers_summary.csv", keep_default_na=False)
    assert summary["route_id"].tolist() == ["R1", "R2", "R3"]
    assert summary["transfer_route_count"].tolist() == [1, 0, 0]
    assert summary["transfer_routes"].tolist() == ["R2", "", ""]

    detail = pd.read_csv(tmp_path / "route_transfers_detail.csv")
    assert len(detail) == 1
    row = detail.iloc[0]
    assert (row["target_route_id"], row["connector_route_id"]) == ("R1", "R2")
    assert row["qualifying_stop_pairs"] == 3  # S3, S4, S5
    assert row["nearest_walk_distance_m"] == 0.0
    assert row["min_transfer_wait_min"] == 25.0


# ---------------------------------------------------------------------------
# main — service dates from calendar.txt / calendar_dates.txt
# ---------------------------------------------------------------------------

_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _calendar_row(service_id: str, days: set[str], start: str, end: str) -> dict[str, str]:
    row = {"service_id": service_id, "start_date": start, "end_date": end}
    row.update({day: "1" if day in days else "0" for day in _DAYS})
    return row


def _write_two_route_feed(
    folder: Path,
    calendar: list[dict[str, str]] | None,
    calendar_dates: list[dict[str, str]] | None = None,
) -> Path:
    """Route A arrives at S1 at 08:00 (service SA); route B leaves S2, ~10 m away, at 08:05 (SB)."""
    folder.mkdir()
    pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["S1", "S2"],
            "stop_lat": ["38.90000", "38.90009"],
            "stop_lon": ["-77.03000", "-77.03000"],
        }
    ).to_csv(folder / "stops.txt", index=False)
    pd.DataFrame(
        {"route_id": ["A", "B"], "route_short_name": ["A", "B"], "route_type": ["3", "3"]}
    ).to_csv(folder / "routes.txt", index=False)
    pd.DataFrame(
        {"route_id": ["A", "B"], "service_id": ["SA", "SB"], "trip_id": ["TA", "TB"]}
    ).to_csv(folder / "trips.txt", index=False)
    pd.DataFrame(
        {
            "trip_id": ["TA", "TB"],
            "arrival_time": ["08:00:00", "08:05:00"],
            "departure_time": ["08:00:00", "08:05:00"],
            "stop_id": ["S1", "S2"],
            "stop_sequence": ["1", "1"],
        }
    ).to_csv(folder / "stop_times.txt", index=False)
    if calendar is not None:
        pd.DataFrame(calendar).to_csv(folder / "calendar.txt", index=False)
    if calendar_dates is not None:
        pd.DataFrame(calendar_dates).to_csv(folder / "calendar_dates.txt", index=False)
    return folder


def _a_transfers(tmp_path: Path, feed: Path, day: str = "weekday") -> str:
    out = tmp_path / "out"
    assert rtc.main(["--gtfs-dir", str(feed), "--output-dir", str(out), "--day", day]) == 0
    summary = pd.read_csv(out / "route_transfers_summary.csv", keep_default_na=False)
    return str(summary.set_index("route_id").loc["A", "transfer_routes"])


@pytest.mark.parametrize("day", ["weekday", "all"])
def test_main_monday_only_arrival_never_meets_friday_only_departure(
    tmp_path: Path, day: str
) -> None:
    feed = _write_two_route_feed(
        tmp_path / "feed",
        [
            _calendar_row("SA", {"monday"}, "20260105", "20260130"),
            _calendar_row("SB", {"friday"}, "20260105", "20260130"),
        ],
    )
    assert _a_transfers(tmp_path, feed, day) == ""


def test_main_same_weekday_service_does_transfer(tmp_path: Path) -> None:
    feed = _write_two_route_feed(
        tmp_path / "feed",
        [
            _calendar_row("SA", {"monday"}, "20260105", "20260130"),
            _calendar_row("SB", {"monday", "friday"}, "20260105", "20260130"),
        ],
    )
    assert _a_transfers(tmp_path, feed) == "B"


def test_main_respects_calendar_date_ranges(tmp_path: Path) -> None:
    # Both run on Mondays, but SA ends in January and SB starts in February.
    feed = _write_two_route_feed(
        tmp_path / "feed",
        [
            _calendar_row("SA", {"monday"}, "20260105", "20260131"),
            _calendar_row("SB", {"monday"}, "20260201", "20260228"),
        ],
    )
    assert _a_transfers(tmp_path, feed) == ""


def test_main_applies_calendar_dates_additions(tmp_path: Path) -> None:
    # Friday-only SB is added on Monday 2026-01-12, a date Monday-only SA also runs.
    feed = _write_two_route_feed(
        tmp_path / "feed",
        [
            _calendar_row("SA", {"monday"}, "20260105", "20260130"),
            _calendar_row("SB", {"friday"}, "20260105", "20260130"),
        ],
        [{"service_id": "SB", "date": "20260112", "exception_type": "1"}],
    )
    assert _a_transfers(tmp_path, feed) == "B"


def test_main_applies_calendar_dates_removals(tmp_path: Path) -> None:
    # The only Monday both run (2026-01-05) is removed from SB.
    feed = _write_two_route_feed(
        tmp_path / "feed",
        [
            _calendar_row("SA", {"monday"}, "20260105", "20260109"),
            _calendar_row("SB", {"monday"}, "20260105", "20260109"),
        ],
        [{"service_id": "SB", "date": "20260105", "exception_type": "2"}],
    )
    assert _a_transfers(tmp_path, feed) == ""


def test_main_reads_calendar_dates_only_feed(tmp_path: Path) -> None:
    feed = _write_two_route_feed(
        tmp_path / "feed",
        None,
        [
            {"service_id": "SA", "date": "20260105", "exception_type": "1"},
            {"service_id": "SB", "date": "20260105", "exception_type": "1"},
        ],
    )
    assert _a_transfers(tmp_path, feed) == "B"


def test_main_rejects_feed_without_any_calendar(tmp_path: Path) -> None:
    feed = _write_two_route_feed(tmp_path / "feed", None)
    out = tmp_path / "out"
    assert rtc.main(["--gtfs-dir", str(feed), "--output-dir", str(out)]) == 1
    assert not (out / "route_transfers_summary.csv").exists()


def test_main_rejects_unknown_day(tmp_path: Path) -> None:
    feed = _write_two_route_feed(tmp_path / "feed", None)
    assert rtc.main(["--gtfs-dir", str(feed), "--output-dir", str(tmp_path), "--day", "x"]) == 2
