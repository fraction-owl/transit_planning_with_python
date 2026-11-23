from __future__ import annotations

import datetime as dt
import logging
import zipfile
from collections.abc import Sequence
from pathlib import Path

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
    parse_service_date,
    resolve_service_day,
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
# resolve_service_day / parse_service_date
# ---------------------------------------------------------------------------

_DAY_COLS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DAY_AFTER_THANKSGIVING = dt.date(2026, 11, 27)


def _special_day_calendar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekday service "1", and service "4" whose exceptions leave it only Nov 27.

    Service 4 carries Monday, Thursday and Friday flags, so calendar.txt alone
    says it runs alongside service 1. calendar_dates.txt removes every one of
    those dates except the day after Thanksgiving.
    """
    calendar = pd.DataFrame(
        {
            "service_id": ["1", "4"],
            **{
                day: ["1", "1" if day in ("monday", "thursday", "friday") else "0"]
                for day in _DAY_COLS[:5]
            },
            "saturday": ["0", "0"],
            "sunday": ["0", "0"],
            "start_date": ["20261102", "20261102"],
            "end_date": ["20261204", "20261204"],
        }
    )
    removed = [
        day
        for day in (dt.date(2026, 11, 2) + dt.timedelta(days=i) for i in range(33))
        if day.weekday() in (0, 3, 4) and day != _DAY_AFTER_THANKSGIVING
    ]
    calendar_dates = pd.DataFrame(
        {
            "service_id": "4",
            "date": [day.strftime("%Y%m%d") for day in removed],
            "exception_type": "2",
        }
    )
    return calendar, calendar_dates


def _running_on(service_dates: dict[str, set[dt.date]], day: dt.date) -> set[str]:
    return {sid for sid, dates in service_dates.items() if day in dates}


def test_resolve_service_day_applies_calendar_dates_exceptions() -> None:
    calendar, calendar_dates = _special_day_calendar()
    chosen, service_dates = resolve_service_day(calendar, calendar_dates, "weekday")
    assert chosen.weekday() < 5 and chosen != _DAY_AFTER_THANKSGIVING
    assert _running_on(service_dates, chosen) == {"1"}
    assert "4" not in service_dates  # service 4 never runs on or beside a regular weekday


def test_resolve_service_day_explicit_date_keeps_weekday_neighbors() -> None:
    calendar, calendar_dates = _special_day_calendar()
    thanksgiving = _DAY_AFTER_THANKSGIVING - dt.timedelta(days=1)
    # Saturday the 28th is outside the weekday filter; Thursday the 26th is kept.
    assert resolve_service_day(calendar, calendar_dates, "weekday", _DAY_AFTER_THANKSGIVING) == (
        _DAY_AFTER_THANKSGIVING,
        {"1": {thanksgiving, _DAY_AFTER_THANKSGIVING}, "4": {_DAY_AFTER_THANKSGIVING}},
    )


def test_resolve_service_day_neighbors_follow_the_day_filter() -> None:
    calendar = pd.DataFrame(
        {
            "service_id": ["WK", "SAT"],
            **{day: ["1", "0"] for day in _DAY_COLS[:5]},
            "saturday": ["0", "1"],
            "sunday": ["0", "0"],
            "start_date": ["20261102", "20261102"],
            "end_date": ["20261204", "20261204"],
        }
    )
    friday, saturday = dt.date(2026, 11, 20), dt.date(2026, 11, 21)
    _, weekday_only = resolve_service_day(calendar, None, "weekday", friday)
    assert weekday_only == {"WK": {friday - dt.timedelta(days=1), friday}}
    _, any_day = resolve_service_day(calendar, None, None, friday)
    assert any_day == {"WK": {friday - dt.timedelta(days=1), friday}, "SAT": {saturday}}
    # A single day name keeps no neighbors.
    chosen, friday_only = resolve_service_day(calendar, None, "friday")
    assert friday_only == {"WK": {chosen}}


def test_resolve_service_day_reads_calendar_dates_only_feed() -> None:
    mondays = [dt.date(2026, 11, 2) + dt.timedelta(weeks=i) for i in range(5)]
    calendar_dates = pd.DataFrame(
        {
            "service_id": "MON",
            "date": [day.strftime("%Y%m%d") for day in mondays],
            "exception_type": "1",
        }
    )
    chosen, service_dates = resolve_service_day(None, calendar_dates, "weekday")
    assert chosen in mondays
    assert service_dates == {"MON": {chosen}}


def test_resolve_service_day_without_calendar_returns_none() -> None:
    assert resolve_service_day(None, None, "weekday") is None
    assert resolve_service_day(pd.DataFrame(), None, "weekday") is None


def test_resolve_service_day_raises_when_nothing_runs() -> None:
    calendar, calendar_dates = _special_day_calendar()
    with pytest.raises(ValueError, match="No service_id is active"):
        resolve_service_day(calendar, calendar_dates, "weekday", dt.date(2026, 11, 28))
    with pytest.raises(ValueError, match="No active dates"):
        resolve_service_day(calendar, calendar_dates, "saturday")


@pytest.mark.parametrize("text", ["2026-11-27", "20261127", " 2026-11-27 "])
def test_parse_service_date_accepts_both_formats(text: str) -> None:
    assert parse_service_date(text) == _DAY_AFTER_THANKSGIVING


def test_parse_service_date_blank_and_invalid() -> None:
    assert parse_service_date("") is None
    with pytest.raises(ValueError, match="not a date"):
        parse_service_date("11/27/2026")


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


def _events(
    can_alight: tuple[bool, bool, bool] = (True, True, True),
    can_board: tuple[bool, bool, bool] = (True, True, True),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "route_id": ["A", "B", "C"],
            "arrival_sec": [28800.0, 29100.0, 29100.0],
            "departure_sec": [28800.0, 29100.0, 29100.0],
            "can_alight": list(can_alight),
            "can_board": list(can_board),
        }
    )


def test_compute_transfers_finds_timed_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 30.0)
    monkeypatch.setattr(rtc, "WALK_SPEED_MPH", 3.0)

    results = compute_transfers(_stops_gdf(), _events(), _routes_table(), radius_m=400.0)

    # A can transfer to B (100 m, 5 min later) but not C (2 km away).
    assert set(results.keys()) == {"A"}
    assert set(results["A"].keys()) == {"B"}
    assert results["A"]["B"]["min_wait_seconds"] == pytest.approx(300.0)


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
    assert set(results["A"].keys()) == {"B"}


@pytest.mark.parametrize("time_check", [True, False])
@pytest.mark.parametrize(
    ("can_alight", "can_board"),
    [
        ((False, True, True), (True, True, True)),  # no drop-off from A at S1
        ((True, True, True), (True, False, True)),  # no pickup onto B at S2
    ],
)
def test_compute_transfers_requires_drop_off_and_pickup(
    monkeypatch: pytest.MonkeyPatch,
    time_check: bool,
    can_alight: tuple[bool, bool, bool],
    can_board: tuple[bool, bool, bool],
) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", time_check)
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 30.0)
    monkeypatch.setattr(rtc, "WALK_SPEED_MPH", 3.0)

    events = _events(can_alight=can_alight, can_board=can_board)
    results = compute_transfers(_stops_gdf(), events, _routes_table(), radius_m=400.0)
    assert results == {}


def test_build_stop_events_reads_pickup_and_drop_off_types() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"] * 5,
            "stop_id": ["S1", "S2", "S3", "S4", "S5"],
            "arrival_time": ["08:00:00"] * 5,
            "departure_time": ["08:00:00"] * 5,
            # 1 = not available; blank, 0, 2 (phone) and 3 (ask driver) still allow it.
            "pickup_type": ["0", "1", None, "2", "3"],
            "drop_off_type": ["1", "0", "3", None, "2"],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1"], "route_id": ["A"], "service_id": ["WK"]})

    events = rtc._build_stop_events(stop_times, trips, keep_service_ids=None)
    assert events["can_board"].tolist() == [True, False, True, True, True]
    assert events["can_alight"].tolist() == [False, True, True, True, True]

    # Both columns are optional in GTFS; without them every stop allows both.
    bare = rtc._build_stop_events(
        stop_times.drop(columns=["pickup_type", "drop_off_type"]), trips, keep_service_ids=None
    )
    assert bare["can_board"].all()
    assert bare["can_alight"].all()


@pytest.mark.parametrize("time_check", [True, False])
def test_main_skips_connector_stop_with_no_pickup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, time_check: bool
) -> None:
    # Feeder A arrives at the hub at 08:00; connector B leaves it at 08:05 but
    # with pickup_type=1, so no rider can board B there.
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", time_check)
    monkeypatch.setattr(rtc, "TARGET_ROUTE_TOKENS", {"A"})
    feed = tmp_path / "feed"
    feed.mkdir()
    (feed / "stops.txt").write_text(
        "stop_id,stop_lat,stop_lon\nHUB,38.90,-77.00\nA0,38.95,-77.00\nB9,38.85,-77.00\n"
    )
    (feed / "routes.txt").write_text("route_id,route_short_name\nA,A\nB,B\n")
    (feed / "trips.txt").write_text("route_id,service_id,trip_id\nA,WK,TA\nB,WK,TB\n")
    stop_times = (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type\n"
        "TA,07:50:00,07:50:00,A0,1,0,1\n"
        "TA,08:00:00,08:00:00,HUB,2,1,0\n"
        "TB,08:05:00,08:05:00,HUB,1,{hub_pickup},1\n"
        "TB,08:15:00,08:15:00,B9,2,1,0\n"
    )
    (feed / "calendar.txt").write_text(
        f"{_CALENDAR_HEADER},start_date,end_date\nWK,1,1,1,1,1,0,0,{_YEAR}\n"
    )
    argv = ["--gtfs-dir", str(feed), "--output-dir", str(tmp_path / "out")]

    (feed / "stop_times.txt").write_text(stop_times.format(hub_pickup="1"))
    assert rtc.main(argv) == 0
    summary = pd.read_csv(tmp_path / "out" / "route_transfers_summary.csv", keep_default_na=False)
    assert summary["transfer_routes"].tolist() == [""]

    # Control: the same feed with boarding allowed at the hub finds A -> B.
    (feed / "stop_times.txt").write_text(stop_times.format(hub_pickup="0"))
    assert rtc.main(argv) == 0
    summary = pd.read_csv(tmp_path / "out" / "route_transfers_summary.csv", keep_default_na=False)
    assert summary["transfer_routes"].tolist() == ["B"]


def test_build_stop_events_uses_stop_sequence_per_visit(caplog: pytest.LogCaptureFixture) -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["L", "L", "L", "N", "N", "N"],
            # L loops HUB -> X -> HUB. N's sequences sort numerically (2 < 10),
            # and its "x" cannot be placed in the trip at all.
            "stop_id": ["HUB", "X", "HUB", "P", "Q", "R"],
            "stop_sequence": ["1", "2", "3", "2", "10", "x"],
            "arrival_time": ["08:00:00"] * 6,
            "departure_time": ["08:00:00"] * 6,
        }
    )
    trips = pd.DataFrame({"trip_id": ["L", "N"], "route_id": ["A", "B"], "service_id": ["WK"] * 2})

    with caplog.at_level(logging.WARNING):
        events = rtc._build_stop_events(stop_times, trips, keep_service_ids=None)
    # Nobody gets off at a trip's first visit or on at its last.
    assert events["can_alight"].tolist() == [False, True, True, False, True, False]
    assert events["can_board"].tolist() == [True, True, False, True, False, False]
    assert events["service_id"].tolist() == ["WK"] * 6
    assert "1 stop time(s) have a missing or non-numeric stop_sequence" in caplog.text


def test_compute_transfers_only_pairs_services_sharing_a_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    monkeypatch.setattr(rtc, "MAX_TRANSFER_WAIT_MINUTES", 30.0)
    monkeypatch.setattr(rtc, "WALK_SPEED_MPH", 3.0)
    # A arrives at S1 at 08:00 on Mondays. B leaves S2 (100 m away) at 08:05
    # on Tuesdays and at 08:20 on Mondays.
    events = pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S2"],
            "route_id": ["A", "B", "B"],
            "service_id": ["MON", "TUE", "MON"],
            "arrival_sec": [28800.0, 29100.0, 30000.0],
            "departure_sec": [28800.0, 29100.0, 30000.0],
            "can_alight": [True] * 3,
            "can_board": [True] * 3,
        }
    )
    service_dates = {"MON": {dt.date(2026, 11, 16)}, "TUE": {dt.date(2026, 11, 17)}}

    matched = compute_transfers(
        _stops_gdf(), events, _routes_table(), radius_m=400.0, service_dates=service_dates
    )
    assert matched["A"]["B"]["min_wait_seconds"] == pytest.approx(1200.0)  # the Monday 08:20

    # Without service matching, the Tuesday 08:05 departure is (wrongly) caught.
    pooled = compute_transfers(_stops_gdf(), events, _routes_table(), radius_m=400.0)
    assert pooled["A"]["B"]["min_wait_seconds"] == pytest.approx(300.0)


_CALENDAR_HEADER = "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday"
_YEAR = "20260101,20261231"
# A ends at HUB at 08:00; B starts at HUB at 08:05 (the other stops are ~5 km away).
_A_ENDS_B_STARTS = (
    "TA,07:50:00,07:50:00,A0,1\n"
    "TA,08:00:00,08:00:00,HUB,2\n"
    "TB,08:05:00,08:05:00,HUB,1\n"
    "TB,08:15:00,08:15:00,B9,2\n"
)


def _write_ab_feed(
    tmp_path: Path, calendar: str, stop_times: str, calendar_dates: str | None = None
) -> Path:
    """Write a feed where route A runs on service SA and route B on service SB."""
    feed = tmp_path / "feed"
    feed.mkdir()
    (feed / "stops.txt").write_text(
        "stop_id,stop_lat,stop_lon\nHUB,38.90,-77.00\nA0,38.95,-77.00\nB9,38.85,-77.00\n"
    )
    (feed / "routes.txt").write_text("route_id,route_short_name\nA,A\nB,B\n")
    (feed / "trips.txt").write_text("route_id,service_id,trip_id\nA,SA,TA\nB,SB,TB\n")
    (feed / "calendar.txt").write_text(calendar)
    if calendar_dates is not None:
        (feed / "calendar_dates.txt").write_text(calendar_dates)
    (feed / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n" + stop_times
    )
    return feed


def _transfers_from_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calendar: str,
    stop_times: str,
    time_check: bool = True,
    calendar_dates: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", time_check)
    monkeypatch.setattr(rtc, "TARGET_ROUTE_TOKENS", {"A"})
    feed = _write_ab_feed(tmp_path, calendar, stop_times, calendar_dates)
    out = tmp_path / "out"
    assert rtc.main(["--gtfs-dir", str(feed), "--output-dir", str(out), *extra_args]) == 0
    summary = pd.read_csv(out / "route_transfers_summary.csv", keep_default_na=False)
    return summary["transfer_routes"].tolist()


@pytest.mark.parametrize("time_check", [True, False])
@pytest.mark.parametrize(
    ("services", "expected"),
    [
        # Monday-only feeder, Tuesday-only connector: never on the same day.
        (f"SA,1,0,0,0,0,0,0,{_YEAR}\nSB,0,1,0,0,0,0,0,{_YEAR}\n", [""]),
        # Different service_ids that both run Mondays still connect.
        (f"SA,1,0,1,0,1,0,0,{_YEAR}\nSB,1,1,1,1,1,0,0,{_YEAR}\n", ["B"]),
        # The same weekdays, but in date ranges that never overlap.
        ("SA,1,1,1,1,1,0,0,20260101,20260630\nSB,1,1,1,1,1,0,0,20260701,20261231\n", [""]),
        # Both Monday-only; the ranges overlap only on Tuesday, June 30.
        ("SA,1,0,0,0,0,0,0,20260101,20260630\nSB,1,0,0,0,0,0,0,20260630,20261231\n", [""]),
    ],
)
def test_main_pairs_only_services_running_on_the_weekday_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_check: bool,
    services: str,
    expected: list[str],
) -> None:
    calendar = f"{_CALENDAR_HEADER},start_date,end_date\n{services}"
    assert (
        _transfers_from_main(tmp_path, monkeypatch, calendar, _A_ENDS_B_STARTS, time_check)
        == expected
    )


def _special_day_feed_files() -> tuple[str, str]:
    """Return calendar.txt and calendar_dates.txt for a special-day service SA.

    SA's Monday/Thursday/Friday flags overlap SB's weekdays, but exceptions
    leave SA only Friday, Nov 27 (the day after Thanksgiving).
    """
    calendar = (
        f"{_CALENDAR_HEADER},start_date,end_date\n"
        "SA,1,0,0,1,1,0,0,20261102,20261204\n"
        "SB,1,1,1,1,1,0,0,20261102,20261204\n"
    )
    removed = [
        day
        for day in (dt.date(2026, 11, 2) + dt.timedelta(days=i) for i in range(33))
        if day.weekday() in (0, 3, 4) and day != _DAY_AFTER_THANKSGIVING
    ]
    calendar_dates = "service_id,date,exception_type\n" + "".join(
        f"SA,{day:%Y%m%d},2\n" for day in removed
    )
    return calendar, calendar_dates


@pytest.mark.parametrize("time_check", [True, False])
@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        # The representative weekday has only SB, so A never feeds B.
        ((), [""]),
        # On the day after Thanksgiving both services run, and A feeds B.
        (("--service-date", "2026-11-27"), ["B"]),
    ],
)
def test_main_applies_calendar_dates_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_check: bool,
    extra_args: tuple[str, ...],
    expected: list[str],
) -> None:
    calendar, calendar_dates = _special_day_feed_files()
    transfers = _transfers_from_main(
        tmp_path,
        monkeypatch,
        calendar,
        _A_ENDS_B_STARTS,
        time_check,
        calendar_dates=calendar_dates,
        extra_args=extra_args,
    )
    assert transfers == expected


def test_main_reports_calendar_without_date_range(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calendar = f"{_CALENDAR_HEADER}\nSA,1,0,0,0,0,0,0\nSB,1,0,0,0,0,0,0\n"
    feed = _write_ab_feed(tmp_path, calendar, _A_ENDS_B_STARTS)
    with caplog.at_level(logging.ERROR):
        assert rtc.main(["--gtfs-dir", str(feed), "--output-dir", str(tmp_path / "out")]) == 1
    assert "start_date" in caplog.text
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("extra_args", [("--day", "someday"), ("--service-date", "11/27/2026")])
def test_main_rejects_unusable_day_settings(tmp_path: Path, extra_args: tuple[str, ...]) -> None:
    calendar, calendar_dates = _special_day_feed_files()
    feed = _write_ab_feed(tmp_path, calendar, _A_ENDS_B_STARTS, calendar_dates)
    argv = ["--gtfs-dir", str(feed), "--output-dir", str(tmp_path / "out"), *extra_args]
    assert rtc.main(argv) == 2


def test_main_errors_when_no_service_runs_on_the_service_date(tmp_path: Path) -> None:
    calendar, calendar_dates = _special_day_feed_files()
    feed = _write_ab_feed(tmp_path, calendar, _A_ENDS_B_STARTS, calendar_dates)
    argv = ["--gtfs-dir", str(feed), "--output-dir", str(tmp_path / "out")]
    assert rtc.main([*argv, "--service-date", "20261128"]) == 1  # a Saturday


@pytest.mark.parametrize("time_check", [True, False])
def test_main_ignores_feeder_origin_and_connector_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, time_check: bool
) -> None:
    # A starts at HUB at 08:00 and leaves; B ends at HUB at 08:05. Nobody
    # arrives at HUB aboard A or continues aboard B.
    calendar = f"{_CALENDAR_HEADER},start_date,end_date\nSA,1,0,0,0,0,0,0,{_YEAR}\n"
    calendar += f"SB,1,0,0,0,0,0,0,{_YEAR}\n"
    stop_times = (
        "TA,08:00:00,08:00:00,HUB,1\n"
        "TA,08:10:00,08:10:00,A0,2\n"
        "TB,07:55:00,07:55:00,B9,1\n"
        "TB,08:05:00,08:05:00,HUB,2\n"
    )
    assert _transfers_from_main(tmp_path, monkeypatch, calendar, stop_times, time_check) == [""]


@pytest.mark.parametrize(("b_leaves", "expected"), [("08:05:00", ["B"]), ("07:35:00", [""])])
def test_main_loop_lets_riders_off_only_on_final_visit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, b_leaves: str, expected: list[str]
) -> None:
    # A loops HUB -> A0 -> HUB, leaving at 07:30 and returning at 08:00. A
    # 07:35 departure of B follows only A's origin visit, which drops no one.
    calendar = f"{_CALENDAR_HEADER},start_date,end_date\nSA,1,0,0,0,0,0,0,{_YEAR}\n"
    calendar += f"SB,1,0,0,0,0,0,0,{_YEAR}\n"
    stop_times = (
        "TA,07:30:00,07:30:00,HUB,1\n"
        "TA,07:45:00,07:45:00,A0,2\n"
        "TA,08:00:00,08:00:00,HUB,3\n"
        f"TB,{b_leaves},{b_leaves},HUB,1\n"
        "TB,09:00:00,09:00:00,B9,2\n"
    )
    assert _transfers_from_main(tmp_path, monkeypatch, calendar, stop_times) == expected


def _two_route_feed(
    tmp_path: Path, stops: str, stop_times: str, extra_files: dict[str, str]
) -> list[str]:
    """Run main on a weekday A/B feed and return A's transfer_routes."""
    feed = tmp_path / "feed"
    feed.mkdir(parents=True)
    (feed / "stops.txt").write_text(stops)
    (feed / "routes.txt").write_text("route_id,route_short_name\nA,A\nB,B\n")
    (feed / "trips.txt").write_text("route_id,service_id,trip_id\nA,WK,TA\nB,WK,TB\n")
    (feed / "calendar.txt").write_text(
        f"{_CALENDAR_HEADER},start_date,end_date\nWK,1,1,1,1,1,0,0,{_YEAR}\n"
    )
    (feed / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n" + stop_times
    )
    for name, text in extra_files.items():
        (feed / name).write_text(text)
    out = tmp_path / "out"
    assert rtc.main(["--gtfs-dir", str(feed), "--output-dir", str(out)]) == 0
    summary = pd.read_csv(out / "route_transfers_summary.csv", keep_default_na=False)
    return summary["transfer_routes"].tolist()


# A alights at HA at 08:00; B boards at HB, about 11 m north, at 08:05.
_NEAR_STOPS = (
    "stop_id,stop_lat,stop_lon,parent_station\n"
    "HA,38.90000,-77.00000,HUB\nHB,38.90010,-77.00000,HUB\n"
    "HUB,38.90005,-77.00000,\nA0,38.95,-77.00,\nB9,38.85,-77.00,\n"
)
_A_AT_HA_B_AT_HB = (
    "TA,07:50:00,07:50:00,A0,1\n"
    "TA,08:00:00,08:00:00,HA,2\n"
    "TB,08:05:00,08:05:00,HB,1\n"
    "TB,08:15:00,08:15:00,B9,2\n"
)


@pytest.mark.parametrize("time_check", [True, False])
@pytest.mark.parametrize(
    ("transfers", "expected"),
    [
        (None, ["B"]),  # control: no transfers.txt
        ("from_stop_id,to_stop_id,transfer_type\nHA,HB,3\n", [""]),
        ("from_stop_id,to_stop_id,transfer_type\nHUB,HUB,3\n", [""]),  # station covers children
        ("from_stop_id,to_stop_id,transfer_type\nHB,HA,3\n", ["B"]),  # other direction only
        ("from_stop_id,to_stop_id,transfer_type\nHA,HB,2\n", ["B"]),  # not a prohibition
    ],
)
def test_main_skips_transfers_marked_not_possible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_check: bool,
    transfers: str | None,
    expected: list[str],
) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", time_check)
    monkeypatch.setattr(rtc, "TARGET_ROUTE_TOKENS", {"A"})
    extra = {"transfers.txt": transfers} if transfers else {}
    assert _two_route_feed(tmp_path, _NEAR_STOPS, _A_AT_HA_B_AT_HB, extra) == expected


def test_transfer_rules_follow_routes_stations_and_specificity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stops = pd.DataFrame({"stop_id": ["HA", "HB", "HUB"], "parent_station": ["HUB", "HUB", None]})
    transfers = pd.DataFrame(
        {
            "from_stop_id": ["HUB", "HA", "HA", "HA"],
            "to_stop_id": ["HUB", "HB", "HB", "HB"],
            "from_route_id": ["", "A", "", ""],
            "to_route_id": ["", "B", "C", ""],
            "from_trip_id": ["", "", "", "T1"],
            "transfer_type": ["3", "0", "3", "0"],
        }
    )
    with caplog.at_level(logging.WARNING):
        rules = rtc.build_transfer_rules(transfers, stops)
    assert "1 row(s) name a from_trip_id or to_trip_id" in caplog.text
    pair = rules[("HA", "HB")]
    # The station-wide ban applies to every route pair the A -> B override does not name.
    assert rtc.transfer_prohibited(pair, "A", "C")
    assert rtc.transfer_prohibited(pair, "D", "B")
    assert not rtc.transfer_prohibited(pair, "A", "B")
    assert rtc.transfer_prohibited(rules[("HB", "HA")], "A", "B")
    assert rtc.build_transfer_rules(None, stops) == {}


def test_main_counts_exact_frequency_departures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B's template leaves HUB at 07:05 and runs every 10 minutes; A arrives at 08:00.
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    monkeypatch.setattr(rtc, "TARGET_ROUTE_TOKENS", {"A"})
    stops = "stop_id,stop_lat,stop_lon\nHUB,38.90,-77.00\nA0,38.95,-77.00\nB9,38.85,-77.00\n"
    stop_times = (
        "TA,07:50:00,07:50:00,A0,1\n"
        "TA,08:00:00,08:00:00,HUB,2\n"
        "TB,07:05:00,07:05:00,HUB,1\n"
        "TB,07:15:00,07:15:00,B9,2\n"
    )
    header = "trip_id,start_time,end_time,headway_secs,exact_times\n"
    exact = {"frequencies.txt": header + "TB,07:05:00,09:00:00,600,1\n"}
    assert _two_route_feed(tmp_path / "exact", stops, stop_times, exact) == ["B"]
    # The window must reach the departure: runs stop before end_time.
    early = {"frequencies.txt": header + "TB,07:05:00,08:05:00,600,1\n"}
    assert _two_route_feed(tmp_path / "early", stops, stop_times, early) == [""]


def test_expand_exact_frequencies_shifts_every_stop() -> None:
    events = pd.DataFrame(
        {
            "trip_id": ["TB", "TB", "TX"],
            "arrival_sec": [300.0, 900.0, 100.0],
            "departure_sec": [300.0, 900.0, 100.0],
        }
    )
    frequencies = pd.DataFrame(
        {
            "trip_id": ["TB"],
            "start_time": ["07:05:00"],
            "end_time": ["07:35:00"],  # exclusive: runs at 07:05, 07:15, 07:25
            "headway_secs": ["600"],
            "exact_times": ["1"],
        }
    )
    out = rtc._expand_exact_frequencies(events, frequencies)
    tb = out[out["trip_id"] == "TB"].sort_values("departure_sec")
    start = 7 * 3600 + 5 * 60
    first_stop = [start, start + 600, start + 1200]
    assert sorted(tb["departure_sec"].tolist()) == sorted(
        first_stop + [t + 600 for t in first_stop]
    )
    assert out.loc[out["trip_id"] == "TX", "departure_sec"].tolist() == [100.0]
    headway_only = frequencies.assign(exact_times="0")
    pd.testing.assert_frame_equal(rtc._expand_exact_frequencies(events, headway_only), events)


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
    assert args.service_date == ""


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


def test_read_feed_zipped_feed_matches_folder(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"
    with zipfile.ZipFile(fixture) as zf:
        zf.extractall(tmp_path / "extracted")
    folder = tmp_path / "extracted" / "dc"

    tables_folder, calendar_folder = rtc._read_feed(str(folder))
    tables_zip, calendar_zip = rtc._read_feed(str(fixture))

    assert tables_zip.keys() == tables_folder.keys()
    for key, frame in tables_folder.items():
        pd.testing.assert_frame_equal(tables_zip[key], frame)
    assert calendar_folder is not None and calendar_zip is not None
    pd.testing.assert_frame_equal(calendar_zip, calendar_folder)


def test_read_feed_missing_required_file_raises(tmp_path: Path) -> None:
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("stops.txt", "stop_id\n1\n")
    with pytest.raises(FileNotFoundError, match="routes.txt"):
        rtc._read_feed(str(zipped))


# ---------------------------------------------------------------------------
# transfers.txt minimum transfer times
# ---------------------------------------------------------------------------

_HUB_STOPS = "stop_id,stop_lat,stop_lon\nHUB,38.90,-77.00\nA0,38.95,-77.00\nB9,38.85,-77.00\n"


@pytest.mark.parametrize(
    ("b_leaves", "transfers", "expected"),
    [
        ("08:02:00", None, ["B"]),  # control: a 2-minute transfer at a shared stop
        ("08:02:00", "HUB,HUB,2,300", [""]),  # transfers.txt requires 5 minutes
        ("08:05:00", "HUB,HUB,2,300", ["B"]),
        ("08:02:00", "HUB,HUB,2,60", ["B"]),
        ("08:02:00", "HUB,HUB,0,300", ["B"]),  # min_transfer_time applies to type 2 only
    ],
)
def test_main_applies_min_transfer_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    b_leaves: str,
    transfers: str | None,
    expected: list[str],
) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    monkeypatch.setattr(rtc, "TARGET_ROUTE_TOKENS", {"A"})
    stop_times = (
        "TA,07:50:00,07:50:00,A0,1\n"
        "TA,08:00:00,08:00:00,HUB,2\n"
        f"TB,{b_leaves},{b_leaves},HUB,1\n"
        "TB,08:20:00,08:20:00,B9,2\n"
    )
    header = "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n"
    extra = {"transfers.txt": header + transfers + "\n"} if transfers else {}
    assert _two_route_feed(tmp_path, _HUB_STOPS, stop_times, extra) == expected


def test_governing_transfer_rule_returns_most_specific_minimum() -> None:
    stops = pd.DataFrame({"stop_id": ["HA", "HB"]})
    transfers = pd.DataFrame(
        {
            "from_stop_id": ["HA", "HA", "HA", "HA"],
            "to_stop_id": ["HB", "HB", "HB", "HB"],
            "from_route_id": ["", "A", "C", "C"],
            "to_route_id": ["", "B", "", ""],
            "transfer_type": ["2", "0", "2", "2"],
            "min_transfer_time": ["300", "", "120", "240"],
        }
    )
    pair = rtc.build_transfer_rules(transfers, stops)[("HA", "HB")]
    assert rtc.governing_transfer_rule(pair, "D", "B") == (False, 300.0)  # stop-wide rule
    assert rtc.governing_transfer_rule(pair, "A", "B") == (False, 0.0)  # route-pair override
    assert rtc.governing_transfer_rule(pair, "C", "B") == (False, 240.0)  # tie: longer minimum


# ---------------------------------------------------------------------------
# Connections across a service-day boundary
# ---------------------------------------------------------------------------

# A arrives at HUB at 24:05 on its service date; B leaves HUB at 00:10 on its own.
_A_AFTER_MIDNIGHT = (
    "TA,23:55:00,23:55:00,A0,1\n"
    "TA,24:05:00,24:05:00,HUB,2\n"
    "TB,00:10:00,00:10:00,HUB,1\n"
    "TB,00:20:00,00:20:00,B9,2\n"
)
# A arrives at 00:05 on its service date; B leaves at 24:10 on the date before.
_B_BEFORE_MIDNIGHT = (
    "TA,23:55:00,23:55:00,A0,1\n"
    "TA,00:05:00,00:05:00,HUB,2\n"
    "TB,24:10:00,24:10:00,HUB,1\n"
    "TB,24:20:00,24:20:00,B9,2\n"
)


@pytest.mark.parametrize(
    ("services", "stop_times", "expected"),
    [
        # Monday-only A, Tuesday-only B: the representative weekday is a Tuesday
        # (Monday holidays are skipped), when A does not run. See the explicit
        # Monday below.
        ("SA,1,0,0,0,0,0,0\nSB,0,1,0,0,0,0,0", _A_AFTER_MIDNIGHT, [""]),
        # One weekday service: Monday's 24:05 arrival catches Tuesday's 00:10.
        ("SA,1,1,1,1,1,0,0\nSB,1,1,1,1,1,0,0", _A_AFTER_MIDNIGHT, ["B"]),
        # Monday at 24:05 does not reach a Wednesday-only 00:10.
        ("SA,1,0,0,0,0,0,0\nSB,0,0,1,0,0,0,0", _A_AFTER_MIDNIGHT, [""]),
        # Saturday service is outside the weekday filter.
        ("SA,0,0,0,0,1,0,0\nSB,0,0,0,0,0,1,0", _A_AFTER_MIDNIGHT, [""]),
        # Tuesday 00:05 arrival catches Monday's 24:10 (also Tuesday 00:10).
        ("SA,0,1,0,0,0,0,0\nSB,1,0,0,0,0,0,0", _B_BEFORE_MIDNIGHT, ["B"]),
    ],
)
def test_main_connects_across_service_day_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    services: str,
    stop_times: str,
    expected: list[str],
) -> None:
    rows = "".join(f"{line},{_YEAR}\n" for line in services.splitlines())
    calendar = f"{_CALENDAR_HEADER},start_date,end_date\n{rows}"
    assert _transfers_from_main(tmp_path, monkeypatch, calendar, stop_times) == expected


@pytest.mark.parametrize(
    ("services", "extra_args", "expected"),
    [
        # Monday trip at 24:05 (Tuesday 00:05) catches the Tuesday 00:10 trip.
        ("SA,1,0,0,0,0,0,0\nSB,0,1,0,0,0,0,0", ("--service-date", "2026-11-16"), ["B"]),
        # On the Tuesday, A does not run, so nothing feeds B.
        ("SA,1,0,0,0,0,0,0\nSB,0,1,0,0,0,0,0", ("--service-date", "2026-11-17"), [""]),
        # Friday night onto Saturday is outside the weekday filter...
        ("SA,0,0,0,0,1,0,0\nSB,0,0,0,0,0,1,0", ("--service-date", "2026-11-20"), [""]),
        # ...and counts once no day filter limits the neighboring days.
        (
            "SA,0,0,0,0,1,0,0\nSB,0,0,0,0,0,1,0",
            ("--service-date", "2026-11-20", "--day", "all"),
            ["B"],
        ),
    ],
)
def test_main_connects_across_midnight_from_a_service_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    services: str,
    extra_args: tuple[str, ...],
    expected: list[str],
) -> None:
    rows = "".join(f"{line},{_YEAR}\n" for line in services.splitlines())
    calendar = f"{_CALENDAR_HEADER},start_date,end_date\n{rows}"
    transfers = _transfers_from_main(
        tmp_path, monkeypatch, calendar, _A_AFTER_MIDNIGHT, extra_args=extra_args
    )
    assert transfers == expected


def test_best_connection_reports_wait_across_midnight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "ENABLE_TIME_CHECK", True)
    feasible, wait = rtc._best_connection(
        {"MON": np.array([24 * 3600.0 + 300.0])},  # 24:05
        {"TUE": np.array([600.0])},  # 00:10
        lambda a, b, offset: offset == 1,
        0.0,
        1800.0,
    )
    assert feasible
    assert wait == pytest.approx(300.0)
