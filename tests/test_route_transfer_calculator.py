from __future__ import annotations

import zipfile
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
    service_ids_active_on_day,
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
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday\nWK,1,1,1,1,1,0,0\n"
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
