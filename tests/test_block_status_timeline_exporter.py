from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

import scripts.gtfs_exports.block_status_timeline_exporter as mod
from scripts.gtfs_exports.block_status_timeline_exporter import (
    _merge_and_filter_data,
    apply_bay_overrides,
    assumptions_text,
    fill_stop_ids_for_dwell_layover_loading,
    find_cluster,
    gap_status,
    get_status_for_minute,
    mark_first_and_last_stops,
    minutes_to_hhmm,
    parse_time_to_minutes,
    process_block,
    resolve_service_ids,
    row_for_inactive,
    run_output_folder,
    validate_folders,
)

# ---------------------------------------------------------------------------
# parse_time_to_minutes
# ---------------------------------------------------------------------------


def test_parse_time_to_minutes_hhmmss() -> None:
    assert parse_time_to_minutes("07:05:00") == 425


def test_parse_time_to_minutes_hhmm_no_seconds() -> None:
    assert parse_time_to_minutes("07:05") == 425


def test_parse_time_to_minutes_past_midnight() -> None:
    assert parse_time_to_minutes("26:30:00") == 1590


def test_parse_time_to_minutes_seconds_rounded() -> None:
    assert parse_time_to_minutes("00:01:30") == 1


# ---------------------------------------------------------------------------
# minutes_to_hhmm
# ---------------------------------------------------------------------------


def test_minutes_to_hhmm_basic() -> None:
    assert minutes_to_hhmm(425) == "07:05"


def test_minutes_to_hhmm_zero() -> None:
    assert minutes_to_hhmm(0) == "00:00"


def test_minutes_to_hhmm_past_midnight() -> None:
    assert minutes_to_hhmm(1590) == "26:30"


# ---------------------------------------------------------------------------
# find_cluster
# ---------------------------------------------------------------------------

_CLUSTERS = [
    {"name": "Hub", "stops": ["100", "101"]},
    {"name": "Terminal", "stops": ["200"]},
]


def test_find_cluster_found() -> None:
    assert find_cluster("100", _CLUSTERS) == "Hub"


def test_find_cluster_second_cluster() -> None:
    assert find_cluster("200", _CLUSTERS) == "Terminal"


def test_find_cluster_not_found() -> None:
    assert find_cluster("999", _CLUSTERS) is None


# ---------------------------------------------------------------------------
# validate_folders
# ---------------------------------------------------------------------------


def test_validate_folders_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        validate_folders(str(tmp_path / "no_such_dir"), str(tmp_path / "out"))


def test_validate_folders_creates_output(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out" / "nested"
    validate_folders(str(src), str(out))
    assert out.is_dir()


# ---------------------------------------------------------------------------
# mark_first_and_last_stops
# ---------------------------------------------------------------------------


def _make_stop_times() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "stop_sequence": [1, 2, 3],
            "stop_id": ["S1", "S2", "S3"],
        }
    )


def test_mark_first_and_last_stops_returns_dataframe() -> None:
    df = mark_first_and_last_stops(_make_stop_times())
    assert isinstance(df, pd.DataFrame)


def test_mark_first_and_last_stops_first_flag() -> None:
    df = mark_first_and_last_stops(_make_stop_times())
    assert df.loc[df["stop_sequence"] == 1, "is_first_stop"].all()


def test_mark_first_and_last_stops_last_flag() -> None:
    df = mark_first_and_last_stops(_make_stop_times())
    assert df.loc[df["stop_sequence"] == 3, "is_last_stop"].all()


def test_mark_first_and_last_stops_middle_neither() -> None:
    df = mark_first_and_last_stops(_make_stop_times())
    middle = df[df["stop_sequence"] == 2]
    assert not middle["is_first_stop"].any()
    assert not middle["is_last_stop"].any()


# ---------------------------------------------------------------------------
# get_status_for_minute
# ---------------------------------------------------------------------------

# stop_info tuple: (arr, dep, stop_id, stop_name, trip_id, is_first, is_last, stop_seq, t_val)

_SEQ_SINGLE = [
    (420, 425, "S1", "Main St", "T1", True, False, 1, 0),
    (430, 435, "S2", "Oak Ave", "T1", False, True, 2, 0),
]


def test_get_status_for_minute_empty_sequence() -> None:
    status, *_ = get_status_for_minute(420, [])
    assert status == "EMPTY"


def test_get_status_for_minute_depart() -> None:
    # DEPART fires when minute == departure_time and stop is the first stop.
    # First stop has arr=420, dep=425, is_first=True.
    status, *_ = get_status_for_minute(425, _SEQ_SINGLE)
    assert status == "DEPART"


def test_get_status_for_minute_arrive() -> None:
    status, *_ = get_status_for_minute(430, _SEQ_SINGLE)
    assert status == "ARRIVE"


def test_get_status_for_minute_traveling() -> None:
    status, *_ = get_status_for_minute(427, _SEQ_SINGLE)
    assert status == "TRAVELING BETWEEN STOPS"


def test_get_status_for_minute_dwell() -> None:
    seq = [(420, 430, "S1", "Main St", "T1", True, False, 1, 0)]
    # 423 is inside the GTFS hold but before the PRE_DEPARTURE_MINUTES window.
    status, *_ = get_status_for_minute(423, seq)
    assert status == "DWELL"


def test_get_status_for_minute_loading_window_before_departure() -> None:
    seq = [(420, 430, "S1", "Main St", "T1", True, False, 1, 0)]
    # The last PRE_DEPARTURE_MINUTES (5) before the 430 departure are LOADING.
    status, *_ = get_status_for_minute(426, seq)
    assert status == "LOADING"


_SEQ_THROUGH = [
    (420, 420, "S1", "Main St", "T1", True, False, 1, 0),
    (430, 430, "S2", "Oak Ave", "T1", False, False, 2, 0),
    (440, 440, "S3", "Elm St", "T1", False, True, 3, 0),
]


def test_get_status_for_minute_through_stop_dwell_window() -> None:
    # A through stop is occupied for THROUGH_DWELL_MINUTES (2) from its stop time.
    assert get_status_for_minute(430, _SEQ_THROUGH)[0] == "ARRIVE/DEPART"
    assert get_status_for_minute(431, _SEQ_THROUGH)[0] == "ARRIVE/DEPART"
    assert get_status_for_minute(432, _SEQ_THROUGH)[0] == "TRAVELING BETWEEN STOPS"


def test_get_status_for_minute_hold_boundaries_are_occupied() -> None:
    # A one-minute timed hold at a mid-trip stop occupies the stop at both of its minutes.
    seq = [
        (420, 420, "S1", "Main St", "T1", True, False, 1, 0),
        (430, 431, "S2", "Oak Ave", "T1", False, False, 2, 0),
        (440, 440, "S3", "Elm St", "T1", False, True, 3, 0),
    ]
    assert [get_status_for_minute(m, seq)[:2] for m in (430, 431, 432)] == [
        ("DWELL", "S2"),
        ("DWELL", "S2"),
        ("TRAVELING BETWEEN STOPS", None),
    ]


def test_get_status_for_minute_later_stop_wins_over_stretched_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "THROUGH_DWELL_MINUTES", 15)
    # S2's window would run to 445, past S3's 440 arrival; the bus has moved on.
    status, stop_id, *_ = get_status_for_minute(440, _SEQ_THROUGH)
    assert (status, stop_id) == ("ARRIVE", "S3")


# ---------------------------------------------------------------------------
# _gap_status / _row_for_inactive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gap", "same_place", "expected"),
    [
        (5, True, ("DWELL", "in bay")),
        (10, True, ("DWELL", "in bay")),  # IN_BAY_LAYOVER_MAX_MINUTES inclusive
        (11, True, ("LAYOVER", "overflow")),
        (20, True, ("LAYOVER", "overflow")),  # LAYOVER_THRESHOLD inclusive
        (21, True, ("LONG BREAK", "overflow")),
        (5, False, ("DEADHEAD", "")),
    ],
)
def test_gap_status(gap: int, same_place: bool, expected: tuple[str, str]) -> None:
    assert gap_status(gap, same_place, mod.occupancy_settings()) == expected


def _trip(trip_id: str, start: int, end: int, first_stop: str, last_stop: str) -> dict:
    return {
        "trip_id": trip_id,
        "start": start,
        "end": end,
        "departure": start,
        "arrival": end,
        "stop_times_sequence": [],
        "route_id": "R1",
        "route_short_name": "R1",
        "trip_headsign": "Downtown",
        "direction_id": "0",
        "first_stop_id": first_stop,
        "first_stop_name": f"Stop {first_stop}",
        "first_stop_seq": 1,
        "last_stop_id": last_stop,
        "last_stop_name": f"Stop {last_stop}",
        "last_stop_seq": 9,
    }


_HUB = [{"name": "Hub", "stops": ["S1", "S2"]}]


def test_row_for_inactive_short_gap_same_stop_is_in_bay_dwell() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 430, 450, "S1", "S9")]
    row = row_for_inactive(423, "B1", trips, [], mod.occupancy_settings())
    assert row["Status"] == "DWELL"
    assert row["Layover Location"] == "in bay"
    assert row["Stop ID"] == "S1"
    assert (row["Prev Trip ID"], row["Next Trip ID"]) == ("T1", "T2")
    assert row["Route Short Name"] == "R1"


def test_row_for_inactive_longer_gap_moves_to_overflow() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 435, 450, "S1", "S9")]
    row = row_for_inactive(425, "B1", trips, [], mod.occupancy_settings())
    assert (row["Status"], row["Layover Location"]) == ("LAYOVER", "overflow")
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 445, 460, "S1", "S9")]
    row = row_for_inactive(430, "B1", trips, [], mod.occupancy_settings())
    assert (row["Status"], row["Layover Location"]) == ("LONG BREAK", "overflow")


def test_row_for_inactive_same_cluster_counts_as_same_place() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 430, 450, "S2", "S9")]
    assert row_for_inactive(423, "B1", trips, [], mod.occupancy_settings())["Status"] == "DEADHEAD"
    row = row_for_inactive(423, "B1", trips, _HUB, mod.occupancy_settings())
    assert row["Status"] == "DWELL"
    assert row["Stop ID"] == "S1"  # the arrival bay


def test_row_for_inactive_deadhead_has_no_stop() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 430, 450, "S5", "S9")]
    row = row_for_inactive(423, "B1", trips, _HUB, mod.occupancy_settings())
    assert row["Status"] == "DEADHEAD"
    assert row["Stop ID"] == ""
    assert (row["Prev Trip ID"], row["Next Trip ID"]) == ("T1", "T2")


def test_row_for_inactive_loading_and_arrive_windows() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 428, 450, "S3", "S9")]
    # Within PRE_DEPARTURE_MINUTES (5) of T2: LOADING at T2's first stop.
    row = row_for_inactive(424, "B1", trips, [], mod.occupancy_settings())
    assert (row["Status"], row["Stop ID"], row["Trip ID"]) == ("LOADING", "S3", "T2")
    # Within POST_ARRIVAL_MINUTES (2) after T1: ARRIVE at T1's last stop.
    row = row_for_inactive(421, "B1", trips, [], mod.occupancy_settings())
    assert (row["Status"], row["Stop ID"], row["Trip ID"]) == ("ARRIVE", "S1", "T1")


def test_row_for_inactive_arrival_buffer_is_post_arrival_minutes_long() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 430, 450, "S1", "S9")]
    settings = {**mod.occupancy_settings(), "POST_ARRIVAL_MINUTES": 2}
    statuses = [row_for_inactive(m, "B1", trips, [], settings)["Status"] for m in (421, 422, 423)]
    assert statuses == ["ARRIVE", "ARRIVE", "DWELL"]  # two minutes after the 420 arrival
    settings["POST_ARRIVAL_MINUTES"] = 1
    assert row_for_inactive(421, "B1", trips, [], settings)["Status"] == "ARRIVE"
    assert row_for_inactive(422, "B1", trips, [], settings)["Status"] == "DWELL"


def test_row_for_inactive_pull_out_and_inactive() -> None:
    trips = [_trip("T1", 428, 450, "S3", "S9")]
    assert row_for_inactive(300, "B1", trips, [], mod.occupancy_settings())["Status"] == "INACTIVE"
    row = row_for_inactive(424, "B1", trips, [], mod.occupancy_settings())
    assert (row["Status"], row["Stop ID"]) == ("LOADING", "S3")  # pull-out
    trips = [_trip("T1", 400, 420, "S9", "S1")]
    assert row_for_inactive(600, "B1", trips, [], mod.occupancy_settings())["Status"] == "INACTIVE"


# ---------------------------------------------------------------------------
# fill_stop_ids_for_dwell_layover_loading
# ---------------------------------------------------------------------------


def _make_status_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Status": ["ARRIVE", "DWELL", "LAYOVER"],
            "Stop ID": ["S1", "", ""],
            "Stop Name": ["Main St", "", ""],
            "Stop Sequence": [1, "", ""],
            "Arrival Time": ["07:00", "", ""],
            "Departure Time": ["07:05", "", ""],
            "Trip ID": ["T1", "", ""],
        }
    )


def test_fill_dwell_stop_id_propagated() -> None:
    df = fill_stop_ids_for_dwell_layover_loading(_make_status_df())
    assert df.loc[1, "Stop ID"] == "S1"


def test_fill_layover_stop_id_propagated() -> None:
    df = fill_stop_ids_for_dwell_layover_loading(_make_status_df())
    assert df.loc[2, "Stop ID"] == "S1"


def test_fill_returns_dataframe() -> None:
    df = fill_stop_ids_for_dwell_layover_loading(_make_status_df())
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# process_block — integration smoke test
# ---------------------------------------------------------------------------


def _make_block_subset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "stop_sequence": [1, 2],
            "stop_id": ["S1", "S2"],
            "stop_name": ["Stop A", "Stop B"],
            "arrival_min": [420, 425],
            "departure_min": [420, 425],
            "is_first_stop": [True, False],
            "is_last_stop": [False, True],
            "route_id": ["R1", "R1"],
            "direction_id": ["0", "0"],
            "timepoint": [0, 0],
        }
    )


def test_process_block_returns_dataframe() -> None:
    df = process_block(_make_block_subset(), "BLK1", range(415, 430), [])
    assert isinstance(df, pd.DataFrame)


def test_process_block_has_expected_columns() -> None:
    df = process_block(_make_block_subset(), "BLK1", range(415, 430), [])
    for col in (
        "Timestamp",
        "Block",
        "Status",
        "Route Short Name",
        "Trip Headsign",
        "Layover Location",
        "Prev Trip ID",
        "Next Trip ID",
    ):
        assert col in df.columns, f"Missing column: {col}"


def test_process_block_row_count_matches_timeline() -> None:
    timeline = range(415, 430)
    df = process_block(_make_block_subset(), "BLK1", timeline, [])
    assert len(df) == len(timeline)


# ---------------------------------------------------------------------------
# run_step1_gtfs_to_blocks — full pipeline against gtfs_basic, real xlsx output
# ---------------------------------------------------------------------------

_GTFS_BASIC = Path(__file__).parent / "fixtures" / "gtfs_basic"


def _gtfs_basic_without_frequencies(tmp_path: Path) -> Path:
    """Copy gtfs_basic without frequencies.txt, so its trip T2 is an ordinary scheduled trip.

    The exporter stops on selected frequency-based trips (see the tests below);
    the tests that use this copy exercise other behavior on the same schedule.
    """
    feed = tmp_path / "gtfs_basic_scheduled"
    shutil.copytree(_GTFS_BASIC, feed)
    (feed / "frequencies.txt").unlink()
    return feed


def test_run_step1_writes_real_block_workbooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.gtfs_exports.block_status_timeline_exporter as mod

    fixture = _gtfs_basic_without_frequencies(tmp_path)
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(fixture))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", ["WKDY"])

    mod.run_step1_gtfs_to_blocks()

    names = sorted(p.name for p in tmp_path.glob("block_*.xlsx"))
    # Six blocks; B1-B3 interline routes R1 and R2, B4-B6 are R3 only.
    assert names == [
        "block_B1_R1_R2.xlsx",
        "block_B2_R1_R2.xlsx",
        "block_B3_R1_R2.xlsx",
        "block_B4_R3.xlsx",
        "block_B5_R3.xlsx",
        "block_B6_R3.xlsx",
    ]

    wb = openpyxl.load_workbook(tmp_path / "block_B1_R1_R2.xlsx")
    ws = wb.active
    header = [c.value for c in ws[1]]
    for col in ("Timestamp", "Block", "Status"):
        assert col in header, f"Missing column: {col}"
    # One row per minute of the 26-hour timeline.
    assert ws.max_row - 1 == mod.DEFAULT_HOURS * 60
    status_col = header.index("Status") + 1
    statuses = {ws.cell(row=r, column=status_col).value for r in range(2, ws.max_row + 1)}
    assert len(statuses) > 1  # active statuses beyond the inactive filler

    # The combined CSV holds every block's rows and names the workbook each came from.
    combined = pd.read_csv(tmp_path / mod.COMBINED_TIMELINE_FILE, dtype=str)
    assert len(combined) == 6 * mod.DEFAULT_HOURS * 60
    assert set(combined["FileName"]) == set(names)
    # The assumptions file records the occupancy settings used.
    assumptions = (tmp_path / mod.ASSUMPTIONS_FILE).read_text(encoding="utf-8")
    assert f"THROUGH_DWELL_MINUTES={mod.THROUGH_DWELL_MINUTES}" in assumptions
    assert "SERVICE_IDS_USED=['WKDY']" in assumptions
    # The run-log sidecar carries the CONFIGURATION block verbatim from the source.
    run_log = (tmp_path / mod.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "CONFIGURATION (verbatim from source)" in run_log
    assert "IN_BAY_LAYOVER_MAX_MINUTES = 10" in run_log
    assert "# === BEGIN CONFIG ===" not in run_log  # markers themselves are excluded
    # The manifest marks the run complete and fingerprints every file it wrote.
    manifest = json.loads((tmp_path / "timeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert set(manifest["block_workbooks"]) == set(names)
    assert set(manifest["files"]) == {
        *names,
        mod.COMBINED_TIMELINE_FILE,
        mod.SCHEDULE_SNAPSHOT_FILE,
        mod.ASSUMPTIONS_FILE,
    }


def _point_at_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _gtfs_basic_without_frequencies(tmp_path)
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(fixture))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", ["WKDY"])
    monkeypatch.setattr(mod, "WRITE_PER_BLOCK_FILES", False)


def test_main_returns_1_when_required_run_log_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _point_at_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "write_run_log", lambda output_dir: False)
    monkeypatch.setattr(mod, "REQUIRE_RUN_LOG", True)
    assert mod.main() == 1
    monkeypatch.setattr(mod, "REQUIRE_RUN_LOG", False)
    assert mod.main() == 0


def test_run_step1_uses_configured_stop_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _point_at_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "PRE_DEPARTURE_MINUTES", 0)
    monkeypatch.setattr(mod, "POST_ARRIVAL_MINUTES", 0)
    monkeypatch.setattr(mod, "BUS_STOP_CLUSTERS_STEP1", [{"name": "Hub", "stops": ["S6", "S7"]}])
    mod.run_step1_gtfs_to_blocks()
    combined = pd.read_csv(tmp_path / mod.COMBINED_TIMELINE_FILE, dtype=str)
    b1 = combined[combined["Block"] == "B1"].set_index("Timestamp")["Status"]
    # B1 turns from S6 (07:25) to S7 (07:30): one place inside the Hub, not a deadhead.
    assert list(b1.loc["07:26":"07:29"]) == ["DWELL"] * 4
    snapshot = json.loads((tmp_path / mod.SCHEDULE_SNAPSHOT_FILE).read_text(encoding="utf-8"))
    assert snapshot["clusters"] == [{"name": "Hub", "stops": ["S6", "S7"]}]


def test_extract_config_block_matches_source() -> None:
    block = mod.extract_config_block(Path(mod.__file__))
    assert block.lstrip().startswith("GTFS_FOLDER_PATH")
    assert "REQUIRE_RUN_LOG: bool = True" in block


def test_run_step1_service_date_and_scenario_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _gtfs_basic_without_frequencies(tmp_path)
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(fixture))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(mod, "SCENARIO_NAME", "alt")
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", [])
    monkeypatch.setattr(mod, "SERVICE_DATE", "20260702")  # a Thursday: WKDY runs
    monkeypatch.setattr(mod, "WRITE_PER_BLOCK_FILES", False)

    mod.run_step1_gtfs_to_blocks()

    run_dir = tmp_path / "alt"
    assert not list(run_dir.glob("block_*.xlsx"))
    combined = pd.read_csv(run_dir / mod.COMBINED_TIMELINE_FILE, dtype=str)
    assert sorted(combined["Block"].unique()) == ["B1", "B2", "B3", "B4", "B5", "B6"]
    assert "SERVICE_IDS_USED=['WKDY']" in (run_dir / mod.ASSUMPTIONS_FILE).read_text()


def test_run_step1_service_date_with_no_service_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "gtfs_basic"
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(fixture))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", [])
    monkeypatch.setattr(mod, "SERVICE_DATE", "20260703")  # removed in calendar_dates.txt
    with pytest.raises(ValueError, match="No service_id is active"):
        mod.run_step1_gtfs_to_blocks()


def test_run_step1_rejects_selected_frequency_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(_GTFS_BASIC))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", ["WKDY"])
    # T2 is repeated by frequencies.txt; one template cannot stand for its vehicles.
    with pytest.raises(ValueError, match=r"frequency-based \(T2\)"):
        mod.run_step1_gtfs_to_blocks()
    manifest = json.loads((tmp_path / "timeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_run_step1_ignores_frequency_trips_outside_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(_GTFS_BASIC))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", ["WKDY"])
    monkeypatch.setattr(mod, "WRITE_PER_BLOCK_FILES", False)
    monkeypatch.setattr(mod, "ROUTE_SHORTNAME_FILTER", ["R3"])  # blocks B4-B6; T2 is on B2
    mod.run_step1_gtfs_to_blocks()
    combined = pd.read_csv(tmp_path / mod.COMBINED_TIMELINE_FILE, dtype=str)
    assert sorted(combined["Block"].unique()) == ["B4", "B5", "B6"]


def test_run_step1_accepts_feed_without_optional_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = tmp_path / "core_only"
    feed.mkdir()
    for name in ("trips.txt", "stop_times.txt", "stops.txt", "routes.txt"):
        shutil.copy(_GTFS_BASIC / name, feed / name)
    (feed / "calendar_dates.txt").write_text("", encoding="utf-8")  # zero-byte: skipped
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(feed))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", ["WKDY"])
    monkeypatch.setattr(mod, "WRITE_PER_BLOCK_FILES", False)
    mod.run_step1_gtfs_to_blocks()
    combined = pd.read_csv(tmp_path / "out" / mod.COMBINED_TIMELINE_FILE, dtype=str)
    assert combined["Block"].nunique() == 6


def test_run_step1_reads_zip_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feed = _gtfs_basic_without_frequencies(tmp_path)
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in feed.iterdir():
            handle.write(path, f"wrapper/{path.name}")  # nested one folder deep
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(archive))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", ["WKDY"])
    monkeypatch.setattr(mod, "WRITE_PER_BLOCK_FILES", False)
    mod.run_step1_gtfs_to_blocks()
    manifest = json.loads((tmp_path / "out" / "timeline_manifest.json").read_text("utf-8"))
    assert manifest["status"] == "complete"


def test_run_step1_extends_timeline_past_default_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = _gtfs_basic_without_frequencies(tmp_path)
    trips = pd.read_csv(feed / "trips.txt", dtype=str)
    late = trips.iloc[[0]].assign(trip_id="TLATE", block_id="B9")
    pd.concat([trips, late]).to_csv(feed / "trips.txt", index=False)
    stop_times = pd.read_csv(feed / "stop_times.txt", dtype=str)
    late_times = pd.DataFrame(
        {
            "trip_id": ["TLATE", "TLATE"],
            "arrival_time": ["25:50:00", "26:10:00"],
            "departure_time": ["25:50:00", "26:10:00"],
            "stop_id": ["S1", "S2"],
            "stop_sequence": ["1", "2"],
        }
    )
    pd.concat([stop_times, late_times]).to_csv(feed / "stop_times.txt", index=False)
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(feed))
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(mod, "CALENDAR_SERVICE_IDS", ["WKDY"])
    monkeypatch.setattr(mod, "WRITE_PER_BLOCK_FILES", False)
    mod.run_step1_gtfs_to_blocks()
    combined = pd.read_csv(tmp_path / "out" / mod.COMBINED_TIMELINE_FILE, dtype=str)
    late_rows = combined[combined["Block"] == "B9"].set_index("Timestamp")["Status"]
    assert late_rows["26:10"] == "ARRIVE"  # DEFAULT_HOURS alone would stop at 25:59
    assert late_rows.index[-1] == "26:12"  # the arrival plus POST_ARRIVAL_MINUTES


def test_available_gtfs_files_folder_zip_and_empty(tmp_path: Path) -> None:
    (tmp_path / "calendar.txt").write_text("service_id\nWKDY\n", encoding="utf-8")
    (tmp_path / "frequencies.txt").write_text("", encoding="utf-8")
    names = ("calendar.txt", "calendar_dates.txt", "frequencies.txt")
    assert mod.available_gtfs_files(str(tmp_path), names) == ("calendar.txt",)
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("feed/calendar_dates.txt", "service_id,date,exception_type\n")
        handle.writestr("feed/frequencies.txt", "")
    assert mod.available_gtfs_files(str(archive), names) == ("calendar_dates.txt",)


# ---------------------------------------------------------------------------
# resolve_service_ids
# ---------------------------------------------------------------------------


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_id": ["WKDY", "SAT"],
            "monday": ["1", "0"],
            "tuesday": ["1", "0"],
            "wednesday": ["1", "0"],
            "thursday": ["1", "0"],
            "friday": ["1", "0"],
            "saturday": ["0", "1"],
            "sunday": ["0", "0"],
            "start_date": ["20260101", "20260101"],
            "end_date": ["20261231", "20261231"],
        }
    )


def _calendar_dates() -> pd.DataFrame:
    # July 3 2026 (a Friday): weekday service removed, Saturday service added.
    return pd.DataFrame(
        {
            "service_id": ["WKDY", "SAT"],
            "date": ["20260703", "20260703"],
            "exception_type": ["2", "1"],
        }
    )


def test_resolve_service_ids_day_of_week() -> None:
    assert resolve_service_ids(_calendar(), _calendar_dates(), "20260702") == ["WKDY"]
    assert resolve_service_ids(_calendar(), _calendar_dates(), "20260704") == ["SAT"]


def test_resolve_service_ids_applies_calendar_dates_exceptions() -> None:
    assert resolve_service_ids(_calendar(), _calendar_dates(), "20260703") == ["SAT"]


def test_resolve_service_ids_calendar_dates_only_feed() -> None:
    assert resolve_service_ids(None, _calendar_dates(), "20260703") == ["SAT"]


def test_resolve_service_ids_rejects_bad_date() -> None:
    with pytest.raises(ValueError, match="YYYYMMDD"):
        resolve_service_ids(_calendar(), None, "2026-07-02")


# ---------------------------------------------------------------------------
# apply_bay_overrides
# ---------------------------------------------------------------------------


def _merged_for_overrides() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_short_name": ["R1", "R1", "R2"],
            "stop_id": ["S1", "S2", "S1"],
            "stop_name": ["Stop S1", "Stop S2", "Stop S1"],
            "stop_code": ["1", "2", "1"],
            "direction_id": ["0", "0", "0"],
        }
    )


_STOPS = pd.DataFrame(
    {
        "stop_id": ["S1", "S2", "S9"],
        "stop_name": ["Stop S1", "Stop S2", "Bay H"],
        "stop_code": ["1", "2", "9"],
    }
)


def test_apply_bay_overrides_rewrites_matching_visits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "BAY_OVERRIDES",
        [{"route_short_name": "R1", "from_stop_ids": ["S1"], "to_stop_id": "S9"}],
    )
    merged = _merged_for_overrides()
    out = apply_bay_overrides(merged, _STOPS)
    assert list(out["stop_id"]) == ["S9", "S2", "S1"]  # R2's visit to S1 is untouched
    assert (out.loc[0, "stop_name"], out.loc[0, "stop_code"]) == ("Bay H", "9")
    assert list(merged["stop_id"]) == ["S1", "S2", "S1"]  # input not mutated


def test_apply_bay_overrides_honours_direction_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "BAY_OVERRIDES",
        [
            {
                "route_short_name": "R1",
                "from_stop_ids": ["S1"],
                "to_stop_id": "S9",
                "direction_id": "1",
            }
        ],
    )
    out = apply_bay_overrides(_merged_for_overrides(), _STOPS)
    assert list(out["stop_id"]) == ["S1", "S2", "S1"]


def test_apply_bay_overrides_noop_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "BAY_OVERRIDES", [])
    merged = _merged_for_overrides()
    assert apply_bay_overrides(merged, _STOPS) is merged


# ---------------------------------------------------------------------------
# _merge_and_filter_data, run_output_folder, assumptions_text
# ---------------------------------------------------------------------------


def test_merge_and_filter_data_orders_stop_sequence_numerically() -> None:
    trips = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "service_id": ["WKDY"],
            "route_id": ["R1"],
            "route_short_name": ["R1"],
            "direction_id": ["0"],
            "block_id": ["B1"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "arrival_time": ["07:00:00", "07:05:00", "07:10:00"],
            "departure_time": ["07:00:00", "07:05:00", "07:10:00"],
            "stop_id": ["S1", "S2", "S3"],
            "stop_sequence": ["1", "9", "10"],  # as read from GTFS: strings
        }
    )
    stops = pd.concat(
        [_STOPS.iloc[:2], pd.DataFrame({"stop_id": ["S3"], "stop_name": ["Stop S3"]})]
    )
    merged = _merge_and_filter_data(trips, stop_times, stops, ["WKDY"])
    # "10" sorts before "9" as text; numerically S3 (seq 10) is the last stop.
    assert merged.loc[merged["is_last_stop"], "stop_id"].tolist() == ["S3"]
    assert merged.loc[merged["is_first_stop"], "stop_id"].tolist() == ["S1"]
    # A stop missing from stops.txt is a broken reference, not a nameless stop.
    with pytest.raises(ValueError, match="absent from stops.txt: S3"):
        _merge_and_filter_data(trips, stop_times, _STOPS.iloc[:2], ["WKDY"])


def test_run_output_folder_appends_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "BLOCK_OUTPUT_FOLDER", "out")
    monkeypatch.setattr(mod, "SCENARIO_NAME", "alt")
    assert run_output_folder() == os.path.join("out", "alt")
    monkeypatch.setattr(mod, "SCENARIO_NAME", "")
    assert run_output_folder() == "out"


def test_assumptions_text_lists_settings_and_service_ids() -> None:
    text = assumptions_text(["WKDY"])
    assert f"PRE_DEPARTURE_MINUTES={mod.PRE_DEPARTURE_MINUTES}" in text
    assert "SERVICE_IDS_USED=['WKDY']" in text
    assert text.endswith("\n")


# ---------------------------------------------------------------------------
# Optional GTFS fields: untimed stops, route and stop names, trips without blocks
# ---------------------------------------------------------------------------

# One vehicle (B1) runs T1 from Elm St (S3) to the Hub (S1), then T2 back out to Oak St (S4).
_FEED = {
    "routes": "route_id,route_short_name,route_long_name,route_type\nR1,1,Line One,3\n",
    "stops": "stop_id,stop_name\nS1,Hub Bay 1\nS2,Hub Bay 2\nS3,Elm St\nS4,Oak St\n",
    "trips": "route_id,service_id,trip_id,direction_id,block_id\n"
    "R1,WKDY,T1,0,B1\nR1,WKDY,T2,1,B1\n",
    "stop_times": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
    "T1,07:00:00,07:00:00,S3,1\nT1,07:10:00,07:10:00,S1,2\n"
    "T2,07:20:00,07:20:00,S1,1\nT2,07:30:00,07:30:00,S4,2\n",
}


def _feed(tmp_path: Path, **tables: str) -> Path:
    """Write a GTFS folder; tables not given come from ``_FEED``."""
    feed = tmp_path / "feed"
    feed.mkdir()
    for name, text in {**_FEED, **tables}.items():
        (feed / f"{name}.txt").write_text(text, encoding="utf-8")
    return feed


def _export(feed: Path, out: Path, monkeypatch: pytest.MonkeyPatch, **settings: object) -> Path:
    """Run Step 1 on *feed* into *out* and return *out*."""
    base = {
        "GTFS_FOLDER_PATH": str(feed),
        "BLOCK_OUTPUT_FOLDER": str(out),
        "SCENARIO_NAME": "",
        "CALENDAR_SERVICE_IDS": ["WKDY"],
        "SERVICE_DATE": "",
        "WRITE_PER_BLOCK_FILES": False,
        "BUS_STOP_CLUSTERS_STEP1": [{"name": "Hub", "stops": ["S1", "S2"]}],
        "ROUTE_SHORTNAME_FILTER": [],
        "STOP_ID_FILTER": [],
        "STOP_CODE_FILTER": [],
        "BAY_OVERRIDES": [],
        "PRE_DEPARTURE_MINUTES": 5,
        "POST_ARRIVAL_MINUTES": 2,
    }
    for name, value in {**base, **settings}.items():
        monkeypatch.setattr(mod, name, value)
    mod.run_step1_gtfs_to_blocks()
    return out


def _timeline(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / mod.COMBINED_TIMELINE_FILE, dtype=str, keep_default_na=False)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_UNTIMED = (
    "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
    "T1,07:00:00,07:00:00,S3,1\nT1,,,S2,2\nT1,,,S4,3\nT1,07:09:00,07:09:00,S1,4\n"
    "T2,07:20:00,07:20:00,S1,1\nT2,07:30:00,07:30:00,S4,2\n"
)


def test_untimed_stops_stop_the_export_unless_interpolation_is_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = _feed(tmp_path, stop_times=_UNTIMED)
    with pytest.raises(ValueError, match=r"2 selected stop visit\(s\) on 1 trip\(s\).*\(T1\)"):
        _export(feed, tmp_path / "strict", monkeypatch)
    assert _json(tmp_path / "strict" / "timeline_manifest.json")["status"] == "failed"

    out = _export(feed, tmp_path / "out", monkeypatch, INTERPOLATE_UNTIMED_STOPS=True)
    rows = _timeline(out).set_index("Timestamp")
    # Evenly by stop order between 07:00 and 07:09: S2 at 07:03, S4 at 07:06.
    assert rows.loc["07:03", ["Stop ID", "Status", "Estimated Time"]].tolist() == [
        "S2",
        "ARRIVE/DEPART",
        "True",
    ]
    assert rows.loc["07:06", ["Stop ID", "Estimated Time"]].tolist() == ["S4", "True"]
    assert rows.loc["07:09", ["Stop ID", "Status", "Estimated Time"]].tolist() == [
        "S1",
        "ARRIVE",
        "False",
    ]
    assert set(rows.loc[rows["Estimated Time"] == "True", "Stop Sequence"]) == {"2", "3"}
    assert "ESTIMATED_STOP_VISITS=2 stop visit(s) on 1 trip(s)" in (
        out / mod.ASSUMPTIONS_FILE
    ).read_text(encoding="utf-8")
    snapshot = _json(out / mod.SCHEDULE_SNAPSHOT_FILE)
    t1 = next(trip for trip in snapshot["trips"] if trip["trip_id"] == "T1")
    assert t1["estimated_stop_sequences"] == [2, 3]
    assert snapshot["estimated_stop_visits"] == 2
    assert _json(out / "timeline_manifest.json")["estimated_stop_visits"] == 2


def _merged(stop_times: str, **trips: str) -> pd.DataFrame:
    """Run _merge_and_filter_data on ``_FEED`` with other stop_times (and trips)."""
    from io import StringIO

    def table(name: str, text: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(text), dtype=str)

    routes = table("routes", _FEED["routes"])
    trip_table = table("trips", trips.get("trips", _FEED["trips"])).merge(routes, on="route_id")
    return _merge_and_filter_data(
        trip_table, table("stop_times", stop_times), table("stops", _FEED["stops"]), ["WKDY"]
    )


def _times(merged: pd.DataFrame, trip: str = "T1") -> list[tuple[str, int, int, bool]]:
    rows = merged[merged["trip_id"] == trip].sort_values("stop_sequence")
    return list(
        zip(rows["stop_id"], rows["arrival_min"], rows["departure_min"], rows["estimated_time"])
    )


def test_interpolation_follows_shape_distance_when_the_whole_stretch_has_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "INTERPOLATE_UNTIMED_STOPS", True)
    stop_times = (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled\n"
        "T1,07:00:00,07:00:00,S3,1,0\nT1,,,S2,2,100\nT1,,,S4,3,800\n"
        "T1,07:09:00,07:09:00,S1,4,900\n"
        "T2,07:20:00,07:20:00,S1,1,0\nT2,07:30:00,07:30:00,S4,2,500\n"
    )
    assert _times(_merged(stop_times)) == [
        ("S3", 420, 420, False),
        ("S2", 421, 421, True),  # 100 of 900 distance units along: one minute in
        ("S4", 428, 428, True),
        ("S1", 429, 429, False),
    ]
    # One stop of the stretch without a distance: the stretch is spaced by stop order.
    merged = _merged(stop_times.replace("T1,,,S4,3,800", "T1,,,S4,3,"))
    assert [row[1] for row in _times(merged)] == [420, 423, 426, 429]
    estimated = merged[merged["estimated_time"]]
    assert set(estimated["timepoint"]) == {0}  # estimated through visits are not timepoints


def test_interpolation_needs_timed_first_and_last_stops_and_valid_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "INTERPOLATE_UNTIMED_STOPS", True)
    untimed_start = _UNTIMED.replace("T1,07:00:00,07:00:00,S3,1", "T1,,,S3,1")
    with pytest.raises(ValueError, match=r"untimed first or last stop \(T1\)"):
        _merged(untimed_start)
    with pytest.raises(ValueError, match="invalid arrival_time, e.g. '7:6x' in trip T2"):
        _merged(_UNTIMED.replace("T2,07:20:00,07:20:00", "T2,7:6x,07:20:00"))
    # A stop that gives only its arrival uses it for both times, flagged as estimated.
    one_sided = _UNTIMED.replace("T1,,,S2,2", "T1,07:04:00,,S2,2")
    assert _times(_merged(one_sided))[1] == ("S2", 424, 424, True)


def test_route_name_mask_uses_long_name_or_route_id_only_without_short_name() -> None:
    routes = pd.DataFrame(
        {
            "route_id": ["R10", "R20", "10"],
            "route_short_name": ["10", "", "5"],
            "route_long_name": ["Ten", "Crosstown", ""],
        }
    )
    assert mod.route_name_mask(routes, ["10"]).tolist() == [True, False, False]
    assert mod.route_name_mask(routes, ["Crosstown"]).tolist() == [False, True, False]
    assert mod.route_name_mask(routes, ["R20"]).tolist() == [False, True, False]
    assert not mod.route_name_mask(routes, ["Ten"]).any()  # R10 is named by its short name


def test_route_without_short_name_is_exported_filtered_and_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    routes = "route_id,route_short_name,route_long_name,route_type\nR1,,Line One,3\n"
    feed = _feed(tmp_path, routes=routes)
    out = _export(
        feed,
        tmp_path / "out",
        monkeypatch,
        ROUTE_SHORTNAME_FILTER=["Line One"],
        BAY_OVERRIDES=[
            {"route_short_name": "Line One", "from_stop_ids": ["S1"], "to_stop_id": "S2"}
        ],
    )
    rows = _timeline(out)
    trip_rows = rows[rows["Trip ID"] == "T1"]
    assert set(trip_rows["Route"]) == {"R1"}
    assert set(trip_rows["Route Short Name"]) == {""}
    assert set(trip_rows["Route Long Name"]) == {"Line One"}
    assert rows.set_index("Timestamp").loc["07:10", "Stop ID"] == "S2"  # override applied
    snapshot = _json(out / mod.SCHEDULE_SNAPSHOT_FILE)
    assert {trip["route_long_name"] for trip in snapshot["trips"]} == {"Line One"}


def test_bay_override_by_route_id_and_configuration_check(monkeypatch: pytest.MonkeyPatch) -> None:
    merged = _merged_for_overrides().assign(route_id=["R1", "R1", "R2"])
    monkeypatch.setattr(
        mod, "BAY_OVERRIDES", [{"route_id": "R2", "from_stop_ids": ["S1"], "to_stop_id": "S9"}]
    )
    assert list(apply_bay_overrides(merged, _STOPS)["stop_id"]) == ["S1", "S2", "S9"]
    monkeypatch.setattr(
        mod,
        "BAY_OVERRIDES",
        [{"route_id": "R2", "route_short_name": "R2", "from_stop_ids": ["S1"], "to_stop_id": "S9"}],
    )
    with pytest.raises(ValueError, match="exactly one of route_short_name and route_id"):
        mod.validate_configuration()


def test_unnamed_stop_is_labeled_by_its_id_and_still_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stops = "stop_id,stop_name\nS1,\nS2,Hub Bay 2\nS3,Elm St\nS4,Oak St\n"
    out = _export(_feed(tmp_path, stops=stops), tmp_path / "out", monkeypatch)
    assert "1 stop(s) used by the selected trips have no stop_name" in caplog.text
    at_hub = _timeline(out).set_index("Timestamp").loc["07:10"]
    assert (at_hub["Stop ID"], at_hub["Stop Name"], at_hub["Status"]) == ("S1", "S1", "ARRIVE")
    snapshot = _json(out / mod.SCHEDULE_SNAPSHOT_FILE)
    assert {"stop_id": "S1", "stop_name": "S1"} in snapshot["stops"]


def test_trip_without_block_stops_the_export_unless_trip_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trips = "route_id,service_id,trip_id,direction_id,block_id\nR1,WKDY,T1,0,B1\nR1,WKDY,T2,1,\n"
    feed = _feed(tmp_path, trips=trips)
    with pytest.raises(ValueError, match=r"1 selected trip\(s\) have no block_id \(T2\)"):
        _export(feed, tmp_path / "strict", monkeypatch)
    # A blockless trip outside the selection does not matter.
    extra = "R2,WKDY,T3,0,\n"
    feed2 = tmp_path / "second"
    feed2.mkdir()
    for name, text in {
        **_FEED,
        "routes": _FEED["routes"] + "R2,2,Line Two,3\n",
        "trips": _FEED["trips"] + extra,
        "stop_times": _FEED["stop_times"]
        + "T3,09:00:00,09:00:00,S3,1\nT3,09:10:00,09:10:00,S4,2\n",
    }.items():
        (feed2 / f"{name}.txt").write_text(text, encoding="utf-8")
    out = _export(feed2, tmp_path / "filtered", monkeypatch, ROUTE_SHORTNAME_FILTER=["1"])
    assert set(_timeline(out)["Block"]) == {"B1"}


def test_trip_only_mode_analyzes_each_blockless_trip_alone_and_discloses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trips = "route_id,service_id,trip_id,direction_id\nR1,WKDY,T1,0\nR1,WKDY,T2,1\n"  # no block_id
    out = _export(
        _feed(tmp_path, trips=trips), tmp_path / "out", monkeypatch, TRIP_ONLY_WITHOUT_BLOCK_ID=True
    )
    rows = _timeline(out)
    assert set(rows["Block"]) == {"trip-only:T1", "trip-only:T2"}
    t1 = rows[rows["Block"] == "trip-only:T1"].set_index("Timestamp")["Status"]
    t2 = rows[rows["Block"] == "trip-only:T2"].set_index("Timestamp")["Status"]
    # T1's arrival buffer and T2's loading are kept; the 07:10-07:20 layover between them
    # belongs to no known vehicle, so neither block shows it.
    assert list(t1.loc["07:10":"07:13"]) == ["ARRIVE", "ARRIVE", "ARRIVE", "INACTIVE"]
    assert list(t2.loc["07:14":"07:20"]) == ["INACTIVE"] + ["LOADING"] * 5 + ["DEPART"]
    assert not rows["Status"].isin({"DWELL", "LAYOVER", "LONG BREAK"}).any()
    assumptions = (out / mod.ASSUMPTIONS_FILE).read_text(encoding="utf-8")
    assert "TRIP_ONLY_TRIPS=2 trip(s) without a block_id" in assumptions
    assert "vehicle continuity are unknown" in assumptions
    assert _json(out / mod.SCHEDULE_SNAPSHOT_FILE)["trip_only_trips"] == ["T1", "T2"]
    assert _json(out / "timeline_manifest.json")["trip_only_trips"] == 2


def test_trip_only_block_ids_cannot_collide_with_real_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "TRIP_ONLY_WITHOUT_BLOCK_ID", True)
    trips = (
        "route_id,service_id,trip_id,direction_id,block_id\n"
        "R1,WKDY,T1,0,trip-only:T2\nR1,WKDY,T2,1,\n"
    )
    with pytest.raises(ValueError, match="'trip-only:T2' is used in trips.txt"):
        _merged(_FEED["stop_times"], trips=trips)


def test_first_stop_hold_occupies_the_bay_and_reports_the_scheduled_departure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # T2 reaches the Hub at 07:12 and holds there until its scheduled 07:20 departure.
    stop_times = _FEED["stop_times"].replace(
        "T2,07:20:00,07:20:00,S1,1", "T2,07:12:00,07:20:00,S1,1"
    )
    out = _export(_feed(tmp_path, stop_times=stop_times), tmp_path / "out", monkeypatch)
    rows = _timeline(out).set_index("Timestamp").loc["07:10":"07:20"]
    assert set(rows["Stop ID"]) == {"S1"}  # the bus is in the bay throughout
    assert list(rows["Status"]) == ["ARRIVE"] + ["LOADING"] * 1 + ["DWELL"] * 3 + [
        "LOADING"
    ] * 5 + ["DEPART"]
    # The between-trip row quotes the schedule: T1's 07:10 arrival, T2's 07:20 departure.
    assert rows.loc["07:11", ["Arrival Time", "Departure Time"]].tolist() == ["07:10", "07:20"]
    snapshot = _json(out / mod.SCHEDULE_SNAPSHOT_FILE)
    assert snapshot["schema_version"] == 2
    t2 = next(trip for trip in snapshot["trips"] if trip["trip_id"] == "T2")
    assert (t2["start"], t2["departure"], t2["arrival"], t2["end"]) == (432, 440, 450, 450)
