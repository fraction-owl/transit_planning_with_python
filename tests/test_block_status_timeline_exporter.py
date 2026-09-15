from __future__ import annotations

import os
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

import scripts.gtfs_exports.block_status_timeline_exporter as mod
from scripts.gtfs_exports.block_status_timeline_exporter import (
    _gap_status,
    _merge_and_filter_data,
    _row_for_inactive,
    apply_bay_overrides,
    assumptions_text,
    fill_stop_ids_for_dwell_layover_loading,
    find_cluster,
    get_status_for_minute,
    mark_first_and_last_stops,
    minutes_to_hhmm,
    parse_time_to_minutes,
    process_block,
    resolve_service_ids,
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
    assert _gap_status(gap, same_place) == expected


def _trip(trip_id: str, start: int, end: int, first_stop: str, last_stop: str) -> dict:
    return {
        "trip_id": trip_id,
        "start": start,
        "end": end,
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
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 428, 450, "S1", "S9")]
    row = _row_for_inactive(422, "B1", trips, [])
    assert row["Status"] == "DWELL"
    assert row["Layover Location"] == "in bay"
    assert row["Stop ID"] == "S1"
    assert (row["Prev Trip ID"], row["Next Trip ID"]) == ("T1", "T2")
    assert row["Route Short Name"] == "R1"


def test_row_for_inactive_longer_gap_moves_to_overflow() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 435, 450, "S1", "S9")]
    row = _row_for_inactive(425, "B1", trips, [])
    assert (row["Status"], row["Layover Location"]) == ("LAYOVER", "overflow")
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 445, 460, "S1", "S9")]
    row = _row_for_inactive(430, "B1", trips, [])
    assert (row["Status"], row["Layover Location"]) == ("LONG BREAK", "overflow")


def test_row_for_inactive_same_cluster_counts_as_same_place() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 428, 450, "S2", "S9")]
    assert _row_for_inactive(422, "B1", trips, [])["Status"] == "DEADHEAD"
    row = _row_for_inactive(422, "B1", trips, _HUB)
    assert row["Status"] == "DWELL"
    assert row["Stop ID"] == "S1"  # the arrival bay


def test_row_for_inactive_deadhead_has_no_stop() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 428, 450, "S5", "S9")]
    row = _row_for_inactive(422, "B1", trips, _HUB)
    assert row["Status"] == "DEADHEAD"
    assert row["Stop ID"] == ""
    assert (row["Prev Trip ID"], row["Next Trip ID"]) == ("T1", "T2")


def test_row_for_inactive_loading_and_arrive_windows() -> None:
    trips = [_trip("T1", 400, 420, "S9", "S1"), _trip("T2", 428, 450, "S3", "S9")]
    # Within PRE_DEPARTURE_MINUTES (5) of T2: LOADING at T2's first stop.
    row = _row_for_inactive(424, "B1", trips, [])
    assert (row["Status"], row["Stop ID"], row["Trip ID"]) == ("LOADING", "S3", "T2")
    # Within POST_ARRIVAL_MINUTES (2) after T1: ARRIVE at T1's last stop.
    row = _row_for_inactive(421, "B1", trips, [])
    assert (row["Status"], row["Stop ID"], row["Trip ID"]) == ("ARRIVE", "S1", "T1")


def test_row_for_inactive_pull_out_and_inactive() -> None:
    trips = [_trip("T1", 428, 450, "S3", "S9")]
    assert _row_for_inactive(300, "B1", trips, [])["Status"] == "INACTIVE"
    row = _row_for_inactive(424, "B1", trips, [])
    assert (row["Status"], row["Stop ID"]) == ("LOADING", "S3")  # pull-out
    trips = [_trip("T1", 400, 420, "S9", "S1")]
    assert _row_for_inactive(600, "B1", trips, [])["Status"] == "INACTIVE"


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


def test_run_step1_writes_real_block_workbooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.gtfs_exports.block_status_timeline_exporter as mod

    fixture = Path(__file__).parent / "fixtures" / "gtfs_basic"
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


def _point_at_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).parent / "fixtures" / "gtfs_basic"
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


def test_extract_config_block_matches_source() -> None:
    block = mod.extract_config_block(Path(mod.__file__))
    assert block.lstrip().startswith("GTFS_FOLDER_PATH")
    assert "REQUIRE_RUN_LOG: bool = True" in block


def test_run_step1_service_date_and_scenario_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "gtfs_basic"
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
    merged = _merge_and_filter_data(trips, stop_times, _STOPS.iloc[:2], ["WKDY"])
    # "10" sorts before "9" as text; numerically S3 (seq 10) is the last stop.
    assert merged.loc[merged["is_last_stop"], "stop_id"].tolist() == ["S3"]
    assert merged.loc[merged["is_first_stop"], "stop_id"].tolist() == ["S1"]


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
