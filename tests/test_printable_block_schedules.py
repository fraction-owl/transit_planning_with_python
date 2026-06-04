from __future__ import annotations

import math
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.field_tools.printable_block_schedules import (
    export_blocks,
    export_to_excel,
    filter_data,
    format_hhmm,
    load_gtfs_data,
    prepare_stop_times,
    time_to_seconds,
)

FIXTURES = Path(__file__).parent / "fixtures"
GTFS_BASIC = FIXTURES / "gtfs_basic"

_MODULE = "scripts.field_tools.printable_block_schedules"

# ---------------------------------------------------------------------------
# time_to_seconds
# ---------------------------------------------------------------------------


def test_time_to_seconds_hhmmss() -> None:
    assert time_to_seconds("08:30:00") == 30600


def test_time_to_seconds_hhmm_no_seconds() -> None:
    assert time_to_seconds("08:30") == 30600


def test_time_to_seconds_midnight() -> None:
    assert time_to_seconds("00:00:00") == 0


def test_time_to_seconds_past_midnight_rolls_over() -> None:
    # 25:10:00 → (25 % 24) = 1 h + 10 min = 4200 s
    assert time_to_seconds("25:10:00") == 4200


def test_time_to_seconds_nan_returns_nan() -> None:
    assert math.isnan(time_to_seconds(float("nan")))


def test_time_to_seconds_bad_string_returns_nan() -> None:
    assert math.isnan(time_to_seconds("not-a-time"))


def test_time_to_seconds_single_part_returns_nan() -> None:
    assert math.isnan(time_to_seconds("08"))


# ---------------------------------------------------------------------------
# format_hhmm
# ---------------------------------------------------------------------------


def test_format_hhmm_basic() -> None:
    assert format_hhmm(30600) == "08:30"


def test_format_hhmm_midnight() -> None:
    assert format_hhmm(0) == "00:00"


def test_format_hhmm_one_hour_boundary() -> None:
    assert format_hhmm(3600) == "01:00"


def test_format_hhmm_negative_returns_empty() -> None:
    assert format_hhmm(-1) == ""


def test_format_hhmm_nan_returns_empty() -> None:
    assert format_hhmm(float("nan")) == ""


def test_format_hhmm_above_24h() -> None:
    # 86400 s = 24:00
    assert format_hhmm(86400) == "24:00"


# ---------------------------------------------------------------------------
# load_gtfs_data
# ---------------------------------------------------------------------------


def test_load_gtfs_data_missing_dir_raises() -> None:
    with pytest.raises(OSError, match="does not exist"):
        load_gtfs_data("/nonexistent/gtfs")


def test_load_gtfs_data_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Missing GTFS files"):
        load_gtfs_data(str(tmp_path), files=["trips.txt"])


def test_load_gtfs_data_returns_all_requested_keys(tmp_path: Path) -> None:
    (tmp_path / "trips.txt").write_text("trip_id,route_id\nT1,R1\n", encoding="utf-8")
    (tmp_path / "stops.txt").write_text("stop_id,stop_name\nS1,Oak St\n", encoding="utf-8")
    data = load_gtfs_data(str(tmp_path), files=["trips.txt", "stops.txt"])
    assert set(data.keys()) == {"trips", "stops"}


def test_load_gtfs_data_correct_row_count(tmp_path: Path) -> None:
    (tmp_path / "trips.txt").write_text("trip_id,route_id\nT1,R1\nT2,R2\n", encoding="utf-8")
    data = load_gtfs_data(str(tmp_path), files=["trips.txt"])
    assert len(data["trips"]) == 2


def test_load_gtfs_data_preserves_leading_zeros_as_str(tmp_path: Path) -> None:
    (tmp_path / "stops.txt").write_text("stop_id,stop_name\n0001,Oak St\n", encoding="utf-8")
    data = load_gtfs_data(str(tmp_path), files=["stops.txt"])
    assert data["stops"]["stop_id"].iloc[0] == "0001"


def test_load_gtfs_data_empty_file_raises(tmp_path: Path) -> None:
    (tmp_path / "trips.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_gtfs_data(str(tmp_path), files=["trips.txt"])


def test_load_gtfs_data_smoke_test_gtfs_basic_fixture() -> None:
    data = load_gtfs_data(
        str(GTFS_BASIC),
        files=["trips.txt", "stops.txt", "routes.txt", "stop_times.txt"],
    )
    assert "trips" in data
    assert len(data["trips"]) > 0
    assert "stops" in data


# ---------------------------------------------------------------------------
# Shared helpers for filter / prepare tests
# ---------------------------------------------------------------------------


def _make_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1", "R1", "R2"],
            "block_id": ["B1", "B1", "B2"],
            "service_id": ["WKDY", "WKDY", "SAT"],
        }
    )


def _make_routes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["R1", "R2"],
            "route_short_name": ["10", "20"],
        }
    )


def _make_stop_times(trip_ids: list[str] | None = None) -> pd.DataFrame:
    tids = trip_ids or ["T1", "T2", "T3"]
    return pd.DataFrame(
        {
            "trip_id": tids,
            "stop_id": [f"S{i + 1}" for i in range(len(tids))],
            "stop_sequence": ["1"] * len(tids),
            "arrival_time": ["07:00:00"] * len(tids),
            "departure_time": ["07:00:00"] * len(tids),
        }
    )


# ---------------------------------------------------------------------------
# filter_data
# ---------------------------------------------------------------------------


def test_filter_data_no_filters_keeps_all_trips() -> None:
    with (
        patch(f"{_MODULE}.FILTER_ROUTE_SHORT_NAMES", []),
        patch(f"{_MODULE}.FILTER_SERVICE_IDS", []),
    ):
        ft, fst = filter_data(_make_trips(), _make_stop_times(), _make_routes())
    assert len(ft) == 3
    assert len(fst) == 3


def test_filter_data_by_route_short_name_keeps_matching_block() -> None:
    with (
        patch(f"{_MODULE}.FILTER_ROUTE_SHORT_NAMES", ["10"]),
        patch(f"{_MODULE}.FILTER_SERVICE_IDS", []),
    ):
        ft, _ = filter_data(_make_trips(), _make_stop_times(), _make_routes())
    # Block B1 belongs to route 10; B2 belongs to route 20
    assert set(ft["block_id"].unique()) == {"B1"}


def test_filter_data_by_service_id() -> None:
    with (
        patch(f"{_MODULE}.FILTER_ROUTE_SHORT_NAMES", []),
        patch(f"{_MODULE}.FILTER_SERVICE_IDS", ["SAT"]),
    ):
        ft, fst = filter_data(_make_trips(), _make_stop_times(), _make_routes())
    assert set(ft["service_id"].unique()) == {"SAT"}
    assert len(fst) == 1


def test_filter_data_no_matching_route_returns_empty_dataframes() -> None:
    with (
        patch(f"{_MODULE}.FILTER_ROUTE_SHORT_NAMES", ["99"]),
        patch(f"{_MODULE}.FILTER_SERVICE_IDS", []),
    ):
        ft, fst = filter_data(_make_trips(), _make_stop_times(), _make_routes())
    assert ft.empty
    assert fst.empty


def test_filter_data_stop_times_restricted_to_filtered_trips() -> None:
    with (
        patch(f"{_MODULE}.FILTER_ROUTE_SHORT_NAMES", ["20"]),
        patch(f"{_MODULE}.FILTER_SERVICE_IDS", []),
    ):
        _, fst = filter_data(_make_trips(), _make_stop_times(), _make_routes())
    assert set(fst["trip_id"].unique()) == {"T3"}


# ---------------------------------------------------------------------------
# prepare_stop_times
# ---------------------------------------------------------------------------


def _make_prep_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "block_id": ["B1", "B2"],
            "route_short_name": ["10", "20"],
            "direction_id": ["0", "1"],
        }
    )


def _make_prep_stop_times() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2"],
            "stop_id": ["S1", "S2", "S3"],
            "stop_sequence": ["1", "2", "1"],
            "arrival_time": ["07:00:00", "07:05:00", "08:00:00"],
            "departure_time": ["07:00:00", "07:05:00", "08:00:00"],
        }
    )


def _make_stops() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "stop_name": ["Oak St", "Elm Ave", "Pine Rd"],
        }
    )


def test_prepare_stop_times_adds_scheduled_time_column() -> None:
    result = prepare_stop_times(_make_prep_trips(), _make_prep_stop_times(), _make_stops())
    assert "scheduled_time_hhmm" in result.columns
    assert result["scheduled_time_hhmm"].iloc[0] == "07:00"


def test_prepare_stop_times_maps_stop_names() -> None:
    result = prepare_stop_times(_make_prep_trips(), _make_prep_stop_times(), _make_stops())
    assert "Oak St" in result["stop_name"].to_numpy()


def test_prepare_stop_times_creates_timepoint_column_when_absent() -> None:
    result = prepare_stop_times(_make_prep_trips(), _make_prep_stop_times(), _make_stops())
    assert "timepoint" in result.columns
    assert (result["timepoint"] == 0).all()


def test_prepare_stop_times_preserves_existing_timepoint_values() -> None:
    st = _make_prep_stop_times()
    st["timepoint"] = ["1", "0", "1"]
    result = prepare_stop_times(_make_prep_trips(), st, _make_stops())
    assert result["timepoint"].tolist() == [1, 0, 1]


def test_prepare_stop_times_unknown_stop_falls_back_to_unknown_stop() -> None:
    stops_partial = _make_stops().iloc[:1]  # Only S1
    result = prepare_stop_times(_make_prep_trips(), _make_prep_stop_times(), stops_partial)
    unknown = result[result["stop_id"] == "S2"]["stop_name"]
    assert (unknown == "Unknown Stop").all()


def test_prepare_stop_times_sorted_by_block_trip_sequence() -> None:
    result = prepare_stop_times(_make_prep_trips(), _make_prep_stop_times(), _make_stops())
    result = result.reset_index(drop=True)
    # Re-sorting should not change the row order if already sorted correctly
    re_sorted = result.sort_values(["block_id", "trip_id", "stop_sequence"])
    assert re_sorted.index.tolist() == list(range(len(result)))


def test_prepare_stop_times_drops_rows_with_missing_block_id() -> None:
    trips = _make_prep_trips().copy()
    trips.loc[0, "block_id"] = None
    result = prepare_stop_times(trips, _make_prep_stop_times(), _make_stops())
    assert result["block_id"].notna().all()


# ---------------------------------------------------------------------------
# export_to_excel
# ---------------------------------------------------------------------------


def test_export_to_excel_creates_xlsx_file(tmp_path: Path) -> None:
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    out = str(tmp_path / "schedule.xlsx")
    export_to_excel(df, out)
    assert os.path.exists(out)


def test_export_to_excel_empty_dataframe_skips_write(tmp_path: Path) -> None:
    out = str(tmp_path / "schedule.xlsx")
    export_to_excel(pd.DataFrame(), out)
    assert not os.path.exists(out)


def test_export_to_excel_creates_nested_parent_directories(tmp_path: Path) -> None:
    df = pd.DataFrame({"Col": [1]})
    out = str(tmp_path / "a" / "b" / "schedule.xlsx")
    export_to_excel(df, out)
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# export_blocks
# ---------------------------------------------------------------------------


def _make_block_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block_id": ["B1", "B1", "B2"],
            "route_short_name": ["10", "10", "20"],
            "direction_id": ["0", "0", "1"],
            "trip_id": ["T1", "T1", "T2"],
            "stop_sequence": [1, 2, 1],
            "timepoint": [0, 0, 1],
            "stop_id": ["S1", "S2", "S3"],
            "stop_name": ["Oak St", "Elm Ave", "Pine Rd"],
            "scheduled_time_hhmm": ["07:00", "07:05", "08:00"],
            "departure_seconds": [25200.0, 25500.0, 28800.0],
        }
    )


def test_export_blocks_creates_one_file_per_block(tmp_path: Path) -> None:
    with patch(f"{_MODULE}.BASE_OUTPUT_PATH", str(tmp_path)):
        export_blocks(_make_block_df())
    xlsx_files = list(tmp_path.glob("*.xlsx"))
    assert len(xlsx_files) == 2


def test_export_blocks_filename_contains_block_id(tmp_path: Path) -> None:
    with patch(f"{_MODULE}.BASE_OUTPUT_PATH", str(tmp_path)):
        export_blocks(_make_block_df())
    names = {f.name for f in tmp_path.glob("*.xlsx")}
    assert "block_B1_schedule_printable.xlsx" in names
    assert "block_B2_schedule_printable.xlsx" in names


def test_export_blocks_output_includes_placeholder_columns(tmp_path: Path) -> None:
    with patch(f"{_MODULE}.BASE_OUTPUT_PATH", str(tmp_path)):
        export_blocks(_make_block_df())
    result = pd.read_excel(tmp_path / "block_B1_schedule_printable.xlsx")
    for col in ("Actual Time", "Boardings", "Alightings", "Comments"):
        assert col in result.columns
