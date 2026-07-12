from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.gtfs_exports.bus_block_exporter import (
    fill_stop_ids_for_dwell_layover_loading,
    find_cluster,
    get_status_for_minute,
    load_gtfs_data,
    mark_first_and_last_stops,
    minutes_to_hhmm,
    parse_time_to_minutes,
    process_block,
    validate_folders,
)

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# parse_time_to_minutes
# ---------------------------------------------------------------------------


def test_parse_time_to_minutes_basic() -> None:
    assert parse_time_to_minutes("08:30:00") == 510


def test_parse_time_to_minutes_past_midnight() -> None:
    assert parse_time_to_minutes("25:00:00") == 1500


def test_parse_time_to_minutes_no_seconds() -> None:
    assert parse_time_to_minutes("08:30") == 510


# ---------------------------------------------------------------------------
# minutes_to_hhmm
# ---------------------------------------------------------------------------


def test_minutes_to_hhmm_basic() -> None:
    assert minutes_to_hhmm(510) == "08:30"


def test_minutes_to_hhmm_midnight() -> None:
    assert minutes_to_hhmm(0) == "00:00"


# ---------------------------------------------------------------------------
# validate_folders
# ---------------------------------------------------------------------------


def test_validate_folders_bad_input_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        validate_folders(tmp_path / "missing", tmp_path / "out")


def test_validate_folders_creates_output_dir(tmp_path: Path) -> None:
    src = tmp_path / "gtfs"
    src.mkdir()
    out = tmp_path / "output"
    validate_folders(src, out)
    assert out.is_dir()


# ---------------------------------------------------------------------------
# find_cluster
# ---------------------------------------------------------------------------

_CLUSTERS = [
    {"name": "Downtown", "stops": ["A", "B"]},
    {"name": "Uptown", "stops": ["C"]},
]


def test_find_cluster_found() -> None:
    assert find_cluster("A", _CLUSTERS) == "Downtown"


def test_find_cluster_not_found() -> None:
    assert find_cluster("Z", _CLUSTERS) is None


# ---------------------------------------------------------------------------
# mark_first_and_last_stops
# ---------------------------------------------------------------------------


def _two_trip_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2"],
            "stop_sequence": [1, 2, 1, 2],
            "stop_id": ["S1", "S2", "S3", "S4"],
        }
    )


def test_mark_first_last_sets_first_flag() -> None:
    df = mark_first_and_last_stops(_two_trip_df())
    assert df.loc[df["stop_id"] == "S1", "is_first_stop"].all()
    assert df.loc[df["stop_id"] == "S3", "is_first_stop"].all()


def test_mark_first_last_sets_last_flag() -> None:
    df = mark_first_and_last_stops(_two_trip_df())
    assert df.loc[df["stop_id"] == "S2", "is_last_stop"].all()
    assert df.loc[df["stop_id"] == "S4", "is_last_stop"].all()


# ---------------------------------------------------------------------------
# get_status_for_minute
# ---------------------------------------------------------------------------

# tuple layout: (arr, dep, stop_id, stop_name, trip_id, is_first, is_last, stop_seq, t_val)
_SEQUENCE = [
    (420, 425, "S1", "First", "T1", True, False, 1, 0),
    (430, 435, "S2", "Last", "T1", False, True, 2, 0),
]


def test_get_status_empty_sequence() -> None:
    status, *_ = get_status_for_minute(420, [])
    assert status == "EMPTY"


def test_get_status_depart() -> None:
    # DEPART fires at the departure minute of the first stop (dep=425).
    status, *_ = get_status_for_minute(425, _SEQUENCE)
    assert status == "DEPART"


def test_get_status_arrive() -> None:
    status, *_ = get_status_for_minute(430, _SEQUENCE)
    assert status == "ARRIVE"


def test_get_status_traveling() -> None:
    status, *_ = get_status_for_minute(427, _SEQUENCE)
    assert status == "TRAVELING BETWEEN STOPS"


# ---------------------------------------------------------------------------
# fill_stop_ids_for_dwell_layover_loading
# ---------------------------------------------------------------------------


def _dwell_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Status": ["ARRIVE", "DWELL", "LOADING"],
            "Stop ID": ["S1", "", ""],
            "Stop Name": ["Oak St", "", ""],
            "Stop Sequence": ["1", "", ""],
            "Arrival Time": ["08:00", "", ""],
            "Departure Time": ["08:05", "", ""],
            "Trip ID": ["T1", "", ""],
            "Trip Start Time": ["08:00", "", ""],
        }
    )


def test_fill_propagates_to_dwell() -> None:
    df = fill_stop_ids_for_dwell_layover_loading(_dwell_df())
    assert df.loc[1, "Stop ID"] == "S1"


def test_fill_propagates_to_loading() -> None:
    df = fill_stop_ids_for_dwell_layover_loading(_dwell_df())
    assert df.loc[2, "Stop ID"] == "S1"


# ---------------------------------------------------------------------------
# load_gtfs_data
# ---------------------------------------------------------------------------


def test_load_gtfs_data_missing_dir_raises() -> None:
    with pytest.raises(OSError):
        load_gtfs_data("/nonexistent/path")


def test_load_gtfs_data_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Missing GTFS files"):
        load_gtfs_data(str(tmp_path), files=["trips.txt"])


def test_load_gtfs_data_loads_files(tmp_path: Path) -> None:
    (tmp_path / "trips.txt").write_text("trip_id,route_id\nT1,R1\n", encoding="utf-8")
    data = load_gtfs_data(str(tmp_path), files=["trips.txt"])
    assert "trips" in data
    assert len(data["trips"]) == 1


# ---------------------------------------------------------------------------
# process_block — smoke test
# ---------------------------------------------------------------------------


def _block_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "stop_sequence": [1, 2],
            "stop_id": ["S1", "S2"],
            "stop_name": ["A", "B"],
            "arrival_min": [420, 425],
            "departure_min": [420, 425],
            "is_first_stop": [True, False],
            "is_last_stop": [False, True],
            "route_short_name": ["R1", "R1"],
            "direction_id": ["0", "0"],
            "timepoint": [2, 2],
        }
    )


def test_process_block_returns_dataframe() -> None:
    df = process_block(_block_df(), "B1", range(418, 428))
    assert isinstance(df, pd.DataFrame)


def test_process_block_row_count() -> None:
    timeline = range(418, 428)
    df = process_block(_block_df(), "B1", timeline)
    assert len(df) == len(timeline)
