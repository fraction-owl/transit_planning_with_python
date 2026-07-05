from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.facilities_tools.bay_usage_analyzer as target

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TEST_CLUSTERS = {
    "Park & Ride": {
        "single_bay_stops": ["100", "101"],
        "double_bay_stops": [],
        "triple_bay_stops": [],
        "overflow_bays": [],
    },
    "Metro": {
        "single_bay_stops": ["200"],
        "double_bay_stops": ["201"],
        "triple_bay_stops": ["202"],
        "overflow_bays": ["layover_A"],
    },
}


def _make_events_df() -> pd.DataFrame:
    """Three buses present at cluster 'Park & Ride' (capacity 2) at 08:00."""
    return pd.DataFrame(
        {
            "Stop ID": ["100", "100", "101", "200", "999"],
            "Status": ["ARRIVE", "LOADING", "DWELL", "ARRIVE", "ARRIVE"],
            "Timestamp": ["08:00", "08:00", "08:00", "08:00", "08:00"],
        }
    )


# ---------------------------------------------------------------------------
# get_all_official_stops
# ---------------------------------------------------------------------------


def test_get_all_official_stops_combines_bay_lists() -> None:
    cinfo = TEST_CLUSTERS["Metro"]
    assert target.get_all_official_stops(cinfo) == ["200", "201", "202"]


def test_get_all_official_stops_excludes_overflow() -> None:
    cinfo = TEST_CLUSTERS["Metro"]
    assert "layover_A" not in target.get_all_official_stops(cinfo)


def test_get_all_official_stops_missing_keys_returns_empty() -> None:
    assert target.get_all_official_stops({}) == []


# ---------------------------------------------------------------------------
# build_cluster_capacities / build_stop_capacities
# ---------------------------------------------------------------------------


def test_build_cluster_capacities_weights_bay_types() -> None:
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        caps = target.build_cluster_capacities()
    assert caps["Park & Ride"] == 2  # 2 singles
    assert caps["Metro"] == 7  # 1 + 2 + 3 + 1 overflow


def test_build_stop_capacities_per_stop_values() -> None:
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        caps = target.build_stop_capacities()
    assert caps["100"] == 1
    assert caps["201"] == 2
    assert caps["202"] == 3
    assert caps["layover_A"] == 1


# ---------------------------------------------------------------------------
# normalize_stop_id
# ---------------------------------------------------------------------------


def test_normalize_stop_id_strips_float_suffix() -> None:
    assert target.normalize_stop_id("2956.0") == "2956"
    assert target.normalize_stop_id(2956.0) == "2956"


def test_normalize_stop_id_nan_returns_none() -> None:
    assert target.normalize_stop_id(float("nan")) is None


def test_normalize_stop_id_strips_whitespace() -> None:
    assert target.normalize_stop_id(" 42 ") == "42"


def test_normalize_stop_id_plain_string_unchanged() -> None:
    assert target.normalize_stop_id("layover_A") == "layover_A"


# ---------------------------------------------------------------------------
# assign_cluster_name
# ---------------------------------------------------------------------------


def test_assign_cluster_name_maps_official_and_overflow_stops() -> None:
    df = pd.DataFrame({"Stop ID": ["100", "201", "layover_A", "999"]})
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        out = target.assign_cluster_name(df)
    assert out.loc[0, "ClusterName"] == "Park & Ride"
    assert out.loc[1, "ClusterName"] == "Metro"
    assert out.loc[2, "ClusterName"] == "Metro"
    assert out.loc[3, "ClusterName"] is None


def test_assign_cluster_name_does_not_mutate_input() -> None:
    df = pd.DataFrame({"Stop ID": ["100"]})
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        target.assign_cluster_name(df)
    assert "ClusterName" not in df.columns


# ---------------------------------------------------------------------------
# find_cluster_conflicts / find_stop_conflicts
# ---------------------------------------------------------------------------


def test_find_cluster_conflicts_flags_over_capacity_timestamp() -> None:
    df = _make_events_df()
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        df = target.assign_cluster_name(df)
        conflicts = target.find_cluster_conflicts(df)
    assert ("Park & Ride", "08:00") in conflicts
    assert ("Metro", "08:00") not in conflicts  # 1 bus, capacity 7


def test_find_cluster_conflicts_ignores_non_presence_statuses() -> None:
    df = pd.DataFrame(
        {
            "Stop ID": ["100", "100", "101"],
            "Status": ["TRAVELING", "TRAVELING", "TRAVELING"],
            "Timestamp": ["08:00"] * 3,
        }
    )
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        df = target.assign_cluster_name(df)
        assert target.find_cluster_conflicts(df) == set()


def test_find_stop_conflicts_flags_single_bay_double_service() -> None:
    df = _make_events_df()
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        conflicts = target.find_stop_conflicts(df)
    # Stop 100 has two passenger-service buses at 08:00 but capacity 1.
    assert ("100", "08:00") in conflicts
    assert ("101", "08:00") not in conflicts


def test_find_stop_conflicts_dwell_does_not_occupy_bay() -> None:
    df = pd.DataFrame(
        {
            "Stop ID": ["100", "100"],
            "Status": ["DWELL", "LAYOVER"],
            "Timestamp": ["08:00", "08:00"],
        }
    )
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        assert target.find_stop_conflicts(df) == set()


def test_find_stop_conflicts_unknown_stop_defaults_capacity_one() -> None:
    df = pd.DataFrame(
        {
            "Stop ID": ["999", "999"],
            "Status": ["ARRIVE", "DEPART"],
            "Timestamp": ["09:00", "09:00"],
        }
    )
    with patch.object(target, "CLUSTER_DEFINITIONS", TEST_CLUSTERS):
        assert ("999", "09:00") in target.find_stop_conflicts(df)


# ---------------------------------------------------------------------------
# annotate_conflicts
# ---------------------------------------------------------------------------


def test_annotate_conflicts_categorises_each_row() -> None:
    df = pd.DataFrame(
        {
            "ClusterName": ["Park & Ride", "Park & Ride", None, "Metro"],
            "Stop ID": ["100", "101", "999", "200"],
            "Timestamp": ["08:00", "08:00", "08:00", "08:00"],
        }
    )
    cluster_conflicts = {("Park & Ride", "08:00")}
    stop_conflicts = {("100", "08:00"), ("999", "08:00")}
    out = target.annotate_conflicts(df, cluster_conflicts, stop_conflicts)
    assert list(out["ConflictType"]) == ["BOTH", "CLUSTER", "STOP", "NONE"]


def test_annotate_conflicts_no_conflicts_all_none() -> None:
    df = pd.DataFrame(
        {
            "ClusterName": ["Metro"],
            "Stop ID": ["200"],
            "Timestamp": ["08:00"],
        }
    )
    out = target.annotate_conflicts(df, set(), set())
    assert list(out["ConflictType"]) == ["NONE"]


# ---------------------------------------------------------------------------
# gather_block_spreadsheets
# ---------------------------------------------------------------------------


def test_gather_block_spreadsheets_concatenates_block_files(tmp_path: Path) -> None:
    df1 = pd.DataFrame({"Timestamp": ["08:00"], "Block": ["B1"]})
    df2 = pd.DataFrame({"Timestamp": ["09:00"], "Block": ["B2"]})
    df1.to_excel(tmp_path / "block_101.xlsx", index=False)
    df2.to_excel(tmp_path / "block_102.xlsx", index=False)
    # A non-block file that must be ignored.
    df1.to_excel(tmp_path / "summary.xlsx", index=False)

    combined = target.gather_block_spreadsheets(str(tmp_path))
    assert len(combined) == 2
    assert set(combined["FileName"]) == {"block_101.xlsx", "block_102.xlsx"}


def test_gather_block_spreadsheets_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        target.gather_block_spreadsheets(str(tmp_path))


# ---------------------------------------------------------------------------
# _is_placeholder_path
# ---------------------------------------------------------------------------


def test_is_placeholder_path_detects_defaults() -> None:
    assert target._is_placeholder_path(r"Path\To\Your\Input_Folder") is True
    assert target._is_placeholder_path("path/to/your/output") is True


def test_is_placeholder_path_real_path_is_false() -> None:
    assert target._is_placeholder_path("/home/user/data/blocks") is False
