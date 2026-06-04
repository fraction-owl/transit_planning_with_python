"""Tests for scripts/gtfs_data_quality/gtfs_stop_proximity_qc.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/gtfs_data_quality").resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import gtfs_stop_proximity_qc as target  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# _meters_to_feet
# ---------------------------------------------------------------------------


def test_meters_to_feet_known_value() -> None:
    result = target._meters_to_feet(1.0)
    assert abs(result - 3.28084) < 0.0001


def test_meters_to_feet_zero() -> None:
    assert target._meters_to_feet(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _euclid_feet
# ---------------------------------------------------------------------------


def test_euclid_feet_pythagorean_triple() -> None:
    # 3 m, 4 m → 5 m → 5 * 3.28084 ft
    result = target._euclid_feet(3.0, 4.0)
    assert result == pytest.approx(5.0 * target.FEET_PER_M, rel=1e-4)


def test_euclid_feet_collinear() -> None:
    result = target._euclid_feet(10.0, 0.0)
    assert result == pytest.approx(10.0 * target.FEET_PER_M, rel=1e-4)


# ---------------------------------------------------------------------------
# _grid_cell
# ---------------------------------------------------------------------------


def test_grid_cell_origin() -> None:
    assert target._grid_cell(0.0, 0.0, 10.0) == (0, 0)


def test_grid_cell_positive_quadrant() -> None:
    assert target._grid_cell(25.0, 15.0, 10.0) == (2, 1)


def test_grid_cell_negative_coords() -> None:
    cx, cy = target._grid_cell(-1.0, -1.0, 10.0)
    assert cx == -1 and cy == -1


# ---------------------------------------------------------------------------
# _neighbor_cells
# ---------------------------------------------------------------------------


def test_neighbor_cells_returns_nine() -> None:
    cells = list(target._neighbor_cells((0, 0)))
    assert len(cells) == 9


def test_neighbor_cells_includes_center() -> None:
    assert (0, 0) in list(target._neighbor_cells((0, 0)))


# ---------------------------------------------------------------------------
# compile_safe_words_regex
# ---------------------------------------------------------------------------


def test_compile_safe_words_regex_matches_whole_word() -> None:
    rx = target.compile_safe_words_regex(["bay"], whole_word=True)
    assert rx.search("Metro Bay Terminal")
    assert not rx.search("bayshore")  # "bay" is a substring here, not a standalone word


def test_compile_safe_words_regex_substring_match() -> None:
    rx = target.compile_safe_words_regex(["bay"], whole_word=False)
    assert rx.search("bayshore")  # "bay" appears as substring


def test_compile_safe_words_regex_empty_list_matches_nothing() -> None:
    rx = target.compile_safe_words_regex([], whole_word=True)
    assert not rx.search("anything")


def test_compile_safe_words_regex_case_insensitive() -> None:
    rx = target.compile_safe_words_regex(["Metro"], whole_word=True)
    assert rx.search("metro station")


# ---------------------------------------------------------------------------
# add_safe_flag
# ---------------------------------------------------------------------------


def _make_stops_df(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": [f"S{i}" for i in range(len(names))],
            "stop_name": names,
            "stop_lat": [38.9] * len(names),
            "stop_lon": [-77.0] * len(names),
        }
    )


def test_add_safe_flag_marks_bay_stop() -> None:
    df = _make_stops_df(["Main St", "Metro Bay", "Oak Ave"])
    result = target.add_safe_flag(df, ["bay"], whole_word=True)
    assert result.loc[1, "is_safe_stop"]
    assert not result.loc[0, "is_safe_stop"]


def test_add_safe_flag_no_safe_words_all_false() -> None:
    df = _make_stops_df(["Main St", "Oak Ave"])
    result = target.add_safe_flag(df, [], whole_word=True)
    assert not result["is_safe_stop"].any()


# ---------------------------------------------------------------------------
# load_stops
# ---------------------------------------------------------------------------


def test_load_stops_raises_on_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "stops.txt"
    bad.write_text("stop_id,stop_lat\nS1,38.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stop_lon"):
        target.load_stops(bad)


def test_load_stops_drops_rows_with_missing_lat(tmp_path: Path) -> None:
    txt = tmp_path / "stops.txt"
    txt.write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nS1,Main,38.9,-77.0\nS2,Oak,,\n",
        encoding="utf-8",
    )
    df = target.load_stops(txt)
    assert len(df) == 1


def test_load_stops_returns_dataframe_with_required_cols(tmp_path: Path) -> None:
    txt = tmp_path / "stops.txt"
    txt.write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nS1,Main,38.9,-77.0\n",
        encoding="utf-8",
    )
    df = target.load_stops(txt)
    for col in ("stop_id", "stop_name", "stop_lat", "stop_lon"):
        assert col in df.columns


# ---------------------------------------------------------------------------
# build_stop_route_direction_index
# ---------------------------------------------------------------------------


def _make_gtfs_dir_with_direction(tmp_path: Path) -> Path:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nS1,Main,38.9,-77.0\nS2,Oak,38.91,-77.0\n",
        encoding="utf-8",
    )
    (gtfs / "trips.txt").write_text(
        "trip_id,route_id,direction_id\nT1,R1,0\nT2,R1,1\n",
        encoding="utf-8",
    )
    (gtfs / "stop_times.txt").write_text(
        "trip_id,stop_id\nT1,S1\nT2,S2\n",
        encoding="utf-8",
    )
    return gtfs


def test_build_stop_route_direction_index_returns_correct_structure(tmp_path: Path) -> None:
    gtfs = _make_gtfs_dir_with_direction(tmp_path)
    index = target.build_stop_route_direction_index(gtfs)
    assert "S1" in index
    assert index["S1"]["R1"] == {0}


def test_build_stop_route_direction_index_missing_files_returns_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty_gtfs"
    empty.mkdir()
    index = target.build_stop_route_direction_index(empty)
    assert index == {}


def test_build_stop_route_direction_index_no_direction_id_returns_empty(tmp_path: Path) -> None:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "trips.txt").write_text("trip_id,route_id\nT1,R1\n", encoding="utf-8")
    (gtfs / "stop_times.txt").write_text("trip_id,stop_id\nT1,S1\n", encoding="utf-8")
    index = target.build_stop_route_direction_index(gtfs)
    assert index == {}


# ---------------------------------------------------------------------------
# is_opposite_direction_pair_same_route
# ---------------------------------------------------------------------------


def _make_index() -> dict[str, dict[str, set[int]]]:
    return {
        "S1": {"R1": {0}},
        "S2": {"R1": {1}},
        "S3": {"R1": {0, 1}},
    }


def test_opposite_direction_pair_detected() -> None:
    index = _make_index()
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is True


def test_same_direction_pair_not_flagged() -> None:
    index = _make_index()
    assert target.is_opposite_direction_pair_same_route("S1", "S3", index) is False


def test_unknown_stop_returns_false() -> None:
    index = _make_index()
    assert target.is_opposite_direction_pair_same_route("S1", "UNKNOWN", index) is False


def test_no_shared_routes_returns_false() -> None:
    index: dict[str, dict[str, set[int]]] = {
        "S1": {"R1": {0}},
        "S2": {"R2": {1}},
    }
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is False


# ---------------------------------------------------------------------------
# find_close_stop_pairs
# ---------------------------------------------------------------------------


def _make_stops_with_safe(rows: list[dict]) -> pd.DataFrame:  # type: ignore[type-arg]
    """Build a minimal stops DataFrame with is_safe_stop column."""
    return pd.DataFrame(rows)


def test_find_close_stop_pairs_raises_without_safe_flag() -> None:
    df = pd.DataFrame(
        {
            "stop_id": ["S1"],
            "stop_name": ["Main"],
            "stop_lat": [38.9],
            "stop_lon": [-77.0],
        }
    )
    with pytest.raises(ValueError, match="is_safe_stop"):
        target.find_close_stop_pairs(df, 50.0, False, False, {})


def test_find_close_stop_pairs_detects_close_pair() -> None:
    # Two stops ~10 ft apart
    df = pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["Stop A", "Stop B"],
            "stop_lat": [38.9000, 38.9001],
            "stop_lon": [-77.0000, -77.0000],
            "is_safe_stop": [False, False],
        }
    )
    pairs = target.find_close_stop_pairs(df, 200.0, False, False, {})
    assert len(pairs) == 1
    assert set(pairs.columns) >= {"stop_id_a", "stop_id_b", "distance_feet"}


def test_find_close_stop_pairs_no_pairs_when_far_apart() -> None:
    df = pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["Stop A", "Stop B"],
            "stop_lat": [38.9, 39.9],
            "stop_lon": [-77.0, -77.0],
            "is_safe_stop": [False, False],
        }
    )
    pairs = target.find_close_stop_pairs(df, 50.0, False, False, {})
    assert pairs.empty


def test_find_close_stop_pairs_safe_stop_skipped() -> None:
    df = pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["Metro Bay", "Stop B"],
            "stop_lat": [38.9000, 38.9001],
            "stop_lon": [-77.0000, -77.0000],
            "is_safe_stop": [True, False],
        }
    )
    pairs = target.find_close_stop_pairs(
        df,
        200.0,
        pass_safe_stops=True,
        exclude_opposite_direction_same_route_pairs=False,
        stop_route_dir_index={},
    )
    assert pairs.empty


def test_find_close_stop_pairs_opposite_direction_excluded() -> None:
    df = pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["Stop A", "Stop B"],
            "stop_lat": [38.9000, 38.9001],
            "stop_lon": [-77.0000, -77.0000],
            "is_safe_stop": [False, False],
        }
    )
    index: dict[str, dict[str, set[int]]] = {"S1": {"R1": {0}}, "S2": {"R1": {1}}}
    pairs = target.find_close_stop_pairs(
        df,
        200.0,
        False,
        exclude_opposite_direction_same_route_pairs=True,
        stop_route_dir_index=index,
    )
    assert pairs.empty


# ---------------------------------------------------------------------------
# summarize_by_stop
# ---------------------------------------------------------------------------


def test_summarize_by_stop_empty_pairs_returns_empty() -> None:
    result = target.summarize_by_stop(pd.DataFrame())
    assert result.empty
    assert "stop_id" in result.columns


def test_summarize_by_stop_counts_correctly() -> None:
    pairs = pd.DataFrame(
        {
            "stop_id_a": ["S1", "S1"],
            "stop_id_b": ["S2", "S3"],
            "distance_feet": [10.0, 20.0],
        }
    )
    summary = target.summarize_by_stop(pairs)
    s1_row = summary[summary["stop_id"] == "S1"]
    assert s1_row["close_neighbor_pairs"].iloc[0] == 2


# ---------------------------------------------------------------------------
# Integration: mock_gtfs_dc fixture
# ---------------------------------------------------------------------------


def _extract_gtfs_dc(tmp_path: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(FIXTURES / "mock_gtfs_dc.zip") as zf:
        zf.extractall(tmp_path)
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_integration_dc_gtfs_load_stops(tmp_path: Path) -> None:
    gtfs_dir = _extract_gtfs_dc(tmp_path)
    df = target.load_stops(gtfs_dir / "stops.txt")
    assert len(df) > 0
    assert {"stop_id", "stop_name", "stop_lat", "stop_lon"}.issubset(df.columns)


def test_integration_dc_gtfs_find_pairs(tmp_path: Path) -> None:
    gtfs_dir = _extract_gtfs_dc(tmp_path)
    stops = target.load_stops(gtfs_dir / "stops.txt")
    stops = target.add_safe_flag(stops, ["bay"], whole_word=True)
    pairs = target.find_close_stop_pairs(
        stops,
        threshold_feet=50.0,
        pass_safe_stops=True,
        exclude_opposite_direction_same_route_pairs=False,
        stop_route_dir_index={},
    )
    # Just ensure it runs without error and returns a DataFrame
    assert isinstance(pairs, pd.DataFrame)


def test_integration_dc_gtfs_summarize(tmp_path: Path) -> None:
    gtfs_dir = _extract_gtfs_dc(tmp_path)
    stops = target.load_stops(gtfs_dir / "stops.txt")
    stops = target.add_safe_flag(stops, [], whole_word=True)
    pairs = target.find_close_stop_pairs(stops, 50.0, False, False, {})
    summary = target.summarize_by_stop(pairs)
    assert isinstance(summary, pd.DataFrame)
