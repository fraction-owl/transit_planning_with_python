"""Tests for scripts/gtfs_data_quality/gtfs_stop_proximity_qc.py."""

from __future__ import annotations

import logging
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


def test_load_stops_keeps_literal_na_like_ids(tmp_path: Path) -> None:
    txt = tmp_path / "stops.txt"
    txt.write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        "NA,None,38.90,-77.0\nnull,Stop null,38.91,-77.0\n"
        "N/A,Stop N/A,38.92,-77.0\nnan,Stop nan,38.93,-77.0\n",
        encoding="utf-8",
    )
    df = target.load_stops(txt)
    assert df["stop_id"].tolist() == ["NA", "null", "N/A", "nan"]
    assert df.loc[0, "stop_name"] == "None"


def test_load_stops_drops_blank_stop_id(tmp_path: Path) -> None:
    txt = tmp_path / "stops.txt"
    txt.write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nS1,Main,38.9,-77.0\n,Oak,38.91,-77.0\n",
        encoding="utf-8",
    )
    assert target.load_stops(txt)["stop_id"].tolist() == ["S1"]


def test_load_stops_drops_non_finite_and_out_of_range_coordinates(tmp_path: Path) -> None:
    txt = tmp_path / "stops.txt"
    txt.write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        "S1,Main,38.9,-77.0\nS2,Inf lat,inf,-77.0\nS3,Inf lon,38.9,-inf\n"
        "S4,Lat 95,95,-77.0\nS5,Lon 181,38.9,181\n",
        encoding="utf-8",
    )
    assert target.load_stops(txt)["stop_id"].tolist() == ["S1"]


def test_load_stops_keeps_only_stops_and_platforms(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    txt = tmp_path / "stops.txt"
    txt.write_text(
        "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
        "STA,Central Station,38.90000,-77.0,1,\n"
        "P1,Platform 1,38.90003,-77.0,0,STA\n"
        "P2,Platform 2,38.90006,-77.0,,STA\n"
        "E1,Entrance,38.89997,-77.0,2,STA\n"
        "N1,Pathway node,,,3,STA\n"
        "BA1,Boarding area,,,4,P1\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        df = target.load_stops(txt)
    assert df["stop_id"].tolist() == ["P1", "P2"]
    # Nodes and boarding areas may omit coordinates; they are not reported as invalid.
    assert "Dropped" not in caplog.text


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
    assert "S1" in index.directions
    assert index.directions["S1"]["R1"] == {0}


def test_build_stop_route_direction_index_missing_files_returns_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty_gtfs"
    empty.mkdir()
    index = target.build_stop_route_direction_index(empty)
    assert index == target.DirectionIndex()


def test_build_stop_route_direction_index_no_direction_id_returns_empty(tmp_path: Path) -> None:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "trips.txt").write_text("trip_id,route_id\nT1,R1\n", encoding="utf-8")
    (gtfs / "stop_times.txt").write_text("trip_id,stop_id\nT1,S1\n", encoding="utf-8")
    index = target.build_stop_route_direction_index(gtfs)
    assert index == target.DirectionIndex()


def _write_trips_and_stop_times(tmp_path: Path, trips: str, stop_times: str) -> Path:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "trips.txt").write_text(trips, encoding="utf-8")
    (gtfs / "stop_times.txt").write_text(stop_times, encoding="utf-8")
    return gtfs


def test_build_stop_route_direction_index_blank_direction_is_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs = _write_trips_and_stop_times(
        tmp_path,
        "trip_id,route_id,direction_id\nT0,X,0\nTU,X,\nT1,X,1\n",
        "trip_id,stop_id\nT0,A\nTU,A\nT1,B\n",
    )
    with caplog.at_level(logging.WARNING):
        index = target.build_stop_route_direction_index(gtfs)
    # The blank trip keeps A from reading as exclusively direction 0.
    assert index.directions == {"A": {"X": {0}}, "B": {"X": {1}}}
    assert index.unknown_direction == {"A": {"X"}}
    assert index.unresolved_stops == set()
    # Blank direction_id is valid GTFS, so it does not warn.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_build_stop_route_direction_index_keeps_route_with_only_unknown_directions(
    tmp_path: Path,
) -> None:
    gtfs = _write_trips_and_stop_times(
        tmp_path,
        "trip_id,route_id,direction_id\nT0,X,0\nL1,L,\n",
        "trip_id,stop_id\nT0,A\nL1,A\n",
    )
    index = target.build_stop_route_direction_index(gtfs)
    assert index.directions["A"] == {"X": {0}, "L": set()}
    assert index.unknown_direction["A"] == {"L"}


@pytest.mark.parametrize(
    ("trips", "stop_times"),
    [
        ("trip_id,route_id,direction_id\nT0,X,0\n", "trip_id,stop_id\nT0,A\nTZ,A\n"),
        ("trip_id,route_id,direction_id\nT0,X,0\nTR,,1\n", "trip_id,stop_id\nT0,A\nTR,A\n"),
    ],
    ids=["trip_missing_from_trips_txt", "blank_route_id"],
)
def test_build_stop_route_direction_index_unresolved_route_marks_stop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, trips: str, stop_times: str
) -> None:
    gtfs = _write_trips_and_stop_times(tmp_path, trips, stop_times)
    with caplog.at_level(logging.WARNING):
        index = target.build_stop_route_direction_index(gtfs)
    assert index.unresolved_stops == {"A"}
    assert index.directions == {"A": {"X": {0}}}
    assert "suppression disabled" in caplog.text


def test_build_stop_route_direction_index_keeps_literal_na_ids(tmp_path: Path) -> None:
    gtfs = _write_trips_and_stop_times(
        tmp_path,
        "trip_id,route_id,direction_id\nT0,NA,0\nNA,NA,1\n",
        "trip_id,stop_id\nT0,A\nNA,A\n",
    )
    index = target.build_stop_route_direction_index(gtfs)
    assert index.directions == {"A": {"NA": {0, 1}}}
    assert index.unresolved_stops == set()


def test_build_stop_route_direction_index_invalid_direction_warns_and_is_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs = _write_trips_and_stop_times(
        tmp_path,
        "trip_id,route_id,direction_id\nT0,X,0\nT1,X,1.5\n",
        "trip_id,stop_id\nT0,A\nT1,B\n",
    )
    with caplog.at_level(logging.WARNING):
        index = target.build_stop_route_direction_index(gtfs)
    assert "other than 0 or 1" in caplog.text
    assert index.directions == {"A": {"X": {0}}, "B": {"X": set()}}
    assert index.unknown_direction == {"B": {"X"}}


def test_build_stop_route_direction_index_all_blank_direction_warns_and_disables(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs = _write_trips_and_stop_times(
        tmp_path,
        "trip_id,route_id,direction_id\nT0,X,\nT1,X,\n",
        "trip_id,stop_id\nT0,A\nT1,B\n",
    )
    with caplog.at_level(logging.WARNING):
        index = target.build_stop_route_direction_index(gtfs)
    assert index == target.DirectionIndex()
    assert "direction-based filtering disabled" in caplog.text


# ---------------------------------------------------------------------------
# is_opposite_direction_pair_same_route
# ---------------------------------------------------------------------------


def _make_index() -> target.DirectionIndex:
    return target.DirectionIndex(
        directions={
            "S1": {"R1": {0}},
            "S2": {"R1": {1}},
            "S3": {"R1": {0, 1}},
        }
    )


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
    index = target.DirectionIndex(
        directions={
            "S1": {"R1": {0}},
            "S2": {"R2": {1}},
        }
    )
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is False


def test_opposite_direction_on_every_shared_route_detected() -> None:
    index = target.DirectionIndex(
        directions={"S1": {"R1": {0}, "R2": {1}}, "S2": {"R1": {1}, "R2": {0}}}
    )
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is True


def test_contradicting_shared_route_keeps_pair() -> None:
    # Opposite on R1, but both stops are served by direction 0 on R2.
    index = target.DirectionIndex(
        directions={"S1": {"R1": {0}, "R2": {0}}, "S2": {"R1": {1}, "R2": {0}}}
    )
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is False


def test_unknown_direction_on_shared_route_keeps_pair() -> None:
    # S1 also has a trip with unknown direction on R1, so it is not exclusively {0}.
    index = target.DirectionIndex(
        directions={"S1": {"R1": {0}}, "S2": {"R1": {1}}},
        unknown_direction={"S1": {"R1"}},
    )
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is False


def test_shared_route_with_only_unknown_directions_keeps_pair() -> None:
    # L must stay in the shared-route intersection even though its sets are empty;
    # otherwise R1 alone would suppress the pair.
    index = target.DirectionIndex(
        directions={"S1": {"R1": {0}, "L": set()}, "S2": {"R1": {1}, "L": set()}},
        unknown_direction={"S1": {"L"}, "S2": {"L"}},
    )
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is False


def test_unknown_direction_on_unshared_route_is_ignored() -> None:
    index = target.DirectionIndex(
        directions={"S1": {"R1": {0}, "L": set()}, "S2": {"R1": {1}}},
        unknown_direction={"S1": {"L"}},
    )
    assert target.is_opposite_direction_pair_same_route("S1", "S2", index) is True


@pytest.mark.parametrize("unresolved_stop", ["S1", "S2"])
def test_unresolved_stop_keeps_pair(unresolved_stop: str) -> None:
    index = target.DirectionIndex(
        directions={"S1": {"R1": {0}}, "S2": {"R1": {1}}},
        unresolved_stops=frozenset({unresolved_stop}),
    )
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
        target.find_close_stop_pairs(df, 50.0, False, False, target.DirectionIndex())


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
    pairs = target.find_close_stop_pairs(df, 200.0, False, False, target.DirectionIndex())
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
    pairs = target.find_close_stop_pairs(df, 50.0, False, False, target.DirectionIndex())
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
        stop_route_dir_index=target.DirectionIndex(),
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
    index = target.DirectionIndex(directions={"S1": {"R1": {0}}, "S2": {"R1": {1}}})
    pairs = target.find_close_stop_pairs(
        df,
        200.0,
        False,
        exclude_opposite_direction_same_route_pairs=True,
        stop_route_dir_index=index,
    )
    assert pairs.empty


def _two_close_stops() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["Stop A", "Stop B"],
            "stop_lat": [38.9000, 38.9001],
            "stop_lon": [-77.0000, -77.0000],
            "is_safe_stop": [False, False],
        }
    )


@pytest.mark.parametrize("threshold", [0.0, -10.0, float("nan"), float("inf")])
def test_find_close_stop_pairs_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold_feet"):
        target.find_close_stop_pairs(
            _two_close_stops(), threshold, False, False, target.DirectionIndex()
        )


def test_find_close_stop_pairs_small_threshold_finds_exact_duplicates() -> None:
    df = _two_close_stops()
    df.loc[1, ["stop_lat", "stop_lon"]] = df.loc[0, ["stop_lat", "stop_lon"]].to_numpy()
    pairs = target.find_close_stop_pairs(df, 1e-6, False, False, target.DirectionIndex())
    assert list(zip(pairs["stop_id_a"], pairs["stop_id_b"])) == [("S1", "S2")]


def test_find_close_stop_pairs_empty_result_keeps_header(tmp_path: Path) -> None:
    df = _two_close_stops()
    df["stop_lat"] = [38.9, 39.9]
    pairs = target.find_close_stop_pairs(df, 50.0, False, False, target.DirectionIndex())
    assert pairs.empty
    assert list(pairs.columns) == target.PAIR_COLUMNS
    # Written and read back as the script does, the CSV still has its header.
    out = tmp_path / "close_stop_pairs.csv"
    pairs.to_csv(out, index=False)
    assert list(pd.read_csv(out).columns) == target.PAIR_COLUMNS


def test_blank_direction_on_shared_route_keeps_close_pair(tmp_path: Path) -> None:
    gtfs = _write_trips_and_stop_times(
        tmp_path,
        "trip_id,route_id,direction_id\nT0,X,0\nTU,X,\nT1,X,1\n",
        "trip_id,stop_id\nT0,A\nTU,A\nT1,B\n",
    )
    (gtfs / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nA,Stop A,38.90000,-77.0\nB,Stop B,38.90003,-77.0\n",
        encoding="utf-8",
    )
    stops = target.add_safe_flag(target.load_stops(gtfs / "stops.txt"), [], whole_word=True)
    index = target.build_stop_route_direction_index(gtfs)
    pairs = target.find_close_stop_pairs(stops, 50.0, False, True, index)
    assert list(zip(pairs["stop_id_a"], pairs["stop_id_b"])) == [("A", "B")]


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
        stop_route_dir_index=target.DirectionIndex(),
    )
    # Just ensure it runs without error and returns a DataFrame
    assert isinstance(pairs, pd.DataFrame)


def test_integration_dc_gtfs_summarize(tmp_path: Path) -> None:
    gtfs_dir = _extract_gtfs_dc(tmp_path)
    stops = target.load_stops(gtfs_dir / "stops.txt")
    stops = target.add_safe_flag(stops, [], whole_word=True)
    pairs = target.find_close_stop_pairs(stops, 50.0, False, False, target.DirectionIndex())
    summary = target.summarize_by_stop(pairs)
    assert isinstance(summary, pd.DataFrame)
