from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from scripts.stop_analysis.audit_turn_clearance import (
    _ensure_output,
    _filter_routes,
    _find_left_turns,
    _read_gtfs,
    _validate_gtfs,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gtfs_dir(tmp_path: Path) -> Path:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "stops.txt").write_text(
        "stop_id,stop_lat,stop_lon,stop_name\nS1,38.7,-77.0,Main St\n",
        encoding="utf-8",
    )
    (gtfs / "routes.txt").write_text(
        "route_id,route_short_name\nR1,101\n",
        encoding="utf-8",
    )
    (gtfs / "trips.txt").write_text(
        "trip_id,route_id,shape_id,direction_id\nT1,R1,SHP1,0\n",
        encoding="utf-8",
    )
    (gtfs / "stop_times.txt").write_text(
        "trip_id,stop_id\nT1,S1\n",
        encoding="utf-8",
    )
    (gtfs / "shapes.txt").write_text(
        "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
        "SHP1,1,38.7,-77.0\nSHP1,2,38.8,-77.0\n",
        encoding="utf-8",
    )
    return gtfs


def _make_valid_dfs() -> dict[str, pd.DataFrame]:
    return {
        "stops": pd.DataFrame(
            {"stop_id": ["S1"], "stop_lat": [38.7], "stop_lon": [-77.0], "stop_name": ["Main"]}
        ),
        "routes": pd.DataFrame({"route_id": ["R1"], "route_short_name": ["101"]}),
        "trips": pd.DataFrame(
            {"trip_id": ["T1"], "route_id": ["R1"], "shape_id": ["SHP1"], "direction_id": [0]}
        ),
        "stop_times": pd.DataFrame({"trip_id": ["T1"], "stop_id": ["S1"]}),
        "shapes": pd.DataFrame(
            {
                "shape_id": ["SHP1"],
                "shape_pt_sequence": [1],
                "shape_pt_lat": [38.7],
                "shape_pt_lon": [-77.0],
            }
        ),
    }


# ---------------------------------------------------------------------------
# _ensure_output
# ---------------------------------------------------------------------------


def test_ensure_output_creates_nested_directory(tmp_path: Path) -> None:
    out = tmp_path / "level1" / "level2"
    _ensure_output(out)
    assert out.is_dir()


def test_ensure_output_returns_path_object(tmp_path: Path) -> None:
    result = _ensure_output(tmp_path / "out")
    assert isinstance(result, Path)


def test_ensure_output_existing_directory_is_ok(tmp_path: Path) -> None:
    _ensure_output(tmp_path)  # already exists – should not raise


# ---------------------------------------------------------------------------
# _read_gtfs
# ---------------------------------------------------------------------------


def test_read_gtfs_from_directory_returns_all_tables(tmp_path: Path) -> None:
    gtfs = _make_gtfs_dir(tmp_path)
    dfs = _read_gtfs(gtfs)
    assert set(dfs.keys()) == {"stops", "routes", "trips", "stop_times", "shapes"}


def test_read_gtfs_from_directory_stops_has_rows(tmp_path: Path) -> None:
    gtfs = _make_gtfs_dir(tmp_path)
    dfs = _read_gtfs(gtfs)
    assert len(dfs["stops"]) == 1


def test_read_gtfs_from_zip(tmp_path: Path) -> None:
    gtfs = _make_gtfs_dir(tmp_path)
    zip_path = tmp_path / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in gtfs.iterdir():
            zf.write(f, f.name)
    dfs = _read_gtfs(zip_path)
    assert "stops" in dfs
    assert "shapes" in dfs


def test_read_gtfs_invalid_path_type_raises(tmp_path: Path) -> None:
    bad = tmp_path / "feed.csv"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory or .zip"):
        _read_gtfs(bad)


# ---------------------------------------------------------------------------
# _validate_gtfs
# ---------------------------------------------------------------------------


def test_validate_gtfs_passes_with_all_required_columns() -> None:
    _validate_gtfs(_make_valid_dfs())  # must not raise


def test_validate_gtfs_raises_on_missing_stop_column() -> None:
    dfs = _make_valid_dfs()
    dfs["stops"] = dfs["stops"].drop(columns=["stop_name"])
    with pytest.raises(ValueError, match="stop_name"):
        _validate_gtfs(dfs)


def test_validate_gtfs_raises_on_missing_trips_column() -> None:
    dfs = _make_valid_dfs()
    dfs["trips"] = dfs["trips"].drop(columns=["direction_id"])
    with pytest.raises(ValueError, match="direction_id"):
        _validate_gtfs(dfs)


# ---------------------------------------------------------------------------
# _filter_routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def routes_and_trips() -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.DataFrame(
        {
            "route_id": ["R1", "R2", "R3"],
            "route_short_name": ["101", "202", "9999A"],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1", "T2", "T3"], "route_id": ["R1", "R2", "R3"]})
    return routes, trips


def test_filter_routes_no_filters_keeps_all(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    r, t = _filter_routes(routes, trips, allow=[], deny=[])
    assert len(r) == 3
    assert len(t) == 3


def test_filter_routes_deny_removes_route(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    r, t = _filter_routes(routes, trips, allow=[], deny=["R3"])
    assert "R3" not in r["route_id"].to_numpy()
    assert "T3" not in t["trip_id"].to_numpy()


def test_filter_routes_allow_restricts_to_listed(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    r, t = _filter_routes(routes, trips, allow=["R1"], deny=[])
    assert list(r["route_id"]) == ["R1"]
    assert list(t["trip_id"]) == ["T1"]


def test_filter_routes_deny_and_allow_combined(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    r, t = _filter_routes(routes, trips, allow=["R1", "R2"], deny=["R2"])
    assert list(r["route_id"]) == ["R1"]


# ---------------------------------------------------------------------------
# _find_left_turns
# ---------------------------------------------------------------------------


def test_find_left_turns_straight_line_has_no_turns() -> None:
    line = LineString([(0, 0), (100, 0), (200, 0)])
    turns = _find_left_turns(line, min_deflect_deg=45.0, min_leg_ft=50.0)
    assert turns == []


def test_find_left_turns_two_point_line_has_no_turns() -> None:
    line = LineString([(0, 0), (100, 0)])
    turns = _find_left_turns(line, min_deflect_deg=45.0, min_leg_ft=50.0)
    assert turns == []


def test_find_left_turns_detects_90_degree_left_turn() -> None:
    # east → north = left turn; cross product of (1,0)×(0,1) = +1 > 0
    line = LineString([(0, 0), (100, 0), (100, 100)])
    turns = _find_left_turns(line, min_deflect_deg=45.0, min_leg_ft=50.0)
    assert len(turns) == 1
    _, angle = turns[0]
    assert angle == pytest.approx(90.0, abs=1.0)


def test_find_left_turns_ignores_right_turn() -> None:
    # east → south = right turn; cross product of (1,0)×(0,-1) = -1 < 0
    line = LineString([(0, 0), (100, 0), (100, -100)])
    turns = _find_left_turns(line, min_deflect_deg=45.0, min_leg_ft=50.0)
    assert turns == []


def test_find_left_turns_ignores_legs_below_minimum_length() -> None:
    # Valid 90° angle but segments are only 10 units long
    line = LineString([(0, 0), (10, 0), (10, 10)])
    turns = _find_left_turns(line, min_deflect_deg=45.0, min_leg_ft=50.0)
    assert turns == []


def test_find_left_turns_dist_along_is_positive() -> None:
    line = LineString([(0, 0), (100, 0), (100, 100)])
    turns = _find_left_turns(line, min_deflect_deg=45.0, min_leg_ft=50.0)
    dist_along, _ = turns[0]
    assert dist_along > 0
