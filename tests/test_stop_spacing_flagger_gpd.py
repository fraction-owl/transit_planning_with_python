from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.stop_analysis.stop_spacing_flagger_gpd import (
    RouteStop,
    _build_deletion_plan,
    _deletion_impact_rows_for_route,
    _drop_stops_from_feed,
    _ensure_output_folder,
    _filter_routes,
    _read_gtfs_tables,
    _resolve_stops_to_delete,
    _validate_columns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gtfs_dir(tmp_path: Path) -> Path:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "stops.txt").write_text(
        "stop_id,stop_lat,stop_lon,stop_name\nS1,38.7,-77.0,Main St\nS2,38.8,-77.0,Oak Ave\n",
        encoding="utf-8",
    )
    (gtfs / "routes.txt").write_text(
        "route_id,route_short_name\nR1,101\nR2,202\n",
        encoding="utf-8",
    )
    (gtfs / "trips.txt").write_text(
        "trip_id,route_id,shape_id,direction_id\nT1,R1,SHP1,0\nT2,R2,SHP2,0\n",
        encoding="utf-8",
    )
    (gtfs / "stop_times.txt").write_text(
        "trip_id,stop_id\nT1,S1\nT1,S2\nT2,S1\n",
        encoding="utf-8",
    )
    (gtfs / "shapes.txt").write_text(
        "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
        "SHP1,1,38.7,-77.0\nSHP1,2,38.8,-77.0\n"
        "SHP2,1,38.7,-77.0\nSHP2,2,38.8,-77.0\n",
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
# _ensure_output_folder
# ---------------------------------------------------------------------------


def test_ensure_output_folder_creates_directory(tmp_path: Path) -> None:
    out = tmp_path / "new" / "nested"
    _ensure_output_folder(out)
    assert out.is_dir()


def test_ensure_output_folder_returns_path(tmp_path: Path) -> None:
    result = _ensure_output_folder(tmp_path / "out")
    assert isinstance(result, Path)


def test_ensure_output_folder_existing_directory_does_not_raise(tmp_path: Path) -> None:
    _ensure_output_folder(tmp_path)  # already exists


# ---------------------------------------------------------------------------
# _read_gtfs_tables
# ---------------------------------------------------------------------------


def test_read_gtfs_tables_from_directory_returns_five_tables(tmp_path: Path) -> None:
    gtfs = _make_gtfs_dir(tmp_path)
    dfs = _read_gtfs_tables(gtfs)
    assert set(dfs.keys()) == {"stops", "routes", "trips", "stop_times", "shapes"}


def test_read_gtfs_tables_stops_has_correct_rows(tmp_path: Path) -> None:
    gtfs = _make_gtfs_dir(tmp_path)
    dfs = _read_gtfs_tables(gtfs)
    assert len(dfs["stops"]) == 2


def test_read_gtfs_tables_from_zip(tmp_path: Path) -> None:
    gtfs = _make_gtfs_dir(tmp_path)
    zip_path = tmp_path / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in gtfs.iterdir():
            zf.write(f, f.name)
    dfs = _read_gtfs_tables(zip_path)
    assert "stops" in dfs
    assert "shapes" in dfs


def test_read_gtfs_tables_raises_on_unsupported_path(tmp_path: Path) -> None:
    bad = tmp_path / "data.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="folder or a .zip"):
        _read_gtfs_tables(bad)


# ---------------------------------------------------------------------------
# _validate_columns
# ---------------------------------------------------------------------------


def test_validate_columns_passes_with_valid_data() -> None:
    _validate_columns(_make_valid_dfs())  # must not raise


def test_validate_columns_raises_on_missing_stops_column() -> None:
    dfs = _make_valid_dfs()
    dfs["stops"] = dfs["stops"].drop(columns=["stop_name"])
    with pytest.raises(ValueError, match="stop_name"):
        _validate_columns(dfs)


def test_validate_columns_raises_on_missing_trips_direction_id() -> None:
    dfs = _make_valid_dfs()
    dfs["trips"] = dfs["trips"].drop(columns=["direction_id"])
    with pytest.raises(ValueError, match="direction_id"):
        _validate_columns(dfs)


def test_validate_columns_raises_on_missing_shapes_column() -> None:
    dfs = _make_valid_dfs()
    dfs["shapes"] = dfs["shapes"].drop(columns=["shape_pt_sequence"])
    with pytest.raises(ValueError, match="shape_pt_sequence"):
        _validate_columns(dfs)


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


def test_filter_routes_empty_filters_keeps_all(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    r, t = _filter_routes(routes, trips, include_ids=[], exclude_ids=[])
    assert len(r) == 3
    assert len(t) == 3


def test_filter_routes_exclude_removes_route(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    r, t = _filter_routes(routes, trips, include_ids=[], exclude_ids=["R3"])
    assert "R3" not in r["route_id"].to_numpy()
    assert "T3" not in t["trip_id"].to_numpy()


def test_filter_routes_include_restricts_to_listed(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    r, t = _filter_routes(routes, trips, include_ids=["R1"], exclude_ids=[])
    assert list(r["route_id"]) == ["R1"]
    assert list(t["trip_id"]) == ["T1"]


def test_filter_routes_exclude_applied_before_include(
    routes_and_trips: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    routes, trips = routes_and_trips
    # Include R1+R2, but exclude R2 → only R1 survives
    r, _ = _filter_routes(routes, trips, include_ids=["R1", "R2"], exclude_ids=["R2"])
    assert list(r["route_id"]) == ["R1"]


# ---------------------------------------------------------------------------
# _resolve_stops_to_delete
# ---------------------------------------------------------------------------


@pytest.fixture()
def deletion_stops() -> pd.DataFrame:
    # "B" is a stop_id AND another stop's stop_code; "200" is shared by two stops.
    return pd.DataFrame(
        {
            "stop_id": ["A", "B", "C", "D", "E"],
            "stop_code": ["100", "200", "200", "B", "500"],
            "stop_name": ["a", "b", "c", "d", "e"],
        }
    )


def test_resolve_stops_matches_plain_stop_id(deletion_stops: pd.DataFrame) -> None:
    assert _resolve_stops_to_delete(deletion_stops, ["A"]) == {"A"}


def test_resolve_stops_shared_stop_code_fans_out(deletion_stops: pd.DataFrame) -> None:
    assert _resolve_stops_to_delete(deletion_stops, ["200"]) == {"B", "C"}


def test_resolve_stops_stop_code_wins_over_stop_id(deletion_stops: pd.DataFrame) -> None:
    # "B" is D's stop_code and B's stop_id; stop_code has priority.
    assert _resolve_stops_to_delete(deletion_stops, ["B"]) == {"D"}


def test_resolve_stops_without_stop_code_column(deletion_stops: pd.DataFrame) -> None:
    stops = deletion_stops.drop(columns=["stop_code"])
    assert _resolve_stops_to_delete(stops, ["B"]) == {"B"}


def test_resolve_stops_ignores_blank_entries(deletion_stops: pd.DataFrame) -> None:
    assert _resolve_stops_to_delete(deletion_stops, ["A", "", "  "]) == {"A"}


def test_resolve_stops_total_miss_logs_format_hint(
    deletion_stops: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        result = _resolve_stops_to_delete(deletion_stops, ["9999"])
    assert result == set()
    assert "None of the STOPS_TO_DELETE entries matched" in caplog.text
    assert "stop_ids look like [A, B, C]" in caplog.text


def test_resolve_stops_total_miss_without_stop_code_says_so(
    deletion_stops: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    stops = deletion_stops.drop(columns=["stop_code"])
    with caplog.at_level(logging.ERROR):
        _resolve_stops_to_delete(stops, ["9999"])
    assert "no stop_code column" in caplog.text


def test_resolve_stops_partial_miss_skips_format_hint(
    deletion_stops: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = _resolve_stops_to_delete(deletion_stops, ["A", "9999"])
    assert result == {"A"}
    assert "Unmatched identifiers" in caplog.text
    assert "None of the STOPS_TO_DELETE entries matched" not in caplog.text


def test_resolve_stops_custom_label_in_messages(
    deletion_stops: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        _resolve_stops_to_delete(deletion_stops, ["9999"], label="STOPS_TO_DELETE_BY_ROUTE['R1']")
    assert "STOPS_TO_DELETE_BY_ROUTE['R1']" in caplog.text


# ---------------------------------------------------------------------------
# _build_deletion_plan
# ---------------------------------------------------------------------------


def test_build_deletion_plan_resolves_both_scopes(deletion_stops: pd.DataFrame) -> None:
    routes = pd.DataFrame({"route_id": ["R1", "R2"], "route_short_name": ["101", "202"]})
    global_ids, by_route = _build_deletion_plan(
        deletion_stops, routes, ["A"], {"R1": ["E"], "R2": ["200"]}
    )
    assert global_ids == {"A"}
    assert by_route == {"R1": {"E"}, "R2": {"B", "C"}}


def test_build_deletion_plan_skips_unknown_route_key(
    deletion_stops: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    routes = pd.DataFrame({"route_id": ["R1"], "route_short_name": ["101"]})
    with caplog.at_level(logging.WARNING):
        global_ids, by_route = _build_deletion_plan(
            deletion_stops, routes, [], {"NOPE": ["A"], "R1": ["E"]}
        )
    assert global_ids == set()
    assert by_route == {"R1": {"E"}}
    assert "does not match any route_id" in caplog.text


def test_build_deletion_plan_drops_route_with_no_matches(deletion_stops: pd.DataFrame) -> None:
    routes = pd.DataFrame({"route_id": ["R1"], "route_short_name": ["101"]})
    _, by_route = _build_deletion_plan(deletion_stops, routes, [], {"R1": ["9999"]})
    assert by_route == {}


# ---------------------------------------------------------------------------
# _drop_stops_from_feed
# ---------------------------------------------------------------------------


@pytest.fixture()
def deletion_feed(deletion_stops: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # T1 runs on R1, T2 on R2; stop B is served by both routes.
    return {
        "stops": deletion_stops,
        "trips": pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "route_id": ["R1", "R2"],
                "shape_id": ["SHP1", "SHP2"],
                "direction_id": [0, 0],
            }
        ),
        "stop_times": pd.DataFrame(
            {
                "trip_id": ["T1", "T1", "T1", "T2", "T2"],
                "stop_id": ["A", "B", "C", "B", "E"],
            }
        ),
        "routes": pd.DataFrame({"route_id": ["R1", "R2"], "route_short_name": ["101", "202"]}),
    }


def test_drop_stops_global_removes_everywhere(deletion_feed: dict[str, pd.DataFrame]) -> None:
    out = _drop_stops_from_feed(deletion_feed, {"B"})
    assert list(out["stop_times"]["stop_id"]) == ["A", "C", "E"]
    assert "B" not in set(out["stops"]["stop_id"])


def test_drop_stops_by_route_only_affects_that_route(
    deletion_feed: dict[str, pd.DataFrame],
) -> None:
    out = _drop_stops_from_feed(deletion_feed, set(), {"R1": {"B"}})
    # B's row on R1's trip T1 is gone; B's row on R2's trip T2 survives.
    assert list(out["stop_times"]["stop_id"]) == ["A", "C", "B", "E"]


def test_drop_stops_by_route_keeps_stop_in_stops_txt(
    deletion_feed: dict[str, pd.DataFrame],
) -> None:
    out = _drop_stops_from_feed(deletion_feed, set(), {"R1": {"B"}})
    assert "B" in set(out["stops"]["stop_id"])


def test_drop_stops_does_not_mutate_input(deletion_feed: dict[str, pd.DataFrame]) -> None:
    _drop_stops_from_feed(deletion_feed, {"B"}, {"R2": {"E"}})
    assert len(deletion_feed["stop_times"]) == 5
    assert len(deletion_feed["stops"]) == 5


def test_drop_stops_shares_untouched_tables(deletion_feed: dict[str, pd.DataFrame]) -> None:
    out = _drop_stops_from_feed(deletion_feed, {"B"})
    assert out["routes"] is deletion_feed["routes"]
    assert out["trips"] is deletion_feed["trips"]


# ---------------------------------------------------------------------------
# _deletion_impact_rows_for_route
# ---------------------------------------------------------------------------


def _stop_sequence() -> list[RouteStop]:
    return [
        RouteStop("s1", "First", 0.0),
        RouteStop("s2", "Second", 500.0),
        RouteStop("s3", "Third", 900.0),
        RouteStop("s4", "Fourth", 1200.0),
        RouteStop("s5", "Fifth", 1500.0),
        RouteStop("s6", "Sixth", 2000.0),
        RouteStop("s7", "Last", 2400.0),
    ]


def _impact_rows(deleted: set[str], global_ids: set[str] | None = None) -> list[dict[str, object]]:
    return _deletion_impact_rows_for_route(
        _stop_sequence(),
        deleted,
        deleted if global_ids is None else global_ids,
        ft_factor=1.0,
        long_threshold_ft=700.0,
        route_id="R1",
        route_short="101",
        direction_id=0,
    )


def test_impact_rows_interior_deletion_merges_gap() -> None:
    (row,) = _impact_rows({"s2"})
    assert row["prev_stop_id"] == "s1"
    assert row["next_stop_id"] == "s3"
    assert row["new_spacing_ft"] == 900.0
    assert row["old_max_spacing_ft"] == 500.0
    assert row["exceeds_long_ft"] == "yes"


def test_impact_rows_consecutive_run_groups_into_one_row() -> None:
    (row,) = _impact_rows({"s4", "s5"})
    assert row["deleted_stop_ids"] == "s4,s5"
    assert row["n_deleted"] == 2
    assert row["new_spacing_ft"] == 1100.0  # s3 (900) → s6 (2000)


def test_impact_rows_gap_at_threshold_not_flagged() -> None:
    # Deleting s3 merges s2 (500) → s4 (1200): exactly 700, not over it.
    (row,) = _impact_rows({"s3"})
    assert row["new_spacing_ft"] == 700.0
    assert row["exceeds_long_ft"] == "no"


def test_impact_rows_terminal_deletion_gets_note_not_flag() -> None:
    (row,) = _impact_rows({"s7"})
    assert row["new_spacing_ft"] == ""
    assert row["exceeds_long_ft"] == ""
    assert "route end" in str(row["note"])


def test_impact_rows_all_stops_deleted_note() -> None:
    (row,) = _impact_rows({"s1", "s2", "s3", "s4", "s5", "s6", "s7"})
    assert "all served stops" in str(row["note"])


def test_impact_rows_scope_all_routes() -> None:
    (row,) = _impact_rows({"s2"}, global_ids={"s2"})
    assert row["deletion_scope"] == "all routes"


def test_impact_rows_scope_this_route() -> None:
    (row,) = _impact_rows({"s2"}, global_ids=set())
    assert row["deletion_scope"] == "this route"


def test_impact_rows_scope_mixed() -> None:
    (row,) = _impact_rows({"s2", "s3"}, global_ids={"s2"})
    assert row["deletion_scope"] == "mixed"


def test_impact_rows_no_deletions_returns_empty() -> None:
    assert _impact_rows(set()) == []
