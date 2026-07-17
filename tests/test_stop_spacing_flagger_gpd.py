from __future__ import annotations

import math
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from scripts.stop_analysis.stop_spacing_flagger_gpd import (
    _apply_proposed_coords,
    _ensure_output_folder,
    _evaluate_proposed_spacing,
    _filter_routes,
    _load_proposed_stops,
    _read_gtfs_tables,
    _resolve_proposed_stops,
    _run_proposed_stops_qa,
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
# _load_proposed_stops
# ---------------------------------------------------------------------------


def test_load_proposed_stops_none_returns_empty() -> None:
    assert _load_proposed_stops(None) == []


def test_load_proposed_stops_empty_list_returns_empty() -> None:
    assert _load_proposed_stops([]) == []


def test_load_proposed_stops_from_list_normalizes_types() -> None:
    result = _load_proposed_stops([(1001, "38.5", "-77.1")])
    assert result == [("1001", 38.5, -77.1)]


def test_load_proposed_stops_from_comma_file(tmp_path: Path) -> None:
    f = tmp_path / "proposed.txt"
    f.write_text("stop_id,new_lat,new_lon\nS1,38.5,-77.1\nS2,38.6,-77.2\n", encoding="utf-8")
    result = _load_proposed_stops(f)
    assert result == [("S1", 38.5, -77.1), ("S2", 38.6, -77.2)]


def test_load_proposed_stops_from_tab_file_with_alt_headers(tmp_path: Path) -> None:
    f = tmp_path / "proposed.txt"
    f.write_text("stop_code\tlat\tlon\n0042\t38.5\t-77.1\n", encoding="utf-8")
    result = _load_proposed_stops(f)
    # Leading zeros in the identifier must survive.
    assert result == [("0042", 38.5, -77.1)]


def test_load_proposed_stops_file_missing_columns_raises(tmp_path: Path) -> None:
    f = tmp_path / "proposed.txt"
    f.write_text("id,y,x\nS1,38.5,-77.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identifier column"):
        _load_proposed_stops(f)


def test_load_proposed_stops_wrong_tuple_size_raises() -> None:
    with pytest.raises(ValueError, match="exactly three fields"):
        _load_proposed_stops([("S1", 38.5)])  # type: ignore[list-item]


def test_load_proposed_stops_non_numeric_coords_raises() -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        _load_proposed_stops([("S1", "north", "-77.1")])


def test_load_proposed_stops_out_of_range_coords_raises() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        _load_proposed_stops([("S1", 138.5, -77.1)])


# ---------------------------------------------------------------------------
# _resolve_proposed_stops
# ---------------------------------------------------------------------------


@pytest.fixture()
def stops_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "stop_code": ["1001", "1002", "S1"],
            "stop_name": ["Main", "Oak", "Elm"],
            "stop_lat": [38.7, 38.71, 38.72],
            "stop_lon": [-77.0, -77.0, -77.0],
        }
    )


def test_resolve_proposed_stops_matches_stop_id(stops_master: pd.DataFrame) -> None:
    resolved = _resolve_proposed_stops([("S2", 38.9, -77.3)], stops_master)
    assert resolved == {"S2": (38.9, -77.3)}


def test_resolve_proposed_stops_falls_back_to_stop_code(stops_master: pd.DataFrame) -> None:
    resolved = _resolve_proposed_stops([("1001", 38.9, -77.3)], stops_master)
    assert resolved == {"S1": (38.9, -77.3)}


def test_resolve_proposed_stops_stop_id_wins_over_stop_code(stops_master: pd.DataFrame) -> None:
    # "S1" is both a stop_id and S3's stop_code → stop_id must win.
    resolved = _resolve_proposed_stops([("S1", 38.9, -77.3)], stops_master)
    assert resolved == {"S1": (38.9, -77.3)}


def test_resolve_proposed_stops_drops_unmatched(stops_master: pd.DataFrame) -> None:
    resolved = _resolve_proposed_stops([("NOPE", 38.9, -77.3)], stops_master)
    assert resolved == {}


def test_resolve_proposed_stops_last_duplicate_wins(stops_master: pd.DataFrame) -> None:
    resolved = _resolve_proposed_stops([("S1", 38.9, -77.3), ("S1", 38.95, -77.35)], stops_master)
    assert resolved == {"S1": (38.95, -77.35)}


def test_resolve_proposed_stops_works_without_stop_code_column(
    stops_master: pd.DataFrame,
) -> None:
    resolved = _resolve_proposed_stops(
        [("S1", 38.9, -77.3)], stops_master.drop(columns=["stop_code"])
    )
    assert resolved == {"S1": (38.9, -77.3)}


# ---------------------------------------------------------------------------
# _apply_proposed_coords / _evaluate_proposed_spacing
# ---------------------------------------------------------------------------

_TEST_CRS = "EPSG:2263"  # feet-based, so spacing_ft == projected units


def _make_spacing_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build a straight 3000-ft test route with four evenly spaced stops.

    Geometries are constructed directly in the projected CRS so distances
    are exact: stops at 0 / 1000 / 2000 / 3000 ft along the line.
    """
    xs = [0.0, 1_000.0, 2_000.0, 3_000.0]
    stops_gdf = gpd.GeoDataFrame(
        {
            "stop_id": ["S1", "S2", "S3", "S4"],
            "stop_name": ["A", "B", "C", "D"],
            "route_id": [["R1"]] * 4,
            "direction_id": [[0]] * 4,
        },
        geometry=[Point(x, 0.0) for x in xs],
        crs=_TEST_CRS,
    )
    routes_gdf = gpd.GeoDataFrame(
        {
            "route_id": ["R1"],
            "direction_id": [0],
            "route_short_name": ["101"],
        },
        geometry=[LineString([(0.0, 0.0), (3_000.0, 0.0)])],
        crs=_TEST_CRS,
    )
    return routes_gdf, stops_gdf


def test_apply_proposed_coords_moves_only_targeted_stop() -> None:
    _, stops_gdf = _make_spacing_layers()
    moved = _apply_proposed_coords(stops_gdf, {"S2": (40.7, -74.0)}, _TEST_CRS)

    expected = gpd.GeoSeries([Point(-74.0, 40.7)], crs="EPSG:4326").to_crs(_TEST_CRS).iloc[0]
    new_s2 = moved.loc[moved["stop_id"] == "S2", "geometry"].iloc[0]
    assert new_s2.distance(expected) < 1e-6

    old_s1 = stops_gdf.loc[stops_gdf["stop_id"] == "S1", "geometry"].iloc[0]
    assert moved.loc[moved["stop_id"] == "S1", "geometry"].iloc[0].equals(old_s1)


def test_apply_proposed_coords_leaves_input_untouched() -> None:
    _, stops_gdf = _make_spacing_layers()
    original = stops_gdf.geometry.copy()
    _apply_proposed_coords(stops_gdf, {"S2": (40.7, -74.0)}, _TEST_CRS)
    assert stops_gdf.geometry.equals(original)


def test_apply_proposed_coords_unserved_stop_is_ignored() -> None:
    _, stops_gdf = _make_spacing_layers()
    moved = _apply_proposed_coords(stops_gdf, {"UNSERVED": (40.7, -74.0)}, _TEST_CRS)
    assert moved.geometry.equals(stops_gdf.geometry)


def _proposed_with_move(stops_gdf: gpd.GeoDataFrame, stop_id: str, x: float) -> gpd.GeoDataFrame:
    proposed = stops_gdf.copy()
    proposed.loc[proposed["stop_id"] == stop_id, "geometry"] = Point(x, 0.0)
    return proposed


def test_evaluate_proposed_spacing_flags_short_and_long() -> None:
    routes_gdf, stops_gdf = _make_spacing_layers()
    # Move S2 from 1000 ft → 1800 ft: S1–S2 becomes 1800 ft (too long),
    # S2–S3 becomes 200 ft (too short).
    proposed = _proposed_with_move(stops_gdf, "S2", 1_800.0)

    report = _evaluate_proposed_spacing(
        routes_gdf, stops_gdf, proposed, {"S2"}, min_spacing_ft=400.0, long_spacing_ft=1_500.0
    )

    assert len(report) == 2
    first, second = report.iloc[0], report.iloc[1]
    assert (first.begin_stop_id, first.end_stop_id) == ("S1", "S2")
    assert first.spacing_ft_after == 1_800.0
    assert first.spacing_ft_before == 1_000.0
    assert first.verdict == "too long"
    assert not first.compliant
    assert (second.begin_stop_id, second.end_stop_id) == ("S2", "S3")
    assert second.spacing_ft_after == 200.0
    assert second.verdict == "too short"
    assert not second.compliant


def test_evaluate_proposed_spacing_compliant_move() -> None:
    routes_gdf, stops_gdf = _make_spacing_layers()
    proposed = _proposed_with_move(stops_gdf, "S2", 1_200.0)

    report = _evaluate_proposed_spacing(
        routes_gdf, stops_gdf, proposed, {"S2"}, min_spacing_ft=400.0, long_spacing_ft=1_500.0
    )

    assert list(report["verdict"]) == ["OK", "OK"]
    assert list(report["compliant"]) == [True, True]
    assert list(report["spacing_ft_after"]) == [1_200.0, 800.0]


def test_evaluate_proposed_spacing_handles_reordering() -> None:
    routes_gdf, stops_gdf = _make_spacing_layers()
    # Move S2 past S3 (1000 ft → 2500 ft): new order is S1, S3, S2, S4.
    proposed = _proposed_with_move(stops_gdf, "S2", 2_500.0)

    report = _evaluate_proposed_spacing(
        routes_gdf, stops_gdf, proposed, {"S2"}, min_spacing_ft=400.0, long_spacing_ft=1_500.0
    )

    pairs = list(zip(report["begin_stop_id"], report["end_stop_id"]))
    assert pairs == [("S3", "S2"), ("S2", "S4")]
    assert list(report["spacing_ft_after"]) == [500.0, 500.0]
    # Before the move S3 sat 1000 ft beyond S2 along the line.
    assert list(report["spacing_ft_before"]) == [1_000.0, 2_000.0]


def test_evaluate_proposed_spacing_untouched_pairs_are_excluded() -> None:
    routes_gdf, stops_gdf = _make_spacing_layers()
    proposed = _proposed_with_move(stops_gdf, "S2", 1_200.0)

    report = _evaluate_proposed_spacing(
        routes_gdf, stops_gdf, proposed, {"S2"}, min_spacing_ft=400.0, long_spacing_ft=1_500.0
    )

    # S3–S4 does not touch the moved stop and must not be reported.
    assert ("S3", "S4") not in list(zip(report["begin_stop_id"], report["end_stop_id"]))
    assert all("S2" == m for m in report["moved_stop_id"])


def test_evaluate_proposed_spacing_no_moved_stops_returns_empty_with_columns() -> None:
    routes_gdf, stops_gdf = _make_spacing_layers()
    report = _evaluate_proposed_spacing(
        routes_gdf, stops_gdf, stops_gdf.copy(), set(), 400.0, 1_500.0
    )
    assert report.empty
    assert "spacing_ft_after" in report.columns
    assert "verdict" in report.columns


# ---------------------------------------------------------------------------
# _run_proposed_stops_qa (end-to-end smoke test)
# ---------------------------------------------------------------------------


def _make_geographic_layers() -> tuple[
    gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, list[float]
]:
    """Build a WGS84-sourced route/stops pair projected to the test CRS."""
    lats = [40.70, 40.71, 40.72, 40.73]
    lon = -74.0
    pts = gpd.GeoSeries([Point(lon, lat) for lat in lats], crs="EPSG:4326").to_crs(_TEST_CRS)

    stops_gdf = gpd.GeoDataFrame(
        {
            "stop_id": ["S1", "S2", "S3", "S4"],
            "stop_name": ["A", "B", "C", "D"],
            "route_id": [["R1"]] * 4,
            "direction_id": [[0]] * 4,
        },
        geometry=list(pts),
        crs=_TEST_CRS,
    )
    routes_gdf = gpd.GeoDataFrame(
        {"route_id": ["R1"], "direction_id": [0], "route_short_name": ["101"]},
        geometry=[LineString([(p.x, p.y) for p in pts])],
        crs=_TEST_CRS,
    )
    stops_master = pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3", "S4"],
            "stop_name": ["A", "B", "C", "D"],
            "stop_lat": lats,
            "stop_lon": [lon] * 4,
        }
    )
    return routes_gdf, stops_gdf, stops_master, lats


def test_run_proposed_stops_qa_writes_csv_and_map(tmp_path: Path) -> None:
    routes_gdf, stops_gdf, stops_master, lats = _make_geographic_layers()
    # Nudge S2 towards S3: still on the line, spacing changes on both sides.
    proposal = [("S2", (lats[1] + lats[2]) / 2.0, -74.0)]
    csv_path = tmp_path / "proposed_spacing_compliance.csv"

    _run_proposed_stops_qa(
        proposal,
        stops_master,
        routes_gdf,
        stops_gdf,
        _TEST_CRS,
        min_spacing_ft=400.0,
        long_spacing_ft=1_000_000.0,
        csv_path=csv_path,
        export_maps=True,
    )

    assert csv_path.is_file()
    report = pd.read_csv(csv_path)
    assert set(report["moved_stop_id"]) == {"S2"}
    assert len(report) == 2
    total_before = report["spacing_ft_before"].sum()
    total_after = report["spacing_ft_after"].sum()
    assert math.isclose(total_before, total_after, rel_tol=0.01)
    assert (tmp_path / "proposed_maps" / "S2.png").is_file()


def test_run_proposed_stops_qa_noop_without_proposals(tmp_path: Path) -> None:
    routes_gdf, stops_gdf, stops_master, _ = _make_geographic_layers()
    csv_path = tmp_path / "proposed_spacing_compliance.csv"

    _run_proposed_stops_qa(
        None,
        stops_master,
        routes_gdf,
        stops_gdf,
        _TEST_CRS,
        400.0,
        1_500.0,
        csv_path,
        export_maps=True,
    )

    assert not csv_path.exists()
    assert not (tmp_path / "proposed_maps").exists()


def test_run_proposed_stops_qa_no_csv_when_nothing_matches(tmp_path: Path) -> None:
    routes_gdf, stops_gdf, stops_master, _ = _make_geographic_layers()
    csv_path = tmp_path / "proposed_spacing_compliance.csv"

    _run_proposed_stops_qa(
        [("NOPE", 40.7, -74.0)],
        stops_master,
        routes_gdf,
        stops_gdf,
        _TEST_CRS,
        400.0,
        1_500.0,
        csv_path,
        export_maps=True,
    )

    assert not csv_path.exists()
