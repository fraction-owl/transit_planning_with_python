from __future__ import annotations

import math
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from scripts.stop_analysis.stop_spacing_flagger_gpd import (
    _ensure_output_folder,
    _feet_factor,
    _filter_routes,
    _flag_long_spacing_csv,
    _flag_short_spacing,
    _read_gtfs_tables,
    _split_into_segments,
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
# Spacing checks across CRS units
# ---------------------------------------------------------------------------

# (along-route ft, offset ft) on a straight 3,000 ft route. A, B and C serve
# route 1: A-B is 300 ft (short) and B-C is 2,000 ft (long). D, F and G serve
# route 2 inside the B-C gap; G lies beyond the 99 ft near buffer.
_STOP_POSITIONS_FT: dict[str, tuple[float, float]] = {
    "A": (0, 0),
    "B": (300, 0),
    "C": (2_300, 0),
    "D": (1_300, 50),
    "F": (360, 30),
    "G": (360, 150),
}


@pytest.fixture(
    params=[
        # CRS units per foot (a US survey foot is within 2 ppm of a foot).
        pytest.param(("EPSG:2263", 1.0), id="EPSG:2263-ftUS"),
        pytest.param(("EPSG:2248", 1.0), id="EPSG:2248-ftUS"),
        pytest.param(("EPSG:26918", 0.3048), id="EPSG:26918-m"),
    ]
)
def crs_units(request: pytest.FixtureRequest) -> tuple[str, float]:
    return request.param


@pytest.fixture()
def spacing_layers(crs_units: tuple[str, float]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    crs, ft = crs_units
    x0, y0 = 1_000_000 * ft, 500_000 * ft
    routes = gpd.GeoDataFrame(
        {"route_id": ["1"], "direction_id": [0], "route_short_name": ["1"]},
        geometry=[LineString([(x0, y0), (x0 + 3_000 * ft, y0)])],
        crs=crs,
    )
    stops = gpd.GeoDataFrame(
        {
            "stop_id": list(_STOP_POSITIONS_FT),
            "stop_name": list(_STOP_POSITIONS_FT),
            "route_id": [["1"] if s in "ABC" else ["2"] for s in _STOP_POSITIONS_FT],
            "direction_id": [[0]] * len(_STOP_POSITIONS_FT),
        },
        geometry=[Point(x0 + dx * ft, y0 + dy * ft) for dx, dy in _STOP_POSITIONS_FT.values()],
        crs=crs,
    )
    return routes, stops


def test_split_into_segments_length_ft_in_feet(
    spacing_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
) -> None:
    routes, stops = spacing_layers
    served = stops[stops["stop_id"].isin(["A", "B", "C"])]
    segs = _split_into_segments(routes, served, routes.crs.to_string())
    assert segs["length_ft"].tolist() == pytest.approx([300.0, 2_000.0, 700.0], abs=0.1)


def test_flag_short_spacing_flags_pair_under_threshold(
    spacing_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame], tmp_path: Path
) -> None:
    routes, stops = spacing_layers
    log_path = tmp_path / "short.txt"
    _flag_short_spacing(routes, stops, 400.0, log_path)
    short = pd.read_csv(log_path, sep="\t")
    assert list(zip(short["begin_stop_id"], short["end_stop_id"])) == [("A", "B")]
    assert short["spacing_ft"].tolist() == pytest.approx([300.0], abs=0.1)


def test_flag_long_spacing_csv_flags_near_stops_along_whole_gap(
    spacing_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame], tmp_path: Path
) -> None:
    routes, stops = spacing_layers
    csv_path = tmp_path / "long.csv"
    _flag_long_spacing_csv(routes, stops, 1_500.0, 99.0, csv_path, summary=False)
    flagged = pd.read_csv(csv_path).set_index("flagged_stop_id")
    # D sits 1,000 ft into the gap; G is 150 ft off the route, beyond the buffer.
    assert sorted(flagged.index) == ["D", "F"]
    assert set(zip(flagged["start_stop_id"], flagged["end_stop_id"])) == {("B", "C")}
    assert flagged["seg_len_ft"].tolist() == pytest.approx([2_000.0, 2_000.0], abs=0.1)
    assert flagged.loc["D", "dist_to_route_ft"] == pytest.approx(50.0, abs=0.1)
    assert flagged.loc["F", "dist_to_route_ft"] == pytest.approx(30.0, abs=0.1)


def test_flag_long_spacing_csv_searches_around_curves(
    crs_units: tuple[str, float], tmp_path: Path
) -> None:
    # U-shaped route: east 1,000 ft, north 500 ft, west 1,000 ft. The gap runs
    # from B at the start to C at the end, directly above B, so a box around
    # the two end stops misses E beside the far leg. H is inside the search
    # box but 250 ft from the route.
    crs, ft = crs_units
    u_shape = [(0, 0), (1_000, 0), (1_000, 500), (0, 500)]
    routes = gpd.GeoDataFrame(
        {"route_id": ["1"], "direction_id": [0], "route_short_name": ["1"]},
        geometry=[LineString([(x * ft, y * ft) for x, y in u_shape])],
        crs=crs,
    )
    stop_xy_ft = {"B": (0, 0), "C": (0, 500), "E": (1_030, 250), "H": (500, 250)}
    stops = gpd.GeoDataFrame(
        {
            "stop_id": list(stop_xy_ft),
            "stop_name": list(stop_xy_ft),
            "route_id": [["1"], ["1"], ["2"], ["2"]],
            "direction_id": [[0]] * len(stop_xy_ft),
        },
        geometry=[Point(x * ft, y * ft) for x, y in stop_xy_ft.values()],
        crs=crs,
    )
    csv_path = tmp_path / "long.csv"
    _flag_long_spacing_csv(routes, stops, 1_500.0, 99.0, csv_path, summary=False)
    flagged = pd.read_csv(csv_path)
    assert flagged["flagged_stop_id"].tolist() == ["E"]
    assert flagged["dist_to_route_ft"].tolist() == pytest.approx([30.0], abs=0.1)


@pytest.mark.parametrize(
    ("crs", "match"),
    [("EPSG:4326", "not projected"), (None, "No CRS")],
)
def test_feet_factor_rejects_crs_without_linear_unit(crs: str | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _feet_factor(crs)


# ---------------------------------------------------------------------------
# Served stops and segment splitting
# ---------------------------------------------------------------------------

# A north-south route in EPSG:2248, whose grid is aligned with DC's meridian, so
# the route's bounding box has no width. FULL runs 10,000 ft; SHORT is a short
# turn ending 100 ft past its last stop (S5). Route 1 stops S0-S10 sit every
# 1,000 ft, 35 ft east of the centreline, and S2b is 150 ft east. X is a route 2
# stop 30 ft west, 3,500 ft along.
_NS_X0, _NS_Y0 = 1_300_000.0, 450_000.0


@pytest.fixture()
def short_turn_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    routes = gpd.GeoDataFrame(
        {"route_id": ["1", "1"], "direction_id": [0, 0], "route_short_name": ["1", "1"]},
        geometry=[
            LineString([(_NS_X0, _NS_Y0), (_NS_X0, _NS_Y0 + 10_000)]),
            LineString([(_NS_X0, _NS_Y0), (_NS_X0, _NS_Y0 + 5_100)]),
        ],
        crs="EPSG:2248",
    )
    offsets = {f"S{i}": (35, i * 1_000) for i in range(11)}
    offsets |= {"S2b": (150, 1_500), "X": (-30, 3_500)}
    stops = gpd.GeoDataFrame(
        {
            "stop_id": list(offsets),
            "stop_name": list(offsets),
            "route_id": [["2"] if s == "X" else ["1"] for s in offsets],
            "direction_id": [[0]] * len(offsets),
        },
        geometry=[Point(_NS_X0 + dx, _NS_Y0 + dy) for dx, dy in offsets.values()],
        crs="EPSG:2248",
    )
    return routes, stops


def test_flag_short_spacing_counts_stops_beside_grid_aligned_route(
    short_turn_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame], tmp_path: Path
) -> None:
    routes, stops = short_turn_layers
    log_path = tmp_path / "short.txt"
    _flag_short_spacing(routes, stops, 600.0, log_path)
    short = pd.read_csv(log_path, sep="\t")
    # S1-S2b-S2 (500 ft apart) on both shapes. S6 onward are 900+ ft past the
    # short turn's end, so they are not served by it: no S5-S6 pair.
    assert sorted(zip(short["begin_stop_id"], short["end_stop_id"])) == sorted(
        [("S1", "S2b"), ("S2b", "S2")] * 2
    )


def test_flag_long_spacing_csv_counts_stops_beside_grid_aligned_route(
    short_turn_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame], tmp_path: Path
) -> None:
    routes, stops = short_turn_layers
    csv_path = tmp_path / "long.csv"
    _flag_long_spacing_csv(routes, stops, 900.0, 99.0, csv_path, summary=False)
    flagged = pd.read_csv(csv_path)
    rows = zip(flagged["start_stop_id"], flagged["end_stop_id"], flagged["flagged_stop_id"])
    assert list(rows) == [("S3", "S4", "X")] * 2  # X in the S3-S4 gap of each shape


def test_split_into_segments_keeps_route_ends_and_skips_stops_past_short_turn(
    short_turn_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
) -> None:
    routes, stops = short_turn_layers
    segs = _split_into_segments(routes, stops, "EPSG:2248")
    full = [1_000.0, 500.0, 500.0] + [1_000.0] * 8
    short_turn = [1_000.0, 500.0, 500.0, 1_000.0, 1_000.0, 1_000.0, 100.0]
    assert segs["length_ft"].tolist() == pytest.approx(full + short_turn, abs=0.1)


def test_split_into_segments_cuts_curved_route_at_each_stop() -> None:
    # Quarter circle of radius 2,000 ft drawn with 90 chords, and stops 20 ft
    # outside it about 800, 1,600 and 2,400 ft along. A stop projected onto a
    # curve rarely lands exactly on it, which left shapely's split unsplit.
    cx, cy, r = 1_300_000.0, 450_000.0, 2_000.0
    arc = LineString(
        [
            (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in range(91)
        ]
    )
    routes = gpd.GeoDataFrame(
        {"route_id": ["1"], "direction_id": [0], "route_short_name": ["1"]},
        geometry=[arc],
        crs="EPSG:2248",
    )
    angles = [d / r for d in (800, 1_600, 2_400)]
    stops = gpd.GeoDataFrame(
        {
            "stop_id": ["P", "Q", "R"],
            "stop_name": ["P", "Q", "R"],
            "route_id": [["1"]] * 3,
            "direction_id": [[0]] * 3,
        },
        geometry=[Point(cx + (r + 20) * math.cos(t), cy + (r + 20) * math.sin(t)) for t in angles],
        crs="EPSG:2248",
    )
    segs = _split_into_segments(routes, stops, "EPSG:2248")
    expected = [800.0, 800.0, 800.0, arc.length - 2_400]
    assert segs["length_ft"].tolist() == pytest.approx(expected, abs=1.0)
