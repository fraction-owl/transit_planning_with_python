from __future__ import annotations

import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point

from scripts.stop_analysis import stop_spacing_flagger_gpd as flagger
from scripts.stop_analysis.stop_spacing_flagger_gpd import (
    UNKNOWN_DIRECTION_ID,
    _build_patterns_gdf,
    _build_routes_gdf,
    _build_shape_lines,
    _build_stops_gdf,
    _ensure_output_folder,
    _export,
    _feet_factor,
    _filter_routes,
    _flag_long_spacing_csv,
    _flag_short_spacing,
    _match_in_order,
    _prepare_tables,
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
        "trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\nT2,S1,1\n",
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
        "stop_times": pd.DataFrame({"trip_id": ["T1"], "stop_id": ["S1"], "stop_sequence": ["1"]}),
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


def test_validate_columns_accepts_trips_without_direction_id() -> None:
    # direction_id is optional in GTFS
    dfs = _make_valid_dfs()
    dfs["trips"] = dfs["trips"].drop(columns=["direction_id"])
    _validate_columns(dfs)  # must not raise


def test_validate_columns_raises_on_missing_stop_sequence() -> None:
    dfs = _make_valid_dfs()
    dfs["stop_times"] = dfs["stop_times"].drop(columns=["stop_sequence"])
    with pytest.raises(ValueError, match="stop_sequence"):
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


def _patterns(rows: list[tuple[str, str, list[str], LineString]], crs: str) -> gpd.GeoDataFrame:
    """Stopping patterns from (route_id, shape_id, stop_ids, line), direction 0."""
    return gpd.GeoDataFrame(
        {
            "route_id": [r[0] for r in rows],
            "direction_id": [0] * len(rows),
            "route_short_name": [r[0] for r in rows],
            "shape_id": [r[1] for r in rows],
            "stop_ids": [tuple(r[2]) for r in rows],
        },
        geometry=[r[3] for r in rows],
        crs=crs,
    )


def _stops(
    xy: dict[str, tuple[float, float]], served: dict[str, list[tuple[str, int]]], crs: str
) -> gpd.GeoDataFrame:
    """Stops at *xy* with the (route_id, direction_id) pairs in *served*."""
    return gpd.GeoDataFrame(
        {"stop_id": list(xy), "stop_name": list(xy), "route_dirs": [served[s] for s in xy]},
        geometry=[Point(x, y) for x, y in xy.values()],
        crs=crs,
    )


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
    line = LineString([(x0, y0), (x0 + 3_000 * ft, y0)])
    patterns = _patterns([("1", "SH1", ["A", "B", "C"], line)], crs)
    stops = _stops(
        {s: (x0 + dx * ft, y0 + dy * ft) for s, (dx, dy) in _STOP_POSITIONS_FT.items()},
        {s: [("1", 0)] if s in "ABC" else [("2", 0)] for s in _STOP_POSITIONS_FT},
        crs,
    )
    return patterns, stops


def test_split_into_segments_length_ft_in_feet(
    spacing_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
) -> None:
    routes, stops = spacing_layers
    segs = _split_into_segments(routes, stops, routes.crs.to_string())
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
    line = LineString([(x * ft, y * ft) for x, y in u_shape])
    routes = _patterns([("1", "U", ["B", "C"], line)], crs)
    stop_xy_ft = {"B": (0, 0), "C": (0, 500), "E": (1_030, 250), "H": (500, 250)}
    stops = _stops(
        {s: (x * ft, y * ft) for s, (x, y) in stop_xy_ft.items()},
        {"B": [("1", 0)], "C": [("1", 0)], "E": [("2", 0)], "H": [("2", 0)]},
        crs,
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
# turn ending 100 ft past S5. Route 1 stops S0-S10 sit every 1,000 ft, 35 ft east
# of the centreline, and S2b is 150 ft east. The SHORT pattern also lists S6,
# 900 ft past its shape's end, as a bad feed might. X is a route 2 stop 30 ft
# west, 3,500 ft along.
_NS_X0, _NS_Y0 = 1_300_000.0, 450_000.0


@pytest.fixture()
def short_turn_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    order = ["S0", "S1", "S2b", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
    routes = _patterns(
        [
            ("1", "FULL", order, LineString([(_NS_X0, _NS_Y0), (_NS_X0, _NS_Y0 + 10_000)])),
            ("1", "SHORT", order[:8], LineString([(_NS_X0, _NS_Y0), (_NS_X0, _NS_Y0 + 5_100)])),
        ],
        "EPSG:2248",
    )
    offsets = {f"S{i}": (35, i * 1_000) for i in range(11)}
    offsets |= {"S2b": (150, 1_500), "X": (-30, 3_500)}
    stops = _stops(
        {s: (_NS_X0 + dx, _NS_Y0 + dy) for s, (dx, dy) in offsets.items()},
        {s: [("2", 0)] if s == "X" else [("1", 0)] for s in offsets},
        "EPSG:2248",
    )
    return routes, stops


def test_flag_short_spacing_counts_stops_beside_grid_aligned_route(
    short_turn_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame], tmp_path: Path
) -> None:
    routes, stops = short_turn_layers
    log_path = tmp_path / "short.txt"
    _flag_short_spacing(routes, stops, 600.0, log_path)
    short = pd.read_csv(log_path, sep="\t")
    # S1-S2b-S2 (500 ft apart) on both shapes. S6 is 900 ft past the short
    # turn's end, so it is skipped there: no S5-S6 pair.
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
    routes = _patterns([("1", "ARC", ["P", "Q", "R"], arc)], "EPSG:2248")
    angles = [d / r for d in (800, 1_600, 2_400)]
    stops = _stops(
        {
            s: (cx + (r + 20) * math.cos(t), cy + (r + 20) * math.sin(t))
            for s, t in zip("PQR", angles)
        },
        {s: [("1", 0)] for s in "PQR"},
        "EPSG:2248",
    )
    segs = _split_into_segments(routes, stops, "EPSG:2248")
    expected = [800.0, 800.0, 800.0, arc.length - 2_400]
    assert segs["length_ft"].tolist() == pytest.approx(expected, abs=1.0)


# ---------------------------------------------------------------------------
# Regression tests: feed reading, patterns, pairs, loops and outputs
# ---------------------------------------------------------------------------

_TO_LL = Transformer.from_crs("EPSG:2248", "EPSG:4326", always_xy=True)


def _write_feed(
    root: Path,
    stops_xy: Mapping[str, tuple[float, float]],
    trips: list[tuple[str, str, str, str, list[str]]],
    shapes: Mapping[str, Sequence[tuple[float, float]]],
    stop_dists: Mapping[str, Sequence[float]] | None = None,
) -> Path:
    """Write a GTFS feed laid out in EPSG:2248 feet around (_NS_X0, _NS_Y0).

    *trips* holds (trip_id, route_id, direction_id, shape_id, stop_ids). With
    *stop_dists*, both stop_times.txt and shapes.txt get shape_dist_traveled.
    """
    root.mkdir()

    def lat_lon(x: float, y: float) -> str:
        lon, lat = _TO_LL.transform(_NS_X0 + x, _NS_Y0 + y)
        return f"{lat:.9f},{lon:.9f}"

    routes = sorted({t[1] for t in trips})
    (root / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        + "".join(f"{s},Stop {s},{lat_lon(*xy)}\n" for s, xy in stops_xy.items()),
        encoding="utf-8",
    )
    (root / "routes.txt").write_text(
        "route_id,route_short_name\n" + "".join(f"{r},{r}\n" for r in routes), encoding="utf-8"
    )
    (root / "trips.txt").write_text(
        "trip_id,route_id,direction_id,shape_id\n"
        + "".join(f"{t},{r},{d},{sh}\n" for t, r, d, sh, _ in trips),
        encoding="utf-8",
    )
    sdt_head = ",shape_dist_traveled" if stop_dists else ""
    (root / "stop_times.txt").write_text(
        f"trip_id,stop_id,stop_sequence{sdt_head}\n"
        + "".join(
            f"{t},{s},{i + 1}" + (f",{stop_dists[t][i]}" if stop_dists else "") + "\n"
            for t, _, _, _, seq in trips
            for i, s in enumerate(seq)
        ),
        encoding="utf-8",
    )
    rows = []
    for sid, pts in shapes.items():
        along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(np.array(pts), axis=0).T))))
        for i, (xy, d) in enumerate(zip(pts, along)):
            rows.append(
                f"{sid},{i + 1},{lat_lon(*xy)}" + (f",{d:.3f}" if stop_dists else "") + "\n"
            )
    (root / "shapes.txt").write_text(
        f"shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon{sdt_head}\n" + "".join(rows),
        encoding="utf-8",
    )
    return root


def _load(
    gtfs: Path, crs: str = "EPSG:2248"
) -> tuple[dict[str, pd.DataFrame], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (tables, stops_gdf, patterns_gdf) for every route in *gtfs*."""
    dfs = _read_gtfs_tables(gtfs)
    _validate_columns(dfs)
    _prepare_tables(dfs)
    stops = _build_stops_gdf(dfs["stops"], dfs["stop_times"], dfs["trips"], dfs["routes"], crs)
    lines = _build_shape_lines(dfs["shapes"], dfs["trips"]["shape_id"], crs)
    patterns = _build_patterns_gdf(dfs["stop_times"], dfs["trips"], dfs["routes"], lines)
    return dfs, stops, patterns


def _pairs(log_path: Path) -> list[tuple[str, str, float]]:
    log = pd.read_csv(log_path, sep="\t", dtype={"begin_stop_id": str, "end_stop_id": str})
    return [
        (b, e, round(d))
        for b, e, d in zip(log["begin_stop_id"], log["end_stop_id"], log["spacing_ft"])
    ]


def test_read_gtfs_tables_keeps_ids_as_text(tmp_path: Path) -> None:
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"007": (0, 30), "NA": (500, 30), "10": (1_000, 30)},
        [("1", "101", "0", "10", ["007", "NA", "10"]), ("2", "101", "1", "", ["10", "007"])],
        {"10": [(0, 0), (2_000, 0)]},
    )
    dfs = _read_gtfs_tables(gtfs)
    assert dfs["stops"]["stop_id"].tolist() == ["007", "NA", "10"]
    assert dfs["routes"]["route_id"].tolist() == ["101"]
    assert dfs["trips"]["shape_id"].tolist() == ["10", ""]
    # Configured IDs are text; integers typed into the config still match.
    for include in (["101"], [101]):
        routes, trips = _filter_routes(dfs["routes"], dfs["trips"], include, [])
        assert len(routes) == 1 and len(trips) == 2


def test_prepare_tables_keeps_trips_with_blank_or_missing_direction() -> None:
    dfs = _make_valid_dfs()
    dfs["trips"] = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1"] * 3,
            "shape_id": ["SHP1"] * 3,
            "direction_id": ["1", "", "x"],
        }
    )
    _prepare_tables(dfs)
    assert dfs["trips"]["direction_id"].tolist() == [1, UNKNOWN_DIRECTION_ID, UNKNOWN_DIRECTION_ID]
    assert dfs["stop_times"]["stop_sequence"].tolist() == [1]

    no_column = _make_valid_dfs()
    no_column["trips"] = no_column["trips"].drop(columns=["direction_id"])
    _prepare_tables(no_column)
    assert no_column["trips"]["direction_id"].tolist() == [UNKNOWN_DIRECTION_ID]


def test_long_spacing_uses_actual_route_direction_pairs(tmp_path: Path) -> None:
    # M is served by A/1 and B/0. It is not an A/0 stop, so it neither splits
    # A/0's 2,000 ft A0-A1 gap nor escapes being flagged in it.
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"A0": (0, 30), "A1": (2_000, 30), "M": (1_000, 30), "N0": (0, 400), "Q": (1_000, 2_000)},
        [
            ("TA0", "A", "0", "SA0", ["A0", "A1"]),
            ("TA1", "A", "1", "SA1", ["M", "N0"]),
            ("TB0", "B", "0", "SB0", ["M", "Q"]),
        ],
        {
            "SA0": [(0, 0), (3_000, 0)],
            "SA1": [(3_000, 60), (0, 60)],
            "SB0": [(1_000, -3_000), (1_000, 3_000)],
        },
    )
    _, stops, patterns = _load(gtfs)
    assert stops.set_index("stop_id").loc["M", "route_dirs"] == [("A", 1), ("B", 0)]
    csv_path = tmp_path / "long.csv"
    _flag_long_spacing_csv(patterns[patterns["shape_id"] == "SA0"], stops, 1_500, 99, csv_path)
    flagged = pd.read_csv(csv_path, dtype=str)
    assert flagged[["start_stop_id", "end_stop_id", "flagged_stop_id"]].to_numpy().tolist() == [
        ["A0", "A1", "M"]
    ]


def test_express_and_local_on_one_shape_are_measured_separately(tmp_path: Path) -> None:
    # The local serves P1 halfway along the express's 2,000 ft P0-P2 gap; R
    # (route F) sits in that gap and must be flagged for the express only.
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"P0": (0, 30), "P1": (1_000, 30), "P2": (2_000, 30), "R": (1_500, 40), "R9": (1_500, 900)},
        [
            ("EXP", "E", "0", "SE", ["P0", "P2"]),
            ("LOC", "E", "0", "SE", ["P0", "P1", "P2"]),
            ("LOC2", "E", "0", "SE", ["P0", "P1", "P2"]),
            ("TF", "F", "0", "SF", ["R", "R9"]),
        ],
        {"SE": [(0, 0), (3_000, 0)], "SF": [(1_500, -500), (1_500, 3_000)]},
    )
    _, stops, patterns = _load(gtfs)
    route_e = patterns[patterns["route_id"] == "E"]
    assert sorted(route_e["stop_ids"]) == [("P0", "P1", "P2"), ("P0", "P2")]

    log_path = tmp_path / "short.txt"
    _flag_short_spacing(route_e, stops, 3_000, log_path)
    assert sorted(_pairs(log_path)) == [
        ("P0", "P1", 1_000),
        ("P0", "P2", 2_000),
        ("P1", "P2", 1_000),
    ]
    csv_path = tmp_path / "long.csv"
    _flag_long_spacing_csv(route_e, stops, 1_500, 99, csv_path)
    flagged = pd.read_csv(csv_path, dtype=str)
    assert flagged[["start_stop_id", "end_stop_id", "flagged_stop_id"]].to_numpy().tolist() == [
        ["P0", "P2", "R"]
    ]


def test_loop_keeps_the_closing_gap_back_to_the_first_stop(tmp_path: Path) -> None:
    # Square loop, 2,000 ft a side; the trip ends where it started (A).
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"A": (-30, -30), "B": (2_030, -30), "C": (2_030, 2_030), "D": (-30, 2_030)},
        [("T", "L", "0", "LOOP", ["A", "B", "C", "D", "A"])],
        {"LOOP": [(0, 0), (2_000, 0), (2_000, 2_000), (0, 2_000), (0, 0)]},
    )
    _, stops, patterns = _load(gtfs)
    log_path = tmp_path / "short.txt"
    _flag_short_spacing(patterns, stops, 2_500, log_path)
    assert _pairs(log_path) == [
        ("A", "B", 2_000),
        ("B", "C", 2_000),
        ("C", "D", 2_000),
        ("D", "A", 2_000),
    ]
    segs = _split_into_segments(patterns, stops, "EPSG:2248")
    assert segs["length_ft"].tolist() == pytest.approx([2_000.0] * 4, abs=0.5)


def test_retraced_street_uses_shape_dist_traveled(tmp_path: Path) -> None:
    # Out along y=0 and back along y=20. Every stop sits at y=25, nearer the
    # return leg, so O3 at x=2,500 fits either pass; shape_dist_traveled says
    # it is on the way out.
    shape = [(0, 0), (3_000, 0), (3_000, 20), (0, 20)]
    stops_xy = {
        "O1": (500, 25),
        "O2": (1_500, 25),
        "O3": (2_500, 25),
        "R1": (2_000, 25),
        "R2": (1_000, 25),
    }
    trip = ("T", "A", "0", "RT", list(stops_xy))
    expected = [("O1", "O2", 1_000), ("O2", "O3", 1_000), ("O3", "R1", 1_520), ("R1", "R2", 1_000)]

    gtfs = _write_feed(
        tmp_path / "gtfs",
        stops_xy,
        [trip],
        {"RT": shape},
        stop_dists={"T": [500, 1_500, 2_500, 4_020, 5_020]},
    )
    _, stops, patterns = _load(gtfs)
    log_path = tmp_path / "short.txt"
    _flag_short_spacing(patterns, stops, 5_000, log_path)
    assert _pairs(log_path) == expected

    # Without it, every stop still keeps the trip's order.
    no_dists = patterns.drop(columns="stop_dists")
    _flag_short_spacing(no_dists, stops, 5_000, log_path)
    pairs = _pairs(log_path)
    assert [p[:2] for p in pairs] == [p[:2] for p in expected]
    assert sum(p[2] for p in pairs) == pytest.approx(sum(p[2] for p in expected), abs=2)


def test_match_in_order_follows_sequence_on_out_and_back_line() -> None:
    # One centreline used both ways: the projections tie, so order decides.
    out_and_back = np.array([[0, 0], [1_000, 0], [0, 0]], dtype=float)
    stops = np.array([[200, -20], [800, -20], [700, 20], [100, 20]], dtype=float)
    along, offset = _match_in_order(out_and_back, stops)
    assert along.tolist() == pytest.approx([200, 800, 1_300, 1_900])
    assert offset.tolist() == pytest.approx([20] * 4)


def test_build_routes_gdf_keeps_every_route_on_a_shared_shape(tmp_path: Path) -> None:
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"S0": (0, 30), "S1": (2_000, 30)},
        [("TA", "A", "0", "S", ["S0", "S1"]), ("TB", "B", "", "S", ["S0", "S1"])],
        {"S": [(0, 0), (3_000, 0)]},
    )
    dfs, _, _ = _load(gtfs)
    lines = _build_shape_lines(dfs["shapes"], dfs["trips"]["shape_id"], "EPSG:2248")
    routes = _build_routes_gdf(lines, dfs["trips"], dfs["routes"], union_shapes=False)
    assert sorted(zip(routes["route_id"], routes["direction_id"])) == [
        ("A", 0),
        ("B", UNKNOWN_DIRECTION_ID),
    ]
    # routes.txt has no route_long_name, which the union must not require
    unioned = _build_routes_gdf(lines, dfs["trips"], dfs["routes"], union_shapes=True)
    assert sorted(unioned["route_id"]) == ["A", "B"]


def test_flag_long_spacing_csv_overwrites_earlier_findings(
    spacing_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame], tmp_path: Path
) -> None:
    routes, stops = spacing_layers
    csv_path = tmp_path / "long.csv"
    _flag_long_spacing_csv(routes, stops, 1_500.0, 99.0, csv_path)
    assert len(pd.read_csv(csv_path)) == 2

    _flag_long_spacing_csv(routes, stops, 5_000.0, 99.0, csv_path)  # nothing this long
    assert pd.read_csv(csv_path).empty
    summary = pd.read_csv(tmp_path / "long_summary.txt", sep="\t")
    assert summary.empty and list(summary.columns) == ["route_id", "direction_id"]


def test_export_writes_list_fields_as_text(tmp_path: Path) -> None:
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1"], "route_id": [["A", "B"]], "direction_id": [[0, 1]]},
        geometry=[Point(_NS_X0, _NS_Y0)],
        crs="EPSG:2248",
    )
    _export(stops, tmp_path, "stops")
    back = gpd.read_file(tmp_path / "stops.shp")
    assert back.loc[0, "route_id"] == "A,B"
    assert back.loc[0, "direction_"] == "0,1"
    assert stops.loc[0, "route_id"] == ["A", "B"]  # caller's frame untouched


def test_main_runs_feed_with_numeric_ids_and_blank_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"01": (0, 30), "02": (300, 30), "03": (2_300, 30), "99": (1_300, 50)},
        [("1", "101", "", "7", ["01", "02", "03"]), ("2", "202", "", "8", ["99"])],
        {"7": [(0, 0), (3_000, 0)], "8": [(1_300, -500), (1_300, 500)]},
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "long_spacing_segments.csv").write_text("stale\n", encoding="utf-8")
    for name, value in {
        "GTFS_PATH": str(gtfs),
        "OUTPUT_FOLDER": str(out),
        "INCLUDE_ROUTE_IDS": ["101"],
        "FILTER_OUT_LIST": [],
    }.items():
        monkeypatch.setattr(flagger, name, value)

    assert flagger.main() == 0
    assert _pairs(out / "short_spacing_segments.txt") == [("01", "02", 300)]
    flagged = pd.read_csv(out / "long_spacing_segments.csv", dtype=str)
    assert flagged[["route_id", "direction_id", "flagged_stop_id"]].to_numpy().tolist() == [
        ["101", str(UNKNOWN_DIRECTION_ID), "99"]
    ]
    exported = gpd.read_file(out / "stops.shp")
    assert sorted(exported["stop_id"]) == ["01", "02", "03"]

    monkeypatch.setattr(flagger, "INCLUDE_ROUTE_IDS", ["no-such-route"])
    assert flagger.main() == 2
