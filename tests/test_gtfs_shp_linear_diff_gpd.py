"""Tests for scripts/stop_analysis/gtfs_shp_linear_diff_gpd.py.

Synthetic feeds and reference shapefiles are drawn in EPSG:2248 (Maryland State
Plane, US feet) around a DC origin, so distances in the fixtures read as feet.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import LineString

from scripts.stop_analysis import gtfs_shp_linear_diff_gpd as mod

_X0, _Y0 = 1_300_000.0, 450_000.0
_TO_LL = Transformer.from_crs("EPSG:2248", "EPSG:4326", always_xy=True)

# ---------------------------------------------------------------------------
# Synthetic network (feet, relative to the origin)
# ---------------------------------------------------------------------------

# Route 10: an east-west local. Direction 1 runs a one-way couplet on a street
# 600 ft north between x=4000 and x=7000; a short turn and a Saturday detour
# exist but should not be picked on a weekday.
S10_0 = [(0, 0), (10_000, 0)]
S10_0_SHORT = [(0, 0), (5_000, 0)]
S10_1 = [(10_000, 20), (7_000, 20), (7_000, 600), (4_000, 600), (4_000, 20), (0, 20)]
S10_SAT = [(0, 0), (0, -3_000), (10_000, -3_000)]
# Route 7: no direction_id (directions come from trip_headsign).
S7_E = [(0, 5_000), (8_000, 5_000)]
S7_W = [(8_000, 5_015), (0, 5_015)]
# Route 40: only in GTFS.
S40 = [(0, -8_000), (6_000, -8_000)]

SHAPES = {
    "S10_0": S10_0,
    "S10_0_SHORT": S10_0_SHORT,
    "S10_1": S10_1,
    "S10_SAT": S10_SAT,
    "S7_E": S7_E,
    "S7_W": S7_W,
    "S40": S40,
}

# (route_id, service_id, direction_id, headsign, shape_id, trip count)
TRIP_SPECS = [
    ("R10", "WK", "0", "East", "S10_0", 20),
    ("R10", "WK", "0", "East", "S10_0_SHORT", 5),
    ("R10", "WK", "1", "West", "S10_1", 18),
    ("R10", "SAT", "0", "East", "S10_SAT", 30),
    ("R10", "SAT", "1", "West", "S10_1", 30),
    ("R7", "WK", "", "East", "S7_E", 12),
    ("R7", "WK", "", "West", "S7_W", 12),
    ("R30", "WK", "0", "North", "", 8),
    ("R40", "WK", "0", "South", "S40", 6),
]

ROUTES = [("R10", "10"), ("R7", "7"), ("R30", "30"), ("R40", "40")]

# Reference lines: route 10 drawn as a main line plus the couplet street,
# route 7 drawn with a detour (a real mismatch), 30 present, 99 extra.
REF_10_MAIN = [(0, 0), (10_000, 0)]
REF_10_COUPLET = [(7_000, 0), (7_000, 600), (4_000, 600), (4_000, 0)]
REF_7 = [(0, 5_000), (3_000, 5_000), (3_000, 6_000), (6_000, 6_000), (6_000, 5_000), (8_000, 5_000)]
REF_30 = [(0, 9_000), (5_000, 9_000)]
REF_99 = [(0, 20_000), (5_000, 20_000)]


def _abs(points: Sequence[tuple[float, float]]) -> LineString:
    return LineString([(_X0 + x, _Y0 + y) for x, y in points])


def _lat_lon(x: float, y: float) -> str:
    lon, lat = _TO_LL.transform(_X0 + x, _Y0 + y)
    return f"{lat:.9f},{lon:.9f}"


def write_feed(
    root: Path,
    trip_specs: Sequence[tuple[str, str, str, str, str, int]] = tuple(TRIP_SPECS),
    shapes: Mapping[str, Sequence[tuple[float, float]]] = SHAPES,
    routes: Sequence[tuple[str, str]] = tuple(ROUTES),
    calendar: bool = True,
) -> Path:
    """Write a GTFS folder from trip specs; returns the folder."""
    root.mkdir(parents=True)
    (root / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name\n"
        + "".join(f"{rid},{short},Route {short}\n" for rid, short in routes),
        encoding="utf-8",
    )
    trip_rows = []
    for rid, sid, direction, headsign, shape_id, n in trip_specs:
        for i in range(n):
            trip_rows.append(
                f"{rid}_{sid}_{shape_id}_{i},{rid},{sid},{direction},{headsign},{shape_id}\n"
            )
    (root / "trips.txt").write_text(
        "trip_id,route_id,service_id,direction_id,trip_headsign,shape_id\n" + "".join(trip_rows),
        encoding="utf-8",
    )
    shape_rows = [
        f"{sid},{i + 1},{_lat_lon(x, y)}\n"
        for sid, pts in shapes.items()
        for i, (x, y) in enumerate(pts)
    ]
    (root / "shapes.txt").write_text(
        "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n" + "".join(shape_rows),
        encoding="utf-8",
    )
    if calendar:
        (root / "calendar.txt").write_text(
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
            "WK,1,1,1,1,1,0,0,20260105,20260327\n"
            "SAT,0,0,0,0,0,1,0,20260105,20260327\n",
            encoding="utf-8",
        )
    return root


def write_reference(
    path: Path,
    features: Sequence[tuple[Any, Sequence[tuple[float, float]]]],
    field: str = "ROUTE",
) -> Path:
    """Write a reference shapefile in EPSG:2248 with one route-name field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {field: [name for name, _ in features]},
        geometry=[_abs(pts) for _, pts in features],
        crs="EPSG:2248",
    )
    gdf.to_file(path)
    return path


DEFAULT_REFERENCE = [
    ("10", REF_10_MAIN),
    ("10", REF_10_COUPLET),
    ("007", REF_7),
    ("30", REF_30),
    ("99", REF_99),
]


def make_config(gtfs: Path, shapes: Path, out: Path, **overrides: Any) -> mod.Config:
    settings: dict[str, Any] = {
        "shp_route_field": "ROUTE",
        "target_crs": "EPSG:2248",
        "service_date": "",
    }
    settings.update(overrides)
    return mod.Config(gtfs_path=gtfs, route_shapes_path=shapes, output_dir=out, **settings)


def route_table(out: Path) -> pd.DataFrame:
    return pd.read_csv(
        out / mod.ROUTE_TABLE_FILENAME,
        dtype={"route_key": str, "route_label": str},
        keep_default_na=False,
    ).set_index("route_key")


# ---------------------------------------------------------------------------
# Route names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ignore", "expected"),
    [
        ("007", True, "7"),
        ("0", True, "0"),
        ("000", True, "0"),
        ("0A", True, "0A"),
        ("010", False, "010"),
        (" purple ", True, "PURPLE"),
        ("", True, ""),
    ],
)
def test_normalize_route_key(text: str, ignore: bool, expected: str) -> None:
    assert mod.normalize_route_key(text, ignore) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(10.0, "10"), (10.5, "10.5"), (float("nan"), ""), (None, ""), (" 12 ", "12"), (7, "7")],
)
def test_route_name_text_drops_float_suffix(value: Any, expected: str) -> None:
    assert mod.route_name_text(value) == expected


def test_apply_route_pattern() -> None:
    pattern = re.compile(r"(?i)route[_ -]*([A-Za-z0-9]+)")
    assert mod.apply_route_pattern("Route_10", pattern) == ("10", True)
    assert mod.apply_route_pattern("ROUTE 50H east", pattern) == ("50H", True)
    assert mod.apply_route_pattern("Circulator", pattern) == ("Circulator", False)
    assert mod.apply_route_pattern("Circulator", None) == ("Circulator", True)
    no_group = re.compile(r"\d+")
    assert mod.apply_route_pattern("Line 42", no_group) == ("42", True)


def test_compile_route_pattern_rejects_bad_regex() -> None:
    assert mod.compile_route_pattern("  ") is None
    with pytest.raises(mod.ConfigError, match="SHP_ROUTE_PATTERN"):
        mod.compile_route_pattern("route_(")


def test_gtfs_route_keys_fall_back_to_route_id() -> None:
    routes = pd.DataFrame(
        {"route_id": ["R1", "R2"], "route_short_name": ["01", ""], "route_long_name": ["A", "B"]}
    )
    keys = mod.gtfs_route_keys(routes, "route_short_name", ignore_leading_zeros=True)
    assert keys == {"R1": ("1", "01"), "R2": ("R2", "B")}
    with pytest.raises(mod.ConfigError, match="route_desc"):
        mod.gtfs_route_keys(routes, "route_desc", ignore_leading_zeros=True)


# ---------------------------------------------------------------------------
# Pattern selection
# ---------------------------------------------------------------------------


def _trips(rows: Sequence[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["route_id", "service_id", "direction_id", "trip_headsign", "shape_id"]
    ).assign(trip_id=lambda d: [f"t{i}" for i in range(len(d))])


def test_direction_keys_fall_back_to_headsign() -> None:
    trips = _trips(
        [("R", "WK", "0", "East", "A"), ("R", "WK", "", "West", "B"), ("R", "WK", "", "", "C")]
    )
    assert mod.direction_keys(trips).tolist() == ["0", "headsign:West", ""]


def test_pick_shapes_rules_and_ties() -> None:
    counts = pd.Series({"B": 10, "A": 10, "C": 3})
    lengths = {"A": 1_000.0, "B": 2_000.0, "C": 9_000.0}
    assert mod._pick_shapes(counts, lengths, "most_trips") == [("B", 10)]  # tie -> longer
    assert mod._pick_shapes(counts, lengths, "longest") == [("C", 3)]
    assert mod._pick_shapes(counts, lengths, "all") == [("B", 10), ("A", 10), ("C", 3)]
    same = {"A": 1_000.0, "B": 1_000.0, "C": 1.0}
    assert mod._pick_shapes(counts, same, "most_trips") == [("A", 10)]  # then lower id


def test_select_patterns_counts_the_service_day_only() -> None:
    trips = _trips(
        [("R", "WK", "0", "", "WEEK")] * 3
        + [("R", "SAT", "0", "", "SAT")] * 9
        + [("R", "WK", "1", "", "BACK")] * 2
        + [("R", "WK", "1", "", "")]
    )
    lengths = {"WEEK": 100.0, "SAT": 100.0, "BACK": 100.0}
    keys = {"R": ("R", "R")}
    choices, notes = mod.select_patterns(trips, lengths, {"WK"}, "most_trips", keys)
    assert [(c.direction, c.shape_id, c.trips, c.direction_trips) for c in choices] == [
        ("0", "WEEK", 3, 3),
        ("1", "BACK", 2, 3),
    ]
    assert all(c.day_basis == "service_day" for c in choices)
    assert notes["R"] == ["1 counted trip(s) have no usable shape"]


def test_select_patterns_falls_back_to_all_trips() -> None:
    trips = _trips([("H", "HOL", "0", "", "LOOP")] * 4)
    choices, notes = mod.select_patterns(
        trips, {"LOOP": 50.0}, {"WK"}, "most_trips", {"H": ("H", "H")}
    )
    assert [(c.shape_id, c.day_basis) for c in choices] == [("LOOP", "all_trips")]
    assert notes["H"] == ["no trips on the service day; counted all trips"]


# ---------------------------------------------------------------------------
# Geometry (planar coordinates stand in for a metric CRS)
# ---------------------------------------------------------------------------


def test_combine_lines_counts_a_two_way_street_once() -> None:
    out = LineString([(0, 0), (3_000, 0)])
    back = LineString([(3_000, 8), (0, 8)])  # same street, drawn 8 m off
    combined = mod.combine_lines([out, back], buffer_dist=25.0, min_segment_dist=45.0)
    assert combined is not None
    assert combined.length == pytest.approx(3_000.0)


def test_combine_lines_keeps_both_streets_of_a_couplet() -> None:
    out = LineString([(0, 0), (3_000, 0)])
    back = LineString([(3_000, 5), (2_000, 5), (2_000, 200), (1_000, 200), (1_000, 5), (0, 5)])
    combined = mod.combine_lines([out, back], buffer_dist=25.0, min_segment_dist=45.0)
    assert combined is not None
    # 3000 + the 1000 m parallel street + two connectors from 25 m to 200 m.
    assert combined.length == pytest.approx(3_000 + 1_000 + 2 * 175, rel=0.01)


def test_combine_lines_drops_terminal_stubs_and_empty_input() -> None:
    out = LineString([(0, 0), (3_000, 0)])
    loop = LineString([(3_000, 0), (3_030, 30), (3_000, 10), (0, 10)])  # tiny turnaround
    combined = mod.combine_lines([out, loop], buffer_dist=25.0, min_segment_dist=45.0)
    assert combined is not None
    assert combined.length == pytest.approx(3_000.0)
    assert mod.combine_lines([], buffer_dist=25.0, min_segment_dist=45.0) is None
    assert mod.combine_lines([LineString()], buffer_dist=25.0, min_segment_dist=45.0) is None


def test_compare_alignments_splits_a_detour() -> None:
    gtfs = LineString([(0, 0), (3_000, 0)])
    ref = LineString([(0, 0), (1_000, 0), (1_000, 500), (2_000, 500), (2_000, 0), (3_000, 0)])
    classes = mod.compare_alignments(gtfs, ref, buffer_dist=25.0, min_segment_dist=45.0)
    lengths = {name: sum(line.length for line in lines) for name, lines in classes.items()}
    assert lengths["matched"] == pytest.approx(2_000 + 50, rel=0.02)
    assert lengths["gtfs_only"] == pytest.approx(1_000 - 50, rel=0.02)
    assert lengths["shp_only"] == pytest.approx(1_000 + 2 * 475, rel=0.02)


def test_compare_alignments_identical_lines_match() -> None:
    line = LineString([(0, 0), (3_000, 0)])
    classes = mod.compare_alignments(
        line, translate(line, 0, 10), buffer_dist=25.0, min_segment_dist=45.0
    )
    assert classes["gtfs_only"] == [] and classes["shp_only"] == []
    assert sum(part.length for part in classes["matched"]) == pytest.approx(3_000.0)


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------


def test_reference_routes_combine_directions_and_numeric_names(tmp_path: Path) -> None:
    shp = write_reference(
        tmp_path / "ref.shp",
        [
            (10.0, [(0, 0), (6_000, 0)]),
            (10.0, [(6_000, 15), (0, 15)]),
            (7.0, [(0, 900), (900, 900)]),
        ],
        field="RTE_NUM",
    )
    refs = mod.load_reference_routes(shp, "rte_num", None, True, "EPSG:2248", 80.0, 150.0)
    assert sorted(refs) == ["10", "7"]
    assert refs["10"].feature_count == 2
    assert refs["10"].geometry.length == pytest.approx(6_000.0)  # EPSG:2248 is in feet
    assert refs["10"].names == ("10",)


def test_reference_routes_from_folder_and_zip(tmp_path: Path) -> None:
    folder = tmp_path / "per_route"
    write_reference(folder / "Route_010.shp", [("x", REF_10_MAIN)], field="NOTE")
    write_reference(folder / "Route_99.shp", [("x", REF_99)], field="NOTE")
    pattern = re.compile(r"(?i)route_(\w+)")
    from_folder = mod.load_reference_routes(folder, "", pattern, True, "EPSG:2248", 25.0, 45.0)
    assert sorted(from_folder) == ["10", "99"]
    assert from_folder["10"].sources == ("Route_010",)

    archive = tmp_path / "routes.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in folder.iterdir():
            zf.write(f, f"exports/{f.name}")
    from_zip = mod.load_reference_routes(archive, "", pattern, True, "EPSG:2248", 25.0, 45.0)
    assert sorted(from_zip) == ["10", "99"]


def test_reference_routes_reject_missing_field_and_crs(tmp_path: Path) -> None:
    shp = write_reference(tmp_path / "ref.shp", [("10", REF_10_MAIN)])
    with pytest.raises(mod.ConfigError, match="no field 'LINE'.*ROUTE"):
        mod.load_reference_routes(shp, "LINE", None, True, "EPSG:2248", 25.0, 45.0)
    shp.with_suffix(".prj").unlink()
    with pytest.raises(ValueError, match="no coordinate system"):
        mod.load_reference_routes(shp, "ROUTE", None, True, "EPSG:2248", 25.0, 45.0)
    with pytest.raises(OSError, match="does not exist"):
        mod.load_reference_routes(
            tmp_path / "nope.shp", "ROUTE", None, True, "EPSG:2248", 25.0, 45.0
        )
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No .shp files"):
        mod.load_reference_routes(tmp_path / "empty", "", None, True, "EPSG:2248", 25.0, 45.0)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_run_classifies_every_route(tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    shp = write_reference(tmp_path / "ref" / "routes.shp", DEFAULT_REFERENCE)
    out = tmp_path / "out"
    summary = mod.run(make_config(gtfs, shp, out))

    table = route_table(out)
    assert table["status"].to_dict() == {
        "7": "mismatch",
        "40": "missing_from_shp",
        "99": "missing_from_gtfs",
        "30": "no_gtfs_geometry",
        "10": "match",
    }
    assert list(table.index) == ["7", "40", "99", "30", "10"]  # mismatches first

    r10 = table.loc["10"]
    assert r10["shape_ids"] == "S10_0;S10_1"  # weekday picks, short turn and Saturday left out
    assert r10["gtfs_only_len_ft"] == 0 and r10["shp_only_len_ft"] == 0
    # Both directions on the shared street count once; the couplet street adds ~4000 ft.
    assert r10["gtfs_len_ft"] == pytest.approx(14_000, rel=0.02)

    r7 = table.loc["7"]
    assert r7["gtfs_route_ids"] == "R7" and r7["shp_route_names"] == "007"
    assert "trip_headsign" in r7["note"]
    assert r7["gtfs_only_len_ft"] == pytest.approx(3_000 - 160, rel=0.03)
    assert r7["shp_only_len_ft"] == pytest.approx(3_000 + 2 * 920, rel=0.03)
    assert "no trip has a usable shapes.txt shape" in table.loc["30", "note"]

    patterns = pd.read_csv(out / mod.PATTERNS_CSV_FILENAME, dtype=str, keep_default_na=False)
    r10_patterns = patterns[patterns["route_id"] == "R10"]
    assert r10_patterns[["direction", "shape_id", "trips"]].to_numpy().tolist() == [
        ["0", "S10_0", "20"],
        ["1", "S10_1", "18"],
    ]
    assert set(patterns.loc[patterns["route_id"] == "R7", "direction"]) == {
        "headsign:East",
        "headsign:West",
    }

    assert (summary.match_count, summary.mismatch_count) == (1, 1)
    assert summary.service_date.startswith("2026-")
    saved = json.loads((out / mod.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert saved["missing_from_gtfs_count"] == 1

    gtfs_only = gpd.read_file(out / mod.SEGMENTS_GTFS_ONLY_FILENAME)
    assert gtfs_only.crs.to_epsg() == 4326
    assert set(gtfs_only["rt_status"]) == {"mismatch", "missing_from_shp"}
    shp_only = gpd.read_file(out / mod.SEGMENTS_SHP_ONLY_FILENAME)
    assert set(shp_only["rt_key"]) == {"7", "99"}
    combined = gpd.read_file(out / mod.COMBINED_ROUTES_FILENAME)
    assert sorted(combined["rt_key"]) == ["10", "40", "7"]
    selected = gpd.read_file(out / mod.PATTERNS_SHP_FILENAME)
    assert len(selected) == len(patterns)

    log = (out / mod.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "PATTERN_RULE" in log and "Match / mismatch:         1 / 1" in log


def test_run_pattern_rules_and_service_day(tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    shp = write_reference(tmp_path / "ref" / "routes.shp", DEFAULT_REFERENCE)

    mod.run(make_config(gtfs, shp, tmp_path / "all", pattern_rule="all"))
    table = route_table(tmp_path / "all")
    assert table.loc["10", "shape_ids"] == "S10_0;S10_1;S10_0_SHORT"
    assert table.loc["10", "status"] == "match"  # the short turn adds nothing new

    mod.run(make_config(gtfs, shp, tmp_path / "sat", service_day="saturday"))
    table = route_table(tmp_path / "sat")
    assert table.loc["10", "shape_ids"] == "S10_SAT;S10_1"
    assert table.loc["10", "status"] == "mismatch"
    assert "no trips on the service day" in table.loc["7", "note"]

    mod.run(make_config(gtfs, shp, tmp_path / "date", service_date="2026-01-10"))  # a Saturday
    assert route_table(tmp_path / "date").loc["10", "shape_ids"] == "S10_SAT;S10_1"


def test_run_without_calendar_counts_every_trip(tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs", calendar=False)
    shp = write_reference(tmp_path / "ref" / "routes.shp", DEFAULT_REFERENCE)
    summary = mod.run(make_config(gtfs, shp, tmp_path / "out"))
    assert summary.service_date == "all trips"
    # Saturday's 30 detour trips now outnumber the weekday's 20 eastbound ones.
    assert route_table(tmp_path / "out").loc["10", "shape_ids"] == "S10_1;S10_SAT"


def test_main_with_zipped_feed_and_per_route_folder(tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in gtfs.iterdir():
            zf.write(f, f"feed/{f.name}")
    folder = tmp_path / "per_route"
    write_reference(folder / "Route_10_a.shp", [("x", REF_10_MAIN)], field="NOTE")
    write_reference(folder / "Route_10_b.shp", [("x", REF_10_COUPLET)], field="NOTE")
    write_reference(folder / "Route_7.shp", [("x", S7_E)], field="NOTE")
    out = tmp_path / "out"
    code = mod.main(
        [
            "--gtfs",
            str(archive),
            "--route-shapes",
            str(folder),
            "--shp-route-pattern",
            r"(?i)route_([0-9]+)",
            "--output-dir",
            str(out),
            "--target-crs",
            "EPSG:2248",
        ]
    )
    assert code == 0
    table = route_table(out)
    assert table.loc["10", "status"] == "match"
    assert table.loc["10", "shp_sources"] == "Route_10_a;Route_10_b"
    assert table.loc["7", "status"] == "match"


def test_main_auto_utm_matches_state_plane(tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    shp = write_reference(tmp_path / "ref" / "routes.shp", DEFAULT_REFERENCE)
    out = tmp_path / "out"
    args = ["--gtfs", str(gtfs), "--route-shapes", str(shp), "--output-dir", str(out)]
    assert mod.main([*args, "--shp-route-field", "ROUTE"]) == 0
    table = route_table(out).sort_index()
    assert table.loc["10", "status"] == "match" and table.loc["7", "status"] == "mismatch"
    summary = json.loads((out / mod.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["projected_crs"] == "EPSG:32618"
    # The same run in State Plane feet measures the same lengths.
    assert mod.main([*args, "--shp-route-field", "ROUTE", "--target-crs", "EPSG:2248"]) == 0
    feet = route_table(out).sort_index()
    assert feet["gtfs_len_ft"].to_numpy() == pytest.approx(
        table["gtfs_len_ft"].to_numpy(), rel=1e-3
    )
    assert feet["shp_only_len_ft"].to_numpy() == pytest.approx(
        table["shp_only_len_ft"].to_numpy(), rel=1e-2
    )


def test_main_returns_2_for_configuration_problems(tmp_path: Path) -> None:
    assert mod.main([]) == 2  # placeholders
    gtfs = write_feed(tmp_path / "gtfs")
    shp = write_reference(tmp_path / "ref" / "routes.shp", DEFAULT_REFERENCE)
    base = ["--gtfs", str(gtfs), "--route-shapes", str(shp), "--output-dir", str(tmp_path / "o")]
    assert mod.main([*base, "--shp-route-field", "LINE"]) == 2
    assert mod.main([*base, "--shp-route-field", "ROUTE", "--shp-route-pattern", "("]) == 2
    assert mod.main([*base, "--shp-route-field", "ROUTE", "--service-date", "03/10/2026"]) == 2
    assert mod.main([*base, "--shp-route-field", "ROUTE", "--buffer-feet", "0"]) == 2
    assert mod.main([*base, "--shp-route-field", "ROUTE", "--service-day", "funday"]) == 2
    assert mod.main([*base, "--shp-route-field", "ROUTE", "--target-crs", "EPSG:4326"]) == 2
    assert mod.main([*base, "--shp-route-field", "ROUTE", "--target-crs", "EPSG:nope"]) == 2


@pytest.mark.parametrize(("crs", "feet"), [("EPSG:2248", 1.000002), ("EPSG:32618", 3.280840)])
def test_crs_feet_per_unit(crs: str, feet: float) -> None:
    assert mod.crs_feet_per_unit(crs) == pytest.approx(feet, abs=1e-6)


def test_main_returns_1_for_missing_inputs(tmp_path: Path) -> None:
    shp = write_reference(tmp_path / "ref" / "routes.shp", DEFAULT_REFERENCE)
    args = ["--gtfs", str(tmp_path / "missing"), "--route-shapes", str(shp)]
    assert mod.main([*args, "--output-dir", str(tmp_path / "o")]) == 1
    gtfs = write_feed(tmp_path / "gtfs")
    (gtfs / "shapes.txt").unlink()
    args = ["--gtfs", str(gtfs), "--route-shapes", str(shp)]
    assert mod.main([*args, "--output-dir", str(tmp_path / "o")]) == 1
