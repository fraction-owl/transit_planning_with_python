from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point

from scripts.stop_analysis.gtfs_linear_diff_gpd import (
    Config,
    build_chord_lines,
    build_correspondence,
    build_route_geometries,
    build_shape_geometries,
    diff_alignments,
    extract_lines,
    filter_short_lines,
    jaccard,
    load_feed,
    main,
    route_stop_sets,
    run,
    total_length_ft,
)

# ---------------------------------------------------------------------------
# extract_lines / filter_short_lines
# ---------------------------------------------------------------------------


def test_extract_lines_passes_linestring_through() -> None:
    line = LineString([(0, 0), (10, 0)])
    assert extract_lines(line) == [line]


def test_extract_lines_merges_touching_parts() -> None:
    mls = MultiLineString([[(0, 0), (5, 0)], [(5, 0), (10, 0)]])
    parts = extract_lines(mls)
    assert len(parts) == 1
    assert parts[0].length == pytest.approx(10.0)


def test_extract_lines_drops_points_from_collections() -> None:
    collection = GeometryCollection([Point(1, 1), LineString([(0, 0), (4, 0)])])
    parts = extract_lines(collection)
    assert len(parts) == 1
    assert parts[0].length == pytest.approx(4.0)


def test_extract_lines_empty_geometry_yields_nothing() -> None:
    assert extract_lines(LineString()) == []


def test_filter_short_lines_drops_slivers() -> None:
    lines = [LineString([(0, 0), (3, 0)]), LineString([(0, 0), (100, 0)])]
    kept = filter_short_lines(lines, min_length_m=10.0)
    assert len(kept) == 1
    assert kept[0].length == pytest.approx(100.0)


def test_total_length_ft_converts_meters() -> None:
    lines = [LineString([(0, 0), (100, 0)])]
    assert total_length_ft(lines) == pytest.approx(328.0839895, rel=1e-9)


# ---------------------------------------------------------------------------
# diff_alignments (planar coordinates stand in for a metric CRS)
# ---------------------------------------------------------------------------


def test_diff_alignments_identical_lines_are_fully_retained() -> None:
    line = LineString([(0, 0), (2000, 0)])
    classes = diff_alignments(line, line, buffer_m=25.0, min_segment_m=45.0)
    assert classes["new"] == []
    assert classes["eliminated"] == []
    assert sum(part.length for part in classes["retained"]) == pytest.approx(2000.0, rel=0.01)


def test_diff_alignments_jitter_within_buffer_is_retained() -> None:
    before = LineString([(0, 0), (2000, 0)])
    after = LineString([(0, 10), (2000, 10)])  # re-digitized 10 m off
    classes = diff_alignments(before, after, buffer_m=25.0, min_segment_m=45.0)
    assert classes["new"] == []
    assert classes["eliminated"] == []
    assert classes["retained"]


def test_diff_alignments_parallel_move_is_new_plus_eliminated() -> None:
    before = LineString([(0, 0), (2000, 0)])
    after = LineString([(0, 500), (2000, 500)])  # moved to a parallel street
    classes = diff_alignments(before, after, buffer_m=25.0, min_segment_m=45.0)
    assert sum(part.length for part in classes["new"]) == pytest.approx(2000.0, rel=0.01)
    assert sum(part.length for part in classes["eliminated"]) == pytest.approx(2000.0, rel=0.01)
    assert classes["retained"] == []


def test_diff_alignments_partial_reroute_mixes_classes() -> None:
    before = LineString([(0, 0), (3000, 0)])
    after = LineString([(0, 0), (1000, 0), (1000, 800), (2000, 800), (2000, 0), (3000, 0)])
    classes = diff_alignments(before, after, buffer_m=25.0, min_segment_m=45.0)
    retained = sum(part.length for part in classes["retained"])
    new = sum(part.length for part in classes["new"])
    eliminated = sum(part.length for part in classes["eliminated"])
    assert retained == pytest.approx(2000.0, rel=0.05)  # shared first/last km
    assert new == pytest.approx(2600.0, rel=0.05)  # 800 up + 1000 across + 800 down
    assert eliminated == pytest.approx(1000.0, rel=0.05)  # skipped middle km


def test_diff_alignments_sliver_filter_applies_to_every_class() -> None:
    before = LineString([(0, 0), (100, 0)])
    after = LineString([(0, 5000), (100, 5000)])
    classes = diff_alignments(before, after, buffer_m=25.0, min_segment_m=200.0)
    assert classes["new"] == []
    assert classes["retained"] == []
    assert classes["eliminated"] == []


# ---------------------------------------------------------------------------
# build_shape_geometries / build_chord_lines
# ---------------------------------------------------------------------------


def test_build_shape_geometries_orders_by_sequence() -> None:
    shapes = pd.DataFrame(
        {
            "shape_id": ["S1", "S1", "S1"],
            "shape_pt_lat": ["0.0", "0.0", "0.0"],
            "shape_pt_lon": ["2.0", "0.0", "1.0"],
            "shape_pt_sequence": ["3", "1", "2"],
        }
    )
    geoms = build_shape_geometries(shapes, "test")
    assert list(geoms["S1"].coords) == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


def test_build_shape_geometries_skips_single_point_shapes() -> None:
    shapes = pd.DataFrame(
        {
            "shape_id": ["S1", "S2", "S2"],
            "shape_pt_lat": ["0.0", "1.0", "1.0"],
            "shape_pt_lon": ["0.0", "0.0", "1.0"],
            "shape_pt_sequence": ["1", "1", "2"],
        }
    )
    geoms = build_shape_geometries(shapes, "test")
    assert set(geoms) == {"S2"}


def test_build_shape_geometries_handles_missing_table() -> None:
    assert build_shape_geometries(None, "test") == {}


def test_build_chord_lines_one_line_per_distinct_pattern() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "stop_id": ["A", "B", "A", "B", "B", "C"],
            "stop_sequence": ["1", "2", "1", "2", "1", "2"],
        }
    )
    stop_xy = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)}
    lines = build_chord_lines(frozenset({"T1", "T2", "T3"}), stop_times, stop_xy)
    assert len(lines) == 2  # T1 and T2 share a pattern


def test_build_chord_lines_skips_unlocatable_patterns() -> None:
    stop_times = pd.DataFrame(
        {"trip_id": ["T1", "T1"], "stop_id": ["A", "X"], "stop_sequence": ["1", "2"]}
    )
    lines = build_chord_lines(frozenset({"T1"}), stop_times, {"A": (0.0, 0.0)})
    assert lines == []


# ---------------------------------------------------------------------------
# correspondence
# ---------------------------------------------------------------------------


def test_jaccard_basic() -> None:
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_build_correspondence_detects_rekey() -> None:
    routes_before = pd.DataFrame({"route_id": ["R1", "R9"], "route_short_name": ["1", "9"]})
    routes_after = pd.DataFrame({"route_id": ["R1", "R9X"], "route_short_name": ["1", "9"]})
    stops_before = {"R1": frozenset({"a"}), "R9": frozenset({"x", "y", "z"})}
    stops_after = {"R1": frozenset({"a"}), "R9X": frozenset({"x", "y", "z"})}
    corr = build_correspondence(routes_before, routes_after, stops_before, stops_after, 0.5)
    assert corr.matched == ["R1"]
    assert corr.rekeyed == {"R9": "R9X"}
    assert corr.eliminated == []
    assert corr.added == []


def test_build_correspondence_low_overlap_is_not_a_rekey() -> None:
    routes_before = pd.DataFrame({"route_id": ["R9"], "route_short_name": ["9"]})
    routes_after = pd.DataFrame({"route_id": ["R9X"], "route_short_name": ["9"]})
    corr = build_correspondence(
        routes_before,
        routes_after,
        {"R9": frozenset({"a", "b", "c"})},
        {"R9X": frozenset({"x", "y", "z"})},
        0.5,
    )
    assert corr.eliminated == ["R9"]
    assert corr.added == ["R9X"]


def test_route_stop_sets_joins_trips_to_stops() -> None:
    trips = pd.DataFrame({"route_id": ["R1", "R1"], "trip_id": ["T1", "T2"]})
    stop_times = pd.DataFrame({"trip_id": ["T1", "T1", "T2"], "stop_id": ["a", "b", "c"]})
    assert route_stop_sets(trips, stop_times) == {"R1": frozenset({"a", "b", "c"})}


# ---------------------------------------------------------------------------
# End-to-end run on synthetic feeds (WGS 84 coordinates near Washington, DC)
# ---------------------------------------------------------------------------


def _write_feed(
    feed_dir: Path,
    routes: list[dict],
    trips: list[dict],
    stop_times: list[dict],
    stops: list[dict],
    shapes: list[dict],
) -> None:
    feed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(routes).to_csv(feed_dir / "routes.txt", index=False)
    pd.DataFrame(trips).to_csv(feed_dir / "trips.txt", index=False)
    pd.DataFrame(stop_times).to_csv(feed_dir / "stop_times.txt", index=False)
    pd.DataFrame(stops).to_csv(feed_dir / "stops.txt", index=False)
    if shapes:
        pd.DataFrame(shapes).to_csv(feed_dir / "shapes.txt", index=False)


def _shape_rows(shape_id: str, coords: list[tuple[float, float]]) -> list[dict]:
    return [
        {
            "shape_id": shape_id,
            "shape_pt_lon": lon,
            "shape_pt_lat": lat,
            "shape_pt_sequence": i + 1,
        }
        for i, (lon, lat) in enumerate(coords)
    ]


def _stop_times_rows(trip_id: str, stop_ids: list[str]) -> list[dict]:
    return [
        {"trip_id": trip_id, "stop_id": sid, "stop_sequence": i + 1}
        for i, sid in enumerate(stop_ids)
    ]


STOPS = [
    {"stop_id": "101", "stop_lat": 38.900, "stop_lon": -77.050},
    {"stop_id": "102", "stop_lat": 38.900, "stop_lon": -77.030},
    {"stop_id": "201", "stop_lat": 38.920, "stop_lon": -77.050},
    {"stop_id": "202", "stop_lat": 38.920, "stop_lon": -77.030},
    {"stop_id": "301", "stop_lat": 38.880, "stop_lon": -77.050},
    {"stop_id": "302", "stop_lat": 38.880, "stop_lon": -77.030},
    {"stop_id": "401", "stop_lat": 38.940, "stop_lon": -77.050},
    {"stop_id": "402", "stop_lat": 38.940, "stop_lon": -77.030},
    {"stop_id": "601", "stop_lat": 38.860, "stop_lon": -77.050},
    {"stop_id": "602", "stop_lat": 38.860, "stop_lon": -77.040},
    {"stop_id": "603", "stop_lat": 38.860, "stop_lon": -77.030},
]


@pytest.fixture()
def diff_run(tmp_path: Path) -> tuple:
    """Run the full pipeline on two synthetic feeds; return (summary, out_dir)."""
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    out_dir = tmp_path / "out"

    # Before feed: R1 (straight), R2 (straight), R3 (to be eliminated),
    # R6 (no shapes -> chord fallback).
    _write_feed(
        before_dir,
        routes=[
            {"route_id": "R1", "route_short_name": "1", "route_long_name": "First"},
            {"route_id": "R2", "route_short_name": "2", "route_long_name": "Second"},
            {"route_id": "R3", "route_short_name": "3", "route_long_name": "Third"},
            {"route_id": "R6", "route_short_name": "6", "route_long_name": "Sixth"},
        ],
        trips=[
            {"route_id": "R1", "trip_id": "T1", "shape_id": "S1"},
            {"route_id": "R2", "trip_id": "T2", "shape_id": "S2B"},
            {"route_id": "R3", "trip_id": "T3", "shape_id": "S3"},
            {"route_id": "R6", "trip_id": "T6", "shape_id": ""},
        ],
        stop_times=(
            _stop_times_rows("T1", ["101", "102"])
            + _stop_times_rows("T2", ["201", "202"])
            + _stop_times_rows("T3", ["301", "302"])
            + _stop_times_rows("T6", ["601", "602", "603"])
        ),
        stops=STOPS,
        shapes=(
            _shape_rows("S1", [(-77.050, 38.900), (-77.030, 38.900)])
            + _shape_rows("S2B", [(-77.050, 38.920), (-77.030, 38.920)])
            + _shape_rows("S3", [(-77.050, 38.880), (-77.030, 38.880)])
        ),
    )

    # After feed: R1 identical geometry under a NEW shape_id (schedule-only
    # change), R2 rerouted through a detour, R3 gone, R4 added, R6 unchanged.
    _write_feed(
        after_dir,
        routes=[
            {"route_id": "R1", "route_short_name": "1", "route_long_name": "First"},
            {"route_id": "R2", "route_short_name": "2", "route_long_name": "Second"},
            {"route_id": "R4", "route_short_name": "4", "route_long_name": "Fourth"},
            {"route_id": "R6", "route_short_name": "6", "route_long_name": "Sixth"},
        ],
        trips=[
            {"route_id": "R1", "trip_id": "T1a", "shape_id": "S1_NEWID"},
            {"route_id": "R2", "trip_id": "T2a", "shape_id": "S2A"},
            {"route_id": "R4", "trip_id": "T4", "shape_id": "S4"},
            {"route_id": "R6", "trip_id": "T6a", "shape_id": ""},
        ],
        stop_times=(
            _stop_times_rows("T1a", ["101", "102"])
            + _stop_times_rows("T2a", ["201", "202"])
            + _stop_times_rows("T4", ["401", "402"])
            + _stop_times_rows("T6a", ["601", "602", "603"])
        ),
        stops=STOPS,
        shapes=(
            _shape_rows("S1_NEWID", [(-77.050, 38.900), (-77.030, 38.900)])
            + _shape_rows(
                "S2A",
                [
                    (-77.050, 38.920),
                    (-77.045, 38.920),
                    (-77.045, 38.930),
                    (-77.035, 38.930),
                    (-77.035, 38.920),
                    (-77.030, 38.920),
                ],
            )
            + _shape_rows("S4", [(-77.050, 38.940), (-77.030, 38.940)])
        ),
    )

    cfg = Config(before_dir=before_dir, after_dir=after_dir, output_dir=out_dir)
    summary = run(cfg)
    return summary, out_dir


def test_run_summary_counts(diff_run) -> None:
    summary, _ = diff_run
    assert summary.matched_count == 3  # R1, R2, R6
    assert summary.added_count == 1  # R4
    assert summary.eliminated_count == 1  # R3
    assert summary.realigned_count == 1  # R2
    assert summary.unchanged_alignment_count == 2  # R1 (schedule-only), R6
    assert summary.no_geometry_count == 0
    assert summary.chord_fallback_route_count == 2  # R6 in each feed


def test_run_route_log_classifications(diff_run) -> None:
    _, out_dir = diff_run
    table = pd.read_csv(out_dir / "routes_linear_changes.csv", dtype={"route_label": str})
    by_label = table.set_index("route_label")

    assert by_label.loc["2", "change_kind"] == "realigned"
    assert bool(by_label.loc["2", "has_linear_change"])
    assert by_label.loc["4", "change_kind"] == "added"
    assert by_label.loc["3", "change_kind"] == "eliminated"
    # Schedule-only routes are explicitly excluded from the change log.
    assert by_label.loc["1", "change_kind"] == "unchanged_alignment"
    assert not bool(by_label.loc["1", "has_linear_change"])
    assert "schedule-only" in str(by_label.loc["1", "note"])
    assert by_label.loc["6", "change_kind"] == "unchanged_alignment"


def test_run_identical_geometry_survives_shape_id_rekey(diff_run) -> None:
    _, out_dir = diff_run
    table = pd.read_csv(
        out_dir / "routes_linear_changes.csv", dtype={"route_label": str}
    ).set_index("route_label")
    # R1's shape_id changed (S1 -> S1_NEWID) but geometry did not: no change.
    assert table.loc["1", "new_len_ft"] == 0.0
    assert table.loc["1", "eliminated_len_ft"] == 0.0
    assert table.loc["1", "retained_len_ft"] > 0.0


def test_run_writes_segment_and_network_shapefiles(diff_run) -> None:
    import geopandas as gpd

    _, out_dir = diff_run
    for name in (
        "linear_segments_new.shp",
        "linear_segments_retained.shp",
        "linear_segments_eliminated.shp",
        "linear_network_new.shp",
        "linear_network_retained.shp",
        "linear_network_eliminated.shp",
    ):
        assert (out_dir / name).exists(), name

    new = gpd.read_file(out_dir / "linear_segments_new.shp")
    assert set(new["route_id"]) == {"R2", "R4"}
    assert str(new.crs).upper() == "EPSG:4326"
    eliminated = gpd.read_file(out_dir / "linear_segments_eliminated.shp")
    assert set(eliminated["route_id"]) == {"R2", "R3"}
    retained = gpd.read_file(out_dir / "linear_segments_retained.shp")
    assert {"R1", "R2", "R6"} <= set(retained["route_id"])
    assert set(retained.loc[retained["route_id"] == "R6", "geom_src"]) == {"stops_chord"}


def test_run_writes_summary_json_and_run_log(diff_run) -> None:
    _, out_dir = diff_run
    summary = json.loads((out_dir / "linear_diff_summary.json").read_text(encoding="utf-8"))
    assert summary["realigned_count"] == 1
    runlog = (out_dir / "gtfs_linear_diff_runlog.txt").read_text(encoding="utf-8")
    assert "GTFS LINEAR DIFF RUN LOG" in runlog
    assert "BUFFER_TOLERANCE_FEET" in runlog


def test_run_r2_lengths_are_plausible(diff_run) -> None:
    _, out_dir = diff_run
    table = pd.read_csv(
        out_dir / "routes_linear_changes.csv", dtype={"route_label": str}
    ).set_index("route_label")
    # Detour adds ~0.01 deg up + 0.01 deg across + 0.01 deg down; skipped middle
    # is ~0.01 deg of the old alignment (~2840 ft).
    assert table.loc["2", "new_len_ft"] == pytest.approx(10150, rel=0.10)
    assert table.loc["2", "eliminated_len_ft"] == pytest.approx(2840, rel=0.15)
    assert table.loc["2", "changed_share"] > 0.5


# ---------------------------------------------------------------------------
# load_feed / build_route_geometries edge cases
# ---------------------------------------------------------------------------


def test_load_feed_missing_required_file_raises(tmp_path: Path) -> None:
    feed_dir = tmp_path / "feed"
    feed_dir.mkdir()
    (feed_dir / "routes.txt").write_text("route_id\nR1\n", encoding="utf-8")
    with pytest.raises(OSError, match="missing required GTFS files"):
        load_feed(feed_dir, "test")


def test_load_feed_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="does not exist"):
        load_feed(tmp_path / "nope", "test")


def test_build_route_geometries_chord_fallback_flagged(tmp_path: Path) -> None:
    feed_dir = tmp_path / "feed"
    _write_feed(
        feed_dir,
        routes=[{"route_id": "R6", "route_short_name": "6", "route_long_name": ""}],
        trips=[{"route_id": "R6", "trip_id": "T6", "shape_id": ""}],
        stop_times=_stop_times_rows("T6", ["601", "602", "603"]),
        stops=STOPS,
        shapes=[],
    )
    feed = load_feed(feed_dir, "test")
    geoms = build_route_geometries(feed, "test", allow_chord_fallback=True)
    assert geoms["R6"].source == "stops_chord"
    assert geoms["R6"].geometry is not None


def test_build_route_geometries_no_fallback_leaves_route_without_geometry(
    tmp_path: Path,
) -> None:
    feed_dir = tmp_path / "feed"
    _write_feed(
        feed_dir,
        routes=[{"route_id": "R6", "route_short_name": "6", "route_long_name": ""}],
        trips=[{"route_id": "R6", "trip_id": "T6", "shape_id": ""}],
        stop_times=_stop_times_rows("T6", ["601", "602", "603"]),
        stops=STOPS,
        shapes=[],
    )
    feed = load_feed(feed_dir, "test")
    geoms = build_route_geometries(feed, "test", allow_chord_fallback=False)
    assert geoms["R6"].source == "none"
    assert geoms["R6"].geometry is None


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------


def test_main_returns_2_for_placeholder_paths() -> None:
    assert main([]) == 2


def test_main_returns_1_for_missing_feed(tmp_path: Path) -> None:
    code = main(
        [
            "--before",
            str(tmp_path / "missing_before"),
            "--after",
            str(tmp_path / "missing_after"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
