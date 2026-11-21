from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point

import scripts.gtfs_data_quality.gtfs_skipped_stop_flagger_gpd as target

# ---------------------------------------------------------------------------
# normalize_direction_id
# ---------------------------------------------------------------------------


def test_normalize_direction_id_ints_become_strings() -> None:
    s = pd.Series([0, 1, 0])
    assert list(target.normalize_direction_id(s)) == ["0", "1", "0"]


def test_normalize_direction_id_preserves_na_token() -> None:
    s = pd.Series([0, None])
    out = target.normalize_direction_id(s)
    assert out.iloc[0] == "0"
    assert out.iloc[1] == "<NA>"


def test_normalize_direction_id_parses_text_values() -> None:
    s = pd.Series(["0", "1", ""])
    assert list(target.normalize_direction_id(s)) == ["0", "1", "<NA>"]


def test_normalize_direction_id_warns_on_non_numeric_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        out = target.normalize_direction_id(pd.Series(["0", "N", "S", ""]))
    assert list(out) == ["0", "<NA>", "<NA>", "<NA>"]
    assert "2 direction_id value(s) are not numbers (e.g., N, S)" in caplog.text


# ---------------------------------------------------------------------------
# load_gtfs_tables
# ---------------------------------------------------------------------------


def _write_text_feed(gtfs_dir: Path, trips_txt: str) -> None:
    (gtfs_dir / "stops.txt").write_text(
        "stop_id,stop_code,stop_name,stop_lat,stop_lon\n"
        "001,0101,First,38.90,-77.03\n"
        "002,,Second,38.91,-77.03\n"
    )
    # "X2" keeps this column text under type inference while stops.txt's
    # stop_id column would be inferred as integers.
    (gtfs_dir / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\nT1,X2,10\nT1,001,9\n")
    (gtfs_dir / "trips.txt").write_text(trips_txt)
    (gtfs_dir / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        "10,38.90,-77.03,1\n"
        "10,38.91,-77.03,2\n"
    )
    (gtfs_dir / "routes.txt").write_text("route_id,route_short_name\nR1,NA\n")


def test_load_gtfs_tables_keeps_identifiers_as_text(tmp_path: Path) -> None:
    _write_text_feed(
        tmp_path,
        "route_id,service_id,trip_id,direction_id,shape_id\nR1,WK,T1,0,10\nR1,WK,T2,1,\n",
    )
    tables = target.load_gtfs_tables(tmp_path)
    assert list(tables["stops"]["stop_id"]) == ["001", "002"]
    assert list(tables["stops"]["stop_code"]) == ["0101", ""]
    assert set(tables["stop_times"]["stop_id"]) == {"001", "X2"}
    # A blank shape_id no longer turns "10" into "10.0" in trips.txt only.
    assert list(tables["trips"]["shape_id"]) == ["10", ""]
    assert list(tables["shapes"]["shape_id"]) == ["10", "10"]
    assert list(tables["routes"]["route_short_name"]) == ["NA"]
    assert list(tables["trips"]["direction_id"]) == ["0", "1"]


def test_load_gtfs_tables_parses_coordinates_and_sequences(tmp_path: Path) -> None:
    _write_text_feed(
        tmp_path,
        "route_id,service_id,trip_id,direction_id,shape_id\nR1,WK,T1,0,10\n",
    )
    tables = target.load_gtfs_tables(tmp_path)
    assert tables["stops"]["stop_lat"].tolist() == [38.90, 38.91]
    assert tables["shapes"]["shape_pt_sequence"].tolist() == [1, 2]
    # Numeric order, not text order ("10" < "9").
    ordered = tables["stop_times"].sort_values("stop_sequence")
    assert ordered["stop_id"].tolist() == ["001", "X2"]


def test_load_gtfs_tables_accepts_feed_without_direction_id(tmp_path: Path) -> None:
    _write_text_feed(tmp_path, "route_id,service_id,trip_id,shape_id\nR1,WK,T1,10\n")
    tables = target.load_gtfs_tables(tmp_path)
    assert list(tables["trips"]["direction_id"]) == ["<NA>"]


def test_load_gtfs_tables_reads_zipped_feed_like_folder(tmp_path: Path) -> None:
    feed = tmp_path / "feed"
    feed.mkdir()
    _write_text_feed(
        feed, "route_id,service_id,trip_id,direction_id,shape_id\nR1,WK,T1,0,10\nR1,WK,T2,1,\n"
    )
    zip_path = tmp_path / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in feed.iterdir():
            zf.write(path, f"wrapper/{path.name}")

    from_folder = target.load_gtfs_tables(feed)
    from_zip = target.load_gtfs_tables(zip_path)
    assert from_zip.keys() == from_folder.keys()
    for key, frame in from_folder.items():
        pd.testing.assert_frame_equal(from_zip[key], frame)


def test_sanitize_token_makes_labels_filename_safe() -> None:
    assert target._sanitize_token("<NA>") == "NA"
    assert target._sanitize_token("stop_id=12/B") == "stop_id_12_B"
    assert target._sanitize_token("Blue Line") == "Blue_Line"
    assert target._sanitize_token("") == "unnamed"


# ---------------------------------------------------------------------------
# choose_representative_trip_ids_max_stops
# ---------------------------------------------------------------------------


def test_choose_representative_trip_picks_trip_with_most_stops() -> None:
    trips = pd.DataFrame(
        {
            "route_id": ["R1", "R1"],
            "direction_id": ["0", "0"],
            "trip_id": ["T_short", "T_long"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T_short", "T_short", "T_long", "T_long", "T_long"],
            "stop_id": ["S1", "S2", "S1", "S2", "S3"],
        }
    )
    reps = target.choose_representative_trip_ids_max_stops(trips, stop_times)
    assert reps[("R1", "0")] == "T_long"


def test_choose_representative_trip_per_direction() -> None:
    trips = pd.DataFrame(
        {
            "route_id": ["R1", "R1"],
            "direction_id": ["0", "1"],
            "trip_id": ["T0", "T1"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T0", "T0", "T1", "T1"],
            "stop_id": ["S1", "S2", "S2", "S1"],
        }
    )
    reps = target.choose_representative_trip_ids_max_stops(trips, stop_times)
    assert reps == {("R1", "0"): "T0", ("R1", "1"): "T1"}


# ---------------------------------------------------------------------------
# build_stop_key_lookup / build_stop_names_lookup
# ---------------------------------------------------------------------------


def _stops_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_code": ["C1", "C2"],
            "stop_name": ["Main & 1st", "Main & 2nd"],
        }
    )


def test_build_stop_key_lookup_maps_stop_id_to_code() -> None:
    lookup = target.build_stop_key_lookup(_stops_df(), "stop_code")
    assert lookup == {"S1": "C1", "S2": "C2"}


def test_build_stop_key_lookup_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="stop_code"):
        target.build_stop_key_lookup(_stops_df().drop(columns=["stop_code"]), "stop_code")


def test_build_stop_names_lookup_keyed_by_stop_key() -> None:
    lookup = target.build_stop_names_lookup(_stops_df(), "stop_code")
    assert lookup["C1"] == "Main & 1st"


def test_build_stop_names_lookup_missing_name_column_raises() -> None:
    with pytest.raises(ValueError, match="stop_name"):
        target.build_stop_names_lookup(_stops_df().drop(columns=["stop_name"]), "stop_code")


def test_build_stop_key_lookup_stop_id_mode_maps_ids_to_themselves() -> None:
    lookup = target.build_stop_key_lookup(_stops_df(), "stop_id")
    assert lookup == {"S1": "S1", "S2": "S2"}


def test_build_stop_key_lookup_blank_codes_fall_back_to_stop_id() -> None:
    stops = pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3", "S4"],
            # S3's stop_id equals S4's real code: the fallback must not collide.
            "stop_code": ["100", "", " ", "S3"],
            "stop_name": ["a", "b", "c", "d"],
        }
    )
    lookup = target.build_stop_key_lookup(stops, "stop_code")
    assert lookup == {"S1": "100", "S2": "stop_id=S2", "S3": "stop_id=S3", "S4": "S3"}
    names = target.build_stop_names_lookup(stops, "stop_code")
    assert names["stop_id=S2"] == "b"


def _shared_code_stops() -> pd.DataFrame:
    """Two platforms sharing stop_code "1" (~4 m apart) and one stop "2"."""
    return pd.DataFrame(
        {
            "stop_id": ["S1a", "S1b", "S2"],
            "stop_code": ["1", "1", "2"],
            "stop_name": ["One NB", "One SB", "Two"],
            "stop_lat": [38.90000, 38.90004, 38.90000],
            "stop_lon": [-77.03000, -77.03000, -77.02770],
        }
    )


def test_build_stops_gdf_places_shared_code_at_mean(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        gdf = target.build_stops_gdf(_shared_code_stops(), "EPSG:4326", "stop_code")
    assert gdf.index.is_unique
    assert list(gdf.index) == ["1", "2"]
    point = gdf.loc["1", "geometry"]
    assert (point.x, point.y) == pytest.approx((-77.03, 38.90002))
    assert "shared by more than one stop_id" in caplog.text


def test_segment_hausdorff_distance_handles_shared_stop_code() -> None:
    gdf = target.build_stops_gdf(_shared_code_stops(), "EPSG:4326", "stop_code")
    gdf_proj = gdf.to_crs("EPSG:26918")
    line = LineString([gdf_proj.loc["1", "geometry"], gdf_proj.loc["2", "geometry"]])
    m0, m1 = target.locate_stops_along_shape(["1", "2"], line, gdf_proj, tolerance=30.0)
    assert m0 is not None and m1 is not None
    distance = target.segment_hausdorff_distance(line, (m0, m1), line, (m0, m1), padding=50.0)
    assert distance == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# locate_stops_along_shape
# ---------------------------------------------------------------------------


def test_locate_stops_along_shape_places_loop_ends_at_both_ends() -> None:
    # A square loop that ends where it starts.
    shape = LineString([(0, 0), (300, 0), (300, 300), (0, 300), (0, 0)])
    stops = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(300, 0), Point(300, 300)],
        index=pd.Index(["A", "B", "C"]),
        crs=target.PROJECTED_CRS,
    )
    measures = target.locate_stops_along_shape(["A", "B", "C", "A"], shape, stops, 30.0)
    assert measures == pytest.approx([0.0, 300.0, 600.0, 1200.0])


def test_locate_stops_along_shape_keeps_visits_on_their_own_pass() -> None:
    # Out along y=0 and back along y=8, with stops on each direction's curb.
    # R and S (return curb) lie 12 m from the outbound pass but belong on the
    # return pass. P is drawn 3 m from the return pass and 5 m from the
    # outbound one, but its neighbours place it outbound.
    shape = LineString([(0, 0), (400, 0), (400, 8), (0, 8)])
    coords = {
        "P1": (50, -4),
        "P": (100, 5),
        "P2": (150, -4),
        "Q": (300, -4),
        "R": (300, 12),
        "S": (100, 12),
    }
    stops = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in coords.values()],
        index=pd.Index(list(coords)),
        crs=target.PROJECTED_CRS,
    )
    seq = ["P1", "P", "P2", "Q", "MISSING", "R", "S"]
    measures = target.locate_stops_along_shape(seq, shape, stops, 30.0)
    assert measures[4] is None
    located = [m for m in measures if m is not None]
    assert located == pytest.approx([50.0, 100.0, 150.0, 300.0, 508.0, 708.0])


# ---------------------------------------------------------------------------
# build_route_sequences
# ---------------------------------------------------------------------------


def test_build_route_sequences_orders_by_stop_sequence_and_dedups() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1", "T1"],
            "stop_id": ["S2", "S1", "S1", "S3"],
            "stop_sequence": [2, 1, 3, 4],
        }
    )
    lookup = {"S1": "C1", "S2": "C2", "S3": "C3"}
    seqs = target.build_route_sequences(stop_times, lookup, {("R1", "0"): "T1"})
    # Sorted: S1(1), S2(2), S1(3), S3(4); consecutive duplicate keys collapse.
    assert seqs[("R1", "0")] == ["C1", "C2", "C1", "C3"]


def test_build_route_sequences_drops_single_stop_trips() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "stop_id": ["S1"],
            "stop_sequence": [1],
        }
    )
    seqs = target.build_route_sequences(stop_times, {"S1": "C1"}, {("R1", "0"): "T1"})
    assert seqs == {}


# ---------------------------------------------------------------------------
# find_aligned_common_stops
# ---------------------------------------------------------------------------


def test_find_aligned_common_stops_in_order() -> None:
    base = ["A", "B", "C", "D"]
    other = ["A", "X", "C", "D"]
    assert target.find_aligned_common_stops(base, other) == [(0, 0), (2, 2), (3, 3)]


def test_find_aligned_common_stops_enforces_direction() -> None:
    base = ["A", "B", "C"]
    other = ["C", "B", "A"]  # reversed: only the first match survives
    assert target.find_aligned_common_stops(base, other) == [(0, 2)]


def test_find_aligned_common_stops_no_overlap() -> None:
    assert target.find_aligned_common_stops(["A"], ["B"]) == []


def test_find_aligned_common_stops_uses_later_occurrences() -> None:
    # The reference visits B and C before looping back through A; only their
    # later occurrences keep all three stops in order.
    base = ["A", "B", "C"]
    other = ["B", "C", "A", "B", "X", "C"]
    assert target.find_aligned_common_stops(base, other) == [(0, 2), (1, 3), (2, 5)]


def test_find_aligned_common_stops_keeps_most_stops_in_order() -> None:
    # Pairing B with the reference's last stop would strand C and D.
    base = ["A", "B", "C", "D"]
    other = ["A", "C", "D", "B"]
    assert target.find_aligned_common_stops(base, other) == [(0, 0), (2, 1), (3, 2)]


# ---------------------------------------------------------------------------
# compare_segments_for_route_pair (coordinates in projected metres)
# ---------------------------------------------------------------------------


def _stops_proj(coords: dict[str, tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in coords.values()],
        index=pd.Index(list(coords), name="stop_code"),
        crs=target.PROJECTED_CRS,
    )


def _compare_pair(
    base_seq: list[str],
    ref_seq: list[str],
    coords: dict[str, tuple[float, float]],
    base_shape: LineString,
    ref_shape: LineString | None = None,
) -> list[tuple[str, str, str]]:
    """Compare two routes with the default thresholds.

    Returns (segment start, segment end, candidate keys) per finding.
    """
    base, ref = ("BASE", "0"), ("REF", "0")
    results = target.compare_segments_for_route_pair(
        base_key=base,
        other_key=ref,
        sequences={base: base_seq, ref: ref_seq},
        stop_names={},
        shapes_proj={base: base_shape, ref: ref_shape or base_shape},
        stops_gdf_proj=_stops_proj(coords),
        max_shape_hausdorff_m=target.MAX_SHAPE_HAUSDORFF_M,
        max_stop_to_shape_m=target.MAX_STOP_TO_SHAPE_M,
        segment_measure_padding_m=target.SEGMENT_MEASURE_PADDING_M,
    )
    return [
        (
            str(r["segment_start_stop_key"]),
            str(r["segment_end_stop_key"]),
            str(r["candidate_missing_stop_keys"]),
        )
        for r in results
    ]


def test_compare_segments_flags_single_skipped_stop() -> None:
    # The base runs A -> C directly; the reference serves B in between.
    coords = {"P": (0, 0), "A": (100, 0), "B": (200, 0), "C": (300, 0), "Q": (400, 0)}
    line = LineString([(0, 0), (400, 0)])
    flags = _compare_pair(["P", "A", "C", "Q"], ["P", "A", "B", "C", "Q"], coords, line)
    assert flags == [("A", "C", "B")]


@pytest.mark.parametrize(("offset_m", "expected"), [(20, [("A", "C", "B")]), (60, [])])
def test_compare_segments_tests_proximity_against_the_segment(
    offset_m: float, expected: list[tuple[str, str, str]]
) -> None:
    # The base doubles back 60 m north of its A-C stretch. At a 60 m offset,
    # B sits on that later stretch: 0 m from the base shape as a whole, but
    # 60 m from the part between A and C.
    coords = {"A": (0, 0), "E": (150, 0), "B": (100, offset_m), "C": (200, 0), "D": (0, 60)}
    base_shape = LineString([(0, 0), (400, 0), (400, 60), (0, 60)])
    ref_shape = LineString([(0, 0), (100, offset_m), (200, 0), (400, 0), (400, 60), (0, 60)])
    flags = _compare_pair(["A", "E", "C", "D"], ["A", "B", "C", "D"], coords, base_shape, ref_shape)
    assert flags == expected


@pytest.mark.parametrize("end_gap_m", [0, 1], ids=["closed-loop", "endpoint-1m-off"])
def test_compare_segments_checks_the_closing_leg_of_a_loop(end_gap_m: float) -> None:
    # Both routes run the same square loop; X lies on the closing leg C -> A.
    # The flag must not depend on whether the shape ends exactly at its start.
    coords = {"A": (0, 0), "B": (300, 0), "C": (300, 300), "X": (0, 150), "Z": (0, end_gap_m)}
    end = "Z" if end_gap_m else "A"
    loop = LineString([(0, 0), (300, 0), (300, 300), (0, 300), (0, end_gap_m)])
    flags = _compare_pair(["A", "B", "C", end], ["A", "B", "C", "X", end], coords, loop)
    assert flags == [("C", end, "X")]


def test_compare_segments_ignores_reference_revisiting_a_boundary_stop() -> None:
    # The reference runs out to L and back through X before continuing to B.
    coords = {"A": (0, 0), "X": (100, 0), "L": (150, 10), "B": (300, 0)}
    base_shape = LineString([(0, 0), (300, 0)])
    ref_shape = LineString([(0, 0), (100, 0), (150, 10), (100, 0), (300, 0)])
    flags = _compare_pair(["A", "X", "B"], ["A", "X", "L", "X", "B"], coords, base_shape, ref_shape)
    assert flags == [("X", "B", "L")]


# ---------------------------------------------------------------------------
# shares_only_terminal_stops
# ---------------------------------------------------------------------------


def test_shares_only_terminal_stops_true_for_terminal_only_overlap() -> None:
    base = ["A", "B", "C"]
    other = ["A", "X", "Y"]
    assert target.shares_only_terminal_stops(base, other) is True


def test_shares_only_terminal_stops_false_when_interior_shared() -> None:
    base = ["A", "B", "C"]
    other = ["X", "B", "Y"]
    assert target.shares_only_terminal_stops(base, other) is False


def test_shares_only_terminal_stops_false_without_overlap() -> None:
    assert target.shares_only_terminal_stops(["A", "B"], ["X", "Y"]) is False


# ---------------------------------------------------------------------------
# sequences_are_reversed
# ---------------------------------------------------------------------------


def test_sequences_are_reversed_detects_opposite_direction() -> None:
    base = ["A", "B", "C", "D"]
    assert target.sequences_are_reversed(base, list(reversed(base))) is True


def test_sequences_are_reversed_false_for_same_direction() -> None:
    base = ["A", "B", "C", "D"]
    assert target.sequences_are_reversed(base, base) is False


def test_sequences_are_reversed_false_with_fewer_than_two_shared() -> None:
    assert target.sequences_are_reversed(["A", "B"], ["B", "X"]) is False


def test_sequences_are_reversed_counts_every_visit_of_a_loop() -> None:
    # The loop's return to A must not outvote A, B, C served in the same order.
    loop = ["A", "B", "C", "A"]
    assert target.sequences_are_reversed(loop, ["A", "B", "X", "C"]) is False
    assert target.sequences_are_reversed(loop, ["C", "X", "B", "A"]) is True
    assert target.sequences_are_reversed(loop, ["A", "C", "B", "A"]) is True


# ---------------------------------------------------------------------------
# unique_preserve_order / _parse_semicolon_list
# ---------------------------------------------------------------------------


def test_unique_preserve_order() -> None:
    assert target.unique_preserve_order(["B", "A", "B", "C", "A"]) == ["B", "A", "C"]


def test_parse_semicolon_list_splits_and_drops_empties() -> None:
    assert target._parse_semicolon_list("A;B;;C") == ["A", "B", "C"]


def test_parse_semicolon_list_non_string_returns_empty() -> None:
    assert target._parse_semicolon_list(None) == []
    assert target._parse_semicolon_list("") == []


# ---------------------------------------------------------------------------
# aggregate_candidates
# ---------------------------------------------------------------------------


def test_aggregate_candidates_counts_distinct_references() -> None:
    df = pd.DataFrame(
        {
            "missing_route_id": ["R1", "R1"],
            "missing_route_direction_id": ["0", "0"],
            "reference_route_id": ["R2", "R3"],
            "segment_start_stop_key": ["A", "A"],
            "candidate_missing_stop_keys": ["X;Y", "X"],
        }
    )
    agg = target.aggregate_candidates(df)
    x_row = agg[agg["stop_key"] == "X"].iloc[0]
    assert x_row["n_reference_routes"] == 2
    assert x_row["reference_route_ids"] == "R2;R3"
    y_row = agg[agg["stop_key"] == "Y"].iloc[0]
    assert y_row["n_reference_routes"] == 1


def test_aggregate_candidates_empty_input_passthrough() -> None:
    df = pd.DataFrame()
    assert target.aggregate_candidates(df).empty


def test_empty_results_keep_their_columns() -> None:
    assert list(target.aggregate_candidates(pd.DataFrame()).columns) == target.AGGREGATE_COLUMNS
    trips = pd.DataFrame({"trip_id": ["T1"], "route_id": ["R1"], "direction_id": ["0"]})
    stop_times = pd.DataFrame(
        {"trip_id": ["T1", "T1"], "stop_id": ["S1", "S2"], "stop_sequence": [1, 2]}
    )
    intra = target.find_intra_route_skipped_stops(trips, stop_times, {"S1": "S1", "S2": "S2"})
    assert intra.empty
    assert list(intra.columns) == target.INTRA_ROUTE_COLUMNS


# ---------------------------------------------------------------------------
# find_intra_route_skipped_stops
# ---------------------------------------------------------------------------


def _intra_route_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three trips on R1 dir 0: two canonical (A,B,C,D) and one skipping C."""
    trips = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1", "R1", "R1"],
            "direction_id": ["0", "0", "0"],
        }
    )
    rows = []
    for tid, seq in [
        ("T1", ["S1", "S2", "S3", "S4"]),
        ("T2", ["S1", "S2", "S3", "S4"]),
        ("T3", ["S1", "S2", "S4"]),
    ]:
        rows += [{"trip_id": tid, "stop_id": s, "stop_sequence": i} for i, s in enumerate(seq)]
    return trips, pd.DataFrame(rows)


def test_find_intra_route_skipped_stops_flags_subset_trip() -> None:
    trips, stop_times = _intra_route_frames()
    lookup = {f"S{i}": f"C{i}" for i in range(1, 5)}
    out = target.find_intra_route_skipped_stops(trips, stop_times, lookup)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["trip_id"] == "T3"
    assert row["missing_stop_keys"] == "C3"
    assert row["n_canonical_trips"] == 2


def test_find_intra_route_skipped_stops_ignores_short_turns() -> None:
    trips = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1"] * 3,
            "direction_id": ["0"] * 3,
        }
    )
    rows = []
    for tid, seq in [
        ("T1", ["S1", "S2", "S3", "S4"]),
        ("T2", ["S1", "S2", "S3", "S4"]),
        ("T3", ["S1", "S2", "S3"]),  # ends early: genuine short-turn
    ]:
        rows += [{"trip_id": tid, "stop_id": s, "stop_sequence": i} for i, s in enumerate(seq)]
    lookup = {f"S{i}": f"C{i}" for i in range(1, 5)}
    out = target.find_intra_route_skipped_stops(trips, pd.DataFrame(rows), lookup)
    assert out.empty


@pytest.mark.parametrize(
    "variant",
    [["S1", "SX", "S3", "S4", "S5"], ["S1", "S4", "S3", "S5"]],
    ids=["adds-stop", "reorders-stops"],
)
def test_find_intra_route_skipped_stops_ignores_variants_that_are_not_omissions(
    variant: list[str],
) -> None:
    trips = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1"] * 3,
            "direction_id": ["0"] * 3,
        }
    )
    rows = []
    for tid, seq in [
        ("T1", ["S1", "S2", "S3", "S4", "S5"]),
        ("T2", ["S1", "S2", "S3", "S4", "S5"]),
        ("T3", variant),  # same endpoints, but not the canonical minus some stops
    ]:
        rows += [{"trip_id": tid, "stop_id": s, "stop_sequence": i} for i, s in enumerate(seq)]
    lookup = {s: s for s in ["S1", "S2", "S3", "S4", "S5", "SX"]}
    out = target.find_intra_route_skipped_stops(trips, pd.DataFrame(rows), lookup)
    assert out.empty


def test_find_intra_route_skipped_stops_flags_skipped_repeat_visit() -> None:
    # The canonical loop serves B twice; T3 skips the second visit only, so
    # both trips serve the same set of stops.
    trips = pd.DataFrame(
        {"trip_id": ["T1", "T2", "T3"], "route_id": ["R1"] * 3, "direction_id": ["0"] * 3}
    )
    rows = []
    for tid, seq in [
        ("T1", ["A", "B", "X", "B", "C"]),
        ("T2", ["A", "B", "X", "B", "C"]),
        ("T3", ["A", "B", "X", "C"]),
    ]:
        rows += [{"trip_id": tid, "stop_id": s, "stop_sequence": i} for i, s in enumerate(seq)]
    lookup = {s: s for s in ["A", "B", "C", "X"]}
    out = target.find_intra_route_skipped_stops(trips, pd.DataFrame(rows), lookup)
    assert out[["trip_id", "missing_stop_keys"]].to_numpy().tolist() == [["T3", "B"]]


def test_find_intra_route_skipped_stops_lists_missing_stops_in_travel_order() -> None:
    trips = pd.DataFrame(
        {"trip_id": ["T1", "T2", "T3"], "route_id": ["R1"] * 3, "direction_id": ["0"] * 3}
    )
    rows = []
    for tid, seq in [
        ("T1", ["A", "Z", "M", "B"]),
        ("T2", ["A", "Z", "M", "B"]),
        ("T3", ["A", "B"]),
    ]:
        rows += [{"trip_id": tid, "stop_id": s, "stop_sequence": i} for i, s in enumerate(seq)]
    lookup = {s: s for s in ["A", "B", "M", "Z"]}
    out = target.find_intra_route_skipped_stops(trips, pd.DataFrame(rows), lookup)
    assert out.iloc[0]["missing_stop_keys"] == "Z;M"


def test_skipped_positions() -> None:
    assert target.skipped_positions(["A", "C"], ["A", "B", "C"]) == [1]
    assert target.skipped_positions(["A", "B", "X", "C"], ["A", "B", "X", "B", "C"]) == [3]
    assert target.skipped_positions(["A", "C", "B", "D"], ["A", "B", "C", "B", "D"]) == [1]
    assert target.skipped_positions(["A", "B"], ["A", "B"]) == []
    assert target.skipped_positions(["C", "A"], ["A", "B", "C"]) is None


def test_is_ordered_subsequence() -> None:
    assert target.is_ordered_subsequence(["A", "C"], ["A", "B", "C"]) is True
    assert target.is_ordered_subsequence(["A", "B", "B"], ["A", "B", "C", "B"]) is True
    assert target.is_ordered_subsequence(["C", "A"], ["A", "B", "C"]) is False
    assert target.is_ordered_subsequence(["A", "X"], ["A", "B", "C"]) is False


# ---------------------------------------------------------------------------
# hausdorff_distance_safe / _find_segment_indices
# ---------------------------------------------------------------------------


def test_hausdorff_distance_safe_none_inputs() -> None:
    line = LineString([(0, 0), (1, 0)])
    assert target.hausdorff_distance_safe(None, line) is None
    assert target.hausdorff_distance_safe(line, None) is None
    assert target.hausdorff_distance_safe(line, line) == 0.0


def test_find_segment_indices_first_occurrences() -> None:
    seq = ["A", "B", "C", "B"]
    assert target._find_segment_indices(seq, "A", "B") == (0, 1)


def test_find_segment_indices_missing_stop_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        target._find_segment_indices(["A", "B"], "Z", "B")
    with pytest.raises(KeyError):
        target._find_segment_indices(["A", "B"], "B", "A")


def test_plot_mismatch_segment_plots_reversed_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reference runs D -> A; the comparison reversed it, so the finding's
    # A -> C segment only exists in the reversed order.
    monkeypatch.setattr(target, "PLOT_DIR", tmp_path)
    base, ref = ("R1", "0"), ("R2", "1")
    sequences = {base: ["A", "C", "D"], ref: ["D", "C", "B", "A"]}
    shapes_geo = {
        base: LineString([(-77.030, 38.9), (-77.027, 38.9)]),
        ref: LineString([(-77.027, 38.9), (-77.030, 38.9)]),
    }
    stops_geo = gpd.GeoDataFrame(
        geometry=[Point(-77.030 + 0.001 * i, 38.9) for i in range(4)],
        index=pd.Index(["A", "B", "C", "D"]),
        crs=target.GTFS_CRS,
    )
    row = pd.Series(
        {
            "missing_route_id": "R1",
            "missing_route_direction_id": "0",
            "reference_route_id": "R2",
            "reference_route_direction_id": "1",
            "segment_start_stop_key": "A",
            "segment_end_stop_key": "C",
            "candidate_missing_stop_keys": "B",
        }
    )
    out_path = target.plot_mismatch_segment(row, sequences, shapes_geo, stops_geo, {})
    assert out_path is not None
    assert out_path.exists()


# ---------------------------------------------------------------------------
# select_representative_shapes / prepare_gtfs_context (synthetic feed)
# ---------------------------------------------------------------------------

_TO_LONLAT = Transformer.from_crs(target.PROJECTED_CRS, target.GTFS_CRS, always_xy=True)
_ORIGIN_X, _ORIGIN_Y = 323_000.0, 4_307_000.0  # Washington, DC in UTM 18N


def _lonlat(x: float, y: float) -> tuple[float, float]:
    return _TO_LONLAT.transform(_ORIGIN_X + x, _ORIGIN_Y + y)


def _write_feed(
    gtfs_dir: Path,
    stops: dict[str, tuple[str, float, float]],
    trips: list[tuple[str, str, str, str, list[str]]],
    shapes: dict[str, list[tuple[float, float]]],
) -> None:
    """Write a feed from metre offsets.

    stops: stop_id -> (stop_code, x, y); trips: (route_id, direction_id,
    trip_id, shape_id, stop_ids); shapes: shape_id -> [(x, y), ...].
    """
    stop_rows = []
    for stop_id, (code, x, y) in stops.items():
        lon, lat = _lonlat(x, y)
        stop_rows.append(
            {
                "stop_id": stop_id,
                "stop_code": code,
                "stop_name": f"Stop {stop_id}",
                "stop_lat": lat,
                "stop_lon": lon,
            }
        )
    trip_rows = []
    stop_time_rows = []
    for route_id, direction_id, trip_id, shape_id, stop_ids in trips:
        trip_rows.append(
            {
                "route_id": route_id,
                "service_id": "WK",
                "trip_id": trip_id,
                "direction_id": direction_id,
                "shape_id": shape_id,
            }
        )
        stop_time_rows += [
            {"trip_id": trip_id, "stop_id": stop_id, "stop_sequence": seq}
            for seq, stop_id in enumerate(stop_ids, start=1)
        ]
    shape_rows = []
    for shape_id, points in shapes.items():
        for seq, (x, y) in enumerate(points, start=1):
            lon, lat = _lonlat(x, y)
            shape_rows.append(
                {
                    "shape_id": shape_id,
                    "shape_pt_lat": lat,
                    "shape_pt_lon": lon,
                    "shape_pt_sequence": seq,
                }
            )
    route_ids = sorted({trip[0] for trip in trips})
    pd.DataFrame(stop_rows).to_csv(gtfs_dir / "stops.txt", index=False)
    pd.DataFrame(trip_rows).to_csv(gtfs_dir / "trips.txt", index=False)
    pd.DataFrame(stop_time_rows).to_csv(gtfs_dir / "stop_times.txt", index=False)
    pd.DataFrame(shape_rows).to_csv(gtfs_dir / "shapes.txt", index=False)
    pd.DataFrame({"route_id": route_ids, "route_short_name": route_ids}).to_csv(
        gtfs_dir / "routes.txt", index=False
    )


# Stops along one street: A=001 (0 m), B=002 (100 m), C=003 (200 m),
# D=004 (300 m), X=006 (350 m), E=005 (400 m, no stop_code).
FEED_STOPS = {
    "001": ("0101", 0, 0),
    "002": ("0102", 100, 0),
    "003": ("0103", 200, 0),
    "004": ("0104", 300, 0),
    "006": ("0106", 350, 0),
    "005": ("", 400, 0),
}
FEED_TRIPS = [
    # R1 skips D; its most common shape belongs to the A-C short turns.
    ("R1", "0", "T1a", "10", ["001", "002", "003", "006", "005"]),
    ("R1", "0", "T1b", "11", ["001", "002", "003"]),
    ("R1", "0", "T1c", "11", ["001", "002", "003"]),
    ("R2", "0", "T2a", "20", ["001", "002", "003", "004", "005"]),
    # R3's third trip omits B.
    ("R3", "0", "T3a", "30", ["001", "002", "003", "004"]),
    ("R3", "0", "T3b", "30", ["001", "002", "003", "004"]),
    ("R3", "0", "T3c", "30", ["001", "003", "004"]),
]
FEED_SHAPES = {
    "10": [(0, 0), (400, 0)],
    "11": [(0, 0), (200, 0)],
    "20": [(0, 0), (400, 0)],
    "30": [(0, 0), (300, 0)],
}


def test_select_representative_shapes_uses_representative_trip() -> None:
    trips = pd.DataFrame(
        {
            "route_id": ["R1"] * 3,
            "direction_id": ["0"] * 3,
            "trip_id": ["short1", "short2", "long"],
            "shape_id": ["S", "S", "L"],
        }
    )
    reps = target.select_representative_shapes(trips, {("R1", "0"): "long"})
    assert reps.to_dict("records") == [{"route_id": "R1", "direction_id": "0", "shape_id": "L"}]


def test_prepare_gtfs_context_takes_shape_from_representative_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_feed(tmp_path, FEED_STOPS, FEED_TRIPS, FEED_SHAPES)
    monkeypatch.setattr(target, "GTFS_DIR", tmp_path)
    ctx = target.prepare_gtfs_context()
    assert ctx.route_sequences[("R1", "0")] == ["0101", "0102", "0103", "0106", "stop_id=005"]
    # T1a's own 400 m shape, not the more common 200 m short-turn shape.
    assert ctx.route_shapes_proj[("R1", "0")].length == pytest.approx(400.0, abs=0.5)


# ---------------------------------------------------------------------------
# route_pairs_to_compare / run_segment_comparison (synthetic feed)
# ---------------------------------------------------------------------------


def test_route_pairs_to_compare_lists_pairs_with_a_base_route_both_ways() -> None:
    a, b, c, w = ("A", "0"), ("B", "0"), ("C", "0"), ("W", "0")
    pairs = target.route_pairs_to_compare([a], [a, b, c, w], {"W"})
    assert pairs == [(a, b), (b, a), (a, c), (c, a)]
    # Two base routes: their shared pair is listed once in each order.
    pairs = target.route_pairs_to_compare([a, b], [a, b], set())
    assert pairs == [(a, b), (b, a)]


# Stops along one street (metres); X is the stop one route skips.
CORRIDOR_STOPS = {"A": ("A", 0, 0), "B": ("B", 100, 0), "X": ("X", 200, 0), "C": ("C", 300, 0)}
CORRIDOR_SHAPES = {"S": [(0, 0), (300, 0)]}


def _segment_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trips: list[tuple[str, str, str, str, list[str]]],
    stops: dict[str, tuple[str, float, float]] = CORRIDOR_STOPS,
    shapes: dict[str, list[tuple[float, float]]] = CORRIDOR_SHAPES,
    target_routes: frozenset[str] = frozenset(),
) -> list[tuple[str, str, str]]:
    """Return (missing route, reference route, candidate keys) per finding."""
    _write_feed(tmp_path, stops, trips, shapes)
    monkeypatch.setattr(target, "GTFS_DIR", tmp_path)
    monkeypatch.setattr(target, "TARGET_ROUTE_IDS", set(target_routes))
    found = target.run_segment_comparison(target.prepare_gtfs_context())
    return sorted(
        zip(
            found["missing_route_id"],
            found["reference_route_id"],
            found["candidate_missing_stop_keys"],
            strict=True,
        )
    )


@pytest.mark.parametrize("target_routes", [frozenset({"151"}), frozenset({"151", "161"}), None])
def test_run_segment_comparison_reports_omissions_on_both_routes_of_a_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_routes: frozenset[str] | None
) -> None:
    # Targeting 151 alone must still report the stop 161 skips.
    trips = [
        ("151", "0", "t151", "S", ["A", "B", "X", "C"]),
        ("161", "0", "t161", "S", ["A", "B", "C"]),
    ]
    flags = _segment_flags(tmp_path, monkeypatch, trips, target_routes=target_routes or frozenset())
    assert flags == [("161", "151", "X")]


@pytest.mark.parametrize(
    ("ref_shape", "skip_terminal_only", "expected"),
    [
        ("S", True, [("161", "151", "X")]),
        ("NO_SUCH_SHAPE", True, []),
        ("NO_SUCH_SHAPE", False, [("161", "151", "X")]),
    ],
    ids=["both-shapes", "reference-without-shape", "skip-setting-off"],
)
def test_run_segment_comparison_two_stop_corridor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ref_shape: str,
    skip_terminal_only: bool,
    expected: list[tuple[str, str, str]],
) -> None:
    # The routes share only their terminals, A and C, on the same street.
    monkeypatch.setattr(target, "SKIP_TERMINAL_ONLY_PAIRS", skip_terminal_only)
    trips = [
        ("151", "0", "t151", ref_shape, ["A", "X", "C"]),
        ("161", "0", "t161", "S", ["A", "C"]),
    ]
    assert _segment_flags(tmp_path, monkeypatch, trips) == expected


def test_run_segment_comparison_terminal_only_pair_on_other_street_is_not_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 151 detours 300 m north between the shared terminals; geometry rules it out.
    stops = {"A": ("A", 0, 0), "X": ("X", 150, 300), "C": ("C", 300, 0)}
    shapes = {"S": [(0, 0), (300, 0)], "N": [(0, 0), (0, 300), (300, 300), (300, 0)]}
    trips = [
        ("151", "0", "t151", "N", ["A", "X", "C"]),
        ("161", "0", "t161", "S", ["A", "C"]),
    ]
    assert _segment_flags(tmp_path, monkeypatch, trips, stops, shapes) == []


@pytest.mark.parametrize(
    ("compare_reversed", "expected"),
    [(False, []), (True, [("151", "161", "X")])],
    ids=["default", "compare-reversed"],
)
@pytest.mark.parametrize("reference_route", ["161", "151"], ids=["other-route", "return-trip"])
def test_run_segment_comparison_reversed_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compare_reversed: bool,
    expected: list[tuple[str, str, str]],
    reference_route: str,
) -> None:
    # The reference runs C -> A on the north curb and serves X, a stop for
    # that direction only, 15 m from the south-curb route's shape.
    monkeypatch.setattr(target, "COMPARE_REVERSED_REFERENCES", compare_reversed)
    stops = {"A": ("A", 0, 5), "B": ("B", 100, 5), "C": ("C", 300, 5), "X": ("X", 250, 15)}
    shapes = {"E": [(0, 0), (300, 0)], "W": [(300, 10), (0, 10)]}
    trips = [
        ("151", "0", "t151", "E", ["A", "B", "C"]),
        (reference_route, "1", "tref", "W", ["C", "X", "B", "A"]),
    ]
    expected = [(m, reference_route, keys) for m, _, keys in expected]
    assert _segment_flags(tmp_path, monkeypatch, trips, stops, shapes) == expected


def test_run_segment_comparison_loop_is_not_mistaken_for_a_reversed_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 151 runs a square loop A -> B -> C -> A; 161 serves X between B and C.
    # With the default COMPARE_REVERSED_REFERENCES = False, 161 must still be
    # compared as a same-direction reference.
    monkeypatch.setattr(target, "COMPARE_REVERSED_REFERENCES", False)
    stops = {"A": ("A", 0, 0), "B": ("B", 300, 0), "X": ("X", 300, 150), "C": ("C", 300, 300)}
    shapes = {
        "LOOP": [(0, 0), (300, 0), (300, 300), (0, 300), (0, 0)],
        "L": [(0, 0), (300, 0), (300, 300)],
    }
    trips = [
        ("151", "0", "t151", "LOOP", ["A", "B", "C", "A"]),
        ("161", "0", "t161", "L", ["A", "B", "X", "C"]),
    ]
    assert _segment_flags(tmp_path, monkeypatch, trips, stops, shapes) == [("151", "161", "X")]


@pytest.mark.parametrize("projected_crs", ["EPSG:26918", "EPSG:2283"], ids=["metres", "us-feet"])
@pytest.mark.parametrize(("offset_m", "expected"), [(20, [("161", "151", "X")]), (40, [])])
def test_run_segment_comparison_thresholds_are_metres_in_any_crs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projected_crs: str,
    offset_m: float,
    expected: list[tuple[str, str, str]],
) -> None:
    # X sits offset_m from the street; MAX_STOP_TO_SHAPE_M is 30 m whether
    # distances are measured in metres or in Virginia State Plane US feet.
    monkeypatch.setattr(target, "PROJECTED_CRS", projected_crs)
    stops = {**CORRIDOR_STOPS, "X": ("X", 200, offset_m)}
    trips = [
        ("151", "0", "t151", "S", ["A", "B", "X", "C"]),
        ("161", "0", "t161", "S", ["A", "B", "C"]),
    ]
    assert _segment_flags(tmp_path, monkeypatch, trips, stops) == expected


def test_prepare_gtfs_context_rejects_a_geographic_projected_crs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target, "PROJECTED_CRS", "EPSG:4326")
    with pytest.raises(ValueError, match="not a projected CRS"):
        target.prepare_gtfs_context()


@pytest.mark.parametrize(
    "drop_shape_id", [False, True], ids=["no-shapes-txt", "no-shape-id-either"]
)
def test_main_runs_within_route_check_without_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    drop_shape_id: bool,
) -> None:
    gtfs_dir = tmp_path / "gtfs"
    shutil.copytree(Path(__file__).parent / "fixtures" / "gtfs_basic", gtfs_dir)
    (gtfs_dir / "shapes.txt").unlink()
    if drop_shape_id:
        trips = pd.read_csv(gtfs_dir / "trips.txt", dtype=str)
        trips.drop(columns="shape_id").to_csv(gtfs_dir / "trips.txt", index=False)
    # T3 skips S4 (stop_code 1004), which R1's other trips serve.
    stop_times = pd.read_csv(gtfs_dir / "stop_times.txt", dtype=str)
    stop_times = stop_times[~((stop_times["trip_id"] == "T3") & (stop_times["stop_id"] == "S4"))]
    stop_times.to_csv(gtfs_dir / "stop_times.txt", index=False)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(target, "GTFS_DIR", gtfs_dir)
    monkeypatch.setattr(target, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(target, "PLOT_DIR", out_dir / "segment_plots")
    monkeypatch.setattr(target, "TARGET_ROUTE_IDS", set())
    monkeypatch.setattr(target, "ROUTE_ID_WHITELIST", set())
    with caplog.at_level(logging.WARNING):
        assert target.main() == 0

    intra = pd.read_csv(out_dir / target.INTRA_ROUTE_FILENAME, dtype=str)
    assert list(zip(intra["trip_id"], intra["missing_stop_keys"], strict=True)) == [("T3", "1004")]
    assert pd.read_csv(out_dir / target.OUTPUT_FILENAME).empty
    assert "3 of 3 route/direction(s) have no usable shape" in caplog.text
    assert "geometry-based checks did not run" in caplog.text


# ---------------------------------------------------------------------------
# main (synthetic feed)
# ---------------------------------------------------------------------------


def _run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_routes: set[str],
    excluded_routes: set[str],
) -> Path:
    """Run main() on the synthetic feed and return the output directory."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_feed(gtfs_dir, FEED_STOPS, FEED_TRIPS, FEED_SHAPES)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(target, "GTFS_DIR", gtfs_dir)
    monkeypatch.setattr(target, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(target, "PLOT_DIR", out_dir / "segment_plots")
    monkeypatch.setattr(target, "TARGET_ROUTE_IDS", target_routes)
    monkeypatch.setattr(target, "ROUTE_ID_WHITELIST", excluded_routes)
    assert target.main() == 0
    return out_dir


@pytest.mark.parametrize(
    ("target_routes", "excluded_routes", "r3_reported"),
    [(set(), set(), True), ({"R1"}, set(), False), (set(), {"R3"}, False)],
    ids=["all-routes", "target-routes", "route-whitelist"],
)
def test_main_limits_intra_route_check_to_analyzed_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_routes: set[str],
    excluded_routes: set[str],
    r3_reported: bool,
) -> None:
    out_dir = _run_main(tmp_path, monkeypatch, target_routes, excluded_routes)

    segments = pd.read_csv(out_dir / target.OUTPUT_FILENAME, dtype=str, keep_default_na=False)
    flagged = set(zip(segments["missing_route_id"], segments["candidate_missing_stop_keys"]))
    assert ("R1", "0104") in flagged  # R1 skips D, which R2 serves
    # R3's trip T3c omits B; it is reported only when R3 is analyzed.
    intra = pd.read_csv(out_dir / target.INTRA_ROUTE_FILENAME, dtype=str)
    assert ("T3c" in set(intra["trip_id"])) is r3_reported


def test_main_writes_csv_headers_when_nothing_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _run_main(tmp_path, monkeypatch, {"NO_SUCH_ROUTE"}, set())
    for filename, columns in [
        (target.OUTPUT_FILENAME, target.SEGMENT_COLUMNS),
        (target.AGGREGATE_FILENAME, target.AGGREGATE_COLUMNS),
        (target.INTRA_ROUTE_FILENAME, target.INTRA_ROUTE_COLUMNS),
    ]:
        written = pd.read_csv(out_dir / filename)
        assert written.empty
        assert list(written.columns) == columns


def test_main_rewrites_csvs_and_keeps_plots_apart_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    plot_dir = out_dir / "segment_plots"
    monkeypatch.setattr(target, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(target, "PLOT_DIR", plot_dir)
    monkeypatch.setattr(target, "TARGET_ROUTE_IDS", set())
    runs = {
        "mismatch": [
            ("151", "0", "a", "S", ["A", "B", "X", "C"]),
            ("161", "0", "b", "S", ["A", "B", "C"]),
        ],
        "clean": [
            ("151", "0", "a", "S", ["A", "B", "X", "C"]),
            ("161", "0", "b", "S", ["A", "B", "X", "C"]),
        ],
        # Single-stop trips: no route sequence can be built.
        "no_sequences": [("151", "0", "a", "S", ["A"]), ("161", "0", "b", "S", ["B"])],
    }
    segment_rows = {}
    plot_folders = []
    for name, trips in runs.items():
        gtfs_dir = tmp_path / name
        gtfs_dir.mkdir()
        _write_feed(gtfs_dir, CORRIDOR_STOPS, trips, CORRIDOR_SHAPES)
        monkeypatch.setattr(target, "GTFS_DIR", gtfs_dir)
        before = set(plot_dir.iterdir()) if plot_dir.exists() else set()
        assert target.main() == 0
        plot_folders.append(set(plot_dir.iterdir()) - before)
        segment_rows[name] = len(pd.read_csv(out_dir / target.OUTPUT_FILENAME))

    # The no-sequence run replaced the earlier CSV instead of leaving it.
    assert segment_rows == {"mismatch": 1, "clean": 0, "no_sequences": 0}
    # Each run that plotted wrote to its own new folder; the clean run's
    # folder holds no segment plot from the mismatch run.
    (mismatch_dir,), (clean_dir,), no_sequence_dirs = plot_folders
    assert [p.name for p in mismatch_dir.glob("seg_*.png")] == ["seg_161_0_vs_151_0_B_C.png"]
    assert list(clean_dir.glob("seg_*.png")) == []
    assert no_sequence_dirs == set()


def test_new_plot_run_dir_never_reuses_an_existing_folder(tmp_path: Path) -> None:
    first = target.new_plot_run_dir(tmp_path)
    assert first.name.startswith("run_")
    assert not first.exists()
    first.mkdir()
    second = target.new_plot_run_dir(tmp_path)
    assert second != first
    assert not second.exists()
