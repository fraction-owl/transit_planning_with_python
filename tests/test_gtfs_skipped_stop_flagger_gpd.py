from __future__ import annotations

import logging
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
    shapes = {("A", "0"): line, ("B", "0"): line}
    distance = target.segment_hausdorff_distance(
        ("A", "0"), ("B", "0"), "1", "2", shapes, gdf_proj, padding_m=50.0
    )
    assert distance == pytest.approx(0.0)


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
