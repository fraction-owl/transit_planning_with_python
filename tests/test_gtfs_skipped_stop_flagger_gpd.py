from __future__ import annotations

import pandas as pd
import pytest

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
    from shapely.geometry import LineString

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
