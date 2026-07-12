from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.stop_analysis.gtfs_stop_diff import (
    _route_display_label,
    _route_sort_key,
    attach_route_context,
    build_modified_description,
    build_stop_routes_table,
    coerce_float,
    compare_stops,
    haversine_meters,
    meters_to_feet,
    normalize_text,
    pick_attribute_columns,
    try_build_nearest_matches,
    validate_stop_ids_unique,
)

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_strips_whitespace() -> None:
    s = pd.Series(["  Main St  ", "  Oak Ave"])
    result = normalize_text(s)
    assert result.tolist() == ["Main St", "Oak Ave"]


def test_normalize_text_fills_nan_with_empty_string() -> None:
    s = pd.Series([None, "Stop A", float("nan")])
    result = normalize_text(s)
    assert result[0] == ""
    assert result[2] == ""


def test_normalize_text_converts_non_string_types() -> None:
    s = pd.Series([123, 456])
    result = normalize_text(s)
    assert result.tolist() == ["123", "456"]


# ---------------------------------------------------------------------------
# coerce_float
# ---------------------------------------------------------------------------


def test_coerce_float_converts_numeric_strings() -> None:
    s = pd.Series(["38.7", "39.0"])
    result = coerce_float(s)
    assert result[0] == pytest.approx(38.7)
    assert result[1] == pytest.approx(39.0)


def test_coerce_float_non_numeric_becomes_nan() -> None:
    s = pd.Series(["bad", "N/A"])
    result = coerce_float(s)
    assert np.isnan(result[0])
    assert np.isnan(result[1])


def test_coerce_float_returns_float_dtype() -> None:
    s = pd.Series(["1.0", "2.0"])
    result = coerce_float(s)
    assert result.dtype == float


# ---------------------------------------------------------------------------
# validate_stop_ids_unique
# ---------------------------------------------------------------------------


def test_validate_stop_ids_unique_keeps_first_occurrence() -> None:
    df = pd.DataFrame({"stop_id": ["S1", "S1", "S2"], "stop_name": ["Alpha", "Beta", "Gamma"]})
    result = validate_stop_ids_unique(df, "test")
    assert len(result) == 2
    assert result.loc[result.stop_id == "S1", "stop_name"].iloc[0] == "Alpha"


def test_validate_stop_ids_unique_no_duplicates_unchanged() -> None:
    df = pd.DataFrame({"stop_id": ["S1", "S2"], "stop_name": ["A", "B"]})
    result = validate_stop_ids_unique(df, "test")
    assert len(result) == 2


def test_validate_stop_ids_unique_raises_when_column_missing() -> None:
    df = pd.DataFrame({"name": ["A", "B"]})
    with pytest.raises(ValueError, match="stop_id"):
        validate_stop_ids_unique(df, "test")


# ---------------------------------------------------------------------------
# haversine_meters
# ---------------------------------------------------------------------------


def test_haversine_meters_same_point_is_zero() -> None:
    dist = haversine_meters(
        np.array([38.7]), np.array([-77.0]), np.array([38.7]), np.array([-77.0])
    )
    assert dist[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_meters_one_degree_latitude() -> None:
    # 1 degree of latitude ≈ 111,195 m at the equator
    dist = haversine_meters(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert dist[0] == pytest.approx(111_195.0, rel=0.01)


def test_haversine_meters_returns_array() -> None:
    dist = haversine_meters(
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 2.0]),
        np.array([0.0, 0.0]),
    )
    assert len(dist) == 2
    assert all(d > 0 for d in dist)


# ---------------------------------------------------------------------------
# meters_to_feet
# ---------------------------------------------------------------------------


def test_meters_to_feet_one_meter() -> None:
    result = meters_to_feet(np.array([1.0]))
    assert result[0] == pytest.approx(3.28084, rel=0.001)


def test_meters_to_feet_zero() -> None:
    result = meters_to_feet(np.array([0.0]))
    assert result[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# pick_attribute_columns
# ---------------------------------------------------------------------------


def test_pick_attribute_columns_returns_shared_columns() -> None:
    before = pd.DataFrame(columns=["stop_id", "stop_name", "stop_code", "zone_id"])
    after = pd.DataFrame(columns=["stop_id", "stop_name", "zone_id", "location_type"])
    cols = pick_attribute_columns(before, after)
    assert "stop_name" in cols
    assert "zone_id" in cols


def test_pick_attribute_columns_excludes_before_only_columns() -> None:
    before = pd.DataFrame(columns=["stop_id", "stop_name", "stop_code"])
    after = pd.DataFrame(columns=["stop_id", "stop_name"])
    cols = pick_attribute_columns(before, after)
    assert "stop_code" not in cols


def test_pick_attribute_columns_excludes_after_only_columns() -> None:
    before = pd.DataFrame(columns=["stop_id", "stop_name"])
    after = pd.DataFrame(columns=["stop_id", "stop_name", "location_type"])
    cols = pick_attribute_columns(before, after)
    assert "location_type" not in cols


# ---------------------------------------------------------------------------
# build_modified_description
# ---------------------------------------------------------------------------


def test_build_modified_description_relocated_with_distance() -> None:
    desc = build_modified_description(relocated=True, changed_fields=[], distance_ft=150.5)
    assert "150.5" in desc
    assert "Relocated" in desc


def test_build_modified_description_attr_change_only() -> None:
    desc = build_modified_description(
        relocated=False, changed_fields=["stop_name"], distance_ft=None
    )
    assert "stop_name" in desc
    assert "Relocated" not in desc


def test_build_modified_description_both_changes() -> None:
    desc = build_modified_description(
        relocated=True, changed_fields=["stop_name"], distance_ft=50.0
    )
    assert "Relocated" in desc
    assert "stop_name" in desc


def test_build_modified_description_no_changes_returns_empty() -> None:
    desc = build_modified_description(relocated=False, changed_fields=[], distance_ft=None)
    assert desc == ""


# ---------------------------------------------------------------------------
# compare_stops
# ---------------------------------------------------------------------------


def _make_stop_df(**kwargs: list) -> pd.DataFrame:
    return pd.DataFrame(
        {"stop_id": kwargs["ids"], "stop_lat": kwargs["lats"], "stop_lon": kwargs["lons"]}
    )


def test_compare_stops_identifies_deleted_stops() -> None:
    before = _make_stop_df(ids=["S1", "S2"], lats=[38.7, 38.8], lons=[-77.0, -77.1])
    after = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    _, deleted, _, _, summary, _ = compare_stops(before, after, relocate_threshold_ft=25.0)
    assert len(deleted) == 1
    assert deleted["stop_id"].iloc[0] == "S2"
    assert summary.deleted_count == 1


def test_compare_stops_identifies_new_stops() -> None:
    before = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    after = _make_stop_df(ids=["S1", "S2"], lats=[38.7, 38.8], lons=[-77.0, -77.1])
    _, _, new, _, summary, _ = compare_stops(before, after, relocate_threshold_ft=25.0)
    assert len(new) == 1
    assert new["stop_id"].iloc[0] == "S2"
    assert summary.new_count == 1


def test_compare_stops_identifies_relocated_stop() -> None:
    before = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    # Moved ~1 degree north → many thousands of feet
    after = _make_stop_df(ids=["S1"], lats=[38.8], lons=[-77.0])
    modified, _, _, _, summary, _ = compare_stops(before, after, relocate_threshold_ft=25.0)
    assert summary.modified_count == 1
    assert summary.relocated_count == 1


def test_compare_stops_unchanged_stop_not_in_modified() -> None:
    before = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    after = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    modified, _, _, unchanged, summary, _ = compare_stops(before, after, relocate_threshold_ft=25.0)
    assert len(modified) == 0
    assert len(unchanged) == 1
    assert summary.unchanged_count == 1


def test_compare_stops_summary_counts_are_consistent() -> None:
    before = _make_stop_df(
        ids=["S1", "S2", "S3"], lats=[38.7, 38.8, 38.9], lons=[-77.0, -77.1, -77.2]
    )
    after = _make_stop_df(
        ids=["S1", "S3", "S4"], lats=[38.7, 38.9, 39.0], lons=[-77.0, -77.2, -77.3]
    )
    _, deleted, new, _, summary, _ = compare_stops(before, after, relocate_threshold_ft=25.0)
    assert summary.deleted_count == len(deleted)
    assert summary.new_count == len(new)
    assert summary.before_stop_count == 3
    assert summary.after_stop_count == 3


# ---------------------------------------------------------------------------
# route context helpers
# ---------------------------------------------------------------------------


def test_route_display_label_prefers_short_name() -> None:
    row = pd.Series({"route_short_name": "10", "route_long_name": "Elm Line", "route_id": "R1"})
    assert _route_display_label(row) == "10"


def test_route_display_label_falls_back_to_long_name() -> None:
    row = pd.Series({"route_short_name": "", "route_long_name": "Elm Line", "route_id": "R1"})
    assert _route_display_label(row) == "Elm Line"


def test_route_display_label_falls_back_to_route_id() -> None:
    row = pd.Series({"route_short_name": "", "route_long_name": "", "route_id": "R1"})
    assert _route_display_label(row) == "R1"


def test_route_sort_key_orders_numeric_before_alpha() -> None:
    labels = ["Red", "2", "10", "1"]
    assert sorted(labels, key=_route_sort_key) == ["1", "2", "10", "Red"]


def test_build_stop_routes_table_aggregates_routes_and_count() -> None:
    pairs = pd.DataFrame(
        {
            "stop_id": ["S1", "S1", "S2"],
            "route_id": ["R2", "R10", "R1"],
            "route_label": ["2", "10", "1"],
        }
    )
    table = build_stop_routes_table(pairs)
    s1 = table.loc[table.stop_id == "S1"].iloc[0]
    assert s1["routes"] == "2; 10"  # numeric sort, not lexical
    assert s1["route_count"] == 2


def test_build_stop_routes_table_none_passthrough() -> None:
    assert build_stop_routes_table(None) is None


def test_attach_route_context_left_join_fills_unserved() -> None:
    stops = _make_stop_df(ids=["S1", "S2"], lats=[38.7, 38.8], lons=[-77.0, -77.1])
    table = pd.DataFrame({"stop_id": ["S1"], "routes": ["10"], "route_count": [1]})
    out = attach_route_context(stops, table, label="test")
    assert out.loc[out.stop_id == "S1", "route_count"].iloc[0] == 1
    # S2 has no service → blank routes, zero count (not NaN)
    assert out.loc[out.stop_id == "S2", "routes"].iloc[0] == ""
    assert out.loc[out.stop_id == "S2", "route_count"].iloc[0] == 0


def test_attach_route_context_none_is_noop() -> None:
    stops = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    out = attach_route_context(stops, None, label="test")
    assert "routes" not in out.columns
    assert out.equals(stops)


def test_route_context_does_not_affect_classification() -> None:
    # Same coords/attrs but different serving routes → still unchanged.
    before = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    before["routes"] = ["10"]
    before["route_count"] = [1]
    after = _make_stop_df(ids=["S1"], lats=[38.7], lons=[-77.0])
    after["routes"] = ["10; 20"]
    after["route_count"] = [2]
    modified, _, _, unchanged, summary, _ = compare_stops(before, after, relocate_threshold_ft=25.0)
    assert summary.modified_count == 0
    assert len(unchanged) == 1
    # Route columns still carried through to the merged output.
    assert "routes_before" in unchanged.columns
    assert "routes_after" in unchanged.columns


def test_nearest_matches_include_routes_when_present() -> None:
    before = _make_stop_df(ids=["B1"], lats=[38.70000], lons=[-77.00000])
    before["routes"] = ["10"]
    after = _make_stop_df(ids=["A1"], lats=[38.70001], lons=[-77.00001])
    after["routes"] = ["10; 20"]
    out = try_build_nearest_matches(before, after, max_feet=500.0)
    assert out is not None
    assert out["after_routes"].iloc[0] == "10; 20"
    assert out["nearest_before_routes"].iloc[0] == "10"
