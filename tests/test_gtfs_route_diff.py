from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.stop_analysis.gtfs_route_diff import (
    Correspondence,
    build_correspondence,
    classify_alignment,
    classify_schedule,
    compare_calendars,
    compare_fare_attributes,
    compare_route_names_attrs,
    compare_routes,
    date_range_relationship,
    detect_splits_merges,
    feed_date_range,
    jaccard,
    median_headway_min,
    normalize_text,
    route_day_types,
    route_fare_map,
    route_interline_partners,
    route_schedule_metrics,
    route_stop_sets,
    route_terminals,
    schedule_fingerprint,
    service_id_day_types,
    summarize_change,
    time_to_seconds,
    trip_endpoints,
)

# Threshold knobs matching the module defaults, for classifier tests.
KNOBS = {
    "rekey_jaccard": 0.50,
    "align_minor_jaccard": 0.80,
    "trips_major": 0.30,
    "span_major": 60.0,
    "headway_major": 0.50,
    "fare_price_major": 0.25,
    "split_merge_enable": 1.0,
    "split_merge_min_coverage": 0.60,
    "split_merge_min_share": 0.30,
}


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


def test_jaccard_identical_sets_is_one() -> None:
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0


def test_jaccard_two_empty_sets_is_one() -> None:
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_jaccard_disjoint_is_zero() -> None:
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_partial_overlap() -> None:
    # {a,b,c} vs {b,c,d} -> intersection 2, union 4
    assert jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# normalize_text / time_to_seconds / median_headway_min
# ---------------------------------------------------------------------------


def test_normalize_text_strips_and_fills() -> None:
    s = pd.Series(["  10 ", None, "20"])
    assert normalize_text(s).tolist() == ["10", "", "20"]


def test_time_to_seconds_extended_clock() -> None:
    s = pd.Series(["08:00:00", "25:30:00"])
    result = time_to_seconds(s)
    assert result[0] == pytest.approx(28800.0)
    assert result[1] == pytest.approx(91800.0)  # 25h30m past midnight


def test_median_headway_min_needs_three_trips() -> None:
    assert np.isnan(median_headway_min(pd.Series([0.0, 600.0])))


def test_median_headway_min_even_spacing() -> None:
    # starts every 600 s -> 10 minute headway
    starts = pd.Series([0.0, 600.0, 1200.0, 1800.0])
    assert median_headway_min(starts) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# classify_alignment
# ---------------------------------------------------------------------------


def test_classify_alignment_identical_is_none() -> None:
    stops = frozenset({"s1", "s2", "s3"})
    assert classify_alignment(stops, stops, terminals_changed=False, minor_jaccard=0.8) == "none"


def test_classify_alignment_small_trim_is_minor() -> None:
    before = frozenset({"s1", "s2", "s3", "s4", "s5"})
    after = frozenset({"s1", "s2", "s3", "s4"})  # jaccard 4/5 = 0.8
    assert classify_alignment(before, after, terminals_changed=True, minor_jaccard=0.8) == "minor"


def test_classify_alignment_large_change_is_major() -> None:
    before = frozenset({"s1", "s2", "s3"})
    after = frozenset({"s4", "s5", "s6"})
    assert classify_alignment(before, after, terminals_changed=True, minor_jaccard=0.8) == "major"


def test_classify_alignment_terminal_only_change_still_classified() -> None:
    # Identical stops but terminals changed -> not "none"; jaccard 1.0 -> minor.
    stops = frozenset({"s1", "s2"})
    assert classify_alignment(stops, stops, terminals_changed=True, minor_jaccard=0.8) == "minor"


# ---------------------------------------------------------------------------
# classify_schedule
# ---------------------------------------------------------------------------


def _sched_row(trips: float, span: float, headway: float) -> pd.Series:
    return pd.Series({"trips_per_day": trips, "span_hours": span, "median_headway_min": headway})


def test_classify_schedule_identical_fingerprint_is_none() -> None:
    row = _sched_row(40, 18.0, 30.0)
    tier, _ = classify_schedule(row, row, identical=True, knobs=KNOBS)
    assert tier == "none"


def test_classify_schedule_shifted_trips_same_metrics_is_minor() -> None:
    # Metrics match exactly, but the fingerprint differs -> a real (minor) change,
    # NOT "none". This is the "literally nothing changed" guard.
    row = _sched_row(40, 18.0, 30.0)
    tier, detail = classify_schedule(row, row, identical=False, knobs=KNOBS)
    assert tier == "minor"
    assert detail  # some description is emitted even when summary metrics match


def test_classify_schedule_missing_side_is_major() -> None:
    tier, detail = classify_schedule(None, _sched_row(40, 18, 30), identical=False, knobs=KNOBS)
    assert tier == "major"
    assert "weekday service" in detail


def test_classify_schedule_both_missing_is_none() -> None:
    tier, _ = classify_schedule(None, None, identical=False, knobs=KNOBS)
    assert tier == "none"


def test_classify_schedule_big_trip_cut_is_major() -> None:
    before = _sched_row(40, 18.0, 30.0)
    after = _sched_row(20, 18.0, 30.0)  # 50% trip cut
    tier, detail = classify_schedule(before, after, identical=False, knobs=KNOBS)
    assert tier == "major"
    assert "trips/day" in detail


def test_classify_schedule_small_span_change_is_minor() -> None:
    before = _sched_row(40, 18.0, 30.0)
    after = _sched_row(40, 18.5, 30.0)  # +30 min span, below major(60)
    tier, _ = classify_schedule(before, after, identical=False, knobs=KNOBS)
    assert tier == "minor"


def test_schedule_fingerprint_distinguishes_shifted_trips() -> None:
    trips = pd.DataFrame({"route_id": ["R1"], "trip_id": ["t1"]})
    st_early = _stop_times([("t1", "a", "1", "08:00:00"), ("t1", "b", "2", "08:30:00")])
    st_late = _stop_times([("t1", "a", "1", "08:10:00"), ("t1", "b", "2", "08:40:00")])
    fp_early = schedule_fingerprint(trips, trip_endpoints(st_early))
    fp_late = schedule_fingerprint(trips, trip_endpoints(st_late))
    assert fp_early["R1"] == fp_early["R1"]
    assert fp_early["R1"] != fp_late["R1"]  # a 10-minute shift is detectable


# ---------------------------------------------------------------------------
# trip_endpoints / route_stop_sets / route_terminals / route_schedule_metrics
# ---------------------------------------------------------------------------


def _stop_times(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    # rows: (trip_id, stop_id, stop_sequence, time)
    return pd.DataFrame(
        rows, columns=["trip_id", "stop_id", "stop_sequence", "departure_time"]
    ).assign(arrival_time=lambda d: d["departure_time"])


def test_trip_endpoints_first_last_and_runtime() -> None:
    st = _stop_times(
        [
            ("t1", "a", "1", "08:00:00"),
            ("t1", "b", "2", "08:30:00"),
            ("t1", "c", "3", "09:00:00"),
        ]
    )
    ends = trip_endpoints(st).set_index("trip_id")
    assert ends.loc["t1", "first_stop"] == "a"
    assert ends.loc["t1", "last_stop"] == "c"
    assert ends.loc["t1", "runtime_sec"] == pytest.approx(3600.0)


def test_route_stop_sets_unions_trips() -> None:
    trips = pd.DataFrame({"route_id": ["R1", "R1"], "trip_id": ["t1", "t2"]})
    st = _stop_times(
        [
            ("t1", "a", "1", "08:00:00"),
            ("t1", "b", "2", "08:10:00"),
            ("t2", "b", "1", "09:00:00"),
            ("t2", "c", "2", "09:10:00"),
        ]
    )
    sets = route_stop_sets(trips, st)
    assert sets["R1"] == frozenset({"a", "b", "c"})


def test_route_terminals_uses_modal_pair() -> None:
    trips = pd.DataFrame(
        {"route_id": ["R1", "R1"], "direction_id": ["0", "0"], "trip_id": ["t1", "t2"]}
    )
    st = _stop_times(
        [
            ("t1", "a", "1", "08:00:00"),
            ("t1", "z", "2", "08:30:00"),
            ("t2", "a", "1", "09:00:00"),
            ("t2", "z", "2", "09:30:00"),
        ]
    )
    terms = route_terminals(trips, trip_endpoints(st))
    assert terms["R1"] == frozenset({("0", "a", "z")})


def test_route_schedule_metrics_counts_trips() -> None:
    trips = pd.DataFrame({"route_id": ["R1", "R1"], "trip_id": ["t1", "t2"]})
    st = _stop_times(
        [
            ("t1", "a", "1", "08:00:00"),
            ("t1", "b", "2", "08:30:00"),
            ("t2", "a", "1", "09:00:00"),
            ("t2", "b", "2", "09:30:00"),
        ]
    )
    metrics = route_schedule_metrics(trips, trip_endpoints(st)).set_index("route_id")
    assert metrics.loc["R1", "trips_per_day"] == 2
    assert metrics.loc["R1", "span_hours"] == pytest.approx(1.5)  # 08:00 -> 09:30


# ---------------------------------------------------------------------------
# route_interline_partners (reblocking input)
# ---------------------------------------------------------------------------


def test_route_interline_partners_shared_block() -> None:
    trips = pd.DataFrame(
        {
            "route_id": ["R1", "R2", "R3"],
            "block_id": ["B1", "B1", "B2"],
        }
    )
    partners = route_interline_partners(trips)
    assert partners["R1"] == frozenset({"R2"})
    assert partners["R3"] == frozenset()


def test_route_interline_partners_none_without_block_id() -> None:
    trips = pd.DataFrame({"route_id": ["R1"], "trip_id": ["t1"]})
    assert route_interline_partners(trips) is None


# ---------------------------------------------------------------------------
# build_correspondence (matched / rekeyed / eliminated / added)
# ---------------------------------------------------------------------------


def test_build_correspondence_matched_and_added_and_eliminated() -> None:
    routes_b = pd.DataFrame({"route_id": ["R1", "R2"], "route_short_name": ["1", "2"]})
    routes_a = pd.DataFrame({"route_id": ["R1", "R3"], "route_short_name": ["1", "3"]})
    stops_b = {"R1": frozenset({"a"}), "R2": frozenset({"b"})}
    stops_a = {"R1": frozenset({"a"}), "R3": frozenset({"c"})}
    corr = build_correspondence(routes_b, routes_a, stops_b, stops_a, rekey_min_jaccard=0.5)
    assert corr.matched == ["R1"]
    assert corr.eliminated == ["R2"]
    assert corr.added == ["R3"]
    assert corr.rekeyed == {}


def test_build_correspondence_detects_rekey() -> None:
    # Same public number ("5") and overlapping stops -> rekey, not elim+add.
    routes_b = pd.DataFrame({"route_id": ["OLD5"], "route_short_name": ["5"]})
    routes_a = pd.DataFrame({"route_id": ["NEW5"], "route_short_name": ["5"]})
    stops_b = {"OLD5": frozenset({"a", "b", "c", "d"})}
    stops_a = {"NEW5": frozenset({"a", "b", "c", "e"})}  # jaccard 3/5 = 0.6 >= 0.5
    corr = build_correspondence(routes_b, routes_a, stops_b, stops_a, rekey_min_jaccard=0.5)
    assert corr.rekeyed == {"OLD5": "NEW5"}
    assert corr.eliminated == []
    assert corr.added == []


def test_build_correspondence_no_rekey_when_jaccard_below_threshold() -> None:
    routes_b = pd.DataFrame({"route_id": ["OLD5"], "route_short_name": ["5"]})
    routes_a = pd.DataFrame({"route_id": ["NEW5"], "route_short_name": ["5"]})
    stops_b = {"OLD5": frozenset({"a", "b"})}
    stops_a = {"NEW5": frozenset({"c", "d"})}  # jaccard 0
    corr = build_correspondence(routes_b, routes_a, stops_b, stops_a, rekey_min_jaccard=0.5)
    assert corr.rekeyed == {}
    assert corr.eliminated == ["OLD5"]
    assert corr.added == ["NEW5"]


# ---------------------------------------------------------------------------
# compare_route_names_attrs
# ---------------------------------------------------------------------------


def test_compare_route_names_attrs_detects_short_name_and_type() -> None:
    before = pd.Series({"route_short_name": "10", "route_long_name": "Main", "route_type": "3"})
    after = pd.Series({"route_short_name": "10A", "route_long_name": "Main", "route_type": "700"})
    short_changed, long_changed, attrs = compare_route_names_attrs(
        before, after, ("route_type", "route_color")
    )
    assert short_changed is True
    assert long_changed is False
    assert attrs == ["route_type"]


# ---------------------------------------------------------------------------
# calendar approach diff
# ---------------------------------------------------------------------------


def _cal(service_id: str, days: str, start: str, end: str) -> dict:
    flags = {
        col: days[i]
        for i, col in enumerate(
            ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        )
    }
    return {"service_id": service_id, **flags, "start_date": start, "end_date": end}


def test_compare_calendars_redefined_days() -> None:
    before = pd.DataFrame([_cal("1", "1111100", "20240101", "20240601")])  # weekday
    after = pd.DataFrame([_cal("1", "1111110", "20240101", "20240601")])  # + Saturday
    result = compare_calendars(before, after)
    row = result[result["service_id"] == "1"].iloc[0]
    assert row["status"] == "redefined"
    assert "days" in row["detail"]


def test_compare_calendars_added_and_removed() -> None:
    before = pd.DataFrame([_cal("1", "1111100", "20240101", "20240601")])
    after = pd.DataFrame([_cal("2", "0000011", "20240101", "20240601")])
    result = compare_calendars(before, after).set_index("service_id")
    assert result.loc["1", "status"] == "removed"
    assert result.loc["2", "status"] == "added"


# ---------------------------------------------------------------------------
# day-type coverage
# ---------------------------------------------------------------------------


def test_service_id_day_types_from_calendar() -> None:
    cal = pd.DataFrame([_cal("WK", "1111100", "20240101", "20240601")])
    types = service_id_day_types(cal, None)
    assert types["WK"] == frozenset({"weekday"})


def test_route_day_types_unions_service_ids() -> None:
    trips = pd.DataFrame({"route_id": ["R1", "R1"], "service_id": ["WK", "SA"]})
    sid_types = {"WK": frozenset({"weekday"}), "SA": frozenset({"saturday"})}
    result = route_day_types(trips, sid_types)
    assert result["R1"] == frozenset({"weekday", "saturday"})


# ---------------------------------------------------------------------------
# fares
# ---------------------------------------------------------------------------


def test_compare_fare_attributes_price_change_major() -> None:
    before = pd.DataFrame({"fare_id": ["full"], "price": ["2.00"], "currency_type": ["USD"]})
    after = pd.DataFrame({"fare_id": ["full"], "price": ["2.50"], "currency_type": ["USD"]})
    result = compare_fare_attributes(before, after, price_major_delta=0.25)
    row = result[result["fare_id"] == "full"].iloc[0]
    assert row["status"] == "changed"
    assert row["price_delta"] == pytest.approx(0.50)
    assert row["severity"] == "major"


def test_compare_fare_attributes_added_removed() -> None:
    before = pd.DataFrame({"fare_id": ["a"], "price": ["1.00"]})
    after = pd.DataFrame({"fare_id": ["b"], "price": ["1.00"]})
    result = compare_fare_attributes(before, after, price_major_delta=0.25).set_index("fare_id")
    assert result.loc["a", "status"] == "removed"
    assert result.loc["b", "status"] == "added"


def test_route_fare_map_groups_fare_ids() -> None:
    rules = pd.DataFrame({"fare_id": ["f1", "f2"], "route_id": ["R1", "R1"]})
    assert route_fare_map(rules)["R1"] == frozenset({"f1", "f2"})


# ---------------------------------------------------------------------------
# date ranges
# ---------------------------------------------------------------------------


def test_feed_date_range_prefers_feed_info() -> None:
    feed = {
        "feed_info": pd.DataFrame({"feed_start_date": ["20240101"], "feed_end_date": ["20240630"]})
    }
    start, end, source = feed_date_range(feed, override=None)
    assert (start, end, source) == ("20240101", "20240630", "feed_info")


def test_feed_date_range_override_wins() -> None:
    start, end, source = feed_date_range({}, override=("20240101", "20240201"))
    assert source == "override"
    assert (start, end) == ("20240101", "20240201")


def test_date_range_relationship_gap_and_overlap() -> None:
    before = ("20240101", "20240131", "feed_info")
    after_gap = ("20240301", "20240331", "feed_info")
    after_overlap = ("20240115", "20240215", "feed_info")
    assert "gap" in date_range_relationship(before, after_gap)
    assert "overlap" in date_range_relationship(before, after_overlap)


# ---------------------------------------------------------------------------
# summarize_change (headline priority)
# ---------------------------------------------------------------------------


def test_summarize_change_eliminated_headline() -> None:
    headline, summary = summarize_change({"status": "eliminated"})
    assert headline == "eliminated"
    assert "eliminated" in summary.lower()


def test_summarize_change_major_alignment_beats_minor_schedule() -> None:
    flags = {
        "status": "present",
        "alignment_change": "major",
        "schedule_change": "minor",
    }
    headline, _ = summarize_change(flags)
    assert headline == "major_alignment"


def test_summarize_change_unchanged_when_no_flags() -> None:
    headline, summary = summarize_change({"status": "present"})
    assert headline == "unchanged"
    assert summary == "No change detected"


def test_summarize_change_detail_parentheses_balanced() -> None:
    # Regression: the detail parenthetical must close even when other parts follow.
    flags = {
        "status": "present",
        "schedule_change": "minor",
        "schedule_detail": "span 0.5h->0.2h",
        "name_change": True,
        "name_detail": "long_name changed",
    }
    _, summary = summarize_change(flags)
    assert "Minor schedule change (span 0.5h->0.2h)" in summary
    assert summary.count("(") == summary.count(")")


# ---------------------------------------------------------------------------
# detect_splits_merges
# ---------------------------------------------------------------------------


def test_detect_splits_merges_flags_a_split() -> None:
    # Before R1 covers stops a..d; after, two routes split that territory.
    stops_before = {"R1": frozenset({"a", "b", "c", "d"})}
    stops_after = {
        "R1A": frozenset({"a", "b"}),  # covers 50% of R1
        "R1B": frozenset({"c", "d"}),  # covers 50% of R1
    }
    result = detect_splits_merges(stops_before, stops_after, min_coverage=0.60, min_share=0.30)
    split = result[result["kind"] == "split"]
    assert len(split) == 1
    assert split.iloc[0]["source_route_id"] == "R1"
    assert set(split.iloc[0]["target_route_ids"].split(";")) == {"R1A", "R1B"}


def test_detect_splits_merges_flags_a_merge() -> None:
    stops_before = {"R1": frozenset({"a", "b"}), "R2": frozenset({"c", "d"})}
    stops_after = {"MERGED": frozenset({"a", "b", "c", "d"})}
    result = detect_splits_merges(stops_before, stops_after, min_coverage=0.60, min_share=0.30)
    merge = result[result["kind"] == "merge"]
    assert len(merge) == 1
    assert merge.iloc[0]["source_route_id"] == "MERGED"


def test_detect_splits_merges_ignores_clean_rename() -> None:
    # A single successor covering the whole route is a rename/reroute, not a split.
    stops_before = {"R1": frozenset({"a", "b", "c", "d"})}
    stops_after = {"R1_NEW": frozenset({"a", "b", "c", "d"})}
    result = detect_splits_merges(stops_before, stops_after, min_coverage=0.60, min_share=0.30)
    assert result.empty


# ---------------------------------------------------------------------------
# compare_routes (end-to-end on synthetic feeds)
# ---------------------------------------------------------------------------


def _calendar_wk() -> pd.DataFrame:
    return pd.DataFrame([_cal("WK", "1111100", "20240101", "20240131")])


def _build_feed(
    routes: list[tuple[str, str]],
    trips: list[tuple[str, str, str]],
    stop_times: list[tuple[str, str, str, str]],
) -> dict:
    return {
        "routes": pd.DataFrame(routes, columns=["route_id", "route_short_name"]).assign(
            route_long_name="", route_type="3"
        ),
        "trips": pd.DataFrame(trips, columns=["route_id", "trip_id", "service_id"]).assign(
            direction_id="0", block_id=""
        ),
        "stop_times": _stop_times(stop_times),
        "calendar": _calendar_wk(),
    }


def test_compare_routes_end_to_end_counts() -> None:
    # before: R1 (stops a,b), R2 (stops e,f). after: R1 unchanged, R3 (stops g,h).
    before = _build_feed(
        routes=[("R1", "1"), ("R2", "2")],
        trips=[("R1", "t1", "WK"), ("R2", "t2", "WK")],
        stop_times=[
            ("t1", "a", "1", "08:00:00"),
            ("t1", "b", "2", "08:30:00"),
            ("t2", "e", "1", "09:00:00"),
            ("t2", "f", "2", "09:30:00"),
        ],
    )
    after = _build_feed(
        routes=[("R1", "1"), ("R3", "3")],
        trips=[("R1", "t1", "WK"), ("R3", "t3", "WK")],
        stop_times=[
            ("t1", "a", "1", "08:00:00"),
            ("t1", "b", "2", "08:30:00"),
            ("t3", "g", "1", "10:00:00"),
            ("t3", "h", "2", "10:30:00"),
        ],
    )
    results = compare_routes(before, after, weekday="tuesday", knobs=KNOBS)
    summary = results["summary"]
    assert summary.matched_count == 1  # R1
    assert summary.eliminated_count == 1  # R2
    assert summary.added_count == 1  # R3
    overview = results["overview"]
    assert len(overview) == 3
    # R1 should be unchanged (identical stops and schedule).
    r1 = overview[overview["route_id"] == "R1"].iloc[0]
    assert r1["headline_change"] == "unchanged"


def test_compare_routes_flags_major_alignment() -> None:
    before = _build_feed(
        routes=[("R1", "1")],
        trips=[("R1", "t1", "WK")],
        stop_times=[
            ("t1", "a", "1", "08:00:00"),
            ("t1", "b", "2", "08:30:00"),
            ("t1", "c", "3", "09:00:00"),
        ],
    )
    after = _build_feed(
        routes=[("R1", "1")],
        trips=[("R1", "t1", "WK")],
        stop_times=[
            ("t1", "x", "1", "08:00:00"),
            ("t1", "y", "2", "08:30:00"),
            ("t1", "z", "3", "09:00:00"),
        ],
    )
    results = compare_routes(before, after, weekday="tuesday", knobs=KNOBS)
    assert results["summary"].alignment_major_count == 1
    r1 = results["overview"].iloc[0]
    assert r1["headline_change"] == "major_alignment"


def test_correspondence_is_dataclass() -> None:
    # Guard the public type stays a frozen dataclass other tools can rely on.
    corr = Correspondence(matched=[], rekeyed={}, eliminated=[], added=[])
    assert corr.matched == []
