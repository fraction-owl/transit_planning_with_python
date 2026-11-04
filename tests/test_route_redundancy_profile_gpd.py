from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import scripts.service_coverage.route_redundancy_profile_gpd as rrp
from scripts.service_coverage.route_redundancy_profile_gpd import (
    build_boarding_events,
    build_route_buffers,
    build_schedule_service_ids,
    compute_area_overlap_pcts,
    compute_pair_stats,
    compute_service_areas,
    filter_platform_stops,
    match_temporal_events,
    resolve_route_pairs,
    run,
    select_temporal_dates,
)

# ---------------------------------------------------------------------------
# build_schedule_service_ids
# ---------------------------------------------------------------------------


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_id": ["wk", "sat"],
            "monday": ["1", "0"],
            "tuesday": ["1", "0"],
            "wednesday": ["1", "0"],
            "thursday": ["1", "0"],
            "friday": ["1", "0"],
            "saturday": ["0", "1"],
            "sunday": ["0", "0"],
            "start_date": ["20250101", "20250101"],
            "end_date": ["20251231", "20251231"],
        }
    )


def test_build_schedule_service_ids_classifies_by_real_dates() -> None:
    trips = pd.DataFrame({"trip_id": ["t1", "t2"], "service_id": ["wk", "sat"]})
    buckets = build_schedule_service_ids(_calendar(), None, trips)
    assert buckets == {"Weekday": {"wk"}, "Saturday": {"sat"}}


def test_build_schedule_service_ids_no_calendar_falls_back_to_all_service() -> None:
    trips = pd.DataFrame({"trip_id": ["t1", "t2"], "service_id": ["a", "b"]})
    buckets = build_schedule_service_ids(None, None, trips)
    assert buckets == {"All Service": {"a", "b"}}


def test_build_schedule_service_ids_excludes_orphan_service_ids() -> None:
    # "ghost" is referenced by trips but absent from the calendar files.
    trips = pd.DataFrame({"trip_id": ["t1", "t2"], "service_id": ["wk", "ghost"]})
    buckets = build_schedule_service_ids(_calendar(), None, trips)
    assert "ghost" not in {sid for ids in buckets.values() for sid in ids}


# ---------------------------------------------------------------------------
# compute_pair_stats
# ---------------------------------------------------------------------------


def _stops_frame() -> pd.DataFrame:
    # S1 (route A) and S2 (route B) are 100 m apart; S3 (route C) is 2 km away.
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "x": [0.0, 100.0, 2000.0],
            "y": [0.0, 0.0, 0.0],
        }
    )


def test_compute_pair_stats_pairs_routes_within_walk_only() -> None:
    serves = {"S1": {"A"}, "S2": {"B"}, "S3": {"C"}}
    pairs = compute_pair_stats(_stops_frame(), serves, radius_m=400.0)
    assert set(pairs) == {("A", "B"), ("B", "A")}
    assert pairs[("A", "B")].shared_stops == {"S1"}
    assert pairs[("A", "B")].min_distance_m == pytest.approx(100.0)


def test_compute_pair_stats_same_stop_counts_as_shared() -> None:
    stops = pd.DataFrame({"stop_id": ["S1"], "x": [0.0], "y": [0.0]})
    pairs = compute_pair_stats(stops, {"S1": {"A", "B"}}, radius_m=400.0)
    assert pairs[("A", "B")].min_distance_m == pytest.approx(0.0)
    assert pairs[("A", "B")].shared_stops == {"S1"}


def test_compute_pair_stats_single_route_has_no_pairs() -> None:
    stops = pd.DataFrame({"stop_id": ["S1", "S2"], "x": [0.0, 50.0], "y": [0.0, 0.0]})
    assert compute_pair_stats(stops, {"S1": {"A"}, "S2": {"A"}}, radius_m=400.0) == {}


def test_compute_pair_stats_timed_check_is_directional() -> None:
    serves = {"S1": {"A"}, "S2": {"B"}, "S3": {"C"}}
    arrivals = {
        ("S1", "A"): np.array([28800.0]),  # A arrives 08:00
        ("S2", "B"): np.array([29100.0]),  # B arrives 08:05
    }
    departures = {
        ("S1", "A"): np.array([28800.0]),
        ("S2", "B"): np.array([29100.0]),  # B departs 08:05
    }
    pairs = compute_pair_stats(
        _stops_frame(),
        serves,
        radius_m=400.0,
        arrivals=arrivals,
        departures=departures,
        check_times=True,
        walk_speed_mph=3.0,
        max_wait_minutes=30.0,
    )
    # A 08:00 -> B 08:05 is catchable; B 08:05 -> A 08:00 already left.
    # The reported wait excludes the 100 m walk (~75 s at 3 mph).
    assert pairs[("A", "B")].timed_feasible
    assert pairs[("A", "B")].min_wait_seconds == pytest.approx(
        300.0 - 100.0 / (3.0 * 1609.34 / 3600.0)
    )
    assert not pairs[("B", "A")].timed_feasible
    assert pairs[("B", "A")].min_wait_seconds is None


def test_compute_pair_stats_max_wait_bounds_feasibility() -> None:
    serves = {"S1": {"A"}, "S2": {"B"}}
    arrivals = {("S1", "A"): np.array([28800.0])}
    departures = {("S2", "B"): np.array([29100.0])}
    pairs = compute_pair_stats(
        _stops_frame(),
        serves,
        radius_m=400.0,
        arrivals=arrivals,
        departures=departures,
        check_times=True,
        max_wait_minutes=2.0,  # the 5-minute-later departure is now too late
    )
    # The pair still exists spatially, but it is not a timed alternative.
    assert not pairs[("A", "B")].timed_feasible


# ---------------------------------------------------------------------------
# compute_service_areas
# ---------------------------------------------------------------------------


def test_compute_service_areas_splits_solo_and_shared() -> None:
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1", "S2"]},
        geometry=[Point(0.0, 0.0), Point(10000.0, 0.0)],
    )
    # A and B both serve S1 (identical buffers); C is alone at S2.
    buffers = build_route_buffers(stops, {"A": {"S1"}, "B": {"S1"}, "C": {"S2"}}, radius_m=400.0)
    areas = compute_service_areas(buffers)
    circle_sqmi = np.pi * 400.0**2 / rrp._SQM_PER_SQMI

    total_a, solo_a, shared_a = areas["A"]
    assert total_a == pytest.approx(circle_sqmi, rel=0.01)
    assert solo_a == pytest.approx(0.0, abs=1e-9)
    assert shared_a == pytest.approx(total_a)

    total_c, solo_c, shared_c = areas["C"]
    assert solo_c == pytest.approx(total_c)
    assert shared_c == pytest.approx(0.0, abs=1e-9)


def test_compute_service_areas_route_with_no_located_stops_is_zero() -> None:
    stops = gpd.GeoDataFrame({"stop_id": ["S1"]}, geometry=[Point(0.0, 0.0)])
    areas = compute_service_areas(build_route_buffers(stops, {"A": {"MISSING"}}, radius_m=400.0))
    assert areas["A"] == (0.0, 0.0, 0.0)


def test_compute_area_overlap_pcts_mutual_and_disjoint() -> None:
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1", "S2"]},
        geometry=[Point(0.0, 0.0), Point(10000.0, 0.0)],
    )
    buffers = build_route_buffers(stops, {"A": {"S1"}, "B": {"S1"}, "C": {"S2"}}, radius_m=400.0)
    pcts = compute_area_overlap_pcts(buffers)

    # Identical buffers overlap 100% both ways; C intersects nobody.
    assert pcts[("A", "B")] == pytest.approx(100.0)
    assert pcts[("B", "A")] == pytest.approx(100.0)
    assert ("A", "C") not in pcts
    assert ("C", "A") not in pcts


def test_compute_service_areas_declared_pair_is_not_shared() -> None:
    stops = gpd.GeoDataFrame({"stop_id": ["S1"]}, geometry=[Point(0.0, 0.0)])
    buffers = build_route_buffers(stops, {"A": {"S1"}, "B": {"S1"}}, radius_m=400.0)
    areas = compute_service_areas(buffers, declared_pairs={frozenset(("A", "B"))})

    # B is A's declared pair, so B's identical buffer no longer makes A's
    # coverage "shared": everything stays solo.
    total_a, solo_a, shared_a = areas["A"]
    assert solo_a == pytest.approx(total_a)
    assert shared_a == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# resolve_route_pairs
# ---------------------------------------------------------------------------


def _routes_table() -> pd.DataFrame:
    return pd.DataFrame({"route_id": ["A", "B", "C"], "route_short_name": ["99A", "99B", "10"]})


def test_resolve_route_pairs_matches_short_names_and_route_ids() -> None:
    declared = resolve_route_pairs([("99A", "99B"), ("A", "10")], _routes_table())
    assert declared == {frozenset(("A", "B")), frozenset(("A", "C"))}


def test_resolve_route_pairs_ignores_unmatched_tokens() -> None:
    assert resolve_route_pairs([("99A", "NOPE")], _routes_table()) == set()


def test_resolve_route_pairs_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError):
        resolve_route_pairs([("99A", "99B", "10")], _routes_table())


# ---------------------------------------------------------------------------
# filter_platform_stops
# ---------------------------------------------------------------------------


def test_filter_platform_stops_drops_stations() -> None:
    stops = pd.DataFrame({"stop_id": ["S1", "S2", "S3"], "location_type": ["", "0", "1"]})
    assert list(filter_platform_stops(stops)["stop_id"]) == ["S1", "S2"]


def test_filter_platform_stops_without_column_is_noop() -> None:
    stops = pd.DataFrame({"stop_id": ["S1"]})
    assert len(filter_platform_stops(stops)) == 1


# ---------------------------------------------------------------------------
# temporal coverage
# ---------------------------------------------------------------------------


def _stop_times(rows: list[tuple[str, str, int, str, str]]) -> pd.DataFrame:
    columns = ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]
    return pd.DataFrame(rows, columns=columns).astype(str)


def _trips(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["trip_id", "route_id", "service_id"])


def test_build_boarding_events_drops_final_stop_and_nonregular_pickup() -> None:
    stop_times = _stop_times(
        [
            ("T1", "S1", 1, "08:00:00", "08:00:00"),
            ("T1", "S2", 2, "08:05:00", "08:05:00"),
            ("T1", "S3", 3, "08:10:00", "08:10:00"),
        ]
    )
    stop_times["pickup_type"] = ["0", "1", "0"]
    events = build_boarding_events(stop_times, _trips([("T1", "A", "wk")]))
    # S2 has no regular pickup and S3 ends the trip, so only S1 is a boarding.
    assert list(events["stop_id"]) == ["S1"]
    assert events["departure_sec"].iloc[0] == pytest.approx(8 * 3600)


def test_build_boarding_events_interpolates_untimed_stops() -> None:
    stop_times = _stop_times(
        [
            ("T1", "S1", 1, "08:00:00", "08:00:00"),
            ("T1", "S2", 2, "", ""),
            ("T1", "S3", 3, "", ""),
            ("T1", "S4", 4, "08:09:00", "08:09:00"),
            ("T1", "S5", 5, "08:20:00", "08:20:00"),
        ]
    )
    events = build_boarding_events(stop_times, _trips([("T1", "A", "wk")]))
    offsets = events.set_index("stop_id")["departure_sec"] - 8 * 3600
    # Evenly spaced by stop order between the 08:00 and 08:09 timed stops.
    assert offsets["S2"] == pytest.approx(180.0)
    assert offsets["S3"] == pytest.approx(360.0)

    stop_times["shape_dist_traveled"] = ["0", "1", "2", "9", "12"]
    events = build_boarding_events(stop_times, _trips([("T1", "A", "wk")]))
    offsets = events.set_index("stop_id")["departure_sec"] - 8 * 3600
    # With distances, S2 sits 1/9 and S3 2/9 of the way along the timed span.
    assert offsets["S2"] == pytest.approx(60.0)
    assert offsets["S3"] == pytest.approx(120.0)


def _boardings(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["route_id", "stop_id", "departure_sec"])
    frame["trip_id"] = [f"t{i}" for i in range(len(frame))]
    frame["absolute_sec"] = frame["departure_sec"]
    return frame


def test_match_temporal_events_finds_later_and_earlier_alternatives() -> None:
    # A boards at S1 at 08:00; B leaves S2 (100 m away) at 07:55 and 08:20.
    events = _boardings([("A", "S1", 28800.0), ("B", "S2", 28500.0), ("B", "S2", 30000.0)])
    matched = match_temporal_events(events, events, _stops_frame(), 400.0, 3.0, set())
    a = matched.loc[matched["route_id"] == "A"].iloc[0]
    walk_min = 100.0 / (3.0 * 1609.344 / 3600.0) / 60.0
    assert bool(a["geographic_neighbor"])
    assert a["alternative_route_id"] == "B"
    assert a["min_wait_after_walk_min"] == pytest.approx(20.0 - walk_min)
    assert a["earlier_alternative_route_id"] == "B"
    assert a["min_earlier_shift_min"] == pytest.approx(5.0 + walk_min)


def test_match_temporal_events_ignores_declared_pair_and_far_routes() -> None:
    # B is A's declared other direction; C runs 2 km away.
    events = _boardings([("A", "S1", 28800.0), ("B", "S2", 28900.0), ("C", "S3", 28900.0)])
    matched = match_temporal_events(
        events, events, _stops_frame(), 400.0, 3.0, {frozenset(("A", "B"))}
    )
    a = matched.loc[matched["route_id"] == "A"].iloc[0]
    assert not bool(a["geographic_neighbor"])
    assert np.isnan(a["min_wait_after_walk_min"])
    assert np.isnan(a["min_earlier_shift_min"])


def test_select_temporal_dates_picks_typical_non_holiday_dates() -> None:
    active = rrp.expand_service_active_dates(_calendar(), None)
    dates = select_temporal_dates(active, {"Weekday": {"wk"}, "Saturday": {"sat"}}, {})
    assert dates["Weekday"].weekday() < 5
    assert dates["Weekday"] not in rrp.federal_holidays_observed(2025)
    assert dates["Saturday"].weekday() == 5


def test_select_temporal_dates_drops_explicit_date_without_service() -> None:
    active = rrp.expand_service_active_dates(_calendar(), None)
    dates = select_temporal_dates(
        active, {"Weekday": {"wk"}}, {"Weekday": "20250106", "Sunday": "20250105"}
    )
    # 2025-01-05 is a Sunday and the calendar has no Sunday service.
    assert dates == {"Weekday": dt.date(2025, 1, 6)}


# ---------------------------------------------------------------------------
# end-to-end: run() on a tiny synthetic feed
# ---------------------------------------------------------------------------


def _write_feed(folder: Path) -> None:
    """Two-route feed: A and B run parallel ~43 m apart on weekdays; A alone Saturdays."""
    folder.mkdir()
    # S1/S2 are ~43 m apart, S3/S4 likewise; the pairs are ~1.7 km from each other.
    pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3", "S4"],
            "stop_name": ["First & Main", "Main & First", "West End", "West End Far Side"],
            "stop_lat": [38.9, 38.9, 38.9, 38.9],
            "stop_lon": [-77.0, -77.0005, -77.02, -77.0205],
        }
    ).to_csv(folder / "stops.txt", index=False)
    pd.DataFrame(
        {
            "route_id": ["A", "B"],
            "route_short_name": ["10", "20"],
            "route_long_name": ["Main St", "First Ave"],
        }
    ).to_csv(folder / "routes.txt", index=False)
    pd.DataFrame(
        {
            "trip_id": ["A_wk", "B_wk", "A_sat"],
            "route_id": ["A", "B", "A"],
            "service_id": ["wk", "wk", "sat"],
        }
    ).to_csv(folder / "trips.txt", index=False)
    pd.DataFrame(
        {
            "trip_id": ["A_wk", "A_wk", "B_wk", "B_wk", "A_sat", "A_sat"],
            "stop_id": ["S1", "S3", "S2", "S4", "S1", "S3"],
            "stop_sequence": [1, 2, 1, 2, 1, 2],
            "arrival_time": [
                "08:00:00",
                "08:10:00",
                "08:05:00",
                "08:15:00",
                "09:00:00",
                "09:10:00",
            ],
            "departure_time": [
                "08:00:00",
                "08:10:00",
                "08:05:00",
                "08:15:00",
                "09:00:00",
                "09:10:00",
            ],
            "timepoint": [1, 1, 1, 0, 1, 1],
        }
    ).to_csv(folder / "stop_times.txt", index=False)
    _calendar().to_csv(folder / "calendar.txt", index=False)


def test_run_end_to_end(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    out = tmp_path / "out"
    _write_feed(feed)

    summary = run(gtfs_path=str(feed), output_dir=out)

    # One row per (schedule, route): Weekday A + B, Saturday A. Weekday first.
    assert list(summary["schedule"]) == ["Weekday", "Weekday", "Saturday"]
    assert (out / rrp.SUMMARY_FILENAME).exists()
    assert (out / rrp.DETAIL_FILENAME).exists()
    assert (out / f"{Path(rrp.SUMMARY_FILENAME).stem}_runlog.txt").exists()

    wk = summary.loc[summary["schedule"] == "Weekday"].set_index("route_id")
    a, b = wk.loc["A"], wk.loc["B"]

    # Angle 1: every stop of A has a B stop across the street, and vice versa.
    assert int(a["n_stops"]) == 2
    assert int(a["n_shared_stops"]) == 2
    assert int(a["n_solo_stops"]) == 0
    assert a["pct_stops_shared"] == pytest.approx(100.0)

    # Angle 2: B is within walking distance of A; only A->B works in time
    # (B departs 5 min after A arrives; A has already left when B arrives).
    assert int(a["n_routes_within_walk"]) == 1
    assert a["routes_within_walk"] == "20"
    assert int(a["n_timed_alternative_routes"]) == 1
    assert a["timed_alternative_routes"] == "20"
    assert int(b["n_timed_alternative_routes"]) == 0

    # Angle 3: A has timepoints at S1 and S3, but B's only timepoint is S2 —
    # so A shares one timepoint (S1~S2) and keeps one solo (S3).
    assert int(a["n_timepoints"]) == 2
    assert int(a["n_shared_timepoints"]) == 1
    assert int(a["n_solo_timepoints"]) == 1
    assert a["pct_timepoints_shared"] == pytest.approx(50.0)
    assert a["timepoint_partner_routes"] == "20"
    assert int(b["n_timepoints"]) == 1
    assert int(b["n_shared_timepoints"]) == 1

    # Angle 4: two disjoint 0.25-mi stop circles ~= 0.39 sq mi; nearly all of
    # it is shared with B's parallel circles 43 m away.
    assert a["service_area_sqmi"] == pytest.approx(0.39, abs=0.02)
    assert 0.0 < a["solo_area_sqmi"] < 0.05
    assert a["shared_area_sqmi"] > 0.3

    # Angle 5: A and B look like a couplet (~93% mutual overlap). With no
    # declaration they are only *suggested* — metrics above are unchanged.
    assert int(a["n_paired_routes"]) == 0
    assert int(a["n_suggested_pairs"]) == 1
    assert a["suggested_pairs"] == "20"
    assert b["suggested_pairs"] == "10"

    # Saturday: A runs alone, so nothing is shared on any angle.
    sat = summary.loc[summary["schedule"] == "Saturday"].iloc[0]
    assert sat["route_id"] == "A"
    assert int(sat["n_shared_stops"]) == 0
    assert int(sat["n_routes_within_walk"]) == 0
    assert int(sat["n_shared_timepoints"]) == 0
    assert sat["solo_area_sqmi"] == pytest.approx(sat["service_area_sqmi"])
    assert sat["pct_area_shared"] == pytest.approx(0.0)
    assert int(sat["n_paired_routes"]) == 0

    detail = pd.read_csv(out / rrp.DETAIL_FILENAME)
    a_to_b = detail.loc[(detail["schedule"] == "Weekday") & (detail["route_id"] == "A")].iloc[0]
    assert a_to_b["partner_route_id"] == "B"
    assert int(a_to_b["n_shared_stops"]) == 2
    assert int(a_to_b["n_shared_timepoints"]) == 1
    assert 100 < a_to_b["nearest_walk_distance_ft"] < 200
    assert str(a_to_b["timed_alternative"]) == "True"
    # B departs 5 min after A arrives; ~0.5 min of that is the walk.
    assert a_to_b["min_alternative_wait_min"] == pytest.approx(4.5)
    assert a_to_b["area_overlap_pct"] > 90.0
    assert str(a_to_b["paired"]) == "False"
    assert str(a_to_b["suggested_pair"]) == "True"


def test_run_declared_pair_excluded_from_metrics(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    out = tmp_path / "out"
    _write_feed(feed)

    summary = run(gtfs_path=str(feed), output_dir=out, route_pairs=[("10", "20")])
    wk = summary.loc[summary["schedule"] == "Weekday"].set_index("route_id")
    a = wk.loc["A"]

    # B is declared as A's other direction, so nothing counts as shared:
    # every stop, timepoint, and square mile of A is solo again.
    assert int(a["n_shared_stops"]) == 0
    assert int(a["n_routes_within_walk"]) == 0
    assert int(a["n_timed_alternative_routes"]) == 0
    assert int(a["n_shared_timepoints"]) == 0
    assert a["solo_area_sqmi"] == pytest.approx(a["service_area_sqmi"])
    assert a["paired_routes"] == "20"
    assert int(a["n_suggested_pairs"]) == 0

    # The pair itself stays visible in the partner detail, flagged as paired.
    detail = pd.read_csv(out / rrp.DETAIL_FILENAME)
    a_to_b = detail.loc[(detail["schedule"] == "Weekday") & (detail["route_id"] == "A")].iloc[0]
    assert a_to_b["partner_route_id"] == "B"
    assert str(a_to_b["paired"]) == "True"
    assert str(a_to_b["suggested_pair"]) == "False"


def test_run_service_label_filter(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    out = tmp_path / "out"
    _write_feed(feed)
    summary = run(gtfs_path=str(feed), output_dir=out, service_labels=["saturday"])
    assert list(summary["schedule"].unique()) == ["Saturday"]


def test_run_unknown_service_label_raises(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    _write_feed(feed)
    with pytest.raises(ValueError):
        run(gtfs_path=str(feed), output_dir=tmp_path / "out", service_labels=["someday"])


def test_run_temporal_coverage_outputs(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    out = tmp_path / "out"
    _write_feed(feed)
    run(gtfs_path=str(feed), output_dir=out)

    daily = pd.read_csv(out / rrp.TEMPORAL_SUMMARY_FILENAME, dtype={"route_id": str})
    wk = daily.loc[(daily["schedule"] == "Weekday") & (daily["max_wait_after_walk_min"] == 30)]
    wk = wk.set_index("route_id")
    # A's 08:00 boarding at S1 can catch B at 08:05 across the street.
    assert int(wk.loc["A", "alternative_available_events"]) == 1
    assert int(wk.loc["A", "earlier_only_alternative_events"]) == 0
    # B's 08:05 boarding has no later alternative, but leaving ~5.5 min sooner
    # catches A at 08:00, within the default 10-minute earlier allowance.
    assert int(wk.loc["B", "alternative_available_events"]) == 1
    assert int(wk.loc["B", "earlier_only_alternative_events"]) == 1
    # Saturday: A runs alone, so its boarding is a geographic gap.
    sat = daily.loc[(daily["schedule"] == "Saturday") & (daily["route_id"] == "A")].iloc[0]
    assert int(sat["geographic_gap_events"]) == 1

    for name in (
        rrp.TEMPORAL_PERIODS_FILENAME,
        rrp.TEMPORAL_HOURLY_FILENAME,
        rrp.STOP_TEMPORAL_FILENAME,
        "temporal_events_Weekday.csv.gz",
        "temporal_events_Saturday.csv.gz",
    ):
        assert (out / name).exists()


def test_run_temporal_without_earlier_allowance(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    out = tmp_path / "out"
    _write_feed(feed)
    run(gtfs_path=str(feed), output_dir=out, temporal_earlier_minutes=0.0)
    daily = pd.read_csv(out / rrp.TEMPORAL_SUMMARY_FILENAME, dtype={"route_id": str})
    b = daily.loc[(daily["schedule"] == "Weekday") & (daily["route_id"] == "B")].iloc[0]
    # Later departures only: A stops nearby but has already left, a timing gap.
    assert int(b["timing_gap_events"]) == 1
    assert int(b["alternative_available_events"]) == 0


def test_run_skips_temporal_for_frequency_feed(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    out = tmp_path / "out"
    _write_feed(feed)
    pd.DataFrame(
        {
            "trip_id": ["A_wk"],
            "start_time": ["06:00:00"],
            "end_time": ["09:00:00"],
            "headway_secs": ["600"],
        }
    ).to_csv(feed / "frequencies.txt", index=False)
    summary = run(gtfs_path=str(feed), output_dir=out)
    # The rest of the profile still runs and is traced in the run log.
    assert not summary.empty
    assert not (out / rrp.TEMPORAL_SUMMARY_FILENAME).exists()
    runlog = (out / f"{Path(rrp.SUMMARY_FILENAME).stem}_runlog.txt").read_text()
    assert "Temporal coverage:    skipped" in runlog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = rrp.build_arg_parser().parse_args([])
    assert args.walk_distance == pytest.approx(0.25)
    assert args.max_wait_minutes == pytest.approx(30.0)
    assert not args.no_time_check
    assert args.temporal_dates == []
    assert args.temporal_earlier_minutes == pytest.approx(10.0)


def test_parse_args_rejects_unknown_tokens() -> None:
    with pytest.raises(SystemExit) as excinfo:
        rrp.build_arg_parser().parse_args(["--input-dir", "somewhere"])
    assert excinfo.value.code == 2


def test_main_placeholder_paths_return_2() -> None:
    assert rrp.main([]) == 2


def test_main_rejects_bad_temporal_date(tmp_path: Path) -> None:
    feed = tmp_path / "gtfs"
    _write_feed(feed)
    argv = ["--gtfs", str(feed), "--output-dir", str(tmp_path / "out")]
    assert rrp.main([*argv, "--temporal-dates", "Weekday:20251340"]) == 2
