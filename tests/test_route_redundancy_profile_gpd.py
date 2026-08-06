from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import scripts.service_coverage.route_redundancy_profile_gpd as rrp
from scripts.service_coverage.route_redundancy_profile_gpd import (
    build_schedule_service_ids,
    compute_pair_stats,
    compute_service_areas,
    filter_platform_stops,
    run,
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
    assert pairs[("A", "B")].timed_feasible
    assert pairs[("A", "B")].min_wait_seconds == pytest.approx(300.0)
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
    areas = compute_service_areas(stops, {"A": {"S1"}, "B": {"S1"}, "C": {"S2"}}, radius_m=400.0)
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
    areas = compute_service_areas(stops, {"A": {"MISSING"}}, radius_m=400.0)
    assert areas["A"] == (0.0, 0.0, 0.0)


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

    # Saturday: A runs alone, so nothing is shared on any angle.
    sat = summary.loc[summary["schedule"] == "Saturday"].iloc[0]
    assert sat["route_id"] == "A"
    assert int(sat["n_shared_stops"]) == 0
    assert int(sat["n_routes_within_walk"]) == 0
    assert int(sat["n_shared_timepoints"]) == 0
    assert sat["solo_area_sqmi"] == pytest.approx(sat["service_area_sqmi"])
    assert sat["pct_area_shared"] == pytest.approx(0.0)

    detail = pd.read_csv(out / rrp.DETAIL_FILENAME)
    a_to_b = detail.loc[(detail["schedule"] == "Weekday") & (detail["route_id"] == "A")].iloc[0]
    assert a_to_b["partner_route_id"] == "B"
    assert int(a_to_b["n_shared_stops"]) == 2
    assert int(a_to_b["n_shared_timepoints"]) == 1
    assert 100 < a_to_b["nearest_walk_distance_ft"] < 200
    assert str(a_to_b["timed_alternative"]) == "True"
    assert a_to_b["min_alternative_wait_min"] == pytest.approx(5.0)


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = rrp.build_arg_parser().parse_args([])
    assert args.walk_distance == pytest.approx(0.25)
    assert args.max_wait_minutes == pytest.approx(30.0)
    assert not args.no_time_check


def test_parse_args_rejects_unknown_tokens() -> None:
    with pytest.raises(SystemExit) as excinfo:
        rrp.build_arg_parser().parse_args(["--input-dir", "somewhere"])
    assert excinfo.value.code == 2


def test_main_placeholder_paths_return_2() -> None:
    assert rrp.main([]) == 2
