"""Tests for otp_by_stop_flagger using the repo TIDES fixtures plus synthetic frames."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import otp_by_stop_flagger as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


@pytest.fixture()
def prepared() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(candidates, scored, in-service trips) built from the repo fixtures."""
    sv = target.load_stop_visits(STOP_VISITS)
    trips = target.add_pattern_key(
        target.filter_in_service(target.load_trips_performed(TRIPS_PERFORMED))
    )
    deviated = target.compute_stop_deviations(target.join_trip_attributes(sv, trips))
    candidates = target.filter_candidate_visits(deviated, True)
    scored = candidates.pipe(target.filter_for_otp, True).pipe(
        target.classify_otp, target.EARLY_MIN, target.LATE_MIN
    )
    return candidates, scored, trips


def _make_detail(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal evaluated detail frame from per-row dicts."""
    defaults = {
        "visits_emitted": 0,
        "visits_skipped": 0,
        "visits_missing_actual": 0,
        "visits_missing_schedule": 0,
        "early": 0,
        "late": 0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_add_pattern_key_prefers_pattern_id() -> None:
    """pattern_id is used when present; blanks fall back to route|direction."""
    trips = pd.DataFrame(
        {
            "trip_id_performed": ["T1", "T2"],
            "route_id": ["101", "101"],
            "direction_id": ["0", "1"],
            "pattern_id": ["P1", None],
        }
    )
    out = target.add_pattern_key(trips)
    assert out["pattern_key"].tolist() == ["P1", "101 | 1"]


def test_add_pattern_key_falls_back_without_pattern_id() -> None:
    """With no pattern_id column at all, route|direction becomes the key."""
    trips = pd.DataFrame(
        {
            "trip_id_performed": ["T1"],
            "route_id": ["101"],
            "direction_id": ["0"],
        }
    )
    out = target.add_pattern_key(trips)
    assert out["pattern_key"].tolist() == ["101 | 0"]


def test_expected_trips_sums_patterns_serving_the_stop() -> None:
    """A stop's expected trips are the trips of every pattern that serves it."""
    membership = pd.DataFrame(
        {
            "pattern_key": ["P1", "P1", "P2"],
            "route_id": ["A", "A", "B"],
            "stop_id": ["S1", "S2", "S1"],
        }
    )
    pattern_trips = pd.DataFrame(
        {
            "pattern_key": ["P1", "P2"],
            "route_id": ["A", "B"],
            "n_trips": [10, 5],
        }
    )
    expected = target.build_expected_trips(membership, pattern_trips)
    lookup = expected.set_index(["stop_id", "route_id"])["expected_trips"]
    assert lookup[("S1", "A")] == 10
    assert lookup[("S1", "B")] == 5
    assert lookup[("S2", "A")] == 10
    assert ("S2", "B") not in lookup.index


def test_detail_splits_visit_failure_causes() -> None:
    """Skipped, missing-actual, and missing-schedule visits are told apart."""
    ts = pd.Timestamp("2025-01-02T06:00:00")
    base = {
        "trip_id_performed": "T1",
        "pattern_key": "P1",
        "route_id": "A",
        "stop_id": "S1",
    }
    candidates = target.compute_stop_deviations(
        pd.DataFrame(
            [
                {
                    **base,
                    "schedule_relationship": "Scheduled",
                    "schedule_departure_time": ts,
                    "schedule_arrival_time": ts,
                    "actual_departure_time": ts,
                    "actual_arrival_time": ts,
                },
                {
                    **base,
                    "schedule_relationship": "Scheduled",
                    "schedule_departure_time": ts,
                    "schedule_arrival_time": ts,
                    "actual_departure_time": pd.NaT,
                    "actual_arrival_time": pd.NaT,
                },
                {
                    **base,
                    "schedule_relationship": "Scheduled",
                    "schedule_departure_time": pd.NaT,
                    "schedule_arrival_time": pd.NaT,
                    "actual_departure_time": ts,
                    "actual_arrival_time": ts,
                },
                {
                    **base,
                    "schedule_relationship": "Skipped",
                    "schedule_departure_time": ts,
                    "schedule_arrival_time": ts,
                    "actual_departure_time": pd.NaT,
                    "actual_arrival_time": pd.NaT,
                },
            ]
        )
    )
    scored = candidates.pipe(target.filter_for_otp, False).pipe(target.classify_otp)
    trips = pd.DataFrame({"trip_id_performed": ["T1"], "pattern_key": ["P1"], "route_id": ["A"]})
    detail = target.build_stop_route_detail(candidates, scored, trips)
    row = detail.iloc[0]
    assert row["visits_emitted"] == 4
    assert row["visits_skipped"] == 1
    assert row["visits_missing_actual"] == 1
    assert row["visits_missing_schedule"] == 1
    assert row["evaluated"] == 1
    assert row["expected_trips"] == 1
    assert row["observed_trips"] == 1


def test_route_baselines_pool_the_route() -> None:
    """Baselines are route-pooled ratios and gaps are stop minus baseline."""
    detail = _make_detail(
        [
            {
                "stop_id": "S1",
                "route_id": "A",
                "expected_trips": 100,
                "observed_trips": 100,
                "evaluated": 100,
                "on_time": 90,
            },
            {
                "stop_id": "S2",
                "route_id": "A",
                "expected_trips": 100,
                "observed_trips": 50,
                "evaluated": 50,
                "on_time": 30,
            },
        ]
    )
    detail["pct_trips_observed"] = detail["observed_trips"] / detail["expected_trips"] * 100
    detail["pct_on_time"] = detail["on_time"] / detail["evaluated"] * 100
    out = target.attach_route_baselines(detail)
    assert out["route_pct_trips_observed"].eq(75.0).all()  # 150 / 200
    assert out["route_pct_on_time"].eq(80.0).all()  # 120 / 150
    s2 = out.loc[out["stop_id"] == "S2"].iloc[0]
    assert s2["coverage_gap"] == pytest.approx(-25.0)
    assert s2["otp_gap"] == pytest.approx(-20.0)


def test_flags_require_multiple_routes(tmp_path: Path) -> None:
    """A stop bad on two routes is flagged; bad on one route is not."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
        min_routes_flagged=2,
        min_expected_trips=10,
        min_scored_visits=10,
    )
    # Routes A and B each have one healthy stop and share problem stop SX;
    # route C has a healthy stop and its own problem stop SY (single route).
    rows = []
    for route, good_stop in (("A", "S1"), ("B", "S2"), ("C", "S3")):
        rows.append(
            {
                "stop_id": good_stop,
                "route_id": route,
                "expected_trips": 100,
                "observed_trips": 98,
                "evaluated": 98,
                "on_time": 95,
            }
        )
    for route in ("A", "B"):
        rows.append(
            {
                "stop_id": "SX",
                "route_id": route,
                "expected_trips": 100,
                "observed_trips": 40,
                "evaluated": 40,
                "on_time": 10,
            }
        )
    rows.append(
        {
            "stop_id": "SY",
            "route_id": "C",
            "expected_trips": 100,
            "observed_trips": 40,
            "evaluated": 40,
            "on_time": 10,
        }
    )
    detail = _make_detail(rows)
    detail["pct_trips_observed"] = detail["observed_trips"] / detail["expected_trips"] * 100
    detail["pct_on_time"] = detail["on_time"] / detail["evaluated"] * 100

    evaluated = target.evaluate_route_level_flags(target.attach_route_baselines(detail), cfg)
    summary = target.build_stop_summary(evaluated, cfg.min_routes_flagged).set_index("stop_id")

    assert summary.loc["SX", "flag_reason"] == "low_coverage+poor_otp"
    assert bool(summary.loc["SX", "flag_low_coverage"])
    assert bool(summary.loc["SX", "flag_poor_otp"])
    assert summary.loc["SY", "flag_reason"] == ""  # only one route agrees
    assert summary.loc["S1", "flag_reason"] == ""
    # Flagged stops sort first.
    assert summary.index[0] == "SX"


def test_absolute_floor_protects_good_stops() -> None:
    """A big gap alone does not flag a stop that is still above the floor."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=Path("."),
        min_expected_trips=10,
        min_scored_visits=10,
        otp_abs_flag_pct=75.0,
    )
    rows = [
        {
            "stop_id": s,
            "route_id": "A",
            "expected_trips": 100,
            "observed_trips": 100,
            "evaluated": 100,
            "on_time": 100,
        }
        for s in ("S1", "S3", "S4")
    ]
    # 15 points below the 95% baseline, but still 80% on-time: above the floor.
    rows.append(
        {
            "stop_id": "S2",
            "route_id": "A",
            "expected_trips": 100,
            "observed_trips": 100,
            "evaluated": 100,
            "on_time": 80,
        }
    )
    detail = _make_detail(rows)
    detail["pct_trips_observed"] = 100.0
    detail["pct_on_time"] = detail["on_time"] / detail["evaluated"] * 100
    evaluated = target.evaluate_route_level_flags(target.attach_route_baselines(detail), cfg)
    s2 = evaluated.loc[evaluated["stop_id"] == "S2"].iloc[0]
    assert s2["otp_gap"] <= -15.0
    assert not s2["poor_otp_route"]  # 80% >= the 75% absolute floor
    # And a small cell is never judged, however bad it looks.
    cfg_small = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=Path("."),
        min_expected_trips=1000,
        min_scored_visits=1000,
    )
    evaluated_small = target.evaluate_route_level_flags(
        target.attach_route_baselines(detail), cfg_small
    )
    assert not evaluated_small["coverage_evaluable"].any()
    assert not evaluated_small["poor_otp_route"].any()


def test_observed_never_exceeds_expected_on_fixtures(
    prepared: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Pattern-inferred expected trips bound the observed trips at every stop."""
    candidates, scored, trips = prepared
    detail = target.build_stop_route_detail(candidates, scored, trips)
    assert not detail.empty
    assert (detail["observed_trips"] <= detail["expected_trips"]).all()
    assert detail["pct_trips_observed"].dropna().between(0, 100).all()
    assert detail["pct_on_time"].dropna().between(0, 100).all()


def test_run_writes_tables_and_runlog(tmp_path: Path) -> None:
    """End-to-end run produces the flags table, the detail table, and a run log."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
    )
    summary = target.run(cfg)
    assert not summary.empty
    assert {"stop_id", "flag_reason", "pct_trips_observed", "pct_on_time"} <= set(summary.columns)

    assert (tmp_path / target.STOP_FLAGS_FILENAME).exists()
    assert (tmp_path / target.STOP_ROUTE_DETAIL_FILENAME).exists()
    runlog = tmp_path / "otp_by_stop_flagger_runlog.txt"
    assert runlog.exists()
    text = runlog.read_text(encoding="utf-8")
    assert "CONFIGURATION (verbatim from source)" in text
    assert "MIN_ROUTES_FLAGGED" in text
