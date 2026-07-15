"""Tests for otp_monthly_panel using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import otp_monthly_panel as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


@pytest.fixture()
def scored() -> pd.DataFrame:
    """Joined, deviated, filtered, classified, month-tagged stop visits."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(sv, tp)
    return (
        joined.pipe(target.compute_stop_deviations)
        .pipe(target.filter_for_otp, True)
        .pipe(target.classify_otp, target.EARLY_MIN, target.LATE_MIN)
        .pipe(target.add_month)
    )


def test_join_brings_route_attributes() -> None:
    """The join attaches route/direction/service-type onto stop visits."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(sv, tp)
    for col in ("route_id", "direction_id", "route_type_agency"):
        assert col in joined.columns
    # Every joined row should carry a route id (inner join on performed trips).
    assert joined["route_id"].notna().all()
    assert set(joined["route_id"].unique()) <= {"101", "202", "303"}


def test_compute_stop_deviations_sign() -> None:
    """Deviation is actual minus scheduled departure, in minutes."""
    df = pd.DataFrame(
        {
            "schedule_departure_time": pd.to_datetime(["2025-01-02T06:00:00"]),
            "schedule_arrival_time": pd.to_datetime(["2025-01-02T06:00:00"]),
            "actual_departure_time": pd.to_datetime(["2025-01-02T06:03:00"]),
            "actual_arrival_time": pd.to_datetime(["2025-01-02T06:03:00"]),
        }
    )
    out = target.compute_stop_deviations(df)
    assert out["dev_min"].iloc[0] == pytest.approx(3.0)


def test_filter_for_otp_drops_skipped_and_nontimepoint(scored: pd.DataFrame) -> None:
    """Only Scheduled timepoint visits with a finite deviation survive."""
    assert (scored["timepoint"].astype(str).str.upper() == "TRUE").all()
    assert (scored["schedule_relationship"] == "Scheduled").all()
    assert scored["dev_min"].notna().all()


def test_summarize_unscorable_splits_reasons() -> None:
    """Missing-actual (AVL dropout) and missing-schedule rows are told apart."""
    ts = pd.Timestamp("2025-01-02T06:00:00")
    df = pd.DataFrame(
        {
            "timepoint": ["TRUE", "TRUE", "TRUE", "FALSE"],
            "schedule_relationship": ["Scheduled"] * 4,
            "schedule_departure_time": [ts, ts, pd.NaT, ts],
            "schedule_arrival_time": [ts, ts, pd.NaT, ts],
            "actual_departure_time": [ts, pd.NaT, ts, pd.NaT],
            "actual_arrival_time": [ts, pd.NaT, ts, pd.NaT],
        }
    )
    out = target.summarize_unscorable(target.compute_stop_deviations(df), timepoints_only=True)
    assert out == {
        "candidates": 3,  # the FALSE-timepoint row is not eligible
        "scored": 1,
        "missing_actual_time": 1,
        "missing_schedule_time": 1,
    }


def test_compute_trip_coverage_counts_unobserved_trips() -> None:
    """Scheduled in-service trips with zero scored visits appear in the denominator."""
    trips = pd.DataFrame(
        {
            "trip_id_performed": ["T1", "T2", "T3", "T4"],
            "route_id": ["101", "101", "101", "101"],
            "service_date": ["2025-01-02"] * 4,
            "schedule_relationship": ["Scheduled", "Scheduled", "Scheduled", "Canceled"],
            "trip_type": ["In service"] * 4,
        }
    )
    scored = pd.DataFrame({"trip_id_performed": ["T1", "T1", "T2"]})
    cov = target.compute_trip_coverage(trips, scored)

    route = cov.loc[(cov["level"] == "route") & (cov["route_id"] == "101")]
    assert len(route) == 1
    row = route.iloc[0]
    assert row["month"] == "2025-01"
    assert row["trips_scheduled"] == 3  # Canceled T4 is excluded
    assert row["trips_observed"] == 2  # T3 produced no visits
    assert row["pct_trips_observed"] == pytest.approx(66.7)
    assert row["evaluated_visits"] == 3
    assert row["visits_per_observed_trip"] == pytest.approx(1.5)

    overall = cov.loc[cov["level"] == "overall"]
    assert set(overall["route_id"].unique()) == {"ALL"}
    assert overall.iloc[0]["trips_scheduled"] == 3


def test_compute_trip_coverage_on_fixtures(scored: pd.DataFrame) -> None:
    """Fixture coverage is well-formed: bounded percentages, both levels present."""
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    cov = target.compute_trip_coverage(tp, scored)
    assert set(cov["level"].unique()) == {"route", "overall"}
    assert (cov["trips_observed"] <= cov["trips_scheduled"]).all()
    assert cov["pct_trips_observed"].between(0, 100).all()


def test_classify_otp_buckets() -> None:
    """Classification respects the inclusive on-time window."""
    df = pd.DataFrame({"dev_min": [-5.0, -1.0, 0.0, 5.0, 7.0]})
    out = target.classify_otp(df, early_min=-1.0, late_min=5.0)
    assert out["otp_class"].tolist() == ["early", "on_time", "on_time", "on_time", "late"]


def test_aggregate_otp_percentages_sum_to_100(scored: pd.DataFrame) -> None:
    """Per-cell early/on-time/late percentages add to 100 and counts reconcile."""
    agg = target.aggregate_otp(scored, ["route_id", "direction_id"])
    assert (agg["early"] + agg["on_time"] + agg["late"] == agg["evaluated"]).all()
    pct_sum = agg["pct_on_time"] + agg["pct_early"] + agg["pct_late"]
    assert pct_sum.round(6).eq(100.0).all()


def test_build_all_levels_and_long_table(scored: pd.DataFrame) -> None:
    """All four standard levels are produced and concatenate into a long table."""
    levels = target.build_all_levels(scored, corridors={})
    assert set(levels) == {"route_direction", "route", "service_type", "overall"}
    long_table = target.make_long_table(levels)
    assert {"level", "group", "month", "pct_on_time"} <= set(long_table.columns)
    # Overall level has exactly one group label.
    overall = long_table.loc[long_table["level"] == "overall"]
    assert set(overall["group"].unique()) == {"ALL"}


def test_corridor_level_added_when_configured(scored: pd.DataFrame) -> None:
    """A corridor mapping introduces a corridor level pooling its routes."""
    levels = target.build_all_levels(scored, corridors={"Downtown": ["101", "202"]})
    assert "corridor" in levels
    long_table = target.make_long_table(levels)
    corr = long_table.loc[long_table["level"] == "corridor"]
    assert set(corr["group"].unique()) == {"Downtown"}


def test_run_writes_tables_and_charts(tmp_path: Path) -> None:
    """End-to-end run produces the processed CSV, pivots, and PNG charts."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
    )
    long_table = target.run(cfg)
    assert not long_table.empty

    assert (tmp_path / target.PROCESSED_FILENAME).exists()
    assert (tmp_path / target.COVERAGE_FILENAME).exists()
    assert (tmp_path / "otp_monthly_route.csv").exists()
    pngs = list((tmp_path / "plots").glob("*.png"))
    assert pngs, "expected at least one OTP chart"
