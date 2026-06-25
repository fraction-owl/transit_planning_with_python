"""Tests for max_load_from_tides using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import max_load_from_tides as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")

FINE_KEYS = ["month", "time_period"]


@pytest.fixture()
def prepared() -> pd.DataFrame:
    """Per-trip maxima joined to attributes, period/month/load-factor tagged."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    per_trip = target.compute_trip_max_load(sv)
    joined = target.join_trip_attributes(per_trip, tp)
    return (
        joined.pipe(target.assign_time_period, target.TIME_PERIODS)
        .pipe(target.add_month)
        .pipe(target.add_load_factor, target.VEHICLE_CAPACITY)
    )


def test_compute_trip_max_load_is_per_trip_peak() -> None:
    """Each trip's max load is the largest departure_load over its sequence."""
    sv = pd.DataFrame(
        {
            "trip_id_performed": ["A", "A", "A", "B", "B"],
            "trip_stop_sequence": [1, 2, 3, 1, 2],
            "stop_id": ["s1", "s2", "s3", "s1", "s2"],
            "departure_load": [2, 9, 4, 1, 1],
            "service_date": pd.to_datetime(["2025-01-02"] * 5),
        }
    )
    out = target.compute_trip_max_load(sv)
    out = out.set_index("trip_id_performed")
    assert out.loc["A", "max_load"] == 9
    assert out.loc["A", "peak_stop_id"] == "s2"
    assert out.loc["B", "max_load"] == 1


def test_compute_trip_max_load_drops_all_nan_trip() -> None:
    """A trip with no usable load reading is dropped, not zero-filled."""
    sv = pd.DataFrame(
        {
            "trip_id_performed": ["A", "A", "B"],
            "trip_stop_sequence": [1, 2, 1],
            "stop_id": ["s1", "s2", "s1"],
            "departure_load": [3, 5, None],
            "service_date": pd.to_datetime(["2025-01-02"] * 3),
        }
    )
    out = target.compute_trip_max_load(sv)
    assert set(out["trip_id_performed"]) == {"A"}


def test_join_attributes_match_fixture_routes() -> None:
    """Per-trip rows pick up route attributes from trips_performed."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(target.compute_trip_max_load(sv), tp)
    assert set(joined["route_id"].unique()) <= {"101", "202", "303"}
    assert joined["max_load"].notna().all()


def test_assign_time_period_uses_trip_start() -> None:
    """A trip is placed in the window containing its start time."""
    df = pd.DataFrame(
        {"schedule_trip_start": pd.to_datetime(["2025-01-02T07:30:00", "2025-01-02T17:00:00"])}
    )
    out = target.assign_time_period(df, target.TIME_PERIODS)
    assert out["time_period"].tolist() == ["AM PEAK", "PM PEAK"]


def test_load_factor_uses_capacity() -> None:
    """Load factor is max load over the configured capacity."""
    df = pd.DataFrame({"max_load": [39.0, 20.0]})
    out = target.add_load_factor(df, 39)
    assert out["load_factor"].tolist() == [1.0, pytest.approx(0.5128, abs=1e-4)]


def test_resolve_grains_drops_period_when_no_windows() -> None:
    """Period-bearing grains are skipped when there are no time windows."""
    all_grains = ("month_and_period", "month", "period", "total")
    assert target.resolve_grains(all_grains, target.TIME_PERIODS) == list(all_grains)
    assert target.resolve_grains(all_grains, {}) == ["month", "total"]


def test_aggregate_stats_and_over_capacity(prepared: pd.DataFrame) -> None:
    """Aggregation reports trip counts, a peak >= mean, and an over-capacity %."""
    agg = target.aggregate_max_load(prepared, ["route_id"], FINE_KEYS, target.VEHICLE_CAPACITY)
    assert (agg["trips"] > 0).all()
    assert (agg["peak_max_load"] >= agg["mean_max_load"]).all()
    assert (agg["pct_trips_over_capacity"] >= 0).all()
    assert agg["trips"].sum() == len(prepared)


def test_total_grain_peak_matches_overall_peak(prepared: pd.DataFrame) -> None:
    """The total grain's peak equals the raw maximum (recomputed, not rolled up)."""
    total = target.aggregate_max_load(prepared, [], [], target.VEHICLE_CAPACITY)
    assert len(total) == 1
    assert total["peak_max_load"].iloc[0] == prepared["max_load"].max()
    assert total["trips"].iloc[0] == len(prepared)


def test_build_all_levels_and_long_table(prepared: pd.DataFrame) -> None:
    """All standard levels are produced and concatenate into a long table."""
    levels = target.build_all_levels(prepared, FINE_KEYS, target.VEHICLE_CAPACITY)
    assert set(levels) == {"route_direction", "route", "service_type", "overall"}
    long_table = target.make_long_table(levels, FINE_KEYS)
    assert {"level", "group", "month", "time_period", "peak_max_load"} <= set(long_table.columns)
    overall = long_table.loc[long_table["level"] == "overall"]
    assert set(overall["group"].unique()) == {"ALL"}


def test_run_writes_all_grain_files(tmp_path: Path) -> None:
    """The end-to-end run writes the per-trip table plus one file per grain."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
        vehicle_capacity=target.VEHICLE_CAPACITY,
        time_periods=target.TIME_PERIODS,
        export_grains=target.EXPORT_GRAINS,
    )
    result = target.run(cfg)
    assert set(result) == {"by_trip", "month_and_period", "month", "period", "total"}
    assert (tmp_path / target.BY_TRIP_FILENAME).exists()
    for grain in ("month_and_period", "month", "period", "total"):
        assert (tmp_path / target.GRAIN_FILENAME[grain]).exists()
    assert (tmp_path / "max_load_from_tides_runlog.txt").exists()
