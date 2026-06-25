"""Tests for ridership_from_tides using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import ridership_from_tides as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")

FINE_KEYS = ["month", "time_period"]


@pytest.fixture()
def prepared() -> pd.DataFrame:
    """Joined, filtered, ridership-tagged, period/month-tagged stop visits."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(sv, tp)
    return (
        joined.pipe(target.filter_for_ridership)
        .pipe(target.add_ridership_columns)
        .pipe(target.assign_time_period, target.TIME_PERIODS)
        .pipe(target.add_month)
    )


def test_join_brings_route_attributes() -> None:
    """The join attaches route/direction/service-type onto stop visits."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(sv, tp)
    for col in ("route_id", "direction_id", "route_type_agency"):
        assert col in joined.columns
    assert joined["route_id"].notna().all()
    assert set(joined["route_id"].unique()) <= {"101", "202", "303"}


def test_boardings_sum_both_doors() -> None:
    """Total boardings/alightings sum the per-door columns."""
    df = pd.DataFrame(
        {
            "boarding_1": [3.0, 0.0],
            "boarding_2": [1.0, 0.0],
            "alighting_1": [0.0, 2.0],
            "alighting_2": [0.0, 1.0],
        }
    )
    out = target.add_ridership_columns(df)
    assert out["boardings"].tolist() == [4.0, 0.0]
    assert out["alightings"].tolist() == [0.0, 3.0]


def test_filter_drops_skipped() -> None:
    """Skipped stop visits (no doors opened) are removed."""
    df = pd.DataFrame({"schedule_relationship": ["Scheduled", "Skipped", "Added"]})
    out = target.filter_for_ridership(df)
    assert "Skipped" not in set(out["schedule_relationship"])
    assert len(out) == 2


def test_assign_time_period_windows() -> None:
    """Each event lands in the window containing its time-of-day; midnight wraps."""
    df = pd.DataFrame(
        {
            "actual_departure_time": pd.to_datetime(
                [
                    "2025-01-02T07:00:00",  # AM PEAK
                    "2025-01-02T12:00:00",  # MIDDAY
                    "2025-01-02T16:00:00",  # PM PEAK
                    "2025-01-02T23:30:00",  # NIGHT (wraps)
                ]
            )
        }
    )
    out = target.assign_time_period(df, target.TIME_PERIODS)
    assert out["time_period"].tolist() == ["AM PEAK", "MIDDAY", "PM PEAK", "NIGHT"]


def test_assign_time_period_empty_is_all_day() -> None:
    """An empty TIME_PERIODS mapping labels everything ALL DAY."""
    df = pd.DataFrame({"actual_departure_time": pd.to_datetime(["2025-01-02T07:00:00"])})
    out = target.assign_time_period(df, {})
    assert out["time_period"].tolist() == ["ALL DAY"]


def test_add_month_is_year_month() -> None:
    """Month is derived as YYYY-MM from the service date."""
    df = pd.DataFrame({"service_date": pd.to_datetime(["2025-02-15"])})
    assert target.add_month(df)["month"].iloc[0] == "2025-02"


def test_resolve_grains_drops_period_when_no_windows() -> None:
    """Period-bearing grains are skipped when there are no time windows."""
    all_grains = ("month_and_period", "month", "period", "total")
    assert target.resolve_grains(all_grains, target.TIME_PERIODS) == list(all_grains)
    assert target.resolve_grains(all_grains, {}) == ["month", "total"]
    # Unknown grains are ignored.
    assert target.resolve_grains(("month", "bogus"), {}) == ["month"]


def test_aggregate_reconciles_with_raw_totals(prepared: pd.DataFrame) -> None:
    """Aggregated boardings equal the raw boardings sum (no double counting)."""
    agg = target.aggregate_ridership(prepared, ["route_id"], FINE_KEYS)
    assert agg["boardings"].sum() == pytest.approx(prepared["boardings"].sum())
    assert (agg["net_boardings"] == agg["boardings"] - agg["alightings"]).all()


def test_total_grain_is_single_row_per_group(prepared: pd.DataFrame) -> None:
    """The total grain collapses to one row per group with no temporal columns."""
    agg = target.aggregate_ridership(prepared, ["route_id"], [])
    assert "month" not in agg.columns
    assert "time_period" not in agg.columns
    assert agg["route_id"].is_unique


def test_coarser_grains_match_finest_totals(prepared: pd.DataFrame) -> None:
    """Each coarser grain sums to the same boardings as the finest grain."""
    fine = target.aggregate_ridership(prepared, ["route_id"], FINE_KEYS)
    by_month = target.aggregate_ridership(prepared, ["route_id"], ["month"])
    total = target.aggregate_ridership(prepared, ["route_id"], [])
    assert by_month["boardings"].sum() == pytest.approx(fine["boardings"].sum())
    assert total["boardings"].sum() == pytest.approx(fine["boardings"].sum())


def test_build_all_levels_and_long_table(prepared: pd.DataFrame) -> None:
    """All standard levels are produced and concatenate into a long table."""
    levels = target.build_all_levels(prepared, FINE_KEYS)
    assert set(levels) == {
        "route_stop",
        "stop",
        "route_direction",
        "route",
        "service_type",
        "overall",
    }
    long_table = target.make_long_table(levels, FINE_KEYS)
    assert {"level", "group", "month", "time_period", "boardings"} <= set(long_table.columns)
    overall = long_table.loc[long_table["level"] == "overall"]
    assert set(overall["group"].unique()) == {"ALL"}
    assert overall["boardings"].sum() == pytest.approx(prepared["boardings"].sum())


def test_by_route_and_stop_export_shape(prepared: pd.DataFrame) -> None:
    """The vendor-style export carries BOARD_ALL/ALIGHT_ALL keyed by stop."""
    levels = target.build_all_levels(prepared, FINE_KEYS)
    brs = target.build_by_route_and_stop(levels)
    for col in ("TIME_PERIOD", "ROUTE_ID", "STOP_ID", "BOARD_ALL", "ALIGHT_ALL"):
        assert col in brs.columns
    assert brs["BOARD_ALL"].sum() == pytest.approx(prepared["boardings"].sum())


def test_run_writes_all_grain_files(tmp_path: Path) -> None:
    """The end-to-end run writes one file per grain, the stop export, run log."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
        time_periods=target.TIME_PERIODS,
        export_grains=target.EXPORT_GRAINS,
    )
    grain_tables = target.run(cfg)
    assert set(grain_tables) == {"month_and_period", "month", "period", "total"}
    for grain in grain_tables:
        assert (tmp_path / target.GRAIN_FILENAME[grain]).exists()
    assert (tmp_path / target.BY_ROUTE_AND_STOP_FILENAME).exists()
    assert (tmp_path / "ridership_from_tides_runlog.txt").exists()
