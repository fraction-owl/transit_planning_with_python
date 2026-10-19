"""Tests for runtime_schedule_builder using synthetic data and the repo TIDES fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import runtime_schedule_builder as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


def _run(tmp_path: Path, *extra: str) -> int:
    return target.main(
        [
            "--stop-visits",
            str(STOP_VISITS),
            "--trips-performed",
            str(TRIPS_PERFORMED),
            "--output-dir",
            str(tmp_path),
            "--no-interactive",
            "--no-charts",
            *extra,
        ]
    )


# --- Step 1: periods ---------------------------------------------------------


def test_ordered_partition_finds_step_breaks() -> None:
    """Groups stay in order and split where the level shifts."""
    values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5])
    sse, back = target.ordered_partition(values, np.ones(len(values), dtype=bool), 3)
    assert sse[3] == pytest.approx(0.0)
    assert target.partition_starts(back, 3, len(values)) == [0, 3, 6]


def test_ordered_partition_respects_can_start() -> None:
    """A group can't begin where can_start is False (e.g. two trips in one minute)."""
    values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    can_start = np.array([True, True, True, False, True, True])
    _, back = target.ordered_partition(values, can_start, 2)
    assert 3 not in target.partition_starts(back, 2, len(values))


def test_propose_periods_breaks_at_peaks() -> None:
    """With a fixed count, the breaks land on the first trip of each level."""
    starts = np.arange(360, 360 + 15 * 24, 15)  # 06:00 every 15 minutes
    ratio = np.where((starts >= 420) & (starts < 540), 1.3, 1.0)  # 07:00-08:59 peak
    summary = pd.DataFrame(
        {
            "route_id": "10",
            "direction_id": "0",
            "pattern_timepoints": "A>B",
            "trip_key": [f"T{s}" for s in starts],
            "start_min": starts.astype(float),
            "ratio": ratio,
            "used_for_breaks": True,
        }
    )
    cfg = target.Config(Path("sv"), Path("tp"), Path("out"), num_periods=3)
    periods = target.propose_periods(summary, cfg)
    assert periods["period_start"].tolist() == ["06:00", "07:00", "09:00"]
    assert periods["period_end"].tolist() == ["07:00", "09:00", ""]


def test_assign_periods_by_start_time() -> None:
    """A trip joins the last period starting at or before it; early trips join period 1."""
    periods = target.number_periods(
        pd.DataFrame({"route_id": "10", "direction_id": "0", "start_min": [400.0, 540.0]})
    )
    trips = pd.DataFrame(
        {"route_id": "10", "direction_id": "0", "start_min": [390.0, 539.5, 540.0, 900.0]}
    )
    out = target.assign_periods(trips, periods)
    assert out["period"].tolist() == [1, 1, 2, 2]


# --- Step 2: runtimes --------------------------------------------------------


def test_runtime_options_discount_early_and_unreachable_goal() -> None:
    """Forgiving early arrivals, the runtime is the goal percentile minus the late allowance.

    Ten runtimes spread evenly over 9 minutes can't fit 85% of trips in a
    6-minute window, so the two-sided goal is flagged unreachable.
    """
    values = np.arange(20.0, 30.0)
    opts = target.runtime_options(values, 85.0, -1.0, 5.0, discount_early=True)
    assert opts["proposed_runtime_min"] == 23  # 9 of 10 runtimes <= 23 + 5
    assert opts["est_otp_proposed_pct"] == pytest.approx(90.0)
    assert opts["best_two_sided_otp_pct"] == pytest.approx(70.0)
    assert opts["goal_reachable_two_sided"] is False
    assert opts["span_for_goal_min"] == pytest.approx(8.0)


def test_runtime_options_counting_early() -> None:
    """Counting early arrivals, the smallest runtime whose window holds the goal is chosen."""
    values = np.array([30.0, 31.0, 31.5, 32.0, 32.0, 33.0, 33.5, 34.0, 34.5, 35.0])
    opts = target.runtime_options(values, 85.0, -1.0, 5.0, discount_early=False)
    assert opts["goal_reachable_two_sided"] is True
    assert opts["proposed_runtime_min"] == 30  # window [29, 35] holds every runtime
    assert opts["est_otp_proposed_pct"] == pytest.approx(100.0)


def test_estimate_final_otp() -> None:
    """On time means within the window; early is forgiven only when asked."""
    actual = np.array([28.0, 30.0, 34.0, 36.0])
    assert target.estimate_final_otp(actual, 30, -1.0, 5.0, discount_early=True) == 75.0
    assert target.estimate_final_otp(actual, 30, -1.0, 5.0, discount_early=False) == 50.0


# --- Step 3: spread ----------------------------------------------------------


def test_allocate_minutes_backloads_leftover_minutes() -> None:
    """Rounding cumulative times down pushes leftover minutes toward the end."""
    assert target.allocate_minutes([48, 26, 26], 10, backload=True) == [4, 3, 3]
    assert target.allocate_minutes([48, 26, 26], 10, backload=False) == [5, 2, 3]
    assert target.allocate_minutes([25, 25, 25, 25], 42, backload=True) == [10, 11, 10, 11]
    assert target.allocate_minutes([25, 25, 25, 25], 42, backload=False) == [11, 10, 11, 10]


def test_allocate_minutes_always_sums_to_total() -> None:
    """Any shares (even ones not summing to 100) divide the whole runtime."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        pcts = rng.uniform(0, 50, size=rng.integers(1, 9))
        total = int(rng.integers(1, 120))
        for backload in (True, False):
            minutes = target.allocate_minutes(pcts, total, backload)
            assert sum(minutes) == total
            assert min(minutes) >= 0


def test_best_cumulative_moves_only_as_far_as_needed() -> None:
    """Among equally good scheduled times, the one nearest the median-share time wins."""
    arrivals = np.array([4.8, 5.0, 5.0, 5.2])
    # Every time from 0.2 to 5.8 puts all arrivals within [-1, +5]: keep the anchor.
    assert target._best_cumulative(arrivals, -1.0, 5.0, anchor=5.0) == pytest.approx(5.0)
    # A padded anchor (arrivals would be early) moves back just to the plateau's edge.
    assert target._best_cumulative(arrivals, -1.0, 5.0, anchor=9.0) == pytest.approx(5.8)
    # No observations: keep the anchor.
    assert target._best_cumulative(np.array([np.nan]), -1.0, 5.0, anchor=7.0) == 7.0


def test_allocate_minutes_rejects_bad_percents() -> None:
    """All-zero or negative shares can't divide a runtime."""
    with pytest.raises(ValueError):
        target.allocate_minutes([0, 0], 10)
    with pytest.raises(ValueError):
        target.allocate_minutes([60, -10, 50], 10)


# --- Step 4: checks ----------------------------------------------------------


def test_check_pattern_consistency_compares_across_skipped_timepoints() -> None:
    """A pattern that skips timepoint B is still compared with A->C on the full pattern."""
    rows = []
    for label, stops, cumulative in (
        ("P1", ["A", "B", "C"], [0, 5, 12]),
        ("P2", ["A", "C"], [0, 10]),
    ):
        for seq, (stop, cum) in enumerate(zip(stops, cumulative), start=1):
            rows.append(
                {
                    "route_id": "10",
                    "direction_id": "0",
                    "pattern_timepoints": ">".join(stops),
                    "period_start_min": 360.0,
                    "pattern": label,
                    "period_start": "06:00",
                    "period_end": "",
                    "tp_seq": seq,
                    "stop_id": stop,
                    "cumulative_min": cum,
                }
            )
    issues = target.check_pattern_consistency(pd.DataFrame(rows))
    assert len(issues) == 1
    assert (issues[0]["from_stop_id"], issues[0]["to_stop_id"]) == ("A", "C")
    assert issues[0]["detail"] == "P1: 12 min; P2: 10 min"


# --- Observations & filters --------------------------------------------------


def test_build_observations_measures_from_first_departure() -> None:
    """Runtime runs from the first timepoint's departure to the last one's arrival."""
    times = ["08:00:00", "08:02:00", "08:05:00", "08:12:00"]
    stop_visits = pd.DataFrame(
        {
            "service_date": pd.Timestamp("2025-02-04"),
            "trip_id_performed": "T1",
            "trip_stop_sequence": [1, 2, 3, 4],
            "stop_id": ["A", "x", "B", "C"],
            "timepoint": [True, False, True, True],
            "schedule_arrival_time": pd.to_datetime([f"2025-02-04T{t}" for t in times]),
            "schedule_departure_time": pd.to_datetime([f"2025-02-04T{t}" for t in times]),
            "actual_arrival_time": pd.to_datetime(
                ["2025-02-04T08:00:00", "", "2025-02-04T08:05:00", "2025-02-04T08:12:00"]
            ),
            "actual_departure_time": pd.to_datetime(
                ["2025-02-04T08:00:00", "", "2025-02-04T08:05:30", "2025-02-04T08:12:00"]
            ),
            "schedule_relationship": "Scheduled",
        }
    )
    trips = pd.DataFrame(
        {
            "service_date": [pd.Timestamp("2025-02-04")],
            "trip_id_performed": ["T1"],
            "route_id": ["10"],
            "direction_id": ["0"],
        }
    )
    trip_obs, tp_obs = target.build_observations(stop_visits, trips)
    trip = trip_obs.iloc[0]
    assert trip["pattern_timepoints"] == "A>B>C"
    assert trip["actual_runtime_min"] == pytest.approx(12.0)
    assert trip["start_min"] == pytest.approx(480.0)
    seg = tp_obs.set_index("stop_id")
    assert seg.loc["B", "seg_actual_min"] == pytest.approx(5.0)
    assert seg.loc["C", "seg_actual_min"] == pytest.approx(6.5)  # excludes dwell at B
    assert seg.loc["C", "cum_actual_min"] == pytest.approx(12.0)


def test_filter_service_drops_weekends_and_federal_holidays() -> None:
    """Only the configured days remain, minus observed federal holidays."""
    dates = pd.to_datetime(["2025-02-14", "2025-02-15", "2025-02-17", "2025-02-18"])
    trips = pd.DataFrame({"service_date": dates, "route_id": "10", "direction_id": "0"})
    cfg = target.Config(Path("sv"), Path("tp"), Path("out"))
    kept = target.filter_service(trips, cfg)["service_date"].dt.strftime("%Y-%m-%d").tolist()
    assert kept == ["2025-02-14", "2025-02-18"]  # Sat 15th and Presidents Day 17th dropped


# --- Prompt parsing ----------------------------------------------------------


def test_parse_time_list() -> None:
    """Typed start times split on commas or spaces, sorted and de-duplicated."""
    assert target.parse_time_list("09:10, 05:00 6:45,6:45") == [300, 405, 550]
    with pytest.raises(ValueError):
        target.parse_time_list("6:45, noon")


def test_parse_number_list() -> None:
    """Typed values must match the expected count and minimum."""
    assert target.parse_number_list("20, 30% 50", 3) == [20.0, 30.0, 50.0]
    with pytest.raises(ValueError):
        target.parse_number_list("20, 30", 3)
    with pytest.raises(ValueError):
        target.parse_number_list("0, 40", 2, minimum=1.0)


# --- End to end --------------------------------------------------------------


def test_main_returns_2_for_placeholder_paths() -> None:
    """Unset CONFIGURATION paths are a configuration error."""
    assert target.main(["--no-interactive"]) == 2


def test_main_end_to_end_on_fixtures(tmp_path: Path) -> None:
    """A full run writes every step, and each runtime table adds up to its runtime."""
    assert _run(tmp_path) == 0
    for name in (
        target.PERIODS_FILENAME,
        target.PERIOD_TRIPS_FILENAME,
        target.RUNTIMES_FILENAME,
        target.SPREAD_FILENAME,
        target.TABLE_FILENAME,
        target.CHECKS_FILENAME,
    ):
        assert (tmp_path / name).exists(), name
    assert list((tmp_path / "runtime_tables").glob("runtime_table_*.csv"))

    run_log = (tmp_path / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "OTP_GOAL_PCT" in run_log
    assert "Step 1 periods:     proposal" in run_log

    table = pd.read_csv(tmp_path / target.TABLE_FILENAME, dtype={"route_id": str})
    keys = ["route_id", "direction_id", "pattern", "period"]
    last = table.sort_values("tp_seq").groupby(keys).tail(1)
    assert (last["cumulative_min"] == last["runtime_min"]).all()
    sums = table.groupby(keys).agg(seg=("segment_min", "sum"), runtime=("runtime_min", "first"))
    assert (sums["seg"] == sums["runtime"]).all()


def test_edited_csvs_are_used_on_the_next_run(tmp_path: Path) -> None:
    """Edited period and runtime CSVs from one run drive the next."""
    first = tmp_path / "first"
    assert _run(first) == 0

    periods = pd.read_csv(first / target.PERIODS_FILENAME, dtype=str)
    one_each = periods.groupby(["route_id", "direction_id"]).head(1)
    one_each.to_csv(tmp_path / "periods_edit.csv", index=False)
    second = tmp_path / "second"
    assert _run(second, "--periods-csv", str(tmp_path / "periods_edit.csv")) == 0
    runtimes = pd.read_csv(second / target.RUNTIMES_FILENAME, dtype=str)
    assert set(runtimes["period"]) == {"1"}

    runtimes["chosen_runtime_min"] = (runtimes["proposed_runtime_min"].astype(int) + 3).astype(str)
    runtimes.to_csv(tmp_path / "runtimes_edit.csv", index=False)
    third = tmp_path / "third"
    assert (
        _run(
            third,
            "--periods-csv",
            str(tmp_path / "periods_edit.csv"),
            "--runtimes-csv",
            str(tmp_path / "runtimes_edit.csv"),
        )
        == 0
    )
    rebuilt = pd.read_csv(third / target.RUNTIMES_FILENAME)
    assert (rebuilt["chosen_runtime_min"] == rebuilt["proposed_runtime_min"] + 3).all()
    spread = pd.read_csv(third / target.SPREAD_FILENAME)
    merged = spread.merge(
        rebuilt[["route_id", "direction_id", "pattern", "period", "chosen_runtime_min"]],
        on=["route_id", "direction_id", "pattern", "period"],
    )
    assert (merged["runtime_min"] == merged["chosen_runtime_min"]).all()
