"""Tests for runtime_by_trip using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import runtime_by_trip as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


@pytest.fixture()
def joined() -> pd.DataFrame:
    """Per-trip runtimes joined with route/direction/DOW context."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    runtimes = target.compute_trip_runtimes(sv)
    return target.join_trip_attributes(runtimes, tp)


def test_compute_trip_runtimes_positive(joined: pd.DataFrame) -> None:
    """Each trip has a positive runtime and start time."""
    assert not joined.empty
    assert (joined["actual_runtime_min"] > 0).all()
    assert joined["start_time"].notna().all()


def test_join_adds_route_and_dow(joined: pd.DataFrame) -> None:
    """Join attaches route/direction, a trip key, and day-of-week."""
    for col in ("route_id", "direction_id", "trip_key", "dow", "start_hhmm"):
        assert col in joined.columns
    assert set(joined["dow"].unique()) <= set(target.DOW_ORDER)


def test_trim_outliers_partitions_rows() -> None:
    """Trimming removes the extreme tails and conserves total rows."""
    df = pd.DataFrame(
        {
            "trip_key": ["T"] * 100,
            "actual_runtime_min": list(range(100)),
        }
    )
    retained, outliers = target.trim_outliers(df, frac=0.05)
    assert len(retained) + len(outliers) == 100
    assert len(outliers) > 0
    # The very smallest and largest values are dropped.
    assert 0 not in retained["actual_runtime_min"].to_numpy()
    assert 99 not in retained["actual_runtime_min"].to_numpy()


def test_trim_outliers_disabled() -> None:
    """frac=0 keeps everything and yields no outliers."""
    df = pd.DataFrame({"trip_key": ["T", "T"], "actual_runtime_min": [10.0, 20.0]})
    retained, outliers = target.trim_outliers(df, frac=0.0)
    assert len(retained) == 2
    assert outliers.empty


def test_compute_trip_stats_flags(joined: pd.DataFrame) -> None:
    """Stats expose mean/median/cv and the high_variation / data_gap flags."""
    retained, _ = target.trim_outliers(joined, frac=target.TRIM_FRAC)
    stats = target.compute_trip_stats(retained)
    for col in (
        "n_obs",
        "runtime_mean_min",
        "runtime_median_min",
        "cv",
        "high_variation",
        "data_gap",
    ):
        assert col in stats.columns
    assert stats["high_variation"].dtype == bool
    assert stats["data_gap"].dtype == bool


def test_dow_anomalies_flags_columns(joined: pd.DataFrame) -> None:
    """DOW table reports counts, means, and low_count / runtime_anomaly flags."""
    retained, _ = target.trim_outliers(joined, frac=target.TRIM_FRAC)
    dow = target.compute_dow_anomalies(retained)
    for col in ("n_obs", "dow_mean_min", "trip_mean_min", "low_count", "runtime_anomaly"):
        assert col in dow.columns
    # The fixture is Monday-heavy, so some DOW buckets carry far fewer trips;
    # the low_count detector should surface at least one of them.
    assert dow["low_count"].any()


def test_scheduled_runtime_is_endpoint_matched(joined: pd.DataFrame) -> None:
    """Scheduled runtime is derived and lands close to the observed runtime."""
    assert "scheduled_runtime_min" in joined.columns
    assert joined["scheduled_runtime_min"].notna().all()
    assert (joined["scheduled_runtime_min"] > 0).all()
    # Endpoint-matched schedule spans should track the observed runtimes closely.
    ratio = joined["actual_runtime_min"] / joined["scheduled_runtime_min"]
    assert ratio.between(0.5, 2.0).all()


def test_join_adds_pattern_key(joined: pd.DataFrame) -> None:
    """The stop pattern a trip ran is carried through the join."""
    assert "pattern_key" in joined.columns
    assert (joined["pattern_key"].str.strip() != "").all()


def _b2b_stats(**overrides: object) -> pd.DataFrame:
    """Build a minimal stats-shaped frame: 4 trips, 20 min apart, one pattern."""
    base = pd.DataFrame(
        {
            "route_id": ["101"] * 4,
            "direction_id": ["0"] * 4,
            "trip_key": ["T0", "T1", "T2", "T3"],
            "pattern_key": ["P1"] * 4,
            "start_hhmm": ["06:00", "06:20", "06:40", "07:00"],
            "n_obs": [10, 10, 10, 10],
            "runtime_median_min": [31.0, 32.0, 33.0, 34.0],
            "act_sched_ratio": [1.03, 1.07, 1.10, 1.13],
        }
    )
    return base.assign(**overrides)


def test_back_to_back_quiet_on_smooth_runtimes() -> None:
    """Neighbors that drift gently are never flagged."""
    pairs = target.compute_back_to_back_flags(_b2b_stats())
    assert len(pairs) == 3
    assert pairs["compared"].all()
    assert not pairs["back_to_back_flag"].any()


def test_back_to_back_flags_ratio_and_diff() -> None:
    """A doubled runtime trips both the ratio and the absolute-minutes test."""
    stats = _b2b_stats(runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    pairs = target.compute_back_to_back_flags(stats)
    flagged = pairs.loc[pairs["back_to_back_flag"]]
    assert set(flagged["to_trip_key"]) | set(flagged["from_trip_key"]) >= {"T2"}
    assert flagged["flag_runtime_ratio"].all()
    assert flagged["flag_runtime_diff"].all()


def test_back_to_back_flags_actual_scheduled_ratio_alone() -> None:
    """Diverging actual/scheduled ratios flag even when runtimes look similar."""
    stats = _b2b_stats(
        runtime_median_min=[31.0, 32.0, 33.0, 34.0],
        act_sched_ratio=[1.0, 1.0, 1.8, 1.0],
    )
    pairs = target.compute_back_to_back_flags(stats)
    flagged = pairs.loc[pairs["back_to_back_flag"]]
    assert not flagged.empty
    assert flagged["flag_act_sched_ratio"].all()
    assert not flagged["flag_runtime_ratio"].any()
    assert not flagged["flag_runtime_diff"].any()


def test_back_to_back_pairs_only_within_a_pattern() -> None:
    """A short-turn is never compared against a full-length trip."""
    stats = _b2b_stats(
        pattern_key=["FULL", "SHORT", "FULL", "SHORT"],
        runtime_median_min=[60.0, 25.0, 61.0, 26.0],
    )
    pairs = target.compute_back_to_back_flags(stats)
    # One pair per pattern, each joining same-pattern trips only.
    assert len(pairs) == 2
    assert set(pairs["pattern_key"]) == {"FULL", "SHORT"}
    assert not pairs["back_to_back_flag"].any()


def test_back_to_back_skipped_without_pattern_id() -> None:
    """Without a usable pattern_id the check refuses to guess."""
    stats = _b2b_stats(pattern_key=[""] * 4, runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    assert target.compute_back_to_back_flags(stats).empty
    # Opting in compares within route/direction instead.
    pairs = target.compute_back_to_back_flags(stats, require_pattern=False)
    assert pairs["back_to_back_flag"].any()


def test_back_to_back_gap_guard_and_min_obs() -> None:
    """Distant neighbors and thin samples are reported but not judged."""
    stats = _b2b_stats(
        start_hhmm=["06:00", "10:00", "14:00", "18:00"],
        runtime_median_min=[31.0, 32.0, 66.0, 33.0],
    )
    pairs = target.compute_back_to_back_flags(stats)
    assert not pairs["compared"].any()
    assert not pairs["back_to_back_flag"].any()
    # Removing the gap limit lets the same pairs be judged.
    assert target.compute_back_to_back_flags(stats, max_gap_min=0)["back_to_back_flag"].any()

    thin = _b2b_stats(n_obs=[1, 1, 1, 1], runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    assert not target.compute_back_to_back_flags(thin)["compared"].any()


def test_back_to_back_thresholds_are_individually_disablable() -> None:
    """Setting a threshold to 0 turns off just that test."""
    stats = _b2b_stats(runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    pairs = target.compute_back_to_back_flags(stats, runtime_ratio=0, runtime_diff_min=0)
    assert pairs["compared"].all()
    assert not pairs["back_to_back_flag"].any()


def test_apply_back_to_back_flag_marks_both_trips() -> None:
    """Every trip in a flagged pair is marked on the stats table."""
    stats = _b2b_stats(runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    pairs = target.compute_back_to_back_flags(stats)
    flagged = target.apply_back_to_back_flag(stats, pairs)
    assert flagged.loc[flagged["trip_key"] == "T2", "back_to_back_flag"].all()
    assert not flagged.loc[flagged["trip_key"] == "T0", "back_to_back_flag"].any()


def test_pattern_key_prefers_stop_visits_then_trips_performed() -> None:
    """stop_visits wins; trips_performed fills in when the visits carry none.

    The converters in this folder synthesize pattern_id on stop visits but leave
    it blank on trips_performed, so the visit table is the better source.
    """
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    tp_marked = tp.assign(pattern_id="FROM_TRIPS")

    both = target.join_trip_attributes(target.compute_trip_runtimes(sv), tp_marked)
    assert (both["pattern_key"] != "FROM_TRIPS").all()

    visits_missing = target.join_trip_attributes(
        target.compute_trip_runtimes(sv.drop(columns=["pattern_id"])), tp_marked
    )
    assert (visits_missing["pattern_key"] == "FROM_TRIPS").all()


def test_pattern_key_empty_when_neither_table_has_one() -> None:
    """With no pattern_id anywhere the key is blank, so the check will skip."""
    sv = target.load_stop_visits(STOP_VISITS).drop(columns=["pattern_id"])
    tp = target.load_trips_performed(TRIPS_PERFORMED).drop(columns=["pattern_id"])
    joined = target.join_trip_attributes(target.compute_trip_runtimes(sv), tp)
    assert (joined["pattern_key"] == "").all()


def test_run_writes_all_outputs(tmp_path: Path) -> None:
    """End-to-end run writes the five CSVs, a run log, and at least one chart."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
    )
    results = target.run(cfg)
    assert set(results) == {"retained", "outliers", "stats", "dow", "back_to_back"}
    assert "back_to_back_flag" in results["stats"].columns

    for name in (
        "trip_runtime_observations.csv",
        "trip_runtime_outliers.csv",
        "trip_runtime_stats.csv",
        "trip_runtime_dow.csv",
        "trip_runtime_back_to_back.csv",
    ):
        assert (tmp_path / name).exists()
    pngs = list((tmp_path / "plots").glob("*.png"))
    assert pngs, "expected at least one runtime chart"


def test_run_writes_run_log_sidecar(tmp_path: Path) -> None:
    """The sidecar records the run and the verbatim CONFIGURATION block."""
    target.run(
        target.Config(
            stop_visits_path=STOP_VISITS,
            trips_performed_path=TRIPS_PERFORMED,
            output_dir=tmp_path,
        )
    )
    log = tmp_path / "runtime_by_trip_runlog.txt"
    assert log.exists()

    text = log.read_text(encoding="utf-8")
    assert "Run timestamp:" in text
    assert "Source script:" in text
    assert "Back-to-back pairs:" in text
    # Config constants are reproduced verbatim so a run can be reconstructed.
    assert "B2B_RUNTIME_RATIO: float = 2.0" in text
    assert f"B2B_MAX_GAP_MIN: float = {target.B2B_MAX_GAP_MIN}" in text
    # The marker lines themselves are excluded from the captured block.
    assert target.CONFIG_BEGIN_MARKER not in text
