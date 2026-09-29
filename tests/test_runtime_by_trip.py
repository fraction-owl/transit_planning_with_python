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


def _visits(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    """Stop visits for trip T1 from (date, seq, arrival HH:MM, departure HH:MM)."""
    return pd.DataFrame(
        {
            "service_date": pd.to_datetime([r[0] for r in rows]),
            "trip_id_performed": ["T1"] * len(rows),
            "trip_stop_sequence": [r[1] for r in rows],
            "actual_arrival_time": pd.to_datetime([f"{r[0]} {r[2]}" for r in rows]),
            "actual_departure_time": pd.to_datetime([f"{r[0]} {r[3]}" for r in rows]),
        }
    )


def test_runtimes_are_per_service_date_for_a_reused_trip_id() -> None:
    """Two daily 20-minute runs of T1 are two runtimes, not one 1,460-minute trip."""
    sv = _visits(
        [
            ("2025-01-06", 1, "06:00", "06:00"),
            ("2025-01-06", 2, "06:20", "06:20"),
            ("2025-01-07", 1, "06:00", "06:00"),
            ("2025-01-07", 2, "06:20", "06:20"),
        ]
    )
    trips = pd.DataFrame(
        {
            "service_date": ["2025-01-06", "2025-01-07"],
            "trip_id_performed": ["T1", "T1"],
            "route_id": ["101", "101"],
            "direction_id": ["0", "0"],
        }
    )
    joined = target.join_trip_attributes(target.compute_trip_runtimes(sv), trips)
    assert joined["actual_runtime_min"].tolist() == [20.0, 20.0]
    assert joined["dow"].tolist() == ["Monday", "Tuesday"]


def test_single_observed_stop_is_not_a_runtime() -> None:
    """One stop with dwell has no distinct endpoints, so no runtime is recorded."""
    sv = _visits([("2025-01-06", 1, "06:00", "06:02")])
    assert target.compute_trip_runtimes(sv).empty


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


def test_run_writes_all_outputs(tmp_path: Path) -> None:
    """End-to-end run writes the four CSVs and at least one chart."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
    )
    results = target.run(cfg)
    assert set(results) == {"retained", "outliers", "stats", "dow"}

    for name in (
        "trip_runtime_observations.csv",
        "trip_runtime_outliers.csv",
        "trip_runtime_stats.csv",
        "trip_runtime_dow.csv",
    ):
        assert (tmp_path / name).exists()
    pngs = list((tmp_path / "plots").glob("*.png"))
    assert pngs, "expected at least one runtime chart"
