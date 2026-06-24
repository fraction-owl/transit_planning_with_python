"""Tests for route_runtime_tides using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import route_runtime_tides as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


@pytest.fixture()
def joined() -> pd.DataFrame:
    """Per-trip runtimes joined with route context."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    runtimes = target.compute_trip_runtimes(sv)
    return target.join_route_attributes(runtimes, tp)


def test_join_adds_route_and_trip_key(joined: pd.DataFrame) -> None:
    """The join attaches route_id and a trip key, with positive runtimes."""
    assert not joined.empty
    for col in ("route_id", "trip_key", "actual_runtime_min"):
        assert col in joined.columns
    assert (joined["actual_runtime_min"] > 0).all()
    assert set(joined["route_id"].unique()) <= {"101", "202", "303"}


def test_aggregate_route_month_schema(joined: pd.DataFrame) -> None:
    """Panel is one row per (route, month) with the runtime statistics."""
    panel = target.aggregate_route_month(target.add_month(joined))
    assert list(panel.columns) == [
        "route_id",
        "month",
        "n_obs",
        "runtime_mean_min",
        "runtime_median_min",
        "runtime_std_min",
    ]
    # No duplicate (route, month) cells.
    assert not panel.duplicated(subset=["route_id", "month"]).any()


def test_aggregate_empty_returns_schema() -> None:
    """An empty input still yields the panel schema (no crash downstream)."""
    panel = target.aggregate_route_month(pd.DataFrame())
    assert panel.empty
    assert "runtime_mean_min" in panel.columns


def test_select_window_trailing_and_end_month() -> None:
    """The window keeps the most recent N months, respecting END_MONTH."""
    months = ["2025-01", "2025-02", "2025-03", "2025-01"]
    assert target.select_window(months, "", 2) == ["2025-02", "2025-03"]
    assert target.select_window(months, "2025-02", 2) == ["2025-01", "2025-02"]
    assert target.select_window(months, "", 0) == ["2025-01", "2025-02", "2025-03"]


def _toy_panel() -> pd.DataFrame:
    """Route A: a light month (mean 10, 1 obs) and a heavy month (mean 20, 3 obs)."""
    return pd.DataFrame(
        {
            "route_id": ["A", "A"],
            "month": ["2025-01", "2025-02"],
            "n_obs": [1, 3],
            "runtime_mean_min": [10.0, 20.0],
            "runtime_median_min": [10.0, 20.0],
            "runtime_std_min": [0.0, 0.0],
        }
    )


def test_reduce_normalized_weights_months_equally() -> None:
    """Normalized rollup averages the monthly means: (10 + 20) / 2 = 15."""
    rollup = target.reduce_to_route(
        _toy_panel(), ["2025-01", "2025-02"], normalize_by_month=True
    )
    row = rollup.iloc[0]
    assert row["runtime_mean_min"] == pytest.approx(15.0)
    assert row["n_months"] == 2
    assert row["n_obs"] == 4
    assert bool(row["normalized"]) is True


def test_reduce_naive_weights_by_observations() -> None:
    """Naive rollup pools observations: (10*1 + 20*3) / 4 = 17.5."""
    rollup = target.reduce_to_route(
        _toy_panel(), ["2025-01", "2025-02"], normalize_by_month=False
    )
    assert rollup.iloc[0]["runtime_mean_min"] == pytest.approx(17.5)


def test_reduce_min_obs_filter_drops_sparse_cells() -> None:
    """Cells below min_obs_per_month are excluded before the rollup."""
    rollup = target.reduce_to_route(
        _toy_panel(), ["2025-01", "2025-02"], normalize_by_month=True, min_obs_per_month=2
    )
    # Only the 3-obs month survives -> mean is that month's mean.
    assert rollup.iloc[0]["runtime_mean_min"] == pytest.approx(20.0)
    assert rollup.iloc[0]["n_months"] == 1


def test_run_writes_panel_rollup_and_runlog(tmp_path: Path) -> None:
    """End-to-end run writes the panel, the rollup, and the run-log sidecar."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
        window_months=0,
    )
    results = target.run(cfg)
    assert set(results) == {"panel", "rollup"}

    assert (tmp_path / target.PANEL_FILENAME).exists()
    assert (tmp_path / target.ROLLUP_FILENAME).exists()
    assert (tmp_path / "route_runtime_tides_runlog.txt").exists()

    rollup = pd.read_csv(tmp_path / target.ROLLUP_FILENAME)
    assert "runtime_mean_min" in rollup.columns
    assert not rollup.empty
