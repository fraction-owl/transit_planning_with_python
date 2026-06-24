"""Tests for otp_by_route: the windowed route-level OTP rollup."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import otp_by_route as target  # noqa: E402
import otp_monthly_tides as otp_tides  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


def _write_panel(path: Path) -> Path:
    """Write a synthetic otp_monthly_processed-style panel and return its path."""
    pd.DataFrame(
        {
            "level": ["route", "route", "overall"],
            "group": ["A", "A", "ALL"],
            "route_id": ["A", "A", ""],
            "month": ["2025-01", "2025-02", "2025-01"],
            "on_time": [8, 30, 38],
            "evaluated": [10, 30, 40],
        }
    ).to_csv(path, index=False)
    return path


def test_load_otp_panel_filters_to_level_and_coerces(tmp_path: Path) -> None:
    """Loading keeps only the requested level with numeric counts."""
    panel = target.load_otp_panel(_write_panel(tmp_path / "p.csv"), level="route")
    assert set(panel["route_id"].unique()) == {"A"}
    assert list(panel.columns) == ["route_id", "month", "on_time", "evaluated"]
    assert panel["evaluated"].dtype.kind in "fi"


def test_load_otp_panel_missing_column_raises(tmp_path: Path) -> None:
    """A panel missing a required column fails loudly."""
    p = tmp_path / "bad.csv"
    pd.DataFrame({"level": ["route"], "route_id": ["A"], "month": ["2025-01"]}).to_csv(
        p, index=False
    )
    with pytest.raises(ValueError, match="missing required column"):
        target.load_otp_panel(p)


def test_load_otp_panel_missing_file_raises(tmp_path: Path) -> None:
    """A missing panel file points the user at otp_monthly_tides."""
    with pytest.raises(FileNotFoundError, match="otp_monthly"):
        target.load_otp_panel(tmp_path / "nope.csv")


def test_select_window_trailing_and_end_month() -> None:
    """The window keeps the most recent N months, respecting END_MONTH."""
    months = ["2025-01", "2025-02", "2025-03"]
    assert target.select_window(months, "", 2) == ["2025-02", "2025-03"]
    assert target.select_window(months, "2025-02", 5) == ["2025-01", "2025-02"]
    assert target.select_window(months, "", 0) == months


def test_select_window_supports_24_month_cadence() -> None:
    """A 24-month (2-year) window is a supported trailing-window length."""
    months = (
        [f"2024-{m:02d}" for m in range(1, 13)]
        + [f"2025-{m:02d}" for m in range(1, 13)]
        + [f"2026-{m:02d}" for m in range(1, 7)]
    )  # 30 months total
    window = target.select_window(months, "", 24)
    assert len(window) == 24
    assert window[0] == "2024-07"
    assert window[-1] == "2026-06"


def _toy_panel() -> pd.DataFrame:
    """Route A: a light 80%-month (10 eval) and a perfect month (30 eval)."""
    return pd.DataFrame(
        {
            "route_id": ["A", "A"],
            "month": ["2025-01", "2025-02"],
            "on_time": [8.0, 30.0],
            "evaluated": [10.0, 30.0],
        }
    )


def test_reduce_normalized_weights_months_equally() -> None:
    """Normalized rollup averages monthly %: (80 + 100) / 2 = 90."""
    rollup = target.reduce_to_route(_toy_panel(), ["2025-01", "2025-02"], normalize_by_month=True)
    row = rollup.iloc[0]
    assert row["pct_on_time"] == pytest.approx(90.0)
    assert row["n_months"] == 2
    assert row["evaluated"] == 40


def test_reduce_naive_pools_counts() -> None:
    """Naive rollup pools counts: 38 / 40 * 100 = 95."""
    rollup = target.reduce_to_route(_toy_panel(), ["2025-01", "2025-02"], normalize_by_month=False)
    assert rollup.iloc[0]["pct_on_time"] == pytest.approx(95.0)


def test_run_writes_rollup_and_runlog(tmp_path: Path) -> None:
    """End-to-end run writes otp_by_route.csv and the run-log sidecar."""
    panel_path = _write_panel(tmp_path / "otp_monthly_processed.csv")
    cfg = target.Config(
        otp_processed_path=panel_path,
        output_dir=tmp_path / "out",
        window_months=0,
    )
    rollup = target.run(cfg)
    assert (tmp_path / "out" / target.ROLLUP_FILENAME).exists()
    assert (tmp_path / "out" / "otp_by_route_runlog.txt").exists()
    assert set(rollup["route_id"]) == {"A"}


def test_consumes_real_otp_monthly_tides_output(tmp_path: Path) -> None:
    """Integration: roll up the panel that otp_monthly_tides actually writes."""
    otp_tides.run(
        otp_tides.Config(
            stop_visits_path=STOP_VISITS,
            trips_performed_path=TRIPS_PERFORMED,
            output_dir=tmp_path,
        )
    )
    processed = tmp_path / otp_tides.PROCESSED_FILENAME
    assert processed.exists()

    rollup = target.run(
        target.Config(
            otp_processed_path=processed,
            output_dir=tmp_path / "rollup",
            window_months=0,
        )
    )
    assert not rollup.empty
    assert {"route_id", "pct_on_time"} <= set(rollup.columns)
    assert (rollup["pct_on_time"].between(0, 100)).all()
