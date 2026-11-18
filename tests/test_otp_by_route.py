"""Tests for otp_by_route: the windowed route-level OTP rollup."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import otp_by_route as target  # noqa: E402

import scripts.operations_tools.otp_runtime_from_tides as exporter  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


def _write_table(path: Path) -> Path:
    """Write a synthetic otp_by_route_and_month-style table and return its path."""
    pd.DataFrame(
        {
            "day_type": ["Weekday", "Saturday", "Weekday"],
            "route_id": ["A", "A", "A"],
            "month": ["2025-01", "2025-01", "2025-02"],
            "on_time": [6, 2, 30],
            "evaluated": [7, 3, 30],
            "pct_on_time": [85.7, 66.7, 100.0],
        }
    ).to_csv(path, index=False)
    return path


def test_load_otp_table_pools_day_types_and_coerces(tmp_path: Path) -> None:
    """Loading sums every day type per (route, month) with numeric counts."""
    panel = target.load_otp_table(_write_table(tmp_path / "p.csv"))
    assert list(panel.columns) == ["route_id", "month", "on_time", "evaluated"]
    assert panel["on_time"].tolist() == [8, 30]
    assert panel["evaluated"].tolist() == [10, 30]
    assert panel["evaluated"].dtype.kind in "fi"


def test_load_otp_table_keeps_only_the_chosen_day_types(tmp_path: Path) -> None:
    """DAY_TYPES_TO_INCLUDE limits the pooled day types; none matching fails loudly."""
    path = _write_table(tmp_path / "p.csv")
    weekday = target.load_otp_table(path, ["Weekday"])
    assert weekday["evaluated"].tolist() == [7, 30]
    with pytest.raises(ValueError, match="No row"):
        target.load_otp_table(path, ["Sunday"])


def test_load_otp_table_missing_column_raises(tmp_path: Path) -> None:
    """A table missing a required column fails loudly."""
    p = tmp_path / "bad.csv"
    pd.DataFrame({"day_type": ["Weekday"], "route_id": ["A"], "month": ["2025-01"]}).to_csv(
        p, index=False
    )
    with pytest.raises(ValueError, match="missing required column"):
        target.load_otp_table(p)


def test_load_otp_table_missing_file_raises(tmp_path: Path) -> None:
    """A missing table points the user at otp_runtime_from_tides."""
    with pytest.raises(FileNotFoundError, match="otp_runtime_from_tides"):
        target.load_otp_table(tmp_path / "nope.csv")


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
    table_path = _write_table(tmp_path / "otp_by_route_and_month.csv")
    cfg = target.Config(
        otp_monthly_path=table_path,
        output_dir=tmp_path / "out",
        window_months=0,
    )
    rollup = target.run(cfg)
    assert (tmp_path / "out" / target.ROLLUP_FILENAME).exists()
    assert (tmp_path / "out" / "otp_by_route_runlog.txt").exists()
    assert set(rollup["route_id"]) == {"A"}
    assert rollup.iloc[0]["pct_on_time"] == pytest.approx(90.0)  # (80 + 100) / 2


def test_main_guards_the_placeholder_path() -> None:
    assert target.main([]) == 2


def test_consumes_real_otp_runtime_from_tides_output(tmp_path: Path) -> None:
    """Integration: roll up the route x month table otp_runtime_from_tides actually writes."""
    periods = exporter.build_time_periods(
        exporter.specs_from_mapping(exporter.TIME_PERIODS), "defaults"
    )
    exporter.run(
        exporter.RunSettings(
            stop_visits_path=STOP_VISITS,
            trips_performed_path=TRIPS_PERFORMED,
            output_dir=tmp_path,
            periods=periods,
            measures=["otp"],
            levels=["route"],
            time_cuts=["month"],
            export_stop_flags=False,
            export_charts=False,
        )
    )
    table = tmp_path / "otp" / exporter.grain_filename("otp", "route", "month")
    assert table.exists()

    rollup = target.run(
        target.Config(
            otp_monthly_path=table,
            output_dir=tmp_path / "rollup",
            window_months=0,
            normalize_by_month=False,
        )
    )
    assert not rollup.empty
    assert {"route_id", "pct_on_time"} <= set(rollup.columns)
    assert (rollup["pct_on_time"].between(0, 100)).all()
    # Pooled counts equal the exporter's route rows summed over day types and months.
    source = pd.read_csv(table, dtype={"route_id": str})
    expected = source.groupby("route_id")["evaluated"].sum()
    assert rollup.set_index("route_id")["evaluated"].to_dict() == expected.to_dict()
