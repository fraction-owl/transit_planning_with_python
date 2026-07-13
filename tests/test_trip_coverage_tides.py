"""Tests for trip_coverage_tides: GTFS scheduled trips vs TIDES trips_performed."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import trip_coverage_tides as target  # noqa: E402

WEEKDAYS = list(pd.date_range("2025-01-06", "2025-01-17", freq="B"))  # 10 weekdays
SATURDAYS = [pd.Timestamp("2025-01-11"), pd.Timestamp("2025-01-18")]


def _write_gtfs(tmp_path: Path, calendar_dates_rows: list[tuple] | None = None) -> Path:
    """Write a minimal GTFS feed: R1 (T1, T2 weekday), R2 (T3 Saturday, T4 weekday)."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "WKDY,1,1,1,1,1,0,0,20250106,20250119\n"
        "SAT,0,0,0,0,0,1,0,20250106,20250119\n",
        encoding="utf-8",
    )
    if calendar_dates_rows:
        lines = ["service_id,date,exception_type"]
        lines += [",".join(row) for row in calendar_dates_rows]
        (gtfs / "calendar_dates.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (gtfs / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id\n"
        "R1,WKDY,T1,0\n"
        "R1,WKDY,T2,0\n"
        "R2,SAT,T3,0\n"
        "R2,WKDY,T4,0\n",
        encoding="utf-8",
    )
    return gtfs


def _write_trips_performed(tmp_path: Path) -> Path:
    """T1 performed daily; T2 performed twice + canceled once; T4 never; TX unknown."""
    rows = []
    for day in WEEKDAYS:
        rows.append((day, f"P_T1_{day.date()}", "T1", "Scheduled", "In service"))
    for day in WEEKDAYS[:2]:
        rows.append((day, f"P_T2_{day.date()}", "T2", "Scheduled", "In service"))
    rows.append((WEEKDAYS[2], f"P_T2_{WEEKDAYS[2].date()}", "T2", "Canceled", "In service"))
    rows.append((WEEKDAYS[0], "P_TX", "TX", "Scheduled", "In service"))
    df = pd.DataFrame(
        rows,
        columns=[
            "service_date",
            "trip_id_performed",
            "trip_id_scheduled",
            "schedule_relationship",
            "trip_type",
        ],
    )
    df["route_id"] = "R" + df["trip_id_scheduled"].str.slice(1, 2)
    path = tmp_path / "trips_performed.csv"
    df.to_csv(path, index=False)
    return path


def _cfg(tmp_path: Path, **overrides: object) -> "target.Config":
    base = {
        "gtfs_path": _write_gtfs(tmp_path),
        "trips_performed_path": _write_trips_performed(tmp_path),
        "output_dir": tmp_path / "out",
    }
    base.update(overrides)
    return target.Config(**base)


def test_summarize_performed_records_splits_recorded_and_performed() -> None:
    """A canceled-only day is recorded but not performed."""
    df = pd.DataFrame(
        {
            "service_date": pd.to_datetime(["2025-01-06", "2025-01-07"]),
            "trip_id_scheduled": ["T2", "T2"],
            "schedule_relationship": ["Canceled", "Scheduled"],
            "trip_type": ["In service", "In service"],
        }
    )
    out = target.summarize_performed_records(df).set_index("service_date")
    assert bool(out.loc[pd.Timestamp("2025-01-06"), "recorded"])
    assert not bool(out.loc[pd.Timestamp("2025-01-06"), "performed"])
    assert bool(out.loc[pd.Timestamp("2025-01-07"), "performed"])


def test_load_trips_performed_missing_column_raises(tmp_path: Path) -> None:
    """A file without trip_id_scheduled fails loudly."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"service_date": ["2025-01-06"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="trip_id_scheduled"):
        target.load_trips_performed(path)


def test_resolve_window_intersects_feed_and_observed() -> None:
    """The automatic window is feed validity intersected with observed dates."""
    calendar = pd.DataFrame(
        {"service_id": ["W"], "start_date": ["20250101"], "end_date": ["20250131"]}
    )
    observed = pd.Series(pd.to_datetime(["2025-01-10", "2025-02-15"]))
    start, end = target.resolve_window(observed, calendar, None)
    assert start == pd.Timestamp("2025-01-10")
    assert end == pd.Timestamp("2025-01-31")


def test_resolve_window_no_overlap_raises() -> None:
    """A feed that predates the data entirely is an actionable error."""
    calendar = pd.DataFrame(
        {"service_id": ["W"], "start_date": ["20240101"], "end_date": ["20240131"]}
    )
    observed = pd.Series(pd.to_datetime(["2025-01-10"]))
    with pytest.raises(ValueError, match="does not overlap"):
        target.resolve_window(observed, calendar, None)


def test_run_end_to_end_flags_and_outputs(tmp_path: Path) -> None:
    """The pipeline flags chronic and total gaps and writes both tables + runlog."""
    cfg = _cfg(tmp_path)
    summary = target.run(cfg).set_index("trip_id")

    assert summary.loc["T1", "pct_days_performed"] == 100.0
    assert not bool(summary.loc["T1", "flag_low_coverage"])

    assert summary.loc["T2", "scheduled_days"] == 10
    assert summary.loc["T2", "days_recorded"] == 3
    assert summary.loc["T2", "days_performed"] == 2
    assert summary.loc["T2", "days_canceled_only"] == 1
    assert summary.loc["T2", "flag_reason"] == "low coverage"

    assert summary.loc["T4", "days_recorded"] == 0
    assert summary.loc["T4", "flag_reason"] == "never recorded"

    # Only one Saturday falls inside the window (observed data ends Jan 17),
    # so T3 sits under MIN_SCHEDULED_DAYS and is never judged.
    assert summary.loc["T3", "scheduled_days"] == 1
    assert not bool(summary.loc["T3", "flag_low_coverage"])

    out = cfg.output_dir
    route_day = pd.read_csv(out / target.ROUTE_DAY_COVERAGE_FILENAME)
    r1 = route_day.loc[(route_day["route_id"] == "R1") & (route_day["day_type"] == "Weekday")].iloc[
        0
    ]
    assert r1["scheduled_instances"] == 20
    assert r1["instances_performed"] == 12
    assert r1["pct_instances_performed"] == 60.0
    assert r1["n_trips_flagged"] == 1

    assert (out / target.TRIP_COVERAGE_FILENAME).exists()
    runlog = (out / "trip_coverage_tides_runlog.txt").read_text(encoding="utf-8")
    assert "CONFIGURATION (verbatim from source)" in runlog
    assert "LOW_COVERAGE_FLAG_PCT" in runlog
    assert "Trip-ID join rate" in runlog


def test_run_excludes_exception_dates(tmp_path: Path) -> None:
    """A calendar_dates holiday removal drops that date from the expectation."""
    gtfs = _write_gtfs(tmp_path, calendar_dates_rows=[("WKDY", "20250113", "2")])
    cfg = target.Config(
        gtfs_path=gtfs,
        trips_performed_path=_write_trips_performed(tmp_path),
        output_dir=tmp_path / "out",
    )
    summary = target.run(cfg).set_index("trip_id")
    assert summary.loc["T1", "scheduled_days"] == 9


def test_run_manual_exclude_dates(tmp_path: Path) -> None:
    """EXCLUDE_DATES drops the named date from both sides of the comparison."""
    cfg = _cfg(tmp_path, exclude_dates=("2025-01-06",))
    summary = target.run(cfg).set_index("trip_id")
    assert summary.loc["T1", "scheduled_days"] == 9
    assert summary.loc["T1", "days_performed"] == 9
