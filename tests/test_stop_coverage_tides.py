"""Tests for stop_coverage_tides: GTFS expected trip-visits vs TIDES stop_visits."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import stop_coverage_tides as target  # noqa: E402

WEEKDAYS = list(pd.date_range("2025-01-06", "2025-01-17", freq="B"))  # 10 weekdays


def _write_gtfs(tmp_path: Path) -> Path:
    """R1: T1 (S1,S2,S3), T2 (S1,S2); R2: T4 (S3). All weekday service."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "WKDY,1,1,1,1,1,0,0,20250106,20250119\n",
        encoding="utf-8",
    )
    (gtfs / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id\nR1,WKDY,T1,0\nR1,WKDY,T2,0\nR2,WKDY,T4,0\n",
        encoding="utf-8",
    )
    (gtfs / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,timepoint\n"
        "T1,07:00:00,07:00:00,S1,1,1\n"
        "T1,07:05:00,07:05:00,S2,2,0\n"
        "T1,07:10:00,07:10:00,S3,3,1\n"
        "T2,08:00:00,08:00:00,S1,1,1\n"
        "T2,08:05:00,08:05:00,S2,2,0\n"
        "T4,09:00:00,09:00:00,S3,1,1\n",
        encoding="utf-8",
    )
    return gtfs


def _write_tides(tmp_path: Path) -> tuple[Path, Path]:
    """Write synthetic TIDES tables exercising every visit outcome.

    T1 is performed daily with visits at S1+S2 (S3 never visited; two S2
    visits lack actuals); T2 is performed on the first two days with S1
    observed and S2 Skipped; T4 is never performed.
    """
    trip_rows = []
    visit_rows = []
    for i, day in enumerate(WEEKDAYS):
        pid = f"P_T1_{day.date()}"
        trip_rows.append((day, pid, "T1", "Scheduled", "In service"))
        visit_rows.append((day, pid, "S1", f"{day.date()}T07:01:00", "Scheduled"))
        # On the first two days, T1's S2 visit is emitted without an actual
        # timestamp (an AVL dropout), pushing S2 below the 50% threshold.
        s2_actual = "" if i < 2 else f"{day.date()}T07:06:00"
        visit_rows.append((day, pid, "S2", s2_actual, "Scheduled"))
    for day in WEEKDAYS[:2]:
        pid = f"P_T2_{day.date()}"
        trip_rows.append((day, pid, "T2", "Scheduled", "In service"))
        visit_rows.append((day, pid, "S1", f"{day.date()}T08:01:00", "Scheduled"))
        visit_rows.append((day, pid, "S2", "", "Skipped"))
    # One orphan visit whose performed trip is unknown: dropped by the bridge.
    visit_rows.append(
        (WEEKDAYS[0], "P_ORPHAN", "S1", f"{WEEKDAYS[0].date()}T10:00:00", "Scheduled")
    )

    trips = pd.DataFrame(
        trip_rows,
        columns=[
            "service_date",
            "trip_id_performed",
            "trip_id_scheduled",
            "schedule_relationship",
            "trip_type",
        ],
    )
    visits = pd.DataFrame(
        visit_rows,
        columns=[
            "service_date",
            "trip_id_performed",
            "stop_id",
            "actual_departure_time",
            "schedule_relationship",
        ],
    )
    trips_path = tmp_path / "trips_performed.csv"
    visits_path = tmp_path / "stop_visits.csv"
    trips.to_csv(trips_path, index=False)
    visits.to_csv(visits_path, index=False)
    return visits_path, trips_path


def _cfg(tmp_path: Path, **overrides: object) -> "target.Config":
    visits_path, trips_path = _write_tides(tmp_path)
    base = {
        "gtfs_path": _write_gtfs(tmp_path),
        "stop_visits_path": visits_path,
        "trips_performed_path": trips_path,
        "output_dir": tmp_path / "out",
    }
    base.update(overrides)
    return target.Config(**base)


def test_bridge_drops_orphan_visits(tmp_path: Path) -> None:
    """Visits with no in-service trips_performed match are dropped."""
    visits_path, trips_path = _write_tides(tmp_path)
    visits = target.load_stop_visits(visits_path)
    trips = target.load_trips_performed(trips_path)
    bridged = target.bridge_visits_to_schedule(visits, trips)
    assert len(bridged) == len(visits) - 1
    assert "trip_id_scheduled" in bridged.columns


def test_summarize_visit_records_flags() -> None:
    """Observed requires an actual timestamp; Skipped rows are marked."""
    df = pd.DataFrame(
        {
            "service_date": pd.to_datetime(["2025-01-06", "2025-01-06"]),
            "trip_id_scheduled": ["T2", "T2"],
            "stop_id": ["S1", "S2"],
            "actual_arrival_time": [pd.NaT, pd.NaT],
            "actual_departure_time": [pd.Timestamp("2025-01-06 08:01"), pd.NaT],
            "schedule_relationship": ["Scheduled", "Skipped"],
        }
    )
    out = target.summarize_visit_records(df).set_index("stop_id")
    assert bool(out.loc["S1", "observed"])
    assert not bool(out.loc["S1", "skipped"])
    assert not bool(out.loc["S2", "observed"])
    assert bool(out.loc["S2", "skipped"])


def test_run_end_to_end_outcomes_and_flags(tmp_path: Path) -> None:
    """Coverage, outcome decomposition, and flags match the constructed data."""
    cfg = _cfg(tmp_path)
    summary = target.run(cfg).set_index("stop_id")

    # S1: T1 observed 10/10, T2 observed 2/10 (8 days trip unrecorded).
    s1 = summary.loc["S1"]
    assert s1["expected_visits"] == 20
    assert s1["observed_visits"] == 12
    assert s1["pct_observed"] == 60.0
    assert s1["n_trip_unrecorded"] == 8
    assert not bool(s1["flag_low_coverage"])

    # S2: T1 observed 8 + 2 without actuals; T2 skipped 2 + unrecorded 8
    # -> 40% observed, flagged.
    s2 = summary.loc["S2"]
    assert s2["expected_visits"] == 20
    assert s2["observed_visits"] == 8
    assert s2["pct_observed"] == 40.0
    assert s2["n_skipped"] == 2
    assert s2["n_visit_without_actual"] == 2
    assert bool(s2["flag_low_coverage"])
    assert s2["flag_reason"] == "low coverage"

    # S3: never visited although T1 ran daily -> invisible, with the
    # trip-performed/visit-missing decomposition intact.
    s3 = summary.loc["S3"]
    assert s3["expected_visits"] == 20
    assert s3["observed_visits"] == 0
    assert s3["n_visit_missing"] == 10  # T1 performed, no row at S3
    assert s3["n_trip_unrecorded"] == 10  # T4 never recorded
    assert bool(s3["flag_invisible"])
    assert s3["flag_reason"] == "invisible (never observed)"

    detail = pd.read_csv(cfg.output_dir / target.STOP_ROUTE_DAY_DETAIL_FILENAME)
    s3_r1 = detail.loc[(detail["stop_id"] == "S3") & (detail["route_id"] == "R1")].iloc[0]
    assert s3_r1["n_visit_missing"] == 10
    s3_r2 = detail.loc[(detail["stop_id"] == "S3") & (detail["route_id"] == "R2")].iloc[0]
    assert s3_r2["n_trip_unrecorded"] == 10

    runlog = (cfg.output_dir / "stop_coverage_tides_runlog.txt").read_text(encoding="utf-8")
    assert "CONFIGURATION (verbatim from source)" in runlog
    assert "Visit join rate" in runlog


def test_run_timepoints_only_restricts_expectation(tmp_path: Path) -> None:
    """TIMEPOINTS_ONLY drops non-timepoint stop_times rows (S2) from scope."""
    cfg = _cfg(tmp_path, timepoints_only=True)
    summary = target.run(cfg).set_index("stop_id")
    assert "S2" not in summary.index
    assert summary.loc["S1", "expected_visits"] == 20


def test_warn_if_timepoint_only_export(caplog: pytest.LogCaptureFixture) -> None:
    """An all-timepoint export with TIMEPOINTS_ONLY off draws a warning."""
    visits = pd.DataFrame({"timepoint": ["TRUE", "TRUE"]})
    with caplog.at_level(logging.WARNING):
        target.warn_if_timepoint_only_export(visits, timepoints_only=False)
    assert any("TIMEPOINTS_ONLY" in message for message in caplog.messages)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        target.warn_if_timepoint_only_export(visits, timepoints_only=True)
    assert not caplog.messages
