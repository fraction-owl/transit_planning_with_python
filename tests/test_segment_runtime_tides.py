"""Tests for segment_runtime_tides using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import segment_runtime_tides as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


@pytest.fixture()
def joined() -> pd.DataFrame:
    """Stop visits joined with trip attributes."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    return target.join_trip_attributes(sv, tp)


def test_build_segments_basic(joined: pd.DataFrame) -> None:
    """Segments carry ordered stop pairs, route context, and runtime columns."""
    seg = target.build_segments(joined, timepoints_only=True)
    assert not seg.empty
    for col in (
        "route_id",
        "direction_id",
        "segment",
        "from_stop_id",
        "to_stop_id",
        "actual_runtime_min",
        "scheduled_runtime_min",
        "diff_min",
    ):
        assert col in seg.columns
    # diff is actual minus scheduled.
    sample = seg.dropna(subset=["actual_runtime_min", "scheduled_runtime_min"]).iloc[0]
    assert sample["diff_min"] == pytest.approx(
        sample["actual_runtime_min"] - sample["scheduled_runtime_min"]
    )
    # Segment label matches its endpoint columns.
    assert sample["segment"] == f"{sample['from_stop_id']} -> {sample['to_stop_id']}"


def test_build_segments_skips_skipped_stops(joined: pd.DataFrame) -> None:
    """A Skipped stop is not used as a segment endpoint."""
    seg = target.build_segments(joined, timepoints_only=True)
    # Stop 1011 on TP20250102_101_0_00 is Skipped in the fixture, so it should
    # not appear as an endpoint of any segment for that trip.
    trip_seg = seg.loc[seg["trip_id_performed"] == "TP20250102_101_0_00"]
    endpoints = set(trip_seg["from_stop_id"]) | set(trip_seg["to_stop_id"])
    assert "1011" not in endpoints


def test_compute_block_recovery() -> None:
    """Recovery is the next trip's scheduled start minus this trip's end."""
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    rec = target.compute_block_recovery(tp)
    assert {"trip_id_performed", "recovery_after_min"} == set(rec.columns)
    # At least one finite recovery value should exist (blocks chain trips).
    assert rec["recovery_after_min"].notna().any()
    # Build a tiny synthetic block to check the arithmetic exactly.
    synth = pd.DataFrame(
        {
            "trip_id_performed": ["A", "B"],
            "block_id": ["BLK", "BLK"],
            "schedule_relationship": ["Scheduled", "Scheduled"],
            "schedule_trip_start": pd.to_datetime(["2025-01-02T06:00:00", "2025-01-02T07:00:00"]),
            "schedule_trip_end": pd.to_datetime(["2025-01-02T06:45:00", "2025-01-02T07:45:00"]),
        }
    )
    rec2 = target.compute_block_recovery(synth).set_index("trip_id_performed")
    assert rec2.loc["A", "recovery_after_min"] == pytest.approx(15.0)
    assert pd.isna(rec2.loc["B", "recovery_after_min"])


def test_summarize_segments_columns(joined: pd.DataFrame) -> None:
    """Summary exposes median/scheduled/diff/recovery per segment."""
    seg = target.build_segments(joined, timepoints_only=True)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    rec = target.compute_block_recovery(tp)
    summary = target.summarize_segments(seg, rec, min_obs=1)
    for col in (
        "route_id",
        "direction_id",
        "segment",
        "n_obs",
        "actual_median_min",
        "scheduled_min",
        "diff_min",
        "recovery_after_min",
    ):
        assert col in summary.columns
    assert (summary["n_obs"] >= 1).all()


def test_run_writes_long_summary_and_pivots(tmp_path: Path) -> None:
    """End-to-end run writes the long table, summary, and per-route pivots."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
    )
    seg = target.run(cfg)
    assert not seg.empty

    assert (tmp_path / target.LONG_FILENAME).exists()
    assert (tmp_path / "segment_runtime_summary.csv").exists()
    pivots = list((tmp_path / "pivots").glob("segment_runtime_*.csv"))
    assert pivots, "expected at least one human pivot"
