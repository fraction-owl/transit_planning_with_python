import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the script directory to path to import the module
# We need to make sure we point to the directory containing the script
script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import trip_runtime_diagnostics_from_avl as target  # noqa: E402

FIXTURE_PATH = Path("tests/fixtures/trips_performed.csv")


def test_load_trip_files_tides_support() -> None:
    """Verify load_trip_files handles TIDES data correctly."""
    # 1. Load the data using the target function
    df = target.load_trip_files([FIXTURE_PATH])

    # 2. Assert basic columns are renamed
    assert "Route" in df.columns, "Route column missing (renamed from route_id)"
    assert "Direction" in df.columns, "Direction column missing (renamed from direction_id)"
    assert "TripID" in df.columns, "TripID column missing (renamed from trip_id_performed)"
    assert "Scheduled Start Time" in df.columns, "Scheduled Start Time column missing"
    assert "Actual Start Time" in df.columns, "Actual Start Time column missing"
    assert "trip_start_time" in df.columns, "trip_start_time column should be derived"

    # 3. Assert filtering
    # Fixture has 288 rows: 282 Scheduled + 6 Canceled (all trip_type = "In service").
    # Canceled trips are dropped; no Deadhead/Pullout/Pullin rows in this fixture.
    # Expected kept: 282 rows.

    assert len(df) == 282, f"Expected 282 rows, got {len(df)}"

    # Check Canceled is gone
    assert "TP20250101_202_0_03" not in df["TripID"].to_numpy(), (
        "Canceled trip should be filtered out"
    )

    # 4. Assert Time Extraction
    # TP20250102_101_0_00: schedule_trip_start 2025-01-02T05:58:00 -> 05:58
    row1 = df[df["TripID"] == "TP20250102_101_0_00"].iloc[0]
    assert row1["trip_start_time"] == "05:58", f"Expected 05:58, got {row1['trip_start_time']}"

    # 5. Assert Direction is string "0"/"1"
    # TP20250102_101_0_00 direction_id is 0
    assert str(row1["Direction"]) == "0", f"Expected direction '0', got {row1['Direction']}"

    # 6. Assert Is Tides flag
    assert "_is_tides" in df.columns
    assert df["_is_tides"].all()

    # 7. Check DateTime conversion
    assert pd.api.types.is_datetime64_any_dtype(df["Scheduled Start Time"]), (
        "Scheduled Start Time not datetime"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["Actual Start Time"]), (
        "Actual Start Time not datetime"
    )


def test_extract_trip_start_time_skip() -> None:
    """Verify extract_trip_start_time returns early if column exists."""
    df = pd.DataFrame({"trip_start_time": ["10:00"], "Trip": ["TRIP_1000"]})
    res = target.extract_trip_start_time(df)
    pd.testing.assert_frame_equal(df, res)


def test_route_column_candidates_skips_service_type_columns() -> None:
    """route_type/route_type_agency hold a mode name, never a route number."""
    header = ["service_date", "route_id", "route_type", "route_type_agency", "direction_id"]
    assert target._route_column_candidates(header) == ["route_id"]

    # Human-readable Route wins, then other non-ID columns, then ID-style ones.
    assert target._route_column_candidates(["RouteID", "routeName", "Route"]) == [
        "Route",
        "routeName",
        "RouteID",
    ]
    assert target._route_column_candidates(["stop_id", "route_type"]) == []


def test_discover_route_csvs_finds_tides_files(tmp_path: Path) -> None:
    """TIDES exports are discovered via route_id, not the route_type column."""
    (tmp_path / "trips.csv").write_bytes(FIXTURE_PATH.read_bytes())
    buckets = target._discover_route_csvs(tmp_path)
    assert set(buckets) == {"101", "202", "303"}
    assert all(len(files) == 1 for files in buckets.values())

    # The whitelist still narrows the result.
    assert set(target._discover_route_csvs(tmp_path, {"101"})) == {"101"}


def _filtered_fixture() -> pd.DataFrame:
    """One route/direction of the fixture, taken through the usual filters."""
    return (
        target.load_trip_files([FIXTURE_PATH])
        .pipe(target.extract_trip_start_time)
        .pipe(target.filter_date_range)
        .pipe(target.filter_routes, {"101"})
        .pipe(target.filter_holidays, target.EXCLUDE_DATES)
        .pipe(target.filter_service_day, "WEEKDAY")
        .pipe(target.add_deviation_cols)
        .pipe(target.add_otp_flag)
    )


def test_write_summary_table_has_runtime_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-start-time runtime statistics are joined onto the summary."""
    monkeypatch.setattr(target, "OUTPUT_DIR", tmp_path)
    summary = target.write_summary_table(_filtered_fixture())

    for col in ("runtime_mean_min", "runtime_median_min", "runtime_p85_min"):
        assert col in summary.columns, f"{col} missing from summary"
        assert summary[col].notna().any()
    assert (tmp_path / f"trip_summary_{(target.SERVICE_DAY_FILTER or 'ALL').upper()}.xlsx").exists()


def _b2b_summary(**overrides: object) -> pd.DataFrame:
    """Summary-shaped frame: 4 start times, 20 min apart, 30 min scheduled."""
    base = pd.DataFrame(
        {
            "trip_start_time": ["06:00", "06:20", "06:40", "07:00"],
            "n_events": [10, 10, 10, 10],
            "scheduled_runtime_mode": [30.0, 30.0, 30.0, 30.0],
            "runtime_median_min": [31.0, 32.0, 33.0, 34.0],
        }
    )
    return base.assign(**overrides)


def _b2b_rows(variants: list[str]) -> pd.DataFrame:
    """Row-level frame carrying each start time's stop pattern."""
    return pd.DataFrame(
        {
            target.TIME_COL_NAME: ["06:00", "06:20", "06:40", "07:00"],
            "pattern_id": variants,
        }
    )


def test_back_to_back_quiet_on_smooth_runtimes() -> None:
    """Neighbors that drift gently are never flagged."""
    pairs = target.flag_back_to_back_runtimes(_b2b_summary(), _b2b_rows(["P1"] * 4))
    assert len(pairs) == 3
    assert pairs["compared"].all()
    assert not pairs["back_to_back_flag"].any()


def test_back_to_back_flags_doubled_runtime() -> None:
    """A doubled runtime trips the ratio, minutes, and act/sched-ratio tests."""
    summary = _b2b_summary(runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    pairs = target.flag_back_to_back_runtimes(summary, _b2b_rows(["P1"] * 4))
    flagged = pairs.loc[pairs["back_to_back_flag"]]
    assert len(flagged) == 2  # the outlier flags against both of its neighbors
    assert flagged["flag_runtime_ratio"].all()
    assert flagged["flag_runtime_diff"].all()
    assert flagged["flag_act_sched_ratio"].all()


def test_back_to_back_pairs_only_within_a_pattern() -> None:
    """A short-turn is never compared against a full-length trip."""
    summary = _b2b_summary(
        runtime_median_min=[60.0, 25.0, 61.0, 26.0],
        scheduled_runtime_mode=[60.0, 25.0, 60.0, 25.0],
    )
    pairs = target.flag_back_to_back_runtimes(
        summary, _b2b_rows(["FULL", "SHORT", "FULL", "SHORT"])
    )
    assert len(pairs) == 2
    assert set(pairs["pattern_key"]) == {"FULL", "SHORT"}
    assert not pairs["back_to_back_flag"].any()


def test_back_to_back_accepts_legacy_variation_column() -> None:
    """Legacy AVL exports name the stop pattern 'Variation'."""
    summary = _b2b_summary(
        runtime_median_min=[60.0, 25.0, 61.0, 26.0],
        scheduled_runtime_mode=[60.0, 25.0, 60.0, 25.0],
    )
    rows = pd.DataFrame(
        {
            target.TIME_COL_NAME: ["06:00", "06:20", "06:40", "07:00"],
            "Variation": ["FULL", "SHORT", "FULL", "SHORT"],
        }
    )
    pairs = target.flag_back_to_back_runtimes(summary, rows)
    assert set(pairs["pattern_key"]) == {"FULL", "SHORT"}
    assert not pairs["back_to_back_flag"].any()


def test_back_to_back_skipped_without_pattern_column() -> None:
    """Without a pattern column the check refuses to guess."""
    summary = _b2b_summary(runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    rows = pd.DataFrame({target.TIME_COL_NAME: ["06:00", "06:20", "06:40", "07:00"]})
    assert target.flag_back_to_back_runtimes(summary, rows).empty
    opted_in = target.flag_back_to_back_runtimes(summary, rows, require_pattern=False)
    assert opted_in["back_to_back_flag"].any()


def test_back_to_back_gap_guard() -> None:
    """Trips hours apart are reported but not judged."""
    summary = _b2b_summary(
        trip_start_time=["06:00", "10:00", "14:00", "18:00"],
        runtime_median_min=[31.0, 32.0, 66.0, 33.0],
    )
    rows = _b2b_rows(["P1"] * 4).assign(
        **{target.TIME_COL_NAME: ["06:00", "10:00", "14:00", "18:00"]}
    )
    pairs = target.flag_back_to_back_runtimes(summary, rows)
    assert not pairs["compared"].any()
    assert not pairs["back_to_back_flag"].any()
    assert target.flag_back_to_back_runtimes(summary, rows, max_gap_min=0)[
        "back_to_back_flag"
    ].any()


def test_write_back_to_back_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flagged pairs are written; a clean run leaves no file behind."""
    monkeypatch.setattr(target, "OUTPUT_DIR", tmp_path)
    day_tag = (target.SERVICE_DAY_FILTER or "ALL").upper()
    out = tmp_path / f"back_to_back_runtime_flags_{day_tag}.csv"

    target.write_back_to_back_flags(
        target.flag_back_to_back_runtimes(_b2b_summary(), _b2b_rows(["P1"] * 4))
    )
    assert not out.exists(), "nothing flagged should mean no exception file"

    summary = _b2b_summary(runtime_median_min=[31.0, 32.0, 66.0, 33.0])
    target.write_back_to_back_flags(
        target.flag_back_to_back_runtimes(summary, _b2b_rows(["P1"] * 4))
    )
    assert out.exists()
    written = pd.read_csv(out)
    assert len(written) == 2
    assert written["back_to_back_flag"].all()


def test_write_run_log_sidecar(tmp_path: Path) -> None:
    """The sidecar records the run and the verbatim CONFIGURATION block."""
    assert target.write_run_log(tmp_path, ["Rows retained:      42"])

    log = tmp_path / "trip_runtime_diagnostics_runlog.txt"
    assert log.exists()

    text = log.read_text(encoding="utf-8")
    assert "Run timestamp:" in text
    assert "Rows retained:      42" in text
    # Config constants are reproduced verbatim so a run can be reconstructed.
    assert "B2B_RUNTIME_RATIO: Final[float] = 2.0" in text
    assert f"B2B_MAX_GAP_MIN: Final[float] = {target.B2B_MAX_GAP_MIN}" in text
    # The marker lines themselves are excluded from the captured block.
    assert target.CONFIG_BEGIN_MARKER not in text
