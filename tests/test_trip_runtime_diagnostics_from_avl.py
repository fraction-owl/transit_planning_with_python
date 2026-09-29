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


def _route_trips() -> pd.DataFrame:
    """Twenty days of two start times, each with one extreme runtime on day 20."""
    rows = []
    for day in range(1, 21):
        for hhmm, runtime in (("06:00", 30), ("07:00", 40)):
            start = pd.Timestamp(f"2025-03-{day:02d} {hhmm}")
            extra = 60 if day == 20 else 0
            rows.append(
                {
                    "Route": "101",
                    "Direction": "NORTHBOUND",
                    "TripID": f"T{hhmm}",
                    "trip_start_time": hhmm,
                    "Scheduled Start Time": start,
                    "Scheduled Finish Time": start + pd.Timedelta(minutes=runtime),
                    "Actual Start Time": start,
                    "Actual Finish Time": start + pd.Timedelta(minutes=runtime + day % 3 + extra),
                }
            )
    return pd.DataFrame(rows).pipe(target.add_deviation_cols).pipe(target.add_otp_flag)


@pytest.fixture()
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(target, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(target, "PLOTS_DIR", tmp_path / "plots")
    return tmp_path


def test_write_summary_table_has_runtime_stats(output_dir: Path) -> None:
    summary = target.write_summary_table(_route_trips())
    assert summary["trip_start_time"].tolist() == ["06:00", "07:00"]
    for col in ("runtime_mean_min", "runtime_median_min", "runtime_p85_min"):
        assert summary[col].notna().all()
    assert (output_dir / f"trip_summary_{target._day_tag()}.xlsx").exists()


def test_retained_and_excluded_exports_do_not_overlap(output_dir: Path) -> None:
    df = _route_trips()
    target.export_trimmed_outliers(df)
    target.write_row_level(df)
    tag = target._day_tag()
    excluded = pd.read_csv(output_dir / f"events_excluded_{tag}.csv")
    retained = pd.read_csv(output_dir / f"events_retained_{tag}.csv")
    key = ["TripID", "Scheduled Start Time"]
    assert not excluded.empty
    assert excluded[key].merge(retained[key], on=key).empty
    assert len(excluded) + len(retained) == len(df)


def test_filter_date_range_keeps_all_of_date_end() -> None:
    df = pd.DataFrame(
        {
            "Scheduled Start Time": [
                target.DATE_START - pd.Timedelta(minutes=1),
                target.DATE_END + pd.Timedelta(hours=18),
                target.DATE_END + pd.Timedelta(days=1),
            ]
        }
    )
    kept = target.filter_date_range(df)
    assert kept["Scheduled Start Time"].tolist() == [target.DATE_END + pd.Timedelta(hours=18)]


def test_suggest_time_bands_handles_fewer_than_two_start_times() -> None:
    one = pd.DataFrame({"trip_start_time": ["06:00"], "runtime_p85_min": [30.0]})
    bands = target.suggest_time_bands(one)
    assert bands[["band_id", "start_time", "end_time", "n_tokens"]].to_numpy().tolist() == [
        [1, "06:00", "06:00", 1]
    ]
    assert target.suggest_time_bands(one.iloc[0:0]).empty
