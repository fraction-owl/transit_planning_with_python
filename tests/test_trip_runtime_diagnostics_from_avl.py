import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the script directory to path to import the module
# We need to make sure we point to the directory containing the script
script_dir = Path("scripts/conversion_tools").resolve()
sys.path.append(str(script_dir))

import trip_runtime_diagnostics_from_avl as target  # noqa: E402

import scripts.conversion_tools.convert_to_tides_stop_visits as stop_converter  # noqa: E402
import scripts.conversion_tools.convert_to_tides_trips_performed as trip_converter  # noqa: E402

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
    # Fixture has 171 rows: 96 "In service" (92 Scheduled + 4 Canceled, 12 of the
    # Scheduled from the 2026-01-01 holiday pass) and 75 nonrevenue
    # Pullout/Pullin/Deadhead rows. Canceled and nonrevenue trips are dropped.
    # Expected kept: 92 rows.

    assert len(df) == 92, f"Expected 92 rows, got {len(df)}"

    # Check Canceled is gone
    assert "TP20260202_DC_R30_weekday_0025" not in df["TripID"].to_numpy(), (
        "Canceled trip should be filtered out"
    )

    # 4. Assert Time Extraction
    # TP20260105_DC_R10_weekday_0032: schedule_trip_start 2026-01-05T21:00:00 -> 21:00
    row1 = df[df["TripID"] == "TP20260105_DC_R10_weekday_0032"].iloc[0]
    assert row1["trip_start_time"] == "21:00", f"Expected 21:00, got {row1['trip_start_time']}"

    # 5. Assert Direction is string "0"/"1"
    # TP20260105_DC_R10_weekday_0032 direction_id is 0
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


# ---------------------------------------------------------------------------
# CLEVER extracts converted to TIDES by the repo's own converters
# ---------------------------------------------------------------------------


def _clever_runtime_row(**overrides: str) -> dict[str, str]:
    """One CLEVER Event Runtime Analysis row: Route 301 on Monday 2026-11-16."""
    row = {
        "Vehicle": "7001",
        "Route": "301 - Telegraph Rd",
        "Direction": "NORTHBOUND",
        "Block": "B1",
        "TripID": "T100",
        "Trip Type": "Revenue",
        "Scheduled Start Time": "11/16/2026 6:00:00 AM",
        "Scheduled Finish Time": "11/16/2026 6:40:00 AM",
        "Actual Start Time": "11/16/2026 6:01:00 AM",
        "Actual Finish Time": "11/16/2026 6:43:00 AM",
        "Operator": "OP1",
        "Last Stop": "S9",
    }
    row.update(overrides)
    return row


def _write_tides_trips(folder: Path, rows: list[dict[str, str]]) -> Path:
    path = folder / "trips_performed.csv"
    trip_converter.convert_to_tides(pd.DataFrame(rows)).to_csv(path, index=False)
    return path


def test_discover_route_csvs_reads_route_id_from_trip_converter_output(tmp_path: Path) -> None:
    """route_type is blank in converter output; discovery must use route_id."""
    path = _write_tides_trips(tmp_path, [_clever_runtime_row()])
    assert dict(target._discover_route_csvs(tmp_path)) == {"301": [path]}


def test_otp_counts_only_trips_with_known_start_deviation(output_dir: Path) -> None:
    start = pd.Timestamp("2026-11-16 06:00")
    df = (
        pd.DataFrame(
            {
                "trip_start_time": ["06:00", "06:00", "07:00"],
                "Scheduled Start Time": [start, start, start + pd.Timedelta(hours=1)],
                "Scheduled Finish Time": [
                    start + pd.Timedelta(minutes=40),
                    start + pd.Timedelta(minutes=40),
                    start + pd.Timedelta(minutes=100),
                ],
                "Actual Start Time": [start + pd.Timedelta(minutes=1), pd.NaT, pd.NaT],
                "Actual Finish Time": [start + pd.Timedelta(minutes=42), pd.NaT, pd.NaT],
            }
        )
        .pipe(target.add_deviation_cols)
        .pipe(target.add_otp_flag)
    )
    assert df["on_time"].isna().tolist() == [False, True, True]
    assert bool(df["on_time"].iloc[0])

    summary = target.write_summary_table(df).set_index("trip_start_time")
    # The one evaluable 06:00 departure was on time; the missing one is not a miss.
    assert summary.loc["06:00", "n_events"] == 1
    assert summary.loc["06:00", "otp_pct"] == 100.0
    assert not summary.loc["06:00", "under_target"]
    # No evaluable 07:00 departure: OTP is unknown, not 0 %.
    assert summary.loc["07:00", "n_events"] == 0
    assert pd.isna(summary.loc["07:00", "otp_pct"])
    assert pd.isna(summary.loc["07:00", "under_target"])


def test_tides_filters_use_service_date_for_owl_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Friday owl trip starting at 00:30 Saturday stays a Friday trip."""
    path = _write_tides_trips(
        tmp_path,
        [
            _clever_runtime_row(
                **{
                    "Scheduled Start Time": "11/21/2026 12:30:00 AM",
                    "Scheduled Finish Time": "11/21/2026 1:10:00 AM",
                    "Actual Start Time": "11/21/2026 12:31:00 AM",
                    "Actual Finish Time": "11/21/2026 1:12:00 AM",
                }
            )
        ],
    )
    df = target.load_trip_files([path])
    assert df["service_date"].tolist() == ["2026-11-20"]
    assert df["trip_start_time"].tolist() == ["00:30"]

    assert len(target.filter_service_day(df, "WEEKDAY")) == 1
    assert target.filter_service_day(df, "SATURDAY").empty
    assert target.filter_holidays(df, ["2026-11-20"]).empty
    assert len(target.filter_holidays(df, ["2026-11-21"])) == 1

    monkeypatch.setattr(target, "DATE_START", pd.Timestamp("2026-11-01"))
    monkeypatch.setattr(target, "DATE_END", pd.Timestamp("2026-11-20"))
    assert len(target.filter_date_range(df)) == 1


def test_load_trip_files_accepts_mixed_clever_and_tides_batch(tmp_path: Path) -> None:
    legacy_path = tmp_path / "clever_runtime.csv"
    pd.DataFrame(
        [
            {
                "Route": "301 - Telegraph Rd",
                "Direction": "north bound",
                "TripID": "T200",
                "Trip": "07:00 T200",
                "Scheduled Start Time": "11/16/2026 7:00:00 AM",
                "Scheduled Finish Time": "11/16/2026 7:40:00 AM",
                "Actual Start Time": "11/16/2026 7:02:00 AM",
                "Actual Finish Time": "11/16/2026 7:45:00 AM",
            }
        ]
    ).to_csv(legacy_path, index=False)
    tides_path = _write_tides_trips(tmp_path, [_clever_runtime_row()])

    df = target.load_trip_files([legacy_path, tides_path]).pipe(target.extract_trip_start_time)

    assert df["_is_tides"].tolist() == [False, True]
    assert df["trip_start_time"].tolist() == ["07:00", "06:00"]
    assert df["Scheduled Start Time"].tolist() == [
        pd.Timestamp("2026-11-16 07:00"),
        pd.Timestamp("2026-11-16 06:00"),
    ]
    assert len(target.filter_routes(df, {"301"})) == 2

    # Only the CLEVER row is normalized; the TIDES row keeps its direction_id.
    with pytest.warns(RuntimeWarning, match="Direction normalized"):
        normalized = target.normalize_legacy_directions(df, target.ALLOWED_DIRECTIONS)
    assert normalized["Direction"].tolist() == ["NORTHBOUND", "0"]


def test_paired_clever_extracts_flow_through_both_converters_into_diagnostics(
    tmp_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime + stop visit extracts -> TIDES converters -> runtime diagnostics."""
    runtime = pd.DataFrame(
        [
            _clever_runtime_row(),
            # Same trip on Tuesday with no AVL actuals.
            _clever_runtime_row(
                **{
                    "Scheduled Start Time": "11/17/2026 6:00:00 AM",
                    "Scheduled Finish Time": "11/17/2026 6:40:00 AM",
                    "Actual Start Time": "",
                    "Actual Finish Time": "",
                }
            ),
        ]
    )
    visits = pd.DataFrame(
        {
            "Date": ["11/16/2026"] * 2,
            "Trip": ["06:00 T100"] * 2,
            "Route": ["301 - Telegraph Rd"] * 2,
            "Direction": ["NORTHBOUND"] * 2,
            "Vehicle": ["7001"] * 2,
            "Variation": ["1"] * 2,
            "Timepoint Order": ["1", "2"],
            "Timepoint ID": ["TP1", "TP2"],
            "Actual Time": ["11/16/2026 6:01:00 AM", "11/16/2026 6:20:00 AM"],
            "Arrival Time": ["11/16/2026 6:01:00 AM", "11/16/2026 6:20:00 AM"],
            "Departure Time": ["11/16/2026 6:01:30 AM", "11/16/2026 6:20:30 AM"],
            "Scheduled Passing Time": ["11/16/2026 6:00:00 AM", "11/16/2026 6:19:00 AM"],
        }
    )
    data_dir = tmp_path / "tides"
    data_dir.mkdir()
    trips = trip_converter.convert_to_tides(runtime)
    trips.to_csv(data_dir / "trips_performed.csv", index=False)
    stops = stop_converter.convert_to_tides(visits)

    joined = stops.merge(trips, on=["service_date", "trip_id_performed"], how="inner")
    assert len(joined) == len(stops) == 2

    routes = target._discover_route_csvs(data_dir)
    assert set(routes) == {"301"}

    monkeypatch.setattr(target, "DATE_START", pd.Timestamp("2026-11-01"))
    monkeypatch.setattr(target, "DATE_END", pd.Timestamp("2026-11-30"))
    df = (
        target.load_trip_files(routes["301"])
        .pipe(target.extract_trip_start_time)
        .pipe(target.filter_date_range)
        .pipe(target.filter_routes, {"301"})
        .pipe(target.filter_holidays, target.EXCLUDE_DATES)
        .pipe(target.filter_service_day, "WEEKDAY")
        .pipe(target.add_deviation_cols)
        .pipe(target.add_otp_flag)
    )
    assert len(df) == 2
    summary = target.write_summary_table(df)
    assert summary[["trip_start_time", "n_events", "otp_pct"]].to_numpy().tolist() == [
        ["06:00", 1, 100.0]
    ]
