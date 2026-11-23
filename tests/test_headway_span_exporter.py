from __future__ import annotations

import datetime as dt
import logging
import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import scripts.gtfs_exports.headway_span_exporter as target

GTFS_BASIC = Path(__file__).parent / "fixtures" / "gtfs_basic"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _calendar_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_id": ["WK", "SAT", "MF_ONLY"],
            "monday": ["1", "0", "1"],
            "tuesday": ["1", "0", "1"],
            "wednesday": ["1", "0", "1"],
            "thursday": ["1", "0", "1"],
            "friday": ["1", "0", "0"],
            "saturday": ["0", "1", "0"],
            "sunday": ["0", "0", "0"],
        }
    )


def _write_gtfs(gtfs_dir: Path) -> None:
    """Minimal weekday feed: route R1 with 3 trips at 06:00/06:30/07:00."""
    (gtfs_dir / "routes.txt").write_text("route_id,route_short_name\nR1,101\nR2,202\n")
    (gtfs_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id\n"
        "R1,WK,T1,0\nR1,WK,T2,0\nR1,WK,T3,0\nR2,SAT,T4,0\n"
    )
    (gtfs_dir / "stop_times.txt").write_text(
        "trip_id,stop_id,stop_sequence,departure_time\n"
        "T1,S1,1,06:00:00\nT1,S2,2,06:10:00\n"
        "T2,S1,1,06:30:00\nT2,S2,2,06:40:00\n"
        "T3,S1,1,07:00:00\nT3,S2,2,07:10:00\n"
        "T4,S1,1,09:00:00\n"
    )
    (gtfs_dir / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday\n"
        "WK,1,1,1,1,1,0,0\nSAT,0,0,0,0,0,1,0\n"
    )


# ---------------------------------------------------------------------------
# parse_time_to_seconds
# ---------------------------------------------------------------------------


def test_parse_time_to_seconds_basic_conversion() -> None:
    assert target.parse_time_to_seconds("06:30:00") == 23400
    assert target.parse_time_to_seconds("06:30") == 23400


def test_parse_time_to_seconds_preserves_times_past_midnight() -> None:
    assert target.parse_time_to_seconds("25:30:00") == 91800


def test_parse_time_to_seconds_keeps_seconds() -> None:
    assert target.parse_time_to_seconds("06:00:31") == 21631
    assert target.parse_time_to_seconds("06:00:30") == 21630


def test_parse_time_to_seconds_invalid_values_return_none() -> None:
    assert target.parse_time_to_seconds("not a time") is None
    assert target.parse_time_to_seconds("06:60:00") is None
    assert target.parse_time_to_seconds(None) is None
    assert target.parse_time_to_seconds(360) is None


# ---------------------------------------------------------------------------
# service_ids_for_day
# ---------------------------------------------------------------------------


def test_service_ids_for_day_weekday_requires_all_five_days() -> None:
    ids = target.service_ids_for_day(_calendar_df(), "weekday")
    assert ids == {"WK"}  # MF_ONLY misses Friday


def test_service_ids_for_day_saturday() -> None:
    assert target.service_ids_for_day(_calendar_df(), "saturday") == {"SAT"}


def test_service_ids_for_day_invalid_day_raises() -> None:
    with pytest.raises(ValueError, match="SERVICE_DAY"):
        target.service_ids_for_day(_calendar_df(), "holiday")


def test_service_ids_for_day_missing_column_raises() -> None:
    cal = _calendar_df().drop(columns=["friday"])
    with pytest.raises(ValueError, match="friday"):
        target.service_ids_for_day(cal, "weekday")


def test_service_ids_for_day_no_match_returns_empty(caplog) -> None:
    cal = _calendar_df()
    with caplog.at_level("WARNING"):
        ids = target.service_ids_for_day(cal, "sunday")
    assert ids == set()
    assert "No service_ids" in caplog.text


# ---------------------------------------------------------------------------
# first_departures
# ---------------------------------------------------------------------------


def test_first_departures_takes_lowest_stop_sequence() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2"],
            "stop_sequence": ["2", "1", "1"],
            "departure_time": ["06:10:00", "06:00:00", "07:00:00"],
        }
    )
    out = target.first_departures(stop_times, {"T1", "T2"})
    lookup = dict(zip(out["trip_id"], out["departure_min"]))
    assert lookup["T1"] == 360
    assert lookup["T2"] == 420


def test_first_departures_filters_to_requested_trips() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T9"],
            "stop_sequence": ["1", "1"],
            "departure_time": ["06:00:00", "08:00:00"],
        }
    )
    out = target.first_departures(stop_times, {"T1"})
    assert set(out["trip_id"]) == {"T1"}


def test_first_departures_keeps_seconds() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "stop_sequence": ["1"],
            "departure_time": ["06:16:20"],
        }
    )
    out = target.first_departures(stop_times, {"T1"})
    assert out.iloc[0]["departure_min"] == pytest.approx(376 + 1 / 3)


def test_stop_time_counts_reports_zero_for_trips_without_rows() -> None:
    stop_times = pd.DataFrame({"trip_id": ["T1", "T1", "T2", "T9"]})
    counts = target.stop_time_counts(stop_times, {"T1", "T2", "T3"})
    assert counts.to_dict() == {"T1": 2, "T2": 1, "T3": 0}


# ---------------------------------------------------------------------------
# compute_headway_span
# ---------------------------------------------------------------------------


def test_compute_headway_span_single_direction() -> None:
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"] * 3,
            "direction_id": ["0"] * 3,
            "departure_min": [360, 390, 420],
        }
    )
    out = target.compute_headway_span(trips_dep)
    row = out.iloc[0]
    assert row["avg_headway_min"] == 30.0
    assert row["span_hrs"] == 1.0
    assert row["trip_count"] == 3


def test_compute_headway_span_averages_across_directions() -> None:
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"] * 4,
            "direction_id": ["0", "0", "1", "1"],
            "departure_min": [360, 380, 360, 420],  # 20-min and 60-min headways
        }
    )
    out = target.compute_headway_span(trips_dep)
    row = out.iloc[0]
    assert row["avg_headway_min"] == 40.0  # mean of 20 and 60
    assert row["span_hrs"] == 1.0  # first to last departure across directions
    assert row["trip_count"] == 4


def test_compute_headway_span_rounds_once_after_averaging_directions() -> None:
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"] * 6,
            "direction_id": ["0", "0", "1", "1", "1", "1"],
            # 30.0-min and 30.666...-min headways: the mean is 30.333...
            "departure_min": [360, 390, 360, 390, 420, 452],
        }
    )
    row = target.compute_headway_span(trips_dep).iloc[0]
    # Rounding each direction first (30.0 and 30.7) would give 30.4.
    assert row["avg_headway_min"] == 30.3


def test_compute_headway_span_spans_first_to_last_departure_across_directions() -> None:
    # AM-inbound / PM-outbound route: each direction alone spans one hour.
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"] * 4,
            "direction_id": ["0", "0", "1", "1"],
            "departure_min": [360, 420, 1020, 1080],  # 06:00, 07:00, 17:00, 18:00
        }
    )
    out = target.compute_headway_span(trips_dep)
    row = out.iloc[0]
    assert row["span_hrs"] == 12.0  # 06:00 to 18:00
    assert row["avg_headway_min"] == 60.0  # headway stays per direction
    assert row["trip_count"] == 4


def test_compute_headway_span_single_trip_gives_nan_headway() -> None:
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"],
            "direction_id": ["0"],
            "departure_min": [360],
        }
    )
    out = target.compute_headway_span(trips_dep)
    row = out.iloc[0]
    assert math.isnan(row["avg_headway_min"])
    assert row["span_hrs"] == 0.0
    assert row["trip_count"] == 1


def test_compute_headway_span_empty_input() -> None:
    trips_dep = pd.DataFrame(columns=["route_id", "direction_id", "departure_min"])
    out = target.compute_headway_span(trips_dep)
    assert out.empty
    assert list(out.columns) == ["route_id", "avg_headway_min", "span_hrs", "trip_count"]


# ---------------------------------------------------------------------------
# load_gtfs / run
# ---------------------------------------------------------------------------


def test_load_gtfs_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="trips.txt"):
        target.load_gtfs(tmp_path)


def test_run_end_to_end_writes_csv(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    out_csv = tmp_path / "out" / "headway_span_by_route.csv"

    result = target.run(gtfs_folder=gtfs_dir, output_path=out_csv, service_day="weekday")

    assert out_csv.exists()
    assert list(result["route_id"]) == ["R1"]  # R2 is Saturday-only
    row = result.iloc[0]
    assert row["avg_headway_min"] == 30.0
    assert row["span_hrs"] == 1.0
    assert row["trip_count"] == 3
    assert row["excluded_trip_count"] == 0
    written = pd.read_csv(out_csv)
    assert len(written) == 1
    assert list(written.columns) == target.OUTPUT_COLUMNS


def test_run_excludes_trips_with_fewer_than_two_stop_times(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    # Single-stop records at 06:15 and 06:45 sit between R1's 30-minute
    # departures; X3 has no stop_times rows at all.
    with (gtfs_dir / "trips.txt").open("a") as handle:
        handle.write("R1,WK,X1,0\nR1,WK,X2,0\nR1,WK,X3,0\n")
    with (gtfs_dir / "stop_times.txt").open("a") as handle:
        handle.write("X1,S1,1,06:15:00\nX2,S1,1,06:45:00\n")

    with caplog.at_level(logging.WARNING):
        result = target.run(
            gtfs_folder=gtfs_dir, output_path=tmp_path / "out.csv", service_day="weekday"
        )

    row = result.iloc[0]
    # Counting X1/X2 as departures would report 5 trips and a 15-minute headway.
    assert row["avg_headway_min"] == 30.0
    assert row["trip_count"] == 3
    assert row["excluded_trip_count"] == 3
    assert "3 selected trip(s) have fewer than 2 stop_times rows" in caplog.text
    assert "X1, X2, X3" in caplog.text


def test_run_keeps_route_whose_trips_are_all_excluded(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    # R2's only trip (T4, one stop) now also runs on weekdays.
    (gtfs_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id\nR1,WK,T1,0\nR1,WK,T2,0\nR1,WK,T3,0\nR2,WK,T4,0\n"
    )
    result = target.run(
        gtfs_folder=gtfs_dir, output_path=tmp_path / "out.csv", service_day="weekday"
    ).set_index("route_id")
    assert result.loc["R1", "trip_count"] == 3
    assert result.loc["R2", "trip_count"] == 0
    assert result.loc["R2", "excluded_trip_count"] == 1
    assert math.isnan(result.loc["R2", "avg_headway_min"])


def test_run_headway_keeps_departure_seconds(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    # Departures 16 min 20 s apart. Rounding each to the minute first
    # (360, 376, 393) would report 16.5.
    (gtfs_dir / "stop_times.txt").write_text(
        "trip_id,stop_id,stop_sequence,departure_time\n"
        "T1,S1,1,06:00:00\nT1,S2,2,06:10:00\n"
        "T2,S1,1,06:16:20\nT2,S2,2,06:26:20\n"
        "T3,S1,1,06:32:40\nT3,S2,2,06:42:40\n"
    )
    result = target.run(
        gtfs_folder=gtfs_dir, output_path=tmp_path / "out.csv", service_day="weekday"
    )
    assert result.iloc[0]["avg_headway_min"] == 16.3


def test_run_stops_on_selected_frequency_based_trip(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    # T2 now repeats every 10 minutes from 07:00 to 09:00 (12 departures).
    (gtfs_dir / "frequencies.txt").write_text(
        "trip_id,start_time,end_time,headway_secs,exact_times\nT2,07:00:00,09:00:00,600,1\n"
    )
    with pytest.raises(ValueError, match=r"selected trip\(s\) \(T2\) are frequency-based"):
        target.run(gtfs_folder=gtfs_dir, output_path=tmp_path / "out.csv", service_day="weekday")
    assert not (tmp_path / "out.csv").exists()


def test_run_ignores_frequency_trips_that_are_not_selected(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    # T4 is Saturday-only, so a weekday run never selects it.
    (gtfs_dir / "frequencies.txt").write_text(
        "trip_id,start_time,end_time,headway_secs,exact_times\nT4,09:00:00,11:00:00,600,1\n"
    )
    result = target.run(
        gtfs_folder=gtfs_dir, output_path=tmp_path / "out.csv", service_day="weekday"
    )
    assert result.iloc[0]["trip_count"] == 3


def test_run_route_filters_can_exclude_everything(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    with pytest.raises(ValueError, match="No trips remain after filtering"):
        target.run(
            gtfs_folder=gtfs_dir,
            output_path=tmp_path / "out.csv",
            service_day="weekday",
            filter_out_route_short_names=["101"],
        )
    assert not (tmp_path / "out.csv").exists()


# ---------------------------------------------------------------------------
# Date-based service resolution (calendar fixtures)
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def test_run_split_weekday_uses_modal_pattern_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = target.run(
            gtfs_folder=FIXTURES / "gtfs_split_weekday",
            output_path=tmp_path / "out.csv",
            service_day="weekday",
        )
    row = result.iloc[0]
    # Tue-Thu is the modal weekday; Monday's 4 and Friday's 3 trips must not
    # be summed on top of it (the old column-based union would report 13).
    assert row["trip_count"] == 6
    assert "varies by day of week" in caplog.text


def test_run_split_weekday_single_day_selection(tmp_path: Path) -> None:
    result = target.run(
        gtfs_folder=FIXTURES / "gtfs_split_weekday",
        output_path=tmp_path / "out.csv",
        service_day="friday",
    )
    assert result.iloc[0]["trip_count"] == 3


def test_run_holiday_negation_keeps_holiday_service_out_of_weekday(tmp_path: Path) -> None:
    result = target.run(
        gtfs_folder=FIXTURES / "gtfs_holiday_negation",
        output_path=tmp_path / "out.csv",
        service_day="weekday",
    )
    # HOL claims Mon/Wed/Fri columns but only runs on holidays; a normal
    # weekday must count WKD's 6 trips alone.
    assert result.iloc[0]["trip_count"] == 6


def test_run_zipped_feed_matches_folder(tmp_path: Path) -> None:
    """A .zip (files in a wrapper folder) gives the same result, calendar_dates included."""
    folder = FIXTURES / "gtfs_holiday_negation"
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"feed/{path.name}")

    from_folder = target.run(
        gtfs_folder=folder, output_path=tmp_path / "a.csv", service_day="weekday"
    )
    from_zip = target.run(gtfs_folder=zipped, output_path=tmp_path / "b.csv", service_day="weekday")
    pd.testing.assert_frame_equal(from_zip, from_folder)
    assert from_zip.iloc[0]["trip_count"] == 6


def test_run_service_date_override_selects_holiday_schedule(tmp_path: Path) -> None:
    result = target.run(
        gtfs_folder=FIXTURES / "gtfs_split_weekday",
        output_path=tmp_path / "out.csv",
        service_day="weekday",
        service_date="20260907",  # Labor Day: Sunday schedule replaces MON
    )
    assert result.iloc[0]["trip_count"] == 2


def test_run_falls_back_when_calendar_lacks_dates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)  # calendar.txt has no start_date/end_date columns
    with caplog.at_level(logging.WARNING):
        result = target.run(
            gtfs_folder=gtfs_dir,
            output_path=tmp_path / "out.csv",
            service_day="weekday",
        )
    assert "falling back" in caplog.text
    assert result.iloc[0]["trip_count"] == 3


@pytest.mark.parametrize("service_date", ["", "20260115"])
def test_run_calendar_dates_only_feed(tmp_path: Path, service_date: str) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    (gtfs_dir / "calendar.txt").unlink()
    # Thursday 2026-01-15 is WK's only service date; SAT runs Saturday 2026-01-17.
    (gtfs_dir / "calendar_dates.txt").write_text(
        "service_id,date,exception_type\nWK,20260115,1\nSAT,20260117,1\n"
    )
    result = target.run(
        gtfs_folder=gtfs_dir,
        output_path=tmp_path / "out.csv",
        service_day="weekday",
        service_date=service_date,
    )
    assert list(result["route_id"]) == ["R1"]
    assert result.iloc[0]["trip_count"] == 3


def _write_november_feed(gtfs_dir: Path) -> None:
    """Weekday service through November 2026, none on Thanksgiving (Nov 26)."""
    _write_gtfs(gtfs_dir)
    (gtfs_dir / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "WK,1,1,1,1,1,0,0,20261101,20261130\nSAT,0,0,0,0,0,1,0,20261101,20261130\n"
    )
    (gtfs_dir / "calendar_dates.txt").write_text("service_id,date,exception_type\nWK,20261126,2\n")


def test_main_no_service_date_inside_feed_writes_header_only_csv(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_november_feed(gtfs_dir)
    out_csv = tmp_path / "thanksgiving.csv"
    argv = ["--gtfs-folder", str(gtfs_dir), "--output", str(out_csv)]

    with caplog.at_level(logging.WARNING):
        assert target.main([*argv, "--service-date", "20261126"]) == 0
    assert "No trips are scheduled on SERVICE_DATE 20261126" in caplog.text
    written = pd.read_csv(out_csv)
    assert written.empty
    assert list(written.columns) == target.OUTPUT_COLUMNS

    # Route filters never turn a no-service day into an error either.
    assert target.main([*argv, "--service-date", "20261126", "--filter-in", "101"]) == 0


def test_main_service_date_outside_feed_calendar_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_november_feed(gtfs_dir)
    out_csv = tmp_path / "out.csv"
    with caplog.at_level(logging.ERROR):
        rc = target.main(
            ["--gtfs-folder", str(gtfs_dir), "--output", str(out_csv), "--service-date", "20261201"]
        )
    assert rc == 1
    assert "outside the period the feed's calendar covers" in caplog.text
    assert "20261101 to 20261130" in caplog.text
    assert not out_csv.exists()


def test_calendar_period_spans_calendar_and_calendar_dates() -> None:
    calendar = pd.DataFrame({"start_date": ["20261101"], "end_date": ["20261130"]})
    calendar_dates = pd.DataFrame({"date": ["20261215", "bad"]})
    assert target.calendar_period(calendar, calendar_dates) == (
        dt.date(2026, 11, 1),
        dt.date(2026, 12, 15),
    )
    assert target.calendar_period(pd.DataFrame(columns=["start_date"]), None) is None


def test_run_requires_calendar_or_calendar_dates(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    (gtfs_dir / "calendar.txt").unlink()
    with pytest.raises(FileNotFoundError, match="neither calendar.txt nor calendar_dates.txt"):
        target.run(gtfs_folder=gtfs_dir, output_path=tmp_path / "out.csv", service_day="weekday")


def test_resolve_service_ids_rejects_malformed_service_date() -> None:
    calendar = _calendar_df()
    with pytest.raises(ValueError, match="SERVICE_DATE"):
        target.resolve_service_ids(calendar, None, "weekday", "2026-09-07")


# ---------------------------------------------------------------------------
# main — full pipeline against gtfs_basic
# ---------------------------------------------------------------------------


def test_main_end_to_end_on_gtfs_basic(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    out_csv = tmp_path / "headway_span_by_route.csv"
    argv = ["--gtfs-folder", str(GTFS_BASIC), "--output", str(out_csv), "--service-day", "weekday"]

    # R1's T2 is a frequency-based trip in frequencies.txt, so the run stops.
    with caplog.at_level(logging.ERROR):
        assert target.main(argv) == 1
    assert "(T2) are frequency-based" in caplog.text
    assert not out_csv.exists()

    assert target.main([*argv, "--filter-out", "R1"]) == 0
    result = pd.read_csv(out_csv).set_index("route_id")
    assert result.index.tolist() == ["R2", "R3"]
    assert result["trip_count"].tolist() == [3, 3]
    # Each route's first and last trips start 10 hours apart. R3's uneven
    # 6 h and 4 h gaps (08:00, 14:00, 18:00) still average 300 minutes.
    assert result["span_hrs"].tolist() == [10.0, 10.0]
    assert result["avg_headway_min"].tolist() == [300.0, 300.0]
