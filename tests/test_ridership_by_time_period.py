"""Tests for ridership_by_time_period using the repo TIDES fixtures."""

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import ridership_by_time_period as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")
EXAMPLE_PERIODS = script_dir / "time_periods_example.txt"

DEFAULT_PERIODS = target.build_time_periods(
    target.specs_from_mapping(target.TIME_PERIODS), "test defaults"
)


def _periods(*specs: str) -> list:
    """Build checked periods from NAME=HH:MM-HH:MM strings."""
    parsed = [target.parse_period_spec(s, f"spec {i}") for i, s in enumerate(specs)]
    return target.build_time_periods(parsed, "test")


def _settings(tmp_path: Path, **overrides: object) -> "target.RunSettings":
    base = {
        "stop_visits_path": STOP_VISITS,
        "trips_performed_path": TRIPS_PERFORMED,
        "output_dir": tmp_path,
        "periods": DEFAULT_PERIODS,
    }
    base.update(overrides)
    return target.RunSettings(**base)


def _expected_visits() -> pd.DataFrame:
    """Independently join the fixtures: in-service, not canceled, not skipped."""
    sv = pd.read_csv(STOP_VISITS, dtype=str)
    tp = pd.read_csv(TRIPS_PERFORMED, dtype=str)
    sv["service_date"] = pd.to_datetime(sv["service_date"])
    tp["service_date"] = pd.to_datetime(tp["service_date"])
    tp = tp[(tp["trip_type"] == "In service") & (tp["schedule_relationship"] != "Canceled")]
    joined = sv.merge(
        tp[["service_date", "trip_id_performed", "route_id"]],
        on=["service_date", "trip_id_performed"],
    )
    joined = joined[joined["schedule_relationship"] != "Skipped"]
    joined["boardings"] = pd.to_numeric(joined["boarding_1"]) + pd.to_numeric(joined["boarding_2"])
    return joined


# --- time-period checks -------------------------------------------------------


def test_default_periods_cover_the_day() -> None:
    """The six default periods read cleanly and add up to 24 hours."""
    assert [p.name for p in DEFAULT_PERIODS] == [
        "Early AM",
        "AM Peak",
        "Midday",
        "PM Peak",
        "Late PM",
        "Night",
    ]
    assert sum(p.minutes for p in DEFAULT_PERIODS) == 24 * 60
    assert target.find_coverage_problems(DEFAULT_PERIODS) == []


def test_example_file_matches_inline_defaults() -> None:
    """time_periods_example.txt holds the same periods as TIME_PERIODS."""
    specs = target.load_time_periods_file(EXAMPLE_PERIODS)
    assert target.build_time_periods(specs, "example") == DEFAULT_PERIODS


def test_gap_is_reported() -> None:
    """A stretch of the day with no period stops the run and names the gap."""
    with pytest.raises(target.TimePeriodError, match="no period covers 02:00-04:00"):
        _periods("Day=04:00-22:00", "Night=22:00-02:00")


def test_gap_across_midnight_is_one_range() -> None:
    """An uncovered stretch that crosses midnight is reported as one range."""
    with pytest.raises(target.TimePeriodError, match="no period covers 23:00-01:00"):
        _periods("A=01:00-23:00")


def test_overlap_names_both_periods() -> None:
    """Overlapping periods are reported with both names and the shared range."""
    with pytest.raises(target.TimePeriodError) as excinfo:
        _periods("AM=00:00-09:30", "Midday=09:00-15:00", "PM=15:00-24:00")
    message = str(excinfo.value)
    assert "'AM' (00:00-09:30) and 'Midday' (09:00-15:00) overlap at 09:00-09:30" in message
    assert "no period covers" not in message


def test_wrapping_overlap_is_reported() -> None:
    """Two periods that both wrap past midnight overlap across it."""
    with pytest.raises(target.TimePeriodError, match="overlap at 23:00-01:00"):
        _periods("Night=22:00-01:00", "Owl=23:00-04:00", "Day=04:00-22:00")


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ("A=6am-24:00", "'6am' is not a 24-hour HH:MM time"),
        ("A=00:00-25:00", "'25:00' is outside 00:00-24:00"),
        ("A=00:00-12:60", "'12:60' is outside 00:00-24:00"),
        ("A=24:00-12:00", "cannot start at 24:00"),
        ("A=04:00-04:00", "starts and ends at 04:00"),
        (" =00:00-24:00", "the period name is blank"),
    ],
)
def test_malformed_period_is_reported(spec: str, fragment: str) -> None:
    """Malformed times, zero-length periods and blank names each get a reason."""
    with pytest.raises(target.TimePeriodError, match=fragment):
        _periods(spec)


def test_repeated_name_is_case_insensitive() -> None:
    """Names that differ only in case count as the same name."""
    with pytest.raises(target.TimePeriodError, match="already used"):
        _periods("Day=00:00-12:00", "DAY=12:00-24:00")


def test_all_problems_reported_together() -> None:
    """Every malformed period is listed in one error, not just the first."""
    with pytest.raises(target.TimePeriodError) as excinfo:
        _periods("A=6am-9am", "B=25:00-03:00", "C=03:00-03:00")
    message = str(excinfo.value)
    assert "'6am'" in message
    assert "'25:00'" in message
    assert "starts and ends at 03:00" in message


def test_full_day_and_midnight_ends() -> None:
    """00:00-24:00 is one whole-day period, and an end of 00:00 means midnight."""
    (whole,) = _periods("All=00:00-24:00")
    assert whole.minutes == 24 * 60
    day, evening = _periods("Day=00:00-18:00", "Evening=18:00-00:00")
    assert evening.end == 24 * 60
    assert evening.minutes == 6 * 60
    assert day.minutes + evening.minutes == 24 * 60


def test_empty_period_list_is_rejected() -> None:
    """At least one period is required."""
    with pytest.raises(target.TimePeriodError, match="No time periods"):
        target.build_time_periods([], "test")


def test_mapping_entry_must_be_a_pair() -> None:
    """An inline TIME_PERIODS value that is not a (start, end) pair is rejected."""
    with pytest.raises(target.TimePeriodError, match=r"TIME_PERIODS\['AM'\]"):
        target.specs_from_mapping({"AM": "06:00-09:00"})


def test_file_reports_bad_lines_with_numbers(tmp_path: Path) -> None:
    """A malformed line in the periods file is named by its line number."""
    path = tmp_path / "periods.txt"
    path.write_text(
        "# header\nDay = 04:00 - 22:00  # inline comment\n\nNight 22:00 - 04:00\n",
        encoding="utf-8",
    )
    with pytest.raises(target.TimePeriodError, match="line 4: expected NAME = HH:MM - HH:MM"):
        target.load_time_periods_file(path)


def test_file_accepts_compact_lines_and_bom(tmp_path: Path) -> None:
    """Spacing around = and - is optional, and a UTF-8 byte-order mark is ignored."""
    path = tmp_path / "periods.txt"
    path.write_text("Day=04:00-22:00\nNight = 22:00 – 04:00\n", encoding="utf-8-sig")
    periods = target.build_time_periods(target.load_time_periods_file(path), "test")
    assert [(p.name, p.start, p.end) for p in periods] == [
        ("Day", 240, 1320),
        ("Night", 1320, 240),
    ]


def test_missing_or_empty_file_is_a_period_error(tmp_path: Path) -> None:
    """A missing file and a file with no periods both raise TimePeriodError."""
    with pytest.raises(target.TimePeriodError, match="Could not read"):
        target.load_time_periods_file(tmp_path / "nope.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(target.TimePeriodError, match="lists no periods"):
        target.load_time_periods_file(empty)


def test_resolve_precedence(tmp_path: Path) -> None:
    """Flags beat the file, and the file beats the inline mapping."""
    path = tmp_path / "periods.txt"
    path.write_text("FromFile = 00:00 - 24:00\n", encoding="utf-8")
    inline = {"Inline": ("00:00", "24:00")}

    periods, source = target.resolve_time_periods(["Flag=00:00-24:00"], str(path), inline)
    assert [p.name for p in periods] == ["Flag"]
    assert "flags" in source

    periods, source = target.resolve_time_periods(None, str(path), inline)
    assert [p.name for p in periods] == ["FromFile"]
    assert "periods.txt" in source

    periods, source = target.resolve_time_periods(None, None, inline)
    assert [p.name for p in periods] == ["Inline"]
    assert "CONFIGURATION" in source


# --- time of day ---------------------------------------------------------------


def test_minute_of_day_formats() -> None:
    """ISO timestamps, offsets across a DST change, time-only and AM/PM values all read."""
    values = pd.Series(
        [
            "2026-01-05T21:03:14",
            "2026-03-07T21:03:14-05:00",  # offsets differ across the DST change,
            "2026-03-09T08:03:14-04:00",  # but the clock time is kept as written
            "2026-01-05 06:00:00",
            "21:03:14",
            "25:10:00",  # GTFS-style hour past midnight
            "1/5/2026 9:03:14 PM",
            "",
            None,
            "not a time",
        ],
        dtype=object,
    )
    minutes = target.minute_of_day(values)
    assert minutes.iloc[:7].tolist() == [1263, 1263, 483, 360, 1263, 70, 1263]
    assert minutes.iloc[7:].isna().all()


def test_period_boundaries_are_start_inclusive() -> None:
    """A visit exactly at a boundary belongs to the period that starts there."""
    df = pd.DataFrame(
        {
            "actual_departure_time": [
                "2026-01-05T03:59:59",
                "2026-01-05T04:00:00",
                "2026-01-05T05:59:59",
                "2026-01-05T06:00:00",
                "2026-01-05T21:59:00",
                "2026-01-05T22:00:00",
                "2026-01-06T00:30:00",
            ]
        }
    )
    out = target.add_time_period(df, DEFAULT_PERIODS)
    assert out["time_period"].tolist() == [
        "Night",
        "Early AM",
        "Early AM",
        "AM Peak",
        "Late PM",
        "Night",
        "Night",
    ]


def test_time_falls_back_through_columns() -> None:
    """A blank actual time falls back to the next column; no time at all drops the row."""
    df = pd.DataFrame(
        {
            "actual_departure_time": ["", None, "2026-01-05T12:00:00"],
            "schedule_departure_time": ["2026-01-05T07:00:00", None, "2026-01-05T23:00:00"],
        }
    )
    out = target.add_time_period(df, DEFAULT_PERIODS)
    assert out.index.tolist() == [0, 2]
    assert out["time_period"].tolist() == ["AM Peak", "Midday"]


def test_missing_time_columns_raise() -> None:
    """With none of TIME_COLUMNS present, visits cannot be placed in a period."""
    with pytest.raises(target.InputDataError, match="TIME_COLUMNS"):
        target.add_time_period(pd.DataFrame({"x": [1]}), DEFAULT_PERIODS)


def test_day_type_from_service_date() -> None:
    """Saturday and Sunday service dates get their own day type."""
    df = pd.DataFrame({"service_date": pd.to_datetime(["2026-01-05", "2026-01-03", "2026-01-04"])})
    assert target.add_day_type(df)["day_type"].tolist() == ["weekday", "saturday", "sunday"]
    assert set(target.add_day_type(df, False)["day_type"]) == {"all days"}


# --- end to end -------------------------------------------------------------------


@pytest.mark.parametrize("group_by", ["system", "route", "route_direction"])
def test_run_reconciles_with_fixture_totals(tmp_path: Path, group_by: str) -> None:
    """Every grain sums back to the same boardings and stop visits as a direct join."""
    expected = _expected_visits()
    table = target.run(_settings(tmp_path, group_by=group_by))

    assert table["total_boardings"].sum() == pytest.approx(expected["boardings"].sum())
    assert table["stop_visits"].sum() == len(expected)
    assert (tmp_path / f"ridership_by_time_period_{group_by}.csv").exists()
    assert (tmp_path / f"ridership_by_time_period_{group_by}_runlog.txt").exists()

    group_cols = target.GROUP_COLUMNS[group_by]
    per_group = table.groupby(["day_type", *group_cols])["time_period"].agg(list)
    names = [p.name for p in DEFAULT_PERIODS]
    assert all(periods == names for periods in per_group)


def test_run_averages_over_service_dates(tmp_path: Path) -> None:
    """Averages divide totals by the service dates of that day type."""
    table = target.run(_settings(tmp_path))
    days = table.groupby("day_type")["days_observed"].first().to_dict()
    assert days == {"weekday": 6, "saturday": 3, "sunday": 3}
    expected = (table["total_boardings"] / table["days_observed"]).round(2)
    assert table["avg_daily_boardings"].tolist() == expected.tolist()

    am = table[(table["day_type"] == "weekday") & (table["time_period"] == "AM Peak")].iloc[0]
    assert am["period_hours"] == 3.0
    assert am["avg_daily_boardings_per_period_hour"] == pytest.approx(
        am["total_boardings"] / 6 / 3, abs=0.01
    )


def test_periods_without_service_are_zero_rows(tmp_path: Path) -> None:
    """The fixture has no 04:00-06:00 visits, so Early AM rows are present and zero."""
    table = target.run(_settings(tmp_path))
    early = table[table["time_period"] == "Early AM"]
    assert len(early) == 3
    assert (early["total_boardings"] == 0).all()
    assert (early["stop_visits"] == 0).all()


def test_route_totals_match_system_totals(tmp_path: Path) -> None:
    """Summing the route grain over routes reproduces the system grain."""
    system = target.run(_settings(tmp_path, group_by="system"))
    route = target.run(_settings(tmp_path, group_by="route"))
    rolled = route.groupby(["day_type", "time_period"])["total_boardings"].sum()
    flat = system.set_index(["day_type", "time_period"])["total_boardings"]
    pd.testing.assert_series_equal(rolled.sort_index(), flat.sort_index(), check_names=False)


def test_route_filters(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Include keeps only the named routes; unknown entries are warned about."""
    with caplog.at_level(logging.WARNING):
        table = target.run(
            _settings(tmp_path, group_by="route", routes_to_include=("DC_R10", "NOPE"))
        )
    assert set(table["route_id"]) == {"DC_R10"}
    assert "NOPE" in caplog.text

    table = target.run(_settings(tmp_path, group_by="route", routes_to_exclude=("DC_R10",)))
    assert "DC_R10" not in set(table["route_id"])


def test_run_rejects_uncovered_periods(tmp_path: Path) -> None:
    """run() re-checks periods built by hand and writes nothing when they leave a gap."""
    periods = [target.TimePeriod("Day", 0, 12 * 60)]
    with pytest.raises(target.TimePeriodError, match="no period covers 12:00-24:00"):
        target.run(_settings(tmp_path, periods=periods))
    assert not list(tmp_path.iterdir())


def test_route_direction_needs_direction_id(tmp_path: Path) -> None:
    """Grouping by direction without a direction_id column is a clear data error."""
    trips = pd.read_csv(TRIPS_PERFORMED, dtype=str).drop(columns="direction_id")
    trips_path = tmp_path / "trips.csv"
    trips.to_csv(trips_path, index=False)
    with pytest.raises(target.InputDataError, match="direction_id"):
        target.run(_settings(tmp_path, trips_performed_path=trips_path, group_by="route_direction"))


# --- main -----------------------------------------------------------------------


def test_main_stops_on_bad_periods_before_reading_data(caplog: pytest.LogCaptureFixture) -> None:
    """Overlapping flags exit 2 with the problem named, even with placeholder paths."""
    with caplog.at_level(logging.ERROR):
        code = target.main(["--time-period", "A=00:00-12:00", "--time-period", "B=11:00-24:00"])
    assert code == 2
    assert "overlap at 11:00-12:00" in caplog.text


def test_main_placeholder_paths_exit_2() -> None:
    """Valid periods but placeholder input paths exit with the config-error code."""
    assert target.main([]) == 2


def test_main_end_to_end(tmp_path: Path) -> None:
    """A full CLI run with a periods file writes the table and a run log naming the file."""
    periods_file = tmp_path / "periods.txt"
    periods_file.write_text("Day = 05:00 - 20:00\nNight = 20:00 - 05:00\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    code = target.main(
        [
            "--stop-visits",
            str(STOP_VISITS),
            "--trips-performed",
            str(TRIPS_PERFORMED),
            "--output-dir",
            str(out_dir),
            "--group-by",
            "route_direction",
            "--time-periods-file",
            str(periods_file),
            "--no-split-by-day-type",
        ]
    )
    assert code == 0
    table = pd.read_csv(out_dir / "ridership_by_time_period_route_direction.csv", dtype=str)
    assert set(table["time_period"]) == {"Day", "Night"}
    assert set(table["day_type"]) == {"all days"}
    log = (out_dir / "ridership_by_time_period_route_direction_runlog.txt").read_text("utf-8")
    assert "periods.txt" in log
    assert "Day = 05:00 - 20:00" in log
    assert "# === BEGIN CONFIG ===" not in log
    assert "TIME_PERIODS: Mapping" in log


def test_main_missing_input_exits_1(tmp_path: Path) -> None:
    """A path that does not exist is a runtime failure, not a crash."""
    code = target.main(
        [
            "--stop-visits",
            str(tmp_path / "missing.csv"),
            "--trips-performed",
            str(TRIPS_PERFORMED),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 1
