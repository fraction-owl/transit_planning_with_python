"""Tests for ridership_from_tides (the TIDES ridership and max-load exporter).

Most end-to-end checks share one run on the repo TIDES fixtures (every level and
time cut, charts off, 2026-02-16 as a holiday) and compare it with an independent
join of the same files.
"""

import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import ridership_from_tides as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")
PERIODS_FIXTURE = Path("tests/fixtures/time_periods.txt")
EXAMPLE_PERIODS = script_dir / "time_periods_example.txt"
HOLIDAY = dt.date(2026, 2, 16)  # Washington's Birthday, a Monday in the fixture

DEFAULT_PERIODS = target.build_time_periods(
    target.specs_from_mapping(target.TIME_PERIODS), "test defaults"
)
ALL_GRAINS = [(level, cut) for level in target.LEVELS for cut in target.TIME_CUTS]


def _periods(*specs: str) -> list:
    """Build checked periods from NAME=HH:MM-HH:MM strings."""
    parsed = [target.parse_period_spec(s, f"spec {i}") for i, s in enumerate(specs)]
    return target.build_time_periods(parsed, "test")


def _settings(tmp_path: Path, **overrides: object) -> "target.RunSettings":
    base: dict = {
        "stop_visits_path": STOP_VISITS,
        "trips_performed_path": TRIPS_PERFORMED,
        "output_dir": tmp_path,
        "periods": DEFAULT_PERIODS,
        "export_charts": False,
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
        tp[["service_date", "trip_id_performed", "route_id", "direction_id"]],
        on=["service_date", "trip_id_performed"],
    )
    joined = joined[joined["schedule_relationship"] != "Skipped"]
    joined["boardings"] = pd.to_numeric(joined["boarding_1"]) + pd.to_numeric(joined["boarding_2"])
    return joined


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> "target.RunResult":
    """Every level and time cut on the fixtures, with one holiday and charts off."""
    out_dir = tmp_path_factory.mktemp("full_run")
    settings = target.RunSettings(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=out_dir,
        periods=DEFAULT_PERIODS,
        levels=list(target.LEVELS),
        time_cuts=list(target.TIME_CUTS),
        holidays=[HOLIDAY],
        export_charts=False,
    )
    return target.run(settings)


# --- time periods ---------------------------------------------------------------


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


@pytest.mark.parametrize("path", [EXAMPLE_PERIODS, PERIODS_FIXTURE])
def test_periods_files_match_inline_defaults(path: Path) -> None:
    """The example file and the system-periods fixture hold the inline defaults."""
    specs = target.load_time_periods_file(path)
    assert target.build_time_periods(specs, str(path)) == DEFAULT_PERIODS


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
    assert day.minutes + evening.minutes == 24 * 60


def test_empty_period_list_and_bad_mapping_entry() -> None:
    """At least one period is required, and an inline entry must be a pair."""
    with pytest.raises(target.TimePeriodError, match="No time periods"):
        target.build_time_periods([], "test")
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
    assert [(p.name, p.start, p.end) for p in periods] == [("Day", 240, 1320), ("Night", 1320, 240)]


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
    assert [p.name for p in periods] == ["Flag"] and "flags" in source
    periods, source = target.resolve_time_periods(None, str(path), inline)
    assert [p.name for p in periods] == ["FromFile"] and "periods.txt" in source
    periods, source = target.resolve_time_periods(None, None, inline)
    assert [p.name for p in periods] == ["Inline"] and "CONFIGURATION" in source


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


def test_service_day_hour() -> None:
    """Next-day stamps, small-hours stamps and GTFS times land on hours 24 and up."""
    values = pd.Series(
        [
            "2026-01-05T21:03:14",  # evening of the service date
            "2026-01-06T00:30:00",  # next calendar day -> 24
            "2026-01-05T01:15:00",  # small hours stamped with the service date -> 25
            "2026-01-06T05:10:00-05:00",  # next day after the cutover -> 29
            "25:10:00",  # GTFS time past midnight
            "02:00:00",  # time-only before the cutover -> 26
            "1/6/2026 0:45:00 AM",  # parsed by pandas, next day -> 24
            "",
        ],
        dtype=object,
    )
    dates = pd.Series(pd.to_datetime(["2026-01-05"] * len(values)))
    hours = target.service_day_hour(values, dates, 4)
    assert hours.iloc[:7].tolist() == [21, 24, 25, 29, 25, 26, 24]
    assert pd.isna(hours.iloc[7])
    assert target.service_day_hour(values.iloc[[2]], dates.iloc[[2]], 1).tolist() == [1]


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
    assert out["_time_text"].tolist() == ["2026-01-05T07:00:00", "2026-01-05T12:00:00"]


def test_missing_time_columns_raise() -> None:
    """With none of TIME_COLUMNS present, visits cannot be placed in a period."""
    with pytest.raises(target.InputDataError, match="TIME_COLUMNS"):
        target.add_time_period(pd.DataFrame({"x": [1]}), DEFAULT_PERIODS)


# --- day types ------------------------------------------------------------------


def test_build_day_types_reads_names_and_abbreviations() -> None:
    """Full names and three-letter abbreviations, in any case, map to weekday numbers."""
    out = target.build_day_types({"Mon-Thu": ("mon", "Tue", "WEDNESDAY", "thu"), "Fri": "Friday"})
    assert out == {"Mon-Thu": (0, 1, 2, 3), "Fri": (4,)}


@pytest.mark.parametrize(
    ("mapping", "fragment"),
    [
        ({"Weekday": ("Funday",)}, "'Funday' is not a weekday name"),
        ({" ": ("Mon",)}, "the day type name is blank"),
        ({"Holiday": ("Mon",)}, "'Holiday' is already used"),
        ({"Weekday": ()}, "lists no weekdays"),
        ({}, "DAY_TYPES is empty"),
    ],
)
def test_build_day_types_problems(mapping: dict, fragment: str) -> None:
    """Unknown weekdays, blank or reserved names and empty entries are reported."""
    with pytest.raises(target.SettingsError, match=fragment):
        target.build_day_types(mapping)


def test_parse_iso_dates_reports_every_bad_value() -> None:
    """Dates must be YYYY-MM-DD; every bad value is named."""
    assert target.parse_iso_dates(["2026-02-16", "2026-01-19", "2026-02-16"], "HOLIDAYS") == [
        dt.date(2026, 1, 19),
        dt.date(2026, 2, 16),
    ]
    with pytest.raises(target.SettingsError) as excinfo:
        target.parse_iso_dates(["2/16/2026", "2026-13-01"], "HOLIDAYS")
    assert "'2/16/2026'" in str(excinfo.value) and "'2026-13-01'" in str(excinfo.value)


def test_date_table_holidays_overlaps_and_unassigned_days() -> None:
    """Holidays get only their own type; overlapping types both apply; others drop out."""
    day_types = target.build_day_types({"Weekday": ("Mon", "Tue"), "Monday": ("Mon",)})
    dates = pd.to_datetime(["2026-02-16", "2026-02-23", "2026-02-24", "2026-02-21"])
    table = target.build_date_table(dates, day_types, [HOLIDAY], "Holiday")
    pairs = sorted((d.strftime("%m-%d"), t) for d, t in table.to_numpy().tolist())
    assert pairs == [
        ("02-16", "Holiday"),
        ("02-23", "Monday"),
        ("02-23", "Weekday"),
        ("02-24", "Weekday"),
    ]  # 02-21 is a Saturday, in no day type


# --- joining --------------------------------------------------------------------


def test_join_takes_each_days_trip_attributes() -> None:
    """A trip ID repeats daily; each day's stop visits get that day's attributes."""
    visits = pd.DataFrame(
        {
            "service_date": ["2026-03-02", "2026-03-03"],
            "trip_id_performed": ["T100", "T100"],
            "stop_id": ["1010", "1010"],
        }
    )
    trips = pd.DataFrame(
        {
            # Written M/D/YYYY, as a spreadsheet-edited export might be.
            "service_date": ["3/2/2026", "3/3/2026"],
            "trip_id_performed": ["T100", "T100"],
            "route_id": ["101", "102"],
            "direction_id": ["0", "1"],
        }
    )
    joined = target.join_trips(visits, trips).sort_values("service_date")
    assert joined[["route_id", "direction_id"]].to_numpy().tolist() == [["101", "0"], ["102", "1"]]


def test_join_drops_canceled_and_non_revenue_but_keeps_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Canceled (any case) and non-revenue trips go; Skipped visits stay until drop_skipped."""
    visits = pd.DataFrame(
        {
            "service_date": ["2026-03-02"] * 4 + ["2026-03-03"],
            "trip_id_performed": ["A", "A", "B", "C", "A"],
            "schedule_relationship": ["Scheduled", "Skipped", "Scheduled", "Scheduled", "x"],
        }
    )
    trips = pd.DataFrame(
        {
            "service_date": ["2026-03-02", "2026-03-02", "2026-03-02"],
            "trip_id_performed": ["A", "B", "C"],
            "schedule_relationship": ["Scheduled", "CANCELED", "Scheduled"],
            "trip_type": ["In service", "In service", "Deadhead"],
        }
    )
    with caplog.at_level(logging.WARNING):
        joined = target.join_trips(visits, trips)
    assert joined["trip_id_performed"].tolist() == ["A", "A"]
    assert "1 stop visit(s) have no trips_performed row" in caplog.text
    assert target.drop_skipped(joined)["schedule_relationship"].tolist() == ["Scheduled"]


def test_trip_key_prefers_the_scheduled_trip() -> None:
    """trip_id is trip_id_scheduled where filled, else trip_id_performed."""
    df = pd.DataFrame({"trip_id_performed": ["P1", "P2"], "trip_id_scheduled": ["S1", " "]})
    assert target.add_trip_key(df)["trip_id"].tolist() == ["S1", "P2"]


def test_boardings_sum_both_doors() -> None:
    """Total boardings/alightings sum the per-door columns; blanks count zero."""
    df = pd.DataFrame(
        {
            "boarding_1": ["3", "0"],
            "boarding_2": ["1", ""],
            "alighting_1": ["0", "2"],
            "alighting_2": [None, "1"],
        }
    )
    out = target.add_ridership(df)
    assert out["boardings"].tolist() == [4.0, 0.0]
    assert out["alightings"].tolist() == [0.0, 3.0]


# --- ridership tables ---------------------------------------------------------------


def test_every_grain_is_written(full_run: "target.RunResult") -> None:
    """One ridership CSV per level and cut, max-load rollups, trip tables and a run log."""
    assert set(full_run.ridership) == set(ALL_GRAINS)
    out_dir = full_run.files[0].parents[1]
    for level, cut in ALL_GRAINS:
        assert (out_dir / "ridership" / target.grain_filename("ridership", level, cut)).exists()
    expected_max = {(lv, c) for lv in target.MAX_LOAD_LEVELS for c in target.TIME_CUTS}
    assert set(full_run.max_load) == expected_max
    assert (out_dir / "max_load" / "max_load_by_performed_trip.csv").exists()
    assert (out_dir / "max_load" / "max_load_by_trip.csv").exists()
    for day in ("weekday", "saturday", "sunday", "holiday"):
        assert (out_dir / "service_cut_inputs" / f"ridership_by_trip_{day}.csv").exists()
    assert (out_dir / "ridership_from_tides_runlog.txt").exists()
    assert not (out_dir / "charts").exists()


def test_grain_filenames() -> None:
    """Each time column adds an _and_<column> suffix."""
    assert target.grain_filename("ridership", "stop", "all_day") == "ridership_by_stop.csv"
    assert (
        target.grain_filename("ridership", "route_direction_stop", "month_time_period")
        == "ridership_by_route_direction_stop_and_month_and_time_period.csv"
    )


@pytest.mark.parametrize(("level", "cut"), ALL_GRAINS)
def test_every_grain_reconciles_with_a_direct_join(
    full_run: "target.RunResult", level: str, cut: str
) -> None:
    """Each table sums back to the boardings and stop visits of a direct join."""
    expected = _expected_visits()
    table = full_run.ridership[(level, cut)]
    assert table["total_boardings"].sum() == pytest.approx(expected["boardings"].sum())
    assert table["stop_visits"].sum() == len(expected)
    keys = ["day_type", *target.level_columns(level, table.columns), *target.TIME_CUTS[cut]]
    assert not table.duplicated(keys).any()


def test_averages_divide_by_observed_dates(full_run: "target.RunResult") -> None:
    """System averages use the dates with data in each day type; the holiday is its own."""
    table = full_run.ridership[("system", "all_day")]
    days = table.set_index("day_type")["days_observed"].to_dict()
    assert days == {"Weekday": 5, "Saturday": 3, "Sunday": 3, "Holiday": 1}
    assert table["day_type"].tolist() == ["Weekday", "Saturday", "Sunday", "Holiday"]
    expected = (table["total_boardings"] / table["days_observed"]).round(2)
    assert table["avg_daily_boardings"].tolist() == expected.tolist()

    expected_visits = _expected_visits()
    holiday_boardings = expected_visits.loc[
        expected_visits["service_date"] == pd.Timestamp(HOLIDAY), "boardings"
    ].sum()
    holiday_row = table.set_index("day_type").loc["Holiday"]
    assert holiday_row["total_boardings"] == pytest.approx(holiday_boardings)


def test_route_averages_share_the_system_divisor(full_run: "target.RunResult") -> None:
    """Route rows divide by the system's dates per day type (counted before filters)."""
    route = full_run.ridership[("route", "all_day")]
    system = full_run.ridership[("system", "all_day")].set_index("day_type")["days_observed"]
    assert (route["days_observed"] == route["day_type"].map(system)).all()


def test_trip_averages_divide_by_the_trips_own_dates(full_run: "target.RunResult") -> None:
    """A trip's average divides by the dates that trip was observed in the day type."""
    trip = full_run.ridership[("trip", "all_day")]
    expected = _expected_visits().assign(trip_id=lambda d: d["trip_id_performed"])
    tp = pd.read_csv(TRIPS_PERFORMED, dtype=str)
    tp["service_date"] = pd.to_datetime(tp["service_date"])
    expected = expected.merge(
        tp[["service_date", "trip_id_performed", "trip_id_scheduled"]],
        on=["service_date", "trip_id_performed"],
    )
    weekday = expected[
        (expected["service_date"].dt.dayofweek < 5)
        & (expected["service_date"] != pd.Timestamp(HOLIDAY))
    ]
    dates = weekday.groupby("trip_id_scheduled")["service_date"].nunique()
    rows = trip[trip["day_type"] == "Weekday"].set_index("trip_id")["days_observed"]
    assert rows.sort_index().to_dict() == dates.sort_index().to_dict()
    assert (rows <= 5).all() and (rows < 5).any()


def test_time_period_rows_are_zero_filled_for_routes_not_stops(
    full_run: "target.RunResult",
) -> None:
    """Early AM has no fixture visits: system rows show zero; stop rows are absent."""
    system = full_run.ridership[("system", "time_period")]
    early = system[system["time_period"] == "Early AM"]
    assert len(early) == 4
    assert (early["total_boardings"] == 0).all() and (early["stop_visits"] == 0).all()
    per_group = system.groupby("day_type")["time_period"].agg(list)
    assert all(names == [p.name for p in DEFAULT_PERIODS] for names in per_group)
    stop = full_run.ridership[("stop", "time_period")]
    assert "Early AM" not in set(stop["time_period"])


def test_period_rates_and_windows(full_run: "target.RunResult") -> None:
    """Period rows carry their window, length and boardings per period hour."""
    table = full_run.ridership[("system", "time_period")]
    am = table[(table["day_type"] == "Weekday") & (table["time_period"] == "AM Peak")].iloc[0]
    assert (am["period_start"], am["period_end"], am["period_hours"]) == ("06:00", "09:00", 3.0)
    assert am["avg_daily_boardings_per_period_hour"] == pytest.approx(
        am["total_boardings"] / 5 / 3, abs=0.01
    )


def test_route_rows_add_up_to_system_rows(full_run: "target.RunResult") -> None:
    """Summing route rows over routes reproduces the system rows, cut by cut."""
    for cut in ("all_day", "time_period", "hour", "month"):
        keys = ["day_type", *target.TIME_CUTS[cut]]
        route = full_run.ridership[("route", cut)].groupby(keys)["total_boardings"].sum()
        system = full_run.ridership[("system", cut)].set_index(keys)["total_boardings"]
        pd.testing.assert_series_equal(route.sort_index(), system.sort_index(), check_names=False)


def test_hours_use_the_service_day_clock(full_run: "target.RunResult") -> None:
    """Owl visits land on hours 24 and up; no hour falls before the cutover."""
    hours = full_run.ridership[("system", "hour")]
    assert hours["hour"].min() >= target.LATE_NIGHT_CUTOVER_HOUR
    assert hours["hour"].max() >= 24
    assert hours.loc[hours["hour"] == 25, "hour_label"].iloc[0] == "25:00"


def test_month_divisors_add_up_to_the_day_type(full_run: "target.RunResult") -> None:
    """Monthly rows divide by that month's observed dates of the day type."""
    monthly = full_run.ridership[("system", "month")]
    assert set(monthly["month"]) == {"2026-01", "2026-02", "2026-03"}
    totals = monthly.groupby("day_type")["days_observed"].sum().to_dict()
    assert totals == {"Weekday": 5, "Saturday": 3, "Sunday": 3, "Holiday": 1}


def test_without_holidays_the_date_counts_as_its_weekday(tmp_path: Path) -> None:
    """With HOLIDAYS empty, the holiday's Monday is one more weekday."""
    result = target.run(_settings(tmp_path, levels=["system"], time_cuts=["all_day"]))
    days = result.ridership[("system", "all_day")].set_index("day_type")["days_observed"]
    assert days.to_dict() == {"Weekday": 6, "Saturday": 3, "Sunday": 3}


def test_day_types_and_date_range(tmp_path: Path) -> None:
    """Custom day types split the week, and the date range keeps only its dates."""
    result = target.run(
        _settings(
            tmp_path,
            levels=["system"],
            time_cuts=["month"],
            day_types={"Weekend": ("Sat", "Sun")},
            start_date=dt.date(2026, 2, 1),
            end_date=dt.date(2026, 3, 31),
        )
    )
    table = result.ridership[("system", "month")]
    assert set(table["day_type"]) == {"Weekend"}
    assert table.set_index("month")["days_observed"].to_dict() == {"2026-02": 2, "2026-03": 2}
    with pytest.raises(target.InputDataError, match="START_DATE"):
        target.run(_settings(tmp_path, start_date=dt.date(2027, 1, 1)))


def test_missing_column_skips_its_levels(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Without direction_id, the direction levels are skipped and the rest still run."""
    trips = pd.read_csv(TRIPS_PERFORMED, dtype=str).drop(columns="direction_id")
    trips_path = tmp_path / "trips.csv"
    trips.to_csv(trips_path, index=False)
    with caplog.at_level(logging.WARNING):
        result = target.run(
            _settings(
                tmp_path,
                trips_performed_path=trips_path,
                levels=["route", "route_direction", "trip"],
                time_cuts=["all_day"],
            )
        )
    assert set(result.ridership) == {("route", "all_day"), ("trip", "all_day")}
    assert "Skipping the route_direction level: the data has no direction_id column" in caplog.text
    assert "direction_id" not in result.ridership[("trip", "all_day")].columns


def test_flags_limit_what_is_written(tmp_path: Path) -> None:
    """Only the chosen levels and cuts are written; maximum load can be switched off."""
    result = target.run(
        _settings(
            tmp_path,
            levels=["stop"],
            time_cuts=["all_day"],
            export_max_load=False,
            export_service_cut_tables=False,
        )
    )
    assert list(result.ridership) == [("stop", "all_day")]
    assert sorted(p.name for p in (tmp_path / "ridership").iterdir()) == ["ridership_by_stop.csv"]
    assert not (tmp_path / "max_load").exists()
    assert not (tmp_path / "service_cut_inputs").exists()


def test_unknown_or_empty_flags_are_settings_errors(tmp_path: Path) -> None:
    """Unknown names and an empty selection stop the run before any data is read."""
    with pytest.raises(target.SettingsError, match="unknown entries: segment"):
        target.run(_settings(tmp_path, levels=["segment"]))
    with pytest.raises(target.SettingsError, match="every entry off"):
        target.resolve_flags({"stop": False}, list(target.LEVELS), "EXPORT_LEVELS")
    with pytest.raises(target.SettingsError, match="CHART_LEVELS"):
        target.run(_settings(tmp_path, chart_levels=["stop"]))
    assert not list(tmp_path.iterdir())


def test_route_filters(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Include keeps only the named routes; unknown entries are warned about."""
    with caplog.at_level(logging.WARNING):
        result = target.run(
            _settings(
                tmp_path,
                levels=["route"],
                time_cuts=["all_day"],
                routes_to_include=("DC_R10", "NOPE"),
            )
        )
    assert set(result.ridership[("route", "all_day")]["route_id"]) == {"DC_R10"}
    assert set(result.max_load_by_trip["route_id"]) == {"DC_R10"}
    assert "NOPE" in caplog.text
    result = target.run(
        _settings(tmp_path, levels=["route"], time_cuts=["all_day"], routes_to_exclude=("DC_R10",))
    )
    assert "DC_R10" not in set(result.ridership[("route", "all_day")]["route_id"])


def test_run_rejects_uncovered_periods(tmp_path: Path) -> None:
    """run() re-checks periods built by hand and writes nothing when they leave a gap."""
    periods = [target.TimePeriod("Day", 0, 12 * 60)]
    with pytest.raises(target.TimePeriodError, match="no period covers 12:00-24:00"):
        target.run(_settings(tmp_path, periods=periods))
    assert not list(tmp_path.iterdir())


def test_service_cut_tables(full_run: "target.RunResult") -> None:
    """Per-day-type trip x stop averages match the trip_stop level's averages."""
    weekday = full_run.service_cut_tables["Weekday"]
    assert list(weekday.columns) == ["trip_id", "stop_id", "avg_daily_boardings"]
    trip_stop = full_run.ridership[("trip_stop", "all_day")]
    trip_stop = trip_stop[trip_stop["day_type"] == "Weekday"]
    merged = weekday.merge(trip_stop, on=["trip_id", "stop_id"], suffixes=("", "_level"))
    assert len(merged) == len(weekday)
    assert (merged["avg_daily_boardings"] == merged["avg_daily_boardings_level"]).all()


# --- maximum load -----------------------------------------------------------------


def test_compute_trip_max_load_is_per_trip_peak() -> None:
    """Each trip's max load is the largest departure_load; ties go to the earliest stop."""
    sv = pd.DataFrame(
        {
            "trip_id_performed": ["A", "A", "A", "B", "B"],
            "trip_stop_sequence": ["1", "2", "3", "1", "2"],
            "stop_id": ["s1", "s2", "s3", "s1", "s2"],
            "departure_load": ["2", "9", "4", "1", "1"],
            "service_date": pd.to_datetime(["2025-01-02"] * 5),
        }
    )
    out = target.compute_trip_max_load(sv).set_index("trip_id_performed")
    assert out.loc["A", "max_load"] == 9
    assert out.loc["A", "peak_stop_id"] == "s2"
    assert out.loc["B", "peak_stop_id"] == "s1"


def test_compute_trip_max_load_drops_all_nan_trip_and_splits_days() -> None:
    """A trip with no load reading is dropped; the same ID on two days is two trips."""
    sv = pd.DataFrame(
        {
            "trip_id_performed": ["T100"] * 4 + ["B"],
            "trip_stop_sequence": [1, 2, 1, 2, 1],
            "stop_id": ["s1", "s2", "s1", "s2", "s1"],
            "departure_load": [7, 3, 2, 11, None],
            "service_date": pd.to_datetime(
                ["2026-03-02"] * 2 + ["2026-03-03"] * 2 + ["2026-03-02"]
            ),
        }
    )
    out = target.compute_trip_max_load(sv).sort_values("service_date")
    assert out["trip_id_performed"].tolist() == ["T100", "T100"]
    assert out["max_load"].tolist() == [7, 11]
    assert out["peak_stop_id"].tolist() == ["s1", "s2"]


def test_max_load_places_trips_by_start_time() -> None:
    """A trip falls in the period and service-day hour of its start time."""
    joined = pd.DataFrame(
        {
            "service_date": pd.to_datetime(["2026-03-02"] * 3),
            "trip_id_performed": ["A", "B", "C"],
            "trip_id": ["A", "B", "C"],
            "departure_load": [10, 20, 39],
            "schedule_trip_start": ["2026-03-02T07:30:00", "2026-03-03T00:15:00", ""],
            "actual_trip_start": ["", "", "2026-03-02T17:00:00"],
        }
    )
    out = target.build_trip_max_load(joined, DEFAULT_PERIODS, 39).set_index("trip_id")
    assert out["time_period"].to_dict() == {"A": "AM Peak", "B": "Night", "C": "PM Peak"}
    assert out["hour"].to_dict() == {"A": 7, "B": 24, "C": 17}
    assert out.loc["C", "load_factor"] == 1.0


def test_max_load_tables(full_run: "target.RunResult") -> None:
    """Per performed trip: one row each; rollups agree with it."""
    by_trip = full_run.max_load_by_trip
    assert not by_trip.duplicated(["service_date", "trip_id_performed", "day_type"]).any()
    expected_trips = _expected_visits()[["service_date", "trip_id_performed"]].drop_duplicates()
    assert len(by_trip) == len(expected_trips)
    system = full_run.max_load[("system", "all_day")].set_index("day_type")
    for day_type, sub in by_trip.groupby("day_type"):
        assert system.loc[day_type, "trips"] == len(sub)
        assert system.loc[day_type, "peak_max_load"] == sub["max_load"].max()
        over = (sub["max_load"] > target.VEHICLE_CAPACITY).mean() * 100
        assert system.loc[day_type, "pct_trips_over_capacity"] == pytest.approx(over, abs=0.05)
    trip = full_run.max_load[("trip", "all_day")]
    assert trip["trips"].sum() == len(by_trip)


# --- low ridership and charts ---------------------------------------------------------


def test_low_ridership_flag(tmp_path: Path) -> None:
    """Trips below the threshold are written with their rates; 0 switches the flag off."""
    result = target.run(
        _settings(
            tmp_path,
            levels=["system"],
            time_cuts=["all_day"],
            low_ridership_threshold=1000.0,
            export_max_load=False,
        )
    )
    rows = result.trip_rows
    assert rows["pass_per_hour"].notna().all()
    flagged = pd.read_csv(tmp_path / "low_ridership_trips.csv")
    assert len(flagged) == len(rows)
    assert flagged["start_time"].str.match(r"^\d{2}:\d{2}$").all()
    off = tmp_path / "off"
    target.run(
        _settings(
            off,
            levels=["system"],
            time_cuts=["all_day"],
            low_ridership_threshold=0.0,
            low_ridership_measure="boardings",
            export_max_load=False,
        )
    )
    assert not (off / "low_ridership_trips.csv").exists()


def test_charts_are_written(tmp_path: Path) -> None:
    """Charts cover the chart levels by hour, plus per-trip charts."""
    target.run(
        _settings(
            tmp_path,
            levels=["system"],
            time_cuts=["hour"],
            export_max_load=False,
            export_charts=True,
            chart_levels=["system"],
            routes_to_include=("DC_R20",),  # one small route keeps the PNG count low
        )
    )
    hour_charts = list((tmp_path / "charts" / "ridership" / "system_by_hour").glob("*.png"))
    # DC_R20 is a weekday-only express, so only the Weekday system chart has data.
    assert [p.name for p in hour_charts] == ["system_Weekday.png"]
    trip_charts = sorted(p.name for p in (tmp_path / "charts" / "trips").glob("*.png"))
    assert trip_charts and all(name.startswith("route_DC_R20_dir") for name in trip_charts)


# --- main -----------------------------------------------------------------------


def test_main_stops_on_bad_periods_before_reading_data(caplog: pytest.LogCaptureFixture) -> None:
    """Overlapping flags exit 2 with the problem named, even with placeholder paths."""
    with caplog.at_level(logging.ERROR):
        code = target.main(["--time-period", "A=00:00-12:00", "--time-period", "B=11:00-24:00"])
    assert code == 2
    assert "overlap at 11:00-12:00" in caplog.text


def test_main_bad_holiday_and_placeholders_exit_2(caplog: pytest.LogCaptureFixture) -> None:
    """A malformed holiday is a settings error; placeholder paths are too."""
    with caplog.at_level(logging.ERROR):
        assert target.main(["--holiday", "Feb 16"]) == 2
    assert "'Feb 16' is not a YYYY-MM-DD date" in caplog.text
    assert target.main([]) == 2


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


def test_main_end_to_end(tmp_path: Path) -> None:
    """A CLI run with the periods fixture writes the chosen grains and a full run log."""
    out_dir = tmp_path / "out"
    code = target.main(
        [
            "--stop-visits",
            str(STOP_VISITS),
            "--trips-performed",
            str(TRIPS_PERFORMED),
            "--output-dir",
            str(out_dir),
            "--time-periods-file",
            str(PERIODS_FIXTURE),
            "--levels",
            "route_direction_stop",
            "route",
            "--time-cuts",
            "time_period",
            "--holiday",
            HOLIDAY.isoformat(),
            "--no-charts",
        ]
    )
    assert code == 0
    written = sorted(p.name for p in (out_dir / "ridership").iterdir())
    assert written == [
        "ridership_by_route_and_time_period.csv",
        "ridership_by_route_direction_stop_and_time_period.csv",
    ]
    table = pd.read_csv(out_dir / "ridership" / written[1], dtype=str)
    assert set(table["day_type"]) == {"Weekday", "Saturday", "Sunday", "Holiday"}
    log = (out_dir / "ridership_from_tides_runlog.txt").read_text("utf-8")
    assert "time_periods.txt" in log
    assert "AM Peak = 06:00 - 09:00" in log
    assert "Holidays (Holiday): 2026-02-16" in log
    assert "Levels:             route, route_direction_stop" in log
    assert "# === BEGIN CONFIG ===" not in log
    assert "EXPORT_LEVELS: Mapping" in log
