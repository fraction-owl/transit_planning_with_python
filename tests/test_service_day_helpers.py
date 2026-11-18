"""Tests for utils/service_day_helpers.py: time periods, the service-day clock and day types."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from utils import service_day_helpers as sdh

DEFAULT_MAPPING = {
    "Early AM": ("04:00", "06:00"),
    "AM Peak": ("06:00", "09:00"),
    "Midday": ("09:00", "15:00"),
    "PM Peak": ("15:00", "18:00"),
    "Late PM": ("18:00", "22:00"),
    "Night": ("22:00", "04:00"),
}
EXAMPLE_PERIODS = Path("sample_data/time_periods_example.txt")
PERIODS_FIXTURE = Path("tests/fixtures/time_periods.txt")


def _periods(*specs: str) -> list[sdh.TimePeriod]:
    """Build checked periods from NAME=HH:MM-HH:MM strings."""
    parsed = [sdh.parse_period_spec(s, f"spec {i}") for i, s in enumerate(specs)]
    return sdh.build_time_periods(parsed, "test")


# --- time periods ---------------------------------------------------------------


def test_inline_periods_cover_the_day() -> None:
    periods = sdh.build_time_periods(sdh.specs_from_mapping(DEFAULT_MAPPING), "test")
    assert [p.name for p in periods] == list(DEFAULT_MAPPING)
    assert sum(p.minutes for p in periods) == sdh.MINUTES_PER_DAY
    assert sdh.find_coverage_problems(periods) == []
    assert periods[-1].span == "22:00-04:00"


@pytest.mark.parametrize("path", [EXAMPLE_PERIODS, PERIODS_FIXTURE])
def test_periods_files_match_the_inline_mapping(path: Path) -> None:
    inline = sdh.build_time_periods(sdh.specs_from_mapping(DEFAULT_MAPPING), "inline")
    assert sdh.build_time_periods(sdh.load_time_periods_file(path), str(path)) == inline


def test_gaps_and_overlaps_are_reported() -> None:
    with pytest.raises(sdh.TimePeriodError, match="no period covers 23:00-01:00"):
        _periods("A=01:00-23:00")
    with pytest.raises(sdh.TimePeriodError, match="overlap at 09:00-09:30"):
        _periods("AM=00:00-09:30", "Midday=09:00-15:00", "PM=15:00-24:00")


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ("A=6am-24:00", "'6am' is not a 24-hour HH:MM time"),
        ("A=00:00-25:00", "'25:00' is outside 00:00-24:00"),
        ("A=24:00-12:00", "cannot start at 24:00"),
        ("A=04:00-04:00", "starts and ends at 04:00"),
        (" =00:00-24:00", "the period name is blank"),
    ],
)
def test_malformed_periods_are_reported(spec: str, fragment: str) -> None:
    with pytest.raises(sdh.TimePeriodError, match=fragment):
        _periods(spec)


def test_midnight_ends_and_whole_day() -> None:
    (whole,) = _periods("All=00:00-24:00")
    assert whole.minutes == sdh.MINUTES_PER_DAY
    _, evening = _periods("Day=00:00-18:00", "Evening=18:00-00:00")
    assert evening.end == sdh.MINUTES_PER_DAY


def test_mapping_entry_must_be_a_pair() -> None:
    with pytest.raises(sdh.TimePeriodError, match=r"TIME_PERIODS\['AM'\]"):
        sdh.specs_from_mapping({"AM": "06:00-09:00"})
    with pytest.raises(sdh.TimePeriodError, match="No time periods"):
        sdh.build_time_periods([], "test")


def test_file_lines_comments_and_errors(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("# header\nDay=04:00-22:00  # note\n\nNight = 22:00 – 04:00\n", "utf-8-sig")
    specs = sdh.load_time_periods_file(good)
    assert [(s.name, s.start, s.end, s.where) for s in specs] == [
        ("Day", "04:00", "22:00", "line 2"),
        ("Night", "22:00", "04:00", "line 4"),
    ]
    bad = tmp_path / "bad.txt"
    bad.write_text("Day = 04:00 - 22:00\nNight 22:00 - 04:00\n", encoding="utf-8")
    with pytest.raises(sdh.TimePeriodError, match="line 2: expected NAME = HH:MM - HH:MM"):
        sdh.load_time_periods_file(bad)
    with pytest.raises(sdh.TimePeriodError, match="Could not read"):
        sdh.load_time_periods_file(tmp_path / "missing.txt")


def test_resolve_precedence(tmp_path: Path) -> None:
    path = tmp_path / "periods.txt"
    path.write_text("FromFile = 00:00 - 24:00\n", encoding="utf-8")
    inline = {"Inline": ("00:00", "24:00")}
    periods, source = sdh.resolve_time_periods(["Flag=00:00-24:00"], str(path), inline)
    assert [p.name for p in periods] == ["Flag"] and "flags" in source
    periods, source = sdh.resolve_time_periods(None, str(path), inline)
    assert [p.name for p in periods] == ["FromFile"] and "periods.txt" in source
    periods, source = sdh.resolve_time_periods(None, None, inline)
    assert [p.name for p in periods] == ["Inline"] and "CONFIGURATION" in source


def test_format_period_and_lookup() -> None:
    day, night = _periods("Day=04:00-22:00", "Night=22:00-04:00")
    assert sdh.format_period(night) == "Night = 22:00 - 04:00"
    lookup = sdh.period_lookup([day, night])
    assert (lookup[0], lookup[239], lookup[240], lookup[1319], lookup[1320]) == (
        "Night",
        "Night",
        "Day",
        "Day",
        "Night",
    )
    assert int(sdh.minute_mask(night).sum()) == night.minutes


# --- service-day clock ------------------------------------------------------------


def test_minute_of_day_reads_clock_time_as_written() -> None:
    values = pd.Series(
        ["2026-03-07T21:03:14-05:00", "21:03:14", "25:10:00", "1/5/2026 9:03:14 PM", "", None],
        dtype=object,
    )
    minutes = sdh.minute_of_day(values)
    assert minutes.iloc[:4].tolist() == [1263, 1263, 70, 1263]
    assert minutes.iloc[4:].isna().all()


def test_service_day_hour_moves_small_hours_past_24() -> None:
    values = pd.Series(
        ["2026-01-05T21:03:14", "2026-01-06T00:30:00", "2026-01-05T01:15:00", "02:00:00", ""],
        dtype=object,
    )
    dates = pd.Series(pd.to_datetime(["2026-01-05"] * len(values)))
    hours = sdh.service_day_hour(values, dates, 4)
    assert hours.iloc[:4].tolist() == [21, 24, 25, 26]
    assert pd.isna(hours.iloc[4])


# --- day types and dates ------------------------------------------------------------


def test_build_day_types_reads_names_and_abbreviations() -> None:
    out = sdh.build_day_types({"Mon-Thu": ("mon", "Tue", "WEDNESDAY", "thu"), "Fri": "Friday"})
    assert out == {"Mon-Thu": (0, 1, 2, 3), "Fri": (4,)}


@pytest.mark.parametrize(
    ("mapping", "fragment"),
    [
        ({"Weekday": ("Funday",)}, "'Funday' is not a weekday name"),
        ({"Holiday": ("Mon",)}, "'Holiday' is already used"),
        ({"Weekday": ()}, "lists no weekdays"),
        ({}, "DAY_TYPES is empty"),
    ],
)
def test_build_day_types_problems(mapping: dict, fragment: str) -> None:
    with pytest.raises(sdh.SettingsError, match=fragment):
        sdh.build_day_types(mapping)


def test_parse_dates() -> None:
    assert sdh.parse_iso_dates(["2026-02-16", "2026-01-19", "2026-02-16"], "HOLIDAYS") == [
        dt.date(2026, 1, 19),
        dt.date(2026, 2, 16),
    ]
    with pytest.raises(sdh.SettingsError, match="'2/16/2026' is not a YYYY-MM-DD date"):
        sdh.parse_iso_dates(["2/16/2026"], "HOLIDAYS")
    assert sdh.parse_optional_date("  ", "START_DATE") is None
    assert sdh.parse_optional_date("2026-01-05", "START_DATE") == dt.date(2026, 1, 5)
