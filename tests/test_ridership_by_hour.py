"""Tests for ridership_by_hour trip-start parsing and workbook loading.

Focuses on the Excel midnight quirk: a 00:00 trip start is stored as time
serial zero, which Excel displays — and some vendor exports save as literal
text — as the epoch date "1/0/1900" rather than a time. Such trips must load
as midnight (service-day hour 24), not be dropped as unparseable.
"""

import datetime

import openpyxl
import pytest

from scripts.ridership_tools import ridership_by_hour

_HEADER = [
    "ROUTE_NUMBER",
    "ROUTE_NAME",
    "DIRECTION_NAME",
    "TRIP_NUMBER",
    "TRIP_START_TIME",
    "PASSENGERS_ON",
    "PASSENGERS_OFF",
    "TRIPS_COUNT",
    "REVENUE_HOURS",
    "REVENUE_MILES",
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("05:13", 313),
        ("05:13:00", 313),
        ("26:30:00", 1590),  # GTFS-style owl time passes straight through
        ("1899-12-30 00:00:00", 0),  # datetime-typed cell stringified by the reader
        ("1900-01-01 05:13:00", 313),
        ("1/0/1900", 0),  # Excel's rendering of time serial zero (midnight)
        ("12/30/1899", 0),
        ("", None),
        (None, None),
        ("not a time", None),
    ],
)
def test_trip_start_minutes(value, expected) -> None:
    assert ridership_by_hour._trip_start_minutes(value) == expected


def test_load_route_trip_workbook_keeps_midnight_trips(tmp_path) -> None:
    """Serial-zero and '1/0/1900'-text midnight starts load as hour 24."""
    path = tmp_path / "RIDERSHIP_BY_ROUTE_AND_TRIP_weekday.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(_HEADER)
    # Midnight stored as a numeric time cell (serial 0, formatted h:mm).
    sheet.append([3695, "RIBS5", "LOOP", 44, 0, 4.14, 4.14, 152, 0.75, 9.04])
    sheet["E2"].number_format = "h:mm"
    # Midnight saved as the literal text Excel displays for serial zero.
    sheet.append([3695, "RIBS5", "LOOP", 46, "1/0/1900", 5.0, 5.0, 150, 0.75, 9.04])
    # An ordinary daytime trip for contrast.
    sheet.append([3695, "RIBS5", "LOOP", 48, datetime.time(5, 13), 4.2, 4.2, 188, 0.9, 13.93])
    workbook.save(path)

    out = ridership_by_hour.load_route_trip_workbook(path, "weekday")

    assert len(out) == 3, "no row should be dropped as unparseable"
    by_trip = out.set_index("trip_id")
    # LATE_NIGHT_CUTOVER_HOUR (default 4) shifts midnight onto the 24+ clock.
    assert by_trip.loc["44", "hour"] == 24
    assert by_trip.loc["44", "start_time"] == "00:00"
    assert by_trip.loc["46", "hour"] == 24
    assert by_trip.loc["46", "start_time"] == "00:00"
    assert by_trip.loc["48", "hour"] == 5
    assert by_trip.loc["48", "start_time"] == "05:13"
    assert (by_trip["direction"] == "LOOP").all()
