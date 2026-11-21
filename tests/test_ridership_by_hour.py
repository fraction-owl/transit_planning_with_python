"""Tests for ridership_by_hour trip-start parsing and workbook loading.

Focuses on the Excel midnight quirk: a 00:00 trip start is stored as time
serial zero, which Excel displays — and some vendor exports save as literal
text — as the epoch date "1/0/1900" rather than a time. Such trips must load
as midnight (service-day hour 24), not be dropped as unparseable.
"""

import datetime
from pathlib import Path

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


@pytest.mark.parametrize(
    ("route", "name", "expected_title", "expected_token"),
    [
        ("3695", "RIBS5", "Route RIBS5 (3695)", "RIBS5"),  # name leads, code kept visible
        ("232", "232", "Route 232", "232"),  # name == code: no redundant suffix
        ("451", "", "Route 451", "451"),  # unnamed route falls back to the code
        ("451", None, "Route 451", "451"),
    ],
)
def test_route_display(route, name, expected_title, expected_token) -> None:
    assert ridership_by_hour._route_display(route, name) == (expected_title, expected_token)


def _ribs5_workbook(tmp_path) -> Path:
    """A two-trip workbook whose route is numbered 3695 but named RIBS5."""
    path = tmp_path / "RIDERSHIP_BY_ROUTE_AND_TRIP_weekday.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(_HEADER)
    sheet.append([3695, "RIBS5", "LOOP", 44, datetime.time(6, 15), 11.0, 11.0, 150, 0.75, 9.04])
    sheet.append([3695, "RIBS5", "LOOP", 46, datetime.time(7, 15), 8.0, 8.0, 150, 0.75, 9.04])
    workbook.save(path)
    return path


def test_route_filter_matches_the_route_name(tmp_path, monkeypatch) -> None:
    """The name planners recognise is what the filters key on."""
    path = _ribs5_workbook(tmp_path)
    monkeypatch.setattr(ridership_by_hour, "ROUTES_TO_EXCLUDE", ("RIBS5",))
    with pytest.raises(ValueError, match="No usable trip rows"):
        ridership_by_hour.build_hourly_from_xlsx({"weekday": str(path)})

    monkeypatch.setattr(ridership_by_hour, "ROUTES_TO_EXCLUDE", ())
    monkeypatch.setattr(ridership_by_hour, "ROUTES_TO_INCLUDE", ("RIBS5",))
    _, _, rows = ridership_by_hour.build_hourly_from_xlsx({"weekday": str(path)})
    assert len(rows) == 2


def test_route_filter_on_the_code_warns_and_names_the_route(tmp_path, monkeypatch, caplog) -> None:
    """A code-based entry no longer filters, and says which name to use instead."""
    path = _ribs5_workbook(tmp_path)
    monkeypatch.setattr(ridership_by_hour, "ROUTES_TO_EXCLUDE", ("3695",))
    with caplog.at_level("WARNING"):
        _, _, rows = ridership_by_hour.build_hourly_from_xlsx({"weekday": str(path)})
    assert len(rows) == 2, "the code must not silently filter the route out"
    assert "RIBS5" in caplog.text and "3695" in caplog.text


def test_unmatched_route_filter_is_never_silent(tmp_path, monkeypatch, caplog) -> None:
    path = _ribs5_workbook(tmp_path)
    monkeypatch.setattr(ridership_by_hour, "ROUTES_TO_EXCLUDE", ("RIBS6",))
    with caplog.at_level("WARNING"):
        ridership_by_hour.build_hourly_from_xlsx({"weekday": str(path)})
    assert "RIBS6" in caplog.text


def test_trip_chart_filename_uses_the_route_name(tmp_path) -> None:
    """Charts file under the name a planner would look for, not the code."""
    path = _ribs5_workbook(tmp_path)
    _, _, rows = ridership_by_hour.build_hourly_from_xlsx({"weekday": str(path)})
    out_dir = tmp_path / "charts"
    assert ridership_by_hour.export_trip_charts(rows, "pass_per_hour", 5.0, out_dir) == 1
    assert (out_dir / "route_RIBS5_LOOP_weekday.png").exists()


def _one_trip_workbook(path: Path, boardings: float, alightings: object = 1.0) -> Path:
    """One weekday trip at 06:15 with one revenue hour; alightings=None drops the column."""
    header = [h for h in _HEADER if alightings is not None or h != "PASSENGERS_OFF"]
    row = [3695, "RIBS5", "LOOP", 44, datetime.time(6, 15), boardings, alightings, 150, 1.0, 9.0]
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(header)
    sheet.append([v for h, v in zip(_HEADER, row) if h in header])
    workbook.save(path)
    return path


def test_rerun_replaces_the_low_ridership_list(tmp_path, monkeypatch) -> None:
    """A clean rerun leaves a header-only CSV, not the previous run's flagged trips."""
    import pandas as pd

    monkeypatch.setattr(ridership_by_hour, "LOW_RIDERSHIP_MEASURE", "pass_per_hour")
    monkeypatch.setattr(ridership_by_hour, "LOW_RIDERSHIP_THRESHOLD", 5.0)
    out_dir = tmp_path / "out"
    low_csv = out_dir / ridership_by_hour.LOW_RIDERSHIP_FILENAME

    first = _one_trip_workbook(tmp_path / "first.xlsx", boardings=2.0)
    ridership_by_hour.run(out_dir, {"weekday": str(first)}, export_charts_flag=False)
    flagged = pd.read_csv(low_csv)
    assert flagged["pass_per_hour"].tolist() == [2.0]

    second = _one_trip_workbook(tmp_path / "second.xlsx", boardings=12.0)
    ridership_by_hour.run(out_dir, {"weekday": str(second)}, export_charts_flag=False)
    rerun = pd.read_csv(low_csv)
    assert rerun.empty
    assert list(rerun.columns) == list(flagged.columns)


def test_missing_alighting_data_exports_blank_not_zero(tmp_path) -> None:
    """No PASSENGERS_OFF column -> blank alightings; a recorded zero stays 0.0."""
    import pandas as pd

    missing = _one_trip_workbook(tmp_path / "missing.xlsx", boardings=4.0, alightings=None)
    ridership_by_hour.run(tmp_path / "missing", {"weekday": str(missing)}, False)
    for name in (ridership_by_hour.ROUTE_OUTPUT_FILENAME, ridership_by_hour.SYSTEM_OUTPUT_FILENAME):
        table = pd.read_csv(tmp_path / "missing" / name)
        assert table["alightings"].isna().all(), name
        assert table["boardings"].tolist() == [4.0], name

    zero = _one_trip_workbook(tmp_path / "zero.xlsx", boardings=4.0, alightings=0.0)
    ridership_by_hour.run(tmp_path / "zero", {"weekday": str(zero)}, False)
    for name in (ridership_by_hour.ROUTE_OUTPUT_FILENAME, ridership_by_hour.SYSTEM_OUTPUT_FILENAME):
        table = pd.read_csv(tmp_path / "zero" / name)
        assert table["alightings"].tolist() == [0.0], name


def test_alighting_chart_draws_with_missing_alighting_data(tmp_path, monkeypatch) -> None:
    """Charting alightings when the source has none writes empty bars instead of failing."""
    monkeypatch.setattr(ridership_by_hour, "CHART_MEASURE", "alightings")
    monkeypatch.setattr(ridership_by_hour, "EXPORT_TRIP_CHARTS", False)
    missing = _one_trip_workbook(tmp_path / "missing.xlsx", boardings=4.0, alightings=None)
    ridership_by_hour.run(tmp_path / "out", {"weekday": str(missing)}, export_charts_flag=True)
    assert list((tmp_path / "out" / ridership_by_hour.CHARTS_SUBDIR).glob("*.png"))
