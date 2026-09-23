"""Tests for the trip-level ridership intake in service_cut_impact_gpd.

The fixtures mirror the shape of a real APC/ridecheck export: one row per trip
per surveyed day, keyed by route number and scheduled start time rather than by
GTFS trip_id.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.stop_analysis.service_cut_impact_gpd import (
    RidershipSpec,
    build_stop_id_lookup,
    load_ridership,
    match_trips_by_route_and_time,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def events() -> pd.DataFrame:
    """Three trips on route 42, plus one after-midnight trip on route 7."""
    return pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t2", "t2", "t3", "owl"],
            "stop_id": ["s1", "s2", "s1", "s2", "s1", "s9"],
            "dep_min": [443, 455, 503, 515, 563, 1530],  # 7:23, 8:23, 9:23, 25:30
            "route_id": ["r42", "r42", "r42", "r42", "r42", "r7"],
        }
    )


@pytest.fixture
def routes() -> pd.DataFrame:
    return pd.DataFrame({"route_id": ["r42", "r7"], "route_short_name": ["42", "7"]})


@pytest.fixture
def vendor_spec() -> RidershipSpec:
    """Column names as they arrive from the vendor workbook."""
    return RidershipSpec(
        boardings_col="PASSENGERS_ON",
        match_mode="route_start_time",
        route_col="ROUTE_NUMBER",
        start_time_col="TRIP_START_TIME",
    )


def gtfs_spec(**overrides: object) -> RidershipSpec:
    """Spec for a feed-native file joined on ``trip_id``.

    Stated explicitly rather than relying on ``RidershipSpec()`` defaults,
    which are deployment config an analyst is expected to edit.
    """
    fields: dict[str, object] = {
        "trip_col": "trip_id",
        "stop_col": "stop_id",
        "boardings_col": "avg_daily_boardings",
        "match_mode": "trip_id",
    }
    fields.update(overrides)
    return RidershipSpec(**fields)  # type: ignore[arg-type]


def _write(tmp_path: Path, frame: pd.DataFrame) -> str:
    path = tmp_path / "ridership.csv"
    frame.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# match_trips_by_route_and_time
# ---------------------------------------------------------------------------


def test_matches_each_row_to_the_nearest_departure(events, routes, vendor_spec) -> None:
    frame = pd.DataFrame(
        {"ROUTE_NUMBER": ["42", "42", "42"], "TRIP_START_TIME": ["7:23", "8:23", "9:23"]}
    )
    matched = match_trips_by_route_and_time(frame, events, routes, vendor_spec)
    assert list(matched) == ["t1", "t2", "t3"]


def test_route_short_name_and_route_id_both_resolve(events, routes, vendor_spec) -> None:
    frame = pd.DataFrame({"ROUTE_NUMBER": ["42", "r42"], "TRIP_START_TIME": ["7:23", "7:23"]})
    matched = match_trips_by_route_and_time(frame, events, routes, vendor_spec)
    assert list(matched) == ["t1", "t1"]


def test_after_midnight_row_matches_a_gtfs_25_30_departure(events, routes, vendor_spec) -> None:
    # The vendor writes 1:30 AM; GTFS holds the same departure as 25:30.
    frame = pd.DataFrame({"ROUTE_NUMBER": ["7"], "TRIP_START_TIME": ["1:30"]})
    matched = match_trips_by_route_and_time(frame, events, routes, vendor_spec)
    assert list(matched) == ["owl"]


def test_row_outside_the_tolerance_does_not_match(events, routes, vendor_spec) -> None:
    frame = pd.DataFrame({"ROUTE_NUMBER": ["42"], "TRIP_START_TIME": ["7:40"]})
    matched = match_trips_by_route_and_time(frame, events, routes, vendor_spec)
    assert list(matched) == [""]


def test_unknown_route_yields_no_match(events, routes, vendor_spec) -> None:
    frame = pd.DataFrame({"ROUTE_NUMBER": ["999"], "TRIP_START_TIME": ["7:23"]})
    matched = match_trips_by_route_and_time(frame, events, routes, vendor_spec)
    assert list(matched) == [""]


def test_unreadable_start_time_yields_no_match(events, routes, vendor_spec) -> None:
    # An Excel time-only cell that lost its time of day.
    frame = pd.DataFrame({"ROUTE_NUMBER": ["42"], "TRIP_START_TIME": ["not a time"]})
    matched = match_trips_by_route_and_time(frame, events, routes, vendor_spec)
    assert list(matched) == [""]


# ---------------------------------------------------------------------------
# load_ridership — column validation
# ---------------------------------------------------------------------------


def test_missing_boardings_column_lists_the_columns_found(tmp_path, events, routes) -> None:
    path = _write(tmp_path, pd.DataFrame({"ROUTE_NUMBER": ["42"], "TRIP_START_TIME": ["7:23"]}))
    spec = RidershipSpec(boardings_col="PASSENGERS_ON", match_mode="route_start_time")
    with pytest.raises(ValueError, match="PASSENGERS_ON"):
        load_ridership(path, spec, events=events, routes=routes)


def test_trip_id_mode_does_not_require_the_route_columns(tmp_path) -> None:
    path = _write(tmp_path, pd.DataFrame({"trip_id": ["t1"], "avg_daily_boardings": [10.0]}))
    stop_sums, trip_sums = load_ridership(path, gtfs_spec())
    assert stop_sums.empty
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t1": 10.0}


def test_unrecognised_match_mode_is_rejected(tmp_path) -> None:
    path = _write(tmp_path, pd.DataFrame({"trip_id": ["t1"], "avg_daily_boardings": [1.0]}))
    with pytest.raises(ValueError, match="RIDERSHIP_TRIP_MATCH_MODE"):
        load_ridership(path, gtfs_spec(match_mode="by_vibes"))


def test_route_start_time_without_a_feed_is_rejected(tmp_path, vendor_spec) -> None:
    path = _write(
        tmp_path,
        pd.DataFrame({"ROUTE_NUMBER": ["42"], "TRIP_START_TIME": ["7:23"], "PASSENGERS_ON": [5.0]}),
    )
    with pytest.raises(ValueError, match="needs the GTFS feed"):
        load_ridership(path, vendor_spec)


# ---------------------------------------------------------------------------
# load_ridership — survey-date averaging
# ---------------------------------------------------------------------------


def _three_day_file() -> pd.DataFrame:
    """Trip t1 observed on three days, t2 on two of them."""
    return pd.DataFrame(
        {
            "SURVEY_DATE": [
                "2026-03-02",
                "2026-03-03",
                "2026-03-04",
                "2026-03-02",
                "2026-03-03",
            ],
            "ROUTE_NUMBER": ["42"] * 5,
            "TRIP_START_TIME": ["7:23", "7:23", "7:23", "8:23", "8:23"],
            "PASSENGERS_ON": [10.0, 20.0, 30.0, 5.0, 7.0],
        }
    )


def test_without_a_date_column_a_multiday_file_sums(tmp_path, events, routes, vendor_spec) -> None:
    # Documents the overcount that RIDERSHIP_DATE_COL exists to prevent.
    path = _write(tmp_path, _three_day_file())
    _, trip_sums = load_ridership(path, vendor_spec, events=events, routes=routes)
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t1": 60.0, "t2": 12.0}


def test_date_column_averages_across_surveyed_days(tmp_path, events, routes) -> None:
    path = _write(tmp_path, _three_day_file())
    spec = RidershipSpec(
        boardings_col="PASSENGERS_ON",
        match_mode="route_start_time",
        date_col="SURVEY_DATE",
    )
    _, trip_sums = load_ridership(path, spec, events=events, routes=routes)
    # t1: (10+20+30)/3 = 20. t2 appeared on 2 of the 3 days: (5+7)/2 = 6.
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t1": 20.0, "t2": 6.0}


def test_several_rows_on_one_day_sum_before_the_average(tmp_path, events, routes) -> None:
    # Both directions of a loop on the same date must total, not average.
    frame = pd.DataFrame(
        {
            "SURVEY_DATE": ["2026-03-02", "2026-03-02", "2026-03-03"],
            "ROUTE_NUMBER": ["42"] * 3,
            "TRIP_START_TIME": ["7:23", "7:23", "7:23"],
            "PASSENGERS_ON": [10.0, 6.0, 20.0],
        }
    )
    path = _write(tmp_path, frame)
    spec = RidershipSpec(
        boardings_col="PASSENGERS_ON",
        match_mode="route_start_time",
        date_col="SURVEY_DATE",
    )
    _, trip_sums = load_ridership(path, spec, events=events, routes=routes)
    # Day one totals 16, day two 20 → mean 18, not (10+6+20)/3.
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t1": 18.0}


def test_date_averaging_applies_to_stop_grain_rows(tmp_path, events, routes) -> None:
    frame = pd.DataFrame(
        {
            "SURVEY_DATE": ["2026-03-02", "2026-03-03"],
            "ROUTE_NUMBER": ["42", "42"],
            "TRIP_START_TIME": ["7:23", "7:23"],
            "stop_id": ["s1", "s1"],
            "PASSENGERS_ON": [8.0, 12.0],
        }
    )
    path = _write(tmp_path, frame)
    spec = RidershipSpec(
        boardings_col="PASSENGERS_ON",
        match_mode="route_start_time",
        date_col="SURVEY_DATE",
    )
    stop_sums, trip_sums = load_ridership(path, spec, events=events, routes=routes)
    assert stop_sums.set_index(["trip_id", "stop_id"])["boardings"].to_dict() == {
        ("t1", "s1"): 10.0
    }
    assert trip_sums.empty


def test_missing_date_column_is_rejected(tmp_path, events, routes) -> None:
    path = _write(tmp_path, _three_day_file())
    spec = RidershipSpec(
        boardings_col="PASSENGERS_ON",
        match_mode="route_start_time",
        date_col="NOT_A_COLUMN",
    )
    with pytest.raises(ValueError, match="RIDERSHIP_DATE_COL"):
        load_ridership(path, spec, events=events, routes=routes)


def test_date_averaging_works_in_trip_id_mode(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "SURVEY_DATE": ["2026-03-02", "2026-03-03"],
            "trip_id": ["t1", "t1"],
            "avg_daily_boardings": [10.0, 20.0],
        }
    )
    path = _write(tmp_path, frame)
    _, trip_sums = load_ridership(path, gtfs_spec(date_col="SURVEY_DATE"))
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t1": 15.0}


# ---------------------------------------------------------------------------
# load_ridership — grain handling and stop_code resolution
# ---------------------------------------------------------------------------


def test_trip_grain_rows_are_dropped_for_trips_that_have_stop_grain(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t2"],
            "stop_id": ["s1", "", ""],
            "avg_daily_boardings": [4.0, 99.0, 7.0],
        }
    )
    path = _write(tmp_path, frame)
    stop_sums, trip_sums = load_ridership(path, gtfs_spec())
    assert stop_sums.set_index(["trip_id", "stop_id"])["boardings"].to_dict() == {("t1", "s1"): 4.0}
    # t1's whole-trip row is ignored so its boardings are not counted twice.
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t2": 7.0}


def test_stop_codes_translate_to_stop_ids(tmp_path) -> None:
    stops = pd.DataFrame({"stop_id": ["s1", "s2"], "stop_code": ["1001", "1002"]})
    lookup = build_stop_id_lookup(stops, "stop_code")
    frame = pd.DataFrame({"trip_id": ["t1"], "stop_id": ["1001"], "avg_daily_boardings": [3.0]})
    path = _write(tmp_path, frame)
    stop_sums, _ = load_ridership(path, gtfs_spec(), stop_lookup=lookup)
    assert stop_sums.set_index(["trip_id", "stop_id"])["boardings"].to_dict() == {("t1", "s1"): 3.0}


def test_non_numeric_boardings_rows_are_dropped(tmp_path) -> None:
    frame = pd.DataFrame({"trip_id": ["t1", "t2"], "avg_daily_boardings": ["10.0", "n/a"]})
    path = _write(tmp_path, frame)
    _, trip_sums = load_ridership(path, gtfs_spec())
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t1": 10.0}


def test_blank_path_disables_ridership() -> None:
    assert load_ridership("", gtfs_spec()) is None


def test_missing_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ridership(str(tmp_path / "nope.csv"), gtfs_spec())


# ---------------------------------------------------------------------------
# load_ridership — Excel workbook cell shapes
#
# A vendor .xlsx stores routes as numbers and start times as date-formatted
# serials near the Excel epoch, which spreadsheets display as "1/0/1900". These
# tests feed load_ridership a real workbook holding every shape at once.
# ---------------------------------------------------------------------------


@pytest.fixture
def vendor_xlsx(tmp_path: Path) -> str:
    """One workbook, four TRIP_START_TIME shapes on numeric route 232."""
    openpyxl = pytest.importorskip("openpyxl")
    import datetime as dt

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["SURVEY_DATE", "ROUTE_NUMBER", "TRIP_START_TIME", "PASSENGERS_ON"])
    # A date-formatted serial: displays as 1/0/1900 but still holds 07:23.
    ws.append([None, 232, 0.30763888888, 4.2])
    ws.cell(row=2, column=3).number_format = "m/d/yyyy"
    # A day-zero datetime, as openpyxl hands back cached date cells.
    ws.append([None, 232, dt.datetime(1899, 12, 31, 8, 23), 7.5])
    # Literal text "1/0/1900": the time of day is genuinely gone.
    ws.append([None, 232, "1/0/1900", 9.9])
    # An AM/PM text time.
    ws.append([None, 232, "9:23:00 AM", 1.1])
    path = tmp_path / "vendor.xlsx"
    wb.save(path)
    return str(path)


@pytest.fixture
def route_232_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3"],
            "stop_id": ["s1", "s1", "s1"],
            "dep_min": [443, 503, 563],  # 7:23, 8:23, 9:23
            "route_id": ["r232", "r232", "r232"],
        }
    )


@pytest.fixture
def route_232() -> pd.DataFrame:
    return pd.DataFrame({"route_id": ["r232"], "route_short_name": ["232"]})


def test_vendor_workbook_shapes_match_and_lost_times_drop(
    vendor_xlsx, route_232_events, route_232, vendor_spec
) -> None:
    _, trip_sums = load_ridership(
        vendor_xlsx, vendor_spec, events=route_232_events, routes=route_232
    )
    # Serial, datetime, and AM/PM rows all match; numeric 232 resolves to the
    # route despite Excel storing it as a float. The text "1/0/1900" row —
    # whose time of day is unrecoverable — is dropped, not guessed at.
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {
        "t1": 4.2,
        "t2": 7.5,
        "t3": 1.1,
    }


def test_shipped_defaults_load_a_vendor_workbook_with_no_overrides(
    vendor_xlsx, route_232_events, route_232
) -> None:
    # RidershipSpec() straight from the module constants, i.e. what an analyst
    # gets after setting only RIDERSHIP_CSV.
    _, trip_sums = load_ridership(
        vendor_xlsx, RidershipSpec(), events=route_232_events, routes=route_232
    )
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {
        "t1": 4.2,
        "t2": 7.5,
        "t3": 1.1,
    }


def test_a_file_where_nothing_matches_is_an_error_not_zero_riders(
    tmp_path, route_232_events, route_232, vendor_spec
) -> None:
    # Every start time has lost its time of day, so no row can match. Reporting
    # zero riders affected would look like a finding.
    frame = pd.DataFrame(
        {
            "ROUTE_NUMBER": ["232", "232"],
            "TRIP_START_TIME": ["1/0/1900", "1/0/1900"],
            "PASSENGERS_ON": [4.2, 7.5],
        }
    )
    path = _write(tmp_path, frame)
    with pytest.raises(ValueError, match="none of the 2 row"):
        load_ridership(path, vendor_spec, events=route_232_events, routes=route_232)


def test_partial_matches_do_not_error(tmp_path, route_232_events, route_232, vendor_spec) -> None:
    # Rows for other service days legitimately fail to match; only a total
    # wipeout is treated as an input failure.
    frame = pd.DataFrame(
        {
            "ROUTE_NUMBER": ["232", "232"],
            "TRIP_START_TIME": ["7:23", "1/0/1900"],
            "PASSENGERS_ON": [4.2, 7.5],
        }
    )
    path = _write(tmp_path, frame)
    _, trip_sums = load_ridership(path, vendor_spec, events=route_232_events, routes=route_232)
    assert trip_sums.set_index("trip_id")["boardings"].to_dict() == {"t1": 4.2}


def test_blank_survey_date_column_makes_date_averaging_a_no_op(
    vendor_xlsx, route_232_events, route_232, vendor_spec
) -> None:
    # A pre-averaged export ships SURVEY_DATE empty; pointing RIDERSHIP_DATE_COL
    # at it collapses to one date group and must not change any number.
    dated = vendor_spec._replace(date_col="SURVEY_DATE")
    _, plain = load_ridership(vendor_xlsx, vendor_spec, events=route_232_events, routes=route_232)
    _, averaged = load_ridership(vendor_xlsx, dated, events=route_232_events, routes=route_232)
    assert plain.equals(averaged)
