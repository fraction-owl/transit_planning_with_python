from __future__ import annotations

import pandas as pd
import pytest

from scripts.gtfs_exports.stop_pattern_exporter import (
    assign_pattern_ids,
    convert_dist_to_miles,
    filter_trips,
    format_service_id_folder_name,
    forward_match_pattern_to_master,
    is_number,
    minutes_to_hhmm,
    parse_time_to_minutes,
)

# ---------------------------------------------------------------------------
# is_number
# ---------------------------------------------------------------------------


def test_is_number_integer_string() -> None:
    assert is_number("42") is True


def test_is_number_float_string() -> None:
    assert is_number("3.14") is True


def test_is_number_plain_int() -> None:
    assert is_number(7) is True


def test_is_number_text_returns_false() -> None:
    assert is_number("hello") is False


def test_is_number_none_returns_false() -> None:
    assert is_number(None) is False


def test_is_number_empty_returns_false() -> None:
    assert is_number("") is False


# ---------------------------------------------------------------------------
# convert_dist_to_miles
# ---------------------------------------------------------------------------


def test_convert_dist_to_miles_meters() -> None:
    import scripts.gtfs_exports.stop_pattern_exporter as mod

    orig = mod.CONVERT_TO_MILES
    mod.CONVERT_TO_MILES = True
    try:
        result = convert_dist_to_miles(1609.34, "meters")
        assert result == pytest.approx(1.0, rel=0.01)
    finally:
        mod.CONVERT_TO_MILES = orig


def test_convert_dist_to_miles_feet() -> None:
    import scripts.gtfs_exports.stop_pattern_exporter as mod

    orig = mod.CONVERT_TO_MILES
    mod.CONVERT_TO_MILES = True
    try:
        result = convert_dist_to_miles(5280.0, "feet")
        assert result == pytest.approx(1.0, rel=0.01)
    finally:
        mod.CONVERT_TO_MILES = orig


def test_convert_dist_to_miles_convert_false_returns_original() -> None:
    import scripts.gtfs_exports.stop_pattern_exporter as mod

    orig = mod.CONVERT_TO_MILES
    mod.CONVERT_TO_MILES = False
    try:
        assert convert_dist_to_miles(1000.0, "meters") == 1000.0
    finally:
        mod.CONVERT_TO_MILES = orig


# ---------------------------------------------------------------------------
# parse_time_to_minutes
# ---------------------------------------------------------------------------


def test_parse_time_to_minutes_basic() -> None:
    assert parse_time_to_minutes("07:05:30") == pytest.approx(425.5)


def test_parse_time_to_minutes_midnight() -> None:
    assert parse_time_to_minutes("00:00:00") == pytest.approx(0.0)


def test_parse_time_to_minutes_not_string_returns_none() -> None:
    assert parse_time_to_minutes(None) is None  # type: ignore[arg-type]


def test_parse_time_to_minutes_bad_format_returns_none() -> None:
    assert parse_time_to_minutes("07:05") is None


# ---------------------------------------------------------------------------
# minutes_to_hhmm
# ---------------------------------------------------------------------------


def test_minutes_to_hhmm_basic() -> None:
    assert minutes_to_hhmm(425) == "07:05"


def test_minutes_to_hhmm_none_returns_empty() -> None:
    assert minutes_to_hhmm(None) == ""


# ---------------------------------------------------------------------------
# format_service_id_folder_name
# ---------------------------------------------------------------------------


def test_format_service_id_folder_name_no_calendar() -> None:
    name = format_service_id_folder_name("5", None)
    assert name == "calendar_5"


def test_format_service_id_folder_name_with_weekday_calendar() -> None:
    cal = pd.DataFrame(
        {
            "service_id": ["3"],
            "monday": ["1"],
            "tuesday": ["1"],
            "wednesday": ["1"],
            "thursday": ["1"],
            "friday": ["1"],
            "saturday": ["0"],
            "sunday": ["0"],
        }
    )
    name = format_service_id_folder_name("3", cal)
    assert "mon" in name and "fri" in name


def test_format_service_id_folder_name_no_match_in_calendar() -> None:
    cal = pd.DataFrame(
        {
            "service_id": ["99"],
            "monday": ["1"],
            "tuesday": ["0"],
            "wednesday": ["0"],
            "thursday": ["0"],
            "friday": ["0"],
            "saturday": ["0"],
            "sunday": ["0"],
        }
    )
    name = format_service_id_folder_name("5", cal)
    assert name == "calendar_5"


# ---------------------------------------------------------------------------
# filter_trips
# ---------------------------------------------------------------------------


def _trips_and_routes() -> tuple[pd.DataFrame, pd.DataFrame]:
    trips = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1", "R1", "R2"],
            "service_id": ["1", "2", "1"],
        }
    )
    routes = pd.DataFrame({"route_id": ["R1", "R2"], "route_short_name": ["101", "202"]})
    return trips, routes


def test_filter_trips_no_filters_returns_all() -> None:
    import scripts.gtfs_exports.stop_pattern_exporter as mod

    orig_in = mod.FILTER_IN_ROUTE_SHORT_NAMES
    orig_out = mod.FILTER_OUT_ROUTE_SHORT_NAMES
    mod.FILTER_IN_ROUTE_SHORT_NAMES = []
    mod.FILTER_OUT_ROUTE_SHORT_NAMES = []
    trips, routes = _trips_and_routes()
    try:
        result = filter_trips(trips, routes, [])
        assert len(result) == 3
    finally:
        mod.FILTER_IN_ROUTE_SHORT_NAMES = orig_in
        mod.FILTER_OUT_ROUTE_SHORT_NAMES = orig_out


def test_filter_trips_by_service_id() -> None:
    import scripts.gtfs_exports.stop_pattern_exporter as mod

    orig_in = mod.FILTER_IN_ROUTE_SHORT_NAMES
    orig_out = mod.FILTER_OUT_ROUTE_SHORT_NAMES
    mod.FILTER_IN_ROUTE_SHORT_NAMES = []
    mod.FILTER_OUT_ROUTE_SHORT_NAMES = []
    trips, routes = _trips_and_routes()
    try:
        result = filter_trips(trips, routes, ["1"])
        assert all(result["service_id"] == "1")
    finally:
        mod.FILTER_IN_ROUTE_SHORT_NAMES = orig_in
        mod.FILTER_OUT_ROUTE_SHORT_NAMES = orig_out


# ---------------------------------------------------------------------------
# forward_match_pattern_to_master
# ---------------------------------------------------------------------------


def test_forward_match_full_match() -> None:
    pattern = [("S1", "-"), ("S2", "1.2")]
    master = [("S1", "Stop A"), ("S2", "Stop B")]
    result = forward_match_pattern_to_master(pattern, master)
    assert result == ["-", "1.2"]


def test_forward_match_partial_pattern() -> None:
    pattern = [("S2", "1.5")]
    master = [("S1", "Stop A"), ("S2", "Stop B")]
    result = forward_match_pattern_to_master(pattern, master)
    assert result[1] == "1.5"
    assert result[0] == ""


def test_forward_match_empty_pattern() -> None:
    master = [("S1", "Stop A"), ("S2", "Stop B")]
    result = forward_match_pattern_to_master([], master)
    assert result == ["", ""]


# ---------------------------------------------------------------------------
# assign_pattern_ids
# ---------------------------------------------------------------------------


def test_assign_pattern_ids_assigns_ids() -> None:
    patterns_dict = {
        ("R1", "0", "1", (("S1", "-"), ("S2", "1.0"))): {
            "route_id": "R1",
            "direction_id": "0",
            "service_id": "1",
            "pattern_stops": (("S1", "-"), ("S2", "1.0")),
            "trip_count": 3,
            "trip_ids": ["T1", "T2", "T3"],
        }
    }
    records = assign_pattern_ids(patterns_dict)
    assert len(records) == 1
    assert records[0]["pattern_id"] == 1


def test_assign_pattern_ids_trip_count_preserved() -> None:
    patterns_dict = {
        ("R1", "0", "1", (("S1", "-"),)): {
            "route_id": "R1",
            "direction_id": "0",
            "service_id": "1",
            "pattern_stops": (("S1", "-"),),
            "trip_count": 5,
            "trip_ids": ["T1", "T2", "T3", "T4", "T5"],
        }
    }
    records = assign_pattern_ids(patterns_dict)
    assert records[0]["trip_count"] == 5
