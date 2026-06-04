from __future__ import annotations

import pandas as pd
import pytest

from scripts.stop_analysis.stop_impacts_target_routes import (
    _apply_service_id_filter_to_trips,
    _dow_code_from_calendar_row,
    _svc_ids_to_dow_list,
    add_stop_level_summary_columns,
    build_stop_service_routes,
    classify_impacts,
    identify_target_route_ids,
)

# ---------------------------------------------------------------------------
# _dow_code_from_calendar_row
# ---------------------------------------------------------------------------


def test_dow_code_weekday_only() -> None:
    row = pd.Series(
        {
            "monday": "1",
            "tuesday": "1",
            "wednesday": "1",
            "thursday": "1",
            "friday": "1",
            "saturday": "0",
            "sunday": "0",
        }
    )
    assert _dow_code_from_calendar_row(row) == "M/T/W/R/F"


def test_dow_code_saturday_only() -> None:
    row = pd.Series(
        {
            "monday": "0",
            "tuesday": "0",
            "wednesday": "0",
            "thursday": "0",
            "friday": "0",
            "saturday": "1",
            "sunday": "0",
        }
    )
    assert _dow_code_from_calendar_row(row) == "S"


def test_dow_code_sunday_only() -> None:
    row = pd.Series(
        {
            "monday": "0",
            "tuesday": "0",
            "wednesday": "0",
            "thursday": "0",
            "friday": "0",
            "saturday": "0",
            "sunday": "1",
        }
    )
    assert _dow_code_from_calendar_row(row) == "U"


def test_dow_code_no_active_days_returns_empty() -> None:
    row = pd.Series(
        {
            "monday": "0",
            "tuesday": "0",
            "wednesday": "0",
            "thursday": "0",
            "friday": "0",
            "saturday": "0",
            "sunday": "0",
        }
    )
    assert _dow_code_from_calendar_row(row) == ""


def test_dow_code_missing_column_treated_as_zero() -> None:
    # Row with no 'saturday' key: defaults to "0"
    row = pd.Series(
        {
            "monday": "1",
            "tuesday": "0",
            "wednesday": "0",
            "thursday": "0",
            "friday": "0",
            "sunday": "0",
        }
    )
    code = _dow_code_from_calendar_row(row)
    assert "M" in code
    assert "S" not in code


# ---------------------------------------------------------------------------
# _svc_ids_to_dow_list
# ---------------------------------------------------------------------------


def test_svc_ids_to_dow_list_maps_known_ids() -> None:
    svc_to_dow = {"1": "M/T/W/R/F", "2": "S", "3": "U"}
    result = _svc_ids_to_dow_list("1,2", svc_to_dow)
    assert "M/T/W/R/F" in result
    assert "S" in result


def test_svc_ids_to_dow_list_falls_back_to_service_id() -> None:
    result = _svc_ids_to_dow_list("UNKNOWN", svc_to_dow={})
    assert "UNKNOWN" in result


def test_svc_ids_to_dow_list_empty_input_returns_empty() -> None:
    assert _svc_ids_to_dow_list("", svc_to_dow={}) == ""


# ---------------------------------------------------------------------------
# _apply_service_id_filter_to_trips
# ---------------------------------------------------------------------------


@pytest.fixture()
def trips_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1", "R1", "R2"],
            "service_id": ["WD", "WE", "WD"],
        }
    )


def test_apply_service_id_filter_keeps_matching_trips(trips_df: pd.DataFrame) -> None:
    result = _apply_service_id_filter_to_trips(trips_df, service_filter={"WD"})
    assert set(result["trip_id"]) == {"T1", "T3"}


def test_apply_service_id_filter_none_returns_all(trips_df: pd.DataFrame) -> None:
    result = _apply_service_id_filter_to_trips(trips_df, service_filter=None)
    assert len(result) == len(trips_df)


def test_apply_service_id_filter_unknown_id_returns_empty(trips_df: pd.DataFrame) -> None:
    result = _apply_service_id_filter_to_trips(trips_df, service_filter={"HOLIDAY"})
    assert result.empty


# ---------------------------------------------------------------------------
# identify_target_route_ids
# ---------------------------------------------------------------------------


@pytest.fixture()
def routes_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["R1", "R2", "R3"],
            "route_short_name": ["101", "202", "303"],
            "route_long_name": ["Route 101", "Route 202", "Route 303"],
        }
    )


def test_identify_target_route_ids_by_short_name(routes_df: pd.DataFrame) -> None:
    result = identify_target_route_ids(routes_df, tokens={"101"})
    assert result == {"R1"}


def test_identify_target_route_ids_by_route_id(routes_df: pd.DataFrame) -> None:
    result = identify_target_route_ids(routes_df, tokens={"R2"})
    assert result == {"R2"}


def test_identify_target_route_ids_multiple_tokens(routes_df: pd.DataFrame) -> None:
    result = identify_target_route_ids(routes_df, tokens={"101", "202"})
    assert result == {"R1", "R2"}


def test_identify_target_route_ids_missing_token_returns_empty(routes_df: pd.DataFrame) -> None:
    result = identify_target_route_ids(routes_df, tokens={"999"})
    assert result == set()


# ---------------------------------------------------------------------------
# build_stop_service_routes + classify_impacts
# ---------------------------------------------------------------------------


@pytest.fixture()
def gtfs_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Minimal GTFS tables: stop S1 served by R1+R2, stop S2 by R1 only."""
    stop_times = pd.DataFrame({"trip_id": ["T1", "T1", "T2"], "stop_id": ["S1", "S2", "S1"]})
    trips = pd.DataFrame(
        {"trip_id": ["T1", "T2"], "route_id": ["R1", "R2"], "service_id": ["WD", "WD"]}
    )
    routes = pd.DataFrame(
        {
            "route_id": ["R1", "R2"],
            "route_short_name": ["101", "202"],
            "route_long_name": ["Route 101", "Route 202"],
        }
    )
    stops = pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["Stop One", "Stop Two"],
            "stop_code": ["100", "200"],
            "stop_lon": [-77.0, -77.1],
            "stop_lat": [38.7, 38.8],
            "location_type": ["0", "0"],
        }
    )
    return stop_times, trips, routes, stops


def test_build_stop_service_routes_returns_expected_columns(
    gtfs_tables: tuple,
) -> None:
    stop_times, trips, routes, _ = gtfs_tables
    result = build_stop_service_routes(stop_times, trips, routes)
    assert "stop_id" in result.columns
    assert "service_id" in result.columns
    assert "route_id_arr" in result.columns


def test_classify_impacts_target_only(gtfs_tables: tuple) -> None:
    stop_times, trips, routes, stops = gtfs_tables
    ssr = build_stop_service_routes(stop_times, trips, routes)
    result = classify_impacts(ssr, stops, target_route_ids={"R1"})
    # S2 is served only by R1 → target_only
    s2_class = result.loc[result.stop_id == "S2", "classification"]
    assert (s2_class == "target_only").all()


def test_classify_impacts_target_plus_other(gtfs_tables: tuple) -> None:
    stop_times, trips, routes, stops = gtfs_tables
    ssr = build_stop_service_routes(stop_times, trips, routes)
    result = classify_impacts(ssr, stops, target_route_ids={"R1"})
    # S1 is served by R1 and R2 → target_plus_other
    s1_class = result.loc[result.stop_id == "S1", "classification"]
    assert (s1_class == "target_plus_other").all()


def test_classify_impacts_includes_stop_attributes(gtfs_tables: tuple) -> None:
    stop_times, trips, routes, stops = gtfs_tables
    ssr = build_stop_service_routes(stop_times, trips, routes)
    result = classify_impacts(ssr, stops, target_route_ids={"R1"})
    assert "stop_name" in result.columns
    assert "stop_lat" in result.columns


# ---------------------------------------------------------------------------
# add_stop_level_summary_columns
# ---------------------------------------------------------------------------


def test_add_stop_level_summary_columns_adds_expected_columns(
    gtfs_tables: tuple,
) -> None:
    stop_times, trips, routes, stops = gtfs_tables
    ssr = build_stop_service_routes(stop_times, trips, routes)
    classified = classify_impacts(ssr, stops, target_route_ids={"R1"})
    flagged = classified[classified["classification"] != "not_target"].copy()
    result = add_stop_level_summary_columns(flagged, svc_to_dow={})
    for col in ("impact_category", "has_eliminated_days", "has_route_loss_days"):
        assert col in result.columns, f"Missing column: {col}"


def test_add_stop_level_summary_columns_eliminated_category(
    gtfs_tables: tuple,
) -> None:
    stop_times, trips, routes, stops = gtfs_tables
    ssr = build_stop_service_routes(stop_times, trips, routes)
    classified = classify_impacts(ssr, stops, target_route_ids={"R1"})
    # Keep only target_only rows (S2)
    only_s2 = classified[classified["stop_id"] == "S2"].copy()
    result = add_stop_level_summary_columns(only_s2, svc_to_dow={})
    assert (result["impact_category"] == "eliminated").all()
