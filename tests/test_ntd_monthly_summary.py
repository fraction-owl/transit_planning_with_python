"""Tests for scripts/ridership_tools/ntd_monthly_summary.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.ridership_tools.ntd_monthly_summary as mod

FIXTURE_CSV = Path("tests/fixtures/ridership_by_route_and_stop.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_CSV)


@pytest.fixture
def minimal_ntd_df() -> pd.DataFrame:
    """One month of NTD rows for two routes across three service periods."""
    return pd.DataFrame(
        {
            "ROUTE_NAME": ["101", "101", "101", "202", "202", "202"],
            "SERVICE_PERIOD": [
                "Weekday",
                "Saturday",
                "Sunday",
                "Weekday",
                "Saturday",
                "Sunday",
            ],
            "MTH_BOARD": [6278.0, 575.0, 556.0, 3801.0, 348.0, 337.0],
            "MTH_REV_HOURS": [803.0, 109.5, 127.7, 682.0, 93.0, 108.5],
            "MTH_PASS_MILES": [24484.0, 2242.0, 2168.0, 15964.0, 1462.0, 1415.0],
            "ASCH_TRIPS": [72.0, 50.0, 43.0, 58.0, 41.0, 35.0],
            "DAYS": [22.0, 4.0, 5.0, 22.0, 4.0, 5.0],
            "REV_MILES": [520.0, 374.4, 338.0, 410.0, 295.2, 266.5],
            "service_type": ["local"] * 6,
            "period": ["Jul-2024"] * 6,
        }
    )


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------


def test_safe_float_plain_number() -> None:
    assert mod.safe_float("1234") == pytest.approx(1234.0)


def test_safe_float_comma_formatted() -> None:
    assert mod.safe_float("1,234.56") == pytest.approx(1234.56)


def test_safe_float_blank_string() -> None:
    assert mod.safe_float("") is None


def test_safe_float_nan() -> None:
    assert mod.safe_float(float("nan")) is None


def test_safe_float_non_numeric_string() -> None:
    assert mod.safe_float("N/A") is None


def test_safe_float_actual_float() -> None:
    assert mod.safe_float(42.5) == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# safe_div
# ---------------------------------------------------------------------------


def test_safe_div_normal_division() -> None:
    assert mod.safe_div(10, 4) == pytest.approx(2.5)


def test_safe_div_zero_denominator_returns_none() -> None:
    assert mod.safe_div(10, 0) is None


def test_safe_div_precision_param() -> None:
    assert mod.safe_div(1, 3, precision=2) == pytest.approx(0.33)


def test_safe_div_none_inputs_returns_none() -> None:
    assert mod.safe_div(None, 5) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalise_columns
# ---------------------------------------------------------------------------


def test_normalise_columns_uppercases_and_underscores() -> None:
    df = pd.DataFrame({"Route Name": [1], "service period": [2]})
    result = mod.normalise_columns(df)
    assert list(result.columns) == ["ROUTE_NAME", "SERVICE_PERIOD"]


def test_normalise_columns_strips_leading_trailing_whitespace() -> None:
    df = pd.DataFrame({"  mth board  ": [5]})
    result = mod.normalise_columns(df)
    assert "MTH_BOARD" in result.columns


def test_normalise_columns_does_not_mutate_original() -> None:
    df = pd.DataFrame({"route name": [1]})
    mod.normalise_columns(df)
    assert "route name" in df.columns


def test_normalise_columns_on_fixture(fixture_df: pd.DataFrame) -> None:
    """Fixture column names already uppercase; renaming one exercises the logic."""
    renamed = fixture_df.rename(columns={"TIME_PERIOD": "time period", "ROUTE_NAME": "route name"})
    result = mod.normalise_columns(renamed)
    assert "TIME_PERIOD" in result.columns
    assert "ROUTE_NAME" in result.columns


# ---------------------------------------------------------------------------
# classify_route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route,expected",
    [
        ("101", "local"),
        ("202", "local"),
        ("303", "express"),
        ("404", "express"),
        ("505", "circulator"),
        ("606", "circulator"),
        ("707", "feeder"),
        ("808", "feeder"),
        ("999", "unknown"),
    ],
)
def test_classify_route(route: str, expected: str) -> None:
    assert mod.classify_route(route) == expected


def test_classify_route_fixture_routes_all_unknown(fixture_df: pd.DataFrame) -> None:
    """No route in the stop-level fixture is in SERVICE_TYPE_DICT."""
    for route in fixture_df["ROUTE_NAME"].unique():
        assert mod.classify_route(str(route)) == "unknown"


# ---------------------------------------------------------------------------
# classify_corridor
# ---------------------------------------------------------------------------


def test_classify_corridor_known_route() -> None:
    result = mod.classify_corridor("101")
    assert "route_one_corridor" in result


def test_classify_corridor_multi_match() -> None:
    # Route 303 belongs to i_2_corridor
    result = mod.classify_corridor("303")
    assert "i_2_corridor" in result


def test_classify_corridor_unknown_route_returns_other() -> None:
    assert mod.classify_corridor("999") == ["other"]


def test_classify_corridor_fixture_routes_all_other(fixture_df: pd.DataFrame) -> None:
    for route in fixture_df["ROUTE_NAME"].unique():
        assert mod.classify_corridor(str(route)) == ["other"]


# ---------------------------------------------------------------------------
# _is_excluded
# ---------------------------------------------------------------------------


def test_is_excluded_no_exclusions() -> None:
    with patch.object(mod, "EXCLUDE_DATA", {}):
        assert mod._is_excluded("Jul-2024", "101") is False


def test_is_excluded_wildcard_excludes_any_route() -> None:
    with patch.object(mod, "EXCLUDE_DATA", {"Jul-2024": "*"}):
        assert mod._is_excluded("Jul-2024", "101") is True


def test_is_excluded_specific_route_in_list() -> None:
    with patch.object(mod, "EXCLUDE_DATA", {"Jul-2024": ["101", "202"]}):
        assert mod._is_excluded("Jul-2024", "101") is True
        assert mod._is_excluded("Jul-2024", "303") is False


def test_is_excluded_different_period_not_excluded() -> None:
    with patch.object(mod, "EXCLUDE_DATA", {"Jul-2024": "*"}):
        assert mod._is_excluded("Aug-2024", "101") is False


# ---------------------------------------------------------------------------
# slice_for_window
# ---------------------------------------------------------------------------


def test_slice_for_window_returns_rows_inside_window() -> None:
    df = pd.DataFrame({"period": ["Jun-2024", "Jul-2024", "Aug-2024", "Jul-2025"]})
    window = mod.TimeWindow("FY25", datetime(2024, 7, 1), datetime(2025, 6, 30))
    result = mod.slice_for_window(df, window)
    assert set(result["period"]) == {"Jul-2024", "Aug-2024"}


def test_slice_for_window_inclusive_boundaries() -> None:
    df = pd.DataFrame({"period": ["Jul-2024", "Jun-2025"]})
    window = mod.TimeWindow("FY25", datetime(2024, 7, 1), datetime(2025, 6, 30))
    result = mod.slice_for_window(df, window)
    assert set(result["period"]) == {"Jul-2024", "Jun-2025"}


def test_slice_for_window_empty_df_returns_empty() -> None:
    df = pd.DataFrame({"period": pd.Series([], dtype=str)})
    window = mod.TimeWindow("FY25", datetime(2024, 7, 1), datetime(2025, 6, 30))
    result = mod.slice_for_window(df, window)
    assert result.empty


def test_slice_for_window_no_match_returns_empty() -> None:
    df = pd.DataFrame({"period": ["Jan-2020", "Feb-2020"]})
    window = mod.TimeWindow("FY25", datetime(2024, 7, 1), datetime(2025, 6, 30))
    result = mod.slice_for_window(df, window)
    assert result.empty


# ---------------------------------------------------------------------------
# calculate_derived_columns
# ---------------------------------------------------------------------------


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ASCH_TRIPS": [72.0],
            "DAYS": [22.0],
            "MTH_BOARD": [6278.0],
            "MTH_REV_HOURS": [803.0],
            "REV_MILES": [520.0],
        }
    )


def test_calculate_derived_columns_total_trips(single_row_df: pd.DataFrame) -> None:
    result = mod.calculate_derived_columns(single_row_df)
    assert result["TOTAL_TRIPS"].iloc[0] == pytest.approx(72 * 22, rel=1e-3)


def test_calculate_derived_columns_boards_per_hour(single_row_df: pd.DataFrame) -> None:
    result = mod.calculate_derived_columns(single_row_df)
    assert result["BOARDS_PER_HOUR"].iloc[0] == pytest.approx(round(6278 / 803, 1))


def test_calculate_derived_columns_rev_miles(single_row_df: pd.DataFrame) -> None:
    result = mod.calculate_derived_columns(single_row_df)
    assert result["MTH_REV_MILES"].iloc[0] == pytest.approx(520 * 22, rel=1e-3)


def test_calculate_derived_columns_passengers_per_trip(single_row_df: pd.DataFrame) -> None:
    result = mod.calculate_derived_columns(single_row_df)
    expected_trips = 72 * 22
    assert result["PASSENGERS_PER_TRIP"].iloc[0] == pytest.approx(round(6278 / expected_trips, 1))


def test_calculate_derived_columns_does_not_mutate_original(
    single_row_df: pd.DataFrame,
) -> None:
    original_cols = set(single_row_df.columns)
    mod.calculate_derived_columns(single_row_df)
    assert set(single_row_df.columns) == original_cols


def test_calculate_derived_columns_zero_hours_gives_nan() -> None:
    df = pd.DataFrame(
        {
            "ASCH_TRIPS": [72.0],
            "DAYS": [22.0],
            "MTH_BOARD": [6278.0],
            "MTH_REV_HOURS": [0.0],
            "REV_MILES": [520.0],
        }
    )
    result = mod.calculate_derived_columns(df)
    assert pd.isna(result["BOARDS_PER_HOUR"].iloc[0])


# ---------------------------------------------------------------------------
# aggregate_by_service_type
# ---------------------------------------------------------------------------


def _service_type_df(**kwargs: list[float]) -> pd.DataFrame:
    base = {
        "service_type": ["local", "express"],
        "MTH_BOARD": [1000.0, 500.0],
        "MTH_REV_HOURS": [100.0, 50.0],
        "MTH_PASS_MILES": [5000.0, 2500.0],
        "MTH_REV_MILES": [400.0, 200.0],
        "TOTAL_TRIPS": [200.0, 100.0],
    }
    base.update(kwargs)  # type: ignore[arg-type]
    return pd.DataFrame(base)


def test_aggregate_by_service_type_appends_total_row() -> None:
    result = mod.aggregate_by_service_type(_service_type_df())
    assert "TOTAL" in result["service_type"].to_numpy()


def test_aggregate_by_service_type_sums_board_correctly() -> None:
    result = mod.aggregate_by_service_type(_service_type_df())
    total_row = result[result["service_type"] == "TOTAL"]
    assert total_row["MTH_BOARD"].iloc[0] == pytest.approx(1500.0)


def test_aggregate_by_service_type_boards_per_hour_total() -> None:
    df = pd.DataFrame(
        {
            "service_type": ["local"],
            "MTH_BOARD": [1000.0],
            "MTH_REV_HOURS": [100.0],
            "MTH_PASS_MILES": [5000.0],
            "MTH_REV_MILES": [400.0],
            "TOTAL_TRIPS": [200.0],
        }
    )
    result = mod.aggregate_by_service_type(df)
    total_row = result[result["service_type"] == "TOTAL"]
    assert total_row["BOARDS_PER_HOUR"].iloc[0] == pytest.approx(10.0)


def test_aggregate_by_service_type_row_count() -> None:
    result = mod.aggregate_by_service_type(_service_type_df())
    # 2 service types + TOTAL row
    assert len(result) == 3


# ---------------------------------------------------------------------------
# route_level_summary
# ---------------------------------------------------------------------------


def test_route_level_summary_aggregates_by_route(minimal_ntd_df: pd.DataFrame) -> None:
    minimal_ntd_df = minimal_ntd_df.copy()
    minimal_ntd_df["TOTAL_TRIPS"] = minimal_ntd_df["ASCH_TRIPS"] * minimal_ntd_df["DAYS"]
    minimal_ntd_df["MTH_REV_MILES"] = minimal_ntd_df["REV_MILES"] * minimal_ntd_df["DAYS"]
    result = mod.route_level_summary(minimal_ntd_df)
    assert set(result["ROUTE_NAME"]) == {"101", "202"}


def test_route_level_summary_sums_boards(minimal_ntd_df: pd.DataFrame) -> None:
    minimal_ntd_df = minimal_ntd_df.copy()
    minimal_ntd_df["TOTAL_TRIPS"] = minimal_ntd_df["ASCH_TRIPS"] * minimal_ntd_df["DAYS"]
    minimal_ntd_df["MTH_REV_MILES"] = minimal_ntd_df["REV_MILES"] * minimal_ntd_df["DAYS"]
    result = mod.route_level_summary(minimal_ntd_df)
    row_101 = result[result["ROUTE_NAME"] == "101"]
    assert row_101["MTH_BOARD"].iloc[0] == pytest.approx(6278 + 575 + 556)


def test_route_level_summary_daily_avg() -> None:
    df = pd.DataFrame(
        {
            "service_type": ["local"],
            "ROUTE_NAME": ["101"],
            "MTH_BOARD": [6600.0],
            "DAYS": [22.0],
            "MTH_REV_HOURS": [800.0],
            "MTH_PASS_MILES": [24000.0],
            "MTH_REV_MILES": [11000.0],
            "TOTAL_TRIPS": [1584.0],
        }
    )
    result = mod.route_level_summary(df)
    assert result["DAILY_AVG"].iloc[0] == pytest.approx(round(6600 / 22, 1))


def test_route_level_summary_sorted_by_route_name() -> None:
    df = pd.DataFrame(
        {
            "service_type": ["local", "local"],
            "ROUTE_NAME": ["202", "101"],
            "MTH_BOARD": [3801.0, 6278.0],
            "DAYS": [22.0, 22.0],
            "MTH_REV_HOURS": [682.0, 803.0],
            "MTH_PASS_MILES": [15964.0, 24484.0],
            "MTH_REV_MILES": [9020.0, 11440.0],
            "TOTAL_TRIPS": [1276.0, 1584.0],
        }
    )
    result = mod.route_level_summary(df)
    assert list(result["ROUTE_NAME"]) == ["101", "202"]


def test_route_level_summary_with_fixture_routes(fixture_df: pd.DataFrame) -> None:
    """Fixture routes are all unknown service type; summary should produce one row per route."""
    ntd = fixture_df.rename(columns={"BOARD_ALL": "MTH_BOARD", "TIME_PERIOD": "SERVICE_PERIOD"})
    ntd = ntd.assign(
        MTH_REV_HOURS=100.0,
        MTH_PASS_MILES=5000.0,
        DAYS=22.0,
        REV_MILES=500.0,
        service_type="unknown",
    )
    ntd["TOTAL_TRIPS"] = 50.0 * ntd["DAYS"]
    ntd["MTH_REV_MILES"] = ntd["REV_MILES"] * ntd["DAYS"]

    result = mod.route_level_summary(ntd)
    fixture_routes = set(fixture_df["ROUTE_NAME"].astype(str).unique())
    assert set(result["ROUTE_NAME"]) == fixture_routes


# ---------------------------------------------------------------------------
# build_monthly_timeseries
# ---------------------------------------------------------------------------


def _timeseries_input(**overrides: object) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "period": ["Jul-2024", "Jul-2024"],
        "ROUTE_NAME": ["101", "202"],
        "SERVICE_PERIOD": ["Weekday", "Weekday"],
        "MTH_BOARD": [1000.0, 2000.0],
        "DAYS": [22.0, 22.0],
        "MTH_REV_HOURS": [100.0, 200.0],
        "TOTAL_TRIPS": [200.0, 400.0],
        "MTH_REV_MILES": [400.0, 800.0],
    }
    base.update(overrides)  # type: ignore[arg-type]  # ty: ignore[no-matching-overload]
    return pd.DataFrame(base)


def test_build_monthly_timeseries_has_systemwide_row() -> None:
    with patch.object(mod, "ORDERED_PERIODS", ["Jul-2024"]):
        result = mod.build_monthly_timeseries(_timeseries_input())
    assert "SYSTEMWIDE" in result["route"].to_numpy()


def test_build_monthly_timeseries_per_route_rows_exist() -> None:
    with patch.object(mod, "ORDERED_PERIODS", ["Jul-2024"]):
        result = mod.build_monthly_timeseries(_timeseries_input())
    route_rows = result[result["route"] != "SYSTEMWIDE"]
    assert set(route_rows["route"]) == {"101", "202"}


def test_build_monthly_timeseries_weekday_avg() -> None:
    df = pd.DataFrame(
        {
            "period": ["Jul-2024"],
            "ROUTE_NAME": ["101"],
            "SERVICE_PERIOD": ["Weekday"],
            "MTH_BOARD": [6600.0],
            "DAYS": [22.0],
            "MTH_REV_HOURS": [800.0],
            "TOTAL_TRIPS": [1584.0],
            "MTH_REV_MILES": [11440.0],
        }
    )
    with patch.object(mod, "ORDERED_PERIODS", ["Jul-2024"]):
        result = mod.build_monthly_timeseries(df)
    row = result[result["route"] == "101"]
    assert row["weekday_avg"].iloc[0] == pytest.approx(round(6600 / 22, 1))


def test_build_monthly_timeseries_pph() -> None:
    df = pd.DataFrame(
        {
            "period": ["Jul-2024"],
            "ROUTE_NAME": ["101"],
            "SERVICE_PERIOD": ["Weekday"],
            "MTH_BOARD": [1000.0],
            "DAYS": [22.0],
            "MTH_REV_HOURS": [100.0],
            "TOTAL_TRIPS": [200.0],
            "MTH_REV_MILES": [400.0],
        }
    )
    with patch.object(mod, "ORDERED_PERIODS", ["Jul-2024"]):
        result = mod.build_monthly_timeseries(df)
    row = result[result["route"] == "101"]
    assert row["pph"].iloc[0] == pytest.approx(round(1000 / 100, 1))


# ---------------------------------------------------------------------------
# weekday_holiday_counts
# ---------------------------------------------------------------------------


def test_weekday_holiday_counts_groups_by_month() -> None:
    # Jul 4 2024 (Thu) and Sep 2 2024 (Mon) are both weekdays.
    holidays = [datetime(2024, 7, 4), datetime(2024, 9, 2)]
    assert mod.weekday_holiday_counts(holidays) == {"Jul-2024": 1, "Sep-2024": 1}


def test_weekday_holiday_counts_ignores_weekend_holidays() -> None:
    # Jul 6 2024 is a Saturday, Jul 7 2024 is a Sunday — both ignored.
    holidays = [datetime(2024, 7, 6), datetime(2024, 7, 7)]
    assert mod.weekday_holiday_counts(holidays) == {}


def test_weekday_holiday_counts_accumulates_same_month() -> None:
    holidays = [datetime(2024, 11, 28), datetime(2024, 11, 29)]  # Thu + Fri
    assert mod.weekday_holiday_counts(holidays) == {"Nov-2024": 2}


def test_weekday_holiday_counts_empty() -> None:
    assert mod.weekday_holiday_counts([]) == {}


# ---------------------------------------------------------------------------
# summarize_service_days
# ---------------------------------------------------------------------------


def test_summarize_service_days_representative_days(minimal_ntd_df: pd.DataFrame) -> None:
    with patch.object(mod, "ORDERED_PERIODS", ["Jul-2024"]):
        result = mod.summarize_service_days(minimal_ntd_df, {"Jul-2024": 1})
    row = result[result["period"] == "Jul-2024"].iloc[0]
    assert row["Weekday"] == 22
    assert row["Saturday"] == 4
    assert row["Sunday"] == 5
    assert row["Holidays"] == 1


def test_summarize_service_days_no_holidays_defaults_zero(minimal_ntd_df: pd.DataFrame) -> None:
    with patch.object(mod, "ORDERED_PERIODS", ["Jul-2024"]):
        result = mod.summarize_service_days(minimal_ntd_df)
    assert result["Holidays"].iloc[0] == 0


def test_summarize_service_days_columns() -> None:
    with patch.object(mod, "ORDERED_PERIODS", ["Jul-2024"]):
        result = mod.summarize_service_days(
            pd.DataFrame(columns=["period", "SERVICE_PERIOD", "DAYS"])
        )
    assert list(result.columns) == ["period", "Weekday", "Saturday", "Sunday", "Holidays"]


# ---------------------------------------------------------------------------
# with_weekday_ex_holiday_columns
# ---------------------------------------------------------------------------


def _weekday_routes_df() -> pd.DataFrame:
    """Two weekday months for one route: 22 + 20 = 42 weekday DAYS, 8800 boards."""
    return pd.DataFrame(
        {
            "service_type": ["local", "local"],
            "ROUTE_NAME": ["101", "101"],
            "period": ["Jul-2024", "Aug-2024"],
            "SERVICE_PERIOD": ["Weekday", "Weekday"],
            "MTH_BOARD": [4400.0, 4400.0],
            "DAYS": [22.0, 20.0],
            "MTH_REV_HOURS": [400.0, 360.0],
            "MTH_PASS_MILES": [10000.0, 9000.0],
            "REV_MILES": [500.0, 500.0],
            "MTH_REV_MILES": [11000.0, 10000.0],
            "TOTAL_TRIPS": [1000.0, 900.0],
        }
    )


def test_with_weekday_ex_holiday_columns_adds_adjusted_columns() -> None:
    weekday_df = _weekday_routes_df()
    summary = mod.route_level_summary(weekday_df)
    # 3 weekday holidays across the two months → denominator 42 - 3 = 39.
    result = mod.with_weekday_ex_holiday_columns(
        summary, weekday_df, {"Jul-2024": 2, "Aug-2024": 1}
    )
    row = result[result["ROUTE_NAME"] == "101"].iloc[0]
    assert row["DAYS_EX_HOLIDAYS"] == pytest.approx(39.0)
    assert row["DAILY_AVG_EX_HOLIDAYS"] == pytest.approx(round(8800 / 39, 1))
    # The raw DAILY_AVG column is left untouched.
    assert row["DAILY_AVG"] == pytest.approx(round(8800 / 42, 1))


def test_with_weekday_ex_holiday_columns_unchanged_when_no_holiday_in_range() -> None:
    weekday_df = _weekday_routes_df()
    summary = mod.route_level_summary(weekday_df)
    result = mod.with_weekday_ex_holiday_columns(summary, weekday_df, {"Dec-2024": 2})
    assert "DAYS_EX_HOLIDAYS" not in result.columns
    assert "DAILY_AVG_EX_HOLIDAYS" not in result.columns


def test_with_weekday_ex_holiday_columns_unchanged_when_no_holidays() -> None:
    weekday_df = _weekday_routes_df()
    summary = mod.route_level_summary(weekday_df)
    result = mod.with_weekday_ex_holiday_columns(summary, weekday_df, {})
    assert "DAYS_EX_HOLIDAYS" not in result.columns


def test_with_weekday_ex_holiday_columns_empty_input() -> None:
    empty = _weekday_routes_df().iloc[0:0]
    summary = mod.route_level_summary(empty)
    result = mod.with_weekday_ex_holiday_columns(summary, empty, {"Jul-2024": 2})
    assert "DAYS_EX_HOLIDAYS" not in result.columns


# ---------------------------------------------------------------------------
# detect_negative_trends_12m
# ---------------------------------------------------------------------------

_TREND_PERIODS = [f"M{i:02d}-2024" for i in range(1, 15)]  # 14 fake periods


def _trend_df(vals: list[float], route: str = "101") -> pd.DataFrame:
    """Build a minimal time-series DataFrame aligned to _TREND_PERIODS."""
    assert len(vals) == len(_TREND_PERIODS)
    return pd.DataFrame(
        {
            "period": _TREND_PERIODS,
            "route": [route] * len(_TREND_PERIODS),
            "weekday_avg": vals,
            "pph": [10.0] * len(_TREND_PERIODS),
            "ppt": [5.0] * len(_TREND_PERIODS),
            "ppm": [2.0] * len(_TREND_PERIODS),
        }
    )


def test_detect_negative_trends_flags_large_decline() -> None:
    # Baseline months 1-12 = 100, month 13 (prev) = 80, month 14 (latest) = 70
    vals = [100.0] * 12 + [80.0, 70.0]
    df = _trend_df(vals)
    with (
        patch.object(mod, "ORDERED_PERIODS", _TREND_PERIODS),
        patch.object(mod, "EXCLUDE_DATA", {}),
    ):
        result = mod.detect_negative_trends_12m(
            df, window=12, pct_threshold=10.0, min_coverage=0.75, confirm_prev=True
        )
    flagged = result[result["route"] == "101"]
    assert not flagged.empty
    assert "weekday_avg" in flagged["metric"].to_numpy()


def test_detect_negative_trends_no_flag_below_threshold() -> None:
    # Only 5% decline — below 10% threshold
    vals = [100.0] * 13 + [95.0]
    df = _trend_df(vals)
    with (
        patch.object(mod, "ORDERED_PERIODS", _TREND_PERIODS),
        patch.object(mod, "EXCLUDE_DATA", {}),
    ):
        result = mod.detect_negative_trends_12m(
            df, window=12, pct_threshold=10.0, min_coverage=0.75, confirm_prev=False
        )
    assert result.empty or "101" not in result["route"].to_numpy()


def test_detect_negative_trends_confirm_prev_blocks_single_month_drop() -> None:
    # Latest is far below baseline, but previous month is NOT below baseline
    vals = [100.0] * 13 + [70.0]
    df = _trend_df(vals)
    with (
        patch.object(mod, "ORDERED_PERIODS", _TREND_PERIODS),
        patch.object(mod, "EXCLUDE_DATA", {}),
    ):
        result = mod.detect_negative_trends_12m(
            df, window=12, pct_threshold=10.0, min_coverage=0.75, confirm_prev=True
        )
    # confirm_prev=True: month 13 == 100 == baseline, so prev is not below — no flag
    assert result.empty or "101" not in result["route"].to_numpy()


def test_detect_negative_trends_returns_empty_dataframe_on_no_flags() -> None:
    vals = [100.0] * 14
    df = _trend_df(vals)
    with (
        patch.object(mod, "ORDERED_PERIODS", _TREND_PERIODS),
        patch.object(mod, "EXCLUDE_DATA", {}),
    ):
        result = mod.detect_negative_trends_12m(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# write_trend_log
# ---------------------------------------------------------------------------


def test_write_trend_log_empty_flags_creates_file(tmp_path: Path) -> None:
    out = mod.write_trend_log(pd.DataFrame(), tmp_path)
    assert out.exists()
    assert "No negative trends" in out.read_text()


def test_write_trend_log_with_flags_mentions_route(tmp_path: Path) -> None:
    flags = pd.DataFrame(
        {
            "route": ["101"],
            "metric": ["weekday_avg"],
            "latest_value": [75.0],
            "baseline_mean": [100.0],
            "pct_change": [-25.0],
            "window_months": [12],
        }
    )
    out = mod.write_trend_log(flags, tmp_path)
    content = out.read_text()
    assert "101" in content
    assert "weekday_avg" in content
    assert "25.0%" in content


def test_write_trend_log_returns_path(tmp_path: Path) -> None:
    out = mod.write_trend_log(pd.DataFrame(), tmp_path)
    assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# extract_config_block
# ---------------------------------------------------------------------------


def test_extract_config_block_reads_from_script() -> None:
    source = Path("scripts/ridership_tools/ntd_monthly_summary.py")
    block = mod.extract_config_block(source)
    assert "DATA_ROOT" in block
    assert "OUTPUT_DIR" in block


def test_extract_config_block_excludes_marker_lines() -> None:
    source = Path("scripts/ridership_tools/ntd_monthly_summary.py")
    block = mod.extract_config_block(source)
    assert mod.CONFIG_BEGIN_MARKER not in block
    assert mod.CONFIG_END_MARKER not in block


def test_extract_config_block_raises_on_missing_markers(tmp_path: Path) -> None:
    f = tmp_path / "no_markers.py"
    f.write_text("# just a comment\n")
    with pytest.raises(ValueError, match="Config markers not found"):
        mod.extract_config_block(f)


def test_extract_config_block_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        mod.extract_config_block(tmp_path / "does_not_exist.py")


# ---------------------------------------------------------------------------
# write_run_log
# ---------------------------------------------------------------------------


def test_write_run_log_returns_true_on_success(tmp_path: Path) -> None:
    assert mod.write_run_log(tmp_path) is True


def test_write_run_log_creates_log_file(tmp_path: Path) -> None:
    mod.write_run_log(tmp_path)
    log_path = tmp_path / "ntd_monthly_summary_runlog.txt"
    assert log_path.exists()


def test_write_run_log_content_includes_header(tmp_path: Path) -> None:
    mod.write_run_log(tmp_path)
    content = (tmp_path / "ntd_monthly_summary_runlog.txt").read_text()
    assert "NTD MONTHLY SUMMARY RUN LOG" in content


def test_write_run_log_content_includes_config(tmp_path: Path) -> None:
    mod.write_run_log(tmp_path)
    content = (tmp_path / "ntd_monthly_summary_runlog.txt").read_text()
    assert "DATA_ROOT" in content


def test_write_run_log_omits_service_day_section_when_none(tmp_path: Path) -> None:
    mod.write_run_log(tmp_path)
    content = (tmp_path / "ntd_monthly_summary_runlog.txt").read_text()
    assert "SERVICE-DAY COUNTS PER MONTH" not in content


def test_write_run_log_includes_service_day_table(tmp_path: Path) -> None:
    service_days = pd.DataFrame(
        {
            "period": ["Jul-2024"],
            "Weekday": [22],
            "Saturday": [4],
            "Sunday": [5],
            "Holidays": [1],
        }
    )
    mod.write_run_log(tmp_path, service_days)
    content = (tmp_path / "ntd_monthly_summary_runlog.txt").read_text()
    assert "SERVICE-DAY COUNTS PER MONTH" in content
    assert "Holidays" in content
    assert "Jul-2024" in content
