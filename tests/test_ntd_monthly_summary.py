"""Tests for scripts/ridership_tools/ntd_monthly_summary.py."""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.ridership_tools.ntd_monthly_summary as mod

FIXTURE_CSV = Path("tests/fixtures/ridership_by_route_and_stop.csv")
EXAMPLE_CONFIG = Path("scripts/ridership_tools/ntd_monthly_config_example.txt")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_CSV)


@pytest.fixture
def config() -> mod.AppConfig:
    """The shipped example configuration."""
    return mod.load_config(EXAMPLE_CONFIG)


def _with_periods(config: mod.AppConfig, periods: list[str], **overrides: object) -> mod.AppConfig:
    """Return *config* covering *periods* (in chronological order) plus *overrides*."""
    specs = {period: mod.PeriodSpec(f"{period}.xlsx", "Sheet1") for period in periods}
    return replace(config, periods=specs, **overrides)


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
def test_classify_route(route: str, expected: str, config: mod.AppConfig) -> None:
    assert mod.classify_route(route, config) == expected


def test_classify_route_fixture_routes_all_unknown(
    fixture_df: pd.DataFrame, config: mod.AppConfig
) -> None:
    """No route in the stop-level fixture is in [service_types]."""
    for route in fixture_df["ROUTE_NAME"].unique():
        assert mod.classify_route(str(route), config) == "unknown"


# ---------------------------------------------------------------------------
# classify_corridor
# ---------------------------------------------------------------------------


def test_classify_corridor_known_route(config: mod.AppConfig) -> None:
    result = mod.classify_corridor("101", config)
    assert "route_one_corridor" in result


def test_classify_corridor_multi_match(config: mod.AppConfig) -> None:
    corridors = {"i_2_corridor": ["303", "404"], "downtown": ["101", "303"]}
    result = mod.classify_corridor("303", replace(config, corridors=corridors))
    assert result == ["i_2_corridor", "downtown"]


def test_classify_corridor_unknown_route_returns_other(config: mod.AppConfig) -> None:
    assert mod.classify_corridor("999", config) == ["other"]


def test_classify_corridor_fixture_routes_all_other(
    fixture_df: pd.DataFrame, config: mod.AppConfig
) -> None:
    for route in fixture_df["ROUTE_NAME"].unique():
        assert mod.classify_corridor(str(route), config) == ["other"]


# ---------------------------------------------------------------------------
# _is_excluded
# ---------------------------------------------------------------------------


def test_is_excluded_no_exclusions(config: mod.AppConfig) -> None:
    assert mod._is_excluded("Jul-2024", "101", replace(config, exclude_data={})) is False


def test_is_excluded_wildcard_excludes_any_route(config: mod.AppConfig) -> None:
    cfg = replace(config, exclude_data={"Jul-2024": "*"})
    assert mod._is_excluded("Jul-2024", "101", cfg) is True


def test_is_excluded_specific_route_in_list(config: mod.AppConfig) -> None:
    cfg = replace(config, exclude_data={"Jul-2024": ["101", "202"]})
    assert mod._is_excluded("Jul-2024", "101", cfg) is True
    assert mod._is_excluded("Jul-2024", "303", cfg) is False


def test_is_excluded_different_period_not_excluded(config: mod.AppConfig) -> None:
    cfg = replace(config, exclude_data={"Jul-2024": "*"})
    assert mod._is_excluded("Aug-2024", "101", cfg) is False


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


def test_route_level_summary_keeps_garage_column_when_present(
    minimal_ntd_df: pd.DataFrame,
) -> None:
    df = minimal_ntd_df.assign(
        TOTAL_TRIPS=minimal_ntd_df["ASCH_TRIPS"] * minimal_ntd_df["DAYS"],
        MTH_REV_MILES=minimal_ntd_df["REV_MILES"] * minimal_ntd_df["DAYS"],
        garage=["north"] * 3 + ["unknown"] * 3,
    )
    result = mod.route_level_summary(df)
    assert dict(zip(result["ROUTE_NAME"], result["garage"], strict=True)) == {
        "101": "north",
        "202": "unknown",
    }


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


@pytest.fixture
def july_config(config: mod.AppConfig) -> mod.AppConfig:
    return _with_periods(config, ["Jul-2024"])


def test_build_monthly_timeseries_has_systemwide_row(july_config: mod.AppConfig) -> None:
    result = mod.build_monthly_timeseries(_timeseries_input(), july_config)
    assert "SYSTEMWIDE" in result["route"].to_numpy()


def test_build_monthly_timeseries_per_route_rows_exist(july_config: mod.AppConfig) -> None:
    result = mod.build_monthly_timeseries(_timeseries_input(), july_config)
    route_rows = result[result["route"] != "SYSTEMWIDE"]
    assert set(route_rows["route"]) == {"101", "202"}


def test_build_monthly_timeseries_weekday_avg(july_config: mod.AppConfig) -> None:
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
    result = mod.build_monthly_timeseries(df, july_config)
    row = result[result["route"] == "101"]
    assert row["weekday_avg"].iloc[0] == pytest.approx(round(6600 / 22, 1))


def test_build_monthly_timeseries_pph(july_config: mod.AppConfig) -> None:
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
    result = mod.build_monthly_timeseries(df, july_config)
    row = result[result["route"] == "101"]
    assert row["pph"].iloc[0] == pytest.approx(round(1000 / 100, 1))


def _systemwide_row(df: pd.DataFrame, config: mod.AppConfig) -> pd.Series:
    result = mod.build_monthly_timeseries(df, config)
    return result[result["route"] == "SYSTEMWIDE"].iloc[0]


def test_build_monthly_timeseries_systemwide_avgs_divide_by_month_service_days(
    july_config: mod.AppConfig,
) -> None:
    # Both routes run the same 20 weekdays, 4 Saturdays and 4 Sundays, so the
    # systemwide averages divide by those counts, not by the summed route-days.
    df = pd.DataFrame(
        {
            "period": ["Jul-2024"] * 6,
            "ROUTE_NAME": ["101", "202"] * 3,
            "SERVICE_PERIOD": ["Weekday"] * 2 + ["Saturday"] * 2 + ["Sunday"] * 2,
            "MTH_BOARD": [2000.0, 4000.0, 400.0, 800.0, 300.0, 500.0],
            "DAYS": [20.0, 20.0, 4.0, 4.0, 4.0, 4.0],
            "MTH_REV_HOURS": [100.0] * 6,
            "TOTAL_TRIPS": [200.0] * 6,
            "MTH_REV_MILES": [400.0] * 6,
        }
    )
    row = _systemwide_row(df, july_config)
    assert row["weekday_avg"] == pytest.approx(6000 / 20)
    assert row["saturday_avg"] == pytest.approx(1200 / 4)
    assert row["sunday_avg"] == pytest.approx(800 / 4)


def test_build_monthly_timeseries_systemwide_avg_with_partial_month_route(
    july_config: mod.AppConfig,
) -> None:
    # Route 303 ran 10 of the month's 20 weekdays: its boardings count toward the
    # 20-day month instead of adding 10 route-days to the denominator.
    df = pd.DataFrame(
        {
            "period": ["Jul-2024"] * 3,
            "ROUTE_NAME": ["101", "202", "303"],
            "SERVICE_PERIOD": ["Weekday"] * 3,
            "MTH_BOARD": [4000.0, 2000.0, 1000.0],
            "DAYS": [20.0, 20.0, 10.0],
            "MTH_REV_HOURS": [100.0] * 3,
            "TOTAL_TRIPS": [200.0] * 3,
            "MTH_REV_MILES": [400.0] * 3,
        }
    )
    assert _systemwide_row(df, july_config)["weekday_avg"] == pytest.approx(7000 / 20)


# ---------------------------------------------------------------------------
# plot_metric_over_time
# ---------------------------------------------------------------------------


def test_plot_metric_over_time_labels_months_without_category_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, config: mod.AppConfig
) -> None:
    periods = ["Jul-2024", "Aug-2024", "Sep-2024"]
    cfg = _with_periods(config, periods, output_dir=tmp_path)
    # The route has no Jul-2024 row, so its line starts at Aug-2024.
    df_time = pd.DataFrame(
        {"period": ["Aug-2024", "Sep-2024"], "route": ["101"] * 2, "weekday_avg": [100.0, 120.0]}
    )
    caplog.set_level(logging.INFO, logger="matplotlib.category")
    with patch.object(mod.plt, "close"):  # keep the figure open for inspection
        mod.plot_metric_over_time(df_time, "weekday_avg", cfg)
    ax = mod.plt.gca()
    labels = [label.get_text() for label in ax.get_xticklabels()]
    x_min = ax.get_xlim()[0]
    mod.plt.close("all")

    assert labels == periods
    # The x-axis still covers only the months with data (Aug at position 1).
    assert x_min > 0
    assert not [r for r in caplog.records if r.name == "matplotlib.category"]
    assert (tmp_path / "plots" / "weekday_avg" / "weekday_avg_route_101.png").exists()


def test_plot_metric_over_time_keeps_missing_catalogue_month_as_gap(
    tmp_path: Path, config: mod.AppConfig
) -> None:
    cfg = _with_periods(config, ["Jul-2024", "Sep-2024"], output_dir=tmp_path)
    df_time = pd.DataFrame(
        {"period": ["Jul-2024", "Sep-2024"], "route": ["101"] * 2, "weekday_avg": [100.0, 120.0]}
    )
    with patch.object(mod.plt, "close"):
        mod.plot_metric_over_time(df_time, "weekday_avg", cfg)
    ax = mod.plt.gca()
    labels = [label.get_text() for label in ax.get_xticklabels()]
    x_data = list(ax.get_lines()[0].get_xdata())
    mod.plt.close("all")

    assert labels == ["Jul-2024", "Aug-2024", "Sep-2024"]
    assert x_data == [0, 1, 2]


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


def test_summarize_service_days_representative_days(
    minimal_ntd_df: pd.DataFrame, july_config: mod.AppConfig
) -> None:
    result = mod.summarize_service_days(minimal_ntd_df, july_config, {"Jul-2024": 1})
    row = result[result["period"] == "Jul-2024"].iloc[0]
    assert row["Weekday"] == 22
    assert row["Saturday"] == 4
    assert row["Sunday"] == 5
    assert row["Holidays"] == 1


def test_summarize_service_days_no_holidays_defaults_zero(
    minimal_ntd_df: pd.DataFrame, july_config: mod.AppConfig
) -> None:
    result = mod.summarize_service_days(minimal_ntd_df, july_config)
    assert result["Holidays"].iloc[0] == 0


@pytest.mark.parametrize("days", [[10.0, 20.0], [20.0, 10.0]])
def test_summarize_service_days_tie_goes_to_larger_count(
    days: list[float], july_config: mod.AppConfig
) -> None:
    df = pd.DataFrame({"period": ["Jul-2024"] * 2, "SERVICE_PERIOD": ["Weekday"] * 2, "DAYS": days})
    result = mod.summarize_service_days(df, july_config)
    assert result["Weekday"].iloc[0] == 20


def test_summarize_service_days_columns(july_config: mod.AppConfig) -> None:
    result = mod.summarize_service_days(
        pd.DataFrame(columns=["period", "SERVICE_PERIOD", "DAYS"]), july_config
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

# 14 consecutive months: Jan-2024 .. Feb-2025.
_TREND_PERIODS = [f"{mod.MONTH_NAMES[i % 12]}-{2024 + i // 12}" for i in range(14)]


def _trend_config(config: mod.AppConfig, *, confirm_prev: bool = True) -> mod.AppConfig:
    return _with_periods(
        config,
        _TREND_PERIODS,
        rolling_window=12,
        decline_thresh_pct=10.0,
        min_coverage=0.75,
        require_two_months=confirm_prev,
        exclude_data={},
    )


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


def test_detect_negative_trends_flags_large_decline(config: mod.AppConfig) -> None:
    # Baseline months 1-12 = 100, month 13 (prev) = 80, month 14 (latest) = 70
    vals = [100.0] * 12 + [80.0, 70.0]
    df = _trend_df(vals)
    result = mod.detect_negative_trends_12m(df, _trend_config(config))
    flagged = result[result["route"] == "101"]
    assert not flagged.empty
    assert "weekday_avg" in flagged["metric"].to_numpy()


def test_detect_negative_trends_no_flag_below_threshold(config: mod.AppConfig) -> None:
    # Only 5% decline — below 10% threshold
    vals = [100.0] * 13 + [95.0]
    df = _trend_df(vals)
    result = mod.detect_negative_trends_12m(df, _trend_config(config, confirm_prev=False))
    assert result.empty or "101" not in result["route"].to_numpy()


def test_detect_negative_trends_confirm_prev_blocks_single_month_drop(
    config: mod.AppConfig,
) -> None:
    # Latest is far below baseline, but previous month is NOT below baseline
    vals = [100.0] * 13 + [70.0]
    df = _trend_df(vals)
    result = mod.detect_negative_trends_12m(df, _trend_config(config))
    # confirm_prev=True: month 13 == 100 == baseline, so prev is not below — no flag
    assert result.empty or "101" not in result["route"].to_numpy()


def test_detect_negative_trends_returns_empty_dataframe_on_no_flags(
    config: mod.AppConfig,
) -> None:
    vals = [100.0] * 14
    df = _trend_df(vals)
    result = mod.detect_negative_trends_12m(df, _trend_config(config))
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_detect_negative_trends_counts_missing_catalogue_month_as_gap(
    config: mod.AppConfig,
) -> None:
    # Without Jun-2024 in the catalogue, the 12-month baseline still spans that
    # calendar month, so only 11 of its 12 months hold data and full coverage fails.
    df = _trend_df([100.0] * 12 + [80.0, 70.0])
    full = replace(_trend_config(config), min_coverage=1.0)
    assert not mod.detect_negative_trends_12m(df, full).empty

    gapped = replace(full, periods={p: s for p, s in full.periods.items() if p != "Jun-2024"})
    assert gapped.timeline == _TREND_PERIODS
    assert mod.detect_negative_trends_12m(df[df["period"] != "Jun-2024"], gapped).empty


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
# load_config / parse_config
# ---------------------------------------------------------------------------


def _write_variant(tmp_path: Path, old: str, new: str) -> Path:
    """Write the example config with *old* (which must occur once) replaced by *new*."""
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "ntd_monthly_config.txt"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_load_config_reads_example(config: mod.AppConfig) -> None:
    assert config.ordered_periods[0] == "Jul-2024"
    assert config.ordered_periods[-1] == "May-2025"
    assert config.periods["Aug-2024"] == mod.PeriodSpec(
        "AUGUST 2024  NTD RIDERSHIP REPORT BY ROUTE.xlsx", "Temporary_Query_N"
    )
    assert [w.label for w in config.time_windows] == [
        "FY21",
        "FY22",
        "FY23",
        "FY24",
        "FY25",
        "FY26",
    ]
    assert len(config.holidays) == 14
    assert config.service_types["express"] == ["303", "404"]
    assert config.garage_assignments == {}
    assert config.plot_style == mod.PlotStyle((9.0, 5.0), "o", "-", 45.0, True)
    assert config.log_level == logging.INFO
    assert config.from_run_log is False


def test_load_config_sorts_periods_chronologically(tmp_path: Path) -> None:
    path = _write_variant(tmp_path, "[periods]\n", "[periods]\nJun-2025 = june.xlsx | Jun. 2025\n")
    assert mod.load_config(path).ordered_periods[-1] == "Jun-2025"


def test_load_config_resolves_relative_paths_beside_config(tmp_path: Path) -> None:
    path = _write_variant(tmp_path, r"data_root = Path\To\Your\NTD_Folder", "data_root = workbooks")
    assert mod.load_config(path).data_root == (tmp_path / "workbooks").resolve()


@pytest.mark.parametrize("value", [r"\\server\share\ntd", r"C:\reports\ntd"])
def test_load_config_keeps_windows_absolute_paths(tmp_path: Path, value: str) -> None:
    path = _write_variant(tmp_path, r"data_root = Path\To\Your\NTD_Folder", f"data_root = {value}")
    assert str(mod.load_config(path).data_root) == value


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[corridors]", "[corridor]", "Sections: missing"),
        ("grid = true", "grid = true\ncolour = red", "unknown settings"),
        ("express = 303, 404", "express = 101, 404", "appears in both"),
        ("FY26 = 2025-07-01 | 2026-06-30", "FY26 = 2025-07-01 | 2026-06-29", "complete months"),
        ("FY21 = 2020-07-01", "plots = 2020-07-01", "reserved"),
        ("# Sep-2024 = *", "Jun-2025 = *", "absent from [periods]"),
        ("2027-01-01", "2026-12-31", "duplicate dates"),
        ("Jul-2024 = JULY", "July-2024 = JULY", "Invalid period"),
        ("level = INFO", "level = LOUD", "choose one of"),
        (r"data_root = Path\To\Your\NTD_Folder", r"data_root = C:reports", "drive-relative"),
    ],
)
def test_load_config_rejects_invalid_settings(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    with pytest.raises(mod.ConfigError, match=re.escape(message)):
        mod.load_config(_write_variant(tmp_path, old, new))


def test_load_config_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(mod.ConfigError):
        mod.load_config(tmp_path / "missing.txt")


def test_timeline_keeps_missing_catalogue_months(config: mod.AppConfig) -> None:
    cfg = _with_periods(config, ["Nov-2024", "Feb-2025"])
    assert cfg.timeline == ["Nov-2024", "Dec-2024", "Jan-2025", "Feb-2025"]


# ---------------------------------------------------------------------------
# write_run_log
# ---------------------------------------------------------------------------


def _run_log_text(output_dir: Path) -> str:
    return (output_dir / "ntd_monthly_summary_runlog.txt").read_text(encoding="utf-8")


def test_write_run_log_returns_true_on_success(tmp_path: Path, config: mod.AppConfig) -> None:
    assert mod.write_run_log(replace(config, output_dir=tmp_path)) is True


def test_write_run_log_returns_false_when_folder_missing(
    tmp_path: Path, config: mod.AppConfig
) -> None:
    assert mod.write_run_log(replace(config, output_dir=tmp_path / "missing")) is False


def test_write_run_log_content_includes_header_and_status(
    tmp_path: Path, config: mod.AppConfig
) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path), status="started")
    content = _run_log_text(tmp_path)
    assert content.splitlines()[1] == "NTD MONTHLY SUMMARY RUN LOG"
    assert "Run status:       started" in content
    assert "Configuration source: config file" in content


def test_write_run_log_content_includes_config_between_markers(
    tmp_path: Path, config: mod.AppConfig
) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path))
    content = _run_log_text(tmp_path)
    expected = f"{mod.REPLAY_CONFIG_BEGIN}\n{config.source_text}\n{mod.REPLAY_CONFIG_END}"
    assert expected in content


def test_write_run_log_omits_service_day_section_when_none(
    tmp_path: Path, config: mod.AppConfig
) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path))
    assert "SERVICE-DAY COUNTS PER MONTH" not in _run_log_text(tmp_path)


def _service_days() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["Jul-2024"],
            "Weekday": [22],
            "Saturday": [4],
            "Sunday": [5],
            "Holidays": [1],
        }
    )


def test_write_run_log_includes_service_day_table(tmp_path: Path, config: mod.AppConfig) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path), _service_days())
    content = _run_log_text(tmp_path)
    assert "SERVICE-DAY COUNTS PER MONTH" in content
    assert "Holidays" in content
    assert "Jul-2024" in content


# ---------------------------------------------------------------------------
# Run-log replay
# ---------------------------------------------------------------------------


def test_load_config_replays_run_log(tmp_path: Path, config: mod.AppConfig) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path), _service_days())
    replayed = mod.load_config(tmp_path / "ntd_monthly_summary_runlog.txt")
    assert replayed.from_run_log is True
    assert replayed.source_text == config.source_text
    assert replayed.periods == config.periods
    assert replayed.data_root == config.data_root
    assert replayed.config_base_dir == EXAMPLE_CONFIG.resolve().parent


def test_load_config_replays_moved_run_log_against_original_directory(
    tmp_path: Path, config: mod.AppConfig
) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path))
    moved = tmp_path / "archive" / "old_runlog.txt"
    moved.parent.mkdir()
    shutil.move(tmp_path / "ntd_monthly_summary_runlog.txt", moved)
    # The example's data_root is relative, so it must still resolve beside the
    # original config file rather than beside the moved log.
    assert mod.load_config(moved).data_root == config.data_root


def test_load_config_replays_unmarked_legacy_run_log(tmp_path: Path, config: mod.AppConfig) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path), _service_days())
    log_path = tmp_path / "ntd_monthly_summary_runlog.txt"
    new_only = (
        mod.REPLAY_CONFIG_BEGIN,
        mod.REPLAY_CONFIG_END,
        "Config base",
        "Configuration source",
    )
    legacy = [
        line for line in _run_log_text(tmp_path).splitlines() if not line.startswith(new_only)
    ]
    log_path.write_text("\n".join(legacy) + "\n", encoding="utf-8")

    replayed = mod.load_config(log_path)
    assert replayed.source_text == config.source_text
    assert replayed.data_root == config.data_root


def test_extract_replay_config_rejects_duplicate_markers(
    tmp_path: Path, config: mod.AppConfig
) -> None:
    mod.write_run_log(replace(config, output_dir=tmp_path))
    text = _run_log_text(tmp_path) + mod.REPLAY_CONFIG_BEGIN + "\n"
    with pytest.raises(mod.ConfigError, match="missing or duplicate"):
        mod.extract_replay_config(text)


def test_extract_replay_config_rejects_truncated_legacy_log() -> None:
    text = "\n".join(
        [
            "=" * 72,
            mod.RUN_LOG_TITLE,
            "=" * 72,
            "Config file:      /reports/ntd_monthly_config.txt",
            "-" * 72,
            "CONFIGURATION (text loaded for this run)",
            "-" * 72,
            "[paths]",
        ]
    )
    with pytest.raises(mod.ConfigError, match="truncated"):
        mod.extract_replay_config(text)


# ---------------------------------------------------------------------------
# main / cli (end to end)
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_root_log_level() -> Iterator[None]:
    """main() sets the root logger level from the config; undo it after the test."""
    root = logging.getLogger()
    level = root.level
    yield
    root.setLevel(level)


def _write_workbook(path: Path, boards_per_day: float) -> None:
    rows = [
        {
            "Route Name": route,
            "Service Period": service_period,
            "MTH_BOARD": boards_per_day * days,
            "MTH_REV_HOURS": 10.0 * days,
            "MTH_PASS_MILES": 4 * boards_per_day * days,
            "ASCH_TRIPS": 50,
            "DAYS": days,
            "REV_MILES": 400.0,
        }
        for route in (101, 202)
        for service_period, days in (("Weekday", 22), ("Saturday", 4), ("Sunday", 5))
    ]
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet1", index=False)


def _write_project(tmp_path: Path) -> Path:
    """Create two monthly workbooks and a config that refers to them relatively."""
    workbooks = tmp_path / "workbooks"
    workbooks.mkdir()
    _write_workbook(workbooks / "jul.xlsx", 100.0)
    _write_workbook(workbooks / "aug.xlsx", 90.0)
    plots = "\n".join(f"{flag} = {flag == 'plot_weekday_avg'}" for flag in mod.PLOT_FLAGS)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "ntd_monthly_config.txt"
    path.write_text(
        f"""\
[paths]
data_root = ../workbooks
output_dir = ../out

[periods]
Jul-2024 = jul.xlsx | Sheet1
Aug-2024 = aug.xlsx | Sheet1

[time_windows]
FY25 = 2024-07-01 | 2025-06-30

[holidays]
dates = 2024-07-04

[service_types]
local = 101, 202

[corridors]
main_street = 101

[garage_assignments]
north = 101

[trend]
rolling_window = 12
min_coverage = 0.75
decline_thresh_pct = 10
require_two_months = true

[exclude_data]

[plots]
{plots}

[plot_style]
figsize = 9, 5
marker = o
linestyle = -
rotation = 45
grid = true

[logging]
level = INFO
require_run_log = true
""",
        encoding="utf-8",
    )
    return path


def test_main_writes_all_outputs(tmp_path: Path, restore_root_log_level: None) -> None:
    assert mod.main(_write_project(tmp_path)) == 0

    out = tmp_path / "out"
    for name in (
        "DetailedAllPeriods_for_plotting.csv",
        "MonthlySheets.xlsx",
        "RouteLevelSummary_Combined.xlsx",
        "NegativeTrendFlags.txt",
        "FY25/detailed_FY25_for_plotting.csv",
        "FY25/AggByServiceType_FY25.xlsx",
        "plots/weekday_avg/weekday_avg_route_SYSTEMWIDE.png",
    ):
        assert (out / name).exists(), name
    weekday = pd.read_excel(out / "RouteLevelSummary_Weekday.xlsx", dtype={"ROUTE_NAME": str})
    assert dict(zip(weekday["ROUTE_NAME"], weekday["garage"], strict=True)) == {
        "101": "north",
        "202": "unknown",
    }
    # Jul 4 2024 is a Thursday, so the weekday summary gains holiday-free columns.
    assert "DAYS_EX_HOLIDAYS" in weekday.columns
    log = _run_log_text(out)
    assert "Run status:       complete" in log
    assert "SERVICE-DAY COUNTS PER MONTH" in log


def test_main_replays_run_log_into_new_folder(tmp_path: Path, restore_root_log_level: None) -> None:
    config_path = _write_project(tmp_path)
    assert mod.main(config_path) == 0
    out = tmp_path / "out"
    original_log = _run_log_text(out)

    # Replay from a moved copy of the log with the original config file gone.
    moved = tmp_path / "archive" / "runlog.txt"
    moved.parent.mkdir()
    shutil.copy(out / "ntd_monthly_summary_runlog.txt", moved)
    config_path.unlink()
    assert mod.main(moved) == 0

    reruns = list(tmp_path.glob("out_rerun_*"))
    assert len(reruns) == 1
    rerun_log = _run_log_text(reruns[0])
    assert "Configuration source: previous run log" in rerun_log
    assert f"Config base directory: {config_path.resolve().parent}" in rerun_log
    pd.testing.assert_frame_equal(
        pd.read_csv(reruns[0] / "DetailedAllPeriods_for_plotting.csv"),
        pd.read_csv(out / "DetailedAllPeriods_for_plotting.csv"),
    )
    assert _run_log_text(out) == original_log


def test_main_missing_workbook_fails_before_creating_output(
    tmp_path: Path, restore_root_log_level: None
) -> None:
    config_path = _write_project(tmp_path)
    (tmp_path / "workbooks" / "aug.xlsx").unlink()
    assert mod.main(config_path) == 1
    assert not (tmp_path / "out").exists()


def test_main_returns_2_on_config_error(tmp_path: Path) -> None:
    assert mod.main(tmp_path / "missing.txt") == 2


def test_cli_validate_config_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], restore_root_log_level: None
) -> None:
    config_path = _write_project(tmp_path)
    with patch.object(
        mod.sys,
        "argv",
        ["ntd_monthly_summary.py", "--config", str(config_path), "--validate-config"],
    ):
        assert mod.cli() == 0
    assert "Configuration valid: 2 months, 1 time windows" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("argv0", "expected"),
    [
        ("/env/lib/ipykernel_launcher.py", True),
        (r"C:\env\Lib\site-packages\ipykernel_launcher.py", True),
        ("ntd_monthly_summary.py", False),
    ],
)
def test_launched_by_notebook(argv0: str, expected: bool) -> None:
    with patch.object(mod.sys, "argv", [argv0, "-f", "kernel.json"]):
        assert mod.launched_by_notebook() is expected
