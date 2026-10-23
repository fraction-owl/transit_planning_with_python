import logging
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts.field_tools import gtfs_service_change_dates as change_dates_mod
from scripts.ridership_tools import ntd_monthly_trends_export as mod

FIXTURE_CSV = Path("tests/fixtures/ntd_monthly_multi_month.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_CSV)


# ---------------------------------------------------------------------------
# parse_month / format_month
# ---------------------------------------------------------------------------


def test_parse_month_returns_month_start() -> None:
    assert mod.parse_month("Jan-2024") == datetime(2024, 1, 1)


def test_parse_month_december() -> None:
    assert mod.parse_month("Dec-2025") == datetime(2025, 12, 1)


def test_format_month_roundtrip() -> None:
    assert mod.format_month(mod.parse_month("Mar-2026")) == "Mar-2026"


# ---------------------------------------------------------------------------
# month_range
# ---------------------------------------------------------------------------


def test_month_range_inclusive() -> None:
    result = mod.month_range(datetime(2025, 12, 1), datetime(2026, 2, 1))
    assert result == [datetime(2025, 12, 1), datetime(2026, 1, 1), datetime(2026, 2, 1)]


def test_month_range_single_month() -> None:
    dt = datetime(2024, 6, 1)
    assert mod.month_range(dt, dt) == [dt]


def test_month_range_empty_when_start_after_end() -> None:
    assert mod.month_range(datetime(2026, 3, 1), datetime(2026, 1, 1)) == []


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------


def test_safe_float_numeric_string() -> None:
    assert mod.safe_float("1234") == 1234.0


def test_safe_float_comma_formatted() -> None:
    assert mod.safe_float("1,234.56") == pytest.approx(1234.56)


def test_safe_float_blank() -> None:
    assert mod.safe_float("") is None


def test_safe_float_na() -> None:
    assert mod.safe_float(float("nan")) is None


def test_safe_float_non_numeric() -> None:
    assert mod.safe_float("N/A") is None


# ---------------------------------------------------------------------------
# normalise_route
# ---------------------------------------------------------------------------


def test_normalise_route_strips_trailing_dot_zero() -> None:
    assert mod.normalise_route("610.0") == "610"


def test_normalise_route_trims_spaces() -> None:
    assert mod.normalise_route("  101  ") == "101"


def test_normalise_route_uppercases() -> None:
    assert mod.normalise_route("abc") == "ABC"


# ---------------------------------------------------------------------------
# normalise_service_period
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("weekday", "Weekday"),
        ("wkday", "Weekday"),
        ("wkdy", "Weekday"),
        ("Weekday", "Weekday"),
        ("sat", "Saturday"),
        ("Saturday", "Saturday"),
        ("sun", "Sunday"),
        ("Sunday", "Sunday"),
        ("unknown", "unknown"),  # passthrough for unrecognised values
    ],
)
def test_normalise_service_period(raw: str, expected: str) -> None:
    assert mod.normalise_service_period(raw) == expected


# ---------------------------------------------------------------------------
# normalise_columns
# ---------------------------------------------------------------------------


def test_normalise_columns_uppercases_and_underscores() -> None:
    df = pd.DataFrame({"Route Name": [1], "service period": [2]})
    result = mod.normalise_columns(df)
    assert list(result.columns) == ["ROUTE_NAME", "SERVICE_PERIOD"]


def test_normalise_columns_strips_leading_trailing_whitespace() -> None:
    df = pd.DataFrame({"  MTH BOARD  ": [5]})
    result = mod.normalise_columns(df)
    assert "MTH_BOARD" in result.columns


# ---------------------------------------------------------------------------
# read_month_workbook
# ---------------------------------------------------------------------------


def test_read_month_workbook_converts_numbers_under_unnormalised_headers(tmp_path: Path) -> None:
    path = tmp_path / "Jan 2026.xlsx"
    pd.DataFrame(
        {
            "Route Name": ["101", "101"],
            "Service Period": ["Weekday", "Weekday"],
            "Mth Board": ["1,000", "2,000"],
            "Days": ["10", "n/a"],
        }
    ).to_excel(path, index=False)

    df = mod.read_month_workbook("Jan-2026", path)

    assert df["MTH_BOARD"].tolist() == [1000.0, 2000.0]
    assert df["DAYS"].iloc[0] == 10.0
    assert pd.isna(df["DAYS"].iloc[1])


# ---------------------------------------------------------------------------
# aggregate_monthly_long
# ---------------------------------------------------------------------------


def test_aggregate_monthly_long_basic(fixture_df: pd.DataFrame) -> None:
    dt_dec = datetime(2025, 12, 1)
    raw = fixture_df[
        (fixture_df["MTH_YR"] == "December 2025") & fixture_df["ROUTE_NAME"].isin([101, 202])
    ].copy()
    raw["ROUTE_NAME"] = raw["ROUTE_NAME"].astype(str)
    raw["SERVICE_PERIOD"] = raw["SERVICE_PERIOD"].apply(mod.normalise_service_period)
    raw["period"] = "Dec-2025"
    raw["period_dt"] = dt_dec

    with (
        patch.object(mod, "ROUTES", ["101", "202"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday", "Sunday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt_dec])

    assert len(monthly_long) == 6  # 2 routes × 3 service periods
    assert ("101", dt_dec, "Weekday") in observed_keys
    assert ("202", dt_dec, "Saturday") in observed_keys


def test_aggregate_monthly_long_empty_raw_returns_na_grid() -> None:
    dt = datetime(2026, 1, 1)
    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(pd.DataFrame(), [dt])

    assert len(monthly_long) == 1
    assert observed_keys == set()
    assert pd.isna(monthly_long.iloc[0]["mth_board"])


def test_aggregate_monthly_long_computes_daily_avg(fixture_df: pd.DataFrame) -> None:
    dt_dec = datetime(2025, 12, 1)
    raw = fixture_df[
        (fixture_df["MTH_YR"] == "December 2025") & (fixture_df["ROUTE_NAME"] == 101)
    ].copy()
    raw["ROUTE_NAME"] = "101"
    raw["SERVICE_PERIOD"] = raw["SERVICE_PERIOD"].apply(mod.normalise_service_period)
    raw["period"] = "Dec-2025"
    raw["period_dt"] = dt_dec

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday", "Sunday"]),
    ):
        monthly_long, _ = mod.aggregate_monthly_long(raw, [dt_dec])

    wd = monthly_long[
        (monthly_long["route"] == "101") & (monthly_long["service_period"] == "Weekday")
    ]
    assert not wd.empty
    # Fixture: MTH_BOARD=6278, DAYS=22
    assert float(wd["daily_avg"].iloc[0]) == pytest.approx(6278 / 22, rel=1e-3)


def _weekday_rows(rows: list[tuple[str, float | None, float | None]]) -> pd.DataFrame:
    """Build Jan-2026 Weekday raw rows from (route, boardings, days)."""
    df = pd.DataFrame(rows, columns=["ROUTE_NAME", "MTH_BOARD", "DAYS"])
    df["SERVICE_PERIOD"] = "Weekday"
    df["period"] = "Jan-2026"
    df["period_dt"] = datetime(2026, 1, 1)
    return df


def test_aggregate_monthly_long_keeps_all_blank_entry_blank() -> None:
    dt = datetime(2026, 1, 1)
    raw = _weekday_rows([("101", None, None), ("101", None, None)])

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt])
        flags = mod.flag_outages(monthly_long, [dt], observed_keys, {dt})

    assert pd.isna(monthly_long["mth_board"].iloc[0])
    assert pd.isna(monthly_long["days"].iloc[0])
    assert flags["flag"].tolist() == ["missing_entry"]


# ---------------------------------------------------------------------------
# flag_outages
# ---------------------------------------------------------------------------


def test_flag_outages_zero_ridership_nonzero_days(fixture_df: pd.DataFrame) -> None:
    """Route 303, Dec-2025, Weekday has MTH_BOARD=0 and DAYS=22 in the fixture."""
    dt_dec = datetime(2025, 12, 1)
    raw = fixture_df[fixture_df["MTH_YR"] == "December 2025"].copy()
    raw["ROUTE_NAME"] = raw["ROUTE_NAME"].astype(str)
    raw["SERVICE_PERIOD"] = raw["SERVICE_PERIOD"].apply(mod.normalise_service_period)
    raw["period"] = "Dec-2025"
    raw["period_dt"] = dt_dec

    with (
        patch.object(mod, "ROUTES", ["101", "202", "303"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday", "Sunday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt_dec])
        flags = mod.flag_outages(monthly_long, [dt_dec], observed_keys, {dt_dec})

    zero_flags = flags[flags["flag"] == "zero_ridership_nonzero_days"]
    assert any(
        r == "303" and p == "Dec-2025" and sp == "Weekday"
        for r, p, sp in zip(zero_flags["route"], zero_flags["period"], zero_flags["service_period"])
    )


def test_flag_outages_zero_days(fixture_df: pd.DataFrame) -> None:
    """Route 404, Feb-2026, Sunday has DAYS=0 in the fixture."""
    dt_feb = datetime(2026, 2, 1)
    raw = fixture_df[fixture_df["MTH_YR"] == "February 2026"].copy()
    raw["ROUTE_NAME"] = raw["ROUTE_NAME"].astype(str)
    raw["SERVICE_PERIOD"] = raw["SERVICE_PERIOD"].apply(mod.normalise_service_period)
    raw["period"] = "Feb-2026"
    raw["period_dt"] = dt_feb

    with (
        patch.object(mod, "ROUTES", ["404"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday", "Sunday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt_feb])
        flags = mod.flag_outages(monthly_long, [dt_feb], observed_keys, {dt_feb})

    zero_day_flags = flags[flags["flag"] == "zero_days"]
    assert any(
        r == "404" and sp == "Sunday"
        for r, sp in zip(zero_day_flags["route"], zero_day_flags["service_period"])
    )


def test_flag_outages_missing_service_period() -> None:
    dt = datetime(2026, 1, 1)
    raw = pd.DataFrame(
        {
            "ROUTE_NAME": ["101"],
            "SERVICE_PERIOD": ["Weekday"],
            "MTH_BOARD": [5000.0],
            "DAYS": [21.0],
            "period": ["Jan-2026"],
            "period_dt": [dt],
        }
    )

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday", "Sunday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt])
        flags = mod.flag_outages(monthly_long, [dt], observed_keys, {dt})

    missing = flags[flags["flag"] == "missing_service_period"]
    missing_sps = set(zip(missing["route"], missing["service_period"]))
    assert ("101", "Saturday") in missing_sps
    assert ("101", "Sunday") in missing_sps


def test_flag_outages_clean_data_no_flags() -> None:
    dt = datetime(2026, 1, 1)
    raw = pd.DataFrame(
        {
            "ROUTE_NAME": ["101", "101", "101"],
            "SERVICE_PERIOD": ["Weekday", "Saturday", "Sunday"],
            "MTH_BOARD": [5000.0, 800.0, 600.0],
            "DAYS": [21.0, 5.0, 4.0],
            "period": ["Jan-2026"] * 3,
            "period_dt": [dt] * 3,
        }
    )

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday", "Sunday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt])
        flags = mod.flag_outages(monthly_long, [dt], observed_keys, {dt})

    assert flags.empty


def test_flag_outages_flags_partly_blank_entry_and_excludes_it_from_yoy() -> None:
    # Two schedule rows for the month; the second has no boardings, so the sum
    # understates the month.
    dt = datetime(2026, 1, 1)
    raw = _weekday_rows([("101", 1000.0, 10.0), ("101", None, 12.0)])

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt])
        flags = mod.flag_outages(monthly_long, [dt], observed_keys, {dt})

    assert monthly_long["missing_value_rows"].iloc[0] == 1
    assert flags["flag"].tolist() == ["incomplete_entry"]
    assert pd.isna(mod.clean_daily_avg(monthly_long, flags)["daily_avg"].iloc[0])


@pytest.mark.parametrize(
    "board,days,flag",
    [(-50.0, 21.0, "negative_boardings"), (500.0, -1.0, "negative_days")],
)
def test_flag_outages_flags_negative_values(board: float, days: float, flag: str) -> None:
    dt = datetime(2026, 1, 1)
    raw = _weekday_rows([("101", board, days)])

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt])
        flags = mod.flag_outages(monthly_long, [dt], observed_keys, {dt})

    assert flags["flag"].tolist() == [flag]


# ---------------------------------------------------------------------------
# apply_manual_fixes
# ---------------------------------------------------------------------------


def test_manual_fix_of_missing_entry_clears_its_flag_and_keeps_its_average() -> None:
    dt = datetime(2026, 1, 1)
    raw = _weekday_rows([("101", 5000.0, 21.0)])

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday"]),
        patch.object(mod, "PROMPT_FOR_FIXES", True),
        patch("builtins.input", side_effect=["800", "4"]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt])
        flags = mod.flag_outages(monthly_long, [dt], observed_keys, {dt})
        monthly_long, fixed_keys = mod.apply_manual_fixes(monthly_long, flags)
        flags = mod.flag_outages(monthly_long, [dt], observed_keys | fixed_keys, {dt})

    assert fixed_keys == {("101", dt, "Saturday")}
    assert flags.empty
    cleaned = mod.clean_daily_avg(monthly_long, flags)
    sat = cleaned[cleaned["service_period"] == "Saturday"]
    assert float(sat["daily_avg"].iloc[0]) == pytest.approx(200.0)


def test_manual_fix_skipped_entry_stays_flagged() -> None:
    dt = datetime(2026, 1, 1)
    raw = _weekday_rows([("101", 1000.0, 10.0), ("101", None, 12.0)])

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday"]),
        patch.object(mod, "PROMPT_FOR_FIXES", True),
        patch("builtins.input", side_effect=["", ""]),
    ):
        monthly_long, observed_keys = mod.aggregate_monthly_long(raw, [dt])
        flags = mod.flag_outages(monthly_long, [dt], observed_keys, {dt})
        monthly_long, fixed_keys = mod.apply_manual_fixes(monthly_long, flags)
        flags = mod.flag_outages(monthly_long, [dt], observed_keys | fixed_keys, {dt})

    assert fixed_keys == set()
    assert flags["flag"].tolist() == ["incomplete_entry"]


# ---------------------------------------------------------------------------
# to_wide
# ---------------------------------------------------------------------------


def test_to_wide_has_expected_columns() -> None:
    dt = datetime(2026, 1, 1)
    monthly_long = pd.DataFrame(
        {
            "route": ["101"] * 3,
            "period_dt": [dt] * 3,
            "period": ["Jan-2026"] * 3,
            "service_period": ["Weekday", "Saturday", "Sunday"],
            "mth_board": [5000.0, 800.0, 600.0],
            "daily_avg": [238.1, 160.0, 150.0],
        }
    )

    wide = mod.to_wide(monthly_long)

    expected_cols = {
        "route",
        "period_dt",
        "period",
        "weekday_total",
        "saturday_total",
        "sunday_total",
        "weekday_avg",
        "saturday_avg",
        "sunday_avg",
    }
    assert expected_cols.issubset(set(wide.columns))


def test_to_wide_one_row_per_route_month() -> None:
    dt_dec = datetime(2025, 12, 1)
    dt_jan = datetime(2026, 1, 1)
    monthly_long = pd.DataFrame(
        {
            "route": ["101"] * 6,
            "period_dt": [dt_dec] * 3 + [dt_jan] * 3,
            "period": ["Dec-2025"] * 3 + ["Jan-2026"] * 3,
            "service_period": ["Weekday", "Saturday", "Sunday"] * 2,
            "mth_board": [6278.0, 575.0, 556.0, 6115.0, 734.0, 454.0],
            "daily_avg": [285.4, 143.8, 111.2, 291.2, 146.8, 113.5],
        }
    )

    wide = mod.to_wide(monthly_long)

    assert len(wide) == 2
    assert list(wide["period"]) == ["Dec-2025", "Jan-2026"]


def test_to_wide_values_match_long(fixture_df: pd.DataFrame) -> None:
    dt_dec = datetime(2025, 12, 1)
    raw = fixture_df[
        (fixture_df["MTH_YR"] == "December 2025") & (fixture_df["ROUTE_NAME"] == 101)
    ].copy()
    raw["ROUTE_NAME"] = "101"
    raw["SERVICE_PERIOD"] = raw["SERVICE_PERIOD"].apply(mod.normalise_service_period)
    raw["period"] = "Dec-2025"
    raw["period_dt"] = dt_dec

    with (
        patch.object(mod, "ROUTES", ["101"]),
        patch.object(mod, "SERVICE_PERIODS", ["Weekday", "Saturday", "Sunday"]),
    ):
        monthly_long, _ = mod.aggregate_monthly_long(raw, [dt_dec])

    wide = mod.to_wide(monthly_long)

    assert wide["weekday_total"].iloc[0] == pytest.approx(6278.0)
    assert wide["saturday_total"].iloc[0] == pytest.approx(575.0)
    assert wide["sunday_total"].iloc[0] == pytest.approx(556.0)


def test_to_wide_keeps_wholly_missing_month() -> None:
    months = [datetime(2026, 1, 1), datetime(2026, 2, 1), datetime(2026, 3, 1)]
    monthly_long = pd.DataFrame(
        {
            "route": ["101"] * 3,
            "period_dt": months,
            "period": ["Jan-2026", "Feb-2026", "Mar-2026"],
            "service_period": ["Weekday"] * 3,
            "mth_board": [1000.0, None, 1100.0],
            "daily_avg": [50.0, None, 52.4],
        }
    )

    wide = mod.to_wide(monthly_long)

    assert list(wide["period"]) == ["Jan-2026", "Feb-2026", "Mar-2026"]
    assert pd.isna(wide["weekday_total"].iloc[1])
    assert pd.isna(wide["weekday_avg"].iloc[1])


# ---------------------------------------------------------------------------
# compute_systemwide_perday_by_sp
# ---------------------------------------------------------------------------


def _raw_rows(period: str, rows: list[tuple[str, str, float | None, float]]) -> pd.DataFrame:
    """Build raw_all-shaped rows from (route, service_period, boardings, days)."""
    df = pd.DataFrame(rows, columns=["ROUTE_NAME", "SERVICE_PERIOD", "MTH_BOARD", "DAYS"])
    df["period"] = period
    df["period_dt"] = mod.parse_month(period)
    return df


def test_systemwide_perday_divides_by_month_service_days() -> None:
    # Four routes carry the same weekday ridership both years; two new low-ridership
    # routes start in Jul-2025. Systemwide boardings per weekday rise 2,800 -> 3,000,
    # whereas boardings per route-day would fall 700 -> 500.
    base = [("101", 1000.0), ("202", 800.0), ("303", 600.0), ("404", 400.0)]
    jul24 = _raw_rows("Jul-2024", [(r, "Weekday", per_day * 23, 23.0) for r, per_day in base])
    jul25 = _raw_rows(
        "Jul-2025",
        [
            (r, "Weekday", per_day * 23, 23.0)
            for r, per_day in [*base, ("505", 100.0), ("606", 100.0)]
        ],
    )

    result = mod.compute_systemwide_perday_by_sp(pd.concat([jul24, jul25], ignore_index=True), {})

    weekday = result["Weekday"]
    assert weekday[datetime(2024, 7, 1)] == pytest.approx(2800.0)
    assert weekday[datetime(2025, 7, 1)] == pytest.approx(3000.0)
    months = mod.month_range(datetime(2024, 7, 1), datetime(2025, 7, 1))
    yoy = mod.monthly_yoy_pct(weekday, months)
    assert yoy[datetime(2025, 7, 1)] == pytest.approx((3000 / 2800 - 1) * 100)


def test_systemwide_perday_counts_partial_month_route_in_proportion() -> None:
    # Route 303 ran 10 of the month's 20 weekdays: its boardings count toward the
    # 20-day month instead of adding 10 route-days to the denominator.
    raw = _raw_rows(
        "Jul-2024",
        [
            ("101", "Weekday", 4000.0, 20.0),
            ("202", "Weekday", 2000.0, 20.0),
            ("303", "Weekday", 1000.0, 10.0),
        ],
    )
    result = mod.compute_systemwide_perday_by_sp(raw, {})
    assert result["Weekday"][datetime(2024, 7, 1)] == pytest.approx(7000 / 20)


def test_systemwide_perday_subtracts_weekday_holidays_once() -> None:
    raw = _raw_rows(
        "Jul-2024",
        [
            ("101", "Weekday", 2300.0, 23.0),
            ("202", "Weekday", 4600.0, 23.0),
            ("101", "Saturday", 400.0, 4.0),
            ("202", "Saturday", 800.0, 4.0),
        ],
    )
    result = mod.compute_systemwide_perday_by_sp(raw, {"Jul-2024": 1})
    assert result["Weekday"][datetime(2024, 7, 1)] == pytest.approx(6900 / 22)
    assert result["Saturday"][datetime(2024, 7, 1)] == pytest.approx(1200 / 4)


def test_systemwide_perday_tie_in_service_days_goes_to_larger_count() -> None:
    raw = _raw_rows(
        "Jul-2024",
        [("101", "Weekday", 2000.0, 20.0), ("202", "Weekday", 1000.0, 10.0)],
    )
    result = mod.compute_systemwide_perday_by_sp(raw, {})
    assert result["Weekday"][datetime(2024, 7, 1)] == pytest.approx(3000 / 20)


def test_systemwide_perday_excludes_total_rows() -> None:
    raw = _raw_rows(
        "Jul-2024",
        [
            ("101", "Weekday", 2000.0, 20.0),
            ("202", "Weekday", 4000.0, 20.0),
            ("TOTAL", "Weekday", 6000.0, 20.0),
        ],
    )
    result = mod.compute_systemwide_perday_by_sp(raw, {})
    assert result["Weekday"][datetime(2024, 7, 1)] == pytest.approx(6000 / 20)


@pytest.mark.parametrize(
    "board,days",
    [(0.0, 20.0), (None, 20.0), (-10.0, 20.0)],
    ids=["zero_boardings", "blank_boardings", "negative_boardings"],
)
def test_systemwide_perday_leaves_out_flagged_route_and_warns(
    board: float | None, days: float, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _raw_rows(
        "Jul-2024",
        [("101", "Weekday", 2000.0, 20.0), ("202", "Weekday", board, days)],
    )

    with caplog.at_level(logging.WARNING):
        result = mod.compute_systemwide_perday_by_sp(raw, {})

    assert result["Weekday"][datetime(2024, 7, 1)] == pytest.approx(2000 / 20)
    assert "Systemwide Weekday Jul-2024" in caplog.text
    assert "202 (" in caplog.text


def test_systemwide_perday_skips_route_with_no_service() -> None:
    raw = _raw_rows(
        "Jul-2024",
        [("101", "Sunday", 400.0, 4.0), ("202", "Sunday", 0.0, 0.0)],
    )
    result = mod.compute_systemwide_perday_by_sp(raw, {})
    assert result["Sunday"][datetime(2024, 7, 1)] == pytest.approx(100.0)


def test_systemwide_perday_warns_on_interior_gap_but_not_route_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 202 reports Jul and Sep but not Aug (a gap); 303 starts in Aug (not a gap).
    raw = pd.concat(
        [
            _raw_rows(
                "Jul-2024", [("101", "Weekday", 2000.0, 20.0), ("202", "Weekday", 1000.0, 20.0)]
            ),
            _raw_rows(
                "Aug-2024", [("101", "Weekday", 2000.0, 20.0), ("303", "Weekday", 400.0, 20.0)]
            ),
            _raw_rows(
                "Sep-2024",
                [
                    ("101", "Weekday", 2000.0, 20.0),
                    ("202", "Weekday", 1000.0, 20.0),
                    ("303", "Weekday", 400.0, 20.0),
                ],
            ),
        ],
        ignore_index=True,
    )

    with caplog.at_level(logging.WARNING):
        weekday = mod.compute_systemwide_perday_by_sp(raw, {})["Weekday"]

    assert weekday[datetime(2024, 7, 1)] == pytest.approx(150.0)
    assert weekday[datetime(2024, 8, 1)] == pytest.approx(2400 / 20)
    assert weekday[datetime(2024, 9, 1)] == pytest.approx(170.0)
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "Weekday Aug-2024" in warnings[0]
    assert "202 (missing_service_period)" in warnings[0]


def test_systemwide_perday_uses_manual_fixes() -> None:
    raw = _raw_rows(
        "Jul-2024",
        [("101", "Weekday", 2000.0, 20.0), ("202", "Weekday", 0.0, 20.0)],
    )
    fixed = pd.DataFrame(
        {
            "route": ["202"],
            "period_dt": [datetime(2024, 7, 1)],
            "service_period": ["Weekday"],
            "mth_board": [1000.0],
            "days": [20.0],
        }
    )

    result = mod.compute_systemwide_perday_by_sp(raw, {}, fixed)

    assert result["Weekday"][datetime(2024, 7, 1)] == pytest.approx(3000 / 20)


def test_systemwide_perday_sums_schedule_rows_before_counting_service_days() -> None:
    # A mid-month pick splits every route into two schedule rows (10 + 12 days);
    # the month still has 22 service days, not 12.
    raw = _raw_rows(
        "Jul-2024",
        [
            ("101", "Weekday", 1000.0, 10.0),
            ("101", "Weekday", 1200.0, 12.0),
            ("202", "Weekday", 500.0, 10.0),
            ("202", "Weekday", 600.0, 12.0),
        ],
    )
    result = mod.compute_systemwide_perday_by_sp(raw, {})
    assert result["Weekday"][datetime(2024, 7, 1)] == pytest.approx(3300 / 22)


# ---------------------------------------------------------------------------
# Service change markers
# ---------------------------------------------------------------------------


def test_load_service_change_dates_reads_gtfs_service_change_dates_output(
    tmp_path: Path,
) -> None:
    # Written by the producing script's own table builder, so a change to its
    # CSV layout breaks this test.
    changes = [
        change_dates_mod.MergedChange(
            date(2024, 6, 9), "Service change", frozenset({"A"}), frozenset(), ("f1",), ()
        ),
        change_dates_mod.MergedChange(
            date(2025, 1, 12), "Service change", frozenset({"B"}), frozenset({"A"}), ("f2",), ()
        ),
        change_dates_mod.MergedChange(
            date(2025, 3, 30), "New service period", frozenset(), frozenset(), ("f3",), ()
        ),
    ]
    csv_path = tmp_path / "service_changes.csv"
    change_dates_mod.build_changes_table(changes).to_csv(csv_path, index=False)

    result = mod.load_service_change_dates(csv_path, datetime(2024, 7, 1), datetime(2025, 3, 1))

    # Jun-2024 is before the window; 30 Mar 2025 falls inside its last month.
    assert result == [datetime(2025, 1, 12), datetime(2025, 3, 30)]


def test_load_service_change_dates_none_draws_no_markers() -> None:
    assert mod.load_service_change_dates(None, datetime(2024, 1, 1), datetime(2025, 1, 1)) == []


def test_load_service_change_dates_warns_on_unreadable_or_wrong_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("date,notes\n2025-01-12,x\n")
    bounds = (datetime(2024, 1, 1), datetime(2025, 12, 1))

    with caplog.at_level(logging.WARNING):
        assert mod.load_service_change_dates(tmp_path / "missing.csv", *bounds) == []
        assert mod.load_service_change_dates(wrong, *bounds) == []

    assert "Could not read SERVICE_CHANGES_CSV" in caplog.text
    assert "no change_date column" in caplog.text


def test_load_service_change_dates_skips_unparseable_dates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "service_changes.csv"
    csv_path.write_text("change_date\n2025-01-12\n1/26/2025\n2025-01-12\n")

    with caplog.at_level(logging.WARNING):
        result = mod.load_service_change_dates(csv_path, datetime(2025, 1, 1), datetime(2025, 1, 1))

    assert result == [datetime(2025, 1, 12)]
    assert "1/26/2025" in caplog.text


def test_draw_service_changes_adds_one_line_per_date_and_one_legend_entry() -> None:
    fig, ax = plt.subplots()
    try:
        mod.draw_service_changes(ax, [datetime(2025, 1, 12), datetime(2025, 3, 30)])
        assert len(ax.lines) == 2
        assert ax.get_legend_handles_labels()[1] == ["GTFS service change"]
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Stale outputs
# ---------------------------------------------------------------------------


def test_plot_route_change_removes_stale_chart_when_nothing_to_draw(tmp_path: Path) -> None:
    stale = tmp_path / "plots" / "yoy_percent_change.png"
    stale.parent.mkdir()
    stale.write_bytes(b"old")
    change = pd.DataFrame(
        {
            "period_dt": [datetime(2026, 1, 1)],
            "weekday_yoy_pct": [float("nan")],
            "weekday_yoy_rolling_pct": [float("nan")],
        }
    )

    with patch.object(mod, "SERVICE_PERIODS", ["Weekday"]):
        mod.plot_route_change(tmp_path, "101", change)

    assert not stale.exists()


def test_export_route_removes_stale_change_outputs_when_disabled(tmp_path: Path) -> None:
    route_dir = tmp_path / "route_101"
    (route_dir / "plots").mkdir(parents=True)
    stale_csv = route_dir / "yoy_percent_change.csv"
    stale_png = route_dir / "plots" / "yoy_percent_change.png"
    stale_csv.write_text("old")
    stale_png.write_bytes(b"old")
    dt = datetime(2026, 1, 1)
    monthly_long = pd.DataFrame(
        {
            "route": ["101"],
            "period_dt": [dt],
            "period": ["Jan-2026"],
            "service_period": ["Weekday"],
            "mth_board": [1000.0],
            "days": [20.0],
            "daily_avg": [50.0],
        }
    )
    flags = pd.DataFrame(columns=["route", "period", "service_period", "flag", "mth_board", "days"])

    with patch.object(mod, "OUTPUT_ROOT", tmp_path):
        mod.export_route("101", monthly_long, mod.to_wide(monthly_long), flags, None)

    assert not stale_csv.exists()
    assert not stale_png.exists()
    assert (route_dir / "plots" / "monthly_totals.png").exists()


# ---------------------------------------------------------------------------
# Integration: main() with mocked read_month_workbook
# ---------------------------------------------------------------------------


def test_main_integration(fixture_df: pd.DataFrame, tmp_path: Path) -> None:
    test_workbooks = {
        "Dec-2025": tmp_path / "dec.xlsx",
        "Jan-2026": tmp_path / "jan.xlsx",
        "Feb-2026": tmp_path / "feb.xlsx",
    }

    def mock_read_month_workbook(period: str, path: Path) -> pd.DataFrame:
        dt = datetime.strptime(period, "%b-%Y")
        fixture_month_str = dt.strftime("%B %Y")
        period_df = fixture_df[fixture_df["MTH_YR"] == fixture_month_str].copy()
        if period_df.empty:
            return pd.DataFrame()
        period_df["ROUTE_NAME"] = period_df["ROUTE_NAME"].apply(mod.normalise_route)
        period_df["SERVICE_PERIOD"] = period_df["SERVICE_PERIOD"].apply(
            mod.normalise_service_period
        )
        period_df["MTH_BOARD"] = pd.to_numeric(period_df["MTH_BOARD"], errors="coerce")
        period_df["DAYS"] = pd.to_numeric(period_df["DAYS"], errors="coerce")
        period_df["period"] = period
        period_df["period_dt"] = dt
        return period_df

    changes_csv = tmp_path / "service_changes.csv"
    changes_csv.write_text("change_date,change_type\n2026-01-12,Service change\n")

    with (
        patch.object(mod, "SERVICE_CHANGES_CSV", changes_csv),
        patch.object(mod, "draw_service_changes", wraps=mod.draw_service_changes) as draw,
        patch.object(mod, "discover_workbooks", return_value=test_workbooks),
        patch.object(mod, "START_MONTH", "Dec-2025"),
        patch.object(mod, "END_MONTH", "Feb-2026"),
        patch.object(mod, "ROUTES", ["101", "202"]),
        patch.object(mod, "DATA_ROOT", tmp_path),
        patch.object(mod, "OUTPUT_ROOT", tmp_path),
        patch.object(mod, "read_month_workbook", side_effect=mock_read_month_workbook),
    ):
        assert mod.main() == 0

    # Totals and averages plots for both routes marked the change date.
    assert draw.call_count == 4
    assert all(list(call.args[1]) == [datetime(2026, 1, 12)] for call in draw.call_args_list)

    # Only configured routes get output directories
    assert (tmp_path / "route_101").exists()
    assert (tmp_path / "route_202").exists()
    assert not (tmp_path / "route_303").exists()

    route_dir = tmp_path / "route_101"
    for fname in ("monthly_long.csv", "monthly_wide.csv", "outage_flags.csv"):
        assert (route_dir / fname).exists(), f"Missing {fname}"
    assert (route_dir / "plots" / "monthly_totals.png").exists()
    assert (route_dir / "plots" / "daily_averages.png").exists()

    # All three months present in monthly_long
    df_long = pd.read_csv(route_dir / "monthly_long.csv")
    assert {"Dec-2025", "Jan-2026", "Feb-2026"} == set(df_long["period"].unique())

    # One row per month in monthly_wide
    df_wide = pd.read_csv(route_dir / "monthly_wide.csv")
    assert len(df_wide) == 3

    # Combined outputs exist
    combined = tmp_path / "_combined"
    assert (combined / "all_routes_monthly_long.csv").exists()
    assert (combined / "all_routes_monthly_wide.csv").exists()
    assert (combined / "all_routes_outage_flags.csv").exists()
