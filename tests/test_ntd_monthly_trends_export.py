from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

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

    with (
        patch.object(mod, "discover_workbooks", return_value=test_workbooks),
        patch.object(mod, "START_MONTH", "Dec-2025"),
        patch.object(mod, "END_MONTH", "Feb-2026"),
        patch.object(mod, "ROUTES", ["101", "202"]),
        patch.object(mod, "DATA_ROOT", tmp_path),
        patch.object(mod, "OUTPUT_ROOT", tmp_path),
        patch.object(mod, "read_month_workbook", side_effect=mock_read_month_workbook),
    ):
        mod.main()

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
