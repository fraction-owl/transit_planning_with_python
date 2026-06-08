"""Tests for scripts/exogenous_tools/clean_eia_gas_prices.py.

Exercised against a trimmed EIA "Data 1" weekly retail price export. The
fixture reproduces the structural quirks the cleaner must handle: a three-row
header block (banner, Sourcekey row, long-name row), opaque sourcekeys that map
to compact tokens, ``DD-Mon-YY`` dates spanning the century rollover, a series
that is empty over the whole file, and a weekly cadence rolled up to monthly.
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.exogenous_tools import clean_eia_gas_prices as mod

FIXTURE_CSV = Path("tests/fixtures/PET_PRI_GND_DCUS_NUS_W_Data_1__sample.csv")


# ---------------------------------------------------------------------------
# load_eia_weekly
# ---------------------------------------------------------------------------


def test_load_eia_weekly_renames_date_axis_and_tokens() -> None:
    df = mod.load_eia_weekly(FIXTURE_CSV)
    assert df.columns[0] == "DATE"
    # Opaque sourcekeys become compact tokens by default.
    assert "gas_regular_all" in df.columns
    assert "gas_allgrade_all" in df.columns
    assert "diesel_no2_all" in df.columns
    # No raw sourcekey leaks through.
    assert not any(c.startswith("EMM_") or c.startswith("EMD_") for c in df.columns)


def test_load_eia_weekly_long_names_uses_eia_labels() -> None:
    df = mod.load_eia_weekly(FIXTURE_CSV, use_long_names=True)
    assert "gas_regular_all" not in df.columns
    assert any("Regular All Formulations" in str(c) for c in df.columns)


# ---------------------------------------------------------------------------
# _parse_dates
# ---------------------------------------------------------------------------


def test_parse_dates_handles_century_rollover() -> None:
    parsed = mod._parse_dates(pd.Series(["20-Aug-90", "20-Apr-26"]))
    assert parsed.iloc[0] == pd.Timestamp("1990-08-20")
    assert parsed.iloc[1] == pd.Timestamp("2026-04-20")


# ---------------------------------------------------------------------------
# clean_weekly
# ---------------------------------------------------------------------------


def test_clean_weekly_parses_sorts_and_drops_empty_series() -> None:
    weekly = mod.clean_weekly(mod.load_eia_weekly(FIXTURE_CSV))
    assert pd.api.types.is_datetime64_any_dtype(weekly["DATE"])
    assert weekly["DATE"].is_monotonic_increasing
    # diesel_no2_lsd has no readings anywhere in the fixture -> dropped.
    assert "diesel_no2_lsd" not in weekly.columns
    # A series with readings survives.
    assert "gas_regular_all" in weekly.columns


# ---------------------------------------------------------------------------
# to_monthly
# ---------------------------------------------------------------------------


@pytest.fixture
def monthly() -> pd.DataFrame:
    return mod.to_monthly(mod.clean_weekly(mod.load_eia_weekly(FIXTURE_CSV)))


def test_to_monthly_row_per_calendar_month(monthly: pd.DataFrame) -> None:
    # Weeks fall in Aug/Sep/Oct 1990 and Apr/May/Jun 2026 -> six months.
    assert len(monthly) == 6
    assert list(monthly.columns[:4]) == ["MONTH_START", "YEAR", "MONTH", "N_WEEKS"]
    assert monthly["MONTH_START"].is_monotonic_increasing


def test_to_monthly_counts_survey_weeks(monthly: pd.DataFrame) -> None:
    n_weeks = dict(zip(monthly["MONTH_START"].dt.strftime("%Y-%m"), monthly["N_WEEKS"]))
    assert n_weeks["1990-08"] == 2
    assert n_weeks["1990-09"] == 4
    assert n_weeks["2026-05"] == 4
    assert monthly["N_WEEKS"].dtype == "int8"


def test_to_monthly_value_is_mean_of_weeks(monthly: pd.DataFrame) -> None:
    # Aug-1990 regular gas = mean(1.191, 1.245) = 1.218.
    aug = monthly[monthly["MONTH_START"] == pd.Timestamp("1990-08-01")]
    assert aug["gas_regular_all"].iloc[0] == pytest.approx(1.218)


def test_to_monthly_rounds_price_means(monthly: pd.DataFrame) -> None:
    value_cols = [c for c in monthly.columns if c not in mod.ID_COLS]
    for col in value_cols:
        non_null = monthly[col].dropna()
        assert (non_null.round(mod.ROUND_DECIMALS) == non_null).all()


# ---------------------------------------------------------------------------
# run (end to end)
# ---------------------------------------------------------------------------


def test_run_writes_monthly_csv_and_returns_frame(tmp_path: Path) -> None:
    out_path = tmp_path / "eia_monthly.csv"
    result = mod.run(FIXTURE_CSV, out_path, write_log=False)

    assert out_path.exists()
    assert len(result) == 6

    reloaded = pd.read_csv(out_path)
    assert list(reloaded.columns[:4]) == ["MONTH_START", "YEAR", "MONTH", "N_WEEKS"]
    assert len(reloaded) == len(result)


def test_run_can_also_write_weekly_frame(tmp_path: Path) -> None:
    out_path = tmp_path / "eia_monthly.csv"
    mod.run(FIXTURE_CSV, out_path, write_log=False, write_weekly=True)
    weekly_path = out_path.with_name(f"{out_path.stem}_weekly{out_path.suffix}")
    assert weekly_path.exists()
    weekly = pd.read_csv(weekly_path)
    assert "DATE" in weekly.columns
    assert len(weekly) == 15  # all dated weekly rows in the fixture
