"""Tests for scripts/national_data_tools/clean_fred_unrate.py.

Exercised against a trimmed FRED UNRATE export (``observation_date,UNRATE``)
carrying the quirks the cleaner is meant to absorb: a not-yet-released month
(blank value -> NaN), US-style ``M/D/YYYY`` dates, and the series-id header.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.national_data_tools import clean_fred_unrate as mod

FIXTURE_CSV = Path("tests/fixtures/UNRATE_sample.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_df() -> pd.DataFrame:
    # Mirror run()'s read: FRED's "." missing marker -> NaN at read time.
    return pd.read_csv(FIXTURE_CSV, na_values=mod.NA_VALUES)


# ---------------------------------------------------------------------------
# _resolve_date_col
# ---------------------------------------------------------------------------


def test_resolve_date_col_prefers_observation_date() -> None:
    cols = pd.Index(["observation_date", "UNRATE"])
    assert mod._resolve_date_col(cols) == "observation_date"


def test_resolve_date_col_falls_back_to_legacy_date() -> None:
    cols = pd.Index(["DATE", "UNRATE"])
    assert mod._resolve_date_col(cols) == "DATE"


def test_resolve_date_col_defaults_to_leftmost_column() -> None:
    cols = pd.Index(["when", "UNRATE"])
    assert mod._resolve_date_col(cols) == "when"


# ---------------------------------------------------------------------------
# _infer_frequency
# ---------------------------------------------------------------------------


def test_infer_frequency_month_start() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=12, freq="MS"))
    assert mod._infer_frequency(dates) == "MS"


def test_infer_frequency_quarter_start() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=8, freq="QS"))
    assert mod._infer_frequency(dates) == "QS"


def test_infer_frequency_irregular_returns_none() -> None:
    dates = pd.to_datetime(pd.Series(["2020-01-01", "2020-01-09", "2020-04-30"]))
    assert mod._infer_frequency(dates) is None


# ---------------------------------------------------------------------------
# clean_series
# ---------------------------------------------------------------------------


def test_clean_series_adds_calendar_columns(raw_df: pd.DataFrame) -> None:
    out = mod.clean_series(raw_df)
    # Calendar columns sit immediately after the date axis.
    assert list(out.columns[:4]) == ["observation_date", "YEAR", "MONTH", "QUARTER"]
    assert out["YEAR"].dtype == "int16"
    assert out["MONTH"].dtype == "int8"
    assert out["QUARTER"].dtype == "int8"


def test_clean_series_parses_and_sorts_dates(raw_df: pd.DataFrame) -> None:
    out = mod.clean_series(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(out["observation_date"])
    assert out["observation_date"].is_monotonic_increasing
    first = out.iloc[0]
    assert (first["YEAR"], first["MONTH"], first["QUARTER"]) == (1948, 1, 1)


def test_clean_series_preserves_missing_observation(raw_df: pd.DataFrame) -> None:
    # The blank Oct-2025 cell is a real hole and must stay NaN, not be filled.
    out = mod.clean_series(raw_df)
    assert int(out["UNRATE"].isna().sum()) == 1
    oct_2025 = out[(out["YEAR"] == 2025) & (out["MONTH"] == 10)]
    assert np.isnan(oct_2025["UNRATE"].iloc[0])


def test_clean_series_keeps_series_id_header_by_default(raw_df: pd.DataFrame) -> None:
    out = mod.clean_series(raw_df)
    assert "UNRATE" in out.columns


def test_clean_series_long_names_renames_known_series(raw_df: pd.DataFrame) -> None:
    out = mod.clean_series(raw_df, use_long_names=True)
    assert "Unemployment Rate" in out.columns
    assert "UNRATE" not in out.columns


def test_clean_series_drops_fully_blank_padding_row() -> None:
    df = pd.DataFrame(
        {
            "observation_date": ["2020-01-01", "2020-02-01", None],
            "UNRATE": [3.5, 3.6, None],
        }
    )
    out = mod.clean_series(df)
    assert len(out) == 2


def test_clean_series_drops_all_empty_value_column() -> None:
    # The multi-series fredgraph shape can carry a column empty over the window.
    df = pd.DataFrame(
        {
            "observation_date": ["2020-01-01", "2020-02-01"],
            "UNRATE": [3.5, 3.6],
            "PAYEMS": [None, None],
        }
    )
    out = mod.clean_series(df)
    assert "PAYEMS" not in out.columns
    assert "UNRATE" in out.columns


# ---------------------------------------------------------------------------
# run (end to end)
# ---------------------------------------------------------------------------


def test_run_writes_csv_and_returns_clean_frame(tmp_path: Path) -> None:
    out_path = tmp_path / "unrate_clean.csv"
    result = mod.run(FIXTURE_CSV, out_path, write_log=False)

    assert out_path.exists()
    assert len(result) == 23
    assert int(result["UNRATE"].isna().sum()) == 1

    reloaded = pd.read_csv(out_path)
    assert list(reloaded.columns[:4]) == ["observation_date", "YEAR", "MONTH", "QUARTER"]
    assert len(reloaded) == len(result)
