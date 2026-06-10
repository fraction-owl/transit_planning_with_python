"""Tests for scripts/national_data_tools/clean_noaa_weather.py.

Exercised against a trimmed NOAA CDO daily-summaries export for a single
station. The fixture carries the semantics the cleaner must respect: WT* event
flags that are blank when the event did not occur (not missing), columns that
are entirely empty for this station (PGTM, TAVG), and US-style dates.
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.national_data_tools import clean_noaa_weather as mod

FIXTURE_CSV = Path("tests/fixtures/4331222_sample.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_df() -> pd.DataFrame:
    # Mirror run()'s read: keep STATION as a string so the numeric-looking id
    # is not mangled by inference.
    return pd.read_csv(FIXTURE_CSV, dtype={"STATION": "string"})


# ---------------------------------------------------------------------------
# clean_weather
# ---------------------------------------------------------------------------


def test_clean_weather_adds_calendar_columns(raw_df: pd.DataFrame) -> None:
    out = mod.clean_weather(raw_df)
    date_pos = out.columns.get_loc("DATE")
    assert list(out.columns[date_pos : date_pos + 4]) == [
        "DATE",
        "YEAR",
        "MONTH",
        "DAY_OF_WEEK",
    ]
    # 2021-01-01 is a Friday.
    first = out.sort_values("DATE").iloc[0]
    assert first["DAY_OF_WEEK"] == "Friday"
    assert first["YEAR"] == 2021
    assert first["MONTH"] == 1


def test_clean_weather_sorts_dates(raw_df: pd.DataFrame) -> None:
    out = mod.clean_weather(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert out["DATE"].is_monotonic_increasing


def test_clean_weather_fills_wt_flags_to_int(raw_df: pd.DataFrame) -> None:
    out = mod.clean_weather(raw_df)
    wt_cols = [c for c in out.columns if c.startswith("WT")]
    assert wt_cols  # flags survived cleaning
    for col in wt_cols:
        assert out[col].dtype == "int8"
        assert out[col].notna().all()  # blanks became 0, not NaN
        assert set(out[col].unique()) <= {0, 1}


def test_clean_weather_keeps_never_fired_flag_as_all_zero(raw_df: pd.DataFrame) -> None:
    # WT05 / WT09 never fire in this fixture: all-zero, kept (not dropped).
    out = mod.clean_weather(raw_df)
    assert "WT05" in out.columns
    assert (out["WT05"] == 0).all()


def test_clean_weather_drops_all_empty_non_flag_columns(raw_df: pd.DataFrame) -> None:
    # PGTM and TAVG have no readings for this station -> dropped.
    out = mod.clean_weather(raw_df)
    assert "PGTM" not in out.columns
    assert "TAVG" not in out.columns


def test_clean_weather_preserves_station_identity(raw_df: pd.DataFrame) -> None:
    out = mod.clean_weather(raw_df)
    assert "STATION" in out.columns
    assert out["STATION"].iloc[0] == "USW00093738"


def test_clean_weather_coerces_measurements_to_numeric(raw_df: pd.DataFrame) -> None:
    out = mod.clean_weather(raw_df)
    assert pd.api.types.is_numeric_dtype(out["PRCP"])
    assert pd.api.types.is_numeric_dtype(out["TMAX"])


def test_clean_weather_long_names_renames_known_codes(raw_df: pd.DataFrame) -> None:
    out = mod.clean_weather(raw_df, use_long_names=True)
    assert "Maximum temperature" in out.columns
    assert "TMAX" not in out.columns


def test_clean_weather_drops_fully_blank_padding_row() -> None:
    df = pd.DataFrame(
        {
            "STATION": ["X", "X", None],
            "NAME": ["n", "n", None],
            "DATE": ["1/1/2021", "1/2/2021", None],
            "PRCP": [0.1, 0.2, None],
            "WT01": [1.0, None, None],
        }
    )
    out = mod.clean_weather(df)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# run (end to end)
# ---------------------------------------------------------------------------


def test_run_writes_csv_and_returns_clean_frame(tmp_path: Path) -> None:
    out_path = tmp_path / "weather_clean.csv"
    result = mod.run(FIXTURE_CSV, out_path, write_log=False)

    assert out_path.exists()
    assert len(result) == 200
    assert "PGTM" not in result.columns

    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == len(result)
    assert "DAY_OF_WEEK" in reloaded.columns
