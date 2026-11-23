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
# aggregate_monthly
# ---------------------------------------------------------------------------


def _daily(
    start: str, end: str, *, station: str = "X", prcp: float | None, snow: float | None
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "STATION": station,
            "NAME": station,
            "DATE": dates.strftime("%Y-%m-%d"),
            # float("nan"), not None: keep an unobserved column float, as read_csv would.
            "PRCP": float("nan") if prcp is None else prcp,
            "SNOW": float("nan") if snow is None else snow,
            "TMAX": 40.0,
            "TMIN": 30.0,
        }
    )


def test_aggregate_monthly_rejects_multiple_stations() -> None:
    # Two stations' rows for one date would count as two days with precipitation.
    raw = pd.concat(
        [
            _daily("2021-01-01", "2021-01-31", station="A", prcp=0.1, snow=1.0),
            _daily("2021-01-01", "2021-01-31", station="B", prcp=0.1, snow=1.0),
        ]
    )
    with pytest.raises(ValueError, match="single station.*2: A, B"):
        mod.aggregate_monthly(mod.clean_weather(raw), units="standard")


def test_aggregate_monthly_unobserved_month_is_na_not_dry() -> None:
    raw = pd.concat(
        [
            _daily("2021-01-01", "2021-01-31", prcp=0.2, snow=0.5),
            _daily("2021-02-01", "2021-02-28", prcp=None, snow=None),
            _daily("2021-03-01", "2021-03-31", prcp=0.0, snow=0.0),
        ]
    )
    monthly = mod.aggregate_monthly(mod.clean_weather(raw), units="standard").set_index("period")
    assert monthly.loc["2021-01", "days_with_precip"] == 31
    assert monthly.loc["2021-01", "total_snow_in"] == pytest.approx(15.5)
    # February has no PRCP/SNOW readings at all: unknown, not 0 wet days / 0 in.
    assert pd.isna(monthly.loc["2021-02", "days_with_precip"])
    assert pd.isna(monthly.loc["2021-02", "total_snow_in"])
    # March has real zero readings: a genuinely dry month stays 0.
    assert monthly.loc["2021-03", "days_with_precip"] == 0
    assert monthly.loc["2021-03", "total_snow_in"] == 0.0


def test_aggregate_monthly_accepts_long_name_headers(raw_df: pd.DataFrame) -> None:
    short = mod.aggregate_monthly(mod.clean_weather(raw_df), units="standard")
    long = mod.aggregate_monthly(mod.clean_weather(raw_df, use_long_names=True), units="standard")
    pd.testing.assert_frame_equal(short, long)


# ---------------------------------------------------------------------------
# run (end to end)
# ---------------------------------------------------------------------------


def test_run_long_names_with_monthly_output(tmp_path: Path) -> None:
    out_path = tmp_path / "weather_clean.csv"
    monthly_path = tmp_path / "weather_monthly.csv"
    mod.run(
        FIXTURE_CSV,
        out_path,
        monthly_path,
        use_long_names=True,
        write_log=False,
        input_units="standard",
    )

    assert "Precipitation" in pd.read_csv(out_path).columns
    monthly = pd.read_csv(monthly_path)
    assert {"period", "days_with_precip", "total_snow_in"} <= set(monthly.columns)


def test_run_multi_station_monthly_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "two_stations.csv"
    pd.concat(
        [
            _daily("2021-01-01", "2021-01-31", station="A", prcp=0.1, snow=0.0),
            _daily("2021-01-01", "2021-01-31", station="B", prcp=0.1, snow=0.0),
        ]
    ).to_csv(src, index=False)
    out_path = tmp_path / "weather_clean.csv"
    monthly_path = tmp_path / "weather_monthly.csv"
    with pytest.raises(ValueError, match="single station"):
        mod.run(src, out_path, monthly_path, write_log=False, input_units="standard")
    assert not out_path.exists()
    assert not monthly_path.exists()


def test_run_writes_csv_and_returns_clean_frame(tmp_path: Path) -> None:
    out_path = tmp_path / "weather_clean.csv"
    result = mod.run(FIXTURE_CSV, out_path, write_log=False)

    assert out_path.exists()
    assert len(result) == 200
    assert "PGTM" not in result.columns

    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == len(result)
    assert "DAY_OF_WEEK" in reloaded.columns


# ---------------------------------------------------------------------------
# Input units and repeated station-dates
# ---------------------------------------------------------------------------


def _frame(csv: str) -> pd.DataFrame:
    import io

    return mod.clean_weather(pd.read_csv(io.StringIO(csv), dtype={"STATION": "string"}))


def test_aggregate_monthly_converts_metric_to_fahrenheit_and_inches() -> None:
    clean = _frame("STATION,NAME,DATE,TAVG,PRCP,SNOW\nX,N,2026-01-01,20,25.4,50.8\n")
    metric = mod.aggregate_monthly(clean, units="metric").iloc[0]
    assert metric["avg_temp_f"] == pytest.approx(68.0)
    assert metric["max_daily_precip_in"] == pytest.approx(1.0)
    assert metric["total_snow_in"] == pytest.approx(2.0)
    standard = mod.aggregate_monthly(clean, units="Standard").iloc[0]
    assert standard["avg_temp_f"] == pytest.approx(20.0)
    assert standard["max_daily_precip_in"] == pytest.approx(25.4)


@pytest.mark.parametrize("units", [None, "", "imperial"])
def test_aggregate_monthly_requires_input_units(units: str | None) -> None:
    clean = _frame("STATION,NAME,DATE,TAVG\nX,N,2026-01-01,20\n")
    with pytest.raises(ValueError, match="INPUT_UNITS"):
        mod.aggregate_monthly(clean, units=units)


def test_run_monthly_without_units_stops_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "INPUT_UNITS", None)
    out_path = tmp_path / "weather_clean.csv"
    with pytest.raises(ValueError, match="INPUT_UNITS"):
        mod.run(FIXTURE_CSV, out_path, tmp_path / "monthly.csv", write_log=False)
    assert not out_path.exists()


def test_aggregate_monthly_counts_an_exact_repeat_once() -> None:
    clean = _frame(
        "STATION,NAME,DATE,PRCP,SNOW\nX,N,2026-01-01,0.5,2\nX,N,2026-01-01,0.5,2\n"
        "X,N,2026-01-02,0,0\n"
    )
    row = mod.aggregate_monthly(clean, units="standard").iloc[0]
    assert row["days_with_precip"] == 1
    assert row["total_snow_in"] == pytest.approx(2.0)


def test_aggregate_monthly_rejects_conflicting_repeat() -> None:
    clean = _frame("STATION,NAME,DATE,PRCP,SNOW\nX,N,2026-01-01,0.5,2\nX,N,2026-01-01,0.5,3\n")
    with pytest.raises(ValueError, match="1 date\\(s\\) appear more than once.*2026-01-01"):
        mod.aggregate_monthly(clean, units="standard")
