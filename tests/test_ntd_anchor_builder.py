"""Tests for scripts/ridership_tools/ntd_anchor_builder.py."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.ridership_tools.ntd_anchor_builder as mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_workbook(path: Path, rows: dict[str, list[object]]) -> Path:
    """Write *rows* to a single-sheet .xlsx at *path* and return it."""
    pd.DataFrame(rows).to_excel(path, index=False, sheet_name="Sheet1")
    return path


def _one_month_rows() -> dict[str, list[object]]:
    """One month of NTD rows: two routes across the three service periods."""
    return {
        "ROUTE_NAME": ["101", "101", "101", "202", "202", "202"],
        "SERVICE_PERIOD": ["Weekday", "Saturday", "Sunday", "Weekday", "Saturday", "Sunday"],
        "MTH_BOARD": [6278.0, 575.0, 556.0, 3801.0, 348.0, 337.0],
        "MTH_REV_HOURS": [803.0, 109.5, 127.7, 682.0, 93.0, 108.5],
        "REV_MILES": [520.0, 374.4, 338.0, 410.0, 295.2, 266.5],
        "DAYS": [22.0, 4.0, 5.0, 22.0, 4.0, 5.0],
    }


def _raw_row(
    route: str,
    period_ym: str,
    board: float,
    hours: float,
    rev_miles: float,
    days: float,
    service_period: str = "Weekday",
) -> dict[str, object]:
    """One tidy pre-aggregation row matching load_raw's output schema."""
    return {
        mod.ROUTE_ID_OUT: route,
        "_service_period": service_period,
        mod._BOARD: board,
        mod._HOURS: hours,
        mod._REVMILES: rev_miles,
        mod._DAYS: days,
        "_period_key": datetime.strptime(period_ym, "%Y-%m").strftime("%b-%Y"),
        "_period_ym": period_ym,
    }


# ---------------------------------------------------------------------------
# parse_month / to_period_ym
# ---------------------------------------------------------------------------


def test_parse_month_returns_month_start() -> None:
    assert mod.parse_month("Jul-2024") == datetime(2024, 7, 1)


def test_parse_month_strips_whitespace() -> None:
    assert mod.parse_month("  Dec-2025 ") == datetime(2025, 12, 1)


def test_to_period_ym_formats_year_month() -> None:
    assert mod.to_period_ym("Jul-2024") == "2024-07"
    assert mod.to_period_ym("Jan-2026") == "2026-01"


# ---------------------------------------------------------------------------
# parse_month_bound
# ---------------------------------------------------------------------------


def test_parse_month_bound_parses_non_blank() -> None:
    assert mod.parse_month_bound("Jul-2024") == datetime(2024, 7, 1)


@pytest.mark.parametrize("value", ["", "   "])
def test_parse_month_bound_blank_is_none(value: str) -> None:
    assert mod.parse_month_bound(value) is None


def test_parse_month_bound_invalid_raises() -> None:
    with pytest.raises(ValueError):
        mod.parse_month_bound("2024-07")


# ---------------------------------------------------------------------------
# parse_filename_period
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("JULY 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Jul-2024"),
        ("NTD RIDERSHIP BY ROUTE-NOVEMBER 2024.xlsx", "Nov-2024"),
        ("NTD RIDERSHIP BY MONTH_DECEMBER 2024.xlsx", "Dec-2024"),
        ("ntd_2025_mar.xlsx", "Mar-2025"),
        ("Apr.2025.finals.xlsx", "Apr-2025"),
    ],
)
def test_parse_filename_period_extracts_key(filename: str, expected: str) -> None:
    assert mod.parse_filename_period(filename) == expected


def test_parse_filename_period_none_when_no_month() -> None:
    assert mod.parse_filename_period("ridership_2024.xlsx") is None


def test_parse_filename_period_none_when_no_year() -> None:
    assert mod.parse_filename_period("july_ridership.xlsx") is None


def test_parse_filename_period_none_when_ambiguous_month() -> None:
    # Two distinct month tokens -> ambiguous -> None.
    assert mod.parse_filename_period("jan_to_march_2025.xlsx") is None


def test_parse_filename_period_none_when_ambiguous_year() -> None:
    assert mod.parse_filename_period("july_2024_vs_2025.xlsx") is None


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


# ---------------------------------------------------------------------------
# normalise_columns / normalise_route / normalise_service_period
# ---------------------------------------------------------------------------


def test_normalise_columns_uppercases_and_underscores() -> None:
    df = pd.DataFrame({"Route Name": [1], "service period": [2]})
    result = mod.normalise_columns(df)
    assert list(result.columns) == ["ROUTE_NAME", "SERVICE_PERIOD"]


def test_normalise_columns_does_not_mutate_original() -> None:
    df = pd.DataFrame({"route name": [1]})
    mod.normalise_columns(df)
    assert "route name" in df.columns


@pytest.mark.parametrize(
    "value,expected",
    [
        ("610.0", "610"),
        (" 42 ", "42"),
        ("101", "101"),
        ("rt 5", "RT5"),
    ],
)
def test_normalise_route(value: object, expected: str) -> None:
    assert mod.normalise_route(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("WEEKDAY", "Weekday"),
        ("Sat", "Saturday"),
        ("sun", "Sunday"),
        ("week day", "Weekday"),
        ("Holiday", "Holiday"),  # unknown labels pass through trimmed
    ],
)
def test_normalise_service_period(value: str, expected: str) -> None:
    assert mod.normalise_service_period(value) == expected


# ---------------------------------------------------------------------------
# column-name helpers
# ---------------------------------------------------------------------------


def test_avg_col_names_daily_average_column() -> None:
    assert mod.avg_col("weekday", mod.BOARDINGS_OUT) == "weekday_avg_ntd_boardings"
    assert mod.avg_col("saturday", mod.HOURS_OUT) == "saturday_avg_revenue_hours"


def test_days_col_names_service_day_count() -> None:
    assert mod.days_col("sunday") == "sunday_service_days"


def test_dv_column_is_weekday_boardings_average() -> None:
    assert mod.DV_COLUMN == "weekday_avg_ntd_boardings"


# ---------------------------------------------------------------------------
# discover_workbooks
# ---------------------------------------------------------------------------


def test_discover_workbooks_maps_period_to_path(tmp_path: Path) -> None:
    _write_workbook(tmp_path / "JULY 2024 NTD.xlsx", _one_month_rows())
    _write_workbook(tmp_path / "AUGUST 2024 NTD.xlsx", _one_month_rows())
    found = mod.discover_workbooks(tmp_path)
    assert set(found) == {"Jul-2024", "Aug-2024"}


def test_discover_workbooks_skips_lock_files(tmp_path: Path) -> None:
    _write_workbook(tmp_path / "JULY 2024 NTD.xlsx", _one_month_rows())
    (tmp_path / "~$JULY 2024 NTD.xlsx").write_text("lock")
    found = mod.discover_workbooks(tmp_path)
    assert set(found) == {"Jul-2024"}


def test_discover_workbooks_warns_on_unparseable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_workbook(tmp_path / "ridership_2024.xlsx", _one_month_rows())
    with caplog.at_level(logging.WARNING):
        found = mod.discover_workbooks(tmp_path)
    assert found == {}
    assert "Could not parse" in caplog.text


def test_discover_workbooks_keeps_first_on_duplicate_period(tmp_path: Path) -> None:
    _write_workbook(tmp_path / "A JULY 2024.xlsx", _one_month_rows())
    _write_workbook(tmp_path / "B JULY 2024.xlsx", _one_month_rows())
    found = mod.discover_workbooks(tmp_path)
    # Sorted-name order keeps "A JULY 2024.xlsx".
    assert found["Jul-2024"].name == "A JULY 2024.xlsx"


# ---------------------------------------------------------------------------
# periods_in_range
# ---------------------------------------------------------------------------


def test_periods_in_range_inclusive_and_sorted() -> None:
    workbooks = {k: Path(f"{k}.xlsx") for k in ["Jun-2024", "Jul-2024", "Aug-2024", "May-2026"]}
    result = mod.periods_in_range(workbooks, datetime(2024, 7, 1), datetime(2024, 8, 1))
    assert result == ["Jul-2024", "Aug-2024"]


def test_periods_in_range_empty_when_none_match() -> None:
    workbooks = {"Jan-2020": Path("Jan-2020.xlsx")}
    assert mod.periods_in_range(workbooks, datetime(2024, 7, 1), datetime(2024, 8, 1)) == []


def _range_workbooks() -> dict[str, Path]:
    return {k: Path(f"{k}.xlsx") for k in ["Jun-2024", "Jul-2024", "Aug-2024", "May-2026"]}


def test_periods_in_range_both_none_returns_all_sorted() -> None:
    result = mod.periods_in_range(_range_workbooks(), None, None)
    assert result == ["Jun-2024", "Jul-2024", "Aug-2024", "May-2026"]


def test_periods_in_range_open_start_clamps_only_end() -> None:
    result = mod.periods_in_range(_range_workbooks(), None, datetime(2024, 7, 1))
    assert result == ["Jun-2024", "Jul-2024"]


def test_periods_in_range_open_end_clamps_only_start() -> None:
    result = mod.periods_in_range(_range_workbooks(), datetime(2024, 8, 1), None)
    assert result == ["Aug-2024", "May-2026"]


# ---------------------------------------------------------------------------
# read_month_workbook
# ---------------------------------------------------------------------------


def test_read_month_workbook_keeps_all_three_service_days(tmp_path: Path) -> None:
    path = _write_workbook(tmp_path / "JULY 2024.xlsx", _one_month_rows())
    out = mod.read_month_workbook("Jul-2024", path)
    assert set(out["_service_period"]) == {"Weekday", "Saturday", "Sunday"}
    assert set(out[mod.ROUTE_ID_OUT]) == {"101", "202"}


def test_read_month_workbook_drops_unknown_service_periods(tmp_path: Path) -> None:
    rows = _one_month_rows()
    rows["SERVICE_PERIOD"] = ["Weekday", "Holiday", "Sunday", "Weekday", "Saturday", "Sunday"]
    path = _write_workbook(tmp_path / "JULY 2024.xlsx", rows)
    out = mod.read_month_workbook("Jul-2024", path)
    assert "Holiday" not in set(out["_service_period"])


def test_read_month_workbook_retains_days_and_monthly_revenue_miles(tmp_path: Path) -> None:
    path = _write_workbook(tmp_path / "JULY 2024.xlsx", _one_month_rows())
    out = mod.read_month_workbook("Jul-2024", path)
    wk_101 = out[(out[mod.ROUTE_ID_OUT] == "101") & (out["_service_period"] == "Weekday")].iloc[0]
    assert wk_101[mod._DAYS] == pytest.approx(22.0)
    assert wk_101[mod._REVMILES] == pytest.approx(520.0 * 22.0)


def test_read_month_workbook_missing_file_returns_empty(tmp_path: Path) -> None:
    out = mod.read_month_workbook("Jul-2024", tmp_path / "nope.xlsx")
    assert out.empty


def test_read_month_workbook_missing_column_returns_empty(tmp_path: Path) -> None:
    rows = _one_month_rows()
    del rows["REV_MILES"]
    path = _write_workbook(tmp_path / "JULY 2024.xlsx", rows)
    out = mod.read_month_workbook("Jul-2024", path)
    assert out.empty


# ---------------------------------------------------------------------------
# build_anchor
# ---------------------------------------------------------------------------


def _raw_three_days_two_months() -> pd.DataFrame:
    """Route 101 across two months x three service days; route 202 weekday-only."""
    return pd.DataFrame(
        [
            _raw_row("101", "2024-07", 6278.0, 803.0, 11440.0, 22.0, "Weekday"),
            _raw_row("101", "2024-07", 575.0, 109.5, 1497.6, 4.0, "Saturday"),
            _raw_row("101", "2024-07", 556.0, 127.7, 1690.0, 5.0, "Sunday"),
            _raw_row("101", "2024-08", 6000.0, 800.0, 11000.0, 21.0, "Weekday"),
            _raw_row("202", "2024-07", 3801.0, 682.0, 9020.0, 22.0, "Weekday"),
        ]
    )


def test_build_anchor_cross_section_wide_schema() -> None:
    result = mod.build_anchor(_raw_three_days_two_months(), "cross_section")
    assert list(result.columns) == mod.output_columns("cross_section")
    # One row per route.
    assert list(result[mod.ROUTE_ID_OUT]) == ["101", "202"]


def test_build_anchor_panel_wide_schema_one_row_per_route_period() -> None:
    result = mod.build_anchor(_raw_three_days_two_months(), "panel")
    assert list(result.columns) == mod.output_columns("panel")
    # 101 appears for 2024-07 and 2024-08; 202 only for 2024-07.
    assert len(result) == 3
    assert list(result[mod.PERIOD_OUT]) == ["2024-07", "2024-08", "2024-07"]


def test_build_anchor_weekday_average_is_day_weighted() -> None:
    """Pooled weekday boardings average = sum(board) / sum(days) across months."""
    result = mod.build_anchor(_raw_three_days_two_months(), "cross_section")
    r101 = result[result[mod.ROUTE_ID_OUT] == "101"].iloc[0]
    expected = (6278.0 + 6000.0) / (22.0 + 21.0)
    assert r101[mod.avg_col("weekday", mod.BOARDINGS_OUT)] == pytest.approx(round(expected, 2))


def test_build_anchor_saturday_average_and_service_days() -> None:
    result = mod.build_anchor(_raw_three_days_two_months(), "cross_section")
    r101 = result[result[mod.ROUTE_ID_OUT] == "101"].iloc[0]
    assert r101[mod.avg_col("saturday", mod.BOARDINGS_OUT)] == pytest.approx(575.0 / 4.0)
    assert r101[mod.days_col("saturday")] == 4


def test_build_anchor_revenue_miles_average_uses_monthly_totals() -> None:
    result = mod.build_anchor(_raw_three_days_two_months(), "cross_section")
    r101 = result[result[mod.ROUTE_ID_OUT] == "101"].iloc[0]
    expected = (11440.0 + 11000.0) / (22.0 + 21.0)
    assert r101[mod.avg_col("weekday", mod.REVMILES_OUT)] == pytest.approx(round(expected, 2))


def test_build_anchor_no_weekend_service_is_nan_with_zero_days() -> None:
    """A weekday-only route has NaN weekend averages and zero weekend service_days."""
    result = mod.build_anchor(_raw_three_days_two_months(), "cross_section")
    r202 = result[result[mod.ROUTE_ID_OUT] == "202"].iloc[0]
    assert pd.isna(r202[mod.avg_col("saturday", mod.BOARDINGS_OUT)])
    assert pd.isna(r202[mod.avg_col("sunday", mod.HOURS_OUT)])
    assert r202[mod.days_col("saturday")] == 0
    assert r202[mod.days_col("sunday")] == 0


def test_build_anchor_empty_panel_has_schema() -> None:
    result = mod.build_anchor(pd.DataFrame(), "panel")
    assert list(result.columns) == mod.output_columns("panel")
    assert result.empty


def test_build_anchor_empty_cross_section_has_schema() -> None:
    result = mod.build_anchor(pd.DataFrame(), "cross_section")
    assert list(result.columns) == mod.output_columns("cross_section")


# ---------------------------------------------------------------------------
# clean_anchor
# ---------------------------------------------------------------------------


def _wide_anchor(
    weekday_boardings: list[float], weekday_hours: list[float] | None = None
) -> pd.DataFrame:
    """A minimal wide cross-section anchor frame with the full column set."""
    n = len(weekday_boardings)
    data: dict[str, list[object]] = {mod.ROUTE_ID_OUT: [str(i) for i in range(n)]}
    for day, _ in mod.SERVICE_DAYS:
        board = weekday_boardings if day == "weekday" else [100.0] * n
        hours = (
            (weekday_hours if weekday_hours is not None else [50.0] * n)
            if day == "weekday"
            else [50.0] * n
        )
        data[mod.avg_col(day, mod.BOARDINGS_OUT)] = board
        data[mod.avg_col(day, mod.HOURS_OUT)] = hours
        data[mod.avg_col(day, mod.REVMILES_OUT)] = [500.0] * n
        data[mod.days_col(day)] = [22] * n
    return pd.DataFrame(data)


def test_clean_anchor_drops_nonpositive_weekday_dv_when_flag_true() -> None:
    anchor = _wide_anchor([100.0, 0.0, -5.0, 200.0])
    with patch.object(mod, "DROP_NONPOSITIVE_BOARDINGS", True):
        result = mod.clean_anchor(anchor)
    assert list(result[mod.DV_COLUMN]) == [100.0, 200.0]


def test_clean_anchor_keeps_nonpositive_when_flag_false() -> None:
    anchor = _wide_anchor([100.0, 0.0, 200.0])
    with patch.object(mod, "DROP_NONPOSITIVE_BOARDINGS", False):
        result = mod.clean_anchor(anchor)
    assert len(result) == 3


def test_clean_anchor_warns_on_nan_weekday_supply(caplog: pytest.LogCaptureFixture) -> None:
    anchor = _wide_anchor([100.0], weekday_hours=[float("nan")])
    with caplog.at_level(logging.WARNING):
        mod.clean_anchor(anchor)
    assert "NaN weekday" in caplog.text


# ---------------------------------------------------------------------------
# main (end-to-end)
# ---------------------------------------------------------------------------


def test_main_writes_wide_anchor_with_all_service_days(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_root.mkdir()
    _write_workbook(data_root / "JULY 2024 NTD.xlsx", _one_month_rows())

    mod.main(
        [
            "--data-root",
            str(data_root),
            "--output-dir",
            str(out_dir),
            "--grain",
            "cross_section",
        ]
    )

    path = out_dir / "ntd_anchor.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == mod.output_columns("cross_section")

    r101 = df[df[mod.ROUTE_ID_OUT].astype(str) == "101"].iloc[0]
    # Weekday boardings average = 6278 / 22 (single month).
    assert r101[mod.DV_COLUMN] == pytest.approx(round(6278.0 / 22.0, 2))
    # Saturday average kept separate on its own day basis.
    assert r101[mod.avg_col("saturday", mod.BOARDINGS_OUT)] == pytest.approx(round(575.0 / 4.0, 2))
    assert r101[mod.days_col("weekday")] == 22

    assert (out_dir / "ntd_anchor_builder_runlog.txt").exists()


def test_main_aborts_when_no_workbooks_in_range(tmp_path: Path) -> None:
    """Empty discovery exits nonzero and writes no anchor (the #96 fail-loud guard)."""
    data_root = tmp_path / "empty"
    out_dir = tmp_path / "out"
    data_root.mkdir()

    with pytest.raises(SystemExit) as exc:
        mod.main(
            [
                "--data-root",
                str(data_root),
                "--output-dir",
                str(out_dir),
                "--grain",
                "cross_section",
            ]
        )
    assert exc.value.code == 1
    assert not (out_dir / "ntd_anchor.csv").exists()


# ---------------------------------------------------------------------------
# extract_config_block / write_run_log
# ---------------------------------------------------------------------------


def test_extract_config_block_reads_from_script() -> None:
    source = Path("scripts/ridership_tools/ntd_anchor_builder.py")
    block = mod.extract_config_block(source)
    assert "DATA_ROOT" in block
    assert "GRAIN" in block


def test_extract_config_block_excludes_marker_lines() -> None:
    source = Path("scripts/ridership_tools/ntd_anchor_builder.py")
    block = mod.extract_config_block(source)
    assert mod.CONFIG_BEGIN_MARKER not in block
    assert mod.CONFIG_END_MARKER not in block


def test_extract_config_block_raises_on_missing_markers(tmp_path: Path) -> None:
    f = tmp_path / "no_markers.py"
    f.write_text("# just a comment\n")
    with pytest.raises(ValueError, match="Config markers not found"):
        mod.extract_config_block(f)


def test_write_run_log_returns_true_and_creates_file(tmp_path: Path) -> None:
    assert mod.write_run_log(tmp_path, ["Grain: panel"]) is True
    assert (tmp_path / "ntd_anchor_builder_runlog.txt").exists()


def test_write_run_log_content_includes_header_summary_and_config(tmp_path: Path) -> None:
    mod.write_run_log(tmp_path, ["Grain:            panel"])
    content = (tmp_path / "ntd_anchor_builder_runlog.txt").read_text()
    assert "NTD ANCHOR BUILD RUN LOG" in content
    assert "Grain:            panel" in content
    assert "DATA_ROOT" in content  # verbatim config block
