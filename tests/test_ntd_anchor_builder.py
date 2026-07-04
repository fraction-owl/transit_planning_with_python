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
    service_period: str = "Weekday",
) -> dict[str, object]:
    """One tidy pre-aggregation row matching load_raw's output schema."""
    return {
        mod.ROUTE_ID_OUT: route,
        "_service_period": service_period,
        mod._BOARD: board,
        mod._HOURS: hours,
        mod._REVMILES: rev_miles,
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


def test_read_month_workbook_keeps_only_selected_periods(tmp_path: Path) -> None:
    path = _write_workbook(tmp_path / "JULY 2024.xlsx", _one_month_rows())
    out = mod.read_month_workbook("Jul-2024", path, ["Weekday"])
    assert set(out["_service_period"]) == {"Weekday"}
    assert set(out[mod.ROUTE_ID_OUT]) == {"101", "202"}


def test_read_month_workbook_revenue_miles_is_per_day_times_days(tmp_path: Path) -> None:
    path = _write_workbook(tmp_path / "JULY 2024.xlsx", _one_month_rows())
    out = mod.read_month_workbook("Jul-2024", path, ["Weekday"])
    row_101 = out[out[mod.ROUTE_ID_OUT] == "101"].iloc[0]
    assert row_101[mod._REVMILES] == pytest.approx(520.0 * 22.0)


def test_read_month_workbook_missing_file_returns_empty(tmp_path: Path) -> None:
    out = mod.read_month_workbook("Jul-2024", tmp_path / "nope.xlsx", ["Weekday"])
    assert out.empty


def test_read_month_workbook_missing_column_returns_empty(tmp_path: Path) -> None:
    rows = _one_month_rows()
    del rows["REV_MILES"]
    path = _write_workbook(tmp_path / "JULY 2024.xlsx", rows)
    out = mod.read_month_workbook("Jul-2024", path, ["Weekday"])
    assert out.empty


# ---------------------------------------------------------------------------
# build_anchor
# ---------------------------------------------------------------------------


def _raw_two_months() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _raw_row("101", "2024-07", 6278.0, 803.0, 11440.0),
            _raw_row("101", "2024-08", 6000.0, 800.0, 11000.0),
            _raw_row("202", "2024-07", 3801.0, 682.0, 9020.0),
        ]
    )


def test_build_anchor_panel_one_row_per_route_period() -> None:
    result = mod.build_anchor(_raw_two_months(), "panel")
    assert list(result.columns) == [
        mod.ROUTE_ID_OUT,
        mod.PERIOD_OUT,
        mod.BOARDINGS_OUT,
        mod.HOURS_OUT,
        mod.REVMILES_OUT,
    ]
    assert len(result) == 3


def test_build_anchor_panel_sorted_by_route_then_period() -> None:
    result = mod.build_anchor(_raw_two_months(), "panel")
    assert list(result[mod.PERIOD_OUT]) == ["2024-07", "2024-08", "2024-07"]
    assert list(result[mod.ROUTE_ID_OUT]) == ["101", "101", "202"]


def test_build_anchor_cross_section_pools_periods() -> None:
    result = mod.build_anchor(_raw_two_months(), "cross_section")
    assert mod.PERIOD_OUT not in result.columns
    row_101 = result[result[mod.ROUTE_ID_OUT] == "101"].iloc[0]
    # 101 pooled across both months: 6278 + 6000 boardings.
    assert row_101[mod.BOARDINGS_OUT] == pytest.approx(6278.0 + 6000.0)


def test_build_anchor_sums_within_group() -> None:
    raw = pd.DataFrame(
        [
            _raw_row("101", "2024-07", 6278.0, 803.0, 11440.0, "Weekday"),
            _raw_row("101", "2024-07", 575.0, 109.5, 1497.6, "Saturday"),
        ]
    )
    result = mod.build_anchor(raw, "panel")
    row = result.iloc[0]
    assert row[mod.BOARDINGS_OUT] == pytest.approx(6278.0 + 575.0)
    assert row[mod.HOURS_OUT] == pytest.approx(803.0 + 109.5)


def test_build_anchor_rounds_supply_to_two_decimals() -> None:
    raw = pd.DataFrame([_raw_row("101", "2024-07", 100.0, 1.005, 2.346)])
    result = mod.build_anchor(raw, "panel")
    row = result.iloc[0]
    assert row[mod.HOURS_OUT] == pytest.approx(1.0)
    assert row[mod.REVMILES_OUT] == pytest.approx(2.35)


def test_build_anchor_empty_panel_has_schema() -> None:
    result = mod.build_anchor(pd.DataFrame(), "panel")
    assert list(result.columns) == [
        mod.ROUTE_ID_OUT,
        mod.PERIOD_OUT,
        mod.BOARDINGS_OUT,
        mod.HOURS_OUT,
        mod.REVMILES_OUT,
    ]
    assert result.empty


def test_build_anchor_empty_cross_section_has_schema() -> None:
    result = mod.build_anchor(pd.DataFrame(), "cross_section")
    assert list(result.columns) == [
        mod.ROUTE_ID_OUT,
        mod.BOARDINGS_OUT,
        mod.HOURS_OUT,
        mod.REVMILES_OUT,
    ]


# ---------------------------------------------------------------------------
# clean_anchor
# ---------------------------------------------------------------------------


def _anchor_frame(boardings: list[float], hours: list[float] | None = None) -> pd.DataFrame:
    n = len(boardings)
    return pd.DataFrame(
        {
            mod.ROUTE_ID_OUT: [str(i) for i in range(n)],
            mod.BOARDINGS_OUT: boardings,
            mod.HOURS_OUT: hours if hours is not None else [100.0] * n,
            mod.REVMILES_OUT: [500.0] * n,
        }
    )


def test_clean_anchor_drops_nonpositive_when_flag_true() -> None:
    anchor = _anchor_frame([100.0, 0.0, -5.0, 200.0])
    with patch.object(mod, "DROP_NONPOSITIVE_BOARDINGS", True):
        result = mod.clean_anchor(anchor)
    assert list(result[mod.BOARDINGS_OUT]) == [100.0, 200.0]


def test_clean_anchor_keeps_nonpositive_when_flag_false() -> None:
    anchor = _anchor_frame([100.0, 0.0, 200.0])
    with patch.object(mod, "DROP_NONPOSITIVE_BOARDINGS", False):
        result = mod.clean_anchor(anchor)
    assert len(result) == 3


def test_clean_anchor_warns_on_nan_supply(caplog: pytest.LogCaptureFixture) -> None:
    anchor = _anchor_frame([100.0], hours=[float("nan")])
    with caplog.at_level(logging.WARNING):
        mod.clean_anchor(anchor)
    assert "NaN" in caplog.text


# ---------------------------------------------------------------------------
# day_type_filename / "each" service-day mode
# ---------------------------------------------------------------------------


def test_day_type_filename_suffixes_stem() -> None:
    assert mod.day_type_filename("ntd_anchor.csv", "weekday") == "ntd_anchor_weekday.csv"
    assert mod.day_type_filename("anchor.v2.csv", "sunday") == "anchor.v2_sunday.csv"


def test_main_each_writes_one_anchor_per_service_day(tmp_path: Path) -> None:
    """'each' builds all three single-day anchors from one workbook pass."""
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
            "--service-day",
            "each",
        ]
    )

    frames: dict[str, pd.DataFrame] = {}
    for day in ("weekday", "saturday", "sunday"):
        path = out_dir / f"ntd_anchor_{day}.csv"
        assert path.exists(), f"missing {path.name}"
        df = pd.read_csv(path)
        assert set(df[mod.SERVICE_DAY_OUT]) == {day}
        frames[day] = df.assign(**{mod.ROUTE_ID_OUT: df[mod.ROUTE_ID_OUT].astype(str)})

    # Each anchor carries only that day's measures: route 101's weekday boardings/
    # hours stay separate from its Saturday figures, keeping DV and supply on the
    # same day-type basis.
    wk_101 = frames["weekday"].set_index(mod.ROUTE_ID_OUT).loc["101"]
    sat_101 = frames["saturday"].set_index(mod.ROUTE_ID_OUT).loc["101"]
    assert wk_101[mod.BOARDINGS_OUT] == pytest.approx(6278.0)
    assert wk_101[mod.HOURS_OUT] == pytest.approx(803.0)
    assert sat_101[mod.BOARDINGS_OUT] == pytest.approx(575.0)
    assert sat_101[mod.HOURS_OUT] == pytest.approx(109.5)

    assert (out_dir / "ntd_anchor_builder_runlog.txt").exists()


def test_main_single_day_keeps_plain_filename(tmp_path: Path) -> None:
    """A single-day run still writes OUTPUT_FILENAME unchanged, stamped with the day."""
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
            "--service-day",
            "saturday",
        ]
    )

    path = out_dir / "ntd_anchor.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert set(df[mod.SERVICE_DAY_OUT]) == {"saturday"}
    row_101 = df[df[mod.ROUTE_ID_OUT].astype(str) == "101"].iloc[0]
    assert row_101[mod.BOARDINGS_OUT] == pytest.approx(575.0)


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
