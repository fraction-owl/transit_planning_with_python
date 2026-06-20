"""Subset and summarize NTD monthly ridership for selected routes (config-driven).

Workbooks are discovered by scanning DATA_ROOT (matching ntd_anchor_builder.py):
each file is assumed to hold a single worksheet, and its month/year are parsed
from the filename (full or 3-letter month, 4-digit year, in any order or
separator). No hand-maintained catalogue is required. The script filters to
configured routes and exports per-route monthly summaries and trend plots:
    - Weekday/Saturday/Sunday monthly totals
    - Weekday/Saturday/Sunday per-day averages by month

It logs warnings when:
    - A month in the requested range has no discovered workbook
    - A workbook file is missing
    - A route-month-service period is missing
    - Ridership is 0 while DAYS > 0 (possible localized data outage)

Optionally prompts users for manual fixes (enter corrected MTH_BOARD and/or DAYS).

Outputs per route (folder: OUTPUT_ROOT/route_<ROUTE>/):
    - monthly_long.csv (month x service_period rows)
    - monthly_wide.csv (one row per month; totals + averages columns)
    - outage_flags.csv
    - yoy_percent_change.csv (per-day-average YoY metrics; when enabled)
    - plots/monthly_totals.png
    - plots/daily_averages.png
    - plots/yoy_percent_change.png (small multiples, one panel per service
      period: raw YoY points, a rolling-mean YoY line, and an optional
      systemwide rolling line + comparison band; when enabled). The weekday
      series can optionally be computed on holiday-free days (see HOLIDAYS).
"""

from __future__ import annotations

import calendar
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Sentinel markers used by extract_config_block / write_run_log to identify
# the configuration block within this file's source. Each string must appear
# exactly once in this file as a stand-alone comment line (other than these
# constant definitions themselves). Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

DATA_ROOT: Final[Path] = Path(r"Path\To\Your\Input_Folder")  # monthly workbooks
OUTPUT_ROOT: Final[Path] = Path(r"Path\To\Your\Output_Folder")

# Glob used to discover workbooks under DATA_ROOT (non-recursive). Switch to
# DATA_ROOT.rglob if the monthly files live in dated subfolders.
WORKBOOK_GLOB: Final[str] = "*.xlsx"

# Requested overall date range (inclusive month starts), as "Mon-YYYY".
#
# Leave either bound BLANK ("") to make it dynamic: an empty START_MONTH starts
# at the earliest workbook on disk, an empty END_MONTH ends at the latest, and
# leaving both blank uses whatever is in DATA_ROOT with no clamping. Set a bound
# explicitly whenever you need control over the window (e.g. to flag a month
# whose workbook has not landed yet).
START_MONTH: Final[str] = ""
END_MONTH: Final[str] = ""

# Route subset.
ROUTES: Final[list[str]] = ["101", "202", "303"]

# Service periods expected.
SERVICE_PERIODS: Final[list[str]] = ["Weekday", "Saturday", "Sunday"]

# Prompt users to manually override missing/zero values.
PROMPT_FOR_FIXES: Final[bool] = False

# -----------------------------------------------------------------------------
#  Percent-change ("trend") charts
# -----------------------------------------------------------------------------

# Export the per-route percent-change chart in addition to the totals/averages
# plots. The chart is a stack of small multiples, one panel per service period
# (Weekday / Saturday / Sunday), so routes that run only part of the week simply
# show fewer panels. Within each panel:
#   - points: raw month-over-same-month-last-year (YoY) % change
#   - line:   rolling mean of that YoY series (see ROLLING_WINDOW_MONTHS); it
#             starts equal to the first YoY point and smooths as data accrues
#   - optionally a dashed systemwide rolling line + shaded band for comparison
# All percent changes are computed on per-day averages (boardings / service
# days), which controls for differing service-day counts month to month. Months
# carrying any outage flag are excluded from the YoY math.
EXPORT_CHANGE_CHARTS: Final[bool] = True

# Trailing window (in months) over which the YoY series is averaged to draw the
# rolling line. MIN_ROLLING_MONTHS is the rolling mean's min_periods: with the
# default of 1 the window expands from a single month up to ROLLING_WINDOW_MONTHS
# as data accrues (raise it to suppress the noisy earliest points).
ROLLING_WINDOW_MONTHS: Final[int] = 12
MIN_ROLLING_MONTHS: Final[int] = 1

# Overlay a systemwide rolling YoY line (aggregated over EVERY route discovered
# on disk, not just ROUTES) as a point of comparison in each panel.
INCLUDE_SYSTEMWIDE_COMPARISON: Final[bool] = True

# Half-width, in percentage points, of the shaded band drawn around the
# systemwide rolling line. A route trending outside systemwide +/- this band is
# notably over/under-performing. Set to 0 to draw the line with no band.
SYSTEMWIDE_BAND_PCT: Final[float] = 10.0

# Routes whose normalised name matches this (case-insensitive) regex are dropped
# before the systemwide aggregation, guarding against subtotal / grand-total /
# non-fixed-route rows that would otherwise double-count. The per-route subset is
# unaffected (it is filtered to ROUTES). Set to "" to disable the guard.
SYSTEMWIDE_EXCLUDE_ROUTES_REGEX: Final[str] = r"^(TOTAL|TOTALS|SYSTEM|SYSTEMWIDE|ALL|GRANDTOTAL)$"

# Weekday holidays to subtract from the WEEKDAY per-day average before its YoY is
# computed (mirrors ntd_monthly_summary.py). NTD counts a holiday weekday as an
# ordinary weekday service day even though ridership is atypically low, biasing
# the weekday average; removing those days corrects it. Year-over-year already
# cancels holidays that fall in the same month both years, so this mainly matters
# when a holiday's weekday count differs across years (a fixed-date holiday
# landing on a weekend one year, Election Day in even years, etc.). Only holidays
# falling on a weekday count; Saturday/Sunday panels are never adjusted. Leave
# the list EMPTY to disable, and populate it across EVERY year you load, e.g.:
#     datetime(2024, 1, 1),    # New Year's Day
#     datetime(2024, 7, 4),    # Independence Day
HOLIDAYS: Final[list[datetime]] = []

# Logging level (INFO recommended; DEBUG if troubleshooting).
LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so the analyst is never
# left with an output directory that lacks a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Required columns for this simplified workflow.
REQUIRED_COLS: Final[list[str]] = ["ROUTE_NAME", "SERVICE_PERIOD", "MTH_BOARD", "DAYS"]

# Month-token lookup (full names and 3-letter abbreviations -> month number),
# plus a regex that finds any of them as a whole word, longest-first.
_MONTH_LOOKUP: Final[dict[str, int]] = {
    name.lower(): num
    for num in range(1, 13)
    for name in (calendar.month_name[num], calendar.month_abbr[num])
}
_MONTH_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_LOOKUP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_YEAR_RE: Final[re.Pattern[str]] = re.compile(r"\b(20\d{2})\b")

PLOT_STYLE: Final[dict[str, Any]] = {
    "figsize": (10, 5),
    "marker": "o",
    "linestyle": "-",
    "grid": True,
    "rotation": 45,
    "dpi": 150,
    # Per-panel height for the stacked small-multiples change chart.
    "panel_height": 3.0,
}

# Per-service-period plot styling for the change chart (marker + line colour).
_SP_STYLE: Final[dict[str, dict[str, str]]] = {
    "Weekday": {"marker": "o", "color": "tab:blue"},
    "Saturday": {"marker": "s", "color": "tab:orange"},
    "Sunday": {"marker": "^", "color": "tab:green"},
}

# =============================================================================
# HELPERS
# =============================================================================


def parse_month(value: str) -> datetime:
    """Parse 'Mon-YYYY' into a month-start datetime."""
    dt = datetime.strptime(value.strip(), "%b-%Y")
    return datetime(dt.year, dt.month, 1)


def format_month(dt: datetime) -> str:
    """Format a month-start datetime as 'Mon-YYYY'."""
    return dt.strftime("%b-%Y")


def month_range(start: datetime, end: datetime) -> list[datetime]:
    """Return inclusive list of month-start datetimes from start to end."""
    if start > end:
        return []
    months = pd.date_range(start=start, end=end, freq="MS").to_pydatetime().tolist()
    return [datetime(m.year, m.month, 1) for m in months]


def safe_float(value: Any) -> float | None:
    """Return float if value looks numeric; else None."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case, trim, and replace spaces with underscores in columns."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    return out


def normalise_route(value: Any) -> str:
    """Normalize route values to a compact token (e.g., 610.0 -> '610')."""
    s = str(value).strip().upper().replace(" ", "")
    s = re.sub(r"\.0$", "", s)
    return s


def normalise_service_period(value: Any) -> str:
    """Normalize service period values to {Weekday, Saturday, Sunday} where possible."""
    s = str(value).strip().lower()
    mapping = {
        "weekday": "Weekday",
        "week day": "Weekday",
        "wkday": "Weekday",
        "wkdy": "Weekday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    return mapping.get(s, str(value).strip())


def parse_month_bound(value: str) -> datetime | None:
    """Parse a range bound, treating a blank string as "unbounded" (``None``).

    Lets START_MONTH / END_MONTH be left empty to mean "use whatever is on disk"
    on that side of the range. A non-blank value is parsed as ``Mon-YYYY``.
    """
    return parse_month(value) if value.strip() else None


def parse_filename_period(filename: str) -> str | None:
    """Extract a ``Mon-YYYY`` period key from a workbook filename.

    Looks for exactly one month token (full or 3-letter, any case) and exactly
    one 4-digit year anywhere in the name, in any order or separator. Returns
    ``None`` if either is missing or the name is ambiguous (multiple candidate
    months or years), so the caller can warn and skip rather than guess.

    Args:
        filename: The workbook's filename (extension is ignored).

    Returns:
        A ``Mon-YYYY`` key (e.g. ``"Jul-2024"``), or ``None``.
    """
    # Underscores are regex word characters, so a name like "MONTH_DECEMBER"
    # or "_2024" hides the token from \b boundaries. Collapse every run of
    # non-alphanumeric characters to a single space before matching.
    stem = re.sub(r"[^0-9A-Za-z]+", " ", Path(filename).stem)
    months = _MONTH_RE.findall(stem)
    years = _YEAR_RE.findall(stem)
    if len(months) != 1 or len(years) != 1:
        return None
    month_num = _MONTH_LOOKUP[months[0].lower()]
    return datetime(int(years[0]), month_num, 1).strftime("%b-%Y")


def discover_workbooks(data_root: Path) -> dict[str, Path]:
    """Scan ``data_root`` and map each ``Mon-YYYY`` period to its workbook.

    Skips Excel lock files (``~$*``), warns on filenames whose month/year cannot
    be parsed, and warns on (then ignores) any second file that maps to a period
    already claimed, keeping the first by sorted name.

    Args:
        data_root: Folder to scan using :data:`WORKBOOK_GLOB`.

    Returns:
        Mapping of period key -> workbook path.
    """
    found: dict[str, Path] = {}
    for path in sorted(data_root.glob(WORKBOOK_GLOB)):
        if path.name.startswith("~$"):
            continue
        period = parse_filename_period(path.name)
        if period is None:
            logging.warning("Could not parse month/year from filename: %s", path.name)
            continue
        if period in found:
            logging.warning(
                "Duplicate workbook for %s: keeping '%s', ignoring '%s'.",
                period,
                found[period].name,
                path.name,
            )
            continue
        found[period] = path
    return found


def periods_in_range(
    workbooks: dict[str, Path],
    start: datetime | None,
    end: datetime | None,
) -> list[str]:
    """Return discovered period keys within ``[start, end]``, chronologically.

    A ``None`` bound is treated as open-ended on that side, so passing both as
    ``None`` returns every discovered period.
    """
    return sorted(
        (
            k
            for k in workbooks
            if (start is None or parse_month(k) >= start) and (end is None or parse_month(k) <= end)
        ),
        key=parse_month,
    )


# =============================================================================
# IO + TRANSFORM
# =============================================================================


def read_month_workbook(period: str, path: Path) -> pd.DataFrame:
    """Read one workbook's only sheet and return normalized rows (unfiltered)."""
    if not path.exists():
        logging.warning("Workbook missing on disk: %s (period=%s)", path, period)
        return pd.DataFrame()

    converters: dict[str, Any] = {"MTH_BOARD": safe_float, "DAYS": safe_float}
    try:
        # sheet_name=0 reads the single worksheet regardless of its name.
        df = pd.read_excel(path, sheet_name=0, converters=converters)
    except Exception:
        logging.exception("Failed to read workbook: %s (period=%s)", path, period)
        return pd.DataFrame()

    df = normalise_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logging.warning(
            "Workbook missing required columns (period=%s file=%s): %s",
            period,
            path.name,
            ", ".join(missing),
        )
        return pd.DataFrame()

    out = df.copy()
    out["ROUTE_NAME"] = out["ROUTE_NAME"].apply(normalise_route)
    out["SERVICE_PERIOD"] = out["SERVICE_PERIOD"].apply(normalise_service_period)

    out["period"] = period
    out["period_dt"] = parse_month(period)
    return out


def load_raw(workbooks: dict[str, Path], periods: list[str]) -> pd.DataFrame:
    """Load all months in `periods`, filtered to configured service periods.

    Every route is retained (not just ROUTES) so the systemwide comparison can
    be derived from the same read; the route subset is filtered downstream.
    """
    frames: list[pd.DataFrame] = []
    sp_set = set(SERVICE_PERIODS)

    for period in periods:
        path = workbooks.get(period)
        if path is None:
            continue
        df = read_month_workbook(period, path)
        if df.empty:
            continue

        df = df[df["SERVICE_PERIOD"].isin(sp_set)].copy()

        frames.append(df)
        logging.info("Loaded %s: %d rows kept", period, len(df))

    if not frames:
        return pd.DataFrame(columns=[*REQUIRED_COLS, "period", "period_dt"])

    return pd.concat(frames, ignore_index=True)


def aggregate_monthly_long(
    raw: pd.DataFrame,
    expected_months: list[datetime],
) -> tuple[pd.DataFrame, set[tuple[str, datetime, str]]]:
    """Aggregate raw rows to monthly-long and reindex to a full grid.

    Returns:
        monthly_long: one row per (route, period_dt, service_period)
        observed_keys: set of (route, period_dt, service_period) observed in raw aggregates
    """
    if raw.empty:
        idx = pd.MultiIndex.from_product(
            [ROUTES, expected_months, SERVICE_PERIODS],
            names=["route", "period_dt", "service_period"],
        )
        monthly_long = idx.to_frame(index=False)
        monthly_long["period"] = monthly_long["period_dt"].apply(format_month)
        monthly_long["mth_board"] = pd.NA
        monthly_long["days"] = pd.NA
        monthly_long["daily_avg"] = pd.NA
        return monthly_long, set()

    agg = (
        raw.groupby(["ROUTE_NAME", "period_dt", "SERVICE_PERIOD"], as_index=False)
        .agg({"MTH_BOARD": "sum", "DAYS": "sum"})
        .rename(
            columns={
                "ROUTE_NAME": "route",
                "SERVICE_PERIOD": "service_period",
                "MTH_BOARD": "mth_board",
                "DAYS": "days",
            }
        )
    )

    observed_keys = set(
        (str(r), dt, str(sp))
        for r, dt, sp in zip(agg["route"], agg["period_dt"], agg["service_period"])
    )

    idx = pd.MultiIndex.from_product(
        [ROUTES, expected_months, SERVICE_PERIODS],
        names=["route", "period_dt", "service_period"],
    )
    monthly_long = (
        agg.set_index(["route", "period_dt", "service_period"]).reindex(idx).reset_index()
    )
    monthly_long["period"] = monthly_long["period_dt"].apply(format_month)

    monthly_long["daily_avg"] = pd.NA
    has_days = pd.to_numeric(monthly_long["days"], errors="coerce") > 0
    monthly_long.loc[has_days, "daily_avg"] = pd.to_numeric(
        monthly_long.loc[has_days, "mth_board"], errors="coerce"
    ) / pd.to_numeric(monthly_long.loc[has_days, "days"], errors="coerce")

    return monthly_long, observed_keys


def flag_outages(
    monthly_long: pd.DataFrame,
    expected_months: list[datetime],
    observed_keys: set[tuple[str, datetime, str]],
    available_months: set[datetime],
) -> pd.DataFrame:
    """Flag missing/zero/suspicious values, and log warnings.

    Args:
        monthly_long: The reindexed monthly-long grid.
        expected_months: Every month-start in the requested range.
        observed_keys: (route, period_dt, service_period) tuples seen in the data.
        available_months: Month-starts for which a workbook was discovered on
            disk; an expected month not in this set is flagged as having no
            workbook.
    """
    flags: list[dict[str, Any]] = []

    expected_month_set = set(expected_months)

    # Expected months with no discovered workbook on disk.
    for m in sorted(expected_month_set):
        if m not in available_months:
            logging.warning("No workbook discovered for expected month: %s", format_month(m))

    # Row-level flags
    for _, r in monthly_long.iterrows():
        route = str(r["route"])
        period_dt = r["period_dt"]
        period = str(r["period"])
        sp = str(r["service_period"])

        key = (route, period_dt, sp)
        was_observed = key in observed_keys

        b_num = pd.to_numeric(r["mth_board"], errors="coerce")
        d_num = pd.to_numeric(r["days"], errors="coerce")

        if not was_observed:
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "missing_service_period",
                    "mth_board": pd.NA,
                    "days": pd.NA,
                }
            )
            continue

        if pd.isna(b_num) and pd.isna(d_num):
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "missing_entry",
                    "mth_board": pd.NA,
                    "days": pd.NA,
                }
            )
            continue

        if not pd.isna(d_num) and d_num == 0:
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "zero_days",
                    "mth_board": b_num,
                    "days": d_num,
                }
            )

        if not pd.isna(b_num) and not pd.isna(d_num) and b_num == 0 and d_num > 0:
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "zero_ridership_nonzero_days",
                    "mth_board": b_num,
                    "days": d_num,
                }
            )

    if not flags:
        out = pd.DataFrame(
            columns=["route", "period", "service_period", "flag", "mth_board", "days"]
        )
    else:
        out = pd.DataFrame(flags).drop_duplicates(ignore_index=True)

    for _, f in out.iterrows():
        logging.warning(
            "Flag: route=%s period=%s service_period=%s flag=%s boards=%s days=%s",
            f["route"],
            f["period"],
            f["service_period"],
            f["flag"],
            f["mth_board"],
            f["days"],
        )

    return out


def apply_manual_fixes(monthly_long: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Prompt user to manually override missing/zero values (if enabled)."""
    if not PROMPT_FOR_FIXES or flags.empty:
        return monthly_long

    updated = monthly_long.copy()
    stop = False

    def prompt_float(msg: str) -> float | None | str:
        raw = input(msg).strip()
        if raw.lower() == "q":
            return "q"
        if raw == "":
            return None
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            logging.warning("Invalid numeric input %r; keeping existing value.", raw)
            return None

    fixable = flags[
        flags["flag"].isin(
            {"missing_service_period", "missing_entry", "zero_ridership_nonzero_days"}
        )
    ].copy()
    fixable = fixable.sort_values(["route", "period", "service_period", "flag"])

    for _, f in fixable.iterrows():
        if stop:
            break

        route = str(f["route"])
        period = str(f["period"])
        sp = str(f["service_period"])
        flag = str(f["flag"])

        mask = (
            (updated["route"] == route)
            & (updated["period"] == period)
            & (updated["service_period"] == sp)
        )
        if mask.sum() != 1:
            logging.warning(
                "Fix skipped; could not uniquely locate row: %s %s %s", route, period, sp
            )
            continue

        cur_days = updated.loc[mask, "days"].iloc[0]

        logging.warning("Interactive fix candidate: %s %s %s (%s)", route, period, sp, flag)

        if flag in {"missing_service_period", "missing_entry"}:
            b_in = prompt_float(
                f"[{route} | {period} | {sp}] Missing. Enter MTH_BOARD (Enter=skip, q=quit): "
            )
            if b_in == "q":
                stop = True
                break
            d_in = prompt_float(
                f"[{route} | {period} | {sp}] Missing. Enter DAYS (Enter=skip, q=quit): "
            )
            if d_in == "q":
                stop = True
                break

            if b_in is not None:
                updated.loc[mask, "mth_board"] = b_in
            if d_in is not None:
                updated.loc[mask, "days"] = d_in

        elif flag == "zero_ridership_nonzero_days":
            b_in = prompt_float(
                f"[{route} | {period} | {sp}] boards=0 days={cur_days}. Enter corrected MTH_BOARD "
                "(Enter=keep 0, q=quit): "
            )
            if b_in == "q":
                stop = True
                break
            if b_in is not None:
                updated.loc[mask, "mth_board"] = b_in

    # Recompute daily_avg after edits
    updated["daily_avg"] = pd.NA
    has_days = pd.to_numeric(updated["days"], errors="coerce") > 0
    updated.loc[has_days, "daily_avg"] = pd.to_numeric(
        updated.loc[has_days, "mth_board"], errors="coerce"
    ) / pd.to_numeric(updated.loc[has_days, "days"], errors="coerce")

    return updated


def to_wide(monthly_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot monthly-long into a single row per month with totals and averages."""
    base = monthly_long[
        ["route", "period_dt", "period", "service_period", "mth_board", "daily_avg"]
    ].copy()

    totals = base.pivot_table(
        index=["route", "period_dt", "period"],
        columns="service_period",
        values="mth_board",
        aggfunc="first",
    )
    avgs = base.pivot_table(
        index=["route", "period_dt", "period"],
        columns="service_period",
        values="daily_avg",
        aggfunc="first",
    )

    totals.columns = [f"{c.lower()}_total" for c in totals.columns]
    avgs.columns = [f"{c.lower()}_avg" for c in avgs.columns]

    out = pd.concat([totals, avgs], axis=1).reset_index()
    out = out.sort_values(["route", "period_dt"], ignore_index=True)
    return out


# =============================================================================
# PERCENT-CHANGE ANALYTICS
# =============================================================================


def weekday_holiday_counts(holidays: list[datetime]) -> dict[str, int]:
    """Count weekday holidays per ``Mon-YYYY`` month (matches ntd_monthly_summary).

    Holidays falling on a Saturday or Sunday are ignored, since the adjustment
    only applies to the weekday service-day denominator.
    """
    counts: dict[str, int] = {}
    for holiday in holidays:
        if holiday.weekday() >= 5:  # Saturday=5, Sunday=6
            continue
        period = holiday.strftime("%b-%Y")
        counts[period] = counts.get(period, 0) + 1
    return counts


def weekday_holiday_free_level(df_sp: pd.DataFrame, holiday_counts: dict[str, int]) -> pd.Series:
    """Weekday per-day average over holiday-free days, indexed by ``period_dt``.

    ``df_sp`` is one route's Weekday rows (indexed by ``period_dt``). Boardings
    are divided by ``days - weekday_holidays_in_month`` instead of raw days. The
    result is NaN wherever the cleaned ``daily_avg`` is NaN (a missing or
    outage-flagged month) so the holiday adjustment never resurrects an excluded
    month, and wherever the holiday-free day count is non-positive.
    """
    board = pd.to_numeric(df_sp["mth_board"], errors="coerce")
    days = pd.to_numeric(df_sp["days"], errors="coerce")
    holidays = df_sp["period"].map(holiday_counts).fillna(0).astype(float)
    adj_days = days - holidays
    level = board / adj_days.where(adj_days > 0)
    return level.where(pd.to_numeric(df_sp["daily_avg"], errors="coerce").notna())


def clean_daily_avg(monthly_long: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Return monthly_long with daily_avg blanked on any outage-flagged month.

    The change math runs on per-day averages, so every (route, period, service
    period) carrying a flag - missing, zero-days, or zero-ridership-with-days -
    has its daily_avg set to NaN here so it is excluded from the YoY series and
    its rolling mean.
    """
    out = monthly_long.copy()
    if flags.empty:
        return out
    flagged = set(
        zip(
            flags["route"].astype(str),
            flags["period"].astype(str),
            flags["service_period"].astype(str),
        )
    )
    keys = zip(
        out["route"].astype(str), out["period"].astype(str), out["service_period"].astype(str)
    )
    mask = pd.Series([k in flagged for k in keys], index=out.index)
    out.loc[mask, "daily_avg"] = pd.NA
    return out


def monthly_yoy_pct(level: pd.Series, expected_months: list[datetime]) -> pd.Series:
    """Year-over-year % change of a per-day series on the continuous month grid.

    Reindexing onto the full grid makes a 12-row shift exactly 12 calendar
    months, so a missing month yields NaN on both sides rather than comparing
    adjacent observations.
    """
    idx = pd.DatetimeIndex(expected_months)
    s = pd.to_numeric(pd.Series(level), errors="coerce").reindex(idx)
    return (s / s.shift(12) - 1.0) * 100.0


def rolling_mean_yoy(yoy: pd.Series) -> pd.Series:
    """Trailing rolling mean of a YoY series (the smoothed line on the chart).

    With MIN_ROLLING_MONTHS == 1 the window expands from a single month up to
    ROLLING_WINDOW_MONTHS, so the line starts equal to the first YoY point and
    smooths as more months become available. NaN (missing/outage) months are
    skipped within each window.
    """
    return yoy.rolling(ROLLING_WINDOW_MONTHS, min_periods=MIN_ROLLING_MONTHS).mean()


def compute_systemwide_perday_by_sp(
    raw_all: pd.DataFrame, holiday_counts: dict[str, int]
) -> dict[str, pd.Series]:
    """Per-service-period systemwide per-day average over all routes on disk.

    For each service period, sums boardings and service days across every route
    (after dropping subtotal/aggregate rows per SYSTEMWIDE_EXCLUDE_ROUTES_REGEX)
    and divides. When ``holiday_counts`` is non-empty, each Weekday row's days are
    reduced by that month's weekday holidays before summing, so the systemwide
    Weekday baseline matches the holiday-free route series. Returns
    ``{service_period: Series indexed by month-start}``.
    """
    if raw_all.empty:
        return {}
    d = raw_all.copy()
    if SYSTEMWIDE_EXCLUDE_ROUTES_REGEX.strip():
        pat = re.compile(SYSTEMWIDE_EXCLUDE_ROUTES_REGEX, re.IGNORECASE)
        excluded = d["ROUTE_NAME"].astype(str).str.fullmatch(pat)
        if excluded.any():
            logging.info(
                "Systemwide guard excluded %d row(s) for routes: %s",
                int(excluded.sum()),
                ", ".join(sorted(d.loc[excluded, "ROUTE_NAME"].astype(str).unique())),
            )
        d = d[~excluded]
    d["_b"] = pd.to_numeric(d["MTH_BOARD"], errors="coerce")
    d["_d"] = pd.to_numeric(d["DAYS"], errors="coerce")
    if holiday_counts:
        is_weekday = d["SERVICE_PERIOD"] == "Weekday"
        holidays = d["period"].map(holiday_counts).fillna(0).astype(float)
        d.loc[is_weekday, "_d"] = d.loc[is_weekday, "_d"] - holidays[is_weekday]
    d = d[d["_b"].notna() & d["_d"].notna() & (d["_d"] > 0)]
    if d.empty:
        return {}
    g = d.groupby(["SERVICE_PERIOD", "period_dt"]).agg(b=("_b", "sum"), dd=("_d", "sum"))
    perday = g["b"] / g["dd"]
    return {sp: perday.xs(sp, level="SERVICE_PERIOD") for sp in perday.index.levels[0]}


def compute_route_change(
    route: str,
    cleaned_long: pd.DataFrame,
    systemwide_by_sp: dict[str, pd.Series] | None,
    expected_months: list[datetime],
    holiday_counts: dict[str, int],
) -> pd.DataFrame:
    """Assemble the per-route, per-service-period percent-change table.

    Columns per service period <sp>:
        <sp>_yoy_pct          raw monthly YoY % of that period's per-day average
        <sp>_yoy_rolling_pct  rolling mean of that YoY series (the route line)
        <sp>_sys_rolling_pct  systemwide rolling YoY % (when supplied)

    When ``holiday_counts`` is non-empty the Weekday level uses holiday-free days
    (see :func:`weekday_holiday_free_level`); other periods always use daily_avg.
    """
    idx = pd.DatetimeIndex(expected_months)
    out = pd.DataFrame(index=idx)

    sub = cleaned_long[cleaned_long["route"] == route]
    for sp in SERVICE_PERIODS:
        spl = sp.lower()
        df_sp = sub[sub["service_period"] == sp].set_index("period_dt")
        if sp == "Weekday" and holiday_counts:
            level = weekday_holiday_free_level(df_sp, holiday_counts)
        else:
            level = df_sp["daily_avg"]
        yoy = monthly_yoy_pct(level, expected_months)
        out[f"{spl}_yoy_pct"] = yoy
        out[f"{spl}_yoy_rolling_pct"] = rolling_mean_yoy(yoy)

        if systemwide_by_sp and sp in systemwide_by_sp:
            sys_yoy = monthly_yoy_pct(systemwide_by_sp[sp], expected_months)
            out[f"{spl}_sys_rolling_pct"] = rolling_mean_yoy(sys_yoy)

    out = out.reset_index(names="period_dt")
    out.insert(0, "route", route)
    out.insert(2, "period", out["period_dt"].apply(format_month))
    return out


# =============================================================================
# PLOTTING + EXPORT
# =============================================================================


def plot_route_totals(route_dir: Path, route: str, wide: pd.DataFrame) -> None:
    """Plot monthly totals for a single route."""
    df = wide[wide["route"] == route].copy()
    if df.empty:
        return

    out_path = route_dir / "plots" / "monthly_totals.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=PLOT_STYLE["figsize"])
    x = df["period_dt"]

    for col in ["weekday_total", "saturday_total", "sunday_total"]:
        if col in df.columns:
            plt.plot(
                x,
                df[col],
                marker=PLOT_STYLE["marker"],
                linestyle=PLOT_STYLE["linestyle"],
                label=col,
            )

    plt.title(f"Monthly Ridership Totals – Route {route}")
    plt.xlabel("Month")
    plt.ylabel("Boardings (Monthly Total)")
    plt.grid(PLOT_STYLE["grid"])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=PLOT_STYLE["rotation"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_STYLE["dpi"])
    plt.close()


def plot_route_avgs(route_dir: Path, route: str, wide: pd.DataFrame) -> None:
    """Plot per-day averages for a single route."""
    df = wide[wide["route"] == route].copy()
    if df.empty:
        return

    out_path = route_dir / "plots" / "daily_averages.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=PLOT_STYLE["figsize"])
    x = df["period_dt"]

    for col in ["weekday_avg", "saturday_avg", "sunday_avg"]:
        if col in df.columns:
            plt.plot(
                x,
                df[col],
                marker=PLOT_STYLE["marker"],
                linestyle=PLOT_STYLE["linestyle"],
                label=col,
            )

    plt.title(f"Per-Day Ridership Averages – Route {route}")
    plt.xlabel("Month")
    plt.ylabel("Boardings (Per Day Average)")
    plt.grid(PLOT_STYLE["grid"])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=PLOT_STYLE["rotation"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_STYLE["dpi"])
    plt.close()


def _panels_with_data(change: pd.DataFrame) -> list[str]:
    """Service periods that have any YoY value to draw (others are skipped)."""
    panels: list[str] = []
    for sp in SERVICE_PERIODS:
        candidates = (f"{sp.lower()}_yoy_pct", f"{sp.lower()}_yoy_rolling_pct")
        cols = [c for c in candidates if c in change]
        if cols and change[cols].notna().to_numpy().sum() > 0:
            panels.append(sp)
    return panels


def plot_route_change(route_dir: Path, route: str, change: pd.DataFrame) -> None:
    """Plot the per-route YoY percent-change chart as stacked small multiples.

    One panel per service period that has data (so partial-week routes show only
    the panels they run), sharing the month axis. Each panel draws the raw YoY
    points, the rolling-mean YoY line, and - when present - the systemwide
    rolling line with a +/- SYSTEMWIDE_BAND_PCT shaded band. Skips quietly when
    there is nothing to draw (e.g. under a year of data).
    """
    panels = _panels_with_data(change)
    if not panels:
        logging.info("No YoY data to plot for route %s; skipping change chart.", route)
        return

    out_path = route_dir / "plots" / "yoy_percent_change.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = change["period_dt"]
    width = PLOT_STYLE["figsize"][0]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(width, PLOT_STYLE["panel_height"] * len(panels) + 1.0),
        sharex=True,
        squeeze=False,
    )

    for ax, sp in zip(axes[:, 0], panels):
        spl = sp.lower()
        style = _SP_STYLE.get(sp, {"marker": "o", "color": "tab:blue"})
        ax.axhline(0, color="0.6", linewidth=1, zorder=1)

        # Systemwide rolling line + comparison band.
        sys_col = f"{spl}_sys_rolling_pct"
        if sys_col in change.columns and change[sys_col].notna().any():
            sys = change[sys_col]
            if SYSTEMWIDE_BAND_PCT > 0:
                ax.fill_between(
                    x,
                    sys - SYSTEMWIDE_BAND_PCT,
                    sys + SYSTEMWIDE_BAND_PCT,
                    color="0.85",
                    zorder=1,
                    label=f"Systemwide ±{SYSTEMWIDE_BAND_PCT:g} pp",
                )
            ax.plot(
                x,
                sys,
                linestyle="--",
                linewidth=1.5,
                color="black",
                label="Systemwide rolling YoY %",
                zorder=2,
            )

        # Raw YoY points then the rolling line, in the period's colour.
        ax.scatter(
            x,
            change[f"{spl}_yoy_pct"],
            marker=style["marker"],
            color=style["color"],
            s=28,
            label=f"{sp} YoY %",
            zorder=3,
        )
        ax.plot(
            x,
            change[f"{spl}_yoy_rolling_pct"],
            linestyle="-",
            linewidth=2.0,
            color=style["color"],
            label=f"{sp} rolling YoY %",
            zorder=2,
        )

        ax.set_title(sp, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("YoY %")
        ax.grid(PLOT_STYLE["grid"])
        ax.legend(fontsize="small", ncol=2, loc="best")

    bottom = axes[-1, 0]
    bottom.set_xlabel("Month")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(bottom.get_xticklabels(), rotation=PLOT_STYLE["rotation"], ha="right")

    fig.suptitle(f"Year-over-Year % Change – Route {route}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)


def export_route(
    route: str,
    monthly_long: pd.DataFrame,
    wide: pd.DataFrame,
    flags: pd.DataFrame,
    change: pd.DataFrame | None,
) -> None:
    """Export CSVs and plots into a per-route folder."""
    route_dir = OUTPUT_ROOT / f"route_{route}"
    route_dir.mkdir(parents=True, exist_ok=True)

    monthly_long[monthly_long["route"] == route].to_csv(route_dir / "monthly_long.csv", index=False)
    wide[wide["route"] == route].to_csv(route_dir / "monthly_wide.csv", index=False)
    flags[flags["route"] == route].to_csv(route_dir / "outage_flags.csv", index=False)

    plot_route_totals(route_dir, route, wide)
    plot_route_avgs(route_dir, route, wide)

    if change is not None:
        change.to_csv(route_dir / "yoy_percent_change.csv", index=False)
        plot_route_change(route_dir, route, change)

    logging.info("Exported %s", route_dir)


# =============================================================================
# RUN LOG
# =============================================================================


# Canonical version lives in utils/run_log.py — keep this copy in sync.
def extract_config_block(source_file: Path) -> str:
    """Return the text between the CONFIG markers in *source_file*.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def write_run_log(output_dir: Path) -> bool:
    """Write a run log of the configuration block into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "ntd_route_trends_runlog.txt"

    try:
        config_text: str = extract_config_block(Path(__file__))
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: list[str] = [
        "=" * 72,
        "NTD ROUTE TRENDS RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {Path(__file__).resolve()}",
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the end-to-end subset workflow."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _DEFAULT_DATA_ROOT = r"Path\To\Your\Input_Folder"
    _DEFAULT_OUTPUT_ROOT = r"Path\To\Your\Output_Folder"
    if str(DATA_ROOT) == _DEFAULT_DATA_ROOT or str(OUTPUT_ROOT) == _DEFAULT_OUTPUT_ROOT:
        logging.warning(
            "File paths are still set to their defaults. Update DATA_ROOT and "
            "OUTPUT_ROOT in the CONFIGURATION section before running."
        )
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        start_bound = parse_month_bound(START_MONTH)
        end_bound = parse_month_bound(END_MONTH)
    except ValueError as exc:
        logging.error(
            "START_MONTH / END_MONTH must be blank or 'Mon-YYYY' (e.g. 'Jan-2024'): %s", exc
        )
        sys.exit(1)
    if start_bound is not None and end_bound is not None and start_bound > end_bound:
        logging.error("START_MONTH (%s) is after END_MONTH (%s).", START_MONTH, END_MONTH)
        sys.exit(1)

    logging.info("Discovering workbooks under %s", DATA_ROOT)
    workbooks = discover_workbooks(DATA_ROOT)
    if not workbooks:
        logging.warning("No workbooks matched '%s' under %s.", WORKBOOK_GLOB, DATA_ROOT)
    else:
        logging.info("Discovered %d workbook(s).", len(workbooks))

    available_months = {parse_month(k) for k in workbooks}

    # Resolve the requested window. A blank bound falls back to the earliest /
    # latest workbook on disk so the expected-month grid (and its missing-month
    # warnings) cover exactly the data that is available.
    start_dt = start_bound if start_bound is not None else min(available_months, default=None)
    end_dt = end_bound if end_bound is not None else max(available_months, default=None)
    if start_dt is None or end_dt is None:
        logging.warning("No workbooks available to define a date range; nothing to export.")
        return

    expected_months = month_range(start_dt, end_dt)

    periods = periods_in_range(workbooks, start_bound, end_bound)
    start_label = (
        START_MONTH if start_bound is not None else format_month(start_dt) + " (earliest on disk)"
    )
    end_label = END_MONTH if end_bound is not None else format_month(end_dt) + " (latest on disk)"
    if not periods:
        logging.warning("No discovered workbooks fall in %s..%s.", start_label, end_label)
    else:
        logging.info("Processing %d period(s): %s..%s", len(periods), start_label, end_label)

    # Read every route once; the route subset and the systemwide comparison are
    # both derived from this single read.
    raw_all = load_raw(workbooks, periods)
    raw = raw_all[raw_all["ROUTE_NAME"].isin(set(ROUTES))].copy()

    monthly_long, observed_keys = aggregate_monthly_long(raw, expected_months)
    flags = flag_outages(monthly_long, expected_months, observed_keys, available_months)

    monthly_long = apply_manual_fixes(monthly_long, flags)

    # Recompute flags after manual edits (so exports reflect final used values).
    flags = flag_outages(monthly_long, expected_months, observed_keys, available_months)

    wide = to_wide(monthly_long)

    # Percent-change analytics (per-day-average basis, per service period).
    change_by_route: dict[str, pd.DataFrame] = {}
    if EXPORT_CHANGE_CHARTS:
        cleaned_long = clean_daily_avg(monthly_long, flags)
        holiday_counts = weekday_holiday_counts(HOLIDAYS)
        if holiday_counts:
            logging.info("Weekday holiday exclusion active for %d month(s).", len(holiday_counts))
        systemwide_by_sp: dict[str, pd.Series] | None = None
        if INCLUDE_SYSTEMWIDE_COMPARISON:
            systemwide_by_sp = compute_systemwide_perday_by_sp(raw_all, holiday_counts)
        for route in ROUTES:
            change_by_route[route] = compute_route_change(
                route, cleaned_long, systemwide_by_sp, expected_months, holiday_counts
            )

    for route in ROUTES:
        export_route(route, monthly_long, wide, flags, change_by_route.get(route))

    combined_dir = OUTPUT_ROOT / "_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    monthly_long.to_csv(combined_dir / "all_routes_monthly_long.csv", index=False)
    wide.to_csv(combined_dir / "all_routes_monthly_wide.csv", index=False)
    flags.to_csv(combined_dir / "all_routes_outage_flags.csv", index=False)
    if change_by_route:
        pd.concat(change_by_route.values(), ignore_index=True).to_csv(
            combined_dir / "all_routes_yoy_percent_change.csv", index=False
        )

    if not write_run_log(OUTPUT_ROOT) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        sys.exit(1)

    logging.info("Done. Outputs written to: %s", OUTPUT_ROOT)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
