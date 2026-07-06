"""Build the NTD regression anchor table from monthly ridership workbooks.

Collapses the per-service-period NTD monthly workbooks (Weekday / Saturday /
Sunday rows) into the single anchor table consumed by the ridership regression
(PART B). The anchor holds the dependent variable plus the "service supplied"
predictors, keyed to whatever grain the model expects:

    - GRAIN = "cross_section" -> one row per route_id, pooled over all periods
      in range (suitable for JOIN_KEYS = ("route_id",)).
    - GRAIN = "panel"         -> one row per route_id x period, with ``period``
      formatted as ``%Y-%m`` (suitable for JOIN_KEYS = ("route_id", "period")).

The workbooks carry separate Weekday / Saturday / Sunday rows. SERVICE_DAY_SELECTION
picks which of those each anchor row represents: a single service day in
isolation ("weekday" / "saturday" / "sunday"), all three summed into a
full-week monthly total ("combined"), or "each" to build all three single-day
anchors in one pass (one CSV per day, the day suffixed onto OUTPUT_FILENAME).
The choice is stamped onto every row as the SERVICE_DAY_OUT column so the
regression can confirm the service day it analyzes.

Workbooks are discovered by scanning DATA_ROOT: each file is assumed to hold a
single worksheet, and its month/year are parsed from the filename (full or
3-letter month, 4-digit year, in any order or separator). No hand-maintained
catalogue is required.

Output is a CSV written to OUTPUT_DIR / OUTPUT_FILENAME (or one day-suffixed CSV
per service day under "each"), plus a run-log sidecar. Point the regression's
ANCHOR_PATH at the CSV for the service day being modeled.

Measure conventions (kept consistent with ntd_monthly_summary.py):
    - boardings   = sum of MTH_BOARD across the selected service periods
      (already monthly).
    - hours       = sum of MTH_REV_HOURS across the selected service periods
      (already monthly).
    - revenue_miles = sum of (REV_MILES * DAYS) across the selected service
      periods, i.e. the monthly revenue-mile total, matching the monthly
      summary's MTH_REV_MILES.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Sequence

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

# -----------------------------------------------------------------------------
#  Input / output paths
# -----------------------------------------------------------------------------

DATA_ROOT: Final[Path] = Path(r"Path\To\Your\NTD_Folder")  # monthly workbooks
OUTPUT_DIR: Final[Path] = Path(r"Path\To\Your\Output\Folder")  # anchor lands here
OUTPUT_FILENAME: Final[str] = "ntd_anchor.csv"

# Glob used to discover workbooks under DATA_ROOT (non-recursive). Switch to
# DATA_ROOT.rglob if the monthly files live in dated subfolders.
WORKBOOK_GLOB: Final[str] = "*.xlsx"

# -----------------------------------------------------------------------------
#  Grain
# -----------------------------------------------------------------------------

# "cross_section" -> one row per route_id (pooled over the in-range periods).
# "panel"         -> one row per route_id x period (period as %Y-%m).
GRAIN: Final[str] = "panel"

# Inclusive month range to pull, as "Mon-YYYY". Workbooks whose parsed period
# falls outside this range are ignored even if present on disk. For a cross-
# sectional pool this window is the pooling window, so choose it deliberately.
#
# Leave either bound BLANK ("") to make it dynamic: an empty START_MONTH starts
# at the earliest workbook on disk, an empty END_MONTH ends at the latest, and
# leaving both blank pulls whatever is in DATA_ROOT with no clamping. Set a bound
# explicitly whenever you need control over the window.
START_MONTH: Final[str] = ""
END_MONTH: Final[str] = ""

# -----------------------------------------------------------------------------
#  Service-day selection
# -----------------------------------------------------------------------------

# Which service-day type each anchor row represents. NTD monthly workbooks carry
# separate Weekday / Saturday / Sunday rows; this selects which to fold into each
# anchor row:
#   - "weekday" / "saturday" / "sunday" -> isolate that single service day.
#   - "combined"                        -> sum all three into a full-week monthly
#                                          total (the historical behaviour).
#   - "each"                            -> build all three single-day anchors in
#                                          one pass, writing one CSV per day with
#                                          the day suffixed onto OUTPUT_FILENAME
#                                          (ntd_anchor_weekday.csv, ...).
# Whatever is chosen, the measures (boardings, hours, revenue miles) are summed
# over exactly the selected service period(s), so a single-day anchor carries that
# day's totals and a combined anchor carries the full-week totals.
SERVICE_DAY_SELECTION: Final[str] = "weekday"

# Output column recording SERVICE_DAY_SELECTION on every anchor row. Set this to
# match the regression's SERVICE_DAY_COLUMN so its service-day filter can assert
# the anchor's service day and abort on a mismatch (e.g. a combined anchor fed to
# a weekday-only run).
SERVICE_DAY_OUT: Final[str] = "service_day"

# -----------------------------------------------------------------------------
#  Output schema (align these with the regression's config)
# -----------------------------------------------------------------------------

# Join key / dependent / supply-predictor column names as they should appear in
# the anchor. ROUTE_ID_OUT must equal the regression's JOIN_KEYS route key;
# BOARDINGS_OUT must equal DEPENDENT_VAR; HOURS_OUT and REVMILES_OUT must appear
# in the regression's PREDICTORS.
ROUTE_ID_OUT: Final[str] = "route_id"
PERIOD_OUT: Final[str] = "period"  # panel only
BOARDINGS_OUT: Final[str] = "ntd_boardings"
REVMILES_OUT: Final[str] = "revenue_miles"

# The workbooks expose revenue hours (MTH_REV_HOURS), not scheduled hours. This
# is emitted honestly as "revenue_hours" by default. The regression config names
# this predictor "scheduled_hours" -> EITHER rename that predictor to
# "revenue_hours", OR set this to "scheduled_hours" to alias revenue hours as the
# supply measure (a documented approximation). Do not leave the two mismatched.
HOURS_OUT: Final[str] = "revenue_hours"

# -----------------------------------------------------------------------------
#  Cleaning behaviour
# -----------------------------------------------------------------------------

# Drop aggregated rows whose total boardings are <= 0. The regression log-
# transforms the dependent variable (LOG_DEPENDENT) and rejects non-positive
# values, so these rows cannot be modeled. When False, they are retained and a
# warning is emitted instead.
DROP_NONPOSITIVE_BOARDINGS: Final[bool] = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so the analyst is never
# left with an output that lacks a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Columns required from each workbook (post-normalisation).
REQUIRED_COLS: Final[list[str]] = [
    "ROUTE_NAME",
    "SERVICE_PERIOD",
    "MTH_BOARD",
    "MTH_REV_HOURS",
    "REV_MILES",
    "DAYS",
]

# Canonical service-period label set for each SERVICE_DAY_SELECTION value. The
# selected labels are both the rows kept from each workbook and the rows summed
# into one anchor row. Labels match normalise_service_period's output.
_SERVICE_DAY_SETS: Final[dict[str, list[str]]] = {
    "weekday": ["Weekday"],
    "saturday": ["Saturday"],
    "sunday": ["Sunday"],
    "combined": ["Weekday", "Saturday", "Sunday"],
}

# The single-day selections built by SERVICE_DAY_SELECTION = "each", in output order.
_EACH_SELECTIONS: Final[list[str]] = ["weekday", "saturday", "sunday"]

# Internal (pre-rename) measure column names.
_BOARD: Final[str] = "_board"
_HOURS: Final[str] = "_hours"
_REVMILES: Final[str] = "_rev_miles"

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


# =============================================================================
# HELPERS
# =============================================================================


def parse_month(value: str) -> datetime:
    """Parse a ``Mon-YYYY`` string into a month-start datetime."""
    dt = datetime.strptime(value.strip(), "%b-%Y")
    return datetime(dt.year, dt.month, 1)


def to_period_ym(value: str) -> str:
    """Convert a ``Mon-YYYY`` period key into the ``%Y-%m`` the model expects."""
    return parse_month(value).strftime("%Y-%m")


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


def safe_float(value: Any) -> float | None:
    """Return ``value`` as a float if it looks numeric, else ``None``."""
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
    """Upper-case, trim, and underscore-join the column labels."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    return out


def normalise_route(value: Any) -> str:
    """Normalise a route token (e.g. ``610.0`` -> ``'610'``) for stable keys."""
    s = str(value).strip().upper().replace(" ", "")
    return re.sub(r"\.0$", "", s)


def normalise_service_period(value: Any) -> str:
    """Map assorted service-period spellings onto {Weekday, Saturday, Sunday}."""
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
    return mapping.get(str(value).strip().lower(), str(value).strip())


def day_type_filename(base: str, service_day: str) -> str:
    """Suffix an anchor filename with its service day (ntd_anchor.csv -> ntd_anchor_weekday.csv)."""
    p = Path(base)
    return f"{p.stem}_{service_day}{p.suffix}"


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


def read_month_workbook(period: str, path: Path, keep_periods: list[str]) -> pd.DataFrame:
    """Read one workbook's only sheet and return tidy measure rows.

    Returns an empty frame (with a warning) if the file is missing or lacks a
    required column, so a single bad month does not abort the build.

    Args:
        period: ``Mon-YYYY`` key for this workbook.
        path: Path to the workbook on disk.
        keep_periods: Service-period labels to retain (rows with any other label
            are dropped). Driven by SERVICE_DAY_SELECTION.
    """
    if not path.exists():
        logging.warning("Workbook missing on disk: %s (period=%s)", path, period)
        return pd.DataFrame()

    converters: dict[str, Any] = {
        c: safe_float for c in ("MTH_BOARD", "MTH_REV_HOURS", "REV_MILES", "DAYS")
    }
    try:
        # sheet_name=0 reads the single worksheet regardless of its name.
        df = pd.read_excel(path, sheet_name=0, converters=converters)
    except Exception:
        logging.exception("Failed to read %s (period=%s)", path, period)
        return pd.DataFrame()

    df = normalise_columns(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logging.warning(
            "Skipping %s (period=%s): missing column(s) %s", path.name, period, ", ".join(missing)
        )
        return pd.DataFrame()

    out = pd.DataFrame()
    out[ROUTE_ID_OUT] = df["ROUTE_NAME"].apply(normalise_route)
    out["_service_period"] = df["SERVICE_PERIOD"].apply(normalise_service_period)
    out[_BOARD] = pd.to_numeric(df["MTH_BOARD"], errors="coerce")
    out[_HOURS] = pd.to_numeric(df["MTH_REV_HOURS"], errors="coerce")
    # Monthly revenue miles = per-day revenue miles * service days, matching
    # ntd_monthly_summary's MTH_REV_MILES derivation.
    out[_REVMILES] = pd.to_numeric(df["REV_MILES"], errors="coerce") * pd.to_numeric(
        df["DAYS"], errors="coerce"
    )
    out["_period_key"] = period
    out["_period_ym"] = to_period_ym(period)

    out = out[out["_service_period"].isin(keep_periods)]
    out = out[out[ROUTE_ID_OUT].astype(bool) & (out[ROUTE_ID_OUT].str.lower() != "nan")]
    return out


def load_raw(
    workbooks: dict[str, Path], periods: list[str], keep_periods: list[str]
) -> pd.DataFrame:
    """Read and concatenate every in-range workbook into one tidy frame.

    Args:
        workbooks: Period -> path map from :func:`discover_workbooks`.
        periods: In-range period keys to read, chronologically.
        keep_periods: Service-period labels to retain (see SERVICE_DAY_SELECTION).
    """
    frames: list[pd.DataFrame] = []
    for period in periods:
        path = workbooks.get(period)
        if path is None:
            continue
        df = read_month_workbook(period, path, keep_periods)
        if df.empty:
            continue
        frames.append(df)
        logging.info("Loaded %s: %d service-period rows", period, len(df))

    if not frames:
        return pd.DataFrame(
            columns=[
                ROUTE_ID_OUT,
                "_service_period",
                _BOARD,
                _HOURS,
                _REVMILES,
                "_period_key",
                "_period_ym",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_anchor(raw: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Collapse service periods and aggregate to the requested grain.

    Args:
        raw: Tidy per-service-period rows from :func:`load_raw`.
        grain: ``"cross_section"`` or ``"panel"``.

    Returns:
        The anchor table with output-schema column names.
    """
    if raw.empty:
        cols = [ROUTE_ID_OUT, BOARDINGS_OUT, HOURS_OUT, REVMILES_OUT]
        if grain == "panel":
            cols.insert(1, PERIOD_OUT)
        return pd.DataFrame(columns=cols)

    group_keys = [ROUTE_ID_OUT] if grain == "cross_section" else [ROUTE_ID_OUT, "_period_ym"]
    agg = raw.groupby(group_keys, as_index=False).agg(
        **{
            BOARDINGS_OUT: (_BOARD, "sum"),
            HOURS_OUT: (_HOURS, "sum"),
            REVMILES_OUT: (_REVMILES, "sum"),
        }
    )

    # Round the supply measures to 2 decimals for the published anchor.
    # round() preserves NaN and sign, so clean_anchor's checks still apply.
    agg[[HOURS_OUT, REVMILES_OUT]] = agg[[HOURS_OUT, REVMILES_OUT]].round(2)

    if grain == "panel":
        agg = agg.rename(columns={"_period_ym": PERIOD_OUT})
        agg = agg.sort_values([ROUTE_ID_OUT, PERIOD_OUT], ignore_index=True)
        ordered = [ROUTE_ID_OUT, PERIOD_OUT, BOARDINGS_OUT, HOURS_OUT, REVMILES_OUT]
    else:
        agg = agg.sort_values([ROUTE_ID_OUT], ignore_index=True)
        ordered = [ROUTE_ID_OUT, BOARDINGS_OUT, HOURS_OUT, REVMILES_OUT]

    return agg[ordered]


def clean_anchor(anchor: pd.DataFrame) -> pd.DataFrame:
    """Warn on / drop non-modelable rows (non-positive boardings, NaN supply)."""
    nan_supply = int(anchor[[HOURS_OUT, REVMILES_OUT]].isna().any(axis=1).sum())
    if nan_supply:
        logging.warning(
            "%d row(s) have NaN %s/%s; the regression will drop these when building "
            "the design matrix.",
            nan_supply,
            HOURS_OUT,
            REVMILES_OUT,
        )

    nonpos = anchor[BOARDINGS_OUT] <= 0
    n_nonpos = int(nonpos.sum())
    if n_nonpos and DROP_NONPOSITIVE_BOARDINGS:
        logging.warning(
            "Dropping %d row(s) with %s <= 0 (cannot be log-transformed downstream).",
            n_nonpos,
            BOARDINGS_OUT,
        )
        anchor = anchor[~nonpos].reset_index(drop=True)
    elif n_nonpos:
        logging.warning(
            "%d row(s) have %s <= 0 and were KEPT; the regression's LOG_DEPENDENT will "
            "abort unless these are removed.",
            n_nonpos,
            BOARDINGS_OUT,
        )
    return anchor


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source.

    Returns ``None`` in interactive contexts (Jupyter/IPython) where ``__file__``
    is undefined, so the run log can degrade gracefully instead of raising.
    """
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


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


def write_run_log(output_dir: Path, summary_lines: list[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    When running interactively (no ``__file__`` on disk), the verbatim config
    cannot be read back from source; the log is still written with a note in
    place of the config block so the sidecar is never silently skipped.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "ntd_anchor_builder_runlog.txt"

    source_file = resolve_source_file()
    if source_file is None:
        config_text = "(config block unavailable: interactive session, no __file__ on disk)"
        source_display = "<interactive>"
    else:
        try:
            config_text = extract_config_block(source_file)
        except (OSError, ValueError) as exc:
            logging.error("Could not extract config block for run log: %s", exc)
            return False
        source_display = str(source_file)

    lines: list[str] = [
        "=" * 72,
        "NTD ANCHOR BUILD RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "BUILD SUMMARY",
        "-" * 72,
        *summary_lines,
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Every option defaults to the matching CONFIGURATION constant, so the script
    behaves identically when run with no flags (notebook / ArcGIS Pro) and can be
    fully driven by an orchestrator (e.g. ``prep_features_private.py``) when
    flags are supplied.
    """
    p = argparse.ArgumentParser(
        description="Build the NTD regression anchor from monthly ridership workbooks."
    )
    p.add_argument("--data-root", default=str(DATA_ROOT), help="Folder of monthly NTD workbooks.")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Where the anchor CSV is written.")
    p.add_argument("--output-filename", default=OUTPUT_FILENAME, help="Anchor CSV filename.")
    p.add_argument(
        "--grain",
        default=GRAIN,
        choices=["cross_section", "panel"],
        help="One row per route ('cross_section') or per route x month ('panel').",
    )
    p.add_argument(
        "--service-day",
        default=SERVICE_DAY_SELECTION,
        help="weekday / saturday / sunday / combined, or 'each' to write all three "
        "single-day anchors in one pass.",
    )
    p.add_argument(
        "--start-month",
        default=START_MONTH,
        help="Inclusive start as 'Mon-YYYY' (blank = earliest).",
    )
    p.add_argument(
        "--end-month", default=END_MONTH, help="Inclusive end as 'Mon-YYYY' (blank = latest)."
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Read the monthly workbooks, build the anchor, and export it."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(argv)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_filename = args.output_filename
    grain = args.grain
    start_month = args.start_month
    end_month = args.end_month

    if grain not in {"cross_section", "panel"}:
        logging.error("--grain must be 'cross_section' or 'panel', got '%s'.", grain)
        sys.exit(1)

    selection = args.service_day.strip().lower()
    if selection not in _SERVICE_DAY_SETS and selection != "each":
        logging.error(
            "--service-day must be one of %s, got '%s'.",
            sorted([*_SERVICE_DAY_SETS, "each"]),
            args.service_day,
        )
        sys.exit(1)
    # "each" reads every service-period row once, then builds the three single-day
    # anchors from the same tidy frame (one workbook pass, three CSVs).
    selections = _EACH_SELECTIONS if selection == "each" else [selection]
    selected_periods = (
        _SERVICE_DAY_SETS["combined"] if selection == "each" else _SERVICE_DAY_SETS[selection]
    )

    if (
        str(data_root) == r"Path\To\Your\NTD_Folder"
        or str(output_dir) == r"Path\To\Your\Output\Folder"
    ):
        logging.warning(
            "File paths are still set to their defaults. Update DATA_ROOT and OUTPUT_DIR "
            "in the CONFIGURATION section, or pass --data-root/--output-dir, before running."
        )
        return

    if HOURS_OUT != "scheduled_hours":
        logging.info(
            "Emitting supply-hours column as '%s'. Ensure the regression's PREDICTORS "
            "names it '%s' (not 'scheduled_hours').",
            HOURS_OUT,
            HOURS_OUT,
        )

    logging.info(
        "Service-day selection: '%s' (periods summed: %s).", selection, ", ".join(selected_periods)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== STEP 0: DISCOVER WORKBOOKS UNDER %s ===", data_root)
    workbooks = discover_workbooks(data_root)
    if not workbooks:
        logging.warning("No workbooks matched '%s' under %s.", WORKBOOK_GLOB, data_root)
    else:
        logging.info("Discovered %d workbook(s).", len(workbooks))

    try:
        start_dt = parse_month_bound(start_month)
        end_dt = parse_month_bound(end_month)
    except ValueError as exc:
        logging.error(
            "--start-month / --end-month must be blank or 'Mon-YYYY' (e.g. 'Jul-2024'): %s", exc
        )
        sys.exit(1)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        logging.error("--start-month (%s) is after --end-month (%s).", start_month, end_month)
        sys.exit(1)

    periods = periods_in_range(workbooks, start_dt, end_dt)
    # Report the window actually applied, naming the dynamic ends explicitly so a
    # blank bound is never mistaken for a silently dropped month.
    start_label = start_month if start_dt is not None else "earliest on disk"
    end_label = end_month if end_dt is not None else "latest on disk"
    if not periods:
        logging.warning("No discovered workbooks fall in %s..%s.", start_label, end_label)
        logging.error(
            "No in-range workbooks were found under %s; the anchor cannot be built.", data_root
        )
        sys.exit(1)

    logging.info("=== STEP 1: READ WORKBOOKS (%s..%s) ===", start_label, end_label)
    raw = load_raw(workbooks, periods, selected_periods)

    summary_lines = [
        f"Grain:            {grain}",
        f"Service day:      {selection} (periods read: {', '.join(selected_periods)})",
        f"Period range:     {start_label}..{end_label}",
        f"Workbooks found:  {len(workbooks)}",
        f"Periods loaded:   {len(periods)} ({', '.join(periods) or 'none'})",
    ]

    for sel in selections:
        sub = (
            raw[raw["_service_period"].isin(_SERVICE_DAY_SETS[sel])] if selection == "each" else raw
        )
        logging.info("=== STEP 2: AGGREGATE TO '%s' GRAIN ('%s') ===", grain, sel)
        anchor = build_anchor(sub, grain)
        anchor = clean_anchor(anchor)

        # Stamp the service-day selection onto every row so the regression can assert
        # which service day it is analyzing.
        insert_at = 2 if grain == "panel" else 1
        anchor.insert(insert_at, SERVICE_DAY_OUT, sel)

        filename = (
            day_type_filename(output_filename, sel) if selection == "each" else output_filename
        )
        out_path = output_dir / filename
        anchor.to_csv(out_path, index=False)
        logging.info("Anchor written: %s (%d rows, %d cols).", out_path, *anchor.shape)

        n_routes = anchor[ROUTE_ID_OUT].nunique() if not anchor.empty else 0
        summary_lines.append(f"[{sel}] {filename}: {len(anchor)} row(s), {n_routes} route(s)")
        if grain == "panel" and not anchor.empty:
            summary_lines.append(f"[{sel}]   unique periods: {anchor[PERIOD_OUT].nunique()}")

    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
        sys.exit(1)

    logging.info("All processing complete. Script completed successfully.")


if __name__ == "__main__":
    main()
