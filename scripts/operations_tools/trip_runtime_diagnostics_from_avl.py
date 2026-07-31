"""Trip-level runtime diagnostics and schedule performance review.

This module analyzes observed bus trips to evaluate how actual runtime and timing
deviations compare to scheduled values. It generates both row-level flags and
summary statistics, emphasizing on-time performance (OTP) and 85th-percentile runtime.

Two data-quality screens run alongside the performance review: start times with
unusually few observations, and back-to-back trips whose runtimes or
actual/scheduled ratios diverge far more than neighbors on the same stop
pattern should.

Designed to support schedule tuning, the script suggests time-of-day bands using
Fisher–Jenks segmentation and provides visual diagnostics for start time, runtime,
and deviation patterns.

Assumes route-wise CSVs of trip observations with key timestamp columns.

Outputs
-------
Written per route (and per direction when ``SPLIT_BY_DIRECTION`` is True) under
``OUTPUT_ROOT_DIR/<route>/[<direction>/]``, where ``<day>`` is the service-day
tag (e.g. ``WEEKDAY``):

- ``events_retained_<day>.csv`` - trips kept for analysis, with deviations and
  OTP compliance flags.
- ``events_excluded_<day>.csv`` - outlier trips trimmed from the analysis
  (only when ``WRITE_EXCLUSIONS`` is True).
- ``trip_summary_<day>.xlsx`` - summarized runtime and OTP statistics per trip.
- ``time_bands_<day>.xlsx`` - suggested time-of-day bands for runtime
  adjustment.
- ``low_sample_start_times_<day>.csv`` - start times with unusually low sample
  counts, for data-quality review.
- ``back_to_back_runtime_flags_<day>.csv`` - consecutive trips on the same stop
  pattern whose runtimes or actual/scheduled ratios differ far more than
  neighbors should, for data-quality review. Written only when something is
  flagged.
- ``trip_runtime_diagnostics_runlog.txt`` - the verbatim CONFIGURATION block,
  a timestamp, and this run's counts, written beside every set of outputs.
- ``plots/*.png`` - diagnostic plots (start/finish/runtime deviation boxplots
  and a scheduled vs. 85th-percentile runtime bar chart).

Typical usage
-------------
Update the paths in the CONFIGURATION section and run from a shell, ArcGIS
Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import difflib
import logging
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Final, Iterable, List, Sequence, TypeAlias

# May need to comment out TypeAlias import for old ArcPro Python versions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sentinel markers used by extract_config_block / write_run_log to identify the
# CONFIGURATION block that is copied verbatim into the run-log sidecar.
CONFIG_BEGIN_MARKER: Final[str] = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: Final[str] = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================

# === BEGIN CONFIG ===

INPUT_ROOT_DIR: Final[Path] = Path(r"Path\To\Your\Data_Folder_with_observed_trips")
OUTPUT_ROOT_DIR: Final[Path] = Path(r"Path\To\Your\Output_Folder")

ROUTES_TO_INCLUDE: Final[set[str]] = {"101", "202"}  # optional whitelist

DATE_START: Final[pd.Timestamp] = pd.Timestamp("2024-06-30")
DATE_END: Final[pd.Timestamp] = pd.Timestamp("2025-07-24")

LOW_SAMPLE_FRAC: Final[float] = 0.20  # 20 % of the median n_events

MAX_TIME_BANDS: Final[int | None] = 6  # None ⇒ no hard cap
ENFORCE_MIN_BAND_SIZE: Final[bool] = True  # toggle merging on/off
MIN_BAND_SIZE: Final[int] = 2  # ignored when above is False

WRITE_EXCLUSIONS: Final[bool] = True  # write events_excluded_*.csv
SPLIT_BY_DIRECTION: Final[bool] = True  # keep current folder depth; False → flat

# Paired stems so all writers reference one source of truth
_RETAINED_STEM: Final[str] = "events_retained"
_EXCLUDED_STEM: Final[str] = "events_excluded"

EXCLUDE_DATES: Final[list[str]] = [
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-10-13",
    "2025-11-11",
    "2025-11-27",
    "2025-11-28",
    "2025-12-25",
]

SERVICE_DAY_FILTER: Final[str | None] = "WEEKDAY"  # "SATURDAY" | "SUNDAY" | None

OTP_EARLY_MIN: Final[int] = -1
OTP_LATE_MIN: Final[int] = 6
OTP_TARGET_PCT: Final[float] = 85.0

TIME_COL_NAME: Final[str] = "trip_start_time"
_DOW_CHOICES: Final[set[str | None]] = {None, "", "WEEKDAY", "SATURDAY", "SUNDAY"}

# Outlier‑trimming settings (per‑trip)
TRIM_OUTLIERS: Final[bool] = True
TRIM_FRAC: Final[float] = 0.01  # drop shortest & longest 1 %

# Back-to-back trip checks. Each start-time token is compared with the next one
# on the same stop pattern. Neighboring trips on one pattern should behave
# alike, so a step change is normally a data or schedule problem rather than
# real operating conditions.
#   * B2B_RUNTIME_RATIO    - flag when the longer median runtime divided by the
#     shorter one reaches this (2.0 = one trip takes twice as long).
#   * B2B_RUNTIME_DIFF_MIN - flag when the median runtimes differ by at least
#     this many minutes.
#   * B2B_SCHED_RATIO_DIFF - flag when the two trips' actual/scheduled runtime
#     ratios differ by at least this much (0.50 = one trip runs 50% further over
#     its scheduled time than its neighbor does).
#   * B2B_MAX_GAP_MIN      - skip pairs whose scheduled starts are more than this
#     many minutes apart (peak-only gaps, where a runtime change is legitimate).
#     Set 0 to compare every consecutive pair regardless of the gap.
#   * B2B_MIN_OBS          - both trips need at least this many observations
#     before the pair is compared.
# A pair is flagged when *any* enabled test trips. Set an individual threshold to
# 0 to disable just that test.
B2B_RUNTIME_RATIO: Final[float] = 2.0
B2B_RUNTIME_DIFF_MIN: Final[float] = 20.0
B2B_SCHED_RATIO_DIFF: Final[float] = 0.50
B2B_MAX_GAP_MIN: Final[float] = 90.0
B2B_MIN_OBS: Final[int] = 3

# Columns tried, in order, to identify a trip's stop pattern: ``pattern_id`` in
# TIDES, ``Variation`` in the legacy AVL exports. Pairing trips across patterns
# is meaningless - a short-turn legitimately runs far shorter than a full-length
# trip - so when neither column is present the check is skipped rather than run
# on mixed patterns. Set B2B_REQUIRE_PATTERN to False to compare anyway, which
# pools branching variants and can flag correct service.
B2B_PATTERN_COLS: Final[tuple[str, ...]] = ("pattern_id", "Variation")
B2B_REQUIRE_PATTERN: Final[bool] = True

# Abort when the run-log sidecar cannot be written, so an output is never left
# without its record. Set False only for genuinely read-only destinations.
REQUIRE_RUN_LOG: Final[bool] = True

# ─── derived paths (initial stubs – reassigned per route in main()) ─── #
OUTPUT_DIR: Path = OUTPUT_ROOT_DIR
PLOTS_DIR: Path = OUTPUT_DIR / "plots"

# Valid directions to enforce and normalize to (keep UPPERCASE strings)
ALLOWED_DIRECTIONS: List[str] = [
    "NORTHBOUND",
    "SOUTHBOUND",
    "EASTBOUND",
    "WESTBOUND",
    "LOOP",
]

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

# ---------------------------------------------------------------------
# TYPE ALIASES
# ---------------------------------------------------------------------

PlotFunc: TypeAlias = Callable[[pd.DataFrame], None]

# =============================================================================
# FUNCTIONS
# =============================================================================


def _detect_sep(path: Path) -> str:
    """Return delimiter based on extension (csv → comma, others → tab)."""
    return "," if path.suffix.lower() == ".csv" else "\t"


def _ensure_plot_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _trim_pct(series: pd.Series, frac: float = TRIM_FRAC) -> pd.Series:
    """Trim the lowest and highest *frac* proportion of values."""
    if series.empty or frac <= 0:
        return series
    lo = series.quantile(frac)
    hi = series.quantile(1 - frac)
    return series[(series >= lo) & (series <= hi)]


def _safe_plot(plot_func: PlotFunc, df: pd.DataFrame) -> None:
    """Safely call a plotting function if the DataFrame is not empty.

    Args:
        plot_func: A function that takes a DataFrame and produces a plot.
        df: Data to plot. If empty, the plot function is not called.
    """
    if not df.empty:
        plot_func(df)
    else:
        logging.warning("   ⚠  No rows after filters; skipping plots.")


def normalize_direction_value(value: str, allowed: List[str]) -> str:
    """Coerce a free-text direction to one of the allowed values.

    Uses difflib to guess the closest allowed label. Always returns one of
    ``allowed``. Emits a warning if the original does not exactly match.

    Examples:
        "NOTHTBOUND" -> "NORTHBOUND"
        "west bound" -> "WESTBOUND"
    """
    val = (value or "").strip().upper()
    if val in allowed:
        return val
    guess = difflib.get_close_matches(val, allowed, n=1, cutoff=0.6)
    if guess:
        fixed = guess[0]
        warnings.warn(f"Direction normalized: {value!r} → {fixed!r}", RuntimeWarning, stacklevel=2)
        return fixed
    warnings.warn(
        f"Unrecognized direction {value!r}; defaulting to 'LOOP'", RuntimeWarning, stacklevel=2
    )
    return "LOOP"


def normalize_directions_column(df: pd.DataFrame, allowed: List[str]) -> pd.DataFrame:
    """Normalize the 'Direction' column in place to the allowed list."""
    df = df.copy()
    df["Direction"] = (
        df["Direction"].astype(str).map(lambda s: normalize_direction_value(s, allowed))
    )
    return df


# -----------------------------------------------------------------------------
# ROUTE DISCOVERY HELPERS
# -----------------------------------------------------------------------------

_route_token_re = re.compile(r"([0-9]{1,4})")
_ROUTE_HDR_RE: Final[re.Pattern[str]] = re.compile(r"\s*route\w*\s*", flags=re.I)
# Route-like headers that describe the service rather than identify the route.
# TIDES carries route_type ("Local Bus Service") and route_type_agency ("LOCAL"),
# neither of which holds a route number.
_ROUTE_ATTR_RE: Final[re.Pattern[str]] = re.compile(r"\s*route_?type\w*\s*", flags=re.I)


def _clean_route_id(raw: str | float | int) -> str:
    """Canonical 1‑to‑4 digit route ID from a Route cell."""
    txt = str(raw)
    m = _route_token_re.search(txt)
    if not m:
        raise ValueError(f"Cannot parse route from value {txt!r}")
    return m.group(1).lstrip("0") or "0"


def _route_column_candidates(header: Iterable[str]) -> List[str]:
    """Rank a file's route-like headers by how likely each identifies a route.

    Human-readable ``Route`` first, then other non-ID route columns (such as
    ``routeName``), then ID-style ones (``route_id``, ``RouteID``). Columns that
    describe the service type are dropped outright: they match the same
    ``route*`` pattern but never hold a route number.

    Args:
        header: Column names read from the file.

    Returns:
        Candidate column names, best first; empty when none look like a route.
    """
    route_like = [
        c for c in header if _ROUTE_HDR_RE.fullmatch(c) and not _ROUTE_ATTR_RE.fullmatch(c)
    ]
    named = [c for c in route_like if c.strip().lower() == "route"]
    rest = [c for c in route_like if c not in named]
    non_id = [c for c in rest if not re.search(r"id\s*$", c, flags=re.I)]
    id_like = [c for c in rest if c not in non_id]
    return [*named, *non_id, *id_like]


def _discover_route_csvs(
    root: Path,
    wanted: set[str] | None = None,
) -> dict[str, list[Path]]:
    """Crawl *root* recursively and build {route → [files]}.

    A file is linked to **every** route ID that appears in its chosen route column
    (prefers a human-readable ``Route`` column over ID-like columns such as
    ``RouteID``). Mixed files get processed by each relevant route run.

    The selected route column may be “Route”, “routeName”, “Route_ID”, etc. Each
    candidate is tried in turn, so a file is only skipped once *no* route-like
    column yields a usable ID - a column full of unparseable values falls through
    to the next candidate instead of dropping the file.

    Honors *wanted* whitelist (1-to-4 digit IDs with leading zeros stripped).
    """
    buckets: dict[str, list[Path]] = defaultdict(list)

    for p in root.rglob("*.csv"):
        try:
            # Read just the header to find the Route-like columns.
            header = pd.read_csv(p, sep=_detect_sep(p), nrows=0).columns.tolist()
        except (OSError, ValueError, UnicodeDecodeError) as e:
            logging.warning("!! %s: %s; skipped", p.name, e)
            continue

        candidates = _route_column_candidates(header)
        if not candidates:
            logging.warning("!! %s: no route-like column; skipped", p.name)
            continue

        ids: set[str] = set()
        last_error: str = ""
        for route_col in candidates:
            try:
                routes = (
                    pd.read_csv(
                        p,
                        sep=_detect_sep(p),
                        usecols=[route_col],
                        dtype=str,
                        low_memory=False,
                    )[route_col]
                    .dropna()
                    .unique()
                )
                ids = {_clean_route_id(r) for r in routes}
            except (OSError, ValueError, KeyError, UnicodeDecodeError) as e:
                last_error = f"{route_col}: {e}"
                continue
            if ids:
                break

        if not ids:
            logging.warning("!! %s: %s; skipped", p.name, last_error or "no route IDs found")
            continue

        if wanted:
            ids &= wanted
        if not ids:
            continue  # file has no wanted routes

        for rid in ids:
            buckets[rid].append(p)

    return buckets


def load_trip_files(files: Iterable[Path]) -> pd.DataFrame:
    """Load and concatenate observed trip CSV files.

    Args:
        files: List of file paths to read.

    Returns:
        Combined DataFrame with parsed time columns and unified schema.
    """
    frames = [pd.read_csv(p, sep=_detect_sep(p), dtype=str, low_memory=False) for p in files]
    df = pd.concat(frames, ignore_index=True)

    # Detect TIDES format
    if "schedule_trip_start" in df.columns:
        # Filter for 'In service' and remove 'Canceled'
        if "trip_type" in df.columns:
            # Normalize to lower/title case if needed, but assuming standard TIDES casing
            # TIDES typically uses "In service", "Deadhead", etc.
            # Use fillna to avoid dropping legacy rows in mixed batches
            df = df.loc[df["trip_type"].fillna("In service") == "In service"].copy()
        if "schedule_relationship" in df.columns:
            # Use fillna to avoid dropping legacy rows
            df = df.loc[df["schedule_relationship"].fillna("Scheduled") != "Canceled"].copy()

        # Rename columns to legacy internal names
        rename_map = {
            "route_id": "Route",
            "direction_id": "Direction",
            "trip_id_performed": "TripID",
            "schedule_trip_start": "Scheduled Start Time",
            "schedule_trip_end": "Scheduled Finish Time",
            "actual_trip_start": "Actual Start Time",
            "actual_trip_end": "Actual Finish Time",
        }
        df = df.rename(columns=rename_map)

        # Convert timestamps
        for col in (
            "Scheduled Start Time",
            "Scheduled Finish Time",
            "Actual Start Time",
            "Actual Finish Time",
        ):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Derive trip_start_time (HH:MM) from the full timestamp
        # Format as HH:MM
        df[TIME_COL_NAME] = df["Scheduled Start Time"].dt.strftime("%H:%M")

        # Ensure Direction is string for consistent handling
        if "Direction" in df.columns:
            df["Direction"] = df["Direction"].astype(str)

        # Flag as TIDES so we can skip legacy normalization steps
        df["_is_tides"] = True

    else:
        # Legacy Logic
        for col in (
            "Scheduled Start Time",
            "Scheduled Finish Time",
            "Actual Start Time",
            "Actual Finish Time",
        ):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["_is_tides"] = False

    return df


def extract_trip_start_time(df: pd.DataFrame, trip_col: str = "Trip") -> pd.DataFrame:
    """Extract trip start time (HH:MM) from a Trip column.

    Args:
        df: Trip data with a string-based Trip column.
        trip_col: Name of the Trip column to extract time from.

    Returns:
        Copy of input DataFrame with a new 'trip_start_time' column.
    """
    # If the time column is already populated (e.g. TIDES data), skip extraction
    if TIME_COL_NAME in df.columns:
        return df

    df = df.copy()
    df[TIME_COL_NAME] = df[trip_col].str.extract(r"^\s*([0-2]?\d:[0-5]\d)")[0]
    return df


def filter_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """Filter trips to only those within the configured date range.

    Args:
        df: Trip records with 'Scheduled Start Time' as datetime.

    Returns:
        Filtered DataFrame with rows inside DATE_START and DATE_END.
    """
    return df.loc[df["Scheduled Start Time"].between(DATE_START, DATE_END)].copy()


def filter_routes(df: pd.DataFrame, wanted: set[str]) -> pd.DataFrame:
    """Filter trips to only those matching desired route IDs.

    Args:
        df: Trip data with a 'Route' column.
        wanted: Canonical route IDs (1–4 digits, leading zeros removed).

    Returns:
        Subset of DataFrame containing only wanted route IDs.
    """
    wanted_canon = {str(x).lstrip("0") for x in wanted}
    route_num = df["Route"].astype(str).str.extract(r"^\s*([0-9]{1,4})")[0].str.lstrip("0")
    return df.loc[route_num.isin(wanted_canon)].copy()


def filter_holidays(df: pd.DataFrame, dates: Iterable[str]) -> pd.DataFrame:
    """Remove trips that occur on excluded holiday dates.

    Args:
        df: Trip data with 'Scheduled Start Time' as datetime.
        dates: Iterable of holiday dates as YYYY-MM-DD strings.

    Returns:
        DataFrame excluding rows on the given dates.
    """
    bad = {pd.to_datetime(d).date() for d in dates}
    return df.loc[~df["Scheduled Start Time"].dt.date.isin(bad)].copy()


def filter_service_day(df: pd.DataFrame, which: str | None) -> pd.DataFrame:
    """Filter trips by service day: WEEKDAY, SATURDAY, or SUNDAY.

    Args:
        df: Trip data with 'Scheduled Start Time' as datetime.
        which: Day type to keep. Must be 'WEEKDAY', 'SATURDAY', 'SUNDAY', or None.

    Returns:
        DataFrame filtered to the selected day type.
    """
    if which not in _DOW_CHOICES:
        raise ValueError("SERVICE_DAY_FILTER must be WEEKDAY, SATURDAY, SUNDAY or None")
    if not which:
        return df
    dow = df["Scheduled Start Time"].dt.dayofweek
    keep = (dow <= 4) if which == "WEEKDAY" else (dow == 5 if which == "SATURDAY" else dow == 6)
    return df.loc[keep].copy()


def add_deviation_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add schedule vs. actual time deviation columns to the DataFrame.

    Args:
        df: Trip data with scheduled and actual time columns.

    Returns:
        DataFrame with new columns for start/finish/runtime deviations.
    """
    df = df.copy()
    df["start_dev_min"] = (
        df["Actual Start Time"] - df["Scheduled Start Time"]
    ).dt.total_seconds() / 60
    df["finish_dev_min"] = (
        df["Actual Finish Time"] - df["Scheduled Finish Time"]
    ).dt.total_seconds() / 60
    df["scheduled_runtime_min"] = (
        df["Scheduled Finish Time"] - df["Scheduled Start Time"]
    ).dt.total_seconds() / 60
    df["actual_runtime_min"] = (
        df["Actual Finish Time"] - df["Actual Start Time"]
    ).dt.total_seconds() / 60
    df["runtime_dev_min"] = df["actual_runtime_min"] - df["scheduled_runtime_min"]
    return df


def add_otp_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add both legacy *on_time* and new *within_window* boolean columns.

    The columns are identical for one release cycle so downstream code
    can migrate from *on_time* to *within_window* at leisure.
    """
    flag = df["start_dev_min"].between(
        OTP_EARLY_MIN,
        OTP_LATE_MIN,
        inclusive="both",
    )
    return df.assign(on_time=flag, within_window=flag)


def _box_by_trip(
    df: pd.DataFrame,
    col: str,
    title: str,
    file_name: str,
    shade_range: tuple[float, float] | None = None,
) -> None:
    data = df[[TIME_COL_NAME, col]].dropna()
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=data, x=TIME_COL_NAME, y=col, whis=(0, 100), showfliers=False)
    if shade_range:
        plt.axhspan(*shade_range, color="g", alpha=0.15)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel("Minutes")
    plt.xlabel("Scheduled trip start")
    plt.xticks(rotation=90)
    plt.tight_layout()
    _ensure_plot_dirs()
    plt.savefig(PLOTS_DIR / file_name, dpi=150)
    plt.close()


def plot_start_dev_shaded(df: pd.DataFrame) -> None:
    """Create a boxplot of start-time deviations with OTP shading.

    Args:
        df: Trip data with 'start_dev_min' and 'trip_start_time' columns.
    """
    _box_by_trip(
        df,
        "start_dev_min",
        "Start‑time deviation – shaded OTP window",
        "box_start_dev_shaded.png",
        shade_range=(OTP_EARLY_MIN, OTP_LATE_MIN),
    )


def plot_start_dev_plain(df: pd.DataFrame) -> None:
    """Box‑plot start‑time deviation without OTP shading.

    Args:
        df: Fully filtered trip‑level data containing
            ``'start_dev_min'`` and ``'trip_start_time'`` columns.
    """
    _box_by_trip(
        df,
        "start_dev_min",
        "Start‑time deviation",
        "box_start_dev.png",
    )


def plot_finish_dev_shaded(df: pd.DataFrame) -> None:
    """Box‑plot finish‑time deviation with the OTP window shaded.

    Args:
        df: Trip‑level DataFrame with ``'finish_dev_min'`` present.
    """
    _box_by_trip(
        df,
        "finish_dev_min",
        "Finish‑time deviation – shaded OTP window",
        "box_finish_dev_shaded.png",
        shade_range=(OTP_EARLY_MIN, OTP_LATE_MIN),  # same thresholds for illustration
    )


def plot_finish_dev_plain(df: pd.DataFrame) -> None:
    """Box‑plot finish‑time deviation without shading.

    Args:
        df: Trip‑level DataFrame with ``'finish_dev_min'`` present.
    """
    _box_by_trip(
        df,
        "finish_dev_min",
        "Finish‑time deviation",
        "box_finish_dev.png",
    )


def plot_runtime_dev(df: pd.DataFrame) -> None:
    """Box‑plot runtime deviation (actual – scheduled) by trip.

    Args:
        df: Trip‑level DataFrame with ``'runtime_dev_min'`` present.
    """
    _box_by_trip(
        df,
        "runtime_dev_min",
        "Runtime deviation by trip",
        "box_runtime_dev.png",
    )


def _day_tag() -> str:
    return (SERVICE_DAY_FILTER or "ALL").upper()


def write_row_level(df: pd.DataFrame) -> None:
    """CSV of events retained for analysis (after all filters)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lead = ["Route", "Direction", "TripID", TIME_COL_NAME]
    ordered = df[lead + [c for c in df.columns if c not in lead]].sort_values(
        by=["Route", "Direction", TIME_COL_NAME, "TripID"],
        ignore_index=True,
    )
    ordered.to_csv(
        OUTPUT_DIR / f"{_RETAINED_STEM}_{_day_tag()}.csv",
        index=False,
    )


def _sched_mode_with_warning(s: pd.Series, trip_id: str | int) -> float:
    """Return the (possibly multi‑modal) mode of *s*, warning if ambiguous."""
    modes = s.mode(dropna=True)
    if modes.empty:
        return float("nan")
    if len(modes) > 1:
        warnings.warn(
            f"TripID {trip_id} has multiple scheduled runtimes "
            f"{modes.tolist()}; using their median.",
            RuntimeWarning,
            stacklevel=2,
        )
    return modes.median() if len(modes) > 1 else modes.iloc[0]


def write_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each start-time token and save an XLSX sheet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trip_grp = df.groupby(TIME_COL_NAME, sort=False)
    summary = trip_grp.agg(
        n_events=("Actual Start Time", lambda s: s.notna().sum()),
        otp_pct=("on_time", lambda s: s.mean() * 100.0),
    )

    summary["scheduled_runtime_mode"] = trip_grp["scheduled_runtime_min"].apply(
        lambda s: _sched_mode_with_warning(s, trip_id=s.name)
    )

    def _runtime_stats(series: pd.Series) -> dict[str, float]:
        data = _trim_pct(series) if TRIM_OUTLIERS else series
        return {
            "runtime_mean_min": data.mean(),
            "runtime_median_min": data.median(),
            "runtime_p85_min": data.quantile(0.85),
        }

    # Built one start-time at a time so the statistics land as columns on every
    # pandas version, rather than depending on how groupby.apply reshapes a
    # Series-returning function.
    _runtime_df = pd.DataFrame.from_dict(
        {token: _runtime_stats(vals) for token, vals in trip_grp["actual_runtime_min"]},
        orient="index",
    )
    _runtime_df.index.name = summary.index.name
    summary = summary.join(_runtime_df)

    # New, clearer columns kept in parallel for one cycle
    summary["pct_within_window"] = summary["otp_pct"]
    summary["under_target"] = summary["otp_pct"] < OTP_TARGET_PCT
    summary["below_target_pct"] = summary["under_target"]

    # Chronological order improves readability
    summary = summary.reset_index(drop=False)
    summary = summary.sort_values(by=TIME_COL_NAME, ignore_index=True)

    summary.to_excel(
        OUTPUT_DIR / f"trip_summary_{_day_tag()}.xlsx",
        index=False,
        engine="openpyxl",
    )
    return summary


def plot_runtime_p85_vs_sched(df: pd.DataFrame) -> None:
    """Bar‐plot scheduled vs. 85th‑percentile runtime for each start time.

    The function computes, for every distinct ``trip_start_time`` token:

    * **Scheduled runtime** – the modal (most common) scheduled runtime
      across trips starting at that time (with ambiguity warnings via
      ``_sched_mode_with_warning``).

    * **85th‑percentile actual runtime** – calculated on actual runtimes
      after optional outlier trimming (controlled by ``TRIM_OUTLIERS`` and
      ``TRIM_FRAC``).

    A grouped bar chart is saved to
    ``PLOTS_DIR / "bar_runtime_p85_vs_sched.png"``.  It visually highlights
    start‑times where the observed P85 runtime exceeds the scheduled value.

    Parameters
    ----------
    df : pandas.DataFrame
        Fully filtered trip‑level data for a single route.

    Notes:
    -----
    * Relies on the global constants ``TIME_COL_NAME``, ``TRIM_OUTLIERS``,
      and ``TRIM_FRAC`` defined earlier in the module.
    * Uses the helper ``_sched_mode_with_warning`` for scheduled runtimes
      and ``_trim_pct`` for optional outlier removal.
    """
    if df.empty:
        logging.warning("   ⚠  No rows after filters; skipping plots.")
        return

    # ── 1.  Aggregate per start‑time token ────────────────────────────
    grp = df.groupby(TIME_COL_NAME, sort=False)

    sched_runtime = grp["scheduled_runtime_min"].apply(
        lambda s: _sched_mode_with_warning(s, trip_id=s.name)
    )

    def _p85(series: pd.Series) -> float:
        data = _trim_pct(series, TRIM_FRAC) if TRIM_OUTLIERS else series
        return data.quantile(0.85)

    p85_runtime = grp["actual_runtime_min"].apply(_p85)

    summary = pd.DataFrame(
        {
            "trip_start_time": sched_runtime.index,
            "scheduled_min": sched_runtime.to_numpy(),
            "p85_min": p85_runtime.to_numpy(),
        }
    ).dropna(subset=["scheduled_min", "p85_min"])

    if summary.empty:
        logging.warning("   ⚠  No valid data for runtime P85 vs. scheduled plot.")
        return

    # ── 2.  Reshape to long format for seaborn ────────────────────────
    tidy = summary.melt(
        id_vars="trip_start_time",
        value_vars=["scheduled_min", "p85_min"],
        var_name="type",
        value_name="runtime_min",
    )

    # ── 3.  Plot ──────────────────────────────────────────────────────
    n_tokens = summary.shape[0]
    fig_w = max(12, n_tokens * 0.45)  # widen for many start‑times
    plt.figure(figsize=(fig_w, 6))
    sns.barplot(
        data=tidy,
        x="trip_start_time",
        y="runtime_min",
        hue="type",
        dodge=True,
    )

    plt.title("Scheduled vs. 85th‑percentile runtime by trip start time")
    plt.xlabel("Scheduled trip start (HH:MM)")
    plt.ylabel("Runtime (minutes)")
    plt.xticks(rotation=90)
    plt.legend(title="", labels=["Scheduled", "85th‑percentile"], loc="upper right")
    plt.tight_layout()

    _ensure_plot_dirs()
    plt.savefig(PLOTS_DIR / "bar_runtime_p85_vs_sched.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# DIRECTION HELPER
# -----------------------------------------------------------------------------

_DIR_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+", flags=re.I)


def _dir_slug(value: str | int | float | None) -> str:
    """Return a filesystem‑safe slug for *value* (e.g., `0_Westbound`)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"

    txt = str(value).strip()
    # Common English labels → deterministic slugs so that 0/1 and text agree
    canonical = {
        "0": "0_Westbound",
        "1": "1_Eastbound",
        "2": "2_Northbound",
        "3": "3_Southbound",
        "WESTBOUND": "0_Westbound",
        "EASTBOUND": "1_Eastbound",
        "NORTHBOUND": "2_Northbound",
        "SOUTHBOUND": "3_Southbound",
    }.get(txt.upper())

    if canonical:
        return canonical

    # Fallback – strip junk and collapse whitespace to “safe” slug
    clean = _DIR_SLUG_RE.sub("_", txt).strip("_")
    return clean or "unknown"


def export_trimmed_outliers(
    df: pd.DataFrame,
    *,
    group_key: str = TIME_COL_NAME,
    runtime_col: str = "actual_runtime_min",
    frac: float = TRIM_FRAC,
) -> None:
    """Write a CSV of rows dropped by the ±*frac* quantile filter."""
    if df.empty or frac <= 0 or not WRITE_EXCLUSIONS:
        return

    keep_frames: list[pd.DataFrame] = []

    for _, sub in df.groupby(group_key, sort=False):
        runtimes = sub[runtime_col]
        if runtimes.empty:
            continue
        lo, hi = runtimes.quantile([frac, 1 - frac])
        mask = (runtimes < lo) | (runtimes > hi)
        if mask.any():
            keep_frames.append(sub.loc[mask])

    if not keep_frames:
        return

    outliers = pd.concat(keep_frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"{_EXCLUDED_STEM}_{_day_tag()}.csv"
    outliers.to_csv(fname, index=False)
    logging.info("   ⤷ %d excluded rows ➜ %s", len(outliers), fname.name)


def log_low_sample_start_times(
    df: pd.DataFrame,
    thresh_frac: float = LOW_SAMPLE_FRAC,
    *,
    exclude_dates: Iterable[str] | None = None,
) -> None:
    """Save a CSV listing start‑times that have *unusually few* observations.

    A start‑time (e.g. “06:15”) is flagged when its row count is **below
    thresh_frac × median(row counts)** across all start‑times in *df*.

    The CSV (one per route) lives alongside the other artefacts and
    contains:

    * `trip_start_time – HH:MM string extracted earlier
    * `n_obs            – observations for that token after filtering
    * `dates_run        – comma‑separated YYYY‑MM‑DD values

    Parameters
    ----------
    df : pandas.DataFrame
        The fully filtered route‑level dataframe.
    thresh_frac : float, default `LOW_SAMPLE_FRAC
        Fraction of the median observation count below which a
        start‑time is considered under‑sampled.
    exclude_dates : Iterable[str] | None, optional
        Your current `EXCLUDE_DATES list (YYYY‑MM‑DD strings).
        Supplying it lets the function tell you which flagged dates are
        *not yet* excluded.
    """
    # ------------------------------------------------------------------ #
    # 1.  Observation counts per start‑time and threshold calculation.   #
    # ------------------------------------------------------------------ #
    counts = df.groupby(TIME_COL_NAME)["Actual Start Time"].count()
    if counts.empty:
        return  # no rows after earlier filters

    cutoff = float(counts.median()) * thresh_frac
    sparse_tokens = counts[counts.lt(cutoff)]
    if sparse_tokens.empty:
        return  # nothing to flag

    # ------------------------------------------------------------------ #
    # 2.  Assemble diagnostic rows, including the dates run.             #
    # ------------------------------------------------------------------ #
    warned_df = (
        df[df[TIME_COL_NAME].isin(sparse_tokens.index)]
        .loc[:, [TIME_COL_NAME, "Scheduled Start Time"]]
        .assign(n_obs=lambda x: x.groupby(TIME_COL_NAME)["Scheduled Start Time"].transform("size"))
    )

    out = (
        warned_df.groupby(TIME_COL_NAME, sort=False)
        .agg(
            n_obs=("n_obs", "first"),
            dates_run=(
                "Scheduled Start Time",
                lambda s: ", ".join(sorted({d.strftime("%Y-%m-%d") for d in s.dt.date})),
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 3.  Write the CSV and emit an actionable console message.          #
    # ------------------------------------------------------------------ #
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"low_sample_start_times_{_day_tag()}.csv"
    out.to_csv(fname, index=False)

    logging.warning(
        "   ⚠  %d low‑sample start‑times logged (<%.0f%% of median obs) ➜ %s",
        len(out),
        thresh_frac * 100,
        fname.name,
    )

    # If an EXCLUDE_DATES list is provided, point out any new dates.
    if exclude_dates is not None:
        flagged_dates: set[str] = {
            d for dates_str in out["dates_run"] for d in dates_str.split(", ")
        }
        missing = flagged_dates - {str(d) for d in exclude_dates}
        if missing:
            logging.warning("      ↪ Consider adding these to EXCLUDE_DATES: %s", sorted(missing))


B2B_COLUMNS: Final[list[str]] = [
    "pattern_key",
    "from_trip_start_time",
    "from_n_obs",
    "from_runtime_median_min",
    "from_scheduled_runtime_min",
    "from_act_sched_ratio",
    "to_trip_start_time",
    "to_n_obs",
    "to_runtime_median_min",
    "to_scheduled_runtime_min",
    "to_act_sched_ratio",
    "gap_min",
    "runtime_diff_min",
    "runtime_ratio",
    "act_sched_ratio_diff",
    "compared",
    "flag_runtime_ratio",
    "flag_runtime_diff",
    "flag_act_sched_ratio",
    "back_to_back_flag",
]


def _representative_value(series: pd.Series) -> str:
    """Return the most common value in *series*, breaking ties alphabetically."""
    vals = series.dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    if vals.empty:
        return ""
    counts = vals.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def flag_back_to_back_runtimes(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    *,
    runtime_ratio: float = B2B_RUNTIME_RATIO,
    runtime_diff_min: float = B2B_RUNTIME_DIFF_MIN,
    sched_ratio_diff: float = B2B_SCHED_RATIO_DIFF,
    max_gap_min: float = B2B_MAX_GAP_MIN,
    min_obs: int = B2B_MIN_OBS,
    require_pattern: bool = B2B_REQUIRE_PATTERN,
) -> pd.DataFrame:
    """Compare each start-time token with the next one on the same pattern.

    Consecutive trips on one pattern should run alike, so a step change between
    neighbors is usually a data or schedule problem rather than real operating
    conditions. Pairs are formed strictly *within* a stop pattern: a short-turn
    legitimately runs far shorter than a full-length trip, so comparing across
    patterns would flag normal service. A pair is flagged when
    any enabled test trips - the median runtimes differ by ``runtime_ratio``, by
    ``runtime_diff_min`` minutes, or the two trips' actual/scheduled runtime
    ratios differ by ``sched_ratio_diff``.

    Args:
        summary: Output of :func:`write_summary_table` for one route/direction.
        df: The row-level trips behind *summary*, used to attach each start-time
            token to its stop pattern.
        runtime_ratio: Longer/shorter median runtime cutoff (0 disables).
        runtime_diff_min: Absolute median-runtime difference in minutes
            (0 disables).
        sched_ratio_diff: Absolute difference between the pair's
            actual/scheduled runtime ratios (0 disables).
        max_gap_min: Skip pairs whose scheduled starts are farther apart than
            this many minutes (0 compares every consecutive pair).
        min_obs: Minimum observations required on *both* trips.
        require_pattern: When True and no pattern column is present, skip the
            check rather than compare trips on different patterns.

    Returns:
        One row per consecutive pair with the gap, the deltas, whether the pair
        was ``compared``, the individual test flags, and ``back_to_back_flag``.
        Empty (with the full column set) when there is nothing to compare.
    """
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in B2B_COLUMNS})
    need = {"trip_start_time", "runtime_median_min", "scheduled_runtime_mode", "n_events"}
    if summary.empty or not need <= set(summary.columns):
        return empty

    pattern_col = next((c for c in B2B_PATTERN_COLS if c in df.columns), None)
    if pattern_col is None:
        if require_pattern:
            logging.warning(
                "   ⚠  Back-to-back check skipped: no pattern/shape column found "
                "(looked for %s). Comparing trips across patterns is meaningless, so set "
                "B2B_REQUIRE_PATTERN = False to compare within the direction anyway.",
                ", ".join(B2B_PATTERN_COLS),
            )
            return empty
        logging.warning(
            "   ⚠  Back-to-back check running without a pattern/shape column; trips on "
            "different patterns may be compared and flagged even when both are correct."
        )
        work = summary.assign(pattern_key="")
    else:
        per_token = (
            df.groupby(TIME_COL_NAME)[pattern_col].agg(_representative_value).rename("pattern_key")
        )
        work = summary.merge(per_token, how="left", left_on="trip_start_time", right_index=True)
        work["pattern_key"] = work["pattern_key"].fillna("").astype(str)

    work = work.assign(
        _t=_hhmm_series_to_minutes(work["trip_start_time"]),
        act_sched_ratio=(work["runtime_median_min"] / work["scheduled_runtime_mode"]).replace(
            [np.inf, -np.inf], np.nan
        ),
    )
    work = work.sort_values(["pattern_key", "_t"], kind="mergesort").reset_index(drop=True)

    nxt = work.groupby("pattern_key", dropna=False, sort=False).shift(-1)
    pairs = pd.DataFrame(
        {
            "pattern_key": work["pattern_key"],
            "from_trip_start_time": work["trip_start_time"],
            "from_n_obs": work["n_events"],
            "from_runtime_median_min": work["runtime_median_min"],
            "from_scheduled_runtime_min": work["scheduled_runtime_mode"],
            "from_act_sched_ratio": work["act_sched_ratio"],
            "to_trip_start_time": nxt["trip_start_time"],
            "to_n_obs": nxt["n_events"],
            "to_runtime_median_min": nxt["runtime_median_min"],
            "to_scheduled_runtime_min": nxt["scheduled_runtime_mode"],
            "to_act_sched_ratio": nxt["act_sched_ratio"],
            "gap_min": nxt["_t"] - work["_t"],
        }
    )
    # The last token on each pattern has no successor.
    pairs = pairs.loc[nxt["trip_start_time"].notna()].reset_index(drop=True)
    if pairs.empty:
        return empty

    runtime_cols = ["from_runtime_median_min", "to_runtime_median_min"]
    pairs["runtime_diff_min"] = pairs["to_runtime_median_min"] - pairs["from_runtime_median_min"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pairs["runtime_ratio"] = (
            pairs[runtime_cols].max(axis=1) / pairs[runtime_cols].min(axis=1)
        ).replace([np.inf, -np.inf], np.nan)
    pairs["act_sched_ratio_diff"] = pairs["to_act_sched_ratio"] - pairs["from_act_sched_ratio"]

    # Which pairs are worth judging at all.
    compared = (pairs["from_n_obs"] >= min_obs) & (pairs["to_n_obs"] >= min_obs)
    if max_gap_min and max_gap_min > 0:
        compared &= pairs["gap_min"].le(max_gap_min)
    pairs["compared"] = compared

    # NaN never satisfies a comparison, so unmeasurable tests simply do not flag.
    zeros = pd.Series(False, index=pairs.index)
    pairs["flag_runtime_ratio"] = (
        (pairs["runtime_ratio"] >= runtime_ratio) if runtime_ratio > 0 else zeros
    ) & compared
    pairs["flag_runtime_diff"] = (
        (pairs["runtime_diff_min"].abs() >= runtime_diff_min) if runtime_diff_min > 0 else zeros
    ) & compared
    pairs["flag_act_sched_ratio"] = (
        (pairs["act_sched_ratio_diff"].abs() >= sched_ratio_diff) if sched_ratio_diff > 0 else zeros
    ) & compared
    pairs["back_to_back_flag"] = (
        pairs["flag_runtime_ratio"] | pairs["flag_runtime_diff"] | pairs["flag_act_sched_ratio"]
    )

    pairs = pairs.sort_values(["pattern_key", "from_trip_start_time"]).reset_index(drop=True)
    return pairs[B2B_COLUMNS].round(
        {
            "gap_min": 1,
            "runtime_diff_min": 2,
            "runtime_ratio": 3,
            "act_sched_ratio_diff": 3,
            "from_runtime_median_min": 2,
            "to_runtime_median_min": 2,
            "from_scheduled_runtime_min": 2,
            "to_scheduled_runtime_min": 2,
            "from_act_sched_ratio": 3,
            "to_act_sched_ratio": 3,
        }
    )


def write_back_to_back_flags(pairs: pd.DataFrame) -> None:
    """Save the flagged back-to-back pairs, if any, as an exception list.

    Nothing is written when no pair trips a threshold, matching the other
    data-quality outputs (a missing file means nothing to review).

    Args:
        pairs: Output of :func:`flag_back_to_back_runtimes`.
    """
    if pairs.empty:
        return

    n_skipped = int((~pairs["compared"].astype(bool)).sum())
    if n_skipped:
        logging.info(
            "   ⤷ %d of %d back-to-back pair(s) not compared (under %d observations "
            "or a scheduled gap over %g min).",
            n_skipped,
            len(pairs),
            B2B_MIN_OBS,
            B2B_MAX_GAP_MIN,
        )

    flagged = pairs.loc[pairs["back_to_back_flag"].astype(bool)]
    if flagged.empty:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"back_to_back_runtime_flags_{_day_tag()}.csv"
    flagged.to_csv(fname, index=False)
    logging.warning(
        "   ⚠  %d back-to-back trip pair(s) flagged (>=%gx or >=%g min apart, or "
        "actual/scheduled ratios >=%g apart) ➜ %s",
        len(flagged),
        B2B_RUNTIME_RATIO,
        B2B_RUNTIME_DIFF_MIN,
        B2B_SCHED_RATIO_DIFF,
        fname.name,
    )


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def extract_config_block(source_file: Path) -> str:
    """Return the text between the CONFIG markers in *source_file*.

    Canonical implementation: ``utils/run_log.py``.

    Args:
        source_file: Path to the Python source file to scan.

    Returns:
        The verbatim text of the configuration block, joined with newlines.

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


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    One sidecar is written per route/direction folder, beside that folder's
    outputs, so every artifact set carries the settings that produced it.

    Args:
        output_dir: Directory holding this route/direction's outputs.
        summary_lines: Human-readable facts about what the run produced.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "trip_runtime_diagnostics_runlog.txt"

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

    lines: List[str] = [
        "=" * 72,
        "TRIP RUNTIME DIAGNOSTICS (AVL) RUN LOG",
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
        logging.info("   ⤷ Run log saved to '%s'.", log_path.name)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


def _cum_sums(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    csum = np.insert(np.cumsum(arr), 0, 0.0)
    csum_sq = np.insert(np.cumsum(arr**2), 0, 0.0)
    return csum, csum_sq


def _ssq(csum: np.ndarray, csum_sq: np.ndarray, i: int, j: int) -> float:
    n = j - i
    if n == 0:
        return 0.0
    s, s2 = csum[j] - csum[i], csum_sq[j] - csum_sq[i]
    return s2 - (s * s) / n


def _fisher_jenks(values: Sequence[float], k: int) -> List[int]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if not 2 <= k <= n:
        raise ValueError("k must be in 2‥n")

    csum, csum_sq = _cum_sums(arr)
    dp = np.full((k, n + 1), np.inf)
    idx = np.zeros((k, n + 1), dtype=int)

    dp[0, 1:] = [_ssq(csum, csum_sq, 0, m) for m in range(1, n + 1)]

    for g in range(1, k):
        for m in range(g + 1, n + 1):
            best, best_s = np.inf, g
            for s in range(g, m):
                cost = dp[g - 1, s] + _ssq(csum, csum_sq, s, m)
                if cost < best:
                    best, best_s = cost, s
            dp[g, m], idx[g, m] = best, best_s

    bkpts, m = [], n
    for g in range(k - 1, 0, -1):
        s = idx[g, m]
        bkpts.append(s)
        m = s
    return sorted(bkpts)  # len == k‑1


def _hhmm_series_to_minutes(s: pd.Series) -> pd.Series:
    """Convert an `HH:MM string to minutes after midnight.

    Accepts hours `00 through 29 so that after‑midnight tokens
    such as `"24:05" parse without error.  Any value that cannot be
    interpreted returns `NaN (float) so it can be removed later with
    `dropna().

    Parameters
    ----------
    s : pandas.Series
        Series of strings in *HH:MM* format.

    Returns:
    -------
    pandas.Series
        Numeric minutes after midnight (float); invalid inputs → NaN.
    """
    parts = s.str.split(":", n=1, expand=True)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    return h * 60 + m


# -----------------------------------------------------------------------------
#  PUBLIC API
# -----------------------------------------------------------------------------
def suggest_time_bands(
    summary: pd.DataFrame,
    *,
    max_bands: int | None = MAX_TIME_BANDS,
    enforce_min_size: bool = ENFORCE_MIN_BAND_SIZE,
    min_band_size: int = MIN_BAND_SIZE,
) -> pd.DataFrame:
    """Create Fisher–Jenks time‑of‑day bands from the 85th‑percentile runtime.

    Handles start‑time tokens up to 29:59 and works with older pandas
    versions that lack `reset_index(names=...).
    """
    need = {"trip_start_time", "runtime_p85_min"}
    miss = need - set(summary.columns)
    if miss:
        raise KeyError(f"summary missing columns {miss}")

    # ── 1. prepare ordered DF with numeric surrogate _t ──────────────
    df = (
        summary[["trip_start_time", "runtime_p85_min"]]
        .assign(_t=_hhmm_series_to_minutes(summary["trip_start_time"]))
        .dropna(subset=["_t", "runtime_p85_min"])
        .sort_values("_t", kind="mergesort")
        .reset_index(drop=True)
    )

    # ── 2. Fisher–Jenks segmentation ─────────────────────────────────
    n = len(df)
    k0 = max(int(np.ceil(np.sqrt(n))), 2)
    k = k0 if max_bands is None or max_bands <= 0 else min(k0, max_bands)

    breaks = _fisher_jenks(df["runtime_p85_min"].to_numpy(), k=k)

    labels = np.zeros(n, dtype=int)
    for _i, b in enumerate(breaks, start=1):
        labels[b:] += 1
    df["_band"] = labels

    # ── 3. optional merge of undersized bands ─────────────────────────
    if enforce_min_size and min_band_size > 1:
        changed = True
        while changed:
            sizes = df["_band"].value_counts().sort_index()
            small = sizes[sizes < min_band_size].index
            if small.empty:
                changed = False
                continue
            for bid in small:
                idx = int(sizes.index.get_loc(bid))
                opts = []
                if idx > 0:
                    left = sizes.index[idx - 1]
                    diff = abs(
                        df.loc[df["_band"] == bid, "runtime_p85_min"].mean()
                        - df.loc[df["_band"] == left, "runtime_p85_min"].mean()
                    )
                    opts.append((left, diff))
                if idx < len(sizes) - 1:
                    right = sizes.index[idx + 1]
                    diff = abs(
                        df.loc[df["_band"] == bid, "runtime_p85_min"].mean()
                        - df.loc[df["_band"] == right, "runtime_p85_min"].mean()
                    )
                    opts.append((right, diff))
                merge_into = min(opts, key=lambda t: t[1])[0]
                df.loc[df["_band"] == bid, "_band"] = merge_into

            # re‑number after merges
            remap = {old: new for new, old in enumerate(sorted(df["_band"].unique()))}
            df["_band"] = df["_band"].map(remap)

    # ── 4. assemble output table (pandas ≤1.5 compatible) ─────────────
    bands = (
        df.groupby("_band", sort=True, observed=True)
        .agg(
            start_time=("trip_start_time", "first"),
            end_time=("trip_start_time", "last"),
            n_tokens=("trip_start_time", "size"),
            p85_mean_min=("runtime_p85_min", "mean"),
        )
        .reset_index()  # '_band' -> column
        .rename(columns={"_band": "band_id"})
        .assign(band_id=lambda x: x["band_id"] + 1)
    )

    return bands


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:  # pragma: no cover
    """Run the end-to-end analysis for every route.

    * If ``SPLIT_BY_DIRECTION`` is **True** (default), rows are subdivided
      by the ``Direction`` column and written to
      ``<OUTPUT_ROOT>/<route>/<direction-slug>/``.
    * If **False**, all rows for the route are written directly to
      ``<OUTPUT_ROOT>/<route>/``.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # ------------------------------------------------------------------ #
    # 0.  Locate all CSVs and assign them to routes                      #
    # ------------------------------------------------------------------ #
    whitelist = {r.lstrip("0") for r in ROUTES_TO_INCLUDE} if ROUTES_TO_INCLUDE else None

    if INPUT_ROOT_DIR == Path(
        r"Path\To\Your\Data_Folder_with_observed_trips"
    ) or OUTPUT_ROOT_DIR == Path(r"Path\To\Your\Output_Folder"):
        logging.warning(
            "INPUT_ROOT_DIR and/or OUTPUT_ROOT_DIR are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2

    if not INPUT_ROOT_DIR.exists():
        logging.warning(
            "INPUT_ROOT_DIR does not exist: %s — update INPUT_ROOT_DIR in the CONFIGURATION "
            "section to your actual folder of observed trip CSV files before running.",
            INPUT_ROOT_DIR,
        )
        logging.info("Completed (no data processed — update INPUT_ROOT_DIR to proceed).")
        return 1

    logging.info("→ Crawling %s for CSVs …", INPUT_ROOT_DIR)
    route_files = _discover_route_csvs(INPUT_ROOT_DIR, whitelist)

    if not route_files:
        logging.warning(
            "No CSV files found under %s — ensure the folder contains CLEVER or TIDES "
            "trip export CSVs and that INPUT_ROOT_DIR points to the correct location.",
            INPUT_ROOT_DIR,
        )
        logging.info("Completed (no data processed — no CSV files found).")
        return 1

    # ------------------------------------------------------------------ #
    # 1.  Process each route                                             #
    # ------------------------------------------------------------------ #
    for route, paths in sorted(route_files.items(), key=lambda t: int(t[0])):
        logging.info("— Processing route %s (%d files) …", route, len(paths))

        # ── 1 a.  Load + global filters (shared before direction split) ──
        base_df = (
            load_trip_files(paths)
            .pipe(extract_trip_start_time)
            .pipe(filter_date_range)
            .pipe(filter_routes, {route})
            .pipe(filter_holidays, EXCLUDE_DATES)
            .pipe(filter_service_day, SERVICE_DAY_FILTER)
            .pipe(add_deviation_cols)
            .pipe(add_otp_flag)  # adds both on_time & within_window
        )

        # Conditionally normalize directions (legacy only)
        # For TIDES, we preserve the 0/1 IDs as-is.
        if not base_df.get("_is_tides", pd.Series([False] * len(base_df))).any():
            base_df = normalize_directions_column(base_df, ALLOWED_DIRECTIONS)

        # Drop the helper column if present
        if "_is_tides" in base_df.columns:
            base_df = base_df.drop(columns=["_is_tides"])

        if base_df.empty:
            logging.warning("   ⚠  No rows left after filtering; skipping route.")
            continue

        # ── 1 b.  Quick sanity printout — median obs per start-time ──────
        logging.info(
            "   observation median: %s",
            base_df.groupby(TIME_COL_NAME)["Actual Start Time"].count().median(),
        )
        log_low_sample_start_times(
            base_df,
            thresh_frac=LOW_SAMPLE_FRAC,
            exclude_dates=EXCLUDE_DATES,
        )

        # ------------------------------------------------------------------
        # 2.  Split by Direction (optional)                                #
        # ------------------------------------------------------------------
        if SPLIT_BY_DIRECTION:
            if "Direction" not in base_df.columns:
                logging.warning(
                    "   ⚠  'Direction' column missing; treating all rows as one direction."
                )
                base_df["Direction"] = "unknown"
            dir_groups = base_df.groupby("Direction", sort=False)
        else:
            # Flatten: treat everything as a single pseudo-direction “all”
            dir_groups = [("all", base_df)]

        for dir_val, dir_df in dir_groups:
            if dir_df.empty:
                continue

            # ---- 2 a.  Sanitise direction for folder names ---------------
            dir_slug = re.sub(r"[^0-9A-Za-z_-]+", "_", str(dir_val)).strip("_") or "all"

            # ---- 2 b.  Point writers/plotters at the correct folder ------
            global OUTPUT_DIR, PLOTS_DIR
            OUTPUT_DIR = (
                OUTPUT_ROOT_DIR / route / dir_slug
                if SPLIT_BY_DIRECTION
                else OUTPUT_ROOT_DIR / route
            )
            PLOTS_DIR = OUTPUT_DIR / "plots"

            logging.info(
                "   ↳ Direction '%s': %d rows ➜ %s",
                dir_val,
                len(dir_df),
                OUTPUT_DIR.relative_to(OUTPUT_ROOT_DIR),
            )

            # ---- 2 c.  Persist outliers (optional) -----------------------
            if TRIM_OUTLIERS:
                export_trimmed_outliers(dir_df)

            # ---- 2 d.  Core exports --------------------------------------
            write_row_level(dir_df)
            summary = write_summary_table(dir_df)
            b2b_pairs = flag_back_to_back_runtimes(summary, dir_df)
            write_back_to_back_flags(b2b_pairs)
            bands = suggest_time_bands(summary)
            bands.to_excel(
                OUTPUT_DIR / f"time_bands_{_day_tag()}.xlsx",
                index=False,
                engine="openpyxl",
            )
            logging.info("      → Suggested %d time bands saved.", len(bands))

            # ---- 2 e.  Plots ---------------------------------------------
            plot_funcs = [
                plot_start_dev_shaded,
                plot_start_dev_plain,
                plot_finish_dev_shaded,
                plot_finish_dev_plain,
                plot_runtime_dev,
                plot_runtime_p85_vs_sched,
            ]
            for func in plot_funcs:
                _safe_plot(func, dir_df)

            # ---- 2 f.  Run-log sidecar -----------------------------------
            n_b2b_flagged = int(b2b_pairs["back_to_back_flag"].sum()) if not b2b_pairs.empty else 0
            n_b2b_compared = int(b2b_pairs["compared"].sum()) if not b2b_pairs.empty else 0
            summary_lines = [
                f"Route / direction:  {route} / {dir_val}",
                f"Rows retained:      {len(dir_df)}",
                f"Start-time tokens:  {len(summary)}",
                f"Under OTP target:   "
                f"{int(summary['under_target'].sum()) if not summary.empty else 0}",
                f"Back-to-back pairs: {len(b2b_pairs)} "
                f"({n_b2b_compared} compared, {n_b2b_flagged} flagged)",
                f"Time bands:         {len(bands)}",
            ]
            if not write_run_log(OUTPUT_DIR, summary_lines) and REQUIRE_RUN_LOG:
                logging.error(
                    "Run log could not be written for %s. Set REQUIRE_RUN_LOG = False to "
                    "continue anyway when a sidecar file is genuinely impossible.",
                    OUTPUT_DIR,
                )
                return 1

            logging.info("      ✓ Direction '%s' done.", dir_val)

        logging.info("✓ Finished route %s", route)

    logging.info("✓✓ All routes processed.")
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
