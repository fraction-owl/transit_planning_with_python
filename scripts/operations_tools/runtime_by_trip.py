"""Per-trip running-time statistics, variation flags, and data diagnostics.

This script derives observed trip running times from a TIDES ``stop_visits``
table, joins route/direction context from ``trips_performed``, and summarizes
each recurring scheduled trip (``trip_id_scheduled``) across all the dates it
ran. For every such trip it reports the mean, median, standard deviation, and
coefficient of variation of the running time, and flags trips whose runtime is
unusually variable.

Three data-quality diagnostics are produced as well:

  * **Apparent data gaps** - scheduled trips with far fewer observations than
    their peers (a sign of missing AVL/feed coverage), flagged relative to the
    median observation count.
  * **Day-of-week anomalies** - per trip, day-of-week buckets that either carry
    very little data (e.g. only a couple of Mondays observed) or whose mean
    running time departs noticeably from the trip's overall mean (e.g. Monday
    trips running materially longer/shorter than Tue-Fri). Both are common
    artifacts of pooling weekdays that actually behave differently.
  * **Back-to-back trip anomalies** - consecutive trips on the same route,
    direction, *and* stop pattern should behave alike; a step change
    between neighbors usually means bad data or a broken schedule rather than
    real operating conditions. Each trip is compared with the next one on its
    pattern and flagged when the median running times differ by a large ratio
    (e.g. 2x) or a large number of minutes (e.g. 20), or when the two trips'
    actual/scheduled runtime ratios diverge. Comparing across patterns is
    meaningless (a short-turn legitimately runs far shorter than a full-length
    trip), so pairs are only formed within a pattern. Pairs whose scheduled
    starts are far apart are skipped, since a genuine peak/midday runtime
    difference is not a data problem.

Outlier trimming is applied per trip before statistics are computed: the
shortest and longest ``TRIM_FRAC`` of observations are dropped (default 1%; set
to 0.05 for 5%, etc.). The trimmed rows are written to their own CSV so nothing
is silently discarded.

Outputs:
  * ``trip_runtime_observations.csv``  - retained per-trip-per-date runtimes.
  * ``trip_runtime_outliers.csv``      - rows removed by trimming.
  * ``trip_runtime_stats.csv``         - per-trip summary statistics + flags.
  * ``trip_runtime_dow.csv``           - per-trip-per-DOW counts/means + flags.
  * ``trip_runtime_back_to_back.csv``  - every consecutive same-pattern trip
    pair, with the gap, the runtime deltas, and the flags below.
  * PNG charts per route: runtime boxplot by trip start, and mean runtime by
    day of week.

Typical usage
-------------
Update the CONFIGURATION paths (or pass the matching CLI flags) and run from a
shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Sentinel markers used by extract_config_block / write_run_log to identify the
# CONFIGURATION block that is copied verbatim into the run-log sidecar.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================

# === BEGIN CONFIG ===

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Outlier trimming: drop the shortest and longest TRIM_FRAC of runtimes per trip
# before computing statistics. 0.01 = 1%, 0.05 = 5%. Set 0 to disable.
TRIM_FRAC: float = 0.01

# A trip is flagged "high variation" when its coefficient of variation
# (std / mean of runtime) exceeds this threshold and it has enough observations.
HIGH_CV_THRESHOLD: float = 0.15
MIN_OBS_FOR_CV: int = 5

# Apparent data gap: a trip is flagged when its observation count is below
# GAP_FRAC * median(observation counts) across all trips.
GAP_FRAC: float = 0.30

# Day-of-week anomaly thresholds.
#   * A (trip, DOW) bucket is "low count" when its observations are below
#     DOW_LOW_COUNT_FRAC * the trip's median per-DOW count.
#   * A bucket's runtime is "anomalous" when its mean differs from the trip's
#     overall mean by more than DOW_RUNTIME_PCT (fraction).
DOW_LOW_COUNT_FRAC: float = 0.40
DOW_RUNTIME_PCT: float = 0.10

# Back-to-back trip checks. Each trip is compared with the *next* scheduled trip
# on the same route, direction, and stop pattern (``pattern_id``). Neighboring
# trips on one pattern should behave alike, so a step change is normally a data or
# schedule problem rather than real operating conditions.
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
#   * B2B_MIN_OBS          - both trips need at least this many retained
#     observations before the pair is compared.
# A pair is flagged when *any* enabled test trips. Set an individual threshold to
# 0 to disable just that test.
B2B_RUNTIME_RATIO: float = 2.0
B2B_RUNTIME_DIFF_MIN: float = 20.0
B2B_SCHED_RATIO_DIFF: float = 0.50
B2B_MAX_GAP_MIN: float = 90.0
B2B_MIN_OBS: int = 3

# Pairing trips across stop patterns is meaningless, so when no usable
# ``pattern_id`` is available the check is skipped rather than run on mixed
# patterns. Set B2B_REQUIRE_PATTERN to False to compare within route/direction
# anyway, which pools branching variants and can flag correct service.
B2B_REQUIRE_PATTERN: bool = True

# Optional route filters (matched against route_id as a string). Empty = all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# Abort when the run-log sidecar cannot be written, so an output is never left
# without its record. Set False only for genuinely read-only destinations.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO

DOW_ORDER: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# === END CONFIG ===

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a trip-runtime flagging run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    trim_frac: float = TRIM_FRAC
    high_cv_threshold: float = HIGH_CV_THRESHOLD
    min_obs_for_cv: int = MIN_OBS_FOR_CV
    gap_frac: float = GAP_FRAC
    dow_low_count_frac: float = DOW_LOW_COUNT_FRAC
    dow_runtime_pct: float = DOW_RUNTIME_PCT
    b2b_runtime_ratio: float = B2B_RUNTIME_RATIO
    b2b_runtime_diff_min: float = B2B_RUNTIME_DIFF_MIN
    b2b_sched_ratio_diff: float = B2B_SCHED_RATIO_DIFF
    b2b_max_gap_min: float = B2B_MAX_GAP_MIN
    b2b_min_obs: int = B2B_MIN_OBS
    b2b_require_pattern: bool = B2B_REQUIRE_PATTERN
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()


# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read ``stop_visits`` and parse timestamps + numeric sequence."""
    df = pd.read_csv(path, dtype=str)
    for col in (
        "actual_arrival_time",
        "actual_departure_time",
        "schedule_arrival_time",
        "schedule_departure_time",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["trip_stop_sequence"] = pd.to_numeric(df["trip_stop_sequence"], errors="coerce")
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read ``trips_performed`` and parse the scheduled start/end timestamps."""
    df = pd.read_csv(path, dtype=str)
    for col in ("schedule_trip_start", "schedule_trip_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def compute_trip_runtimes(stop_visits: pd.DataFrame) -> pd.DataFrame:
    """Derive observed running time for each performed trip.

    The running time is the last usable actual arrival minus the first usable
    actual departure on the trip. Skipped stop visits (no actual time) are
    excluded before picking the endpoints, so a trip's runtime spans its first
    to last *served* stop.

    The scheduled running time is measured between those same two endpoints
    (the schedule times on the first and last served stop) rather than over the
    whole booked trip, so the actual/scheduled ratio is not skewed low whenever
    AVL coverage stops short of the terminals.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.

    Returns:
        DataFrame with one row per ``trip_id_performed`` and columns
        ``service_date``, ``start_time`` (first actual departure),
        ``actual_runtime_min`` (NaN when fewer than two usable stops exist),
        ``scheduled_runtime_min`` (NaN when the endpoints carry no schedule
        times), and ``pattern_id`` when the visits carry one.
    """
    work = stop_visits
    if "schedule_relationship" in work.columns:
        work = work.loc[work["schedule_relationship"].fillna("Scheduled") != "Skipped"]
    work = work.dropna(subset=["actual_arrival_time", "actual_departure_time"], how="all")
    work = work.sort_values(["trip_id_performed", "trip_stop_sequence"])

    has_sched = {"schedule_departure_time", "schedule_arrival_time"} <= set(work.columns)
    has_pattern = "pattern_id" in work.columns

    rows: List[Dict[str, object]] = []
    for trip_id, g in work.groupby("trip_id_performed", sort=False):
        # Row labels of the endpoints, so schedule times come from the same stops.
        dep_idx = g["actual_departure_time"].first_valid_index()
        arr_idx = g["actual_arrival_time"].last_valid_index()
        if dep_idx is None or arr_idx is None:
            continue
        start = g.loc[dep_idx, "actual_departure_time"]
        end = g.loc[arr_idx, "actual_arrival_time"]
        runtime = (end - start).total_seconds() / 60.0

        sched_runtime = float("nan")
        if has_sched:
            sched_start = g.loc[dep_idx, "schedule_departure_time"]
            sched_end = g.loc[arr_idx, "schedule_arrival_time"]
            if pd.notna(sched_start) and pd.notna(sched_end):
                sched_runtime = (sched_end - sched_start).total_seconds() / 60.0

        row: Dict[str, object] = {
            "trip_id_performed": trip_id,
            "service_date": g["service_date"].iloc[0],
            "start_time": start,
            "actual_runtime_min": runtime,
            "scheduled_runtime_min": sched_runtime,
        }
        if has_pattern:
            # One pattern per performed trip; take the first non-null visit.
            pattern = g["pattern_id"].dropna()
            row["pattern_id"] = pattern.iloc[0] if not pattern.empty else pd.NA
        rows.append(row)

    return pd.DataFrame(rows)


def join_trip_attributes(
    trip_runtimes: pd.DataFrame, trips_performed: pd.DataFrame
) -> pd.DataFrame:
    """Attach route/direction/scheduled-trip context and a day-of-week column.

    Canceled / non-in-service trips are dropped via the join. Adds:
    ``route_id``, ``direction_id``, ``trip_id_scheduled``, ``start_hhmm`` (the
    *scheduled* start label, stable across dates, used to order trips), ``dow``
    (day name), and ``pattern_key`` (the stop pattern the trip ran; empty when
    neither table carries a usable ``pattern_id``).

    ``pattern_id`` is preferred from ``stop_visits`` and only then from
    ``trips_performed``: it is optional in TIDES, and the converters in this
    folder synthesize it on stop visits while leaving it blank on trips.

    The start label is taken from the scheduled trip start rather than the
    observed departure, so the same recurring trip carries one consistent label
    on every date it ran (the actual departure jitters by seconds day to day).

    Any trip whose scheduled runtime could not be measured from the stop-level
    schedule times falls back to the booked trip start/end on
    ``trips_performed``.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [
        c
        for c in (
            "route_id",
            "direction_id",
            "trip_id_scheduled",
            "route_type_agency",
            "schedule_trip_start",
            "schedule_trip_end",
            "pattern_id",
        )
        if c in trips.columns
    ]
    trips_small = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")

    merged = trip_runtimes.merge(
        trips_small, on="trip_id_performed", how="inner", suffixes=("", "_trips")
    )
    merged["dow"] = merged["service_date"].dt.day_name()

    # Stop pattern the trip ran; pairing trips across patterns is meaningless.
    blank = pd.Series("", index=merged.index)
    pattern = blank
    for col in ("pattern_id", "pattern_id_trips"):
        if col in merged.columns:
            values = merged[col].astype("string").fillna("").str.strip()
            pattern = pattern.where(pattern != "", values)
    merged["pattern_key"] = pattern.fillna("")

    # Fall back to the booked trip span when the stop-level schedule was missing.
    if "scheduled_runtime_min" not in merged.columns:
        merged["scheduled_runtime_min"] = float("nan")
    if {"schedule_trip_start", "schedule_trip_end"} <= set(merged.columns):
        booked = (
            merged["schedule_trip_end"] - merged["schedule_trip_start"]
        ).dt.total_seconds() / 60.0
        merged["scheduled_runtime_min"] = merged["scheduled_runtime_min"].fillna(booked)

    # Prefer the scheduled start for the (stable) label; fall back to actual.
    if "schedule_trip_start" in merged.columns:
        sched_hhmm = merged["schedule_trip_start"].dt.strftime("%H:%M")
    else:
        sched_hhmm = pd.Series(pd.NA, index=merged.index)
    merged["start_hhmm"] = sched_hhmm.fillna(merged["start_time"].dt.strftime("%H:%M"))

    # Trip key falls back to a route/direction/start label when no scheduled id.
    if "trip_id_scheduled" in merged.columns:
        merged["trip_key"] = merged["trip_id_scheduled"].astype(str)
    else:
        merged["trip_key"] = (
            merged["route_id"].astype(str)
            + "_"
            + merged["direction_id"].astype(str)
            + "_"
            + merged["start_hhmm"].astype(str)
        )
    return merged


# =============================================================================
# OUTLIER TRIMMING
# =============================================================================


def trim_outliers(
    df: pd.DataFrame,
    frac: float = TRIM_FRAC,
    group_col: str = "trip_key",
    value_col: str = "actual_runtime_min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations into retained and trimmed-outlier frames.

    Within each ``group_col`` group, observations below the ``frac`` quantile or
    above the ``1 - frac`` quantile of ``value_col`` are treated as outliers.

    Args:
        df: Per-trip-per-date observations.
        frac: Tail fraction to trim from each end (0 disables trimming).
        group_col: Column identifying each recurring trip.
        value_col: Numeric column to trim on.

    Returns:
        ``(retained, outliers)`` DataFrames. ``outliers`` is empty when
        ``frac <= 0`` or no rows fall in the tails.
    """
    if frac <= 0 or df.empty:
        return df.copy(), df.iloc[0:0].copy()

    keep_mask = pd.Series(True, index=df.index)
    for _, sub in df.groupby(group_col, sort=False):
        vals = sub[value_col]
        if vals.notna().sum() < 2:
            continue
        lo = vals.quantile(frac)
        hi = vals.quantile(1 - frac)
        out = (vals < lo) | (vals > hi)
        keep_mask.loc[sub.index[out.to_numpy()]] = False

    retained = df.loc[keep_mask].copy()
    outliers = df.loc[~keep_mask].copy()
    return retained, outliers


# =============================================================================
# STATISTICS & FLAGS
# =============================================================================


def _representative_label(series: pd.Series) -> str:
    """Return the most common start label, breaking ties by the earliest time."""
    s = series.dropna().astype(str)
    if s.empty:
        return ""
    counts = s.value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def compute_trip_stats(
    df: pd.DataFrame,
    high_cv_threshold: float = HIGH_CV_THRESHOLD,
    min_obs_for_cv: int = MIN_OBS_FOR_CV,
    gap_frac: float = GAP_FRAC,
) -> pd.DataFrame:
    """Compute per-trip runtime statistics and quality flags.

    Args:
        df: Retained observations (after trimming) with a ``trip_key`` column.
        high_cv_threshold: Coefficient-of-variation cutoff for the
            ``high_variation`` flag.
        min_obs_for_cv: Minimum observations before a CV flag is trusted.
        gap_frac: Fraction of the median observation count below which a trip is
            flagged as a likely data gap.

    Returns:
        DataFrame with one row per ``trip_key`` and statistics + boolean flags
        ``high_variation`` and ``data_gap``. Also carries the trip's
        ``pattern_key``, its median ``sched_runtime_median_min``, and
        ``act_sched_ratio`` (median actual over median scheduled runtime), which
        :func:`compute_back_to_back_flags` compares between neighbors.
    """
    if df.empty:
        return df.copy()

    aggs = {
        "start_hhmm": ("start_hhmm", _representative_label),
        "n_obs": ("actual_runtime_min", "count"),
        "runtime_mean_min": ("actual_runtime_min", "mean"),
        "runtime_median_min": ("actual_runtime_min", "median"),
        "runtime_std_min": ("actual_runtime_min", "std"),
        "runtime_min_min": ("actual_runtime_min", "min"),
        "runtime_max_min": ("actual_runtime_min", "max"),
    }
    if "pattern_key" in df.columns:
        aggs["pattern_key"] = ("pattern_key", _representative_label)
    if "scheduled_runtime_min" in df.columns:
        aggs["sched_runtime_median_min"] = ("scheduled_runtime_min", "median")

    grouped = df.groupby(["route_id", "direction_id", "trip_key"], dropna=False)
    stats = grouped.agg(**aggs).reset_index()

    if "pattern_key" not in stats.columns:
        stats["pattern_key"] = ""
    if "sched_runtime_median_min" not in stats.columns:
        stats["sched_runtime_median_min"] = float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        stats["cv"] = stats["runtime_std_min"] / stats["runtime_mean_min"]
        stats["act_sched_ratio"] = (
            stats["runtime_median_min"] / stats["sched_runtime_median_min"]
        ).replace([np.inf, -np.inf], np.nan)

    stats["high_variation"] = (stats["cv"] > high_cv_threshold) & (stats["n_obs"] >= min_obs_for_cv)

    median_obs = stats["n_obs"].median()
    cutoff = median_obs * gap_frac if pd.notna(median_obs) else 0.0
    stats["data_gap"] = stats["n_obs"] < cutoff

    stats = stats.sort_values(["route_id", "direction_id", "start_hhmm"]).reset_index(drop=True)
    return stats.round(
        {
            "runtime_mean_min": 2,
            "runtime_median_min": 2,
            "runtime_std_min": 2,
            "sched_runtime_median_min": 2,
            "cv": 3,
            "act_sched_ratio": 3,
        }
    )


def compute_dow_anomalies(
    df: pd.DataFrame,
    low_count_frac: float = DOW_LOW_COUNT_FRAC,
    runtime_pct: float = DOW_RUNTIME_PCT,
) -> pd.DataFrame:
    """Flag day-of-week buckets with sparse data or anomalous runtimes.

    For each ``trip_key`` the overall mean runtime is compared against the mean
    within each day-of-week. A bucket is flagged ``low_count`` when its
    observation count is far below the trip's typical per-DOW count, and
    ``runtime_anomaly`` when its mean runtime differs from the trip's overall
    mean by more than ``runtime_pct``.

    Args:
        df: Retained observations with ``trip_key`` and ``dow`` columns.
        low_count_frac: Threshold (fraction of the trip's median per-DOW count).
        runtime_pct: Relative runtime-deviation threshold (fraction).

    Returns:
        DataFrame with one row per ``(trip_key, dow)`` and the diagnostic
        columns ``n_obs``, ``dow_mean_min``, ``trip_mean_min``, ``pct_diff``,
        ``low_count``, ``runtime_anomaly``.
    """
    if df.empty:
        return df.copy()

    trip_mean = df.groupby("trip_key")["actual_runtime_min"].transform("mean")
    work = df.assign(_trip_mean=trip_mean)

    grouped = work.groupby(["route_id", "direction_id", "trip_key", "dow"], dropna=False)
    out = grouped.agg(
        n_obs=("actual_runtime_min", "count"),
        dow_mean_min=("actual_runtime_min", "mean"),
        trip_mean_min=("_trip_mean", "first"),
    ).reset_index()

    median_dow_count = out.groupby("trip_key")["n_obs"].transform("median")
    out["low_count"] = out["n_obs"] < (median_dow_count * low_count_frac)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["pct_diff"] = (out["dow_mean_min"] - out["trip_mean_min"]) / out["trip_mean_min"]
    out["runtime_anomaly"] = out["pct_diff"].abs() > runtime_pct

    # Order day names naturally where possible.
    out["dow"] = pd.Categorical(out["dow"], categories=DOW_ORDER, ordered=True)
    out = out.sort_values(["route_id", "direction_id", "trip_key", "dow"]).reset_index(drop=True)
    return out.round({"dow_mean_min": 2, "trip_mean_min": 2, "pct_diff": 3})


# =============================================================================
# BACK-TO-BACK TRIP CHECKS
# =============================================================================

B2B_COLUMNS: List[str] = [
    "route_id",
    "direction_id",
    "pattern_key",
    "from_trip_key",
    "from_start_hhmm",
    "from_n_obs",
    "from_runtime_median_min",
    "from_act_sched_ratio",
    "to_trip_key",
    "to_start_hhmm",
    "to_n_obs",
    "to_runtime_median_min",
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


def _hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """Convert an ``HH:MM`` label to minutes after midnight.

    Hours beyond 23 (e.g. ``"24:05"`` for an after-midnight trip) parse fine, so
    owl trips stay in schedule order. Anything unparseable becomes NaN.

    Args:
        s: Series of ``HH:MM`` strings.

    Returns:
        Float Series of minutes after midnight; invalid inputs are NaN.
    """
    parts = s.astype(str).str.split(":", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=s.index, dtype=float)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    return hours * 60 + minutes


def compute_back_to_back_flags(
    stats: pd.DataFrame,
    runtime_ratio: float = B2B_RUNTIME_RATIO,
    runtime_diff_min: float = B2B_RUNTIME_DIFF_MIN,
    sched_ratio_diff: float = B2B_SCHED_RATIO_DIFF,
    max_gap_min: float = B2B_MAX_GAP_MIN,
    min_obs: int = B2B_MIN_OBS,
    require_pattern: bool = B2B_REQUIRE_PATTERN,
) -> pd.DataFrame:
    """Compare each trip with the next one on its route/direction/pattern.

    Consecutive trips on the same pattern should run alike, so a step change
    between neighbors is usually a data or schedule problem. Pairs are formed
    strictly *within* a stop pattern: a short-turn legitimately runs far
    shorter than a full-length trip, so comparing across patterns would flag
    normal service. A pair is flagged when any enabled test trips - the runtimes
    differ by ``runtime_ratio``, by ``runtime_diff_min`` minutes, or the two
    trips' actual/scheduled ratios differ by ``sched_ratio_diff``.

    Args:
        stats: Output of :func:`compute_trip_stats`.
        runtime_ratio: Longer/shorter median runtime cutoff (0 disables).
        runtime_diff_min: Absolute median-runtime difference in minutes
            (0 disables).
        sched_ratio_diff: Absolute difference between the pair's
            actual/scheduled runtime ratios (0 disables).
        max_gap_min: Skip pairs whose scheduled starts are farther apart than
            this many minutes (0 compares every consecutive pair).
        min_obs: Minimum retained observations required on *both* trips.
        require_pattern: When True and no usable ``pattern_id`` was available,
            skip the check rather than compare trips on different patterns.

    Returns:
        One row per consecutive pair with the gap, the deltas, whether the pair
        was ``compared``, the individual test flags, and ``back_to_back_flag``.
        Empty (with the full column set) when there is nothing to compare.
    """
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in B2B_COLUMNS})
    if stats.empty:
        return empty

    work = stats.copy()
    if "pattern_key" not in work.columns:
        work["pattern_key"] = ""
    work["pattern_key"] = work["pattern_key"].fillna("").astype(str)

    if not (work["pattern_key"].str.strip() != "").any():
        if require_pattern:
            logging.warning(
                "Back-to-back check skipped: neither stop_visits nor trips_performed has a "
                "usable pattern_id. Comparing trips across stop patterns is meaningless, so "
                "set B2B_REQUIRE_PATTERN = False to compare within route/direction anyway."
            )
            return empty
        logging.warning(
            "Back-to-back check running without pattern_id; branching variants within a "
            "direction are pooled, so correct trips may be compared and flagged."
        )

    if "act_sched_ratio" not in work.columns:
        work["act_sched_ratio"] = float("nan")

    work["_start_min"] = _hhmm_to_minutes(work["start_hhmm"])
    group_cols = ["route_id", "direction_id", "pattern_key"]
    work = work.sort_values([*group_cols, "_start_min"], kind="mergesort").reset_index(drop=True)

    nxt = work.groupby(group_cols, dropna=False, sort=False).shift(-1)
    pairs = pd.DataFrame(
        {
            "route_id": work["route_id"],
            "direction_id": work["direction_id"],
            "pattern_key": work["pattern_key"],
            "from_trip_key": work["trip_key"],
            "from_start_hhmm": work["start_hhmm"],
            "from_n_obs": work["n_obs"],
            "from_runtime_median_min": work["runtime_median_min"],
            "from_act_sched_ratio": work["act_sched_ratio"],
            "to_trip_key": nxt["trip_key"],
            "to_start_hhmm": nxt["start_hhmm"],
            "to_n_obs": nxt["n_obs"],
            "to_runtime_median_min": nxt["runtime_median_min"],
            "to_act_sched_ratio": nxt["act_sched_ratio"],
            "gap_min": nxt["_start_min"] - work["_start_min"],
        }
    )
    # The last trip on each pattern has no successor.
    pairs = pairs.loc[nxt["trip_key"].notna()].reset_index(drop=True)
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

    pairs = pairs.sort_values(
        ["route_id", "direction_id", "pattern_key", "from_start_hhmm"]
    ).reset_index(drop=True)
    return pairs[B2B_COLUMNS].round(
        {
            "gap_min": 1,
            "runtime_diff_min": 2,
            "runtime_ratio": 3,
            "act_sched_ratio_diff": 3,
            "from_runtime_median_min": 2,
            "to_runtime_median_min": 2,
            "from_act_sched_ratio": 3,
            "to_act_sched_ratio": 3,
        }
    )


def apply_back_to_back_flag(stats: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Mark every trip that takes part in a flagged back-to-back pair.

    Args:
        stats: Output of :func:`compute_trip_stats`.
        pairs: Output of :func:`compute_back_to_back_flags`.

    Returns:
        Copy of *stats* with a boolean ``back_to_back_flag`` column.
    """
    out = stats.copy()
    if out.empty:
        out["back_to_back_flag"] = pd.Series(dtype=bool)
        return out
    if pairs.empty:
        out["back_to_back_flag"] = False
        return out

    flagged = pairs.loc[pairs["back_to_back_flag"].astype(bool)]
    keys = set(flagged["from_trip_key"]) | set(flagged["to_trip_key"])
    out["back_to_back_flag"] = out["trip_key"].isin(keys)
    return out


# =============================================================================
# PLOTS
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


def plot_route_runtime_box(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Boxplot of trip runtime, one box per scheduled trip, per route/direction.

    Each box pools all observed runtimes for one ``trip_key``; boxes are ordered
    and labeled by the trip's representative scheduled start time.
    """
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []

    for (route, direction), g in df.groupby(["route_id", "direction_id"], dropna=False):
        # One box per scheduled trip, ordered by its representative start time.
        trip_boxes: List[tuple[str, object]] = []
        for _trip_key, tg in g.groupby("trip_key", dropna=False):
            vals = tg["actual_runtime_min"].dropna().to_numpy()
            if len(vals):
                label = _representative_label(tg["start_hhmm"])
                trip_boxes.append((label, vals))
        if not trip_boxes:
            continue
        trip_boxes.sort(key=lambda t: t[0])
        labels = [lbl for lbl, _ in trip_boxes]
        data = [vals for _, vals in trip_boxes]
        plt.figure(figsize=(max(8, len(labels) * 0.5), 5))
        positions = range(1, len(data) + 1)
        plt.boxplot(data, positions=positions)
        plt.xticks(list(positions), labels, rotation=90)
        plt.title(f"Route {route} dir {direction} - runtime by trip start")
        plt.ylabel("Runtime (minutes)")
        plt.xlabel("Scheduled start (HH:MM)")
        plt.tight_layout()
        p = plots_dir / f"runtime_box_{route}_dir{direction}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        written.append(p)

    return written


def plot_dow_runtime(dow_table: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Bar chart of mean runtime by day of week, one PNG per route/direction."""
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []
    if dow_table.empty:
        return written

    for (route, direction), g in dow_table.groupby(["route_id", "direction_id"], dropna=False):
        means = g.groupby("dow", observed=True)["dow_mean_min"].mean().reindex(DOW_ORDER).dropna()
        if means.empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.bar(means.index.astype(str), means.to_numpy())
        plt.title(f"Route {route} dir {direction} - mean runtime by day of week")
        plt.ylabel("Mean runtime (minutes)")
        plt.xlabel("Day of week")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        p = plots_dir / f"runtime_dow_{route}_dir{direction}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        written.append(p)

    return written


# =============================================================================
# OUTPUT
# =============================================================================


def export_outputs(
    retained: pd.DataFrame,
    outliers: pd.DataFrame,
    stats: pd.DataFrame,
    dow_table: pd.DataFrame,
    back_to_back: pd.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """Write the observation, outlier, stats, DOW-anomaly, and pair CSVs."""
    ensure_dir(out_dir)
    written: List[Path] = []

    obs_path = out_dir / "trip_runtime_observations.csv"
    retained.round({"actual_runtime_min": 2}).to_csv(obs_path, index=False)
    written.append(obs_path)

    out_path = out_dir / "trip_runtime_outliers.csv"
    outliers.round({"actual_runtime_min": 2}).to_csv(out_path, index=False)
    written.append(out_path)

    stats_path = out_dir / "trip_runtime_stats.csv"
    stats.to_csv(stats_path, index=False)
    written.append(stats_path)

    dow_path = out_dir / "trip_runtime_dow.csv"
    dow_table.to_csv(dow_path, index=False)
    written.append(dow_path)

    b2b_path = out_dir / "trip_runtime_back_to_back.csv"
    back_to_back.to_csv(b2b_path, index=False)
    written.append(b2b_path)

    return written


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Optional[Path]:
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

    Args:
        output_dir: Directory holding this run's outputs.
        summary_lines: Human-readable facts about what the run produced.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "runtime_by_trip_runlog.txt"

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
        "TRIP RUNTIME DIAGNOSTICS RUN LOG",
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
# PIPELINE
# =============================================================================


def run(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Execute the full trip-runtime pipeline and write all artifacts.

    Returns:
        Mapping with keys ``retained``, ``outliers``, ``stats``, ``dow``,
        ``back_to_back``.
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)

    runtimes = compute_trip_runtimes(stop_visits)
    joined = join_trip_attributes(runtimes, trips)

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        joined = joined.loc[joined["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        joined = joined.loc[~joined["route_id"].astype(str).isin(drop)]

    retained, outliers = trim_outliers(joined, cfg.trim_frac)
    stats = compute_trip_stats(retained, cfg.high_cv_threshold, cfg.min_obs_for_cv, cfg.gap_frac)
    dow_table = compute_dow_anomalies(retained, cfg.dow_low_count_frac, cfg.dow_runtime_pct)

    back_to_back = compute_back_to_back_flags(
        stats,
        cfg.b2b_runtime_ratio,
        cfg.b2b_runtime_diff_min,
        cfg.b2b_sched_ratio_diff,
        cfg.b2b_max_gap_min,
        cfg.b2b_min_obs,
        cfg.b2b_require_pattern,
    )
    stats = apply_back_to_back_flag(stats, back_to_back)
    if not back_to_back.empty:
        n_flagged = int(back_to_back["back_to_back_flag"].sum())
        n_skipped = int((~back_to_back["compared"].astype(bool)).sum())
        logging.info(
            "Back-to-back check: %d of %d consecutive same-pattern pairs flagged "
            "(%d not compared - too few observations or a scheduled gap over %g min).",
            n_flagged,
            len(back_to_back),
            n_skipped,
            cfg.b2b_max_gap_min,
        )

    paths = export_outputs(retained, outliers, stats, dow_table, back_to_back, cfg.output_dir)
    for p in paths:
        logging.info("Wrote: %s", p)

    box_paths = plot_route_runtime_box(retained, cfg.output_dir)
    dow_paths = plot_dow_runtime(dow_table, cfg.output_dir)
    logging.info("Wrote %d runtime charts.", len(box_paths) + len(dow_paths))

    n_b2b_flagged = int(back_to_back["back_to_back_flag"].sum()) if not back_to_back.empty else 0
    n_b2b_compared = int(back_to_back["compared"].sum()) if not back_to_back.empty else 0
    summary_lines = [
        f"Trips summarized:   {len(stats)}",
        f"Observations kept:  {len(retained)} (trimmed {len(outliers)})",
        f"High variation:     {int(stats['high_variation'].sum()) if not stats.empty else 0}",
        f"Data gaps:          {int(stats['data_gap'].sum()) if not stats.empty else 0}",
        f"Back-to-back pairs: {len(back_to_back)} "
        f"({n_b2b_compared} compared, {n_b2b_flagged} flagged)",
        f"Charts written:     {len(box_paths) + len(dow_paths)}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return {
        "retained": retained,
        "outliers": outliers,
        "stats": stats,
        "dow": dow_table,
        "back_to_back": back_to_back,
    }


# =============================================================================
# CLI / MAIN
# =============================================================================


def notebook_safe_argv(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Return the argv to parse, shielding notebook kernels from stray flags.

    When a script's ``main()`` runs with no explicit ``argv`` inside a
    Jupyter/IPython kernel, ``sys.argv`` holds kernel plumbing (for example
    ``-f /path/kernel.json``) rather than flags meant for the script, and
    strict ``argparse.parse_args`` would reject it and abort.  This helper
    detects the notebook case and substitutes an empty argument list so the
    CONFIGURATION constants stay in charge, while shell runs keep strict
    parsing (a typo in a flag fails loudly instead of being silently ignored).

    Canonical implementation: ``utils/cli_helpers.py``.

    Args:
        argv: Explicit argument list passed to ``main()``, or ``None`` to
            fall back to ``sys.argv``.

    Returns:
        ``list(argv)`` when *argv* was provided; ``[]`` when running inside a
        notebook kernel; otherwise ``None`` so argparse reads ``sys.argv[1:]``.
    """
    if argv is not None:
        return list(argv)
    if "ipykernel" in sys.modules:
        return []
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Per-trip runtime statistics and flags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--trim-frac",
        type=float,
        default=TRIM_FRAC,
        help="Tail fraction trimmed per trip (e.g. 0.01 for 1%%, 0.05 for 5%%).",
    )
    p.add_argument(
        "--high-cv", type=float, default=HIGH_CV_THRESHOLD, help="CV cutoff for high variation."
    )
    p.add_argument(
        "--b2b-runtime-ratio",
        type=float,
        default=B2B_RUNTIME_RATIO,
        help="Back-to-back: longer/shorter median runtime cutoff (0 disables).",
    )
    p.add_argument(
        "--b2b-runtime-diff-min",
        type=float,
        default=B2B_RUNTIME_DIFF_MIN,
        help="Back-to-back: median runtime difference in minutes (0 disables).",
    )
    p.add_argument(
        "--b2b-sched-ratio-diff",
        type=float,
        default=B2B_SCHED_RATIO_DIFF,
        help="Back-to-back: difference between the pair's actual/scheduled ratios (0 disables).",
    )
    p.add_argument(
        "--b2b-max-gap-min",
        type=float,
        default=B2B_MAX_GAP_MIN,
        help="Back-to-back: skip pairs whose starts are farther apart than this (0 = no limit).",
    )
    p.add_argument(
        "--b2b-min-obs",
        type=int,
        default=B2B_MIN_OBS,
        help="Back-to-back: minimum observations required on both trips in a pair.",
    )
    p.add_argument(
        "--b2b-allow-mixed-patterns",
        action="store_true",
        help="Back-to-back: compare within route/direction even when no pattern/shape column "
        "exists (off by default, since cross-pattern pairs are meaningless).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    if args.stop_visits == STOP_VISITS_PATH or args.trips_performed == TRIPS_PERFORMED_PATH:
        logging.warning(
            "STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. Update the "
            "CONFIGURATION section or pass --stop-visits/--trips-performed before running."
        )
        return 2

    cfg = Config(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        trim_frac=args.trim_frac,
        high_cv_threshold=args.high_cv,
        b2b_runtime_ratio=args.b2b_runtime_ratio,
        b2b_runtime_diff_min=args.b2b_runtime_diff_min,
        b2b_sched_ratio_diff=args.b2b_sched_ratio_diff,
        b2b_max_gap_min=args.b2b_max_gap_min,
        b2b_min_obs=args.b2b_min_obs,
        b2b_require_pattern=not args.b2b_allow_mixed_patterns,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
    )

    if not cfg.stop_visits_path.exists():
        logging.warning("stop_visits not found: %s", cfg.stop_visits_path)
        return 1
    if not cfg.trips_performed_path.exists():
        logging.warning("trips_performed not found: %s", cfg.trips_performed_path)
        return 1

    run(cfg)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
