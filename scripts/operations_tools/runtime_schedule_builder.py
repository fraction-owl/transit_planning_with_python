"""Build proposed scheduled running times from TIDES stop and trip events.

Walks a planner through setting scheduled running times for one day type (e.g.
weekdays). At each step the script proposes an answer from observed data and
the planner accepts or changes it before the next step builds on it:

1. **Running time periods** - an order-preserving Jenks (Fisher) optimal split
   of each route-direction's trips, in start-time order, on their
   goal-percentile runtimes. Each period is one continuous span, breaks can
   fall mid-hour, and a trip belongs to the period its scheduled start falls
   in. All patterns of a route-direction share one set of periods; each trip's
   runtime is divided by its pattern's typical runtime so patterns pool.
2. **Full runtimes** - per pattern and period, the smallest whole-minute
   runtime at which ``OTP_GOAL_PCT`` of observed trips would be on time at the
   last timepoint under the ``EARLY_MIN``/``LATE_MIN`` window (early arrivals
   there are forgiven when ``DISCOUNT_EARLY_AT_FINAL``). Warns when runtimes
   vary too much for any runtime to reach the goal counting early and late.
3. **Percent spread by timepoint** - proposes each segment's share of the
   runtime from observed segment times, then turns shares into whole minutes
   that add up to the runtime. With ``BACKLOAD_ROUNDING``, cumulative times are
   rounded down so leftover minutes fall toward the end of the trip.
4. **Checks** - flags minutes that differ between patterns sharing the same
   timepoints, 0-minute segments, segments unobserved in a period, thin data,
   and unreachable OTP goals.

Layover and blocking are out of scope. Choices are made at prompts (terminal or
notebook, with ``INTERACTIVE``) or by editing the CSV a step writes and passing
it back (``PERIODS_CSV`` / ``RUNTIMES_CSV`` / ``SPREAD_CSV``).

Inputs
------
- TIDES ``stop_visits`` and ``trips_performed`` CSVs (e.g. from
  ``convert_to_tides_stop_visits.py`` / ``convert_to_tides_trips_performed.py``).
- Optionally, edited step CSVs from an earlier run of this script.

Outputs
-------
- ``step1_periods.csv`` / ``step1_period_trips.csv`` - chosen periods; each
  scheduled trip's runtimes and period.
- ``step2_runtimes.csv`` - per pattern and period: observed runtimes, proposed
  and chosen runtime, estimated OTP at the last timepoint.
- ``step3_spread.csv`` - per segment: observed minutes, proposed and chosen
  percent, scheduled minutes, estimated early / on-time / late arrivals.
- ``step4_runtime_table.csv`` and ``runtime_tables/`` - the runtime table by
  timepoint (long, plus one wide CSV per pattern).
- ``step4_checks.csv`` - items to review.
- ``charts/`` - PNGs of the period split and of the chosen runtimes.
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags)
and run from a shell, ArcGIS Pro's Python window, or a Jupyter notebook. To
revise a choice later, edit the step's CSV and pass it back with
``--periods-csv`` / ``--runtimes-csv`` / ``--spread-csv``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# TIDES inputs: stop_visits (timepoint arrivals/departures) and trips_performed
# (route, direction, scheduled trip id, scheduled start).
STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Edited choices from an earlier run. Leave "" to use this run's proposals (or
# prompts). Each is a CSV this script wrote: step1_periods.csv,
# step2_runtimes.csv, step3_spread.csv.
PERIODS_CSV: str = ""
RUNTIMES_CSV: str = ""
SPREAD_CSV: str = ""

# Prompt to accept or change each proposal when a user is present (terminal or
# notebook). Without a user (cron, CI, piped input) the proposals are used.
INTERACTIVE: bool = True

# --- Service to analyze (one day type per run) ---
# Optional route filters (matched against route_id as a string). Empty = all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()
# Service-date range as "YYYY-MM-DD"; "" = no limit.
SERVICE_DATE_START: str = ""
SERVICE_DATE_END: str = ""
# Days of week to pool: Monday=0 ... Sunday=6. Use (5,) for Saturday, (6,) for Sunday.
DAYS_OF_WEEK: Sequence[int] = (0, 1, 2, 3, 4)
# Drop observed U.S. federal holidays and any extra dates ("YYYY-MM-DD").
EXCLUDE_FEDERAL_HOLIDAYS: bool = True
EXCLUDE_DATES: Sequence[str] = ()

# --- On-time performance target ---
OTP_GOAL_PCT: float = 85.0
# A timepoint is on time when actual minus scheduled falls in [EARLY_MIN, LATE_MIN].
EARLY_MIN: float = -1.0
LATE_MIN: float = 5.0
# Forgive early arrival at the last timepoint (nobody boards there).
DISCOUNT_EARLY_AT_FINAL: bool = True

# --- Step 1: running time periods ---
# Fixed number of periods per route-direction; 0 = choose automatically: the
# fewest periods whose goodness of variance fit reaches PERIOD_GVF_TARGET.
NUM_PERIODS: int = 0
PERIOD_GVF_TARGET: float = 0.80
MAX_PERIODS: int = 8
# Scheduled trips observed fewer times than this don't shape the breaks (they
# are still assigned to a period by start time).
MIN_OBS_PER_TRIP: int = 5

# --- Step 3: percent spread ---
# "median_share": each segment's share of the summed median segment times.
# "otp_cumulative": start from median_share, then move each intermediate
#   timepoint's cumulative time the least amount that maximizes estimated
#   on-time arrivals there (early counted).
SPREAD_METHOD: str = "median_share"
# True rounds cumulative minutes down at each timepoint so leftover minutes
# fall toward the end of the trip; False rounds to the nearest minute.
BACKLOAD_ROUNDING: bool = True

# --- Checks ---
# Flag pattern/period runtimes built from fewer observed trips than this.
MIN_OBS_WARN: int = 10

MAKE_CHARTS: bool = True

LOG_LEVEL: int = logging.INFO

# Output filenames.
PERIODS_FILENAME: str = r"step1_periods.csv"
PERIOD_TRIPS_FILENAME: str = r"step1_period_trips.csv"
RUNTIMES_FILENAME: str = r"step2_runtimes.csv"
SPREAD_FILENAME: str = r"step3_spread.csv"
TABLE_FILENAME: str = r"step4_runtime_table.csv"
CHECKS_FILENAME: str = r"step4_checks.csv"
RUN_LOG_FILENAME: str = r"runtime_schedule_builder_runlog.txt"

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# TIDES identifies a performed trip by (service_date, trip_id_performed): the same
# trip_id_performed may recur on other dates, so joins and grouping use both.
TRIP_KEY: List[str] = ["service_date", "trip_id_performed"]
ROUTE_DIR: List[str] = ["route_id", "direction_id"]
# A pattern is the ordered list of timepoint stop_ids a trip serves.
PATTERN_KEY: List[str] = [*ROUTE_DIR, "pattern_timepoints"]
GROUP_KEY: List[str] = [*PATTERN_KEY, "period_start_min"]
PATTERN_SEP: str = ">"

SV_REQUIRED: List[str] = [
    "service_date",
    "trip_id_performed",
    "trip_stop_sequence",
    "stop_id",
    "timepoint",
    "schedule_arrival_time",
    "schedule_departure_time",
    "actual_arrival_time",
    "actual_departure_time",
]
TP_REQUIRED: List[str] = ["service_date", "trip_id_performed", "route_id", "direction_id"]
# Stop visits with no usable actual time.
UNUSABLE_VISITS: Tuple[str, ...] = ("Skipped", "Missing")
TRUE_STRINGS: Tuple[str, ...] = ("TRUE", "T", "1", "Y", "YES")
SPREAD_METHODS: Tuple[str, ...] = ("median_share", "otp_cumulative")
PROMPT_HINT: str = "Enter = accept, 'all' = accept this and the rest"

# Chart colors (light surface; categorical slots validated as a set).
CHART_SURFACE: str = "#fcfcfb"
CHART_INK: str = "#0b0b0b"
CHART_INK_2: str = "#52514e"
CHART_GRID: str = "#e6e5e1"
CHART_SERIES: Tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
CHART_OTHER: str = "#a3a29b"

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class Config(NamedTuple):
    """Runtime configuration for a runtime-schedule build."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    periods_csv: Optional[Path] = None
    runtimes_csv: Optional[Path] = None
    spread_csv: Optional[Path] = None
    interactive: bool = INTERACTIVE
    routes_to_include: Sequence[str] = ROUTES_TO_INCLUDE
    routes_to_exclude: Sequence[str] = ROUTES_TO_EXCLUDE
    service_date_start: str = SERVICE_DATE_START
    service_date_end: str = SERVICE_DATE_END
    days_of_week: Sequence[int] = DAYS_OF_WEEK
    exclude_federal_holidays: bool = EXCLUDE_FEDERAL_HOLIDAYS
    exclude_dates: Sequence[str] = EXCLUDE_DATES
    otp_goal_pct: float = OTP_GOAL_PCT
    early_min: float = EARLY_MIN
    late_min: float = LATE_MIN
    discount_early_at_final: bool = DISCOUNT_EARLY_AT_FINAL
    num_periods: int = NUM_PERIODS
    period_gvf_target: float = PERIOD_GVF_TARGET
    max_periods: int = MAX_PERIODS
    min_obs_per_trip: int = MIN_OBS_PER_TRIP
    spread_method: str = SPREAD_METHOD
    backload_rounding: bool = BACKLOAD_ROUNDING
    min_obs_warn: int = MIN_OBS_WARN
    make_charts: bool = MAKE_CHARTS


# =============================================================================
# REUSABLE HELPERS (copied from utils/)
# =============================================================================


def parse_time_to_minutes(time_value: Optional[str]) -> Optional[int]:
    """Convert an ``HH:MM[:SS]`` time string to integer minutes past midnight.

    GTFS times may exceed 24:00 (e.g. ``"25:30:00"`` for a 1:30 AM trip on
    the following calendar day); those values are preserved as integers
    greater than or equal to 1440. Seconds, when present, are rounded to the
    nearest minute.

    Args:
        time_value: Time string such as ``"7:05"``, ``"07:05:00"``, or
            ``"26:30:00"``. Leading/trailing whitespace is ignored.
            Non-string or malformed values yield ``None``.

    Returns:
        Minutes since midnight, or ``None`` if the value cannot be parsed.
    """
    if not isinstance(time_value, str):
        return None
    parts = time_value.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return hours * 60 + minutes + round(seconds / 60)


def minutes_to_hhmm(minutes: Optional[float], missing: str = "") -> str:
    """Convert minutes past midnight to a zero-padded ``HH:MM`` string.

    GTFS service days may exceed 24 hours, so values of 1440 minutes or more
    format with hours >= 24 (e.g. ``1590`` -> ``"26:30"``).

    Args:
        minutes: Minutes since midnight (may be fractional; rounded to the
            nearest minute). ``None`` and NaN yield ``missing``.
        missing: String returned for missing values, e.g. ``""`` or a
            sentinel such as ``"–"``.

    Returns:
        Zero-padded ``HH:MM`` string, or ``missing`` when *minutes* is
        ``None``/NaN.
    """
    if minutes is None or pd.isna(minutes):
        return missing
    hours, mins = divmod(int(round(minutes)), 60)
    return f"{hours:02d}:{mins:02d}"


def federal_holidays_observed(year: int) -> set[dt.date]:
    """Return the observed dates of the U.S. federal holidays of *year*.

    Covers the eleven holidays of 5 U.S.C. 6103: New Year's Day, Birthday of
    Martin Luther King Jr. (3rd Monday of January), Washington's Birthday
    (3rd Monday of February), Memorial Day (last Monday of May), Juneteenth
    (June 19, from its 2021 establishment onward), Independence Day, Labor
    Day (1st Monday of September), Columbus Day (2nd Monday of October),
    Veterans Day, Thanksgiving (4th Thursday of November), and Christmas.

    Fixed-date holidays falling on a Saturday are observed on the preceding
    Friday and those falling on a Sunday on the following Monday, so an
    observed date can land in the *previous* calendar year (e.g. New Year's
    Day 2022 was observed on 2021-12-31). Callers classifying a span of dates
    should therefore union this set over ``range(first_year, last_year + 2)``.

    Args:
        year: Calendar year whose holidays are computed.

    Returns:
        The observed dates of *year*'s federal holidays.
    """

    def nth_weekday(month: int, weekday: int, n: int) -> dt.date:
        first = dt.date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + dt.timedelta(days=offset + 7 * (n - 1))

    def last_monday(month: int) -> dt.date:
        next_month = dt.date(year + (month == 12), month % 12 + 1, 1)
        last = next_month - dt.timedelta(days=1)
        return last - dt.timedelta(days=last.weekday())

    def observed(day: dt.date) -> dt.date:
        if day.weekday() == 5:  # Saturday -> preceding Friday
            return day - dt.timedelta(days=1)
        if day.weekday() == 6:  # Sunday -> following Monday
            return day + dt.timedelta(days=1)
        return day

    fixed = [
        dt.date(year, 1, 1),  # New Year's Day
        dt.date(year, 7, 4),  # Independence Day
        dt.date(year, 11, 11),  # Veterans Day
        dt.date(year, 12, 25),  # Christmas Day
    ]
    if year >= 2021:
        fixed.append(dt.date(year, 6, 19))  # Juneteenth
    floating = [
        nth_weekday(1, 0, 3),  # Birthday of Martin Luther King Jr.
        nth_weekday(2, 0, 3),  # Washington's Birthday
        last_monday(5),  # Memorial Day
        nth_weekday(9, 0, 1),  # Labor Day
        nth_weekday(10, 0, 2),  # Columbus Day
        nth_weekday(11, 3, 4),  # Thanksgiving Day
    ]
    return {observed(day) for day in fixed} | set(floating)


# =============================================================================
# LOADING & FILTERING
# =============================================================================


def _minutes(delta: pd.Series) -> pd.Series:
    """Convert a timedelta Series to float minutes."""
    return delta.dt.total_seconds() / 60.0


def _check_columns(df: pd.DataFrame, required: Sequence[str], label: str, path: Path) -> None:
    """Raise a readable ValueError when *df* lacks any *required* column."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} '{path}' is missing required column(s): {missing}. "
            f"Expected at least {list(required)}."
        )


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read TIDES ``stop_visits`` and parse timestamps, sequence, and flags.

    Args:
        path: Path to the ``stop_visits`` CSV.

    Returns:
        DataFrame with parsed timestamps and ``service_date``, a numeric
        ``trip_stop_sequence``, a boolean ``timepoint``, and
        ``schedule_relationship`` defaulted to ``Scheduled``.

    Raises:
        ValueError: If a required column is missing.
    """
    df = pd.read_csv(path, dtype=str)
    _check_columns(df, SV_REQUIRED, "stop_visits", path)
    for col in (
        "schedule_arrival_time",
        "schedule_departure_time",
        "actual_arrival_time",
        "actual_departure_time",
    ):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["trip_stop_sequence"] = pd.to_numeric(df["trip_stop_sequence"], errors="coerce")
    df["stop_id"] = df["stop_id"].fillna("").astype(str).str.strip()
    df["timepoint"] = df["timepoint"].fillna("").astype(str).str.strip().str.upper()
    df["timepoint"] = df["timepoint"].isin(TRUE_STRINGS)
    if "schedule_relationship" not in df.columns:
        df["schedule_relationship"] = "Scheduled"
    df["schedule_relationship"] = df["schedule_relationship"].fillna("Scheduled")
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read TIDES ``trips_performed``, keeping in-service, non-canceled trips.

    Raises:
        ValueError: If a required column is missing.
    """
    df = pd.read_csv(path, dtype=str)
    _check_columns(df, TP_REQUIRED, "trips_performed", path)
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    if "schedule_trip_start" in df.columns:
        df["schedule_trip_start"] = pd.to_datetime(df["schedule_trip_start"], errors="coerce")
    if "schedule_relationship" in df.columns:
        df = df.loc[df["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in df.columns:
        df = df.loc[df["trip_type"].fillna("In service") == "In service"]
    df = df.dropna(subset=["service_date", "route_id"]).copy()
    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["direction_id"] = df["direction_id"].fillna("").astype(str).str.strip()
    return df.drop_duplicates(TRIP_KEY)


def filter_service(trips: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep the trips of the routes, dates, and day type being scheduled."""
    out = trips
    if cfg.routes_to_include:
        out = out.loc[out["route_id"].isin({str(r) for r in cfg.routes_to_include})]
    if cfg.routes_to_exclude:
        out = out.loc[~out["route_id"].isin({str(r) for r in cfg.routes_to_exclude})]
    if cfg.service_date_start:
        out = out.loc[out["service_date"] >= pd.Timestamp(cfg.service_date_start)]
    if cfg.service_date_end:
        out = out.loc[out["service_date"] <= pd.Timestamp(cfg.service_date_end)]
    out = out.loc[out["service_date"].dt.dayofweek.isin(list(cfg.days_of_week))]

    drop_dates = {pd.Timestamp(d).date() for d in cfg.exclude_dates}
    if cfg.exclude_federal_holidays and not out.empty:
        first_year = int(out["service_date"].dt.year.min())
        last_year = int(out["service_date"].dt.year.max())
        for year in range(first_year, last_year + 2):
            drop_dates |= federal_holidays_observed(year)
    if drop_dates:
        dates = out["service_date"].dt.date
        n_dropped = int(dates.isin(drop_dates).sum())
        if n_dropped:
            logging.info("Dropped %d trip(s) on holidays or excluded dates.", n_dropped)
        out = out.loc[~dates.isin(drop_dates)]
    return out


# =============================================================================
# TRIP & TIMEPOINT OBSERVATIONS
# =============================================================================


def build_observations(
    stop_visits: pd.DataFrame, trips: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derive per-trip runtimes and per-timepoint times from stop visits.

    Only timepoint visits are used (``Added`` visits are not part of the
    pattern). ``Skipped``/``Missing`` visits keep their place in the pattern
    but have no actual time. A trip's pattern is its ordered timepoint
    stop_ids; a performed trip whose pattern differs from the most common
    pattern of its scheduled trip (usually a missing AVL row) is dropped.

    Times are measured from the departure at the first timepoint:
    ``actual_runtime_min`` is the arrival at the last timepoint minus that
    departure; ``seg_actual_min`` is the arrival at a timepoint minus the
    departure from the previous one (excludes dwell and holding at
    timepoints); ``cum_actual_min`` is the arrival at a timepoint minus the
    first departure.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.
        trips: Output of :func:`load_trips_performed` after filtering.

    Returns:
        ``(trip_obs, tp_obs)``: one row per performed trip (route, direction,
        pattern, ``trip_key``, ``start_min``, actual and scheduled runtime) and
        one row per timepoint visit of those trips.
    """
    attr_cols = [c for c in ("trip_id_scheduled", "schedule_trip_start") if c in trips.columns]
    sv = stop_visits.merge(
        trips[[*TRIP_KEY, *ROUTE_DIR, *attr_cols]], on=TRIP_KEY, how="inner"
    ).dropna(subset=["trip_stop_sequence"])
    tp = sv.loc[sv["timepoint"] & (sv["schedule_relationship"] != "Added")]
    tp = tp.sort_values([*TRIP_KEY, "trip_stop_sequence"], kind="mergesort").reset_index(drop=True)

    usable = ~tp["schedule_relationship"].isin(UNUSABLE_VISITS)
    tp["_arr"] = tp["actual_arrival_time"].fillna(tp["actual_departure_time"]).where(usable)
    tp["_dep"] = tp["actual_departure_time"].fillna(tp["actual_arrival_time"]).where(usable)
    tp["_sched_arr"] = tp["schedule_arrival_time"].fillna(tp["schedule_departure_time"])
    tp["_sched_dep"] = tp["schedule_departure_time"].fillna(tp["schedule_arrival_time"])

    by_trip = tp.groupby(TRIP_KEY, sort=False)
    tp["tp_seq"] = by_trip.cumcount() + 1
    tp["n_tp"] = by_trip["stop_id"].transform("size")
    patterns = by_trip["stop_id"].agg(PATTERN_SEP.join).rename("pattern_timepoints")
    tp = tp.merge(patterns.reset_index(), on=TRIP_KEY, how="left")

    first_row = tp["tp_seq"] == 1
    for col in ("_dep", "_sched_dep"):
        tp["_tmp"] = tp[col].where(first_row)
        tp["_origin" + col] = tp.groupby(TRIP_KEY, sort=False)["_tmp"].transform("max")
    prev = tp.groupby(TRIP_KEY, sort=False)[["_dep", "_sched_dep"]].shift(1)
    tp["seg_actual_min"] = _minutes(tp["_arr"] - prev["_dep"])
    tp["seg_sched_min"] = _minutes(tp["_sched_arr"] - prev["_sched_dep"])
    tp["cum_actual_min"] = _minutes(tp["_arr"] - tp["_origin_dep"])
    tp["cum_sched_min"] = _minutes(tp["_sched_arr"] - tp["_origin_sched_dep"])
    tp.loc[first_row, ["cum_actual_min", "cum_sched_min"]] = 0.0
    tp.loc[tp["seg_actual_min"] < 0, "seg_actual_min"] = np.nan

    trip_cols = [*TRIP_KEY, *ROUTE_DIR, *attr_cols, "pattern_timepoints", "n_tp"]
    trip_obs = tp.loc[tp["tp_seq"] == tp["n_tp"], [*trip_cols, "cum_actual_min", "cum_sched_min"]]
    trip_obs = trip_obs.rename(
        columns={"cum_actual_min": "actual_runtime_min", "cum_sched_min": "sched_runtime_min"}
    )
    origin_sched = tp.loc[first_row, [*TRIP_KEY, "_origin_sched_dep"]]
    trip_obs = trip_obs.merge(origin_sched, on=TRIP_KEY, how="left")
    if "schedule_trip_start" in trip_obs.columns:
        sched_start = trip_obs["schedule_trip_start"].fillna(trip_obs["_origin_sched_dep"])
    else:
        sched_start = trip_obs["_origin_sched_dep"]
    trip_obs["start_min"] = _minutes(sched_start - trip_obs["service_date"])

    n_short = int((trip_obs["n_tp"] < 2).sum())
    if n_short:
        logging.warning("%d trip(s) have fewer than two timepoints and were skipped.", n_short)
    trip_obs = trip_obs.loc[(trip_obs["n_tp"] >= 2) & trip_obs["start_min"].notna()].copy()
    trip_obs.loc[trip_obs["actual_runtime_min"] <= 0, "actual_runtime_min"] = np.nan

    fallback_key = (
        trip_obs["route_id"]
        + "|"
        + trip_obs["direction_id"]
        + "|"
        + trip_obs["pattern_timepoints"]
        + "|"
        + trip_obs["start_min"].map(minutes_to_hhmm)
    )
    if "trip_id_scheduled" in trip_obs.columns:
        trip_obs["trip_key"] = trip_obs["trip_id_scheduled"].fillna(fallback_key)
    else:
        trip_obs["trip_key"] = fallback_key

    modal = (
        trip_obs.groupby([*ROUTE_DIR, "trip_key"])["pattern_timepoints"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("_modal")
        .reset_index()
    )
    trip_obs = trip_obs.merge(modal, on=[*ROUTE_DIR, "trip_key"], how="left")
    off_pattern = trip_obs["pattern_timepoints"] != trip_obs["_modal"]
    if off_pattern.any():
        logging.warning(
            "%d performed trip(s) served a different set of timepoints than their scheduled "
            "trip usually does (often a missing AVL record) and were skipped.",
            int(off_pattern.sum()),
        )
    trip_obs = trip_obs.loc[~off_pattern]

    keep_trip_cols = [
        *TRIP_KEY,
        *PATTERN_KEY,
        "trip_key",
        "start_min",
        "n_tp",
        "actual_runtime_min",
        "sched_runtime_min",
    ]
    trip_obs = trip_obs[keep_trip_cols].reset_index(drop=True)
    tp_cols = [
        *TRIP_KEY,
        "tp_seq",
        "stop_id",
        "seg_actual_min",
        "seg_sched_min",
        "cum_actual_min",
        "cum_sched_min",
    ]
    tp_obs = tp[tp_cols].merge(
        trip_obs[[*TRIP_KEY, *PATTERN_KEY, "start_min", "n_tp"]], on=TRIP_KEY, how="inner"
    )
    return trip_obs, tp_obs


def label_patterns(trip_obs: pd.DataFrame) -> pd.DataFrame:
    """Name each route-direction's patterns P1, P2, ... by observed trip count."""
    counts = trip_obs.groupby(PATTERN_KEY).size().rename("n_trip_obs").reset_index()
    counts = counts.sort_values(
        [*ROUTE_DIR, "n_trip_obs", "pattern_timepoints"], ascending=[True, True, False, True]
    )
    counts["pattern"] = "P" + (counts.groupby(ROUTE_DIR).cumcount() + 1).astype(str)
    return counts.reset_index(drop=True)


# =============================================================================
# STEP 1: RUNNING TIME PERIODS
# =============================================================================


def _mode_or_median(series: pd.Series) -> float:
    """Return the modal value of a numeric series, breaking ties by median."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    modes = s.mode()
    if len(modes) == 1:
        return float(modes.iloc[0])
    return float(s.median())


def summarize_scheduled_trips(
    trip_obs: pd.DataFrame, goal_pct: float, min_obs: int
) -> pd.DataFrame:
    """Summarize each scheduled trip's observed runtimes across dates.

    ``ratio`` is the trip's goal-percentile runtime divided by the median of
    that value over its pattern's trips, so a short turn and a full trip that
    both run 10% long in the PM peak read alike and can share period breaks.

    Returns:
        One row per scheduled trip with ``start_min``, ``n_obs``,
        ``runtime_p50_min``, ``runtime_goal_pctile_min``,
        ``current_runtime_min``, ``ratio``, and ``used_for_breaks``.
    """
    obs = trip_obs.dropna(subset=["actual_runtime_min"])
    keys = [*PATTERN_KEY, "trip_key"]
    grouped = obs.groupby(keys)
    summary = grouped.agg(
        start_min=("start_min", "median"),
        n_obs=("actual_runtime_min", "count"),
        runtime_p50_min=("actual_runtime_min", "median"),
        current_runtime_min=("sched_runtime_min", _mode_or_median),
    ).reset_index()
    goal = grouped["actual_runtime_min"].quantile(goal_pct / 100.0)
    summary = summary.merge(
        goal.rename("runtime_goal_pctile_min").reset_index(), on=keys, how="left"
    )
    summary["used_for_breaks"] = summary["n_obs"] >= min_obs
    reference = (
        summary.loc[summary["used_for_breaks"]]
        .groupby(PATTERN_KEY)["runtime_goal_pctile_min"]
        .median()
        .rename("_reference")
        .reset_index()
    )
    summary = summary.merge(reference, on=PATTERN_KEY, how="left")
    summary["ratio"] = summary["runtime_goal_pctile_min"] / summary["_reference"]
    summary.loc[summary["_reference"] <= 0, "ratio"] = np.nan
    return summary.drop(columns="_reference").sort_values([*ROUTE_DIR, "start_min"])


def ordered_partition(
    values: np.ndarray, can_start: np.ndarray, max_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the best split of ordered *values* into 1..max_k contiguous groups.

    This is Fisher's exact optimal partition (the method behind Jenks natural
    breaks) without sorting the values first, so groups stay contiguous in the
    given order. Each split minimizes the total within-group sum of squared
    deviations.

    Args:
        values: Values in their fixed order (here: trips by start time).
        can_start: Boolean per position; a group may begin at position ``i``
            only where ``can_start[i]`` (used to forbid splitting trips that
            start in the same minute).
        max_k: Largest number of groups to evaluate.

    Returns:
        ``(sse, back)``: ``sse[k]`` is the smallest within-group sum of squares
        with ``k`` groups (``inf`` when infeasible; index 0 unused) and
        ``back`` holds the split points for :func:`partition_starts`.
    """
    n = len(values)
    s1 = np.concatenate([[0.0], np.cumsum(values)])
    s2 = np.concatenate([[0.0], np.cumsum(values**2)])

    def cost(i: np.ndarray, j: int) -> np.ndarray:
        total = s1[j] - s1[i]
        return np.maximum((s2[j] - s2[i]) - total * total / (j - i), 0.0)

    best = np.full((max_k + 1, n + 1), np.inf)
    back = np.zeros((max_k + 1, n + 1), dtype=int)
    ends = np.arange(1, n + 1)
    best[1, 1:] = np.maximum(s2[ends] - s1[ends] ** 2 / ends, 0.0)
    for k in range(2, max_k + 1):
        for j in range(k, n + 1):
            starts = np.arange(k - 1, j)
            candidates = np.where(can_start[starts], best[k - 1, starts] + cost(starts, j), np.inf)
            idx = int(np.argmin(candidates))
            best[k, j] = candidates[idx]
            back[k, j] = starts[idx]
    return best[:, n], back


def partition_starts(back: np.ndarray, k: int, n: int) -> List[int]:
    """Return the first position of each of the *k* groups found by :func:`ordered_partition`."""
    starts = [0]
    j = n
    for level in range(k, 1, -1):
        j = int(back[level, j])
        starts.append(j)
    return sorted(starts)


def partition_gvf(values: np.ndarray, group_ids: np.ndarray) -> float:
    """Return the goodness of variance fit (1 - within SSE / total SSE) of a grouping."""
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 0:
        return 1.0
    frame = pd.DataFrame({"v": values, "g": group_ids})
    within = float((frame["v"] - frame.groupby("g")["v"].transform("mean")).pow(2).sum())
    return 1.0 - within / total


def propose_periods(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Propose running time period start times for each route-direction.

    Trips with at least ``min_obs_per_trip`` observations are ordered by
    scheduled start and split with :func:`ordered_partition` on ``ratio``. The
    number of periods is ``num_periods`` when set, otherwise the fewest whose
    GVF reaches ``period_gvf_target`` (capped at ``max_periods``). Each
    period starts at the whole minute of its first trip's start; the first
    period starts at the route-direction's earliest trip.

    Returns:
        One row per period: ``route_id``, ``direction_id``, ``start_min``.
    """
    rows: List[Dict[str, Any]] = []
    for (route, direction), g in summary.groupby(ROUTE_DIR, sort=True):
        first_start = math.floor(g["start_min"].min())
        rows.append({"route_id": route, "direction_id": direction, "start_min": first_start})
        used = g.loc[g["used_for_breaks"] & g["ratio"].notna()].sort_values(
            "start_min", kind="mergesort"
        )
        if len(used) < 2:
            logging.info(
                "Route %s dir %s: too few well-observed trips to split; one period.",
                route,
                direction,
            )
            continue
        values = used["ratio"].to_numpy(float)
        minute = np.floor(used["start_min"].to_numpy(float))
        can_start = np.concatenate([[True], minute[1:] > minute[:-1]])
        cap = cfg.num_periods if cfg.num_periods > 0 else cfg.max_periods
        max_k = max(1, min(cap, int(can_start[1:].sum()) + 1))
        sse, back = ordered_partition(values, can_start, max_k)
        feasible = [k for k in range(1, max_k + 1) if np.isfinite(sse[k])]
        gvf = {k: (1.0 - sse[k] / sse[1]) if sse[1] > 0 else 1.0 for k in feasible}
        if cfg.num_periods > 0:
            k = feasible[-1]
        else:
            reaching = [k for k in feasible if gvf[k] >= cfg.period_gvf_target]
            k = reaching[0] if reaching else feasible[-1]
        logging.info(
            "Route %s dir %s: proposing %d period(s) (GVF %.2f; by count: %s).",
            route,
            direction,
            k,
            gvf[k],
            ", ".join(f"{kk}={gvf[kk]:.2f}" for kk in feasible),
        )
        for idx in partition_starts(back, k, len(values))[1:]:
            rows.append({"route_id": route, "direction_id": direction, "start_min": minute[idx]})
    return number_periods(pd.DataFrame(rows, columns=[*ROUTE_DIR, "start_min"]))


def number_periods(periods: pd.DataFrame) -> pd.DataFrame:
    """Sort periods, drop duplicate starts, and add ``period``, start and end labels."""
    out = periods.copy()
    out["start_min"] = out["start_min"].astype(float)
    out = out.drop_duplicates([*ROUTE_DIR, "start_min"]).sort_values([*ROUTE_DIR, "start_min"])
    out["period"] = out.groupby(ROUTE_DIR).cumcount() + 1
    out["end_min"] = out.groupby(ROUTE_DIR)["start_min"].shift(-1)
    out["period_start"] = out["start_min"].map(minutes_to_hhmm)
    out["period_end"] = out["end_min"].map(minutes_to_hhmm)
    return out.reset_index(drop=True)


def assign_periods(df: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's period by its ``start_min``.

    A trip belongs to the last period starting at or before its scheduled
    start; trips earlier than the first period start join the first period.

    Returns:
        Copy of *df* with ``period``, ``period_start_min``, ``period_start``,
        and ``period_end``.
    """
    lookup = {key: p for key, p in periods.groupby(ROUTE_DIR, sort=False)}
    parts = []
    for key, g in df.groupby(ROUTE_DIR, sort=False):
        p = lookup.get(key)
        if p is None:
            continue
        starts = p["start_min"].to_numpy(float)
        idx = np.searchsorted(starts, g["start_min"].to_numpy(float), side="right") - 1
        idx = np.clip(idx, 0, len(starts) - 1)
        parts.append(
            g.assign(
                period=p["period"].to_numpy()[idx],
                period_start_min=starts[idx],
                period_start=p["period_start"].to_numpy()[idx],
                period_end=p["period_end"].to_numpy()[idx],
            )
        )
    if not parts:
        return df.iloc[0:0].assign(
            period=pd.Series(dtype=int),
            period_start_min=pd.Series(dtype=float),
            period_start=pd.Series(dtype=str),
            period_end=pd.Series(dtype=str),
        )
    return pd.concat(parts).sort_index()


def describe_periods(periods: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Add trip counts, trip spans, runtime level, and GVF to *periods* for review."""
    assigned = assign_periods(summary, periods)
    stats = (
        assigned.groupby([*ROUTE_DIR, "period"])
        .agg(
            n_trips=("trip_key", "size"),
            first_trip_min=("start_min", "min"),
            last_trip_min=("start_min", "max"),
            runtime_vs_typical_pct=("ratio", "mean"),
        )
        .reset_index()
    )
    stats["runtime_vs_typical_pct"] = (stats["runtime_vs_typical_pct"] * 100.0).round(1)
    out = periods.merge(stats, on=[*ROUTE_DIR, "period"], how="left")
    out["n_trips"] = out["n_trips"].fillna(0).astype(int)
    out["first_trip"] = out["first_trip_min"].map(minutes_to_hhmm)
    out["last_trip"] = out["last_trip_min"].map(minutes_to_hhmm)

    gvf_rows = []
    for key, g in assigned.loc[assigned["used_for_breaks"] & assigned["ratio"].notna()].groupby(
        ROUTE_DIR
    ):
        gvf_rows.append(
            {
                "route_id": key[0],
                "direction_id": key[1],
                "gvf": partition_gvf(g["ratio"].to_numpy(float), g["period"].to_numpy()),
            }
        )
    gvf = pd.DataFrame(gvf_rows, columns=[*ROUTE_DIR, "gvf"])
    out = out.merge(gvf, on=ROUTE_DIR, how="left")
    out["gvf"] = out["gvf"].round(3)
    cols = [
        "route_id",
        "direction_id",
        "period",
        "period_start",
        "period_end",
        "n_trips",
        "first_trip",
        "last_trip",
        "runtime_vs_typical_pct",
        "gvf",
    ]
    return out[cols]


def read_periods_csv(path: Path, proposed: pd.DataFrame) -> pd.DataFrame:
    """Read period start times chosen in a CSV (``route_id``, ``direction_id``, ``period_start``).

    Route-directions missing from the file keep the proposal; rows for
    route-directions not in this run's data are ignored. Both are logged.

    Raises:
        ValueError: If a required column is missing or a time can't be read.
    """
    df = pd.read_csv(path, dtype=str)
    _check_columns(df, [*ROUTE_DIR, "period_start"], "Periods CSV", path)
    df["route_id"] = df["route_id"].fillna("").str.strip()
    df["direction_id"] = df["direction_id"].fillna("").str.strip()
    df["start_min"] = df["period_start"].map(parse_time_to_minutes)
    bad = df.loc[df["start_min"].isna(), "period_start"].tolist()
    if bad:
        raise ValueError(f"Periods CSV '{path}' has unreadable period_start value(s): {bad}.")
    chosen = df[[*ROUTE_DIR, "start_min"]]

    in_data = set(map(tuple, proposed[ROUTE_DIR].drop_duplicates().to_numpy().tolist()))
    in_file = set(map(tuple, chosen[ROUTE_DIR].drop_duplicates().to_numpy().tolist()))
    if in_file - in_data:
        logging.warning(
            "Periods CSV rows for route-direction(s) not in this run were ignored: %s",
            sorted(in_file - in_data),
        )
    if in_data - in_file:
        logging.warning(
            "Route-direction(s) not in the periods CSV keep the proposed periods: %s",
            sorted(in_data - in_file),
        )
    keep = chosen.loc[[tuple(k) in in_data for k in chosen[ROUTE_DIR].to_numpy().tolist()]]
    fill = proposed.loc[
        [tuple(k) in in_data - in_file for k in proposed[ROUTE_DIR].to_numpy().tolist()],
        [*ROUTE_DIR, "start_min"],
    ]
    return number_periods(pd.concat([keep, fill], ignore_index=True))


# =============================================================================
# STEP 2: FULL RUNTIMES
# =============================================================================


def runtime_options(
    values: np.ndarray,
    goal_pct: float,
    early: float,
    late: float,
    discount_early: bool,
) -> Dict[str, Any]:
    """Score whole-minute runtimes against observed runtimes and pick one.

    A trip counts as on time at the last timepoint when its observed runtime
    minus the scheduled runtime falls in ``[early, late]`` (only ``<= late``
    when *discount_early*); departures from the first timepoint are assumed
    on time. The proposal is the smallest whole minute meeting *goal_pct*; if
    none does (possible only when early arrivals count), the one with the
    highest on-time share.

    Args:
        values: Observed runtimes in minutes (NaNs are ignored).
        goal_pct: Target on-time percentage.
        early: Earliest on-time deviation in minutes (negative).
        late: Latest on-time deviation in minutes.
        discount_early: Forgive early arrivals at the last timepoint.

    Returns:
        Dict with ``proposed_runtime_min``, ``est_otp_proposed_pct``,
        ``best_two_sided_otp_pct`` (the highest share any runtime reaches with
        early and late both counted), ``goal_reachable_two_sided``, and
        ``span_for_goal_min`` (the narrowest range holding *goal_pct* of the
        observed runtimes; compare it with the window's width).
    """
    v = np.sort(values[~np.isnan(values)])
    n = len(v)
    if n == 0:
        return {
            "proposed_runtime_min": np.nan,
            "est_otp_proposed_pct": np.nan,
            "best_two_sided_otp_pct": np.nan,
            "goal_reachable_two_sided": False,
            "span_for_goal_min": np.nan,
        }
    goal = goal_pct / 100.0
    candidates = np.arange(max(1.0, math.floor(v[0] - late)), math.ceil(v[-1] - early) + 1.0)
    on_or_early = np.searchsorted(v, candidates + late, side="right")
    too_early = np.searchsorted(v, candidates + early, side="left")
    two_sided = (on_or_early - too_early) / n
    share = on_or_early / n if discount_early else two_sided
    meets = np.flatnonzero(share >= goal - 1e-9)
    pick = int(meets[0]) if len(meets) else int(np.argmax(share))

    m = max(1, math.ceil(goal * n - 1e-9))
    span = float((v[m - 1 :] - v[: n - m + 1]).min())
    best_two = float(two_sided.max())
    return {
        "proposed_runtime_min": int(candidates[pick]),
        "est_otp_proposed_pct": round(float(share[pick]) * 100.0, 1),
        "best_two_sided_otp_pct": round(best_two * 100.0, 1),
        "goal_reachable_two_sided": bool(best_two >= goal - 1e-9),
        "span_for_goal_min": round(span, 1),
    }


def estimate_final_otp(
    actual: np.ndarray, scheduled: Any, early: float, late: float, discount_early: bool
) -> float:
    """Return the % of trips on time at the last timepoint for given scheduled runtime(s)."""
    dev = np.asarray(actual, float) - np.asarray(scheduled, float)
    dev = dev[~np.isnan(dev)]
    if len(dev) == 0:
        return float("nan")
    on_time = dev <= late if discount_early else (dev >= early) & (dev <= late)
    return round(float(on_time.mean()) * 100.0, 1)


def propose_runtimes(trip_obs: pd.DataFrame, patterns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Propose a whole-minute runtime per pattern and period (see :func:`runtime_options`).

    Logs a warning for each pattern-period whose runtimes vary too much to
    reach the OTP goal when both early and late arrivals count.

    Args:
        trip_obs: Trip observations with periods assigned.
        patterns: Output of :func:`label_patterns`.
        cfg: Run configuration.

    Returns:
        One row per (route, direction, pattern, period) with observed
        statistics, ``proposed_runtime_min``, and ``chosen_runtime_min``
        (initially the proposal).
    """
    obs = trip_obs.dropna(subset=["actual_runtime_min", "period_start_min"])
    rows: List[Dict[str, Any]] = []
    window = cfg.late_min - cfg.early_min
    for key, g in obs.groupby(GROUP_KEY, sort=True):
        actual = g["actual_runtime_min"].to_numpy(float)
        opts = runtime_options(
            actual, cfg.otp_goal_pct, cfg.early_min, cfg.late_min, cfg.discount_early_at_final
        )
        row = dict(zip(GROUP_KEY, key))
        row.update(
            {
                "period": int(g["period"].iloc[0]),
                "period_start": g["period_start"].iloc[0],
                "period_end": g["period_end"].iloc[0],
                "n_trips": int(g["trip_key"].nunique()),
                "n_obs": len(actual),
                "current_runtime_min": round(_mode_or_median(g["sched_runtime_min"]), 1),
                "actual_p50_min": round(float(np.median(actual)), 1),
                "actual_goal_pctile_min": round(
                    float(np.quantile(actual, cfg.otp_goal_pct / 100.0)), 1
                ),
                "est_otp_current_pct": estimate_final_otp(
                    actual,
                    g["sched_runtime_min"].to_numpy(float),
                    cfg.early_min,
                    cfg.late_min,
                    cfg.discount_early_at_final,
                ),
                **opts,
            }
        )
        rows.append(row)
        if not opts["goal_reachable_two_sided"]:
            label = patterns.set_index(PATTERN_KEY).loc[key[:3], "pattern"]
            logging.warning(
                "Route %s dir %s %s, %s-%s: runtimes vary too much to reach %.0f%% OTP counting "
                "both early and late arrivals (best %.0f%%). The narrowest range holding "
                "%.0f%% of runtimes is %.1f min; the on-time window is %.0f min wide.",
                key[0],
                key[1],
                label,
                row["period_start"],
                row["period_end"] or "end",
                cfg.otp_goal_pct,
                opts["best_two_sided_otp_pct"],
                cfg.otp_goal_pct,
                opts["span_for_goal_min"],
                window,
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(patterns[[*PATTERN_KEY, "pattern"]], on=PATTERN_KEY, how="left")
    out["chosen_runtime_min"] = out["proposed_runtime_min"]
    return out


def read_choice_csv(
    path: Path, value_col: str, label: str, extra_keys: Sequence[str] = ()
) -> pd.DataFrame:
    """Read a step CSV's choice column keyed by route, direction, pattern, and period.

    Raises:
        ValueError: If a required column is missing or a value can't be read.
    """
    df = pd.read_csv(path, dtype=str)
    needed = [*PATTERN_KEY, "period_start", *extra_keys, value_col]
    _check_columns(df, needed, label, path)
    for col in PATTERN_KEY:
        df[col] = df[col].fillna("").str.strip()
    df["period_start_min"] = df["period_start"].map(parse_time_to_minutes).astype(float)
    for col in extra_keys:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    bad = df[["period_start_min", *extra_keys, value_col]].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{label} '{path}' has {int(bad.sum())} row(s) with an unreadable period_start, "
            f"key, or {value_col} (first at CSV row {int(np.flatnonzero(bad.to_numpy())[0]) + 2})."
        )
    for col in extra_keys:
        df[col] = df[col].astype(int)
    keys = [*GROUP_KEY, *extra_keys]
    duplicated = df.duplicated(keys, keep="last")
    if duplicated.any():
        logging.warning(
            "%s '%s' repeats %d key(s); the last row of each is used.",
            label,
            path,
            int(duplicated.sum()),
        )
    return df.loc[~duplicated, [*keys, value_col]]


def apply_choices(
    proposed: pd.DataFrame,
    choices: pd.DataFrame,
    keys: Sequence[str],
    value_col: str,
    label: str,
) -> pd.DataFrame:
    """Overwrite *value_col* in *proposed* with matching rows from *choices*.

    Unmatched rows on either side are logged: proposals without a choice keep
    the proposed value; choices with no matching proposal are ignored (they
    usually come from periods or patterns that have since changed).
    """
    merged = proposed.merge(
        choices.rename(columns={value_col: "_chosen"}),
        on=list(keys),
        how="left",
        indicator=True,
    )
    matched = merged["_merge"] == "both"
    n_unmatched_file = len(choices) - int(matched.sum())
    if (~matched).any():
        logging.warning(
            "%s: %d row(s) had no match in the file and keep the proposed value.",
            label,
            int((~matched).sum()),
        )
    if n_unmatched_file > 0:
        logging.warning(
            "%s: %d file row(s) matched nothing in this run (changed periods or patterns?) "
            "and were ignored.",
            label,
            n_unmatched_file,
        )
    merged[value_col] = merged["_chosen"].where(matched, merged[value_col])
    return merged.drop(columns=["_chosen", "_merge"])


def finalize_runtimes(runtimes: pd.DataFrame, trip_obs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Validate chosen runtimes and add the estimated OTP at each chosen runtime.

    Raises:
        ValueError: If a chosen runtime is missing or below one minute.
    """
    out = runtimes.copy()
    bad = out["chosen_runtime_min"].isna() | (out["chosen_runtime_min"] < 1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} chosen runtime(s) are missing or under 1 minute; "
            "fix chosen_runtime_min and run again."
        )
    out["chosen_runtime_min"] = out["chosen_runtime_min"].round().astype(int)
    obs = {k: g for k, g in trip_obs.groupby(GROUP_KEY, sort=False)}
    est = []
    for row in out.itertuples(index=False):
        g = obs[tuple(getattr(row, c) for c in GROUP_KEY)]
        est.append(
            estimate_final_otp(
                g["actual_runtime_min"].to_numpy(float),
                row.chosen_runtime_min,
                cfg.early_min,
                cfg.late_min,
                cfg.discount_early_at_final,
            )
        )
    out["est_otp_chosen_pct"] = est
    return out


# =============================================================================
# STEP 3: PERCENT SPREAD BY TIMEPOINT
# =============================================================================


def allocate_minutes(pcts: Sequence[float], total: int, backload: bool = True) -> List[int]:
    """Turn segment percentages into whole minutes that add up to *total*.

    Percentages are normalized to sum to 100 and accumulated, so rounding
    happens on the cumulative time at each timepoint and never drifts. With
    *backload*, each cumulative time is rounded down, so the leftover minutes
    land toward the end of the trip and a timepoint is never scheduled later
    than its exact share; otherwise cumulative times round to the nearest
    minute (halves up).

    Args:
        pcts: Share of the runtime for each segment, in trip order.
        total: Whole-minute runtime to divide.
        backload: Round cumulative times down (True) or to nearest (False).

    Returns:
        Minutes per segment; they sum to *total*.

    Raises:
        ValueError: If a percentage is negative or all are zero.
    """
    p = np.asarray(pcts, float)
    if (p < 0).any() or p.sum() <= 0:
        raise ValueError("Segment percentages must be zero or positive and not all zero.")
    exact = np.cumsum(p) / p.sum() * total
    cumulative = np.floor(exact + 1e-9) if backload else np.floor(exact + 0.5 + 1e-9)
    cumulative[-1] = total
    cumulative = np.maximum.accumulate(cumulative)
    return [int(m) for m in np.diff(np.concatenate([[0.0], cumulative]))]


def _best_cumulative(cum: np.ndarray, early: float, late: float, anchor: float) -> float:
    """Return the scheduled time nearest *anchor* that puts the most arrivals in the window.

    Candidate times step by 0.1 minute. Many times usually tie for the most
    on-time arrivals; the one closest to *anchor* (the median-share time) wins,
    so a timepoint only moves as far as its on-time share requires.
    """
    v = np.sort(cum[~np.isnan(cum)])
    if len(v) == 0:
        return anchor
    grid = np.arange(math.floor(v[0] - late), math.ceil(v[-1] - early) + 0.1, 0.1)
    on_time = np.searchsorted(v, grid + late, side="right") - np.searchsorted(
        v, grid + early, side="left"
    )
    best = grid[on_time == on_time.max()]
    return float(best[int(np.argmin(np.abs(best - anchor)))])


def propose_spread(
    tp_obs: pd.DataFrame, runtimes: pd.DataFrame, patterns: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Propose each segment's percent of the runtime for every pattern and period.

    The observed segment time is the median of ``seg_actual_min`` in that
    period; when a segment has no observations there, the median over all
    periods of the pattern is used, then the current scheduled time
    (``observed_source`` says which). ``median_share`` makes each segment's
    percent its share of those medians. ``otp_cumulative`` starts from those
    shares and moves each intermediate timepoint's cumulative time the least
    amount that maximizes its estimated on-time arrivals (early counted),
    never back past the previous timepoint, and lets the last segment take
    the rest of the chosen runtime.

    Returns:
        One row per segment with ``segment_seq``, ``from_stop_id``,
        ``to_stop_id``, ``n_obs``, ``observed_median_min``,
        ``observed_source``, ``proposed_pct``, and ``chosen_pct``.
    """
    seg = tp_obs.loc[tp_obs["tp_seq"] >= 2]
    by_period = {k: g for k, g in seg.groupby(GROUP_KEY, sort=False)}
    all_periods = seg.groupby([*PATTERN_KEY, "tp_seq"])["seg_actual_min"].median().to_dict()
    sched = seg.groupby([*PATTERN_KEY, "tp_seq"])["seg_sched_min"].median().to_dict()

    rows: List[Dict[str, Any]] = []
    for r in runtimes.itertuples(index=False):
        key = tuple(getattr(r, c) for c in GROUP_KEY)
        stops = str(r.pattern_timepoints).split(PATTERN_SEP)
        g = by_period.get(key, seg.iloc[0:0])
        medians, counts, sources = [], [], []
        for tp_seq in range(2, len(stops) + 1):
            vals = g.loc[g["tp_seq"] == tp_seq, "seg_actual_min"].dropna()
            fallback_all = all_periods.get((*key[:3], tp_seq), np.nan)
            fallback_sched = sched.get((*key[:3], tp_seq), np.nan)
            if len(vals):
                medians.append(float(vals.median()))
                sources.append("this period")
            elif pd.notna(fallback_all):
                medians.append(float(fallback_all))
                sources.append("all periods")
            elif pd.notna(fallback_sched):
                medians.append(max(float(fallback_sched), 0.0))
                sources.append("current schedule")
            else:
                medians.append(np.nan)
                sources.append("none (equal split)")
            counts.append(len(vals))
        med = np.asarray(medians, float)
        if np.isnan(med).all() or np.nansum(med) <= 0:
            shares = np.full(len(med), 100.0 / len(med))
        else:
            med_filled = np.where(np.isnan(med), np.nanmean(med), med)
            shares = med_filled / med_filled.sum() * 100.0

        if cfg.spread_method == "otp_cumulative":
            total = float(r.chosen_runtime_min)
            anchors = np.cumsum(shares) / 100.0 * total
            cum_points = [0.0]
            for i, tp_seq in enumerate(range(2, len(stops))):
                cum = g.loc[g["tp_seq"] == tp_seq, "cum_actual_min"].to_numpy(float)
                point = _best_cumulative(cum, cfg.early_min, cfg.late_min, anchors[i])
                cum_points.append(min(max(point, cum_points[-1]), total))
            cum_points.append(total)
            shares = np.diff(cum_points) / total * 100.0

        for i, tp_seq in enumerate(range(2, len(stops) + 1)):
            rows.append(
                {
                    **dict(zip(GROUP_KEY, key)),
                    "pattern": r.pattern,
                    "period": r.period,
                    "period_start": r.period_start,
                    "period_end": r.period_end,
                    "segment_seq": tp_seq - 1,
                    "from_stop_id": stops[tp_seq - 2],
                    "to_stop_id": stops[tp_seq - 1],
                    "n_obs": counts[i],
                    "observed_median_min": round(medians[i], 2) if pd.notna(medians[i]) else np.nan,
                    "observed_source": sources[i],
                    "proposed_pct": round(float(shares[i]), 1),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["chosen_pct"] = out["proposed_pct"]
    return out


def evaluate_spread(
    spread: pd.DataFrame, runtimes: pd.DataFrame, tp_obs: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Turn chosen percents into minutes and estimate arrivals at each timepoint.

    For each timepoint, an observed trip is estimated to arrive early, on
    time, or late by comparing its observed time from the first departure
    (``cum_actual_min``) with the new scheduled cumulative minutes, using the
    ``EARLY_MIN``/``LATE_MIN`` window (early arrivals at the last timepoint
    count as on time when ``DISCOUNT_EARLY_AT_FINAL``). Departures from the
    first timepoint are assumed on time. Early arrivals at an intermediate
    timepoint mean holding (or running hot); late ones are OTP failures.

    Raises:
        ValueError: If a group's chosen percentages are negative or all zero.
    """
    total_by_key = runtimes.set_index(GROUP_KEY)["chosen_runtime_min"].to_dict()
    cum_obs = {k: g for k, g in tp_obs.groupby(GROUP_KEY, sort=False)}
    parts = []
    for key, g in spread.groupby(GROUP_KEY, sort=False):
        g = g.sort_values("segment_seq")
        pcts = g["chosen_pct"].to_numpy(float)
        if abs(pcts.sum() - 100.0) > 0.5:
            logging.warning(
                "Route %s dir %s %s, period %s: chosen percents sum to %.1f; scaled to 100.",
                key[0],
                key[1],
                g["pattern"].iloc[0],
                g["period_start"].iloc[0],
                pcts.sum(),
            )
        try:
            minutes = allocate_minutes(pcts, int(total_by_key[key]), cfg.backload_rounding)
        except ValueError as exc:
            raise ValueError(
                f"Route {key[0]} dir {key[1]} {g['pattern'].iloc[0]}, period "
                f"{g['period_start'].iloc[0]}: {exc}"
            ) from exc
        cumulative = np.cumsum(minutes)
        obs = cum_obs.get(key)
        early_pct, on_pct, late_pct = [], [], []
        n_seg = len(minutes)
        for i, tp_seq in enumerate(g["segment_seq"].to_numpy() + 1):
            if obs is None:
                vals = np.array([])
            else:
                vals = obs.loc[obs["tp_seq"] == tp_seq, "cum_actual_min"].dropna().to_numpy(float)
            if len(vals) == 0:
                early_pct.append(np.nan)
                on_pct.append(np.nan)
                late_pct.append(np.nan)
                continue
            dev = vals - cumulative[i]
            early = dev < cfg.early_min
            late = dev > cfg.late_min
            on = ~late if (i == n_seg - 1 and cfg.discount_early_at_final) else ~early & ~late
            early_pct.append(round(float(early.mean()) * 100.0, 1))
            on_pct.append(round(float(on.mean()) * 100.0, 1))
            late_pct.append(round(float(late.mean()) * 100.0, 1))
        parts.append(
            g.assign(
                runtime_min=int(total_by_key[key]),
                segment_min=minutes,
                cumulative_min=cumulative.astype(int),
                est_arrive_early_pct=early_pct,
                est_on_time_pct=on_pct,
                est_arrive_late_pct=late_pct,
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else spread


# =============================================================================
# STEP 4: RUNTIME TABLE & CHECKS
# =============================================================================


def build_runtime_table(spread: pd.DataFrame) -> pd.DataFrame:
    """Lay out the chosen minutes by timepoint, with a 0-minute row for the first timepoint."""
    rows: List[Dict[str, Any]] = []
    for key, g in spread.groupby(GROUP_KEY, sort=False):
        g = g.sort_values("segment_seq")
        base = {
            **dict(zip(GROUP_KEY, key)),
            "pattern": g["pattern"].iloc[0],
            "period": g["period"].iloc[0],
            "period_start": g["period_start"].iloc[0],
            "period_end": g["period_end"].iloc[0],
            "runtime_min": g["runtime_min"].iloc[0],
        }
        rows.append(
            {
                **base,
                "tp_seq": 1,
                "stop_id": g["from_stop_id"].iloc[0],
                "segment_min": np.nan,
                "cumulative_min": 0,
            }
        )
        for r in g.itertuples(index=False):
            rows.append(
                {
                    **base,
                    "tp_seq": int(r.segment_seq) + 1,
                    "stop_id": r.to_stop_id,
                    "segment_min": r.segment_min,
                    "cumulative_min": r.cumulative_min,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values([*ROUTE_DIR, "pattern", "period", "tp_seq"]).reset_index(drop=True)


def _minutes_between(stops: List[str], cumulative: List[float], a: str, b: str) -> Optional[float]:
    """Return scheduled minutes from timepoint *a* to a later *b* in one pattern, if both appear."""
    if a not in stops:
        return None
    i = stops.index(a)
    if b not in stops[i + 1 :]:
        return None
    j = stops.index(b, i + 1)
    return cumulative[j] - cumulative[i]


def check_pattern_consistency(table: pd.DataFrame) -> List[Dict[str, Any]]:
    """Flag timepoint pairs whose scheduled minutes differ between patterns.

    Within each route, direction, and period, every pair of timepoints that
    are consecutive in at least one pattern is compared with the minutes
    between the same two timepoints in every other pattern that serves both
    in that order (so a pattern skipping a middle timepoint is still compared).
    """
    issues: List[Dict[str, Any]] = []
    for (route, direction, _start), g in table.groupby([*ROUTE_DIR, "period_start_min"]):
        pats = {}
        for pt, pg in g.groupby("pattern_timepoints", sort=False):
            pg = pg.sort_values("tp_seq")
            pats[pt] = (
                pg["stop_id"].tolist(),
                pg["cumulative_min"].astype(float).tolist(),
                pg["pattern"].iloc[0],
            )
        if len(pats) < 2:
            continue
        seen = set()
        for stops, _, _ in pats.values():
            for k in range(1, len(stops)):
                pair = (stops[k - 1], stops[k])
                if pair in seen:
                    continue
                seen.add(pair)
                found = {}
                for q_stops, q_cum, q_label in pats.values():
                    minutes = _minutes_between(q_stops, q_cum, *pair)
                    if minutes is not None:
                        found[q_label] = minutes
                if len(found) >= 2 and len(set(found.values())) > 1:
                    issues.append(
                        {
                            "check": "pattern_mismatch",
                            "route_id": route,
                            "direction_id": direction,
                            "pattern": ", ".join(sorted(found)),
                            "period_start": g["period_start"].iloc[0],
                            "period_end": g["period_end"].iloc[0],
                            "from_stop_id": pair[0],
                            "to_stop_id": pair[1],
                            "detail": "; ".join(
                                f"{label}: {m:g} min" for label, m in sorted(found.items())
                            ),
                        }
                    )
    return issues


def collect_checks(
    runtimes: pd.DataFrame, spread: pd.DataFrame, table: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Gather everything worth a second look into one table."""
    issues: List[Dict[str, Any]] = []
    for r in runtimes.itertuples(index=False):
        base = {
            "route_id": r.route_id,
            "direction_id": r.direction_id,
            "pattern": r.pattern,
            "period_start": r.period_start,
            "period_end": r.period_end,
        }
        if not r.goal_reachable_two_sided:
            issues.append(
                {
                    **base,
                    "check": "otp_goal_unreachable",
                    "detail": (
                        f"Best possible OTP counting early and late is "
                        f"{r.best_two_sided_otp_pct:g}%; {cfg.otp_goal_pct:g}% of runtimes span "
                        f"{r.span_for_goal_min:g} min vs a "
                        f"{cfg.late_min - cfg.early_min:g}-min window."
                    ),
                }
            )
        if r.n_obs < cfg.min_obs_warn:
            issues.append(
                {
                    **base,
                    "check": "thin_data",
                    "detail": f"Runtime based on {r.n_obs} observed trip(s).",
                }
            )
    issues.extend(check_pattern_consistency(table))
    for r in spread.itertuples(index=False):
        base = {
            "route_id": r.route_id,
            "direction_id": r.direction_id,
            "pattern": r.pattern,
            "period_start": r.period_start,
            "period_end": r.period_end,
            "from_stop_id": r.from_stop_id,
            "to_stop_id": r.to_stop_id,
        }
        if r.segment_min == 0:
            issues.append(
                {**base, "check": "zero_minute_segment", "detail": "Segment rounds to 0 minutes."}
            )
        if r.observed_source != "this period":
            issues.append(
                {
                    **base,
                    "check": "segment_not_observed",
                    "detail": f"No observations in this period; used {r.observed_source}.",
                }
            )
    cols = [
        "check",
        "route_id",
        "direction_id",
        "pattern",
        "period_start",
        "period_end",
        "from_stop_id",
        "to_stop_id",
        "detail",
    ]
    return pd.DataFrame(issues, columns=cols)


def build_wide_tables(table: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """One wide table per pattern: a row per period, a column per timepoint (cumulative min)."""
    tables: Dict[str, pd.DataFrame] = {}
    for (route, direction, pt), g in table.groupby(PATTERN_KEY, sort=True):
        stops = pt.split(PATTERN_SEP)
        headers: List[str] = []
        for stop in stops:
            n = sum(1 for h in headers if h == stop or h.startswith(f"{stop} ("))
            headers.append(stop if n == 0 else f"{stop} ({n + 1})")
        rows = []
        for (_, start, end, runtime), pg in g.groupby(
            ["period", "period_start", "period_end", "runtime_min"], sort=True
        ):
            row: Dict[str, Any] = {"period_start": start, "period_end": end, "runtime_min": runtime}
            for header, cum in zip(headers, pg.sort_values("tp_seq")["cumulative_min"]):
                row[header] = cum
            rows.append(row)
        label = g["pattern"].iloc[0]
        tables[f"{_safe_name(route)}_dir{_safe_name(direction)}_{label}"] = pd.DataFrame(rows)
    return tables


def _safe_name(value: Any) -> str:
    """Make *value* safe to use in a filename."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-") or "blank"


# =============================================================================
# INTERACTIVE CHOICES
# =============================================================================


def stdin_is_interactive() -> bool:
    """Return True when ``input()`` can reach a live user.

    True inside a Jupyter/IPython kernel (ipykernel routes ``input()`` to a
    notebook prompt widget) or when stdin is a real terminal. False under
    captured or redirected stdin — CI runners, orchestrator pipelines, cron —
    where an ``input()`` call would hang or crash rather than guide anyone.
    Scripts use this to decide between prompting for missing configuration
    (guided setup) and failing fast with an exit code.

    Canonical implementation: ``utils/cli_helpers.py``.

    Returns:
        True when prompting a user is possible, False otherwise.
    """
    if "ipykernel" in sys.modules:
        return True
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _split_tokens(raw: str) -> List[str]:
    """Split typed input on commas and/or whitespace."""
    return [t for t in re.split(r"[,\s]+", raw.strip()) if t]


def parse_time_list(raw: str) -> List[int]:
    """Parse typed period start times such as ``"05:00, 06:45 9:10"`` into sorted minutes.

    Raises:
        ValueError: If nothing is entered or a token is not a time.
    """
    tokens = _split_tokens(raw)
    if not tokens:
        raise ValueError("Enter at least one start time.")
    minutes = []
    for token in tokens:
        value = parse_time_to_minutes(token)
        if value is None:
            raise ValueError(f"'{token}' is not a time like 06:45.")
        minutes.append(value)
    return sorted(set(minutes))


def parse_number_list(raw: str, expected: int, minimum: float = 0.0) -> List[float]:
    """Parse exactly *expected* typed numbers, each at least *minimum*.

    Raises:
        ValueError: If the count is wrong or a token is not a valid number.
    """
    tokens = _split_tokens(raw)
    if len(tokens) != expected:
        raise ValueError(f"Expected {expected} value(s), got {len(tokens)}.")
    values = []
    for token in tokens:
        try:
            value = float(token.rstrip("%"))
        except ValueError:
            raise ValueError(f"'{token}' is not a number.") from None
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"'{token}' must be at least {minimum:g}.")
        values.append(value)
    return values


def prompt_override(prompt: str, parser: Callable[[str], Any]) -> Tuple[str, Any]:
    """Ask once (re-asking on bad input) whether to accept a proposal or replace it.

    Returns:
        ``("accept", None)`` on blank input, ``("all", None)`` for ``all``,
        else ``("edit", parsed_value)``.

    Raises:
        KeyboardInterrupt: The user cancelled with Ctrl+C.
        EOFError: Stdin closed mid-prompt.
    """
    while True:
        raw = input(prompt).strip()
        if not raw:
            return "accept", None
        if raw.lower() == "all":
            return "all", None
        try:
            return "edit", parser(raw)
        except ValueError as exc:
            logging.warning("%s Try again, or press Enter to accept the proposal.", exc)


def _in_review_order(frame: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Order rows by route, direction, pattern number (P1, P2, ...), then *extra* columns."""
    order = frame.assign(_n=frame["pattern"].str[1:].astype(int))
    return order.sort_values([*ROUTE_DIR, "_n", *extra], kind="mergesort").drop(columns="_n")


def _show(title: str, frame: pd.DataFrame) -> None:
    """Log a small table for the planner to read before a prompt."""
    logging.info("%s\n%s", title, frame.to_string(index=False))


def choose_periods_interactively(proposed: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Show each route-direction's proposed periods and accept typed replacements."""
    described = describe_periods(proposed, summary)
    chosen = []
    accept_rest = False
    for key, p in proposed.groupby(ROUTE_DIR, sort=True):
        if not accept_rest:
            mask = (described["route_id"] == key[0]) & (described["direction_id"] == key[1])
            _show(
                f"Route {key[0]} direction {key[1]}: proposed running time periods",
                described.loc[mask].drop(columns=ROUTE_DIR),
            )
            action, value = prompt_override(
                f"{PROMPT_HINT}, or type period start times (e.g. 05:00, 06:45, 09:10): ",
                parse_time_list,
            )
            if action == "all":
                accept_rest = True
            elif action == "edit":
                p = pd.DataFrame({"route_id": key[0], "direction_id": key[1], "start_min": value})
        chosen.append(p[[*ROUTE_DIR, "start_min"]])
    return number_periods(pd.concat(chosen, ignore_index=True))


def choose_runtimes_interactively(runtimes: pd.DataFrame) -> pd.DataFrame:
    """Show each pattern's proposed runtimes by period and accept typed replacements."""
    out = runtimes.copy()
    show_cols = [
        "period",
        "period_start",
        "period_end",
        "n_obs",
        "current_runtime_min",
        "actual_p50_min",
        "actual_goal_pctile_min",
        "best_two_sided_otp_pct",
        "est_otp_current_pct",
        "proposed_runtime_min",
        "est_otp_proposed_pct",
    ]
    accept_rest = False
    for key, g in _in_review_order(out, ["period"]).groupby(PATTERN_KEY, sort=False):
        if accept_rest:
            break
        _show(
            f"Route {key[0]} direction {key[1]} {g['pattern'].iloc[0]} ({key[2]}): "
            "proposed runtimes",
            g[show_cols],
        )
        action, value = prompt_override(
            f"{PROMPT_HINT}, or type {len(g)} runtime(s) in period order (e.g. 42, 48): ",
            partial(parse_number_list, expected=len(g), minimum=1.0),
        )
        if action == "all":
            accept_rest = True
        elif action == "edit":
            out.loc[g.index, "chosen_runtime_min"] = [int(round(v)) for v in value]
    return out


def choose_spread_interactively(
    spread: pd.DataFrame, runtimes: pd.DataFrame, tp_obs: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Show each pattern-period's proposed percents and accept typed replacements."""
    out = spread.copy()
    preview = evaluate_spread(spread, runtimes, tp_obs, cfg).set_index([*GROUP_KEY, "segment_seq"])
    show_cols = [
        "segment_seq",
        "from_stop_id",
        "to_stop_id",
        "observed_median_min",
        "proposed_pct",
        "segment_min",
        "cumulative_min",
        "est_arrive_early_pct",
        "est_on_time_pct",
        "est_arrive_late_pct",
    ]
    accept_rest = False
    ordered = _in_review_order(out, ["period", "segment_seq"])
    for key, g in ordered.groupby(GROUP_KEY, sort=False):
        if accept_rest:
            break
        shown = preview.loc[[(*key, s) for s in g["segment_seq"]]].reset_index()
        _show(
            f"Route {key[0]} direction {key[1]} {g['pattern'].iloc[0]}, period "
            f"{g['period_start'].iloc[0]}-{g['period_end'].iloc[0] or 'end'} "
            f"({int(shown['runtime_min'].iloc[0])} min): proposed spread",
            shown[show_cols],
        )
        action, value = prompt_override(
            f"{PROMPT_HINT}, or type {len(g)} percent(s) in segment order (e.g. 20, 30, 50): ",
            partial(parse_number_list, expected=len(g), minimum=0.0),
        )
        if action == "all":
            accept_rest = True
        elif action == "edit":
            out.loc[g.index, "chosen_pct"] = value
    return out


# =============================================================================
# CHARTS
# =============================================================================


def _style_axes(ax: Any) -> None:
    """Apply a quiet, single-axis style: hairline grid, muted ticks, no box."""
    ax.set_facecolor(CHART_SURFACE)
    ax.grid(True, color=CHART_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHART_GRID)
    ax.tick_params(colors=CHART_INK_2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: minutes_to_hhmm(x * 60)))


def plot_period_charts(
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    patterns: pd.DataFrame,
    out_dir: Path,
    status: str = "chosen",
) -> List[Path]:
    """Chart each route-direction's trips by start time with the (*status*) period breaks."""
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    labeled = summary.merge(patterns[[*PATTERN_KEY, "pattern"]], on=PATTERN_KEY, how="left")
    labeled = assign_periods(labeled, periods)
    for (route, direction), g in labeled.groupby(ROUTE_DIR, sort=True):
        used = g.loc[g["used_for_breaks"] & g["ratio"].notna()]
        fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=CHART_SURFACE)
        _style_axes(ax)
        for i, (label, pg) in enumerate(used.groupby("pattern", sort=True)):
            is_other = i >= len(CHART_SERIES)
            ax.scatter(
                pg["start_min"] / 60.0,
                pg["ratio"] * 100.0,
                s=36,
                color=CHART_OTHER if is_other else CHART_SERIES[i],
                edgecolors=CHART_SURFACE,
                linewidths=1.5,
                zorder=3,
                label="Other patterns" if i == len(CHART_SERIES) else (None if is_other else label),
            )
        p = periods.loc[(periods["route_id"] == route) & (periods["direction_id"] == direction)]
        last_x = g["start_min"].max() / 60.0 + 0.25
        for j, r in enumerate(p.itertuples(index=False)):
            x0 = r.start_min / 60.0
            x1 = r.end_min / 60.0 if pd.notna(r.end_min) else last_x
            mean = used.loc[used["period"] == r.period, "ratio"].mean() * 100.0
            if j:
                ax.axvline(x0, color=CHART_INK_2, linewidth=0.8, alpha=0.5, zorder=1)
            if pd.notna(mean):
                ax.hlines(
                    mean,
                    x0,
                    x1,
                    color=CHART_INK_2,
                    linewidth=2,
                    zorder=2,
                    label="Period average" if j == 0 else None,
                )
        ax.set_title(
            f"Route {route} direction {direction}: {status} running time periods",
            color=CHART_INK,
            loc="left",
        )
        ax.set_xlabel("Scheduled trip start", color=CHART_INK_2)
        ax.set_ylabel("Goal-percentile runtime, % of pattern typical", color=CHART_INK_2)
        ax.legend(frameon=False, loc="upper left", labelcolor=CHART_INK_2)
        fig.tight_layout()
        path = chart_dir / f"step1_periods_{_safe_name(route)}_dir{_safe_name(direction)}.png"
        fig.savefig(path, dpi=150, facecolor=CHART_SURFACE)
        plt.close(fig)
        written.append(path)
    return written


def plot_runtime_charts(
    trip_obs: pd.DataFrame, runtimes: pd.DataFrame, periods: pd.DataFrame, cfg: Config
) -> List[Path]:
    """Chart each pattern's observed runtimes against the current and chosen runtimes."""
    chart_dir = cfg.output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    trips = trip_obs.dropna(subset=["actual_runtime_min"])
    for key, rt in runtimes.groupby(PATTERN_KEY, sort=True):
        route, direction, pt = key
        g = trips.loc[
            (trips["route_id"] == route)
            & (trips["direction_id"] == direction)
            & (trips["pattern_timepoints"] == pt)
        ]
        if g.empty:
            continue
        label = rt["pattern"].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=CHART_SURFACE)
        _style_axes(ax)
        ax.scatter(
            g["start_min"] / 60.0,
            g["actual_runtime_min"],
            s=14,
            color=CHART_SERIES[0],
            alpha=0.45,
            linewidths=0,
            zorder=2,
            label="Observed runtime",
        )
        current = g.groupby("trip_key").agg(
            start=("start_min", "median"), sched=("sched_runtime_min", "median")
        )
        ax.scatter(
            current["start"] / 60.0,
            current["sched"],
            s=36,
            marker="D",
            color=CHART_SERIES[2],
            edgecolors=CHART_SURFACE,
            linewidths=1.5,
            zorder=4,
            label="Current scheduled runtime",
        )
        p = periods.loc[(periods["route_id"] == route) & (periods["direction_id"] == direction)]
        p = p.set_index("start_min")
        last_x = g["start_min"].max() / 60.0 + 0.25
        first = True
        for r in rt.sort_values("period").itertuples(index=False):
            end = p.loc[r.period_start_min, "end_min"]
            x0 = r.period_start_min / 60.0
            x1 = end / 60.0 if pd.notna(end) else last_x
            ax.fill_between(
                [x0, x1],
                r.chosen_runtime_min + cfg.early_min,
                r.chosen_runtime_min + cfg.late_min,
                color=CHART_SERIES[1],
                alpha=0.10,
                linewidth=0,
                zorder=1,
                label="On-time window" if first else None,
            )
            ax.hlines(
                r.chosen_runtime_min,
                x0,
                x1,
                color=CHART_SERIES[1],
                linewidth=2,
                zorder=3,
                label="Chosen runtime" if first else None,
            )
            if not first:
                ax.axvline(x0, color=CHART_INK_2, linewidth=0.8, alpha=0.5, zorder=1)
            first = False
        stops = pt.split(PATTERN_SEP)
        ax.set_title(
            f"Route {route} direction {direction} {label}: {stops[0]} to {stops[-1]} "
            f"({len(stops)} timepoints)",
            color=CHART_INK,
            loc="left",
        )
        ax.set_xlabel("Scheduled trip start", color=CHART_INK_2)
        ax.set_ylabel("Runtime (minutes)", color=CHART_INK_2)
        ax.legend(frameon=False, loc="upper left", labelcolor=CHART_INK_2)
        fig.tight_layout()
        path = (
            chart_dir / f"step2_runtimes_{_safe_name(route)}_dir{_safe_name(direction)}_{label}.png"
        )
        fig.savefig(path, dpi=150, facecolor=CHART_SURFACE)
        plt.close(fig)
        written.append(path)
    return written


# =============================================================================
# OUTPUT & RUN LOG
# =============================================================================


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write *frame* to *path* (creating folders) and log it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logging.info("Wrote: %s", path)
    return path


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def extract_config_block(source_file: Path) -> str:
    r"""Return the text between the CONFIG markers in *source_file*.

    Reads ``source_file`` as UTF-8 text and slices out the lines strictly
    *between* the first occurrence of ``# === BEGIN CONFIG ===`` and the first
    subsequent occurrence of ``# === END CONFIG ===``.  The marker lines
    themselves are excluded; whitespace and inline comments inside the block
    are preserved verbatim.

    Args:
        source_file: Path to the Python source file to scan (typically
            ``Path(__file__)`` from the calling script).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    _BEGIN = "# === BEGIN CONFIG ==="
    _END = "# === END CONFIG ==="

    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == _BEGIN:
            begin_idx = i
        elif begin_idx is not None and stripped == _END:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. Expected '{_BEGIN}' and '{_END}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / RUN_LOG_FILENAME

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
        "RUNTIME SCHEDULE BUILDER RUN LOG",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "BUILD SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source; CLI flags in the summary above win)",
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


def _finish_run_log(cfg: Config, summary_lines: List[str]) -> None:
    """Write the run log, raising when it fails and ``REQUIRE_RUN_LOG`` is set."""
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )


def describe_settings(cfg: Config) -> List[str]:
    """Return the effective settings (after CLI flags) for the run log."""
    return [
        f"Stop visits:        {cfg.stop_visits_path}",
        f"Trips performed:    {cfg.trips_performed_path}",
        f"Routes:             include={list(cfg.routes_to_include) or 'all'} "
        f"exclude={list(cfg.routes_to_exclude) or 'none'}",
        f"Service dates:      {cfg.service_date_start or 'start'} .. "
        f"{cfg.service_date_end or 'end'}",
        f"Days of week:       {list(cfg.days_of_week)} (Monday=0)",
        f"Holidays dropped:   federal={cfg.exclude_federal_holidays} "
        f"extra={list(cfg.exclude_dates)}",
        f"OTP goal / window:  {cfg.otp_goal_pct:g}% within [{cfg.early_min:g}, "
        f"{cfg.late_min:g}] min; early forgiven at last timepoint={cfg.discount_early_at_final}",
        f"Periods:            num={cfg.num_periods or 'auto'} gvf_target={cfg.period_gvf_target:g} "
        f"max={cfg.max_periods} min_obs_per_trip={cfg.min_obs_per_trip}",
        f"Spread:             method={cfg.spread_method} backload={cfg.backload_rounding}",
    ]


# =============================================================================
# PIPELINE
# =============================================================================


def run(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Execute all four steps, prompting or reading choices, and write all artifacts.

    Each step's CSV is written as soon as its choices are made, so a run
    stopped at a later prompt keeps the earlier choices on disk.

    Returns:
        Mapping with keys ``periods``, ``runtimes``, ``spread``, ``table``,
        ``checks``.

    Raises:
        ValueError: If no usable trips remain or an input is malformed.
    """
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompting = cfg.interactive and stdin_is_interactive()
    if cfg.interactive and not prompting:
        logging.info("No user at the keyboard; using proposals (or CSV choices) without prompts.")
    summary_lines = describe_settings(cfg)

    try:
        stop_visits = load_stop_visits(cfg.stop_visits_path)
        trips = filter_service(load_trips_performed(cfg.trips_performed_path), cfg)
        if trips.empty:
            raise ValueError("No in-service trips remain after the route, date, and day filters.")
        trip_obs, tp_obs = build_observations(stop_visits, trips)
        if trip_obs["actual_runtime_min"].notna().sum() == 0:
            raise ValueError(
                "No trip has an observed runtime. Check that stop_visits marks timepoints and "
                "carries actual times, and that its trips appear in trips_performed."
            )
        patterns = label_patterns(trip_obs)
        logging.info(
            "Using %d performed trips (%d scheduled trips, %d patterns, %d route-directions).",
            len(trip_obs),
            trip_obs["trip_key"].nunique(),
            len(patterns),
            len(patterns[ROUTE_DIR].drop_duplicates()),
        )
        summary_lines.append(
            f"Trips used:         {len(trip_obs)} performed, "
            f"{trip_obs['trip_key'].nunique()} scheduled, {len(patterns)} pattern(s)"
        )

        # --- Step 1: periods ---
        summary = summarize_scheduled_trips(trip_obs, cfg.otp_goal_pct, cfg.min_obs_per_trip)
        proposed = propose_periods(summary, cfg)
        if cfg.make_charts and prompting and not cfg.periods_csv:
            plot_period_charts(summary, proposed, patterns, out_dir, status="proposed")
        if cfg.periods_csv:
            periods = read_periods_csv(cfg.periods_csv, proposed)
            source = f"CSV {cfg.periods_csv}"
        elif prompting:
            logging.info("Step 1 charts: %s", out_dir / "charts")
            periods = choose_periods_interactively(proposed, summary)
            source = "prompts"
        else:
            periods, source = proposed, "proposal"
        summary_lines.append(f"Step 1 periods:     {source}")
        if cfg.make_charts:
            plot_period_charts(summary, periods, patterns, out_dir)
        trip_obs = assign_periods(trip_obs, periods)
        tp_obs = tp_obs.merge(
            trip_obs[[*TRIP_KEY, "period", "period_start_min", "period_start", "period_end"]],
            on=TRIP_KEY,
            how="inner",
        )
        trip_table = assign_periods(summary, periods).merge(
            assign_periods(summary, proposed)[[*PATTERN_KEY, "trip_key", "period"]].rename(
                columns={"period": "proposed_period"}
            ),
            on=[*PATTERN_KEY, "trip_key"],
            how="left",
        )
        trip_table = trip_table.merge(
            patterns[[*PATTERN_KEY, "pattern"]], on=PATTERN_KEY, how="left"
        )
        trip_table["start_hhmm"] = trip_table["start_min"].map(minutes_to_hhmm)
        write_csv(describe_periods(periods, summary), out_dir / PERIODS_FILENAME)
        write_csv(
            trip_table[
                [
                    "route_id",
                    "direction_id",
                    "pattern",
                    "pattern_timepoints",
                    "trip_key",
                    "start_hhmm",
                    "n_obs",
                    "runtime_p50_min",
                    "runtime_goal_pctile_min",
                    "current_runtime_min",
                    "ratio",
                    "used_for_breaks",
                    "proposed_period",
                    "period",
                ]
            ].round(3),
            out_dir / PERIOD_TRIPS_FILENAME,
        )

        # --- Step 2: full runtimes ---
        runtimes = propose_runtimes(trip_obs, patterns, cfg)
        if cfg.runtimes_csv:
            choices = read_choice_csv(cfg.runtimes_csv, "chosen_runtime_min", "Runtimes CSV")
            runtimes = apply_choices(
                runtimes, choices, GROUP_KEY, "chosen_runtime_min", "Runtimes CSV"
            )
            source = f"CSV {cfg.runtimes_csv}"
        elif prompting:
            runtimes = choose_runtimes_interactively(runtimes)
            source = "prompts"
        else:
            source = "proposal"
        runtimes = finalize_runtimes(runtimes, trip_obs, cfg)
        summary_lines.append(f"Step 2 runtimes:    {source}")
        runtime_cols = [
            "route_id",
            "direction_id",
            "pattern",
            "pattern_timepoints",
            "period",
            "period_start",
            "period_end",
            "n_trips",
            "n_obs",
            "current_runtime_min",
            "actual_p50_min",
            "actual_goal_pctile_min",
            "span_for_goal_min",
            "best_two_sided_otp_pct",
            "goal_reachable_two_sided",
            "est_otp_current_pct",
            "proposed_runtime_min",
            "est_otp_proposed_pct",
            "chosen_runtime_min",
            "est_otp_chosen_pct",
        ]
        write_csv(runtimes[runtime_cols], out_dir / RUNTIMES_FILENAME)

        # --- Step 3: percent spread ---
        spread = propose_spread(tp_obs, runtimes, patterns, cfg)
        if cfg.spread_csv:
            choices = read_choice_csv(
                cfg.spread_csv, "chosen_pct", "Spread CSV", extra_keys=["segment_seq"]
            )
            spread = apply_choices(
                spread, choices, [*GROUP_KEY, "segment_seq"], "chosen_pct", "Spread CSV"
            )
            source = f"CSV {cfg.spread_csv}"
        elif prompting:
            spread = choose_spread_interactively(spread, runtimes, tp_obs, cfg)
            source = "prompts"
        else:
            source = "proposal"
        spread = evaluate_spread(spread, runtimes, tp_obs, cfg)
        summary_lines.append(f"Step 3 spread:      {source}")
        spread_cols = [
            "route_id",
            "direction_id",
            "pattern",
            "pattern_timepoints",
            "period",
            "period_start",
            "period_end",
            "runtime_min",
            "segment_seq",
            "from_stop_id",
            "to_stop_id",
            "n_obs",
            "observed_median_min",
            "observed_source",
            "proposed_pct",
            "chosen_pct",
            "segment_min",
            "cumulative_min",
            "est_arrive_early_pct",
            "est_on_time_pct",
            "est_arrive_late_pct",
        ]
        write_csv(spread[spread_cols], out_dir / SPREAD_FILENAME)

        # --- Step 4: runtime table & checks ---
        table = build_runtime_table(spread)
        checks = collect_checks(runtimes, spread, table, cfg)
        write_csv(
            table[
                [
                    "route_id",
                    "direction_id",
                    "pattern",
                    "pattern_timepoints",
                    "period",
                    "period_start",
                    "period_end",
                    "runtime_min",
                    "tp_seq",
                    "stop_id",
                    "segment_min",
                    "cumulative_min",
                ]
            ],
            out_dir / TABLE_FILENAME,
        )
        for name, wide in build_wide_tables(table).items():
            write_csv(wide, out_dir / "runtime_tables" / f"runtime_table_{name}.csv")
        write_csv(checks, out_dir / CHECKS_FILENAME)
        if cfg.make_charts:
            charts = plot_runtime_charts(trip_obs, runtimes, periods, cfg)
            logging.info("Wrote %d runtime chart(s) to %s", len(charts), out_dir / "charts")

        counts = checks["check"].value_counts().to_dict() if not checks.empty else {}
        summary_lines.append(f"Checks:             {counts or 'none'}")
        if counts:
            logging.warning("Items to review in %s: %s", CHECKS_FILENAME, counts)
    except (KeyboardInterrupt, EOFError, OSError, ValueError, RuntimeError) as exc:
        stopped = isinstance(exc, (KeyboardInterrupt, EOFError))
        summary_lines.append(
            "Run stopped at a prompt before finishing."
            if stopped
            else f"Run failed before finishing: {exc}"
        )
        write_run_log(out_dir, summary_lines)
        raise

    _finish_run_log(cfg, summary_lines)
    return {
        "periods": periods,
        "runtimes": runtimes,
        "spread": spread,
        "table": table,
        "checks": checks,
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
        description="Propose and build scheduled running times from TIDES data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--periods-csv",
        default=PERIODS_CSV,
        help="Edited step1_periods.csv to use instead of the proposal (empty = propose).",
    )
    p.add_argument(
        "--runtimes-csv",
        default=RUNTIMES_CSV,
        help="Edited step2_runtimes.csv to use instead of the proposal (empty = propose).",
    )
    p.add_argument(
        "--spread-csv",
        default=SPREAD_CSV,
        help="Edited step3_spread.csv to use instead of the proposal (empty = propose).",
    )
    p.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=INTERACTIVE,
        help="Prompt to accept or change each proposal when a user is present.",
    )
    p.add_argument(
        "--routes",
        default=",".join(ROUTES_TO_INCLUDE),
        help="Comma-separated route_ids to include (empty = all).",
    )
    p.add_argument("--start-date", default=SERVICE_DATE_START, help="First service date.")
    p.add_argument("--end-date", default=SERVICE_DATE_END, help="Last service date.")
    p.add_argument(
        "--days-of-week",
        default=",".join(str(d) for d in DAYS_OF_WEEK),
        help="Comma-separated days to pool, Monday=0 ... Sunday=6.",
    )
    p.add_argument("--otp-goal", type=float, default=OTP_GOAL_PCT, help="Target %% on time.")
    p.add_argument("--early-min", type=float, default=EARLY_MIN, help="Earliest on-time minutes.")
    p.add_argument("--late-min", type=float, default=LATE_MIN, help="Latest on-time minutes.")
    p.add_argument(
        "--discount-early-at-final",
        action=argparse.BooleanOptionalAction,
        default=DISCOUNT_EARLY_AT_FINAL,
        help="Forgive early arrival at the last timepoint.",
    )
    p.add_argument(
        "--num-periods", type=int, default=NUM_PERIODS, help="Periods per route-dir (0 = auto)."
    )
    p.add_argument(
        "--gvf-target", type=float, default=PERIOD_GVF_TARGET, help="GVF for auto period count."
    )
    p.add_argument("--max-periods", type=int, default=MAX_PERIODS, help="Most periods (auto).")
    p.add_argument(
        "--min-obs-per-trip",
        type=int,
        default=MIN_OBS_PER_TRIP,
        help="Observations a trip needs to shape the period breaks.",
    )
    p.add_argument(
        "--spread-method", choices=SPREAD_METHODS, default=SPREAD_METHOD, help="Spread proposal."
    )
    p.add_argument(
        "--backload",
        action=argparse.BooleanOptionalAction,
        default=BACKLOAD_ROUNDING,
        help="Round cumulative minutes down so leftover time falls late in the trip.",
    )
    p.add_argument(
        "--charts", action=argparse.BooleanOptionalAction, default=MAKE_CHARTS, help="Draw PNGs."
    )
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from parsed CLI arguments.

    Raises:
        ValueError: If a list or date argument can't be read.
    """

    def optional_path(value: str) -> Optional[Path]:
        return Path(value).expanduser() if value else None

    try:
        days = tuple(int(d) for d in _split_tokens(args.days_of_week))
    except ValueError:
        raise ValueError(
            f"--days-of-week must be integers 0-6, got '{args.days_of_week}'."
        ) from None
    for value, flag in ((args.start_date, "--start-date"), (args.end_date, "--end-date")):
        if value:
            try:
                pd.Timestamp(value)
            except ValueError:
                raise ValueError(f"{flag} must look like YYYY-MM-DD, got '{value}'.") from None
    routes = tuple(_split_tokens(args.routes))
    return Config(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        periods_csv=optional_path(args.periods_csv),
        runtimes_csv=optional_path(args.runtimes_csv),
        spread_csv=optional_path(args.spread_csv),
        interactive=args.interactive,
        routes_to_include=routes,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
        service_date_start=args.start_date,
        service_date_end=args.end_date,
        days_of_week=days,
        exclude_federal_holidays=EXCLUDE_FEDERAL_HOLIDAYS,
        exclude_dates=EXCLUDE_DATES,
        otp_goal_pct=args.otp_goal,
        early_min=args.early_min,
        late_min=args.late_min,
        discount_early_at_final=args.discount_early_at_final,
        num_periods=args.num_periods,
        period_gvf_target=args.gvf_target,
        max_periods=args.max_periods,
        min_obs_per_trip=args.min_obs_per_trip,
        spread_method=args.spread_method,
        backload_rounding=args.backload,
        min_obs_warn=MIN_OBS_WARN,
        make_charts=args.charts,
    )


def validate_config(cfg: Config) -> List[str]:
    """Return a list of human-readable problems with *cfg* (empty when valid)."""
    problems = []
    if not 0 < cfg.otp_goal_pct <= 100:
        problems.append("OTP_GOAL_PCT must be above 0 and at most 100.")
    if cfg.early_min > 0 or cfg.late_min < 0:
        problems.append("EARLY_MIN must be <= 0 and LATE_MIN must be >= 0.")
    if not set(cfg.days_of_week) <= set(range(7)) or not cfg.days_of_week:
        problems.append("DAYS_OF_WEEK must list days 0 (Monday) to 6 (Sunday).")
    if cfg.num_periods < 0 or cfg.max_periods < 1:
        problems.append("NUM_PERIODS must be >= 0 and MAX_PERIODS >= 1.")
    if not 0 < cfg.period_gvf_target <= 1:
        problems.append("PERIOD_GVF_TARGET must be above 0 and at most 1.")
    if cfg.spread_method not in SPREAD_METHODS:
        problems.append(f"SPREAD_METHOD must be one of {SPREAD_METHODS}.")
    for date in cfg.exclude_dates:
        try:
            pd.Timestamp(date)
        except ValueError:
            problems.append(f"EXCLUDE_DATES has an unreadable date: '{date}'.")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders or invalid.
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

    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2
    problems = validate_config(cfg)
    for problem in problems:
        logging.error("%s", problem)
    if problems:
        return 2

    for path in (
        cfg.stop_visits_path,
        cfg.trips_performed_path,
        cfg.periods_csv,
        cfg.runtimes_csv,
        cfg.spread_csv,
    ):
        if path is not None and not path.exists():
            logging.error("Input file not found: %s", path)
            return 1

    try:
        run(cfg)
    except (KeyboardInterrupt, EOFError):
        logging.warning(
            "Stopped at a prompt. Finished steps were saved in %s; pass them back with "
            "--periods-csv / --runtimes-csv to pick up where you left off.",
            cfg.output_dir,
        )
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
