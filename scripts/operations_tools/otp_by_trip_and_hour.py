"""Trip-level and hour-of-day on-time performance (OTP) from TIDES ``stop_visits``.

This script slices on-time performance two ways that the monthly panel
(``otp_monthly_panel.py``) does not: by *scheduled trip* (which trips of the
day chronically run hot or late) and by *hour of the service day* (when in the
day OTP sags), each computed separately per configurable day-type bucket
(default: Weekday, Saturday, Sunday).

It reads a TIDES-style ``stop_visits`` table (stop-level arrival/departure
events), optionally joined to a ``trips_performed`` table (trip-level
attributes such as ``route_id``, ``direction_id``, and the schedule-stable
``trip_id_scheduled``). The scoring rules deliberately mirror
``otp_monthly_panel.py`` -- timepoint-only (``TIMEPOINTS_ONLY``),
``Scheduled``-relationship-only, departure with arrival fallback, inclusive
``[EARLY_MIN, LATE_MIN]`` window -- so "on-time" means the same thing across
this repo's OTP outputs.

Day-type buckets
----------------
``DAY_TYPES`` maps a bucket name to the weekdays it pools; each bucket is
scored separately. Buckets may overlap -- e.g. add ``"Weekend": ("Saturday",
"Sunday")`` alongside the individual days to get a pooled series too (a visit
is then counted once per bucket it belongs to, which is why per-bucket rows
are never summed across buckets). Weekdays in no bucket are dropped, so a
weekend-only analysis is just a two-bucket mapping. The same mapping can be
passed on the CLI as ``--day-types "Weekday=Mon,Tue,Wed,Thu,Fri;Saturday=Sat;
Sunday=Sun"``.

Hour of the service day
-----------------------
Visits are bucketed by their *scheduled* time (a late trip is charged to the
hour it was supposed to serve), measured as hours since midnight of
``service_date`` rather than the timestamp's own clock hour. TIDES timestamps
are full datetimes, so an owl trip scheduled past midnight lands on the next
calendar date while ``service_date`` stays on the operational day; measuring
from the service date keeps that trip at hour 24+ (GTFS-style) instead of
folding it into the next morning's 0-1 AM.

Trip identity
-------------
Pooling a trip across many service dates needs a schedule-stable ID. When
``trips_performed`` is provided, ``trip_id_scheduled`` is used (falling back
to ``trip_id_performed`` where blank); without it the script falls back to
``trip_id_performed`` and warns when those IDs turn out to be unique per day
(each "trip" then spans a single date and the table is per-performed-trip,
not per-scheduled-trip).

Data sanity warnings
--------------------
Checks are limited to what changes the validity of *these* OTP numbers:
duplicated visit rows (double counting), unscorable visits split by cause
(missing schedule vs. missing actual timestamp), deviations beyond
``DEV_ABS_WARN_MIN`` (schedule-join or clock defects hiding inside the
early/late buckets), scheduled times implausibly far from ``service_date``,
and day-type buckets backed by very few service dates. Deeper TIDES schema
validation belongs upstream (in the AVL-to-TIDES converters or a standalone
validator), not duplicated in every consumer script.

Outputs (all in ``OUTPUT_DIR``):
  1) ``otp_by_trip.csv``     - one row per (day type, trip): scheduled start,
     dates observed, early/on-time/late counts and percentages.
  2) ``otp_by_hour.csv``     - one row per (day type, service-day hour).
  3) ``otp_by_day_type.csv`` - one summary row per day type.
  4) ``plots/otp_by_hour_<day_type>.png`` - % on-time by hour, as bars (bars
     rather than a line so hours with no service read as gaps instead of
     being bridged), each annotated with its evaluated visit count.
  5) ``plots/otp_by_trip_[<route>_]<day_type>.png`` - % on-time per trip in
     scheduled-start order (skipped above ``MAX_TRIPS_PER_CHART`` bars).
  6) A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--stop-visits`` /
``--trips-performed`` / ``--output-dir``) and run from a shell, ArcGIS Pro's
Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"

# Optional trips_performed table. When set, route/direction attributes and the
# schedule-stable trip_id_scheduled are joined on, and route filters become
# available. Leave "" to analyze stop_visits alone.
TRIPS_PERFORMED_PATH: str = ""

OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# OTP window (minutes). A timepoint departure is "on time" when its deviation
# (actual - scheduled) falls within [EARLY_MIN, LATE_MIN], inclusive. The
# common transit convention is up to 1 minute early through 5 minutes late.
EARLY_MIN: float = -1.0
LATE_MIN: float = 5.0

# Only evaluate OTP at timepoint stops (timepoint == TRUE). Set False to score
# every Scheduled stop visit.
TIMEPOINTS_ONLY: bool = True

# Day-type buckets: bucket name -> weekdays pooled into it. Each bucket is
# scored separately; buckets may overlap (e.g. add "Weekend": ("Saturday",
# "Sunday") to get a pooled series alongside the individual days). Weekdays in
# no bucket are dropped from the analysis.
DAY_TYPES: Mapping[str, Sequence[str]] = {
    "Weekday": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"),
    "Saturday": ("Saturday",),
    "Sunday": ("Sunday",),
}

# Optional route filters (matched against route_id as a string). Empty = keep
# all. Requires TRIPS_PERFORMED_PATH, which is where route_id lives in TIDES.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# Agency OTP standard (percent) -- drawn as a dashed reference line on charts.
OTP_STANDARD: float = 85.0

# Warn when a scored deviation exceeds this many minutes in either direction;
# such rows are almost always schedule-join or vehicle-clock defects, and they
# sit inside the early/late buckets where they quietly distort percentages.
DEV_ABS_WARN_MIN: float = 120.0

# Warn when a day-type bucket is backed by fewer distinct service dates than
# this (its percentages then hinge on a handful of days).
THIN_DAY_TYPE_DATES: int = 4

# Trip charts with more bars than this are skipped with a pointer to the route
# filter (an unreadable 500-bar chart helps no one; the CSV still has it all).
MAX_TRIPS_PER_CHART: int = 120

LOG_LEVEL: int = logging.INFO

# Filenames
TRIP_FILENAME: str = "otp_by_trip.csv"
HOURLY_FILENAME: str = "otp_by_hour.csv"
DAY_TYPE_FILENAME: str = "otp_by_day_type.csv"

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Weekday-name tokens accepted in DAY_TYPES / --day-types (case-insensitive),
# mapped to pandas dayofweek indices (Monday=0 .. Sunday=6).
_WEEKDAY_ALIASES: Dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

# Scheduled times outside [0, this) hours from service_date midnight are left
# unbucketed (and warned about): even owl service rarely passes hour 30, so
# anything this far out means service_date and the timestamps disagree.
_SERVICE_DAY_HOUR_MAX: int = 36

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for an OTP-by-trip-and-hour run."""

    stop_visits_path: Path
    output_dir: Path
    trips_performed_path: Optional[Path] = None
    early_min: float = EARLY_MIN
    late_min: float = LATE_MIN
    timepoints_only: bool = TIMEPOINTS_ONLY
    day_types: Mapping[str, Sequence[int]] = field(default_factory=dict)
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()
    otp_standard: float = OTP_STANDARD
    make_plots: bool = True


# =============================================================================
# DAY-TYPE BUCKETS
# =============================================================================


def normalize_day_types(raw: Mapping[str, Sequence[str]]) -> Dict[str, List[int]]:
    """Resolve weekday-name buckets to pandas dayofweek indices (Mon=0..Sun=6).

    Args:
        raw: Mapping of bucket name -> weekday names (full names or 3-letter
            abbreviations, case-insensitive).

    Returns:
        Mapping of bucket name -> sorted, de-duplicated weekday indices, in the
        original bucket order.

    Raises:
        ValueError: On an empty mapping, an empty bucket, or an unrecognized
            weekday token.
    """
    if not raw:
        raise ValueError("No day-type buckets configured; at least one is required.")
    out: Dict[str, List[int]] = {}
    for name, day_names in raw.items():
        label = str(name).strip()
        if not label:
            raise ValueError("Day-type bucket names must be non-empty.")
        days: List[int] = []
        for token in day_names:
            key = str(token).strip().lower()
            if key not in _WEEKDAY_ALIASES:
                raise ValueError(
                    f"Unknown weekday {token!r} in day-type bucket {label!r}; expected "
                    "full names or 3-letter abbreviations (e.g. 'Monday' or 'Mon')."
                )
            idx = _WEEKDAY_ALIASES[key]
            if idx not in days:
                days.append(idx)
        if not days:
            raise ValueError(f"Day-type bucket {label!r} lists no weekdays.")
        out[label] = sorted(days)

    counts: Dict[int, int] = {}
    for days in out.values():
        for idx in days:
            counts[idx] = counts.get(idx, 0) + 1
    if any(n > 1 for n in counts.values()):
        logging.info(
            "Day-type buckets overlap; overlapping days are counted once per bucket "
            "(never sum rows across buckets)."
        )
    return out


def serialize_day_types(raw: Mapping[str, Sequence[str]]) -> str:
    """Render a day-type mapping in the ``--day-types`` CLI spec format."""
    return ";".join(f"{name}={','.join(str(d) for d in days)}" for name, days in raw.items())


def parse_day_types_spec(spec: str) -> Dict[str, List[int]]:
    """Parse a ``--day-types`` spec like ``"Weekday=Mon,Tue;Saturday=Sat"``.

    Buckets are separated by ``;``, each as ``Name=day,day,...`` with weekday
    tokens as accepted by :func:`normalize_day_types`.

    Args:
        spec: The CLI spec string.

    Returns:
        Mapping of bucket name -> weekday indices.

    Raises:
        ValueError: On a malformed spec, duplicate bucket name, or unknown
            weekday token.
    """
    raw: Dict[str, List[str]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Bad --day-types bucket {part!r}; expected 'Name=Mon,Tue,...' entries "
                "separated by ';'."
            )
        name, _, days_txt = part.partition("=")
        name = name.strip()
        if name in raw:
            raise ValueError(f"Duplicate day-type bucket name {name!r} in --day-types.")
        raw[name] = [tok for tok in (t.strip() for t in days_txt.split(",")) if tok]
    return normalize_day_types(raw)


def add_day_type(df: pd.DataFrame, day_types: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    """Replicate rows into their day-type buckets via a ``day_type`` column.

    A row lands in every bucket whose weekdays contain its ``service_date``'s
    weekday, so overlapping buckets (e.g. Saturday + a pooled Weekend) each get
    their own copy. Rows whose weekday is in no bucket are dropped.

    Args:
        df: Stop visits with a parsed ``service_date`` column.
        day_types: Output of :func:`normalize_day_types`.

    Returns:
        Concatenated per-bucket copies of ``df`` with a ``day_type`` column.
    """
    dow = df["service_date"].dt.dayofweek
    frames: List[pd.DataFrame] = []
    for name, days in day_types.items():
        sub = df.loc[dow.isin(list(days))].copy()
        if sub.empty:
            logging.info("Day type %r matched no visits.", name)
            continue
        sub["day_type"] = name
        frames.append(sub)

    covered = {idx for days in day_types.values() for idx in days}
    n_dropped = int((~dow.isin(sorted(covered)) & dow.notna()).sum())
    if n_dropped:
        logging.info("Dropped %d visits on weekdays outside every day-type bucket.", n_dropped)

    if not frames:
        out = df.iloc[0:0].copy()
        out["day_type"] = pd.Series(dtype="object")
        return out
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read a TIDES ``stop_visits`` CSV and parse its timestamp columns.

    Args:
        path: Path to the ``stop_visits`` CSV export.

    Returns:
        DataFrame with the four schedule/actual timestamp columns parsed to
        datetimes and ``service_date`` parsed to a date.
    """
    df = pd.read_csv(path, dtype=str)
    for col in (
        "schedule_arrival_time",
        "schedule_departure_time",
        "actual_arrival_time",
        "actual_departure_time",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read a TIDES ``trips_performed`` CSV (trip-level attributes).

    Args:
        path: Path to the ``trips_performed`` CSV export.

    Returns:
        DataFrame with string columns (timestamps left as strings; only the
        attribute columns are needed for the join here).
    """
    return pd.read_csv(path, dtype=str)


# Attributes carried over from trips_performed onto each stop visit.
_TRIP_ATTR_COLS: List[str] = [
    "trip_id_scheduled",
    "route_id",
    "direction_id",
    "route_type_agency",
]


def join_trip_attributes(
    stop_visits: pd.DataFrame,
    trips_performed: pd.DataFrame,
) -> pd.DataFrame:
    """Attach route/direction/scheduled-trip attributes to each stop visit.

    Trips that were Canceled (or not in revenue service) in ``trips_performed``
    are dropped, since their stop visits are not meaningful for OTP. The join
    key is ``trip_id_performed``, unique per performed trip in TIDES.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Stop visits with the ``_TRIP_ATTR_COLS`` attributes joined on.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [c for c in _TRIP_ATTR_COLS if c in trips.columns]
    trips = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")

    return stop_visits.merge(trips, on="trip_id_performed", how="inner")


def assign_trip_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``trip_id`` column used to pool a trip across service dates.

    Prefers ``trip_id_scheduled`` (stable across dates by construction), falling
    back to ``trip_id_performed`` where it is blank or when the column is absent
    entirely (stop_visits-only runs).

    Args:
        df: Stop visits, optionally carrying ``trip_id_scheduled``.

    Returns:
        Copy of ``df`` with a ``trip_id`` column.
    """
    df = df.copy()
    if "trip_id_scheduled" in df.columns and df["trip_id_scheduled"].notna().any():
        df["trip_id"] = df["trip_id_scheduled"].fillna(df["trip_id_performed"])
        logging.info("Trip identity: trip_id_scheduled (trip_id_performed where blank).")
    else:
        df["trip_id"] = df["trip_id_performed"]
        logging.info(
            "Trip identity: trip_id_performed (no trips_performed table / no "
            "trip_id_scheduled column)."
        )
    return df


# =============================================================================
# DATA SANITY CHECKS
# =============================================================================


def sanity_check_stop_visits(sv: pd.DataFrame) -> None:
    """Log warnings for input defects that would distort the OTP tables.

    Checks unparseable ``service_date`` values and duplicated visit rows (same
    service date, performed trip, and stop sequence -- each duplicate counts
    twice in every percentage). Rows are never dropped here; the warnings exist
    so oddities are chased upstream (converter or a TIDES validator) rather
    than silently absorbed.

    Args:
        sv: Output of :func:`load_stop_visits`.
    """
    bad_dates = int(sv["service_date"].isna().sum())
    if bad_dates:
        logging.warning(
            "%d rows have an unparseable service_date; they cannot be bucketed by "
            "day type and will drop out.",
            bad_dates,
        )

    dup_key = ["service_date", "trip_id_performed", "trip_stop_sequence"]
    if set(dup_key) <= set(sv.columns):
        n_dupes = int(sv.duplicated(subset=dup_key, keep="first").sum())
        if n_dupes:
            logging.warning(
                "%d duplicated visit rows (same service_date, trip_id_performed, "
                "trip_stop_sequence); each duplicate is double-counted in the "
                "percentages -- deduplicate upstream.",
                n_dupes,
            )


def summarize_unscorable(
    df: pd.DataFrame,
    timepoints_only: bool = TIMEPOINTS_ONLY,
) -> Dict[str, int]:
    """Break down why otherwise-eligible visits could not be scored.

    ``dev_min`` going NaN has two very different causes: no actual timestamp
    (an AVL dropout on a row the export chose to emit anyway) versus no
    schedule timestamp (a data-quality defect in the export's schedule join).
    In an observed-only export the first should be near zero, so a nonzero
    ``missing_schedule_time`` is the number to chase with the AVL vendor.

    Args:
        df: Stop visits with a ``dev_min`` column, before :func:`filter_for_otp`.
        timepoints_only: Apply the same timepoint filter as scoring, so the
            counts describe the same candidate pool.

    Returns:
        Dict with ``candidates`` (eligible visits), ``scored`` (finite
        deviation), ``missing_actual_time``, and ``missing_schedule_time``.
        Rows missing both timestamps count under ``missing_schedule_time``.
    """
    sub = df
    if timepoints_only and "timepoint" in sub.columns:
        sub = sub.loc[sub["timepoint"].astype(str).str.upper() == "TRUE"]
    if "schedule_relationship" in sub.columns:
        sub = sub.loc[sub["schedule_relationship"].fillna("Scheduled") == "Scheduled"]

    unscorable = sub.loc[sub["dev_min"].isna()]
    if "schedule_departure_time" in unscorable.columns:
        sched = unscorable["schedule_departure_time"]
        if "schedule_arrival_time" in unscorable.columns:
            sched = sched.fillna(unscorable["schedule_arrival_time"])
    else:
        sched = pd.Series(pd.NaT, index=unscorable.index)
    missing_schedule = int(sched.isna().sum())

    return {
        "candidates": int(len(sub)),
        "scored": int(len(sub) - len(unscorable)),
        "missing_actual_time": int(len(unscorable) - missing_schedule),
        "missing_schedule_time": missing_schedule,
    }


def warn_extreme_deviations(scored: pd.DataFrame, warn_abs_min: float = DEV_ABS_WARN_MIN) -> None:
    """Warn when scored deviations exceed ``warn_abs_min`` minutes either way.

    Such rows are kept (they classify as early/late like any other), but a
    multi-hour "deviation" is nearly always a schedule joined to the wrong
    trip or a vehicle clock defect, and enough of them will drag the early/late
    percentages away from operational reality.

    Args:
        scored: Visits with a finite ``dev_min`` column.
        warn_abs_min: Absolute deviation threshold in minutes.
    """
    extreme = scored["dev_min"].abs() > warn_abs_min
    if extreme.any():
        logging.warning(
            "%d scored visits deviate more than %.0f minutes from schedule "
            "(worst: %.0f min). These classify as early/late but usually indicate "
            "schedule-join or clock defects -- inspect before trusting the splits.",
            int(extreme.sum()),
            warn_abs_min,
            float(scored.loc[extreme, "dev_min"].abs().max()),
        )


def warn_thin_day_types(scored: pd.DataFrame, min_dates: int = THIN_DAY_TYPE_DATES) -> None:
    """Warn for day-type buckets observed on fewer than ``min_dates`` dates.

    Args:
        scored: Day-typed scored visits.
        min_dates: Minimum distinct service dates for a bucket to pass quietly.
    """
    per_bucket = scored.groupby("day_type", dropna=False)["service_date"].nunique()
    for name, n_dates in per_bucket.items():
        if int(n_dates) < min_dates:
            logging.warning(
                "Day type %r is backed by only %d service date(s); its percentages "
                "hinge on a handful of days.",
                name,
                int(n_dates),
            )


def warn_if_trip_ids_do_not_repeat(trip_table: pd.DataFrame, n_dates_total: int) -> None:
    """Warn when trip pooling degenerated to one service date per trip.

    Args:
        trip_table: Output of :func:`build_trip_table`.
        n_dates_total: Distinct service dates in the scored data.
    """
    if trip_table.empty or n_dates_total < 2:
        return
    if int(trip_table["n_dates"].max()) == 1:
        logging.warning(
            "No trip ID repeats across the %d service dates, so each 'trip' row covers "
            "a single date. Provide trips_performed (for trip_id_scheduled) to pool "
            "trips across dates.",
            n_dates_total,
        )


# =============================================================================
# DEVIATION & OTP SCORING
# =============================================================================


def compute_stop_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``dev_min`` (actual minus scheduled departure, minutes) and ``sched_used_time``.

    Departure is the standard reference for OTP. Where a departure timestamp is
    missing (scheduled or actual), the corresponding arrival timestamp is used
    as a fallback so terminal/first stops are still scored when possible --
    the same rule as ``otp_monthly_panel.py``. The scheduled timestamp the
    deviation was measured against is kept as ``sched_used_time`` so hour-of-day
    bucketing charges each visit to its scheduled hour.

    Args:
        df: Stop visits with parsed timestamp columns.

    Returns:
        Copy of ``df`` with float ``dev_min`` (NaN where either side is missing)
        and datetime ``sched_used_time`` columns.
    """
    df = df.copy()
    sched = df["schedule_departure_time"].fillna(df["schedule_arrival_time"])
    actual = df["actual_departure_time"].fillna(df["actual_arrival_time"])
    df["dev_min"] = (actual - sched).dt.total_seconds() / 60.0
    df["sched_used_time"] = sched
    return df


def filter_for_otp(df: pd.DataFrame, timepoints_only: bool = TIMEPOINTS_ONLY) -> pd.DataFrame:
    """Keep only stop visits that can be scored for OTP.

    Drops non-timepoint visits (when ``timepoints_only``), visits whose
    ``schedule_relationship`` is not ``Scheduled`` (Skipped/Added carry no
    comparable used time), and visits with a missing deviation.

    Args:
        df: Stop visits with a ``dev_min`` column.
        timepoints_only: When True, retain only ``timepoint == TRUE`` rows.

    Returns:
        Filtered copy suitable for OTP aggregation.
    """
    out = df
    if timepoints_only and "timepoint" in out.columns:
        out = out.loc[out["timepoint"].astype(str).str.upper() == "TRUE"]
    if "schedule_relationship" in out.columns:
        out = out.loc[out["schedule_relationship"].fillna("Scheduled") == "Scheduled"]
    out = out.loc[out["dev_min"].notna()]
    return out.copy()


def classify_otp(
    df: pd.DataFrame,
    early_min: float = EARLY_MIN,
    late_min: float = LATE_MIN,
) -> pd.DataFrame:
    """Classify each scored visit as ``early``/``on_time``/``late``.

    Args:
        df: Stop visits with a ``dev_min`` column.
        early_min: Lower (inclusive) bound of the on-time window, in minutes
            (typically negative, e.g. -1.0).
        late_min: Upper (inclusive) bound of the on-time window, in minutes.

    Returns:
        Copy of ``df`` with a string ``otp_class`` column.
    """
    df = df.copy()
    dev = df["dev_min"]
    conditions = [dev < early_min, dev > late_min]
    df["otp_class"] = np.select(conditions, ["early", "late"], default="on_time")
    return df


def add_service_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sched_minutes`` and ``hour``: scheduled time measured from service-date midnight.

    ``hour`` is the whole hour since midnight of ``service_date`` (GTFS-style:
    an owl visit scheduled 1:30 AM on the next calendar date is hour 25, not
    hour 1). Hours outside ``[0, _SERVICE_DAY_HOUR_MAX)`` mean the timestamps
    and ``service_date`` disagree; those are warned about and left blank rather
    than charged to a fictitious hour.

    Args:
        df: Stop visits with ``sched_used_time`` and parsed ``service_date``.

    Returns:
        Copy of ``df`` with float ``sched_minutes`` and nullable-int ``hour``.
    """
    df = df.copy()
    minutes = (df["sched_used_time"] - df["service_date"]).dt.total_seconds() / 60.0
    hour = np.floor(minutes / 60.0)
    in_range = minutes.notna() & (hour >= 0) & (hour < _SERVICE_DAY_HOUR_MAX)
    n_out = int((minutes.notna() & ~in_range).sum())
    if n_out:
        logging.warning(
            "%d visits have a scheduled time before service_date or %d+ hours after "
            "it; their hour-of-day is left blank -- check that service_date matches "
            "the timestamps (post-midnight trips should keep the operational date).",
            n_out,
            _SERVICE_DAY_HOUR_MAX,
        )
    df["sched_minutes"] = minutes
    df["hour"] = hour.where(in_range).astype("Int64")
    return df


# =============================================================================
# AGGREGATION
# =============================================================================


def aggregate_otp(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate scored visits to OTP counts and percentages per group.

    Args:
        df: Classified stop visits (must contain ``otp_class``).
        group_cols: Grouping columns (must be non-empty; every table here
            groups at least by ``day_type``).

    Returns:
        Tidy DataFrame with one row per group and columns ``early``,
        ``on_time``, ``late``, ``evaluated``, ``pct_on_time``, ``pct_early``,
        ``pct_late``.
    """
    keys = list(group_cols)
    counts = df.assign(_n=1).pivot_table(
        index=keys,
        columns="otp_class",
        values="_n",
        aggfunc="sum",
        fill_value=0,
        dropna=False,
    )
    for cls in ("early", "on_time", "late"):
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[["early", "on_time", "late"]]
    counts["evaluated"] = counts.sum(axis=1)
    # pivot_table(dropna=False) expands multi-key indexes to the cartesian
    # product of key values; combinations that never occurred (0 visits) are
    # artifacts, not groups, so they are dropped rather than reported as 0%.
    counts = counts.loc[counts["evaluated"] > 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        counts["pct_on_time"] = counts["on_time"] / counts["evaluated"] * 100.0
        counts["pct_early"] = counts["early"] / counts["evaluated"] * 100.0
        counts["pct_late"] = counts["late"] / counts["evaluated"] * 100.0

    return counts.reset_index().sort_values(keys).reset_index(drop=True)


def _sort_by_day_type(
    df: pd.DataFrame, day_type_order: Sequence[str], then_by: Sequence[str]
) -> pd.DataFrame:
    """Sort ``df`` by configured day-type order, then by ``then_by`` columns."""
    rank = {name: i for i, name in enumerate(day_type_order)}
    out = df.assign(_rank=df["day_type"].map(rank))
    out = out.sort_values(["_rank", *then_by], kind="stable").drop(columns="_rank")
    return out.reset_index(drop=True)


def _format_service_minutes(minutes: float) -> str:
    """Render minutes-from-service-midnight as a GTFS-style clock (may pass 24:00)."""
    total = int(round(minutes))
    return f"{total // 60:02d}:{total % 60:02d}"


def build_trip_table(scored: pd.DataFrame, day_type_order: Sequence[str]) -> pd.DataFrame:
    """One row per (day type, trip): scheduled start, dates observed, OTP splits.

    The scheduled start is the median across dates of each date's earliest
    scheduled timepoint time, in service-day clock terms (so an owl trip can
    read ``24:15``). Route/direction columns are carried when present.

    Args:
        scored: Day-typed, classified visits with ``trip_id``, ``sched_minutes``.
        day_type_order: Bucket order for sorting (configured DAY_TYPES order).

    Returns:
        Tidy trip-level OTP table sorted by day type then scheduled start.
    """
    keys = ["day_type"]
    keys += [c for c in ("route_id", "direction_id") if c in scored.columns]
    keys += ["trip_id"]

    agg = aggregate_otp(scored, keys)

    n_dates = (
        scored.groupby(keys, dropna=False).agg(n_dates=("service_date", "nunique")).reset_index()
    )
    agg = agg.merge(n_dates, on=keys, how="left")

    per_date_start = (
        scored.groupby([*keys, "service_date"], dropna=False)
        .agg(_start=("sched_minutes", "min"))
        .reset_index()
    )
    starts = (
        per_date_start.groupby(keys, dropna=False)
        .agg(scheduled_start_minutes=("_start", "median"))
        .reset_index()
    )
    agg = agg.merge(starts, on=keys, how="left")
    agg["scheduled_start_minutes"] = agg["scheduled_start_minutes"].round(0)
    agg["scheduled_start"] = agg["scheduled_start_minutes"].map(
        lambda m: _format_service_minutes(m) if pd.notna(m) else pd.NA
    )

    front = [*keys, "scheduled_start", "scheduled_start_minutes", "n_dates"]
    ordered = front + [c for c in agg.columns if c not in front]
    agg = agg[ordered]
    return _sort_by_day_type(agg, day_type_order, ["scheduled_start_minutes", "trip_id"])


def build_hourly_table(scored: pd.DataFrame, day_type_order: Sequence[str]) -> pd.DataFrame:
    """One row per (day type, service-day hour) with OTP splits.

    Args:
        scored: Day-typed, classified visits with the ``hour`` column.
        day_type_order: Bucket order for sorting (configured DAY_TYPES order).

    Returns:
        Tidy hourly OTP table sorted by day type then hour.
    """
    sub = scored.loc[scored["hour"].notna()].copy()
    if sub.empty:
        cols = ("day_type", "hour", "early", "on_time", "late", "evaluated")
        pcts = ("pct_on_time", "pct_early", "pct_late")
        return pd.DataFrame({c: [] for c in cols + pcts})
    sub["hour"] = sub["hour"].astype(int)
    agg = aggregate_otp(sub, ["day_type", "hour"])
    return _sort_by_day_type(agg, day_type_order, ["hour"])


def build_day_type_table(scored: pd.DataFrame, day_type_order: Sequence[str]) -> pd.DataFrame:
    """One summary row per day type: OTP splits plus dates and trips observed.

    Args:
        scored: Day-typed, classified visits with ``trip_id``.
        day_type_order: Bucket order for sorting (configured DAY_TYPES order).

    Returns:
        Tidy day-type summary table.
    """
    agg = aggregate_otp(scored, ["day_type"])
    extras = (
        scored.groupby("day_type", dropna=False)
        .agg(n_dates=("service_date", "nunique"), n_trips=("trip_id", "nunique"))
        .reset_index()
    )
    agg = agg.merge(extras, on="day_type", how="left")
    front = ["day_type", "n_dates", "n_trips"]
    agg = agg[front + [c for c in agg.columns if c not in front]]
    return _sort_by_day_type(agg, day_type_order, [])


# =============================================================================
# OUTPUT
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _slug(value: object) -> str:
    """Return a filesystem-safe token for a group identifier."""
    txt = str(value).strip()
    for ch in (" ", "|", "/", "\\", ":"):
        txt = txt.replace(ch, "_")
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt.strip("_") or "group"


def plot_hourly(
    hourly: pd.DataFrame,
    out_dir: Path,
    otp_standard: float,
    day_type_order: Sequence[str],
) -> List[Path]:
    """Render % on-time by service-day hour as bar charts, one PNG per day type.

    Bars are used (rather than a connected line) so hours with no service read
    as gaps instead of being bridged. Each bar is annotated with its evaluated
    visit count so thin hours are visible at a glance.

    Args:
        hourly: Output of :func:`build_hourly_table`.
        out_dir: Output directory; charts land under ``out_dir/plots``.
        otp_standard: Dashed reference line (percent).
        day_type_order: Bucket order (chart iteration order).

    Returns:
        Paths of the PNG files written.
    """
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []

    for day_type in day_type_order:
        g = hourly.loc[hourly["day_type"] == day_type].sort_values("hour")
        if g.empty:
            continue
        hours = g["hour"].to_numpy(dtype=int)
        pct = g["pct_on_time"].to_numpy(dtype=float)
        n = g["evaluated"].to_numpy(dtype=int)

        plt.figure(figsize=(10, 5))
        plt.bar(hours, pct, width=0.8, label="% On-time")
        plt.axhline(
            y=otp_standard,
            linestyle="--",
            color="red",
            linewidth=1,
            label=f"OTP Standard ({otp_standard:.0f}%)",
        )
        for h, p, cnt in zip(hours, pct, n):
            if np.isfinite(p):
                plt.annotate(
                    f"n={cnt}",
                    (h, p),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=7,
                    rotation=90 if cnt >= 1000 else 0,
                )
        plt.xticks(range(int(hours.min()), int(hours.max()) + 1))
        plt.ylim(0, 105)
        plt.xlabel("Scheduled service-day hour (24+ = after midnight)")
        plt.ylabel("% On-time")
        plt.title(f"{day_type} - OTP by hour of day")
        plt.legend()
        plt.tight_layout()
        out_path = plots_dir / f"otp_by_hour_{_slug(day_type)}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        written.append(out_path)

    return written


def plot_trips(
    trip_table: pd.DataFrame,
    out_dir: Path,
    otp_standard: float,
    max_trips_per_chart: int = MAX_TRIPS_PER_CHART,
) -> List[Path]:
    """Render % on-time per trip in scheduled-start order, one PNG per group.

    Charts are grouped by day type (and by route, when route attributes are
    present) so each stays readable; a group with more trips than
    ``max_trips_per_chart`` is skipped with a pointer to the route filter.

    Args:
        trip_table: Output of :func:`build_trip_table`.
        out_dir: Output directory; charts land under ``out_dir/plots``.
        otp_standard: Dashed reference line (percent).
        max_trips_per_chart: Bar-count ceiling per chart.

    Returns:
        Paths of the PNG files written.
    """
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []

    group_cols = ["day_type"] + (["route_id"] if "route_id" in trip_table.columns else [])
    for group_key, g in trip_table.groupby(group_cols, sort=False):
        parts = list(group_key) if isinstance(group_key, tuple) else [group_key]
        label = " | ".join(str(p) for p in parts)
        if len(g) > max_trips_per_chart:
            logging.warning(
                "Skipping trip chart for %s: %d trips exceeds MAX_TRIPS_PER_CHART=%d. "
                "Filter to a route (ROUTES_TO_INCLUDE / --routes) for a readable chart; "
                "the CSV still carries every trip.",
                label,
                len(g),
                max_trips_per_chart,
            )
            continue
        g = g.sort_values("scheduled_start_minutes", kind="stable")
        x = np.arange(len(g))
        labels = g["scheduled_start"].fillna("?").astype(str).tolist()

        plt.figure(figsize=(max(8.0, len(g) * 0.18), 5))
        plt.bar(x, g["pct_on_time"].to_numpy(dtype=float), width=0.8, label="% On-time")
        plt.axhline(
            y=otp_standard,
            linestyle="--",
            color="red",
            linewidth=1,
            label=f"OTP Standard ({otp_standard:.0f}%)",
        )
        step = max(1, int(np.ceil(len(g) / 30)))
        plt.xticks(ticks=x[::step], labels=labels[::step], rotation=90, fontsize=7)
        plt.ylim(0, 100)
        plt.xlabel("Trip (by scheduled start)")
        plt.ylabel("% On-time")
        plt.title(f"{label} - OTP by trip")
        plt.legend()
        plt.tight_layout()
        out_path = plots_dir / f"otp_by_trip_{_slug(label)}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        written.append(out_path)

    return written


# =============================================================================
# RUN LOG
# =============================================================================


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
    log_path = output_dir / "otp_by_trip_and_hour_runlog.txt"

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
        "OTP BY TRIP AND HOUR RUN LOG",
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
    """Execute the full trip/hour OTP pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        Mapping of table name (``trip`` / ``hourly`` / ``day_type``) to the
        DataFrame written for it.

    Raises:
        ValueError: When a route filter is requested but ``trips_performed``
            carries no ``route_id`` column.
        RuntimeError: When no visits survive scoring and day-type bucketing,
            or the run log cannot be written while ``REQUIRE_RUN_LOG`` is set.
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    logging.info("Read %d stop visits from %s", len(stop_visits), cfg.stop_visits_path)
    sanity_check_stop_visits(stop_visits)

    if cfg.trips_performed_path is not None:
        trips = load_trips_performed(cfg.trips_performed_path)
        # Route filters are applied to trips_performed (route_id's source of
        # truth), then the inner join narrows stop_visits to the kept trips.
        if (cfg.routes_to_include or cfg.routes_to_exclude) and "route_id" not in trips.columns:
            raise ValueError(
                f"trips_performed '{cfg.trips_performed_path}' has no route_id column, "
                "so the configured route filter cannot be applied."
            )
        if cfg.routes_to_include:
            keep = {str(r) for r in cfg.routes_to_include}
            trips = trips.loc[trips["route_id"].astype(str).isin(keep)]
        if cfg.routes_to_exclude:
            drop = {str(r) for r in cfg.routes_to_exclude}
            trips = trips.loc[~trips["route_id"].astype(str).isin(drop)]
        joined = join_trip_attributes(stop_visits, trips)
        logging.info("Joined trips_performed: %d visits carry trip attributes.", len(joined))
    else:
        joined = stop_visits

    deviated = compute_stop_deviations(joined)
    unscorable = summarize_unscorable(deviated, cfg.timepoints_only)
    if unscorable["missing_schedule_time"]:
        logging.warning(
            "%d of %d eligible timepoint visits lack a schedule timestamp and cannot "
            "be scored -- a data-quality defect in the export's schedule join, not an "
            "AVL gap.",
            unscorable["missing_schedule_time"],
            unscorable["candidates"],
        )
    if unscorable["missing_actual_time"]:
        logging.info(
            "%d of %d eligible timepoint visits have a schedule but no actual "
            "timestamp (within-row AVL dropouts).",
            unscorable["missing_actual_time"],
            unscorable["candidates"],
        )

    scored = (
        deviated.pipe(filter_for_otp, cfg.timepoints_only)
        .pipe(classify_otp, cfg.early_min, cfg.late_min)
        .pipe(add_service_hour)
        .pipe(assign_trip_id)
    )
    warn_extreme_deviations(scored)

    scored = add_day_type(scored, cfg.day_types)
    if scored.empty:
        raise RuntimeError(
            "No visits remain after scoring and day-type bucketing; nothing to "
            "aggregate. Check DAY_TYPES and the input's service_date/timestamps."
        )
    warn_thin_day_types(scored)

    day_type_order = list(cfg.day_types)
    trip_table = build_trip_table(scored, day_type_order)
    hourly_table = build_hourly_table(scored, day_type_order)
    day_type_table = build_day_type_table(scored, day_type_order)

    n_dates_total = int(scored["service_date"].nunique())
    warn_if_trip_ids_do_not_repeat(trip_table, n_dates_total)

    ensure_dir(cfg.output_dir)
    for name, table in (
        (TRIP_FILENAME, trip_table),
        (HOURLY_FILENAME, hourly_table),
        (DAY_TYPE_FILENAME, day_type_table),
    ):
        path = cfg.output_dir / name
        table.to_csv(path, index=False)
        logging.info("Wrote table: %s", path)

    if cfg.make_plots:
        plot_paths = plot_hourly(hourly_table, cfg.output_dir, cfg.otp_standard, day_type_order)
        plot_paths += plot_trips(trip_table, cfg.output_dir, cfg.otp_standard)
        logging.info("Wrote %d OTP charts to %s", len(plot_paths), cfg.output_dir / "plots")

    for _, r in day_type_table.iterrows():
        logging.info(
            "%s: %.1f%% on time (%d evaluated over %d date(s); %.1f%% early, %.1f%% late).",
            r["day_type"],
            r["pct_on_time"],
            r["evaluated"],
            r["n_dates"],
            r["pct_early"],
            r["pct_late"],
        )

    summary_lines = [
        f"Stop visits read:   {len(stop_visits)}",
        f"Visits scored:      {unscorable['scored']} of {unscorable['candidates']} eligible",
        f"Service dates:      {n_dates_total}",
        f"Day types:          {', '.join(day_type_order)}",
        f"Trips in table:     {len(trip_table)} rows",
        f"OTP window (min):   [{cfg.early_min:g}, {cfg.late_min:g}]",
        f"Route filter:       include={list(cfg.routes_to_include) or 'all'} "
        f"exclude={list(cfg.routes_to_exclude) or 'none'}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return {"trip": trip_table, "hourly": hourly_table, "day_type": day_type_table}


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


def _split_csv_arg(value: str) -> List[str]:
    """Split a comma-separated CLI value into stripped, non-empty tokens."""
    return [tok for tok in (t.strip() for t in value.split(",")) if tok]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Trip-level and hour-of-day OTP from TIDES stop_visits, per day type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed",
        default=TRIPS_PERFORMED_PATH,
        help="Path to trips_performed CSV (optional; enables route filters and "
        "schedule-stable trip IDs).",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument("--early-min", type=float, default=EARLY_MIN, help="On-time window lower bound.")
    p.add_argument("--late-min", type=float, default=LATE_MIN, help="On-time window upper bound.")
    p.add_argument(
        "--day-types",
        default=serialize_day_types(DAY_TYPES),
        help="Day-type buckets as 'Name=Mon,Tue,...' entries separated by ';'.",
    )
    p.add_argument(
        "--routes",
        default=",".join(str(r) for r in ROUTES_TO_INCLUDE),
        help="Comma-separated route_ids to include (empty = all; needs --trips-performed).",
    )
    p.add_argument(
        "--exclude-routes",
        default=",".join(str(r) for r in ROUTES_TO_EXCLUDE),
        help="Comma-separated route_ids to exclude (needs --trips-performed).",
    )
    p.add_argument("--otp-standard", type=float, default=OTP_STANDARD, help="OTP standard (%%).")
    p.add_argument("--no-plots", action="store_true", help="Disable writing PNG charts.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders or inconsistent.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    if args.stop_visits == STOP_VISITS_PATH:
        logging.warning(
            "STOP_VISITS_PATH is still a placeholder. Update the CONFIGURATION section "
            "or pass --stop-visits before running."
        )
        return 2

    try:
        day_types = parse_day_types_spec(args.day_types)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    routes_include = _split_csv_arg(args.routes)
    routes_exclude = _split_csv_arg(args.exclude_routes)
    trips_performed = Path(args.trips_performed).expanduser() if args.trips_performed else None
    if trips_performed is None and (routes_include or routes_exclude):
        logging.error(
            "A route filter is configured but no trips_performed table was given. "
            "route_id lives in trips_performed in TIDES -- set TRIPS_PERFORMED_PATH "
            "(or pass --trips-performed) to filter by route."
        )
        return 2

    cfg = Config(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        trips_performed_path=trips_performed,
        early_min=args.early_min,
        late_min=args.late_min,
        timepoints_only=TIMEPOINTS_ONLY,
        day_types=day_types,
        routes_to_include=routes_include,
        routes_to_exclude=routes_exclude,
        otp_standard=args.otp_standard,
        make_plots=not args.no_plots,
    )

    if not cfg.stop_visits_path.exists():
        logging.warning("stop_visits not found: %s", cfg.stop_visits_path)
        return 1
    if cfg.trips_performed_path is not None and not cfg.trips_performed_path.exists():
        logging.warning("trips_performed not found: %s", cfg.trips_performed_path)
        return 1

    try:
        run(cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
