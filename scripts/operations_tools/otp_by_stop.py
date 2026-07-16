"""Stop-level OTP and AVL data coverage from TIDES, with cross-route flags.

This script computes on-time performance and AVL data coverage per stop (and
per stop/route) from TIDES ``stop_visits`` (stop-level arrival/departure
events) joined to ``trips_performed`` (trip-level attributes). By default only
timepoint visits are scored (``TIMEPOINTS_ONLY``), so "per stop" means per
timepoint stop -- the same scoring rule as ``otp_monthly_panel.py``.

The flagging layered on top exists because route-level rollups (see
``otp_monthly_panel.py``) average over every stop a route serves, so a single
problem stop hides inside the route mean and a stop-specific defect is easy to
misread as a route problem. This script inverts the view: it asks whether a
stop looks bad on *several* of the routes serving it at once. A stop that
underperforms its routes' own baselines on multiple independent routes is
unlikely to be the routes' fault -- it points at the stop itself: a mis-placed
AVL geofence, a stop moved in the field but not in the AVL configuration, a
construction detour, or chronic congestion at that location. Whether the cause
is operational or a data defect, the stop deserves a look, and the per-stop
diagnostics here separate the two failure modes as far as the data allows.

Everything is derived from the TIDES tables alone -- no GTFS feed is required
(see "Limitations" for what that costs).

Method
------
For each (stop, route) pair, over the whole analysis window:

* **Data coverage** -- the share of expected trip-visits that produced a scored
  timepoint record. Expected service is inferred from the export itself: every
  pattern (``pattern_id``, falling back to route + direction) that ever emitted
  a visit at the stop is assumed to serve that stop on every scheduled
  in-service performed trip of that pattern (``trips_performed`` lists those,
  including trips whose AVL never reported). Observed = distinct trips that
  produced at least one scorable visit at the stop.
* **OTP** -- percent of scored visits classified on-time, where a visit is
  on-time when its deviation (actual minus scheduled departure) falls within
  ``[EARLY_MIN, LATE_MIN]``, inclusive.

Each (stop, route) value is then compared to the route's own systemwide
baseline. A route that is uniformly late, or uniformly under-covered because a
garage's AVL units are failing, does not flag every stop it serves -- the gap
against the route baseline isolates the stop-specific component. A (stop,
route) cell is judged *low-coverage* when it is both far below the route
baseline (``COVERAGE_GAP_FLAG_PCT`` points) and below an absolute floor
(``COVERAGE_ABS_FLAG_PCT``); *poor-OTP* is analogous. Cells with too little
data (``MIN_EXPECTED_TRIPS`` / ``MIN_SCORED_VISITS``) are never judged. A stop
is flagged only when at least ``MIN_ROUTES_FLAGGED`` routes independently
agree; set that to 1 to also flag stops served by a single route (at the cost
of no longer being able to distinguish a stop problem from a route problem).

Outputs
-------
  1) ``otp_by_stop.csv`` - one row per stop: pooled coverage/OTP, flags and
     flag reasons, worst gaps versus route baselines, per-cause visit
     diagnostics (skipped, missing-actual, missing-schedule), sorted with
     flagged stops first.
  2) ``otp_by_stop_route_detail.csv`` - the per-(stop, route) table behind
     each flag: expected/observed trips, coverage and OTP with route baselines
     and gaps, and the per-route verdicts.
  3) A run-log sidecar capturing the verbatim CONFIGURATION block.

Reading the diagnostics
-----------------------
``visits_skipped`` (stop marked Skipped) and ``visits_missing_actual``
(schedule present, no AVL timestamp) point at operations or AVL hardware;
``visits_missing_schedule`` (no schedule timestamp on an emitted row) points at
a data defect in the export's schedule join. Low *trip* coverage with clean
visit diagnostics means whole trips leave no trace at the stop -- typical of a
geofence/stop-matching problem when the stop's routes are otherwise healthy.

Limitations (the GTFS-shaped hole)
----------------------------------
Observed-only exports cannot reveal a stop that never appears in them: a stop
whose AVL matching is completely dead for the whole window is invisible here.
Catching that case requires comparing against GTFS ``stop_times``, which is
deliberately out of scope for now. Likewise, expected service is inferred from
the window as a whole, so a stop added to (or dropped from) a pattern mid-window
will look under-covered; run separate windows around known service changes.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# Sentinel markers used by extract_config_block / write_run_log to identify the
# configuration block within this file's source. Each string must appear exactly
# once in this file as a stand-alone comment line. Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# OTP window (minutes). A timepoint departure is "on time" when its deviation
# (actual - scheduled) falls within [EARLY_MIN, LATE_MIN], inclusive. The
# common transit convention is up to 1 minute early through 5 minutes late.
EARLY_MIN: float = -1.0
LATE_MIN: float = 5.0

# Only evaluate stops at timepoint visits (timepoint == TRUE). Set False to
# score every stop visit the export emits.
TIMEPOINTS_ONLY: bool = True

# Optional route filters (matched against route_id as a string). Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# A stop is flagged when at least this many routes independently judge it
# low-coverage (or poor-OTP). 2 is the "across multiple routes" reading that
# separates stop problems from route problems; set 1 to also flag stops served
# by a single route.
MIN_ROUTES_FLAGGED: int = 2

# A (stop, route) cell is low-coverage when its percent-of-trips-observed sits
# at least COVERAGE_GAP_FLAG_PCT points below the route's own baseline AND
# below the absolute floor COVERAGE_ABS_FLAG_PCT. The gap keeps a uniformly
# under-covered route from flagging all of its stops; the floor keeps a small
# dip on a near-perfect route from flagging a stop that is actually fine.
COVERAGE_GAP_FLAG_PCT: float = 15.0
COVERAGE_ABS_FLAG_PCT: float = 80.0

# Same two-part rule for OTP: at least OTP_GAP_FLAG_PCT points below the
# route's baseline AND below the absolute floor OTP_ABS_FLAG_PCT.
OTP_GAP_FLAG_PCT: float = 15.0
OTP_ABS_FLAG_PCT: float = 75.0

# Minimum sample sizes before a (stop, route) cell is judged at all: expected
# in-service trips for the coverage rule, scored visits for the OTP rule.
MIN_EXPECTED_TRIPS: int = 20
MIN_SCORED_VISITS: int = 20

LOG_LEVEL: int = logging.INFO

# Filenames.
OTP_BY_STOP_FILENAME: str = "otp_by_stop.csv"
OTP_BY_STOP_ROUTE_DETAIL_FILENAME: str = "otp_by_stop_route_detail.csv"

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a stop-flagger run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    early_min: float = EARLY_MIN
    late_min: float = LATE_MIN
    timepoints_only: bool = TIMEPOINTS_ONLY
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()
    min_routes_flagged: int = MIN_ROUTES_FLAGGED
    coverage_gap_flag_pct: float = COVERAGE_GAP_FLAG_PCT
    coverage_abs_flag_pct: float = COVERAGE_ABS_FLAG_PCT
    otp_gap_flag_pct: float = OTP_GAP_FLAG_PCT
    otp_abs_flag_pct: float = OTP_ABS_FLAG_PCT
    min_expected_trips: int = MIN_EXPECTED_TRIPS
    min_scored_visits: int = MIN_SCORED_VISITS


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
        attribute columns are needed here).
    """
    return pd.read_csv(path, dtype=str)


def filter_in_service(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Keep scheduled, in-revenue-service performed trips, one row per trip.

    Canceled trips never served any stop and non-revenue trips carry no
    passengers, so neither belongs in the expected-service denominator or the
    OTP pool. This is the same filter ``otp_monthly_panel.py`` applies before
    its join, so the two scripts see the same trip pool.

    Args:
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Filtered copy, deduplicated on ``trip_id_performed``.
    """
    out = trips_performed
    if "schedule_relationship" in out.columns:
        out = out.loc[out["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in out.columns:
        out = out.loc[out["trip_type"].fillna("In service") == "In service"]
    return out.drop_duplicates("trip_id_performed").copy()


def add_pattern_key(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Add a ``pattern_key`` column identifying each trip's stop pattern.

    ``pattern_id`` is the TIDES field for this, but it is optional; when it is
    absent (or blank on a row), ``route_id | direction_id`` is used instead.
    The fallback pools branching variants within a direction, which can
    overstate expected service at branch-only stops -- hence the warning.

    Args:
        trips_performed: In-service trips (output of :func:`filter_in_service`).

    Returns:
        Copy of the input with a non-null string ``pattern_key`` column.
    """
    out = trips_performed.copy()
    direction = (
        out["direction_id"].astype(str)
        if "direction_id" in out.columns
        else pd.Series("", index=out.index)
    )
    fallback = out["route_id"].astype(str) + " | " + direction
    if "pattern_id" in out.columns and out["pattern_id"].notna().any():
        out["pattern_key"] = out["pattern_id"].fillna(fallback)
    else:
        logging.warning(
            "trips_performed has no usable pattern_id; falling back to route + direction "
            "as the pattern key. Branching variants within a direction are pooled, which "
            "can overstate expected service (and understate coverage) at branch-only stops."
        )
        out["pattern_key"] = fallback
    return out


# Attributes carried over from trips_performed onto each stop visit.
_TRIP_ATTR_COLS: List[str] = [
    "route_id",
    "direction_id",
    "route_type_agency",
    "pattern_key",
]


def join_trip_attributes(
    stop_visits: pd.DataFrame,
    trips_performed: pd.DataFrame,
) -> pd.DataFrame:
    """Attach route/direction/pattern attributes to each stop visit.

    The join key is ``trip_id_performed``, unique per performed trip in TIDES.
    The visit table's own ``pattern_id`` (when present) is dropped in favor of
    the trip-level ``pattern_key`` so one pattern definition is used throughout.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.
        trips_performed: In-service trips with a ``pattern_key`` (outputs of
            :func:`filter_in_service` then :func:`add_pattern_key`).

    Returns:
        Stop visits with the ``_TRIP_ATTR_COLS`` attributes joined on (inner
        join, so visits from canceled/non-revenue trips are dropped).
    """
    attr_cols = [c for c in _TRIP_ATTR_COLS if c in trips_performed.columns]
    trips = trips_performed[["trip_id_performed", *attr_cols]]
    visits = stop_visits.drop(columns=["pattern_id"], errors="ignore")
    return visits.merge(trips, on="trip_id_performed", how="inner")


# =============================================================================
# DEVIATION & OTP SCORING
# =============================================================================


def compute_stop_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``dev_min`` column: actual minus scheduled departure, in minutes.

    Departure is the standard reference for OTP. Where a departure timestamp is
    missing (scheduled or actual), the corresponding arrival timestamp is used
    as a fallback so terminal/first stops are still scored when possible.

    Args:
        df: Stop visits with parsed timestamp columns.

    Returns:
        Copy of ``df`` with a float ``dev_min`` column (NaN where neither pair
        of timestamps is available).
    """
    df = df.copy()
    sched = df["schedule_departure_time"].fillna(df["schedule_arrival_time"])
    actual = df["actual_departure_time"].fillna(df["actual_arrival_time"])
    df["dev_min"] = (actual - sched).dt.total_seconds() / 60.0
    return df


def filter_candidate_visits(
    df: pd.DataFrame,
    timepoints_only: bool = TIMEPOINTS_ONLY,
) -> pd.DataFrame:
    """Keep the visits that define each stop's expected-service pool.

    Candidates are every emitted visit at an evaluable stop: timepoints only
    (when configured), minus ``Added`` visits (extra service that was never
    scheduled, so it should not define a pattern's stop membership) and minus
    rows with no ``stop_id`` (nothing to attribute them to). Skipped and
    unscorable visits stay in -- they mark the stop as scheduled-to-be-served
    and feed the per-cause diagnostics.

    Args:
        df: Stop visits with trip attributes and a ``dev_min`` column.
        timepoints_only: When True, retain only ``timepoint == TRUE`` rows.

    Returns:
        Filtered copy suitable for membership inference and diagnostics.
    """
    out = df
    if timepoints_only and "timepoint" in out.columns:
        out = out.loc[out["timepoint"].astype(str).str.upper() == "TRUE"]
    if "schedule_relationship" in out.columns:
        out = out.loc[out["schedule_relationship"].fillna("Scheduled") != "Added"]
    n_no_stop = int(out["stop_id"].isna().sum())
    if n_no_stop:
        logging.warning("Dropping %d visit(s) with no stop_id.", n_no_stop)
        out = out.loc[out["stop_id"].notna()]
    return out.copy()


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


# =============================================================================
# EXPECTED SERVICE (SCHEDULE-FREE DENOMINATOR)
# =============================================================================


def count_pattern_trips(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Count in-service performed trips per pattern.

    Args:
        trips_performed: In-service trips with a ``pattern_key`` column.

    Returns:
        One row per (``pattern_key``, ``route_id``) with its ``n_trips`` count.
    """
    return (
        trips_performed.groupby(["pattern_key", "route_id"])
        .agg(n_trips=("trip_id_performed", "nunique"))
        .reset_index()
    )


def infer_pattern_stop_membership(candidates: pd.DataFrame) -> pd.DataFrame:
    """Infer which stops each pattern serves from the visits it emitted.

    A pattern is taken to serve a stop if any of its trips ever emitted a
    candidate visit there (scored or not -- a Skipped or AVL-dropped visit
    still proves the stop belongs to the pattern). This is the schedule-free
    stand-in for GTFS ``stop_times``.

    Args:
        candidates: Output of :func:`filter_candidate_visits`.

    Returns:
        Unique (``pattern_key``, ``route_id``, ``stop_id``) rows.
    """
    return candidates[["pattern_key", "route_id", "stop_id"]].drop_duplicates(ignore_index=True)


def build_expected_trips(
    membership: pd.DataFrame,
    pattern_trips: pd.DataFrame,
) -> pd.DataFrame:
    """Expected trip-visits per (stop, route): trips of the patterns serving it.

    Because ``trips_performed`` lists every scheduled in-service trip -- even
    ones whose AVL never reported -- this denominator counts whole-trip AVL
    gaps against the stop, exactly like ``otp_monthly_panel.py`` does per route.

    Args:
        membership: Output of :func:`infer_pattern_stop_membership`.
        pattern_trips: Output of :func:`count_pattern_trips`.

    Returns:
        One row per (``stop_id``, ``route_id``) with ``expected_trips``.
    """
    merged = membership.merge(pattern_trips, on=["pattern_key", "route_id"], how="left")
    merged["n_trips"] = merged["n_trips"].fillna(0).astype(int)
    return (
        merged.groupby(["stop_id", "route_id"]).agg(expected_trips=("n_trips", "sum")).reset_index()
    )


# =============================================================================
# STOP x ROUTE DETAIL
# =============================================================================


def _summarize_candidate_visits(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per (stop, route) visit diagnostics split by failure cause.

    Args:
        candidates: Output of :func:`filter_candidate_visits` (with ``dev_min``).

    Returns:
        One row per (``stop_id``, ``route_id``) with ``visits_emitted``,
        ``visits_skipped``, ``visits_missing_actual`` (Scheduled row, schedule
        timestamp present, no scorable actual -- an AVL dropout), and
        ``visits_missing_schedule`` (Scheduled row with no schedule timestamp
        -- an export data defect; rows missing both count here).
    """
    df = candidates.copy()
    if "schedule_relationship" in df.columns:
        rel = df["schedule_relationship"].fillna("Scheduled")
    else:
        rel = pd.Series("Scheduled", index=df.index)
    sched_ts = df["schedule_departure_time"].fillna(df["schedule_arrival_time"])
    unscored = rel.eq("Scheduled") & df["dev_min"].isna()
    df["_skipped"] = rel.eq("Skipped")
    df["_missing_schedule"] = unscored & sched_ts.isna()
    df["_missing_actual"] = unscored & sched_ts.notna()
    return (
        df.groupby(["stop_id", "route_id"])
        .agg(
            visits_emitted=("stop_id", "size"),
            visits_skipped=("_skipped", "sum"),
            visits_missing_actual=("_missing_actual", "sum"),
            visits_missing_schedule=("_missing_schedule", "sum"),
        )
        .reset_index()
    )


def _aggregate_stop_route_otp(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (stop, route) OTP counts and observed-trip counts from scored visits.

    Args:
        scored: Classified visits (post :func:`filter_for_otp` and
            :func:`classify_otp`).

    Returns:
        One row per (``stop_id``, ``route_id``) with ``observed_trips``,
        ``evaluated``, and ``early``/``on_time``/``late`` counts.
    """
    df = scored.copy()
    for cls in ("early", "on_time", "late"):
        df[f"_{cls}"] = df["otp_class"].eq(cls)
    return (
        df.groupby(["stop_id", "route_id"])
        .agg(
            observed_trips=("trip_id_performed", "nunique"),
            evaluated=("otp_class", "size"),
            early=("_early", "sum"),
            on_time=("_on_time", "sum"),
            late=("_late", "sum"),
        )
        .reset_index()
    )


def build_stop_route_detail(
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    trips_performed: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the per-(stop, route) coverage and OTP evidence table.

    Args:
        candidates: Output of :func:`filter_candidate_visits`.
        scored: Classified visits (post :func:`filter_for_otp` and
            :func:`classify_otp`).
        trips_performed: In-service trips with a ``pattern_key`` column.

    Returns:
        One row per (``stop_id``, ``route_id``) carrying expected/observed
        trips, ``pct_trips_observed``, the visit diagnostics, OTP counts, and
        ``pct_on_time`` (NaN when nothing was scored).
    """
    expected = build_expected_trips(
        infer_pattern_stop_membership(candidates),
        count_pattern_trips(trips_performed),
    )
    detail = expected.merge(
        _summarize_candidate_visits(candidates), on=["stop_id", "route_id"], how="left"
    ).merge(_aggregate_stop_route_otp(scored), on=["stop_id", "route_id"], how="left")
    count_cols = [
        "visits_emitted",
        "visits_skipped",
        "visits_missing_actual",
        "visits_missing_schedule",
        "observed_trips",
        "evaluated",
        "early",
        "on_time",
        "late",
    ]
    detail[count_cols] = detail[count_cols].fillna(0).astype(int)
    detail["pct_trips_observed"] = np.where(
        detail["expected_trips"] > 0,
        detail["observed_trips"] / detail["expected_trips"].replace(0, np.nan) * 100.0,
        np.nan,
    ).round(1)
    detail["pct_on_time"] = np.where(
        detail["evaluated"] > 0,
        detail["on_time"] / detail["evaluated"].replace(0, np.nan) * 100.0,
        np.nan,
    ).round(1)
    return detail.sort_values(["stop_id", "route_id"], ignore_index=True)


def attach_route_baselines(detail: pd.DataFrame) -> pd.DataFrame:
    """Add each route's systemwide baselines and the stop-vs-route gaps.

    The coverage baseline is the route's pooled percent-of-expected-trip-visits
    observed (across all its stops); the OTP baseline is the route's pooled
    percent on-time. Negative gaps mean the stop underperforms its route.

    Args:
        detail: Output of :func:`build_stop_route_detail`.

    Returns:
        Copy of ``detail`` with ``route_pct_trips_observed``,
        ``route_pct_on_time``, ``coverage_gap``, and ``otp_gap`` columns.
    """
    out = detail.copy()
    grp = out.groupby("route_id")
    expected_sum = grp["expected_trips"].transform("sum")
    observed_sum = grp["observed_trips"].transform("sum")
    evaluated_sum = grp["evaluated"].transform("sum")
    on_time_sum = grp["on_time"].transform("sum")
    out["route_pct_trips_observed"] = np.where(
        expected_sum > 0,
        observed_sum / expected_sum.replace(0, np.nan) * 100.0,
        np.nan,
    ).round(1)
    out["route_pct_on_time"] = np.where(
        evaluated_sum > 0,
        on_time_sum / evaluated_sum.replace(0, np.nan) * 100.0,
        np.nan,
    ).round(1)
    out["coverage_gap"] = (out["pct_trips_observed"] - out["route_pct_trips_observed"]).round(1)
    out["otp_gap"] = (out["pct_on_time"] - out["route_pct_on_time"]).round(1)
    return out


def evaluate_route_level_flags(detail: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Judge each (stop, route) cell against the coverage and OTP rules.

    A cell is judged only when it clears the minimum sample size
    (``coverage_evaluable`` / ``otp_evaluable``); a judged cell is bad when it
    is both far below the route baseline and below the absolute floor.

    Args:
        detail: Output of :func:`attach_route_baselines`.
        cfg: Resolved configuration (thresholds).

    Returns:
        Copy of ``detail`` with boolean ``coverage_evaluable``,
        ``low_coverage_route``, ``otp_evaluable``, and ``poor_otp_route``
        columns.
    """
    out = detail.copy()
    out["coverage_evaluable"] = out["expected_trips"] >= cfg.min_expected_trips
    out["low_coverage_route"] = (
        out["coverage_evaluable"]
        & (out["coverage_gap"] <= -cfg.coverage_gap_flag_pct)
        & (out["pct_trips_observed"] < cfg.coverage_abs_flag_pct)
    )
    out["otp_evaluable"] = out["evaluated"] >= cfg.min_scored_visits
    out["poor_otp_route"] = (
        out["otp_evaluable"]
        & (out["otp_gap"] <= -cfg.otp_gap_flag_pct)
        & (out["pct_on_time"] < cfg.otp_abs_flag_pct)
    )
    return out


# =============================================================================
# STOP SUMMARY & FLAGS
# =============================================================================


def build_stop_summary(
    detail: pd.DataFrame, min_routes_flagged: int = MIN_ROUTES_FLAGGED
) -> pd.DataFrame:
    """Roll the per-route verdicts up to one flagged/unflagged row per stop.

    Args:
        detail: Output of :func:`evaluate_route_level_flags`.
        min_routes_flagged: Number of routes that must independently judge the
            stop bad before it is flagged.

    Returns:
        One row per ``stop_id`` with pooled coverage/OTP, worst gaps among the
        judged cells, per-cause visit diagnostics, route counts, the two flags,
        and a ``flag_reason`` string -- sorted flagged-first, worst-first.
    """
    df = detail.copy()
    df["_cov_gap_judged"] = df["coverage_gap"].where(df["coverage_evaluable"])
    df["_otp_gap_judged"] = df["otp_gap"].where(df["otp_evaluable"])
    summary = (
        df.groupby("stop_id")
        .agg(
            n_routes=("route_id", "nunique"),
            expected_trips=("expected_trips", "sum"),
            observed_trips=("observed_trips", "sum"),
            evaluated=("evaluated", "sum"),
            on_time=("on_time", "sum"),
            visits_emitted=("visits_emitted", "sum"),
            visits_skipped=("visits_skipped", "sum"),
            visits_missing_actual=("visits_missing_actual", "sum"),
            visits_missing_schedule=("visits_missing_schedule", "sum"),
            routes_coverage_evaluable=("coverage_evaluable", "sum"),
            routes_low_coverage=("low_coverage_route", "sum"),
            routes_otp_evaluable=("otp_evaluable", "sum"),
            routes_poor_otp=("poor_otp_route", "sum"),
            worst_coverage_gap=("_cov_gap_judged", "min"),
            worst_otp_gap=("_otp_gap_judged", "min"),
        )
        .reset_index()
    )
    summary["pct_trips_observed"] = np.where(
        summary["expected_trips"] > 0,
        summary["observed_trips"] / summary["expected_trips"].replace(0, np.nan) * 100.0,
        np.nan,
    ).round(1)
    summary["pct_on_time"] = np.where(
        summary["evaluated"] > 0,
        summary["on_time"] / summary["evaluated"].replace(0, np.nan) * 100.0,
        np.nan,
    ).round(1)
    summary["flag_low_coverage"] = summary["routes_low_coverage"] >= min_routes_flagged
    summary["flag_poor_otp"] = summary["routes_poor_otp"] >= min_routes_flagged
    summary["flag_reason"] = np.select(
        [
            summary["flag_low_coverage"] & summary["flag_poor_otp"],
            summary["flag_low_coverage"],
            summary["flag_poor_otp"],
        ],
        ["low_coverage+poor_otp", "low_coverage", "poor_otp"],
        default="",
    )

    severity = summary[["worst_coverage_gap", "worst_otp_gap"]].min(axis=1)
    summary["_rank"] = summary["flag_low_coverage"].astype(int) + summary["flag_poor_otp"].astype(
        int
    )
    summary["_severity"] = severity
    summary = summary.sort_values(
        ["_rank", "_severity", "stop_id"],
        ascending=[False, True, True],
        ignore_index=True,
    ).drop(columns=["_rank", "_severity"])

    cols = [
        "stop_id",
        "flag_reason",
        "flag_low_coverage",
        "flag_poor_otp",
        "n_routes",
        "expected_trips",
        "observed_trips",
        "pct_trips_observed",
        "worst_coverage_gap",
        "routes_low_coverage",
        "routes_coverage_evaluable",
        "evaluated",
        "on_time",
        "pct_on_time",
        "worst_otp_gap",
        "routes_poor_otp",
        "routes_otp_evaluable",
        "visits_emitted",
        "visits_skipped",
        "visits_missing_actual",
        "visits_missing_schedule",
    ]
    return summary[cols]


# =============================================================================
# OUTPUT
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def export_tables(summary: pd.DataFrame, detail: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Write the stop summary and the per-(stop, route) detail tables.

    Args:
        summary: Output of :func:`build_stop_summary`.
        detail: Output of :func:`evaluate_route_level_flags`.
        out_dir: Directory to write into (created if needed).

    Returns:
        Paths of the files written.
    """
    ensure_dir(out_dir)
    summary_path = out_dir / OTP_BY_STOP_FILENAME
    summary.to_csv(summary_path, index=False)
    detail_path = out_dir / OTP_BY_STOP_ROUTE_DETAIL_FILENAME
    detail.to_csv(detail_path, index=False)
    return [summary_path, detail_path]


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


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "otp_by_stop_runlog.txt"

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
        "TIDES STOP OTP FLAGGER RUN LOG",
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


def run(cfg: Config) -> pd.DataFrame:
    """Execute the full stop-flagging pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        The per-stop summary table (also written to disk).
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)

    # Route filters are applied to trips_performed (route_id's source of truth)
    # so the OTP pool and the expected-service denominator see the same trips.
    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        trips = trips.loc[trips["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        trips = trips.loc[~trips["route_id"].astype(str).isin(drop)]

    trips = add_pattern_key(filter_in_service(trips))
    joined = join_trip_attributes(stop_visits, trips)
    deviated = compute_stop_deviations(joined)
    candidates = filter_candidate_visits(deviated, cfg.timepoints_only)
    scored = candidates.pipe(filter_for_otp, cfg.timepoints_only).pipe(
        classify_otp, cfg.early_min, cfg.late_min
    )

    detail = (
        build_stop_route_detail(candidates, scored, trips)
        .pipe(attach_route_baselines)
        .pipe(evaluate_route_level_flags, cfg)
    )
    summary = build_stop_summary(detail, cfg.min_routes_flagged)

    flagged = summary.loc[summary["flag_reason"] != ""]
    if flagged.empty:
        logging.info(
            "No stops flagged: no stop was judged bad by %d or more routes.",
            cfg.min_routes_flagged,
        )
    else:
        worst = flagged.iloc[0]
        logging.warning(
            "%d stop(s) flagged (%d low-coverage, %d poor-OTP). Worst: stop %s (%s; "
            "coverage %.1f%%, OTP %.1f%%). See %s for per-route evidence.",
            len(flagged),
            int(flagged["flag_low_coverage"].sum()),
            int(flagged["flag_poor_otp"].sum()),
            worst["stop_id"],
            worst["flag_reason"],
            worst["pct_trips_observed"],
            worst["pct_on_time"],
            OTP_BY_STOP_ROUTE_DETAIL_FILENAME,
        )

    paths = export_tables(summary, detail, cfg.output_dir)
    for p in paths:
        logging.info("Wrote table: %s", p)

    summary_lines = [
        f"Stops analyzed:        {summary['stop_id'].nunique()}",
        f"Stop-route pairs:      {len(detail)}",
        f"Routes:                {detail['route_id'].nunique()}",
        f"Stops flagged:         {len(flagged)}",
        f"  low coverage:        {int(summary['flag_low_coverage'].sum())}",
        f"  poor OTP:            {int(summary['flag_poor_otp'].sum())}",
        f"Scored visits:         {int(detail['evaluated'].sum())}",
        f"Expected trip-visits:  {int(detail['expected_trips'].sum())}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return summary


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
        description="Flag stops with poor AVL coverage and/or OTP across multiple routes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
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
        early_min=EARLY_MIN,
        late_min=LATE_MIN,
        timepoints_only=TIMEPOINTS_ONLY,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
        min_routes_flagged=MIN_ROUTES_FLAGGED,
        coverage_gap_flag_pct=COVERAGE_GAP_FLAG_PCT,
        coverage_abs_flag_pct=COVERAGE_ABS_FLAG_PCT,
        otp_gap_flag_pct=OTP_GAP_FLAG_PCT,
        otp_abs_flag_pct=OTP_ABS_FLAG_PCT,
        min_expected_trips=MIN_EXPECTED_TRIPS,
        min_scored_visits=MIN_SCORED_VISITS,
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
