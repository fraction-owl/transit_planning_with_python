"""Export TIDES on-time performance and running times at every grain in one run.

Reads TIDES ``stop_visits`` joined to ``trips_performed`` and writes one CSV per grain for
three measures: on-time performance (OTP) at timepoint visits, whole-trip running time,
and running time between consecutive timepoints (segments). A grain is an aggregation
level crossed with a time cut; ``EXPORT_MEASURES``, ``EXPORT_LEVELS`` and
``EXPORT_TIME_CUTS`` switch each one on or off, and all are on by default.

- Levels: system, service_type, route, route_direction, stop and route_stop (OTP only),
  trip, and segment (segment running time only).
- Time cuts: all_day, time_period, hour, month, month_time_period, day_of_week.

Every table is split by day type, found from GTFS and ``HOLIDAYS`` the same way as in
``ridership_from_tides.py``. ``TRIP_TIME_BASIS`` places a trip in a time period and hour by
its scheduled (default) or actual start; ``STOP_TIME_BASIS`` places a stop visit or a
segment by the scheduled (default) or actual time at its stop. Time periods come from
``TIME_PERIODS``, a ``TIME_PERIODS_FILE`` (see ``sample_data/time_periods_example.txt``)
or ``--time-period`` flags, and must cover the day exactly once.

Inputs
------
- TIDES ``stop_visits.csv`` and ``trips_performed.csv`` (for example from
  ``convert_to_tides_stop_visits.py`` and ``convert_to_tides_trips_performed.py``).
- Optional: GTFS feeds covering the TIDES dates, and a time-periods ``.txt`` file.

Outputs
-------
- ``otp/otp_by_<level>[_and_<cut>].csv``: early / on-time / late counts and shares and
  the deviation from schedule; ``otp_stop_flags.csv`` and ``otp_stop_route_detail.csv``
  flag stops whose data coverage or OTP lags on several routes.
- ``trip_runtime/``: actual vs. scheduled running time per grain, each trip's
  observations and trimmed outliers, and variation, data-gap and day-of-week flags.
- ``segment_runtime/``: the same per timepoint segment, each traversal, and one table
  per route, direction and day type with the segments in stop order.
- ``date_classification.csv``, ``trip_coverage.csv`` (with GTFS), ``charts/`` and a
  run log with the CONFIGURATION block.

Limitations
-----------
- Timestamps are read as written, ignoring any UTC offset, so a running time that
  crosses a daylight-saving change is off by an hour.
- Canceled and non-revenue trips are dropped. OTP scores Scheduled visits only, and
  running times leave out Skipped visits. A trip's last stop is scored on arrival;
  ``LAST_STOP_TREATMENT`` can count an early arrival there as on time, or leave the
  stop out, on express routes (``EXPRESS_ROUTES_FILE``) or every route.

Typical usage
-------------
Update the paths in CONFIGURATION (or pass ``--stop-visits`` / ``--trips-performed`` /
``--output-dir``) and run from a shell, ArcGIS Pro's Python window, or a Jupyter notebook.

See Also:
    - ``ridership_from_tides.py``
    - ``otp_monthly_panel.py``
    - ``otp_by_route.py``
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import logging
import os
import re
import sys
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import (
    IO,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# MEASURES --------------------------------------------------------------------
#   otp             - on-time performance at stop visits (timepoints by default)
#   trip_runtime    - running time from a trip's first served stop to its last
#   segment_runtime - running time between consecutive timepoints on a trip
EXPORT_MEASURES: Mapping[str, bool] = {
    "otp": True,
    "trip_runtime": True,
    "segment_runtime": True,
}

# GRAINS ----------------------------------------------------------------------
# Aggregation levels. Each level switched on is written once for every time cut
# switched on below, for each measure it applies to. Set an entry to False to skip it.
#   system          - all routes pooled
#   service_type    - per route_type_agency (e.g. LOCAL, EXPRESS)
#   route           - per route_id
#   route_direction - per route_id x direction_id
#   stop            - per stop_id, all routes pooled (OTP only)
#   route_stop      - per route_id x stop_id, both directions pooled (OTP only)
#   trip            - per scheduled trip (trip_id_scheduled, else trip_id_performed)
#   segment         - per route_id x direction_id x timepoint pair (segment running time only)
EXPORT_LEVELS: Mapping[str, bool] = {
    "system": True,
    "service_type": True,
    "route": True,
    "route_direction": True,
    "stop": True,
    "route_stop": True,
    "trip": True,
    "segment": True,
}

# Time cuts, each crossed with every level switched on above:
#   all_day           - one row per day type
#   time_period       - per TIME_PERIODS window (see the TIME BASIS settings)
#   hour              - per hour of the service day (see LATE_NIGHT_CUTOVER_HOUR)
#   month             - per calendar month of service_date
#   month_time_period - per month x time period
#   day_of_week       - per weekday of service_date (Monday ... Sunday)
EXPORT_TIME_CUTS: Mapping[str, bool] = {
    "all_day": True,
    "time_period": True,
    "hour": True,
    "month": True,
    "month_time_period": True,
    "day_of_week": True,
}

# TIME BASIS ------------------------------------------------------------------
# The clock time that places a whole trip (trip running time) in a time period and
# an hour: "scheduled_start" (schedule_trip_start, else the scheduled departure at
# its first served stop) or "actual_start" (actual_trip_start, else the actual
# departure at its first served stop).
TRIP_TIME_BASIS: str = "scheduled_start"
# The clock time that places a stop visit (OTP) or a segment (by the time at its
# first timepoint) in a time period and an hour: "scheduled" (scheduled departure,
# else arrival) or "actual" (actual departure, else arrival); a trip's last stop
# uses arrival first. Scheduled time charges a late bus to the period it was
# supposed to serve.
STOP_TIME_BASIS: str = "scheduled"

# Clock hours before this belong to the previous service day and appear as hours 24
# and up in the hour cut (a 00:30 visit is hour 24). The default 4 gives the usual
# 04:00-27:59 service day.
LATE_NIGHT_CUTOVER_HOUR: int = 4

# TIME PERIODS ----------------------------------------------------------------
# Time periods as {name: ("HH:MM", "HH:MM")} on the 24-hour clock, start inclusive
# and end exclusive. An end earlier than the start wraps past midnight, and "24:00"
# may be used as an end. Together the periods must cover every minute of the day
# exactly once; the script stops before reading any data if there is a gap or an
# overlap. Output rows follow the order listed.
TIME_PERIODS: Mapping[str, Tuple[str, str]] = {
    "Early AM": ("04:00", "06:00"),
    "AM Peak": ("06:00", "09:00"),
    "Midday": ("09:00", "15:00"),
    "PM Peak": ("15:00", "18:00"),
    "Late PM": ("18:00", "22:00"),
    "Night": ("22:00", "04:00"),
}

# Optional plain-text file of time periods. When set, it replaces TIME_PERIODS.
# One period per line as "NAME = HH:MM - HH:MM"; blank lines and anything after
# "#" are ignored. See sample_data/time_periods_example.txt. The same checks
# apply. --time-period flags on the command line replace both.
TIME_PERIODS_FILE: Optional[str] = None  # e.g. r"C:\Data\time_periods.txt"

# DAY TYPES -------------------------------------------------------------------
# Day types as {name: weekdays} (full names or three-letter abbreviations). Every
# table is split by day type, in this order. A weekday in no entry is left out; a
# weekday in several entries counts in each. To pool every date, use
# {"All days": ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")}.
DAY_TYPES: Mapping[str, Sequence[str]] = {
    "Weekday": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"),
    "Saturday": ("Saturday",),
    "Sunday": ("Sunday",),
}

# GTFS feeds (folders or .zip files) whose calendars cover the TIDES dates, the
# recommended source of holidays. List a series oldest first; where two feeds
# schedule service on the same date, the later feed is used. A date becomes a
# holiday when none of the services scheduled that day is a regular service for
# its weekday (holiday-only service on Labor Day, or Sunday service on a Wednesday
# Veterans Day). Leave empty to rely on HOLIDAYS alone.
GTFS_PATHS: Sequence[str] = ()

# Extra holiday dates as "YYYY-MM-DD", added to any GTFS finds.
HOLIDAYS: Sequence[str] = ()
# Name of the day type holidays form; they leave the day type of their weekday.
HOLIDAY_DAY_TYPE: str = "Holiday"

# Optional service-date range as "YYYY-MM-DD", both ends inclusive. None = no limit.
START_DATE: Optional[str] = None
END_DATE: Optional[str] = None

# DATE CHECKS -----------------------------------------------------------------
# date_classification.csv flags a service date when more than
# MAX_UNSCHEDULED_SHARE of the revenue trips that ran are not scheduled that day in
# GTFS, when GTFS schedules no service that day, when more than MAX_CANCELED_SHARE
# of the day's revenue trips are Canceled, when more than MAX_UNRECORDED_SHARE of
# the trips GTFS schedules that day have no trips_performed row at all, or when it
# has conflicting duplicate stop visits. trip_coverage.csv (with GTFS) flags a
# scheduled trip with no record on more than MAX_TRIP_UNRECORDED_SHARE of its
# scheduled dates. Set a share to 1.0 to switch its check off.
MAX_UNSCHEDULED_SHARE: float = 0.10
MAX_CANCELED_SHARE: float = 0.20
MAX_UNRECORDED_SHARE: float = 0.50
MAX_TRIP_UNRECORDED_SHARE: float = 0.50
# False keeps flagged dates in every table (they are only listed). True leaves them
# out of every table.
EXCLUDE_FLAGGED_DATES: bool = False

# FILTERS ---------------------------------------------------------------------
# Optional route filters, matched against route_id as text. Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# ON-TIME PERFORMANCE ---------------------------------------------------------
# A visit is on time when its deviation (actual minus scheduled departure, falling
# back to arrival where a departure is missing; arrival at a trip's last stop, see
# below) falls within [EARLY_MIN, LATE_MIN] minutes, inclusive. The common
# convention is 1 minute early to 5 minutes late.
EARLY_MIN: float = -1.0
LATE_MIN: float = 5.0
# Score timepoint visits only (timepoint TRUE). False scores every Scheduled visit.
TIMEPOINTS_ONLY: bool = True
# Agency OTP standard (percent), drawn as a dashed line on the charts.
OTP_STANDARD: float = 85.0
# Warn when a scored deviation exceeds this many minutes either way; such rows are
# almost always schedule-join or vehicle-clock defects.
DEV_ABS_WARN_MIN: float = 120.0
# Warn when a day type is backed by fewer service dates than this.
THIN_DAY_TYPE_DATES: int = 4

# Last stop: a trip's trip_end_stop_id (trips_performed) at its last visit. OTP there
# is scored on arrival (actual minus scheduled arrival, each falling back to
# departure), since a bus leaving the end of the line is no longer serving riders.
# Trips without a trip_end_stop_id are scored on departure like any other stop.
# LAST_STOP_TREATMENT sets how the last stop counts on the routes LAST_STOP_ROUTES
# picks ("express" or "all"); every other route uses "score":
#   "score"    - score it like any other timepoint.
#   "early_ok" - riders only get off there, so an early arrival counts as on time.
#                A late arrival still counts as late, and dev_min keeps the
#                measured deviation.
#   "exclude"  - leave it out of OTP.
LAST_STOP_TREATMENT: str = "score"
LAST_STOP_ROUTES: str = "express"
# Express routes: the route_ids in EXPRESS_ROUTES_FILE (a .txt with one route_id per
# line; see sample_data/optional/express_routes.txt) and in EXPRESS_ROUTES together.
# When neither lists any, routes whose route_type_agency is in EXPRESS_SERVICE_TYPES
# (compared without regard to case) are express.
EXPRESS_ROUTES_FILE: Optional[str] = None  # e.g. r"C:\Data\express_routes.txt"
EXPRESS_ROUTES: Sequence[str] = ()
EXPRESS_SERVICE_TYPES: Sequence[str] = ("EXPRESS",)

# Stop flags (otp_stop_flags.csv): a stop is flagged when at least
# MIN_ROUTES_FLAGGED of its routes each judge it low-coverage or poor-OTP. A
# (stop, route) cell is low-coverage when its share of expected trips observed is
# at least COVERAGE_GAP_FLAG_PCT points below the route's own share AND below
# COVERAGE_ABS_FLAG_PCT; poor-OTP uses OTP_GAP_FLAG_PCT and OTP_ABS_FLAG_PCT the
# same way. Cells with fewer than MIN_EXPECTED_TRIPS expected trips (coverage) or
# MIN_SCORED_VISITS scored visits (OTP) are not judged.
EXPORT_STOP_FLAGS: bool = True
MIN_ROUTES_FLAGGED: int = 2
COVERAGE_GAP_FLAG_PCT: float = 15.0
COVERAGE_ABS_FLAG_PCT: float = 80.0
OTP_GAP_FLAG_PCT: float = 15.0
OTP_ABS_FLAG_PCT: float = 75.0
MIN_EXPECTED_TRIPS: int = 20
MIN_SCORED_VISITS: int = 20

# RUNNING TIME ----------------------------------------------------------------
# Drop the shortest and longest TRIM_FRAC of each scheduled trip's running times
# before any trip running-time table is built (0.01 = 1%; 0 switches trimming off).
# Trimmed trips are written to trip_runtime_outliers.csv. Segments are not trimmed.
TRIM_FRAC: float = 0.01
# Percentiles (0-100) of the actual running time reported in every running-time
# table, each as an actual_pNN_min column, beside the mean and median.
RUNTIME_PERCENTILES: Sequence[float] = (1, 5, 85, 95, 99)
# Build segments between consecutive timepoints only. False builds them between
# every pair of consecutive served stops.
SEGMENT_TIMEPOINTS_ONLY: bool = True

# Trip flags (trip_runtime_flags.csv): a trip is "high_variation" when its running
# time's coefficient of variation exceeds HIGH_CV_THRESHOLD with at least
# MIN_OBS_FOR_CV observations, and a "data_gap" when it has fewer than GAP_FRAC x
# the median observation count of the trips in its day type.
# trip_runtime_dow_flags.csv marks a trip's day of week "low_count" when it has
# fewer than DOW_LOW_COUNT_FRAC x the trip's median per-weekday count, and a
# "runtime_anomaly" when its mean differs from the trip's mean by more than
# DOW_RUNTIME_PCT (a fraction).
HIGH_CV_THRESHOLD: float = 0.15
MIN_OBS_FOR_CV: int = 5
GAP_FRAC: float = 0.30
DOW_LOW_COUNT_FRAC: float = 0.40
DOW_RUNTIME_PCT: float = 0.10

# CHARTS ----------------------------------------------------------------------
# Bar charts of % on time and of running time by time period, hour, month and day
# of week for CHART_LEVELS (any of system, service_type, route, route_direction),
# one PNG per group and day type; per-trip % on-time charts and running-time box
# plots per route, direction and day type. Per-trip charts with more trips than
# MAX_TRIPS_PER_CHART are skipped (the CSVs hold every trip).
EXPORT_CHARTS: bool = True
CHART_LEVELS: Sequence[str] = ("system", "route")
MAX_TRIPS_PER_CHART: int = 120

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# CONSTANTS
# =============================================================================

# Level -> its key columns as (column, required). A level whose required column is
# missing from the data is skipped with a warning; optional columns (the route and
# direction beside a trip or segment) are carried when present.
LEVELS: Dict[str, Tuple[Tuple[str, bool], ...]] = {
    "system": (),
    "service_type": (("route_type_agency", True),),
    "route": (("route_id", True),),
    "route_direction": (("route_id", True), ("direction_id", True)),
    "stop": (("stop_id", True),),
    "route_stop": (("route_id", True), ("stop_id", True)),
    "trip": (("route_id", False), ("direction_id", False), ("trip_id", True)),
    "segment": (
        ("route_id", True),
        ("direction_id", False),
        ("from_stop_id", True),
        ("to_stop_id", True),
    ),
}

# Measure -> the levels it is reported at, in output order.
MEASURE_LEVELS: Dict[str, Tuple[str, ...]] = {
    "otp": ("system", "service_type", "route", "route_direction", "stop", "route_stop", "trip"),
    "trip_runtime": ("system", "service_type", "route", "route_direction", "trip"),
    "segment_runtime": ("segment",),
}

# Time cut -> the time columns it adds to each level's keys.
TIME_CUTS: Dict[str, Tuple[str, ...]] = {
    "all_day": (),
    "time_period": ("time_period",),
    "hour": ("hour",),
    "month": ("month",),
    "month_time_period": ("month", "time_period"),
    "day_of_week": ("day_of_week",),
}

# Time columns that come from the service date rather than a clock time.
DATE_CUT_COLUMNS: Tuple[str, ...] = ("month", "day_of_week")

# Time cuts drawn as charts, with one bar per period, hour, month or weekday.
CHART_CUTS: Tuple[str, ...] = ("time_period", "hour", "month", "day_of_week")

# Levels that can be charted (one chart per group).
CHARTABLE_LEVELS: Tuple[str, ...] = ("system", "service_type", "route", "route_direction")

TRIP_TIME_BASES: Tuple[str, ...] = ("scheduled_start", "actual_start")
STOP_TIME_BASES: Tuple[str, ...] = ("scheduled", "actual")
LAST_STOP_TREATMENTS: Tuple[str, ...] = ("score", "early_ok", "exclude")
LAST_STOP_ROUTE_CHOICES: Tuple[str, ...] = ("express", "all")

WEEKDAY_NAMES: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

MINUTES_PER_DAY: int = 24 * 60
BLANK_LABEL: str = "(blank)"
UNCLASSIFIED: str = "UNCLASSIFIED"

# TIDES identifies a performed trip by (service_date, trip_id_performed).
TRIP_KEY: List[str] = ["service_date", "trip_id_performed"]

# The four stop_visits timestamps, parsed once into the "_ts" columns.
TIMESTAMP_COLUMNS: Tuple[str, ...] = (
    "schedule_arrival_time",
    "schedule_departure_time",
    "actual_arrival_time",
    "actual_departure_time",
)

# trips_performed columns carried onto each stop visit (where present).
TRIP_ATTRIBUTE_COLUMNS: Tuple[str, ...] = (
    "route_id",
    "direction_id",
    "route_type_agency",
    "trip_id_scheduled",
    "pattern_id",
    "block_id",
    "trip_end_stop_id",
    "schedule_trip_start",
    "schedule_trip_end",
    "actual_trip_start",
    "actual_trip_end",
)

# Columns used as group keys; blanks in them are reported as BLANK_LABEL.
GROUP_KEY_COLUMNS: Tuple[str, ...] = (
    "route_id",
    "direction_id",
    "route_type_agency",
    "stop_id",
    "trip_id",
    "from_stop_id",
    "to_stop_id",
)

# Values of the timepoint column that mean TRUE (compared without regard to case).
TRUE_TEXT: frozenset[str] = frozenset({"true", "t", "1", "yes", "y"})

OTP_SUBDIR: str = "otp"
TRIP_RUNTIME_SUBDIR: str = "trip_runtime"
SEGMENT_RUNTIME_SUBDIR: str = "segment_runtime"
SEGMENT_ROUTE_SUBDIR: str = "by_route"
CHARTS_SUBDIR: str = "charts"
STOP_FLAGS_FILENAME: str = "otp_stop_flags.csv"
STOP_ROUTE_DETAIL_FILENAME: str = "otp_stop_route_detail.csv"
TRIP_OBSERVATIONS_FILENAME: str = "trip_runtime_observations.csv"
TRIP_OUTLIERS_FILENAME: str = "trip_runtime_outliers.csv"
TRIP_FLAGS_FILENAME: str = "trip_runtime_flags.csv"
TRIP_DOW_FLAGS_FILENAME: str = "trip_runtime_dow_flags.csv"
SEGMENT_OBSERVATIONS_FILENAME: str = "segment_runtime_observations.csv"
DATE_CLASSIFICATION_FILENAME: str = "date_classification.csv"
TRIP_COVERAGE_FILENAME: str = "trip_coverage.csv"
RUN_LOG_FILENAME: str = "otp_runtime_from_tides_runlog.txt"
RUN_LOG_TITLE: str = "OTP AND RUNNING TIME FROM TIDES RUN LOG"

# The CONFIGURATION placeholders, kept here so an edited path is never mistaken
# for one (comparing against the settings themselves rejects every edited path).
PLACEHOLDER_PATHS: frozenset[str] = frozenset(
    {
        r"Path\To\Your\stop_visits.csv",
        r"Path\To\Your\trips_performed.csv",
        r"Path\To\Your\Output_Folder",
    }
)

# date_classification.csv columns, plus the two internal ones dropped before writing.
CLASSIFICATION_COLUMNS: Tuple[str, ...] = (
    "service_date",
    "weekday",
    "day_type",
    "day_type_source",
    "gtfs_feed",
    "gtfs_services",
    "gtfs_service_labels",
    "revenue_trips",
    "canceled_trips",
    "trips_run",
    "trips_checked_against_gtfs",
    "trips_not_scheduled",
    "trips_not_in_gtfs",
    "trips_scheduled",
    "trips_without_record",
    "share_without_record",
    "duplicate_visits_dropped",
    "conflicting_visits_dropped",
    "flagged",
    "flag_reason",
    "_stamp",
    "_day_types",
)

# Raised when no stop visit joins an in-service trip.
NO_MATCH_MESSAGE: str = (
    "No stop visits matched an in-service trip in trips_performed; check that the two "
    "files cover the same service dates and trip_id_performed values."
)

# Columns that identify one stop visit (the trip_stop_sequence is optional in some
# exports; without it only exact duplicates are removed).
VISIT_KEY: Tuple[str, ...] = ("service_date", "trip_id_performed", "trip_stop_sequence")

# A GTFS service is a regular service for a weekday when it runs on at least this
# share of that weekday's dates between its first and last service date.
REGULAR_WEEKDAY_SHARE: float = 0.5

# Chart styling: series hues and neutral ink from the repo's chart palette.
_SERIES_COLORS: Tuple[str, ...] = ("#2a78d6", "#e08b2c")
_INK_COLOR: str = "#52514e"
_GRID_COLOR: str = "#e3e2de"
# Status red for below-standard trips and the standard line; never the only
# encoding (the dashed line and the legend carry the same meaning).
_FLAG_COLOR: str = "#d03b3b"

# =============================================================================
# ERRORS & SETTINGS
# =============================================================================


class TimePeriodError(ValueError):
    """Raised when the time periods are malformed or do not cover the day exactly once."""


class SettingsError(ValueError):
    """Raised when a setting (day types, dates, grain flags, measure) is unusable."""


class InputDataError(ValueError):
    """Raised when the TIDES tables cannot support the requested exports."""


class PeriodSpec(NamedTuple):
    """A time period as written, before its times are checked."""

    name: str
    start: str
    end: str
    where: str  # where it was written, for error messages


class TimePeriod(NamedTuple):
    """A checked time period in minutes since midnight.

    ``start`` runs 0-1439 and ``end`` runs 1-1440. A period whose ``end`` is not
    after its ``start`` wraps past midnight.
    """

    name: str
    start: int
    end: int

    @property
    def minutes(self) -> int:
        """Length of the period in minutes."""
        if self.end > self.start:
            return self.end - self.start
        return self.end + MINUTES_PER_DAY - self.start

    @property
    def span(self) -> str:
        """The period's window as ``HH:MM-HH:MM``."""
        return f"{_clock(self.start)}-{_clock(self.end)}"


class RunSettings(NamedTuple):
    """Resolved settings for one run.

    ``None`` for ``measures``, ``levels``, ``time_cuts`` or ``day_types`` means "take
    them from the CONFIGURATION block" (``EXPORT_MEASURES``, ``EXPORT_LEVELS``,
    ``EXPORT_TIME_CUTS``, ``DAY_TYPES``).
    """

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    periods: Sequence[TimePeriod]
    periods_source: str = "TIME_PERIODS in the CONFIGURATION block"
    measures: Optional[Sequence[str]] = None
    levels: Optional[Sequence[str]] = None
    time_cuts: Optional[Sequence[str]] = None
    trip_time_basis: str = TRIP_TIME_BASIS
    stop_time_basis: str = STOP_TIME_BASIS
    cutover_hour: int = LATE_NIGHT_CUTOVER_HOUR
    day_types: Optional[Mapping[str, Sequence[str]]] = None
    gtfs_paths: Sequence[str] = GTFS_PATHS
    holidays: Sequence[dt.date] = ()
    holiday_day_type: str = HOLIDAY_DAY_TYPE
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    max_unscheduled_share: float = MAX_UNSCHEDULED_SHARE
    max_canceled_share: float = MAX_CANCELED_SHARE
    max_unrecorded_share: float = MAX_UNRECORDED_SHARE
    max_trip_unrecorded_share: float = MAX_TRIP_UNRECORDED_SHARE
    exclude_flagged_dates: bool = EXCLUDE_FLAGGED_DATES
    routes_to_include: Sequence[str] = ROUTES_TO_INCLUDE
    routes_to_exclude: Sequence[str] = ROUTES_TO_EXCLUDE
    early_min: float = EARLY_MIN
    late_min: float = LATE_MIN
    last_stop_treatment: str = LAST_STOP_TREATMENT
    last_stop_routes: str = LAST_STOP_ROUTES
    express_routes_file: Optional[str] = EXPRESS_ROUTES_FILE
    express_routes: Sequence[str] = EXPRESS_ROUTES
    express_service_types: Sequence[str] = EXPRESS_SERVICE_TYPES
    timepoints_only: bool = TIMEPOINTS_ONLY
    otp_standard: float = OTP_STANDARD
    dev_abs_warn_min: float = DEV_ABS_WARN_MIN
    thin_day_type_dates: int = THIN_DAY_TYPE_DATES
    export_stop_flags: bool = EXPORT_STOP_FLAGS
    min_routes_flagged: int = MIN_ROUTES_FLAGGED
    coverage_gap_flag_pct: float = COVERAGE_GAP_FLAG_PCT
    coverage_abs_flag_pct: float = COVERAGE_ABS_FLAG_PCT
    otp_gap_flag_pct: float = OTP_GAP_FLAG_PCT
    otp_abs_flag_pct: float = OTP_ABS_FLAG_PCT
    min_expected_trips: int = MIN_EXPECTED_TRIPS
    min_scored_visits: int = MIN_SCORED_VISITS
    trim_frac: float = TRIM_FRAC
    percentiles: Sequence[float] = RUNTIME_PERCENTILES
    segment_timepoints_only: bool = SEGMENT_TIMEPOINTS_ONLY
    high_cv_threshold: float = HIGH_CV_THRESHOLD
    min_obs_for_cv: int = MIN_OBS_FOR_CV
    gap_frac: float = GAP_FRAC
    dow_low_count_frac: float = DOW_LOW_COUNT_FRAC
    dow_runtime_pct: float = DOW_RUNTIME_PCT
    export_charts: bool = EXPORT_CHARTS
    chart_levels: Sequence[str] = CHART_LEVELS
    max_trips_per_chart: int = MAX_TRIPS_PER_CHART
    require_run_log: bool = REQUIRE_RUN_LOG


class RunResult(NamedTuple):
    """Tables built by one run (each also written to disk)."""

    otp: Dict[Tuple[str, str], pd.DataFrame]
    trip_runtime: Dict[Tuple[str, str], pd.DataFrame]
    segment_runtime: Dict[Tuple[str, str], pd.DataFrame]
    stop_flags: pd.DataFrame
    stop_route_detail: pd.DataFrame
    trip_observations: pd.DataFrame
    trip_outliers: pd.DataFrame
    trip_flags: pd.DataFrame
    trip_dow_flags: pd.DataFrame
    segment_observations: pd.DataFrame
    date_classification: pd.DataFrame
    trip_coverage: pd.DataFrame
    files: List[Path]


# =============================================================================
# CANONICAL HELPERS (copied verbatim from utils/ — see CONTRIBUTING.md)
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


def load_gtfs_data(
    gtfs_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_path: Absolute or relative path to the folder containing the
            GTFS feed, or to a ``.zip`` archive of it — the form GTFS
            producers and most open-data portals distribute feeds in. Zip
            members may sit at the archive root or nested one level inside
            a single wrapper folder; both layouts are handled.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.
        logger: Logger for progress messages. Defaults to this module's
            logger (``logging.getLogger(__name__)``) rather than the root
            logger, so callers keep control of handler configuration.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Path missing, one of *files* not present in the feed, or
            an OS-level failure while reading a file.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, a requested file matches more than one location inside
            the zip, a file is empty, or the CSV parser fails.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    if not os.path.exists(gtfs_path):
        raise OSError(f"The path '{gtfs_path}' does not exist.")

    if files is None:
        files = (
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        )

    is_zip = os.path.isfile(gtfs_path) and gtfs_path.lower().endswith(".zip")
    if not is_zip and not os.path.isdir(gtfs_path):
        raise ValueError(f"'{gtfs_path}' is neither a directory nor a .zip file.")

    archive: zipfile.ZipFile | None = None
    members_by_name: dict[str, list[str]] = {}
    if is_zip:
        try:
            archive = zipfile.ZipFile(gtfs_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"'{gtfs_path}' is not a valid zip archive.") from exc
        for name in archive.namelist():
            members_by_name.setdefault(os.path.basename(name), []).append(name)

    try:
        missing: list[str] = []
        ambiguous: list[str] = []
        resolved: dict[str, str] = {}
        for file_name in files:
            if archive is None:
                if not os.path.exists(os.path.join(gtfs_path, file_name)):
                    missing.append(file_name)
                continue
            candidates = members_by_name.get(file_name, [])
            if not candidates:
                missing.append(file_name)
            elif len(candidates) > 1:
                ambiguous.append(file_name)
            else:
                resolved[file_name] = candidates[0]

        if ambiguous:
            raise ValueError(
                f"Ambiguous GTFS files in '{gtfs_path}' (found in multiple "
                f"locations): {', '.join(ambiguous)}"
            )
        if missing:
            raise OSError(f"Missing GTFS files in '{gtfs_path}': {', '.join(missing)}")

        data: dict[str, pd.DataFrame] = {}
        for file_name in files:
            key = file_name.replace(".txt", "")
            try:
                if archive is None:
                    df = pd.read_csv(
                        os.path.join(gtfs_path, file_name), dtype=dtype, low_memory=False
                    )
                else:
                    with archive.open(resolved[file_name]) as handle:
                        df = pd.read_csv(handle, dtype=dtype, low_memory=False)
                data[key] = df
                log.info("Loaded %s (%d records).", file_name, len(df))

            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"File '{file_name}' in '{gtfs_path}' is empty.") from exc

            except pd.errors.ParserError as exc:
                raise ValueError(f"Parser error in '{file_name}' in '{gtfs_path}': {exc}") from exc

        return data
    finally:
        if archive is not None:
            archive.close()


@contextmanager
def open_gtfs_file(
    gtfs_path: str | os.PathLike[str], file_name: str
) -> Generator[IO[bytes], None, None]:
    """Open one GTFS text file from a feed folder or a ``.zip`` of the feed.

    The counterpart of :func:`load_gtfs_data` for scripts that read a file with
    their own options (``pandas.read_csv`` with ``usecols`` or
    ``keep_default_na=False``, or the :mod:`csv` module), so they accept a
    zipped feed without extracting it. Zip members are found the way
    :func:`load_gtfs_data` finds them: at the archive root or inside a single
    wrapper folder.

    Args:
        gtfs_path: Folder containing the GTFS feed, or a ``.zip`` archive of it.
        file_name: GTFS file to open, e.g. ``"stops.txt"``.

    Yields:
        A binary file handle. ``pandas.read_csv`` reads it directly; wrap it in
        :class:`io.TextIOWrapper` for the :mod:`csv` module.

    Raises:
        FileNotFoundError: *gtfs_path* does not exist, or the feed has no
            *file_name*.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, or *file_name* is found in more than one place in the zip.
    """
    path = os.fspath(gtfs_path)
    if os.path.isdir(path):
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        with open(file_path, "rb") as handle:
            yield handle
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    if not (os.path.isfile(path) and path.lower().endswith(".zip")):
        raise ValueError(f"'{path}' is neither a directory nor a .zip file.")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc
    with archive:
        matches = [name for name in archive.namelist() if os.path.basename(name) == file_name]
        if not matches:
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous GTFS file in '{path}' (found in multiple locations): {file_name}"
            )
        with archive.open(matches[0]) as handle:
            yield handle


def expand_service_active_dates(
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame] = None,
    max_days_per_service: int = 1830,
    today: Optional[dt.date] = None,
) -> dict[str, set[dt.date]]:
    """Expand each service_id into its real set of active calendar dates.

    Builds the base date set from each ``calendar.txt`` row (day-of-week
    pattern × ``start_date``–``end_date`` range), then applies
    ``calendar_dates.txt`` exceptions (``exception_type`` 1 adds a date,
    2 removes it). Handles calendar_dates-only feeds (*calendar_df* empty or
    ``None``), redundant additions, and fully negated base patterns — the
    returned sets reflect only the dates a service truly operates.

    Rows with unparseable or reversed dates are skipped with a warning.
    A date range longer than *max_days_per_service* (a common placeholder
    pattern, e.g. 2000–2099) is clamped to a window of that length centred
    on *today* and logged, so expansion stays fast and downstream per-year
    statistics stay meaningful.

    Args:
        calendar_df: Parsed ``calendar.txt``, or ``None`` if the feed has
            none. Expected columns: ``service_id``, the seven day-of-week
            flags, ``start_date``, ``end_date``.
        calendar_dates_df: Parsed ``calendar_dates.txt`` or ``None``.
            Expected columns: ``service_id``, ``date``, ``exception_type``.
        max_days_per_service: Longest date range expanded per service before
            clamping kicks in. The default (1830 ≈ 5 years) is far beyond
            any real service span but well short of placeholder ranges.
        today: Anchor date for clamping oversized ranges. Defaults to the
            current date; pass a fixed date for deterministic tests.

    Returns:
        Mapping of ``service_id`` (as ``str``) to the set of dates the
        service operates. Services whose dates never parse map to an empty
        set rather than being dropped, so callers can report them.

    Raises:
        ValueError: If *calendar_df* is provided but lacks ``service_id``,
            ``start_date``, or ``end_date`` columns.
    """
    day_cols = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    anchor = dt.date.today() if today is None else today
    active: dict[str, set[dt.date]] = {}

    if calendar_df is not None and not calendar_df.empty:
        required = {"service_id", "start_date", "end_date"}
        missing = required - set(calendar_df.columns)
        if missing:
            raise ValueError(f"calendar.txt is missing required column(s): {sorted(missing)}")
        for _, row in calendar_df.iterrows():
            sid = str(row["service_id"]).strip()
            try:
                start = dt.datetime.strptime(str(row["start_date"]).strip(), "%Y%m%d").date()
                end = dt.datetime.strptime(str(row["end_date"]).strip(), "%Y%m%d").date()
            except ValueError:
                logging.warning("Service %s: unparseable start/end date — skipping row.", sid)
                active.setdefault(sid, set())
                continue
            if end < start:
                logging.warning(
                    "Service %s: end_date %s precedes start_date %s — skipping row.",
                    sid,
                    end,
                    start,
                )
                active.setdefault(sid, set())
                continue
            if (end - start).days + 1 > max_days_per_service:
                half = max_days_per_service // 2
                clamped_start = max(start, anchor - dt.timedelta(days=half))
                clamped_end = min(end, anchor + dt.timedelta(days=half))
                logging.warning(
                    "Service %s: date range %s–%s looks like a placeholder; "
                    "clamping expansion to %s–%s.",
                    sid,
                    start,
                    end,
                    clamped_start,
                    clamped_end,
                )
                start, end = clamped_start, clamped_end
            pattern = [str(row.get(c, "0")).strip() == "1" for c in day_cols]
            dates = active.setdefault(sid, set())
            d = start
            while d <= end:
                if pattern[d.weekday()]:
                    dates.add(d)
                d += dt.timedelta(days=1)

    if calendar_dates_df is not None and not calendar_dates_df.empty:
        bad_rows = 0
        for _, row in calendar_dates_df.iterrows():
            sid = str(row["service_id"]).strip()
            try:
                d = dt.datetime.strptime(str(row["date"]).strip(), "%Y%m%d").date()
            except ValueError:
                bad_rows += 1
                continue
            etype = str(row.get("exception_type", "")).strip()
            dates = active.setdefault(sid, set())
            if etype == "1":
                dates.add(d)
            elif etype == "2":
                dates.discard(d)
            else:
                bad_rows += 1
        if bad_rows:
            logging.warning(
                "calendar_dates.txt: skipped %d row(s) with unparseable date/exception_type.",
                bad_rows,
            )

    return active


def classify_service_ids(
    active_dates: Mapping[str, set[dt.date]],
    holiday_max_days_per_year: float = 25.0,
    dow_share: float = 0.80,
) -> dict[str, set[str]]:
    """Classify each service_id by its real active-date pattern.

    A service operating at or below *holiday_max_days_per_year* is labelled
    ``Holiday`` — this catches holiday-only services regardless of what
    their day-of-week columns claim (scheduling-software exports often stamp
    a weekday pattern on a service that really runs five days a year).
    Otherwise, day-of-week shares of the active dates determine the labels.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        holiday_max_days_per_year: Services active at or below this annual
            rate are labelled ``Holiday``.
        dow_share: Minimum fraction of active dates on a day-of-week bucket
            (Mon–Fri, Saturday, Sunday) to earn that bucket's label.

    Returns:
        Mapping of ``service_id`` to a set of labels drawn from
        ``{"Weekday", "Saturday", "Sunday", "Holiday"}``. A service with no
        active dates maps to an empty set.
    """
    result: dict[str, set[str]] = {}
    for sid, dates in active_dates.items():
        if not dates:
            result[sid] = set()
            logging.info("Service %s: empty (0 active dates).", sid)
            continue
        span_days = (max(dates) - min(dates)).days + 1
        per_year = len(dates) / max(span_days / 365.25, 0.1)
        labels: set[str] = set()
        if per_year <= holiday_max_days_per_year:
            labels.add("Holiday")
        else:
            n = len(dates)
            wd = sum(1 for d in dates if d.weekday() < 5)
            sat = sum(1 for d in dates if d.weekday() == 5)
            sun = sum(1 for d in dates if d.weekday() == 6)
            if wd / n >= dow_share:
                labels.add("Weekday")
            if sat / n >= dow_share:
                labels.add("Saturday")
            if sun / n >= dow_share:
                labels.add("Sunday")
            if not labels:
                if wd > 0:
                    labels.add("Weekday")
                if sat > 0:
                    labels.add("Saturday")
                if sun > 0:
                    labels.add("Sunday")
        result[sid] = labels
        logging.info(
            "Service %s -> %s (%d dates, %.1f/yr).",
            sid,
            sorted(labels),
            len(dates),
            per_year,
        )
    return result


def load_id_set(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
    *,
    kind: str = "id",
) -> set[str]:
    """Union an inline list and an optional text file of ids into one set.

    Used to resolve override lists (express routes, express origin stops, …) that
    a caller may supply inline, in an external file, or both — without repeating
    the parsing for each one.

    Args:
        inline_ids: Id values supplied directly (e.g. a config list). ``None`` is
            treated as empty.
        txt_path: Path to a text file with one id per line. Blank lines are
            skipped and ``#`` starts a comment (whole-line or inline). ``None``
            skips the file. A path that is set but missing is logged as a warning
            and skipped — the inline ids are still returned.
        kind: Human-readable noun used only in log messages (e.g.
            ``"express route"``, ``"express origin stop"``).

    Returns:
        The unioned set of id strings (possibly empty). Every id is coerced to a
        trimmed ``str`` so it matches GTFS values, which are read as strings.
    """
    ids: set[str] = set()

    for raw in inline_ids or ():
        text = str(raw).strip()
        if text:
            ids.add(text)

    if txt_path:
        if not os.path.exists(txt_path):
            logging.warning(
                "%s file '%s' not found; using inline ids only.", kind.capitalize(), txt_path
            )
        else:
            with open(txt_path, encoding="utf-8") as handle:
                for line in handle:
                    text = line.split("#", 1)[0].strip()
                    if text:
                        ids.add(text)
            logging.info("Loaded %s ids from '%s'.", kind, txt_path)

    logging.info("Resolved %d %s id(s).", len(ids), kind)
    return ids


def load_express_route_ids(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
) -> set[str]:
    """Resolve the set of express-route ``route_id`` values (see ``load_id_set``).

    Thin wrapper kept for readable call sites and backwards compatibility.
    """
    return load_id_set(inline_ids, txt_path, kind="express route")


# =============================================================================
# TIME PERIODS
# =============================================================================

# Copied verbatim from utils/service_day_helpers.py (the canonical copy) through the
# SERVICE-DAY CLOCK section; keep them identical to it and to ridership_from_tides.py.

_CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})")

_SPEC_RE = re.compile(r"(?P<name>[^=]*?)\s*=\s*(?P<start>[^\s\-–]+)\s*[-–]\s*(?P<end>[^\s\-–]+)")


def _clock(minutes: int) -> str:
    """Format minutes since midnight as ``HH:MM`` (1440 becomes ``24:00``)."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _parse_clock(text: str, *, is_end: bool) -> int:
    """Return minutes since midnight for a 24-hour ``HH:MM`` string.

    Args:
        text: The time as written.
        is_end: Whether this is a period's end time (only an end may be
            ``24:00``).

    Returns:
        Minutes since midnight, 0-1440.

    Raises:
        ValueError: With a short reason if the time is malformed or out of range.
    """
    match = _CLOCK_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"{text.strip()!r} is not a 24-hour HH:MM time")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if minutes > 59 or hours > 24 or (hours == 24 and minutes > 0):
        raise ValueError(f"{text.strip()!r} is outside 00:00-24:00")
    if hours == 24 and not is_end:
        raise ValueError("a period cannot start at 24:00 (use 00:00)")
    return hours * 60 + minutes


def minute_mask(period: TimePeriod) -> np.ndarray:
    """Return a 1440-long boolean array that is True for the minutes in *period*."""
    mask = np.zeros(MINUTES_PER_DAY, dtype=bool)
    if period.end > period.start:
        mask[period.start : period.end] = True
    else:
        mask[period.start :] = True
        mask[: period.end] = True
    return mask


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return the ``(start, end)`` minute ranges where *mask* is True.

    The day is treated as a circle, so a run that crosses midnight comes back as
    one range whose end is earlier than its start.
    """
    if mask.all():
        return [(0, MINUTES_PER_DAY)]
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    runs = list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == MINUTES_PER_DAY:
        runs = [(runs[-1][0], runs[0][1]), *runs[1:-1]]
    return runs


def find_coverage_problems(periods: Sequence[TimePeriod]) -> List[str]:
    """Describe every gap and overlap in *periods* across the 24-hour clock.

    Args:
        periods: Time periods whose individual times are already valid.

    Returns:
        One message per uncovered range and per overlapping pair of periods;
        empty when the periods cover every minute exactly once.
    """
    masks = [minute_mask(p) for p in periods]
    covered = np.zeros(MINUTES_PER_DAY, dtype=int)
    for mask in masks:
        covered += mask
    problems = [f"no period covers {_clock(s)}-{_clock(e)}" for s, e in _runs(covered == 0)]
    for (first, first_mask), (second, second_mask) in itertools.combinations(
        zip(periods, masks), 2
    ):
        for s, e in _runs(first_mask & second_mask):
            problems.append(
                f"{first.name!r} ({first.span}) and {second.name!r} ({second.span}) "
                f"overlap at {_clock(s)}-{_clock(e)}"
            )
    return problems


def _problem_report(header: str, problems: Sequence[str]) -> str:
    """Join *problems* under *header* as an indented bullet list."""
    return "\n".join([header, *(f"  - {p}" for p in problems)])


def build_time_periods(specs: Sequence[PeriodSpec], source: str) -> List[TimePeriod]:
    """Check *specs* and return them as time periods that cover the day exactly once.

    Every problem found is reported together: blank or repeated names
    (case-insensitive), malformed or out-of-range times, zero-length periods,
    and then, once every period reads cleanly, each gap and overlap.

    Args:
        specs: The time periods as written.
        source: Where they came from, for the error message.

    Returns:
        The periods in the order given.

    Raises:
        TimePeriodError: If any check fails.
    """
    if not specs:
        raise TimePeriodError(f"No time periods were given ({source}).")
    problems: List[str] = []
    periods: List[TimePeriod] = []
    seen: Dict[str, str] = {}
    for spec in specs:
        name = spec.name.strip()
        if not name:
            problems.append(f"{spec.where}: the period name is blank")
            continue
        if name.casefold() in seen:
            problems.append(
                f"{spec.where}: the name {name!r} is already used ({seen[name.casefold()]})"
            )
            continue
        seen[name.casefold()] = spec.where
        try:
            start = _parse_clock(spec.start, is_end=False)
            end = _parse_clock(spec.end, is_end=True)
        except ValueError as exc:
            problems.append(f"{spec.where}: period {name!r}: {exc}")
            continue
        if start == end:
            problems.append(
                f"{spec.where}: period {name!r} starts and ends at {spec.start.strip()}; "
                "to cover the whole day use 00:00 - 24:00"
            )
            continue
        periods.append(TimePeriod(name, start, end or MINUTES_PER_DAY))
    if not problems:
        problems = find_coverage_problems(periods)
    if problems:
        raise TimePeriodError(
            _problem_report(f"The time periods from {source} are not usable:", problems)
            + "\nTime periods must use 24-hour HH:MM times and, together, cover every "
            "minute of the day exactly once. Fix them and run again."
        )
    return periods


def specs_from_mapping(mapping: Mapping[str, Sequence[str]]) -> List[PeriodSpec]:
    """Read the inline ``TIME_PERIODS`` mapping into period specs.

    Args:
        mapping: ``{name: (start, end)}`` as in the CONFIGURATION block.

    Returns:
        One spec per entry, in mapping order.

    Raises:
        TimePeriodError: If an entry is not a ``(start, end)`` pair.
    """
    specs: List[PeriodSpec] = []
    for name, window in mapping.items():
        where = f"TIME_PERIODS[{name!r}]"
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            raise TimePeriodError(
                f'{where} must be a (start, end) pair such as ("06:00", "09:00"), not {window!r}.'
            )
        specs.append(PeriodSpec(str(name), str(window[0]), str(window[1]), where))
    return specs


def parse_period_spec(text: str, where: str) -> PeriodSpec:
    """Parse one ``NAME = HH:MM - HH:MM`` line or ``NAME=HH:MM-HH:MM`` flag.

    Args:
        text: The period as written.
        where: Where it was written, for error messages.

    Returns:
        The unchecked period spec.

    Raises:
        TimePeriodError: If *text* is not in that form.
    """
    match = _SPEC_RE.fullmatch(text.strip())
    if match is None:
        raise TimePeriodError(f"{where}: expected NAME = HH:MM - HH:MM, got {text.strip()!r}")
    return PeriodSpec(match["name"], match["start"], match["end"], where)


def load_time_periods_file(path: Path) -> List[PeriodSpec]:
    """Read a time-periods text file (one ``NAME = HH:MM - HH:MM`` per line).

    Blank lines and anything after ``#`` are ignored.

    Args:
        path: The text file.

    Returns:
        One spec per period line, in file order.

    Raises:
        TimePeriodError: If the file cannot be read, a line is not in that
            form, or the file lists no periods.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise TimePeriodError(f"Could not read the time-periods file '{path}': {exc}") from exc
    specs: List[PeriodSpec] = []
    problems: List[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            specs.append(parse_period_spec(line, f"line {number}"))
        except TimePeriodError as exc:
            problems.append(str(exc))
    if problems:
        raise TimePeriodError(
            _problem_report(f"Could not read the time-periods file '{path}':", problems)
        )
    if not specs:
        raise TimePeriodError(f"The time-periods file '{path}' lists no periods.")
    return specs


def resolve_time_periods(
    cli_periods: Optional[Sequence[str]],
    periods_file: Optional[str],
    inline: Mapping[str, Sequence[str]],
) -> Tuple[List[TimePeriod], str]:
    """Pick the time periods to use and check them.

    Precedence: ``--time-period`` flags, then the time-periods file, then the
    inline ``TIME_PERIODS`` mapping.

    Args:
        cli_periods: ``NAME=HH:MM-HH:MM`` strings from the command line, if any.
        periods_file: Path to a time-periods text file, if any.
        inline: The ``TIME_PERIODS`` mapping.

    Returns:
        The checked periods and a description of where they came from.

    Raises:
        TimePeriodError: If the chosen periods fail any check.
    """
    if cli_periods:
        if periods_file:
            logging.info("Using --time-period flags; ignoring time-periods file %s.", periods_file)
        source = "the --time-period flags"
        specs = [parse_period_spec(v, f"--time-period {v!r}") for v in cli_periods]
    elif periods_file:
        path = Path(periods_file).expanduser()
        source = f"time-periods file '{path}'"
        specs = load_time_periods_file(path)
    else:
        source = "TIME_PERIODS in the CONFIGURATION block"
        specs = specs_from_mapping(inline)
    return build_time_periods(specs, source), source


def format_period(period: TimePeriod) -> str:
    """Format *period* as a time-periods file line (``NAME = HH:MM - HH:MM``)."""
    return f"{period.name} = {_clock(period.start)} - {_clock(period.end)}"


def period_lookup(periods: Sequence[TimePeriod]) -> np.ndarray:
    """Return a 1440-long array giving the period name for each minute of the day."""
    lookup = np.full(MINUTES_PER_DAY, None, dtype=object)
    for period in periods:
        lookup[minute_mask(period)] = period.name
    return lookup


# =============================================================================
# OTHER SETTINGS
# =============================================================================


def resolve_flags(flags: Mapping[str, bool], known: Sequence[str], setting: str) -> List[str]:
    """Return the names switched on in a ``{name: bool}`` flag mapping, in *known* order.

    Args:
        flags: The flag mapping (an entry left out counts as off).
        known: Every valid name, in output order.
        setting: The setting's name, for error messages.

    Returns:
        The names whose flag is True.

    Raises:
        SettingsError: If the mapping names an unknown entry or switches everything off.
    """
    unknown = [str(k) for k in flags if k not in known]
    if unknown:
        raise SettingsError(
            f"{setting} has unknown entries: {', '.join(unknown)}. Known: {', '.join(known)}."
        )
    chosen = [name for name in known if flags.get(name, False)]
    if not chosen:
        raise SettingsError(f"{setting} switches every entry off; nothing would be exported.")
    return chosen


def check_names(values: Sequence[str], known: Sequence[str], setting: str) -> List[str]:
    """Check a list of names against *known* and return it in *known* order.

    Args:
        values: The names given.
        known: Every valid name, in output order.
        setting: The setting's name, for error messages.

    Returns:
        The given names, de-duplicated and in *known* order.

    Raises:
        SettingsError: If a name is unknown or the list is empty.
    """
    unknown = [str(v) for v in values if v not in known]
    if unknown:
        raise SettingsError(
            f"{setting} has unknown entries: {', '.join(unknown)}. Known: {', '.join(known)}."
        )
    chosen = [name for name in known if name in set(values)]
    if not chosen:
        raise SettingsError(f"{setting} is empty; nothing would be exported.")
    return chosen


def build_day_types(
    mapping: Mapping[str, Sequence[str]], holiday_day_type: str = HOLIDAY_DAY_TYPE
) -> Dict[str, Tuple[int, ...]]:
    """Check ``DAY_TYPES`` and return each day type's weekday numbers (Monday = 0).

    Args:
        mapping: ``{name: weekdays}``; weekdays are full English names or their
            three-letter abbreviations, in any case. A single name may be given
            as a plain string.
        holiday_day_type: The holiday day type's name, which no entry may reuse.

    Returns:
        ``{name: weekday numbers}`` in the order given.

    Raises:
        SettingsError: Listing every problem found (blank or repeated names, an
            unknown weekday, an entry with no weekdays, or an empty mapping).
    """
    problems: List[str] = []
    out: Dict[str, Tuple[int, ...]] = {}
    lookup: Dict[str, int] = {}
    for number, day in enumerate(WEEKDAY_NAMES):
        lookup[day.casefold()] = number
        lookup[day[:3].casefold()] = number
    for raw_name, days in mapping.items():
        name = str(raw_name).strip()
        where = f"DAY_TYPES[{raw_name!r}]"
        if not name:
            problems.append(f"{where}: the day type name is blank")
            continue
        taken = {n.casefold() for n in out} | {holiday_day_type.strip().casefold()}
        if name.casefold() in taken:
            problems.append(f"{where}: the name {name!r} is already used")
            continue
        entries = [days] if isinstance(days, str) else list(days)
        numbers: List[int] = []
        for entry in entries:
            key = str(entry).strip().casefold()
            if key not in lookup:
                problems.append(f"{where}: {entry!r} is not a weekday name")
                continue
            numbers.append(lookup[key])
        if not entries:
            problems.append(f"{where} lists no weekdays")
        out[name] = tuple(sorted(set(numbers)))
    if not mapping:
        problems.append("DAY_TYPES is empty")
    if problems:
        raise SettingsError(_problem_report("DAY_TYPES is not usable:", problems))
    return out


def parse_iso_dates(values: Iterable[str], setting: str) -> List[dt.date]:
    """Parse ``YYYY-MM-DD`` strings, reporting every unreadable one together.

    Args:
        values: The dates as written.
        setting: The setting's name, for error messages.

    Returns:
        The parsed dates, sorted and de-duplicated.

    Raises:
        SettingsError: If any value is not a ``YYYY-MM-DD`` date.
    """
    dates: set[dt.date] = set()
    problems: List[str] = []
    for value in values:
        text = str(value).strip()
        try:
            dates.add(dt.date.fromisoformat(text))
        except ValueError:
            problems.append(f"{text!r} is not a YYYY-MM-DD date")
    if problems:
        raise SettingsError(_problem_report(f"{setting} is not usable:", problems))
    return sorted(dates)


def parse_optional_date(value: Optional[str], setting: str) -> Optional[dt.date]:
    """Parse one optional ``YYYY-MM-DD`` setting (``None`` or blank stays ``None``).

    Args:
        value: The date as written, or ``None``.
        setting: The setting's name, for error messages.

    Returns:
        The date, or ``None``.

    Raises:
        SettingsError: If the value is not a ``YYYY-MM-DD`` date.
    """
    if value is None or not str(value).strip():
        return None
    return parse_iso_dates([str(value)], setting)[0]


# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_tides_csv(path: Path, table: str, keep_text: bool = False) -> pd.DataFrame:
    """Read a TIDES CSV with every column as text.

    Args:
        path: The CSV file.
        table: The table's name, for error messages.
        keep_text: Treat only empty fields as blank, keeping placeholder text
            such as ``n/a`` or ``NULL`` (which pandas would otherwise blank), so
            a count column holding it reads as a missing count, not a blank door.

    Returns:
        The table, all columns as strings.

    Raises:
        InputDataError: If the file is empty or cannot be parsed as CSV.
    """
    try:
        if keep_text:
            return pd.read_csv(
                path, dtype=str, low_memory=False, keep_default_na=False, na_values=[""]
            )
        return pd.read_csv(path, dtype=str, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise InputDataError(f"Could not read {table} from '{path}': {exc}") from exc


def _require_columns(df: pd.DataFrame, columns: Sequence[str], table: str, why: str) -> None:
    """Raise :class:`InputDataError` naming any of *columns* missing from *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InputDataError(f"{table} has no {', '.join(missing)} column(s), which {why}.")


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse a service_date column to midnight timestamps (NaT where unreadable)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce").dt.normalize()


def _casefolded(values: pd.Series, blank: str) -> pd.Series:
    """Return *values* stripped and casefolded, with blanks replaced by *blank*."""
    text = values.fillna("").astype(str).str.strip()
    return text.where(text != "", blank).str.casefold()


def join_trips(
    stop_visits: pd.DataFrame, trips_performed: pd.DataFrame, require_rows: bool = True
) -> pd.DataFrame:
    """Attach trip attributes to each stop visit and keep revenue service only.

    The join keys are ``service_date`` and ``trip_id_performed`` (TIDES makes
    the trip ID unique only within a service date); dates are compared parsed,
    since the two exports may format them differently. Trips marked Canceled or
    with a ``trip_type`` other than In service are dropped (compared without
    regard to case). Visits with no same-day trip are counted in a warning.
    Skipped visits are kept here; :func:`drop_skipped` removes them for ridership.

    Args:
        stop_visits: The TIDES ``stop_visits`` table.
        trips_performed: The TIDES ``trips_performed`` table.
        require_rows: Raise when no visit survives (False returns an empty table,
            so a caller can still report on the trips).

    Returns:
        Stop visits with the :data:`TRIP_ATTRIBUTE_COLUMNS` present in
        ``trips_performed`` joined on, and ``service_date`` parsed.

    Raises:
        InputDataError: If a join key is missing, or no visit survives and
            *require_rows* is True.
    """
    keys = ["service_date", "trip_id_performed"]
    _require_columns(stop_visits, keys, "stop_visits", "the join to trips_performed needs")
    _require_columns(trips_performed, keys, "trips_performed", "the join to stop_visits needs")

    visits = stop_visits.assign(service_date=_parse_dates(stop_visits["service_date"]))
    trips = trips_performed.assign(service_date=_parse_dates(trips_performed["service_date"]))
    bad_dates = visits["service_date"].isna()
    if bad_dates.any():
        logging.warning(
            "%d stop visit(s) have an unreadable service_date; they are left out.",
            int(bad_dates.sum()),
        )
        visits = visits.loc[~bad_dates]
    trips = trips.loc[trips["service_date"].notna()]
    duplicated = trips.duplicated(keys)
    if duplicated.any():
        logging.warning(
            "trips_performed repeats %d service_date + trip_id_performed pair(s); "
            "the first row of each is used.",
            int(duplicated.sum()),
        )
        trips = trips.loc[~duplicated]

    trip_index = pd.MultiIndex.from_frame(trips[keys])
    unmatched = ~pd.MultiIndex.from_frame(visits[keys]).isin(trip_index)
    if unmatched.any():
        logging.warning(
            "%d stop visit(s) have no trips_performed row with the same service_date and "
            "trip_id_performed; they are left out.",
            int(unmatched.sum()),
        )

    revenue = pd.Series(True, index=trips.index)
    if "schedule_relationship" in trips.columns:
        status = _casefolded(trips["schedule_relationship"], "scheduled")
        revenue &= ~status.isin(["canceled", "cancelled"])
    if "trip_type" in trips.columns:
        revenue &= _casefolded(trips["trip_type"], "in service") == "in service"
    kept_index = pd.MultiIndex.from_frame(trips.loc[revenue, keys])
    on_dropped = ~unmatched & ~pd.MultiIndex.from_frame(visits[keys]).isin(kept_index)
    if on_dropped.any():
        logging.info(
            "Left out %d stop visit(s) on canceled or non-revenue trips.", int(on_dropped.sum())
        )

    attr_cols = [c for c in TRIP_ATTRIBUTE_COLUMNS if c in trips.columns]
    joined = visits.drop(columns=attr_cols, errors="ignore").merge(
        trips.loc[revenue, [*keys, *attr_cols]], on=keys, how="inner"
    )
    if joined.empty and require_rows:
        raise InputDataError(NO_MATCH_MESSAGE)
    return joined.reset_index(drop=True)


def dedupe_stop_visits(stop_visits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact duplicate stop visits and the later copies of conflicting ones.

    Visits conflict when they share ``service_date`` (compared parsed),
    ``trip_id_performed`` and ``trip_stop_sequence`` but differ elsewhere; the
    first is kept. Only visits whose three key values are all filled are
    compared: a blank ``trip_stop_sequence`` says nothing about which stop a
    visit is, so such visits (and every visit, when the column is absent) are
    only checked for exact duplicates.

    Args:
        stop_visits: The TIDES ``stop_visits`` table as read.

    Returns:
        The de-duplicated table, and ``duplicate_visits_dropped`` /
        ``conflicting_visits_dropped`` per service date (midnight timestamps).
    """
    exact = stop_visits.duplicated(keep="first")
    out = stop_visits.loc[~exact]
    conflict = pd.Series(False, index=out.index)
    if all(col in out.columns for col in VISIT_KEY):
        keyed = out[list(VISIT_KEY)].assign(service_date=_parse_dates(out["service_date"]))
        text = keyed.drop(columns="service_date").astype("string")
        filled = keyed["service_date"].notna()
        for col in text.columns:
            filled &= (text[col].str.strip() != "").fillna(False).astype(bool)
        conflict.loc[filled] = keyed.loc[filled].duplicated(keep="first")
        if (~filled).any():
            logging.info(
                "%d stop visit(s) have a blank %s and are only checked for exact duplicates.",
                int((~filled).sum()),
                " / ".join(VISIT_KEY),
            )
    if exact.any():
        logging.warning("Removed %d exact duplicate stop visit(s).", int(exact.sum()))
    if conflict.any():
        logging.warning(
            "%d stop visit(s) repeat a service_date + trip_id_performed + trip_stop_sequence "
            "with different values; the first of each is kept and the dates are flagged.",
            int(conflict.sum()),
        )
    dates = _parse_dates(stop_visits["service_date"])
    stats = pd.DataFrame(
        {
            "duplicate_visits_dropped": exact.groupby(dates).sum(),
            "conflicting_visits_dropped": conflict.reindex(stop_visits.index, fill_value=False)
            .groupby(dates)
            .sum(),
        }
    ).astype(int)
    return out.loc[~conflict], stats


def add_trip_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``trip_id``: the scheduled (GTFS) trip, else the performed trip ID.

    Args:
        df: Joined stop visits.

    Returns:
        A copy of *df* with a text ``trip_id`` column.
    """
    out = df.copy()
    performed = out["trip_id_performed"].fillna("").astype(str).str.strip()
    if "trip_id_scheduled" in out.columns:
        scheduled = out["trip_id_scheduled"].fillna("").astype(str).str.strip()
        out["trip_id"] = scheduled.where(scheduled != "", performed)
    else:
        out["trip_id"] = performed
    return out


def filter_dates(
    df: pd.DataFrame,
    start: Optional[dt.date],
    end: Optional[dt.date],
    require_rows: bool = True,
) -> pd.DataFrame:
    """Keep service dates from *start* to *end* (inclusive; ``None`` = no limit).

    Args:
        df: Rows with a parsed ``service_date``.
        start: First date kept, or ``None``.
        end: Last date kept, or ``None``.
        require_rows: Raise when no row is in range (False returns an empty table).

    Returns:
        The rows in range.

    Raises:
        InputDataError: If no rows are in range and *require_rows* is True.
    """
    if start is None and end is None:
        return df
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df["service_date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= df["service_date"] <= pd.Timestamp(end)
    if (~keep).any():
        logging.info("Left out %d stop visit(s) outside the date range.", int((~keep).sum()))
    out = df.loc[keep]
    if out.empty and require_rows:
        raise InputDataError(out_of_range_message(start, end))
    return out


def out_of_range_message(start: Optional[dt.date], end: Optional[dt.date]) -> str:
    """Explain that no stop visit falls in the requested date range."""
    return (
        f"No stop visits fall between {start or 'the first date'} and "
        f"{end or 'the last date'}; check START_DATE / END_DATE."
    )


# =============================================================================
# DAY TYPES
# =============================================================================


class GtfsFeed(NamedTuple):
    """The parts of one GTFS feed the day-type logic needs."""

    name: str
    services_by_date: Dict[dt.date, frozenset[str]]  # date -> service_ids scheduled
    service_labels: Dict[str, frozenset[str]]  # service_id -> Weekday/Saturday/Sunday/Holiday
    trip_services: Dict[str, str]  # trip_id -> service_id
    regular_weekdays: Dict[str, frozenset[int]]  # service_id -> weekdays it regularly runs
    trip_routes: Dict[str, str]  # trip_id -> route_id
    trip_directions: Dict[str, str]  # trip_id -> direction_id ("" when absent)
    trip_starts: Dict[str, float]  # trip_id -> first scheduled departure, minutes (GTFS clock)


_GTFS_TIME_RE = r"^\s*(\d{1,3}):([0-5]\d)(?::[0-5]\d)?\s*$"


def gtfs_trip_starts(path: str) -> Dict[str, float]:
    """Return each trip's first scheduled departure from ``stop_times.txt``.

    Only the trip ID and time columns are read. Times keep the GTFS clock, so a
    trip leaving at 25:10 is 1510 minutes.

    Args:
        path: A GTFS folder or ``.zip``.

    Returns:
        ``{trip_id: minutes}``; empty when the feed has no usable ``stop_times.txt``.
    """
    try:
        with open_gtfs_file(path, "stop_times.txt") as handle:
            header = pd.read_csv(handle, nrows=0).columns
        wanted = [c for c in ("trip_id", "departure_time", "arrival_time") if c in header]
        with open_gtfs_file(path, "stop_times.txt") as handle:
            times = pd.read_csv(handle, usecols=wanted, dtype=str)
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        logging.info(
            "GTFS feed %s: no usable stop_times.txt (%s); trip start times are blank.", path, exc
        )
        return {}
    if "trip_id" not in times.columns:
        return {}
    minutes = pd.Series(np.nan, index=times.index, dtype="float64")
    for col in ("departure_time", "arrival_time"):
        if col not in times.columns:
            continue
        parts = times[col].astype("string").str.extract(_GTFS_TIME_RE)
        clock = pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(
            parts[1], errors="coerce"
        )
        minutes = minutes.fillna(clock.astype("float64"))
    first = minutes.groupby(times["trip_id"].astype(str).str.strip()).min()
    return {str(trip): float(value) for trip, value in first.items() if pd.notna(value)}


def regular_weekdays(
    dates: Iterable[dt.date],
    is_holiday_only: bool = False,
    min_share: float = REGULAR_WEEKDAY_SHARE,
) -> frozenset[int]:
    """Return the weekdays (Monday = 0) a service regularly runs on.

    A weekday is regular when the service runs on at least *min_share* of that
    weekday's dates between its first and last service date, so a Monday-to-
    Saturday service is regular on Saturdays too, while Sunday service added on
    one holiday Wednesday is not regular on Wednesdays. A holiday-only service
    (labeled Holiday by ``classify_service_ids``) is never regular.

    Args:
        dates: The service's active dates.
        is_holiday_only: Whether the service is holiday-only.
        min_share: Share of a weekday's dates the service must run on.

    Returns:
        The regular weekdays.
    """
    days = sorted(set(dates))
    if is_holiday_only or not days:
        return frozenset()
    possible = [0] * 7
    span = (days[-1] - days[0]).days + 1
    for offset in range(span):
        possible[(days[0] + dt.timedelta(days=offset)).weekday()] += 1
    ran = [0] * 7
    for day in days:
        ran[day.weekday()] += 1
    return frozenset(w for w in range(7) if possible[w] and ran[w] / possible[w] >= min_share)


def load_gtfs_feed(path: str) -> GtfsFeed:
    """Read one feed's calendar and trips, and label each service by its active dates.

    Services are labeled Weekday, Saturday, Sunday or Holiday from the dates they
    actually run (``classify_service_ids``), not from their names.

    Args:
        path: A GTFS folder or ``.zip``.

    Returns:
        The feed's scheduled services per date, service labels and trip services.

    Raises:
        InputDataError: If the feed cannot be read, or has neither
            ``calendar.txt`` nor ``calendar_dates.txt``.
    """
    try:
        trips = load_gtfs_data(path, files=["trips.txt"])["trips"]
    except (OSError, ValueError) as exc:
        raise InputDataError(f"Could not read trips.txt from GTFS feed '{path}': {exc}") from exc
    tables: Dict[str, Optional[pd.DataFrame]] = {}
    for name in ("calendar", "calendar_dates"):
        try:
            tables[name] = load_gtfs_data(path, files=[f"{name}.txt"])[name]
        except OSError:
            tables[name] = None  # either calendar file may be absent
        except ValueError as exc:
            if "is empty" not in str(exc):
                raise InputDataError(f"GTFS feed '{path}': {exc}") from exc
            tables[name] = None
    if tables["calendar"] is None and tables["calendar_dates"] is None:
        raise InputDataError(
            f"GTFS feed '{path}' has neither calendar.txt nor calendar_dates.txt, so its "
            "service dates are unknown."
        )
    logging.disable(logging.INFO)  # the calendar helpers log one line per service
    try:
        active = expand_service_active_dates(tables["calendar"], tables["calendar_dates"])
        labels = classify_service_ids(active)
    except ValueError as exc:
        raise InputDataError(f"GTFS feed '{path}': {exc}") from exc
    finally:
        logging.disable(logging.NOTSET)
    by_date: Dict[dt.date, set[str]] = {}
    for service_id, dates in active.items():
        for day in dates:
            by_date.setdefault(day, set()).add(service_id)
    trip_ids = trips["trip_id"].fillna("").astype(str).str.strip()

    def trip_column(column: str) -> Dict[str, str]:
        """Map trip_id to one trips.txt column ("" when the column is absent)."""
        if column not in trips.columns:
            return dict.fromkeys(trip_ids, "")
        return dict(zip(trip_ids, trips[column].fillna("").astype(str).str.strip()))

    trip_services = trip_column("service_id")
    logging.info(
        "GTFS feed %s: %d service(s), %d trip(s), service on %d date(s).",
        Path(path).name,
        len(active),
        len(trip_services),
        len(by_date),
    )
    return GtfsFeed(
        Path(path).name,
        {day: frozenset(sids) for day, sids in by_date.items()},
        {sid: frozenset(labs) for sid, labs in labels.items()},
        trip_services,
        {
            sid: regular_weekdays(dates, "Holiday" in labels.get(sid, ()))
            for sid, dates in active.items()
        },
        trip_column("route_id"),
        trip_column("direction_id"),
        gtfs_trip_starts(path),
    )


def trip_date_stats(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Count each date's revenue trips and canceled trips, and list the scheduled trips run.

    Args:
        trips_performed: The TIDES ``trips_performed`` table (all rows).

    Returns:
        One row per service date (midnight timestamp index) with
        ``revenue_trips``, ``canceled_trips``, ``trips_run`` and ``ran_trip_ids``
        (the non-blank ``trip_id_scheduled`` of the revenue trips that ran).
    """
    trips = trips_performed.assign(service_date=_parse_dates(trips_performed["service_date"]))
    trips = trips.loc[trips["service_date"].notna()].drop_duplicates(
        ["service_date", "trip_id_performed"]
    )
    revenue = pd.Series(True, index=trips.index)
    if "trip_type" in trips.columns:
        revenue = _casefolded(trips["trip_type"], "in service") == "in service"
    canceled = pd.Series(False, index=trips.index)
    if "schedule_relationship" in trips.columns:
        status = _casefolded(trips["schedule_relationship"], "scheduled")
        canceled = status.isin(["canceled", "cancelled"])
    if "trip_id_scheduled" in trips.columns:
        scheduled = trips["trip_id_scheduled"].fillna("").astype(str).str.strip()
    else:
        scheduled = pd.Series("", index=trips.index)
    work = pd.DataFrame(
        {
            "service_date": trips["service_date"],
            "canceled": canceled,
            "scheduled": scheduled,
        }
    ).loc[revenue]
    ran = work.loc[~work["canceled"] & (work["scheduled"] != "")]
    stats = work.groupby("service_date").agg(
        revenue_trips=("canceled", "size"), canceled_trips=("canceled", "sum")
    )
    stats["canceled_trips"] = stats["canceled_trips"].astype(int)
    stats["trips_run"] = stats["revenue_trips"] - stats["canceled_trips"]
    ids = ran.groupby("service_date")["scheduled"].agg(list)
    stats["ran_trip_ids"] = [ids.get(stamp, []) for stamp in stats.index]
    return stats


def governing_feed(feeds: Sequence[GtfsFeed], day: dt.date) -> Optional[GtfsFeed]:
    """Return the last feed that schedules any service on *day* (later feeds win)."""
    chosen: Optional[GtfsFeed] = None
    for feed in feeds:
        if feed.services_by_date.get(day):
            chosen = feed
    return chosen


def trip_records_by_date(
    trips_performed: pd.DataFrame,
) -> Dict[pd.Timestamp, Tuple[frozenset[str], frozenset[str]]]:
    """Return, per service date, the scheduled trip IDs with any row and those canceled.

    Args:
        trips_performed: The TIDES ``trips_performed`` table (all rows).

    Returns:
        ``{date: (recorded trip_id_scheduled values, canceled ones)}``.
    """
    if "trip_id_scheduled" not in trips_performed.columns:
        return {}
    work = pd.DataFrame(
        {
            "service_date": _parse_dates(trips_performed["service_date"]),
            "trip": trips_performed["trip_id_scheduled"].fillna("").astype(str).str.strip(),
            "canceled": (
                _casefolded(trips_performed["schedule_relationship"], "scheduled").isin(
                    ["canceled", "cancelled"]
                )
                if "schedule_relationship" in trips_performed.columns
                else False
            ),
        }
    )
    work = work.loc[work["service_date"].notna() & (work["trip"] != "")]
    records: Dict[pd.Timestamp, Tuple[frozenset[str], frozenset[str]]] = {}
    for stamp, sub in work.groupby("service_date"):
        records[pd.Timestamp(str(stamp))] = (
            frozenset(sub["trip"]),
            frozenset(sub.loc[sub["canceled"], "trip"]),
        )
    return records


def schedule_coverage(
    dates: Iterable[pd.Timestamp],
    feeds: Sequence[GtfsFeed],
    records: Mapping[pd.Timestamp, Tuple[frozenset[str], frozenset[str]]],
) -> pd.DataFrame:
    """List every GTFS trip scheduled on each date and whether TIDES recorded it.

    Each date uses its governing feed (:func:`governing_feed`). A trip counts as
    recorded when ``trips_performed`` has a row with that ``trip_id_scheduled``
    on that date, whatever its status; a Canceled row is recorded (its outcome is
    known) and also counted as canceled.

    Args:
        dates: Service dates to check (midnight timestamps).
        feeds: Loaded GTFS feeds, oldest first.
        records: Output of :func:`trip_records_by_date`.

    Returns:
        One row per date and scheduled trip: ``service_date``, ``trip_id``,
        ``route_id``, ``direction_id``, ``start_minutes`` (GTFS clock, NaN when
        unknown), ``recorded`` and ``canceled``.
    """
    columns = [
        "service_date",
        "trip_id",
        "route_id",
        "direction_id",
        "start_minutes",
        "recorded",
        "canceled",
    ]
    # Keyed by object identity: feeds in a series often share a file name.
    trips_by_service: Dict[int, Dict[str, List[str]]] = {}
    for feed in feeds:
        by_service: Dict[str, List[str]] = {}
        for trip_id, service_id in feed.trip_services.items():
            by_service.setdefault(service_id, []).append(trip_id)
        trips_by_service[id(feed)] = by_service
    frames: List[pd.DataFrame] = []
    for stamp in sorted({pd.Timestamp(d) for d in dates}):
        feed = governing_feed(feeds, stamp.date())
        if feed is None:
            continue
        trip_ids = [
            trip_id
            for service_id in sorted(feed.services_by_date[stamp.date()])
            for trip_id in trips_by_service[id(feed)].get(service_id, [])
        ]
        if not trip_ids:
            continue
        recorded, canceled = records.get(stamp, (frozenset(), frozenset()))
        frames.append(
            pd.DataFrame(
                {
                    "service_date": stamp,
                    "trip_id": trip_ids,
                    "route_id": [feed.trip_routes.get(t, "") for t in trip_ids],
                    "direction_id": [feed.trip_directions.get(t, "") for t in trip_ids],
                    "start_minutes": [feed.trip_starts.get(t, np.nan) for t in trip_ids],
                    "recorded": [t in recorded for t in trip_ids],
                    "canceled": [t in canceled for t in trip_ids],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def coverage_by_date(schedule: pd.DataFrame) -> pd.DataFrame:
    """Count each date's scheduled trips and those with no TIDES record.

    Args:
        schedule: Output of :func:`schedule_coverage`.

    Returns:
        ``trips_scheduled`` and ``trips_without_record`` per service date.
    """
    if schedule.empty:
        return pd.DataFrame(columns=["trips_scheduled", "trips_without_record"])
    grouped = schedule.groupby("service_date")
    return pd.DataFrame(
        {
            "trips_scheduled": grouped.size(),
            "trips_without_record": grouped["recorded"].agg(lambda r: int((~r).sum())),
        }
    )


def build_trip_coverage(
    schedule: pd.DataFrame,
    classification: pd.DataFrame,
    periods: Sequence[TimePeriod],
    day_order: Sequence[str],
    max_trip_unrecorded_share: float = MAX_TRIP_UNRECORDED_SHARE,
) -> pd.DataFrame:
    """Summarize, per scheduled trip and day type, how often TIDES recorded it.

    Args:
        schedule: Output of :func:`schedule_coverage`.
        classification: Output of :func:`classify_service_dates` (its
            ``_stamp`` and ``_day_types`` give each date's day types).
        periods: The checked time periods (to place each trip by its start).
        day_order: Output order of the day types.
        max_trip_unrecorded_share: Trips with no record on more than this share
            of their scheduled dates are flagged.

    Returns:
        One row per day type and trip with ``days_scheduled``,
        ``days_recorded``, ``days_canceled``, ``days_without_record``,
        ``share_without_record`` and ``flagged``.
    """
    columns = [
        "day_type",
        "route_id",
        "direction_id",
        "trip_id",
        "start_time",
        "time_period",
        "days_scheduled",
        "days_recorded",
        "days_canceled",
        "days_without_record",
        "share_without_record",
        "flagged",
    ]
    dates = classification[["_stamp", "_day_types"]].explode("_day_types").dropna()
    dates = dates.rename(columns={"_stamp": "service_date", "_day_types": "day_type"})
    work = schedule.merge(dates, on="service_date", how="inner")
    if work.empty:
        return pd.DataFrame(columns=columns)
    keys = ["day_type", "route_id", "direction_id", "trip_id"]
    out = work.groupby(keys, as_index=False).agg(
        start_minutes=("start_minutes", "min"),
        days_scheduled=("service_date", "nunique"),
        days_recorded=("recorded", "sum"),
        days_canceled=("canceled", "sum"),
    )
    out["days_recorded"] = out["days_recorded"].astype(int)
    out["days_canceled"] = out["days_canceled"].astype(int)
    out["days_without_record"] = out["days_scheduled"] - out["days_recorded"]
    out["share_without_record"] = (out["days_without_record"] / out["days_scheduled"]).round(3)
    out["flagged"] = out["days_without_record"] / out["days_scheduled"] > max_trip_unrecorded_share
    out["start_time"] = out["start_minutes"].map(minutes_to_hhmm)
    lookup = period_lookup(periods)
    known = out["start_minutes"].notna()
    labels = pd.Series("", index=out.index, dtype="object")
    labels.loc[known] = lookup[(out.loc[known, "start_minutes"] % MINUTES_PER_DAY).astype(int)]
    out["time_period"] = labels
    out = _sort_table(
        out, ["day_type", "route_id", "direction_id", "start_minutes", "trip_id"], day_order
    )
    return out[columns]


def classify_service_dates(
    dates: Iterable[pd.Timestamp],
    day_types: Mapping[str, Sequence[int]],
    holidays: Iterable[dt.date] = (),
    holiday_day_type: str = HOLIDAY_DAY_TYPE,
    feeds: Sequence[GtfsFeed] = (),
    trip_stats: Optional[pd.DataFrame] = None,
    max_unscheduled_share: float = MAX_UNSCHEDULED_SHARE,
    max_canceled_share: float = MAX_CANCELED_SHARE,
    visit_stats: Optional[pd.DataFrame] = None,
    coverage_stats: Optional[pd.DataFrame] = None,
    max_unrecorded_share: float = MAX_UNRECORDED_SHARE,
) -> pd.DataFrame:
    """Give each service date its day type(s), the reason, and any data-quality flag.

    A date is a holiday when it is in *holidays*, or when GTFS schedules service
    that day but none of it regularly runs on that weekday (a Wednesday with only
    Sunday or holiday-only service; see :func:`regular_weekdays`). Any other date
    gets every ``DAY_TYPES`` entry listing its weekday. With several feeds, the
    last one scheduling service on a date governs it.

    A date is flagged when GTFS schedules no service that day, when more than
    *max_unscheduled_share* of the scheduled trip IDs that ran are not scheduled
    that day in the governing feed, when more than *max_canceled_share* of its
    revenue trips were canceled, when more than *max_unrecorded_share* of the GTFS
    trips scheduled that day have no ``trips_performed`` row at all, or when it has
    conflicting duplicate stop visits.

    Args:
        dates: Service dates (midnight timestamps).
        day_types: Output of :func:`build_day_types`.
        holidays: Holiday dates given by hand.
        holiday_day_type: Name of the holiday day type.
        feeds: Loaded GTFS feeds, oldest first.
        trip_stats: Output of :func:`trip_date_stats`, or ``None``.
        max_unscheduled_share: Flag threshold for trips run but not scheduled.
        max_canceled_share: Flag threshold for canceled revenue trips.
        visit_stats: Per-date stop-visit counts (``duplicate_visits_dropped``,
            ``conflicting_visits_dropped``), or ``None``.
        coverage_stats: Output of :func:`coverage_by_date`, or ``None``.
        max_unrecorded_share: Flag threshold for scheduled trips with no record.

    Returns:
        One row per date, in date order, with the published columns (see
        ``date_classification.csv``) plus ``_stamp`` and ``_day_types`` for
        building the date table.
    """
    manual = set(holidays)
    stats: Dict[Any, Dict[Any, Any]] = {} if trip_stats is None else trip_stats.to_dict("index")
    visit_counts: Dict[Any, Dict[Any, Any]] = (
        {} if visit_stats is None else visit_stats.to_dict("index")
    )
    coverage: Dict[Any, Dict[Any, Any]] = (
        {} if coverage_stats is None else coverage_stats.to_dict("index")
    )
    rows: List[Dict[str, Any]] = []
    for stamp in sorted({pd.Timestamp(d) for d in dates}):
        day = stamp.date()
        weekday = WEEKDAY_NAMES[day.weekday()]
        governing = governing_feed(feeds, day)
        services = sorted(governing.services_by_date[day]) if governing else []
        labels = sorted(
            set().union(*(governing.service_labels.get(s, frozenset()) for s in services))
            if governing
            else set()
        )
        regular = governing is not None and any(
            day.weekday() in governing.regular_weekdays.get(s, frozenset()) for s in services
        )
        if day in manual:
            types, source = [holiday_day_type], "HOLIDAYS"
        elif services and not regular:
            types, source = [holiday_day_type], f"GTFS: no service that regularly runs {weekday}s"
        else:
            types = [name for name, numbers in day_types.items() if day.weekday() in numbers]
            source = "DAY_TYPES" if types else "DAY_TYPES (weekday in no day type)"

        reasons: List[str] = []
        if feeds and not services:
            reasons.append("GTFS schedules no service on this date")
        revenue = canceled = run = checked = not_scheduled = not_in_gtfs = 0
        stat = stats.get(stamp)
        if stat is not None:
            revenue, canceled = int(stat["revenue_trips"]), int(stat["canceled_trips"])
            run = int(stat["trips_run"])
            if revenue and canceled / revenue > max_canceled_share:
                reasons.append(f"{canceled / revenue:.0%} of revenue trips canceled")
            ids = list(stat["ran_trip_ids"])
            if governing is not None and ids:
                checked = len(ids)
                active = governing.services_by_date[day]
                not_in_gtfs = sum(1 for t in ids if t not in governing.trip_services)
                not_scheduled = sum(
                    1
                    for t in ids
                    if t in governing.trip_services and governing.trip_services[t] not in active
                )
                share = (not_in_gtfs + not_scheduled) / checked
                if share > max_unscheduled_share:
                    reasons.append(
                        f"{share:.0%} of trips run are not scheduled on this date in GTFS "
                        f"({not_in_gtfs} not in trips.txt)"
                    )
        scheduled = int(coverage.get(stamp, {}).get("trips_scheduled", 0) or 0)
        unrecorded = int(coverage.get(stamp, {}).get("trips_without_record", 0) or 0)
        unrecorded_share = unrecorded / scheduled if scheduled else 0.0
        if scheduled and unrecorded_share > max_unrecorded_share:
            reasons.append(
                f"{unrecorded_share:.0%} of scheduled trips have no TIDES record "
                f"({unrecorded} of {scheduled})"
            )
        counts = {
            col: int(visit_counts.get(stamp, {}).get(col, 0) or 0)
            for col in ("duplicate_visits_dropped", "conflicting_visits_dropped")
        }
        if counts["conflicting_visits_dropped"]:
            reasons.append(
                f"{counts['conflicting_visits_dropped']} conflicting duplicate stop visit(s) "
                "(the first of each kept)"
            )
        rows.append(
            {
                "service_date": day.isoformat(),
                "weekday": weekday,
                "day_type": " / ".join(types),
                "day_type_source": source,
                "gtfs_feed": governing.name if governing else "",
                "gtfs_services": ";".join(services),
                "gtfs_service_labels": ";".join(labels),
                "revenue_trips": revenue,
                "canceled_trips": canceled,
                "trips_run": run,
                "trips_checked_against_gtfs": checked,
                "trips_not_scheduled": not_scheduled,
                "trips_not_in_gtfs": not_in_gtfs,
                "trips_scheduled": scheduled,
                "trips_without_record": unrecorded,
                "share_without_record": round(unrecorded_share, 3),
                **counts,
                "flagged": bool(reasons),
                "flag_reason": "; ".join(reasons),
                "_stamp": stamp,
                "_day_types": types,
            }
        )
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def date_table_from_classification(
    classification: pd.DataFrame, exclude_flagged: bool = False
) -> pd.DataFrame:
    """Expand the classification into one row per date and day type.

    Dates in no day type drop out; flagged dates drop out too when
    *exclude_flagged* is True.

    Args:
        classification: Output of :func:`classify_service_dates`.
        exclude_flagged: Whether to leave flagged dates out.

    Returns:
        ``service_date`` (midnight timestamp) and ``day_type`` columns.
    """
    rows = [
        (stamp, day_type)
        for stamp, types, flagged in zip(
            classification["_stamp"], classification["_day_types"], classification["flagged"]
        )
        if not (exclude_flagged and flagged)
        for day_type in types
    ]
    return pd.DataFrame(rows, columns=["service_date", "day_type"])


def log_date_classification(classification: pd.DataFrame, holiday_day_type: str) -> None:
    """Log the holidays found, the flagged dates, and dates in no day type."""
    holidays = classification.loc[classification["_day_types"].map(lambda t: holiday_day_type in t)]
    for _, row in holidays.iterrows():
        logging.info(
            "%s (%s) is in the %s day type: %s.",
            row["service_date"],
            row["weekday"],
            holiday_day_type,
            row["day_type_source"],
        )
    flagged = classification.loc[classification["flagged"]]
    if not flagged.empty:
        listed = [f"{d} ({why})" for d, why in zip(flagged["service_date"], flagged["flag_reason"])]
        logging.warning(
            "%d service date(s) are flagged in %s: %s",
            len(flagged),
            DATE_CLASSIFICATION_FILENAME,
            "; ".join(listed[:10]) + ("; ..." if len(listed) > 10 else ""),
        )
    untyped = classification.loc[classification["day_type"] == "", "service_date"].tolist()
    if untyped:
        logging.info(
            "%d service date(s) fall on a weekday in no day type and are left out: %s",
            len(untyped),
            ", ".join(untyped[:10]) + (" ..." if len(untyped) > 10 else ""),
        )


def day_type_order(
    day_types: Mapping[str, Sequence[int]], holiday_day_type: str = HOLIDAY_DAY_TYPE
) -> List[str]:
    """Return the output order of day types: ``DAY_TYPES`` order, then the holiday type."""
    return [*day_types, holiday_day_type]


def add_day_type(df: pd.DataFrame, date_table: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its date's day type(s); rows on dates in no day type drop out.

    Args:
        df: Rows with a parsed ``service_date``.
        date_table: Output of :func:`build_date_table`.

    Returns:
        *df* with a ``day_type`` column (a row appears once per day type of its date).
    """
    return df.merge(date_table, on="service_date", how="inner")


# =============================================================================
# SERVICE-DAY CLOCK
# =============================================================================

# Clock time inside an ISO 8601 timestamp (2026-01-05T21:03:14-05:00 -> 21:03).
_ISO_CLOCK_RE = r"^\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})"
# Date and hour of an ISO 8601 timestamp, for the service-day hour.
_ISO_DATE_HOUR_RE = r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}):\d{2}"
# Time-only value, hours possibly past 23 as in GTFS (25:10:00 -> 01:10).
_TIME_ONLY_RE = r"^(\d{1,2}):([0-5]\d)(?::[0-5]\d(?:\.\d+)?)?$"


def minute_of_day(values: pd.Series) -> pd.Series:
    """Return each value's clock time as minutes since midnight (0-1439).

    ISO 8601 timestamps are read as written, ignoring any UTC offset, so a
    daylight-saving change within the data does not shift them. Time-only
    values may run past 24:00 (GTFS style) and wrap to the next day's clock.
    Anything else is handed to :func:`pandas.to_datetime`.

    Args:
        values: Timestamps or times as text.

    Returns:
        Float minutes since midnight, NaN where the value is blank or unreadable.
    """
    text = values.astype("string").str.strip()
    text = text.mask(text == "", pd.NA)
    minutes = pd.Series(np.nan, index=values.index, dtype="float64")

    iso = text.str.extract(_ISO_CLOCK_RE)
    hours = pd.to_numeric(iso[0], errors="coerce")
    mins = pd.to_numeric(iso[1], errors="coerce")
    valid = (hours < 24) & (mins < 60)
    minutes = minutes.mask(valid, hours * 60 + mins)

    rest = minutes.isna() & text.notna()
    if rest.any():
        parts = text[rest].str.extract(_TIME_ONLY_RE)
        clock = pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(
            parts[1], errors="coerce"
        )
        minutes.loc[rest] = (clock % MINUTES_PER_DAY).to_numpy(dtype="float64")

    rest = minutes.isna() & text.notna()
    if rest.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stamp = pd.to_datetime(text[rest], errors="coerce")
            except (ValueError, TypeError):
                stamp = None
        if stamp is not None and pd.api.types.is_datetime64_any_dtype(stamp):
            parsed = stamp.dt.hour * 60 + stamp.dt.minute
            minutes.loc[rest] = parsed.to_numpy(dtype="float64", na_value=np.nan)
    return minutes


def _to_service_hour(clock: pd.Series, offset: pd.Series, cutover_hour: int) -> pd.Series:
    """Move clock hours onto the service-day clock.

    A stamp dated the day after its service date gains 24 hours (00:30 -> 24).
    Otherwise an hour before *cutover_hour* gains 24 too, for vendors that stamp
    small-hours events with the operating day's date.

    Args:
        clock: Clock hours 0-23.
        offset: Days between each stamp's date and its service date (NaN unknown).
        cutover_hour: First hour of the service day.

    Returns:
        Service-day hours.
    """
    next_day = offset == 1
    early = clock < cutover_hour
    return clock.where(~(next_day | early), clock + 24)


def service_day_hour(values: pd.Series, service_dates: pd.Series, cutover_hour: int) -> pd.Series:
    """Return each value's hour on the service-day clock (cutover to cutover + 23).

    ISO 8601 timestamps keep the date and hour as written and are compared with
    the service date (a stamp on the next calendar day lands at hour 24 or later).
    Time-only values past 23:59 keep their GTFS hour; earlier ones before the
    cutover move to the 24+ end. Anything else is handed to
    :func:`pandas.to_datetime`.

    Args:
        values: Timestamps or times as text.
        service_dates: Each row's service date.
        cutover_hour: First hour of the service day (``LATE_NIGHT_CUTOVER_HOUR``).

    Returns:
        Float service-day hours, NaN where the value is blank or unreadable.
    """
    text = values.astype("string").str.strip()
    text = text.mask(text == "", pd.NA)
    days = pd.to_datetime(service_dates, errors="coerce").dt.normalize()
    hours = pd.Series(np.nan, index=values.index, dtype="float64")

    iso = text.str.extract(_ISO_DATE_HOUR_RE)
    stamp_day = pd.to_datetime(iso[0], format="%Y-%m-%d", errors="coerce")
    clock = pd.to_numeric(iso[1], errors="coerce")
    offset = (stamp_day - days).dt.days
    valid = clock.notna() & stamp_day.notna() & (clock < 24)
    hours = hours.mask(valid, _to_service_hour(clock, offset, cutover_hour))

    rest = hours.isna() & text.notna()
    if rest.any():
        parts = text[rest].str.extract(_TIME_ONLY_RE)
        hour = pd.to_numeric(parts[0], errors="coerce")
        hour = hour.where(hour >= 24, hour.where(hour >= cutover_hour, hour + 24))
        hours.loc[rest] = hour.to_numpy(dtype="float64")

    rest = hours.isna() & text.notna()
    if rest.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stamp = pd.to_datetime(text[rest], errors="coerce")
            except (ValueError, TypeError):
                stamp = None
        if stamp is not None and pd.api.types.is_datetime64_any_dtype(stamp):
            if stamp.dt.tz is not None:
                stamp = stamp.dt.tz_localize(None)
            later = (stamp.dt.normalize() - days[rest]).dt.days
            shifted = _to_service_hour(stamp.dt.hour.astype("float64"), later, cutover_hour)
            hours.loc[rest] = shifted.to_numpy(dtype="float64", na_value=np.nan)
    return hours


# =============================================================================
# TIMESTAMPS & TIME KEYS
# =============================================================================

# An ISO 8601 date and time at the start of a value, and a trailing UTC offset or
# "Z" on one; the offset is dropped so the clock time is read as written.
_ISO_START_RE = r"^\d{4}-\d{2}-\d{2}[T ]\d"
_OFFSET_RE = r"(?:Z|[+-]\d{2}:?\d{2})$"

# pandas 2 reads ISO 8601 and mixed formats only when asked (format="ISO8601" /
# "mixed"); pandas 1 (e.g. older ArcGIS Pro) has neither and parses value by value.
_PANDAS_MAJOR: int = int(pd.__version__.split(".", 1)[0])


def _naive(stamps: pd.Series) -> pd.Series:
    """Return *stamps* as naive datetimes (anything unreadable becomes NaT)."""
    if not pd.api.types.is_datetime64_any_dtype(stamps):
        stamps = pd.to_datetime(stamps, errors="coerce", utc=True)
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        stamps = stamps.dt.tz_localize(None)
    return stamps


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse timestamps as written, ignoring any UTC offset.

    ISO 8601 text is read with its offset (or ``Z``) dropped, so every value keeps
    the local clock time it was written in and differences between them are plain
    clock differences. Anything else is handed to pandas' mixed-format parser.

    Args:
        values: Timestamps as text.

    Returns:
        Naive datetimes, NaT where the value is blank or unreadable.
    """
    text = values.astype("string").str.strip()
    text = text.mask(text == "", pd.NA)
    iso = text.str.match(_ISO_START_RE).fillna(False).astype(bool)
    text = text.where(~iso, text.str.replace(_OFFSET_RE, "", regex=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _PANDAS_MAJOR < 2:
            return _naive(pd.to_datetime(text.astype(object), errors="coerce"))
        stamps = _naive(pd.to_datetime(text, errors="coerce", format="ISO8601"))
        rest = stamps.isna() & text.notna()
        if rest.any():
            try:
                others = _naive(pd.to_datetime(text[rest], errors="coerce", format="mixed"))
            except (ValueError, TypeError):
                others = pd.Series(pd.NaT, index=text[rest].index, dtype="datetime64[ns]")
            stamps = stamps.astype("datetime64[ns]")
            stamps.loc[rest] = others.astype("datetime64[ns]")
    return stamps


def _first_present(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return the first non-blank value per row across *columns* (NA where none)."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        text = df[col].astype("string").str.strip()
        filled = text.notna() & (text != "").fillna(False)
        out = out.where(out.notna() | ~filled, text.astype(object))
    return out


def add_parsed_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the stop-visit timestamps once and add the columns every measure uses.

    Adds ``<column>_ts`` for each of :data:`TIMESTAMP_COLUMNS` (NaT where the column
    is absent), ``_sched_ts`` / ``_actual_ts`` (departure, else arrival) and the
    matching text in ``_sched_text`` / ``_actual_text``.

    Args:
        df: Joined stop visits.

    Returns:
        A copy of *df* with the parsed columns.
    """
    out = df.copy()
    for col in TIMESTAMP_COLUMNS:
        if col in out.columns:
            out[f"{col}_ts"] = parse_timestamps(out[col])
        else:
            out[f"{col}_ts"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    out["_sched_ts"] = out["schedule_departure_time_ts"].fillna(out["schedule_arrival_time_ts"])
    out["_actual_ts"] = out["actual_departure_time_ts"].fillna(out["actual_arrival_time_ts"])
    out["_sched_text"] = _first_present(out, ("schedule_departure_time", "schedule_arrival_time"))
    out["_actual_text"] = _first_present(out, ("actual_departure_time", "actual_arrival_time"))
    return out


def service_day_minutes(
    values: pd.Series, service_dates: pd.Series, cutover_hour: int
) -> pd.Series:
    """Return each value's time as minutes on the service-day clock (may pass 1440).

    Args:
        values: Timestamps or times as text.
        service_dates: Each row's service date.
        cutover_hour: First hour of the service day.

    Returns:
        Float minutes (an 00:30 stamp after its service date is 1470), NaN where the
        value is blank or unreadable.
    """
    minutes = minute_of_day(values)
    hours = service_day_hour(values, service_dates, cutover_hour)
    clock = (minutes // 60).astype("float64")
    hours = hours.fillna(clock.where(clock >= cutover_hour, clock + 24))
    return hours * 60 + minutes % 60


def add_time_keys(
    df: pd.DataFrame,
    text: pd.Series,
    periods: Sequence[TimePeriod],
    cutover_hour: int,
    what: str,
) -> pd.DataFrame:
    """Add the time-cut columns: ``time_period``, ``hour``, ``month`` and ``day_of_week``.

    The time period and hour come from *text* (the clock time the time basis
    chose); the month and weekday come from ``service_date``. Rows with no readable
    time keep a blank ``time_period`` and ``hour``: they still count in the all-day,
    month and day-of-week tables and are left out of the time-period and hour ones.

    Args:
        df: Rows with a parsed ``service_date``.
        text: Each row's clock time as text (same index as *df*).
        periods: Checked time periods covering the whole day.
        cutover_hour: First hour of the service day.
        what: What the rows are (e.g. ``"scored visits"``), for the log.

    Returns:
        A copy of *df* with the four columns.
    """
    out = df.copy()
    sample = text.dropna().head(1000).astype(str).str.strip()
    if not sample.empty and sample.str.contains(r"(?:Z|[+-]00:?00)$").mean() > 0.5:
        logging.warning(
            "The %s times look like UTC. Time periods and hours use the clock time as "
            "written, so convert the timestamps to local time first if they are UTC.",
            what,
        )
    minutes = minute_of_day(text)
    known = minutes.notna()
    lookup = period_lookup(periods)
    labels = pd.Series(None, index=out.index, dtype="object")
    labels.loc[known] = lookup[minutes.loc[known].to_numpy(dtype=int)]
    out["time_period"] = labels
    hours = service_day_hour(text, out["service_date"], cutover_hour)
    clock = (minutes // 60).astype("float64")
    out["hour"] = hours.fillna(clock.where(clock >= cutover_hour, clock + 24))
    out["month"] = out["service_date"].dt.strftime("%Y-%m")
    out["day_of_week"] = out["service_date"].dt.dayofweek.map(dict(enumerate(WEEKDAY_NAMES)))
    if (~known).any():
        logging.warning(
            "%d %s have no readable time for the time basis; they count in the all-day, "
            "month and day-of-week tables but not the time-period and hour tables.",
            int((~known).sum()),
            what,
        )
    return out


def _not_skipped(df: pd.DataFrame) -> pd.Series:
    """Return a mask of the visits not marked Skipped."""
    if "schedule_relationship" not in df.columns:
        return pd.Series(True, index=df.index)
    return _casefolded(df["schedule_relationship"], "scheduled") != "skipped"


def _sorted_by_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Sort visits by performed trip and numeric ``trip_stop_sequence`` (row order if absent)."""
    if "trip_stop_sequence" not in df.columns:
        return df.sort_values(TRIP_KEY, kind="stable")
    seq = pd.to_numeric(df["trip_stop_sequence"], errors="coerce")
    return df.assign(_seq=seq).sort_values([*TRIP_KEY, "_seq"], kind="stable").drop(columns="_seq")


def build_trip_frame(visits: pd.DataFrame, cutover_hour: int) -> pd.DataFrame:
    """Return one row per performed trip with its attributes and both start times.

    ``_start_scheduled`` is ``schedule_trip_start``, else the scheduled time at the
    trip's first served stop; ``_start_actual`` is ``actual_trip_start``, else the
    actual time there. ``_start_min`` is the scheduled start on the service-day
    clock, used to label and order trips.

    Args:
        visits: Stop visits with :func:`add_parsed_timestamps` columns.
        cutover_hour: First hour of the service day.

    Returns:
        One row per ``TRIP_KEY``.
    """
    attrs = [
        c
        for c in ("route_id", "direction_id", "route_type_agency", "trip_id", "block_id")
        if c in visits.columns
    ]
    served = _sorted_by_sequence(visits.loc[_not_skipped(visits)])
    first_sched = (
        served.loc[served["_sched_text"].notna()]
        .groupby(TRIP_KEY, sort=False)
        .head(1)[[*TRIP_KEY, "_sched_text"]]
    )
    first_actual = (
        served.loc[served["_actual_text"].notna()]
        .groupby(TRIP_KEY, sort=False)
        .head(1)[[*TRIP_KEY, "_actual_text"]]
    )
    starts = [c for c in ("schedule_trip_start", "actual_trip_start") if c in visits.columns]
    trips = visits.drop_duplicates(TRIP_KEY)[[*TRIP_KEY, *attrs, *starts]]
    trips = trips.merge(first_sched, on=TRIP_KEY, how="left").merge(
        first_actual, on=TRIP_KEY, how="left"
    )
    trips["_start_scheduled"] = _first_present(trips, ("schedule_trip_start", "_sched_text"))
    trips["_start_actual"] = _first_present(trips, ("actual_trip_start", "_actual_text"))
    trips["_start_min"] = service_day_minutes(
        trips["_start_scheduled"], trips["service_date"], cutover_hour
    )
    return trips.drop(columns=["_sched_text", "_actual_text", *starts]).reset_index(drop=True)


# =============================================================================
# GRAIN HELPERS
# =============================================================================


def level_columns(level: str, columns: Iterable[str]) -> Optional[List[str]]:
    """Return *level*'s key columns found in *columns*, or ``None`` if a required one is missing.

    Args:
        level: A key of :data:`LEVELS`.
        columns: The columns available.

    Returns:
        The key columns, in order, or ``None``.
    """
    available = set(columns)
    cols: List[str] = []
    for col, required in LEVELS[level]:
        if col in available:
            cols.append(col)
        elif required:
            return None
    return cols


def missing_level_columns(level: str, columns: Iterable[str]) -> List[str]:
    """Return the required key columns of *level* that are not in *columns*."""
    available = set(columns)
    return [col for col, required in LEVELS[level] if required and col not in available]


def _sort_table(
    table: pd.DataFrame,
    sort_cols: Sequence[str],
    day_order: Sequence[str],
    periods: Sequence[TimePeriod] = (),
) -> pd.DataFrame:
    """Sort *table* by *sort_cols*, with day types, periods and weekdays in their own order."""
    work = table.copy()
    order_cols: List[str] = []
    helpers: List[str] = []
    ranks: Dict[str, Dict[str, int]] = {
        "day_type": {d: i for i, d in enumerate(day_order)},
        "time_period": {p.name: i for i, p in enumerate(periods)},
        "day_of_week": {d: i for i, d in enumerate(WEEKDAY_NAMES)},
    }
    for col in sort_cols:
        rank = ranks.get(col)
        if rank:
            helper = f"_{col}_rank"
            work[helper] = work[col].map(rank).fillna(len(rank))
            order_cols.append(helper)
            helpers.append(helper)
        else:
            order_cols.append(col)
    work = work.sort_values(order_cols, kind="stable")
    return work.drop(columns=helpers).reset_index(drop=True)


def _cut_keys(cut: str) -> Tuple[List[str], List[str]]:
    """Split a time cut's columns into date-level keys and time-of-day keys.

    Args:
        cut: A key of :data:`TIME_CUTS`.

    Returns:
        ``(date keys, time keys)``: ``["day_type"]`` plus ``"month"`` or
        ``"day_of_week"`` when the cut has one, and ``[]``, ``["time_period"]`` or
        ``["hour"]``.
    """
    temporal = TIME_CUTS[cut]
    date_keys = ["day_type", *[t for t in temporal if t in DATE_CUT_COLUMNS]]
    time_keys = [t for t in temporal if t not in DATE_CUT_COLUMNS]
    return date_keys, time_keys


def _rows_with_time(rows: pd.DataFrame, time_keys: Sequence[str]) -> pd.DataFrame:
    """Keep the rows placed on the clock when the cut needs a time of day."""
    if not time_keys:
        return rows
    work = rows.loc[rows[time_keys[0]].notna()]
    if list(time_keys) == ["hour"]:
        work = work.assign(hour=work["hour"].astype(int))
    return work


def _add_time_columns(
    agg: pd.DataFrame, time_keys: Sequence[str], periods: Sequence[TimePeriod]
) -> Tuple[pd.DataFrame, List[str]]:
    """Add the period window or hour label beside a cut's time keys.

    Returns:
        The table and its time columns in output order.
    """
    if list(time_keys) == ["time_period"]:
        by_name = {p.name: p for p in periods}
        agg["period_start"] = agg["time_period"].map(lambda n: _clock(by_name[n].start))
        agg["period_end"] = agg["time_period"].map(lambda n: _clock(by_name[n].end))
        agg["period_hours"] = agg["time_period"].map(lambda n: round(by_name[n].minutes / 60, 2))
        return agg, ["time_period", "period_start", "period_end", "period_hours"]
    if list(time_keys) == ["hour"]:
        agg["hour"] = agg["hour"].astype(int)
        agg["hour_label"] = agg["hour"].map(lambda h: minutes_to_hhmm(h * 60))
        return agg, ["hour", "hour_label"]
    return agg, list(time_keys)


def _pct(part: pd.Series, whole: pd.Series) -> pd.Series:
    """Return ``part / whole * 100`` rounded to one decimal (NaN where *whole* is 0)."""
    return (part / whole.where(whole > 0) * 100.0).round(1)


def _start_label(minutes: pd.Series) -> pd.Series:
    """Format service-day minutes as ``HH:MM`` (may pass 24:00); blank where missing."""
    return minutes.map(lambda m: minutes_to_hhmm(m))


def _trip_sort_cols(date_keys: Sequence[str], level_cols: Sequence[str]) -> List[str]:
    """Sort keys for a trip table: date keys, route and direction, then scheduled start."""
    lead = [c for c in level_cols if c != "trip_id"]
    return [*date_keys, *lead, "_start", "trip_id"]


def grain_filename(prefix: str, level: str, cut: str) -> str:
    """Return the CSV name for one grain, e.g. ``otp_by_route_and_time_period.csv``."""
    suffix = "".join(f"_and_{col}" for col in TIME_CUTS[cut])
    return f"{prefix}_by_{level}{suffix}.csv"


# =============================================================================
# ON-TIME PERFORMANCE
# =============================================================================


def is_true(values: pd.Series) -> pd.Series:
    """Return a mask of the values that read as TRUE (``TRUE``, ``1``, ``yes``, ...)."""
    return values.fillna("").astype(str).str.strip().str.casefold().isin(TRUE_TEXT)


def otp_candidates(visits: pd.DataFrame, timepoints_only: bool) -> pd.DataFrame:
    """Return the visits OTP is judged on: timepoints (when set) marked Scheduled.

    Args:
        visits: Stop visits with :func:`add_parsed_timestamps` columns.
        timepoints_only: Keep only ``timepoint`` TRUE visits.

    Returns:
        The candidate visits (scored or not).
    """
    out = visits
    if timepoints_only:
        if "timepoint" in out.columns:
            out = out.loc[is_true(out["timepoint"])]
        else:
            logging.warning("stop_visits has no timepoint column; every Scheduled visit is scored.")
    if "schedule_relationship" in out.columns:
        out = out.loc[_casefolded(out["schedule_relationship"], "scheduled") == "scheduled"]
    return out


def last_stop_mask(visits: pd.DataFrame) -> pd.Series:
    """Mark each visit made at its trip's last stop.

    A visit is at the last stop when its ``stop_id`` is the trip's
    ``trip_end_stop_id`` and it is the trip's last visit by ``trip_stop_sequence``
    (row order where that is absent). Requiring both keeps a loop trip's first
    visit, at the same stop, from counting, and keeps an earlier stop from counting
    when the last-stop visit is missing from the export. Trips without a
    ``trip_end_stop_id`` have no last stop and are counted in a warning.

    Args:
        visits: Joined stop visits.

    Returns:
        A boolean mask aligned to *visits*.
    """
    none = pd.Series(False, index=visits.index)
    if visits.empty or "stop_id" not in visits.columns:
        return none
    if "trip_end_stop_id" not in visits.columns:
        logging.warning(
            "trips_performed has no trip_end_stop_id, so no trip's last stop is known: "
            "every stop is scored on departure and LAST_STOP_TREATMENT has no effect."
        )
        return none
    end_stop = visits["trip_end_stop_id"].fillna("").astype(str).str.strip()
    no_end = end_stop.eq("")
    if no_end.any():
        logging.warning(
            "%d performed trip(s) have no trip_end_stop_id; their last stop is scored on "
            "departure like any other stop, and LAST_STOP_TREATMENT does not apply to it.",
            int(visits.loc[no_end, TRIP_KEY].drop_duplicates().shape[0]),
        )
    if "trip_stop_sequence" in visits.columns:
        seq = pd.to_numeric(visits["trip_stop_sequence"], errors="coerce")
    else:
        seq = pd.Series(np.arange(len(visits), dtype="float64"), index=visits.index)
    is_last = seq.eq(seq.groupby([visits[c] for c in TRIP_KEY]).transform("max"))
    stop = visits["stop_id"].fillna("").astype(str).str.strip()
    return is_last & stop.eq(end_stop) & ~no_end


def resolve_express_routes(
    visits: pd.DataFrame,
    routes_file: Optional[str],
    inline: Sequence[str],
    service_types: Sequence[str],
) -> Tuple[frozenset[str], str]:
    """Pick the express route_ids and say where they came from.

    The route_ids in *inline* and the express-routes file together (see
    :func:`load_express_route_ids`); when neither lists any, the routes whose
    ``route_type_agency`` is one of *service_types* (compared without regard to
    case). Listed routes that do not appear in *visits* are named in a warning.

    Args:
        visits: Joined stop visits (``route_id`` and, for the fallback,
            ``route_type_agency``).
        routes_file: Path to a text file of route_ids, if any.
        inline: The ``EXPRESS_ROUTES`` list.
        service_types: The ``EXPRESS_SERVICE_TYPES`` values.

    Returns:
        The express route_ids and a description of their source.
    """
    listed = load_express_route_ids(inline, routes_file)
    if listed:
        sources = ["EXPRESS_ROUTES"] if inline else []
        if routes_file:
            sources.append(f"express-routes file '{routes_file}'")
        known = (
            set(visits["route_id"].dropna().astype(str).str.strip())
            if "route_id" in visits.columns
            else set()
        )
        absent = sorted(listed - known)
        if absent:
            logging.warning("Express route(s) not in the data: %s.", ", ".join(absent))
        return frozenset(listed), " and ".join(sources)
    source = f"route_type_agency in EXPRESS_SERVICE_TYPES ({', '.join(service_types)})"
    if "route_type_agency" not in visits.columns or "route_id" not in visits.columns:
        logging.warning("No express routes: the visits have no route_type_agency column.")
        return frozenset(), source
    wanted = {str(t).strip().casefold() for t in service_types}
    express = visits.loc[_casefolded(visits["route_type_agency"], "").isin(wanted), "route_id"]
    return frozenset(express.dropna().astype(str).str.strip()) - {""}, source


def _arrival_at_last_stop(candidates: pd.DataFrame) -> pd.DataFrame:
    """Switch last-stop visits' time columns to arrival, each falling back to departure."""
    at_last = candidates["_at_last_stop"].to_numpy(dtype=bool)
    if not at_last.any():
        return candidates
    out = candidates.copy()
    for prefix, kind in (("_sched", "schedule"), ("_actual", "actual")):
        arrival = out[f"{kind}_arrival_time_ts"].fillna(out[f"{kind}_departure_time_ts"])
        out[f"{prefix}_ts"] = out[f"{prefix}_ts"].where(~at_last, arrival)
        text = _first_present(out, (f"{kind}_arrival_time", f"{kind}_departure_time"))
        out[f"{prefix}_text"] = out[f"{prefix}_text"].where(~at_last, text)
    return out


def score_otp(
    visits: pd.DataFrame,
    early_min: float = EARLY_MIN,
    late_min: float = LATE_MIN,
    timepoints_only: bool = TIMEPOINTS_ONLY,
    last_stop_treatment: str = LAST_STOP_TREATMENT,
    treated_routes: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Score each candidate visit as early, on time or late.

    The deviation is the actual minus the scheduled departure in minutes, each
    falling back to the arrival where the departure is missing; at a trip's last
    stop (:func:`last_stop_mask`) it is arrival, falling back to departure, and the
    visit is placed in time by its arrival too. Visits without both sides cannot be
    scored.

    Args:
        visits: Stop visits with :func:`add_parsed_timestamps` columns.
        early_min: Lower (inclusive) bound of the on-time window, in minutes.
        late_min: Upper (inclusive) bound of the on-time window, in minutes.
        timepoints_only: Score timepoint visits only.
        last_stop_treatment: How the last stop counts on *treated_routes*: one of
            :data:`LAST_STOP_TREATMENTS` (see ``LAST_STOP_TREATMENT``).
        treated_routes: The route_ids the treatment applies to; ``None`` applies it
            to every route.

    Returns:
        The scored visits with ``dev_min`` and ``otp_class``, and counts of the
        candidates, the scored visits, why the rest were not scored, the last-stop
        visits left out, and the early last-stop visits scored on time.
    """
    candidates = otp_candidates(
        visits.assign(_at_last_stop=last_stop_mask(visits)), timepoints_only
    )
    treated = candidates["_at_last_stop"].to_numpy(dtype=bool)
    if treated_routes is not None:
        wanted = {str(r).strip() for r in treated_routes}
        if "route_id" in candidates.columns:
            route = candidates["route_id"].fillna("").astype(str).str.strip()
            treated = treated & route.isin(wanted).to_numpy(dtype=bool)
        else:
            treated = np.zeros(len(candidates), dtype=bool)
    excluded = 0
    if last_stop_treatment == "exclude":
        excluded = int(treated.sum())
        candidates, treated = candidates.loc[~treated], treated[~treated]
    candidates = _arrival_at_last_stop(candidates.assign(_treated=treated))
    dev = (candidates["_actual_ts"] - candidates["_sched_ts"]) / pd.Timedelta(minutes=1)
    no_schedule = candidates["_sched_ts"].isna()
    counts = {
        "candidates": len(candidates),
        "scored": int(dev.notna().sum()),
        "missing_schedule_time": int(no_schedule.sum()),
        "missing_actual_time": int((~no_schedule & candidates["_actual_ts"].isna()).sum()),
        "last_stop_excluded": excluded,
        "early_ok_at_last_stop": 0,
    }
    scored = candidates.assign(dev_min=dev).loc[dev.notna()].copy()
    scored["otp_class"] = np.select(
        [scored["dev_min"] < early_min, scored["dev_min"] > late_min],
        ["early", "late"],
        default="on_time",
    )
    if last_stop_treatment == "early_ok":
        forgiven = scored["otp_class"].eq("early") & scored["_treated"]
        scored.loc[forgiven, "otp_class"] = "on_time"
        counts["early_ok_at_last_stop"] = int(forgiven.sum())
    return scored.drop(columns=["_at_last_stop", "_treated"]), counts


def warn_extreme_deviations(scored: pd.DataFrame, warn_abs_min: float = DEV_ABS_WARN_MIN) -> None:
    """Warn when scored deviations exceed *warn_abs_min* minutes either way.

    Such rows are kept (they score as early or late like any other), but a
    multi-hour deviation is nearly always a schedule joined to the wrong trip or a
    vehicle clock defect.

    Args:
        scored: Scored visits with ``dev_min``.
        warn_abs_min: Absolute deviation threshold in minutes.
    """
    extreme = scored["dev_min"].abs() > warn_abs_min
    if extreme.any():
        logging.warning(
            "%d scored visits deviate more than %.0f minutes from schedule (worst: %.0f min). "
            "They score as early or late but usually point to schedule-join or clock defects.",
            int(extreme.sum()),
            warn_abs_min,
            float(scored.loc[extreme, "dev_min"].abs().max()),
        )


def warn_thin_day_types(rows: pd.DataFrame, min_dates: int = THIN_DAY_TYPE_DATES) -> None:
    """Warn for day types observed on fewer than *min_dates* service dates.

    Args:
        rows: Day-typed rows with ``service_date``.
        min_dates: Minimum distinct service dates for a day type to pass quietly.
    """
    per_day_type = rows.groupby("day_type", dropna=False)["service_date"].nunique()
    for name, n_dates in per_day_type.items():
        if int(n_dates) < min_dates:
            logging.warning(
                "Day type %r is backed by only %d service date(s); its figures hinge on a "
                "handful of days.",
                name,
                int(n_dates),
            )


def summarize_otp(
    scored: pd.DataFrame,
    level: str,
    cut: str,
    periods: Sequence[TimePeriod],
    day_order: Sequence[str],
) -> pd.DataFrame:
    """Aggregate scored visits to one OTP table (one level crossed with one time cut).

    Args:
        scored: Day-typed scored visits with the time-cut columns and the level's keys.
        level: A key of :data:`LEVELS`; its required columns must be present.
        cut: A key of :data:`TIME_CUTS`.
        periods: The checked time periods (their order sets the row order).
        day_order: Output order of the day types.

    Returns:
        The table: date keys, level keys, time columns, then ``early``, ``on_time``,
        ``late``, ``evaluated``, the three percentages, the mean and median deviation,
        ``trips_observed`` and ``days_observed`` (a trip table adds its scheduled start).
    """
    level_cols = level_columns(level, scored.columns) or []
    date_keys, time_keys = _cut_keys(cut)
    keys = [*date_keys, *level_cols, *time_keys]
    work = _rows_with_time(scored, time_keys).assign(
        _early=lambda d: d["otp_class"].eq("early"),
        _on_time=lambda d: d["otp_class"].eq("on_time"),
        _late=lambda d: d["otp_class"].eq("late"),
        _trip=lambda d: d["service_date"].astype(str) + "|" + d["trip_id_performed"].astype(str),
    )
    specs: Dict[str, Tuple[str, Any]] = {
        "early": ("_early", "sum"),
        "on_time": ("_on_time", "sum"),
        "late": ("_late", "sum"),
        "evaluated": ("dev_min", "size"),
        "mean_deviation_min": ("dev_min", "mean"),
        "median_deviation_min": ("dev_min", "median"),
        "trips_observed": ("_trip", "nunique"),
        "days_observed": ("service_date", "nunique"),
    }
    if level == "trip" and "_start_min" in work.columns:
        specs["_start"] = ("_start_min", "median")
    agg = work.groupby(keys, sort=False, dropna=False).agg(**specs).reset_index()
    for col in ("early", "on_time", "late", "evaluated", "trips_observed", "days_observed"):
        agg[col] = agg[col].astype(int)
    agg["pct_on_time"] = _pct(agg["on_time"], agg["evaluated"])
    agg["pct_early"] = _pct(agg["early"], agg["evaluated"])
    agg["pct_late"] = _pct(agg["late"], agg["evaluated"])
    agg["mean_deviation_min"] = agg["mean_deviation_min"].round(2)
    agg["median_deviation_min"] = agg["median_deviation_min"].round(2)
    agg, time_cols = _add_time_columns(agg, time_keys, periods)
    lead: List[str] = []
    sort_cols = keys
    if "_start" in agg.columns:
        agg["scheduled_start"] = _start_label(agg["_start"])
        lead = ["scheduled_start"]
        sort_cols = [*_trip_sort_cols(date_keys, level_cols), *time_keys]
    measures = [
        "early",
        "on_time",
        "late",
        "evaluated",
        "pct_on_time",
        "pct_early",
        "pct_late",
        "mean_deviation_min",
        "median_deviation_min",
        "trips_observed",
        "days_observed",
    ]
    table = _sort_table(agg, sort_cols, day_order, periods)
    return table[[*date_keys, *level_cols, *lead, *time_cols, *measures]]


# --- stop flags (coverage and OTP by stop, judged route by route) ---------------


def add_pattern_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``pattern_key``: ``pattern_id``, else ``route_id | direction_id``.

    The fallback pools branching variants within a direction, which can overstate
    expected service at branch-only stops, hence the warning.

    Args:
        df: Trips or visits with ``route_id`` (and ``direction_id``, ``pattern_id``
            where present).

    Returns:
        A copy with a text ``pattern_key`` column.
    """
    out = df.copy()
    direction = (
        out["direction_id"].astype(str)
        if "direction_id" in out.columns
        else pd.Series("", index=out.index)
    )
    fallback = out["route_id"].astype(str) + " | " + direction
    if "pattern_id" in out.columns:
        pattern = out["pattern_id"].fillna("").astype(str).str.strip()
        if (pattern != "").any():
            out["pattern_key"] = pattern.where(pattern != "", fallback)
            return out
    logging.warning(
        "trips_performed has no usable pattern_id; the stop flags use route + direction as "
        "the pattern. Branching variants within a direction are pooled, which can overstate "
        "expected service (and understate coverage) at branch-only stops."
    )
    out["pattern_key"] = fallback
    return out


def stop_flag_candidates(visits: pd.DataFrame, timepoints_only: bool) -> pd.DataFrame:
    """Return the visits that define each stop's expected service.

    Every emitted visit at a judged stop counts: timepoints only (when set), minus
    Added visits (never scheduled, so they should not define a pattern's stops) and
    rows with no ``stop_id``. Skipped and unscored visits stay in: they mark the stop
    as scheduled to be served and feed the per-cause diagnostics.

    Args:
        visits: Stop visits with :func:`add_parsed_timestamps` columns.
        timepoints_only: Keep only ``timepoint`` TRUE visits.

    Returns:
        The candidate visits with ``dev_min``.
    """
    out = visits
    if timepoints_only and "timepoint" in out.columns:
        out = out.loc[is_true(out["timepoint"])]
    if "schedule_relationship" in out.columns:
        out = out.loc[_casefolded(out["schedule_relationship"], "scheduled") != "added"]
    has_stop = out["stop_id"].fillna("").astype(str).str.strip().isin(["", BLANK_LABEL])
    if has_stop.any():
        logging.warning(
            "The stop flags leave out %d visit(s) with no stop_id.", int(has_stop.sum())
        )
        out = out.loc[~has_stop]
    dev = (out["_actual_ts"] - out["_sched_ts"]) / pd.Timedelta(minutes=1)
    return out.assign(dev_min=dev)


def count_pattern_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """Count the in-service performed trips per (``pattern_key``, ``route_id``)."""
    return (
        trips.drop_duplicates(TRIP_KEY)
        .groupby(["pattern_key", "route_id"])
        .size()
        .rename("n_trips")
        .reset_index()
    )


def build_expected_trips(candidates: pd.DataFrame, pattern_trips: pd.DataFrame) -> pd.DataFrame:
    """Expected trip-visits per (stop, route): the trips of every pattern serving the stop.

    A pattern serves a stop if any of its trips ever emitted a candidate visit
    there; ``trips_performed`` lists every in-service trip, even those whose AVL
    never reported, so whole-trip AVL gaps count against the stop.

    Args:
        candidates: Output of :func:`stop_flag_candidates` with ``pattern_key``.
        pattern_trips: Output of :func:`count_pattern_trips`.

    Returns:
        One row per (``stop_id``, ``route_id``) with ``expected_trips``.
    """
    membership = candidates[["pattern_key", "route_id", "stop_id"]].drop_duplicates()
    merged = membership.merge(pattern_trips, on=["pattern_key", "route_id"], how="left")
    merged["n_trips"] = merged["n_trips"].fillna(0).astype(int)
    return (
        merged.groupby(["stop_id", "route_id"]).agg(expected_trips=("n_trips", "sum")).reset_index()
    )


def summarize_candidate_visits(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per (stop, route) visit diagnostics split by failure cause.

    Args:
        candidates: Output of :func:`stop_flag_candidates`.

    Returns:
        ``visits_emitted``, ``visits_skipped``, ``visits_missing_actual`` (an AVL
        dropout) and ``visits_missing_schedule`` (an export defect; rows missing both
        count here) per (``stop_id``, ``route_id``).
    """
    if "schedule_relationship" in candidates.columns:
        status = _casefolded(candidates["schedule_relationship"], "scheduled")
    else:
        status = pd.Series("scheduled", index=candidates.index)
    unscored = status.eq("scheduled") & candidates["dev_min"].isna()
    work = candidates.assign(
        _skipped=status.eq("skipped"),
        _missing_schedule=unscored & candidates["_sched_ts"].isna(),
        _missing_actual=unscored & candidates["_sched_ts"].notna(),
    )
    return (
        work.groupby(["stop_id", "route_id"])
        .agg(
            visits_emitted=("stop_id", "size"),
            visits_skipped=("_skipped", "sum"),
            visits_missing_actual=("_missing_actual", "sum"),
            visits_missing_schedule=("_missing_schedule", "sum"),
        )
        .reset_index()
    )


def build_stop_route_detail(
    candidates: pd.DataFrame, scored: pd.DataFrame, trips: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the per-(stop, route) coverage and OTP evidence behind the stop flags.

    Args:
        candidates: Output of :func:`stop_flag_candidates` with ``pattern_key``.
        scored: Scored visits (one row per visit).
        trips: In-service trips with ``pattern_key`` on the dates and routes analyzed.

    Returns:
        One row per (``stop_id``, ``route_id``) with expected and observed trips,
        ``pct_trips_observed``, the visit diagnostics, OTP counts and ``pct_on_time``.
    """
    keys = ["stop_id", "route_id"]
    expected = build_expected_trips(candidates, count_pattern_trips(trips))
    work = scored.assign(
        _early=scored["otp_class"].eq("early"),
        _on_time=scored["otp_class"].eq("on_time"),
        _late=scored["otp_class"].eq("late"),
    )
    observed = (
        work.drop_duplicates([*keys, *TRIP_KEY]).groupby(keys).size().rename("observed_trips")
    )
    otp = work.groupby(keys).agg(
        evaluated=("otp_class", "size"),
        early=("_early", "sum"),
        on_time=("_on_time", "sum"),
        late=("_late", "sum"),
    )
    otp = otp.join(observed).reset_index()
    detail = expected.merge(summarize_candidate_visits(candidates), on=keys, how="left").merge(
        otp, on=keys, how="left"
    )
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
    detail["pct_trips_observed"] = _pct(detail["observed_trips"], detail["expected_trips"])
    detail["pct_on_time"] = _pct(detail["on_time"], detail["evaluated"])
    return detail.sort_values(keys, ignore_index=True)


def attach_route_baselines(detail: pd.DataFrame) -> pd.DataFrame:
    """Add each route's pooled coverage and OTP and every stop's gap from them.

    Args:
        detail: Output of :func:`build_stop_route_detail`.

    Returns:
        A copy with ``route_pct_trips_observed``, ``route_pct_on_time``,
        ``coverage_gap`` and ``otp_gap`` (negative = the stop lags its route).
    """
    out = detail.copy()
    grp = out.groupby("route_id")
    out["route_pct_trips_observed"] = _pct(
        grp["observed_trips"].transform("sum"), grp["expected_trips"].transform("sum")
    )
    out["route_pct_on_time"] = _pct(
        grp["on_time"].transform("sum"), grp["evaluated"].transform("sum")
    )
    out["coverage_gap"] = (out["pct_trips_observed"] - out["route_pct_trips_observed"]).round(1)
    out["otp_gap"] = (out["pct_on_time"] - out["route_pct_on_time"]).round(1)
    return out


def evaluate_route_level_flags(
    detail: pd.DataFrame,
    min_expected_trips: int = MIN_EXPECTED_TRIPS,
    min_scored_visits: int = MIN_SCORED_VISITS,
    coverage_gap_flag_pct: float = COVERAGE_GAP_FLAG_PCT,
    coverage_abs_flag_pct: float = COVERAGE_ABS_FLAG_PCT,
    otp_gap_flag_pct: float = OTP_GAP_FLAG_PCT,
    otp_abs_flag_pct: float = OTP_ABS_FLAG_PCT,
) -> pd.DataFrame:
    """Judge each (stop, route) cell against the coverage and OTP rules.

    A cell is judged only with enough data; a judged cell is bad when it is both far
    below its route's baseline and below the absolute floor.

    Args:
        detail: Output of :func:`attach_route_baselines`.
        min_expected_trips: Expected trips needed to judge coverage.
        min_scored_visits: Scored visits needed to judge OTP.
        coverage_gap_flag_pct: Points below the route's coverage that count as low.
        coverage_abs_flag_pct: Coverage floor (percent).
        otp_gap_flag_pct: Points below the route's OTP that count as poor.
        otp_abs_flag_pct: OTP floor (percent).

    Returns:
        A copy with ``coverage_evaluable``, ``low_coverage_route``, ``otp_evaluable``
        and ``poor_otp_route``.
    """
    out = detail.copy()
    out["coverage_evaluable"] = out["expected_trips"] >= min_expected_trips
    out["low_coverage_route"] = (
        out["coverage_evaluable"]
        & (out["coverage_gap"] <= -coverage_gap_flag_pct)
        & (out["pct_trips_observed"] < coverage_abs_flag_pct)
    )
    out["otp_evaluable"] = out["evaluated"] >= min_scored_visits
    out["poor_otp_route"] = (
        out["otp_evaluable"]
        & (out["otp_gap"] <= -otp_gap_flag_pct)
        & (out["pct_on_time"] < otp_abs_flag_pct)
    )
    return out


def build_stop_summary(
    detail: pd.DataFrame, min_routes_flagged: int = MIN_ROUTES_FLAGGED
) -> pd.DataFrame:
    """Roll the per-route verdicts up to one row per stop, flagged stops first.

    Args:
        detail: Output of :func:`evaluate_route_level_flags`.
        min_routes_flagged: Routes that must each judge the stop bad before it is flagged.

    Returns:
        One row per ``stop_id`` with pooled coverage and OTP, the worst judged gaps,
        the visit diagnostics, route counts, the two flags and ``flag_reason``.
    """
    df = detail.assign(
        _cov_gap=detail["coverage_gap"].where(detail["coverage_evaluable"]),
        _otp_gap=detail["otp_gap"].where(detail["otp_evaluable"]),
    )
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
            worst_coverage_gap=("_cov_gap", "min"),
            worst_otp_gap=("_otp_gap", "min"),
        )
        .reset_index()
    )
    summary["pct_trips_observed"] = _pct(summary["observed_trips"], summary["expected_trips"])
    summary["pct_on_time"] = _pct(summary["on_time"], summary["evaluated"])
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
    summary["_rank"] = summary["flag_low_coverage"].astype(int) + summary["flag_poor_otp"].astype(
        int
    )
    summary["_severity"] = summary[["worst_coverage_gap", "worst_otp_gap"]].min(axis=1)
    summary = summary.sort_values(
        ["_rank", "_severity", "stop_id"], ascending=[False, True, True], ignore_index=True
    )
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


def build_stop_flags(
    visits: pd.DataFrame,
    scored: pd.DataFrame,
    trips: pd.DataFrame,
    settings: RunSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``otp_stop_flags.csv`` and its per-(stop, route) detail over the whole window.

    Args:
        visits: Stop visits analyzed (one row per visit) with parsed timestamps.
        scored: The same visits' scored subset.
        trips: In-service trips on the dates and routes analyzed.
        settings: The run settings (thresholds).

    Returns:
        ``(stop summary, stop x route detail)``.
    """
    candidates = add_pattern_key(stop_flag_candidates(visits, settings.timepoints_only))
    detail = build_stop_route_detail(candidates, scored, add_pattern_key(trips))
    detail = evaluate_route_level_flags(
        attach_route_baselines(detail),
        settings.min_expected_trips,
        settings.min_scored_visits,
        settings.coverage_gap_flag_pct,
        settings.coverage_abs_flag_pct,
        settings.otp_gap_flag_pct,
        settings.otp_abs_flag_pct,
    )
    summary = build_stop_summary(detail, settings.min_routes_flagged)
    flagged = summary.loc[summary["flag_reason"] != ""]
    if not flagged.empty:
        logging.warning(
            "%d stop(s) are flagged for low coverage or poor OTP on %d+ routes; see %s.",
            len(flagged),
            settings.min_routes_flagged,
            STOP_FLAGS_FILENAME,
        )
    return summary, detail


# =============================================================================
# RUNNING TIME
# =============================================================================


def percentile_column(pct: float) -> str:
    """Return the column name for a percentile, e.g. 85 -> ``actual_p85_min``."""
    label = f"{pct:g}".replace(".", "_")
    if float(pct) < 10 and "_" not in label:
        label = f"0{label}"
    return f"actual_p{label}_min"


def _quantile(pct: float) -> Any:
    """Return an aggregation callable for the given percentile (0-100)."""

    def agg(series: pd.Series) -> float:
        return float(series.quantile(pct / 100.0)) if series.notna().any() else float("nan")

    return agg


def _mode_or_median(series: pd.Series) -> float:
    """Return the modal value of a numeric series, breaking ties by the median."""
    values = series.dropna()
    if values.empty:
        return float("nan")
    modes = values.mode()
    if len(modes) == 1:
        return float(modes.iloc[0])
    return float(values.median())


def compute_trip_runtimes(visits: pd.DataFrame) -> pd.DataFrame:
    """Derive each performed trip's actual and scheduled running time.

    The actual running time runs from the first actual departure to the last
    actual arrival among the trip's served (not Skipped) stops; the scheduled one
    covers the same two stops (scheduled departure, else arrival, at the first;
    scheduled arrival, else departure, at the last). A trip whose arrival stop does
    not come after its departure stop, or arrives no later than it departs, has no
    running time (a single observed stop would read as zero) and is counted in a
    warning.

    Args:
        visits: Stop visits with :func:`add_parsed_timestamps` columns.

    Returns:
        One row per ``TRIP_KEY`` with ``actual_runtime_min``,
        ``scheduled_runtime_min`` and ``deviation_min`` (actual minus scheduled).
    """
    columns = [*TRIP_KEY, "actual_runtime_min", "scheduled_runtime_min", "deviation_min"]
    work = _sorted_by_sequence(visits.loc[_not_skipped(visits)])
    work = work.assign(_pos=work.groupby(TRIP_KEY, sort=False).cumcount())
    dep = work["actual_departure_time_ts"].notna()
    arr = work["actual_arrival_time_ts"].notna()
    first = work.loc[dep].groupby(TRIP_KEY, sort=False).head(1)
    last = work.loc[arr].groupby(TRIP_KEY, sort=False).tail(1)
    first = pd.DataFrame(
        {
            "service_date": first["service_date"],
            "trip_id_performed": first["trip_id_performed"],
            "_start_pos": first["_pos"],
            "_start": first["actual_departure_time_ts"],
            "_sched_start": first["schedule_departure_time_ts"].fillna(
                first["schedule_arrival_time_ts"]
            ),
        }
    )
    last = pd.DataFrame(
        {
            "service_date": last["service_date"],
            "trip_id_performed": last["trip_id_performed"],
            "_end_pos": last["_pos"],
            "_end": last["actual_arrival_time_ts"],
            "_sched_end": last["schedule_arrival_time_ts"].fillna(
                last["schedule_departure_time_ts"]
            ),
        }
    )
    trips = work[TRIP_KEY].drop_duplicates()
    ends = first.merge(last, on=TRIP_KEY, how="inner")
    usable = (ends["_end_pos"] > ends["_start_pos"]) & (ends["_end"] > ends["_start"])
    n_unusable = len(trips) - int(usable.sum())
    if n_unusable:
        logging.warning(
            "%d performed trip(s) lack a departure followed by a later stop's arrival (e.g. "
            "only one observed stop); no running time is recorded for them.",
            n_unusable,
        )
    ends = ends.loc[usable]
    if ends.empty:
        return pd.DataFrame(columns=columns)
    out = ends[TRIP_KEY].copy()
    out["actual_runtime_min"] = (ends["_end"] - ends["_start"]) / pd.Timedelta(minutes=1)
    out["scheduled_runtime_min"] = (ends["_sched_end"] - ends["_sched_start"]) / pd.Timedelta(
        minutes=1
    )
    out["deviation_min"] = out["actual_runtime_min"] - out["scheduled_runtime_min"]
    return out[columns].reset_index(drop=True)


def trim_outliers(
    rows: pd.DataFrame,
    frac: float = TRIM_FRAC,
    group_col: str = "trip_id",
    value_col: str = "actual_runtime_min",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations into retained rows and trimmed outliers.

    Within each *group_col* group, observations below the *frac* quantile or above
    the ``1 - frac`` quantile of *value_col* are outliers.

    Args:
        rows: Per-trip-per-date observations.
        frac: Tail fraction trimmed from each end (0 disables trimming).
        group_col: Column identifying each recurring trip.
        value_col: Numeric column to trim on.

    Returns:
        ``(retained, outliers)``; ``outliers`` is empty when *frac* is 0.
    """
    if frac <= 0 or rows.empty:
        return rows.copy(), rows.iloc[0:0].copy()
    grouped = rows.groupby(group_col, sort=False)[value_col]
    count = grouped.transform("count")
    lo = grouped.transform(lambda s: s.quantile(frac))
    hi = grouped.transform(lambda s: s.quantile(1 - frac))
    out = (count >= 2) & ((rows[value_col] < lo) | (rows[value_col] > hi))
    return rows.loc[~out].copy(), rows.loc[out].copy()


def summarize_runtime(
    rows: pd.DataFrame,
    level: str,
    cut: str,
    periods: Sequence[TimePeriod],
    day_order: Sequence[str],
    percentiles: Sequence[float] = RUNTIME_PERCENTILES,
) -> pd.DataFrame:
    """Aggregate running-time observations to one table (one level by one time cut).

    Used for whole trips and for segments alike.

    Args:
        rows: Day-typed observations with the time-cut columns, the level's keys,
            ``actual_runtime_min``, ``scheduled_runtime_min`` and ``deviation_min``
            (segments add ``segment``, ``seq`` and ``recovery_after_min``).
        level: A key of :data:`LEVELS`; its required columns must be present.
        cut: A key of :data:`TIME_CUTS`.
        periods: The checked time periods (their order sets the row order).
        day_order: Output order of the day types.
        percentiles: Percentiles of the actual running time to report.

    Returns:
        The table: keys, then ``n_obs``, ``days_observed``, the actual mean, median,
        standard deviation, coefficient of variation, minimum, maximum and
        percentiles, the modal ``scheduled_min``, ``diff_min`` (median actual minus
        scheduled) and ``mean_deviation_min``.
    """
    level_cols = level_columns(level, rows.columns) or []
    date_keys, time_keys = _cut_keys(cut)
    keys = [*date_keys, *level_cols, *time_keys]
    work = _rows_with_time(rows.loc[rows["actual_runtime_min"].notna()], time_keys)
    pct_cols = [percentile_column(p) for p in percentiles]
    specs: Dict[str, Tuple[str, Any]] = {
        "n_obs": ("actual_runtime_min", "size"),
        "days_observed": ("service_date", "nunique"),
        "actual_mean_min": ("actual_runtime_min", "mean"),
        "actual_median_min": ("actual_runtime_min", "median"),
        "actual_std_min": ("actual_runtime_min", "std"),
        "actual_min_min": ("actual_runtime_min", "min"),
        "actual_max_min": ("actual_runtime_min", "max"),
        **{col: ("actual_runtime_min", _quantile(p)) for col, p in zip(pct_cols, percentiles)},
        "scheduled_min": ("scheduled_runtime_min", _mode_or_median),
        "mean_deviation_min": ("deviation_min", "mean"),
    }
    lead: List[str] = []
    if level == "segment":
        specs = {"segment": ("segment", "first"), "seq": ("seq", "min"), **specs}
        specs["recovery_after_min"] = ("recovery_after_min", "median")
        lead = ["segment", "seq"]
    if level == "trip" and "_start_min" in work.columns:
        specs["_start"] = ("_start_min", "median")
    agg = work.groupby(keys, sort=False, dropna=False).agg(**specs).reset_index()
    agg["n_obs"] = agg["n_obs"].astype(int)
    agg["days_observed"] = agg["days_observed"].astype(int)
    agg["actual_cv"] = (
        agg["actual_std_min"] / agg["actual_mean_min"].where(agg["actual_mean_min"] > 0)
    ).round(3)
    agg["diff_min"] = agg["actual_median_min"] - agg["scheduled_min"]
    agg, time_cols = _add_time_columns(agg, time_keys, periods)
    sort_cols: List[str] = keys
    if level == "segment":
        lead_keys = [c for c in level_cols if c in ("route_id", "direction_id")]
        sort_cols = [*date_keys, *lead_keys, "seq", "from_stop_id", "to_stop_id", *time_keys]
    if "_start" in agg.columns:
        agg["scheduled_start"] = _start_label(agg["_start"])
        lead = ["scheduled_start"]
        sort_cols = [*_trip_sort_cols(date_keys, level_cols), *time_keys]
    measures = [
        "n_obs",
        "days_observed",
        "actual_mean_min",
        "actual_median_min",
        "actual_std_min",
        "actual_cv",
        "actual_min_min",
        "actual_max_min",
        *pct_cols,
        "scheduled_min",
        "diff_min",
        "mean_deviation_min",
    ]
    if level == "segment":
        measures.append("recovery_after_min")
    rounded = [m for m in measures if m.endswith("_min")]
    agg[rounded] = agg[rounded].astype(float).round(2)
    table = _sort_table(agg, sort_cols, day_order, periods)
    return table[[*date_keys, *level_cols, *lead, *time_cols, *measures]]


def build_trip_flags(
    rows: pd.DataFrame,
    high_cv_threshold: float = HIGH_CV_THRESHOLD,
    min_obs_for_cv: int = MIN_OBS_FOR_CV,
    gap_frac: float = GAP_FRAC,
    day_order: Sequence[str] = (),
) -> pd.DataFrame:
    """Flag trips with highly variable running times or far fewer observations than their peers.

    Args:
        rows: Day-typed, trimmed trip observations.
        high_cv_threshold: Coefficient of variation above which a trip is ``high_variation``.
        min_obs_for_cv: Observations needed before the variation flag is trusted.
        gap_frac: Share of the day type's median observation count below which a trip
            is a ``data_gap``.
        day_order: Output order of the day types.

    Returns:
        One row per (day type, trip) with its statistics and the two flags.
    """
    keys = ["day_type", *[c for c in ("route_id", "direction_id") if c in rows.columns], "trip_id"]
    columns = [*keys, "scheduled_start", "n_obs", "runtime_mean_min", "runtime_median_min"]
    columns += ["runtime_std_min", "cv", "high_variation", "data_gap"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    stats = (
        rows.groupby(keys, sort=False, dropna=False)
        .agg(
            _start=("_start_min", "median"),
            n_obs=("actual_runtime_min", "count"),
            runtime_mean_min=("actual_runtime_min", "mean"),
            runtime_median_min=("actual_runtime_min", "median"),
            runtime_std_min=("actual_runtime_min", "std"),
        )
        .reset_index()
    )
    stats["cv"] = (
        stats["runtime_std_min"] / stats["runtime_mean_min"].where(stats["runtime_mean_min"] > 0)
    ).round(3)
    stats["high_variation"] = (stats["cv"] > high_cv_threshold) & (stats["n_obs"] >= min_obs_for_cv)
    median_obs = stats.groupby("day_type")["n_obs"].transform("median")
    stats["data_gap"] = stats["n_obs"] < median_obs * gap_frac
    stats["scheduled_start"] = _start_label(stats["_start"])
    for col in ("runtime_mean_min", "runtime_median_min", "runtime_std_min"):
        stats[col] = stats[col].round(2)
    return _sort_table(stats, [*keys[:-1], "_start", "trip_id"], day_order)[columns]


def build_dow_flags(
    rows: pd.DataFrame,
    low_count_frac: float = DOW_LOW_COUNT_FRAC,
    runtime_pct: float = DOW_RUNTIME_PCT,
    day_order: Sequence[str] = (),
) -> pd.DataFrame:
    """Flag a trip's weekdays that are thinly observed or run unusually long or short.

    Args:
        rows: Day-typed, trimmed trip observations.
        low_count_frac: Share of the trip's median per-weekday count below which a
            weekday is ``low_count``.
        runtime_pct: Relative difference from the trip's mean above which a weekday's
            mean is a ``runtime_anomaly``.
        day_order: Output order of the day types.

    Returns:
        One row per (day type, trip, ``day_of_week``) with ``n_obs``,
        ``dow_mean_min``, ``trip_mean_min``, ``pct_diff`` and the two flags.
    """
    keys = ["day_type", *[c for c in ("route_id", "direction_id") if c in rows.columns], "trip_id"]
    columns = [*keys, "day_of_week", "n_obs", "dow_mean_min", "trip_mean_min", "pct_diff"]
    columns += ["low_count", "runtime_anomaly"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    work = rows.assign(
        _trip_mean=rows.groupby(keys, dropna=False)["actual_runtime_min"].transform("mean")
    )
    out = (
        work.groupby([*keys, "day_of_week"], sort=False, dropna=False)
        .agg(
            n_obs=("actual_runtime_min", "count"),
            dow_mean_min=("actual_runtime_min", "mean"),
            trip_mean_min=("_trip_mean", "first"),
            _start=("_start_min", "median"),
        )
        .reset_index()
    )
    median_count = out.groupby(keys, dropna=False)["n_obs"].transform("median")
    out["low_count"] = out["n_obs"] < median_count * low_count_frac
    out["pct_diff"] = (
        (out["dow_mean_min"] - out["trip_mean_min"])
        / out["trip_mean_min"].where(out["trip_mean_min"] > 0)
    ).round(3)
    out["runtime_anomaly"] = out["pct_diff"].abs() > runtime_pct
    out["dow_mean_min"] = out["dow_mean_min"].round(2)
    out["trip_mean_min"] = out["trip_mean_min"].round(2)
    return _sort_table(out, [*keys[:-1], "_start", "trip_id", "day_of_week"], day_order)[columns]


def compute_block_recovery(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Return the scheduled recovery after each trip before the next trip on its block.

    Trips are chained per (``service_date``, ``block_id``) in scheduled-start order,
    non-revenue trips included (a pull-in or deadhead is the next movement); Canceled
    trips are left out. A block's last trip has no recovery.

    Args:
        trips_performed: The TIDES ``trips_performed`` table as read.

    Returns:
        ``TRIP_KEY`` columns and ``recovery_after_min``, one row per performed trip;
        empty when the block or schedule columns are missing.
    """
    columns = [*TRIP_KEY, "recovery_after_min"]
    needed = ["block_id", "schedule_trip_start", "schedule_trip_end"]
    if any(c not in trips_performed.columns for c in needed):
        logging.info(
            "trips_performed has no %s; segment tables leave recovery_after_min blank.",
            " / ".join(c for c in needed if c not in trips_performed.columns),
        )
        return pd.DataFrame(columns=columns)
    trips = trips_performed.assign(service_date=_parse_dates(trips_performed["service_date"]))
    if "schedule_relationship" in trips.columns:
        status = _casefolded(trips["schedule_relationship"], "scheduled")
        trips = trips.loc[~status.isin(["canceled", "cancelled"])]
    trips = trips.assign(
        _start=parse_timestamps(trips["schedule_trip_start"]),
        _end=parse_timestamps(trips["schedule_trip_end"]),
        block_id=trips["block_id"].fillna("").astype(str).str.strip(),
    )
    trips = trips.loc[trips["_start"].notna() & trips["_end"].notna() & (trips["block_id"] != "")]
    trips = trips.drop_duplicates(TRIP_KEY).sort_values(["service_date", "block_id", "_start"])
    next_start = trips.groupby(["service_date", "block_id"])["_start"].shift(-1)
    out = trips[TRIP_KEY].copy()
    out["recovery_after_min"] = (next_start - trips["_end"]) / pd.Timedelta(minutes=1)
    return out[columns].reset_index(drop=True)


def build_segments(
    visits: pd.DataFrame, timepoints_only: bool = SEGMENT_TIMEPOINTS_ONLY
) -> pd.DataFrame:
    """Build one row per traversal of a segment between consecutive (timepoint) stops.

    Skipped visits are left out first, so a segment bridges a skipped timepoint to
    the next served one. The actual running time is the downstream actual arrival
    minus the upstream actual departure; the scheduled one is the same on the
    scheduled times.

    Args:
        visits: Stop visits with :func:`add_parsed_timestamps` columns.
        timepoints_only: Anchor segments at timepoints only.

    Returns:
        One row per segment traversal with ``from_stop_id``, ``to_stop_id``,
        ``segment`` (``"A -> B"``), ``seq`` (its place along the trip),
        ``actual_runtime_min``, ``scheduled_runtime_min``, ``deviation_min``, the
        trip attributes, and the upstream stop's scheduled and actual time as text
        (``_sched_text`` / ``_actual_text``) for the time basis.
    """
    work = visits.loc[_not_skipped(visits)]
    if timepoints_only:
        if "timepoint" in work.columns:
            work = work.loc[is_true(work["timepoint"])]
        else:
            logging.warning("stop_visits has no timepoint column; segments join every served stop.")
    work = _sorted_by_sequence(work)
    nxt = work.groupby(TRIP_KEY, sort=False)[
        ["stop_id", "actual_arrival_time_ts", "schedule_arrival_time_ts"]
    ].shift(-1)
    pos = work.groupby(TRIP_KEY, sort=False).cumcount() + 1
    has_next = nxt["stop_id"].notna()
    attrs = [
        c
        for c in ("route_id", "direction_id", "route_type_agency", "block_id", "trip_id")
        if c in work.columns
    ]
    up = work.loc[has_next]
    dn = nxt.loc[has_next]
    out = up[[*TRIP_KEY, *attrs, "_sched_text", "_actual_text"]].copy()
    out["from_stop_id"] = up["stop_id"].astype(str)
    out["to_stop_id"] = dn["stop_id"].astype(str)
    out["segment"] = out["from_stop_id"] + " -> " + out["to_stop_id"]
    out["seq"] = pos.loc[has_next].astype(int)
    out["actual_runtime_min"] = (
        dn["actual_arrival_time_ts"] - up["actual_departure_time_ts"]
    ) / pd.Timedelta(minutes=1)
    out["scheduled_runtime_min"] = (
        dn["schedule_arrival_time_ts"] - up["schedule_departure_time_ts"]
    ) / pd.Timedelta(minutes=1)
    out["deviation_min"] = out["actual_runtime_min"] - out["scheduled_runtime_min"]
    return out.reset_index(drop=True)


def build_segment_route_tables(
    table: pd.DataFrame, day_order: Sequence[str]
) -> Dict[str, pd.DataFrame]:
    """Split the all-day segment table into one table per route, direction and day type.

    Args:
        table: The segment x all-day running-time table.
        day_order: Output order of the day types.

    Returns:
        ``{file stem: table}`` with the segments of each group in stop order.
    """
    out: Dict[str, pd.DataFrame] = {}
    if table.empty:
        return out
    group_cols = ["day_type", *[c for c in ("route_id", "direction_id") if c in table.columns]]
    for keys, sub in table.groupby(group_cols, sort=False):
        values = keys if isinstance(keys, tuple) else (keys,)
        parts = dict(zip(group_cols, values))
        stem = f"segment_runtime_{_sanitize_token(parts.get('route_id', 'all'))}"
        if "direction_id" in parts:
            stem += f"_dir{_sanitize_token(parts['direction_id'])}"
        stem += f"_{_sanitize_token(parts['day_type']).lower()}"
        out[stem] = sub.sort_values(["seq", "from_stop_id", "to_stop_id"], kind="stable").drop(
            columns=group_cols
        )
    return out


# =============================================================================
# FILTERING
# =============================================================================


def filter_routes(
    df: pd.DataFrame,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    warn: bool = True,
) -> pd.DataFrame:
    """Keep or drop routes by ``route_id`` (compared as stripped text).

    Filter entries that match no route in the data are named in a warning.

    Args:
        df: Rows with ``route_id``.
        include: Route IDs to keep; empty keeps all.
        exclude: Route IDs to drop.
        warn: Whether to warn about filter entries that match no route.

    Returns:
        The filtered rows.

    Raises:
        InputDataError: If no rows remain.
    """
    if not include and not exclude:
        return df
    route = df["route_id"].fillna("").astype(str).str.strip()
    known = set(route)
    wanted = {str(r).strip() for r in include}
    unwanted = {str(r).strip() for r in exclude}
    if warn:
        for label, values in (("ROUTES_TO_INCLUDE", wanted), ("ROUTES_TO_EXCLUDE", unwanted)):
            absent = sorted(values - known)
            if absent:
                logging.warning("%s entries match no route_id: %s", label, ", ".join(absent))
    keep = pd.Series(True, index=df.index)
    if wanted:
        keep &= route.isin(wanted)
    if unwanted:
        keep &= ~route.isin(unwanted)
    out = df.loc[keep]
    if out.empty:
        raise InputDataError(
            "No stop visits are left after the route filters; check ROUTES_TO_INCLUDE / "
            "ROUTES_TO_EXCLUDE."
        )
    return out


def label_blank_groups(
    df: pd.DataFrame, group_cols: Sequence[str], warn: bool = True
) -> pd.DataFrame:
    """Replace blank grouping values with ``"(blank)"`` so they form their own group.

    Args:
        df: Rows to label.
        group_cols: The grouping columns (any not in *df* are ignored).
        warn: Whether to warn about the blanks found.

    Returns:
        A copy of *df* with the grouping columns as stripped text.
    """
    out = df.copy()
    for col in group_cols:
        if col not in out.columns:
            continue
        text = out[col].fillna("").astype(str).str.strip()
        blank = text == ""
        if blank.any() and warn:
            logging.warning(
                "%d row(s) have a blank %s; they are reported as %s.",
                int(blank.sum()),
                col,
                BLANK_LABEL,
            )
        out[col] = text.where(~blank, BLANK_LABEL)
    return out


# =============================================================================
# CHARTS
# =============================================================================


def _sanitize_token(name: object) -> str:
    """Reduce a label to a filesystem-safe token."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip()).strip("_")
    return token or "unnamed"


def _style_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply the shared chart styling (light grid, no top/right spines)."""
    ax.tick_params(axis="y", labelsize=8, colors=_INK_COLOR)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK_COLOR)
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID_COLOR)


def draw_bar_chart(
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    out_path: Path,
    reference: Optional[Tuple[str, float]] = None,
) -> None:
    """Draw one bar chart PNG with one or two series side by side.

    Args:
        labels: Category labels along the x axis.
        series: ``(name, values)`` pairs, one value per label (NaN draws no bar); a
            legend is drawn when there is more than one series or a reference line.
        title: Chart title.
        ylabel: Y-axis label.
        out_path: PNG destination.
        reference: Optional ``(name, value)`` drawn as a dashed horizontal line.
    """
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    x = np.arange(len(labels))
    width = 0.72 / max(len(series), 1)
    for i, (name, values) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.bar(x + offset, list(values), width, color=color, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(labels), rotation=45, ha="right", fontsize=7, color=_INK_COLOR)
    _style_axes(ax, title, ylabel)
    if reference is not None:
        ax.axhline(
            reference[1], linestyle="--", linewidth=1.2, color=_FLAG_COLOR, label=reference[0]
        )
    if len(series) > 1 or reference is not None:
        # Above the plot, beside the title, so the legend never covers a bar.
        ax.legend(
            fontsize=7,
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=len(series) + (reference is not None),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_summary_charts(
    tables: Mapping[Tuple[str, str], pd.DataFrame],
    chart_levels: Sequence[str],
    series_columns: Sequence[Tuple[str, str]],
    ylabel: str,
    out_dir: Path,
    periods: Sequence[TimePeriod],
    reference: Optional[Tuple[str, float]] = None,
) -> int:
    """Draw one chart per group and day type for each chartable table.

    Charts cover the levels in *chart_levels* crossed with the time period, hour,
    month and day-of-week cuts (whichever were exported). Hours run continuously
    across a table's range so charts of one table line up; a period, hour, month or
    weekday with no row draws no bar.

    Args:
        tables: ``{(level, cut): table}``.
        chart_levels: Levels to chart.
        series_columns: ``(legend name, column)`` pairs to draw.
        ylabel: Y-axis label.
        out_dir: Folder for this family of charts.
        periods: The checked time periods (x order for time-period charts).
        reference: Optional ``(name, value)`` drawn as a dashed line on every chart.

    Returns:
        Number of PNGs written.
    """
    written = 0
    for (level, cut), table in tables.items():
        if level not in chart_levels or cut not in CHART_CUTS or table.empty:
            continue
        x_field = TIME_CUTS[cut][0]
        if x_field == "time_period":
            categories: List[object] = [p.name for p in periods]
        elif x_field == "hour":
            categories = list(range(int(table["hour"].min()), int(table["hour"].max()) + 1))
        elif x_field == "day_of_week":
            present = set(table["day_of_week"])
            categories = [d for d in WEEKDAY_NAMES if d in present]
        else:
            categories = sorted(table[x_field].dropna().unique().tolist())
        if x_field == "hour":
            labels = [minutes_to_hhmm(int(str(h)) * 60) for h in categories]
        else:
            labels = [str(c) for c in categories]
        level_cols = level_columns(level, table.columns) or []
        folder = out_dir / f"{level}_by_{cut}"
        folder.mkdir(parents=True, exist_ok=True)
        for keys, sub in table.groupby(["day_type", *level_cols], sort=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            day_type, group_values = key_tuple[0], key_tuple[1:]
            label, token = _group_label(level, group_values)
            series = [
                (
                    name,
                    sub.groupby(x_field)[column]
                    .first()
                    .reindex(categories)
                    .to_numpy(dtype=float)
                    .tolist(),
                )
                for name, column in series_columns
            ]
            out_path = folder / f"{_sanitize_token(token)}_{_sanitize_token(day_type)}.png"
            title = f"{label} — {day_type}, by {x_field.replace('_', ' ')}"
            draw_bar_chart(labels, series, title, ylabel, out_path, reference)
            written += 1
    return written


def _route_direction_groups(
    rows: pd.DataFrame,
) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    """Yield ``(title label, file stem, rows)`` per day type, route and direction."""
    group_cols = ["day_type", *[c for c in ("route_id", "direction_id") if c in rows.columns]]
    for keys, sub in rows.groupby(group_cols, sort=False):
        values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        route = values.get("route_id", "all routes")
        label = f"Route {route}"
        stem = f"route_{_sanitize_token(route)}"
        direction = values.get("direction_id")
        if direction is not None and str(direction) not in ("", BLANK_LABEL):
            label = f"{label} direction {direction}"
            stem = f"{stem}_dir{_sanitize_token(direction)}"
        day_type = str(values["day_type"])
        yield f"{label} — {day_type}", f"{stem}_{_sanitize_token(day_type)}", sub


def export_trip_otp_charts(
    trip_table: pd.DataFrame, standard: float, out_dir: Path, max_trips: int
) -> int:
    """Write one per-trip % on-time chart per day type, route and direction.

    Bars run in scheduled-start order; trips below *standard* are highlighted and
    the standard is drawn as a dashed line. Groups with more than *max_trips* trips
    are skipped (the CSV holds every trip).

    Args:
        trip_table: The trip x all-day OTP table.
        standard: Agency OTP standard (percent).
        out_dir: Charts folder (created if needed).
        max_trips: Largest number of bars drawn on one chart.

    Returns:
        Number of PNGs written.
    """
    if trip_table.empty:
        return 0
    rows = trip_table.copy()
    labels = rows["scheduled_start"].fillna("").astype(str).str.strip()
    rows["_label"] = labels.where(labels != "", rows["trip_id"].astype(str))
    written = 0
    for title, stem, sub in _route_direction_groups(rows):
        if len(sub) > max_trips:
            logging.info(
                "Skipped the per-trip OTP chart for %s: %d trips is more than "
                "MAX_TRIPS_PER_CHART (%d).",
                title,
                len(sub),
                max_trips,
            )
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        draw_trip_chart(
            sub.sort_values(["_label", "trip_id"]),
            f"{title} % on time by trip",
            "pct_on_time",
            "% on time",
            standard,
            out_dir / f"{stem}.png",
            100.0,
        )
        written += 1
    return written


def export_runtime_boxplots(rows: pd.DataFrame, out_dir: Path, max_trips: int) -> int:
    """Write one running-time box plot per day type, route and direction.

    Each box pools one scheduled trip's observed running times; boxes run in
    scheduled-start order. Groups with more than *max_trips* trips are skipped.

    Args:
        rows: Day-typed, trimmed trip observations.
        out_dir: Charts folder (created if needed).
        max_trips: Largest number of boxes drawn on one chart.

    Returns:
        Number of PNGs written.
    """
    written = 0
    for title, stem, sub in _route_direction_groups(rows):
        boxes = []
        for trip_id, trip_rows in sub.groupby("trip_id", sort=False):
            values = trip_rows["actual_runtime_min"].dropna().to_numpy(dtype=float)
            if values.size:
                start = trip_rows["_start_min"].median()
                label = minutes_to_hhmm(start) or str(trip_id)
                boxes.append((start if pd.notna(start) else np.inf, label, values))
        if not boxes:
            continue
        if len(boxes) > max_trips:
            logging.info(
                "Skipped the running-time box plot for %s: %d trips is more than "
                "MAX_TRIPS_PER_CHART (%d).",
                title,
                len(boxes),
                max_trips,
            )
            continue
        boxes.sort(key=lambda b: (b[0], b[1]))
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(8.0, len(boxes) * 0.35), 3.6))
        ax.boxplot(
            [b[2] for b in boxes],
            patch_artist=True,
            boxprops={"facecolor": _SERIES_COLORS[0], "edgecolor": _INK_COLOR, "alpha": 0.8},
            medianprops={"color": "#0b0b0b"},
            whiskerprops={"color": _INK_COLOR},
            capprops={"color": _INK_COLOR},
            flierprops={"markeredgecolor": _INK_COLOR, "markersize": 3},
        )
        ax.set_xticks(range(1, len(boxes) + 1))
        ax.set_xticklabels([b[1] for b in boxes], rotation=90, fontsize=7, color=_INK_COLOR)
        _style_axes(ax, f"{title} running time by trip", "Running time (minutes)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close(fig)
        written += 1
    return written


def _group_label(level: str, values: Sequence[object]) -> Tuple[str, str]:
    """Return a chart title fragment and filename token for one group of *level*."""
    parts = [str(v) for v in values]
    if level == "system":
        return "All routes", "system"
    if level == "service_type":
        return f"{parts[0]} service", parts[0]
    if level == "route":
        return f"Route {parts[0]}", parts[0]
    return f"Route {parts[0]} direction {parts[1]}", f"{parts[0]}_dir{parts[1]}"


def draw_trip_chart(
    sub: pd.DataFrame,
    title: str,
    column: str,
    ylabel: str,
    threshold: float,
    out_path: Path,
    y_max: Optional[float],
) -> None:
    """Draw one per-trip bar chart PNG for a single route, direction and day type.

    One bar per trip in start-time order. When *threshold* is positive it is
    drawn as a dashed line and bars below it are highlighted, so the flag is
    carried by both position and color; a legend names both.

    Args:
        sub: Per-trip rows, already sorted by start time, with a ``_label`` column.
        title: Chart title.
        column: Per-trip column to plot.
        ylabel: Y-axis label.
        threshold: Low-ridership threshold; <= 0 draws no line or highlight.
        out_path: PNG destination.
        y_max: Shared y-axis limit (None lets matplotlib autoscale).
    """
    values = sub[column].astype(float).tolist()
    colors = [_FLAG_COLOR if threshold > 0 and v < threshold else _SERIES_COLORS[0] for v in values]

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(range(len(values)), values, color=colors, width=0.72)
    # Floor the axis width so a route with few trips keeps bars proportioned.
    ax.set_xlim(-0.5, max(len(values), 12) - 0.5)
    step = max(1, len(values) // 24)  # keep x labels legible on busy routes
    ticks = list(range(0, len(values), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [sub["_label"].iloc[i] for i in ticks],
        rotation=45,
        ha="right",
        fontsize=7,
        color=_INK_COLOR,
    )
    _style_axes(ax, title, ylabel)
    if y_max is not None and y_max > 0:
        ax.set_ylim(0, y_max)
    if threshold > 0:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.2,
            color=_FLAG_COLOR,
            label=f"threshold ({threshold:g})",
        )
        if any(c == _FLAG_COLOR for c in colors):
            ax.bar(0, 0, color=_FLAG_COLOR, label="below threshold")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def write_run_log(
    log_path: Path,
    settings_lines: Sequence[str],
    periods: Sequence[TimePeriod],
    periods_source: str,
    summary_lines: Sequence[str],
) -> bool:
    """Write the run-log sidecar into the output folder.

    Args:
        log_path: Where to write the log.
        settings_lines: The settings actually used (after CLI overrides).
        periods: The time periods used.
        periods_source: Where the time periods came from.
        summary_lines: Build summary lines.

    Returns:
        True if the log was written, False otherwise.
    """
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
        RUN_LOG_TITLE,
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {log_path.parent}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "SETTINGS USED",
        "-" * 72,
        *settings_lines,
        "",
        "-" * 72,
        f"TIME PERIODS USED (from {periods_source})",
        "-" * 72,
        "# Same format as a time-periods file; copy these lines into one to reuse them.",
        *(format_period(p) for p in periods),
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
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to '%s'.", log_path)
    return True


# =============================================================================
# PIPELINE
# =============================================================================


def _write_csv(table: pd.DataFrame, path: Path, files: List[Path]) -> None:
    """Write *table* to *path* (creating its folder) and record the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    files.append(path)


def schedule_for_run(
    audit_dates: Iterable[pd.Timestamp],
    feeds: Sequence[GtfsFeed],
    trips_performed: pd.DataFrame,
    settings: RunSettings,
) -> pd.DataFrame:
    """Build the run's GTFS schedule (see :func:`schedule_coverage`) for the routes in TIDES.

    The window runs from ``START_DATE`` (else the first audited date) to
    ``END_DATE`` (else the last), so dates with no TIDES records at all are
    included. Only routes with a ``trips_performed`` row in the window (and kept
    by the route filters) are checked; GTFS routes the export lacks entirely are
    named in a warning instead of flagging every date.

    Args:
        audit_dates: Dates already audited from the TIDES data.
        feeds: Loaded GTFS feeds (empty skips the check).
        trips_performed: The TIDES ``trips_performed`` table (all rows).
        settings: The run settings (date range and route filters).

    Returns:
        The schedule rows; empty without feeds or dates.
    """
    dates = sorted(audit_dates)
    if not feeds or (not dates and (settings.start_date is None or settings.end_date is None)):
        return schedule_coverage([], feeds, {})
    start = pd.Timestamp(settings.start_date) if settings.start_date else dates[0]
    end = pd.Timestamp(settings.end_date) if settings.end_date else dates[-1]
    schedule = schedule_coverage(
        pd.date_range(start, end, freq="D"), feeds, trip_records_by_date(trips_performed)
    )
    if schedule.empty or "route_id" not in trips_performed.columns:
        return schedule
    trip_dates = _parse_dates(trips_performed["service_date"])
    in_window = (trip_dates >= start) & (trip_dates <= end)
    tides_routes = set(trips_performed.loc[in_window, "route_id"].dropna().astype(str).str.strip())
    tides_routes.discard("")
    wanted = set(schedule["route_id"])
    if settings.routes_to_include:
        wanted &= {str(r).strip() for r in settings.routes_to_include}
    wanted -= {str(r).strip() for r in settings.routes_to_exclude}
    absent = sorted(wanted - tides_routes)
    if absent:
        logging.warning(
            "%d GTFS route(s) scheduled in the date range have no TIDES records at all and "
            "are left out of the schedule checks: %s",
            len(absent),
            ", ".join(absent[:10]) + (" ..." if len(absent) > 10 else ""),
        )
    return schedule.loc[schedule["route_id"].isin(wanted & tides_routes)].reset_index(drop=True)


def in_service_trips(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Return the revenue trips of ``trips_performed``, one row per performed trip.

    Uses the same filter as :func:`join_trips`: Canceled trips and any ``trip_type``
    other than In service are dropped (compared without regard to case).

    Args:
        trips_performed: The TIDES ``trips_performed`` table as read.

    Returns:
        The kept trips with ``service_date`` parsed.
    """
    trips = trips_performed.assign(service_date=_parse_dates(trips_performed["service_date"]))
    trips = trips.loc[trips["service_date"].notna()].drop_duplicates(TRIP_KEY)
    if "schedule_relationship" in trips.columns:
        status = _casefolded(trips["schedule_relationship"], "scheduled")
        trips = trips.loc[~status.isin(["canceled", "cancelled"])]
    if "trip_type" in trips.columns:
        trips = trips.loc[_casefolded(trips["trip_type"], "in service") == "in service"]
    return trips.reset_index(drop=True)


def _observation_export(rows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Order and format observation rows for CSV (ISO service dates, rounded minutes)."""
    out = rows.copy()
    if "_start_min" in out.columns:
        out["scheduled_start"] = _start_label(out["_start_min"])
    out = out[[c for c in columns if c in out.columns]].copy()
    out["service_date"] = pd.to_datetime(out["service_date"]).dt.strftime("%Y-%m-%d")
    for col in [c for c in out.columns if c.endswith("_min")]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    if "hour" in out.columns:
        out["hour"] = out["hour"].astype("Int64")
    return out


def _measure_levels(levels: Sequence[str], measure: str) -> List[str]:
    """Return the chosen levels that apply to *measure*, in output order."""
    return [lv for lv in levels if lv in MEASURE_LEVELS[measure]]


def _write_grains(
    rows: pd.DataFrame,
    measure: str,
    levels: Sequence[str],
    cuts: Sequence[str],
    build: Any,
    out_dir: Path,
    files: List[Path],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Build and write every (level, cut) table of one measure.

    Args:
        rows: Day-typed rows for the measure.
        measure: A key of :data:`MEASURE_LEVELS` (also the file prefix).
        levels: The chosen levels.
        cuts: The chosen time cuts.
        build: ``build(rows, level, cut)`` returning one table.
        out_dir: Folder for this measure's tables.
        files: Written paths, appended to.

    Returns:
        ``{(level, cut): table}``.
    """
    tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    for level in _measure_levels(levels, measure):
        missing = missing_level_columns(level, rows.columns)
        if missing:
            logging.warning(
                "Skipping the %s level for %s: the data has no %s column.",
                level,
                measure,
                ", ".join(missing),
            )
            continue
        for cut in cuts:
            table = build(rows, level, cut)
            tables[(level, cut)] = table
            _write_csv(table, out_dir / grain_filename(measure, level, cut), files)
    logging.info("Wrote %d %s table(s) to %s.", len(tables), measure, out_dir)
    return tables


def _resolve_run_settings(
    settings: RunSettings,
) -> Tuple[List[str], List[str], List[str], Dict[str, Tuple[int, ...]], List[str]]:
    """Check the settings that ``run`` cannot take on trust.

    Args:
        settings: The run settings.

    Returns:
        ``(measures, levels, time cuts, day types, chart levels)``.

    Raises:
        TimePeriodError: If the periods do not cover the day exactly once.
        SettingsError: If any other setting is unusable.
    """
    problems = find_coverage_problems(settings.periods)
    if problems:
        raise TimePeriodError(
            _problem_report("The time periods do not cover the day exactly once:", problems)
        )
    if settings.measures is None:
        measures = resolve_flags(EXPORT_MEASURES, list(MEASURE_LEVELS), "EXPORT_MEASURES")
    else:
        measures = check_names(settings.measures, list(MEASURE_LEVELS), "measures")
    if settings.levels is None:
        levels = resolve_flags(EXPORT_LEVELS, list(LEVELS), "EXPORT_LEVELS")
    else:
        levels = check_names(settings.levels, list(LEVELS), "levels")
    if settings.time_cuts is None:
        cuts = resolve_flags(EXPORT_TIME_CUTS, list(TIME_CUTS), "EXPORT_TIME_CUTS")
    else:
        cuts = check_names(settings.time_cuts, list(TIME_CUTS), "time cuts")
    if not any(_measure_levels(levels, m) for m in measures):
        raise SettingsError(
            f"None of the levels ({', '.join(levels)}) applies to the measures "
            f"({', '.join(measures)}); nothing would be exported. Segment running time "
            "uses the segment level, and the stop levels are OTP only."
        )
    day_types = build_day_types(
        DAY_TYPES if settings.day_types is None else settings.day_types,
        settings.holiday_day_type,
    )
    unknown_charts = [lv for lv in settings.chart_levels if lv not in CHARTABLE_LEVELS]
    if unknown_charts:
        raise SettingsError(
            f"CHART_LEVELS has entries that cannot be charted: {', '.join(unknown_charts)}. "
            f"Use any of {', '.join(CHARTABLE_LEVELS)}."
        )
    if settings.trip_time_basis not in TRIP_TIME_BASES:
        raise SettingsError(
            f"TRIP_TIME_BASIS must be one of {', '.join(TRIP_TIME_BASES)}, "
            f"not {settings.trip_time_basis!r}."
        )
    if settings.stop_time_basis not in STOP_TIME_BASES:
        raise SettingsError(
            f"STOP_TIME_BASIS must be one of {', '.join(STOP_TIME_BASES)}, "
            f"not {settings.stop_time_basis!r}."
        )
    if not 0 <= settings.cutover_hour <= 23:
        raise SettingsError(f"LATE_NIGHT_CUTOVER_HOUR must be 0-23, not {settings.cutover_hour!r}.")
    if settings.early_min > settings.late_min:
        raise SettingsError(
            f"EARLY_MIN ({settings.early_min:g}) is later than LATE_MIN ({settings.late_min:g})."
        )
    if settings.last_stop_treatment not in LAST_STOP_TREATMENTS:
        raise SettingsError(
            f"LAST_STOP_TREATMENT must be one of {', '.join(LAST_STOP_TREATMENTS)}, "
            f"not {settings.last_stop_treatment!r}."
        )
    if settings.last_stop_routes not in LAST_STOP_ROUTE_CHOICES:
        raise SettingsError(
            f"LAST_STOP_ROUTES must be one of {', '.join(LAST_STOP_ROUTE_CHOICES)}, "
            f"not {settings.last_stop_routes!r}."
        )
    if not 0.0 <= settings.trim_frac < 0.5:
        raise SettingsError(
            f"TRIM_FRAC must be at least 0 and below 0.5, not {settings.trim_frac!r}."
        )
    bad = [p for p in settings.percentiles if not 0 <= float(p) <= 100]
    if bad:
        raise SettingsError(
            f"RUNTIME_PERCENTILES must lie between 0 and 100: {', '.join(f'{p:g}' for p in bad)}."
        )
    for name, value in (
        ("MAX_UNSCHEDULED_SHARE", settings.max_unscheduled_share),
        ("MAX_CANCELED_SHARE", settings.max_canceled_share),
        ("MAX_UNRECORDED_SHARE", settings.max_unrecorded_share),
        ("MAX_TRIP_UNRECORDED_SHARE", settings.max_trip_unrecorded_share),
    ):
        if not 0.0 <= value <= 1.0:
            raise SettingsError(f"{name} must be between 0 and 1, not {value!r}.")
    if settings.start_date and settings.end_date and settings.start_date > settings.end_date:
        raise SettingsError(
            f"START_DATE {settings.start_date} is after END_DATE {settings.end_date}."
        )
    return measures, levels, cuts, day_types, list(settings.chart_levels)


TRIP_OBSERVATION_COLUMNS: Tuple[str, ...] = (
    "service_date",
    "day_of_week",
    "route_id",
    "direction_id",
    "route_type_agency",
    "trip_id",
    "trip_id_performed",
    "block_id",
    "scheduled_start",
    "time_period",
    "hour",
    "actual_runtime_min",
    "scheduled_runtime_min",
    "deviation_min",
)

SEGMENT_OBSERVATION_COLUMNS: Tuple[str, ...] = (
    "service_date",
    "day_of_week",
    "route_id",
    "direction_id",
    "route_type_agency",
    "block_id",
    "trip_id",
    "trip_id_performed",
    "seq",
    "segment",
    "from_stop_id",
    "to_stop_id",
    "time_period",
    "hour",
    "actual_runtime_min",
    "scheduled_runtime_min",
    "deviation_min",
    "recovery_after_min",
)


def run(settings: RunSettings) -> RunResult:
    """Build every enabled table, write them with the charts, and write the run log.

    Args:
        settings: Resolved settings.

    Returns:
        The tables built (also written under ``settings.output_dir``).

    Raises:
        TimePeriodError: If ``settings.periods`` do not cover the day exactly once.
        SettingsError: If another setting is unusable.
        InputDataError: If the TIDES tables cannot support the exports.
        OSError: If an input cannot be read or an output cannot be written.
        RuntimeError: If the run log cannot be written and it is required.
    """
    measures, levels, cuts, day_types, chart_levels = _resolve_run_settings(settings)
    holiday = settings.holiday_day_type.strip()
    order = day_type_order(day_types, holiday)
    periods = list(settings.periods)

    stop_visits = load_tides_csv(settings.stop_visits_path, "stop_visits")
    trips = load_tides_csv(settings.trips_performed_path, "trips_performed")
    if settings.routes_to_include or settings.routes_to_exclude:
        _require_columns(trips, ["route_id"], "trips_performed", "the route filters need")

    stop_visits, duplicate_stats = dedupe_stop_visits(stop_visits)
    # Neither step may stop the run yet: the date audit below is written first.
    matched = join_trips(stop_visits, trips, require_rows=False)
    joined = filter_dates(matched, settings.start_date, settings.end_date, require_rows=False)
    joined = add_trip_key(joined)

    # --- service dates ---------------------------------------------------------
    trip_stats = trip_date_stats(trips)
    in_range = pd.Series(True, index=trip_stats.index)
    if settings.start_date is not None:
        in_range &= trip_stats.index >= pd.Timestamp(settings.start_date)
    if settings.end_date is not None:
        in_range &= trip_stats.index <= pd.Timestamp(settings.end_date)
    audit_dates = set(joined["service_date"].unique()) | set(trip_stats.index[in_range])
    feeds = [load_gtfs_feed(str(path)) for path in settings.gtfs_paths]
    schedule = schedule_for_run(audit_dates, feeds, trips, settings)
    coverage_stats = coverage_by_date(schedule) if feeds else None
    audit_dates |= set(schedule["service_date"].unique())
    classification = classify_service_dates(
        sorted(audit_dates),
        day_types,
        settings.holidays,
        holiday,
        feeds,
        trip_stats,
        settings.max_unscheduled_share,
        settings.max_canceled_share,
        duplicate_stats,
        coverage_stats,
        settings.max_unrecorded_share,
    )
    classification["excluded"] = (classification["day_type"] == "") | (
        classification["flagged"] & settings.exclude_flagged_dates
    )
    log_date_classification(classification, holiday)
    published = classification.drop(columns=["_stamp", "_day_types"])
    files: List[Path] = []
    out_dir = settings.output_dir
    _write_csv(published, out_dir / DATE_CLASSIFICATION_FILENAME, files)
    trip_coverage = pd.DataFrame()
    if feeds:
        trip_coverage = build_trip_coverage(
            schedule, classification, periods, order, settings.max_trip_unrecorded_share
        )
        _write_csv(trip_coverage, out_dir / TRIP_COVERAGE_FILENAME, files)
        flagged_trips = int(trip_coverage["flagged"].sum()) if not trip_coverage.empty else 0
        if flagged_trips:
            logging.warning(
                "%d scheduled trip(s) have no TIDES record on more than %.0f%% of their "
                "scheduled dates; see %s.",
                flagged_trips,
                settings.max_trip_unrecorded_share * 100,
                TRIP_COVERAGE_FILENAME,
            )
    elif (out_dir / TRIP_COVERAGE_FILENAME).exists():
        (out_dir / TRIP_COVERAGE_FILENAME).unlink()
        logging.info("No GTFS feed given; removed the previous %s.", TRIP_COVERAGE_FILENAME)
    written = f" {DATE_CLASSIFICATION_FILENAME} was still written."
    if matched.empty:
        raise InputDataError(NO_MATCH_MESSAGE + written)
    if joined.empty:
        raise InputDataError(out_of_range_message(settings.start_date, settings.end_date) + written)
    date_table = date_table_from_classification(classification, settings.exclude_flagged_dates)
    if date_table.empty:
        raise InputDataError(
            "No service date is left in any day type; check DAY_TYPES and "
            "EXCLUDE_FLAGGED_DATES." + written
        )

    # --- shared preparation ----------------------------------------------------
    kept_dates = set(date_table["service_date"])
    visits = joined.loc[joined["service_date"].isin(kept_dates)]
    if visits.empty:
        raise InputDataError(
            "No stop visits are left on a service date in any day type; check DAY_TYPES and "
            "EXCLUDE_FLAGGED_DATES."
        )
    visits = filter_routes(visits, settings.routes_to_include, settings.routes_to_exclude)
    visits = add_parsed_timestamps(label_blank_groups(visits, GROUP_KEY_COLUMNS))
    trip_frame = build_trip_frame(visits, settings.cutover_hour)
    starts = trip_frame[[*TRIP_KEY, "_start_min"]]
    stop_text = "_sched_text" if settings.stop_time_basis == "scheduled" else "_actual_text"
    trip_text = (
        "_start_scheduled" if settings.trip_time_basis == "scheduled_start" else "_start_actual"
    )

    def otp_table(rows: pd.DataFrame, level: str, cut: str) -> pd.DataFrame:
        return summarize_otp(rows, level, cut, periods, order)

    def runtime_table(rows: pd.DataFrame, level: str, cut: str) -> pd.DataFrame:
        return summarize_runtime(rows, level, cut, periods, order, settings.percentiles)

    # --- on-time performance -----------------------------------------------------
    otp_tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    stop_flags = stop_detail = pd.DataFrame()
    otp_counts: Dict[str, int] = {}
    last_stop_scope = "all routes"
    if "otp" in measures:
        treated_routes: Optional[frozenset[str]] = None
        if settings.last_stop_treatment != "score" and settings.last_stop_routes == "express":
            treated_routes, express_source = resolve_express_routes(
                visits,
                settings.express_routes_file,
                settings.express_routes,
                settings.express_service_types,
            )
            last_stop_scope = (
                f"express routes ({', '.join(sorted(treated_routes)) or 'none'}; "
                f"from {express_source})"
            )
            if not treated_routes:
                logging.warning(
                    "LAST_STOP_TREATMENT %r applies to express routes, but none were found; "
                    "every last stop is scored normally.",
                    settings.last_stop_treatment,
                )
        scored, otp_counts = score_otp(
            visits,
            settings.early_min,
            settings.late_min,
            settings.timepoints_only,
            settings.last_stop_treatment,
            treated_routes,
        )
        logging.info(
            "OTP: %d candidate visit(s), %d scored; %d had no actual time and %d no "
            "scheduled time.",
            otp_counts["candidates"],
            otp_counts["scored"],
            otp_counts["missing_actual_time"],
            otp_counts["missing_schedule_time"],
        )
        if settings.last_stop_treatment == "exclude":
            logging.info(
                "OTP: %d last-stop visit(s) on %s left out (LAST_STOP_TREATMENT 'exclude').",
                otp_counts["last_stop_excluded"],
                last_stop_scope,
            )
        elif settings.last_stop_treatment == "early_ok":
            logging.info(
                "OTP: %d early last-stop visit(s) on %s scored on time "
                "(LAST_STOP_TREATMENT 'early_ok').",
                otp_counts["early_ok_at_last_stop"],
                last_stop_scope,
            )
        if scored.empty:
            logging.warning("No visit could be scored for OTP; the OTP tables are skipped.")
        else:
            warn_extreme_deviations(scored, settings.dev_abs_warn_min)
            rows = scored.merge(starts, on=TRIP_KEY, how="left")
            rows = add_time_keys(
                rows, rows[stop_text], periods, settings.cutover_hour, "scored visits"
            )
            rows = add_day_type(rows, date_table)
            warn_thin_day_types(rows, settings.thin_day_type_dates)
            otp_tables = _write_grains(
                rows, "otp", levels, cuts, otp_table, out_dir / OTP_SUBDIR, files
            )
            if settings.export_stop_flags and "stop_id" in visits.columns:
                trips_kept = in_service_trips(trips)
                trips_kept = trips_kept.loc[trips_kept["service_date"].isin(kept_dates)]
                try:
                    trips_kept = filter_routes(
                        trips_kept,
                        settings.routes_to_include,
                        settings.routes_to_exclude,
                        warn=False,
                    )
                except InputDataError:
                    trips_kept = trips_kept.iloc[0:0]
                trips_kept = label_blank_groups(trips_kept, GROUP_KEY_COLUMNS, warn=False)
                stop_flags, stop_detail = build_stop_flags(visits, scored, trips_kept, settings)
                _write_csv(stop_flags, out_dir / OTP_SUBDIR / STOP_FLAGS_FILENAME, files)
                _write_csv(stop_detail, out_dir / OTP_SUBDIR / STOP_ROUTE_DETAIL_FILENAME, files)

    # --- trip running time -------------------------------------------------------
    trip_tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    trip_observations = trip_outliers = trip_flags = dow_flags = pd.DataFrame()
    trip_rows = pd.DataFrame()
    if "trip_runtime" in measures:
        per_trip = compute_trip_runtimes(visits)
        if per_trip.empty:
            logging.warning("No trip has a running time; the trip running-time tables are skipped.")
        else:
            rows = per_trip.merge(trip_frame, on=TRIP_KEY, how="left")
            rows = add_time_keys(rows, rows[trip_text], periods, settings.cutover_hour, "trips")
            retained, outliers = trim_outliers(rows, settings.trim_frac)
            if not outliers.empty:
                logging.info(
                    "Trimmed %d trip running time(s) in the outer %.0f%% tails of their trips; "
                    "see %s.",
                    len(outliers),
                    settings.trim_frac * 100,
                    TRIP_OUTLIERS_FILENAME,
                )
            trip_rows = add_day_type(retained, date_table)
            trip_tables = _write_grains(
                trip_rows,
                "trip_runtime",
                levels,
                cuts,
                runtime_table,
                out_dir / TRIP_RUNTIME_SUBDIR,
                files,
            )
            trip_observations = _observation_export(retained, TRIP_OBSERVATION_COLUMNS)
            trip_outliers = _observation_export(outliers, TRIP_OBSERVATION_COLUMNS)
            trip_flags = build_trip_flags(
                trip_rows,
                settings.high_cv_threshold,
                settings.min_obs_for_cv,
                settings.gap_frac,
                order,
            )
            dow_flags = build_dow_flags(
                trip_rows, settings.dow_low_count_frac, settings.dow_runtime_pct, order
            )
            runtime_dir = out_dir / TRIP_RUNTIME_SUBDIR
            _write_csv(trip_observations, runtime_dir / TRIP_OBSERVATIONS_FILENAME, files)
            _write_csv(trip_outliers, runtime_dir / TRIP_OUTLIERS_FILENAME, files)
            _write_csv(trip_flags, runtime_dir / TRIP_FLAGS_FILENAME, files)
            _write_csv(dow_flags, runtime_dir / TRIP_DOW_FLAGS_FILENAME, files)
            n_variable = int(trip_flags["high_variation"].sum())
            n_gaps = int(trip_flags["data_gap"].sum())
            if n_variable or n_gaps:
                logging.warning(
                    "%d trip(s) have highly variable running times and %d look like data gaps; "
                    "see %s.",
                    n_variable,
                    n_gaps,
                    TRIP_FLAGS_FILENAME,
                )

    # --- segment running time ----------------------------------------------------
    segment_tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    segment_observations = pd.DataFrame()
    if "segment_runtime" in measures:
        segments = build_segments(visits, settings.segment_timepoints_only)
        if segments.empty:
            logging.warning("No trip has two served timepoints; the segment tables are skipped.")
        else:
            recovery = compute_block_recovery(trips)
            if recovery.empty:
                segments["recovery_after_min"] = np.nan
            else:
                segments = segments.merge(recovery, on=TRIP_KEY, how="left")
            segments = add_time_keys(
                segments, segments[stop_text], periods, settings.cutover_hour, "segment traversals"
            )
            segment_rows = add_day_type(segments, date_table)
            segment_dir = out_dir / SEGMENT_RUNTIME_SUBDIR
            segment_tables = _write_grains(
                segment_rows, "segment_runtime", levels, cuts, runtime_table, segment_dir, files
            )
            if not missing_level_columns("segment", segment_rows.columns):
                by_route = segment_tables.get(("segment", "all_day"))
                if by_route is None:
                    by_route = runtime_table(segment_rows, "segment", "all_day")
                for stem, table in build_segment_route_tables(by_route, order).items():
                    _write_csv(table, segment_dir / SEGMENT_ROUTE_SUBDIR / f"{stem}.csv", files)
            segment_observations = _observation_export(segments, SEGMENT_OBSERVATION_COLUMNS)
            _write_csv(segment_observations, segment_dir / SEGMENT_OBSERVATIONS_FILENAME, files)

    if not (otp_tables or trip_tables or segment_tables):
        raise InputDataError(
            "No table could be built: no visit was scored for OTP and no running time could "
            "be measured. Check the timestamps, the timepoint column and the measures."
        )

    # --- charts ------------------------------------------------------------------
    charts = 0
    if settings.export_charts:
        charts_dir = out_dir / CHARTS_SUBDIR
        charts += export_summary_charts(
            otp_tables,
            chart_levels,
            [("% on time", "pct_on_time")],
            "% on time",
            charts_dir / "otp",
            periods,
            (f"standard ({settings.otp_standard:g}%)", settings.otp_standard),
        )
        charts += export_summary_charts(
            trip_tables,
            chart_levels,
            [("Median actual", "actual_median_min"), ("Scheduled", "scheduled_min")],
            "Running time (minutes)",
            charts_dir / "trip_runtime",
            periods,
        )
        if ("trip", "all_day") in otp_tables:
            charts += export_trip_otp_charts(
                otp_tables[("trip", "all_day")],
                settings.otp_standard,
                charts_dir / "otp_trips",
                settings.max_trips_per_chart,
            )
        if not trip_rows.empty:
            charts += export_runtime_boxplots(
                trip_rows, charts_dir / "trip_runtime_boxplots", settings.max_trips_per_chart
            )
        logging.info("Wrote %d chart PNG(s) to %s.", charts, charts_dir)

    # --- run log -----------------------------------------------------------------
    holidays_text = ", ".join(d.isoformat() for d in settings.holidays) or "(none)"
    settings_lines = [
        f"stop_visits:        {settings.stop_visits_path}",
        f"trips_performed:    {settings.trips_performed_path}",
        f"Measures:           {', '.join(measures)}",
        f"Levels:             {', '.join(levels)}",
        f"Time cuts:          {', '.join(cuts)}",
        f"Trip time basis:    {settings.trip_time_basis}",
        f"Stop time basis:    {settings.stop_time_basis} (stop visits and segments)",
        f"Service day starts: {settings.cutover_hour:02d}:00",
        "Day types:          "
        + "; ".join(
            f"{name} = {', '.join(WEEKDAY_NAMES[n][:3] for n in nums)}"
            for name, nums in day_types.items()
        ),
        f"HOLIDAYS list:      {holidays_text} (day type {holiday})",
        f"GTFS feeds:         {', '.join(str(p) for p in settings.gtfs_paths) or '(none)'}",
        f"Date checks:        flag above {settings.max_unscheduled_share:.0%} of trips "
        f"unscheduled, {settings.max_canceled_share:.0%} canceled or "
        f"{settings.max_unrecorded_share:.0%} of scheduled trips without a record; "
        f"exclude flagged dates: {settings.exclude_flagged_dates}",
        f"Date range:         {settings.start_date or '(start)'} to {settings.end_date or '(end)'}",
        f"Routes included:    {', '.join(settings.routes_to_include) or '(all)'}",
        f"Routes excluded:    {', '.join(settings.routes_to_exclude) or '(none)'}",
        f"On-time window:     {settings.early_min:g} to {settings.late_min:g} min; "
        f"timepoints only: {settings.timepoints_only}; standard {settings.otp_standard:g}%",
        "Last stop:          scored on arrival; "
        + (
            "scored like any timepoint"
            if settings.last_stop_treatment == "score"
            else f"{settings.last_stop_treatment} on {last_stop_scope}"
        ),
        f"Running time:       trim {settings.trim_frac:g} per tail; percentiles "
        f"{', '.join(f'{p:g}' for p in settings.percentiles)}; segments between "
        f"{'timepoints' if settings.segment_timepoints_only else 'served stops'}",
    ]
    service_dates = (
        date_table.loc[date_table["service_date"].isin(set(visits["service_date"]))]
        .groupby("day_type")["service_date"]
        .nunique()
        .to_dict()
    )
    summary_lines = [
        f"OTP tables:         {len(otp_tables)}",
        f"Trip runtime tables: {len(trip_tables)}",
        f"Segment tables:     {len(segment_tables)}",
        f"Visits scored:      {otp_counts.get('scored', 0)} of {otp_counts.get('candidates', 0)}",
        f"Trip running times: {len(trip_observations)} kept, {len(trip_outliers)} trimmed",
        f"Segment traversals: {len(segment_observations)}",
        "Service dates:      "
        + ", ".join(f"{d} {service_dates[d]}" for d in order if d in service_dates),
        "Holiday dates:      "
        + (
            ", ".join(
                f"{d} ({src})"
                for d, src, types in zip(
                    published["service_date"],
                    published["day_type_source"],
                    classification["_day_types"],
                )
                if holiday in types
            )
            or "(none)"
        ),
        "Flagged dates:      "
        + (
            "; ".join(
                f"{d}: {why}"
                for d, why in zip(published["service_date"], published["flag_reason"])
                if why
            )
            or "(none)"
        ),
        "Flagged stops:      "
        + (str(int((stop_flags["flag_reason"] != "").sum())) if not stop_flags.empty else "(none)"),
        f"Charts written:     {charts}",
    ]
    logged = write_run_log(
        out_dir / RUN_LOG_FILENAME,
        settings_lines,
        periods,
        settings.periods_source,
        summary_lines,
    )
    if not logged and settings.require_run_log:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return RunResult(
        otp_tables,
        trip_tables,
        segment_tables,
        stop_flags,
        stop_detail,
        trip_observations,
        trip_outliers,
        trip_flags,
        dow_flags,
        segment_observations,
        published,
        trip_coverage,
        files,
    )


# =============================================================================
# CLI / MAIN
# =============================================================================


def _percentile_list(text: str) -> List[float]:
    """Parse ``--percentiles`` (comma-separated numbers) for argparse."""
    try:
        values = [float(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"not a comma-separated list of numbers: {text!r}"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("give at least one percentile")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser (defaults are the CONFIGURATION values)."""
    p = argparse.ArgumentParser(
        description="Export TIDES on-time performance and running times at every grain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="TIDES stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="TIDES trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for the tables and run log.")
    p.add_argument(
        "--measures",
        nargs="+",
        choices=list(MEASURE_LEVELS),
        default=None,
        help="Measures to export; replaces EXPORT_MEASURES.",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        choices=list(LEVELS),
        default=None,
        help="Levels to export; replaces EXPORT_LEVELS.",
    )
    p.add_argument(
        "--time-cuts",
        nargs="+",
        choices=list(TIME_CUTS),
        default=None,
        help="Time cuts to export; replaces EXPORT_TIME_CUTS.",
    )
    p.add_argument(
        "--trip-time-basis",
        choices=list(TRIP_TIME_BASES),
        default=TRIP_TIME_BASIS,
        help="Clock time that places a trip in a time period and hour.",
    )
    p.add_argument(
        "--stop-time-basis",
        choices=list(STOP_TIME_BASES),
        default=STOP_TIME_BASIS,
        help="Clock time that places a stop visit or segment in a time period and hour.",
    )
    periods = p.add_mutually_exclusive_group()
    periods.add_argument(
        "--time-periods-file",
        default=TIME_PERIODS_FILE,
        help="Text file of time periods, one 'NAME = HH:MM - HH:MM' per line; replaces "
        "TIME_PERIODS.",
    )
    periods.add_argument(
        "--time-period",
        action="append",
        metavar="NAME=HH:MM-HH:MM",
        help="One time period; repeat the flag for each. Replaces TIME_PERIODS and any "
        "time-periods file. Leave unset to use those.",
    )
    p.add_argument("--start-date", default=START_DATE, help="First service date (YYYY-MM-DD).")
    p.add_argument("--end-date", default=END_DATE, help="Last service date (YYYY-MM-DD).")
    p.add_argument(
        "--gtfs",
        nargs="+",
        default=None,
        metavar="FEED",
        help="GTFS feeds (folders or .zip), oldest first; replaces GTFS_PATHS.",
    )
    p.add_argument(
        "--holiday",
        action="append",
        metavar="YYYY-MM-DD",
        help="One holiday date; repeat the flag for each. Replaces HOLIDAYS.",
    )
    p.add_argument(
        "--exclude-flagged-dates",
        action=argparse.BooleanOptionalAction,
        default=EXCLUDE_FLAGGED_DATES,
        help="Leave flagged dates out of every table.",
    )
    p.add_argument(
        "--routes-include",
        nargs="*",
        default=list(ROUTES_TO_INCLUDE),
        metavar="ROUTE_ID",
        help="Keep only these route_ids.",
    )
    p.add_argument(
        "--routes-exclude",
        nargs="*",
        default=list(ROUTES_TO_EXCLUDE),
        metavar="ROUTE_ID",
        help="Drop these route_ids.",
    )
    p.add_argument(
        "--early-min", type=float, default=EARLY_MIN, help="Earliest on-time deviation (minutes)."
    )
    p.add_argument(
        "--late-min", type=float, default=LATE_MIN, help="Latest on-time deviation (minutes)."
    )
    p.add_argument(
        "--last-stop-treatment",
        choices=LAST_STOP_TREATMENTS,
        default=LAST_STOP_TREATMENT,
        help="How a trip's last stop counts toward OTP on the --last-stop-routes.",
    )
    p.add_argument(
        "--last-stop-routes",
        choices=LAST_STOP_ROUTE_CHOICES,
        default=LAST_STOP_ROUTES,
        help="Apply --last-stop-treatment to express routes only or to every route.",
    )
    p.add_argument(
        "--express-routes-file",
        default=EXPRESS_ROUTES_FILE,
        help="Text file of express route_ids, one per line.",
    )
    p.add_argument(
        "--express-routes",
        nargs="+",
        metavar="ROUTE_ID",
        help="Express route_ids (replace EXPRESS_ROUTES; added to the express-routes file).",
    )
    p.add_argument(
        "--timepoints-only",
        action=argparse.BooleanOptionalAction,
        default=TIMEPOINTS_ONLY,
        help="Score OTP at timepoint visits only.",
    )
    p.add_argument(
        "--stop-flags",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_STOP_FLAGS,
        help="Write otp_stop_flags.csv and its per-route detail.",
    )
    p.add_argument(
        "--trim-frac",
        type=float,
        default=TRIM_FRAC,
        help="Share of each scheduled trip's running times trimmed from each tail.",
    )
    p.add_argument(
        "--percentiles",
        type=_percentile_list,
        default=list(RUNTIME_PERCENTILES),
        metavar="P1,P2,...",
        help="Percentiles of the actual running time to report.",
    )
    p.add_argument(
        "--charts",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_CHARTS,
        help="Write chart PNGs (--charts) or skip them (--no-charts).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the settings, then build and write every enabled table.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv`` (or nothing in a
            notebook).

    Returns:
        Process exit code: 0 on success, 1 on a runtime failure, 2 if a setting fails
        its checks or the input paths are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))

    try:
        periods, source = resolve_time_periods(
            args.time_period, args.time_periods_file, TIME_PERIODS
        )
    except TimePeriodError as exc:
        logging.error("%s", exc)
        return 2
    logging.info("Time periods from %s:", source)
    for period in periods:
        logging.info("  %s", format_period(period))

    try:
        holiday_setting = "--holiday" if args.holiday else "HOLIDAYS"
        holidays = parse_iso_dates(args.holiday or HOLIDAYS, holiday_setting)
        start = parse_optional_date(args.start_date, "START_DATE")
        end = parse_optional_date(args.end_date, "END_DATE")
        settings = RunSettings(
            stop_visits_path=Path(args.stop_visits).expanduser(),
            trips_performed_path=Path(args.trips_performed).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            periods=periods,
            periods_source=source,
            measures=args.measures,
            levels=args.levels,
            time_cuts=args.time_cuts,
            trip_time_basis=args.trip_time_basis,
            stop_time_basis=args.stop_time_basis,
            gtfs_paths=tuple(args.gtfs) if args.gtfs else tuple(GTFS_PATHS),
            holidays=holidays,
            start_date=start,
            end_date=end,
            exclude_flagged_dates=args.exclude_flagged_dates,
            routes_to_include=tuple(args.routes_include),
            routes_to_exclude=tuple(args.routes_exclude),
            early_min=args.early_min,
            late_min=args.late_min,
            last_stop_treatment=args.last_stop_treatment,
            last_stop_routes=args.last_stop_routes,
            express_routes_file=args.express_routes_file,
            express_routes=tuple(args.express_routes or EXPRESS_ROUTES),
            timepoints_only=args.timepoints_only,
            export_stop_flags=args.stop_flags,
            trim_frac=args.trim_frac,
            percentiles=tuple(args.percentiles),
            export_charts=args.charts,
        )
        _resolve_run_settings(settings)
    except SettingsError as exc:
        logging.error("%s", exc)
        return 2

    placeholders = [
        name
        for name, value in (
            ("STOP_VISITS_PATH", args.stop_visits),
            ("TRIPS_PERFORMED_PATH", args.trips_performed),
            ("OUTPUT_DIR", args.output_dir),
        )
        if str(value).strip() in PLACEHOLDER_PATHS
    ]
    if placeholders:
        logging.warning(
            "%s still hold the placeholder path. Update the CONFIGURATION section or pass "
            "--stop-visits / --trips-performed / --output-dir before running.",
            ", ".join(placeholders),
        )
        return 2
    for path in (settings.stop_visits_path, settings.trips_performed_path):
        if not path.is_file():
            logging.error("Input file not found: %s", path)
            return 1
    for feed in settings.gtfs_paths:
        if not Path(feed).expanduser().exists():
            logging.error("GTFS feed not found: %s", feed)
            return 1

    try:
        run(settings)
    except (InputDataError, SettingsError, OSError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
