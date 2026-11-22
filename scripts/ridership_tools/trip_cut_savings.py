"""Screen potential trip cuts using ridership, GTFS blocks, and optional deadhead data.

The output shortlists individual trip cuts and frequency-reduction sessions, each
evaluated independently against the original schedule and rated excellent, good, poor,
or ineligible by boardings on the removed trips per modeled vehicle-hour saved, using
editable thresholds. Ratings do not establish actual ridership loss, paid hours saved,
or whether the service should be removed, and option savings are not additive.

A cut can trim a block boundary, remove a whole block, or open a mid-block gap long
enough for a garage release. Other mid-block cuts merely replace runtime with waiting.
Frequency sessions retain existing departure times and blocks, price their removed
trips together, and protect daily service endpoints by default. Reblocking is not
implemented, and the schedule itself is never changed.

Neighbors: ``ridership_by_hour.py`` and ``ridership_from_tides.py`` write
``ridership_by_trip_<day>.csv`` tables this script reads in ``trip_id`` mode, and flag
trips against a systemwide minimum-productivity standard; this script screens achievable
vehicle-time savings. Pass a reviewed option's trip_ids to ``service_cut_impact_gpd.py``
as a ``trip_ids`` scenario for stop-level coverage and rider impacts.

Inputs
------
- A GTFS feed (folder or .zip) whose trips.txt carries ``block_id``.
- One trip-level ridership file (.csv or .xlsx/.xls) of average daily boardings per day
  type, matched to that day's GTFS trips by ``trip_id`` or by route + start time
  (``RIDERSHIP_TRIP_MATCH_MODE``), refined by direction name and block when present.
- Optional: an extracted TODS folder (``TODS_PATH``), such as ``tsde_to_tods.py``
  writes, for scheduled pull-out, pull-in, and deadhead times.

Outputs
-------
Per-day CSVs are named ``trip_cut_<table>_<day>.csv``:

- Options: ``grouped_shortlist``, ``alternatives``, ``shortlist``, ``options`` (all,
  including ineligible), ``sessions`` / ``session_trips``, and ``option_blocks`` /
  ``option_gaps`` (each option's block costs and changed departure gaps).
- Trips: ``review`` and ``trips``, with every trip's rating, ridership, observation
  count, block position, and what cutting it alone would save.
- Context: ``blocks``, ``cycle_times``, and ``vehicle_profile`` (baseline diagnostics);
  ``shared_routes``, ``linked_groups``, and a paste-ready ``linked_route_config`` .txt.
- Audit: ``ridership_match``, ``block_qa``, ``excluded_trips``, ``deadhead_audit``.
- ``proposal_workbooks/*.xlsx``: one proposal per workbook (index: ``workbook_index``);
  ``workbook_trips`` and a JSON context let ``--workbooks-from`` re-export them.
- ``trip_cut_summary.csv`` (one column per day type) and a run-log sidecar with the
  verbatim CONFIGURATION block, resolved settings, and input SHA-256 fingerprints.

Limitations
-----------
Without TODS, each pull-out + pull-in pair costs a flat ``PULL_DEADHEAD_MIN`` (half for a
missing garage leg); with TODS, new direct connections without an estimate are rejected.
Vehicle hours are not operator pay hours: runcutting rules, guarantees, and split or
spread premiums are not modeled. ``frequencies.txt`` service is not expanded.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags) and run
from a shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Union, cast

import pandas as pd

RouteListSource = Union[str, Path, Sequence[str]]

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS feed: the unzipped folder or the .zip as published. trips.txt must carry
# block_id — blocks decide whether a cut saves vehicle time.
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"

# Trip-level ridership (average daily boardings), one file per day type, as .csv or
# .xlsx/.xls. Keys are "weekday", "saturday", "sunday" (or any of "monday".."sunday").
# Each file is matched to the GTFS trips of its own day type and analysed on its
# own. An empty path skips that day type. For the ridership_by_trip_<day>.csv tables
# from ridership_by_hour.py or ridership_from_tides.py, set
# RIDERSHIP_TRIP_MATCH_MODE = "trip_id" and the column names below.
RIDERSHIP_PATHS: Mapping[str, str] = {
    "weekday": r"Path\To\Your\weekday\RIDERSHIP_BY_ROUTE_AND_TRIP.xlsx",
    "saturday": r"Path\To\Your\saturday\RIDERSHIP_BY_ROUTE_AND_TRIP.xlsx",
    "sunday": r"Path\To\Your\sunday\RIDERSHIP_BY_ROUTE_AND_TRIP.xlsx",
}

OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# A representative date of each day type is chosen from the feed's calendar
# (skipping observed federal holidays). To pin an explicit date instead, give
# "YYYY-MM-DD" keyed like RIDERSHIP_PATHS, e.g. {"weekday": "2026-10-14"}.
SERVICE_DATES: Mapping[str, str] = {}

# --- Ridership matching ----------------------------------------------------------
# "trip_id": RIDERSHIP_TRIP_ID_COL holds GTFS trip_ids.
# "route_start_time": each row matches the GTFS trip on the same route
#   (route_short_name or route_id) whose first departure is nearest the row's
#   start time, within RIDERSHIP_START_TIME_TOLERANCE_MIN, narrowed by direction
#   and block when the columns below are set. Matching is one-to-one: a trip
#   takes at most one row (per-stop/per-date rows may form one group). Ties,
#   competing equal-evidence rows, and unresolved directions stay unmatched.
#   Rows are never reassigned to a less suitable trip merely to improve coverage.
RIDERSHIP_TRIP_MATCH_MODE: str = "route_start_time"
RIDERSHIP_TRIP_ID_COL: str = "trip_id"
RIDERSHIP_BOARDINGS_COL: str = "PASSENGERS_ON"
RIDERSHIP_ROUTE_COL: str = "ROUTE_NAME"
RIDERSHIP_START_TIME_COL: str = "TRIP_START_TIME"
RIDERSHIP_START_TIME_TOLERANCE_MIN: int = 2
# route_start_time only. Direction names (e.g. EASTBOUND) are mapped to GTFS
# direction_id route by route, learned from the rows whose start time fits trips of
# only one direction; each row then matches only trips in its direction. The
# learned mapping is in the ridership match table. "" = ignore direction.
RIDERSHIP_DIRECTION_COL: str = "DIRECTION_NAME"
# route_start_time only. When a row's block equals the GTFS block_id of a candidate
# trip, only trips in that block are considered; otherwise matching falls back to
# route, time, and direction (and the run warns if that happens for most rows).
# "" = ignore blocks.
RIDERSHIP_BLOCK_COL: str = "BLOCK"
# Observation count behind each row's average (Ridecheck TRIPS_COUNT).
# Values below LOW_OBSERVATION_SHARE x the median are flagged. When
# PROTECT_LOW_OBSERVATIONS is True, flagged trips cannot be selected.
# "" = no check.
RIDERSHIP_OBS_COL: str = "TRIPS_COUNT"
LOW_OBSERVATION_SHARE: float = 0.25
# Optional stop column. Rows with a stop value are summed to their trip; a trip
# carrying both per-stop rows and a whole-trip row keeps only the per-stop rows.
RIDERSHIP_STOP_ID_COL: str = "stop_id"
# For a file with one row per trip per surveyed day, name its date column so
# boardings are averaged across days instead of summed. "" = already averaged.
RIDERSHIP_DATE_COL: str = ""

# --- Candidate screening ---------------------------------------------------------
# Every trip and frequency session is rated on its own; no package of cuts is selected.
# Boardings on the removed trip / modeled vehicle-hours saved by cutting it alone.
# Lower is better. These are illustrative starting thresholds, NOT agency standards.
# <= excellent limit: excellent; <= good limit: good; above good limit: poor.
# Ineligible trips (route exclusions, no time saved, missing data, etc.) are not rated.
EXCELLENT_MAX_BOARDINGS_PER_VEHICLE_HOUR_SAVED: float = 5.0
GOOD_MAX_BOARDINGS_PER_VEHICLE_HOUR_SAVED: float = 10.0

# --- Independent frequency-reduction sessions -----------------------------------
# Each route/window/retained-departure pattern is ONE option, evaluated alone.
# A lower frequency means a LONGER headway. Manual windows supplement the search below.
# Empty routes/days means all routes/day types. Use quoted route names or route_ids.
# Set this to [] AND HEADWAY_SEARCH to {} to disable sessions.
HEADWAY_SESSIONS: Sequence[Mapping[str, Any]] = [
    {
        "name": "Midday_30_to_60",
        "start": "10:00",
        "end": "12:00",
        "from_headway_min": 30,
        "to_headway_min": 60,
        "routes": [],
        "days": [],
    },
]
# Generate overlapping windows between these service-day times. {} disables search.
# Each route/group is evaluated separately; none of these alternatives are combined.
HEADWAY_SEARCH: Mapping[str, Any] = {
    "start": "06:00",
    "end": "22:00",
    "durations_min": [120, 180, 240, 360],
    "step_min": 30,
    "headways": [[15, 30], [20, 40], [30, 60]],
    "routes": [],
    "days": [],
}
# Retain the first AND last SERVICE-DAY departure of each route/direction.
# Window endpoints may be removed. False allows the daily span to shrink;
# PROTECT_ROUTE_SPAN still applies. No departure is retimed.
# Session bounds are inclusive; all departures outside the window remain unchanged.
SESSION_PRESERVE_FIRST_LAST: bool = True
# Existing and resulting headways must match the requested values within this tolerance.
# Shorter transition gaps are allowed when protected boundary trips are retained.
SESSION_HEADWAY_TOLERANCE_MIN: int = 2
# Top options of EACH type (single_trip and frequency_session); 0 exports all eligible.
SHORTLIST_LIMIT_PER_TYPE: int = 25
# Suppress a lower-ranked session if its removed-trip set has this Jaccard overlap
# (intersection / union) with a better session for the same routes and headways.
# All alternatives remain in the detailed outputs; 0 disables near-duplicate suppression.
SESSION_SHORTLIST_OVERLAP: float = 0.75
# "auto": manual groups plus qualifying pairs discovered from shared GTFS blocks.
# "manual": only groups below. "off": single-route sessions only (still log pairs).
# Auto pairs must share at least this many valid blocks AND have a direct route
# transition on a block. Indirect links never merge whole connected networks.
LINKED_ROUTE_MODE: str = "auto"
LINKED_ROUTE_MIN_SHARED_BLOCKS: int = 2
# Route names or IDs, as lists or one-route-per-line .txt paths, e.g.
# {"101_202": ["101", "202"], "another_group": r"C:\\data\\routes.txt"}
LINKED_ROUTE_GROUPS: Mapping[str, RouteListSource] = {}
# Every retained-departure phase is considered; oversized groups are explicitly
# reported as unsupported rather than silently sampling phase combinations.
SESSION_MAX_PHASE_COMBINATIONS: int = 256

# Proposal review workbooks. Each option is evaluated/exported independently.
# Explicit option IDs override the automatic selection; copy IDs from options.csv.
# With an empty list, export up to five grouped representatives of EACH option type.
# Identical cut sets across types are exported once. 0 means all representatives.
EXPORT_PROPOSAL_WORKBOOKS: bool = True
WORKBOOK_OPTION_IDS: Sequence[str] = []
WORKBOOK_LIMIT_PER_TYPE: int = 5
# Optional label for test runs; appears on every worksheet. Leave blank for real data.
WORKBOOK_DATA_LABEL: str = ""
# Future work: reblocking could reveal additional savings. This version NEVER moves
# a retained trip to another block or changes its departure time. Session savings
# come from repricing the complete affected original blocks, including other routes.

# --- Block savings rules ---------------------------------------------------------
# A gap between consecutive trips of a block at least this long lets the vehicle
# pull in and pull out again later, so the gap costs only PULL_DEADHEAD_MIN. A
# shorter gap is layover the vehicle sits through: a mid-block cut leaving a gap
# shorter than this frees no vehicle time.
MIN_RELEASE_GAP_MIN: int = 90
# Non-revenue minutes charged per pull-out + pull-in pair (garage to first stop and
# last stop to garage). GTFS has no deadhead, so this is a flat allowance. It must
# be below MIN_RELEASE_GAP_MIN. 0 ignores deadhead.
PULL_DEADHEAD_MIN: int = 30

# EDIT HERE: extracted TODS directory. Empty retains the flat pair allowance.
# Example: r"C:\path\to\tods_folder"
TODS_PATH: str = ""
# Minimum useful time at the garage when inferring a new mid-block release.
MIN_GARAGE_IDLE_MIN: int = 10
# Optional verified (route short name or route_id, vendor direction) -> GTFS direction.
# Leave unknown mappings out: unresolved competing directions are not assigned.
DIRECTION_OVERRIDES: Mapping[tuple[str, str], str] = {}
# Avoid choosing very sparsely observed trips solely because their averages are low.
PROTECT_LOW_OBSERVATIONS: bool = True

# --- Guards (off by default; their effects are reported either way) --------------
# Never cut a trip carrying more than this many boardings per revenue hour.
# 0 = no guard.
PROTECT_TRIPS_ABOVE: float = 0.0
# Reject a cut that would open a gap between consecutive departures on the same
# route and direction longer than this many minutes. 0 = no guard.
MAX_RESULTING_HEADWAY_MIN: int = 0
# Reject cuts that remove a route-direction's first or last trip of the day.
PROTECT_ROUTE_SPAN: bool = False
# Routes whose trips may be cut (route_short_name or route_id). Empty = all. Every
# route still counts toward the baseline totals.
ROUTES_TO_INCLUDE: RouteListSource = []

# --- Route exclusions: EDIT THESE LISTS ------------------------------------------
# Each setting accepts a list OR a .txt file path. These can be mixed by category:
#   COVERAGE_ROUTES = ["101", "202"]
#   COVERAGE_ROUTES = r"C:\Planning\route_lists\coverage_routes.txt"
# Use route_short_name or route_id as text. [] or "" = none.
# Files: UTF-8, one route per line, no header; blank lines and # comments ignored.
# Relative file paths are relative to this script (not the launch directory).
# In a pasted notebook cell without __file__, relative paths use the working directory.
# These passenger-service routes remain in baseline totals but cannot be cut.
# A route may appear in multiple lists; duplicates are combined automatically.
COVERAGE_ROUTES: RouteListSource = []
POLICY_ROUTES: RouteListSource = []  # e.g. ["101"]
GRANT_ROUTES: RouteListSource = []  # e.g. ["202"]
OTHER_EXCLUDED_ROUTES: RouteListSource = []

# Bookkeeping/non-public routes: removed from BOTH baseline totals and screening.
# Keep actual passenger-service routes in one of the lists above instead.
NON_REVENUE_ROUTES: RouteListSource = []  # e.g. ["9999A", "9999B"]

# Additional uncategorized exclusions; retains the earlier configuration option.
# This and ROUTES_TO_INCLUDE also accept a list OR a .txt file path.
ROUTES_TO_EXCLUDE: RouteListSource = []

# Combined at run time; add another named category here if needed.
# Exclusions override ROUTES_TO_INCLUDE. Missing/unreadable files stop the run.
ROUTE_EXCLUSION_GROUPS: Mapping[str, RouteListSource] = {
    "coverage": COVERAGE_ROUTES,
    "policy": POLICY_ROUTES,
    "grant": GRANT_ROUTES,
    "other": OTHER_EXCLUDED_ROUTES,
}

# --- Cycle-time diagnostics ------------------------------------------------------
# Named clock windows ("HH:MM" start, end) for the cycle-time table. A trip belongs
# to the window containing its start time; a window may wrap past midnight. An
# empty mapping reports one all-day row per route.
TIME_PERIODS: Mapping[str, tuple[str, str]] = {
    "AM PEAK": ("06:00", "09:00"),
    "MIDDAY": ("09:00", "15:00"),
    "PM PEAK": ("15:00", "18:00"),
    "EVENING": ("18:00", "22:00"),
    "NIGHT": ("22:00", "06:00"),
}
# Recovery (cycle time minus round-trip running time) as a percent of running time
# that the schedule is taken to need. Periods carrying more recovery than this
# could free vehicles by trimming running or recovery time; the table counts how
# many at each period's headway.
RECOVERY_TARGET_PCT: float = 15.0

# Bin width, in minutes, of the vehicles-in-service profile.
VEHICLE_PROFILE_BIN_MIN: int = 15

REVIEW_FILENAME: str = "trip_cut_review.csv"
OPTIONS_FILENAME: str = "trip_cut_options.csv"
SHORTLIST_FILENAME: str = "trip_cut_shortlist.csv"
SESSIONS_FILENAME: str = "trip_cut_sessions.csv"
SESSION_TRIPS_FILENAME: str = "trip_cut_session_trips.csv"
TRIPS_FILENAME: str = "trip_cut_trips.csv"
BLOCKS_FILENAME: str = "trip_cut_blocks.csv"
CYCLE_TIMES_FILENAME: str = "trip_cut_cycle_times.csv"
VEHICLE_PROFILE_FILENAME: str = "trip_cut_vehicle_profile.csv"
RIDERSHIP_MATCH_FILENAME: str = "trip_cut_ridership_match.csv"
SUMMARY_FILENAME: str = "trip_cut_summary.csv"
RUN_LOG_FILENAME: str = "trip_cut_savings_runlog.txt"

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

_PLACEHOLDER_PREFIX: str = "Path\\To\\"
_EXCEL_SUFFIXES: tuple[str, ...] = (".xls", ".xlsx", ".xlsm", ".xltx", ".xltm")
_MATCH_MODES: tuple[str, ...] = ("trip_id", "route_start_time")
_DAY_TYPES: tuple[str, ...] = (
    "weekday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
# Below this share of ridership rows matching an analysis-day trip, the unmatched-row
# warning adds a hint that the file may describe another day type or use other ids.
_RIDERSHIP_MATCH_WARN_SHARE: float = 0.5
# A direction name whose winning direction_id carries less than this share of its
# unambiguous rows is reported as an uncertain mapping.
_DIRECTION_VOTE_WARN_SHARE: float = 0.8
# Below this share of rows whose block matched a GTFS block_id, warn that the two
# block numberings may differ.
_BLOCK_MATCH_WARN_SHARE: float = 0.5
# Ridership match statuses, as written to the match table.
_MS_MATCHED: str = "matched"
_MS_BAD_BOARDINGS: str = "invalid or negative boardings"
_MS_BLANK_TRIP: str = "blank trip_id"
_MS_OFF_DAY: str = "trip_id not on analysis day"
_MS_UNKNOWN_ROUTE: str = "route not in routes.txt"
_MS_BAD_TIME: str = "unreadable start time"
_MS_NO_TRIP: str = "no trip on route within tolerance"
_MS_WRONG_DIRECTION: str = "no trip in mapped direction within tolerance"
_MS_TAKEN: str = "nearest trips claimed by closer rows"
_MS_SUPERSEDED: str = "whole-trip row superseded by per-stop rows"
_NO_BLOCK_PREFIX: str = "(no block) "
_ALL_DAY: str = "ALL DAY"

# =============================================================================
# REUSABLE FUNCTIONS (verbatim copies of utils/; CI audits them for drift)
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


def service_ids_active_on(
    active_dates: Mapping[str, set[dt.date]],
    target_date: dt.date,
) -> set[str]:
    """Return the service_ids operating on *target_date*.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        target_date: The calendar date to query.

    Returns:
        Set of service_id strings active on that date (possibly empty).
    """
    return {sid for sid, dates in active_dates.items() if target_date in dates}


def representative_service_date(
    active_dates: Mapping[str, set[dt.date]],
    service_day: str,
    override_date: Optional[dt.date] = None,
    exclude_dates: Optional[set[dt.date]] = None,
) -> tuple[dt.date, set[str]]:
    """Pick a typical date for *service_day* and the service_ids active on it.

    Rather than trusting any single date or unioning every service whose
    columns mention a weekday (which double-counts agencies running distinct
    Monday / midweek / Friday schedules), this scans every candidate date of
    the requested day type, groups them by their exact set of active
    service_ids, and returns the median date of the **modal** (most common)
    set. A few miscoded dates therefore cannot steer the result, and per-day
    math (headways, spans, trip counts) reflects one real operating day.

    Warnings are logged when the choice is ambiguous: when the modal set
    covers under half the candidate dates, and — for ``"weekday"`` — when
    Monday-through-Friday do not all share one service pattern, so the user
    knows a single representative day cannot speak for the whole week.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        service_day: ``"weekday"`` or one of ``"monday"`` … ``"sunday"``.
        override_date: Skip selection entirely and use this date (the
            explicit user override). Logged, and a warning is emitted if no
            service is active on it.
        exclude_dates: Dates to skip as candidates — typically observed
            holidays, so a holiday cannot masquerade as a typical day.

    Returns:
        Tuple of (chosen date, set of service_id strings active on it).

    Raises:
        ValueError: If *service_day* is not recognised, or no candidate
            dates exist for it.
    """
    day_names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    key = service_day.strip().lower()
    if key == "weekday":
        allowed = {0, 1, 2, 3, 4}
    elif key in day_names:
        allowed = {day_names.index(key)}
    else:
        raise ValueError(
            f"service_day must be 'weekday' or one of {', '.join(day_names)}; got {service_day!r}"
        )

    if override_date is not None:
        ids = service_ids_active_on(active_dates, override_date)
        if not ids:
            logging.warning("No service is active on override date %s.", override_date)
        else:
            logging.info(
                "Using override date %s (%s): %d service_id(s).",
                override_date,
                day_names[override_date.weekday()],
                len(ids),
            )
        return override_date, ids

    skip = exclude_dates or set()
    candidates = sorted(
        {d for dates in active_dates.values() for d in dates if d.weekday() in allowed} - skip
    )
    if not candidates:
        raise ValueError(f"No active dates found for service_day={service_day!r}.")

    by_set: dict[frozenset[str], list[dt.date]] = {}
    for d in candidates:
        by_set.setdefault(frozenset(service_ids_active_on(active_dates, d)), []).append(d)

    modal_ids = max(by_set, key=lambda ids: (len(by_set[ids]), -min(by_set[ids]).toordinal()))
    modal_dates = by_set[modal_ids]
    chosen = modal_dates[len(modal_dates) // 2]
    share = len(modal_dates) / len(candidates)

    if key == "weekday":
        per_day: dict[str, frozenset[str]] = {}
        for dow in sorted({d.weekday() for d in candidates}):
            day_candidates = [d for d in candidates if d.weekday() == dow]
            day_sets: dict[frozenset[str], int] = {}
            for d in day_candidates:
                s = frozenset(service_ids_active_on(active_dates, d))
                day_sets[s] = day_sets.get(s, 0) + 1
            per_day[day_names[dow]] = max(day_sets, key=lambda s: day_sets[s])
        if len(set(per_day.values())) > 1:
            detail = "; ".join(
                f"{day}={sorted(ids) if ids else '{}'}" for day, ids in per_day.items()
            )
            logging.warning(
                "Weekday service varies by day of week (%s). Using %s (%s) as the "
                "representative weekday; pass an explicit service date to analyse "
                "a different day.",
                detail,
                chosen,
                day_names[chosen.weekday()],
            )
    if share < 0.5:
        logging.warning(
            "The chosen service pattern covers only %.0f%% of candidate %s dates — "
            "this feed's %s service is irregular; consider an explicit service date.",
            share * 100,
            service_day,
            service_day,
        )

    logging.info(
        "Representative %s: %s (%s) with %d service_id(s), matching %.0f%% of candidate dates.",
        service_day,
        chosen,
        day_names[chosen.weekday()],
        len(modal_ids),
        share * 100,
    )
    return chosen, set(modal_ids)


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


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class RidershipSpec(NamedTuple):
    """Where the ridership file keeps each field, and how its rows map to trips."""

    match_mode: str = RIDERSHIP_TRIP_MATCH_MODE
    trip_col: str = RIDERSHIP_TRIP_ID_COL
    boardings_col: str = RIDERSHIP_BOARDINGS_COL
    route_col: str = RIDERSHIP_ROUTE_COL
    start_time_col: str = RIDERSHIP_START_TIME_COL
    tolerance_min: int = RIDERSHIP_START_TIME_TOLERANCE_MIN
    stop_col: str = RIDERSHIP_STOP_ID_COL
    date_col: str = RIDERSHIP_DATE_COL
    direction_col: str = RIDERSHIP_DIRECTION_COL
    block_col: str = RIDERSHIP_BLOCK_COL
    obs_col: str = RIDERSHIP_OBS_COL
    low_obs_share: float = LOW_OBSERVATION_SHARE


class RidershipResult(NamedTuple):
    """One day type's ridership, as matched to that day's trips.

    Attributes:
        boardings: Average daily boardings per matched trip_id.
        observations: Observation count per matched trip_id (empty when the file
            has no observation column).
        match_table: Every input row with its matched trip_id or the reason it
            matched none.
        rows_in: Rows in the file.
        rows_matched: Rows that matched an analysis-day trip.
    """

    boardings: pd.Series
    observations: pd.Series
    match_table: pd.DataFrame
    rows_in: int
    rows_matched: int


class BlockRules(NamedTuple):
    """How gaps between a block's trips turn into vehicle time."""

    min_release_gap_min: int = MIN_RELEASE_GAP_MIN
    pull_deadhead_min: int = PULL_DEADHEAD_MIN
    model: Optional[DeadheadModel] = None


class Guards(NamedTuple):
    """Optional limits on which cuts may be selected."""

    protect_trips_above: float = PROTECT_TRIPS_ABOVE
    max_resulting_headway_min: int = MAX_RESULTING_HEADWAY_MIN
    protect_route_span: bool = PROTECT_ROUTE_SPAN


class Config(NamedTuple):
    """Resolved settings for one run."""

    gtfs_path: str
    ridership_paths: Mapping[str, str]
    output_dir: Path
    service_dates: Mapping[str, str] = SERVICE_DATES
    tods_path: str = TODS_PATH
    ridership: RidershipSpec = RidershipSpec()
    rules: BlockRules = BlockRules()
    guards: Guards = Guards()
    routes_to_include: RouteListSource = ROUTES_TO_INCLUDE
    routes_to_exclude: RouteListSource = ROUTES_TO_EXCLUDE
    time_periods: Mapping[str, tuple[str, str]] = TIME_PERIODS
    recovery_target_pct: float = RECOVERY_TARGET_PCT
    vehicle_profile_bin_min: int = VEHICLE_PROFILE_BIN_MIN
    route_exclusion_groups: Mapping[str, RouteListSource] = ROUTE_EXCLUSION_GROUPS
    non_revenue_routes: RouteListSource = NON_REVENUE_ROUTES
    excellent_max_boardings_per_vehicle_hour_saved: float = (
        EXCELLENT_MAX_BOARDINGS_PER_VEHICLE_HOUR_SAVED
    )
    good_max_boardings_per_vehicle_hour_saved: float = GOOD_MAX_BOARDINGS_PER_VEHICLE_HOUR_SAVED
    headway_sessions: Sequence[Mapping[str, Any]] = HEADWAY_SESSIONS
    session_preserve_first_last: bool = SESSION_PRESERVE_FIRST_LAST
    session_headway_tolerance_min: int = SESSION_HEADWAY_TOLERANCE_MIN
    shortlist_limit_per_type: int = SHORTLIST_LIMIT_PER_TYPE
    headway_search: Mapping[str, Any] = HEADWAY_SEARCH
    session_shortlist_overlap: float = SESSION_SHORTLIST_OVERLAP
    linked_route_mode: str = LINKED_ROUTE_MODE
    linked_route_min_shared_blocks: int = LINKED_ROUTE_MIN_SHARED_BLOCKS
    linked_route_groups: Mapping[str, RouteListSource] = LINKED_ROUTE_GROUPS
    session_max_phase_combinations: int = SESSION_MAX_PHASE_COMBINATIONS
    export_proposal_workbooks: bool = EXPORT_PROPOSAL_WORKBOOKS
    workbook_option_ids: Sequence[str] = WORKBOOK_OPTION_IDS
    workbook_limit_per_type: int = WORKBOOK_LIMIT_PER_TYPE
    workbook_data_label: str = WORKBOOK_DATA_LABEL


class HeadwaySession(NamedTuple):
    """One requested frequency reduction; departure times and blocks stay fixed."""

    name: str
    start: int
    end: int
    from_headway: int
    to_headway: int
    routes: tuple[str, ...]
    days: tuple[str, ...]


def parse_headway_sessions(sources: Sequence[Mapping[str, Any]]) -> list[HeadwaySession]:
    """Validate regular-service thinning requests; times use the GTFS service-day clock."""
    sessions = []
    seen: set[str] = set()
    fields = {"name", "start", "end", "from_headway_min", "to_headway_min", "routes", "days"}
    for source in sources:
        if not isinstance(source, Mapping) or set(source) - fields:
            raise ValueError(
                "HEADWAY_SESSIONS entries must be dictionaries using the documented fields."
            )
        name = str(source.get("name", "")).strip()
        if not name or name in seen:
            raise ValueError("HEADWAY_SESSIONS requires a unique, nonempty name for each window.")
        seen.add(name)
        start = parse_time_to_minutes(str(source.get("start", "")))
        end = parse_time_to_minutes(str(source.get("end", "")))
        if start is None or end is None or start < 0 or end <= start or end - start > 1440:
            raise ValueError(
                f"Session {name}: require start < end within one service day; use 25:00 after "
                "midnight."
            )
        before, after = source.get("from_headway_min"), source.get("to_headway_min")
        if (
            type(before) is not int
            or type(after) is not int
            or before <= 0
            or after <= before
            or after % before
            or after // before > 12
        ):
            raise ValueError(
                f"Session {name}: headways must be positive whole minutes; target must be 2-12 "
                "times the original."
            )
        lists = []
        for field in ("routes", "days"):
            values = source.get(field, [])
            if (
                isinstance(values, str)
                or not isinstance(values, Sequence)
                or any(not isinstance(v, str) for v in values)
            ):
                raise ValueError(f"Session {name}: {field} must be a list of quoted strings.")
            lists.append(tuple(dict.fromkeys(v.strip() for v in values if v.strip())))
        routes, days = lists
        if set(days) - set(_DAY_TYPES):
            raise ValueError(f"Session {name}: unrecognized day type in {days}.")
        sessions.append(HeadwaySession(name, start, end, before, after, routes, days))
    return sessions


class ResolvedRouteFilters(NamedTuple):
    """Route IDs/names and source fingerprints loaded once for an entire run."""

    include: tuple[str, ...]
    exclude: tuple[str, ...]
    non_revenue: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    fingerprints: tuple[str, ...]
    linked_groups: Mapping[str, tuple[str, ...]] = {}


def read_route_list(
    source: RouteListSource, name: str, base_dir: Path
) -> tuple[tuple[str, ...], Optional[str]]:
    """Read a list or UTF-8 text path, retaining textual IDs and stable order.

    Text files contain one route per line, with optional blank lines and full-line
    ``#`` comments. UTF-8 BOMs are accepted. A scalar string is always a file path;
    even a single inline route must be inside a list. Returns the routes and an
    optional fingerprint of the exact file bytes that were parsed.

    Raises:
        OSError: If the configured file cannot be read.
        ValueError: If the text is not UTF-8 or an inline entry is not a string.
    """
    fingerprint: Optional[str] = None
    if isinstance(source, (str, Path)):
        if isinstance(source, str) and not source.strip():
            return (), None
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise OSError(f"Cannot read route list {name}: {path}: {exc}") from exc
        try:
            lines = raw.decode("utf-8-sig").splitlines()
        except UnicodeError as exc:
            raise ValueError(f"Route list {name} must be UTF-8 text: {path}") from exc
        entries = [line.strip() for line in lines if not line.lstrip().startswith("#")]
        fingerprint = (
            f"route_list/{name}  {len(raw):,} bytes  "
            f"sha256={hashlib.sha256(raw).hexdigest()}  {path}"
        )
        logging.info("Loaded %s from %s.", name, path)
    else:
        if not isinstance(source, Sequence) or any(not isinstance(v, str) for v in source):
            raise ValueError(f"{name} must be a list of quoted route IDs/names or a .txt path.")
        entries = [v.strip() for v in source]
    routes = tuple(dict.fromkeys(v for v in entries if v))
    logging.info("%s: %d unique routes: %s", name, len(routes), list(routes))
    return routes, fingerprint


def resolve_route_filters(cfg: Config) -> ResolvedRouteFilters:
    """Resolve all route-list sources before feed loading or output writes."""
    script = globals().get("__file__")
    base_dir = Path(script).resolve().parent if script else Path.cwd()
    fingerprints: list[str] = []

    def read(source: RouteListSource, name: str) -> tuple[str, ...]:
        routes, fingerprint = read_route_list(source, name, base_dir)
        if fingerprint:
            fingerprints.append(fingerprint)
        return routes

    include = read(cfg.routes_to_include, "ROUTES_TO_INCLUDE")
    additional = read(cfg.routes_to_exclude, "ROUTES_TO_EXCLUDE")
    non_revenue = read(cfg.non_revenue_routes, "NON_REVENUE_ROUTES")
    groups = {
        name: read(source, f"ROUTE_EXCLUSION_GROUPS[{name!r}]")
        for name, source in cfg.route_exclusion_groups.items()
    }
    exclude = tuple(
        dict.fromkeys([*additional, *non_revenue, *(v for group in groups.values() for v in group)])
    )
    linked = {}
    for name, source in cfg.linked_route_groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("LINKED_ROUTE_GROUPS requires nonempty string names.")
        values = read(source, f"LINKED_ROUTE_GROUPS[{name!r}]")
        if len(values) < 2:
            raise ValueError(f"Linked group {name!r} requires at least two distinct routes.")
        linked[name] = values
    return ResolvedRouteFilters(include, exclude, non_revenue, groups, tuple(fingerprints), linked)


class RunCut(NamedTuple):
    """What removing trips ``i``..``j`` (inclusive) of a block would do."""

    i: int
    j: int
    cut_type: str
    revenue_min: int
    vehicle_min_saved: int
    gap_left_min: Optional[int]
    feasible: bool = True


class BlockState:
    """One block's remaining trips, in start order, as cuts are applied.

    Attributes:
        block_id: The GTFS block_id (or a synthetic id for a trip without one).
        trip_ids: Remaining trip_ids in start order.
        starts: First departure of each remaining trip, minutes past midnight.
        ends: Last arrival of each remaining trip, minutes past midnight.
        version: Incremented on every cut, so stale candidates can be recognized.
    """

    def __init__(
        self, block_id: str, trip_ids: List[str], starts: List[int], ends: List[int]
    ) -> None:
        """Hold one block's trips; the three lists are parallel and start-ordered."""
        self.block_id = block_id
        self.trip_ids = list(trip_ids)
        self.starts = list(starts)
        self.ends = list(ends)
        self.version = 0

    def remove_run(self, i: int, j: int) -> List[str]:
        """Drop trips ``i``..``j`` and return their trip_ids."""
        removed = self.trip_ids[i : j + 1]
        del self.trip_ids[i : j + 1]
        del self.starts[i : j + 1]
        del self.ends[i : j + 1]
        self.version += 1
        return removed


class TripLookup(NamedTuple):
    """Per-trip attributes the screening code reads many times, as plain dicts."""

    route_dir: dict[str, tuple[str, str]]
    label: dict[str, str]


# =============================================================================
# GTFS LOADING
# =============================================================================


def load_feed(gtfs_path: str) -> dict[str, Optional[pd.DataFrame]]:
    """Load the GTFS tables this analysis uses.

    Args:
        gtfs_path: GTFS folder or .zip path.

    Blank cells load as ``""`` rather than NaN, so ids and names compare as text
    throughout.

    Returns:
        Mapping with ``routes``, ``trips``, ``stop_times``, ``stops`` (always present)
        and ``calendar``, ``calendar_dates``, ``frequencies`` (``None`` when absent).
    """
    required = load_gtfs_data(
        gtfs_path, files=("routes.txt", "trips.txt", "stop_times.txt", "stops.txt")
    )
    feed: dict[str, Optional[pd.DataFrame]] = {k: v.fillna("") for k, v in required.items()}
    for optional in ("calendar", "calendar_dates", "frequencies"):
        try:
            table = load_gtfs_data(gtfs_path, files=(f"{optional}.txt",))[optional]
            feed[optional] = table.fillna("")
        except (OSError, ValueError):
            feed[optional] = None
    freq = feed.get("frequencies")
    if freq is not None and not freq.empty:
        logging.warning(
            "frequencies.txt is present (%d rows) but is NOT expanded — frequency-based "
            "service is missing from every block and total.",
            len(freq),
        )
    return feed


def select_service_ids(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    service_day: str,
    service_date: str,
) -> tuple[Optional[dt.date], Optional[set[str]]]:
    """Choose the representative analysis date and its active service_ids.

    Args:
        calendar: Parsed ``calendar.txt`` or ``None``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.
        service_day: ``"weekday"`` or a single day name.
        service_date: Optional explicit ``YYYY-MM-DD`` date; overrides selection.

    Returns:
        Tuple of (chosen date, active service_ids); both ``None`` when the feed
        has no usable calendar, in which case callers keep every trip.

    Raises:
        ValueError: If *service_date* is set but not parseable as ``YYYY-MM-DD``.
    """
    if (calendar is None or calendar.empty) and (calendar_dates is None or calendar_dates.empty):
        logging.warning(
            "Feed has no calendar.txt or calendar_dates.txt — keeping all trips, so every "
            "service day is pooled into one."
        )
        return None, None

    active = expand_service_active_dates(calendar, calendar_dates)
    if not any(active.values()):
        logging.warning("No service has any active dates — keeping all trips.")
        return None, None

    override: Optional[dt.date] = None
    if service_date.strip():
        try:
            override = dt.datetime.strptime(service_date.strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"SERVICE_DATES value must be YYYY-MM-DD; got {service_date!r}."
            ) from exc

    all_dates = {d for dates in active.values() for d in dates}
    holidays: set[dt.date] = set()
    for year in range(min(all_dates).year, max(all_dates).year + 2):
        holidays |= federal_holidays_observed(year)

    chosen, service_ids = representative_service_date(
        active, service_day, override_date=override, exclude_dates=holidays
    )
    return chosen, service_ids


def route_labels(routes: pd.DataFrame) -> dict[str, str]:
    """Map route_id to the label planners use: route_short_name, else route_id."""
    r = routes.drop_duplicates(subset=["route_id"]).copy()
    r["route_id"] = r["route_id"].astype(str).str.strip()
    short = r.get("route_short_name", pd.Series("", index=r.index)).fillna("").astype(str)
    short = short.str.strip()
    return {rid: (s if s else rid) for rid, s in zip(r["route_id"], short)}


def build_trip_table(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes: pd.DataFrame,
    service_ids: Optional[set[str]],
) -> pd.DataFrame:
    """Reduce the analysis day's trips to one row each with start, end, and block.

    Args:
        trips: Parsed ``trips.txt``.
        stop_times: Parsed ``stop_times.txt``.
        routes: Parsed ``routes.txt``.
        service_ids: Active service_ids (``None`` keeps every trip).

    Returns:
        DataFrame indexed by ``trip_id`` with ``route_id``, ``route_label``,
        ``direction_id``, ``block_id`` (``""`` when missing), ``start_min``,
        ``end_min``, and ``runtime_min``, sorted by block and start.

    Raises:
        ValueError: If trips.txt has no ``block_id`` values, or no trip with
            usable times remains on the analysis day.
    """
    if "block_id" not in trips.columns or trips["block_id"].fillna("").str.strip().eq("").all():
        raise ValueError(
            "trips.txt has no block_id values. This analysis needs vehicle blocks to tell "
            "which cuts free vehicle time; use a feed exported with blocking."
        )
    t = trips.drop_duplicates(subset=["trip_id"]).copy()
    for col in ("trip_id", "route_id", "service_id", "block_id"):
        t[col] = t[col].fillna("").astype(str).str.strip()
    t["direction_id"] = (
        t["direction_id"].fillna("").astype(str).str.strip() if "direction_id" in t else ""
    )
    if service_ids is not None:
        t = t[t["service_id"].isin(service_ids)]
        if t.empty:
            raise ValueError(
                "No trips remain on the analysis day's service_ids. Check the RIDERSHIP_PATHS "
                "day type / SERVICE_DATES against the feed's calendar."
            )

    if trips.trip_id.duplicated().any():
        raise ValueError("Duplicate GTFS trip_id; resolve before analysis.")
    st = stop_times[
        ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]
    ].copy()
    st["trip_id"] = st["trip_id"].astype(str).str.strip()
    st = st[st["trip_id"].isin(set(t["trip_id"]))]
    dep = st["departure_time"].map(parse_time_to_minutes)
    arr = st["arrival_time"].map(parse_time_to_minutes)
    st["dep_min"] = dep.where(dep.notna(), arr)
    st["arr_min"] = arr.where(arr.notna(), dep)
    st["_seq"] = pd.to_numeric(st["stop_sequence"], errors="raise")
    if st.duplicated(["trip_id", "_seq"]).any() or (st._seq < 0).any() or (st._seq % 1 != 0).any():
        raise ValueError("Invalid or duplicate GTFS trip_id/stop_sequence.")
    st = st.sort_values(["trip_id", "_seq"])
    first = st.groupby("trip_id", sort=False).head(1).set_index("trip_id")
    last = st.groupby("trip_id", sort=False).tail(1).set_index("trip_id")
    spans = pd.DataFrame(
        {
            "start_min": first.dep_min,
            "end_min": last.arr_min,
            "start_stop_id": first.stop_id,
            "end_stop_id": last.stop_id,
        }
    ).dropna()
    if (spans.end_min < spans.start_min).any():
        raise ValueError("GTFS trip ends before it starts; use service-day times beyond 24:00.")

    table = t.set_index("trip_id")[["route_id", "direction_id", "block_id", "service_id"]].join(
        spans, how="inner"
    )
    dropped = len(t) - len(table)
    if dropped:
        logging.warning("Dropped %d analysis-day trip(s) with no parseable stop times.", dropped)
    if table.empty:
        raise ValueError("No analysis-day trip has parseable stop_times.")
    table["start_min"] = table["start_min"].astype(int)
    table["end_min"] = table["end_min"].astype(int)
    table["runtime_min"] = (table["end_min"] - table["start_min"]).clip(lower=0)
    # Session thinning must not silently combine branches/short turns with the same route label.
    patterns = st.groupby("trip_id", sort=False)["stop_id"].agg(list)
    table["pattern_key"] = patterns.map(
        lambda stops: hashlib.sha256(
            json.dumps(stops, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    labels = route_labels(routes)
    table["route_label"] = table["route_id"].map(labels).fillna(table["route_id"])
    no_block = int(table["block_id"].eq("").sum())
    if no_block:
        logging.warning(
            "%d analysis-day trip(s) have no block_id. Each counts as its own block in the "
            "baseline totals, but none is eligible to cut — its savings cannot be checked.",
            no_block,
        )
    table.index.name = "trip_id"
    return table.sort_values(["block_id", "start_min", "end_min"])


# =============================================================================
# RIDERSHIP
# =============================================================================


def _cell_to_text(value: Any) -> str:
    """Render one spreadsheet cell as table text.

    Whole-number floats collapse to their integer form because Excel stores
    every number as a float — otherwise a numeric ``trip_id`` arrives as
    ``"4821.0"`` and joins against nothing. Missing cells become ``""``.
    """
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _read_table(path: Path, label: str) -> pd.DataFrame:
    """Read a ridership table as strings, whether it is a CSV or an Excel workbook.

    The format is decided by file content first (the OLE2 header of ``.xls``, the
    zip header of ``.xlsx``) and by extension second, so a workbook saved with a
    ``.csv`` name still reads. Text files are tried as UTF-8 (tolerating Excel's
    BOM) and then as cp1252, the usual Windows export encoding.

    Args:
        path: File to read.
        label: Human-readable name of the input, used in error messages.

    Returns:
        DataFrame of text values.

    Raises:
        ValueError: If the file is neither readable text nor a workbook the
            installed Excel engines can open.
    """
    with open(path, "rb") as handle:
        head = handle.read(8)
    suffix = path.suffix.lower()

    excel_engine: Optional[str] = None
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        excel_engine = "xlrd"
    elif head.startswith(b"PK\x03\x04"):
        excel_engine = "openpyxl"
    elif suffix in _EXCEL_SUFFIXES:
        excel_engine = "xlrd" if suffix == ".xls" else "openpyxl"

    if excel_engine is not None:
        logging.info("%s: reading %s as an Excel workbook.", label, path.name)
        try:
            frame = pd.read_excel(path, engine=excel_engine)
        except ImportError as exc:
            raise ValueError(
                f"{label} {path} is an Excel workbook, but the {excel_engine} engine is not "
                f"installed. Install it, or re-save the file as a CSV and re-run."
            ) from exc
        for column in frame.columns:
            frame[column] = frame[column].map(_cell_to_text)
        return frame

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"{label} {path} is not readable as UTF-8 or Windows-1252 text, and is not an Excel "
        f"workbook. Re-save it from Excel as 'CSV UTF-8 (Comma delimited)' and re-run."
    )


def _as_excel_serial(value: Any, text: str) -> Optional[int]:
    """Read an Excel date serial as minutes past midnight, if it holds a time.

    A whole number carries no time of day (a date-only cell format lost it), so it
    is reported as unreadable rather than silently taken for midnight.

    Args:
        value: The raw cell value.
        text: Its stripped text form.

    Returns:
        Minutes past midnight, or ``None`` when the value is not a serial or carries
        no time of day.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(text)
        except ValueError:
            return None
    fraction = number % 1.0
    if fraction == 0.0:
        return None
    return int(round(fraction * 1440)) % 1440


def parse_clock_value(value: Any) -> Optional[int]:
    """Read a trip start time from whatever the export stored, as minutes.

    Handles ``datetime``/``time`` values, Excel serials, ``"HH:MM[:SS]"`` (keeping
    GTFS-style hours past 24), and other forms pandas can parse such as
    ``"7:23:00 AM"``. Exact midnight is rejected: a column whose time was lost to a
    date-only cell format arrives as all-midnight, a far more common mistake than a
    genuine 00:00 departure (which ``trip_id`` matching handles).

    Args:
        value: The raw cell value.

    Returns:
        Minutes past midnight, or ``None`` when missing or uninterpretable.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    minutes: Optional[int]
    if isinstance(value, (dt.datetime, dt.time)):
        minutes = value.hour * 60 + value.minute + round(value.second / 60)
    elif pd.isna(value):
        return None
    else:
        text = str(value).strip()
        minutes = parse_time_to_minutes(text)
        if minutes is None:
            minutes = _as_excel_serial(value, text)
        if minutes is None:
            try:
                stamp = pd.to_datetime(text, errors="raise")
            except (ValueError, TypeError, pd.errors.ParserError):
                return None
            if pd.isna(stamp):
                return None
            minutes = int(stamp.hour) * 60 + int(stamp.minute) + round(int(stamp.second) / 60)
    return minutes if minutes else None


def _norm_block(value: Any) -> str:
    """Normalize a block id for comparison.

    Trims whitespace and drops leading zeros from an all-digit id, so a GTFS
    ``"01612"`` equals a Ridecheck ``1612``. Other ids compare as trimmed text.
    """
    text = str(value).strip()
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def match_trips_by_route_and_time(
    frame: pd.DataFrame,
    trip_table: pd.DataFrame,
    routes: pd.DataFrame,
    spec: RidershipSpec,
) -> pd.DataFrame:
    """Match ridership rows one-to-one to analysis-day trips on route and first departure.

    A row's candidates are the trips of its route whose first departure lies within
    ``spec.tolerance_min`` of its start time; a small-hours time is also tried a
    service day later (1:30 AM matches a GTFS 25:30 departure). Two refinements
    narrow the candidates when their columns are set:

    - Direction: each direction name is mapped to a GTFS ``direction_id`` per route,
      by majority over the rows whose candidates all share one direction (their
      start time leaves no doubt which way they ran). A row whose name is mapped
      keeps only candidates in that direction; a name with no such row stays
      unmapped and its rows match without direction.
    - Block: when a candidate's ``block_id`` equals the row's block, only candidates
      in that block are kept.

    Rows that are parts of one trip and agree in route, time, direction name, and
    block (per-stop rows, or per-date rows when a date column is set) form one
    group; every whole-trip row is a group of its own. Groups are assigned to
    trips one-to-one: block-confirmed pairs first, then nearest departure.
    Ties and conflicting equally supported claims are left unmatched; a rejected
    row is not assigned to its next-nearest trip.

    Args:
        frame: Ridership rows with numeric boardings.
        trip_table: Output of :func:`build_trip_table`.
        routes: Parsed ``routes.txt`` (route_short_name or route_id tokens).
        spec: Column names and tolerance.

    Returns:
        DataFrame aligned to ``frame.index`` with ``_trip`` (``""`` when unmatched),
        ``_match_status``, ``_match_delta_min``, ``_matched_on_block``, and
        ``_mapped_direction_id``.
    """
    r = routes.drop_duplicates(subset=["route_id"]).copy()
    r["route_id"] = r["route_id"].astype(str).str.strip()
    short = r.get("route_short_name", pd.Series("", index=r.index)).fillna("").astype(str)
    token_to_id = dict(zip(r["route_id"], r["route_id"]))
    token_to_id.update({s.strip(): rid for s, rid in zip(short, r["route_id"]) if s.strip()})
    labels = route_labels(routes)

    by_route: dict[str, list[tuple[int, str, str, str]]] = {}
    for trip_id, row in trip_table.iterrows():
        by_route.setdefault(str(row["route_id"]), []).append(
            (
                int(row["start_min"]),
                str(trip_id),
                str(row["direction_id"]),
                _norm_block(row["block_id"]),
            )
        )

    dir_col = spec.direction_col.strip()
    block_col = spec.block_col.strip()
    blank = pd.Series("", index=frame.index, dtype=object)
    dir_values = frame[dir_col] if dir_col else blank
    block_values = frame[block_col] if block_col else blank
    # Rows that are parts of one trip (per-stop rows, or per-date rows when a date
    # column is set) share a key; every other row is its own key, so two whole-trip
    # rows that look alike still go to two different trips.
    stop_col = spec.stop_col.strip()
    part_of_trip = (
        pd.Series(True, index=frame.index)
        if spec.date_col.strip()
        else (
            frame[stop_col].fillna("").astype(str).str.strip().ne("")
            if stop_col and stop_col in frame.columns
            else pd.Series(False, index=frame.index)
        )
    )

    # 1. Key rows by (route_id, start minutes, direction name, block, own-row id).
    row_status: dict[Any, str] = {}
    rows_of_key: dict[tuple[str, int, str, str, Any], list[Any]] = {}
    unknown_routes: set[str] = set()
    for idx, token, raw_time, name, block, shared in zip(
        frame.index,
        frame[spec.route_col],
        frame[spec.start_time_col],
        dir_values,
        block_values,
        part_of_trip,
    ):
        route_id = token_to_id.get(str(token).strip())
        if route_id is None:
            unknown_routes.add(str(token).strip())
            row_status[idx] = _MS_UNKNOWN_ROUTE
            continue
        minutes = parse_clock_value(raw_time)
        if minutes is None:
            row_status[idx] = _MS_BAD_TIME
            continue
        own = None if shared else idx
        key = (route_id, minutes, str(name).strip().upper(), _norm_block(block), own)
        rows_of_key.setdefault(key, []).append(idx)

    def candidates(route_id: str, minutes: int) -> list[tuple[int, str, str, str]]:
        found = []
        for start, tid, direction, block in by_route.get(route_id, []):
            delta = min(abs(start - minutes), abs(start - (minutes + 1440)))
            if delta <= spec.tolerance_min:
                found.append((delta, tid, direction, block))
        return found

    def narrow_to_block(
        found: list[tuple[int, str, str, str]], block: str
    ) -> tuple[list[tuple[int, str, str, str]], bool]:
        if not block:
            return found, False
        same = [c for c in found if c[3] == block]
        return (same, True) if same else (found, False)

    def describe(route_id: str, name: str) -> str:
        return f"{labels.get(route_id, route_id)} {name}".strip()

    # 2. Learn each route's direction names from the rows whose time fits one direction.
    direction_map: dict[tuple[str, str], str] = {}
    if dir_col:
        votes: dict[tuple[str, str], dict[str, int]] = {}
        for route_id, minutes, name, block, _ in rows_of_key:
            if not name:
                continue
            found, _ = narrow_to_block(candidates(route_id, minutes), block)
            directions = {c[2] for c in found}
            if len(directions) == 1:
                tally = votes.setdefault((route_id, name), {})
                direction = directions.pop()
                tally[direction] = tally.get(direction, 0) + 1
        doubtful: list[str] = []
        for (route_id, name), tally in sorted(votes.items()):
            ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
            counts = ", ".join(f"direction_id {d or '(blank)'} x{n}" for d, n in ranked)
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                doubtful.append(f"{describe(route_id, name)} [{counts}; tied, left unmapped]")
                continue
            if ranked[0][1] / sum(tally.values()) < _DIRECTION_VOTE_WARN_SHARE:
                doubtful.append(f"{describe(route_id, name)} [{counts}; left unmapped]")
            else:
                direction_map[(route_id, name)] = ranked[0][0]
        names_by_direction: dict[tuple[str, str], list[str]] = {}
        for (route_id, name), direction in direction_map.items():
            names_by_direction.setdefault((route_id, direction), []).append(name)
        for (route_id, direction), names in sorted(names_by_direction.items()):
            if len(names) > 1 and len({c[2] for c in by_route.get(route_id, [])}) > 1:
                doubtful.append(
                    f"{labels.get(route_id, route_id)} {' and '.join(sorted(names))} "
                    f"[both map to direction_id {direction or '(blank)'}]"
                )
        unvoted = sorted(
            {
                describe(route_id, name)
                for route_id, _, name, _, _ in rows_of_key
                if name and (route_id, name) not in votes
            }
        )
        logging.info(
            "Ridership: mapped %d route direction name(s) to GTFS direction_id.",
            len(direction_map),
        )
        if doubtful:
            logging.warning(
                "Ridership: %d direction mapping(s) are uncertain (rows that fit one "
                "direction, by direction_id): %s%s. Check mapped_direction_id in the "
                "ridership match table.",
                len(doubtful),
                "; ".join(doubtful[:10]),
                " ..." if len(doubtful) > 10 else "",
            )
        if unvoted:
            logging.warning(
                "Ridership: %d route direction name(s) have no row whose start time fits "
                "a single direction, so their rows require unambiguous candidates or manual "
                "direction overrides: %s%s.",
                len(unvoted),
                ", ".join(unvoted[:10]),
                " ..." if len(unvoted) > 10 else "",
            )

    for (route_token, name), direction in DIRECTION_OVERRIDES.items():
        route_id = token_to_id.get(str(route_token).strip())
        if route_id is None:
            raise ValueError(f"Unknown route in DIRECTION_OVERRIDES: {route_token}")
        known = {entry[2] for entry in by_route.get(route_id, [])}
        if known and str(direction) not in known:
            raise ValueError(f"Invalid direction override: {route_token}, {name}, {direction}")
        direction_map[(route_id, name.strip().upper())] = str(direction)

    # 3. Narrow each group's candidates, then assign groups to trips one-to-one.
    keys = list(rows_of_key)
    key_status: dict[int, str] = {}
    key_direction: dict[int, str] = {}
    key_on_block: dict[int, bool] = {}
    first_choice: dict[int, str] = {}
    pairs: list[tuple[int, int, str, int]] = []
    for order, (route_id, minutes, name, block, _) in enumerate(keys):
        found = candidates(route_id, minutes)
        mapped = direction_map.get((route_id, name))
        key_direction[order] = mapped if mapped is not None else ""
        if not found:
            key_status[order] = _MS_NO_TRIP
            continue
        if mapped is not None:
            found = [c for c in found if c[2] == mapped]
            if not found:
                key_status[order] = _MS_WRONG_DIRECTION
                continue
        found, on_block = narrow_to_block(found, block)
        if mapped is None and len({c[2] for c in found}) > 1:
            key_status[order] = "ambiguous direction; supply DIRECTION_OVERRIDES"
            continue
        key_on_block[order] = on_block
        found.sort(key=lambda c: (c[0], c[1]))
        best_delta = found[0][0]
        nearest = [c for c in found if c[0] == best_delta]
        if len(nearest) != 1:
            key_status[order] = "ambiguous equally close trips"
            continue
        first_choice[order] = nearest[0][1]
        for delta, tid, _, _ in nearest:
            pairs.append((0 if on_block else 1, delta, tid, order))
    pairs.sort()
    # Equal-evidence claims cannot be resolved by row order or a trip_id tie-break.
    claims: dict[str, list[tuple[int, int, int]]] = {}
    for priority, delta, tid, order in pairs:
        claims.setdefault(tid, []).append((priority, delta, order))
    blocked: set[int] = set()
    for contenders in claims.values():
        best = min((p, d) for p, d, _ in contenders)
        tied = [o for p, d, o in contenders if (p, d) == best]
        if len(tied) > 1:
            for order in tied:
                key_status[order] = "ambiguous competing ridership rows"
            blocked.update(o for _, _, o in contenders)
    assigned: dict[int, tuple[str, int]] = {}
    taken: set[str] = set()
    for _, delta, tid, order in pairs:
        if order in blocked or order in assigned or tid in taken:
            continue
        assigned[order] = (tid, delta)
        taken.add(tid)

    per_row: dict[Any, tuple[str, str, Optional[int], Optional[bool], str]] = {
        idx: ("", status, None, None, "") for idx, status in row_status.items()
    }
    contested: list[str] = []
    contested_rows = 0
    moved_rows = 0
    for order, key in enumerate(keys):
        trip, delta, status = "", None, key_status.get(order, "")
        if not status:
            if order in assigned:
                trip, delta = assigned[order]
                status = _MS_MATCHED
                if trip != first_choice[order]:
                    moved_rows += len(rows_of_key[key])
            else:
                status = _MS_TAKEN
                contested_rows += len(rows_of_key[key])
                contested.append(f"{describe(key[0], key[2])} {minutes_to_hhmm(key[1])}")
        for idx in rows_of_key[key]:
            per_row[idx] = (trip, status, delta, key_on_block.get(order), key_direction[order])

    result = pd.DataFrame(
        [per_row[idx] for idx in frame.index],
        index=frame.index,
        columns=pd.Index(
            [
                "_trip",
                "_match_status",
                "_match_delta_min",
                "_matched_on_block",
                "_mapped_direction_id",
            ]
        ),
    )
    result["_match_delta_min"] = pd.to_numeric(result["_match_delta_min"]).astype("Int64")

    if unknown_routes:
        logging.warning(
            "Ridership: %d route value(s) match nothing in routes.txt (e.g. %s); their rows "
            "are ignored.",
            len(unknown_routes),
            ", ".join(sorted(unknown_routes)[:5]),
        )
    unparsed = sum(1 for status in row_status.values() if status == _MS_BAD_TIME)
    if unparsed:
        logging.warning(
            "Ridership: %d row(s) have an unreadable %r value and are ignored. An Excel "
            "time-only cell that reads as a date (1/0/1900) has lost its time of day; "
            "re-export that column as text (HH:MM).",
            unparsed,
            spec.start_time_col,
        )
    if block_col:
        with_block = [o for o, k in enumerate(keys) if k[3] and o in key_on_block]
        rows_with_block = sum(len(rows_of_key[keys[o]]) for o in with_block)
        rows_on_block = sum(len(rows_of_key[keys[o]]) for o in with_block if key_on_block[o])
        if rows_with_block:
            share = rows_on_block / rows_with_block
            level = logging.WARNING if share < _BLOCK_MATCH_WARN_SHARE else logging.INFO
            logging.log(
                level,
                "Ridership: %d of %d row(s) (%.0f%%) with a candidate trip had a %r value "
                "equal to a candidate's GTFS block_id.%s",
                rows_on_block,
                rows_with_block,
                100.0 * share,
                block_col,
                " The two block numberings may differ; those rows fell back to route, "
                "time, and direction."
                if level == logging.WARNING
                else "",
            )
    if contested:
        logging.warning(
            "Ridership: %d row(s) found every candidate trip claimed by a closer row and "
            "stay unmatched, e.g. %s.",
            contested_rows,
            ", ".join(contested[:5]),
        )
    if moved_rows:
        logging.info(
            "Ridership: %d row(s) lost their nearest trip to a closer row and took their "
            "next-nearest trip instead.",
            moved_rows,
        )
    ambiguous = result["_match_status"].str.startswith("ambiguous").sum()
    if ambiguous:
        logging.warning(
            "Ridership: %d ambiguous rows left unmatched; review match output.", ambiguous
        )
    return result


def load_ridership(
    path: str,
    spec: RidershipSpec,
    trip_table: pd.DataFrame,
    routes: pd.DataFrame,
) -> RidershipResult:
    """Load average daily boardings per analysis-day trip, and how every row matched.

    Args:
        path: Ridership file (.csv or .xlsx/.xls).
        spec: Column names and the trip-matching strategy.
        trip_table: Output of :func:`build_trip_table`.
        routes: Parsed ``routes.txt``.

    Returns:
        :class:`RidershipResult` with boardings and observation counts per matched
        trip, and the per-row match table.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a required column is missing, or no row matches an
            analysis-day trip.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ridership file not found: {file_path}")
    frame = _read_table(file_path, "Ridership file")

    mode = spec.match_mode.strip().lower()
    if mode == "trip_id":
        required = [spec.trip_col, spec.boardings_col]
    else:
        required = [spec.route_col, spec.start_time_col, spec.boardings_col]
    date_col = spec.date_col.strip()
    if date_col:
        required.append(date_col)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Ridership file is missing column(s) {missing}. Columns found: "
            f"{', '.join(map(str, frame.columns))}. Set the RIDERSHIP_*_COL settings to match."
        )
    optional = {"obs_col": spec.obs_col}
    if mode == "route_start_time":
        optional.update(direction_col=spec.direction_col, block_col=spec.block_col)
    absent = {field: c for field, c in optional.items() if c.strip() and c not in frame.columns}
    for column in absent.values():
        logging.warning(
            "Ridership: optional column %r is not in the file; continuing without it.", column
        )
    spec = spec._replace(
        obs_col="" if "obs_col" in absent else spec.obs_col,
        direction_col="" if "direction_col" in absent else spec.direction_col,
        block_col="" if "block_col" in absent else spec.block_col,
    )

    rows_in = len(frame)
    frame["_source_row"] = range(2, rows_in + 2)
    frame["_boardings"] = pd.to_numeric(frame[spec.boardings_col], errors="coerce")
    numeric = frame["_boardings"].map(lambda v: pd.notna(v) and math.isfinite(v) and v >= 0)
    if mode == "trip_id":
        trips = frame[spec.trip_col].astype(str).str.strip()
        on_day = trips.isin(set(trip_table.index))
        status = pd.Series(_MS_MATCHED, index=frame.index, dtype=object)
        status[~on_day] = _MS_OFF_DAY
        status[trips.eq("")] = _MS_BLANK_TRIP
        match = pd.DataFrame({"_trip": trips.where(on_day, ""), "_match_status": status})
    else:
        match = match_trips_by_route_and_time(frame[numeric], trip_table, routes, spec)
    frame = frame.join(match)
    frame["_trip"] = frame["_trip"].fillna("").astype(str)
    frame.loc[~numeric, "_trip"] = ""
    frame.loc[~numeric, "_match_status"] = _MS_BAD_BOARDINGS

    matched = frame["_match_status"].eq(_MS_MATCHED)
    unmatched = int((~matched).sum())
    if unmatched:
        counts = frame.loc[~matched, "_match_status"].value_counts()
        detail = "; ".join(f"{n} {status}" for status, n in counts.items())
        hint = ""
        if rows_in and (rows_in - unmatched) / rows_in < _RIDERSHIP_MATCH_WARN_SHARE:
            hint = " Under half the rows matched: check that the file describes this day type"
            if mode == "trip_id":
                hint += (
                    f" and that {spec.trip_col!r} holds GTFS trip_ids (if not, set "
                    "RIDERSHIP_TRIP_MATCH_MODE = 'route_start_time')"
                )
            hint += "."
        logging.warning(
            "Ridership: %d of %d row(s) matched no analysis-day trip (%s); their boardings "
            "are not used. Every row's status is in the ridership match table.%s",
            unmatched,
            rows_in,
            detail,
            hint,
        )
    if not matched.any():
        raise ValueError(
            f"None of the ridership file's {rows_in} row(s) matched a trip on the analysis "
            "day, so no trip can be scored. Check the day type, the id column, and the "
            "match mode."
        )

    used = frame[matched]
    has_stop = pd.Series(False, index=used.index)
    if spec.stop_col and spec.stop_col in used.columns:
        has_stop = used[spec.stop_col].fillna("").astype(str).str.strip().ne("")
        stop_trips = set(used.loc[has_stop, "_trip"])
        double = ~has_stop & used["_trip"].isin(stop_trips)
        if double.any():
            logging.warning(
                "Ridership: %d trip(s) have both per-stop rows and a whole-trip row; keeping "
                "the per-stop rows only.",
                used.loc[double, "_trip"].nunique(),
            )
            frame.loc[double[double].index, "_match_status"] = _MS_SUPERSEDED
            used = used[~double]
            has_stop = has_stop[~double]

    if date_col:
        used = used.assign(_date=used[date_col].astype(str).str.strip())
        per_day = used.groupby(["_trip", "_date"])["_boardings"].sum()
        boardings = per_day.groupby(level="_trip").mean()
        logging.info(
            "Ridership: averaged across %d distinct %r value(s).",
            used["_date"].nunique(),
            date_col,
        )
    else:
        repeats = used.loc[~has_stop].groupby("_trip").size()
        repeats = repeats[repeats > 1]
        if not repeats.empty:
            logging.warning(
                "Ridership: %d trip(s) carry more than one whole-trip row and the file has "
                "no date column; their boardings are summed (e.g. %s).",
                len(repeats),
                ", ".join(map(str, repeats.index[:5])),
            )
        boardings = used.groupby("_trip")["_boardings"].sum()
    boardings.index.name = "trip_id"

    observations = pd.Series(dtype=float)
    obs_col = spec.obs_col.strip()
    if obs_col:
        counts = pd.to_numeric(used[obs_col], errors="coerce")
        observations = counts.groupby(used["_trip"]).max().dropna()
        observations.index.name = "trip_id"

    logging.info(
        "Ridership: %d of %d analysis-day trip(s) have boardings (%.1f%%).",
        len(boardings),
        len(trip_table),
        100.0 * len(boardings) / len(trip_table),
    )

    source_cols = [
        c
        for c in dict.fromkeys(
            [
                spec.trip_col,
                spec.route_col,
                spec.direction_col,
                spec.start_time_col,
                spec.block_col,
                spec.stop_col,
                date_col,
                obs_col,
                spec.boardings_col,
            ]
        )
        if c and c in frame.columns
    ]
    match_cols = [
        c
        for c in (
            "_trip",
            "_match_status",
            "_match_delta_min",
            "_matched_on_block",
            "_mapped_direction_id",
        )
        if c in frame.columns
    ]
    table = frame[["_source_row", *source_cols, *match_cols]].rename(
        columns={
            "_source_row": "source_row",
            "_trip": "matched_trip_id",
            "_match_status": "match_status",
            "_match_delta_min": "match_delta_min",
            "_matched_on_block": "matched_on_block",
            "_mapped_direction_id": "mapped_direction_id",
        }
    )
    gtfs = trip_table[["route_label", "direction_id", "block_id"]].rename(
        columns={
            "route_label": "gtfs_route",
            "direction_id": "gtfs_direction_id",
            "block_id": "gtfs_block_id",
        }
    )
    gtfs["gtfs_start"] = trip_table["start_min"].map(minutes_to_hhmm)
    table = table.join(gtfs, on="matched_trip_id").reset_index(drop=True)
    rows_matched = int(frame["_match_status"].eq(_MS_MATCHED).sum())
    return RidershipResult(
        boardings.astype(float), observations.astype(float), table, rows_in, rows_matched
    )


# =============================================================================
# BLOCK ARITHMETIC
# =============================================================================


# =============================================================================
# OPTIONAL TODS DEADHEAD INPUTS
# =============================================================================


class Movement(NamedTuple):
    """Minutes of travel and the evidence used for one directional movement."""

    minutes: int
    source: str
    synthetic: bool = False
    wait_min: int = 0


class Connection(NamedTuple):
    """Charge for the gap between two surviving passenger trips."""

    charge: int
    deadhead: int
    release: bool
    feasible: bool
    source: str
    synthetic: bool = False


def _tods_table(folder: Path, name: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read an optional UTF-8 CSV-format .txt file, preserving identifiers."""
    path = folder / name
    if not path.is_file():
        return pd.DataFrame(columns=pd.Index(columns))
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    return frame.apply(lambda col: col.str.strip())


def _synthetic(row: Mapping[str, Any]) -> bool:
    return str(row.get("is_synthetic", "")).lower() in {"1", "true", "yes"}


class DeadheadModel:
    """Read a limited, explicit TODS profile and price changed block connections.

    Supports added non-revenue trips in trips_supplement/stop_times_supplement,
    operational stops in stops_supplement, and two custom analytical extensions:
    block_garages.txt and deadhead_travel_times.txt. This is not a general TODS
    merger: edits/deletions to public GTFS trips must be applied upstream.
    """

    def __init__(
        self,
        folder: str,
        feed: Mapping[str, Optional[pd.DataFrame]],
        trips: pd.DataFrame,
        rules: BlockRules,
    ) -> None:
        """Read the TODS folder (empty string = flat allowance only) for *trips*."""
        self.enabled = bool(folder.strip())
        self.trips = trips.to_dict("index")
        self.rules = rules
        self.scheduled: dict[tuple[str, str], Movement] = {}
        self.schedule_times: dict[tuple[str, str], tuple[int, int]] = {}
        self.garages: dict[tuple[str, str], str] = {}
        self.matrix: dict[tuple[str, str, str], list[tuple[int, int, Movement]]] = {}
        self.original_pairs: set[tuple[str, str]] = set()
        self._connections: dict[tuple[str, str], Connection] = {}
        self._edges: dict[tuple[str, bool], Movement] = {}
        for _, rows in trips[trips.block_id.ne("")].groupby("block_id"):
            ids = list(rows.sort_values(["start_min", "end_min"]).index.astype(str))
            self.original_pairs.update(zip(ids, ids[1:]))
        if not self.enabled:
            return
        root = Path(folder).expanduser()
        if not root.is_dir():
            raise ValueError(f"TODS_PATH must be an extracted directory: {root}")
        public_trips = _required(feed, "trips").set_index("trip_id")
        stops = _required(feed, "stops")
        stop_ids = set(stops.stop_id)
        services = set(public_trips.service_id)
        supplement = _tods_table(root, "stops_supplement.txt", ["stop_id"])
        if supplement.stop_id.duplicated().any():
            raise ValueError("Duplicate stop_id in stops_supplement.txt")
        if "TODS_delete" in supplement and supplement.TODS_delete.eq("1").any():
            raise ValueError("Apply TODS stop deletions upstream before using this profile.")
        stop_ids.update(supplement.stop_id)
        garages = _tods_table(
            root, "block_garages.txt", ["service_id", "block_id", "garage_stop_id"]
        )
        known_blocks = set(zip(public_trips.service_id, public_trips.block_id))
        for row in garages.to_dict("records"):
            key = (row["service_id"], row["block_id"])
            if (
                key in self.garages
                or key not in known_blocks
                or row["garage_stop_id"] not in stop_ids
            ):
                raise ValueError(f"Duplicate or invalid block garage: {key}")
            self.garages[key] = row["garage_stop_id"]
        matrix = _tods_table(
            root,
            "deadhead_travel_times.txt",
            [
                "service_id",
                "from_stop_id",
                "to_stop_id",
                "start_time",
                "end_time",
                "duration_min",
            ],
        )
        for row in matrix.to_dict("records"):
            a, b = row["from_stop_id"], row["to_stop_id"]
            lo = parse_time_to_minutes(row["start_time"])
            hi = parse_time_to_minutes(row["end_time"])
            duration = float(row["duration_min"])
            if (
                a not in stop_ids
                or b not in stop_ids
                or (row["service_id"] and row["service_id"] not in services)
                or lo is None
                or hi is None
                or lo >= hi
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError(f"Invalid travel-time row: {row}")
            key = (row["service_id"], a, b)
            move = Movement(
                math.ceil(duration), "matrix:" + row.get("source", "unspecified"), _synthetic(row)
            )
            self.matrix.setdefault(key, []).append((lo, hi, move))
        for key, bands in self.matrix.items():
            bands.sort(key=lambda band: band[0])
            if any(bands[i][1] > bands[i + 1][0] for i in range(len(bands) - 1)):
                raise ValueError(f"Overlapping or duplicate travel-time bands: {key}")
        nonrev = _tods_table(
            root,
            "trips_supplement.txt",
            [
                "trip_id",
                "route_id",
                "service_id",
                "block_id",
                "TODS_trip_type",
            ],
        )
        times = _tods_table(
            root,
            "stop_times_supplement.txt",
            [
                "trip_id",
                "stop_id",
                "stop_sequence",
                "arrival_time",
                "departure_time",
            ],
        )
        if nonrev.empty and matrix.empty:
            raise ValueError("TODS folder contains no non-revenue trips or travel-time estimates.")
        if (
            nonrev.trip_id.duplicated().any()
            or times.duplicated(["trip_id", "stop_sequence"]).any()
        ):
            raise ValueError("Duplicate TODS trip or stop-time primary key.")
        if set(nonrev.trip_id) & set(public_trips.index):
            raise ValueError(
                "Apply TODS edits to public trips upstream; this reader adds deadheads only."
            )
        if set(times.trip_id) - set(nonrev.trip_id):
            raise ValueError("stop_times_supplement references undefined non-revenue trips.")
        if "TODS_delete" in nonrev and nonrev.TODS_delete.eq("1").any():
            raise ValueError("Apply TODS trip deletions upstream before using this profile.")
        if "TODS_delete" in times and times.TODS_delete.eq("1").any():
            raise ValueError("Apply TODS stop-time deletions upstream before using this profile.")
        # TODS adds non-revenue routes in routes_supplement.txt.
        routes = set(_required(feed, "routes").route_id) | set(
            _tods_table(root, "routes_supplement.txt", ["route_id"]).route_id
        )
        timed = {str(t): g for t, g in times.groupby("trip_id")}
        all_block_groups = {str(b): g for b, g in trips.groupby("block_id")}
        active_groups = {
            (str(key[0]), str(key[1])): g.sort_values("start_min")
            for raw_key, g in trips.groupby(["service_id", "block_id"])
            for key in [cast("tuple[str, str]", raw_key)]
        }
        occupied: dict[str, list[tuple[int, int, str]]] = {}
        skipped_types: set[str] = set()
        for row in nonrev.to_dict("records"):
            tid, service, block = row["trip_id"], row["service_id"], row["block_id"]
            if (service, block) not in known_blocks or row["route_id"] not in routes:
                raise ValueError(f"TODS trip {tid}: unknown service/block/route.")
            kind = row["TODS_trip_type"].strip().lower().replace("-", "_").replace(" ", "_")
            kind = {
                "pullout": "pull_out",
                "pullin": "pull_in",
                "pull_back": "pull_in",  # the TODS spelling; tsde_to_tods.py writes it
                "pullback": "pull_in",
                "between_trip": "deadhead",
            }.get(kind, kind)
            if kind not in {"pull_out", "pull_in", "deadhead"}:
                skipped_types.add(kind)
                continue
            st = timed.get(tid)
            if st is None or len(st) < 2:
                raise ValueError(f"TODS trip {tid}: needs at least two endpoint stop-times.")
            st = st.assign(_seq=pd.to_numeric(st.stop_sequence, errors="raise")).sort_values("_seq")
            if st._seq.duplicated().any() or (st._seq < 0).any() or (st._seq % 1 != 0).any():
                raise ValueError(f"TODS trip {tid}: invalid stop_sequence.")
            parsed = [
                (parse_time_to_minutes(a), parse_time_to_minutes(d))
                for a, d in zip(st.arrival_time, st.departure_time)
            ]
            previous = -1
            for arrival, departure in parsed:
                if (
                    arrival is None
                    or departure is None
                    or arrival < previous
                    or departure < arrival
                ):
                    raise ValueError(f"TODS trip {tid}: missing or non-monotonic times.")
                previous = departure
            start, end = parsed[0][1], parsed[-1][0]
            assert start is not None and end is not None
            origin, dest = str(st.stop_id.iloc[0]), str(st.stop_id.iloc[-1])
            if set(st.stop_id) - stop_ids:
                raise ValueError(f"TODS trip {tid}: unknown stop.")
            group = active_groups.get((service, block))
            if group is None:
                continue
            occupied.setdefault(block, []).append((start, end, tid))
            before = group[group.end_min <= start]
            after = group[group.start_min >= end]
            left = str(before.index[-1]) if len(before) else ""
            right = str(after.index[0]) if len(after) else ""
            if kind == "pull_out":
                left = ""
            if kind == "pull_in":
                right = ""
            # Optional analytical links disambiguate duplicate terminal times.
            left = str(row.get("from_trip_id", left))
            right = str(row.get("to_trip_id", right))
            if (kind == "pull_out" and (left or not right)) or (
                kind == "pull_in" and (right or not left)
            ):
                raise ValueError(f"TODS trip {tid}: invalid pull-in/out anchors.")
            if kind == "deadhead" and (left, right) not in self.original_pairs:
                raise ValueError(f"TODS trip {tid}: endpoints are not consecutive public trips.")
            for anchor in (left, right):
                if anchor and (
                    anchor not in self.trips
                    or self.trips[anchor]["block_id"] != block
                    or self.trips[anchor]["service_id"] != service
                ):
                    raise ValueError(f"TODS trip {tid}: invalid linked revenue trip {anchor}.")
            if left and (
                self.trips[left]["end_stop_id"] != origin or self.trips[left]["end_min"] > start
            ):
                raise ValueError(f"TODS trip {tid}: invalid origin/time.")
            if right and (
                self.trips[right]["start_stop_id"] != dest or self.trips[right]["start_min"] < end
            ):
                raise ValueError(f"TODS trip {tid}: invalid destination/time.")
            block_group = all_block_groups[block]
            if ((block_group.start_min < end) & (block_group.end_min > start)).any():
                raise ValueError(f"TODS trip {tid} overlaps passenger service.")
            if kind in {"pull_in", "pull_out"}:
                garage = origin if kind == "pull_out" else dest
                key = (service, block)
                if key in self.garages and self.garages[key] != garage:
                    raise ValueError(
                        f"Conflicting garages for {key}; multiple garages are unsupported."
                    )
                self.garages[key] = garage
            key = (left, right)
            if key in self.scheduled:
                raise ValueError(f"Conflicting scheduled movement links: {key}")
            self.scheduled[key] = Movement(end - start, "scheduled:" + tid, _synthetic(row))
            self.schedule_times[key] = (start, end)
        for block, spans in occupied.items():
            spans.sort()
            if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
                raise ValueError(f"Overlapping scheduled deadheads in block {block}.")
        if skipped_types:
            logging.warning("Ignored unsupported TODS_trip_type values: %s", sorted(skipped_types))
        if any(m.synthetic for m in self.scheduled.values()) or any(
            m.synthetic for bands in self.matrix.values() for _, _, m in bands
        ):
            logging.warning(
                "SYNTHETIC TODS TEST DATA: resulting savings are not operational estimates."
            )
        logging.info(
            "TODS: %d active scheduled movements; %d directional/service matrix keys.",
            len(self.scheduled),
            len(self.matrix),
        )

    def estimate(
        self, service: str, a: str, b: str, minute: int, arrive: bool = False
    ) -> Optional[Movement]:
        """Resolve service-specific bands before wildcard service, without wrapping midnight."""
        if a == b and a:
            return Movement(0, "same_location")
        for svc in (service, ""):
            found = []
            for lo, hi, move in self.matrix.get((svc, a, b), []):
                departure = min(minute - move.minutes, hi - 1) if arrive else minute
                if lo <= departure < hi:
                    waiting = minute - departure - move.minutes if arrive else 0
                    found.append((departure, move._replace(wait_min=waiting)))
            if found:
                return max(found, key=lambda item: item[0])[1]
        return None

    def edge(self, tid: str, outbound: bool) -> Movement:
        """Scheduled leg, then location-specific estimate, then half the pair allowance."""
        key = (tid, outbound)
        if key in self._edges:
            return self._edges[key]
        scheduled = ("", tid) if outbound else (tid, "")
        move = self.scheduled.get(scheduled)
        row = self.trips[tid]
        garage = self.garages.get((row["service_id"], row["block_id"]), "")
        if move is None and garage:
            a, b = (garage, row["start_stop_id"]) if outbound else (row["end_stop_id"], garage)
            move = self.estimate(
                row["service_id"],
                a,
                b,
                int(row["start_min"] if outbound else row["end_min"]),
                outbound,
            )
        if move is None:
            # Odd allowances split deterministically and still sum to the configured pair.
            duration = self.rules.pull_deadhead_min // 2
            if not outbound:
                duration = self.rules.pull_deadhead_min - duration
            move = Movement(duration, "fallback_half_pair")
        self._edges[key] = move
        return move

    def edge_charge(self, tid: str, outbound: bool) -> int:
        """Include terminal waiting between a scheduled garage leg and passenger service."""
        key = ("", tid) if outbound else (tid, "")
        if key in self.schedule_times:
            depart, arrive = self.schedule_times[key]
            row = self.trips[tid]
            return int(row["start_min"]) - depart if outbound else arrive - int(row["end_min"])
        movement = self.edge(tid, outbound)
        return movement.minutes + movement.wait_min

    def connection(self, left: str, right: str) -> Connection:
        """Use the complete scheduled gap or estimate the new connection after a cut."""
        key = (left, right)
        if key in self._connections:
            return self._connections[key]
        a, b = self.trips[left], self.trips[right]
        gap = int(b["start_min"] - a["end_min"])
        if gap < 0:
            answer = Connection(0, 0, False, False, "overlap")
        else:
            pin, pout = self.edge(left, False), self.edge(right, True)
            pin_charge, pout_charge = self.edge_charge(left, False), self.edge_charge(right, True)
            movement = self.scheduled.get(key)
            actual_release = (
                key in self.original_pairs
                and (left, "") in self.scheduled
                and ("", right) in self.scheduled
            )
            if actual_release:
                idle = self.schedule_times[("", right)][0] - self.schedule_times[(left, "")][1]
                answer = Connection(
                    pin_charge + pout_charge,
                    pin.minutes + pout.minutes,
                    True,
                    idle >= 0,
                    pin.source + ";" + pout.source,
                    pin.synthetic or pout.synthetic,
                )
            elif movement is not None:
                answer = Connection(
                    gap,
                    movement.minutes,
                    False,
                    movement.minutes <= gap,
                    movement.source,
                    movement.synthetic,
                )
            else:
                movement = self.estimate(
                    a["service_id"], a["end_stop_id"], b["start_stop_id"], int(a["end_min"])
                )
                # Existing GTFS adjacency is retained when there is no scheduled/matrix evidence.
                unknown = movement is None
                if movement is None:
                    movement = Movement(
                        0,
                        "gtfs_gap_unverified"
                        if key in self.original_pairs
                        else "missing_new_connection",
                    )
                garage_a = self.garages.get((a["service_id"], a["block_id"]))
                garage_b = self.garages.get((b["service_id"], b["block_id"]))
                garage_compatible = not self.enabled or (bool(garage_a) and garage_a == garage_b)
                can_release = (
                    gap >= self.rules.min_release_gap_min
                    and garage_compatible
                    and gap - pin_charge - pout_charge >= MIN_GARAGE_IDLE_MIN
                )
                if can_release:
                    answer = Connection(
                        pin_charge + pout_charge,
                        pin.minutes + pout.minutes,
                        True,
                        True,
                        pin.source + ";" + pout.source,
                        pin.synthetic or pout.synthetic,
                    )
                else:
                    feasible = movement.minutes <= gap and (
                        not unknown or key in self.original_pairs or not self.enabled
                    )
                    answer = Connection(
                        gap, movement.minutes, False, feasible, movement.source, movement.synthetic
                    )
        self._connections[key] = answer
        return answer

    def metrics(self, ids: Sequence[str]) -> dict[str, int]:
        """Separate movement, waiting, and passenger runtime without counting a gap twice."""
        if not ids:
            return dict(
                trips=0, pieces=0, revenue_min=0, layover_min=0, deadhead_min=0, vehicle_min=0
            )
        revenue = sum(int(self.trips[t]["runtime_min"]) for t in ids)
        connections = [self.connection(a, b) for a, b in zip(ids, ids[1:])]
        if any(not c.feasible for c in connections):
            raise ValueError("Cannot price a block containing an infeasible connection.")
        edge = self.edge(ids[0], True).minutes + self.edge(ids[-1], False).minutes
        deadhead = edge + sum(c.deadhead for c in connections)
        wait = (
            self.edge_charge(ids[0], True)
            + self.edge_charge(ids[-1], False)
            - edge
            + sum(c.charge - c.deadhead for c in connections)
        )
        return dict(
            trips=len(ids),
            pieces=1 + sum(c.release for c in connections),
            revenue_min=revenue,
            layover_min=wait,
            deadhead_min=deadhead,
            vehicle_min=revenue + wait + deadhead,
        )

    def pieces(self, ids: Sequence[str], include_deadhead: bool = False) -> list[tuple[int, int]]:
        """Return a block's pieces as (start, end), optionally with pull-out/in time."""
        if not ids:
            return []
        groups: list[list[str]] = [[ids[0]]]
        for left, right in zip(ids, ids[1:]):
            if self.connection(left, right).release:
                groups.append([])
            groups[-1].append(right)
        return [
            (
                int(self.trips[g[0]]["start_min"])
                - (self.edge_charge(g[0], True) if include_deadhead else 0),
                int(self.trips[g[-1]]["end_min"])
                + (self.edge_charge(g[-1], False) if include_deadhead else 0),
            )
            for g in groups
        ]

    def audit(self, blocks: Mapping[str, BlockState], scenario: str) -> pd.DataFrame:
        """List every block's pull-out and pull-in charge and where it came from."""
        rows = []
        for block in blocks.values():
            ids = block.trip_ids
            if not ids:
                continue
            for tid, outbound in ((ids[0], True), (ids[-1], False)):
                m = self.edge(tid, outbound)
                charge = self.edge_charge(tid, outbound)
                rows.append(
                    dict(
                        scenario=scenario,
                        block_id=block.block_id,
                        from_trip_id="" if outbound else tid,
                        to_trip_id=tid if outbound else "",
                        kind="pull_out" if outbound else "pull_in",
                        charged_min=charge,
                        deadhead_min=m.minutes,
                        waiting_min=charge - m.minutes,
                        source=m.source,
                        is_synthetic=m.synthetic,
                        feasible=True,
                    )
                )
            for a, b in zip(ids, ids[1:]):
                c = self.connection(a, b)
                rows.append(
                    dict(
                        scenario=scenario,
                        block_id=block.block_id,
                        from_trip_id=a,
                        to_trip_id=b,
                        kind="garage_release" if c.release else "between_trips",
                        charged_min=c.charge,
                        deadhead_min=c.deadhead,
                        waiting_min=c.charge - c.deadhead,
                        source=c.source,
                        is_synthetic=c.synthetic,
                        feasible=c.feasible,
                    )
                )
        return pd.DataFrame(rows)


def validate_analysis_blocks(
    table: pd.DataFrame,
    model: DeadheadModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarantine whole blocks with overlapping trips or impossible known connections."""
    bad: dict[str, str] = {}
    details = []
    for block, rows in table[table.block_id.ne("")].groupby("block_id"):
        ids = list(rows.sort_values(["start_min", "end_min"]).index.astype(str))
        furthest = -1
        previous = ""
        for tid in ids:
            row = model.trips[tid]
            reason = ""
            if int(row["start_min"]) < furthest:
                reason = "overlapping passenger trips"
            elif previous and not model.connection(previous, tid).feasible:
                reason = "known deadhead cannot fit scheduled gap"
            if reason:
                bad[str(block)] = reason
                details.append(
                    dict(block_id=block, previous_trip_id=previous, trip_id=tid, reason=reason)
                )
            furthest = max(furthest, int(row["end_min"]))
            previous = tid
    if bad:
        logging.warning(
            "Excluded %d invalid blocks (%d trips) from the baseline and cuts; see block QA.",
            len(bad),
            int(table.block_id.isin(bad).sum()),
        )
    return table[~table.block_id.isin(bad)].copy(), pd.DataFrame(
        details, columns=pd.Index(["block_id", "previous_trip_id", "trip_id", "reason"])
    )


def gap_charge(gap: int, rules: BlockRules) -> int:
    """Vehicle minutes a gap between consecutive trips of a block costs.

    A gap shorter than the release threshold is layover the vehicle sits through and
    costs its full length; a longer gap releases the vehicle and costs only the
    pull-in + pull-out deadhead allowance.
    """
    return gap if gap < rules.min_release_gap_min else rules.pull_deadhead_min


def block_gaps(starts: Sequence[int], ends: Sequence[int]) -> list[int]:
    """Gaps between consecutive trips (next start minus previous end), floored at 0."""
    return [max(0, starts[k + 1] - ends[k]) for k in range(len(starts) - 1)]


def block_metrics(
    starts: Sequence[int],
    ends: Sequence[int],
    rules: BlockRules,
    trip_ids: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """Summarize one block's trips, layover, pieces, and vehicle minutes.

    A *piece* is a stretch of the block between releases (gaps of at least
    ``min_release_gap_min``). Vehicle minutes are revenue minutes, plus layover
    shorter than the release threshold, plus the deadhead allowance per piece.

    Args:
        starts: Trip first departures in start order.
        ends: Trip last arrivals, parallel to *starts*.
        rules: Release threshold and deadhead allowance.
        trip_ids: The block's trip_ids, parallel to *starts*; required when a TODS
            deadhead model is active.

    Returns:
        Dict with ``trips``, ``pieces``, ``revenue_min``, ``layover_min``,
        ``deadhead_min``, and ``vehicle_min``.
    """
    if rules.model is not None:
        if trip_ids is None:
            raise ValueError("trip_ids required for deadhead accounting.")
        return rules.model.metrics(trip_ids)
    if any(starts[k + 1] < max(ends[: k + 1]) for k in range(len(starts) - 1)):
        raise ValueError("Overlapping trips cannot be summed as a vehicle block.")
    if not starts:
        return {
            "trips": 0,
            "pieces": 0,
            "revenue_min": 0,
            "layover_min": 0,
            "deadhead_min": 0,
            "vehicle_min": 0,
        }
    gaps = block_gaps(starts, ends)
    revenue = sum(max(0, e - s) for s, e in zip(starts, ends))
    layover = sum(g for g in gaps if g < rules.min_release_gap_min)
    pieces = 1 + sum(1 for g in gaps if g >= rules.min_release_gap_min)
    return {
        "trips": len(starts),
        "pieces": pieces,
        "revenue_min": revenue,
        "layover_min": layover,
        "deadhead_min": rules.pull_deadhead_min * pieces,
        "vehicle_min": revenue + layover + rules.pull_deadhead_min * pieces,
    }


def block_pieces(
    starts: Sequence[int], ends: Sequence[int], rules: BlockRules
) -> list[tuple[int, int]]:
    """Return each piece of a block as (first departure, last arrival)."""
    pieces: list[tuple[int, int]] = []
    if not starts:
        return pieces
    piece_start = starts[0]
    for k, gap in enumerate(block_gaps(starts, ends)):
        if gap >= rules.min_release_gap_min:
            pieces.append((piece_start, ends[k]))
            piece_start = starts[k + 1]
    pieces.append((piece_start, max(ends[-1], piece_start)))
    return pieces


def enumerate_runs(
    starts: Sequence[int],
    ends: Sequence[int],
    rules: BlockRules,
    eligible: Optional[Sequence[bool]] = None,
    single_trips_only: bool = False,
    trip_ids: Optional[Sequence[str]] = None,
) -> list[RunCut]:
    """Score contiguous cuts, including changed garage legs and connection feasibility."""
    n = len(starts)
    if n == 0:
        return []
    if any(starts[k + 1] < max(ends[: k + 1]) for k in range(n - 1)):
        raise ValueError("Overlapping trips cannot be scored as one vehicle block.")
    model = rules.model
    ids = list(trip_ids) if trip_ids is not None else []
    if model is not None and len(ids) != n:
        raise ValueError("Endpoint-aware scoring requires one trip_id per trip.")
    flags = list(eligible) if eligible is not None else [True] * n
    connections = [model.connection(a, b) for a, b in zip(ids, ids[1:])] if model else []
    if any(not c.feasible for c in connections):
        raise ValueError("Baseline block contains an infeasible connection.")
    charges = (
        [c.charge for c in connections]
        if model
        else [gap_charge(g, rules) for g in block_gaps(starts, ends)]
    )
    out = [model.edge_charge(t, True) for t in ids] if model else [rules.pull_deadhead_min // 2] * n
    ins = (
        [model.edge_charge(t, False) for t in ids]
        if model
        else [rules.pull_deadhead_min - rules.pull_deadhead_min // 2] * n
    )
    rt, ch = [0], [0]
    for s, e in zip(starts, ends):
        rt.append(rt[-1] + e - s)
    for value in charges:
        ch.append(ch[-1] + value)
    total = out[0] + ins[-1] + rt[-1] + ch[-1]
    runs = []
    for i in range(n):
        if not flags[i]:
            continue
        for j in range(i, i + 1 if single_trips_only else n):
            if not flags[j]:
                break
            revenue = rt[j + 1] - rt[i]
            gap, feasible = None, True
            if i == 0 and j == n - 1:
                kind, saved = "whole_block", total
            elif i == 0:
                kind = "block_start_trim"
                saved = revenue + ch[j + 1] + out[0] - out[j + 1]
            elif j == n - 1:
                kind = "block_end_trim"
                saved = revenue + ch[n - 1] - ch[i - 1] + ins[-1] - ins[i - 1]
            else:
                gap = starts[j + 1] - ends[i - 1]
                if model:
                    c = model.connection(ids[i - 1], ids[j + 1])
                    charge, release, feasible = c.charge, c.release, c.feasible
                else:
                    charge, release = gap_charge(gap, rules), gap >= rules.min_release_gap_min
                saved = revenue + ch[j + 1] - ch[i - 1] - charge
                kind = "mid_block_release" if release else "mid_block_layover_only"
                if not feasible:
                    kind = "infeasible_connection"
            runs.append(RunCut(i, j, kind, revenue, saved, gap, feasible))
    return runs


def build_blocks(trip_table: pd.DataFrame) -> dict[str, BlockState]:
    """Group the analysis day's trips into blocks, each in start order.

    Trips without a ``block_id`` become single-trip blocks under a synthetic id so
    the baseline totals still count them.
    """
    blocks: dict[str, BlockState] = {}
    keyed = trip_table.assign(
        _block=[
            bid if bid else f"{_NO_BLOCK_PREFIX}{tid}"
            for tid, bid in zip(trip_table.index, trip_table["block_id"])
        ]
    ).sort_values(["_block", "start_min", "end_min"])
    for block_id, rows in keyed.groupby("_block", sort=True):
        blocks[str(block_id)] = BlockState(
            str(block_id),
            [str(t) for t in rows.index],
            [int(v) for v in rows["start_min"]],
            [int(v) for v in rows["end_min"]],
        )
    overlaps = sum(
        1 for b in blocks.values() for k in range(len(b.starts) - 1) if b.starts[k + 1] < b.ends[k]
    )
    if overlaps:
        logging.warning(
            "%d block gap(s) are negative (a trip starts before the previous trip in its "
            "block ends); they are treated as zero-minute gaps.",
            overlaps,
        )
    return blocks


# =============================================================================
# ROUTE-DIRECTION STATE (headway and span guards)
# =============================================================================


def build_trip_lookup(trip_table: pd.DataFrame) -> TripLookup:
    """Copy the route-direction and route label of every trip into dicts."""
    ids = [str(t) for t in trip_table.index]
    route_dir = {
        tid: (str(r), str(d))
        for tid, r, d in zip(ids, trip_table["route_id"], trip_table["direction_id"])
    }
    return TripLookup(route_dir, dict(zip(ids, trip_table["route_label"].astype(str))))


def build_route_dir_index(trip_table: pd.DataFrame) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Map each (route_id, direction_id) to its trips' (start, trip_id), sorted."""
    index: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for tid, r, d, start in zip(
        trip_table.index,
        trip_table["route_id"],
        trip_table["direction_id"],
        trip_table["start_min"],
    ):
        index.setdefault((str(r), str(d)), []).append((int(start), str(tid)))
    for entries in index.values():
        entries.sort()
    return index


def service_effects(
    trip_ids: Sequence[str],
    lookup: TripLookup,
    route_dirs: Mapping[tuple[str, str], list[tuple[int, str]]],
) -> tuple[bool, Optional[int]]:
    """Report what removing *trip_ids* does to service on their routes and directions.

    Args:
        trip_ids: Trips to remove.
        lookup: Output of :func:`build_trip_lookup`.
        route_dirs: Current (start, trip_id) lists per route-direction.

    Returns:
        Tuple of (whether the cut removes a route-direction's current first or last
        trip, the longest new gap between remaining consecutive departures in
        minutes — ``None`` when the cut opens no interior gap).
    """
    removed_by_key: dict[tuple[str, str], set[str]] = {}
    for tid in trip_ids:
        removed_by_key.setdefault(lookup.route_dir[tid], set()).add(tid)

    shortens_span = False
    worst: Optional[int] = None
    for key, removed in removed_by_key.items():
        entries = route_dirs.get(key, [])
        if not entries:
            continue
        if entries[0][1] in removed or entries[-1][1] in removed:
            shortens_span = True
        remaining = [start for start, tid in entries if tid not in removed]
        if len(remaining) < 2:
            continue
        for start, tid in entries:
            if tid not in removed:
                continue
            k = bisect.bisect_left(remaining, start)
            if 0 < k < len(remaining):
                gap = remaining[k] - remaining[k - 1]
                worst = gap if worst is None else max(worst, gap)
    return shortens_span, worst


# =============================================================================
# SELECTION
# =============================================================================


def _guard_reason(shortens_span: bool, new_headway: Optional[int], guards: Guards) -> Optional[str]:
    """Name the guard a cut violates, or ``None`` when it passes."""
    if guards.protect_route_span and shortens_span:
        return "route_span"
    if (
        guards.max_resulting_headway_min > 0
        and new_headway is not None
        and new_headway > guards.max_resulting_headway_min
    ):
        return "max_headway"
    return None


# =============================================================================
# REPORT TABLES
# =============================================================================


def trip_report(
    trip_table: pd.DataFrame,
    baseline_blocks: Mapping[str, BlockState],
    boardings: Mapping[str, float],
    eligibility: Mapping[str, str],
    rules: BlockRules,
    observations: Mapping[str, float],
    low_observation_trips: set[str],
) -> pd.DataFrame:
    """One row per analysis-day trip: ridership, block position, and solo-cut savings.

    Args:
        trip_table: Output of :func:`build_trip_table`.
        baseline_blocks: Blocks before any cut.
        boardings: Boardings per trip_id.
        eligibility: trip_id -> ``""`` when cuttable, else the reason it is not.
        rules: Block savings rules.
        observations: Observation count per trip_id (may be empty).
        low_observation_trips: Trips flagged by :func:`flag_low_observations`.

    Returns:
        The trip table, in block and start order.
    """
    solo: dict[str, dict[str, Any]] = {}
    for block in baseline_blocks.values():
        n = len(block.trip_ids)
        for run in enumerate_runs(
            block.starts, block.ends, rules, single_trips_only=True, trip_ids=block.trip_ids
        ):
            position = (
                "only"
                if n == 1
                else "first"
                if run.i == 0
                else "last"
                if run.i == n - 1
                else "middle"
            )
            short = (
                rules.min_release_gap_min - run.gap_left_min
                if run.cut_type == "mid_block_layover_only" and run.gap_left_min is not None
                else None
            )
            solo[block.trip_ids[run.i]] = {
                "block_key": block.block_id,
                "block_position": position,
                "block_trip_count": n,
                "solo_cut_type": run.cut_type,
                "solo_vehicle_min_saved": run.vehicle_min_saved if run.feasible else None,
                "solo_gap_left_min": run.gap_left_min,
                "solo_min_short_of_release": short,
            }

    out = trip_table.reset_index()
    out["start"] = out["start_min"].map(minutes_to_hhmm)
    out["end"] = out["end_min"].map(minutes_to_hhmm)
    out["boardings"] = out["trip_id"].map(boardings)
    hours = out["runtime_min"] / 60.0
    out["boardings_per_rev_hour"] = (out["boardings"] / hours.where(hours > 0)).round(2)
    out["productivity_pctile"] = (out["boardings_per_rev_hour"].rank(pct=True) * 100).round(1)
    out["observations"] = out["trip_id"].map(observations)
    out["low_observations"] = out["trip_id"].isin(low_observation_trips)
    details = pd.DataFrame.from_dict(solo, orient="index")
    out = out.join(details, on="trip_id")
    out["eligible"] = out["trip_id"].map(lambda t: eligibility.get(t, "") == "")
    out["ineligible_reason"] = out["trip_id"].map(lambda t: eligibility.get(t, ""))
    columns = [
        "trip_id",
        "route_label",
        "route_id",
        "direction_id",
        "block_id",
        "start",
        "end",
        "runtime_min",
        "boardings",
        "boardings_per_rev_hour",
        "productivity_pctile",
        "observations",
        "low_observations",
        "block_position",
        "block_trip_count",
        "solo_cut_type",
        "solo_vehicle_min_saved",
        "solo_gap_left_min",
        "solo_min_short_of_release",
        "eligible",
        "ineligible_reason",
    ]
    return out.sort_values(["block_key", "start_min"])[columns].reset_index(drop=True)


def candidate_review(
    trips_out: pd.DataFrame,
    trip_table: pd.DataFrame,
    excluded: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Classify individual cuts; retain excluded trips with an explicit reason.

    Ratings use boardings per modeled vehicle-hour saved. Each rating is against
    the original schedule: individual savings are not additive. Mid-block trips
    are ineligible only when cutting them alone cannot produce feasible savings.
    """
    lookup = build_trip_lookup(trip_table)
    route_dirs = build_route_dir_index(trip_table)
    rows: list[dict[str, Any]] = []
    for row in trips_out.to_dict("records"):
        tid = str(row["trip_id"])
        span, headway = service_effects([tid], lookup, route_dirs)
        saved = row["solo_vehicle_min_saved"]
        reason = str(row["ineligible_reason"])
        if reason.startswith("route in ROUTES_TO_EXCLUDE"):
            reason = reason.replace("route in ROUTES_TO_EXCLUDE", "protected route", 1)
        else:
            reason = {
                "no block_id": "missing GTFS block assignment",
                "route not in ROUTES_TO_INCLUDE": "route is outside the allowed route list",
                "no ridership": "missing or ambiguous ridership match",
                "zero running time": "no positive scheduled running time",
                "above PROTECT_TRIPS_ABOVE": "exceeds configured boarding-productivity guard",
            }.get(reason, reason)
        ratio: Optional[float] = None
        rating = "ineligible"
        flags = []
        if span:
            flags.append("shortens route-direction service span")
        if row["low_observations"]:
            flags.append("low observation count")
        if not reason:
            if pd.isna(saved):
                reason = "new connection is infeasible or has no travel-time estimate"
            elif saved <= 0:
                reason = (
                    "mid-block: cutting this trip alone saves no vehicle time"
                    if row["block_position"] == "middle"
                    else "cutting this trip alone saves no vehicle time"
                )
                if saved < 0:
                    reason += " (modeled vehicle time increases)"
            else:
                guard = _guard_reason(span, headway, cfg.guards)
                if guard == "route_span":
                    reason = "protected route-direction service span"
                elif guard == "max_headway":
                    reason = "resulting departure gap exceeds configured maximum"
        if not reason:
            ratio = float(row["boardings"]) / (float(saved) / 60.0)
            if ratio <= cfg.excellent_max_boardings_per_vehicle_hour_saved:
                rating = "excellent"
                reason = "at or below the excellent boardings-per-vehicle-hour threshold"
            elif ratio <= cfg.good_max_boardings_per_vehicle_hour_saved:
                rating = "good"
                reason = "above the excellent threshold and at or below the good threshold"
            else:
                rating = "poor"
                reason = "above the good boardings-per-vehicle-hour threshold"
        rows.append(
            {
                **row,
                "candidate_classification": rating,
                "classification_reason": reason,
                "in_analysis_baseline": True,
                "solo_cut_eligible": rating != "ineligible",
                "boardings_per_vehicle_hour_saved": ratio,
                "solo_shortens_span": span,
                "solo_new_max_headway_min": headway,
                "review_flags": "; ".join(flags),
                "classification_scope": "individual trip cut alone",
            }
        )
    for row in excluded.reset_index().to_dict("records"):
        rows.append(
            {
                **{k: row.get(k) for k in trips_out.columns},
                "start": minutes_to_hhmm(int(row["start_min"])),
                "end": minutes_to_hhmm(int(row["end_min"])),
                "candidate_classification": "ineligible",
                "classification_reason": row["exclusion_reason"],
                "in_analysis_baseline": False,
                "solo_cut_eligible": False,
                "eligible": False,
                "ineligible_reason": row["exclusion_reason"],
                "review_flags": "excluded from analysis totals",
                "classification_scope": "individual trip cut alone",
            }
        )
    review = pd.DataFrame(rows)
    review["solo_new_max_headway_min"] = review["solo_new_max_headway_min"].astype("Int64")
    review["solo_shortens_span"] = review["solo_shortens_span"].astype("boolean")
    order = {"excellent": 0, "good": 1, "poor": 2, "ineligible": 3}
    review["_order"] = review["candidate_classification"].map(order)
    review = review.sort_values(
        ["_order", "boardings_per_vehicle_hour_saved", "route_label", "start", "trip_id"],
        kind="stable",
    ).drop(columns="_order")
    leading = [
        "trip_id",
        "route_label",
        "direction_id",
        "block_id",
        "start",
        "end",
        "candidate_classification",
        "classification_reason",
        "boardings",
        "solo_vehicle_min_saved",
        "boardings_per_vehicle_hour_saved",
        "block_position",
        "solo_cut_type",
        "solo_cut_eligible",
        "in_analysis_baseline",
        "solo_shortens_span",
        "solo_new_max_headway_min",
        "review_flags",
    ]
    return review[leading + [c for c in review.columns if c not in leading]].reset_index(drop=True)


def configured_headway_sessions(cfg: Config) -> list[HeadwaySession]:
    """Expand manual and sliding windows, deduplicating identical requests."""
    sources = list(cfg.headway_sessions)
    search = cfg.headway_search
    if not isinstance(search, Mapping):
        raise ValueError("HEADWAY_SEARCH must be a dictionary; use {} to disable it.")
    if search:
        fields = {"start", "end", "durations_min", "step_min", "headways", "routes", "days"}
        if not isinstance(search, Mapping) or set(search) - fields:
            raise ValueError("HEADWAY_SEARCH has unknown fields or is not a dictionary.")
        start = parse_time_to_minutes(str(search.get("start", "")))
        end = parse_time_to_minutes(str(search.get("end", "")))
        durations = search.get("durations_min", [])
        step = search.get("step_min")
        headways = search.get("headways", [])
        if (
            start is None
            or end is None
            or start < 0
            or end <= start
            or end - start > 1440
            or type(step) is not int
            or step <= 0
        ):
            raise ValueError(
                "HEADWAY_SEARCH requires start < end (maximum 24 hours) and positive integer "
                "step_min."
            )
        if (
            not isinstance(durations, (list, tuple))
            or not durations
            or any(type(d) is not int or d <= 0 or d > end - start for d in durations)
        ):
            raise ValueError(
                "HEADWAY_SEARCH durations_min must be positive integers fitting its time range."
            )
        if (
            not isinstance(headways, (list, tuple))
            or not headways
            or any(not isinstance(h, (list, tuple)) or len(h) != 2 for h in headways)
        ):
            raise ValueError("HEADWAY_SEARCH headways must contain [original, target] pairs.")
        generated = []
        for duration in dict.fromkeys(durations):
            for window_start in range(start, end - duration + 1, step):
                for before, after in headways:
                    generated.append(
                        {
                            "name": f"Search_{window_start}_{duration}_{before}_to_{after}",
                            "start": minutes_to_hhmm(window_start),
                            "end": minutes_to_hhmm(window_start + duration),
                            "from_headway_min": before,
                            "to_headway_min": after,
                            "routes": search.get("routes", []),
                            "days": search.get("days", []),
                        }
                    )
        # Duplicate headway pairs are a configuration error, not a silent expansion.
        sources.extend(generated)
    parsed = parse_headway_sessions(sources)
    unique = {}
    for spec in parsed:
        unique.setdefault(
            (spec.start, spec.end, spec.from_headway, spec.to_headway, spec.routes, spec.days), spec
        )
    return list(unique.values())


_SHARED_ROUTE_COLUMNS = [
    "route_a_id",
    "route_a_label",
    "route_b_id",
    "route_b_label",
    "shared_blocks",
    "shared_block_ids",
    "direct_transitions",
    "qualifies_for_auto",
    "auto_reason",
    "config_group_name",
    "config_routes",
    "used_automatically",
]
SHARED_ROUTES_FILENAME = "trip_cut_shared_routes.csv"
LINKED_GROUPS_FILENAME = "trip_cut_linked_groups.csv"


def linked_route_groups(
    trip_table: pd.DataFrame, cfg: Config, filters: ResolvedRouteFilters
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Report direct block sharing and resolve bounded, independently tested groups.

    Pair discovery is per active service day and uses only valid passenger blocks.
    A common block does not itself prove a feasible or worthwhile frequency cut.
    """
    labels = {str(r): str(part.route_label.iloc[0]) for r, part in trip_table.groupby("route_id")}
    allowed = {
        rid
        for rid, label in labels.items()
        if not {rid, label}.intersection(filters.exclude)
        and (not filters.include or {rid, label}.intersection(filters.include))
    }
    shared: dict[tuple[str, str], set[str]] = {}
    transitions: dict[tuple[str, str], int] = {}
    for block_id, part in trip_table.groupby("block_id", sort=True):
        if not str(block_id).strip() or str(block_id).startswith(_NO_BLOCK_PREFIX):
            continue
        route_ids = part.sort_values("start_min").route_id.astype(str).tolist()
        for pair in combinations(sorted(set(route_ids)), 2):
            shared.setdefault(pair, set()).add(str(block_id))
        for left, right in zip(route_ids, route_ids[1:]):
            if left != right:
                pair = tuple(sorted((left, right)))
                transitions[pair] = transitions.get(pair, 0) + 1
    rows: list[dict[str, Any]] = []
    active: dict[str, tuple[str, ...]] = {}
    for (left, right), block_ids in sorted(shared.items()):
        count = transitions.get((left, right), 0)
        reason = (
            "route excluded or outside include list"
            if {left, right} - allowed
            else "fewer shared blocks than configured minimum"
            if len(block_ids) < cfg.linked_route_min_shared_blocks
            else "no direct consecutive route transition"
            if not count
            else "qualifying direct shared-block pair"
        )
        qualifies = (
            not ({left, right} - allowed)
            and len(block_ids) >= cfg.linked_route_min_shared_blocks
            and count > 0
        )
        # IDs make the export unambiguous even when short names repeat.
        name = f"auto_{left}_{right}"
        used = qualifies and cfg.linked_route_mode == "auto"
        if used:
            active[name] = (left, right)
        rows.append(
            dict(
                route_a_id=left,
                route_a_label=labels[left],
                route_b_id=right,
                route_b_label=labels[right],
                shared_blocks=len(block_ids),
                shared_block_ids=json.dumps(sorted(block_ids)),
                direct_transitions=count,
                qualifies_for_auto=qualifies,
                auto_reason=reason,
                config_group_name=name,
                config_routes=json.dumps([left, right]),
                used_automatically=used,
            )
        )
    if cfg.linked_route_mode != "off":
        for name, tokens in filters.linked_groups.items():
            resolved = []
            absent = []
            for token in tokens:
                matches = (
                    [token]
                    if token in labels
                    else [rid for rid, label in labels.items() if label == token]
                )
                if len(matches) > 1:
                    raise ValueError(
                        f"Linked group {name!r}: route name {token!r} is ambiguous; use route_id."
                    )
                if not matches:
                    absent.append(token)
                else:
                    resolved.extend(matches)
            ids = tuple(sorted(set(resolved)))
            if absent or set(ids) - allowed:
                logging.warning(
                    "Skipping linked group %s: absent routes %s; excluded routes %s.",
                    name,
                    absent,
                    sorted(set(ids) - allowed),
                )
                continue
            if len(ids) < 2:
                raise ValueError(
                    f"Linked group {name!r} resolves to fewer than two distinct routes."
                )
            # The explicit name wins over an identical discovered pair.
            active = {key: value for key, value in active.items() if value != ids}
            for row in rows:
                if tuple(json.loads(row["config_routes"])) == ids:
                    row["used_automatically"] = False
            if name in active and active[name] != ids:
                raise ValueError(f"Linked group name {name!r} conflicts with a discovered group.")
            active[name] = ids
    return pd.DataFrame(rows, columns=pd.Index(_SHARED_ROUTE_COLUMNS)), active


def shared_route_config_text(shared: pd.DataFrame, day: str) -> str:
    """Return copyable Python configuration with route IDs and explanatory labels."""
    lines = [
        f"# Shared-block route pairs for {day}; review before copying into CONFIGURATION.",
        "# Only pairs passing the configured route filters and discovery threshold appear here.",
        "# This is a suggestion file; it is never executed or read automatically.",
        '# Use LINKED_ROUTE_MODE = "auto" to rediscover pairs on each run.',
        'LINKED_ROUTE_MODE = "manual"',
        "LINKED_ROUTE_GROUPS = {",
    ]
    for row in shared.to_dict("records"):
        if row["qualifies_for_auto"]:
            labels = f"{row['route_a_label']} + {row['route_b_label']}".replace("\n", " ").replace(
                "\r", " "
            )
            lines.append(
                f"    {row['config_group_name']!r}: {json.loads(row['config_routes'])!r},  # "
                f"{labels}; {row['shared_blocks']} blocks"
            )
    lines.extend(["}", ""])
    return "\n".join(lines)


def option_scoring_context(
    trip_table: pd.DataFrame, baseline: Mapping[str, BlockState], cfg: Config
) -> dict[str, Any]:
    """Cache immutable baseline indexes and block metrics for a session search."""
    return {
        "lookup": build_trip_lookup(trip_table),
        "route_dirs": build_route_dir_index(trip_table),
        "valid_ids": set(trip_table.index),
        "block_of": {tid: key for key, block in baseline.items() for tid in block.trip_ids},
        "before": {
            key: block_metrics(b.starts, b.ends, cfg.rules, b.trip_ids)
            for key, b in baseline.items()
        },
    }


def score_fixed_block_option(
    removed: set[str],
    trip_table: pd.DataFrame,
    baseline: Mapping[str, BlockState],
    boardings: Mapping[str, float],
    eligibility: Mapping[str, str],
    cfg: Config,
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Price one independent cut unit against complete original blocks, without reblocking.

    Removed trips may be nonconsecutive and on multiple blocks. Reprice each affected
    block after ALL removals in this option; never add individual-trip savings.
    """
    context = context if context is not None else option_scoring_context(trip_table, baseline, cfg)
    affected_ids = {context["block_of"][t] for t in removed if t in context["block_of"]}
    affected = {key: baseline[key] for key in sorted(affected_ids)}
    reason = "; ".join(sorted({eligibility[t] for t in removed if eligibility.get(t)}))
    missing = removed - context["valid_ids"]
    if missing:
        reason = "session includes trips outside the valid analysis baseline"
    known_ridership = all(t in boardings for t in removed)
    lost = sum(boardings[t] for t in removed) if known_ridership else None
    span, headway = service_effects(
        sorted(removed - missing), context["lookup"], context["route_dirs"]
    )
    record: dict[str, Any] = {
        "trip_count": len(removed),
        "trip_ids": json.dumps(sorted(removed)),
        "affected_blocks": json.dumps(sorted(affected)),
        "boardings": lost,
        "revenue_min_removed": int(trip_table.loc[list(removed - missing), "runtime_min"].sum())
        if not missing
        else None,
        "vehicle_min_saved": None,
        "deadhead_min_change": None,
        "waiting_min_saved": None,
        "boardings_per_vehicle_hour_saved": None,
        "shortens_span": span,
        "new_max_headway_min": headway,
        "candidate_classification": "ineligible",
        "classification_reason": reason,
        "deadhead_inputs_synthetic": False,
    }
    if reason:
        return record
    if lost is None:
        record["classification_reason"] = "missing or ambiguous ridership match"
        return record
    totals = {key: 0 for key in ("vehicle_min", "deadhead_min", "layover_min")}
    changed: dict[str, BlockState] = {}
    for key, block in affected.items():
        kept = [i for i, tid in enumerate(block.trip_ids) if tid not in removed]
        post = BlockState(
            key,
            [block.trip_ids[i] for i in kept],
            [block.starts[i] for i in kept],
            [block.ends[i] for i in kept],
        )
        changed[key] = post
        before = context["before"][key]
        try:
            after = block_metrics(post.starts, post.ends, cfg.rules, post.trip_ids)
        except ValueError as exc:
            record["classification_reason"] = f"infeasible connection after session cuts: {exc}"
            return record
        for metric in totals:
            totals[metric] += before[metric] - after[metric]
    saved = totals["vehicle_min"]
    record.update(
        vehicle_min_saved=saved,
        deadhead_min_change=-totals["deadhead_min"],
        waiting_min_saved=totals["layover_min"],
    )
    if cfg.rules.model is not None:
        audit = pd.concat(
            [cfg.rules.model.audit(affected, "before"), cfg.rules.model.audit(changed, "after")],
            ignore_index=True,
        )
        if not audit.empty:
            record["deadhead_inputs_synthetic"] = bool(audit.is_synthetic.any())
    guard = _guard_reason(span, headway, cfg.guards)
    if guard:
        reason = (
            "protected route-direction service span"
            if guard == "route_span"
            else "resulting departure gap exceeds configured maximum"
        )
    elif saved <= 0:
        reason = "session saves no vehicle time within existing blocks"
        if saved < 0:
            reason += " (modeled vehicle time increases)"
    else:
        ratio = float(lost) / (saved / 60.0)
        rating = (
            "excellent"
            if ratio <= cfg.excellent_max_boardings_per_vehicle_hour_saved
            else "good"
            if ratio <= cfg.good_max_boardings_per_vehicle_hour_saved
            else "poor"
        )
        record.update(candidate_classification=rating, boardings_per_vehicle_hour_saved=ratio)
        reason = (
            f"{rating} under the configured boardings-per-vehicle-hour thresholds; session "
            "evaluated alone"
        )
    record["classification_reason"] = reason
    return record


_OPTION_COLUMNS = [
    "option_id",
    "option_type",
    "route_id",
    "route_label",
    "direction_id",
    "session_name",
    "window_start",
    "window_end",
    "from_headway_min",
    "to_headway_min",
    "candidate_classification",
    "classification_reason",
    "trip_count",
    "trip_ids",
    "affected_blocks",
    "boardings",
    "revenue_min_removed",
    "vehicle_min_saved",
    "deadhead_min_change",
    "waiting_min_saved",
    "boardings_per_vehicle_hour_saved",
    "shortens_span",
    "new_max_headway_min",
    "preserves_service_day_endpoints",
    "route_ids",
    "linked_group",
    "equivalent_to_option_id",
    "retained_departures",
    "cycle_min_diagnostic",
    "deadhead_inputs_synthetic",
    "review_flags",
]
_SESSION_TRIP_COLUMNS = [
    "option_id",
    "session_name",
    "route_id",
    "route_label",
    "direction_id",
    "trip_id",
    "block_id",
    "start",
    "end",
    "action",
    "boardings",
    "window_endpoint",
    "service_day_endpoint",
]


def frequency_session_options(
    source_table: pd.DataFrame,
    trip_table: pd.DataFrame,
    baseline: Mapping[str, BlockState],
    boardings: Mapping[str, float],
    eligibility: Mapping[str, str],
    cfg: Config,
    day: str,
    linked_groups: Optional[Mapping[str, tuple[str, ...]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test single-route and linked-route windows without retiming or reblocking.

    Each route-direction is a separate stream. All phase combinations are scored
    as one cut unit across its original blocks. Exact duplicate cut sets reuse
    their scores; near duplicates are consolidated only in the shortlist.
    """
    rows: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    tol = cfg.session_headway_tolerance_min
    specs = [spec for spec in configured_headway_sessions(cfg) if not spec.days or day in spec.days]
    if not specs:
        return pd.DataFrame(columns=pd.Index(_OPTION_COLUMNS)), pd.DataFrame(
            columns=pd.Index(_SESSION_TRIP_COLUMNS)
        )
    context = option_scoring_context(trip_table, baseline, cfg)
    cycles = observed_cycles(baseline, context["lookup"], cfg.rules)
    # Raw passenger data protects the true daily endpoints even after block QA.
    source_dirs = build_route_dir_index(source_table)
    daily_endpoints = {entries[i][1] for entries in source_dirs.values() for i in (0, -1)}
    routes = {str(rid): part for rid, part in source_table.groupby("route_id", sort=True)}
    groups: list[tuple[str, tuple[str, ...], pd.DataFrame]] = [
        ("", (rid,), route) for rid, route in routes.items()
    ]
    groups.extend(
        (name, ids, source_table[source_table.route_id.isin(ids)])
        for name, ids in (linked_groups or {}).items()
    )
    scored: dict[frozenset[str], dict[str, Any]] = {}
    equivalents: dict[tuple[Any, ...], str] = {}
    logging.info(
        "Searching %d windows/headways across %d routes and %d linked groups.",
        len(specs),
        len(routes),
        len(groups) - len(routes),
    )
    for group_index, (group_name, route_ids, route) in enumerate(groups, start=1):
        label = " + ".join(str(routes[rid].route_label.iloc[0]) for rid in route_ids)
        streams = sorted(set(zip(route.route_id.astype(str), route.direction_id.astype(str))))
        for spec_index, spec in enumerate(specs, start=1):
            # A requested route subset must allow EVERY member of a linked group.
            if spec.routes and any(
                not {rid, str(routes[rid].route_label.iloc[0])}.intersection(spec.routes)
                for rid in route_ids
            ):
                continue
            window = route[route.start_min.between(spec.start, spec.end)].sort_values(
                ["route_id", "direction_id", "start_min", "end_min"]
            )
            if window.empty:
                continue
            window_ids = set(window.index.astype(str))
            prefix = f"{day}:session:{spec_index}:group{group_index}"
            cycle_values = [cycles[t] for t in window_ids if t in cycles]
            base: dict[str, Any] = {
                "option_type": "frequency_session",
                "route_id": route_ids[0] if len(route_ids) == 1 else "",
                "route_ids": json.dumps(list(route_ids)),
                "route_label": label,
                "linked_group": group_name,
                "direction_id": ";".join(sorted(route.direction_id.unique())),
                "session_name": spec.name,
                "window_start": minutes_to_hhmm(spec.start),
                "window_end": minutes_to_hhmm(spec.end),
                "from_headway_min": spec.from_headway,
                "to_headway_min": spec.to_headway,
                "preserves_service_day_endpoints": cfg.session_preserve_first_last,
                "cycle_min_diagnostic": float(pd.Series(cycle_values).median())
                if cycle_values
                else None,
                "candidate_classification": "ineligible",
                "trip_count": 0,
                "trip_ids": "[]",
                "equivalent_to_option_id": "",
                "review_flags": "evaluated alone; existing blocks and departure times retained",
            }
            reason = ""
            choices: list[list[tuple[str, tuple[str, ...]]]] = []
            endpoints: set[str] = set()
            factor = spec.to_headway // spec.from_headway
            if factor ** len(streams) > cfg.session_max_phase_combinations:
                reason = (
                    "phase combinations exceed SESSION_MAX_PHASE_COMBINATIONS; split the linked "
                    "group or raise the limit"
                )
            if window_ids - context["valid_ids"]:
                reason = "session window includes trips in quarantined blocks"
            for rid, direction in streams:
                if reason:
                    break
                part = window[window.route_id.eq(rid) & window.direction_id.eq(direction)]
                if len(part) < factor + 1:
                    reason = (
                        "too few departures in one or more route-directions for the requested "
                        "reduction"
                    )
                    break
                if "pattern_key" not in part or len(set(part.pattern_key.dropna())) != 1:
                    reason = (
                        "mixed or unknown stop patterns in a direction; session requires a single "
                        "pattern"
                    )
                    break
                differences = part.start_min.diff().dropna()
                if (differences.sub(spec.from_headway).abs() > tol).any() or differences.le(
                    0
                ).any():
                    reason = "existing departures do not match the requested original headway"
                    break
                ids = list(part.index.astype(str))
                endpoints.update((ids[0], ids[-1]))
                variants = []
                seen_kept: set[tuple[str, ...]] = set()
                for phase in range(factor):
                    kept_set = set(ids[phase::factor])
                    if cfg.session_preserve_first_last:
                        kept_set.update(daily_endpoints & set(ids))
                    kept = tuple(t for t in ids if t in kept_set)
                    if len(kept) < len(ids) and kept not in seen_kept:
                        # JSON keys are unambiguous for labels containing punctuation.
                        stream_key = (
                            json.dumps([rid, direction]) if len(route_ids) > 1 else direction
                        )
                        variants.append((stream_key, kept))
                        seen_kept.add(kept)
                if not variants:
                    reason = (
                        "protected daily endpoints leave no frequency reduction in one or more "
                        "directions"
                    )
                    break
                choices.append(variants)
            if reason:
                rows.append(
                    {**base, "option_id": prefix + ":unavailable", "classification_reason": reason}
                )
                continue
            seen_removed: set[frozenset[str]] = set()
            window_records = window.to_dict("index")
            for combination in product(*choices):
                kept_ids = {tid for _, kept in combination for tid in kept}
                removed = window_ids - kept_ids
                frozen = frozenset(removed)
                if not removed or frozen in seen_removed:
                    continue
                seen_removed.add(frozen)
                option_id = f"{prefix}:{len(seen_removed)}"
                if frozen not in scored:
                    scored[frozen] = score_fixed_block_option(
                        removed, trip_table, baseline, boardings, eligibility, cfg, context
                    )
                score = dict(scored[frozen])
                if (
                    score["new_max_headway_min"] is not None
                    and score["new_max_headway_min"] > spec.to_headway + tol
                ):
                    score.update(
                        candidate_classification="ineligible",
                        classification_reason=(
                            "session would open a boundary/interior departure gap longer than the "
                            "target headway"
                        ),
                    )
                departures = {
                    direction: [minutes_to_hhmm(int(window_records[t]["start_min"])) for t in kept]
                    for direction, kept in combination
                }
                short_transitions = any(
                    int(window_records[b]["start_min"]) - int(window_records[a]["start_min"])
                    < spec.to_headway - tol
                    for _, kept in combination
                    for a, b in zip(kept, kept[1:])
                )
                identity = (frozen, spec.from_headway, spec.to_headway)
                row = {
                    **base,
                    **score,
                    "option_id": option_id,
                    "retained_departures": json.dumps(departures),
                    "equivalent_to_option_id": equivalents.get(identity, ""),
                }
                equivalents.setdefault(identity, option_id)
                if short_transitions:
                    row["review_flags"] += (
                        "; shorter transition gaps retain daily boundary departures"
                    )
                rows.append(row)
                for tid, trip in window_records.items():
                    members.append(
                        {
                            "option_id": option_id,
                            "session_name": spec.name,
                            "route_id": str(trip["route_id"]),
                            "route_label": str(trip["route_label"]),
                            "direction_id": trip["direction_id"],
                            "trip_id": str(tid),
                            "block_id": trip["block_id"],
                            "start": minutes_to_hhmm(int(trip["start_min"])),
                            "end": minutes_to_hhmm(int(trip["end_min"])),
                            "action": "remove" if tid in removed else "retain",
                            "boardings": boardings.get(str(tid)),
                            "window_endpoint": tid in endpoints,
                            "service_day_endpoint": tid in daily_endpoints,
                        }
                    )
        if group_index % 10 == 0 or group_index == len(groups):
            logging.info(
                "Session search: %d/%d route/group sets checked; %d scored cut sets.",
                group_index,
                len(groups),
                len(scored),
            )
    return pd.DataFrame(rows, columns=pd.Index(_OPTION_COLUMNS)), pd.DataFrame(
        members, columns=pd.Index(_SESSION_TRIP_COLUMNS)
    )


def option_shortlist(
    review: pd.DataFrame, sessions: pd.DataFrame, cfg: Config, day: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine independently rated singles and sessions without selecting a cut package."""
    singles = []
    for row in review.to_dict("records"):
        if not row["in_analysis_baseline"]:
            continue
        singles.append(
            {
                "option_id": f"{day}:trip:{row['trip_id']}",
                "option_type": "single_trip",
                "route_id": row["route_id"],
                "route_ids": json.dumps([row["route_id"]]),
                "route_label": row["route_label"],
                "direction_id": row["direction_id"],
                "window_start": row["start"],
                "window_end": row["end"],
                "candidate_classification": row["candidate_classification"],
                "classification_reason": row["classification_reason"],
                "trip_count": 1,
                "trip_ids": json.dumps([row["trip_id"]]),
                "affected_blocks": json.dumps([row["block_id"]]),
                "boardings": row["boardings"],
                "revenue_min_removed": row["runtime_min"],
                "vehicle_min_saved": row["solo_vehicle_min_saved"]
                if pd.notna(row["solo_vehicle_min_saved"])
                else None,
                "boardings_per_vehicle_hour_saved": row["boardings_per_vehicle_hour_saved"],
                "shortens_span": row["solo_shortens_span"],
                "new_max_headway_min": row["solo_new_max_headway_min"],
                "deadhead_inputs_synthetic": row["deadhead_inputs_synthetic"],
                "review_flags": row["review_flags"],
            }
        )
    parts = [pd.DataFrame(singles, columns=pd.Index(_OPTION_COLUMNS))]
    if not sessions.empty:
        parts.append(sessions)
    nonempty = [part for part in parts if not part.empty]
    options = pd.concat(nonempty, ignore_index=True) if nonempty else parts[0].copy()
    options["_class_order"] = options.candidate_classification.map(
        {"excellent": 0, "good": 1, "poor": 2, "ineligible": 3}
    )
    options = (
        options.sort_values(
            [
                "option_type",
                "_class_order",
                "boardings_per_vehicle_hour_saved",
                "vehicle_min_saved",
                "option_id",
            ],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
        .drop(columns="_class_order")
        .reset_index(drop=True)
    )
    options["shortlist_suppressed_by"] = ""
    options["shortlist_suppression_reason"] = ""
    # Compare removed-trip sets, not window names; disjoint sessions stay separate.
    accepted: dict[tuple[str, int, int], list[tuple[str, set[str]]]] = {}
    for index, row in options.iterrows():
        if row.option_type != "frequency_session" or row.candidate_classification == "ineligible":
            continue
        family = (str(row.route_ids), int(row.from_headway_min), int(row.to_headway_min))
        removed = set(json.loads(row.trip_ids))
        for previous_id, previous in accepted.get(family, []):
            exact = removed == previous
            overlap = len(removed & previous) / len(removed | previous)
            if exact or (
                cfg.session_shortlist_overlap > 0 and overlap >= cfg.session_shortlist_overlap
            ):
                options.loc[index, "shortlist_suppressed_by"] = previous_id
                options.loc[index, "shortlist_suppression_reason"] = (
                    "identical removed trips" if exact else f"removed-trip overlap {overlap:.1%}"
                )
                break
        else:
            accepted.setdefault(family, []).append((row.option_id, removed))
    options["rank_within_type"] = pd.Series(pd.NA, index=options.index, dtype="Int64")
    eligible = options.candidate_classification.ne(
        "ineligible"
    ) & options.shortlist_suppressed_by.eq("")
    options.loc[eligible, "rank_within_type"] = (
        options[eligible].groupby("option_type").cumcount() + 1
    )
    shortlist = options[eligible]
    if cfg.shortlist_limit_per_type:
        shortlist = shortlist[shortlist.rank_within_type.le(cfg.shortlist_limit_per_type)]
    return options, shortlist.reset_index(drop=True)


# Report-only annotations: these never change a rating, score, or departure.
OPTION_BLOCKS_FILENAME = "trip_cut_option_blocks.csv"
OPTION_GAPS_FILENAME = "trip_cut_option_gaps.csv"
GROUPED_SHORTLIST_FILENAME = "trip_cut_grouped_shortlist.csv"
ALTERNATIVES_FILENAME = "trip_cut_alternatives.csv"

_MECHANISM_COUNTS = {
    "whole_blocks_removed": "whole block removed",
    "blocks_starting_later": "later block start",
    "blocks_finishing_earlier": "earlier block finish",
    "new_interior_releases": "new interior release",
    "changed_existing_releases": "changed existing release",
    "interior_gaps_without_release": "interior waiting",
    "infeasible_interior_connections": "infeasible interior connection",
}
_BLOCK_EFFECT_COLUMNS = [
    "option_id",
    "block_id",
    "mechanisms",
    "trips_before",
    "trips_removed",
    "trips_after",
    "removed_trip_ids",
    "vehicle_min_before",
    "vehicle_min_after",
    "vehicle_min_saved",
    "revenue_min_removed",
    "deadhead_min_change",
    "waiting_min_saved",
    "pricing_status",
    "vehicle_start_before",
    "vehicle_start_after",
    "vehicle_finish_before",
    "vehicle_finish_after",
    "start_later_min",
    "finish_earlier_min",
    "first_departure_before",
    "first_departure_after",
    "last_arrival_before",
    "last_arrival_after",
    "first_revenue_trip_removed",
    "last_revenue_trip_removed",
    "interior_changes",
    *_MECHANISM_COUNTS,
]
_GAP_EFFECT_COLUMNS = [
    "option_id",
    "route_id",
    "route_label",
    "direction_id",
    "effect",
    "interval_start",
    "interval_end",
    "previous_retained_trip_id",
    "next_retained_trip_id",
    "removed_trip_ids",
    "removed_departures",
    "original_headways_min",
    "resulting_gap_min",
]
_EFFECT_COLUMNS = [
    "search_window_start",
    "search_window_end",
    "session_scope",
    "equivalent_single_trip_option_id",
    "removed_departure_first",
    "removed_departure_last",
    "affected_service_summary",
    "changed_gap_count",
    "savings_mechanism",
    "mechanism_status",
    *_MECHANISM_COUNTS,
]


def _clock(minutes: int) -> str:
    """Format a service-day time, including a modeled pull-out before midnight."""
    return ("-" if minutes < 0 else "") + minutes_to_hhmm(abs(minutes))


def _removed_runs(ids: Sequence[str], removed: set[str]) -> list[tuple[int, int]]:
    """Return maximal contiguous removed ranges in a specified trip order."""
    runs = []
    start = None
    for index, tid in enumerate(ids):
        if tid in removed:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(ids) - 1))
    return runs


def _report_connection(block: BlockState, left: int, right: int, rules: BlockRules) -> Connection:
    """Read the same release/feasibility rule used by the cost calculation."""
    if rules.model is not None:
        return rules.model.connection(block.trip_ids[left], block.trip_ids[right])
    gap = block.starts[right] - block.ends[left]
    release = gap >= rules.min_release_gap_min
    return Connection(
        gap_charge(gap, rules),
        rules.pull_deadhead_min if release else 0,
        release,
        gap >= 0,
        "flat gap rule",
    )


def _vehicle_bounds(block: BlockState, rules: BlockRules) -> tuple[int, int]:
    """Return modeled first pull-out and last pull-in, including boundary waiting."""
    if rules.model is not None:
        outward = rules.model.edge_charge(block.trip_ids[0], True)
        inward = rules.model.edge_charge(block.trip_ids[-1], False)
    else:
        outward = rules.pull_deadhead_min // 2
        inward = rules.pull_deadhead_min - outward
    return block.starts[0] - outward, block.ends[-1] + inward


def option_block_effects(
    removed: set[str], baseline: Mapping[str, BlockState], cfg: Config, context: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Describe each affected block and reconcile its complete before/after cost.

    Mechanism counts describe operations, not an additive allocation of savings.
    A release spanning one or more original releases is a changed existing release,
    never a new release. An infeasible changed block has no after-cost estimate.
    """
    rows = []
    keys = sorted({context["block_of"][t] for t in removed if t in context["block_of"]})
    for key in keys:
        block = baseline[key]
        kept = [i for i, tid in enumerate(block.trip_ids) if tid not in removed]
        post = BlockState(
            key,
            [block.trip_ids[i] for i in kept],
            [block.starts[i] for i in kept],
            [block.ends[i] for i in kept],
        )
        counts = dict.fromkeys(_MECHANISM_COUNTS, 0)
        before = context["before"][key]
        start_before, finish_before = _vehicle_bounds(block, cfg.rules)
        start_after, finish_after = _vehicle_bounds(post, cfg.rules) if kept else (None, None)
        interior = []
        if not kept:
            counts["whole_blocks_removed"] = 1
        else:
            counts["blocks_starting_later"] = int(
                start_after is not None and start_after > start_before
            )
            counts["blocks_finishing_earlier"] = int(
                finish_after is not None and finish_after < finish_before
            )
            for first, last in _removed_runs(block.trip_ids, removed):
                if first == 0 or last == len(block.trip_ids) - 1:
                    continue
                connection = _report_connection(block, first - 1, last + 1, cfg.rules)
                old_releases = sum(
                    _report_connection(block, i, i + 1, cfg.rules).release
                    for i in range(first - 1, last + 1)
                )
                kind = (
                    "infeasible_interior_connections"
                    if not connection.feasible
                    else "changed_existing_releases"
                    if connection.release and old_releases
                    else "new_interior_releases"
                    if connection.release
                    else "interior_gaps_without_release"
                )
                counts[kind] += 1
                interior.append(
                    {
                        "mechanism": _MECHANISM_COUNTS[kind],
                        "previous_trip_id": block.trip_ids[first - 1],
                        "next_trip_id": block.trip_ids[last + 1],
                        "gap_min": block.starts[last + 1] - block.ends[first - 1],
                        "original_releases_spanned": old_releases,
                        "after_is_release": connection.release,
                        "feasible": connection.feasible,
                        "source": connection.source,
                        "is_synthetic": connection.synthetic,
                    }
                )
        error = ""
        try:
            after = block_metrics(post.starts, post.ends, cfg.rules, post.trip_ids)
        except ValueError as exc:
            after = None
            error = str(exc)
        mechanisms = [_MECHANISM_COUNTS[name] for name, count in counts.items() if count]
        if not mechanisms:
            mechanisms = ["changed block boundary; no earlier finish or later start"]
        rows.append(
            {
                "block_id": key,
                "mechanisms": "; ".join(mechanisms),
                **counts,
                "trips_before": len(block.trip_ids),
                "trips_removed": len(block.trip_ids) - len(kept),
                "trips_after": len(kept),
                "removed_trip_ids": json.dumps([t for t in block.trip_ids if t in removed]),
                "vehicle_min_before": before["vehicle_min"],
                "vehicle_min_after": after["vehicle_min"] if after is not None else None,
                "vehicle_min_saved": before["vehicle_min"] - after["vehicle_min"]
                if after is not None
                else None,
                "revenue_min_removed": sum(
                    e - s
                    for t, s, e in zip(block.trip_ids, block.starts, block.ends)
                    if t in removed
                ),
                "deadhead_min_change": after["deadhead_min"] - before["deadhead_min"]
                if after is not None
                else None,
                "waiting_min_saved": before["layover_min"] - after["layover_min"]
                if after is not None
                else None,
                "pricing_status": "modeled" if after is not None else f"infeasible: {error}",
                "vehicle_start_before": _clock(start_before),
                "vehicle_start_after": _clock(start_after) if start_after is not None else "",
                "vehicle_finish_before": _clock(finish_before),
                "vehicle_finish_after": _clock(finish_after) if finish_after is not None else "",
                "start_later_min": start_after - start_before if start_after is not None else None,
                "finish_earlier_min": finish_before - finish_after
                if finish_after is not None
                else None,
                "first_departure_before": _clock(block.starts[0]),
                "first_departure_after": _clock(post.starts[0]) if kept else "",
                "last_arrival_before": _clock(block.ends[-1]),
                "last_arrival_after": _clock(post.ends[-1]) if kept else "",
                "first_revenue_trip_removed": block.trip_ids[0] in removed,
                "last_revenue_trip_removed": block.trip_ids[-1] in removed,
                "interior_changes": json.dumps(interior),
            }
        )
    return rows


def option_departure_effects(
    removed: set[str],
    route_dirs: Mapping[tuple[str, str], list[tuple[int, str]]],
    lookup: TripLookup,
) -> list[dict[str, Any]]:
    """Describe only changed departure gaps or spans, including outside-window neighbors."""
    rows = []
    keys = sorted({lookup.route_dir[t] for t in removed if t in lookup.route_dir})
    for rid, direction in keys:
        entries = route_dirs[(rid, direction)]
        ids = [tid for _, tid in entries]
        for first, last in _removed_runs(ids, removed):
            left = first - 1 if first else None
            right = last + 1 if last + 1 < len(entries) else None
            a = left if left is not None else first
            b = right if right is not None else last
            effect = (
                "enlarged departure gap"
                if left is not None and right is not None
                else "later first departure"
                if right is not None
                else "earlier last departure"
                if left is not None
                else "route-direction service removed"
            )
            rows.append(
                {
                    "route_id": rid,
                    "route_label": lookup.label[ids[first]],
                    "direction_id": direction,
                    "effect": effect,
                    "interval_start": _clock(entries[a][0]),
                    "interval_end": _clock(entries[b][0]),
                    "previous_retained_trip_id": ids[left] if left is not None else "",
                    "next_retained_trip_id": ids[right] if right is not None else "",
                    "removed_trip_ids": json.dumps(ids[first : last + 1]),
                    "removed_departures": json.dumps(
                        [_clock(entries[i][0]) for i in range(first, last + 1)]
                    ),
                    "original_headways_min": json.dumps(
                        [entries[i + 1][0] - entries[i][0] for i in range(a, b)]
                    ),
                    "resulting_gap_min": entries[right][0] - entries[left][0]
                    if left is not None and right is not None
                    else None,
                }
            )
    return rows


def annotate_option_effects(
    options: pd.DataFrame,
    source_table: pd.DataFrame,
    trip_table: pd.DataFrame,
    baseline: Mapping[str, BlockState],
    cfg: Config,
    day: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add operational explanations after ranking, leaving all existing scores untouched."""
    result = options.copy()
    context = option_scoring_context(trip_table, baseline, cfg)
    route_dirs = build_route_dir_index(source_table)
    lookup = build_trip_lookup(source_table)
    starts = source_table.start_min.to_dict()
    cache: dict[
        frozenset[str], tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]
    ] = {}
    annotations = []
    blocks = []
    gaps = []
    for row in options.to_dict("records"):
        removed = frozenset(json.loads(row["trip_ids"]))
        is_session = row["option_type"] == "frequency_session"
        if removed not in cache:
            block_rows = (
                option_block_effects(set(removed), baseline, cfg, context) if removed else []
            )
            gap_rows = option_departure_effects(set(removed), route_dirs, lookup) if removed else []
            counts = {name: sum(b[name] for b in block_rows) for name in _MECHANISM_COUNTS}
            summary = [
                f"{count} {_MECHANISM_COUNTS[name]}" for name, count in counts.items() if count
            ]
            if block_rows and not summary:
                summary = ["changed block boundary; no earlier finish or later start"]
            descriptions: dict[tuple[str, str], list[str]] = {}
            for gap in gap_rows:
                text = f"{gap['interval_start']}–{gap['interval_end']}"
                if gap["resulting_gap_min"] is not None:
                    text += f" ({gap['resulting_gap_min']} min gap)"
                else:
                    text += f" ({gap['effect']})"
                descriptions.setdefault((gap["route_label"], gap["direction_id"]), []).append(text)
            departures = [int(starts[t]) for t in removed if t in starts]
            info = {
                **counts,
                "savings_mechanism": "; ".join(summary) if removed else "not evaluated",
                "mechanism_status": "not evaluated"
                if not removed
                else "infeasible"
                if any(b["pricing_status"] != "modeled" for b in block_rows)
                else "modeled",
                "removed_departure_first": _clock(min(departures)) if departures else "",
                "removed_departure_last": _clock(max(departures)) if departures else "",
                "affected_service_summary": "; ".join(
                    f"{label} direction {direction}: {', '.join(values)}"
                    for (label, direction), values in descriptions.items()
                ),
                "changed_gap_count": sum(g["effect"] == "enlarged departure gap" for g in gap_rows),
            }
            cache[removed] = info, block_rows, gap_rows
        info, block_rows, gap_rows = cache[removed]
        annotations.append(
            {
                **info,
                "search_window_start": row["window_start"] if is_session else "",
                "search_window_end": row["window_end"] if is_session else "",
                "session_scope": "unsupported window"
                if not removed
                else "individual trip"
                if not is_session
                else "single-trip cut found by session search"
                if len(removed) == 1
                else "frequency reduction",
                "equivalent_single_trip_option_id": f"{day}:trip:{next(iter(removed))}"
                if is_session and len(removed) == 1
                else "",
            }
        )
        blocks.extend({"option_id": row["option_id"], **b} for b in block_rows)
        gaps.extend({"option_id": row["option_id"], **g} for g in gap_rows)
    extras = pd.DataFrame(annotations, index=result.index, columns=pd.Index(_EFFECT_COLUMNS))
    result = pd.concat([result, extras], axis=1)
    return (
        result,
        pd.DataFrame(blocks, columns=pd.Index(_BLOCK_EFFECT_COLUMNS)),
        pd.DataFrame(gaps, columns=pd.Index(_GAP_EFFECT_COLUMNS)),
    )


_FAMILY_COLUMNS = [
    "review_group_id",
    "review_group_rank",
    "representative_option_id",
    "is_group_representative",
    "alternative_rank",
    "distinct_alternatives",
    "grouping_reason",
    "exact_duplicate_of",
]


def grouped_option_review(
    options: pd.DataFrame,
    cfg: Config,
    day: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group eligible alternatives around their best-ranked representative, never transitively.

    Singles stay separate. Sessions must have the same routes and headway request.
    Nested removal sets belong together, as do overlapping cuts to the same blocks
    or high-Jaccard-overlap alternatives. Disjoint sets never join. Full details keep
    exact duplicates; the alternatives file displays each distinct removal set once.
    """
    result = options.copy()
    annotations: list[dict[str, Any]] = [{} for _ in range(len(result))]
    families: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    ranks: dict[str, int] = {}
    selected_ids: set[str] = set()
    family_members: dict[str, list[int]] = {}
    for index, row in enumerate(options.to_dict("records")):
        if row["candidate_classification"] == "ineligible":
            continue
        kind = row["option_type"]
        removed = set(json.loads(row["trip_ids"]))
        block_ids = set(json.loads(row.get("affected_blocks") or "[]"))
        routes = tuple(sorted(json.loads(row["route_ids"])))
        key = (
            (kind, routes, row["from_headway_min"], row["to_headway_min"])
            if kind == "frequency_session"
            else (kind, row["option_id"])
        )
        chosen: Optional[dict[str, Any]] = None
        reason = "highest-ranked option in this review group"
        for family in families.get(key, []):
            previous = family["removed"]
            intersection = removed & previous
            if not intersection:
                continue
            if removed == previous:
                reason = "identical removed trips to representative"
            elif removed <= previous or previous <= removed:
                reason = "nested removed-trip sets"
            elif block_ids and block_ids == family["blocks"]:
                reason = "overlapping cuts to the same blocks"
            elif (
                cfg.session_shortlist_overlap > 0
                and len(intersection) / len(removed | previous) >= cfg.session_shortlist_overlap
            ):
                reason = "high removed-trip overlap with representative"
            else:
                continue
            chosen = family
            break
        if chosen is None:
            ranks[kind] = ranks.get(kind, 0) + 1
            group_id = f"{day}:review:{kind}:{ranks[kind]}"
            chosen = {
                "id": group_id,
                "rank": ranks[kind],
                "option_id": row["option_id"],
                "removed": removed,
                "blocks": block_ids,
                "distinct": {},
            }
            families.setdefault(key, []).append(chosen)
            if not cfg.shortlist_limit_per_type or ranks[kind] <= cfg.shortlist_limit_per_type:
                selected_ids.add(group_id)
        frozen = frozenset(removed)
        duplicate = chosen["distinct"].get(frozen)
        if duplicate is None:
            chosen["distinct"][frozen] = (row["option_id"], len(chosen["distinct"]) + 1)
        original_id, alternative_rank = chosen["distinct"][frozen]
        annotations[index] = {
            "review_group_id": chosen["id"],
            "review_group_rank": chosen["rank"],
            "representative_option_id": chosen["option_id"],
            "is_group_representative": row["option_id"] == chosen["option_id"],
            "alternative_rank": alternative_rank,
            "grouping_reason": reason,
            "exact_duplicate_of": original_id if duplicate is not None else "",
        }
        family_members.setdefault(chosen["id"], []).append(index)
    for collection in families.values():
        for family in collection:
            for index in family_members[family["id"]]:
                annotations[index]["distinct_alternatives"] = len(family["distinct"])
    extras = pd.DataFrame(annotations, index=result.index, columns=pd.Index(_FAMILY_COLUMNS))
    for name in ("review_group_rank", "alternative_rank", "distinct_alternatives"):
        extras[name] = extras[name].astype("Int64")
    result = pd.concat([result, extras], axis=1)
    chosen_rows = result.review_group_id.isin(selected_ids)
    leaders = result[chosen_rows & result.is_group_representative.eq(True)].copy()
    alternatives = result[chosen_rows & result.exact_duplicate_of.eq("")].copy()
    alternatives = alternatives.sort_values(
        ["option_type", "review_group_rank", "alternative_rank"], kind="stable"
    )
    return result, planner_column_order(leaders), planner_column_order(alternatives)


def planner_column_order(table: pd.DataFrame) -> pd.DataFrame:
    """Put evidence, scores, and actual service effects ahead of long identifier lists."""
    leading = [
        "review_group_id",
        "review_group_rank",
        "alternative_rank",
        "distinct_alternatives",
        "option_id",
        "option_type",
        "route_label",
        "direction_id",
        "candidate_classification",
        "boardings",
        "vehicle_min_saved",
        "boardings_per_vehicle_hour_saved",
        "savings_mechanism",
        "session_scope",
        "removed_departure_first",
        "removed_departure_last",
        "affected_service_summary",
        "search_window_start",
        "search_window_end",
        "from_headway_min",
        "to_headway_min",
        "trip_count",
    ]
    return table[
        [c for c in leading if c in table] + [c for c in table.columns if c not in leading]
    ].reset_index(drop=True)


# =============================================================================
# FUTURE MODULE BOUNDARY: PROPOSAL WORKBOOK EXPORTER
# =============================================================================
# This section consumes saved analysis tables; it must not score options, match
# ridership again, select a new service day, or reblock vehicles. The handoff is
# WORKBOOK_TRIPS_FILENAME + the four existing option tables below + a small JSON
# metadata file. Move this section to proposal_workbooks.py if splitting later.
# The analysis-side adapter build_workbook_trip_context() follows this section.

WORKBOOK_TRIPS_FILENAME = "trip_cut_workbook_trips.csv"
WORKBOOK_INDEX_FILENAME = "trip_cut_workbook_index.csv"
_WORKBOOK_TABLES = (
    WORKBOOK_TRIPS_FILENAME,
    OPTIONS_FILENAME,
    GROUPED_SHORTLIST_FILENAME,
    OPTION_BLOCKS_FILENAME,
    OPTION_GAPS_FILENAME,
)
_WORKBOOK_NUMERIC_FIELDS = {
    "start_min",
    "end_min",
    "runtime_min",
    "boardings",
    "observations",
    "trip_count",
    "vehicle_min_saved",
    "revenue_min_removed",
    "deadhead_min_change",
    "waiting_min_saved",
    "boardings_per_vehicle_hour_saved",
    "from_headway_min",
    "to_headway_min",
    "review_group_rank",
    "distinct_alternatives",
    "resulting_gap_min",
    "trips_before",
    "trips_removed",
    "trips_after",
    "vehicle_min_before",
    "vehicle_min_after",
    "start_later_min",
    "finish_earlier_min",
}


def select_workbook_options(tables: Mapping[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Select independent options without changing scores or merging alternatives.

    Explicit IDs are matched within this day's tables; the caller checks that all
    requested IDs were found across the configured days. Explicit IDs may select
    an alternative, but cannot select an ineligible option.
    """
    options = tables[OPTIONS_FILENAME]
    if cfg.workbook_option_ids:
        indexed = options.set_index("option_id", drop=False)
        ids = [key for key in dict.fromkeys(cfg.workbook_option_ids) if key in indexed.index]
        selected = indexed.loc[ids].reset_index(drop=True)
        bad = selected[selected.candidate_classification.eq("ineligible")]
        if not bad.empty:
            raise ValueError("Cannot export ineligible proposals: " + ", ".join(bad.option_id))
        return selected
    representatives = tables[GROUPED_SHORTLIST_FILENAME]
    selected_rows = []
    counts: dict[str, int] = {}
    seen: set[frozenset[str]] = set()
    for row in representatives.to_dict("records"):
        kind = str(row["option_type"])
        removed = frozenset(json.loads(row["trip_ids"]))
        if row["candidate_classification"] == "ineligible" or not removed or removed in seen:
            continue
        if cfg.workbook_limit_per_type and counts.get(kind, 0) >= cfg.workbook_limit_per_type:
            continue
        selected_rows.append(row)
        seen.add(removed)
        counts[kind] = counts.get(kind, 0) + 1
    return pd.DataFrame(selected_rows, columns=options.columns)


def _report_value(value: Any) -> Any:
    """Normalize pandas scalars for XLSX without turning absent data into zero."""
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _report_range(first: Any, last: Any) -> str:
    """Format an optional pair consistently for in-memory and saved CSV inputs."""
    values = []
    for value in (first, last):
        value = _report_value(value)
        values.append(
            "" if value is None else f"{value:g}" if isinstance(value, (int, float)) else str(value)
        )
    return " to ".join(values) if any(values) else ""


def _report_sheet_name(label: str, used: set[str]) -> str:
    """Return a unique Excel-safe title, preserving the full identifier in cells."""
    base = re.sub(r"[\x00-\x1f\[\]:*?/\\]", "_", label).strip(" '") or "Sheet"
    name = base[:31]
    suffix = 1
    while name.casefold() in used:
        suffix += 1
        tail = f"_{suffix}"
        name = base[: 31 - len(tail)] + tail
    used.add(name.casefold())
    return name


def _report_cell(ws: Any, row: int, column: int, value: Any) -> Any:
    """Write identifiers and externally supplied text literally, never as formulas."""
    value = _report_value(value)
    if isinstance(value, str):
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    cell = ws.cell(row, column, value)
    if isinstance(value, str):
        cell.data_type = "s"
    return cell


def _report_sheet(book: Any, name: str, title: str, context: str, label: str) -> Any:
    """Create a plain sheet with space for context and a table starting on row 6."""
    from openpyxl.styles import Font

    ws = book.create_sheet(name)
    ws.sheet_view.showGridLines = False
    _report_cell(ws, 2, 1, title).font = Font(name="Arial", size=14, bold=True)
    _report_cell(ws, 3, 1, context).font = Font(name="Arial", size=10)
    if label:
        _report_cell(ws, 4, 1, label).font = Font(name="Arial", size=10, bold=True, color="9C0006")
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    return ws


def _report_table(
    ws: Any,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    cut_rows: Optional[set[int]] = None,
    time_columns: Sequence[int] = (),
    start_row: int = 6,
) -> None:
    """Write a filterable static report; row offsets in cut_rows are zero-based."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    for column, (header, width) in enumerate(zip(headers, widths), 1):
        cell = _report_cell(ws, start_row, column, header)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="34465A")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(column)].width = width
    ws.row_dimensions[start_row].height = 34
    for offset, values in enumerate(rows):
        row_num = start_row + 1 + offset
        ws.row_dimensions[row_num].height = 30
        for column, value in enumerate(values, 1):
            cell = _report_cell(ws, row_num, column, value)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = "#,##0" if float(cell.value).is_integer() else "#,##0.00"
            if column in time_columns:
                cell.number_format = "[h]:mm"
            if cut_rows and offset in cut_rows:
                cell.fill = PatternFill("solid", fgColor="FCE4D6")
    last = get_column_letter(len(headers))
    ws.auto_filter.ref = f"A{start_row}:{last}{max(start_row, ws.max_row)}"
    ws.freeze_panes = f"C{start_row + 1}"
    ws.print_title_rows = f"1:{start_row}"
    ws.print_options.horizontalCentered = False
    ws.print_area = f"A1:{last}{ws.max_row}"


def workbook_trip_rows(
    trips: pd.DataFrame, removed: set[str], route_view: bool
) -> tuple[list[list[Any]], set[int]]:
    """Create schedule rows and gaps; a route gap is measured within each direction.

    Block gaps are arrival-to-departure intervals, NOT estimates of layover or
    release time. A missing endpoint prevents gap comparisons for that group.
    No block or departure is reassigned. Times retain the service-day clock.
    """
    keys = ["direction_id"] if route_view else ["block_key"]
    ordered = trips.sort_values([*keys, "start_min", "trip_id"], na_position="last").copy()
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    incomplete: set[str] = set()
    for _, group in ordered.groupby(keys, dropna=False, sort=False):
        if group[["start_min", "end_min"]].isna().any().any():
            incomplete.update(group.trip_id)
            continue
        kept = group[~group.trip_id.isin(removed)]
        for data, result in ((group, before), (kept, after)):
            previous = None
            for row in data.to_dict("records"):
                if previous is not None:
                    result[row["trip_id"]] = row["start_min"] - previous
                previous = row["start_min"] if route_view else row["end_min"]
    rows: list[list[Any]] = []
    cuts: set[int] = set()
    for record in ordered.to_dict("records"):
        tid = record["trip_id"]
        if tid in removed:
            cuts.add(len(rows))
        flags = str(record.get("data_flags", ""))
        if tid in incomplete:
            flags += "; gap unavailable: incomplete schedule"
        rows.append(
            [
                "PROPOSED CUT" if tid in removed else "KEEP",
                record["block_id"] or "UNASSIGNED",
                record["route_label"],
                record["direction_id"],
                tid,
                record["start_min"] / 1440 if pd.notna(record["start_min"]) else None,
                record["end_min"] / 1440 if pd.notna(record["end_min"]) else None,
                record.get("boardings"),
                before.get(tid),
                after.get(tid),
                record.get("observations"),
                flags.strip("; "),
                record.get("start_stop_name", ""),
                record.get("end_stop_name", ""),
                record["route_id"],
                record["service_id"],
                record.get("start_stop_id", ""),
                record.get("end_stop_id", ""),
            ]
        )
    return rows, cuts


def write_proposal_workbook(
    target: Path,
    option: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one complete, independent proposal from the analysis handoff.

    Include every block serving a directly impacted route, including every other
    route on those blocks. Do not expand again through those interlined routes.
    Static values are intentional: editing this report does not rerun the model.
    """
    from openpyxl import Workbook

    context = tables[WORKBOOK_TRIPS_FILENAME]
    removed = set(json.loads(option["trip_ids"]))
    available = set(context.trip_id)
    if not removed or removed - available:
        raise ValueError(f"Option {option['option_id']} has empty or missing workbook trip IDs.")
    if context.trip_id.duplicated().any():
        raise ValueError("Duplicate trip IDs in workbook handoff.")
    cut = context[context.trip_id.isin(removed)]
    route_ids = set(cut.route_id)
    routes = context[context.route_id.isin(route_ids)]
    block_keys = set(routes.block_key)
    scoped = context[context.block_key.isin(block_keys)]
    # Reconcile exact removed-trip ridership against the scored option.
    actual = pd.to_numeric(cut.boardings, errors="coerce")
    if actual.isna().any() or not math.isclose(
        float(actual.sum()), float(option["boardings"]), rel_tol=1e-7, abs_tol=0.011
    ):
        raise ValueError(f"Workbook ridership does not reconcile for {option['option_id']}.")
    effects = tables[OPTION_BLOCKS_FILENAME]
    effects = effects[effects.option_id.eq(option["option_id"])]
    if effects.empty or not math.isclose(
        float(pd.to_numeric(effects.vehicle_min_saved).sum()),
        float(option["vehicle_min_saved"]),
        rel_tol=1e-7,
        abs_tol=0.011,
    ):
        raise ValueError(f"Workbook block savings do not reconcile for {option['option_id']}.")
    effect_lookup = effects.set_index("block_id").to_dict("index")
    book = Workbook()
    book.remove(book.active)
    used: set[str] = {"summary", "block overview", "changed gaps"}
    route_tabs = {
        rid: _report_sheet_name(f"Route {group.iloc[0].route_label}", used)
        for rid, group in routes.groupby("route_id", sort=True)
    }
    block_tabs = {key: _report_sheet_name(f"Block {key}", used) for key in sorted(block_keys)}
    label = str(metadata.get("data_label", ""))
    if bool(option.get("deadhead_inputs_synthetic", False)):
        label = "; ".join(filter(None, [label, "SYNTHETIC DEADHEAD INPUTS: testing only"]))
    day_context = (
        f"{metadata.get('day', '')}, {metadata.get('analysis_date', '')}, {option['option_id']}"
    )
    summary = _report_sheet(
        book, "Summary", f"Proposal: {option['route_label']}", day_context, label
    )
    summary.sheet_properties.tabColor = "34465A"
    metrics = [
        ("Candidate classification", option["candidate_classification"]),
        ("Option type", option["option_type"]),
        ("Trips proposed for removal", len(removed)),
        ("Average daily boardings on removed trips", option["boardings"]),
        ("Modeled vehicle minutes saved", option["vehicle_min_saved"]),
        ("Boardings per vehicle hour saved", option["boardings_per_vehicle_hour_saved"]),
        ("Savings mechanism", option.get("savings_mechanism", "")),
        ("Session scope", option.get("session_scope", "")),
        (
            "Search window",
            _report_range(option.get("search_window_start"), option.get("search_window_end")),
        ),
        (
            "Removed departures",
            _report_range(
                option.get("removed_departure_first"), option.get("removed_departure_last")
            ),
        ),
        (
            "Requested headways (minutes)",
            _report_range(option.get("from_headway_min"), option.get("to_headway_min")),
        ),
        ("Service effects", option.get("affected_service_summary", "")),
        ("Review group", option.get("review_group_id", "")),
        ("Distinct alternatives in group", option.get("distinct_alternatives")),
        (
            "Blocks with proposed cuts / blocks shown",
            f"{len(set(cut.block_key))} / {len(block_keys)}",
        ),
        ("Trips shown without matched ridership", int(scoped.boardings.isna().sum())),
        (
            "Interpretation",
            "One independent alternative. Savings cannot be added across overlapping proposals. "
            "Vehicle time is not paid operator time.",
        ),
        (
            "Workbook use",
            "Static report. Regenerate after changing inputs or proposals. Orange rows are "
            "proposed cuts; KEEP rows remain scheduled.",
        ),
        (
            "Schedule scope",
            "Full service day for all blocks serving the directly impacted routes, including their "
            "other routes. Unchanged blocks are included.",
        ),
        (
            "Departure gaps",
            "Route gaps compare consecutive departures within a direction, across all patterns. "
            "They do not prove equivalent stop coverage.",
        ),
        (
            "Block gaps",
            "Arrival-to-departure minutes, not layover or deadhead estimates. Proposed times and "
            "block assignments remain unchanged.",
        ),
        (
            "Blocking",
            "No reblocking. Combining remaining work from different blocks requires a separate "
            "feasibility analysis.",
        ),
        ("Deadhead basis", metadata.get("deadhead_basis", "")),
        ("Service IDs", metadata.get("service_ids", "")),
        ("GTFS source", metadata.get("gtfs_path", "")),
        ("Ridership source", metadata.get("ridership_file", "")),
        (
            "Run log",
            "trip_cut_savings_runlog.txt in the source results folder records input fingerprints "
            "and settings.",
        ),
    ]
    _report_table(summary, ["Measure", "Value"], metrics, [45, 105])
    summary.auto_filter.ref = None
    summary.freeze_panes = None
    for number in range(7, summary.max_row + 1):
        summary.row_dimensions[number].height = 32
    overview = _report_sheet(
        book, "Block overview", "Blocks serving the impacted routes", day_context, label
    )
    block_rows = []
    removed_blocks: set[int] = set()
    for key, group in scoped.groupby("block_key", sort=True):
        count = int(group.trip_id.isin(removed).sum())
        entire = count == len(group)
        if entire:
            removed_blocks.add(len(block_rows))
        effect = effect_lookup.get(key, {})
        block_rows.append(
            [
                key,
                "WHOLE BLOCK CUT" if entire else "PARTIAL CUT" if count else "UNCHANGED",
                ", ".join(sorted(set(group.route_label))),
                len(group),
                count,
                len(group) - count,
                effect.get("vehicle_min_saved", 0),
                effect.get("mechanisms", ""),
                effect.get("vehicle_start_before"),
                effect.get("vehicle_start_after"),
                effect.get("vehicle_finish_before"),
                effect.get("vehicle_finish_after"),
                block_tabs[key],
            ]
        )
    _report_table(
        overview,
        [
            "Block",
            "Proposal status",
            "Routes",
            "Trips before",
            "Trips cut",
            "Trips after",
            "Vehicle min saved",
            "Mechanism",
            "Vehicle start before",
            "Vehicle start after",
            "Vehicle finish before",
            "Vehicle finish after",
            "Open block tab",
        ],
        block_rows,
        [19, 23, 24, 11, 11, 11, 15, 40, 15, 15, 15, 15, 30],
        removed_blocks,
    )
    for number, record in enumerate(block_rows, 7):
        overview.cell(
            number, 13
        ).hyperlink = f"#'{str(record[12]).replace(chr(39), chr(39) * 2)}'!A1"
    for route_view, groups, tabs in (
        (True, routes.groupby("route_id", sort=True), route_tabs),
        (False, scoped.groupby("block_key", sort=True), block_tabs),
    ):
        for key, group in groups:
            heading = (
                f"Route {group.iloc[0].route_label}: all blocks"
                if route_view
                else f"Block {key}: complete day"
            )
            ws = _report_sheet(book, tabs[key], heading, day_context, label)
            rows, highlighted = workbook_trip_rows(group, removed, route_view)
            gap_label = "Departure gap" if route_view else "Inter-trip gap"
            _report_table(
                ws,
                [
                    "Action",
                    "Block",
                    "Route",
                    "Direction",
                    "Trip ID",
                    "Departure",
                    "Arrival",
                    "Avg daily boardings",
                    f"{gap_label} before (min)",
                    f"{gap_label} after (min)",
                    "Observations",
                    "Data flags",
                    "From terminal",
                    "To terminal",
                    "Route ID",
                    "Service ID",
                    "From stop ID",
                    "To stop ID",
                ],
                rows,
                [20, 16, 11, 11, 16, 12, 12, 16, 18, 18, 14, 44, 48, 48, 14, 14, 16, 16],
                highlighted,
                time_columns=(6, 7),
            )
            if not route_view and len(highlighted) == len(rows):
                ws.sheet_properties.tabColor = "C65911"
            for number, record in enumerate(rows, 7):
                if record[7] is None or pd.isna(record[7]):
                    from openpyxl.styles import PatternFill

                    ws.cell(number, 8).fill = PatternFill("solid", fgColor="FFF2CC")
    gaps = tables[OPTION_GAPS_FILENAME]
    gaps = gaps[gaps.option_id.eq(option["option_id"])]
    if not gaps.empty:
        ws = _report_sheet(
            book,
            "Changed gaps",
            "Changed route departure gaps and service spans",
            day_context,
            label,
        )
        fields = [
            "route_label",
            "direction_id",
            "effect",
            "interval_start",
            "interval_end",
            "original_headways_min",
            "resulting_gap_min",
            "removed_departures",
            "previous_retained_trip_id",
            "next_retained_trip_id",
        ]
        _report_table(
            ws,
            [
                "Route",
                "Direction",
                "Effect",
                "Interval start",
                "Interval end",
                "Original gaps (min)",
                "Resulting gap (min)",
                "Removed departures",
                "Previous retained trip",
                "Next retained trip",
            ],
            gaps[fields].to_numpy().tolist(),
            [13, 12, 32, 16, 16, 26, 20, 48, 24, 24],
        )
    # Atomic replacement prevents a failed export leaving an apparently complete file.
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".xlsx", dir=target.parent)
    os.close(fd)
    try:
        book.save(temporary)
        os.replace(temporary, target)
    finally:
        book.close()
        if Path(temporary).exists():
            Path(temporary).unlink()
    return {
        "option_id": option["option_id"],
        "option_type": option["option_type"],
        "route_label": option["route_label"],
        "candidate_classification": option["candidate_classification"],
        "trips_cut": len(removed),
        "blocks_shown": len(block_keys),
        "boardings": option["boardings"],
        "vehicle_min_saved": option["vehicle_min_saved"],
        "workbook": str(target.relative_to(target.parent.parent)),
    }


def export_proposal_workbooks(
    tables: Mapping[str, pd.DataFrame], metadata: Mapping[str, Any], cfg: Config, day: str
) -> pd.DataFrame:
    """Export this day's selected options and a manifest; never delete older files."""
    selected = select_workbook_options(tables, cfg)
    entries = []
    for option in selected.to_dict("records"):
        name = re.sub(r"[^A-Za-z0-9_-]", "_", str(option["option_id"]))[:90]
        digest = hashlib.sha256(str(option["option_id"]).encode()).hexdigest()[:10]
        target = cfg.output_dir / "proposal_workbooks" / f"{name}_{digest}.xlsx"
        info = dict(metadata)
        if cfg.workbook_data_label:
            info["data_label"] = cfg.workbook_data_label
        entries.append(write_proposal_workbook(target, option, tables, info))
        logging.info("Wrote proposal workbook: %s", target)
    result = pd.DataFrame(
        entries,
        columns=pd.Index(
            [
                "option_id",
                "option_type",
                "route_label",
                "candidate_classification",
                "trips_cut",
                "blocks_shown",
                "boardings",
                "vehicle_min_saved",
                "workbook",
            ]
        ),
    )
    result.to_csv(cfg.output_dir / _day_filename(WORKBOOK_INDEX_FILENAME, day), index=False)
    return result


def export_saved_workbooks(source: Path, cfg: Config) -> None:
    """Re-export version-1 handoffs without reading inputs or rerunning analysis."""
    paths = sorted(source.glob("trip_cut_workbook_context_*.json"))
    if not paths:
        raise ValueError("No workbook handoff found. Run the updated analysis once first.")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("handoff_version") != 1:
            raise ValueError(f"Unsupported workbook handoff: {path}")
        day = metadata["day"]
        tables = {}
        for filename in _WORKBOOK_TABLES:
            table = pd.read_csv(
                source / _day_filename(filename, day), dtype=str, keep_default_na=False
            )
            for field in _WORKBOOK_NUMERIC_FIELDS.intersection(table.columns):
                table[field] = pd.to_numeric(table[field], errors="coerce")
            if "deadhead_inputs_synthetic" in table:
                table["deadhead_inputs_synthetic"] = table.deadhead_inputs_synthetic.eq("True")
            tables[filename] = table
        found.update(tables[OPTIONS_FILENAME].option_id)
        export_proposal_workbooks(tables, metadata, cfg, day)
    missing = set(cfg.workbook_option_ids) - found
    if missing:
        raise ValueError(
            "Requested workbook option IDs were not found: " + ", ".join(sorted(missing))
        )


# =============================================================================
# END FUTURE WORKBOOK MODULE / ANALYSIS-SIDE HANDOFF ADAPTER
# =============================================================================


def build_workbook_trip_context(
    passenger_trips: pd.DataFrame,
    trips_out: pd.DataFrame,
    feed: Mapping[str, Optional[pd.DataFrame]],
    service_ids: Optional[set[str]],
    filters: ResolvedRouteFilters,
) -> pd.DataFrame:
    """Attach the original schedule and already-matched ridership for later review.

    Retain QA-excluded blocks and trips lacking usable endpoint times as context,
    with blank ridership and explicit flags. Configured fake routes remain omitted.
    This adapter does not reprice any option or rematch any ridership records.
    """
    context = passenger_trips.reset_index().copy()
    raw = _required(feed, "trips").copy()
    if service_ids is not None:
        raw = raw[raw.service_id.isin(service_ids)]
    raw = raw[~raw.trip_id.isin(context.trip_id)].copy()
    route_table = _required(feed, "routes")
    labels = (
        route_table.set_index("route_id").route_short_name.fillna("").to_dict()
        if "route_short_name" in route_table
        else {}
    )
    raw["route_label"] = raw.route_id.map(lambda rid: labels.get(rid, "") or rid)
    raw = raw[~raw.route_id.isin(filters.non_revenue) & ~raw.route_label.isin(filters.non_revenue)]
    if not raw.empty:
        context = pd.concat([context, raw.reindex(columns=context.columns)], ignore_index=True)
    for field in ("block_id", "direction_id", "start_stop_id", "end_stop_id"):
        context[field] = context[field].fillna("").astype(str)
    context["block_key"] = [
        block or _NO_BLOCK_PREFIX + tid for block, tid in zip(context.block_id, context.trip_id)
    ]
    details = trips_out.set_index("trip_id")
    for field in ("boardings", "observations", "low_observations", "ineligible_reason"):
        context[field] = context.trip_id.map(details[field])
    stops = feed.get("stops")
    stop_names = (
        stops.set_index("stop_id").stop_name.to_dict()
        if stops is not None and "stop_name" in stops
        else {}
    )
    for end in ("start", "end"):
        context[f"{end}_stop_name"] = context[f"{end}_stop_id"].map(stop_names).fillna("")
    flags = []
    for row in context.to_dict("records"):
        notes = []
        if pd.isna(row["boardings"]):
            notes.append("MISSING RIDERSHIP")
        if row["low_observations"] is True:
            notes.append("LOW OBSERVATIONS")
        if row["trip_id"] not in details.index:
            notes.append("QA EXCLUDED: context only")
        if pd.isna(row["start_min"]) or pd.isna(row["end_min"]):
            notes.append("MISSING ENDPOINT TIMES")
        if not row["block_id"]:
            notes.append("NO BLOCK ASSIGNMENT")
        reason = _report_value(row["ineligible_reason"])
        if reason:
            notes.append(str(reason))
        flags.append("; ".join(notes))
    context["data_flags"] = flags
    return context.sort_values(
        ["block_key", "start_min", "trip_id"], na_position="last"
    ).reset_index(drop=True)


def block_report(
    baseline: Mapping[str, BlockState],
    trip_table: pd.DataFrame,
    rules: BlockRules,
) -> pd.DataFrame:
    """Per block: trips, span, pieces, and revenue, layover, deadhead, and vehicle hours."""
    rows: list[dict[str, Any]] = []
    for block_id, block in baseline.items():
        m = block_metrics(block.starts, block.ends, rules, block.trip_ids)
        labels = trip_table.loc[block.trip_ids, "route_label"].astype(str)
        rows.append(
            {
                "block_id": block_id,
                "routes": ";".join(dict.fromkeys(labels)),
                "trips": m["trips"],
                "start": minutes_to_hhmm(block.starts[0]) if block.starts else "",
                "end": minutes_to_hhmm(max(block.ends)) if block.ends else "",
                "pieces": m["pieces"],
                "revenue_hours": round(m["revenue_min"] / 60.0, 2),
                "layover_hours": round(m["layover_min"] / 60.0, 2),
                "deadhead_hours": round(m["deadhead_min"] / 60.0, 2),
                "vehicle_hours": round(m["vehicle_min"] / 60.0, 2),
            }
        )
    return pd.DataFrame(rows)


def parse_periods(periods: Mapping[str, tuple[str, str]]) -> list[tuple[str, int, int]]:
    """Turn ``{"NAME": ("HH:MM", "HH:MM")}`` into (name, start, end) minute windows.

    Raises:
        ValueError: If a bound is not ``HH:MM`` or a window has zero length.
    """
    parsed: list[tuple[str, int, int]] = []
    for name, bounds in periods.items():
        if len(bounds) != 2:
            raise ValueError(f"TIME_PERIODS[{name!r}] must be a (start, end) pair.")
        start = parse_time_to_minutes(str(bounds[0]))
        end = parse_time_to_minutes(str(bounds[1]))
        if start is None or end is None:
            raise ValueError(f"TIME_PERIODS[{name!r}] bounds must be HH:MM; got {bounds!r}.")
        if start % 1440 == end % 1440:
            raise ValueError(f"TIME_PERIODS[{name!r}] has zero length.")
        parsed.append((str(name), start % 1440, end % 1440))
    return parsed


def assign_period(start_min: int, periods: Sequence[tuple[str, int, int]]) -> str:
    """Name the window containing a trip's start (clock time, wrapping midnight)."""
    if not periods:
        return _ALL_DAY
    clock = start_min % 1440
    for name, start, end in periods:
        inside = start <= clock < end if start < end else clock >= start or clock < end
        if inside:
            return name
    return "UNASSIGNED"


def observed_cycles(
    blocks: Mapping[str, BlockState], lookup: TripLookup, rules: BlockRules
) -> dict[str, int]:
    """Minutes from each trip's start to the vehicle's next start in the same direction.

    Follows the trip's block forward while the vehicle stays on the same route and is
    not released; the first later trip in the same direction ends the cycle. Trips
    whose vehicle interlines, is released, or never returns get no cycle.
    """
    cycles: dict[str, int] = {}
    for block in blocks.values():
        if block.block_id.startswith(_NO_BLOCK_PREFIX):
            continue
        ids = block.trip_ids
        routes = [lookup.route_dir[t][0] for t in ids]
        dirs = [lookup.route_dir[t][1] for t in ids]
        for k in range(len(ids)):
            for m in range(k + 1, len(ids)):
                if routes[m] != routes[k]:
                    break
                release = (
                    rules.model.connection(ids[m - 1], ids[m]).release
                    if rules.model
                    else block.starts[m] - block.ends[m - 1] >= rules.min_release_gap_min
                )
                if release:
                    break
                if dirs[m] == dirs[k]:
                    cycles[ids[k]] = block.starts[m] - block.starts[k]
                    break
    return cycles


def cycle_time_table(
    trip_table: pd.DataFrame,
    blocks: Mapping[str, BlockState],
    rules: BlockRules,
    periods: Sequence[tuple[str, int, int]],
    recovery_target_pct: float,
) -> pd.DataFrame:
    """Per route × time period: running time, recovery, cycle time, headway, vehicles.

    Round-trip running time sums the mean running time of each direction operating
    in the period. Cycle time is the mean of :func:`observed_cycles`, and recovery
    is cycle time minus running time. Headway is the median gap between consecutive
    departures in the period's busier direction, ordered on the window's own clock
    so a window wrapping midnight is not split in two. Vehicles needed are cycle time
    over headway, rounded up, both at the observed cycle and at running time plus
    ``recovery_target_pct``. When the first exceeds the second, the difference is how
    many vehicles trimming running or recovery time could free at that headway.

    Args:
        trip_table: Trips to describe (one row per trip, indexed by trip_id).
        blocks: The blocks those trips run in.
        rules: Block savings rules (release threshold ends a cycle).
        periods: Output of :func:`parse_periods`.
        recovery_target_pct: Recovery the schedule is taken to need, percent.

    Returns:
        One row per (route, period) with at least one trip.
    """
    if trip_table.empty:
        return pd.DataFrame()
    cycles = observed_cycles(blocks, build_trip_lookup(trip_table), rules)
    frame = trip_table.reset_index()
    frame["period"] = frame["start_min"].map(lambda m: assign_period(int(m), periods))
    frame["cycle_min"] = frame["trip_id"].map(cycles)
    window_start = {name: start for name, start, _ in periods}
    frame["window_min"] = [
        (int(m) - window_start[p]) % 1440 if p in window_start else int(m)
        for m, p in zip(frame["start_min"], frame["period"])
    ]
    rows: list[dict[str, Any]] = []
    for group_key, sub in frame.groupby(["route_id", "period"], sort=True):
        route_id, period = cast("tuple[str, str]", group_key)
        by_dir = sub.groupby("direction_id")
        running = float(by_dir["runtime_min"].mean().sum())
        busiest: Optional[pd.DataFrame] = None
        for _, dir_rows in by_dir:
            if busiest is None or len(dir_rows) > len(busiest):
                busiest = dir_rows
        headway = float("nan")
        if busiest is not None and len(busiest) >= 2:
            headway = float(busiest["window_min"].sort_values().diff().dropna().median())
        cycle = float(sub["cycle_min"].mean()) if sub["cycle_min"].notna().any() else float("nan")
        recovery = cycle - running
        target_recovery = running * recovery_target_pct / 100.0
        need_cycle = _vehicles(cycle, headway)
        need_target = _vehicles(running + target_recovery, headway)
        rows.append(
            {
                "route_id": route_id,
                "route_label": str(sub["route_label"].iloc[0]),
                "period": period,
                "trips": len(sub),
                "round_trip_running_min": round(running, 1),
                "cycle_min": round(cycle, 1),
                "recovery_min": round(recovery, 1),
                "recovery_pct": round(100.0 * recovery / running, 1) if running > 0 else math.nan,
                "excess_recovery_min": round(recovery - target_recovery, 1),
                "headway_min": round(headway, 1),
                "vehicles_at_cycle": need_cycle,
                "vehicles_at_target_recovery": need_target,
                "vehicles_freeable_by_runtime_trim": (
                    max(0, need_cycle - need_target)
                    if need_cycle is not None and need_target is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _vehicles(cycle: float, headway: float) -> Optional[int]:
    """Vehicles a cycle needs at a headway, rounded up; ``None`` if either is unknown."""
    if math.isnan(cycle) or math.isnan(headway) or headway <= 0:
        return None
    return int(math.ceil(cycle / headway - 1e-9))


def vehicle_profile(
    baseline: Mapping[str, BlockState],
    rules: BlockRules,
    bin_min: int,
) -> pd.DataFrame:
    """Vehicles in service at the start of each time bin on the analysis day.

    Includes pull-out and pull-in durations when the endpoint-aware model is active.
    Negative labels indicate inferred activity before service-day midnight. This
    sampled profile is not a fleet-reduction or operator-cost guarantee.
    """
    pieces = [
        p
        for b in baseline.values()
        for p in (
            rules.model.pieces(b.trip_ids, include_deadhead=True)
            if rules.model
            else block_pieces(b.starts, b.ends, rules)
        )
    ]
    if not pieces:
        return pd.DataFrame(columns=pd.Index(["time", "vehicles"]))
    first = min(p[0] for p in pieces) // bin_min * bin_min
    last = max(p[1] for p in pieces)
    rows = []
    for t in range(first, last + 1, bin_min):
        rows.append(
            {
                "time": ("-" + minutes_to_hhmm(-t)) if t < 0 else minutes_to_hhmm(t),
                "vehicles": sum(1 for start, end in pieces if start <= t < end),
            }
        )
    return pd.DataFrame(rows)


def eligibility_reasons(
    trip_table: pd.DataFrame,
    boardings: Mapping[str, float],
    guards: Guards,
    routes_to_include: Sequence[str],
    routes_to_exclude: Sequence[str],
    non_revenue_routes: Sequence[str] = (),
    exclusion_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict[str, str]:
    """Return each trip's reason it cannot be cut, or ``""`` when it can.

    Unmatched route entries generate warnings. Category names are included in
    exclusion reasons when their routes match.
    """
    tokens = {
        tid: {str(row["route_id"]), str(row["route_label"])} for tid, row in trip_table.iterrows()
    }
    known = set().union(*tokens.values()) if tokens else set()
    for name, values in (
        ("ROUTES_TO_INCLUDE", routes_to_include),
        ("ROUTES_TO_EXCLUDE", routes_to_exclude),
    ):
        unknown = sorted({str(v).strip() for v in values} - known - set(non_revenue_routes))
        if unknown:
            logging.warning("%s entries match no analysis-day route: %s", name, unknown)
    include = {str(v).strip() for v in routes_to_include}
    exclude = {str(v).strip() for v in routes_to_exclude}
    group_tokens = {
        name: {str(v).strip() for v in values} for name, values in (exclusion_groups or {}).items()
    }

    reasons: dict[str, str] = {}
    for tid, row in trip_table.iterrows():
        reason = ""
        rides = boardings.get(str(tid))
        hours = float(row["runtime_min"]) / 60.0
        if not str(row["block_id"]):
            reason = "no block_id"
        elif include and not (tokens[tid] & include):
            reason = "route not in ROUTES_TO_INCLUDE"
        elif tokens[tid] & exclude:
            reason = "route in ROUTES_TO_EXCLUDE"
            matched_groups = [name for name, values in group_tokens.items() if tokens[tid] & values]
            if matched_groups:
                reason += f" ({', '.join(matched_groups)})"
        elif rides is None:
            reason = "no ridership"
        elif hours <= 0:
            reason = "zero running time"
        elif guards.protect_trips_above > 0 and rides / hours > guards.protect_trips_above:
            reason = "above PROTECT_TRIPS_ABOVE"
        reasons[str(tid)] = reason
    return reasons


def flag_low_observations(
    observations: pd.Series, trip_table: pd.DataFrame, share: float
) -> set[str]:
    """Flag trips whose observation count is a low outlier, and warn about them.

    A trip is flagged when its count is below *share* times the median count of
    the matched trips. run_day protects flagged trips from selection when
    PROTECT_LOW_OBSERVATIONS is enabled.

    Args:
        observations: Observation count per matched trip_id.
        trip_table: Output of :func:`build_trip_table` (for route labels and times).
        share: Fraction of the median below which a count is flagged; 0 disables.

    Returns:
        The flagged trip_ids (empty when there are no counts or *share* is 0).
    """
    if observations.empty or share <= 0:
        return set()
    positive = observations[observations.map(lambda v: math.isfinite(v) and v > 0)]
    median = float(positive.median()) if not positive.empty else 0.0
    threshold = median * share
    low = observations[
        observations.map(lambda v: not math.isfinite(v) or v <= 0 or v < threshold)
    ].sort_values()
    if low.empty:
        return set()
    examples = ", ".join(
        f"{trip_table.loc[t, 'route_label']} {minutes_to_hhmm(trip_table.loc[t, 'start_min'])} "
        f"({n:g})"
        for t, n in low.head(5).items()
    )
    logging.warning(
        "Ridership: %d trip(s) have invalid/zero counts or fewer than %.1f observations (%.0f%% of "
        "the median "
        "%.1f), e.g. %s. Screening follows PROTECT_LOW_OBSERVATIONS; see observations and "
        "low_observations in the trips table.",
        len(low),
        threshold,
        share * 100,
        median,
        examples,
    )
    return {str(t) for t in low.index}


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of *path*, read in chunks.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprints(gtfs_path: str, ridership_paths: Mapping[str, str]) -> list[str]:
    """Fingerprint every input file, failing fast if one is missing.

    A zipped feed is hashed whole; a feed folder is hashed per GTFS table this
    script reads.

    Args:
        gtfs_path: GTFS folder or .zip.
        ridership_paths: Day type -> ridership file (empty paths already removed).

    Returns:
        One ``label  size  sha256  path`` line per file, for the run log.

    Raises:
        FileNotFoundError: If an input file does not exist.
    """
    files: list[tuple[str, Path]] = []
    gtfs = Path(gtfs_path)
    if gtfs.is_dir():
        for name in (
            "routes",
            "trips",
            "stop_times",
            "stops",
            "calendar",
            "calendar_dates",
            "frequencies",
        ):
            candidate = gtfs / f"{name}.txt"
            if candidate.exists():
                files.append((f"gtfs/{name}.txt", candidate))
    else:
        files.append(("gtfs", gtfs))
    files.extend((f"ridership/{day}", Path(p)) for day, p in ridership_paths.items())

    lines: list[str] = []
    for label, path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found ({label}): {path}")
        digest = sha256_file(path)
        lines.append(f"{label:<24} {path.stat().st_size:>12,} bytes  sha256={digest}  {path}")
        logging.info("Fingerprinted %s: sha256=%s", label, digest)
    return lines


def settings_lines(cfg: Config, route_filters: Optional[ResolvedRouteFilters] = None) -> list[str]:
    """List every resolved setting of *cfg* (CLI overrides included), one per line."""
    lines: list[str] = []
    for name, value in cfg._asdict().items():
        if hasattr(value, "_asdict"):
            for sub, sub_value in value._asdict().items():
                lines.append(f"{name}.{sub:<28} {sub_value!r}")
        else:
            lines.append(f"{name:<37} {value!r}")
    if route_filters is not None:
        lines.extend(
            [
                f"resolved_routes_to_include            {route_filters.include!r}",
                f"resolved_routes_to_exclude            {route_filters.exclude!r}",
                f"resolved_non_revenue_routes           {route_filters.non_revenue!r}",
                f"resolved_route_exclusion_groups       {dict(route_filters.groups)!r}",
                f"resolved_linked_route_groups          {dict(route_filters.linked_groups)!r}",
            ]
        )
    return lines


def write_run_log(
    output_dir: Path,
    summary_lines: List[str],
    fingerprint_lines: List[str],
    resolved_lines: List[str],
) -> bool:
    """Write the run summary, input fingerprints, resolved settings, and config block.

    Args:
        output_dir: Directory the log is written to.
        summary_lines: Headline results.
        fingerprint_lines: Output of :func:`input_fingerprints`.
        resolved_lines: Output of :func:`settings_lines`.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / RUN_LOG_FILENAME
    source_file = resolve_source_file()
    if source_file is None:
        config_text = (
            "(verbatim block unavailable: interactive session with no __file__ on disk. "
            "RESOLVED SETTINGS above records every analysis setting this run used.)"
        )
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
        "TRIP CUT SAVINGS RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "RUN SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "INPUT FINGERPRINTS",
        "-" * 72,
        *fingerprint_lines,
        "",
        "-" * 72,
        "RESOLVED SETTINGS",
        "-" * 72,
        *resolved_lines,
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


def validate_config(cfg: Config) -> list[str]:
    """Return a message per invalid setting (empty when the configuration is usable)."""
    errors: list[str] = []
    if not isinstance(cfg.export_proposal_workbooks, bool):
        errors.append("EXPORT_PROPOSAL_WORKBOOKS must be True or False.")
    if type(cfg.workbook_limit_per_type) is not int or cfg.workbook_limit_per_type < 0:
        errors.append("WORKBOOK_LIMIT_PER_TYPE must be a nonnegative integer.")
    if isinstance(cfg.workbook_option_ids, str) or any(
        not isinstance(value, str) or not value.strip() for value in cfg.workbook_option_ids
    ):
        errors.append("WORKBOOK_OPTION_IDS must be a list of nonempty quoted option IDs.")
    try:
        configured_headway_sessions(cfg)
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
    if type(cfg.session_headway_tolerance_min) is not int or cfg.session_headway_tolerance_min < 0:
        errors.append("SESSION_HEADWAY_TOLERANCE_MIN must be a nonnegative integer.")
    if type(cfg.shortlist_limit_per_type) is not int or cfg.shortlist_limit_per_type < 0:
        errors.append("SHORTLIST_LIMIT_PER_TYPE must be a nonnegative integer.")
    if not isinstance(cfg.session_preserve_first_last, bool):
        errors.append("SESSION_PRESERVE_FIRST_LAST must be True or False.")
    if cfg.linked_route_mode not in {"auto", "manual", "off"}:
        errors.append("LINKED_ROUTE_MODE must be auto, manual, or off.")
    for name, value in (
        ("LINKED_ROUTE_MIN_SHARED_BLOCKS", cfg.linked_route_min_shared_blocks),
        ("SESSION_MAX_PHASE_COMBINATIONS", cfg.session_max_phase_combinations),
    ):
        if type(value) is not int or value <= 0:
            errors.append(f"{name} must be a positive integer.")
    if (
        not math.isfinite(cfg.session_shortlist_overlap)
        or not 0 <= cfg.session_shortlist_overlap <= 1
    ):
        errors.append("SESSION_SHORTLIST_OVERLAP must be between 0 and 1 (0 disables).")
    excellent = cfg.excellent_max_boardings_per_vehicle_hour_saved
    good = cfg.good_max_boardings_per_vehicle_hour_saved
    if not (math.isfinite(excellent) and math.isfinite(good) and 0 <= excellent < good):
        errors.append("Candidate thresholds must be finite and satisfy 0 <= excellent < good.")
    if MIN_GARAGE_IDLE_MIN < 0:
        errors.append("MIN_GARAGE_IDLE_MIN must be non-negative.")
    if cfg.ridership.match_mode.strip().lower() not in _MATCH_MODES:
        errors.append(f"RIDERSHIP_TRIP_MATCH_MODE must be one of {list(_MATCH_MODES)}.")
    if cfg.rules.min_release_gap_min <= 0:
        errors.append("MIN_RELEASE_GAP_MIN must be positive.")
    if not 0 <= cfg.rules.pull_deadhead_min < cfg.rules.min_release_gap_min:
        errors.append("PULL_DEADHEAD_MIN must be at least 0 and below MIN_RELEASE_GAP_MIN.")
    if cfg.vehicle_profile_bin_min <= 0:
        errors.append("VEHICLE_PROFILE_BIN_MIN must be positive.")
    if cfg.recovery_target_pct < 0:
        errors.append("RECOVERY_TARGET_PCT must not be negative.")
    if not 0 <= cfg.ridership.low_obs_share < 1:
        errors.append("LOW_OBSERVATION_SHARE must be at least 0 and below 1.")
    bad_days = sorted(d for d in cfg.ridership_paths if d not in _DAY_TYPES)
    if bad_days:
        errors.append(f"RIDERSHIP_PATHS keys must be among {list(_DAY_TYPES)}; got {bad_days}.")
    if not any(str(p).strip() for p in cfg.ridership_paths.values()):
        errors.append("RIDERSHIP_PATHS has no file to analyse; give at least one day type a path.")
    for day, value in cfg.service_dates.items():
        if day not in cfg.ridership_paths:
            errors.append(f"SERVICE_DATES key {day!r} has no RIDERSHIP_PATHS entry.")
        elif str(value).strip():
            try:
                dt.datetime.strptime(str(value).strip(), "%Y-%m-%d")
            except ValueError:
                errors.append(f"SERVICE_DATES[{day!r}] must be YYYY-MM-DD; got {value!r}.")
    return errors


def run_day(
    cfg: Config,
    feed: Mapping[str, Optional[pd.DataFrame]],
    periods: Sequence[tuple[str, int, int]],
    day: str,
    ridership_path: str,
    route_filters: Optional[ResolvedRouteFilters] = None,
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, Any]]]:
    """Analyse one day type: match its ridership, rate cut options, and build its tables.

    Args:
        cfg: Resolved configuration.
        feed: Output of :func:`load_feed`.
        periods: Output of :func:`parse_periods`.
        day: Day type (a ``RIDERSHIP_PATHS`` key).
        ridership_path: That day type's ridership file.
        route_filters: Lists already resolved for all days; omitted for a single-day call.

    Returns:
        The day's output tables keyed by base filename, and its summary metrics.
    """
    if route_filters is None:
        route_filters = resolve_route_filters(cfg)
    routes, trips, stop_times = (_required(feed, k) for k in ("routes", "trips", "stop_times"))
    chosen_date, service_ids = select_service_ids(
        feed.get("calendar"),
        feed.get("calendar_dates"),
        day,
        str(cfg.service_dates.get(day, "")),
    )
    all_trips = build_trip_table(trips, stop_times, routes, service_ids)
    excluded_routes = all_trips.route_label.isin(
        route_filters.non_revenue
    ) | all_trips.route_id.isin(route_filters.non_revenue)
    excluded = all_trips[excluded_routes].copy()
    excluded["exclusion_reason"] = "configured non-revenue route"
    trip_table = all_trips[~excluded_routes].copy()
    session_source = trip_table.copy()
    model = DeadheadModel(cfg.tods_path, feed, trip_table, cfg.rules)
    valid, block_qa = validate_analysis_blocks(trip_table, model)
    invalid = trip_table[~trip_table.index.isin(valid.index)].copy()
    invalid["exclusion_reason"] = "invalid block; see block QA"
    excluded = pd.concat([excluded, invalid])
    trip_table = valid
    if trip_table.empty:
        raise ValueError("No valid passenger-service blocks remain after QA exclusions.")
    cfg = cfg._replace(rules=cfg.rules._replace(model=model))
    if len(excluded):
        logging.warning(
            "%d trips excluded from analysis totals; see excluded_trips output.", len(excluded)
        )
    ridership = load_ridership(ridership_path, cfg.ridership, trip_table, routes)
    boardings = {str(k): float(v) for k, v in ridership.boardings.items()}
    observations = {str(k): float(v) for k, v in ridership.observations.items()}
    low_obs = flag_low_observations(ridership.observations, trip_table, cfg.ridership.low_obs_share)

    baseline = build_blocks(trip_table)
    eligibility = eligibility_reasons(
        trip_table,
        boardings,
        cfg.guards,
        route_filters.include,
        route_filters.exclude,
        route_filters.non_revenue,
        route_filters.groups,
    )
    if PROTECT_LOW_OBSERVATIONS:
        for tid in low_obs:
            if not eligibility.get(tid):
                eligibility[tid] = "low observation count"
    eligible = {t for t, reason in eligibility.items() if not reason}

    base_totals = {
        "trips": 0,
        "pieces": 0,
        "revenue_min": 0,
        "layover_min": 0,
        "deadhead_min": 0,
        "vehicle_min": 0,
    }
    for block in baseline.values():
        for key, value in block_metrics(
            block.starts, block.ends, cfg.rules, block.trip_ids
        ).items():
            base_totals[key] += value

    no_ridership = trip_table.index[~trip_table.index.isin(list(boardings))]
    no_ridership_min = int(trip_table.loc[no_ridership, "runtime_min"].sum())
    if len(no_ridership):
        logging.warning(
            "%d of %d analysis-day trip(s) (%.1f revenue hours, %.1f%% of the day's) have no "
            "ridership. They stay in the baseline but are ineligible for screening. See "
            "ineligible_reason 'no ridership'.",
            len(no_ridership),
            len(trip_table),
            no_ridership_min / 60.0,
            100.0 * no_ridership_min / base_totals["revenue_min"]
            if base_totals["revenue_min"]
            else 0.0,
        )

    trips_out = trip_report(
        trip_table, baseline, boardings, eligibility, cfg.rules, observations, low_obs
    )
    review = candidate_review(trips_out, trip_table, excluded, cfg)
    blocks_out = block_report(baseline, trip_table, cfg.rules)
    cycles_out = cycle_time_table(trip_table, baseline, cfg.rules, periods, cfg.recovery_target_pct)
    profile = vehicle_profile(baseline, cfg.rules, cfg.vehicle_profile_bin_min)

    total_boardings = float(sum(boardings.values()))
    solo_layover_only = int(
        (trips_out["eligible"] & trips_out["solo_cut_type"].eq("mid_block_layover_only")).sum()
    )
    summary_pairs: list[tuple[str, Any]] = [
        ("ridership_file", Path(ridership_path).name),
        ("analysis_date", str(chosen_date) if chosen_date else "all trips (no calendar)"),
        ("service_ids", ";".join(sorted(service_ids)) if service_ids else ""),
        ("trips_in_feed_on_day", len(all_trips)),
        ("trips_excluded_from_analysis", len(excluded)),
        ("trips_on_day", len(trip_table)),
        ("baseline_scope", "valid passenger-service blocks only"),
        ("blocks", int(sum(1 for b in baseline if not b.startswith(_NO_BLOCK_PREFIX)))),
        ("ridership_rows", ridership.rows_in),
        ("ridership_rows_matched", ridership.rows_matched),
        ("trips_with_ridership", len(boardings)),
        ("trips_without_ridership", len(no_ridership)),
        ("revenue_hours_without_ridership", round(no_ridership_min / 60.0, 1)),
        ("trips_low_observations", len(low_obs)),
        ("trips_eligible_to_cut", len(eligible)),
        ("eligible_trips_whose_solo_cut_is_layover_only", solo_layover_only),
        ("baseline_boardings", round(total_boardings, 1)),
        ("baseline_revenue_hours", round(base_totals["revenue_min"] / 60.0, 1)),
        ("baseline_layover_hours", round(base_totals["layover_min"] / 60.0, 1)),
        ("baseline_deadhead_hours", round(base_totals["deadhead_min"] / 60.0, 1)),
        ("baseline_vehicle_hours", round(base_totals["vehicle_min"] / 60.0, 1)),
        ("peak_vehicles", int(profile["vehicles"].max()) if not profile.empty else 0),
        ("tods_enabled", model.enabled),
    ]

    audit = model.audit(baseline, "baseline")
    synthetic_used = bool(audit.is_synthetic.any()) if not audit.empty else False
    review["deadhead_inputs_synthetic"] = synthetic_used
    shared_routes, linked_groups = linked_route_groups(trip_table, cfg, route_filters)
    logging.info(
        "Shared routes: %d pairs reported; %d linked groups enabled (%s).",
        len(shared_routes),
        len(linked_groups),
        cfg.linked_route_mode,
    )
    sessions, session_trips = frequency_session_options(
        session_source, trip_table, baseline, boardings, eligibility, cfg, day, linked_groups
    )
    if not sessions.empty:
        sessions["deadhead_inputs_synthetic"] = (
            sessions.deadhead_inputs_synthetic.eq(True) | synthetic_used
        )
    options, shortlist = option_shortlist(review, sessions, cfg, day)
    shortlist_ids = shortlist.option_id.tolist()
    options, option_blocks, option_gaps = annotate_option_effects(
        options, session_source, trip_table, baseline, cfg, day
    )
    options, grouped_shortlist, alternatives = grouped_option_review(options, cfg, day)
    option_index = options.set_index("option_id", drop=False)
    shortlist = planner_column_order(option_index.loc[shortlist_ids].reset_index(drop=True))
    # Append annotations to the original session table without changing its order or scores.
    for column in [*_EFFECT_COLUMNS, *_FAMILY_COLUMNS]:
        sessions[column] = sessions.option_id.map(option_index[column])
    logging.info(
        "Grouped review: %d representatives and %d distinct alternatives; ratings unchanged.",
        len(grouped_shortlist),
        len(alternatives),
    )
    logging.info(
        "Independent options: %d single-trip rows, %d session rows; %d shortlisted.",
        len(options) - len(sessions),
        len(sessions),
        len(shortlist),
    )
    summary_pairs.extend(
        [
            ("frequency_session_options", len(sessions)),
            (
                "frequency_session_options_eligible",
                int(sessions.candidate_classification.ne("ineligible").sum()),
            ),
            ("shortlisted_single_trip_options", int(shortlist.option_type.eq("single_trip").sum())),
            (
                "shortlisted_frequency_session_options",
                int(shortlist.option_type.eq("frequency_session").sum()),
            ),
            ("shared_route_pairs", len(shared_routes)),
            ("linked_route_groups_enabled", len(linked_groups)),
            ("session_alternatives_suppressed", int(options.shortlist_suppressed_by.ne("").sum())),
            ("grouped_shortlist_representatives", len(grouped_shortlist)),
            ("grouped_shortlist_distinct_alternatives", len(alternatives)),
            ("option_savings_are_additive", False),
        ]
    )
    for rating in ("excellent", "good", "poor", "ineligible"):
        summary_pairs.append(
            (f"review_trips_{rating}", int(review.candidate_classification.eq(rating).sum()))
        )
    fallback_rows = (
        int(audit.source.str.contains("fallback|unverified|missing").sum())
        if not audit.empty
        else 0
    )
    summary_pairs.extend(
        [
            ("synthetic_deadhead_used", synthetic_used),
            ("deadhead_audit_fallback_or_unverified_rows", fallback_rows),
        ]
    )
    if fallback_rows:
        logging.warning("Deadhead audit contains %d fallback/unverified rows.", fallback_rows)
    tables: dict[str, pd.DataFrame] = {
        WORKBOOK_TRIPS_FILENAME: build_workbook_trip_context(
            session_source, trips_out, feed, service_ids, route_filters
        ),
        "trip_cut_deadhead_audit.csv": audit,
        "trip_cut_block_qa.csv": block_qa,
        "trip_cut_excluded_trips.csv": excluded.reset_index(),
        TRIPS_FILENAME: trips_out,
        REVIEW_FILENAME: review,
        OPTIONS_FILENAME: options,
        SHORTLIST_FILENAME: shortlist,
        GROUPED_SHORTLIST_FILENAME: grouped_shortlist,
        ALTERNATIVES_FILENAME: alternatives,
        OPTION_BLOCKS_FILENAME: option_blocks,
        OPTION_GAPS_FILENAME: option_gaps,
        SESSIONS_FILENAME: sessions,
        SESSION_TRIPS_FILENAME: session_trips,
        SHARED_ROUTES_FILENAME: shared_routes,
        LINKED_GROUPS_FILENAME: pd.DataFrame(
            [
                {"group_name": name, "route_ids": json.dumps(ids)}
                for name, ids in linked_groups.items()
            ],
            columns=pd.Index(["group_name", "route_ids"]),
        ),
        BLOCKS_FILENAME: blocks_out,
        CYCLE_TIMES_FILENAME: cycles_out,
        VEHICLE_PROFILE_FILENAME: profile,
        RIDERSHIP_MATCH_FILENAME: ridership.match_table,
    }
    return tables, summary_pairs


def _day_filename(name: str, day: str) -> str:
    """Insert the day type before the extension: ``a.csv`` -> ``a_weekday.csv``."""
    path = Path(name)
    return f"{path.stem}_{day}{path.suffix}"


def run(cfg: Config) -> dict[str, dict[str, pd.DataFrame]]:
    """Analyse every configured day type and write every output.

    Inputs are fingerprinted first, so a missing file stops the run before any
    work. Each day type's tables are written with the day as a filename suffix;
    the summary holds one column per day type.

    Args:
        cfg: Resolved configuration.

    Returns:
        Each day type's output tables keyed by base filename.
    """
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))
    route_filters = resolve_route_filters(cfg)
    periods = parse_periods(cfg.time_periods)
    days = {day: str(p).strip() for day, p in cfg.ridership_paths.items() if str(p).strip()}
    skipped = [day for day in cfg.ridership_paths if day not in days]
    if skipped:
        logging.info("No ridership file for %s; skipping.", ", ".join(skipped))
    fingerprints = input_fingerprints(cfg.gtfs_path, days)
    fingerprints.extend(route_filters.fingerprints)
    if cfg.tods_path:
        manifest_path = Path(cfg.tods_path) / "mock_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not Path(cfg.gtfs_path).is_file() or manifest.get("source_sha256") != sha256_file(
                Path(cfg.gtfs_path)
            ):
                raise ValueError(
                    "Mock TODS was generated for a different GTFS ZIP. Regenerate the fixtures."
                )
            logging.warning("Mock TODS manifest found: use this run only for software testing.")
        for path in sorted(Path(cfg.tods_path).glob("*.txt")):
            fingerprints.append(f"tods/{path.name}: sha256={sha256_file(path)}")
    feed = load_feed(cfg.gtfs_path)

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, pd.DataFrame]] = {}
    summaries: dict[str, list[tuple[str, Any]]] = {}
    for day, path in days.items():
        logging.info("%s %s %s", "=" * 30, day.upper(), "=" * 30)
        tables, pairs = run_day(cfg, feed, periods, day, path, route_filters)
        for name, table in tables.items():
            target = out / _day_filename(name, day)
            table.to_csv(target, index=False)
            logging.info("Wrote %s (%d rows).", target, len(table))
        config_path = out / f"trip_cut_linked_route_config_{day}.txt"
        config_path.write_text(
            shared_route_config_text(tables[SHARED_ROUTES_FILENAME], day), encoding="utf-8"
        )
        logging.info("Wrote paste-ready shared-route configuration: %s", config_path)
        metadata = {
            "handoff_version": 1,
            "day": day,
            "analysis_date": dict(pairs)["analysis_date"],
            "service_ids": dict(pairs)["service_ids"],
            "gtfs_path": cfg.gtfs_path,
            "ridership_file": path,
            "data_label": cfg.workbook_data_label,
            "deadhead_basis": (
                f"TODS supplements: {cfg.tods_path}; see deadhead audit for fallbacks."
                if cfg.tods_path
                else f"Flat assumptions: {cfg.rules.pull_deadhead_min} min combined pull-in/out; "
                f"{cfg.rules.min_release_gap_min} min minimum release gap. No verified movement "
                "times."
            ),
        }
        (out / f"trip_cut_workbook_context_{day}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if cfg.export_proposal_workbooks:
            index = export_proposal_workbooks(tables, metadata, cfg, day)
            tables[WORKBOOK_INDEX_FILENAME] = index
            pairs.append(("proposal_workbooks_written", len(index)))
        results[day] = tables
        summaries[day] = pairs

    metrics = [metric for metric, _ in next(iter(summaries.values()))]
    summary = pd.DataFrame({"metric": metrics})
    for day, pairs in summaries.items():
        summary[day] = [value for _, value in pairs]
    summary.to_csv(out / SUMMARY_FILENAME, index=False)
    logging.info("Wrote %s (%d rows).", out / SUMMARY_FILENAME, len(summary))
    if cfg.export_proposal_workbooks and cfg.workbook_option_ids:
        available = {
            key for tables in results.values() for key in tables[OPTIONS_FILENAME].option_id
        }
        missing = set(cfg.workbook_option_ids) - available
        if missing:
            raise ValueError(
                "Requested workbook option IDs were not found: " + ", ".join(sorted(missing))
            )

    summary_lines: list[str] = []
    for day, pairs in summaries.items():
        summary_lines.append(f"[{day}]")
        summary_lines.extend(f"  {metric:<46} {value}" for metric, value in pairs)
    for line in summary_lines:
        logging.info(line)
    if (
        not write_run_log(out, summary_lines, fingerprints, settings_lines(cfg, route_filters))
        and REQUIRE_RUN_LOG
    ):
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return results


def _required(feed: Mapping[str, Optional[pd.DataFrame]], key: str) -> pd.DataFrame:
    """Return a required GTFS table, raising if the loader left it empty."""
    table = feed.get(key)
    if table is None:
        raise ValueError(f"GTFS feed has no {key}.txt.")
    return table


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser; every default is a CONFIGURATION constant."""
    p = argparse.ArgumentParser(
        description=(
            "Shortlist independent trip cuts and frequency-reduction sessions using ridership "
            "and modeled vehicle-time savings, preserving existing blocks and departure times."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gtfs", default=GTFS_PATH, help="GTFS folder or .zip.")
    p.add_argument("--tods", default=TODS_PATH, help="Extracted optional TODS profile directory.")
    p.add_argument(
        "--ridership",
        action="append",
        metavar="DAY=PATH",
        help="Ridership file for a day type; repeat per day type. Replaces RIDERSHIP_PATHS.",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--proposal-workbooks",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_PROPOSAL_WORKBOOKS,
        help="Export one XLSX per selected independent proposal.",
    )
    p.add_argument(
        "--workbook-option",
        action="append",
        metavar="OPTION_ID",
        help="Export this option; repeat for multiple IDs. Overrides automatic workbook selection.",
    )
    p.add_argument(
        "--workbook-limit",
        type=int,
        default=WORKBOOK_LIMIT_PER_TYPE,
        help="Maximum automatic workbooks per option type; 0 exports all grouped representatives.",
    )
    p.add_argument(
        "--workbook-data-label",
        default=WORKBOOK_DATA_LABEL,
        help="Optional test-data label printed on every worksheet.",
    )
    p.add_argument(
        "--workbooks-from",
        metavar="RESULTS_FOLDER",
        help="Export workbooks from a saved results folder without rerunning analysis.",
    )
    p.add_argument(
        "--excellent-max",
        type=float,
        default=EXCELLENT_MAX_BOARDINGS_PER_VEHICLE_HOUR_SAVED,
        help="Excellent candidate maximum boardings per modeled vehicle-hour saved.",
    )
    p.add_argument(
        "--good-max",
        type=float,
        default=GOOD_MAX_BOARDINGS_PER_VEHICLE_HOUR_SAVED,
        help="Good candidate maximum boardings per modeled vehicle-hour saved.",
    )
    p.add_argument(
        "--service-date",
        action="append",
        metavar="DAY=YYYY-MM-DD",
        help="Explicit date for a day type; repeat per day type. Replaces SERVICE_DATES.",
    )
    p.add_argument(
        "--match-mode",
        default=RIDERSHIP_TRIP_MATCH_MODE,
        choices=_MATCH_MODES,
        help="How ridership rows map to GTFS trips.",
    )
    p.add_argument("--trip-id-col", default=RIDERSHIP_TRIP_ID_COL, help="Ridership trip_id column.")
    p.add_argument(
        "--boardings-col", default=RIDERSHIP_BOARDINGS_COL, help="Ridership boardings column."
    )
    p.add_argument("--route-col", default=RIDERSHIP_ROUTE_COL, help="Ridership route column.")
    p.add_argument(
        "--start-time-col", default=RIDERSHIP_START_TIME_COL, help="Ridership start-time column."
    )
    p.add_argument(
        "--date-col", default=RIDERSHIP_DATE_COL, help="Survey-date column to average across."
    )
    p.add_argument(
        "--direction-col",
        default=RIDERSHIP_DIRECTION_COL,
        help="Ridership direction-name column ('' = ignore direction).",
    )
    p.add_argument(
        "--block-col", default=RIDERSHIP_BLOCK_COL, help="Ridership block column ('' = ignore)."
    )
    p.add_argument(
        "--obs-col",
        default=RIDERSHIP_OBS_COL,
        help="Observation-count column for the low-count warning ('' = no check).",
    )
    p.add_argument(
        "--min-release-gap",
        type=int,
        default=MIN_RELEASE_GAP_MIN,
        help="Minutes of gap that release a vehicle mid-block.",
    )
    p.add_argument(
        "--pull-deadhead",
        type=int,
        default=PULL_DEADHEAD_MIN,
        help="Deadhead minutes charged per pull-out + pull-in.",
    )
    p.add_argument(
        "--protect-above",
        type=float,
        default=PROTECT_TRIPS_ABOVE,
        help="Never cut trips above this many boardings per revenue hour (0 = off).",
    )
    p.add_argument(
        "--max-headway",
        type=int,
        default=MAX_RESULTING_HEADWAY_MIN,
        help="Reject cuts opening a longer departure gap, minutes (0 = off).",
    )
    p.add_argument(
        "--protect-span",
        action=argparse.BooleanOptionalAction,
        default=PROTECT_ROUTE_SPAN,
        help="Reject cuts of a route-direction's first or last trip.",
    )
    p.add_argument(
        "--recovery-target",
        type=float,
        default=RECOVERY_TARGET_PCT,
        help="Recovery percent of running time the cycle-time table measures against.",
    )
    return p


def _parse_day_pairs(values: Optional[Sequence[str]], flag: str) -> Optional[dict[str, str]]:
    """Parse repeated ``DAY=VALUE`` flags into a dict; ``None`` when the flag was not given.

    Raises:
        ValueError: If an item lacks ``=`` or a day name.
    """
    if not values:
        return None
    pairs: dict[str, str] = {}
    for item in values:
        day, sep, value = item.partition("=")
        if not sep or not day.strip():
            raise ValueError(f"{flag} expects DAY=VALUE; got {item!r}.")
        pairs[day.strip().lower()] = value.strip()
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths and settings before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 on a configuration error.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))

    try:
        ridership_paths = _parse_day_pairs(args.ridership, "--ridership")
        service_dates = _parse_day_pairs(args.service_date, "--service-date")
    except ValueError as exc:
        logging.error("%s", exc)
        return 2
    if ridership_paths is None:
        ridership_paths = {str(k).strip().lower(): str(v) for k, v in RIDERSHIP_PATHS.items()}
    if service_dates is None:
        service_dates = {str(k).strip().lower(): str(v) for k, v in SERVICE_DATES.items()}

    placeholders = [
        name
        for name, value in (
            ("GTFS_PATH (--gtfs)", args.gtfs),
            *((f"RIDERSHIP_PATHS[{d!r}] (--ridership)", p) for d, p in ridership_paths.items()),
            ("OUTPUT_DIR (--output-dir)", args.output_dir),
        )
        if str(value).startswith(_PLACEHOLDER_PREFIX)
    ]
    if placeholders and not args.workbooks_from:
        logging.error(
            "Still a placeholder: %s. Update the CONFIGURATION section or pass the flags.",
            ", ".join(placeholders),
        )
        return 2

    cfg = Config(
        gtfs_path=str(Path(args.gtfs).expanduser()),
        tods_path=str(Path(args.tods).expanduser()) if args.tods.strip() else "",
        ridership_paths={
            d: (str(Path(p).expanduser()) if str(p).strip() else "")
            for d, p in ridership_paths.items()
        },
        output_dir=Path(args.output_dir).expanduser(),
        service_dates=service_dates,
        ridership=RidershipSpec(
            match_mode=args.match_mode,
            trip_col=args.trip_id_col,
            boardings_col=args.boardings_col,
            route_col=args.route_col,
            start_time_col=args.start_time_col,
            date_col=args.date_col,
            direction_col=args.direction_col,
            block_col=args.block_col,
            obs_col=args.obs_col,
        ),
        rules=BlockRules(args.min_release_gap, args.pull_deadhead),
        guards=Guards(args.protect_above, args.max_headway, args.protect_span),
        recovery_target_pct=args.recovery_target,
        excellent_max_boardings_per_vehicle_hour_saved=args.excellent_max,
        good_max_boardings_per_vehicle_hour_saved=args.good_max,
        export_proposal_workbooks=args.proposal_workbooks,
        workbook_option_ids=args.workbook_option
        if args.workbook_option is not None
        else WORKBOOK_OPTION_IDS,
        workbook_limit_per_type=args.workbook_limit,
        workbook_data_label=args.workbook_data_label,
    )
    errors = validate_config(cfg)
    if errors:
        for message in errors:
            logging.error("Configuration: %s", message)
        return 2

    try:
        if args.workbooks_from:
            export_saved_workbooks(Path(args.workbooks_from).expanduser(), cfg)
        else:
            run(cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Trip cut savings analysis completed successfully.")
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        # A notebook cell: SystemExit would print a spurious traceback, so just run.
        main()
    else:
        raise SystemExit(main())
