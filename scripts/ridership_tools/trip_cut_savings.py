"""Find trip cuts that return real vehicle-hour savings, given trip ridership and GTFS blocks.

Cutting a low-ridership trip saves money only when the vehicle time it used goes away:
the cut trims the start or end of a block, removes a whole block, or opens a mid-block
gap long enough to pull the vehicle in (``MIN_RELEASE_GAP_MIN``). A trip cut from the
middle of a block with a shorter gap only turns its running time into layover — revenue
hours fall, vehicle hours do not. This script scores every contiguous run of trips in
every block by the vehicle time its removal frees, then takes runs in order of fewest
boardings lost per hour saved until the cuts reach ``TARGET_CUT_PERCENT`` of the
analysis day's hours. A per-route cycle-time table, before and after the cuts, shows
where recovery time exceeds ``RECOVERY_TARGET_PCT`` — savings that trimming running
time, not trips, would reach.

Neighbors: ``ridership_by_hour.py`` writes the default input
(``ridership_by_trip_<day>.csv``) and, through ``LOW_RIDERSHIP_THRESHOLD``, flags trips
against a systemwide minimum-productivity standard; this script instead cuts to an
hours target. Pass the selected trip_ids to ``service_cut_impact_gpd.py`` as a
``trip_ids`` scenario for stop-level coverage and rider impacts.

Inputs
------
- A GTFS feed (folder or .zip) whose trips.txt carries ``block_id``: routes, trips,
  stop_times, plus calendar / calendar_dates when present.
- A trip-level ridership file (.csv or .xlsx/.xls) of average daily boardings for the
  same day type, matched by ``trip_id`` or by route + start time
  (``RIDERSHIP_TRIP_MATCH_MODE``). Stop-level rows are summed to their trip.

Outputs
-------
- ``trip_cut_groups.csv``: the selected cuts in order, with hours saved, boardings
  lost, and running totals against the target.
- ``trip_cut_trips.csv``: every analysis-day trip with its ridership, block position,
  what cutting it alone would save, and whether it was selected.
- ``trip_cut_candidates.csv``: every baseline cut that frees vehicle time, ranked.
- ``trip_cut_blocks.csv`` / ``trip_cut_cycle_times.csv`` /
  ``trip_cut_vehicle_profile.csv``: per block, per route × time period (running time,
  recovery, cycle, headway, vehicles needed), and by time of day, before and after.
- ``trip_cut_summary.csv``: headline totals.
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Limitations
-----------
GTFS has no deadhead, so each pull-out + pull-in is charged a flat
``PULL_DEADHEAD_MIN``. Vehicle hours are not operator pay hours: runcutting rules,
guarantees, and split or spread premiums are not modeled. ``frequencies.txt`` service is
not expanded. Selection is greedy, and the final cut may overshoot the target.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags) and run
from a shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import heapq
import logging
import math
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS feed: the unzipped folder or the .zip as published. trips.txt must carry
# block_id — blocks decide whether a cut saves vehicle time.
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"

# Trip-level ridership (average daily boardings) for the day type set below, as
# .csv or .xlsx/.xls. The defaults read ridership_by_hour.py's
# ridership_by_trip_<day>.csv; for a vendor export keyed by route and start time,
# set RIDERSHIP_TRIP_MATCH_MODE = "route_start_time" and the column names below.
RIDERSHIP_PATH: str = r"Path\To\Your\ridership_by_trip_weekday.csv"

OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Day the analysis represents: "weekday" or "monday".."sunday". A representative
# date of that day type is chosen from the feed's calendar (skipping observed
# federal holidays); SERVICE_DATE ("YYYY-MM-DD") pins one explicit date instead.
# Match it to the day type the ridership file describes.
SERVICE_DAY: str = "weekday"
SERVICE_DATE: str = ""

# --- Ridership matching ----------------------------------------------------------
# "trip_id": RIDERSHIP_TRIP_ID_COL holds GTFS trip_ids.
# "route_start_time": each row matches the GTFS trip on the same route
#   (route_short_name or route_id) whose first departure is nearest the row's
#   start time, within RIDERSHIP_START_TIME_TOLERANCE_MIN.
RIDERSHIP_TRIP_MATCH_MODE: str = "trip_id"
RIDERSHIP_TRIP_ID_COL: str = "trip_id"
RIDERSHIP_BOARDINGS_COL: str = "avg_daily_boardings"
RIDERSHIP_ROUTE_COL: str = "ROUTE_NUMBER"
RIDERSHIP_START_TIME_COL: str = "TRIP_START_TIME"
RIDERSHIP_START_TIME_TOLERANCE_MIN: int = 2
# Optional stop column. Rows with a stop value are summed to their trip; a trip
# carrying both per-stop rows and a whole-trip row keeps only the per-stop rows.
RIDERSHIP_STOP_ID_COL: str = "stop_id"
# For a file with one row per trip per surveyed day, name its date column so
# boardings are averaged across days instead of summed. "" = already averaged.
RIDERSHIP_DATE_COL: str = ""

# --- Cut target ------------------------------------------------------------------
# Share of the analysis day's hours to cut, in percent. Cuts are taken in order of
# fewest boardings lost per hour saved until the running total reaches the target.
# 0 = select nothing; every table except the groups is still written.
TARGET_CUT_PERCENT: float = 5.0
# Which hours the target counts:
#   "vehicle_hours" - vehicle time visible in GTFS: running time, layovers shorter
#                     than MIN_RELEASE_GAP_MIN, and PULL_DEADHEAD_MIN per pull-out.
#   "revenue_hours" - trip running time.
# Either way, only cuts that free vehicle time are eligible.
TARGET_HOURS_BASIS: str = "vehicle_hours"

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
# route still counts toward the baseline hours the target is a share of.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

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

# Also write every baseline cut that frees vehicle time, ranked.
EXPORT_CANDIDATES: bool = True

GROUPS_FILENAME: str = "trip_cut_groups.csv"
TRIPS_FILENAME: str = "trip_cut_trips.csv"
CANDIDATES_FILENAME: str = "trip_cut_candidates.csv"
BLOCKS_FILENAME: str = "trip_cut_blocks.csv"
CYCLE_TIMES_FILENAME: str = "trip_cut_cycle_times.csv"
VEHICLE_PROFILE_FILENAME: str = "trip_cut_vehicle_profile.csv"
SUMMARY_FILENAME: str = "trip_cut_summary.csv"
RUN_LOG_FILENAME: str = "trip_cut_savings_runlog.txt"

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

_PLACEHOLDER_PREFIX: str = "Path\\To\\"
_EXCEL_SUFFIXES: tuple[str, ...] = (".xls", ".xlsx", ".xlsm", ".xltx", ".xltm")
_TARGET_BASES: tuple[str, ...] = ("vehicle_hours", "revenue_hours")
_MATCH_MODES: tuple[str, ...] = ("trip_id", "route_start_time")
# Below this share of ridership trip_ids matching an analysis-day trip, warn that
# the file may describe another day type or use other ids.
_RIDERSHIP_MATCH_WARN_SHARE: float = 0.5
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


class BlockRules(NamedTuple):
    """How gaps between a block's trips turn into vehicle time."""

    min_release_gap_min: int = MIN_RELEASE_GAP_MIN
    pull_deadhead_min: int = PULL_DEADHEAD_MIN


class Guards(NamedTuple):
    """Optional limits on which cuts may be selected."""

    protect_trips_above: float = PROTECT_TRIPS_ABOVE
    max_resulting_headway_min: int = MAX_RESULTING_HEADWAY_MIN
    protect_route_span: bool = PROTECT_ROUTE_SPAN


class Config(NamedTuple):
    """Resolved settings for one run."""

    gtfs_path: str
    ridership_path: str
    output_dir: Path
    service_day: str = SERVICE_DAY
    service_date: str = SERVICE_DATE
    ridership: RidershipSpec = RidershipSpec()
    target_cut_percent: float = TARGET_CUT_PERCENT
    target_hours_basis: str = TARGET_HOURS_BASIS
    rules: BlockRules = BlockRules()
    guards: Guards = Guards()
    routes_to_include: Sequence[str] = tuple(ROUTES_TO_INCLUDE)
    routes_to_exclude: Sequence[str] = tuple(ROUTES_TO_EXCLUDE)
    time_periods: Mapping[str, tuple[str, str]] = TIME_PERIODS
    recovery_target_pct: float = RECOVERY_TARGET_PCT
    vehicle_profile_bin_min: int = VEHICLE_PROFILE_BIN_MIN
    export_candidates: bool = EXPORT_CANDIDATES


class RunCut(NamedTuple):
    """What removing trips ``i``..``j`` (inclusive) of a block would do."""

    i: int
    j: int
    cut_type: str
    revenue_min: int
    vehicle_min_saved: int
    gap_left_min: Optional[int]


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
    """Per-trip attributes the selection loop reads many times, as plain dicts."""

    route_dir: dict[str, tuple[str, str]]
    label: dict[str, str]


class SelectionResult(NamedTuple):
    """The cuts chosen to meet the target, and why other candidates were passed over."""

    groups: pd.DataFrame
    cut_trip_ids: set[str]
    group_of_trip: dict[str, int]
    rejected: dict[str, int]


# =============================================================================
# GTFS LOADING
# =============================================================================


def load_feed(gtfs_path: str) -> dict[str, Optional[pd.DataFrame]]:
    """Load the GTFS tables this analysis uses.

    Args:
        gtfs_path: GTFS folder or .zip path.

    Returns:
        Mapping with ``routes``, ``trips``, ``stop_times`` (always present) and
        ``calendar``, ``calendar_dates``, ``frequencies`` (``None`` when absent).
    """
    required = load_gtfs_data(gtfs_path, files=("routes.txt", "trips.txt", "stop_times.txt"))
    feed: dict[str, Optional[pd.DataFrame]] = dict(required)
    for optional in ("calendar", "calendar_dates", "frequencies"):
        try:
            feed[optional] = load_gtfs_data(gtfs_path, files=(f"{optional}.txt",))[optional]
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
            raise ValueError(f"SERVICE_DATE must be YYYY-MM-DD; got {service_date!r}.") from exc

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
                "No trips remain on the analysis day's service_ids. Check SERVICE_DAY / "
                "SERVICE_DATE against the feed's calendar."
            )

    st = stop_times[["trip_id", "arrival_time", "departure_time"]].copy()
    st["trip_id"] = st["trip_id"].astype(str).str.strip()
    st = st[st["trip_id"].isin(set(t["trip_id"]))]
    dep = st["departure_time"].map(parse_time_to_minutes)
    arr = st["arrival_time"].map(parse_time_to_minutes)
    st["dep_min"] = dep.where(dep.notna(), arr)
    st["arr_min"] = arr.where(arr.notna(), dep)
    spans = st.groupby("trip_id").agg(start_min=("dep_min", "min"), end_min=("arr_min", "max"))
    spans = spans.dropna()

    table = t.set_index("trip_id")[["route_id", "direction_id", "block_id"]].join(
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


def match_trips_by_route_and_time(
    frame: pd.DataFrame,
    trip_table: pd.DataFrame,
    routes: pd.DataFrame,
    spec: RidershipSpec,
) -> pd.Series:
    """Match ridership rows to analysis-day trips on route plus first departure.

    Each row goes to the trip of the same route whose first departure is nearest
    the row's start time, within ``spec.tolerance_min``. A small-hours time is also
    tried a service day later (1:30 AM matches a GTFS 25:30 departure), and ties go
    to the lowest ``trip_id`` so reruns are stable.

    Args:
        frame: Raw ridership rows.
        trip_table: Output of :func:`build_trip_table`.
        routes: Parsed ``routes.txt`` (route_short_name or route_id tokens).
        spec: Column names and tolerance.

    Returns:
        A ``trip_id`` per row aligned to ``frame.index``; ``""`` where nothing matched.
    """
    r = routes.drop_duplicates(subset=["route_id"]).copy()
    r["route_id"] = r["route_id"].astype(str).str.strip()
    short = r.get("route_short_name", pd.Series("", index=r.index)).fillna("").astype(str)
    token_to_id = dict(zip(r["route_id"], r["route_id"]))
    token_to_id.update({s.strip(): rid for s, rid in zip(short, r["route_id"]) if s.strip()})

    by_route: dict[str, list[tuple[int, str]]] = {}
    for trip_id, row in trip_table.iterrows():
        by_route.setdefault(str(row["route_id"]), []).append((int(row["start_min"]), str(trip_id)))

    matched: list[str] = []
    unknown_routes: set[str] = set()
    unparsed = 0
    no_trip = 0
    for token, raw_time in zip(frame[spec.route_col], frame[spec.start_time_col]):
        route_id = token_to_id.get(str(token).strip())
        if route_id is None:
            unknown_routes.add(str(token).strip())
            matched.append("")
            continue
        minutes = parse_clock_value(raw_time)
        if minutes is None:
            unparsed += 1
            matched.append("")
            continue
        best_trip = ""
        best_delta: Optional[int] = None
        for departure, trip_id in by_route.get(route_id, []):
            for candidate in (minutes, minutes + 1440):
                delta = abs(departure - candidate)
                if delta > spec.tolerance_min:
                    continue
                if (
                    best_delta is None
                    or delta < best_delta
                    or (delta == best_delta and trip_id < best_trip)
                ):
                    best_trip, best_delta = trip_id, delta
        if not best_trip:
            no_trip += 1
        matched.append(best_trip)

    if unknown_routes:
        logging.warning(
            "Ridership: %d route value(s) match nothing in routes.txt (e.g. %s); their rows "
            "are ignored.",
            len(unknown_routes),
            ", ".join(sorted(unknown_routes)[:5]),
        )
    if unparsed:
        logging.warning(
            "Ridership: %d row(s) have an unreadable %r value and are ignored. An Excel "
            "time-only cell that reads as a date (1/0/1900) has lost its time of day; "
            "re-export that column as text (HH:MM).",
            unparsed,
            spec.start_time_col,
        )
    if no_trip:
        logging.info(
            "Ridership: %d row(s) have no trip on their route starting within %d minute(s) "
            "of the listed time — expected for rows describing other service days.",
            no_trip,
            spec.tolerance_min,
        )
    return pd.Series(matched, index=frame.index, dtype=object)


def load_ridership(
    path: str,
    spec: RidershipSpec,
    trip_table: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.Series:
    """Load average daily boardings per analysis-day trip.

    Args:
        path: Ridership file (.csv or .xlsx/.xls).
        spec: Column names and the trip-matching strategy.
        trip_table: Output of :func:`build_trip_table`.
        routes: Parsed ``routes.txt``.

    Returns:
        Boardings indexed by ``trip_id``, for trips on the analysis day only.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a required column is missing, the match mode is unknown, or
            no row matches an analysis-day trip.
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

    rows_in = len(frame)
    if mode == "trip_id":
        frame["_trip"] = frame[spec.trip_col].astype(str).str.strip()
    else:
        frame["_trip"] = match_trips_by_route_and_time(frame, trip_table, routes, spec)
    frame["_boardings"] = pd.to_numeric(frame[spec.boardings_col], errors="coerce")
    bad = int(frame["_boardings"].isna().sum())
    if bad:
        logging.warning("Ridership: dropped %d row(s) with non-numeric boardings.", bad)
    frame = frame[frame["_boardings"].notna() & frame["_trip"].ne("")]

    on_day = frame["_trip"].isin(set(trip_table.index))
    file_trips = frame["_trip"].nunique()
    matched_trips = frame.loc[on_day, "_trip"].nunique()
    if (
        mode == "trip_id"
        and file_trips
        and matched_trips / file_trips < _RIDERSHIP_MATCH_WARN_SHARE
    ):
        logging.warning(
            "Ridership: only %d of %d trip_id(s) in the file run on the analysis day. Check "
            "that the file and SERVICE_DAY describe the same day type and that %r holds GTFS "
            "trip_ids (if not, set RIDERSHIP_TRIP_MATCH_MODE = 'route_start_time').",
            matched_trips,
            file_trips,
            spec.trip_col,
        )
    frame = frame[on_day]
    if frame.empty:
        raise ValueError(
            f"None of the ridership file's {rows_in} row(s) matched a trip on the analysis "
            "day, so no trip can be scored. Check the day type, the id column, and the "
            "match mode."
        )

    if spec.stop_col and spec.stop_col in frame.columns:
        has_stop = frame[spec.stop_col].fillna("").astype(str).str.strip().ne("")
        stop_trips = set(frame.loc[has_stop, "_trip"])
        double = ~has_stop & frame["_trip"].isin(stop_trips)
        if double.any():
            logging.warning(
                "Ridership: %d trip(s) have both per-stop rows and a whole-trip row; keeping "
                "the per-stop rows only.",
                frame.loc[double, "_trip"].nunique(),
            )
            frame = frame[~double]

    if date_col:
        frame = frame.assign(_date=frame[date_col].astype(str).str.strip())
        per_day = frame.groupby(["_trip", "_date"])["_boardings"].sum()
        boardings = per_day.groupby(level="_trip").mean()
        logging.info(
            "Ridership: averaged across %d distinct %r value(s).",
            frame["_date"].nunique(),
            date_col,
        )
    else:
        boardings = frame.groupby("_trip")["_boardings"].sum()
    boardings.index.name = "trip_id"
    logging.info(
        "Ridership: %d of %d analysis-day trip(s) have boardings (%.1f%%).",
        len(boardings),
        len(trip_table),
        100.0 * len(boardings) / len(trip_table),
    )
    return boardings.astype(float)


# =============================================================================
# BLOCK ARITHMETIC
# =============================================================================


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


def block_metrics(starts: Sequence[int], ends: Sequence[int], rules: BlockRules) -> dict[str, int]:
    """Summarize one block's trips, layover, pieces, and vehicle minutes.

    A *piece* is a stretch of the block between releases (gaps of at least
    ``min_release_gap_min``). Vehicle minutes are revenue minutes, plus layover
    shorter than the release threshold, plus the deadhead allowance per piece.

    Args:
        starts: Trip first departures in start order.
        ends: Trip last arrivals, parallel to *starts*.
        rules: Release threshold and deadhead allowance.

    Returns:
        Dict with ``trips``, ``pieces``, ``revenue_min``, ``layover_min``, and
        ``vehicle_min``.
    """
    if not starts:
        return {"trips": 0, "pieces": 0, "revenue_min": 0, "layover_min": 0, "vehicle_min": 0}
    gaps = block_gaps(starts, ends)
    revenue = sum(max(0, e - s) for s, e in zip(starts, ends))
    layover = sum(g for g in gaps if g < rules.min_release_gap_min)
    pieces = 1 + sum(1 for g in gaps if g >= rules.min_release_gap_min)
    return {
        "trips": len(starts),
        "pieces": pieces,
        "revenue_min": revenue,
        "layover_min": layover,
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
) -> list[RunCut]:
    """Score every contiguous run of trips in a block by the vehicle time it frees.

    Removing trips ``i``..``j`` deletes their running time and the gaps on either
    side, and — when trips remain on both sides — opens one new gap from the end of
    trip ``i-1`` to the start of trip ``j+1``. The vehicle minutes saved are the
    charges removed minus the new gap's charge, so a mid-block run whose new gap is
    shorter than the release threshold saves exactly nothing. Prefix sums make each
    run O(1).

    Args:
        starts: Trip first departures in start order.
        ends: Trip last arrivals, parallel to *starts*.
        rules: Release threshold and deadhead allowance.
        eligible: Per-trip flags; runs containing an ineligible trip are skipped.
            ``None`` treats every trip as eligible.
        single_trips_only: Score only one-trip runs (each trip cut on its own).

    Returns:
        One :class:`RunCut` per run scored, including runs that save nothing.
    """
    n = len(starts)
    if n == 0:
        return []
    flags = list(eligible) if eligible is not None else [True] * n
    runtimes = [max(0, e - s) for s, e in zip(starts, ends)]
    charges = [gap_charge(g, rules) for g in block_gaps(starts, ends)]
    rt_prefix = [0]
    for value in runtimes:
        rt_prefix.append(rt_prefix[-1] + value)
    ch_prefix = [0]
    for value in charges:
        ch_prefix.append(ch_prefix[-1] + value)
    total = rules.pull_deadhead_min + rt_prefix[n] + ch_prefix[n - 1]

    runs: list[RunCut] = []
    for i in range(n):
        if not flags[i]:
            continue
        last = i if single_trips_only else n - 1
        for j in range(i, last + 1):
            if not flags[j]:
                break
            revenue = rt_prefix[j + 1] - rt_prefix[i]
            gap_left: Optional[int] = None
            if i == 0 and j == n - 1:
                cut_type, saved = "whole_block", total
            elif i == 0:
                cut_type = "block_start_trim"
                saved = revenue + ch_prefix[j + 1] - ch_prefix[0]
            elif j == n - 1:
                cut_type = "block_end_trim"
                saved = revenue + ch_prefix[n - 1] - ch_prefix[i - 1]
            else:
                gap_left = max(0, starts[j + 1] - ends[i - 1])
                saved = revenue + ch_prefix[j + 1] - ch_prefix[i - 1] - gap_charge(gap_left, rules)
                cut_type = (
                    "mid_block_release"
                    if gap_left >= rules.min_release_gap_min
                    else "mid_block_layover_only"
                )
            runs.append(RunCut(i, j, cut_type, revenue, saved, gap_left))
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


def basis_minutes(run: RunCut, basis: str) -> int:
    """Minutes a run counts toward the target under *basis*."""
    return run.vehicle_min_saved if basis == "vehicle_hours" else run.revenue_min


def run_record(
    run: RunCut,
    block: BlockState,
    lookup: TripLookup,
    boardings: Mapping[str, float],
    basis: str,
    shortens_span: bool,
    new_headway: Optional[int],
) -> dict[str, Any]:
    """Describe one cut as an output row."""
    trip_ids = block.trip_ids[run.i : run.j + 1]
    labels = [lookup.label[t] for t in trip_ids]
    lost = float(sum(boardings.get(t, 0.0) for t in trip_ids))
    hours = basis_minutes(run, basis) / 60.0
    return {
        "block_id": block.block_id,
        "cut_type": run.cut_type,
        "routes": ";".join(dict.fromkeys(labels)),
        "trip_count": len(trip_ids),
        "trip_ids": ";".join(trip_ids),
        "first_start": minutes_to_hhmm(block.starts[run.i]),
        "last_end": minutes_to_hhmm(block.ends[run.j]),
        "revenue_hours_cut": round(run.revenue_min / 60.0, 2),
        "vehicle_hours_saved": round(run.vehicle_min_saved / 60.0, 2),
        "boardings_lost": round(lost, 1),
        "boardings_per_hour_saved": round(lost / hours, 2) if hours > 0 else float("nan"),
        "gap_left_min": run.gap_left_min,
        "shortens_span": shortens_span,
        "new_max_headway_min": new_headway,
    }


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


def baseline_candidates(
    blocks: Mapping[str, BlockState],
    trip_table: pd.DataFrame,
    boardings: Mapping[str, float],
    eligible: set[str],
    rules: BlockRules,
    basis: str,
) -> pd.DataFrame:
    """Every baseline cut of eligible trips that frees vehicle time, best first.

    Args:
        blocks: Baseline blocks from :func:`build_blocks`.
        trip_table: Output of :func:`build_trip_table`.
        boardings: Boardings per trip_id.
        eligible: Trip_ids that may be cut.
        rules: Block savings rules.
        basis: ``"vehicle_hours"`` or ``"revenue_hours"``.

    Returns:
        One row per candidate, ranked by boardings lost per hour saved.
    """
    lookup = build_trip_lookup(trip_table)
    route_dirs = build_route_dir_index(trip_table)
    rows: list[dict[str, Any]] = []
    for block in blocks.values():
        flags = [t in eligible for t in block.trip_ids]
        for run in enumerate_runs(block.starts, block.ends, rules, flags):
            if run.vehicle_min_saved <= 0 or basis_minutes(run, basis) <= 0:
                continue
            span, headway = service_effects(block.trip_ids[run.i : run.j + 1], lookup, route_dirs)
            rows.append(run_record(run, block, lookup, boardings, basis, span, headway))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["boardings_per_hour_saved", "vehicle_hours_saved", "block_id", "first_start"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def select_cuts(
    blocks: Mapping[str, BlockState],
    trip_table: pd.DataFrame,
    boardings: Mapping[str, float],
    eligible: set[str],
    rules: BlockRules,
    guards: Guards,
    basis: str,
    target_min: float,
) -> SelectionResult:
    """Pick cuts greedily, fewest boardings lost per hour saved first, up to the target.

    Every eligible run that frees vehicle time sits in a priority queue. The best is
    popped, re-checked against the guards on the current schedule, and applied; its
    block's runs are then re-scored, because trips on either side of a cut can now
    combine into new runs. A queued run whose block has since changed is skipped.
    *blocks* is modified in place to the post-cut schedule.

    Args:
        blocks: Blocks from :func:`build_blocks` (mutated).
        trip_table: Output of :func:`build_trip_table`.
        boardings: Boardings per trip_id.
        eligible: Trip_ids that may be cut.
        rules: Block savings rules.
        guards: Optional limits on selectable cuts.
        basis: ``"vehicle_hours"`` or ``"revenue_hours"``.
        target_min: Minutes to cut on *basis*; 0 selects nothing.

    Returns:
        The selected groups, the cut trip_ids, each cut trip's group number, and a
        count of candidates rejected per guard.
    """
    lookup = build_trip_lookup(trip_table)
    route_dirs = build_route_dir_index(trip_table)
    heap: list[tuple[float, int, int, str, int, RunCut]] = []
    sequence = 0

    def push_block(block: BlockState) -> None:
        nonlocal sequence
        flags = [t in eligible for t in block.trip_ids]
        for run in enumerate_runs(block.starts, block.ends, rules, flags):
            minutes = basis_minutes(run, basis)
            if run.vehicle_min_saved <= 0 or minutes <= 0:
                continue
            lost = sum(boardings.get(t, 0.0) for t in block.trip_ids[run.i : run.j + 1])
            # sequence is unique, so tuple comparison never reaches the RunCut.
            entry = (
                lost / (minutes / 60.0),
                -minutes,
                sequence,
                block.block_id,
                block.version,
                run,
            )
            heapq.heappush(heap, entry)
            sequence += 1

    if target_min > 0:
        for block in blocks.values():
            push_block(block)

    groups: list[dict[str, Any]] = []
    cut_trip_ids: set[str] = set()
    group_of_trip: dict[str, int] = {}
    rejected: dict[str, int] = {"route_span": 0, "max_headway": 0}
    total_min = 0.0
    lost_total = 0.0
    while heap and total_min < target_min:
        _, _, _, block_id, version, run = heapq.heappop(heap)
        block = blocks[block_id]
        if block.version != version:
            continue
        trip_ids = block.trip_ids[run.i : run.j + 1]
        span, headway = service_effects(trip_ids, lookup, route_dirs)
        reason = _guard_reason(span, headway, guards)
        if reason is not None:
            rejected[reason] += 1
            continue
        record = run_record(run, block, lookup, boardings, basis, span, headway)
        total_min += basis_minutes(run, basis)
        lost_total += record["boardings_lost"]
        group_id = len(groups) + 1
        groups.append(
            {
                "group_id": group_id,
                **record,
                "cumulative_hours": round(total_min / 60.0, 2),
                "cumulative_boardings_lost": round(lost_total, 1),
            }
        )

        removed = set(block.remove_run(run.i, run.j))
        cut_trip_ids |= removed
        for tid in removed:
            group_of_trip[tid] = group_id
        for key in {lookup.route_dir[t] for t in removed}:
            route_dirs[key] = [e for e in route_dirs[key] if e[1] not in removed]
        push_block(block)

    return SelectionResult(pd.DataFrame(groups), cut_trip_ids, group_of_trip, rejected)


# =============================================================================
# REPORT TABLES
# =============================================================================


def trip_report(
    trip_table: pd.DataFrame,
    baseline_blocks: Mapping[str, BlockState],
    boardings: Mapping[str, float],
    eligibility: Mapping[str, str],
    selection: SelectionResult,
    rules: BlockRules,
) -> pd.DataFrame:
    """One row per analysis-day trip: ridership, block position, and solo-cut savings.

    Args:
        trip_table: Output of :func:`build_trip_table`.
        baseline_blocks: Blocks before any cut.
        boardings: Boardings per trip_id.
        eligibility: trip_id -> ``""`` when cuttable, else the reason it is not.
        selection: Output of :func:`select_cuts`.
        rules: Block savings rules.

    Returns:
        The trip table, in block and start order.
    """
    solo: dict[str, dict[str, Any]] = {}
    for block in baseline_blocks.values():
        n = len(block.trip_ids)
        for run in enumerate_runs(block.starts, block.ends, rules, single_trips_only=True):
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
                "solo_vehicle_min_saved": run.vehicle_min_saved,
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
    details = pd.DataFrame.from_dict(solo, orient="index")
    out = out.join(details, on="trip_id")
    out["eligible"] = out["trip_id"].map(lambda t: eligibility.get(t, "") == "")
    out["ineligible_reason"] = out["trip_id"].map(lambda t: eligibility.get(t, ""))
    out["selected"] = out["trip_id"].isin(selection.cut_trip_ids)
    out["group_id"] = out["trip_id"].map(selection.group_of_trip).astype("Int64")
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
        "block_position",
        "block_trip_count",
        "solo_cut_type",
        "solo_vehicle_min_saved",
        "solo_gap_left_min",
        "solo_min_short_of_release",
        "eligible",
        "ineligible_reason",
        "selected",
        "group_id",
    ]
    return out.sort_values(["block_key", "start_min"])[columns].reset_index(drop=True)


def block_report(
    baseline: Mapping[str, BlockState],
    after: Mapping[str, BlockState],
    trip_table: pd.DataFrame,
    rules: BlockRules,
) -> pd.DataFrame:
    """Per block, trips and hours before and after the cuts."""
    rows: list[dict[str, Any]] = []
    for block_id, before in baseline.items():
        post = after[block_id]
        m0 = block_metrics(before.starts, before.ends, rules)
        m1 = block_metrics(post.starts, post.ends, rules)
        labels = trip_table.loc[before.trip_ids, "route_label"].astype(str)
        rows.append(
            {
                "block_id": block_id,
                "routes": ";".join(dict.fromkeys(labels)),
                "trips_before": m0["trips"],
                "trips_after": m1["trips"],
                "start_before": minutes_to_hhmm(before.starts[0]) if before.starts else "",
                "end_before": minutes_to_hhmm(max(before.ends)) if before.ends else "",
                "start_after": minutes_to_hhmm(post.starts[0]) if post.starts else "",
                "end_after": minutes_to_hhmm(max(post.ends)) if post.ends else "",
                "pieces_before": m0["pieces"],
                "pieces_after": m1["pieces"],
                "revenue_hours_before": round(m0["revenue_min"] / 60.0, 2),
                "revenue_hours_after": round(m1["revenue_min"] / 60.0, 2),
                "layover_hours_before": round(m0["layover_min"] / 60.0, 2),
                "layover_hours_after": round(m1["layover_min"] / 60.0, 2),
                "vehicle_hours_before": round(m0["vehicle_min"] / 60.0, 2),
                "vehicle_hours_after": round(m1["vehicle_min"] / 60.0, 2),
                "vehicle_hours_saved": round((m0["vehicle_min"] - m1["vehicle_min"]) / 60.0, 2),
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
                if block.starts[m] - block.ends[m - 1] >= rules.min_release_gap_min:
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
    for (route_id, period), sub in frame.groupby(["route_id", "period"], sort=True):
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
    after: Mapping[str, BlockState],
    rules: BlockRules,
    bin_min: int,
) -> pd.DataFrame:
    """Vehicles in service at the start of each time bin, before and after the cuts.

    A vehicle is in service from the first departure to the last arrival of each
    block piece; deadhead allowances are not drawn.
    """

    def pieces(blocks: Mapping[str, BlockState]) -> list[tuple[int, int]]:
        return [p for b in blocks.values() for p in block_pieces(b.starts, b.ends, rules)]

    before_pieces, after_pieces = pieces(baseline), pieces(after)
    if not before_pieces:
        return pd.DataFrame(columns=["time", "vehicles_before", "vehicles_after"])
    first = min(p[0] for p in before_pieces) // bin_min * bin_min
    last = max(p[1] for p in before_pieces)
    rows = []
    for t in range(first, last + 1, bin_min):
        rows.append(
            {
                "time": minutes_to_hhmm(t),
                "vehicles_before": sum(1 for s, e in before_pieces if s <= t < e),
                "vehicles_after": sum(1 for s, e in after_pieces if s <= t < e),
            }
        )
    return pd.DataFrame(rows)


def copy_blocks(blocks: Mapping[str, BlockState]) -> dict[str, BlockState]:
    """Independent copy of a block mapping, so selection can mutate one of them."""
    return {k: BlockState(b.block_id, b.trip_ids, b.starts, b.ends) for k, b in blocks.items()}


def eligibility_reasons(
    trip_table: pd.DataFrame,
    boardings: Mapping[str, float],
    guards: Guards,
    routes_to_include: Sequence[str],
    routes_to_exclude: Sequence[str],
) -> dict[str, str]:
    """Return each trip's reason it cannot be cut, or ``""`` when it can.

    Raises:
        ValueError: If a route filter matches no route on the analysis day.
    """
    tokens = {
        tid: {str(row["route_id"]), str(row["route_label"])} for tid, row in trip_table.iterrows()
    }
    known = set().union(*tokens.values()) if tokens else set()
    for name, values in (
        ("ROUTES_TO_INCLUDE", routes_to_include),
        ("ROUTES_TO_EXCLUDE", routes_to_exclude),
    ):
        unknown = sorted({str(v).strip() for v in values} - known)
        if unknown:
            logging.warning("%s entries match no analysis-day route: %s", name, unknown)
    include = {str(v).strip() for v in routes_to_include}
    exclude = {str(v).strip() for v in routes_to_exclude}

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
        elif rides is None:
            reason = "no ridership"
        elif hours <= 0:
            reason = "zero running time"
        elif guards.protect_trips_above > 0 and rides / hours > guards.protect_trips_above:
            reason = "above PROTECT_TRIPS_ABOVE"
        reasons[str(tid)] = reason
    return reasons


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

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
    if cfg.target_hours_basis not in _TARGET_BASES:
        errors.append(f"TARGET_HOURS_BASIS must be one of {list(_TARGET_BASES)}.")
    if not 0 <= cfg.target_cut_percent <= 100:
        errors.append("TARGET_CUT_PERCENT must be between 0 and 100.")
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
    return errors


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    """Execute the analysis and write every output.

    Args:
        cfg: Resolved configuration.

    Returns:
        The output tables keyed by name (also written to ``cfg.output_dir``).
    """
    periods = parse_periods(cfg.time_periods)
    feed = load_feed(cfg.gtfs_path)
    routes, trips, stop_times = (_required(feed, k) for k in ("routes", "trips", "stop_times"))
    chosen_date, service_ids = select_service_ids(
        feed.get("calendar"), feed.get("calendar_dates"), cfg.service_day, cfg.service_date
    )
    trip_table = build_trip_table(trips, stop_times, routes, service_ids)
    boardings_series = load_ridership(cfg.ridership_path, cfg.ridership, trip_table, routes)
    boardings = {str(k): float(v) for k, v in boardings_series.items()}

    baseline = build_blocks(trip_table)
    eligibility = eligibility_reasons(
        trip_table, boardings, cfg.guards, cfg.routes_to_include, cfg.routes_to_exclude
    )
    eligible = {t for t, reason in eligibility.items() if not reason}
    basis = cfg.target_hours_basis

    base_totals = {"trips": 0, "pieces": 0, "revenue_min": 0, "layover_min": 0, "vehicle_min": 0}
    for block in baseline.values():
        for key, value in block_metrics(block.starts, block.ends, cfg.rules).items():
            base_totals[key] += value
    basis_total = (
        base_totals["vehicle_min"] if basis == "vehicle_hours" else base_totals["revenue_min"]
    )
    target_min = basis_total * cfg.target_cut_percent / 100.0

    candidates = baseline_candidates(baseline, trip_table, boardings, eligible, cfg.rules, basis)
    after = copy_blocks(baseline)
    selection = select_cuts(
        after, trip_table, boardings, eligible, cfg.rules, cfg.guards, basis, target_min
    )
    groups = selection.groups
    if not groups.empty and basis_total > 0:
        groups.insert(
            groups.columns.get_loc("cumulative_hours") + 1,
            "cumulative_pct_of_baseline",
            (100.0 * groups["cumulative_hours"] * 60.0 / basis_total).round(2),
        )

    after_totals = {"trips": 0, "pieces": 0, "revenue_min": 0, "layover_min": 0, "vehicle_min": 0}
    for block in after.values():
        for key, value in block_metrics(block.starts, block.ends, cfg.rules).items():
            after_totals[key] += value

    trips_out = trip_report(trip_table, baseline, boardings, eligibility, selection, cfg.rules)
    blocks_out = block_report(baseline, after, trip_table, cfg.rules)
    remaining = trip_table.loc[~trip_table.index.isin(selection.cut_trip_ids)]
    cycle_before = cycle_time_table(
        trip_table, baseline, cfg.rules, periods, cfg.recovery_target_pct
    )
    cycle_after = cycle_time_table(remaining, after, cfg.rules, periods, cfg.recovery_target_pct)
    cycles_out = _merge_before_after(cycle_before, cycle_after)
    profile = vehicle_profile(baseline, after, cfg.rules, cfg.vehicle_profile_bin_min)

    total_boardings = float(sum(boardings.values()))
    lost = float(groups["boardings_lost"].sum()) if not groups.empty else 0.0
    saved_basis = (
        (base_totals["vehicle_min"] - after_totals["vehicle_min"])
        if basis == "vehicle_hours"
        else (base_totals["revenue_min"] - after_totals["revenue_min"])
    )
    solo_layover_only = int(
        (trips_out["eligible"] & trips_out["solo_cut_type"].eq("mid_block_layover_only")).sum()
    )
    summary_pairs: list[tuple[str, Any]] = [
        ("analysis_date", str(chosen_date) if chosen_date else "all trips (no calendar)"),
        ("service_ids", ";".join(sorted(service_ids)) if service_ids else ""),
        ("trips_on_day", len(trip_table)),
        ("blocks", int(sum(1 for b in baseline if not b.startswith(_NO_BLOCK_PREFIX)))),
        ("trips_with_ridership", len(boardings)),
        ("trips_eligible_to_cut", len(eligible)),
        ("eligible_trips_whose_solo_cut_is_layover_only", solo_layover_only),
        ("baseline_boardings", round(total_boardings, 1)),
        ("baseline_revenue_hours", round(base_totals["revenue_min"] / 60.0, 1)),
        ("baseline_layover_hours", round(base_totals["layover_min"] / 60.0, 1)),
        ("baseline_vehicle_hours", round(base_totals["vehicle_min"] / 60.0, 1)),
        ("target_basis", basis),
        ("target_percent", cfg.target_cut_percent),
        ("target_hours", round(target_min / 60.0, 1)),
        ("groups_selected", len(groups)),
        ("trips_cut", len(selection.cut_trip_ids)),
        (
            "revenue_hours_cut",
            round((base_totals["revenue_min"] - after_totals["revenue_min"]) / 60.0, 1),
        ),
        (
            "vehicle_hours_saved",
            round((base_totals["vehicle_min"] - after_totals["vehicle_min"]) / 60.0, 1),
        ),
        (
            "achieved_percent_of_basis",
            round(100.0 * saved_basis / basis_total, 2) if basis_total else 0.0,
        ),
        ("boardings_lost", round(lost, 1)),
        (
            "boardings_lost_percent",
            round(100.0 * lost / total_boardings, 2) if total_boardings else 0.0,
        ),
        ("peak_vehicles_before", int(profile["vehicles_before"].max()) if not profile.empty else 0),
        ("peak_vehicles_after", int(profile["vehicles_after"].max()) if not profile.empty else 0),
        ("candidates_rejected_route_span", selection.rejected["route_span"]),
        ("candidates_rejected_max_headway", selection.rejected["max_headway"]),
    ]
    summary = pd.DataFrame(summary_pairs, columns=["metric", "value"])

    if target_min > 0 and saved_basis < target_min:
        logging.warning(
            "Target not reached: cuts that free vehicle time total %.1f of the %.1f %s "
            "targeted. The rest of the low-ridership trips sit mid-block where cutting "
            "them only adds layover, or are protected by a guard.",
            saved_basis / 60.0,
            target_min / 60.0,
            basis.replace("_", " "),
        )

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {
        GROUPS_FILENAME: groups,
        TRIPS_FILENAME: trips_out,
        BLOCKS_FILENAME: blocks_out,
        CYCLE_TIMES_FILENAME: cycles_out,
        VEHICLE_PROFILE_FILENAME: profile,
        SUMMARY_FILENAME: summary,
    }
    if cfg.export_candidates:
        tables[CANDIDATES_FILENAME] = candidates
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
        logging.info("Wrote %s (%d rows).", out / name, len(table))

    summary_lines = [f"{metric:<48} {value}" for metric, value in summary_pairs]
    for line in summary_lines:
        logging.info(line)
    if not write_run_log(out, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return tables


def _required(feed: Mapping[str, Optional[pd.DataFrame]], key: str) -> pd.DataFrame:
    """Return a required GTFS table, raising if the loader left it empty."""
    table = feed.get(key)
    if table is None:
        raise ValueError(f"GTFS feed has no {key}.txt.")
    return table


def _merge_before_after(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Join two cycle-time tables on route and period, suffixing their measures."""
    keys = ["route_id", "route_label", "period"]
    if before.empty:
        return before
    if after.empty:
        after = pd.DataFrame(columns=before.columns)
    merged = before.merge(after, on=keys, how="left", suffixes=("_before", "_after"))
    return merged


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser; every default is a CONFIGURATION constant."""
    p = argparse.ArgumentParser(
        description=(
            "Select low-ridership trip cuts that free vehicle time in GTFS blocks, up to a "
            "target share of hours."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gtfs", default=GTFS_PATH, help="GTFS folder or .zip.")
    p.add_argument("--ridership", default=RIDERSHIP_PATH, help="Trip-level ridership file.")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument("--service-day", default=SERVICE_DAY, help="weekday or monday..sunday.")
    p.add_argument("--service-date", default=SERVICE_DATE, help="Explicit YYYY-MM-DD date.")
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
        "--target-percent",
        type=float,
        default=TARGET_CUT_PERCENT,
        help="Percent of the day's hours to cut (0 = select nothing).",
    )
    p.add_argument(
        "--target-basis",
        default=TARGET_HOURS_BASIS,
        choices=_TARGET_BASES,
        help="Hours the target counts.",
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
    p.add_argument(
        "--export-candidates",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_CANDIDATES,
        help="Write the ranked baseline candidate list.",
    )
    return p


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

    placeholders = [
        name
        for name, value in (
            ("GTFS_PATH (--gtfs)", args.gtfs),
            ("RIDERSHIP_PATH (--ridership)", args.ridership),
            ("OUTPUT_DIR (--output-dir)", args.output_dir),
        )
        if str(value).startswith(_PLACEHOLDER_PREFIX)
    ]
    if placeholders:
        logging.error(
            "Still a placeholder: %s. Update the CONFIGURATION section or pass the flags.",
            ", ".join(placeholders),
        )
        return 2

    cfg = Config(
        gtfs_path=str(Path(args.gtfs).expanduser()),
        ridership_path=str(Path(args.ridership).expanduser()),
        output_dir=Path(args.output_dir).expanduser(),
        service_day=args.service_day,
        service_date=args.service_date,
        ridership=RidershipSpec(
            match_mode=args.match_mode,
            trip_col=args.trip_id_col,
            boardings_col=args.boardings_col,
            route_col=args.route_col,
            start_time_col=args.start_time_col,
            date_col=args.date_col,
        ),
        target_cut_percent=args.target_percent,
        target_hours_basis=args.target_basis,
        rules=BlockRules(args.min_release_gap, args.pull_deadhead),
        guards=Guards(args.protect_above, args.max_headway, args.protect_span),
        recovery_target_pct=args.recovery_target,
        export_candidates=args.export_candidates,
    )
    errors = validate_config(cfg)
    if errors:
        for message in errors:
            logging.error("Configuration: %s", message)
        return 2

    try:
        run(cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Trip cut savings analysis completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
