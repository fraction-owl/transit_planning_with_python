"""Compute the routes each target route can transfer to within one GTFS feed.

For every *target* route the user designates, this script finds the other routes a
rider can transfer to, subject to a **space buffer** (how far a rider will walk
between an alighting stop and a boarding stop) and an optional **time buffer**
(the connecting trip must depart after the feeder trip arrives — accounting for
walk time — and within a maximum wait).

Targets are chosen with ``TARGET_ROUTE_TOKENS`` (matched against
``route_short_name`` or ``route_id``); every route in the feed is eligible as a
*connector*. To find transfers between agencies, first merge their feeds with
``gtfs_feed_merger.py``, which combines duplicate stops (so a stop both
agencies serve counts as one shared stop) and aligns their calendars, then run
this script on the merged feed. ``TARGET_AGENCY_IDS`` can restrict targets to
one agency's routes, and connectors from another agency are labelled
``"<agency_id>:<route>"``.

Transfers are directional: an A -> B transfer means a rider can ride route A, get off, and
board route B. The time check uses A's arrival times at the alighting stop and B's departure
times at the (possibly different) nearby boarding stop. Riders must be allowed off A at the
alighting stop and onto B at the boarding stop: stop times with ``drop_off_type`` 1 never
count as feeder arrivals, and stop times with ``pickup_type`` 1 never count as connector
departures, whether or not the time check is enabled. A stop time counts as a feeder arrival
only if its trip has an earlier ``stop_sequence`` and as a connector departure only if it
has a later one, judged per visit (so a loop still lets riders off when it returns to its
origin). Stop pairs that transfers.txt marks not possible (``transfer_type`` 3) never count,
and trips with ``exact_times=1`` in frequencies.txt count at every scheduled run.

By default (``DAY_OF_WEEK = "weekday"``) one representative weekday is analysed,
the date ``gtfs_calendar_inspector.py`` reports: from calendar.txt and
calendar_dates.txt, the median Monday–Friday date (federal holidays skipped)
running the most common set of services. Only that date's trips are kept, so
feeder and connector trips always run on the same day. A day name picks a
representative date of that day; ``SERVICE_DATE`` analyses one explicit date
(e.g. the day after Thanksgiving); ``None`` / ``--day all`` pools every trip
(which can pair trips that never run on the same day, inflating counts). A
calendar without usable dates falls back to calendar.txt's weekday columns.

Inputs:
    - A GTFS feed folder or ``.zip`` with stops.txt, routes.txt, trips.txt, and
      stop_times.txt (a single agency's feed, or a merged multi-agency feed
      from gtfs_feed_merger.py). calendar.txt and calendar_dates.txt are used
      when a day filter or SERVICE_DATE is set.

Outputs:
    - A summary CSV keyed on ``route_id``: one row per target route with a
      transfer count and the comma-separated list of routes it can transfer to.
    - An optional detail CSV: one row per (target route, connector route) pair
      with the number of qualifying stop pairs, nearest walk distance, and the
      shortest feasible wait observed.

Defaults come from the CONFIGURATION block; ``--gtfs-dir`` / ``--output-dir`` /
``--day`` / ``--service-date`` / ``--log-level`` override them, so the script can run standalone or
under the prep_features_public.py orchestrator.

Typical usage:
    Update the paths in the CONFIGURATION block (or pass the CLI flags above)
    and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import zipfile
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, List, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS feed folder or .zip containing stops.txt, routes.txt, trips.txt, and
# stop_times.txt. For transfers between agencies, merge their feeds first with
# gtfs_feed_merger.py and point this at the merged feed. Use raw strings (r"")
# for Windows paths.
GTFS_FOLDER: str = r"Path\To\Your\GTFS_Folder"

# Folder where the output CSV(s) are written.
OUTPUT_DIR: Path = Path(r"Path\To\Output_Folder")

# Routes we want transfers FOR. Each token is matched against route_short_name OR
# route_id. Leave empty to treat every route as a target.
TARGET_ROUTE_TOKENS: set[str] = set()  # e.g. {"10", "20"}; empty = every route is a target

# Optionally restrict targets to these routes.agency_id values, e.g. {"FFX"} in a
# merged multi-agency feed. None = any agency. Connectors are always drawn from
# every agency regardless of this setting.
TARGET_AGENCY_IDS: Optional[set[str]] = None

# --- Space buffer ---------------------------------------------------------
# Maximum walking distance between an alighting stop and a boarding stop for the
# pair to count as a transfer opportunity. Shared stops have distance 0.
MAX_TRANSFER_DISTANCE: float = 0.25
MAX_TRANSFER_DISTANCE_UNIT = "miles"  # "miles" | "feet" | "meters"

# Projected CRS (in METERS) used for distance math. Pick one appropriate for your
# study area; the default is NAD83 / Conterminous US Albers, fine for short
# walking distances anywhere in the US. For non-US feeds, set a local metric CRS.
PROJECTED_CRS = "EPSG:5070"

# --- Time buffer ----------------------------------------------------------
# When True, a connector route only counts if at least one scheduled connection
# is feasible: feeder arrival + walk time <= connector departure <= feeder
# arrival + MAX_TRANSFER_WAIT_MINUTES. When False, any spatial co-location of the
# two routes counts (a pure proximity transfer map). Either way, the feeder stop
# must allow drop-off and the connector stop must allow pickup, and under a day
# filter the two trips' services must share an operating date.
ENABLE_TIME_CHECK = True

# Assumed walking speed used to convert the stop-to-stop distance into a minimum
# transfer time. 3.0 mph is a common planning value.
WALK_SPEED_MPH: float = 3.0

# Extra slack (minutes) added on top of the computed walk time before a connector
# departure can be caught (e.g., time to find the stop). 0 = walk time only.
MIN_TRANSFER_BUFFER_MINUTES: float = 0.0

# Longest a rider will wait at the connector stop, in minutes.
MAX_TRANSFER_WAIT_MINUTES: float = 30.0

# Service-day filter. "weekday" (the default) keeps the trips that run on a
# representative weekday: the date, picked from calendar.txt and
# calendar_dates.txt, that runs the most common Monday-Friday set of services
# (observed federal holidays skipped). A single day name (e.g., "friday") picks
# a representative date of that day. None keeps all trips regardless of
# service day. Feeds with no calendar at all keep all trips, with a warning.
DAY_OF_WEEK: Optional[str] = "weekday"

# Explicit service date (YYYYMMDD or YYYY-MM-DD), e.g. "20261127" for the day
# after Thanksgiving. When set, it replaces DAY_OF_WEEK: only the trips of the
# services running on that date are kept. "" = use DAY_OF_WEEK.
SERVICE_DATE: str = ""

# --- Output ---------------------------------------------------------------
SUMMARY_FILENAME = "route_transfers_summary.csv"
DETAIL_FILENAME = "route_transfers_detail.csv"
WRITE_DETAIL = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

_PLACEHOLDER_FEED = r"Path\To\Your\GTFS_Folder"
_PLACEHOLDER_OUTPUT = Path(r"Path\To\Output_Folder")
_REQUIRED_FILES = ("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
_DAY_COLUMNS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

# === END CONFIG ===

# =============================================================================
# FUNCTIONS
# =============================================================================


def parse_gtfs_time(value: object) -> Optional[float]:
    """Convert an HH:MM:SS GTFS time into seconds past midnight.

    Hours may exceed 24 (service after midnight), and those values are preserved.

    Args:
        value: A time string such as ``"07:35:00"`` or ``"25:10:00"``.

    Returns:
        Seconds past midnight as a float, or None when the value is missing or
        malformed.
    """
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    except (TypeError, ValueError):
        return None
    return hours * 3600 + minutes * 60 + seconds


def distance_to_meters(distance: float, unit: str) -> float:
    """Convert a distance to meters.

    Args:
        distance: The numeric distance value.
        unit: One of ``"miles"``, ``"feet"``, or ``"meters"`` (case-insensitive).

    Returns:
        The distance expressed in meters.

    Raises:
        ValueError: If ``unit`` is not recognized.
    """
    factors = {"miles": 1609.34, "feet": 0.3048, "meters": 1.0}
    key = unit.lower()
    if key not in factors:
        raise ValueError(f"Unknown distance unit '{unit}'. Use miles, feet, or meters.")
    return distance * factors[key]


def service_ids_active_on_day(calendar_df: Optional[pd.DataFrame], day: str) -> Optional[set[str]]:
    """Return the set of service_ids whose regular service runs on the given day.

    The fallback of :func:`select_services` when the calendar has no usable
    dates. Only ``calendar.txt`` weekly flags are consulted; date-specific exceptions in
    ``calendar_dates.txt`` are not mapped to weekdays, so holiday-exception
    service never qualifies. When the calendar is unavailable, None is returned
    to signal "do not filter".

    Args:
        calendar_df: Parsed calendar.txt, or None if the feed has none.
        day: Lowercase day name (e.g. ``"monday"``), or ``"weekday"`` to keep
            every service_id that runs on at least one of Monday-Friday.

    Returns:
        A set of service_id strings active on ``day``, or None when no calendar is
        available to filter on.
    """
    key = day.lower()
    if key != "weekday" and key not in _DAY_COLUMNS:
        raise ValueError(f"DAY_OF_WEEK must be 'weekday' or one of {_DAY_COLUMNS}, got '{day}'.")
    if calendar_df is None or calendar_df.empty:
        return None

    day_cols = _DAY_COLUMNS[:5] if key == "weekday" else (key,)
    present = [c for c in day_cols if c in calendar_df.columns]
    if not present:
        return None

    mask = pd.Series(False, index=calendar_df.index)
    for col in present:
        mask |= calendar_df[col].astype(str) == "1"
    active = calendar_df.loc[mask, "service_id"]
    return {str(s) for s in active}


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


def eligible_service_dates(
    calendar_df: Optional[pd.DataFrame], day: str, service_ids: set[str]
) -> Optional[dict[str, set[dt.date]]]:
    """Map each service_id to the dates it runs on the day filter's days.

    Dates come from ``calendar.txt`` alone (weekly flags within
    ``start_date``-``end_date``), matching :func:`service_ids_active_on_day`:
    ``calendar_dates.txt`` exceptions are not applied. Two services can share a
    transfer only if their date sets intersect.

    Args:
        calendar_df: Parsed calendar.txt, or None if the feed has none.
        day: Lowercase day name (e.g. ``"monday"``), or ``"weekday"`` for
            Monday-Friday.
        service_ids: The service_ids the day filter kept.

    Returns:
        A mapping of each of ``service_ids`` to its eligible operating dates
        (empty when it never runs on one of the filter's days within its date
        range, or its dates cannot be parsed), or None when there is no
        calendar. A calendar without ``start_date``/``end_date`` columns is
        matched by weekday alone.
    """
    key = day.lower()
    if key != "weekday" and key not in _DAY_COLUMNS:
        raise ValueError(f"DAY_OF_WEEK must be 'weekday' or one of {_DAY_COLUMNS}, got '{day}'.")
    if calendar_df is None or calendar_df.empty:
        return None
    day_cols = _DAY_COLUMNS[:5] if key == "weekday" else (key,)
    weekdays = {_DAY_COLUMNS.index(col) for col in day_cols}
    if not {"start_date", "end_date"} <= set(calendar_df.columns):
        logging.warning(
            "calendar.txt has no start_date/end_date columns; matching trips by weekday only."
        )
        # One Monday-Sunday week stands in for the missing date range, so each
        # service's dates reduce to the weekdays it runs.
        calendar_df = calendar_df.assign(start_date="20000103", end_date="20000109")
    active = expand_service_active_dates(calendar_df)
    return {
        sid: {d for d in active.get(sid.strip(), set()) if d.weekday() in weekdays}
        for sid in service_ids
    }


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


def parse_service_date(text: str) -> Optional[dt.date]:
    """Parse SERVICE_DATE (``YYYYMMDD`` or ``YYYY-MM-DD``); ``""`` gives ``None``.

    Raises:
        ValueError: If *text* is not blank and not a date in either format.
    """
    value = str(text).strip()
    if not value:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"SERVICE_DATE {value!r} is not a date; use YYYYMMDD or YYYY-MM-DD.")


def select_services(
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame],
    day: Optional[str],
    service_date: Optional[dt.date] = None,
) -> tuple[Optional[set[str]], Optional[dict[str, set[dt.date]]]]:
    """Choose the services to analyse: one date's services, or every trip.

    With *service_date*, the services running on that date. Otherwise, with a
    day filter, the services running on a representative date of *day* (see
    :func:`representative_service_date`), with calendar_dates.txt exceptions
    applied and observed federal holidays skipped. Either way every kept trip
    runs on the same date, so feeder and connector trips need no date matching.

    If the calendar has no usable dates (no calendar files, or calendar.txt
    without ``start_date``/``end_date``), the day filter falls back to
    calendar.txt's weekday columns via :func:`service_ids_active_on_day`, and
    services pair only on shared dates from :func:`eligible_service_dates`.

    Args:
        calendar_df: Parsed calendar.txt, or None if the feed has none.
        calendar_dates_df: Parsed calendar_dates.txt, or None if the feed has none.
        day: ``"weekday"``, a day name, or None for no day filter.
        service_date: Explicit date that replaces *day*, or None.

    Returns:
        ``(service_ids, service_dates)``: the service_ids to keep (None keeps every
        trip) and, in the weekday-column fallback only, each service's eligible
        dates for :func:`compute_transfers` (otherwise None).

    Raises:
        ValueError: If *day* is not recognised, or no service runs on *service_date*.
    """
    if day is not None and day.lower() != "weekday" and day.lower() not in _DAY_COLUMNS:
        raise ValueError(f"DAY_OF_WEEK must be 'weekday' or one of {_DAY_COLUMNS}, got '{day}'.")
    if service_date is None and day is None:
        return None, None
    try:
        active = expand_service_active_dates(calendar_df, calendar_dates_df)
        if service_date is not None:
            ids = service_ids_active_on(active, service_date)
            if not ids:
                raise ValueError(f"No service runs on SERVICE_DATE {service_date.isoformat()}.")
            logging.info(
                "Using SERVICE_DATE %s (%s): service_id(s) %s.",
                service_date.isoformat(),
                service_date.strftime("%A"),
                ", ".join(sorted(ids)),
            )
            return ids, None
        years = {d.year for dates in active.values() for d in dates}
        holidays: set[dt.date] = set()
        for year in range(min(years, default=0), max(years, default=-1) + 2):
            holidays |= federal_holidays_observed(year)
        chosen, ids = representative_service_date(active, str(day), exclude_dates=holidays)
        logging.info(
            "Analysing %s (%s): service_id(s) %s.",
            chosen.isoformat(),
            chosen.strftime("%A"),
            ", ".join(sorted(ids)),
        )
        return ids, None
    except ValueError as exc:
        if service_date is not None:
            raise
        logging.warning(
            "Could not pick a representative %s from the calendar dates (%s); "
            "falling back to calendar.txt weekday columns.",
            day,
            exc,
        )
    keep = service_ids_active_on_day(calendar_df, str(day))
    if keep is None:
        logging.warning("No usable calendar.txt for the day filter; keeping all trips.")
        return None, None
    return keep, eligible_service_dates(calendar_df, str(day), keep)


def has_timed_connection(
    arrivals: np.ndarray,
    departures: np.ndarray,
    min_wait_seconds: float,
    max_wait_seconds: float,
) -> tuple[bool, Optional[float]]:
    """Check whether any feeder arrival can catch a connector departure in time.

    A connection is feasible when, for some feeder arrival ``a`` and connector
    departure ``d``, ``min_wait_seconds <= d - a <= max_wait_seconds``.

    Args:
        arrivals: Sorted feeder arrival times (seconds past midnight).
        departures: Sorted connector departure times (seconds past midnight).
        min_wait_seconds: Earliest the rider can board after arriving (walk time
            plus any buffer).
        max_wait_seconds: Longest the rider will wait.

    Returns:
        A tuple ``(feasible, shortest_wait_seconds)``. ``shortest_wait_seconds`` is
        the smallest qualifying ``d - a`` across all pairs, or None when infeasible.
    """
    if arrivals.size == 0 or departures.size == 0:
        return False, None

    best_wait: Optional[float] = None
    for arrival in arrivals:
        earliest = arrival + min_wait_seconds
        latest = arrival + max_wait_seconds
        # First departure at or after the earliest boardable time.
        idx = int(np.searchsorted(departures, earliest, side="left"))
        if idx < departures.size and departures[idx] <= latest:
            wait = float(departures[idx] - arrival)
            if best_wait is None or wait < best_wait:
                best_wait = wait
    return best_wait is not None, best_wait


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


def _read_feed(folder: str) -> tuple[dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Load the GTFS tables this script needs from one feed folder or ``.zip``.

    Args:
        folder: Path to the GTFS feed folder, or to a ``.zip`` of the feed.

    Returns:
        A tuple ``(tables, calendar)``: ``stops``, ``routes``, ``trips`` and
        ``stop_times`` frames, plus ``transfers``, ``frequencies`` and
        ``calendar_dates`` when the feed has them, and calendar.txt (None when
        the feed has none).

    Raises:
        FileNotFoundError: If a required GTFS file is missing.
    """
    for filename in _REQUIRED_FILES:
        with open_gtfs_file(folder, filename):
            pass  # raises FileNotFoundError naming the file when it is missing

    def _read(name: str) -> pd.DataFrame:
        with open_gtfs_file(folder, name) as handle:
            return pd.read_csv(handle, dtype=str, low_memory=False)

    stops = _read("stops.txt")
    routes = _read("routes.txt")
    trips = _read("trips.txt")
    stop_times = _read("stop_times.txt")

    calendar: Optional[pd.DataFrame] = None
    try:
        with open_gtfs_file(folder, "calendar.txt") as handle:
            calendar = pd.read_csv(handle, dtype=str)
    except FileNotFoundError:
        pass  # optional; a day filter then has nothing to filter on
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logging.warning("Could not read calendar.txt (%s).", exc)

    logging.info(
        "Feed: %d stops, %d routes, %d trips, %d stop_times.",
        len(stops),
        len(routes),
        len(trips),
        len(stop_times),
    )
    tables = {"stops": stops, "routes": routes, "trips": trips, "stop_times": stop_times}
    for name in ("transfers", "frequencies", "calendar_dates"):
        try:
            tables[name] = _read(f"{name}.txt")
        except FileNotFoundError:
            pass  # optional
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logging.warning("Could not read %s.txt (%s); it is ignored.", name, exc)
    return tables, calendar


def _build_route_table(routes: pd.DataFrame) -> pd.DataFrame:
    """Add display labels and target flags to the routes table.

    Args:
        routes: Parsed routes.txt.

    Returns:
        A routes frame indexed by ``route_id`` with ``agency_id`` (blank when
        the feed omits it), ``route_label`` (short name, else long name, else
        route_id) and an ``is_target`` flag.
    """
    routes = routes.copy()
    routes["route_id"] = routes["route_id"].astype(str)
    for col in ("agency_id", "route_short_name", "route_long_name"):
        if col not in routes.columns:
            routes[col] = ""
        routes[col] = routes[col].fillna("").astype(str).str.strip()

    label = routes["route_short_name"].where(routes["route_short_name"] != "")
    label = label.fillna(routes["route_long_name"].where(routes["route_long_name"] != ""))
    routes["route_label"] = label.fillna(routes["route_id"])

    if TARGET_ROUTE_TOKENS:
        tokens = {str(t) for t in TARGET_ROUTE_TOKENS}
        is_target = routes["route_short_name"].isin(tokens) | routes["route_id"].isin(tokens)
    else:
        is_target = pd.Series(True, index=routes.index)

    if TARGET_AGENCY_IDS:
        is_target = is_target & routes["agency_id"].isin({str(a) for a in TARGET_AGENCY_IDS})

    routes["is_target"] = is_target
    return routes.set_index("route_id", drop=False)


def _project_stops(stops: pd.DataFrame) -> gpd.GeoDataFrame:
    """Project unique stops to ``PROJECTED_CRS`` and attach metric x/y columns.

    Args:
        stops: Stops frame with ``stop_id``, ``stop_lat``, ``stop_lon``.

    Returns:
        A GeoDataFrame of unique stops with numeric ``x`` and ``y`` (meters).
    """
    unique = stops.drop_duplicates("stop_id").copy()
    unique["stop_id"] = unique["stop_id"].astype(str)
    unique["stop_lat"] = pd.to_numeric(unique["stop_lat"], errors="coerce")
    unique["stop_lon"] = pd.to_numeric(unique["stop_lon"], errors="coerce")
    before = len(unique)
    unique = unique.dropna(subset=["stop_lat", "stop_lon"])
    if len(unique) < before:
        logging.warning("Dropped %d stop(s) with missing coordinates.", before - len(unique))

    gdf = gpd.GeoDataFrame(
        unique,
        geometry=[Point(lon, lat) for lon, lat in zip(unique["stop_lon"], unique["stop_lat"])],
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def _expand_exact_frequencies(
    events: pd.DataFrame, frequencies: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Replace each exact-times frequency trip's template with its scheduled runs.

    A frequencies.txt row with ``exact_times=1`` schedules a run at
    ``start_time`` and every ``headway_secs`` after it, up to but not including
    ``end_time``. The trip's stop_times give each stop's time relative to its
    first stop, so every run is the template shifted to leave its first stop at
    that start. Headway-based rows (``exact_times`` blank or 0) have no exact
    departures; those trips keep their template times, with a warning.

    Args:
        events: Stop events with ``trip_id``, ``arrival_sec`` and ``departure_sec``.
        frequencies: Parsed frequencies.txt, or None when the feed has none.

    Returns:
        *events* with every exact-times trip's rows repeated once per run.
    """
    if frequencies is None or frequencies.empty or events.empty:
        return events
    required = ("trip_id", "start_time", "end_time", "headway_secs")
    missing = [col for col in required if col not in frequencies.columns]
    if missing:
        logging.warning("frequencies.txt has no %s column(s); it is ignored.", ", ".join(missing))
        return events

    freq = frequencies.assign(trip_id=frequencies["trip_id"].astype(str).str.strip())
    freq = freq[freq["trip_id"].isin(set(events["trip_id"]))]
    exact_col = freq["exact_times"] if "exact_times" in freq.columns else pd.Series("", freq.index)
    exact = exact_col.fillna("").astype(str).str.strip() == "1"
    headway_only = freq.loc[~exact, "trip_id"].nunique()
    if headway_only:
        logging.warning(
            "%d trip(s) in frequencies.txt run on a headway without exact times "
            "(exact_times 0); only their stop_times template times are used.",
            headway_only,
        )

    runs: list[tuple[str, float]] = []
    invalid = 0
    for trip_id, start_text, end_text, headway_text in zip(
        freq.loc[exact, "trip_id"],
        freq.loc[exact, "start_time"],
        freq.loc[exact, "end_time"],
        freq.loc[exact, "headway_secs"],
    ):
        start, end = parse_gtfs_time(start_text), parse_gtfs_time(end_text)
        headway = pd.to_numeric(headway_text, errors="coerce")
        if start is None or end is None or not headway > 0 or start >= end:
            invalid += 1
            continue
        runs.extend((trip_id, float(run)) for run in np.arange(start, end, float(headway)))
    if invalid:
        logging.warning(
            "%d exact_times row(s) in frequencies.txt have an unreadable time window or "
            "headway; their trips keep only their stop_times template times.",
            invalid,
        )
    if not runs:
        return events

    starts = pd.DataFrame(runs, columns=["trip_id", "_run_start"])
    expanded_ids = set(starts["trip_id"])
    template = events[events["trip_id"].isin(expanded_ids)]
    first = template.groupby("trip_id")["departure_sec"].min().rename("_template_start")
    expanded = template.merge(starts, on="trip_id").join(first, on="trip_id")
    offset = expanded["_run_start"] - expanded["_template_start"]
    expanded["arrival_sec"] = expanded["arrival_sec"] + offset
    expanded["departure_sec"] = expanded["departure_sec"] + offset
    logging.info(
        "frequencies.txt: %d exact-times trip(s) expanded to %d scheduled run(s).",
        len(expanded_ids),
        len(starts),
    )
    rest = events[~events["trip_id"].isin(expanded_ids)]
    expanded = expanded.drop(columns=["_run_start", "_template_start"])
    return pd.concat([rest, expanded], ignore_index=True)


def _build_stop_events(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    keep_service_ids: Optional[set[str]],
    frequencies: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join stop_times to trips and resolve feeder/connector event times.

    Args:
        stop_times: Parsed stop_times.txt.
        trips: Parsed trips.txt.
        keep_service_ids: service_ids to keep (day filter), or None to keep
            every trip.
        frequencies: Parsed frequencies.txt, or None. Trips with
            ``exact_times=1`` rows get one set of events per scheduled run
            (see :func:`_expand_exact_frequencies`).

    Returns:
        A frame with ``stop_id``, ``route_id``, ``service_id``,
        ``arrival_sec`` (feeder time), ``departure_sec`` (connector time),
        ``can_alight`` (``drop_off_type`` is not 1 and the trip has an earlier
        ``stop_sequence``) and ``can_board`` (``pickup_type`` is not 1 and the
        trip has a later ``stop_sequence``).
    """
    trip_cols = trips[["trip_id", "route_id", "service_id"]].astype(str)
    events = stop_times.astype({"trip_id": str, "stop_id": str}).merge(
        trip_cols, on="trip_id", how="inner"
    )

    if keep_service_ids is not None:
        mask = events["service_id"].isin(keep_service_ids)
        logging.info("Day-of-week filter kept %d/%d stop events.", int(mask.sum()), len(events))
        events = events.loc[mask].copy()

    arrival = events["arrival_time"].map(parse_gtfs_time)
    departure = events["departure_time"].map(parse_gtfs_time)
    # Feeders use arrival (fall back to departure); connectors use departure
    # (fall back to arrival) so single-time rows still participate.
    events["arrival_sec"] = arrival.fillna(departure)
    events["departure_sec"] = departure.fillna(arrival)
    # pickup_type / drop_off_type 1 means riders cannot board / get off here.
    # Blank, 0, 2 (phone agency) and 3 (coordinate with driver) still allow it.
    for flag, column in (("can_alight", "drop_off_type"), ("can_board", "pickup_type")):
        if column in events.columns:
            events[flag] = events[column].fillna("").astype(str).str.strip() != "1"
        else:
            events[flag] = True
    # Riders get off a trip after riding it from an earlier stop and board it to
    # ride on to a later one. Each visit is judged by its own stop_sequence, so a
    # loop that returns to its origin still lets riders off on the final visit.
    if "stop_sequence" in events.columns:
        sequence = pd.to_numeric(events["stop_sequence"], errors="coerce")
        by_trip = sequence.groupby(events["trip_id"])
        events["can_alight"] &= sequence > by_trip.transform("min")
        events["can_board"] &= sequence < by_trip.transform("max")
        unplaced = int(sequence.isna().sum())
        if unplaced:
            logging.warning(
                "%d stop time(s) have a missing or non-numeric stop_sequence; "
                "they are not used as transfer points.",
                unplaced,
            )
    else:
        logging.warning(
            "stop_times.txt has no stop_sequence column, so trip origins cannot be told "
            "from destinations; every stop counts for both boarding and alighting."
        )
    events = _expand_exact_frequencies(events, frequencies)
    return events[
        [
            "stop_id",
            "route_id",
            "service_id",
            "arrival_sec",
            "departure_sec",
            "can_alight",
            "can_board",
        ]
    ]


# Service key used for every event when trips are not matched by service.
_ANY_SERVICE = ""

_StopRouteServiceArrays = dict[tuple[str, str], dict[str, np.ndarray]]


def _index_events(
    events: pd.DataFrame, by_service: bool
) -> tuple[
    dict[str, set[str]], dict[str, set[str]], _StopRouteServiceArrays, _StopRouteServiceArrays
]:
    """Build lookups of routes-per-stop and sorted time arrays per (stop, route, service).

    Alighting lookups use only events where riders may get off
    (``can_alight``); boarding lookups use only events where riders may get on
    (``can_board``).

    Args:
        events: Output of :func:`_build_stop_events`.
        by_service: Keep each ``service_id``'s times apart so trips can be
            matched by operating date. When False, every event falls under one
            shared service key.

    Returns:
        A tuple ``(alight_routes, board_routes, arrivals, departures)`` where
        ``alight_routes`` maps a ``stop_id`` to the set of routes riders can get
        off at it, ``board_routes`` maps it to the set of routes riders can get
        on at it, ``arrivals`` maps ``(stop_id, route_id)`` to
        ``{service_id: sorted feeder arrival seconds}``, and ``departures`` maps
        it to ``{service_id: sorted connector departure seconds}``. A service
        appears only if riders can get off (or on) one of its trips there.
    """
    alight_routes: dict[str, set[str]] = {}
    board_routes: dict[str, set[str]] = {}
    for stop_id, route_id, can_alight, can_board in zip(
        events["stop_id"], events["route_id"], events["can_alight"], events["can_board"]
    ):
        if can_alight:
            alight_routes.setdefault(stop_id, set()).add(route_id)
        if can_board:
            board_routes.setdefault(stop_id, set()).add(route_id)

    service = events["service_id"] if by_service else pd.Series(_ANY_SERVICE, index=events.index)
    arrivals: _StopRouteServiceArrays = {}
    departures: _StopRouteServiceArrays = {}
    for (stop_id, route_id, service_id), group in events.groupby(
        [events["stop_id"], events["route_id"], service], sort=False
    ):
        key = (str(stop_id), str(route_id))
        if group["can_alight"].any():
            alight = group.loc[group["can_alight"], "arrival_sec"].dropna()
            arrivals.setdefault(key, {})[str(service_id)] = np.sort(alight.to_numpy(dtype=float))
        if group["can_board"].any():
            board = group.loc[group["can_board"], "departure_sec"].dropna()
            departures.setdefault(key, {})[str(service_id)] = np.sort(board.to_numpy(dtype=float))
    return alight_routes, board_routes, arrivals, departures


def _best_connection(
    arrivals_by_service: Mapping[str, np.ndarray],
    departures_by_service: Mapping[str, np.ndarray],
    same_day: Callable[[str, str], bool],
    min_wait_seconds: float,
    max_wait_seconds: float,
) -> tuple[bool, Optional[float]]:
    """Check whether any feeder trip can reach any connector trip on a shared day.

    Only (feeder service, connector service) pairs that ``same_day`` accepts
    are compared. With the time check off, any such pair is enough.

    Args:
        arrivals_by_service: Feeder arrival seconds at the alighting stop,
            keyed by service_id.
        departures_by_service: Connector departure seconds at the boarding
            stop, keyed by service_id.
        same_day: Returns True when two service_ids share an operating date.
        min_wait_seconds: Earliest the rider can board after arriving.
        max_wait_seconds: Longest the rider will wait.

    Returns:
        A tuple ``(feasible, shortest_wait_seconds)``. ``shortest_wait_seconds``
        is None when infeasible or when the time check is disabled.
    """
    best_wait: Optional[float] = None
    for service_a, arrivals in arrivals_by_service.items():
        for service_b, departures in departures_by_service.items():
            if not same_day(service_a, service_b):
                continue
            if not ENABLE_TIME_CHECK:
                return True, None
            feasible, wait = has_timed_connection(
                arrivals, departures, min_wait_seconds, max_wait_seconds
            )
            if feasible and wait is not None and (best_wait is None or wait < best_wait):
                best_wait = wait
    return best_wait is not None, best_wait


# One transfers.txt row as it applies to a stop pair: (from_route_id or None,
# to_route_id or None, stop rank, prohibited). Stop rank counts the row's stop
# IDs that name the stop itself rather than its parent station (0-2).
TransferRule = tuple[Optional[str], Optional[str], int, bool]


def build_transfer_rules(
    transfers: Optional[pd.DataFrame], stops: pd.DataFrame
) -> dict[tuple[str, str], list[TransferRule]]:
    """Index transfers.txt rows by the (from stop, to stop) pairs they govern.

    A ``from_stop_id`` or ``to_stop_id`` naming a station governs each stop
    whose ``parent_station`` is that station. Rows naming a trip
    (``from_trip_id``/``to_trip_id``) are skipped with a warning, because this
    script counts transfers between routes rather than individual trips. Rows
    without both stop IDs, and in-seat rows (``transfer_type`` 4 or 5), say
    nothing about walking between stops and are skipped too.

    Args:
        transfers: Parsed transfers.txt, or None when the feed has none.
        stops: Parsed stops.txt (``parent_station`` is optional).

    Returns:
        Mapping of ``(from_stop_id, to_stop_id)`` to the rules that apply there;
        :func:`transfer_prohibited` decides between them.
    """
    if transfers is None or transfers.empty:
        return {}

    def column(name: str) -> pd.Series:
        if name not in transfers.columns:
            return pd.Series("", index=transfers.index)
        return transfers[name].fillna("").astype(str).str.strip()

    children: dict[str, set[str]] = {}
    if "parent_station" in stops.columns:
        parents = stops["parent_station"].fillna("").astype(str).str.strip()
        for stop_id, parent in zip(stops["stop_id"].astype(str), parents):
            if parent:
                children.setdefault(parent, set()).add(stop_id)

    def governed(stop_id: str) -> list[tuple[str, int]]:
        return [(stop_id, 1), *((child, 0) for child in sorted(children.get(stop_id, ())))]

    rules: dict[tuple[str, str], list[TransferRule]] = {}
    trip_rows = 0
    prohibiting = 0
    kinds = column("transfer_type").replace("", "0")
    for from_stop, to_stop, from_route, to_route, from_trip, to_trip, kind in zip(
        column("from_stop_id"),
        column("to_stop_id"),
        column("from_route_id"),
        column("to_route_id"),
        column("from_trip_id"),
        column("to_trip_id"),
        kinds,
    ):
        if from_trip or to_trip:
            trip_rows += 1
            continue
        if not from_stop or not to_stop or kind not in {"0", "1", "2", "3"}:
            continue
        prohibited = kind == "3"
        prohibiting += prohibited
        for stop_a, rank_a in governed(from_stop):
            for stop_b, rank_b in governed(to_stop):
                rules.setdefault((stop_a, stop_b), []).append(
                    (from_route or None, to_route or None, rank_a + rank_b, prohibited)
                )
    if trip_rows:
        logging.warning(
            "transfers.txt: %d row(s) name a from_trip_id or to_trip_id and are not "
            "applied; this script compares routes, not individual trips.",
            trip_rows,
        )
    if prohibiting:
        logging.info("transfers.txt: %d row(s) mark a transfer as not possible.", prohibiting)
    return rules


def transfer_prohibited(rules: Sequence[TransferRule], from_route: str, to_route: str) -> bool:
    """Return True when the most specific matching rule says the transfer is not possible.

    A rule matches when each route it names equals the feeder (*from_route*)
    or connector (*to_route*). Rules naming more of the two routes are more
    specific; among those naming as many, rules naming the stops themselves
    beat rules naming their parent stations. Equally specific rules that
    disagree resolve to not possible.

    Args:
        rules: The rules for one stop pair, from :func:`build_transfer_rules`.
        from_route: The feeder route_id.
        to_route: The connector route_id.

    Returns:
        True when the transfer is not possible (``transfer_type`` 3).
    """
    best: Optional[tuple[int, int]] = None
    prohibited = False
    for rule_from, rule_to, stop_rank, rule_prohibited in rules:
        if (rule_from and rule_from != from_route) or (rule_to and rule_to != to_route):
            continue
        rank = (int(rule_from is not None) + int(rule_to is not None), stop_rank)
        if best is None or rank > best:
            best, prohibited = rank, rule_prohibited
        elif rank == best:
            prohibited = prohibited or rule_prohibited
    return prohibited


def compute_transfers(
    stops_gdf: gpd.GeoDataFrame,
    events: pd.DataFrame,
    routes: pd.DataFrame,
    radius_m: float,
    service_dates: Optional[Mapping[str, set[dt.date]]] = None,
    transfer_rules: Optional[Mapping[tuple[str, str], Sequence[TransferRule]]] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute feasible transfers from each target route to connector routes.

    A stop pair counts only where riders can get off the target route at the
    alighting stop and get on the connector at the boarding stop, and where
    transfers.txt does not mark the transfer as not possible, whether or not
    the time check is enabled. When ``service_dates`` is given, the feeder
    and connector trips must also belong to services that share an operating
    date.

    Args:
        stops_gdf: Projected unique stops with ``x``/``y`` in meters.
        events: Stop events from :func:`_build_stop_events`.
        routes: Route table indexed by ``route_id`` with ``is_target``.
        radius_m: Space buffer in meters.
        service_dates: Each service_id's eligible operating dates, from
            :func:`eligible_service_dates`. A service missing from it pairs
            with nothing. None pools every trip regardless of service.
        transfer_rules: transfers.txt rules by stop pair, from
            :func:`build_transfer_rules`. None applies no rules.

    Returns:
        A nested mapping ``target_route_id -> connector_route_id -> stats``,
        where stats holds ``stop_pairs``, ``min_distance_m``, and
        ``min_wait_seconds`` (NaN when the time check is disabled).
    """
    alight_routes, board_routes, arrivals, departures = _index_events(
        events, by_service=service_dates is not None
    )
    target_ids = set(routes.index[routes["is_target"]])

    shares_date: dict[tuple[str, str], bool] = {}

    def same_day(service_a: str, service_b: str) -> bool:
        if service_dates is None:
            return True
        key = (service_a, service_b)
        if key not in shares_date:
            dates_a = service_dates.get(service_a, set())
            shares_date[key] = not dates_a.isdisjoint(service_dates.get(service_b, set()))
        return shares_date[key]

    coords = np.column_stack([stops_gdf["x"].to_numpy(), stops_gdf["y"].to_numpy()])
    stop_ids = stops_gdf["stop_id"].to_numpy()
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius_m, p=2.0)

    walk_mps = WALK_SPEED_MPH * 1609.34 / 3600.0
    buffer_sec = MIN_TRANSFER_BUFFER_MINUTES * 60.0
    max_wait_sec = MAX_TRANSFER_WAIT_MINUTES * 60.0

    results: dict[str, dict[str, dict[str, float]]] = {}

    for i, neigh in enumerate(neighbors):
        stop_a = stop_ids[i]
        targets_here = [r for r in alight_routes.get(stop_a, ()) if r in target_ids]
        if not targets_here:
            continue
        for j in neigh:
            stop_b = stop_ids[j]
            routes_b = board_routes.get(stop_b)
            if not routes_b:
                continue
            distance = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
            min_wait_sec = distance / walk_mps + buffer_sec if walk_mps > 0 else buffer_sec
            pair_rules = transfer_rules.get((stop_a, stop_b)) if transfer_rules else None

            for route_a in targets_here:
                arr = arrivals.get((stop_a, route_a), {})
                for route_b in routes_b:
                    if route_b == route_a:
                        continue
                    if pair_rules and transfer_prohibited(pair_rules, route_a, route_b):
                        continue
                    dep = departures.get((stop_b, route_b), {})
                    feasible, wait = _best_connection(
                        arr, dep, same_day, min_wait_sec, max_wait_sec
                    )
                    if not feasible:
                        continue

                    conn = results.setdefault(route_a, {}).setdefault(
                        route_b,
                        {"stop_pairs": 0.0, "min_distance_m": np.inf, "min_wait_seconds": np.inf},
                    )
                    conn["stop_pairs"] += 1
                    conn["min_distance_m"] = min(conn["min_distance_m"], distance)
                    if wait is not None:
                        conn["min_wait_seconds"] = min(conn["min_wait_seconds"], wait)

    return results


def _connector_label(connector_id: str, target_agency: str, routes: pd.DataFrame) -> str:
    """Format a connector route for the output list, prefixing other agencies' routes.

    Args:
        connector_id: The connector ``route_id``.
        target_agency: The target route's ``agency_id`` (may be blank).
        routes: Route table indexed by ``route_id``.

    Returns:
        The route label, prefixed with ``"<agency_id>:"`` when the connector
        belongs to a different agency than the target.
    """
    row = routes.loc[connector_id]
    label = str(row["route_label"])
    agency = str(row["agency_id"])
    return label if agency in ("", target_agency) else f"{agency}:{label}"


def build_output_tables(
    results: dict[str, dict[str, dict[str, float]]], routes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the transfer results into a summary and a detail DataFrame.

    Args:
        results: Output of :func:`compute_transfers`.
        routes: Route table indexed by ``route_id``.

    Returns:
        A ``(summary, detail)`` tuple. The summary has one row per target route
        (including targets with zero transfers), keyed on ``route_id`` so it can
        be joined downstream (e.g. by the prep_features_public.py orchestrator);
        the detail has one row per (target, connector) pair.
    """
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    target_ids = list(routes.index[routes["is_target"]])
    for target_id in target_ids:
        target = routes.loc[target_id]
        target_agency = str(target["agency_id"])
        connectors = results.get(target_id, {})

        labelled = sorted(
            (
                (_connector_label(cid, target_agency, routes), cid, stats)
                for cid, stats in connectors.items()
            ),
            key=lambda item: item[0],
        )

        summary_rows.append(
            {
                "agency_id": target_agency,
                "route_id": str(target["route_id"]),
                "route_short_name": str(target["route_short_name"]),
                "route_long_name": str(target["route_long_name"]),
                "transfer_route_count": len(labelled),
                "transfer_routes": ", ".join(label for label, _, _ in labelled),
            }
        )

        for label, cid, stats in labelled:
            connector = routes.loc[cid]
            min_wait = stats["min_wait_seconds"]
            detail_rows.append(
                {
                    "target_agency_id": target_agency,
                    "target_route_id": str(target["route_id"]),
                    "target_route_short_name": str(target["route_short_name"]),
                    "connector_agency_id": str(connector["agency_id"]),
                    "connector_route_id": str(connector["route_id"]),
                    "connector_route_short_name": str(connector["route_short_name"]),
                    "connector_label": label,
                    "cross_agency": str(connector["agency_id"]) != target_agency,
                    "qualifying_stop_pairs": int(stats["stop_pairs"]),
                    "nearest_walk_distance_m": round(stats["min_distance_m"], 1),
                    "min_transfer_wait_min": (
                        round(min_wait / 60.0, 1) if np.isfinite(min_wait) else ""
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["agency_id", "route_short_name", "route_id"])
    detail = pd.DataFrame(detail_rows)
    return summary, detail


# =============================================================================
# MAIN
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args strictly, defaulting to the configuration block.

    Notebook kernels' injected argv is shielded via ``notebook_safe_argv``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the routes each target route can transfer to within one GTFS feed. "
            "Defaults come from the CONFIGURATION block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-dir",
        default=GTFS_FOLDER,
        help="GTFS feed folder or .zip (merge several agencies first with gtfs_feed_merger.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory the summary/detail CSVs are written to.",
    )
    parser.add_argument(
        "--day",
        default=DAY_OF_WEEK if DAY_OF_WEEK is not None else "all",
        help=(
            "Service-day filter: 'weekday' (a representative Mon-Fri date), a single "
            "day name (monday..sunday) for a representative date of that day, or "
            "'all' to keep every trip."
        ),
    )
    parser.add_argument(
        "--service-date",
        default=SERVICE_DATE,
        help="Explicit service date (YYYYMMDD or YYYY-MM-DD); replaces --day when set.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    args = parser.parse_args(notebook_safe_argv(argv))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the route transfer calculation and write the output CSV(s).

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    gtfs_dir = str(args.gtfs_dir)
    output_dir = Path(args.output_dir)
    if not gtfs_dir or gtfs_dir == _PLACEHOLDER_FEED or output_dir == _PLACEHOLDER_OUTPUT:
        logging.warning(
            "The GTFS feed folder and/or the output folder are still set to placeholder "
            "values. Update the CONFIGURATION section or pass --gtfs-dir/--output-dir."
        )
        return 2

    day_token = str(args.day).strip().lower()
    day_filter: Optional[str] = None if day_token in ("", "all", "none") else day_token
    try:
        service_date = parse_service_date(args.service_date)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2
    if day_filter is not None and day_filter != "weekday" and day_filter not in _DAY_COLUMNS:
        logging.error(
            "--day must be 'weekday', 'all' or one of %s; got %r.", _DAY_COLUMNS, day_token
        )
        return 2

    try:
        feed, calendar = _read_feed(gtfs_dir)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        logging.error("Failed to load feed '%s': %s", gtfs_dir, exc)
        return 1

    routes = _build_route_table(feed["routes"])
    n_targets = int(routes["is_target"].sum())
    if n_targets == 0:
        logging.warning("No routes matched the target criteria. Nothing to do.")
        return 1
    logging.info("Identified %d target route(s).", n_targets)

    if service_date is None and day_filter is not None:
        logging.info("Service-day filter: %s.", day_filter)
    elif service_date is None:
        logging.info("No service-day filter: pooling all trips regardless of service day.")
    try:
        keep_service_ids, service_dates = select_services(
            calendar, feed.get("calendar_dates"), day_filter, service_date
        )
    except ValueError as exc:
        logging.error("%s", exc)
        return 1

    stops_gdf = _project_stops(feed["stops"])
    events = _build_stop_events(
        feed["stop_times"], feed["trips"], keep_service_ids, feed.get("frequencies")
    )
    transfer_rules = build_transfer_rules(feed.get("transfers"), feed["stops"])

    radius_m = distance_to_meters(MAX_TRANSFER_DISTANCE, MAX_TRANSFER_DISTANCE_UNIT)
    logging.info(
        "Space buffer: %.1f m. Time check: %s.",
        radius_m,
        "on" if ENABLE_TIME_CHECK else "off",
    )

    results = compute_transfers(stops_gdf, events, routes, radius_m, service_dates, transfer_rules)
    summary, detail = build_output_tables(results, routes)

    os.makedirs(output_dir, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    logging.info("Wrote summary for %d target route(s) -> %s", len(summary), summary_path)

    if WRITE_DETAIL:
        detail_path = output_dir / DETAIL_FILENAME
        detail.to_csv(detail_path, index=False)
        logging.info("Wrote %d transfer pair(s) -> %s", len(detail), detail_path)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
