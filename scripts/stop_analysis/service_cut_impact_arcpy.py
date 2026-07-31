"""Assess stop-level service impacts of cutting bus routes, trips, or stops, by scenario.

ArcPy port of ``service_cut_impact_gpd.py`` for environments running ArcGIS
Pro's bundled Python (arcpy + pandas + numpy, no geopandas/shapely). Every
service cut — eliminating whole routes, dropping individual trips (by id or by
route/direction/departure window), or removing stops from a route (truncation) —
reduces to removing rows from ``stop_times``. This script exploits that: each
configured scenario is resolved to a removal mask over the baseline feed's stop
events, and every impact metric is a before/after comparison of products derived
from the surviving rows. Scenarios are evaluated independently against the same
baseline, so alternatives (drop route 101 *or* drop 202 and 303) can be compared
in one run.

Metrics are computed for a single representative service day (default: weekday),
chosen from the feed's real calendar the same way ``headway_span_exporter.py``
does, so holiday and weekend services cannot leak into weekday counts. Coverage
uses straight-line stop buffers — no pedestrian network — and optional
demographic and trip-level ridership inputs upgrade the impact accounting from
areas to people.

Inputs:
    - A GTFS feed (folder or .zip): stops, routes, trips, stop_times, plus calendar /
      calendar_dates and shapes when present.
    - Optional demographics polygon layer (any feature class arcpy can read —
      shapefile, file geodatabase, ...) with numeric fields to apportion over
      lost coverage.
    - Optional trip-level ridership CSV of average daily boardings by ``trip_id`` (and
      optionally ``stop_id``), from APC/ridecheck processing.
    - Optional stop-level ridership CSV at route × stop grain (a vendor "stop usage"
      export) for boardings-lost estimates when trip-level data is unavailable.

Outputs:
    - ``scenario_summary.csv``: one row per scenario — trips cut/truncated, stops
      eliminated/reduced, stop×time-bin service lost, revenue hours/miles cut, coverage
      area lost, demographics lost, and riders affected. In screening mode
      (``SCENARIOS = "each_route"`` or ``--each-route``) this is the only table:
      every route's elimination is evaluated and ranked by what is at stake.
    - ``<scenario>_stop_impacts.csv``: per impacted stop — baseline vs scenario trips,
      span, largest gap, time bins lost, nearest remaining stop, boardings lost.
    - ``<scenario>_stop_time_bins.csv``: long-format stop × time-bin trip counts,
      baseline vs scenario.
    - ``<scenario>_lost_coverage.shp``: polygon of walk-buffer coverage lost.
    - Map layers per scenario (``EXPORT_MAP_LAYERS``): ``_remaining_lines.shp``
      and ``_removed_lines.shp`` (route alignments still run / no longer run by
      any trip, from shapes.txt) and ``_remaining_coverage.shp`` (walk buffer of
      the surviving network).
    - ``<scenario>_gtfs/`` per scenario (``EXPORT_SCENARIO_GTFS``): the surviving
      network as a standalone GTFS feed (cuts applied across all service days),
      so any other tool can run baseline-vs-scenario comparisons.
    - A run-log sidecar capturing the verbatim CONFIGURATION block.

Limitations:
    Coverage is straight-line buffer, not a walkshed. Revenue hours/miles are GTFS
    revenue-service approximations — deadhead, pull-ins/outs, and layover need scheduled
    operations data (TODS) to be accurate, and truncation mileage changes are not
    estimated. ``frequencies.txt``-based feeds are not expanded. Access-quality metrics
    (replacement direction match, added wait) and a Title VI equity screen are future
    work.

Requires:
    ArcGIS Pro (arcpy) plus pandas and numpy, all bundled with Pro. Outputs land
    in a filesystem folder (shapefiles + CSVs), not a geodatabase.

Typical usage:
    Update the paths and SCENARIOS in the CONFIGURATION section (or pass the matching
    CLI flags) and run from a shell, ArcGIS Pro's Python window, or a Jupyter notebook
    using ArcGIS Pro's bundled Python.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

import arcpy
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

# GTFS feed: a folder of .txt files or a .zip archive of one.
GTFS_DIR: str = r"Path\To\Your\GTFS_Folder"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Scenarios are evaluated independently against the same baseline feed. Set the
# whole variable to the string "each_route" instead of a list for SCREENING MODE:
# one route-elimination scenario is generated per route serving the analysis day,
# per-scenario detail files are skipped, and the summary becomes a route exposure
# ranking (sorted by riders at stake, else revenue hours). For an explicit list,
# every key except "name" is optional, and any combination is unioned into one cut:
#   routes       - drop every trip on these routes (route_short_name or route_id).
#   trip_ids     - drop these exact trips.txt trip_ids.
#   trip_windows - drop trips by first departure, e.g. {"route": "101",
#                  "direction_id": "0", "depart_after": "21:00", "depart_before":
#                  "27:00"}. Bounds are inclusive HH:MM[:SS]; either bound and the
#                  route/direction scope may be omitted. Times past midnight (25:30)
#                  follow the GTFS convention.
#   stops        - drop these stop_ids from every trip that visits them.
#   route_stops  - {"route token": [stop_ids]}: drop the stops only from that
#                  route's trips (truncation / short-turn).
SCENARIOS: "Sequence[Mapping[str, Any]] | str" = (
    {
        "name": "cut_route_101",
        "description": "Eliminate route 101 entirely.",
        "routes": ["101"],
    },
    {
        "name": "cut_202_and_303",
        "description": "Eliminate routes 202 and 303 together.",
        "routes": ["202", "303"],
    },
)

# Day the analysis represents: "weekday" or "monday".."sunday". A representative
# date of that day type is chosen from the feed's real calendar (skipping observed
# federal holidays); SERVICE_DATE ("YYYY-MM-DD") pins one explicit date instead.
SERVICE_DAY: str = "weekday"
SERVICE_DATE: str = ""

# Width of the time-of-day bins used for the stop × time-bin service matrix.
TIME_BIN_MINUTES: int = 60

# Walk-access buffer radius around stops for the coverage comparison.
STOP_BUFFER_MILES: float = 0.25

# Projected spatial reference (WKID) used for buffering, distances, and areas.
# 2248 = NAD83 / Maryland State Plane (US feet), the DC-area default. Any
# projected WKID works — its linear unit is read from the spatial reference, so
# no unit constant has to be kept in sync by hand.
PROJECTED_CRS_WKID: int = 2248

# Keep only platform stops (location_type 0 or blank) in stop-level outputs.
FILTER_TO_PLATFORM_STOPS: bool = True

# Also export per-scenario mapping layers: remaining route lines, removed route
# lines (shapes no surviving trip uses; needs shapes.txt), and the remaining
# network's walk-buffer coverage. Line layers reflect whole-trip cuts —
# truncations do not redraw alignments.
EXPORT_MAP_LAYERS: bool = True

# Also write each scenario's surviving network as a standalone GTFS feed folder
# (<name>_gtfs/), so any other script in this repo can run baseline-vs-scenario
# comparisons on ordinary feeds. Cuts are applied across ALL service days —
# not just the analysis day — so the exported feed is the scenario network.
EXPORT_SCENARIO_GTFS: bool = False

# Optional demographics polygon layer; numeric DEMOGRAPHIC_FIELDS are apportioned
# onto lost coverage by area weighting. Leave the path "" to skip.
DEMOGRAPHICS_PATH: str = r""
DEMOGRAPHIC_FIELDS: Sequence[str] = ("population",)

# Optional trip-level ridership CSV of average daily boardings. Rows carry a
# trip_id, an optional stop_id (blank/absent = whole-trip total), and a numeric
# boardings column. Leave the path "" to skip ridership accounting.
RIDERSHIP_CSV: str = r""
RIDERSHIP_TRIP_ID_COL: str = "trip_id"
RIDERSHIP_STOP_ID_COL: str = "stop_id"
RIDERSHIP_BOARDINGS_COL: str = "avg_daily_boardings"

# Optional stop-level ridership CSV at route × stop grain (a vendor "stop usage"
# export: one row per route/direction/stop with average daily boardings). Route
# values match route_short_name or route_id. Enables boardings-lost estimates —
# exact for stops/route-pairs losing all service, prorated by trip share for
# partial cuts — without needing trip-level data. Leave the path "" to skip.
STOP_RIDERSHIP_CSV: str = r""
STOP_RIDERSHIP_ROUTE_COL: str = "ROUTE_NUMBER"
STOP_RIDERSHIP_STOP_ID_COL: str = "STOP_ID"
STOP_RIDERSHIP_BOARDINGS_COL: str = "XBOARDINGS"

# Cross-scenario summary filename (per-scenario files are prefixed by scenario name).
SUMMARY_FILENAME: str = r"scenario_summary.csv"

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Scenario keys accepted by resolve_drop_mask (anything else is a config error).
_SCENARIO_KEYS: frozenset[str] = frozenset(
    {"name", "description", "routes", "trip_ids", "trip_windows", "stops", "route_stops"}
)

# Stops buffered per arcpy Multipoint before the chunk buffers are unioned. One
# huge multipoint is slower to buffer than a handful of moderate ones, and the
# chunk unions keep peak geometry size bounded on large feeds.
_BUFFER_CHUNK_STOPS: int = 500

# Rows compared per numpy distance block. Distance work is done as chunked
# broadcasting rather than a spatial index so the script needs nothing beyond
# the numpy that ships with ArcGIS Pro.
_DISTANCE_CHUNK: int = 256

# Longest string a shapefile text field can hold (dBASE limit).
_SHP_TEXT_MAX: int = 254

# Excel workbook suffixes accepted for the optional ridership tables. Legacy
# .xls is read with xlrd, the .xlsx family with openpyxl — both ship with
# ArcGIS Pro.
_EXCEL_SUFFIXES: tuple[str, ...] = (".xls", ".xlsx", ".xlsm", ".xltx", ".xltm")


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


def convert_distance(
    value: Any,
    input_unit: str,
    output_unit: Literal["miles", "km"] = "miles",
) -> Optional[float]:
    """Convert a distance value between transit-planning units.

    Args:
        value: Distance as a number or numeric string. ``None``, NaN, and
            empty/whitespace strings yield ``None``.
        input_unit: Unit of *value*: ``"feet"``, ``"meters"``, ``"km"``, or
            ``"miles"`` (case-insensitive).
        output_unit: Unit to convert to: ``"miles"`` or ``"km"``.

    Returns:
        The converted distance as a float, or ``None`` when *value* is
        missing or cannot be interpreted as a number.

    Raises:
        ValueError: If *input_unit* or *output_unit* is not a supported unit.
    """
    meters_per_input_unit = {"feet": 0.3048, "meters": 1.0, "km": 1000.0, "miles": 1609.344}
    meters_per_output_unit = {"miles": 1609.344, "km": 1000.0}

    input_factor = meters_per_input_unit.get(str(input_unit).strip().lower())
    if input_factor is None:
        raise ValueError(
            f"Unsupported input_unit {input_unit!r}; "
            f"expected one of {sorted(meters_per_input_unit)}."
        )
    output_factor = meters_per_output_unit.get(str(output_unit).strip().lower())
    if output_factor is None:
        raise ValueError(
            f"Unsupported output_unit {output_unit!r}; "
            f"expected one of {sorted(meters_per_output_unit)}."
        )

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric * input_factor / output_factor


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


class Config(NamedTuple):
    """Runtime configuration for a service-cut impact run."""

    gtfs_dir: str
    output_dir: Path
    scenarios: "Sequence[Mapping[str, Any]] | str" = SCENARIOS
    service_day: str = SERVICE_DAY
    service_date: str = SERVICE_DATE
    time_bin_minutes: int = TIME_BIN_MINUTES
    stop_buffer_miles: float = STOP_BUFFER_MILES
    crs_wkid: int = PROJECTED_CRS_WKID
    filter_platform_stops: bool = FILTER_TO_PLATFORM_STOPS
    export_map_layers: bool = EXPORT_MAP_LAYERS
    export_scenario_gtfs: bool = EXPORT_SCENARIO_GTFS
    demographics_path: str = DEMOGRAPHICS_PATH
    demographic_fields: Sequence[str] = tuple(DEMOGRAPHIC_FIELDS)
    ridership_csv: str = RIDERSHIP_CSV
    stop_ridership_csv: str = STOP_RIDERSHIP_CSV


class Baseline(NamedTuple):
    """Baseline products every scenario is compared against.

    The baseline is identical for every scenario, so it is built once per run
    instead of once per scenario — the arcpy buffer union in particular is far
    too slow to repeat for each of the hundreds of scenarios screening mode
    generates.

    Attributes:
        events: The baseline stop-event table (:func:`build_events`).
        stats: Per-stop service statistics (:func:`summarize_stops`).
        bins: Per-stop × time-bin trip counts (:func:`bin_counts`).
        coverage: Walk-buffer union of every stop with baseline service
            (``None`` when no stop has usable coordinates).
        coverage_sqmi: Area of *coverage* in square miles.
    """

    events: pd.DataFrame
    stats: pd.DataFrame
    bins: pd.DataFrame
    coverage: Optional[Any]
    coverage_sqmi: float


# =============================================================================
# SMALL PRIVATE HELPERS
# =============================================================================


def _spatial_reference(wkid: int) -> Any:
    """Return the projected spatial reference for *wkid*.

    Raises:
        ValueError: If the WKID is unknown to arcpy or is not projected —
            buffering, lengths, and areas are meaningless in degrees.
    """
    unknown = f"PROJECTED_CRS_WKID {wkid!r} is not a spatial reference arcpy knows."
    try:
        sr = arcpy.SpatialReference(int(wkid))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(unknown) from exc
    if not sr.name or sr.name == "Unknown":
        raise ValueError(unknown)
    if sr.type != "Projected":
        raise ValueError(
            f"PROJECTED_CRS_WKID must reference a projected spatial reference; "
            f"got {sr.name!r} ({sr.type})."
        )
    return sr


def _miles_to_sr_units(miles: float, sr: Any) -> float:
    """Convert miles to the linear unit of the projected spatial reference."""
    meters_per_unit = float(sr.metersPerUnit or 1.0)
    return miles * 1609.344 / meters_per_unit


def _sr_units_to_miles(value: float, sr: Any) -> float:
    """Convert a length in spatial-reference units to miles."""
    meters = float(value) * float(sr.metersPerUnit or 1.0)
    miles = convert_distance(meters, input_unit="meters", output_unit="miles")
    return float("nan") if miles is None else miles


def _area_to_sqmi(area: float, sr: Any) -> float:
    """Convert an area in squared spatial-reference units to square miles."""
    per_mile = _miles_to_sr_units(1.0, sr)
    return float(area) / (per_mile * per_mile)


def _sanitize_name(name: str) -> str:
    """Reduce a scenario name to a filesystem-safe token."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return token or "scenario"


def _route_label_series(routes: pd.DataFrame) -> pd.Series:
    """Return a route_id-indexed Series of display labels (short name, else id)."""
    r = routes.drop_duplicates(subset=["route_id"]).copy()
    short = r.get("route_short_name", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    label = short.where(short.str.len() > 0, r["route_id"].astype(str))
    return pd.Series(label.to_numpy(), index=r["route_id"].astype(str))


def _as_sorted_csv(values: Sequence[str]) -> str:
    """Join unique, non-blank values into one sorted comma-separated string."""
    uniq = sorted({str(v).strip() for v in values if str(v).strip()})
    return ",".join(uniq)


def _shp_text(value: Any) -> str:
    """Trim a value to what a shapefile text field can hold."""
    text = "" if value is None else str(value)
    if len(text) > _SHP_TEXT_MAX:
        return text[:_SHP_TEXT_MAX]
    return text


# =============================================================================
# GEOMETRY HELPERS (arcpy)
# =============================================================================


def _is_empty(geometry: Optional[Any]) -> bool:
    """Return True when *geometry* is absent or encloses no area."""
    return geometry is None or float(geometry.area) <= 0.0


def _project_lonlat(
    lons: Sequence[float], lats: Sequence[float], sr: Any
) -> Tuple[List[float], List[float]]:
    """Project WGS84 coordinate pairs into *sr*.

    Args:
        lons: Longitudes in decimal degrees.
        lats: Latitudes in decimal degrees.
        sr: Target projected spatial reference.

    Returns:
        Tuple of (x values, y values) in *sr* units, aligned to the inputs.
    """
    wgs84 = arcpy.SpatialReference(4326)
    xs: List[float] = []
    ys: List[float] = []
    for lon, lat in zip(lons, lats):
        point = arcpy.PointGeometry(arcpy.Point(float(lon), float(lat)), wgs84).projectAs(sr)
        xs.append(float(point.firstPoint.X))
        ys.append(float(point.firstPoint.Y))
    return xs, ys


def _xy_array(points: pd.DataFrame) -> np.ndarray:
    """Return an (n, 2) float array of the projected ``x``/``y`` columns."""
    if points.empty:
        return np.empty((0, 2), dtype=float)
    return points[["x", "y"]].to_numpy(dtype=float)


def _nearest_neighbors(
    source_xy: np.ndarray, target_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Find each source point's nearest target point by planar distance.

    Chunked numpy broadcasting stands in for a spatial index so nothing beyond
    ArcGIS Pro's bundled numpy is required; distances match the planar nearest
    join the GeoPandas twin performs.

    Args:
        source_xy: (n, 2) array of projected query coordinates.
        target_xy: (m, 2) array of projected candidate coordinates.

    Returns:
        Tuple of (index into *target_xy* of each source point's nearest
        neighbour, distance to it in spatial-reference units). Both arrays are
        empty when either input is.
    """
    n = len(source_xy)
    if n == 0 or len(target_xy) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    idx = np.zeros(n, dtype=int)
    dist = np.zeros(n, dtype=float)
    for start in range(0, n, _DISTANCE_CHUNK):
        block = source_xy[start : start + _DISTANCE_CHUNK]
        dx = block[:, 0][:, None] - target_xy[:, 0][None, :]
        dy = block[:, 1][:, None] - target_xy[:, 1][None, :]
        squared = dx * dx + dy * dy
        nearest = squared.argmin(axis=1)
        idx[start : start + len(block)] = nearest
        dist[start : start + len(block)] = np.sqrt(squared[np.arange(len(block)), nearest])
    return idx, dist


def _within_radius_mask(target_xy: np.ndarray, source_xy: np.ndarray, radius: float) -> np.ndarray:
    """Flag the target points lying within *radius* of any source point.

    Args:
        target_xy: (m, 2) array of projected coordinates to test.
        source_xy: (n, 2) array of projected reference coordinates.
        radius: Distance threshold in spatial-reference units.

    Returns:
        Boolean array of length ``len(target_xy)``.
    """
    mask = np.zeros(len(target_xy), dtype=bool)
    if len(target_xy) == 0 or len(source_xy) == 0:
        return mask
    limit = float(radius) ** 2
    for start in range(0, len(source_xy), _DISTANCE_CHUNK):
        block = source_xy[start : start + _DISTANCE_CHUNK]
        dx = target_xy[:, 0][:, None] - block[:, 0][None, :]
        dy = target_xy[:, 1][:, None] - block[:, 1][None, :]
        mask |= ((dx * dx + dy * dy) <= limit).any(axis=1)
    return mask


def buffer_union(points: pd.DataFrame, radius: float, sr: Any) -> Optional[Any]:
    """Buffer projected points by *radius* and dissolve the result.

    Points are buffered in chunks and the chunk polygons unioned, which keeps
    each intermediate geometry small enough for arcpy to handle comfortably on
    agency-sized feeds.

    Args:
        points: Table carrying projected ``x``/``y`` columns.
        radius: Buffer radius in spatial-reference units.
        sr: Projected spatial reference of the coordinates.

    Returns:
        The dissolved buffer polygon, or ``None`` when *points* is empty.
    """
    if points.empty:
        return None
    coords = _xy_array(points)
    union: Optional[Any] = None
    for start in range(0, len(coords), _BUFFER_CHUNK_STOPS):
        block = coords[start : start + _BUFFER_CHUNK_STOPS]
        multipoint = arcpy.Multipoint(
            arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in block]), sr
        )
        chunk_buffer = multipoint.buffer(radius)
        union = chunk_buffer if union is None else union.union(chunk_buffer)
    return union


def _extents_overlap(a: Any, b: Any) -> bool:
    """Return True when two arcpy extents intersect (a cheap prefilter)."""
    return not (a.XMax < b.XMin or a.XMin > b.XMax or a.YMax < b.YMin or a.YMin > b.YMax)


def write_feature_shapefile(
    output_dir: Path,
    filename: str,
    geometry_type: str,
    sr: Any,
    fields: Sequence[Tuple[str, str, int]],
    rows: Sequence[Sequence[Any]],
) -> Path:
    """Create a shapefile and insert *rows* into it, replacing any existing file.

    Args:
        output_dir: Folder the shapefile and its sidecar files are written to.
        filename: Shapefile name including the ``.shp`` extension.
        geometry_type: arcpy geometry type, e.g. ``"POLYGON"``/``"POLYLINE"``.
        sr: Spatial reference of the geometries.
        fields: Field definitions as ``(name, arcpy field type, text length)``;
            the length is ignored for non-text fields. Shapefile field names
            are capped at 10 characters by the dBASE format.
        rows: One sequence per feature: the field values in *fields* order,
            followed by the geometry.

    Returns:
        Path of the shapefile written.
    """
    out_path = output_dir / filename
    if arcpy.Exists(str(out_path)):
        arcpy.management.Delete(str(out_path))
    arcpy.management.CreateFeatureclass(
        str(output_dir), filename, geometry_type, spatial_reference=sr
    )
    for name, field_type, length in fields:
        if field_type == "TEXT":
            arcpy.management.AddField(str(out_path), name, field_type, field_length=length)
        else:
            arcpy.management.AddField(str(out_path), name, field_type)

    cursor_fields = [name for name, _, _ in fields] + ["SHAPE@"]
    with arcpy.da.InsertCursor(str(out_path), cursor_fields) as cursor:
        for row in rows:
            cursor.insertRow(tuple(row))
    return out_path


# =============================================================================
# LOADING & BASELINE
# =============================================================================


def load_feed(gtfs_dir: str) -> dict[str, Optional[pd.DataFrame]]:
    """Load the required and optional GTFS tables for the analysis.

    Args:
        gtfs_dir: GTFS folder or .zip path.

    Returns:
        Mapping with keys ``stops``, ``routes``, ``trips``, ``stop_times`` (always
        present) and ``calendar``, ``calendar_dates``, ``shapes``, ``frequencies``
        (``None`` when the feed lacks the file).
    """
    required = load_gtfs_data(
        gtfs_dir, files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
    )
    feed: dict[str, Optional[pd.DataFrame]] = dict(required)
    for optional in ("calendar", "calendar_dates", "shapes", "frequencies"):
        try:
            feed[optional] = load_gtfs_data(gtfs_dir, files=(f"{optional}.txt",))[optional]
        except (OSError, ValueError):
            feed[optional] = None
    freq = feed.get("frequencies")
    if freq is not None and not freq.empty:
        logging.warning(
            "frequencies.txt is present (%d rows) but is NOT expanded — trip counts, "
            "headways, and revenue hours will undercount frequency-based service.",
            len(freq),
        )
    return feed


def select_service_ids(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    service_day: str,
    service_date: str,
) -> Optional[set[str]]:
    """Choose the service_ids of the representative analysis day.

    Args:
        calendar: Parsed ``calendar.txt`` or ``None``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.
        service_day: ``"weekday"`` or a single day name.
        service_date: Optional explicit ``YYYY-MM-DD`` date; overrides selection.

    Returns:
        The set of active service_ids, or ``None`` when the feed has no usable
        calendar information (callers should then keep all trips).

    Raises:
        ValueError: If *service_date* is set but not parseable as ``YYYY-MM-DD``.
    """
    if (calendar is None or calendar.empty) and (calendar_dates is None or calendar_dates.empty):
        logging.warning(
            "Feed has no calendar.txt or calendar_dates.txt — keeping all trips; "
            "metrics will pool every service day together."
        )
        return None

    active = expand_service_active_dates(calendar, calendar_dates)
    if not any(active.values()):
        logging.warning("No service has any active dates — keeping all trips.")
        return None

    override: Optional[dt.date] = None
    if service_date.strip():
        try:
            override = dt.datetime.strptime(service_date.strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"SERVICE_DATE must be YYYY-MM-DD; got {service_date!r}.") from exc

    all_dates = {d for dates in active.values() for d in dates}
    years = range(min(all_dates).year, max(all_dates).year + 2)
    holidays: set[dt.date] = set()
    for year in years:
        holidays |= federal_holidays_observed(year)

    _, service_ids = representative_service_date(
        active, service_day, override_date=override, exclude_dates=holidays
    )
    return service_ids


def build_events(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    service_ids: Optional[set[str]],
) -> pd.DataFrame:
    """Build the baseline stop-event table every metric derives from.

    One row per (trip, stop visit) on the analysis day, with a parsed departure
    minute. Rows whose departure and arrival times both fail to parse are dropped
    with a warning — they cannot participate in time-based metrics.

    Args:
        stop_times: Parsed ``stop_times.txt``.
        trips: Parsed ``trips.txt``.
        routes: Parsed ``routes.txt``.
        service_ids: Service filter from :func:`select_service_ids` (``None`` keeps
            all trips).

    Returns:
        DataFrame with columns ``trip_id``, ``stop_id``, ``dep_min``, ``route_id``,
        ``route_label``, ``direction_id``.

    Raises:
        ValueError: If no trips remain after the service-day filter.
    """
    t = trips.drop_duplicates(subset=["trip_id"]).copy()
    t["trip_id"] = t["trip_id"].astype(str)
    t["route_id"] = t["route_id"].astype(str)
    if service_ids is not None:
        t = t[t["service_id"].astype(str).isin(service_ids)].copy()
        if t.empty:
            raise ValueError(
                "No trips remain after filtering to the analysis day's service_ids. "
                "Check SERVICE_DAY / SERVICE_DATE against the feed's calendar."
            )

    st = stop_times[["trip_id", "stop_id", "departure_time", "arrival_time"]].copy()
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_id"] = st["stop_id"].astype(str)
    events = st.merge(
        t[["trip_id", "route_id", "direction_id"]]
        if "direction_id" in t.columns
        else t[["trip_id", "route_id"]],
        on="trip_id",
        how="inner",
        validate="many_to_one",
    )
    if "direction_id" not in events.columns:
        events["direction_id"] = ""
    events["direction_id"] = events["direction_id"].fillna("").astype(str)

    time_source = events["departure_time"].where(
        events["departure_time"].notna(), events["arrival_time"]
    )
    events["dep_min"] = time_source.map(parse_time_to_minutes)
    bad = int(events["dep_min"].isna().sum())
    if bad:
        logging.warning(
            "Dropped %d stop_times row(s) with unparseable departure/arrival times.", bad
        )
        events = events[events["dep_min"].notna()].copy()
    events["dep_min"] = events["dep_min"].astype(int)

    labels = _route_label_series(routes)
    events["route_label"] = events["route_id"].map(labels).fillna(events["route_id"])
    events = events.drop(columns=["departure_time", "arrival_time"])
    logging.info(
        "Baseline events: %d stop visits, %d trips, %d stops, %d routes.",
        len(events),
        events["trip_id"].nunique(),
        events["stop_id"].nunique(),
        events["route_id"].nunique(),
    )
    return events.reset_index(drop=True)


def prepare_stops_table(stops: pd.DataFrame, filter_platform_stops: bool, sr: Any) -> pd.DataFrame:
    """Project stops into the analysis spatial reference, keeping platforms by default.

    Args:
        stops: Parsed ``stops.txt``.
        filter_platform_stops: Keep only ``location_type`` 0/blank rows.
        sr: Projected spatial reference of the analysis.

    Returns:
        DataFrame with ``stop_id``, ``stop_name``, ``stop_lat``, ``stop_lon`` and
        the projected ``x``/``y`` coordinates.
    """
    s = stops.drop_duplicates(subset=["stop_id"]).copy()
    s["stop_id"] = s["stop_id"].astype(str)
    if "stop_name" not in s.columns:
        s["stop_name"] = ""
    if filter_platform_stops and "location_type" in s.columns:
        loc = s["location_type"].fillna("").astype(str).str.strip()
        s = s[loc.isin(["", "0"])].copy()
    s["stop_lat"] = pd.to_numeric(s["stop_lat"], errors="coerce")
    s["stop_lon"] = pd.to_numeric(s["stop_lon"], errors="coerce")
    bad = int(s["stop_lat"].isna().sum() + s["stop_lon"].isna().sum())
    if bad:
        logging.warning("Dropped stop rows with unparseable coordinates (%d bad values).", bad)
    s = s[s["stop_lat"].notna() & s["stop_lon"].notna()].copy()

    out = s[["stop_id", "stop_name", "stop_lat", "stop_lon"]].reset_index(drop=True)
    xs, ys = _project_lonlat(out["stop_lon"].tolist(), out["stop_lat"].tolist(), sr)
    out["x"] = xs
    out["y"] = ys
    logging.info("Projected %d stop(s) to %s.", len(out), sr.name)
    return out


# =============================================================================
# SCENARIO RESOLUTION
# =============================================================================


def _resolve_route_ids(routes: pd.DataFrame, tokens: Sequence[str]) -> set[str]:
    """Resolve route tokens (short names or ids) to route_ids, warning on misses."""
    r = routes.copy()
    r["route_id"] = r["route_id"].astype(str).str.strip()
    short = r.get("route_short_name", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    wanted = {str(tok).strip() for tok in tokens}
    mask = r["route_id"].isin(wanted) | short.isin(wanted)
    found_ids = set(r.loc[mask, "route_id"])
    matched_tokens = set(r.loc[mask, "route_id"]) | set(short[mask])
    missing = sorted(wanted - matched_tokens)
    if missing:
        logging.warning(
            "Route token(s) not found in routes.txt (route_short_name or route_id): %s",
            ", ".join(missing),
        )
    return found_ids


def resolve_drop_mask(
    scenario: Mapping[str, Any], events: pd.DataFrame, routes: pd.DataFrame
) -> pd.Series:
    """Resolve one scenario definition into a boolean drop mask over *events*.

    Args:
        scenario: One SCENARIOS entry (see the CONFIGURATION block for keys).
        events: Baseline event table from :func:`build_events`.
        routes: Parsed ``routes.txt`` (for route-token resolution).

    Returns:
        Boolean Series aligned to ``events.index``; ``True`` marks rows the
        scenario removes.

    Raises:
        ValueError: On unknown scenario keys or a trip_windows entry with no
            usable criteria.
    """
    unknown = set(scenario) - _SCENARIO_KEYS
    if unknown:
        raise ValueError(
            f"Scenario {scenario.get('name', '?')!r} has unknown key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(_SCENARIO_KEYS)}."
        )

    mask = pd.Series(False, index=events.index)

    route_tokens = scenario.get("routes") or ()
    if route_tokens:
        route_ids = _resolve_route_ids(routes, route_tokens)
        mask |= events["route_id"].isin(route_ids)

    trip_ids = {str(t).strip() for t in (scenario.get("trip_ids") or ())}
    if trip_ids:
        present = set(events.loc[events["trip_id"].isin(trip_ids), "trip_id"])
        missing = sorted(trip_ids - present)
        if missing:
            logging.warning("trip_id(s) not found on the analysis day: %s", ", ".join(missing[:20]))
        mask |= events["trip_id"].isin(trip_ids)

    windows = scenario.get("trip_windows") or ()
    if windows:
        first_dep = events.groupby("trip_id")["dep_min"].min()
        trip_route = events.drop_duplicates("trip_id").set_index("trip_id")
        for window in windows:
            selected = pd.Series(True, index=first_dep.index)
            route_token = str(window.get("route", "")).strip()
            if route_token:
                ids = _resolve_route_ids(routes, [route_token])
                selected &= trip_route["route_id"].reindex(first_dep.index).isin(ids)
            direction = str(window.get("direction_id", "")).strip()
            if direction:
                selected &= (
                    trip_route["direction_id"].reindex(first_dep.index).astype(str) == direction
                )
            after = parse_time_to_minutes(window.get("depart_after"))
            before = parse_time_to_minutes(window.get("depart_before"))
            if after is None and before is None and not route_token and not direction:
                raise ValueError(
                    f"trip_windows entry {window!r} has no usable criteria — set at least "
                    "one of route, direction_id, depart_after, depart_before."
                )
            if after is not None:
                selected &= first_dep >= after
            if before is not None:
                selected &= first_dep <= before
            window_trips = set(selected[selected].index)
            if not window_trips:
                logging.warning("trip_windows entry %r matched no trips.", window)
            mask |= events["trip_id"].isin(window_trips)

    stop_ids = {str(s).strip() for s in (scenario.get("stops") or ())}
    if stop_ids:
        present = set(events.loc[events["stop_id"].isin(stop_ids), "stop_id"])
        missing = sorted(stop_ids - present)
        if missing:
            logging.warning("stop_id(s) not served on the analysis day: %s", ", ".join(missing))
        mask |= events["stop_id"].isin(stop_ids)

    route_stops = scenario.get("route_stops") or {}
    for route_token, stops_for_route in route_stops.items():
        ids = _resolve_route_ids(routes, [route_token])
        wanted_stops = {str(s).strip() for s in stops_for_route}
        scoped = events["route_id"].isin(ids) & events["stop_id"].isin(wanted_stops)
        if not bool(scoped.any()):
            logging.warning(
                "route_stops entry %r -> %s matched no events.", route_token, sorted(wanted_stops)
            )
        mask |= scoped

    return mask


def build_screening_scenarios(events: pd.DataFrame) -> "list[dict[str, Any]]":
    """Generate one route-elimination scenario per route serving the analysis day.

    Screening mode's scenario generator: the whole-route cut is the upper bound
    on what is at stake for each route, so evaluating every route's elimination
    in one run turns the summary table into a route exposure ranking.

    Args:
        events: Baseline event table.

    Returns:
        One scenario dict per route, targeted by ``route_id`` (exact), named
        after the route label with the route_id appended when labels collide.
    """
    label_by_id = (
        events.drop_duplicates(subset=["route_id"])
        .set_index("route_id")["route_label"]
        .astype(str)
        .to_dict()
    )
    ordered = sorted(label_by_id, key=lambda rid: (label_by_id[rid], rid))
    base_names = {rid: f"cut_route_{_sanitize_name(label_by_id[rid])}" for rid in ordered}
    counts: dict[str, int] = {}
    for name in base_names.values():
        counts[name] = counts.get(name, 0) + 1

    scenarios: list[dict[str, Any]] = []
    for rid in ordered:
        name = base_names[rid]
        if counts[name] > 1:
            name = f"{name}_{_sanitize_name(rid)}"
        scenarios.append(
            {
                "name": name,
                "description": f"Screening: eliminate route {label_by_id[rid]} entirely.",
                "routes": [rid],
            }
        )
    return scenarios


# =============================================================================
# METRICS
# =============================================================================


def summarize_stops(events: pd.DataFrame) -> pd.DataFrame:
    """Reduce an event table to one row of service statistics per stop.

    Args:
        events: Event table (baseline or scenario-surviving rows).

    Returns:
        DataFrame indexed by ``stop_id`` with ``trips``, ``first_dep_min``,
        ``last_dep_min``, ``max_gap_min`` (largest gap between consecutive
        departures; NaN with fewer than two), and ``routes`` (label CSV).
    """

    def _max_gap(dep_minutes: pd.Series) -> float:
        uniq = sorted(set(dep_minutes.tolist()))
        if len(uniq) < 2:
            return float("nan")
        return float(max(b - a for a, b in zip(uniq, uniq[1:])))

    grouped = events.groupby("stop_id")
    out = pd.DataFrame(
        {
            "trips": grouped["trip_id"].nunique(),
            "first_dep_min": grouped["dep_min"].min(),
            "last_dep_min": grouped["dep_min"].max(),
            "max_gap_min": grouped["dep_min"].apply(_max_gap),
            "routes": grouped["route_label"].apply(lambda s: _as_sorted_csv(s.tolist())),
        }
    )
    return out


def bin_counts(events: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    """Count distinct trips serving each stop in each time-of-day bin.

    Args:
        events: Event table (baseline or scenario-surviving rows).
        bin_minutes: Bin width in minutes (e.g. 60).

    Returns:
        DataFrame with columns ``stop_id``, ``bin_start_min``, ``trips``.
    """
    binned = events.copy()
    binned["bin_start_min"] = (binned["dep_min"] // bin_minutes) * bin_minutes
    counts = (
        binned.groupby(["stop_id", "bin_start_min"])["trip_id"].nunique().reset_index(name="trips")
    )
    return counts


def compute_stop_impacts(
    base_stats: pd.DataFrame,
    scen_stats: pd.DataFrame,
    base_bins: pd.DataFrame,
    scen_bins: pd.DataFrame,
    stops: pd.DataFrame,
) -> pd.DataFrame:
    """Compare baseline and scenario stop statistics and classify each stop.

    Args:
        base_stats: :func:`summarize_stops` of the baseline events.
        scen_stats: :func:`summarize_stops` of the surviving events.
        base_bins: :func:`bin_counts` of the baseline events.
        scen_bins: :func:`bin_counts` of the surviving events.
        stops: Projected stops table (for names and coordinates).

    Returns:
        One row per stop with baseline service, sorted with eliminated stops
        first; ``classification`` is ``eliminated`` / ``reduced`` / ``unchanged``.
    """
    merged = base_stats.add_suffix("_baseline").join(scen_stats.add_suffix("_scenario"))
    merged["trips_scenario"] = merged["trips_scenario"].fillna(0).astype(int)
    merged["trips_delta"] = merged["trips_scenario"] - merged["trips_baseline"]
    merged["classification"] = "unchanged"
    merged.loc[merged["trips_delta"] < 0, "classification"] = "reduced"
    merged.loc[merged["trips_scenario"] == 0, "classification"] = "eliminated"

    bins = base_bins.merge(
        scen_bins, on=["stop_id", "bin_start_min"], how="left", suffixes=("_base", "_scen")
    )
    bins["trips_scen"] = bins["trips_scen"].fillna(0)
    lost = bins[bins["trips_scen"] == 0]
    lost_by_stop = lost.groupby("stop_id")["bin_start_min"].agg(
        time_bins_lost="count",
        time_bins_lost_list=lambda s: ",".join(minutes_to_hhmm(m) for m in sorted(s)),
    )
    merged = merged.join(lost_by_stop)
    merged["time_bins_lost"] = merged["time_bins_lost"].fillna(0).astype(int)
    merged["time_bins_lost_list"] = merged["time_bins_lost_list"].fillna("")

    for col in ("first_dep", "last_dep"):
        for side in ("baseline", "scenario"):
            merged[f"{col}_{side}"] = merged[f"{col}_min_{side}"].map(minutes_to_hhmm)
        merged = merged.drop(columns=[f"{col}_min_baseline", f"{col}_min_scenario"])

    lookup = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].set_index("stop_id")
    merged = merged.join(lookup, how="left")
    merged["routes_scenario"] = merged["routes_scenario"].fillna("")
    order = pd.CategoricalDtype(["eliminated", "reduced", "unchanged"], ordered=True)
    merged["classification"] = merged["classification"].astype(order)
    return merged.sort_values(["classification", "trips_delta", "stop_id"]).reset_index()


def add_nearest_remaining(impacts: pd.DataFrame, stops: pd.DataFrame, sr: Any) -> pd.DataFrame:
    """Attach each eliminated stop's nearest stop that keeps service.

    Args:
        impacts: Output of :func:`compute_stop_impacts`.
        stops: Projected stops table.
        sr: Projected spatial reference (for the distance conversion).

    Returns:
        *impacts* with ``nearest_remaining_stop_id`` and
        ``nearest_remaining_stop_miles`` filled for eliminated stops.
    """
    impacts = impacts.copy()
    impacts["nearest_remaining_stop_id"] = ""
    impacts["nearest_remaining_stop_miles"] = float("nan")

    eliminated_ids = set(impacts.loc[impacts["classification"] == "eliminated", "stop_id"])
    remaining_ids = set(impacts.loc[impacts["classification"] != "eliminated", "stop_id"])
    eliminated = stops[stops["stop_id"].isin(eliminated_ids)].reset_index(drop=True)
    remaining = stops[stops["stop_id"].isin(remaining_ids)].reset_index(drop=True)
    if eliminated.empty or remaining.empty:
        return impacts

    idx, dist = _nearest_neighbors(_xy_array(eliminated), _xy_array(remaining))
    near_id = pd.Series(
        remaining["stop_id"].to_numpy()[idx], index=eliminated["stop_id"].to_numpy()
    )
    near_miles = pd.Series(
        [_sr_units_to_miles(d, sr) for d in dist], index=eliminated["stop_id"].to_numpy()
    )

    is_elim = impacts["classification"] == "eliminated"
    impacts.loc[is_elim, "nearest_remaining_stop_id"] = (
        impacts.loc[is_elim, "stop_id"].map(near_id).fillna("")
    )
    impacts.loc[is_elim, "nearest_remaining_stop_miles"] = pd.to_numeric(
        impacts.loc[is_elim, "stop_id"].map(near_miles), errors="coerce"
    ).round(3)
    return impacts


def coverage_stats(
    base_stop_ids: Set[str],
    scen_stop_ids: Set[str],
    stops: pd.DataFrame,
    buffer_miles: float,
    sr: Any,
    base_union: Optional[Any],
    base_sqmi: float,
    include_remaining: bool = True,
) -> Tuple[float, float, Optional[Any], Optional[Any]]:
    """Compare walk-buffer coverage of served stops before and after the cut.

    The scenario's buffer union is a subset of the baseline's, so the lost area
    is computed from the eliminated stops alone: only surviving stops within two
    buffer radii of an eliminated stop can cover any of its buffer, and the
    remaining coverage is the baseline minus what was lost. That is identical to
    the twin's ``baseline.difference(scenario)`` while buffering a handful of
    stops per scenario instead of the whole network.

    Args:
        base_stop_ids: Stops with baseline service.
        scen_stop_ids: Stops with surviving service.
        stops: Projected stops table.
        buffer_miles: Walk-access buffer radius in miles.
        sr: Projected spatial reference.
        base_union: Precomputed baseline buffer union (see :class:`Baseline`).
        base_sqmi: Area of *base_union* in square miles.
        include_remaining: Compute the remaining-coverage polygon. Set False
            when no map layer needs it (screening mode) to skip the overlay.

    Returns:
        Tuple of (baseline area sq mi, scenario area sq mi, lost-coverage
        geometry, remaining-coverage geometry) — geometries in the analysis
        spatial reference, ``None`` when they enclose no area.
    """
    eliminated_ids = set(base_stop_ids) - set(scen_stop_ids)
    eliminated = stops[stops["stop_id"].isin(eliminated_ids)]
    if base_union is None or eliminated.empty:
        return base_sqmi, base_sqmi, None, base_union if include_remaining else None

    radius = _miles_to_sr_units(buffer_miles, sr)
    lost = buffer_union(eliminated, radius, sr)
    if lost is None:
        return base_sqmi, base_sqmi, None, base_union if include_remaining else None

    surviving = stops[stops["stop_id"].isin(set(scen_stop_ids))]
    nearby = surviving[
        _within_radius_mask(_xy_array(surviving), _xy_array(eliminated), 2.0 * radius)
    ]
    nearby_union = buffer_union(nearby, radius, sr)
    if nearby_union is not None:
        lost = lost.difference(nearby_union)

    if _is_empty(lost):
        return base_sqmi, base_sqmi, None, base_union if include_remaining else None

    lost_sqmi = _area_to_sqmi(lost.area, sr)
    remaining = base_union.difference(lost) if include_remaining else None
    return base_sqmi, base_sqmi - lost_sqmi, lost, remaining


def load_demographics(path: str, fields: Sequence[str], sr: Any) -> List[Dict[str, Any]]:
    """Read the demographics polygons, projected into the analysis reference.

    Args:
        path: Feature class or shapefile path (any layer arcpy can read).
        fields: Numeric field names to carry through for apportionment.
        sr: Projected spatial reference the geometries are projected into.

    Returns:
        One record per polygon with its ``geometry``, ``extent``, projected
        ``area``, and the requested field ``values``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If arcpy cannot read the layer's fields or rows — most often
            a shapefile whose .dbf declares the wrong codepage.
    """
    if not arcpy.Exists(path):
        raise FileNotFoundError(f"Demographics layer not found: {path}")

    logging.info("Reading demographics layer: %s", path)
    try:
        available = {f.name.lower(): f.name for f in arcpy.ListFields(path)}
    except (RuntimeError, UnicodeDecodeError, arcpy.ExecuteError) as exc:
        raise ValueError(f"Could not list the fields of demographics layer {path}: {exc}") from exc
    resolved: List[Tuple[str, str]] = []
    for name in fields:
        actual = available.get(str(name).strip().lower())
        if actual is None:
            logging.warning("Demographic field %r not found in the demographics layer.", name)
        else:
            resolved.append((str(name), actual))

    cursor_fields = ["SHAPE@"] + [actual for _, actual in resolved]
    records: List[Dict[str, Any]] = []
    try:
        with arcpy.da.SearchCursor(path, cursor_fields, spatial_reference=sr) as cursor:
            for row in cursor:
                geometry = row[0]
                if geometry is None:
                    continue
                values: Dict[str, float] = {}
                for offset, (name, _) in enumerate(resolved, start=1):
                    number = pd.to_numeric(row[offset], errors="coerce")
                    values[name] = 0.0 if pd.isna(number) else float(number)
                records.append(
                    {
                        "geometry": geometry,
                        "extent": geometry.extent,
                        "area": float(geometry.area),
                        "values": values,
                    }
                )
    except (RuntimeError, UnicodeDecodeError, arcpy.ExecuteError) as exc:
        raise ValueError(f"Could not read demographics layer {path}: {exc}") from exc
    logging.info(
        "Demographics: %d polygon(s); fields: %s",
        len(records),
        ", ".join(name for name, _ in resolved) or "(none resolved)",
    )
    return records


def apportion_demographics(
    lost_geom: Optional[Any],
    demographics: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, float]:
    """Area-weight demographic fields onto the lost-coverage polygon.

    Assumes each field is uniformly distributed within its polygon — the standard
    area-apportionment simplification.

    Args:
        lost_geom: Lost-coverage geometry (analysis CRS, may be ``None``).
        demographics: Records from :func:`load_demographics`.
        fields: Numeric field names to apportion.

    Returns:
        Mapping of ``<field>_lost`` to the apportioned total (0.0 when the lost
        geometry is empty).
    """
    out = {f"{name}_lost": 0.0 for name in fields}
    if _is_empty(lost_geom) or not demographics:
        return out

    lost_extent = lost_geom.extent
    for record in demographics:
        if record["area"] <= 0 or not _extents_overlap(record["extent"], lost_extent):
            continue
        clipped = record["geometry"].intersect(lost_geom, 4)  # 4 = polygon dimension
        if clipped is None:
            continue
        frac = float(clipped.area) / record["area"]
        if frac <= 0:
            continue
        for name in fields:
            value = record["values"].get(name)
            if value:
                out[f"{name}_lost"] += value * frac
    return out


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


def _read_optional_table(path: Path, label: str) -> pd.DataFrame:
    """Read an optional ridership table as strings, whatever Windows produced.

    The format is decided by file *content* first — Excel's two container
    signatures (the OLE2 header of legacy ``.xls``, the zip header of
    ``.xlsx``) — and by extension second, so a workbook saved with a ``.csv``
    name is still read as Excel instead of failing with a cryptic ``utf-8
    codec`` error. Text files are read as UTF-8 (tolerating the BOM Excel's
    "CSV UTF-8" export writes, which would otherwise hide the first column
    behind an invisible character) and then as cp1252, the usual Windows
    export encoding.

    Args:
        path: File to read.
        label: Human-readable name of the input, used in error messages.

    Returns:
        DataFrame of text values.

    Raises:
        ValueError: If the file is neither readable text nor a workbook the
            bundled Excel engines can open.
    """
    logging.info("%s: reading %s", label, path)
    with open(path, "rb") as handle:
        head = handle.read(8)
    suffix = path.suffix.lower()

    excel_engine: Optional[str] = None
    if head.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2 compound document = .xls
        excel_engine = "xlrd"
    elif head.startswith(b"PK\x03\x04"):  # zip container = .xlsx family
        excel_engine = "openpyxl"
    elif suffix in _EXCEL_SUFFIXES:  # Excel name, unrecognized content
        excel_engine = "xlrd" if suffix == ".xls" else "openpyxl"

    if excel_engine is not None:
        logging.info("%s: reading %s as an Excel workbook.", label, path.name)
        try:
            frame = pd.read_excel(path, engine=excel_engine)
        except ImportError as exc:
            raise ValueError(
                f"{label} {path} is an Excel workbook, but the {excel_engine} engine is not "
                f"available in this Python environment. Re-save it as a CSV and re-run."
            ) from exc
        for column in frame.columns:
            frame[column] = frame[column].map(_cell_to_text)
        return frame

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"{label} {path} is not readable as UTF-8 or Windows-1252 text, and is not an Excel "
        f"workbook. Re-save it from Excel as 'CSV UTF-8 (Comma delimited)' and re-run."
    )


def load_ridership(
    path: str, trip_col: str, stop_col: str, boardings_col: str
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """Load the optional trip-level ridership CSV and split it by grain.

    Args:
        path: CSV path ("" disables ridership accounting).
        trip_col: Column holding GTFS ``trip_id``.
        stop_col: Column holding ``stop_id`` (optional column; blank rows are
            whole-trip totals).
        boardings_col: Numeric average-daily-boardings column.

    Returns:
        ``None`` when disabled, else a tuple of (stop-grain rows summed by
        (trip_id, stop_id), trip-grain rows summed by trip_id). Trips that carry
        stop-grain rows have their trip-grain rows ignored (warned) so totals are
        never double-counted.

    Raises:
        FileNotFoundError: If *path* is set but does not exist.
        ValueError: If required columns are missing.
    """
    if not path.strip():
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ridership CSV not found: {csv_path}")
    ridership = _read_optional_table(csv_path, "Ridership CSV")
    for col in (trip_col, boardings_col):
        if col not in ridership.columns:
            raise ValueError(
                f"Ridership CSV is missing required column {col!r}. "
                f"Columns found: {', '.join(map(str, ridership.columns))}."
            )
    ridership = ridership.rename(columns={trip_col: "trip_id", boardings_col: "boardings"})
    ridership["trip_id"] = ridership["trip_id"].astype(str).str.strip()
    ridership["boardings"] = pd.to_numeric(ridership["boardings"], errors="coerce")
    bad = int(ridership["boardings"].isna().sum())
    if bad:
        logging.warning("Ridership CSV: dropped %d row(s) with non-numeric boardings.", bad)
        ridership = ridership[ridership["boardings"].notna()]

    if stop_col in ridership.columns:
        ridership = ridership.rename(columns={stop_col: "stop_id"})
        ridership["stop_id"] = ridership["stop_id"].fillna("").astype(str).str.strip()
    else:
        ridership["stop_id"] = ""

    stop_grain = ridership[ridership["stop_id"] != ""]
    trip_grain = ridership[ridership["stop_id"] == ""]
    overlap = set(stop_grain["trip_id"]) & set(trip_grain["trip_id"])
    if overlap:
        logging.warning(
            "Ridership CSV: %d trip(s) have both stop-grain and trip-grain rows; "
            "ignoring their trip-grain rows to avoid double counting.",
            len(overlap),
        )
        trip_grain = trip_grain[~trip_grain["trip_id"].isin(overlap)]

    stop_sums = stop_grain.groupby(["trip_id", "stop_id"])["boardings"].sum().reset_index()
    trip_sums = trip_grain.groupby("trip_id")["boardings"].sum().reset_index()
    logging.info(
        "Ridership loaded: %d stop-grain pair(s), %d trip-grain trip(s).",
        len(stop_sums),
        len(trip_sums),
    )
    return stop_sums, trip_sums


def ridership_impacts(
    dropped: pd.DataFrame,
    cut_trip_ids: set[str],
    eliminated_stop_ids: set[str],
    ridership: Optional[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[dict[str, float], pd.Series]:
    """Compute riders affected by the scenario's cut.

    Args:
        dropped: Removed event rows.
        cut_trip_ids: Trips with zero surviving events.
        eliminated_stop_ids: Stops whose service goes to zero.
        ridership: Output of :func:`load_ridership`, already scoped by the caller
            to trips operating on the analysis day (``None`` when disabled).

    Returns:
        Tuple of (summary dict with ``riders_on_cut_service`` and
        ``riders_at_eliminated_stops``, per-stop boardings-lost Series indexed by
        stop_id — empty when stop-grain data is unavailable).
    """
    empty = pd.Series(dtype=float)
    if ridership is None:
        return {}, empty
    stop_sums, trip_sums = ridership

    pairs = dropped[["trip_id", "stop_id"]].drop_duplicates()
    on_cut_stop_grain = pairs.merge(stop_sums, on=["trip_id", "stop_id"], how="inner")
    covered_trips = set(stop_sums["trip_id"])
    on_cut_trip_grain = trip_sums[trip_sums["trip_id"].isin(cut_trip_ids - covered_trips)]
    riders_on_cut = float(on_cut_stop_grain["boardings"].sum()) + float(
        on_cut_trip_grain["boardings"].sum()
    )

    if stop_sums.empty:
        riders_at_eliminated = float("nan")
        if eliminated_stop_ids:
            logging.warning(
                "Ridership CSV has no stop-grain rows — riders_at_eliminated_stops "
                "cannot be computed."
            )
        per_stop = empty
    else:
        at_elim = stop_sums[stop_sums["stop_id"].isin(eliminated_stop_ids)]
        riders_at_eliminated = float(at_elim["boardings"].sum())
        per_stop = on_cut_stop_grain.groupby("stop_id")["boardings"].sum()

    return (
        {
            "riders_on_cut_service": round(riders_on_cut, 1),
            "riders_at_eliminated_stops": round(riders_at_eliminated, 1),
        },
        per_stop,
    )


def load_stop_ridership(
    path: str,
    route_col: str,
    stop_col: str,
    boardings_col: str,
    routes: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Load the optional route × stop stop-usage CSV and resolve its route ids.

    Args:
        path: CSV path ("" disables stop-level ridership accounting).
        route_col: Column holding the route (matched against route_short_name
            or route_id, like scenario route tokens).
        stop_col: Column holding GTFS ``stop_id``.
        boardings_col: Numeric average-daily-boardings column.
        routes: Parsed ``routes.txt`` (for route resolution).

    Returns:
        ``None`` when disabled, else a frame with ``route_id``, ``stop_id``,
        ``boardings`` summed by pair. Rows whose route value matches nothing in
        routes.txt are dropped with a warning.

    Raises:
        FileNotFoundError: If *path* is set but does not exist.
        ValueError: If required columns are missing.
    """
    if not path.strip():
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Stop ridership CSV not found: {csv_path}")
    usage = _read_optional_table(csv_path, "Stop ridership CSV")
    for col in (route_col, stop_col, boardings_col):
        if col not in usage.columns:
            raise ValueError(
                f"Stop ridership CSV is missing required column {col!r}. "
                f"Columns found: {', '.join(map(str, usage.columns))}."
            )
    usage = usage.rename(
        columns={route_col: "route_token", stop_col: "stop_id", boardings_col: "boardings"}
    )
    usage["route_token"] = usage["route_token"].astype(str).str.strip()
    usage["stop_id"] = usage["stop_id"].astype(str).str.strip()
    usage["boardings"] = pd.to_numeric(usage["boardings"], errors="coerce")
    bad = int(usage["boardings"].isna().sum())
    if bad:
        logging.warning("Stop ridership CSV: dropped %d row(s) with non-numeric boardings.", bad)
        usage = usage[usage["boardings"].notna()]

    r = routes.drop_duplicates(subset=["route_id"]).copy()
    r["route_id"] = r["route_id"].astype(str).str.strip()
    short = r.get("route_short_name", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    token_to_id = dict(zip(r["route_id"], r["route_id"]))
    token_to_id.update({s: rid for s, rid in zip(short, r["route_id"]) if s})
    usage["route_id"] = usage["route_token"].map(token_to_id)
    unmatched = usage.loc[usage["route_id"].isna(), "route_token"].unique()
    if len(unmatched):
        logging.warning(
            "Stop ridership CSV: %d route value(s) match nothing in routes.txt "
            "(e.g. %s) — their rows are ignored.",
            len(unmatched),
            ", ".join(sorted(unmatched)[:5]),
        )
        usage = usage[usage["route_id"].notna()]
    out = usage.groupby(["route_id", "stop_id"], as_index=False)["boardings"].sum()
    logging.info("Stop ridership loaded: %d route × stop pair(s).", len(out))
    return out


def stop_ridership_impacts(
    events: pd.DataFrame,
    drop_mask: pd.Series,
    eliminated_stop_ids: set[str],
    stop_usage: Optional[pd.DataFrame],
) -> tuple[dict[str, float], pd.Series]:
    """Estimate boardings lost from route × stop stop-usage averages.

    For each (route, stop) pair, the pair's daily boardings are scaled by the
    share of that pair's trips the scenario removes — exact when the pair loses
    all service, a proportional estimate for partial cuts (which assumes riders
    are spread evenly across the pair's trips).

    Args:
        events: Baseline event table.
        drop_mask: Boolean drop mask over *events*.
        eliminated_stop_ids: Stops whose service goes to zero.
        stop_usage: Output of :func:`load_stop_ridership` (``None`` disables).

    Returns:
        Tuple of (summary dict with ``stop_boardings_lost_est`` and
        ``stop_riders_at_eliminated_stops``, per-stop estimated-boardings-lost
        Series indexed by stop_id).
    """
    empty = pd.Series(dtype=float)
    if stop_usage is None:
        return {}, empty

    pair_base = events.groupby(["route_id", "stop_id"])["trip_id"].nunique().rename("trips_base")
    pair_dropped = (
        events[drop_mask]
        .groupby(["route_id", "stop_id"])["trip_id"]
        .nunique()
        .rename("trips_dropped")
    )
    pairs = pd.concat([pair_base, pair_dropped], axis=1).fillna(0)
    pairs["share_cut"] = pairs["trips_dropped"] / pairs["trips_base"]

    joined = stop_usage.merge(
        pairs.reset_index(), on=["route_id", "stop_id"], how="inner", validate="one_to_one"
    )
    matched_pairs = len(joined)
    if matched_pairs < len(stop_usage):
        logging.info(
            "Stop ridership: %d of %d pair(s) matched service on the analysis day.",
            matched_pairs,
            len(stop_usage),
        )
    joined["boardings_lost"] = joined["boardings"] * joined["share_cut"]
    per_stop = joined.groupby("stop_id")["boardings_lost"].sum()
    per_stop = per_stop[per_stop > 0]

    # Scope to pairs with analysis-day service so boardings attributed to routes
    # that only run other day types (e.g. a holiday loop) don't inflate the total.
    at_eliminated = float(
        joined.loc[joined["stop_id"].isin(eliminated_stop_ids), "boardings"].sum()
    )
    return (
        {
            "stop_boardings_lost_est": round(float(joined["boardings_lost"].sum()), 1),
            "stop_riders_at_eliminated_stops": round(at_eliminated, 1),
        },
        per_stop,
    )


def export_scenario_gtfs(
    gtfs_dir: str,
    feed: Mapping[str, Optional[pd.DataFrame]],
    events_all: pd.DataFrame,
    drop_mask_all: pd.Series,
    output_dir: Path,
    token: str,
) -> None:
    """Write the scenario's surviving network as a standalone GTFS feed folder.

    The cut is applied across every service day (*events_all* spans the whole
    feed, not just the analysis day): masked (trip, stop) pairs are removed from
    ``stop_times``, trips left with no stop events are removed from ``trips``
    (and ``frequencies``), routes left with no trips are removed from
    ``routes``, and every other ``.txt`` in the source feed is copied through
    verbatim — so the folder is a complete feed any GTFS tool can consume.

    Args:
        gtfs_dir: Source feed folder or .zip (for copy-through files).
        feed: Loaded feed tables from :func:`load_feed`.
        events_all: All-days event table from :func:`build_events` with no
            service filter.
        drop_mask_all: The scenario's drop mask resolved over *events_all*.
        output_dir: Destination folder.
        token: Sanitized scenario name; the feed lands in ``<token>_gtfs/``.
    """
    dest = output_dir / f"{token}_gtfs"
    dest.mkdir(parents=True, exist_ok=True)

    dropped = events_all[drop_mask_all]
    surviving_trips = set(events_all.loc[~drop_mask_all, "trip_id"])
    removed_trips = set(events_all["trip_id"]) - surviving_trips
    dropped_pairs = set(zip(dropped["trip_id"], dropped["stop_id"]))

    stop_times = feed["stop_times"]
    trips = feed["trips"]
    routes = feed["routes"]
    assert isinstance(stop_times, pd.DataFrame)
    assert isinstance(trips, pd.DataFrame)
    assert isinstance(routes, pd.DataFrame)

    st = stop_times.copy()
    st["_trip"] = st["trip_id"].astype(str)
    pair_key = pd.Series(
        list(zip(st["_trip"], st["stop_id"].astype(str))), index=st.index, dtype=object
    )
    keep = ~st["_trip"].isin(removed_trips) & ~pair_key.isin(dropped_pairs)
    st_out = st[keep].drop(columns="_trip")

    kept_trip_ids = set(st_out["trip_id"].astype(str))
    trips_out = trips[trips["trip_id"].astype(str).isin(kept_trip_ids)]
    kept_route_ids = set(trips_out["route_id"].astype(str))
    routes_out = routes[routes["route_id"].astype(str).isin(kept_route_ids)]
    removed_routes = len(routes) - len(routes_out)

    st_out.to_csv(dest / "stop_times.txt", index=False)
    trips_out.to_csv(dest / "trips.txt", index=False)
    routes_out.to_csv(dest / "routes.txt", index=False)
    rewritten = {"stop_times.txt", "trips.txt", "routes.txt"}

    freq = feed.get("frequencies")
    if freq is not None and not freq.empty:
        freq_out = freq[~freq["trip_id"].astype(str).isin(removed_trips)]
        freq_out.to_csv(dest / "frequencies.txt", index=False)
        rewritten.add("frequencies.txt")

    src = Path(gtfs_dir)
    if src.is_dir():
        for txt in sorted(src.glob("*.txt")):
            if txt.name not in rewritten:
                (dest / txt.name).write_bytes(txt.read_bytes())
    elif src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as archive:
            for member in archive.namelist():
                base = os.path.basename(member)
                if base.endswith(".txt") and base not in rewritten:
                    (dest / base).write_bytes(archive.read(member))

    logging.info(
        "Wrote: %s (scenario GTFS — removed %d trip(s), %d route(s), %d stop event(s))",
        dest,
        len(removed_trips),
        removed_routes,
        len(stop_times) - len(st_out),
    )


def build_shape_lines(shapes: Optional[pd.DataFrame], sr: Any) -> Dict[str, Any]:
    """Assemble one projected polyline per shape from ``shapes.txt``.

    Args:
        shapes: Parsed ``shapes.txt`` or ``None``.
        sr: Projected spatial reference of the analysis.

    Returns:
        Mapping of ``shape_id`` to its projected arcpy polyline (empty when
        shapes are missing or no shape has two usable points).
    """
    lines: Dict[str, Any] = {}
    if shapes is None or shapes.empty:
        return lines
    pts = shapes.copy()
    pts["shape_pt_sequence"] = pd.to_numeric(pts["shape_pt_sequence"], errors="coerce")
    pts["shape_pt_lat"] = pd.to_numeric(pts["shape_pt_lat"], errors="coerce")
    pts["shape_pt_lon"] = pd.to_numeric(pts["shape_pt_lon"], errors="coerce")
    pts = pts.dropna(subset=["shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"])
    pts = pts.sort_values(["shape_id", "shape_pt_sequence"])

    wgs84 = arcpy.SpatialReference(4326)
    for shape_id, group in pts.groupby("shape_id"):
        if len(group) < 2:
            continue
        vertices = arcpy.Array(
            [
                arcpy.Point(float(lon), float(lat))
                for lon, lat in zip(group["shape_pt_lon"], group["shape_pt_lat"])
            ]
        )
        lines[str(shape_id)] = arcpy.Polyline(vertices, wgs84).projectAs(sr)
    return lines


def shape_length_miles(shape_lines: Mapping[str, Any], sr: Any) -> pd.Series:
    """Measure each shape line's length in miles.

    Args:
        shape_lines: Output of :func:`build_shape_lines`.
        sr: Projected spatial reference of the lines.

    Returns:
        Series of miles indexed by ``shape_id`` (empty when no lines exist).
    """
    if not shape_lines:
        return pd.Series(dtype=float)
    return pd.Series(
        [_sr_units_to_miles(line.length, sr) for line in shape_lines.values()],
        index=list(shape_lines),
        dtype=float,
    )


def export_line_layers(
    events: pd.DataFrame,
    drop_mask: pd.Series,
    trips: pd.DataFrame,
    shape_lines: Mapping[str, Any],
    output_dir: Path,
    token: str,
    sr: Any,
) -> None:
    """Write the remaining- and removed-alignment line shapefiles for a scenario.

    A shape is *removed* when baseline trips ran it but no surviving trip does,
    and *remaining* otherwise. Truncations keep their (unchanged) shape in the
    remaining layer. Each feature carries the routes using the shape and its
    baseline/surviving trip counts.

    Args:
        events: Baseline event table.
        drop_mask: Boolean drop mask over *events*.
        trips: Parsed ``trips.txt`` (for the trip -> shape_id mapping).
        shape_lines: Output of :func:`build_shape_lines` (non-empty).
        output_dir: Destination folder.
        token: Sanitized scenario name used as the filename prefix.
        sr: Projected spatial reference of the lines.
    """
    trip_info = events.drop_duplicates(subset=["trip_id"])[["trip_id", "route_label"]]
    if "shape_id" not in trips.columns:
        return
    shape_map = trips.drop_duplicates(subset=["trip_id"])[["trip_id", "shape_id"]].copy()
    shape_map["trip_id"] = shape_map["trip_id"].astype(str)
    shape_map["shape_id"] = shape_map["shape_id"].fillna("").astype(str)
    usage = trip_info.merge(shape_map, on="trip_id", how="left")
    usage = usage[usage["shape_id"] != ""]
    if usage.empty:
        logging.warning("No baseline trip has a shape_id — line layers skipped.")
        return

    surviving_trips = set(events.loc[~drop_mask, "trip_id"])
    usage["surviving"] = usage["trip_id"].isin(surviving_trips)
    per_shape = usage.groupby("shape_id").agg(
        routes=("route_label", lambda s: _as_sorted_csv(s.tolist())),
        trips_base=("trip_id", "nunique"),
        trips_left=("surviving", "sum"),
    )
    per_shape["trips_left"] = per_shape["trips_left"].astype(int)
    drawable = per_shape[per_shape.index.isin(shape_lines)]
    missing_geom = len(per_shape) - len(drawable)
    if missing_geom:
        logging.warning("%d used shape(s) have no drawable geometry in shapes.txt.", missing_geom)

    fields = (
        ("shape_id", "TEXT", 100),
        ("routes", "TEXT", _SHP_TEXT_MAX),
        ("trips_base", "LONG", 0),
        ("trips_left", "LONG", 0),
    )
    for label, subset in (
        ("remaining_lines", drawable[drawable["trips_left"] > 0]),
        ("removed_lines", drawable[drawable["trips_left"] == 0]),
    ):
        if subset.empty:
            continue
        rows = [
            (
                _shp_text(shape_id),
                _shp_text(row.routes),
                int(row.trips_base),
                int(row.trips_left),
                shape_lines[shape_id],
            )
            for shape_id, row in zip(subset.index, subset.itertuples(index=False))
        ]
        path = write_feature_shapefile(
            output_dir, f"{token}_{label}.shp", "POLYLINE", sr, fields, rows
        )
        logging.info("Wrote: %s (%d shape(s))", path, len(subset))


def operational_savings(
    events: pd.DataFrame,
    drop_mask: pd.Series,
    trips: pd.DataFrame,
    shape_miles: pd.Series,
) -> dict[str, Any]:
    """Estimate revenue service removed by the scenario, from GTFS alone.

    Revenue hours are the change in summed per-trip spans (first to last stop
    event), so truncations that shorten a trip's ends count. Revenue miles are
    summed shape lengths of wholly-cut trips only — truncation mileage changes
    are not estimated. Deadhead, pull-ins/outs, and layover are invisible to
    GTFS; scheduled-operations data (TODS) would be needed for true operating
    savings.

    Args:
        events: Baseline event table.
        drop_mask: Boolean drop mask over *events*.
        trips: Parsed ``trips.txt`` (for the trip -> shape_id mapping).
        shape_miles: Output of :func:`shape_length_miles`.

    Returns:
        Dict with ``trips_cut``, ``trips_truncated``, ``revenue_hours_cut``, and
        ``revenue_miles_cut`` (NaN when no cut trip has a measurable shape).
    """
    surviving = events[~drop_mask]
    dropped = events[drop_mask]
    base_span = events.groupby("trip_id")["dep_min"].agg(["min", "max"])
    scen_span = surviving.groupby("trip_id")["dep_min"].agg(["min", "max"])
    base_hours = float((base_span["max"] - base_span["min"]).sum()) / 60.0
    scen_hours = float((scen_span["max"] - scen_span["min"]).sum()) / 60.0

    cut_trip_ids = set(base_span.index) - set(scen_span.index)
    truncated = (set(dropped["trip_id"]) & set(scen_span.index)) if not dropped.empty else set()

    miles_cut = float("nan")
    if cut_trip_ids:
        t = trips.drop_duplicates(subset=["trip_id"]).copy()
        t["trip_id"] = t["trip_id"].astype(str)
        shape_ids = (
            t.set_index("trip_id")["shape_id"].astype(str)
            if "shape_id" in t.columns
            else pd.Series(dtype=str)
        )
        cut_shapes = shape_ids.reindex(sorted(cut_trip_ids))
        per_trip = cut_shapes.map(shape_miles)
        measured = int(per_trip.notna().sum())
        if measured:
            miles_cut = round(float(per_trip.sum()), 1)
        if measured < len(cut_trip_ids):
            logging.warning(
                "Revenue miles reflect %d of %d cut trip(s) — the rest have no "
                "measurable shape in shapes.txt.",
                measured,
                len(cut_trip_ids),
            )
    return {
        "trips_cut": len(cut_trip_ids),
        "trips_truncated": len(truncated),
        "revenue_hours_cut": round(base_hours - scen_hours, 1),
        "revenue_miles_cut": miles_cut,
    }


# =============================================================================
# SCENARIO PIPELINE
# =============================================================================


def build_baseline(events: pd.DataFrame, stops: pd.DataFrame, cfg: Config, sr: Any) -> Baseline:
    """Compute the products every scenario is compared against, once per run.

    Args:
        events: Baseline event table.
        stops: Projected stops table.
        cfg: Resolved configuration.
        sr: Projected spatial reference.

    Returns:
        The populated :class:`Baseline`.
    """
    stats = summarize_stops(events)
    bins = bin_counts(events, cfg.time_bin_minutes)
    served = stops[stops["stop_id"].isin(set(stats.index))]
    missing_geom = len(set(stats.index)) - len(served)
    if missing_geom > 0:
        logging.warning(
            "%d served stop(s) have no usable coordinates (or were filtered out) and "
            "are excluded from the coverage comparison.",
            missing_geom,
        )
    radius = _miles_to_sr_units(cfg.stop_buffer_miles, sr)
    coverage = buffer_union(served, radius, sr)
    coverage_sqmi = 0.0 if coverage is None else _area_to_sqmi(coverage.area, sr)
    logging.info(
        "Baseline walk-access coverage: %.2f sq mi from %d served stop(s) at %.2f mi.",
        coverage_sqmi,
        len(served),
        cfg.stop_buffer_miles,
    )
    return Baseline(
        events=events, stats=stats, bins=bins, coverage=coverage, coverage_sqmi=coverage_sqmi
    )


def run_scenario(
    scenario: Mapping[str, Any],
    baseline: Baseline,
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    stops: pd.DataFrame,
    shape_lines: Mapping[str, Any],
    shape_miles: pd.Series,
    demographics: Optional[Sequence[Mapping[str, Any]]],
    ridership: Optional[tuple[pd.DataFrame, pd.DataFrame]],
    stop_usage: Optional[pd.DataFrame],
    feed: Mapping[str, Optional[pd.DataFrame]],
    events_all: Optional[pd.DataFrame],
    cfg: Config,
    sr: Any,
    write_details: bool = True,
) -> dict[str, Any]:
    """Evaluate one scenario end-to-end and write its per-scenario outputs.

    Args:
        scenario: One SCENARIOS entry.
        baseline: Shared baseline products from :func:`build_baseline`.
        routes: Parsed ``routes.txt``.
        trips: Parsed ``trips.txt``.
        stops: Projected stops table.
        shape_lines: Projected shape lines from :func:`build_shape_lines`.
        shape_miles: Shape lengths from :func:`shape_length_miles`.
        demographics: Records from :func:`load_demographics` or ``None``.
        ridership: Output of :func:`load_ridership` or ``None``.
        stop_usage: Output of :func:`load_stop_ridership` or ``None``.
        feed: Loaded feed tables (for the scenario GTFS export).
        events_all: All-days event table for the GTFS export, or ``None`` when
            the export is disabled.
        cfg: Resolved configuration.
        sr: Projected spatial reference.
        write_details: When False (screening mode), compute the summary row but
            skip every per-scenario file — impact/matrix CSVs, shapefiles, and
            the scenario GTFS export.

    Returns:
        The scenario's summary row (one dict) for ``scenario_summary.csv``.
    """
    name = str(scenario.get("name", "")).strip()
    if not name:
        raise ValueError(f"Every scenario needs a 'name'; got {scenario!r}.")
    token = _sanitize_name(name)
    logging.info("--- Scenario %r ---", name)

    events = baseline.events
    drop_mask = resolve_drop_mask(scenario, events, routes)
    dropped = events[drop_mask]
    surviving = events[~drop_mask]
    if dropped.empty:
        logging.warning("Scenario %r removes nothing — check its route/trip/stop keys.", name)

    scen_stats = summarize_stops(surviving)
    scen_bins = bin_counts(surviving, cfg.time_bin_minutes)

    impacts = compute_stop_impacts(baseline.stats, scen_stats, baseline.bins, scen_bins, stops)
    if write_details:
        impacts = add_nearest_remaining(impacts, stops, sr)

    eliminated_ids = set(impacts.loc[impacts["classification"] == "eliminated", "stop_id"])
    reduced_ids = set(impacts.loc[impacts["classification"] == "reduced", "stop_id"])
    cut_trip_ids = set(events["trip_id"]) - set(surviving["trip_id"])

    ops = operational_savings(events, drop_mask, trips, shape_miles)
    base_sqmi, scen_sqmi, lost_geom, remaining_geom = coverage_stats(
        set(baseline.stats.index),
        set(scen_stats.index),
        stops,
        cfg.stop_buffer_miles,
        sr,
        baseline.coverage,
        baseline.coverage_sqmi,
        include_remaining=write_details and cfg.export_map_layers,
    )
    demo_lost: dict[str, float] = {}
    if demographics is not None:
        demo_lost = apportion_demographics(lost_geom, demographics, cfg.demographic_fields)
        demo_lost = {k: round(v, 1) for k, v in demo_lost.items()}

    riders, boardings_lost_by_stop = ridership_impacts(
        dropped, cut_trip_ids, eliminated_ids, ridership
    )
    if not boardings_lost_by_stop.empty:
        impacts["boardings_lost"] = (
            impacts["stop_id"].map(boardings_lost_by_stop).fillna(0.0).round(1)
        )
    stop_riders, est_lost_by_stop = stop_ridership_impacts(
        events, drop_mask, eliminated_ids, stop_usage
    )
    if not est_lost_by_stop.empty:
        impacts["stop_boardings_lost_est"] = (
            impacts["stop_id"].map(est_lost_by_stop).fillna(0.0).round(1)
        )

    if write_details:
        impacted = impacts[impacts["classification"] != "unchanged"].copy()
        impacts_path = cfg.output_dir / f"{token}_stop_impacts.csv"
        impacted.to_csv(impacts_path, index=False)
        logging.info("Wrote: %s (%d impacted stops)", impacts_path, len(impacted))

        matrix = baseline.bins.merge(
            scen_bins,
            on=["stop_id", "bin_start_min"],
            how="outer",
            suffixes=("_baseline", "_scenario"),
        )
        matrix[["trips_baseline", "trips_scenario"]] = (
            matrix[["trips_baseline", "trips_scenario"]].fillna(0).astype(int)
        )
        matrix["trips_delta"] = matrix["trips_scenario"] - matrix["trips_baseline"]
        matrix["bin_start"] = matrix["bin_start_min"].map(minutes_to_hhmm)
        names = stops.set_index("stop_id")["stop_name"]
        matrix["stop_name"] = matrix["stop_id"].map(names).fillna("")
        matrix = matrix.sort_values(["stop_id", "bin_start_min"])
        matrix_path = cfg.output_dir / f"{token}_stop_time_bins.csv"
        matrix[
            ["stop_id", "stop_name", "bin_start", "trips_baseline", "trips_scenario", "trips_delta"]
        ].to_csv(matrix_path, index=False)
        logging.info("Wrote: %s", matrix_path)

    lost_sqmi = 0.0 if _is_empty(lost_geom) else _area_to_sqmi(lost_geom.area, sr)
    if write_details and not _is_empty(lost_geom):
        coverage_path = write_feature_shapefile(
            cfg.output_dir,
            f"{token}_lost_coverage.shp",
            "POLYGON",
            sr,
            (("scenario", "TEXT", _SHP_TEXT_MAX),),
            [(_shp_text(name), lost_geom)],
        )
        logging.info("Wrote: %s (%.2f sq mi lost)", coverage_path, lost_sqmi)

    if write_details and cfg.export_map_layers:
        if not _is_empty(remaining_geom):
            remaining_path = write_feature_shapefile(
                cfg.output_dir,
                f"{token}_remaining_coverage.shp",
                "POLYGON",
                sr,
                (("scenario", "TEXT", _SHP_TEXT_MAX),),
                [(_shp_text(name), remaining_geom)],
            )
            logging.info("Wrote: %s (%.2f sq mi remaining)", remaining_path, scen_sqmi)
        if shape_lines:
            export_line_layers(events, drop_mask, trips, shape_lines, cfg.output_dir, token, sr)

    if write_details and cfg.export_scenario_gtfs and events_all is not None:
        drop_mask_all = resolve_drop_mask(scenario, events_all, routes)
        export_scenario_gtfs(cfg.gtfs_dir, feed, events_all, drop_mask_all, cfg.output_dir, token)

    stop_bins_lost_total = int(impacts["time_bins_lost"].sum())
    routes_affected = _as_sorted_csv(dropped["route_label"].tolist()) if not dropped.empty else ""

    summary: dict[str, Any] = {
        "scenario": name,
        "description": str(scenario.get("description", "")),
        "routes_affected": routes_affected,
        "events_removed": int(len(dropped)),
        "stops_eliminated": len(eliminated_ids),
        "stops_reduced": len(reduced_ids),
        "stop_time_bins_lost": stop_bins_lost_total,
        "baseline_coverage_sqmi": round(base_sqmi, 2),
        "coverage_lost_sqmi": round(lost_sqmi, 2),
        **ops,
        **demo_lost,
        **riders,
        **stop_riders,
    }
    hours = summary.get("revenue_hours_cut") or 0
    if riders and hours:
        summary["riders_per_revenue_hour_cut"] = round(riders["riders_on_cut_service"] / hours, 1)
    return summary


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
    log_path = output_dir / "service_cut_impact_runlog.txt"

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
        "SERVICE CUT IMPACT RUN LOG",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
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


def run(cfg: Config) -> pd.DataFrame:
    """Execute every scenario and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        The cross-scenario summary table (also written to disk).
    """
    sr = _spatial_reference(cfg.crs_wkid)
    feed = load_feed(cfg.gtfs_dir)
    stops_txt = feed["stops"]
    routes = feed["routes"]
    trips = feed["trips"]
    stop_times = feed["stop_times"]
    assert isinstance(stops_txt, pd.DataFrame)
    assert isinstance(routes, pd.DataFrame)
    assert isinstance(trips, pd.DataFrame)
    assert isinstance(stop_times, pd.DataFrame)

    service_ids = select_service_ids(
        feed["calendar"], feed["calendar_dates"], cfg.service_day, cfg.service_date
    )
    events = build_events(stop_times, trips, routes, service_ids)
    stops = prepare_stops_table(stops_txt, cfg.filter_platform_stops, sr)
    logging.info("Building route alignments from shapes.txt …")
    shape_lines = build_shape_lines(feed["shapes"], sr)
    shape_miles = shape_length_miles(shape_lines, sr)
    logging.info("Built %d route alignment(s) from shapes.txt.", len(shape_lines))
    if not shape_lines:
        logging.warning(
            "No usable shapes.txt — revenue_miles_cut will be blank and line layers skipped."
        )

    demographics: Optional[List[Dict[str, Any]]] = None
    if cfg.demographics_path.strip():
        demographics = load_demographics(cfg.demographics_path, cfg.demographic_fields, sr)

    ridership = load_ridership(
        cfg.ridership_csv, RIDERSHIP_TRIP_ID_COL, RIDERSHIP_STOP_ID_COL, RIDERSHIP_BOARDINGS_COL
    )
    if ridership is not None:
        # Keep every ridership metric on the same day as the service metrics: rows
        # for trips that do not operate on the analysis day are set aside.
        day_trips = set(events["trip_id"])
        stop_sums, trip_sums = ridership
        off_day = len(set(stop_sums["trip_id"]) | set(trip_sums["trip_id"])) - len(
            (set(stop_sums["trip_id"]) | set(trip_sums["trip_id"])) & day_trips
        )
        if off_day:
            logging.info(
                "Ridership: ignoring rows for %d trip(s) not operating on the analysis day.",
                off_day,
            )
        ridership = (
            stop_sums[stop_sums["trip_id"].isin(day_trips)].reset_index(drop=True),
            trip_sums[trip_sums["trip_id"].isin(day_trips)].reset_index(drop=True),
        )

    stop_usage = load_stop_ridership(
        cfg.stop_ridership_csv,
        STOP_RIDERSHIP_ROUTE_COL,
        STOP_RIDERSHIP_STOP_ID_COL,
        STOP_RIDERSHIP_BOARDINGS_COL,
        routes,
    )

    screening = isinstance(cfg.scenarios, str)
    if screening:
        if str(cfg.scenarios).strip().lower() != "each_route":
            raise ValueError(
                f"SCENARIOS must be a list of scenario dicts or the string "
                f"'each_route'; got {cfg.scenarios!r}."
            )
        scenarios = build_screening_scenarios(events)
        logging.info(
            "Screening mode: evaluating the elimination of each of %d route(s). "
            "Per-scenario detail files are skipped — rerun with an explicit "
            "SCENARIOS entry for full outputs on a route of interest.",
            len(scenarios),
        )
    else:
        scenarios = list(cfg.scenarios)

    events_all: Optional[pd.DataFrame] = None
    if cfg.export_scenario_gtfs and not screening:
        # The exported feeds apply each cut across every service day, so the
        # scenario masks are re-resolved over an unfiltered event table.
        logging.info("Scenario GTFS export enabled — building the all-days event table.")
        events_all = build_events(stop_times, trips, routes, service_ids=None)

    names = [str(s.get("name", "")).strip() for s in scenarios]
    if len(names) != len(set(names)):
        raise ValueError(f"Scenario names must be unique; got {names}.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline(events, stops, cfg, sr)
    summary_rows = [
        run_scenario(
            scenario,
            baseline,
            routes,
            trips,
            stops,
            shape_lines,
            shape_miles,
            demographics,
            ridership,
            stop_usage,
            feed,
            events_all,
            cfg,
            sr,
            write_details=not screening,
        )
        for scenario in scenarios
    ]
    summary = pd.DataFrame(summary_rows)
    if screening:
        # Rank the exposure table: riders at stake when ridership is loaded,
        # else the operational scale of the route.
        rank_col = next(
            (
                col
                for col in ("riders_on_cut_service", "stop_boardings_lost_est", "revenue_hours_cut")
                if col in summary.columns and summary[col].notna().any()
            ),
            None,
        )
        if rank_col:
            summary = summary.sort_values(rank_col, ascending=False, ignore_index=True)
    summary_path = cfg.output_dir / SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    logging.info("Wrote: %s (%d scenario(s))", summary_path, len(summary))

    summary_lines = [
        f"Scenarios evaluated: {len(summary_rows)}",
        f"Service day:         {cfg.service_date or cfg.service_day}",
        f"Spatial reference:   {sr.name} (WKID {cfg.crs_wkid})",
    ] + [
        f"  {row['scenario']}: {row['trips_cut']} trip(s) cut, "
        f"{row['stops_eliminated']} stop(s) eliminated, "
        f"{row['coverage_lost_sqmi']} sq mi coverage lost"
        for row in summary_rows
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    SCENARIOS is deliberately configuration-only — its nested structure does not
    map to flags. Edit it in the CONFIGURATION section.
    """
    p = argparse.ArgumentParser(
        description="Scenario-based service-cut impact analysis for a GTFS feed (ArcPy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gtfs-dir", default=GTFS_DIR, help="GTFS folder or .zip path.")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--service-day", default=SERVICE_DAY, help="'weekday' or a day name, e.g. 'saturday'."
    )
    p.add_argument(
        "--service-date",
        default=SERVICE_DATE,
        help="Explicit YYYY-MM-DD analysis date (overrides --service-day selection).",
    )
    p.add_argument(
        "--time-bin-minutes",
        type=int,
        default=TIME_BIN_MINUTES,
        help="Width of time-of-day bins for the stop x time-bin matrix.",
    )
    p.add_argument(
        "--buffer-miles",
        type=float,
        default=STOP_BUFFER_MILES,
        help="Walk-access buffer radius around stops, in miles.",
    )
    p.add_argument(
        "--wkid",
        "--epsg",
        dest="wkid",
        type=int,
        default=PROJECTED_CRS_WKID,
        help="Projected spatial reference WKID (EPSG code) for buffers and areas.",
    )
    p.add_argument(
        "--demographics",
        default=DEMOGRAPHICS_PATH,
        help="Optional demographics polygon layer ('' disables).",
    )
    p.add_argument(
        "--demographic-fields",
        nargs="*",
        default=list(DEMOGRAPHIC_FIELDS),
        help="Numeric demographics fields to apportion onto lost coverage.",
    )
    p.add_argument(
        "--ridership",
        default=RIDERSHIP_CSV,
        help="Optional trip-level ridership CSV ('' disables).",
    )
    p.add_argument(
        "--map-layers",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_MAP_LAYERS,
        help="Write remaining/removed line and remaining-coverage layers per scenario.",
    )
    p.add_argument(
        "--stop-ridership",
        default=STOP_RIDERSHIP_CSV,
        help="Optional route x stop stop-usage CSV ('' disables).",
    )
    p.add_argument(
        "--scenario-gtfs",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_SCENARIO_GTFS,
        help="Write each scenario's surviving network as a GTFS feed folder.",
    )
    p.add_argument(
        "--each-route",
        action="store_true",
        default=False,
        help=(
            "Screening mode: ignore SCENARIOS and evaluate the elimination of every "
            "route serving the analysis day, writing only the ranked summary."
        ),
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

    if args.gtfs_dir == GTFS_DIR and GTFS_DIR.startswith("Path"):
        logging.warning(
            "GTFS_DIR is still a placeholder. Update the CONFIGURATION section or pass "
            "--gtfs-dir before running."
        )
        return 2
    if args.output_dir == OUTPUT_DIR and OUTPUT_DIR.startswith("Path"):
        logging.warning(
            "OUTPUT_DIR is still a placeholder. Update the CONFIGURATION section or pass "
            "--output-dir before running."
        )
        return 2
    if str(args.output_dir).lower().endswith((".gdb", ".sde")):
        logging.error(
            "OUTPUT_DIR must be a filesystem folder — this script writes CSVs and "
            "shapefiles alongside each other, which a geodatabase cannot hold."
        )
        return 2

    arcpy.env.overwriteOutput = True

    cfg = Config(
        gtfs_dir=args.gtfs_dir,
        output_dir=Path(args.output_dir).expanduser(),
        scenarios="each_route" if args.each_route else SCENARIOS,
        service_day=args.service_day,
        service_date=args.service_date,
        time_bin_minutes=args.time_bin_minutes,
        stop_buffer_miles=args.buffer_miles,
        crs_wkid=args.wkid,
        demographics_path=args.demographics,
        demographic_fields=list(args.demographic_fields),
        ridership_csv=args.ridership,
        stop_ridership_csv=args.stop_ridership,
        export_map_layers=bool(args.map_layers),
        export_scenario_gtfs=bool(args.scenario_gtfs),
    )

    try:
        run(cfg)
    except (OSError, ValueError, RuntimeError, arcpy.ExecuteError) as exc:
        # The exception type is part of the message: a bare decode or lookup
        # error text alone leaves the user with nothing to act on.
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
