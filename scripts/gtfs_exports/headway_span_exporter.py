"""Compute route-level headway and span of service from a GTFS feed.

Produces a single CSV with one row per ``route_id``, ready to join to the
ridership-model anchor table on that key.

Metrics
-------
avg_headway_min
    Mean gap (minutes) between consecutive trip departures from the first
    stop of each trip.  Computed per direction then averaged across
    directions, so a two-way route is not credited with half its true
    headway.
span_hrs
    Hours from the first departure to the last departure.  GTFS times
    beyond 24:00 (e.g. ``"25:30:00"``) are handled correctly — they are
    preserved as integers ≥ 1440 minutes before the span is calculated.
trip_count
    Total one-way trips across all directions (diagnostic; not used by
    the model directly).

Weekday service is the default because the ridership model uses weekday
NTD data.  A single day (e.g. ``"friday"``) can be selected via
``SERVICE_DAY`` for agencies whose Monday / midweek / Friday schedules
differ, and ``SERVICE_DATE`` pins the analysis to one explicit date.

Outputs
-------
- ``headway_span_by_route.csv`` (``OUTPUT_FILENAME``) in ``OUTPUT_DIR``: one
  row per ``route_id`` with ``avg_headway_min``, ``span_hrs``, and
  ``trip_count``.

Notes:
-----
Service selection is date-based: ``calendar.txt`` and ``calendar_dates.txt``
are expanded into each service_id's real set of active dates, and metrics
are computed for a single representative date of the requested day type
(the median date of the most common service pattern, skipping observed
federal holidays). This keeps holiday-only services out of weekday
headways and avoids double-counting agencies that run distinct Monday /
midweek / Friday schedules — the script warns when it detects that split.
Feeds whose calendar lacks usable dates fall back to the day-of-week
columns with a warning; ``SERVICE_DATE`` overrides selection entirely.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--gtfs-folder`` /
``--output`` / ``--service-day`` / ``--service-date`` / ``--filter-in`` /
``--filter-out``) and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, List, Optional, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me
OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me
OUTPUT_FILENAME: str = "headway_span_by_route.csv"

# "weekday" (a typical Mon–Fri day) or a single day name ("monday" …
# "sunday"). Use a single day for agencies whose weekday schedules differ
# by day of week (e.g. separate Monday / Tue–Thu / Friday service).
SERVICE_DAY: str = "weekday"

# Optional explicit analysis date ("YYYYMMDD"). When set, SERVICE_DAY's
# automatic date selection is skipped and metrics reflect exactly this date.
SERVICE_DATE: str = ""

# Optional route filters — leave empty to process all routes.
FILTER_IN_ROUTE_SHORT_NAMES: list[str] = []
FILTER_OUT_ROUTE_SHORT_NAMES: list[str] = []

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# CONSTANTS
# =============================================================================

REQ_FILES: Final[tuple[str, ...]] = (
    "trips.txt",
    "stop_times.txt",
    "routes.txt",
    "calendar.txt",
)
_DAY_COLS: Final[dict[str, list[str]]] = {
    "weekday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "monday": ["monday"],
    "tuesday": ["tuesday"],
    "wednesday": ["wednesday"],
    "thursday": ["thursday"],
    "friday": ["friday"],
    "saturday": ["saturday"],
    "sunday": ["sunday"],
}
_SERVICE_DAY_CHOICES: Final[tuple[str, ...]] = (
    "weekday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

# =============================================================================
# FUNCTIONS
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


def load_gtfs(folder: Path) -> dict[str, pd.DataFrame]:
    """Load the required GTFS files from *folder* into a keyed dict."""
    missing = [f for f in REQ_FILES if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing GTFS file(s): {', '.join(missing)}")
    return {f[:-4]: pd.read_csv(folder / f, dtype=str, low_memory=False) for f in REQ_FILES}


def service_ids_for_day(calendar: pd.DataFrame, service_day: str) -> set[str]:
    """Return service_ids that operate on every day covered by *service_day*.

    Fallback used only when the feed's calendar lacks usable dates — see
    :func:`resolve_service_ids` for the preferred date-based path.

    Args:
        calendar: DataFrame from ``calendar.txt``.
        service_day: ``"weekday"`` or one of the seven day names.

    Raises:
        ValueError: If *service_day* is not a recognised key or if a
            required day column is absent from *calendar*.
    """
    cols = _DAY_COLS.get(service_day)
    if cols is None:
        raise ValueError(
            f"SERVICE_DAY must be 'weekday' or a day name ('monday'…'sunday'); got {service_day!r}"
        )
    cal = calendar.copy()
    for c in cols:
        if c not in cal.columns:
            raise ValueError(f"calendar.txt is missing expected column '{c}'.")
        cal[c] = pd.to_numeric(cal[c], errors="coerce").fillna(0)
    mask = (cal[cols] == 1).all(axis=1)
    ids = set(cal.loc[mask, "service_id"].astype(str))
    if not ids:
        logging.warning(
            "No service_ids found for SERVICE_DAY=%r — check calendar.txt.",
            service_day,
        )
    return ids


# ---- REUSABLE HELPERS (copied from utils/calendar_helpers.py) --------------


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


# ---- REUSABLE HELPERS (copied from utils/time_helpers.py) -------------------


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


def resolve_service_ids(
    calendar: pd.DataFrame,
    calendar_dates: Optional[pd.DataFrame],
    service_day: str,
    service_date_text: str = "",
) -> set[str]:
    """Resolve the service_ids to analyse, preferring real calendar dates.

    Expands ``calendar.txt`` + ``calendar_dates.txt`` into each service_id's
    active dates and picks a representative date of *service_day* (skipping
    observed federal holidays), or uses the explicit *service_date_text*
    override. Feeds whose calendar rows lack usable dates fall back to the
    day-of-week columns via :func:`service_ids_for_day`, with a warning.

    Args:
        calendar: Parsed ``calendar.txt``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.
        service_day: ``"weekday"`` or one of the seven day names.
        service_date_text: Optional explicit ``YYYYMMDD`` analysis date.

    Returns:
        Set of service_id strings to analyse (possibly empty).

    Raises:
        ValueError: If *service_date_text* is set but not a valid YYYYMMDD
            date, or the override date cannot be resolved against the feed.
    """
    override: Optional[dt.date] = None
    if service_date_text.strip():
        try:
            override = dt.datetime.strptime(service_date_text.strip(), "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError(
                f"SERVICE_DATE must be a YYYYMMDD date; got {service_date_text!r}"
            ) from exc

    try:
        active = expand_service_active_dates(calendar, calendar_dates)
        if override is not None:
            _, ids = representative_service_date(active, service_day, override_date=override)
            return ids
        years = {d.year for dates in active.values() for d in dates}
        holidays: set[dt.date] = set()
        for year in range(min(years), max(years) + 2):
            holidays |= federal_holidays_observed(year)
        _, ids = representative_service_date(active, service_day, exclude_dates=holidays)
        return ids
    except ValueError as exc:
        if override is not None:
            raise
        logging.warning(
            "Date-based service resolution failed (%s); falling back to "
            "calendar.txt day-of-week columns.",
            exc,
        )
        return service_ids_for_day(calendar, service_day)


def first_departures(stop_times: pd.DataFrame, trip_ids: set[str]) -> pd.DataFrame:
    """Return one row per trip with the departure time of its first stop.

    Args:
        stop_times: Full ``stop_times.txt`` DataFrame.
        trip_ids: Set of trip_id strings to process.

    Returns:
        DataFrame with columns ``trip_id`` and ``departure_min``.
    """
    st = stop_times[stop_times["trip_id"].isin(trip_ids)].copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st["departure_min"] = st["departure_time"].map(parse_time_to_minutes)
    st = st.dropna(subset=["stop_sequence", "departure_min"])
    first = st.sort_values("stop_sequence").groupby("trip_id", sort=False).first().reset_index()
    return first[["trip_id", "departure_min"]]


def compute_headway_span(trips_dep: pd.DataFrame) -> pd.DataFrame:
    """Compute avg_headway_min, span_hrs, and trip_count per route.

    Args:
        trips_dep: DataFrame with columns ``route_id``, ``direction_id``,
            and ``departure_min`` (one row per trip).

    Returns:
        One row per ``route_id`` with ``avg_headway_min``, ``span_hrs``,
        and ``trip_count``.
    """
    per_dir: list[dict] = []
    for (route_id, direction_id), grp in trips_dep.groupby(["route_id", "direction_id"]):
        deps = grp["departure_min"].sort_values().to_numpy()
        n = len(deps)
        span_min = float(deps[-1] - deps[0]) if n > 1 else 0.0
        headway = span_min / (n - 1) if n > 1 else float("nan")
        per_dir.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "avg_headway_min": round(headway, 1),
                "span_min": span_min,
                "trip_count": n,
            }
        )

    if not per_dir:
        return pd.DataFrame(columns=["route_id", "avg_headway_min", "span_hrs", "trip_count"])

    df = pd.DataFrame(per_dir)
    out = (
        df.groupby("route_id")
        .agg(
            avg_headway_min=("avg_headway_min", "mean"),
            span_min=("span_min", "max"),  # longest span across directions
            trip_count=("trip_count", "sum"),
        )
        .reset_index()
    )
    out["avg_headway_min"] = out["avg_headway_min"].round(1)
    out["span_hrs"] = (out["span_min"] / 60).round(2)
    return out[["route_id", "avg_headway_min", "span_hrs", "trip_count"]]


def run(
    gtfs_folder: Path | None = None,
    output_path: Path | None = None,
    service_day: str | None = None,
    service_date: str | None = None,
    filter_in_route_short_names: Sequence[str] | None = None,
    filter_out_route_short_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute headway and span for one service day and write to CSV.

    Unset args fall back to the config block at the top of this file, so
    ``m.GTFS_FOLDER = ...; m.run()`` works after a plain import.
    """
    gtfs_folder = GTFS_FOLDER if gtfs_folder is None else Path(gtfs_folder)
    output_path = (OUTPUT_DIR / OUTPUT_FILENAME) if output_path is None else Path(output_path)
    service_day = SERVICE_DAY if service_day is None else service_day
    service_date = SERVICE_DATE if service_date is None else str(service_date)
    filter_in = (
        FILTER_IN_ROUTE_SHORT_NAMES
        if filter_in_route_short_names is None
        else list(filter_in_route_short_names)
    )
    filter_out = (
        FILTER_OUT_ROUTE_SHORT_NAMES
        if filter_out_route_short_names is None
        else list(filter_out_route_short_names)
    )

    gtfs = load_gtfs(gtfs_folder)

    calendar_dates_path = gtfs_folder / "calendar_dates.txt"
    calendar_dates = (
        pd.read_csv(calendar_dates_path, dtype=str, low_memory=False)
        if calendar_dates_path.exists()
        else None
    )

    svc_ids = resolve_service_ids(gtfs["calendar"], calendar_dates, service_day, service_date)
    logging.info("%d service_id(s) matched for SERVICE_DAY=%r.", len(svc_ids), service_day)

    trips = gtfs["trips"].copy()
    trips = trips[trips["service_id"].isin(svc_ids)]

    routes = gtfs["routes"][["route_id", "route_short_name"]]
    trips = trips.merge(routes, on="route_id", how="left")

    if filter_in:
        trips = trips[trips["route_short_name"].isin(filter_in)]
    if filter_out:
        trips = trips[~trips["route_short_name"].isin(filter_out)]

    if trips.empty:
        logging.error("No trips remain after filtering. Check SERVICE_DAY and route filter lists.")
        sys.exit(1)

    logging.info("%d trips after filtering.", len(trips))

    # direction_id is optional in GTFS; default to "0" when absent.
    if "direction_id" not in trips.columns:
        trips["direction_id"] = "0"
    else:
        trips["direction_id"] = trips["direction_id"].fillna("0")

    trip_ids = set(trips["trip_id"].astype(str))
    dep = first_departures(gtfs["stop_times"], trip_ids)
    trips["trip_id"] = trips["trip_id"].astype(str)

    trips_dep = trips[["trip_id", "route_id", "direction_id"]].merge(dep, on="trip_id", how="inner")
    result = compute_headway_span(trips_dep)
    logging.info("Computed metrics for %d route(s).", len(result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logging.info("Written → %s", output_path)
    return result


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
    """Parse command-line arguments, defaulting to the config block values."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute route-level headway and span of service from a GTFS feed. "
            "Defaults come from the configuration block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-folder", type=Path, default=GTFS_FOLDER, help="Path to the GTFS folder."
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / OUTPUT_FILENAME, help="Output CSV path."
    )
    parser.add_argument(
        "--service-day",
        default=SERVICE_DAY,
        choices=_SERVICE_DAY_CHOICES,
        help="Service day to summarize ('weekday' or a single day name).",
    )
    parser.add_argument(
        "--service-date",
        default=SERVICE_DATE,
        metavar="YYYYMMDD",
        help="Explicit analysis date; overrides automatic date selection.",
    )
    parser.add_argument(
        "--filter-in",
        nargs="*",
        default=FILTER_IN_ROUTE_SHORT_NAMES,
        metavar="ROUTE_SHORT_NAME",
        help="Only keep these route_short_name values (empty = keep all).",
    )
    parser.add_argument(
        "--filter-out",
        nargs="*",
        default=FILTER_OUT_ROUTE_SHORT_NAMES,
        metavar="ROUTE_SHORT_NAME",
        help="Drop these route_short_name values (empty = drop none).",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(notebook_safe_argv(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the config block.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sentinels = {
        Path(r"Path\To\Your\GTFS_Folder"),
        Path(r"Path\To\Your\Output_Folder") / "headway_span_by_route.csv",
    }
    if args.gtfs_folder in sentinels or args.output in sentinels:
        logging.warning(
            "GTFS_FOLDER and/or OUTPUT_DIR are still placeholders. Update the configuration "
            "block or pass --gtfs-folder/--output before running."
        )
        return 2
    try:
        run(
            gtfs_folder=args.gtfs_folder,
            output_path=args.output,
            service_day=args.service_day,
            service_date=args.service_date,
            filter_in_route_short_names=args.filter_in,
            filter_out_route_short_names=args.filter_out,
        )
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the config block stays in charge.
    raise SystemExit(main())
