"""List the routes that interline (share vehicle blocks) in a GTFS feed.

Two routes interline when one vehicle block (a ``block_id`` on one service
date) runs trips on both. For each day type a representative date is picked
from calendar.txt / calendar_dates.txt (skipping federal holidays), the trips
running that date are grouped into blocks, and each block's trips are put in
order by scheduled start time. The script then reports which routes share
blocks and every point where a block's next trip is on a different route (a
hand-off), with the layover between the two trips.

``bus_block_exporter.py`` traces individual blocks minute by minute, and
``gtfs_route_diff.py`` flags routes whose interline partners changed between
two feeds; this script is the feed-wide list.

Inputs
------
- A GTFS feed folder or ``.zip`` with routes, trips (with ``block_id``) and
  stop_times, plus calendar.txt and/or calendar_dates.txt. stops.txt adds stop
  names to the hand-off table when present.

Outputs
-------
- ``interlined_routes_by_route.csv``: one row per route and day type — trips,
  blocks, blocks shared with other routes, and the routes it interlines with.
- ``interlined_route_pairs.csv``: one row per interlined route pair and day
  type — shared blocks, hand-offs in each direction, layover range, block_ids.
- ``interline_hand_offs.csv``: every hand-off — block, both trips, the arrival
  and departure times, the stops, and the layover in minutes.
- ``interlined_routes_runlog.txt``: run-log sidecar with the verbatim
  CONFIGURATION block and the service dates used.

Method
------
- Routes are identified by route_id. Two route_ids that share a short name
  are reported as separate routes, so a block moving between them is listed.
- Trips without a block_id are counted per route but cannot interline.
- Trips in frequencies.txt count once each; headways are not expanded.
- A negative layover means two trips on one block overlap in time, usually a
  block_id reused by more than one vehicle on that date; it is logged.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags)
and run from a shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# === BEGIN CONFIG ===

# GTFS feed: a folder of .txt files or the published .zip.
GTFS_PATH: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me
OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me

# Day types to analyze: "weekday" or a day name ("monday" … "sunday"). A
# representative date is picked for each from calendar.txt / calendar_dates.txt,
# skipping federal holidays.
DAY_TYPES: List[str] = ["weekday", "saturday", "sunday"]

# Analyze this one date instead of DAY_TYPES, as "YYYY-MM-DD" or "YYYYMMDD"
# (e.g. a holiday or a special-event day). Leave "" to use DAY_TYPES.
SERVICE_DATE: str = ""

# Output names (written inside OUTPUT_DIR).
ROUTES_FILENAME: str = r"interlined_routes_by_route.csv"
PAIRS_FILENAME: str = r"interlined_route_pairs.csv"
HAND_OFFS_FILENAME: str = r"interline_hand_offs.csv"
RUN_LOG_FILENAME: str = r"interlined_routes_runlog.txt"

# When True, a failed run-log write aborts the script so no output is left
# without its configuration record. Set False only for read-only destinations.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

REQUIRED_GTFS_FILES: List[str] = ["routes.txt", "trips.txt", "stop_times.txt"]

DAY_NAMES: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

# Day-type label used when SERVICE_DATE replaces DAY_TYPES.
SERVICE_DATE_LABEL: str = "service_date"

ROUTE_COLUMNS: List[str] = [
    "day_type",
    "service_date",
    "route",
    "route_id",
    "route_long_name",
    "trips",
    "trips_without_block_id",
    "blocks",
    "interlined_blocks",
    "interlines_with",
]
PAIR_COLUMNS: List[str] = [
    "day_type",
    "service_date",
    "route_a",
    "route_b",
    "route_a_id",
    "route_b_id",
    "shared_blocks",
    "hand_offs",
    "hand_offs_a_to_b",
    "hand_offs_b_to_a",
    "hand_offs_same_stop",
    "min_layover_min",
    "max_layover_min",
    "block_ids",
]
HAND_OFF_COLUMNS: List[str] = [
    "day_type",
    "service_date",
    "block_id",
    "from_route",
    "from_route_id",
    "from_trip_id",
    "from_headsign",
    "arrives",
    "from_end_stop_id",
    "from_end_stop_name",
    "to_route",
    "to_route_id",
    "to_trip_id",
    "to_headsign",
    "departs",
    "to_start_stop_id",
    "to_start_stop_name",
    "layover_min",
    "same_stop",
]

# =============================================================================
# HELPERS (canonical copies — see utils/)
# =============================================================================


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


# =============================================================================
# FUNCTIONS
# =============================================================================


class ConfigurationError(ValueError):
    """A setting is invalid (an unknown day type or a malformed date).

    Reported without a traceback; the script exits with code 2.
    """


def natural_sort_key(label: str) -> str:
    """Return a sort key that orders embedded numbers numerically ("2" < "10")."""
    return re.sub(r"\d+", lambda m: m.group().zfill(8), label)


def natural_sort_series(series: pd.Series) -> pd.Series:
    """Return *series* with text mapped through :func:`natural_sort_key` (a sort key).

    Numeric columns are returned unchanged, so ``sort_values(key=...)`` can mix
    text and number columns.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype(str).map(natural_sort_key)


def clean_text(series: pd.Series) -> pd.Series:
    """Return *series* as trimmed strings, with missing values as ``""``."""
    return series.fillna("").astype(str).str.strip()


def load_optional_table(gtfs_path: Path, file_name: str) -> Optional[pd.DataFrame]:
    """Load one optional GTFS file, returning ``None`` when it is absent or unreadable."""
    try:
        return load_gtfs_data(str(gtfs_path), files=(file_name,))[file_name[:-4]]
    except (OSError, ValueError) as exc:
        logging.info("%s not usable (%s) — continuing without it.", file_name, exc)
        return None


def parse_service_date(value: str) -> Optional[dt.date]:
    """Parse SERVICE_DATE (``YYYY-MM-DD`` or ``YYYYMMDD``); ``""`` gives ``None``.

    Raises:
        ConfigurationError: If *value* is set but is not a valid date.
    """
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ConfigurationError(
        f"SERVICE_DATE {text!r} is not a date; use YYYY-MM-DD (e.g. 2026-03-11) or YYYYMMDD."
    )


def normalize_day_types(day_types: Sequence[str]) -> List[str]:
    """Return *day_types* lower-cased and de-duplicated, in order.

    Raises:
        ConfigurationError: If a value is not ``"weekday"`` or a day name.
    """
    cleaned = list(dict.fromkeys(str(d).strip().lower() for d in day_types if str(d).strip()))
    unknown = [d for d in cleaned if d != "weekday" and d not in DAY_NAMES]
    if unknown:
        raise ConfigurationError(
            f"DAY_TYPES {unknown} not recognised; use 'weekday' or a day name "
            "('monday' … 'sunday')."
        )
    return cleaned


def resolve_service_days(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    day_types: Sequence[str],
    service_date: Optional[dt.date],
) -> dict[str, tuple[dt.date, set[str]]]:
    """Pick the date to analyze, and its active service_ids, for each day type.

    Args:
        calendar: Parsed calendar.txt or ``None``.
        calendar_dates: Parsed calendar_dates.txt or ``None``.
        day_types: Normalized day types (see :func:`normalize_day_types`).
        service_date: An explicit date that replaces *day_types*, or ``None``.

    Returns:
        Mapping of day-type label to (date, service_ids). Day types with no
        candidate dates are left out with a warning.

    Raises:
        ValueError: If the feed has no usable calendar, or no day type
            resolves to a date with service.
        ConfigurationError: If neither *day_types* nor *service_date* is set.
    """
    no_calendar = calendar is None or calendar.empty
    no_dates = calendar_dates is None or calendar_dates.empty
    if no_calendar and no_dates:
        raise ValueError(
            "The feed has no usable calendar.txt or calendar_dates.txt, so the trips that "
            "run on a given day cannot be identified."
        )
    active = expand_service_active_dates(calendar, calendar_dates)

    if service_date is not None:
        ids = service_ids_active_on(active, service_date)
        if not ids:
            raise ValueError(f"No service_id is active on SERVICE_DATE {service_date}.")
        logging.info(
            "Using SERVICE_DATE %s (%s): service_ids %s.",
            service_date,
            DAY_NAMES[service_date.weekday()],
            ", ".join(sorted(ids)),
        )
        return {SERVICE_DATE_LABEL: (service_date, ids)}

    if not day_types:
        raise ConfigurationError("Set DAY_TYPES (e.g. ['weekday']) or SERVICE_DATE.")
    years = {d.year for dates in active.values() for d in dates}
    holidays: set[dt.date] = set()
    if years:
        for year in range(min(years), max(years) + 2):
            holidays |= federal_holidays_observed(year)

    selection: dict[str, tuple[dt.date, set[str]]] = {}
    for day_type in day_types:
        try:
            selection[day_type] = representative_service_date(
                active, day_type, exclude_dates=holidays
            )
        except ValueError as exc:
            logging.warning("Skipping %s: %s", day_type, exc)
    if not selection:
        raise ValueError(f"None of DAY_TYPES {list(day_types)} has an active date in the feed.")
    return selection


def build_route_labels(routes: pd.DataFrame) -> pd.DataFrame:
    """Return one row per route_id with its display label and long name.

    The label is the route_short_name, else the route_long_name, else the
    route_id.

    Args:
        routes: Parsed routes.txt.

    Returns:
        DataFrame indexed by route_id with ``route`` (label) and
        ``route_long_name`` columns.
    """
    route_ids = clean_text(routes["route_id"])
    short = clean_text(routes["route_short_name"]) if "route_short_name" in routes else ""
    long_name = clean_text(routes["route_long_name"]) if "route_long_name" in routes else ""
    table = pd.DataFrame(
        {"route_id": route_ids, "short": short, "route_long_name": long_name}, index=routes.index
    )
    table["route"] = table["short"].where(table["short"] != "", table["route_long_name"])
    table["route"] = table["route"].where(table["route"] != "", table["route_id"])
    table = table.drop_duplicates("route_id", keep="first").set_index("route_id")
    return table[["route", "route_long_name"]]


def normalize_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """Return the trip columns this script uses as trimmed strings.

    Args:
        trips: Parsed trips.txt.

    Returns:
        DataFrame with ``trip_id``, ``route_id``, ``service_id``,
        ``block_id`` and ``trip_headsign`` (``""`` when the column is absent).
    """
    out = pd.DataFrame(index=trips.index)
    for col in ("trip_id", "route_id", "service_id", "block_id", "trip_headsign"):
        out[col] = clean_text(trips[col]) if col in trips.columns else ""
    return out.reset_index(drop=True)


def build_trip_ends(stop_times: pd.DataFrame, trip_ids: set[str]) -> pd.DataFrame:
    """Return each trip's first and last stop and its scheduled start and end.

    The start is the first stop's departure time (its arrival time when the
    departure is blank); the end is the last stop's arrival time (its
    departure time when the arrival is blank). Stops are ordered by
    ``stop_sequence``.

    Args:
        stop_times: Parsed stop_times.txt.
        trip_ids: Trips to keep.

    Returns:
        DataFrame with ``trip_id``, ``start_stop_id``, ``end_stop_id``,
        ``start_min`` and ``end_min`` (minutes past midnight; ``NaN`` when the
        time cannot be parsed).
    """
    columns = ["trip_id", "start_stop_id", "end_stop_id", "start_min", "end_min"]
    st = pd.DataFrame(
        {
            "trip_id": clean_text(stop_times["trip_id"]),
            "stop_id": clean_text(stop_times["stop_id"]),
            "arrival_time": clean_text(stop_times.get("arrival_time", pd.Series(dtype=str))),
            "departure_time": clean_text(stop_times.get("departure_time", pd.Series(dtype=str))),
            "seq": pd.to_numeric(stop_times["stop_sequence"], errors="coerce"),
        }
    )
    st = st[st["trip_id"].isin(trip_ids)]
    if st.empty:
        return pd.DataFrame(columns=columns)
    st = st.sort_values(["trip_id", "seq"], kind="stable")
    first = st.drop_duplicates("trip_id", keep="first").set_index("trip_id")
    last = st.drop_duplicates("trip_id", keep="last").set_index("trip_id")

    def to_minutes(primary: pd.Series, fallback: pd.Series) -> pd.Series:
        times = primary.where(primary != "", fallback)
        return pd.to_numeric(times.map(parse_time_to_minutes), errors="coerce")

    ends = pd.DataFrame(
        {
            "start_stop_id": first["stop_id"],
            "end_stop_id": last.loc[first.index, "stop_id"],
            "start_min": to_minutes(first["departure_time"], first["arrival_time"]),
            "end_min": to_minutes(
                last.loc[first.index, "arrival_time"], last.loc[first.index, "departure_time"]
            ),
        }
    )
    return ends.rename_axis("trip_id").reset_index()[columns]


def find_hand_offs(block_trips: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return each place a block's next trip is on a different route.

    Args:
        block_trips: Trips of one service date with a block_id, with
            ``route_id``, ``start_min``, ``end_min``, ``start_stop_id`` and
            ``end_stop_id`` columns. Trips without times are ignored here.

    Returns:
        A tuple of (hand-offs, overlapping blocks). Hand-offs have one row
        per consecutive trip pair whose route changes, with ``from_*`` /
        ``to_*`` trip columns and ``layover_min``. Overlapping blocks is the
        number of blocks with any two consecutive trips (same route or not)
        that overlap in time.
    """
    timed = block_trips.dropna(subset=["start_min", "end_min"])
    timed = timed.sort_values(["block_id", "start_min", "end_min", "trip_id"], kind="stable")
    keep = ["trip_id", "route_id", "trip_headsign", "end_stop_id", "end_min"]
    prev = timed.groupby("block_id", sort=False)[keep].shift(1)
    pairs = timed.assign(**{f"prev_{c}": prev[c] for c in keep})
    pairs = pairs[pairs["prev_trip_id"].notna()].copy()
    pairs["layover_min"] = pairs["start_min"] - pairs["prev_end_min"]
    overlapping = int(pairs.loc[pairs["layover_min"] < 0, "block_id"].nunique())

    hand_offs = pairs[pairs["route_id"] != pairs["prev_route_id"]]
    out = pd.DataFrame(
        {
            "block_id": hand_offs["block_id"],
            "from_route_id": hand_offs["prev_route_id"],
            "from_trip_id": hand_offs["prev_trip_id"],
            "from_headsign": hand_offs["prev_trip_headsign"],
            "from_end_stop_id": hand_offs["prev_end_stop_id"],
            "from_end_min": hand_offs["prev_end_min"],
            "to_route_id": hand_offs["route_id"],
            "to_trip_id": hand_offs["trip_id"],
            "to_headsign": hand_offs["trip_headsign"],
            "to_start_stop_id": hand_offs["start_stop_id"],
            "to_start_min": hand_offs["start_min"],
            "layover_min": hand_offs["layover_min"],
        }
    )
    out["same_stop"] = out["from_end_stop_id"] == out["to_start_stop_id"]
    return out.reset_index(drop=True), overlapping


def find_shared_blocks(block_trips: pd.DataFrame) -> pd.DataFrame:
    """Return every (block, route pair) where one block runs trips on both routes.

    Args:
        block_trips: Trips of one service date with a block_id.

    Returns:
        DataFrame with ``block_id``, ``route_id_x`` and ``route_id_y``; each
        unordered pair appears once per block, with ``route_id_x`` <
        ``route_id_y``.
    """
    members = block_trips[["block_id", "route_id"]].drop_duplicates()
    pairs = members.merge(members, on="block_id")
    return pairs[pairs["route_id_x"] < pairs["route_id_y"]].reset_index(drop=True)


def order_pair(a: str, b: str, labels: Mapping[str, str]) -> tuple[str, str]:
    """Return route_ids *a* and *b* ordered by display label, then route_id."""
    key_a = (natural_sort_key(labels.get(a, a)), a)
    key_b = (natural_sort_key(labels.get(b, b)), b)
    return (a, b) if key_a <= key_b else (b, a)


def summarize_pairs(
    shared: pd.DataFrame, hand_offs: pd.DataFrame, labels: Mapping[str, str]
) -> pd.DataFrame:
    """Return one row per interlined route pair for one service date.

    Args:
        shared: Output of :func:`find_shared_blocks`.
        hand_offs: Hand-offs from :func:`find_hand_offs`.
        labels: route_id → display label.

    Returns:
        Pair table with the :data:`PAIR_COLUMNS` after ``service_date``.
    """
    rows = []
    blocks_by_pair: dict[tuple[str, str], set[str]] = {}
    for block, x, y in zip(shared["block_id"], shared["route_id_x"], shared["route_id_y"]):
        blocks_by_pair.setdefault(order_pair(x, y, labels), set()).add(block)

    hand_offs_by_pair: dict[tuple[str, str], list[tuple[bool, float, bool]]] = {}
    for src, dst, gap, same in zip(
        hand_offs["from_route_id"],
        hand_offs["to_route_id"],
        hand_offs["layover_min"],
        hand_offs["same_stop"],
    ):
        pair = order_pair(src, dst, labels)
        hand_offs_by_pair.setdefault(pair, []).append((src == pair[0], float(gap), bool(same)))

    for (a, b), blocks in blocks_by_pair.items():
        found = hand_offs_by_pair.get((a, b), [])
        gaps = [gap for _, gap, _ in found]
        rows.append(
            {
                "route_a": labels.get(a, a),
                "route_b": labels.get(b, b),
                "route_a_id": a,
                "route_b_id": b,
                "shared_blocks": len(blocks),
                "hand_offs": len(found),
                "hand_offs_a_to_b": sum(1 for a_to_b, _, _ in found if a_to_b),
                "hand_offs_b_to_a": sum(1 for a_to_b, _, _ in found if not a_to_b),
                "hand_offs_same_stop": sum(1 for _, _, same in found if same),
                "min_layover_min": int(min(gaps)) if gaps else None,
                "max_layover_min": int(max(gaps)) if gaps else None,
                "block_ids": "; ".join(sorted(blocks, key=natural_sort_key)),
            }
        )
    out = pd.DataFrame(rows, columns=PAIR_COLUMNS[2:])
    out["min_layover_min"] = out["min_layover_min"].astype("Int64")
    out["max_layover_min"] = out["max_layover_min"].astype("Int64")
    if out.empty:
        return out
    return out.sort_values(
        ["route_a", "route_b", "route_a_id", "route_b_id"], key=natural_sort_series
    ).reset_index(drop=True)


def summarize_routes(
    day_trips: pd.DataFrame, shared: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per route running on one service date.

    Args:
        day_trips: Every trip of the date (with or without a block_id).
        shared: Output of :func:`find_shared_blocks` for the date.
        labels: Output of :func:`build_route_labels`.

    Returns:
        Route table with the :data:`ROUTE_COLUMNS` after ``service_date``.
    """
    label_map = labels["route"].to_dict()
    long_names = labels["route_long_name"].to_dict()
    partners: dict[str, set[str]] = {}
    interlined_blocks: dict[str, set[str]] = {}
    for block, x, y in zip(shared["block_id"], shared["route_id_x"], shared["route_id_y"]):
        partners.setdefault(x, set()).add(y)
        partners.setdefault(y, set()).add(x)
        interlined_blocks.setdefault(x, set()).add(block)
        interlined_blocks.setdefault(y, set()).add(block)

    rows = []
    for route_id, group in day_trips.groupby("route_id", sort=False):
        blocked = group[group["block_id"] != ""]
        partner_labels = sorted(
            (label_map.get(p, p) for p in partners.get(route_id, set())), key=natural_sort_key
        )
        rows.append(
            {
                "route": label_map.get(route_id, route_id),
                "route_id": route_id,
                "route_long_name": long_names.get(route_id, ""),
                "trips": len(group),
                "trips_without_block_id": len(group) - len(blocked),
                "blocks": blocked["block_id"].nunique(),
                "interlined_blocks": len(interlined_blocks.get(route_id, set())),
                "interlines_with": ", ".join(partner_labels),
            }
        )
    out = pd.DataFrame(rows, columns=ROUTE_COLUMNS[2:])
    if out.empty:
        return out
    return out.sort_values(["route", "route_id"], key=natural_sort_series).reset_index(drop=True)


def format_hand_offs(
    hand_offs: pd.DataFrame, labels: Mapping[str, str], stop_names: Mapping[str, str]
) -> pd.DataFrame:
    """Return the hand-off table in output form (labels, HH:MM times, stop names)."""
    out = hand_offs.copy()
    out["from_route"] = out["from_route_id"].map(lambda r: labels.get(r, r))
    out["to_route"] = out["to_route_id"].map(lambda r: labels.get(r, r))
    out["arrives"] = out["from_end_min"].map(minutes_to_hhmm)
    out["departs"] = out["to_start_min"].map(minutes_to_hhmm)
    out["from_end_stop_name"] = out["from_end_stop_id"].map(lambda s: stop_names.get(s, ""))
    out["to_start_stop_name"] = out["to_start_stop_id"].map(lambda s: stop_names.get(s, ""))
    out["layover_min"] = out["layover_min"].round().astype("Int64")
    out = out.sort_values(["block_id", "to_start_min"], key=natural_sort_series)
    return out[HAND_OFF_COLUMNS[2:]].reset_index(drop=True)


def _extract_config_from_text(source_text: str, source_label: str) -> str:
    """Return the text between the CONFIG markers in *source_text*.

    Same slicing as :func:`extract_config_block`, for source that is already
    in memory (a notebook cell) rather than on disk.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
    """
    lines = source_text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_label}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )
    return "\n".join(lines[begin_idx + 1 : end_idx])


def _notebook_path(ip: Any) -> Optional[str]:
    """Return the notebook file path from the IPython kernel namespace, or None.

    Tries the known frontend-specific variables in order:
      * ``__vsc_ipynb_file__``  – VS Code / Pylance Jupyter extension
      * ``__session__``         – JupyterLab ≥ 4 kernel session path

    Returns None when running in an environment that doesn't expose a path
    (ArcGIS Pro notebooks, classic Notebook, plain IPython console, etc.).
    """
    for var in ("__vsc_ipynb_file__", "__session__"):
        val = ip.user_ns.get(var)
        if val:
            return str(val)
    return None


def resolve_config_source() -> tuple[str, str]:
    """Return ``(config_text, source_label)`` for the running configuration.

    Resolution order:

    1. **Jupyter / IPython kernel** – walks the ``In`` cell history in reverse
       and uses the most recent cell that contains both CONFIG markers, so
       pasting this whole script into a cell still yields a faithful run log.
       ``__file__`` does not exist in a kernel, which is why this comes first.
    2. **Plain script / imported module** – reads ``__file__``.

    Raises:
        RuntimeError: If neither source can be located.
        ValueError: If the located source has no CONFIG markers.
        OSError: If the source file cannot be read.
    """
    # IPython registers itself in sys.modules when a kernel starts; avoid a
    # direct import so the dependency stays optional. get_ipython() returns
    # None outside a live kernel.
    _ipython = sys.modules.get("IPython")
    ip = _ipython.get_ipython() if _ipython is not None else None  # type: ignore[attr-defined]
    if ip is not None:
        history: List[str] = ip.user_ns.get("In", [])
        for cell in reversed(history):
            if CONFIG_BEGIN_MARKER in cell and CONFIG_END_MARKER in cell:
                label = _notebook_path(ip) or "<Jupyter cell>"
                return _extract_config_from_text(cell, label), label

    file_attr: Optional[str] = globals().get("__file__")
    if file_attr is not None:
        source_path = Path(file_attr).resolve()
        return extract_config_block(source_path), str(source_path)

    raise RuntimeError(
        "Cannot locate the script source for the run log: __file__ is not defined "
        "and no notebook cell containing the CONFIG markers was found. Run the cell "
        "that holds the CONFIGURATION block (or the whole script) before writing output."
    )


def write_run_log(
    output_dir: Path, summary_lines: List[str], config_text: str, source_label: str
) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    Args:
        output_dir: Folder that receives ``RUN_LOG_FILENAME``.
        summary_lines: Human-readable lines describing this run.
        config_text: The CONFIG block, from :func:`resolve_config_source`.
        source_label: Where the config came from (file path or notebook).

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / RUN_LOG_FILENAME
    lines: List[str] = [
        "=" * 72,
        "INTERLINED ROUTES EXPORTER RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_label}",
        "",
        "-" * 72,
        "RUN SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim)",
        "-" * 72,
        CONFIG_BEGIN_MARKER,
        config_text,
        CONFIG_END_MARKER,
        "",
    ]
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logging.error("Could not write run log '%s': %s", log_path, exc)
        return False
    logging.info("Run log written → %s", log_path)
    return True


# =============================================================================
# ORCHESTRATION
# =============================================================================


def run(
    gtfs_path: Path | None = None,
    output_dir: Path | None = None,
    day_types: Optional[Sequence[str]] = None,
    service_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Find interlined routes for each day type and write the three CSVs.

    Unset args fall back to the CONFIGURATION block, so ``m.GTFS_PATH = ...;
    m.run()`` works after a plain import.

    Returns:
        ``{"routes": ..., "pairs": ..., "hand_offs": ...}``, the tables as
        written to ROUTES_FILENAME, PAIRS_FILENAME and HAND_OFFS_FILENAME.

    Raises:
        ConfigurationError: If DAY_TYPES or SERVICE_DATE is invalid.
        OSError: If the feed is missing files, or the run log is required but
            cannot be written.
        ValueError: If the feed has no block_id values, no usable calendar, or
            no service on the requested days.
    """
    gtfs_path = GTFS_PATH if gtfs_path is None else Path(gtfs_path)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    day_types = normalize_day_types(DAY_TYPES if day_types is None else day_types)
    chosen_date = parse_service_date(SERVICE_DATE if service_date is None else service_date)

    # Locate the CONFIG block first so an untraceable run aborts before any
    # output is written (REQUIRE_RUN_LOG), not after.
    try:
        config_text, source_label = resolve_config_source()
    except (OSError, ValueError, RuntimeError) as exc:
        if REQUIRE_RUN_LOG:
            raise OSError(f"Run log source unavailable and REQUIRE_RUN_LOG is True: {exc}") from exc
        logging.warning("Run log will not record the configuration: %s", exc)
        config_text = "(config block unavailable — see warning in the console log)"
        source_label = "<unknown>"

    gtfs = load_gtfs_data(str(gtfs_path), files=REQUIRED_GTFS_FILES)
    calendar = load_optional_table(gtfs_path, "calendar.txt")
    calendar_dates = load_optional_table(gtfs_path, "calendar_dates.txt")
    stops = load_optional_table(gtfs_path, "stops.txt")
    frequencies = load_optional_table(gtfs_path, "frequencies.txt")

    trips = normalize_trips(gtfs["trips"])
    if not (trips["block_id"] != "").any():
        raise ValueError(
            "trips.txt has no block_id values, so it does not say which trips share a "
            "vehicle and interlining cannot be derived from this feed."
        )
    labels = build_route_labels(gtfs["routes"])
    label_map: dict[str, str] = labels["route"].to_dict()
    stop_names: dict[str, str] = {}
    if stops is not None and "stop_id" in stops.columns and "stop_name" in stops.columns:
        stop_names = dict(zip(clean_text(stops["stop_id"]), clean_text(stops["stop_name"])))

    day_selection = resolve_service_days(calendar, calendar_dates, day_types, chosen_date)
    selected_ids = set().union(*(ids for _, ids in day_selection.values()))
    trips = trips[trips["service_id"].isin(selected_ids)].reset_index(drop=True)
    if frequencies is not None and "trip_id" in frequencies.columns:
        if trips["trip_id"].isin(set(clean_text(frequencies["trip_id"]))).any():
            logging.warning(
                "Some trips are headway-based (frequencies.txt); each counts as one trip here."
            )

    blocked_ids = set(trips.loc[trips["block_id"] != "", "trip_id"])
    trips = trips.merge(build_trip_ends(gtfs["stop_times"], blocked_ids), on="trip_id", how="left")

    route_tables: List[pd.DataFrame] = []
    pair_tables: List[pd.DataFrame] = []
    hand_off_tables: List[pd.DataFrame] = []
    day_lines: List[str] = []
    for day_type, (date, service_ids) in day_selection.items():
        day_trips = trips[trips["service_id"].isin(service_ids)]
        block_trips = day_trips[day_trips["block_id"] != ""]
        untimed = int(block_trips["start_min"].isna().sum())
        if untimed:
            logging.warning(
                "%s: %d trip(s) with a block_id have no usable first/last stop time; they "
                "count toward shared blocks but not hand-offs.",
                day_type,
                untimed,
            )
        shared = find_shared_blocks(block_trips)
        hand_offs, overlapping = find_hand_offs(block_trips)
        if overlapping:
            logging.warning(
                "%s: %d block(s) have consecutive trips that overlap in time — the block_id "
                "may be used by more than one vehicle that day (see negative layover_min).",
                day_type,
                overlapping,
            )
        stamp = {"day_type": day_type, "service_date": date.isoformat()}
        routes_out = summarize_routes(day_trips, shared, labels)
        pairs_out = summarize_pairs(shared, hand_offs, label_map)
        hand_offs_out = format_hand_offs(hand_offs, label_map, stop_names)
        route_tables.append(routes_out.assign(**stamp)[ROUTE_COLUMNS])
        pair_tables.append(pairs_out.assign(**stamp)[PAIR_COLUMNS])
        hand_off_tables.append(hand_offs_out.assign(**stamp)[HAND_OFF_COLUMNS])

        n_interlined = int((routes_out["interlined_blocks"] > 0).sum())
        logging.info(
            "%s %s: %d route(s), %d interlined; %d route pair(s); %d hand-off(s).",
            day_type,
            date.isoformat(),
            len(routes_out),
            n_interlined,
            len(pairs_out),
            len(hand_offs_out),
        )
        day_lines.append(
            f"  {day_type}: {date.isoformat()} ({date:%A}) — service_ids "
            f"{', '.join(sorted(service_ids)) or '-'}; {len(routes_out)} route(s), "
            f"{n_interlined} interlined, {len(pairs_out)} pair(s), "
            f"{len(hand_offs_out)} hand-off(s), {len(block_trips)} "
            f"trip(s) with a block_id, {len(day_trips) - len(block_trips)} without"
        )

    tables = {
        "routes": pd.concat(route_tables, ignore_index=True),
        "pairs": pd.concat(pair_tables, ignore_index=True),
        "hand_offs": pd.concat(hand_off_tables, ignore_index=True),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "routes": output_dir / ROUTES_FILENAME,
        "pairs": output_dir / PAIRS_FILENAME,
        "hand_offs": output_dir / HAND_OFFS_FILENAME,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
        logging.info("Wrote %d row(s) → %s", len(tables[key]), path)

    summary_lines = [
        f"GTFS feed:          {gtfs_path}",
        f"Day types:          {', '.join(day_types) or '-'}",
        f"Service date:       {chosen_date.isoformat() if chosen_date else '- (day types used)'}",
        "Dates analyzed:",
        *day_lines,
        f"Routes CSV:         {paths['routes']}",
        f"Pairs CSV:          {paths['pairs']}",
        f"Hand-offs CSV:      {paths['hand_offs']}",
    ]
    if not write_run_log(output_dir, summary_lines, config_text, source_label) and REQUIRE_RUN_LOG:
        raise OSError(
            f"Run log could not be written to '{output_dir}' and REQUIRE_RUN_LOG is True."
        )

    logging.info(
        "Interlined route export complete — %d route pair row(s), %d hand-off(s).",
        len(tables["pairs"]),
        len(tables["hand_offs"]),
    )
    return tables


# =============================================================================
# MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; every flag defaults to its CONFIGURATION constant."""
    parser = argparse.ArgumentParser(
        description=(
            "List the routes that share vehicle blocks (interline) in a GTFS feed, per day "
            "type. Defaults come from the configuration block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-path", type=Path, default=GTFS_PATH, help="GTFS feed folder or .zip archive."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for the CSVs and run log."
    )
    parser.add_argument(
        "--day-types",
        nargs="*",
        default=DAY_TYPES,
        metavar="DAY",
        help="Day types to analyze: 'weekday' or day names.",
    )
    parser.add_argument(
        "--service-date",
        default=SERVICE_DATE,
        help="Analyze this date (YYYY-MM-DD or YYYYMMDD) instead of --day-types.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the config block.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders or a setting is invalid.
    """
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sentinels = {Path(r"Path\To\Your\GTFS_Folder"), Path(r"Path\To\Your\Output_Folder")}
    if args.gtfs_path in sentinels or args.output_dir in sentinels:
        logging.warning(
            "GTFS_PATH and/or OUTPUT_DIR are still placeholders. Update the configuration "
            "block or pass --gtfs-path/--output-dir before running."
        )
        return 2
    try:
        run(
            gtfs_path=args.gtfs_path,
            output_dir=args.output_dir,
            day_types=args.day_types,
            service_date=args.service_date,
        )
    except ConfigurationError as exc:
        logging.error("%s", exc)
        return 2
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


# Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
# injected argv away from argparse so the config block stays in charge.
if __name__ == "__main__":
    raise SystemExit(main())
