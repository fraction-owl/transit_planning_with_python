"""Total scheduled revenue and nonrevenue vehicle hours from GTFS plus TODS.

Reads a GTFS feed and, optionally, its TODS supplement files, rebuilds each
vehicle block's day (pull-out, revenue trips, layovers, deadheads, pull-in) and
totals the hours by service set (the GTFS services that run together on a date),
by day type (Weekday / Saturday / Sunday / Holiday) and by reporting period, such
as a month. TODS supplements are applied by the TODS 2.1 rules (delete, update, add; trips on
a deleted route are ignored) before anything is counted. Until TODS_MOVEMENTS_COMPLETE confirms
the TODS holds every block's movements, only revenue trip hours and layover are reported.

Blocks are rebuilt once for each service set that runs on some date, from the
trips of every service in that set: GTFS defines a block by its block_id and
the days its trips share, so one block can join trips of different service_ids.

Each block's time is split into revenue trip hours (first departure to last
arrival of each revenue trip), layover (the vehicle waits at one place between
movements), TODS pull-out, pull-in and deadhead trips (other TODS_trip_type
values are totalled as other nonrevenue), and unresolved gaps, where the vehicle
ends one movement somewhere other than where the next begins with no scheduled
movement in between. Gaps are measured from the end of the block's latest-ending earlier
movement, so time any movement covers is never layover. Vehicle revenue hours add layover
to revenue trip hours when LAYOVER_IN_REVENUE_HOURS is True; vehicle hours run from
pull-out to pull-in. A trip that starts before an earlier trip of its block ends is an
overlap: one vehicle cannot run both, so vehicle revenue, nonrevenue and vehicle hours are
left blank for that block and every total that includes it until its block_id is fixed.

A date's day type is Weekday, Saturday or Sunday, or Holiday for HOLIDAY_DATES (and optionally
U.S. federal holidays) and, with the GTFS calendar, dates running a service SERVICE_DAY_TYPES
calls Holiday. Period totals count each date's services from the GTFS calendar (calendar.txt,
calendar_dates.txt and their TODS supplements) or, with PERIOD_CALENDAR = "config", the
services of the date's day type: those SERVICE_DAY_TYPES lists (they must run together on some
date), or else the day type's typical set: the service set that runs on the most dates of that
type in the feed's calendar (for Holiday, only sets run mostly on holidays); other sets are
logged. Period days are also counted by the typical day whose service they run: a Christmas
on Sunday service is a holiday and a Sunday schedule day.

Inputs
------
- GTFS_PATH: GTFS feed, as a folder or ``.zip``. A feed with frequency-based
  trips (rows in frequencies.txt) stops the run: their repetitions are not expanded.
- TODS_PATH (optional): TODS 2.1 supplement files for that feed, as a folder or ``.zip``
  (for example from ``tsde_to_tods.py``); TODS v1 deadhead tables stop the run.

Outputs
-------
- ``service_hours_by_block.csv``: hours per block and service set, with QA counts.
- ``service_hours_by_service.csv``: hours for one day of each service set.
- ``service_hours_by_day_type.csv``: hours on a typical day of each day type.
- ``service_hours_by_period.csv``: total hours, day counts and service sets per period.
- ``service_hours.xlsx``: the same period totals as Excel formulas, traceable
  through Calendar, Services and Blocks sheets to each trip's start and end time,
  with a layover-as-revenue switch, so totals can be checked and adjusted by hand.
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags)
and run from a shell, ArcGIS Pro's Python window, or a Jupyter notebook.

See Also:
    - ``tsde_to_tods.py``
    - ``gtfs_route_features_gpd.py``
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import os
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.workbook.properties import CalcProperties

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS feed (folder or .zip) and its TODS supplement files (folder or .zip).
# Set TODS_PATH = None to report revenue trip hours and layover only.
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder_or.zip"
TODS_PATH: Optional[str] = r"Path\To\Your\TODS_Folder_or.zip"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Set True only once the TODS files hold every block's pull-out, pull-in and
# deadhead trips. Until then, pull-out, pull-in, deadhead, nonrevenue and vehicle
# hours are left blank, because a movement missing from TODS would count as zero.
# The log reports how many blocks have TODS movements, to check before confirming.
TODS_MOVEMENTS_COMPLETE: bool = False

BLOCK_FILENAME: str = "service_hours_by_block.csv"
SERVICE_FILENAME: str = "service_hours_by_service.csv"
DAY_TYPE_FILENAME: str = "service_hours_by_day_type.csv"
PERIOD_FILENAME: str = "service_hours_by_period.csv"
# Formula workbook: period totals as Excel formulas back to each trip's times.
WORKBOOK_FILENAME: str = "service_hours.xlsx"

# Reporting periods as (name, first date, last date), dates as YYYYMMDD.
PERIODS: List[tuple[str, str, str]] = [("November 2026", "20261101", "20261130")]

# Which services run on each date of a period:
#   "gtfs"   - the GTFS calendar (with TODS calendar supplements), date by date.
#   "config" - each date's day type: Weekday / Saturday / Sunday, or Holiday for
#              the dates below; the services of that day type run (from
#              SERVICE_DAY_TYPES, or else the set that runs on the most dates of
#              that day type in the GTFS calendar).
PERIOD_CALENDAR: str = "gtfs"

# service_id -> day type(s): "Weekday", "Saturday", "Sunday", "Holiday", or a
# comma-separated list such as "Saturday, Sunday". The services listed under one
# day type must run together on some date of the GTFS calendar. With the GTFS
# calendar, a date running a Holiday service counts as a holiday. Empty: config
# mode runs each day type's typical service set.
SERVICE_DAY_TYPES: dict[str, str] = {}

# Holiday dates (YYYYMMDD) and whether to add the observed U.S. federal
# holidays; these dates count as holidays in both calendar modes.
# PERIOD_CALENDAR = "config" only: whose services run on a holiday (for example
# "Sunday" if holidays run the Sunday schedule).
HOLIDAY_DATES: List[str] = []
USE_FEDERAL_HOLIDAYS: bool = True
HOLIDAY_SERVICE_DAY_TYPE: str = "Holiday"

# Count layover/recovery time as revenue time (True) or as nonrevenue time
# (False). It is always reported separately as well; set this to match the
# definition you report against.
LAYOVER_IN_REVENUE_HOURS: bool = True

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

RUN_LOG_FILENAME = "scheduled_service_hours_runlog.txt"
DAY_TYPES = ("Weekday", "Saturday", "Sunday", "Holiday")
WEEKDAY_LABELS = ("Weekday",) * 5 + ("Saturday", "Sunday")
# TODS_trip_type -> hour category; any other non-blank type is "other nonrevenue".
NONREVENUE_CATEGORIES = {
    "pull-out": "pull_out",
    "pull-back": "pull_in",
    "pull-in": "pull_in",
    "deadhead": "deadhead",
}
COMPONENTS = (
    "revenue_trip_hours",
    "layover_hours",
    "pull_out_hours",
    "pull_in_hours",
    "deadhead_hours",
    "other_nonrevenue_hours",
    "unresolved_gap_hours",
)
# Hour column -> the movement/gap category whose seconds it totals.
CATEGORY_OF = {
    "revenue_trip_hours": "revenue",
    "layover_hours": "layover",
    "pull_out_hours": "pull_out",
    "pull_in_hours": "pull_in",
    "deadhead_hours": "deadhead",
    "other_nonrevenue_hours": "other_nonrevenue",
    "unresolved_gap_hours": "unresolved_gap",
}
TODS_COMPONENTS = ("pull_out_hours", "pull_in_hours", "deadhead_hours", "other_nonrevenue_hours")
DERIVED = ("vehicle_revenue_hours", "nonrevenue_hours", "vehicle_hours")
_PLACEHOLDER_MARKERS = ("path\\to\\", "path/to/")


@dataclass
class Config:
    """Resolved settings for one run (mirrors the CONFIGURATION block)."""

    gtfs_path: Path
    tods_path: Optional[Path]
    output_dir: Path
    periods: List[tuple[str, str, str]] = field(default_factory=lambda: list(PERIODS))
    period_calendar: str = PERIOD_CALENDAR
    service_day_types: dict[str, str] = field(default_factory=lambda: dict(SERVICE_DAY_TYPES))
    holiday_dates: List[str] = field(default_factory=lambda: list(HOLIDAY_DATES))
    use_federal_holidays: bool = USE_FEDERAL_HOLIDAYS
    holiday_service_day_type: str = HOLIDAY_SERVICE_DAY_TYPE
    layover_in_revenue_hours: bool = LAYOVER_IN_REVENUE_HOURS
    tods_movements_complete: bool = TODS_MOVEMENTS_COMPLETE


# =============================================================================
# REUSABLE HELPERS (verbatim copies of utils/ canonical versions)
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
# LOADING AND TODS SUPPLEMENTS
# =============================================================================


def feed_members(path: Path) -> set[str]:
    """Return the file names present in a feed folder or ``.zip`` (basenames)."""
    if path.is_dir():
        return {p.name for p in path.iterdir() if p.is_file()}
    with zipfile.ZipFile(path) as archive:
        return {Path(name).name for name in archive.namelist() if not name.endswith("/")}


TODS_V1_TABLES = ("deadheads.txt", "deadhead_times.txt")


def reject_tods_v1(path: Path) -> None:
    """Stop when a TODS input holds TODS v1 deadhead tables.

    This script reads TODS 2.1 supplements (pull-outs, pull-ins and deadheads
    as ``trips_supplement.txt`` rows). A v1 folder would otherwise load without
    its deadheads and report plausible but incomplete vehicle hours.

    Raises:
        ValueError: If *path* contains ``deadheads.txt`` or ``deadhead_times.txt``.
    """
    found = sorted(set(TODS_V1_TABLES) & feed_members(path))
    if found:
        raise ValueError(
            f"'{path}' contains TODS v1 deadhead tables ({', '.join(found)}). This script "
            "expects TODS 2.1 supplements (trips_supplement.txt, stop_times_supplement.txt); "
            "convert the files to TODS 2.1 first."
        )


def load_tables(path: Path, wanted: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Load the files of *wanted* that exist in the feed, as all-text DataFrames.

    Raises:
        OSError: If *path* does not exist.
    """
    if not path.exists():
        raise OSError(f"The path '{path}' does not exist.")
    present = [name for name in wanted if name in feed_members(path)]
    if not present:
        return {}
    tables = load_gtfs_data(str(path), files=present)
    return {name: df.fillna("") for name, df in tables.items()}


def apply_supplement(
    base: Optional[pd.DataFrame], supplement: Optional[pd.DataFrame], keys: Sequence[str]
) -> pd.DataFrame:
    """Apply one TODS supplement file to its GTFS table (TODS 2.1 evaluation rules).

    A row whose primary key matches the GTFS table deletes that row when
    ``TODS_delete`` is ``1`` and otherwise overwrites it with the row's
    non-empty values; a row with a new key is added whole.

    Args:
        base: The GTFS table (``None`` or empty if the feed lacks it).
        supplement: The ``*_supplement.txt`` table, or ``None``.
        keys: The GTFS file's primary-key columns.

    Returns:
        The supplemented table, all values as text.

    Raises:
        ValueError: If the supplement lacks a key column or both deletes and
            re-adds the same primary key.
    """
    base = pd.DataFrame(columns=list(keys)) if base is None else base.fillna("")
    if supplement is None or supplement.empty:
        return base.copy()
    supplement = supplement.fillna("")
    missing = [k for k in keys if k not in supplement.columns]
    if missing:
        raise ValueError(f"Supplement file is missing primary-key column(s): {missing}")
    rows: dict[tuple, dict[str, Any]] = {
        tuple(r[k] for k in keys): r for r in base.to_dict("records")
    }
    order = list(rows)
    deleted: set[tuple] = set()
    written: set[tuple] = set()
    for record in supplement.to_dict("records"):
        key = tuple(record[k] for k in keys)
        values = {c: v for c, v in record.items() if c != "TODS_delete"}
        if str(record.get("TODS_delete", "")).strip() == "1":
            if key in rows:
                deleted.add(key)
            else:
                logging.warning("TODS_delete for %s matches no GTFS row; ignored.", key)
            continue
        written.add(key)
        if key in rows:
            rows[key].update({c: v for c, v in values.items() if v != ""})
        else:
            rows[key] = values
            order.append(key)
    if deleted & written:
        raise ValueError(f"Supplement both deletes and re-adds key(s): {sorted(deleted & written)}")
    out = pd.DataFrame([rows[k] for k in order if k not in deleted])
    return out.reindex(columns=list(dict.fromkeys([*base.columns, *out.columns]))).fillna("")


def drop_trips_on_deleted_routes(
    trips: pd.DataFrame, routes_before: pd.DataFrame, routes_after: pd.DataFrame
) -> pd.DataFrame:
    """Drop trips whose route a TODS ``routes_supplement.txt`` deletes.

    TODS ignores trips that reference a deleted route unless they are
    reassigned, so pass *trips* after ``trips_supplement.txt`` has been
    applied: a trip moved to a surviving route keeps its new ``route_id`` and
    stays. Each dropped trip is counted in a warning.

    Args:
        trips: Trips table with a ``route_id`` column, already supplemented.
        routes_before: ``routes.txt`` before ``routes_supplement.txt``.
        routes_after: The same table after :func:`apply_supplement`.

    Returns:
        *trips* without the rows on deleted routes.
    """
    deleted = set(routes_before.get("route_id", pd.Series(dtype=str)).astype(str)) - set(
        routes_after.get("route_id", pd.Series(dtype=str)).astype(str)
    )
    if not deleted:
        return trips
    on_deleted = trips["route_id"].astype(str).isin(deleted)
    if on_deleted.any():
        logging.warning(
            "Ignoring %d trip(s) on route(s) deleted in routes_supplement.txt (%s); "
            "reassign them to another route in trips_supplement.txt to keep them.",
            int(on_deleted.sum()),
            ", ".join(sorted(deleted)),
        )
    return trips.loc[~on_deleted]


@dataclass
class Schedule:
    """Supplemented trips with their timing, plus calendar and stop locations."""

    movements: pd.DataFrame
    active_dates: dict[str, set[dt.date]]
    stop_places: dict[str, tuple[str, str]]
    tods_supplied: bool


def _seconds(value: str) -> Optional[int]:
    """Parse a GTFS ``H:MM:SS`` time into seconds; blank or malformed gives None."""
    parts = str(value).strip().split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    h, m, s = (int(p) for p in parts)
    return h * 3600 + m * 60 + s


def build_movements(trips: pd.DataFrame, stop_times: pd.DataFrame) -> pd.DataFrame:
    """Return one row per trip: block, category, start/end seconds and end stops.

    Raises:
        ValueError: If a trip has no stop_times or no time at its first/last stop.
    """
    st = stop_times.loc[stop_times["trip_id"].isin(set(trips["trip_id"]))].copy()
    st["_seq"] = pd.to_numeric(st["stop_sequence"], errors="raise")
    st = st.sort_values(["trip_id", "_seq"])
    first = st.groupby("trip_id").first()
    last = st.groupby("trip_id").last()
    tods_type = trips["TODS_trip_type"] if "TODS_trip_type" in trips.columns else ""
    out = pd.DataFrame(
        {
            "trip_id": trips["trip_id"],
            "service_id": trips["service_id"],
            "block_id": trips["block_id"] if "block_id" in trips.columns else "",
            "route_id": trips["route_id"],
            "tods_trip_type": tods_type,
        }
    )
    out["tods_trip_type"] = out["tods_trip_type"].astype(str).str.strip()
    missing = sorted(set(out["trip_id"]) - set(first.index))
    if missing:
        raise ValueError(f"{len(missing)} trip(s) have no stop_times, e.g. {missing[:3]}.")
    out["first_stop"] = out["trip_id"].map(first["stop_id"])
    out["last_stop"] = out["trip_id"].map(last["stop_id"])
    starts = first["departure_time"].where(first["departure_time"] != "", first["arrival_time"])
    ends = last["arrival_time"].where(last["arrival_time"] != "", last["departure_time"])
    out["start"] = out["trip_id"].map(starts.map(_seconds))
    out["end"] = out["trip_id"].map(ends.map(_seconds))
    bad = out.loc[out["start"].isna() | out["end"].isna(), "trip_id"].tolist()
    if bad:
        raise ValueError(f"{len(bad)} trip(s) lack a first/last stop time, e.g. {bad[:3]}.")
    out["category"] = [
        "revenue" if not t else NONREVENUE_CATEGORIES.get(t.lower(), "other_nonrevenue")
        for t in out["tods_trip_type"]
    ]
    return out.reset_index(drop=True)


def load_schedule(gtfs_path: Path, tods_path: Optional[Path]) -> Schedule:
    """Load GTFS (and TODS), apply the supplements and time every trip.

    Raises:
        OSError: If an input path is missing.
        ValueError: If required GTFS tables are missing or inconsistent, the
            feed has frequency-based trips, the TODS input holds TODS v1
            deadhead tables, or the supplements leave no trips.
    """
    gtfs = load_tables(
        gtfs_path,
        (
            "trips.txt",
            "stop_times.txt",
            "stops.txt",
            "routes.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "frequencies.txt",
        ),
    )
    for name in ("trips", "stop_times"):
        if name not in gtfs:
            raise ValueError(f"GTFS feed '{gtfs_path}' has no {name}.txt.")
    frequencies = gtfs.get("frequencies")
    if frequencies is not None and not frequencies.empty:
        # Each frequencies.txt row repeats a template trip; counting the template
        # once would undercount trips and hours, so stop rather than guess.
        ids = sorted(set(frequencies.get("trip_id", pd.Series(dtype=str))))
        listed = ", ".join(ids[:5]) + (", ..." if len(ids) > 5 else "")
        raise ValueError(
            f"GTFS feed '{gtfs_path}' has frequency-based trips ({listed}) in frequencies.txt, "
            "which this script does not expand: each would count as a single trip. Expand them "
            "into individually scheduled trips in trips.txt and stop_times.txt, then rerun."
        )
    tods: dict[str, pd.DataFrame] = {}
    if tods_path is not None:
        tods = load_tables(
            tods_path,
            (
                "trips_supplement.txt",
                "stop_times_supplement.txt",
                "stops_supplement.txt",
                "routes_supplement.txt",
                "calendar_supplement.txt",
                "calendar_dates_supplement.txt",
            ),
        )
        reject_tods_v1(tods_path)
        if not tods:
            raise ValueError(f"No TODS supplement files found in '{tods_path}'.")
    trips = apply_supplement(gtfs["trips"], tods.get("trips_supplement"), ["trip_id"])
    routes = apply_supplement(gtfs.get("routes"), tods.get("routes_supplement"), ["route_id"])
    trips = drop_trips_on_deleted_routes(
        trips, gtfs.get("routes", pd.DataFrame(columns=["route_id"])), routes
    )
    if trips.empty:
        raise ValueError(
            f"No trips remain after applying the TODS supplements in '{tods_path}': every trip "
            "was deleted or is on a route deleted in routes_supplement.txt."
        )
    stop_times = apply_supplement(
        gtfs["stop_times"], tods.get("stop_times_supplement"), ["trip_id", "stop_sequence"]
    )
    stops = apply_supplement(gtfs.get("stops"), tods.get("stops_supplement"), ["stop_id"])
    calendar = apply_supplement(
        gtfs.get("calendar"), tods.get("calendar_supplement"), ["service_id"]
    )
    calendar_dates = apply_supplement(
        gtfs.get("calendar_dates"), tods.get("calendar_dates_supplement"), ["service_id", "date"]
    )
    active = expand_service_active_dates(
        calendar if not calendar.empty else None,
        calendar_dates if not calendar_dates.empty else None,
    )
    places = {
        str(r["stop_id"]): (str(r.get("stop_lat", "")), str(r.get("stop_lon", "")))
        for r in stops.to_dict("records")
    }
    return Schedule(build_movements(trips, stop_times), active, places, tods_path is not None)


# =============================================================================
# HOURS
# =============================================================================


def _coordinates(place: Optional[tuple[str, str]]) -> Optional[tuple[float, float]]:
    """Parse a (stop_lat, stop_lon) text pair; None when missing or not numeric."""
    if place is None:
        return None
    try:
        lat, lon = float(place[0]), float(place[1])
    except ValueError:
        return None
    return (lat, lon) if math.isfinite(lat) and math.isfinite(lon) else None


def same_place(a: str, b: str, places: Mapping[str, tuple[str, str]]) -> bool:
    """Return True if stops *a* and *b* are one place (same id or same coordinates).

    Coordinates are compared as numbers, so ``38.8`` and ``38.800000`` match,
    while stops at different coordinates (such as nearby bays) stay apart.
    """
    if a == b:
        return True
    pa = _coordinates(places.get(a))
    return pa is not None and pa == _coordinates(places.get(b))


def expand_by_service_set(movements: pd.DataFrame, sets: Sequence[tuple[str, ...]]) -> pd.DataFrame:
    """Return the movements of each service set, labelled in a ``service_set`` column.

    A trip appears once for every set that includes its service_id, so each set's
    blocks can be rebuilt from all the trips that run together on its dates.

    Args:
        movements: Output of :func:`build_movements`.
        sets: Service sets, each a sorted tuple of the service_ids running on a date.
    """
    parts = [
        movements.loc[movements["service_id"].isin(members)].assign(service_set="|".join(members))
        for members in sets
    ]
    if not parts:
        return movements.iloc[0:0].assign(service_set="")
    return pd.concat(parts, ignore_index=True)


def block_hours(movements: pd.DataFrame, places: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """Split each block's service day into the hour components.

    Trips are grouped by service set and block_id, so a block's trips of different
    service_ids that run on the same dates form one vehicle day. Trips with a blank
    block_id each form their own block. The vehicle is free once every earlier
    movement has ended, at the stop where the latest-ending one finished: a gap from
    then to the next movement is layover if the vehicle stays put and unresolved if
    it changes place, so time any movement covers is never layover. A movement that
    starts before the vehicle is free is counted in ``overlaps``: one vehicle cannot
    run both, so :func:`add_derived` leaves such a block's vehicle-level hours blank.

    Args:
        movements: Output of :func:`expand_by_service_set`.
        places: stop_id -> (stop_lat, stop_lon).

    Returns:
        One row per (service set, block_id), the set in ``service_ids``, with the
        COMPONENTS (hours) and ``revenue_trips``, ``nonrevenue_trips``, ``routes``,
        ``start``, ``end``, ``has_tods``, ``overlaps`` and ``location_jumps``.
    """
    keyed = movements.assign(
        _block=movements["block_id"].where(
            movements["block_id"] != "", "trip:" + movements["trip_id"]
        )
    )
    rows = []
    for (service_set, block), group in keyed.groupby(["service_set", "_block"], sort=True):
        legs = group.sort_values(["start", "end"]).to_dict("records")
        seconds = Counter()
        overlaps = jumps = 0
        for leg in legs:
            seconds[leg["category"]] += leg["end"] - leg["start"]
        latest = legs[0]  # The earlier movement that ends last (the later one on ties).
        for leg in legs[1:]:
            gap = leg["start"] - latest["end"]
            moved = not same_place(latest["last_stop"], leg["first_stop"], places)
            jumps += moved
            if gap < 0:
                overlaps += 1
            elif gap > 0:
                seconds["unresolved_gap" if moved else "layover"] += gap
            if leg["end"] >= latest["end"]:
                latest = leg
        revenue = [leg for leg in legs if leg["category"] == "revenue"]
        row = {
            "service_ids": service_set,
            "block_id": "" if block.startswith("trip:") else block,
            "routes": "|".join(sorted({leg["route_id"] for leg in revenue})),
            "revenue_trips": len(revenue),
            "nonrevenue_trips": len(legs) - len(revenue),
            "start": _clock(min(leg["start"] for leg in legs)),
            "end": _clock(max(leg["end"] for leg in legs)),
            "has_tods": len(legs) > len(revenue),
            "overlaps": overlaps,
            "location_jumps": jumps,
        }
        for column in COMPONENTS:
            row[column] = seconds[CATEGORY_OF[column]] / 3600
        rows.append(row)
    return pd.DataFrame(rows)


def warn_on_overlaps(blocks: pd.DataFrame) -> None:
    """Log the blocks whose trips overlap in time, whose vehicle hours are withheld."""
    flagged = blocks.loc[blocks["overlaps"] > 0] if not blocks.empty else blocks
    if flagged.empty:
        return
    labels = [f"{b} ({s})" for s, b in zip(flagged["service_ids"], flagged["block_id"])]
    logging.warning(
        "%d block(s) have a trip that starts before an earlier trip of the block ends, "
        "e.g. %s. One vehicle cannot run both, so vehicle revenue, nonrevenue and vehicle "
        "hours are left blank for these blocks and for every total that includes them "
        "until the block_ids are fixed (see overlaps in %s, and the negative gaps on the "
        "Trips sheet of %s).",
        len(flagged),
        ", ".join(labels[:5]) + (", ..." if len(labels) > 5 else ""),
        BLOCK_FILENAME,
        WORKBOOK_FILENAME,
    )


def _clock(seconds: float) -> str:
    """Format seconds after midnight as ``HH:MM`` (24+ hours allowed)."""
    total = int(seconds) // 60
    return f"{total // 60:02d}:{total % 60:02d}"


def add_derived(df: pd.DataFrame, layover_in_revenue: bool, movements_known: bool) -> pd.DataFrame:
    """Add vehicle revenue, nonrevenue and vehicle hours.

    When *movements_known* is False (no TODS, or TODS not confirmed complete),
    the TODS columns, nonrevenue hours and vehicle hours are blank (NaN). Rows
    with ``overlaps`` (a block, or a total that includes one, whose trips overlap
    in time) have all three blank: the trips need more than one vehicle, so
    their vehicle time is unknown until the block is fixed.
    """
    out = df.copy()
    layover = out["layover_hours"]
    out["vehicle_revenue_hours"] = out["revenue_trip_hours"] + (
        layover if layover_in_revenue else 0
    )
    out["nonrevenue_hours"] = out[list(TODS_COMPONENTS)].sum(axis=1) + (
        0 if layover_in_revenue else layover
    )
    out["vehicle_hours"] = out[list(COMPONENTS)].sum(axis=1)
    if not movements_known:
        # Without complete TODS the pull-outs, pull-ins and deadheads are unknown, not zero.
        out[[*TODS_COMPONENTS, "nonrevenue_hours", "vehicle_hours"]] = float("nan")
    if "overlaps" in out.columns:
        out.loc[out["overlaps"] > 0, list(DERIVED)] = float("nan")
    return out


def movement_coverage_confirmed(blocks: pd.DataFrame, tods_supplied: bool, confirmed: bool) -> bool:
    """Return True if nonrevenue and vehicle hours can be reported; log why not.

    They need TODS, the user's confirmation (TODS_MOVEMENTS_COMPLETE) that the TODS
    files hold every block's movements, and at least one nonrevenue trip in them.

    Args:
        blocks: Output of :func:`block_hours`.
        tods_supplied: Whether TODS supplement files were applied.
        confirmed: TODS_MOVEMENTS_COMPLETE.
    """
    if not tods_supplied:
        return False
    with_tods = int(blocks["has_tods"].sum()) if not blocks.empty else 0
    coverage = f"{with_tods} of {len(blocks)} block(s) have TODS nonrevenue trips"
    if not confirmed:
        logging.warning(
            "Pull-out, pull-in, deadhead, nonrevenue and vehicle hours are left blank: "
            "TODS_MOVEMENTS_COMPLETE is False. %s. Set it to True (or pass "
            "--tods-movements-complete) once the TODS files hold every block's pull-out, "
            "pull-in and deadhead trips.",
            coverage,
        )
        return False
    if with_tods == 0:
        logging.warning(
            "TODS_MOVEMENTS_COMPLETE is True, but the TODS files hold no nonrevenue trips "
            "(no trip has a TODS_trip_type); nonrevenue and vehicle hours are left blank."
        )
        return False
    if with_tods < len(blocks):
        logging.warning(
            "%s; the others' pull-out and pull-in time is missing from vehicle hours "
            "(see has_tods in %s).",
            coverage,
            BLOCK_FILENAME,
        )
    return True


def service_hours(blocks: pd.DataFrame) -> pd.DataFrame:
    """Total the block hours per service set (one day on which that set runs)."""
    grouped = blocks.groupby("service_ids")
    out = grouped[list(COMPONENTS)].sum()
    out.insert(0, "blocks", grouped.size())
    out.insert(1, "blocks_without_tods", grouped["has_tods"].apply(lambda s: int((~s).sum())))
    out.insert(2, "revenue_trips", grouped["revenue_trips"].sum())
    out.insert(3, "nonrevenue_trips", grouped["nonrevenue_trips"].sum())
    out["overlaps"] = grouped["overlaps"].sum()
    out["location_jumps"] = grouped["location_jumps"].sum()
    return out


def service_day_types(
    active: Mapping[str, set[dt.date]], explicit: Mapping[str, str]
) -> dict[str, set[str]]:
    """Return service_id -> day types from *explicit* (SERVICE_DAY_TYPES); empty if it is.

    Raises:
        ValueError: If *explicit* names a day type outside DAY_TYPES.
    """
    if not explicit:
        return {}
    result: dict[str, set[str]] = {}
    for sid, value in explicit.items():
        labels = {v.strip().title() for v in str(value).split(",") if v.strip()}
        unknown = labels - set(DAY_TYPES)
        if unknown:
            raise ValueError(f"SERVICE_DAY_TYPES[{sid!r}] has unknown day type(s): {unknown}")
        result[str(sid)] = labels
    for sid in sorted(set(active) - set(result)):
        logging.warning("Service %s is not in SERVICE_DAY_TYPES; left out of day types.", sid)
    return result


def ranked_service_sets(calendar: pd.DataFrame, label: str) -> list[tuple[str, int]]:
    """Return the service sets running on *label* dates, most dates first (ties: earliest)."""
    sets = Counter(
        s for s, d in zip(calendar["service_set"], calendar["day_type"]) if d == label and s
    )
    return sets.most_common()


def typical_service_sets(calendar: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """Return each day type's candidate service sets, the typical one (most dates) first.

    Holiday ranks only sets that run on holidays on most of their dates: a holiday
    that runs the weekday or Sunday schedule does not make it a holiday schedule.
    Day types with no candidate set are left out.
    """
    ranked = {label: ranked_service_sets(calendar, label) for label in DAY_TYPES}
    total = Counter(s for s in calendar["service_set"] if s)
    ranked["Holiday"] = [(s, n) for s, n in ranked["Holiday"] if 2 * n > total[s]]
    return {label: sets for label, sets in ranked.items() if sets}


def config_day_types(
    explicit: Mapping[str, set[str]], gtfs_days: pd.DataFrame, holiday_service_day_type: str
) -> dict[str, set[str]]:
    """Return service_id -> day types for PERIOD_CALENDAR = "config".

    Config mode runs every service of a date's day type together, so the services
    of one day type must be ones that run together. Services that never share a
    date (a Monday-Thursday and a Friday schedule, or the schedules before and
    after a service change) would otherwise add several days' trips to each date.

    Args:
        explicit: Output of :func:`service_day_types` (SERVICE_DAY_TYPES).
        gtfs_days: Output of :func:`build_calendar` in GTFS calendar mode over the
            feed's dates.
        holiday_service_day_type: HOLIDAY_SERVICE_DAY_TYPE.

    Returns:
        *explicit*, once checked; when it is empty, each day type's typical
        service set from :func:`typical_service_sets`.

    Raises:
        ValueError: If *explicit* lists services under one day type that never
            all run on one date of the GTFS calendar.
    """
    observed = [set(running) for running in gtfs_days["running"] if running]
    if explicit:
        dated = set().union(*observed)
        for label in DAY_TYPES:
            members = {s for s, labels in explicit.items() if label in labels and s in dated}
            if len(members) > 1 and not any(members <= running for running in observed):
                combos = Counter("|".join(sorted(r & members)) for r in observed if r & members)
                runs = "; ".join(f"{s} ({n} dates)" for s, n in combos.most_common(5))
                raise ValueError(
                    f"SERVICE_DAY_TYPES lists {', '.join(sorted(members))} as {label}, but they "
                    "never all run on the same date of the GTFS calendar, so config mode would "
                    "add their trips together. List only the services that run together on one "
                    f"{label}. The GTFS calendar runs: {runs}."
                )
        return dict(explicit)
    result: dict[str, set[str]] = {}
    for label, ranked in typical_service_sets(gtfs_days).items():
        typical, n = ranked[0]
        logging.info(
            "Config calendar: %s dates run %s, the set on %d of %d %s dates in the GTFS calendar.",
            label,
            typical,
            n,
            sum(count for _set, count in ranked_service_sets(gtfs_days, label)),
            label,
        )
        for sid in typical.split("|"):
            result.setdefault(sid, set()).add(label)
    if holiday_service_day_type.title() == "Holiday" and not any(
        "Holiday" in labels for labels in result.values()
    ):
        logging.warning(
            "No service set in the GTFS calendar runs mostly on holidays, so config-mode "
            "holidays get no service. Set HOLIDAY_SERVICE_DAY_TYPE to the day type whose "
            "schedule runs on holidays, or list the holiday services in SERVICE_DAY_TYPES."
        )
    return result


def day_type_hours(services: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Report the hours of a typical day of each day type in *calendar*.

    A day type can have more than one version, i.e. more than one service set
    running on its dates (a Friday-only service, school-day trips, one service_id
    per weekday). The typical day is the set that runs on the most dates of that
    type (ties go to the earliest; see :func:`typical_service_sets`); the other
    sets are logged.

    Args:
        services: Output of :func:`service_hours`, indexed by service set.
        calendar: Output of :func:`build_calendar`.

    Returns:
        One row per day type with service: the set (``service_ids``), the dates
        of that type with service (``days``) and running the set (``typical_days``),
        its ``blocks`` and ``revenue_trips``, the COMPONENTS (hours) and its
        ``overlaps``.
    """
    rows = []
    typical_sets = typical_service_sets(calendar)
    for label in DAY_TYPES:
        ranked = ranked_service_sets(calendar, label)
        if label not in typical_sets:
            if ranked:
                logging.info("%s dates run only other day types' schedules.", label)
            continue
        typical, typical_days = typical_sets[label][0]
        others = [(s, n) for s, n in ranked if s != typical]
        days = sum(n for _set, n in ranked)
        if others:
            logging.warning(
                "%s: %d service sets run on %s dates; the typical %s is %s (%d of %d days). "
                "Others: %s.",
                label,
                len(ranked),
                label,
                label,
                typical,
                typical_days,
                days,
                ", ".join(f"{s} ({n} days)" for s, n in others),
            )
        sub = services.loc[typical]
        row: dict[str, Any] = {
            "day_type": label,
            "service_ids": typical,
            "days": days,
            "typical_days": typical_days,
            "blocks": int(sub["blocks"]),
            "revenue_trips": int(sub["revenue_trips"]),
        }
        row.update(sub[list(COMPONENTS)].astype(float).to_dict())
        row["overlaps"] = int(sub["overlaps"])
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_date(value: str) -> dt.date:
    """Parse ``YYYYMMDD``."""
    return datetime.strptime(str(value).strip(), "%Y%m%d").date()


def feed_span(active: Mapping[str, set[dt.date]], known: set[str]) -> Optional[tuple[str, str]]:
    """Return the first and last active dates (YYYYMMDD) of the *known* services, or None."""
    dates = [d for sid in known for d in active.get(sid, ())]
    if not dates:
        return None
    return min(dates).strftime("%Y%m%d"), max(dates).strftime("%Y%m%d")


def build_calendar(
    known: set[str],
    active: Mapping[str, set[dt.date]],
    types: Mapping[str, set[str]],
    cfg: Config,
    periods: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    """Return one row per (period, date) with its day type and the services running.

    Args:
        known: service_ids that have trips; other services are ignored.
        active: service_id -> active dates.
        types: service_id -> day types: SERVICE_DAY_TYPES with the GTFS calendar
            (a date running a Holiday service is a holiday), the services of each
            day type in config mode (:func:`config_day_types`).
        cfg: Resolved configuration (calendar mode and holidays).
        periods: (name, first date, last date) of each period, dates as YYYYMMDD.

    Returns:
        Columns ``period``, ``date``, ``weekday``, ``day_type``, ``schedule_day_type``
        (the day type whose services run; blank on a date without service. With
        PERIOD_CALENDAR = "gtfs" it starts as ``day_type``; see
        :func:`schedule_day_types`), ``running`` (sorted tuple of service_ids) and
        ``service_set`` (``running`` joined by ``|``).

    Raises:
        ValueError: If a period's dates are invalid, period names repeat, or
            PERIOD_CALENDAR is unknown.
    """
    if cfg.period_calendar not in {"gtfs", "config"}:
        raise ValueError(
            f"PERIOD_CALENDAR must be 'gtfs' or 'config', not {cfg.period_calendar!r}."
        )
    names = [name for name, _first, _last in periods]
    if len(names) != len(set(names)):
        raise ValueError(f"Period names must be unique: {names}")
    rows = []
    for name, first, last in periods:
        start, end = _parse_date(first), _parse_date(last)
        if end < start:
            raise ValueError(f"Period {name!r} ends before it starts.")
        holidays = {_parse_date(d) for d in cfg.holiday_dates}
        if cfg.use_federal_holidays:
            for year in range(start.year, end.year + 2):
                holidays |= federal_holidays_observed(year)
        day = start
        while day <= end:
            if cfg.period_calendar == "gtfs":
                running = service_ids_active_on(active, day) & known
                holiday = day in holidays or any("Holiday" in types.get(s, ()) for s in running)
                lookup = "Holiday" if holiday else WEEKDAY_LABELS[day.weekday()]
            else:
                holiday = day in holidays
                lookup = cfg.holiday_service_day_type if holiday else WEEKDAY_LABELS[day.weekday()]
                running = {s for s in known if lookup in types.get(s, set())}
            rows.append(
                {
                    "period": name,
                    "date": day,
                    "weekday": day.strftime("%A"),
                    "day_type": "Holiday" if holiday else WEEKDAY_LABELS[day.weekday()],
                    "schedule_day_type": lookup.title() if running else "",
                    "running": tuple(sorted(running)),
                    "service_set": "|".join(sorted(running)),
                }
            )
            day += dt.timedelta(days=1)
    columns = ["period", "date", "weekday", "day_type", "schedule_day_type", "running"]
    return pd.DataFrame(rows, columns=[*columns, "service_set"])


def schedule_day_types(calendar: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """Return the day type whose schedule runs on each date of *calendar* (GTFS calendar).

    A date runs day type X's schedule when its service set is X's typical set
    in *reference* (:func:`typical_service_sets`): a Christmas that runs the
    typical Sunday set is a Sunday schedule day. A set that is the typical set of
    no day type takes the day type of most of its dates in *reference* (a set
    run only on a few Fridays, or a weekday set run on a holiday, is a weekday
    schedule), or the date's own day type if it has none there. A set typical of
    several (a single weekend service) keeps the date's own day type if it is one
    of them, else takes the first in DAY_TYPES.

    Args:
        calendar: Output of :func:`build_calendar` (GTFS calendar mode).
        reference: The calendar the typical sets come from (the feed's whole span).

    Returns:
        One day type per row of *calendar*; blank where no service runs.
    """
    typical_of: dict[str, list[str]] = {}
    for label, ranked in typical_service_sets(reference).items():
        typical_of.setdefault(ranked[0][0], []).append(label)
    usual: dict[str, Counter[str]] = {}
    for service_set, day_type in zip(reference["service_set"], reference["day_type"]):
        usual.setdefault(service_set, Counter())[day_type] += 1
    out = []
    for service_set, day_type in zip(calendar["service_set"], calendar["day_type"]):
        labels = typical_of.get(service_set, [])
        if not service_set:
            out.append("")
        elif not labels and service_set in usual:
            out.append(usual[service_set].most_common(1)[0][0])
        elif not labels or day_type in labels:
            out.append(day_type)
        else:
            out.append(labels[0])
    return out


def period_hours(services: pd.DataFrame, calendar: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Total each period as days x hours per day of each service set, summed over sets.

    ``service_sets`` lists each set run in the period with its number of days, so a
    date running an unusual set (a Friday-only schedule) stays visible. ``overlaps``
    totals the sets' overlaps the same way (days x per day).
    """
    rows = []
    for name, first, last in cfg.periods:
        sub = calendar.loc[calendar["period"] == name]
        days = Counter(sub["day_type"])
        schedules = Counter(sub["schedule_day_type"])
        set_days = Counter(s for s in sub["service_set"] if s)
        totals = pd.Series(0.0, index=list(COMPONENTS))
        overlaps = 0
        for service_set, n_days in set_days.items():
            totals += n_days * services.loc[service_set, list(COMPONENTS)].astype(float)
            overlaps += n_days * int(services.loc[service_set, "overlaps"])
        no_service = int((sub["service_set"] == "").sum())
        row: dict[str, Any] = {
            "period": name,
            "start_date": _parse_date(first).isoformat(),
            "end_date": _parse_date(last).isoformat(),
            "calendar": cfg.period_calendar,
            "days": len(sub),
            "weekdays": days["Weekday"],
            "saturdays": days["Saturday"],
            "sundays": days["Sunday"],
            "holidays": days["Holiday"],
            "weekday_schedule_days": schedules["Weekday"],
            "saturday_schedule_days": schedules["Saturday"],
            "sunday_schedule_days": schedules["Sunday"],
            "holiday_schedule_days": schedules["Holiday"],
            "days_without_service": no_service,
            "service_sets": "; ".join(
                f"{s} ({n} day{'s' if n != 1 else ''})" for s, n in set_days.most_common()
            ),
        }
        row.update(totals.to_dict())
        row["overlaps"] = overlaps
        rows.append(row)
        if no_service:
            logging.warning("Period %s: %d day(s) with no service counted.", name, no_service)
    return pd.DataFrame(rows)


# =============================================================================
# FORMULA WORKBOOK
# =============================================================================

TRIP_TYPE_LABELS = {
    "revenue": "revenue",
    "pull_out": "pull-out",
    "pull_in": "pull-in",
    "deadhead": "deadhead",
    "other_nonrevenue": "other nonrevenue",
}
COMPONENT_LABELS = {
    "revenue_trip_hours": "Revenue trip h",
    "layover_hours": "Layover h",
    "pull_out_hours": "Pull-out h",
    "pull_in_hours": "Pull-in h",
    "deadhead_hours": "Deadhead h",
    "other_nonrevenue_hours": "Other nonrevenue h",
    "unresolved_gap_hours": "Unresolved gap h",
}
DERIVED_LABELS = {
    "vehicle_revenue_hours": "Vehicle revenue h",
    "nonrevenue_hours": "Nonrevenue h",
    "vehicle_hours": "Vehicle h",
}
LAYOVER_CELL = "Settings!$B$2"
HOURS_FORMAT = "0.00"
CLOCK_FORMAT = "[h]:mm:ss"


def movement_table(movements: pd.DataFrame, places: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """Return every trip in block order, flagging when the vehicle stayed put since the last.

    *movements* is the output of :func:`expand_by_service_set`, so a trip appears
    once per service set it runs in. Trips with a blank block_id each form their
    own block, labelled ``(none: <trip_id>)``. ``stays_put`` is True when the trip
    starts where the vehicle is once every earlier trip of the block has ended: the
    stop where the latest-ending one finished (same stop ID or same coordinates), as
    in :func:`block_hours`.
    """
    df = movements.copy()
    df["block_key"] = df["block_id"].where(df["block_id"] != "", "(none: " + df["trip_id"] + ")")
    df = df.sort_values(["service_set", "block_key", "start", "end"], kind="stable")
    df = df.reset_index(drop=True)
    same_block = df["service_set"].eq(df["service_set"].shift()) & df["block_key"].eq(
        df["block_key"].shift()
    )
    df["same_block_as_previous"] = same_block
    stays_put: list[bool] = []
    latest_stop, latest_end = "", 0.0
    for sb, first, last, end in zip(same_block, df["first_stop"], df["last_stop"], df["end"]):
        stays_put.append(bool(sb) and same_place(str(latest_stop), str(first), places))
        if not sb or end >= latest_end:
            latest_stop, latest_end = last, end
    df["stays_put"] = stays_put
    df["trip_type"] = df["category"].map(TRIP_TYPE_LABELS)
    return df


def _withheld_if_overlaps(formula: str, overlaps: str) -> str:
    """Wrap *formula* to show "n/a" while the *overlaps* cell is above zero."""
    return f'=IF({overlaps}>0,"n/a",{formula[1:]})'


def _derived_formulas(
    cols: Mapping[str, str], r: int, tods: bool, overlaps_col: str
) -> dict[str, Any]:
    """Formulas for vehicle revenue, nonrevenue and vehicle hours on row *r*.

    Each is "n/a" while the row's overlaps (column *overlaps_col*) are above zero.
    """
    c = {k: f"{v}{r}" for k, v in cols.items()}
    revenue = f"={c['revenue_trip_hours']}+IF({LAYOVER_CELL},{c['layover_hours']},0)"
    if not tods:
        formulas = {
            "vehicle_revenue_hours": revenue,
            "nonrevenue_hours": "n/a",
            "vehicle_hours": "n/a",
        }
    else:
        moves = "+".join(c[k] for k in TODS_COMPONENTS)
        first, last = cols["revenue_trip_hours"], cols["unresolved_gap_hours"]
        formulas = {
            "vehicle_revenue_hours": revenue,
            "nonrevenue_hours": f"={moves}+IF({LAYOVER_CELL},0,{c['layover_hours']})",
            "vehicle_hours": f"=SUM({first}{r}:{last}{r})",
        }
    return {
        k: _withheld_if_overlaps(v, f"{overlaps_col}{r}") if v.startswith("=") else v
        for k, v in formulas.items()
    }


def _sheet(wb: Any, title: str, headers: Sequence[str], widths: Mapping[int, int]) -> Any:
    """Add a sheet with a bold, frozen header row and the given column widths."""
    ws = wb.create_sheet(title)
    ws.append(list(headers))
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    for idx in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(idx)].width = widths.get(idx, 14)
    return ws


def write_workbook(
    path: Path,
    trips: pd.DataFrame,
    calendar: pd.DataFrame,
    types: Mapping[str, set[str]],
    cfg: Config,
    movements_known: bool,
) -> Path:
    """Write a workbook whose period totals are Excel formulas back to trip times.

    Sheets: Periods (days x hours per day of each service set), Settings (the
    layover toggle and inputs), Period days and Calendar (which service set runs
    on each date; editable), Services and Blocks (sums of the rows below them) and
    Trips (start/end times; durations, gaps and layover as formulas).

    Args:
        path: Output ``.xlsx`` path.
        trips: Output of :func:`movement_table`.
        calendar: Output of :func:`build_calendar` for the periods.
        types: service set -> day types of the dates it runs on.
        cfg: Resolved configuration.
        movements_known: False leaves the TODS columns, nonrevenue and vehicle
            hours as "n/a".

    Returns:
        *path*.
    """
    wb = Workbook()
    wb.remove(wb.active)
    services = sorted(set(trips["service_set"]))
    component_cols = list(COMPONENTS)

    def n_a(column: str) -> bool:
        return not movements_known and column in TODS_COMPONENTS

    # --- Trips -----------------------------------------------------------------
    ws_t = _sheet(
        wb,
        "Trips",
        [
            "Services",
            "Block",
            "Trip ID",
            "Type",
            "TODS_trip_type",
            "Route",
            "From stop",
            "To stop",
            "Start",
            "End",
            "Duration h",
            "Same block as previous row",
            "Gap before h",
            "Stays at the same place",
            "Layover h",
            "Unresolved gap h",
            "Trip service",
        ],
        {2: 26, 3: 30, 5: 16, 7: 18, 8: 18},
    )
    block_rows: dict[tuple[str, str], list[int]] = {}
    block_start = 2
    for i, t in enumerate(trips.to_dict("records")):
        r = i + 2
        if not t["same_block_as_previous"]:
            block_start = r
        ws_t.append(
            [
                t["service_set"],
                t["block_key"],
                t["trip_id"],
                t["trip_type"],
                t["tods_trip_type"],
                t["route_id"],
                t["first_stop"],
                t["last_stop"],
                t["start"] / 86400,
                t["end"] / 86400,
                f"=(J{r}-I{r})*24",
                bool(t["same_block_as_previous"]),
                (f"=IF(L{r},(I{r}-MAX(J{block_start}:J{r - 1}))*24,0)" if r > block_start else 0),
                bool(t["stays_put"]),
                f"=IF(AND(M{r}>0,N{r}),M{r},0)",
                f"=IF(AND(M{r}>0,NOT(N{r})),M{r},0)",
                t["service_id"],
            ]
        )
        for col in ("I", "J"):
            ws_t[f"{col}{r}"].number_format = CLOCK_FORMAT
        for col in ("K", "M", "O", "P"):
            ws_t[f"{col}{r}"].number_format = HOURS_FORMAT
        block_rows.setdefault((t["service_set"], t["block_key"]), []).append(r)

    # --- Blocks ----------------------------------------------------------------
    block_headers = [
        "Services",
        "Block",
        "Routes",
        "Trips rows",
        "Revenue trips",
        "Nonrevenue trips",
        *COMPONENT_LABELS.values(),
        *DERIVED_LABELS.values(),
        "Overlaps",
    ]
    ws_b = _sheet(wb, "Blocks", block_headers, {2: 26, 3: 18})
    b_cols = {c: get_column_letter(7 + i) for i, c in enumerate(component_cols)}
    b_overlaps = get_column_letter(7 + len(component_cols) + len(DERIVED))
    sums = {
        "revenue_trip_hours": ("K", "revenue"),
        "pull_out_hours": ("K", "pull-out"),
        "pull_in_hours": ("K", "pull-in"),
        "deadhead_hours": ("K", "deadhead"),
        "other_nonrevenue_hours": ("K", "other nonrevenue"),
    }
    service_block_rows: dict[str, list[int]] = {}
    for i, ((sid, block), rows) in enumerate(block_rows.items()):
        r = i + 2
        a, b = rows[0], rows[-1]
        routes = trips.loc[(trips["service_set"] == sid) & (trips["block_key"] == block)]
        routes = routes.loc[routes["category"] == "revenue", "route_id"]
        values: dict[str, Any] = {}
        for column in component_cols:
            if n_a(column):
                values[column] = "n/a"
            elif column in sums:
                src, kind = sums[column]
                values[column] = (
                    f'=SUMIFS(Trips!${src}${a}:${src}${b},Trips!$D${a}:$D${b},"{kind}")'
                )
            else:
                src = "O" if column == "layover_hours" else "P"
                values[column] = f"=SUM(Trips!${src}${a}:${src}${b})"
        values.update(_derived_formulas(b_cols, r, movements_known, b_overlaps))
        ws_b.append(
            [
                sid,
                block,
                "|".join(sorted(set(routes))),
                f"{a}-{b}",
                f'=COUNTIFS(Trips!$D${a}:$D${b},"revenue")',
                f"=ROWS(Trips!$D${a}:$D${b})-E{r}",
                *(values[c] for c in (*component_cols, *DERIVED)),
                f'=COUNTIF(Trips!$M${a}:$M${b},"<0")',
            ]
        )
        for idx in range(7, 7 + len(component_cols) + len(DERIVED)):
            ws_b.cell(r, idx).number_format = HOURS_FORMAT
        service_block_rows.setdefault(sid, []).append(r)

    # --- Services --------------------------------------------------------------
    ws_s = _sheet(
        wb,
        "Services",
        [
            "Services",
            "Day types",
            "Blocks rows",
            "Blocks",
            "Revenue trips",
            *COMPONENT_LABELS.values(),
            *DERIVED_LABELS.values(),
            "Overlaps",
        ],
        {2: 20},
    )
    s_cols = {c: get_column_letter(6 + i) for i, c in enumerate(component_cols)}
    s_overlaps = get_column_letter(6 + len(component_cols) + len(DERIVED))
    service_row: dict[str, int] = {}
    for i, sid in enumerate(services):
        r = i + 2
        a, b = service_block_rows[sid][0], service_block_rows[sid][-1]
        values = {
            c: "n/a" if n_a(c) else f"=SUM(Blocks!{b_cols[c]}{a}:{b_cols[c]}{b})"
            for c in component_cols
        }
        values.update(_derived_formulas(s_cols, r, movements_known, s_overlaps))
        ws_s.append(
            [
                sid,
                ", ".join(d for d in DAY_TYPES if d in types.get(sid, set())),
                f"{a}-{b}",
                f"=ROWS(Blocks!A{a}:A{b})",
                f"=SUM(Blocks!E{a}:E{b})",
                *(values[c] for c in (*component_cols, *DERIVED)),
                f"=SUM(Blocks!{b_overlaps}{a}:{b_overlaps}{b})",
            ]
        )
        for idx in range(6, 6 + len(component_cols) + len(DERIVED)):
            ws_s.cell(r, idx).number_format = HOURS_FORMAT
        service_row[sid] = r

    # --- Calendar --------------------------------------------------------------
    ws_c = _sheet(
        wb,
        "Calendar",
        [
            "Period",
            "Date",
            "Weekday",
            "Day type",
            "Schedule day type",
            *services,
            "Service sets running",
        ],
        {1: 18, 2: 12},
    )
    c_cols = {sid: get_column_letter(6 + i) for i, sid in enumerate(services)}
    running_col = get_column_letter(6 + len(services))
    last_service_col = get_column_letter(5 + len(services))
    for i, day in enumerate(calendar.to_dict("records")):
        r = i + 2
        ws_c.append(
            [
                day["period"],
                day["date"],
                day["weekday"],
                day["day_type"],
                day["schedule_day_type"],
                *(1 if sid == day["service_set"] else 0 for sid in services),
                f"=SUM(F{r}:{last_service_col}{r})",
            ]
        )
        ws_c[f"B{r}"].number_format = "yyyy-mm-dd"

    # --- Period days -----------------------------------------------------------
    ws_d = _sheet(
        wb,
        "Period days",
        [
            "Period",
            "Start",
            "End",
            "Days",
            "Weekdays",
            "Saturdays",
            "Sundays",
            "Holidays",
            *(f"{label} schedule days" for label in DAY_TYPES),
            "Days without service",
        ],
        {1: 18, 2: 12, 3: 12},
    )
    for i, (name, first, last) in enumerate(cfg.periods):
        r = i + 2
        key = f"Calendar!$A:$A,$A{r}"
        ws_d.append(
            [
                name,
                _parse_date(first),
                _parse_date(last),
                f"=COUNTIFS({key})",
                *(f'=COUNTIFS({key},Calendar!$D:$D,"{label}")' for label in DAY_TYPES),
                *(f'=COUNTIFS({key},Calendar!$E:$E,"{label}")' for label in DAY_TYPES),
                f"=COUNTIFS({key},Calendar!${running_col}:${running_col},0)",
            ]
        )
        for col in ("B", "C"):
            ws_d[f"{col}{r}"].number_format = "yyyy-mm-dd"

    # --- Periods ---------------------------------------------------------------
    ws_p = _sheet(
        wb,
        "Periods",
        [
            "Period",
            "Start",
            "End",
            "Services",
            "Service days",
            *COMPONENT_LABELS.values(),
            *DERIVED_LABELS.values(),
            "Overlaps",
        ],
        {1: 18, 2: 12, 3: 12, 4: 16},
    )
    p_cols = {c: get_column_letter(6 + i) for i, c in enumerate(component_cols)}
    p_overlaps = get_column_letter(6 + len(component_cols) + len(DERIVED))
    r = 1
    for name, first, last in cfg.periods:
        first_row = r + 1
        for sid in services:
            r += 1
            srow = service_row[sid]
            values = {
                c: "n/a" if n_a(c) else f"=$E{r}*Services!{s_cols[c]}{srow}" for c in component_cols
            }
            values.update(_derived_formulas(p_cols, r, movements_known, p_overlaps))
            ws_p.append(
                [
                    name,
                    _parse_date(first),
                    _parse_date(last),
                    sid,
                    f"=SUMIFS(Calendar!{c_cols[sid]}:{c_cols[sid]},Calendar!$A:$A,$A{r})",
                    *(values[c] for c in (*component_cols, *DERIVED)),
                    f"=$E{r}*Services!{s_overlaps}{srow}",
                ]
            )
        r += 1
        totals: list[Any] = []
        for idx in range(6, 6 + len(component_cols) + len(DERIVED)):
            col = get_column_letter(idx)
            header = ws_p.cell(1, idx).value
            unknown = not movements_known and header in (
                *(COMPONENT_LABELS[c] for c in TODS_COMPONENTS),
                DERIVED_LABELS["nonrevenue_hours"],
                DERIVED_LABELS["vehicle_hours"],
            )
            total = f"=SUM({col}{first_row}:{col}{r - 1})"
            if header in DERIVED_LABELS.values():
                total = _withheld_if_overlaps(total, f"{p_overlaps}{r}")
            totals.append("n/a" if unknown else total)
        totals.append(f"=SUM({p_overlaps}{first_row}:{p_overlaps}{r - 1})")
        ws_p.append([name, _parse_date(first), _parse_date(last), "All service sets", "", *totals])
        for cell in ws_p[r]:
            cell.font = Font(bold=True)
        for row in range(first_row, r + 1):
            for col in ("B", "C"):
                ws_p[f"{col}{row}"].number_format = "yyyy-mm-dd"
            for idx in range(6, 6 + len(component_cols) + len(DERIVED)):
                ws_p.cell(row, idx).number_format = HOURS_FORMAT

    # --- Settings --------------------------------------------------------------
    ws_set = _sheet(wb, "Settings", ["Setting", "Value", "Notes"], {1: 34, 2: 40, 3: 90})
    notes = [
        (
            "Layover counts as revenue time",
            cfg.layover_in_revenue_hours,
            "TRUE or FALSE. Every vehicle revenue / nonrevenue total recalculates.",
        ),
        ("GTFS", str(cfg.gtfs_path), ""),
        ("TODS", str(cfg.tods_path) if cfg.tods_path else "(none: revenue only)", ""),
        (
            "TODS movements complete",
            cfg.tods_movements_complete,
            "TODS_MOVEMENTS_COMPLETE: the TODS files hold every block's pull-out, pull-in "
            "and deadhead trips. Nonrevenue and vehicle hours need it.",
        ),
        ("Period calendar", cfg.period_calendar, "gtfs: GTFS calendar; config: day types"),
        ("Generated", datetime.now().isoformat(timespec="seconds"), ""),
        ("", "", ""),
        ("How the numbers are built", "", ""),
        (
            "Trips",
            "",
            "One row per revenue (GTFS) or nonrevenue (TODS) trip, in block order, "
            "under each service set (the services that run together on a date) that "
            "includes the trip's service. Duration h = (End - Start) x 24. Gap before h = "
            "Start - the latest End among the block's earlier rows, so time any trip covers "
            "is never a gap. A positive gap where the vehicle stays at the same place (where "
            "that latest-ending trip finished) is layover; a positive gap where it changes "
            "place with no trip in between is an unresolved gap. A negative gap is an "
            "overlap: the trip starts before an earlier trip of the block ends.",
        ),
        (
            "Overlaps",
            "",
            "Trips that start before an earlier trip of their block ends. One vehicle cannot "
            "run both, so vehicle revenue, nonrevenue and vehicle hours are n/a for a block "
            "with overlaps and for every service set and period total that includes it, "
            "until the block_ids are fixed.",
        ),
        ("Blocks", "", "Sums of each block's Trips rows (row range shown)."),
        (
            "Services",
            "",
            "Sums of the service set's Blocks rows: hours for one day on which that set runs.",
        ),
        (
            "Calendar",
            "",
            "1 under the service set that runs on a date. Move the 1 (e.g. for a holiday) "
            "and the period totals follow. Schedule day type is the day type whose typical "
            "service set runs that date (e.g. Sunday for a holiday on Sunday service); "
            "it does not follow edits.",
        ),
        (
            "Period days",
            "",
            "Day counts per period, from the Calendar: by calendar day type and by schedule.",
        ),
        (
            "Periods",
            "",
            "Days in the period x hours per day of each service set, then totalled.",
        ),
    ]
    if not movements_known:
        notes.append(
            (
                "No complete TODS",
                "",
                "Pull-out, pull-in, deadhead, nonrevenue and vehicle hours are n/a: there is "
                "no TODS, TODS_MOVEMENTS_COMPLETE is False, or the TODS has no nonrevenue trips.",
            )
        )
    for row in notes:
        ws_set.append(list(row))
    for row in ws_set.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    order = ("Periods", "Settings", "Period days", "Calendar", "Services", "Blocks", "Trips")
    for idx, name in enumerate(order):
        wb.move_sheet(name, offset=idx - wb.sheetnames.index(name))
    # openpyxl stores formulas without results; have Excel compute them on open.
    wb.calculation = CalcProperties(fullCalcOnLoad=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Optional[Path]:
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
        "SCHEDULED SERVICE HOURS RUN LOG",
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


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    """Compute every table and write the CSVs and run log.

    Args:
        cfg: Resolved configuration.

    Returns:
        The ``block``, ``service``, ``day_type`` and ``period`` tables.

    Raises:
        RuntimeError: If the run log is required and could not be written.
        ValueError: If no service with trips runs on any date.
    """
    schedule = load_schedule(cfg.gtfs_path, cfg.tods_path)
    known = set(schedule.movements["service_id"])
    types = service_day_types(schedule.active_dates, cfg.service_day_types)
    # Typical days come from every date of the feed; the periods if it has no dates.
    span = feed_span(schedule.active_dates, known)
    if cfg.period_calendar == "config":
        gtfs_days = build_calendar(
            known,
            schedule.active_dates,
            types,
            replace(cfg, period_calendar="gtfs"),
            [("feed", *span)] if span is not None else cfg.periods,
        )
        types = config_day_types(types, gtfs_days, cfg.holiday_service_day_type)
    calendar = build_calendar(known, schedule.active_dates, types, cfg, cfg.periods)
    feed_calendar = (
        build_calendar(known, schedule.active_dates, types, cfg, [("feed", *span)])
        if span is not None
        else calendar
    )
    if cfg.period_calendar == "gtfs":
        calendar["schedule_day_type"] = schedule_day_types(calendar, feed_calendar)
    days = pd.concat([calendar, feed_calendar], ignore_index=True)
    sets = sorted({running for running in days["running"] if running})
    if not sets:
        raise ValueError(
            "No service with trips runs on any date of the feed calendar or the periods; "
            "check calendar.txt / calendar_dates.txt (or SERVICE_DAY_TYPES in config mode)."
        )
    idle = sorted(known - {sid for members in sets for sid in members})
    if idle:
        logging.warning("Service(s) with trips that never run, left out: %s", ", ".join(idle))
    movements = expand_by_service_set(schedule.movements, sets)
    blocks = block_hours(movements, schedule.stop_places)
    warn_on_overlaps(blocks)
    known = movement_coverage_confirmed(blocks, schedule.tods_supplied, cfg.tods_movements_complete)
    services = service_hours(blocks)
    set_day_types: dict[str, set[str]] = {}
    for service_set, day_type in zip(days["service_set"], days["day_type"]):
        if service_set:
            set_day_types.setdefault(service_set, set()).add(day_type)
    tables = {
        "block": blocks,
        "service": services.reset_index(),
        "day_type": day_type_hours(services, feed_calendar),
        "period": period_hours(services, calendar, cfg),
    }
    for key, table in tables.items():
        if not table.empty:
            table = add_derived(table, cfg.layover_in_revenue_hours, known)
            hours = [c for c in (*COMPONENTS, *DERIVED) if c in table.columns]
            table[hours] = table[hours].round(2)
        tables[key] = table

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("block", BLOCK_FILENAME),
        ("service", SERVICE_FILENAME),
        ("day_type", DAY_TYPE_FILENAME),
        ("period", PERIOD_FILENAME),
    ):
        tables[key].to_csv(cfg.output_dir / filename, index=False)
        logging.info("Wrote %s", cfg.output_dir / filename)
    workbook = write_workbook(
        cfg.output_dir / WORKBOOK_FILENAME,
        movement_table(movements, schedule.stop_places),
        calendar,
        set_day_types,
        cfg,
        known,
    )
    logging.info("Wrote %s (formulas; open in Excel to check the totals)", workbook)

    summary = [
        f"GTFS input:        {cfg.gtfs_path}",
        f"TODS input:        {cfg.tods_path if cfg.tods_path else '(none: revenue only)'}",
        f"TODS movements:    {'confirmed complete' if known else 'not used (n/a)'}",
        f"Service sets:      {len(services)}",
        f"Blocks (per set):  {len(blocks)}",
        f"Overlap blocks:    {int((blocks['overlaps'] > 0).sum()) if not blocks.empty else 0}",
        f"Period calendar:   {cfg.period_calendar}",
    ]

    def _hours(value: float, width: int) -> str:
        return ("n/a" if pd.isna(value) else f"{value:.2f}").rjust(width)

    for row in tables["day_type"].to_dict("records"):
        summary.append(
            f"{row['day_type']:<9} vehicle revenue h {_hours(row['vehicle_revenue_hours'], 9)}  "
            f"nonrevenue h {_hours(row['nonrevenue_hours'], 8)}"
        )
    if not write_run_log(cfg.output_dir, summary) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return tables


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser; every default is a CONFIGURATION constant."""
    p = argparse.ArgumentParser(
        description="Scheduled revenue and nonrevenue vehicle hours from GTFS plus TODS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gtfs", default=GTFS_PATH, help="GTFS folder or .zip.")
    p.add_argument("--tods", default=TODS_PATH, help="TODS supplement folder or .zip.")
    p.add_argument("--no-tods", action="store_true", help="Ignore TODS (revenue hours only).")
    p.add_argument(
        "--tods-movements-complete",
        action=argparse.BooleanOptionalAction,
        default=TODS_MOVEMENTS_COMPLETE,
        help="Confirm the TODS files hold every block's pull-out, pull-in and deadhead "
        "trips, so nonrevenue and vehicle hours are reported.",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for the outputs.")
    p.add_argument(
        "--period",
        nargs=3,
        action="append",
        metavar=("NAME", "START", "END"),
        help="Reporting period (dates YYYYMMDD); repeat for several. Default: PERIODS.",
    )
    p.add_argument(
        "--period-calendar",
        choices=("gtfs", "config"),
        default=PERIOD_CALENDAR,
        help="Count each date's services from the GTFS calendar or from its day type.",
    )
    p.add_argument(
        "--layover-in-revenue",
        action=argparse.BooleanOptionalAction,
        default=LAYOVER_IN_REVENUE_HOURS,
        help="Count layover/recovery time as revenue time.",
    )
    return p


def is_placeholder_path(path: object) -> bool:
    """Return True if *path* still points at a default placeholder location."""
    text = str(path).lower()
    return any(marker in text for marker in _PLACEHOLDER_MARKERS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv`` (or uses the
            CONFIGURATION constants inside a notebook).

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    tods = None if args.no_tods or args.tods is None else args.tods
    placeholders = [
        name
        for name, value in (
            ("GTFS_PATH", args.gtfs),
            ("TODS_PATH", tods),
            ("OUTPUT_DIR", args.output_dir),
        )
        if value is not None and is_placeholder_path(value)
    ]
    if placeholders:
        logging.warning(
            "%s still placeholder(s). Update the CONFIGURATION section or pass --gtfs, "
            "--tods / --no-tods and --output-dir before running.",
            ", ".join(placeholders),
        )
        return 2
    cfg = Config(
        gtfs_path=Path(args.gtfs).expanduser(),
        tods_path=Path(tods).expanduser() if tods else None,
        output_dir=Path(args.output_dir).expanduser(),
        periods=[tuple(p) for p in args.period] if args.period else list(PERIODS),
        period_calendar=args.period_calendar,
        layover_in_revenue_hours=args.layover_in_revenue,
        tods_movements_complete=args.tods_movements_complete,
    )
    try:
        tables = run(cfg)
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        logging.error("%s", exc)
        return 1
    for row in tables["period"].to_dict("records"):
        logging.info(
            "%s: %s vehicle revenue hours, %s nonrevenue hours.",
            row["period"],
            *(
                "n/a" if pd.isna(row[c]) else f"{row[c]:.1f}"
                for c in ("vehicle_revenue_hours", "nonrevenue_hours")
            ),
        )
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
