"""Total scheduled revenue and nonrevenue vehicle hours from GTFS plus TODS.

Reads a GTFS feed and, optionally, its TODS supplement files, rebuilds each
vehicle block's day (pull-out, revenue trips, layovers, deadheads, pull-in) and
totals the hours by GTFS service, by day type (Weekday / Saturday / Sunday /
Holiday) and by reporting period, such as a month. TODS supplements are applied
to the GTFS tables by the TODS 2.1 rules (delete, update, add) before anything
is counted. Without TODS the run reports revenue trip hours and layover only.

Each block's time is split into revenue trip hours (first departure to last
arrival of each revenue trip), layover (the vehicle waits at one place between
movements), TODS pull-out, pull-in and deadhead trips (other TODS_trip_type
values are totalled as other nonrevenue), and unresolved gaps, where the vehicle
ends one movement somewhere other than where the next begins with no scheduled
movement in between. Vehicle revenue hours add layover to revenue trip hours
when LAYOVER_IN_REVENUE_HOURS is True; vehicle hours run from pull-out to pull-in.

Day types come from SERVICE_DAY_TYPES or, when that is empty, from each
service's GTFS active dates. Period totals count each date's services from the
GTFS calendar (calendar.txt, calendar_dates.txt and their TODS supplements) or,
with PERIOD_CALENDAR = "config", from each date's day type, treating
HOLIDAY_DATES (and optionally U.S. federal holidays) as holidays.

Inputs
------
- GTFS_PATH: GTFS feed, as a folder or ``.zip``.
- TODS_PATH (optional): TODS supplement files for that feed, as a folder or
  ``.zip`` (for example from ``tsde_to_tods.py``).

Outputs
-------
- ``service_hours_by_block.csv``: hours per block and service, with QA counts.
- ``service_hours_by_service.csv``: hours per service day of each service_id.
- ``service_hours_by_day_type.csv``: hours on a typical day of each day type.
- ``service_hours_by_period.csv``: total hours and day counts per period.
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
import os
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS feed (folder or .zip) and its TODS supplement files (folder or .zip).
# Set TODS_PATH = None to report revenue trip hours and layover only.
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder_or.zip"
TODS_PATH: Optional[str] = r"Path\To\Your\TODS_Folder_or.zip"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

BLOCK_FILENAME: str = "service_hours_by_block.csv"
SERVICE_FILENAME: str = "service_hours_by_service.csv"
DAY_TYPE_FILENAME: str = "service_hours_by_day_type.csv"
PERIOD_FILENAME: str = "service_hours_by_period.csv"

# Reporting periods as (name, first date, last date), dates as YYYYMMDD.
PERIODS: List[tuple[str, str, str]] = [("November 2026", "20261101", "20261130")]

# Which services run on each date of a period:
#   "gtfs"   - the GTFS calendar (with TODS calendar supplements), date by date.
#   "config" - each date's day type: Weekday / Saturday / Sunday, or Holiday for
#              the dates below; the services of that day type run.
PERIOD_CALENDAR: str = "gtfs"

# service_id -> day type(s): "Weekday", "Saturday", "Sunday", "Holiday", or a
# comma-separated list such as "Saturday, Sunday". Empty: classify each service
# from its GTFS active dates.
SERVICE_DAY_TYPES: dict[str, str] = {}

# PERIOD_CALENDAR = "config" only: holiday dates (YYYYMMDD), whether to add the
# observed U.S. federal holidays, and whose services run on a holiday (for
# example "Sunday" if holidays run the Sunday schedule).
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
        ValueError: If required GTFS tables are missing or inconsistent.
    """
    gtfs = load_tables(
        gtfs_path,
        ("trips.txt", "stop_times.txt", "stops.txt", "calendar.txt", "calendar_dates.txt"),
    )
    for name in ("trips", "stop_times"):
        if name not in gtfs:
            raise ValueError(f"GTFS feed '{gtfs_path}' has no {name}.txt.")
    tods: dict[str, pd.DataFrame] = {}
    if tods_path is not None:
        tods = load_tables(
            tods_path,
            (
                "trips_supplement.txt",
                "stop_times_supplement.txt",
                "stops_supplement.txt",
                "calendar_supplement.txt",
                "calendar_dates_supplement.txt",
            ),
        )
        if not tods:
            raise ValueError(f"No TODS supplement files found in '{tods_path}'.")
    trips = apply_supplement(gtfs["trips"], tods.get("trips_supplement"), ["trip_id"])
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


def same_place(a: str, b: str, places: Mapping[str, tuple[str, str]]) -> bool:
    """Return True if stops *a* and *b* are one place (same id or same coordinates)."""
    if a == b:
        return True
    pa, pb = places.get(a), places.get(b)
    return pa is not None and pa == pb and all(pa)


def block_hours(movements: pd.DataFrame, places: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """Split each block's service day into the hour components.

    Trips with a blank block_id each form their own block. Gaps where the vehicle
    stays put are layover; gaps where it changes place with no movement are
    unresolved. Overlapping movements are counted, not summed as gaps.

    Returns:
        One row per (service_id, block_id) with the COMPONENTS (hours) and
        ``revenue_trips``, ``nonrevenue_trips``, ``routes``, ``start``, ``end``,
        ``has_tods``, ``overlaps`` and ``location_jumps``.
    """
    keyed = movements.assign(
        _block=movements["block_id"].where(
            movements["block_id"] != "", "trip:" + movements["trip_id"]
        )
    )
    rows = []
    for (service_id, block), group in keyed.groupby(["service_id", "_block"], sort=True):
        legs = group.sort_values(["start", "end"]).to_dict("records")
        seconds = Counter()
        overlaps = jumps = 0
        for leg in legs:
            seconds[leg["category"]] += leg["end"] - leg["start"]
        for before, after in zip(legs, legs[1:]):
            gap = after["start"] - before["end"]
            moved = not same_place(before["last_stop"], after["first_stop"], places)
            jumps += moved
            if gap < 0:
                overlaps += 1
            elif gap > 0:
                seconds["unresolved_gap" if moved else "layover"] += gap
        revenue = [leg for leg in legs if leg["category"] == "revenue"]
        row = {
            "service_id": service_id,
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


def _clock(seconds: float) -> str:
    """Format seconds after midnight as ``HH:MM`` (24+ hours allowed)."""
    total = int(seconds) // 60
    return f"{total // 60:02d}:{total % 60:02d}"


def add_derived(df: pd.DataFrame, layover_in_revenue: bool, tods_supplied: bool) -> pd.DataFrame:
    """Add vehicle revenue, nonrevenue and vehicle hours; blank TODS columns without TODS."""
    out = df.copy()
    layover = out["layover_hours"]
    out["vehicle_revenue_hours"] = out["revenue_trip_hours"] + (
        layover if layover_in_revenue else 0
    )
    out["nonrevenue_hours"] = out[list(TODS_COMPONENTS)].sum(axis=1) + (
        0 if layover_in_revenue else layover
    )
    out["vehicle_hours"] = out[list(COMPONENTS)].sum(axis=1)
    if not tods_supplied:
        # Without TODS the pull-outs, pull-ins and deadheads are unknown, not zero.
        out[[*TODS_COMPONENTS, "nonrevenue_hours", "vehicle_hours"]] = float("nan")
    return out


def service_hours(blocks: pd.DataFrame) -> pd.DataFrame:
    """Total the block hours per service_id (one service day of each service)."""
    grouped = blocks.groupby("service_id")
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
    """Return service_id -> day types, from *explicit* or from GTFS active dates.

    Raises:
        ValueError: If *explicit* names a day type outside DAY_TYPES.
    """
    if not explicit:
        return classify_service_ids(active)
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


def day_type_hours(services: pd.DataFrame, types: Mapping[str, set[str]]) -> pd.DataFrame:
    """Sum the services of each day type into one typical day of that type."""
    rows = []
    for label in DAY_TYPES:
        members = sorted(s for s in services.index if label in types.get(s, set()))
        if not members:
            continue
        sub = services.loc[members]
        row: dict[str, Any] = {
            "day_type": label,
            "service_ids": "|".join(members),
            "blocks": int(sub["blocks"].sum()),
            "revenue_trips": int(sub["revenue_trips"].sum()),
        }
        row.update(sub[list(COMPONENTS)].sum().to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_date(value: str) -> dt.date:
    """Parse ``YYYYMMDD``."""
    return datetime.strptime(str(value).strip(), "%Y%m%d").date()


def period_hours(
    services: pd.DataFrame,
    active: Mapping[str, set[dt.date]],
    types: Mapping[str, set[str]],
    cfg: Config,
) -> pd.DataFrame:
    """Total the service hours over each configured period, date by date.

    Raises:
        ValueError: If a period's dates are invalid or PERIOD_CALENDAR is unknown.
    """
    if cfg.period_calendar not in {"gtfs", "config"}:
        raise ValueError(
            f"PERIOD_CALENDAR must be 'gtfs' or 'config', not {cfg.period_calendar!r}."
        )
    known = set(services.index)
    rows = []
    for name, first, last in cfg.periods:
        start, end = _parse_date(first), _parse_date(last)
        if end < start:
            raise ValueError(f"Period {name!r} ends before it starts.")
        holidays = {_parse_date(d) for d in cfg.holiday_dates}
        if cfg.use_federal_holidays:
            for year in range(start.year, end.year + 2):
                holidays |= federal_holidays_observed(year)
        days: Counter = Counter()
        totals = pd.Series(0.0, index=list(COMPONENTS))
        no_service = 0
        day = start
        while day <= end:
            if cfg.period_calendar == "gtfs":
                running = service_ids_active_on(active, day) & known
                holiday = any("Holiday" in types.get(s, set()) for s in running)
            else:
                holiday = day in holidays
                lookup = cfg.holiday_service_day_type if holiday else WEEKDAY_LABELS[day.weekday()]
                running = {s for s in known if lookup in types.get(s, set())}
            days["Holiday" if holiday else WEEKDAY_LABELS[day.weekday()]] += 1
            no_service += not running
            for sid in running:
                totals += services.loc[sid, list(COMPONENTS)].astype(float)
            day += dt.timedelta(days=1)
        row: dict[str, Any] = {
            "period": name,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "calendar": cfg.period_calendar,
            "days": (end - start).days + 1,
            "weekdays": days["Weekday"],
            "saturdays": days["Saturday"],
            "sundays": days["Sunday"],
            "holidays": days["Holiday"],
            "days_without_service": no_service,
        }
        row.update(totals.to_dict())
        rows.append(row)
        if no_service:
            logging.warning("Period %s: %d day(s) with no service counted.", name, no_service)
    return pd.DataFrame(rows)


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
    """
    schedule = load_schedule(cfg.gtfs_path, cfg.tods_path)
    blocks = block_hours(schedule.movements, schedule.stop_places)
    services = service_hours(blocks)
    types = service_day_types(schedule.active_dates, cfg.service_day_types)
    tables = {
        "block": blocks,
        "service": services.reset_index(),
        "day_type": day_type_hours(services, types),
        "period": period_hours(services, schedule.active_dates, types, cfg),
    }
    for key, table in tables.items():
        if not table.empty:
            table = add_derived(table, cfg.layover_in_revenue_hours, schedule.tods_supplied)
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

    summary = [
        f"GTFS input:        {cfg.gtfs_path}",
        f"TODS input:        {cfg.tods_path if cfg.tods_path else '(none: revenue only)'}",
        f"Blocks:            {len(blocks)}",
        f"Period calendar:   {cfg.period_calendar}",
    ]
    for row in tables["day_type"].to_dict("records"):
        summary.append(
            f"{row['day_type']:<9} vehicle revenue h {row['vehicle_revenue_hours']:>9.2f}  "
            f"nonrevenue h {row['nonrevenue_hours']:>8.2f}"
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
    )
    try:
        tables = run(cfg)
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        logging.error("%s", exc)
        return 1
    for row in tables["period"].to_dict("records"):
        logging.info(
            "%s: %.1f vehicle revenue hours, %s nonrevenue hours.",
            row["period"],
            row["vehicle_revenue_hours"],
            "n/a" if pd.isna(row["nonrevenue_hours"]) else f"{row['nonrevenue_hours']:.1f}",
        )
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
