"""Summarize a GTFS feed's service calendar so planners can see what runs when.

Agencies encode service days in wildly different ways: some publish a clean
weekday/Saturday/Sunday calendar, some run distinct Monday / midweek / Friday
schedules, and some scheduling-software exports stamp a weekday pattern on a
holiday-only service and then negate it date-by-date in ``calendar_dates.txt``.
This script expands every service_id into its real set of active dates
(calendar pattern × date range, plus exceptions) and reports what each one
actually is — the five-minute first look that tells you which service_ids
matter before configuring any other tool in this repository.

It also reports, for each day type (weekday / Saturday / Sunday), the
representative date and service_id set the repository's date-based scripts
would select, so their automatic choices are visible and checkable.

Inputs
------
- A GTFS feed (folder or ``.zip``) containing ``calendar.txt`` and/or
  ``calendar_dates.txt``. ``trips.txt``, when present, adds per-service
  trip counts.

Outputs
-------
- ``calendar_service_summary.csv``: one row per service_id — inferred labels
  (Weekday / Saturday / Sunday / Holiday), active-date count and range,
  per-year rate, day-of-week breakdown, holiday overlap, trip count, and
  sample dates.
- ``calendar_day_type_summary.csv``: one row per day type — the
  representative date, its service_ids, and their combined trip count.
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--gtfs`` /
``--output-dir``) and run from a shell, ArcGIS Pro's Python window, or a
Jupyter notebook. Read the INFO log or the CSVs — each service_id's real
operating pattern is spelled out either way.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
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
# === BEGIN CONFIG ===

GTFS_PATH: Path = Path(r"Path\To\Your\GTFS_Folder")  # folder or .zip  # ←–– change me
OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me
SERVICE_SUMMARY_FILENAME: str = r"calendar_service_summary.csv"
DAY_TYPE_SUMMARY_FILENAME: str = r"calendar_day_type_summary.csv"

# Classification thresholds — see utils/calendar_helpers.py for guidance on
# when to raise or lower them.
HOLIDAY_MAX_DAYS_PER_YEAR: float = 25.0
WEEKDAY_DOW_SHARE: float = 0.80

# How many example active dates to show per service_id in the summary CSV.
SAMPLE_DATES_SHOWN: int = 5

# When True, a failed run-log write aborts the script so outputs are never
# left untraced. Set False only for genuinely read-only output locations.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

# =============================================================================
# FUNCTIONS
# =============================================================================

# ---- REUSABLE HELPERS (copied from utils/gtfs_helpers.py) ------------------


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


# ---- REUSABLE HELPERS (copied from utils/time_helpers.py) ------------------


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


# ---- REUSABLE HELPERS (copied from utils/run_log.py) -----------------------


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


# ---- REUSABLE HELPERS (copied from utils/cli_helpers.py) -------------------


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


# ---- SCRIPT-SPECIFIC LOGIC --------------------------------------------------


def _load_optional(gtfs_path: Path, file_name: str) -> Optional[pd.DataFrame]:
    """Load one GTFS file via ``load_gtfs_data``, returning ``None`` if absent."""
    try:
        return load_gtfs_data(str(gtfs_path), files=(file_name,))[file_name[:-4]]
    except (OSError, ValueError) as exc:
        logging.info("%s not usable (%s) — continuing without it.", file_name, exc)
        return None


def _holidays_for(active_dates: Mapping[str, set[dt.date]]) -> set[dt.date]:
    """Observed federal holidays across the feed's active years (plus spillover)."""
    years = {d.year for dates in active_dates.values() for d in dates}
    holidays: set[dt.date] = set()
    if years:
        for year in range(min(years), max(years) + 2):
            holidays |= federal_holidays_observed(year)
    return holidays


def build_service_summary(
    active_dates: Mapping[str, set[dt.date]],
    labels: Mapping[str, set[str]],
    trip_counts: Mapping[str, int],
    holidays: set[dt.date],
    sample_dates_shown: int = SAMPLE_DATES_SHOWN,
) -> pd.DataFrame:
    """Build the one-row-per-service_id summary table.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        labels: Output of :func:`classify_service_ids`.
        trip_counts: Trips per service_id (empty mapping when trips.txt is
            unavailable).
        holidays: Observed federal holidays overlapping the feed.
        sample_dates_shown: Number of example dates listed per service.

    Returns:
        DataFrame sorted by service_id with classification and diagnostics.
    """
    day_names = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    rows: list[dict[str, Any]] = []
    for sid in sorted(active_dates):
        dates = active_dates[sid]
        row: dict[str, Any] = {
            "service_id": sid,
            "labels": " + ".join(sorted(labels.get(sid, set()))) or "(no active dates)",
            "active_days": len(dates),
        }
        if dates:
            span_days = (max(dates) - min(dates)).days + 1
            row["first_active_date"] = min(dates).isoformat()
            row["last_active_date"] = max(dates).isoformat()
            row["days_per_year"] = round(len(dates) / max(span_days / 365.25, 0.1), 1)
        else:
            row["first_active_date"] = ""
            row["last_active_date"] = ""
            row["days_per_year"] = 0.0
        for dow, name in enumerate(day_names):
            row[f"n_{name}"] = sum(1 for d in dates if d.weekday() == dow)
        row["holiday_days"] = sum(1 for d in dates if d in holidays)
        row["trip_count"] = trip_counts.get(sid, 0)
        row["sample_dates"] = "; ".join(d.isoformat() for d in sorted(dates)[:sample_dates_shown])
        rows.append(row)
    return pd.DataFrame(rows)


def build_day_type_summary(
    active_dates: Mapping[str, set[dt.date]],
    trip_counts: Mapping[str, int],
    holidays: set[dt.date],
) -> pd.DataFrame:
    """Build the representative-date table for weekday / Saturday / Sunday.

    Shows the date and service_id set the repository's date-based scripts
    would automatically select for each day type, with the warnings those
    selections would emit surfaced in the log.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        trip_counts: Trips per service_id (may be empty).
        holidays: Dates excluded as representative candidates.

    Returns:
        DataFrame with one row per day type.
    """
    rows: list[dict[str, Any]] = []
    for day_type in ("weekday", "saturday", "sunday"):
        row: dict[str, Any] = {"day_type": day_type}
        try:
            chosen, ids = representative_service_date(
                active_dates, day_type, exclude_dates=holidays
            )
            row["representative_date"] = chosen.isoformat()
            row["service_ids"] = "; ".join(sorted(ids))
            row["trip_count"] = sum(trip_counts.get(sid, 0) for sid in ids)
        except ValueError as exc:
            logging.warning("No representative %s date: %s", day_type, exc)
            row["representative_date"] = ""
            row["service_ids"] = ""
            row["trip_count"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "gtfs_calendar_inspector_runlog.txt"
    try:
        config_text = extract_config_block(Path(__file__))
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: List[str] = [
        "=" * 72,
        "GTFS CALENDAR INSPECTOR RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {Path(__file__).resolve()}",
        "",
        "-" * 72,
        "RUN SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim)",
        "-" * 72,
        "# === BEGIN CONFIG ===",
        config_text,
        "# === END CONFIG ===",
        "",
    ]
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logging.error("Could not write run log '%s': %s", log_path, exc)
        return False
    logging.info("Run log written → %s", log_path)
    return True


def run(
    gtfs_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect the feed's calendar and write both summary CSVs.

    Unset args fall back to the config block at the top of this file, so
    ``m.GTFS_PATH = ...; m.run()`` works after a plain import.

    Returns:
        Tuple of (service summary, day-type summary) DataFrames.

    Raises:
        ValueError: If the feed contains neither ``calendar.txt`` nor
            ``calendar_dates.txt``.
        OSError: If the run log is required but cannot be written.
    """
    gtfs_path = GTFS_PATH if gtfs_path is None else Path(gtfs_path)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)

    calendar = _load_optional(gtfs_path, "calendar.txt")
    calendar_dates = _load_optional(gtfs_path, "calendar_dates.txt")
    trips = _load_optional(gtfs_path, "trips.txt")
    if calendar is None and calendar_dates is None:
        raise ValueError(
            f"'{gtfs_path}' contains neither calendar.txt nor calendar_dates.txt — "
            "not a usable GTFS feed."
        )

    active = expand_service_active_dates(calendar, calendar_dates)
    labels = classify_service_ids(
        active,
        holiday_max_days_per_year=HOLIDAY_MAX_DAYS_PER_YEAR,
        dow_share=WEEKDAY_DOW_SHARE,
    )
    holidays = _holidays_for(active)
    trip_counts: dict[str, int] = {}
    if trips is not None and "service_id" in trips.columns:
        trip_counts = trips["service_id"].astype(str).str.strip().value_counts().to_dict()

    service_summary = build_service_summary(active, labels, trip_counts, holidays)
    day_type_summary = build_day_type_summary(active, trip_counts, holidays)

    for _, row in service_summary.iterrows():
        logging.info(
            "Service %-20s %-24s %5d active day(s)  %s → %s",
            row["service_id"],
            row["labels"],
            row["active_days"],
            row["first_active_date"] or "-",
            row["last_active_date"] or "-",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    service_path = output_dir / SERVICE_SUMMARY_FILENAME
    day_type_path = output_dir / DAY_TYPE_SUMMARY_FILENAME
    service_summary.to_csv(service_path, index=False)
    day_type_summary.to_csv(day_type_path, index=False)
    logging.info("Written → %s", service_path)
    logging.info("Written → %s", day_type_path)

    summary_lines = [
        f"GTFS feed:            {gtfs_path}",
        f"Service_ids:          {len(service_summary)}",
        f"Service summary CSV:  {service_path}",
        f"Day-type summary CSV: {day_type_path}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise OSError(
            f"Run log could not be written to '{output_dir}' and REQUIRE_RUN_LOG is True."
        )

    logging.info("Calendar inspection complete — %d service_id(s).", len(service_summary))
    return service_summary, day_type_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the config block values."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a GTFS feed's service calendar: what each service_id "
            "really is, and which dates the repository's date-based scripts "
            "would pick. Defaults come from the configuration block at the "
            "top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs", type=Path, default=GTFS_PATH, help="Path to the GTFS folder or .zip."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for the summary CSVs."
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
    sentinels = {Path(r"Path\To\Your\GTFS_Folder"), Path(r"Path\To\Your\Output_Folder")}
    if args.gtfs in sentinels or args.output_dir in sentinels:
        logging.warning(
            "GTFS_PATH and/or OUTPUT_DIR are still placeholders. Update the configuration "
            "block or pass --gtfs/--output-dir before running."
        )
        return 2
    try:
        run(gtfs_path=args.gtfs, output_dir=args.output_dir)
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the config block stays in charge.
    raise SystemExit(main())
