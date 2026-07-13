"""Vet stop-level ridership data coverage against GTFS scheduled service.

Compares a stop-level ridership export (vendor ridecheck output, APC rollup,
or the ``ridership_by_route_and_stop`` table produced by
``ridership_from_tides.py``) against what the GTFS feed says should be served
on each day type, and flags both directions of disagreement:

* **Missing** -- stops (and, when a route column is configured, route-stop
  pairs) that GTFS schedules on a day type but that have *no record at all*
  in that day type's ridership data. Whether the cause is a broken data
  pipeline, a stop dropped from the counting system, or a stale feed, the
  gap deserves a look before the data is used.
* **Unexpected** -- ridership records at stops the feed does not know, or on
  day types the feed schedules no service for. These catch stale stop IDs,
  renumbered stops, and feed errors.

A fundamental ambiguity is designed around rather than hidden: most vendor
exports drop stops with zero boardings, so "absent from the data" cannot be
distinguished from "genuinely unused". The missing-stop output therefore
carries the *scheduled service intensity* (average scheduled visits per day)
as a severity cue -- a stop scheduled 40 times a day with no record points at
the pipeline; a twice-a-day stop is probably just unused -- and rows are
sorted by it.

Day types
---------
Ridership exports rarely carry a day-type column, so each day type's file(s)
are designated explicitly in ``RIDERSHIP_FILES``. Provide any subset: a day
type with no files is skipped. On the GTFS side, the calendars are expanded
date by date over the analysis window (defaulting to the feed's validity
span), exception dates are dropped by default so holidays do not dilute the
weekly pattern, and a stop is *expected* on a day type when it is scheduled
on at least ``MIN_SHARE_OF_DAYS`` of that day type's service dates -- so a
one-Saturday special event does not make a stop "Saturday-served".

Matching
--------
Vendor exports identify stops by ``stop_id`` or by the public-facing
``stop_code``, and routes by ``route_id`` or the rider-facing short name;
``STOP_MATCH_FIELD`` / ``ROUTE_MATCH_FIELD`` choose which GTFS field the
export's values are compared against. Excel float artifacts (``1001.0``) are
cleaned automatically. When several GTFS stops share a ``stop_code`` (or
several routes a short name), they are pooled under that key and the
``n_gtfs_stops`` / ``n_gtfs_routes`` columns say so.

Outputs
-------
  1) ``ridership_missing_stops.csv`` -- expected stops with no ridership
     record, per day type, heaviest scheduled service first.
  2) ``ridership_unexpected_stops.csv`` -- ridership records with no matching
     expectation, per day type, with a ``reason`` column.
  3) ``ridership_missing_route_stops.csv`` /
     ``ridership_unexpected_route_stops.csv`` -- the same at the (route,
     stop) grain, written only when ``ROUTE_COLUMN`` is configured. The
     route-level missing table is restricted to stops that *do* appear in
     that day type's data (a wholly missing stop is already reported once at
     stop level).
  4) A ``_runlog.txt`` sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--gtfs`` /
``--output-dir``) and run from a shell, ArcGIS Pro's Python window, or a
Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import os
import zipfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_PATH: str = r"Path\To\Your\GTFS_Folder_Or_Zip"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Map each day type to the ridership export file(s) covering it. Provide any
# subset of the three day types; multiple files per day type are pooled.
# CSV and Excel (.xlsx) files are supported.
RIDERSHIP_FILES: Mapping[str, Sequence[str]] = {
    "Weekday": (r"Path\To\Your\weekday_ridership.csv",),
    "Saturday": (),
    "Sunday": (),
}

# Ridership column names. ROUTE_COLUMN may be "" when the export has no route
# column -- the two route-level outputs are then skipped. BOARDINGS_COLUMN is
# optional context for the unexpected-stops output ("" to skip).
STOP_ID_COLUMN: str = "STOP_ID"
ROUTE_COLUMN: str = "ROUTE_NAME"
BOARDINGS_COLUMN: str = "BOARD_ALL"

# Which GTFS field the export's stop/route values are matched against:
# STOP_MATCH_FIELD:  "stop_id" or "stop_code"
# ROUTE_MATCH_FIELD: "route_id" or "route_short_name"
STOP_MATCH_FIELD: str = "stop_id"
ROUTE_MATCH_FIELD: str = "route_short_name"

# Analysis window, "YYYY-MM-DD". Leave empty to use the feed's validity span.
# Narrow it to the span the ridership data actually covers when the feed is
# much longer-lived than the export.
START_DATE: str = ""
END_DATE: str = ""

# Drop dates carrying any calendar_dates.txt exception (holidays / special
# service), so day-type expectations reflect the normal weekly pattern.
# Ignored (with a warning) for feeds defined only by calendar_dates.txt.
EXCLUDE_EXCEPTION_DATES: bool = True

# Additional dates to drop, "YYYY-MM-DD" (e.g. weather shutdowns).
EXCLUDE_DATES: Sequence[str] = ()

# A stop (or route-stop pair) is *expected* on a day type when it is scheduled
# on at least this share of that day type's service dates. 0.5 means "a
# majority of days", so special-event service does not create expectations.
MIN_SHARE_OF_DAYS: float = 0.5

LOG_LEVEL: int = logging.INFO

# Filenames.
MISSING_STOPS_FILENAME: str = r"ridership_missing_stops.csv"
UNEXPECTED_STOPS_FILENAME: str = r"ridership_unexpected_stops.csv"
MISSING_ROUTE_STOPS_FILENAME: str = r"ridership_missing_route_stops.csv"
UNEXPECTED_ROUTE_STOPS_FILENAME: str = r"ridership_unexpected_route_stops.csv"

# When True, a failed run-log write aborts the script so an output is never
# left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

VALID_DAY_TYPES: tuple = ("Weekday", "Saturday", "Sunday")

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class Config(NamedTuple):
    """Runtime configuration for a ridership-coverage run."""

    gtfs_path: Path
    output_dir: Path
    ridership_files: Mapping[str, Sequence[str]] = RIDERSHIP_FILES
    stop_id_column: str = STOP_ID_COLUMN
    route_column: str = ROUTE_COLUMN
    boardings_column: str = BOARDINGS_COLUMN
    stop_match_field: str = STOP_MATCH_FIELD
    route_match_field: str = ROUTE_MATCH_FIELD
    start_date: str = START_DATE
    end_date: str = END_DATE
    exclude_exception_dates: bool = EXCLUDE_EXCEPTION_DATES
    exclude_dates: Sequence[str] = EXCLUDE_DATES
    min_share_of_days: float = MIN_SHARE_OF_DAYS


# =============================================================================
# CANONICAL HELPERS (copied verbatim from utils/ per CONTRIBUTING.md)
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


def expand_service_dates(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Expand GTFS calendars into one row per active (service_date, service_id).

    Both ``calendar.txt`` (weekly patterns bounded by a date range) and
    ``calendar_dates.txt`` (per-date add/remove exceptions) are honored, so
    feeds that use either convention -- or both -- expand correctly. Exception
    type 1 rows add a (date, service) pair; type 2 rows remove one, including
    pairs the weekly pattern would otherwise activate.

    Args:
        calendar: Parsed ``calendar.txt`` with string columns, or ``None`` /
            empty when the feed has no such file.
        calendar_dates: Parsed ``calendar_dates.txt`` with string columns, or
            ``None`` / empty when the feed has no such file.
        start_date: First service date (inclusive) of the expansion window.
        end_date: Last service date (inclusive) of the expansion window.

    Returns:
        DataFrame with normalized datetime ``service_date`` and string
        ``service_id`` columns, one row per active pair, de-duplicated and
        sorted. May be empty if no service falls inside the window.

    Raises:
        ValueError: If the window is inverted, both calendar inputs are
            missing/empty, or a required column is absent.
    """
    window_start = start_date.normalize()
    window_end = end_date.normalize()
    if window_start > window_end:
        raise ValueError(f"start_date {window_start.date()} is after end_date {window_end.date()}.")

    has_calendar = calendar is not None and not calendar.empty
    has_dates = calendar_dates is not None and not calendar_dates.empty
    if not has_calendar and not has_dates:
        raise ValueError(
            "Neither calendar.txt nor calendar_dates.txt has rows; cannot expand service dates."
        )

    day_columns = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    frames: list[pd.DataFrame] = []

    if has_calendar:
        required = ["service_id", "start_date", "end_date", *day_columns]
        missing = [col for col in required if col not in calendar.columns]
        if missing:
            raise ValueError(f"calendar.txt is missing required column(s): {', '.join(missing)}")
        cal = calendar.copy()
        cal["_start"] = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
        cal["_end"] = pd.to_datetime(cal["end_date"], format="%Y%m%d", errors="coerce")
        cal_long = cal.melt(
            id_vars=["service_id", "_start", "_end"],
            value_vars=day_columns,
            var_name="_day_name",
            value_name="_runs",
        )
        cal_long["_dow"] = cal_long["_day_name"].map(
            {name: i for i, name in enumerate(day_columns)}
        )
        dates = pd.DataFrame({"service_date": pd.date_range(window_start, window_end, freq="D")})
        dates["_dow"] = dates["service_date"].dt.dayofweek
        merged = dates.merge(cal_long, on="_dow", how="inner")
        keep = (
            (merged["_runs"].astype(str).str.strip() == "1")
            & (merged["service_date"] >= merged["_start"])
            & (merged["service_date"] <= merged["_end"])
        )
        frames.append(merged.loc[keep, ["service_date", "service_id"]])

    removed: Optional[pd.DataFrame] = None
    if has_dates:
        required = ["service_id", "date", "exception_type"]
        missing = [col for col in required if col not in calendar_dates.columns]
        if missing:
            raise ValueError(
                f"calendar_dates.txt is missing required column(s): {', '.join(missing)}"
            )
        cd = calendar_dates.copy()
        cd["service_date"] = pd.to_datetime(cd["date"], format="%Y%m%d", errors="coerce")
        cd = cd.loc[cd["service_date"].between(window_start, window_end)]
        exception = cd["exception_type"].astype(str).str.strip()
        frames.append(cd.loc[exception == "1", ["service_date", "service_id"]])
        removed = cd.loc[exception == "2", ["service_date", "service_id"]]

    active = pd.concat(frames, ignore_index=True)
    active["service_id"] = active["service_id"].astype(str)
    active = active.drop_duplicates()

    if removed is not None and not removed.empty:
        removed = removed.copy()
        removed["service_id"] = removed["service_id"].astype(str)
        active = active.merge(
            removed.drop_duplicates(),
            on=["service_date", "service_id"],
            how="left",
            indicator=True,
        )
        active = active.loc[active["_merge"] == "left_only", ["service_date", "service_id"]]

    return active.sort_values(["service_date", "service_id"], ignore_index=True)


def classify_day_type(dates: pd.Series) -> pd.Series:
    """Label each date ``Weekday`` / ``Saturday`` / ``Sunday`` by day of week.

    Holiday-aware relabeling (e.g. a Monday running Sunday service) is
    deliberately out of scope: callers that want holiday-like dates excluded
    should drop the dates returned by ``find_exception_dates`` first, so both
    sides of a schedule-vs-observed comparison see the same date pool.

    Args:
        dates: Series of datetime-like values.

    Returns:
        String Series aligned to ``dates`` holding ``"Weekday"``,
        ``"Saturday"``, or ``"Sunday"``.
    """
    dow = pd.to_datetime(dates).dt.dayofweek
    return dow.map({5: "Saturday", 6: "Sunday"}).fillna("Weekday").astype(str)


def find_exception_dates(
    calendar_dates: Optional[pd.DataFrame],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> set[pd.Timestamp]:
    """Return dates carrying any ``calendar_dates.txt`` exception row.

    These are the holiday-like dates whose service differs from the weekly
    pattern (service added, removed, or swapped to another day's schedule).
    Coverage comparisons typically exclude them so a Monday running a Sunday
    schedule does not dilute the Weekday expectation -- mirroring how vendor
    ridership and ridecheck exports usually treat holidays. Only meaningful
    for feeds that define a weekly baseline in ``calendar.txt``; for feeds
    built purely from ``calendar_dates.txt``, every service day is an
    "exception" and callers should skip the exclusion instead.

    Args:
        calendar_dates: Parsed ``calendar_dates.txt`` with string columns, or
            ``None`` / empty.
        start_date: Optional inclusive lower bound on the returned dates.
        end_date: Optional inclusive upper bound on the returned dates.

    Returns:
        Set of normalized ``pd.Timestamp`` dates having at least one exception
        row of either type (possibly empty).
    """
    if calendar_dates is None or calendar_dates.empty or "date" not in calendar_dates.columns:
        return set()
    parsed = pd.to_datetime(calendar_dates["date"], format="%Y%m%d", errors="coerce").dropna()
    if start_date is not None:
        parsed = parsed.loc[parsed >= start_date.normalize()]
    if end_date is not None:
        parsed = parsed.loc[parsed <= end_date.normalize()]
    return set(parsed)


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
# GTFS LOADING & EXPECTED SERVICE
# =============================================================================


def load_optional_gtfs_table(gtfs_path: str, file_name: str) -> Optional[pd.DataFrame]:
    """Load one GTFS file, returning ``None`` when the feed does not carry it.

    Args:
        gtfs_path: GTFS folder or ``.zip`` archive path.
        file_name: File to attempt, e.g. ``"calendar.txt"``.

    Returns:
        The parsed table, or ``None`` when the file is absent (many feeds ship
        only one of ``calendar.txt`` / ``calendar_dates.txt``).
    """
    try:
        key = file_name.replace(".txt", "")
        return load_gtfs_data(gtfs_path, files=(file_name,))[key]
    except OSError:
        logging.info("GTFS file %s not present in feed; continuing without it.", file_name)
        return None


def resolve_feed_window(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    start_override: str = "",
    end_override: str = "",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve the analysis window from the feed validity span and overrides.

    Args:
        calendar: Parsed ``calendar.txt`` or ``None``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.
        start_override: ``"YYYY-MM-DD"`` or empty for the feed's first date.
        end_override: ``"YYYY-MM-DD"`` or empty for the feed's last date.

    Returns:
        Tuple of normalized (window_start, window_end).

    Raises:
        ValueError: If no calendar dates are parseable or the window is empty.
    """
    candidates: List[pd.Timestamp] = []
    if calendar is not None and not calendar.empty:
        starts = pd.to_datetime(calendar["start_date"], format="%Y%m%d", errors="coerce")
        ends = pd.to_datetime(calendar["end_date"], format="%Y%m%d", errors="coerce")
        candidates.extend([starts.min(), ends.max()])
    if calendar_dates is not None and not calendar_dates.empty:
        dates = pd.to_datetime(calendar_dates["date"], format="%Y%m%d", errors="coerce")
        candidates.extend([dates.min(), dates.max()])
    candidates = [ts for ts in candidates if pd.notna(ts)]
    if not candidates:
        raise ValueError(
            "Could not determine the feed's validity span: no parseable dates in "
            "calendar.txt or calendar_dates.txt."
        )
    feed_start = min(candidates).normalize()
    feed_end = max(candidates).normalize()
    logging.info("Feed validity span: %s to %s.", feed_start.date(), feed_end.date())

    window_start = pd.Timestamp(start_override).normalize() if start_override else feed_start
    window_end = pd.Timestamp(end_override).normalize() if end_override else feed_end
    if window_start > window_end:
        raise ValueError(
            f"Analysis window is empty ({window_start.date()} to {window_end.date()}). "
            "Check START_DATE/END_DATE against the feed's validity span."
        )
    logging.info("Analysis window: %s to %s.", window_start.date(), window_end.date())
    return window_start, window_end


def resolve_excluded_dates(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    exclude_exception_dates: bool = EXCLUDE_EXCEPTION_DATES,
    exclude_dates: Sequence[str] = EXCLUDE_DATES,
) -> set:
    """Union the exception-date exclusion with manually configured dates.

    Args:
        calendar: Parsed ``calendar.txt`` or ``None``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.
        window_start: Analysis window start (inclusive).
        window_end: Analysis window end (inclusive).
        exclude_exception_dates: Drop dates carrying calendar_dates exceptions.
        exclude_dates: Extra ``"YYYY-MM-DD"`` dates to drop.

    Returns:
        Set of normalized timestamps to exclude from the analysis.
    """
    excluded: set = set()
    if exclude_exception_dates:
        if calendar is None or calendar.empty:
            logging.warning(
                "EXCLUDE_EXCEPTION_DATES is on, but this feed defines service only via "
                "calendar_dates.txt, so every service day is an 'exception'. Skipping the "
                "exclusion; holidays in the window will dilute day-type expectations."
            )
        else:
            excluded |= find_exception_dates(calendar_dates, window_start, window_end)
    excluded |= {pd.Timestamp(d).normalize() for d in exclude_dates}
    if excluded:
        logging.info(
            "Excluding %d date(s) from the analysis (exceptions/holidays and "
            "EXCLUDE_DATES entries).",
            len(excluded),
        )
    return excluded


def build_scheduled_instances(
    trips: pd.DataFrame,
    service_dates: pd.DataFrame,
    excluded_dates: set,
) -> pd.DataFrame:
    """Build one row per scheduled (service_date, trip) instance.

    Args:
        trips: Parsed GTFS ``trips.txt``.
        service_dates: Output of :func:`expand_service_dates`.
        excluded_dates: Dates to drop (see :func:`resolve_excluded_dates`).

    Returns:
        DataFrame with ``service_date``, ``trip_id``, ``route_id``,
        ``direction_id``, and ``day_type`` columns.

    Raises:
        ValueError: If ``trips.txt`` lacks a required column.
    """
    required = ["trip_id", "service_id", "route_id"]
    missing = [col for col in required if col not in trips.columns]
    if missing:
        raise ValueError(f"trips.txt is missing required column(s): {', '.join(missing)}")
    t = trips.copy()
    if "direction_id" not in t.columns:
        t["direction_id"] = ""
    t["service_id"] = t["service_id"].astype(str)
    instances = service_dates.merge(
        t[["trip_id", "service_id", "route_id", "direction_id"]],
        on="service_id",
        how="inner",
    )
    if excluded_dates:
        instances = instances.loc[~instances["service_date"].isin(list(excluded_dates))]
    instances = instances.drop(columns=["service_id"]).copy()
    instances["day_type"] = classify_day_type(instances["service_date"])
    return instances


def attach_stop_times(scheduled: pd.DataFrame, stop_times: pd.DataFrame) -> pd.DataFrame:
    """Expand scheduled trip instances into expected (date, trip, stop) visits.

    Args:
        scheduled: Output of :func:`build_scheduled_instances`.
        stop_times: Parsed GTFS ``stop_times.txt``.

    Returns:
        One row per expected trip-visit: ``service_date``, ``trip_id``,
        ``stop_id``, ``route_id``, ``direction_id``, ``day_type``.

    Raises:
        ValueError: If ``stop_times.txt`` lacks a required column.
    """
    required = ["trip_id", "stop_id"]
    missing = [col for col in required if col not in stop_times.columns]
    if missing:
        raise ValueError(f"stop_times.txt is missing required column(s): {', '.join(missing)}")
    expected = scheduled.merge(stop_times[["trip_id", "stop_id"]], on="trip_id", how="inner")
    return expected.drop_duplicates(["service_date", "trip_id", "stop_id"], ignore_index=True)


def attach_match_keys(
    expected: pd.DataFrame,
    stops: pd.DataFrame,
    routes: pd.DataFrame,
    stop_match_field: str = STOP_MATCH_FIELD,
    route_match_field: str = ROUTE_MATCH_FIELD,
) -> pd.DataFrame:
    """Attach the configured stop/route matching keys to expected visits.

    Args:
        expected: Output of :func:`attach_stop_times`.
        stops: Parsed GTFS ``stops.txt``.
        routes: Parsed GTFS ``routes.txt``.
        stop_match_field: ``"stop_id"`` or ``"stop_code"``.
        route_match_field: ``"route_id"`` or ``"route_short_name"``.

    Returns:
        ``expected`` with ``stop_key`` and ``route_key`` columns. Rows whose
        stop (or route) has a blank matching key are dropped with a warning --
        they cannot be compared against the export.

    Raises:
        ValueError: If a match field is invalid or missing from the feed.
    """
    if stop_match_field not in ("stop_id", "stop_code"):
        raise ValueError(
            f"STOP_MATCH_FIELD must be 'stop_id' or 'stop_code', got '{stop_match_field}'."
        )
    if route_match_field not in ("route_id", "route_short_name"):
        raise ValueError(
            f"ROUTE_MATCH_FIELD must be 'route_id' or 'route_short_name', got "
            f"'{route_match_field}'."
        )
    out = expected

    if stop_match_field == "stop_id":
        out = out.copy()
        out["stop_key"] = normalize_key_series(out["stop_id"])
    else:
        if "stop_code" not in stops.columns:
            raise ValueError(
                "STOP_MATCH_FIELD is 'stop_code' but this feed's stops.txt has no stop_code column."
            )
        lookup = stops[["stop_id", "stop_code"]].copy()
        lookup["stop_key"] = normalize_key_series(lookup["stop_code"])
        out = out.merge(lookup[["stop_id", "stop_key"]], on="stop_id", how="left")

    if route_match_field == "route_id":
        out = out.copy()
        out["route_key"] = normalize_key_series(out["route_id"])
    else:
        if "route_short_name" not in routes.columns:
            raise ValueError(
                "ROUTE_MATCH_FIELD is 'route_short_name' but this feed's routes.txt has "
                "no route_short_name column."
            )
        lookup = routes[["route_id", "route_short_name"]].copy()
        lookup["route_key"] = normalize_key_series(lookup["route_short_name"])
        out = out.merge(lookup[["route_id", "route_key"]], on="route_id", how="left")

    blank = (
        out["stop_key"].isna()
        | (out["stop_key"] == "")
        | out["route_key"].isna()
        | (out["route_key"] == "")
    )
    n_blank = int(blank.sum())
    if n_blank:
        logging.warning(
            "Dropping %d expected trip-visit(s) whose stop or route has a blank "
            "%s/%s value -- they cannot be matched against the ridership export.",
            n_blank,
            stop_match_field,
            route_match_field,
        )
        out = out.loc[~blank]
    return out.copy()


def count_service_days(service_dates: pd.DataFrame, excluded_dates: set) -> pd.Series:
    """Count distinct service dates per day type, after exclusions.

    Args:
        service_dates: Output of :func:`expand_service_dates`.
        excluded_dates: Dates to drop (see :func:`resolve_excluded_dates`).

    Returns:
        Series indexed by day type with the number of distinct dates on which
        the feed activates any service.
    """
    dates = service_dates.drop_duplicates("service_date")
    if excluded_dates:
        dates = dates.loc[~dates["service_date"].isin(list(excluded_dates))]
    day_types = classify_day_type(dates["service_date"])
    counts = day_types.value_counts()
    for day_type, count in counts.items():
        logging.info("Service dates in window: %s = %d", day_type, int(count))
    return counts


def summarize_expected_service(
    expected_keyed: pd.DataFrame,
    n_days_by_type: pd.Series,
    group_cols: List[str],
    min_share_of_days: float = MIN_SHARE_OF_DAYS,
) -> pd.DataFrame:
    """Summarize expected visits to *group_cols*, judging expectedness.

    Args:
        expected_keyed: Output of :func:`attach_match_keys`.
        n_days_by_type: Output of :func:`count_service_days`.
        group_cols: e.g. ``["day_type", "stop_key"]`` or
            ``["day_type", "route_key", "stop_key"]``.
        min_share_of_days: Share of the day type's dates a group must be
            scheduled on to count as expected.

    Returns:
        One row per group with ``days_served``, ``n_days``,
        ``share_of_days_served``, ``avg_daily_sched_visits``, an ``expected``
        boolean, and representative ``stop_id`` / ``n_gtfs_stops`` (plus
        ``n_gtfs_routes`` at the route grain) identification columns.
    """
    agg_spec: dict = {
        "days_served": ("service_date", "nunique"),
        "scheduled_visits": ("service_date", "size"),
        "stop_id": ("stop_id", "first"),
        "n_gtfs_stops": ("stop_id", "nunique"),
    }
    if "route_key" in group_cols:
        agg_spec["n_gtfs_routes"] = ("route_id", "nunique")
    grouped = expected_keyed.groupby(group_cols, as_index=False).agg(**agg_spec)
    grouped["n_days"] = grouped["day_type"].map(n_days_by_type).fillna(0).astype(int)
    grouped["share_of_days_served"] = (
        grouped["days_served"] / grouped["n_days"].where(grouped["n_days"] > 0)
    ).round(3)
    grouped["avg_daily_sched_visits"] = (
        grouped["scheduled_visits"] / grouped["n_days"].where(grouped["n_days"] > 0)
    ).round(1)
    grouped["expected"] = grouped["share_of_days_served"].fillna(0) >= min_share_of_days
    return grouped


# =============================================================================
# RIDERSHIP LOADING
# =============================================================================


def normalize_key_series(values: pd.Series) -> pd.Series:
    """Normalize ID-like values for matching: trim and drop Excel's ``.0``.

    Args:
        values: Raw ID values (any dtype).

    Returns:
        String Series with whitespace trimmed and the float artifact Excel
        adds to numeric IDs (``"1001.0"``) stripped back to ``"1001"``.
    """
    s = values.astype(str).str.strip()
    is_float_int = s.str.fullmatch(r"\d+\.0")
    return s.mask(is_float_int, s.str.slice(0, -2))


def read_ridership_table(path: Path) -> pd.DataFrame:
    """Read one ridership export as strings, by file extension.

    Args:
        path: ``.csv`` or ``.xlsx`` file path.

    Returns:
        The parsed table with every column as strings.

    Raises:
        ValueError: If the extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, dtype=str)
    raise ValueError(
        f"Unsupported ridership file type '{suffix}' for '{path}' (use .csv or .xlsx)."
    )


def load_ridership_records(
    ridership_files: Mapping[str, Sequence[str]],
    stop_id_column: str = STOP_ID_COLUMN,
    route_column: str = ROUTE_COLUMN,
    boardings_column: str = BOARDINGS_COLUMN,
) -> pd.DataFrame:
    """Load and pool every configured ridership file into one tidy table.

    Args:
        ridership_files: Day type -> file path(s) mapping.
        stop_id_column: Column holding the stop identifier.
        route_column: Column holding the route identifier ("" = none).
        boardings_column: Column holding boardings ("" = none).

    Returns:
        One row per input record with ``day_type``, ``stop_key``,
        ``route_key`` ("" when unconfigured), and float ``boardings`` (NaN
        when unconfigured).

    Raises:
        ValueError: On an unknown day-type key, a missing file, or a missing
            required column.
    """
    unknown = [day for day in ridership_files if day not in VALID_DAY_TYPES]
    if unknown:
        raise ValueError(
            f"RIDERSHIP_FILES has unknown day type key(s) {unknown}; "
            f"valid keys are {list(VALID_DAY_TYPES)}."
        )
    frames: List[pd.DataFrame] = []
    for day_type, paths in ridership_files.items():
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.exists():
                raise ValueError(f"Ridership file for {day_type} not found: {path}")
            df = read_ridership_table(path)
            needed = [stop_id_column] + ([route_column] if route_column else [])
            missing = [col for col in needed if col not in df.columns]
            if missing:
                raise ValueError(
                    f"Ridership file '{path}' is missing column(s): {', '.join(missing)}. "
                    "Check STOP_ID_COLUMN/ROUTE_COLUMN against the file's header."
                )
            tidy = pd.DataFrame(
                {
                    "day_type": day_type,
                    "stop_key": normalize_key_series(df[stop_id_column]),
                }
            )
            tidy["route_key"] = normalize_key_series(df[route_column]) if route_column else ""
            if boardings_column and boardings_column in df.columns:
                tidy["boardings"] = pd.to_numeric(df[boardings_column], errors="coerce")
            else:
                tidy["boardings"] = float("nan")
            tidy = tidy.loc[tidy["stop_key"] != ""]
            frames.append(tidy)
            logging.info("Loaded %d %s ridership record(s) from '%s'.", len(tidy), day_type, path)
    if not frames:
        raise ValueError("RIDERSHIP_FILES configures no files at all; nothing to check.")
    return pd.concat(frames, ignore_index=True)


def summarize_ridership_presence(
    records: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """Collapse ridership records to one row per *group_cols*.

    Args:
        records: Output of :func:`load_ridership_records`.
        group_cols: e.g. ``["day_type", "stop_key"]``.

    Returns:
        One row per group with ``n_records`` and summed ``boardings``.
    """
    return records.groupby(group_cols, as_index=False).agg(
        n_records=("stop_key", "size"),
        boardings=("boardings", "sum"),
    )


# =============================================================================
# COMPARISONS
# =============================================================================


def find_missing_stops(
    expected_stop: pd.DataFrame,
    presence_stop: pd.DataFrame,
    expected_route: pd.DataFrame,
    stop_names: pd.DataFrame,
    day_types_checked: Sequence[str],
) -> pd.DataFrame:
    """Expected stops with no ridership record at all on a day type.

    Args:
        expected_stop: Stop-grain output of :func:`summarize_expected_service`.
        presence_stop: Stop-grain output of
            :func:`summarize_ridership_presence`.
        expected_route: Route-grain output of
            :func:`summarize_expected_service` (for the routes-serving list).
        stop_names: ``stops.txt`` columns ``stop_id`` / ``stop_name``.
        day_types_checked: Day types that actually have ridership files.

    Returns:
        One row per missing (day type, stop), heaviest scheduled service
        first.
    """
    candidates = expected_stop.loc[
        expected_stop["expected"] & expected_stop["day_type"].isin(list(day_types_checked))
    ]
    merged = candidates.merge(
        presence_stop[["day_type", "stop_key"]],
        on=["day_type", "stop_key"],
        how="left",
        indicator=True,
    )
    missing = merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    routes = (
        expected_route.loc[expected_route["expected"]]
        .groupby(["day_type", "stop_key"], as_index=False)
        .agg(routes_serving=("route_key", lambda keys: ", ".join(sorted(set(keys)))))
    )
    missing = missing.merge(routes, on=["day_type", "stop_key"], how="left")
    if "stop_name" in stop_names.columns:
        missing = missing.merge(
            stop_names[["stop_id", "stop_name"]].drop_duplicates("stop_id"),
            on="stop_id",
            how="left",
        )
    columns = [
        "day_type",
        "stop_key",
        "stop_id",
        "stop_name",
        "routes_serving",
        "avg_daily_sched_visits",
        "days_served",
        "n_days",
        "share_of_days_served",
        "n_gtfs_stops",
    ]
    missing = missing[[col for col in columns if col in missing.columns]]
    return missing.sort_values(
        ["avg_daily_sched_visits", "day_type", "stop_key"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def find_unexpected_stops(
    presence_stop: pd.DataFrame,
    expected_stop: pd.DataFrame,
    stop_names: pd.DataFrame,
    min_share_of_days: float = MIN_SHARE_OF_DAYS,
) -> pd.DataFrame:
    """Ridership stop records with no matching GTFS expectation.

    Args:
        presence_stop: Stop-grain output of
            :func:`summarize_ridership_presence`.
        expected_stop: Stop-grain output of :func:`summarize_expected_service`
            (all rows, including below-share ones).
        stop_names: ``stops.txt`` columns ``stop_id`` / ``stop_name`` /
            optionally ``stop_code``.
        min_share_of_days: The configured threshold (used in the reason text).

    Returns:
        One row per unexpected (day type, stop) with a ``reason`` column,
        heaviest ridership first.
    """
    merged = presence_stop.merge(
        expected_stop[["day_type", "stop_key", "stop_id", "expected", "share_of_days_served"]],
        on=["day_type", "stop_key"],
        how="left",
    )
    known_keys = set(expected_stop["stop_key"])
    unexpected = merged.loc[~merged["expected"].eq(True)].copy()
    if unexpected.empty:
        return pd.DataFrame(
            columns=[
                "day_type",
                "stop_key",
                "stop_id",
                "stop_name",
                "reason",
                "n_records",
                "boardings",
                "share_of_days_served",
            ]
        )

    unexpected["reason"] = "no service scheduled this day type"
    unexpected.loc[unexpected["share_of_days_served"].notna(), "reason"] = (
        f"scheduled on under {min_share_of_days:.0%} of days"
    )
    unexpected.loc[~unexpected["stop_key"].isin(known_keys), "reason"] = "stop not in GTFS"

    if "stop_name" in stop_names.columns:
        unexpected = unexpected.merge(
            stop_names[["stop_id", "stop_name"]].drop_duplicates("stop_id"),
            on="stop_id",
            how="left",
        )
    columns = [
        "day_type",
        "stop_key",
        "stop_id",
        "stop_name",
        "reason",
        "n_records",
        "boardings",
        "share_of_days_served",
    ]
    unexpected = unexpected[[col for col in columns if col in unexpected.columns]]
    return unexpected.sort_values(
        ["boardings", "day_type", "stop_key"],
        ascending=[False, True, True],
        ignore_index=True,
        na_position="last",
    )


def find_missing_route_stops(
    expected_route: pd.DataFrame,
    presence_route: pd.DataFrame,
    presence_stop: pd.DataFrame,
    stop_names: pd.DataFrame,
    day_types_checked: Sequence[str],
) -> pd.DataFrame:
    """Expected route-stop pairs absent from data at stops that ARE present.

    A stop entirely absent from a day type's data is reported once at stop
    level; this view catches the subtler case where the stop reports for some
    routes but a scheduled route is missing from it.

    Args:
        expected_route: Route-grain output of
            :func:`summarize_expected_service`.
        presence_route: Route-grain output of
            :func:`summarize_ridership_presence`.
        presence_stop: Stop-grain output of
            :func:`summarize_ridership_presence`.
        stop_names: ``stops.txt`` columns ``stop_id`` / ``stop_name``.
        day_types_checked: Day types that actually have ridership files.

    Returns:
        One row per missing (day type, route, stop), heaviest scheduled
        service first.
    """
    candidates = expected_route.loc[
        expected_route["expected"] & expected_route["day_type"].isin(list(day_types_checked))
    ]
    present_stops = presence_stop[["day_type", "stop_key"]].drop_duplicates()
    candidates = candidates.merge(present_stops, on=["day_type", "stop_key"], how="inner")
    merged = candidates.merge(
        presence_route[["day_type", "route_key", "stop_key"]],
        on=["day_type", "route_key", "stop_key"],
        how="left",
        indicator=True,
    )
    missing = merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    if "stop_name" in stop_names.columns:
        missing = missing.merge(
            stop_names[["stop_id", "stop_name"]].drop_duplicates("stop_id"),
            on="stop_id",
            how="left",
        )
    columns = [
        "day_type",
        "route_key",
        "stop_key",
        "stop_id",
        "stop_name",
        "avg_daily_sched_visits",
        "days_served",
        "n_days",
        "share_of_days_served",
        "n_gtfs_routes",
    ]
    missing = missing[[col for col in columns if col in missing.columns]]
    return missing.sort_values(
        ["avg_daily_sched_visits", "day_type", "route_key", "stop_key"],
        ascending=[False, True, True, True],
        ignore_index=True,
    )


def find_unexpected_route_stops(
    presence_route: pd.DataFrame,
    expected_route: pd.DataFrame,
    expected_stop: pd.DataFrame,
    min_share_of_days: float = MIN_SHARE_OF_DAYS,
) -> pd.DataFrame:
    """Ridership route-stop pairs with no matching GTFS expectation.

    Stops unknown to GTFS are already reported at stop level and are skipped
    here; this view catches known stops paired with routes GTFS does not
    schedule there on that day type, and route identifiers GTFS does not know.

    Args:
        presence_route: Route-grain output of
            :func:`summarize_ridership_presence`.
        expected_route: Route-grain output of
            :func:`summarize_expected_service` (all rows).
        expected_stop: Stop-grain output of :func:`summarize_expected_service`
            (to identify stops known to GTFS).
        min_share_of_days: The configured threshold (used in the reason text).

    Returns:
        One row per unexpected (day type, route, stop) with a ``reason``
        column, heaviest ridership first.
    """
    known_stop_keys = set(expected_stop["stop_key"])
    known_route_keys = set(expected_route["route_key"])
    pool = presence_route.loc[presence_route["stop_key"].isin(known_stop_keys)]
    merged = pool.merge(
        expected_route[["day_type", "route_key", "stop_key", "expected", "share_of_days_served"]],
        on=["day_type", "route_key", "stop_key"],
        how="left",
    )
    unexpected = merged.loc[~merged["expected"].eq(True)].copy()
    if unexpected.empty:
        return pd.DataFrame(
            columns=[
                "day_type",
                "route_key",
                "stop_key",
                "reason",
                "n_records",
                "boardings",
                "share_of_days_served",
            ]
        )

    unexpected["reason"] = "route does not serve this stop on this day type"
    unexpected.loc[unexpected["share_of_days_served"].notna(), "reason"] = (
        f"scheduled on under {min_share_of_days:.0%} of days"
    )
    unexpected.loc[~unexpected["route_key"].isin(known_route_keys), "reason"] = "route not in GTFS"

    columns = [
        "day_type",
        "route_key",
        "stop_key",
        "reason",
        "n_records",
        "boardings",
        "share_of_days_served",
    ]
    unexpected = unexpected[[col for col in columns if col in unexpected.columns]]
    return unexpected.sort_values(
        ["boardings", "day_type", "route_key", "stop_key"],
        ascending=[False, True, True, True],
        ignore_index=True,
        na_position="last",
    )


# =============================================================================
# EXPORT & RUN LOG
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def export_tables(
    tables: Mapping[str, pd.DataFrame],
    out_dir: Path,
) -> List[Path]:
    """Write each named table to *out_dir* as CSV.

    Args:
        tables: Filename -> DataFrame mapping.
        out_dir: Directory to write into (created if needed).

    Returns:
        Paths of the files written.
    """
    ensure_dir(out_dir)
    paths: List[Path] = []
    for filename, table in tables.items():
        path = out_dir / filename
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


def resolve_source_file() -> Optional[Path]:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "ridership_gtfs_coverage_runlog.txt"

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
        "RIDERSHIP VS GTFS COVERAGE RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
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
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# PIPELINE
# =============================================================================


def run(cfg: Config) -> dict:
    """Execute the full ridership-coverage pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        Dict of the comparison tables, keyed by output filename (also written
        to disk).
    """
    gtfs_path = str(cfg.gtfs_path)
    gtfs = load_gtfs_data(
        gtfs_path, files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
    )
    stops = gtfs["stops"]
    routes = gtfs["routes"]
    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]
    calendar = load_optional_gtfs_table(gtfs_path, "calendar.txt")
    calendar_dates = load_optional_gtfs_table(gtfs_path, "calendar_dates.txt")

    window_start, window_end = resolve_feed_window(
        calendar, calendar_dates, cfg.start_date, cfg.end_date
    )
    excluded = resolve_excluded_dates(
        calendar,
        calendar_dates,
        window_start,
        window_end,
        cfg.exclude_exception_dates,
        cfg.exclude_dates,
    )

    service_dates = expand_service_dates(calendar, calendar_dates, window_start, window_end)
    if service_dates.empty:
        raise ValueError(
            "The GTFS calendars activate no service inside the analysis window "
            f"({window_start.date()} to {window_end.date()})."
        )

    scheduled = build_scheduled_instances(trips, service_dates, excluded)
    expected_visits = attach_stop_times(scheduled, stop_times)
    expected_keyed = attach_match_keys(
        expected_visits, stops, routes, cfg.stop_match_field, cfg.route_match_field
    )
    if expected_keyed.empty:
        raise ValueError("No expected trip-visits remain in the analysis window after exclusions.")
    n_days_by_type = count_service_days(service_dates, excluded)

    expected_stop = summarize_expected_service(
        expected_keyed, n_days_by_type, ["day_type", "stop_key"], cfg.min_share_of_days
    )
    expected_route = summarize_expected_service(
        expected_keyed,
        n_days_by_type,
        ["day_type", "route_key", "stop_key"],
        cfg.min_share_of_days,
    )

    records = load_ridership_records(
        cfg.ridership_files, cfg.stop_id_column, cfg.route_column, cfg.boardings_column
    )
    day_types_checked = sorted(records["day_type"].unique())
    for day_type in day_types_checked:
        if int(n_days_by_type.get(day_type, 0)) == 0:
            logging.warning(
                "Ridership file(s) supplied for %s, but the feed schedules no %s service "
                "in the window -- every %s stop will be reported as unexpected.",
                day_type,
                day_type,
                day_type,
            )
    skipped = [d for d in VALID_DAY_TYPES if d not in day_types_checked]
    if skipped:
        logging.info("No ridership files for: %s. Those day types are skipped.", ", ".join(skipped))

    presence_stop = summarize_ridership_presence(records, ["day_type", "stop_key"])
    missing_stops = find_missing_stops(
        expected_stop, presence_stop, expected_route, stops, day_types_checked
    )
    unexpected_stops = find_unexpected_stops(
        presence_stop, expected_stop, stops, cfg.min_share_of_days
    )
    tables = {
        MISSING_STOPS_FILENAME: missing_stops,
        UNEXPECTED_STOPS_FILENAME: unexpected_stops,
    }

    if cfg.route_column:
        presence_route = summarize_ridership_presence(
            records, ["day_type", "route_key", "stop_key"]
        )
        tables[MISSING_ROUTE_STOPS_FILENAME] = find_missing_route_stops(
            expected_route, presence_route, presence_stop, stops, day_types_checked
        )
        tables[UNEXPECTED_ROUTE_STOPS_FILENAME] = find_unexpected_route_stops(
            presence_route, expected_route, expected_stop, cfg.min_share_of_days
        )

    if missing_stops.empty:
        logging.info("No expected stops are missing from the ridership data.")
    else:
        worst = missing_stops.iloc[0]
        logging.warning(
            "%d expected (day type, stop) row(s) have no ridership record. Heaviest: "
            "stop %s on %s (%.1f scheduled visits/day, routes: %s).",
            len(missing_stops),
            worst["stop_key"],
            worst["day_type"],
            worst["avg_daily_sched_visits"],
            worst.get("routes_serving", ""),
        )
    if not unexpected_stops.empty:
        logging.warning(
            "%d ridership (day type, stop) row(s) have no matching GTFS expectation "
            "(see the reason column).",
            len(unexpected_stops),
        )

    paths = export_tables(tables, cfg.output_dir)
    for path in paths:
        logging.info("Wrote table: %s", path)

    summary_lines = [
        f"Analysis window:            {window_start.date()} to {window_end.date()}",
        f"Dates excluded:             {len(excluded)}",
        f"Day types checked:          {', '.join(day_types_checked)}",
        f"Ridership records:          {len(records)}",
        f"Expected (day, stop) rows:  {int(expected_stop['expected'].sum())}",
        f"Missing stops:              {len(missing_stops)}",
        f"Unexpected stops:           {len(unexpected_stops)}",
    ]
    if cfg.route_column:
        summary_lines.extend(
            [
                f"Missing route-stops:        {len(tables[MISSING_ROUTE_STOPS_FILENAME])}",
                f"Unexpected route-stops:     {len(tables[UNEXPECTED_ROUTE_STOPS_FILENAME])}",
            ]
        )
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return tables


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Vet stop-level ridership data coverage against GTFS scheduled service."
    )
    parser.add_argument("--gtfs", default=GTFS_PATH, help="Path to GTFS folder or .zip.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point. Validates placeholder paths before doing any work."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(argv)

    configured_files = [p for paths in RIDERSHIP_FILES.values() for p in paths]
    if (
        args.gtfs == GTFS_PATH
        or not configured_files
        or any(p.startswith("Path\\To\\") for p in configured_files)
    ):
        logging.warning(
            "GTFS_PATH and/or RIDERSHIP_FILES are still placeholders. Update the "
            "CONFIGURATION section (and optionally pass --gtfs) before running."
        )
        return

    cfg = Config(
        gtfs_path=Path(args.gtfs).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        ridership_files=RIDERSHIP_FILES,
        stop_id_column=STOP_ID_COLUMN,
        route_column=ROUTE_COLUMN,
        boardings_column=BOARDINGS_COLUMN,
        stop_match_field=STOP_MATCH_FIELD,
        route_match_field=ROUTE_MATCH_FIELD,
        start_date=START_DATE,
        end_date=END_DATE,
        exclude_exception_dates=EXCLUDE_EXCEPTION_DATES,
        exclude_dates=EXCLUDE_DATES,
        min_share_of_days=MIN_SHARE_OF_DAYS,
    )

    if not cfg.gtfs_path.exists():
        logging.warning("GTFS feed not found: %s", cfg.gtfs_path)
        return

    run(cfg)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
