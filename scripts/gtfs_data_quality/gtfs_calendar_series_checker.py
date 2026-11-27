"""Check a series of GTFS feed versions for overlapping service dates.

Point the script at the feed versions one agency has published: ZIPs,
extracted folders, or a folder holding them. Every input is treated as one
series, successive versions of the same feed. When two versions schedule
service on the same date, or declare overlapping validity periods, it is
unclear which timetable applies, so each overlapping pair is logged as a
warning naming the feed files and the GTFS files inside them
(``calendar.txt``, ``calendar_dates.txt``, ``feed_info.txt``) to review.
Resolve each overlap by hand: edit one feed so every date belongs to a single
version, or drop a version from the series. Inputs are never edited.

Inputs
------
- ``GTFS_PATH``: one feed (``.zip`` or folder) or a folder of feeds;
  ``GTFS_PATHS`` optionally lists feeds explicitly. ``RECURSIVE_SCAN`` also
  searches subfolders.
- Each feed needs ``calendar.txt`` and/or ``calendar_dates.txt``.
  ``trips.txt`` limits a feed's dates to the services its trips use;
  ``feed_info.txt`` supplies the declared validity period; ``agency.txt``
  supplies the agency names compared across the series.

Outputs
-------
- ``feed_inventory.csv``: one row per feed: status (OK / REVIEW / ERROR),
  agency names, declared vs. actual service dates, warnings, errors.
- ``feed_overlaps.csv``: one row per overlapping pair: shared service dates,
  overlapping declared periods, and the files to review in each feed.
- ``feed_overlap_dates.csv``: each shared service date, with the service_ids
  each feed runs on it.
- ``gtfs_calendar_series_checker_runlog.txt``: verbatim configuration,
  resolved inputs, and a run summary.

Method and limits
-----------------
Active dates are ``calendar.txt`` patterns plus ``calendar_dates.txt``
exceptions, expanded in full (long placeholder ranges are not clamped), for
the services ``trips.txt`` references. Dates are GTFS service dates, not
clock times after midnight. A feed with invalid calendar data is reported as
ERROR and left out of the comparison. Feeds whose agency names differ are
flagged, since they may not belong to the same series.
GTFS reference: https://gtfs.org/documentation/schedule/reference/

Typical usage
-------------
Edit the CONFIGURATION paths, or run from a shell:
    python gtfs_calendar_series_checker.py --gtfs old.zip new.zip --output-dir results
    python gtfs_calendar_series_checker.py --gtfs feed_folder --output-dir results
Exit status is 0 when every feed was read (overlaps are warnings), 1 when any
feed failed, and 2 when the placeholder paths are unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import os
import sys
import zipfile
from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import IO, Any, List, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# One feed (.zip or folder), or a folder holding several versions of one feed.
GTFS_PATH: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me
# Optional explicit list of feeds; when populated, it overrides GTFS_PATH.
# Example: [Path(r"C:\GTFS\old.zip"), Path(r"C:\GTFS\new.zip")]
GTFS_PATHS: list[Path] = []
# Also search subfolders of a collection folder for feeds.
RECURSIVE_SCAN: bool = False
# Calendar ranges longer than this are reported as errors, never truncated.
MAX_CALENDAR_SPAN_DAYS: int = 36600

OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me
INVENTORY_FILENAME: str = "feed_inventory.csv"
OVERLAPS_FILENAME: str = "feed_overlaps.csv"
OVERLAP_DATES_FILENAME: str = "feed_overlap_dates.csv"

# When True, a failed run-log write aborts the script so outputs are never
# left untraced. Set False only for genuinely read-only output locations.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

# =============================================================================
# FUNCTIONS
# =============================================================================

# ---- REUSABLE HELPERS (copied from utils/gtfs_helpers.py) ------------------


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


def require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    """Reject missing fields before calculating dates or trip counts."""
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}.")


def required_id(value: Any, label: str) -> str:
    """Preserve text identifiers, including leading zeros and literal 'NA'."""
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label}: blank identifier.")
    return result


def parse_date(value: Any, label: str) -> dt.date:
    """Require an eight-digit, valid GTFS date."""
    text = str(value).strip()
    if len(text) != 8 or not text.isascii() or not text.isdigit():
        raise ValueError(f"{label}: expected YYYYMMDD, got {text!r}.")
    try:
        return dt.datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"{label}: invalid date {text!r}.") from exc


def _load_optional(gtfs_path: Path, file_name: str) -> pd.DataFrame | None:
    """Read one GTFS file as text, or return None when the feed lacks it.

    Blank cells stay ``""`` and values such as ``NA`` stay text
    (``keep_default_na=False``), so identifiers compare exactly. A file that is
    present but empty, unparseable, or found twice in a zip raises ValueError.
    """
    try:
        with open_gtfs_file(gtfs_path, file_name) as handle:
            table = pd.read_csv(
                handle, dtype=str, keep_default_na=False, encoding="utf-8-sig", low_memory=False
            )
    except FileNotFoundError:
        return None
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{file_name} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Parser error in {file_name}: {exc}") from exc
    table.columns = [str(column).strip() for column in table.columns]
    return table.apply(lambda col: col.str.strip())


def expand_calendar_dates(
    calendar_df: pd.DataFrame | None,
    calendar_dates_df: pd.DataFrame | None = None,
    max_days_per_service: int = MAX_CALENDAR_SPAN_DAYS,
) -> tuple[dict[str, set[dt.date]], dict[str, set[dt.date]], dict[str, set[dt.date]]]:
    """Expand calendar ranges in full and apply calendar_dates exceptions exactly.

    Unlike ``expand_service_active_dates`` in ``utils/calendar_helpers.py``,
    invalid rows raise ValueError and long spans are rejected rather than
    clamped to a window around today: clamping could hide an overlap between
    historical feed versions. Only the calendar fields this check uses are
    validated, not the whole GTFS.

    Returns:
        ``(active, pattern, added)``, each keyed by service_id: the dates each
        service runs, the dates its ``calendar.txt`` pattern gives before
        exceptions, and the dates ``calendar_dates.txt`` adds. The last two
        say which file schedules a given date.
    """
    if max_days_per_service < 1:
        raise ValueError("MAX_CALENDAR_SPAN_DAYS must be positive.")
    day_cols = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    pattern: dict[str, set[dt.date]] = {}
    added: dict[str, set[dt.date]] = {}
    if calendar_df is not None:
        require_columns(
            calendar_df, {"service_id", "start_date", "end_date", *day_cols}, "calendar.txt"
        )
        if calendar_df["service_id"].duplicated().any():
            raise ValueError("calendar.txt contains duplicate service_id values.")
        for row in calendar_df.to_dict("records"):
            sid = required_id(row["service_id"], "calendar.txt service_id")
            start = parse_date(row["start_date"], f"{sid} start_date")
            end = parse_date(row["end_date"], f"{sid} end_date")
            span = (end - start).days + 1
            if not 1 <= span <= max_days_per_service:
                raise ValueError(
                    f"Service {sid}: invalid or excessive date span "
                    f"{start} to {end} ({span} days); limit={max_days_per_service}."
                )
            flags = [str(row[c]).strip() for c in day_cols]
            if any(flag not in {"0", "1"} for flag in flags):
                raise ValueError(f"Service {sid}: day-of-week flags must be 0 or 1.")
            dates = pattern.setdefault(sid, set())
            for offset in range(span):
                date = start + dt.timedelta(days=offset)
                if flags[date.weekday()] == "1":
                    dates.add(date)
    active = {sid: set(dates) for sid, dates in pattern.items()}
    if calendar_dates_df is not None:
        require_columns(
            calendar_dates_df, {"service_id", "date", "exception_type"}, "calendar_dates.txt"
        )
        if calendar_dates_df.duplicated(["service_id", "date"]).any():
            raise ValueError("calendar_dates.txt contains duplicate service_id/date pairs.")
        for row in calendar_dates_df.to_dict("records"):
            sid = required_id(row["service_id"], "calendar_dates.txt service_id")
            date = parse_date(row["date"], f"{sid} exception date")
            kind = str(row["exception_type"]).strip()
            dates = active.setdefault(sid, set())
            if kind == "1":
                dates.add(date)
                added.setdefault(sid, set()).add(date)
            elif kind == "2":
                dates.discard(date)
            else:
                raise ValueError(f"Service {sid}: invalid exception_type {kind!r}.")
    return active, pattern, added


def discover_feeds(inputs: Sequence[Path], recursive: bool = False) -> list[Path]:
    """Find ZIPs and extracted feed folders; deduplicate resolved input paths.

    A folder containing calendar.txt/calendar_dates.txt is one feed. Otherwise
    inspect its immediate children, or descendants when recursive=True. ZIPs
    are opened during inspection, so corrupt archives remain in the inventory.
    """
    found: dict[Path, None] = {}
    visited: set[Path] = set()

    def is_feed(path: Path) -> bool:
        return any((path / name).is_file() for name in ("calendar.txt", "calendar_dates.txt"))

    def visit(path: Path, explicit: bool = False) -> None:
        path = path.expanduser().resolve()
        if path in visited:
            return
        visited.add(path)
        if path.is_file() and path.suffix.lower() == ".zip":
            found[path] = None
        elif path.is_dir() and is_feed(path):
            found[path] = None
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() == ".zip":
                    visit(child)
                elif child.is_dir() and (is_feed(child) or recursive):
                    visit(child)
        elif explicit:
            # Report explicit missing/invalid paths alongside usable feeds.
            found[path] = None

    for item in inputs:
        visit(Path(item), explicit=True)
    if not found:
        raise ValueError("No GTFS ZIPs or extracted calendar folders found in the inputs.")
    return sorted(found)


class FeedAudit:
    """One feed version's identity, service dates, and review notes."""

    def __init__(self, path: Path, label: str | None = None) -> None:
        """Start an empty audit for the (resolved) feed at *path*."""
        suffix = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
        self.path = path
        self.feed_id = f"{path.stem}_{suffix}"
        self.label = label or path.name
        self.active: dict[str, set[dt.date]] = {}
        self.pattern: dict[str, set[dt.date]] = {}
        self.added: dict[str, set[dt.date]] = {}
        self.operating: dict[str, set[dt.date]] = {}
        self.trip_counts: dict[str, int] = {}
        self.agency_names: tuple[str, ...] = ()
        self.metadata: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.error = ""

    @property
    def dates(self) -> set[dt.date]:
        """Get dates with referenced service, or calendar dates if trips is absent."""
        return {date for dates in self.operating.values() for date in dates}

    @property
    def status(self) -> str:
        """ERROR when unreadable, REVIEW when warnings exist, otherwise OK."""
        return "ERROR" if self.error else "REVIEW" if self.warnings else "OK"


def inspect_feed(path: Path, label: str | None = None) -> FeedAudit:
    """Read one feed's calendar, trips, feed_info, and agency names.

    A failure is kept on the returned audit (``error``) rather than raised, so
    the rest of the series can still be checked.
    """
    path = path.expanduser().resolve()
    feed = FeedAudit(path, label)
    logging.info("Inspecting %s", path)
    try:
        if not path.exists():
            raise OSError(f"Input does not exist: {path}")
        if not path.is_dir() and path.suffix.lower() != ".zip":
            raise ValueError("Expected a GTFS folder or .zip archive.")
        calendar = _load_optional(path, "calendar.txt")
        exceptions = _load_optional(path, "calendar_dates.txt")
        if calendar is None and exceptions is None:
            raise ValueError("Neither calendar.txt nor calendar_dates.txt is present.")
        feed.active, feed.pattern, feed.added = expand_calendar_dates(
            calendar, exceptions, MAX_CALENDAR_SPAN_DAYS
        )
        trips = _load_optional(path, "trips.txt")
        if trips is None:
            feed.operating = dict(feed.active)
            feed.metadata["date_basis"] = "calendar_only"
            feed.warnings.append("trips.txt absent: calendar dates are unverified by trips.")
        else:
            require_columns(trips, {"trip_id", "service_id"}, "trips.txt")
            for column in ("trip_id", "service_id"):
                if trips[column].eq("").any():
                    raise ValueError(f"trips.txt contains blank {column} values.")
            if trips["trip_id"].duplicated().any():
                raise ValueError("trips.txt contains duplicate trip_id values.")
            feed.trip_counts = {
                str(sid): int(count) for sid, count in trips["service_id"].value_counts().items()
            }
            missing = set(feed.trip_counts) - set(feed.active)
            if missing:
                raise ValueError(f"Trips reference undefined service_ids: {sorted(missing)}")
            feed.operating = {sid: feed.active[sid] for sid in feed.trip_counts}
            feed.metadata["date_basis"] = "services_referenced_by_trips"
            unused = set(feed.active) - set(feed.trip_counts)
            feed.metadata["unused_service_ids"] = "; ".join(sorted(unused))
            if unused:
                feed.warnings.append(
                    f"{len(unused)} unused service_id(s) excluded from feed dates."
                )
            if trips.empty:
                feed.warnings.append("trips.txt has no trips; no operating dates established.")
        dates = feed.dates
        if not dates:
            feed.warnings.append("No active service dates.")
        info = _load_optional(path, "feed_info.txt")
        declared_start = declared_end = None
        if info is not None and not info.empty:
            if len(info) != 1:
                raise ValueError("feed_info.txt must contain exactly one data row when present.")
            row = info.iloc[0]
            for name in ("feed_version", "feed_publisher_name"):
                feed.metadata[name] = str(row.get(name, ""))
            start_text = str(row.get("feed_start_date", ""))
            end_text = str(row.get("feed_end_date", ""))
            if start_text:
                declared_start = parse_date(start_text, "feed_info feed_start_date")
            if end_text:
                declared_end = parse_date(end_text, "feed_info feed_end_date")
            if declared_start and declared_end and declared_end < declared_start:
                raise ValueError("feed_info feed_end_date precedes feed_start_date.")
            if bool(declared_start) != bool(declared_end):
                feed.warnings.append(
                    "feed_info.txt gives only one of feed_start_date/feed_end_date; "
                    "its declared period is not compared."
                )
        feed.metadata["declared_start_date"] = declared_start
        feed.metadata["declared_end_date"] = declared_end
        if declared_start and declared_end:
            outside = {d for d in dates if not declared_start <= d <= declared_end}
            feed.metadata["active_days_outside_declared_period"] = len(outside)
            if outside:
                feed.warnings.append(
                    f"{len(outside)} scheduled date(s) fall outside the feed_info.txt period; "
                    "they are still compared."
                )
        agency = _load_optional(path, "agency.txt")
        if agency is not None and "agency_name" in agency.columns:
            feed.agency_names = tuple(sorted(set(agency["agency_name"]) - {""}))
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        feed.error = str(exc)
        logging.error("%s: %s", feed.label, exc)
    for warning in feed.warnings:
        logging.warning("%s: %s", feed.label, warning)
    return feed


def check_agency_names(feeds: Sequence[FeedAudit]) -> None:
    """Flag feeds whose agency names differ from the rest of the series.

    Every input is treated as one series, so a differing ``agency_name``
    suggests a feed from another system (or a renamed agency). Names are
    compared ignoring case and spacing. Feeds outside the single most common
    set of names are flagged; when no set is most common, every named feed is.
    """
    readable = [feed for feed in feeds if not feed.error]
    keys = {
        feed.feed_id: frozenset(" ".join(name.lower().split()) for name in feed.agency_names)
        for feed in readable
    }
    for feed in readable:
        if not keys[feed.feed_id]:
            feed.warnings.append(
                "agency.txt gives no agency_name; cannot confirm this feed belongs to the series."
            )
            logging.warning("%s: %s", feed.label, feed.warnings[-1])
    ranked = Counter(key for key in keys.values() if key).most_common()
    if len(ranked) < 2:
        return
    common = ranked[0][0] if ranked[0][1] > ranked[1][1] else None
    groups = []
    for key, _ in ranked:
        members = [feed for feed in readable if keys[feed.feed_id] == key]
        names = "; ".join(members[0].agency_names)
        groups.append(f"{names} in {', '.join(feed.label for feed in members)}")
    logging.warning("Feeds in the series carry different agency names: %s.", " | ".join(groups))
    for feed in readable:
        key = keys[feed.feed_id]
        if key and key != common:
            feed.warnings.append(
                f"agency_name ({'; '.join(feed.agency_names)}) differs from other feeds in "
                "the series; confirm this feed belongs to it."
            )
            logging.warning("%s: %s", feed.label, feed.warnings[-1])


def _id_list(ids: set[str], limit: int = 10) -> str:
    """Join sorted ids, shortening long lists to the first *limit*."""
    ordered = sorted(ids)
    shown = ", ".join(ordered[:limit])
    return shown if len(ordered) <= limit else f"{shown} and {len(ordered) - limit} more"


def _declared_overlap(left: FeedAudit, right: FeedAudit) -> tuple[dt.date, dt.date] | None:
    """Return the overlap of two complete feed_info.txt periods, or None."""
    starts = [feed.metadata.get("declared_start_date") for feed in (left, right)]
    ends = [feed.metadata.get("declared_end_date") for feed in (left, right)]
    if None in starts or None in ends:
        return None
    start, end = max(starts), min(ends)
    return (start, end) if start <= end else None


def files_to_review(
    feed: FeedAudit, shared: set[dt.date], declared: tuple[dt.date, dt.date] | None
) -> str:
    """Name the GTFS files (and service_ids) in *feed* that create an overlap.

    A shared date comes from ``calendar.txt`` when the service's day-of-week
    pattern covers it and from ``calendar_dates.txt`` when an exception adds
    it (both when an addition repeats the pattern). ``feed_info.txt`` is named
    when the two feeds' declared periods overlap.
    """
    from_calendar: set[str] = set()
    from_exceptions: set[str] = set()
    for sid, dates in feed.operating.items():
        hit = dates & shared
        if hit & feed.pattern.get(sid, set()):
            from_calendar.add(sid)
        if hit & feed.added.get(sid, set()):
            from_exceptions.add(sid)
    parts: list[str] = []
    if from_calendar:
        parts.append(f"calendar.txt (service_id {_id_list(from_calendar)})")
    if from_exceptions:
        parts.append(f"calendar_dates.txt (dates added for service_id {_id_list(from_exceptions)})")
    if declared is not None:
        start = feed.metadata["declared_start_date"]
        end = feed.metadata["declared_end_date"]
        parts.append(f"feed_info.txt (feed_start_date {start:%Y%m%d}, feed_end_date {end:%Y%m%d})")
    return "; ".join(parts)


def compare_feeds(feeds: Sequence[FeedAudit]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find every pair of readable feeds that overlap and say what to review.

    Feeds are ordered by first service date, so ``feed_a`` is the earlier
    version. A pair is reported when the feeds share a service date or their
    complete ``feed_info.txt`` periods overlap. Each reported pair is logged as
    a warning and noted on both feeds' inventory rows; nothing is changed.
    Service IDs are local to each feed and never compared across feeds.
    """
    columns = [
        "feed_a",
        "feed_b",
        "file_a",
        "file_b",
        "shared_service_days",
        "shared_first_date",
        "shared_last_date",
        "declared_overlap_start",
        "declared_overlap_end",
        "a_only_active_days",
        "b_only_active_days",
        "active_date_relationship",
        "files_to_review_a",
        "files_to_review_b",
    ]
    detail_columns = ["feed_a", "feed_b", "date", "day_of_week", "service_ids_a", "service_ids_b"]
    readable = [feed for feed in feeds if not feed.error]
    feed_dates = {feed.feed_id: feed.dates for feed in readable}
    readable.sort(key=lambda feed: (min(feed_dates[feed.feed_id], default=dt.date.max), feed.label))
    pairs: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for left, right in combinations(readable, 2):
        a_dates, b_dates = feed_dates[left.feed_id], feed_dates[right.feed_id]
        shared = a_dates & b_dates
        declared = _declared_overlap(left, right)
        if not shared and declared is None:
            continue
        if not a_dates or not b_dates:
            relation = "one_or_both_empty"
        elif a_dates == b_dates:
            relation = "identical_active_dates"
        elif a_dates < b_dates:
            relation = "a_dates_contained_in_b"
        elif b_dates < a_dates:
            relation = "b_dates_contained_in_a"
        elif shared:
            relation = "partial_overlap"
        else:
            relation = "no_shared_active_dates"
        review_a = files_to_review(left, shared, declared)
        review_b = files_to_review(right, shared, declared)
        pairs.append(
            {
                "feed_a": left.feed_id,
                "feed_b": right.feed_id,
                "file_a": left.label,
                "file_b": right.label,
                "shared_service_days": len(shared),
                "shared_first_date": min(shared) if shared else None,
                "shared_last_date": max(shared) if shared else None,
                "declared_overlap_start": declared[0] if declared else None,
                "declared_overlap_end": declared[1] if declared else None,
                "a_only_active_days": len(a_dates - b_dates),
                "b_only_active_days": len(b_dates - a_dates),
                "active_date_relationship": relation,
                "files_to_review_a": review_a,
                "files_to_review_b": review_b,
            }
        )
        for date in sorted(shared):
            details.append(
                {
                    "feed_a": left.feed_id,
                    "feed_b": right.feed_id,
                    "date": date,
                    "day_of_week": date.strftime("%A"),
                    "service_ids_a": "; ".join(
                        sorted(sid for sid, dates in left.operating.items() if date in dates)
                    ),
                    "service_ids_b": "; ".join(
                        sorted(sid for sid, dates in right.operating.items() if date in dates)
                    ),
                }
            )
        found = []
        if shared:
            found.append(f"{len(shared)} shared service date(s), {min(shared)} to {max(shared)}")
        if declared:
            found.append(f"overlapping feed_info.txt periods, {declared[0]} to {declared[1]}")
        what = "; ".join(found)
        logging.warning(
            "Series overlap between %s and %s: %s. Resolve by hand: edit one feed so each "
            "date belongs to only one version, or drop one version from the series. "
            "Files to review: %s: %s | %s: %s",
            left.label,
            right.label,
            what,
            left.label,
            review_a,
            right.label,
            review_b,
        )
        for feed, other in ((left, right), (right, left)):
            feed.warnings.append(f"Overlaps {other.label} ({what}); see {OVERLAPS_FILENAME}.")
    return pd.DataFrame(pairs, columns=columns), pd.DataFrame(details, columns=detail_columns)


def feed_inventory(feeds: Sequence[FeedAudit]) -> pd.DataFrame:
    """List each feed's identity, declared vs. actual dates, and review notes."""
    rows: list[dict[str, Any]] = []
    for feed in feeds:
        ok = not feed.error
        dates = feed.dates if ok else set()
        rows.append(
            {
                "feed_id": feed.feed_id,
                "feed_name": feed.label,
                "feed_path": str(feed.path),
                "status": feed.status,
                "agency_names": "; ".join(feed.agency_names),
                "feed_version": feed.metadata.get("feed_version", ""),
                "feed_publisher_name": feed.metadata.get("feed_publisher_name", ""),
                "declared_start_date": feed.metadata.get("declared_start_date") if ok else None,
                "declared_end_date": feed.metadata.get("declared_end_date") if ok else None,
                "first_active_date": min(dates) if dates else None,
                "last_active_date": max(dates) if dates else None,
                "active_days": len(dates) if ok else None,
                "active_days_outside_declared_period": (
                    feed.metadata.get("active_days_outside_declared_period") if ok else None
                ),
                "service_ids": len(feed.active) if ok else None,
                "unused_service_ids": feed.metadata.get("unused_service_ids", ""),
                "trip_rows": (
                    sum(feed.trip_counts.values())
                    if ok and feed.metadata.get("date_basis") == "services_referenced_by_trips"
                    else None
                ),
                "date_basis": feed.metadata.get("date_basis", "") if ok else "",
                "warnings": " | ".join(feed.warnings),
                "error": feed.error,
            }
        )
    return pd.DataFrame(rows)


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "gtfs_calendar_series_checker_runlog.txt"
    source_name = globals().get("__file__")
    if source_name is None:
        config_text = "Notebook configuration (resolved values):\n" + "\n".join(
            f"{name} = {value!r}"
            for name, value in globals().items()
            if name.isupper() and not name.startswith("_")
        )
    else:
        try:
            config_text = extract_config_block(Path(source_name))
        except (OSError, ValueError) as exc:
            logging.error("Could not extract config block for run log: %s", exc)
            return False

    source_label = str(Path(source_name).resolve()) if source_name else "notebook"
    lines: List[str] = [
        "=" * 72,
        "GTFS CALENDAR SERIES CHECKER RUN LOG",
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
    gtfs_paths: str | Path | Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
    recursive: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Check the series and write the inventory, overlap, and run-log outputs.

    Unset arguments fall back to the CONFIGURATION block, so
    ``m.GTFS_PATH = ...; m.run()`` works after a plain import. Feed errors and
    overlaps are reported in the outputs, not raised.

    Returns:
        Output tables keyed ``feed_inventory``, ``feed_overlaps``, and
        ``feed_overlap_dates``.

    Raises:
        ValueError: No GTFS ZIPs or calendar folders were found in the inputs.
        OSError: The run log is required but cannot be written.
    """
    if gtfs_paths is None:
        inputs = list(GTFS_PATHS or [GTFS_PATH])
    elif isinstance(gtfs_paths, (str, os.PathLike)):
        inputs = [Path(gtfs_paths)]
    else:
        inputs = [Path(path) for path in gtfs_paths]
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    recursive = RECURSIVE_SCAN if recursive is None else recursive

    paths = discover_feeds(inputs, recursive)
    name_counts = Counter(path.name for path in paths)
    feeds = [
        inspect_feed(path, path.name if name_counts[path.name] == 1 else str(path))
        for path in paths
    ]
    check_agency_names(feeds)
    overlaps, overlap_dates = compare_feeds(feeds)
    tables = {
        "feed_inventory": feed_inventory(feeds),
        "feed_overlaps": overlaps,
        "feed_overlap_dates": overlap_dates,
    }

    failed = sum(bool(feed.error) for feed in feeds)
    summary = [
        *[f"Resolved input: {path}" for path in paths],
        f"Feeds: {len(feeds)}; read: {len(feeds) - failed}; failed: {failed} "
        "(failed feeds are not compared)",
        f"Overlapping pairs: {len(overlaps)}",
        f"Recursive scan: {recursive}; maximum calendar span: {MAX_CALENDAR_SPAN_DAYS} days",
        "Series: every input is one series; differing agency names are flagged.",
        "Overlap: a shared service date, or overlapping complete feed_info.txt periods.",
        "Active dates: full calendar.txt ranges plus calendar_dates.txt exceptions, for the "
        "services trips.txt references (all services when trips.txt is absent).",
        "Dates are GTFS service dates, not clock times after midnight. Inputs are not modified.",
        *[
            f"{feed.label}: {feed.status}"
            + (f"; {feed.error}" if feed.error else "")
            + (f"; {' | '.join(feed.warnings)}" if feed.warnings else "")
            for feed in feeds
        ],
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Trace the resolved inputs before publishing the CSVs.
    if not write_run_log(output_dir, summary) and REQUIRE_RUN_LOG:
        raise OSError(f"Cannot write required run log to {output_dir}.")
    filenames = {
        "feed_inventory": INVENTORY_FILENAME,
        "feed_overlaps": OVERLAPS_FILENAME,
        "feed_overlap_dates": OVERLAP_DATES_FILENAME,
    }
    for key, table in tables.items():
        path = output_dir / filenames[key]
        table.to_csv(path, index=False, encoding="utf-8-sig")
        logging.info("Written %s (%d rows)", path, len(table))
    if len(overlaps):
        logging.warning(
            "%d overlapping pair(s) need manual resolution; see the warnings above and %s.",
            len(overlaps),
            OVERLAPS_FILENAME,
        )
    logging.info(
        "Series check complete: %d feed(s), %d failed, %d overlapping pair(s).",
        len(feeds),
        failed,
        len(overlaps),
    )
    return tables


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the config block values."""
    parser = argparse.ArgumentParser(
        description=(
            "Check a series of GTFS feed versions for overlapping service dates. "
            "Defaults come from the configuration block at the top of this file."
        )
    )
    parser.add_argument(
        "--gtfs",
        type=Path,
        nargs="+",
        default=GTFS_PATHS or [GTFS_PATH],
        help="Feed ZIP(s), extracted folder(s), or a folder holding feeds.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for the CSVs and run log."
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=RECURSIVE_SCAN,
        help="Search subfolders beneath collection folders.",
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
        Process exit code: 0 when every feed was read (overlaps are warnings,
        not failures), 1 when any feed failed or no feeds were found, 2 when
        the CONFIGURATION paths are still placeholders.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sentinels = {Path(r"Path\To\Your\GTFS_Folder"), Path(r"Path\To\Your\Output_Folder")}
    if any(path in sentinels for path in [*args.gtfs, args.output_dir]):
        logging.warning(
            "GTFS_PATH/GTFS_PATHS and/or OUTPUT_DIR are still placeholders. Update the "
            "configuration block or pass --gtfs/--output-dir before running."
        )
        return 2
    try:
        tables = run(args.gtfs, args.output_dir, args.recursive)
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return int(tables["feed_inventory"]["status"].eq("ERROR").any())


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the config block stays in charge.
    raise SystemExit(main())
