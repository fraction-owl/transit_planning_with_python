"""Stop-level data coverage: GTFS expected trip-visits vs TIDES ``stop_visits``.

For each stop, compares the trip-visits GTFS ``stop_times`` says should happen
(per Weekday / Saturday / Sunday) against the visits the TIDES ``stop_visits``
export actually contains, and flags stops whose coverage is low or zero. This
fills the "GTFS-shaped hole" that ``tides_stop_otp_flagger.py`` documents: an
observed-only export cannot reveal a stop whose AVL stop matching is
completely dead, because such a stop never appears in the export at all. With
GTFS as the denominator, those invisible stops are exactly what surfaces
first. It also differs from that flagger's route-baseline comparison on
purpose: here a uniformly under-covered route *does* flag, because the
baseline is the schedule, not the route's own average.

``trips_performed`` is a required input even though the coverage question is
about stops: TIDES ``stop_visits`` carries only ``trip_id_performed``, so the
trip table is the sole bridge to the GTFS ``trip_id`` (via
``trip_id_scheduled``) and to route attribution -- and it lets low coverage be
decomposed into "the trip never ran / was never recorded" (a trip-level
problem; see ``trip_coverage_tides.py``) versus "the trip ran but left no
trace at this stop" (the geofence / stop-matching defects this script hunts).

Method
------
1. Expand the GTFS calendars into per-date active service over the analysis
   window (feed validity intersected with the export's date span, unless
   ``START_DATE``/``END_DATE`` override it), dropping exception dates by
   default so holidays do not dilute day-type expectations.
2. Build one expected row per (date, trip, stop) from ``stop_times``, carrying
   route, direction, and day type.
3. Reduce ``stop_visits`` (bridged through in-service ``trips_performed``
   rows) to one observed record per (date, scheduled trip, stop), noting
   whether any row carried an actual timestamp or was marked Skipped.
4. Classify every expected trip-visit into exactly one outcome:
   ``observed`` (a visit with an actual timestamp), ``skipped`` (emitted but
   marked Skipped), ``visit_without_actual`` (emitted, schedule present, no
   AVL timestamp), ``visit_missing`` (trip performed, but no row at this stop
   -- the geofence/stop-matching signature), or ``trip_unrecorded`` (the trip
   itself has no usable record that day).
5. Roll up to (stop, day type) and (stop, route, day type): percent observed,
   per-outcome counts, and flags. A row is judged only when at least
   ``MIN_EXPECTED_VISITS`` visits are expected; it flags *invisible* when
   nothing was ever observed, and *low coverage* below
   ``LOW_COVERAGE_FLAG_PCT``.

Like the trip script, the GTFS-to-TIDES trip-ID match rate is reported as a
headline diagnostic (``MIN_VISIT_JOIN_RATE_PCT``): when the feed does not
correspond to the export's signup, coverage numbers are noise.

Reading the outcomes
--------------------
High ``n_visit_missing`` with healthy trip coverage points at the stop itself
(mis-placed geofence, stop moved in the field, stop absent from the AVL
configuration). High ``n_trip_unrecorded`` means the gap is upstream at the
trip level -- run ``trip_coverage_tides.py`` to localize it. ``skipped`` and
``visit_without_actual`` mirror the per-cause diagnostics in
``tides_stop_otp_flagger.py`` (operations vs AVL hardware vs export defects).

Timepoint-only exports
----------------------
Some AVL exports emit visits only at timepoints. Against a full ``stop_times``
expectation, every non-timepoint stop would then look invisible. Set
``TIMEPOINTS_ONLY = True`` to restrict both sides (GTFS rows with
``timepoint == 1`` and visits with ``timepoint == TRUE``); the script warns
when the export looks timepoint-only but the setting is off.

Outputs
-------
  1) ``stop_coverage.csv`` -- one row per (stop, day type) with coverage,
     outcome counts, and flags; flagged rows first.
  2) ``stop_route_day_detail.csv`` -- the same per (stop, route, day type),
     the evidence view that localizes a problem to specific routes.
  3) A ``_runlog.txt`` sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--gtfs`` /
``--stop-visits`` / ``--trips-performed`` / ``--output-dir``) and run from a
shell, ArcGIS Pro's Python window, or a Jupyter notebook.
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
STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Analysis window, "YYYY-MM-DD". Leave empty to use the intersection of the
# feed's validity span and the stop_visits date span. Set both explicitly to
# audit a window the export *should* cover -- the automatic window cannot see
# dates the export is entirely missing.
START_DATE: str = ""
END_DATE: str = ""

# Drop dates carrying any calendar_dates.txt exception (holidays / special
# service), so day-type expectations reflect the normal weekly pattern.
# Ignored (with a warning) for feeds defined only by calendar_dates.txt.
EXCLUDE_EXCEPTION_DATES: bool = True

# Additional dates to drop, "YYYY-MM-DD" (e.g. weather shutdowns).
EXCLUDE_DATES: Sequence[str] = ()

# Optional route filters (matched against GTFS route_id as a string).
# Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# Restrict both sides to timepoints: GTFS stop_times rows with timepoint == 1
# and visits with timepoint == TRUE. Use when the AVL export only emits
# timepoint visits, so ordinary stops are not misread as invisible.
TIMEPOINTS_ONLY: bool = False

# A (stop, day type) cell is flagged when its percent of expected trip-visits
# observed falls below this, provided at least MIN_EXPECTED_VISITS visits are
# expected (so a stop served a handful of times cannot flag on noise). Cells
# with zero observed visits are additionally flagged "invisible".
LOW_COVERAGE_FLAG_PCT: float = 50.0
MIN_EXPECTED_VISITS: int = 10

# Headline data-health diagnostic: the share of bridged visit records (in
# window) whose (date, scheduled trip, stop) matches an expected GTFS
# trip-visit. Below this, the feed likely does not correspond to the export
# (trip-ID drift between signups) and coverage numbers are not trustworthy.
MIN_VISIT_JOIN_RATE_PCT: float = 75.0

LOG_LEVEL: int = logging.INFO

# Filenames.
STOP_COVERAGE_FILENAME: str = r"stop_coverage.csv"
STOP_ROUTE_DAY_DETAIL_FILENAME: str = r"stop_route_day_detail.csv"

# When True, a failed run-log write aborts the script so an output is never
# left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class Config(NamedTuple):
    """Runtime configuration for a stop-coverage run."""

    gtfs_path: Path
    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    start_date: str = START_DATE
    end_date: str = END_DATE
    exclude_exception_dates: bool = EXCLUDE_EXCEPTION_DATES
    exclude_dates: Sequence[str] = EXCLUDE_DATES
    routes_to_include: Sequence[str] = ROUTES_TO_INCLUDE
    routes_to_exclude: Sequence[str] = ROUTES_TO_EXCLUDE
    timepoints_only: bool = TIMEPOINTS_ONLY
    low_coverage_flag_pct: float = LOW_COVERAGE_FLAG_PCT
    min_expected_visits: int = MIN_EXPECTED_VISITS
    min_visit_join_rate_pct: float = MIN_VISIT_JOIN_RATE_PCT


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
# LOADING
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


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read a TIDES ``stop_visits`` CSV and prepare it for matching.

    Args:
        path: Path to the ``stop_visits`` CSV export.

    Returns:
        DataFrame with string columns, ``service_date`` parsed and normalized,
        actual timestamp columns parsed, and rows lacking a service date, a
        ``trip_id_performed``, or a ``stop_id`` dropped (nothing to attribute
        them to).

    Raises:
        ValueError: If a required column is missing.
    """
    df = pd.read_csv(path, dtype=str)
    required = ["service_date", "trip_id_performed", "stop_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"stop_visits at '{path}' is missing required column(s): {', '.join(missing)}"
        )
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.normalize()
    for col in ("actual_arrival_time", "actual_departure_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT
    unusable = (
        df["service_date"].isna()
        | df["trip_id_performed"].isna()
        | df["stop_id"].isna()
        | (df["stop_id"].str.strip() == "")
    )
    n_unusable = int(unusable.sum())
    if n_unusable:
        logging.warning(
            "Dropping %d stop_visits row(s) with no service_date, trip_id_performed, or stop_id.",
            n_unusable,
        )
        df = df.loc[~unusable]
    return df.copy()


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read a TIDES ``trips_performed`` CSV and prepare it for matching.

    Args:
        path: Path to the ``trips_performed`` CSV export.

    Returns:
        DataFrame with string columns, ``service_date`` parsed and normalized,
        and rows lacking a service date or a ``trip_id_scheduled`` dropped
        (they cannot be matched to the schedule).

    Raises:
        ValueError: If a required column is missing.
    """
    df = pd.read_csv(path, dtype=str)
    required = ["service_date", "trip_id_scheduled"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"trips_performed at '{path}' is missing required column(s): {', '.join(missing)}"
        )
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.normalize()
    blank_id = df["trip_id_scheduled"].isna() | (df["trip_id_scheduled"].str.strip() == "")
    n_unmatchable = int((df["service_date"].isna() | blank_id).sum())
    if n_unmatchable:
        logging.warning(
            "Dropping %d trips_performed row(s) with no service_date or no "
            "trip_id_scheduled (typically Added/unscheduled trips).",
            n_unmatchable,
        )
        df = df.loc[df["service_date"].notna() & ~blank_id]
    return df.copy()


def filter_in_service(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Keep scheduled, in-revenue-service performed trips, one row per trip.

    Canceled trips never served any stop and non-revenue trips carry no
    passengers, so neither belongs in the visit bridge. This is the same
    filter the other TIDES consumers in this repo apply before their joins.

    Args:
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Filtered copy, deduplicated on ``trip_id_performed``.
    """
    out = trips_performed
    if "schedule_relationship" in out.columns:
        out = out.loc[out["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in out.columns:
        out = out.loc[out["trip_type"].fillna("In service") == "In service"]
    return out.drop_duplicates("trip_id_performed").copy()


def summarize_performed_records(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``trips_performed`` to one row per (service_date, scheduled trip).

    ``recorded`` means the operational export contains any row for the trip
    that day -- the AVL/CAD system knew about it. ``performed`` narrows that
    to rows representing revenue service: ``schedule_relationship`` not
    ``Canceled`` and ``trip_type`` in service (each filter is skipped when its
    column is absent, matching the other TIDES consumers in this repo).

    Args:
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        One row per (``service_date``, ``trip_id_scheduled``) with boolean
        ``recorded`` / ``performed`` and an ``n_records`` count.
    """
    df = trips_performed.copy()
    ok = pd.Series(True, index=df.index)
    if "schedule_relationship" in df.columns:
        ok &= df["schedule_relationship"].fillna("Scheduled") != "Canceled"
    if "trip_type" in df.columns:
        ok &= df["trip_type"].fillna("In service") == "In service"
    df["_performed"] = ok
    grouped = df.groupby(["service_date", "trip_id_scheduled"], as_index=False).agg(
        performed=("_performed", "any"),
        n_records=("_performed", "size"),
    )
    grouped["recorded"] = True
    return grouped


# =============================================================================
# ANALYSIS WINDOW
# =============================================================================


def feed_validity_span(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (first, last) service date the feed declares any service for.

    Args:
        calendar: Parsed ``calendar.txt`` or ``None``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.

    Returns:
        Tuple of normalized timestamps spanning both files' declared dates.

    Raises:
        ValueError: If neither file yields a parseable date.
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
    return min(candidates).normalize(), max(candidates).normalize()


def resolve_window(
    observed_dates: pd.Series,
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    start_override: str = "",
    end_override: str = "",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve the analysis window and log every span that shaped it.

    The automatic window is the intersection of the feed validity span and the
    observed data span, so the schedule is never compared against dates one
    side knows nothing about. Explicit overrides win over the intersection for
    their bound.

    Args:
        observed_dates: Parsed ``service_date`` values from the observed data.
        calendar: Parsed ``calendar.txt`` or ``None``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.
        start_override: ``"YYYY-MM-DD"`` or empty for automatic.
        end_override: ``"YYYY-MM-DD"`` or empty for automatic.

    Returns:
        Tuple of normalized (window_start, window_end).

    Raises:
        ValueError: If the resolved window is empty -- most commonly a feed
            that does not overlap the observed data at all.
    """
    feed_start, feed_end = feed_validity_span(calendar, calendar_dates)
    obs_start = observed_dates.min()
    obs_end = observed_dates.max()
    logging.info(
        "Feed validity span: %s to %s. Observed data span: %s to %s.",
        feed_start.date(),
        feed_end.date(),
        obs_start.date(),
        obs_end.date(),
    )

    window_start = pd.Timestamp(start_override) if start_override else max(feed_start, obs_start)
    window_end = pd.Timestamp(end_override) if end_override else min(feed_end, obs_end)
    window_start = window_start.normalize()
    window_end = window_end.normalize()

    if window_start > window_end:
        raise ValueError(
            f"Analysis window is empty ({window_start.date()} to {window_end.date()}). "
            "The GTFS feed's validity span likely does not overlap the observed data "
            "-- use the feed that was in effect when the data was collected, or set "
            "START_DATE/END_DATE explicitly."
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


# =============================================================================
# EXPECTED & OBSERVED TRIP-VISITS
# =============================================================================


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


def build_expected_visits(
    scheduled: pd.DataFrame,
    stop_times: pd.DataFrame,
    timepoints_only: bool = TIMEPOINTS_ONLY,
) -> pd.DataFrame:
    """Expand scheduled trip instances into expected (date, trip, stop) visits.

    Args:
        scheduled: Output of :func:`build_scheduled_instances`.
        stop_times: Parsed GTFS ``stop_times.txt``.
        timepoints_only: When True, keep only ``timepoint == 1`` rows. If the
            feed has no ``timepoint`` column, every time is exact per the GTFS
            spec, so no filtering is needed and a note is logged.

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
    st = stop_times[
        [col for col in ("trip_id", "stop_id", "timepoint") if col in stop_times.columns]
    ]
    if timepoints_only:
        if "timepoint" in st.columns:
            st = st.loc[st["timepoint"].astype(str).str.strip() == "1"]
            if st.empty:
                raise ValueError(
                    "TIMEPOINTS_ONLY is on but no stop_times rows have timepoint == 1."
                )
        else:
            logging.info(
                "TIMEPOINTS_ONLY is on but stop_times has no timepoint column; per the "
                "GTFS spec all times are then exact, so every stop is kept."
            )
    expected = scheduled.merge(st[["trip_id", "stop_id"]], on="trip_id", how="inner")
    return expected.drop_duplicates(["service_date", "trip_id", "stop_id"], ignore_index=True)


def bridge_visits_to_schedule(
    stop_visits: pd.DataFrame,
    trips_performed: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``trip_id_scheduled`` and route attribution to each stop visit.

    The join key is ``trip_id_performed``, unique per performed trip in TIDES;
    only in-service, non-canceled trips participate (see
    :func:`filter_in_service`).

    Args:
        stop_visits: Output of :func:`load_stop_visits`.
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Stop visits carrying ``trip_id_scheduled`` (inner join -- visits from
        unknown, canceled, or non-revenue trips are dropped with a warning).
    """
    trips = filter_in_service(trips_performed)
    bridge = trips[["trip_id_performed", "trip_id_scheduled"]]
    out = stop_visits.merge(bridge, on="trip_id_performed", how="inner")
    n_dropped = len(stop_visits) - len(out)
    if n_dropped:
        logging.warning(
            "%d stop_visits row(s) (%.1f%%) have no in-service trips_performed match on "
            "trip_id_performed and were dropped from the observed side.",
            n_dropped,
            100.0 * n_dropped / len(stop_visits) if len(stop_visits) else 0.0,
        )
    return out


def summarize_visit_records(bridged_visits: pd.DataFrame) -> pd.DataFrame:
    """Collapse bridged visits to one row per (date, scheduled trip, stop).

    Args:
        bridged_visits: Output of :func:`bridge_visits_to_schedule`.

    Returns:
        One row per (``service_date``, ``trip_id_scheduled``, ``stop_id``)
        with ``emitted`` (a row exists), ``observed`` (any row carries an
        actual arrival or departure timestamp), and ``skipped`` (any row is
        marked Skipped) flags.
    """
    df = bridged_visits.copy()
    df["_has_actual"] = df["actual_arrival_time"].notna() | df["actual_departure_time"].notna()
    if "schedule_relationship" in df.columns:
        df["_skipped"] = df["schedule_relationship"].fillna("Scheduled") == "Skipped"
    else:
        df["_skipped"] = False
    grouped = df.groupby(["service_date", "trip_id_scheduled", "stop_id"], as_index=False).agg(
        observed=("_has_actual", "any"),
        skipped=("_skipped", "any"),
    )
    grouped["emitted"] = True
    return grouped


def warn_if_timepoint_only_export(stop_visits: pd.DataFrame, timepoints_only: bool) -> None:
    """Warn when the export looks timepoint-only but TIMEPOINTS_ONLY is off.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.
        timepoints_only: The configured ``TIMEPOINTS_ONLY`` value.
    """
    if timepoints_only or "timepoint" not in stop_visits.columns or stop_visits.empty:
        return
    is_timepoint = stop_visits["timepoint"].astype(str).str.upper() == "TRUE"
    if bool(is_timepoint.all()):
        logging.warning(
            "Every stop_visits row is a timepoint. If this export only emits timepoint "
            "visits, every non-timepoint stop will be misread as invisible -- set "
            "TIMEPOINTS_ONLY = True to restrict the GTFS expectation to timepoints too."
        )


def classify_visit_outcomes(
    expected: pd.DataFrame,
    performed_records: pd.DataFrame,
    visit_records: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each expected trip-visit exactly one outcome.

    Outcomes, in priority order: ``observed``, ``skipped``,
    ``visit_without_actual``, ``visit_missing`` (trip performed, no row at
    this stop), ``trip_unrecorded`` (no usable trip record that day).

    Args:
        expected: Output of :func:`build_expected_visits`.
        performed_records: Output of :func:`summarize_performed_records`,
            limited to the analysis date pool.
        visit_records: Output of :func:`summarize_visit_records`.

    Returns:
        ``expected`` with boolean helper columns and a string ``outcome``.
    """
    out = expected.merge(
        performed_records[["service_date", "trip_id_scheduled", "performed"]],
        left_on=["service_date", "trip_id"],
        right_on=["service_date", "trip_id_scheduled"],
        how="left",
    ).drop(columns=["trip_id_scheduled"])
    out = out.merge(
        visit_records,
        left_on=["service_date", "trip_id", "stop_id"],
        right_on=["service_date", "trip_id_scheduled", "stop_id"],
        how="left",
    ).drop(columns=["trip_id_scheduled"])
    for col in ("performed", "emitted", "observed", "skipped"):
        out[col] = out[col].eq(True)

    out["outcome"] = "trip_unrecorded"
    out.loc[out["performed"], "outcome"] = "visit_missing"
    out.loc[out["emitted"], "outcome"] = "visit_without_actual"
    out.loc[out["emitted"] & out["skipped"], "outcome"] = "skipped"
    out.loc[out["observed"], "outcome"] = "observed"
    return out


def report_visit_join_health(
    expected: pd.DataFrame,
    visit_records: pd.DataFrame,
    min_join_rate_pct: float = MIN_VISIT_JOIN_RATE_PCT,
) -> dict:
    """Compute and log the share of visit records matching an expected visit.

    Args:
        expected: Output of :func:`build_expected_visits`.
        visit_records: Output of :func:`summarize_visit_records`, limited to
            the analysis date pool.
        min_join_rate_pct: Threshold below which a loud warning is emitted.

    Returns:
        Dict with ``n_visit_records``, ``n_matched``, and ``join_rate_pct``
        for the run log.
    """
    expected_keys = pd.MultiIndex.from_frame(expected[["service_date", "trip_id", "stop_id"]])
    visit_keys = pd.MultiIndex.from_frame(
        visit_records[["service_date", "trip_id_scheduled", "stop_id"]]
    )
    n_records = len(visit_records)
    n_matched = int(visit_keys.isin(expected_keys).sum())
    join_rate = 100.0 * n_matched / n_records if n_records else float("nan")
    if n_records and join_rate < min_join_rate_pct:
        logging.error(
            "Only %.1f%% of observed visit records (%d of %d) match an expected GTFS "
            "trip-visit (threshold %.0f%%). The feed likely does not correspond to this "
            "export's signup (trip-ID drift) -- coverage results below are NOT "
            "trustworthy.",
            join_rate,
            n_matched,
            n_records,
            min_join_rate_pct,
        )
    else:
        logging.info(
            "Visit join health: %.1f%% of observed visit records (%d of %d) match an "
            "expected GTFS trip-visit.",
            join_rate,
            n_matched,
            n_records,
        )
    return {"n_visit_records": n_records, "n_matched": n_matched, "join_rate_pct": join_rate}


# =============================================================================
# SUMMARIES
# =============================================================================


def _aggregate_outcomes(outcomes: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate outcome rows to *group_cols* with per-outcome counts."""
    df = outcomes.copy()
    for name in ("observed", "skipped", "visit_without_actual", "visit_missing", "trip_unrecorded"):
        df[f"_is_{name}"] = df["outcome"] == name
    grouped = df.groupby(group_cols, as_index=False).agg(
        expected_visits=("outcome", "size"),
        observed_visits=("_is_observed", "sum"),
        n_skipped=("_is_skipped", "sum"),
        n_visit_without_actual=("_is_visit_without_actual", "sum"),
        n_visit_missing=("_is_visit_missing", "sum"),
        n_trip_unrecorded=("_is_trip_unrecorded", "sum"),
    )
    grouped["pct_observed"] = (
        100.0 * grouped["observed_visits"] / grouped["expected_visits"]
    ).round(1)
    return grouped


def _apply_flags(
    grouped: pd.DataFrame,
    low_coverage_flag_pct: float,
    min_expected_visits: int,
) -> pd.DataFrame:
    """Add invisible / low-coverage flags and a combined reason column."""
    judged = grouped["expected_visits"] >= min_expected_visits
    grouped["flag_invisible"] = judged & (grouped["observed_visits"] == 0)
    grouped["flag_low_coverage"] = judged & (grouped["pct_observed"] < low_coverage_flag_pct)
    grouped["flag_reason"] = ""
    grouped.loc[grouped["flag_low_coverage"], "flag_reason"] = "low coverage"
    grouped.loc[grouped["flag_invisible"], "flag_reason"] = "invisible (never observed)"
    return grouped.sort_values(
        ["flag_invisible", "flag_low_coverage", "pct_observed", "expected_visits"],
        ascending=[False, False, True, False],
        ignore_index=True,
    )


def summarize_stop_coverage(
    outcomes: pd.DataFrame,
    low_coverage_flag_pct: float = LOW_COVERAGE_FLAG_PCT,
    min_expected_visits: int = MIN_EXPECTED_VISITS,
) -> pd.DataFrame:
    """Roll outcome rows up to one row per (stop, day type), with flags.

    Args:
        outcomes: Output of :func:`classify_visit_outcomes`.
        low_coverage_flag_pct: Flag threshold on percent observed.
        min_expected_visits: Minimum expected visits before a row is judged.

    Returns:
        Per-(stop, day type) coverage with flags, flagged rows first.
    """
    routes = outcomes.groupby(["stop_id", "day_type"], as_index=False).agg(
        n_routes=("route_id", "nunique")
    )
    grouped = _aggregate_outcomes(outcomes, ["stop_id", "day_type"])
    grouped = grouped.merge(routes, on=["stop_id", "day_type"], how="left")
    return _apply_flags(grouped, low_coverage_flag_pct, min_expected_visits)


def summarize_stop_route_detail(
    outcomes: pd.DataFrame,
    low_coverage_flag_pct: float = LOW_COVERAGE_FLAG_PCT,
    min_expected_visits: int = MIN_EXPECTED_VISITS,
) -> pd.DataFrame:
    """Roll outcome rows up to one row per (stop, route, day type), with flags.

    Args:
        outcomes: Output of :func:`classify_visit_outcomes`.
        low_coverage_flag_pct: Flag threshold on percent observed.
        min_expected_visits: Minimum expected visits before a row is judged.

    Returns:
        Per-(stop, route, day type) coverage with flags, flagged rows first.
    """
    grouped = _aggregate_outcomes(outcomes, ["stop_id", "route_id", "day_type"])
    return _apply_flags(grouped, low_coverage_flag_pct, min_expected_visits)


# =============================================================================
# EXPORT & RUN LOG
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def export_tables(
    stop_summary: pd.DataFrame,
    detail: pd.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """Write the per-stop summary and the per-(stop, route) detail tables.

    Args:
        stop_summary: Output of :func:`summarize_stop_coverage`.
        detail: Output of :func:`summarize_stop_route_detail`.
        out_dir: Directory to write into (created if needed).

    Returns:
        Paths of the files written.
    """
    ensure_dir(out_dir)
    summary_path = out_dir / STOP_COVERAGE_FILENAME
    stop_summary.to_csv(summary_path, index=False)
    detail_path = out_dir / STOP_ROUTE_DAY_DETAIL_FILENAME
    detail.to_csv(detail_path, index=False)
    return [summary_path, detail_path]


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
    log_path = output_dir / "stop_coverage_tides_runlog.txt"

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
        "STOP COVERAGE (GTFS VS TIDES) RUN LOG",
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


def run(cfg: Config) -> pd.DataFrame:
    """Execute the full stop-coverage pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        The per-(stop, day type) coverage table (also written to disk).
    """
    gtfs_path = str(cfg.gtfs_path)
    gtfs = load_gtfs_data(gtfs_path, files=("trips.txt", "stop_times.txt"))
    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]
    calendar = load_optional_gtfs_table(gtfs_path, "calendar.txt")
    calendar_dates = load_optional_gtfs_table(gtfs_path, "calendar_dates.txt")

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        trips = trips.loc[trips["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        trips = trips.loc[~trips["route_id"].astype(str).isin(drop)]
    if trips.empty:
        raise ValueError("No GTFS trips remain after ROUTES_TO_INCLUDE/EXCLUDE filtering.")

    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips_performed = load_trips_performed(cfg.trips_performed_path)
    warn_if_timepoint_only_export(stop_visits, cfg.timepoints_only)
    if cfg.timepoints_only and "timepoint" in stop_visits.columns:
        stop_visits = stop_visits.loc[stop_visits["timepoint"].astype(str).str.upper() == "TRUE"]

    window_start, window_end = resolve_window(
        stop_visits["service_date"], calendar, calendar_dates, cfg.start_date, cfg.end_date
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
    expected = build_expected_visits(scheduled, stop_times, cfg.timepoints_only)
    if expected.empty:
        raise ValueError("No expected trip-visits remain in the analysis window after exclusions.")

    def in_pool(dates: pd.Series) -> pd.Series:
        pool = dates.between(window_start, window_end)
        if excluded:
            pool &= ~dates.isin(list(excluded))
        return pool

    performed_records = summarize_performed_records(
        trips_performed.loc[in_pool(trips_performed["service_date"])]
    )
    bridged = bridge_visits_to_schedule(
        stop_visits.loc[in_pool(stop_visits["service_date"])], trips_performed
    )
    visit_records = summarize_visit_records(bridged)

    join_stats = report_visit_join_health(expected, visit_records, cfg.min_visit_join_rate_pct)
    outcomes = classify_visit_outcomes(expected, performed_records, visit_records)
    stop_summary = summarize_stop_coverage(
        outcomes, cfg.low_coverage_flag_pct, cfg.min_expected_visits
    )
    detail = summarize_stop_route_detail(
        outcomes, cfg.low_coverage_flag_pct, cfg.min_expected_visits
    )

    flagged = stop_summary.loc[stop_summary["flag_low_coverage"]]
    invisible = stop_summary.loc[stop_summary["flag_invisible"]]
    if flagged.empty:
        logging.info(
            "No stops flagged: every judged (stop, day type) is at or above %.0f%% of "
            "expected trip-visits observed.",
            cfg.low_coverage_flag_pct,
        )
    else:
        worst = flagged.iloc[0]
        logging.warning(
            "%d (stop, day type) row(s) flagged below %.0f%% coverage, of which %d are "
            "invisible (never observed). Worst: stop %s on %s (%.1f%% of %d expected). "
            "See %s for per-route evidence.",
            len(flagged),
            cfg.low_coverage_flag_pct,
            len(invisible),
            worst["stop_id"],
            worst["day_type"],
            worst["pct_observed"],
            int(worst["expected_visits"]),
            STOP_ROUTE_DAY_DETAIL_FILENAME,
        )

    paths = export_tables(stop_summary, detail, cfg.output_dir)
    for path in paths:
        logging.info("Wrote table: %s", path)

    pct_overall = (
        100.0 * (outcomes["outcome"] == "observed").sum() / len(outcomes)
        if len(outcomes)
        else float("nan")
    )
    summary_lines = [
        f"Analysis window:          {window_start.date()} to {window_end.date()}",
        f"Dates excluded:           {len(excluded)}",
        f"Expected trip-visits:     {len(outcomes)}",
        f"Observed trip-visits:     {int((outcomes['outcome'] == 'observed').sum())} "
        f"({pct_overall:.1f}%)",
        f"Stops (x day type):       {len(stop_summary)}",
        f"Flagged low coverage:     {len(flagged)}",
        f"  of which invisible:     {len(invisible)}",
        f"Visit records in pool:    {join_stats['n_visit_records']}",
        f"Visit join rate:          {join_stats['join_rate_pct']:.1f}%",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return stop_summary


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Vet TIDES stop_visits coverage against GTFS expected trip-visits."
    )
    parser.add_argument("--gtfs", default=GTFS_PATH, help="Path to GTFS folder or .zip.")
    parser.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    parser.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
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

    if (
        args.gtfs == GTFS_PATH
        or args.stop_visits == STOP_VISITS_PATH
        or args.trips_performed == TRIPS_PERFORMED_PATH
    ):
        logging.warning(
            "GTFS_PATH/STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. "
            "Update the CONFIGURATION section or pass --gtfs/--stop-visits/"
            "--trips-performed before running."
        )
        return

    cfg = Config(
        gtfs_path=Path(args.gtfs).expanduser(),
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        start_date=START_DATE,
        end_date=END_DATE,
        exclude_exception_dates=EXCLUDE_EXCEPTION_DATES,
        exclude_dates=EXCLUDE_DATES,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
        timepoints_only=TIMEPOINTS_ONLY,
        low_coverage_flag_pct=LOW_COVERAGE_FLAG_PCT,
        min_expected_visits=MIN_EXPECTED_VISITS,
        min_visit_join_rate_pct=MIN_VISIT_JOIN_RATE_PCT,
    )

    if not cfg.gtfs_path.exists():
        logging.warning("GTFS feed not found: %s", cfg.gtfs_path)
        return
    if not cfg.stop_visits_path.exists():
        logging.warning("stop_visits not found: %s", cfg.stop_visits_path)
        return
    if not cfg.trips_performed_path.exists():
        logging.warning("trips_performed not found: %s", cfg.trips_performed_path)
        return

    run(cfg)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
