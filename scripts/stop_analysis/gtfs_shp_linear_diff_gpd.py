"""Compare a GTFS feed's route alignments with an agency's route shapefile(s).

A sibling of ``gtfs_linear_diff_gpd.py`` that checks one GTFS feed against a
reference route layer instead of a second feed: the recurring check that
``shapes.txt`` and the GIS team's route lines still agree. Route shapefiles
usually hold one undirected line per route, so for each route the script picks
one GTFS pattern per direction and compares the directions together with it.

Method
------
- Pattern choice: trips are counted on one service day (default: a typical
  weekday picked from the calendar, skipping federal holidays; ``SERVICE_DATE``
  overrides). Per route and ``direction_id`` (``trip_headsign`` when it is
  blank), ``PATTERN_RULE`` keeps the shape run by the most trips (ties go to
  the longer shape), the longest shape, or every shape. A route with no trips
  that day counts all of its trips.
- Comparison: the buffered overlay of ``gtfs_linear_diff_gpd.py``, on each
  route's chosen GTFS shapes and reference features as drawn. GTFS line beyond
  ``BUFFER_TOLERANCE_FEET`` of every reference feature is ``gtfs_only``, the
  rest ``matched``; reference line beyond it of every shape is ``shp_only``.
- Lengths: each class is measured across the route's shapes, busiest first,
  so a street drawn both ways counts once (one-way couplets keep both streets;
  a shape straying past the tolerance for under ``MIN_SEGMENT_LENGTH_FEET``
  stays the same street). Difference pieces under that length count as
  ``short``, not as differences, and each side's total is the sum of its
  classes. A route is a mismatch when ``gtfs_only`` plus ``shp_only`` reaches
  ``MISMATCH_MIN_FEET``.
- Routes match on ``GTFS_ROUTE_FIELD`` against ``SHP_ROUTE_FIELD``, or against
  each file's name when that is blank, ignoring case and leading zeros;
  ``SHP_ROUTE_ALIASES`` renames shapefile routes that still differ.

Inputs
------
- A GTFS feed (folder or ``.zip``) with ``routes.txt``, ``trips.txt`` and
  ``shapes.txt``, plus ``calendar.txt`` and/or ``calendar_dates.txt``.
- Reference route lines: one ``.shp`` holding every route, a folder of ``.shp``
  files (one per route), or a ``.zip`` of either. Each needs a ``.prj``.

Outputs
-------
- ``gtfs_shp_segments_gtfs_only.shp`` / ``_shp_only.shp`` / ``_matched.shp``.
- ``gtfs_shp_combined_routes.shp``: each route's GTFS directions merged into one line.
- ``gtfs_shp_selected_patterns.shp`` / ``.csv``: the shape picked per direction.
- ``gtfs_shp_route_comparison.csv``: one row per route with its status
  (match, mismatch, missing from either side) and lengths.
- ``gtfs_shp_summary.json`` and a run-log sidecar with the CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--gtfs``,
``--route-shapes`` and ``--output-dir``) and run from a shell or a Jupyter
notebook. Shapefiles are written in WGS 84 (EPSG:4326).

See Also:
    - ``gtfs_shp_linear_diff_arcpy.py``
    - ``gtfs_linear_diff_gpd.py``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import zipfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, List, Literal, NamedTuple, Optional, Sequence

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

# Sentinel markers used by extract_config_block / write_run_log to identify the
# configuration block within this file's source. Each string must appear exactly
# once in this file as a stand-alone comment line. Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS feed to check: a folder or a .zip (with routes.txt, trips.txt,
# shapes.txt, and calendar.txt and/or calendar_dates.txt).
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"

# Reference route lines: one .shp holding every route, a folder of .shp files
# (one per route), or a .zip of either.
ROUTE_SHAPES_PATH: str = r"Path\To\Your\Route_Shapefiles"

OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Field in the route shapefile(s) that names each route. Leave blank ("") when
# each .shp holds one route and its file name says which (e.g. Route_10.shp).
SHP_ROUTE_FIELD: str = ""

# Optional regular expression whose first capture group pulls the route name
# out of the field value or file name, e.g. r"(?i)route[_ -]*([A-Za-z0-9]+)"
# turns "Route_10" into "10". Blank uses the whole value.
SHP_ROUTE_PATTERN: str = r""

# Shapefile route names to read as another route name, for names the rules
# above do not reconcile, e.g. {"Blue 1": "B1", "7 AM": "7", "7 PM": "7"}.
# Keys are compared after SHP_ROUTE_PATTERN, ignoring case. Features that end
# up with the same route name are compared as one route.
SHP_ROUTE_ALIASES: Mapping[str, str] = {}

# routes.txt column matched against the shapefile's route names.
GTFS_ROUTE_FIELD: Literal["route_short_name", "route_id"] = "route_short_name"

# Ignore leading zeros when matching route names, so "010" matches "10".
# Matching always ignores case.
IGNORE_LEADING_ZEROS: bool = True

# Service day whose trips are counted to pick each direction's pattern:
# "weekday" or a day name ("saturday", "sunday", ...). A typical date of that
# kind is picked from the calendar, skipping federal holidays. SERVICE_DATE
# ("YYYY-MM-DD") overrides the pick.
SERVICE_DAY: str = "weekday"
SERVICE_DATE: str = ""

# How each route direction's GTFS pattern is chosen:
#   "most_trips" - the shape run by the most trips that day (ties: longer shape)
#   "longest"    - the longest shape run that day, for reference lines drawn
#                  to the route's full extent
#   "all"        - every shape run that day, branches and short turns included
PATTERN_RULE: Literal["most_trips", "longest", "all"] = "most_trips"

# Lines within this distance of each other count as the same street. Also used
# to count a street drawn in both directions once. ~80 ft (about 25 m) absorbs digitizing
# differences without swallowing a move to a parallel street.
BUFFER_TOLERANCE_FEET: float = 80.0

# Difference pieces shorter than this (crossing artifacts, terminal stubs) are
# not reported as gtfs_only / shp_only; their length goes in *_short_len_ft.
MIN_SEGMENT_LENGTH_FEET: float = 150.0

# A route present on both sides is flagged as a mismatch when its gtfs_only +
# shp_only length reaches this.
MISMATCH_MIN_FEET: float = 500.0

# Projected CRS for buffering/length math, in feet or meters, e.g. "EPSG:2248"
# (Maryland State Plane, US feet, covering DC). Empty = auto-pick a UTM zone
# from the feed's extent.
# Output shapefiles are always written in WGS 84 (EPSG:4326).
TARGET_CRS: str = ""

# Unit of the converted length columns (named *_mi or *_km to match) and of
# the summary lengths: "miles" or "km".
DISTANCE_OUTPUT_UNIT: Literal["miles", "km"] = "miles"

# Output filenames (all written inside OUTPUT_DIR).
SEGMENTS_GTFS_ONLY_FILENAME: str = r"gtfs_shp_segments_gtfs_only.shp"
SEGMENTS_SHP_ONLY_FILENAME: str = r"gtfs_shp_segments_shp_only.shp"
SEGMENTS_MATCHED_FILENAME: str = r"gtfs_shp_segments_matched.shp"
COMBINED_ROUTES_FILENAME: str = r"gtfs_shp_combined_routes.shp"
PATTERNS_SHP_FILENAME: str = r"gtfs_shp_selected_patterns.shp"
PATTERNS_CSV_FILENAME: str = r"gtfs_shp_selected_patterns.csv"
ROUTE_TABLE_FILENAME: str = r"gtfs_shp_route_comparison.csv"
SUMMARY_FILENAME: str = r"gtfs_shp_summary.json"
RUN_LOG_FILENAME: str = r"gtfs_shp_linear_diff_runlog.txt"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

FEET_PER_METER: float = 3.280839895013123
DistanceUnit = Literal["miles", "km"]
PLACEHOLDER_PREFIX: str = "Path\\To\\Your"

_REQUIRED_FILES: tuple[str, ...] = ("routes.txt", "trips.txt", "shapes.txt")
_CALENDAR_FILES: tuple[str, ...] = ("calendar.txt", "calendar_dates.txt")
_PATTERN_RULES: tuple[str, ...] = ("most_trips", "longest", "all")
_GTFS_ROUTE_FIELDS: tuple[str, ...] = ("route_short_name", "route_id")
_SHP_TEXT_MAX: int = 254
_SHAPEFILE_PARTS: tuple[str, ...] = (
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qix",
    ".sbn",
    ".sbx",
    ".shp.xml",
)

# Route status values, in the order the route table lists them.
STATUS_ORDER: tuple[str, ...] = (
    "mismatch",
    "missing_from_shp",
    "missing_from_gtfs",
    "no_gtfs_geometry",
    "match",
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ConfigError(ValueError):
    """A CONFIGURATION value is unusable; reported without a traceback (exit code 2)."""


class Config(NamedTuple):
    """Runtime configuration for one GTFS-vs-shapefile comparison."""

    gtfs_path: Path
    route_shapes_path: Path
    output_dir: Path
    shp_route_field: str = SHP_ROUTE_FIELD
    shp_route_pattern: str = SHP_ROUTE_PATTERN
    shp_route_aliases: Mapping[str, str] = SHP_ROUTE_ALIASES
    gtfs_route_field: str = GTFS_ROUTE_FIELD
    ignore_leading_zeros: bool = IGNORE_LEADING_ZEROS
    service_day: str = SERVICE_DAY
    service_date: str = SERVICE_DATE
    pattern_rule: str = PATTERN_RULE
    buffer_tolerance_feet: float = BUFFER_TOLERANCE_FEET
    min_segment_length_feet: float = MIN_SEGMENT_LENGTH_FEET
    mismatch_min_feet: float = MISMATCH_MIN_FEET
    target_crs: str = TARGET_CRS
    distance_output_unit: DistanceUnit = DISTANCE_OUTPUT_UNIT


class PatternChoice(NamedTuple):
    """One GTFS shape picked to represent a route direction."""

    route_id: str
    route_key: str
    route_label: str
    direction: str  # direction_id, "headsign:<text>", or "" when neither is set
    shape_id: str
    trips: int  # trips on this shape in this direction on the counted day(s)
    direction_trips: int  # all trips in this direction on the counted day(s)
    shapes_in_direction: int  # distinct usable shapes run in this direction
    length_ft: float
    day_basis: str  # "service_day" or "all_trips"


class GtfsRoute(NamedTuple):
    """The GTFS side of one matched route key (one or more GTFS routes)."""

    route_key: str
    route_label: str
    route_ids: tuple[str, ...]
    patterns: tuple[PatternChoice, ...]
    lines: tuple[BaseGeometry, ...]  # chosen shapes, busiest first, projected CRS
    note: str


class ReferenceRoute(NamedTuple):
    """The reference side of one route key (every feature carrying that name)."""

    route_key: str
    names: tuple[str, ...]  # route names as written in the shapefile(s)
    sources: tuple[str, ...]  # shapefile names the features came from
    feature_count: int
    lines: tuple[BaseGeometry, ...]  # features chained end to end, longest first


class SegmentRecord(NamedTuple):
    """One exportable comparison segment."""

    route_key: str
    route_label: str
    route_ids: str
    segment_class: str  # gtfs_only / shp_only / matched
    status: str
    geometry: BaseGeometry


class RouteComparison(NamedTuple):
    """Per-route comparison outcome (one row of the route table)."""

    route_key: str
    route_label: str
    gtfs_route_ids: str
    shp_route_names: str
    shp_sources: str
    status: str
    pattern_count: int
    shape_ids: str
    gtfs_len_ft: float  # = matched + gtfs_only + gtfs_short
    matched_len_ft: float
    gtfs_only_len_ft: float
    gtfs_short_len_ft: float
    shp_len_ft: float  # = shp_matched + shp_only + shp_short
    shp_matched_len_ft: float
    shp_only_len_ft: float
    shp_short_len_ft: float
    mismatch_len_ft: float
    mismatch_share: float
    note: str


class Summary(NamedTuple):
    """Run-level metrics for the comparison."""

    gtfs_route_count: int
    shp_route_count: int
    compared_count: int
    match_count: int
    mismatch_count: int
    missing_from_shp_count: int
    missing_from_gtfs_count: int
    no_gtfs_geometry_count: int
    service_date: str
    pattern_rule: str
    length_unit: str
    matched_length: float
    gtfs_only_length: float
    shp_only_length: float
    buffer_tolerance_feet: float
    min_segment_length_feet: float
    mismatch_min_feet: float
    projected_crs: str


# =============================================================================
# REUSABLE HELPERS (verbatim copies; canonical versions live in utils/)
# =============================================================================


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


def load_shapefile(path: str, member: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load a shapefile given a direct ``.shp`` path or a ``.zip`` archive.

    Args:
        path: Path to a ``.shp`` file, or to a ``.zip`` archive containing
            one. A zipped shapefile is opened directly via GDAL's ``zip://``
            virtual filesystem, so the archive is never extracted to disk.
        member: For a ``.zip`` containing more than one ``.shp``, the name
            of the one to load (matched case-insensitively against its
            base filename, e.g. ``"Hospitals.shp"``). Ignored when *path*
            is a direct ``.shp`` file. Required when the archive contains
            more than one shapefile. macOS metadata entries (anything under
            ``__MACOSX/`` and ``._*`` files) are not counted as shapefiles.

    Returns:
        The loaded :class:`geopandas.GeoDataFrame`.

    Raises:
        OSError: *path* does not exist.
        ValueError: *path* is neither a ``.shp`` nor a ``.zip`` file; the
            zip is unreadable/corrupt; the zip contains no shapefile; the
            zip contains more than one shapefile and *member* was not
            supplied; or *member* does not match any shapefile in the zip.
    """
    if not os.path.exists(path):
        raise OSError(f"The path '{path}' does not exist.")

    lower_path = path.lower()
    if lower_path.endswith(".shp"):
        logging.info("Reading %s", path)
        return gpd.read_file(path)

    if not lower_path.endswith(".zip"):
        raise ValueError(f"'{path}' is neither a .shp nor a .zip file.")

    try:
        with zipfile.ZipFile(path) as archive:
            # Zips made with macOS Finder add __MACOSX/._<name>.shp metadata
            # files next to the real ones; they are not shapefiles.
            shp_members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".shp")
                and not name.startswith("__MACOSX/")
                and not os.path.basename(name).startswith("._")
            ]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc

    if not shp_members:
        raise ValueError(f"No .shp file found inside '{path}'.")

    if member is not None:
        matches = [name for name in shp_members if os.path.basename(name).lower() == member.lower()]
        if not matches:
            raise ValueError(f"'{member}' not found inside '{path}'. Available: {shp_members}")
        chosen = matches[0]
    elif len(shp_members) == 1:
        chosen = shp_members[0]
    else:
        raise ValueError(
            f"'{path}' contains {len(shp_members)} shapefiles; pass `member=` to "
            f"choose one of: {shp_members}"
        )

    source = f"zip://{path}!{chosen}"
    logging.info("Reading %s", source)
    return gpd.read_file(source)


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
# GTFS LOADING AND SERVICE DAY
# =============================================================================


def _read_csv(path: Path | IO[bytes], usecols: Optional[object] = None) -> pd.DataFrame:
    """Read a GTFS table as all-string columns with missing values as empty strings.

    Args:
        path: Path to the ``.txt`` file, or an open binary handle to it (as
            yielded by ``open_gtfs_file``).
        usecols: Optional value forwarded to ``pandas.read_csv(usecols=...)``; a
            callable is convenient for keeping only columns that exist.

    Returns:
        The parsed table; every column is ``str`` and absent values are ``""``.
    """
    return pd.read_csv(
        path,
        dtype=str,
        usecols=usecols,  # type: ignore[arg-type]
        encoding="utf-8-sig",
        low_memory=False,
        keep_default_na=False,
        na_filter=False,
    )


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize a text column for comparisons (fill NA, cast to str, strip)."""
    return series.fillna("").astype(str).str.strip()


def load_feed(gtfs_path: Path) -> dict[str, pd.DataFrame]:
    """Load the GTFS tables the comparison needs.

    Args:
        gtfs_path: GTFS feed folder, or a ``.zip`` of the feed.

    Returns:
        Mapping of file stem to table. ``routes``, ``trips`` and ``shapes`` are
        always present; ``calendar`` and ``calendar_dates`` only when the feed
        has them. Text columns are stripped, and ``trips`` always carries
        ``service_id``, ``direction_id``, ``trip_headsign`` and ``shape_id``.

    Raises:
        OSError: The feed is missing, or lacks a required file.
        ValueError: The feed is not a folder or valid ``.zip``, or a required
            column is absent.
    """
    if not os.path.exists(gtfs_path):
        raise OSError(f"GTFS feed '{gtfs_path}' does not exist.")

    def _has(name: str) -> bool:
        try:
            with open_gtfs_file(gtfs_path, name):
                return True
        except FileNotFoundError:
            return False

    missing = [name for name in _REQUIRED_FILES if not _has(name)]
    if missing:
        raise OSError(f"GTFS feed '{gtfs_path}' is missing {', '.join(missing)}.")

    trip_cols = {"route_id", "service_id", "trip_id", "direction_id", "trip_headsign", "shape_id"}
    shape_cols = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    feed: dict[str, pd.DataFrame] = {}
    for name in _REQUIRED_FILES + _CALENDAR_FILES:
        if name in _CALENDAR_FILES and not _has(name):
            continue
        with open_gtfs_file(gtfs_path, name) as handle:
            if name == "trips.txt":
                frame = _read_csv(handle, usecols=lambda c: c in trip_cols)
            elif name == "shapes.txt":
                frame = _read_csv(handle, usecols=lambda c: c in shape_cols)
            else:
                frame = _read_csv(handle)
        for col in frame.columns:
            frame[col] = normalize_text(frame[col])
        feed[name.replace(".txt", "")] = frame
        logging.info("Loaded %s (%d records).", name, len(frame))

    for table, cols in (("routes", ("route_id",)), ("trips", ("route_id", "trip_id"))):
        absent = [c for c in cols if c not in feed[table].columns]
        if absent:
            raise ValueError(f"{table}.txt is missing required column(s): {', '.join(absent)}")
    for col in ("service_id", "direction_id", "trip_headsign", "shape_id"):
        if col not in feed["trips"].columns:
            feed["trips"][col] = ""
    return feed


def select_service_ids(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    service_day: str,
    service_date: str,
) -> tuple[Optional[dt.date], Optional[set[str]]]:
    """Choose the service day whose trips are counted to pick patterns.

    Args:
        calendar: Parsed ``calendar.txt`` or ``None``.
        calendar_dates: Parsed ``calendar_dates.txt`` or ``None``.
        service_day: ``"weekday"`` or a single day name.
        service_date: Optional explicit ``YYYY-MM-DD`` date; overrides the pick.

    Returns:
        Tuple of (chosen date, its active service_ids). Both are ``None`` when
        the feed has no usable calendar, in which case every trip counts.

    Raises:
        ConfigError: *service_date* is not ``YYYY-MM-DD`` or *service_day* is
            not a recognized day.
    """
    if (calendar is None or calendar.empty) and (calendar_dates is None or calendar_dates.empty):
        logging.warning(
            "Feed has no calendar.txt or calendar_dates.txt; counting every trip to pick "
            "patterns, all service days together."
        )
        return None, None

    override: Optional[dt.date] = None
    if service_date.strip():
        try:
            override = dt.datetime.strptime(service_date.strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ConfigError(f"SERVICE_DATE must be YYYY-MM-DD; got {service_date!r}.") from exc

    active = expand_service_active_dates(calendar, calendar_dates)
    if not any(active.values()) and override is None:
        logging.warning("No service has any active dates; counting every trip.")
        return None, None

    holidays: set[dt.date] = set()
    all_dates = {d for dates in active.values() for d in dates}
    if all_dates:
        for year in range(min(all_dates).year, max(all_dates).year + 2):
            holidays |= federal_holidays_observed(year)
    try:
        chosen, service_ids = representative_service_date(
            active, service_day, override_date=override, exclude_dates=holidays
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return chosen, service_ids


# =============================================================================
# ROUTE NAMES
# =============================================================================


def compile_route_pattern(pattern: str) -> Optional[re.Pattern[str]]:
    """Compile ``SHP_ROUTE_PATTERN``; blank means no pattern.

    Raises:
        ConfigError: The pattern is not a valid regular expression.
    """
    if not pattern.strip():
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"SHP_ROUTE_PATTERN {pattern!r} is not a valid regex: {exc}") from exc


def route_name_text(value: Any) -> str:
    """Return a route name as text, writing whole-number floats without ``.0``.

    Shapefile number fields come back as floats, so route 10 stored in one reads
    as ``10.0``; GTFS holds it as ``"10"``.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def apply_route_pattern(text: str, pattern: Optional[re.Pattern[str]]) -> tuple[str, bool]:
    """Pull the route name out of *text* with *pattern*.

    Returns:
        Tuple of (name, matched). Without a pattern, or when it finds nothing,
        the text is returned unchanged; *matched* is False only in the latter case.
    """
    if pattern is None or not text:
        return text, True
    found = pattern.search(text)
    if found is None:
        return text, False
    name = found.group(1) if found.groups() else found.group(0)
    return (name or "").strip(), True


def normalize_route_key(text: str, ignore_leading_zeros: bool) -> str:
    """Return the key two route names are matched on: upper case, zeros optional.

    Leading zeros are dropped only before a digit, so ``"007"`` becomes ``"7"``
    and ``"0"`` stays ``"0"``, while ``"0A"`` is left alone.
    """
    key = text.strip().upper()
    if ignore_leading_zeros:
        key = re.sub(r"^0+(?=\d)", "", key)
    return key


def route_display_label(row: Mapping[Any, Any]) -> str:
    """Best human-readable label for a route: short name, else long name, else ID."""
    for col in ("route_short_name", "route_long_name", "route_id"):
        value = str(row.get(col) or "").strip()
        if value and value.lower() != "nan":
            return value
    return ""


def gtfs_route_keys(
    routes: pd.DataFrame, field: str, ignore_leading_zeros: bool
) -> dict[str, tuple[str, str]]:
    """Map each GTFS route_id to (match key, display label).

    A route whose *field* is blank falls back to its route_id, with a warning.

    Raises:
        ConfigError: routes.txt has no *field* column.
    """
    if field not in routes.columns:
        raise ConfigError(
            f"routes.txt has no {field!r} column; set GTFS_ROUTE_FIELD to one of "
            f"{', '.join(c for c in _GTFS_ROUTE_FIELDS if c in routes.columns)}."
        )
    out: dict[str, tuple[str, str]] = {}
    fell_back: list[str] = []
    for row in routes.drop_duplicates(subset="route_id").to_dict("records"):
        route_id = str(row["route_id"])
        name = str(row.get(field) or "").strip()
        if not name:
            name = route_id
            fell_back.append(route_id)
        out[route_id] = (normalize_route_key(name, ignore_leading_zeros), route_display_label(row))
    if fell_back:
        logging.warning(
            "%d route(s) have a blank %s and are matched on route_id instead: %s",
            len(fell_back),
            field,
            ", ".join(fell_back),
        )
    return out


# =============================================================================
# PATTERN SELECTION
# =============================================================================


def direction_keys(trips: pd.DataFrame) -> pd.Series:
    """Label each trip's direction: direction_id, else ``headsign:<trip_headsign>``.

    Trips with neither share the label ``""``.
    """
    direction = trips["direction_id"].astype(str)
    headsign = trips["trip_headsign"].astype(str)
    fallback = ("headsign:" + headsign).where(headsign != "", "")
    return direction.where(direction != "", fallback)


def _pick_shapes(
    counts: pd.Series, shape_lengths_ft: Mapping[str, float], rule: str
) -> list[tuple[str, int]]:
    """Choose shape(s) from one direction's trip counts per shape_id.

    Args:
        counts: Trips per shape_id in the direction (usable shapes only).
        shape_lengths_ft: Length of each shape in feet.
        rule: ``"most_trips"``, ``"longest"`` or ``"all"``.

    Returns:
        ``(shape_id, trips)`` pairs, busiest first. Ties go to the longer shape,
        then to the lower shape_id, so the choice is repeatable.
    """
    candidates = sorted(
        ((str(sid), int(n)) for sid, n in counts.items()),
        key=lambda c: (-c[1], -shape_lengths_ft[c[0]], c[0]),
    )
    if rule == "all":
        return candidates
    if rule == "longest":
        return [min(candidates, key=lambda c: (-shape_lengths_ft[c[0]], -c[1], c[0]))]
    return candidates[:1]


def select_patterns(
    trips: pd.DataFrame,
    shape_lengths_ft: Mapping[str, float],
    service_ids: Optional[set[str]],
    rule: str,
    route_keys: Mapping[str, tuple[str, str]],
) -> tuple[list[PatternChoice], dict[str, list[str]]]:
    """Pick the GTFS shape(s) that represent each route direction.

    Args:
        trips: Parsed ``trips.txt`` (see :func:`load_feed`).
        shape_lengths_ft: Length in feet of every shape with usable geometry.
        service_ids: Service_ids of the counted day, or ``None`` to count every
            trip.
        rule: ``PATTERN_RULE`` value.
        route_keys: Output of :func:`gtfs_route_keys`.

    Returns:
        Tuple of (choices, notes). Choices are grouped by route and ordered
        busiest first within a route; notes maps route_id to remarks such as
        "no trips on the service day".
    """
    choices: list[PatternChoice] = []
    notes: dict[str, list[str]] = {}
    for route_id, grp in trips.groupby("route_id", sort=True):
        rid = str(route_id)
        key, label = route_keys.get(rid, (normalize_route_key(rid, False), rid))
        route_notes = notes.setdefault(rid, [])
        day = grp
        basis = "all_trips"
        if service_ids is not None:
            day = grp[grp["service_id"].isin(service_ids)]
            basis = "service_day"
            if day.empty:
                day, basis = grp, "all_trips"
                route_notes.append("no trips on the service day; counted all trips")
        directions = direction_keys(day)
        if (day["direction_id"] == "").any():
            route_notes.append("blank direction_id; directions taken from trip_headsign")
        route_choices: list[PatternChoice] = []
        for direction in sorted(directions.unique()):
            in_dir = day[directions == direction]
            shaped = in_dir[in_dir["shape_id"].isin(shape_lengths_ft)]
            if shaped.empty:
                continue
            counts = shaped["shape_id"].value_counts()
            for shape_id, n in _pick_shapes(counts, shape_lengths_ft, rule):
                route_choices.append(
                    PatternChoice(
                        route_id=rid,
                        route_key=key,
                        route_label=label,
                        direction=str(direction),
                        shape_id=shape_id,
                        trips=n,
                        direction_trips=len(in_dir),
                        shapes_in_direction=len(counts),
                        length_ft=round(shape_lengths_ft[shape_id], 1),
                        day_basis=basis,
                    )
                )
        unshaped = int((~day["shape_id"].isin(shape_lengths_ft)).sum())
        if unshaped:
            route_notes.append(f"{unshaped} counted trip(s) have no usable shape")
        route_choices.sort(key=lambda c: (-c.trips, -c.length_ft, c.direction, c.shape_id))
        choices.extend(route_choices)
    return choices, notes


# =============================================================================
# GEOMETRY (shapely)
# =============================================================================


def build_shape_geometries(shapes: Optional[pd.DataFrame]) -> dict[str, LineString]:
    """Build one WGS 84 LineString per shape_id from a ``shapes.txt`` table.

    Points are ordered by numeric ``shape_pt_sequence``; rows with unparseable
    coordinates or sequence values are dropped, and shapes left with fewer than
    two points are skipped (with a warning).

    Args:
        shapes: Parsed ``shapes.txt`` (or None when the file is absent).

    Returns:
        Mapping of shape_id -> LineString.
    """
    if shapes is None or shapes.empty:
        return {}
    required = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    missing = sorted(required - set(shapes.columns))
    if missing:
        logging.warning("shapes.txt missing columns %s; ignoring shapes.", missing)
        return {}

    df = shapes.copy()
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
    df = df.sort_values(["shape_id", "shape_pt_sequence"])

    out: dict[str, LineString] = {}
    skipped = 0
    for shape_id, grp in df.groupby("shape_id"):
        coords = list(zip(grp["shape_pt_lon"].astype(float), grp["shape_pt_lat"].astype(float)))
        if len(coords) < 2:
            skipped += 1
            continue
        out[str(shape_id)] = LineString(coords)
    if skipped:
        logging.warning("Skipped %d shape(s) with fewer than 2 valid points.", skipped)
    return out


def resolve_projected_crs(geometries: Sequence[BaseGeometry], target_crs: str) -> Any:
    """Pick the projected CRS used for buffering and length math.

    Args:
        geometries: WGS 84 geometries (used for auto-UTM).
        target_crs: Explicit CRS string from the CONFIGURATION block, or ""
            to estimate a UTM zone from the geometries' extent.

    Returns:
        A CRS accepted by ``GeoSeries.to_crs``.

    Raises:
        ValueError: No geometry is available to estimate a CRS from.
    """
    if target_crs.strip():
        return target_crs.strip()
    if not geometries:
        raise ValueError("No shapes.txt geometry available to estimate a projected CRS from.")
    return gpd.GeoSeries(list(geometries), crs="EPSG:4326").estimate_utm_crs()


def crs_feet_per_unit(crs: Any) -> float:
    """Return how many feet one linear unit of a projected *crs* spans.

    Buffers and lengths are computed in the CRS's own unit (meters for UTM,
    US survey feet for most State Plane zones) and converted with this factor.

    Raises:
        ConfigError: *crs* is not a recognized projected CRS.
    """
    try:
        parsed = CRS.from_user_input(crs)
    except CRSError as exc:
        raise ConfigError(f"TARGET_CRS {crs!r} is not a recognized CRS: {exc}") from exc
    if not parsed.is_projected:
        raise ConfigError(
            f"TARGET_CRS must be a projected CRS (lengths in feet or meters); got {parsed.name!r}."
        )
    return float(parsed.axis_info[0].unit_conversion_factor) * FEET_PER_METER


def project_geometries(
    geoms: Mapping[str, BaseGeometry], src_crs: Any, dst_crs: Any
) -> dict[str, BaseGeometry]:
    """Reproject a mapping of geometries from *src_crs* to *dst_crs*."""
    if not geoms:
        return {}
    series = gpd.GeoSeries(list(geoms.values()), crs=src_crs).to_crs(dst_crs)
    return dict(zip(geoms.keys(), series))


def extract_lines(geom: BaseGeometry) -> list[LineString]:
    """Flatten a geometry to its LineString parts, merging touching pieces.

    Points and polygons that an intersection/difference can emit (e.g. where two
    alignments cross at a single vertex) are discarded — only linear geometry is
    meaningful in these layers.

    Args:
        geom: Any shapely geometry.

    Returns:
        A list of LineStrings (possibly empty).
    """
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return [merged]
        return list(merged.geoms)
    if isinstance(geom, GeometryCollection):
        parts: list[LineString] = []
        for part in geom.geoms:
            parts.extend(extract_lines(part))
        if len(parts) > 1:
            return extract_lines(MultiLineString(parts))
        return parts
    return []


def filter_short_lines(lines: Sequence[LineString], min_length: float) -> list[LineString]:
    """Drop overlay slivers: lines shorter than ``min_length`` (CRS units)."""
    return [line for line in lines if line.length >= min_length]


def total_length_ft(lines: Sequence[LineString], feet_per_unit: float) -> float:
    """Sum the length of projected lines, in feet."""
    return float(sum(line.length for line in lines)) * feet_per_unit


def geometry_bounds(geom: BaseGeometry) -> tuple[float, float, float, float]:
    """Return ``(xmin, ymin, xmax, ymax)`` of a geometry."""
    xmin, ymin, xmax, ymax = geom.bounds
    return (float(xmin), float(ymin), float(xmax), float(ymax))


def union_all(geoms: Sequence[BaseGeometry]) -> BaseGeometry:
    """Union geometries into one (callers pass at least one)."""
    return unary_union(list(geoms))


def chain_lines(lines: Sequence[BaseGeometry]) -> list[LineString]:
    """Join lines that meet end to end, without splitting them where they cross.

    Reference routes are often drawn as many street segments. Chaining them
    first means :func:`measure_lines` sees one line per run of segments rather
    than a meeting of separate lines at every joint.
    """
    parts = [part for line in lines for part in extract_lines(line)]
    if not parts:
        return []
    return extract_lines(MultiLineString(parts))


def _boxes_near(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], dist: float
) -> bool:
    """True when two ``(xmin, ymin, xmax, ymax)`` boxes come within *dist* of each other."""
    return (
        a[0] - dist <= b[2] and b[0] - dist <= a[2] and a[1] - dist <= b[3] and b[1] - dist <= a[3]
    )


def measure_lines(lines: Sequence[Any], buffer_dist: float, min_added: float = 0.0) -> list[Any]:
    """Keep what each line adds beyond the lines before it, so a street counts once.

    Lines are taken in order, and each keeps only its parts farther than
    *buffer_dist* from the lines kept before it. A piece left over from that
    trimming that is shorter than *min_added* (a line weaving just beyond the
    tolerance, a stub at a terminal) counts as the same street and is not
    kept; pieces the trimming did not touch are kept at any length. A line is
    never compared with itself, so its own pieces are never trimmed where they
    meet. Pass a route's shapes busiest first, so the busiest one is kept whole.

    Args:
        lines: Line geometries (possibly multipart or empty) in a projected CRS.
        buffer_dist: Distance within which two lines count as the same street.
        min_added: Shortest leftover piece kept after trimming, in CRS units.

    Returns:
        The kept single-part lines.
    """
    kept: list[Any] = []
    covers: list[Any] = []
    boxes: list[tuple[float, float, float, float]] = []
    for line in lines:
        parts = extract_lines(line)
        if not parts:
            continue
        box = geometry_bounds(line)
        near = [
            cover for cover, other in zip(covers, boxes) if _boxes_near(box, other, buffer_dist)
        ]
        if near:
            untouched = {round(part.length, 6) for part in parts}
            trimmed = extract_lines(line.difference(union_all(near)))
            parts = [
                part
                for part in trimmed
                if part.length >= min_added or round(part.length, 6) in untouched
            ]
        kept.extend(parts)
        covers.extend(part.buffer(buffer_dist) for part in parts)
        boxes.extend(geometry_bounds(part) for part in parts)
    return kept


def buffer_lines(lines: Sequence[Any], dist: float) -> Any:
    """Buffer each line and union the buffers.

    Gives the same area as buffering the union of the lines, without first
    splitting overlapping lines at every crossing, which is far slower on
    shapes that run side by side.
    """
    return union_all([line.buffer(dist) for line in lines])


def split_by_length(lines: Sequence[Any], min_length: float) -> tuple[list[Any], list[Any]]:
    """Split pieces into those at least *min_length* long and the shorter ones."""
    return (
        [line for line in lines if line.length >= min_length],
        [line for line in lines if line.length < min_length],
    )


def intersect_lines(geom: BaseGeometry, area: BaseGeometry) -> BaseGeometry:
    """Return the part of a line geometry inside a polygon."""
    return geom.intersection(area)


def compare_alignments(
    gtfs_lines: Sequence[Any],
    shp_lines: Sequence[Any],
    buffer_dist: float,
    min_segment_dist: float,
) -> dict[str, list[Any]]:
    """Classify and measure a route's GTFS shapes and reference lines (projected CRS).

    Each line is classified as drawn, against the complete linework of the
    other side: GTFS line farther than *buffer_dist* from every reference line
    is ``gtfs_only``, the rest of it ``matched``; reference line farther than
    *buffer_dist* from every GTFS shape is ``shp_only``, the rest of it
    ``shp_matched``. Each class is then measured with :func:`measure_lines`, so
    a street drawn in both directions counts once. Difference pieces shorter
    than *min_segment_dist* go to ``gtfs_short`` / ``shp_short`` instead of
    being reported. Each side's total is the sum of its three classes, so the
    classes always add up to it.

    Args:
        gtfs_lines: The route's chosen GTFS shapes, busiest first.
        shp_lines: The route's reference lines, longest first (same CRS).
        buffer_dist: Match tolerance in CRS units.
        min_segment_dist: Shortest difference piece reported, in CRS units.

    Returns:
        Lines by class: ``matched``, ``gtfs_only``, ``gtfs_short``,
        ``shp_matched``, ``shp_only`` and ``shp_short``.
    """
    shp_area = buffer_lines(shp_lines, buffer_dist)
    gtfs_area = buffer_lines(gtfs_lines, buffer_dist)
    gtfs_out = measure_lines(
        [line.difference(shp_area) for line in gtfs_lines], buffer_dist, min_segment_dist
    )
    shp_out = measure_lines(
        [line.difference(gtfs_area) for line in shp_lines], buffer_dist, min_segment_dist
    )
    gtfs_only, gtfs_short = split_by_length(gtfs_out, min_segment_dist)
    shp_only, shp_short = split_by_length(shp_out, min_segment_dist)
    return {
        "matched": measure_lines(
            [intersect_lines(line, shp_area) for line in gtfs_lines], buffer_dist, min_segment_dist
        ),
        "gtfs_only": gtfs_only,
        "gtfs_short": gtfs_short,
        "shp_matched": measure_lines(
            [intersect_lines(line, gtfs_area) for line in shp_lines], buffer_dist, min_segment_dist
        ),
        "shp_only": shp_only,
        "shp_short": shp_short,
    }


def measure_one_side(
    lines: Sequence[Any], side: str, buffer_dist: float, min_segment_dist: float
) -> dict[str, list[Any]]:
    """Measure a route found on one side only: all of it is a difference.

    Args:
        lines: The route's GTFS shapes or reference lines.
        side: ``"gtfs"`` or ``"shp"``.
        buffer_dist: Distance within which two lines count as the same street.
        min_segment_dist: Shortest difference piece reported.

    Returns:
        ``{"<side>_only": [...], "<side>_short": [...]}``.
    """
    measured = measure_lines(lines, buffer_dist, min_segment_dist)
    reported, short = split_by_length(measured, min_segment_dist)
    return {f"{side}_only": reported, f"{side}_short": short}


# =============================================================================
# REFERENCE ROUTE SHAPEFILE(S)
# =============================================================================


def _zip_shapefile_members(path: Path) -> list[str]:
    """List the ``.shp`` members of a zip, skipping macOS metadata entries."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc
    return sorted(
        name
        for name in names
        if name.lower().endswith(".shp")
        and not name.startswith("__MACOSX/")
        and not os.path.basename(name).startswith("._")
    )


def read_reference_layers(path: Path) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Read every shapefile ``ROUTE_SHAPES_PATH`` points to.

    Args:
        path: A ``.shp``, a ``.zip`` holding one or more shapefiles, or a folder
            of ``.shp`` files.

    Returns:
        ``(file stem, layer)`` pairs in file-name order.

    Raises:
        OSError: *path* does not exist.
        ValueError: *path* holds no shapefile, or is not a ``.shp``, ``.zip`` or
            folder.
    """
    if not path.exists():
        raise OSError(f"ROUTE_SHAPES_PATH '{path}' does not exist.")
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".shp")
        if not files:
            raise ValueError(f"No .shp files found in folder '{path}'.")
        return [(p.stem, load_shapefile(str(p))) for p in files]
    if path.suffix.lower() == ".zip":
        members = _zip_shapefile_members(path)
        if len(members) <= 1:
            stem = Path(members[0]).stem if members else path.stem
            return [(stem, load_shapefile(str(path)))]
        return [
            (Path(m).stem, load_shapefile(str(path), member=os.path.basename(m))) for m in members
        ]
    return [(path.stem, load_shapefile(str(path)))]


def _find_field(columns: Sequence[str], wanted: str) -> Optional[str]:
    """Return the column named *wanted*, ignoring case (shapefile fields vary)."""
    for col in columns:
        if col.lower() == wanted.lower():
            return col
    return None


def read_reference_features(
    path: Path, route_field: str, projected_crs: Any
) -> tuple[list[tuple[str, str, BaseGeometry]], list[str]]:
    """Read every reference line feature with its route name, projected.

    Args:
        path: ``ROUTE_SHAPES_PATH``.
        route_field: ``SHP_ROUTE_FIELD``; blank names each feature by its file.
        projected_crs: CRS the comparison runs in.

    Returns:
        Tuple of (``(layer name, raw route name, geometry)`` per line feature,
        names of every layer read).

    Raises:
        OSError, ValueError: The path is unusable, or a layer has no CRS.
        ConfigError: *route_field* is missing from a layer.
    """
    layers = read_reference_layers(path)
    features: list[tuple[str, str, BaseGeometry]] = []
    for stem, layer in layers:
        if layer.crs is None:
            raise ValueError(
                f"Shapefile '{stem}' has no coordinate system (.prj); define one in GIS first."
            )
        lines = layer[layer.geometry.notna() & ~layer.geometry.is_empty]
        lines = lines[lines.geom_type.isin(["LineString", "MultiLineString"])]
        if lines.empty:
            logging.warning("Shapefile '%s' holds no line features; skipped.", stem)
            continue
        if route_field.strip():
            col = _find_field(list(lines.columns), route_field.strip())
            if col is None:
                fields = ", ".join(c for c in lines.columns if c != lines.geometry.name)
                raise ConfigError(
                    f"Shapefile '{stem}' has no field {route_field!r}. Its fields: {fields}."
                )
            raw_names = [route_name_text(v) for v in lines[col]]
        else:
            raw_names = [stem] * len(lines)
        projected = lines.geometry.to_crs(projected_crs)
        features.extend((stem, raw, geom) for raw, geom in zip(raw_names, projected))
    return features, [stem for stem, _ in layers]


def load_reference_routes(
    path: Path,
    route_field: str,
    pattern: Optional[re.Pattern[str]],
    ignore_leading_zeros: bool,
    projected_crs: Any,
    aliases: Optional[Mapping[str, str]] = None,
) -> dict[str, ReferenceRoute]:
    """Load the reference route lines and group them by route.

    Every feature is assigned a route name from *route_field* (or its file's
    name when *route_field* is blank), renamed through *aliases*, and grouped
    with the other features of that name. A route's features are chained end
    to end (:func:`chain_lines`) and kept longest first, otherwise as drawn.

    Args:
        path: ``ROUTE_SHAPES_PATH``.
        route_field: ``SHP_ROUTE_FIELD``; blank names features by file.
        pattern: Compiled ``SHP_ROUTE_PATTERN`` or ``None``.
        ignore_leading_zeros: ``IGNORE_LEADING_ZEROS``.
        projected_crs: CRS the comparison runs in.
        aliases: ``SHP_ROUTE_ALIASES``.

    Returns:
        Mapping of route key to :class:`ReferenceRoute`.

    Raises:
        OSError, ValueError: The path is unusable, or a layer has no CRS.
        ConfigError: *route_field* is missing from a layer.
    """
    features, layer_names = read_reference_features(path, route_field, projected_crs)
    if not route_field.strip() and len(layer_names) == 1:
        logging.warning(
            "SHP_ROUTE_FIELD is blank, so every feature in '%s' counts as route %r. "
            "Set SHP_ROUTE_FIELD if this shapefile holds several routes.",
            layer_names[0],
            layer_names[0],
        )
    alias_keys = {
        normalize_route_key(name, ignore_leading_zeros): normalize_route_key(
            target, ignore_leading_zeros
        )
        for name, target in (aliases or {}).items()
    }

    pieces: dict[str, list[BaseGeometry]] = {}
    names: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    unmatched: set[str] = set()
    used_aliases: set[str] = set()
    blank = 0
    for stem, raw, geom in features:
        name, matched = apply_route_pattern(raw, pattern)
        if not matched:
            unmatched.add(raw)
        key = normalize_route_key(name, ignore_leading_zeros)
        if key in alias_keys:
            used_aliases.add(key)
            key = alias_keys[key]
        if not key:
            blank += 1
            continue
        pieces.setdefault(key, []).append(geom)
        names.setdefault(key, set()).add(raw)
        sources.setdefault(key, set()).add(stem)
        counts[key] = counts.get(key, 0) + 1
    if unmatched:
        logging.warning(
            "SHP_ROUTE_PATTERN found no route name in %d value(s), used whole: %s",
            len(unmatched),
            ", ".join(sorted(unmatched)[:20]),
        )
    if blank:
        logging.warning("Skipped %d reference feature(s) with a blank route name.", blank)
    if used_aliases:
        logging.info(
            "Renamed shapefile route(s) through SHP_ROUTE_ALIASES: %s",
            ", ".join(f"{k} -> {alias_keys[k]}" for k in sorted(used_aliases)),
        )
    unused = sorted(set(alias_keys) - used_aliases)
    if unused:
        logging.warning(
            "SHP_ROUTE_ALIASES name(s) not found in the shapefile(s): %s", ", ".join(unused)
        )

    out: dict[str, ReferenceRoute] = {}
    for key in sorted(pieces):
        lines = tuple(sorted(chain_lines(pieces[key]), key=lambda line: -line.length))
        if not lines:
            continue
        out[key] = ReferenceRoute(
            route_key=key,
            names=tuple(sorted(names[key])),
            sources=tuple(sorted(sources[key])),
            feature_count=counts[key],
            lines=lines,
        )
    logging.info("Loaded %d reference route(s) from %d layer(s).", len(out), len(layer_names))
    return out


# =============================================================================
# GTFS ROUTE LINES
# =============================================================================


def unique_shape_ids(patterns: Sequence[PatternChoice]) -> list[str]:
    """Return the patterns' shape_ids once each, in pattern order.

    Both directions of a route often run one shared shape.
    """
    return list(dict.fromkeys(p.shape_id for p in patterns))


def build_gtfs_routes(
    choices: Sequence[PatternChoice],
    notes: Mapping[str, list[str]],
    route_keys: Mapping[str, tuple[str, str]],
    trip_route_ids: Sequence[str],
    shape_geoms: Mapping[str, BaseGeometry],
) -> dict[str, GtfsRoute]:
    """Group each route key's chosen patterns with their shapes, busiest first.

    Args:
        choices: Output of :func:`select_patterns`.
        notes: Per-route_id remarks from :func:`select_patterns`.
        route_keys: Output of :func:`gtfs_route_keys`.
        trip_route_ids: Route_ids that have at least one trip.
        shape_geoms: Projected shape geometries by shape_id.

    Returns:
        Mapping of route key to :class:`GtfsRoute`. Routes without a usable
        shape carry no lines.
    """
    ids_by_key: dict[str, list[str]] = {}
    for rid in sorted(set(trip_route_ids)):
        key = route_keys.get(rid, (normalize_route_key(rid, False), rid))[0]
        ids_by_key.setdefault(key, []).append(rid)

    out: dict[str, GtfsRoute] = {}
    for key, route_ids in sorted(ids_by_key.items()):
        patterns = [c for c in choices if c.route_key == key]
        patterns.sort(key=lambda c: (-c.trips, -c.length_ft, c.route_id, c.shape_id))
        remarks = [
            f"{rid}: {n}" if len(route_ids) > 1 else n
            for rid in route_ids
            for n in notes.get(rid, [])
        ]
        if len(route_ids) > 1:
            remarks.insert(0, f"{len(route_ids)} GTFS routes share this name; combined")
        if not patterns:
            remarks.append("no trip has a usable shapes.txt shape")
        label = route_keys.get(route_ids[0], (key, route_ids[0]))[1]
        out[key] = GtfsRoute(
            route_key=key,
            route_label=label,
            route_ids=tuple(route_ids),
            patterns=tuple(patterns),
            lines=tuple(shape_geoms[sid] for sid in unique_shape_ids(patterns)),
            note="; ".join(remarks),
        )
    return out


# =============================================================================
# COMPARISON DRIVER
# =============================================================================


def _shp_text(value: Any) -> str:
    """Trim a value to what a shapefile text field can hold."""
    text = "" if value is None else str(value)
    return text[:_SHP_TEXT_MAX]


def compare_routes(
    gtfs_routes: Mapping[str, GtfsRoute],
    ref_routes: Mapping[str, ReferenceRoute],
    buffer_dist: float,
    min_segment_dist: float,
    mismatch_min_ft: float,
    feet_per_unit: float,
    idle_route_keys: frozenset[str] = frozenset(),
) -> tuple[list[RouteComparison], list[SegmentRecord], dict[str, list[Any]]]:
    """Compare every route key found on either side.

    Routes on both sides go through :func:`compare_alignments`; a route on one
    side only is measured whole as a difference. *idle_route_keys* names
    routes listed in routes.txt that no trip runs, so a shapefile route of that
    name is noted rather than unexplained.

    Returns:
        Tuple of (route rows, segment records, and each route's measured GTFS
        line pieces for the combined-route layer), geometry in the projected CRS.
    """
    rows: list[RouteComparison] = []
    segments: list[SegmentRecord] = []
    combined: dict[str, list[Any]] = {}
    for key in sorted(set(gtfs_routes) | set(ref_routes)):
        g = gtfs_routes.get(key)
        r = ref_routes.get(key)
        label = g.route_label if g is not None else key
        route_ids = ",".join(g.route_ids) if g is not None else ""
        notes = [g.note] if g is not None and g.note else []
        classes: dict[str, list[Any]] = {}

        status = ""
        if g is not None and not g.lines:
            status = "no_gtfs_geometry"
            if r is not None:
                classes = measure_one_side(r.lines, "shp", buffer_dist, min_segment_dist)
            else:
                notes.append("no reference line with this route name")
        elif g is not None and r is not None:
            classes = compare_alignments(g.lines, r.lines, buffer_dist, min_segment_dist)
        elif g is not None:
            status = "missing_from_shp"
            classes = measure_one_side(g.lines, "gtfs", buffer_dist, min_segment_dist)
            notes.append("no reference line with this route name")
        else:
            status = "missing_from_gtfs"
            if r is not None:
                classes = measure_one_side(r.lines, "shp", buffer_dist, min_segment_dist)
            if key in idle_route_keys:
                notes.append("routes.txt lists this route, but no trip runs it")
            else:
                notes.append("no GTFS route with trips has this name")

        lengths = {
            name: total_length_ft(classes.get(name, []), feet_per_unit)
            for name in (
                "matched",
                "gtfs_only",
                "gtfs_short",
                "shp_matched",
                "shp_only",
                "shp_short",
            )
        }
        mismatch_len = lengths["gtfs_only"] + lengths["shp_only"]
        if g is not None and r is not None and g.lines:
            status = "mismatch" if mismatch_len >= mismatch_min_ft else "match"
        gtfs_len = lengths["matched"] + lengths["gtfs_only"] + lengths["gtfs_short"]
        shp_len = lengths["shp_matched"] + lengths["shp_only"] + lengths["shp_short"]
        denom = max(gtfs_len, shp_len)
        for segment_class in ("gtfs_only", "shp_only", "matched"):
            segments.extend(
                SegmentRecord(key, label, route_ids, segment_class, status, line)
                for line in classes.get(segment_class, [])
            )
        pieces = [
            line
            for name in ("matched", "gtfs_only", "gtfs_short")
            for line in classes.get(name, [])
        ]
        if pieces:
            combined[key] = pieces
        rows.append(
            RouteComparison(
                route_key=key,
                route_label=label,
                gtfs_route_ids=route_ids,
                shp_route_names=";".join(r.names) if r is not None else "",
                shp_sources=";".join(r.sources) if r is not None else "",
                status=status,
                pattern_count=len(g.patterns) if g is not None else 0,
                shape_ids=";".join(unique_shape_ids(g.patterns)) if g is not None else "",
                gtfs_len_ft=gtfs_len,
                matched_len_ft=lengths["matched"],
                gtfs_only_len_ft=lengths["gtfs_only"],
                gtfs_short_len_ft=lengths["gtfs_short"],
                shp_len_ft=shp_len,
                shp_matched_len_ft=lengths["shp_matched"],
                shp_only_len_ft=lengths["shp_only"],
                shp_short_len_ft=lengths["shp_short"],
                mismatch_len_ft=mismatch_len,
                mismatch_share=(mismatch_len / denom) if denom > 0 else 0.0,
                note="; ".join(notes),
            )
        )
    return rows, segments, combined


# =============================================================================
# EXPORT
# =============================================================================


def unit_suffix(output_unit: DistanceUnit) -> str:
    """Column-name suffix for converted lengths: ``"mi"`` or ``"km"``."""
    return "km" if output_unit == "km" else "mi"


def remove_layer(path: Path) -> bool:
    """Delete a shapefile and its sidecar files; True when anything was removed."""
    removed = False
    for part in path.parent.glob(f"{path.stem}.*"):
        if part.name[len(path.stem) :].lower() in _SHAPEFILE_PARTS:
            part.unlink()
            removed = True
    return removed


def _length_unit_value(feet: float, output_unit: DistanceUnit) -> Optional[float]:
    """Convert feet to *output_unit* rounded for an output column."""
    value = convert_distance(feet, "feet", output_unit)
    return None if value is None else round(value, 3)


def write_line_shapefile(
    records: Sequence[Mapping[str, Any]],
    geometries: Sequence[BaseGeometry],
    path: Path,
    projected_crs: Any,
    feet_per_unit: float,
    output_unit: DistanceUnit,
) -> int:
    """Write line features to a shapefile in WGS 84, adding length fields.

    With no features, a shapefile an earlier run left at *path* is deleted, so
    every layer in the output folder belongs to this run.

    Args:
        records: One attribute mapping per feature (field names of 10
            characters or fewer).
        geometries: Feature geometries in *projected_crs*, aligned with *records*.
        path: Destination ``.shp`` path.
        projected_crs: CRS the geometries are in.
        feet_per_unit: Feet per linear unit of *projected_crs*.
        output_unit: Unit of the ``len_mi`` (or ``len_km``) field.

    Returns:
        Number of features written (0 when there was nothing to write).
    """
    if not records:
        if remove_layer(path):
            logging.info("No features for %s; removed the copy from an earlier run.", path.name)
        else:
            logging.info("No features for %s; file not written.", path.name)
        return 0
    frame = pd.DataFrame(
        [{k: _shp_text(v) if isinstance(v, str) else v for k, v in r.items()} for r in records]
    )
    frame["len_ft"] = [round(g.length * feet_per_unit, 1) for g in geometries]
    frame[f"len_{unit_suffix(output_unit)}"] = [
        _length_unit_value(ft, output_unit) for ft in frame["len_ft"]
    ]
    gdf = gpd.GeoDataFrame(frame, geometry=list(geometries), crs=projected_crs)
    gdf.to_crs("EPSG:4326").to_file(path)
    logging.info("Wrote %d feature(s): %s", len(gdf), path)
    return len(gdf)


def write_segment_layers(
    segments: Sequence[SegmentRecord],
    output_dir: Path,
    projected_crs: Any,
    feet_per_unit: float,
    output_unit: DistanceUnit,
) -> dict[str, int]:
    """Write the gtfs_only, shp_only and matched segment shapefiles.

    Returns:
        Feature count per shapefile name.
    """
    written: dict[str, int] = {}
    for segment_class, filename in (
        ("gtfs_only", SEGMENTS_GTFS_ONLY_FILENAME),
        ("shp_only", SEGMENTS_SHP_ONLY_FILENAME),
        ("matched", SEGMENTS_MATCHED_FILENAME),
    ):
        subset = [s for s in segments if s.segment_class == segment_class]
        records = [
            {
                "rt_key": s.route_key,
                "rt_label": s.route_label,
                "gtfs_rts": s.route_ids,
                "seg_class": s.segment_class,
                "rt_status": s.status,
            }
            for s in subset
        ]
        written[filename] = write_line_shapefile(
            records,
            [s.geometry for s in subset],
            output_dir / filename,
            projected_crs,
            feet_per_unit,
            output_unit,
        )
    return written


def write_gtfs_route_layers(
    gtfs_routes: Mapping[str, GtfsRoute],
    combined: Mapping[str, Sequence[Any]],
    shape_geoms: Mapping[str, BaseGeometry],
    output_dir: Path,
    projected_crs: Any,
    feet_per_unit: float,
    output_unit: DistanceUnit,
) -> dict[str, int]:
    """Write the combined-route and selected-pattern shapefiles.

    The combined-route line is each route's measured GTFS pieces from
    :func:`compare_routes`, so its length equals the route table's
    ``gtfs_len_ft``.

    Returns:
        Feature count per shapefile name.
    """
    written: dict[str, int] = {}
    drawn = [g for g in gtfs_routes.values() if combined.get(g.route_key)]
    written[COMBINED_ROUTES_FILENAME] = write_line_shapefile(
        [
            {
                "rt_key": g.route_key,
                "rt_label": g.route_label,
                "gtfs_rts": ",".join(g.route_ids),
                "n_patterns": len(g.patterns),
                "shape_ids": ";".join(unique_shape_ids(g.patterns)),
            }
            for g in drawn
        ],
        [union_all(combined[g.route_key]) for g in drawn],
        output_dir / COMBINED_ROUTES_FILENAME,
        projected_crs,
        feet_per_unit,
        output_unit,
    )
    patterns = [p for g in gtfs_routes.values() for p in g.patterns]
    written[PATTERNS_SHP_FILENAME] = write_line_shapefile(
        [
            {
                "route_id": p.route_id,
                "rt_key": p.route_key,
                "rt_label": p.route_label,
                "direction": p.direction,
                "shape_id": p.shape_id,
                "trips": p.trips,
                "dir_trips": p.direction_trips,
                "n_shapes": p.shapes_in_direction,
                "day_basis": p.day_basis,
            }
            for p in patterns
        ],
        [shape_geoms[p.shape_id] for p in patterns],
        output_dir / PATTERNS_SHP_FILENAME,
        projected_crs,
        feet_per_unit,
        output_unit,
    )
    return written


def patterns_frame(gtfs_routes: Mapping[str, GtfsRoute], output_unit: DistanceUnit) -> pd.DataFrame:
    """Tabulate the chosen patterns, one row per route direction and shape."""
    rows = [p._asdict() for g in gtfs_routes.values() for p in g.patterns]
    df = pd.DataFrame(rows, columns=list(PatternChoice._fields))
    df["trip_share"] = [
        round(t / d, 4) if d else 0.0 for t, d in zip(df["trips"], df["direction_trips"])
    ]
    df[f"length_{unit_suffix(output_unit)}"] = [
        _length_unit_value(ft, output_unit) for ft in df["length_ft"]
    ]
    return df


def route_table_frame(rows: Sequence[RouteComparison], output_unit: DistanceUnit) -> pd.DataFrame:
    """Assemble the per-route comparison table, mismatches first."""
    df = pd.DataFrame([r._asdict() for r in rows], columns=list(RouteComparison._fields))
    for col in (
        "gtfs_len_ft",
        "matched_len_ft",
        "gtfs_only_len_ft",
        "gtfs_short_len_ft",
        "shp_len_ft",
        "shp_matched_len_ft",
        "shp_only_len_ft",
        "shp_short_len_ft",
        "mismatch_len_ft",
    ):
        unit_col = col.replace("_ft", f"_{unit_suffix(output_unit)}")
        df[unit_col] = [_length_unit_value(v, output_unit) for v in df[col]]
        df[col] = df[col].astype(float).round(1)
    df["mismatch_share"] = df["mismatch_share"].astype(float).round(4)
    df["_order"] = df["status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    df = df.sort_values(["_order", "mismatch_len_ft", "route_key"], ascending=[True, False, True])
    return df.drop(columns="_order").reset_index(drop=True)


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def setting_literal(value: Any) -> str:
    """Write a setting the way the CONFIGURATION block spells it (paths as raw strings)."""
    if isinstance(value, Path):
        return f'r"{value.absolute()}"'
    if isinstance(value, Mapping):
        return repr(dict(value))
    return repr(value)


def write_run_log(output_dir: Path, summary_lines: List[str], settings: Mapping[str, Any]) -> bool:
    """Write the run summary, the settings used, and the verbatim config block.

    *settings* are the values the run actually used, after command-line flags
    override the CONFIGURATION block, so a run made with flags can be repeated.

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
        "GTFS VS ROUTE SHAPEFILE LINEAR DIFF RUN LOG",
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
        "SETTINGS USED (CONFIGURATION plus command-line flags)",
        "-" * 72,
        *(f"{name} = {setting_literal(value)}" for name, value in settings.items()),
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


def validate_config(cfg: Config) -> None:
    """Reject unusable CONFIGURATION values before any data is read.

    Raises:
        ConfigError: A value is out of range or not one of the allowed choices.
    """
    if cfg.pattern_rule not in _PATTERN_RULES:
        raise ConfigError(
            f"PATTERN_RULE must be one of {', '.join(_PATTERN_RULES)}; got {cfg.pattern_rule!r}."
        )
    if cfg.gtfs_route_field not in _GTFS_ROUTE_FIELDS:
        raise ConfigError(
            f"GTFS_ROUTE_FIELD must be one of {', '.join(_GTFS_ROUTE_FIELDS)}; "
            f"got {cfg.gtfs_route_field!r}."
        )
    if cfg.distance_output_unit not in ("miles", "km"):
        raise ConfigError(
            f"DISTANCE_OUTPUT_UNIT must be 'miles' or 'km'; got {cfg.distance_output_unit!r}."
        )
    for name, value in (
        ("BUFFER_TOLERANCE_FEET", cfg.buffer_tolerance_feet),
        ("MIN_SEGMENT_LENGTH_FEET", cfg.min_segment_length_feet),
        ("MISMATCH_MIN_FEET", cfg.mismatch_min_feet),
    ):
        if value < 0 or (name == "BUFFER_TOLERANCE_FEET" and value == 0):
            raise ConfigError(f"{name} must be positive; got {value!r}.")
    compile_route_pattern(cfg.shp_route_pattern)
    for name, target in cfg.shp_route_aliases.items():
        if not str(name).strip():
            raise ConfigError(f"SHP_ROUTE_ALIASES has a blank route name (for {target!r}).")
    if cfg.target_crs.strip():
        crs_feet_per_unit(cfg.target_crs.strip())


def run(cfg: Config) -> Summary:
    """Execute the comparison end-to-end and write all outputs.

    Args:
        cfg: Resolved configuration.

    Returns:
        Run-level :class:`Summary` (also written to the summary JSON).

    Raises:
        ConfigError: A CONFIGURATION value is unusable.
        OSError, ValueError: An input is missing or malformed.
        RuntimeError: The run log could not be written and ``REQUIRE_RUN_LOG``
            is True.
    """
    validate_config(cfg)
    pattern = compile_route_pattern(cfg.shp_route_pattern)

    feed = load_feed(cfg.gtfs_path)
    service_date, service_ids = select_service_ids(
        feed.get("calendar"), feed.get("calendar_dates"), cfg.service_day, cfg.service_date
    )
    route_keys = gtfs_route_keys(feed["routes"], cfg.gtfs_route_field, cfg.ignore_leading_zeros)
    trip_route_ids = set(feed["trips"]["route_id"])
    active_keys = {key for rid, (key, _) in route_keys.items() if rid in trip_route_ids}
    idle = {rid: key for rid, (key, _) in route_keys.items() if rid not in trip_route_ids}
    idle_route_keys = frozenset(set(idle.values()) - active_keys)
    if idle:
        logging.info(
            "%d route(s) in routes.txt have no trips and are left out: %s",
            len(idle),
            ", ".join(route_keys[rid][1] for rid in sorted(idle)),
        )

    shapes_ll = build_shape_geometries(feed["shapes"])
    projected_crs = resolve_projected_crs(list(shapes_ll.values()), cfg.target_crs)
    feet_per_unit = crs_feet_per_unit(projected_crs)
    logging.info("Projected CRS for overlay/lengths: %s", projected_crs)
    buffer_dist = cfg.buffer_tolerance_feet / feet_per_unit
    min_segment_dist = cfg.min_segment_length_feet / feet_per_unit
    shape_geoms = project_geometries(shapes_ll, "EPSG:4326", projected_crs)
    shape_lengths_ft = {sid: g.length * feet_per_unit for sid, g in shape_geoms.items()}

    choices, notes = select_patterns(
        feed["trips"], shape_lengths_ft, service_ids, cfg.pattern_rule, route_keys
    )
    gtfs_routes = build_gtfs_routes(
        choices,
        notes,
        route_keys,
        feed["trips"]["route_id"].tolist(),
        shape_geoms,
    )
    logging.info(
        "Picked %d pattern(s) for %d GTFS route(s) (rule: %s).",
        len(choices),
        len(gtfs_routes),
        cfg.pattern_rule,
    )
    ref_routes = load_reference_routes(
        cfg.route_shapes_path,
        cfg.shp_route_field,
        pattern,
        cfg.ignore_leading_zeros,
        projected_crs,
        cfg.shp_route_aliases,
    )
    rows, segments, combined = compare_routes(
        gtfs_routes,
        ref_routes,
        buffer_dist,
        min_segment_dist,
        cfg.mismatch_min_feet,
        feet_per_unit,
        idle_route_keys,
    )

    by_status = {s: [r for r in rows if r.status == s] for s in STATUS_ORDER}
    for status in ("mismatch", "missing_from_shp", "missing_from_gtfs", "no_gtfs_geometry"):
        if by_status[status]:
            logging.warning(
                "%d route(s) %s: %s",
                len(by_status[status]),
                status.replace("_", " "),
                ", ".join(r.route_label for r in by_status[status]),
            )
    logging.info("%d route(s) match within tolerance.", len(by_status["match"]))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    unit = cfg.distance_output_unit
    layers = write_segment_layers(segments, cfg.output_dir, projected_crs, feet_per_unit, unit)
    layers.update(
        write_gtfs_route_layers(
            gtfs_routes, combined, shape_geoms, cfg.output_dir, projected_crs, feet_per_unit, unit
        )
    )
    for frame, filename in (
        (patterns_frame(gtfs_routes, unit), PATTERNS_CSV_FILENAME),
        (route_table_frame(rows, unit), ROUTE_TABLE_FILENAME),
    ):
        frame.to_csv(cfg.output_dir / filename, index=False, encoding="utf-8")
        logging.info("Wrote: %s", cfg.output_dir / filename)

    compared = [r for r in rows if r.status in ("match", "mismatch")]

    def total_in_unit(feet: float) -> float:
        value = convert_distance(feet, "feet", unit)
        return round(value or 0.0, 2)

    summary = Summary(
        gtfs_route_count=len(gtfs_routes),
        shp_route_count=len(ref_routes),
        compared_count=len(compared),
        match_count=len(by_status["match"]),
        mismatch_count=len(by_status["mismatch"]),
        missing_from_shp_count=len(by_status["missing_from_shp"]),
        missing_from_gtfs_count=len(by_status["missing_from_gtfs"]),
        no_gtfs_geometry_count=len(by_status["no_gtfs_geometry"]),
        service_date=service_date.isoformat() if service_date else "all trips",
        pattern_rule=cfg.pattern_rule,
        length_unit=unit,
        matched_length=total_in_unit(sum(r.matched_len_ft for r in compared)),
        gtfs_only_length=total_in_unit(sum(r.gtfs_only_len_ft for r in compared)),
        shp_only_length=total_in_unit(sum(r.shp_only_len_ft for r in compared)),
        buffer_tolerance_feet=cfg.buffer_tolerance_feet,
        min_segment_length_feet=cfg.min_segment_length_feet,
        mismatch_min_feet=cfg.mismatch_min_feet,
        projected_crs=str(projected_crs),
    )
    summary_path = cfg.output_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary._asdict(), indent=2), encoding="utf-8")
    logging.info("Wrote: %s", summary_path)

    summary_lines = [
        f"GTFS / shapefile routes:  {summary.gtfs_route_count} / {summary.shp_route_count}",
        f"Service day counted:      {summary.service_date} ({cfg.service_day})",
        f"Pattern rule:             {summary.pattern_rule}",
        f"Match / mismatch:         {summary.match_count} / {summary.mismatch_count}",
        f"Missing from shapefile:   {summary.missing_from_shp_count}",
        f"Missing from GTFS:        {summary.missing_from_gtfs_count}",
        f"No GTFS geometry:         {summary.no_gtfs_geometry_count}",
        f"GTFS-only / shp-only:     {summary.gtfs_only_length} / {summary.shp_only_length} {unit}",
        f"Projected CRS:            {summary.projected_crs}",
        "Layers written:           "
        + (", ".join(f"{name} ({n})" for name, n in layers.items() if n) or "none"),
        "Layers with no features:  "
        + (", ".join(name for name, n in layers.items() if not n) or "none"),
    ]
    settings = {name.upper(): value for name, value in cfg._asdict().items()}
    if not write_run_log(cfg.output_dir, summary_lines, settings) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return summary


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_route_aliases(values: Sequence[str]) -> dict[str, str]:
    """Parse ``--shp-route-alias NAME=ROUTE`` values into a mapping.

    Raises:
        ConfigError: A value has no ``=`` or a blank name.
    """
    aliases: dict[str, str] = {}
    for value in values:
        name, sep, target = value.partition("=")
        if not sep or not name.strip():
            raise ConfigError(f"--shp-route-alias expects NAME=ROUTE; got {value!r}.")
        aliases[name.strip()] = target.strip()
    return aliases


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Compare GTFS route alignments with route shapefile(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gtfs", default=GTFS_PATH, help="GTFS folder or .zip.")
    p.add_argument(
        "--route-shapes",
        default=ROUTE_SHAPES_PATH,
        help="Route .shp, folder of .shp files (one per route), or .zip of either.",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--shp-route-field",
        default=SHP_ROUTE_FIELD,
        help="Shapefile field naming the route; blank uses each file's name.",
    )
    p.add_argument(
        "--shp-route-pattern",
        default=SHP_ROUTE_PATTERN,
        help="Regex whose first group pulls the route name from the field or file name.",
    )
    p.add_argument(
        "--shp-route-alias",
        action="append",
        metavar="NAME=ROUTE",
        help="Read shapefile route NAME as ROUTE; repeat for each alias. "
        "Replaces SHP_ROUTE_ALIASES when given.",
    )
    p.add_argument(
        "--gtfs-route-field",
        choices=_GTFS_ROUTE_FIELDS,
        default=GTFS_ROUTE_FIELD,
        help="routes.txt column matched against the shapefile's route names.",
    )
    p.add_argument(
        "--ignore-leading-zeros",
        action=argparse.BooleanOptionalAction,
        default=IGNORE_LEADING_ZEROS,
        help='Match "010" to "10".',
    )
    p.add_argument(
        "--service-day",
        default=SERVICE_DAY,
        help='"weekday" or a day name; its trips pick each direction\'s pattern.',
    )
    p.add_argument("--service-date", default=SERVICE_DATE, help="Explicit YYYY-MM-DD service date.")
    p.add_argument(
        "--pattern-rule",
        choices=_PATTERN_RULES,
        default=PATTERN_RULE,
        help="How each direction's pattern is chosen.",
    )
    p.add_argument(
        "--buffer-feet",
        type=float,
        default=BUFFER_TOLERANCE_FEET,
        help="Distance within which two lines count as the same street, in feet.",
    )
    p.add_argument(
        "--min-segment-feet",
        type=float,
        default=MIN_SEGMENT_LENGTH_FEET,
        help="Drop overlay pieces shorter than this, in feet.",
    )
    p.add_argument(
        "--mismatch-min-feet",
        type=float,
        default=MISMATCH_MIN_FEET,
        help="gtfs_only + shp_only length that flags a route as a mismatch, in feet.",
    )
    p.add_argument(
        "--target-crs",
        default=TARGET_CRS,
        help='Projected CRS for the overlay, e.g. "EPSG:2248"; empty auto-picks a UTM zone.',
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths and settings before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if a CONFIGURATION
        value is still a placeholder or is unusable.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    placeholders = [
        flag
        for flag, value in (
            ("--gtfs (GTFS_PATH)", args.gtfs),
            ("--route-shapes (ROUTE_SHAPES_PATH)", args.route_shapes),
            ("--output-dir (OUTPUT_DIR)", args.output_dir),
        )
        if str(value).startswith(PLACEHOLDER_PREFIX)
    ]
    if placeholders:
        logging.warning(
            "These paths are still placeholders: %s. Update the CONFIGURATION section "
            "or pass the matching flags before running.",
            ", ".join(placeholders),
        )
        return 2
    try:
        aliases = (
            parse_route_aliases(args.shp_route_alias)
            if args.shp_route_alias is not None
            else dict(SHP_ROUTE_ALIASES)
        )
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2

    cfg = Config(
        gtfs_path=Path(args.gtfs).expanduser(),
        route_shapes_path=Path(args.route_shapes).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        shp_route_field=args.shp_route_field,
        shp_route_pattern=args.shp_route_pattern,
        shp_route_aliases=aliases,
        gtfs_route_field=args.gtfs_route_field,
        ignore_leading_zeros=args.ignore_leading_zeros,
        service_day=args.service_day,
        service_date=args.service_date,
        pattern_rule=args.pattern_rule,
        buffer_tolerance_feet=args.buffer_feet,
        min_segment_length_feet=args.min_segment_feet,
        mismatch_min_feet=args.mismatch_min_feet,
        target_crs=args.target_crs,
        distance_output_unit=DISTANCE_OUTPUT_UNIT,
    )
    try:
        run(cfg)
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
