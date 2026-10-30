"""Map every distinct stop/path pattern of each GTFS route, one panel per pattern.

Official route maps usually show one version of a route. This script finds every
distinct way a route's trips actually run — a pattern is a unique combination of
route, direction, path (shape geometry) and ordered stop list, optionally
including pickup/drop-off restrictions — and draws each one in its own map
panel: the pattern's line with arrows for the direction of travel, its stops,
and the route's other patterns in light gray for context. Each panel lists the
pattern's trip counts and first/last departures on a representative weekday,
Saturday and Sunday, and how it differs from the direction's main pattern (P1,
the one with the most trips), so readers can tell the main pattern from
occasional variants.

``stop_pattern_exporter.py`` lists patterns as Excel stop tables, and
``gtfs_to_shapefile_*.py`` export shapes for GIS; this script is the visual
companion.

Inputs
------
- A GTFS feed folder or ``.zip`` with routes, trips, stop_times and stops.
  shapes.txt, calendar.txt and calendar_dates.txt are used when present;
  without shapes, patterns are drawn stop to stop.
- Optional ``BACKDROP_PATH``: street linework drawn faintly behind each panel.

Outputs
-------
- ``pattern_maps/route_<name>_dir<id>.png``: one figure per route and
  direction, one panel per pattern (long lists continue in ``_p2``, ``_p3`` …).
- ``pattern_summary.csv``: one row per pattern — trips, day-type counts and
  first/last departures, terminals, length, and differences from P1.
- ``pattern_stops.csv``: the ordered stop list of every pattern.
- ``pattern_map_exporter_runlog.txt``: run-log sidecar with the verbatim
  CONFIGURATION block and the representative dates used.

Method
------
- Shapes with identical coordinates count as one path even if their shape_ids
  differ. Shapes that differ only slightly still split patterns; the panel then
  reads "different shape, same streets".
- A shape drawn opposite to its trips' stop order (e.g. one shape reused for
  both directions) is reversed so the arrows follow the direction of travel,
  and each line is trimmed to run from the pattern's first stop to its last.
- Trip counts treat each trip in trips.txt once; frequencies.txt is not expanded.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags)
and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import os
import re
import sys
import textwrap
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from pyproj import CRS
from shapely.geometry import LineString, Point, box
from shapely.ops import substring

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# === BEGIN CONFIG ===

# GTFS feed: a folder of .txt files or the published .zip.
GTFS_PATH: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me
OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me

# Routes to map, e.g. ["101", "102"]. Each name is matched against
# route_short_name first and route_id second. Leave empty to map every route.
ROUTE_NAMES: List[str] = []

# Optional text file with one route name per line ('#' starts a comment).
# Its names are unioned with ROUTE_NAMES. None or "" to skip.
ROUTE_NAMES_FILE: Optional[str] = None

# Optional service_id filter (values from trips.txt). Leave empty to find
# patterns across every service day in the feed.
SERVICE_IDS_TO_INCLUDE: List[str] = []

# Day types to report trip counts and first/last departures for: "weekday" or a
# day name ("monday" … "sunday"). A representative date is picked for each from
# calendar.txt / calendar_dates.txt, skipping federal holidays. Leave empty to
# report feed-wide trip counts only.
DAY_TYPES: List[str] = ["weekday", "saturday", "sunday"]

# If True, trips serving the same stops but with different pickup/drop-off
# restrictions (e.g. one stop drop-off only) count as separate patterns.
SPLIT_ON_BOARDING_RESTRICTIONS: bool = True

# Projected CRS used for drawing and distances (feet or metre units both work).
# Default: Maryland State Plane, US feet (DC region). None = the feed's UTM zone.
PROJECTED_CRS: Optional[str] = "EPSG:2248"

# Optional street linework drawn faintly behind each panel: a .shp, a zipped
# shapefile, a .gpkg or a .geojson. None for a blank background.
BACKDROP_PATH: Optional[str] = None

# When BACKDROP_PATH is a .zip holding several shapefiles, the one to draw,
# e.g. "road_centerlines.shp". None when the zip holds a single shapefile.
BACKDROP_ZIP_MEMBER: Optional[str] = None

# A pattern must run farther than this from P1's line to count as using
# different streets.
OFF_PATH_TOLERANCE_FT: float = 100.0

# Figure layout: panels per figure (more patterns continue in _p2, _p3 …
# files), panels per row, and PNG resolution.
MAX_PANELS_PER_FIGURE: int = 9
PANEL_COLUMNS: int = 3
FIGURE_DPI: int = 150

# Roughly how many direction arrows fit across a panel's diagonal; arrow
# spacing is the same in every panel of a route, so longer patterns get more.
ARROWS_ACROSS_PANEL: int = 12

# Output names (written inside OUTPUT_DIR).
MAPS_SUBDIR: str = r"pattern_maps"
SUMMARY_FILENAME: str = r"pattern_summary.csv"
STOPS_FILENAME: str = r"pattern_stops.csv"
RUN_LOG_FILENAME: str = r"pattern_map_exporter_runlog.txt"

# When True, a failed run-log write aborts the script so no output is left
# without its configuration record. Set False only for read-only destinations.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

REQUIRED_GTFS_FILES: List[str] = ["routes.txt", "trips.txt", "stop_times.txt", "stops.txt"]

# Drawing style.
PATTERN_COLOR: str = "#1f5fa8"
CONTEXT_COLOR: str = "#c4c4c4"
BACKDROP_COLOR: str = "#e3e3e3"
START_COLOR: str = "#2e7d32"
END_COLOR: str = "#b71c1c"
NO_SERVICE_COLOR: str = "#7a7a7a"
PANEL_WIDTH_IN: float = 4.4
PANEL_HEADER_IN: float = 0.95
ARROW_HEAD_SIZE: float = 13.0

DAY_TYPE_ABBREV: dict[str, str] = {
    "weekday": "Wkdy",
    "monday": "Mon",
    "tuesday": "Tue",
    "wednesday": "Wed",
    "thursday": "Thu",
    "friday": "Fri",
    "saturday": "Sat",
    "sunday": "Sun",
}

FEET_PER_MILE: float = 5280.0

# =============================================================================
# HELPERS (canonical copies — see utils/)
# =============================================================================


class ServiceSelectionError(ValueError):
    """A requested GTFS service_id is not used by any trip in the feed.

    A configuration problem rather than a crash: callers report it without a
    traceback and exit with code 2.
    """


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


def load_id_set(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
    *,
    kind: str = "id",
) -> set[str]:
    """Union an inline list and an optional text file of ids into one set.

    Used to resolve override lists (express routes, express origin stops, …) that
    a caller may supply inline, in an external file, or both — without repeating
    the parsing for each one.

    Args:
        inline_ids: Id values supplied directly (e.g. a config list). ``None`` is
            treated as empty.
        txt_path: Path to a text file with one id per line. Blank lines are
            skipped and ``#`` starts a comment (whole-line or inline). ``None``
            skips the file. A path that is set but missing is logged as a warning
            and skipped — the inline ids are still returned.
        kind: Human-readable noun used only in log messages (e.g.
            ``"express route"``, ``"express origin stop"``).

    Returns:
        The unioned set of id strings (possibly empty). Every id is coerced to a
        trimmed ``str`` so it matches GTFS values, which are read as strings.
    """
    ids: set[str] = set()

    for raw in inline_ids or ():
        text = str(raw).strip()
        if text:
            ids.add(text)

    if txt_path:
        if not os.path.exists(txt_path):
            logging.warning(
                "%s file '%s' not found; using inline ids only.", kind.capitalize(), txt_path
            )
        else:
            with open(txt_path, encoding="utf-8") as handle:
                for line in handle:
                    text = line.split("#", 1)[0].strip()
                    if text:
                        ids.add(text)
            logging.info("Loaded %s ids from '%s'.", kind, txt_path)

    logging.info("Resolved %d %s id(s).", len(ids), kind)
    return ids


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
            more than one shapefile.

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
            shp_members = [name for name in archive.namelist() if name.lower().endswith(".shp")]
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


def select_trips_by_service(
    trips_df: pd.DataFrame,
    service_ids: Sequence[str],
    *,
    default_service_ids: Sequence[str] = (),
) -> pd.DataFrame:
    """Return the trips that run on *service_ids*; every trip when it is empty.

    Warns when *service_ids* still equals *default_service_ids* (the calling
    script's shipped default), so an unedited default is noticed, and warns that an
    empty selection combines every service day.

    Args:
        trips_df: Parsed ``trips.txt`` (needs a ``service_id`` column).
        service_ids: The service_id values to keep. Empty keeps every trip.
        default_service_ids: The calling script's default, used only for the warning.

    Returns:
        The trips whose ``service_id`` is in *service_ids*.

    Raises:
        ServiceSelectionError: If a requested service_id is used by no trip. The
            message lists the feed's service_ids with their trip counts.
    """
    wanted = [str(s).strip() for s in service_ids if str(s).strip()]
    if not wanted:
        logging.warning(
            "No service_id selected: using every trip in the feed, all service days "
            "combined. Set SERVICE_IDS_TO_INCLUDE to analyze a single service day."
        )
        return trips_df
    defaults = [str(s).strip() for s in default_service_ids]
    if defaults and wanted == defaults:
        logging.warning(
            "SERVICE_IDS_TO_INCLUDE is still the default %s. Check it against calendar.txt "
            "and change it if this feed uses other service_ids or you want another day.",
            wanted,
        )
    trip_service = trips_df["service_id"].astype(str).str.strip()
    counts = trip_service.value_counts().sort_index()
    missing = [s for s in wanted if s not in counts.index]
    if missing:
        available = ", ".join(f"{sid} ({n} trips)" for sid, n in counts.items())
        raise ServiceSelectionError(
            f"service_id(s) {missing} are not used by any trip in trips.txt. This feed's "
            f"service_ids: {available}. Set SERVICE_IDS_TO_INCLUDE to the service day to "
            "analyze (calendar.txt shows which days each service_id runs)."
        )
    kept = trips_df[trip_service.isin(wanted)]
    logging.info("Service filter %s: trips %d -> %d.", wanted, len(trips_df), len(kept))
    return kept


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


def route_label(route_short_name: str, route_id: str) -> str:
    """Return the display label for a route: its short name, else its route_id."""
    short = str(route_short_name).strip()
    return short if short and short.lower() != "nan" else str(route_id).strip()


def identify_target_route_ids(routes: pd.DataFrame, names: set[str]) -> tuple[set[str], set[str]]:
    """Resolve route names to route_id values.

    Each name is compared (trimmed) against ``route_short_name`` first and
    ``route_id`` second, so a name that names a short name shared by several
    ``route_id`` values (a common pattern for per-calendar route ids) selects
    all of them.

    Args:
        routes: The routes.txt table (string columns).
        names: Route names to resolve.

    Returns:
        A tuple ``(matched_route_ids, unmatched_names)``.
    """
    route_ids = routes["route_id"].astype(str).str.strip()
    if "route_short_name" in routes.columns:
        short_names = routes["route_short_name"].fillna("").astype(str).str.strip()
    else:
        short_names = pd.Series("", index=routes.index, dtype=str)

    matched: set[str] = set()
    unmatched: set[str] = set()
    for name in sorted(names):
        hits = route_ids[(short_names == name) | (route_ids == name)]
        if hits.empty:
            unmatched.add(name)
        else:
            matched.update(hits.tolist())
    return matched, unmatched


def sanitize_filename_component(value: str, max_len: int = 40) -> str:
    """Make a string safe to use as (part of) a file name.

    Collapses every character outside ``[A-Za-z0-9_]`` to an underscore,
    trims leading/trailing underscores, and truncates to *max_len*
    characters.

    Args:
        value: Raw string (e.g. a route_short_name or direction_id).
        max_len: Maximum length of the returned string.

    Returns:
        A non-empty, filesystem-safe string ("unnamed" if nothing survives).
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return cleaned[:max_len].rstrip("_") or "unnamed"


def natural_sort_key(label: str) -> str:
    """Return a sort key that orders embedded numbers numerically ("2" < "10")."""
    return re.sub(r"\d+", lambda m: m.group().zfill(8), label)


def load_optional_table(gtfs_path: Path, file_name: str) -> Optional[pd.DataFrame]:
    """Load one optional GTFS file, returning ``None`` when it is absent or unreadable."""
    try:
        return load_gtfs_data(str(gtfs_path), files=(file_name,))[file_name[:-4]]
    except (OSError, ValueError) as exc:
        logging.info("%s not usable (%s) — continuing without it.", file_name, exc)
        return None


def build_shape_lines(shapes: Optional[pd.DataFrame]) -> dict[str, LineString]:
    """Build one WGS 84 LineString per shape_id from shapes.txt.

    Args:
        shapes: Parsed shapes.txt, or ``None`` when the feed has none.

    Returns:
        Mapping of shape_id to LineString (lon/lat). Shapes with fewer than
        two distinct valid points are skipped with a warning. Empty when
        *shapes* is missing or lacks the required columns.
    """
    if shapes is None or shapes.empty:
        return {}
    required = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    missing = [c for c in required if c not in shapes.columns]
    if missing:
        logging.warning("shapes.txt lacks %s; drawing patterns stop to stop.", missing)
        return {}

    df = shapes[required].copy()
    df["shape_id"] = df["shape_id"].astype(str).str.strip()
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_bad = int(df[["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]].isna().any(axis=1).sum())
    if n_bad:
        logging.warning("Dropped %d shapes.txt row(s) with unparseable values.", n_bad)
    df = df.dropna().sort_values(["shape_id", "shape_pt_sequence"], kind="mergesort")

    lines: dict[str, LineString] = {}
    for shape_id, grp in df.groupby("shape_id", sort=False):
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        if len(set(coords)) < 2:
            logging.warning("Shape %s has fewer than two distinct points; skipped.", shape_id)
            continue
        lines[str(shape_id)] = LineString(coords)
    return lines


def group_identical_shapes(lines: Mapping[str, LineString], decimals: int = 6) -> dict[str, str]:
    """Map every shape_id to a canonical shape_id with identical coordinates.

    Some feeds publish a separate shape_id per trip even when the path is the
    same; grouping them keeps those trips in one pattern.

    Args:
        lines: Output of :func:`build_shape_lines`.
        decimals: Coordinates are rounded to this many decimal degrees
            (6 ≈ 0.1 m) before comparison.

    Returns:
        Mapping of each shape_id to the alphabetically first shape_id that has
        the same rounded coordinate sequence (itself when unique).
    """
    first_by_coords: dict[tuple[tuple[float, float], ...], str] = {}
    canonical: dict[str, str] = {}
    for shape_id in sorted(lines):
        key = tuple((round(x, decimals), round(y, decimals)) for x, y in lines[shape_id].coords)
        canonical[shape_id] = first_by_coords.setdefault(key, shape_id)
    n_merged = sum(1 for sid, canon in canonical.items() if sid != canon)
    if n_merged:
        logging.info("%d shape_id(s) duplicate another shape's coordinates; merged.", n_merged)
    return canonical


def boarding_status(pickup_type: pd.Series, drop_off_type: pd.Series) -> np.ndarray:
    """Classify each stop-time as regular, pickup_only, dropoff_only or no_service.

    GTFS ``pickup_type``/``drop_off_type`` value 1 means "not available"; every
    other value (blank, 0, 2 = phone agency, 3 = coordinate with driver) still
    lets riders board or alight.

    Args:
        pickup_type: stop_times.txt pickup_type values (strings, may be blank).
        drop_off_type: stop_times.txt drop_off_type values.

    Returns:
        Array of status labels aligned with the inputs.
    """
    no_pickup = pickup_type.fillna("").astype(str).str.strip() == "1"
    no_drop = drop_off_type.fillna("").astype(str).str.strip() == "1"
    return np.select(
        [no_pickup & no_drop, no_drop, no_pickup],
        ["no_service", "pickup_only", "dropoff_only"],
        default="regular",
    )


def build_trip_table(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes: pd.DataFrame,
    shape_canon: Mapping[str, str],
) -> pd.DataFrame:
    """Build one row per trip with its route, direction, path key and stop list.

    Args:
        trips: trips.txt, already filtered to the routes and services to map.
        stop_times: stop_times.txt.
        routes: routes.txt.
        shape_canon: Output of :func:`group_identical_shapes`.

    Returns:
        DataFrame with ``trip_id``, ``route_id``, ``route_label``,
        ``route_long_name``, ``service_id``, ``direction_id``, ``shape_id``,
        ``path_key`` (canonical shape_id, blank when the trip has no usable
        shape), ``trip_headsign``, ``stop_ids`` and ``boarding`` (tuples in
        stop order) and ``start_min`` (first departure, minutes past midnight).
    """
    table = pd.DataFrame(
        {
            "trip_id": trips["trip_id"].astype(str).str.strip(),
            "route_id": trips["route_id"].astype(str).str.strip(),
            "service_id": trips["service_id"].astype(str).str.strip(),
        }
    )
    for col in ("direction_id", "shape_id", "trip_headsign"):
        table[col] = trips[col].fillna("").astype(str).str.strip() if col in trips.columns else ""

    short = (
        routes["route_short_name"].fillna("")
        if "route_short_name" in routes.columns
        else pd.Series("", index=routes.index)
    )
    long_name = (
        routes["route_long_name"].fillna("").astype(str).str.strip()
        if "route_long_name" in routes.columns
        else pd.Series("", index=routes.index)
    )
    route_info = pd.DataFrame(
        {
            "route_id": routes["route_id"].astype(str).str.strip(),
            "route_label": [route_label(s, r) for s, r in zip(short, routes["route_id"])],
            "route_long_name": long_name,
        }
    ).drop_duplicates("route_id")
    table = table.merge(route_info, on="route_id", how="left")
    table["route_label"] = table["route_label"].fillna(table["route_id"])
    table["route_long_name"] = table["route_long_name"].fillna("")

    st = stop_times[stop_times["trip_id"].astype(str).str.strip().isin(set(table["trip_id"]))]
    st = st.copy()
    st["trip_id"] = st["trip_id"].astype(str).str.strip()
    st["stop_id"] = st["stop_id"].astype(str).str.strip()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    n_bad = int(st["stop_sequence"].isna().sum())
    if n_bad:
        logging.warning("Dropped %d stop_times row(s) with unparseable stop_sequence.", n_bad)
        st = st.dropna(subset=["stop_sequence"])
    st = st.sort_values(["trip_id", "stop_sequence"], kind="mergesort")

    blank = pd.Series("", index=st.index)
    st["boarding"] = boarding_status(
        st["pickup_type"] if "pickup_type" in st.columns else blank,
        st["drop_off_type"] if "drop_off_type" in st.columns else blank,
    )
    departure = st["departure_time"] if "departure_time" in st.columns else blank
    arrival = st["arrival_time"] if "arrival_time" in st.columns else blank
    departure = departure.fillna("").astype(str).str.strip()
    st["time"] = departure.where(departure != "", arrival.fillna("").astype(str).str.strip())
    st["time"] = st["time"].replace("", np.nan)

    grouped = st.groupby("trip_id", sort=False)
    signatures = pd.DataFrame(
        {
            "stop_ids": grouped["stop_id"].agg(tuple),
            "boarding": grouped["boarding"].agg(tuple),
            "first_time": grouped["time"].first(),
        }
    )
    signatures["start_min"] = pd.to_numeric(
        signatures["first_time"].map(parse_time_to_minutes), errors="coerce"
    )

    before = len(table)
    table = table.merge(
        signatures[["stop_ids", "boarding", "start_min"]],
        left_on="trip_id",
        right_index=True,
        how="inner",
    )
    if len(table) < before:
        logging.warning("%d trip(s) have no stop_times rows; skipped.", before - len(table))

    table["path_key"] = table["shape_id"].map(shape_canon).fillna("")
    n_no_shape = int((table["path_key"] == "").sum())
    if n_no_shape:
        logging.info("%d trip(s) have no usable shape; drawn stop to stop.", n_no_shape)
    return table.reset_index(drop=True)


def find_patterns(
    trip_table: pd.DataFrame, split_on_boarding: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group trips into unique patterns and rank them within each route/direction.

    A pattern is a unique (route label, direction_id, path key, stop list)
    combination, plus the per-stop boarding restrictions when
    *split_on_boarding* is True. Patterns are grouped by route *label* (the
    public route number), so per-calendar route_ids that share a short name
    map together. Within each route and direction, patterns are ranked by trip
    count (then by stop count); ``P1`` is the main pattern.

    Args:
        trip_table: Output of :func:`build_trip_table`.
        split_on_boarding: Treat boarding-restriction differences as distinct
            patterns.

    Returns:
        Tuple of (patterns, trip_table). ``patterns`` has one row per pattern
        in rank order; ``trip_table`` gains a ``pattern_index`` column holding
        the row position of each trip's pattern.
    """
    restrictions = trip_table["boarding"] if split_on_boarding else [()] * len(trip_table)
    keys = list(
        zip(
            trip_table["route_label"],
            trip_table["direction_id"],
            trip_table["path_key"],
            trip_table["stop_ids"],
            restrictions,
        )
    )
    positions: dict[tuple[Any, ...], list[int]] = {}
    for pos, key in enumerate(keys):
        positions.setdefault(key, []).append(pos)

    records: list[dict[str, Any]] = []
    for key, pos_list in positions.items():
        grp = trip_table.iloc[pos_list]
        label, direction, path_key, stop_ids, _ = key
        headsigns = grp.loc[grp["trip_headsign"] != "", "trip_headsign"]
        starts = grp["start_min"].dropna()
        first_trip = grp.sort_values(["start_min", "trip_id"], kind="mergesort").iloc[0]
        records.append(
            {
                "route_label": label,
                "route_ids": ", ".join(sorted(set(grp["route_id"]))),
                "route_long_name": grp["route_long_name"].iloc[0],
                "direction_id": direction,
                "path_key": path_key,
                "shape_ids": ", ".join(sorted(set(grp["shape_id"]) - {""})),
                "stop_ids": stop_ids,
                "boarding": Counter(grp["boarding"]).most_common(1)[0][0],
                "headsign": headsigns.mode().iloc[0] if not headsigns.empty else "",
                "n_trips": len(grp),
                "first_departure_min": starts.min() if not starts.empty else np.nan,
                "last_departure_min": starts.max() if not starts.empty else np.nan,
                "service_ids": ", ".join(sorted(set(grp["service_id"]))),
                "example_trip_id": first_trip["trip_id"],
                "_positions": pos_list,
            }
        )

    patterns = pd.DataFrame(records)
    patterns["n_stops"] = patterns["stop_ids"].map(len)
    patterns["_route_sort"] = patterns["route_label"].map(natural_sort_key)
    patterns["_stop_key"] = patterns["stop_ids"].map("|".join)
    patterns = patterns.sort_values(
        [
            "_route_sort",
            "route_label",
            "direction_id",
            "n_trips",
            "n_stops",
            "path_key",
            "_stop_key",
        ],
        ascending=[True, True, True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    patterns["pattern_rank"] = patterns.groupby(["route_label", "direction_id"]).cumcount() + 1
    patterns["pattern_id"] = "P" + patterns["pattern_rank"].astype(str)
    patterns["share_of_direction_trips"] = (
        patterns["n_trips"]
        / patterns.groupby(["route_label", "direction_id"])["n_trips"].transform("sum")
    ).round(3)

    pattern_index = np.empty(len(trip_table), dtype=int)
    for idx, pos_list in enumerate(patterns["_positions"]):
        pattern_index[pos_list] = idx
    trip_table = trip_table.assign(pattern_index=pattern_index)
    patterns = patterns.drop(columns=["_positions", "_route_sort", "_stop_key"])
    return patterns, trip_table


def project_stops(
    stops: pd.DataFrame, crs: Optional[str]
) -> tuple[dict[str, tuple[float, float]], CRS]:
    """Project stop coordinates into the working CRS.

    Args:
        stops: stops.txt (string columns).
        crs: Target projected CRS, or ``None`` to use the feed's UTM zone.

    Returns:
        Tuple of (stop_id → (x, y) in the projected CRS, the CRS used).

    Raises:
        ValueError: If stops.txt lacks coordinates, or the CRS is geographic.
    """
    if not {"stop_lat", "stop_lon"}.issubset(stops.columns):
        raise ValueError("stops.txt lacks stop_lat/stop_lon; cannot map patterns.")
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    valid = lat.notna() & lon.notna()
    if (~valid).any():
        logging.warning("%d stop(s) lack valid coordinates; not drawn.", int((~valid).sum()))
    points = gpd.GeoSeries(gpd.points_from_xy(lon[valid], lat[valid]), crs="EPSG:4326")
    target = CRS.from_user_input(crs) if crs else CRS.from_user_input(points.estimate_utm_crs())
    if not target.is_projected:
        raise ValueError(f"PROJECTED_CRS {crs!r} is not a projected CRS.")
    projected = points.to_crs(target)
    stop_ids = stops.loc[valid, "stop_id"].astype(str).str.strip()
    stop_xy = {sid: (pt.x, pt.y) for sid, pt in zip(stop_ids, projected)}
    return stop_xy, target


def crs_unit_to_feet(crs: CRS) -> float:
    """Return how many feet one unit of *crs*'s x axis spans."""
    return crs.axis_info[0].unit_conversion_factor / 0.3048


def orient_line_to_stops(
    line: LineString, stop_points: Sequence[tuple[float, float]]
) -> tuple[LineString, bool]:
    """Reverse *line* when its drawing order runs against the stop order.

    Each stop is projected onto the line; if more consecutive stop pairs move
    backward along the line than forward, the line is reversed. Counting pairs
    (rather than comparing only the first and last stop) keeps loops, whose
    terminals coincide, oriented correctly.

    Args:
        line: The pattern's shape, in the projected CRS.
        stop_points: The pattern's stops in travel order, same CRS.

    Returns:
        Tuple of (line in travel order, whether it was reversed).
    """
    if len(stop_points) < 2:
        return line, False
    along = np.array([line.project(Point(p)) for p in stop_points])
    steps = np.diff(along)
    if int((steps < 0).sum()) > int((steps > 0).sum()):
        return LineString(list(line.coords)[::-1]), True
    return line, False


def trim_line_to_stops(
    line: LineString, stop_points: Sequence[tuple[float, float]], slack: float
) -> tuple[LineString, bool]:
    """Cut *line* down to the stretch between the first and the last stop.

    A short-turn often reuses the full route's shape, and many shapes run on
    past the terminal to a layover point; trimming keeps each panel to the
    path riders actually travel. The cut is skipped when it would move any
    stop more than *slack* farther from the line, which guards loops and
    out-and-back routes where a terminal projects onto the wrong pass.

    Args:
        line: Pattern line already in travel order (projected CRS).
        stop_points: The pattern's stops in travel order, same CRS.
        slack: Allowed increase in a stop's distance to the line, and the
            minimum overhang worth trimming, in CRS units.

    Returns:
        Tuple of (possibly trimmed line, whether it was trimmed).
    """
    if len(stop_points) < 2:
        return line, False
    start = line.project(Point(stop_points[0]))
    end = line.project(Point(stop_points[-1]))
    if end <= start or (start <= slack and line.length - end <= slack):
        return line, False
    trimmed = substring(line, start, end)
    if not isinstance(trimmed, LineString) or trimmed.length <= 0:
        return line, False
    for p in stop_points:
        point = Point(p)
        if trimmed.distance(point) > line.distance(point) + slack:
            return line, False
    return trimmed, True


def build_pattern_geometries(
    patterns: pd.DataFrame,
    shape_lines: Mapping[str, LineString],
    stop_xy: Mapping[str, tuple[float, float]],
    crs: CRS,
) -> pd.DataFrame:
    """Attach a projected line, its source, and its length to every pattern.

    Args:
        patterns: Output of :func:`find_patterns`.
        shape_lines: Output of :func:`build_shape_lines` (WGS 84).
        stop_xy: Projected stop coordinates from :func:`project_stops`.
        crs: The projected CRS.

    Returns:
        *patterns* with ``geometry`` (LineString or ``None``),
        ``geometry_source`` ("shapes.txt", "stop-to-stop" or "none"),
        ``shape_reversed``, ``shape_trimmed_to_stops`` and ``length_mi``
        (first to last stop) columns.
    """
    keys = sorted({k for k in patterns["path_key"] if k and k in shape_lines})
    projected: dict[str, LineString] = {}
    if keys:
        series = gpd.GeoSeries([shape_lines[k] for k in keys], crs="EPSG:4326").to_crs(crs)
        projected = dict(zip(keys, series))
    to_miles = crs_unit_to_feet(crs) / FEET_PER_MILE
    slack = 50.0 / crs_unit_to_feet(crs)

    geoms: list[Optional[LineString]] = []
    sources: list[str] = []
    reversed_flags: list[bool] = []
    trimmed_flags: list[bool] = []
    reversed_shapes: set[str] = set()
    for row in patterns.itertuples(index=False):
        points = [stop_xy[s] for s in row.stop_ids if s in stop_xy]
        line = projected.get(row.path_key)
        if line is not None:
            line, flipped = orient_line_to_stops(line, points)
            if flipped:
                reversed_shapes.add(row.path_key)
            line, trimmed = trim_line_to_stops(line, points, slack)
            geoms.append(line)
            sources.append("shapes.txt")
            reversed_flags.append(flipped)
            trimmed_flags.append(trimmed)
            continue
        deduped = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
        if len(set(deduped)) >= 2:
            geoms.append(LineString(deduped))
            sources.append("stop-to-stop")
        else:
            geoms.append(None)
            sources.append("none")
        reversed_flags.append(False)
        trimmed_flags.append(False)

    if reversed_shapes:
        logging.info(
            "%d shape(s) are drawn opposite to their trips' stop order and were reversed "
            "for mapping: %s",
            len(reversed_shapes),
            ", ".join(sorted(reversed_shapes)),
        )
    patterns = patterns.copy()
    patterns["geometry"] = geoms
    patterns["geometry_source"] = sources
    patterns["shape_reversed"] = reversed_flags
    patterns["shape_trimmed_to_stops"] = trimmed_flags
    patterns["length_mi"] = [
        round(g.length * to_miles, 2) if g is not None else np.nan for g in geoms
    ]
    return patterns


def describe_difference(
    row: Mapping[str, Any],
    main: Mapping[str, Any],
    added: int,
    dropped: int,
    off_path_mi: float,
) -> str:
    """Summarize in a few words how a pattern differs from the main pattern."""
    parts: list[str] = []
    if added or dropped:
        counts = ([f"+{added}"] if added else []) + ([f"−{dropped}"] if dropped else [])
        parts.append(" / ".join(counts) + " stops")
    elif row["stop_ids"] != main["stop_ids"]:
        parts.append("same stops, different order")
    if row["geometry_source"] != main["geometry_source"]:
        if row["geometry_source"] != "shapes.txt":
            parts.append("no shape in feed (drawn stop to stop)")
        else:
            parts.append(f"{main['pattern_id']} has no shape in feed")
    elif not math.isnan(off_path_mi) and off_path_mi >= 0.05:
        parts.append(f"{off_path_mi:.1f} mi off {main['pattern_id']}'s path")
    elif row["path_key"] != main["path_key"]:
        parts.append("different shape, same streets")
    if row["stop_ids"] == main["stop_ids"] and row["boarding"] != main["boarding"]:
        parts.append("different pickup/drop-off rules")
    if not parts:
        return f"same stops and path as {main['pattern_id']}"
    return f"vs {main['pattern_id']}: " + "; ".join(parts)


def compare_with_main(
    patterns: pd.DataFrame, unit_to_ft: float, tolerance_ft: float
) -> pd.DataFrame:
    """Compare every pattern with its route/direction's main pattern (P1).

    Args:
        patterns: Output of :func:`build_pattern_geometries`.
        unit_to_ft: Feet per CRS unit, from :func:`crs_unit_to_feet`.
        tolerance_ft: Distance from P1's line within which a pattern still
            counts as on P1's streets.

    Returns:
        *patterns* with ``main_pattern_id``, ``stops_added_vs_main``,
        ``stops_dropped_vs_main``, ``added_stop_ids``, ``dropped_stop_ids``,
        ``off_main_path_mi`` and ``differs_from_main`` columns.
    """
    patterns = patterns.copy()
    columns: dict[str, list[Any]] = {
        "main_pattern_id": [],
        "stops_added_vs_main": [],
        "stops_dropped_vs_main": [],
        "added_stop_ids": [],
        "dropped_stop_ids": [],
        "off_main_path_mi": [],
        "differs_from_main": [],
    }
    mains = {
        (row["route_label"], row["direction_id"]): row
        for row in patterns[patterns["pattern_rank"] == 1].to_dict("records")
    }
    buffers: dict[tuple[str, str], Any] = {}
    for row in patterns.to_dict("records"):
        group = (row["route_label"], row["direction_id"])
        main = mains[group]
        columns["main_pattern_id"].append(main["pattern_id"])
        if row["pattern_rank"] == 1:
            columns["stops_added_vs_main"].append(0)
            columns["stops_dropped_vs_main"].append(0)
            columns["added_stop_ids"].append("")
            columns["dropped_stop_ids"].append("")
            columns["off_main_path_mi"].append(0.0)
            columns["differs_from_main"].append("main pattern (most trips)")
            continue

        own, ref = set(row["stop_ids"]), set(main["stop_ids"])
        added = [s for s in dict.fromkeys(row["stop_ids"]) if s not in ref]
        dropped = [s for s in dict.fromkeys(main["stop_ids"]) if s not in own]
        off_path_mi = np.nan
        comparable = row["geometry_source"] == main["geometry_source"]
        if comparable and row["geometry"] is not None and main["geometry"] is not None:
            if group not in buffers:
                buffers[group] = main["geometry"].buffer(tolerance_ft / unit_to_ft)
            stray = row["geometry"].difference(buffers[group])
            off_path_mi = round(stray.length * unit_to_ft / FEET_PER_MILE, 2)
        columns["stops_added_vs_main"].append(len(added))
        columns["stops_dropped_vs_main"].append(len(dropped))
        columns["added_stop_ids"].append("; ".join(added))
        columns["dropped_stop_ids"].append("; ".join(dropped))
        columns["off_main_path_mi"].append(off_path_mi)
        columns["differs_from_main"].append(
            describe_difference(row, main, len(added), len(dropped), off_path_mi)
        )
    for name, values in columns.items():
        patterns[name] = values
    return patterns


def resolve_day_types(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    day_types: Sequence[str],
) -> dict[str, tuple[dt.date, set[str]]]:
    """Pick a representative date, and its active service_ids, per day type.

    Args:
        calendar: Parsed calendar.txt or ``None``.
        calendar_dates: Parsed calendar_dates.txt or ``None``.
        day_types: Values such as ``"weekday"``, ``"saturday"``, ``"sunday"``.

    Returns:
        Mapping of day type to (date, service_ids). Day types with no
        candidate dates are omitted with a warning; the mapping is empty when
        the feed has no calendar files.
    """
    if not day_types:
        return {}
    if (calendar is None or calendar.empty) and (calendar_dates is None or calendar_dates.empty):
        logging.warning("No calendar.txt / calendar_dates.txt; reporting feed-wide trips only.")
        return {}
    active = expand_service_active_dates(calendar, calendar_dates)
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
            logging.warning("No representative %s date: %s", day_type, exc)
    return selection


def add_day_type_columns(
    patterns: pd.DataFrame,
    trip_table: pd.DataFrame,
    day_selection: Mapping[str, tuple[dt.date, set[str]]],
) -> pd.DataFrame:
    """Add per-day-type trip counts and first/last departures to each pattern.

    Args:
        patterns: Pattern table (row position = ``pattern_index``).
        trip_table: Trip table with ``pattern_index``, ``service_id`` and
            ``start_min``.
        day_selection: Output of :func:`resolve_day_types`.

    Returns:
        *patterns* with ``<day>_trips``, ``<day>_first_min`` and
        ``<day>_last_min`` columns for every resolved day type.
    """
    patterns = patterns.copy()
    for day_type, (_, service_ids) in day_selection.items():
        on_day = trip_table[trip_table["service_id"].isin(service_ids)]
        stats = on_day.groupby("pattern_index")["start_min"].agg(["size", "min", "max"])
        patterns[f"{day_type}_trips"] = (
            stats["size"].reindex(patterns.index).fillna(0).astype(int).to_numpy()
        )
        patterns[f"{day_type}_first_min"] = stats["min"].reindex(patterns.index).to_numpy()
        patterns[f"{day_type}_last_min"] = stats["max"].reindex(patterns.index).to_numpy()
    return patterns


def build_pattern_stops_table(patterns: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Expand every pattern into its ordered stop list.

    Args:
        patterns: Pattern table with ``stop_ids`` and ``boarding`` tuples.
        stops: stops.txt.

    Returns:
        One row per (pattern, stop) with ``stop_order`` starting at 1.
    """
    rows: list[dict[str, Any]] = []
    for row in patterns.itertuples(index=False):
        for order, (stop_id, status) in enumerate(zip(row.stop_ids, row.boarding), start=1):
            rows.append(
                {
                    "route": row.route_label,
                    "direction_id": row.direction_id,
                    "pattern_id": row.pattern_id,
                    "stop_order": order,
                    "stop_id": stop_id,
                    "boarding": status,
                }
            )
    table = pd.DataFrame(rows)
    stop_cols = [
        c for c in ("stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon") if c in stops
    ]
    attrs = stops[stop_cols].copy()
    attrs["stop_id"] = attrs["stop_id"].astype(str).str.strip()
    return table.merge(attrs.drop_duplicates("stop_id"), on="stop_id", how="left")


def arrow_segments(
    line: LineString, spacing: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return short (tail, head) point pairs along *line*, pointing in travel order.

    Arrows sit at ``spacing / 2``, ``3 * spacing / 2`` … along the line; a
    line shorter than *spacing* gets one arrow at its midpoint.

    Args:
        line: Pattern line in travel order.
        spacing: Distance between arrows, in the line's units.

    Returns:
        List of ((x_tail, y_tail), (x_head, y_head)) pairs.
    """
    length = line.length
    if length <= 0 or spacing <= 0:
        return []
    positions = [length / 2] if length < spacing else list(np.arange(spacing / 2, length, spacing))
    eps = min(spacing, length) * 0.02
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for s in positions:
        tail = line.interpolate(max(s - eps, 0.0))
        head = line.interpolate(min(s + eps, length))
        if tail.distance(head) > 0:
            segments.append(((tail.x, tail.y), (head.x, head.y)))
    return segments


def route_extent(
    geometries: Sequence[Optional[LineString]],
    points: Sequence[tuple[float, float]],
    min_aspect: float = 0.55,
    max_aspect: float = 1.6,
) -> tuple[float, float, float, float]:
    """Return a padded map extent covering every line and stop of a route.

    The extent is widened (never squeezed) so its height/width ratio stays
    within [*min_aspect*, *max_aspect*], keeping panels a readable shape.
    """
    xs: list[float] = [p[0] for p in points]
    ys: list[float] = [p[1] for p in points]
    for geom in geometries:
        if geom is not None:
            minx, miny, maxx, maxy = geom.bounds
            xs += [minx, maxx]
            ys += [miny, maxy]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    if height / width < min_aspect:
        height = width * min_aspect
    elif height / width > max_aspect:
        width = height / max_aspect
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    width, height = width * 1.08, height * 1.08
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def truncate(text: str, max_len: int) -> str:
    """Shorten *text* to *max_len* characters, marking the cut with an ellipsis."""
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def day_type_line(row: Mapping[str, Any], day_types: Sequence[str]) -> str:
    """Format the panel line with trips and first/last departures per day type."""
    if not day_types:
        span = (
            f" ({minutes_to_hhmm(row['first_departure_min'])}–"
            f"{minutes_to_hhmm(row['last_departure_min'])})"
            if not pd.isna(row["first_departure_min"])
            else ""
        )
        return f"{row['n_trips']} trips in feed{span}"
    if all(row[f"{d}_trips"] == 0 for d in day_types):
        return (
            f"Not on the representative days — {row['n_trips']} trip(s) in feed "
            f"(service: {truncate(row['service_ids'], 30)})"
        )
    parts: list[str] = []
    for day in day_types:
        n = row[f"{day}_trips"]
        abbrev = DAY_TYPE_ABBREV.get(day, day[:3].title())
        if n:
            first = minutes_to_hhmm(row[f"{day}_first_min"], missing="?")
            last = minutes_to_hhmm(row[f"{day}_last_min"], missing="?")
            parts.append(f"{abbrev} {n} ({first}–{last})")
        else:
            parts.append(f"{abbrev} —")
    return " · ".join(parts)


def draw_pattern_panel(
    ax: Axes,
    row: Mapping[str, Any],
    context: Sequence[LineString],
    extent: tuple[float, float, float, float],
    arrow_spacing: float,
    stop_xy: Mapping[str, tuple[float, float]],
    stop_names: Mapping[str, str],
    backdrop: Optional[gpd.GeoDataFrame],
    day_types: Sequence[str],
) -> None:
    """Draw one pattern's map panel and its header text.

    Args:
        ax: Axes to draw into.
        row: The pattern's row from the pattern table.
        context: Lines of the direction's other patterns (drawn gray).
        extent: Shared (xmin, ymin, xmax, ymax) of the route's panels.
        arrow_spacing: Distance between direction arrows.
        stop_xy: Projected stop coordinates.
        stop_names: stop_id → stop_name.
        backdrop: Optional street linework clipped to the route.
        day_types: Resolved day types (keys of the day selection).
    """
    if backdrop is not None and not backdrop.empty:
        backdrop.plot(ax=ax, color=BACKDROP_COLOR, linewidth=0.5, zorder=0)
    for other in context:
        x, y = other.xy
        ax.plot(x, y, color=CONTEXT_COLOR, linewidth=1.6, solid_capstyle="round", zorder=1)

    geom = row["geometry"]
    if geom is not None:
        x, y = geom.xy
        ax.plot(x, y, color=PATTERN_COLOR, linewidth=2.4, solid_capstyle="round", zorder=3)
        for tail, head in arrow_segments(geom, arrow_spacing):
            ax.annotate(
                "",
                xy=head,
                xytext=tail,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": PATTERN_COLOR,
                    "linewidth": 0.8,
                    "mutation_scale": ARROW_HEAD_SIZE,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "path_effects": [withStroke(linewidth=2.2, foreground="white")],
                },
                zorder=6,
            )

    by_status: dict[str, list[tuple[float, float]]] = {}
    for stop_id, status in zip(row["stop_ids"], row["boarding"]):
        if stop_id in stop_xy:
            by_status.setdefault(status, []).append(stop_xy[stop_id])
    regular = by_status.get("regular", [])
    restricted = by_status.get("pickup_only", []) + by_status.get("dropoff_only", [])
    no_service = by_status.get("no_service", [])
    if regular:
        ax.scatter(
            *zip(*regular), s=16, c=PATTERN_COLOR, edgecolors="white", linewidths=0.6, zorder=5
        )
    if restricted:
        ax.scatter(
            *zip(*restricted),
            s=18,
            facecolors="white",
            edgecolors=PATTERN_COLOR,
            linewidths=1.1,
            zorder=5,
        )
    if no_service:
        ax.scatter(
            *zip(*no_service), s=16, marker="x", c=NO_SERVICE_COLOR, linewidths=1.0, zorder=5
        )

    stop_ids = row["stop_ids"]
    first_id, last_id = stop_ids[0], stop_ids[-1]
    if first_id in stop_xy:
        ax.scatter(
            *stop_xy[first_id], s=70, marker="s", c=START_COLOR, edgecolors="white", zorder=6
        )
    if last_id in stop_xy:
        ax.scatter(*stop_xy[last_id], s=45, marker="o", c=END_COLOR, edgecolors="white", zorder=7)

    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#9e9e9e")
        spine.set_linewidth(0.6)

    length = "" if pd.isna(row["length_mi"]) else f" · {row['length_mi']:.1f} mi"
    headsign = f" · {truncate(row['headsign'], 34)}" if row["headsign"] else ""
    ax.set_title(
        f"{row['pattern_id']}{headsign} · {row['n_stops']} stops{length}",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=38,
    )
    first_name = truncate(stop_names.get(first_id, first_id), 32)
    last_name = truncate(stop_names.get(last_id, last_id), 32)
    details = [
        f"{first_name} → {last_name}",
        day_type_line(row, day_types),
        row["differs_from_main"]
        + (
            " · no shape: drawn stop to stop"
            if row["geometry_source"] != "shapes.txt" and "no shape" not in row["differs_from_main"]
            else ""
        ),
    ]
    ax.annotate(
        "\n".join(details),
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7.3,
        va="bottom",
        ha="left",
        linespacing=1.35,
    )


def stop_categories(rows: pd.DataFrame) -> set[str]:
    """Return the stop symbol categories ("regular", "restricted", "no_service") in *rows*."""
    statuses = {status for boarding in rows["boarding"] for status in boarding}
    categories = {"regular"} if "regular" in statuses else set()
    if statuses & {"pickup_only", "dropoff_only"}:
        categories.add("restricted")
    if "no_service" in statuses:
        categories.add("no_service")
    return categories


def legend_handles(categories: set[str], has_context: bool) -> list[Line2D]:
    """Build figure legend entries for the symbols that appear in a figure."""
    handles = [
        Line2D([], [], color=PATTERN_COLOR, linewidth=2.4, label="This pattern (arrows = travel)")
    ]
    if has_context:
        handles.append(
            Line2D(
                [], [], color=CONTEXT_COLOR, linewidth=1.6, label="Other patterns, same direction"
            )
        )
    if "regular" in categories:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=PATTERN_COLOR,
                markeredgecolor="white",
                markersize=5,
                label="Stop",
            )
        )
    if "restricted" in categories:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor=PATTERN_COLOR,
                markersize=5,
                label="Pickup-only or drop-off-only stop",
            )
        )
    if "no_service" in categories:
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="",
                color=NO_SERVICE_COLOR,
                markersize=5,
                label="Listed, no pickup or drop-off",
            )
        )
    handles.append(
        Line2D(
            [], [], marker="s", linestyle="", color=START_COLOR, markersize=7, label="First stop"
        )
    )
    handles.append(
        Line2D([], [], marker="o", linestyle="", color=END_COLOR, markersize=7, label="Last stop")
    )
    return handles


def day_type_footnote(day_selection: Mapping[str, tuple[dt.date, set[str]]]) -> str:
    """Return the footnote naming the representative date behind each day type."""
    if not day_selection:
        return "Trip counts cover every selected service_id in the feed."
    dates = " · ".join(
        f"{DAY_TYPE_ABBREV.get(day, day[:3].title())} = {date:%a %Y-%m-%d}"
        for day, (date, _) in day_selection.items()
    )
    return f"Trips and first–last departures on representative dates (holidays skipped): {dates}"


def plot_route_direction(
    rows: pd.DataFrame,
    extent: tuple[float, float, float, float],
    stop_xy: Mapping[str, tuple[float, float]],
    stop_names: Mapping[str, str],
    backdrop: Optional[gpd.GeoDataFrame],
    day_selection: Mapping[str, tuple[dt.date, set[str]]],
    out_dir: Path,
    base_name: str,
    max_panels: int,
    n_columns: int,
    dpi: int,
    arrows_across: int,
) -> list[tuple[Path, list[int]]]:
    """Write the figure(s) for one route and direction, one panel per pattern.

    Args:
        rows: The direction's patterns in rank order (index = pattern_index).
        extent: Shared map extent of the route.
        stop_xy: Projected stop coordinates.
        stop_names: stop_id → stop_name.
        backdrop: Optional street linework clipped to the route.
        day_selection: Output of :func:`resolve_day_types`.
        out_dir: Folder for the PNG files.
        base_name: File stem, e.g. ``route_10_dir0``.
        max_panels: Panels per figure before continuing on a new page.
        n_columns: Panels per row.
        dpi: PNG resolution.
        arrows_across: Arrows across the panel diagonal.

    Returns:
        List of (PNG path, pattern_index values drawn in it).
    """
    xmin, ymin, xmax, ymax = extent
    panel_h = PANEL_WIDTH_IN * (ymax - ymin) / (xmax - xmin)
    spacing = math.hypot(xmax - xmin, ymax - ymin) / max(arrows_across, 1)
    day_types = list(day_selection)
    first = rows.iloc[0]
    direction = first["direction_id"] or "(none)"
    title = f"Route {first['route_label']}"
    if first["route_long_name"]:
        title += f" — {first['route_long_name']}"
    title += f"\nDirection {direction} · {len(rows)} pattern(s)"
    footnote = (
        "Gray lines: this direction's other patterns. P1 = most trips. " if len(rows) > 1 else ""
    ) + day_type_footnote(day_selection)

    pages = [rows.iloc[i : i + max_panels] for i in range(0, len(rows), max_panels)]
    written: list[tuple[Path, list[int]]] = []
    for page_no, page in enumerate(pages, start=1):
        n = len(page)
        ncols = max(1, min(n_columns, n))
        nrows = math.ceil(n / ncols)
        side_in, gap_in, top_in = 0.15, 0.25, 0.4
        width = ncols * PANEL_WIDTH_IN + (ncols - 1) * gap_in + 2 * side_in
        handles = legend_handles(stop_categories(page), has_context=len(rows) > 1)
        legend_cols = min(len(handles), {1: 2, 2: 4}.get(ncols, 7))
        legend_in = 0.22 * math.ceil(len(handles) / legend_cols)
        note = textwrap.fill(footnote, width=int(width * 15))
        note_in = 0.14 * (note.count("\n") + 1)
        bottom_in = 0.2 + legend_in + note_in
        height = top_in + nrows * (panel_h + PANEL_HEADER_IN) + bottom_in
        fig = Figure(figsize=(width, height))
        axes = fig.subplots(nrows, ncols, squeeze=False)
        fig.subplots_adjust(
            left=side_in / width,
            right=1 - side_in / width,
            top=1 - (top_in + PANEL_HEADER_IN) / height,
            bottom=bottom_in / height,
            wspace=gap_in / PANEL_WIDTH_IN,
            hspace=PANEL_HEADER_IN / panel_h,
        )
        for ax, (idx, row) in zip(axes.flat, page.iterrows()):
            context = [g for j, g in rows["geometry"].items() if j != idx and g is not None]
            draw_pattern_panel(
                ax,
                row.to_dict(),
                context,
                extent,
                spacing,
                stop_xy,
                stop_names,
                backdrop,
                day_types,
            )
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)

        page_note = f" (page {page_no} of {len(pages)})" if len(pages) > 1 else ""
        fig.suptitle(
            title + page_note, x=side_in / width, ha="left", fontsize=12, fontweight="bold"
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, (bottom_in - 0.12) / height),
            ncol=legend_cols,
            fontsize=7.5,
            frameon=False,
        )
        fig.text(0.5, 0.05 / height, note, ha="center", va="bottom", fontsize=7, color="#555555")

        suffix = f"_p{page_no}" if len(pages) > 1 else ""
        path = out_dir / f"{base_name}{suffix}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        written.append((path, list(page.index)))
    return written


def load_backdrop(
    path: Optional[str], crs: CRS, zip_member: Optional[str] = None
) -> Optional[gpd.GeoDataFrame]:
    """Load optional street linework for panel backgrounds, projected to *crs*.

    Args:
        path: A .shp, a zipped shapefile, a .gpkg or a .geojson; ``None`` skips.
        crs: The projected CRS the maps are drawn in.
        zip_member: Shapefile to use when *path* is a zip holding several.

    Returns:
        The linework, or ``None`` (with a warning) when the layer cannot be
        used, so a bad backdrop never stops the maps from being drawn.
    """
    if not path:
        return None
    try:
        if path.lower().endswith((".shp", ".zip")):
            layer = load_shapefile(path, member=zip_member or None)
        else:
            layer = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 - any read failure just drops the backdrop
        logging.warning("Backdrop '%s' could not be read (%s); drawing without it.", path, exc)
        return None
    if layer.empty or layer.crs is None:
        logging.warning("Backdrop '%s' is empty or has no CRS; drawing without it.", path)
        return None
    layer = layer[layer.geometry.notna()].to_crs(crs)
    logging.info("Loaded backdrop with %d feature(s).", len(layer))
    return layer[["geometry"]]


def clip_backdrop(
    backdrop: Optional[gpd.GeoDataFrame], extent: tuple[float, float, float, float]
) -> Optional[gpd.GeoDataFrame]:
    """Return the backdrop features inside *extent* (``None`` when there are none)."""
    if backdrop is None:
        return None
    window = box(*extent)
    subset = backdrop[backdrop.intersects(window)]
    if subset.empty:
        return None
    return gpd.clip(subset, window)


def unique_base_names(groups: Sequence[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Map each (route_label, direction_id) to a unique, filesystem-safe file stem.

    Stems are deduplicated case-insensitively so they stay unique on Windows.
    """
    used: set[str] = set()
    names: dict[tuple[str, str], str] = {}
    for label, direction in groups:
        stem = f"route_{sanitize_filename_component(label)}_dir"
        stem += sanitize_filename_component(direction) if direction else "none"
        candidate, n = stem, 2
        while candidate.lower() in used:
            candidate, n = f"{stem}_{n}", n + 1
        used.add(candidate.lower())
        names[(label, direction)] = candidate
    return names


def build_summary_table(
    patterns: pd.DataFrame,
    stop_names: Mapping[str, str],
    day_types: Sequence[str],
) -> pd.DataFrame:
    """Format the one-row-per-pattern summary written to SUMMARY_FILENAME."""
    out = pd.DataFrame(
        {
            "route": patterns["route_label"],
            "route_ids": patterns["route_ids"],
            "route_long_name": patterns["route_long_name"],
            "direction_id": patterns["direction_id"],
            "pattern_id": patterns["pattern_id"],
            "headsign": patterns["headsign"],
            "n_trips": patterns["n_trips"],
            "share_of_direction_trips": patterns["share_of_direction_trips"],
        }
    )
    for day in day_types:
        out[f"{day}_trips"] = patterns[f"{day}_trips"]
        out[f"{day}_first_departure"] = patterns[f"{day}_first_min"].map(minutes_to_hhmm)
        out[f"{day}_last_departure"] = patterns[f"{day}_last_min"].map(minutes_to_hhmm)
    out["first_departure"] = patterns["first_departure_min"].map(minutes_to_hhmm)
    out["last_departure"] = patterns["last_departure_min"].map(minutes_to_hhmm)
    out["n_stops"] = patterns["n_stops"]
    out["first_stop_id"] = patterns["stop_ids"].map(lambda s: s[0])
    out["first_stop_name"] = out["first_stop_id"].map(stop_names)
    out["last_stop_id"] = patterns["stop_ids"].map(lambda s: s[-1])
    out["last_stop_name"] = out["last_stop_id"].map(stop_names)
    out["length_mi"] = patterns["length_mi"]
    for col in (
        "geometry_source",
        "shape_ids",
        "shape_reversed",
        "shape_trimmed_to_stops",
        "main_pattern_id",
        "differs_from_main",
        "stops_added_vs_main",
        "stops_dropped_vs_main",
        "added_stop_ids",
        "dropped_stop_ids",
        "off_main_path_mi",
    ):
        out[col] = patterns[col]
    out["restricted_stops"] = [
        "; ".join(f"{s} ({b})" for s, b in zip(ids, statuses) if b != "regular")
        for ids, statuses in zip(patterns["stop_ids"], patterns["boarding"])
    ]
    out["service_ids"] = patterns["service_ids"]
    out["example_trip_id"] = patterns["example_trip_id"]
    out["map_file"] = patterns["map_file"]
    return out


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
        "PATTERN MAP EXPORTER RUN LOG",
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
    route_names: Optional[Sequence[str]] = None,
    route_names_file: Optional[str] = None,
    service_ids: Optional[Sequence[str]] = None,
    day_types: Optional[Sequence[str]] = None,
    split_on_boarding: Optional[bool] = None,
    projected_crs: Optional[str] = None,
    backdrop_path: Optional[str] = None,
    backdrop_zip_member: Optional[str] = None,
    off_path_tolerance_ft: Optional[float] = None,
    max_panels: Optional[int] = None,
    panel_columns: Optional[int] = None,
    dpi: Optional[int] = None,
    arrows_across: Optional[int] = None,
) -> pd.DataFrame:
    """Find every route pattern, draw the panel maps, and write the CSVs.

    Unset args fall back to the CONFIGURATION block, so ``m.GTFS_PATH = ...;
    m.run()`` works after a plain import.

    Returns:
        The pattern summary table (as written to SUMMARY_FILENAME).

    Raises:
        OSError: If the feed is missing files, or the run log is required but
            cannot be written.
        ValueError: If a setting is invalid, no route name matches, or no
            trips remain to map.
        ServiceSelectionError: If a requested service_id is not in trips.txt.
    """
    gtfs_path = GTFS_PATH if gtfs_path is None else Path(gtfs_path)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    route_names = ROUTE_NAMES if route_names is None else route_names
    route_names_file = ROUTE_NAMES_FILE if route_names_file is None else route_names_file
    service_ids = SERVICE_IDS_TO_INCLUDE if service_ids is None else service_ids
    day_types = DAY_TYPES if day_types is None else day_types
    split_on_boarding = (
        SPLIT_ON_BOARDING_RESTRICTIONS if split_on_boarding is None else split_on_boarding
    )
    projected_crs = PROJECTED_CRS if projected_crs is None else (projected_crs or None)
    backdrop_path = BACKDROP_PATH if backdrop_path is None else (backdrop_path or None)
    backdrop_zip_member = (
        BACKDROP_ZIP_MEMBER if backdrop_zip_member is None else (backdrop_zip_member or None)
    )
    tolerance_ft = OFF_PATH_TOLERANCE_FT if off_path_tolerance_ft is None else off_path_tolerance_ft
    max_panels = MAX_PANELS_PER_FIGURE if max_panels is None else max_panels
    panel_columns = PANEL_COLUMNS if panel_columns is None else panel_columns
    dpi = FIGURE_DPI if dpi is None else dpi
    arrows_across = ARROWS_ACROSS_PANEL if arrows_across is None else arrows_across

    day_types = [str(d).strip().lower() for d in day_types if str(d).strip()]
    unknown_days = [d for d in day_types if d not in DAY_TYPE_ABBREV]
    if unknown_days:
        raise ValueError(
            f"DAY_TYPES {unknown_days} not recognised; use 'weekday' or a day name "
            "('monday' … 'sunday')."
        )
    if max_panels < 1 or panel_columns < 1:
        raise ValueError("MAX_PANELS_PER_FIGURE and PANEL_COLUMNS must be at least 1.")

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
    routes, trips, stop_times, stops = (
        gtfs["routes"],
        gtfs["trips"],
        gtfs["stop_times"],
        gtfs["stops"],
    )
    shapes = load_optional_table(gtfs_path, "shapes.txt")
    calendar = load_optional_table(gtfs_path, "calendar.txt")
    calendar_dates = load_optional_table(gtfs_path, "calendar_dates.txt")
    frequencies = load_optional_table(gtfs_path, "frequencies.txt")

    names = load_id_set(route_names, route_names_file or None, kind="route")
    unmatched: set[str] = set()
    if names:
        target_route_ids, unmatched = identify_target_route_ids(routes, names)
        if unmatched:
            logging.warning(
                "%d route name(s) matched no route_short_name or route_id and were skipped: %s",
                len(unmatched),
                ", ".join(sorted(unmatched)),
            )
        if not target_route_ids:
            raise ValueError(
                f"None of the route names {sorted(names)} matched a route_short_name or "
                "route_id in routes.txt."
            )
        trips = trips[trips["route_id"].astype(str).str.strip().isin(target_route_ids)]
    else:
        logging.info("No route names given; mapping every route in the feed.")

    if service_ids:
        trips = select_trips_by_service(trips, service_ids)
    if trips.empty:
        raise ValueError("No trips remain after the route and service filters.")
    if frequencies is not None and not frequencies.empty and "trip_id" in frequencies.columns:
        if trips["trip_id"].isin(set(frequencies["trip_id"])).any():
            logging.warning(
                "Some trips are headway-based (frequencies.txt); each counts as one trip here."
            )

    shape_lines = build_shape_lines(shapes)
    shape_canon = group_identical_shapes(shape_lines)
    trip_table = build_trip_table(trips, stop_times, routes, shape_canon)
    if trip_table.empty:
        raise ValueError("None of the selected trips has stop_times rows.")
    patterns, trip_table = find_patterns(trip_table, split_on_boarding)

    stop_xy, crs = project_stops(stops, projected_crs)
    logging.info("Drawing in %s (%s).", crs.name, crs.to_string())
    unit_to_ft = crs_unit_to_feet(crs)
    patterns = build_pattern_geometries(patterns, shape_lines, stop_xy, crs)
    patterns = compare_with_main(patterns, unit_to_ft, tolerance_ft)

    day_selection = resolve_day_types(calendar, calendar_dates, day_types)
    patterns = add_day_type_columns(patterns, trip_table, day_selection)

    stop_names = dict(
        zip(
            stops["stop_id"].astype(str).str.strip(),
            stops["stop_name"].fillna("").astype(str).str.strip()
            if "stop_name" in stops.columns
            else stops["stop_id"].astype(str),
        )
    )
    backdrop = load_backdrop(backdrop_path, crs, backdrop_zip_member)

    maps_dir = output_dir / MAPS_SUBDIR
    maps_dir.mkdir(parents=True, exist_ok=True)
    groups = list(dict.fromkeys(zip(patterns["route_label"], patterns["direction_id"])))
    base_names = unique_base_names(groups)
    map_files: dict[int, str] = {}
    n_figures = 0
    for label, route_rows in patterns.groupby("route_label", sort=False):
        points = [stop_xy[s] for ids in route_rows["stop_ids"] for s in ids if s in stop_xy]
        if not points and route_rows["geometry"].isna().all():
            logging.warning("Route %s has no mappable geometry or stops; skipped.", label)
            continue
        extent = route_extent(list(route_rows["geometry"]), points)
        route_backdrop = clip_backdrop(backdrop, extent)
        for direction, rows in route_rows.groupby("direction_id", sort=False):
            written = plot_route_direction(
                rows,
                extent,
                stop_xy,
                stop_names,
                route_backdrop,
                day_selection,
                maps_dir,
                base_names[(str(label), str(direction))],
                max_panels,
                panel_columns,
                dpi,
                arrows_across,
            )
            for path, indexes in written:
                for idx in indexes:
                    map_files[idx] = str(Path(MAPS_SUBDIR) / path.name)
            n_figures += len(written)
            logging.info(
                "Route %s direction %s: %d pattern(s) — %s",
                label,
                direction or "(none)",
                len(rows),
                ", ".join(f"{p} {n} trips" for p, n in zip(rows["pattern_id"], rows["n_trips"])),
            )
    patterns["map_file"] = [map_files.get(i, "") for i in range(len(patterns))]

    summary = build_summary_table(patterns, stop_names, list(day_selection))
    stop_table = build_pattern_stops_table(patterns, stops)
    summary_path = output_dir / SUMMARY_FILENAME
    stops_path = output_dir / STOPS_FILENAME
    summary.to_csv(summary_path, index=False)
    stop_table.to_csv(stops_path, index=False)
    logging.info("Wrote %d pattern(s) → %s", len(summary), summary_path)
    logging.info("Wrote %d pattern stop row(s) → %s", len(stop_table), stops_path)

    day_lines = [
        f"  {day}: {date.isoformat()} ({date:%A}) — service_ids {', '.join(sorted(ids)) or '-'}"
        for day, (date, ids) in day_selection.items()
    ] or ["  (none — feed-wide trip counts only)"]
    summary_lines = [
        f"GTFS feed:            {gtfs_path}",
        f"Route names:          {', '.join(sorted(names)) or 'all routes'}",
        f"Unmatched names:      {', '.join(sorted(unmatched)) or '-'}",
        f"Service filter:       {', '.join(service_ids) or 'all service_ids'}",
        f"Split on boarding:    {split_on_boarding}",
        f"Projected CRS:        {crs.to_string()} ({crs.name})",
        f"Backdrop:             {backdrop_path or '-'}"
        + (f" ({backdrop_zip_member})" if backdrop_zip_member else ""),
        f"Routes mapped:        {patterns['route_label'].nunique()}",
        f"Patterns found:       {len(patterns)}",
        f"Figures written:      {n_figures} in {maps_dir}",
        f"Summary CSV:          {summary_path}",
        f"Pattern stops CSV:    {stops_path}",
        "Representative dates:",
        *day_lines,
    ]
    if not write_run_log(output_dir, summary_lines, config_text, source_label) and REQUIRE_RUN_LOG:
        raise OSError(
            f"Run log could not be written to '{output_dir}' and REQUIRE_RUN_LOG is True."
        )

    logging.info(
        "Pattern map export complete — %d pattern(s) across %d route(s), %d figure(s).",
        len(patterns),
        patterns["route_label"].nunique(),
        n_figures,
    )
    return summary


# =============================================================================
# MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; every flag defaults to its CONFIGURATION constant."""
    parser = argparse.ArgumentParser(
        description=(
            "Draw every distinct stop/path pattern of each GTFS route, one map panel per "
            "pattern, with direction arrows and stops. Defaults come from the configuration "
            "block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-path", type=Path, default=GTFS_PATH, help="GTFS feed folder or .zip archive."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for maps, CSVs and run log."
    )
    parser.add_argument(
        "--routes",
        nargs="*",
        default=ROUTE_NAMES,
        metavar="ROUTE",
        help="Route short names and/or route_ids to map (empty = all routes).",
    )
    parser.add_argument(
        "--routes-file",
        default=ROUTE_NAMES_FILE,
        help="Text file of route names, one per line ('#' comments allowed).",
    )
    parser.add_argument(
        "--service-ids",
        nargs="*",
        default=SERVICE_IDS_TO_INCLUDE,
        metavar="SERVICE_ID",
        help="Restrict to these service_ids from trips.txt (empty = all).",
    )
    parser.add_argument(
        "--day-types",
        nargs="*",
        default=DAY_TYPES,
        metavar="DAY",
        help="Day types for trip counts: 'weekday' or day names (empty = feed-wide only).",
    )
    parser.add_argument(
        "--split-on-boarding-restrictions",
        action=argparse.BooleanOptionalAction,
        default=SPLIT_ON_BOARDING_RESTRICTIONS,
        help="Treat pickup/drop-off restriction differences as separate patterns.",
    )
    parser.add_argument(
        "--projected-crs",
        default=PROJECTED_CRS,
        help="Projected CRS for drawing; pass '' to use the feed's UTM zone.",
    )
    parser.add_argument(
        "--backdrop-path", default=BACKDROP_PATH, help="Optional street linework for panels."
    )
    parser.add_argument(
        "--backdrop-zip-member",
        default=BACKDROP_ZIP_MEMBER,
        help="Shapefile to use when --backdrop-path is a zip holding several.",
    )
    parser.add_argument(
        "--off-path-tolerance-ft",
        type=float,
        default=OFF_PATH_TOLERANCE_FT,
        help="Distance from P1's line still counted as the same streets.",
    )
    parser.add_argument(
        "--max-panels-per-figure", type=int, default=MAX_PANELS_PER_FIGURE, help="Panels per PNG."
    )
    parser.add_argument("--panel-columns", type=int, default=PANEL_COLUMNS, help="Panels per row.")
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="PNG resolution.")
    parser.add_argument(
        "--arrows-across-panel",
        type=int,
        default=ARROWS_ACROSS_PANEL,
        help="Approximate number of direction arrows across a panel's diagonal.",
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
        CONFIGURATION values are still placeholders or a service_id is unknown.
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
            route_names=args.routes,
            route_names_file=args.routes_file,
            service_ids=args.service_ids,
            day_types=args.day_types,
            split_on_boarding=args.split_on_boarding_restrictions,
            projected_crs=args.projected_crs,
            backdrop_path=args.backdrop_path,
            backdrop_zip_member=args.backdrop_zip_member,
            off_path_tolerance_ft=args.off_path_tolerance_ft,
            max_panels=args.max_panels_per_figure,
            panel_columns=args.panel_columns,
            dpi=args.dpi,
            arrows_across=args.arrows_across_panel,
        )
    except ServiceSelectionError as exc:
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
