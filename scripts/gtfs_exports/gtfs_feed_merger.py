"""Merge several agencies' GTFS feeds into one valid GTFS feed, combining duplicate stops.

Built for regional work where riders move between systems, e.g. Fairfax
Connector + WMATA Metrobus + Montgomery County Ride On. Feeds are listed in
priority order and the first is *dominant*: wherever two feeds collide, the
higher-priority feed keeps its IDs and route names, and the lower-priority feed
is renamed (IDs, and route names another agency already uses, which become
``"<name>_<agency_name>"``) or merged into it (duplicate stops).

A lower-priority street stop merges into a higher-priority stop within
``STOP_MATCH_DISTANCE_FT`` (default 50 ft) when the buses at both travel the
same way (``HEADING_TOLERANCE_DEG``), so stops across the street or around the
corner stay separate. Each stop absorbs at most one stop per feed, closest
pairs first. Bus bays (stops with a ``parent_station``, a name matching
``BAY_NAME_PATTERNS``, or inside an optional transit-center polygon layer) sit
closer together than street stops, so they use ``BAY_MATCH_DISTANCE_FT``
(default 0 = never merged automatically) and are listed for review instead.

Each feed keeps its own service_ids. By default every calendar is clipped to
the date window all feeds cover and trips left with no dates are dropped, so
one agency's current schedule is never paired with another's expired one.

Inputs
------
- Two or more GTFS feeds (folders or .zip), each with agency.txt, stops.txt,
  routes.txt, trips.txt, stop_times.txt, and calendar.txt and/or
  calendar_dates.txt.
- Optional overrides CSV with columns ``feed, stop_id, target_feed,
  target_stop_id`` (blank target = keep the stop separate).
- Optional transit-center polygon layer, e.g. the zones drawn for
  ``cluster_stops_from_zones_gpd.py`` (requires geopandas).

Outputs
-------
- ``merged_gtfs/`` and ``merged_gtfs.zip``: agency, stops, routes, trips,
  stop_times, calendar, calendar_dates, shapes, frequencies, transfers,
  levels, pathways, and a new feed_info.txt. Fares and other files are not
  carried over; each dropped file is logged.
- ``merge_stop_matches.csv``: every cross-feed stop pair within the match
  distance, with distance, heading difference, name similarity and outcome.
- ``merge_id_crosswalk.csv``: original to merged value of every ID and
  renamed route name in every feed.
- ``merge_feed_summary.csv``: one row per input feed.
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI
flags) and run from a shell or a Jupyter notebook. Review
``merge_stop_matches.csv``, add overrides where needed, and rerun.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import logging
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import IO, Any, List, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS feeds to merge (folders or .zip), in PRIORITY ORDER: the first feed is
# dominant. Where feeds collide, the higher-priority feed keeps its IDs.
GTFS_FEEDS: list[str] = [
    r"Path\To\Dominant_Agency_GTFS",  # e.g. Fairfax Connector
    r"Path\To\Second_Agency_GTFS",  # e.g. WMATA Metrobus
    # r"Path\To\Third_Agency_GTFS",  # e.g. Montgomery County Ride On
]

# Short label per feed, parallel to GTFS_FEEDS, e.g. ["FFX", "WMATA", "RO"].
# Labels prefix renamed IDs ("WMATA_1001") and identify feeds in the reports and
# the overrides CSV. Empty = each feed's folder or .zip name.
FEED_LABELS: list[str] = []

OUTPUT_DIR: str = r"Path\To\Output_Folder"
MERGED_FEED_NAME: str = r"merged_gtfs"  # folder (and .zip) written in OUTPUT_DIR
WRITE_ZIP: bool = True

# --- ID collisions ------------------------------------------------------------
# "on_collision": a lower-priority ID is prefixed with its feed label only when
#   a higher-priority feed already uses it (route "1" in two feeds -> "1" and
#   "RO_1"). Unique IDs keep their original values.
# "always": every lower-priority ID is prefixed, so IDs do not change when a
#   later feed update happens to create or remove a collision. agency_ids are
#   only ever renamed on collision.
ID_PREFIX_MODE: str = "on_collision"
ID_PREFIX_SEPARATOR: str = "_"

# --- Duplicate street stops ---------------------------------------------------
# A lower-priority stop merges into a higher-priority stop within this distance.
# 0 = never merge street stops automatically.
STOP_MATCH_DISTANCE_FT: float = 50.0

# Two street stops only merge when some bus at one travels within this many
# degrees of some bus at the other (headings run from the previous stop to the
# next stop of each trip). Stops across the street (~180 deg) or around the
# corner (~90 deg) stay separate, and so do stops with no scheduled trips (no
# direction to compare). Bays are matched on distance alone. 180 = no check.
HEADING_TOLERANCE_DEG: float = 60.0

# --- Bus bays -----------------------------------------------------------------
# A stop is a bay when it has a parent_station, its stop_name matches one of
# these case-insensitive regular expressions, or it lies inside a polygon of
# BAY_ZONES_PATH. Bays merge only within BAY_MATCH_DISTANCE_FT; bay pairs
# within STOP_MATCH_DISTANCE_FT are listed for review either way.
BAY_NAME_PATTERNS: list[str] = [r"\bbay\b"]
BAY_ZONES_PATH: str = r""  # optional polygon layer, e.g. one polygon per transit center
BAY_MATCH_DISTANCE_FT: float = 0.0  # 0 = never merge bays automatically

# --- Route names --------------------------------------------------------------
# When agencies share a route name (route_short_name, or route_long_name for
# routes without one), the name is suffixed with the agency_name, e.g. "10" and
# "10_WMATA", plus "_2", "_3"... if that is taken too. Every shared name is
# logged as a warning and each rename is listed in the ID crosswalk.
# "lower_priority": the highest-priority agency keeps the plain name.
# "all": every agency sharing the name is renamed.
# "keep": names stay as published (warning only).
DUPLICATE_ROUTE_NAMES: str = "lower_priority"

# Optional CSV of manual stop decisions, applied before automatic matching:
#   feed, stop_id, target_feed, target_stop_id
# feed/stop_id name a lower-priority stop (original IDs). A target_feed and
# target_stop_id from a higher-priority feed merge it there; a blank target
# keeps it separate.
STOP_OVERRIDES_PATH: str = r""

# --- Service calendars --------------------------------------------------------
# True: clip every calendar to the date window all feeds cover and drop trips
# left with no service dates. False: keep each feed's full calendar.
CLIP_TO_COMMON_WINDOW: bool = True
# Optional window override as YYYYMMDD (used only when clipping). Blank = the
# common window derived from the feeds.
WINDOW_START: str = ""
WINDOW_END: str = ""

# --- feed_info.txt ------------------------------------------------------------
# Publisher of the merged feed. Blank = the dominant feed's publisher (or agency).
FEED_PUBLISHER_NAME: str = ""
FEED_PUBLISHER_URL: str = ""

# --- Reports ------------------------------------------------------------------
STOP_MATCHES_FILENAME: str = r"merge_stop_matches.csv"
CROSSWALK_FILENAME: str = r"merge_id_crosswalk.csv"
FEED_SUMMARY_FILENAME: str = r"merge_feed_summary.csv"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output is never
# left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

_PLACEHOLDER_PREFIX = "Path\\To\\"

# Mean Earth radius (6,371,008.8 m) in feet.
EARTH_RADIUS_FT: float = 6_371_008.8 / 0.3048

# GTFS tables carried into the merged feed, in output order. feed_info.txt is
# read from each feed but rebuilt for the merged feed.
CARRIED_TABLES: tuple[str, ...] = (
    "agency",
    "stops",
    "routes",
    "trips",
    "stop_times",
    "calendar",
    "calendar_dates",
    "shapes",
    "frequencies",
    "transfers",
    "levels",
    "pathways",
)
READ_TABLES: tuple[str, ...] = (*CARRIED_TABLES, "feed_info")
REQUIRED_TABLES: tuple[str, ...] = ("agency", "stops", "routes", "trips", "stop_times")

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "agency": ("agency_name", "agency_timezone"),
    "stops": ("stop_id",),
    "routes": ("route_id",),
    "trips": ("route_id", "service_id", "trip_id"),
    "stop_times": ("trip_id", "stop_id", "stop_sequence"),
    "calendar": ("service_id", "start_date", "end_date"),
    "calendar_dates": ("service_id", "date", "exception_type"),
}

# Every column that holds an ID, by table, with the ID type it belongs to. IDs
# only collide within one type: route "10" and stop "10" are unrelated.
ID_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "agency": (("agency_id", "agency"),),
    "stops": (("stop_id", "stop"), ("parent_station", "stop"), ("level_id", "level")),
    "routes": (("route_id", "route"), ("agency_id", "agency")),
    "trips": (
        ("trip_id", "trip"),
        ("route_id", "route"),
        ("service_id", "service"),
        ("shape_id", "shape"),
        ("block_id", "block"),
    ),
    "stop_times": (("trip_id", "trip"), ("stop_id", "stop")),
    "calendar": (("service_id", "service"),),
    "calendar_dates": (("service_id", "service"),),
    "shapes": (("shape_id", "shape"),),
    "frequencies": (("trip_id", "trip"),),
    "transfers": (
        ("from_stop_id", "stop"),
        ("to_stop_id", "stop"),
        ("from_route_id", "route"),
        ("to_route_id", "route"),
        ("from_trip_id", "trip"),
        ("to_trip_id", "trip"),
    ),
    "levels": (("level_id", "level"),),
    "pathways": (("pathway_id", "pathway"), ("from_stop_id", "stop"), ("to_stop_id", "stop")),
}

TRANSFER_KEY: tuple[str, ...] = (
    "from_stop_id",
    "to_stop_id",
    "from_route_id",
    "to_route_id",
    "from_trip_id",
    "to_trip_id",
)

STOP_MATCH_COLUMNS: list[str] = [
    "feed",
    "stop_id",
    "stop_name",
    "target_feed",
    "target_stop_id",
    "target_stop_name",
    "merged_stop_id",
    "distance_ft",
    "heading_diff_deg",
    "bay",
    "name_similarity",
    "outcome",
]

_NO_HEADINGS = np.empty(0)

# =============================================================================
# FUNCTIONS
# =============================================================================


class MergeError(ValueError):
    """An input problem that stops the merge, with an actionable message."""


class MergeSettings(NamedTuple):
    """Matching, ID and calendar settings for one merge run."""

    id_prefix_mode: str = ID_PREFIX_MODE
    id_prefix_separator: str = ID_PREFIX_SEPARATOR
    duplicate_route_names: str = DUPLICATE_ROUTE_NAMES
    stop_match_distance_ft: float = STOP_MATCH_DISTANCE_FT
    heading_tolerance_deg: float = HEADING_TOLERANCE_DEG
    bay_name_patterns: tuple[str, ...] = tuple(BAY_NAME_PATTERNS)
    bay_match_distance_ft: float = BAY_MATCH_DISTANCE_FT
    clip_to_common_window: bool = CLIP_TO_COMMON_WINDOW
    window_start: str = WINDOW_START
    window_end: str = WINDOW_END
    feed_publisher_name: str = FEED_PUBLISHER_NAME
    feed_publisher_url: str = FEED_PUBLISHER_URL


class Feed(NamedTuple):
    """One input feed: its label, source path, GTFS tables and uncarried files.

    ``tables`` is a mutable dict; the merge replaces its entries in place.
    """

    label: str
    path: str
    tables: dict[str, pd.DataFrame]
    dropped_files: list[str]


class MergeResult(NamedTuple):
    """Everything one merge produces."""

    tables: dict[str, pd.DataFrame]
    stop_matches: pd.DataFrame
    crosswalk: pd.DataFrame
    feed_summary: pd.DataFrame
    window: Optional[tuple[dt.date, dt.date]]


class _StopPool:
    """Platform stops already in the merged feed, which later feeds match against.

    Parallel lists, one entry per stop: merged stop_id, source feed label,
    original stop_id, stop_name, Earth-centred x/y/z (feet), headings, bay flag.
    """

    def __init__(self) -> None:
        """Start an empty pool."""
        self.ids: list[str] = []
        self.feeds: list[str] = []
        self.orig_ids: list[str] = []
        self.names: list[str] = []
        self.xyz: list[np.ndarray] = []
        self.headings: list[np.ndarray] = []
        self.bay: list[bool] = []


# -----------------------------------------------------------------------------
# Reading feeds
# -----------------------------------------------------------------------------


def _read_gtfs_csv(source: Union[str, Path, IO[bytes]], label: str) -> pd.DataFrame:
    """Read one GTFS text file with every value kept as text.

    Blank cells stay ``""`` (not NaN) and values such as ``"NA"`` stay text, so
    the merged feed reproduces the source values exactly.

    Args:
        source: File path or open binary handle.
        label: Name used in error messages.

    Returns:
        The table, with column names stripped of whitespace. An empty file
        gives an empty table.

    Raises:
        ValueError: The file cannot be parsed as CSV.
    """
    try:
        df = pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            encoding="utf-8-sig",
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse {label}: {exc}") from exc
    df.columns = [str(col).strip() for col in df.columns]
    return df


def read_gtfs_tables(gtfs_path: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read every GTFS table this script carries from a feed folder or .zip.

    Zip members may sit at the archive root or one folder down. macOS
    resource-fork entries (``__MACOSX/``, ``._*``) are ignored.

    Args:
        gtfs_path: Feed folder or .zip archive.

    Returns:
        A tuple ``(tables, other_files)``: tables keyed by file stem
        (``"stops"``), and the names of any other .txt/.geojson files in the
        feed, which are not carried into the merged feed.

    Raises:
        OSError: The path does not exist.
        ValueError: The path is not a folder or valid .zip, a file appears in
            more than one zip folder, or a file cannot be parsed.
    """
    path = Path(gtfs_path)
    if not path.exists():
        raise OSError(f"GTFS feed not found: {gtfs_path}")
    tables: dict[str, pd.DataFrame] = {}
    other_files: list[str] = []

    def _is_data_file(name: str) -> bool:
        return name.lower().endswith((".txt", ".geojson")) and not name.startswith("._")

    if path.is_dir():
        for file in sorted(path.iterdir()):
            if not file.is_file() or not _is_data_file(file.name):
                continue
            if file.suffix.lower() == ".txt" and file.stem in READ_TABLES:
                tables[file.stem] = _read_gtfs_csv(file, f"{gtfs_path}/{file.name}")
            else:
                other_files.append(file.name)
        return tables, other_files

    if path.suffix.lower() != ".zip":
        raise ValueError(f"'{gtfs_path}' is neither a folder nor a .zip file.")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{gtfs_path}' is not a valid zip archive.") from exc
    with archive:
        members: dict[str, list[str]] = {}
        for name in archive.namelist():
            base = name.rsplit("/", 1)[-1]
            if name.startswith("__MACOSX/") or not base or not _is_data_file(base):
                continue
            members.setdefault(base, []).append(name)
        for base, names in sorted(members.items()):
            stem = base.rsplit(".", 1)[0]
            if not base.lower().endswith(".txt") or stem not in READ_TABLES:
                other_files.append(base)
                continue
            if len(names) > 1:
                raise ValueError(f"'{gtfs_path}' holds {base} in more than one folder: {names}")
            with archive.open(names[0]) as handle:
                tables[stem] = _read_gtfs_csv(handle, f"{gtfs_path}:{base}")
    return tables, other_files


def validate_feed_tables(label: str, tables: Mapping[str, pd.DataFrame]) -> None:
    """Check that a feed has the files and columns the merge needs.

    Args:
        label: Feed label, for messages.
        tables: The feed's tables from :func:`read_gtfs_tables`.

    Raises:
        MergeError: A required file or column is missing.
    """
    missing = [name for name in REQUIRED_TABLES if name not in tables]
    if "calendar" not in tables and "calendar_dates" not in tables:
        missing.append("calendar or calendar_dates")
    if missing:
        raise MergeError(f"Feed '{label}' is missing required file(s): {', '.join(missing)}.")
    for name, columns in REQUIRED_COLUMNS.items():
        df = tables.get(name)
        if df is None or (df.empty and len(df.columns) == 0 and name not in REQUIRED_TABLES):
            continue
        absent = [col for col in columns if col not in df.columns]
        if absent:
            raise MergeError(
                f"Feed '{label}': {name}.txt is missing required column(s): {', '.join(absent)}."
            )


def default_feed_labels(paths: Sequence[str]) -> list[str]:
    """Label each feed by its folder or .zip name, de-duplicated.

    Args:
        paths: Feed paths.

    Returns:
        Unique labels parallel to *paths* (a repeated name gets ``_2``, ``_3``...).
    """
    labels: list[str] = []
    for i, raw in enumerate(paths):
        path = Path(raw)
        name = path.stem if path.suffix.lower() == ".zip" else path.name
        base = name or f"feed{i + 1}"
        label, n = base, 1
        while label in labels:
            n += 1
            label = f"{base}_{n}"
        labels.append(label)
    return labels


def load_feed(path: str, label: str) -> Feed:
    """Read and validate one feed.

    Args:
        path: Feed folder or .zip.
        label: Unique feed label.

    Returns:
        The loaded :class:`Feed`.
    """
    tables, other_files = read_gtfs_tables(path)
    validate_feed_tables(label, tables)
    stops = tables["stops"]
    for col in ("stop_lat", "stop_lon", "stop_name"):
        if col not in stops.columns:
            stops[col] = ""
    logging.info(
        "Feed '%s': %d stops, %d routes, %d trips, %d stop_times.",
        label,
        len(stops),
        len(tables["routes"]),
        len(tables["trips"]),
        len(tables["stop_times"]),
    )
    if other_files:
        logging.warning(
            "Feed '%s': not carried into the merged feed: %s.", label, ", ".join(other_files)
        )
    return Feed(label, path, tables, other_files)


def load_stop_overrides(
    path: str, labels: Sequence[str]
) -> dict[tuple[str, str], Optional[tuple[str, str]]]:
    """Read the manual stop-decision CSV.

    Args:
        path: CSV with columns ``feed, stop_id, target_feed, target_stop_id``,
            or "" for none.
        labels: Feed labels in priority order.

    Returns:
        Mapping ``(feed, stop_id)`` to ``(target_feed, target_stop_id)``, or
        to ``None`` to keep the stop separate.

    Raises:
        OSError: The file does not exist.
        MergeError: A column is missing, a label is unknown, a row points
            at an equal- or lower-priority feed, or a stop is listed twice.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise OSError(f"Stop overrides file not found: {path}")
    df = _read_gtfs_csv(path, path)
    required = ["feed", "stop_id", "target_feed", "target_stop_id"]
    absent = [col for col in required if col not in df.columns]
    if absent:
        raise MergeError(f"Stop overrides file is missing column(s): {', '.join(absent)}.")
    rank = {label: i for i, label in enumerate(labels)}
    overrides: dict[tuple[str, str], Optional[tuple[str, str]]] = {}
    for line, record in enumerate(df[required].to_dict("records"), start=2):
        feed, stop_id, target_feed, target_stop = (str(record[c]).strip() for c in required)
        where = f"Stop overrides line {line}"
        if feed not in rank:
            raise MergeError(f"{where}: unknown feed label '{feed}' (labels: {list(labels)}).")
        if rank[feed] == 0:
            raise MergeError(
                f"{where}: '{feed}' is the dominant feed, whose stops are never merged "
                "into another feed. List the lower-priority stop instead."
            )
        if (feed, stop_id) in overrides:
            raise MergeError(f"{where}: stop {feed}/{stop_id} is listed more than once.")
        if not target_feed and not target_stop:
            overrides[(feed, stop_id)] = None
            continue
        if target_feed not in rank or rank[target_feed] >= rank[feed]:
            raise MergeError(
                f"{where}: target_feed '{target_feed}' must be a feed listed before '{feed}'."
            )
        if not target_stop:
            raise MergeError(f"{where}: target_stop_id is blank.")
        overrides[(feed, stop_id)] = (target_feed, target_stop)
    logging.info("Loaded %d stop override(s) from %s.", len(overrides), path)
    return overrides


def load_bay_zones(zones_path: str) -> Any:
    """Read the optional transit-center polygon layer.

    Args:
        zones_path: Polygon layer in any format geopandas reads.

    Returns:
        A GeoDataFrame of the polygons.

    Raises:
        OSError: The file does not exist.
        MergeError: geopandas is not installed or the layer has no CRS.
    """
    if not os.path.exists(zones_path):
        raise OSError(f"Bay zone layer not found: {zones_path}")
    try:
        import geopandas as gpd  # only needed when BAY_ZONES_PATH is set
    except ImportError as exc:
        raise MergeError(
            "BAY_ZONES_PATH needs geopandas (pip install -r requirements.txt); "
            "leave it blank to identify bays by parent_station and name only."
        ) from exc
    zones = gpd.read_file(zones_path)
    if zones.crs is None:
        raise MergeError(f"Bay zone layer '{zones_path}' has no CRS (missing .prj?).")
    logging.info("Loaded %d bay zone polygon(s) from %s.", len(zones), zones_path)
    return zones


def stops_in_zones(stops: pd.DataFrame, zones: Any) -> set[str]:
    """Return the stop_ids that fall inside or on the edge of any zone polygon.

    Args:
        stops: A feed's stops.txt.
        zones: GeoDataFrame from :func:`load_bay_zones`.

    Returns:
        The stop_ids inside a zone.
    """
    import geopandas as gpd  # only needed when BAY_ZONES_PATH is set

    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    valid = lat.notna() & lon.notna()
    points = gpd.GeoDataFrame(
        stops.loc[valid, ["stop_id"]],
        geometry=gpd.points_from_xy(lon[valid], lat[valid]),
        crs="EPSG:4326",
    ).to_crs(zones.crs)
    joined = gpd.sjoin(points, zones[["geometry"]], how="inner", predicate="intersects")
    return set(joined["stop_id"])


# -----------------------------------------------------------------------------
# Agencies and calendars
# -----------------------------------------------------------------------------


def normalize_agency_ids(feed: Feed) -> None:
    """Give every agency and route an agency_id, as a multi-agency feed requires.

    A single-agency feed may leave agency_id blank; it is set to the feed label,
    and blank routes.agency_id values point at that agency.

    Args:
        feed: Feed whose agency and routes tables are updated in place.

    Raises:
        MergeError: A multi-agency feed leaves an agency_id or a route's
            agency_id blank, so routes cannot be assigned.
    """
    agency = feed.tables["agency"]
    routes = feed.tables["routes"]
    if "agency_id" not in agency.columns:
        agency["agency_id"] = ""
    if "agency_id" not in routes.columns:
        routes["agency_id"] = ""
    blank = agency["agency_id"].str.strip() == ""
    if blank.any():
        if len(agency) > 1:
            raise MergeError(
                f"Feed '{feed.label}': agency.txt lists {len(agency)} agencies but not "
                "all have an agency_id."
            )
        agency.loc[blank, "agency_id"] = feed.label
        logging.info("Feed '%s': blank agency_id set to '%s'.", feed.label, feed.label)
    blank_routes = routes["agency_id"].str.strip() == ""
    if blank_routes.any():
        if len(agency) > 1:
            raise MergeError(
                f"Feed '{feed.label}': {int(blank_routes.sum())} route(s) have no agency_id "
                "in a feed with several agencies."
            )
        routes.loc[blank_routes, "agency_id"] = agency["agency_id"].iloc[0]


def check_timezones(feeds: Sequence[Feed]) -> None:
    """Require one agency_timezone across all feeds, as GTFS requires.

    Args:
        feeds: The feeds to merge.

    Raises:
        MergeError: The feeds' agencies use different timezones.
    """
    found: dict[str, list[str]] = {}
    for feed in feeds:
        for tz in feed.tables["agency"]["agency_timezone"].str.strip().unique():
            found.setdefault(tz, []).append(feed.label)
    if len(found) > 1:
        detail = "; ".join(f"{tz}: {', '.join(labels)}" for tz, labels in sorted(found.items()))
        raise MergeError(
            "GTFS allows one agency_timezone per feed, but these feeds use several "
            f"({detail}). Merge only feeds in the same timezone."
        )


def feed_service_dates(tables: Mapping[str, pd.DataFrame]) -> dict[str, set[dt.date]]:
    """Expand a feed's calendars into each service_id's active dates.

    Args:
        tables: The feed's tables.

    Returns:
        Mapping of service_id to the dates it operates.
    """
    return expand_service_active_dates(tables.get("calendar"), tables.get("calendar_dates"))


def feed_active_window(
    tables: Mapping[str, pd.DataFrame], active: Mapping[str, set[dt.date]]
) -> Optional[tuple[dt.date, dt.date]]:
    """Return the first and last date any of the feed's trips operates.

    Args:
        tables: The feed's tables.
        active: Output of :func:`feed_service_dates`.

    Returns:
        ``(first, last)``, or None when no trip has an active date.
    """
    dates: set[dt.date] = set()
    for service_id in tables["trips"]["service_id"].unique():
        dates |= active.get(str(service_id).strip(), set())
    if not dates:
        return None
    return min(dates), max(dates)


def _parse_yyyymmdd(value: str, name: str) -> dt.date:
    """Parse a YYYYMMDD setting.

    Raises:
        MergeError: The value is not a valid YYYYMMDD date.
    """
    try:
        return datetime.strptime(value.strip(), "%Y%m%d").date()
    except ValueError as exc:
        raise MergeError(f"{name} must be a YYYYMMDD date, got '{value}'.") from exc


def resolve_window(
    windows: Mapping[str, Optional[tuple[dt.date, dt.date]]], settings: MergeSettings
) -> Optional[tuple[dt.date, dt.date]]:
    """Pick the date window to clip calendars to.

    Args:
        windows: Each feed's active window, by label.
        settings: Merge settings (clip flag and optional override dates).

    Returns:
        The ``(start, end)`` window, or None when clipping is off.

    Raises:
        MergeError: A feed has no active service, or the window is empty.
    """
    empty = [label for label, window in windows.items() if window is None]
    if empty:
        raise MergeError(
            f"Feed(s) {', '.join(empty)} have no trips with active service dates; "
            "check their calendar.txt / calendar_dates.txt."
        )
    spans = {label: w for label, w in windows.items() if w is not None}
    start = max(w[0] for w in spans.values())
    end = min(w[1] for w in spans.values())
    if not settings.clip_to_common_window:
        if start > end:
            logging.warning(
                "The feeds' service windows do not overlap: no single date has service "
                "from every feed."
            )
        return None
    if settings.window_start:
        start = _parse_yyyymmdd(settings.window_start, "WINDOW_START")
    if settings.window_end:
        end = _parse_yyyymmdd(settings.window_end, "WINDOW_END")
    if start > end:
        detail = "; ".join(f"{label}: {w[0]} to {w[1]}" for label, w in spans.items())
        raise MergeError(
            f"The calendar window {start} to {end} is empty ({detail}). Use feeds that "
            "cover a common period, set WINDOW_START/WINDOW_END, or set "
            "CLIP_TO_COMMON_WINDOW = False."
        )
    return start, end


def clip_feed_calendars(feed: Feed, start: dt.date, end: dt.date) -> int:
    """Clip a feed's calendars to ``start``-``end`` and drop trips with no dates left.

    calendar.txt ranges are trimmed to the window, calendar_dates.txt rows
    outside it are removed, and service_ids with no remaining date are removed
    with their trips, stop_times, frequencies and trip-level transfers.

    Args:
        feed: Feed whose tables are replaced in place.
        start: First date kept.
        end: Last date kept.

    Returns:
        Number of trips dropped.
    """
    tables = feed.tables
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    calendar = tables.get("calendar")
    if calendar is not None and not calendar.empty:
        calendar = calendar.copy()
        first = pd.to_datetime(calendar["start_date"].str.strip(), format="%Y%m%d", errors="coerce")
        last = pd.to_datetime(calendar["end_date"].str.strip(), format="%Y%m%d", errors="coerce")
        parsed = first.notna() & last.notna()
        first = first.where(first >= start_ts, start_ts)
        last = last.where(last <= end_ts, end_ts)
        calendar.loc[parsed, "start_date"] = first[parsed].dt.strftime("%Y%m%d")
        calendar.loc[parsed, "end_date"] = last[parsed].dt.strftime("%Y%m%d")
        tables["calendar"] = calendar.loc[~parsed | (first <= last)].reset_index(drop=True)

    calendar_dates = tables.get("calendar_dates")
    if calendar_dates is not None and not calendar_dates.empty:
        day = pd.to_datetime(calendar_dates["date"].str.strip(), format="%Y%m%d", errors="coerce")
        in_window = day.isna() | ((day >= start_ts) & (day <= end_ts))
        tables["calendar_dates"] = calendar_dates.loc[in_window].reset_index(drop=True)

    active = feed_service_dates(tables)
    live = {service_id for service_id, dates in active.items() if dates}
    for name in ("calendar", "calendar_dates"):
        df = tables.get(name)
        if df is not None and not df.empty:
            tables[name] = df.loc[df["service_id"].str.strip().isin(live)].reset_index(drop=True)

    trips = tables["trips"]
    keep = trips["service_id"].str.strip().isin(live)
    dropped = set(trips.loc[~keep, "trip_id"])
    if not dropped:
        return 0
    tables["trips"] = trips.loc[keep].reset_index(drop=True)
    for name in ("stop_times", "frequencies"):
        df = tables.get(name)
        if df is not None and "trip_id" in df.columns:
            tables[name] = df.loc[~df["trip_id"].isin(dropped)].reset_index(drop=True)
    transfers = tables.get("transfers")
    if transfers is not None and not transfers.empty:
        mask = pd.Series(True, index=transfers.index)
        for col in ("from_trip_id", "to_trip_id"):
            if col in transfers.columns:
                mask &= ~transfers[col].isin(dropped)
        tables["transfers"] = transfers.loc[mask].reset_index(drop=True)
    return len(dropped)


# -----------------------------------------------------------------------------
# Stop matching
# -----------------------------------------------------------------------------


def is_platform_stop(stops: pd.DataFrame) -> pd.Series:
    """Flag stops a rider boards at (``location_type`` 0 or blank).

    Args:
        stops: stops.txt.

    Returns:
        Boolean Series aligned to *stops*.
    """
    if "location_type" not in stops.columns:
        return pd.Series(True, index=stops.index)
    return stops["location_type"].str.strip().isin(("", "0"))


def to_xyz_ft(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude in degrees to Earth-centred x/y/z in feet.

    Straight-line (chord) distance in this space ranks pairs exactly like
    distance along the ground, anywhere on Earth, so no projected CRS is needed.

    Args:
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.

    Returns:
        An ``(n, 3)`` array.
    """
    phi, lam = np.radians(lat), np.radians(lon)
    return EARTH_RADIUS_FT * np.column_stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
    )


def ground_ft_to_chord_ft(distance_ft: float) -> float:
    """Convert a ground distance to the matching x/y/z chord length (feet)."""
    return 2.0 * EARTH_RADIUS_FT * float(np.sin(distance_ft / (2.0 * EARTH_RADIUS_FT)))


def chord_ft_to_ground_ft(chord_ft: float) -> float:
    """Convert an x/y/z chord length to the matching ground distance (feet)."""
    ratio = min(chord_ft / (2.0 * EARTH_RADIUS_FT), 1.0)
    return 2.0 * EARTH_RADIUS_FT * float(np.arcsin(ratio))


def initial_bearing_deg(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Compass bearing (0-360, 0 = north) from point 1 to point 2.

    Args:
        lat1: Start latitudes in degrees.
        lon1: Start longitudes in degrees.
        lat2: End latitudes in degrees.
        lon2: End longitudes in degrees.

    Returns:
        Bearings in degrees; NaN where the two points coincide or a coordinate
        is missing.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return np.where(np.hypot(x, y) < 1e-12, np.nan, bearing)


def compute_stop_headings(stops: pd.DataFrame, stop_times: pd.DataFrame) -> dict[str, np.ndarray]:
    """Collect the directions buses travel through each stop.

    A trip's heading at a stop runs from its previous stop to its next stop (from
    the stop itself at the first stop, to it at the last), rounded to whole
    degrees.

    Args:
        stops: stops.txt with stop_lat/stop_lon.
        stop_times: stop_times.txt.

    Returns:
        Mapping of stop_id to a sorted array of distinct headings. Stops
        without scheduled trips are absent.
    """
    if stop_times.empty:
        return {}
    unique_stops = stops.drop_duplicates("stop_id").set_index("stop_id")
    lat_by_stop = pd.to_numeric(unique_stops["stop_lat"], errors="coerce")
    lon_by_stop = pd.to_numeric(unique_stops["stop_lon"], errors="coerce")

    visits = stop_times[["trip_id", "stop_id"]].copy()
    visits["seq"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    visits = visits.dropna(subset=["seq"]).sort_values(["trip_id", "seq"], kind="mergesort")
    lat = visits["stop_id"].map(lat_by_stop)
    lon = visits["stop_id"].map(lon_by_stop)
    trip = visits["trip_id"]
    has_prev = trip.eq(trip.shift(1))
    has_next = trip.eq(trip.shift(-1))
    from_lat = lat.shift(1).where(has_prev).fillna(lat)
    from_lon = lon.shift(1).where(has_prev).fillna(lon)
    to_lat = lat.shift(-1).where(has_next).fillna(lat)
    to_lon = lon.shift(-1).where(has_next).fillna(lon)
    bearing = initial_bearing_deg(
        from_lat.to_numpy(dtype=float),
        from_lon.to_numpy(dtype=float),
        to_lat.to_numpy(dtype=float),
        to_lon.to_numpy(dtype=float),
    )
    visits["heading"] = np.round(bearing) % 360.0
    visits = visits.dropna(subset=["heading"])
    return {
        str(stop_id): np.unique(group.to_numpy(dtype=float))
        for stop_id, group in visits.groupby("stop_id")["heading"]
    }


def heading_difference_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Smallest angle between any heading in *a* and any heading in *b*.

    Args:
        a: Headings in degrees.
        b: Headings in degrees.

    Returns:
        The smallest difference (0-180), or None when either side has no
        headings.
    """
    if a.size == 0 or b.size == 0:
        return None
    diff = np.abs(a[:, None] - b[None, :]) % 360.0
    return float(np.minimum(diff, 360.0 - diff).min())


def flag_bay_stops(
    stops: pd.DataFrame, name_patterns: Sequence[str], zone_stop_ids: set[str]
) -> np.ndarray:
    """Flag the stops that are bus bays rather than street stops.

    Args:
        stops: stops.txt.
        name_patterns: Case-insensitive regular expressions matched against
            stop_name.
        zone_stop_ids: stop_ids inside a bay zone polygon.

    Returns:
        Boolean array aligned to *stops*.
    """
    is_bay = stops["stop_id"].isin(zone_stop_ids)
    if "parent_station" in stops.columns:
        is_bay |= stops["parent_station"].str.strip() != ""
    if name_patterns:
        pattern = "|".join(f"(?:{p})" for p in name_patterns)
        is_bay |= stops["stop_name"].str.contains(pattern, case=False, regex=True)
    return is_bay.to_numpy(dtype=bool)


def name_similarity(a: str, b: str) -> float:
    """Similarity of two stop names from 0 to 1, ignoring case and punctuation."""

    def _norm(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())

    return round(difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio(), 2)


def match_feed_stops(
    label: str,
    stops: pd.DataFrame,
    headings: Mapping[str, np.ndarray],
    is_bay: np.ndarray,
    pool: _StopPool,
    settings: MergeSettings,
    overrides: Mapping[tuple[str, str], Optional[tuple[str, str]]],
    merged_stop_ids: Mapping[tuple[str, str], str],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Decide which of a lower-priority feed's stops merge into existing stops.

    Overrides are applied first. Remaining platform stops are paired with every
    pool stop within the match distance; pairs that pass the bay distance rule
    and (street stops only) the service and heading rules are accepted closest first, each
    pool stop absorbing at most one stop from this feed and each stop merging at
    most once.

    Args:
        label: This feed's label.
        stops: This feed's stops.txt (original IDs).
        headings: Output of :func:`compute_stop_headings` for this feed.
        is_bay: Output of :func:`flag_bay_stops` for this feed.
        pool: Platform stops from higher-priority feeds.
        settings: Merge settings.
        overrides: Output of :func:`load_stop_overrides`.
        merged_stop_ids: ``(feed, original stop_id)`` to merged stop_id for
            every higher-priority stop.

    Returns:
        A tuple ``(matches, rows)``: original stop_id to pool position for each
        merged stop, and one review row per evaluated pair.
    """
    matches: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    stop_ids = stops["stop_id"].tolist()
    names = dict(zip(stop_ids, stops["stop_name"].tolist()))
    pool_position = {stop_id: pos for pos, stop_id in enumerate(pool.ids)}
    taken: set[int] = set()

    def _row(stop_id: str, pos: Optional[int], outcome: str, **extra: Any) -> dict[str, Any]:
        row: dict[str, Any] = dict.fromkeys(STOP_MATCH_COLUMNS, "")
        row.update(feed=label, stop_id=stop_id, stop_name=names.get(stop_id, ""), outcome=outcome)
        if pos is not None:
            row.update(
                target_feed=pool.feeds[pos],
                target_stop_id=pool.orig_ids[pos],
                target_stop_name=pool.names[pos],
                merged_stop_id=pool.ids[pos],
                name_similarity=name_similarity(names.get(stop_id, ""), pool.names[pos]),
            )
        row.update(extra)
        return row

    lat = pd.to_numeric(stops["stop_lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce").to_numpy(dtype=float)
    platform = is_platform_stop(stops).to_numpy()
    candidate = platform & np.isfinite(lat) & np.isfinite(lon)

    # 1. Manual overrides.
    overridden: set[str] = set()
    for (feed, stop_id), target in overrides.items():
        if feed != label:
            continue
        if stop_id not in names:
            logging.warning("Stop override: %s/%s is not in that feed; ignored.", feed, stop_id)
            continue
        overridden.add(stop_id)
        if target is not None and not platform[stop_ids.index(stop_id)]:
            logging.warning(
                "Stop override: %s/%s is a station or entrance, not a boarding stop; "
                "kept separate.",
                feed,
                stop_id,
            )
            target = None
        if target is None:
            rows.append(_row(stop_id, None, "kept_override"))
            continue
        pos = pool_position.get(merged_stop_ids.get(target, ""))
        if pos is None:
            logging.warning(
                "Stop override: target %s/%s is not a boarding stop of that feed; "
                "%s/%s kept separate.",
                *target,
                feed,
                stop_id,
            )
            rows.append(_row(stop_id, None, "kept_override"))
            continue
        if pos in taken:
            logging.warning(
                "Stop override: %s/%s already absorbed another %s stop; %s kept separate.",
                *target,
                label,
                stop_id,
            )
            rows.append(_row(stop_id, pos, "kept_target_taken"))
            continue
        matches[stop_id] = pos
        taken.add(pos)
        distance: Any = ""
        i = stop_ids.index(stop_id)
        if np.isfinite(lat[i]) and np.isfinite(lon[i]):
            chord = float(
                np.linalg.norm(to_xyz_ft(lat[i : i + 1], lon[i : i + 1])[0] - pool.xyz[pos])
            )
            distance = round(chord_ft_to_ground_ft(chord), 1)
        rows.append(_row(stop_id, pos, "merged_override", distance_ft=distance))

    # 2. Automatic matching.
    street_ft = settings.stop_match_distance_ft
    bay_ft = settings.bay_match_distance_ft
    radius_ft = max(street_ft, bay_ft)
    if radius_ft <= 0 or not pool.ids:
        return matches, rows
    positions = [i for i in np.flatnonzero(candidate) if stop_ids[i] not in overridden]
    if not positions:
        return matches, rows
    pool_xyz = np.vstack(pool.xyz)
    xyz = to_xyz_ft(lat[positions], lon[positions])
    neighbors = cKDTree(pool_xyz).query_ball_point(xyz, r=ground_ft_to_chord_ft(radius_ft), p=2.0)

    eligible: list[tuple[float, str, str, int]] = []
    pending: dict[tuple[str, int], dict[str, Any]] = {}
    for k, i in enumerate(positions):
        stop_id = stop_ids[i]
        for pos in neighbors[k]:
            dist = chord_ft_to_ground_ft(float(np.linalg.norm(xyz[k] - pool_xyz[pos])))
            bay = bool(is_bay[i]) or pool.bay[pos]
            if dist > (max(street_ft, bay_ft) if bay else street_ft):
                continue
            diff = heading_difference_deg(headings.get(stop_id, _NO_HEADINGS), pool.headings[pos])
            row = _row(
                stop_id,
                pos,
                "",
                distance_ft=round(dist, 1),
                heading_diff_deg="" if diff is None else round(diff),
                bay=bay,
            )
            check_heading = not bay and settings.heading_tolerance_deg < 180
            if bay and dist > bay_ft:
                row["outcome"] = "kept_bay"
            elif check_heading and diff is None:
                row["outcome"] = "kept_unserved"
            elif check_heading and diff is not None and diff > settings.heading_tolerance_deg:
                row["outcome"] = "kept_heading"
            else:
                eligible.append((dist, stop_id, pool.ids[pos], pos))
                pending[(stop_id, pos)] = row
                continue
            rows.append(row)

    for _, stop_id, _, pos in sorted(eligible):
        row = pending[(stop_id, pos)]
        if stop_id in matches:
            row["outcome"] = "kept_matched_elsewhere"
        elif pos in taken:
            row["outcome"] = "kept_target_taken"
        else:
            matches[stop_id] = pos
            taken.add(pos)
            row["outcome"] = "merged"
        rows.append(row)
    return matches, rows


# -----------------------------------------------------------------------------
# IDs
# -----------------------------------------------------------------------------


def collect_feed_ids(tables: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Gather every non-blank ID a feed uses, by ID type.

    Args:
        tables: The feed's tables.

    Returns:
        Mapping of ID type (``"stop"``, ``"route"``, ...) to sorted IDs.
    """
    found: dict[str, set[str]] = {}
    for table, columns in ID_COLUMNS.items():
        df = tables.get(table)
        if df is None:
            continue
        for col, id_type in columns:
            if col in df.columns:
                values = df[col]
                found.setdefault(id_type, set()).update(values[values.str.strip() != ""])
    return {id_type: sorted(ids) for id_type, ids in found.items()}


def allocate_ids(
    ids: Sequence[str],
    used: set[str],
    label: str,
    *,
    dominant: bool,
    always_prefix: bool,
    separator: str,
) -> dict[str, str]:
    """Assign merged IDs for one feed's IDs of one type.

    Args:
        ids: The feed's original IDs.
        used: Merged IDs of this type already taken; updated in place.
        label: Feed label, used as the prefix.
        dominant: True for the dominant feed, whose IDs never change.
        always_prefix: Prefix every ID rather than only colliding ones.
        separator: Text between the label and the original ID.

    Returns:
        Mapping of original ID to merged ID.
    """
    mapping: dict[str, str] = {}
    for original in ids:
        if dominant or (not always_prefix and original not in used):
            new = original
        else:
            new = f"{label}{separator}{original}"
            n = 1
            while new in used:
                n += 1
                new = f"{label}{separator}{original}{separator}{n}"
        used.add(new)
        mapping[original] = new
    return mapping


def apply_id_maps(tables: dict[str, pd.DataFrame], maps: Mapping[str, Mapping[str, str]]) -> None:
    """Rewrite every ID column of a feed's tables through the ID maps.

    Args:
        tables: The feed's tables, updated in place.
        maps: ID type to (original ID to merged ID).
    """
    for table, columns in ID_COLUMNS.items():
        df = tables.get(table)
        if df is None:
            continue
        for col, id_type in columns:
            mapping = maps.get(id_type)
            if mapping and col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])


# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------


def rename_duplicate_route_names(
    routes: pd.DataFrame, agency: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Suffix route names that more than one agency uses with the agency_name.

    A route's name is its route_short_name, or its route_long_name when the
    short name is blank. Routes of one agency that share a name (variants of
    one public route) are not duplicates and always get the same new name.

    Args:
        routes: Merged routes.txt.
        agency: Merged agency.txt, in priority order.
        mode: ``"lower_priority"`` (the highest-priority agency keeps the plain
            name), ``"all"`` (every sharing agency is renamed) or ``"keep"``.

    Returns:
        A tuple ``(routes, renames)``: the routes table with names updated, and
        one record per renamed (agency, name) with keys ``agency_id``,
        ``field``, ``original_name`` and ``new_name``.
    """
    routes = routes.copy()
    for col in ("route_short_name", "route_long_name"):
        if col not in routes.columns:
            routes[col] = ""
    short = routes["route_short_name"].str.strip()
    names = pd.DataFrame(
        {
            "name": short.where(short != "", routes["route_long_name"].str.strip()),
            "field": np.where(short != "", "route_short_name", "route_long_name"),
            "agency_id": routes["agency_id"],
        },
        index=routes.index,
    )
    names = names.loc[names["name"] != ""]
    rank = {agency_id: i for i, agency_id in enumerate(agency["agency_id"])}
    agency_name = dict(zip(agency["agency_id"], agency["agency_name"].str.strip()))
    taken = set(names["name"])

    shared: list[str] = []
    renames: list[dict[str, Any]] = []
    for name, group in names.groupby("name", sort=True):
        sharing = sorted(group["agency_id"].unique(), key=lambda a: rank.get(a, len(rank)))
        if len(sharing) < 2:
            continue
        shared.append(str(name))
        to_rename = {"all": sharing, "keep": []}.get(mode, sharing[1:])
        for agency_id in to_rename:
            base = f"{name}_{agency_name.get(agency_id) or agency_id}"
            new, n = base, 1
            while new in taken:
                n += 1
                new = f"{base}_{n}"
            taken.add(new)
            rows = group.loc[group["agency_id"] == agency_id]
            for field_name, field_rows in rows.groupby("field"):
                routes.loc[field_rows.index, str(field_name)] = new
                renames.append(
                    {
                        "agency_id": agency_id,
                        "field": field_name,
                        "original_name": str(name),
                        "new_name": new,
                    }
                )

    if renames:
        examples = ", ".join(f"{r['original_name']} -> {r['new_name']}" for r in renames[:10])
        logging.warning(
            "%d route name(s) are used by more than one agency; renamed %d agency route "
            "name(s) (e.g. %s). The full list is in the ID crosswalk (id_type route_name).",
            len(shared),
            len(renames),
            examples,
        )
    elif shared:
        logging.warning(
            "%d route name(s) are used by more than one agency and were kept as published "
            "(e.g. %s); tools that select routes by name will match all of them.",
            len(shared),
            ", ".join(shared[:10]),
        )
    return routes, renames


def build_feed_info(
    feeds: Sequence[Feed], window: Optional[tuple[dt.date, dt.date]], settings: MergeSettings
) -> pd.DataFrame:
    """Build feed_info.txt for the merged feed.

    Args:
        feeds: The input feeds, dominant first.
        window: The clipped calendar window, if any.
        settings: Merge settings (publisher overrides).

    Returns:
        A one-row feed_info table.
    """
    dominant = feeds[0].tables
    info = dominant.get("feed_info")
    info_row: dict[str, str] = {} if info is None or info.empty else info.iloc[0].to_dict()
    agency_row = dominant["agency"].iloc[0].to_dict()
    versions = []
    for feed in feeds:
        feed_info = feed.tables.get("feed_info")
        version = ""
        if feed_info is not None and not feed_info.empty and "feed_version" in feed_info.columns:
            version = str(feed_info["feed_version"].iloc[0]).strip()
        versions.append(f"{feed.label}={version or 'unversioned'}")
    row = {
        "feed_publisher_name": settings.feed_publisher_name
        or info_row.get("feed_publisher_name")
        or agency_row.get("agency_name", ""),
        "feed_publisher_url": settings.feed_publisher_url
        or info_row.get("feed_publisher_url")
        or agency_row.get("agency_url", ""),
        "feed_lang": info_row.get("feed_lang") or agency_row.get("agency_lang") or "en",
        "feed_version": "merged: " + "; ".join(versions),
    }
    if window is not None:
        row["feed_start_date"] = window[0].strftime("%Y%m%d")
        row["feed_end_date"] = window[1].strftime("%Y%m%d")
    return pd.DataFrame([row])


def find_broken_references(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """List references in the merged feed that point at a missing record.

    Args:
        tables: Merged feed tables.

    Returns:
        One message per broken reference type (empty when all resolve).
    """
    checks = [
        ("stop_times", "stop_id", "stops", "stop_id"),
        ("stop_times", "trip_id", "trips", "trip_id"),
        ("trips", "route_id", "routes", "route_id"),
        ("trips", "shape_id", "shapes", "shape_id"),
        ("routes", "agency_id", "agency", "agency_id"),
        ("stops", "parent_station", "stops", "stop_id"),
        ("frequencies", "trip_id", "trips", "trip_id"),
        ("transfers", "from_stop_id", "stops", "stop_id"),
        ("transfers", "to_stop_id", "stops", "stop_id"),
    ]
    messages: list[str] = []
    for child, col, parent, parent_col in checks:
        child_df, parent_df = tables.get(child), tables.get(parent)
        if child_df is None or col not in child_df.columns:
            continue
        values = child_df[col][child_df[col].str.strip() != ""]
        known = set() if parent_df is None else set(parent_df.get(parent_col, []))
        missing = sorted(set(values) - known)
        if missing:
            messages.append(
                f"{len(missing)} {child}.{col} value(s) missing from {parent}.txt "
                f"(e.g. {', '.join(missing[:5])})"
            )
    services: set[str] = set()
    for name in ("calendar", "calendar_dates"):
        df = tables.get(name)
        if df is not None and "service_id" in df.columns:
            services |= set(df["service_id"])
    missing_services = sorted(set(tables["trips"]["service_id"]) - services)
    if missing_services:
        messages.append(
            f"{len(missing_services)} trips.service_id value(s) missing from the calendars "
            f"(e.g. {', '.join(missing_services[:5])})"
        )
    return messages


def merge_feeds(
    feeds: Sequence[Feed],
    settings: MergeSettings,
    overrides: Optional[Mapping[tuple[str, str], Optional[tuple[str, str]]]] = None,
    zones: Any = None,
) -> MergeResult:
    """Merge the feeds, dominant first, into one set of GTFS tables.

    Args:
        feeds: Loaded feeds in priority order. Their tables are modified.
        settings: Merge settings.
        overrides: Output of :func:`load_stop_overrides`.
        zones: Output of :func:`load_bay_zones`, or None.

    Returns:
        The merged tables plus the review, crosswalk and summary tables.

    Raises:
        MergeError: The feeds cannot be merged as configured.
    """
    if len(feeds) < 2:
        raise MergeError("At least two feeds are needed to merge.")
    if settings.id_prefix_mode not in ("on_collision", "always"):
        raise MergeError("ID_PREFIX_MODE must be 'on_collision' or 'always'.")
    if settings.duplicate_route_names not in ("lower_priority", "all", "keep"):
        raise MergeError("DUPLICATE_ROUTE_NAMES must be 'lower_priority', 'all' or 'keep'.")
    overrides = overrides or {}

    for feed in feeds:
        normalize_agency_ids(feed)
    check_timezones(feeds)

    windows: dict[str, Optional[tuple[dt.date, dt.date]]] = {}
    for feed in feeds:
        windows[feed.label] = feed_active_window(feed.tables, feed_service_dates(feed.tables))
        span = windows[feed.label]
        logging.info(
            "Feed '%s' service dates: %s",
            feed.label,
            "none" if span is None else f"{span[0]} to {span[1]}",
        )
    window = resolve_window(windows, settings)
    trips_in = {feed.label: len(feed.tables["trips"]) for feed in feeds}
    trips_dropped = dict.fromkeys(trips_in, 0)
    if window is not None:
        logging.info("Clipping every calendar to %s to %s.", window[0], window[1])
        for feed in feeds:
            trips_dropped[feed.label] = clip_feed_calendars(feed, window[0], window[1])
            if trips_dropped[feed.label]:
                logging.info(
                    "Feed '%s': dropped %d trip(s) with no service in the window.",
                    feed.label,
                    trips_dropped[feed.label],
                )

    pool = _StopPool()
    used: dict[str, set[str]] = {}
    merged_stop_ids: dict[tuple[str, str], str] = {}
    match_rows: list[dict[str, Any]] = []
    crosswalk_frames: list[pd.DataFrame] = []
    agency_feed: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    for rank, feed in enumerate(feeds):
        tables = feed.tables
        stops = tables["stops"]
        headings = compute_stop_headings(stops, tables["stop_times"])
        zone_ids = stops_in_zones(stops, zones) if zones is not None else set()
        is_bay = flag_bay_stops(stops, settings.bay_name_patterns, zone_ids)

        matches: dict[str, int] = {}
        if rank > 0:
            matches, rows = match_feed_stops(
                feed.label, stops, headings, is_bay, pool, settings, overrides, merged_stop_ids
            )
            match_rows.extend(rows)

        maps: dict[str, dict[str, str]] = {}
        for id_type, ids in collect_feed_ids(tables).items():
            keep = [i for i in ids if not (id_type == "stop" and i in matches)]
            maps[id_type] = allocate_ids(
                keep,
                used.setdefault(id_type, set()),
                feed.label,
                dominant=rank == 0,
                always_prefix=settings.id_prefix_mode == "always" and id_type != "agency",
                separator=settings.id_prefix_separator,
            )
        maps.setdefault("stop", {}).update(
            {stop_id: pool.ids[pos] for stop_id, pos in matches.items()}
        )

        # Grow the pool with this feed's unmerged boarding stops; merged stops lend
        # their headings to the stop that absorbed them.
        lat = pd.to_numeric(stops["stop_lat"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(stops["stop_lon"], errors="coerce").to_numpy(dtype=float)
        platform = is_platform_stop(stops).to_numpy()
        for i, (stop_id, name) in enumerate(zip(stops["stop_id"], stops["stop_name"])):
            own = headings.get(stop_id, _NO_HEADINGS)
            if stop_id in matches:
                pos = matches[stop_id]
                pool.headings[pos] = np.union1d(pool.headings[pos], own)
            elif platform[i] and np.isfinite(lat[i]) and np.isfinite(lon[i]):
                pool.ids.append(maps["stop"].get(stop_id, stop_id))
                pool.feeds.append(feed.label)
                pool.orig_ids.append(stop_id)
                pool.names.append(name)
                pool.xyz.append(to_xyz_ft(lat[i : i + 1], lon[i : i + 1])[0])
                pool.headings.append(own)
                pool.bay.append(bool(is_bay[i]))
        for stop_id in stops["stop_id"]:
            merged_stop_ids[(feed.label, stop_id)] = maps["stop"].get(stop_id, stop_id)

        renamed = 0
        for id_type, mapping in maps.items():
            if not mapping:
                continue
            frame = pd.DataFrame(
                {
                    "id_type": id_type,
                    "feed": feed.label,
                    "original_id": list(mapping.keys()),
                    "merged_id": list(mapping.values()),
                }
            )
            frame["change"] = np.where(
                frame["original_id"] == frame["merged_id"], "unchanged", "renamed"
            )
            if id_type == "stop":
                frame.loc[frame["original_id"].isin(list(matches)), "change"] = "merged"
            renamed += int((frame["change"] == "renamed").sum())
            crosswalk_frames.append(frame)

        tables["stops"] = stops.loc[~stops["stop_id"].isin(list(matches))].reset_index(drop=True)
        apply_id_maps(tables, maps)
        agency_feed.update(dict.fromkeys(tables["agency"]["agency_id"], feed.label))

        span = windows[feed.label]
        summary_rows.append(
            {
                "priority": rank + 1,
                "feed": feed.label,
                "path": feed.path,
                "agencies": "; ".join(tables["agency"]["agency_name"]),
                "service_start": "" if span is None else span[0].isoformat(),
                "service_end": "" if span is None else span[1].isoformat(),
                "trips": trips_in[feed.label],
                "trips_dropped_outside_window": trips_dropped[feed.label],
                "stops": len(stops),
                "stops_merged_into_other_feeds": len(matches),
                "ids_renamed": renamed,
                "files_not_carried": ", ".join(feed.dropped_files),
            }
        )
        logging.info(
            "Feed '%s': %d stop(s) merged into higher-priority stops, %d ID(s) renamed.",
            feed.label,
            len(matches),
            renamed,
        )

    merged: dict[str, pd.DataFrame] = {}
    for name in CARRIED_TABLES:
        frames = [f.tables[name] for f in feeds if name in f.tables and len(f.tables[name].columns)]
        if frames:
            merged[name] = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    transfers = merged.get("transfers")
    if transfers is not None:
        key = [col for col in TRANSFER_KEY if col in transfers.columns]
        deduped = transfers.drop_duplicates(subset=key, keep="first")
        if len(deduped) < len(transfers):
            logging.info(
                "Dropped %d duplicate transfer row(s) created by merged stops.",
                len(transfers) - len(deduped),
            )
        merged["transfers"] = deduped.reset_index(drop=True)
    merged["feed_info"] = build_feed_info(feeds, window, settings)

    merged["routes"], renames = rename_duplicate_route_names(
        merged["routes"], merged["agency"], settings.duplicate_route_names
    )
    if renames:
        crosswalk_frames.append(
            pd.DataFrame(
                {
                    "id_type": "route_name",
                    "feed": [agency_feed.get(r["agency_id"], "") for r in renames],
                    "original_id": [r["original_name"] for r in renames],
                    "merged_id": [r["new_name"] for r in renames],
                    "change": "renamed",
                }
            )
        )
    renamed_by_feed = pd.Series(
        [agency_feed.get(r["agency_id"], "") for r in renames]
    ).value_counts()
    for row in summary_rows:
        row["route_names_renamed"] = int(renamed_by_feed.get(row["feed"], 0))
    for message in find_broken_references(merged):
        logging.warning("Merged feed: %s.", message)

    stop_matches = pd.DataFrame(match_rows, columns=STOP_MATCH_COLUMNS)
    crosswalk = (
        pd.concat(crosswalk_frames, ignore_index=True)
        if crosswalk_frames
        else pd.DataFrame(columns=["id_type", "feed", "original_id", "merged_id", "change"])
    )
    return MergeResult(
        tables=merged,
        stop_matches=stop_matches,
        crosswalk=crosswalk,
        feed_summary=pd.DataFrame(summary_rows),
        window=window,
    )


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def write_gtfs_feed(tables: Mapping[str, pd.DataFrame], folder: Path, write_zip: bool) -> None:
    """Write the merged tables as a GTFS folder and, optionally, a .zip beside it.

    Files already in *folder* that this run did not write are left in place and
    reported, since they would otherwise be mistaken for part of the merged feed.

    Args:
        tables: Merged feed tables keyed by file stem.
        folder: Output feed folder (created if needed).
        write_zip: Also write ``<folder>.zip``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    names = [name for name in (*CARRIED_TABLES, "feed_info") if name in tables]
    for name in names:
        tables[name].to_csv(folder / f"{name}.txt", index=False, lineterminator="\n")
    stale = sorted(p.name for p in folder.glob("*.txt") if p.stem not in names)
    if stale:
        logging.warning(
            "%s also holds file(s) this run did not write (%s); delete them before "
            "publishing the folder. The .zip leaves them out.",
            folder,
            ", ".join(stale),
        )
    logging.info("Wrote merged feed (%d files) -> %s", len(names), folder)
    if write_zip:
        zip_path = folder.with_name(f"{folder.name}.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(folder / f"{name}.txt", arcname=f"{name}.txt")
        logging.info("Wrote %s", zip_path)


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


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


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    The sidecar is named after :data:`MERGED_FEED_NAME` (``_runlog.txt``
    suffix) per CONTRIBUTING.md.

    Args:
        output_dir: Directory that received this run's outputs.
        summary_lines: Human-readable lines describing what the run did.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / f"{MERGED_FEED_NAME}_runlog.txt"
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
        "GTFS FEED MERGER RUN LOG",
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
    except OSError as exc:
        logging.error("Could not write run log '%s': %s", log_path, exc)
        return False
    logging.info("Run log saved to '%s'.", log_path)
    return True


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


# =============================================================================
# MAIN
# =============================================================================


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser (defaults = CONFIGURATION)."""
    p = argparse.ArgumentParser(
        description=(
            "Merge several agencies' GTFS feeds into one, combining duplicate stops. "
            "Feeds are listed in priority order; the first is dominant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gtfs-feeds",
        nargs="+",
        default=list(GTFS_FEEDS),
        help="Feed folders or .zip files, dominant first.",
    )
    p.add_argument(
        "--feed-labels",
        nargs="+",
        default=list(FEED_LABELS),
        help="Short label per feed, parallel to --gtfs-feeds.",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for all outputs.")
    p.add_argument(
        "--stop-match-distance-ft",
        type=float,
        default=STOP_MATCH_DISTANCE_FT,
        help="Merge street stops within this distance (0 = never).",
    )
    p.add_argument(
        "--heading-tolerance-deg",
        type=float,
        default=HEADING_TOLERANCE_DEG,
        help="Largest travel-direction difference for a merge (180 = no check).",
    )
    p.add_argument(
        "--bay-match-distance-ft",
        type=float,
        default=BAY_MATCH_DISTANCE_FT,
        help="Merge bus bays within this distance (0 = never).",
    )
    p.add_argument("--bay-zones", default=BAY_ZONES_PATH, help="Optional bay polygon layer.")
    p.add_argument(
        "--stop-overrides", default=STOP_OVERRIDES_PATH, help="Optional stop overrides CSV."
    )
    p.add_argument(
        "--id-prefix-mode",
        choices=("on_collision", "always"),
        default=ID_PREFIX_MODE,
        help="When to prefix lower-priority IDs with their feed label.",
    )
    p.add_argument(
        "--duplicate-route-names",
        choices=("lower_priority", "all", "keep"),
        default=DUPLICATE_ROUTE_NAMES,
        help="Which agencies' shared route names get an agency_name suffix.",
    )
    p.add_argument(
        "--clip-to-common-window",
        action=argparse.BooleanOptionalAction,
        default=CLIP_TO_COMMON_WINDOW,
        help="Clip calendars to the date window every feed covers.",
    )
    p.add_argument("--window-start", default=WINDOW_START, help="Window override (YYYYMMDD).")
    p.add_argument("--window-end", default=WINDOW_END, help="Window override (YYYYMMDD).")
    p.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=WRITE_ZIP,
        help="Also write the merged feed as a .zip.",
    )
    p.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the configured feeds and write the merged feed and reports.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders or invalid.
    """
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    paths = [str(p) for p in args.gtfs_feeds if str(p).strip()]
    if any(p.startswith(_PLACEHOLDER_PREFIX) for p in paths) or str(args.output_dir).startswith(
        _PLACEHOLDER_PREFIX
    ):
        logging.warning(
            "GTFS_FEEDS and/or OUTPUT_DIR are still placeholders. Update the CONFIGURATION "
            "section or pass --gtfs-feeds/--output-dir."
        )
        return 2
    if len(paths) < 2:
        logging.error("List at least two feeds to merge (dominant first).")
        return 2
    labels = [str(label).strip() for label in args.feed_labels] or default_feed_labels(paths)
    if len(labels) != len(paths) or len(set(labels)) != len(labels) or not all(labels):
        logging.error("FEED_LABELS needs one unique, non-blank label per feed (or none).")
        return 2
    if args.stop_match_distance_ft < 0 or args.bay_match_distance_ft < 0:
        logging.error("Match distances cannot be negative.")
        return 2
    if not 0 <= args.heading_tolerance_deg <= 180:
        logging.error("HEADING_TOLERANCE_DEG must be between 0 and 180.")
        return 2

    settings = MergeSettings(
        id_prefix_mode=args.id_prefix_mode,
        duplicate_route_names=args.duplicate_route_names,
        stop_match_distance_ft=args.stop_match_distance_ft,
        heading_tolerance_deg=args.heading_tolerance_deg,
        bay_match_distance_ft=args.bay_match_distance_ft,
        clip_to_common_window=args.clip_to_common_window,
        window_start=args.window_start,
        window_end=args.window_end,
    )
    output_dir = Path(args.output_dir)
    try:
        overrides = load_stop_overrides(args.stop_overrides, labels)
        zones = load_bay_zones(args.bay_zones) if args.bay_zones else None
        feeds = [load_feed(path, label) for path, label in zip(paths, labels)]
        result = merge_feeds(feeds, settings, overrides, zones)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_gtfs_feed(result.tables, output_dir / MERGED_FEED_NAME, args.zip)
        result.stop_matches.to_csv(output_dir / STOP_MATCHES_FILENAME, index=False)
        result.crosswalk.to_csv(output_dir / CROSSWALK_FILENAME, index=False)
        result.feed_summary.to_csv(output_dir / FEED_SUMMARY_FILENAME, index=False)
    except (MergeError, OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    outcomes = result.stop_matches["outcome"].value_counts()
    summary_lines = [
        f"Feeds (priority order): {', '.join(f'{lab}={p}' for lab, p in zip(labels, paths))}",
        "Calendar window: "
        + ("not clipped" if result.window is None else f"{result.window[0]} to {result.window[1]}"),
        f"Merged stops: {len(result.tables['stops'])}",
        f"Stop pair outcomes: {', '.join(f'{k}={v}' for k, v in outcomes.items()) or 'none'}",
        f"IDs renamed: {int((result.crosswalk['change'] == 'renamed').sum())}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to continue without one."
        )
        return 1
    logging.info(
        "Done: %d stop(s) merged across feeds; review %s.",
        int(outcomes.get("merged", 0) + outcomes.get("merged_override", 0)),
        output_dir / STOP_MATCHES_FILENAME,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
