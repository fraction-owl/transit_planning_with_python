"""Estimate door-to-door transit times on ONE service date across GTFS feeds.

Python 3.9+; standard library only. No ArcPy, pandas, or API key needed.

Inputs:
    One or more GTFS feeds, each an unzipped folder or a .zip (GTFS_FEEDS, or
    GTFS_PATH for a single feed). pathways.txt, transfers.txt and
    frequencies.txt are used when present. Endpoints and OD pairs are set in
    LOCATIONS and OD_PAIRS; BENCHMARK_ESTIMATES optionally holds reference
    travel times from another method for comparison.

Typical usage:
    Save this single file and run: python gtfs_od_travel_times.py
    Edit paths, SERVICE_DATE, LOCATIONS and OD_PAIRS in the configuration below,
    or override scalar settings with flags, for example:
        python gtfs_od_travel_times.py --feed agency_a=agency_a.zip --output-dir od_runs
    Run with --help for every flag and its default. LOCATIONS, OD_PAIRS,
    PERIODS and the benchmark tables are set in the configuration only.
    Use GTFS_FEEDS to add/remove agencies or substitute planned-service feeds.
    A new timestamped output folder is created for every run.
    Set DISCOVERY_ONLY=True to inspect calendars and stop candidates first.

Service selection:
    SERVICE_DATE selects one local service day, applying calendar.txt weekday/date
    ranges, then calendar_dates.txt additions/removals. Several service IDs may
    run concurrently. Do not select a service_id by its name alone: a calendar
    row can be active on the date yet have zero trips. Check service_selection.csv
    for the IDs and modes actually used on the date.
    IDs are prefixed with feed names to prevent collisions across agencies.
    Stop codes remain unchanged; ambiguous codes require a prefixed stop_id.
    For a single-calendar scenario, set USE_MULTIPLE_FEEDS=False,
    SERVICE_DATE=None, and GTFS_PATH / SERVICE_ID. That mode ignores exceptions.
    Times above 24:00 remain on the selected service day; no adjacent day's trips
    are added. This limits late-night/early-morning analyses across day boundaries.

Endpoints and sampling:
    Each location is a fixed WGS84 point or one exact stop. Arrival means alight
    at a scheduled stop and walk to the point. A building or place pin is not a
    verified pedestrian entrance. When a site has several entrances or gates,
    configure each as its own location and OD pair rather than a centroid.
    The router chooses the fastest available itinerary; it is not restricted to
    any particular route or transfer point. Routes absent from the input feeds
    cannot be evaluated.
    Peak/off-peak clock windows are analysis assumptions; set PERIODS to your
    agency's definitions. Every-minute origin departure samples include
    scheduled waits.

Method:
    Each boarding round scans all trips with a feasible boarding. Fixed labels
    from the prior round handle overtaking without depending on trip order.
    Earliest arrival is the objective; fewer boardings break final arrival ties.
    Waiting is unrestricted within MAX_JOURNEY_MINUTES. One surface walking link
    is allowed for access, for each transfer, and for egress. Links cannot be
    chained to bypass walking limits or transfer bans. No total walking budget
    is imposed. Local ties prefer less walking; globally least walking is not
    guaranteed among equal arrival times.
    Outdoor distances are straight-line great-circle distances at WALK_SPEED_MPH.
    Where pathways.txt is present, station access/egress and platform transfers
    use its directed station pathways and their traversal times. Mileage limits
    apply to the outdoor link only; station pathway distance/time is additional.
    Platforms cannot be reached directly from the street without an entrance
    path. With REQUIRE_RAIL_PATHWAYS=True, active subway/metro (route_type 1)
    platforms without entrance/exit paths fail explicitly; set it to False for
    feeds that publish no pathways, so those platforms are reached on foot.
    If DERIVE_ZERO_TIME_WALKWAYS=True, zero-time mode-1 pathways use length divided
    by WALK_SPEED_MPH, rounded up with a one-second minimum. Any adjustment is
    recorded in station_pathways.csv. Exit gates are restricted to their forward
    direction, including where the feed incorrectly marks them bidirectional.
    A transfer buffer is added AFTER physical walking. Any GTFS minimum can
    require more elapsed time. A timed transfer never holds a vehicle.

Outputs (CSV, UTF-8 with BOM, suitable for Excel):
    service_selection.csv / service_inventory.csv: active IDs, modes, trip counts.
    feed_metadata.csv: publisher, feed version and advertised validity dates.
    stop_catalog.csv / stop_candidates.csv: IDs, codes, coordinates and matches.
    resolved_locations.csv / endpoint_connections.csv: endpoint and access audit.
    od_results.csv / od_summary.csv: sampled results, coverage and statistics.
    itinerary_legs.csv: ordered WALK / WAIT / RIDE legs and original source IDs.
    network_transit_segments.csv / network_walk_links.csv: network tables.
    station_pathways.csv / stop_time_adjustments.csv: pathway/time input audits.
    benchmark_comparison.csv: GTFS means against reported benchmark totals.
    benchmark_method_totals.csv / benchmark_method_legs.csv: formula audit.
    methods.txt, settings.json and run.log: assumptions and diagnostics.
    gtfs_od_travel_times_runlog.txt: the verbatim CONFIGURATION block, any
    command-line overrides, the run timestamp and the source script path.
    station_pathway_minutes is INCLUDED in walk_minutes, not an extra cost.
    Means and percentiles cover successful samples only; always check coverage.

Benchmark comparison (optional):
    BENCHMARK_ESTIMATES holds totals from a sketch method: distance / assumed
    speed + half headway for each transit leg, plus walking, 1 minute per
    intermediate metro station and 5 minutes per metro use. This router uses
    scheduled running/dwell times, actual sampled waits and station pathways.
    Do not add half a headway, metro dwell or fixed entry/exit time again. The
    reference audit retains the benchmark's distinct assumptions. Rounded source
    values and different endpoints can prevent exact agreement. Endpoint
    variants mapped to one benchmark through PAIR_BENCHMARK_IDS are not
    independent OD pairs; this script does not calculate a network average.

Limitations:
    Surface walks ignore highways, fences, water, grade separations and security
    restrictions. Check access at secured sites such as military bases. Station
    routing permits stairs/escalators; it is not wheelchair-accessible routing.
    No elevator outage, fare or gate-opening-hours model is included.
    p90 describes departure-time variation, not operational reliability.
    Only regular pickup/drop-off (types 0/blank) is allowed. No inferred through
    service from block IDs. Route/trip-qualified and linked-trip transfer rules
    fail explicitly; ordinary stop/station transfer types 0-3 are supported.
    Exact frequency schedules (exact_times=1) are expanded. Headway-only service
    (exact_times=0) fails rather than inventing scheduled departures.
    Missing interior times are optionally interpolated at timepoint=0 stops.
    Feed validity dates alone do not guarantee all modes operate on a given day:
    inspect service_selection.csv and warnings about absent feed/route types.

Reference: https://gtfs.org/documentation/schedule/reference/
"""

from __future__ import annotations

import argparse
import csv
import heapq
import io
import json
import logging
import math
import re
import statistics
import sys
import zipfile
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# =============================================================================
# CONFIGURATION -- edit here; IDs/codes must remain strings, including leading 0s.
# Scalar and list settings can also be passed as command-line flags (see --help).
# =============================================================================
# === BEGIN CONFIG ===
GTFS_PATH = Path(r"Path\To\Agency_A_GTFS")  # Unzipped GTFS folder or .zip.
SECOND_GTFS_PATH = Path(r"Path\To\Agency_B_GTFS.zip")  # Unzipped GTFS folder or .zip.
# Default: one service DATE, all concurrent calendars, from every feed listed.
# Feed names become ID prefixes; add, remove or rename entries as needed.
USE_MULTIPLE_FEEDS = True
GTFS_FEEDS = {"agency_a": GTFS_PATH, "agency_b": SECOND_GTFS_PATH}
SERVICE_DATE: Optional[str] = "20260115"  # YYYYMMDD, in agency local time.
# For a single-calendar scenario: USE_MULTIPLE_FEEDS=False,
# SERVICE_DATE=None, and set GTFS_PATH / SERVICE_ID. Exceptions are then ignored.
OUTPUT_DIR = Path(r"Path\To\Output_Folder")  # A timestamped run folder is created inside.
SERVICE_ID: Optional[str] = None  # Used ONLY when SERVICE_DATE=None, for a single feed.
DISCOVERY_ONLY = False  # True: inventories only, no routing.

WALK_SPEED_MPH = 3.0
MAX_ACCESS_WALK_MILES = 0.50
MAX_TRANSFER_WALK_MILES = 0.25
MAX_EGRESS_WALK_MILES = 0.50  # Try 1.0 as a walking-limit sensitivity test.
MIN_TRANSFER_MINUTES = 2.0  # Added AFTER walking; GTFS minimum can require more.
INITIAL_BOARDING_BUFFER_MINUTES = 0.0
MAX_TRANSFERS = 2  # Two transfers = up to three separate vehicle boardings.
MAX_JOURNEY_MINUTES = 180.0
ALLOW_WALK_ONLY = True
MAX_WALK_ONLY_MILES = 1.0

# Build the network once, then run the selected periods. These windows are analysis
# assumptions; replace them with your agency's peak/off-peak definitions.
# Each entry: (start, end, [OD pair IDs]). Every OD pair must appear in at least one.
USE_PERIODS = True
PERIODS = {
    "AM_peak": ("06:00", "08:59", ["1", "2"]),
    "Midday": ("10:00", "13:59", ["3"]),
    "PM_peak": ("15:00", "17:59", ["1", "2"]),
}
DEPARTURE_STEP_MINUTES = 1  # Every-minute sampling includes schedule-dependent waits.
# These three settings apply only when USE_PERIODS=False (all OD pairs then run).
EXPLICIT_DEPARTURE_TIMES: List[str] = []  # Example: ["07:00", "08:00", "17:00"].
DEPARTURE_START = "06:00"
DEPARTURE_END = "08:59"
INCLUDE_REVERSE_PAIRS = False
INTERPOLATE_MISSING_STOP_TIMES = True
DERIVE_ZERO_TIME_WALKWAYS = True  # Audit zero-time mode-1 pathways using length / walk speed.
# True: subway/metro (route_type 1) platforms must be reached through pathways.txt.
# False: platforms without pathways are reached by surface walking like bus stops.
REQUIRE_RAIL_PATHWAYS = True
EXPORT_NETWORK = True
LOG_LEVEL = logging.INFO
# True: if the _runlog.txt sidecar cannot be written, stop before routing so a run
# folder is never left without a record of its configuration.
REQUIRE_RUN_LOG = True

# Fixed WGS84 points allow existing/planned comparisons without moving endpoints
# when stops move. Replace a lat/lon pair with ONE stop_code or stop_id if desired,
# e.g. "lat": None, "lon": None, "stop_code": "1234". Keep IDs/codes quoted.
# Model a site with several entrances or gates as separate locations, not a centroid.
# The placeholders below are example points; replace them with your own endpoints.
LOCATIONS: Dict[str, Dict[str, Any]] = {
    "location_a": {
        "name": "Example Origin A",
        "lat": 38.8100,
        "lon": -77.0160,
        "stop_code": None,
        "stop_id": None,
        "candidate_pattern": "MAIN ST|EXAMPLE",
        "note": "Placeholder point; replace with a real origin.",
    },
    "location_b": {
        "name": "Example Destination B",
        "lat": 38.9800,
        "lon": -77.0160,
        "stop_code": None,
        "stop_id": None,
        "candidate_pattern": "MAIN ST|EXAMPLE",
        "note": "Placeholder point; replace with a real destination.",
    },
    "location_c": {
        "name": "Example Destination C",
        "lat": 38.8960,
        "lon": -77.1000,
        "stop_code": None,
        "stop_id": None,
        "candidate_pattern": "MAIN ST|EXAMPLE",
        "note": "Placeholder point; replace with a real destination.",
    },
}
# Each entry: (pair_id, origin location key, destination location key).
OD_PAIRS = [
    ("1", "location_a", "location_b"),
    ("2", "location_a", "location_c"),
    ("3", "location_b", "location_c"),
]

# Optional reference estimates from another method (for example, a sketch-planning
# worksheet), retained for comparison; NOT routing costs. The entries below are
# examples; replace them with your own reference totals, or set to {} to skip.
# Leg tuple: route, distance_miles, assumed_speed_mph, displayed_headway_minutes,
# displayed_wait_minutes, intermediate_metro_stations. The latter two preserve
# source rounding; the formula audit independently uses half the displayed headway.
# Route names should match the feed's route_short_name (or route_long_name when the
# short name is blank) so the comparison can flag routes absent from the service day.
BENCHMARK_WALK_SPEED_MPH = 3.0
BENCHMARK_METRO_SPEED_MPH = 45.0
BENCHMARK_METRO_STATION_MINUTES = 1.0
BENCHMARK_METRO_ENTRY_EXIT_MINUTES = 5.0
BENCHMARK_METRO_HEADWAYS = {"Peak": 8.0, "Off-Peak": 12.0}
# Benchmark legs on these routes use the metro speed, headway and station values above.
BENCHMARK_METRO_ROUTES: set = set()  # Example: {"Red Line", "Blue Line"}.
# Per-pair tuple: period, reported_total, access_walk_miles, final_walk_miles,
# number_of_metro_uses, list_of_legs.
BENCHMARK_ESTIMATES = {
    "1": ("Peak", 57.6, 0.0, 0.0, 0, [("10", 11.7, 14.0, 15.0, 7.5, 0)]),
    "2": (
        "Peak",
        55.3,
        0.0,
        0.0,
        0,
        [
            ("10", 5.9, 14.0, 15.0, 7.5, 0),
            ("20", 4.6, 22.0, 20.0, 10.0, 0),
        ],
    ),
}
# Map endpoint variants to one benchmark pair, e.g. {"1A": "1", "1B": "1"}.
PAIR_BENCHMARK_IDS: Dict[str, str] = {}
# === END CONFIG ===
# =============================================================================

EARTH_RADIUS_M = 6371008.8
METERS_PER_MILE = 1609.344
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
Row = Dict[str, Any]


@dataclass(frozen=True)
class Stop:
    """A GTFS stop or station, with its original string identifiers."""

    id: str
    code: str
    name: str
    lat: float
    lon: float
    parent: str = ""
    location_type: str = "0"
    feed_id: str = ""
    source_id: str = ""


@dataclass(frozen=True)
class Event:
    """An arrival and departure at one ordered stop on a trip."""

    stop: str
    arrival: int
    departure: int
    pickup: bool
    dropoff: bool
    sequence: int


@dataclass(frozen=True)
class Trip:
    """One selected scheduled run, optionally expanded from an exact frequency."""

    id: str
    source_id: str
    route: str
    route_id: str
    direction: str
    headsign: str
    events: Tuple[Event, ...]
    feed_id: str = ""
    service_id: str = ""
    route_type: str = ""


@dataclass(frozen=True)
class Place:
    """A single real endpoint rather than a collection of zero-cost stop anchors."""

    id: str
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Leg:
    """An immutable itinerary leg; stop keys use 'stop:' and places use 'place:'."""

    mode: str
    origin: str
    destination: str
    start: int
    end: int
    distance_m: float = 0.0
    route: str = ""
    route_id: str = ""
    trip_id: str = ""
    source_trip_id: str = ""
    direction: str = ""
    headsign: str = ""
    feed_id: str = ""
    service_id: str = ""
    route_type: str = ""
    walk_basis: str = ""
    pathway_id: str = ""


@dataclass(frozen=True)
class Label:
    """A physical arrival time and an immutable path achieving it."""

    time: int
    legs: Tuple[Leg, ...]


@dataclass(frozen=True)
class BoardLabel:
    """A physical arrival plus the earliest allowable next boarding time."""

    label: Label
    ready: int


@dataclass(frozen=True)
class WalkStep:
    """A surface segment or a directed GTFS station pathway, in seconds/meters."""

    origin: str
    destination: str
    distance_m: float
    seconds: int
    pathway_id: str = ""


@dataclass(frozen=True)
class EndpointLink:
    """A complete place/platform connection, including station paths if needed."""

    stop: str
    distance_m: float
    seconds: int
    steps: Tuple[WalkStep, ...]


@dataclass(frozen=True)
class WalkLink:
    """One physical transfer walk and the required elapsed time before boarding."""

    to_stop: str
    distance_m: float
    seconds: int
    min_elapsed: int
    steps: Tuple[WalkStep, ...] = ()


def parse_time(value: str) -> int:
    """Parse HH:MM[:SS] without wrapping service times above 24 hours."""
    if not re.fullmatch(r"\d+:\d{2}(?::\d{2})?", value):
        raise ValueError(f"Invalid time: {value!r}; use HH:MM[:SS].")
    fields = [int(item) for item in value.split(":")]
    hour, minute = fields[:2]
    second = fields[2] if len(fields) == 3 else 0
    if minute > 59 or second > 59:
        raise ValueError(f"Invalid time: {value!r}.")
    return hour * 3600 + minute * 60 + second


def format_time(seconds: int) -> str:
    """Format seconds on the GTFS service-day clock, retaining hours above 24."""
    hour, remaining = divmod(seconds, 3600)
    minute, second = divmod(remaining, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return straight-line great-circle distance on the Earth's surface."""
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = a2 - a1, math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, value))))


def walk_seconds(meters: float) -> int:
    """Round walking upward to the next second to avoid optimistic connections."""
    return math.ceil(meters / (WALK_SPEED_MPH * METERS_PER_MILE / 3600))


def valid_coordinates(lat: Any, lon: Any) -> Tuple[float, float]:
    """Validate WGS84 degrees, rejecting blanks, infinities, and NaNs."""
    lat, lon = float(lat), float(lon)
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90:
        raise ValueError("Latitude must be finite and between -90 and 90.")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be finite and between -180 and 180.")
    return lat, lon


class Feed:
    """Stream UTF-8 GTFS tables from a folder or a ZIP, preserving literal IDs."""

    def __init__(self, path: Path) -> None:
        """Open a GTFS folder or ZIP and index its table names."""
        self.path = path
        self.archive: Optional[zipfile.ZipFile] = None
        if path.is_dir():
            self.names = {item.name.lower(): item for item in path.iterdir() if item.is_file()}
        elif path.is_file() and zipfile.is_zipfile(path):
            self.archive = zipfile.ZipFile(path)
            self.names = {}
            for name in self.archive.namelist():
                base = Path(name).name.lower()
                if base.endswith(".txt") and not name.startswith("__MACOSX/"):
                    if base in self.names:
                        raise ValueError(f"Duplicate table {base} in ZIP; supply one GTFS feed.")
                    self.names[base] = name
        else:
            raise FileNotFoundError(f"GTFS_PATH is not a GTFS folder or ZIP: {path}")

    def rows(self, filename: str, required: Sequence[str] = ()) -> Iterator[Row]:
        """Yield a table; absent optional tables yield no rows."""
        source = self.names.get(filename)
        if source is None:
            if required:
                raise ValueError(f"Missing required GTFS file: {filename}.")
            return
        stream = (
            io.TextIOWrapper(self.archive.open(str(source)), encoding="utf-8-sig", newline="")
            if self.archive
            else Path(source).open(encoding="utf-8-sig", newline="")
        )
        with stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                if required:
                    raise ValueError(f"{filename} is empty.")
                return
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"Duplicate columns in {filename}.")
            missing = set(required) - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{filename} is missing columns: {sorted(missing)}.")
            for number, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f"Extra CSV fields in {filename}, line {number}.")
                yield {key: (value or "").strip() for key, value in row.items()}

    def close(self) -> None:
        """Close an opened ZIP."""
        if self.archive:
            self.archive.close()


class CombinedFeed:
    """Stream multiple feeds with scoped IDs; never merge identical source IDs."""

    ID_FIELDS = {
        "stop_id",
        "parent_station",
        "trip_id",
        "route_id",
        "service_id",
        "shape_id",
        "agency_id",
        "block_id",
        "pathway_id",
        "level_id",
        "from_stop_id",
        "to_stop_id",
        "from_route_id",
        "to_route_id",
        "from_trip_id",
        "to_trip_id",
    }

    def __init__(self, paths: Dict[str, Path]) -> None:
        """Open each named feed, closing any already opened if one fails."""
        self.feeds: Dict[str, Feed] = {}
        try:
            if not paths:
                raise ValueError("GTFS_FEEDS is empty.")
            for key, path in paths.items():
                if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                    raise ValueError("Feed names must use letters, digits, underscores or hyphens.")
                self.feeds[key] = Feed(Path(path))
            self.names = set().union(*(feed.names for feed in self.feeds.values()))
        except Exception:
            self.close()
            raise

    def rows(self, filename: str, required: Sequence[str] = ()) -> Iterator[Row]:
        """Prefix identifier references; preserve public codes and raw trip/stop IDs."""
        for key, feed in self.feeds.items():
            for row in feed.rows(filename, required):
                result = dict(row)
                for field in self.ID_FIELDS & row.keys():
                    if row[field]:
                        result[field] = key + ":" + row[field]
                result["_feed_id"] = key
                result["_source_id"] = row.get("trip_id") or row.get("stop_id", "")
                yield result

    def close(self) -> None:
        """Close all constituent archives, including partially initialized ones."""
        for feed in self.feeds.values():
            feed.close()


def active_services(feed: Feed, date_text: str) -> set:
    """Apply weekly service and dated additions/removals for ONE local service day."""
    if not re.fullmatch(r"\d{8}", date_text):
        raise ValueError("SERVICE_DATE must be YYYYMMDD, for example '20261022'.")
    date = datetime.strptime(date_text, "%Y%m%d").date()
    weekday = WEEKDAYS[date.weekday()]
    calendars = unique_rows(feed.rows("calendar.txt"), "service_id", "calendar")
    selected = set()
    for service, row in calendars.items():
        start = datetime.strptime(row["start_date"], "%Y%m%d").date()
        end = datetime.strptime(row["end_date"], "%Y%m%d").date()
        if start > end or any(row.get(day) not in {"0", "1"} for day in WEEKDAYS):
            raise ValueError(f"Invalid calendar.txt row for {service}.")
        if start <= date <= end and row[weekday] == "1":
            selected.add(service)
    seen = set()
    for row in feed.rows("calendar_dates.txt"):
        key = (row["service_id"], row["date"])
        if key in seen:
            raise ValueError(f"Duplicate calendar_dates service/date: {key}.")
        seen.add(key)
        if row["exception_type"] not in {"1", "2"}:
            raise ValueError(f"Invalid exception_type for {key}.")
        if row["date"] == date_text:
            if row["exception_type"] == "1":
                selected.add(row["service_id"])
            else:
                selected.discard(row["service_id"])
    return selected


def select_services(feed: Feed, trip_rows: Dict[str, Row], output: Path) -> set:
    """Select dated service or the legacy exact ID; audit coverage by feed/route type."""
    if SERVICE_DATE is None:
        if USE_MULTIPLE_FEEDS:
            raise ValueError("Multiple feeds require SERVICE_DATE; raw IDs are feed-specific.")
        selected = {SERVICE_ID} if SERVICE_ID is not None else set()
        logging.warning("Generic SERVICE_ID=%s: date exceptions are NOT applied.", SERVICE_ID)
    else:
        selected = active_services(feed, SERVICE_DATE)
    counts = Counter(row["service_id"] for row in trip_rows.values())
    unused = selected - counts.keys()
    if unused:
        logging.info("Active calendar rows with zero trips (unused): %s", sorted(unused))
    selected &= counts.keys()
    routes = unique_rows(feed.rows("routes.txt", ["route_id"]), "route_id", "routes")
    selected_counts = Counter(
        (
            row.get("_feed_id", "single"),
            row["service_id"],
            routes[row["route_id"]].get("route_type", ""),
        )
        for row in trip_rows.values()
        if row["service_id"] in selected
    )
    records = [
        {
            "feed": key[0],
            "service_date": SERVICE_DATE or "",
            "service_id": key[1],
            "route_type": key[2],
            "trip_count": count,
        }
        for key, count in sorted(selected_counts.items())
    ]
    write_csv(
        output / "service_selection.csv",
        records,
        ["feed", "service_date", "service_id", "route_type", "trip_count"],
    )
    for record in records:
        logging.info("Selected service: %s", record)
    if not selected:
        raise ValueError("No trips for selected date/ID; see service_inventory.csv.")
    # A stale rail calendar can coexist with a valid bus calendar. Report missing
    # modes explicitly rather than silently describing a bus-only feed as complete.
    all_modes = {
        (row.get("_feed_id", "single"), routes[row["route_id"]].get("route_type", ""))
        for row in trip_rows.values()
    }
    found_modes = {(key[0], key[2]) for key in selected_counts}
    missing = all_modes - found_modes
    if missing:
        logging.warning("NO ACTIVE TRIPS for these feed/route types: %s", sorted(missing))
    timezones = {row.get("agency_timezone", "") for row in feed.rows("agency.txt")}
    normalized = {"America/New_York" if zone == "US/Eastern" else zone for zone in timezones}
    if len(normalized) > 1:
        raise ValueError(f"Mixed agency time zones are not supported: {sorted(timezones)}")
    write_csv(
        output / "feed_metadata.csv",
        feed.rows("feed_info.txt"),
        [
            "_feed_id",
            "feed_publisher_name",
            "feed_version",
            "feed_start_date",
            "feed_end_date",
        ],
    )
    return selected


def unique_rows(rows: Iterable[Row], key: str, table: str) -> Dict[str, Row]:
    """Index a GTFS table without silently overwriting duplicate identifiers."""
    result = {}
    for row in rows:
        value = row.get(key, "")
        if not value or value in result:
            raise ValueError(f"Blank or duplicate {key}={value!r} in {table}.")
        result[value] = row
    return result


def write_csv(path: Path, rows: Iterable[Row], fields: Sequence[str]) -> None:
    """Write deterministic headers even when there are no output records."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_stops(feed: Feed) -> Dict[str, Stop]:
    """Read stop/station coordinates; unsupported unlocated nodes are excluded."""
    result = {}
    rows = unique_rows(feed.rows("stops.txt", ["stop_id", "stop_name"]), "stop_id", "stops")
    for key, row in rows.items():
        kind = row.get("location_type", "") or "0"
        if not row.get("stop_lat") or not row.get("stop_lon"):
            if kind in {"0", "1"}:
                raise ValueError(f"Missing coordinates for stop/station {key!r}.")
            continue
        lat, lon = valid_coordinates(row["stop_lat"], row["stop_lon"])
        result[key] = Stop(
            key,
            row.get("stop_code", ""),
            row["stop_name"],
            lat,
            lon,
            row.get("parent_station", ""),
            kind,
            row.get("_feed_id", ""),
            row.get("_source_id", key),
        )
    return result


def export_inventories(
    feed: Feed,
    trip_rows: Dict[str, Row],
    stops: Dict[str, Stop],
    output: Path,
    served: Optional[Dict[str, set]] = None,
    selected_services: Optional[set] = None,
) -> None:
    """Export service choices and endpoint candidates without guessing identities."""
    counts = Counter(row["service_id"] for row in trip_rows.values())
    calendars = unique_rows(feed.rows("calendar.txt"), "service_id", "calendar")
    exceptions = Counter(row["service_id"] for row in feed.rows("calendar_dates.txt"))
    inventory = []
    for service in sorted(set(counts) | set(calendars) | set(exceptions)):
        inventory.append(
            {
                **calendars.get(service, {}),
                "service_id": service,
                "trip_count": counts[service],
                "exception_rows": exceptions[service],
                "selected": service in (selected_services or set()),
                "service_date": SERVICE_DATE or "",
            }
        )
    write_csv(
        output / "service_inventory.csv",
        inventory,
        [
            "service_id",
            "selected",
            "trip_count",
            *WEEKDAYS,
            "start_date",
            "end_date",
            "exception_rows",
            "service_date",
        ],
    )
    catalog = []
    for stop in sorted(stops.values(), key=lambda item: item.id):
        catalog.append(
            {
                "stop_id": stop.id,
                "stop_code": stop.code,
                "stop_name": stop.name,
                "latitude": stop.lat,
                "longitude": stop.lon,
                "location_type": stop.location_type,
                "parent_station": stop.parent,
                "feed_id": stop.feed_id,
                "source_stop_id": stop.source_id,
                "served_selected_calendar": "" if served is None else stop.id in served,
                "selected_routes": ""
                if served is None
                else ", ".join(sorted(served.get(stop.id, []))),
            }
        )
    columns = list(catalog[0]) if catalog else ["stop_id", "stop_code", "stop_name"]
    write_csv(output / "stop_catalog.csv", catalog, columns)
    candidates = []
    for place_id, config in LOCATIONS.items():
        pattern = re.compile(config["candidate_pattern"], re.IGNORECASE)
        matches = [row for row in catalog if pattern.search(row["stop_name"])]
        if not matches:
            matches = [{"stop_name": "NO NAME MATCH; use stop_catalog.csv or coordinates"}]
        candidates.extend(
            {"location_id": place_id, "location": config["name"], **row} for row in matches
        )
    write_csv(output / "stop_candidates.csv", candidates, ["location_id", "location", *columns])


def make_events(rows: List[Row], trip_id: str, adjustments: List[Row]) -> Tuple[Event, ...]:
    """Validate stop order and fill permitted interior times, logging each estimate."""
    rows.sort(key=lambda row: int(row["stop_sequence"]))
    sequences = [int(row["stop_sequence"]) for row in rows]
    if len(rows) < 2 or len(sequences) != len(set(sequences)) or min(sequences) < 0:
        raise ValueError(f"Trip {trip_id}: needs >=2 stops with unique nonnegative sequences.")
    times: List[Optional[Tuple[int, int]]] = []
    for row in rows:
        arr, dep = row.get("arrival_time", ""), row.get("departure_time", "")
        if arr or dep:
            times.append((parse_time(arr or dep), parse_time(dep or arr)))
            if not (arr and dep):
                adjustments.append(
                    {
                        "trip_id": trip_id,
                        "stop_sequence": row["stop_sequence"],
                        "method": "copied_other_time",
                        "arrival": arr or dep,
                        "departure": dep or arr,
                    }
                )
        else:
            if not INTERPOLATE_MISSING_STOP_TIMES or row.get("timepoint", "1") != "0":
                raise ValueError(
                    f"Trip {trip_id}, stop {row['stop_sequence']}: missing required time."
                )
            times.append(None)
    if times[0] is None or times[-1] is None:
        raise ValueError(f"Trip {trip_id}: first/last stop times cannot be missing.")
    left = 0
    while left < len(rows) - 1:
        right = left + 1
        while times[right] is None:
            right += 1
        if right > left + 1:
            start, end = times[left][1], times[right][0]
            if end < start:
                raise ValueError(f"Trip {trip_id}: decreasing interpolation anchors.")
            try:
                distances = [
                    float(row.get("shape_dist_traveled", "")) for row in rows[left : right + 1]
                ]
                use_distance = (
                    all(math.isfinite(value) for value in distances)
                    and distances[-1] > distances[0]
                    and all(a <= b for a, b in zip(distances, distances[1:]))
                )
            except (TypeError, ValueError):
                use_distance = False
            for index in range(left + 1, right):
                fraction = (
                    (distances[index - left] - distances[0]) / (distances[-1] - distances[0])
                    if use_distance
                    else (index - left) / (right - left)
                )
                value = round(start + fraction * (end - start))
                times[index] = (value, value)
                adjustments.append(
                    {
                        "trip_id": trip_id,
                        "stop_sequence": rows[index]["stop_sequence"],
                        "method": "shape_distance" if use_distance else "stop_index",
                        "arrival": format_time(value),
                        "departure": format_time(value),
                    }
                )
        left = right
    events = []
    previous_departure = -1
    for row, time in zip(rows, times):
        arrival, departure = time
        if arrival < previous_departure or departure < arrival:
            raise ValueError(f"Trip {trip_id}, sequence {row['stop_sequence']}: decreasing times.")
        pickup, dropoff = row.get("pickup_type", "") or "0", row.get("drop_off_type", "") or "0"
        if pickup not in {"0", "1", "2", "3"} or dropoff not in {"0", "1", "2", "3"}:
            raise ValueError(f"Trip {trip_id}: invalid pickup/drop-off type.")
        events.append(
            Event(
                row["stop_id"],
                arrival,
                departure,
                pickup == "0",
                dropoff == "0",
                int(row["stop_sequence"]),
            )
        )
        previous_departure = departure
    return tuple(events)


def load_trips(
    feed: Feed,
    trip_rows: Dict[str, Row],
    stops: Dict[str, Stop],
    output: Path,
    service_ids: Optional[set] = None,
) -> List[Trip]:
    """Filter selected services, validate times, and expand exact frequencies."""
    services = {SERVICE_ID} if service_ids is None else service_ids
    selected = {key: row for key, row in trip_rows.items() if row["service_id"] in services}
    if not selected:
        raise ValueError(f"No trips for SERVICE_ID={SERVICE_ID!r}; see service_inventory.csv.")
    routes = unique_rows(feed.rows("routes.txt", ["route_id"]), "route_id", "routes")
    grouped: Dict[str, List[Row]] = defaultdict(list)
    for row in feed.rows("stop_times.txt", ["trip_id", "stop_id", "stop_sequence"]):
        if row["trip_id"] in selected:
            if row["stop_id"] not in stops or stops[row["stop_id"]].location_type != "0":
                raise ValueError(f"Missing or unsupported platform stop {row['stop_id']!r}.")
            if row.get("start_pickup_drop_off_window") or row.get("end_pickup_drop_off_window"):
                raise ValueError("Flexible pickup windows are not supported by this router.")
            grouped[row["trip_id"]].append(row)
    frequencies: Dict[str, List[Row]] = defaultdict(list)
    for row in feed.rows("frequencies.txt"):
        if row["trip_id"] in selected:
            if row.get("exact_times", "0") != "1":
                raise ValueError(
                    f"Trip {row['trip_id']} uses headway-only frequencies. Exact departure routing "
                    "needs scheduled trips; a frequency-based wait model is not implemented."
                )
            frequencies[row["trip_id"]].append(row)
    trips = []
    adjustments: List[Row] = []
    for trip_id, row in sorted(selected.items()):
        route = routes.get(row["route_id"])
        if route is None:
            raise ValueError(f"Trip {trip_id}: unknown route {row['route_id']}.")
        events = make_events(grouped[trip_id], trip_id, adjustments)
        route_type = route.get("route_type", "")
        route_name = (
            route.get("route_short_name") or route.get("route_long_name") or row["route_id"]
        )
        trip = Trip(
            trip_id,
            row.get("_source_id", trip_id),
            route_name,
            row["route_id"],
            row.get("direction_id", ""),
            row.get("trip_headsign", ""),
            events,
            row.get("_feed_id", ""),
            row["service_id"],
            route_type,
        )
        if trip_id not in frequencies:
            trips.append(trip)
            continue
        previous_end = -1
        intervals = sorted(frequencies[trip_id], key=lambda item: parse_time(item["start_time"]))
        for frequency in intervals:
            start, end = parse_time(frequency["start_time"]), parse_time(frequency["end_time"])
            headway = int(frequency["headway_secs"])
            if headway <= 0 or start >= end or start < previous_end:
                raise ValueError(f"Trip {trip_id}: invalid or overlapping frequency intervals.")
            for departure in range(start, end, headway):
                offset = departure - events[0].departure
                shifted = tuple(
                    replace(
                        event, arrival=event.arrival + offset, departure=event.departure + offset
                    )
                    for event in events
                )
                trips.append(replace(trip, id=f"{trip_id}@{departure}", events=shifted))
            previous_end = end
    write_csv(
        output / "stop_time_adjustments.csv",
        adjustments,
        ["trip_id", "stop_sequence", "method", "arrival", "departure"],
    )
    if adjustments:
        logging.warning(
            "Inferred/interpolated %d stop-time rows; see adjustment audit.", len(adjustments)
        )
    logging.info("Selected %d source trips; %d scheduled runs.", len(selected), len(trips))
    return trips


class SpatialIndex:
    """Index 3-D Earth-surface coordinates in cubic cells; no GIS dependency."""

    def __init__(self, stops: Dict[str, Stop], cell_size: float) -> None:
        """Assign every stop to a cubic cell of the given size in meters."""
        self.stops = stops
        self.cell_size = max(cell_size, 1.0)
        self.cells: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        for stop in stops.values():
            self.cells[self.key(stop.lat, stop.lon)].append(stop.id)

    def key(self, lat: float, lon: float) -> Tuple[int, int, int]:
        """Map WGS84 coordinates onto a cell containing a point on a sphere."""
        lat, lon = math.radians(lat), math.radians(lon)
        xyz = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
        return tuple(math.floor(value * EARTH_RADIUS_M / self.cell_size) for value in xyz)

    def nearby(self, lat: float, lon: float, radius: float) -> List[Tuple[str, float]]:
        """Return all stops within a great-circle radius, with no nearest-N truncation."""
        center = self.key(lat, lon)
        extent = math.ceil(radius / self.cell_size)
        matches = []
        for x in range(center[0] - extent, center[0] + extent + 1):
            for y in range(center[1] - extent, center[1] + extent + 1):
                for z in range(center[2] - extent, center[2] + extent + 1):
                    for stop_id in self.cells.get((x, y, z), []):
                        stop = self.stops[stop_id]
                        meters = distance_m(lat, lon, stop.lat, stop.lon)
                        if meters <= radius + 1e-6:
                            matches.append((stop_id, meters))
        return sorted(matches, key=lambda item: (item[1], item[0]))


def transfer_rules(
    feed: Feed,
    stops: Dict[str, Stop],
    active: set,
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Expand ordinary station/stop transfer rules to active platform pairs."""
    descendants: Dict[str, set] = defaultdict(set)
    for stop_id in active:
        current, seen = stop_id, set()
        while current:
            if current in seen:
                raise ValueError(f"Cycle in parent_station hierarchy at {current}.")
            seen.add(current)
            descendants[current].add(stop_id)
            if current not in stops:
                raise ValueError(f"Missing parent station {current!r}.")
            current = stops[current].parent
    chosen: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    for row in feed.rows("transfers.txt"):
        qualifiers = ["from_route_id", "to_route_id", "from_trip_id", "to_trip_id"]
        if any(row.get(key) for key in qualifiers):
            raise ValueError("Route/trip-specific transfer rules are not supported by this router.")
        kind = int(row.get("transfer_type", "") or "0")
        if kind not in {0, 1, 2, 3}:
            raise ValueError("Linked-trip transfer types 4/5 are not supported by this router.")
        source, target = row.get("from_stop_id", ""), row.get("to_stop_id", "")
        if not source or not target or source not in stops or target not in stops:
            raise ValueError("transfers.txt references a blank or missing stop/station.")
        if kind == 2 and not row.get("min_transfer_time"):
            raise ValueError("transfer_type=2 requires min_transfer_time.")
        minimum = int(row.get("min_transfer_time", "") or "0") if kind == 2 else 0
        if minimum < 0:
            raise ValueError("Negative min_transfer_time.")
        if kind == 1:
            logging.warning(
                "Timed transfer %s -> %s: using schedule; no vehicle hold assumed.", source, target
            )
        specificity = int(source in active) + int(target in active)
        for first in descendants[source]:
            for second in descendants[target]:
                key = (first, second)
                old = chosen.get(key)
                if old and old[0] == specificity and old[1:] != (kind, minimum):
                    raise ValueError(f"Conflicting equally specific transfer rules for {key}.")
                if old is None or specificity > old[0]:
                    chosen[key] = (specificity, kind, minimum)
    return {key: (value[1], value[2]) for key, value in chosen.items()}


def build_walk_links(
    index: SpatialIndex,
    rules: Dict[Tuple[str, str], Tuple[int, int]],
) -> Dict[str, List[WalkLink]]:
    """Build single-step transfer links; forbidden links are never added."""
    result = {}
    buffer_seconds = math.ceil(MIN_TRANSFER_MINUTES * 60)
    for stop in index.stops.values():
        links = []
        nearby = index.nearby(stop.lat, stop.lon, MAX_TRANSFER_WALK_MILES * METERS_PER_MILE)
        for target, meters in nearby:
            kind, minimum = rules.get((stop.id, target), (0, 0))
            if kind == 3:
                continue
            seconds = walk_seconds(meters)
            links.append(WalkLink(target, meters, seconds, max(seconds + buffer_seconds, minimum)))
        result[stop.id] = links
    return result


class StationWalking:
    """Join surface walks to directed station paths without teleporting to platforms.

    Configured mileage caps apply to each outdoor surface connection. Station
    lengths and traversal times are additional. This is not wheelchair routing.
    """

    def __init__(self, feed: Feed, stops: Dict[str, Stop], trips: List[Trip], output: Path) -> None:
        """Build directed pathway graphs and entrance/platform connections."""
        self.stops = stops
        self.active = {event.stop for trip in trips for event in trip.events}
        graph: Dict[str, List[WalkStep]] = defaultdict(list)
        pathway_rows = unique_rows(feed.rows("pathways.txt"), "pathway_id", "pathways")
        exported = []
        for key, row in pathway_rows.items():
            first, last = row["from_stop_id"], row["to_stop_id"]
            if first not in stops or last not in stops:
                raise ValueError(f"Pathway {key} references an unknown/unlocated node.")
            if stops[first].location_type == "1" or stops[last].location_type == "1":
                raise ValueError(f"Pathway {key} must not end at a parent station.")
            mode, bidirectional = row.get("pathway_mode"), row.get("is_bidirectional")
            source_bidirectional = bidirectional
            if mode not in {str(n) for n in range(1, 8)} or bidirectional not in {"0", "1"}:
                raise ValueError(f"Invalid pathway mode/direction for {key}.")
            if mode == "7" and bidirectional == "1":
                bidirectional = "0"
                logging.warning("Exit gate %s marked bidirectional: using forward direction.", key)
            # Use feed durations for stairs, elevators, escalators and gates.
            # Do not replace missing durations with an unverified straight-line shortcut.
            if not row.get("traversal_time") or not row.get("length"):
                raise ValueError(f"Pathway {key} requires traversal_time and length in this model.")
            seconds, meters = int(row["traversal_time"]), float(row["length"])
            if seconds < 0 or not math.isfinite(meters) or meters < 0:
                raise ValueError(f"Invalid pathway time/length for {key}.")
            original_seconds = seconds
            if seconds == 0:
                if mode != "1" or not DERIVE_ZERO_TIME_WALKWAYS:
                    raise ValueError(f"Nonpositive traversal_time for pathway {key}.")
                seconds = max(1, walk_seconds(meters))
                logging.warning(
                    "Zero-time walkway %s: derived %d second(s) from length.", key, seconds
                )
            step = WalkStep("stop:" + first, "stop:" + last, meters, seconds, key)
            graph[first].append(step)
            if bidirectional == "1":
                graph[last].append(replace(step, origin=step.destination, destination=step.origin))
            exported.append(
                {
                    "pathway_id": key,
                    "from_stop_id": first,
                    "to_stop_id": last,
                    "pathway_mode": mode,
                    "is_bidirectional": bidirectional,
                    "length_m": meters,
                    "traversal_seconds": seconds,
                    "source_traversal_seconds": original_seconds,
                    "source_is_bidirectional": source_bidirectional,
                    "time_basis": "FEED" if original_seconds else "LENGTH_WALK_SPEED",
                }
            )
        write_csv(
            output / "station_pathways.csv",
            exported,
            [
                "pathway_id",
                "from_stop_id",
                "to_stop_id",
                "pathway_mode",
                "is_bidirectional",
                "length_m",
                "traversal_seconds",
                "source_traversal_seconds",
                "time_basis",
                "source_is_bidirectional",
            ],
        )
        nodes = set(graph) | {s.destination[5:] for steps in graph.values() for s in steps}
        rail = (
            {e.stop for t in trips if t.route_type == "1" for e in t.events}
            if REQUIRE_RAIL_PATHWAYS
            else set()
        )
        platforms = (self.active & nodes) | rail
        entrances = {key for key in nodes if stops[key].location_type == "2"}
        self.paths = {key: self.shortest_paths(graph, key) for key in platforms | entrances}
        self.inbound: Dict[str, Dict[str, Tuple[WalkStep, ...]]] = defaultdict(dict)
        self.outbound: Dict[str, Dict[str, Tuple[WalkStep, ...]]] = defaultdict(dict)
        for stop in self.active - platforms:
            self.inbound[stop][stop] = ()
            self.outbound[stop][stop] = ()
        for entrance in entrances:
            for platform in platforms:
                if platform in self.paths[entrance]:
                    self.inbound[entrance][platform] = self.paths[entrance][platform]
                if entrance in self.paths[platform]:
                    self.outbound[platform][entrance] = self.paths[platform][entrance]
        unreachable = [
            key
            for key in platforms
            if not self.outbound[key] or not any(key in value for value in self.inbound.values())
        ]
        if unreachable:
            raise ValueError(
                "No entrance/exit pathway for active platforms: " + ", ".join(sorted(unreachable))
            )
        terminals = set(self.inbound) | {s for items in self.outbound.values() for s in items}
        radius = max(MAX_ACCESS_WALK_MILES, MAX_TRANSFER_WALK_MILES, MAX_EGRESS_WALK_MILES)
        self.index = SpatialIndex({s: stops[s] for s in terminals}, radius * METERS_PER_MILE)
        self.platforms = platforms
        logging.info(
            "Station network: %d platform stops, %d entrances, %d pathways.",
            len(platforms),
            len(entrances),
            len(pathway_rows),
        )

    @staticmethod
    def shortest_paths(
        graph: Dict[str, List[WalkStep]], source: str
    ) -> Dict[str, Tuple[WalkStep, ...]]:
        """Find fastest directed paths, preferring shorter distance for equal times."""
        costs = {source: (0, 0.0)}
        paths: Dict[str, Tuple[WalkStep, ...]] = {source: ()}
        queue = [(0, 0.0, source)]
        while queue:
            seconds, meters, current = heapq.heappop(queue)
            if costs[current] != (seconds, meters):
                continue
            for step in graph.get(current, []):
                target = step.destination[5:]
                cost = (seconds + step.seconds, meters + step.distance_m)
                if cost < costs.get(target, (math.inf, math.inf)):
                    costs[target] = cost
                    paths[target] = paths[current] + (step,)
                    heapq.heappush(queue, (*cost, target))
        return paths

    @staticmethod
    def cost(steps: Tuple[WalkStep, ...]) -> Tuple[int, float]:
        """Return total seconds then meters for a complete physical connection."""
        return sum(s.seconds for s in steps), sum(s.distance_m for s in steps)

    def endpoints(self, places: Dict[str, Place], entering: bool) -> Dict[str, List[EndpointLink]]:
        """Build place-to-platform access or platform-to-place egress links."""
        limit = MAX_ACCESS_WALK_MILES if entering else MAX_EGRESS_WALK_MILES
        outward: Dict[str, Dict[str, Tuple[WalkStep, ...]]] = defaultdict(dict)
        if not entering:
            for platform, terminals in self.outbound.items():
                for terminal, steps in terminals.items():
                    outward[terminal][platform] = steps
        result = {}
        for key, place in places.items():
            best = {}
            nearby = self.index.nearby(place.lat, place.lon, limit * METERS_PER_MILE)
            for terminal, meters in nearby:
                first, last = "place:" + key, "stop:" + terminal
                if not entering:
                    first, last = last, first
                surface = WalkStep(first, last, meters, walk_seconds(meters))
                for platform, steps in (self.inbound if entering else outward)[terminal].items():
                    path = (surface,) + steps if entering else steps + (surface,)
                    seconds, length = self.cost(path)
                    candidate = EndpointLink(platform, length, seconds, path)
                    old = best.get(platform)
                    if old is None or (seconds, length) < (old.seconds, old.distance_m):
                        best[platform] = candidate
            result[key] = [best[s] for s in sorted(best)]
        return result

    def transfers(self, rules: Dict[Tuple[str, str], Tuple[int, int]]) -> Dict[str, List[WalkLink]]:
        """Combine one surface walk with any required exit and entry pathways."""
        result = {}
        buffer = math.ceil(MIN_TRANSFER_MINUTES * 60)
        radius = MAX_TRANSFER_WALK_MILES * METERS_PER_MILE
        for source in sorted(self.active):
            candidates = {source: ()}
            for target, steps in self.paths.get(source, {}).items():
                if target in self.active:
                    candidates[target] = steps
            for exit_stop, exits in self.outbound[source].items():
                location = self.stops[exit_stop]
                for entry_stop, meters in self.index.nearby(location.lat, location.lon, radius):
                    walk = WalkStep(
                        "stop:" + exit_stop, "stop:" + entry_stop, meters, walk_seconds(meters)
                    )
                    for target, entries in self.inbound[entry_stop].items():
                        steps = exits + (walk,) + entries
                        if target not in candidates or self.cost(steps) < self.cost(
                            candidates[target]
                        ):
                            candidates[target] = steps
            links = []
            for target, steps in sorted(candidates.items()):
                kind, minimum = rules.get((source, target), (0, 0))
                if kind == 3:
                    continue
                seconds, meters = self.cost(steps)
                links.append(
                    WalkLink(target, meters, seconds, max(seconds + buffer, minimum), steps)
                )
            result[source] = links
        return result


class IndexedTrips(list):
    """Index pickup departures so each routing round scans only boardable trips."""

    def __init__(self, trips: List[Trip]) -> None:
        """Store trips and index their pickup departures by stop."""
        super().__init__(trips)
        events = defaultdict(list)
        for index, trip in enumerate(trips):
            for event in trip.events:
                if event.pickup:
                    events[event.stop].append((event.departure, index))
        self.by_stop = {}
        for key, values in events.items():
            values.sort()
            self.by_stop[key] = ([v[0] for v in values], [v[1] for v in values])

    def eligible(self, boarding: Dict[str, BoardLabel], deadline: int) -> Iterator[Trip]:
        """Yield every trip with at least one feasible boarding, in stable order."""
        indices = set()
        for stop, label in boarding.items():
            times, trips = self.by_stop.get(stop, ([], []))
            indices.update(trips[bisect_left(times, label.ready) : bisect_right(times, deadline)])
        return (self[index] for index in sorted(indices))


def resolve_places(stops: Dict[str, Stop]) -> Tuple[Dict[str, Place], Dict[str, str]]:
    """Resolve only explicit coordinates or unique exact stop IDs/codes."""
    places, errors = {}, {}
    for key, config in LOCATIONS.items():
        try:
            has_coordinates = config.get("lat") is not None or config.get("lon") is not None
            methods = (
                int(has_coordinates)
                + int(bool(config.get("stop_code")))
                + int(bool(config.get("stop_id")))
            )
            if methods != 1:
                raise ValueError("Set exactly one method: lat/lon, stop_code, or stop_id.")
            if has_coordinates:
                lat, lon = valid_coordinates(config["lat"], config["lon"])
            else:
                field = "code" if config.get("stop_code") else "id"
                value = config.get("stop_code") or config.get("stop_id")
                if not isinstance(value, str):
                    raise ValueError("Keep stop IDs and codes in quotes, including leading zeros.")
                matches = [stop for stop in stops.values() if getattr(stop, field) == value]
                if len(matches) != 1:
                    raise ValueError(
                        f"{field}={value!r} matches {len(matches)} stops; choose an exact stop_id."
                    )
                lat, lon = matches[0].lat, matches[0].lon
            places[key] = Place(key, config["name"], lat, lon)
        except (ValueError, TypeError) as exc:
            errors[key] = str(exc)
            logging.warning("Unresolved location %s: %s", key, exc)
    return places, errors


def add_walk(
    label: Label,
    source: str,
    target: str,
    meters: float,
    seconds: Optional[int] = None,
    pathway_id: str = "",
) -> Label:
    """Append a walk, including zero-distance endpoint walks for traceability."""
    end = label.time + (walk_seconds(meters) if seconds is None else seconds)
    leg = Leg(
        "WALK",
        source,
        target,
        label.time,
        end,
        meters,
        walk_basis="GTFS_PATHWAY" if pathway_id else "STRAIGHT_LINE",
        pathway_id=pathway_id,
    )
    return Label(end, label.legs + (leg,))


def append_steps(label: Label, steps: Tuple[WalkStep, ...]) -> Label:
    """Append physical station/surface segments with their actual modeled durations."""
    for step in steps:
        label = add_walk(
            label, step.origin, step.destination, step.distance_m, step.seconds, step.pathway_id
        )
    return label


def endpoint_link(item: Any, place: str, entering: bool) -> EndpointLink:
    """Accept legacy (stop, meters) links as well as complete station-aware paths."""
    if isinstance(item, EndpointLink):
        return item
    stop, meters = item
    first, last = "place:" + place, "stop:" + stop
    if not entering:
        first, last = last, first
    seconds = walk_seconds(meters)
    return EndpointLink(stop, meters, seconds, (WalkStep(first, last, meters, seconds),))


def scan_trips(
    trips: List[Trip],
    boarding: Dict[str, BoardLabel],
    deadline: int,
) -> Dict[str, Label]:
    """Find earliest alightings after exactly one more boarding.

    Boarding labels are fixed throughout this scan. An earlier feasible position on
    a given run can reach every later event on that run, including no-alight stops.
    Staying on a trip bypasses pickup restrictions and includes intermediate dwell.
    """
    arrivals: Dict[str, Label] = {}
    selected = trips.eligible(boarding, deadline) if isinstance(trips, IndexedTrips) else trips
    for trip in selected:
        board_event: Optional[Event] = None
        prefix: Tuple[Leg, ...] = ()
        board_walk = math.inf
        for event in trip.events:
            if event.arrival > deadline:
                break
            if board_event is not None:
                old = arrivals.get(event.stop)
                better = (
                    old is None
                    or event.arrival < old.time
                    or (
                        event.arrival == old.time
                        and board_walk < sum(leg.distance_m for leg in old.legs)
                    )
                )
                if event.dropoff and better:
                    ride = Leg(
                        "RIDE",
                        "stop:" + board_event.stop,
                        "stop:" + event.stop,
                        board_event.departure,
                        event.arrival,
                        route=trip.route,
                        route_id=trip.route_id,
                        trip_id=trip.id,
                        source_trip_id=trip.source_id,
                        direction=trip.direction,
                        headsign=trip.headsign,
                        feed_id=trip.feed_id,
                        service_id=trip.service_id,
                        route_type=trip.route_type,
                    )
                    arrivals[event.stop] = Label(event.arrival, prefix + (ride,))
            if event.pickup and event.departure <= deadline:
                candidate = boarding.get(event.stop)
                if candidate and candidate.ready <= event.departure:
                    candidate_walk = sum(leg.distance_m for leg in candidate.label.legs)
                    if board_event is not None and candidate_walk >= board_walk:
                        continue
                    board_event = event
                    board_walk = candidate_walk
                    prefix = candidate.label.legs
                    if event.departure > candidate.label.time:
                        prefix += (
                            Leg(
                                "WAIT",
                                "stop:" + event.stop,
                                "stop:" + event.stop,
                                candidate.label.time,
                                event.departure,
                            ),
                        )
    return arrivals


def transfer_boarding(
    arrivals: Dict[str, Label],
    links: Dict[str, List[WalkLink]],
) -> Dict[str, BoardLabel]:
    """Construct next-round labels with walking and both transfer-time requirements."""
    boarding: Dict[str, BoardLabel] = {}
    for source, label in arrivals.items():
        for link in links[source]:
            ready = label.time + link.min_elapsed
            old = boarding.get(link.to_stop)
            meters = sum(leg.distance_m for leg in label.legs) + link.distance_m
            if (
                old is None
                or ready < old.ready
                or (ready == old.ready and meters < sum(leg.distance_m for leg in old.label.legs))
            ):
                walked = (
                    append_steps(label, link.steps)
                    if link.steps
                    else add_walk(
                        label,
                        "stop:" + source,
                        "stop:" + link.to_stop,
                        link.distance_m,
                        link.seconds,
                    )
                )
                boarding[link.to_stop] = BoardLabel(walked, ready)
    return boarding


def route_origin(
    origin: Place,
    destinations: Dict[str, Place],
    departure: int,
    trips: List[Trip],
    access: Dict[str, List[Tuple[str, float]]],
    egress: Dict[str, List[Tuple[str, float]]],
    links: Dict[str, List[WalkLink]],
) -> Dict[str, Tuple[Label, int]]:
    """Route one departure from one origin to all requested destinations."""
    deadline = departure + math.floor(MAX_JOURNEY_MINUTES * 60)
    answers: Dict[str, Tuple[Label, int]] = {}
    start = Label(departure, ())
    for key, destination in destinations.items():
        meters = distance_m(origin.lat, origin.lon, destination.lat, destination.lon)
        if ALLOW_WALK_ONLY and meters <= MAX_WALK_ONLY_MILES * METERS_PER_MILE:
            label = add_walk(start, "place:" + origin.id, "place:" + key, meters)
            if label.time <= deadline:
                answers[key] = (label, 0)
    boarding = {}
    for item in access[origin.id]:
        connection = endpoint_link(item, origin.id, True)
        label = append_steps(start, connection.steps)
        ready = label.time + math.ceil(INITIAL_BOARDING_BUFFER_MINUTES * 60)
        boarding[connection.stop] = BoardLabel(label, ready)
    candidates = (
        trips
        if isinstance(trips, IndexedTrips)
        else [
            trip
            for trip in trips
            if trip.events[-1].arrival >= departure and trip.events[0].departure <= deadline
        ]
    )
    for count in range(1, MAX_TRANSFERS + 2):
        if not boarding:
            break
        arrivals = scan_trips(candidates, boarding, deadline)
        for key in destinations:
            for item in egress[key]:
                connection = endpoint_link(item, key, False)
                label = arrivals.get(connection.stop)
                if label is None:
                    continue
                end = label.time + connection.seconds
                old = answers.get(key)
                walking = sum(leg.distance_m for leg in label.legs) + connection.distance_m
                better = (
                    old is None
                    or end < old[0].time
                    or (
                        end == old[0].time
                        and count == old[1]
                        and walking < sum(leg.distance_m for leg in old[0].legs)
                    )
                )
                if end <= deadline and better:
                    answers[key] = (append_steps(label, connection.steps), count)
        boarding = transfer_boarding(arrivals, links) if count <= MAX_TRANSFERS else {}
    return answers


def departures(start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[int]:
    """Return sorted unique departure samples, with an inclusive range end."""
    if start_time is None and end_time is None and EXPLICIT_DEPARTURE_TIMES:
        return sorted({parse_time(value) for value in EXPLICIT_DEPARTURE_TIMES})
    start = parse_time(DEPARTURE_START if start_time is None else start_time)
    end = parse_time(DEPARTURE_END if end_time is None else end_time)
    if end < start:
        raise ValueError("DEPARTURE_END must be >= start; use 25:00 for 1 AM next service day.")
    if not isinstance(DEPARTURE_STEP_MINUTES, int) or DEPARTURE_STEP_MINUTES <= 0:
        raise ValueError("DEPARTURE_STEP_MINUTES must be a positive integer.")
    return list(range(start, end + 1, DEPARTURE_STEP_MINUTES * 60))


def configured_queries() -> List[Tuple[str, List[Tuple[str, str, str]], List[int]]]:
    """Resolve period-specific OD pairs, samples and optional reverse directions."""
    batches = []
    known = {pair[0] for pair in OD_PAIRS}
    if USE_PERIODS:
        if not PERIODS:
            raise ValueError("USE_PERIODS=True requires at least one PERIODS entry.")
        assigned = set()
        for label, (start, end, identifiers) in PERIODS.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Period labels must be nonempty strings.")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"Duplicate OD IDs in period {label}.")
            unknown = set(identifiers) - known
            if unknown:
                raise ValueError(f"Unknown OD IDs in period {label}: {sorted(unknown)}")
            selected = [pair for pair in OD_PAIRS if pair[0] in identifiers]
            assigned.update(identifiers)
            if selected:
                batches.append((label, selected, departures(start, end)))
        if known - assigned:
            raise ValueError(f"OD IDs missing from PERIODS: {sorted(known - assigned)}")
    else:
        batches.append(("Custom", list(OD_PAIRS), departures()))
    if INCLUDE_REVERSE_PAIRS:
        expanded = []
        for label, pairs, times in batches:
            pairs = pairs + [(pid + "_reverse", target, origin) for pid, origin, target in pairs]
            if len({pair[0] for pair in pairs}) != len(pairs):
                raise ValueError("Reverse pair IDs conflict with configured pair IDs.")
            expanded.append((label, pairs, times))
        batches = expanded
    return batches


def endpoint(key: str, stops: Dict[str, Stop]) -> Row:
    """Expand a leg endpoint into its public name and original stop identifiers."""
    kind, value = key.split(":", 1)
    if kind == "stop":
        stop = stops[value]
        return {
            "name": stop.name,
            "stop_id": stop.id,
            "stop_code": stop.code,
            "feed_id": stop.feed_id,
            "source_stop_id": stop.source_id,
        }
    return {"name": LOCATIONS[value]["name"], "stop_id": "", "stop_code": ""}


def result_rows(
    pairs: List[Tuple[str, str, str]],
    places: Dict[str, Place],
    errors: Dict[str, str],
    times: List[int],
    trips: List[Trip],
    access: Dict[str, List[Tuple[str, float]]],
    egress: Dict[str, List[Tuple[str, float]]],
    links: Dict[str, List[WalkLink]],
    stops: Dict[str, Stop],
) -> Tuple[List[Row], List[Row]]:
    """Batch by origin/departure and keep one auditable itinerary per OD result."""
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair[1]].append(pair)
    results, legs = [], []
    for origin, origin_pairs in grouped.items():
        destinations = {target: places[target] for _, _, target in origin_pairs if target in places}
        for departure in times:
            answers = {}
            if origin in places:
                answers = route_origin(
                    places[origin], destinations, departure, trips, access, egress, links
                )
            for pair_id, _, target in origin_pairs:
                row = {
                    "pair_id": pair_id,
                    "origin": LOCATIONS[origin]["name"],
                    "destination": LOCATIONS[target]["name"],
                    "service_date": SERVICE_DATE or "",
                    "service_id": "" if SERVICE_DATE else SERVICE_ID,
                    "requested_departure": format_time(departure),
                    "status": "NO_PATH_WITHIN_LIMITS",
                    "notes": "",
                }
                if origin in errors or target in errors:
                    row.update(
                        status="UNRESOLVED_LOCATION",
                        notes="; ".join(
                            f"{key}: {errors[key]}" for key in (origin, target) if key in errors
                        ),
                    )
                elif target not in answers:
                    if not access[origin]:
                        row["notes"] += "No selected-service stops within access limit. "
                    if not egress[target]:
                        row["notes"] += "No selected-service stops within egress limit. "
                    row["notes"] += (
                        "No feasible path under the configured service/time/walk/transfer limits."
                    )
                else:
                    label, boardings = answers[target]
                    durations = Counter()
                    for leg in label.legs:
                        durations[leg.mode] += leg.end - leg.start
                    assert sum(durations.values()) == label.time - departure
                    row.update(
                        {
                            "status": "OK_TRANSIT" if boardings else "OK_WALK_ONLY",
                            "arrival": format_time(label.time),
                            "total_minutes": round((label.time - departure) / 60, 3),
                            "walk_minutes": round(durations["WALK"] / 60, 3),
                            "wait_minutes": round(durations["WAIT"] / 60, 3),
                            "ride_minutes": round(durations["RIDE"] / 60, 3),
                            "station_pathway_minutes": round(
                                sum(leg.end - leg.start for leg in label.legs if leg.pathway_id)
                                / 60,
                                3,
                            ),
                            "walk_miles": round(
                                sum(item.distance_m for item in label.legs) / METERS_PER_MILE, 4
                            ),
                            "boardings": boardings,
                            "transfers": max(0, boardings - 1),
                            "routes": " > ".join(
                                item.route for item in label.legs if item.mode == "RIDE"
                            ),
                            "feeds": " > ".join(
                                item.feed_id for item in label.legs if item.mode == "RIDE"
                            ),
                            "used_service_ids": "; ".join(
                                sorted(
                                    {item.service_id for item in label.legs if item.mode == "RIDE"}
                                )
                            ),
                        }
                    )
                    for number, leg in enumerate(label.legs, 1):
                        record = {
                            "pair_id": pair_id,
                            "requested_departure": format_time(departure),
                            "leg_number": number,
                            **asdict(leg),
                            "start": format_time(leg.start),
                            "end": format_time(leg.end),
                            "minutes": round((leg.end - leg.start) / 60, 3),
                        }
                        for side, point in [("from", leg.origin), ("to", leg.destination)]:
                            record.update(
                                {
                                    f"{side}_{key}": value
                                    for key, value in endpoint(point, stops).items()
                                }
                            )
                        legs.append(record)
                results.append(row)
        logging.info(
            "Completed %s for %d departure samples.", LOCATIONS[origin]["name"], len(times)
        )
    return results, legs


def summarize(results: List[Row], pairs: List[Tuple[str, str, str]]) -> List[Row]:
    """Summarize successful departure samples while explicitly counting failures."""
    output = []
    for pair_id, origin, target in pairs:
        subset = [row for row in results if row["pair_id"] == pair_id]
        found = [row for row in subset if row["status"].startswith("OK_")]
        values = sorted(row["total_minutes"] for row in found)
        row = {
            "pair_id": pair_id,
            "origin": LOCATIONS[origin]["name"],
            "destination": LOCATIONS[target]["name"],
            "samples": len(subset),
            "found": len(found),
            "not_found_or_unresolved": len(subset) - len(found),
            "transit_results": sum(item["status"] == "OK_TRANSIT" for item in found),
            "walk_only_results": sum(item["status"] == "OK_WALK_ONLY" for item in found),
            "statistic_scope": "Successful sampled departures only; not operating reliability",
        }
        if values:
            fastest = min(found, key=lambda item: item["total_minutes"])
            row.update(
                min_minutes=values[0],
                mean_minutes=round(statistics.mean(values), 3),
                median_minutes=round(statistics.median(values), 3),
                p90_departure_sample_minutes=values[math.ceil(len(values) * 0.9) - 1],
                max_minutes=values[-1],
                fastest_requested_departure=fastest["requested_departure"],
            )
            for field in (
                "walk_minutes",
                "wait_minutes",
                "ride_minutes",
                "walk_miles",
                "station_pathway_minutes",
            ):
                row["mean_" + field] = round(statistics.mean(item[field] for item in found), 3)
            row["route_counts"] = json.dumps(dict(Counter(item["routes"] for item in found)))
        row["status"] = (
            "UNRESOLVED_LOCATION"
            if any(x["status"] == "UNRESOLVED_LOCATION" for x in subset)
            else "NO_PATH_WITHIN_LIMITS"
            if not found
            else "PARTIAL_PATH_COVERAGE"
            if len(found) < len(subset)
            else "OK"
        )
        output.append(row)
    return output


def export_benchmark_reference(output: Path) -> Dict[str, Row]:
    """Audit the supplied assumed-speed/half-headway method without altering routing."""
    totals, legs = {}, []
    for pair_id, data in BENCHMARK_ESTIMATES.items():
        period, reported, access_miles, egress_miles, metro_uses, source_legs = data
        ride_total = wait_total = displayed_wait_total = station_total = 0.0
        for number, (route, miles, speed, headway, displayed_wait, passed) in enumerate(
            source_legs, 1
        ):
            metro = route in BENCHMARK_METRO_ROUTES
            effective_speed = BENCHMARK_METRO_SPEED_MPH if metro else speed
            effective_headway = BENCHMARK_METRO_HEADWAYS[period] if metro else headway
            if effective_speed <= 0 or effective_headway < 0 or miles < 0 or passed < 0:
                raise ValueError(f"Invalid benchmark reference inputs for pair {pair_id}.")
            ride = miles / effective_speed * 60
            wait = effective_headway / 2
            stations = passed * BENCHMARK_METRO_STATION_MINUTES if metro else 0.0
            ride_total += ride
            wait_total += wait
            displayed_wait_total += displayed_wait
            station_total += stations
            legs.append(
                {
                    "benchmark_pair_id": pair_id,
                    "reference_period": period,
                    "leg_number": number,
                    "route": route,
                    "distance_miles": miles,
                    "assumed_speed_mph": effective_speed,
                    "headway_minutes": effective_headway,
                    "displayed_wait_minutes": displayed_wait,
                    "half_headway_minutes": wait,
                    "calculated_ride_minutes": round(ride, 6),
                    "intermediate_metro_stations": passed,
                    "metro_station_minutes": stations,
                    "ride_plus_wait_plus_stations_minutes": round(ride + wait + stations, 6),
                }
            )
        if BENCHMARK_WALK_SPEED_MPH <= 0:
            raise ValueError("BENCHMARK_WALK_SPEED_MPH must be positive.")
        walk = (access_miles + egress_miles) / BENCHMARK_WALK_SPEED_MPH * 60
        metro_access = metro_uses * BENCHMARK_METRO_ENTRY_EXIT_MINUTES
        total = ride_total + wait_total + station_total + walk + metro_access
        totals[pair_id] = {
            "benchmark_pair_id": pair_id,
            "reference_period": period,
            "reference_routes": " > ".join(leg[0] for leg in source_legs),
            "reported_total_minutes": reported,
            "ride_minutes": round(ride_total, 6),
            "half_headway_wait_minutes": wait_total,
            "displayed_wait_minutes": displayed_wait_total,
            "walk_minutes": walk,
            "metro_station_minutes": station_total,
            "metro_entry_exit_minutes": metro_access,
            "recomputed_total_minutes": round(total, 6),
            "recomputed_minus_reported_minutes": round(total - reported, 6),
            "recomputed_using_displayed_wait_minutes": round(
                total - wait_total + displayed_wait_total, 6
            ),
            "note": "Source distances/headways/waits are rounded; reported total retained.",
        }
    write_csv(output / "benchmark_method_legs.csv", legs, list(legs[0]) if legs else [])
    write_csv(
        output / "benchmark_method_totals.csv",
        totals.values(),
        list(next(iter(totals.values()))) if totals else [],
    )
    return totals


def export_benchmark_comparison(
    output: Path,
    summaries: List[Row],
    references: Dict[str, Row],
    routes: set,
) -> None:
    """Compare period means with original totals, retaining endpoint and method caveats."""
    rows = []
    for summary in summaries:
        pair_id = summary["pair_id"]
        benchmark_id = PAIR_BENCHMARK_IDS.get(pair_id, pair_id)
        reference = references.get(benchmark_id)
        row = {
            key: summary.get(key, "")
            for key in (
                "pair_id",
                "period",
                "origin",
                "destination",
                "status",
                "samples",
                "found",
                "mean_minutes",
                "mean_walk_minutes",
                "mean_wait_minutes",
                "mean_ride_minutes",
                "route_counts",
            )
        }
        row["comparison_note"] = "Different methods and possibly different endpoints/itineraries."
        if reference:
            reported = reference["reported_total_minutes"]
            row.update(
                benchmark_pair_id=benchmark_id,
                benchmark_reference_period=reference["reference_period"],
                benchmark_reported_minutes=reported,
                benchmark_recomputed_minutes=reference["recomputed_total_minutes"],
                benchmark_routes=reference["reference_routes"],
                gtfs_mean_minus_reported_minutes=(
                    round(summary["mean_minutes"] - reported, 3)
                    if "mean_minutes" in summary
                    else ""
                ),
                reference_routes_not_in_selected_service="; ".join(
                    name
                    for name in reference["reference_routes"].split(" > ")
                    if name not in routes
                ),
            )
        else:
            row["comparison_note"] = "No original benchmark for this configured OD direction."
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output / "benchmark_comparison.csv", rows, fields)


def export_network(trips: List[Trip], links: Dict[str, List[WalkLink]], output: Path) -> None:
    """Export timed transit arcs and physical transfer links without graph libraries."""
    segments = (
        {
            "trip_id": trip.id,
            "source_trip_id": trip.source_id,
            "route": trip.route,
            "feed_id": trip.feed_id,
            "service_id": trip.service_id,
            "route_type": trip.route_type,
            "route_id": trip.route_id,
            "direction": trip.direction,
            "from_stop_id": first.stop,
            "to_stop_id": second.stop,
            "from_sequence": first.sequence,
            "to_sequence": second.sequence,
            "departure": format_time(first.departure),
            "arrival": format_time(second.arrival),
            "segment_minutes": round((second.arrival - first.departure) / 60, 3),
            "pickup_at_from": first.pickup,
            "dropoff_at_to": second.dropoff,
        }
        for trip in trips
        for first, second in zip(trip.events, trip.events[1:])
    )
    write_csv(
        output / "network_transit_segments.csv",
        segments,
        [
            "trip_id",
            "source_trip_id",
            "route",
            "route_id",
            "direction",
            "from_stop_id",
            "to_stop_id",
            "from_sequence",
            "to_sequence",
            "departure",
            "arrival",
            "segment_minutes",
            "pickup_at_from",
            "dropoff_at_to",
            "feed_id",
            "service_id",
            "route_type",
        ],
    )
    write_csv(
        output / "network_walk_links.csv",
        (
            {
                "from_stop_id": source,
                "to_stop_id": link.to_stop,
                "distance_m": round(link.distance_m, 3),
                "walk_seconds": link.seconds,
                "minimum_arrival_to_boarding_seconds": link.min_elapsed,
                "pathway_ids": "; ".join(s.pathway_id for s in link.steps if s.pathway_id),
            }
            for source, items in links.items()
            for link in items
        ),
        [
            "from_stop_id",
            "to_stop_id",
            "distance_m",
            "walk_seconds",
            "minimum_arrival_to_boarding_seconds",
            "pathway_ids",
        ],
    )


def validate_config() -> None:
    """Fail on invalid numeric limits and ambiguous OD identifiers."""
    positive = {"WALK_SPEED_MPH": WALK_SPEED_MPH, "MAX_JOURNEY_MINUTES": MAX_JOURNEY_MINUTES}
    nonnegative = {
        "MAX_ACCESS_WALK_MILES": MAX_ACCESS_WALK_MILES,
        "MAX_TRANSFER_WALK_MILES": MAX_TRANSFER_WALK_MILES,
        "MAX_EGRESS_WALK_MILES": MAX_EGRESS_WALK_MILES,
        "MAX_WALK_ONLY_MILES": MAX_WALK_ONLY_MILES,
        "MIN_TRANSFER_MINUTES": MIN_TRANSFER_MINUTES,
        "INITIAL_BOARDING_BUFFER_MINUTES": INITIAL_BOARDING_BUFFER_MINUTES,
    }
    for name, value in {**positive, **nonnegative}.items():
        if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
            raise ValueError(f"Invalid {name}: {value}.")
    if not isinstance(MAX_TRANSFERS, int) or MAX_TRANSFERS < 0:
        raise ValueError("MAX_TRANSFERS must be a nonnegative integer.")
    if SERVICE_ID is not None and (not isinstance(SERVICE_ID, str) or not SERVICE_ID):
        raise ValueError("SERVICE_ID must be one nonempty quoted string, or None for discovery.")
    if len({pair[0] for pair in OD_PAIRS}) != len(OD_PAIRS):
        raise ValueError("OD pair IDs must be unique.")
    for _, origin, target in OD_PAIRS:
        if origin not in LOCATIONS or target not in LOCATIONS:
            raise ValueError("OD_PAIRS references an unknown location key.")


def run(output: Path) -> int:
    """Build a single-service-day network and export all configured OD queries."""
    validate_config()
    batches = configured_queries()
    references = export_benchmark_reference(output)
    (output / "methods.txt").write_text(__doc__ or "", encoding="utf-8")
    feed = CombinedFeed(GTFS_FEEDS) if USE_MULTIPLE_FEEDS else Feed(GTFS_PATH)
    try:
        trip_rows = unique_rows(
            feed.rows("trips.txt", ["trip_id", "route_id", "service_id"]), "trip_id", "trips"
        )
        stops = load_stops(feed)
        export_inventories(feed, trip_rows, stops, output)
        if DISCOVERY_ONLY or (SERVICE_DATE is None and SERVICE_ID is None):
            logging.warning(
                "Discovery complete. Select SERVICE_DATE and resolve LOCATIONS; see %s", output
            )
            return 0 if DISCOVERY_ONLY else 2
        services = select_services(feed, trip_rows, output)
        logging.warning(
            "Straight-line walking ignores barriers/access restrictions; "
            "scheduled times are estimates."
        )
        trips = IndexedTrips(load_trips(feed, trip_rows, stops, output, services))
        served: Dict[str, set] = defaultdict(set)
        for trip in trips:
            for event in trip.events:
                served[event.stop].add(trip.route)
        export_inventories(feed, trip_rows, stops, output, served, services)
        active = {key: stops[key] for key in served}
        walking = StationWalking(feed, stops, trips, output)
        links = walking.transfers(transfer_rules(feed, stops, set(active)))
        logging.info(
            "Network: %d served stops, %d directed transfer links.",
            len(active),
            sum(map(len, links.values())),
        )
        places, errors = resolve_places(stops)
        access = walking.endpoints(places, entering=True)
        egress = walking.endpoints(places, entering=False)
        write_csv(
            output / "endpoint_connections.csv",
            (
                {
                    "location_id": place,
                    "kind": kind,
                    "stop_id": link.stop,
                    "distance_m": round(link.distance_m, 3),
                    "seconds": link.seconds,
                    "pathway_ids": "; ".join(s.pathway_id for s in link.steps if s.pathway_id),
                }
                for kind, connections in [("access", access), ("egress", egress)]
                for place, items in connections.items()
                for link in items
            ),
            ["location_id", "kind", "stop_id", "distance_m", "seconds", "pathway_ids"],
        )
        write_csv(
            output / "resolved_locations.csv",
            (
                {
                    "location_id": key,
                    "name": config["name"],
                    "latitude": places[key].lat if key in places else "",
                    "longitude": places[key].lon if key in places else "",
                    "access_stops": len(access.get(key, [])),
                    "egress_stops": len(egress.get(key, [])),
                    "status": "RESOLVED" if key in places else errors[key],
                    "note": config.get("note", ""),
                }
                for key, config in LOCATIONS.items()
            ),
            [
                "location_id",
                "name",
                "latitude",
                "longitude",
                "access_stops",
                "egress_stops",
                "status",
                "note",
            ],
        )
        if EXPORT_NETWORK:
            export_network(trips, links, output)
        results, legs, summaries = [], [], []
        for period, pairs, times in batches:
            logging.info(
                "Routing %s: %d OD pairs, %d departure samples.", period, len(pairs), len(times)
            )
            period_results, period_legs = result_rows(
                pairs, places, errors, times, trips, access, egress, links, stops
            )
            period_summaries = summarize(period_results, pairs)
            for group in (period_results, period_legs, period_summaries):
                for row in group:
                    row["period"] = period
            results.extend(period_results)
            legs.extend(period_legs)
            summaries.extend(period_summaries)
        write_csv(
            output / "od_results.csv",
            results,
            [
                "pair_id",
                "period",
                "origin",
                "destination",
                "service_date",
                "service_id",
                "requested_departure",
                "status",
                "arrival",
                "total_minutes",
                "walk_minutes",
                "wait_minutes",
                "ride_minutes",
                "station_pathway_minutes",
                "walk_miles",
                "boardings",
                "transfers",
                "routes",
                "feeds",
                "used_service_ids",
                "notes",
            ],
        )
        write_csv(
            output / "itinerary_legs.csv",
            legs,
            [
                "pair_id",
                "period",
                "requested_departure",
                "leg_number",
                "mode",
                "from_name",
                "from_stop_code",
                "from_stop_id",
                "to_name",
                "to_stop_code",
                "to_stop_id",
                "start",
                "end",
                "minutes",
                "distance_m",
                "route",
                "route_id",
                "trip_id",
                "source_trip_id",
                "direction",
                "headsign",
                "feed_id",
                "service_id",
                "route_type",
                "walk_basis",
                "pathway_id",
                "from_feed_id",
                "to_feed_id",
                "from_source_stop_id",
                "to_source_stop_id",
            ],
        )
        write_csv(
            output / "od_summary.csv",
            summaries,
            [
                "pair_id",
                "period",
                "origin",
                "destination",
                "status",
                "samples",
                "found",
                "not_found_or_unresolved",
                "transit_results",
                "walk_only_results",
                "min_minutes",
                "mean_minutes",
                "median_minutes",
                "p90_departure_sample_minutes",
                "max_minutes",
                "fastest_requested_departure",
                "mean_walk_minutes",
                "mean_wait_minutes",
                "mean_ride_minutes",
                "mean_walk_miles",
                "mean_station_pathway_minutes",
                "route_counts",
                "statistic_scope",
            ],
        )
        export_benchmark_comparison(output, summaries, references, {trip.route for trip in trips})
        counts = Counter(row["status"] for row in results)
        logging.info("Results by status: %s. Outputs: %s", dict(counts), output)
        if counts["UNRESOLVED_LOCATION"]:
            logging.warning(
                "Completed with unresolved endpoints (exit code 2). "
                "Resolved pairs were processed; see resolved_locations.csv."
            )
        return 2 if counts["UNRESOLVED_LOCATION"] else 0
    finally:
        feed.close()


def run_once(overrides: Dict[str, Any]) -> int:
    """Create a separate run folder, retain settings, and report actionable failures."""
    feeds = list(GTFS_FEEDS.values()) if USE_MULTIPLE_FEEDS else [GTFS_PATH]
    unset = [
        str(path)
        for path in [*feeds, OUTPUT_DIR]
        if str(path).replace("\\", "/").startswith("Path/To/")
    ]
    if unset:
        logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
        logging.warning(
            "Placeholder paths are still set: %s. Update the CONFIGURATION section or pass "
            "--feed (or --gtfs-path) and --output-dir.",
            ", ".join(unset),
        )
        return 2
    output = OUTPUT_DIR / ("gtfs_od_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    try:
        output.mkdir(parents=True, exist_ok=False)
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(output / "run.log", encoding="utf-8"),
            ],
            force=True,
        )
        settings = {
            key: value
            for key, value in globals().items()
            if key.isupper() and key not in {"EARTH_RADIUS_M", "METERS_PER_MILE", "WEEKDAYS"}
        }
        (output / "settings.json").write_text(
            json.dumps(settings, indent=2, default=str), encoding="utf-8"
        )
        if not write_run_log(output, overrides) and REQUIRE_RUN_LOG:
            logging.error(
                "Run log could not be written; stopping before routing. Set "
                "REQUIRE_RUN_LOG = False (or pass --no-require-run-log) to continue without it."
            )
            return 1
        return run(output)
    except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as exc:
        logging.exception("Failed: %s. Completed diagnostics remain in %s", exc, output)
        return 1


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


def config_block_source() -> Tuple[str, str]:
    """Return the CONFIGURATION block text and its source, from this file or a notebook.

    A script run reads its own file. In a notebook, where __file__ is undefined, the
    most recent input cell containing both CONFIG markers is used instead.

    Raises:
        ValueError: If neither source holds both CONFIG markers.
        OSError: If the script file cannot be read.
    """
    file_attr = globals().get("__file__")
    if file_attr:
        source = Path(file_attr).resolve()
        return extract_config_block(source), str(source)
    begin, end = "# === BEGIN CONFIG ===", "# === END CONFIG ==="
    ipython = sys.modules.get("IPython")
    shell = getattr(ipython, "get_ipython", lambda: None)() if ipython is not None else None
    cells = shell.user_ns.get("In", []) if shell is not None else []
    for cell in reversed(cells):
        if begin in cell and end in cell.split(begin, 1)[1]:
            return cell.split(begin, 1)[1].split(end, 1)[0].strip("\n"), "<Jupyter cell>"
    raise ValueError(
        "CONFIG markers not found: __file__ is undefined and no notebook cell holds them."
    )


def write_run_log(output: Path, overrides: Dict[str, Any]) -> bool:
    """Write the run folder's _runlog.txt sidecar: configuration, overrides and source.

    Args:
        output: The run folder that receives every output of this run.
        overrides: Settings whose command-line values replaced the CONFIGURATION ones.

    Returns:
        True if the sidecar was written, False otherwise.
    """
    log_path = output / "gtfs_od_travel_times_runlog.txt"
    try:
        config_text, source = config_block_source()
    except (OSError, ValueError) as exc:
        logging.error("Could not extract the CONFIGURATION block for the run log: %s", exc)
        return False
    shown = {**overrides}
    if "LOG_LEVEL" in shown:
        shown["LOG_LEVEL"] = logging.getLevelName(shown["LOG_LEVEL"])
    override_lines = [f"{name} = {json.dumps(value, default=str)}" for name, value in shown.items()]
    lines = [
        "=" * 72,
        "GTFS OD TRAVEL TIMES RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output}",
        f"Source script:    {source}",
        "",
        "-" * 72,
        "COMMAND-LINE OVERRIDES (these replace the CONFIGURATION values below)",
        "-" * 72,
        *(override_lines or ["(none)"]),
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
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to %s", log_path)
    return True


def feed_spec(text: str) -> Tuple[str, Path]:
    """Parse one --feed value written as NAME=PATH."""
    name, separator, path = text.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            f"Use NAME=PATH, for example agency_a=agency_a.zip; got {text!r}."
        )
    return name.strip(), Path(path.strip())


def optional_text(text: str) -> Optional[str]:
    """Return None for a blank value or 'none'; keep other text, including leading zeros."""
    return None if text.strip().lower() in {"", "none"} else text.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the parser; each flag's default is the matching CONFIGURATION value."""
    p = argparse.ArgumentParser(
        description="Estimate door-to-door transit times between configured OD pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configured = ", ".join(f"{name}={path}" for name, path in GTFS_FEEDS.items())
    p.add_argument(
        "--feed",
        dest="feeds",
        action="append",
        type=feed_spec,
        default=argparse.SUPPRESS,
        metavar="NAME=PATH",
        help="GTFS feed as a folder or .zip; repeat for each agency. Replaces GTFS_FEEDS "
        f"({configured}).",
    )
    p.add_argument(
        "--gtfs-path",
        dest="GTFS_PATH",
        type=Path,
        default=GTFS_PATH,
        help="Single GTFS feed (folder or .zip), used with --no-multiple-feeds.",
    )
    p.add_argument(
        "--service-date",
        dest="SERVICE_DATE",
        type=optional_text,
        default=SERVICE_DATE,
        help="Service day as YYYYMMDD; 'none' uses --service-id instead.",
    )
    p.add_argument(
        "--service-id",
        dest="SERVICE_ID",
        type=optional_text,
        default=SERVICE_ID,
        help="Single-feed service_id, used only when the service date is none.",
    )
    p.add_argument(
        "--output-dir",
        dest="OUTPUT_DIR",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder that receives a new timestamped run folder.",
    )
    numbers = [
        ("--walk-speed-mph", "WALK_SPEED_MPH", float, "Walking speed."),
        (
            "--max-access-walk-miles",
            "MAX_ACCESS_WALK_MILES",
            float,
            "Longest walk from the origin to a stop or station entrance.",
        ),
        (
            "--max-transfer-walk-miles",
            "MAX_TRANSFER_WALK_MILES",
            float,
            "Longest outdoor walk between stops when transferring.",
        ),
        (
            "--max-egress-walk-miles",
            "MAX_EGRESS_WALK_MILES",
            float,
            "Longest walk from a stop or station exit to the destination.",
        ),
        (
            "--min-transfer-minutes",
            "MIN_TRANSFER_MINUTES",
            float,
            "Buffer added after each transfer walk.",
        ),
        (
            "--initial-boarding-buffer-minutes",
            "INITIAL_BOARDING_BUFFER_MINUTES",
            float,
            "Buffer before the first boarding.",
        ),
        ("--max-transfers", "MAX_TRANSFERS", int, "Transfers allowed (2 = three boardings)."),
        (
            "--max-journey-minutes",
            "MAX_JOURNEY_MINUTES",
            float,
            "Longest door-to-door journey, including waits.",
        ),
        (
            "--max-walk-only-miles",
            "MAX_WALK_ONLY_MILES",
            float,
            "Longest walk-only trip when walk-only results are allowed.",
        ),
        (
            "--departure-step-minutes",
            "DEPARTURE_STEP_MINUTES",
            int,
            "Minutes between sampled departures.",
        ),
    ]
    for flag, name, kind, text in numbers:
        p.add_argument(flag, dest=name, type=kind, default=globals()[name], help=text)
    p.add_argument(
        "--departure-times",
        dest="EXPLICIT_DEPARTURE_TIMES",
        nargs="*",
        default=EXPLICIT_DEPARTURE_TIMES,
        metavar="HH:MM",
        help="Exact departures for --no-use-periods; replaces the start/end range.",
    )
    p.add_argument(
        "--departure-start",
        dest="DEPARTURE_START",
        default=DEPARTURE_START,
        metavar="HH:MM",
        help="First sampled departure for --no-use-periods.",
    )
    p.add_argument(
        "--departure-end",
        dest="DEPARTURE_END",
        default=DEPARTURE_END,
        metavar="HH:MM",
        help="Last sampled departure for --no-use-periods.",
    )
    switches = [
        ("--multiple-feeds", "USE_MULTIPLE_FEEDS", "Combine every feed in GTFS_FEEDS / --feed."),
        ("--discovery-only", "DISCOVERY_ONLY", "Write calendar and stop inventories only."),
        (
            "--allow-walk-only",
            "ALLOW_WALK_ONLY",
            "Allow walk-only results up to --max-walk-only-miles.",
        ),
        (
            "--use-periods",
            "USE_PERIODS",
            "Sample the PERIODS windows; otherwise use the --departure-* settings.",
        ),
        ("--include-reverse-pairs", "INCLUDE_REVERSE_PAIRS", "Also route each OD pair reversed."),
        (
            "--interpolate-missing-stop-times",
            "INTERPOLATE_MISSING_STOP_TIMES",
            "Interpolate blank times at timepoint=0 stops.",
        ),
        (
            "--derive-zero-time-walkways",
            "DERIVE_ZERO_TIME_WALKWAYS",
            "Derive zero walkway traversal times from length and walking speed.",
        ),
        (
            "--require-rail-pathways",
            "REQUIRE_RAIL_PATHWAYS",
            "Require pathways.txt entrances for subway/metro platforms.",
        ),
        ("--export-network", "EXPORT_NETWORK", "Write the network segment and walk-link tables."),
        (
            "--require-run-log",
            "REQUIRE_RUN_LOG",
            "Stop before routing if the _runlog.txt sidecar cannot be written.",
        ),
    ]
    for flag, name, text in switches:
        p.add_argument(
            flag,
            dest=name,
            action=argparse.BooleanOptionalAction,
            default=globals()[name],
            help=text,
        )
    p.add_argument(
        "--log-level",
        dest="LOG_LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=logging.getLevelName(LOG_LEVEL),
        help="Logging detail.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Apply command-line overrides to the CONFIGURATION values for one run.

    Overrides last only for this call; the CONFIGURATION values are restored
    afterward, so repeated notebook calls start from the same settings.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if paths are still
        placeholders, no service was selected, or an endpoint is unresolved.
    """
    parser = build_arg_parser()
    args = vars(parser.parse_args(notebook_safe_argv(argv)))
    if "feeds" in args:
        feeds = args.pop("feeds")
        if len({name for name, _ in feeds}) != len(feeds):
            parser.error("Each --feed NAME must be unique.")
        args["GTFS_FEEDS"] = dict(feeds)
    args["LOG_LEVEL"] = getattr(logging, args["LOG_LEVEL"])
    overrides = {name: value for name, value in args.items() if value != globals()[name]}
    saved = {name: globals()[name] for name in args}
    globals().update(args)
    try:
        return run_once(overrides)
    finally:
        globals().update(saved)


if __name__ == "__main__":
    raise SystemExit(main())
