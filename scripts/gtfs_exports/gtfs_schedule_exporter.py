"""Export GTFS timetables, distance patterns, runtime bands, and speed bands.

EDIT HERE: change GTFS_PATH and OUTPUT_DIR in CONFIGURATION, confirm the feed's
shape_dist_traveled unit, then run this file in Python, ArcGIS Pro, or a notebook.
Requires Python 3.9+ and openpyxl. No pandas, ArcPy, repository, or CLI required.

One workbook per route_id/service_id contains branch-aware timetable sheets,
separate metric sheets for each direction/stop pattern, a trip index, and issues.
Every run has a text log and issues.csv; optional detail CSVs retain trip lineage.
Each run gets a new subfolder by default, so old flagged files are never confused
with current results. Save failures return a nonzero exit code.

Definitions:
* Pattern ID describes the ordered DISPLAYED stop IDs, including repeat visits.
  Full Pattern ID describes ALL stops before filtering. Both are stable hashes.
* Runtime default: next retained stop's arrival minus previous retained stop's
  departure. This includes travel/dwell at omitted intermediate stops, but not
  holding at the downstream retained stop. Holding is exported separately.
* Optional departure_to_departure runtime includes downstream holding, including
  terminal dwell. Speed uses the selected runtime basis and is scheduled speed.
* Mean speed = total distance / total selected segment runtime, only when ALL
  segment distances/runtimes are known and all segment runtimes are positive.
* Consecutive bands split on profile changes within each displayed pattern,
  including changes in full stop sequence, shape_id, distance, runtime or holding.
  First/Last Departure are observed trip starts, NOT inclusive validity windows.
* Missing values remain blank; no interpolation, guessed distance units, or
  straight-line fallback is applied. Hours beyond 24 and seconds are preserved.
* Frequency-based trips are rejected by default (or explicitly skipped). They
  are not expanded and are never counted as ordinary scheduled departures.
* Service labels describe the calendar, not a date-specific service selection.
  Use service-ID filters. Calendar exceptions are included in labels/metadata;
  this exporter does not combine services into a particular operating date.

Sections intentionally separate feed loading, metrics, timetable layout, and
export, so they can become modules later without changing the configuration.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import logging
import math
import os
import re
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

# =============================================================================
# CONFIGURATION -- EDIT HERE
# =============================================================================
# === BEGIN CONFIG ===
GTFS_PATH = Path(r"Path\To\Your\GTFS_Folder")  # Folder OR ZIP; UNC paths accepted.
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

INCLUDE_ROUTES: list[str] = []  # route_short_name; empty = all. Values as strings.
EXCLUDE_ROUTES: list[str] = []
INCLUDE_SERVICE_IDS: list[str] = []
EXCLUDE_SERVICE_IDS: list[str] = []
SERVICE_LABEL_OVERRIDES: dict[str, str] = {}  # Example: {"4": "Weekday"}.

EXPORT_TIMETABLES = True
EXPORT_DISTANCES = True
EXPORT_RUNTIMES = True
EXPORT_SPEEDS = True
EXPORT_DETAIL_CSV = True  # trips.csv and segments.csv for traceable calculations.

TIMEPOINTS_ONLY = True
KEEP_TRIP_ENDPOINTS = True  # Preserve full trip coverage even if endpoints are 0.
INPUT_DISTANCE_UNIT = "meters"  # "feet", "meters", "km", "miles"; confirm this!
OUTPUT_DISTANCE_UNIT = "miles"  # Same options; speeds always mph.
RUNTIME_BASIS = "departure_to_arrival"  # Or "departure_to_departure".
BAND_MODE = "consecutive"  # Or "profile_summary" for all-day first/last summaries.
MAX_BAND_GAP_MINUTES: float | None = 120.0  # None disables gap splitting.

TIME_FORMAT = "24"  # "24" preserves 25:15; "12" shows 1:15 AM (+1d).
SHOW_SECONDS = False  # False rounds DISPLAY ONLY to nearest minute, halves up.
STOP_LABEL_ID = "stop_code"  # Or "stop_id". Internal joins always use stop_id.
FREQUENCY_POLICY = "error"  # "error" or "skip"; no silent template-trip counting.
SPEED_WARNING_MPH = 65.0  # Review flag, not an automatic cap or deletion.
CREATE_RUN_SUBFOLDER = True
WARNING_FILENAME_SUFFIX = "_warning"  # Empty string disables suffix, not warnings.
LOG_LEVEL = logging.INFO
# === END CONFIG ===

LOGGER = logging.getLogger("gtfs_schedule_exporter")
VERSION = "1.0.0"
METERS_PER_UNIT = {"feet": 0.3048, "meters": 1.0, "km": 1000.0, "miles": 1609.344}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class Config:
    """Effective run settings; also accepts programmatic overrides for testing."""

    gtfs_path: Path
    output_dir: Path
    include_routes: tuple[str, ...] = ()
    exclude_routes: tuple[str, ...] = ()
    include_services: tuple[str, ...] = ()
    exclude_services: tuple[str, ...] = ()
    service_labels: dict[str, str] = field(default_factory=dict)
    timetables: bool = True
    distances: bool = True
    runtimes: bool = True
    speeds: bool = True
    detail_csv: bool = True
    timepoints_only: bool = True
    keep_endpoints: bool = True
    input_unit: str = "meters"
    output_unit: str = "miles"
    runtime_basis: str = "departure_to_arrival"
    band_mode: str = "consecutive"
    max_band_gap: float | None = 120.0
    time_format: str = "24"
    show_seconds: bool = False
    stop_label_id: str = "stop_code"
    frequency_policy: str = "error"
    speed_warning: float = 65.0
    run_subfolder: bool = True
    warning_suffix: str = "_warning"


def configured_settings() -> Config:
    """Read live globals, including settings changed in an interactive session."""
    return Config(
        gtfs_path=Path(GTFS_PATH),
        output_dir=Path(OUTPUT_DIR),
        include_routes=tuple(str(x) for x in INCLUDE_ROUTES),
        exclude_routes=tuple(str(x) for x in EXCLUDE_ROUTES),
        include_services=tuple(str(x) for x in INCLUDE_SERVICE_IDS),
        exclude_services=tuple(str(x) for x in EXCLUDE_SERVICE_IDS),
        service_labels=dict(SERVICE_LABEL_OVERRIDES),
        timetables=EXPORT_TIMETABLES,
        distances=EXPORT_DISTANCES,
        runtimes=EXPORT_RUNTIMES,
        speeds=EXPORT_SPEEDS,
        detail_csv=EXPORT_DETAIL_CSV,
        timepoints_only=TIMEPOINTS_ONLY,
        keep_endpoints=KEEP_TRIP_ENDPOINTS,
        input_unit=INPUT_DISTANCE_UNIT,
        output_unit=OUTPUT_DISTANCE_UNIT,
        runtime_basis=RUNTIME_BASIS,
        band_mode=BAND_MODE,
        max_band_gap=MAX_BAND_GAP_MINUTES,
        time_format=TIME_FORMAT,
        show_seconds=SHOW_SECONDS,
        stop_label_id=STOP_LABEL_ID,
        frequency_policy=FREQUENCY_POLICY,
        speed_warning=SPEED_WARNING_MPH,
        run_subfolder=CREATE_RUN_SUBFOLDER,
        warning_suffix=WARNING_FILENAME_SUFFIX,
    )


@dataclass(frozen=True)
class Issue:
    """One traceable input, calculation, layout, or export finding."""

    code: str
    message: str
    route_id: str = ""
    service_id: str = ""
    trip_id: str = ""
    severity: str = "WARNING"


@dataclass
class Audit:
    """Retain every finding while keeping repetitive console logging bounded."""

    issues: list[Issue] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        code: str,
        message: str,
        trip: Mapping[str, str] | None = None,
        severity: str = "WARNING",
    ) -> None:
        """Record a global or trip-specific issue."""
        row = trip or {}
        issue = Issue(
            code,
            message,
            row.get("route_id", ""),
            row.get("service_id", ""),
            row.get("trip_id", ""),
            severity,
        )
        self.issues.append(issue)
        self.counts[code] += 1
        if self.counts[code] <= 5:
            LOGGER.log(
                logging.ERROR if severity == "ERROR" else logging.WARNING,
                "%s | trip=%s | %s",
                code,
                issue.trip_id or "-",
                message,
            )
        elif self.counts[code] == 6:
            LOGGER.warning("Further %s findings will be listed in issues.csv.", code)

    def relevant(self, route_id: str, service_id: str) -> list[Issue]:
        """Include global findings and this workbook's route/service findings."""
        return [
            i
            for i in self.issues
            if (not i.route_id or i.route_id == route_id)
            and (not i.service_id or i.service_id == service_id)
        ]


@dataclass(frozen=True)
class Stop:
    """One stop and its preferred public label components."""

    stop_id: str
    name: str
    code: str


@dataclass(frozen=True)
class Event:
    """One stop visit; numeric values remain in seconds and input distance units."""

    stop_id: str
    sequence: int
    arrival: int | None
    departure: int | None
    distance: float | None
    timepoint: str


@dataclass(frozen=True)
class Trip:
    """Normalized trip plus its displayed segment calculations."""

    trip_id: str
    route_id: str
    route_name: str
    service_id: str
    direction: str
    block_id: str
    headsign: str
    shape_id: str
    events: tuple[Event, ...]
    final_sequence: int
    full_pattern: tuple[str, ...]
    full_pattern_id: str
    pattern_id: str
    start: int | None
    end: int | None
    distances: tuple[float | None, ...]  # One per displayed stop, first = None.
    running: tuple[int | None, ...]  # Departure -> next arrival.
    elapsed: tuple[int | None, ...]  # Departure -> next departure.
    holding: tuple[int | None, ...]
    full_distance: float | None
    full_stop_count: int

    @property
    def pattern(self) -> tuple[str, ...]:
        """Keep repeated visits as distinct ordered entries."""
        return tuple(e.stop_id for e in self.events)

    def runtimes(self, config: Config) -> tuple[int | None, ...]:
        """Select the explicitly configured interval definition."""
        return self.running if config.runtime_basis == "departure_to_arrival" else self.elapsed


@dataclass
class Feed:
    """Selected feed tables and grouped raw stop-time rows."""

    routes: dict[str, dict[str, str]]
    stops: dict[str, Stop]
    trips: dict[str, dict[str, str]]
    stop_times: dict[str, list[dict[str, str]]]
    service_labels: dict[str, str]
    timepoints_provided: bool
    input_trip_count: int
    selected_trip_count: int
    frequency_skipped: int


# =============================================================================
# PART 1 -- FEED LOADING AND VALIDATION
# =============================================================================


class FeedReader:
    """Read root or nested GTFS CSV members without extracting an archive."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive: zipfile.ZipFile | None = None
        self.members: dict[str, list[str]] = defaultdict(list)

    def __enter__(self) -> FeedReader:
        if self.path.is_dir():
            for path in self.path.iterdir():
                if path.is_file():
                    self.members[path.name].append(path.name)
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.archive = zipfile.ZipFile(self.path)
            for member in self.archive.namelist():
                if not member.endswith("/") and "__MACOSX/" not in member:
                    self.members[member.rsplit("/", 1)[-1]].append(member)
        else:
            raise ValueError(f"GTFS_PATH is not a feed directory or valid ZIP: {self.path}")
        return self

    def __exit__(self, *_args: object) -> None:
        if self.archive is not None:
            self.archive.close()

    @contextmanager
    def open_text(self, name: str) -> Generator[TextIO, None, None]:
        """Reject ambiguous members instead of silently choosing one feed."""
        members = self.members.get(name, [])
        if not members:
            raise FileNotFoundError(f"Missing required GTFS file: {name}")
        if len(members) != 1:
            raise ValueError(f"Ambiguous GTFS file {name}: {members}")
        if self.archive is None:
            with (self.path / name).open(encoding="utf-8-sig", newline="") as handle:
                yield handle
        else:
            with self.archive.open(members[0]) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as handle:
                    yield handle

    def rows(
        self, name: str, required: Sequence[str], optional: bool = False
    ) -> Iterator[dict[str, str]]:
        """Read IDs literally, including 'NA', 'NULL', and leading zeros."""
        if optional and name not in self.members:
            return
        with self.open_text(name) as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            if not columns:
                raise ValueError(f"{name} is empty or has no header.")
            if len(columns) != len(set(columns)):
                raise ValueError(f"{name} has duplicate column names.")
            missing = set(required) - set(columns)
            if missing:
                raise ValueError(f"{name} is missing columns: {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f"{name}, row {line}: more cells than header columns.")
                yield {key: (value or "").strip() for key, value in row.items() if key is not None}


def keyed_rows(rows: Iterator[dict[str, str]], key: str, name: str) -> dict[str, dict[str, str]]:
    """Validate unique nonblank IDs before building any joins."""
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"{name}: blank or duplicate {key} {value!r}.")
        result[value] = row
    return result


def calendar_labels(
    reader: FeedReader, services: set[str], config: Config, audit: Audit
) -> dict[str, str]:
    """Label services without assuming agency-specific ID/day mappings."""
    calendar = keyed_rows(
        reader.rows("calendar.txt", ["service_id", *WEEKDAYS], True), "service_id", "calendar.txt"
    )
    added: dict[str, set[int]] = defaultdict(set)
    for row in reader.rows("calendar_dates.txt", ["service_id", "date", "exception_type"], True):
        try:
            day = datetime.strptime(row["date"], "%Y%m%d").weekday()
            if row["exception_type"] not in {"1", "2"}:
                raise ValueError("exception_type must be 1 or 2")
        except ValueError:
            audit.add("INVALID_CALENDAR_DATE", f"Invalid calendar_dates row: {row}")
            continue
        if row["exception_type"] == "1":
            added[row["service_id"]].add(day)
    labels = {}
    common = {
        (0, 1, 2, 3, 4): "Weekday",
        (5,): "Saturday",
        (6,): "Sunday",
        (5, 6): "Weekend",
        tuple(range(7)): "Daily",
    }
    for service_id in services:
        if service_id in calendar:
            days = tuple(i for i, day in enumerate(WEEKDAYS) if calendar[service_id][day] == "1")
            label = common.get(days, "-".join(WEEKDAYS[i][:3].title() for i in days))
            label = label or "Exceptions only"
        elif service_id in added:
            label = (
                "Dates only (" + ",".join(WEEKDAYS[i][:3] for i in sorted(added[service_id])) + ")"
            )
        else:
            label = "Unspecified service"
            audit.add(
                "UNDEFINED_SERVICE",
                f"Service {service_id!r} has no usable calendar definition.",
                {"service_id": service_id},
            )
        labels[service_id] = config.service_labels.get(service_id, label)
    return labels


def load_feed(config: Config, audit: Audit) -> Feed:
    """Load only needed files; scan stop_times once and retain selected trips."""
    with FeedReader(config.gtfs_path) as reader:
        routes = keyed_rows(reader.rows("routes.txt", ["route_id"]), "route_id", "routes.txt")
        stop_rows = keyed_rows(
            reader.rows("stops.txt", ["stop_id", "stop_name"]), "stop_id", "stops.txt"
        )
        stops = {
            sid: Stop(sid, row["stop_name"] or sid, row.get("stop_code", ""))
            for sid, row in stop_rows.items()
        }
        all_trips = keyed_rows(
            reader.rows("trips.txt", ["trip_id", "route_id", "service_id"]), "trip_id", "trips.txt"
        )
        route_names = {
            rid: row.get("route_short_name", "") or row.get("route_long_name", "") or rid
            for rid, row in routes.items()
        }
        for name in sorted(set(config.include_routes) - set(route_names.values())):
            audit.add(
                "FILTER_NOT_FOUND", f"Included route label {name!r} is absent from routes.txt."
            )
        for sid in sorted(
            set(config.include_services) - {t["service_id"] for t in all_trips.values()}
        ):
            audit.add("FILTER_NOT_FOUND", f"Included service_id {sid!r} is absent from trips.txt.")
        trips = {}
        for trip_id, trip in all_trips.items():
            rid, sid = trip["route_id"], trip["service_id"]
            if rid not in routes or not sid:
                raise ValueError(f"Trip {trip_id}: unknown route_id or blank service_id.")
            name = route_names[rid]
            if (
                (config.include_routes and name not in config.include_routes)
                or name in config.exclude_routes
                or (config.include_services and sid not in config.include_services)
                or sid in config.exclude_services
            ):
                continue
            trips[trip_id] = {
                **trip,
                "route_name": name,
                "direction_id": trip.get("direction_id", ""),
            }
        selected_count = len(trips)
        frequency_ids = {
            row["trip_id"] for row in reader.rows("frequencies.txt", ["trip_id"], True)
        }
        selected_frequency = frequency_ids & trips.keys()
        if selected_frequency and config.frequency_policy == "error":
            raise ValueError(
                f"{len(selected_frequency)} selected trips use frequencies.txt. "
                "Expansion is unsupported. Use FREQUENCY_POLICY='skip' explicitly "
                "or provide an expanded fixed-schedule feed."
            )
        for trip_id in sorted(selected_frequency):
            audit.add(
                "FREQUENCY_SKIPPED",
                "Frequency template excluded; not counted as a departure.",
                trips[trip_id],
            )
            del trips[trip_id]
        labels = calendar_labels(reader, {t["service_id"] for t in trips.values()}, config, audit)
        stop_times: dict[str, list[dict[str, str]]] = defaultdict(list)
        timepoints_provided = False
        count = 0
        for row in reader.rows("stop_times.txt", ["trip_id", "stop_id", "stop_sequence"]):
            count += 1
            timepoints_provided = timepoints_provided or bool(row.get("timepoint", ""))
            if row["trip_id"] in trips:
                stop_times[row["trip_id"]].append(row)
            if count % 250000 == 0:
                LOGGER.info("Read %s stop-time rows...", f"{count:,}")
        LOGGER.info(
            "Loaded %s trips; selected %s; retained %s stop-time rows.",
            f"{len(all_trips):,}",
            f"{len(trips):,}",
            f"{sum(map(len, stop_times.values())):,}",
        )
    return Feed(
        routes,
        stops,
        trips,
        dict(stop_times),
        labels,
        timepoints_provided,
        len(all_trips),
        selected_count,
        len(selected_frequency),
    )


def parse_seconds(value: str) -> int | None:
    """Parse HH:MM[:SS] without rounding or wrapping extended service-day hours."""
    match = re.fullmatch(r"(\d+):(\d{2})(?::(\d{2}))?", value.strip())
    if not match:
        return None
    hours, minutes, seconds = int(match[1]), int(match[2]), int(match[3] or 0)
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def stable_id(value: object) -> str:
    """A deterministic display ID; full tuples, not hashes, remain grouping keys."""
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def duration(start: int | None, end: int | None) -> int | None:
    """Leave missing or reversed intervals unknown; never invent a midnight wrap."""
    return end - start if start is not None and end is not None and end >= start else None


def complete_sum(values: Sequence[float | int | None]) -> float | None:
    """Prevent partial observations from masquerading as complete totals."""
    if not values or any(value is None for value in values):
        return None
    return math.fsum(value for value in values if value is not None)


def distance_value(value: str, audit: Audit, trip: dict[str, str], sequence: int) -> float | None:
    """Coerce valid cumulative distances; retain blanks as missing."""
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        result = float("nan")
    if not math.isfinite(result) or result < 0:
        audit.add(
            "INVALID_DISTANCE", f"Stop sequence {sequence}: invalid distance {value!r}.", trip
        )
        return None
    return result


def parse_events(
    rows: list[dict[str, str]], metadata: dict[str, str], feed: Feed, audit: Audit
) -> tuple[list[Event], bool, bool]:
    """Validate every stop visit before any timepoint filtering takes place."""
    events = []
    sequences = set()
    for row in rows:
        token = row["stop_sequence"]
        if not token.isdigit() or int(token) in sequences:
            raise ValueError(
                f"Trip {metadata['trip_id']}: invalid/duplicate stop_sequence {token!r}."
            )
        sequence = int(token)
        sequences.add(sequence)
        if row["stop_id"] not in feed.stops:
            raise ValueError(
                f"Trip {metadata['trip_id']}: unknown/blank stop_id {row['stop_id']!r}."
            )
        arr_raw, dep_raw = row.get("arrival_time", ""), row.get("departure_time", "")
        arrival, departure = parse_seconds(arr_raw), parse_seconds(dep_raw)
        if (arr_raw and arrival is None) or (dep_raw and departure is None):
            audit.add(
                "INVALID_TIME",
                f"Sequence {sequence}: invalid arrival/departure {arr_raw!r}/{dep_raw!r}.",
                metadata,
            )
        if (arrival is None) != (departure is None):
            audit.add(
                "TIME_FALLBACK",
                f"Sequence {sequence}: using available time for both arrival/departure; holding unknown.",
                metadata,
            )
        # Keep original missingness: fallback is applied explicitly to intervals,
        # while holding remains unknown if either original event time is absent.
        tp = row.get("timepoint", "")
        if tp not in {"", "0", "1"}:
            audit.add(
                "INVALID_TIMEPOINT",
                f"Sequence {sequence}: invalid timepoint {tp!r}; retained for review.",
                metadata,
            )
        events.append(
            Event(
                row["stop_id"],
                sequence,
                arrival,
                departure,
                distance_value(row.get("shape_dist_traveled", ""), audit, metadata, sequence),
                tp,
            )
        )
    events.sort(key=lambda event: event.sequence)
    time_ok = distance_ok = True
    last_time = last_distance = None
    for event in events:
        for value in (event.arrival, event.departure):
            if value is not None:
                if last_time is not None and value < last_time:
                    time_ok = False
                last_time = value
        if event.distance is not None:
            if last_distance is not None and event.distance < last_distance:
                distance_ok = False
            last_distance = event.distance
    if not time_ok:
        audit.add(
            "TIME_REVERSAL",
            "Time decreases within trip. Times retained for review; runtime/speed metrics left blank.",
            metadata,
        )
    if not distance_ok:
        audit.add(
            "DISTANCE_REVERSAL",
            "Cumulative distance decreases. Distance/speed metrics left blank.",
            metadata,
        )
    return events, time_ok, distance_ok


def arrival(event: Event) -> int | None:
    """Use departure only when arrival is unavailable."""
    return event.arrival if event.arrival is not None else event.departure


def departure(event: Event) -> int | None:
    """Use arrival only when departure is unavailable."""
    return event.departure if event.departure is not None else event.arrival


def build_trip(
    metadata: dict[str, str], rows: list[dict[str, str]], feed: Feed, config: Config, audit: Audit
) -> Trip:
    """Calculate all metrics from the same retained visits and unrounded values."""
    events, time_ok, distance_ok = parse_events(rows, metadata, feed, audit)
    if len(events) < 2:
        raise ValueError(f"Trip {metadata['trip_id']}: fewer than two stop-time rows.")
    if metadata.get("direction_id", "") not in {"", "0", "1"}:
        audit.add(
            "INVALID_DIRECTION",
            "Nonstandard direction_id retained as a separate direction.",
            metadata,
        )
    if feed.timepoints_provided and any(e.timepoint == "" for e in events):
        audit.add(
            "PARTIAL_TIMEPOINTS",
            "Blank timepoint flags in a partly populated feed are retained for review.",
            metadata,
        )
    selected = [
        e
        for e in events
        if not config.timepoints_only
        or not feed.timepoints_provided
        or e.timepoint != "0"
        or (config.keep_endpoints and e in (events[0], events[-1]))
    ]
    if not selected:
        raise ValueError(
            f"Trip {metadata['trip_id']}: no retained stops. Enable KEEP_TRIP_ENDPOINTS or all-stop export."
        )
    if selected[0] != events[0] or selected[-1] != events[-1]:
        audit.add(
            "PARTIAL_TRIP_COVERAGE",
            "Displayed stops exclude a trip endpoint; displayed metric totals cover only the retained span.",
            metadata,
        )
    distances: list[float | None] = [None]
    running: list[int | None] = [None]
    elapsed: list[int | None] = [None]
    for previous, current in zip(selected, selected[1:]):
        dist = (
            current.distance - previous.distance
            if distance_ok and current.distance is not None and previous.distance is not None
            else None
        )
        distances.append(dist)
        running.append(duration(departure(previous), arrival(current)) if time_ok else None)
        elapsed.append(duration(departure(previous), departure(current)) if time_ok else None)
    holding = tuple(duration(e.arrival, e.departure) if time_ok else None for e in selected)
    full_distance = (
        events[-1].distance - events[0].distance
        if distance_ok and events[-1].distance is not None and events[0].distance is not None
        else None
    )
    trip = Trip(
        metadata["trip_id"],
        metadata["route_id"],
        metadata["route_name"],
        metadata["service_id"],
        metadata.get("direction_id", ""),
        metadata.get("block_id", ""),
        metadata.get("trip_headsign", ""),
        metadata.get("shape_id", ""),
        tuple(selected),
        events[-1].sequence,
        tuple(e.stop_id for e in events),
        stable_id([e.stop_id for e in events]),
        stable_id([e.stop_id for e in selected]),
        departure(events[0]),
        arrival(events[-1]),
        tuple(distances),
        tuple(running),
        tuple(elapsed),
        holding,
        full_distance,
        len(events),
    )
    if trip.start is None or trip.end is None:
        audit.add(
            "MISSING_TRIP_TIME",
            "Missing first departure or last arrival; trip retained with blank times.",
            metadata,
        )
    if (config.distances or config.speeds) and any(d is None for d in trip.distances[1:]):
        audit.add(
            "INCOMPLETE_DISTANCE",
            "One or more displayed segment distances unavailable; total displayed distance/speed left blank.",
            metadata,
        )
    runtimes = trip.runtimes(config)
    if (config.runtimes or config.speeds) and any(t is None for t in runtimes[1:]):
        audit.add(
            "INCOMPLETE_RUNTIME",
            "One or more displayed runtimes unavailable; total runtime/speed left blank.",
            metadata,
        )
    if config.speeds and any(t == 0 for t in runtimes[1:]):
        audit.add(
            "ZERO_RUNTIME",
            "Zero-duration segment; speed for that segment and overall mean left blank.",
            metadata,
        )
    for dist, seconds in zip(trip.distances[1:], runtimes[1:]):
        speed = speed_mph(dist, seconds, config)
        if config.speeds and speed is not None and speed > config.speed_warning:
            audit.add(
                "HIGH_SPEED",
                f"Scheduled segment speed {speed:.1f} mph exceeds {config.speed_warning:g}; check distance units/times.",
                metadata,
            )
            break
    displayed_total = complete_sum(trip.distances[1:])
    if (
        selected[0] == events[0]
        and selected[-1] == events[-1]
        and displayed_total is not None
        and full_distance is not None
    ):
        if not math.isclose(displayed_total, full_distance, rel_tol=1e-9, abs_tol=1e-8):
            audit.add(
                "DISTANCE_RECONCILIATION",
                "Displayed segment sum differs from unfiltered endpoint distance.",
                metadata,
            )
    return trip


def speed_mph(distance: float | None, seconds: float | int | None, config: Config) -> float | None:
    """Unknown, negative, and zero-duration cases never become numeric speeds."""
    if distance is None or seconds is None or distance < 0 or seconds <= 0:
        return None
    return distance * METERS_PER_UNIT[config.input_unit] / 1609.344 * 3600 / seconds


def mean_speed(trip: Trip, config: Config) -> float | None:
    """Use this trip's distance/runtime, never another trip's speed signature."""
    runtimes = trip.runtimes(config)[1:]
    if any(value is None or value <= 0 for value in runtimes):
        return None
    return speed_mph(complete_sum(trip.distances[1:]), complete_sum(runtimes), config)


# =============================================================================
# PART 2 -- PATTERN PROFILES AND TIMETABLE LAYOUT
# =============================================================================


def profile_key(trip: Trip, config: Config) -> tuple[object, ...]:
    """Share band boundaries across metrics; do not group by rounded speeds."""
    return (
        trip.full_pattern,
        trip.shape_id,
        trip.distances,
        trip.running,
        trip.elapsed,
        trip.holding,
        config.runtime_basis,
    )


def trip_order(trip: Trip) -> tuple[bool, int, str]:
    """Order using original first-stop departure, not a filtered timepoint."""
    return trip.start is None, trip.start if trip.start is not None else 0, trip.trip_id


def make_bands(trips: Sequence[Trip], config: Config) -> list[list[Trip]]:
    """Group one pattern's trips, splitting intervening profiles and large gaps."""
    ordered = sorted(trips, key=trip_order)
    if config.band_mode == "profile_summary":
        groups: dict[tuple[object, ...], list[Trip]] = defaultdict(list)
        for trip in ordered:
            # Missing departures are traceable rows, never silently omitted.
            key = profile_key(trip, config) + ((trip.trip_id,) if trip.start is None else ())
            groups[key].append(trip)
        return list(groups.values())
    bands: list[list[Trip]] = []
    for trip in ordered:
        new_band = not bands
        if bands:
            previous = bands[-1][-1]
            new_band = (
                profile_key(trip, config) != profile_key(previous, config)
                or trip.start is None
                or previous.start is None
            )
            if (
                trip.start is not None
                and previous.start is not None
                and config.max_band_gap is not None
            ):
                new_band = new_band or trip.start - previous.start > config.max_band_gap * 60
        if new_band:
            bands.append([])
        bands[-1].append(trip)
    return bands


def merge_stop_patterns(base: Sequence[str], pattern: Sequence[str]) -> list[str]:
    """Shortest common supersequence for compatible branch/short-turn columns."""
    n, m = len(base), len(pattern)
    common = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            common[i][j] = (
                common[i + 1][j + 1] + 1
                if base[i] == pattern[j]
                else max(common[i + 1][j], common[i][j + 1])
            )
    merged = []
    i = j = 0
    while i < n and j < m:
        if base[i] == pattern[j]:
            merged.append(base[i])
            i += 1
            j += 1
        elif common[i + 1][j] >= common[i][j + 1]:
            merged.append(base[i])
            i += 1
        else:
            merged.append(pattern[j])
            j += 1
    return merged + list(base[i:]) + list(pattern[j:])


def build_layouts(trips: Sequence[Trip]) -> tuple[list[list[str]], dict[tuple[str, ...], int]]:
    """Merge ordinary branches; keep conflicting/ambiguous loop patterns separate."""
    counts = Counter(trip.pattern for trip in trips)
    layouts: list[list[str]] = []
    assignment = {}
    for pattern in sorted(counts, key=lambda p: (-counts[p], -len(p), p)):
        for index, columns in enumerate(layouts):
            # Different loop patterns have ambiguous visit correspondence.
            # Separate tables preserve every visit without inventing that match.
            if len(set(columns)) < len(columns) or len(set(pattern)) < len(pattern):
                continue
            merged = merge_stop_patterns(columns, pattern)
            if len(merged) == len(set(columns) | set(pattern)):
                layouts[index] = merged
                assignment[pattern] = index
                break
        else:
            assignment[pattern] = len(layouts)
            layouts.append(list(pattern))
    return layouts, assignment


def embed_pattern(pattern: Sequence[str], columns: Sequence[str]) -> list[int]:
    """Embed every visit in order, including repeat visits in an exact loop layout."""
    result = []
    start = 0
    for stop_id in pattern:
        for index in range(start, len(columns)):
            if columns[index] == stop_id:
                result.append(index)
                start = index + 1
                break
        else:
            raise ValueError(f"Pattern {pattern} cannot be embedded in {columns}.")
    return result


def chronological_order(grid: Sequence[Sequence[int | None]]) -> tuple[list[int], bool]:
    """Stable topological sort from times at shared columns; flag overtaking cycles."""
    if not grid:
        return [], True
    successors: list[set[int]] = [set() for _ in grid]
    indegrees = [0] * len(grid)
    for column in range(len(grid[0])):
        groups: dict[int, list[int]] = defaultdict(list)
        for row_index, row in enumerate(grid):
            if row[column] is not None:
                value = row[column]
                if value is not None:
                    groups[value].append(row_index)
        previous: list[int] = []
        for value in sorted(groups):
            current = groups[value]
            for earlier in previous:
                for later in current:
                    if later not in successors[earlier]:
                        successors[earlier].add(later)
                        indegrees[later] += 1
            previous = current
    ready = [i for i, degree in enumerate(indegrees) if degree == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        current = heapq.heappop(ready)
        result.append(current)
        for later in sorted(successors[current]):
            indegrees[later] -= 1
            if indegrees[later] == 0:
                heapq.heappush(ready, later)
    return (result, True) if len(result) == len(grid) else (list(range(len(grid))), False)


def format_time(seconds: int | None, config: Config, exact: bool = False) -> str:
    """Format service-day time; internal calculations never use these strings."""
    if seconds is None:
        return ""
    show_seconds = config.show_seconds or exact
    value = seconds if show_seconds else ((seconds + 30) // 60) * 60
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    tail = f":{secs:02d}" if show_seconds else ""
    if config.time_format == "24" or exact:
        return f"{hours:02d}:{minutes:02d}{tail}"
    clock = hours % 24
    day = f" (+{hours // 24}d)" if hours >= 24 else ""
    return f"{clock % 12 or 12}:{minutes:02d}{tail} {'AM' if clock < 12 else 'PM'}{day}"


def stop_headers(pattern: Sequence[str], stops: dict[str, Stop], config: Config) -> list[str]:
    """Disambiguate codes, names, and repeated visits without changing stop IDs."""
    codes = Counter(stop.code for stop in stops.values() if stop.code)
    seen: Counter[str] = Counter()
    used: set[str] = set()
    headers = []
    for stop_id in pattern:
        stop = stops[stop_id]
        label_id = stop.code if config.stop_label_id == "stop_code" and stop.code else stop_id
        if config.stop_label_id == "stop_code" and stop.code and codes[stop.code] > 1:
            label_id += f"; id={stop_id}"
        label = f"{stop.name} [{label_id}]"
        seen[stop_id] += 1
        if seen[stop_id] > 1:
            label += f" (visit {seen[stop_id]})"
        while label in used:
            label += f" [{stop_id}]"
        used.add(label)
        headers.append(label)
    return headers


# =============================================================================
# PART 3 -- EXCEL AND AUDIT EXPORTS
# =============================================================================


def safe_component(value: str, limit: int = 45) -> str:
    """Filename-safe component, with a hash when cleaning/truncation could collide."""
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .") or "unknown"
    if clean != value or len(clean) > limit:
        return clean[:limit] + "_" + stable_id(value)[:8]
    return clean


def sheet_name(workbook: Workbook, wanted: str) -> str:
    """Generate an Excel-safe, case-insensitively unique sheet name."""
    base = re.sub(r"[:\\/*?\[\]]", "_", wanted).strip("'") or "Sheet"
    name = base[:31]
    count = 1
    existing = {name.casefold() for name in workbook.sheetnames}
    while name.casefold() in existing:
        count += 1
        suffix = f"_{count}"
        name = base[: 31 - len(suffix)] + suffix
    return name


def append_row(sheet: Worksheet, values: Sequence[object]) -> None:
    """Write literal text (including IDs beginning '=') and genuine numeric cells."""
    row = sheet.max_row + 1 if sheet.cell(1, 1).value is not None else 1
    for column, value in enumerate(values, 1):
        if value is None:
            continue
        cell = sheet.cell(row, column)
        if isinstance(value, str):
            cell.value = value
            cell.data_type = "s"
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            cell.value = value
            cell.number_format = "0.00" if isinstance(value, float) else "0"
        else:
            raise TypeError(f"Unsupported spreadsheet value: {type(value).__name__}")


def new_sheet(workbook: Workbook, name: str, header: Sequence[str]) -> Worksheet:
    """Create a plain table with a frozen header and readable column widths."""
    if len(header) > 16384:
        raise ValueError(f"{name}: exceeds Excel's column limit.")
    sheet = workbook.create_sheet(sheet_name(workbook, name))
    append_row(sheet, header)
    sheet.freeze_panes = "A2"
    maximum_lines = 1
    for index, title in enumerate(header, 1):
        width = min(max(len(title) + 2, 14), 36)
        sheet.column_dimensions[get_column_letter(index)].width = width
        sheet.cell(1, index).alignment = Alignment(wrap_text=True, vertical="top")
        maximum_lines = max(maximum_lines, math.ceil(len(title) / max(width - 4, 1)))
    sheet.row_dimensions[1].height = min(180, max(30, 16 * maximum_lines))
    return sheet


def display_distance(value: float | None, config: Config) -> float | None:
    """Convert units without rounding the stored number."""
    return (
        None
        if value is None
        else value * METERS_PER_UNIT[config.input_unit] / METERS_PER_UNIT[config.output_unit]
    )


def minutes(value: float | int | None) -> float | None:
    """Represent durations as numeric minutes, preserving fractional minutes."""
    return None if value is None else value / 60


def trip_totals(trip: Trip, config: Config) -> list[object]:
    """Common totals for workbook and CSV trip indexes."""
    return [
        display_distance(complete_sum(trip.distances[1:]), config),
        display_distance(trip.full_distance, config),
        minutes(complete_sum(trip.runtimes(config)[1:])),
        minutes(duration(trip.start, trip.end)),
        mean_speed(trip, config),
    ]


def write_timetables(
    workbook: Workbook, trips: list[Trip], feed: Feed, config: Config, audit: Audit
) -> None:
    """Retain branch-aware timetables and shared-stop trip ordering."""
    by_direction: dict[str, list[Trip]] = defaultdict(list)
    for trip in trips:
        by_direction[trip.direction].append(trip)
    for direction, group in sorted(by_direction.items()):
        layouts, assignment = build_layouts(group)
        for layout_index, columns in enumerate(layouts):
            selected = [t for t in group if assignment[t.pattern] == layout_index]
            selected.sort(key=trip_order)
            grid: list[list[int | None]] = []
            for trip in selected:
                row: list[int | None] = [None] * len(columns)
                for index, event in zip(embed_pattern(trip.pattern, columns), trip.events):
                    row[index] = (
                        arrival(event)
                        if event.sequence == trip.final_sequence
                        else departure(event)
                    )
                grid.append(row)
            order, consistent = chronological_order(grid)
            if not consistent:
                audit.add(
                    "OVERTAKING_OR_ORDER_CONFLICT",
                    f"Direction {direction or 'unspecified'}, layout {layout_index + 1}: no trip order is chronological at every shared stop. Original-departure order retained.",
                    {"route_id": group[0].route_id, "service_id": group[0].service_id},
                )
            sheet = new_sheet(
                workbook,
                f"D{direction or 'X'}_Timetable_{layout_index + 1}",
                [
                    "Block ID",
                    "Trip ID",
                    "Pattern ID",
                    "Headsign",
                    "Trip Start",
                    *stop_headers(columns, feed.stops, config),
                ],
            )
            for index in order:
                trip = selected[index]
                append_row(
                    sheet,
                    [
                        trip.block_id,
                        trip.trip_id,
                        trip.pattern_id,
                        trip.headsign,
                        format_time(trip.start, config),
                        *[format_time(t, config) for t in grid[index]],
                    ],
                )
            sheet.auto_filter.ref = sheet.dimensions


def band_prefix(band: list[Trip], label: str, config: Config) -> list[object]:
    """Use observed first/last departures; count every retained trip."""
    starts = [t.start for t in band if t.start is not None]
    return [
        label,
        band[0].pattern_id,
        band[0].full_pattern_id,
        band[0].shape_id,
        format_time(min(starts) if starts else None, config),
        format_time(max(starts) if starts else None, config),
        len(band),
        band[0].trip_id,
    ]


def write_metrics(
    workbook: Workbook, trips: list[Trip], feed: Feed, config: Config
) -> dict[str, str]:
    """Write exact-pattern tables and return trip-ID to shared band-ID mapping."""
    groups: dict[tuple[str, tuple[str, ...]], list[Trip]] = defaultdict(list)
    for trip in trips:
        groups[(trip.direction, trip.pattern)].append(trip)
    index = new_sheet(
        workbook,
        "Patterns",
        [
            "Direction",
            "Pattern ID",
            "Trip Count",
            "Retained Stops",
            "Full Pattern Count",
            "Table Prefix",
            "First Departure",
            "Last Departure",
        ],
    )
    band_lookup = {}
    for number, ((direction, pattern), group) in enumerate(sorted(groups.items()), 1):
        prefix = f"D{direction or 'X'}_P{number:02d}"
        starts = [t.start for t in group if t.start is not None]
        append_row(
            index,
            [
                direction,
                group[0].pattern_id,
                len(group),
                len(pattern),
                len({t.full_pattern for t in group}),
                prefix,
                format_time(min(starts) if starts else None, config),
                format_time(max(starts) if starts else None, config),
            ],
        )
        labels = stop_headers(pattern, feed.stops, config)
        segments = [f"{labels[i - 1]} -> {labels[i]}" for i in range(1, len(labels))]
        common = [
            "Band ID",
            "Pattern ID",
            "Full Pattern ID",
            "Shape ID",
            "First Departure",
            "Last Departure",
            "Trip Count",
            "Example Trip ID",
        ]
        distance_sheet = (
            new_sheet(
                workbook,
                prefix + "_Dist",
                [
                    *common,
                    f"Total ({config.output_unit})",
                    *[f"{name} ({config.output_unit})" for name in segments],
                ],
            )
            if config.distances
            else None
        )
        runtime_header = [*common, "Total Segment Runtime (min)"]
        runtime_header.append(f"{labels[0]} Holding (min)")
        for name, label in zip(segments, labels[1:]):
            runtime_header.extend([f"{name} Runtime (min)", f"{label} Holding (min)"])
        runtime_sheet = (
            new_sheet(workbook, prefix + "_Run", runtime_header) if config.runtimes else None
        )
        speed_sheet = (
            new_sheet(
                workbook,
                prefix + "_Speed",
                [*common, "Mean mph", *[f"{name} mph" for name in segments]],
            )
            if config.speeds
            else None
        )
        for band_number, band in enumerate(make_bands(group, config), 1):
            label = f"{prefix}_B{band_number:03d}"
            for trip in band:
                band_lookup[trip.trip_id] = label
            trip = band[0]
            base = band_prefix(band, label, config)
            if distance_sheet is not None:
                append_row(
                    distance_sheet,
                    [
                        *base,
                        display_distance(complete_sum(trip.distances[1:]), config),
                        *[display_distance(d, config) for d in trip.distances[1:]],
                    ],
                )
            if runtime_sheet is not None:
                runtimes = trip.runtimes(config)
                row: list[object] = [
                    *base,
                    minutes(complete_sum(runtimes[1:])),
                    minutes(trip.holding[0]),
                ]
                for run, hold in zip(runtimes[1:], trip.holding[1:]):
                    row.extend([minutes(run), minutes(hold)])
                append_row(runtime_sheet, row)
            if speed_sheet is not None:
                append_row(
                    speed_sheet,
                    [
                        *base,
                        mean_speed(trip, config),
                        *[
                            speed_mph(d, t, config)
                            for d, t in zip(trip.distances[1:], trip.runtimes(config)[1:])
                        ],
                    ],
                )
        for sheet in (distance_sheet, runtime_sheet, speed_sheet):
            if sheet is not None:
                sheet.auto_filter.ref = sheet.dimensions
    return band_lookup


def trip_header(config: Config) -> list[str]:
    """Common metadata schema for workbook and CSV outputs."""
    return [
        "Block ID",
        "Trip ID",
        "Route ID",
        "Route",
        "Service ID",
        "Direction",
        "Pattern ID",
        "Full Pattern ID",
        "Band ID",
        "Shape ID",
        "Headsign",
        "Start",
        "End",
        "Full Stop Count",
        "Retained Stop Count",
        f"Displayed Distance ({config.output_unit})",
        f"Full Distance ({config.output_unit})",
        "Total Segment Runtime (min)",
        "Trip Elapsed (min)",
        "Mean mph",
    ]


def trip_row(trip: Trip, band_id: str, config: Config) -> list[object]:
    """Preserve all IDs as text and export true full-trip times."""
    return [
        trip.block_id,
        trip.trip_id,
        trip.route_id,
        trip.route_name,
        trip.service_id,
        trip.direction,
        trip.pattern_id,
        trip.full_pattern_id,
        band_id,
        trip.shape_id,
        trip.headsign,
        format_time(trip.start, config, exact=True),
        format_time(trip.end, config, exact=True),
        trip.full_stop_count,
        len(trip.events),
        *trip_totals(trip, config),
    ]


def save_workbook(workbook: Workbook, path: Path) -> None:
    """Atomic replacement; a failed save leaves the previous workbook intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".gtfs_", suffix=".xlsx", dir=path.parent)
    os.close(descriptor)
    try:
        workbook.save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
        workbook.close()


def export_workbook(
    trips: list[Trip], feed: Feed, config: Config, output: Path, audit: Audit
) -> tuple[Path, dict[str, str]]:
    """Export one route/service independently; caller records any failure."""
    first = trips[0]
    workbook = Workbook()
    active = workbook.active
    if active is not None:
        workbook.remove(active)
    summary = new_sheet(workbook, "Readme", ["Setting", "Value"])
    for row in [
        ["Script version", VERSION],
        ["Source", str(config.gtfs_path)],
        ["Route ID", first.route_id],
        ["Route", first.route_name],
        ["Service ID", first.service_id],
        ["Service label", feed.service_labels.get(first.service_id, first.service_id)],
        ["Trips", len(trips)],
        ["Input distance unit", config.input_unit],
        ["Output distance unit", config.output_unit],
        ["Runtime basis", config.runtime_basis],
        ["Band mode", config.band_mode],
        ["Band gap limit (minutes)", config.max_band_gap],
        ["Band times", "First/last actual departures of matching trips; NOT validity windows."],
        [
            "Holding",
            "Arrival-to-departure at retained stops. Runtime departure_to_arrival excludes downstream holding; intermediate-stop dwell remains included.",
        ],
        [
            "Timetable times",
            "Departure at each visit; arrival at the ORIGINAL final stop. Missing preferred times use the other field and are flagged.",
        ],
        [
            "Timetable layouts",
            "Compatible branches share columns; conflicting stop order or ambiguous loop visits use separate sheets.",
        ],
        ["Missing data", "Blank cells mean unknown; totals require every contributing segment."],
        [
            "Pattern ID",
            "Ordered retained stop IDs, including repeats. Full Pattern ID includes all original stops.",
        ],
        [
            "Metric profiles",
            "All metrics share band boundaries; shape/full-pattern/distance/time/holding changes split bands.",
        ],
        [
            "Units",
            "GTFS distances are unitless by specification; INPUT_DISTANCE_UNIT must match your producer.",
        ],
        [
            "Calendar",
            "Service IDs are kept separate; labels do not select an operating date or apply holiday exceptions to other service IDs.",
        ],
    ]:
        append_row(summary, row)
    summary.column_dimensions["B"].width = 100
    if config.timetables:
        write_timetables(workbook, trips, feed, config, audit)
    bands = write_metrics(workbook, trips, feed, config)
    trip_sheet = new_sheet(workbook, "Trips", trip_header(config))
    for trip in sorted(trips, key=trip_order):
        append_row(trip_sheet, trip_row(trip, bands[trip.trip_id], config))
    relevant = audit.relevant(first.route_id, first.service_id)
    issue_sheet = new_sheet(workbook, "Issues", ["Severity", "Code", "Trip ID", "Message"])
    for issue in relevant:
        append_row(issue_sheet, [issue.severity, issue.code, issue.trip_id, issue.message])
    folder = output / (
        safe_component(feed.service_labels.get(first.service_id, "Service"))
        + "_sid_"
        + safe_component(first.service_id)
    )
    basename = (
        f"route_{safe_component(first.route_name)}_rid_{safe_component(first.route_id)}"
        f"_sid_{safe_component(first.service_id)}_schedule"
    )
    suffix = config.warning_suffix if relevant else ""
    path = folder / f"{basename}{suffix}.xlsx"
    save_workbook(workbook, path)
    # With stable output folders, remove only this exporter's exact alternate name,
    # and only after the new version is safely written.
    alternate = folder / f"{basename}{'' if relevant else config.warning_suffix}.xlsx"
    if alternate != path:
        alternate.unlink(missing_ok=True)
    LOGGER.info("Wrote %s (%d trips, %d sheets).", path.name, len(trips), len(workbook.sheetnames))
    return path, bands


def write_csv(path: Path, header: Sequence[str], rows: Iterator[Sequence[object]]) -> None:
    """Write typed audit values with standard CSV quoting and UTF-8 BOM for Excel."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def export_details(
    trips: Sequence[Trip], bands: dict[str, str], output: Path, config: Config
) -> None:
    """Expose per-trip and per-segment calculation lineage without merging IDs."""
    write_csv(
        output / "trips.csv",
        trip_header(config),
        (trip_row(trip, bands.get(trip.trip_id, ""), config) for trip in trips),
    )
    header = [
        "Route ID",
        "Service ID",
        "Direction",
        "Trip ID",
        "Pattern ID",
        "Band ID",
        "Segment",
        "From Stop ID",
        "To Stop ID",
        "From Sequence",
        "To Sequence",
        "From Departure",
        "To Arrival",
        "To Departure",
        f"Distance ({config.output_unit})",
        "Departure-Arrival (seconds)",
        "Departure-Departure (seconds)",
        "Downstream Holding (seconds)",
        "Selected Runtime (seconds)",
        "mph",
    ]

    def segment_rows() -> Iterator[Sequence[object]]:
        for trip in trips:
            for i in range(1, len(trip.events)):
                previous, current = trip.events[i - 1], trip.events[i]
                yield [
                    trip.route_id,
                    trip.service_id,
                    trip.direction,
                    trip.trip_id,
                    trip.pattern_id,
                    bands.get(trip.trip_id, ""),
                    i,
                    previous.stop_id,
                    current.stop_id,
                    previous.sequence,
                    current.sequence,
                    format_time(departure(previous), config, True),
                    format_time(arrival(current), config, True),
                    format_time(departure(current), config, True),
                    display_distance(trip.distances[i], config),
                    trip.running[i],
                    trip.elapsed[i],
                    trip.holding[i],
                    trip.runtimes(config)[i],
                    speed_mph(trip.distances[i], trip.runtimes(config)[i], config),
                ]

    write_csv(output / "segments.csv", header, segment_rows())


# =============================================================================
# PART 4 -- RUN COORDINATION AND LOGGING
# =============================================================================


def validate_config(config: Config) -> None:
    """Fail early on unsupported settings or accidental placeholder paths."""
    if str(config.gtfs_path).startswith("Path\\To\\") or str(config.output_dir).startswith(
        "Path\\To\\"
    ):
        raise ValueError("Edit GTFS_PATH and OUTPUT_DIR in CONFIGURATION before running.")
    if config.input_unit not in METERS_PER_UNIT or config.output_unit not in METERS_PER_UNIT:
        raise ValueError("Distance unit must be feet, meters, km, or miles.")
    choices = [
        (config.runtime_basis, {"departure_to_arrival", "departure_to_departure"}, "RUNTIME_BASIS"),
        (config.band_mode, {"consecutive", "profile_summary"}, "BAND_MODE"),
        (config.frequency_policy, {"error", "skip"}, "FREQUENCY_POLICY"),
        (config.time_format, {"12", "24"}, "TIME_FORMAT"),
        (config.stop_label_id, {"stop_id", "stop_code"}, "STOP_LABEL_ID"),
    ]
    for value, allowed, label in choices:
        if value not in allowed:
            raise ValueError(f"{label} must be one of {sorted(allowed)}.")
    if not any((config.timetables, config.distances, config.runtimes, config.speeds)):
        raise ValueError("Enable at least one workbook export.")
    if config.max_band_gap is not None and (
        not math.isfinite(config.max_band_gap) or config.max_band_gap <= 0
    ):
        raise ValueError("MAX_BAND_GAP_MINUTES must be positive or None.")
    if not math.isfinite(config.speed_warning) or config.speed_warning <= 0:
        raise ValueError("SPEED_WARNING_MPH must be finite and positive.")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', config.warning_suffix):
        raise ValueError("WARNING_FILENAME_SUFFIX contains invalid filename characters.")


def write_audit(
    output: Path,
    config: Config,
    audit: Audit,
    summary: dict[str, object],
    written: list[Path],
    failed: list[str],
) -> None:
    """Always capture effective settings, issue counts, failed trips, and exports."""
    write_csv(
        output / "issues.csv",
        ["Severity", "Code", "Route ID", "Service ID", "Trip ID", "Message"],
        (
            [i.severity, i.code, i.route_id, i.service_id, i.trip_id, i.message]
            for i in audit.issues
        ),
    )
    lines = [
        "GTFS SCHEDULE EXPORTER RUN LOG",
        f"Version: {VERSION}",
        f"Timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "RUN SUMMARY",
        json.dumps(summary, indent=2, default=str),
        "",
        "EFFECTIVE CONFIGURATION",
        json.dumps(asdict(config), indent=2, default=str),
        "",
        "ISSUE COUNTS",
        json.dumps(dict(sorted(audit.counts.items())), indent=2),
        "",
        "WRITTEN WORKBOOKS",
        *[str(p.relative_to(output)) for p in written],
        "",
        "FAILED TRIPS / EXPORTS",
        *(failed or ["None"]),
        "",
        "ALL ISSUES (also available in issues.csv)",
        *[
            f"{i.severity} | {i.code} | route={i.route_id} | service={i.service_id} | trip={i.trip_id} | {i.message}"
            for i in audit.issues
        ],
    ]
    (output / "gtfs_schedule_exporter_runlog.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(config: Config) -> int:
    """Export selected trips, continue independent work after failures, reconcile counts."""
    validate_config(config)
    output = config.output_dir
    if config.run_subfolder:
        output = output / datetime.now().strftime("gtfs_schedule_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    written: list[Path] = []
    failed: list[str] = []
    trips: list[Trip] = []
    rejected_ids: set[str] = set()
    bands: dict[str, str] = {}
    summary: dict[str, object] = {"output_folder": str(output)}
    exported_count = 0
    LOGGER.info("Starting combined GTFS export. Output: %s", output)
    try:
        feed = load_feed(config, audit)
        summary.update(
            input_trips=feed.input_trip_count,
            selected_trips=feed.selected_trip_count,
            frequency_trips_skipped=feed.frequency_skipped,
        )
        for index, (trip_id, metadata) in enumerate(sorted(feed.trips.items()), 1):
            try:
                trips.append(
                    build_trip(metadata, feed.stop_times.get(trip_id, []), feed, config, audit)
                )
            except ValueError as exc:
                rejected_ids.add(trip_id)
                audit.add("TRIP_REJECTED", str(exc), metadata, "ERROR")
                failed.append(f"Trip {trip_id}: {exc}")
            if index % 1000 == 0:
                LOGGER.info("Calculated %s / %s trips...", f"{index:,}", f"{len(feed.trips):,}")
        groups: dict[tuple[str, str], list[Trip]] = defaultdict(list)
        for trip in trips:
            groups[(trip.route_id, trip.service_id)].append(trip)
        for (route_id, service_id), group in sorted(groups.items()):
            try:
                path, lookup = export_workbook(group, feed, config, output, audit)
                written.append(path)
                bands.update(lookup)
                exported_count += len(group)
            except Exception as exc:  # Isolate export failures; never report a successful full run.
                LOGGER.exception("Workbook failed: route=%s, service=%s", route_id, service_id)
                audit.add(
                    "EXPORT_FAILED",
                    str(exc),
                    {"route_id": route_id, "service_id": service_id},
                    "ERROR",
                )
                failed.append(f"Route {route_id}, service {service_id}: {exc}")
        if not feed.trips:
            audit.add(
                "NO_SELECTED_TRIPS", "No trips remain after route/service/frequency filtering."
            )
        if config.detail_csv:
            export_details(trips, bands, output, config)
        valid_ids = {trip.trip_id for trip in trips}
        reconciled = (
            len(valid_ids) == len(trips)
            and valid_ids.isdisjoint(rejected_ids)
            and valid_ids | rejected_ids == set(feed.trips)
            and feed.selected_trip_count
            == feed.frequency_skipped + len(valid_ids) + len(rejected_ids)
        )
        if not reconciled:
            raise RuntimeError(
                "Selected-trip reconciliation failed; inspect rejected trips and outputs."
            )
        summary.update(
            valid_trips=len(trips),
            rejected_trips=len(rejected_ids),
            workbook_trips=exported_count,
            valid_trips_not_in_workbooks=len(valid_ids - bands.keys()),
            workbooks=len(written),
            selected_reconciliation=reconciled,
        )
    except Exception as exc:
        LOGGER.exception("GTFS run failed")
        audit.add("RUN_FAILED", str(exc), severity="ERROR")
        failed.append(str(exc))
    summary.update(issues=len(audit.issues), failures=len(failed), complete=not failed)
    try:
        write_audit(output, config, audit, summary, written, failed)
    except OSError:
        LOGGER.exception("Could not write the mandatory run log / issues.csv.")
        return 1
    if failed:
        LOGGER.error(
            "Finished with %d failure(s). Review %s",
            len(failed),
            output / "gtfs_schedule_exporter_runlog.txt",
        )
        return 1
    LOGGER.info(
        "Completed: %d trips in %d workbooks; %d review findings. Output: %s",
        exported_count,
        len(written),
        len(audit.issues),
        output,
    )
    return 0


def main(config: Config | None = None) -> int:
    """Entry point usable unchanged from a notebook or ordinary Python process."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        return run(config if config is not None else configured_settings())
    except (OSError, ValueError) as exc:
        LOGGER.error("Cannot start export: %s", exc)
        return 1


if __name__ == "__main__":
    status = main()
    if "ipykernel" not in sys.modules:
        raise SystemExit(status)
