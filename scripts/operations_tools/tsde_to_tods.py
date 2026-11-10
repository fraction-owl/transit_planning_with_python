"""Reconstruct Trapeze TSDE blocks and export conservative TODS 2.1 supplements.

Python 3.9+; standard library only. Edit CONFIG below or use --help.
Input: TSDE XML ZIP/folder, optionally the corresponding GTFS ZIP/folder.
Output: timestamped audit folder and, with matched GTFS, draft_tods.zip.

Event meanings below are inferred from the supplied TSDE 19.0.17 export.
They have structural checks, but have not been confirmed against Trapeze's ICD.
Whole blocks with unresolved movements or incomplete GTFS matches are excluded.
Run pieces/pay components are audited; operator run_events are deferred in v0.1.
Reference: https://tods-transit.org/spec/ (TODS 2.1).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import IO, Optional

# =============================================================================
# EDIT HERE: absolute paths also work in an IDE, notebook, or from another folder.
# Relative paths are resolved against this script, NOT the working directory.
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TSDE_PATH = SCRIPT_DIR / "2026-June.zip"
GTFS_PATH: Optional[Path] = None  # Set to the corresponding GTFS ZIP or folder.
OUTPUT_DIR = SCRIPT_DIR / "output"
EXPORT_PREFIX: Optional[str] = None  # e.g. "20260522"; required if several exports.
STOP_CODE_ATTRIBUTE = ""  # Blank: TSDE stop pid -> GTFS stop_code (verified here).
GTFS_STOP_KEY = "stop_code"  # Alternative: "stop_id".
MAX_STOP_DISTANCE_METERS = 75.0
COORDINATE_SCALE = 1_000_000.0
# Exact timing at endpoints and GTFS timepoint=1 stops; interpolated stops may vary.
TIME_TOLERANCE_SECONDS = 0
EVENT_CODES_VERIFIED = False  # Set only after independent ICD/paddle validation.
REQUIRE_ALL_BLOCKS = False  # True: suppress the ZIP if any block is ineligible.

VERSION = "0.1.0"
EVENT_CODES = {
    "1": "pull_out_start",
    "15": "pull_out_end",
    "3": "scheduled_trip_start",
    "16": "pull_in_start",
    "2": "pull_in_end",
    "17": "intertrip_deadhead_start",
    "18": "intertrip_deadhead_end",
}
PAIRS = {"1": ("15", "pull-out"), "16": ("2", "pull-back"), "17": ("18", "deadhead")}
WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
Row = dict[str, str]
PointKey = tuple[str, str]
TripKey = tuple[str, str]
BlockKey = tuple[str, str]


@dataclass
class Config:
    tsde: Path = TSDE_PATH
    gtfs: Optional[Path] = GTFS_PATH
    output: Path = OUTPUT_DIR
    prefix: Optional[str] = EXPORT_PREFIX
    stop_attribute: str = STOP_CODE_ATTRIBUTE
    stop_key: str = GTFS_STOP_KEY
    max_stop_distance: float = MAX_STOP_DISTANCE_METERS
    coordinate_scale: float = COORDINATE_SCALE
    time_tolerance: int = TIME_TOLERANCE_SECONDS
    event_codes_verified: bool = EVENT_CODES_VERIFIED
    require_all_blocks: bool = REQUIRE_ALL_BLOCKS


@dataclass
class Issue:
    severity: str
    code: str
    service_id: str = ""
    block_id: str = ""
    trip_id: str = ""
    detail: str = ""


@dataclass
class QA:
    issues: list[Issue] = field(default_factory=list)

    def add(
        self,
        severity: str,
        code: str,
        detail: str,
        block: Optional[BlockKey] = None,
        trip: str = "",
    ) -> None:
        sid, bid = block if block is not None else ("", "")
        self.issues.append(Issue(severity, code, sid, bid, trip, detail))

    def bad_blocks(self) -> set[BlockKey]:
        return {
            (i.service_id, i.block_id) for i in self.issues if i.severity == "ERROR" and i.block_id
        }


class Source:
    """Read a ZIP or folder without extracting archive members."""

    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.is_file() else None
        if self.zip is not None:
            self.names = [n for n in self.zip.namelist() if not n.endswith("/")]
        elif path.is_dir():
            self.names = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]
        else:
            raise FileNotFoundError(path)

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()

    def find(self, basename: str, required: bool = True) -> Optional[str]:
        matches = [n for n in self.names if Path(n).name.lower() == basename.lower()]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous input: multiple {basename} files in {self.path}.")
        if not matches:
            if required:
                raise ValueError(f"Required file {basename} missing from {self.path}.")
            return None
        return matches[0]

    @contextmanager
    def open(self, name: str) -> Generator[IO[bytes], None, None]:
        stream = self.zip.open(name) if self.zip is not None else (self.path / name).open("rb")
        try:
            yield stream
        finally:
            stream.close()

    def text(self, basename: str) -> str:
        name = self.find(basename, required=False)
        if name is None:
            return ""
        with self.open(name) as stream:
            return stream.read().decode("utf-8-sig")

    def xml(self, prefix: str, tag: str, required: bool = True) -> Iterator[Row]:
        name = self.find(f"{prefix}_{tag}.xml", required)
        if name is None:
            return
        with self.open(name) as stream:
            context = ET.iterparse(stream, events=("start", "end"))
            _, root = next(context)
            if root.tag != "file":
                raise ValueError(f"Unexpected XML root in {name}: {root.tag}.")
            for event, element in context:
                if event == "end" and element.tag == tag:
                    yield {child.tag: (child.text or "").strip() for child in element}
                    root.clear()

    def csv(self, basename: str, required: bool = True) -> list[Row]:
        name = self.find(basename, required)
        if name is None:
            return []
        with self.open(name) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
            result = []
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"Malformed CSV row in {basename}.")
                result.append({k.strip(): v.strip() for k, v in row.items()})
            return result


def number(row: Row, key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing/invalid integer {key}: {row}") from exc


def clock(seconds: int) -> str:
    if seconds < 0:
        raise ValueError("Negative time cannot be formatted as a GTFS time.")
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", value)
    if match is None:
        raise ValueError(f"Invalid or missing GTFS time: {value!r}")
    h, m, s = map(int, match.groups())
    return h * 3600 + m * 60 + s


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def dates_between(start: date, end: date) -> list[date]:
    if end < start or (end - start).days > 3660:
        raise ValueError(f"Unsupported calendar span: {start} to {end}.")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def unique(rows: Iterable[Row], key: str) -> dict[str, Row]:
    result: dict[str, Row] = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}.")
        result[value] = row
    return result


def route_name(name: str) -> str:
    return "".join(name.upper().split())


def point_key(row: Row, prefix: str = "") -> PointKey:
    return row[f"{prefix}ptype"], row[f"{prefix}pid"]


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    v = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6_371_008.8 * 2 * math.asin(min(1.0, math.sqrt(v)))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Call:
    sequence: int
    point: PointKey
    time: int


@dataclass
class Trip:
    sid: str
    tid: str
    lid: str
    direction: str
    route: str
    ecid: str
    calls: list[Call] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def nodes(self) -> list[Call]:
        return [c for c in self.calls if c.point[0] == "no"]

    @property
    def stops(self) -> list[Call]:
        return [c for c in self.calls if c.point[0] == "st"]


@dataclass
class TSDE:
    version: Row
    prefix: str
    points: dict[PointKey, Row]
    services: dict[str, Row]
    service_dates: dict[str, set[date]]
    trips: dict[TripKey, Trip]
    blocks: dict[BlockKey, list[Row]]
    tables: dict[str, list[Row]]
    export_version: str


# =============================================================================
# STAGE 1: Source ingestion and calendar interpretation.
# =============================================================================
def read_tsde(source: Source, cfg: Config, qa: QA) -> TSDE:
    prefixes = sorted(
        {Path(n).name[:-8] for n in source.names if Path(n).name.lower().endswith("_ver.xml")}
    )
    prefix = cfg.prefix
    if prefix is None:
        if len(prefixes) != 1:
            raise ValueError(f"Choose --prefix; schedule exports found: {prefixes}.")
        prefix = prefixes[0]
    versions = list(source.xml(prefix, "ver"))
    if len(versions) != 1:
        raise ValueError("Expected exactly one schedule version record.")
    version = versions[0]
    tables: dict[str, list[Row]] = {}
    for tag in (
        "pnt",
        "svc",
        "cal",
        "lin",
        "blk",
        "deh",
        "run",
        "rua",
        "pty",
        "div",
        "dgr",
        "exc",
        "ecb",
    ):
        tables[tag] = list(
            source.xml(prefix, tag, required=tag in {"pnt", "svc", "cal", "lin", "blk", "deh"})
        )
        logging.info("Read %-4s: %s records", tag, f"{len(tables[tag]):,}")
        for row in tables[tag]:
            if row.get("verid", version["verid"]) != version["verid"]:
                raise ValueError(f"Mixed schedule versions in {tag}.")
    points: dict[PointKey, Row] = {}
    for row in tables["pnt"]:
        key = point_key(row)
        if key in points:
            raise ValueError(f"Duplicate typed point: {key}.")
        points[key] = dict(row)
        points[key]["stop_code"] = row["pid"] if row["ptype"] == "st" else ""
    for row in source.xml(prefix, "pta", required=False):
        if row["aname"] == cfg.stop_attribute and row["avalue"] and point_key(row) in points:
            points[point_key(row)]["stop_code"] = row["avalue"]
    services = unique(tables["svc"], "sid")
    days = dates_between(parse_date(version["fdate"]), parse_date(version["tdate"]))
    active: dict[str, set[date]] = {}
    for sid, row in services.items():
        mask = row["wktplt"]
        if len(mask) != 7 or set(mask) - {"0", "1"}:
            raise ValueError(f"Invalid weekly template for service {sid}: {mask}.")
        active[sid] = {d for d in days if mask[(d.weekday() + 1) % 7] == "1"}
    overrides: dict[date, str] = {}
    for row in tables["cal"]:
        if row.get("ecid", "0") != "0":
            raise ValueError("Calendar exception combinations require a documented mapping.")
        day, sid = parse_date(row["cdate"]), row["sid"]
        if sid not in active or (day in overrides and overrides[day] != sid):
            raise ValueError(f"Ambiguous calendar override on {day}.")
        overrides[day] = sid
        if day in days:
            for ds in active.values():
                ds.discard(day)
            active[sid].add(day)
    lines: dict[str, str] = {}
    for row in tables["lin"]:
        name = route_name(row["sname"])
        if row["lid"] in lines and lines[row["lid"]] != name:
            raise ValueError("Line ID has conflicting route names.")
        lines[row["lid"]] = name
    trips: dict[TripKey, Trip] = {}
    for row in source.xml(prefix, "trp"):
        if row["verid"] != version["verid"]:
            raise ValueError("Mixed versions in trip table.")
        key = row["sid"], row["tid"]
        metadata = (row["lid"], row["did"], row["ecid"])
        if key not in trips:
            trips[key] = Trip(*key, row["lid"], row["did"], lines[row["lid"]], row["ecid"])
        trip = trips[key]
        if row.get("wktplt") != services[trip.sid]["wktplt"]:
            if "Unsupported trip-specific weekly template." not in trip.errors:
                trip.errors.append("Unsupported trip-specific weekly template.")
        if metadata != (trip.lid, trip.direction, trip.ecid):
            trip.errors.append("Conflicting metadata/exception variants for the same trip ID.")
        trip.calls.append(Call(number(row, "lpsnum"), point_key(row), number(row, "dtime")))
    for trip in trips.values():
        trip.calls.sort(key=lambda c: (c.sequence, c.point[0], c.point[1]))
        if not trip.nodes:
            trip.errors.append("Missing terminal nodes.")
        if trip.ecid != "0":
            trip.errors.append("Trip uses an unsupported exception combination.")
        if any(c.time < 0 or c.point not in points for c in trip.calls):
            trip.errors.append("Negative time or missing point reference.")
        if len({(c.sequence, c.point[0]) for c in trip.calls}) != len(trip.calls):
            trip.errors.append("Ambiguous repeated point type at a sequence position.")
        for calls in (trip.nodes, trip.stops):
            if any(b.time < a.time for a, b in zip(calls, calls[1:])):
                trip.errors.append("Times decrease along the trip.")
        if trip.nodes and (
            trip.nodes[0].sequence != trip.calls[0].sequence
            or trip.nodes[-1].sequence != trip.calls[-1].sequence
        ):
            trip.errors.append("Terminal nodes do not bracket the trip.")
    blocks: dict[BlockKey, list[Row]] = defaultdict(list)
    for row in tables["blk"]:
        blocks[row["sid"], row["bid"]].append(row)
    for rows in blocks.values():
        rows.sort(key=lambda r: number(r, "snum"))
    export_log = source.text("log.log")
    for line in export_log.splitlines():
        if "*E*" in line or "not completed successfully" in line:
            qa.add("WARNING", "SOURCE_EXPORT_WARNING", line.strip())
    if not cfg.event_codes_verified:
        qa.add(
            "WARNING",
            "INFERRED_EVENT_CODES",
            "Event meanings inferred from TSDE 19.0.17 sequences; independently verify before operational use.",
        )
    logging.info(
        "Read %s distinct trips and %s service/block combinations",
        len(trips),
        len(blocks),
    )
    return TSDE(
        version,
        prefix,
        points,
        services,
        active,
        trips,
        dict(blocks),
        tables,
        source.text("TSDEVersion.txt"),
    )


@dataclass
class Movement:
    sid: str
    bid: str
    sequence: int
    kind: str
    start: int
    end: int
    origin: PointKey
    destination: PointKey
    from_trip: str = ""
    to_trip: str = ""
    source_codes: str = ""

    @property
    def key(self) -> BlockKey:
        return self.sid, self.bid


# =============================================================================
# STAGE 2: Reconstruct movements; quarantine entire ambiguous blocks.
# =============================================================================
def reconstruct(data: TSDE, qa: QA) -> dict[BlockKey, list[Movement]]:
    result: dict[BlockKey, list[Movement]] = {}
    ownership: dict[TripKey, set[BlockKey]] = defaultdict(set)
    for key, rows in data.blocks.items():
        sid, bid = key
        movements: list[Movement] = []
        if len({r["snum"] for r in rows}) != len(rows):
            qa.add("ERROR", "DUPLICATE_BLOCK_SEQUENCE", "Duplicate sequence numbers.", key)
        if sum(r["etype"] == "1" for r in rows) != 1 or sum(r["etype"] == "2" for r in rows) != 1:
            qa.add(
                "ERROR",
                "MULTIPLE_BLOCK_EXTENTS",
                "Expected one pull-out and one pull-in; source may contain variants or split vehicle work.",
                key,
            )
        consumed: set[int] = set()
        for i, row in enumerate(rows):
            code, tid = row["etype"], row["tid"]
            if code not in EVENT_CODES:
                qa.add("ERROR", "UNKNOWN_EVENT_CODE", code, key, tid)
            if row.get("ecid", "0") != "0" or row.get("ecin", "N") != "N":
                qa.add(
                    "ERROR",
                    "BLOCK_EXCEPTION",
                    "Unsupported block exception semantics.",
                    key,
                    tid,
                )
            if point_key(row) not in data.points or number(row, "etime") < 0:
                qa.add(
                    "ERROR",
                    "INVALID_BLOCK_EVENT",
                    "Missing point or negative event time.",
                    key,
                    tid,
                )
            if code == "3":
                trip = data.trips.get((sid, tid))
                ownership[sid, tid].add(key)
                if trip is None or trip.errors or not trip.nodes:
                    qa.add(
                        "ERROR",
                        "INVALID_TRIP",
                        "; ".join(trip.errors) if trip else "Trip not found.",
                        key,
                        tid,
                    )
                    continue
                first, last = trip.nodes[0], trip.nodes[-1]
                if first.time != number(row, "etime") or first.point != point_key(row):
                    qa.add(
                        "ERROR",
                        "TRIP_START_CONFLICT",
                        "Block event differs from trip's first node.",
                        key,
                        tid,
                    )
                movements.append(
                    Movement(
                        sid,
                        bid,
                        number(row, "snum"),
                        "scheduled_trip",
                        first.time,
                        last.time,
                        first.point,
                        last.point,
                        tid,
                        tid,
                        code,
                    )
                )
            elif code in PAIRS:
                end_code, kind = PAIRS[code]
                if i + 1 >= len(rows) or rows[i + 1]["etype"] != end_code:
                    qa.add(
                        "ERROR",
                        "UNPAIRED_EVENT",
                        f"{code} must be followed by {end_code}.",
                        key,
                        tid,
                    )
                    continue
                end = rows[i + 1]
                consumed.add(i + 1)
                if (row["tid"], row["ttid"]) != (end["tid"], end["ttid"]):
                    qa.add(
                        "ERROR",
                        "PAIR_TRIP_CONFLICT",
                        "Paired events reference different trips.",
                        key,
                        tid,
                    )
                movement = Movement(
                    sid,
                    bid,
                    number(row, "snum"),
                    kind,
                    number(row, "etime"),
                    number(end, "etime"),
                    point_key(row),
                    point_key(end),
                    tid if code != "1" else "",
                    row["ttid"] if code != "16" else "",
                    f"{code}/{end_code}",
                )
                movements.append(movement)
                if movement.end < movement.start or (
                    movement.end == movement.start and movement.origin != movement.destination
                ):
                    qa.add(
                        "ERROR",
                        "INVALID_MOVEMENT_TIME",
                        "Negative duration or zero-duration movement between different points.",
                        key,
                        tid,
                    )
                for trip_id, endpoint, event_time, is_start in (
                    (movement.from_trip, movement.origin, movement.start, True),
                    (movement.to_trip, movement.destination, movement.end, False),
                ):
                    if not trip_id:
                        continue
                    trip = data.trips.get((sid, trip_id))
                    if trip is None or not trip.nodes:
                        qa.add(
                            "ERROR",
                            "MISSING_ANCHOR_TRIP",
                            "Deadhead anchor trip/node missing.",
                            key,
                            trip_id,
                        )
                        continue
                    call = trip.nodes[-1] if is_start else trip.nodes[0]
                    wrong_time = event_time < call.time if is_start else event_time > call.time
                    if call.point != endpoint or wrong_time:
                        qa.add(
                            "ERROR",
                            "DEADHEAD_ANCHOR_CONFLICT",
                            "Deadhead endpoint/time conflicts with its associated trip.",
                            key,
                            trip_id,
                        )
                if code == "1" and movement.origin[0] != "ga":
                    qa.add("ERROR", "PULLOUT_NOT_GARAGE", str(movement.origin), key)
                if code == "16" and movement.destination[0] != "ga":
                    qa.add("ERROR", "PULLIN_NOT_GARAGE", str(movement.destination), key)
            elif code in {"15", "2", "18"} and i not in consumed:
                qa.add("ERROR", "ORPHAN_END_EVENT", code, key, tid)
        movements.sort(key=lambda m: m.sequence)
        with_gaps: list[Movement] = []
        for i, movement in enumerate(movements):
            if i:
                previous = movements[i - 1]
                if movement.start < previous.end:
                    qa.add(
                        "ERROR",
                        "MOVEMENT_OVERLAP",
                        f"{previous.sequence} ends after {movement.sequence} starts.",
                        key,
                    )
                elif previous.destination != movement.origin:
                    qa.add(
                        "ERROR",
                        "UNRESOLVED_LOCATION_GAP",
                        f"{previous.destination} to {movement.origin} has no explicit movement.",
                        key,
                    )
                elif movement.start > previous.end:
                    with_gaps.append(
                        Movement(
                            sid,
                            bid,
                            0,
                            "layover",
                            previous.end,
                            movement.start,
                            previous.destination,
                            movement.origin,
                            previous.from_trip,
                            movement.to_trip,
                            "derived_same_point_gap",
                        )
                    )
            with_gaps.append(movement)
        result[key] = with_gaps
    for (_sid, tid), owners in ownership.items():
        if len(owners) > 1:
            for key in owners:
                qa.add(
                    "ERROR",
                    "TRIP_MULTIPLE_BLOCKS",
                    "Trip assigned to multiple blocks in the same service.",
                    key,
                    tid,
                )
    return result


@dataclass
class GTFS:
    trips: dict[str, Row]
    stops: dict[str, Row]
    routes: dict[str, Row]
    calls: dict[str, list[Row]]
    service_dates: dict[str, set[date]]
    block_trips: dict[tuple[str, str], set[str]]
    agencies: dict[str, Row]


@dataclass
class Matches:
    trips: dict[TripKey, str] = field(default_factory=dict)
    blocks: dict[BlockKey, tuple[str, str]] = field(default_factory=dict)
    trip_rows: list[dict] = field(default_factory=list)
    point_rows: list[dict] = field(default_factory=list)
    service_rows: list[dict] = field(default_factory=list)


# =============================================================================
# STAGE 3: Match dates, stops, trips, and COMPLETE vehicle blocks to GTFS.
# =============================================================================
def read_gtfs(source: Source) -> GTFS:
    if source.csv("frequencies.txt", required=False):
        raise ValueError(
            "Frequency-based GTFS requires explicit trip-instance expansion; unsupported."
        )
    trips = unique(source.csv("trips.txt"), "trip_id")
    stops = unique(source.csv("stops.txt"), "stop_id")
    routes = unique(source.csv("routes.txt"), "route_id")
    agency_rows = source.csv("agency.txt")
    for r in agency_rows:
        r.setdefault("agency_id", "")
    agencies = unique(agency_rows, "agency_id")
    calls: dict[str, list[Row]] = defaultdict(list)
    for row in source.csv("stop_times.txt"):
        if row["trip_id"] not in trips or row["stop_id"] not in stops:
            raise ValueError("GTFS stop_times contains an unknown trip or stop reference.")
        calls[row["trip_id"]].append(row)
    for tid, rows in calls.items():
        rows.sort(key=lambda r: number(r, "stop_sequence"))
        if len({number(r, "stop_sequence") for r in rows}) != len(rows):
            raise ValueError(f"Duplicate GTFS stop_sequence on trip {tid}.")
        previous = -1
        for row in rows:
            if not row.get("arrival_time") or not row.get("departure_time"):
                if row in (rows[0], rows[-1]) or row.get("timepoint", "1") != "0":
                    raise ValueError(f"GTFS trip {tid} has an untimed required stop.")
                continue
            arrival, departure = (
                seconds(row["arrival_time"]),
                seconds(row["departure_time"]),
            )
            if arrival < previous or departure < arrival:
                raise ValueError(f"GTFS trip {tid} has nonchronological stop times.")
            previous = departure
    active: dict[str, set[date]] = defaultdict(set)
    for row in source.csv("calendar.txt", required=False):
        if row["service_id"] in active:
            raise ValueError("Duplicate GTFS calendar service ID.")
        active[row["service_id"]] = {
            d
            for d in dates_between(parse_date(row["start_date"]), parse_date(row["end_date"]))
            if row[WEEKDAYS[d.weekday()]] == "1"
        }
    seen_dates: set[tuple[str, str]] = set()
    for row in source.csv("calendar_dates.txt", required=False):
        key = row["service_id"], row["date"]
        if key in seen_dates:
            raise ValueError("Duplicate GTFS calendar_dates key.")
        seen_dates.add(key)
        day = parse_date(row["date"])
        if row["exception_type"] == "1":
            active[key[0]].add(day)
        elif row["exception_type"] == "2":
            active[key[0]].discard(day)
        else:
            raise ValueError("Invalid calendar exception_type.")
    block_trips: dict[tuple[str, str], set[str]] = defaultdict(set)
    for tid, row in trips.items():
        if row["route_id"] not in routes or row["service_id"] not in active or tid not in calls:
            raise ValueError(f"GTFS trip {tid} has a missing route, calendar, or stop times.")
        block_trips[row["service_id"], row.get("block_id", "")].add(tid)
    return GTFS(trips, stops, routes, dict(calls), dict(active), dict(block_trips), agencies)


def match_gtfs(data: TSDE, gtfs: GTFS, cfg: Config, qa: QA) -> Matches:
    matches = Matches()
    code_index: dict[str, list[str]] = defaultdict(list)
    for stop_id, row in gtfs.stops.items():
        if row.get(cfg.stop_key):
            code_index[row[cfg.stop_key]].append(stop_id)
    point_map: dict[PointKey, str] = {}
    for key, row in data.points.items():
        if key[0] != "st":
            continue
        candidates = code_index.get(row["stop_code"], [])
        close: list[tuple[str, float]] = []
        try:
            coord = (
                float(row["lat"]) / cfg.coordinate_scale,
                float(row["lon"]) / cfg.coordinate_scale,
            )
            for candidate in candidates:
                other = gtfs.stops[candidate]
                distance = distance_m(coord, (float(other["stop_lat"]), float(other["stop_lon"])))
                if distance <= cfg.max_stop_distance:
                    close.append((candidate, distance))
        except ValueError:
            close = []
        status = "matched" if len(close) == 1 else "ambiguous" if len(close) > 1 else "unmatched"
        if len(close) == 1:
            point_map[key] = close[0][0]
        matches.point_rows.append(
            {
                "tsde_point_type": key[0],
                "tsde_point_id": key[1],
                "stop_code": row["stop_code"],
                "status": status,
                "gtfs_stop_id": close[0][0] if len(close) == 1 else "",
                "distance_m": round(close[0][1], 2) if len(close) == 1 else "",
            }
        )
    service_map: dict[str, set[str]] = {}
    for sid, dates in data.service_dates.items():
        same = {g for g, other in gtfs.service_dates.items() if dates and dates == other}
        service_map[sid] = same
        matches.service_rows.append(
            {
                "tsde_service_id": sid,
                "tsde_service_name": data.services[sid]["sname"],
                "active_dates": len(dates),
                "gtfs_service_ids": "|".join(sorted(same)),
                "status": "matched" if same else "unmatched",
            }
        )
        if not same:
            qa.add(
                "WARNING",
                "SERVICE_UNMATCHED",
                f"TSDE service {sid}: no identical GTFS date set.",
            )
    index: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for tid, row in gtfs.trips.items():
        route = gtfs.routes[row["route_id"]]
        signature = (
            route_name(route.get("route_short_name") or route["route_id"]),
            tuple(r["stop_id"] for r in gtfs.calls[tid]),
        )
        index[signature].append(tid)
    for key, trip in data.trips.items():
        calls = trip.stops
        row: dict = {
            "tsde_service_id": key[0],
            "tsde_trip_id": key[1],
            "route": trip.route,
            "tsde_direction": trip.direction,
            "status": "unmatched",
            "gtfs_trip_id": "",
            "gtfs_service_id": "",
            "gtfs_block_id": "",
            "gtfs_direction": "",
            "max_interpolated_difference_seconds": "",
            "reason": "",
        }
        if trip.errors:
            row["reason"] = "; ".join(sorted(set(trip.errors)))
        elif not calls or any(c.point not in point_map for c in calls):
            row["reason"] = "Missing/ambiguous GTFS stop mapping."
        else:
            signature = trip.route, tuple(point_map[c.point] for c in calls)
            candidates = []
            for tid in index.get(signature, []):
                if gtfs.trips[tid]["service_id"] not in service_map.get(trip.sid, set()):
                    continue
                rows = gtfs.calls[tid]
                timed = [0, len(rows) - 1] + [
                    i for i, r in enumerate(rows) if r.get("timepoint", "1") != "0"
                ]
                if all(
                    abs(seconds(rows[i]["departure_time"]) - calls[i].time) <= cfg.time_tolerance
                    for i in timed
                ):
                    candidates.append(tid)
            if len(candidates) == 1:
                tid = candidates[0]
                matches.trips[key] = tid
                gr = gtfs.trips[tid]
                differences = [
                    abs(seconds(r["departure_time"]) - c.time)
                    for r, c in zip(gtfs.calls[tid], calls)
                    if r.get("timepoint", "1") == "0" and r.get("departure_time")
                ]
                row.update(
                    status="matched",
                    gtfs_trip_id=tid,
                    gtfs_service_id=gr["service_id"],
                    gtfs_block_id=gr.get("block_id", ""),
                    gtfs_direction=gr.get("direction_id", ""),
                    max_interpolated_difference_seconds=max(differences, default=0),
                )
            else:
                row["reason"] = (
                    "Multiple matching trips."
                    if candidates
                    else "No exact date/route/stop-order/timepoint match."
                )
                row["status"] = "ambiguous" if candidates else "unmatched"
        matches.trip_rows.append(row)
    # Reject many-to-one trip matches even when each individual query was unique.
    inverse: dict[str, list[TripKey]] = defaultdict(list)
    for key, tid in matches.trips.items():
        inverse[tid].append(key)
    for tid, keys in inverse.items():
        if len(keys) > 1:
            for key in keys:
                del matches.trips[key]
            for row in matches.trip_rows:
                if row["gtfs_trip_id"] == tid:
                    row.update(
                        status="ambiguous",
                        reason="Multiple TSDE trips map to this GTFS trip.",
                        gtfs_trip_id="",
                    )
    bad = qa.bad_blocks()
    for key, rows in data.blocks.items():
        source_trips = [(key[0], r["tid"]) for r in rows if r["etype"] == "3"]
        if not source_trips or len(set(source_trips)) != len(source_trips):
            qa.add(
                "ERROR",
                "REPEATED_OR_EMPTY_BLOCK_TRIPS",
                "No trips or repeated trip starts.",
                key,
            )
            continue
        unmapped = [k[1] for k in source_trips if k not in matches.trips]
        if unmapped:
            qa.add(
                "ERROR",
                "BLOCK_UNMATCHED_TRIPS",
                f"{len(unmapped)} unmatched trip(s): {','.join(unmapped)}",
                key,
            )
            continue
        mapped = {matches.trips[k] for k in source_trips}
        gkeys = {(gtfs.trips[t]["service_id"], gtfs.trips[t].get("block_id", "")) for t in mapped}
        if len(gkeys) != 1 or not next(iter(gkeys))[1]:
            qa.add(
                "ERROR",
                "BLOCK_ID_CONFLICT",
                "Matched trips span multiple GTFS blocks/services or have blank block IDs.",
                key,
            )
            continue
        gkey = next(iter(gkeys))
        if gtfs.block_trips[gkey] != mapped:
            qa.add(
                "ERROR",
                "INCOMPLETE_GTFS_BLOCK",
                f"GTFS block has {len(gtfs.block_trips[gkey] - mapped)} additional trips.",
                key,
            )
            continue
        if key not in bad:
            matches.blocks[key] = gkey
    inverse_blocks: dict[tuple[str, str], list[BlockKey]] = defaultdict(list)
    for key, gkey in matches.blocks.items():
        inverse_blocks[gkey].append(key)
    for gkey, keys in inverse_blocks.items():
        if len(keys) > 1:
            for key in keys:
                del matches.blocks[key]
                qa.add(
                    "ERROR",
                    "MULTIPLE_TSDE_BLOCKS",
                    f"Several TSDE blocks map to {gkey}.",
                    key,
                )
    return matches


# =============================================================================
# STAGE 4: Export only complete eligible blocks as TODS 2.1 supplements.
# =============================================================================
def export_tods(
    data: TSDE,
    gtfs: GTFS,
    matches: Matches,
    movements: dict[BlockKey, list[Movement]],
    out: Path,
    cfg: Config,
    qa: QA,
) -> dict:
    prefix = f"TODS_TSDE_{data.version['verid']}_"
    tables: dict[str, list[Row]] = {
        name: []
        for name in (
            "routes_supplement.txt",
            "stops_supplement.txt",
            "trips_supplement.txt",
            "stop_times_supplement.txt",
        )
    }
    fields = {
        "routes_supplement.txt": [
            "route_id",
            "agency_id",
            "route_long_name",
            "route_type",
        ],
        "stops_supplement.txt": [
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "location_type",
            "TODS_location_type",
        ],
        "trips_supplement.txt": [
            "route_id",
            "service_id",
            "trip_id",
            "block_id",
            "TODS_trip_type",
        ],
        "stop_times_supplement.txt": [
            "trip_id",
            "arrival_time",
            "departure_time",
            "stop_id",
            "stop_sequence",
            "pickup_type",
            "drop_off_type",
            "timepoint",
        ],
    }
    new_points: dict[str, Row] = {}
    new_routes: dict[str, Row] = {}
    exported: list[BlockKey] = []
    exported_movements: list[dict] = []

    def internal_point(key: PointKey) -> str:
        stop_id = f"{prefix}{key[0]}_{key[1]}"
        if stop_id in gtfs.stops:
            raise ValueError(f"Generated stop ID collides with GTFS: {stop_id}.")
        row = data.points[key]
        lat, lon = (
            float(row["lat"]) / cfg.coordinate_scale,
            float(row["lon"]) / cfg.coordinate_scale,
        )
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            raise ValueError(f"Invalid coordinates for internal point {key}.")
        new_points[stop_id] = {
            "stop_id": stop_id,
            "stop_name": row["lname"] or row["sname"],
            "stop_lat": f"{lat:.7f}",
            "stop_lon": f"{lon:.7f}",
            "location_type": "0",
            "TODS_location_type": "garage" if key[0] == "ga" else "operational_point",
        }
        return stop_id

    for key, (service_id, block_id) in sorted(matches.blocks.items()):
        pending_trips: list[Row] = []
        pending_times: list[Row] = []
        pending_audit: list[dict] = []
        try:
            for movement in movements[key]:
                if movement.kind not in {"pull-out", "pull-back", "deadhead"}:
                    continue
                trip_id = f"{prefix}{key[0]}_{key[1]}_{movement.sequence}"
                if trip_id in gtfs.trips:
                    raise ValueError("Generated trip ID collides with GTFS.")
                anchor = movement.to_trip or movement.from_trip
                anchor_id = matches.trips[key[0], anchor]
                route = gtfs.routes[gtfs.trips[anchor_id]["route_id"]]
                agency_id = route.get("agency_id", "")
                if not agency_id and len(gtfs.agencies) == 1:
                    agency_id = next(iter(gtfs.agencies))
                if agency_id not in gtfs.agencies:
                    raise ValueError("Unresolved agency for nonrevenue route.")
                route_type = route["route_type"]
                route_id = f"{prefix}deadhead_{agency_id}_{route_type}"
                if route_id in gtfs.routes:
                    raise ValueError("Generated route ID collides with GTFS.")
                new_routes[route_id] = {
                    "route_id": route_id,
                    "agency_id": agency_id,
                    "route_long_name": "Scheduled nonrevenue movements",
                    "route_type": route_type,
                }
                # A node may represent several bays: resolve its endpoint via THIS trip.
                if movement.from_trip:
                    gr = gtfs.calls[matches.trips[key[0], movement.from_trip]][-1]
                    origin = gr["stop_id"]
                    if movement.start < seconds(gr["departure_time"]):
                        raise ValueError("Deadhead begins before its GTFS anchor trip ends.")
                else:
                    origin = internal_point(movement.origin)
                if movement.to_trip:
                    gr = gtfs.calls[matches.trips[key[0], movement.to_trip]][0]
                    destination = gr["stop_id"]
                    if movement.end > seconds(gr["departure_time"]):
                        raise ValueError("Deadhead ends after its GTFS anchor trip departs.")
                else:
                    destination = internal_point(movement.destination)
                pending_trips.append(
                    {
                        "route_id": route_id,
                        "service_id": service_id,
                        "trip_id": trip_id,
                        "block_id": block_id,
                        "TODS_trip_type": movement.kind,
                    }
                )
                for sequence, stop_id, time in (
                    (1, origin, movement.start),
                    (2, destination, movement.end),
                ):
                    pending_times.append(
                        {
                            "trip_id": trip_id,
                            "arrival_time": clock(time),
                            "departure_time": clock(time),
                            "stop_id": stop_id,
                            "stop_sequence": str(sequence),
                            "pickup_type": "1",
                            "drop_off_type": "1",
                            "timepoint": "1",
                        }
                    )
                pending_audit.append(
                    {
                        "tsde_service_id": key[0],
                        "tsde_block_id": key[1],
                        "tsde_sequence": movement.sequence,
                        "tods_trip_id": trip_id,
                        "gtfs_service_id": service_id,
                        "gtfs_block_id": block_id,
                        "movement_type": movement.kind,
                    }
                )
        except (ValueError, KeyError) as exc:
            qa.add("ERROR", "EXPORT_BLOCK_REJECTED", str(exc), key)
            continue
        tables["trips_supplement.txt"].extend(pending_trips)
        tables["stop_times_supplement.txt"].extend(pending_times)
        exported_movements.extend(pending_audit)
        exported.append(key)
    used_routes = {r["route_id"] for r in tables["trips_supplement.txt"]}
    used_stops = {r["stop_id"] for r in tables["stop_times_supplement.txt"]}
    tables["routes_supplement.txt"] = [new_routes[k] for k in sorted(used_routes)]
    tables["stops_supplement.txt"] = [new_points[k] for k in sorted(used_stops) if k in new_points]
    if cfg.require_all_blocks and len(exported) != len(data.blocks):
        qa.add(
            "WARNING",
            "EXPORT_SUPPRESSED",
            "require_all_blocks=True and one or more source blocks were rejected.",
        )
        return {
            "exported_blocks": 0,
            "exported_deadheads": 0,
            "exported_block_keys": [],
            "tods_zip": None,
        }
    if not tables["trips_supplement.txt"]:
        return {
            "exported_blocks": 0,
            "exported_deadheads": 0,
            "exported_block_keys": [],
            "tods_zip": None,
        }
    # Validate generated primary and foreign keys before writing the ZIP.
    trip_ids = [r["trip_id"] for r in tables["trips_supplement.txt"]]
    if len(trip_ids) != len(set(trip_ids)):
        raise ValueError("Duplicate generated trip IDs.")
    for row in tables["trips_supplement.txt"]:
        if row["service_id"] not in gtfs.service_dates or row["route_id"] not in used_routes:
            raise ValueError("Generated trip has a dangling service/route reference.")
    for row in tables["stop_times_supplement.txt"]:
        if row["stop_id"] not in gtfs.stops and row["stop_id"] not in new_points:
            raise ValueError("Generated stop_time has a dangling stop reference.")
    folder = out / "tods"
    folder.mkdir()
    for name, rows in tables.items():
        write_csv(folder / name, fields[name], rows)
    zip_path = out / "draft_tods.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in tables:
            archive.write(folder / name, name)
    write_csv(
        out / "tods_movement_crosswalk.csv",
        [
            "tsde_service_id",
            "tsde_block_id",
            "tsde_sequence",
            "tods_trip_id",
            "gtfs_service_id",
            "gtfs_block_id",
            "movement_type",
        ],
        exported_movements,
    )
    return {
        "exported_blocks": len(exported),
        "exported_deadheads": len(trip_ids),
        "exported_block_keys": [list(k) for k in exported],
        "tods_zip": zip_path.name,
    }


def write_audit(
    data: TSDE,
    movements: dict[BlockKey, list[Movement]],
    matches: Matches,
    qa: QA,
    out: Path,
    summary: dict,
) -> None:
    points = []
    for key, row in sorted(data.points.items()):
        points.append(
            {
                "point_type": key[0],
                "point_id": key[1],
                "name": row["lname"],
                "abbreviation": row["sname"],
                "stop_code": row["stop_code"],
                "source_latitude": row["lat"],
                "source_longitude": row["lon"],
            }
        )
    write_csv(
        out / "points.csv",
        [
            "point_type",
            "point_id",
            "name",
            "abbreviation",
            "stop_code",
            "source_latitude",
            "source_longitude",
        ],
        points,
    )
    raw_events = []
    movement_rows = []
    block_rows = []
    bad = qa.bad_blocks()
    exported = {tuple(k) for k in summary.get("exported_block_keys", [])}
    block_issues: dict[BlockKey, set[str]] = defaultdict(set)
    for issue in qa.issues:
        if issue.block_id:
            block_issues[issue.service_id, issue.block_id].add(issue.code)
    for key, rows in data.blocks.items():
        for row in rows:
            raw_events.append(
                {
                    **row,
                    "event_label_inferred": EVENT_CODES.get(row["etype"], "UNKNOWN"),
                    "event_time": clock(number(row, "etime")) if number(row, "etime") >= 0 else "",
                    "point_name": data.points.get(point_key(row), {}).get("lname", ""),
                }
            )
        totals: Counter[str] = Counter()
        for n, movement in enumerate(movements[key], start=1):
            totals[movement.kind] += max(0, movement.end - movement.start)
            movement_rows.append(
                {
                    "tsde_service_id": key[0],
                    "service_name": data.services[key[0]]["sname"],
                    "tsde_block_id": key[1],
                    "block_number": rows[0]["bnum"],
                    "timeline_sequence": n,
                    "source_sequence": movement.sequence,
                    "movement_type": movement.kind,
                    "start_time": clock(movement.start) if movement.start >= 0 else "",
                    "end_time": clock(movement.end) if movement.end >= 0 else "",
                    "duration_seconds": movement.end - movement.start,
                    "from_point_type": movement.origin[0],
                    "from_point_id": movement.origin[1],
                    "from_name": data.points.get(movement.origin, {}).get("lname", ""),
                    "to_point_type": movement.destination[0],
                    "to_point_id": movement.destination[1],
                    "to_name": data.points.get(movement.destination, {}).get("lname", ""),
                    "from_tsde_trip_id": movement.from_trip,
                    "to_tsde_trip_id": movement.to_trip,
                    "source_codes": movement.source_codes,
                    "block_has_errors": key in bad,
                    "block_exported": key in exported,
                }
            )
        trip_keys = [(key[0], r["tid"]) for r in rows if r["etype"] == "3"]
        gsid, gbid = matches.blocks.get(key, ("", ""))
        block_rows.append(
            {
                "tsde_service_id": key[0],
                "service_name": data.services[key[0]]["sname"],
                "tsde_block_id": key[1],
                "block_number": rows[0]["bnum"],
                "division_id": rows[0]["did"],
                "block_label_route": rows[0]["lgabbr"],
                "routes": "|".join(
                    sorted({data.trips[k].route for k in trip_keys if k in data.trips})
                ),
                "scheduled_trip_count": len(trip_keys),
                "matched_trip_count": sum(k in matches.trips for k in trip_keys),
                "scheduled_trip_minutes": round(totals["scheduled_trip"] / 60, 2),
                "pullout_minutes": round(totals["pull-out"] / 60, 2),
                "pullin_minutes": round(totals["pull-back"] / 60, 2),
                "intertrip_deadhead_minutes": round(totals["deadhead"] / 60, 2),
                "layover_minutes": round(totals["layover"] / 60, 2),
                "gtfs_service_id": gsid,
                "gtfs_block_id": gbid,
                "has_errors": key in bad,
                "exported": key in exported,
                "issue_codes": "|".join(sorted(block_issues[key])),
            }
        )
    raw_fields = (
        list(data.tables["blk"][0]) if data.tables["blk"] else ["sid", "bid", "snum", "etype"]
    )
    write_csv(
        out / "block_events_raw.csv",
        raw_fields + ["event_label_inferred", "event_time", "point_name"],
        raw_events,
    )
    movement_fields = [
        "tsde_service_id",
        "service_name",
        "tsde_block_id",
        "block_number",
        "timeline_sequence",
        "source_sequence",
        "movement_type",
        "start_time",
        "end_time",
        "duration_seconds",
        "from_point_type",
        "from_point_id",
        "from_name",
        "to_point_type",
        "to_point_id",
        "to_name",
        "from_tsde_trip_id",
        "to_tsde_trip_id",
        "source_codes",
        "block_has_errors",
        "block_exported",
    ]
    write_csv(out / "block_movements.csv", movement_fields, movement_rows)
    block_fields = [
        "tsde_service_id",
        "service_name",
        "tsde_block_id",
        "block_number",
        "division_id",
        "block_label_route",
        "routes",
        "scheduled_trip_count",
        "matched_trip_count",
        "scheduled_trip_minutes",
        "pullout_minutes",
        "pullin_minutes",
        "intertrip_deadhead_minutes",
        "layover_minutes",
        "gtfs_service_id",
        "gtfs_block_id",
        "has_errors",
        "exported",
        "issue_codes",
    ]
    write_csv(out / "block_summary.csv", block_fields, block_rows)
    lookup = []
    for row in data.tables["deh"]:
        lookup.append(
            {
                **row,
                "duration_defined": number(row, "dhtime") >= 0,
                "distance_positive": number(row, "dist") > 0,
                "from_name": data.points.get(point_key(row, "f"), {}).get("lname", ""),
                "to_name": data.points.get(point_key(row, "t"), {}).get("lname", ""),
            }
        )
    lookup_fields = (
        list(data.tables["deh"][0])
        if data.tables["deh"]
        else ["sid", "fpid", "tpid", "dhtime", "dist"]
    )
    write_csv(
        out / "deadhead_lookup.csv",
        lookup_fields + ["duration_defined", "distance_positive", "from_name", "to_name"],
        lookup,
    )
    for tag, name in (
        ("run", "run_pieces_raw.csv"),
        ("div", "divisions.csv"),
        ("dgr", "division_garages.csv"),
    ):
        rows = data.tables[tag]
        write_csv(out / name, list(rows[0]) if rows else ["no_records"], rows)
    pay_types = {r["ptid"]: r["ptname"] for r in data.tables["pty"]}
    rows = [{**r, "pay_type_name": pay_types.get(r["ptid"], "UNKNOWN")} for r in data.tables["rua"]]
    write_csv(out / "run_pay_components.csv", list(rows[0]) if rows else ["no_records"], rows)
    write_csv(
        out / "trip_crosswalk.csv",
        [
            "tsde_service_id",
            "tsde_trip_id",
            "route",
            "tsde_direction",
            "status",
            "gtfs_trip_id",
            "gtfs_service_id",
            "gtfs_block_id",
            "gtfs_direction",
            "max_interpolated_difference_seconds",
            "reason",
        ],
        matches.trip_rows,
    )
    write_csv(
        out / "stop_crosswalk.csv",
        [
            "tsde_point_type",
            "tsde_point_id",
            "stop_code",
            "status",
            "gtfs_stop_id",
            "distance_m",
        ],
        matches.point_rows,
    )
    write_csv(
        out / "service_crosswalk.csv",
        [
            "tsde_service_id",
            "tsde_service_name",
            "active_dates",
            "gtfs_service_ids",
            "status",
        ],
        matches.service_rows,
    )
    write_csv(
        out / "qa_issues.csv",
        ["severity", "code", "service_id", "block_id", "trip_id", "detail"],
        (asdict(i) for i in qa.issues),
    )
    counts = Counter(i.code for i in qa.issues)
    lines = [
        "# TSDE to TODS run report",
        "",
        f"Signup: **{data.version['name']}**. Converter: {VERSION}.",
        "",
        f"- Source trips: {len(data.trips):,}; service/block combinations: {len(data.blocks):,}.",
        f"- Trips matched to GTFS: {len(matches.trips):,}.",
        f"- Blocks exported: {summary.get('exported_blocks', 0):,}.",
        f"- Nonrevenue trips exported: {summary.get('exported_deadheads', 0):,}.",
        "",
        "This is a **partial draft supplement**, not a complete operational feed. Apply only to the exact GTFS fingerprint in manifest.json.",
        "No source GTFS rows are overwritten. Missing blocks are documented in block_summary.csv; do not treat their absent deadheads as zero.",
        "",
        "## Interpretation",
        "",
        "- Event meanings are inferred unless explicitly marked verified in the configuration.",
        "- Scheduled trip durations include internal service; they are not automatically revenue hours.",
        "- Same-point gaps are labeled layover; paid-break status and actual operator duties are not inferred.",
        "- Deadheads come from paired block events. Lookup entries are not instantiated as trips.",
        "- Negative lookup durations and zero distances are preserved, not converted to zero travel time or mileage.",
        "- Stop matching uses the configured code and a coordinate check. Trip matching checks route, all stops in order, dates, terminal times, and published timepoints.",
        "- Direction numbers are preserved in the crosswalk but not assumed to have the same meaning across systems.",
        "- Run pieces and pay attributes are included for review; run_events.txt and employee/fleet assignments are not generated in this version.",
        "- Failed source geometry exports are recorded. No nonrevenue geometry or distance is fabricated.",
        "",
        "## QA counts",
        "",
        "| Code | Records |",
        "|---|---:|",
    ]
    lines.extend(f"| {code} | {count:,} |" for code, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## Suggested manual checks",
            "",
            "Check a weekday block from each garage, an interlined block, and an overnight block against paddles.",
            "Confirm the 1/15, 16/2, and 17/18 event pairs against Trapeze documentation before relying on the output for operations or savings estimates.",
            "",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def fingerprint(source: Source) -> dict[str, str]:
    """Hash exact inputs so supplements cannot be confused across revisions."""
    files: dict[str, str] = {}
    for name in sorted(source.names):
        digest = hashlib.sha256()
        with source.open(name) as stream:
            for chunk in iter(lambda: stream.read(1_048_576), b""):
                digest.update(chunk)
        files[name] = digest.hexdigest()
    return files


def run(cfg: Config) -> Path:
    if cfg.stop_key not in {"stop_id", "stop_code"} or cfg.coordinate_scale <= 0:
        raise ValueError("Invalid stop key or coordinate scale.")
    if cfg.max_stop_distance < 0 or cfg.time_tolerance < 0:
        raise ValueError("Matching tolerances must be nonnegative.")
    cfg.output.mkdir(parents=True, exist_ok=True)
    out = cfg.output / datetime.now().strftime("tsde_to_tods_%Y%m%d_%H%M%S_%f")
    out.mkdir()
    handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    sources: list[Source] = []
    try:
        qa = QA()
        source = Source(cfg.tsde)
        sources.append(source)
        data = read_tsde(source, cfg, qa)
        if data.version.get("prod") == "N":
            qa.add(
                "WARNING",
                "SOURCE_PROD_FLAG_N",
                "Schedule version has prod=N; retained as source metadata, not assumed to invalidate the export.",
            )
        logging.info("Reconstructing vehicle movements")
        movements = reconstruct(data, qa)
        reconstruction_bad = len(qa.bad_blocks())
        matches = Matches()
        summary: dict = {
            "exported_blocks": 0,
            "exported_deadheads": 0,
            "exported_block_keys": [],
            "tods_zip": None,
        }
        gtfs_hashes: dict[str, str] = {}
        gtfs_trip_count = 0
        if cfg.gtfs is not None:
            logging.info("Reading and matching GTFS: %s", cfg.gtfs)
            gtfs_source = Source(cfg.gtfs)
            sources.append(gtfs_source)
            gtfs = read_gtfs(gtfs_source)
            gtfs_trip_count = len(gtfs.trips)
            matches = match_gtfs(data, gtfs, cfg, qa)
            logging.info(
                "Matched %s trips; %s complete blocks eligible before export",
                len(matches.trips),
                len(matches.blocks),
            )
            summary = export_tods(data, gtfs, matches, movements, out, cfg, qa)
            gtfs_hashes = fingerprint(gtfs_source)
        else:
            qa.add(
                "WARNING",
                "NO_GTFS",
                "Audit only: matching GTFS was not supplied, so no TODS ZIP was generated.",
            )
        summary.update(
            tsde_trips=len(data.trips),
            tsde_blocks=len(data.blocks),
            matched_trips=len(matches.trips),
            gtfs_trips=gtfs_trip_count,
            reconstruction_rejected_blocks=reconstruction_bad,
            excluded_blocks=len(data.blocks) - summary["exported_blocks"],
            qa_counts=dict(Counter(i.code for i in qa.issues)),
        )
        write_audit(data, movements, matches, qa, out, summary)
        manifest = {
            "converter_version": VERSION,
            "tods_version": "2.1",
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "partial_draft" if summary["tods_zip"] else "audit_only",
            "configuration": asdict(cfg),
            "source_version": data.version,
            "source_export_metadata": data.export_version,
            "event_codes": EVENT_CODES,
            "event_codes_verified": cfg.event_codes_verified,
            "tsde_sha256": fingerprint(source),
            "gtfs_sha256": gtfs_hashes,
            "summary": summary,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        logging.info(
            "Complete: %s blocks, %s nonrevenue trips exported. Output: %s",
            summary["exported_blocks"],
            summary["exported_deadheads"],
            out,
        )
        return out
    except Exception:
        logging.exception("Conversion failed; partial files are not a usable feed")
        # A completed ZIP must never survive a later audit/manifest failure.
        (out / "draft_tods.zip").unlink(missing_ok=True)
        raise
    finally:
        for item in sources:
            item.close()
        logging.getLogger().removeHandler(handler)
        handler.close()


def parse_args(argv: Optional[list[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tsde", type=Path, default=TSDE_PATH, help="TSDE XML ZIP or folder")
    parser.add_argument(
        "--gtfs",
        type=Path,
        default=GTFS_PATH,
        help="Matching GTFS ZIP or folder; omit for audit only",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Parent output folder")
    parser.add_argument(
        "--prefix",
        default=EXPORT_PREFIX,
        help="Date prefix if the TSDE folder has several exports",
    )
    parser.add_argument("--stop-key", choices=("stop_code", "stop_id"), default=GTFS_STOP_KEY)
    parser.add_argument("--stop-attribute", default=STOP_CODE_ATTRIBUTE)
    parser.add_argument("--max-stop-distance", type=float, default=MAX_STOP_DISTANCE_METERS)
    parser.add_argument("--time-tolerance", type=int, default=TIME_TOLERANCE_SECONDS)
    parser.add_argument("--event-codes-verified", action="store_true", default=EVENT_CODES_VERIFIED)
    parser.add_argument("--require-all-blocks", action="store_true", default=REQUIRE_ALL_BLOCKS)
    # Notebook execution uses EDIT HERE values and ignores the kernel's -f argument.
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return (
            path.expanduser().resolve()
            if path.is_absolute()
            else (SCRIPT_DIR / path.expanduser()).resolve()
        )

    return Config(
        resolve(args.tsde),
        resolve(args.gtfs) if args.gtfs is not None else None,
        resolve(args.output),
        args.prefix,
        args.stop_attribute,
        args.stop_key,
        args.max_stop_distance,
        COORDINATE_SCALE,
        args.time_tolerance,
        args.event_codes_verified,
        args.require_all_blocks,
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        run(parse_args(argv))
    except (OSError, ValueError, KeyError, ET.ParseError, zipfile.BadZipFile) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
