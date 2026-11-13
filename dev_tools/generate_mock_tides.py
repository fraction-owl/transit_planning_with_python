"""Generate coherent sample fixtures for ``stop_visits.csv`` and ``trips_performed.csv``.

This script emits a pair of TIDES-shaped CSV fixtures suitable for testing
analyses that group by month, by route/direction, or by month x route x
direction. The two files share keys (``service_date``, ``trip_id_performed``,
``pattern_id``, ``vehicle_id``) so they can be joined the way real exports
allow -- generating them separately risks orphan stop visits.

The actuals are drawn from the mock GTFS feed and its TODS supplements
(``tests/fixtures/mock_gtfs_dc.zip`` and ``mock_tods_dc.zip``, both written by
``generate_mock_gtfs.py``), so every TIDES identifier resolves against them:
``trip_id_scheduled``, ``route_id``, ``shape_id``, ``block_id`` and ``stop_id``
come from GTFS, service dates are days the GTFS calendar runs, and each
sampled block's pull-out, deadheads and pull-in appear as ``Pullout`` /
``Deadhead`` / ``Pullin`` rows whose ``trip_id_scheduled`` is the TODS trip.
The sampling unit is a whole vehicle block on one date, run by one vehicle.

A second pass (``HOLIDAY_PASS``) appends every block of the holiday-only service
on each date it runs inside the month range -- with the base feed, the 50H loop
on 2026-01-01, a Thursday -- so day-type and holiday detection have a real
holiday to find. It draws from its own random stream, so the main sample's rows
do not change when the pass is edited.

Sizing is configurable at the top of the module; defaults keep the files
small enough to commit while leaving enough rows for meaningful aggregation.
A fixed random seed makes the output stable run-to-run.

Inputs:
    - GTFS_PATH: the mock GTFS feed (folder or ``.zip``).
    - TODS_PATH (optional): its TODS supplements (folder or ``.zip``).

Outputs:
    <OUTPUT_DIR>/trips_performed.csv  (service_date as YYYY-MM-DD)
    <OUTPUT_DIR>/stop_visits.csv      (service_date as M/D/YYYY)
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import random
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Mock GTFS feed and TODS supplements to draw from (folder or .zip). Set
# TODS_PATH to None to emit revenue trips only.
GTFS_PATH: Final[Path] = REPO_ROOT / "tests" / "fixtures" / "mock_gtfs_dc.zip"
TODS_PATH: Final[Optional[Path]] = REPO_ROOT / "tests" / "fixtures" / "mock_tods_dc.zip"

OUTPUT_DIR: Final[Path] = Path("./fixtures_generated")

# GTFS route_ids to sample: a local, an express and a crosstown local. Route 40
# (close stop spacing, ~176 stops a trip) and the holiday loop 50H are left out
# by default to keep stop_visits.csv small.
ROUTES: Final[list[str]] = ["DC_R10", "DC_R20", "DC_R30"]

# Inclusive month range, inside the GTFS calendar (2026 for the base feed).
START_MONTH: Final[tuple[int, int]] = (2026, 1)  # (year, month)
END_MONTH: Final[tuple[int, int]] = (2026, 3)

# Service dates sampled per month for each GTFS service_id, evenly spread over
# the dates that service runs (holidays drop out where calendar_dates says so).
DAYS_PER_MONTH: Final[dict[str, int]] = {"weekday": 2, "saturday": 1, "sunday": 1}

# Holiday pass, run after the main sample: {service_id: route_ids}. On every date
# the service runs inside the month range, each listed route runs all of its
# blocks for that service. The base feed's holiday service is the 50H loop on
# Jan 1, Nov 11, Nov 26 and Dec 25, so the default Jan-Mar range adds one date.
# Rows come from their own random stream (RANDOM_SEED + 1) and are appended, so
# the main sample above is unchanged. Set to {} to skip the pass.
HOLIDAY_PASS: Final[dict[str, list[str]]] = {"holiday": ["DC_R50H"]}

# Each (service_id, route) gets a fixed rotation of blocks, chosen once; the
# sampled dates of a month take them in turn, one block per route per date.
# Reusing blocks means each trip is observed on several dates (per-trip runtime
# and OTP analyses need repeats). The rotation holds the fewest blocks that run
# every direction (express blocks run one direction), led by the owl block
# below; DAYS_PER_MONTH must be at least the rotation length for every month to
# cover every route-direction. Every trip in a sampled block is emitted, so stop
# visits scale with block length.
#
# For routes that run past OWL_START_MIN, the rotation starts with the block
# that has the latest-starting trip. That puts
# late-night trips in the fixture and -- more importantly -- exercises the
# post-midnight footgun: in TIDES, schedule_* and actual_* fields are full ISO
# datetimes, so a trip that ends after midnight has its end timestamp on the
# *next* calendar day while ``service_date`` stays on the operational service
# day. Code that strips the date, groups by ``timestamp.date()`` instead of
# ``service_date``, or filters by hour-of-day without handling rollover will
# break on these rows.
INCLUDE_OWL_TRIPS: Final[bool] = True
OWL_START_MIN: Final[int] = 23 * 60  # GTFS start time, minutes after midnight

# Edge-case rates. Kept small so the bulk of the data is clean for analysis.
P_SKIPPED_STOP: Final[float] = 0.03  # one stop on a trip gets schedule_relationship=Skipped
P_ADDED_STOP: Final[float] = 0.02  # one stop on a trip gets schedule_relationship=Added
P_CANCELED_TRIP: Final[float] = 0.02  # revenue trip emitted with no actuals

# Vehicles are drawn without replacement from this fleet for each service date.
FLEET: Final[list[str]] = [str(1001 + i) for i in range(60)]

RANDOM_SEED: Final[int] = 42

# Route metadata, keyed by GTFS route_id. route_type and ntd_mode use TIDES enums.
ROUTE_META: Final[dict[str, dict[str, str]]] = {
    "DC_R10": {
        "route_type": "Local Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "LOCAL",
        "operator_id": "OP_10",
    },
    "DC_R20": {
        "route_type": "Express Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "EXPRESS",
        "operator_id": "OP_20",
    },
    "DC_R30": {
        "route_type": "Local Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "LOCAL",
        "operator_id": "OP_30",
    },
    "DC_R40": {
        "route_type": "Local Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "LOCAL",
        "operator_id": "OP_40",
    },
    "DC_R50H": {
        "route_type": "Local Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "LOCAL",
        "operator_id": "OP_50",
    },
}

# TODS_trip_type -> TIDES trip_type for nonrevenue trips.
TODS_TO_TIDES_TRIP_TYPE: Final[dict[str, str]] = {
    "pull-out": "Pullout",
    "pull-back": "Pullin",
    "deadhead": "Deadhead",
}

# Column order for each output file. Matches the uploaded samples exactly.
TRIPS_COLS: Final[list[str]] = [
    "service_date",
    "trip_id_performed",
    "vehicle_id",
    "trip_id_scheduled",
    "route_id",
    "route_type",
    "ntd_mode",
    "route_type_agency",
    "shape_id",
    "pattern_id",
    "direction_id",
    "operator_id",
    "block_id",
    "trip_start_stop_id",
    "trip_end_stop_id",
    "schedule_trip_start",
    "schedule_trip_end",
    "actual_trip_start",
    "actual_trip_end",
    "trip_type",
    "schedule_relationship",
]

STOPS_COLS: Final[list[str]] = [
    "service_date",
    "trip_id_performed",
    "trip_stop_sequence",
    "scheduled_stop_sequence",
    "pattern_id",
    "vehicle_id",
    "dwell",
    "stop_id",
    "timepoint",
    "schedule_arrival_time",
    "schedule_departure_time",
    "actual_arrival_time",
    "actual_departure_time",
    "distance",
    "boarding_1",
    "alighting_1",
    "boarding_2",
    "alighting_2",
    "departure_load",
    "door_open",
    "door_close",
    "door_status",
    "ramp_deployed_time",
    "ramp_failure",
    "kneel_deployed_time",
    "lift_deployed_time",
    "bike_rack_deployed",
    "bike_load",
    "revenue",
    "number_of_transactions",
    "schedule_relationship",
]

METERS_PER_FOOT: Final[float] = 0.3048


# =============================================================================
# HELPERS
# =============================================================================


@dataclass(frozen=True)
class ScheduledTrip:
    """One scheduled trip (GTFS revenue or TODS nonrevenue) in a block."""

    trip_id: str
    route_id: str
    direction_id: str
    shape_id: str
    block_id: str
    start: int  # seconds after the service date's midnight (GTFS time)
    end: int
    calls: tuple[dict[str, str], ...]  # stop_times rows in sequence order
    tods_type: str = ""  # "" for GTFS revenue trips


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


def _iter_months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Yield (year, month) tuples from start to end inclusive."""
    months: list[tuple[int, int]] = []
    y, m = start
    while (y, m) <= end:
        months.append((y, m))
        m += 1
        if m == 13:
            y += 1
            m = 1
    return months


def _pick_service_days(candidates: list[date], n: int, rng: random.Random) -> list[date]:
    """Pick *n* dates from *candidates* (sorted), evenly spread with a little jitter."""
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    picks: set[date] = set()
    for i in range(n):
        # Small jitter so we don't always get the same days of every month.
        idx = min(max(int(i * step) + rng.randint(-1, 1), 0), len(candidates) - 1)
        picks.add(candidates[idx])
    return sorted(picks)


def _gtfs_seconds(value: str) -> int:
    """Parse a GTFS ``H:MM:SS`` time (24+ hours allowed) into seconds."""
    h, m, s = (int(part) for part in value.split(":"))
    return h * 3600 + m * 60 + s


def _at(d: date, seconds: float) -> datetime:
    """Combine a service date with seconds after its midnight into a datetime."""
    return datetime(d.year, d.month, d.day) + timedelta(seconds=seconds)


def _iso(ts: datetime | None) -> str:
    """ISO-8601 string with seconds, or empty string if *ts* is None."""
    return "" if ts is None else ts.strftime("%Y-%m-%dT%H:%M:%S")


def _format_mdy(d: date) -> str:
    """Format a date as ``M/D/YYYY`` portably (Windows-safe)."""
    return f"{d.month}/{d.day}/{d.year}"


def _pattern_id(trip: ScheduledTrip) -> str:
    """Stable pattern_id from the trip's shape and direction."""
    return f"{trip.shape_id}_{trip.direction_id}"


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the great-circle distance in meters between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    v = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6_371_000.0 * 2 * math.asin(math.sqrt(v))


# =============================================================================
# INPUTS
# =============================================================================


@dataclass
class Schedule:
    """The scheduled blocks and service dates the actuals are drawn from."""

    active_dates: dict[str, set[date]]
    # (service_id, route_id) -> block_id -> trips (revenue and nonrevenue) by start
    blocks: dict[tuple[str, str], dict[str, list[ScheduledTrip]]]
    stop_coords: dict[str, tuple[float, float]]


def load_schedule(gtfs_path: Path, tods_path: Optional[Path]) -> Schedule:
    """Read the GTFS feed (and TODS supplements) into blocks per service and route.

    Nonrevenue TODS trips are filed under the route of their block's revenue trips.

    Raises:
        ValueError: If a TODS trip names a block with no GTFS revenue trips.
    """
    gtfs = load_gtfs_data(
        str(gtfs_path),
        files=("trips.txt", "stop_times.txt", "stops.txt", "calendar.txt", "calendar_dates.txt"),
    )
    active = expand_service_active_dates(gtfs["calendar"], gtfs["calendar_dates"])
    stop_coords = {
        row.stop_id: (float(row.stop_lat), float(row.stop_lon))
        for row in gtfs["stops"].itertuples(index=False)
    }
    trips = gtfs["trips"].fillna("")
    stop_times = gtfs["stop_times"].fillna("")
    if tods_path is not None:
        tods = load_gtfs_data(
            str(tods_path),
            files=("trips_supplement.txt", "stop_times_supplement.txt", "stops_supplement.txt"),
        )
        tods_trips = tods["trips_supplement"].fillna("")
        stop_times = pd.concat([stop_times, tods["stop_times_supplement"].fillna("")])
        for row in tods["stops_supplement"].itertuples(index=False):
            stop_coords[row.stop_id] = (float(row.stop_lat), float(row.stop_lon))
    else:
        tods_trips = pd.DataFrame(columns=["trip_id", "service_id", "block_id", "TODS_trip_type"])

    calls: dict[str, list[dict[str, str]]] = {}
    for row in stop_times.to_dict("records"):
        calls.setdefault(row["trip_id"], []).append(row)
    for rows in calls.values():
        rows.sort(key=lambda r: int(r["stop_sequence"]))

    def scheduled(row: dict[str, str], route_id: str, tods_type: str = "") -> ScheduledTrip:
        rows = calls[row["trip_id"]]
        return ScheduledTrip(
            trip_id=row["trip_id"],
            route_id=route_id,
            direction_id=row.get("direction_id", ""),
            shape_id=row.get("shape_id", ""),
            block_id=row["block_id"],
            start=_gtfs_seconds(rows[0]["departure_time"]),
            end=_gtfs_seconds(rows[-1]["arrival_time"]),
            calls=tuple(rows),
            tods_type=tods_type,
        )

    blocks: dict[tuple[str, str], dict[str, list[ScheduledTrip]]] = {}
    block_route: dict[tuple[str, str], str] = {}
    for row in trips.to_dict("records"):
        blocks.setdefault((row["service_id"], row["route_id"]), {}).setdefault(
            row["block_id"], []
        ).append(scheduled(row, row["route_id"]))
        block_route[row["service_id"], row["block_id"]] = row["route_id"]
    for row in tods_trips.to_dict("records"):
        key = (row["service_id"], row["block_id"])
        if key not in block_route:
            raise ValueError(f"TODS trip {row['trip_id']} names unknown block {key}.")
        route_id = block_route[key]
        blocks[row["service_id"], route_id][row["block_id"]].append(
            scheduled(row, route_id, row["TODS_trip_type"])
        )
    for by_block in blocks.values():
        for chain in by_block.values():
            chain.sort(key=lambda t: (t.start, t.end))
    return Schedule(active, blocks, stop_coords)


# =============================================================================
# TRIP + STOP GENERATION
# =============================================================================


def _generate_trip_row(
    day: date,
    trip: ScheduledTrip,
    rng: random.Random,
    vehicle_id: str,
    not_before: datetime | None,
) -> tuple[dict[str, object], bool]:
    """Build one trips_performed row.

    The actual start never precedes ``not_before`` (when the vehicle finished
    its previous trip). Returns the row and a ``canceled`` flag so the matching
    stop_visits rows can react.
    """
    sched_start = _at(day, trip.start)
    sched_end = _at(day, trip.end)
    sched_dur = (sched_end - sched_start).total_seconds()
    revenue = not trip.tods_type

    # Decide canceled BEFORE drawing actuals, so we don't leak it.
    canceled = revenue and rng.random() < P_CANCELED_TRIP

    if canceled:
        actual_start: datetime | None = None
        actual_end: datetime | None = None
        sched_rel = "Canceled"
    else:
        # Real-world spread: most trips within +/- 2 minutes of schedule.
        actual_start = sched_start + timedelta(seconds=int(rng.gauss(60, 75)))
        if not_before is not None and actual_start < not_before:
            actual_start = not_before
        # Runtime drift independent of start drift.
        actual_dur = sched_dur + rng.gauss(60, 120)
        actual_end = actual_start + timedelta(seconds=int(max(actual_dur, sched_dur / 2, 60)))
        sched_rel = "Scheduled"

    meta = ROUTE_META.get(trip.route_id, {}) if revenue else {}
    row: dict[str, object] = {
        "service_date": day.strftime("%Y-%m-%d"),
        "trip_id_performed": f"TP{day.strftime('%Y%m%d')}_{trip.trip_id}",
        "vehicle_id": vehicle_id,
        # GTFS trip_id for revenue trips, the TODS trip_id for nonrevenue ones.
        "trip_id_scheduled": trip.trip_id,
        "route_id": trip.route_id if revenue else "",
        "route_type": meta.get("route_type", ""),
        "ntd_mode": meta.get("ntd_mode", ""),
        "route_type_agency": meta.get("route_type_agency", ""),
        "shape_id": trip.shape_id,
        "pattern_id": _pattern_id(trip) if revenue else "",
        "direction_id": trip.direction_id,
        "operator_id": ROUTE_META.get(trip.route_id, {}).get("operator_id", ""),
        "block_id": trip.block_id,
        "trip_start_stop_id": trip.calls[0]["stop_id"],
        "trip_end_stop_id": trip.calls[-1]["stop_id"],
        "schedule_trip_start": _iso(sched_start),
        "schedule_trip_end": _iso(sched_end),
        "actual_trip_start": _iso(actual_start),
        "actual_trip_end": _iso(actual_end),
        "trip_type": TODS_TO_TIDES_TRIP_TYPE[trip.tods_type] if trip.tods_type else "In service",
        "schedule_relationship": sched_rel,
    }
    return row, canceled


def _generate_stop_rows(
    day: date,
    trip: ScheduledTrip,
    trip_row: dict[str, object],
    stop_coords: Mapping[str, tuple[float, float]],
    rng: random.Random,
    force: str = "",
) -> list[dict[str, object]]:
    """Build the stop_visits rows for a single revenue trip (one per GTFS stop_time).

    Canceled trips emit no stop visits. ``force`` ("Skipped" or "Added") makes
    the trip carry that edge case instead of leaving it to chance.
    """
    if trip_row["schedule_relationship"] == "Canceled":
        return []

    actual_start = datetime.strptime(str(trip_row["actual_trip_start"]), "%Y-%m-%dT%H:%M:%S")
    actual_end = datetime.strptime(str(trip_row["actual_trip_end"]), "%Y-%m-%dT%H:%M:%S")
    calls = trip.calls
    n_stops = len(calls)
    pattern_id = str(trip_row["pattern_id"])
    vehicle_id = str(trip_row["vehicle_id"])

    # Actual times follow the schedule's spacing, stretched to the actual runtime.
    sched_total = max(trip.end - trip.start, 1)
    actual_total = (actual_end - actual_start).total_seconds()

    # Decide if this trip will have one skipped or added stop.
    skip_idx: int | None = None
    added_idx: int | None = None
    r = rng.random()
    if force == "Skipped":
        r = 0.0
    elif force == "Added":
        r = P_SKIPPED_STOP
    if r < P_SKIPPED_STOP and n_stops >= 3:
        skip_idx = rng.randint(1, n_stops - 2)
    elif r < P_SKIPPED_STOP + P_ADDED_STOP and n_stops >= 3:
        added_idx = rng.randint(1, n_stops - 2)

    rows: list[dict[str, object]] = []
    load = 0
    previous_departure: datetime | None = None
    for i, call in enumerate(calls):
        sched_arr_s = _gtfs_seconds(call["arrival_time"])
        sched_dep_s = _gtfs_seconds(call["departure_time"])
        frac = (sched_arr_s - trip.start) / sched_total
        actual_arr = actual_start + timedelta(
            seconds=int(actual_total * frac) + rng.randint(-10, 10)
        )
        if previous_departure is not None and actual_arr <= previous_departure:
            actual_arr = previous_departure + timedelta(seconds=5)

        # GTFS timepoint: 1 or empty means exact times.
        timepoint = call.get("timepoint", "") != "0"

        if i == skip_idx:
            # Bus drove past without stopping.
            actual_dep = actual_arr
            dwell = 0
            boarding = 0
            alighting = 0
            door_status = "Doors did not open"
            door_open = ""
            door_close = ""
            sched_rel = "Skipped"
        elif i == added_idx:
            # Unscheduled stop -- no scheduled times.
            actual_dep = actual_arr + timedelta(seconds=rng.randint(20, 50))
            dwell = int((actual_dep - actual_arr).total_seconds())
            boarding = rng.randint(0, 3)
            alighting = rng.randint(0, 2)
            door_status = "All doors opened"
            door_open = _iso(actual_arr)
            door_close = _iso(actual_dep)
            sched_rel = "Added"
        else:
            actual_dep = actual_arr + timedelta(seconds=rng.randint(15, 45))
            dwell = int((actual_dep - actual_arr).total_seconds())
            boarding = rng.choices([0, 1, 2, 3, 4], weights=[3, 4, 3, 2, 1])[0]
            alighting = rng.choices([0, 1, 2], weights=[5, 3, 2])[0]
            door_status = (
                "Front door opened and back doors remain closed"
                if alighting == 0
                else "All doors opened"
            )
            door_open = _iso(actual_arr)
            door_close = _iso(actual_dep)
            sched_rel = "Scheduled"
        previous_departure = actual_dep

        load = max(0, load + boarding - alighting)
        # TIDES distance: meters traveled from the previous stop.
        if i == 0:
            distance = 0
        elif call.get("shape_dist_traveled") and calls[i - 1].get("shape_dist_traveled"):
            feet = float(call["shape_dist_traveled"]) - float(calls[i - 1]["shape_dist_traveled"])
            distance = round(feet * METERS_PER_FOOT)
        else:
            a, b = stop_coords[calls[i - 1]["stop_id"]], stop_coords[call["stop_id"]]
            distance = round(_haversine_m(a, b))

        if sched_rel == "Added":
            sched_arr_str = ""
            sched_dep_str = ""
            scheduled_stop_sequence: int | str = ""
        else:
            sched_arr_str = _iso(_at(day, sched_arr_s))
            sched_dep_str = _iso(_at(day, sched_dep_s))
            scheduled_stop_sequence = int(call["stop_sequence"])

        rows.append(
            {
                "service_date": _format_mdy(day),
                "trip_id_performed": trip_row["trip_id_performed"],
                "trip_stop_sequence": i + 1,
                "scheduled_stop_sequence": scheduled_stop_sequence,
                "pattern_id": pattern_id,
                "vehicle_id": vehicle_id,
                "dwell": dwell,
                "stop_id": call["stop_id"],
                "timepoint": "TRUE" if timepoint else "FALSE",
                "schedule_arrival_time": sched_arr_str,
                "schedule_departure_time": sched_dep_str,
                "actual_arrival_time": _iso(actual_arr),
                "actual_departure_time": _iso(actual_dep),
                "distance": distance,
                "boarding_1": boarding,
                "alighting_1": alighting,
                "boarding_2": 0,
                "alighting_2": 0,
                "departure_load": load,
                "door_open": door_open,
                "door_close": door_close,
                "door_status": door_status,
                "ramp_deployed_time": 0,
                "ramp_failure": "FALSE",
                "kneel_deployed_time": 0,
                "lift_deployed_time": 0,
                "bike_rack_deployed": "FALSE",
                "bike_load": 0,
                "revenue": f"{boarding * 2.0:.2f}",
                "number_of_transactions": boarding,
                "schedule_relationship": sched_rel,
            }
        )

    return rows


# =============================================================================
# DRIVER
# =============================================================================


def _block_rotation(by_block: dict[str, list[ScheduledTrip]], rng: random.Random) -> list[str]:
    """Return the fixed block rotation for one (service_id, route).

    Leads with the owl block (if INCLUDE_OWL_TRIPS and the route runs one), then
    adds blocks until every direction the route runs is covered.
    """

    def directions(block_id: str) -> set[str]:
        return {t.direction_id for t in by_block[block_id] if not t.tods_type}

    def latest_start(block_id: str) -> int:
        return max(t.start for t in by_block[block_id] if not t.tods_type)

    block_ids = sorted(by_block)
    needed = set().union(*(directions(b) for b in block_ids))
    rotation: list[str] = []
    owl = max(block_ids, key=latest_start)
    if INCLUDE_OWL_TRIPS and latest_start(owl) >= OWL_START_MIN * 60:
        rotation.append(owl)
    while not rotation or needed - set().union(*(directions(b) for b in rotation)):
        covered = set().union(*(directions(b) for b in rotation))
        rest = [b for b in block_ids if b not in rotation]
        preferred = [b for b in rest if directions(b) - covered]
        rotation.append(rng.choice(preferred or rest))
    return rotation


def _run_block(
    day: date,
    trips: Sequence[ScheduledTrip],
    schedule: Schedule,
    rng: random.Random,
    vehicle_id: str,
    forced: Iterator[str],
    trip_rows: list[dict[str, object]],
    stop_rows: list[dict[str, object]],
) -> None:
    """Emit one vehicle block on one date: a trip row per trip, stop rows per revenue trip.

    One vehicle runs the whole block, trip after trip; a canceled trip gets a
    trip row but no stop visits. Rows are appended to *trip_rows* / *stop_rows*.
    """
    not_before: datetime | None = None
    for trip in trips:
        row, canceled = _generate_trip_row(day, trip, rng, vehicle_id, not_before)
        trip_rows.append(row)
        if canceled:
            continue
        not_before = datetime.strptime(str(row["actual_trip_end"]), "%Y-%m-%dT%H:%M:%S")
        if not trip.tods_type:
            visits = _generate_stop_rows(
                day, trip, row, schedule.stop_coords, rng, force=next(forced, "")
            )
            stop_rows.extend(visits)


def _holiday_pass(
    schedule: Schedule,
    trip_rows: list[dict[str, object]],
    stop_rows: list[dict[str, object]],
) -> None:
    """Append every block of each ``HOLIDAY_PASS`` service on each date it runs in range.

    Uses its own random stream so editing the pass never changes the main sample.
    """
    rng = random.Random(RANDOM_SEED + 1)
    months = set(_iter_months(START_MONTH, END_MONTH))
    for service_id, routes in HOLIDAY_PASS.items():
        dates = sorted(
            d for d in schedule.active_dates.get(service_id, set()) if (d.year, d.month) in months
        )
        for day in dates:
            fleet = iter(rng.sample(FLEET, len(FLEET)))
            for route_id in routes:
                by_block = schedule.blocks.get((service_id, route_id), {})
                for block_id in sorted(by_block):
                    _run_block(
                        day,
                        by_block[block_id],
                        schedule,
                        rng,
                        next(fleet),
                        iter(()),
                        trip_rows,
                        stop_rows,
                    )


def generate(schedule: Schedule, rng: random.Random) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build both fixtures and return them as DataFrames."""
    trip_rows: list[dict[str, object]] = []
    stop_rows: list[dict[str, object]] = []
    # The first two observed revenue trips are guaranteed one skipped and one
    # added stop, so the fixture always carries both edge cases.
    forced = iter(("Skipped", "Added"))
    rotations = {
        key: _block_rotation(by_block, rng)
        for key, by_block in sorted(schedule.blocks.items())
        if key[1] in ROUTES
    }

    for year, month in _iter_months(START_MONTH, END_MONTH):
        for service_id, n_days in DAYS_PER_MONTH.items():
            routes = [r for r in ROUTES if (service_id, r) in schedule.blocks]
            candidates = sorted(
                d
                for d in schedule.active_dates.get(service_id, set())
                if (d.year, d.month) == (year, month)
            )
            if not routes or not candidates:
                continue
            for day_idx, day in enumerate(_pick_service_days(candidates, n_days, rng)):
                fleet = iter(rng.sample(FLEET, len(FLEET)))
                for route_id in routes:
                    by_block = schedule.blocks[service_id, route_id]
                    rotation = rotations[service_id, route_id]
                    block_id = rotation[day_idx % len(rotation)]
                    vehicle_id = next(fleet)
                    _run_block(
                        day,
                        by_block[block_id],
                        schedule,
                        rng,
                        vehicle_id,
                        forced,
                        trip_rows,
                        stop_rows,
                    )

    _holiday_pass(schedule, trip_rows, stop_rows)
    trips_df = pd.DataFrame(trip_rows, columns=TRIPS_COLS)
    stops_df = pd.DataFrame(stop_rows, columns=STOPS_COLS)
    return trips_df, stops_df


def main() -> None:
    """Generate both fixtures and write them to ``OUTPUT_DIR``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rng = random.Random(RANDOM_SEED)
    trips_df, stops_df = generate(load_schedule(GTFS_PATH, TODS_PATH), rng)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trips_path = OUTPUT_DIR / "trips_performed.csv"
    stops_path = OUTPUT_DIR / "stop_visits.csv"

    trips_df.to_csv(trips_path, index=False)
    stops_df.to_csv(stops_path, index=False)

    logging.info("Wrote %d trips -> %s", len(trips_df), trips_path)
    logging.info("Wrote %d stop visits -> %s", len(stops_df), stops_path)

    # Quick sanity summary so the user can eyeball coherence at-a-glance.
    by_month_route_dir = (
        trips_df.assign(month=trips_df["service_date"].str.slice(0, 7))
        .groupby(["month", "trip_type", "route_id", "direction_id"], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "trips"})
    )
    logging.info(
        "Trips by month x trip type x route x direction:\n%s",
        by_month_route_dir.to_string(index=False),
    )


if __name__ == "__main__":
    main()
