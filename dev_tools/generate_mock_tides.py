"""Generate coherent sample fixtures for ``stop_visits.csv`` and ``trips_performed.csv``.

This script emits a pair of TIDES-shaped CSV fixtures suitable for testing
analyses that group by month, by route/direction, or by month x route x
direction. The two files share keys (``service_date``, ``trip_id_performed``,
``pattern_id``, ``vehicle_id``) so they can be joined the way real exports
allow -- generating them separately risks orphan stop visits.

Sizing is configurable at the top of the module; defaults produce a few
hundred trips and ~a thousand stop visits, which keeps the files small
enough to commit while leaving enough rows for meaningful aggregation.

A fixed random seed makes the output stable run-to-run.

Outputs:
    <OUTPUT_DIR>/trips_performed.csv  (service_date as YYYY-MM-DD)
    <OUTPUT_DIR>/stop_visits.csv      (service_date as M/D/YYYY)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR: Final[Path] = Path("./fixtures_generated")

# Routes and directions. Direction 0/1 follows the TIDES convention.
ROUTES: Final[list[str]] = ["101", "202", "303"]
DIRECTIONS: Final[list[int]] = [0, 1]

# Inclusive month range. Service days are sampled within each month.
START_MONTH: Final[tuple[int, int]] = (2025, 1)  # (year, month)
END_MONTH: Final[tuple[int, int]] = (2025, 3)

# How many service days to sample per (route, direction, month). Mondays,
# Wednesdays, and Fridays are favoured to keep the sample compact but spread.
SERVICE_DAYS_PER_MONTH: Final[int] = 4

# How many trips per (route, direction, service_day).
TRIPS_PER_DAY: Final[int] = 3

# Include one extra "owl" trip per (route, direction, day) starting in the
# 23:00-23:45 window. This is realistic late-night service and -- more
# importantly -- exercises the post-midnight footgun: in TIDES, schedule_*
# and actual_* fields are full ISO datetimes, so a trip that ends after
# midnight has its end timestamp on the *next* calendar day while
# ``service_date`` stays on the operational service day. Code that strips
# the date, groups by ``timestamp.date()`` instead of ``service_date``, or
# filters by hour-of-day without handling rollover will break on these rows.
INCLUDE_OWL_TRIPS: Final[bool] = True
OWL_START_MIN: Final[int] = 23 * 60  # earliest owl start (minutes since midnight)
OWL_START_MAX: Final[int] = 23 * 60 + 45  # latest owl start

# How many stops per trip (constant for simplicity; varied via skip/added rows).
STOPS_PER_TRIP: Final[int] = 5

# Edge-case rates. Kept small so the bulk of the data is clean for analysis.
P_SKIPPED_STOP: Final[float] = 0.03  # one stop on a trip gets schedule_relationship=Skipped
P_ADDED_STOP: Final[float] = 0.02  # one stop on a trip gets schedule_relationship=Added
P_CANCELED_TRIP: Final[float] = 0.02  # trip emitted with no actuals
P_DEADHEAD_TRIP: Final[float] = 0.0  # 0 keeps stop_visits joinable for every trip row

RANDOM_SEED: Final[int] = 42

# Route metadata. Kept compact -- one row per route.
ROUTE_META: Final[dict[str, dict[str, str]]] = {
    "101": {
        "route_type": "Local Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "LOCAL",
        "operator_id": "OP_01",
    },
    "202": {
        "route_type": "Express Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "EXPRESS",
        "operator_id": "OP_02",
    },
    "303": {
        "route_type": "Local Bus Service",
        "ntd_mode": "Bus",
        "route_type_agency": "LOCAL",
        "operator_id": "OP_03",
    },
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


# =============================================================================
# HELPERS
# =============================================================================


@dataclass(frozen=True)
class TripKey:
    """Identifies one performed trip uniquely within the fixture."""

    service_date: date
    route: str
    direction: int
    trip_idx: int  # 0..TRIPS_PER_DAY-1


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


def _pick_service_days(year: int, month: int, n: int, rng: random.Random) -> list[date]:
    """Pick *n* weekday-ish dates from a given month, evenly spread."""
    # Build all weekdays in the month, then choose evenly across them.
    d = date(year, month, 1)
    weekdays: list[date] = []
    while d.month == month:
        if d.weekday() < 5:  # Mon..Fri
            weekdays.append(d)
        d += timedelta(days=1)
    if len(weekdays) <= n:
        return weekdays
    step = len(weekdays) / n
    picks = [weekdays[int(i * step)] for i in range(n)]
    # Small jitter so we don't always get the 1st/8th/15th/22nd of every month.
    jittered: list[date] = []
    for p in picks:
        shift = rng.randint(-1, 1)
        candidate = p + timedelta(days=shift)
        if candidate.month == month and candidate.weekday() < 5:
            jittered.append(candidate)
        else:
            jittered.append(p)
    return sorted(set(jittered))


def _schedule_start_times(n: int, rng: random.Random) -> list[int]:
    """Return *n* scheduled trip start times in minutes-since-midnight.

    Spread across roughly 06:00-20:00 so service-period analysis works.
    """
    span_min = 6 * 60  # 06:00
    span_max = 20 * 60  # 20:00
    # Even slots with small jitter.
    slots = [span_min + int((span_max - span_min) * i / n) for i in range(n)]
    return [s + rng.randint(-10, 10) for s in slots]


def _minutes_to_dt(d: date, minutes: int) -> datetime:
    """Combine a date with minutes-since-midnight into a datetime."""
    return datetime(d.year, d.month, d.day) + timedelta(minutes=minutes)


def _iso(ts: datetime | None) -> str:
    """ISO-8601 string with seconds, or empty string if *ts* is None."""
    return "" if ts is None else ts.strftime("%Y-%m-%dT%H:%M:%S")


def _format_mdy(d: date) -> str:
    """Format a date as ``M/D/YYYY`` portably (Windows-safe)."""
    return f"{d.month}/{d.day}/{d.year}"


def _pattern_id(route: str, direction: int) -> str:
    """Stable pattern_id from route and direction."""
    return f"shp-{route}-{'01' if direction == 0 else '51'}"


def _shape_id(route: str, direction: int) -> str:
    """Stable shape_id from route and direction."""
    suffix = "OUT" if direction == 0 else "IN"
    return f"SHAPE_{route}_{suffix}"


def _stop_ids_for(route: str, direction: int) -> list[int]:
    """Stable list of stop ids for a (route, direction) pair."""
    # Direction 0 walks ascending; direction 1 walks descending across the
    # same physical corridor so the two directions feel like a real loop.
    base = int(route) * 10
    forward = [base + i for i in range(STOPS_PER_TRIP + 2)]
    return forward if direction == 0 else list(reversed(forward))


# =============================================================================
# TRIP + STOP GENERATION
# =============================================================================


def _generate_trip_row(
    key: TripKey,
    sched_start_min: int,
    rng: random.Random,
    vehicle_id: str,
    block_id: str,
) -> tuple[dict[str, object], bool, bool]:
    """Build one trips_performed row.

    Returns the row, a ``canceled`` flag, and a ``deadhead`` flag so the
    matching stop_visits rows can react to both.
    """
    sched_start = _minutes_to_dt(key.service_date, sched_start_min)
    # Trips are ~30-45 minutes scheduled, with bigger spread on route 303.
    sched_dur = rng.randint(28, 46)
    sched_end = sched_start + timedelta(minutes=sched_dur)

    # Decide canceled/deadhead BEFORE drawing actuals, so we don't leak them.
    canceled = rng.random() < P_CANCELED_TRIP
    deadhead = (not canceled) and rng.random() < P_DEADHEAD_TRIP

    if canceled:
        actual_start: datetime | None = None
        actual_end: datetime | None = None
        sched_rel = "Canceled"
        trip_type = "In service"
    else:
        # Real-world spread: most trips within +/- 2 minutes of schedule.
        actual_start = sched_start + timedelta(seconds=int(rng.gauss(60, 75)))
        # Runtime drift independent of start drift.
        actual_dur = sched_dur + rng.gauss(1.0, 2.0)
        actual_end = actual_start + timedelta(minutes=max(actual_dur, 5))
        sched_rel = "Scheduled"
        trip_type = "Deadhead" if deadhead else "In service"

    meta = ROUTE_META[key.route]
    pattern_id = _pattern_id(key.route, key.direction)
    stops = _stop_ids_for(key.route, key.direction)

    # trip_id_performed: stable, sortable, no collision risk across the fixture.
    date_compact = key.service_date.strftime("%Y%m%d")
    trip_id_performed = f"TP{date_compact}_{key.route}_{key.direction}_{key.trip_idx:02d}"

    # trip_id_scheduled is blank for non-revenue legs in the real exports.
    trip_id_scheduled = "" if deadhead else f"TRIP_{key.route}_{key.direction}_{key.trip_idx:02d}"

    row: dict[str, object] = {
        "service_date": key.service_date.strftime("%Y-%m-%d"),
        "trip_id_performed": trip_id_performed,
        "vehicle_id": vehicle_id,
        "trip_id_scheduled": trip_id_scheduled,
        "route_id": key.route,
        "route_type": meta["route_type"],
        "ntd_mode": meta["ntd_mode"],
        "route_type_agency": meta["route_type_agency"],
        "shape_id": _shape_id(key.route, key.direction),
        "pattern_id": pattern_id,
        "direction_id": key.direction,
        "operator_id": meta["operator_id"],
        "block_id": block_id,
        "trip_start_stop_id": stops[0],
        "trip_end_stop_id": stops[-1],
        "schedule_trip_start": _iso(sched_start),
        "schedule_trip_end": _iso(sched_end),
        "actual_trip_start": _iso(actual_start),
        "actual_trip_end": _iso(actual_end),
        "trip_type": trip_type,
        "schedule_relationship": sched_rel,
    }
    return row, canceled, deadhead


def _generate_stop_rows(
    key: TripKey,
    trip_row: dict[str, object],
    canceled: bool,
    deadhead: bool,
    rng: random.Random,
) -> list[dict[str, object]]:
    """Build the stop_visits rows for a single trip.

    Canceled trips emit no stop visits. Deadheads do, but with no boardings.
    """
    if canceled:
        return []

    sched_start = datetime.strptime(str(trip_row["schedule_trip_start"]), "%Y-%m-%dT%H:%M:%S")
    sched_end = datetime.strptime(str(trip_row["schedule_trip_end"]), "%Y-%m-%dT%H:%M:%S")
    actual_start = datetime.strptime(str(trip_row["actual_trip_start"]), "%Y-%m-%dT%H:%M:%S")
    actual_end = datetime.strptime(str(trip_row["actual_trip_end"]), "%Y-%m-%dT%H:%M:%S")

    n_stops = STOPS_PER_TRIP
    stops = _stop_ids_for(key.route, key.direction)[:n_stops]
    pattern_id = str(trip_row["pattern_id"])
    vehicle_id = str(trip_row["vehicle_id"])

    # Evenly-spaced scheduled and actual timestamps along the trip.
    sched_total = (sched_end - sched_start).total_seconds()
    actual_total = (actual_end - actual_start).total_seconds()

    # Decide if this trip will have one skipped or added stop.
    skip_idx: int | None = None
    added_idx: int | None = None
    r = rng.random()
    if r < P_SKIPPED_STOP and n_stops >= 3:
        skip_idx = rng.randint(1, n_stops - 2)
    elif r < P_SKIPPED_STOP + P_ADDED_STOP and n_stops >= 3:
        added_idx = rng.randint(1, n_stops - 2)

    rows: list[dict[str, object]] = []
    load = 0
    for i in range(n_stops):
        frac = i / max(n_stops - 1, 1)
        sched_arr = sched_start + timedelta(seconds=int(sched_total * frac))
        sched_dep = sched_arr + timedelta(seconds=rng.randint(15, 30))
        actual_arr = actual_start + timedelta(
            seconds=int(actual_total * frac) + rng.randint(-10, 10)
        )

        timepoint = i in (0, n_stops - 1) or i % 2 == 1  # endpoints + alternates

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
            # Deadheads have no riders. Otherwise small positive counts.
            boarding = 0 if deadhead else rng.choices([0, 1, 2, 3, 4], weights=[3, 4, 3, 2, 1])[0]
            alighting = 0 if deadhead else rng.choices([0, 1, 2], weights=[5, 3, 2])[0]
            door_status = (
                "Front door opened and back doors remain closed"
                if alighting == 0
                else "All doors opened"
            )
            door_open = _iso(actual_arr)
            door_close = _iso(actual_dep)
            sched_rel = "Scheduled"

        load = max(0, load + boarding - alighting)
        # Distance: 0 for the first stop, then growing increments.
        distance = 0 if i == 0 else rng.randint(150, 700)

        if sched_rel == "Added":
            sched_arr_str = ""
            sched_dep_str = ""
            scheduled_stop_sequence: int | str = ""
        else:
            sched_arr_str = _iso(sched_arr)
            sched_dep_str = _iso(sched_dep)
            # scheduled_stop_sequence may differ from trip_stop_sequence
            # to reflect skipped scheduled stops upstream; here we just match.
            scheduled_stop_sequence = i + 1

        rows.append(
            {
                "service_date": _format_mdy(key.service_date),
                "trip_id_performed": trip_row["trip_id_performed"],
                "trip_stop_sequence": i + 1,
                "scheduled_stop_sequence": scheduled_stop_sequence,
                "pattern_id": pattern_id,
                "vehicle_id": vehicle_id,
                "dwell": dwell,
                "stop_id": stops[i],
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


def generate(rng: random.Random) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build both fixtures and return them as DataFrames."""
    trip_rows: list[dict[str, object]] = []
    stop_rows: list[dict[str, object]] = []

    for year, month in _iter_months(START_MONTH, END_MONTH):
        for route in ROUTES:
            for direction in DIRECTIONS:
                days = _pick_service_days(year, month, SERVICE_DAYS_PER_MONTH, rng)
                for day in days:
                    # One vehicle and one block per (route, day) for realism.
                    vehicle_id = f"VEH_{int(route):03d}_{day.day:02d}"
                    block_id = f"BLOCK_{route}_{day.strftime('%m%d')}"
                    starts = _schedule_start_times(TRIPS_PER_DAY, rng)
                    # Offset direction 1 by ~30 min so the two directions
                    # don't have identical start times.
                    if direction == 1:
                        starts = [s + 30 for s in starts]
                    # Add one owl trip per (route, direction, day) so that
                    # late-night, midnight-crossing trips are present in the
                    # fixture (see INCLUDE_OWL_TRIPS docstring above).
                    if INCLUDE_OWL_TRIPS:
                        starts.append(rng.randint(OWL_START_MIN, OWL_START_MAX))
                    for idx, sched_start_min in enumerate(starts):
                        key = TripKey(day, route, direction, idx)
                        trip_row, canceled, deadhead = _generate_trip_row(
                            key, sched_start_min, rng, vehicle_id, block_id
                        )
                        trip_rows.append(trip_row)
                        stop_rows.extend(
                            _generate_stop_rows(key, trip_row, canceled, deadhead, rng)
                        )

    trips_df = pd.DataFrame(trip_rows, columns=TRIPS_COLS)
    stops_df = pd.DataFrame(stop_rows, columns=STOPS_COLS)
    return trips_df, stops_df


def main() -> None:
    """Generate both fixtures and write them to ``OUTPUT_DIR``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rng = random.Random(RANDOM_SEED)
    trips_df, stops_df = generate(rng)

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
        .groupby(["month", "route_id", "direction_id"], as_index=False)
        .size()
        .rename(columns={"size": "trips"})
    )
    logging.info(
        "Trips by month x route x direction:\n%s",
        by_month_route_dir.to_string(index=False),
    )


if __name__ == "__main__":
    main()
