"""Generate a rich mock GTFS feed for dev and testing.

Produces a ``gtfs_mock/`` folder (default: ``tests/fixtures/gtfs_mock/``) with
all required GTFS .txt files plus ``shapes.txt`` and ``calendar_dates.txt``.

Network design
--------------
Ten routes spanning Washington DC and Ottawa regions, covering the cardinal and
diagonal directions plus a square loop:

  Route  Dir      Service pattern          Region    Headway  Spacing
  -----  -------  -----------------------  --------  -------  --------
  101    N-S      Weekday all-day + late    DC        15 min   1,000 ft
  102    E-W      Weekday all-day + late    DC        30 min   500 ft
  103    NW-SE    Weekday all-day           DC        60 min   2,000 ft
  104    NE-SW    Express peak only (1-dir) DC        30 min   1,000 ft
  105    Square   All-day + late night      DC        20 min   1,000 ft
  201    N-S      Weekday all-day + late    Ottawa    15 min   1,000 ft
  202    E-W      Weekday all-day + late    Ottawa    30 min   500 ft
  203    NW-SE    Weekday all-day           Ottawa    60 min   2,000 ft
  204    NE-SW    Express peak only (1-dir) Ottawa    30 min   1,000 ft
  205    Square   All-day + late night      Ottawa    20 min   1,000 ft

Service IDs
-----------
  WKDY      Mon-Fri
  SAT       Saturday
  SUN       Sunday
  HOL       Holiday-only (calendar_dates.txt exceptions)

Special patterns
----------------
* Route 104 / 204 (NE-SW express): weekday peak only, one direction (NE→SW
  in AM peak, SW→NE in PM peak), no weekend service.
* Route 103 / 203 (NW-SE): weekday only, no late night.
* Route 901 (holiday-only): runs on holidays only via calendar_dates.txt.
* Stops on route 102 between S_EW_04 and S_EW_06 are skipped (served by 101
  which overlaps that segment) — demonstrating a skip-stop / two-route overlap.
* Route 105 / 205 (square loop) share corner stops with the cardinal routes,
  demonstrating overlapping stop usage.

Headways
--------
All departure times fall on multiples of 5 minutes.  Headways are 5, 15, 20,
30, or 60 minutes.

Blocks
------
Each block is capped at 8 hours of scheduled run time.  A new block is started
whenever adding the next trip would push the block past the 8-hour cap.

Usage
-----
Run from the repo root::

    python dev_tools/build_gtfs_fixtures.py

An optional ``--output`` flag overrides the default output directory::

    python dev_tools/build_gtfs_fixtures.py --output /tmp/my_gtfs
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_OUTPUT: Final[Path] = Path("tests/fixtures/gtfs_mock")

FEED_START: Final[str] = "20260101"
FEED_END: Final[str] = "20261231"

# Holiday dates for calendar_dates.txt (US federal holidays in 2026)
HOLIDAY_DATES: Final[list[str]] = [
    "20260101",  # New Year's Day
    "20260119",  # MLK Day
    "20260216",  # Presidents' Day
    "20260525",  # Memorial Day
    "20260704",  # Independence Day
    "20260907",  # Labor Day
    "20261109",  # Veterans Day (observed)
    "20261126",  # Thanksgiving
    "20261225",  # Christmas
]

# Approximate feet-per-degree latitude at ~38-45 N
_FT_PER_DEG_LAT: Final[float] = 364_000.0
_FT_PER_DEG_LON_DC: Final[float] = 289_000.0   # ~38.9 N
_FT_PER_DEG_LON_OTT: Final[float] = 255_000.0  # ~45.4 N

MAX_BLOCK_SECONDS: Final[int] = 8 * 3600  # 8-hour block cap

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================


def _step_lat(ft: float) -> float:
    """Convert feet to degrees latitude."""
    return ft / _FT_PER_DEG_LAT


def _step_lon_dc(ft: float) -> float:
    """Convert feet to degrees longitude at DC latitude."""
    return ft / _FT_PER_DEG_LON_DC


def _step_lon_ott(ft: float) -> float:
    """Convert feet to degrees longitude at Ottawa latitude."""
    return ft / _FT_PER_DEG_LON_OTT


def _haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in feet between two WGS-84 points."""
    r_ft = 20_925_524.9
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r_ft * math.asin(math.sqrt(a))


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Stop:
    """A single GTFS stop."""

    stop_id: str
    stop_name: str
    lat: float
    lon: float


@dataclass
class RouteSpec:
    """Everything needed to generate trips for one route."""

    route_id: str
    agency_id: str
    short_name: str
    long_name: str
    route_type: int
    stops: list[Stop]
    # service_ids -> list of (start_hhmm, end_hhmm, headway_min) windows
    service_windows: dict[str, list[tuple[int, int, int]]]
    # seconds per stop dwell + travel to next stop (index i = travel *from* stop i)
    travel_times: list[int]
    # if True, only outbound trips are generated (express one-direction pattern)
    peak_express: bool = False
    # shape_id to assign (generated automatically if empty)
    shape_id: str = ""

    def __post_init__(self) -> None:
        """Assign default shape_id."""
        if not self.shape_id:
            self.shape_id = f"SHP_{self.route_id}"


# =============================================================================
# STOP DEFINITIONS
# =============================================================================

# ---- DC region anchor -------------------------------------------------------
_DC_LAT: Final[float] = 38.900
_DC_LON: Final[float] = -77.050


def _dc_stops_ns() -> list[Stop]:
    """Ten N-S stops spaced ~1,000 ft apart, DC region."""
    dlat = _step_lat(1000)
    stops = []
    for i in range(10):
        lat = _DC_LAT - 4 * dlat + i * dlat
        stops.append(Stop(f"S_NS_{i+1:02d}", f"North-South Ave & {_ordinal(i+1)} Cross St", lat, _DC_LON))
    return stops


def _dc_stops_ew() -> list[Stop]:
    """Fourteen E-W stops spaced ~500 ft apart, DC region."""
    dlon = _step_lon_dc(500)
    stops = []
    for i in range(14):
        lon = _DC_LON - 3.5 * dlon + i * dlon
        stops.append(Stop(f"S_EW_{i+1:02d}", f"Main Corridor & Stop {i+1}", _DC_LAT, lon))
    return stops


def _dc_stops_nwse() -> list[Stop]:
    """Eight NW-SE stops spaced ~2,000 ft apart, DC region."""
    dlat = _step_lat(2000)
    dlon = _step_lon_dc(2000)
    stops = []
    for i in range(8):
        lat = _DC_LAT + (3.5 - i) * dlat
        lon = _DC_LON - (3.5 - i) * dlon
        stops.append(Stop(f"S_NWSE_{i+1:02d}", f"Diagonal NW/SE & Marker {i+1}", lat, lon))
    return stops


def _dc_stops_nesw() -> list[Stop]:
    """Eight NE-SW stops spaced ~1,000 ft apart, DC region (express)."""
    dlat = _step_lat(1000)
    dlon = _step_lon_dc(1000)
    stops = []
    for i in range(8):
        lat = _DC_LAT + (3.5 - i) * dlat
        lon = _DC_LON + (3.5 - i) * dlon
        stops.append(Stop(f"S_NESW_{i+1:02d}", f"Express NE/SW & Point {i+1}", lat, lon))
    return stops


def _dc_stops_square() -> list[Stop]:
    """Twelve stops forming a square loop, DC region, ~1,000 ft spacing."""
    dlat = _step_lat(1000)
    dlon = _step_lon_dc(1000)
    lat0, lon0 = _DC_LAT - 0.003, _DC_LON - 0.003
    # 3 stops per side (corners shared): N side, E side, S side, W side
    coords = []
    for i in range(3):
        coords.append((lat0 + 3 * dlat, lon0 + i * dlon))   # N, going E
    for i in range(3):
        coords.append((lat0 + (3 - i) * dlat, lon0 + 3 * dlon))  # E, going S
    for i in range(3):
        coords.append((lat0, lon0 + (3 - i) * dlon))          # S, going W
    for i in range(3):
        coords.append((lat0 + i * dlat, lon0))                 # W, going N
    return [
        Stop(f"S_SQ_{i+1:02d}", f"Square Loop Stop {i+1}", lat, lon)
        for i, (lat, lon) in enumerate(coords)
    ]


# ---- Ottawa region anchor ---------------------------------------------------
_OTT_LAT: Final[float] = 45.420
_OTT_LON: Final[float] = -75.690


def _ott_stops_ns() -> list[Stop]:
    """Ten N-S stops, Ottawa region."""
    dlat = _step_lat(1000)
    stops = []
    for i in range(10):
        lat = _OTT_LAT - 4 * dlat + i * dlat
        stops.append(Stop(f"O_NS_{i+1:02d}", f"Chemin Nord-Sud & Rue {_ordinal(i+1)}", lat, _OTT_LON))
    return stops


def _ott_stops_ew() -> list[Stop]:
    """Fourteen E-W stops, Ottawa region."""
    dlon = _step_lon_ott(500)
    stops = []
    for i in range(14):
        lon = _OTT_LON - 3.5 * dlon + i * dlon
        stops.append(Stop(f"O_EW_{i+1:02d}", f"Promenade Est-Ouest & Arrêt {i+1}", _OTT_LAT, lon))
    return stops


def _ott_stops_nwse() -> list[Stop]:
    """Eight NW-SE stops, Ottawa region."""
    dlat = _step_lat(2000)
    dlon = _step_lon_ott(2000)
    stops = []
    for i in range(8):
        lat = _OTT_LAT + (3.5 - i) * dlat
        lon = _OTT_LON - (3.5 - i) * dlon
        stops.append(Stop(f"O_NWSE_{i+1:02d}", f"Diagonale NO/SE & Repère {i+1}", lat, lon))
    return stops


def _ott_stops_nesw() -> list[Stop]:
    """Eight NE-SW stops, Ottawa region (express)."""
    dlat = _step_lat(1000)
    dlon = _step_lon_ott(1000)
    stops = []
    for i in range(8):
        lat = _OTT_LAT + (3.5 - i) * dlat
        lon = _OTT_LON + (3.5 - i) * dlon
        stops.append(Stop(f"O_NESW_{i+1:02d}", f"Express NE/SO & Point {i+1}", lat, lon))
    return stops


def _ott_stops_square() -> list[Stop]:
    """Twelve stops forming a square loop, Ottawa region."""
    dlat = _step_lat(1000)
    dlon = _step_lon_ott(1000)
    lat0, lon0 = _OTT_LAT - 0.003, _OTT_LON - 0.003
    coords = []
    for i in range(3):
        coords.append((lat0 + 3 * dlat, lon0 + i * dlon))
    for i in range(3):
        coords.append((lat0 + (3 - i) * dlat, lon0 + 3 * dlon))
    for i in range(3):
        coords.append((lat0, lon0 + (3 - i) * dlon))
    for i in range(3):
        coords.append((lat0 + i * dlat, lon0))
    return [
        Stop(f"O_SQ_{i+1:02d}", f"Boucle Carré Arrêt {i+1}", lat, lon)
        for i, (lat, lon) in enumerate(coords)
    ]


def _ordinal(n: int) -> str:
    """Return simple English ordinal string for n (1→'1st', etc.)."""
    sfx = {1: "st", 2: "nd", 3: "rd"}
    return f"{n}{sfx.get(n if n < 20 else n % 10, 'th')}"


# =============================================================================
# TRAVEL TIME ESTIMATION
# =============================================================================

_MPH_LOCAL: Final[float] = 15.0
_MPH_EXPRESS: Final[float] = 25.0
_DWELL_SEC: Final[int] = 30


def _travel_times(stops: list[Stop], express: bool = False) -> list[int]:
    """Return a list of travel+dwell seconds for each inter-stop segment.

    Index i = seconds between stop i and stop i+1 (includes dwell at stop i).
    The list has len(stops)-1 entries.
    """
    mph = _MPH_EXPRESS if express else _MPH_LOCAL
    fps = mph * 5280 / 3600
    times = []
    for i in range(len(stops) - 1):
        dist = _haversine_ft(stops[i].lat, stops[i].lon, stops[i + 1].lat, stops[i + 1].lon)
        travel = int(dist / fps)
        times.append(_DWELL_SEC + travel)
    return times


# =============================================================================
# SERVICE WINDOWS
# =============================================================================

# Each tuple: (start_hhmm_as_int, end_hhmm_as_int, headway_minutes)
# Times in GTFS can exceed 2400 for post-midnight service.

_WKDY_ALL_DAY_LATE: Final[list[tuple[int, int, int]]] = [
    (500, 900, 15),    # early AM, 15-min
    (900, 1500, 30),   # midday, 30-min
    (1500, 1900, 15),  # PM peak, 15-min
    (1900, 2200, 30),  # evening, 30-min
    (2200, 2500, 60),  # late night (post-midnight), 60-min
]

_WKDY_ALL_DAY: Final[list[tuple[int, int, int]]] = [
    (600, 900, 30),
    (900, 1500, 60),
    (1500, 1900, 30),
    (1900, 2200, 60),
]

_WKDY_PEAK_EXPRESS_AM: Final[list[tuple[int, int, int]]] = [
    (630, 930, 30),
]

_WKDY_PEAK_EXPRESS_PM: Final[list[tuple[int, int, int]]] = [
    (1600, 1900, 30),
]

_SAT_ALL_DAY_LATE: Final[list[tuple[int, int, int]]] = [
    (700, 1200, 30),
    (1200, 2000, 30),
    (2000, 2400, 60),
]

_SUN_ALL_DAY: Final[list[tuple[int, int, int]]] = [
    (800, 1200, 60),
    (1200, 2000, 30),
    (2000, 2200, 60),
]

_HOL_REDUCED: Final[list[tuple[int, int, int]]] = [
    (900, 1800, 60),
]


# =============================================================================
# ROUTE CATALOG
# =============================================================================


def _build_route_specs() -> list[RouteSpec]:
    """Instantiate all RouteSpec objects."""
    dc_ns = _dc_stops_ns()
    dc_ew = _dc_stops_ew()
    dc_nwse = _dc_stops_nwse()
    dc_nesw = _dc_stops_nesw()
    dc_sq = _dc_stops_square()

    ott_ns = _ott_stops_ns()
    ott_ew = _ott_stops_ew()
    ott_nwse = _ott_stops_nwse()
    ott_nesw = _ott_stops_nesw()
    ott_sq = _ott_stops_square()

    # Route 102/202 skip stops 4-6 (served by 101/201 overlap segment):
    # keep stops 1-3 and 7-14 in the stop list; stops 4-6 appear in 101/201.
    dc_ew_skipped = dc_ew[:3] + dc_ew[6:]
    ott_ew_skipped = ott_ew[:3] + ott_ew[6:]

    return [
        # --- DC region -------------------------------------------------------
        RouteSpec(
            route_id="101",
            agency_id="DC_TRANSIT",
            short_name="101",
            long_name="North-South Line",
            route_type=3,
            stops=dc_ns,
            service_windows={
                "WKDY": _WKDY_ALL_DAY_LATE,
                "SAT": _SAT_ALL_DAY_LATE,
                "SUN": _SUN_ALL_DAY,
            },
            travel_times=_travel_times(dc_ns),
        ),
        RouteSpec(
            route_id="102",
            agency_id="DC_TRANSIT",
            short_name="102",
            long_name="East-West Corridor",
            route_type=3,
            stops=dc_ew_skipped,   # stops 4-6 skipped (overlap with 101)
            service_windows={
                "WKDY": _WKDY_ALL_DAY_LATE,
                "SAT": _SAT_ALL_DAY_LATE,
                "SUN": _SUN_ALL_DAY,
            },
            travel_times=_travel_times(dc_ew_skipped),
        ),
        RouteSpec(
            route_id="103",
            agency_id="DC_TRANSIT",
            short_name="103",
            long_name="Northwest-Southeast Diagonal",
            route_type=3,
            stops=dc_nwse,
            service_windows={
                "WKDY": _WKDY_ALL_DAY,
            },
            travel_times=_travel_times(dc_nwse),
        ),
        RouteSpec(
            route_id="104",
            agency_id="DC_TRANSIT",
            short_name="104",
            long_name="Northeast-Southwest Express",
            route_type=3,
            stops=dc_nesw,
            service_windows={
                "WKDY": _WKDY_PEAK_EXPRESS_AM + _WKDY_PEAK_EXPRESS_PM,
            },
            travel_times=_travel_times(dc_nesw, express=True),
            peak_express=True,
        ),
        RouteSpec(
            route_id="105",
            agency_id="DC_TRANSIT",
            short_name="105",
            long_name="Downtown Square Loop",
            route_type=3,
            stops=dc_sq,
            service_windows={
                "WKDY": [
                    (500, 900, 20),
                    (900, 1500, 20),
                    (1500, 1900, 20),
                    (1900, 2200, 20),
                    (2200, 2500, 60),
                ],
                "SAT": [(700, 2400, 20)],
                "SUN": [(800, 2200, 20)],
            },
            travel_times=_travel_times(dc_sq),
        ),
        # --- Ottawa region ---------------------------------------------------
        RouteSpec(
            route_id="201",
            agency_id="OC_TRANSIT",
            short_name="201",
            long_name="Ligne Nord-Sud",
            route_type=3,
            stops=ott_ns,
            service_windows={
                "WKDY": _WKDY_ALL_DAY_LATE,
                "SAT": _SAT_ALL_DAY_LATE,
                "SUN": _SUN_ALL_DAY,
            },
            travel_times=_travel_times(ott_ns),
        ),
        RouteSpec(
            route_id="202",
            agency_id="OC_TRANSIT",
            short_name="202",
            long_name="Couloir Est-Ouest",
            route_type=3,
            stops=ott_ew_skipped,
            service_windows={
                "WKDY": _WKDY_ALL_DAY_LATE,
                "SAT": _SAT_ALL_DAY_LATE,
                "SUN": _SUN_ALL_DAY,
            },
            travel_times=_travel_times(ott_ew_skipped),
        ),
        RouteSpec(
            route_id="203",
            agency_id="OC_TRANSIT",
            short_name="203",
            long_name="Diagonale NO-SE",
            route_type=3,
            stops=ott_nwse,
            service_windows={
                "WKDY": _WKDY_ALL_DAY,
            },
            travel_times=_travel_times(ott_nwse),
        ),
        RouteSpec(
            route_id="204",
            agency_id="OC_TRANSIT",
            short_name="204",
            long_name="Express NE-SO",
            route_type=3,
            stops=ott_nesw,
            service_windows={
                "WKDY": _WKDY_PEAK_EXPRESS_AM + _WKDY_PEAK_EXPRESS_PM,
            },
            travel_times=_travel_times(ott_nesw, express=True),
            peak_express=True,
        ),
        RouteSpec(
            route_id="205",
            agency_id="OC_TRANSIT",
            short_name="205",
            long_name="Boucle Centre-Ville",
            route_type=3,
            stops=ott_sq,
            service_windows={
                "WKDY": [
                    (500, 900, 20),
                    (900, 1500, 20),
                    (1500, 1900, 20),
                    (1900, 2200, 20),
                    (2200, 2500, 60),
                ],
                "SAT": [(700, 2400, 20)],
                "SUN": [(800, 2200, 20)],
            },
            travel_times=_travel_times(ott_sq),
        ),
        # --- Holiday-only route (DC) -----------------------------------------
        RouteSpec(
            route_id="901",
            agency_id="DC_TRANSIT",
            short_name="901",
            long_name="Holiday Shuttle",
            route_type=3,
            stops=dc_ns[:5],  # shorter shuttle on 5 stops
            service_windows={
                "HOL": _HOL_REDUCED,
            },
            travel_times=_travel_times(dc_ns[:5]),
        ),
    ]


# =============================================================================
# TIME UTILITIES
# =============================================================================


def _hhmm_to_sec(hhmm: int) -> int:
    """Convert integer HHMM (e.g. 1435) to seconds since midnight."""
    return (hhmm // 100) * 3600 + (hhmm % 100) * 60


def _sec_to_gtfs(sec: int) -> str:
    """Format seconds-since-midnight as HH:MM:SS (GTFS allows >24:00:00)."""
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _round_up_to_5(sec: int) -> int:
    """Round seconds up to the next multiple of 5 minutes."""
    five = 5 * 60
    return ((sec + five - 1) // five) * five


# =============================================================================
# TRIP / STOP-TIME GENERATION
# =============================================================================


@dataclass
class TripRow:
    """One row destined for trips.txt."""

    route_id: str
    service_id: str
    trip_id: str
    shape_id: str
    direction_id: int
    block_id: str
    trip_headsign: str


@dataclass
class StopTimeRow:
    """One row destined for stop_times.txt."""

    trip_id: str
    arrival: str
    departure: str
    stop_id: str
    stop_sequence: int
    timepoint: int


def _generate_trips_for_route(
    spec: RouteSpec,
    trip_counter: list[int],
    block_counter: list[int],
) -> tuple[list[TripRow], list[StopTimeRow]]:
    """Generate all trips and stop-times for a single RouteSpec.

    Args:
        spec: The route specification.
        trip_counter: Single-element list used as a mutable integer counter.
        block_counter: Single-element list used as a mutable integer counter.

    Returns:
        Tuple of (trip_rows, stop_time_rows).
    """
    trips: list[TripRow] = []
    stop_times: list[StopTimeRow] = []

    trip_duration = sum(spec.travel_times)

    for service_id, windows in spec.service_windows.items():
        # Collapse all windows for this service into an ordered departure list.
        departures_outbound: list[int] = []
        for start_hhmm, end_hhmm, headway_min in windows:
            start_sec = _round_up_to_5(_hhmm_to_sec(start_hhmm))
            end_sec = _hhmm_to_sec(end_hhmm)
            hw_sec = headway_min * 60
            t = start_sec
            while t < end_sec:
                departures_outbound.append(t)
                t += hw_sec

        # For peak_express: AM window = outbound (NE→SW), PM = inbound (SW→NE).
        # For normal routes: generate both directions for all departures.

        def _emit_direction(
            dep_list: list[int],
            stops: list[Stop],
            travel: list[int],
            direction_id: int,
            headsign: str,
        ) -> None:
            """Emit trip + stop-time rows for one direction's departure list."""
            block_start: int | None = None
            block_id: str | None = None

            for dep_sec in dep_list:
                # Start a new block if none open or current would exceed 8 h.
                if block_id is None or (dep_sec + trip_duration - block_start) > MAX_BLOCK_SECONDS:  # type: ignore[operator]
                    block_counter[0] += 1
                    block_id = f"BLK_{block_counter[0]:04d}"
                    block_start = dep_sec

                trip_counter[0] += 1
                trip_id = f"T_{trip_counter[0]:05d}"

                trips.append(
                    TripRow(
                        route_id=spec.route_id,
                        service_id=service_id,
                        trip_id=trip_id,
                        shape_id=spec.shape_id,
                        direction_id=direction_id,
                        block_id=block_id,
                        trip_headsign=headsign,
                    )
                )

                cur = dep_sec
                for seq, stop in enumerate(stops, start=1):
                    stop_times.append(
                        StopTimeRow(
                            trip_id=trip_id,
                            arrival=_sec_to_gtfs(cur),
                            departure=_sec_to_gtfs(cur),
                            stop_id=stop.stop_id,
                            stop_sequence=seq,
                            timepoint=1 if (seq == 1 or seq == len(stops)) else 0,
                        )
                    )
                    if seq < len(stops):
                        cur += travel[seq - 1]

        if spec.peak_express:
            # Split departures into AM (first window) and PM (second window).
            # AM = outbound (direction 0), PM = inbound (direction 1, reversed stops).
            midpoint = len(windows) // 2
            am_deps: list[int] = []
            pm_deps: list[int] = []
            dep_idx = 0
            for win_i, (start_hhmm, end_hhmm, headway_min) in enumerate(windows):
                start_sec = _round_up_to_5(_hhmm_to_sec(start_hhmm))
                end_sec = _hhmm_to_sec(end_hhmm)
                hw_sec = headway_min * 60
                t = start_sec
                while t < end_sec:
                    if win_i < midpoint:
                        am_deps.append(t)
                    else:
                        pm_deps.append(t)
                    t += hw_sec

            term_out = spec.stops[-1].stop_name
            term_in = spec.stops[0].stop_name
            _emit_direction(am_deps, spec.stops, spec.travel_times, 0, f"To {term_out}")
            rev_stops = list(reversed(spec.stops))
            rev_travel = list(reversed(spec.travel_times))
            _emit_direction(pm_deps, rev_stops, rev_travel, 1, f"To {term_in}")
        else:
            term_out = spec.stops[-1].stop_name
            term_in = spec.stops[0].stop_name
            _emit_direction(departures_outbound, spec.stops, spec.travel_times, 0, f"To {term_out}")
            # Inbound: reverse stops and travel times.
            rev_stops = list(reversed(spec.stops))
            rev_travel = list(reversed(spec.travel_times))
            inbound_deps = [_round_up_to_5(d + trip_duration) for d in departures_outbound]
            _emit_direction(inbound_deps, rev_stops, rev_travel, 1, f"To {term_in}")

    return trips, stop_times


# =============================================================================
# SHAPES
# =============================================================================


def _build_shapes(specs: list[RouteSpec]) -> list[dict[str, str]]:
    """Build shapes.txt rows from route stop sequences.

    Each RouteSpec yields one shape (outbound direction).  The shape points
    follow the stop lat/lon in order, with dist_traveled in feet.
    """
    rows: list[dict[str, str]] = []
    for spec in specs:
        cum_dist = 0.0
        for seq, stop in enumerate(spec.stops):
            rows.append(
                {
                    "shape_id": spec.shape_id,
                    "shape_pt_lat": f"{stop.lat:.6f}",
                    "shape_pt_lon": f"{stop.lon:.6f}",
                    "shape_pt_sequence": str(seq + 1),
                    "shape_dist_traveled": f"{cum_dist:.1f}",
                }
            )
            if seq < len(spec.stops) - 1:
                cum_dist += _haversine_ft(
                    stop.lat, stop.lon, spec.stops[seq + 1].lat, spec.stops[seq + 1].lon
                )
    return rows


# =============================================================================
# WRITERS
# =============================================================================


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write a GTFS .txt file (CSV with header)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>6,} rows  -> {path.name}")


# =============================================================================
# MAIN ASSEMBLY
# =============================================================================


def build(output_dir: Path) -> None:
    """Generate all GTFS files in *output_dir*.

    Args:
        output_dir: Directory to write GTFS .txt files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_route_specs()

    # --- agency.txt ----------------------------------------------------------
    agencies = [
        {
            "agency_id": "DC_TRANSIT",
            "agency_name": "DC Metro Transit",
            "agency_url": "https://www.dcmetrotransit.example",
            "agency_timezone": "America/New_York",
            "agency_lang": "en",
        },
        {
            "agency_id": "OC_TRANSIT",
            "agency_name": "Ottawa City Transit",
            "agency_url": "https://www.ottawacitytransit.example",
            "agency_timezone": "America/Toronto",
            "agency_lang": "en",
        },
    ]
    _write_csv(
        output_dir / "agency.txt",
        ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang"],
        agencies,
    )

    # --- stops.txt -----------------------------------------------------------
    all_stops: dict[str, Stop] = {}
    for spec in specs:
        for s in spec.stops:
            all_stops[s.stop_id] = s
    stop_rows = [
        {
            "stop_id": s.stop_id,
            "stop_name": s.stop_name,
            "stop_lat": f"{s.lat:.6f}",
            "stop_lon": f"{s.lon:.6f}",
        }
        for s in all_stops.values()
    ]
    _write_csv(
        output_dir / "stops.txt",
        ["stop_id", "stop_name", "stop_lat", "stop_lon"],
        stop_rows,
    )

    # --- routes.txt ----------------------------------------------------------
    route_rows = [
        {
            "route_id": s.route_id,
            "agency_id": s.agency_id,
            "route_short_name": s.short_name,
            "route_long_name": s.long_name,
            "route_type": str(s.route_type),
        }
        for s in specs
    ]
    _write_csv(
        output_dir / "routes.txt",
        ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
        route_rows,
    )

    # --- calendar.txt --------------------------------------------------------
    calendar_rows = [
        {
            "service_id": "WKDY",
            "monday": "1", "tuesday": "1", "wednesday": "1",
            "thursday": "1", "friday": "1",
            "saturday": "0", "sunday": "0",
            "start_date": FEED_START, "end_date": FEED_END,
        },
        {
            "service_id": "SAT",
            "monday": "0", "tuesday": "0", "wednesday": "0",
            "thursday": "0", "friday": "0",
            "saturday": "1", "sunday": "0",
            "start_date": FEED_START, "end_date": FEED_END,
        },
        {
            "service_id": "SUN",
            "monday": "0", "tuesday": "0", "wednesday": "0",
            "thursday": "0", "friday": "0",
            "saturday": "0", "sunday": "1",
            "start_date": FEED_START, "end_date": FEED_END,
        },
        {
            "service_id": "HOL",
            "monday": "0", "tuesday": "0", "wednesday": "0",
            "thursday": "0", "friday": "0",
            "saturday": "0", "sunday": "0",
            "start_date": FEED_START, "end_date": FEED_END,
        },
    ]
    _write_csv(
        output_dir / "calendar.txt",
        [
            "service_id", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday", "start_date", "end_date",
        ],
        calendar_rows,
    )

    # --- calendar_dates.txt --------------------------------------------------
    cal_date_rows = [
        {"service_id": "HOL", "date": d, "exception_type": "1"}
        for d in HOLIDAY_DATES
    ]
    # Also add WKDY exceptions: remove WKDY service on each holiday.
    cal_date_rows += [
        {"service_id": "WKDY", "date": d, "exception_type": "2"}
        for d in HOLIDAY_DATES
    ]
    _write_csv(
        output_dir / "calendar_dates.txt",
        ["service_id", "date", "exception_type"],
        cal_date_rows,
    )

    # --- shapes.txt ----------------------------------------------------------
    shape_rows = _build_shapes(specs)
    _write_csv(
        output_dir / "shapes.txt",
        ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence", "shape_dist_traveled"],
        shape_rows,
    )

    # --- trips.txt + stop_times.txt ------------------------------------------
    all_trips: list[TripRow] = []
    all_stop_times: list[StopTimeRow] = []
    trip_counter = [0]
    block_counter = [0]

    for spec in specs:
        t, st = _generate_trips_for_route(spec, trip_counter, block_counter)
        all_trips.extend(t)
        all_stop_times.extend(st)

    _write_csv(
        output_dir / "trips.txt",
        ["route_id", "service_id", "trip_id", "shape_id", "direction_id", "block_id", "trip_headsign"],
        [
            {
                "route_id": tr.route_id,
                "service_id": tr.service_id,
                "trip_id": tr.trip_id,
                "shape_id": tr.shape_id,
                "direction_id": str(tr.direction_id),
                "block_id": tr.block_id,
                "trip_headsign": tr.trip_headsign,
            }
            for tr in all_trips
        ],
    )

    _write_csv(
        output_dir / "stop_times.txt",
        ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence", "timepoint"],
        [
            {
                "trip_id": st.trip_id,
                "arrival_time": st.arrival,
                "departure_time": st.departure,
                "stop_id": st.stop_id,
                "stop_sequence": str(st.stop_sequence),
                "timepoint": str(st.timepoint),
            }
            for st in all_stop_times
        ],
    )

    print(
        f"\nDone. {len(all_stops)} stops, {len(all_trips)} trips, "
        f"{len(all_stop_times)} stop-times in {output_dir}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """Parse CLI args and run the generator."""
    parser = argparse.ArgumentParser(description="Generate mock GTFS fixture files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
