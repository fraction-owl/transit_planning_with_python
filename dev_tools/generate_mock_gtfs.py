"""Generate mock GTFS feeds (composed of .txt files) for development and testing.

Produces one self-contained GTFS feed per region (DC and Ottawa by default).
Each feed exercises a range of route archetypes, service profiles, headways,
stop spacings, and block lengths to serve as fixture data for downstream
scripts in this repository (e.g., schedule exporters, stop-spacing flaggers,
skipped-stop flaggers, GTFS-to-shapefile converters).

What it produces
----------------
For each region, a complete GTFS package containing:
    agency.txt, stops.txt, routes.txt, trips.txt, stop_times.txt,
    calendar.txt, calendar_dates.txt, shapes.txt, feed_info.txt

Route archetypes per region (five routes each):
    10  N-S Local         — all-day weekday/Sat/Sun, weekday late-night tail
    20  E-W Express       — weekday peak only, one direction per peak
    30  NW-SE Crosstown   — all-day weekday/Sat/Sun
    40  NE-SW Diagonal    — all-day weekday/Sat/Sun, close stop spacing
    50H Holiday Loop      — holiday only, clockwise; shares east leg with R10

Geographic basis
----------------
Each region's bounding box is taken from a Census shapefile fixture if one is
configured and reachable; otherwise the script falls back to a hardcoded
approximate bbox. Shapes are drawn so they (nearly) hit the edges of the
bounding box, with a small inset so they sit inside the study area.

Inputs:
    - OUTPUT_DIR: Folder where the feeds will be written.
    - DC_FIXTURE_PATH, OTTAWA_FIXTURE_PATH (optional): Paths to any
      geopandas-readable file (.shp, .geojson, .gpkg, ...) used to derive
      the region bounding box. Geopandas is imported lazily — the script
      runs fine without it when no fixture path is configured.

Outputs:
    - Two GTFS sub-folders under OUTPUT_DIR (dc/ and ottawa/), each holding
      the .txt files listed above.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

# Optional fixture paths. Any geopandas-readable file is fine. Leave as the
# placeholder strings to skip fixture lookup and use the hardcoded bboxes.
DC_FIXTURE_PATH: Optional[Path] = None
OTTAWA_FIXTURE_PATH: Optional[Path] = None

# Hardcoded fallback bounding boxes (WGS84): (min_lon, min_lat, max_lon, max_lat).
# Approximate metro envelopes — replace with fixture-derived bboxes when available.
DEFAULT_BBOX_DC = (-77.120, 38.790, -76.910, 39.000)
DEFAULT_BBOX_OTTAWA = (-76.000, 45.300, -75.500, 45.550)

# Inset the drawing area inside each bbox so shapes do not sit on the boundary.
BBOX_INSET_FRACTION = 0.05

# Operator policy: maximum scheduled revenue duration per block, in hours.
# Treated as the default ceiling; tune per agency policy.
MAX_BLOCK_HOURS = 8.0

# Layover (recovery) time inserted between trips chained into the same block.
BLOCK_LAYOVER_MINUTES = 5

# Feed-wide service window (calendar.txt start/end dates) and feed publisher info.
FEED_START_DATE = "20260101"
FEED_END_DATE = "20261231"
FEED_VERSION = "2026.05-mock"
FEED_LANG = "en"

# A small set of representative U.S./Canada holidays falling in the FEED window.
# Used to (a) populate the "holiday" service with active dates, and
# (b) remove those dates from the regular "weekday" service via calendar_dates.txt.
HOLIDAYS = (
    "20260101",  # New Year's Day
    "20261225",  # Christmas Day
)

# Stop-spacing standards exposed by archetype, in feet.
STOP_SPACING_LOCAL_FT = 1000.0
STOP_SPACING_EXPRESS_FT = 2000.0
STOP_SPACING_CLOSE_FT = 500.0
STOP_SPACING_HOLIDAY_FT = 2000.0

# Average revenue speeds, in miles per hour, used to compute trip runtime.
AVG_SPEED_LOCAL_MPH = 14.0
AVG_SPEED_EXPRESS_MPH = 22.0
AVG_SPEED_CROSSTOWN_MPH = 15.0
AVG_SPEED_DIAGONAL_MPH = 13.0
AVG_SPEED_LOOP_MPH = 12.0

# Dwell at each stop (seconds). Added on top of inter-stop run time.
STOP_DWELL_SECONDS = 20

# Spacing of shape_pt rows along each shape leg (meters). Smaller = smoother shape.
SHAPE_POINT_SPACING_M = 100.0

# Earth radius (meters) for great-circle calculations.
EARTH_RADIUS_M = 6_371_000.0

# Unit conversions.
FEET_PER_METER = 3.28084
METERS_PER_MILE = 1609.344

# Logging.
LOG_LEVEL = logging.INFO

# --------------------------------------------------------------------------------------------------
# REGION DEFINITIONS
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One target region for mock feed generation.

    Attributes:
        key: Short identifier used for output folder and id prefixes (e.g., ``"dc"``).
        agency_id: GTFS agency_id.
        agency_name: Human-readable agency name.
        agency_url: Public-facing URL for the (fictional) agency.
        agency_timezone: IANA timezone string.
        fixture_path: Optional path to a shapefile/geopandas-readable file
            used to derive the bounding box. ``None`` falls back to
            ``default_bbox``.
        default_bbox: Fallback bbox as ``(min_lon, min_lat, max_lon, max_lat)``.
    """

    key: str
    agency_id: str
    agency_name: str
    agency_url: str
    agency_timezone: str
    fixture_path: Optional[Path]
    default_bbox: Tuple[float, float, float, float]


REGIONS: Tuple[Region, ...] = (
    Region(
        key="dc",
        agency_id="MOCK_DC",
        agency_name="Mock DC Transit",
        agency_url="https://example.org/mock-dc",
        agency_timezone="America/New_York",
        fixture_path=DC_FIXTURE_PATH,
        default_bbox=DEFAULT_BBOX_DC,
    ),
    Region(
        key="ottawa",
        agency_id="MOCK_OTW",
        agency_name="Mock Ottawa Transit",
        agency_url="https://example.org/mock-ottawa",
        agency_timezone="America/Toronto",
        fixture_path=OTTAWA_FIXTURE_PATH,
        default_bbox=DEFAULT_BBOX_OTTAWA,
    ),
)

# --------------------------------------------------------------------------------------------------
# ROUTE SPECIFICATIONS
# --------------------------------------------------------------------------------------------------

# Each headway window is (start_hhmm, end_hhmm, headway_minutes). Inclusive start,
# exclusive end. End times may exceed "24:00" to express service past midnight.

ServiceWindow = Tuple[str, str, int]


@dataclass(frozen=True)
class RouteSpec:
    """Static description of one mock route's behavior.

    Attributes:
        short_name: Public route designator (e.g., ``"10"``).
        long_name: Public route name.
        shape_kind: Geometry archetype: ``"ns"``, ``"ew"``, ``"nwse"``,
            ``"nesw"``, or ``"loop"``.
        stop_spacing_ft: Target spacing between stops along the shape (feet).
        avg_speed_mph: Average revenue speed for runtime computation.
        windows_by_service: Maps service_id to a list of operating windows.
        direction_pattern: ``"both"`` (alternating 0/1 trips), ``"am_in_pm_out"``
            (peak express convention), or ``"cw_only"`` (loop, direction 0 only).
    """

    short_name: str
    long_name: str
    shape_kind: str
    stop_spacing_ft: float
    avg_speed_mph: float
    windows_by_service: Dict[str, List[ServiceWindow]] = field(default_factory=dict)
    direction_pattern: str = "both"


def _route_specs() -> Tuple[RouteSpec, ...]:
    """Return the canonical five-route lineup applied to every region."""
    return (
        # ------- Route 10: N-S Local — all day + late night ------------------------------------
        RouteSpec(
            short_name="10",
            long_name="North-South Local",
            shape_kind="ns",
            stop_spacing_ft=STOP_SPACING_LOCAL_FT,
            avg_speed_mph=AVG_SPEED_LOCAL_MPH,
            windows_by_service={
                "weekday": [
                    ("05:00", "23:00", 30),
                    ("23:00", "26:00", 60),  # late-night tail past midnight
                ],
                "saturday": [("06:00", "24:00", 30)],
                "sunday": [("07:00", "22:00", 60)],
            },
            direction_pattern="both",
        ),
        # ------- Route 20: E-W Express — peak only, directional ---------------------------------
        RouteSpec(
            short_name="20",
            long_name="East-West Express",
            shape_kind="ew",
            stop_spacing_ft=STOP_SPACING_EXPRESS_FT,
            avg_speed_mph=AVG_SPEED_EXPRESS_MPH,
            windows_by_service={
                "weekday": [
                    ("06:30", "09:00", 5),    # AM peak — direction 0 only
                    ("15:30", "18:30", 5),    # PM peak — direction 1 only
                ],
            },
            direction_pattern="am_in_pm_out",
        ),
        # ------- Route 30: NW-SE Crosstown ------------------------------------------------------
        RouteSpec(
            short_name="30",
            long_name="NW-SE Crosstown",
            shape_kind="nwse",
            stop_spacing_ft=STOP_SPACING_LOCAL_FT,
            avg_speed_mph=AVG_SPEED_CROSSTOWN_MPH,
            windows_by_service={
                "weekday": [("06:00", "22:00", 15)],
                "saturday": [("07:00", "22:00", 30)],
                "sunday": [("07:00", "22:00", 30)],
            },
            direction_pattern="both",
        ),
        # ------- Route 40: NE-SW Diagonal — close stop spacing ----------------------------------
        RouteSpec(
            short_name="40",
            long_name="NE-SW Diagonal",
            shape_kind="nesw",
            stop_spacing_ft=STOP_SPACING_CLOSE_FT,
            avg_speed_mph=AVG_SPEED_DIAGONAL_MPH,
            windows_by_service={
                "weekday": [("05:30", "22:30", 30)],
                "saturday": [("07:00", "22:00", 60)],
                "sunday": [("08:00", "21:00", 60)],
            },
            direction_pattern="both",
        ),
        # ------- Route 50H: Holiday Loop — clockwise only, skipped-stop demo --------------------
        RouteSpec(
            short_name="50H",
            long_name="Holiday Loop",
            shape_kind="loop",
            stop_spacing_ft=STOP_SPACING_HOLIDAY_FT,
            avg_speed_mph=AVG_SPEED_LOOP_MPH,
            windows_by_service={
                "holiday": [("08:00", "20:00", 60)],
            },
            direction_pattern="cw_only",
        ),
    )


# ==================================================================================================
# GEOGRAPHIC HELPERS
# ==================================================================================================


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two WGS84 points in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the initial bearing in degrees from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _destination_point(
    lat: float, lon: float, bearing_deg: float, distance_m: float
) -> Tuple[float, float]:
    """Return the lat/lon reached by traveling ``distance_m`` along ``bearing_deg``."""
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    ang = distance_m / EARTH_RADIUS_M
    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _polyline_length_m(vertices: Sequence[Tuple[float, float]]) -> float:
    """Return cumulative great-circle length of a (lat, lon) polyline."""
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(vertices[:-1], vertices[1:]):
        total += _haversine_m(lat1, lon1, lat2, lon2)
    return total


def _interpolate_along_polyline(
    vertices: Sequence[Tuple[float, float]], distance_m: float
) -> Tuple[float, float]:
    """Return the (lat, lon) at ``distance_m`` along the polyline from its start.

    Tolerates small floating-point drift at the terminus: if the requested
    distance is within 1 mm of the polyline length, returns the last vertex.

    Raises:
        ValueError: If ``distance_m`` exceeds the polyline length by more
            than the floating-point tolerance.
    """
    tolerance_m = 1e-3
    remaining = distance_m
    for (lat1, lon1), (lat2, lon2) in zip(vertices[:-1], vertices[1:]):
        seg_len = _haversine_m(lat1, lon1, lat2, lon2)
        if remaining <= seg_len:
            bearing = _initial_bearing_deg(lat1, lon1, lat2, lon2)
            return _destination_point(lat1, lon1, bearing, max(remaining, 0.0))
        remaining -= seg_len
    if remaining <= tolerance_m:
        return vertices[-1]
    raise ValueError(
        f"distance_m exceeds polyline length by {remaining:.6f} m (>{tolerance_m} m tol)"
    )


def _bbox_with_inset(
    bbox: Tuple[float, float, float, float], inset_fraction: float
) -> Tuple[float, float, float, float]:
    """Inset a bbox by a fractional margin on every side."""
    min_lon, min_lat, max_lon, max_lat = bbox
    dlon = max_lon - min_lon
    dlat = max_lat - min_lat
    return (
        min_lon + dlon * inset_fraction,
        min_lat + dlat * inset_fraction,
        max_lon - dlon * inset_fraction,
        max_lat - dlat * inset_fraction,
    )


def _bbox_from_fixture(fixture_path: Optional[Path]) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(min_lon, min_lat, max_lon, max_lat)`` from a fixture, or None.

    Geopandas is imported lazily so a missing dependency does not break the
    script when no fixture path is configured.
    """
    if fixture_path is None:
        return None
    if not Path(fixture_path).exists():
        logging.warning("Fixture path does not exist; falling back to default bbox: %s",
                        fixture_path)
        return None
    try:
        import geopandas as gpd  # noqa: PLC0415 — lazy import is intentional
    except ImportError:
        logging.warning("geopandas not installed; falling back to default bbox for %s",
                        fixture_path)
        return None
    gdf = gpd.read_file(fixture_path)
    if gdf.empty:
        logging.warning("Fixture is empty; falling back to default bbox: %s", fixture_path)
        return None
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


# ==================================================================================================
# SHAPE / STOP BUILDERS
# ==================================================================================================


def _build_shape_vertices(
    shape_kind: str, bbox: Tuple[float, float, float, float]
) -> List[Tuple[float, float]]:
    """Return a list of (lat, lon) vertices for a shape archetype within a bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0

    if shape_kind == "ns":
        return [(min_lat, mid_lon), (max_lat, mid_lon)]
    if shape_kind == "ew":
        return [(mid_lat, min_lon), (mid_lat, max_lon)]
    if shape_kind == "nwse":
        return [(max_lat, min_lon), (min_lat, max_lon)]
    if shape_kind == "nesw":
        return [(max_lat, max_lon), (min_lat, min_lon)]
    if shape_kind == "loop":
        # The loop's east leg is intentionally aligned with the N-S corridor
        # (at mid_lon) so the two routes share physical stops on that segment.
        # The loop sits on the western half of the bbox, with its north and
        # south extents at 25% / 75% of the bbox latitude span.
        east_lon = mid_lon
        west_lon = mid_lon - 0.25 * (max_lon - min_lon)
        south_lat = min_lat + 0.25 * (max_lat - min_lat)
        north_lat = min_lat + 0.75 * (max_lat - min_lat)
        # SW → NW → NE → SE → SW. The NE→SE leg (vertices 2→3) is the
        # shared east leg; direction-of-travel along the loop is set in
        # _build_trip_stop_visits.
        return [
            (south_lat, west_lon),
            (north_lat, west_lon),
            (north_lat, east_lon),
            (south_lat, east_lon),
            (south_lat, west_lon),
        ]
    raise ValueError(f"Unknown shape_kind: {shape_kind!r}")


def _densify_shape(
    vertices: Sequence[Tuple[float, float]], spacing_m: float
) -> List[Tuple[float, float]]:
    """Insert intermediate points along each leg at roughly ``spacing_m``."""
    if len(vertices) < 2:
        return list(vertices)

    densified: List[Tuple[float, float]] = [vertices[0]]
    for (lat1, lon1), (lat2, lon2) in zip(vertices[:-1], vertices[1:]):
        seg_len = _haversine_m(lat1, lon1, lat2, lon2)
        if seg_len <= spacing_m:
            densified.append((lat2, lon2))
            continue
        bearing = _initial_bearing_deg(lat1, lon1, lat2, lon2)
        steps = int(seg_len // spacing_m)
        for i in range(1, steps + 1):
            d = i * spacing_m
            if d >= seg_len:
                break
            densified.append(_destination_point(lat1, lon1, bearing, d))
        densified.append((lat2, lon2))
    return densified


def _place_stops_along_shape(
    vertices: Sequence[Tuple[float, float]], spacing_ft: float
) -> List[Tuple[float, float, float]]:
    """Place stops at ``spacing_ft`` along the polyline.

    Returns:
        A list of (lat, lon, shape_dist_traveled_ft) tuples. The first stop sits
        at the polyline origin and the last sits at the terminus.
    """
    spacing_m = spacing_ft / FEET_PER_METER
    total_m = _polyline_length_m(vertices)
    if total_m <= 0:
        return []

    n_intervals = max(1, round(total_m / spacing_m))
    actual_spacing_m = total_m / n_intervals

    stops: List[Tuple[float, float, float]] = []
    for i in range(n_intervals + 1):
        d_m = i * actual_spacing_m
        if i == n_intervals:
            d_m = total_m  # clamp to terminus to avoid float drift
        lat, lon = _interpolate_along_polyline(vertices, d_m)
        stops.append((lat, lon, d_m * FEET_PER_METER))
    return stops


# ==================================================================================================
# TIME HELPERS
# ==================================================================================================


def _hhmm_to_seconds(hhmm: str) -> int:
    """Convert ``"HH:MM"`` (allowing >= 24) into seconds since service-day start."""
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got {hhmm!r}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60


def _seconds_to_hhmmss(seconds: int) -> str:
    """Format seconds since service-day start as GTFS ``HH:MM:SS`` (>=24h allowed)."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _round_up_to_5(seconds: int) -> int:
    """Round a second-of-day value up to the next multiple of 5 minutes."""
    block = 5 * 60
    return ((seconds + block - 1) // block) * block


# ==================================================================================================
# CALENDAR / SERVICE
# ==================================================================================================


def _build_calendar_rows() -> List[Dict[str, str]]:
    """Return calendar.txt rows for the four base services."""
    base = {
        "start_date": FEED_START_DATE,
        "end_date": FEED_END_DATE,
    }
    return [
        {
            "service_id": "weekday",
            "monday": "1", "tuesday": "1", "wednesday": "1", "thursday": "1", "friday": "1",
            "saturday": "0", "sunday": "0",
            **base,
        },
        {
            "service_id": "saturday",
            "monday": "0", "tuesday": "0", "wednesday": "0", "thursday": "0", "friday": "0",
            "saturday": "1", "sunday": "0",
            **base,
        },
        {
            "service_id": "sunday",
            "monday": "0", "tuesday": "0", "wednesday": "0", "thursday": "0", "friday": "0",
            "saturday": "0", "sunday": "1",
            **base,
        },
        {
            "service_id": "holiday",
            # No regular days of week; activated only via calendar_dates.txt.
            "monday": "0", "tuesday": "0", "wednesday": "0", "thursday": "0", "friday": "0",
            "saturday": "0", "sunday": "0",
            **base,
        },
    ]


def _build_calendar_dates_rows() -> List[Dict[str, str]]:
    """Return calendar_dates.txt rows turning weekday service off and holiday on."""
    rows: List[Dict[str, str]] = []
    for date in HOLIDAYS:
        rows.append({"service_id": "weekday", "date": date, "exception_type": "2"})
        rows.append({"service_id": "holiday", "date": date, "exception_type": "1"})
    return rows


# ==================================================================================================
# TRIP / STOP_TIMES BUILDERS
# ==================================================================================================


@dataclass
class TripPlan:
    """One scheduled trip, prior to writing into trips.txt / stop_times.txt."""

    trip_id: str
    route_id: str
    service_id: str
    shape_id: str
    direction_id: int
    block_id: str
    headsign: str
    start_seconds: int
    end_seconds: int
    stop_visits: List[Tuple[str, int, float]]  # (stop_id, arrival_seconds, shape_dist_ft)


def _route_runtime_seconds(spec: RouteSpec, shape_length_m: float) -> int:
    """Estimate scheduled runtime for one trip, in seconds, incl. stop dwell."""
    miles = shape_length_m / METERS_PER_MILE
    travel_hours = miles / spec.avg_speed_mph
    travel_seconds = travel_hours * 3600.0
    # Approximate dwell: STOP_DWELL_SECONDS per stop, ignoring first stop.
    n_intervals = max(1, round(shape_length_m / (spec.stop_spacing_ft / FEET_PER_METER)))
    dwell_seconds = STOP_DWELL_SECONDS * n_intervals
    return int(round(travel_seconds + dwell_seconds))


def _allowed_directions_for_window(
    spec: RouteSpec, window_index: int
) -> Tuple[int, ...]:
    """Return which direction_ids run during a given window of a route."""
    if spec.direction_pattern == "cw_only":
        return (0,)
    if spec.direction_pattern == "am_in_pm_out":
        # By spec ordering: window 0 = AM peak (dir 0), window 1 = PM peak (dir 1).
        return (window_index % 2,)
    return (0, 1)


def _build_trip_stop_visits(
    spec: RouteSpec,
    stops: Sequence[Tuple[str, float, float, float]],
    start_seconds: int,
    direction_id: int,
    apply_skip: bool,
) -> Tuple[List[Tuple[str, int, float]], int]:
    """Return (stop_visits, end_seconds) for one trip.

    Args:
        spec: Route spec.
        stops: List of ``(stop_id, lat, lon, shape_dist_ft)`` in shape order.
        start_seconds: Departure time at the first stop.
        direction_id: 0 = shape-order, 1 = reverse shape-order.
        apply_skip: If True, omit exactly one shared east-leg stop from
            the visit list (skipped-stop demo). A stop is considered
            "shared east-leg" if its stop_id matches the N-S corridor
            prefix used by the N-S route.

    Returns:
        A tuple (stop_visits, end_seconds), where stop_visits is
        ``(stop_id, arrival_seconds, shape_dist_ft)`` in trip order.
    """
    ordered = list(stops) if direction_id == 0 else list(reversed(stops))
    spacing_m = spec.stop_spacing_ft / FEET_PER_METER
    seg_seconds = (spacing_m / METERS_PER_MILE) / spec.avg_speed_mph * 3600.0

    skip_idx: Optional[int] = None
    if apply_skip:
        shared_indices = [
            i for i, (sid, *_rest) in enumerate(ordered) if "_NS_" in sid
        ]
        if shared_indices:
            skip_idx = shared_indices[len(shared_indices) // 2]

    visits: List[Tuple[str, int, float]] = []
    cursor = float(start_seconds)
    for idx, (stop_id, _lat, _lon, dist_ft) in enumerate(ordered):
        if idx > 0:
            cursor += seg_seconds + STOP_DWELL_SECONDS
        if idx == skip_idx:
            continue
        visits.append((stop_id, int(round(cursor)), dist_ft))
    end_seconds = visits[-1][1] if visits else start_seconds
    return visits, end_seconds


def _materialize_trips(
    region_key: str,
    spec: RouteSpec,
    stops: Sequence[Tuple[str, float, float, float]],
    shape_length_m: float,
) -> List[TripPlan]:
    """Materialize all concrete trips for a route across every service it runs."""
    route_id = f"{region_key.upper()}_R{spec.short_name}"
    shape_id = f"{route_id}_shp"
    trips: List[TripPlan] = []

    for service_id, windows in spec.windows_by_service.items():
        per_service_seq = 0
        for w_idx, (start_hhmm, end_hhmm, headway_min) in enumerate(windows):
            start_s = _round_up_to_5(_hhmm_to_seconds(start_hhmm))
            end_s = _hhmm_to_seconds(end_hhmm)
            allowed_dirs = _allowed_directions_for_window(spec, w_idx)
            headway_s = headway_min * 60
            t = start_s
            dir_rotation = 0
            while t < end_s:
                direction_id = allowed_dirs[dir_rotation % len(allowed_dirs)]
                trip_id = f"{route_id}_{service_id}_{per_service_seq:04d}"
                visits, end_seconds = _build_trip_stop_visits(
                    spec=spec,
                    stops=stops,
                    start_seconds=t,
                    direction_id=direction_id,
                    apply_skip=spec.shape_kind == "loop",
                )
                headsign = _headsign_for(spec, direction_id)
                trips.append(
                    TripPlan(
                        trip_id=trip_id,
                        route_id=route_id,
                        service_id=service_id,
                        shape_id=shape_id,
                        direction_id=direction_id,
                        block_id="",  # filled in by _assign_blocks
                        headsign=headsign,
                        start_seconds=t,
                        end_seconds=end_seconds,
                        stop_visits=visits,
                    )
                )
                per_service_seq += 1
                dir_rotation += 1
                t += headway_s

    _ = shape_length_m  # currently informational only
    return trips


def _headsign_for(spec: RouteSpec, direction_id: int) -> str:
    """Return a simple, generic trip headsign per archetype and direction."""
    mapping = {
        "ns": ("Northbound", "Southbound"),
        "ew": ("Eastbound", "Westbound"),
        "nwse": ("Southeast", "Northwest"),
        "nesw": ("Southwest", "Northeast"),
        "loop": ("Loop", "Loop"),
    }
    pair = mapping.get(spec.shape_kind, ("Outbound", "Inbound"))
    return pair[direction_id]


# ==================================================================================================
# BLOCK ASSIGNMENT
# ==================================================================================================


def _assign_blocks(trips: List[TripPlan]) -> None:
    """Assign block_ids in place, chaining trips on the same route+service.

    Algorithm (interval scheduling):
        - Group by (route_id, service_id) — a vehicle stays on its route.
        - Within each group, walk trips in start-time order.
        - Maintain a pool of currently-open blocks (each with its first
          trip's start time and its most recent trip's end time).
        - For each new trip, pick the open block whose most recent trip
          ended earliest, provided that block can still accept this trip
          while honoring the layover and the MAX_BLOCK_HOURS cap.
        - Otherwise, open a new block for the trip.

    This produces realistic block chaining: a single bus runs trip → layover →
    next available trip in the same service, accumulating revenue hours up
    to the policy cap.
    """
    max_block_seconds = int(MAX_BLOCK_HOURS * 3600)
    layover_seconds = BLOCK_LAYOVER_MINUTES * 60

    groups: Dict[Tuple[str, str], List[TripPlan]] = {}
    for trip in trips:
        groups.setdefault((trip.route_id, trip.service_id), []).append(trip)

    for (route_id, service_id), group in groups.items():
        group.sort(key=lambda t: t.start_seconds)

        # Open blocks for this group: each entry is [block_id, start_s, end_s].
        open_blocks: List[List] = []
        next_block_idx = 1

        for trip in group:
            # Find the open block with the earliest end time that can still
            # accept this trip without violating layover or duration limits.
            best_idx: Optional[int] = None
            best_end = math.inf
            for i, (bid, bstart, bend) in enumerate(open_blocks):
                if trip.start_seconds < bend + layover_seconds:
                    continue
                if trip.end_seconds - bstart > max_block_seconds:
                    continue
                if bend < best_end:
                    best_end = bend
                    best_idx = i

            if best_idx is not None:
                bid, bstart, _bend = open_blocks[best_idx]
                open_blocks[best_idx][2] = trip.end_seconds
                trip.block_id = bid
            else:
                bid = f"{route_id}_{service_id}_BLK{next_block_idx:03d}"
                next_block_idx += 1
                open_blocks.append([bid, trip.start_seconds, trip.end_seconds])
                trip.block_id = bid


# ==================================================================================================
# WRITERS
# ==================================================================================================


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    """Write a GTFS .txt (CSV) file with explicit headers and UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_agency(region: Region, out_dir: Path) -> None:
    """Write agency.txt for one region."""
    _write_csv(
        out_dir / "agency.txt",
        ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang"],
        [{
            "agency_id": region.agency_id,
            "agency_name": region.agency_name,
            "agency_url": region.agency_url,
            "agency_timezone": region.agency_timezone,
            "agency_lang": FEED_LANG,
        }],
    )


def _write_calendar(out_dir: Path) -> None:
    """Write calendar.txt and calendar_dates.txt."""
    _write_csv(
        out_dir / "calendar.txt",
        ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
         "saturday", "sunday", "start_date", "end_date"],
        _build_calendar_rows(),
    )
    _write_csv(
        out_dir / "calendar_dates.txt",
        ["service_id", "date", "exception_type"],
        _build_calendar_dates_rows(),
    )


def _write_feed_info(region: Region, out_dir: Path) -> None:
    """Write feed_info.txt."""
    _write_csv(
        out_dir / "feed_info.txt",
        ["feed_publisher_name", "feed_publisher_url", "feed_lang",
         "feed_start_date", "feed_end_date", "feed_version"],
        [{
            "feed_publisher_name": region.agency_name,
            "feed_publisher_url": region.agency_url,
            "feed_lang": FEED_LANG,
            "feed_start_date": FEED_START_DATE,
            "feed_end_date": FEED_END_DATE,
            "feed_version": FEED_VERSION,
        }],
    )


def _write_routes(region: Region, specs: Sequence[RouteSpec], out_dir: Path) -> None:
    """Write routes.txt."""
    rows = []
    for spec in specs:
        rows.append({
            "route_id": f"{region.key.upper()}_R{spec.short_name}",
            "agency_id": region.agency_id,
            "route_short_name": spec.short_name,
            "route_long_name": spec.long_name,
            "route_type": "3",  # bus
        })
    _write_csv(
        out_dir / "routes.txt",
        ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
        rows,
    )


def _write_stops(
    stops_by_route: Dict[str, List[Tuple[str, float, float, float]]],
    region_key: str,
    out_dir: Path,
) -> None:
    """Write stops.txt, deduplicating shared stops across routes."""
    seen: Dict[str, Tuple[float, float]] = {}
    rows: List[Dict[str, str]] = []
    for stops in stops_by_route.values():
        for stop_id, lat, lon, _dist in stops:
            if stop_id in seen:
                continue
            seen[stop_id] = (lat, lon)
            rows.append({
                "stop_id": stop_id,
                "stop_code": stop_id,
                "stop_name": _stop_name_from_id(stop_id, region_key),
                "stop_lat": f"{lat:.6f}",
                "stop_lon": f"{lon:.6f}",
            })
    rows.sort(key=lambda r: r["stop_id"])
    _write_csv(
        out_dir / "stops.txt",
        ["stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon"],
        rows,
    )


def _stop_name_from_id(stop_id: str, region_key: str) -> str:
    """Build a human-readable stop name from a stop_id."""
    return f"{region_key.upper()} Stop {stop_id.split('_')[-1]}"


def _write_shapes(
    shape_vertices_by_id: Dict[str, List[Tuple[float, float]]], out_dir: Path
) -> None:
    """Write shapes.txt, densifying each shape to SHAPE_POINT_SPACING_M."""
    rows: List[Dict[str, str]] = []
    for shape_id, vertices in shape_vertices_by_id.items():
        densified = _densify_shape(vertices, SHAPE_POINT_SPACING_M)
        cum_m = 0.0
        prev: Optional[Tuple[float, float]] = None
        for seq, (lat, lon) in enumerate(densified, start=1):
            if prev is not None:
                cum_m += _haversine_m(prev[0], prev[1], lat, lon)
            rows.append({
                "shape_id": shape_id,
                "shape_pt_lat": f"{lat:.6f}",
                "shape_pt_lon": f"{lon:.6f}",
                "shape_pt_sequence": str(seq),
                "shape_dist_traveled": f"{cum_m * FEET_PER_METER:.2f}",
            })
            prev = (lat, lon)
    _write_csv(
        out_dir / "shapes.txt",
        ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence",
         "shape_dist_traveled"],
        rows,
    )


def _write_trips_and_stop_times(trips: Sequence[TripPlan], out_dir: Path) -> None:
    """Write trips.txt and stop_times.txt."""
    trip_rows = [
        {
            "route_id": t.route_id,
            "service_id": t.service_id,
            "trip_id": t.trip_id,
            "trip_headsign": t.headsign,
            "direction_id": str(t.direction_id),
            "block_id": t.block_id,
            "shape_id": t.shape_id,
        }
        for t in trips
    ]
    _write_csv(
        out_dir / "trips.txt",
        ["route_id", "service_id", "trip_id", "trip_headsign",
         "direction_id", "block_id", "shape_id"],
        trip_rows,
    )

    st_rows: List[Dict[str, str]] = []
    for t in trips:
        for seq, (stop_id, arr_s, dist_ft) in enumerate(t.stop_visits, start=1):
            hhmmss = _seconds_to_hhmmss(arr_s)
            st_rows.append({
                "trip_id": t.trip_id,
                "arrival_time": hhmmss,
                "departure_time": hhmmss,
                "stop_id": stop_id,
                "stop_sequence": str(seq),
                "shape_dist_traveled": f"{dist_ft:.2f}",
                "timepoint": "1" if seq == 1 or seq == len(t.stop_visits) else "0",
            })
    _write_csv(
        out_dir / "stop_times.txt",
        ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
         "shape_dist_traveled", "timepoint"],
        st_rows,
    )


# ==================================================================================================
# REGION ASSEMBLY
# ==================================================================================================


def _build_region_feed(region: Region, out_dir: Path) -> None:
    """Build and write a complete GTFS feed for one region."""
    logging.info("Building feed for region: %s", region.key)

    bbox = _bbox_from_fixture(region.fixture_path) or region.default_bbox
    drawing_bbox = _bbox_with_inset(bbox, BBOX_INSET_FRACTION)
    logging.info("  bbox: %s", tuple(f"{v:.5f}" for v in drawing_bbox))

    region_dir = out_dir / region.key
    region_dir.mkdir(parents=True, exist_ok=True)

    specs = _route_specs()

    # Build the shared N-S stop sequence first so the holiday loop can reuse it
    # on its east leg (skipped-stop demo and route overlap).
    ns_spec = next(s for s in specs if s.shape_kind == "ns")
    ns_vertices = _build_shape_vertices("ns", drawing_bbox)
    ns_stops_raw = _place_stops_along_shape(ns_vertices, ns_spec.stop_spacing_ft)
    ns_stops = [
        (f"{region.key.upper()}_NS_{i:03d}", lat, lon, dist)
        for i, (lat, lon, dist) in enumerate(ns_stops_raw)
    ]

    # Build per-route shape vertices and stops.
    stops_by_route: Dict[str, List[Tuple[str, float, float, float]]] = {}
    shape_vertices_by_id: Dict[str, List[Tuple[float, float]]] = {}
    all_trips: List[TripPlan] = []

    for spec in specs:
        route_id = f"{region.key.upper()}_R{spec.short_name}"
        shape_id = f"{route_id}_shp"

        if spec.shape_kind == "ns":
            stops = ns_stops
            vertices = ns_vertices
        elif spec.shape_kind == "loop":
            vertices = _build_shape_vertices("loop", drawing_bbox)
            stops = _build_loop_stops_with_shared_east_leg(
                vertices=vertices,
                spec=spec,
                shared_ns_stops=ns_stops,
                region_key=region.key,
                ns_vertices=ns_vertices,
            )
        else:
            vertices = _build_shape_vertices(spec.shape_kind, drawing_bbox)
            stops_raw = _place_stops_along_shape(vertices, spec.stop_spacing_ft)
            stops = [
                (f"{region.key.upper()}_R{spec.short_name}_{i:03d}", lat, lon, dist)
                for i, (lat, lon, dist) in enumerate(stops_raw)
            ]

        stops_by_route[route_id] = stops
        shape_vertices_by_id[shape_id] = vertices

        shape_length_m = _polyline_length_m(vertices)
        trips = _materialize_trips(region.key, spec, stops, shape_length_m)
        all_trips.extend(trips)

    _assign_blocks(all_trips)

    # Write all GTFS tables.
    _write_agency(region, region_dir)
    _write_calendar(region_dir)
    _write_feed_info(region, region_dir)
    _write_routes(region, specs, region_dir)
    _write_stops(stops_by_route, region.key, region_dir)
    _write_shapes(shape_vertices_by_id, region_dir)
    _write_trips_and_stop_times(all_trips, region_dir)

    logging.info("  wrote %d trips across %d routes to %s",
                 len(all_trips), len(specs), region_dir)


def _build_loop_stops_with_shared_east_leg(
    vertices: Sequence[Tuple[float, float]],
    spec: RouteSpec,
    shared_ns_stops: Sequence[Tuple[str, float, float, float]],
    region_key: str,
    ns_vertices: Sequence[Tuple[float, float]],
) -> List[Tuple[str, float, float, float]]:
    """Build the loop's stop sequence leg by leg.

    Layout (vertices from _build_shape_vertices('loop', ...)):
        Leg 0: SW → NW   (west side, going north)
        Leg 1: NW → NE   (north side, going east)
        Leg 2: NE → SE   (east side, going south) — shared with N-S corridor
        Leg 3: SE → SW   (south side, going west)

    On the shared east leg the loop adopts the N-S route's stop ids and
    locations directly (so the two routes share physical stops). On the
    other three legs the loop places its own stops at its nominal spacing.
    The first stop of each new leg duplicates the previous leg's last
    stop and is dropped to keep the sequence clean.

    The caller passes spec.skip_stop_offset as the number of east-leg
    stops to skip past before omitting one (skipped-stop demo). The
    omission is applied later in _build_trip_stop_visits.
    """
    spacing_m = spec.stop_spacing_ft / FEET_PER_METER
    legs_out: List[List[Tuple[str, float, float, float]]] = []

    for leg_idx in range(4):
        v_start = vertices[leg_idx]
        v_end = vertices[leg_idx + 1]
        if leg_idx == 2:
            # Shared east leg — borrow N-S stops whose latitude falls inside
            # the leg's lat span, ordered to match direction of travel
            # (NE → SE means north-to-south, descending latitude).
            lat_min = min(v_start[0], v_end[0])
            lat_max = max(v_start[0], v_end[0])
            east_stops = [
                (sid, lat, lon)
                for (sid, lat, lon, _d) in shared_ns_stops
                if lat_min <= lat <= lat_max
            ]
            east_stops.sort(key=lambda s: -s[1])
            legs_out.append([(sid, lat, lon, 0.0) for (sid, lat, lon) in east_stops])
        else:
            seg_len = _haversine_m(v_start[0], v_start[1], v_end[0], v_end[1])
            n_intervals = max(1, round(seg_len / spacing_m))
            actual_spacing_m = seg_len / n_intervals
            bearing = _initial_bearing_deg(v_start[0], v_start[1], v_end[0], v_end[1])
            leg_stops: List[Tuple[str, float, float, float]] = []
            for i in range(n_intervals + 1):
                lat, lon = _destination_point(v_start[0], v_start[1], bearing,
                                              min(i * actual_spacing_m, seg_len))
                sid = f"{region_key.upper()}_R{spec.short_name}_L{leg_idx}_{i:03d}"
                leg_stops.append((sid, lat, lon, 0.0))
            legs_out.append(leg_stops)

    # Concatenate, dropping the first stop of each subsequent leg when it
    # sits on top of the previous leg's last stop. Then compute cumulative
    # shape_dist_traveled in feet.
    out: List[Tuple[str, float, float, float]] = []
    for leg in legs_out:
        if not leg:
            continue
        start = 1 if out and _haversine_m(out[-1][1], out[-1][2], leg[0][1], leg[0][2]) < 30 else 0
        out.extend(leg[start:])

    cum_m = 0.0
    finished: List[Tuple[str, float, float, float]] = []
    prev: Optional[Tuple[float, float]] = None
    for (sid, lat, lon, _) in out:
        if prev is not None:
            cum_m += _haversine_m(prev[0], prev[1], lat, lon)
        finished.append((sid, lat, lon, cum_m * FEET_PER_METER))
        prev = (lat, lon)

    _ = ns_vertices  # informational; not needed after the snap is structural
    return finished


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> None:
    """Generate one mock GTFS feed per configured region."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")

    logging.info("====================================================")
    logging.info("Mock GTFS Generator")
    logging.info("Output folder: %s", OUTPUT_DIR)
    logging.info("====================================================")

    if str(OUTPUT_DIR).startswith("Path") or OUTPUT_DIR == Path("Path\\To\\Your\\Output_Folder"):
        logging.warning(
            "OUTPUT_DIR is still the placeholder. Edit the CONFIGURATION block "
            "at the top of this script to point at a real folder, then re-run."
        )
        return

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for region in REGIONS:
            _build_region_feed(region, OUTPUT_DIR)
        logging.info("Script completed successfully.")
    except (OSError, ValueError, RuntimeError) as err:
        logging.error("%s", err)
    except Exception as err:  # noqa: BLE001 — last-resort catch for unforeseen issues
        logging.exception("Unexpected error: %s", err)


if __name__ == "__main__":
    main()
