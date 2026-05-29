"""Generate a mock road-centerline network to accompany the mock GTFS feeds.

This is the geometric backbone for the geotools fixtures. It produces a clean,
named, topologically-noded road network per region whose corridors the mock
GTFS shapes will trace (built in a later step). Building the road graph *first*
makes two fixture requirements structural rather than coincidental:

    1. Every GTFS stop sits on a road centerline, so stop names can be derived
       from (and tested against) the adjacent road name — see
       ``stop_vs_roadname_checker_{gpd,arcpy}.py``.
    2. The centerlines form a clean network (segments split at every
       intersection, shared nodes coincide exactly, every road is a connected
       polyline), so the network/turn-clearance tools have well-formed input —
       see ``audit_turn_clearance.py`` and ``stop_removal_impact_gpd.py``.

Topology
--------
A single normalized grid (the unit square) is shared by every region; only the
names and the geographic extent change. The grid is four named arterials (the
GTFS route corridors) plus four named cross streets:

        C1      A1      C2
         |       |       |
    -----+-------+-------+----- C4   (v = 0.75)
         |       |       |
    -----+-------+-------+----- A2   (v = 0.50)   E-W arterial
         |       |       |
    -----+-------+-------+----- C3   (v = 0.25)
         |       |       |
       (u=.25) (u=.50) (u=.75)

    A1  u=0.50 vertical    N-S arterial      (route 10; loop east leg)
    A2  v=0.50 horizontal  E-W arterial      (route 20)
    A3  NW->SE diagonal     u + v = 1         (route 30)
    A4  NE->SW diagonal     u = v             (route 40)
    C1..C4  cross streets; the holiday loop (route 50H) is a clockwise path
            over C1 (west), C4 (north), A1 (east, shared with route 10), and
            C3 (south) -- so the loop needs no geometry of its own.

The two diagonals pass exactly through the grid centre and through the loop's
NW and SW corners, producing high-degree nodes on purpose: good stress cases
for the splitter and for the typo checker's stop-to-road spatial join.

Outputs
-------
For each region, the centerline network is written as BOTH a shapefile and a
GeoPackage layer (government users typically arrive with shapefiles; GPKG is the
cleaner single-file form). Attribute schema matches what the typo checker
expects: ``RW_PREFIX, RW_TYPE_US, RW_SUFFIX, RW_SUFFIX_, FULLNAME`` plus
``road_key`` and ``seg_id`` for traceability.

NOTE: This module is the network core. GTFS-shape-from-graph, stop placement +
naming (with seeded typos), and optional road-asphalt polygons are added in
subsequent steps and will share this module's frame and topology.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
from shapely.ops import linemerge, substring, unary_union

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

# Region extents (WGS84): (min_lon, min_lat, max_lon, max_lat). Kept in sync with
# the mock GTFS generator so the road network and the GTFS shapes share a frame.
DEFAULT_BBOX_DC = (-77.120, 38.790, -76.910, 39.000)
DEFAULT_BBOX_OTTAWA = (-76.000, 45.300, -75.500, 45.550)

# Inset the drawing area inside each bbox so geometry does not sit on the boundary.
BBOX_INSET_FRACTION = 0.05

# Earth model + unit conversions (matched to the GTFS generator's spherical model
# so a "foot" means the same thing in both).
EARTH_RADIUS_M = 6_371_000.0
FEET_PER_METER = 3.28084

# Coordinate snapping grid (decimal places, in planar feet). Rounding both roads'
# endpoints to the same grid guarantees bit-identical shared nodes at junctions.
COORD_DECIMALS_FT = 3

# Topology guards.
MIN_SEG_FT = 10.0  # reject slivers shorter than this

# --- Asphalt (road-surface) polygons -------------------------------------------------------------
# Per road-class half-width (feet) used to buffer centerlines into road-surface
# polygons. Consumed by scripts/data_quality/stop_v_conflict_checker_gpd.py, which
# flags GTFS stops that INTERSECT the road surface (i.e., sit in the roadbed).
ASPHALT_HALF_WIDTH_FT: Dict[str, float] = {
    "PRI": 24.0,  # primary arterial: ~4 lanes + median
    "SEC": 18.0,  # secondary: ~3 lanes
    "LOC": 12.0,  # local: ~2 lanes
}

# --- Stop decoration -----------------------------------------------------------------------------
# Normal stops are nudged this far off the centerline (feet), placing them on the
# "sidewalk": clear of the widest asphalt half-width (so no false conflict) yet
# well within the roadname checker's 50 ft stop buffer (so the name still matches
# the adjacent road). Conflict-positive stops use a zero offset (left in the roadbed).
STOP_OFFSET_FT = 35.0

# Fraction of stops deliberately seeded as known positives, selected deterministically
# by stop_id so the manifest is stable across runs. Documented in the run manifest.
TYPO_RATE = 0.04  # ~4% of stops get a single-character typo in the road portion of the name
CONFLICT_RATE = 0.05  # ~5% of stops are left in the roadbed (intersect the asphalt)

# Cross-street naming: a stop within this distance (feet) of an intersection is named
# "ON_ROAD @ CROSS_ROAD"; otherwise "ON_ROAD" alone (mid-block).
CROSS_STREET_NEAR_FT = 250.0

# Output CRS for all fixtures. CRS handling is intentionally unchanged: fixtures
# are emitted in WGS84. The provided Fairfax sample (EPSG:2283, US survey feet)
# was used to match the DBF attribute *schema*, not to set the output projection;
# the fixture-reading path elsewhere already reprojects 2283 -> 4326 correctly.
OUTPUT_CRS = "EPSG:4326"

# Fairfax-schema metadata reproduced on every feature (constant in the source).
SCHEMA_STATUS = "B"  # "built" / active segment
SCHEMA_CREATOR = "MockGenerator"
SCHEMA_EDITOR = "MockGenerator"
SCHEMA_TIMESTAMP = "2026-05-03T00:00:00Z"  # ISO-8601 w/ Z, as in source CreationDa/EditDate
SCHEMA_EFF_DATE = "2007-09-12T00:00:00Z"

# Deterministic ID bases (mirroring the magnitude of the real fields:
# OBJECTID ~9-digit, TRANS_ID 5000xxxxx, SEGID ~6-digit). Per-region offset keeps
# ids unique across regions.
ID_BASE_OBJECTID = 150_000_000
ID_BASE_TRANS_ID = 500_000_000
ID_BASE_SEGID = 300_000
ID_REGION_STRIDE = 10_000

LOG_LEVEL = logging.INFO

# --------------------------------------------------------------------------------------------------
# ROAD MODEL
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoadDef:
    """One named road, as a normalized polyline in the unit square.

    Attribute names and semantics mirror the Fairfax County
    ``Roadway_Centerlines`` schema. In that schema the *bare* road name lives in
    ``RW_NAME`` (e.g. ``"HUNTINGTON"``), the type in ``RW_TYPE_US`` (e.g.
    ``"ST"``), and ``FULLNAME`` is the uppercase space-join of the non-empty
    ``[prefix, name, type, suffix]`` parts (e.g. ``"N HUNTINGTON ST"``). Values
    are uppercase, matching the source data.

    Attributes:
        key: Short topology identifier shared across regions (e.g., ``"A1"``).
        prefix: ``RW_PREFIX`` (directional prefix, e.g., ``"N"``; ``""`` if none).
        name: ``RW_NAME`` (bare road name, no type/prefix/suffix).
        type_us: ``RW_TYPE_US`` (street type, e.g., ``"ST"``).
        suffix: ``RW_SUFFIX`` (directional suffix, e.g., ``"SE"``; ``""`` if none).
        suffix2: ``RW_SUFFIX_`` (additional suffix; all-null in the source data,
            kept present and empty for schema fidelity).
        road_class: ``ROAD_CLASS`` — ``"PRI"`` | ``"SEC"`` | ``"LOC"``.
        cfcc: ``CFCC`` Census Feature Class Code (e.g. ``"A21"`` primary,
            ``"A31"`` secondary, ``"A41"`` local).
        speed_limit: ``SPEED_LIMI`` posted speed (mph).
        route_alias: ``ROUTE_ALIA`` route alias string (``""`` if none).
        norm_vertices: ``(u, v)`` vertices in ``[0, 1]``; ``u`` west->east,
            ``v`` south->north.
        is_route_corridor: True if a GTFS route rides this road (documentation
            only at this stage; consumed by the GTFS-shape step).
    """

    key: str
    prefix: str
    name: str
    type_us: str
    suffix: str
    suffix2: str
    road_class: str
    cfcc: str
    speed_limit: int
    route_alias: str
    norm_vertices: Tuple[Tuple[float, float], ...]
    is_route_corridor: bool

    @property
    def full_name(self) -> str:
        """Compose ``FULLNAME`` the Fairfax way: uppercase ``[PFX] NAME TYPE [SFX]``."""
        parts = [self.prefix, self.name, self.type_us, self.suffix]
        return " ".join(p for p in (s.strip() for s in parts) if p).upper()


# Shared normalized topology: (u, v) endpoints of each straight road in the unit square.
_TOPOLOGY: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "A1": ((0.50, 0.00), (0.50, 1.00)),  # N-S arterial      (route 10 + loop east leg)
    "A2": ((0.00, 0.50), (1.00, 0.50)),  # E-W arterial      (route 20)
    "A3": ((0.00, 1.00), (1.00, 0.00)),  # NW->SE diagonal   (route 30)
    "A4": ((1.00, 1.00), (0.00, 0.00)),  # NE->SW diagonal   (route 40)
    "C1": ((0.25, 0.00), (0.25, 1.00)),  # cross street      (loop west leg)
    "C2": ((0.75, 0.00), (0.75, 1.00)),  # cross street
    "C3": ((0.00, 0.25), (1.00, 0.25)),  # cross street      (loop south leg)
    "C4": ((0.00, 0.75), (1.00, 0.75)),  # cross street      (loop north leg)
}

_ROUTE_CORRIDOR_KEYS = {"A1", "A2", "A3", "A4"}

# Holiday-loop traversal as (road_key, from_param, to_param), where param is the
# road's own normalized position (0 at first vertex -> 1 at last). Clockwise:
# SW -> NW -> NE -> SE -> SW. Documentation for the GTFS-shape step; not consumed here.
LOOP_CORRIDOR: Tuple[Tuple[str, float, float], ...] = (
    ("C1", 0.25, 0.75),  # west leg, south->north
    ("C4", 0.25, 0.50),  # north leg, west->east
    ("A1", 0.75, 0.25),  # east leg, north->south  (shared with route 10)
    ("C3", 0.50, 0.25),  # south leg, east->west
)

ROUTE_CORRIDORS: Dict[str, Tuple[Tuple[str, float, float], ...]] = {
    "10": (("A1", 0.0, 1.0),),
    "20": (("A2", 0.0, 1.0),),
    "30": (("A3", 0.0, 1.0),),
    "40": (("A4", 0.0, 1.0),),
    "50H": LOOP_CORRIDOR,
}


def route_corridor_keys_for_region(region_key: str) -> Dict[str, List[str]]:
    """Map GTFS route_id (``{REGION}_R{short}``) -> the road keys that route rides.

    Consumed by :func:`decorate_stops` to disambiguate which road each stop sits
    on. Mirrors the GTFS generator's route_id convention.
    """
    out: Dict[str, List[str]] = {}
    for short, corridor in ROUTE_CORRIDORS.items():
        route_id = f"{region_key.upper()}_R{short}"
        out[route_id] = sorted({leg[0] for leg in corridor})
    return out


def corridor_legs_wgs(region_key: str, route_short: str) -> List[List[Tuple[float, float]]]:
    """Trace a route's corridor as a list of per-leg densified WGS84 polylines.

    Each leg corresponds to one entry in ``ROUTE_CORRIDORS[route_short]`` (one road
    the route rides), as an ordered list of ``(lat, lon)`` densified at
    :data:`SHAPE_DENSIFY_FT`. Straight routes return a single leg; the loop returns
    one leg per side. The GTFS loop builder consumes these so its shared-east-leg
    reuse operates on the same centerline geometry as everything else.
    """
    roads = {r.key: r for r in _region_roads(region_key)}
    inset = _bbox_with_inset(REGION_BBOXES[region_key], BBOX_INSET_FRACTION)
    _w, _h, norm_to_planar, planar_to_wgs, _wgs_to_planar = _make_frame(inset)
    planar_lines = {
        k: LineString([norm_to_planar(u, v) for (u, v) in r.norm_vertices])
        for k, r in roads.items()
    }

    legs: List[List[Tuple[float, float]]] = []
    for key, p_from, p_to in ROUTE_CORRIDORS[route_short]:
        line = planar_lines[key]
        a = line.interpolate(p_from * line.length)
        b = line.interpolate(p_to * line.length)
        dense = _densify_planar_polyline([(a.x, a.y), (b.x, b.y)], SHAPE_DENSIFY_FT)
        legs.append([(lat, lon) for (lon, lat) in (planar_to_wgs(x, y) for (x, y) in dense)])
    return legs


def corridor_polyline_wgs(region_key: str, route_short: str) -> List[Tuple[float, float]]:
    """Trace a route's corridor along the centerlines into one densified WGS84 shape.

    Concatenates :func:`corridor_legs_wgs`, dropping the duplicate junction point
    shared between consecutive legs. The GTFS generator uses this as the route's
    shape vertices (replacing the standalone ``_build_shape_vertices`` archetypes);
    because the shape *is* the centerline, stops placed along it sit on the
    centerline by construction.
    """
    out: List[Tuple[float, float]] = []
    for leg in corridor_legs_wgs(region_key, route_short):
        if out and math.dist(out[-1], leg[0]) < 1e-9:
            out.extend(leg[1:])
        else:
            out.extend(leg)
    return out


# Shared per-key classification (topology is shared across regions, so road tier
# is too). Arterials A1/A2 are primary, the diagonals A3/A4 secondary, cross
# streets local. Values mirror the source schema's CFCC / ROAD_CLASS / speed.
_CLASS_TABLE: Dict[str, Tuple[str, str, int]] = {
    # key: (road_class, cfcc, speed_limit_mph)
    "A1": ("PRI", "A21", 35),
    "A2": ("PRI", "A21", 35),
    "A3": ("SEC", "A31", 30),
    "A4": ("SEC", "A31", 30),
    "C1": ("LOC", "A41", 25),
    "C2": ("LOC", "A41", 25),
    "C3": ("LOC", "A41", 25),
    "C4": ("LOC", "A41", 25),
}

# Per-region name tables, keyed by topology id. Each value is the Fairfax-style
# name decomposition: (prefix, name, type_us, suffix, route_alias). FULLNAME is
# composed from these (see RoadDef.full_name). Stored uppercase to match source.
# The set exercises the typo checker: directional prefixes (N/S), quadrant
# suffixes (SE/NW), a multi-word name (RHODE ISLAND), an apostrophe (O'CONNOR),
# and varied street types.
_NAME_TABLES: Dict[str, Dict[str, Tuple[str, str, str, str, str]]] = {
    "dc": {
        "A1": ("", "CAPITOL", "ST", "", "801"),
        "A2": ("", "CONSTITUTION", "AVE", "", "802"),
        "A3": ("", "PENNSYLVANIA", "AVE", "SE", "803"),
        "A4": ("", "MASSACHUSETTS", "AVE", "NW", "804"),
        "C1": ("N", "DELAWARE", "ST", "", ""),
        "C2": ("S", "MARYLAND", "ST", "", ""),
        "C3": ("", "VIRGINIA", "RD", "", ""),
        "C4": ("", "RHODE ISLAND", "DR", "", ""),
    },
    "ottawa": {
        "A1": ("", "BANK", "ST", "", "417"),
        "A2": ("", "WELLINGTON", "ST", "", "418"),
        "A3": ("", "RIDEAU", "ST", "", "419"),
        "A4": ("", "LAURIER", "AVE", "", "420"),
        "C1": ("", "ELGIN", "ST", "", ""),
        "C2": ("", "KENT", "ST", "", ""),
        "C3": ("", "O'CONNOR", "ST", "", ""),
        "C4": ("", "GLADSTONE", "AVE", "", ""),
    },
}

# Per-region Virginia FIPS for V_FIPS. The source schema is Virginia-specific;
# for a non-VA region the field is realistically left blank.
_V_FIPS: Dict[str, str] = {"dc": "51059", "ottawa": ""}

REGION_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "dc": DEFAULT_BBOX_DC,
    "ottawa": DEFAULT_BBOX_OTTAWA,
}


def _region_roads(region_key: str) -> Tuple[RoadDef, ...]:
    """Assemble the RoadDef tuple for a region from shared topology + names + class."""
    names = _NAME_TABLES[region_key]
    roads: List[RoadDef] = []
    for key, verts in _TOPOLOGY.items():
        prefix, name, type_us, suffix, route_alias = names[key]
        road_class, cfcc, speed_limit = _CLASS_TABLE[key]
        roads.append(
            RoadDef(
                key=key,
                prefix=prefix,
                name=name,
                type_us=type_us,
                suffix=suffix,
                suffix2="",
                road_class=road_class,
                cfcc=cfcc,
                speed_limit=speed_limit,
                route_alias=route_alias,
                norm_vertices=verts,
                is_route_corridor=key in _ROUTE_CORRIDOR_KEYS,
            )
        )
    return tuple(roads)


# ==================================================================================================
# PLANAR FRAME
# ==================================================================================================

PlanarPt = Tuple[float, float]


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


def _make_frame(
    inset_bbox: Tuple[float, float, float, float],
) -> Tuple[
    float,
    float,
    Callable[[float, float], PlanarPt],
    Callable[[float, float], PlanarPt],
    Callable[[float, float], PlanarPt],
]:
    """Build a local equirectangular frame (feet) for an inset bbox.

    Returns:
        ``(width_ft, height_ft, norm_to_planar, planar_to_wgs, wgs_to_planar)``.
        The frame's origin is the bbox SW corner; ``norm_to_planar`` maps
        unit-square ``(u, v)`` to planar feet, ``planar_to_wgs`` maps planar feet
        to ``(lon, lat)``, and ``wgs_to_planar`` maps ``(lon, lat)`` back to planar
        feet. Distortion is negligible at metro scale and, more importantly, the
        topology is exact in the planar space where it is built.
    """
    min_lon, min_lat, max_lon, max_lat = inset_bbox
    mid_lat = 0.5 * (min_lat + max_lat)
    ft_per_deg_lat = EARTH_RADIUS_M * math.pi / 180.0 * FEET_PER_METER
    ft_per_deg_lon = ft_per_deg_lat * math.cos(math.radians(mid_lat))
    width_ft = (max_lon - min_lon) * ft_per_deg_lon
    height_ft = (max_lat - min_lat) * ft_per_deg_lat

    def norm_to_planar(u: float, v: float) -> PlanarPt:
        return (u * width_ft, v * height_ft)

    def planar_to_wgs(x: float, y: float) -> PlanarPt:
        return (min_lon + x / ft_per_deg_lon, min_lat + y / ft_per_deg_lat)

    def wgs_to_planar(lon: float, lat: float) -> PlanarPt:
        return ((lon - min_lon) * ft_per_deg_lon, (lat - min_lat) * ft_per_deg_lat)

    return width_ft, height_ft, norm_to_planar, planar_to_wgs, wgs_to_planar


# Spacing (feet) for densifying traced corridor shapes handed to the GTFS
# generator. Fine enough that great-circle interpolation between adjacent shape
# points is indistinguishable from the straight centerline (sub-foot), so stops
# placed along the shape land on the centerline.
SHAPE_DENSIFY_FT = 150.0


def _densify_planar_polyline(pts: Sequence[PlanarPt], spacing_ft: float) -> List[PlanarPt]:
    """Insert intermediate points along each leg of a planar polyline."""
    if len(pts) < 2:
        return list(pts)
    out: List[PlanarPt] = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= spacing_ft:
            out.append((x1, y1))
            continue
        steps = int(seg // spacing_ft)
        for i in range(1, steps + 1):
            f = (i * spacing_ft) / seg
            if f >= 1.0:
                break
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
        out.append((x1, y1))
    return out


# ==================================================================================================
# NETWORK ASSEMBLY
# ==================================================================================================


def _extract_points(geom: object) -> List[Point]:
    """Return the Point components of a shapely intersection result.

    Logs a warning for LineString overlaps, which would indicate two roads share
    a collinear stretch (a duplicate-geometry bug this fixture is designed to avoid).
    """
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, MultiPoint):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        pts: List[Point] = []
        for part in geom.geoms:
            pts.extend(_extract_points(part))
        return pts
    if isinstance(geom, LineString):
        logging.warning(
            "Collinear road overlap detected (%s); roads should meet only at points.", geom.wkt[:60]
        )
        return []
    return []


@dataclass
class _Segment:
    """One noded centerline segment in planar feet.

    Carries a reference to its parent :class:`RoadDef`, the rounded topology
    endpoints (shared exactly with adjacent segments at intersection nodes), the
    great-circle-densified planar polyline used for geometry/asphalt/geocoding,
    and the great-circle length in feet (used for ``Shape__Len``).
    """

    road: RoadDef
    endpoints: Tuple[PlanarPt, PlanarPt]
    densified: List[PlanarPt]
    length_ft: float

    @property
    def road_key(self) -> str:
        return self.road.key

    @property
    def full_name(self) -> str:
        return self.road.full_name


def _build_segments(
    roads: Sequence[RoadDef],
    norm_to_planar: Callable[[float, float], PlanarPt],
    planar_to_wgs: Callable[[float, float], PlanarPt],
    wgs_to_planar: Callable[[float, float], PlanarPt],
) -> List[_Segment]:
    """Node the road set at intersections and return attributed planar segments.

    Each road is split at every point where another road crosses it. Splitting is
    done by chainage (``project`` + ``substring``) rather than ``shapely.split`` to
    avoid touching-tolerance failures, and all endpoint coordinates are rounded to
    a common grid so shared junction nodes coincide exactly. Each segment's
    interior is then densified along the great circle between its endpoints so the
    centerline follows the same geodesic as the GTFS shapes and stops.
    """
    lines: Dict[str, LineString] = {
        r.key: LineString([norm_to_planar(u, v) for (u, v) in r.norm_vertices]) for r in roads
    }
    by_key: Dict[str, RoadDef] = {r.key: r for r in roads}

    # Gather per-road intersection chainages.
    inter_pts: Dict[str, set] = {r.key: set() for r in roads}
    keys = list(lines)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for pt in _extract_points(lines[keys[i]].intersection(lines[keys[j]])):
                inter_pts[keys[i]].add(
                    (round(pt.x, COORD_DECIMALS_FT), round(pt.y, COORD_DECIMALS_FT))
                )
                inter_pts[keys[j]].add(
                    (round(pt.x, COORD_DECIMALS_FT), round(pt.y, COORD_DECIMALS_FT))
                )

    segments: List[_Segment] = []
    for key, line in lines.items():
        rd = by_key[key]
        chainages = {0.0, line.length}
        for px, py in inter_pts[key]:
            chainages.add(line.project(Point(px, py)))
        ordered = sorted(chainages)
        for a, b in zip(ordered, ordered[1:]):
            if b - a < MIN_SEG_FT:
                continue  # collapse a near-coincident split pair
            seg = substring(line, a, b)
            (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
            c0 = (round(x0, COORD_DECIMALS_FT), round(y0, COORD_DECIMALS_FT))
            c1 = (round(x1, COORD_DECIMALS_FT), round(y1, COORD_DECIMALS_FT))
            # Centerlines are straight in the planar frame; the GTFS shapes that
            # trace them are densified at handoff (see corridor_polyline_wgs), so
            # stop placement walks short chords and stops land on the centerline.
            densified = [c0, c1]
            length_ft = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
            segments.append(
                _Segment(road=rd, endpoints=(c0, c1), densified=densified, length_ft=length_ft)
            )
    return segments


def validate_network(segments: Sequence[_Segment], width_ft: float, height_ft: float) -> None:
    """Assert the topology contract; raise ValueError listing every violation.

    Checks:
        * No dangling interior node (degree-1 nodes must lie on the bbox edge).
        * No segment shorter than ``MIN_SEG_FT``.
        * No duplicate segment geometry.
        * Each road's segments form a single connected polyline.
    """
    issues: List[str] = []
    w_r = round(width_ft, COORD_DECIMALS_FT)
    h_r = round(height_ft, COORD_DECIMALS_FT)

    def on_boundary(c: PlanarPt) -> bool:
        x, y = c
        return x == 0.0 or y == 0.0 or x == w_r or y == h_r

    endpoint_count: Counter = Counter()
    for s in segments:
        endpoint_count[s.endpoints[0]] += 1
        endpoint_count[s.endpoints[1]] += 1
    for node, deg in endpoint_count.items():
        if deg == 1 and not on_boundary(node):
            issues.append(f"dangling interior node at {node} (degree 1, not on bbox edge)")

    for s in segments:
        (x0, y0), (x1, y1) = s.endpoints
        if math.hypot(x1 - x0, y1 - y0) < MIN_SEG_FT:
            issues.append(f"sub-min segment on {s.road_key} ({s.full_name})")

    seen: set = set()
    for s in segments:
        key = tuple(sorted(s.endpoints))
        if key in seen:
            issues.append(f"duplicate segment geometry on {s.road_key} ({s.full_name})")
        seen.add(key)

    by_road: Dict[str, List[LineString]] = defaultdict(list)
    for s in segments:
        by_road[s.road_key].append(LineString(s.endpoints))
    for road_key, parts in by_road.items():
        merged = linemerge(parts)
        if merged.geom_type != "LineString":
            issues.append(
                f"road {road_key} is not a single connected polyline ({merged.geom_type})"
            )

    if issues:
        raise ValueError("Network validation failed:\n  - " + "\n  - ".join(issues))


def build_region_network(region_key: str) -> Tuple[gpd.GeoDataFrame, Dict[str, object]]:
    """Build, validate, and return the centerline GeoDataFrame (WGS84) for a region.

    Attribute columns mirror a realistic subset of the Fairfax County
    ``Roadway_Centerlines`` schema: the full name-field group
    (``FULLNAME, RW_PREFIX, RW_NAME, RW_TYPE_US, RW_SUFFIX, RW_SUFFIX_``) plus
    identifiers, classification, and edit-metadata fields. Geometry is WGS84;
    ``Shape__Len`` is the planar length in feet (the attribute's documented unit),
    independent of the WGS84 geometry's degree-based ``.length``.

    Returns:
        ``(gdf, stats)`` where ``stats`` summarizes the build (segment/node counts
        and the node-degree histogram).
    """
    roads = _region_roads(region_key)
    inset = _bbox_with_inset(REGION_BBOXES[region_key], BBOX_INSET_FRACTION)
    width_ft, height_ft, norm_to_planar, planar_to_wgs, wgs_to_planar = _make_frame(inset)

    segments = _build_segments(roads, norm_to_planar, planar_to_wgs, wgs_to_planar)
    validate_network(segments, width_ft, height_ft)

    region_offset = list(REGION_BBOXES).index(region_key) * ID_REGION_STRIDE
    v_fips = _V_FIPS.get(region_key, "")

    records = []
    geoms = []
    for i, s in enumerate(segments):
        rd = s.road
        seg_no = region_offset + i
        records.append(
            {
                # --- identifiers ---
                "OBJECTID": ID_BASE_OBJECTID + seg_no,
                "TRANS_ID": ID_BASE_TRANS_ID + seg_no,
                "SEGID": ID_BASE_SEGID + seg_no,
                # --- name fields (the typo-checker's required group) ---
                "FULLNAME": rd.full_name,
                "RW_PREFIX": rd.prefix,
                "RW_NAME": rd.name,
                "RW_TYPE_US": rd.type_us,
                "RW_SUFFIX": rd.suffix,
                "RW_SUFFIX_": rd.suffix2,
                # --- classification ---
                "CFCC": rd.cfcc,
                "ROAD_CLASS": rd.road_class,
                "FFX_CLASS": rd.road_class,
                "SPEED_LIMI": rd.speed_limit,
                "ONEWAY": "N",
                "DIVIDED": "N",
                "ROUTE_ALIA": rd.route_alias,
                # --- status / jurisdiction ---
                "STATUS": SCHEMA_STATUS,
                "V_FIPS": v_fips,
                "EFF_DATE": SCHEMA_EFF_DATE,
                # --- edit metadata ---
                "GlobalID": str(uuid.uuid5(uuid.NAMESPACE_OID, f"{region_key}:{seg_no}")),
                "CreationDa": SCHEMA_TIMESTAMP,
                "Creator": SCHEMA_CREATOR,
                "EditDate": SCHEMA_TIMESTAMP,
                "Editor": SCHEMA_EDITOR,
                "Shape__Len": round(s.length_ft, 6),
                # --- traceability (not in source schema; useful for the GTFS step) ---
                "road_key": rd.key,
            }
        )
        geoms.append(LineString([planar_to_wgs(x, y) for (x, y) in s.densified]))

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=OUTPUT_CRS)

    node_count: Counter = Counter()
    for s in segments:
        node_count[s.endpoints[0]] += 1
        node_count[s.endpoints[1]] += 1
    degree_hist = Counter(node_count.values())
    stats = {
        "segments": len(segments),
        "nodes": len(node_count),
        "degree_hist": dict(sorted(degree_hist.items())),
        "roads": len(roads),
    }
    return gdf, stats


# ==================================================================================================
# REGION GEOMETRY (shared by asphalt + geocoder + stop decoration)
# ==================================================================================================


@dataclass
class RegionGeom:
    """Bundled planar geometry for a region, built once and shared by consumers.

    Attributes:
        roads: The region's RoadDefs.
        road_lines: Planar (feet) full LineString per road key (unsplit).
        nodes: Mapping of rounded planar node -> set of road keys meeting there
            (intersection nodes have >= 2 keys).
        asphalt: Dissolved planar road-surface polygon (union of per-class buffers).
        planar_to_wgs: Frame map planar feet -> (lon, lat).
        wgs_to_planar: Frame map (lon, lat) -> planar feet.
    """

    roads: Tuple[RoadDef, ...]
    road_lines: Dict[str, LineString]
    nodes: Dict[PlanarPt, set]
    asphalt: object  # shapely (Multi)Polygon in planar feet
    planar_to_wgs: Callable[[float, float], PlanarPt]
    wgs_to_planar: Callable[[float, float], PlanarPt]


def _build_region_geom(region_key: str) -> RegionGeom:
    """Build the shared planar geometry (lines, nodes, asphalt) for a region.

    Reuses :func:`_build_segments` so the geocoder's road lines, the asphalt
    buffers, and the exported centerlines are all the same great-circle-densified
    geometry.
    """
    roads = _region_roads(region_key)
    inset = _bbox_with_inset(REGION_BBOXES[region_key], BBOX_INSET_FRACTION)
    _w, _h, norm_to_planar, planar_to_wgs, wgs_to_planar = _make_frame(inset)
    by_key = {r.key: r for r in roads}

    segments = _build_segments(roads, norm_to_planar, planar_to_wgs, wgs_to_planar)

    # Merge each road's densified segments into one planar polyline.
    seg_lines: Dict[str, List[LineString]] = defaultdict(list)
    nodes: Dict[PlanarPt, set] = defaultdict(set)
    for s in segments:
        seg_lines[s.road_key].append(LineString(s.densified))
        nodes[s.endpoints[0]].add(s.road_key)
        nodes[s.endpoints[1]].add(s.road_key)
    road_lines: Dict[str, LineString] = {}
    for k, parts in seg_lines.items():
        merged = linemerge(parts)
        road_lines[k] = merged if merged.geom_type == "LineString" else parts[0]

    # Intersection nodes: those shared by >= 2 roads.
    inter_nodes = {pt: keys for pt, keys in nodes.items() if len(keys) >= 2}

    # Asphalt: buffer each road's densified line by its class half-width, dissolve.
    buffers = [
        road_lines[k].buffer(ASPHALT_HALF_WIDTH_FT[by_key[k].road_class], cap_style="flat")
        for k in road_lines
    ]
    asphalt = unary_union(buffers)

    return RegionGeom(
        roads=roads,
        road_lines=road_lines,
        nodes=dict(inter_nodes),
        asphalt=asphalt,
        planar_to_wgs=planar_to_wgs,
        wgs_to_planar=wgs_to_planar,
    )


def build_asphalt_polygons(region_key: str) -> gpd.GeoDataFrame:
    """Return the road-surface (asphalt) polygons for a region in WGS84.

    Centerlines are buffered by a per-road-class half-width and dissolved per
    road class, yielding one multipolygon row per class present. Consumed by
    ``stop_v_conflict_checker_gpd.py`` (point-in-polygon overlap test), which is
    geometry-only and does not require attributes; ``road_class`` and
    ``half_width_ft`` are carried for traceability.
    """
    geom = _build_region_geom(region_key)
    by_key = {r.key: r for r in geom.roads}

    by_class: Dict[str, list] = defaultdict(list)
    for k, line in geom.road_lines.items():
        rc = by_key[k].road_class
        by_class[rc].append(line.buffer(ASPHALT_HALF_WIDTH_FT[rc], cap_style="flat"))

    records, geoms = [], []
    for rc, polys in sorted(by_class.items()):
        dissolved = unary_union(polys)
        wgs = _planar_poly_to_wgs(dissolved, geom.planar_to_wgs)
        records.append({"road_class": rc, "half_width_ft": ASPHALT_HALF_WIDTH_FT[rc]})
        geoms.append(wgs)
    return gpd.GeoDataFrame(records, geometry=geoms, crs=OUTPUT_CRS)


def _planar_poly_to_wgs(poly: object, planar_to_wgs: Callable[[float, float], PlanarPt]) -> object:
    """Reproject a planar shapely polygon/multipolygon to WGS84 via the frame map."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import transform

    def _tx(
        x: float,
        y: float,
        z: float | None = None,  # noqa: ANN202
    ) -> tuple[float, float]:
        lon, lat = planar_to_wgs(x, y)
        return (lon, lat)

    if isinstance(poly, (Polygon, MultiPolygon)):
        return transform(_tx, poly)
    return transform(_tx, poly)


# ==================================================================================================
# GEOCODER + STOP DECORATION
# ==================================================================================================


class RegionGeocoder:
    """Names stops from adjacent roads and offsets them off the roadbed.

    Built from a :class:`RegionGeom`. All spatial work is in planar feet; inputs
    and outputs are WGS84 ``(lat, lon)``.
    """

    def __init__(self, geom: RegionGeom) -> None:  # noqa: D107
        self._geom = geom

    def nearest_road(self, lat: float, lon: float, candidates: Sequence[str]) -> str:
        """Return the candidate road key whose centerline is nearest the point."""
        x, y = self._geom.wgs_to_planar(lon, lat)
        p = Point(x, y)
        return min(candidates, key=lambda k: self._geom.road_lines[k].distance(p))

    def name_at(self, lat: float, lon: float, on_road_key: str) -> Tuple[str, Optional[str]]:
        """Return ``(stop_name, cross_road_key)`` for a stop on ``on_road_key``.

        Name is ``"ON_ROAD @ CROSS_ROAD"`` when the stop is within
        ``CROSS_STREET_NEAR_FT`` of an intersection on the on-road, else the
        on-road name alone (mid-block).
        """
        by_key = {r.key: r for r in self._geom.roads}
        on_name = by_key[on_road_key].full_name
        x, y = self._geom.wgs_to_planar(lon, lat)
        p = Point(x, y)

        best_cross: Optional[str] = None
        best_dist = math.inf
        for node, road_keys in self._geom.nodes.items():
            if on_road_key not in road_keys:
                continue
            others = [k for k in road_keys if k != on_road_key]
            if not others:
                continue
            d = math.hypot(p.x - node[0], p.y - node[1])
            if d < best_dist:
                best_dist = d
                best_cross = sorted(others)[0]  # deterministic pick

        if best_cross is not None and best_dist <= CROSS_STREET_NEAR_FT:
            return f"{on_name} @ {by_key[best_cross].full_name}", best_cross
        return on_name, None

    def offset_off_road(
        self, lat: float, lon: float, on_road_key: str, feet: float
    ) -> Tuple[float, float]:
        """Nudge a point perpendicular to its on-road centerline by ``feet``."""
        if feet == 0.0:
            return lat, lon
        line = self._geom.road_lines[on_road_key]
        x, y = self._geom.wgs_to_planar(lon, lat)
        p = Point(x, y)
        s = line.project(p)
        a = line.interpolate(max(0.0, s - 1.0))
        b = line.interpolate(min(line.length, s + 1.0))
        dx, dy = (b.x - a.x), (b.y - a.y)
        norm = math.hypot(dx, dy) or 1.0
        # Left-hand normal in direction of increasing chainage (consistent side).
        nx, ny = (-dy / norm, dx / norm)
        ox, oy = (x + nx * feet, y + ny * feet)
        lon2, lat2 = self._geom.planar_to_wgs(ox, oy)
        return lat2, lon2

    def in_asphalt(self, lat: float, lon: float) -> bool:
        """True if the WGS84 point falls within the (planar) asphalt polygon."""
        x, y = self._geom.wgs_to_planar(lon, lat)
        return self._geom.asphalt.contains(Point(x, y))


def _hash_unit(text: str, salt: str) -> float:
    """Deterministic float in [0, 1) from a string + salt (stable across runs)."""
    h = hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _longest_alpha_token(bare_road_name: str) -> str:
    """Return the longest purely-alphabetic token of a bare road name."""
    tokens = re.findall(r"[A-Za-z]+", bare_road_name)
    return max(tokens, key=len) if tokens else ""


def _typo_is_catchable(bare_road_name: str, threshold: int = 80) -> bool:
    """True if a single-edit typo on this name stays above the checker threshold.

    A one-character substitution in a length-L token scores ``(1 - 1/L) * 100`` on
    the checker's ratio metric, so the longest token must be long enough to clear
    ``threshold`` (e.g. >= 5 chars for an 80 threshold). Short names like ``BANK``
    cannot host a catchable typo, so they are skipped rather than seeded with a
    positive the checker would silently miss.
    """
    tok = _longest_alpha_token(bare_road_name)
    return bool(tok) and (1.0 - 1.0 / len(tok)) * 100.0 >= threshold


def _apply_typo(name: str, bare_road_name: str, stop_id: str) -> str:
    """Corrupt a single character of the longest token of the bare road name.

    Deterministic per stop_id. Swaps one interior alphabetic character of the
    bare road name's longest token (e.g. ``"WELLINGTON"`` in ``"WELLINGTON ST"``)
    for a neighbour, producing a single-edit typo. The longest token is targeted
    (rather than the street type or a short directional) so the typo survives the
    roadname checker's modifier-stripping normalization and stays above its
    similarity threshold.
    """
    token = _longest_alpha_token(bare_road_name)
    idx_tok_in_name = name.find(token)
    if not token or idx_tok_in_name < 0:
        return name
    alpha_positions = [i for i in range(1, len(token) - 1) if token[i].isalpha()]
    if not alpha_positions:
        return name
    pick = alpha_positions[int(_hash_unit(stop_id, "typo_pos") * len(alpha_positions))]
    orig = token[pick]
    shifted = chr((ord(orig.upper()) - ord("A") + 1) % 26 + ord("A"))
    new_token = token[:pick] + shifted + token[pick + 1 :]
    return name[:idx_tok_in_name] + new_token + name[idx_tok_in_name + len(token) :]


def _utm_epsg_for_lon(lon: float) -> int:
    """Return the northern-hemisphere UTM EPSG code for a longitude."""
    zone = int((lon + 180.0) / 6.0) + 1
    return 32600 + zone


def decorate_stops(
    region_key: str,
    stops_by_route: Dict[str, List[Tuple[str, float, float, float]]],
    route_corridor_keys: Dict[str, Sequence[str]],
) -> Tuple[Dict[str, Tuple[float, float, str]], Dict[str, list]]:
    """Name, offset, and seed positives for every unique stop in a region.

    Args:
        region_key: Region identifier.
        stops_by_route: Mapping route_id -> list of ``(stop_id, lat, lon, dist_ft)``.
        route_corridor_keys: Mapping route_id -> the road keys that route rides
            (used to disambiguate which road each stop sits on).

    Returns:
        ``(stop_meta, manifest)`` where ``stop_meta`` maps stop_id ->
        ``(lat, lon, stop_name)`` (offset applied, typo applied where seeded), and
        ``manifest`` documents the seeded positives:
        ``{"typos": [...], "conflicts": [...]}`` for downstream test assertions.

    Containment is evaluated in a projected (UTM) CRS so the conflict manifest
    matches what ``stop_v_conflict_checker_gpd.py`` computes (it projects to a
    metric CRS and runs a point-in-polygon test). Normal stops that would land in
    a crossing road's pavement near a corner are actively nudged clear, so the
    conflict set stays small and intentional (≈ the seeded rate) rather than
    accumulating incidental corner overlaps.
    """
    from pyproj import Transformer  # noqa: PLC0415

    geom = _build_region_geom(region_key)
    geocoder = RegionGeocoder(geom)
    by_key = {r.key: r for r in geom.roads}

    # Asphalt in a projected CRS, plus a WGS84->UTM point transformer.
    inset = _bbox_with_inset(REGION_BBOXES[region_key], BBOX_INSET_FRACTION)
    centroid_lon = 0.5 * (inset[0] + inset[2])
    utm_epsg = _utm_epsg_for_lon(centroid_lon)
    asphalt_utm = build_asphalt_polygons(region_key).to_crs(epsg=utm_epsg).geometry.union_all()
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    def in_asphalt_proj(lat: float, lon: float) -> bool:
        x, y = to_utm.transform(lon, lat)
        return asphalt_utm.contains(Point(x, y))

    # Collect each unique stop with the union of candidate roads across its routes.
    candidates: Dict[str, set] = defaultdict(set)
    coords: Dict[str, Tuple[float, float]] = {}
    for route_id, stops in stops_by_route.items():
        cand = route_corridor_keys.get(route_id, list(geom.road_lines))
        for stop_id, lat, lon, _dist in stops:
            candidates[stop_id].update(cand)
            coords.setdefault(stop_id, (lat, lon))

    stop_meta: Dict[str, Tuple[float, float, str]] = {}
    manifest: Dict[str, list] = {"typos": [], "conflicts": []}

    for stop_id in sorted(coords):
        lat, lon = coords[stop_id]
        on_key = geocoder.nearest_road(lat, lon, sorted(candidates[stop_id]))
        base_name, _cross = geocoder.name_at(lat, lon, on_key)

        seeded_conflict = _hash_unit(stop_id, "conflict") < CONFLICT_RATE
        is_typo = _hash_unit(stop_id, "typo") < TYPO_RATE and _typo_is_catchable(
            by_key[on_key].name
        )

        if seeded_conflict:
            nlat, nlon = lat, lon  # left in the roadbed
        else:
            # Offset off the roadbed; if that lands in a crossing road's pavement
            # near a corner, try the other side, then a wider offset, until clear.
            nlat, nlon = lat, lon
            for feet in (STOP_OFFSET_FT, -STOP_OFFSET_FT, 48.0, -48.0):
                clat, clon = geocoder.offset_off_road(lat, lon, on_key, feet)
                if not in_asphalt_proj(clat, clon):
                    nlat, nlon = clat, clon
                    break
            else:
                nlat, nlon = geocoder.offset_off_road(lat, lon, on_key, STOP_OFFSET_FT)

        name = base_name
        if is_typo:
            name = _apply_typo(base_name, by_key[on_key].name, stop_id)
            manifest["typos"].append(
                {"stop_id": stop_id, "true_road": by_key[on_key].full_name, "stop_name": name}
            )

        # The conflict manifest is the oracle for stop_v_conflict_checker: record
        # every stop whose final position intersects the asphalt (seeded, or an
        # unavoidable corner overlap), evaluated in the projected CRS.
        if in_asphalt_proj(nlat, nlon):
            manifest["conflicts"].append(
                {"stop_id": stop_id, "on_road": by_key[on_key].full_name, "seeded": seeded_conflict}
            )

        stop_meta[stop_id] = (nlat, nlon, name)

    return stop_meta, manifest


# ==================================================================================================
# EXPORT
# ==================================================================================================


def export_network(gdf: gpd.GeoDataFrame, out_dir: Path, region_key: str) -> Tuple[Path, Path]:
    """Write the centerline network as BOTH a shapefile and a GeoPackage layer."""
    region_dir = out_dir / region_key
    region_dir.mkdir(parents=True, exist_ok=True)
    shp_path = region_dir / f"{region_key}_road_centerlines.shp"
    gpkg_path = region_dir / f"{region_key}_road_centerlines.gpkg"
    gdf.to_file(shp_path)  # ESRI Shapefile (all attribute names <= 10 chars; no truncation)
    gdf.to_file(gpkg_path, layer="road_centerlines", driver="GPKG")
    return shp_path, gpkg_path


def export_asphalt(gdf: gpd.GeoDataFrame, out_dir: Path, region_key: str) -> Tuple[Path, Path]:
    """Write the asphalt (road-surface) polygons as BOTH shapefile and GeoPackage."""
    region_dir = out_dir / region_key
    region_dir.mkdir(parents=True, exist_ok=True)
    shp_path = region_dir / f"{region_key}_road_asphalt.shp"
    gpkg_path = region_dir / f"{region_key}_road_asphalt.gpkg"
    gdf.to_file(shp_path)
    gdf.to_file(gpkg_path, layer="road_asphalt", driver="GPKG")
    return shp_path, gpkg_path


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> None:
    """Build, validate, and export the mock road network for every region."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
    logging.info("====================================================")
    logging.info("Mock Road-Centerline Generator")
    logging.info("Output folder: %s", OUTPUT_DIR)
    logging.info("====================================================")

    if str(OUTPUT_DIR).startswith("Path"):
        logging.warning(
            "OUTPUT_DIR is still the placeholder. Edit the CONFIGURATION block "
            "at the top of this script to point at a real folder, then re-run."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for region_key in REGION_BBOXES:
        gdf, stats = build_region_network(region_key)
        shp_path, gpkg_path = export_network(gdf, OUTPUT_DIR, region_key)
        asphalt_gdf = build_asphalt_polygons(region_key)
        asp_shp, asp_gpkg = export_asphalt(asphalt_gdf, OUTPUT_DIR, region_key)
        logging.info(
            "  %s: %d segments / %d nodes (degree hist %s) -> %s, %s",
            region_key,
            stats["segments"],
            stats["nodes"],
            stats["degree_hist"],
            shp_path.name,
            gpkg_path.name,
        )
        logging.info(
            "       asphalt: %d class polygons -> %s, %s",
            len(asphalt_gdf),
            asp_shp.name,
            asp_gpkg.name,
        )
    logging.info("Script completed successfully.")


# ==================================================================================================
# NOTEBOOK SHIM
# ==================================================================================================
# Allow a separate Jupyter cell to do `import generate_mock_roads as roads` without
# having this file on disk anywhere. When this code runs in a notebook the module
# name is `__main__`, so we additionally publish it under the expected name. When
# the file IS on disk and imported normally, this assignment is a harmless no-op
# (it just points the entry at itself).
import sys as _sys  # noqa: E402,I001

_sys.modules["generate_mock_roads"] = _sys.modules[__name__]
del _sys


if __name__ == "__main__":
    main()
