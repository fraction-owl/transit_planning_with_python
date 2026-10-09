"""Audit a pedestrian line network around GTFS stops for gaps and connectivity problems.

Builds straight-line review buffers around GTFS platform stops, keeps the whole
sidewalk, trail, and crossing lines that reach them (plus a margin), and builds
a diagnostic undirected graph from the lines' existing vertices. It flags
disconnected endpoints, near-miss gaps, crossings without a shared vertex,
overlaps, short or coarse segments, small isolated groups, and stops far from
any line. Optional road polygons mark gaps that span a street as possible
missing crossings. Inputs are never edited. GeoPandas port of
``walking_network_qa_arcpy.py``.

Lines connect only where eligible vertices coincide within
``COINCIDENCE_TOLERANCE_FT`` (any vertex, or endpoints only) and their
constant ``LEVEL_FIELD`` values match; two blank levels connect. Gap and
crossing thresholds are review screens, never snap distances. Lines with
geometry problems stay in the network layers but are left out of the graph.

Inputs
------
- One or more pedestrian line layers (any format geopandas reads) with a
  defined CRS, each with an optional SQL filter, stable ID field, and level field.
- GTFS feed folder or .zip: stops.txt, plus routes/trips/stop_times when
  filtering by route.
- Optional road polygon layer (street area).
- Optional approved-snaps CSV: an edited copy of a previous repair candidates CSV.

Outputs
-------
- ``walking_network_qa.gpkg``: stops, buffers, study/context areas, road
  context, original and working networks, components, issue points and
  lines, and optional map clips per radius (overwritten on each run).
- CSVs of issues, per-stop coverage, components, source attributes, geometry
  preflight, repair candidates, and repairs applied; a JSON run summary.
- ``walking_network_qa_runlog.txt`` capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags)
and run from a shell or a Jupyter notebook (requires geopandas; see
``requirements.txt``). Leave repairs off for the first run and review the issue
layers. To close tiny gaps, copy the repair candidates CSV elsewhere, set
APPROVE to YES on chosen rows, and rerun with ``APPROVED_SNAPS_CSV`` set.

Limitations
-----------
- Buffers are straight-line review radii, not walk sheds. Components touching
  the context edge may connect outside the retained data.
- XY diagnostics only: Z values never decide connectivity, and nothing here
  certifies grade, legal access, slope, curb ramps, or a routable network.
- Road polygons suggest where a crossing may be missing, not that one is
  permitted or safe. GDAL linearizes curved geometry on read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, NamedTuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import LineString, MultiLineString, Point

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Pedestrian line layers to audit. Add crossings or trails only when they are
# separate, non-duplicated inputs. Keys:
#   LABEL        unique name (letter first; letters, digits, underscores)
#   PATH         any line layer geopandas reads (.shp, .gpkg, .geojson, ...)
#   LAYER        layer inside a multi-layer file (GeoPackage, FileGDB); "" = first
#   WHERE        optional SQL filter, e.g. "STATUS = 'EXISTING'"
#   ID_FIELD     optional stable ID field; "" = the source feature ID (FID)
#   LEVEL_FIELD  optional field with ONE grade level for the whole feature (never
#                from/to elevations). Equal levels connect; two blanks connect.
NETWORK_SOURCES: list[dict[str, str]] = [
    {
        "LABEL": "sidewalks",
        "PATH": r"Path\To\Your\Sidewalks.shp",
        "LAYER": "",
        "WHERE": "",
        "ID_FIELD": "",
        "LEVEL_FIELD": "",
    },
]

# GTFS feed folder or .zip. Platform stops (location_type blank or 0) are used.
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"

# Optional stop filters. Route short names keep stops served by any trip on
# those routes (no calendar or date filter); stop IDs narrow the result further.
GTFS_ROUTE_SHORT_NAMES: list[str] = []
GTFS_STOP_IDS: list[str] = []

# Optional road polygons (street area), used only to annotate gaps that span a
# street. Leave "" to skip the road checks.
ROAD_POLYGONS_PATH: str = r""
ROAD_POLYGONS_LAYER: str = ""
ROAD_WHERE: str = ""

# Output folder and file names. Each run overwrites these files.
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"
REVIEW_GPKG_FILENAME: str = r"walking_network_qa.gpkg"
ISSUES_FILENAME: str = r"walking_network_issues.csv"
STOP_SUMMARY_FILENAME: str = r"walking_network_stop_summary.csv"
COMPONENTS_FILENAME: str = r"walking_network_components.csv"
SOURCE_ATTRIBUTES_FILENAME: str = r"walking_network_source_attributes.csv"
PREFLIGHT_FILENAME: str = r"walking_network_geometry_preflight.csv"
REPAIR_CANDIDATES_FILENAME: str = r"walking_network_repair_candidates.csv"
REPAIR_LOG_FILENAME: str = r"walking_network_repair_log.csv"
SUMMARY_FILENAME: str = r"walking_network_summary.json"

# Local projected CRS in feet or meters; "" uses the first network layer's CRS.
# Default: NAD83 / Maryland State Plane (US feet), the repo's Washington, DC CRS.
ANALYSIS_CRS: str = "EPSG:2248"

# Straight-line review radii around each stop, in miles (not walk sheds).
STOP_BUFFER_MILES: list[float] = [0.25]
# Whole lines within this distance of the largest buffer are kept as context.
CONTEXT_MARGIN_FT: float = 200.0
# Also write a map-only clip of the network for each radius (never re-audit it).
WRITE_CLIPPED_NETWORK: bool = True
# "ANY_VERTEX" or "ENDPOINT": which coincident vertices join two lines. Match
# the connectivity you intend to use for routing.
CONNECTIVITY_POLICY: str = "ANY_VERTEX"
# Vertices this close count as one point. Coordinate precision, NOT a snap
# distance: keep it tiny.
COINCIDENCE_TOLERANCE_FT: float = 0.001

# Review thresholds: starting points, not established accuracy standards.
GAP_REVIEW_FT: float = 10.0
CROSSING_SEARCH_FT: float = 120.0  # Applied only when a gap overlaps road area.
MIN_ROAD_OVERLAP_FT: float = 1.0  # Ignore negligible touches at a road edge.
ROAD_EDGE_REVIEW_FT: float = 6.0
SHORT_SEGMENT_FT: float = 0.5
COARSE_SEGMENT_FT: float = 150.0
SPIKE_LEG_FT: float = 5.0
SPIKE_TURN_DEGREES: float = 150.0
SMALL_COMPONENT_LENGTH_FT: float = 200.0
STOP_NEAR_NETWORK_FT: float = 75.0
LOW_DENSITY_MI_PER_SQ_MI: float = 2.0  # Screening only; 0 turns the flag off.

# Repairs are OFF by default and only ever change the working network output.
# Duplicate-vertex cleanup applies to 2D lines whose only problem is a repeat.
REMOVE_EXACT_DUPLICATE_VERTICES: bool = False
# Edited copy of a previous repair candidates CSV, with APPROVE = YES rows.
APPROVED_SNAPS_CSV: str = r""
MAX_APPROVED_SNAP_FT: float = 1.0

# Every output must be traceable: a failed run-log write aborts the script.
# Set to False only when writing to a genuinely read-only location.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO

# === END CONFIG ===

CONFIG_BEGIN_MARKER = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER = "# === END CONFIG ==="

PLACEHOLDER_MARK = "Path\\To\\Your"
METERS_PER_FOOT = 0.3048
FEET_PER_MILE = 5280.0
POLICIES = ("ANY_VERTEX", "ENDPOINT")
WEB_MERCATOR_CODES = frozenset({3785, 3857, 900913, 102100, 102113})
LINE_TYPES = ("LineString", "MultiLineString")
POLYGON_TYPES = ("Polygon", "MultiPolygon")
LABEL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}")

ISSUE_COLUMNS = [
    "ISSUE_ID",
    "ISSUE_TYPE",
    "CONFIDENCE",
    "SOURCE_KEY",
    "OTHER_KEY",
    "PART_NO",
    "OTHER_PART",
    "VALUE_FT",
    "ROAD_FT",
    "DETAIL",
    "REVIEW_STATUS",
]
REPAIR_COLUMNS = [
    "APPROVE",
    "ISSUE_ID",
    "SOURCE_KEY",
    "SOURCE_PART",
    "SOURCE_ENDPOINT",
    "TARGET_KEY",
    "TARGET_PART",
    "TARGET_ENDPOINT",
    "GAP_FT",
    "SOURCE_HASH",
    "TARGET_HASH",
    "REASON",
]
REPAIR_LOG_COLUMNS = ["SOURCE_KEY", "ACTION", "MOVE_FT", "OLD_WKT", "NEW_WKT"]
COMPONENT_COLUMNS = [
    "COMP_ID",
    "PART_COUNT",
    "FEATURE_COUNT",
    "CONTEXT_LENGTH_FT",
    "STUDY_LENGTH_FT",
    "CONTEXT_EDGE",
]
STOP_SUMMARY_COLUMNS = [
    "STOP_ID",
    "STOP_NAME",
    "RADIUS_MI",
    "MAPPED_LENGTH_FT",
    "BUFFER_SQ_MI",
    "DENSITY_MI_PER_SQ_MI",
    "COMPONENT_COUNT",
    "NEAREST_LINE_FT",
    "NEAREST_COMPONENT",
    "PRE_STOP_SCREEN_ISSUES",
    "GEOMETRY_EXCLUDED_IN_RUN",
    "STOP_CONNECTION_VERIFIED",
]
PREFLIGHT_COLUMNS = ["SOURCE_LABEL", "SOURCE_FID", "PROBLEM", "SCOPE"]
SOURCE_ATTRIBUTE_COLUMNS = [
    "SOURCE_KEY",
    "SOURCE_LABEL",
    "SOURCE_FID",
    "SOURCE_ID",
    "ATTRIBUTES_JSON",
]

LIMITATIONS = [
    "Review buffers are straight-line, not network service areas.",
    "Undirected XY diagnostic graph; routing and stop access are not certified.",
    "Unknown levels, legal access, barriers, slope and curb ramps need review.",
    "Lines crossing without shared eligible vertices remain disconnected.",
    "Components touching the context boundary may connect outside retained data.",
    "Excluded geometry can cause apparent gaps and low coverage.",
    "Mapped length includes duplicate or overlapping lines until resolved.",
    "Density flags need land-use or imagery context; they do not establish omission.",
    "Issue types overlap; issues_per_mapped_mile is not an accuracy percentage.",
    "Features without geometry cannot be located; see the geometry preflight CSV.",
    "Clips acquire artificial endpoints; do not use them to rerun this audit.",
]

XY = tuple[float, float]


class ConfigError(ValueError):
    """A configuration value is missing, inconsistent, or unusable."""


# =============================================================================
# SETTINGS AND DATA MODELS
# =============================================================================


class SourceSpec(NamedTuple):
    """One pedestrian line layer from NETWORK_SOURCES."""

    label: str
    path: str
    layer: str = ""
    where: str = ""
    id_field: str = ""
    level_field: str = ""


class Settings(NamedTuple):
    """Run settings: the CONFIGURATION values with any CLI overrides applied."""

    network_sources: tuple[SourceSpec, ...]
    gtfs_path: str
    output_dir: Path
    route_short_names: frozenset[str] = frozenset()
    stop_ids: frozenset[str] = frozenset()
    road_polygons_path: str = ""
    road_polygons_layer: str = ""
    road_where: str = ""
    analysis_crs: str = ANALYSIS_CRS
    stop_buffer_miles: tuple[float, ...] = tuple(STOP_BUFFER_MILES)
    context_margin_ft: float = CONTEXT_MARGIN_FT
    write_clipped_network: bool = WRITE_CLIPPED_NETWORK
    connectivity_policy: str = CONNECTIVITY_POLICY
    coincidence_tolerance_ft: float = COINCIDENCE_TOLERANCE_FT
    gap_review_ft: float = GAP_REVIEW_FT
    crossing_search_ft: float = CROSSING_SEARCH_FT
    min_road_overlap_ft: float = MIN_ROAD_OVERLAP_FT
    road_edge_review_ft: float = ROAD_EDGE_REVIEW_FT
    short_segment_ft: float = SHORT_SEGMENT_FT
    coarse_segment_ft: float = COARSE_SEGMENT_FT
    spike_leg_ft: float = SPIKE_LEG_FT
    spike_turn_degrees: float = SPIKE_TURN_DEGREES
    small_component_length_ft: float = SMALL_COMPONENT_LENGTH_FT
    stop_near_network_ft: float = STOP_NEAR_NETWORK_FT
    low_density_mi_per_sq_mi: float = LOW_DENSITY_MI_PER_SQ_MI
    remove_exact_duplicate_vertices: bool = REMOVE_EXACT_DUPLICATE_VERTICES
    approved_snaps_csv: str = APPROVED_SNAPS_CSV
    max_approved_snap_ft: float = MAX_APPROVED_SNAP_FT


class Feature:
    """A selected source line with its projected geometry and lineage."""

    __slots__ = (
        "key",
        "label",
        "fid",
        "source_id",
        "level",
        "original",
        "geometry",
        "attrs",
        "problems",
        "excluded_reason",
    )

    def __init__(
        self,
        key: str,
        label: str,
        fid: str,
        source_id: str,
        level: str,
        geometry: Any,
        attrs: dict[str, Any],
        problems: list[str],
    ) -> None:
        """Store lineage; the working geometry starts as the projected original."""
        self.key = key
        self.label = label
        self.fid = fid
        self.source_id = source_id
        self.level = level
        self.original = geometry
        self.geometry = geometry
        self.attrs = attrs
        self.problems = problems
        self.excluded_reason = ""

    @property
    def has_z(self) -> bool:
        """Whether the source geometry carries Z values."""
        return bool(shapely.has_z(self.original))


class Part(NamedTuple):
    """One continuous line part; parts of a multipart feature are not joined."""

    pid: int
    feature_key: str
    part_number: int
    points: tuple[XY, ...]
    level: str


class Issue(NamedTuple):
    """A diagnostic observation; confidence describes its reading as an error."""

    kind: str
    points: tuple[XY, ...]
    pid: int = -1
    other_pid: int = -1
    value_ft: Optional[float] = None
    confidence: str = "REVIEW"
    detail: str = ""
    road_ft: float = 0.0
    feature_key: str = ""
    issue_id: str = ""

    def geometry(self) -> Any:
        """Return a point for one location, a line for several, or None."""
        if not self.points:
            return None
        if len(self.points) == 1:
            return Point(self.points[0])
        return LineString(self.points)


@contextmanager
def timed_stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    """Log a stage's start and duration, and record the duration in *timings*."""
    logging.info("START: %s", name)
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    timings[name] = round(elapsed, 3)
    logging.info("DONE: %s [%.1f s]", name, elapsed)


# =============================================================================
# PURE GEOMETRY HELPERS
# =============================================================================


class UnionFind:
    """Track undirected connected groups of line parts."""

    def __init__(self, size: int) -> None:
        """Start with every part in its own group."""
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        """Return the representative of *item*'s group."""
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        """Merge the groups holding *first* and *second*."""
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def distance(a: XY, b: XY) -> float:
    """Return the planar distance between two points in CRS units."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_on_segment(p: XY, a: XY, b: XY) -> tuple[XY, float]:
    """Return the closest point on segment a-b to *p* and its fraction along a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom
    t = max(0.0, min(1.0, t))
    return (a[0] + t * dx, a[1] + t * dy), t


def segment_intersection(a: XY, b: XY, c: XY, d: XY, epsilon: float) -> tuple[str, list[XY]]:
    """Classify how segments a-b and c-d meet: NONE, POINT, or OVERLAP.

    Args:
        a: Start of the first segment.
        b: End of the first segment.
        c: Start of the second segment.
        d: End of the second segment.
        epsilon: Coordinate-precision tolerance in CRS units (not a snap distance).

    Returns:
        The kind of contact and its location(s): one point, or the two ends of
        a shared stretch.
    """
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    lr, ls = math.hypot(rx, ry), math.hypot(sx, sy)
    if lr <= epsilon or ls <= epsilon:
        return "NONE", []
    qx, qy = c[0] - a[0], c[1] - a[1]
    cross = rx * sy - ry * sx
    if abs(cross) <= epsilon * max(lr, ls):
        # Near-parallel lines are only collinear when both ends of c-d lie on a-b.
        if abs(qx * ry - qy * rx) / lr > epsilon:
            return "NONE", []
        if abs((d[0] - a[0]) * ry - (d[1] - a[1]) * rx) / lr > epsilon:
            return "NONE", []
        t0 = (qx * rx + qy * ry) / (lr * lr)
        t1 = ((d[0] - a[0]) * rx + (d[1] - a[1]) * ry) / (lr * lr)
        lo, hi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
        if hi < lo - epsilon / lr:
            return "NONE", []
        lo, hi = min(1.0, max(0.0, lo)), min(1.0, max(0.0, hi))
        first = a[0] + lo * rx, a[1] + lo * ry
        last = a[0] + hi * rx, a[1] + hi * ry
        if distance(first, last) <= epsilon:
            return "POINT", [first]
        return "OVERLAP", [first, last]
    t = (qx * sy - qy * sx) / cross
    u = (qx * ry - qy * rx) / cross
    if -epsilon / lr <= t <= 1 + epsilon / lr and -epsilon / ls <= u <= 1 + epsilon / ls:
        t = max(0.0, min(1.0, t))
        return "POINT", [(a[0] + t * rx, a[1] + t * ry)]
    return "NONE", []


def deduplicate_consecutive(points: Sequence[XY]) -> list[XY]:
    """Drop exactly repeated consecutive coordinates; closed loops stay closed."""
    output: list[XY] = []
    for point in points:
        if not output or point != output[-1]:
            output.append(point)
    return output


def levels_connect(first: str, second: str) -> bool:
    """Equal levels connect, including two unknown (blank) levels."""
    return first == second


def geometry_paths(geometry: Any) -> list[list[XY]]:
    """Return each part of a line as a list of XY tuples."""
    return [
        [(float(x), float(y)) for x, y in shapely.get_coordinates(part)]
        for part in shapely.get_parts(geometry)
    ]


def rebuild_2d(paths: Sequence[Sequence[XY]]) -> Any:
    """Rebuild a 2D line from XY paths; callers must reject Z geometry first."""
    if len(paths) == 1:
        return LineString(paths[0])
    return MultiLineString([list(path) for path in paths])


def shape_hash(geometry: Any) -> str:
    """Fingerprint a geometry so stale repair approvals can be rejected."""
    return hashlib.sha256(shapely.to_wkb(geometry)).hexdigest()


def check_line_geometry(geometry: Any) -> list[str]:
    """Return the problems that keep a line out of the diagnostic graph.

    Covers the polyline checks ArcGIS Check Geometry makes: null or empty
    shapes, non-line types, non-finite coordinates or Z values, empty or
    single-point parts, consecutive duplicate XY vertices, and invalid shapes.
    """
    if geometry is None:
        return ["NULL_GEOMETRY"]
    if geometry.is_empty:
        return ["EMPTY_GEOMETRY"]
    if geometry.geom_type not in LINE_TYPES:
        return [f"NOT_A_LINE: {geometry.geom_type}"]
    if not np.isfinite(shapely.get_coordinates(geometry)).all():
        return ["NONFINITE_COORDINATES"]
    problems: list[str] = []
    if shapely.has_z(geometry):
        z_values = shapely.get_coordinates(geometry, include_z=True)[:, 2]
        if not np.isfinite(z_values).all():
            problems.append("EMPTY_Z_VALUES")
    for part in shapely.get_parts(geometry):
        coords = shapely.get_coordinates(part)
        if len(coords) == 0:
            problems.append("EMPTY_PART")
        elif len(coords) < 2:
            problems.append("SINGLE_POINT_PART")
        elif (coords[1:] == coords[:-1]).all(axis=1).any():
            problems.append("DUPLICATE_VERTEX")
    if not problems and not shapely.is_valid(geometry):
        problems.append(f"INVALID: {shapely.is_valid_reason(geometry)}")
    return list(dict.fromkeys(problems))


def text_value(value: Any) -> str:
    """Return an attribute value as text; missing values become ""."""
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def normalize_level(value: Any) -> str:
    """Treat numeric 0 and 0.0 as the same level; keep other level codes as text."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return format(number, ".15g") if math.isfinite(number) else ""
    return text_value(value).strip()


def json_safe(value: Any) -> Any:
    """Convert numpy, pandas, and date values to plain JSON-ready values."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


# =============================================================================
# DIAGNOSTIC NETWORK
# =============================================================================


class Network:
    """Index existing vertices and segments without snapping or splitting lines."""

    def __init__(self, parts: list[Part], epsilon: float, policy: str) -> None:
        """Index eligible vertices and segments, then group connected parts.

        Args:
            parts: Continuous line parts, numbered by position (``pid``).
            epsilon: Coincidence tolerance in CRS units.
            policy: ``"ANY_VERTEX"`` or ``"ENDPOINT"``.
        """
        self.parts = parts
        self.epsilon = epsilon
        self.policy = policy
        self.v_pid: list[int] = []
        self.v_pos: list[int] = []
        vertex_xy: list[XY] = []
        seg_pid: list[int] = []
        seg_pos: list[int] = []
        seg_ends: list[tuple[XY, XY]] = []
        for part in parts:
            last = len(part.points) - 1
            for position, point in enumerate(part.points):
                if policy == "ANY_VERTEX" or position in (0, last):
                    self.v_pid.append(part.pid)
                    self.v_pos.append(position)
                    vertex_xy.append(point)
            for position, (a, b) in enumerate(zip(part.points, part.points[1:])):
                if distance(a, b) > epsilon:
                    seg_pid.append(part.pid)
                    seg_pos.append(position)
                    seg_ends.append((a, b))
        self.vertex_geoms = shapely.points(np.array(vertex_xy, dtype=float).reshape(-1, 2))
        self.vertex_tree = shapely.STRtree(self.vertex_geoms)
        self.seg_pid = seg_pid
        self.seg_pos = seg_pos
        self.seg_ends = seg_ends
        self.segment_geoms = shapely.linestrings(np.array(seg_ends, dtype=float).reshape(-1, 2, 2))
        self.segment_tree = shapely.STRtree(self.segment_geoms)
        self._part_geoms: Optional[np.ndarray] = None
        self._dangles: Optional[list[tuple[int, int, XY]]] = None

        self.components = UnionFind(len(parts))
        level_codes = {level: code for code, level in enumerate({p.level for p in parts})}
        part_level = np.array([level_codes[p.level] for p in parts], dtype=np.int64)
        v_pid = np.array(self.v_pid, dtype=np.int64)
        left, right = self.vertex_tree.query(
            self.vertex_geoms, predicate="dwithin", distance=epsilon
        )
        joined = (left < right) & (v_pid[left] != v_pid[right])
        joined &= part_level[v_pid[left]] == part_level[v_pid[right]]
        for first, second in zip(v_pid[left[joined]].tolist(), v_pid[right[joined]].tolist()):
            self.components.union(first, second)

    @property
    def part_geometries(self) -> np.ndarray:
        """Return one shapely LineString per part, in ``pid`` order."""
        if self._part_geoms is None:
            geoms = np.empty(len(self.parts), dtype=object)
            for part in self.parts:
                geoms[part.pid] = LineString(part.points)
            self._part_geoms = geoms
        return self._part_geoms

    def at_vertex(self, pid: int, point: XY, eligible_only: bool = True) -> bool:
        """Return whether *point* is an original vertex of part *pid*.

        With ``eligible_only`` False under the ENDPOINT policy, interior vertices
        count too, which separates policy mismatches from missing vertices.
        """
        if not eligible_only and self.policy != "ANY_VERTEX":
            return any(distance(p, point) <= self.epsilon for p in self.parts[pid].points)
        hits = self.vertex_tree.query(Point(point), predicate="dwithin", distance=self.epsilon)
        return any(self.v_pid[v] == pid for v in hits.tolist())

    def degrees(self, queries: Sequence[tuple[int, XY]]) -> list[int]:
        """Count the line arms meeting at each (pid, point), using that part's level."""
        if not queries:
            return []
        points = shapely.points(np.array([xy for _, xy in queries], dtype=float))
        q_idx, v_idx = self.vertex_tree.query(points, predicate="dwithin", distance=self.epsilon)
        arms: list[set[tuple[int, int]]] = [set() for _ in queries]
        for q, v in zip(q_idx.tolist(), v_idx.tolist()):
            other = self.parts[self.v_pid[v]]
            if not levels_connect(self.parts[queries[q][0]].level, other.level):
                continue
            position = self.v_pos[v]
            if position > 0:
                arms[q].add((other.pid, position - 1))
            if position < len(other.points) - 1:
                arms[q].add((other.pid, position))
        return [len(found) for found in arms]

    def degree(self, pid: int, point: XY) -> int:
        """Count line arms at one point (a closed loop's start counts twice)."""
        return self.degrees([(pid, point)])[0]

    def nearest_parts(self, point: XY, radius: float) -> dict[int, tuple[float, XY]]:
        """Return each part within *radius* with its gap and closest location."""
        found: dict[int, tuple[float, XY]] = {}
        hits = self.segment_tree.query(Point(point), predicate="dwithin", distance=radius)
        for sid in sorted(hits.tolist()):
            a, b = self.seg_ends[sid]
            closest, _ = nearest_on_segment(point, a, b)
            gap = distance(point, closest)
            pid = self.seg_pid[sid]
            if gap <= radius and (pid not in found or gap < found[pid][0]):
                found[pid] = (gap, closest)
        return found

    def dangles(self) -> list[tuple[int, int, XY]]:
        """Return degree-one endpoints as (pid, point index, point)."""
        if self._dangles is None:
            ends = [
                (part.pid, index, part.points[index])
                for part in self.parts
                for index in (0, len(part.points) - 1)
            ]
            counts = self.degrees([(pid, point) for pid, _, point in ends])
            self._dangles = [end for end, count in zip(ends, counts) if count == 1]
        return self._dangles

    def component_count(self) -> int:
        """Return the number of connected groups of parts."""
        return len({self.components.find(p.pid) for p in self.parts})


def make_parts(features: list[Feature], epsilon: float) -> tuple[list[Part], list[Issue]]:
    """Split features into graph parts, excluding problem geometry explicitly.

    Args:
        features: Selected features; ``excluded_reason`` is set on each.
        epsilon: Coincidence tolerance in CRS units.

    Returns:
        The audited parts and issues for excluded, multipart, or degenerate lines.
    """
    parts: list[Part] = []
    issues: list[Issue] = []
    for feature in features:
        geom = feature.geometry
        if feature.problems:
            feature.excluded_reason = "GEOMETRY_CHECK: " + "; ".join(feature.problems)
        elif geom.length <= epsilon:
            feature.excluded_reason = "ZERO_LENGTH"
        else:
            feature.excluded_reason = ""
        if feature.excluded_reason:
            located = [path[0] for path in geometry_paths(geom) if path]
            issues.append(
                Issue(
                    "EXCLUDED_GEOMETRY",
                    tuple(located[:1]),
                    confidence="CONFIRMED",
                    detail=feature.excluded_reason,
                    feature_key=feature.key,
                )
            )
            continue
        paths = geometry_paths(geom)
        if len(paths) > 1:
            issues.append(
                Issue(
                    "MULTIPART_FEATURE",
                    (paths[0][0],),
                    detail="Parts are separate; multipart does not imply connectivity.",
                    feature_key=feature.key,
                )
            )
        for part_number, points in enumerate(paths):
            length = sum(distance(a, b) for a, b in zip(points, points[1:]))
            if len(points) < 2 or length <= epsilon:
                issues.append(
                    Issue(
                        "DEGENERATE_PART",
                        tuple(points[:1]),
                        confidence="CONFIRMED",
                        detail="Part excluded from graph: no measurable line length.",
                        feature_key=feature.key,
                    )
                )
                continue
            parts.append(Part(len(parts), feature.key, part_number, tuple(points), feature.level))
    return parts, issues


# =============================================================================
# REVIEW SCANS
# =============================================================================


def scan_intersections(network: Network, feet_per_unit: float) -> list[Issue]:
    """Flag unnoded crossings, overlaps, self-intersections, and level mismatches."""
    issues: list[Issue] = []
    seen: set[tuple[Any, ...]] = set()
    eps = network.epsilon
    left, right = network.segment_tree.query(
        network.segment_geoms, predicate="dwithin", distance=2 * eps
    )
    keep = left < right
    left, right = left[keep], right[keep]
    order = np.lexsort((right, left))
    for i, j in zip(left[order].tolist(), right[order].tolist()):
        (a1, b1), (a2, b2) = network.seg_ends[i], network.seg_ends[j]
        kind, points = segment_intersection(a1, b1, a2, b2, eps)
        if kind == "NONE":
            continue
        a, b = network.parts[network.seg_pid[i]], network.parts[network.seg_pid[j]]
        pos_i, pos_j = network.seg_pos[i], network.seg_pos[j]
        same = a.pid == b.pid
        adjacent = same and abs(pos_i - pos_j) == 1
        closed_neighbors = (
            same
            and {pos_i, pos_j} == {0, len(a.points) - 2}
            and distance(a.points[0], a.points[-1]) <= eps
        )
        if kind == "POINT" and (adjacent or closed_neighbors):
            continue
        value: Optional[float] = None
        if kind == "OVERLAP":
            code = "SELF_OVERLAP" if same else "LINE_OVERLAP"
            detail = "Shared line length; inspect duplicate data, attribution, and grade."
            value = distance(points[0], points[-1]) * feet_per_unit
        else:
            point = points[0]
            eligible = network.at_vertex(a.pid, point) and network.at_vertex(b.pid, point)
            shared = network.at_vertex(a.pid, point, False) and network.at_vertex(
                b.pid, point, False
            )
            if same:
                code, detail = "SELF_INTERSECTION", "Nonadjacent segments intersect."
            elif eligible and levels_connect(a.level, b.level):
                continue
            elif not shared:
                code = "CROSSING_NO_SHARED_VERTEX"
                detail = "Lines meet in XY without a vertex on BOTH lines; review grade/access."
            elif not eligible:
                code = "JUNCTION_POLICY_MISMATCH"
                detail = "Shared vertices are not both endpoints under ENDPOINT policy."
            else:
                code = "JUNCTION_LEVEL_MISMATCH"
                detail = "Coincident eligible vertices have different/incomplete level values."
        if a.level and b.level and a.level != b.level:
            detail += " Known level values differ: likely intentional grade separation."
            confidence = "LIKELY_INTENTIONAL"
        else:
            confidence = "REVIEW"
        # Report a crossing once, even when several incident segments find it.
        key = (code, min(a.pid, b.pid), max(a.pid, b.pid)) + tuple(
            (round(p[0] / eps), round(p[1] / eps)) for p in points
        )
        if key not in seen:
            seen.add(key)
            issues.append(Issue(code, tuple(points), a.pid, b.pid, value, confidence, detail))
    return issues


def scan_shape_detail(parts: list[Part], feet_per_unit: float, cfg: Settings) -> list[Issue]:
    """Flag short segments, coarse vertex spacing, and short sharp reversals."""
    issues: list[Issue] = []
    for part in parts:
        for a, b in zip(part.points, part.points[1:]):
            length_ft = distance(a, b) * feet_per_unit
            if length_ft < cfg.short_segment_ft:
                issues.append(
                    Issue(
                        "SHORT_SEGMENT",
                        (a, b),
                        part.pid,
                        value_ft=length_ft,
                        detail="Short segment; may be legitimate junction detail.",
                    )
                )
            elif length_ft > cfg.coarse_segment_ft:
                issues.append(
                    Issue(
                        "COARSE_VERTEX_SPACING",
                        (a, b),
                        part.pid,
                        value_ft=length_ft,
                        confidence="SCREENING",
                        detail="Long straight segment; verify against imagery.",
                    )
                )
        for a, b, c in zip(part.points, part.points[1:], part.points[2:]):
            first, second = distance(a, b), distance(b, c)
            if min(first, second) == 0 or max(first, second) * feet_per_unit > cfg.spike_leg_ft:
                continue
            dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / (first * second)))))
            if angle >= cfg.spike_turn_degrees:
                issues.append(
                    Issue(
                        "SHORT_SPIKE",
                        (a, b, c),
                        part.pid,
                        value_ft=(first + second) * feet_per_unit,
                        detail="Short sharp reversal; check geometry or a real turn.",
                    )
                )
    return issues


class RoadContext:
    """Road polygons used only to annotate gaps, never as walking edges."""

    def __init__(
        self, geometries: Sequence[Any], fids: Sequence[str], units_per_foot: float
    ) -> None:
        """Index projected road polygons for overlap and proximity queries."""
        self.geometries = np.empty(len(geometries), dtype=object)
        self.geometries[:] = list(geometries)
        self.fids = list(fids)
        self.units_per_foot = units_per_foot
        self.tree = shapely.STRtree(self.geometries)

    def overlap_ft(self, points: Sequence[XY]) -> float:
        """Return how much of a proposed gap line lies on road area, in feet."""
        if not len(self.geometries) or len(points) < 2 or points[0] == points[-1]:
            return 0.0
        line = LineString(points)
        hits = self.tree.query(line, predicate="intersects")
        if not len(hits):
            return 0.0
        pieces = shapely.intersection(line, self.geometries[hits])
        pieces = pieces[shapely.length(pieces) > 0]
        if not len(pieces):
            return 0.0
        return float(shapely.union_all(pieces).length) / self.units_per_foot

    def near(self, point: XY, distance_ft: float) -> bool:
        """Return whether *point* is inside or within *distance_ft* of a road polygon."""
        if not len(self.geometries):
            return False
        hits = self.tree.query(
            Point(point), predicate="dwithin", distance=distance_ft * self.units_per_foot
        )
        return bool(len(hits))


def scan_gaps(
    network: Network, roads: RoadContext, units_per_foot: float, cfg: Settings
) -> list[Issue]:
    """Inspect degree-one endpoints for gaps, possible crossings, and road-edge ends."""
    issues: list[Issue] = []
    seen_pairs: set[tuple[Any, ...]] = set()
    radius_ft = (
        max(cfg.gap_review_ft, cfg.crossing_search_ft)
        if len(roads.geometries)
        else cfg.gap_review_ft
    )
    for pid, endpoint, point in network.dangles():
        part = network.parts[pid]
        issues.append(
            Issue(
                "DANGLING_ENDPOINT",
                (point,),
                pid,
                detail="Degree-one endpoint; may be an intentional sidewalk end.",
            )
        )
        if roads.near(point, cfg.road_edge_review_ft):
            issues.append(
                Issue(
                    "END_NEAR_ROAD",
                    (point,),
                    pid,
                    detail="Check crossing/curb access; proximity is not proof.",
                )
            )
        nearby = network.nearest_parts(point, radius_ft * units_per_foot)
        for other_pid, (gap, target) in nearby.items():
            if other_pid == pid or gap <= network.epsilon:
                continue  # Exact but unnoded contacts are scan_intersections' job.
            other = network.parts[other_pid]
            road_ft = roads.overlap_ft([point, target])
            if gap / units_per_foot > cfg.gap_review_ft and road_ft < cfg.min_road_overlap_ft:
                continue
            last = len(other.points) - 1
            target_end = (
                0
                if distance(target, other.points[0]) <= distance(target, other.points[last])
                else last
            )
            to_endpoint = distance(target, other.points[target_end]) <= network.epsilon
            if to_endpoint:
                pair = tuple(sorted([(pid, endpoint), (other_pid, target_end)]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
            crossing = road_ft >= cfg.min_road_overlap_ft
            if crossing:
                code = "POSSIBLE_MISSING_CROSSING"
            else:
                code = "NEAR_ENDPOINT_GAP" if to_endpoint else "NEAR_EDGE_GAP"
            detail = "Nearby endpoint is disconnected here. " + (
                "Candidate crosses road polygon area; verify grade and permitted crossing."
                if crossing
                else "Check intended continuity, barriers, grade, and alternative targets."
            )
            if network.components.find(pid) == network.components.find(other_pid):
                detail += " Both parts already connect elsewhere in the retained graph."
            if part.level and other.level and part.level != other.level:
                detail += " Known levels differ; do not snap."
            issues.append(
                Issue(
                    code,
                    (point, target),
                    pid,
                    other_pid,
                    gap / units_per_foot,
                    "REVIEW",
                    detail,
                    road_ft,
                )
            )
    return issues


def scope_issue(issue: Issue, study: Any, features: dict[str, Feature]) -> bool:
    """Keep issues that touch the study area; unlocated issues are always kept."""
    if not issue.points:
        return True
    if issue.feature_key in features:
        # A feature's first point can lie outside the study area while its
        # middle is inside, so test the whole feature.
        return bool(shapely.intersects(study, features[issue.feature_key].geometry))
    return bool(shapely.intersects(study, issue.geometry()))


def identify_issue(issue: Issue, parts: list[Part]) -> str:
    """Return a deterministic ID from issue type, lineage, and location."""
    owners = [
        (parts[i].feature_key, parts[i].part_number) for i in (issue.pid, issue.other_pid) if i >= 0
    ]
    data = [
        issue.kind,
        sorted(owners),
        issue.feature_key,
        [[round(x, 8), round(y, 8)] for x, y in issue.points],
    ]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]


# =============================================================================
# CONSERVATIVE REPAIRS -- OPT-IN, WORKING COPY ONLY
# =============================================================================


def remove_repeated_vertices(features: list[Feature]) -> list[dict[str, Any]]:
    """Remove exact consecutive repeats from 2D lines whose only problem is a repeat."""
    changes: list[dict[str, Any]] = []
    for feature in features:
        if feature.has_z or not feature.problems:
            continue
        if any(not p.endswith("DUPLICATE_VERTEX") for p in feature.problems):
            continue
        raw = geometry_paths(feature.geometry)
        cleaned = [deduplicate_consecutive(path) for path in raw]
        if raw == cleaned or any(len(path) < 2 for path in cleaned):
            continue
        revised = rebuild_2d(cleaned)
        if geometry_paths(revised) != cleaned or not revised.equals(feature.geometry):
            logging.warning("Skipped duplicate cleanup that changed geometry: %s", feature.key)
            continue
        changes.append(
            {
                "SOURCE_KEY": feature.key,
                "ACTION": "REMOVE_REPEATED_VERTICES",
                "MOVE_FT": 0.0,
                "OLD_WKT": feature.geometry.wkt,
                "NEW_WKT": revised.wkt,
            }
        )
        feature.geometry = revised
        feature.problems = check_line_geometry(revised)
    return changes


def repair_candidates(
    issues: list[Issue],
    network: Network,
    features: dict[str, Feature],
    units_per_foot: float,
    cfg: Settings,
) -> list[dict[str, Any]]:
    """Offer only tiny, unambiguous, same-level 2D endpoint pairs for approval."""
    output: list[dict[str, Any]] = []
    for issue in issues:
        if issue.kind != "NEAR_ENDPOINT_GAP" or issue.value_ft is None:
            continue
        if issue.value_ft > cfg.max_approved_snap_ft or issue.road_ft > 0:
            continue
        first, second = network.parts[issue.pid], network.parts[issue.other_pid]
        a, b = features[first.feature_key], features[second.feature_key]
        if a.key == b.key or a.level != b.level or a.has_z or b.has_z:
            continue
        ends: list[str] = []
        safe = True
        for part, point, other_pid in (
            (first, issue.points[0], second.pid),
            (second, issue.points[-1], first.pid),
        ):
            last = len(part.points) - 1
            endpoint = (
                0 if distance(part.points[0], point) <= distance(part.points[last], point) else last
            )
            if distance(part.points[endpoint], point) > network.epsilon:
                safe = False
            if network.degree(part.pid, point) != 1:
                safe = False
            nearby = network.nearest_parts(point, cfg.gap_review_ft * units_per_foot)
            if set(nearby) - {part.pid} != {other_pid}:
                safe = False
            ends.append("START" if endpoint == 0 else "END")
        if safe:
            output.append(
                {
                    "APPROVE": "",
                    "ISSUE_ID": issue.issue_id,
                    "SOURCE_KEY": a.key,
                    "SOURCE_PART": first.part_number,
                    "SOURCE_ENDPOINT": ends[0],
                    "TARGET_KEY": b.key,
                    "TARGET_PART": second.part_number,
                    "TARGET_ENDPOINT": ends[1],
                    "GAP_FT": issue.value_ft,
                    "SOURCE_HASH": shape_hash(a.geometry),
                    "TARGET_HASH": shape_hash(b.geometry),
                    "REASON": "Verify same level, no barrier, and intended continuity.",
                }
            )
    return output


def read_approvals(path: str) -> list[dict[str, str]]:
    """Read APPROVE = YES rows from an edited repair candidates CSV."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in REPAIR_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Approved snaps CSV is missing columns {missing}: {path}")
    approved = table.loc[table["APPROVE"].str.strip().str.upper() == "YES"]
    return [{str(k): str(v) for k, v in row.items()} for row in approved.to_dict("records")]


def apply_approved_snaps(
    candidates: list[dict[str, Any]],
    features: dict[str, Feature],
    network: Network,
    approvals: list[dict[str, str]],
    units_per_foot: float,
) -> list[dict[str, Any]]:
    """Validate every approval, then move approved endpoints in the working copy.

    Approvals must still match a current candidate (including both shape
    fingerprints), may not share a feature, and may not create any new contact
    other than the intended target endpoint. Nothing is applied unless all
    approvals pass.

    Args:
        candidates: Current repair candidates from this run.
        features: Selected features keyed by SOURCE_KEY.
        network: The network built before the snaps.
        approvals: APPROVE = YES rows from the approved snaps CSV.
        units_per_foot: CRS units per foot.

    Returns:
        One repair-log row per applied snap.

    Raises:
        ValueError: An approval is stale, overlapping, or unsafe.
    """
    lookup = {str(c["ISSUE_ID"]): c for c in candidates}
    used: set[str] = set()
    pending: list[tuple[Feature, Any, dict[str, Any]]] = []
    eps = network.epsilon
    for row in approvals:
        current = lookup.get(row["ISSUE_ID"])
        if current is None:
            raise ValueError(f"Approved candidate is no longer eligible: {row['ISSUE_ID']}")
        for name in REPAIR_COLUMNS:
            if name not in ("APPROVE", "REASON", "GAP_FT") and str(current[name]) != row[name]:
                raise ValueError(f"Stale approval field {name} for {row['ISSUE_ID']}")
        keys = {row["SOURCE_KEY"], row["TARGET_KEY"]}
        if keys & used:
            raise ValueError("Approved snaps share a source/target feature; run separate rounds.")
        used.update(keys)
        source, target = features[row["SOURCE_KEY"]], features[row["TARGET_KEY"]]
        paths, targets = geometry_paths(source.geometry), geometry_paths(target.geometry)
        part_no, target_no = int(row["SOURCE_PART"]), int(row["TARGET_PART"])
        endpoint = 0 if row["SOURCE_ENDPOINT"] == "START" else len(paths[part_no]) - 1
        target_end = 0 if row["TARGET_ENDPOINT"] == "START" else len(targets[target_no]) - 1
        old, new = paths[part_no][endpoint], targets[target_no][target_end]
        neighbor = paths[part_no][1 if endpoint == 0 else -2]
        if distance(new, neighbor) <= eps:
            raise ValueError(f"Approved snap would collapse its end segment: {row['ISSUE_ID']}")
        moving_position = 0 if endpoint == 0 else len(paths[part_no]) - 2
        hits = network.segment_tree.query(
            LineString([neighbor, new]), predicate="dwithin", distance=2 * eps
        )
        for sid in sorted(hits.tolist()):
            owner = network.parts[network.seg_pid[sid]]
            if (
                owner.feature_key == source.key
                and owner.part_number == part_no
                and network.seg_pos[sid] == moving_position
            ):
                continue
            seg_a, seg_b = network.seg_ends[sid]
            kind, contacts = segment_intersection(neighbor, new, seg_a, seg_b, eps)
            if kind == "NONE":
                continue
            allowed_target = (
                kind == "POINT"
                and owner.feature_key == target.key
                and all(distance(p, new) <= eps for p in contacts)
            )
            old_kind, old_contacts = segment_intersection(neighbor, old, seg_a, seg_b, eps)
            unchanged = kind == old_kind == "POINT" and all(
                any(distance(p, q) <= eps for q in old_contacts) for p in contacts
            )
            if not allowed_target and not unchanged:
                raise ValueError(f"Snap creates an unintended contact: {row['ISSUE_ID']}")
        paths[part_no][endpoint] = new
        revised = rebuild_2d(paths)
        if geometry_paths(revised) != paths:
            raise ValueError(f"Rebuilding the snapped line changed it: {row['ISSUE_ID']}")
        change = {
            "SOURCE_KEY": source.key,
            "ACTION": "APPROVED_ENDPOINT_SNAP",
            "MOVE_FT": distance(old, new) / units_per_foot,
            "OLD_WKT": source.geometry.wkt,
            "NEW_WKT": revised.wkt,
        }
        pending.append((source, revised, change))
    for source, revised, _ in pending:
        source.geometry = revised
    return [change for _, _, change in pending]


# =============================================================================
# INPUTS: GTFS STOPS, NETWORK SOURCES, ROAD POLYGONS
# =============================================================================


def read_gtfs_table(
    gtfs_path: str, filename: str, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Read one GTFS table as text, keeping IDs such as "NA" and leading zeroes.

    Args:
        gtfs_path: GTFS folder or .zip (members at the root or in one subfolder).
        filename: Table name, e.g. ``"stops.txt"``.
        columns: Optional subset of columns to read.

    Returns:
        The table with every column as text and blanks as "".

    Raises:
        OSError: The table is missing from a folder feed.
        ValueError: The zip is unreadable, the table is missing or ambiguous in
            it, or a requested column is absent.
    """
    path = Path(gtfs_path)
    use: Any = None if columns is None else (lambda name: name.strip() in columns)
    if path.is_dir():
        table_path = path / filename
        if not table_path.is_file():
            raise OSError(f"{filename} not found in GTFS folder '{path}'.")
        table = pd.read_csv(
            table_path, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=use
        )
    else:
        try:
            with zipfile.ZipFile(path) as archive:
                matches = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == filename]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected exactly one {filename} in GTFS zip '{path}'; "
                        f"found {len(matches)}."
                    )
                with archive.open(matches[0]) as handle:
                    table = pd.read_csv(
                        handle, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=use
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError(f"'{path}' is not a GTFS folder or a valid .zip file.") from exc
    table.columns = [str(c).strip() for c in table.columns]
    if columns is not None:
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")
    return table


def select_gtfs_stops(cfg: Settings) -> pd.DataFrame:
    """Select platform stops, optionally limited by route short names and stop IDs.

    Returns:
        One row per selected stop with ``stop_id``, ``stop_name``, ``lat``, ``lon``.

    Raises:
        ValueError: Blank or duplicate stop IDs, bad coordinates on a selected
            stop, unknown route names or stop IDs, or no stops left.
    """
    stops = read_gtfs_table(cfg.gtfs_path, "stops.txt")
    for column in ("stop_id", "stop_lat", "stop_lon"):
        if column not in stops.columns:
            raise ValueError(f"stops.txt is missing required column '{column}'.")
    if stops.empty:
        raise ValueError("stops.txt has no rows.")
    ids = stops["stop_id"]
    bad_ids = ids.loc[(ids == "") | ids.duplicated()]
    if not bad_ids.empty:
        raise ValueError(f"Blank or duplicate stop_id in stops.txt: {bad_ids.iloc[0]!r}")
    keep = pd.Series(True, index=stops.index)
    if "location_type" in stops.columns:
        keep &= stops["location_type"].str.strip().isin(["", "0"])
    allowed: Optional[set[str]] = None
    if cfg.route_short_names:
        routes = read_gtfs_table(cfg.gtfs_path, "routes.txt", ["route_id", "route_short_name"])
        trips = read_gtfs_table(cfg.gtfs_path, "trips.txt", ["trip_id", "route_id"])
        times = read_gtfs_table(cfg.gtfs_path, "stop_times.txt", ["trip_id", "stop_id"])
        names = routes["route_short_name"].str.strip()
        missing = sorted(cfg.route_short_names - set(names))
        if missing:
            raise ValueError(f"Route short names absent from GTFS: {missing}")
        route_ids = set(routes.loc[names.isin(cfg.route_short_names), "route_id"])
        trip_ids = set(trips.loc[trips["route_id"].isin(route_ids), "trip_id"])
        allowed = set(times.loc[times["trip_id"].isin(trip_ids), "stop_id"])
        if allowed - set(ids):
            raise ValueError("Selected route stop_times reference stop IDs missing from stops.txt.")
        keep &= ids.isin(allowed)
    if cfg.stop_ids:
        keep &= ids.isin(cfg.stop_ids)
    selected = stops.loc[keep]
    if cfg.stop_ids - set(selected["stop_id"]):
        raise ValueError("Some requested stop IDs are absent or excluded by other filters.")
    if selected.empty:
        raise ValueError("No GTFS platform stops remain after filtering.")
    lat = pd.to_numeric(selected["stop_lat"], errors="coerce")
    lon = pd.to_numeric(selected["stop_lon"], errors="coerce")
    bad = ~(lat.between(-90, 90) & lon.between(-180, 180))
    if bad.any():
        raise ValueError(
            f"Missing or out-of-range coordinates for stop {selected.loc[bad, 'stop_id'].iloc[0]}."
        )
    names_column = selected["stop_name"] if "stop_name" in selected.columns else ""
    return pd.DataFrame(
        {"stop_id": selected["stop_id"], "stop_name": names_column, "lat": lat, "lon": lon}
    ).reset_index(drop=True)


def project_stops(stops: pd.DataFrame, crs: CRS) -> gpd.GeoDataFrame:
    """Turn stop coordinates into projected points in the analysis CRS."""
    points = gpd.GeoDataFrame(
        {"STOP_ID": stops["stop_id"], "STOP_NAME": stops["stop_name"]},
        geometry=gpd.points_from_xy(stops["lon"], stops["lat"]),
        crs="EPSG:4326",
    ).to_crs(crs)
    if not np.isfinite(shapely.get_coordinates(points.geometry.to_numpy())).all():
        raise ValueError("Some stops could not be projected into the analysis CRS.")
    logging.info("Selected %d GTFS platform stops.", len(points))
    return points


def build_scope(
    stops: gpd.GeoDataFrame, cfg: Settings, units_per_foot: float
) -> tuple[gpd.GeoDataFrame, dict[float, Any], Any, Any]:
    """Build per-stop review buffers, the dissolved scope per radius, and the context.

    Returns:
        Stop buffers, the dissolved area per radius, the study area (largest
        radius), and the context area (study area plus the context margin).
    """
    stop_points = stops.geometry.to_numpy()
    frames: list[gpd.GeoDataFrame] = []
    scopes: dict[float, Any] = {}
    for radius in sorted(cfg.stop_buffer_miles):
        polygons = shapely.buffer(stop_points, radius * FEET_PER_MILE * units_per_foot)
        scopes[radius] = shapely.union_all(polygons)
        frames.append(
            gpd.GeoDataFrame(
                {"STOP_ID": stops["STOP_ID"].to_numpy(), "RADIUS_MI": radius},
                geometry=polygons,
                crs=stops.crs,
            )
        )
    buffers = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=stops.crs)
    study = scopes[max(cfg.stop_buffer_miles)]
    context = shapely.buffer(study, cfg.context_margin_ft * units_per_foot)
    shapely.prepare(study)
    shapely.prepare(context)
    return buffers, scopes, study, context


def read_layer(path: str, layer: str = "", where: str = "") -> gpd.GeoDataFrame:
    """Read a vector layer with its data-source feature IDs as the index."""
    options: dict[str, Any] = {"engine": "pyogrio", "fid_as_index": True}
    if layer:
        options["layer"] = layer
    if where:
        options["where"] = where
    return gpd.read_file(path, **options)


def select_near(layer: gpd.GeoDataFrame, context: Any, crs: CRS) -> gpd.GeoDataFrame:
    """Return rows of *layer* (in its own CRS) that intersect the context area."""
    if layer.crs is None:
        raise ValueError("Every input layer needs a defined CRS.")
    in_layer_crs = gpd.GeoSeries([context], crs=crs).to_crs(layer.crs).iloc[0]
    hits = np.sort(layer.sindex.query(in_layer_crs, predicate="intersects"))
    return layer.iloc[hits]


def load_sources(
    cfg: Settings, context: Any, crs: CRS, preflight: list[dict[str, Any]]
) -> list[Feature]:
    """Read each network layer and keep whole lines that reach the context area.

    Geometry problems are checked on the source and projected shapes and
    recorded in *preflight*; features without geometry cannot be located, so
    they are listed there for the whole layer.

    Args:
        cfg: Run settings.
        context: Context polygon in the analysis CRS.
        crs: Analysis CRS.
        preflight: Rows of the geometry preflight CSV, appended in place.

    Returns:
        Selected features in source order.

    Raises:
        ValueError: A layer has no CRS or non-line shapes, a configured field is
            missing, IDs are blank or duplicated, or projection fails.
    """
    features: list[Feature] = []
    for spec in cfg.network_sources:
        source = read_layer(spec.path, spec.layer, spec.where)
        if source.crs is None:
            raise ValueError(f"Network layer '{spec.label}' has no CRS; define it in GIS first.")
        for field_name in (spec.id_field, spec.level_field):
            if field_name and field_name not in source.columns:
                raise ValueError(f"Field '{field_name}' not found in {spec.path}")
        absent = source.geometry.isna() | source.geometry.is_empty
        for fid in source.index[absent]:
            preflight.append(
                {
                    "SOURCE_LABEL": spec.label,
                    "SOURCE_FID": str(fid),
                    "PROBLEM": "NULL_OR_EMPTY_GEOMETRY",
                    "SCOPE": "UNLOCATED",
                }
            )
        if absent.any():
            logging.warning(
                "%s: %d feature(s) have no geometry and cannot be located; see the preflight CSV.",
                spec.label,
                int(absent.sum()),
            )
        located = source.loc[~absent]
        other_types = sorted(set(located.geom_type) - set(LINE_TYPES))
        if other_types:
            raise ValueError(f"Network layer '{spec.label}' must hold lines; found {other_types}.")
        selected = select_near(located, context, crs)
        logging.info(
            "%s: selected %d of %d features near stops.", spec.label, len(selected), len(source)
        )
        same_crs = CRS.from_user_input(selected.crs) == crs
        projected = selected.geometry if same_crs else selected.geometry.to_crs(crs)
        attr_columns = [c for c in selected.columns if c != selected.geometry.name]
        records = selected[attr_columns].to_dict("records")
        seen_ids: set[str] = set()
        kept = 0
        for fid, source_geom, geom, attrs in zip(
            selected.index, selected.geometry, projected, records
        ):
            identity = attrs.get(spec.id_field) if spec.id_field else fid
            source_id = text_value(identity)
            if not source_id or source_id in seen_ids:
                raise ValueError(
                    f"Blank or duplicate source ID in selected {spec.label}: {source_id!r}"
                )
            seen_ids.add(source_id)
            problems = check_line_geometry(source_geom)
            if not same_crs:
                problems += [
                    "PROJECTED: " + p for p in check_line_geometry(geom) if p not in problems
                ]
            if "PROJECTED: NONFINITE_COORDINATES" in problems:
                raise ValueError(
                    f"Projecting {spec.label} feature {fid} produced non-finite coordinates; "
                    "check the layer's CRS."
                )
            for problem in problems:
                preflight.append(
                    {
                        "SOURCE_LABEL": spec.label,
                        "SOURCE_FID": str(fid),
                        "PROBLEM": problem,
                        "SCOPE": "PROJECTED" if problem.startswith("PROJECTED") else "SOURCE",
                    }
                )
            if not shapely.intersects(context, geom):
                continue
            level = normalize_level(attrs.get(spec.level_field)) if spec.level_field else ""
            features.append(
                Feature(
                    f"{spec.label}:{source_id}",
                    spec.label,
                    str(fid),
                    source_id,
                    level,
                    geom,
                    {str(k): json_safe(v) for k, v in attrs.items()},
                    problems,
                )
            )
            kept += 1
        logging.info("%s: kept %d whole features in the context area.", spec.label, kept)
    z_count = sum(f.has_z for f in features)
    if z_count:
        logging.warning("%d Z-enabled features use XY diagnostics; review grade.", z_count)
    return features


def load_roads(cfg: Settings, context: Any, crs: CRS, units_per_foot: float) -> RoadContext:
    """Load valid road polygons that reach the context area (empty when unset)."""
    if not cfg.road_polygons_path:
        return RoadContext([], [], units_per_foot)
    roads = read_layer(cfg.road_polygons_path, cfg.road_polygons_layer, cfg.road_where)
    roads = roads.loc[~(roads.geometry.isna() | roads.geometry.is_empty)]
    other_types = sorted(set(roads.geom_type) - set(POLYGON_TYPES))
    if other_types:
        raise ValueError(f"ROAD_POLYGONS_PATH must hold polygons; found {other_types}.")
    selected = select_near(roads, context, crs)
    projected = selected.geometry.to_crs(crs)
    keep = shapely.intersects(context, projected.to_numpy())
    projected = projected.loc[keep]
    invalid = projected.index[~projected.is_valid].tolist()
    if invalid:
        raise ValueError(
            f"{len(invalid)} road polygon(s) are invalid (FIDs {invalid[:5]}); repair them in GIS."
        )
    logging.info("Loaded %d road polygons in the context area.", len(projected))
    if projected.empty:
        logging.warning("No road polygons in context; road-based crossing checks are off.")
    return RoadContext(projected.to_list(), [str(f) for f in projected.index], units_per_foot)


# =============================================================================
# COMPONENTS AND STOP-LEVEL COVERAGE
# =============================================================================


def component_outputs(
    network: Network, study: Any, context: Any, units_per_foot: float, cfg: Settings
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, Any]], list[Issue]]:
    """Group parts into components and flag small groups away from the context edge.

    Returns:
        Component ID per part (by ``pid``), the per-part component table, one
        summary row per component, and small-group / isolated-part issues.
    """
    geoms = network.part_geometries
    lengths_ft = shapely.length(geoms) / units_per_foot
    inside = shapely.contains(study, geoms)
    study_ft = np.where(inside, lengths_ft, 0.0)
    partial = ~inside & shapely.intersects(study, geoms)
    if partial.any():
        study_ft[partial] = shapely.length(shapely.intersection(geoms[partial], study)) / (
            units_per_foot
        )
    at_edge = ~shapely.contains_properly(context, geoms)

    groups: dict[int, list[Part]] = defaultdict(list)
    for part in network.parts:
        groups[network.components.find(part.pid)].append(part)
    # Stable numbering by source keys and part numbers, not by read order.
    ordered = sorted(groups.values(), key=lambda g: min((p.feature_key, p.part_number) for p in g))
    part_component = np.zeros(len(network.parts), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    issues: list[Issue] = []
    for component_id, group in enumerate(ordered, 1):
        pids = [p.pid for p in group]
        total = float(lengths_ft[pids].sum())
        in_study = float(study_ft[pids].sum())
        edge = bool(at_edge[pids].any())
        part_component[pids] = component_id
        summaries.append(
            {
                "COMP_ID": component_id,
                "PART_COUNT": len(group),
                "FEATURE_COUNT": len({p.feature_key for p in group}),
                "CONTEXT_LENGTH_FT": total,
                "STUDY_LENGTH_FT": in_study,
                "CONTEXT_EDGE": int(edge),
            }
        )
        for part in group:
            rows.append(
                {
                    "SOURCE_KEY": part.feature_key,
                    "PART_NO": part.part_number,
                    "COMP_ID": component_id,
                    "CONTEXT_EDGE": int(edge),
                    "LENGTH_FT": float(lengths_ft[part.pid]),
                    "STUDY_FT": float(study_ft[part.pid]),
                    "geometry": geoms[part.pid],
                }
            )
        if in_study > 0 and not edge and total <= cfg.small_component_length_ft:
            representative = next(p for p in group if study_ft[p.pid] > 0)
            issues.append(
                Issue(
                    "SMALL_DISCONNECTED_GROUP",
                    representative.points,
                    representative.pid,
                    value_ft=total,
                    detail="Small separate component; review nearby links.",
                )
            )
        elif in_study > 0 and not edge and len(group) == 1:
            issues.append(
                Issue(
                    "ISOLATED_PART",
                    group[0].points,
                    group[0].pid,
                    value_ft=total,
                    detail="Part connects to no other audited part.",
                )
            )
    return part_component, pd.DataFrame(rows), summaries, issues


def stop_outputs(
    network: Network,
    part_component: np.ndarray,
    issues: list[Issue],
    stops: gpd.GeoDataFrame,
    buffers: gpd.GeoDataFrame,
    units_per_foot: float,
    excluded_count: int,
    cfg: Settings,
) -> tuple[list[dict[str, Any]], list[Issue]]:
    """Measure mapped coverage in each stop buffer; nearby lines are not verified access.

    Returns:
        One summary row per stop and radius, and stop-level issues (far from
        the network, low local coverage) for the largest radius.
    """
    part_geoms = network.part_geometries
    buffer_geoms = buffers.geometry.to_numpy()
    stop_row = {sid: i for i, sid in enumerate(stops["STOP_ID"])}
    stop_points = stops.geometry.to_numpy()
    buffer_stop = np.array([stop_row[s] for s in buffers["STOP_ID"]], dtype=np.int64)

    b_idx, p_idx = shapely.STRtree(part_geoms).query(buffer_geoms, predicate="intersects")
    clipped_ft = (
        shapely.length(shapely.intersection(part_geoms[p_idx], buffer_geoms[b_idx]))
        / units_per_foot
    )
    keep = clipped_ft > 0  # Lines that only touch a buffer at a point do not count.
    b_idx, p_idx, clipped_ft = b_idx[keep], p_idx[keep], clipped_ft[keep]
    pairs = pd.DataFrame(
        {
            "buffer": b_idx,
            "pid": p_idx,
            "length_ft": clipped_ft,
            "gap_ft": shapely.distance(stop_points[buffer_stop[b_idx]], part_geoms[p_idx])
            / units_per_foot,
            "component": part_component[p_idx],
        }
    )
    grouped = pairs.groupby("buffer")
    lengths = grouped["length_ft"].sum()
    component_counts = grouped["component"].nunique()
    nearest = pairs.sort_values(["buffer", "gap_ft", "pid"]).drop_duplicates("buffer")
    nearest_gap = dict(zip(nearest["buffer"].tolist(), nearest["gap_ft"].tolist()))
    nearest_component = dict(zip(nearest["buffer"].tolist(), nearest["component"].tolist()))

    mapped = [g for g in (i.geometry() for i in issues) if g is not None]
    issue_tree = shapely.STRtree(np.array(mapped, dtype=object))
    hit_buffers, _ = issue_tree.query(buffer_geoms, predicate="intersects")
    issue_counts = np.bincount(hit_buffers, minlength=len(buffers))

    largest = max(cfg.stop_buffer_miles)
    rows: list[dict[str, Any]] = []
    added: list[Issue] = []
    for i, (sid, radius, polygon) in enumerate(
        zip(buffers["STOP_ID"], buffers["RADIUS_MI"], buffer_geoms)
    ):
        length_ft = float(lengths.get(i, 0.0))
        area_sq_mi = polygon.area / (units_per_foot * FEET_PER_MILE) ** 2
        density = length_ft / FEET_PER_MILE / area_sq_mi if area_sq_mi else 0.0
        nearest_ft = nearest_gap.get(i)
        rows.append(
            {
                "STOP_ID": sid,
                "STOP_NAME": stops["STOP_NAME"].iloc[buffer_stop[i]],
                "RADIUS_MI": radius,
                "MAPPED_LENGTH_FT": length_ft,
                "BUFFER_SQ_MI": area_sq_mi,
                "DENSITY_MI_PER_SQ_MI": density,
                "COMPONENT_COUNT": int(component_counts.get(i, 0)),
                "NEAREST_LINE_FT": nearest_ft,
                "NEAREST_COMPONENT": nearest_component.get(i),
                "PRE_STOP_SCREEN_ISSUES": int(issue_counts[i]),
                "GEOMETRY_EXCLUDED_IN_RUN": excluded_count,
                "STOP_CONNECTION_VERIFIED": "NO",
            }
        )
        if radius != largest:
            continue
        point = shapely.get_coordinates(stop_points[buffer_stop[i]])[0]
        location = ((float(point[0]), float(point[1])),)
        if nearest_ft is None or nearest_ft > cfg.stop_near_network_ft:
            added.append(
                Issue(
                    "STOP_FAR_FROM_NETWORK",
                    location,
                    value_ft=nearest_ft,
                    detail=f"Stop {sid}: no audited line within {cfg.stop_near_network_ft:g} ft.",
                    feature_key=f"GTFS:{sid}",
                )
            )
        if cfg.low_density_mi_per_sq_mi > 0 and density < cfg.low_density_mi_per_sq_mi:
            added.append(
                Issue(
                    "LOW_LOCAL_COVERAGE",
                    location,
                    confidence="SCREENING",
                    detail=(
                        f"Stop {sid}: {density:.3f} mi/sq mi in {radius:.3f}-mi radius; "
                        "review local context."
                    ),
                    feature_key=f"GTFS:{sid}",
                )
            )
    return rows, added


# =============================================================================
# OUTPUT WRITERS
# =============================================================================


def network_layers(
    features: list[Feature], crs: CRS, use_original: bool
) -> dict[str, gpd.GeoDataFrame]:
    """Return network rows split into 2D and Z layers so no Z values are invented."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in features:
        groups["_z" if feature.has_z else ""].append(
            {
                "SOURCE_KEY": feature.key,
                "SOURCE_LABEL": feature.label,
                "SOURCE_FID": feature.fid,
                "SOURCE_ID": feature.source_id,
                "LEVEL_VALUE": feature.level,
                "QA_STATE": feature.excluded_reason or "XY_DIAGNOSTIC_ONLY",
                "geometry": feature.original if use_original else feature.geometry,
            }
        )
    if not groups:
        groups[""] = []
    columns = ["SOURCE_KEY", "SOURCE_LABEL", "SOURCE_FID", "SOURCE_ID", "LEVEL_VALUE", "QA_STATE"]
    return {
        suffix: frame_of(pd.DataFrame(rows, columns=[*columns, "geometry"]), crs)
        for suffix, rows in sorted(groups.items())
    }


def frame_of(table: pd.DataFrame, crs: CRS) -> gpd.GeoDataFrame:
    """Wrap a table with a ``geometry`` column as a GeoDataFrame in *crs*."""
    return gpd.GeoDataFrame(table, geometry=gpd.GeoSeries(table["geometry"], crs=crs), crs=crs)


def issue_record(issue: Issue, parts: list[Part]) -> dict[str, Any]:
    """Build a source-linked issue row shared by the CSV and the map layers."""
    first = parts[issue.pid] if issue.pid >= 0 else None
    second = parts[issue.other_pid] if issue.other_pid >= 0 else None
    return {
        "ISSUE_ID": issue.issue_id,
        "ISSUE_TYPE": issue.kind,
        "CONFIDENCE": issue.confidence,
        "SOURCE_KEY": first.feature_key if first else issue.feature_key,
        "OTHER_KEY": second.feature_key if second else "",
        "PART_NO": first.part_number if first else None,
        "OTHER_PART": second.part_number if second else None,
        "VALUE_FT": issue.value_ft,
        "ROAD_FT": issue.road_ft,
        "DETAIL": issue.detail,
        "REVIEW_STATUS": "OPEN",
    }


def issue_tables(
    issues: list[Issue], parts: list[Part], crs: CRS
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return the issues CSV table and the point and line review layers."""
    csv_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    for issue in issues:
        record = issue_record(issue, parts)
        geometry = issue.geometry()
        if isinstance(geometry, Point):
            point_rows.append({**record, "geometry": geometry})
        elif geometry is not None:
            line_rows.append({**record, "geometry": geometry})
        csv_rows.append(
            {
                **record,
                "X": issue.points[0][0] if issue.points else None,
                "Y": issue.points[0][1] if issue.points else None,
                "POINTS_JSON": json.dumps([list(p) for p in issue.points]),
            }
        )
    table = pd.DataFrame(csv_rows, columns=[*ISSUE_COLUMNS, "X", "Y", "POINTS_JSON"])
    points = frame_of(pd.DataFrame(point_rows, columns=[*ISSUE_COLUMNS, "geometry"]), crs)
    lines = frame_of(pd.DataFrame(line_rows, columns=[*ISSUE_COLUMNS, "geometry"]), crs)
    return table, points, lines


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    """Write rows to an Excel-friendly UTF-8 CSV, with a header even when empty."""
    pd.DataFrame(list(rows), columns=list(columns)).to_csv(path, index=False, encoding="utf-8-sig")


def radius_suffix(radius: float) -> str:
    """Return a layer-name-safe suffix for a radius, e.g. 0.25 -> "0p25"."""
    return repr(float(radius)).replace(".", "p").replace("-", "m")


def notebook_config_block() -> Optional[str]:
    """Return the CONFIG block from the latest notebook cell that holds one, if any."""
    get_ipython = getattr(sys.modules.get("IPython"), "get_ipython", None)
    shell = get_ipython() if get_ipython is not None else None
    if shell is None:
        return None
    for cell in reversed(shell.user_ns.get("In", [])):
        lines = cell.splitlines()
        stripped = [line.strip() for line in lines]
        if CONFIG_BEGIN_MARKER in stripped:
            start = stripped.index(CONFIG_BEGIN_MARKER)
            if CONFIG_END_MARKER in stripped[start:]:
                return "\n".join(lines[start + 1 : stripped.index(CONFIG_END_MARKER, start)])
    return None


# Canonical version lives in utils/run_log.py — keep this copy in sync.
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


def write_run_log(output_file: Path, summary_lines: Sequence[str]) -> bool:
    """Write a ``_runlog.txt`` sidecar next to *output_file*.

    The log records the run time, the source script, a short run summary, and
    the CONFIGURATION block verbatim. In a notebook, the latest cell holding
    the CONFIG markers is used; otherwise this script's file on disk.

    Args:
        output_file: The review GeoPackage this run produced.
        summary_lines: Short "label: value" lines describing the run.

    Returns:
        True if the log was written, False otherwise.
    """
    log_path = output_file.with_name(f"{output_file.stem}_runlog.txt")
    file_attr = globals().get("__file__")
    try:
        if file_attr is not None:
            source_label = str(Path(file_attr).resolve())
            config_text = extract_config_block(Path(file_attr))
        else:
            source_label = "<Jupyter cell>"
            notebook_text = notebook_config_block()
            if notebook_text is None:
                raise ValueError("no notebook cell contains the CONFIG markers")
            config_text = notebook_text
    except (OSError, ValueError) as exc:
        logging.error("Could not extract the config block for the run log: %s", exc)
        return False
    lines = [
        "=" * 72,
        "WALKING NETWORK QA RUN LOG",
        "=" * 72,
        f"Run timestamp:  {datetime.now().isoformat(timespec='seconds')}",
        f"Review output:  {output_file}",
        f"Source script:  {source_label}",
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
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to '%s'.", log_path)
    return True


def settings_record(cfg: Settings) -> dict[str, Any]:
    """Return the run settings as JSON-ready values for the summary."""
    record: dict[str, Any] = {}
    for key, value in cfg._asdict().items():
        if key == "network_sources":
            record[key] = [spec._asdict() for spec in value]
        elif isinstance(value, frozenset):
            record[key] = sorted(value)
        elif isinstance(value, Path):
            record[key] = str(value)
        else:
            record[key] = value
    return record


# =============================================================================
# PIPELINE
# =============================================================================


def resolve_analysis_crs(cfg: Settings) -> tuple[CRS, float]:
    """Return the analysis CRS and CRS units per foot.

    Raises:
        ConfigError: The CRS is not a local projected CRS with linear units.
    """
    if cfg.analysis_crs:
        try:
            crs = CRS.from_user_input(cfg.analysis_crs)
        except CRSError as exc:
            raise ConfigError(f"ANALYSIS_CRS is not a recognized CRS: {exc}") from exc
    else:
        first = cfg.network_sources[0]
        sample = gpd.read_file(first.path, layer=first.layer or None, rows=1, engine="pyogrio")
        if sample.crs is None:
            raise ConfigError(f"'{first.path}' has no CRS; set ANALYSIS_CRS instead.")
        crs = CRS.from_user_input(sample.crs)
    if not crs.is_projected:
        raise ConfigError(
            f"Analysis CRS {crs.name} is not projected; use a local projected CRS "
            "in feet or meters."
        )
    if crs.to_epsg() in WEB_MERCATOR_CODES or "mercator" in crs.name.lower():
        raise ConfigError("Mercator distorts local distances; choose a local projected CRS.")
    unit_factor = crs.axis_info[0].unit_conversion_factor if crs.axis_info else 0.0
    if not unit_factor or unit_factor <= 0:
        raise ConfigError(f"Analysis CRS {crs.name} has no linear axis unit.")
    return crs, METERS_PER_FOOT / unit_factor


def run_audit(cfg: Settings) -> dict[str, Any]:
    """Run the audit and write every output.

    Args:
        cfg: Validated run settings.

    Returns:
        The run summary that was also written to the summary JSON.

    Raises:
        ConfigError: The analysis CRS is unusable.
        OSError: An input is unreadable or an output cannot be written.
        ValueError: Input data fails a check (IDs, coordinates, geometry, approvals).
        RuntimeError: The run log could not be written and REQUIRE_RUN_LOG is set.
    """
    timings: dict[str, float] = {}
    crs, units_per_foot = resolve_analysis_crs(cfg)
    feet_per_unit = 1.0 / units_per_foot
    epsilon = cfg.coincidence_tolerance_ft * units_per_foot
    logging.info("Analysis CRS: %s (%s).", crs.name, crs.axis_info[0].unit_name)

    with timed_stage("Select stops and build review buffers", timings):
        stops = project_stops(select_gtfs_stops(cfg), crs)
        buffers, scopes, study, context = build_scope(stops, cfg, units_per_foot)
    preflight: list[dict[str, Any]] = []
    with timed_stage("Load network sources and road polygons", timings):
        features = load_sources(cfg, context, crs, preflight)
        if not features:
            logging.warning("No network features intersect the stop context area.")
        roads = load_roads(cfg, context, crs, units_per_foot)
    feature_map = {f.key: f for f in features}

    def build(found: list[Feature]) -> tuple[list[Part], list[Issue], Network]:
        parts, part_issues = make_parts(found, epsilon)
        return parts, part_issues, Network(parts, epsilon, cfg.connectivity_policy)

    with timed_stage("Build existing-vertex network", timings):
        parts, initial, network = build(features)
        before = {
            "audited_parts": len(parts),
            "components": network.component_count(),
            "dangles": len(network.dangles()),
        }
        logging.info(
            "Indexed %d parts, %d segments, %d dangles.",
            len(parts),
            len(network.segment_geoms),
            before["dangles"],
        )
        changes = remove_repeated_vertices(features) if cfg.remove_exact_duplicate_vertices else []
        if changes:
            parts, initial, network = build(features)

    with timed_stage("Gap and possible-crossing review", timings):
        gap_issues = [
            i._replace(issue_id=identify_issue(i, parts))
            for i in scan_gaps(network, roads, units_per_foot, cfg)
            if scope_issue(i, study, feature_map)
        ]
        if cfg.approved_snaps_csv:
            candidates = repair_candidates(gap_issues, network, feature_map, units_per_foot, cfg)
            approvals = read_approvals(cfg.approved_snaps_csv)
            snaps = apply_approved_snaps(
                candidates, feature_map, network, approvals, units_per_foot
            )
            logging.info("Applied %d approved endpoint snap(s).", len(snaps))
            changes.extend(snaps)
            if snaps:
                parts, initial, network = build(features)
                gap_issues = scan_gaps(network, roads, units_per_foot, cfg)

    with timed_stage("Intersection, overlap, and shape review", timings):
        issues = initial + gap_issues + scan_intersections(network, feet_per_unit)
        issues += scan_shape_detail(parts, feet_per_unit, cfg)

    with timed_stage("Component and stop coverage review", timings):
        part_component, component_parts, component_rows, component_issues = component_outputs(
            network, study, context, units_per_foot, cfg
        )
        issues = [i for i in issues + component_issues if scope_issue(i, study, feature_map)]
        excluded = sum(bool(f.excluded_reason) for f in features)
        stop_rows, stop_issues = stop_outputs(
            network, part_component, issues, stops, buffers, units_per_foot, excluded, cfg
        )
        unique: dict[str, Issue] = {}
        for issue in issues + stop_issues:
            identified = issue._replace(issue_id=identify_issue(issue, parts))
            unique[identified.issue_id] = identified
        issues = list(unique.values())

    with timed_stage("Write outputs", timings):
        output_dir = cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        gpkg = output_dir / REVIEW_GPKG_FILENAME
        gpkg.unlink(missing_ok=True)  # Drop stale layers (e.g. clips for old radii).

        layers: dict[str, gpd.GeoDataFrame] = {
            "gtfs_stops": stops,
            "stop_buffers": buffers,
            "study_area": gpd.GeoDataFrame({"AREA": ["study"]}, geometry=[study], crs=crs),
            "context_area": gpd.GeoDataFrame({"AREA": ["context"]}, geometry=[context], crs=crs),
        }
        if cfg.road_polygons_path:
            layers["road_context"] = gpd.GeoDataFrame(
                {"SOURCE_FID": roads.fids}, geometry=list(roads.geometries), crs=crs
            )
        for suffix, frame in network_layers(features, crs, use_original=True).items():
            layers[f"network_original{suffix}"] = frame
        for suffix, frame in network_layers(features, crs, use_original=False).items():
            layers[f"network_working{suffix}"] = frame
        components = frame_of(
            component_parts
            if not component_parts.empty
            else pd.DataFrame(
                columns=[
                    "SOURCE_KEY",
                    "PART_NO",
                    "COMP_ID",
                    "CONTEXT_EDGE",
                    "LENGTH_FT",
                    "STUDY_FT",
                    "geometry",
                ]
            ),
            crs,
        )
        layers["network_components"] = components
        issue_table, issue_points, issue_lines = issue_tables(issues, parts, crs)
        layers["issues_points"] = issue_points
        layers["issues_lines"] = issue_lines
        if cfg.write_clipped_network:
            for radius, scope in scopes.items():
                layers[f"network_clip_{radius_suffix(radius)}"] = gpd.clip(
                    components, scope, keep_geom_type=True
                )
        for name, frame in layers.items():
            frame.to_file(gpkg, layer=name, driver="GPKG", engine="pyogrio")
        logging.info("Wrote %d layers to %s", len(layers), gpkg)

        issue_table.to_csv(output_dir / ISSUES_FILENAME, index=False, encoding="utf-8-sig")
        write_csv(output_dir / STOP_SUMMARY_FILENAME, stop_rows, STOP_SUMMARY_COLUMNS)
        write_csv(output_dir / COMPONENTS_FILENAME, component_rows, COMPONENT_COLUMNS)
        write_csv(output_dir / PREFLIGHT_FILENAME, preflight, PREFLIGHT_COLUMNS)
        write_csv(
            output_dir / REPAIR_CANDIDATES_FILENAME,
            repair_candidates(issues, network, feature_map, units_per_foot, cfg),
            REPAIR_COLUMNS,
        )
        write_csv(output_dir / REPAIR_LOG_FILENAME, changes, REPAIR_LOG_COLUMNS)
        write_csv(
            output_dir / SOURCE_ATTRIBUTES_FILENAME,
            [
                {
                    "SOURCE_KEY": f.key,
                    "SOURCE_LABEL": f.label,
                    "SOURCE_FID": f.fid,
                    "SOURCE_ID": f.source_id,
                    "ATTRIBUTES_JSON": json.dumps(f.attrs, default=str),
                }
                for f in features
            ],
            SOURCE_ATTRIBUTE_COLUMNS,
        )
        working_problems = sum(bool(check_line_geometry(f.geometry)) for f in features)

    study_length_ft = sum(r["STUDY_LENGTH_FT"] for r in component_rows)
    counts = Counter(i.kind for i in issues)
    summary: dict[str, Any] = {
        "status": "REVIEW_REQUIRED",
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "geopandas": gpd.__version__,
        "shapely": shapely.__version__,
        "settings": settings_record(cfg),
        "analysis_crs": crs.name,
        "coincidence_tolerance_ft": cfg.coincidence_tolerance_ft,
        "stops": len(stops),
        "selected_features": len(features),
        "audited_parts": len(parts),
        "excluded_features": excluded,
        "geometry_preflight_problems": len(preflight),
        "working_geometry_problems": working_problems,
        "audited_study_length_ft": study_length_ft,
        "issue_counts": dict(sorted(counts.items())),
        "issues_per_mapped_mile": (
            len(issues) / (study_length_ft / FEET_PER_MILE) if study_length_ft else None
        ),
        "repairs_applied": len(changes),
        "before_repairs_context": before,
        "after_repairs_context": {
            "audited_parts": len(parts),
            "components": len(component_rows),
            "dangles": len(network.dangles()),
        },
        "z_features_using_xy_diagnostics": sum(f.has_z for f in features),
        "unknown_level_features": sum(not f.level for f in features),
        "road_polygons_in_context": len(roads.geometries),
        "stage_seconds": timings,
        "limitations": LIMITATIONS,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logging.info("Issue counts: %s", dict(sorted(counts.items())))
    logging.info("Repairs applied: %d; excluded features: %d.", len(changes), excluded)

    summary_lines = [
        f"Analysis CRS:       {crs.name}",
        f"Stops:              {len(stops)}",
        f"Selected features:  {len(features)} ({excluded} excluded from the graph)",
        f"Audited parts:      {len(parts)} in {len(component_rows)} component(s)",
        f"Issues:             {len(issues)}",
        f"Repairs applied:    {len(changes)}",
    ]
    if not write_run_log(gpkg, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return summary


# =============================================================================
# CLI / MAIN
# =============================================================================


def source_specs(entries: Sequence[dict[str, str]]) -> tuple[SourceSpec, ...]:
    """Convert NETWORK_SOURCES dictionaries into source specs."""
    return tuple(
        SourceSpec(
            label=str(entry.get("LABEL", "")).strip(),
            path=str(entry.get("PATH", "")).strip(),
            layer=str(entry.get("LAYER", "")).strip(),
            where=str(entry.get("WHERE", "")).strip(),
            id_field=str(entry.get("ID_FIELD", "")).strip(),
            level_field=str(entry.get("LEVEL_FIELD", "")).strip(),
        )
        for entry in entries
    )


def specs_from_paths(paths: Sequence[str]) -> tuple[SourceSpec, ...]:
    """Build unfiltered source specs from CLI paths, labeled by file name."""
    specs: list[SourceSpec] = []
    used: set[str] = set()
    for path in paths:
        base = re.sub(r"[^A-Za-z0-9_]", "_", Path(path).stem) or "source"
        base = (base if base[0].isalpha() else f"src_{base}")[:70]
        label, n = base, 1
        while label in used:
            n += 1
            label = f"{base}_{n}"
        used.add(label)
        specs.append(SourceSpec(label=label, path=path))
    return tuple(specs)


def validate_settings(cfg: Settings) -> None:
    """Check paths, labels, and thresholds before any data is read.

    Raises:
        ConfigError: A setting is missing, inconsistent, or out of range.
    """
    if not cfg.network_sources:
        raise ConfigError("Configure at least one NETWORK_SOURCES entry.")
    labels: set[str] = set()
    inputs: set[tuple[str, str, str]] = set()
    for spec in cfg.network_sources:
        if not LABEL_PATTERN.fullmatch(spec.label) or spec.label in labels:
            raise ConfigError(
                "Source labels must be unique: a letter, then letters/digits/underscores, max 80."
            )
        labels.add(spec.label)
        identity = (spec.path.lower().replace("/", "\\"), spec.layer, spec.where)
        if identity in inputs:
            raise ConfigError("The same network path, layer, and WHERE filter appear twice.")
        inputs.add(identity)
        if not Path(spec.path).exists():
            raise ConfigError(f"Network layer not found: {spec.path}")
    if not Path(cfg.gtfs_path).exists():
        raise ConfigError(f"GTFS_PATH not found: {cfg.gtfs_path}")
    if cfg.road_polygons_path and not Path(cfg.road_polygons_path).exists():
        raise ConfigError(f"ROAD_POLYGONS_PATH not found: {cfg.road_polygons_path}")
    if cfg.approved_snaps_csv and not Path(cfg.approved_snaps_csv).is_file():
        raise ConfigError(f"APPROVED_SNAPS_CSV not found: {cfg.approved_snaps_csv}")
    if cfg.connectivity_policy not in POLICIES:
        raise ConfigError(f"CONNECTIVITY_POLICY must be one of {POLICIES}.")
    radii = cfg.stop_buffer_miles
    if not radii or len(set(radii)) != len(radii):
        raise ConfigError("STOP_BUFFER_MILES must hold unique positive radii.")
    positive = [
        *radii,
        cfg.context_margin_ft,
        cfg.coincidence_tolerance_ft,
        cfg.gap_review_ft,
        cfg.crossing_search_ft,
        cfg.min_road_overlap_ft,
        cfg.short_segment_ft,
        cfg.coarse_segment_ft,
        cfg.max_approved_snap_ft,
    ]
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        raise ConfigError("Distances, radii, and tolerances must be finite positive numbers.")
    if cfg.context_margin_ft < max(cfg.gap_review_ft, cfg.crossing_search_ft):
        raise ConfigError(
            "CONTEXT_MARGIN_FT must cover both the gap and crossing search distances."
        )
    if cfg.max_approved_snap_ft > cfg.gap_review_ft:
        raise ConfigError("MAX_APPROVED_SNAP_FT cannot exceed GAP_REVIEW_FT.")
    if cfg.coincidence_tolerance_ft >= cfg.max_approved_snap_ft:
        raise ConfigError("COINCIDENCE_TOLERANCE_FT must be far smaller than any snap distance.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser (defaults are the CONFIGURATION values)."""
    p = argparse.ArgumentParser(
        description="Audit a pedestrian line network around GTFS stops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--network",
        nargs="+",
        default=[entry["PATH"] for entry in NETWORK_SOURCES],
        help="Pedestrian line layer(s). Paths other than the configured ones replace "
        "NETWORK_SOURCES, labeled by file name, with no filter, ID, or level field.",
    )
    p.add_argument("--gtfs", default=GTFS_PATH, help="GTFS folder or .zip.")
    p.add_argument("--roads", default=ROAD_POLYGONS_PATH, help="Optional road polygon layer.")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for all outputs.")
    p.add_argument(
        "--crs", default=ANALYSIS_CRS, help='Local projected analysis CRS ("" = first layer).'
    )
    p.add_argument(
        "--buffer-miles",
        nargs="+",
        type=float,
        default=STOP_BUFFER_MILES,
        help="Straight-line review radii around each stop, in miles.",
    )
    p.add_argument(
        "--routes",
        nargs="*",
        default=GTFS_ROUTE_SHORT_NAMES,
        help="Route short names whose stops are audited (none = all platform stops).",
    )
    p.add_argument("--stop-ids", nargs="*", default=GTFS_STOP_IDS, help="Optional stop_id filter.")
    p.add_argument(
        "--connectivity-policy",
        choices=POLICIES,
        default=CONNECTIVITY_POLICY,
        help="Which coincident vertices join two lines.",
    )
    p.add_argument(
        "--gap-review-ft", type=float, default=GAP_REVIEW_FT, help="Gap review distance (ft)."
    )
    p.add_argument(
        "--crossing-search-ft",
        type=float,
        default=CROSSING_SEARCH_FT,
        help="Search distance for gaps that cross road polygons (ft).",
    )
    p.add_argument(
        "--approved-snaps",
        default=APPROVED_SNAPS_CSV,
        help="Edited copy of a repair candidates CSV with APPROVE = YES rows.",
    )
    p.add_argument(
        "--remove-duplicate-vertices",
        action=argparse.BooleanOptionalAction,
        default=REMOVE_EXACT_DUPLICATE_VERTICES,
        help="Remove exact repeated vertices from 2D lines in the working copy.",
    )
    p.add_argument(
        "--write-clips",
        action=argparse.BooleanOptionalAction,
        default=WRITE_CLIPPED_NETWORK,
        help="Write a map-only network clip for each radius.",
    )
    return p


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


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Combine parsed CLI arguments with the remaining CONFIGURATION values."""
    configured = source_specs(NETWORK_SOURCES)
    if list(args.network) == [spec.path for spec in configured]:
        sources = configured
    else:
        sources = specs_from_paths([str(p) for p in args.network])
    return Settings(
        network_sources=sources,
        gtfs_path=str(args.gtfs),
        output_dir=Path(args.output_dir).expanduser(),
        route_short_names=frozenset(str(r).strip() for r in args.routes if str(r).strip()),
        stop_ids=frozenset(str(s) for s in args.stop_ids if str(s)),
        road_polygons_path=str(args.roads),
        road_polygons_layer=ROAD_POLYGONS_LAYER,
        road_where=ROAD_WHERE,
        analysis_crs=str(args.crs),
        stop_buffer_miles=tuple(float(r) for r in args.buffer_miles),
        write_clipped_network=bool(args.write_clips),
        connectivity_policy=str(args.connectivity_policy),
        gap_review_ft=float(args.gap_review_ft),
        crossing_search_ft=float(args.crossing_search_ft),
        remove_exact_duplicate_vertices=bool(args.remove_duplicate_vertices),
        approved_snaps_csv=str(args.approved_snaps),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the walking network audit.

    Returns:
        Process exit code: 0 on success, 1 on a runtime failure, 2 when
        CONFIGURATION values are still placeholders or invalid.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    cfg = settings_from_args(args)

    paths = [cfg.gtfs_path, str(cfg.output_dir), *(s.path for s in cfg.network_sources)]
    if any(PLACEHOLDER_MARK in path for path in paths):
        logging.warning(
            "NETWORK_SOURCES, GTFS_PATH, and/or OUTPUT_DIR are still placeholders. Update the "
            "CONFIGURATION section or pass --network/--gtfs/--output-dir before running."
        )
        return 2
    try:
        validate_settings(cfg)
        summary = run_audit(cfg)
    except ConfigError as exc:
        logging.error("Configuration error: %s", exc)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.warning("Outputs are review evidence, not a certified routable network.")
    logging.info(
        "Script completed successfully: %d issue(s) for %d stop(s); outputs in %s",
        sum(summary["issue_counts"].values()),
        summary["stops"],
        cfg.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
