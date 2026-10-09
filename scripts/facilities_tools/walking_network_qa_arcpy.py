"""Audit a pedestrian line network near GTFS stops with ArcGIS Pro / Python 3.9+.

Edit CONFIGURATION, then run this file in the ArcGIS Pro Python environment.
Requires ArcPy, but no Network Analyst, Data Reviewer, pandas, or NetworkX.
Use file geodatabases/shapefiles with Basic; enterprise geometry checks may require
a higher ArcGIS license, according to the input storage type.
Inputs are never edited. Each run writes a NEW timestamped output directory.

QUICK START
    1. Set NETWORK_SOURCES, GTFS_PATH, OUTPUT_FOLDER, and optional ROAD_POLYGONS.
    2. Use a suitable local projected CRS. ANALYSIS_WKID=0 uses the first input's CRS.
    3. Start with all repair options disabled. Inspect issues_points, issues_lines,
       network_components, stop_summary.csv, and summary.json in the run folder.
    4. Correct ambiguous crossings/grade separation in your source GIS and rerun.
       Optional: copy repair_candidates.csv elsewhere, mark selected APPROVE cells
       YES, and set APPROVED_SNAPS_CSV to that file. Stale candidates are rejected.

PERFORMANCE AND PROGRESS
    - Buffers and whole-feature selection precede the detailed network checks.
      By default, Check Geometry also runs only on the selected context features.
      Null shapes cannot be located by a spatial selection. Set
      FULL_SOURCE_GEOMETRY_CHECK=True to restore the whole-source preflight.
    - USE_LOCAL_WORKSPACE=True builds in Windows TEMP, then copies completed
      results to OUTPUT_FOLDER. Set LOCAL_WORK_ROOT to another local folder if
      desired. The final folder contains run.log and run_status.json immediately;
      review.gdb and CSVs arrive when publication completes. Failed runs retain
      their working folder, whose full path appears in the log and status file.
    - Smaller indexed pieces of the study/context polygons accelerate repeated
      spatial tests. They do not cut sidewalk lines or introduce network nodes.
      Set USE_TILED_SCOPE=False to use the undivided polygons for comparison.
    - Buffers, graph results, and review geometries are reused; the graph is
      rebuilt only after actual repairs. Original and working outputs remain.
    - Stop coordinates are inserted in WGS84 and projected in one batch. Setup
      logs distinguish schema creation, coordinate insertion, projection, buffer
      creation, and dissolve. Geoprocessing runs after write cursors are closed.
    - Temporary source/road layers have unique names for each use and are removed
      on normal exit or Python exceptions. A hard interruption may leave a layer
      behind; its name will not be reused on the next run.
    - Network sources are staged locally with their original OIDs, projected by
      one native Project operation per source, and spatially selected in the
      analysis CRS. Original attributes are read separately without geometry.
      ID sets and Z/M dimensions are checked; geometry problems from both source
      and projected copies remain excluded. Project does not add vertices for
      shape preservation. Curves remain excluded even if projection densifies them.
      BATCH_PROJECT_SOURCES=False restores the slower per-feature loading path.
    - KEEP_SOURCE_STAGING=True retains the intermediate source_stage_* feature
      classes for investigation. Failed runs retain them regardless. Projected
      geometry-check tables include SOURCE_OID for the original source ID.
    - Stage starts/completions and elapsed-time heartbeats appear in the console
      and run.log. Heartbeats are not percentages. timings.csv and summary.json
      record the measured stage durations; runtime depends on the selected data.
    - If a diagnostic line becomes empty during ArcPy geometry construction, its
      map marker falls back to its first point. MAP_GEOM and GEOM_NOTE document
      this; issues.csv preserves the original coordinates in POINTS_JSON and the
      measured VALUE_FT. Unmappable issues remain in the CSV and run summary.
      Display fallbacks never change source coordinates or graph connectivity.

IMPORTANT INTERPRETATION
    - STOP_BUFFER_MILES defines straight-line REVIEW radii, not walk service areas.
      Features outside the largest radius cannot contribute to a walking route
      shorter than that radius measured from a stop. This does not apply to an
      unrestricted route search. The network itself is NOT clipped for the audit.
    - Whole features intersecting the largest buffer plus CONTEXT_MARGIN_FT are
      retained. Components touching the context boundary have incomplete context.
      network_clip_* layers contain only audited parts and are convenient map
      extracts, NOT audit/routing inputs. Excluded geometry remains in working copies.
    - This is a diagnostic undirected XY graph, not an ArcGIS network dataset.
      ENDPOINT / ANY_VERTEX must match your intended routing configuration.
      Coincidence uses coordinate resolution, never the gap review threshold.
      Crossings without eligible coincident vertices are reported, never joined.
    - LEVEL_FIELD is optional and must describe a CONSTANT level for the entire
      feature. Do not map from/to elevation fields or a ramp to this field.
      Equal known levels connect; different levels do not; two unknown levels
      assume XY connectivity; known-versus-unknown remains unresolved. Z values
      alone are NOT used to determine connectivity. Grade/access review remains
      necessary even for a visually clean graph. No slope/ADA certification.
    - Road polygons are evidence of a potential street crossing, not evidence
      that crossing is permitted, safe, at grade, or missing from the source.
    - GTFS platform stops (location_type blank or 0) are used. Optional route
      filters select stops from trips in the feed, WITHOUT calendar/date filtering.
      Missing/malformed stop coordinates fail the run instead of shrinking scope.
    - True curves and geometry flagged by Check Geometry are retained in the
      network outputs but excluded from the diagnostic graph (except a duplicate-
      vertex-only problem repaired by the optional exact-duplicate cleanup).
      Their exclusion is explicit; affected results are marked incomplete.
    - Sparse coverage and long vertex spacing are review indicators, not errors.
      Component counts are context-dependent and cannot certify completeness.
    - Repairs are limited to consecutive identical XY vertices on 2D lines and
      explicitly approved, unique, tiny endpoint-to-endpoint snaps. No automatic
      street crossing, unnoded intersection, smoothing, or feature deletion.
      Source attributes are retained in source_attributes.csv as JSON, linked by
      SOURCE_KEY. Output network feature classes have a normalized review schema;
      they do NOT carry a configured routing impedance/restriction model.

OUTPUTS
    review.gdb: gtfs_stops; stop_buffers; study_area; context_area; road_context;
      network_original; network_working; network_components;
      issues_points; issues_lines; network_clip_<radius>; geometry_check_<source>.
    Network outputs with Z/M dimensions have _z, _m, or _zm suffixes. Dimension
      groups are kept separate so missing elevation values are not invented.
    issues.csv: location, type, confidence, source/part IDs, reason, road context.
    stop_summary.csv: radius, nearby line length/density, component count,
      nearest-line distance, and review-issue counts. Overlapping buffers are
      intentionally summarized separately; never add their lengths together.
    components.csv; source_attributes.csv; repair_candidates.csv; repair_log.csv;
      summary.json; timings.csv; run_status.json; run.log. The manifest records
      assumptions and parameters. study_area_<radius> polygons cache smaller
      dissolved scopes when map clips are enabled.

Use stable ID_FIELD values where available. Otherwise SOURCE_KEY uses source label
and original OID and is only stable while the source OIDs remain unchanged.

References (ArcGIS Pro documentation):
    https://pro.arcgis.com/en/pro-app/latest/help/analysis/networks/understanding-connectivity.htm
    https://pro.arcgis.com/en/pro-app/latest/arcpy/classes/geometry.htm
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/check-geometry.htm
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import arcpy
except ImportError:
    arcpy = None  # Allows the pure geometry helpers to be regression-tested elsewhere.


# =============================================================================
# CONFIGURATION -- full paths; no arcpy.env.workspace is required.
# =============================================================================

NETWORK_SOURCES = [
    {
        "LABEL": "sidewalks",
        "PATH": r"Path\To\Your\Sidewalks.shp",
        "WHERE": "",  # Optional SQL filter, e.g. public, existing, walkable features.
        "ID_FIELD": "",  # Optional stable source ID. Blank retains original OID.
        "LEVEL_FIELD": "",  # Optional constant grade level for the whole feature.
    },
    # Add crossings/trails only when they are separate, nonduplicated inputs:
    # {
    #     "LABEL": "crossings",
    #     "PATH": r"C:\GIS\pedestrian_network.gdb\crossings",
    #     "WHERE": "",
    #     "ID_FIELD": "",
    #     "LEVEL_FIELD": "",
    # },
]

# An extracted feed folder OR a .zip feed.
GTFS_PATH = r"Path\To\Your\GTFS_Folder"
GTFS_ROUTE_SHORT_NAMES: Set[str] = set()  # Empty = all platform stops in stops.txt.
GTFS_STOP_IDS: Set[str] = set()  # Optional additional stop_id filter (intersection).
ROAD_POLYGONS = ""  # Optional: r"C:\GIS\transportation.gdb\road_polygons"
ROAD_WHERE = ""
OUTPUT_FOLDER = r"Path\To\Your\Output_Folder"

ANALYSIS_WKID = 2283  # NAD 1983 StatePlane Virginia North, US survey feet
#ANALYSIS_WKID = 0  # 0 = first input CRS; otherwise supply a LOCAL projected WKID.
GEOGRAPHIC_TRANSFORMATION = ""  # Blank = first valid ArcGIS-listed transformation.
STOP_BUFFER_MILES = (0.25,)
CONTEXT_MARGIN_FT = 200.0
WRITE_CLIPPED_NETWORK = True
CONNECTIVITY_POLICY = "ANY_VERTEX"  # "ANY_VERTEX" or "ENDPOINT".

# These are starting review thresholds, NOT established accuracy standards.
GAP_REVIEW_FT = 10.0
CROSSING_SEARCH_FT = 120.0  # Applied only when the proposed gap overlaps road area.
MIN_ROAD_OVERLAP_FT = 1.0  # Ignore negligible touches at a road polygon edge.
ROAD_EDGE_REVIEW_FT = 6.0
SHORT_SEGMENT_FT = 0.5
COARSE_SEGMENT_FT = 150.0
SPIKE_LEG_FT = 5.0
SPIKE_TURN_DEGREES = 150.0
SMALL_COMPONENT_LENGTH_FT = 200.0
STOP_NEAR_NETWORK_FT = 75.0
LOW_DENSITY_MI_PER_SQ_MI = 2.0  # Context-dependent screening only; 0 disables flag.
SPATIAL_GRID_FT = 100.0
COINCIDENCE_RESOLUTION_MULTIPLIER = 1.1  # Not a repair/snap distance.

# Repairs are OFF for the first run. All changes affect new outputs only.
REMOVE_EXACT_DUPLICATE_VERTICES = False
APPROVED_SNAPS_CSV = ""  # Previously exported repair_candidates.csv, APPROVE=YES.
MAX_APPROVED_SNAP_FT = 1.0

# Performance settings. Original paths, review thresholds, and repairs are retained.
USE_LOCAL_WORKSPACE = True  # Build locally, then copy finished outputs to OUTPUT_FOLDER.
LOCAL_WORK_ROOT = ""  # Blank = Windows TEMP; alternatively set a full local folder path.
KEEP_LOCAL_WORKSPACE = False  # Failed runs are ALWAYS retained for diagnosis/recovery.
FULL_SOURCE_GEOMETRY_CHECK = False  # False = check selected context features only.
BATCH_PROJECT_SOURCES = True  # Native local export/Project; preserve original OIDs separately.
KEEP_SOURCE_STAGING = False  # Delete successful intermediate copies before publishing.
SOURCE_PROGRESS_ROWS = 5000  # Also log the first and 100th row of each reading stage.
USE_TILED_SCOPE = True  # Index exact polygon pieces; never simplify the network.
SCOPE_TILE_SIZE_FT = 5280.0
MAX_SCOPE_TILES = 10000
PROGRESS_LOG_SECONDS = 30.0

LOG_LEVEL = "INFO"
SCRIPT_VERSION = "1.2.1"


# =============================================================================
# DATA MODELS AND PURE GEOMETRY HELPERS
# =============================================================================

XY = Tuple[float, float]
Box = Tuple[float, float, float, float]
LOGGER = logging.getLogger("walking_network_qa")


@contextmanager
def timed_stage(name: str, timings: List[Dict[str, Any]]) -> Iterator[None]:
    """Log start, elapsed-time heartbeats, and completion without background ArcPy.

    The background thread only writes log messages. All GIS operations stay on
    the main thread. A heartbeat indicates a running stage, not completed work.
    """
    started = time.perf_counter()
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(PROGRESS_LOG_SECONDS):
            LOGGER.info("Still running: %s [%.1f min elapsed]", name,
                        (time.perf_counter() - started) / 60)

    LOGGER.info("START: %s", name)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    status = "COMPLETED"
    try:
        yield
    except BaseException:
        status = "FAILED"
        raise
    finally:
        stopped.set()
        worker.join()
        elapsed = time.perf_counter() - started
        timings.append({"STAGE": name, "SECONDS": elapsed, "STATUS": status})
        LOGGER.info("%s: %s [%.1f sec]", status, name, elapsed)


@dataclass
class Feature:
    """A source feature with projected geometry and retained lineage."""

    key: str
    label: str
    oid: str
    source_id: str
    level: str
    geometry: Any
    original: Any
    attrs: Dict[str, Any]
    problems: List[str] = field(default_factory=list)
    excluded_reason: str = ""
    dimension_flags: Optional[Tuple[bool, bool]] = None
    source_has_curves: bool = False


@dataclass
class Part:
    """One continuous line part; multipart features do not imply connections."""

    pid: int
    feature_key: str
    part_number: int
    points: List[XY]
    level: str = ""

    @property
    def length(self) -> float:
        return sum(distance(a, b) for a, b in zip(self.points, self.points[1:]))


@dataclass
class Segment:
    """A straight segment belonging to a continuous line part."""

    sid: int
    pid: int
    position: int
    a: XY
    b: XY


@dataclass
class Issue:
    """A diagnostic observation. Confidence concerns the interpretation as an error."""

    kind: str
    points: List[XY]
    pid: int = -1
    other_pid: int = -1
    value_ft: Optional[float] = None
    confidence: str = "REVIEW"
    detail: str = ""
    road_ft: float = 0.0
    feature_key: str = ""
    issue_id: str = ""
    map_geometry: Any = field(default=None, init=False, repr=False, compare=False)
    map_geometry_kind: str = field(default="UNMAPPED", init=False, repr=False, compare=False)
    map_geometry_note: str = field(default="", init=False, repr=False, compare=False)
    map_geometry_ready: bool = field(default=False, init=False, repr=False, compare=False)


class Grid:
    """Spatial hash with exact bounding-box filtering and a large-object fallback."""

    def __init__(self, cell_size: float) -> None:
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("Grid cell size must be finite and positive.")
        self.size = cell_size
        self.cells: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.boxes: Dict[int, Box] = {}
        self.large: Set[int] = set()

    def _cells(self, box: Box) -> Optional[List[Tuple[int, int]]]:
        validate_box(box)
        x0, y0, x1, y1 = (math.floor(v / self.size) for v in box)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 20000:
            return None
        return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def add(self, item: int, box: Box) -> None:
        cells = self._cells(box)
        self.boxes[item] = box
        if cells is None:
            self.large.add(item)
        else:
            for cell in cells:
                self.cells[cell].add(item)

    def query(self, box: Box) -> List[int]:
        cells = self._cells(box)
        found = set(self.boxes) if cells is None else set(self.large)
        if cells is not None:
            for cell in cells:
                found.update(self.cells.get(cell, ()))
        return sorted(i for i in found if boxes_touch(self.boxes[i], box))


class UnionFind:
    """Track undirected connected groups without an optional graph dependency."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def distance(a: XY, b: XY) -> float:
    """Return planar distance in analysis coordinate units."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox(points: Sequence[XY], pad: float = 0.0) -> Box:
    """Return a padded bounding box."""
    xs, ys = zip(*points)
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def boxes_touch(a: Box, b: Box) -> bool:
    """Return whether two closed boxes intersect."""
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def nearest_on_segment(p: XY, a: XY, b: XY) -> Tuple[XY, float]:
    """Return the closest location and clamped fractional position on a segment."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom
    t = max(0.0, min(1.0, t))
    return (a[0] + t * dx, a[1] + t * dy), t


def segment_intersection(
    a: XY, b: XY, c: XY, d: XY, epsilon: float
) -> Tuple[str, List[XY]]:
    """Return NONE, POINT, or OVERLAP without creating network connections.

    Epsilon is coordinate-resolution scale, not a user review or snap distance.
    """
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    lr, ls = math.hypot(rx, ry), math.hypot(sx, sy)
    if lr <= epsilon or ls <= epsilon:
        return "NONE", []
    qx, qy = c[0] - a[0], c[1] - a[1]
    cross = rx * sy - ry * sx
    if abs(cross) <= epsilon * max(lr, ls):
        # Check both endpoints; near-parallel lines are not necessarily collinear.
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
        return ("POINT", [first]) if distance(first, last) <= epsilon else (
            "OVERLAP", [first, last]
        )
    t = (qx * sy - qy * sx) / cross
    u = (qx * ry - qy * rx) / cross
    if -epsilon / lr <= t <= 1 + epsilon / lr and -epsilon / ls <= u <= 1 + epsilon / ls:
        t = max(0.0, min(1.0, t))
        return "POINT", [(a[0] + t * rx, a[1] + t * ry)]
    return "NONE", []


def deduplicate_consecutive(points: Sequence[XY]) -> List[XY]:
    """Remove only exactly repeated consecutive coordinates; retain closed loops."""
    output: List[XY] = []
    for point in points:
        if not output or point != output[-1]:
            output.append(point)
    return output


def levels_connect(first: str, second: str) -> bool:
    """Equal levels connect, including the explicitly documented two-unknown case."""
    return first == second


def eligible_vertex(part: Part, position: int, policy: str) -> bool:
    """Identify a vertex used by the selected diagnostic connectivity policy."""
    return policy == "ANY_VERTEX" or position in (0, len(part.points) - 1)


class Network:
    """Index the existing vertices and segments without snapping or planarizing."""

    def __init__(self, parts: List[Part], epsilon: float, cell: float, policy: str) -> None:
        self.parts = parts
        self.epsilon = epsilon
        self.policy = policy
        self.segments: List[Segment] = []
        self.segment_grid = Grid(cell)
        self.vertices: List[Tuple[int, int, XY]] = []
        self.vertex_grid = Grid(max(epsilon * 2, 1e-9))
        self.components = UnionFind(len(parts))
        self._degrees: Dict[Tuple[int, XY], int] = {}
        self._dangles: Optional[List[Tuple[int, int, XY]]] = None
        self._geometries: Dict[int, Any] = {}
        for part in parts:
            for position, point in enumerate(part.points):
                if eligible_vertex(part, position, policy):
                    vid = len(self.vertices)
                    for other in self.vertex_grid.query(bbox([point], epsilon)):
                        other_pid, _, other_point = self.vertices[other]
                        if distance(point, other_point) <= epsilon and levels_connect(
                            part.level, parts[other_pid].level
                        ):
                            self.components.union(part.pid, other_pid)
                    self.vertices.append((part.pid, position, point))
                    self.vertex_grid.add(vid, bbox([point]))
            for position, (a, b) in enumerate(zip(part.points, part.points[1:])):
                if distance(a, b) <= epsilon:
                    continue
                segment = Segment(len(self.segments), part.pid, position, a, b)
                self.segments.append(segment)
                self.segment_grid.add(segment.sid, bbox([a, b]))

    def at_vertex(self, pid: int, point: XY, eligible_only: bool = True) -> bool:
        """Test whether a point is represented by an eligible ORIGINAL vertex."""
        if not eligible_only and self.policy != "ANY_VERTEX":
            return any(distance(p, point) <= self.epsilon for p in self.parts[pid].points)
        return any(
            self.vertices[vid][0] == pid
            and distance(self.vertices[vid][2], point) <= self.epsilon
            for vid in self.vertex_grid.query(bbox([point], self.epsilon))
        )

    def degree(self, pid: int, point: XY) -> int:
        """Count incident line arms at an eligible vertex, including closed loops."""
        cache_key = (pid, point)
        if cache_key in self._degrees:
            return self._degrees[cache_key]
        arms: Set[Tuple[int, int]] = set()
        for vid in self.vertex_grid.query(bbox([point], self.epsilon)):
            other_pid, position, other_point = self.vertices[vid]
            part = self.parts[other_pid]
            if distance(point, other_point) > self.epsilon or not levels_connect(
                self.parts[pid].level, part.level
            ):
                continue
            if position > 0:
                arms.add((other_pid, position - 1))
            if position < len(part.points) - 1:
                arms.add((other_pid, position))
        self._degrees[cache_key] = len(arms)
        return len(arms)

    def nearest_parts(self, point: XY, radius: float) -> Dict[int, Tuple[float, XY]]:
        """Find each nearby part's true closest location, not just its vertices."""
        found: Dict[int, Tuple[float, XY]] = {}
        for sid in self.segment_grid.query(bbox([point], radius)):
            seg = self.segments[sid]
            q, _ = nearest_on_segment(point, seg.a, seg.b)
            gap = distance(point, q)
            if gap <= radius and (seg.pid not in found or gap < found[seg.pid][0]):
                found[seg.pid] = gap, q
        return found

    def dangles(self) -> List[Tuple[int, int, XY]]:
        """Return actual degree-one endpoints; no line clipping is performed."""
        if self._dangles is None:
            self._dangles = [
                (p.pid, endpoint, p.points[endpoint])
                for p in self.parts
                for endpoint in (0, len(p.points) - 1)
                if self.degree(p.pid, p.points[endpoint]) == 1
            ]
        return self._dangles

    def geometry(self, pid: int, sr: Any) -> Any:
        """Reuse ArcPy part geometry in component and stop coverage calculations."""
        if pid not in self._geometries:
            self._geometries[pid] = line_geom(self.parts[pid].points, sr)
        return self._geometries[pid]


def scan_intersections(network: Network, feet_per_unit: float) -> List[Issue]:
    """Flag unnoded crossings, overlaps, self intersections, and grade mismatches."""
    issues: List[Issue] = []
    seen: Set[Tuple[Any, ...]] = set()
    eps = network.epsilon
    for count, first in enumerate(network.segments):
        if count and count % 50000 == 0:
            LOGGER.info("Intersection review: %s / %s segments", count, len(network.segments))
        for sid in network.segment_grid.query(bbox([first.a, first.b], eps)):
            if sid <= first.sid:
                continue
            second = network.segments[sid]
            kind, points = segment_intersection(first.a, first.b, second.a, second.b, eps)
            if kind == "NONE":
                continue
            a, b = network.parts[first.pid], network.parts[second.pid]
            same = a.pid == b.pid
            adjacent = same and abs(first.position - second.position) == 1
            closed_neighbors = same and {first.position, second.position} == {
                0, len(a.points) - 2
            } and distance(a.points[0], a.points[-1]) <= eps
            if kind == "POINT" and (adjacent or closed_neighbors):
                continue
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
                value = None
            if a.level and b.level and a.level != b.level:
                detail += " Known level values differ: likely intentional grade separation."
                confidence = "LIKELY_INTENTIONAL"
            else:
                confidence = "REVIEW"
            # Do not duplicate the same crossing reported by several incident segments.
            key = (code, min(a.pid, b.pid), max(a.pid, b.pid)) + tuple(
                (round(p[0] / eps), round(p[1] / eps)) for p in points
            )
            if key not in seen:
                seen.add(key)
                issues.append(Issue(code, points, a.pid, b.pid, value, confidence, detail))
    return issues


def scan_shape_detail(parts: List[Part], feet_per_unit: float) -> List[Issue]:
    """Find short segments, coarse vertex spacing, and short sharp reversals."""
    issues: List[Issue] = []
    for part in parts:
        for a, b in zip(part.points, part.points[1:]):
            length_ft = distance(a, b) * feet_per_unit
            if length_ft < SHORT_SEGMENT_FT:
                issues.append(Issue("SHORT_SEGMENT", [a, b], part.pid, value_ft=length_ft,
                                    detail="Short segment; may be legitimate junction detail."))
            elif length_ft > COARSE_SEGMENT_FT:
                issues.append(Issue("COARSE_VERTEX_SPACING", [a, b], part.pid,
                                    value_ft=length_ft, confidence="SCREENING",
                                    detail="Long straight segment; verify against imagery."))
        for a, b, c in zip(part.points, part.points[1:], part.points[2:]):
            first, second = distance(a, b), distance(b, c)
            if min(first, second) == 0 or max(first, second) * feet_per_unit > SPIKE_LEG_FT:
                continue
            dot = ((b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / (first * second)))))
            if angle >= SPIKE_TURN_DEGREES:
                issues.append(Issue("SHORT_SPIKE", [a, b, c], part.pid,
                                    value_ft=(first + second) * feet_per_unit,
                                    detail="Short sharp reversal; check geometry or a real turn."))
    return issues


# =============================================================================
# ARCPY I/O, GTFS, STUDY AREA, AND SOURCE PREPARATION
# =============================================================================


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    """Write a consistent UTF-8 CSV, including its header when there are no rows."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_gtfs(filename: str) -> List[Dict[str, str]]:
    """Read one GTFS table, retaining text IDs such as NA and leading zeroes."""
    path = Path(GTFS_PATH)
    if path.is_dir():
        with (path / filename).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == filename]
        if len(matches) != 1:
            raise ValueError("Expected exactly one {} in GTFS ZIP.".format(filename))
        with archive.open(matches[0]) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))


def required_columns(rows: List[Dict[str, str]], columns: Sequence[str], name: str) -> None:
    """Fail explicitly on an empty or malformed required GTFS table."""
    if not rows or not set(columns).issubset(rows[0]):
        raise ValueError("{} is empty or missing required columns: {}".format(name, columns))


def select_gtfs_stops() -> List[Dict[str, str]]:
    """Select platform stops, optionally limited by feed route names and stop IDs."""
    stops = read_gtfs("stops.txt")
    required_columns(stops, ["stop_id", "stop_lat", "stop_lon"], "stops.txt")
    allowed: Optional[Set[str]] = None
    if GTFS_ROUTE_SHORT_NAMES:
        routes, trips, times = read_gtfs("routes.txt"), read_gtfs("trips.txt"), read_gtfs(
            "stop_times.txt"
        )
        required_columns(routes, ["route_id", "route_short_name"], "routes.txt")
        required_columns(trips, ["trip_id", "route_id"], "trips.txt")
        required_columns(times, ["trip_id", "stop_id"], "stop_times.txt")
        route_names = {r["route_short_name"].strip() for r in routes}
        missing = GTFS_ROUTE_SHORT_NAMES - route_names
        if missing:
            raise ValueError("Route short names absent from GTFS: {}".format(sorted(missing)))
        route_ids = {r["route_id"] for r in routes if r["route_short_name"].strip()
                     in GTFS_ROUTE_SHORT_NAMES}
        trip_ids = {r["trip_id"] for r in trips if r["route_id"] in route_ids}
        allowed = {r["stop_id"] for r in times if r["trip_id"] in trip_ids}
    selected, seen = [], set()
    for row in stops:
        sid = row["stop_id"]
        if not sid or sid in seen:
            raise ValueError("Blank or duplicate stop_id in stops.txt: {!r}".format(sid))
        seen.add(sid)
        if row.get("location_type", "").strip() not in ("", "0"):
            continue
        if allowed is not None and sid not in allowed:
            continue
        if GTFS_STOP_IDS and sid not in GTFS_STOP_IDS:
            continue
        try:
            lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid coordinates for selected stop {}".format(sid)) from exc
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90
                and -180 <= lon <= 180):
            raise ValueError("Out-of-range coordinates for selected stop {}".format(sid))
        selected.append(row)
    if GTFS_STOP_IDS - {r["stop_id"] for r in selected}:
        raise ValueError("Some requested stop IDs are absent or excluded by other filters.")
    if allowed is not None and allowed - seen:
        raise ValueError("Selected route stop_times reference stop IDs missing from stops.txt.")
    if not selected:
        raise ValueError("No GTFS platform stops remain after filtering.")
    return selected


def transformation(source: Any, target: Any, extent: Any = None) -> str:
    """Choose an available transformation when the geographic datums differ."""
    if source.name == "Unknown" or target.name == "Unknown":
        raise ValueError("Every input must have a defined coordinate system.")
    if source.GCS.datumName == target.GCS.datumName:
        return ""
    choices = arcpy.ListTransformations(source, target, extent) if extent is not None else (
        arcpy.ListTransformations(source, target)
    )
    if GEOGRAPHIC_TRANSFORMATION:
        if GEOGRAPHIC_TRANSFORMATION not in choices:
            raise ValueError("Configured transformation is not available for {} -> {}".format(
                source.name, target.name
            ))
        chosen = GEOGRAPHIC_TRANSFORMATION
    elif choices:
        chosen = choices[0]
    else:
        raise ValueError("No geographic transformation available for {} -> {}".format(
            source.name, target.name
        ))
    LOGGER.info("Geographic transformation %s -> %s: %s", source.name, target.name, chosen)
    return chosen


def project(geometry: Any, target: Any, method: str = "") -> Any:
    """Project using the selected geographic transformation, if needed."""
    if geometry.spatialReference.exportToString() == target.exportToString():
        return geometry
    return geometry.projectAs(target, method) if method else geometry.projectAs(target)


def create_fc(gdb: str, name: str, kind: str, sr: Any,
              fields: Sequence[Tuple[str, str, int]], has_z: bool = False,
              has_m: bool = False) -> str:
    """Create a feature class with explicit paths and a small portable field schema."""
    path = os.path.join(gdb, name)
    LOGGER.info("Creating %s feature class: %s", kind, name)
    arcpy.management.CreateFeatureclass(
        gdb, name, kind, has_m="ENABLED" if has_m else "DISABLED",
        has_z="ENABLED" if has_z else "DISABLED", spatial_reference=sr
    )
    if fields:
        LOGGER.info("Adding %s fields in one operation: %s", len(fields), name)
        descriptions = [[field_name, field_type, None, length if field_type == "TEXT" else None]
                        for field_name, field_type, length in fields]
        arcpy.management.AddFields(path, descriptions)
    LOGGER.info("Feature class ready: %s", name)
    return path


def point_geom(point: XY, sr: Any) -> Any:
    """Create an ArcPy point with an explicit spatial reference."""
    return arcpy.PointGeometry(arcpy.Point(*point), sr)


def line_geom(points: Sequence[XY], sr: Any) -> Any:
    """Create a two-dimensional line for diagnostic outputs."""
    return arcpy.Polyline(arcpy.Array([arcpy.Point(*p) for p in points]), sr)


def issue_geometry(issue: Issue, sr: Any) -> Any:
    """Cache a valid display shape, preserving tiny collapsed lines as point markers.

    ArcPy may simplify a short diagnostic line to an empty geometry. A display
    fallback preserves the issue's original points, measured value, and identity.
    Invalid coordinates or failed point construction leave a CSV-only issue.
    This function must not be used to repair source or network geometries.
    """
    if issue.map_geometry_ready:
        return issue.map_geometry
    if not issue.points:
        issue.map_geometry_note = "NO_COORDINATES: retained in CSV only."
    elif not all(math.isfinite(v) for point in issue.points for v in point):
        issue.map_geometry_note = "NONFINITE_COORDINATES: retained in CSV only."
    else:
        is_line = len(issue.points) > 1
        try:
            geometry = line_geom(issue.points, sr) if is_line else point_geom(issue.points[0], sr)
            geom_box(geometry)
        except (ValueError, RuntimeError) as exc:
            issue.map_geometry_note = "DISPLAY_GEOMETRY_INVALID: {}".format(str(exc)[:300])
            if is_line:
                try:
                    geometry = point_geom(issue.points[0], sr)
                    geom_box(geometry)
                except (ValueError, RuntimeError) as point_exc:
                    issue.map_geometry_note += "; POINT_FAILED: {}".format(str(point_exc)[:300])
                else:
                    issue.map_geometry = geometry
                    issue.map_geometry_kind = "POINT_FALLBACK"
                    issue.map_geometry_note += "; marker at first original coordinate."
        else:
            issue.map_geometry = geometry
            issue.map_geometry_kind = "POLYLINE" if is_line else "POINT"
    issue.map_geometry_ready = True
    return issue.map_geometry


def validate_box(box: Box) -> None:
    """Reject unusable extents explicitly before spatial hashing or comparison."""
    if not all(math.isfinite(v) for v in box) or box[0] > box[2] or box[1] > box[3]:
        raise ValueError("Geometry extent must contain finite, ordered XY bounds: {!r}".format(box))


def geom_box(geometry: Any) -> Box:
    """Return validated XY bounds; empty or nonfinite geometry has no usable extent."""
    if geometry is None or geometry.pointCount == 0:
        raise ValueError("Geometry is empty; no usable XY extent.")
    e = geometry.extent
    box = (e.XMin, e.YMin, e.XMax, e.YMax)
    validate_box(box)
    return box


def box_distance(point: XY, box: Box) -> float:
    """Return a lower bound on point-to-feature distance for nearest-line pruning."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


class AreaIndex:
    """Index polygon pieces so each small feature sees only its local scope.

    Tile edges are computational boundaries, not network breaks. Line clips are
    unioned before measuring a line that spans tiles, preventing double-counted
    length on shared tile edges. Context-edge checks use a union of all local
    pieces so internal tile seams never become artificial outside boundaries.
    """

    def __init__(self, pieces: Sequence[Any], cell: float, padding: float) -> None:
        self.pieces = list(pieces)
        self.grid = Grid(cell)
        self.padding = padding
        for i, geometry in enumerate(self.pieces):
            self.grid.add(i, geom_box(geometry))

    def candidates(self, geometry: Any) -> List[Any]:
        """Return all potentially relevant pieces, including both sides of seams."""
        x0, y0, x1, y1 = geom_box(geometry)
        pad = self.padding
        return [self.pieces[i] for i in self.grid.query((x0-pad, y0-pad, x1+pad, y1+pad))]

    def disjoint(self, geometry: Any) -> bool:
        """Test intersection with exact local polygon pieces."""
        return all(geometry.disjoint(piece) for piece in self.candidates(geometry))

    def clipped_length(self, geometry: Any) -> float:
        """Measure the line inside the scope, counting tile seams once."""
        covered = None
        for piece in self.candidates(geometry):
            if geometry.disjoint(piece):
                continue
            clipped = geometry.intersect(piece, 2)
            if clipped.length:
                covered = clipped if covered is None else covered.union(clipped)
        return covered.length if covered is not None else 0.0

    def touches_edge(self, geometry: Any) -> bool:
        """Test the actual area's boundary/extent, not individual tile boundaries."""
        pieces = self.candidates(geometry)
        if any(piece.contains(geometry, "PROPER") for piece in pieces):
            return False
        if not pieces:
            return True
        local = pieces[0]
        for piece in pieces[1:]:
            local = local.union(piece)
        return not geometry.disjoint(local.boundary()) or not geometry.within(local)


def index_area(gdb: str, name: str, area: Any, sr: Any, units_per_foot: float) -> AreaIndex:
    """Split a scope polygon once using native overlay and build a spatial index."""
    cell = SCOPE_TILE_SIZE_FT * units_per_foot
    padding = max(float(sr.XYTolerance), float(sr.XYResolution) * 2)
    if not USE_TILED_SCOPE:
        return AreaIndex([area], cell, padding)
    e = area.extent
    width, height = e.XMax - e.XMin, e.YMax - e.YMin
    columns, rows = max(1, math.ceil(width / cell)), max(1, math.ceil(height / cell))
    while columns * rows > MAX_SCOPE_TILES:
        cell *= max(1.01, math.sqrt(columns * rows / MAX_SCOPE_TILES))
        columns, rows = max(1, math.ceil(width / cell)), max(1, math.ceil(height / cell))
    fishnet, clipped = os.path.join(gdb, name + "_grid"), os.path.join(gdb, name + "_tiles")
    arcpy.management.CreateFishnet(
        fishnet, "{} {}".format(e.XMin, e.YMin),
        "{} {}".format(e.XMin, e.YMin + cell), cell, cell, rows, columns,
        labels="NO_LABELS", geometry_type="POLYGON"
    )
    arcpy.analysis.Clip(fishnet, os.path.join(gdb, name), clipped)
    with arcpy.da.SearchCursor(clipped, ["SHAPE@"]) as cursor:
        pieces = [geom for (geom,) in cursor if geom is not None and geom.area > 0]
    if not pieces:
        raise RuntimeError("No scope tiles produced for {}".format(name))
    arcpy.management.Delete(fishnet)
    arcpy.management.Delete(clipped)
    LOGGER.info("Indexed %s: %s polygon pieces, %.0f-ft tiles", name, len(pieces),
                cell / units_per_foot)
    return AreaIndex(pieces, cell, padding)


def area_disjoint(area: Any, geometry: Any) -> bool:
    """Allow unchanged predicates against either a polygon or an indexed scope."""
    return area.disjoint(geometry) if isinstance(area, AreaIndex) else geometry.disjoint(area)


def merged_geometry(path: str) -> Any:
    """Load every polygon record, including multipart dissolve outputs."""
    output = None
    with arcpy.da.SearchCursor(path, ["SHAPE@"]) as rows:
        for (geometry,) in rows:
            if geometry is not None and geometry.pointCount:
                output = geometry if output is None else output.union(geometry)
    if output is None:
        raise ValueError("Empty study/context geometry: {}".format(path))
    return output


def radius_scope_path(gdb: str, radius: float) -> str:
    """Return the cached dissolved polygon for a requested radius."""
    if radius == max(STOP_BUFFER_MILES):
        return os.path.join(gdb, "study_area")
    suffix = repr(float(radius)).replace(".", "p").replace("-", "m").replace("+", "p")
    return os.path.join(gdb, "study_area_" + suffix)


def prepare_stop_features(
    gdb: str, sr: Any, timings: List[Dict[str, Any]]
) -> Tuple[str, List[Any]]:
    """Insert GTFS coordinates efficiently, then project all stops in one operation.

    Args:
        gdb: Full path of the run's working file geodatabase.
        sr: Validated local projected analysis spatial reference.
        timings: Shared stage-duration records for the run.

    Returns:
        Projected feature-class path and (stop ID, name, geometry) records.
    """
    with timed_stage("Read and validate GTFS stops", timings):
        rows = select_gtfs_stops()
        for row in rows:
            if len(row["stop_id"]) > 255 or len(row.get("stop_name", "")) > 500:
                raise ValueError("GTFS stop text exceeds output schema length.")
        LOGGER.info("Selected %s GTFS platform stops", len(rows))
        wgs = arcpy.SpatialReference(4326)
        lons = [float(r["stop_lon"]) for r in rows]
        lats = [float(r["stop_lat"]) for r in rows]
        method = transformation(wgs, sr, arcpy.Extent(min(lons), min(lats), max(lons), max(lats)))
    with timed_stage("Create WGS84 stop feature class", timings):
        raw_fc = create_fc(gdb, "gtfs_stops_wgs84", "POINT", wgs,
                           [("STOP_ID", "TEXT", 255), ("STOP_NAME", "TEXT", 500)])
    with timed_stage("Write GTFS stop coordinates", timings):
        with arcpy.da.InsertCursor(raw_fc, ["SHAPE@XY", "STOP_ID", "STOP_NAME"]) as cursor:
            for count, row in enumerate(rows, 1):
                xy = (float(row["stop_lon"]), float(row["stop_lat"]))
                cursor.insertRow([xy, row["stop_id"], row.get("stop_name", "")])
                if count % 500 == 0 or count == len(rows):
                    LOGGER.info("Wrote GTFS stops: %s / %s", count, len(rows))
        del cursor
    stop_fc = os.path.join(gdb, "gtfs_stops")
    with timed_stage("Project all GTFS stops to analysis CRS", timings):
        arcpy.management.Project(raw_fc, stop_fc, sr, transform_method=method)
    with timed_stage("Read and validate projected stops", timings):
        by_id = {}
        with arcpy.da.SearchCursor(stop_fc, ["STOP_ID", "STOP_NAME", "SHAPE@"]) as cursor:
            for sid, name, geometry in cursor:
                if geometry is None or geometry.pointCount == 0:
                    raise ValueError("Empty projected geometry for stop {}".format(sid))
                p = geometry.firstPoint
                if not (math.isfinite(p.X) and math.isfinite(p.Y)):
                    raise ValueError("Invalid projected coordinates for stop {}".format(sid))
                if sid in by_id:
                    raise ValueError("Duplicate stop ID after projection: {}".format(sid))
                by_id[sid] = (sid, name, geometry)
        del cursor
        if set(by_id) != {r["stop_id"] for r in rows}:
            raise ValueError("Projected stop IDs do not match the selected GTFS stops.")
        stops = [by_id[r["stop_id"]] for r in rows]
        arcpy.management.Delete(raw_fc)
    return stop_fc, stops


def build_scope(gdb: str, sr: Any, units_per_foot: float,
                timings: Optional[List[Dict[str, Any]]] = None) -> Tuple[Any, Any, List[Any]]:
    """Build review scope with individually timed operations and closed write cursors."""
    steps = timings if timings is not None else []
    stop_fc, stops = prepare_stop_features(gdb, sr, steps)
    with timed_stage("Create combined stop-buffer feature class", steps):
        buffer_fc = create_fc(gdb, "stop_buffers", "POLYGON", sr,
                              [("STOP_ID", "TEXT", 255), ("RADIUS_MI", "DOUBLE", 0)])
    for i, radius in enumerate(sorted(STOP_BUFFER_MILES)):
        temporary = os.path.join(gdb, "buffer_stage_{}".format(i))
        with timed_stage("Build {:.3f}-mile geodesic buffers".format(radius), steps):
            LOGGER.info("Building %.3f-mile buffers for %s stops", radius, len(stops))
            arcpy.analysis.Buffer(stop_fc, temporary, "{} Miles".format(radius),
                                  dissolve_option="NONE", method="GEODESIC")
        with timed_stage("Store {:.3f}-mile stop buffers".format(radius), steps):
            with arcpy.da.InsertCursor(buffer_fc, ["SHAPE@", "STOP_ID", "RADIUS_MI"]) as cursor:
                with arcpy.da.SearchCursor(temporary, ["SHAPE@", "STOP_ID"]) as source:
                    for count, (geometry, sid) in enumerate(source, 1):
                        cursor.insertRow([geometry, sid, radius])
                        if count % 500 == 0 or count == len(stops):
                            LOGGER.info("Stored %.3f-mile buffers: %s / %s", radius,
                                        count, len(stops))
            del cursor, source
        if radius == max(STOP_BUFFER_MILES) or WRITE_CLIPPED_NETWORK:
            with timed_stage("Dissolve {:.3f}-mile scope".format(radius), steps):
                arcpy.management.Dissolve(temporary, radius_scope_path(gdb, radius))
        arcpy.management.Delete(temporary)
    study_fc = os.path.join(gdb, "study_area")
    context_fc = os.path.join(gdb, "context_area")
    with timed_stage("Build context margin", steps):
        arcpy.analysis.Buffer(study_fc, context_fc, CONTEXT_MARGIN_FT * units_per_foot,
                              dissolve_option="ALL", method="PLANAR")
    with timed_stage("Load dissolved study and context polygons", steps):
        study, context = merged_geometry(study_fc), merged_geometry(context_fc)
    LOGGER.info("Study area: %s stops, radii %s miles", len(stops), STOP_BUFFER_MILES)
    return study, context, stops


def geometry_paths(geometry: Any) -> List[List[XY]]:
    """Extract continuous paths; true curves must have been excluded by the caller."""
    paths: List[List[XY]] = []
    for part in geometry:
        points: List[XY] = []
        for point in part:
            if point is None:
                if points:
                    paths.append(points)
                points = []
            else:
                points.append((float(point.X), float(point.Y)))
        if points:
            paths.append(points)
    return paths


def shape_hash(geometry: Any) -> str:
    """Fingerprint the entire projected source geometry to reject stale repairs."""
    return hashlib.sha256(geometry.JSON.encode("utf-8")).hexdigest()


def dimensions(geometry: Any) -> Tuple[bool, bool]:
    """Read documented Esri JSON flags; Geometry.hasZ/hasM are not portable APIs."""
    data = json.loads(geometry.JSON)
    return bool(data.get("hasZ", False)), bool(data.get("hasM", False))


def feature_dimensions(feature: Feature) -> Tuple[bool, bool]:
    """Read schema dimensions once; supported repairs never change dimensions."""
    if feature.dimension_flags is None:
        feature.dimension_flags = dimensions(feature.geometry)
    return feature.dimension_flags


def normalize_level(value: Any) -> str:
    """Treat numeric 0 and 0.0 as the same level; retain nonnumeric level codes."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return format(value, ".15g") if math.isfinite(value) else ""
    return str(value).strip()


@contextmanager
def temporary_feature_layer(path: str, prefix: str, where: str = "") -> Iterator[str]:
    """Create a uniquely named layer and release it when its scope exits.

    Cleanup targets only the layer created here. Unique names permit notebook
    reruns even when an earlier hard interruption prevented cleanup. A cleanup
    failure is logged without replacing the original processing exception.
    """
    name = "{}_{}".format(prefix, uuid.uuid4().hex)
    created = False
    try:
        arcpy.management.MakeFeatureLayer(path, name, where)
        created = True
        yield name
    finally:
        if created:
            try:
                arcpy.management.Delete(name)
            except Exception:
                LOGGER.warning("Temporary-layer cleanup failed: %s", name, exc_info=True)


def load_sources_legacy(gdb: str, sr: Any, context: Any, context_index: Any = None
                 ) -> Tuple[List[Feature], List[Dict[str, Any]]]:
    """Read original IDs/attributes, preflight geometry, and retain whole selected lines."""
    features: List[Feature] = []
    unlocated: List[Dict[str, Any]] = []
    for number, config in enumerate(NETWORK_SOURCES):
        path, label = config["PATH"], config["LABEL"]
        desc = arcpy.Describe(path)
        source_sr = desc.spatialReference
        same_crs = source_sr.exportToString() == sr.exportToString()
        if desc.shapeType != "Polyline":
            raise ValueError("Network input must be a polyline feature class: {}".format(path))
        to_analysis = transformation(source_sr, sr, desc.extent)
        to_source = transformation(sr, source_sr, context.extent)
        with temporary_feature_layer(
            path, "qa_source_{}".format(number), config.get("WHERE", "")
        ) as layer:
            check_table = os.path.join(gdb, "geometry_check_{}".format(number))
            if FULL_SOURCE_GEOMETRY_CHECK:
                LOGGER.info("Checking ALL SQL-selected source geometry: %s", label)
                check_started = time.perf_counter()
                arcpy.management.CheckGeometry(layer, check_table, "ESRI")
                LOGGER.info("Check Geometry completed for %s [%.1f sec]", label,
                            time.perf_counter() - check_started)
            LOGGER.info("Selecting whole %s features near stops before detailed processing", label)
            arcpy.management.SelectLayerByLocation(
                layer, "INTERSECT", project(context, source_sr, to_source),
                selection_type="NEW_SELECTION"
            )
            selected_count = int(arcpy.management.GetCount(layer)[0])
            LOGGER.info("Selected %s context features from %s", selected_count, label)
            if not FULL_SOURCE_GEOMETRY_CHECK and selected_count:
                LOGGER.info("Checking selected context geometry: %s", label)
                check_started = time.perf_counter()
                arcpy.management.CheckGeometry(layer, check_table, "ESRI")
                LOGGER.info("Check Geometry completed for %s [%.1f sec]", label,
                            time.perf_counter() - check_started)
            problems: Dict[str, List[str]] = defaultdict(list)
            if arcpy.Exists(check_table):
                scope = "FULL_SOURCE" if FULL_SOURCE_GEOMETRY_CHECK else "SELECTED_CONTEXT"
                with arcpy.da.SearchCursor(check_table, ["FEATURE_ID", "PROBLEM"]) as cursor:
                    for oid, problem in cursor:
                        problems[str(oid)].append(str(problem))
                        unlocated.append({"SOURCE_LABEL": label, "SOURCE_OID": str(oid),
                                          "PROBLEM": str(problem), "SCOPE": scope})
            names = [f.name for f in arcpy.ListFields(layer)
                     if f.type not in ("OID", "Geometry", "Blob", "Raster")]
            for field_name in (config.get("ID_FIELD", ""), config.get("LEVEL_FIELD", "")):
                if field_name and field_name not in names:
                    raise ValueError("Field {} not found in {}".format(field_name, path))
            seen_ids: Set[str] = set()
            LOGGER.info("Starting feature read/projection: %s (%s selected rows)", label,
                        selected_count)
            with arcpy.da.SearchCursor(layer, ["OID@", "SHAPE@"] + names) as cursor:
                for count, row in enumerate(cursor, 1):
                    if count % 10000 == 0:
                        LOGGER.info("Reading/projecting %s: %s / %s", label, count, selected_count)
                    oid, geometry = str(row[0]), row[1]
                    attrs = dict(zip(names, row[2:]))
                    if geometry is None or geometry.pointCount == 0:
                        continue  # Unlocatable problems require the full-source preflight.
                    identity = attrs.get(config.get("ID_FIELD", ""), oid)
                    source_id = str(identity) if identity is not None else ""
                    if not source_id or source_id in seen_ids:
                        raise ValueError("Blank/duplicate source ID in selected {}: {}".format(
                            label, source_id
                        ))
                    seen_ids.add(source_id)
                    if len(source_id) > 400:
                        raise ValueError("Source ID exceeds the 400-character output field.")
                    key = label + ":" + source_id
                    if len(key) > 500:
                        raise ValueError("SOURCE_KEY exceeds 500 characters: {}".format(key))
                    raw_level = attrs.get(config.get("LEVEL_FIELD", ""))
                    level = normalize_level(raw_level)
                    if len(level) > 100:
                        raise ValueError("Level value exceeds the 100-character output field.")
                    if not same_crs:
                        geometry = geometry.projectAs(sr, to_analysis) if to_analysis else (
                            geometry.projectAs(sr)
                        )
                    scope = context_index if context_index is not None else context
                    if area_disjoint(scope, geometry):
                        continue
                    feature = Feature(key, label, oid, source_id, level, geometry, geometry,
                                      attrs, problems.get(oid, []),
                                      dimension_flags=(bool(desc.hasZ), bool(desc.hasM)))
                    features.append(feature)
        LOGGER.info("Loaded %s; cumulative selected features: %s", label, len(features))
    return features, unlocated


class SourceProgress:
    """Report completed rows and measured throughput without predicting later stages."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total = label, total
        self.started = self.last_log = time.perf_counter()
        self.last_count = -1

    def update(self, count: int, force: bool = False) -> None:
        """Log early milestones, configured row intervals, and elapsed-time intervals."""
        now = time.perf_counter()
        if count == self.last_count:
            return
        if force or count in (1, 100, self.total) or count % SOURCE_PROGRESS_ROWS == 0 or (
            now - self.last_log >= PROGRESS_LOG_SECONDS
        ):
            elapsed = max(now - self.started, 1e-9)
            LOGGER.info("%s: %s / %s rows [%.1f sec; %.0f rows/sec]", self.label,
                        count, self.total, elapsed, count / elapsed)
            self.last_log, self.last_count = now, count


def read_source_attributes(layer: str, config: Dict[str, Any], desc: Any,
                           problems: Dict[str, List[str]], total: int) -> Dict[str, Feature]:
    """Read original attributes and IDs without constructing source geometry objects."""
    names = [f.name for f in arcpy.ListFields(layer)
             if f.type not in ("OID", "Geometry", "Blob", "Raster")]
    for name in (config.get("ID_FIELD", ""), config.get("LEVEL_FIELD", "")):
        if name and name not in names:
            raise ValueError("Field {} not found in {}".format(name, config["PATH"]))
    label = config["LABEL"]
    flags = bool(desc.hasZ), bool(desc.hasM)
    records: Dict[str, Feature] = {}
    seen_ids: Set[str] = set()
    progress = SourceProgress("Read {} attributes".format(label), total)
    with arcpy.da.SearchCursor(layer, ["OID@"] + names) as cursor:
        for count, row in enumerate(cursor, 1):
            oid, attrs = str(row[0]), dict(zip(names, row[1:]))
            identity = attrs.get(config.get("ID_FIELD", ""), oid)
            source_id = str(identity) if identity is not None else ""
            if not source_id or source_id in seen_ids or oid in records:
                raise ValueError("Blank/duplicate source ID in selected {}: {}".format(
                    label, source_id
                ))
            seen_ids.add(source_id)
            key = label + ":" + source_id
            level = normalize_level(attrs.get(config.get("LEVEL_FIELD", "")))
            if len(source_id) > 400 or len(key) > 500 or len(level) > 100:
                raise ValueError("Source ID, key, or level exceeds output schema: {}".format(key))
            records[oid] = Feature(key, label, oid, source_id, level, None, None, attrs,
                                   list(problems.get(oid, [])), dimension_flags=flags)
            progress.update(count)
    if len(records) != total:
        raise ValueError("Selected row count changed while reading {}".format(label))
    progress.update(len(records), force=True)
    return records


def export_source_stage(layer: str, gdb: str, name: str, desc: Any) -> str:
    """Export selected whole geometry and the original OID without editing the source.

    Only the lineage field is exported. Original attributes are read separately,
    so field renaming, recalculated shape fields, and domains cannot alter them.
    """
    oid_map = arcpy.FieldMap()
    oid_map.addInputField(layer, desc.OIDFieldName)
    output_field = oid_map.outputField
    output_field.name = "QA_ORIG_ID"
    output_field.aliasName = "Original source ObjectID"
    output_field.type = "String"
    output_field.length = 40
    output_field.required = False
    output_field.editable = True
    oid_map.outputField = output_field
    mappings = arcpy.FieldMappings()
    mappings.addFieldMap(oid_map)
    # Explicitly keep the source CRS here: the outer environment uses the analysis CRS.
    with arcpy.EnvManager(outputCoordinateSystem=desc.spatialReference, extent=None,
                          outputZFlag="Same As Input", outputMFlag="Same As Input"):
        arcpy.conversion.FeatureClassToFeatureClass(layer, gdb, name, field_mapping=mappings)
    return os.path.join(gdb, name)


def verify_stage_dimensions(path: str, flags: Tuple[bool, bool]) -> None:
    """Reject an intermediate copy that lost or invented Z/M dimensions."""
    desc = arcpy.Describe(path)
    if (bool(desc.hasZ), bool(desc.hasM)) != flags:
        raise ValueError("Z/M dimensions changed during staging/projection: {}".format(path))


def verify_staged_geometry(path: str, records: Dict[str, Feature], label: str
                           ) -> Dict[str, int]:
    """Check source IDs, retain curve exclusions, and record original vertex counts."""
    counts: Dict[str, int] = {}
    progress = SourceProgress("Verify staged {} geometry".format(label), len(records))
    with arcpy.da.SearchCursor(path, ["QA_ORIG_ID", "SHAPE@"]) as cursor:
        for count, (raw_oid, geometry) in enumerate(cursor, 1):
            oid = str(raw_oid)
            if oid not in records or oid in counts:
                raise ValueError("Invalid or duplicate staged source OID: {}".format(oid))
            if geometry is None or geometry.pointCount == 0:
                raise ValueError("Empty staged geometry for {}:{}; working copy retained.".format(
                    label, oid
                ))
            counts[oid] = geometry.pointCount
            records[oid].source_has_curves = bool(geometry.hasCurves)
            progress.update(count)
    if counts.keys() != records.keys():
        raise ValueError("Staging omitted selected source IDs from {}".format(label))
    progress.update(len(counts), force=True)
    return counts


def projected_geometry_check(path: str, table: str, records: Dict[str, Feature],
                             label: str) -> List[Dict[str, Any]]:
    """Check projected geometry and translate its OIDs back to original source IDs."""
    arcpy.management.CheckGeometry(path, table, "ESRI")
    by_output_oid: Dict[str, List[str]] = defaultdict(list)
    with arcpy.da.SearchCursor(table, ["FEATURE_ID", "PROBLEM"]) as cursor:
        for oid, problem in cursor:
            by_output_oid[str(oid)].append(str(problem))
    seen: Set[str] = set()
    problem_lineage: Dict[str, str] = {}
    reports: List[Dict[str, Any]] = []
    progress = SourceProgress("Verify projected {} IDs".format(label), len(records))
    with arcpy.da.SearchCursor(path, ["OID@", "QA_ORIG_ID"]) as cursor:
        for count, (output_oid, raw_oid) in enumerate(cursor, 1):
            oid = str(raw_oid)
            if oid not in records or oid in seen:
                raise ValueError("Missing, altered, or duplicated projected source OID: {}".format(
                    oid
                ))
            seen.add(oid)
            for problem in by_output_oid.get(str(output_oid), []):
                problem_lineage[str(output_oid)] = oid
                records[oid].problems.append("PROJECTED: " + problem)
                reports.append({"SOURCE_LABEL": label, "SOURCE_OID": oid,
                                "PROBLEM": problem, "SCOPE": "PROJECTED_CONTEXT"})
            progress.update(count)
    if seen != records.keys():
        raise ValueError("Projection omitted selected source IDs from {}".format(label))
    if by_output_oid.keys() != problem_lineage.keys():
        raise ValueError("Could not link all projected geometry problems to source IDs.")
    arcpy.management.AddField(table, "SOURCE_OID", "TEXT", field_length=40)
    with arcpy.da.UpdateCursor(table, ["FEATURE_ID", "SOURCE_OID"]) as cursor:
        for row in cursor:
            row[1] = problem_lineage[str(row[0])]
            cursor.updateRow(row)
    if any("null geometry" in r["PROBLEM"].lower() or "empty geometry" in r["PROBLEM"].lower()
           for r in reports):
        raise ValueError("Projection produced empty geometry; inspect {}".format(table))
    progress.update(len(seen), force=True)
    return reports


def read_projected_selection(layer: str, records: Dict[str, Feature], counts: Dict[str, int],
                             label: str, total: int) -> List[Dict[str, Any]]:
    """Load already projected geometry without per-row projection or spatial predicates."""
    seen: Set[str] = set()
    reports: List[Dict[str, Any]] = []
    progress = SourceProgress("Read projected {} geometry".format(label), total)
    with arcpy.da.SearchCursor(layer, ["QA_ORIG_ID", "SHAPE@"]) as cursor:
        for count, (raw_oid, geometry) in enumerate(cursor, 1):
            oid = str(raw_oid)
            if oid not in records or oid in seen:
                raise ValueError("Invalid lineage in selected projected geometry: {}".format(oid))
            seen.add(oid)
            if geometry is None or geometry.pointCount == 0:
                raise ValueError("Empty projected geometry for {}:{}".format(label, oid))
            feature = records[oid]
            feature.geometry = feature.original = geometry
            if not feature.source_has_curves and geometry.pointCount != counts[oid]:
                problem = "STAGING_TO_PROJECTION_VERTEX_COUNT_CHANGED"
                feature.problems.append(problem)
                reports.append({"SOURCE_LABEL": label, "SOURCE_OID": oid,
                                "PROBLEM": problem, "SCOPE": "PROJECTED_CONTEXT"})
            progress.update(count)
    if len(seen) != total:
        raise ValueError("Projected selection changed while reading {}".format(label))
    progress.update(len(seen), force=True)
    return reports


def load_sources(gdb: str, sr: Any, context: Any, context_index: Any = None,
                 timings: Optional[List[Dict[str, Any]]] = None
                 ) -> Tuple[List[Feature], List[Dict[str, Any]]]:
    """Stage selected whole lines locally and project each source in one native call.

    Original OIDs travel in QA_ORIG_ID, never inferred from new OIDs or cursor
    order. Attribute dictionaries retain original source field names and values.
    Intermediate copies survive failures; the input datasets remain read-only.
    """
    steps = timings if timings is not None else []
    if not BATCH_PROJECT_SOURCES:
        with timed_stage("Legacy per-feature source loading", steps):
            return load_sources_legacy(gdb, sr, context, context_index)
    features: List[Feature] = []
    preflight: List[Dict[str, Any]] = []
    for number, config in enumerate(NETWORK_SOURCES):
        path, label = config["PATH"], config["LABEL"]
        desc = arcpy.Describe(path)
        if desc.shapeType != "Polyline":
            raise ValueError("Network input must be a polyline feature class: {}".format(path))
        source_sr = desc.spatialReference
        same_crs = source_sr.exportToString() == sr.exportToString()
        flags = bool(desc.hasZ), bool(desc.hasM)
        forward = transformation(source_sr, sr, desc.extent)
        reverse = transformation(sr, source_sr, context.extent)
        with temporary_feature_layer(path, "qa_source_{}".format(number),
                                     config.get("WHERE", "")) as layer:
            table = os.path.join(gdb, "geometry_check_{}".format(number))
            if FULL_SOURCE_GEOMETRY_CHECK:
                with timed_stage("Check full-source {} geometry".format(label), steps):
                    arcpy.management.CheckGeometry(layer, table, "ESRI")
            with timed_stage("Select {} features near stops".format(label), steps):
                arcpy.management.SelectLayerByLocation(
                    layer, "INTERSECT", project(context, source_sr, reverse),
                    selection_type="NEW_SELECTION"
                )
                total = int(arcpy.management.GetCount(layer)[0])
                LOGGER.info("Selected %s context features from %s", total, label)
            if not FULL_SOURCE_GEOMETRY_CHECK and total:
                with timed_stage("Check selected {} geometry".format(label), steps):
                    arcpy.management.CheckGeometry(layer, table, "ESRI")
            problems: Dict[str, List[str]] = defaultdict(list)
            if arcpy.Exists(table):
                scope = "FULL_SOURCE" if FULL_SOURCE_GEOMETRY_CHECK else "SELECTED_CONTEXT"
                with arcpy.da.SearchCursor(table, ["FEATURE_ID", "PROBLEM"]) as cursor:
                    for oid, problem in cursor:
                        problems[str(oid)].append(str(problem))
                        preflight.append({"SOURCE_LABEL": label, "SOURCE_OID": str(oid),
                                          "PROBLEM": str(problem), "SCOPE": scope})
            if not total:
                LOGGER.info("No selected %s features; skipping export/projection", label)
                continue
            with timed_stage("Read {} source attributes (no geometry)".format(label), steps):
                records = read_source_attributes(layer, config, desc, problems, total)
            with timed_stage("Copy selected {} geometry to local staging".format(label), steps):
                native = export_source_stage(layer, gdb, "source_stage_{}_native".format(number),
                                             desc)
                verify_stage_dimensions(native, flags)
        with timed_stage("Verify staged {} IDs and geometry".format(label), steps):
            counts = verify_staged_geometry(native, records, label)
        projected = native
        if not same_crs:
            projected = os.path.join(gdb, "source_stage_{}_projected".format(number))
            with timed_stage("Batch-project {} geometry".format(label), steps):
                arcpy.management.Project(native, projected, sr, transform_method=forward,
                                         preserve_shape="NO_PRESERVE_SHAPE", vertical="NO_VERTICAL")
                verify_stage_dimensions(projected, flags)
        with timed_stage("Check projected {} geometry and lineage".format(label), steps):
            table = os.path.join(gdb, "geometry_check_projected_{}".format(number))
            preflight.extend(projected_geometry_check(projected, table, records, label))
        with temporary_feature_layer(projected, "qa_projected_{}".format(number)) as layer:
            with timed_stage("Filter projected {} by context".format(label), steps):
                arcpy.management.SelectLayerByLocation(
                    layer, "INTERSECT", os.path.join(gdb, "context_area"),
                    selection_type="NEW_SELECTION"
                )
                retained = int(arcpy.management.GetCount(layer)[0])
                LOGGER.info("Retained %s / %s projected %s features", retained, total, label)
            with timed_stage("Read local projected {} geometry".format(label), steps):
                preflight.extend(read_projected_selection(layer, records, counts, label, retained))
        # Preserve source cursor order even when native export/Project reorders rows.
        features.extend(f for f in records.values() if f.geometry is not None)
        if not KEEP_SOURCE_STAGING:
            with timed_stage("Remove completed {} staging copies".format(label), steps):
                for stage in dict.fromkeys((native, projected)):
                    arcpy.management.Delete(stage)
        LOGGER.info("Loaded %s; cumulative retained features: %s", label, len(features))
    return features, preflight


class RoadContext:
    """Use road polygons to annotate gaps; never use them as walking edges."""

    def __init__(self, gdb: str, sr: Any, context: Any, units_per_foot: float,
                 context_index: Any = None) -> None:
        self.sr, self.units_per_foot = sr, units_per_foot
        self.geometries: List[Any] = []
        self.grid = Grid(SPATIAL_GRID_FT * units_per_foot)
        if not ROAD_POLYGONS:
            return
        desc = arcpy.Describe(ROAD_POLYGONS)
        if desc.shapeType != "Polygon":
            raise ValueError("ROAD_POLYGONS must be a polygon feature class.")
        method = transformation(desc.spatialReference, sr, desc.extent)
        reverse = transformation(sr, desc.spatialReference, context.extent)
        same_crs = desc.spatialReference.exportToString() == sr.exportToString()
        with temporary_feature_layer(ROAD_POLYGONS, "qa_roads", ROAD_WHERE) as layer:
            arcpy.management.SelectLayerByLocation(
                layer, "INTERSECT", project(context, desc.spatialReference, reverse),
                selection_type="NEW_SELECTION"
            )
            output = create_fc(gdb, "road_context", "POLYGON", sr, [("SOURCE_OID", "TEXT", 40)])
            with arcpy.da.InsertCursor(output, ["SHAPE@", "SOURCE_OID"]) as write:
                with arcpy.da.SearchCursor(layer, ["OID@", "SHAPE@"]) as rows:
                    for oid, geometry in rows:
                        if geometry is None or geometry.pointCount == 0:
                            raise ValueError("Null selected road geometry: {}".format(oid))
                        if not same_crs:
                            geometry = geometry.projectAs(sr, method) if method else (
                                geometry.projectAs(sr)
                            )
                        scope = context_index if context_index is not None else context
                        if area_disjoint(scope, geometry):
                            continue
                        idx = len(self.geometries)
                        self.geometries.append(geometry)
                        self.grid.add(idx, geom_box(geometry))
                        write.insertRow([geometry, str(oid)])
        check = os.path.join(gdb, "geometry_check_roads")
        arcpy.management.CheckGeometry(output, check, "ESRI")
        if int(arcpy.management.GetCount(check)[0]):
            raise ValueError("Road polygons have invalid geometry; inspect {}".format(check))
        LOGGER.info("Loaded %s road polygons", len(self.geometries))
        if not self.geometries:
            LOGGER.warning("No road polygons in context; road-based crossing checks unavailable.")

    def overlap_ft(self, points: Sequence[XY]) -> float:
        """Measure road-covered proposed gap length without double-counting polygons."""
        if not self.geometries or len(points) < 2 or points[0] == points[-1]:
            return 0.0
        candidates = self.grid.query(bbox(points))
        if not candidates:
            return 0.0
        line = line_geom(points, self.sr)
        covered = None
        for idx in candidates:
            polygon = self.geometries[idx]
            if line.disjoint(polygon):
                continue
            piece = line.intersect(polygon, 2)
            if piece is not None and piece.length > 0:
                covered = piece if covered is None else covered.union(piece)
        return 0.0 if covered is None else covered.length / self.units_per_foot

    def near(self, point: XY, distance_ft: float) -> bool:
        """Return whether a point is inside or close to any road polygon."""
        radius = distance_ft * self.units_per_foot
        candidates = self.grid.query(bbox([point], radius))
        if not candidates:
            return False
        p = point_geom(point, self.sr)
        return any(p.distanceTo(self.geometries[i]) <= radius
                   for i in candidates)


def make_parts(features: List[Feature], epsilon: float) -> Tuple[List[Part], List[Issue]]:
    """Build continuous parts while explicitly excluding unsupported/problem geometry."""
    parts: List[Part] = []
    issues: List[Issue] = []
    for feature in features:
        geom = feature.geometry
        if feature.problems:
            feature.excluded_reason = "CHECK_GEOMETRY: " + "; ".join(feature.problems)
        elif feature.source_has_curves or geom.hasCurves:
            feature.excluded_reason = "TRUE_CURVE_NOT_AUDITED"
        elif geom.length <= epsilon:
            feature.excluded_reason = "ZERO_LENGTH"
        else:
            feature.excluded_reason = ""
        if feature.excluded_reason:
            p = geom.firstPoint
            issues.append(Issue("EXCLUDED_GEOMETRY", [(p.X, p.Y)], confidence="CONFIRMED",
                                detail=feature.excluded_reason, feature_key=feature.key))
            continue
        paths = geometry_paths(geom)
        if len(paths) > 1:
            p = geom.firstPoint
            issues.append(Issue("MULTIPART_FEATURE", [(p.X, p.Y)],
                                detail="Parts are separate; multipart does not imply connectivity.",
                                feature_key=feature.key))
        for part_number, raw_points in enumerate(paths):
            points = deduplicate_consecutive(raw_points)
            length = sum(distance(a, b) for a, b in zip(points, points[1:]))
            if len(points) < 2 or length <= epsilon:
                issues.append(Issue("DEGENERATE_PART", points[:1], confidence="CONFIRMED",
                                    detail="Part excluded from graph: no measurable line length.",
                                    feature_key=feature.key))
                continue
            part = Part(len(parts), feature.key, part_number, points, feature.level)
            parts.append(part)
            if len(points) < len(raw_points):
                issues.append(Issue("REPEATED_VERTEX", points[:1], part.pid,
                                    confidence="CONFIRMED",
                                    detail="Consecutive identical XY vertices; source unedited."))
    return parts, issues


def scope_issue(issue: Issue, study: Any, sr: Any,
                features: Optional[Dict[str, Feature]] = None) -> bool:
    """Keep spatially relevant issues and retain unlocatable evidence in the CSV."""
    if not issue.points:
        return True
    if issue.feature_key and features and issue.feature_key in features:
        # An excluded/multipart feature's first point can lie outside the scope
        # even though its middle is inside. Do not silently lose that warning.
        return not area_disjoint(study, features[issue.feature_key].geometry)
    geometry = issue_geometry(issue, sr)
    return geometry is None or not area_disjoint(study, geometry)


def identify_issue(issue: Issue, parts: List[Part]) -> str:
    """Create a deterministic ID from lineage, issue type, and precise location."""
    owners = [(parts[i].feature_key, parts[i].part_number)
              for i in (issue.pid, issue.other_pid) if i >= 0]
    data = [issue.kind, sorted(owners), issue.feature_key,
            [[round(x, 8), round(y, 8)] for x, y in issue.points]]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]


def scan_gaps(network: Network, roads: RoadContext, units_per_foot: float) -> List[Issue]:
    """Inspect degree-one endpoints for gaps, potential crossings, and road-edge ends."""
    issues: List[Issue] = []
    seen_pairs: Set[Tuple[Any, ...]] = set()
    radius = (max(GAP_REVIEW_FT, CROSSING_SEARCH_FT) if roads.geometries else GAP_REVIEW_FT)
    dangles = network.dangles()
    for count, (pid, endpoint, point) in enumerate(dangles):
        if count and count % 5000 == 0:
            LOGGER.info("Gap review: %s / %s endpoints", count, len(dangles))
        part = network.parts[pid]
        issues.append(Issue("DANGLING_ENDPOINT", [point], pid,
                            detail="Degree-one endpoint; may be an intentional sidewalk end."))
        near_road = roads.near(point, ROAD_EDGE_REVIEW_FT)
        if near_road:
            issues.append(Issue("END_NEAR_ROAD", [point], pid,
                                detail="Check crossing/curb access; proximity is not proof."))
        nearby = network.nearest_parts(point, radius * units_per_foot)
        for other_pid, (gap, target) in nearby.items():
            if other_pid == pid:
                continue
            other = network.parts[other_pid]
            if gap <= network.epsilon:
                continue  # Exact but unnoded intersections are handled by scan_intersections.
            road_ft = roads.overlap_ft([point, target])
            if gap / units_per_foot > GAP_REVIEW_FT and road_ft < MIN_ROAD_OVERLAP_FT:
                continue
            target_end = min((0, len(other.points) - 1),
                             key=lambda i: distance(target, other.points[i]))
            to_endpoint = distance(target, other.points[target_end]) <= network.epsilon
            if to_endpoint:
                pair = tuple(sorted([(pid, endpoint), (other_pid, target_end)]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
            if road_ft >= MIN_ROAD_OVERLAP_FT:
                code = "POSSIBLE_MISSING_CROSSING"
            else:
                code = "NEAR_ENDPOINT_GAP" if to_endpoint else "NEAR_EDGE_GAP"
            detail = "Nearby endpoint is disconnected here. "
            detail += ("Candidate crosses road polygon area; verify grade and permitted crossing."
                       if road_ft >= MIN_ROAD_OVERLAP_FT else
                       "Check intended continuity, barriers, grade, and alternative targets.")
            if network.components.find(pid) == network.components.find(other_pid):
                detail += " Both parts already connect elsewhere in the retained graph."
            if part.level and other.level and part.level != other.level:
                detail += " Known levels differ; do not snap."
            issues.append(Issue(code, [point, target], pid, other_pid, gap / units_per_foot,
                                "REVIEW", detail, road_ft))
    return issues


# =============================================================================
# CONSERVATIVE REPAIRS -- EXPLICITLY OPT-IN, NEVER SOURCE EDITS
# =============================================================================


def rebuild_2d(paths: Sequence[Sequence[XY]], sr: Any) -> Any:
    """Rebuild a straight 2D feature; callers must reject curves and Z/M geometry."""
    return arcpy.Polyline(arcpy.Array([
        arcpy.Array([arcpy.Point(*p) for p in path]) for path in paths
    ]), sr)


def remove_repeated_vertices(features: List[Feature], sr: Any) -> List[Dict[str, Any]]:
    """Optionally remove exact consecutive repeats from otherwise valid 2D geometry."""
    changes: List[Dict[str, Any]] = []
    if not REMOVE_EXACT_DUPLICATE_VERTICES:
        return changes
    for feature in features:
        geom = feature.geometry
        if feature.source_has_curves or geom.hasCurves or any(feature_dimensions(feature)):
            continue
        if any("duplicate vertex" not in p.lower() for p in feature.problems):
            continue
        raw = geometry_paths(geom)
        cleaned = [deduplicate_consecutive(p) for p in raw]
        if raw == cleaned or any(len(p) < 2 for p in cleaned):
            continue
        revised = rebuild_2d(cleaned, sr)
        # ArcPy constructors can simplify shapes. Reject any unrequested change.
        if geometry_paths(revised) != cleaned or not revised.equals(geom):
            LOGGER.warning("Skipped duplicate cleanup that changed geometry: %s", feature.key)
            continue
        feature.geometry = revised
        feature.problems = []
        changes.append({"SOURCE_KEY": feature.key, "ACTION": "REMOVE_REPEATED_VERTICES",
                        "MOVE_FT": 0, "OLD_JSON": geom.JSON, "NEW_JSON": revised.JSON})
    return changes


REPAIR_FIELDS = ["APPROVE", "ISSUE_ID", "SOURCE_KEY", "SOURCE_PART", "SOURCE_ENDPOINT",
                 "TARGET_KEY", "TARGET_PART", "TARGET_ENDPOINT", "GAP_FT", "SOURCE_HASH",
                 "TARGET_HASH", "REASON"]


def repair_candidates(issues: List[Issue], network: Network, features: Dict[str, Feature],
                      units_per_foot: float) -> List[Dict[str, Any]]:
    """Offer only tiny, unique, 2D endpoint pairs for explicit approval."""
    output: List[Dict[str, Any]] = []
    for issue in issues:
        if issue.kind != "NEAR_ENDPOINT_GAP" or issue.value_ft is None:
            continue
        if issue.value_ft > MAX_APPROVED_SNAP_FT or issue.road_ft > 0:
            continue
        first, second = network.parts[issue.pid], network.parts[issue.other_pid]
        a, b = features[first.feature_key], features[second.feature_key]
        if a.key == b.key or a.level != b.level:
            continue
        if any(any(feature_dimensions(f)) or f.source_has_curves or f.geometry.hasCurves
               for f in (a, b)):
            continue
        ends = []
        safe = True
        for part, point, other_pid in ((first, issue.points[0], second.pid),
                                       (second, issue.points[-1], first.pid)):
            endpoint = min((0, len(part.points) - 1), key=lambda i: distance(part.points[i], point))
            if distance(part.points[endpoint], point) > network.epsilon:
                safe = False
            if network.degree(part.pid, point) != 1:
                safe = False
            nearby = network.nearest_parts(point, GAP_REVIEW_FT * units_per_foot)
            if set(nearby) - {part.pid} != {other_pid}:
                safe = False
            ends.append("START" if endpoint == 0 else "END")
        if safe:
            output.append({"APPROVE": "", "ISSUE_ID": issue.issue_id, "SOURCE_KEY": a.key,
                           "SOURCE_PART": first.part_number, "SOURCE_ENDPOINT": ends[0],
                           "TARGET_KEY": b.key, "TARGET_PART": second.part_number,
                           "TARGET_ENDPOINT": ends[1], "GAP_FT": issue.value_ft,
                           "SOURCE_HASH": shape_hash(a.geometry),
                           "TARGET_HASH": shape_hash(b.geometry),
                           "REASON": "Verify same level, no barrier, and intended continuity."})
    return output


def apply_approved_snaps(candidates: List[Dict[str, Any]], features: Dict[str, Feature],
                         network: Network, sr: Any, units_per_foot: float) -> List[Dict[str, Any]]:
    """Validate ALL approvals before applying endpoint moves to in-memory copies.

    Feature-disjoint approvals prevent chained/order-dependent snapping. A moved
    terminal segment may not acquire new intersections other than its intended
    target endpoint. Every source/target fingerprint must still match.
    """
    if not APPROVED_SNAPS_CSV:
        return []
    with open(APPROVED_SNAPS_CSV, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(REPAIR_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Approved snaps CSV must use the exported candidate schema.")
        approvals = [r for r in reader if r["APPROVE"].strip().upper() == "YES"]
    lookup = {r["ISSUE_ID"]: r for r in candidates}
    used: Set[str] = set()
    pending: List[Tuple[Feature, Any, Dict[str, Any]]] = []
    for row in approvals:
        current = lookup.get(row["ISSUE_ID"])
        if current is None:
            raise ValueError("Approved candidate no longer eligible: {}".format(row["ISSUE_ID"]))
        for name in REPAIR_FIELDS:
            if name not in ("APPROVE", "REASON", "GAP_FT") and str(current[name]) != row[name]:
                raise ValueError("Stale approval field {} for {}".format(name, row["ISSUE_ID"]))
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
        if distance(new, neighbor) <= network.epsilon:
            raise ValueError("Approved snap would collapse its terminal segment.")
        for sid in network.segment_grid.query(bbox([neighbor, new], network.epsilon)):
            seg = network.segments[sid]
            owner = network.parts[seg.pid]
            is_moving = owner.feature_key == source.key and owner.part_number == part_no and (
                seg.position == (0 if endpoint == 0 else len(paths[part_no]) - 2)
            )
            if is_moving:
                continue
            kind, hits = segment_intersection(neighbor, new, seg.a, seg.b, network.epsilon)
            if kind == "NONE":
                continue
            allowed_target = kind == "POINT" and owner.feature_key == target.key and all(
                distance(p, new) <= network.epsilon for p in hits
            )
            old_kind, old_hits = segment_intersection(neighbor, old, seg.a, seg.b, network.epsilon)
            unchanged = kind == old_kind == "POINT" and all(
                any(distance(p, q) <= network.epsilon for q in old_hits) for p in hits
            )
            if not allowed_target and not unchanged:
                raise ValueError("Snap creates an unintended intersection/overlap: {}".format(
                    row["ISSUE_ID"]
                ))
        paths[part_no][endpoint] = new
        revised = rebuild_2d(paths, sr)
        actual = geometry_paths(revised)
        if len(actual) != len(paths) or any(len(a) != len(b) for a, b in zip(actual, paths)):
            raise ValueError("ArcPy simplified an approved snap unexpectedly.")
        if any(distance(a, b) > network.epsilon for pa, pb in zip(actual, paths)
               for a, b in zip(pa, pb)):
            raise ValueError("ArcPy moved additional coordinates during an approved snap.")
        change = {"SOURCE_KEY": source.key, "ACTION": "APPROVED_ENDPOINT_SNAP",
                  "MOVE_FT": distance(old, new) / units_per_foot,
                  "OLD_JSON": source.geometry.JSON, "NEW_JSON": revised.JSON}
        pending.append((source, revised, change))
    for source, revised, _ in pending:
        source.geometry = revised
    return [change for _, _, change in pending]


# =============================================================================
# OUTPUTS, COMPONENTS, AND STOP-LEVEL COVERAGE
# =============================================================================


def component_outputs(
    gdb: str, network: Network, study: Any, context: Any, sr: Any, units_per_foot: float
) -> Tuple[Dict[int, int], List[Dict[str, Any]], List[Issue]]:
    """Write part components and flag small groups without penalizing boundary cuts."""
    groups: Dict[int, List[Part]] = defaultdict(list)
    for part in network.parts:
        groups[network.components.find(part.pid)].append(part)
    # Stable ordering by original source keys/part numbers, not accidental cursor order.
    ordered = sorted(groups.values(), key=lambda g: min((p.feature_key, p.part_number) for p in g))
    fields = [("SOURCE_KEY", "TEXT", 500), ("PART_NO", "LONG", 0), ("COMP_ID", "LONG", 0),
              ("CONTEXT_EDGE", "SHORT", 0), ("LENGTH_FT", "DOUBLE", 0),
              ("STUDY_FT", "DOUBLE", 0)]
    path = create_fc(gdb, "network_components", "POLYLINE", sr, fields)
    part_to_component, summaries, issues = {}, [], []
    boundary = None if isinstance(context, AreaIndex) else context.boundary()
    completed_parts = 0
    with arcpy.da.InsertCursor(path, ["SHAPE@"] + [f[0] for f in fields]) as cursor:
        for component_id, group in enumerate(ordered, 1):
            total, in_study, edge = 0.0, 0.0, False
            records = []
            for part in group:
                geometry = network.geometry(part.pid, sr)
                length_ft = geometry.length / units_per_foot
                study_length = study.clipped_length(geometry) if isinstance(study, AreaIndex) else (
                    geometry.intersect(study, 2).length
                )
                study_ft = study_length / units_per_foot
                touches = context.touches_edge(geometry) if isinstance(context, AreaIndex) else (
                    not geometry.disjoint(boundary) or not geometry.within(context)
                )
                edge = edge or touches
                total += length_ft
                in_study += study_ft
                part_to_component[part.pid] = component_id
                records.append((part, geometry, length_ft, study_ft))
                completed_parts += 1
                if completed_parts % 10000 == 0:
                    LOGGER.info("Component coverage: %s / %s parts", completed_parts,
                                len(network.parts))
            summaries.append({"COMP_ID": component_id, "PART_COUNT": len(group),
                              "FEATURE_COUNT": len({p.feature_key for p in group}),
                              "CONTEXT_LENGTH_FT": total, "STUDY_LENGTH_FT": in_study,
                              "CONTEXT_EDGE": int(edge)})
            for part, geometry, length_ft, study_ft in records:
                cursor.insertRow([geometry, part.feature_key, part.part_number, component_id,
                                  int(edge), length_ft, study_ft])
            if in_study > 0 and total <= SMALL_COMPONENT_LENGTH_FT and not edge:
                representative = next(p for p, _, _, sf in records if sf > 0)
                issues.append(Issue("SMALL_DISCONNECTED_GROUP", representative.points,
                                    representative.pid, value_ft=total,
                                    detail="Small separate component; review nearby links."))
            elif in_study > 0 and len(group) == 1 and not edge:
                issues.append(Issue("ISOLATED_PART", group[0].points, group[0].pid,
                                    value_ft=total,
                                    detail="Part connects to no other audited part."))
    return part_to_component, summaries, issues


NETWORK_FIELDS = [("SOURCE_KEY", "TEXT", 500), ("SOURCE_LABEL", "TEXT", 80),
                  ("SOURCE_OID", "TEXT", 40), ("SOURCE_ID", "TEXT", 400),
                  ("LEVEL_VALUE", "TEXT", 100), ("QA_STATE", "TEXT", 500)]


def write_network(gdb: str, name: str, features: List[Feature], sr: Any,
                  original: bool = False) -> List[str]:
    """Preserve complete projected features and their Z/M dimensions in a new output."""
    groups: Dict[Tuple[bool, bool], List[Feature]] = defaultdict(list)
    for feature in features:
        groups[feature_dimensions(feature)].append(feature)
    if not groups:
        groups[(False, False)] = []
    paths = []
    for (has_z, has_m), group in sorted(groups.items()):
        suffix = ("_" + ("z" if has_z else "") + ("m" if has_m else "")) if (
            has_z or has_m
        ) else ""
        path = create_fc(gdb, name + suffix, "POLYLINE", sr, NETWORK_FIELDS, has_z, has_m)
        with arcpy.da.InsertCursor(path, ["SHAPE@"] + [f[0] for f in NETWORK_FIELDS]) as cursor:
            for feature in group:
                cursor.insertRow([feature.original if original else feature.geometry, feature.key,
                                  feature.label, feature.oid, feature.source_id, feature.level,
                                  feature.excluded_reason or "XY_DIAGNOSTIC_ONLY"])
        paths.append(path)
    return paths


ISSUE_FIELDS = [("ISSUE_ID", "TEXT", 40), ("ISSUE_TYPE", "TEXT", 60),
                ("CONFIDENCE", "TEXT", 30), ("SOURCE_KEY", "TEXT", 500),
                ("OTHER_KEY", "TEXT", 500), ("PART_NO", "LONG", 0),
                ("OTHER_PART", "LONG", 0), ("VALUE_FT", "DOUBLE", 0),
                ("ROAD_FT", "DOUBLE", 0), ("DETAIL", "TEXT", 1500),
                ("REVIEW_STATUS", "TEXT", 30), ("MAP_GEOM", "TEXT", 30),
                ("GEOM_NOTE", "TEXT", 1000)]


def issue_record(issue: Issue, parts: List[Part]) -> Dict[str, Any]:
    """Build a source-linked row shared by the CSV and map outputs."""
    first = parts[issue.pid] if issue.pid >= 0 else None
    second = parts[issue.other_pid] if issue.other_pid >= 0 else None
    return {"ISSUE_ID": issue.issue_id, "ISSUE_TYPE": issue.kind,
            "CONFIDENCE": issue.confidence,
            "SOURCE_KEY": first.feature_key if first else issue.feature_key,
            "OTHER_KEY": second.feature_key if second else "",
            "PART_NO": first.part_number if first else None,
            "OTHER_PART": second.part_number if second else None,
            "VALUE_FT": issue.value_ft, "ROAD_FT": issue.road_ft, "DETAIL": issue.detail,
            "REVIEW_STATUS": "OPEN", "X": issue.points[0][0] if issue.points else None,
            "Y": issue.points[0][1] if issue.points else None,
            "MAP_GEOM": issue.map_geometry_kind, "GEOM_NOTE": issue.map_geometry_note,
            "POINTS_JSON": json.dumps([
                [v if math.isfinite(v) else str(v) for v in point] for point in issue.points
            ], allow_nan=False)}


def write_issues(gdb: str, folder: Path, issues: List[Issue], parts: List[Part], sr: Any) -> None:
    """Write point/line review layers and an Excel-readable CSV."""
    point_fc = create_fc(gdb, "issues_points", "POINT", sr, ISSUE_FIELDS)
    line_fc = create_fc(gdb, "issues_lines", "POLYLINE", sr, ISSUE_FIELDS)
    fields = [f[0] for f in ISSUE_FIELDS]
    with arcpy.da.InsertCursor(point_fc, ["SHAPE@"] + fields) as pc:
        with arcpy.da.InsertCursor(line_fc, ["SHAPE@"] + fields) as lc:
            def records() -> Iterator[Dict[str, Any]]:
                for count, issue in enumerate(issues, 1):
                    geometry = issue_geometry(issue, sr)
                    row = issue_record(issue, parts)
                    if geometry is not None:
                        values = [row[f] for f in fields]
                        cursor = lc if issue.map_geometry_kind == "POLYLINE" else pc
                        cursor.insertRow([geometry] + values)
                    if count % 25000 == 0:
                        LOGGER.info("Writing issues: %s / %s", count, len(issues))
                    yield row

            write_csv(folder / "issues.csv", records(), fields + ["X", "Y", "POINTS_JSON"])
    display_counts = Counter(i.map_geometry_kind for i in issues)
    LOGGER.info("Issue map geometry counts: %s", dict(sorted(display_counts.items())))
    if display_counts["POINT_FALLBACK"] or display_counts["UNMAPPED"]:
        LOGGER.warning("Diagnostic display: %s point fallbacks; %s CSV-only issues. "
                       "Review MAP_GEOM, GEOM_NOTE, and POINTS_JSON in issues.csv.",
                       display_counts["POINT_FALLBACK"], display_counts["UNMAPPED"])


def stop_outputs(gdb: str, network: Network, components: Dict[int, int], issues: List[Issue],
                 stops: List[Any], sr: Any, units_per_foot: float,
                 excluded_count: int) -> Tuple[List[Dict[str, Any]], List[Issue]]:
    """Measure local mapped coverage; nearby geometry is not a verified stop entrance."""
    by_stop = {sid: (name, geometry) for sid, name, geometry in stops}
    part_geometries = [network.geometry(p.pid, sr) for p in network.parts]
    part_grid = Grid(SPATIAL_GRID_FT * units_per_foot)
    for i, geom in enumerate(part_geometries):
        part_grid.add(i, geom_box(geom))
    issue_grid = Grid(SPATIAL_GRID_FT * units_per_foot)
    for i, issue in enumerate(issues):
        geometry = issue_geometry(issue, sr)
        if geometry is not None:
            issue_grid.add(i, geom_box(geometry))
    rows, added = [], []
    nearby_stops: Set[str] = set()
    buffer_fc = os.path.join(gdb, "stop_buffers")
    with arcpy.da.SearchCursor(buffer_fc, ["STOP_ID", "RADIUS_MI", "SHAPE@"]) as cursor:
        for count, (sid, radius, polygon) in enumerate(cursor):
            if count and count % 250 == 0:
                LOGGER.info("Stop coverage: %s buffers summarized", count)
            name, stop_geometry = by_stop[sid]
            point = (stop_geometry.firstPoint.X, stop_geometry.firstPoint.Y)
            length_ft, comp_ids = 0.0, set()
            nearest_distance, nearest_pid = None, None
            # Visit likely nearest parts first. Bounding-box lower bounds let us
            # skip most expensive point-to-line distances without guessing.
            candidates = part_grid.query(geom_box(polygon))
            candidates.sort(key=lambda pid: (box_distance(point, part_grid.boxes[pid]), pid))
            for pid in candidates:
                geom = part_geometries[pid]
                if geom.disjoint(polygon):
                    continue
                # Preserve overlay semantics even for lines that retrace themselves.
                clipped_length = geom.intersect(polygon, 2).length
                if clipped_length <= 0:
                    continue
                length_ft += clipped_length / units_per_foot
                comp_ids.add(components[pid])
                lower_bound = box_distance(point, part_grid.boxes[pid]) / units_per_foot
                if nearest_distance is not None and lower_bound > nearest_distance:
                    continue
                gap = stop_geometry.distanceTo(geom) / units_per_foot
                if nearest_distance is None or gap < nearest_distance or (
                    gap == nearest_distance and pid < nearest_pid
                ):
                    nearest_distance, nearest_pid = gap, pid
            area_sq_mi = polygon.area / (units_per_foot * 5280.0) ** 2
            density = length_ft / 5280.0 / area_sq_mi if area_sq_mi else 0.0
            issue_count = 0
            for i in issue_grid.query(geom_box(polygon)):
                if not issue_geometry(issues[i], sr).disjoint(polygon):
                    issue_count += 1
            rows.append({"STOP_ID": sid, "STOP_NAME": name, "RADIUS_MI": radius,
                         "MAPPED_LENGTH_FT": length_ft, "BUFFER_SQ_MI": area_sq_mi,
                         "DENSITY_MI_PER_SQ_MI": density, "COMPONENT_COUNT": len(comp_ids),
                         "NEAREST_LINE_FT": nearest_distance,
                         "NEAREST_COMPONENT": components.get(nearest_pid),
                         "PRE_STOP_SCREEN_ISSUES": issue_count,
                         "GEOMETRY_EXCLUDED_IN_RUN": excluded_count,
                         "STOP_CONNECTION_VERIFIED": "NO"})
            if radius == max(STOP_BUFFER_MILES) and sid not in nearby_stops:
                nearby_stops.add(sid)
                if nearest_distance is None or nearest_distance > STOP_NEAR_NETWORK_FT:
                    added.append(Issue("STOP_FAR_FROM_NETWORK", [point], value_ft=nearest_distance,
                                       detail="Stop {}: no audited line within {} ft.".format(
                                           sid, STOP_NEAR_NETWORK_FT
                                       ), feature_key="GTFS:" + sid))
                if LOW_DENSITY_MI_PER_SQ_MI > 0 and density < LOW_DENSITY_MI_PER_SQ_MI:
                    added.append(Issue("LOW_LOCAL_COVERAGE", [point], confidence="SCREENING",
                                       detail=("Stop {}: {:.3f} mi/sq mi in {:.3f}-mi radius; "
                                               "review local context.").format(
                                           sid, density, radius
                                       ), feature_key="GTFS:" + sid))
    return rows, added


def write_clips(gdb: str, network_fc: str) -> None:
    """Make requested map clips AFTER the audit, never feed their ends back into QA."""
    if not WRITE_CLIPPED_NETWORK:
        return
    for radius in sorted(STOP_BUFFER_MILES):
        LOGGER.info("Writing %.3f-mile map clip using the cached dissolved scope", radius)
        dissolved = radius_scope_path(gdb, radius)
        suffix = repr(float(radius)).replace(".", "p").replace("-", "m").replace("+", "p")
        output = os.path.join(gdb, "network_clip_" + suffix)
        arcpy.analysis.Clip(network_fc, dissolved, output)


def configuration() -> Dict[str, Any]:
    """Collect user-facing settings for the run manifest."""
    return {key: sorted(value) if isinstance(value, set) else value
            for key, value in globals().items()
            if key.isupper() and isinstance(value, (str, int, float, bool, list, tuple, set))
            and key not in ("REPAIR_FIELDS", "NETWORK_FIELDS", "ISSUE_FIELDS")}


def validate_config() -> Any:
    """Validate paths, thresholds, source labels, and the analysis coordinate system."""
    if arcpy is None:
        raise RuntimeError("ArcPy is unavailable. Run using ArcGIS Pro's Python environment.")
    if not NETWORK_SOURCES:
        raise ValueError("Configure at least one NETWORK_SOURCES entry.")
    labels: Set[str] = set()
    input_filters: Set[Tuple[str, str]] = set()
    for source in NETWORK_SOURCES:
        label = source.get("LABEL", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", label) or label in labels:
            raise ValueError("Source labels must be unique letters/digits/underscores, max 80.")
        labels.add(label)
        identity = (source["PATH"].lower().replace("/", "\\"), source.get("WHERE", "").strip())
        if identity in input_filters:
            raise ValueError("The same network path/WHERE filter was configured twice.")
        input_filters.add(identity)
        if not arcpy.Exists(source["PATH"]):
            raise ValueError("Network input not found: {}. Edit CONFIGURATION.".format(
                source["PATH"]
            ))
    if not Path(GTFS_PATH).exists():
        raise ValueError("GTFS_PATH does not exist; edit CONFIGURATION.")
    if ROAD_POLYGONS and not arcpy.Exists(ROAD_POLYGONS):
        raise ValueError("ROAD_POLYGONS does not exist.")
    if APPROVED_SNAPS_CSV and not Path(APPROVED_SNAPS_CSV).is_file():
        raise ValueError("APPROVED_SNAPS_CSV does not exist.")
    if CONNECTIVITY_POLICY not in ("ANY_VERTEX", "ENDPOINT"):
        raise ValueError("CONNECTIVITY_POLICY must be ANY_VERTEX or ENDPOINT.")
    if not STOP_BUFFER_MILES or len(set(STOP_BUFFER_MILES)) != len(STOP_BUFFER_MILES):
        raise ValueError("STOP_BUFFER_MILES must contain unique positive radii.")
    positive = list(STOP_BUFFER_MILES) + [CONTEXT_MARGIN_FT, GAP_REVIEW_FT, CROSSING_SEARCH_FT,
                                         MIN_ROAD_OVERLAP_FT, SHORT_SEGMENT_FT, COARSE_SEGMENT_FT,
                                         SPATIAL_GRID_FT, MAX_APPROVED_SNAP_FT,
                                         COINCIDENCE_RESOLUTION_MULTIPLIER]
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        raise ValueError("Distances/radii/grid settings must be finite positive numbers.")
    if CONTEXT_MARGIN_FT < max(GAP_REVIEW_FT, CROSSING_SEARCH_FT):
        raise ValueError("CONTEXT_MARGIN_FT must cover both gap and crossing search distances.")
    if MAX_APPROVED_SNAP_FT > GAP_REVIEW_FT:
        raise ValueError("Repair limit cannot exceed the gap review distance.")
    if not math.isfinite(SCOPE_TILE_SIZE_FT) or SCOPE_TILE_SIZE_FT <= 0:
        raise ValueError("SCOPE_TILE_SIZE_FT must be finite and positive.")
    if not isinstance(MAX_SCOPE_TILES, int) or MAX_SCOPE_TILES < 1:
        raise ValueError("MAX_SCOPE_TILES must be a positive integer.")
    if not math.isfinite(PROGRESS_LOG_SECONDS) or PROGRESS_LOG_SECONDS < 5:
        raise ValueError("PROGRESS_LOG_SECONDS must be at least 5 seconds.")
    if not isinstance(SOURCE_PROGRESS_ROWS, int) or SOURCE_PROGRESS_ROWS < 1:
        raise ValueError("SOURCE_PROGRESS_ROWS must be a positive integer.")
    sr = arcpy.SpatialReference(ANALYSIS_WKID) if ANALYSIS_WKID else arcpy.Describe(
        NETWORK_SOURCES[0]["PATH"]
    ).spatialReference
    if sr.type != "Projected" or not sr.metersPerUnit or sr.name == "Unknown":
        raise ValueError("Use a suitable local projected ANALYSIS_WKID; degrees are not supported.")
    if sr.factoryCode in (3857, 102100, 102113) or "Web_Mercator" in sr.name:
        raise ValueError("Web Mercator distorts local distances. Choose a local projected CRS.")
    return sr


def execute_audit(folder: Path, sr: Any, timings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the audit in the chosen working folder and return its quality summary."""
    units_per_foot = 0.3048 / sr.metersPerUnit
    epsilon = max(float(sr.XYResolution) * COINCIDENCE_RESOLUTION_MULTIPLIER, 1e-9)
    LOGGER.info("CRS=%s; units=%s; diagnostic coincidence=%.9f ft",
                sr.name, sr.linearUnitName, epsilon / units_per_foot)
    gdb = str(folder / "review.gdb")
    arcpy.management.CreateFileGDB(str(folder), "review.gdb")
    study, context, stops = build_scope(gdb, sr, units_per_foot, timings)
    with timed_stage("Index local study-area polygons", timings):
        study_index = index_area(gdb, "study_area", study, sr, units_per_foot)
        context_index = index_area(gdb, "context_area", context, sr, units_per_foot)
    features, preflight = load_sources(gdb, sr, context, context_index, timings)
    with timed_stage("Summarize selected sources", timings):
        if not features:
            LOGGER.warning("No network features intersect the stop context area.")
        z_count = sum(feature_dimensions(f)[0] for f in features)
        if z_count:
            LOGGER.warning("%s Z-enabled features use XY diagnostics; review grade separation.",
                           z_count)
        write_csv(folder / "source_geometry_preflight.csv", preflight,
                  ["SOURCE_LABEL", "SOURCE_OID", "PROBLEM", "SCOPE"])
        feature_map = {f.key: f for f in features}
    with timed_stage("Load road polygon context", timings):
        roads = RoadContext(gdb, sr, context, units_per_foot, context_index)
    with timed_stage("Build existing-vertex network", timings):
        parts, initial = make_parts(features, epsilon)
        network = Network(parts, epsilon, SPATIAL_GRID_FT * units_per_foot, CONNECTIVITY_POLICY)
        before_counts = {
            "audited_parts": len(parts),
            "components": len({network.components.find(p.pid) for p in parts}),
            "dangles": len(network.dangles()),
        }
        LOGGER.info("Indexed %s parts / %s segments / %s dangles", len(parts),
                    len(network.segments), before_counts["dangles"])
    with timed_stage("Write original network and optional vertex cleanup", timings):
        original_outputs = write_network(gdb, "network_original", features, sr, original=True)
        changes = remove_repeated_vertices(features, sr)
        if changes:
            parts, initial = make_parts(features, epsilon)
            network = Network(parts, epsilon, SPATIAL_GRID_FT * units_per_foot,
                              CONNECTIVITY_POLICY)
    with timed_stage("Gap and possible-crossing review", timings):
        gap_issues = [i for i in scan_gaps(network, roads, units_per_foot)
                      if scope_issue(i, study_index, sr, feature_map)]
        for issue in gap_issues:
            issue.issue_id = identify_issue(issue, parts)
        if APPROVED_SNAPS_CSV:
            candidates = repair_candidates(gap_issues, network, feature_map, units_per_foot)
            snap_changes = apply_approved_snaps(candidates, feature_map, network, sr,
                                                units_per_foot)
            changes.extend(snap_changes)
            if snap_changes:
                parts, initial = make_parts(features, epsilon)
                network = Network(parts, epsilon, SPATIAL_GRID_FT * units_per_foot,
                                  CONNECTIVITY_POLICY)
                gap_issues = scan_gaps(network, roads, units_per_foot)
    with timed_stage("Intersection, overlap, and shape review", timings):
        issues = initial + gap_issues + scan_intersections(network, 1 / units_per_foot)
        issues.extend(scan_shape_detail(parts, 1 / units_per_foot))
    with timed_stage("Component coverage and boundary review", timings):
        part_components, component_rows, component_issues = component_outputs(
            gdb, network, study_index, context_index, sr, units_per_foot
        )
        issues.extend(component_issues)
        issues = [i for i in issues if scope_issue(i, study_index, sr, feature_map)]
        excluded = sum(bool(f.excluded_reason) for f in features)
    with timed_stage("Per-stop coverage and issue counts", timings):
        stop_rows, stop_issues = stop_outputs(gdb, network, part_components, issues, stops,
                                             sr, units_per_foot, excluded)
        issues.extend(stop_issues)
        unique: Dict[str, Issue] = {}
        for issue in issues:
            issue.issue_id = identify_issue(issue, parts)
            unique[issue.issue_id] = issue
        issues = list(unique.values())
    with timed_stage("Write working network and final geometry check", timings):
        if changes:
            working = write_network(gdb, "network_working", features, sr)
        else:
            # Native copy replaces a second Python row-by-row write of identical geometry.
            working = []
            for original in original_outputs:
                name = os.path.basename(original).replace("network_original", "network_working", 1)
                output = os.path.join(gdb, name)
                arcpy.management.CopyFeatures(original, output)
                working.append(output)
        final_check = os.path.join(gdb, "geometry_check_working")
        arcpy.management.CheckGeometry(working, final_check, "ESRI")
        final_problems = int(arcpy.management.GetCount(final_check)[0])
        if final_problems:
            LOGGER.warning("Working output has %s geometry problems; inspect final check table",
                           final_problems)
    with timed_stage("Write review layers and CSV summaries", timings):
        write_issues(gdb, folder, issues, parts, sr)
        final_candidates = repair_candidates(issues, network, feature_map, units_per_foot)
        write_csv(folder / "repair_candidates.csv", final_candidates, REPAIR_FIELDS)
        write_csv(folder / "repair_log.csv", changes,
                  ["SOURCE_KEY", "ACTION", "MOVE_FT", "OLD_JSON", "NEW_JSON"])
        write_csv(folder / "components.csv", component_rows,
                  ["COMP_ID", "PART_COUNT", "FEATURE_COUNT", "CONTEXT_LENGTH_FT",
                   "STUDY_LENGTH_FT", "CONTEXT_EDGE"])
        write_csv(folder / "stop_summary.csv", stop_rows,
                  ["STOP_ID", "STOP_NAME", "RADIUS_MI", "MAPPED_LENGTH_FT", "BUFFER_SQ_MI",
                   "DENSITY_MI_PER_SQ_MI", "COMPONENT_COUNT", "NEAREST_LINE_FT",
                   "NEAREST_COMPONENT", "PRE_STOP_SCREEN_ISSUES", "GEOMETRY_EXCLUDED_IN_RUN",
                   "STOP_CONNECTION_VERIFIED"])
        write_csv(folder / "source_attributes.csv",
                  ({"SOURCE_KEY": f.key, "SOURCE_LABEL": f.label, "SOURCE_OID": f.oid,
                    "SOURCE_ID": f.source_id,
                    "ATTRIBUTES_JSON": json.dumps(f.attrs, default=str)} for f in features),
                  ["SOURCE_KEY", "SOURCE_LABEL", "SOURCE_OID", "SOURCE_ID", "ATTRIBUTES_JSON"])
    with timed_stage("Write requested map clips", timings):
        write_clips(gdb, os.path.join(gdb, "network_components"))
    length_ft = sum(r["STUDY_LENGTH_FT"] for r in component_rows)
    counts = Counter(i.kind for i in issues)
    LOGGER.info("Issue counts: %s", dict(sorted(counts.items())))
    LOGGER.info("Repairs: %s; excluded features: %s", len(changes), excluded)
    return {
        "script_version": SCRIPT_VERSION, "status": "REVIEW_REQUIRED",
        "python": sys.version, "arcgis": arcpy.GetInstallInfo(),
        "configuration": configuration(), "analysis_crs": sr.name,
        "xy_resolution": sr.XYResolution, "diagnostic_coincidence_ft": epsilon / units_per_foot,
        "selected_features": len(features), "audited_parts": len(parts),
        "excluded_features": excluded,
        "source_geometry_preflight_problems": sum(
            r["SCOPE"] != "PROJECTED_CONTEXT" for r in preflight
        ),
        "projected_geometry_problems": sum(r["SCOPE"] == "PROJECTED_CONTEXT" for r in preflight),
        "source_geometry_check_scope": "FULL_SOURCE" if FULL_SOURCE_GEOMETRY_CHECK else (
            "SELECTED_CONTEXT"
        ),
        "working_geometry_problems": final_problems, "audited_study_length_ft": length_ft,
        "issue_counts": dict(sorted(counts.items())),
        "issue_map_geometry_counts": dict(sorted(Counter(
            i.map_geometry_kind for i in issues
        ).items())),
        "issues_per_mapped_mile": len(issues) / (length_ft / 5280) if length_ft else None,
        "repairs_applied": len(changes), "before_repairs_context": before_counts,
        "after_repairs_context": {"audited_parts": len(parts),
                                  "components": len(component_rows),
                                  "dangles": len(network.dangles())},
        "z_features_using_xy_diagnostics": z_count,
        "unknown_level_features": sum(not f.level for f in features),
        "road_polygons_in_context": len(roads.geometries),
        "scope_piece_counts": {"study": len(study_index.pieces),
                               "context": len(context_index.pieces)},
        "limitations": [
            "Review buffers are straight-line, not network service areas.",
            "Undirected XY diagnostic graph; routing/stop access is not certified.",
            "Unknown levels, legal access, barriers, slope and curb ramps need review.",
            "Lines crossing without shared eligible vertices remain disconnected.",
            "Components touching the context boundary may connect outside retained data.",
            "Excluded geometry can cause apparent gaps and low coverage.",
            "Mapped length includes duplicate/overlapping lines until resolved.",
            "Density flags need land-use/imagery context; they do not establish omission.",
            "Issue types overlap; issues_per_mapped_mile is not an accuracy percentage.",
            "Display point fallbacks preserve raw issue coordinates and values in issues.csv.",
            "Unmapped issues remain in issues.csv but cannot enter spatial per-stop counts.",
            "Selected-context geometry checks cannot locate null shapes or check outside features.",
            "Enable FULL_SOURCE_GEOMETRY_CHECK for the original whole-source preflight.",
            "Clips acquire artificial endpoints; do not use them to rerun this audit.",
        ],
    }


def publish_outputs(folder: Path, destination: Path) -> None:
    """Copy a completed local geodatabase and sidecars to the configured output folder.

    ArcPy Copy handles the geodatabase as a dataset. Do not copy individual .gdb
    files or lock files with a filesystem walk. Failed publication retains the
    complete local working folder and leaves the final run status as FAILED.
    """
    if folder == destination:
        return
    gdb = str(folder / "review.gdb")
    arcpy.management.ClearWorkspaceCache(gdb)
    arcpy.management.Copy(gdb, str(destination / "review.gdb"))
    for source in folder.iterdir():
        if source.is_file():
            shutil.copy2(str(source), str(destination / source.name))


def write_run_status(destination: Path, status: str, working_folder: Path) -> None:
    """Distinguish complete published results from running or failed partial output."""
    record = {"status": status, "working_folder": str(working_folder),
              "updated": datetime.now().isoformat(), "script_version": SCRIPT_VERSION}
    (destination / "run_status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    """Run timed stages, optionally build locally, and publish completed results."""
    sr = validate_config()
    started = time.perf_counter()
    name = datetime.now().strftime("walking_qa_%Y%m%d_%H%M%S_%f")
    destination = Path(OUTPUT_FOLDER) / name
    destination.mkdir(parents=True, exist_ok=False)
    folder = destination
    if USE_LOCAL_WORKSPACE:
        root = Path(LOCAL_WORK_ROOT) if LOCAL_WORK_ROOT else Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=name + "_", dir=str(root)))
    LOGGER.setLevel(getattr(logging, LOG_LEVEL.upper()))
    for previous in LOGGER.handlers:
        previous.close()
    LOGGER.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(destination / "run.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.info("Walking network QA %s; final output: %s", SCRIPT_VERSION, destination)
    LOGGER.info("Working folder: %s", folder)
    LOGGER.warning("Outputs are review evidence, not a certified routable network.")
    if not FULL_SOURCE_GEOMETRY_CHECK:
        LOGGER.info("Fast preflight: selected context features only; null shapes are unlocated.")
    timings: List[Dict[str, Any]] = []
    write_run_status(destination, "RUNNING", folder)
    try:
        with arcpy.EnvManager(overwriteOutput=False, outputCoordinateSystem=sr,
                              extent=None, XYTolerance=None, XYResolution=None,
                              addOutputsToMap=False):
            summary = execute_audit(folder, sr, timings)
            (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                 encoding="utf-8")
            with timed_stage("Publish completed outputs", timings):
                publish_outputs(folder, destination)
        summary["total_elapsed_seconds"] = time.perf_counter() - started
        summary["stage_timings"] = timings
        summary["output_folder"] = str(destination)
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                  encoding="utf-8")
        write_csv(destination / "timings.csv", timings, ["STAGE", "SECONDS", "STATUS"])
        write_run_status(destination, "COMPLETE", folder)
    except BaseException:
        LOGGER.exception("Run failed. Working data retained at %s; log: %s", folder, destination)
        try:
            write_csv(destination / "timings.csv", timings, ["STAGE", "SECONDS", "STATUS"])
            write_run_status(destination, "FAILED", folder)
        except OSError:
            LOGGER.exception("Could not update final status; retain the working folder above.")
        raise
    LOGGER.info("COMPLETE in %.1f minutes. Output: %s",
                (time.perf_counter() - started) / 60, destination)
    if USE_LOCAL_WORKSPACE and not KEEP_LOCAL_WORKSPACE:
        try:
            arcpy.management.ClearWorkspaceCache(str(folder / "review.gdb"))
            shutil.rmtree(str(folder))
        except (OSError, arcpy.ExecuteError):
            LOGGER.warning("Results are published; temporary files remain at %s", folder)


if __name__ == "__main__":
    main()
