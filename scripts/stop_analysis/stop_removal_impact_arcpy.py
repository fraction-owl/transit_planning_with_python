"""Analyze sidewalk-access impacts from GTFS stop removals using a spatial network (ArcPy).

This module quantifies pedestrian-access changes caused by bus stop removals by
combining GTFS stop data with a segmented sidewalk network. It measures the
change in 0.25-mile sidewalk-buffer coverage before and after stop deletion
and calculates network-based walking distances from each removed stop to its
nearest retained stop.

Key Features:
    * Builds a segmented ("physical") pedestrian graph from sidewalk or road
      centerlines, treating each single-part segment as a distinct edge.
    * Snaps stops to their nearest sidewalk segment via a uniform-grid spatial
      index and measures the snap position along that segment, so a stop can
      enter the network between two endpoints.
    * Calculates coverage area lost when stops are removed.
    * Computes both linear and network-based walking distances to the nearest
      retained stop within the search radius.
    * Exports CSV, shapefile, and QA map outputs showing removed stops and
      shortest walking paths.

This script is intended for transportation planners and GIS analysts assessing
the accessibility impacts of stop consolidation, relocation, or removal.

``stop_removal_impact_gpd.py`` is the GeoPandas twin of this script and runs
without an ArcGIS license. Both produce the same outputs from the same inputs;
this version swaps GeoPandas/Shapely for ArcPy geometry so it can run inside
ArcGIS Pro. Two deliberate differences: the shortest-path polylines here are
built vertex-ordered from the removed stop to the retained stop, and the CSV
omits the geometry column (an ArcPy geometry has no useful text form — the
paths shapefile carries the geometry).

Distances are reported in feet and miles regardless of the projected CRS's
linear units; internal measures stay in the CRS's own units and are converted
at the reporting boundary.

Outputs:
    * ``deleted_stops_distances.csv``: per-removed-stop linear and network walking
      distances (miles) to the nearest retained stop, with sanity flags.
    * ``deleted_stops.shp``: point shapefile of the removed stops.
    * ``deleted_to_nearest_paths.shp``: shortest-path polylines from each removed
      stop to its nearest retained stop (when a network path exists).
    * ``lost_coverage.shp``: sidewalk-buffer coverage lost by the removals
      (written only when non-empty).
    * ``maps/<stop_id>.png``: one QA map per removed stop, when ``EXPORT_MAPS``
      is enabled.
    * ``stop_removal_impact_runlog.txt``: run-log sidecar capturing the verbatim
      CONFIGURATION block, the resolved settings, and input fingerprints.

Typical usage:
    Update the paths in the CONFIGURATION section (or pass the matching CLI
    flags) and run from ArcGIS Pro's Python window, a Jupyter notebook, or a
    shell whose environment provides ``arcpy`` (an ArcGIS Pro install).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import arcpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Path to this script's own source. Undefined in a Jupyter kernel (no __file__),
# so fall back to None; the run log handles a missing source gracefully.
try:
    SELF_PATH: Optional[Path] = Path(__file__).resolve()
except NameError:
    SELF_PATH = None

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# INPUTS ----------------------------------------------------------------------
# Sidewalk or road centerlines used to build the walking network. This must be a
# *functional* network: segments that should connect need endpoints within
# NODE_GRID_FT of one another, or the graph will be split into islands.
SIDEWALK_FC: str = r"Path\To\Your\Sidewalks_Centerline.shp"

# Folder containing stops.txt.
GTFS_DIR: str = r"Path\To\Your\GTFS_Folder"

# Plotting-only backdrop (never used for analysis). Set to "" to skip it.
PLOT_SIDEWALKS_FC: str = r"Path\To\Your\Sidewalks_Centerline.shp"
SIDEWALK_BACKDROP_PAD_FT: float = 300.0  # how far to expand the map view when clipping

# Optional geographic (datum) transformation applied when projecting inputs,
# e.g. "WGS_1984_(ITRF00)_To_NAD_1983". Empty means ArcGIS picks a default.
GEO_TRANSFORMATION: str = ""

# OUTPUTS ---------------------------------------------------------------------
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"
DISTANCES_CSV_NAME: str = "deleted_stops_distances.csv"
DELETED_STOPS_NAME: str = "deleted_stops.shp"
PATHS_NAME: str = "deleted_to_nearest_paths.shp"
LOST_COVERAGE_NAME: str = "lost_coverage.shp"
MAPS_SUBDIR: str = "maps"
RUN_LOG_NAME: str = "stop_removal_impact_runlog.txt"

EXPORT_MAPS: bool = True  # write a small PNG map per deleted stop

# ANALYSIS PARAMETERS ---------------------------------------------------------
PROJECTED_WKID: int = 6447  # VA State Plane (US ft)
BUFFER_MILES: float = 0.25
FT_PER_MILE: float = 5280.0
MAX_SNAP_FT: float = 1500.0  # skip stops farther than this from the network

# Endpoint merging: snap endpoints to a grid to connect near-coincident nodes.
NODE_GRID_FT: float = 5.0  # merge endpoints within ~±2.5 ft

# Cell size of the uniform grid used to find each stop's nearest segment.
# Roughly one block; smaller cells mean more memory but fewer distance tests.
INDEX_CELL_FT: float = 500.0

# "Across-the-street" sanity guard
ACROSS_STREET_MAX_FT: float = 120.0  # straight-line threshold to consider "across street"
ACROSS_STREET_RATIO: float = 8.0  # network/linear ratio considered absurd
ACROSS_STREET_ABS_FT: float = 2000.0  # or absolute detour threshold

# STOP SELECTION --------------------------------------------------------------
DELETED_STOP_IDS: List[str] = ["1001", "1002"]

# Which field an entry in DELETED_STOP_IDS is matched against first.
IDENTIFIER_PRIORITY: Tuple[str, str] = ("stop_code", "stop_id")

# RUN LOG ---------------------------------------------------------------------
# When True, a failed run-log write aborts the script so the analyst is never
# left with outputs that lack a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# In a Jupyter kernel __file__ is undefined, so the run log cannot read this
# script's own source to capture the config block verbatim. Optionally point
# this at the .py on disk to restore verbatim capture. If left empty, the run
# log falls back to a snapshot of the live config values instead.
SOURCE_FILE_OVERRIDE: str = r""

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

# -----------------------------------------------------------------------------
# TYPES AND MODULE CONSTANTS
# -----------------------------------------------------------------------------

NodeKey = Tuple[float, float]  # quantized (x, y)
EdgeID = int

# Default grid step for the canonical quantize_node() copy below.
DEFAULT_NODE_GRID_FT: float = NODE_GRID_FT

# Coordinates closer than this (in CRS units) are treated as the same vertex
# when stitching path pieces together.
COORD_TOL: float = 1e-6

# A segment whose envelope would occupy more grid cells than this is held in a
# fallback list that every query scans, instead of bloating the index.
MAX_INDEX_CELLS_PER_SEGMENT: int = 4_096

# Shapefile field names are capped at 10 characters. These are the names the
# GeoPandas twin's shapefile driver produces after truncation, kept identical
# here so outputs from either script share one attribute schema.
FIELD_NEAREST_STOP: str = "nearest_st"  # nearest_stop_id
FIELD_LINEAR_DIST: str = "linear_dis"  # linear_dist_miles
FIELD_NETWORK_DIST: str = "network_di"  # network_dist_miles

# Sentinel written when the nearest retained stop is beyond the search buffer.
BEYOND_BUFFER: str = "> {:.2f}"

# Result-record keys that stay internal to the run and never reach the CSV.
INTERNAL_RESULT_FIELDS: Tuple[str, ...] = ("path_geom", "network_dist_u")

# =============================================================================
# SPATIAL REFERENCE AND UNITS
# =============================================================================


def _get_projected_sr(wkid: int) -> arcpy.SpatialReference:
    """Return the projected spatial reference for the given WKID."""
    sr = arcpy.SpatialReference(wkid)
    if sr.name == "Unknown":
        raise ValueError(f"Spatial reference WKID {wkid} is not recognized.")
    return sr


def _feet_factor(sr: arcpy.SpatialReference) -> float:
    """Return factor to convert from SR linear units to feet."""
    name = (sr.linearUnitName or "").lower()
    if "foot" in name:
        return 1.0
    return 3.28084


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================


def _is_empty_polyline(geom: Optional[arcpy.Polyline]) -> bool:
    """Return True if a Polyline is None, has no points, or has zero length."""
    if geom is None:
        return True
    if getattr(geom, "pointCount", 0) == 0:
        return True
    if getattr(geom, "length", 0.0) == 0.0:
        return True
    return False


def polyline_length(line: arcpy.Polyline) -> float:
    """Return a polyline's planar length in CRS units as a plain float."""
    return float(line.length)


# Canonical version lives in utils/network_helpers.py — keep this copy in sync.
def quantize_node(x: float, y: float, step: float = DEFAULT_NODE_GRID_FT) -> NodeKey:
    """Snap an ``(x, y)`` coordinate to a square grid of size ``step``.

    Args:
        x: X coordinate in the layer's CRS units.
        y: Y coordinate in the layer's CRS units.
        step: Grid size used to merge near-coincident endpoints into a shared
            node. Expressed in the same linear units as the coordinates.

    Returns:
        The grid-snapped ``(x, y)`` tuple, suitable as a hashable node key.
    """
    return (round(float(x) / step) * step, round(float(y) / step) * step)


def _polyline_points(geom: arcpy.Polyline) -> List[arcpy.Point]:
    """Return every non-null vertex of a polyline, in order across all parts."""
    points: List[arcpy.Point] = []
    for part in geom:
        for point in part:
            if point is not None:
                points.append(point)
    return points


def _polyline_parts_xy(geom: arcpy.Polyline) -> List[Tuple[List[float], List[float]]]:
    """Return (xs, ys) coordinate lists for each drawable part of a polyline."""
    parts: List[Tuple[List[float], List[float]]] = []
    for part in geom:
        xs: List[float] = []
        ys: List[float] = []
        for point in part:
            if point is None:
                continue
            xs.append(float(point.X))
            ys.append(float(point.Y))
        if len(xs) >= 2:
            parts.append((xs, ys))
    return parts


def _endpoints_xy(geom: arcpy.Polyline) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the first and last vertex of a polyline as (x, y) tuples."""
    first = geom.firstPoint
    last = geom.lastPoint
    return (float(first.X), float(first.Y)), (float(last.X), float(last.Y))


def _reverse_polyline(geom: arcpy.Polyline, sr: arcpy.SpatialReference) -> arcpy.Polyline:
    """Return a single-part polyline with its vertex order reversed."""
    points = _polyline_points(geom)
    points.reverse()
    return arcpy.Polyline(arcpy.Array(points), sr)


def _oriented_from(
    geom: arcpy.Polyline, node: NodeKey, sr: arcpy.SpatialReference
) -> arcpy.Polyline:
    """Return *geom* oriented so its first vertex is the end nearer to *node*."""
    (x0, y0), (x1, y1) = _endpoints_xy(geom)
    d_start = math.hypot(x0 - node[0], y0 - node[1])
    d_end = math.hypot(x1 - node[0], y1 - node[1])
    if d_end < d_start:
        return _reverse_polyline(geom, sr)
    return geom


def polyline_substring(
    line: arcpy.Polyline, start_u: float, end_u: float
) -> Optional[arcpy.Polyline]:
    """Return the portion of a polyline between two measures, or None.

    Measures are absolute distances along the line in CRS units and are clamped
    to the line's extent; passing them out of order is allowed. A zero-length
    result (both measures land on the same position) returns None, since such a
    piece contributes nothing to a stitched path.

    Args:
        line: Source polyline.
        start_u: First measure along the line, in CRS units.
        end_u: Second measure along the line, in CRS units.

    Returns:
        The polyline between the two measures, or None when the result would be
        empty or the geometry cannot be cut.
    """
    total = polyline_length(line)
    if total <= 0.0:
        return None

    s_0 = max(0.0, min(total, float(start_u)))
    s_1 = max(0.0, min(total, float(end_u)))
    if s_1 < s_0:
        s_0, s_1 = s_1, s_0
    if (s_1 - s_0) <= 0.0:
        return None

    try:
        piece = line.segmentAlongLine(s_0, s_1, use_percentage=False)
    except (arcpy.ExecuteError, RuntimeError, ValueError) as exc:
        logging.debug("segmentAlongLine failed (%s); skipping piece.", exc)
        return None

    return None if _is_empty_polyline(piece) else piece


def _concat_polylines(
    pieces: Sequence[Optional[arcpy.Polyline]], sr: arcpy.SpatialReference
) -> Optional[arcpy.Polyline]:
    """Stitch ordered polyline pieces into one polyline, dropping repeated vertices."""
    points: List[arcpy.Point] = []
    for piece in pieces:
        if piece is None:
            continue
        for point in _polyline_points(piece):
            if points:
                prev = points[-1]
                if abs(point.X - prev.X) <= COORD_TOL and abs(point.Y - prev.Y) <= COORD_TOL:
                    continue
            points.append(point)

    if len(points) < 2:
        return None
    return arcpy.Polyline(arcpy.Array(points), sr)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_centerlines(
    path: str, sr: arcpy.SpatialReference, transformation: str = ""
) -> List[arcpy.Polyline]:
    """Read centerline geometries, projected into the target spatial reference.

    Args:
        path: Polyline feature class or shapefile of sidewalk/road centerlines.
        sr: Target projected spatial reference.
        transformation: Optional geographic (datum) transformation name.

    Returns:
        The layer's polyline geometries in *sr*.

    Raises:
        OSError: The layer does not exist.
        ValueError: The layer has no spatial reference, is not a polyline
            layer, or holds no usable geometry.
    """
    if not arcpy.Exists(path):
        raise OSError(f"Centerline layer not found: {path}")

    desc = arcpy.Describe(path)
    if getattr(desc, "shapeType", "") != "Polyline":
        raise ValueError(
            f"Centerline layer must be a polyline layer; '{path}' is "
            f"'{getattr(desc, 'shapeType', 'unknown')}'."
        )
    source_sr = getattr(desc, "spatialReference", None)
    if source_sr is None or source_sr.name in ("", "Unknown"):
        raise ValueError(
            f"Centerline layer '{path}' has no spatial reference; define one before running."
        )

    if transformation:
        arcpy.env.geographicTransformations = transformation

    lines: List[arcpy.Polyline] = []
    with arcpy.da.SearchCursor(path, ["SHAPE@"], spatial_reference=sr) as cursor:
        for (geom,) in cursor:
            if geom is not None:
                lines.append(geom)

    if not lines:
        raise ValueError(f"Centerline layer '{path}' contains no geometry.")

    logging.info("Read %d centerline features.", len(lines))
    return lines


def explode_segments(
    lines: Iterable[arcpy.Polyline], sr: arcpy.SpatialReference
) -> List[arcpy.Polyline]:
    """Split multipart centerlines into single-part segments.

    Each returned segment is one graph edge; its position in the list is its
    stable ``edge_id``.

    Args:
        lines: Polyline geometries to explode.
        sr: Spatial reference assigned to the rebuilt segments.

    Returns:
        Single-part polylines, in input order.
    """
    segments: List[arcpy.Polyline] = []
    for geom in lines:
        for part in geom:
            points = [point for point in part if point is not None]
            if len(points) < 2:
                continue
            segment = arcpy.Polyline(arcpy.Array(points), sr)
            if _is_empty_polyline(segment):
                continue
            segments.append(segment)

    logging.info("Exploded to %d single-part segments.", len(segments))
    return segments


def build_graph(
    segments: Sequence[arcpy.Polyline], node_grid_u: float
) -> Tuple[nx.MultiGraph, Dict[EdgeID, Tuple[NodeKey, NodeKey]]]:
    """Build an undirected MultiGraph from exploded segments.

    Args:
        segments: Single-part polylines; the list index becomes the edge_id.
        node_grid_u: Endpoint-merging grid size in CRS units.

    Returns:
        (graph, edge_endpoints) where edge_endpoints maps each edge_id to its
        quantized (u, v) node keys. Edge attributes are ``edge_id``,
        ``length`` (CRS units), and ``segment`` (the polyline).
    """
    graph = nx.MultiGraph()
    edge_endpoints: Dict[EdgeID, Tuple[NodeKey, NodeKey]] = {}

    for edge_id, geom in enumerate(segments):
        (x_1, y_1), (x_2, y_2) = _endpoints_xy(geom)
        node_u = quantize_node(x_1, y_1, node_grid_u)
        node_v = quantize_node(x_2, y_2, node_grid_u)

        for node in (node_u, node_v):
            if node not in graph:
                graph.add_node(node, x=node[0], y=node[1])

        graph.add_edge(
            node_u,
            node_v,
            edge_id=int(edge_id),
            segment=geom,
            length=polyline_length(geom),
        )
        edge_endpoints[int(edge_id)] = (node_u, node_v)

    logging.info("Graph: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges())
    return graph, edge_endpoints


def load_gtfs_stops(
    gtfs_dir: str, sr: arcpy.SpatialReference, transformation: str = ""
) -> pd.DataFrame:
    """Load GTFS stops.txt as a DataFrame of projected point geometries.

    Args:
        gtfs_dir: Folder containing stops.txt.
        sr: Target projected spatial reference.
        transformation: Optional geographic (datum) transformation name.

    Returns:
        DataFrame with stop_id, stop_name, stop_code, geometry (PointGeometry
        in *sr*), and the projected x / y coordinates.

    Raises:
        OSError: stops.txt is missing or unreadable.
        ValueError: stops.txt lacks required columns or has no usable rows.
    """
    stops_csv = Path(gtfs_dir) / "stops.txt"
    if not stops_csv.is_file():
        raise OSError(f"stops.txt not found in GTFS folder: {gtfs_dir}")

    try:
        raw = pd.read_csv(stops_csv, dtype=str)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise OSError(f"Could not read '{stops_csv}': {exc}") from exc

    missing = {"stop_id", "stop_lat", "stop_lon"} - set(raw.columns)
    if missing:
        raise ValueError(f"stops.txt missing columns: {', '.join(sorted(missing))}")

    raw = raw.drop_duplicates("stop_id")
    wgs84 = arcpy.SpatialReference(4326)

    records: List[Dict[str, Any]] = []
    skipped = 0
    for row in raw.itertuples(index=False):
        stop_id = str(row.stop_id)
        try:
            lon = float(row.stop_lon)
            lat = float(row.stop_lat)
        except (TypeError, ValueError):
            logging.debug("Skipping stop %s: invalid coordinates.", stop_id)
            skipped += 1
            continue

        point = arcpy.PointGeometry(arcpy.Point(lon, lat), wgs84)
        projected = point.projectAs(sr, transformation) if transformation else point.projectAs(sr)
        records.append(
            {
                "stop_id": stop_id,
                "stop_name": str(getattr(row, "stop_name", "") or ""),
                "stop_code": str(getattr(row, "stop_code", "") or ""),
                "geometry": projected,
                "x": float(projected.firstPoint.X),
                "y": float(projected.firstPoint.Y),
            }
        )

    if skipped:
        logging.warning("Skipped %d stop(s) with invalid coordinates.", skipped)
    if not records:
        raise ValueError(f"No usable stops found in '{stops_csv}'.")

    return pd.DataFrame(records)


def resolve_deleted_stop_ids(
    stops: pd.DataFrame,
    identifiers: Sequence[str],
    prefer_stop_code: bool = True,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Resolve human-entered identifiers to canonical GTFS stop_ids.

    Args:
        stops: Stops table carrying stop_id and (optionally) stop_code.
        identifiers: Values typed by the analyst, matched against either field.
        prefer_stop_code: Try stop_code before stop_id when both could match.

    Returns:
        (resolved_ids, match_map) where resolved_ids lists every matched
        stop_id and match_map records what each input identifier matched.
    """
    sid_series = stops["stop_id"].astype(str)
    if "stop_code" in stops.columns:
        sc_series = stops["stop_code"].astype(str)
    else:
        sc_series = pd.Series([""] * len(stops), index=stops.index, dtype=str)

    id_by_stop_id: Dict[str, List[str]] = (
        sid_series.to_frame(name="stop_id").groupby("stop_id")["stop_id"].apply(list).to_dict()
    )
    id_by_stop_code: Dict[str, List[str]] = (
        pd.DataFrame({"stop_code": sc_series, "stop_id": sid_series})
        .groupby("stop_code")["stop_id"]
        .apply(list)
        .to_dict()
    )

    first, second = ("stop_code", "stop_id") if prefer_stop_code else ("stop_id", "stop_code")
    lookups = {"stop_id": id_by_stop_id, "stop_code": id_by_stop_code}

    resolved: List[str] = []
    match_map: Dict[str, List[str]] = {}

    for raw in identifiers:
        key = str(raw)
        matched: List[str] = []
        for field in (first, second):
            hits = lookups[field].get(key, [])
            if hits:
                matched = hits
                break
        match_map[key] = matched
        resolved.extend(matched)

    n_in = len(identifiers)
    n_res = len(set(resolved))
    n_only_sid = sum(
        1 for k, v in match_map.items() if v and (k in id_by_stop_id) and (k not in id_by_stop_code)
    )
    n_only_sc = sum(
        1 for k, v in match_map.items() if v and (k in id_by_stop_code) and (k not in id_by_stop_id)
    )
    n_both = sum(
        1 for k, v in match_map.items() if v and (k in id_by_stop_code) and (k in id_by_stop_id)
    )
    n_none = sum(1 for v in match_map.values() if not v)

    logging.info(
        "Deleted list resolved: %d identifiers → %d unique stop_ids "
        "(only stop_code: %d, only stop_id: %d, both: %d, unmatched: %d)",
        n_in,
        n_res,
        n_only_sc,
        n_only_sid,
        n_both,
        n_none,
    )
    if n_none:
        lost = [k for k, v in match_map.items() if not v]
        logging.warning(
            "Unmatched identifiers (neither stop_id nor stop_code): %s",
            ", ".join(lost[:20]) + ("…" if len(lost) > 20 else ""),
        )

    return resolved, match_map


# =============================================================================
# SPATIAL INDEX
# =============================================================================


class SegmentIndex:
    """Uniform-grid index over segment envelopes for nearest-segment queries.

    ArcPy has no public R-tree, so segments are bucketed into square cells by
    their envelopes. A query walks outward ring by ring and stops once no
    unvisited cell could hold anything closer than the best match so far.
    """

    def __init__(self, segments: Sequence[arcpy.Polyline], cell_u: float) -> None:
        """Build the grid index.

        Args:
            segments: Single-part polylines; list position is the edge_id.
            cell_u: Grid cell size in CRS units.

        Raises:
            ValueError: cell_u is not positive.
        """
        if cell_u <= 0.0:
            raise ValueError("Index cell size must be positive.")

        self.segments = segments
        self.cell_u = float(cell_u)
        self._cells: Dict[Tuple[int, int], List[EdgeID]] = {}
        self._oversized: List[EdgeID] = []

        for edge_id, geom in enumerate(segments):
            extent = geom.extent
            col_min, row_min = self._cell_of(extent.XMin, extent.YMin)
            col_max, row_max = self._cell_of(extent.XMax, extent.YMax)
            n_cells = (col_max - col_min + 1) * (row_max - row_min + 1)

            if n_cells > MAX_INDEX_CELLS_PER_SEGMENT:
                self._oversized.append(int(edge_id))
                continue

            for col in range(col_min, col_max + 1):
                for row in range(row_min, row_max + 1):
                    self._cells.setdefault((col, row), []).append(int(edge_id))

        if self._oversized:
            logging.debug(
                "%d oversized segment(s) held outside the grid index.", len(self._oversized)
            )

    def _cell_of(self, x: float, y: float) -> Tuple[int, int]:
        """Return the (column, row) grid cell holding a coordinate."""
        return (int(math.floor(x / self.cell_u)), int(math.floor(y / self.cell_u)))

    def _ring_cells(self, col: int, row: int, ring: int) -> Iterable[Tuple[int, int]]:
        """Yield the cells whose Chebyshev distance from (col, row) equals *ring*."""
        if ring == 0:
            yield (col, row)
            return
        for offset in range(-ring, ring + 1):
            yield (col + offset, row - ring)
            yield (col + offset, row + ring)
        for offset in range(-ring + 1, ring):
            yield (col - ring, row + offset)
            yield (col + ring, row + offset)

    def nearest(
        self, point: arcpy.PointGeometry, max_dist_u: float
    ) -> Optional[Tuple[EdgeID, float]]:
        """Return the nearest segment within a distance cap.

        Args:
            point: Query point, in the index's spatial reference.
            max_dist_u: Farthest distance to consider, in CRS units.

        Returns:
            (edge_id, distance in CRS units), or None when nothing qualifies.
        """
        col_0, row_0 = self._cell_of(float(point.firstPoint.X), float(point.firstPoint.Y))

        best_eid: Optional[EdgeID] = None
        best_dist = math.inf
        seen: set[EdgeID] = set()

        for edge_id in self._oversized:
            seen.add(edge_id)
            dist = float(self.segments[edge_id].distanceTo(point))
            if dist < best_dist:
                best_eid, best_dist = edge_id, dist

        max_ring = int(math.ceil(max_dist_u / self.cell_u)) + 2
        for ring in range(max_ring + 1):
            for cell in self._ring_cells(col_0, row_0, ring):
                for edge_id in self._cells.get(cell, ()):
                    if edge_id in seen:
                        continue
                    seen.add(edge_id)
                    dist = float(self.segments[edge_id].distanceTo(point))
                    if dist < best_dist:
                        best_eid, best_dist = edge_id, dist

            # Nothing outside the rings scanned so far can beat this bound: a
            # segment reaching within (ring - 1) cells of the query point is
            # already indexed in a cell that has been visited.
            covered_u = max(0.0, (ring - 1) * self.cell_u)
            if best_eid is not None and best_dist <= covered_u:
                break
            if covered_u > max_dist_u:
                break

        if best_eid is None or best_dist > max_dist_u:
            return None
        return best_eid, best_dist


# =============================================================================
# SNAP LOGIC
# =============================================================================


def snap_stops_to_segments(
    stops: pd.DataFrame,
    index: SegmentIndex,
    segments: Sequence[arcpy.Polyline],
    edge_endpoints: Dict[EdgeID, Tuple[NodeKey, NodeKey]],
    max_snap_u: float,
) -> pd.DataFrame:
    """Snap each stop to its nearest segment and measure offsets to the endpoints.

    Args:
        stops: Stops table with stop_id and projected point geometry.
        index: Grid index over *segments*.
        segments: Single-part polylines making up the network.
        edge_endpoints: edge_id → (u, v) node keys.
        max_snap_u: Farthest snap distance to accept, in CRS units.

    Returns:
        DataFrame with stop_id, stop_name, stop_code, x, y, edge_id, s_u (snap
        measure along the segment), len_u, u_node, v_node, to_u_u, and to_v_u.
        Measure columns are CRS units; off-network stops carry NaN/None.
    """
    rows: List[Dict[str, Any]] = []
    off_network = 0

    for row in stops.itertuples(index=False):
        base = {
            "stop_id": str(row.stop_id),
            "stop_name": str(row.stop_name),
            "stop_code": str(row.stop_code),
            "x": float(row.x),
            "y": float(row.y),
        }
        unsnapped = dict(
            base,
            edge_id=np.nan,
            s_u=np.nan,
            len_u=np.nan,
            u_node=None,
            v_node=None,
            to_u_u=np.nan,
            to_v_u=np.nan,
        )

        hit = index.nearest(row.geometry, max_snap_u)
        if hit is None:
            rows.append(unsnapped)
            off_network += 1
            continue

        edge_id, _dist_u = hit
        segment = segments[edge_id]
        measure = float(segment.measureOnLine(row.geometry, use_percentage=False))
        if not math.isfinite(measure):
            rows.append(unsnapped)
            off_network += 1
            continue

        length_u = polyline_length(segment)
        node_u, node_v = edge_endpoints[int(edge_id)]
        rows.append(
            dict(
                base,
                edge_id=int(edge_id),
                s_u=measure,
                len_u=length_u,
                u_node=node_u,
                v_node=node_v,
                to_u_u=measure,
                to_v_u=length_u - measure,
            )
        )

    if off_network:
        logging.info("%d stop(s) are farther than the snap limit from the network.", off_network)
    return pd.DataFrame(rows)


# =============================================================================
# SHORTEST PATHS
# =============================================================================

# Small symmetric cache keyed by unordered node pair.
_node_dist_cache: Dict[Tuple[NodeKey, NodeKey], float] = {}


def _node_dist_u(graph: nx.MultiGraph, node_a: NodeKey, node_b: NodeKey) -> float:
    """Return the cached node-to-node shortest path length in CRS units."""
    if node_a == node_b:
        return 0.0
    key = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
    cached = _node_dist_cache.get(key)
    if cached is not None:
        return cached
    dist = float(nx.shortest_path_length(graph, node_a, node_b, weight="length"))
    _node_dist_cache[key] = dist
    return dist


def _min_edge_data(graph: nx.MultiGraph, node_u: NodeKey, node_v: NodeKey) -> Dict[str, Any]:
    """Return edge data for the shortest parallel edge between two nodes."""
    data = graph.get_edge_data(node_u, node_v)
    if not data:
        raise nx.NetworkXNoPath(f"No edge between {node_u} and {node_v}")
    key_min = min(data, key=lambda k: data[k]["length"])
    return data[key_min]


def _build_path_geometry(
    graph: nx.MultiGraph,
    a_seg: arcpy.Polyline,
    a_s_u: float,
    a_to_u: bool,
    node_path: Sequence[NodeKey],
    b_seg: arcpy.Polyline,
    b_s_u: float,
    b_from_u: bool,
    sr: arcpy.SpatialReference,
) -> Optional[arcpy.Polyline]:
    """Stitch the full walking path: partial A segment, node path, partial B segment.

    Every piece is oriented so the returned polyline runs from stop A's snap
    position to stop B's snap position.

    Args:
        graph: The pedestrian graph.
        a_seg: Segment stop A snapped to.
        a_s_u: Stop A's measure along a_seg, in CRS units.
        a_to_u: True when the path leaves a_seg through its u node.
        node_path: Node sequence between the two segments.
        b_seg: Segment stop B snapped to.
        b_s_u: Stop B's measure along b_seg, in CRS units.
        b_from_u: True when the path enters b_seg through its u node.
        sr: Spatial reference for the assembled geometry.

    Returns:
        The stitched polyline, or None when no piece produced geometry.
    """
    len_a = polyline_length(a_seg)
    len_b = polyline_length(b_seg)

    if a_to_u:
        a_part = polyline_substring(a_seg, 0.0, a_s_u)
        if a_part is not None:
            a_part = _reverse_polyline(a_part, sr)  # run from the stop toward node u
    else:
        a_part = polyline_substring(a_seg, a_s_u, len_a)

    middles: List[arcpy.Polyline] = []
    for node_u, node_v in zip(node_path[:-1], node_path[1:]):
        edge = _min_edge_data(graph, node_u, node_v)
        middles.append(_oriented_from(edge["segment"], node_u, sr))

    if b_from_u:
        b_part = polyline_substring(b_seg, 0.0, b_s_u)
    else:
        b_part = polyline_substring(b_seg, b_s_u, len_b)
        if b_part is not None:
            b_part = _reverse_polyline(b_part, sr)  # run from node v toward the stop

    return _concat_polylines([a_part, *middles, b_part], sr)


def stop_to_stop_network(
    graph: nx.MultiGraph,
    segments: Sequence[arcpy.Polyline],
    edge_endpoints: Dict[EdgeID, Tuple[NodeKey, NodeKey]],
    snap_map: pd.DataFrame,
    sid_a: str,
    sid_b: str,
    sr: arcpy.SpatialReference,
) -> Tuple[float, Optional[arcpy.Polyline]]:
    """Compute the network distance and path geometry between two snapped stops.

    Args:
        graph: The pedestrian graph.
        segments: Single-part polylines making up the network.
        edge_endpoints: edge_id → (u, v) node keys.
        snap_map: Output of :func:`snap_stops_to_segments`.
        sid_a: Origin stop_id.
        sid_b: Destination stop_id.
        sr: Spatial reference for the assembled geometry.

    Returns:
        (distance in CRS units, path geometry). The distance is ``inf`` and the
        geometry None when either stop is off-network or no path exists.
    """
    rec_a = snap_map.loc[snap_map.stop_id == sid_a]
    rec_b = snap_map.loc[snap_map.stop_id == sid_b]
    if rec_a.empty or rec_b.empty:
        return math.inf, None

    row_a = rec_a.iloc[0]
    row_b = rec_b.iloc[0]
    if pd.isna(row_a.edge_id) or pd.isna(row_b.edge_id):
        return math.inf, None

    eid_a = int(row_a.edge_id)
    eid_b = int(row_b.edge_id)
    u_a, v_a = edge_endpoints[eid_a]
    u_b, v_b = edge_endpoints[eid_b]
    seg_a = segments[eid_a]
    seg_b = segments[eid_b]

    candidates: List[Tuple[float, Tuple[NodeKey, NodeKey], Tuple[bool, bool]]] = []
    for a_end, a_cost, a_is_u in (
        (u_a, float(row_a.to_u_u), True),
        (v_a, float(row_a.to_v_u), False),
    ):
        for b_end, b_cost, b_is_u in (
            (u_b, float(row_b.to_u_u), True),
            (v_b, float(row_b.to_v_u), False),
        ):
            try:
                middle = _node_dist_u(graph, a_end, b_end)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            candidates.append((a_cost + middle + b_cost, (a_end, b_end), (a_is_u, b_is_u)))

    if not candidates:
        return math.inf, None

    total_u, (a_end, b_end), (a_is_u, b_is_u) = min(candidates, key=lambda item: item[0])

    try:
        node_path: List[NodeKey] = nx.shortest_path(graph, a_end, b_end, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf, None

    path_geom = _build_path_geometry(
        graph,
        a_seg=seg_a,
        a_s_u=float(row_a.s_u),
        a_to_u=a_is_u,
        node_path=node_path,
        b_seg=seg_b,
        b_s_u=float(row_b.s_u),
        b_from_u=b_is_u,
        sr=sr,
    )
    return total_u, path_geom


# =============================================================================
# COVERAGE
# =============================================================================


def _scratch_point_fc(
    points: Sequence[arcpy.PointGeometry], sr: arcpy.SpatialReference, name: str
) -> str:
    """Write point geometries to an in-memory feature class and return its path."""
    result = arcpy.management.CreateFeatureclass("memory", name, "POINT", spatial_reference=sr)
    fc_path = str(result[0])
    with arcpy.da.InsertCursor(fc_path, ["SHAPE@"]) as cursor:
        for point in points:
            cursor.insertRow([point])
    return fc_path


def coverage_polygon(
    points: Sequence[arcpy.PointGeometry],
    buffer_miles: float,
    sr: arcpy.SpatialReference,
    name: str,
) -> Optional[arcpy.Polygon]:
    """Return one dissolved buffer polygon around the given stops.

    Args:
        points: Projected stop locations.
        buffer_miles: Buffer radius in miles; ArcGIS converts it to CRS units.
        sr: Spatial reference of the stops.
        name: Unique base name for the scratch feature classes.

    Returns:
        The dissolved coverage polygon, or None when there are no stops.
    """
    if not points:
        return None

    src_fc = _scratch_point_fc(points, sr, f"{name}_pts")
    buf_fc = f"memory\\{name}_buf"
    try:
        arcpy.analysis.Buffer(
            src_fc,
            buf_fc,
            f"{buffer_miles} Miles",
            dissolve_option="ALL",
        )
        merged: Optional[arcpy.Polygon] = None
        with arcpy.da.SearchCursor(buf_fc, ["SHAPE@"]) as cursor:
            for (geom,) in cursor:
                if geom is None:
                    continue
                merged = geom if merged is None else merged.union(geom)
        return merged
    finally:
        for scratch in (src_fc, buf_fc):
            if arcpy.Exists(scratch):
                arcpy.management.Delete(scratch)


# =============================================================================
# EXPORTS
# =============================================================================


def _create_shapefile(
    out_dir: Path,
    filename: str,
    geometry_type: str,
    sr: arcpy.SpatialReference,
    fields: Sequence[Tuple[str, str, Optional[int]]],
) -> str:
    """Create an empty shapefile with the given fields and return its path."""
    name = Path(filename).stem
    fc_path = os.path.join(str(out_dir), f"{name}.shp")
    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(str(out_dir), name, geometry_type, spatial_reference=sr)
    for field_name, field_type, length in fields:
        arcpy.management.AddField(fc_path, field_name, field_type, field_length=length)
    return fc_path


def _as_text(value: Any) -> str:
    """Render a result value for a shapefile text field ("" for missing)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def export_results(
    results: Dict[str, Dict[str, Any]],
    sr: arcpy.SpatialReference,
    out_dir: Path,
    csv_name: str,
    stops_name: str,
    paths_name: str,
) -> None:
    """Write the distances CSV, the removed-stop points, and the path polylines.

    Args:
        results: Per-removed-stop result records.
        sr: Spatial reference of the outputs.
        out_dir: Destination folder.
        csv_name: Filename for the distances CSV.
        stops_name: Filename for the removed-stop point shapefile.
        paths_name: Filename for the shortest-path polyline shapefile.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(results.values())

    # Geometry has no useful text form (the paths shapefile carries it), and the
    # raw CRS-unit distance is bookkeeping for the map titles.
    frame = pd.DataFrame(
        [{k: v for k, v in rec.items() if k not in INTERNAL_RESULT_FIELDS} for rec in records]
    )
    csv_path = out_dir / csv_name
    frame.to_csv(csv_path, index=False)
    logging.info("Wrote %s (%d rows).", csv_path, len(frame))

    point_fields = [
        ("stop_name", "TEXT", 128),
        ("stop_id", "TEXT", 64),
        ("stop_code", "TEXT", 64),
        (FIELD_LINEAR_DIST, "TEXT", 24),
        (FIELD_NETWORK_DIST, "TEXT", 24),
    ]
    pts_path = _create_shapefile(out_dir, stops_name, "POINT", sr, point_fields)
    insert_fields = [name for name, _type, _len in point_fields] + ["SHAPE@"]
    with arcpy.da.InsertCursor(pts_path, insert_fields) as cursor:
        for rec in records:
            geom = arcpy.PointGeometry(arcpy.Point(rec["x"], rec["y"]), sr)
            cursor.insertRow(
                [
                    _as_text(rec["stop_name"]),
                    _as_text(rec["stop_id"]),
                    _as_text(rec["stop_code"]),
                    _as_text(rec["linear_dist_miles"]),
                    _as_text(rec["network_dist_miles"]),
                    geom,
                ]
            )
    logging.info("Wrote %s (%d features).", pts_path, len(records))

    with_paths = [rec for rec in records if rec.get("path_geom") is not None]
    if not with_paths:
        logging.info("No shortest paths to export; skipping %s.", paths_name)
        return

    line_fields = [
        ("stop_name", "TEXT", 128),
        ("stop_id", "TEXT", 64),
        ("stop_code", "TEXT", 64),
        (FIELD_NEAREST_STOP, "TEXT", 64),
        (FIELD_LINEAR_DIST, "TEXT", 24),
        (FIELD_NETWORK_DIST, "TEXT", 24),
    ]
    lines_path = _create_shapefile(out_dir, paths_name, "POLYLINE", sr, line_fields)
    insert_fields = [name for name, _type, _len in line_fields] + ["SHAPE@"]
    with arcpy.da.InsertCursor(lines_path, insert_fields) as cursor:
        for rec in with_paths:
            cursor.insertRow(
                [
                    _as_text(rec["stop_name"]),
                    _as_text(rec["stop_id"]),
                    _as_text(rec["stop_code"]),
                    _as_text(rec["nearest_stop_id"]),
                    _as_text(rec["linear_dist_miles"]),
                    _as_text(rec["network_dist_miles"]),
                    rec["path_geom"],
                ]
            )
    logging.info("Wrote %s (%d features).", lines_path, len(with_paths))


def export_lost_coverage(
    lost: Optional[arcpy.Polygon],
    sr: arcpy.SpatialReference,
    out_dir: Path,
    filename: str,
) -> None:
    """Write the lost-coverage polygon, skipping empty results."""
    if lost is None or lost.area == 0.0:
        logging.info("No lost coverage to export; skipping %s.", filename)
        return

    fc_path = _create_shapefile(out_dir, filename, "POLYGON", sr, [("area_sqmi", "DOUBLE", None)])
    with arcpy.da.InsertCursor(fc_path, ["area_sqmi", "SHAPE@"]) as cursor:
        cursor.insertRow([float(lost.getArea("PLANAR", "SQUAREMILES")), lost])
    logging.info("Wrote %s.", fc_path)


# =============================================================================
# QA MAPS
# =============================================================================


def _load_backdrop_for_plots(
    path: str, sr: arcpy.SpatialReference, transformation: str = ""
) -> Optional[List[arcpy.Polyline]]:
    """Load a linework backdrop for plotting only, or None when unusable."""
    if not path or path.startswith(r"Path\To\Your"):
        return None
    try:
        return load_centerlines(path, sr, transformation)
    except (OSError, ValueError, arcpy.ExecuteError) as exc:
        logging.warning("Backdrop load failed (%s); continuing without it.", exc)
        return None


def _plot_lines_within(
    ax: Any,
    lines: Optional[Sequence[arcpy.Polyline]],
    envelope: arcpy.Extent,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    """Plot the portion of each line that falls inside an envelope."""
    if not lines:
        return
    for geom in lines:
        extent = geom.extent
        if (
            extent.XMax < envelope.XMin
            or extent.XMin > envelope.XMax
            or extent.YMax < envelope.YMin
            or extent.YMin > envelope.YMax
        ):
            continue
        try:
            clipped = geom.clip(envelope)
        except (arcpy.ExecuteError, RuntimeError, ValueError):
            clipped = geom
        if _is_empty_polyline(clipped):
            continue
        for xs, ys in _polyline_parts_xy(clipped):
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _safe_filename(value: str) -> str:
    """Return a filename-safe version of a stop identifier."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(value))
    return cleaned or "stop"


def export_stop_maps(
    stops: pd.DataFrame,
    segments: Sequence[arcpy.Polyline],
    results: Dict[str, Dict[str, Any]],
    sr: arcpy.SpatialReference,
    out_dir: Path,
    backdrop: Optional[Sequence[arcpy.Polyline]],
    pad_u: float,
    ft_per_unit: float,
) -> None:
    """Save one PNG QA map per removed stop that resolved to a network path.

    Each map draws the analysis network (and the optional backdrop) clipped to
    the path's padded extent, the shortest path, and the removed and retained
    stops labelled with names, IDs, and the network distance in feet.

    Args:
        stops: Stops table with geometry, used for the two plotted points.
        segments: Network segments used to build the graph.
        results: Per-removed-stop result records.
        sr: Spatial reference of the data.
        out_dir: Parent folder; maps land in its ``maps`` subfolder.
        backdrop: Optional plot-only linework.
        pad_u: Map padding around the path extent, in CRS units.
        ft_per_unit: Feet per CRS linear unit.
    """
    map_dir = out_dir / MAPS_SUBDIR
    map_dir.mkdir(parents=True, exist_ok=True)

    geom_by_id = dict(zip(stops.stop_id.astype(str), stops.geometry))
    name_by_id = dict(zip(stops.stop_id.astype(str), stops.stop_name.astype(str)))
    written = 0

    for sid, rec in results.items():
        path_geom = rec.get("path_geom")
        target_sid = rec.get("nearest_stop_id")
        if path_geom is None or target_sid is None:
            continue

        removed_pt = geom_by_id.get(str(sid))
        kept_pt = geom_by_id.get(str(target_sid))
        if removed_pt is None or kept_pt is None:
            continue

        extent = path_geom.extent
        envelope = arcpy.Extent(
            extent.XMin - pad_u,
            extent.YMin - pad_u,
            extent.XMax + pad_u,
            extent.YMax + pad_u,
            spatial_reference=sr,
        )

        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
        _plot_lines_within(ax, backdrop, envelope, "0.75", 0.6, 0.6, 0)
        _plot_lines_within(ax, segments, envelope, "0.45", 0.8, 0.8, 1)

        for xs, ys in _polyline_parts_xy(path_geom):
            ax.plot(xs, ys, color="tab:blue", linewidth=2.0, zorder=2)

        removed_name = name_by_id.get(str(sid), "")
        kept_name = name_by_id.get(str(target_sid), "")
        for point, color, label in (
            (removed_pt, "red", f"{removed_name} ({sid})"),
            (kept_pt, "green", f"{kept_name} ({target_sid})"),
        ):
            ax.scatter([point.firstPoint.X], [point.firstPoint.Y], c=color, s=35, zorder=3)
            ax.annotate(
                label,
                xy=(point.firstPoint.X, point.firstPoint.Y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                zorder=4,
            )

        net_u = rec.get("network_dist_u")
        if isinstance(net_u, float) and math.isfinite(net_u):
            dist_txt = f"{net_u * ft_per_unit:,.0f} ft"
        else:
            dist_txt = _as_text(rec.get("network_dist_miles")) or "N/A"
            if dist_txt.startswith(">"):
                dist_txt = f"> {float(dist_txt.lstrip('> ')) * FT_PER_MILE:,.0f} ft"
        ax.set_title(
            f"Deleted {removed_name} ({sid}) → {kept_name} ({target_sid}) [Network: {dist_txt}]",
            fontsize=8,
        )

        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(
            map_dir / f"{_safe_filename(sid)}.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
        written += 1

    logging.info("Wrote %d QA map(s) to %s.", written, map_dir)


# =============================================================================
# RUN LOG
# =============================================================================


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


def _resolve_source_path() -> Optional[Path]:
    """Return a readable path to this script's source, or None.

    Prefers SOURCE_FILE_OVERRIDE (for notebook runs), then SELF_PATH (defined
    when running as a .py file). Returns None when neither is available.
    """
    override = SOURCE_FILE_OVERRIDE.strip()
    if override:
        path = Path(override)
        if path.is_file():
            return path
        logging.warning("SOURCE_FILE_OVERRIDE set but not found: %s", path)
    return SELF_PATH


def _live_config_snapshot() -> str:
    """Build a best-effort config record from live module globals.

    Used when the source file is unavailable (e.g. running in a Jupyter kernel),
    so the run log still carries a record of the configuration actually used.
    Captures UPPER_SNAKE_CASE globals holding simple scalar/sequence values.
    """
    module_globals = globals()
    lines: List[str] = []
    for name in sorted(module_globals):
        if name.startswith("_") or name != name.upper():
            continue
        value = module_globals[name]
        if isinstance(value, (str, int, float, bool, list, tuple, type(None))):
            lines.append(f"{name} = {value!r}")
    return "\n".join(lines)


def _sha256_of_file(path: str) -> Optional[str]:
    """Return the SHA-256 digest of a file, or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def input_fingerprints(sidewalk_fc: str, gtfs_dir: str) -> List[str]:
    """Return SHA-256 fingerprint lines for the primary inputs, for provenance.

    Args:
        sidewalk_fc: The centerline layer actually used for this run.
        gtfs_dir: The GTFS folder actually used for this run.

    Returns:
        Formatted "label: path" / "sha256: digest" line pairs for the run log.
    """
    targets: List[Tuple[str, str]] = [("Centerlines", sidewalk_fc)]
    dbf = os.path.splitext(sidewalk_fc)[0] + ".dbf"
    if os.path.isfile(dbf):
        targets.append(("Centerlines .dbf", dbf))
    targets.append(("GTFS stops.txt", os.path.join(gtfs_dir, "stops.txt")))

    lines: List[str] = []
    for label, path in targets:
        digest = _sha256_of_file(path)
        lines.append(f"{label}: {path}")
        lines.append(f"  sha256: {digest if digest else 'unreadable'}")
    return lines


def write_run_log(
    output_dir: str,
    log_name: str,
    effective_settings: Sequence[str],
    fingerprints: Sequence[str],
) -> bool:
    """Write a run log of the configuration into *output_dir*.

    When the script's source is readable, the config block is captured
    verbatim. Otherwise (e.g. a Jupyter kernel with no __file__) it falls back
    to a live snapshot of the config globals so a run still produces a
    configuration record. The effective-settings section records the values
    actually used, which may come from CLI flags rather than the constants.

    Args:
        output_dir: Directory the run log is written into.
        log_name: Filename for the run log sidecar.
        effective_settings: Pre-formatted lines describing the resolved
            settings for this run.
        fingerprints: Pre-formatted input fingerprint lines (from
            :func:`input_fingerprints`).

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = Path(output_dir) / log_name

    source_path = _resolve_source_path()
    if source_path is not None:
        try:
            config_text = extract_config_block(source_path)
            config_heading = "CONFIGURATION (verbatim from source)"
            source_label = str(source_path)
        except (OSError, ValueError) as exc:
            logging.warning(
                "Could not extract config block from '%s' (%s); falling back to a live snapshot.",
                source_path,
                exc,
            )
            config_text = _live_config_snapshot()
            config_heading = "CONFIGURATION (live snapshot — source unreadable)"
            source_label = f"{source_path} (unreadable)"
    else:
        config_text = _live_config_snapshot()
        config_heading = "CONFIGURATION (live snapshot — source file unavailable)"
        source_label = "unavailable (running without __file__, e.g. Jupyter)"

    lines: List[str] = [
        "=" * 72,
        "STOP REMOVAL IMPACT RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output folder:    {output_dir}",
        f"Source script:    {source_label}",
        "",
        "-" * 72,
        "EFFECTIVE SETTINGS (constants or CLI flags, as resolved for this run)",
        "-" * 72,
        *effective_settings,
        "",
        "-" * 72,
        "INPUT FINGERPRINTS",
        "-" * 72,
        *fingerprints,
        "",
        "-" * 72,
        config_heading,
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# PIPELINE
# =============================================================================


def _result_record(
    row: Any,
    nearest_stop_id: Optional[str],
    linear_dist_miles: Any,
    network_dist_miles: Any,
    path_geom: Optional[arcpy.Polyline],
    sanity_flag: Optional[str],
    network_dist_u: float = math.inf,
) -> Dict[str, Any]:
    """Assemble one per-removed-stop result record."""
    return {
        "stop_name": str(row.stop_name),
        "stop_id": str(row.stop_id),
        "stop_code": str(row.stop_code),
        "x": float(row.x),
        "y": float(row.y),
        "nearest_stop_id": nearest_stop_id,
        "linear_dist_miles": linear_dist_miles,
        "network_dist_miles": network_dist_miles,
        "sanity_flag": sanity_flag,
        "path_geom": path_geom,
        "network_dist_u": network_dist_u,
    }


def _effective_settings(args: argparse.Namespace, sr: arcpy.SpatialReference) -> List[str]:
    """Return run-log lines describing the settings resolved for this run."""
    return [
        f"Centerlines:          {args.sidewalk_fc}",
        f"GTFS folder:          {args.gtfs_dir}",
        f"Backdrop (plot only): {args.plot_sidewalks_fc or '(none)'}",
        f"Output folder:        {args.output_dir}",
        f"Projected WKID:       {args.projected_wkid} ({sr.name}, {sr.linearUnitName})",
        f"Geo transformation:   {args.geo_transformation or '(default)'}",
        f"Buffer (miles):       {args.buffer_miles}",
        f"Max snap (ft):        {args.max_snap_ft}",
        f"Node grid (ft):       {args.node_grid_ft}",
        f"Index cell (ft):      {args.index_cell_ft}",
        f"Across-street guard:  max {args.across_street_max_ft} ft, "
        f"ratio {args.across_street_ratio}, abs {args.across_street_abs_ft} ft",
        f"Identifier priority:  {args.identifier_priority}",
        f"Deleted identifiers:  {', '.join(str(i) for i in args.deleted_stop_ids) or '(none)'}",
        f"Export maps:          {args.export_maps}",
    ]


def _nearest_kept_stop(
    args: argparse.Namespace,
    row: Any,
    kept: pd.DataFrame,
    kd_tree: cKDTree,
    graph: nx.MultiGraph,
    segments: Sequence[arcpy.Polyline],
    edge_endpoints: Dict[EdgeID, Tuple[NodeKey, NodeKey]],
    snap_map: pd.DataFrame,
    sr: arcpy.SpatialReference,
    units_per_ft: float,
    ft_per_unit: float,
) -> Dict[str, Any]:
    """Find the nearest retained stop to one removed stop and build its record.

    Candidates come from a Euclidean prefilter within the coverage buffer; each
    is scored by network distance. A candidate that is a short straight-line hop
    away but an absurd detour by network is treated as across-the-street and
    scored linearly instead, flagged as ``across_street_override``.

    Args:
        args: Resolved settings.
        row: The removed stop's row.
        kept: Retained stops.
        kd_tree: KD-tree over the retained stops' coordinates.
        graph: The pedestrian graph.
        segments: Network segments.
        edge_endpoints: edge_id → (u, v) node keys.
        snap_map: Output of :func:`snap_stops_to_segments`.
        sr: Spatial reference of the data.
        units_per_ft: CRS units per foot.
        ft_per_unit: Feet per CRS linear unit.

    Returns:
        The result record for this removed stop.
    """
    sid = str(row.stop_id)
    beyond = BEYOND_BUFFER.format(args.buffer_miles)

    snapped = snap_map.loc[snap_map.stop_id == sid]
    off_network = snapped.empty or pd.isna(snapped.iloc[0].edge_id)

    buffer_u = args.buffer_miles * FT_PER_MILE * units_per_ft
    neighbors = kd_tree.query_ball_point(x=[row.x, row.y], r=buffer_u, p=2.0)

    if off_network or not neighbors:
        return _result_record(
            row,
            nearest_stop_id=None,
            linear_dist_miles=beyond,
            network_dist_miles=beyond,
            path_geom=None,
            sanity_flag="off_network" if off_network else "no_kept_within_buffer",
        )

    best_net_u = math.inf
    best_lin_u: Optional[float] = None
    best_sid: Optional[str] = None
    best_geom: Optional[arcpy.Polyline] = None
    best_flag: Optional[str] = None

    for position in neighbors:
        target = kept.iloc[position]
        target_sid = str(target.stop_id)

        linear_u = float(row.geometry.distanceTo(target.geometry))
        network_u, path_geom = stop_to_stop_network(
            graph, segments, edge_endpoints, snap_map, sid, target_sid, sr
        )

        flag: Optional[str] = None
        linear_ft = linear_u * ft_per_unit
        network_ft = network_u * ft_per_unit
        if linear_ft <= args.across_street_max_ft and (
            math.isinf(network_u)
            or network_ft > max(args.across_street_abs_ft, args.across_street_ratio * linear_ft)
        ):
            path_geom = arcpy.Polyline(
                arcpy.Array(
                    [
                        arcpy.Point(row.x, row.y),
                        arcpy.Point(float(target.x), float(target.y)),
                    ]
                ),
                sr,
            )
            network_u = linear_u
            flag = "across_street_override"

        if network_u < best_net_u:
            best_net_u = network_u
            best_lin_u = linear_u
            best_sid = target_sid
            best_geom = path_geom
            best_flag = flag

    if math.isinf(best_net_u):
        return _result_record(
            row,
            nearest_stop_id=None,
            linear_dist_miles=(
                round(best_lin_u * ft_per_unit / FT_PER_MILE, 4) if best_lin_u is not None else None
            ),
            network_dist_miles=beyond,
            path_geom=None,
            sanity_flag=best_flag,
        )

    return _result_record(
        row,
        nearest_stop_id=best_sid,
        linear_dist_miles=round(float(best_lin_u) * ft_per_unit / FT_PER_MILE, 4),
        network_dist_miles=round(best_net_u * ft_per_unit / FT_PER_MILE, 4),
        path_geom=best_geom,
        sanity_flag=best_flag,
        network_dist_u=best_net_u,
    )


def run_analysis(args: argparse.Namespace) -> None:
    """Run the full stop-removal impact analysis with resolved settings.

    Args:
        args: Parsed CLI arguments (defaults mirror the CONFIGURATION block).

    Raises:
        OSError: An input path is missing or unreadable.
        ValueError: Invalid settings or inputs that cannot support the analysis.
        RuntimeError: The run log could not be written and REQUIRE_RUN_LOG is
            True.
        arcpy.ExecuteError: A geoprocessing step failed.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = _get_projected_sr(int(args.projected_wkid))
    ft_per_unit = _feet_factor(sr)
    units_per_ft = 1.0 / ft_per_unit
    logging.info("Using SR: %s (1 unit ≈ %.4f ft)", sr.name, ft_per_unit)

    if args.geo_transformation:
        arcpy.env.geographicTransformations = args.geo_transformation

    logging.info("Reading centerlines …")
    centerlines = load_centerlines(args.sidewalk_fc, sr, args.geo_transformation)

    logging.info("Exploding to segments …")
    segments = explode_segments(centerlines, sr)
    if not segments:
        raise ValueError(
            f"No usable line segments in '{args.sidewalk_fc}'; the network cannot be built."
        )

    logging.info("Building graph …")
    _node_dist_cache.clear()
    graph, edge_endpoints = build_graph(segments, args.node_grid_ft * units_per_ft)

    logging.info("Building spatial index …")
    index = SegmentIndex(segments, args.index_cell_ft * units_per_ft)

    logging.info("Loading GTFS stops …")
    stops = load_gtfs_stops(args.gtfs_dir, sr, args.geo_transformation)
    logging.info("Total unique stops: %d", len(stops))

    logging.info("Snapping stops to the network …")
    snap_map = snap_stops_to_segments(
        stops, index, segments, edge_endpoints, args.max_snap_ft * units_per_ft
    )

    logging.info("Resolving deleted identifiers …")
    resolved_ids, _match_map = resolve_deleted_stop_ids(
        stops,
        args.deleted_stop_ids,
        prefer_stop_code=args.identifier_priority == "stop_code",
    )
    if not resolved_ids:
        raise ValueError(
            "None of the deleted identifiers matched a GTFS stop_id or stop_code; "
            "check DELETED_STOP_IDS / --deleted-stop-ids."
        )
    stops["is_deleted"] = stops["stop_id"].isin(resolved_ids)
    kept = stops.loc[~stops["is_deleted"]].reset_index(drop=True)

    logging.info("Calculating coverage polygons …")
    original_cov = coverage_polygon(list(stops.geometry), args.buffer_miles, sr, "cov_all")
    kept_cov = coverage_polygon(list(kept.geometry), args.buffer_miles, sr, "cov_kept")
    if original_cov is None:
        lost_cov = None
    elif kept_cov is None:
        lost_cov = original_cov
    else:
        lost_cov = original_cov.difference(kept_cov)
    lost_area_sqmi = (
        float(lost_cov.getArea("PLANAR", "SQUAREMILES")) if lost_cov is not None else 0.0
    )
    logging.info("Area lost: %.4f sq mi", lost_area_sqmi)

    logging.info("Computing network distances …")
    if kept.empty:
        logging.warning("No kept stops; every removed stop is reported beyond the buffer.")
    kept_coords = (
        np.array([(float(x), float(y)) for x, y in zip(kept.x, kept.y)])
        if not kept.empty
        else np.empty((0, 2))
    )
    kd_tree = cKDTree(kept_coords) if kept_coords.size else None

    results: Dict[str, Dict[str, Any]] = {}
    unique_deleted = sorted(set(resolved_ids))
    for sid in unique_deleted:
        match = stops.loc[stops.stop_id == sid]
        if match.empty:
            logging.warning("Deleted stop_id %s not found in GTFS.", sid)
            continue
        row = match.iloc[0]

        if kd_tree is None:
            beyond = BEYOND_BUFFER.format(args.buffer_miles)
            results[sid] = _result_record(
                row,
                nearest_stop_id=None,
                linear_dist_miles=beyond,
                network_dist_miles=beyond,
                path_geom=None,
                sanity_flag=None,
            )
            continue

        results[sid] = _nearest_kept_stop(
            args,
            row,
            kept,
            kd_tree,
            graph,
            segments,
            edge_endpoints,
            snap_map,
            sr,
            units_per_ft,
            ft_per_unit,
        )

    built = sum(1 for rec in results.values() if rec["path_geom"] is not None)
    logging.info("Paths built for %d/%d deleted stops", built, len(unique_deleted))

    logging.info("Exporting CSV and shapefiles …")
    export_results(
        results,
        sr,
        out_dir,
        args.distances_csv_name,
        args.deleted_stops_name,
        args.paths_name,
    )
    export_lost_coverage(lost_cov, sr, out_dir, args.lost_coverage_name)

    if args.export_maps:
        logging.info("Exporting QA maps …")
        backdrop = _load_backdrop_for_plots(args.plot_sidewalks_fc, sr, args.geo_transformation)
        export_stop_maps(
            stops,
            segments,
            results,
            sr,
            out_dir,
            backdrop,
            args.backdrop_pad_ft * units_per_ft,
            ft_per_unit,
        )

    wrote_log = write_run_log(
        args.output_dir,
        args.run_log_name,
        _effective_settings(args, sr),
        input_fingerprints(args.sidewalk_fc, args.gtfs_dir),
    )
    if not wrote_log and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written and REQUIRE_RUN_LOG is True. Outputs were "
            "written but are untraced; fix the output folder permissions and re-run."
        )

    logging.info("All done! Outputs in: %s", out_dir)


# =============================================================================
# CLI / MAIN
# =============================================================================


# Canonical version lives in utils/cli_helpers.py — keep this copy in sync.
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser (defaults mirror CONFIGURATION)."""
    parser = argparse.ArgumentParser(
        description="Analyze sidewalk-access impacts of GTFS stop removals (ArcPy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sidewalk-fc",
        default=SIDEWALK_FC,
        help="Sidewalk/road centerline layer used to build the walking network.",
    )
    parser.add_argument(
        "--gtfs-dir",
        default=GTFS_DIR,
        help="GTFS folder containing stops.txt.",
    )
    parser.add_argument(
        "--plot-sidewalks-fc",
        default=PLOT_SIDEWALKS_FC,
        help="Plot-only linework backdrop for the QA maps (empty to skip).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for the CSV, shapefiles, maps, and run log.",
    )
    parser.add_argument(
        "--distances-csv-name",
        default=DISTANCES_CSV_NAME,
        help="Filename for the per-stop distances CSV.",
    )
    parser.add_argument(
        "--deleted-stops-name",
        default=DELETED_STOPS_NAME,
        help="Filename for the removed-stop point shapefile.",
    )
    parser.add_argument(
        "--paths-name",
        default=PATHS_NAME,
        help="Filename for the shortest-path polyline shapefile.",
    )
    parser.add_argument(
        "--lost-coverage-name",
        default=LOST_COVERAGE_NAME,
        help="Filename for the lost-coverage polygon shapefile.",
    )
    parser.add_argument(
        "--run-log-name",
        default=RUN_LOG_NAME,
        help="Filename for the run-log sidecar.",
    )
    parser.add_argument(
        "--deleted-stop-ids",
        nargs="*",
        default=DELETED_STOP_IDS,
        help="Stop identifiers to remove, matched against stop_code or stop_id.",
    )
    parser.add_argument(
        "--identifier-priority",
        default=IDENTIFIER_PRIORITY[0],
        choices=("stop_code", "stop_id"),
        help="Field each deleted identifier is matched against first.",
    )
    parser.add_argument(
        "--projected-wkid",
        type=int,
        default=PROJECTED_WKID,
        help="Projected CRS WKID used for all measurements.",
    )
    parser.add_argument(
        "--geo-transformation",
        default=GEO_TRANSFORMATION,
        help="Optional geographic (datum) transformation for projecting inputs.",
    )
    parser.add_argument(
        "--buffer-miles",
        type=float,
        default=BUFFER_MILES,
        help="Coverage buffer radius and nearest-stop search radius, in miles.",
    )
    parser.add_argument(
        "--max-snap-ft",
        type=float,
        default=MAX_SNAP_FT,
        help="Stops farther than this from the network are treated as off-network.",
    )
    parser.add_argument(
        "--node-grid-ft",
        type=float,
        default=NODE_GRID_FT,
        help="Endpoint-merging grid size used to connect near-coincident nodes.",
    )
    parser.add_argument(
        "--index-cell-ft",
        type=float,
        default=INDEX_CELL_FT,
        help="Cell size of the grid index used for nearest-segment lookups.",
    )
    parser.add_argument(
        "--across-street-max-ft",
        type=float,
        default=ACROSS_STREET_MAX_FT,
        help="Straight-line distance under which a pair counts as across-the-street.",
    )
    parser.add_argument(
        "--across-street-ratio",
        type=float,
        default=ACROSS_STREET_RATIO,
        help="Network/linear ratio treated as an absurd detour.",
    )
    parser.add_argument(
        "--across-street-abs-ft",
        type=float,
        default=ACROSS_STREET_ABS_FT,
        help="Absolute detour distance treated as absurd.",
    )
    parser.add_argument(
        "--backdrop-pad-ft",
        type=float,
        default=SIDEWALK_BACKDROP_PAD_FT,
        help="How far to expand the QA map view around each path.",
    )
    parser.add_argument(
        "--export-maps",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_MAPS,
        help="Write one PNG QA map per removed stop.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the analysis, validating placeholder paths before doing any work.

    Args:
        argv: Optional explicit argument list; None reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 on runtime failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    still_placeholder = [
        name
        for name, value in (
            ("SIDEWALK_FC / --sidewalk-fc", args.sidewalk_fc),
            ("GTFS_DIR / --gtfs-dir", args.gtfs_dir),
            ("OUTPUT_DIR / --output-dir", args.output_dir),
        )
        if str(value).startswith(r"Path\To\Your")
    ]
    if still_placeholder:
        logging.warning(
            "Placeholder value(s) still set for: %s. Update the CONFIGURATION "
            "section or pass the matching CLI flags before running.",
            "; ".join(still_placeholder),
        )
        return 2

    arcpy.env.overwriteOutput = True

    try:
        run_analysis(args)
    except (OSError, ValueError, RuntimeError, arcpy.ExecuteError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
