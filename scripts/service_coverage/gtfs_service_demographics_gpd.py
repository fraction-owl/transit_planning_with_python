"""Performs spatial analysis on GTFS transit data and demographic shapefiles.

Generates a transit *service area* and estimates the population, household, and
employment characteristics that fall within it. The script offers two
independent choices:

* **Analysis mode** — how results are grouped: ``"network"`` (one combined
  surface), ``"route"`` (one surface per route), or ``"stop"`` (one surface per
  stop).
* **Service-area method** — how each catchment polygon is built:
    - ``"stop_buffer"``: a fixed-radius buffer around each transit stop, with an
      optional wider drive-access buffer for express origin (park-and-ride) stops.
    - ``"route_buffer"``: a fixed-radius buffer around the route-line geometry
      taken from GTFS ``shapes.txt``.
    - ``"isochrone"``: a walk-time isochrone (walkshed) around each stop,
      traced over a pedestrian centerline network.

In ``"route"`` mode each route is labeled by ``service_type``
(``"express"``/``"local"``) and ``express_direction``
(``"unidirectional"``/``"bidirectional"``/``""``). Express routes get *access-mode
accounting* under the ``"stop_buffer"`` method: population is drive-scaled at the
park-and-ride origin stops (wide buffer) and walk-scaled elsewhere, while
employment is always walk-scaled (riders walk from the alighting stop to the
job). Unidirectional (commuter) routes additionally drop the two cross-terms —
origin employment and destination population — which can be zeroed
independently; bidirectional routes keep both ends. See the express settings in
the CONFIGURATION block.

Intended for use in Jupyter notebooks with appropriate EPSG settings. The
projected CRS (``CRS_EPSG_CODE``) is assumed to use **metres** as its linear
unit, matching the miles-to-metres conversions used throughout.

Typical inputs:
    - GTFS folder containing: trips.txt, stop_times.txt, routes.txt,
      stops.txt, calendar.txt (and shapes.txt for the ``route_buffer`` method).
    - Demographic shapefile with fields to estimate.
    - A pedestrian centerline shapefile (only for the ``isochrone`` method).
    - Configurable filter lists, buffer, and isochrone settings in the script.

Outputs:
    - Shapefiles (.shp) and Excel summaries (.xlsx) for each analysis unit.
    - Optional matplotlib plots for visual inspection.
    - Optional ``express_origin_candidates.csv`` advisory when express routes are
      set: stops ranked by nearby jobs, flagging the job-poor ones as likely
      express origins for review (it never changes any buffer).
"""

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

# Select analysis mode: "network", "route", or "stop"
ANALYSIS_MODE = "network"  # Options: "network", "route", "stop"

# Select how the service-area polygon is built around the transit it serves:
#   "stop_buffer"  → fixed-radius buffer around each stop (uses BUFFER_DISTANCE,
#                    plus the wider express-origin buffer for the stops listed
#                    in the express-origin settings below).
#   "route_buffer" → fixed-radius buffer around the route-line geometry from
#                    shapes.txt (uses BUFFER_DISTANCE). Falls back to
#                    "stop_buffer" when route geometry is unavailable.
#   "isochrone"    → walk-time isochrone around each stop traced over the
#                    pedestrian network (uses the ISOCHRONE_* settings below).
SERVICE_AREA_METHOD = "stop_buffer"  # Options: "stop_buffer", "route_buffer", "isochrone"

# Paths
GTFS_DATA_PATH = r"Path\To\GTFS_data_folder"
DEMOGRAPHICS_SHP_PATH = r"Path\To\census_blocks.shp"
OUTPUT_DIRECTORY = r"Path\To\Output"

# Pedestrian centerline (sidewalk/road) shapefile — only used when
# SERVICE_AREA_METHOD == "isochrone". Lines should be routable (split at
# intersections) for best results.
PEDESTRIAN_NETWORK_PATH = r"Path\To\centerlines.shp"

# Calendar / service-pattern filter. Leave empty to auto-select the full Monday–Friday
# service(s) straight from calendar.txt (recommended — robust to feed-specific service_id
# values); set explicit ids (e.g. ["2"]) to force a particular service pattern.
SERVICE_IDS_TO_INCLUDE: Final[list[str]] = []

# Route filters:
# 1) ROUTES_TO_INCLUDE: If non-empty, only these routes are considered.
# 2) ROUTES_TO_EXCLUDE: If non-empty, these routes are removed.
# If both are empty, all routes in routes.txt are used.
ROUTES_TO_INCLUDE: list[str] = []  # empty = all routes; set e.g. ["101", "202"] for a subset
ROUTES_TO_EXCLUDE: list[str] = []  # e.g. [] for no exclude filter

# Express routes:
# Express/commuter routes whose service areas are unreliable under the standard
# walk-buffer methods (a long highway corridor served by only a few stops). In
# "route" mode each output row gets a ``service_type`` column of "express" or
# "local" (so the downstream model can fit them separately) plus an
# ``express_direction`` column of "unidirectional", "bidirectional", or "".
#
# Express routes split into two service patterns. Each is matched against GTFS
# ``route_id`` and is the union of an inline list and an optional text file (one
# route_id per line, "#" starts a comment; set the file to None for inline only):
#
#   * UNIDIRECTIONAL (commuter) express — a directional home->work flow (AM
#     inbound, PM outbound). Its ridership market is population at the origin
#     (residential / park-and-ride end) plus employment at the destination
#     (job / CBD end), so in "route" mode with the "stop_buffer" method these get
#     *directional accounting*: population is summed from the origin catchment and
#     employment from the destination catchment, and the two cross-terms can be
#     zeroed (see the directional-accounting knobs below). Origin stops must be
#     listed in the express-origin settings so the two ends can be told apart;
#     a route with no origin stop falls back to symmetric accounting (warned).
EXPRESS_UNIDIRECTIONAL_ROUTE_IDS: list[str] = []
EXPRESS_UNIDIRECTIONAL_ROUTES_FILE: str | None = None
#
#   * BIDIRECTIONAL express — connects two mixed job/residential centers and
#     carries load both ways, so both ends contribute. It keeps both ends'
#     population and employment (the cross-term knobs are unidirectional-only),
#     but still uses access-mode buffering: population is drive-scaled at the
#     park-and-ride origins (wide buffer) and employment is walk-scaled at every
#     stop. (Counting jobs over a wide drive buffer would imply driving miles
#     from the bus to the desk — riders walk that last leg, so jobs stay walk.)
EXPRESS_BIDIRECTIONAL_ROUTE_IDS: list[str] = []
EXPRESS_BIDIRECTIONAL_ROUTES_FILE: str | None = None
#
# EXPRESS_ROUTE_IDS / EXPRESS_ROUTES_FILE is the original, direction-agnostic
# list. Routes listed here (and not in either list above) are still labeled
# "express" and get the wide origin buffer, but keep symmetric accounting — i.e.
# they behave like bidirectional express. Kept for backward compatibility.
EXPRESS_ROUTE_IDS: list[str] = []
EXPRESS_ROUTES_FILE: str | None = None

# Stop filters:
# 1) STOP_IDS_TO_INCLUDE: If non-empty, only these stops are considered (after route filter).
# 2) STOP_IDS_TO_EXCLUDE: If non-empty, these stops are removed (after route filter).
# If both are empty, all stops belonging to final routes are used.
STOP_IDS_TO_INCLUDE: list[
    str
] = []  # e.g. [] for no include filter or [1005, 1007] for include filter
STOP_IDS_TO_EXCLUDE: list[
    str
] = []  # e.g. [] for no include filter or [1010, 1011] for exclude filter

# Buffer distances in miles (used by the "stop_buffer" and "route_buffer" methods)
BUFFER_DISTANCE = 0.25  # Standard walk-access buffer (≈ a quarter-mile walkshed)
EXPRESS_ORIGIN_BUFFER_DISTANCE = 2.0  # Drive-access buffer for express origin stops

# Express origin stops:
# stop_ids that are *origin* stops on express/commuter routes — park-and-rides
# and similar boarding points people reach by car, not on foot. For an express
# route they mark the drive-access end: population near them is captured with the
# wide EXPRESS_ORIGIN_BUFFER_DISTANCE drive-shed (instead of the near-empty
# quarter-mile around a highway lot), while every other stop uses the walk
# BUFFER_DISTANCE for population. Employment is always counted on the walk buffer
# (riders walk from the bus to the job), at every stop including the origins.
# Listing a route's origins is also what lets the unidirectional split tell its
# two ends apart. (Applies to the "stop_buffer" method.) The set is the union of
# two sources, matched against GTFS ``stop_id``:
#   1) EXPRESS_ORIGIN_STOP_IDS: stop_id values listed inline.
#   2) EXPRESS_ORIGIN_STOPS_FILE: path to a text file with one stop_id per line
#      ("#" starts a comment). Set to None to use the inline list only.
# Tip: set the express route lists first; those routes' stops are the only
# candidates, and the advisory can suggest which of them are origins.
EXPRESS_ORIGIN_STOP_IDS: list[str] = []
EXPRESS_ORIGIN_STOPS_FILE: str | None = None

# Express origin advisory:
# When EXPRESS_ROUTE_IDS / file is set and the demographics layer carries a jobs
# field, score each express-route stop by the jobs within a walk buffer and flag
# the job-poor ones as *likely* origins (park-and-rides sit far from employment;
# downtown destinations do not). Results are written to
# express_origin_candidates.csv with a warning. Advisory only — it never changes
# any buffer; review the flagged stop_ids and copy the real origins into
# EXPRESS_ORIGIN_STOP_IDS yourself.
EXPRESS_ORIGIN_ADVISORY: bool = True
EXPRESS_ORIGIN_JOBS_FIELD: str = "tot_empl"  # demographics column holding job counts
# A stop is flagged when its nearby jobs fall below this fraction of the
# job-richest stop on the same route (a relative, per-route cutoff).
EXPRESS_ORIGIN_JOBS_SHARE_THRESHOLD: float = 0.2

# Directional accounting (UNIDIRECTIONAL express routes only):
# A commuter express is a one-way flow — riders board near home (the origin /
# park-and-ride end) and alight near work (the destination / CBD end) — so its
# market is population@origin + employment@destination. Those two "keystone"
# terms are always kept. The two *cross-terms* are mostly irrelevant (only the
# rare reverse / night-shift rider uses them) and can be zeroed independently:
#   * EXPRESS_ZERO_ORIGIN_EMPLOYMENT — drop the employment counted in the origin
#     stops' walk buffer (employment is always walk-scaled, never drive-scaled).
#     Park-and-rides sit far from jobs by construction, so this term is usually
#     negligible; leave True.
#   * EXPRESS_ZERO_DESTINATION_POPULATION — drop population counted in the
#     destination (CBD) catchment. Set False where the destination has real
#     housing inside the walk buffer (e.g. mixed-use cores like Tysons or
#     Bethesda) and you want those residents kept; True suits job-only downtowns
#     (e.g. a CBD or office park with little housing within a quarter mile).
# Applies only in "route" mode with the "stop_buffer" method; other methods fall
# back to symmetric accounting with a warning.
EXPRESS_ZERO_ORIGIN_EMPLOYMENT: bool = True
EXPRESS_ZERO_DESTINATION_POPULATION: bool = True

# Which SYNTHETIC_FIELDS (below) are employment (job) counts. Employment is
# walk-scaled at every stop; every other synthetic field is treated as
# residence-based (population / households / residents), drive-scaled at the
# origin stops and walk-scaled elsewhere.
EXPRESS_EMPLOYMENT_FIELDS: list[str] = ["tot_empl", "low_wage", "mid_wage", "high_wage"]

# Isochrone settings (only used when SERVICE_AREA_METHOD == "isochrone")
ISOCHRONE_WALK_TIME_MIN = 10.0  # Walk-time budget in minutes
WALK_SPEED_MPH = 3.0  # Assumed pedestrian walking speed

# Optional FIPS filter (list of codes). Empty list = no filter.
FIPS_FILTER: list[str] = []  # Replace with FIPS code(s) for desired jurisdictions (e.g. "11001")

# Fields in the demographics shapefile to multiply by the area ratio. Each must be an
# additive block-level COUNT (never a percentage). The first group is block-native
# (decennial population/households, LEHD jobs); the second is tract-level estimates that
# uscensus_tiger_join_gpd disaggregates down to blocks (see TRACT_COUNT_DISAGG there), so
# they are area-weightable too. A field absent from the layer is reported once and
# skipped, so this list can stay ambitious even when an input table was not supplied.
SYNTHETIC_FIELDS = [
    "total_pop",
    "total_hh",
    "tot_empl",
    "low_wage",
    "mid_wage",
    "high_wage",
    "low_income",  # households under the low-income bands
    "minority",  # non-white-alone residents
    "lep",  # limited-English-proficiency residents
    "lo_veh_hh",  # households with 0-1 vehicles
    "youth",  # residents age 15-21
    "elderly",  # residents age 65+
    # Commuting (ACS S0801) worker counts, disaggregated to blocks by
    # uscensus_tiger_join_gpd. All additive, so they area-weight cleanly. A catchment
    # transit share is cmt_trnst / cmt_wrkrs; mean travel time is cmt_pmin / cmt_wrkrs.
    "cmt_wrkrs",  # workers 16+
    "cmt_trnst",  # commute by public transit
    "cmt_drove",  # drove alone
    "cmt_carpl",  # carpooled
    "cmt_wfh",  # worked from home
    "cmt_pmin",  # worker-minutes (mean travel time = cmt_pmin / cmt_wrkrs)
]

# EPSG code for projected coordinate system used in area calculations
CRS_EPSG_CODE = 3395  # Replace with EPSG for your study area

# GTFS files always required
REQUIRED_GTFS_FILES = [
    "trips.txt",
    "stop_times.txt",
    "routes.txt",
    "stops.txt",
    "calendar.txt",
]

# Additional GTFS file required only for the "route_buffer" service-area method.
# Loaded opportunistically; if absent the method falls back to "stop_buffer".
ROUTE_GEOMETRY_GTFS_FILE = "shapes.txt"

# Conversion factor: metres per mile (the projected CRS is assumed metric).
METERS_PER_MILE: Final[float] = 1609.34

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def filter_weekday_service(calendar_df: pd.DataFrame) -> pd.Series:
    """Return service_ids that run every weekday (Monday through Friday).

    calendar.txt is frequently loaded with every column as a string, so the day flags
    are coerced to numeric before the ``== 1`` comparison; a service must run on all
    five weekdays to qualify.

    :param calendar_df: DataFrame from calendar.txt.
    :return: Series of service_id values available on all weekdays.
    """
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    flags = calendar_df[days].apply(pd.to_numeric, errors="coerce").fillna(0)
    weekday_filter = (flags == 1).all(axis=1)
    return calendar_df.loc[weekday_filter, "service_id"]


def get_included_stops(
    stops_df: pd.DataFrame,
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
) -> pd.DataFrame:
    """Determine which stops to keep by applying inclusion/exclusion lists.

    Args:
        stops_df: DataFrame from stops.txt (or an already merged subset).
        stop_ids_to_include: Stop IDs to include. If non-empty, only these remain.
        stop_ids_to_exclude: Stop IDs to exclude. If non-empty, these are removed.

    Returns:
        DataFrame containing only the final included stops.
    """
    filtered = stops_df.copy()

    filtered["stop_id"] = filtered["stop_id"].astype(str)
    include = [str(s) for s in stop_ids_to_include]
    exclude = [str(s) for s in stop_ids_to_exclude]

    if include:
        filtered = filtered[filtered["stop_id"].isin(include)]

    if exclude:
        filtered = filtered[~filtered["stop_id"].isin(exclude)]

    logging.info(
        "Including %d stops after applying stop include/exclude lists.",
        len(filtered),
    )
    return filtered


def get_included_routes(
    routes_df: pd.DataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
) -> pd.DataFrame:
    """Filter routes by route_short_name include/exclude lists."""
    filtered = routes_df.copy()

    if "route_short_name" not in filtered.columns:
        raise KeyError("routes_df is missing required column: 'route_short_name'")

    filtered["route_short_name"] = filtered["route_short_name"].astype(str)
    include = [str(r) for r in routes_to_include]
    exclude = [str(r) for r in routes_to_exclude]

    if include:
        filtered = filtered[filtered["route_short_name"].isin(include)]

    if exclude:
        filtered = filtered[~filtered["route_short_name"].isin(exclude)]

    logging.info("Including %d routes after route include/exclude lists.", len(filtered))
    return filtered


def pick_buffer_distance(
    stop_id: str, normal_buffer: float, large_buffer: float, large_buffer_ids: list[str]
) -> float:
    """Determine the buffer distance for a given stop_id.

    :param stop_id: The stop_id to check.
    :param normal_buffer: The standard buffer distance in miles.
    :param large_buffer: The larger buffer distance in miles.
    :param large_buffer_ids: List of stop_ids that require the larger buffer.
    :return: Buffer distance in miles.
    """
    # Convert as needed to match what large_buffer_ids contain
    # for consistent comparison
    str_stop_id = str(stop_id)
    large_buffer_str_ids = [str(s) for s in large_buffer_ids]

    if str_stop_id in large_buffer_str_ids:
        return large_buffer
    else:
        return normal_buffer


# -----------------------------------------------------------------------------
# PEDESTRIAN NETWORK HELPERS
#
# The two functions below are copied verbatim from utils/network_helpers.py so
# this script stays self-contained (see utils/run_log.py for the same
# convention). The canonical versions live in utils/network_helpers.py — keep
# these copies in sync when updating either. Only the walking-network builder
# is reproduced; the isochrone method walks this graph from each stop.
# -----------------------------------------------------------------------------

FT_PER_MILE: float = 5_280.0
SECONDS_PER_HOUR: float = 3_600.0
DEFAULT_WALK_SPEED_MPH: float = 3.0
DEFAULT_WALK_SPEED_FT_PER_S: float = DEFAULT_WALK_SPEED_MPH * FT_PER_MILE / SECONDS_PER_HOUR
DEFAULT_NODE_GRID_FT: float = 5.0

NodeKey = Tuple[float, float]  # quantized (x, y)
EdgeID = int


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


def build_pedestrian_time_network(
    centerlines: gpd.GeoDataFrame,
    *,
    walk_speed: float = DEFAULT_WALK_SPEED_FT_PER_S,
    node_grid: float = DEFAULT_NODE_GRID_FT,
) -> tuple[nx.MultiGraph, dict[EdgeID, Tuple[NodeKey, NodeKey]]]:
    """Build a walking travel-time graph from a centerline layer.

    Each input centerline is exploded into simple :class:`LineString` segments,
    and every segment becomes one undirected edge whose endpoints are snapped to
    a grid (see :func:`quantize_node`) so adjacent segments share nodes. Edges
    carry the segment ``geometry``, its ``length`` (CRS units), and ``time_s``,
    the walking time in seconds (``length / walk_speed``).

    Args:
        centerlines: Sidewalk or road centerlines in a projected CRS. The CRS
            must be set; a geographic CRS triggers a warning because lengths
            would be measured in degrees.
        walk_speed: Walking speed in the layer's linear CRS units **per second**
            (defaults to ~3 mph in feet, for a US-foot CRS). Must be positive.
        node_grid: Grid size for merging near-coincident endpoints, in the
            layer's linear CRS units.

    Returns:
        A tuple of:
            * the undirected :class:`networkx.MultiGraph`; each node has ``x``
              and ``y`` attributes, each edge has ``edge_id``, ``geometry``,
              ``length``, and ``time_s``.
            * a mapping of ``edge_id`` to its ``(u_node, v_node)`` endpoint keys,
              for callers that need to relate edges back to graph nodes.

    Raises:
        ValueError: If ``centerlines`` has no CRS or ``walk_speed`` is not
            positive.
    """
    if centerlines.crs is None:
        raise ValueError("centerlines has no CRS; cannot build a metric walking network.")
    if walk_speed <= 0:
        raise ValueError(f"walk_speed must be positive, got {walk_speed}.")
    if centerlines.crs.is_geographic:
        logging.warning(
            "centerlines CRS '%s' is geographic; segment lengths and travel "
            "times will be meaningless. Reproject to a projected CRS first.",
            centerlines.crs,
        )

    segments = centerlines.explode(index_parts=False, ignore_index=True)
    segments = segments[segments.geometry.notna()]
    segments = segments[segments.geom_type == "LineString"]

    graph = nx.MultiGraph()
    edge_endpoints: dict[EdgeID, Tuple[NodeKey, NodeKey]] = {}

    edge_id = 0
    for geom in segments.geometry.to_numpy():
        length = float(geom.length)
        if length == 0.0:
            continue

        x1, y1 = geom.coords[0]
        x2, y2 = geom.coords[-1]
        u = quantize_node(x1, y1, node_grid)
        v = quantize_node(x2, y2, node_grid)
        if u == v:
            continue  # degenerate loop after snapping

        for node, (xx, yy) in ((u, u), (v, v)):
            if node not in graph:
                graph.add_node(node, x=xx, y=yy)

        graph.add_edge(
            u,
            v,
            edge_id=edge_id,
            geometry=geom,
            length=length,
            time_s=length / walk_speed,
        )
        edge_endpoints[edge_id] = (u, v)
        edge_id += 1

    logging.info(
        "Built pedestrian time network: %d nodes, %d edges (walk_speed=%.3f units/s).",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        walk_speed,
    )
    return graph, edge_endpoints


# -----------------------------------------------------------------------------
# SERVICE-AREA GEOMETRY BUILDERS
# -----------------------------------------------------------------------------


def build_route_shapes_gdf(
    shapes_df: Optional[pd.DataFrame],
    trips: pd.DataFrame,
    final_routes_df: pd.DataFrame,
    crs_epsg_code: int,
) -> gpd.GeoDataFrame:
    """Build dissolved route-line geometry, keyed by ``route_short_name``.

    Reconstructs a :class:`~shapely.geometry.LineString` for each ``shape_id``
    in *shapes_df*, attributes it to a route via *trips*/*final_routes_df*, and
    dissolves to one (multi)line per ``route_short_name`` in the projected CRS.

    Args:
        shapes_df: GTFS ``shapes.txt`` table, or ``None`` if unavailable.
        trips: GTFS ``trips`` table (must include ``route_id`` and ``shape_id``).
        final_routes_df: Already-filtered routes (``route_id``,
            ``route_short_name``).
        crs_epsg_code: EPSG code of the projected CRS to return geometry in.

    Returns:
        A GeoDataFrame with columns ``route_short_name`` and ``geometry``.
        Empty (but validly typed) when route geometry cannot be derived.
    """
    empty = gpd.GeoDataFrame(
        {"route_short_name": pd.Series(dtype=str)},
        geometry=gpd.GeoSeries([], crs=f"EPSG:{crs_epsg_code}"),
    )

    needed = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    if shapes_df is None or shapes_df.empty:
        logging.warning("No shapes.txt data available; route geometry cannot be built.")
        return empty
    if not needed <= set(shapes_df.columns):
        logging.warning(
            "shapes.txt is missing required columns %s; route geometry unavailable.",
            sorted(needed - set(shapes_df.columns)),
        )
        return empty
    if "shape_id" not in trips.columns:
        logging.warning("trips.txt has no 'shape_id' column; route geometry unavailable.")
        return empty

    # Map each shape_id to a route_short_name through the filtered trips.
    trip_routes = trips.merge(final_routes_df[["route_id", "route_short_name"]], on="route_id")
    shape_to_route = (
        trip_routes.dropna(subset=["shape_id"])
        .drop_duplicates(subset=["shape_id"])[["shape_id", "route_short_name"]]
        .astype({"shape_id": str})
    )
    if shape_to_route.empty:
        logging.warning("No shape_id values map to the selected routes.")
        return empty

    pts = shapes_df.copy()
    pts["shape_id"] = pts["shape_id"].astype(str)
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        pts[col] = pd.to_numeric(pts[col], errors="coerce")
    pts = pts.dropna(subset=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
    pts = pts.sort_values(["shape_id", "shape_pt_sequence"])

    lines: list[dict[str, Any]] = []
    for shape_id, group in pts.groupby("shape_id", sort=False):
        if len(group) < 2:
            continue  # need at least two points for a line
        coords = list(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
        lines.append({"shape_id": str(shape_id), "geometry": LineString(coords)})

    if not lines:
        logging.warning("No usable line geometry reconstructed from shapes.txt.")
        return empty

    shapes_gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(
        epsg=crs_epsg_code
    )
    shapes_gdf = shapes_gdf.merge(shape_to_route, on="shape_id", how="inner")
    if shapes_gdf.empty:
        logging.warning("No route shapes remain after joining shapes to routes.")
        return empty

    dissolved = shapes_gdf.dissolve(by="route_short_name").reset_index()
    return dissolved[["route_short_name", "geometry"]]


def build_walk_isochrone(
    stop_points_gdf: gpd.GeoDataFrame,
    ped_graph: nx.MultiGraph,
    *,
    walk_time_min: float,
    walk_speed_units_per_s: float,
) -> Optional[gpd.GeoDataFrame]:
    """Build a walk-time isochrone (walkshed) around the given stop points.

    Each stop is snapped to the nearest pedestrian-network node, and Dijkstra
    expands outward up to ``walk_time_min`` minutes. Every reachable node is
    buffered by the distance still walkable with its leftover time budget, and
    the union of those buffers forms the walkshed. Stops without a reachable
    node (e.g. an empty graph) are skipped.

    Args:
        stop_points_gdf: Stop *point* geometry in the projected CRS.
        ped_graph: Walking graph from :func:`build_pedestrian_time_network`
            (edges weighted by ``time_s``; nodes carry ``x``/``y``).
        walk_time_min: Walk-time budget in minutes.
        walk_speed_units_per_s: Walking speed in projected-CRS units per second
            (must match the speed used to build ``ped_graph``).

    Returns:
        A single-row GeoDataFrame holding the dissolved walkshed polygon, or
        ``None`` if nothing was reachable.
    """
    if ped_graph.number_of_nodes() == 0:
        logging.warning("Pedestrian network is empty; cannot build an isochrone.")
        return None

    cutoff_s = walk_time_min * 60.0
    node_keys = list(ped_graph.nodes)
    node_xy = np.array([(ped_graph.nodes[n]["x"], ped_graph.nodes[n]["y"]) for n in node_keys])
    tree = cKDTree(node_xy)

    polygons: list[Any] = []
    for geom in stop_points_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        _, idx = tree.query((geom.x, geom.y))
        source = node_keys[int(idx)]

        # Reachable nodes (and their walk time) within the budget.
        lengths = nx.single_source_dijkstra_path_length(
            ped_graph, source, cutoff=cutoff_s, weight="time_s"
        )
        for node, time_s in lengths.items():
            residual_units = (cutoff_s - time_s) * walk_speed_units_per_s
            if residual_units <= 0:
                continue
            node_x = ped_graph.nodes[node]["x"]
            node_y = ped_graph.nodes[node]["y"]
            polygons.append(Point(node_x, node_y).buffer(residual_units))

    if not polygons:
        logging.warning("No pedestrian-network nodes were reachable from the stops.")
        return None

    iso = unary_union(polygons)
    return gpd.GeoDataFrame(geometry=[iso], crs=stop_points_gdf.crs)


def build_service_area_polygon(
    stop_points_gdf: gpd.GeoDataFrame,
    *,
    method: str,
    buffer_distance_mi: float,
    large_buffer_distance_mi: float,
    stop_ids_large_buffer: list[str],
    route_shapes_gdf: Optional[gpd.GeoDataFrame] = None,
    ped_graph: Optional[nx.MultiGraph] = None,
    walk_time_min: float = 0.0,
    walk_speed_units_per_s: float = 0.0,
) -> Optional[gpd.GeoDataFrame]:
    """Build a single dissolved service-area polygon for a set of stops.

    Dispatches on ``method`` to produce the catchment geometry for one analysis
    unit (the whole network, a single route, or a single stop):

    * ``"stop_buffer"``: variable-radius buffer around each stop, dissolved.
    * ``"route_buffer"``: fixed-radius buffer around *route_shapes_gdf*; falls
      back to ``"stop_buffer"`` when no route geometry is supplied.
    * ``"isochrone"``: walk-time walkshed via :func:`build_walk_isochrone`.

    Args:
        stop_points_gdf: Stop *point* geometry in the projected CRS.
        method: One of ``"stop_buffer"``, ``"route_buffer"``, ``"isochrone"``.
        buffer_distance_mi: Standard buffer radius in miles.
        large_buffer_distance_mi: Larger buffer radius in miles for select stops.
        stop_ids_large_buffer: Stop IDs that should use the larger radius.
        route_shapes_gdf: Route-line geometry for the ``"route_buffer"`` method.
        ped_graph: Pedestrian graph for the ``"isochrone"`` method.
        walk_time_min: Walk-time budget (minutes) for the ``"isochrone"`` method.
        walk_speed_units_per_s: Walking speed (CRS units/s) for the isochrone.

    Returns:
        A single-row GeoDataFrame with the dissolved service area, or ``None``
        if no geometry could be produced.
    """
    if method == "isochrone":
        if ped_graph is None:
            logging.warning(
                "Isochrone method selected but no pedestrian network is loaded; "
                "falling back to stop buffers."
            )
        else:
            return build_walk_isochrone(
                stop_points_gdf,
                ped_graph,
                walk_time_min=walk_time_min,
                walk_speed_units_per_s=walk_speed_units_per_s,
            )

    if method == "route_buffer":
        if route_shapes_gdf is None or route_shapes_gdf.empty:
            logging.warning(
                "Route-buffer method selected but no route geometry is available; "
                "falling back to stop buffers."
            )
        else:
            buffered = route_shapes_gdf.geometry.buffer(buffer_distance_mi * METERS_PER_MILE)
            area = unary_union(list(buffered.values))
            return gpd.GeoDataFrame(geometry=[area], crs=stop_points_gdf.crs)

    # Default / fallback: per-stop buffers, dissolved into one polygon.
    if stop_points_gdf.empty:
        return None
    buffer_m = stop_points_gdf["stop_id"].map(
        lambda sid: (
            pick_buffer_distance(
                sid,
                normal_buffer=buffer_distance_mi,
                large_buffer=large_buffer_distance_mi,
                large_buffer_ids=stop_ids_large_buffer,
            )
            * METERS_PER_MILE
        )
    )
    buffered = stop_points_gdf.geometry.buffer(buffer_m)
    area = unary_union(list(buffered.values))
    return gpd.GeoDataFrame(geometry=[area], crs=stop_points_gdf.crs)


def clip_and_calculate_synthetic_fields(
    demographics_gdf: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    synthetic_fields: list[str],
) -> gpd.GeoDataFrame:
    """Clip *demographics_gdf* with *buffer_gdf* and compute synthetic totals.

    Steps
    -----
    1.  Ensure an original-area column exists (acres).
    2.  Clip polygons to the buffer.
    3.  Compute clipped-area and area-percentage.
    4.  For each requested field that exists, multiply by area percentage
        to create ``synthetic_<field>`` columns.
       * Missing fields are reported once and silently skipped.
    """
    # ---------------------------------------------------------------
    # 1. Original area (acres) — if not already present
    # ---------------------------------------------------------------
    if "area_ac_og" not in demographics_gdf.columns:
        demographics_gdf["area_ac_og"] = demographics_gdf.geometry.area / 4046.86

    # ---------------------------------------------------------------
    # 2. Clip to buffer
    # ---------------------------------------------------------------
    clipped_gdf = gpd.clip(demographics_gdf, buffer_gdf)

    # ---------------------------------------------------------------
    # 3. Clipped area + percentage
    # ---------------------------------------------------------------
    clipped_gdf["area_ac_cl"] = clipped_gdf.geometry.area / 4046.86
    clipped_gdf["area_perc"] = clipped_gdf["area_ac_cl"] / clipped_gdf["area_ac_og"]

    # Handle divide-by-zero and NaN without chained-assignment warnings
    clipped_gdf["area_perc"] = (
        clipped_gdf["area_perc"].replace([float("inf"), -float("inf")], 0).fillna(0)
    )

    # ---------------------------------------------------------------
    # 4. Synthetic fields — skip any that are missing
    # ---------------------------------------------------------------
    missing = [f for f in synthetic_fields if f not in clipped_gdf.columns]
    if missing:
        logging.warning("Synthetic field(s) not found and will be skipped: %s", missing)

    for field in synthetic_fields:
        if field not in clipped_gdf.columns:
            continue  # silently skip after the single warning above

        numeric = pd.to_numeric(clipped_gdf[field], errors="coerce").fillna(0)
        clipped_gdf[f"synthetic_{field}"] = clipped_gdf["area_perc"] * numeric

    return clipped_gdf


def _present_synthetic_cols(clipped: gpd.GeoDataFrame, synthetic_fields: list[str]) -> list[str]:
    """Return the ``synthetic_<field>`` columns that clipping actually produced.

    ``clip_and_calculate_synthetic_fields`` only emits a ``synthetic_<field>`` column
    for a field present in the demographics layer, so a layer that lacks some
    ``SYNTHETIC_FIELDS`` (e.g. no LEHD jobs table, or tract estimates the census step
    did not produce) simply yields fewer of them. Selecting only the columns that
    materialized keeps a missing field from ``KeyError``-ing the whole analysis.
    """
    return [f"synthetic_{f}" for f in synthetic_fields if f"synthetic_{f}" in clipped.columns]


def export_summary_to_excel(totals_dict: dict, output_path: str) -> None:
    """Write a dictionary of aggregated synthetic fields to a single-row Excel file.

    :param totals_dict: A dictionary of {synthetic_field_name: numeric_total}.
    :param output_path: File path for the .xlsx output.
    """
    # Convert the dictionary to a single-row DataFrame
    summary_data = {k: [v] for k, v in totals_dict.items()}
    summary_df = pd.DataFrame(summary_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_excel(output_path, index=False)
    logging.info("Exported Excel summary: %s", output_path)


def _stops_to_points_gdf(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    stops_df: pd.DataFrame,
    final_routes_df: pd.DataFrame,
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
) -> Optional[gpd.GeoDataFrame]:
    """Merge GTFS tables and return filtered stop *points* in the projected CRS.

    Args:
        trips: GTFS ``trips`` table (already calendar-filtered).
        stop_times: GTFS ``stop_times`` table.
        stops_df: GTFS ``stops`` table.
        final_routes_df: Already route-filtered routes (``route_id``,
            ``route_short_name``).
        stop_ids_to_include: Stop IDs to include (empty = no include filter).
        stop_ids_to_exclude: Stop IDs to exclude (empty = no exclude filter).

    Returns:
        A GeoDataFrame of stop points with ``route_short_name`` and ``stop_id``
        columns, or ``None`` if no stops survive the filters.
    """
    trips_merged = trips.merge(final_routes_df[["route_id", "route_short_name"]], on="route_id")
    merged_data = stop_times.merge(trips_merged, on="trip_id")
    merged_data = merged_data.merge(stops_df, on="stop_id")

    final_stops_df = get_included_stops(merged_data, stop_ids_to_include, stop_ids_to_exclude)
    if final_stops_df.empty:
        return None

    final_stops_df["geometry"] = final_stops_df.apply(
        lambda row: Point(float(row["stop_lon"]), float(row["stop_lat"])),
        axis=1,
    )
    return gpd.GeoDataFrame(final_stops_df, geometry="geometry", crs="EPSG:4326").to_crs(
        epsg=CRS_EPSG_CODE
    )


def flag_express_origin_candidates(stop_jobs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Flag job-poor express-route stops as likely origins (pure ranking step).

    On an express route, every stop is essentially either a suburban
    park-and-ride *origin* (few jobs nearby) or an employment-anchor
    *destination* (many jobs nearby), so jobs-near-the-stop separates the two.
    A stop is flagged when its nearby-jobs count falls below ``threshold`` times
    the job-richest stop on the *same* route — a relative, per-route rule that is
    scale-free across big and small job markets. Routes whose stops all show zero
    nearby jobs are inconclusive and flag nothing.

    Args:
        stop_jobs: One row per (route, stop) with at least ``route_short_name``,
            ``stop_id``, and ``nearby_jobs`` columns.
        threshold: Fraction of a route's maximum stop-jobs below which a stop is
            flagged (e.g. ``0.2`` flags stops under 20% of the route's richest).

    Returns:
        A copy of *stop_jobs* with added ``jobs_share`` (the stop's share of its
        route's total nearby jobs) and ``likely_origin`` (bool) columns, sorted
        by route then ascending nearby jobs.
    """
    out = stop_jobs.copy()
    out["nearby_jobs"] = pd.to_numeric(out["nearby_jobs"], errors="coerce").fillna(0.0)

    grouped = out.groupby("route_short_name")["nearby_jobs"]
    route_max = grouped.transform("max")
    route_sum = grouped.transform("sum")

    out["jobs_share"] = (out["nearby_jobs"] / route_sum).where(route_sum > 0, 0.0)
    out["likely_origin"] = (route_max > 0) & (out["nearby_jobs"] < threshold * route_max)

    return out.sort_values(["route_short_name", "nearby_jobs"]).reset_index(drop=True)


def suggest_express_origin_stops(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    *,
    express_route_ids: set[str],
    buffer_distance_mi: float,
    jobs_field: str,
    threshold: float,
    output_dir: str,
) -> Optional[pd.DataFrame]:
    """Suggest which express-route stops are likely origins, from nearby jobs.

    Advisory only: this never changes any buffer. For every stop on an express
    route (``route_id`` in *express_route_ids*) it counts the jobs within a walk
    buffer, ranks them per route via :func:`flag_express_origin_candidates`, and
    writes ``express_origin_candidates.csv`` plus a warning naming the flagged
    stops — so the flagged ``stop_id`` values can be reviewed and copied into
    ``EXPRESS_ORIGIN_STOP_IDS``.

    No-ops (returning ``None``) when there are no express routes, the demographics
    layer lacks *jobs_field*, or no express stops resolve.

    Args:
        trips: Calendar-filtered GTFS ``trips`` table.
        stop_times: GTFS ``stop_times`` table.
        routes_df: GTFS ``routes`` table (needs ``route_id``/``route_short_name``).
        stops_df: GTFS ``stops`` table.
        demographics_gdf: Projected demographics layer carrying *jobs_field*.
        express_route_ids: ``route_id`` values treated as express.
        buffer_distance_mi: Walk-buffer radius used to count nearby jobs.
        jobs_field: Demographics column holding the job count (e.g. ``tot_empl``).
        threshold: Per-route flag cutoff passed through to the ranking step.
        output_dir: Directory the candidates CSV is written to.

    Returns:
        The ranked candidates DataFrame, or ``None`` when the advisory cannot run.
    """
    if not express_route_ids:
        return None
    if jobs_field not in demographics_gdf.columns:
        logging.info(
            "Express-origin advisory skipped: demographics layer has no jobs field '%s'.",
            jobs_field,
        )
        return None

    express_routes_df = routes_df[routes_df["route_id"].astype(str).isin(express_route_ids)]
    if express_routes_df.empty:
        logging.info("Express-origin advisory: no express route_id matched routes.txt; skipping.")
        return None

    stops_gdf = _stops_to_points_gdf(trips, stop_times, stops_df, express_routes_df, [], [])
    if stops_gdf is None or stops_gdf.empty:
        logging.info("Express-origin advisory: no stops on the express routes; skipping.")
        return None

    # Jobs near a stop depend only on the stop, so compute once per stop_id and
    # attribute back to each (route, stop) pair for per-route normalization.
    keep_cols = ["stop_id", "route_short_name"]
    if "stop_name" in stops_gdf.columns:
        keep_cols.append("stop_name")
    pairs = stops_gdf[keep_cols].drop_duplicates().reset_index(drop=True)

    jobs_col = f"synthetic_{jobs_field}"
    jobs_by_stop: dict[str, float] = {}
    for stop_id, geom in (
        stops_gdf.drop_duplicates(subset="stop_id").set_index("stop_id").geometry.items()
    ):
        key = str(stop_id)
        single = gpd.GeoDataFrame({"stop_id": [key]}, geometry=[geom], crs=stops_gdf.crs)
        area = build_service_area_polygon(
            single,
            method="stop_buffer",
            buffer_distance_mi=buffer_distance_mi,
            large_buffer_distance_mi=buffer_distance_mi,
            stop_ids_large_buffer=[],
        )
        if area is None or area.empty:
            jobs_by_stop[key] = 0.0
            continue
        clipped = clip_and_calculate_synthetic_fields(demographics_gdf, area, [jobs_field])
        jobs_by_stop[key] = float(clipped[jobs_col].sum()) if jobs_col in clipped.columns else 0.0

    pairs["nearby_jobs"] = pairs["stop_id"].astype(str).map(jobs_by_stop).fillna(0.0)
    ranked = flag_express_origin_candidates(pairs, threshold)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "express_origin_candidates.csv")
    ranked.to_csv(csv_path, index=False)

    flagged = ranked[ranked["likely_origin"]]
    if flagged.empty:
        logging.info(
            "Express-origin advisory: no likely origins flagged (wrote %s for review).", csv_path
        )
    else:
        logging.warning(
            "Express-origin advisory: %d stop(s) MAY be express origins (low nearby jobs). "
            "Review %s, then add confirmed stop_ids to EXPRESS_ORIGIN_STOP_IDS. Flagged: %s",
            len(flagged),
            csv_path,
            ", ".join(
                f"{r.stop_id} (route {r.route_short_name})" for r in flagged.itertuples(index=False)
            ),
        )
    return ranked


def do_network_analysis(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
    buffer_distance_mi: float,
    large_buffer_distance_mi: float,
    stop_ids_large_buffer: list[str],
    output_dir: str,
    synthetic_fields: list[str],
    *,
    service_area_method: str = "stop_buffer",
    shapes_df: Optional[pd.DataFrame] = None,
    ped_graph: Optional[nx.MultiGraph] = None,
    walk_time_min: float = 0.0,
    walk_speed_units_per_s: float = 0.0,
    express_route_ids: Optional[set[str]] = None,
) -> None:
    """Run a single network-wide service-area/clip analysis.

    The function filters routes and stops, builds one combined service area
    using the selected ``service_area_method``, clips demographic polygons, and
    exports both geometry and an Excel summary.

    Args:
        trips: DataFrame from *trips.txt*.
        stop_times: DataFrame from *stop_times.txt*.
        routes_df: DataFrame from *routes.txt*.
        stops_df: DataFrame from *stops.txt*.
        demographics_gdf: GeoDataFrame containing demographic data.
        routes_to_include: List of route_short_names to include.
        routes_to_exclude: List of route_short_names to exclude.
        stop_ids_to_include: List of stop_ids to include.
        stop_ids_to_exclude: List of stop_ids to exclude.
        buffer_distance_mi: Standard buffer distance in miles.
        large_buffer_distance_mi: Larger buffer distance in miles for specific stops.
        stop_ids_large_buffer: List of stop_ids that should use the large buffer distance.
        output_dir: Directory to save output files.
        synthetic_fields: List of demographic fields to synthesize.
        service_area_method: ``"stop_buffer"``, ``"route_buffer"``, or
            ``"isochrone"``.
        shapes_df: GTFS ``shapes.txt`` (for the ``route_buffer`` method).
        ped_graph: Pedestrian network (for the ``isochrone`` method).
        walk_time_min: Walk-time budget in minutes (isochrone method).
        walk_speed_units_per_s: Walking speed in CRS units/s (isochrone method).
        express_route_ids: Express ``route_id`` values. Accepted for a uniform
            dispatch signature but unused here — the ``service_type`` label is a
            per-route output, emitted only by ``do_route_by_route_analysis``.

    Returns:
        - A single shapefile (all_routes_service_buffer_data.shp)
        - A single Excel summary (all_routes_service_buffer_data.xlsx)
    """
    logging.info("\n=== Network-wide Analysis (%s) ===", service_area_method)

    final_routes_df = get_included_routes(routes_df, routes_to_include, routes_to_exclude)
    if final_routes_df.empty:
        logging.info("No routes remain after route filters. Aborting network analysis.")
        return

    stops_gdf = _stops_to_points_gdf(
        trips, stop_times, stops_df, final_routes_df, stop_ids_to_include, stop_ids_to_exclude
    )
    if stops_gdf is None:
        logging.info("No stops remain after stop filters. Aborting network analysis.")
        return
    unique_stops_gdf = stops_gdf.drop_duplicates(subset="stop_id")

    route_shapes_gdf = build_route_shapes_gdf(shapes_df, trips, final_routes_df, CRS_EPSG_CODE)

    service_area_gdf = build_service_area_polygon(
        unique_stops_gdf,
        method=service_area_method,
        buffer_distance_mi=buffer_distance_mi,
        large_buffer_distance_mi=large_buffer_distance_mi,
        stop_ids_large_buffer=stop_ids_large_buffer,
        route_shapes_gdf=route_shapes_gdf,
        ped_graph=ped_graph,
        walk_time_min=walk_time_min,
        walk_speed_units_per_s=walk_speed_units_per_s,
    )
    if service_area_gdf is None or service_area_gdf.empty:
        logging.info("Could not build a network service area. Aborting network analysis.")
        return

    clipped_result = clip_and_calculate_synthetic_fields(
        demographics_gdf,
        service_area_gdf,
        synthetic_fields,
    )
    synthetic_cols = _present_synthetic_cols(clipped_result, synthetic_fields)
    totals = clipped_result[synthetic_cols].sum().round(0)

    logging.info("Network-wide totals:")
    for col, value in totals.items():
        display_col = str(col).replace("synthetic_", "").replace("_", " ").title()
        logging.info("  Total Synthetic %s: %d", display_col, int(value))

    os.makedirs(output_dir, exist_ok=True)
    shp_path = os.path.join(output_dir, "all_routes_service_buffer_data.shp")
    clipped_result.to_file(shp_path)
    logging.info("Exported network shapefile: %s", shp_path)

    xlsx_path = os.path.join(output_dir, "all_routes_service_buffer_data.xlsx")
    final_dict = {col: int(val) for col, val in totals.items()}
    export_summary_to_excel(final_dict, xlsx_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    service_area_gdf.plot(ax=ax, alpha=0.5, label="Service Area")
    unique_stops_gdf.plot(ax=ax, color="black", markersize=2, label="Stops")
    plt.title(f"Network Service Area ({service_area_method})")
    plt.legend()
    plt.show()


def express_route_totals(
    route_stops_gdf: gpd.GeoDataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    synthetic_fields: list[str],
    *,
    origin_stop_ids: set[str],
    walk_buffer_mi: float,
    origin_buffer_mi: float,
    employment_fields: Sequence[str],
    zero_origin_employment: bool,
    zero_destination_population: bool,
) -> Tuple[Optional[dict[str, int]], Optional[gpd.GeoDataFrame]]:
    """Access-mode demographic totals for one express route.

    Buffer width follows how each demographic reaches the bus, not a stop's
    label:

    * **Population** is drive-accessed at origin (park-and-ride) stops — those in
      *origin_stop_ids* get the wide ``origin_buffer_mi`` drive-shed — and
      walk-accessed (``walk_buffer_mi``) at every other stop.
    * **Employment** is always walk-accessed: riders walk from the alighting stop
      to the job, so every stop uses ``walk_buffer_mi`` for jobs. A wide buffer
      would imply driving miles from the bus to the desk and overcount jobs.

    The two directional cross-terms are then dropped per the flags:

    * ``zero_destination_population`` — count population only from the origin
      (drive) end, not the non-origin (destination / walk) stops.
    * ``zero_origin_employment`` — count employment only from the non-origin
      (destination / walk) stops, not the origin stops' walk buffer.

    A unidirectional express sets both flags True (population@origin +
    employment@destination); a bidirectional express sets both False (both ends
    contribute population *and* employment, employment still walk-scaled).

    Args:
        route_stops_gdf: This route's stop points (projected CRS), deduped by stop.
        demographics_gdf: Projected demographics layer carrying *synthetic_fields*.
        synthetic_fields: Demographic count fields to total.
        origin_stop_ids: stop_ids treated as drive-access origins (park-and-rides).
        walk_buffer_mi: Walk-access buffer radius in miles.
        origin_buffer_mi: Drive-access buffer radius in miles for origin stops.
        employment_fields: Subset of *synthetic_fields* that are job counts; every
            other field is treated as residence-based (population).
        zero_origin_employment: Drop the origin stops' (walk-buffer) employment.
        zero_destination_population: Drop the non-origin stops' population.

    Returns:
        ``(route_totals, combined_clip)`` where *route_totals* maps each present
        bare field name to its integer total and *combined_clip* is the clipped
        catchment for the per-route shapefile; or ``(None, None)`` when a required
        catchment is empty (e.g. a unidirectional route with no origin stop or no
        non-origin stop), so the caller can fall back to symmetric accounting.
    """
    sid = route_stops_gdf["stop_id"].astype(str)
    origin_ids = {str(s) for s in origin_stop_ids}
    origin_stops = route_stops_gdf[sid.isin(origin_ids)]
    other_stops = route_stops_gdf[~sid.isin(origin_ids)]

    def _buffer(stops: gpd.GeoDataFrame, radius_mi: float) -> Optional[gpd.GeoDataFrame]:
        if stops.empty:
            return None
        return build_service_area_polygon(
            stops,
            method="stop_buffer",
            buffer_distance_mi=radius_mi,
            large_buffer_distance_mi=radius_mi,
            stop_ids_large_buffer=[],
        )

    # Population: drive-shed at the origin stops, walk buffer elsewhere. Dropping
    # destination population leaves only the origin drive-shed; the not-dropped
    # case is dissolved into one polygon so the wide and walk parts do not
    # double-count where they overlap.
    if zero_destination_population:
        population_area = _buffer(origin_stops, origin_buffer_mi)
    else:
        population_area = build_service_area_polygon(
            route_stops_gdf,
            method="stop_buffer",
            buffer_distance_mi=walk_buffer_mi,
            large_buffer_distance_mi=origin_buffer_mi,
            stop_ids_large_buffer=sorted(origin_ids),
        )

    # Employment: walk buffer only — the non-origin stops, plus the origin stops'
    # walk buffer unless origin employment is dropped.
    if zero_origin_employment:
        employment_area = _buffer(other_stops, walk_buffer_mi)
    else:
        employment_area = _buffer(route_stops_gdf, walk_buffer_mi)

    if (
        population_area is None
        or population_area.empty
        or employment_area is None
        or employment_area.empty
    ):
        return None, None

    pop_clip = clip_and_calculate_synthetic_fields(
        demographics_gdf, population_area, synthetic_fields
    )
    empl_clip = clip_and_calculate_synthetic_fields(
        demographics_gdf, employment_area, synthetic_fields
    )

    # Only emit fields that materialized, matching the symmetric path (a field
    # missing from the layer is omitted, not zeroed).
    present = {
        col.replace("synthetic_", "")
        for col in (
            *_present_synthetic_cols(pop_clip, synthetic_fields),
            *_present_synthetic_cols(empl_clip, synthetic_fields),
        )
    }
    employment = set(employment_fields)

    def _total(clipped: gpd.GeoDataFrame, field: str) -> float:
        col = f"synthetic_{field}"
        return float(clipped[col].sum()) if col in clipped.columns else 0.0

    # Iterate synthetic_fields to preserve the symmetric path's column order;
    # employment fields come from the walk catchment, the rest from population.
    route_totals: dict[str, int] = {}
    for field in synthetic_fields:
        if field not in present:
            continue
        source = empl_clip if field in employment else pop_clip
        route_totals[field] = int(round(_total(source, field)))

    # One dissolved catchment (population area ∪ employment area) for the
    # per-route shapefile, so overlapping blocks are not written twice.
    combined_geom = unary_union(list(population_area.geometry) + list(employment_area.geometry))
    combined_area = gpd.GeoDataFrame(geometry=[combined_geom], crs=population_area.crs)
    combined_clip = clip_and_calculate_synthetic_fields(
        demographics_gdf, combined_area, synthetic_fields
    )
    return route_totals, combined_clip


def do_route_by_route_analysis(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
    buffer_distance_mi: float,
    large_buffer_distance_mi: float,
    stop_ids_large_buffer: list[str],
    output_dir: str,
    synthetic_fields: list[str],
    *,
    service_area_method: str = "stop_buffer",
    shapes_df: Optional[pd.DataFrame] = None,
    ped_graph: Optional[nx.MultiGraph] = None,
    walk_time_min: float = 0.0,
    walk_speed_units_per_s: float = 0.0,
    express_route_ids: Optional[set[str]] = None,
    unidirectional_route_ids: Optional[set[str]] = None,
    employment_fields: Optional[Sequence[str]] = None,
    zero_origin_employment: bool = True,
    zero_destination_population: bool = True,
) -> None:
    """Perform a service-area/clip analysis for each individual route.

    The procedure repeats the selected ``service_area_method`` workflow for
    every ``route_short_name`` in the filtered set, exporting per-route
    shapefiles plus one combined route-level CSV of totals.

    Exports:
      - Per route_short_name R: a shapefile R_service_buffer_data.shp.
      - One combined ``service_demographics_by_route.csv`` keyed on ``route_id``
        (one row per route_id) — the table the feature bundle / monthly_ridership_model.py
        consume. The route identity lives in a real column here, not just the
        filename, so the orchestrator can join it onto the ridership anchor. A
        ``service_type`` column labels each route ``"express"`` (its route_id is
        in ``express_route_ids``) or ``"local"`` so the model can fit the two
        differently, and an ``express_direction`` column records
        ``"unidirectional"``, ``"bidirectional"``, or ``""``.

    Express routes (route_id in *express_route_ids*) get access-mode accounting
    via :func:`express_route_totals`: population is drive-scaled at the
    park-and-ride origin stops (wide buffer) and walk-scaled elsewhere, while
    employment is always walk-scaled. Unidirectional routes (in
    *unidirectional_route_ids*) additionally drop the two cross-terms per
    *zero_origin_employment* / *zero_destination_population*; bidirectional routes
    keep both ends. This needs the ``"stop_buffer"`` method (and, for
    unidirectional, at least one origin and one non-origin stop); otherwise the
    route falls back to symmetric accounting (warned). Local routes keep the
    symmetric single-catchment totals.
    """
    logging.info("\n=== Route-by-Route Analysis (%s) ===", service_area_method)

    final_routes_df = get_included_routes(routes_df, routes_to_include, routes_to_exclude)
    if final_routes_df.empty:
        logging.info("No routes remain after route filters. Aborting route-by-route analysis.")
        return

    stops_gdf = _stops_to_points_gdf(
        trips, stop_times, stops_df, final_routes_df, stop_ids_to_include, stop_ids_to_exclude
    )
    if stops_gdf is None:
        logging.info("No stops remain after stop filters. Aborting route-by-route analysis.")
        return

    # Route-line geometry (used only by the "route_buffer" method).
    route_shapes_gdf = build_route_shapes_gdf(shapes_df, trips, final_routes_df, CRS_EPSG_CODE)

    os.makedirs(output_dir, exist_ok=True)

    # Map each route_short_name back to its route_id(s) so the combined summary
    # below is keyed on route_id (what monthly_ridership_model.py joins on). A short name that
    # maps to several route_ids yields one summary row per route_id.
    short_to_route_ids = (
        final_routes_df.groupby("route_short_name")["route_id"].apply(list).to_dict()
    )
    # Express routes are labeled via a service_type column on each output row (an
    # empty set leaves every route "local"); unidirectional express routes also
    # get directional (origin-pop + destination-jobs) accounting.
    express_set = {str(r) for r in (express_route_ids or set())}
    uni_set = {str(r) for r in (unidirectional_route_ids or set())}
    empl_fields = list(employment_fields) if employment_fields is not None else []
    summary_records: list[dict[str, object]] = []
    express_fallback_warned = False

    for route_name in stops_gdf["route_short_name"].unique():
        logging.info("\nProcessing route: %s", route_name)
        route_stops_gdf = stops_gdf[stops_gdf["route_short_name"] == route_name].drop_duplicates(
            subset="stop_id"
        )
        if route_stops_gdf.empty:
            logging.info("No stops found for route '%s' - skipping.", route_name)
            continue

        route_only_shapes = (
            route_shapes_gdf[route_shapes_gdf["route_short_name"] == route_name]
            if not route_shapes_gdf.empty
            else route_shapes_gdf
        )

        route_id_list = short_to_route_ids.get(route_name, [route_name])
        route_is_express = any(str(rid) in express_set for rid in route_id_list)
        route_is_unidirectional = bool(uni_set) and any(
            str(rid) in uni_set for rid in route_id_list
        )

        # Express routes get access-mode accounting (population drive-scaled at
        # park-and-ride origins, employment always walk-scaled). Unidirectional
        # routes additionally drop the two cross-terms via the zero_* knobs;
        # bidirectional routes keep both ends. Local routes use the symmetric
        # single-catchment totals in the else branch.
        route_totals: Optional[dict[str, int]] = None
        clipped_result: Optional[gpd.GeoDataFrame] = None
        if route_is_express:
            if service_area_method == "stop_buffer":
                # The cross-term knobs are unidirectional-only; bidirectional
                # routes keep both ends' population and employment.
                zero_origin_empl = zero_origin_employment if route_is_unidirectional else False
                zero_dest_pop = zero_destination_population if route_is_unidirectional else False
                route_totals, clipped_result = express_route_totals(
                    route_stops_gdf,
                    demographics_gdf,
                    synthetic_fields,
                    origin_stop_ids=set(stop_ids_large_buffer),
                    walk_buffer_mi=buffer_distance_mi,
                    origin_buffer_mi=large_buffer_distance_mi,
                    employment_fields=empl_fields,
                    zero_origin_employment=zero_origin_empl,
                    zero_destination_population=zero_dest_pop,
                )
                if route_totals is None and route_is_unidirectional:
                    logging.warning(
                        "Route '%s' is flagged unidirectional express but its stops do not "
                        "split into an origin and a destination end (need at least one "
                        "express-origin stop and one other stop on the route); using "
                        "symmetric accounting. List the route's park-and-ride stop_ids in "
                        "the express-origin settings.",
                        route_name,
                    )
            elif not express_fallback_warned:
                logging.warning(
                    "Express access-mode accounting needs the 'stop_buffer' method; "
                    "'%s' is in use, so express routes use symmetric accounting.",
                    service_area_method,
                )
                express_fallback_warned = True

        if route_totals is not None and clipped_result is not None:
            for field, val in route_totals.items():
                display = field.replace("_", " ").title()
                logging.info("  Express %s for route %s: %d", display, route_name, int(val))
        else:
            service_area_gdf = build_service_area_polygon(
                route_stops_gdf,
                method=service_area_method,
                buffer_distance_mi=buffer_distance_mi,
                large_buffer_distance_mi=large_buffer_distance_mi,
                stop_ids_large_buffer=stop_ids_large_buffer,
                route_shapes_gdf=route_only_shapes,
                ped_graph=ped_graph,
                walk_time_min=walk_time_min,
                walk_speed_units_per_s=walk_speed_units_per_s,
            )
            if service_area_gdf is None or service_area_gdf.empty:
                logging.info(
                    "Could not build a service area for route '%s' - skipping.", route_name
                )
                continue

            clipped_result = clip_and_calculate_synthetic_fields(
                demographics_gdf, service_area_gdf, synthetic_fields
            )

            synthetic_cols = _present_synthetic_cols(clipped_result, synthetic_fields)
            totals = clipped_result[synthetic_cols].sum().round(0)
            for col, val in totals.items():
                display_col = str(col).replace("synthetic_", "").replace("_", " ").title()
                logging.info(
                    "  Total Synthetic %s for route %s: %d", display_col, route_name, int(val)
                )
            route_totals = {
                str(col).replace("synthetic_", ""): int(val) for col, val in totals.items()
            }

        shp_path = os.path.join(output_dir, f"{route_name}_service_buffer_data.shp")
        clipped_result.to_file(shp_path)
        logging.info("Exported shapefile for route %s: %s", route_name, shp_path)

        # Accumulate this route's totals into one combined, route_id-keyed table
        # written after the loop. The per-route Excel and blocking plt.show() are
        # intentionally gone: the combined CSV supersedes the former, and the
        # latter stalls headless subprocess runs under the orchestrator.
        for route_id in route_id_list:
            rid = str(route_id)
            if rid in uni_set:
                direction = "unidirectional"
            elif rid in express_set:
                direction = "bidirectional"
            else:
                direction = ""
            summary_records.append(
                {
                    "route_id": route_id,
                    "route_short_name": route_name,
                    "service_type": "express" if rid in express_set else "local",
                    "express_direction": direction,
                    **route_totals,
                }
            )

    # Write one combined, route_id-keyed CSV for the bundle. prep_features_public.py
    # collects this; monthly_ridership_model.py joins it onto the ridership anchor by route_id.
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        csv_path = os.path.join(output_dir, "service_demographics_by_route.csv")
        summary_df.to_csv(csv_path, index=False)
        logging.info(
            "Wrote route-level demographics summary: %s (%d row(s)).", csv_path, len(summary_df)
        )
    else:
        logging.info("No per-route demographics produced; combined CSV not written.")


def do_stop_by_stop_analysis(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
    buffer_distance_mi: float,
    large_buffer_distance_mi: float,
    stop_ids_large_buffer: list[str],
    output_dir: str,
    synthetic_fields: list[str],
    *,
    service_area_method: str = "stop_buffer",
    shapes_df: Optional[pd.DataFrame] = None,
    ped_graph: Optional[nx.MultiGraph] = None,
    walk_time_min: float = 0.0,
    walk_speed_units_per_s: float = 0.0,
    express_route_ids: Optional[set[str]] = None,
) -> None:
    """Compute a service area and demographic catchment for each stop.

    Each GTFS stop that survives the route, trip, and stop filters gets its own
    service area built with the selected ``service_area_method``, clipped
    against the demographic layer, and written to individual shapefile/Excel
    pairs. The ``"route_buffer"`` method is not meaningful for a single stop, so
    it is treated as ``"stop_buffer"`` here.

    Exports, for each stop_id S:
      - A shapefile named stop_S_service_buffer_data.shp
      - A summary Excel named stop_S_service_buffer_data.xlsx
    """
    logging.info("\n=== Stop-by-Stop Analysis (%s) ===", service_area_method)

    effective_method = service_area_method
    if effective_method == "route_buffer":
        logging.warning(
            "'route_buffer' is not meaningful per stop; using 'stop_buffer' for "
            "stop-by-stop analysis."
        )
        effective_method = "stop_buffer"

    final_routes_df = get_included_routes(routes_df, routes_to_include, routes_to_exclude)
    if final_routes_df.empty:
        logging.info("No routes remain after route filters. Aborting stop-by-stop analysis.")
        return

    stops_gdf = _stops_to_points_gdf(
        trips, stop_times, stops_df, final_routes_df, stop_ids_to_include, stop_ids_to_exclude
    )
    if stops_gdf is None:
        logging.info("No stops remain after stop filters. Aborting stop-by-stop analysis.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for sid in stops_gdf["stop_id"].unique():
        single_stop_gdf = stops_gdf[stops_gdf["stop_id"] == sid].drop_duplicates(subset="stop_id")
        if single_stop_gdf.empty:
            continue

        stop_id_str = str(sid)
        service_area_gdf = build_service_area_polygon(
            single_stop_gdf,
            method=effective_method,
            buffer_distance_mi=buffer_distance_mi,
            large_buffer_distance_mi=large_buffer_distance_mi,
            stop_ids_large_buffer=stop_ids_large_buffer,
            ped_graph=ped_graph,
            walk_time_min=walk_time_min,
            walk_speed_units_per_s=walk_speed_units_per_s,
        )
        if service_area_gdf is None or service_area_gdf.empty:
            logging.info("Could not build a service area for stop %s - skipping.", stop_id_str)
            continue

        clipped_result = clip_and_calculate_synthetic_fields(
            demographics_gdf, service_area_gdf, synthetic_fields
        )
        synthetic_cols = _present_synthetic_cols(clipped_result, synthetic_fields)
        totals = clipped_result[synthetic_cols].sum().round(0)

        logging.info("\nStop %s totals:", stop_id_str)
        for col, val in totals.items():
            display_col = str(col).replace("synthetic_", "").replace("_", " ").title()
            logging.info("  Total Synthetic %s: %d", display_col, int(val))

        shp_path = os.path.join(output_dir, f"stop_{stop_id_str}_service_buffer_data.shp")
        clipped_result.to_file(shp_path)
        logging.info("Exported shapefile for stop %s: %s", stop_id_str, shp_path)

        xlsx_path = os.path.join(output_dir, f"stop_{stop_id_str}_service_buffer_data.xlsx")
        final_dict = {col: int(val) for col, val in totals.items()}
        export_summary_to_excel(final_dict, xlsx_path)

        fig, ax = plt.subplots(figsize=(8, 8))
        service_area_gdf.plot(ax=ax, alpha=0.5, label=f"Stop {stop_id_str} Service Area")
        single_stop_gdf.plot(ax=ax, color="black", markersize=8, label="Stop")
        plt.title(f"Stop {stop_id_str} Service Area ({effective_method})")
        plt.legend()
        plt.show()


def apply_fips_filter(
    demog_gdf: gpd.GeoDataFrame,
    fips_filter: list[str],
    fips_col: str = "FIPS",
) -> gpd.GeoDataFrame:
    """Filter *demog_gdf* by county FIPS codes.

    If *fips_col* is absent the function tries to derive it from the first
    column whose name starts with ``GEOID`` (block, tract, etc.), slicing the
    first 5 characters.  If that also fails, the filter is skipped with a
    warning.
    """
    if not fips_filter:
        logging.info("No FIPS filter provided; processing all features.")
        return demog_gdf

    if fips_col not in demog_gdf.columns:
        # attempt automatic derivation
        geo_cols = [c for c in demog_gdf.columns if c.lower().startswith("geoid")]
        if geo_cols:
            src = geo_cols[0]
            demog_gdf[fips_col] = demog_gdf[src].str[:5]
            logging.info("Derived %s from %s (first 5 chars) for FIPS filtering.", fips_col, src)
        else:
            logging.warning(
                "FIPS filter requested (%s) but no '%s' column or GEOID-like "
                "field found.  Skipping the filter.",
                fips_filter,
                fips_col,
            )
            return demog_gdf

    before = len(demog_gdf)
    demog_gdf = demog_gdf[demog_gdf[fips_col].isin(fips_filter)]
    logging.info(
        "Applied FIPS filter %s — %d → %d features.",
        fips_filter,
        before,
        len(demog_gdf),
    )
    return demog_gdf


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

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

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


def load_id_set(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
    *,
    kind: str = "id",
) -> set[str]:
    """Union an inline list and an optional text file of ids into one set.

    Copied verbatim from utils/gtfs_helpers.py so this script stays
    self-contained (same convention as the helpers above). Keep both copies in
    sync when updating either.

    Used to resolve override lists (express routes, express origin stops, …) that
    a caller may supply inline, in an external file, or both — without repeating
    the parsing for each one.

    Args:
        inline_ids: Id values supplied directly (e.g. a config list). ``None`` is
            treated as empty.
        txt_path: Path to a text file with one id per line. Blank lines are
            skipped and ``#`` starts a comment (whole-line or inline). ``None``
            skips the file. A path that is set but missing is logged as a warning
            and skipped — the inline ids are still returned.
        kind: Human-readable noun used only in log messages (e.g.
            ``"express route"``, ``"express origin stop"``).

    Returns:
        The unioned set of id strings (possibly empty). Every id is coerced to a
        trimmed ``str`` so it matches GTFS values, which are read as strings.
    """
    ids: set[str] = set()

    for raw in inline_ids or ():
        text = str(raw).strip()
        if text:
            ids.add(text)

    if txt_path:
        if not os.path.exists(txt_path):
            logging.warning(
                "%s file '%s' not found; using inline ids only.", kind.capitalize(), txt_path
            )
        else:
            with open(txt_path, encoding="utf-8") as handle:
                for line in handle:
                    text = line.split("#", 1)[0].strip()
                    if text:
                        ids.add(text)
            logging.info("Loaded %s ids from '%s'.", kind, txt_path)

    logging.info("Resolved %d %s id(s).", len(ids), kind)
    return ids


def load_express_route_ids(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
) -> set[str]:
    """Resolve the set of express-route ``route_id`` values (see ``load_id_set``).

    Thin wrapper kept for readable call sites and backwards compatibility. Copied
    verbatim from utils/gtfs_helpers.py — keep both copies in sync.
    """
    return load_id_set(inline_ids, txt_path, kind="express route")


# =============================================================================
# MAIN
# =============================================================================


def run(
    analysis_mode: str | None = None,
    service_area_method: str | None = None,
    gtfs_data_path: str | Path | None = None,
    demographics_shp_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    pedestrian_network_path: str | Path | None = None,
    service_ids_to_include: Sequence[str] | None = None,
    routes_to_include: Sequence[str] | None = None,
    routes_to_exclude: Sequence[str] | None = None,
    stop_ids_to_include: Sequence[str] | None = None,
    stop_ids_to_exclude: Sequence[str] | None = None,
    buffer_distance: float | None = None,
    express_origin_buffer_distance: float | None = None,
    express_origin_stop_ids: Sequence[str] | None = None,
    express_origin_stops_file: str | Path | None = None,
    isochrone_walk_time_min: float | None = None,
    walk_speed_mph: float | None = None,
    fips_filter: Sequence[str] | None = None,
    crs_epsg_code: int | None = None,
    express_route_ids: Sequence[str] | None = None,
    express_routes_file: str | Path | None = None,
    express_origin_advisory: bool | None = None,
    express_origin_jobs_field: str | None = None,
    express_origin_jobs_threshold: float | None = None,
    express_unidirectional_route_ids: Sequence[str] | None = None,
    express_unidirectional_routes_file: str | Path | None = None,
    express_bidirectional_route_ids: Sequence[str] | None = None,
    express_bidirectional_routes_file: str | Path | None = None,
    express_zero_origin_employment: bool | None = None,
    express_zero_destination_population: bool | None = None,
    express_employment_fields: Sequence[str] | None = None,
) -> None:
    """Run the catchment-area analysis.

    Unset args fall back to the CONFIGURATION block at the top of this file, so
    ``m.GTFS_DATA_PATH = ...; m.run()`` works after a plain import. The structural
    settings ``SYNTHETIC_FIELDS``, ``REQUIRED_GTFS_FILES``, and
    ``ROUTE_GEOMETRY_GTFS_FILE`` are always read from the config block.
    """
    analysis_mode = ANALYSIS_MODE if analysis_mode is None else analysis_mode
    service_area_method = (
        SERVICE_AREA_METHOD if service_area_method is None else service_area_method
    )
    gtfs_data_path = GTFS_DATA_PATH if gtfs_data_path is None else gtfs_data_path
    demographics_shp_path = (
        DEMOGRAPHICS_SHP_PATH if demographics_shp_path is None else demographics_shp_path
    )
    output_directory = OUTPUT_DIRECTORY if output_directory is None else output_directory
    pedestrian_network_path = (
        PEDESTRIAN_NETWORK_PATH if pedestrian_network_path is None else pedestrian_network_path
    )
    service_ids_to_include = list(
        SERVICE_IDS_TO_INCLUDE if service_ids_to_include is None else service_ids_to_include
    )
    routes_to_include = list(ROUTES_TO_INCLUDE if routes_to_include is None else routes_to_include)
    routes_to_exclude = list(ROUTES_TO_EXCLUDE if routes_to_exclude is None else routes_to_exclude)
    stop_ids_to_include = list(
        STOP_IDS_TO_INCLUDE if stop_ids_to_include is None else stop_ids_to_include
    )
    stop_ids_to_exclude = list(
        STOP_IDS_TO_EXCLUDE if stop_ids_to_exclude is None else stop_ids_to_exclude
    )
    buffer_distance = BUFFER_DISTANCE if buffer_distance is None else buffer_distance
    express_origin_buffer_distance = (
        EXPRESS_ORIGIN_BUFFER_DISTANCE
        if express_origin_buffer_distance is None
        else express_origin_buffer_distance
    )
    express_origin_stop_ids = list(
        EXPRESS_ORIGIN_STOP_IDS if express_origin_stop_ids is None else express_origin_stop_ids
    )
    express_origin_stops_file = (
        EXPRESS_ORIGIN_STOPS_FILE
        if express_origin_stops_file is None
        else express_origin_stops_file
    )
    isochrone_walk_time_min = (
        ISOCHRONE_WALK_TIME_MIN if isochrone_walk_time_min is None else isochrone_walk_time_min
    )
    walk_speed_mph = WALK_SPEED_MPH if walk_speed_mph is None else walk_speed_mph
    fips_filter = list(FIPS_FILTER if fips_filter is None else fips_filter)
    crs_epsg_code = CRS_EPSG_CODE if crs_epsg_code is None else crs_epsg_code
    express_route_ids = list(EXPRESS_ROUTE_IDS if express_route_ids is None else express_route_ids)
    express_routes_file = (
        EXPRESS_ROUTES_FILE if express_routes_file is None else express_routes_file
    )
    express_unidirectional_route_ids = list(
        EXPRESS_UNIDIRECTIONAL_ROUTE_IDS
        if express_unidirectional_route_ids is None
        else express_unidirectional_route_ids
    )
    express_unidirectional_routes_file = (
        EXPRESS_UNIDIRECTIONAL_ROUTES_FILE
        if express_unidirectional_routes_file is None
        else express_unidirectional_routes_file
    )
    express_bidirectional_route_ids = list(
        EXPRESS_BIDIRECTIONAL_ROUTE_IDS
        if express_bidirectional_route_ids is None
        else express_bidirectional_route_ids
    )
    express_bidirectional_routes_file = (
        EXPRESS_BIDIRECTIONAL_ROUTES_FILE
        if express_bidirectional_routes_file is None
        else express_bidirectional_routes_file
    )
    # Union each inline list with its optional .txt. Unidirectional and
    # bidirectional express routes both count as "express" (the service_type
    # label + wide origin buffer + advisory); the legacy direction-agnostic list
    # is folded in as bidirectional (symmetric). Only the unidirectional set
    # drives directional accounting. Origin stop ids drive the drive-access buffer.
    unidirectional_route_id_set = load_id_set(
        express_unidirectional_route_ids,
        str(express_unidirectional_routes_file) if express_unidirectional_routes_file else None,
        kind="unidirectional express route",
    )
    bidirectional_route_id_set = load_id_set(
        express_bidirectional_route_ids,
        str(express_bidirectional_routes_file) if express_bidirectional_routes_file else None,
        kind="bidirectional express route",
    )
    conflicting_express = unidirectional_route_id_set & bidirectional_route_id_set
    if conflicting_express:
        logging.warning(
            "Route id(s) %s are in both the unidirectional and bidirectional express "
            "lists; treating them as unidirectional.",
            sorted(conflicting_express),
        )
    express_route_id_set = (
        load_express_route_ids(
            express_route_ids,
            str(express_routes_file) if express_routes_file else None,
        )
        | unidirectional_route_id_set
        | bidirectional_route_id_set
    )
    express_origin_stop_id_set = load_id_set(
        express_origin_stop_ids,
        str(express_origin_stops_file) if express_origin_stops_file else None,
        kind="express origin stop",
    )
    express_origin_advisory = (
        EXPRESS_ORIGIN_ADVISORY if express_origin_advisory is None else express_origin_advisory
    )
    express_origin_jobs_field = (
        EXPRESS_ORIGIN_JOBS_FIELD
        if express_origin_jobs_field is None
        else express_origin_jobs_field
    )
    express_origin_jobs_threshold = (
        EXPRESS_ORIGIN_JOBS_SHARE_THRESHOLD
        if express_origin_jobs_threshold is None
        else express_origin_jobs_threshold
    )
    express_zero_origin_employment = (
        EXPRESS_ZERO_ORIGIN_EMPLOYMENT
        if express_zero_origin_employment is None
        else express_zero_origin_employment
    )
    express_zero_destination_population = (
        EXPRESS_ZERO_DESTINATION_POPULATION
        if express_zero_destination_population is None
        else express_zero_destination_population
    )
    express_employment_fields = list(
        EXPRESS_EMPLOYMENT_FIELDS
        if express_employment_fields is None
        else express_employment_fields
    )

    try:
        # --------------------------------------------------------------
        # 0) VALIDATE SERVICE-AREA METHOD
        # --------------------------------------------------------------
        service_area_method = service_area_method.lower()
        valid_methods = {"stop_buffer", "route_buffer", "isochrone"}
        if service_area_method not in valid_methods:
            raise ValueError(
                f"Invalid SERVICE_AREA_METHOD: {service_area_method!r}. "
                f"Choose one of {sorted(valid_methods)}."
            )

        # --------------------------------------------------------------
        # 1) LOAD GTFS
        # --------------------------------------------------------------
        gtfs_raw = load_gtfs_data(
            str(gtfs_data_path),
            files=REQUIRED_GTFS_FILES,
            dtype=str,  # keep everything as strings
        )
        trips = gtfs_raw["trips"]
        stop_times = gtfs_raw["stop_times"]
        routes_df = gtfs_raw["routes"]
        stops_df = gtfs_raw["stops"]

        # Route geometry from shapes.txt — required for the "route_buffer"
        # method, optional otherwise. Loaded opportunistically.
        shapes_df: Optional[pd.DataFrame] = None
        shapes_path = os.path.join(str(gtfs_data_path), ROUTE_GEOMETRY_GTFS_FILE)
        if os.path.exists(shapes_path):
            shapes_df = pd.read_csv(shapes_path, dtype=str, low_memory=False)
            logging.info("Loaded %s (%d records).", ROUTE_GEOMETRY_GTFS_FILE, len(shapes_df))
        elif service_area_method == "route_buffer":
            logging.warning(
                "SERVICE_AREA_METHOD is 'route_buffer' but %s was not found in %s; "
                "the analysis will fall back to stop buffers.",
                ROUTE_GEOMETRY_GTFS_FILE,
                gtfs_data_path,
            )

        # --------------------------------------------------------------
        # 1b) PEDESTRIAN NETWORK (only for the "isochrone" method)
        # --------------------------------------------------------------
        ped_graph: Optional[nx.MultiGraph] = None
        # Walking speed expressed in projected-CRS units per second (metres/s,
        # since CRS_EPSG_CODE is assumed metric).
        walk_speed_units_per_s = walk_speed_mph * METERS_PER_MILE / 3_600.0
        if service_area_method == "isochrone":
            ped_path = Path(pedestrian_network_path)
            if not ped_path.is_file():
                raise FileNotFoundError(
                    f"Pedestrian network shapefile not found: {ped_path}. "
                    "It is required for the 'isochrone' method."
                )
            centerlines = gpd.read_file(ped_path).to_crs(epsg=crs_epsg_code)
            ped_graph, _ = build_pedestrian_time_network(
                centerlines, walk_speed=walk_speed_units_per_s
            )

        # --------------------------------------------------------------
        # 2) OPTIONAL CALENDAR FILTER
        # --------------------------------------------------------------
        if not service_ids_to_include:
            # Empty -> auto-select the full Monday–Friday service(s) straight from
            # calendar.txt, so weekday service is chosen for any feed instead of a
            # hardcoded id (service_id values differ per agency/feed).
            service_ids_to_include = [
                str(s) for s in filter_weekday_service(gtfs_raw["calendar"]).tolist()
            ]
            if service_ids_to_include:
                logging.info(
                    "Auto-selected weekday service_id(s) from calendar.txt: %s",
                    service_ids_to_include,
                )
            else:
                logging.warning("No Monday–Friday service found in calendar.txt; using all trips.")

        if service_ids_to_include:  # explicit ids or the auto-selected weekday set
            before = len(trips)
            trips = trips[trips["service_id"].isin(service_ids_to_include)]
            logging.info(
                "Applied calendar filter %s — trips: %d → %d",
                service_ids_to_include,
                before,
                len(trips),
            )
        else:
            logging.info("No calendar filter applied; using all %d trips.", len(trips))

        # --------------------------------------------------------------
        # 3) DEMOGRAPHICS LAYER
        # --------------------------------------------------------------
        demographics_path = Path(demographics_shp_path)
        if not demographics_path.is_file():
            raise FileNotFoundError(f"Demographics shapefile not found: {demographics_path}")

        demographics_gdf = gpd.read_file(demographics_path)
        demographics_gdf = apply_fips_filter(demographics_gdf, fips_filter)
        demographics_gdf = demographics_gdf.to_crs(epsg=crs_epsg_code)

        # --------------------------------------------------------------
        # 3b) EXPRESS-ORIGIN ADVISORY (optional, non-fatal)
        # --------------------------------------------------------------
        # Suggest which express-route stops look like origins, for review. It
        # must never break the real analysis, so failures are logged and ignored.
        if express_origin_advisory and express_route_id_set:
            try:
                suggest_express_origin_stops(
                    trips,
                    stop_times,
                    routes_df,
                    stops_df,
                    demographics_gdf,
                    express_route_ids=express_route_id_set,
                    buffer_distance_mi=buffer_distance,
                    jobs_field=express_origin_jobs_field,
                    threshold=express_origin_jobs_threshold,
                    output_dir=str(output_directory),
                )
            except Exception:  # noqa: BLE001 — advisory must not abort the analysis
                logging.warning("Express-origin advisory failed; continuing.", exc_info=True)

        # --------------------------------------------------------------
        # 4) ANALYSIS DISPATCH
        # --------------------------------------------------------------
        mode = analysis_mode.lower()
        # Shared positional arguments for all three analysis modes. Kwargs are
        # passed literally (not via a dict) so the static type checker keeps each
        # parameter's type. Directional accounting is a route-mode concept, so
        # only that analyzer additionally receives those settings.
        common_args = (
            trips,
            stop_times,
            routes_df,
            stops_df,
            demographics_gdf,
            routes_to_include,
            routes_to_exclude,
            stop_ids_to_include,
            stop_ids_to_exclude,
            buffer_distance,
            express_origin_buffer_distance,
            sorted(express_origin_stop_id_set),
            str(output_directory),
            SYNTHETIC_FIELDS,
        )
        if mode == "route":
            do_route_by_route_analysis(
                *common_args,
                service_area_method=service_area_method,
                shapes_df=shapes_df,
                ped_graph=ped_graph,
                walk_time_min=isochrone_walk_time_min,
                walk_speed_units_per_s=walk_speed_units_per_s,
                express_route_ids=express_route_id_set,
                unidirectional_route_ids=unidirectional_route_id_set,
                employment_fields=express_employment_fields,
                zero_origin_employment=express_zero_origin_employment,
                zero_destination_population=express_zero_destination_population,
            )
        elif mode in ("network", "stop"):
            analyzer = do_network_analysis if mode == "network" else do_stop_by_stop_analysis
            analyzer(
                *common_args,
                service_area_method=service_area_method,
                shapes_df=shapes_df,
                ped_graph=ped_graph,
                walk_time_min=isochrone_walk_time_min,
                walk_speed_units_per_s=walk_speed_units_per_s,
                express_route_ids=express_route_id_set,
            )
        else:
            raise ValueError(f"Invalid ANALYSIS_MODE: {analysis_mode}")

        logging.info("\nAnalysis completed successfully.")

    except Exception:
        # Re-raise after logging: a swallowed error here exits 0 and looks
        # identical to "legitimately produced nothing" to the prep_features_public
        # orchestrator, which is what hid a missing-demographics-input chain.
        logging.error("Analysis terminated due to an error", exc_info=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block.

    The structural ``SYNTHETIC_FIELDS`` list stays in the config block.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate demographics within a GTFS-derived service area. Defaults come "
            "from the CONFIGURATION block at the top of this file; SYNTHETIC_FIELDS "
            "stays in the config block."
        )
    )
    parser.add_argument(
        "--analysis-mode",
        default=ANALYSIS_MODE,
        choices=("network", "route", "stop"),
        help="How results are grouped.",
    )
    parser.add_argument(
        "--service-area-method",
        default=SERVICE_AREA_METHOD,
        choices=("stop_buffer", "route_buffer", "isochrone"),
        help="How each catchment polygon is built.",
    )
    parser.add_argument("--gtfs-path", default=GTFS_DATA_PATH, help="GTFS data folder.")
    parser.add_argument(
        "--demographics-shp", default=DEMOGRAPHICS_SHP_PATH, help="Demographics shapefile."
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIRECTORY, help="Output directory.")
    parser.add_argument(
        "--pedestrian-network",
        default=PEDESTRIAN_NETWORK_PATH,
        help="Pedestrian centerline shapefile (isochrone method only).",
    )
    parser.add_argument(
        "--service-ids",
        nargs="*",
        default=SERVICE_IDS_TO_INCLUDE,
        metavar="SERVICE_ID",
        help="Calendar service_id values to keep (default: all).",
    )
    parser.add_argument(
        "--routes-include",
        nargs="*",
        default=ROUTES_TO_INCLUDE,
        metavar="ROUTE_ID",
        help="Only these routes (default: all).",
    )
    parser.add_argument(
        "--routes-exclude",
        nargs="*",
        default=ROUTES_TO_EXCLUDE,
        metavar="ROUTE_ID",
        help="Drop these routes (default: none).",
    )
    parser.add_argument(
        "--stops-include",
        nargs="*",
        default=STOP_IDS_TO_INCLUDE,
        metavar="STOP_ID",
        help="Only these stops (default: all).",
    )
    parser.add_argument(
        "--stops-exclude",
        nargs="*",
        default=STOP_IDS_TO_EXCLUDE,
        metavar="STOP_ID",
        help="Drop these stops (default: none).",
    )
    parser.add_argument(
        "--buffer-distance",
        type=float,
        default=BUFFER_DISTANCE,
        help="Standard buffer distance in miles.",
    )
    parser.add_argument(
        "--express-origin-buffer-distance",
        type=float,
        default=EXPRESS_ORIGIN_BUFFER_DISTANCE,
        help="Drive-access buffer distance in miles for express origin stops.",
    )
    parser.add_argument(
        "--express-origin-stops",
        nargs="*",
        default=EXPRESS_ORIGIN_STOP_IDS,
        metavar="STOP_ID",
        help="Express origin (park-and-ride) stop_ids that use the drive-access "
        "buffer (stop_buffer method only).",
    )
    parser.add_argument(
        "--express-origin-stops-file",
        default=EXPRESS_ORIGIN_STOPS_FILE,
        help="Path to a text file of express origin stop_ids, one per line ('#' comments).",
    )
    parser.add_argument(
        "--express-origin-advisory",
        action=argparse.BooleanOptionalAction,
        default=EXPRESS_ORIGIN_ADVISORY,
        help="Suggest likely express origin stops from nearby jobs (when express routes are set).",
    )
    parser.add_argument(
        "--express-origin-jobs-field",
        default=EXPRESS_ORIGIN_JOBS_FIELD,
        help="Demographics column holding job counts for the express-origin advisory.",
    )
    parser.add_argument(
        "--express-origin-jobs-threshold",
        type=float,
        default=EXPRESS_ORIGIN_JOBS_SHARE_THRESHOLD,
        help="Flag a stop when its nearby jobs fall below this fraction of its route's max.",
    )
    parser.add_argument(
        "--isochrone-walk-time",
        type=float,
        default=ISOCHRONE_WALK_TIME_MIN,
        help="Walk-time budget in minutes (isochrone method).",
    )
    parser.add_argument(
        "--walk-speed-mph",
        type=float,
        default=WALK_SPEED_MPH,
        help="Assumed pedestrian walking speed.",
    )
    parser.add_argument(
        "--fips",
        nargs="*",
        default=FIPS_FILTER,
        metavar="FIPS",
        help="Demographics FIPS codes to keep (default: all).",
    )
    parser.add_argument(
        "--crs-epsg",
        type=int,
        default=CRS_EPSG_CODE,
        help="Projected (metric) EPSG code for area calculations.",
    )
    parser.add_argument(
        "--express-routes",
        nargs="*",
        default=EXPRESS_ROUTE_IDS,
        metavar="ROUTE_ID",
        help="route_id values to label 'express' in route mode (default: none).",
    )
    parser.add_argument(
        "--express-routes-file",
        default=EXPRESS_ROUTES_FILE,
        help="Path to a text file of express route_ids, one per line ('#' comments). "
        "Direction-agnostic; treated as bidirectional (symmetric) accounting.",
    )
    parser.add_argument(
        "--express-unidirectional-routes",
        nargs="*",
        default=EXPRESS_UNIDIRECTIONAL_ROUTE_IDS,
        metavar="ROUTE_ID",
        help="route_id values for unidirectional (commuter) express routes, which get "
        "directional accounting in route mode (default: none).",
    )
    parser.add_argument(
        "--express-unidirectional-routes-file",
        default=EXPRESS_UNIDIRECTIONAL_ROUTES_FILE,
        help="Path to a text file of unidirectional express route_ids, one per line.",
    )
    parser.add_argument(
        "--express-bidirectional-routes",
        nargs="*",
        default=EXPRESS_BIDIRECTIONAL_ROUTE_IDS,
        metavar="ROUTE_ID",
        help="route_id values for bidirectional express routes (symmetric accounting; "
        "default: none).",
    )
    parser.add_argument(
        "--express-bidirectional-routes-file",
        default=EXPRESS_BIDIRECTIONAL_ROUTES_FILE,
        help="Path to a text file of bidirectional express route_ids, one per line.",
    )
    parser.add_argument(
        "--express-zero-origin-employment",
        action=argparse.BooleanOptionalAction,
        default=EXPRESS_ZERO_ORIGIN_EMPLOYMENT,
        help="Zero employment counted at the origin (park-and-ride) end of "
        "unidirectional express routes.",
    )
    parser.add_argument(
        "--express-zero-destination-population",
        action=argparse.BooleanOptionalAction,
        default=EXPRESS_ZERO_DESTINATION_POPULATION,
        help="Zero population counted at the destination (CBD) end of unidirectional "
        "express routes.",
    )
    parser.add_argument(
        "--express-employment-fields",
        nargs="*",
        default=EXPRESS_EMPLOYMENT_FIELDS,
        metavar="FIELD",
        help="SYNTHETIC_FIELDS that are employment counts, attributed to the destination "
        "end of unidirectional express routes.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. Defaults fall back to the CONFIGURATION block."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if (
        str(args.gtfs_path) == GTFS_DATA_PATH
        or str(args.demographics_shp) == DEMOGRAPHICS_SHP_PATH
        or str(args.output_dir) == OUTPUT_DIRECTORY
    ):
        logging.warning(
            "GTFS_DATA_PATH, DEMOGRAPHICS_SHP_PATH, and/or OUTPUT_DIRECTORY are still set to "
            "their default placeholder values. Update the CONFIGURATION section (or pass "
            "--gtfs-path/--demographics-shp/--output-dir) before running."
        )
        return
    try:
        run(
            analysis_mode=args.analysis_mode,
            service_area_method=args.service_area_method,
            gtfs_data_path=args.gtfs_path,
            demographics_shp_path=args.demographics_shp,
            output_directory=args.output_dir,
            pedestrian_network_path=args.pedestrian_network,
            service_ids_to_include=args.service_ids,
            routes_to_include=args.routes_include,
            routes_to_exclude=args.routes_exclude,
            stop_ids_to_include=args.stops_include,
            stop_ids_to_exclude=args.stops_exclude,
            buffer_distance=args.buffer_distance,
            express_origin_buffer_distance=args.express_origin_buffer_distance,
            express_origin_stop_ids=args.express_origin_stops,
            express_origin_stops_file=args.express_origin_stops_file,
            isochrone_walk_time_min=args.isochrone_walk_time,
            walk_speed_mph=args.walk_speed_mph,
            fips_filter=args.fips,
            crs_epsg_code=args.crs_epsg,
            express_route_ids=args.express_routes,
            express_routes_file=args.express_routes_file,
            express_origin_advisory=args.express_origin_advisory,
            express_origin_jobs_field=args.express_origin_jobs_field,
            express_origin_jobs_threshold=args.express_origin_jobs_threshold,
            express_unidirectional_route_ids=args.express_unidirectional_routes,
            express_unidirectional_routes_file=args.express_unidirectional_routes_file,
            express_bidirectional_route_ids=args.express_bidirectional_routes,
            express_bidirectional_routes_file=args.express_bidirectional_routes_file,
            express_zero_origin_employment=args.express_zero_origin_employment,
            express_zero_destination_population=args.express_zero_destination_population,
            express_employment_fields=args.express_employment_fields,
        )
    except Exception:
        # run() already logged the traceback; exit non-zero so the orchestrator
        # records a real failure instead of "produced no tables".
        sys.exit(1)


def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":  # pragma: no cover
    # In a notebook (pasted cell or %run), use the CONFIGURATION block instead
    # of argparse, which would otherwise try to parse the kernel's own argv.
    if _in_ipython():
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        run()
    else:
        main()
