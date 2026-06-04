"""Reusable builders for travel-time networks used in accessibility analysis.

This module centralizes the two graph builders that downstream service- and
coverage-analysis scripts depend on:

    * :func:`build_pedestrian_time_network` — a walking graph built from a
      centerline (sidewalk/road) layer, with each edge weighted by walking
      travel time.
    * :func:`build_gtfs_transit_time_network` — an in-vehicle transit graph
      built from GTFS ``stop_times``, with each edge weighted by the scheduled
      ride time between consecutive stops.

Both builders return :mod:`networkx` graphs whose edges carry a ``time_s``
attribute (travel time in seconds), so callers can route uniformly with
``networkx.shortest_path(..., weight="time_s")`` regardless of mode.

Notes:
    These builders intentionally model only the in-mode travel network. Wait
    time, transfers, and pedestrian access/egress connectors are the concern of
    the consuming script (for example, an isochrone builder), which knows the
    departure assumptions and how the modes should be stitched together.
"""

from __future__ import annotations

import logging
from typing import Tuple

import geopandas as gpd
import networkx as nx
import pandas as pd

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

FT_PER_MILE: float = 5_280.0
SECONDS_PER_HOUR: float = 3_600.0

# Default walking speed (~3 mph) expressed in feet per second, matching the
# project's customary US-foot projected CRS (e.g. EPSG:6447). Override
# ``walk_speed`` when the centerline CRS uses different linear units.
DEFAULT_WALK_SPEED_MPH: float = 3.0
DEFAULT_WALK_SPEED_FT_PER_S: float = DEFAULT_WALK_SPEED_MPH * FT_PER_MILE / SECONDS_PER_HOUR

# Endpoint merging: snap segment endpoints to a grid so near-coincident nodes
# (e.g. centerlines that don't share an exact vertex) connect into one node.
DEFAULT_NODE_GRID_FT: float = 5.0

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

NodeKey = Tuple[float, float]  # quantized (x, y)
EdgeID = int

# -----------------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------------


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


def _parse_gtfs_time(value: object) -> float:
    """Parse a GTFS ``HH:MM:SS`` time string into seconds past midnight.

    GTFS permits hour values of 24 or greater for trips that run past midnight,
    so this does not wrap at 24:00:00.

    Args:
        value: A GTFS time string (e.g. ``"25:14:00"``). Blank or missing
            values yield ``nan``.

    Returns:
        Seconds past midnight as a float, or ``float("nan")`` if the value is
        missing or malformed.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    parts = text.split(":")
    if len(parts) != 3:
        return float("nan")
    try:
        hours, minutes, seconds = (int(p) for p in parts)
    except ValueError:
        return float("nan")
    return float(hours * 3_600 + minutes * 60 + seconds)


# -----------------------------------------------------------------------------
# PEDESTRIAN TIME NETWORK
# -----------------------------------------------------------------------------


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
# GTFS TRANSIT TIME NETWORK
# -----------------------------------------------------------------------------


def build_gtfs_transit_time_network(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame | None = None,
) -> nx.MultiDiGraph:
    """Build an in-vehicle transit travel-time graph from GTFS ``stop_times``.

    Stops become nodes and each consecutive pair of stops within a trip becomes
    a directed edge weighted by the scheduled ride time between them. Edges carry
    ``time_s`` (arrival at the next stop minus departure from the current stop),
    plus ``trip_id`` and, when ``trips`` is supplied, ``route_id`` for filtering
    or attribution. Because a :class:`networkx.MultiDiGraph` is returned, every
    trip's segment is preserved; routing with ``weight="time_s"`` naturally uses
    the fastest parallel edge between any two stops.

    Wait time at the origin, transfers between trips, and pedestrian
    access/egress are **not** modeled here — they depend on departure
    assumptions the caller owns.

    Args:
        stop_times: GTFS ``stop_times`` with columns ``trip_id``,
            ``stop_id``, ``stop_sequence``, ``arrival_time``, and
            ``departure_time``.
        trips: Optional GTFS ``trips`` table; when provided (with ``trip_id``
            and ``route_id``), each edge is tagged with its ``route_id``.

    Returns:
        A directed :class:`networkx.MultiDiGraph`. Each edge has ``time_s``,
        ``trip_id``, and (if available) ``route_id``.

    Raises:
        ValueError: If required columns are missing from ``stop_times``.
    """
    required = {"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"}
    missing = required - set(stop_times.columns)
    if missing:
        raise ValueError(f"stop_times is missing required columns: {sorted(missing)}")

    route_by_trip: dict[str, str] = {}
    if trips is not None and {"trip_id", "route_id"} <= set(trips.columns):
        route_by_trip = dict(zip(trips["trip_id"].astype(str), trips["route_id"].astype(str)))

    st = stop_times[
        ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]
    ].copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st["_arr_s"] = st["arrival_time"].map(_parse_gtfs_time)
    st["_dep_s"] = st["departure_time"].map(_parse_gtfs_time)
    st = st.sort_values(["trip_id", "stop_sequence"])

    graph = nx.MultiDiGraph()

    n_edges = 0
    n_skipped = 0
    for trip_id, group in st.groupby("trip_id", sort=False):
        trip_key = str(trip_id)
        route_id = route_by_trip.get(trip_key)

        stops = group["stop_id"].astype(str).tolist()
        dep = group["_dep_s"].tolist()
        arr = group["_arr_s"].tolist()

        for i in range(len(stops) - 1):
            time_s = arr[i + 1] - dep[i]
            if pd.isna(time_s) or time_s < 0:
                n_skipped += 1
                continue

            attrs: dict[str, object] = {"time_s": float(time_s), "trip_id": trip_key}
            if route_id is not None:
                attrs["route_id"] = route_id
            graph.add_edge(stops[i], stops[i + 1], **attrs)
            n_edges += 1

    logging.info(
        "Built GTFS transit time network: %d stops, %d edges (%d segments skipped "
        "for missing/negative times).",
        graph.number_of_nodes(),
        n_edges,
        n_skipped,
    )
    return graph


__all__ = [
    "build_gtfs_transit_time_network",
    "build_pedestrian_time_network",
    "quantize_node",
]
