"""Compute the routes each target route can transfer to across one or more GTFS feeds.

For every *target* route the user designates, this script finds the other routes a
rider can transfer to, subject to a **space buffer** (how far a rider will walk
between an alighting stop and a boarding stop) and an optional **time buffer**
(the connecting trip must depart after the feeder trip arrives — accounting for
walk time — and within a maximum wait).

Multiple feeds are supported. All feeds are pooled into one network, so a target
route in one agency's feed can transfer to a route in another agency's feed.
Targets are chosen with ``TARGET_ROUTE_TOKENS`` (matched against
``route_short_name`` or ``route_id``) and may optionally be restricted to a
subset of feeds with ``TARGET_FEED_LABELS``; every route in every feed is
eligible as a *connector*.

Transfers are directional: an A -> B transfer means a rider can ride route A,
get off, and board route B. The time check uses A's arrival times at the
alighting stop and B's departure times at the (possibly different) nearby
boarding stop.

By default only regular weekday service is considered: ``DAY_OF_WEEK`` is
``"weekday"``, which keeps trips whose calendar.txt service runs on at least one
of Monday–Friday. Weekend-only service is excluded, and because
``calendar_dates.txt`` exceptions are never consulted, holiday-exception service
never enters the pool either. Set a single day name to filter to that day, or
``None`` / ``--day all`` to pool every trip regardless of service day (which can
pair trips that never run on the same day — inflating counts).

Inputs:
    - One or more GTFS feed folders, each with stops.txt, routes.txt, trips.txt,
      and stop_times.txt. calendar.txt is used when a day filter is set.

Outputs:
    - A summary CSV keyed on ``route_id``: one row per target route with a
      transfer count and the comma-separated list of routes it can transfer to.
    - An optional detail CSV: one row per (target route, connector route) pair
      with the number of qualifying stop pairs, nearest walk distance, and the
      shortest feasible wait observed.

Defaults come from the CONFIGURATION block; ``--gtfs-dirs`` / ``--output-dir`` /
``--day`` / ``--log-level`` override them, so the script can run standalone or
under the prep_features_public.py orchestrator.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

# =============================================================================
# CONFIGURATION
# =============================================================================

# One or more GTFS feed folders. Each must contain stops.txt, routes.txt,
# trips.txt, and stop_times.txt. Use raw strings (r"") for Windows paths.
GTFS_FEEDS: list[str] = [
    r"Path\To\Your\GTFS_Folder",
    # r"Path\To\Another_Agency_GTFS_Folder",
]

# Optional human-readable labels for each feed, parallel to GTFS_FEEDS. Used to
# tag routes/stops in the output. Leave as None to label each feed by its folder
# name (duplicates are de-duplicated automatically).
FEED_LABELS: Optional[list[str]] = None

# Folder where the output CSV(s) are written.
OUTPUT_DIR: Path = Path(r"Path\To\Output_Folder")

# Routes we want transfers FOR. Each token is matched against route_short_name OR
# route_id, across every feed. Leave empty to treat every route as a target.
TARGET_ROUTE_TOKENS: set[str] = set()  # e.g. {"10", "20"}; empty = every route is a target

# Optionally restrict which feeds' routes can be targets (by label). None = any
# feed. Connectors are always drawn from all feeds regardless of this setting.
TARGET_FEED_LABELS: Optional[set[str]] = None

# --- Space buffer ---------------------------------------------------------
# Maximum walking distance between an alighting stop and a boarding stop for the
# pair to count as a transfer opportunity. Shared stops have distance 0.
MAX_TRANSFER_DISTANCE: float = 0.25
MAX_TRANSFER_DISTANCE_UNIT = "miles"  # "miles" | "feet" | "meters"

# Projected CRS (in METERS) used for distance math. Pick one appropriate for your
# study area; the default is NAD83 / Conterminous US Albers, fine for short
# walking distances anywhere in the US. For non-US feeds, set a local metric CRS.
PROJECTED_CRS = "EPSG:5070"

# --- Time buffer ----------------------------------------------------------
# When True, a connector route only counts if at least one scheduled connection
# is feasible: feeder arrival + walk time <= connector departure <= feeder
# arrival + MAX_TRANSFER_WAIT_MINUTES. When False, any spatial co-location of the
# two routes counts (a pure proximity transfer map).
ENABLE_TIME_CHECK = True

# Assumed walking speed used to convert the stop-to-stop distance into a minimum
# transfer time. 3.0 mph is a common planning value.
WALK_SPEED_MPH: float = 3.0

# Extra slack (minutes) added on top of the computed walk time before a connector
# departure can be caught (e.g., time to find the stop). 0 = walk time only.
MIN_TRANSFER_BUFFER_MINUTES: float = 0.0

# Longest a rider will wait at the connector stop, in minutes.
MAX_TRANSFER_WAIT_MINUTES: float = 30.0

# Service-day filter. "weekday" (the default) keeps trips whose calendar.txt
# service runs on at least one of Monday-Friday — regular weekday service, with
# weekend-only service excluded and holiday exceptions naturally absent because
# calendar_dates.txt is never consulted. A single day name (e.g., "monday")
# keeps only that day's service. None keeps all trips regardless of service day.
# Feeds without calendar.txt fall back to all trips with a warning.
DAY_OF_WEEK: Optional[str] = "weekday"

# --- Output ---------------------------------------------------------------
SUMMARY_FILENAME = "route_transfers_summary.csv"
DETAIL_FILENAME = "route_transfers_detail.csv"
WRITE_DETAIL = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

_PLACEHOLDER_FEED = r"Path\To\Your\GTFS_Folder"
_PLACEHOLDER_OUTPUT = Path(r"Path\To\Output_Folder")
_REQUIRED_FILES = ("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
_DAY_COLUMNS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

# =============================================================================
# FUNCTIONS
# =============================================================================


def parse_gtfs_time(value: object) -> Optional[float]:
    """Convert an HH:MM:SS GTFS time into seconds past midnight.

    Hours may exceed 24 (service after midnight), and those values are preserved.

    Args:
        value: A time string such as ``"07:35:00"`` or ``"25:10:00"``.

    Returns:
        Seconds past midnight as a float, or None when the value is missing or
        malformed.
    """
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    except (TypeError, ValueError):
        return None
    return hours * 3600 + minutes * 60 + seconds


def distance_to_meters(distance: float, unit: str) -> float:
    """Convert a distance to meters.

    Args:
        distance: The numeric distance value.
        unit: One of ``"miles"``, ``"feet"``, or ``"meters"`` (case-insensitive).

    Returns:
        The distance expressed in meters.

    Raises:
        ValueError: If ``unit`` is not recognized.
    """
    factors = {"miles": 1609.34, "feet": 0.3048, "meters": 1.0}
    key = unit.lower()
    if key not in factors:
        raise ValueError(f"Unknown distance unit '{unit}'. Use miles, feet, or meters.")
    return distance * factors[key]


def service_ids_active_on_day(calendar_df: Optional[pd.DataFrame], day: str) -> Optional[set[str]]:
    """Return the set of service_ids whose regular service runs on the given day.

    Only ``calendar.txt`` weekly flags are consulted; date-specific exceptions in
    ``calendar_dates.txt`` are not mapped to weekdays, so holiday-exception
    service never qualifies. When the calendar is unavailable, None is returned
    to signal "do not filter".

    Args:
        calendar_df: Parsed calendar.txt, or None if the feed has none.
        day: Lowercase day name (e.g. ``"monday"``), or ``"weekday"`` to keep
            every service_id that runs on at least one of Monday-Friday.

    Returns:
        A set of service_id strings active on ``day``, or None when no calendar is
        available to filter on.
    """
    key = day.lower()
    if key != "weekday" and key not in _DAY_COLUMNS:
        raise ValueError(f"DAY_OF_WEEK must be 'weekday' or one of {_DAY_COLUMNS}, got '{day}'.")
    if calendar_df is None or calendar_df.empty:
        return None

    day_cols = _DAY_COLUMNS[:5] if key == "weekday" else (key,)
    present = [c for c in day_cols if c in calendar_df.columns]
    if not present:
        return None

    mask = pd.Series(False, index=calendar_df.index)
    for col in present:
        mask |= calendar_df[col].astype(str) == "1"
    active = calendar_df.loc[mask, "service_id"]
    return {str(s) for s in active}


def has_timed_connection(
    arrivals: np.ndarray,
    departures: np.ndarray,
    min_wait_seconds: float,
    max_wait_seconds: float,
) -> tuple[bool, Optional[float]]:
    """Check whether any feeder arrival can catch a connector departure in time.

    A connection is feasible when, for some feeder arrival ``a`` and connector
    departure ``d``, ``min_wait_seconds <= d - a <= max_wait_seconds``.

    Args:
        arrivals: Sorted feeder arrival times (seconds past midnight).
        departures: Sorted connector departure times (seconds past midnight).
        min_wait_seconds: Earliest the rider can board after arriving (walk time
            plus any buffer).
        max_wait_seconds: Longest the rider will wait.

    Returns:
        A tuple ``(feasible, shortest_wait_seconds)``. ``shortest_wait_seconds`` is
        the smallest qualifying ``d - a`` across all pairs, or None when infeasible.
    """
    if arrivals.size == 0 or departures.size == 0:
        return False, None

    best_wait: Optional[float] = None
    for arrival in arrivals:
        earliest = arrival + min_wait_seconds
        latest = arrival + max_wait_seconds
        # First departure at or after the earliest boardable time.
        idx = int(np.searchsorted(departures, earliest, side="left"))
        if idx < departures.size and departures[idx] <= latest:
            wait = float(departures[idx] - arrival)
            if best_wait is None or wait < best_wait:
                best_wait = wait
    return best_wait is not None, best_wait


def _unique_labels(feeds: list[str], labels: Optional[list[str]]) -> list[str]:
    """Derive a unique label for each feed folder.

    Args:
        feeds: Feed folder paths.
        labels: Optional explicit labels parallel to ``feeds``.

    Returns:
        A list of unique, non-empty labels parallel to ``feeds``.
    """
    if labels is not None:
        if len(labels) != len(feeds):
            raise ValueError("FEED_LABELS must have the same length as GTFS_FEEDS.")
        raw = [str(name).strip() for name in labels]
    else:
        raw = [Path(path).name or f"feed{i + 1}" for i, path in enumerate(feeds)]

    seen: dict[str, int] = {}
    out: list[str] = []
    for name in raw:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _read_feed(folder: str, label: str) -> dict[str, pd.DataFrame]:
    """Load the required GTFS tables for one feed and tag them with the label.

    IDs are namespaced as ``"<label>::<id>"`` so that ids never collide across
    feeds when the network is pooled.

    Args:
        folder: Path to the GTFS feed folder.
        label: Unique feed label.

    Returns:
        A mapping with namespaced ``stops``, ``routes``, ``trips``,
        ``stop_times`` frames and an optional ``calendar`` frame.

    Raises:
        FileNotFoundError: If a required GTFS file is missing.
    """
    for filename in _REQUIRED_FILES:
        if not os.path.exists(os.path.join(folder, filename)):
            raise FileNotFoundError(f"Feed '{label}': missing {filename} in {folder}")

    def _read(name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(folder, name), dtype=str, low_memory=False)

    def _ns(series: pd.Series) -> pd.Series:
        return label + "::" + series.astype(str)

    stops = _read("stops.txt")
    routes = _read("routes.txt")
    trips = _read("trips.txt")
    stop_times = _read("stop_times.txt")

    stops["feed"] = label
    stops["gstop_id"] = _ns(stops["stop_id"])

    routes["feed"] = label
    routes["groute_id"] = _ns(routes["route_id"])

    trips["feed"] = label
    trips["groute_id"] = _ns(trips["route_id"])
    trips["gtrip_id"] = _ns(trips["trip_id"])

    stop_times["feed"] = label
    stop_times["gtrip_id"] = _ns(stop_times["trip_id"])
    stop_times["gstop_id"] = _ns(stop_times["stop_id"])

    calendar_path = os.path.join(folder, "calendar.txt")
    calendar: Optional[pd.DataFrame] = None
    if os.path.exists(calendar_path):
        try:
            calendar = pd.read_csv(calendar_path, dtype=str)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logging.warning("Feed '%s': could not read calendar.txt (%s).", label, exc)

    logging.info(
        "Feed '%s': %d stops, %d routes, %d trips, %d stop_times.",
        label,
        len(stops),
        len(routes),
        len(trips),
        len(stop_times),
    )
    return {
        "stops": stops,
        "routes": routes,
        "trips": trips,
        "stop_times": stop_times,
        "calendar": calendar,
    }


def _build_route_table(routes_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-feed route tables and add display labels and target flags.

    Args:
        routes_frames: One namespaced routes frame per feed.

    Returns:
        A routes frame indexed by ``groute_id`` with ``route_label`` (short name,
        else long name, else route_id) and an ``is_target`` flag.
    """
    routes = pd.concat(routes_frames, ignore_index=True)
    for col in ("route_short_name", "route_long_name"):
        if col not in routes.columns:
            routes[col] = ""
    routes["route_short_name"] = routes["route_short_name"].fillna("").astype(str).str.strip()
    routes["route_long_name"] = routes["route_long_name"].fillna("").astype(str).str.strip()

    label = routes["route_short_name"].where(routes["route_short_name"] != "")
    label = label.fillna(routes["route_long_name"].where(routes["route_long_name"] != ""))
    routes["route_label"] = label.fillna(routes["route_id"].astype(str))

    if TARGET_ROUTE_TOKENS:
        tokens = {str(t) for t in TARGET_ROUTE_TOKENS}
        is_target = routes["route_short_name"].isin(tokens) | routes["route_id"].astype(str).isin(
            tokens
        )
    else:
        is_target = pd.Series(True, index=routes.index)

    if TARGET_FEED_LABELS:
        is_target = is_target & routes["feed"].isin({str(f) for f in TARGET_FEED_LABELS})

    routes["is_target"] = is_target
    return routes.set_index("groute_id")


def _project_stops(stops: pd.DataFrame) -> gpd.GeoDataFrame:
    """Project unique stops to ``PROJECTED_CRS`` and attach metric x/y columns.

    Args:
        stops: Combined stops frame with ``gstop_id``, ``stop_lat``, ``stop_lon``.

    Returns:
        A GeoDataFrame of unique stops with numeric ``x`` and ``y`` (meters).
    """
    unique = stops.drop_duplicates("gstop_id").copy()
    unique["stop_lat"] = pd.to_numeric(unique["stop_lat"], errors="coerce")
    unique["stop_lon"] = pd.to_numeric(unique["stop_lon"], errors="coerce")
    before = len(unique)
    unique = unique.dropna(subset=["stop_lat", "stop_lon"])
    if len(unique) < before:
        logging.warning("Dropped %d stop(s) with missing coordinates.", before - len(unique))

    gdf = gpd.GeoDataFrame(
        unique,
        geometry=[Point(lon, lat) for lon, lat in zip(unique["stop_lon"], unique["stop_lat"])],
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def _build_stop_events(
    stop_times: pd.DataFrame, trips: pd.DataFrame, keep_service_ids: dict[str, Optional[set[str]]]
) -> pd.DataFrame:
    """Join stop_times to trips and resolve feeder/connector event times.

    Args:
        stop_times: Combined namespaced stop_times frame.
        trips: Combined namespaced trips frame.
        keep_service_ids: Per-feed set of service_ids to keep (day filter), or
            None for a feed to keep everything.

    Returns:
        A frame with ``gstop_id``, ``groute_id``, ``feed``, ``arrival_sec``
        (feeder time), and ``departure_sec`` (connector time).
    """
    trip_cols = trips[["gtrip_id", "groute_id", "feed", "service_id"]]
    events = stop_times.merge(trip_cols, on="gtrip_id", how="inner", suffixes=("", "_trip"))
    events["feed"] = events["feed_trip"] if "feed_trip" in events.columns else events["feed"]

    # Apply the optional day-of-week service filter per feed.
    if any(ids is not None for ids in keep_service_ids.values()):
        mask = pd.Series(True, index=events.index)
        for feed_label, ids in keep_service_ids.items():
            if ids is None:
                continue
            feed_rows = events["feed"] == feed_label
            mask &= ~feed_rows | events["service_id"].astype(str).isin(ids)
        kept = int(mask.sum())
        logging.info("Day-of-week filter kept %d/%d stop events.", kept, len(events))
        events = events[mask]

    arrival = events["arrival_time"].map(parse_gtfs_time)
    departure = events["departure_time"].map(parse_gtfs_time)
    # Feeders use arrival (fall back to departure); connectors use departure
    # (fall back to arrival) so single-time rows still participate.
    events["arrival_sec"] = arrival.fillna(departure)
    events["departure_sec"] = departure.fillna(arrival)
    return events[["gstop_id", "groute_id", "feed", "arrival_sec", "departure_sec"]]


_StopRouteArrays = dict[tuple[str, str], np.ndarray]


def _index_events(
    events: pd.DataFrame,
) -> tuple[dict[str, set[str]], _StopRouteArrays, _StopRouteArrays]:
    """Build lookups of routes-per-stop and sorted time arrays per (stop, route).

    Args:
        events: Output of :func:`_build_stop_events`.

    Returns:
        A tuple ``(serves, arrivals, departures)`` where ``serves`` maps a
        ``gstop_id`` to the set of routes serving it, ``arrivals`` maps
        ``(gstop_id, groute_id)`` to a sorted array of feeder arrival seconds, and
        ``departures`` maps it to a sorted array of connector departure seconds.
    """
    serves: dict[str, set[str]] = {}
    for gstop_id, groute_id in zip(events["gstop_id"], events["groute_id"]):
        serves.setdefault(gstop_id, set()).add(groute_id)

    arrivals: dict[tuple[str, str], np.ndarray] = {}
    departures: dict[tuple[str, str], np.ndarray] = {}
    for (gstop_id, groute_id), group in events.groupby(["gstop_id", "groute_id"], sort=False):
        arr = group["arrival_sec"].dropna().to_numpy(dtype=float)
        dep = group["departure_sec"].dropna().to_numpy(dtype=float)
        arr.sort()
        dep.sort()
        arrivals[(gstop_id, groute_id)] = arr
        departures[(gstop_id, groute_id)] = dep
    return serves, arrivals, departures


def compute_transfers(
    stops_gdf: gpd.GeoDataFrame,
    events: pd.DataFrame,
    routes: pd.DataFrame,
    radius_m: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute feasible transfers from each target route to connector routes.

    Args:
        stops_gdf: Projected unique stops with ``x``/``y`` in meters.
        events: Stop events from :func:`_build_stop_events`.
        routes: Route table indexed by ``groute_id`` with ``is_target``.
        radius_m: Space buffer in meters.

    Returns:
        A nested mapping ``target_groute_id -> connector_groute_id -> stats``,
        where stats holds ``stop_pairs``, ``min_distance_m``, and
        ``min_wait_seconds`` (NaN when the time check is disabled).
    """
    serves, arrivals, departures = _index_events(events)
    target_ids = set(routes.index[routes["is_target"]])

    coords = np.column_stack([stops_gdf["x"].to_numpy(), stops_gdf["y"].to_numpy()])
    gstop_ids = stops_gdf["gstop_id"].to_numpy()
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius_m, p=2.0)

    walk_mps = WALK_SPEED_MPH * 1609.34 / 3600.0
    buffer_sec = MIN_TRANSFER_BUFFER_MINUTES * 60.0
    max_wait_sec = MAX_TRANSFER_WAIT_MINUTES * 60.0

    results: dict[str, dict[str, dict[str, float]]] = {}

    for i, neigh in enumerate(neighbors):
        stop_a = gstop_ids[i]
        targets_here = [r for r in serves.get(stop_a, ()) if r in target_ids]
        if not targets_here:
            continue
        for j in neigh:
            stop_b = gstop_ids[j]
            routes_b = serves.get(stop_b)
            if not routes_b:
                continue
            distance = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
            min_wait_sec = distance / walk_mps + buffer_sec if walk_mps > 0 else buffer_sec

            for route_a in targets_here:
                for route_b in routes_b:
                    if route_b == route_a:
                        continue
                    wait: Optional[float]
                    if ENABLE_TIME_CHECK:
                        arr = arrivals.get((stop_a, route_a))
                        dep = departures.get((stop_b, route_b))
                        if arr is None or dep is None:
                            continue
                        feasible, wait = has_timed_connection(arr, dep, min_wait_sec, max_wait_sec)
                        if not feasible:
                            continue
                    else:
                        wait = None

                    conn = results.setdefault(route_a, {}).setdefault(
                        route_b,
                        {"stop_pairs": 0.0, "min_distance_m": np.inf, "min_wait_seconds": np.inf},
                    )
                    conn["stop_pairs"] += 1
                    conn["min_distance_m"] = min(conn["min_distance_m"], distance)
                    if wait is not None:
                        conn["min_wait_seconds"] = min(conn["min_wait_seconds"], wait)

    return results


def _connector_label(connector_id: str, target_feed: str, routes: pd.DataFrame) -> str:
    """Format a connector route for the output list, prefixing cross-feed routes.

    Args:
        connector_id: The connector ``groute_id``.
        target_feed: The target route's feed label.
        routes: Route table indexed by ``groute_id``.

    Returns:
        The route label, prefixed with ``"<feed>:"`` when it differs from the
        target's feed.
    """
    row = routes.loc[connector_id]
    label = str(row["route_label"])
    feed = str(row["feed"])
    return label if feed == target_feed else f"{feed}:{label}"


def build_output_tables(
    results: dict[str, dict[str, dict[str, float]]], routes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the transfer results into a summary and a detail DataFrame.

    Args:
        results: Output of :func:`compute_transfers`.
        routes: Route table indexed by ``groute_id``.

    Returns:
        A ``(summary, detail)`` tuple. The summary has one row per target route
        (including targets with zero transfers), keyed on ``route_id`` so it can
        be joined downstream (e.g. by the prep_features_public.py orchestrator);
        the detail has one row per (target, connector) pair.
    """
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    target_ids = list(routes.index[routes["is_target"]])
    for target_id in target_ids:
        target = routes.loc[target_id]
        target_feed = str(target["feed"])
        connectors = results.get(target_id, {})

        labelled = sorted(
            (
                (_connector_label(cid, target_feed, routes), cid, stats)
                for cid, stats in connectors.items()
            ),
            key=lambda item: item[0],
        )

        summary_rows.append(
            {
                "feed": target_feed,
                "route_id": str(target["route_id"]),
                "route_short_name": str(target["route_short_name"]),
                "route_long_name": str(target["route_long_name"]),
                "transfer_route_count": len(labelled),
                "transfer_routes": ", ".join(label for label, _, _ in labelled),
            }
        )

        for label, cid, stats in labelled:
            connector = routes.loc[cid]
            min_wait = stats["min_wait_seconds"]
            detail_rows.append(
                {
                    "target_feed": target_feed,
                    "target_route_id": str(target["route_id"]),
                    "target_route_short_name": str(target["route_short_name"]),
                    "connector_feed": str(connector["feed"]),
                    "connector_route_id": str(connector["route_id"]),
                    "connector_route_short_name": str(connector["route_short_name"]),
                    "connector_label": label,
                    "cross_feed": str(connector["feed"]) != target_feed,
                    "qualifying_stop_pairs": int(stats["stop_pairs"]),
                    "nearest_walk_distance_m": round(stats["min_distance_m"], 1),
                    "min_transfer_wait_min": (
                        round(min_wait / 60.0, 1) if np.isfinite(min_wait) else ""
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["feed", "route_short_name", "route_id"])
    detail = pd.DataFrame(detail_rows)
    return summary, detail


# =============================================================================
# MAIN
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the configuration block.

    ``parse_known_args`` is used so a notebook kernel's injected argv (or the
    orchestrator's extra ``--input-dir`` token) does not raise ``SystemExit: 2``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the routes each target route can transfer to across one or more "
            "GTFS feeds. Defaults come from the CONFIGURATION block at the top of this file."
        )
    )
    parser.add_argument(
        "--gtfs-dirs",
        nargs="+",
        default=list(GTFS_FEEDS),
        help="One or more GTFS feed folders (pooled into a single network).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory the summary/detail CSVs are written to.",
    )
    parser.add_argument(
        "--day",
        default=DAY_OF_WEEK if DAY_OF_WEEK is not None else "all",
        help=(
            "Service-day filter: 'weekday' (any Mon-Fri service; default), a single "
            "day name (monday..sunday), or 'all' to keep every trip."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the route transfer calculation and write the output CSV(s)."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    feeds = [str(f) for f in args.gtfs_dirs if f and str(f) != _PLACEHOLDER_FEED]
    output_dir = Path(args.output_dir)
    if not feeds or output_dir == _PLACEHOLDER_OUTPUT:
        logging.warning(
            "GTFS feed folder(s) and/or the output folder are still set to placeholder "
            "values. Update the CONFIGURATION section or pass --gtfs-dirs/--output-dir."
        )
        return

    day_token = str(args.day).strip().lower()
    day_filter: Optional[str] = None if day_token in ("", "all", "none") else day_token

    labels = _unique_labels(feeds, FEED_LABELS)

    loaded = []
    for folder, label in zip(feeds, labels):
        try:
            loaded.append(_read_feed(folder, label))
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            logging.error("Failed to load feed '%s': %s", label, exc)
            return

    routes = _build_route_table([feed["routes"] for feed in loaded])
    n_targets = int(routes["is_target"].sum())
    if n_targets == 0:
        logging.warning("No routes matched the target criteria. Nothing to do.")
        return
    logging.info("Identified %d target route(s) across %d feed(s).", n_targets, len(loaded))

    stops = pd.concat([feed["stops"] for feed in loaded], ignore_index=True)
    trips = pd.concat([feed["trips"] for feed in loaded], ignore_index=True)
    stop_times = pd.concat([feed["stop_times"] for feed in loaded], ignore_index=True)

    keep_service_ids: dict[str, Optional[set[str]]] = {}
    if day_filter is not None:
        logging.info("Service-day filter: %s.", day_filter)
        for feed in loaded:
            label = str(feed["routes"]["feed"].iloc[0])
            ids = service_ids_active_on_day(feed["calendar"], day_filter)
            if ids is None:
                logging.warning(
                    "Feed '%s': no usable calendar.txt for day filter; keeping all trips.",
                    label,
                )
            keep_service_ids[label] = ids
    else:
        logging.info("No service-day filter: pooling all trips regardless of service day.")
        keep_service_ids = {str(feed["routes"]["feed"].iloc[0]): None for feed in loaded}

    stops_gdf = _project_stops(stops)
    events = _build_stop_events(stop_times, trips, keep_service_ids)

    radius_m = distance_to_meters(MAX_TRANSFER_DISTANCE, MAX_TRANSFER_DISTANCE_UNIT)
    logging.info(
        "Space buffer: %.1f m. Time check: %s.",
        radius_m,
        "on" if ENABLE_TIME_CHECK else "off",
    )

    results = compute_transfers(stops_gdf, events, routes, radius_m)
    summary, detail = build_output_tables(results, routes)

    os.makedirs(output_dir, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    logging.info("Wrote summary for %d target route(s) -> %s", len(summary), summary_path)

    if WRITE_DETAIL:
        detail_path = output_dir / DETAIL_FILENAME
        detail.to_csv(detail_path, index=False)
        logging.info("Wrote %d transfer pair(s) -> %s", len(detail), detail_path)

    logging.info("Done.")


if __name__ == "__main__":
    main()
