"""Compute the routes each target route can transfer to within one GTFS feed.

For every *target* route the user designates, this script finds the other routes a
rider can transfer to, subject to a **space buffer** (how far a rider will walk
between an alighting stop and a boarding stop) and an optional **time buffer**
(the connecting trip must depart after the feeder trip arrives — accounting for
walk time — and within a maximum wait).

Targets are chosen with ``TARGET_ROUTE_TOKENS`` (matched against
``route_short_name`` or ``route_id``); every route in the feed is eligible as a
*connector*. To find transfers between agencies, first merge their feeds with
``gtfs_feed_merger.py``, which combines duplicate stops (so a stop both
agencies serve counts as one shared stop) and aligns their calendars, then run
this script on the merged feed. ``TARGET_AGENCY_IDS`` can restrict targets to
one agency's routes, and connectors from another agency are labelled
``"<agency_id>:<route>"``.

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
    - A GTFS feed folder with stops.txt, routes.txt, trips.txt, and
      stop_times.txt (a single agency's feed, or a merged multi-agency feed
      from gtfs_feed_merger.py). calendar.txt is used when a day filter is set.

Outputs:
    - A summary CSV keyed on ``route_id``: one row per target route with a
      transfer count and the comma-separated list of routes it can transfer to.
    - An optional detail CSV: one row per (target route, connector route) pair
      with the number of qualifying stop pairs, nearest walk distance, and the
      shortest feasible wait observed.

Defaults come from the CONFIGURATION block; ``--gtfs-dir`` / ``--output-dir`` /
``--day`` / ``--log-level`` override them, so the script can run standalone or
under the prep_features_public.py orchestrator.

Typical usage:
    Update the paths in the CONFIGURATION block (or pass the CLI flags above)
    and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

# =============================================================================
# CONFIGURATION
# =============================================================================

# GTFS feed folder containing stops.txt, routes.txt, trips.txt, and
# stop_times.txt. For transfers between agencies, merge their feeds first with
# gtfs_feed_merger.py and point this at the merged feed. Use raw strings (r"")
# for Windows paths.
GTFS_FOLDER: str = r"Path\To\Your\GTFS_Folder"

# Folder where the output CSV(s) are written.
OUTPUT_DIR: Path = Path(r"Path\To\Output_Folder")

# Routes we want transfers FOR. Each token is matched against route_short_name OR
# route_id. Leave empty to treat every route as a target.
TARGET_ROUTE_TOKENS: set[str] = set()  # e.g. {"10", "20"}; empty = every route is a target

# Optionally restrict targets to these routes.agency_id values, e.g. {"FFX"} in a
# merged multi-agency feed. None = any agency. Connectors are always drawn from
# every agency regardless of this setting.
TARGET_AGENCY_IDS: Optional[set[str]] = None

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


def _read_feed(folder: str) -> tuple[dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Load the GTFS tables this script needs from one feed folder.

    Args:
        folder: Path to the GTFS feed folder.

    Returns:
        A tuple ``(tables, calendar)``: ``stops``, ``routes``, ``trips`` and
        ``stop_times`` frames, and calendar.txt (None when the feed has none).

    Raises:
        FileNotFoundError: If a required GTFS file is missing.
    """
    for filename in _REQUIRED_FILES:
        if not os.path.exists(os.path.join(folder, filename)):
            raise FileNotFoundError(f"Missing {filename} in {folder}")

    def _read(name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(folder, name), dtype=str, low_memory=False)

    stops = _read("stops.txt")
    routes = _read("routes.txt")
    trips = _read("trips.txt")
    stop_times = _read("stop_times.txt")

    calendar_path = os.path.join(folder, "calendar.txt")
    calendar: Optional[pd.DataFrame] = None
    if os.path.exists(calendar_path):
        try:
            calendar = pd.read_csv(calendar_path, dtype=str)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logging.warning("Could not read calendar.txt (%s).", exc)

    logging.info(
        "Feed: %d stops, %d routes, %d trips, %d stop_times.",
        len(stops),
        len(routes),
        len(trips),
        len(stop_times),
    )
    tables = {"stops": stops, "routes": routes, "trips": trips, "stop_times": stop_times}
    return tables, calendar


def _build_route_table(routes: pd.DataFrame) -> pd.DataFrame:
    """Add display labels and target flags to the routes table.

    Args:
        routes: Parsed routes.txt.

    Returns:
        A routes frame indexed by ``route_id`` with ``agency_id`` (blank when
        the feed omits it), ``route_label`` (short name, else long name, else
        route_id) and an ``is_target`` flag.
    """
    routes = routes.copy()
    routes["route_id"] = routes["route_id"].astype(str)
    for col in ("agency_id", "route_short_name", "route_long_name"):
        if col not in routes.columns:
            routes[col] = ""
        routes[col] = routes[col].fillna("").astype(str).str.strip()

    label = routes["route_short_name"].where(routes["route_short_name"] != "")
    label = label.fillna(routes["route_long_name"].where(routes["route_long_name"] != ""))
    routes["route_label"] = label.fillna(routes["route_id"])

    if TARGET_ROUTE_TOKENS:
        tokens = {str(t) for t in TARGET_ROUTE_TOKENS}
        is_target = routes["route_short_name"].isin(tokens) | routes["route_id"].isin(tokens)
    else:
        is_target = pd.Series(True, index=routes.index)

    if TARGET_AGENCY_IDS:
        is_target = is_target & routes["agency_id"].isin({str(a) for a in TARGET_AGENCY_IDS})

    routes["is_target"] = is_target
    return routes.set_index("route_id", drop=False)


def _project_stops(stops: pd.DataFrame) -> gpd.GeoDataFrame:
    """Project unique stops to ``PROJECTED_CRS`` and attach metric x/y columns.

    Args:
        stops: Stops frame with ``stop_id``, ``stop_lat``, ``stop_lon``.

    Returns:
        A GeoDataFrame of unique stops with numeric ``x`` and ``y`` (meters).
    """
    unique = stops.drop_duplicates("stop_id").copy()
    unique["stop_id"] = unique["stop_id"].astype(str)
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
    stop_times: pd.DataFrame, trips: pd.DataFrame, keep_service_ids: Optional[set[str]]
) -> pd.DataFrame:
    """Join stop_times to trips and resolve feeder/connector event times.

    Args:
        stop_times: Parsed stop_times.txt.
        trips: Parsed trips.txt.
        keep_service_ids: service_ids to keep (day filter), or None to keep
            every trip.

    Returns:
        A frame with ``stop_id``, ``route_id``, ``arrival_sec`` (feeder time),
        and ``departure_sec`` (connector time).
    """
    trip_cols = trips[["trip_id", "route_id", "service_id"]].astype(str)
    events = stop_times.astype({"trip_id": str, "stop_id": str}).merge(
        trip_cols, on="trip_id", how="inner"
    )

    if keep_service_ids is not None:
        mask = events["service_id"].isin(keep_service_ids)
        logging.info("Day-of-week filter kept %d/%d stop events.", int(mask.sum()), len(events))
        events = events.loc[mask].copy()

    arrival = events["arrival_time"].map(parse_gtfs_time)
    departure = events["departure_time"].map(parse_gtfs_time)
    # Feeders use arrival (fall back to departure); connectors use departure
    # (fall back to arrival) so single-time rows still participate.
    events["arrival_sec"] = arrival.fillna(departure)
    events["departure_sec"] = departure.fillna(arrival)
    return events[["stop_id", "route_id", "arrival_sec", "departure_sec"]]


_StopRouteArrays = dict[tuple[str, str], np.ndarray]


def _index_events(
    events: pd.DataFrame,
) -> tuple[dict[str, set[str]], _StopRouteArrays, _StopRouteArrays]:
    """Build lookups of routes-per-stop and sorted time arrays per (stop, route).

    Args:
        events: Output of :func:`_build_stop_events`.

    Returns:
        A tuple ``(serves, arrivals, departures)`` where ``serves`` maps a
        ``stop_id`` to the set of routes serving it, ``arrivals`` maps
        ``(stop_id, route_id)`` to a sorted array of feeder arrival seconds, and
        ``departures`` maps it to a sorted array of connector departure seconds.
    """
    serves: dict[str, set[str]] = {}
    for stop_id, route_id in zip(events["stop_id"], events["route_id"]):
        serves.setdefault(stop_id, set()).add(route_id)

    arrivals: dict[tuple[str, str], np.ndarray] = {}
    departures: dict[tuple[str, str], np.ndarray] = {}
    for (stop_id, route_id), group in events.groupby(["stop_id", "route_id"], sort=False):
        arr = group["arrival_sec"].dropna().to_numpy(dtype=float)
        dep = group["departure_sec"].dropna().to_numpy(dtype=float)
        arr.sort()
        dep.sort()
        arrivals[(stop_id, route_id)] = arr
        departures[(stop_id, route_id)] = dep
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
        routes: Route table indexed by ``route_id`` with ``is_target``.
        radius_m: Space buffer in meters.

    Returns:
        A nested mapping ``target_route_id -> connector_route_id -> stats``,
        where stats holds ``stop_pairs``, ``min_distance_m``, and
        ``min_wait_seconds`` (NaN when the time check is disabled).
    """
    serves, arrivals, departures = _index_events(events)
    target_ids = set(routes.index[routes["is_target"]])

    coords = np.column_stack([stops_gdf["x"].to_numpy(), stops_gdf["y"].to_numpy()])
    stop_ids = stops_gdf["stop_id"].to_numpy()
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius_m, p=2.0)

    walk_mps = WALK_SPEED_MPH * 1609.34 / 3600.0
    buffer_sec = MIN_TRANSFER_BUFFER_MINUTES * 60.0
    max_wait_sec = MAX_TRANSFER_WAIT_MINUTES * 60.0

    results: dict[str, dict[str, dict[str, float]]] = {}

    for i, neigh in enumerate(neighbors):
        stop_a = stop_ids[i]
        targets_here = [r for r in serves.get(stop_a, ()) if r in target_ids]
        if not targets_here:
            continue
        for j in neigh:
            stop_b = stop_ids[j]
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


def _connector_label(connector_id: str, target_agency: str, routes: pd.DataFrame) -> str:
    """Format a connector route for the output list, prefixing other agencies' routes.

    Args:
        connector_id: The connector ``route_id``.
        target_agency: The target route's ``agency_id`` (may be blank).
        routes: Route table indexed by ``route_id``.

    Returns:
        The route label, prefixed with ``"<agency_id>:"`` when the connector
        belongs to a different agency than the target.
    """
    row = routes.loc[connector_id]
    label = str(row["route_label"])
    agency = str(row["agency_id"])
    return label if agency in ("", target_agency) else f"{agency}:{label}"


def build_output_tables(
    results: dict[str, dict[str, dict[str, float]]], routes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the transfer results into a summary and a detail DataFrame.

    Args:
        results: Output of :func:`compute_transfers`.
        routes: Route table indexed by ``route_id``.

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
        target_agency = str(target["agency_id"])
        connectors = results.get(target_id, {})

        labelled = sorted(
            (
                (_connector_label(cid, target_agency, routes), cid, stats)
                for cid, stats in connectors.items()
            ),
            key=lambda item: item[0],
        )

        summary_rows.append(
            {
                "agency_id": target_agency,
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
                    "target_agency_id": target_agency,
                    "target_route_id": str(target["route_id"]),
                    "target_route_short_name": str(target["route_short_name"]),
                    "connector_agency_id": str(connector["agency_id"]),
                    "connector_route_id": str(connector["route_id"]),
                    "connector_route_short_name": str(connector["route_short_name"]),
                    "connector_label": label,
                    "cross_agency": str(connector["agency_id"]) != target_agency,
                    "qualifying_stop_pairs": int(stats["stop_pairs"]),
                    "nearest_walk_distance_m": round(stats["min_distance_m"], 1),
                    "min_transfer_wait_min": (
                        round(min_wait / 60.0, 1) if np.isfinite(min_wait) else ""
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["agency_id", "route_short_name", "route_id"])
    detail = pd.DataFrame(detail_rows)
    return summary, detail


# =============================================================================
# MAIN
# =============================================================================


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args strictly, defaulting to the configuration block.

    Notebook kernels' injected argv is shielded via ``notebook_safe_argv``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the routes each target route can transfer to within one GTFS feed. "
            "Defaults come from the CONFIGURATION block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-dir",
        default=GTFS_FOLDER,
        help="GTFS feed folder (merge several agencies first with gtfs_feed_merger.py).",
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
            "Service-day filter: 'weekday' (any Mon-Fri service), a single "
            "day name (monday..sunday), or 'all' to keep every trip."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    args = parser.parse_args(notebook_safe_argv(argv))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the route transfer calculation and write the output CSV(s).

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    gtfs_dir = str(args.gtfs_dir)
    output_dir = Path(args.output_dir)
    if not gtfs_dir or gtfs_dir == _PLACEHOLDER_FEED or output_dir == _PLACEHOLDER_OUTPUT:
        logging.warning(
            "The GTFS feed folder and/or the output folder are still set to placeholder "
            "values. Update the CONFIGURATION section or pass --gtfs-dir/--output-dir."
        )
        return 2

    day_token = str(args.day).strip().lower()
    day_filter: Optional[str] = None if day_token in ("", "all", "none") else day_token

    try:
        feed, calendar = _read_feed(gtfs_dir)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        logging.error("Failed to load feed '%s': %s", gtfs_dir, exc)
        return 1

    routes = _build_route_table(feed["routes"])
    n_targets = int(routes["is_target"].sum())
    if n_targets == 0:
        logging.warning("No routes matched the target criteria. Nothing to do.")
        return 1
    logging.info("Identified %d target route(s).", n_targets)

    keep_service_ids: Optional[set[str]] = None
    if day_filter is not None:
        logging.info("Service-day filter: %s.", day_filter)
        keep_service_ids = service_ids_active_on_day(calendar, day_filter)
        if keep_service_ids is None:
            logging.warning("No usable calendar.txt for the day filter; keeping all trips.")
    else:
        logging.info("No service-day filter: pooling all trips regardless of service day.")

    stops_gdf = _project_stops(feed["stops"])
    events = _build_stop_events(feed["stop_times"], feed["trips"], keep_service_ids)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
