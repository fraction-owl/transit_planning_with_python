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

Connections are evaluated on actual service dates. ``calendar.txt`` (each
service's weekly pattern within its ``start_date``–``end_date`` range) and
``calendar_dates.txt`` (dates added or removed) are expanded into the dates each
``service_id`` operates, and a feeder trip and a connector trip pair up only when
both operate on at least one common date. ``DAY_OF_WEEK`` limits which dates
count: ``"weekday"`` (the default) needs a common Monday–Friday date, a single
day name needs that day, and ``None`` / ``--day all`` accepts any date. A
Monday-only arrival therefore never connects to a Friday-only departure. Times
are compared within one service date, so a connection after midnight to the next
service date's trips is not counted.

Inputs:
    - A GTFS feed folder with stops.txt, routes.txt, trips.txt, and
      stop_times.txt (a single agency's feed, or a merged multi-agency feed
      from gtfs_feed_merger.py), plus calendar.txt and/or calendar_dates.txt to
      give each service's dates. A feed with neither is rejected rather than
      treated as running every trip together.

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
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

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

# Service-day filter. A transfer only counts when the feeder and connector trips
# both operate on a common service date (from calendar.txt date ranges and
# calendar_dates.txt additions/removals). "weekday" (the default) requires that
# common date to be Monday-Friday; a single day name (e.g., "monday") requires
# that day; None accepts any date. Feeds with neither calendar.txt nor
# calendar_dates.txt are rejected.
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


def expand_service_active_dates(
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame] = None,
    max_days_per_service: int = 1830,
    today: Optional[dt.date] = None,
) -> dict[str, set[dt.date]]:
    """Expand each service_id into its real set of active calendar dates.

    Builds the base date set from each ``calendar.txt`` row (day-of-week
    pattern × ``start_date``–``end_date`` range), then applies
    ``calendar_dates.txt`` exceptions (``exception_type`` 1 adds a date,
    2 removes it). Handles calendar_dates-only feeds (*calendar_df* empty or
    ``None``), redundant additions, and fully negated base patterns — the
    returned sets reflect only the dates a service truly operates.

    Rows with unparseable or reversed dates are skipped with a warning.
    A date range longer than *max_days_per_service* (a common placeholder
    pattern, e.g. 2000–2099) is clamped to a window of that length centred
    on *today* and logged, so expansion stays fast and downstream per-year
    statistics stay meaningful.

    Args:
        calendar_df: Parsed ``calendar.txt``, or ``None`` if the feed has
            none. Expected columns: ``service_id``, the seven day-of-week
            flags, ``start_date``, ``end_date``.
        calendar_dates_df: Parsed ``calendar_dates.txt`` or ``None``.
            Expected columns: ``service_id``, ``date``, ``exception_type``.
        max_days_per_service: Longest date range expanded per service before
            clamping kicks in. The default (1830 ≈ 5 years) is far beyond
            any real service span but well short of placeholder ranges.
        today: Anchor date for clamping oversized ranges. Defaults to the
            current date; pass a fixed date for deterministic tests.

    Returns:
        Mapping of ``service_id`` (as ``str``) to the set of dates the
        service operates. Services whose dates never parse map to an empty
        set rather than being dropped, so callers can report them.

    Raises:
        ValueError: If *calendar_df* is provided but lacks ``service_id``,
            ``start_date``, or ``end_date`` columns.
    """
    day_cols = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    anchor = dt.date.today() if today is None else today
    active: dict[str, set[dt.date]] = {}

    if calendar_df is not None and not calendar_df.empty:
        required = {"service_id", "start_date", "end_date"}
        missing = required - set(calendar_df.columns)
        if missing:
            raise ValueError(f"calendar.txt is missing required column(s): {sorted(missing)}")
        for _, row in calendar_df.iterrows():
            sid = str(row["service_id"]).strip()
            try:
                start = dt.datetime.strptime(str(row["start_date"]).strip(), "%Y%m%d").date()
                end = dt.datetime.strptime(str(row["end_date"]).strip(), "%Y%m%d").date()
            except ValueError:
                logging.warning("Service %s: unparseable start/end date — skipping row.", sid)
                active.setdefault(sid, set())
                continue
            if end < start:
                logging.warning(
                    "Service %s: end_date %s precedes start_date %s — skipping row.",
                    sid,
                    end,
                    start,
                )
                active.setdefault(sid, set())
                continue
            if (end - start).days + 1 > max_days_per_service:
                half = max_days_per_service // 2
                clamped_start = max(start, anchor - dt.timedelta(days=half))
                clamped_end = min(end, anchor + dt.timedelta(days=half))
                logging.warning(
                    "Service %s: date range %s–%s looks like a placeholder; "
                    "clamping expansion to %s–%s.",
                    sid,
                    start,
                    end,
                    clamped_start,
                    clamped_end,
                )
                start, end = clamped_start, clamped_end
            pattern = [str(row.get(c, "0")).strip() == "1" for c in day_cols]
            dates = active.setdefault(sid, set())
            d = start
            while d <= end:
                if pattern[d.weekday()]:
                    dates.add(d)
                d += dt.timedelta(days=1)

    if calendar_dates_df is not None and not calendar_dates_df.empty:
        bad_rows = 0
        for _, row in calendar_dates_df.iterrows():
            sid = str(row["service_id"]).strip()
            try:
                d = dt.datetime.strptime(str(row["date"]).strip(), "%Y%m%d").date()
            except ValueError:
                bad_rows += 1
                continue
            etype = str(row.get("exception_type", "")).strip()
            dates = active.setdefault(sid, set())
            if etype == "1":
                dates.add(d)
            elif etype == "2":
                dates.discard(d)
            else:
                bad_rows += 1
        if bad_rows:
            logging.warning(
                "calendar_dates.txt: skipped %d row(s) with unparseable date/exception_type.",
                bad_rows,
            )

    return active


def weekdays_for_filter(day: Optional[str]) -> set[int]:
    """Return the weekday numbers (Monday = 0) a ``DAY_OF_WEEK`` value allows.

    Args:
        day: ``"weekday"``, a lowercase day name (e.g. ``"monday"``), or None
            for every day.

    Returns:
        The allowed ``datetime.date.weekday()`` values.

    Raises:
        ValueError: If ``day`` is not recognized.
    """
    if day is None:
        return set(range(7))
    key = day.strip().lower()
    if key == "weekday":
        return set(range(5))
    if key in _DAY_COLUMNS:
        return {_DAY_COLUMNS.index(key)}
    raise ValueError(f"DAY_OF_WEEK must be 'weekday' or one of {_DAY_COLUMNS}, got '{day}'.")


def compatible_service_ids(
    active_dates: Mapping[str, set[dt.date]], weekdays: set[int]
) -> dict[str, set[str]]:
    """Map each service_id to the service_ids sharing an active date on *weekdays*.

    Trip times repeat on every date a service operates, so a feeder trip and a
    connector trip can meet exactly when their services share an active date.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        weekdays: Allowed weekday numbers, from :func:`weekdays_for_filter`.

    Returns:
        Mapping of service_id to the service_ids (itself included) operating on
        at least one common allowed date. Services with no allowed date are
        absent.
    """
    by_date: dict[dt.date, set[str]] = {}
    for sid, dates in active_dates.items():
        for d in dates:
            if d.weekday() in weekdays:
                by_date.setdefault(d, set()).add(sid)

    compatible: dict[str, set[str]] = {}
    for sids in {frozenset(ids) for ids in by_date.values()}:
        for sid in sids:
            compatible.setdefault(sid, set()).update(sids)
    return compatible


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


def _read_feed(
    folder: str,
) -> tuple[dict[str, pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load the GTFS tables this script needs from one feed folder.

    Args:
        folder: Path to the GTFS feed folder.

    Returns:
        A tuple ``(tables, calendar, calendar_dates)``: ``stops``, ``routes``,
        ``trips`` and ``stop_times`` frames, then calendar.txt and
        calendar_dates.txt (each None when the feed lacks it).

    Raises:
        FileNotFoundError: If a required GTFS file is missing, or the feed has
            neither calendar.txt nor calendar_dates.txt.
    """
    for filename in _REQUIRED_FILES:
        if not os.path.exists(os.path.join(folder, filename)):
            raise FileNotFoundError(f"Missing {filename} in {folder}")

    def _read(name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(folder, name), dtype=str, low_memory=False)

    def _read_optional(name: str) -> Optional[pd.DataFrame]:
        return _read(name) if os.path.exists(os.path.join(folder, name)) else None

    stops = _read("stops.txt")
    routes = _read("routes.txt")
    trips = _read("trips.txt")
    stop_times = _read("stop_times.txt")

    calendar = _read_optional("calendar.txt")
    calendar_dates = _read_optional("calendar_dates.txt")
    if calendar is None and calendar_dates is None:
        raise FileNotFoundError(
            f"Missing calendar.txt and calendar_dates.txt in {folder}; service dates are "
            "needed to pair only trips that run on the same date."
        )

    logging.info(
        "Feed: %d stops, %d routes, %d trips, %d stop_times.",
        len(stops),
        len(routes),
        len(trips),
        len(stop_times),
    )
    tables = {"stops": stops, "routes": routes, "trips": trips, "stop_times": stop_times}
    return tables, calendar, calendar_dates


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
    stop_times: pd.DataFrame, trips: pd.DataFrame, keep_service_ids: set[str]
) -> pd.DataFrame:
    """Join stop_times to trips and resolve feeder/connector event times.

    Args:
        stop_times: Parsed stop_times.txt.
        trips: Parsed trips.txt.
        keep_service_ids: service_ids operating on at least one selected date;
            other trips are dropped.

    Returns:
        A frame with ``stop_id``, ``route_id``, ``service_id``, ``arrival_sec``
        (feeder time), and ``departure_sec`` (connector time).
    """
    trip_cols = trips[["trip_id", "route_id", "service_id"]].astype(str)
    # Calendar service_ids are whitespace-stripped (expand_service_active_dates).
    trip_cols["service_id"] = trip_cols["service_id"].str.strip()
    events = stop_times.astype({"trip_id": str, "stop_id": str}).merge(
        trip_cols, on="trip_id", how="inner"
    )

    mask = events["service_id"].isin(keep_service_ids)
    logging.info("Service-date filter kept %d/%d stop events.", int(mask.sum()), len(events))
    events = events.loc[mask].copy()

    arrival = events["arrival_time"].map(parse_gtfs_time)
    departure = events["departure_time"].map(parse_gtfs_time)
    # Feeders use arrival (fall back to departure); connectors use departure
    # (fall back to arrival) so single-time rows still participate.
    events["arrival_sec"] = arrival.fillna(departure)
    events["departure_sec"] = departure.fillna(arrival)
    return events[["stop_id", "route_id", "service_id", "arrival_sec", "departure_sec"]]


_StopRouteArrays = dict[tuple[str, str], dict[str, np.ndarray]]


def _index_events(
    events: pd.DataFrame,
) -> tuple[dict[str, set[str]], _StopRouteArrays, _StopRouteArrays]:
    """Build lookups of routes-per-stop and sorted time arrays per (stop, route).

    Args:
        events: Output of :func:`_build_stop_events`.

    Returns:
        A tuple ``(serves, arrivals, departures)`` where ``serves`` maps a
        ``stop_id`` to the set of routes serving it, ``arrivals`` maps
        ``(stop_id, route_id)`` to ``{service_id: sorted feeder arrival
        seconds}``, and ``departures`` maps it to ``{service_id: sorted
        connector departure seconds}``.
    """
    serves: dict[str, set[str]] = {}
    for stop_id, route_id in zip(events["stop_id"], events["route_id"]):
        serves.setdefault(stop_id, set()).add(route_id)

    arrivals: _StopRouteArrays = {}
    departures: _StopRouteArrays = {}
    for (stop_id, route_id, service_id), group in events.groupby(
        ["stop_id", "route_id", "service_id"], sort=False
    ):
        arr = group["arrival_sec"].dropna().to_numpy(dtype=float)
        dep = group["departure_sec"].dropna().to_numpy(dtype=float)
        arr.sort()
        dep.sort()
        arrivals.setdefault((stop_id, route_id), {})[service_id] = arr
        departures.setdefault((stop_id, route_id), {})[service_id] = dep
    return serves, arrivals, departures


def compute_transfers(
    stops_gdf: gpd.GeoDataFrame,
    events: pd.DataFrame,
    routes: pd.DataFrame,
    radius_m: float,
    compatible: Mapping[str, set[str]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute feasible transfers from each target route to connector routes.

    A stop pair only counts when the feeder and connector serve it under
    services sharing an active date (per *compatible*); with the time check on,
    the timed connection must also work between such services.

    Args:
        stops_gdf: Projected unique stops with ``x``/``y`` in meters.
        events: Stop events from :func:`_build_stop_events`.
        routes: Route table indexed by ``route_id`` with ``is_target``.
        radius_m: Space buffer in meters.
        compatible: Output of :func:`compatible_service_ids`.

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
                arr_by_service = arrivals.get((stop_a, route_a), {})
                for route_b in routes_b:
                    if route_b == route_a:
                        continue
                    dep_by_service = departures.get((stop_b, route_b), {})
                    # Service pairs that operate on a common date at these stops.
                    service_pairs = [
                        (sid_a, sid_b)
                        for sid_a in arr_by_service
                        for sid_b in dep_by_service.keys() & compatible.get(sid_a, set())
                    ]
                    if not service_pairs:
                        continue
                    wait: Optional[float] = None
                    if ENABLE_TIME_CHECK:
                        for sid_a, sid_b in service_pairs:
                            _, pair_wait = has_timed_connection(
                                arr_by_service[sid_a],
                                dep_by_service[sid_b],
                                min_wait_sec,
                                max_wait_sec,
                            )
                            if pair_wait is not None and (wait is None or pair_wait < wait):
                                wait = pair_wait
                        if wait is None:
                            continue

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
            "Service-day filter for the common service date both trips must run on: "
            "'weekday' (Mon-Fri), a single day name (monday..sunday), or 'all' (any date)."
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
        weekdays = weekdays_for_filter(day_filter)
    except ValueError as exc:
        logging.error("%s Use --day all for any date.", exc)
        return 2

    try:
        feed, calendar, calendar_dates = _read_feed(gtfs_dir)
        active_dates = expand_service_active_dates(calendar, calendar_dates)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        logging.error("Failed to load feed '%s': %s", gtfs_dir, exc)
        return 1

    routes = _build_route_table(feed["routes"])
    n_targets = int(routes["is_target"].sum())
    if n_targets == 0:
        logging.warning("No routes matched the target criteria. Nothing to do.")
        return 1
    logging.info("Identified %d target route(s).", n_targets)

    compatible = compatible_service_ids(active_dates, weekdays)
    if not compatible:
        logging.error(
            "No service in calendar.txt / calendar_dates.txt operates on a %s date.",
            day_filter or "service",
        )
        return 1
    unscheduled = set(feed["trips"]["service_id"].astype(str).str.strip()) - set(active_dates)
    if unscheduled:
        logging.warning(
            "Trips with service_id(s) absent from calendar.txt and calendar_dates.txt are "
            "excluded: %s",
            ", ".join(sorted(unscheduled)),
        )
    logging.info(
        "Service-day filter: %s. Transfers need both trips on a common service date.",
        day_filter or "all",
    )

    stops_gdf = _project_stops(feed["stops"])
    events = _build_stop_events(feed["stop_times"], feed["trips"], set(compatible))

    radius_m = distance_to_meters(MAX_TRANSFER_DISTANCE, MAX_TRANSFER_DISTANCE_UNIT)
    logging.info(
        "Space buffer: %.1f m. Time check: %s.",
        radius_m,
        "on" if ENABLE_TIME_CHECK else "off",
    )

    results = compute_transfers(stops_gdf, events, routes, radius_m, compatible)
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
