"""Profile how redundant each transit route is, from four angles, per schedule calendar.

"How redundant is this route?" has no single number, so this script captures
four angles of it at once and emits one row per (schedule, route):

1. **Shared vs. solo stops** — a route's stop is *shared* when any other route
   serves a stop within walking distance of it (the same stop counts, and so
   does an across-the-street stop with a different stop_id); it is *solo* when
   no other route comes that close. Solo stops are the places only this route
   serves.
2. **Routes within walking distance** — every other route a rider standing at
   one of this route's stops could walk to, plus (when the time check is on)
   the subset that is a *timed alternative*: for at least one of this route's
   scheduled stop visits, the other route has a departure nearby that the rider
   could catch within walking time plus a maximum wait (default 30 minutes).
3. **Shared timepoints** — the walking-distance test again, restricted to
   timepoints (stop_times rows with ``timepoint == 1``; feeds that mark none
   fall back to every timed stop). Agencies put timepoints at the major
   destinations along a route, so two routes whose timepoints coincide serve
   the same major destinations — a stronger redundancy signal than sharing
   ordinary local stops.
4. **Solo vs. shared service area** — a route's service area is the union of
   walk-distance buffers around its stops. The *solo* portion is covered by no
   other route; the *shared* portion overlaps at least one other route's
   service area.

Every angle is computed independently per schedule calendar. Each service_id is
classified from its real active-date pattern (calendar.txt × calendar_dates.txt
exceptions) into Weekday / Saturday / Sunday / Holiday, so a route that is
redundant on weekdays but the only service running on Sundays shows both facts.
Feeds with no usable calendar files are profiled once under an "All Service"
label.

Caveats: distances are straight-line in a projected CRS (no pedestrian
network), service areas are stop buffers (not parcels reachable on foot), and
the timed-alternative check compares this route's scheduled arrivals against
the other route's scheduled departures — it asks "was another bus coming
soon nearby", not "does the other route continue to my destination".

Inputs:
    - A GTFS feed (folder or .zip) with stops.txt, routes.txt, trips.txt, and
      stop_times.txt. calendar.txt / calendar_dates.txt, when present, drive
      the per-schedule classification.

Outputs:
    - ``route_redundancy_profile.csv``: one row per (schedule, route) with
      shared/solo stop counts, routes within walking distance, timed
      alternatives, shared timepoint counts, and total/solo/shared service
      area in square miles.
    - ``route_redundancy_partners.csv`` (optional): one row per
      (schedule, route, partner route) pair with shared stop and timepoint
      counts, the nearest walk distance, and the shortest feasible wait.
    - A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage:
    Update the paths in the CONFIGURATION section (or pass the matching CLI
    flags) and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import union_all
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"  # folder of .txt files or a .zip
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Walking distance that defines "shared" throughout: stop sharing, nearby
# routes, timepoint sharing, and the service-area buffer radius all use it.
WALK_DISTANCE: float = 0.25
WALK_DISTANCE_UNIT: str = "miles"  # "miles" | "feet" | "meters"

# --- Timed-alternative check ------------------------------------------------
# When True, a nearby route also gets tested as a timed alternative: some
# scheduled visit of this route must have a departure of the other route
# within walking time plus MAX_ALTERNATIVE_WAIT_MINUTES. When False, the
# timed-alternative columns are left blank and only spatial proximity is
# reported.
ENABLE_TIME_CHECK: bool = True
WALK_SPEED_MPH: float = 3.0
MAX_ALTERNATIVE_WAIT_MINUTES: float = 30.0

# Schedule calendars to profile, matched against the classification labels
# (Weekday, Saturday, Sunday, Holiday — or "All Service" for feeds with no
# usable calendar files). Empty = profile every label found in the feed.
SERVICE_LABELS: list[str] = []

# Route filters, matched against route_id OR route_short_name. Empty = all.
FILTER_IN_ROUTES: list[str] = []
FILTER_OUT_ROUTES: list[str] = []

# Optional service_id filter applied before schedule classification. Empty =
# every service_id in the feed.
FILTER_SERVICE_IDS: list[str] = []

# Keep only platform stops (location_type 0 or blank); parent stations and
# entrances would otherwise double-count shared locations.
FILTER_TO_PLATFORM_STOPS: bool = True

# Projected CRS (in METERS) used for distance and area math. The default is
# NAD83 / Conterminous US Albers, fine for walking distances anywhere in the
# US. For non-US feeds, set a local metric CRS.
PROJECTED_CRS: str = "EPSG:5070"

# Output filenames (written inside OUTPUT_DIR).
SUMMARY_FILENAME: str = r"route_redundancy_profile.csv"
DETAIL_FILENAME: str = r"route_redundancy_partners.csv"
WRITE_DETAIL: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output is never
# left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Columns required in each loaded GTFS table (headers only; values may be blank).
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "stops": {"stop_id", "stop_lat", "stop_lon"},
    "routes": {"route_id"},
    "trips": {"trip_id", "route_id", "service_id"},
    "stop_times": {"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"},
}

# Column order of the two output tables.
SUMMARY_COLUMNS: List[str] = [
    "schedule",
    "route_id",
    "route_short_name",
    "route_long_name",
    "n_stops",
    "n_shared_stops",
    "n_solo_stops",
    "pct_stops_shared",
    "n_routes_within_walk",
    "routes_within_walk",
    "n_timed_alternative_routes",
    "timed_alternative_routes",
    "n_timepoints",
    "n_shared_timepoints",
    "n_solo_timepoints",
    "pct_timepoints_shared",
    "n_timepoint_partner_routes",
    "timepoint_partner_routes",
    "service_area_sqmi",
    "solo_area_sqmi",
    "shared_area_sqmi",
    "pct_area_shared",
]
DETAIL_COLUMNS: List[str] = [
    "schedule",
    "route_id",
    "route_short_name",
    "partner_route_id",
    "partner_route_short_name",
    "n_shared_stops",
    "n_shared_timepoints",
    "nearest_walk_distance_ft",
    "timed_alternative",
    "min_alternative_wait_min",
]

# Fixed display order for the schedule labels produced by classification.
_LABEL_ORDER: dict[str, int] = {"Weekday": 0, "Saturday": 1, "Sunday": 2, "Holiday": 3}

_SQM_PER_SQMI: float = 1609.344**2
_FEET_PER_METER: float = 3.280839895

# =============================================================================
# CORE LOGIC
# =============================================================================


def validate_required_columns(data: Mapping[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` if any required GTFS column is missing.

    Args:
        data: Mapping of table name → DataFrame from :func:`load_gtfs_data`.

    Raises:
        ValueError: Naming every missing table.column so the user can fix
            the feed (or the export that produced it) in one pass.
    """
    problems: list[str] = []
    for table, needed in REQUIRED_COLUMNS.items():
        if table not in data:
            continue
        missing = needed - set(data[table].columns)
        if missing:
            problems.append(f"{table}.txt is missing column(s): {', '.join(sorted(missing))}")
    if problems:
        raise ValueError("GTFS validation failed:\n  " + "\n  ".join(problems))


def filter_trips(
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    service_ids: Sequence[str] = (),
    routes_include: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Apply the schedule and route filters to *trips*.

    Route filters match against both ``route_id`` and ``route_short_name``
    so users can list whichever identifier they know.

    Args:
        trips: Parsed *trips.txt*.
        routes: Parsed *routes.txt* (for the short-name lookup).
        service_ids: Schedules to keep; empty keeps all schedules.
        routes_include: Routes to keep; empty keeps all routes.
        routes_exclude: Routes to drop.

    Returns:
        The filtered trips table.
    """
    kept = trips.copy()
    if service_ids:
        wanted = {str(s) for s in service_ids}
        kept = kept.loc[kept["service_id"].astype(str).isin(wanted)]

    short_names: dict[str, str] = {}
    if "route_short_name" in routes.columns:
        short_names = dict(zip(routes["route_id"], routes["route_short_name"].fillna("")))

    include = {str(r) for r in routes_include}
    exclude = {str(r) for r in routes_exclude}
    if include or exclude:

        def _keys(route_id: str) -> set[str]:
            return {str(route_id), str(short_names.get(route_id, ""))}

        keep_mask = kept["route_id"].map(
            lambda rid: (not include or bool(_keys(rid) & include)) and not (_keys(rid) & exclude)
        )
        kept = kept.loc[keep_mask]

    if kept.empty:
        logging.warning("No trips remain after applying the schedule/route filters.")
    return kept


def select_timepoint_rows(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Return the stop_times rows that count as timepoints.

    Rows with ``timepoint == 1`` are used when the feed marks any. When the
    column is absent — or present but nothing is marked — every row with a
    scheduled arrival or departure time is used instead.

    Args:
        stop_times: Raw *stop_times.txt* DataFrame (string dtypes).

    Returns:
        The timepoint rows, with ``stop_sequence`` coerced to numeric and
        rows lacking a usable sequence dropped.
    """
    out = stop_times.copy()
    out["stop_sequence"] = pd.to_numeric(out["stop_sequence"], errors="coerce")
    n_bad_seq = int(out["stop_sequence"].isna().sum())
    if n_bad_seq:
        logging.warning("Dropped %d stop_times row(s) with non-numeric stop_sequence.", n_bad_seq)
        out = out.loc[out["stop_sequence"].notna()]

    if "timepoint" in out.columns:
        marked = out["timepoint"].fillna("").astype(str).str.strip() == "1"
        if marked.any():
            logging.info("Using %d stop_times rows marked timepoint=1.", int(marked.sum()))
            return out.loc[marked].copy()
        logging.warning(
            "'timepoint' column present but no row is marked 1 — "
            "treating every timed stop as a timepoint."
        )
    else:
        logging.warning(
            "No 'timepoint' column in stop_times.txt — treating every timed stop as a timepoint."
        )

    timed = out["arrival_time"].fillna("").astype(str).str.strip().ne("") | out[
        "departure_time"
    ].fillna("").astype(str).str.strip().ne("")
    logging.info("Using %d stop_times rows with scheduled times as timepoints.", int(timed.sum()))
    return out.loc[timed].copy()


def build_schedule_service_ids(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    trips: pd.DataFrame,
) -> dict[str, set[str]]:
    """Group the feed's service_ids into schedule-calendar buckets.

    Service_ids are classified from their real expanded active dates
    (:func:`expand_service_active_dates` + :func:`classify_service_ids`), so
    day-of-week columns that lie — and holiday-only services stamped with a
    weekday pattern — cannot mislabel a schedule. A service whose dates span
    several buckets (e.g. daily service) lands in each of them.

    Args:
        calendar: Parsed *calendar.txt*, or ``None`` when the feed has none.
        calendar_dates: Parsed *calendar_dates.txt*, or ``None``.
        trips: The (already filtered) trips table, used to fall back to a
            single "All Service" bucket and to report unclassifiable ids.

    Returns:
        Mapping of schedule label → set of service_id strings. Feeds with no
        usable calendar information yield ``{"All Service": <all ids>}``.
    """
    trip_service_ids = {str(s) for s in trips["service_id"]}
    active = expand_service_active_dates(calendar, calendar_dates)
    if not active:
        logging.warning(
            "No usable calendar.txt / calendar_dates.txt — profiling every trip "
            "under a single 'All Service' schedule."
        )
        return {"All Service": trip_service_ids}

    buckets: dict[str, set[str]] = {}
    for sid, labels in classify_service_ids(active).items():
        for label in labels:
            buckets.setdefault(label, set()).add(sid)

    orphans = trip_service_ids - {str(s) for s in active}
    if orphans:
        logging.warning(
            "%d service_id(s) referenced by trips are absent from the calendar files "
            "and cannot be classified; their trips are excluded: %s",
            len(orphans),
            ", ".join(sorted(orphans)),
        )
    return buckets


def project_stops(stops: pd.DataFrame, projected_crs: str = PROJECTED_CRS) -> gpd.GeoDataFrame:
    """Project stops to *projected_crs* and attach metric x/y columns.

    Args:
        stops: Parsed *stops.txt* with ``stop_id``, ``stop_lat``, ``stop_lon``.
        projected_crs: Metric CRS used for all distance and area math.

    Returns:
        A GeoDataFrame of unique stops with numeric ``x`` and ``y`` (meters).
        Stops with missing or non-numeric coordinates are dropped with a
        warning.
    """
    unique = stops.drop_duplicates("stop_id").copy()
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
    ).to_crs(projected_crs)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def build_stop_events(stop_times: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Join stop_times to trips and resolve event times in seconds.

    Args:
        stop_times: The schedule's *stop_times.txt* rows.
        trips: The schedule's *trips.txt* rows (``trip_id`` → ``route_id``).

    Returns:
        A frame with ``stop_id``, ``route_id``, ``arrival_sec``, and
        ``departure_sec``; each time falls back to the other field when blank
        so single-time rows still participate.
    """
    trip_cols = trips[["trip_id", "route_id"]].drop_duplicates("trip_id")
    events = stop_times.merge(trip_cols, on="trip_id", how="inner")
    arrival = events["arrival_time"].map(parse_gtfs_time)
    departure = events["departure_time"].map(parse_gtfs_time)
    events["arrival_sec"] = arrival.fillna(departure)
    events["departure_sec"] = departure.fillna(arrival)
    return events[["stop_id", "route_id", "arrival_sec", "departure_sec"]]


_StopRouteArrays = dict[tuple[str, str], np.ndarray]


def index_stop_events(
    events: pd.DataFrame,
) -> tuple[dict[str, set[str]], _StopRouteArrays, _StopRouteArrays]:
    """Build lookups of routes-per-stop and sorted time arrays per (stop, route).

    Args:
        events: Output of :func:`build_stop_events`.

    Returns:
        A tuple ``(serves, arrivals, departures)`` where ``serves`` maps a
        ``stop_id`` to the set of routes serving it, ``arrivals`` maps
        ``(stop_id, route_id)`` to a sorted array of scheduled arrival seconds,
        and ``departures`` maps it to a sorted array of departure seconds.
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


@dataclass
class PairStats:
    """Accumulated proximity statistics for one directed (route, partner) pair.

    Attributes:
        shared_stops: The route's stops that have a partner stop within
            walking distance.
        min_distance_m: Closest observed stop-to-stop walk distance in meters.
        timed_feasible: True when at least one scheduled visit of the route
            can catch a partner departure within the wait window.
        min_wait_seconds: Shortest qualifying wait observed, or ``None``.
    """

    shared_stops: set[str] = field(default_factory=set)
    min_distance_m: float = float("inf")
    timed_feasible: bool = False
    min_wait_seconds: Optional[float] = None


def compute_pair_stats(
    stops_gdf: gpd.GeoDataFrame,
    serves: Mapping[str, set[str]],
    radius_m: float,
    arrivals: Optional[Mapping[tuple[str, str], np.ndarray]] = None,
    departures: Optional[Mapping[tuple[str, str], np.ndarray]] = None,
    check_times: bool = False,
    walk_speed_mph: float = WALK_SPEED_MPH,
    max_wait_minutes: float = MAX_ALTERNATIVE_WAIT_MINUTES,
) -> dict[tuple[str, str], PairStats]:
    """Find every directed route pair whose stops come within walking distance.

    For each pair of stops within *radius_m* (a stop is its own neighbor, so
    two routes serving the same stop always pair), every route at the first
    stop is paired with every other route at the second stop. When
    *check_times* is on, each stop pair is also tested for a catchable
    connection: some scheduled arrival of the first route followed by a
    departure of the second route after the walk and within the wait window.

    Args:
        stops_gdf: Projected stops with ``stop_id``, ``x``, ``y`` (meters).
        serves: Mapping of stop_id → routes serving it (this schedule).
        radius_m: Walking distance in meters.
        arrivals: Sorted arrival arrays per (stop, route); required when
            *check_times* is True.
        departures: Sorted departure arrays per (stop, route); required when
            *check_times* is True.
        check_times: Whether to evaluate the timed-alternative feasibility.
        walk_speed_mph: Assumed walking speed for stop-to-stop walk time.
        max_wait_minutes: Longest wait a rider accepts for an alternative.

    Returns:
        Mapping of ``(route_id, partner_route_id)`` → :class:`PairStats`.
    """
    sub = stops_gdf.loc[stops_gdf["stop_id"].isin(set(serves))]
    if sub.empty:
        return {}
    coords = np.column_stack([sub["x"].to_numpy(), sub["y"].to_numpy()])
    stop_ids = sub["stop_id"].to_numpy()
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius_m, p=2.0)

    walk_mps = walk_speed_mph * 1609.34 / 3600.0
    max_wait_sec = max_wait_minutes * 60.0

    pairs: dict[tuple[str, str], PairStats] = {}
    for i, neigh in enumerate(neighbors):
        stop_a = stop_ids[i]
        routes_a = serves.get(stop_a)
        if not routes_a:
            continue
        for j in neigh:
            stop_b = stop_ids[j]
            routes_b = serves.get(stop_b)
            if not routes_b:
                continue
            distance = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
            for route_a in routes_a:
                for route_b in routes_b:
                    if route_b == route_a:
                        continue
                    stats = pairs.setdefault((route_a, route_b), PairStats())
                    stats.shared_stops.add(stop_a)
                    stats.min_distance_m = min(stats.min_distance_m, distance)
                    if not check_times or arrivals is None or departures is None:
                        continue
                    arr = arrivals.get((stop_a, route_a))
                    dep = departures.get((stop_b, route_b))
                    if arr is None or dep is None:
                        continue
                    walk_sec = distance / walk_mps if walk_mps > 0 else 0.0
                    feasible, wait = has_timed_connection(arr, dep, walk_sec, max_wait_sec)
                    if feasible and wait is not None:
                        stats.timed_feasible = True
                        if stats.min_wait_seconds is None or wait < stats.min_wait_seconds:
                            stats.min_wait_seconds = wait
    return pairs


def compute_service_areas(
    stops_gdf: gpd.GeoDataFrame,
    route_stops: Mapping[str, set[str]],
    radius_m: float,
) -> dict[str, tuple[float, float, float]]:
    """Compute each route's total, solo, and shared service area.

    A route's service area is the union of *radius_m* buffers around its
    stops. The solo portion is what remains after subtracting every other
    route's service area; shared is the difference.

    Args:
        stops_gdf: Projected stops with ``stop_id`` and point geometry in a
            metric CRS.
        route_stops: Mapping of route_id → the stop_ids it serves.
        radius_m: Buffer radius in meters.

    Returns:
        Mapping of route_id → ``(total_sqmi, solo_sqmi, shared_sqmi)``.
    """
    geoms: dict[str, Any] = {}
    for route_id, stop_ids in route_stops.items():
        subset = stops_gdf.loc[stops_gdf["stop_id"].isin(stop_ids)]
        geoms[route_id] = (
            subset.geometry.buffer(radius_m).union_all() if not subset.empty else Polygon()
        )

    route_ids = list(geoms)
    geom_list = [geoms[r] for r in route_ids]
    tree = STRtree(geom_list)

    areas: dict[str, tuple[float, float, float]] = {}
    for i, route_id in enumerate(route_ids):
        geom = geom_list[i]
        if geom.is_empty:
            areas[route_id] = (0.0, 0.0, 0.0)
            continue
        others = [geom_list[int(j)] for j in tree.query(geom) if int(j) != i]
        overlap = union_all(others) if others else Polygon()
        solo = geom.difference(overlap)
        total_sqmi = geom.area / _SQM_PER_SQMI
        solo_sqmi = solo.area / _SQM_PER_SQMI
        areas[route_id] = (total_sqmi, solo_sqmi, max(total_sqmi - solo_sqmi, 0.0))
    return areas


def build_route_lookup(routes: pd.DataFrame) -> pd.DataFrame:
    """Index routes by route_id with display names and a compact label.

    Args:
        routes: Parsed *routes.txt*.

    Returns:
        A frame indexed by ``route_id`` with ``route_short_name``,
        ``route_long_name``, and ``route_label`` (short name, else route_id).
    """
    lookup = routes.drop_duplicates("route_id").copy()
    for col in ("route_short_name", "route_long_name"):
        if col not in lookup.columns:
            lookup[col] = ""
        lookup[col] = lookup[col].fillna("").astype(str).str.strip()
    short = lookup["route_short_name"]
    lookup["route_label"] = short.where(short != "", lookup["route_id"].astype(str))
    return lookup.set_index("route_id")


def build_profile_tables(
    schedule: str,
    serves: Mapping[str, set[str]],
    tp_serves: Mapping[str, set[str]],
    pair_stats: Mapping[tuple[str, str], PairStats],
    tp_pair_stats: Mapping[tuple[str, str], PairStats],
    areas: Mapping[str, tuple[float, float, float]],
    route_lookup: pd.DataFrame,
    check_times: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Assemble the summary and partner-detail rows for one schedule.

    Args:
        schedule: The schedule-calendar label (e.g. ``"Weekday"``).
        serves: Mapping of stop_id → routes serving it this schedule.
        tp_serves: Same mapping restricted to timepoint stops.
        pair_stats: All-stop pair statistics from :func:`compute_pair_stats`.
        tp_pair_stats: Timepoint-only pair statistics (spatial only).
        areas: Output of :func:`compute_service_areas`.
        route_lookup: Output of :func:`build_route_lookup`.
        check_times: Whether the timed-alternative columns carry values.

    Returns:
        A ``(summary_rows, detail_rows)`` tuple of row dicts, routes sorted
        by display label.
    """
    route_stops: dict[str, set[str]] = {}
    for stop_id, route_ids in serves.items():
        for route_id in route_ids:
            route_stops.setdefault(route_id, set()).add(stop_id)
    tp_route_stops: dict[str, set[str]] = {}
    for stop_id, route_ids in tp_serves.items():
        for route_id in route_ids:
            tp_route_stops.setdefault(route_id, set()).add(stop_id)

    partners: dict[str, dict[str, PairStats]] = {}
    for (route_a, route_b), stats in pair_stats.items():
        partners.setdefault(route_a, {})[route_b] = stats
    tp_partners: dict[str, dict[str, PairStats]] = {}
    for (route_a, route_b), stats in tp_pair_stats.items():
        tp_partners.setdefault(route_a, {})[route_b] = stats

    def _label(route_id: str) -> str:
        if route_id in route_lookup.index:
            return str(route_lookup.loc[route_id, "route_label"])
        return str(route_id)

    def _names(route_id: str) -> tuple[str, str]:
        if route_id in route_lookup.index:
            row = route_lookup.loc[route_id]
            return str(row["route_short_name"]), str(row["route_long_name"])
        return "", ""

    def _pct(part: int, whole: int) -> float:
        return round(100.0 * part / whole, 1) if whole else 0.0

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for route_id in sorted(route_stops, key=_label):
        short_name, long_name = _names(route_id)
        n_stops = len(route_stops[route_id])
        route_partners = partners.get(route_id, {})
        shared_stops: set[str] = set()
        for stats in route_partners.values():
            shared_stops |= stats.shared_stops
        n_shared = len(shared_stops)

        n_timepoints = len(tp_route_stops.get(route_id, set()))
        route_tp_partners = tp_partners.get(route_id, {})
        shared_timepoints: set[str] = set()
        for stats in route_tp_partners.values():
            shared_timepoints |= stats.shared_stops
        n_tp_shared = len(shared_timepoints)

        walk_ids = sorted(route_partners, key=_label)
        timed_ids = [p for p in walk_ids if route_partners[p].timed_feasible]
        total_sqmi, solo_sqmi, shared_sqmi = areas.get(route_id, (0.0, 0.0, 0.0))

        summary_rows.append(
            {
                "schedule": schedule,
                "route_id": route_id,
                "route_short_name": short_name,
                "route_long_name": long_name,
                "n_stops": n_stops,
                "n_shared_stops": n_shared,
                "n_solo_stops": n_stops - n_shared,
                "pct_stops_shared": _pct(n_shared, n_stops),
                "n_routes_within_walk": len(walk_ids),
                "routes_within_walk": ", ".join(_label(p) for p in walk_ids),
                "n_timed_alternative_routes": len(timed_ids) if check_times else None,
                "timed_alternative_routes": (
                    ", ".join(_label(p) for p in timed_ids) if check_times else None
                ),
                "n_timepoints": n_timepoints,
                "n_shared_timepoints": n_tp_shared,
                "n_solo_timepoints": n_timepoints - n_tp_shared,
                "pct_timepoints_shared": _pct(n_tp_shared, n_timepoints),
                "n_timepoint_partner_routes": len(route_tp_partners),
                "timepoint_partner_routes": ", ".join(
                    _label(p) for p in sorted(route_tp_partners, key=_label)
                ),
                "service_area_sqmi": round(total_sqmi, 3),
                "solo_area_sqmi": round(solo_sqmi, 3),
                "shared_area_sqmi": round(shared_sqmi, 3),
                "pct_area_shared": (
                    round(100.0 * shared_sqmi / total_sqmi, 1) if total_sqmi else 0.0
                ),
            }
        )

        for partner_id in walk_ids:
            stats = route_partners[partner_id]
            partner_short, _ = _names(partner_id)
            tp_stats = route_tp_partners.get(partner_id)
            wait = stats.min_wait_seconds
            detail_rows.append(
                {
                    "schedule": schedule,
                    "route_id": route_id,
                    "route_short_name": short_name,
                    "partner_route_id": partner_id,
                    "partner_route_short_name": partner_short,
                    "n_shared_stops": len(stats.shared_stops),
                    "n_shared_timepoints": len(tp_stats.shared_stops) if tp_stats else 0,
                    "nearest_walk_distance_ft": round(stats.min_distance_m * _FEET_PER_METER),
                    "timed_alternative": stats.timed_feasible if check_times else None,
                    "min_alternative_wait_min": (
                        round(wait / 60.0, 1) if check_times and wait is not None else None
                    ),
                }
            )

    return summary_rows, detail_rows


# =============================================================================
# SMALL HELPERS
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


def filter_platform_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """Keep only platform stops (``location_type`` 0 or blank).

    Args:
        stops: Parsed *stops.txt*.

    Returns:
        The filtered stops table (unchanged when the column is absent).
    """
    if "location_type" not in stops.columns:
        return stops
    loc = stops["location_type"].fillna("").astype(str).str.strip()
    kept = stops.loc[loc.isin(("", "0"))]
    dropped = len(stops) - len(kept)
    if dropped:
        logging.info("Excluded %d non-platform stop record(s) (stations, entrances, ...).", dropped)
    return kept


# =============================================================================
# OUTPUT / RUN LOG
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


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


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    The sidecar is named after :data:`SUMMARY_FILENAME` (same stem,
    ``_runlog.txt`` suffix) per CONTRIBUTING.md.

    Args:
        output_dir: Directory that received this run's output CSVs.
        summary_lines: Human-readable lines describing what the run did.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / f"{Path(SUMMARY_FILENAME).stem}_runlog.txt"

    source_file = resolve_source_file()
    if source_file is None:
        config_text = "(config block unavailable: interactive session, no __file__ on disk)"
        source_display = "<interactive>"
    else:
        try:
            config_text = extract_config_block(source_file)
        except (OSError, ValueError) as exc:
            logging.error("Could not extract config block for run log: %s", exc)
            return False
        source_display = str(source_file)

    lines: List[str] = [
        "=" * 72,
        "ROUTE REDUNDANCY PROFILE RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
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
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# REUSABLE FUNCTIONS (canonical copies from utils/ — do not edit here)
# =============================================================================


def load_gtfs_data(
    gtfs_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_path: Absolute or relative path to the folder containing the
            GTFS feed, or to a ``.zip`` archive of it — the form GTFS
            producers and most open-data portals distribute feeds in. Zip
            members may sit at the archive root or nested one level inside
            a single wrapper folder; both layouts are handled.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.
        logger: Logger for progress messages. Defaults to this module's
            logger (``logging.getLogger(__name__)``) rather than the root
            logger, so callers keep control of handler configuration.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Path missing, one of *files* not present in the feed, or
            an OS-level failure while reading a file.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, a requested file matches more than one location inside
            the zip, a file is empty, or the CSV parser fails.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    if not os.path.exists(gtfs_path):
        raise OSError(f"The path '{gtfs_path}' does not exist.")

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

    is_zip = os.path.isfile(gtfs_path) and gtfs_path.lower().endswith(".zip")
    if not is_zip and not os.path.isdir(gtfs_path):
        raise ValueError(f"'{gtfs_path}' is neither a directory nor a .zip file.")

    archive: zipfile.ZipFile | None = None
    members_by_name: dict[str, list[str]] = {}
    if is_zip:
        try:
            archive = zipfile.ZipFile(gtfs_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"'{gtfs_path}' is not a valid zip archive.") from exc
        for name in archive.namelist():
            members_by_name.setdefault(os.path.basename(name), []).append(name)

    try:
        missing: list[str] = []
        ambiguous: list[str] = []
        resolved: dict[str, str] = {}
        for file_name in files:
            if archive is None:
                if not os.path.exists(os.path.join(gtfs_path, file_name)):
                    missing.append(file_name)
                continue
            candidates = members_by_name.get(file_name, [])
            if not candidates:
                missing.append(file_name)
            elif len(candidates) > 1:
                ambiguous.append(file_name)
            else:
                resolved[file_name] = candidates[0]

        if ambiguous:
            raise ValueError(
                f"Ambiguous GTFS files in '{gtfs_path}' (found in multiple "
                f"locations): {', '.join(ambiguous)}"
            )
        if missing:
            raise OSError(f"Missing GTFS files in '{gtfs_path}': {', '.join(missing)}")

        data: dict[str, pd.DataFrame] = {}
        for file_name in files:
            key = file_name.replace(".txt", "")
            try:
                if archive is None:
                    df = pd.read_csv(
                        os.path.join(gtfs_path, file_name), dtype=dtype, low_memory=False
                    )
                else:
                    with archive.open(resolved[file_name]) as handle:
                        df = pd.read_csv(handle, dtype=dtype, low_memory=False)
                data[key] = df
                log.info("Loaded %s (%d records).", file_name, len(df))

            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"File '{file_name}' in '{gtfs_path}' is empty.") from exc

            except pd.errors.ParserError as exc:
                raise ValueError(f"Parser error in '{file_name}' in '{gtfs_path}': {exc}") from exc

        return data
    finally:
        if archive is not None:
            archive.close()


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


def classify_service_ids(
    active_dates: Mapping[str, set[dt.date]],
    holiday_max_days_per_year: float = 25.0,
    dow_share: float = 0.80,
) -> dict[str, set[str]]:
    """Classify each service_id by its real active-date pattern.

    A service operating at or below *holiday_max_days_per_year* is labelled
    ``Holiday`` — this catches holiday-only services regardless of what
    their day-of-week columns claim (scheduling-software exports often stamp
    a weekday pattern on a service that really runs five days a year).
    Otherwise, day-of-week shares of the active dates determine the labels.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        holiday_max_days_per_year: Services active at or below this annual
            rate are labelled ``Holiday``.
        dow_share: Minimum fraction of active dates on a day-of-week bucket
            (Mon–Fri, Saturday, Sunday) to earn that bucket's label.

    Returns:
        Mapping of ``service_id`` to a set of labels drawn from
        ``{"Weekday", "Saturday", "Sunday", "Holiday"}``. A service with no
        active dates maps to an empty set.
    """
    result: dict[str, set[str]] = {}
    for sid, dates in active_dates.items():
        if not dates:
            result[sid] = set()
            logging.info("Service %s: empty (0 active dates).", sid)
            continue
        span_days = (max(dates) - min(dates)).days + 1
        per_year = len(dates) / max(span_days / 365.25, 0.1)
        labels: set[str] = set()
        if per_year <= holiday_max_days_per_year:
            labels.add("Holiday")
        else:
            n = len(dates)
            wd = sum(1 for d in dates if d.weekday() < 5)
            sat = sum(1 for d in dates if d.weekday() == 5)
            sun = sum(1 for d in dates if d.weekday() == 6)
            if wd / n >= dow_share:
                labels.add("Weekday")
            if sat / n >= dow_share:
                labels.add("Saturday")
            if sun / n >= dow_share:
                labels.add("Sunday")
            if not labels:
                if wd > 0:
                    labels.add("Weekday")
                if sat > 0:
                    labels.add("Saturday")
                if sun > 0:
                    labels.add("Sunday")
        result[sid] = labels
        logging.info(
            "Service %s -> %s (%d dates, %.1f/yr).",
            sid,
            sorted(labels),
            len(dates),
            per_year,
        )
    return result


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


# =============================================================================
# ORCHESTRATION
# =============================================================================


def run(
    gtfs_path: str,
    output_dir: Path,
    walk_distance: float = WALK_DISTANCE,
    walk_distance_unit: str = WALK_DISTANCE_UNIT,
    check_times: bool = ENABLE_TIME_CHECK,
    walk_speed_mph: float = WALK_SPEED_MPH,
    max_wait_minutes: float = MAX_ALTERNATIVE_WAIT_MINUTES,
    service_labels: Sequence[str] = (),
    routes_include: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
    service_ids: Sequence[str] = (),
    projected_crs: str = PROJECTED_CRS,
    write_detail: bool = WRITE_DETAIL,
) -> pd.DataFrame:
    """Compute the redundancy profile for every route and write all artifacts.

    Args:
        gtfs_path: GTFS feed folder or ``.zip`` archive.
        output_dir: Directory receiving the CSVs and the run log.
        walk_distance: Walking distance defining "shared" (also the
            service-area buffer radius).
        walk_distance_unit: Unit of *walk_distance*: miles, feet, or meters.
        check_times: Evaluate the timed-alternative check.
        walk_speed_mph: Assumed walking speed for the time check.
        max_wait_minutes: Longest acceptable wait for a timed alternative.
        service_labels: Schedule labels to profile; empty profiles all.
        routes_include: Routes to keep (route_id or route_short_name).
        routes_exclude: Routes to drop (route_id or route_short_name).
        service_ids: service_ids to keep; empty keeps all.
        projected_crs: Metric CRS for distance and area math.
        write_detail: Also write the per-pair partner detail CSV.

    Returns:
        The summary DataFrame (also written to disk).

    Raises:
        OSError: The feed path or a required file is missing.
        ValueError: The feed fails column validation or cannot be parsed.
        RuntimeError: The run log could not be written while
            ``REQUIRE_RUN_LOG`` is True.
    """
    data = load_gtfs_data(
        gtfs_path, files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
    )
    validate_required_columns(data)

    calendar: Optional[pd.DataFrame] = None
    calendar_dates: Optional[pd.DataFrame] = None
    for name in ("calendar", "calendar_dates"):
        try:
            frame = load_gtfs_data(gtfs_path, files=(f"{name}.txt",))[name]
        except (OSError, ValueError):
            frame = None
            logging.info("No usable %s.txt in the feed.", name)
        if name == "calendar":
            calendar = frame
        else:
            calendar_dates = frame

    trips = filter_trips(data["trips"], data["routes"], service_ids, routes_include, routes_exclude)
    route_lookup = build_route_lookup(data["routes"])

    stops = filter_platform_stops(data["stops"])
    stops_gdf = project_stops(stops, projected_crs)
    known_stop_ids = set(stops_gdf["stop_id"])

    radius_m = distance_to_meters(walk_distance, walk_distance_unit)
    logging.info(
        "Walking distance: %.0f m. Timed-alternative check: %s (max wait %.0f min).",
        radius_m,
        "on" if check_times else "off",
        max_wait_minutes,
    )

    schedules = build_schedule_service_ids(calendar, calendar_dates, trips)
    if service_labels:
        wanted = {str(s).strip().lower() for s in service_labels}
        unknown = wanted - {label.lower() for label in schedules}
        if unknown:
            logging.warning(
                "SERVICE_LABELS entr%s not found in this feed: %s. Available: %s.",
                "y" if len(unknown) == 1 else "ies",
                ", ".join(sorted(unknown)),
                ", ".join(sorted(schedules)),
            )
        schedules = {label: ids for label, ids in schedules.items() if label.lower() in wanted}
    if not schedules:
        raise ValueError("No schedule calendars left to profile — check SERVICE_LABELS.")

    ordered_labels = sorted(schedules, key=lambda s: (_LABEL_ORDER.get(s, 9), s))
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for label in ordered_labels:
        label_trips = trips.loc[trips["service_id"].astype(str).isin(schedules[label])]
        if label_trips.empty:
            logging.info("Schedule '%s': no trips after filtering — skipped.", label)
            continue
        label_stop_times = data["stop_times"].loc[
            data["stop_times"]["trip_id"].isin(set(label_trips["trip_id"]))
        ]

        events = build_stop_events(label_stop_times, label_trips)
        n_unknown = int((~events["stop_id"].isin(known_stop_ids)).sum())
        if n_unknown:
            logging.warning(
                "Schedule '%s': dropped %d stop event(s) at stops with no usable "
                "coordinates or filtered location types.",
                label,
                n_unknown,
            )
            events = events.loc[events["stop_id"].isin(known_stop_ids)]
        if events.empty:
            logging.info("Schedule '%s': no usable stop events — skipped.", label)
            continue

        serves, arrivals, departures = index_stop_events(events)

        timepoints = select_timepoint_rows(label_stop_times)
        tp_events = timepoints.merge(
            label_trips[["trip_id", "route_id"]].drop_duplicates("trip_id"),
            on="trip_id",
            how="inner",
        )
        tp_serves: dict[str, set[str]] = {}
        for stop_id, route_id in zip(tp_events["stop_id"], tp_events["route_id"]):
            if stop_id in known_stop_ids:
                tp_serves.setdefault(stop_id, set()).add(route_id)

        pair_stats = compute_pair_stats(
            stops_gdf,
            serves,
            radius_m,
            arrivals=arrivals,
            departures=departures,
            check_times=check_times,
            walk_speed_mph=walk_speed_mph,
            max_wait_minutes=max_wait_minutes,
        )
        tp_pair_stats = compute_pair_stats(stops_gdf, tp_serves, radius_m, check_times=False)

        route_stops: dict[str, set[str]] = {}
        for stop_id, route_ids in serves.items():
            for route_id in route_ids:
                route_stops.setdefault(route_id, set()).add(stop_id)
        areas = compute_service_areas(stops_gdf, route_stops, radius_m)

        label_summary, label_detail = build_profile_tables(
            label, serves, tp_serves, pair_stats, tp_pair_stats, areas, route_lookup, check_times
        )
        summary_rows.extend(label_summary)
        detail_rows.extend(label_detail)
        logging.info(
            "Schedule '%s': profiled %d route(s) across %d stop(s).",
            label,
            len(label_summary),
            len(serves),
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    detail = pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS)

    ensure_dir(output_dir)
    summary_path = output_dir / SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    logging.info("Wrote %d route profile row(s) → %s", len(summary), summary_path)
    if write_detail:
        detail_path = output_dir / DETAIL_FILENAME
        detail.to_csv(detail_path, index=False)
        logging.info("Wrote %d partner pair row(s) → %s", len(detail), detail_path)

    summary_lines = [
        f"GTFS feed:            {gtfs_path}",
        f"Walking distance:     {walk_distance:g} {walk_distance_unit} ({radius_m:.0f} m)",
        f"Timed alternatives:   "
        f"{f'on (max wait {max_wait_minutes:g} min)' if check_times else 'off'}",
        f"Schedules profiled:   {', '.join(ordered_labels)}",
        f"Route profile rows:   {len(summary)}",
        f"Partner pair rows:    {len(detail)}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return summary


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile how redundant each route is — shared/solo stops, routes within "
            "walking distance (with an optional timed-alternative check), shared "
            "timepoints, and solo/shared service area — per schedule calendar."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs",
        default=GTFS_PATH,
        help="GTFS feed: a folder of .txt files or a .zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for the output CSVs and the run log.",
    )
    parser.add_argument(
        "--walk-distance",
        type=float,
        default=WALK_DISTANCE,
        help="Walking distance that defines 'shared' (also the service-area buffer).",
    )
    parser.add_argument(
        "--walk-distance-unit",
        default=WALK_DISTANCE_UNIT,
        choices=("miles", "feet", "meters"),
        help="Unit of --walk-distance.",
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=MAX_ALTERNATIVE_WAIT_MINUTES,
        help="Longest wait accepted by the timed-alternative check.",
    )
    parser.add_argument(
        "--no-time-check",
        action="store_true",
        default=not ENABLE_TIME_CHECK,
        help="Skip the timed-alternative check; report spatial proximity only.",
    )
    parser.add_argument(
        "--service-labels",
        nargs="*",
        default=SERVICE_LABELS,
        help="Schedule calendars to profile (Weekday, Saturday, ...); omit for all.",
    )
    parser.add_argument(
        "--routes-include",
        nargs="*",
        default=FILTER_IN_ROUTES,
        help="Routes to keep, matched on route_id or route_short_name; omit to keep all.",
    )
    parser.add_argument(
        "--routes-exclude",
        nargs="*",
        default=FILTER_OUT_ROUTES,
        help="Routes to drop, matched on route_id or route_short_name.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Args:
        argv: Optional explicit argument list (used by tests); ``None``
            reads ``sys.argv`` outside notebook kernels.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    if args.gtfs == r"Path\To\Your\GTFS_Folder" or args.output_dir == r"Path\To\Your\Output_Folder":
        logging.warning(
            "GTFS_PATH and/or OUTPUT_DIR are still set to placeholder values. "
            "Update the CONFIGURATION section (or pass --gtfs / --output-dir) before running."
        )
        return 2

    try:
        run(
            gtfs_path=args.gtfs,
            output_dir=Path(args.output_dir).expanduser(),
            walk_distance=args.walk_distance,
            walk_distance_unit=args.walk_distance_unit,
            check_times=not args.no_time_check,
            max_wait_minutes=args.max_wait_minutes,
            service_labels=args.service_labels,
            routes_include=args.routes_include,
            routes_exclude=args.routes_exclude,
            service_ids=FILTER_SERVICE_IDS,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
