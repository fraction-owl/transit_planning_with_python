"""Flag vehicle blocks scheduled at the same stop at the same time.

This is the quick, GTFS-only "easy version" of bus conflict checking. Each
block (or blockless trip) is treated as present at a stop from its scheduled
arrival through its scheduled departure, and every pair of distinct blocks
whose presence at one stop overlaps within the same operating schedule is
flagged. Service_ids that actually run on the same calendar dates (per
calendar.txt / calendar_dates.txt) are checked together, so a school tripper
under its own service_id still conflicts with the regular weekday service.
An optional companion check flags trips *within* one block that overlap each
other in time — a block scheduled to be in two places at once.

For operational depth — minute-by-minute statuses, layover and deadhead
inference, bay capacities and cluster semantics — use the two-step pipeline
instead: ``scripts.gtfs_exports.block_status_timeline_exporter`` (Step 1)
feeding ``scripts.facilities_tools.bay_usage_analyzer`` (Step 2). This
checker trades that fidelity for a zero-setup screen that runs straight off
a feed: no cluster definitions, no intermediate workbooks.

Inputs
------
- A GTFS feed (folder or .zip) with stops.txt, routes.txt, trips.txt, and
  stop_times.txt. calendar.txt / calendar_dates.txt, when present, group
  service_ids that operate on the same dates into schedules; without them
  each service_id is checked on its own. frequencies.txt-based trips are
  checked only at their template times (repetitions are not expanded).

Outputs
-------
- block_conflict_flags.csv — one row per pair of blocks scheduled at the
  same stop at overlapping times (schedule, stop, blocks, trips, routes,
  scheduled windows, overlap span and minutes).
- block_conflict_summary.csv — flagged stops rolled up with conflict-pair
  counts, blocks involved, and the peak number of simultaneous buses.
- block_self_overlap_flags.csv — trips within one block whose scheduled
  spans overlap each other (the block is in two places at once).
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
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
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"  # folder of .txt files or a .zip
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Minimum shared scheduled minutes for a pair of blocks at one stop to be
# flagged. Two blocks share N minutes when their [arrival, departure] windows
# overlap for N clock minutes, counting both endpoints — so 1 flags any shared
# minute, including exact hand-offs where one block departs the minute the
# other arrives. Raise to 2+ to require genuinely simultaneous presence.
MIN_OVERLAP_MINUTES: int = 1

# Check service_ids that operate on the same calendar dates together (one
# combined schedule per unique set of co-active service_ids). Requires
# calendar.txt and/or calendar_dates.txt; when False — or when the feed has
# neither file — each service_id is checked on its own and conflicts between
# blocks of different service_ids are NOT detected.
COMBINE_SAME_DAY_SERVICES: bool = True

# Compare stops by parent_station (where stops.txt provides one) instead of
# stop_id, so two buses at different bays of one station count as sharing a
# place. Off by default: distinct bays usually hold distinct buses by design.
GROUP_BY_PARENT_STATION: bool = False

# Also report trips within one block whose scheduled spans overlap each other
# (beyond a single hand-off minute) — a block scheduled in two places at once.
CHECK_BLOCK_SELF_OVERLAPS: bool = True

# Analyze exactly these service_ids together as ONE schedule (e.g. the ids
# you know run on a given day). Empty = automatic grouping per the flags above.
FILTER_SERVICE_IDS: list[str] = []

# Route filters, matched against route_id OR route_short_name. Empty = all.
FILTER_IN_ROUTES: list[str] = []
FILTER_OUT_ROUTES: list[str] = []

# Output filenames (written inside OUTPUT_DIR).
DETAIL_FILENAME: str = r"block_conflict_flags.csv"
SUMMARY_FILENAME: str = r"block_conflict_summary.csv"
SELF_OVERLAP_FILENAME: str = r"block_self_overlap_flags.csv"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output is never
# left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Trips of one block may hand off back-to-back (one span ends the minute the
# next begins); only spans sharing at least this many minutes are self-overlaps.
SELF_OVERLAP_MIN_MINUTES: int = 2

# Columns required in each loaded GTFS table (headers only; values may be blank).
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "stops": {"stop_id"},
    "routes": {"route_id"},
    "trips": {"trip_id", "route_id", "service_id"},
    "stop_times": {"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"},
}

# Column order of the per-visit event table built by parse_stop_events.
EVENT_COLUMNS: List[str] = [
    "trip_id",
    "vehicle",
    "is_block",
    "service_id",
    "route_name",
    "stop_id",
    "place_id",
    "place_name",
    "arr_min",
    "dep_min",
]

# Column order of the merged presence intervals built by merge_presence_intervals.
INTERVAL_COLUMNS: List[str] = [
    "place_id",
    "place_name",
    "vehicle",
    "arr_min",
    "dep_min",
    "trip_ids",
    "stop_ids",
    "route_names",
    "service_ids",
]

# Column order of the raw conflict pairs built by find_conflict_pairs
# (times in minutes; formatted to HH:MM at assembly).
PAIR_COLUMNS: List[str] = [
    "place_id",
    "place_name",
    "block_a",
    "block_b",
    "routes_a",
    "routes_b",
    "trips_a",
    "trips_b",
    "stops_a",
    "stops_b",
    "services_a",
    "services_b",
    "arrive_a",
    "depart_a",
    "arrive_b",
    "depart_b",
    "overlap_start",
    "overlap_end",
    "overlap_minutes",
]

# Column order of the per-place concurrency stats built by find_conflict_pairs.
PLACE_STAT_COLUMNS: List[str] = ["place_id", "place_name", "max_simultaneous", "peak_minute"]

# Column order of the raw self-overlap pairs built by find_block_self_overlaps.
SELF_PAIR_COLUMNS: List[str] = [
    "block_id",
    "trip_a",
    "route_a",
    "service_a",
    "start_a",
    "end_a",
    "trip_b",
    "route_b",
    "service_b",
    "start_b",
    "end_b",
    "overlap_minutes",
]

# Column order of the written detail CSV.
DETAIL_COLUMNS: List[str] = ["schedule", "service_group", "n_dates", *PAIR_COLUMNS]

# Column order of the written summary CSV.
SUMMARY_COLUMNS: List[str] = [
    "place_id",
    "place_name",
    "schedules",
    "n_conflict_pairs",
    "n_blocks",
    "max_simultaneous_buses",
    "peak_time",
    "max_overlap_minutes",
]

# Column order of the written self-overlap CSV.
SELF_OVERLAP_COLUMNS: List[str] = ["schedule", "service_group", "n_dates", *SELF_PAIR_COLUMNS]

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


def label_service_group(dates: set[dt.date]) -> str:
    """Name an operating schedule after the days of week it runs.

    Args:
        dates: The calendar dates on which the schedule's service_ids are
            all active together.

    Returns:
        ``"Weekday"``, ``"Saturday"``, ``"Sunday"``, ``"Weekend"``,
        ``"Daily"``, or ``"Mixed"`` — from the actual day-of-week pattern of
        *dates* rather than any calendar.txt column, so a school service
        running only some weekdays still labels as ``"Weekday"``. An empty
        set yields ``"Empty"``.
    """
    dows = {d.weekday() for d in dates}
    if not dows:
        return "Empty"
    if dows <= {0, 1, 2, 3, 4}:
        return "Weekday"
    if dows == {5}:
        return "Saturday"
    if dows == {6}:
        return "Sunday"
    if dows <= {5, 6}:
        return "Weekend"
    if dows == set(range(7)):
        return "Daily"
    return "Mixed"


def build_service_groups(
    trips: pd.DataFrame,
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    explicit_service_ids: Sequence[str] = (),
    combine_same_day_services: bool = True,
) -> list[dict[str, Any]]:
    """Group the feed's service_ids into the operating schedules to check.

    Conflicts are physical events on a single operating day, so blocks are
    only compared within a set of service_ids that actually run together.
    Explicit ids trump everything; otherwise each unique combination of
    co-active service_ids (per the expanded calendars) becomes one schedule,
    and service_ids with no resolvable dates fall back to being checked on
    their own.

    Args:
        trips: The (already filtered) *trips.txt* table — only service_ids
            appearing here are grouped.
        calendar: Parsed *calendar.txt*, or ``None`` when the feed has none.
        calendar_dates: Parsed *calendar_dates.txt*, or ``None``.
        explicit_service_ids: When non-empty, analyze exactly these ids
            together as one schedule and skip the calendar math.
        combine_same_day_services: When False, skip the calendar math and
            check each service_id on its own.

    Returns:
        One dict per schedule: ``label`` (day-type name or raw service_id),
        ``service_ids`` (frozenset of ids checked together), and ``n_dates``
        (dates this exact combination operates, or ``None`` when unknown).
    """
    used = sorted({str(s).strip() for s in trips["service_id"].dropna()})
    if explicit_service_ids:
        wanted = frozenset(str(s).strip() for s in explicit_service_ids)
        return [{"label": "Selected services", "service_ids": wanted, "n_dates": None}]

    groups: list[dict[str, Any]] = []
    leftovers: list[str] = list(used)
    if combine_same_day_services and (calendar is not None or calendar_dates is not None):
        active = expand_service_active_dates(calendar, calendar_dates)
        used_set = set(used)
        dates_by_set: dict[frozenset[str], set[dt.date]] = {}
        all_dates = sorted({d for sid in used for d in active.get(sid, set())})
        for day in all_dates:
            sids = frozenset(service_ids_active_on(active, day) & used_set)
            if sids:
                dates_by_set.setdefault(sids, set()).add(day)
        covered: set[str] = set()
        for sids in sorted(dates_by_set, key=lambda s: min(dates_by_set[s])):
            dates = dates_by_set[sids]
            groups.append(
                {
                    "label": label_service_group(dates),
                    "service_ids": sids,
                    "n_dates": len(dates),
                }
            )
            covered.update(sids)
        leftovers = [sid for sid in used if sid not in covered]
        if leftovers:
            logging.warning(
                "%d service_id(s) have no active dates in calendar.txt/calendar_dates.txt "
                "(%s) — checking each on its own.",
                len(leftovers),
                ", ".join(leftovers),
            )
    elif combine_same_day_services:
        logging.warning(
            "No calendar.txt or calendar_dates.txt — cannot tell which service_ids run "
            "together, so conflicts between blocks of different service_ids are NOT "
            "checked. Set FILTER_SERVICE_IDS to analyze a known combination as one day."
        )

    for sid in leftovers:
        groups.append({"label": sid, "service_ids": frozenset({sid}), "n_dates": None})
    return groups


def parse_stop_events(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    stops: pd.DataFrame,
    service_ids: frozenset[str],
    group_by_parent_station: bool = False,
) -> pd.DataFrame:
    """Build one timed stop-visit row per stop_times entry of a schedule's trips.

    Each row carries the vehicle identity (``block_id``, or ``trip:<trip_id>``
    for blockless trips, which are treated as single-trip vehicles), the
    place identity (``stop_id``, or ``parent_station`` when requested and
    present), and the scheduled window in minutes past midnight. Rows with a
    blank arrival or departure fall back to the other field; rows with
    neither are skipped, and rows departing before they arrive are corrected
    to (earlier, later) with a warning.

    Args:
        stop_times: Parsed *stop_times.txt* (string dtypes).
        trips: Parsed *trips.txt*, already filtered to the run's scope.
        routes: Parsed *routes.txt* (route_short_name lookup).
        stops: Parsed *stops.txt* (stop_name / parent_station lookup).
        service_ids: The schedule's service_ids; only their trips are kept.
        group_by_parent_station: Compare stops by parent_station when one is
            present, instead of stop_id.

    Returns:
        One row per timed stop visit, with the columns listed in
        :data:`EVENT_COLUMNS`.
    """
    kept = trips.loc[trips["service_id"].astype(str).str.strip().isin(service_ids)].copy()
    if kept.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    if "block_id" in kept.columns:
        block = kept["block_id"].fillna("").astype(str).str.strip()
    else:
        block = pd.Series("", index=kept.index)
    kept["vehicle"] = block.where(block != "", "trip:" + kept["trip_id"].astype(str))
    kept["is_block"] = block != ""

    short_names: dict[str, str] = {}
    if "route_short_name" in routes.columns:
        short_names = dict(zip(routes["route_id"], routes["route_short_name"].fillna("")))
    kept["route_name"] = kept["route_id"].map(lambda rid: str(short_names.get(rid) or rid))

    events = stop_times.merge(
        kept[["trip_id", "vehicle", "is_block", "service_id", "route_name"]],
        on="trip_id",
        how="inner",
    )
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    events["arr_min"] = events["arrival_time"].map(parse_time_to_minutes)
    events["dep_min"] = events["departure_time"].map(parse_time_to_minutes)
    events["arr_min"] = events["arr_min"].fillna(events["dep_min"])
    events["dep_min"] = events["dep_min"].fillna(events["arr_min"])
    n_untimed = int(events["arr_min"].isna().sum())
    if n_untimed:
        logging.info("Skipped %d stop_times row(s) with no parseable scheduled time.", n_untimed)
        events = events.loc[events["arr_min"].notna()]
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    backwards = events["dep_min"] < events["arr_min"]
    if backwards.any():
        logging.warning(
            "%d stop_times row(s) depart before they arrive (first example: trip '%s') — "
            "using the earlier time as the arrival.",
            int(backwards.sum()),
            events.loc[backwards].iloc[0]["trip_id"],
        )
        earlier = events[["arr_min", "dep_min"]].min(axis=1)
        later = events[["arr_min", "dep_min"]].max(axis=1)
        events["arr_min"] = earlier
        events["dep_min"] = later

    stop_attrs = stops.copy()
    if "stop_name" not in stop_attrs.columns:
        stop_attrs["stop_name"] = stop_attrs["stop_id"]
    if "parent_station" not in stop_attrs.columns:
        stop_attrs["parent_station"] = ""
    stop_attrs = stop_attrs[["stop_id", "stop_name", "parent_station"]].drop_duplicates("stop_id")
    name_by_stop = dict(zip(stop_attrs["stop_id"], stop_attrs["stop_name"].fillna("")))
    parent_by_stop = dict(zip(stop_attrs["stop_id"], stop_attrs["parent_station"].fillna("")))

    def _place(stop_id: str) -> str:
        parent = str(parent_by_stop.get(stop_id, "") or "").strip()
        if group_by_parent_station and parent:
            return parent
        return str(stop_id)

    def _place_name(row_place: str, row_stop: str) -> str:
        named = str(name_by_stop.get(row_place, "") or "")
        if named:
            return named
        return str(name_by_stop.get(row_stop, "") or row_stop)

    events["place_id"] = events["stop_id"].map(_place)
    events["place_name"] = [
        _place_name(place, stop) for place, stop in zip(events["place_id"], events["stop_id"])
    ]
    events["arr_min"] = events["arr_min"].astype(int)
    events["dep_min"] = events["dep_min"].astype(int)
    return events[EVENT_COLUMNS].reset_index(drop=True)


def merge_presence_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Merge each vehicle's touching stop visits into continuous presence intervals.

    Within one place, visits by the same vehicle that overlap or share a
    minute (e.g. a trip ending and the block's next trip beginning at the
    same stop) become a single interval, so a block is never reported as
    conflicting with itself and duplicate stop_times rows collapse. Trip,
    stop, route, and service identifiers of the merged visits are joined
    with ``+`` in visit order.

    Args:
        events: Output of :func:`parse_stop_events`.

    Returns:
        One row per continuous (place, vehicle) presence interval, with the
        columns listed in :data:`INTERVAL_COLUMNS`.
    """
    if events.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)

    ordered = events.sort_values(["place_id", "vehicle", "arr_min", "dep_min"], kind="mergesort")
    merged: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    current_dep = -1
    for row in ordered.itertuples(index=False):
        arr = int(row.arr_min)
        dep = int(row.dep_min)
        continues_current = (
            current is not None
            and current["place_id"] == row.place_id
            and current["vehicle"] == row.vehicle
            and arr <= current_dep
        )
        if continues_current and current is not None:
            current_dep = max(current_dep, dep)
            current["dep_min"] = current_dep
            for key, value in (
                ("trip_ids", row.trip_id),
                ("stop_ids", row.stop_id),
                ("route_names", row.route_name),
                ("service_ids", row.service_id),
            ):
                if value not in current[key]:
                    current[key].append(value)
            continue
        if current is not None:
            merged.append(current)
        current = {
            "place_id": row.place_id,
            "place_name": row.place_name,
            "vehicle": row.vehicle,
            "arr_min": arr,
            "dep_min": dep,
            "trip_ids": [row.trip_id],
            "stop_ids": [row.stop_id],
            "route_names": [row.route_name],
            "service_ids": [row.service_id],
        }
        current_dep = dep
    if current is not None:
        merged.append(current)

    out = pd.DataFrame(merged)
    for col in ("trip_ids", "stop_ids", "route_names", "service_ids"):
        out[col] = out[col].map("+".join)
    return out[INTERVAL_COLUMNS]


def find_conflict_pairs(
    intervals: pd.DataFrame,
    min_overlap_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep each place for overlapping presence intervals of distinct vehicles.

    Two intervals share ``min(dep) - max(arr) + 1`` scheduled minutes
    (endpoints inclusive), so an exact hand-off — one block departing the
    minute another arrives — counts as 1 shared minute. Pairs sharing at
    least *min_overlap_minutes* are flagged. Per-place concurrency stats
    (peak simultaneous vehicles and when) are computed over all intervals
    regardless of flagging.

    Args:
        intervals: Output of :func:`merge_presence_intervals`.
        min_overlap_minutes: Flag pairs sharing at least this many minutes.

    Returns:
        Tuple of (conflict pairs with :data:`PAIR_COLUMNS`, per-place stats
        with :data:`PLACE_STAT_COLUMNS`); times remain integer minutes.
    """
    pair_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    if intervals.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS), pd.DataFrame(columns=PLACE_STAT_COLUMNS)

    for place_id, place_df in intervals.groupby("place_id", sort=True):
        rows = list(
            place_df.sort_values(["arr_min", "dep_min"], kind="mergesort").itertuples(index=False)
        )
        active: list[Any] = []
        peak_n, peak_minute = 0, int(rows[0].arr_min)
        for row in rows:
            arr = int(row.arr_min)
            dep = int(row.dep_min)
            active = [a for a in active if int(a.dep_min) >= arr]
            for other in active:
                if other.vehicle == row.vehicle:
                    continue
                overlap_start = arr
                overlap_end = min(dep, int(other.dep_min))
                shared = overlap_end - overlap_start + 1
                if shared < min_overlap_minutes:
                    continue
                first, second = sorted((other, row), key=lambda r: (str(r.vehicle), int(r.arr_min)))
                pair_rows.append(
                    {
                        "place_id": place_id,
                        "place_name": row.place_name,
                        "block_a": first.vehicle,
                        "block_b": second.vehicle,
                        "routes_a": first.route_names,
                        "routes_b": second.route_names,
                        "trips_a": first.trip_ids,
                        "trips_b": second.trip_ids,
                        "stops_a": first.stop_ids,
                        "stops_b": second.stop_ids,
                        "services_a": first.service_ids,
                        "services_b": second.service_ids,
                        "arrive_a": int(first.arr_min),
                        "depart_a": int(first.dep_min),
                        "arrive_b": int(second.arr_min),
                        "depart_b": int(second.dep_min),
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "overlap_minutes": shared,
                    }
                )
            active.append(row)
            present = len({a.vehicle for a in active})
            if present > peak_n:
                peak_n, peak_minute = present, arr
        stat_rows.append(
            {
                "place_id": place_id,
                "place_name": rows[0].place_name,
                "max_simultaneous": peak_n,
                "peak_minute": peak_minute,
            }
        )
    return (
        pd.DataFrame(pair_rows, columns=PAIR_COLUMNS),
        pd.DataFrame(stat_rows, columns=PLACE_STAT_COLUMNS),
    )


def find_block_self_overlaps(events: pd.DataFrame) -> pd.DataFrame:
    """Flag trips within one block whose scheduled spans overlap in time.

    A block's trips should run back to back; two trips of one block sharing
    :data:`SELF_OVERLAP_MIN_MINUTES` or more scheduled minutes mean the
    vehicle is booked in two places at once — a blocking error in the feed.
    A single shared hand-off minute (one trip ending as the next begins) is
    normal and not flagged. Blockless pseudo-vehicles have one trip each and
    are skipped.

    Args:
        events: Output of :func:`parse_stop_events`.

    Returns:
        One row per overlapping trip pair, with the columns listed in
        :data:`SELF_PAIR_COLUMNS`; times remain integer minutes.
    """
    if events.empty:
        return pd.DataFrame(columns=SELF_PAIR_COLUMNS)
    blocks = events.loc[events["is_block"]]
    if blocks.empty:
        return pd.DataFrame(columns=SELF_PAIR_COLUMNS)

    spans = (
        blocks.groupby(["vehicle", "trip_id", "route_name", "service_id"], sort=False)
        .agg(start_min=("arr_min", "min"), end_min=("dep_min", "max"))
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for vehicle, veh_df in spans.groupby("vehicle", sort=True):
        ordered = list(
            veh_df.sort_values(["start_min", "end_min"], kind="mergesort").itertuples(index=False)
        )
        active: list[Any] = []
        for row in ordered:
            start = int(row.start_min)
            end = int(row.end_min)
            active = [a for a in active if int(a.end_min) >= start]
            for other in active:
                shared = min(end, int(other.end_min)) - start + 1
                if shared < SELF_OVERLAP_MIN_MINUTES:
                    continue
                rows.append(
                    {
                        "block_id": vehicle,
                        "trip_a": other.trip_id,
                        "route_a": other.route_name,
                        "service_a": other.service_id,
                        "start_a": int(other.start_min),
                        "end_a": int(other.end_min),
                        "trip_b": row.trip_id,
                        "route_b": row.route_name,
                        "service_b": row.service_id,
                        "start_b": start,
                        "end_b": end,
                        "overlap_minutes": shared,
                    }
                )
            active.append(row)
    return pd.DataFrame(rows, columns=SELF_PAIR_COLUMNS)


def collapse_across_schedules(flags: pd.DataFrame) -> pd.DataFrame:
    """Merge identical flag rows discovered under more than one schedule.

    A conflict between two weekday blocks surfaces in every schedule whose
    service_id combination includes both — e.g. school and non-school
    weekdays. Rows identical in everything but their schedule columns are
    collapsed into one, joining the schedule labels and service groups with
    ``"; "`` and summing ``n_dates`` (total dates in the feed on which the
    conflict occurs).

    Args:
        flags: Concatenated per-schedule flag rows whose first three columns
            are ``schedule``, ``service_group``, and ``n_dates``.

    Returns:
        The collapsed rows, in the input column order.
    """
    if flags.empty:
        return flags

    identity = [c for c in flags.columns if c not in ("schedule", "service_group", "n_dates")]

    def _join_unique(values: pd.Series) -> str:
        return "; ".join(dict.fromkeys(str(v) for v in values))

    out = (
        flags.groupby(identity, dropna=False, sort=False)
        .agg(
            schedule=("schedule", _join_unique),
            service_group=("service_group", _join_unique),
            n_dates=("n_dates", lambda s: s.sum(min_count=1)),
        )
        .reset_index()
    )
    return out[list(flags.columns)]


def summarize_conflicts(detail: pd.DataFrame, place_stats: pd.DataFrame) -> pd.DataFrame:
    """Roll the flagged conflicts up to one row per place.

    Args:
        detail: Collapsed conflict pairs (times still integer minutes).
        place_stats: Concatenated per-schedule outputs of
            :func:`find_conflict_pairs`; the peak across schedules is kept.

    Returns:
        One row per flagged place with the columns listed in
        :data:`SUMMARY_COLUMNS`, sorted by descending conflict-pair count.
    """
    if detail.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    def _join_unique(values: pd.Series) -> str:
        return "; ".join(dict.fromkeys(str(v) for v in values))

    grouped = (
        detail.groupby(["place_id", "place_name"], dropna=False)
        .agg(
            schedules=("schedule", _join_unique),
            n_conflict_pairs=("place_id", "size"),
            max_overlap_minutes=("overlap_minutes", "max"),
        )
        .reset_index()
    )
    blocks = pd.concat(
        [
            detail[["place_id", "block_a"]].rename(columns={"block_a": "block"}),
            detail[["place_id", "block_b"]].rename(columns={"block_b": "block"}),
        ],
        ignore_index=True,
    )
    n_blocks = blocks.groupby("place_id")["block"].nunique().rename("n_blocks").reset_index()

    peaks = place_stats.sort_values(
        ["place_id", "max_simultaneous", "peak_minute"], ascending=[True, False, True]
    ).drop_duplicates("place_id")
    peaks = peaks.assign(peak_time=peaks["peak_minute"].map(minutes_to_hhmm))
    peaks = peaks.rename(columns={"max_simultaneous": "max_simultaneous_buses"})

    summary = grouped.merge(n_blocks, on="place_id", how="left").merge(
        peaks[["place_id", "max_simultaneous_buses", "peak_time"]], on="place_id", how="left"
    )
    return summary[SUMMARY_COLUMNS].sort_values(
        ["n_conflict_pairs", "place_id"], ascending=[False, True], ignore_index=True
    )


# =============================================================================
# OUTPUT & RUN LOG
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
    """Write the verbatim config block plus a review summary into *output_dir*.

    The sidecar is named after :data:`DETAIL_FILENAME` (same stem,
    ``_runlog.txt`` suffix) per CONTRIBUTING.md.

    Args:
        output_dir: Directory that received this run's output CSVs.
        summary_lines: Human-readable lines describing what the run did.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / f"{Path(DETAIL_FILENAME).stem}_runlog.txt"

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
        "BLOCK CONFLICT REVIEW RUN LOG",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "REVIEW SUMMARY",
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


def parse_time_to_minutes(time_value: Optional[str]) -> Optional[int]:
    """Convert an ``HH:MM[:SS]`` time string to integer minutes past midnight.

    GTFS times may exceed 24:00 (e.g. ``"25:30:00"`` for a 1:30 AM trip on
    the following calendar day); those values are preserved as integers
    greater than or equal to 1440. Seconds, when present, are rounded to the
    nearest minute.

    Args:
        time_value: Time string such as ``"7:05"``, ``"07:05:00"``, or
            ``"26:30:00"``. Leading/trailing whitespace is ignored.
            Non-string or malformed values yield ``None``.

    Returns:
        Minutes since midnight, or ``None`` if the value cannot be parsed.
    """
    if not isinstance(time_value, str):
        return None
    parts = time_value.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return hours * 60 + minutes + round(seconds / 60)


def minutes_to_hhmm(minutes: Optional[float], missing: str = "") -> str:
    """Convert minutes past midnight to a zero-padded ``HH:MM`` string.

    GTFS service days may exceed 24 hours, so values of 1440 minutes or more
    format with hours >= 24 (e.g. ``1590`` -> ``"26:30"``).

    Args:
        minutes: Minutes since midnight (may be fractional; rounded to the
            nearest minute). ``None`` and NaN yield ``missing``.
        missing: String returned for missing values, e.g. ``""`` or a
            sentinel such as ``"–"``.

    Returns:
        Zero-padded ``HH:MM`` string, or ``missing`` when *minutes* is
        ``None``/NaN.
    """
    if minutes is None or pd.isna(minutes):
        return missing
    hours, mins = divmod(int(round(minutes)), 60)
    return f"{hours:02d}:{mins:02d}"


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


def service_ids_active_on(
    active_dates: Mapping[str, set[dt.date]],
    target_date: dt.date,
) -> set[str]:
    """Return the service_ids operating on *target_date*.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        target_date: The calendar date to query.

    Returns:
        Set of service_id strings active on that date (possibly empty).
    """
    return {sid for sid, dates in active_dates.items() if target_date in dates}


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
# PIPELINE
# =============================================================================


def _load_optional(gtfs_path: str, file_name: str) -> Optional[pd.DataFrame]:
    """Load one optional GTFS file, returning ``None`` when absent or unreadable."""
    try:
        return load_gtfs_data(gtfs_path, files=(file_name,))[file_name.replace(".txt", "")]
    except (OSError, ValueError):
        return None


def _format_times(flags: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Format the integer-minute *columns* of *flags* as ``HH:MM`` strings."""
    out = flags.copy()
    for col in columns:
        out[col] = out[col].map(minutes_to_hhmm)
    if "n_dates" in out.columns:
        out["n_dates"] = out["n_dates"].astype("Int64")
    return out


def run(
    gtfs_path: str,
    output_dir: Path,
    min_overlap_minutes: int = MIN_OVERLAP_MINUTES,
    service_ids: Sequence[str] = (),
    routes_include: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
    combine_same_day_services: bool = COMBINE_SAME_DAY_SERVICES,
    group_by_parent_station: bool = GROUP_BY_PARENT_STATION,
    check_block_self_overlaps: bool = CHECK_BLOCK_SELF_OVERLAPS,
) -> pd.DataFrame:
    """Execute the block conflict review and write all artifacts.

    Args:
        gtfs_path: GTFS feed folder or ``.zip`` archive.
        output_dir: Directory receiving the CSVs and the run log.
        min_overlap_minutes: Flag block pairs sharing at least this many
            scheduled minutes at one stop (endpoints inclusive).
        service_ids: Analyze exactly these service_ids together as one
            schedule; empty groups schedules automatically.
        routes_include: Routes to keep (route_id or route_short_name).
        routes_exclude: Routes to drop (route_id or route_short_name).
        combine_same_day_services: Check service_ids that operate on the
            same calendar dates together.
        group_by_parent_station: Compare stops by parent_station when one
            is present.
        check_block_self_overlaps: Also report trips within one block that
            overlap each other in time.

    Returns:
        The flagged conflict pairs (also written to disk).

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
    calendar = _load_optional(gtfs_path, "calendar.txt")
    calendar_dates = _load_optional(gtfs_path, "calendar_dates.txt")
    if calendar is None and calendar_dates is None:
        logging.info("No usable calendar.txt or calendar_dates.txt in the feed.")

    trips = filter_trips(data["trips"], data["routes"], service_ids, routes_include, routes_exclude)

    if "block_id" in trips.columns:
        blockless = trips["block_id"].fillna("").astype(str).str.strip() == ""
        n_blockless = int(blockless.sum())
    else:
        n_blockless = len(trips)
    if n_blockless:
        logging.info(
            "%d trip(s) have no block_id — each is treated as its own single-trip vehicle.",
            n_blockless,
        )

    frequencies = _load_optional(gtfs_path, "frequencies.txt")
    if frequencies is not None and "trip_id" in frequencies.columns:
        n_freq = int(frequencies["trip_id"].isin(set(trips["trip_id"])).sum())
        if n_freq:
            logging.warning(
                "frequencies.txt defines %d in-scope template trip(s); their headway "
                "repetitions are NOT expanded, so conflicts among repetitions go unchecked.",
                n_freq,
            )

    groups = build_service_groups(
        trips, calendar, calendar_dates, service_ids, combine_same_day_services
    )
    logging.info("Checking %d schedule(s) for block conflicts.", len(groups))

    detail_parts: list[pd.DataFrame] = []
    self_parts: list[pd.DataFrame] = []
    stat_parts: list[pd.DataFrame] = []
    for group in groups:
        events = parse_stop_events(
            data["stop_times"],
            trips,
            data["routes"],
            data["stops"],
            group["service_ids"],
            group_by_parent_station,
        )
        intervals = merge_presence_intervals(events)
        pairs, stats = find_conflict_pairs(intervals, min_overlap_minutes)
        n_dates = float("nan") if group["n_dates"] is None else group["n_dates"]
        service_group = "+".join(sorted(group["service_ids"]))
        logging.info(
            "Schedule '%s' (%s): %d presence interval(s), %d conflict pair(s).",
            group["label"],
            service_group,
            len(intervals),
            len(pairs),
        )
        if not pairs.empty:
            detail_parts.append(
                pairs.assign(schedule=group["label"], service_group=service_group, n_dates=n_dates)
            )
        if not stats.empty:
            stat_parts.append(stats)
        if check_block_self_overlaps:
            self_pairs = find_block_self_overlaps(events)
            if not self_pairs.empty:
                self_parts.append(
                    self_pairs.assign(
                        schedule=group["label"], service_group=service_group, n_dates=n_dates
                    )
                )

    if detail_parts:
        detail = collapse_across_schedules(pd.concat(detail_parts, ignore_index=True))
        detail = detail.sort_values(
            ["place_id", "overlap_start", "block_a", "block_b"], ignore_index=True
        )
    else:
        detail = pd.DataFrame(columns=DETAIL_COLUMNS)
    place_stats = (
        pd.concat(stat_parts, ignore_index=True)
        if stat_parts
        else pd.DataFrame(columns=PLACE_STAT_COLUMNS)
    )
    summary = summarize_conflicts(detail, place_stats)
    if self_parts:
        self_overlaps = collapse_across_schedules(pd.concat(self_parts, ignore_index=True))
        self_overlaps = self_overlaps.sort_values(
            ["block_id", "start_a", "trip_a"], ignore_index=True
        )
    else:
        self_overlaps = pd.DataFrame(columns=SELF_OVERLAP_COLUMNS)

    detail = _format_times(
        detail[DETAIL_COLUMNS],
        ("arrive_a", "depart_a", "arrive_b", "depart_b", "overlap_start", "overlap_end"),
    )
    self_overlaps = _format_times(
        self_overlaps[SELF_OVERLAP_COLUMNS], ("start_a", "end_a", "start_b", "end_b")
    )

    ensure_dir(output_dir)
    detail_path = output_dir / DETAIL_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    self_path = output_dir / SELF_OVERLAP_FILENAME
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    self_overlaps.to_csv(self_path, index=False)
    logging.info("Wrote %d conflict pair(s) → %s", len(detail), detail_path)
    logging.info("Wrote %d summary row(s) → %s", len(summary), summary_path)
    logging.info("Wrote %d block self-overlap(s) → %s", len(self_overlaps), self_path)
    if detail.empty:
        logging.info(
            "No two blocks share %d+ scheduled minute(s) at any stop — nothing to review.",
            min_overlap_minutes,
        )

    summary_lines = [
        f"GTFS feed:            {gtfs_path}",
        f"Min shared minutes:   {min_overlap_minutes}",
        f"Schedules checked:    {len(groups)}",
        f"Trips in scope:       {trips['trip_id'].nunique()}",
        f"Conflict pairs:       {len(detail)}",
        f"Stops with conflicts: {len(summary)}",
        f"Block self-overlaps:  {len(self_overlaps)}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return detail


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Flag vehicle blocks scheduled at the same stop at overlapping times within one "
            "operating schedule of a GTFS feed, plus blocks scheduled in two places at once."
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
        help="Directory for the three output CSVs and the run log.",
    )
    parser.add_argument(
        "--min-overlap-minutes",
        type=int,
        default=MIN_OVERLAP_MINUTES,
        help="Flag block pairs sharing at least this many scheduled minutes at one stop.",
    )
    parser.add_argument(
        "--service-ids",
        nargs="*",
        default=FILTER_SERVICE_IDS,
        help="Analyze exactly these service_ids together as one schedule; omit for automatic.",
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
    parser.add_argument(
        "--combine-same-day-services",
        action=argparse.BooleanOptionalAction,
        default=COMBINE_SAME_DAY_SERVICES,
        help="Check service_ids that operate on the same calendar dates together.",
    )
    parser.add_argument(
        "--group-by-parent-station",
        action=argparse.BooleanOptionalAction,
        default=GROUP_BY_PARENT_STATION,
        help="Compare stops by parent_station (where present) instead of stop_id.",
    )
    parser.add_argument(
        "--check-block-self-overlaps",
        action=argparse.BooleanOptionalAction,
        default=CHECK_BLOCK_SELF_OVERLAPS,
        help="Also report trips within one block that overlap each other in time.",
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
            min_overlap_minutes=args.min_overlap_minutes,
            service_ids=args.service_ids,
            routes_include=args.routes_include,
            routes_exclude=args.routes_exclude,
            combine_same_day_services=args.combine_same_day_services,
            group_by_parent_station=args.group_by_parent_station,
            check_block_self_overlaps=args.check_block_self_overlaps,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
