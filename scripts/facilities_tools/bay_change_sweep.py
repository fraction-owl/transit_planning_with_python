"""Rank candidate bay reassignments and time shifts at a transit center (Step 3).

Reads Step 1 timelines (one scenario folder per occupancy standard, for
example a "direct conflict" and a "likely conflict" run), inventories what
could move, and scores candidate changes with the same conflict rule Step 2
uses: buses occupying the same bay in the same minute. Nothing moves unless
the CONFIGURATION lists it.

Discover pass (``DISCOVER_ONLY = True``)
----------------------------------------
Run this first. It writes ``<label>_discover.xlsx`` with every route-end found
at the cluster (``<route> arrive``, ``<route> depart``, ``<route> through
<direction>``) and the bay(s) it uses, every block's chain of trips with the
gap between them and where it is spent, and the interline hand-offs whose
gaps a time shift has to respect. It also logs a ``BAY_CANDIDATES`` /
``SHIFT_CANDIDATES`` stub to paste into this CONFIGURATION.

Sweep (``DISCOVER_ONLY = False``)
---------------------------------
Every listed single change is applied to the scheduled trips. Occupancy is
rebuilt for affected blocks, including loading, arrival buffers and layover
thresholds. A route-end shift moves the entire trips serving that end. Each
change is also checked against the block chains: a shift that
reduces any gap on any affected block below the minimums is rejected with the
reason. Counts are shown per standard and never summed across them; conflicts
are broken out by kind (boarding/boarding, boarding/waiting, waiting/waiting)
rather than weighted.

Inputs
------
- One Step 1 output folder per standard (``TIMELINES``): the combined CSV
  ``block_status_timeline_exporter.py`` writes, or its ``block_*.xlsx``
  workbooks. Sweeps require the revised exporter's ``schedule_snapshot.json``
  and ``timeline_manifest.json`` sidecars. Rerun Step 1 for every standard.
  Discover can inspect legacy timelines, but exact sweeps cannot use them.

Outputs
-------
- ``<SCENARIO_LABEL>_discover.xlsx`` (discover pass): "Route-ends", "Block
  chains" and "Interlines" sheets per standard, plus the "Config stub".
- ``<SCENARIO_LABEL>_sweep.xlsx`` (sweep): "Baseline" conflict minutes per
  bay and by kind; "Easy wins" (feasible single changes that do not raise the
  count on the standards named in ``REQUIRE_NO_WORSE_ON``, with before/after,
  the conflicts removed and created, the tightest gap after the change and
  an off-clockface note); "Pairs" (the best pairs, including pairs whose
  members are not wins on their own, such as a bay swap between two routes);
  "Packages" and "Package finalists" (a beam search over singles and the
  best pairs, up to ``MAX_CHANGES_PER_PACKAGE`` steps); "Rejected"
  (infeasible or non-improving changes and why); and "Config used".
- A ``_runlog.txt`` sidecar next to each workbook capturing the verbatim
  CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section, run once with
``DISCOVER_ONLY = True``, paste the candidates you are willing to consider
into ``BAY_CANDIDATES`` / ``SHIFT_CANDIDATES``, set ``DISCOVER_ONLY = False``
and run again, from a shell, ArcGIS Pro's Python window, or a Jupyter
notebook.

The three scripts remain standalone: the sweep contains an identical copy of
the exporter's occupancy renderer (canonical in utils/block_timeline_helpers.py).
Search is bounded by BEAM_WIDTH and the step limit; the returned packages are
candidates, not a guaranteed optimum.
Overlapping shift selectors apply the same shift once or reject contradictory
values. All standards must describe the same source trips and bay assignments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
# === BEGIN CONFIG ===

# --- Inputs ---------------------------------------------------------------------------------------
# One Step 1 scenario folder per standard. Every candidate is scored on all of
# them; the report shows each standard side by side and never sums across them.
# The comments give the Step 1 occupancy preset each folder was produced with.
TIMELINES: Dict[str, str] = {
    "Direct conflict": r"Path\To\block_status_output\direct_conflict",  # 1 / 0 / 0 / 10
    "Likely conflict": r"Path\To\block_status_output\likely_conflict",  # 2 / 5 / 2 / 10
}
COMBINED_TIMELINE_FILE = r"all_blocks_timeline.csv"  # read if present, else block_*.xlsx
OUTPUT_FOLDER = r"Path\To\bay_change_sweep_output"
SCENARIO_LABEL = "baseline"

# Same cluster as Step 2. Stops are GTFS stop_ids; the bay label is what the
# report shows. Bays not listed in CLUSTER_CAPACITY hold one bus.
CLUSTER_NAME = "Metro"
CLUSTER_STOPS: Dict[str, str] = {  # stop_id -> bay label
    "2373": "A",
    "2832": "B",
}
CLUSTER_CAPACITY: Dict[str, int] = {"B": 2}  # bay label -> capacity; default 1

# --- Discover pass --------------------------------------------------------------------------------
# Run with DISCOVER_ONLY = True first. The script logs every route-end it
# finds, its current bay(s), visits per day, the interline hand-offs on its
# blocks, and a stub of BAY_CANDIDATES / SHIFT_CANDIDATES to paste below.
# Nothing moves unless it is listed.
DISCOVER_ONLY = True

# --- Bay reassignment candidates -----------------------------------------------------------------
# A route-end is "<route> arrive" (trips ending here), "<route> depart"
# (trips starting here) or "<route> through <direction_id>" (mid-trip
# visits). A route-end moves as a whole, all day. Anything not listed is
# locked.
BAY_CANDIDATES: List[Dict[str, Any]] = [
    # {"route_end": "101 arrive", "bays": ["A", "B"]},
]

# Route-ends that must share a bay (moved together, or not at all).
SAME_BAY_GROUPS: List[List[str]] = [
    # ["101 arrive", "101 depart"],
]

# --- Time-shift candidates ------------------------------------------------------------------------
# Shifts in minutes, per route-end: only the trips that make up that
# route-end move ("101 arrive" shifts the trips that end at the cluster).
# Use the route name alone ("101") to shift every trip of the route. Route
# names must not contain spaces. Every gap on every affected block is checked
# against the minimums below.
SHIFT_CANDIDATES: Dict[str, List[int]] = {
    # "101 arrive": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    # "102": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
}

# Route-ends (or routes) that must shift together, e.g. an interline the
# discover pass reports as a hand-off of two minutes.
SHIFT_TOGETHER: List[List[str]] = [
    # ["101 arrive", "102 depart"],
]

# --- Feasibility (hard constraints, from the block chains) ---------------------------------------
MIN_INTERLINE_TURN_MINUTES = 2  # gap between two different routes on one block
MIN_LAYOVER_MINUTES = 3  # any other gap between trips on one block
FLAG_OFF_CLOCKFACE = True  # note, do not reject, when a shift takes departures off :00/:30

# --- Search ---------------------------------------------------------------------------------------
REQUIRE_NO_WORSE_ON = "all"  # "all": no standard may get worse; "any": at least one must not
MIN_IMPROVEMENT_MINUTES = 1
MAX_CHANGES_PER_PACKAGE = 4  # beam-search steps; a step adds a single change or a best pair
BEAM_WIDTH = 8  # partial packages kept alive at each step
ENUMERATE_PAIRS = True  # score every compatible pair of feasible singles
TOP_N = 30

# Every output must be traceable: a failed run-log write aborts the script.
# Set to False only when writing to a genuinely read-only location.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL = logging.INFO

# === END CONFIG ===

# Statuses (from Step 1)
BAY_STATUSES = {"ARRIVE", "DEPART", "ARRIVE/DEPART", "LOADING", "DWELL"}
BOARDING_STATUSES = {"ARRIVE", "DEPART", "ARRIVE/DEPART"}
WAITING_STATUSES = {"LOADING", "DWELL"}

BIG = 100000  # minute key spacing per bay (bay index * BIG + minute)

TRIP_COLUMNS = [
    "trip_id",
    "block",
    "route",
    "headsign",
    "direction",
    "start",
    "end",
    "first_stop",
    "first_bay",
    "last_stop",
    "last_bay",
    "through_bays",
    "departure",
    "arrival",
    "through_visits",
]
CHAIN_COLUMNS = [
    "block",
    "gap_min",
    "where",
    "from_route",
    "from_headsign",
    "from_trip",
    "arrives",
    "to_route",
    "to_headsign",
    "to_trip",
    "departs",
    "interline",
    "from_route_end",
    "to_route_end",
]

# ==================================================================================================
# LOADING
# ==================================================================================================


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
def timestamp_to_minutes(ts: object) -> Optional[int]:
    """Parse an HH:MM timeline timestamp, retaining hours beyond midnight."""
    if ts is None or pd.isna(ts):
        return None
    parts = str(ts).strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hours, minute = map(int, parts)
    except ValueError:
        return None
    return hours * 60 + minute if hours >= 0 and 0 <= minute < 60 else None


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


# Canonical versions live in utils/block_timeline_helpers.py -- keep these copies in sync.
def read_run_manifest(folder: str) -> Optional[dict[str, Any]]:
    """Read a completed run manifest; refuse failed or interrupted exporter runs."""
    path = Path(folder) / "timeline_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"Exporter run in {folder} is not complete. Rerun Step 1 successfully.")
    if manifest.get("interval_minutes") != 1:
        raise ValueError("Timeline input must use one-minute intervals.")
    return manifest


def verified_run_file(folder: str, name: str, manifest: dict[str, Any]) -> Path:
    """Resolve one manifest-listed file and verify it belongs to that completed run."""
    if not name or Path(name).name != name or "\\" in name:
        raise ValueError(f"Expected a filename inside the run folder, got {name!r}.")
    expected = manifest.get("files", {}).get(name)
    path = Path(folder) / name
    if not expected or not path.is_file():
        raise ValueError(f"Missing current-run file {name!r}; rerun Step 1.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{name} has changed since Step 1 completed; rerun Step 1.")
    return path


def read_timeline_files(folder: str, combined_name: str) -> DataFrame:
    """Read only current-run outputs, with an explicit legacy-input fallback."""
    manifest = read_run_manifest(folder)
    if manifest is not None:
        combined = manifest.get("combined_timeline", "")
        if combined_name and combined:
            if combined_name != combined:
                raise ValueError("COMBINED_TIMELINE_FILE does not match the exporter manifest.")
            path = verified_run_file(folder, combined, manifest)
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            names = manifest.get("block_workbooks", [])
            if not names:
                raise ValueError("This run has no block workbooks; enable the combined CSV input.")
            frames = []
            for name in names:
                part = pd.read_excel(
                    verified_run_file(folder, name, manifest), dtype=str, keep_default_na=False
                )
                frames.append(part.assign(FileName=name))
            frame = pd.concat(frames, ignore_index=True)
    else:
        logging.warning(
            "Legacy timeline without a run manifest: current-run identity cannot be checked."
        )
        combined_path = Path(folder) / combined_name
        if combined_name and combined_path.is_file():
            frame = pd.read_csv(combined_path, dtype=str, keep_default_na=False)
        else:
            paths = sorted(Path(folder).glob("block_*.xlsx"))
            if not paths:
                raise FileNotFoundError(f"No timeline CSV or block workbooks in {folder}.")
            frame = pd.concat(
                [
                    pd.read_excel(path, dtype=str, keep_default_na=False).assign(FileName=path.name)
                    for path in paths
                ],
                ignore_index=True,
            )
    required = {"Timestamp", "Block", "Trip ID", "Stop ID", "Status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Timeline is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Timeline contains no rows.")
    if frame["Block"].eq("").any():
        raise ValueError("Timeline contains blank block IDs.")
    if frame.duplicated(["Block", "Timestamp"]).any():
        raise ValueError("Duplicate block/timestamp rows: check for mixed or stale input files.")
    minute = frame["Timestamp"].map(timestamp_to_minutes)
    if minute.isna().any():
        raise ValueError("Timeline contains invalid timestamps.")
    if frame.assign(_minute=minute).duplicated(["Block", "_minute"]).any():
        raise ValueError("Timeline contains duplicate block/minute rows.")
    for _, group in frame.assign(_minute=minute).groupby("Block", sort=False):
        differences = group["_minute"].sort_values().diff().dropna()
        if not differences.empty and not differences.eq(1).all():
            raise ValueError("Timeline must contain consecutive one-minute rows for each block.")
    return frame


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
def find_cluster(stop_id: str, clusters: list[dict[str, Any]]) -> Optional[str]:
    """Return cluster name containing the given stop ID, if any."""
    for cluster_item in clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


# Canonical versions live in utils/block_timeline_helpers.py -- keep these copies in sync.
def status_for_same_trip(
    minute: int, stop_info: tuple, settings: dict[str, int]
) -> Optional[tuple]:
    """Return a visit's occupancy, including both boundaries of a scheduled hold."""
    arr, dep, sid, name, tid, first, last, seq, timepoint = stop_info
    detail = (sid, name, minutes_to_hhmm(arr), minutes_to_hhmm(dep), tid, seq, timepoint)
    if first and minute == dep:
        return ("DEPART",) + detail
    if last and minute == arr:
        return ("ARRIVE",) + detail
    if arr <= minute <= dep:
        if not first and not last and arr == dep:
            return ("ARRIVE/DEPART",) + detail
        if first and minute >= dep - settings["PRE_DEPARTURE_MINUTES"]:
            return ("LOADING",) + detail
        return ("DWELL",) + detail
    if not first and not last and arr == dep:
        if arr <= minute < arr + settings["THROUGH_DWELL_MINUTES"]:
            return ("ARRIVE/DEPART",) + detail
    return None


def gap_status(gap: int, same_place: bool, settings: dict[str, int]) -> tuple[str, str]:
    """Classify a between-trip gap using this scenario's occupancy assumptions."""
    if not same_place:
        return "DEADHEAD", ""
    if gap <= settings["IN_BAY_LAYOVER_MAX_MINUTES"]:
        return "DWELL", "in bay"
    if gap <= settings["LAYOVER_THRESHOLD"]:
        return "LAYOVER", "overflow"
    return "LONG BREAK", "overflow"


def make_timeline_row(
    minute: int,
    block_id: str,
    status: str,
    trip: Optional[dict[str, Any]] = None,
    stop_id: str = "",
    stop_name: str = "",
    stop_seq: Any = "",
    arr_str: str = "",
    dep_str: str = "",
    trip_id: str = "",
    timepoint: int = 0,
    layover_location: str = "",
    prev_trip_id: str = "",
    next_trip_id: str = "",
    stop_role: str = "",
) -> dict[str, Any]:
    """Build one timeline row with explicit visit role and scheduled trip bounds."""
    return {
        "Timestamp": minutes_to_hhmm(minute),
        "Block": block_id,
        "Route": trip["route_id"] if trip else "",
        "Route Short Name": trip["route_short_name"] if trip else "",
        "Direction": trip["direction_id"] if trip else "",
        "Trip Headsign": trip["trip_headsign"] if trip else "",
        "Trip ID": trip_id,
        "Stop ID": stop_id or "",
        "Stop Name": stop_name or "",
        "Stop Sequence": stop_seq if stop_seq is not None else "",
        "Arrival Time": arr_str or "",
        "Departure Time": dep_str or "",
        "Status": status,
        "Layover Location": layover_location,
        "Prev Trip ID": prev_trip_id,
        "Next Trip ID": next_trip_id,
        "Timepoint": timepoint,
        "Stop Role": stop_role,
        "Trip Start Minute": trip["start"] if trip else "",
        "Trip End Minute": trip["end"] if trip else "",
    }


def row_for_inactive(
    minute: int,
    block_id: str,
    all_trips: list[dict[str, Any]],
    bus_stop_clusters: list[dict[str, Any]],
    settings: dict[str, int],
) -> dict[str, Any]:
    """Recalculate loading, post-arrival occupancy and layovers between trips."""
    previous = [trip for trip in all_trips if trip["end"] < minute]
    upcoming = [trip for trip in all_trips if trip["start"] > minute]
    prev = max(previous, key=lambda trip: trip["end"]) if previous else None
    nxt = min(upcoming, key=lambda trip: trip["start"]) if upcoming else None
    prev_id = prev["trip_id"] if prev else ""
    next_id = nxt["trip_id"] if nxt else ""
    arr = minutes_to_hhmm(prev["end"]) if prev else ""
    dep = minutes_to_hhmm(nxt["start"]) if nxt else ""
    if nxt and minute >= nxt["start"] - settings["PRE_DEPARTURE_MINUTES"]:
        return make_timeline_row(
            minute,
            block_id,
            "LOADING",
            nxt,
            nxt["first_stop_id"],
            nxt["first_stop_name"],
            nxt["first_stop_seq"],
            arr,
            dep,
            next_id,
            layover_location="in bay",
            prev_trip_id=prev_id,
            next_trip_id=next_id,
            stop_role="depart",
        )
    if prev and minute <= prev["end"] + settings["POST_ARRIVAL_MINUTES"]:
        return make_timeline_row(
            minute,
            block_id,
            "ARRIVE",
            prev,
            prev["last_stop_id"],
            prev["last_stop_name"],
            prev["last_stop_seq"],
            arr,
            dep,
            prev_id,
            layover_location="in bay",
            prev_trip_id=prev_id,
            next_trip_id=next_id,
            stop_role="arrive",
        )
    if prev and nxt:
        a = find_cluster(prev["last_stop_id"], bus_stop_clusters)
        b = find_cluster(nxt["first_stop_id"], bus_stop_clusters)
        same_place = prev["last_stop_id"] == nxt["first_stop_id"] or (a is not None and a == b)
        status, location = gap_status(nxt["start"] - prev["end"], same_place, settings)
        if status != "DEADHEAD":
            return make_timeline_row(
                minute,
                block_id,
                status,
                prev,
                prev["last_stop_id"],
                prev["last_stop_name"],
                prev["last_stop_seq"],
                arr,
                dep,
                prev_id,
                layover_location=location,
                prev_trip_id=prev_id,
                next_trip_id=next_id,
                stop_role="arrive",
            )
        return make_timeline_row(
            minute, block_id, status, prev_trip_id=prev_id, next_trip_id=next_id
        )
    return make_timeline_row(
        minute, block_id, "INACTIVE", prev_trip_id=prev_id, next_trip_id=next_id
    )


def build_schedule_rows(
    trips_summary: list[dict[str, Any]],
    timeline: range,
    block_id: str,
    bus_stop_clusters: list[dict[str, Any]],
    settings: dict[str, int],
    occupancy_only: bool = False,
) -> list[dict[str, Any]]:
    """Render a block from scheduled visits, with one physical vehicle per minute.

    Args:
        trips_summary: Complete scheduled trips on this block.
        timeline: Minutes to render; the pipeline requires a one-minute step.
        block_id: Physical vehicle block identifier.
        bus_stop_clusters: Stops treated as one facility for layover decisions.
        settings: Explicit occupancy settings for this scenario.
        occupancy_only: Omit traveling, inactive and deadhead rows for rescoring.

    Returns:
        Timeline row dictionaries in chronological order.
    """
    if timeline.step != 1:
        raise ValueError("The bay-analysis pipeline requires TIME_INTERVAL_MINUTES = 1.")
    active: dict[int, list[dict[str, Any]]] = {}
    visits: dict[tuple[str, int], tuple] = {}
    for trip in trips_summary:
        for minute in range(
            max(timeline.start, trip["start"]), min(timeline.stop, trip["end"] + 1)
        ):
            active.setdefault(minute, []).append(trip)
        sequence = trip["stop_times_sequence"]
        for i, stop in enumerate(sequence):
            arr, dep = stop[:2]
            finish = dep
            if not stop[5] and not stop[6] and arr == dep:
                finish = arr + settings["THROUGH_DWELL_MINUTES"] - 1
            if i + 1 < len(sequence):
                # Once the next visit begins, an earlier through dwell cannot reappear.
                finish = min(finish, sequence[i + 1][0] - 1)
            finish = min(finish, trip["end"], timeline.stop - 1)
            for minute in range(max(arr, timeline.start), finish + 1):
                status = status_for_same_trip(minute, stop, settings)
                if status is not None:
                    role = "depart" if stop[5] else "arrive" if stop[6] else "through"
                    visits[trip["trip_id"], minute] = (*status, role)
    rows: list[dict[str, Any]] = []
    for minute in timeline:
        candidates = [
            (trip, visits[trip["trip_id"], minute])
            for trip in active.get(minute, [])
            if (trip["trip_id"], minute) in visits
        ]
        if candidates:
            trip, status = min(
                candidates,
                key=lambda item: (
                    item[1][0] != "DEPART",
                    item[1][7] == 0,
                    item[1][6],
                    item[0]["trip_id"],
                ),
            )
            state, sid, name, arr, dep, tid, seq, point, role = status
            rows.append(
                make_timeline_row(
                    minute,
                    block_id,
                    state,
                    trip,
                    sid,
                    name,
                    seq,
                    arr,
                    dep,
                    tid,
                    point,
                    layover_location="in bay" if state in {"DWELL", "LOADING"} else "",
                    stop_role=role,
                )
            )
        elif minute in active:
            if not occupancy_only:
                trip = min(active[minute], key=lambda value: (value["start"], value["trip_id"]))
                rows.append(
                    make_timeline_row(
                        minute, block_id, "TRAVELING BETWEEN STOPS", trip, trip_id=trip["trip_id"]
                    )
                )
        else:
            row = row_for_inactive(minute, block_id, trips_summary, bus_stop_clusters, settings)
            if not occupancy_only or row["Status"] not in {"INACTIVE", "DEADHEAD"}:
                rows.append(row)
    return rows


def load_timeline(folder: str) -> DataFrame:
    """Load and validate one current exporter run, retaining its canonical schedule."""
    df = read_timeline_files(folder, COMBINED_TIMELINE_FILE)
    for column in (
        "Route Short Name",
        "Trip Headsign",
        "Direction",
        "Prev Trip ID",
        "Next Trip ID",
        "Layover Location",
    ):
        if column not in df:
            df[column] = ""
    df["Minute"] = df["Timestamp"].map(timestamp_to_minutes).astype(int)
    df["Bay"] = df["Stop ID"].map(CLUSTER_STOPS).fillna("")
    df.attrs["schedule_snapshot"] = load_schedule_snapshot(folder)
    logging.info("Loaded %d rows and %d blocks from %s.", len(df), df["Block"].nunique(), folder)
    return df


# ==================================================================================================
# TRIPS, ROUTE-ENDS, BLOCK CHAINS
# ==================================================================================================


def build_trips(df: DataFrame) -> DataFrame:
    """One row per trip: route, headsign, direction, block, scheduled start/end and end stops.

    Start is the first stop's scheduled departure (the DEPART row); end is the
    last stop's scheduled arrival (the largest Arrival Time on the trip).
    """
    if df.attrs.get("schedule_snapshot") is not None:
        return trips_from_snapshot(df.attrs["schedule_snapshot"])
    logging.warning("Discover is inferring legacy trip bounds. Rerun Step 1 before the sweep.")
    rows = []
    on_trip = df[
        (df["Trip ID"] != "") & df["Status"].isin(BAY_STATUSES | {"TRAVELING BETWEEN STOPS"})
    ]
    for tid, g in on_trip.groupby("Trip ID", sort=False):
        dep = g[g["Status"] == "DEPART"]
        arr_minutes = g["Arrival Time"].map(timestamp_to_minutes)
        arr_times = arr_minutes.dropna().tolist()
        if not arr_times or dep.empty:
            raise ValueError(f"Cannot recover scheduled endpoints for trip {tid}. Rerun Step 1.")
        start = int(dep["Arrival Time"].map(timestamp_to_minutes).dropna().iloc[0])
        arrivals = g[g["Status"].eq("ARRIVE")]
        if arrivals.empty:
            raise ValueError(f"Cannot recover the scheduled arrival for trip {tid}. Rerun Step 1.")
        end = int(arrivals["Arrival Time"].map(timestamp_to_minutes).dropna().max())
        first = dep.iloc[0] if not dep.empty else g.sort_values("Minute").iloc[0]
        last_rows = g[arr_minutes == end]
        if last_rows.empty:
            last_rows = g
        last = last_rows.sort_values("Minute").iloc[-1]
        through = g[(g["Status"] == "ARRIVE/DEPART") & (g["Bay"] != "")]["Bay"]
        rows.append(
            {
                "trip_id": tid,
                "block": first["Block"],
                "route": first["Route Short Name"],
                "headsign": first["Trip Headsign"],
                "direction": first["Direction"],
                "start": start,
                "end": end,
                "first_stop": first["Stop ID"],
                "first_bay": CLUSTER_STOPS.get(first["Stop ID"], ""),
                "last_stop": last["Stop ID"],
                "last_bay": CLUSTER_STOPS.get(last["Stop ID"], ""),
                "through_bays": ",".join(sorted(set(through))),
                "departure": int(dep["Minute"].iloc[0]),
                "arrival": int(end),
                "through_visits": [
                    (r["Bay"], int(r["Minute"]))
                    for _, r in g[g["Status"].eq("ARRIVE/DEPART") & g["Bay"].ne("")]
                    .drop_duplicates(["Stop Sequence", "Arrival Time"])
                    .iterrows()
                ],
            }
        )
    trips = DataFrame(rows, columns=TRIP_COLUMNS)
    return trips.sort_values(["block", "start"]).reset_index(drop=True)


def route_end_of(trip: pd.Series) -> List[Tuple[str, str]]:
    """Which route-ends this trip contributes at the cluster: ``[(route_end, bay), ...]``."""
    out = []
    if trip["first_bay"]:
        out.append((f"{trip['route']} depart", trip["first_bay"]))
    if trip["last_bay"]:
        out.append((f"{trip['route']} arrive", trip["last_bay"]))
    for bay in [b for b in str(trip["through_bays"]).split(",") if b]:
        out.append((f"{trip['route']} through {trip['direction']}", bay))
    return out


def build_block_chains(trips: DataFrame) -> DataFrame:
    """Consecutive trip pairs on each block with the gap and where it is spent."""
    rows = []
    for block, g in trips.groupby("block", sort=False):
        g = g.sort_values("start")
        prev = None
        for _, t in g.iterrows():
            if prev is not None:
                gap = int(t["start"] - prev["end"])
                if prev["last_stop"] == t["first_stop"]:
                    bay = f" (Bay {prev['last_bay']})" if prev["last_bay"] else ""
                    where = f"same stop{bay}"
                elif prev["last_bay"] and t["first_bay"]:
                    where = f"cluster (Bay {prev['last_bay']} to Bay {t['first_bay']})"
                else:
                    where = "elsewhere"
                rows.append(
                    {
                        "block": block,
                        "gap_min": gap,
                        "where": where,
                        "from_route": prev["route"],
                        "from_headsign": prev["headsign"],
                        "from_trip": prev["trip_id"],
                        "arrives": minutes_to_hhmm(int(prev["end"])),
                        "to_route": t["route"],
                        "to_headsign": t["headsign"],
                        "to_trip": t["trip_id"],
                        "departs": minutes_to_hhmm(int(t["start"])),
                        "interline": prev["route"] != t["route"],
                        "from_route_end": f"{prev['route']} arrive" if prev["last_bay"] else "",
                        "to_route_end": f"{t['route']} depart" if t["first_bay"] else "",
                    }
                )
            prev = t
    return DataFrame(rows, columns=CHAIN_COLUMNS)


def build_route_ends(trips: DataFrame, chains: DataFrame) -> DataFrame:
    """Inventory of route-ends at the cluster with bays, visits, times and tightest gaps."""
    visits: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)  # -> [(bay, minute, trip)]
    for _, t in trips.iterrows():
        for route_end, bay in route_end_of(t):
            if " through " in route_end:
                for visit_bay, minute in t["through_visits"]:
                    if visit_bay == bay:
                        visits[route_end].append((bay, int(minute), t["trip_id"]))
            else:
                minute = t["departure"] if route_end.endswith("depart") else t["arrival"]
                visits[route_end].append((bay, int(minute), t["trip_id"]))
    rows = []
    for route_end, lst in sorted(visits.items()):
        bays = sorted({b for b, _, _ in lst})
        mins = [m for _, m, _ in lst]
        tids = {tid for _, _, tid in lst}
        before = chains[chains["to_trip"].isin(tids)]
        after = chains[chains["from_trip"].isin(tids)]
        partners = set(before.loc[before["interline"], "from_route"]) | set(
            after.loc[after["interline"], "to_route"]
        )
        rows.append(
            {
                "route_end": route_end,
                "bays": ",".join(bays),
                "visits_per_day": len(lst),
                "first": minutes_to_hhmm(min(mins)),
                "last": minutes_to_hhmm(max(mins)),
                "min_gap_before": int(before["gap_min"].min()) if not before.empty else "",
                "min_gap_after": int(after["gap_min"].min()) if not after.empty else "",
                "interlines_with": ",".join(sorted(partners)),
            }
        )
    cols = [
        "route_end",
        "bays",
        "visits_per_day",
        "first",
        "last",
        "min_gap_before",
        "min_gap_after",
        "interlines_with",
    ]
    return DataFrame(rows, columns=cols)


def interline_summary(chains: DataFrame) -> DataFrame:
    """Minimum hand-off gap per ordered route pair, with the count and the tightest example."""
    cols = [
        "from_route",
        "to_route",
        "hand_offs_per_day",
        "min_gap_min",
        "max_gap_min",
        "tightest_example",
        "shift_together_needed",
    ]
    if chains.empty:
        return DataFrame(columns=cols)
    inter = chains[chains["interline"]]
    rows = []
    for (a, b), g in inter.groupby(["from_route", "to_route"]):
        tight = g.sort_values("gap_min").iloc[0]
        rows.append(
            {
                "from_route": a,
                "to_route": b,
                "hand_offs_per_day": len(g),
                "min_gap_min": int(g["gap_min"].min()),
                "max_gap_min": int(g["gap_min"].max()),
                "tightest_example": (
                    f"{tight['arrives']} {tight['from_headsign']} -> "
                    f"{tight['departs']} {tight['to_headsign']} ({tight['where']})"
                ),
                "shift_together_needed": bool(g["gap_min"].min() <= MIN_INTERLINE_TURN_MINUTES + 3),
            }
        )
    return DataFrame(rows, columns=cols).sort_values("min_gap_min")


# ==================================================================================================
# DISCOVER
# ==================================================================================================


def config_stub(route_ends: DataFrame, inter: DataFrame) -> str:
    """Text the user can paste into CONFIGURATION."""
    lines = ["BAY_CANDIDATES: List[Dict[str, Any]] = ["]
    for _, r in route_ends.iterrows():
        first_bay = r["bays"].split(",")[0]
        lines.append(
            f'    # {{"route_end": "{r["route_end"]}", "bays": ["{first_bay}"]}},'
            f"  # now {r['bays']}, {r['visits_per_day']} visits/day"
        )
    lines.append("]")
    lines.append("SHIFT_CANDIDATES: Dict[str, List[int]] = {")
    for _, r in route_ends.iterrows():
        lines.append(f'    # "{r["route_end"]}": [-3, -2, -1, 1, 2, 3],')
    for route in sorted({str(re_).split(" ")[0] for re_ in route_ends["route_end"]}):
        lines.append(f'    # "{route}": [-3, -2, -1, 1, 2, 3],  # whole route')
    lines.append("}")
    lines.append("SHIFT_TOGETHER: List[List[str]] = [")
    if not inter.empty:
        for _, r in inter[inter["shift_together_needed"]].iterrows():
            lines.append(
                f'    # ["{r["from_route"]} arrive", "{r["to_route"]} depart"],'
                f"  # hand-off as tight as {r['min_gap_min']} min"
            )
    lines.append("]")
    return "\n".join(lines)


def run_discover() -> str:
    """Write the discover workbook and log the CONFIGURATION stub.

    Returns:
        Path of the workbook written.
    """
    validate_sweep_configuration()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, f"{SCENARIO_LABEL}_discover.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        stub = ""
        for tier, folder in TIMELINES.items():
            df = load_timeline(folder)
            trips = build_trips(df)
            chains = build_block_chains(trips)
            route_ends = build_route_ends(trips, chains)
            inter = interline_summary(chains)
            tag = "".join(ch for ch in tier if ch.isalnum())[:12]
            route_ends.to_excel(writer, sheet_name=f"Route-ends {tag}", index=False)
            chains.to_excel(writer, sheet_name=f"Block chains {tag}", index=False)
            if not inter.empty:
                inter.to_excel(writer, sheet_name=f"Interlines {tag}", index=False)
            logging.info(
                "%s: %d trips, %d route-ends, %d block hand-offs (%d interline)",
                tier,
                len(trips),
                len(route_ends),
                len(chains),
                int(chains["interline"].sum()) if not chains.empty else 0,
            )
            if not stub:
                stub = config_stub(route_ends, inter)
        DataFrame({"config_stub": stub.split("\n")}).to_excel(
            writer, sheet_name="Config stub", index=False
        )
    logging.info(
        "Paste into CONFIGURATION (uncomment what may move) -- also on the "
        "'Config stub' sheet:\n%s",
        stub,
    )
    logging.info("Discover workbook written to %s", out_path)
    require_run_log(write_run_log(Path(out_path)))
    return out_path


# ==================================================================================================
# SWEEP
# ==================================================================================================


class Change:
    """Absolute bay assignments and shifts; overlapping selectors never add shifts twice."""

    def __init__(
        self, moves: Optional[Dict[str, str]] = None, shifts: Optional[Dict[str, int]] = None
    ) -> None:
        """Copy candidate settings so search combinations do not mutate their inputs."""
        self.moves = dict(moves or {})
        self.shifts = {key: value for key, value in (shifts or {}).items() if value}

    def key(self) -> tuple:
        """Return an order-independent identity for deduplication and caching."""
        return tuple(sorted(self.moves.items())), tuple(sorted(self.shifts.items()))

    def label(self) -> str:
        """Describe a candidate consistently regardless of the order in which it was assembled."""
        return "; ".join(
            [f"{key} to Bay {bay}" for key, bay in sorted(self.moves.items())]
            + [f"{key} {value:+d} min" for key, value in sorted(self.shifts.items())]
        )

    def combine(self, other: Change) -> Change:
        """Combine consistent absolute changes, rejecting conflicting assignments."""
        for left, right in ((self.moves, other.moves), (self.shifts, other.shifts)):
            if any(left[key] != right[key] for key in set(left) & set(right)):
                raise ValueError("Package contains conflicting bay assignments or time shifts.")
        return Change({**self.moves, **other.moves}, {**self.shifts, **other.shifts})

    def compatible(self, other: Change) -> bool:
        """Return whether another change contributes a consistent new assignment."""
        try:
            return self.combine(other).key() != self.key()
        except ValueError:
            return False


def expand_groups(change: Change) -> Change:
    """Expand overlapping groups to closure and reject contradictory member assignments."""

    def expand(values: dict, groups: list) -> dict:
        result = dict(values)
        while True:
            before = dict(result)
            for group in groups:
                assigned = {result[key] for key in group if key in result}
                if len(assigned) > 1:
                    raise ValueError(f"Conflicting assignments within group {group}.")
                if assigned:
                    value = next(iter(assigned))
                    result.update({key: value for key in group})
            if before == result:
                return result

    return Change(expand(change.moves, SAME_BAY_GROUPS), expand(change.shifts, SHIFT_TOGETHER))


def trips_from_snapshot(snapshot: dict[str, Any]) -> DataFrame:
    """Build complete trip metadata directly from scheduled visits, including zero-gap handoffs."""
    rows = []
    for trip in snapshot["trips"]:
        first, last = trip["stop_times_sequence"][0], trip["stop_times_sequence"][-1]
        through = [
            (CLUSTER_STOPS[stop[2]], int(stop[0]))
            for stop in trip["stop_times_sequence"][1:-1]
            if stop[2] in CLUSTER_STOPS
        ]
        rows.append(
            {
                "trip_id": trip["trip_id"],
                "block": trip["block"],
                "route": trip["route_short_name"],
                "headsign": trip["trip_headsign"],
                "direction": trip["direction_id"],
                "start": int(trip["start"]),
                "end": int(trip["end"]),
                "departure": int(first[1]),
                "arrival": int(last[0]),
                "first_stop": first[2],
                "first_bay": CLUSTER_STOPS.get(first[2], ""),
                "last_stop": last[2],
                "last_bay": CLUSTER_STOPS.get(last[2], ""),
                "through_bays": ",".join(sorted({bay for bay, _ in through})),
                "through_visits": through,
            }
        )
    return DataFrame(rows).sort_values(["block", "start"]).reset_index(drop=True)


def load_schedule_snapshot(folder: str) -> Optional[dict[str, Any]]:
    """Load the canonical schedule belonging to a completed exporter run."""
    manifest = read_run_manifest(folder)
    if manifest is None:
        return None
    path = verified_run_file(folder, manifest.get("schedule_snapshot", ""), manifest)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or snapshot.get("interval_minutes") != 1:
        raise ValueError("Unsupported schedule snapshot. Rerun the revised exporter.")
    if not snapshot.get("trips"):
        raise ValueError("Schedule snapshot has no trips.")
    return snapshot


class Standard:
    """One scenario with exact occupancy regeneration for changed vehicle blocks."""

    def __init__(self, name: str, df: DataFrame, trips: DataFrame) -> None:
        """Retain the schedule snapshot, baseline occupancy and scenario assumptions."""
        self.name = name
        snapshot = df.attrs.get("schedule_snapshot")
        if snapshot is None:
            raise ValueError(
                "The sweep requires schedule_snapshot.json from the revised exporter. "
                "Rerun Step 1 for every TIMELINES scenario first."
            )
        self.snapshot = snapshot
        self.trips = trips
        if trips["route"].astype(str).str.contains(r"\s", regex=True).any():
            raise ValueError("Route short names cannot contain whitespace in sweep selectors.")
        self.bay_names = sorted(set(CLUSTER_STOPS.values()))
        self.bay_index = {name: i for i, name in enumerate(self.bay_names)}
        self.cap = np.array([CLUSTER_CAPACITY.get(name, 1) for name in self.bay_names], dtype=int)
        self.trip_index = {trip["trip_id"]: trip for trip in snapshot["trips"]}
        self.block_trips: Dict[str, List[dict[str, Any]]] = defaultdict(list)
        for trip in snapshot["trips"]:
            self.block_trips[trip["block"]].append(trip)
        self.stop_names = {stop["stop_id"]: stop["stop_name"] for stop in snapshot.get("stops", [])}
        for trip in snapshot["trips"]:
            self.stop_names.update({stop[2]: stop[3] for stop in trip["stop_times_sequence"]})
        self.target_stops = {
            bay: next(stop for stop, label in CLUSTER_STOPS.items() if label == bay)
            for bay in self.bay_names
        }
        for stop in CLUSTER_STOPS:
            if stop not in self.stop_names:
                raise ValueError(f"Cluster stop {stop!r} is absent from the source GTFS snapshot.")
        facilities = {find_cluster(stop, snapshot["clusters"]) for stop in CLUSTER_STOPS}
        if None in facilities or len(facilities) != 1:
            raise ValueError("CLUSTER_STOPS must belong to one cluster in the exporter settings.")
        self.shift_model = Feasibility(trips, DataFrame(columns=CHAIN_COLUMNS))
        self.occ = self._occupancy(df)
        self.mask_route_end = {
            key: self.occ["route_end"].eq(key).to_numpy()
            for key in self.shift_model.masks
            if " " in key
        }
        self.cache: OrderedDict[tuple, DataFrame] = OrderedDict()

    def _occupancy(self, frame: DataFrame) -> DataFrame:
        """Select bay occupancy and identify visits by their explicit role."""
        if frame.empty:
            return DataFrame(columns=["Block", "Minute", "Bay", "Status", "route_end"])
        frame = frame.copy()
        # Keep the large canonical snapshot on Standard, not on every cached DataFrame.
        frame.attrs = {}
        frame["Bay"] = frame["Stop ID"].map(CLUSTER_STOPS).fillna("")
        frame = frame[frame["Status"].isin(BAY_STATUSES) & frame["Bay"].ne("")].copy()
        frame["Minute"] = frame["Timestamp"].map(timestamp_to_minutes).astype(int)
        roles = frame["Stop Role"].fillna("")
        if not roles.isin({"arrive", "depart", "through"}).all():
            raise ValueError("Bay occupancy is missing a visit role; rerun the revised exporter.")
        frame["route_end"] = frame["Route Short Name"] + " " + roles
        through = roles.eq("through")
        frame.loc[through, "route_end"] += " " + frame.loc[through, "Direction"].fillna("")
        return frame.reset_index(drop=True)

    def apply(self, change: Change) -> DataFrame:
        """Move scheduled visits/trips, then rebuild every affected block's complete occupancy."""
        if not change.moves and not change.shifts:
            return self.occ
        key = change.key()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        shifts = self.shift_model.trip_shift(change.shifts)
        by_trip = dict(zip(self.trips["trip_id"], shifts))
        moved_routes = {route_end.split(" ", 1)[0] for route_end in change.moves}
        changed: Dict[str, List[dict[str, Any]]] = {}
        for block, originals in self.block_trips.items():
            revised = []
            affected = False
            for original in originals:
                shift = int(by_trip[original["trip_id"]])
                if not shift and original["route_short_name"] not in moved_routes:
                    revised.append(original)
                    continue
                trip = copy.deepcopy(original)
                affected |= bool(shift)
                trip["start"] += shift
                trip["end"] += shift
                stops = []
                for original_stop in trip["stop_times_sequence"]:
                    stop = list(original_stop)
                    role = (
                        "depart"
                        if stop[5]
                        else "arrive"
                        if stop[6]
                        else f"through {trip['direction_id']}"
                    )
                    route_end = f"{trip['route_short_name']} {role}"
                    if stop[2] in CLUSTER_STOPS and route_end in change.moves:
                        target = self.target_stops[change.moves[route_end]]
                        affected |= target != stop[2]
                        stop[2], stop[3] = target, self.stop_names[target]
                    stop[0] += shift
                    stop[1] += shift
                    stops.append(stop)
                trip["stop_times_sequence"] = stops
                for prefix, stop in (("first", stops[0]), ("last", stops[-1])):
                    trip[f"{prefix}_stop_id"], trip[f"{prefix}_stop_name"] = stop[2], stop[3]
                revised.append(trip)
            if affected:
                changed[block] = revised
        pieces = [self.occ[~self.occ["Block"].isin(changed)]]
        for block, trips in changed.items():
            end = max(
                self.snapshot["timeline_end"],
                max(trip["end"] for trip in trips)
                + self.snapshot["settings"]["POST_ARRIVAL_MINUTES"]
                + 1,
            )
            if min(trip["start"] for trip in trips) < 0 or end >= BIG:
                raise ValueError("Shifted schedule falls outside the supported service-day range.")
            rows = build_schedule_rows(
                sorted(trips, key=lambda trip: (trip["start"], trip["trip_id"])),
                range(end),
                block,
                self.snapshot["clusters"],
                self.snapshot["settings"],
                occupancy_only=True,
            )
            pieces.append(self._occupancy(DataFrame(rows)))
        result = pd.concat(pieces, ignore_index=True)
        if result.duplicated(["Block", "Minute"]).any():
            raise ValueError("Rebuilt schedule contains duplicate vehicle/minute occupancy.")
        self.cache[key] = result
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        return result

    def score(self, frame: DataFrame) -> Tuple[int, Dict[str, int], np.ndarray]:
        """Count over-capacity bay/minute pairs; one vehicle has one row per minute."""
        bays = frame["Bay"].map(self.bay_index).to_numpy(dtype=int)
        minutes = frame["Minute"].to_numpy(dtype=int)
        if ((minutes < 0) | (minutes >= BIG)).any():
            raise ValueError("Timeline minute lies outside the supported service-day range.")
        keys = bays.astype(np.int64) * BIG + minutes
        unique, counts = np.unique(keys, return_counts=True)
        conflicts = unique[counts > self.cap[unique // BIG]]
        per = {
            self.bay_names[int(index)]: int(count)
            for index, count in zip(*np.unique(conflicts // BIG, return_counts=True))
        }
        return int(len(conflicts)), per, conflicts

    def detail(
        self, frame: DataFrame, conflict: np.ndarray
    ) -> Tuple[Dict[int, Tuple[str, ...]], Dict[str, int]]:
        """Describe the actual rebuilt rows involved in each conflict."""
        keys = frame["Bay"].map(self.bay_index).to_numpy(dtype=np.int64) * BIG + frame[
            "Minute"
        ].to_numpy(dtype=int)
        sub = frame.loc[np.isin(keys, conflict), ["route_end", "Status"]].copy()
        sub["key"] = keys[np.isin(keys, conflict)]
        who = {}
        kinds: Dict[str, int] = defaultdict(int)
        for key, group in sub.groupby("key", sort=False):
            who[int(key)] = tuple(sorted(set(group["route_end"])))
            waiting = group["Status"].isin(WAITING_STATUSES)
            kind = (
                "waiting/waiting"
                if waiting.all()
                else "boarding/boarding"
                if not waiting.any()
                else "boarding/waiting"
            )
            kinds[kind] += 1
        return who, dict(kinds)

    def key_label(self, key: int) -> Tuple[str, int]:
        """Convert an internal conflict key to its bay and minute."""
        return self.bay_names[int(key // BIG)], int(key % BIG)


class Feasibility:
    """Vectorized gap check over the block chains for a change's shifts."""

    def __init__(self, trips: DataFrame, chains: DataFrame) -> None:
        """Index the chains by trip so a change's shifts can be applied to every gap at once."""
        self.trips = trips.reset_index(drop=True)
        self.chains = chains.reset_index(drop=True)
        idx = {tid: i for i, tid in enumerate(self.trips["trip_id"])}
        empty = np.array([], dtype=int)
        if chains.empty:
            self.from_idx = self.to_idx = self.gap = self.floor = empty
        else:
            self.from_idx = self.chains["from_trip"].map(idx).to_numpy(dtype=int)
            self.to_idx = self.chains["to_trip"].map(idx).to_numpy(dtype=int)
            self.gap = self.chains["gap_min"].to_numpy(dtype=int)
            self.floor = np.where(
                self.chains["interline"].to_numpy(dtype=bool),
                MIN_INTERLINE_TURN_MINUTES,
                MIN_LAYOVER_MINUTES,
            )
        ends = [dict(route_end_of(t)) for _, t in self.trips.iterrows()]
        self.masks: Dict[str, np.ndarray] = {}
        for r in set(self.trips["route"]):
            self.masks[r] = (self.trips["route"] == r).to_numpy()
        for re_ in {k for e in ends for k in e}:
            self.masks[re_] = np.array([re_ in e for e in ends])
        self.first_bay = self.trips["first_bay"].to_numpy()
        self.start = self.trips["departure"].to_numpy(dtype=int)
        self.active_start = self.trips["start"].to_numpy(dtype=int)
        self.active_end = self.trips["end"].to_numpy(dtype=int)
        self.route = self.trips["route"].to_numpy()

    def trip_shift(self, shifts: Dict[str, int]) -> np.ndarray:
        """Minutes each trip moves under *shifts* (route-end and whole-route keys)."""
        out = np.zeros(len(self.trips), dtype=int)
        assigned = np.zeros(len(self.trips), dtype=bool)
        for key, k in shifts.items():
            if not isinstance(k, int) or isinstance(k, bool):
                raise ValueError("Shifts must be integer minutes.")
            if key not in self.masks:
                raise ValueError(f"Unknown shift selector: {key}")
            mask = self.masks[key]
            if (mask & assigned & (out != k)).any():
                raise ValueError("Overlapping trip selectors request different shifts.")
            out[mask] = k
            assigned |= mask
        return out

    def check(self, change: Change) -> Tuple[bool, str, Optional[int]]:
        """Whether the change keeps every affected gap at or above its minimum.

        Returns:
            ``(feasible, reason, tightest_gap_after)``. A gap already below
            its minimum is only rejected if the change shrinks it further.
        """
        try:
            ts = self.trip_shift(change.shifts)
        except ValueError as exc:
            return False, str(exc), None
        if ((self.active_start + ts < 0) | (self.active_end + ts >= BIG - 1)).any():
            return False, "Shifted trip falls outside the supported service-day range", None
        if not change.shifts or len(self.gap) == 0:
            return True, "", None
        new = self.gap + ts[self.to_idx] - ts[self.from_idx]
        changed = new != self.gap
        bad = changed & (new < self.gap) & (new < self.floor)
        if bad.any():
            i = int(np.argmax(bad))
            c = self.chains.iloc[i]
            reason = (
                f"gap {c['from_route']} {c['arrives']} -> {c['to_route']} {c['departs']} "
                f"({c['where']}) goes {int(self.gap[i])} -> {int(new[i])} min, "
                f"below {int(self.floor[i])}"
            )
            return False, reason, None
        return True, "", (int(new[changed].min()) if changed.any() else None)

    def off_clockface(self, change: Change) -> str:
        """Note naming routes whose :00/:30 cluster departures a shift takes off the clockface."""
        if not FLAG_OFF_CLOCKFACE or not change.shifts:
            return ""
        ts = self.trip_shift(change.shifts)
        m = (
            (self.first_bay != "")
            & (ts != 0)
            & (self.start % 30 == 0)
            & ((self.start + ts) % 30 != 0)
        )
        return f"departures off :00/:30: {', '.join(sorted(set(self.route[m])))}" if m.any() else ""


def describe_delta(
    std: Standard, before: Dict[int, Tuple[str, ...]], after: Dict[int, Tuple[str, ...]]
) -> Tuple[str, str]:
    """Conflicts removed and created, collapsed into windows with the route pair."""

    def windows(keys: List[int], src: Dict[int, Tuple[str, ...]]) -> str:
        out = []
        labelled = sorted((std.key_label(k), k) for k in keys)
        by_bay: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for (bay, minute), k in labelled:
            by_bay[bay].append((minute, k))
        for bay, lst in by_bay.items():
            start = prev = lst[0][0]
            pr = src[lst[0][1]]
            for minute, k in lst[1:] + [(None, None)]:
                if minute is not None and k is not None and minute == prev + 1 and src[k] == pr:
                    prev = minute
                    continue
                span = minutes_to_hhmm(start)
                if prev != start:
                    span += f"-{minutes_to_hhmm(prev)}"
                out.append(f"{bay} {span} {'/'.join(x.split(' ')[0] for x in pr)}")
                if minute is not None and k is not None:
                    start = prev = minute
                    pr = src[k]
        return "; ".join(out)

    removed = [k for k in before if k not in after]
    created = [k for k in after if k not in before]
    return windows(removed, before) if removed else "", windows(created, after) if created else ""


def single_candidates(standards: Dict[str, Standard]) -> List[Change]:
    """Every listed single change that exists in the timelines: one bay move or one shift."""
    known = set()
    for std in standards.values():
        known |= set(std.mask_route_end)
    first = next(iter(standards.values()))
    out: List[Change] = []
    for c in BAY_CANDIDATES:
        re_ = c["route_end"]
        if re_ not in known:
            logging.warning(
                "BAY_CANDIDATES: route-end '%s' not found in the timelines; skipped.", re_
            )
            continue
        current = set(first.occ.loc[first.occ["route_end"] == re_, "Bay"])
        for bay in c["bays"]:
            if current != {bay} and bay in first.bay_index:
                out.append(Change(moves={re_: bay}))
    for key, ks in SHIFT_CANDIDATES.items():
        if key not in known and not any(re_.startswith(key + " ") for re_ in known):
            logging.warning("SHIFT_CANDIDATES: '%s' not found in the timelines; skipped.", key)
            continue
        for k in ks:
            if k:
                out.append(Change(shifts={key: k}))
    unique = {expand_groups(change).key(): expand_groups(change) for change in out}
    return list(unique.values())


class Scorer:
    """Score rebuilt schedules against a baseline on every occupancy standard."""

    def __init__(self, standards: Dict[str, Standard], feas: Feasibility) -> None:
        """Initialize baseline scores and the shared block-feasibility model."""
        self.standards = standards
        self.feas = feas
        self.base_change = Change()
        self.base: Dict[str, Tuple[int, Dict[str, int], np.ndarray]] = {}
        self.rebase(Change())

    def rebase(self, change: Change) -> None:
        """Measure subsequent changes relative to this package."""
        self.base_change = change
        self.base = {
            name: standard.score(standard.apply(change))
            for name, standard in self.standards.items()
        }

    def quick(self, change: Change) -> Optional[Dict[str, int]]:
        """Return score improvements, or None when a combined change is infeasible."""
        try:
            combined = expand_groups(self.base_change.combine(change))
            if not self.feas.check(combined)[0]:
                return None
            return {
                name: self.base[name][0] - standard.score(standard.apply(combined))[0]
                for name, standard in self.standards.items()
            }
        except ValueError:
            return None

    def full(self, change: Change) -> Dict[str, Any]:
        """Describe feasibility, removed/created conflicts and the complete rebuilt score."""
        row: Dict[str, Any] = {
            "change": change.label(),
            "feasible": False,
            "reason": "",
            "tightest gap after": "",
            "note": "",
            "_change": change,
            "_improvement": {},
        }
        try:
            combined = expand_groups(self.base_change.combine(change))
            ok, reason, tightest = self.feas.check(combined)
            if not ok:
                raise ValueError(reason)
            rebuilt = {name: standard.apply(combined) for name, standard in self.standards.items()}
            row.update(feasible=True, note=self.feas.off_clockface(combined))
            row["tightest gap after"] = tightest if tightest is not None else ""
            for name, standard in self.standards.items():
                before, _, before_keys = self.base[name]
                total, per, keys = standard.score(rebuilt[name])
                old_who, _ = standard.detail(standard.apply(self.base_change), before_keys)
                who, kinds = standard.detail(rebuilt[name], keys)
                removed, created = describe_delta(standard, old_who, who)
                row.update(
                    {
                        f"{name} before": before,
                        f"{name} after": total,
                        f"{name} per bay after": " ".join(
                            f"{bay}:{count}" for bay, count in sorted(per.items())
                        ),
                        f"{name} removed": removed,
                        f"{name} created": created,
                        f"{name} kinds after": ", ".join(
                            f"{kind} {count}" for kind, count in sorted(kinds.items())
                        ),
                    }
                )
                row["_improvement"][name] = before - total
        except ValueError as exc:
            row["reason"] = str(exc)
            for name in self.standards:
                row[f"{name} before"] = self.base[name][0]
                row[f"{name} after"] = ""
        return row


def is_win(imp: Optional[Dict[str, int]]) -> bool:
    """Feasible, no worse on the required standards, and better by the minimum on at least one."""
    if not imp:
        return False
    if REQUIRE_NO_WORSE_ON == "any":
        no_worse = any(v >= 0 for v in imp.values())
    else:
        no_worse = all(v >= 0 for v in imp.values())
    return no_worse and max(imp.values()) >= MIN_IMPROVEMENT_MINUTES


def rank_key(imp: Dict[str, int]) -> tuple:
    """Sort key: biggest improvement first, judged on the last-listed standard first."""
    names = list(imp)
    return tuple(-imp[n] for n in reversed(names))


def _public(rows: List[Dict[str, Any]]) -> DataFrame:
    """Report rows without the internal ``_``-prefixed keys."""
    return DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])


def _search_packages(
    scorer: Scorer, pool: List[Change], standards: Dict[str, Standard]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retain neutral/intermediate packages in a bounded beam; report only final wins.

    This is a heuristic search, not a guarantee of a globally optimal package.
    Feasible neutral or temporarily worse partial packages can survive the beam.
    """
    beam = [Change()]
    best_per_step = []
    winners: Dict[tuple, Tuple[tuple, Change]] = {}
    seen: set[tuple] = set()
    scorer.rebase(Change())
    for step in range(1, MAX_CHANGES_PER_PACKAGE + 1):
        expanded = {}
        for package in beam:
            for addition in pool:
                if not package.compatible(addition):
                    continue
                try:
                    candidate = expand_groups(package.combine(addition))
                except ValueError:
                    continue
                key = candidate.key()
                if key in seen:
                    continue
                seen.add(key)
                improvement = scorer.quick(candidate)
                if improvement is not None:
                    expanded[key] = (rank_key(improvement), candidate, improvement)
        ranked = sorted(expanded.values(), key=lambda item: (item[0], item[1].key()))
        if not ranked:
            break
        beam = [candidate for _, candidate, _ in ranked[:BEAM_WIDTH]]
        wins = [(rank, candidate) for rank, candidate, improvement in ranked if is_win(improvement)]
        for rank, candidate in wins:
            winners[candidate.key()] = rank, candidate
        if wins:
            best = wins[0][1]
            result = scorer.full(best)
            row = {"step": step, "package": best.label()}
            for name in standards:
                row[f"{name} conflict minutes"] = result[f"{name} after"]
                row[f"{name} per bay"] = result[f"{name} per bay after"]
                row[f"{name} kinds"] = result[f"{name} kinds after"]
            row["tightest gap after"] = result["tightest gap after"]
            row["note"] = result["note"]
            best_per_step.append(row)
    finalists = sorted(winners.values(), key=lambda item: (item[0], item[1].key()))[:5]
    return best_per_step, [scorer.full(candidate) for _, candidate in finalists]


def validate_sweep_configuration() -> None:
    """Validate search limits, physical bay capacities and integer candidate shifts."""
    if not TIMELINES or not CLUSTER_STOPS:
        raise ValueError("TIMELINES and CLUSTER_STOPS must be populated.")
    if REQUIRE_NO_WORSE_ON not in {"all", "any"}:
        raise ValueError("REQUIRE_NO_WORSE_ON must be 'all' or 'any'.")
    values = {
        "BEAM_WIDTH": BEAM_WIDTH,
        "TOP_N": TOP_N,
        "MAX_CHANGES_PER_PACKAGE": MAX_CHANGES_PER_PACKAGE,
        "MIN_IMPROVEMENT_MINUTES": MIN_IMPROVEMENT_MINUTES,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for bay, capacity in CLUSTER_CAPACITY.items():
        if bay not in CLUSTER_STOPS.values() or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"Invalid bay/capacity configuration: {bay}={capacity}")
    if any(not isinstance(bay, str) or not bay or "," in bay for bay in CLUSTER_STOPS.values()):
        raise ValueError("Bay labels must be nonempty strings without commas.")
    for candidates in SHIFT_CANDIDATES.values():
        if any(not isinstance(value, int) or isinstance(value, bool) for value in candidates):
            raise ValueError("Shift candidates must be integer minutes.")
    for candidate in BAY_CANDIDATES:
        if set(candidate["bays"]) - set(CLUSTER_STOPS.values()):
            raise ValueError(f"Candidate contains unknown bays: {candidate}")
    if MIN_INTERLINE_TURN_MINUTES < 0 or MIN_LAYOVER_MINUTES < 0:
        raise ValueError("Minimum gaps must be nonnegative.")


def run_sweep() -> str:
    """Score every listed change, pair and package and write the sweep workbook.

    Feasibility is judged on the block chains of the first-listed standard:
    the schedule is the same across standards, only the occupancy assumptions
    differ.

    Returns:
        Path of the workbook written.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    standards: Dict[str, Standard] = {}
    trips: Optional[DataFrame] = None
    chains: Optional[DataFrame] = None
    schedule_identity = None
    validate_sweep_configuration()
    for name, folder in TIMELINES.items():
        df = load_timeline(folder)
        snapshot = df.attrs.get("schedule_snapshot")
        if snapshot is None:
            raise ValueError(
                "Rerun the revised exporter for every scenario before running the sweep."
            )
        identity = json.dumps(
            {
                "trips": sorted(snapshot["trips"], key=lambda trip: trip["trip_id"]),
                "clusters": snapshot["clusters"],
            },
            sort_keys=True,
        )
        if schedule_identity is not None and identity != schedule_identity:
            raise ValueError(
                "TIMELINES must use the same trips, times, stops and clusters; "
                "only occupancy assumptions may differ."
            )
        schedule_identity = identity
        t = build_trips(df)
        if trips is None or chains is None:
            trips, chains = t, build_block_chains(t)
        standards[name] = Standard(name, df, t)
    if trips is None or chains is None:
        raise ValueError("TIMELINES is empty; list at least one Step 1 scenario folder.")
    feas = Feasibility(trips, chains)
    scorer = Scorer(standards, feas)
    for n, (total, per, _) in scorer.base.items():
        logging.info("%s baseline: %d conflict minutes %s", n, total, dict(sorted(per.items())))

    # ---- singles
    singles = single_candidates(standards)
    logging.info("Scoring %d single changes.", len(singles))
    single_imp = [(c, scorer.quick(c)) for c in singles]
    win_rows = sorted(
        [scorer.full(c) for c, imp in single_imp if is_win(imp)],
        key=lambda r: rank_key(r["_improvement"]),
    )
    rejected_rows = []
    for c, imp in single_imp:
        if is_win(imp):
            continue
        r = scorer.full(c)
        if r["feasible"] and not r["reason"] and imp is not None:
            if max(imp.values()) < MIN_IMPROVEMENT_MINUTES:
                r["reason"] = "no improvement"
            else:
                r["reason"] = "worsens " + ", ".join(n for n, v in imp.items() if v < 0)
        rejected_rows.append(r)
    logging.info("%d single wins, %d rejected.", len(win_rows), len(rejected_rows))

    # ---- pairs: every compatible pair of feasible singles, scored quickly, best ones described
    pair_rows: List[Dict[str, Any]] = []
    if ENUMERATE_PAIRS:
        feasible = singles  # individually infeasible shifts may be feasible together
        scored = []
        n_pairs = 0
        for i in range(len(feasible)):
            for j in range(i + 1, len(feasible)):
                if not feasible[i].compatible(feasible[j]):
                    continue
                n_pairs += 1
                pair = feasible[i].combine(feasible[j])
                imp = scorer.quick(pair)
                if is_win(imp) and imp is not None:
                    scored.append((rank_key(imp), pair))
        scored.sort(key=lambda x: x[0])
        logging.info("%d pairs scored; %d are wins.", n_pairs, len(scored))
        pair_rows = [scorer.full(pair) for _, pair in scored[:TOP_N]]

    # ---- packages
    pool = singles + [r["_change"] for r in pair_rows]
    packages, finalist_rows = _search_packages(scorer, pool, standards)

    out_path = os.path.join(OUTPUT_FOLDER, f"{SCENARIO_LABEL}_sweep.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        base_rows = []
        for n, s in standards.items():
            total, per, conf = scorer.base[n]
            _, kinds = s.detail(s.occ, conf)
            base_rows.append(
                {
                    "standard": n,
                    "total": total,
                    **{f"Bay {b}": per.get(b, 0) for b in s.bay_names},
                    **kinds,
                }
            )
        DataFrame(base_rows).to_excel(writer, sheet_name="Baseline", index=False)
        _public(win_rows[:TOP_N]).to_excel(writer, sheet_name="Easy wins", index=False)
        if ENUMERATE_PAIRS:
            _public(pair_rows).to_excel(writer, sheet_name="Pairs", index=False)
        DataFrame(packages).to_excel(writer, sheet_name="Packages", index=False)
        _public(finalist_rows).to_excel(writer, sheet_name="Package finalists", index=False)
        _public(rejected_rows).to_excel(writer, sheet_name="Rejected", index=False)
        cfg = {
            "CLUSTER_NAME": CLUSTER_NAME,
            "CLUSTER_STOPS": CLUSTER_STOPS,
            "CLUSTER_CAPACITY": CLUSTER_CAPACITY,
            "TIMELINES": TIMELINES,
            "BAY_CANDIDATES": BAY_CANDIDATES,
            "SAME_BAY_GROUPS": SAME_BAY_GROUPS,
            "SHIFT_CANDIDATES": SHIFT_CANDIDATES,
            "SHIFT_TOGETHER": SHIFT_TOGETHER,
            "MIN_INTERLINE_TURN_MINUTES": MIN_INTERLINE_TURN_MINUTES,
            "MIN_LAYOVER_MINUTES": MIN_LAYOVER_MINUTES,
            "REQUIRE_NO_WORSE_ON": REQUIRE_NO_WORSE_ON,
            "MIN_IMPROVEMENT_MINUTES": MIN_IMPROVEMENT_MINUTES,
            "MAX_CHANGES_PER_PACKAGE": MAX_CHANGES_PER_PACKAGE,
            "BEAM_WIDTH": BEAM_WIDTH,
            "ENUMERATE_PAIRS": ENUMERATE_PAIRS,
        }
        DataFrame({"setting": list(cfg), "value": [str(v) for v in cfg.values()]}).to_excel(
            writer, sheet_name="Config used", index=False
        )
    logging.info("Sweep workbook for %s written to %s", CLUSTER_NAME, out_path)
    require_run_log(write_run_log(Path(out_path)))
    return out_path


# --------------------------------------------------------------------------------------------------
# RUN LOG
# --------------------------------------------------------------------------------------------------


class RunLogError(RuntimeError):
    """Raised when the required ``_runlog.txt`` sidecar could not be written."""


# Canonical version lives in utils/run_log.py -- keep this copy in sync.
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


def resolve_source_file() -> Optional[Path]:
    """Path of this script on disk, or ``None`` in an interactive session without ``__file__``."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def write_run_log(output_file: Path) -> bool:
    """Write the ``_runlog.txt`` sidecar for *output_file* (same folder, same stem).

    The log captures this script's CONFIGURATION block verbatim, between the
    ``# === BEGIN CONFIG ===`` / ``# === END CONFIG ===`` markers, and appends
    the effective runtime values, including notebook edits.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_file.with_name(f"{output_file.stem}_runlog.txt")

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
        "BAY CHANGE SWEEP RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output file:      {output_file}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "",
        "EFFECTIVE RUNTIME SETTINGS",
        json.dumps(
            {name: value for name, value in globals().items() if name.isupper()},
            indent=2,
            default=str,
        ),
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to %s", log_path)
    return True


def require_run_log(written: bool) -> None:
    """Raise :class:`RunLogError` when a run log failed and ``REQUIRE_RUN_LOG`` is set."""
    if not written and REQUIRE_RUN_LOG:
        raise RunLogError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )


# ==================================================================================================
# MAIN
# ==================================================================================================

_PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "path\\to\\",
    "path/to/",
)


def _is_placeholder_path(p: str) -> bool:
    """Return True if *p* still points at a default placeholder location."""
    s = str(p).lower()
    return any(marker in s for marker in _PLACEHOLDER_MARKERS)


def main() -> int:
    """Run the discover pass or the sweep, depending on ``DISCOVER_ONLY``.

    Returns:
        Process exit code: 0 on success, 1 if the input or configuration is
        invalid or the required run log could not be written, 2 if required
        CONFIGURATION values are still placeholders or a Step 1 folder does not exist.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    paths = dict(TIMELINES)
    paths["OUTPUT_FOLDER"] = OUTPUT_FOLDER
    unset = [name for name, p in paths.items() if _is_placeholder_path(p)]
    if unset:
        logging.warning(
            "Default placeholder paths detected for: %s. Update the CONFIGURATION section "
            "before running.",
            ", ".join(unset),
        )
        return 2
    missing = [name for name, folder in TIMELINES.items() if not os.path.isdir(folder)]
    if missing:
        logging.warning(
            "Step 1 folder not found for: %s. Run block_status_timeline_exporter.py with the "
            "matching SCENARIO_NAME first.",
            ", ".join(missing),
        )
        return 2
    try:
        if DISCOVER_ONLY:
            run_discover()
        else:
            run_sweep()
    except (RunLogError, ValueError, OSError, KeyError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
