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
gaps between them and where they are spent, and the interline hand-offs whose
gaps a time shift has to respect. It also logs a ``BAY_CANDIDATES`` /
``SHIFT_CANDIDATES`` stub to paste into this CONFIGURATION.

Two gaps are reported between consecutive trips of a block. The schedule gap
runs from the previous trip's scheduled arrival (last stop) to the next trip's
scheduled departure (first stop): the layover a timetable shows. The occupancy
gap runs from the previous trip's last scheduled time (last-stop departure) to
the next trip's first (first-stop arrival); a scheduled hold at the first stop
belongs to the next trip, so this gap is shorter than the schedule gap when
there is one.

Routes are identified by route_id. Reports and CONFIGURATION name a route by
its short name, else its long name, else its route_id; where two routes would
share a name the route_id is added in brackets (``Express [R12]``). Names may
contain spaces. A route_id is accepted wherever a route name is.

Sweep (``DISCOVER_ONLY = False``)
---------------------------------
Every listed single change is applied to the scheduled trips. Occupancy is
rebuilt for affected blocks, including loading, arrival buffers and layover
thresholds. A route-end shift moves the entire trips serving that end. Each
change is also checked against the block chains and rejected, with the
reason, if it shrinks a schedule gap below its minimum or makes a trip begin
before the previous trip on its vehicle ends (a negative occupancy gap).
Counts are shown per standard and never summed across them; conflicts are
broken out by kind (boarding/boarding, boarding/waiting, waiting/waiting)
rather than weighted.

Inputs
------
- One Step 1 output folder per standard (``TIMELINES``): the combined CSV
  ``block_status_timeline_exporter.py`` writes, or its ``block_*.xlsx``
  workbooks. Sweeps require the revised exporter's ``schedule_snapshot.json``
  and ``timeline_manifest.json`` sidecars. Rerun Step 1 for every standard.
  Discover can inspect legacy timelines, but exact sweeps cannot use them.
  Both refuse runs that Step 1 exported in trip-only mode (trips without a
  block_id), because block chains need every trip's vehicle.

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
- With REPORTS_ONLY (or WRITE_BAY_REPORTS after a sweep), selected bay-only
  reports in ``<SCENARIO_LABEL>_bay_reports``: one ``<cluster>_Conflicts``
  workbook per configuration and standard, and a comparison workbook.
  Each report includes Summary, Assignments, Conflict Events, Overflow Events,
  Conflict Minutes, Changes vs Baseline, AllStops and individual space sheets.
  Complete rebuilt timelines are saved as CSVs. Reports include all cluster
  presence, not only conflicting buses. Baseline is rebuilt and compared with
  the input timeline before any reports are published. Times and blocks must
  remain unchanged; each detailed count must equal the sweep scorer.
- A ``_runlog.txt`` sidecar next to each workbook capturing the verbatim
  CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section, run once with
``DISCOVER_ONLY = True``, paste the candidates you are willing to consider
into ``BAY_CANDIDATES`` / ``SHIFT_CANDIDATES``, set ``DISCOVER_ONLY = False``
and run again, from a shell, ArcGIS Pro's Python window, or a Jupyter
notebook.

For detailed reports on bay assignments you have already chosen, list them in
``BAY_REPORT_CONFIGURATIONS``, set ``DISCOVER_ONLY = False`` and
``REPORTS_ONLY = True``, and run: no candidate search runs, and each
configuration is compared with the baseline. Keep only the standards for which
you have current Step 1 outputs.

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
import re
import tempfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
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
# locked. <route> is the route's name as the discover pass lists it, or its
# route_id.
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
# Use the route name alone ("101") to shift every trip of the route. Every
# gap on every affected block is checked against the minimums below.
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
# Minimum schedule gaps (scheduled arrival to next scheduled departure). A shift
# is also rejected when a trip would begin before the previous one on its
# vehicle ends.
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

# --- Detailed bay-only reports --------------------------------------------------------------------
# REPORTS_ONLY = True writes only the reports below: no candidate search runs.
# DISCOVER_ONLY takes precedence. With REPORTS_ONLY = False, WRITE_BAY_REPORTS
# = True adds the same reports after the sweep.
REPORTS_ONLY = False
WRITE_BAY_REPORTS = False
# Baseline is always included. These are absolute assignments, not sequential
# edits. They need not appear in BAY_CANDIDATES. Unknown movements fail loudly.
BAY_REPORT_CONFIGURATIONS: Dict[str, Dict[str, str]] = {
    # "option_a": {"101 arrive": "B", "102 depart": "A"},
}
# Match Step 2. These are separate physical spaces, NOT passenger bays.
REPORT_OVERFLOW_ROUTING: Dict[str, List[str]] = {
    "layover_bay_A": ["LAYOVER"],
    "layover_bay_B": ["LONG BREAK"],
}
REPORT_OVERFLOW_CAPACITY: Dict[str, int] = {
    "layover_bay_A": 1,
    "layover_bay_B": 1,
}
# Optional historical checks, keyed by configuration then exact TIMELINES name.
# Leave empty when testing a new feed or different occupancy assumptions.
# Example: {"baseline": {"Likely conflict": 120}, "option_a": {"Likely conflict": 64}}
EXPECTED_REPORT_TOTALS: Dict[str, Dict[str, int]] = {}

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
    "route_name",
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
    "schedule_gap_min",
    "occupancy_gap_min",
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
    """Build one timeline row with explicit visit role and scheduled trip bounds.

    ``Estimated Time`` is True when the row's stop visit has an interpolated
    time; ``Trip Start Minute`` / ``Trip End Minute`` are the trip's occupancy
    bounds (first-stop arrival, last-stop departure).
    """
    return {
        "Timestamp": minutes_to_hhmm(minute),
        "Block": block_id,
        "Route": trip["route_id"] if trip else "",
        "Route Short Name": trip["route_short_name"] if trip else "",
        "Route Long Name": trip.get("route_long_name", "") if trip else "",
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
        "Estimated Time": bool(trip) and stop_seq in trip.get("estimated_stop_sequences", ()),
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
    """Recalculate loading, post-arrival occupancy and layovers between trips.

    Presence follows each trip's occupancy bounds -- ``start`` (first-stop
    arrival) to ``end`` (last-stop departure) -- so a scheduled hold before the
    next departure stays occupied and the layover is classified on the
    occupancy gap between them. The Arrival/Departure Time columns quote the
    schedule: the previous trip's last-stop arrival and the next trip's
    first-stop departure.
    """
    previous = [trip for trip in all_trips if trip["end"] < minute]
    upcoming = [trip for trip in all_trips if trip["start"] > minute]
    prev = max(previous, key=lambda trip: trip["end"]) if previous else None
    nxt = min(upcoming, key=lambda trip: trip["start"]) if upcoming else None
    prev_id = prev["trip_id"] if prev else ""
    next_id = nxt["trip_id"] if nxt else ""
    arr = minutes_to_hhmm(prev["arrival"]) if prev else ""
    dep = minutes_to_hhmm(nxt["departure"]) if nxt else ""
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
        occupancy_gap = nxt["start"] - prev["end"]
        status, location = gap_status(occupancy_gap, same_place, settings)
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
        "Route Long Name",
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
# ROUTE NAMES AND ROUTE-ENDS
# ==================================================================================================


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
def route_display_name(short_name: object, long_name: object, route_id: object) -> str:
    """Name a route for reports: its short name, else its long name, else its route_id.

    GTFS requires only one of route_short_name and route_long_name, so either
    may be blank; route_id, which is always present, identifies the route.
    """
    for value in (short_name, long_name, route_id):
        text = "" if value is None or pd.isna(value) else str(value).strip()
        if text:
            return text
    return ""


@dataclass(frozen=True, order=True)
class RouteEnd:
    """What a candidate moves or shifts: a route's arrivals, departures or through visits.

    ``route`` is the route_id. ``role`` is ``"arrive"``, ``"depart"``,
    ``"through"``, or ``""`` for every trip of the route; ``direction`` (a
    direction_id) applies to ``"through"`` only. ``name`` is the route's report
    label and takes no part in comparisons.
    """

    route: str
    role: str = ""
    direction: str = ""
    name: str = field(default="", compare=False)

    @property
    def label(self) -> str:
        """The route's report label, or its route_id when none is attached."""
        return self.name or self.route

    def __str__(self) -> str:
        """Render as CONFIGURATION writes it, e.g. ``"101 arrive"`` or ``"101 through 0"``."""
        role = f"{self.role} {self.direction}".strip() if self.role == "through" else self.role
        return f"{self.label} {role}".strip()


class RouteNames:
    """Route labels for reports, and the route each CONFIGURATION name refers to.

    A route's label is its display name (see :func:`route_display_name`); where
    that name is shared with another route, or is another route's route_id, the
    route_id is added: ``"Express [R12]"``. CONFIGURATION may name a route by
    its label, its display name or its route_id, provided the name fits one
    route only.
    """

    def __init__(self, display: Dict[str, str]) -> None:
        """Label each route from its display name (``route_id -> name``)."""
        counts = Counter(display.values())
        self.label = {
            route: name
            if counts[name] == 1 and (name == route or name not in display)
            else f"{name} [{route}]"
            for route, name in display.items()
        }
        self.routes: Dict[str, set[str]] = defaultdict(set)
        for route, name in display.items():
            for alias in (route, name, self.label[route]):
                self.routes[alias].add(route)

    def parse(self, text: str) -> Optional[RouteEnd]:
        """Read ``"<route>"``, ``"<route> arrive|depart"`` or ``"<route> through [<dir>]"``.

        Returns:
            The route-end, or ``None`` if no route has that name.

        Raises:
            ValueError: If the text fits more than one route or reading.
        """
        text = str(text).strip()
        readings = [(text, "", "")]
        for role in ("arrive", "depart"):
            if text.endswith(" " + role):
                readings.append((text[: -len(role)].strip(), role, ""))
        through = re.fullmatch(r"(.+) through(?: (\S+))?", text)
        if through:
            readings.append((through.group(1).strip(), "through", through.group(2) or ""))
        found = {
            RouteEnd(route, role, direction, self.label[route])
            for name, role, direction in readings
            for route in self.routes.get(name, ())
        }
        if len(found) > 1:
            raise ValueError(
                f"{text!r} could mean {', '.join(sorted(map(str, found)))}; name the route "
                "by the label the discover pass lists, or by its route_id."
            )
        return found.pop() if found else None


def snapshot_route_names(snapshot: dict[str, Any]) -> RouteNames:
    """Names of the routes in a Step 1 schedule snapshot."""
    return RouteNames(
        {
            trip["route_id"]: route_display_name(
                trip["route_short_name"], trip.get("route_long_name"), trip["route_id"]
            )
            for trip in snapshot["trips"]
        }
    )


# ==================================================================================================
# TRIPS, ROUTE-ENDS, BLOCK CHAINS
# ==================================================================================================


def build_trips(df: DataFrame) -> DataFrame:
    """Build one row per trip with scheduled times, endpoints, and block details.

    Uses the run's schedule snapshot when there is one. A legacy timeline has
    no snapshot: departure is then the first stop's scheduled departure (the
    DEPART row's Departure Time, after any scheduled hold) and start its
    Arrival Time; end and arrival are the last stop's scheduled arrival (the
    largest Arrival Time on the trip's ARRIVE rows). Missing times on
    individual timeline rows are excluded.

    Args:
        df: Step 1 timeline with Minute and Bay columns added by load_timeline.

    Returns:
        Trip records sorted by block and start time.

    Raises:
        ValueError: If a legacy trip has no readable arrival times, no DEPART
            row, no readable first-stop arrival or departure time, or no
            readable final arrival.
    """
    if df.attrs.get("schedule_snapshot") is not None:
        return trips_from_snapshot(df.attrs["schedule_snapshot"])
    logging.warning("Discover is inferring legacy trip bounds. Rerun Step 1 before the sweep.")
    rows = []
    on_trip = df[
        (df["Trip ID"] != "") & df["Status"].isin(BAY_STATUSES | {"TRAVELING BETWEEN STOPS"})
    ]
    for tid, g in on_trip.groupby("Trip ID", sort=False):
        block = g["Block"].iloc[0]
        dep = g[g["Status"] == "DEPART"]
        arr_minutes = g["Arrival Time"].map(timestamp_to_minutes)
        # Pandas can convert None to NaN; dropna handles both.
        arr_times = arr_minutes.dropna().astype(int).tolist()
        if not arr_times:
            raise ValueError(
                f"Trip {tid!r} in block {block!r} has no readable arrival times. "
                "Cannot determine its end time or build reliable block chains."
            )
        if dep.empty:
            raise ValueError(f"Trip {tid!r} in block {block!r} has no DEPART row. Rerun Step 1.")
        first = dep.iloc[0]
        # Read the scheduled departure rather than the timeline row's minute; the first
        # stop's arrival is kept separately as the trip's occupancy start.
        departure = timestamp_to_minutes(first["Departure Time"])
        start = timestamp_to_minutes(first["Arrival Time"])
        if departure is None or start is None:
            raise ValueError(
                f"Trip {tid!r} in block {block!r} has no readable first-stop "
                "arrival and departure times."
            )
        arrivals = arr_minutes[g["Status"].eq("ARRIVE")].dropna()
        if arrivals.empty:
            raise ValueError(
                f"Trip {tid!r} in block {block!r} has no readable final arrival. Rerun Step 1."
            )
        end = int(arrivals.max())
        last_rows = g[arr_minutes == end]
        if last_rows.empty:
            last_rows = g
        last = last_rows.sort_values("Minute").iloc[-1]
        through = g[(g["Status"] == "ARRIVE/DEPART") & (g["Bay"] != "")]["Bay"]
        rows.append(
            {
                "trip_id": tid,
                "block": first["Block"],
                "route": first["Route"],
                "route_name": route_display_name(
                    first["Route Short Name"], first.get("Route Long Name"), first["Route"]
                ),
                "headsign": first["Trip Headsign"],
                "direction": first["Direction"],
                "start": start,
                "end": end,
                "first_stop": first["Stop ID"],
                "first_bay": CLUSTER_STOPS.get(first["Stop ID"], ""),
                "last_stop": last["Stop ID"],
                "last_bay": CLUSTER_STOPS.get(last["Stop ID"], ""),
                "through_bays": ",".join(sorted(set(through))),
                "departure": departure,
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
    names = RouteNames(dict(zip(trips["route"], trips["route_name"])))
    trips["route_name"] = trips["route"].map(names.label)
    return trips.sort_values(["block", "start"]).reset_index(drop=True)


def route_end_of(trip: pd.Series) -> List[Tuple[RouteEnd, str]]:
    """Which route-ends this trip contributes at the cluster: ``[(route_end, bay), ...]``."""
    out = []
    route, name = str(trip["route"]), str(trip.get("route_name", "") or "")
    if trip["first_bay"]:
        out.append((RouteEnd(route, "depart", "", name), trip["first_bay"]))
    if trip["last_bay"]:
        out.append((RouteEnd(route, "arrive", "", name), trip["last_bay"]))
    for bay in [b for b in str(trip["through_bays"]).split(",") if b]:
        out.append((RouteEnd(route, "through", str(trip["direction"]), name), bay))
    return out


def build_block_chains(trips: DataFrame) -> DataFrame:
    """Consecutive trip pairs on each block with both gaps and where they are spent.

    ``schedule_gap_min`` runs from the previous trip's scheduled arrival to the
    next trip's scheduled departure: the layover a timetable shows, which the
    minimums in CONFIGURATION apply to. ``occupancy_gap_min`` runs from the
    previous trip's last scheduled time (last-stop departure) to the next
    trip's first (first-stop arrival): the time the vehicle belongs to neither
    trip, which may not drop below zero.
    """
    rows = []
    for block, g in trips.groupby("block", sort=False):
        g = g.sort_values("start")
        prev = None
        for _, t in g.iterrows():
            if prev is not None:
                if prev["last_stop"] == t["first_stop"]:
                    bay = f" (Bay {prev['last_bay']})" if prev["last_bay"] else ""
                    where = f"same stop{bay}"
                elif prev["last_bay"] and t["first_bay"]:
                    where = f"cluster (Bay {prev['last_bay']} to Bay {t['first_bay']})"
                else:
                    where = "elsewhere"
                arriving = RouteEnd(prev["route"], "arrive", "", prev["route_name"])
                departing = RouteEnd(t["route"], "depart", "", t["route_name"])
                rows.append(
                    {
                        "block": block,
                        "schedule_gap_min": int(t["departure"] - prev["arrival"]),
                        "occupancy_gap_min": int(t["start"] - prev["end"]),
                        "where": where,
                        "from_route": prev["route_name"],
                        "from_headsign": prev["headsign"],
                        "from_trip": prev["trip_id"],
                        "arrives": minutes_to_hhmm(int(prev["arrival"])),
                        "to_route": t["route_name"],
                        "to_headsign": t["headsign"],
                        "to_trip": t["trip_id"],
                        "departs": minutes_to_hhmm(int(t["departure"])),
                        "interline": prev["route"] != t["route"],
                        "from_route_end": str(arriving) if prev["last_bay"] else "",
                        "to_route_end": str(departing) if t["first_bay"] else "",
                    }
                )
            prev = t
    return DataFrame(rows, columns=CHAIN_COLUMNS)


def build_route_ends(trips: DataFrame, chains: DataFrame) -> DataFrame:
    """Inventory of route-ends at the cluster with bays, visits, times and tightest gaps."""
    visits: Dict[RouteEnd, List[Tuple[str, int, str]]] = defaultdict(list)  # [(bay, min, trip)]
    for _, t in trips.iterrows():
        for route_end, bay in route_end_of(t):
            if route_end.role == "through":
                for visit_bay, minute in t["through_visits"]:
                    if visit_bay == bay:
                        visits[route_end].append((bay, int(minute), t["trip_id"]))
            else:
                minute = t["departure"] if route_end.role == "depart" else t["arrival"]
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
                "route_end": str(route_end),
                "route_name": route_end.label,
                "bays": ",".join(bays),
                "visits_per_day": len(lst),
                "first": minutes_to_hhmm(min(mins)),
                "last": minutes_to_hhmm(max(mins)),
                "min_schedule_gap_before": (
                    int(before["schedule_gap_min"].min()) if not before.empty else ""
                ),
                "min_schedule_gap_after": (
                    int(after["schedule_gap_min"].min()) if not after.empty else ""
                ),
                "interlines_with": ",".join(sorted(partners)),
            }
        )
    cols = [
        "route_end",
        "route_name",
        "bays",
        "visits_per_day",
        "first",
        "last",
        "min_schedule_gap_before",
        "min_schedule_gap_after",
        "interlines_with",
    ]
    return DataFrame(rows, columns=cols)


def interline_summary(chains: DataFrame) -> DataFrame:
    """Tightest hand-off per ordered route pair, with the count and the tightest example."""
    cols = [
        "from_route",
        "to_route",
        "hand_offs_per_day",
        "min_schedule_gap_min",
        "max_schedule_gap_min",
        "min_occupancy_gap_min",
        "tightest_example",
        "shift_together_needed",
    ]
    if chains.empty:
        return DataFrame(columns=cols)
    inter = chains[chains["interline"]]
    rows = []
    for (a, b), g in inter.groupby(["from_route", "to_route"]):
        tight = g.sort_values("schedule_gap_min").iloc[0]
        rows.append(
            {
                "from_route": a,
                "to_route": b,
                "hand_offs_per_day": len(g),
                "min_schedule_gap_min": int(g["schedule_gap_min"].min()),
                "max_schedule_gap_min": int(g["schedule_gap_min"].max()),
                "min_occupancy_gap_min": int(g["occupancy_gap_min"].min()),
                "tightest_example": (
                    f"{tight['arrives']} {tight['from_headsign']} -> "
                    f"{tight['departs']} {tight['to_headsign']} ({tight['where']})"
                ),
                "shift_together_needed": bool(
                    g["schedule_gap_min"].min() <= MIN_INTERLINE_TURN_MINUTES + 3
                ),
            }
        )
    return DataFrame(rows, columns=cols).sort_values("min_schedule_gap_min")


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
    for route in sorted(set(route_ends["route_name"])):
        lines.append(f'    # "{route}": [-3, -2, -1, 1, 2, 3],  # whole route')
    lines.append("}")
    lines.append("SHIFT_TOGETHER: List[List[str]] = [")
    if not inter.empty:
        for _, r in inter[inter["shift_together_needed"]].iterrows():
            lines.append(
                f'    # ["{r["from_route"]} arrive", "{r["to_route"]} depart"],'
                f"  # hand-off as tight as {r['min_schedule_gap_min']} min"
            )
    lines.append("]")
    return "\n".join(lines)


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
@contextmanager
def open_report_workbook(path: str) -> Iterator[pd.ExcelWriter]:
    """Open an openpyxl workbook writer whose cleanup never hides a writing error.

    Build every sheet's data before entering, so a failed calculation never
    leaves an empty workbook behind. If the body raises, the writer is closed
    with any error from closing suppressed (an unfinished workbook cannot be
    saved), the partial file is removed and the original error propagates.
    """
    writer = pd.ExcelWriter(path, engine="openpyxl")
    try:
        yield writer
    except BaseException:
        try:
            writer.close()
        except Exception:  # noqa: BLE001 -- the error from the body is the one to report
            logging.debug("Discarding unfinished workbook %s.", path, exc_info=True)
        Path(path).unlink(missing_ok=True)
        raise
    writer.close()


def run_discover() -> str:
    """Write the discover workbook and log the CONFIGURATION stub.

    Every sheet is calculated before the workbook is opened, so a timeline that
    cannot be read stops the pass with its own error and writes no workbook.

    Returns:
        Path of the workbook written.

    Raises:
        ValueError: If no input scenarios are configured or trip times are invalid.
    """
    validate_sweep_configuration()
    sheets: List[Tuple[str, DataFrame]] = []
    stub = ""
    # Finish calculations first so a data error cannot be hidden by Excel cleanup.
    for tier, folder in TIMELINES.items():
        df = load_timeline(folder)
        trips = build_trips(df)
        chains = build_block_chains(trips)
        route_ends = build_route_ends(trips, chains)
        inter = interline_summary(chains)
        tag = "".join(ch for ch in tier if ch.isalnum())[:12]
        sheets += [(f"Route-ends {tag}", route_ends), (f"Block chains {tag}", chains)]
        if not inter.empty:
            sheets.append((f"Interlines {tag}", inter))
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
    sheets.append(("Config stub", DataFrame({"config_stub": stub.split("\n")})))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, f"{SCENARIO_LABEL}_discover.xlsx")
    with open_report_workbook(out_path) as writer:
        for sheet_name, frame in sheets:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
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
        self,
        moves: Optional[Dict[RouteEnd, str]] = None,
        shifts: Optional[Dict[RouteEnd, int]] = None,
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


@dataclass
class Selections:
    """The CONFIGURATION candidates and groups, read as route-ends of the loaded schedule."""

    bay_candidates: List[Tuple[RouteEnd, List[str]]] = field(default_factory=list)
    shift_candidates: List[Tuple[RouteEnd, List[int]]] = field(default_factory=list)
    same_bay_groups: List[List[RouteEnd]] = field(default_factory=list)
    shift_together: List[List[RouteEnd]] = field(default_factory=list)


def resolve_selections(names: RouteNames) -> Selections:
    """Read BAY_CANDIDATES, SHIFT_CANDIDATES and the groups as structured route-ends.

    A name no route has is skipped with a warning, as a route-end absent from
    the timelines is. A name that fits more than one route stops the sweep.
    """

    def read(text: str, setting: str) -> Optional[RouteEnd]:
        route_end = names.parse(text)
        if route_end is None:
            logging.warning("%s: no route in the timelines is named '%s'; skipped.", setting, text)
        return route_end

    selections = Selections()
    for candidate in BAY_CANDIDATES:
        route_end = read(candidate["route_end"], "BAY_CANDIDATES")
        if route_end is not None:
            selections.bay_candidates.append((route_end, list(candidate["bays"])))
    for key, shifts in SHIFT_CANDIDATES.items():
        route_end = read(key, "SHIFT_CANDIDATES")
        if route_end is not None:
            selections.shift_candidates.append((route_end, list(shifts)))
    for setting, groups, resolved in (
        ("SAME_BAY_GROUPS", SAME_BAY_GROUPS, selections.same_bay_groups),
        ("SHIFT_TOGETHER", SHIFT_TOGETHER, selections.shift_together),
    ):
        for group in groups:
            members = [read(member, setting) for member in group]
            resolved.append([member for member in members if member is not None])
    return selections


def expand_groups(change: Change, selections: Selections) -> Change:
    """Expand overlapping groups to closure and reject contradictory member assignments."""

    def expand(values: dict, groups: list) -> dict:
        result = dict(values)
        while True:
            before = dict(result)
            for group in groups:
                assigned = {result[key] for key in group if key in result}
                if len(assigned) > 1:
                    raise ValueError(
                        f"Conflicting assignments within group {', '.join(map(str, group))}."
                    )
                if assigned:
                    value = next(iter(assigned))
                    result.update({key: value for key in group})
            if before == result:
                return result

    return Change(
        expand(change.moves, selections.same_bay_groups),
        expand(change.shifts, selections.shift_together),
    )


def trips_from_snapshot(snapshot: dict[str, Any]) -> DataFrame:
    """Build complete trip metadata directly from scheduled visits, including zero-gap handoffs."""
    names = snapshot_route_names(snapshot)
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
                "route": trip["route_id"],
                "route_name": names.label[trip["route_id"]],
                "headsign": trip["trip_headsign"],
                "direction": trip["direction_id"],
                "start": int(trip["start"]),
                "end": int(trip["end"]),
                "departure": int(trip["departure"]),
                "arrival": int(trip["arrival"]),
                "first_stop": first[2],
                "first_bay": CLUSTER_STOPS.get(first[2], ""),
                "last_stop": last[2],
                "last_bay": CLUSTER_STOPS.get(last[2], ""),
                "through_bays": ",".join(sorted({bay for bay, _ in through})),
                "through_visits": through,
            }
        )
    trips = DataFrame(rows, columns=TRIP_COLUMNS)
    return trips.sort_values(["block", "start"]).reset_index(drop=True)


def load_schedule_snapshot(folder: str) -> Optional[dict[str, Any]]:
    """Load the canonical schedule belonging to a completed exporter run.

    Raises:
        ValueError: If the snapshot predates the current exporter, has no trips,
            or holds trips exported without a block_id (trip-only mode), whose
            vehicles' other trips the block chains would need.
    """
    manifest = read_run_manifest(folder)
    if manifest is None:
        return None
    path = verified_run_file(folder, manifest.get("schedule_snapshot", ""), manifest)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 2 or snapshot.get("interval_minutes") != 1:
        raise ValueError(
            f"The schedule snapshot in {folder} is from an earlier exporter. Rerun Step 1."
        )
    if not snapshot.get("trips"):
        raise ValueError("Schedule snapshot has no trips.")
    trip_only = snapshot.get("trip_only_trips", [])
    if trip_only:
        raise ValueError(
            f"{folder} was exported in trip-only mode: {len(trip_only)} trip(s) have no "
            "block_id, so their vehicles' other trips are unknown and block chains cannot be "
            "built. Fill in block_id, or filter those trips out, and rerun Step 1."
        )
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
        self.timeline = df.copy()
        self.timeline.attrs = {}
        self.trips = trips
        self.names = snapshot_route_names(snapshot)
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
        codes = {key: code for code, key in enumerate(self.shift_model.masks)}
        found = self.occ["route_end"].map(codes)
        self.mask_route_end = {
            key: found.eq(code).to_numpy() for key, code in codes.items() if key.role
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
        directions = frame["Direction"].fillna("").astype(str).where(roles.eq("through"), "")
        frame["route_end"] = [
            RouteEnd(route, role, direction, self.names.label.get(route, route))
            for route, role, direction in zip(frame["Route"].astype(str), roles, directions)
        ]
        return frame.reset_index(drop=True)

    def changed_block_trips(self, change: Change) -> Dict[str, List[dict[str, Any]]]:
        """Apply candidate decisions to copies of affected blocks' scheduled trips."""
        shifts = self.shift_model.trip_shift(change.shifts)
        by_trip = dict(zip(self.trips["trip_id"], shifts))
        moved_routes = {route_end.route for route_end in change.moves}
        changed: Dict[str, List[dict[str, Any]]] = {}
        for block, originals in self.block_trips.items():
            revised = []
            affected = False
            for original in originals:
                shift = int(by_trip[original["trip_id"]])
                if not shift and original["route_id"] not in moved_routes:
                    revised.append(original)
                    continue
                trip = copy.deepcopy(original)
                affected |= bool(shift)
                for bound in ("start", "end", "departure", "arrival"):
                    trip[bound] += shift
                stops = []
                for original_stop in trip["stop_times_sequence"]:
                    stop = list(original_stop)
                    role = "depart" if stop[5] else "arrive" if stop[6] else "through"
                    direction = str(trip["direction_id"]) if role == "through" else ""
                    route_end = RouteEnd(str(trip["route_id"]), role, direction)
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
        return changed

    def apply(self, change: Change) -> DataFrame:
        """Move scheduled visits/trips, then rebuild every affected block's complete occupancy."""
        if not change.moves and not change.shifts:
            return self.occ
        key = change.key()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        changed = self.changed_block_trips(change)
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
    ) -> Tuple[Dict[int, Tuple[RouteEnd, ...]], Dict[str, int]]:
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
    """Vectorized check of a change's shifts against the block chains.

    A shift may not shrink a schedule gap below its minimum, nor make a trip
    begin before the previous trip on its vehicle ends.
    """

    def __init__(self, trips: DataFrame, chains: DataFrame) -> None:
        """Index the chains by trip so a change's shifts can be applied to every gap at once."""
        self.trips = trips.reset_index(drop=True)
        self.chains = chains.reset_index(drop=True)
        idx = {tid: i for i, tid in enumerate(self.trips["trip_id"])}
        empty = np.array([], dtype=int)
        if chains.empty:
            self.from_idx = self.to_idx = self.floor = empty
            self.schedule_gap = self.occupancy_gap = empty
        else:
            self.from_idx = self.chains["from_trip"].map(idx).to_numpy(dtype=int)
            self.to_idx = self.chains["to_trip"].map(idx).to_numpy(dtype=int)
            self.schedule_gap = self.chains["schedule_gap_min"].to_numpy(dtype=int)
            self.occupancy_gap = self.chains["occupancy_gap_min"].to_numpy(dtype=int)
            self.floor = np.where(
                self.chains["interline"].to_numpy(dtype=bool),
                MIN_INTERLINE_TURN_MINUTES,
                MIN_LAYOVER_MINUTES,
            )
        ends = [dict(route_end_of(t)) for _, t in self.trips.iterrows()]
        labels = dict(zip(self.trips["route"], self.trips["route_name"]))
        self.masks: Dict[RouteEnd, np.ndarray] = {}
        for r in sorted(set(self.trips["route"])):
            self.masks[RouteEnd(r, name=labels[r])] = (self.trips["route"] == r).to_numpy()
        for re_ in sorted({k for e in ends for k in e}):
            self.masks[re_] = np.array([re_ in e for e in ends])
        self.first_bay = self.trips["first_bay"].to_numpy()
        self.start = self.trips["departure"].to_numpy(dtype=int)
        self.active_start = self.trips["start"].to_numpy(dtype=int)
        self.active_end = self.trips["end"].to_numpy(dtype=int)
        self.route = self.trips["route_name"].to_numpy()

    def trip_shift(self, shifts: Dict[RouteEnd, int]) -> np.ndarray:
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
        """Whether the change keeps every affected block chain workable.

        Returns:
            ``(feasible, reason, tightest_schedule_gap_after)``. A schedule gap
            already below its minimum is only rejected if the change shrinks it
            further; an occupancy gap may never drop below zero.
        """
        try:
            ts = self.trip_shift(change.shifts)
        except ValueError as exc:
            return False, str(exc), None
        if ((self.active_start + ts < 0) | (self.active_end + ts >= BIG - 1)).any():
            return False, "Shifted trip falls outside the supported service-day range", None
        if not change.shifts or len(self.schedule_gap) == 0:
            return True, "", None
        moved = ts[self.to_idx] - ts[self.from_idx]
        changed = moved != 0
        schedule_gap = self.schedule_gap + moved
        occupancy_gap = self.occupancy_gap + moved
        short = changed & (schedule_gap < self.schedule_gap) & (schedule_gap < self.floor)
        if short.any():
            i = int(np.argmax(short))
            c = self.chains.iloc[i]
            reason = (
                f"schedule gap {c['from_route']} {c['arrives']} -> {c['to_route']} "
                f"{c['departs']} ({c['where']}) goes {int(self.schedule_gap[i])} -> "
                f"{int(schedule_gap[i])} min, below {int(self.floor[i])}"
            )
            return False, reason, None
        overlap = changed & (occupancy_gap < 0)
        if overlap.any():
            i = int(np.argmax(overlap))
            c = self.chains.iloc[i]
            reason = (
                f"trip {c['to_trip']} ({c['to_route']}) would begin {-int(occupancy_gap[i])} "
                f"min before trip {c['from_trip']} ({c['from_route']}) ends on block "
                f"{c['block']}"
            )
            return False, reason, None
        return True, "", (int(schedule_gap[changed].min()) if changed.any() else None)

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
    std: Standard,
    before: Dict[int, Tuple[RouteEnd, ...]],
    after: Dict[int, Tuple[RouteEnd, ...]],
) -> Tuple[str, str]:
    """Conflicts removed and created, collapsed into windows with the route pair."""

    def windows(keys: List[int], src: Dict[int, Tuple[RouteEnd, ...]]) -> str:
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
                out.append(f"{bay} {span} {'/'.join(x.label for x in pr)}")
                if minute is not None and k is not None:
                    start = prev = minute
                    pr = src[k]
        return "; ".join(out)

    removed = [k for k in before if k not in after]
    created = [k for k in after if k not in before]
    return windows(removed, before) if removed else "", windows(created, after) if created else ""


def single_candidates(standards: Dict[str, Standard], selections: Selections) -> List[Change]:
    """Every listed single change that exists in the timelines: one bay move or one shift."""
    known: set[RouteEnd] = set()
    for std in standards.values():
        known |= set(std.mask_route_end)
    first = next(iter(standards.values()))
    out: List[Change] = []
    for re_, bays in selections.bay_candidates:
        if re_ not in known:
            logging.warning(
                "BAY_CANDIDATES: route-end '%s' not found in the timelines; skipped.", re_
            )
            continue
        current = set(first.occ.loc[first.occ["route_end"] == re_, "Bay"])
        for bay in bays:
            if current != {bay} and bay in first.bay_index:
                out.append(Change(moves={re_: bay}))
    for key, ks in selections.shift_candidates:
        whole_route = not key.role and any(re_.route == key.route for re_ in known)
        if key not in known and not whole_route:
            logging.warning("SHIFT_CANDIDATES: '%s' not found in the timelines; skipped.", key)
            continue
        for k in ks:
            if k:
                out.append(Change(shifts={key: k}))
    unique = {
        expand_groups(change, selections).key(): expand_groups(change, selections) for change in out
    }
    return list(unique.values())


class Scorer:
    """Score rebuilt schedules against a baseline on every occupancy standard."""

    def __init__(
        self, standards: Dict[str, Standard], feas: Feasibility, selections: Selections
    ) -> None:
        """Initialize baseline scores, the shared block-feasibility model and the groups."""
        self.standards = standards
        self.feas = feas
        self.selections = selections
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
            combined = expand_groups(self.base_change.combine(change), self.selections)
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
            "tightest schedule gap after": "",
            "note": "",
            "_change": change,
            "_improvement": {},
        }
        try:
            combined = expand_groups(self.base_change.combine(change), self.selections)
            ok, reason, tightest = self.feas.check(combined)
            if not ok:
                raise ValueError(reason)
            rebuilt = {name: standard.apply(combined) for name, standard in self.standards.items()}
            row.update(feasible=True, note=self.feas.off_clockface(combined))
            row["tightest schedule gap after"] = tightest if tightest is not None else ""
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
                    candidate = expand_groups(package.combine(addition), scorer.selections)
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
            row["tightest schedule gap after"] = result["tightest schedule gap after"]
            row["note"] = result["note"]
            best_per_step.append(row)
    finalists = sorted(winners.values(), key=lambda item: (item[0], item[1].key()))[:5]
    return best_per_step, [scorer.full(candidate) for _, candidate in finalists]


def validate_sweep_configuration() -> None:
    """Validate search limits, physical bay capacities and integer candidate shifts."""
    if not TIMELINES:
        raise ValueError("TIMELINES is empty; list at least one Step 1 scenario folder.")
    if not CLUSTER_STOPS:
        raise ValueError("CLUSTER_STOPS is empty; map the cluster's stop_ids to bay labels.")
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
        if not isinstance(candidate.get("route_end"), str):
            raise ValueError(f"Each bay candidate needs a route_end string: {candidate}")
        if set(candidate["bays"]) - set(CLUSTER_STOPS.values()):
            raise ValueError(f"Candidate contains unknown bays: {candidate}")
    names = [*SHIFT_CANDIDATES, *(m for g in [*SAME_BAY_GROUPS, *SHIFT_TOGETHER] for m in g)]
    if any(not isinstance(name, str) for name in names):
        raise ValueError("Shift candidates and groups must name route-ends as strings.")
    if MIN_INTERLINE_TURN_MINUTES < 0 or MIN_LAYOVER_MINUTES < 0:
        raise ValueError("Minimum gaps must be nonnegative.")


def load_sweep_standards() -> Tuple[Dict[str, Standard], DataFrame, DataFrame, RouteNames, int]:
    """Load compatible current-run inputs for either the search or detailed reports."""
    standards: Dict[str, Standard] = {}
    trips: Optional[DataFrame] = None
    chains: Optional[DataFrame] = None
    names: Optional[RouteNames] = None
    estimated_visits = 0
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
            names = snapshot_route_names(snapshot)
            estimated_visits = int(snapshot.get("estimated_stop_visits", 0))
        standards[name] = Standard(name, df, t)
    if trips is None or chains is None or names is None:
        raise ValueError("TIMELINES is empty; list at least one Step 1 scenario folder.")
    if estimated_visits:
        logging.warning(
            "The schedule has %d stop visit(s) with interpolated times; conflicts involving "
            "them rest on estimated times.",
            estimated_visits,
        )
    return standards, trips, chains, names, estimated_visits


def run_sweep() -> str:
    """Score every listed change, pair and package and write the sweep workbook.

    Feasibility is judged on the block chains of the first-listed standard:
    the schedule is the same across standards, only the occupancy assumptions
    differ.

    Returns:
        Path of the workbook written.
    """
    standards, trips, chains, names, estimated_visits = load_sweep_standards()
    selections = resolve_selections(names)
    feas = Feasibility(trips, chains)
    scorer = Scorer(standards, feas, selections)
    for n, (total, per, _) in scorer.base.items():
        logging.info("%s baseline: %d conflict minutes %s", n, total, dict(sorted(per.items())))

    # ---- singles
    singles = single_candidates(standards, selections)
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
        "REPORTS_ONLY": REPORTS_ONLY,
        "WRITE_BAY_REPORTS": WRITE_BAY_REPORTS,
        "BAY_REPORT_CONFIGURATIONS": BAY_REPORT_CONFIGURATIONS,
        "REPORT_OVERFLOW_ROUTING": REPORT_OVERFLOW_ROUTING,
        "REPORT_OVERFLOW_CAPACITY": REPORT_OVERFLOW_CAPACITY,
        "EXPECTED_REPORT_TOTALS": EXPECTED_REPORT_TOTALS,
        "Stop visits with interpolated times (Step 1)": estimated_visits,
    }
    sheets = [
        ("Baseline", DataFrame(base_rows)),
        ("Easy wins", _public(win_rows[:TOP_N])),
        *([("Pairs", _public(pair_rows))] if ENUMERATE_PAIRS else []),
        ("Packages", DataFrame(packages)),
        ("Package finalists", _public(finalist_rows)),
        ("Rejected", _public(rejected_rows)),
        ("Config used", DataFrame({"setting": list(cfg), "value": [str(v) for v in cfg.values()]})),
    ]
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, f"{SCENARIO_LABEL}_sweep.xlsx")
    with open_report_workbook(out_path) as writer:
        for sheet_name, frame in sheets:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    logging.info("Sweep workbook for %s written to %s", CLUSTER_NAME, out_path)
    require_run_log(write_run_log(Path(out_path)))
    if WRITE_BAY_REPORTS:
        write_selected_bay_reports(standards)
    return out_path


# ==================================================================================================
# DETAILED BAY-ONLY REPORTS
# ==================================================================================================


def report_slug(value: str) -> str:
    """Return a short, portable output name; callers check for collisions."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:70] or "report"


def scheduled_identity(trips: list[dict[str, Any]]) -> str:
    """Hash every trip field except the stop IDs/names a bay move may change."""
    unchanged = copy.deepcopy(trips)
    for trip in unchanged:
        for prefix in ("first", "last"):
            trip.pop(f"{prefix}_stop_id", None)
            trip.pop(f"{prefix}_stop_name", None)
        trip["stop_times_sequence"] = [
            [*stop[:2], *stop[4:]] for stop in trip["stop_times_sequence"]
        ]
    payload = json.dumps(sorted(unchanged, key=lambda trip: trip["trip_id"]), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rebuild_bay_timeline(standard: Standard, change: Change) -> Tuple[DataFrame, str]:
    """Rebuild all vehicle minutes and verify that only permitted bay fields changed."""
    if change.shifts:
        raise ValueError("Detailed reports currently accept bay changes only, not time shifts.")
    revised = standard.changed_block_trips(change)
    all_trips = [
        trip
        for block, originals in standard.block_trips.items()
        for trip in revised.get(block, originals)
    ]
    signature = scheduled_identity(all_trips)
    if signature != scheduled_identity(standard.snapshot["trips"]):
        raise ValueError("Bay report changed a scheduled time, trip, block or other fixed field.")
    rows = []
    for block, originals in standard.block_trips.items():
        trips = revised.get(block, originals)
        rows.extend(
            build_schedule_rows(
                sorted(trips, key=lambda trip: (trip["start"], trip["trip_id"])),
                range(standard.snapshot["timeline_end"]),
                block,
                standard.snapshot["clusters"],
                standard.snapshot["settings"],
            )
        )
    frame = DataFrame(rows)
    if frame.duplicated(["Block", "Timestamp"]).any():
        raise ValueError("Detailed timeline has duplicate vehicle/minute rows.")
    return frame, signature


def verify_baseline_timeline(standard: Standard, rebuilt: DataFrame) -> None:
    """Compare every canonical timeline cell, ignoring only input bookkeeping columns."""
    columns = list(rebuilt.columns)
    missing = set(columns) - set(standard.timeline.columns)
    if missing:
        raise ValueError(
            f"Baseline lacks current exporter columns: {sorted(missing)}. Rerun Step 1."
        )

    def comparable(frame: DataFrame) -> DataFrame:
        return (
            frame[columns]
            .fillna("")
            .astype(str)
            .sort_values(["Block", "Timestamp"])
            .reset_index(drop=True)
        )

    original, actual = comparable(standard.timeline), comparable(rebuilt)
    if original.shape != actual.shape:
        raise ValueError(
            f"{standard.name}: rebuilt baseline has {len(actual)} rows; input has "
            f"{len(original)}. Reports were not published. Rerun matching Step 1."
        )
    different = original.ne(actual)
    if different.to_numpy().any():
        row, column = np.argwhere(different.to_numpy())[0]
        label = columns[column]
        raise ValueError(
            f"{standard.name}: baseline does not reproduce Step 1 at block "
            f"{original.iloc[row]['Block']}, {original.iloc[row]['Timestamp']}, {label}: "
            f"input {original.iloc[row, column]!r}, rebuilt {actual.iloc[row, column]!r}. "
            "Reports were not published. Use matching exporter and sweep versions."
        )


def resolve_report_changes(standard: Standard) -> Dict[str, Change]:
    """Resolve explicit report selections strictly, honoring shared-bay groups."""
    changes = {"baseline": Change()}
    selections = Selections()
    for group in SAME_BAY_GROUPS:
        members = [standard.names.parse(text) for text in group]
        if any(member is None or not member.role for member in members):
            raise ValueError(f"Unknown or incomplete SAME_BAY_GROUPS movement: {group}")
        selections.same_bay_groups.append([member for member in members if member is not None])
    for name, assignments in BAY_REPORT_CONFIGURATIONS.items():
        if not isinstance(name, str) or not name.strip() or name.casefold() == "baseline":
            raise ValueError("Bay report names must be nonblank and cannot be 'baseline'.")
        if not isinstance(assignments, dict) or not assignments:
            raise ValueError(f"Report {name!r} must contain at least one bay assignment.")
        moves: Dict[RouteEnd, str] = {}
        for text, bay in assignments.items():
            movement = standard.names.parse(text)
            if movement is None or not movement.role:
                raise ValueError(f"Report {name!r}: unknown or incomplete route-end {text!r}.")
            if movement in moves and moves[movement] != bay:
                raise ValueError(f"Report {name!r}: conflicting aliases for {movement}.")
            moves[movement] = bay
        change = expand_groups(Change(moves=moves), selections)
        for movement, bay in change.moves.items():
            if movement not in standard.shift_model.masks:
                raise ValueError(f"Report {name!r}: movement {movement} has no cluster visits.")
            if bay not in standard.bay_names:
                raise ValueError(f"Report {name!r}: unknown destination bay {bay!r}.")
        changes[name] = change
    slugs = [report_slug(name).casefold() for name in changes]
    if len(set(slugs)) != len(slugs):
        raise ValueError("Bay report configuration names produce duplicate filenames.")
    return changes


def report_assignment_table(standard: Standard, change: Change) -> DataFrame:
    """Inventory every scheduled cluster movement, including unchanged assignments."""
    visits: Dict[RouteEnd, List[str]] = defaultdict(list)
    for trip in standard.snapshot["trips"]:
        for stop in trip["stop_times_sequence"]:
            if stop[2] not in CLUSTER_STOPS:
                continue
            role = "depart" if stop[5] else "arrive" if stop[6] else "through"
            movement = RouteEnd(
                trip["route_id"],
                role,
                str(trip["direction_id"]) if role == "through" else "",
                standard.names.label[trip["route_id"]],
            )
            visits[movement].append(CLUSTER_STOPS[stop[2]])
    rows = []
    for movement, bays in sorted(visits.items()):
        before = sorted(set(bays))
        after = [change.moves[movement]] if movement in change.moves else before
        rows.append(
            {
                "Movement": str(movement),
                "Route ID": movement.route,
                "Current bay(s)": ", ".join(before),
                "Proposed bay(s)": ", ".join(after),
                "Changed": before != after,
                "Scheduled visits": len(bays),
            }
        )
    return DataFrame(
        rows,
        columns=[
            "Movement",
            "Route ID",
            "Current bay(s)",
            "Proposed bay(s)",
            "Changed",
            "Scheduled visits",
        ],
    )


def report_space_definitions(standard: Standard) -> Dict[str, Tuple[str, int]]:
    """Validate physical layover routing and keep its capacity separate from passenger bays."""
    spaces = {
        bay: ("Passenger bay", int(standard.cap[i])) for i, bay in enumerate(standard.bay_names)
    }
    assigned = []
    if set(REPORT_OVERFLOW_ROUTING) != set(REPORT_OVERFLOW_CAPACITY):
        raise ValueError(
            "REPORT_OVERFLOW_ROUTING and REPORT_OVERFLOW_CAPACITY must name the same spaces."
        )
    for name, statuses in REPORT_OVERFLOW_ROUTING.items():
        capacity = REPORT_OVERFLOW_CAPACITY[name]
        if not name or name in spaces:
            raise ValueError(f"Invalid or duplicate layover space name: {name!r}")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"Layover space {name!r} needs a positive integer capacity.")
        assigned.extend(statuses)
        spaces[name] = ("Layover space", capacity)
    if sorted(assigned) != ["LAYOVER", "LONG BREAK"]:
        raise ValueError(
            "Route LAYOVER and LONG BREAK exactly once each in REPORT_OVERFLOW_ROUTING."
        )
    return spaces


def detailed_occupancy(
    standard: Standard, timeline: DataFrame, spaces: Dict[str, Tuple[str, int]]
) -> Tuple[DataFrame, DataFrame]:
    """Count distinct physical blocks independently for every occupied space and minute."""
    overflow = {
        status: name for name, statuses in REPORT_OVERFLOW_ROUTING.items() for status in statuses
    }
    frame = timeline.loc[
        timeline["Stop ID"].isin(CLUSTER_STOPS)
        & timeline["Status"].isin(BAY_STATUSES | set(overflow))
    ].copy()
    frame["Space"] = frame["Stop ID"].map(CLUSTER_STOPS)
    mask = frame["Status"].isin(overflow)
    frame.loc[mask, "Space"] = frame.loc[mask, "Status"].map(overflow)
    frame["Space type"] = frame["Space"].map({s: kind for s, (kind, _) in spaces.items()})
    frame["Minute"] = frame["Timestamp"].map(timestamp_to_minutes).astype(int)
    frame["Route name"] = frame["Route"].map(standard.names.label).fillna("")
    frame["Capacity"] = frame["Space"].map({s: cap for s, (_, cap) in spaces.items()})
    frame["Buses"] = frame.groupby(["Space", "Minute"])["Block"].transform("nunique")
    frame["Conflict"] = frame["Buses"] > frame["Capacity"]
    frame = frame.sort_values(["Minute", "Space", "Block"]).reset_index(drop=True)
    first_columns = [
        "Timestamp",
        "Space",
        "Space type",
        "Buses",
        "Capacity",
        "Conflict",
        "Block",
        "Route name",
        "Direction",
        "Trip ID",
        "Status",
        "Stop Role",
        "Arrival Time",
        "Departure Time",
        "Stop ID",
        "Stop Name",
    ]
    frame = frame[first_columns + [column for column in frame if column not in first_columns]]
    columns = [
        "Space type",
        "Space",
        "Minute",
        "Time",
        "Buses",
        "Capacity",
        "Conflict",
        "Blocks",
        "Routes",
        "Trips",
        "Activities",
        "Participants",
    ]
    minutes = []
    for (space, minute), group in frame.groupby(["Space", "Minute"], sort=True):
        first = group.iloc[0]
        participants = [
            f"Block {row['Block']}: route {row['Route name']}, trip {row['Trip ID']}, "
            f"{row['Status']} ({row['Stop Role']})"
            for _, row in group.iterrows()
        ]
        minutes.append(
            {
                "Space type": first["Space type"],
                "Space": space,
                "Minute": int(minute),
                "Time": first["Timestamp"],
                "Buses": int(first["Buses"]),
                "Capacity": int(first["Capacity"]),
                "Conflict": bool(first["Conflict"]),
                "Blocks": ", ".join(sorted(set(group["Block"]))),
                "Routes": ", ".join(sorted(set(group["Route name"]))),
                "Trips": ", ".join(sorted(set(group["Trip ID"]))),
                "Activities": ", ".join(sorted(set(group["Status"]))),
                "Participants": "; ".join(participants),
            }
        )
    return frame, DataFrame(minutes, columns=columns).astype(
        {
            "Minute": "int64",
            "Buses": "int64",
            "Capacity": "int64",
            "Conflict": bool,
        }
    )


def report_conflict_events(minutes: DataFrame, space_type: str) -> DataFrame:
    """Collapse consecutive conflict minute marks; event endpoints are inclusive."""
    columns = [
        "Space",
        "Start",
        "End (inclusive)",
        "Conflict minutes",
        "Peak buses",
        "Capacity",
        "Blocks during event",
        "Routes during event",
        "Trips during event",
    ]
    conflicts = minutes[minutes["Conflict"] & minutes["Space type"].eq(space_type)]
    rows = []
    for space, group in conflicts.groupby("Space", sort=True):
        group = group.sort_values("Minute")
        runs = group["Minute"].diff().ne(1).cumsum()
        for _, event in group.groupby(runs, sort=False):
            row = {
                "Space": space,
                "Start": event.iloc[0]["Time"],
                "End (inclusive)": event.iloc[-1]["Time"],
                "Conflict minutes": len(event),
                "Peak buses": int(event["Buses"].max()),
                "Capacity": int(event.iloc[0]["Capacity"]),
            }
            for column in ("Blocks", "Routes", "Trips"):
                row[f"{column} during event"] = ", ".join(
                    sorted({item for value in event[column] for item in value.split(", ") if item})
                )
            rows.append(row)
    return DataFrame(rows, columns=columns)


def report_conflict_comparison(before: DataFrame, after: DataFrame) -> DataFrame:
    """Compare space/minute keys; expose changed participants at retained conflicts."""
    columns = [
        "Space type",
        "Space",
        "Minute",
        "Time",
        "Change",
        "Participants changed",
        "Before buses",
        "After buses",
        "Before participants",
        "After participants",
    ]
    old = {(row["Space"], row["Minute"]): row for row in before.to_dict("records")}
    new = {(row["Space"], row["Minute"]): row for row in after.to_dict("records")}
    old_keys = {key for key, row in old.items() if row["Conflict"]}
    new_keys = {key for key, row in new.items() if row["Conflict"]}
    rows = []
    for key in sorted(old_keys | new_keys):
        a, b = old.get(key, {}), new.get(key, {})
        description = b or a
        retained = key in old_keys and key in new_keys
        rows.append(
            {
                "Space type": description["Space type"],
                "Space": key[0],
                "Minute": key[1],
                "Time": minutes_to_hhmm(key[1]),
                "Change": "Retained" if retained else "Removed" if key in old_keys else "Created",
                "Participants changed": retained and a["Participants"] != b["Participants"],
                "Before buses": a.get("Buses", 0),
                "After buses": b.get("Buses", 0),
                "Before participants": a.get("Participants", ""),
                "After participants": b.get("Participants", ""),
            }
        )
    return DataFrame(rows, columns=columns)


def verify_report_score(standard: Standard, change: Change, minutes: DataFrame) -> int:
    """Require independent detailed counts and exact conflict keys to match the sweep scorer."""
    scored = standard.apply(change)
    if scored.duplicated(["Block", "Minute"]).any():
        raise ValueError("Sweep occupancy contains duplicate vehicle/minute rows.")
    total, _, keys = standard.score(scored)
    actual = minutes[minutes["Conflict"] & minutes["Space type"].eq("Passenger bay")]
    detailed_keys = set(zip(actual["Space"], actual["Minute"]))
    sweep_keys = {standard.key_label(int(key)) for key in keys}
    if detailed_keys != sweep_keys or len(actual) != total:
        raise ValueError(f"{standard.name}: detailed conflict minutes do not match sweep scoring.")
    return total


def write_bay_report_workbook(path: Path, sheets: List[Tuple[str, DataFrame]]) -> None:
    """Write readable tables with frozen headers, filters and conflict highlighting."""
    used: set[str] = set()
    with open_report_workbook(str(path)) as writer:
        for requested, frame in sheets:
            stem = re.sub(r"[\[\]:*?/\\]", "_", requested).strip("'")[:31] or "Sheet"
            name, suffix = stem, 1
            while name.casefold() in used:
                suffix += 1
                name = f"{stem[:26]}_{suffix}"
            used.add(name.casefold())
            if len(frame) > 1_048_575:
                raise ValueError(
                    f"{requested} exceeds Excel's row limit. Use a shorter input period."
                )
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.zoomScale = 85
            sheet.row_dimensions[1].height = 32
            for cell in sheet[1]:
                cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="24435B")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for i, column in enumerate(frame.columns, 1):
                lengths = [len(str(column)), *[len(str(v)) for v in frame[column].head(300)]]
                width = min(64, max(14, max(lengths, default=14) + 2))
                sheet.column_dimensions[get_column_letter(i)].width = width
            flag = list(frame.columns).index("Conflict") if "Conflict" in frame else None
            for row in sheet.iter_rows(min_row=2):
                conflict = flag is not None and row[flag].value is True
                height = 15
                for cell in row:
                    # All report strings are literal input or labels, never Excel formulas.
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
                        width = sheet.column_dimensions[cell.column_letter].width
                        height = max(
                            height, min(180, 15 * (1 + len(cell.value) // max(1, int(width))))
                        )
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                    if conflict:
                        cell.fill = PatternFill("solid", fgColor="FCE4D6")
                        cell.font = Font(name="Calibri", size=11, bold=True)
                sheet.row_dimensions[row[0].row].height = height


def write_selected_bay_reports(standards: Dict[str, Standard]) -> str:
    """Build selected bay reports, validate all results, then publish the output files.

    Baseline and each named configuration are rebuilt independently. The
    comparison counts distinct space/minute keys; a retained key may involve
    different buses, which the minute comparison identifies explicitly.
    """
    if not standards:
        raise ValueError("Detailed bay reports require at least one occupancy standard.")
    tags = [report_slug(name).casefold() for name in standards]
    if len(set(tags)) != len(tags):
        raise ValueError("Occupancy standard names produce duplicate report filenames.")
    for configuration, expected in EXPECTED_REPORT_TOTALS.items():
        if configuration not in {"baseline", *BAY_REPORT_CONFIGURATIONS} or set(expected) - set(
            standards
        ):
            raise ValueError(
                f"Unknown configuration/standard in EXPECTED_REPORT_TOTALS: {configuration}"
            )
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in expected.values()):
            raise ValueError("Expected report totals must be nonnegative integer minutes.")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    destination = Path(OUTPUT_FOLDER) / f"{report_slug(SCENARIO_LABEL)}_bay_reports"
    comparisons, comparison_minutes = [], []
    # Preserve existing reports when calculation, validation or workbook generation fails.
    with tempfile.TemporaryDirectory(prefix="bay_reports_", dir=OUTPUT_FOLDER) as temporary:
        staged = Path(temporary)
        for standard_name, standard in standards.items():
            changes = resolve_report_changes(standard)
            spaces = report_space_definitions(standard)
            baseline, _ = rebuild_bay_timeline(standard, Change())
            verify_baseline_timeline(standard, baseline)
            _, before = detailed_occupancy(standard, baseline, spaces)
            logging.info(
                "%s: rebuilt baseline matches all %d input rows.", standard_name, len(baseline)
            )
            for configuration, change in changes.items():
                logging.info("Building bay report: %s / %s", standard_name, configuration)
                timeline, signature = rebuild_bay_timeline(standard, change)
                occupancy, minutes = detailed_occupancy(standard, timeline, spaces)
                total = verify_report_score(standard, change, minutes)
                expected = EXPECTED_REPORT_TOTALS.get(configuration, {}).get(standard_name)
                if expected is not None and total != expected:
                    raise ValueError(
                        f"{configuration} / {standard_name}: expected {expected} "
                        f"conflict bay-minutes, rebuilt {total}. Reports were not published. "
                        "Check feed and assumptions."
                    )
                delta = report_conflict_comparison(before, minutes)
                assignments = report_assignment_table(standard, change)
                summary = []
                for space, (kind, capacity) in spaces.items():
                    old = before[before["Space"].eq(space) & before["Conflict"]]
                    new = minutes[minutes["Space"].eq(space) & minutes["Conflict"]]
                    change_rows = delta[delta["Space"].eq(space)]
                    counts = change_rows["Change"].value_counts()
                    removed, created, retained = (
                        int(counts.get(k, 0)) for k in ("Removed", "Created", "Retained")
                    )
                    if len(old) - removed + created != len(new) or retained + created != len(new):
                        raise ValueError(
                            "Conflict comparison does not reconcile with detailed counts."
                        )
                    row = {
                        "Space type": kind,
                        "Space": space,
                        "Capacity": capacity,
                        "Current conflict minutes": len(old),
                        "Proposed conflict minutes": len(new),
                        "Removed": removed,
                        "Retained": retained,
                        "Created": created,
                        "Net reduction": len(old) - len(new),
                    }
                    summary.append(row)
                    comparisons.append(
                        {"Standard": standard_name, "Configuration": configuration, **row}
                    )
                for kind, label in (
                    ("Passenger bay", "All passenger bays"),
                    ("Layover space", "All layover spaces"),
                ):
                    members = [row for row in summary if row["Space type"] == kind]
                    summary.append(
                        {
                            "Space type": f"{kind} total",
                            "Space": label,
                            "Capacity": "",
                            **{
                                column: sum(row[column] for row in members)
                                for column in (
                                    "Current conflict minutes",
                                    "Proposed conflict minutes",
                                    "Removed",
                                    "Retained",
                                    "Created",
                                    "Net reduction",
                                )
                            },
                        }
                    )
                comparison_minutes.append(
                    delta.assign(Standard=standard_name, Configuration=configuration)
                )
                events = report_conflict_events(minutes, "Passenger bay")
                overflow_events = report_conflict_events(minutes, "Layover space")
                if int(events["Conflict minutes"].sum()) != total:
                    raise ValueError(
                        "Conflict event durations do not sum to the passenger-bay total."
                    )
                overflow_total = int(
                    (minutes["Conflict"] & minutes["Space type"].eq("Layover space")).sum()
                )
                if int(overflow_events["Conflict minutes"].sum()) != overflow_total:
                    raise ValueError("Overflow event durations do not reconcile.")
                checks = DataFrame(
                    [
                        {
                            "Check": "Baseline canonical rows match Step 1",
                            "Result": "PASS",
                            "Value": len(baseline),
                        },
                        {
                            "Check": "Scheduled times, trips and blocks unchanged",
                            "Result": "PASS",
                            "Value": signature,
                        },
                        {
                            "Check": "Detailed passenger-bay keys match sweep scorer",
                            "Result": "PASS",
                            "Value": total,
                        },
                        {
                            "Check": "Event durations and removed/created counts reconcile",
                            "Result": "PASS",
                            "Value": total,
                        },
                        {
                            "Check": "Optional historical total",
                            "Result": "PASS" if expected is not None else "Not requested",
                            "Value": expected if expected is not None else "",
                        },
                    ]
                )
                explanation = {
                    "Configuration": configuration,
                    "Occupancy standard": standard_name,
                    "Passenger-bay conflict minutes": total,
                    "Distinct minutes with any passenger-bay conflict": int(
                        minutes.loc[
                            minutes["Conflict"] & minutes["Space type"].eq("Passenger bay"),
                            "Minute",
                        ].nunique()
                    ),
                    "Assignment movements changed": int(assignments["Changed"].sum()),
                    "Counting rule": (
                        "One space above capacity for one minute is one conflict minute, "
                        "regardless of the number of excess buses. Passenger bays and layover "
                        "spaces are reported separately."
                    ),
                    "How to check": (
                        "Summary totals equal the event durations. Filter Conflict Minutes by "
                        "space/time; AllStops and the individual space sheets show every bus "
                        "and its scheduled times."
                    ),
                    "Time labels": (
                        "Times are minute marks. Event start and end are inclusive: 11:28 "
                        "through 11:30 is three conflict minutes. Times beyond midnight retain "
                        "hours of 24 or more."
                    ),
                    "Retained conflicts": (
                        "Retained means the same space and minute remain above capacity. "
                        "Participants may change; see Changes vs Baseline."
                    ),
                    "Event participants": (
                        "An event lists all buses present across its conflict minutes; they "
                        "need not all be present simultaneously. Conflict Minutes identifies "
                        "the participants each minute."
                    ),
                    "Layover locations": (
                        "Physical layover spaces are assigned by REPORT_OVERFLOW_ROUTING. The "
                        "original Stop ID on a layover row identifies the associated stop, not "
                        "the physical layover position; use Space."
                    ),
                    "Scope": (
                        "Occupancy follows this standard's settings, including in-bay DWELL. "
                        "No schedule shift or displaced-bus routing is simulated. These counts "
                        "are modeled occupancy overlaps, not measured delays."
                    ),
                    "Selection": (
                        "These are explicitly selected configurations, not a new optimization "
                        "result. A changed feed or counting rule may change earlier totals."
                    ),
                }
                assumptions = {
                    "Source folder": TIMELINES.get(standard_name, ""),
                    "Cluster stops": json.dumps(CLUSTER_STOPS, sort_keys=True),
                    "Bay capacities": json.dumps(CLUSTER_CAPACITY, sort_keys=True),
                    "Overflow routing": json.dumps(REPORT_OVERFLOW_ROUTING, sort_keys=True),
                    "Overflow capacities": json.dumps(REPORT_OVERFLOW_CAPACITY, sort_keys=True),
                    "Selected changes": change.label() or "Current assignments",
                    "Schedule identity": signature,
                    "Service IDs": str(standard.snapshot.get("service_ids", [])),
                    **standard.snapshot["settings"],
                }
                sheets = [
                    ("Summary", DataFrame(summary)),
                    (
                        "Read me",
                        DataFrame({"Item": list(explanation), "Value": list(explanation.values())}),
                    ),
                    ("Assignments", assignments),
                    ("Conflict Events", events),
                    ("Overflow Events", overflow_events),
                    ("Conflict Minutes", minutes[minutes["Conflict"]]),
                    ("Changes vs Baseline", delta),
                    ("AllStops", occupancy),
                    *[
                        (
                            f"Bay {space}" if kind == "Passenger bay" else space,
                            occupancy[occupancy["Space"].eq(space)],
                        )
                        for space, (kind, _) in spaces.items()
                    ],
                    ("Checks", checks),
                    (
                        "Assumptions",
                        DataFrame(
                            {
                                "Setting": list(assumptions),
                                "Value": [str(v) for v in assumptions.values()],
                            }
                        ),
                    ),
                ]
                stem = (
                    f"{report_slug(CLUSTER_NAME)}_Conflicts_"
                    f"{report_slug(configuration)}_{report_slug(standard_name)}"
                )
                workbook_path = staged / f"{stem}.xlsx"
                write_bay_report_workbook(workbook_path, sheets)
                require_run_log(write_run_log(workbook_path))
                timeline.to_csv(staged / f"{stem}_timeline.csv", index=False)
                logging.info(
                    "%s / %s: %d passenger-bay conflict minutes.",
                    standard_name,
                    configuration,
                    total,
                )
        comparison_path = staged / f"{report_slug(CLUSTER_NAME)}_Bay_Comparison.xlsx"
        comparison_by_space = DataFrame(comparisons)
        comparison_totals = comparison_by_space.groupby(
            ["Standard", "Configuration", "Space type"], sort=False, as_index=False
        )[
            [
                "Current conflict minutes",
                "Proposed conflict minutes",
                "Removed",
                "Retained",
                "Created",
                "Net reduction",
            ]
        ].sum()
        write_bay_report_workbook(
            comparison_path,
            [
                ("Comparison totals", comparison_totals),
                ("Comparison by space", comparison_by_space),
                ("Conflict minute changes", pd.concat(comparison_minutes, ignore_index=True)),
            ],
        )
        require_run_log(write_run_log(comparison_path))
        destination.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            os.replace(path, destination / path.name)
    logging.info("Detailed bay reports written to %s", destination)
    return str(destination)


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
    """Run discovery, selected bay reports, or the candidate sweep.

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
        elif REPORTS_ONLY:
            standards, _, _, _, _ = load_sweep_standards()
            write_selected_bay_reports(standards)
        else:
            run_sweep()
    except (RunLogError, ValueError, OSError, KeyError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
