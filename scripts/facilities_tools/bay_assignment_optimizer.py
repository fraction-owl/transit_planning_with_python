"""Choose bus-bay assignments that reduce conflicts at a transit center, straight from GTFS.

Runs the whole bay-analysis pipeline in one pass. Every vehicle block that
visits the facility is rendered minute by minute, as
``block_status_timeline_exporter.py`` renders it, under two occupancy
standards ("direct" and "likely" conflict). Passenger-bay, layover-space and
facility-wide conflict minutes are counted as ``bay_usage_analyzer.py`` counts
them. Every movement that could change bay (``<route> arrive``, ``<route>
depart``, ``<route> through <direction>``) is listed. PuLP's CBC solver then
chooses bay reassignments that minimize a weighted count of conflict minutes.
The time-shift search stays in ``bay_change_sweep.py``.

Method
------
Schedules are fixed: only the bay a movement uses may change. A changed
movement uses one permitted bay for every visit, while KEEP preserves its
scheduled stops. A bay-minute is a conflict when it holds more buses than the
bay's capacity. Each conflict minute is weighted by standard (direct, or likely
only) and by peak/off-peak period; a direct conflict is never charged as likely
as well. ``OPTIMIZER_OBJECTIVE`` ranks plans by that score, then by changed
routes, then by changed movements, or minimizes changes under a hard score
ceiling, subject to route budgets, bays per route, consistent boarding bays
per passenger direction, same-bay groups and a direct-conflict policy. The same
renderer rebuilds every returned plan, and its rows must match the solver's
model before the plan is reported. ``NAMED_PROPOSALS`` (e.g. a consultant's
layout) are scored the same way, with rule breaks flagged. Layover-space and
facility-wide totals are reported but not optimized; a bay-only change cannot
alter them.

Inputs
------
- A GTFS folder or ``.zip`` with trips, stop_times, stops and routes, plus
  calendar.txt / calendar_dates.txt when ``SERVICE_DATE`` selects the day.
  Fields are read as written, so IDs such as ``NA`` stay text.

Outputs
-------
Each run writes a new ``<SCENARIO_LABEL>_<timestamp>`` folder in ``OUTPUT_DIR``:
- ``<SCENARIO_LABEL>_bay_plans.xlsx``: Read me, Conflict summary, Plan
  comparison, Assignments, Route bay usage, Boarding bays, Conflict minutes,
  Route-ends, Interlines, Block chains, Bay options and Configuration. It is rewritten after
  each route budget, so a long solve leaves results that can be used.
- ``<SCENARIO_LABEL>_bay_plans_runlog.txt``, the run-log sidecar capturing the
  verbatim CONFIGURATION block and the effective settings.
- ``solver/``: CBC's log and working files (model, warm start, solution) per
  route budget.

Typical usage
-------------
Set the GTFS path, output folder, service day and facility in CONFIGURATION
(or pass the matching CLI flags; ``--config`` takes a JSON file of further
overrides) and run from a shell, ArcGIS Pro's Python window, or a Jupyter
notebook. Run once with ``OPTIMIZE_MODE = "never"`` to review the Route-ends
and Bay options sheets, restrict ``BAY_OPTIONS`` or add locks if needed, then
run with ``"if_conflicts"``.

Requires
--------
pandas, numpy and openpyxl. Optimizing also needs PuLP 3.x
(``pip install PuLP==3.2.2``), whose bundled CBC solver is used.
Results are proposals only: review bay access, bus suitability and walking
paths before adopting a plan.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.metadata
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
# === BEGIN CONFIG ===

# --- Inputs and outputs ---------------------------------------------------------------------------
GTFS_PATH = r"Path\To\GTFS_folder_or_zip"
OUTPUT_DIR = r"Path\To\bay_assignment_output"  # each run writes a new subfolder here
SCENARIO_LABEL = "baseline"

# Service day: give ONE date (YYYYMMDD) or the service_ids that operate together.
# A date applies calendar_dates.txt exceptions automatically. service_id values are
# agency-specific -- REPLACE WITH YOUR VALUES (the placeholder assumes a weekday service).
SERVICE_DATE = ""
SERVICE_IDS: List[str] = ["4"]

# --- Facility -------------------------------------------------------------------------------------
# One GTFS stop_id per physical passenger bay, mapped to the bay label the reports show.
CLUSTER_NAME = "Metro"
CLUSTER_STOPS: Dict[str, str] = {
    "2373": "A",
    "2832": "B",
}
CLUSTER_CAPACITY: Dict[str, int] = {"B": 2}  # buses per bay (1-3); unlisted bays hold one

# Layover spaces at the facility, separate from the passenger bays and never reassigned.
# A between-trip gap longer than IN_BAY_LAYOVER_MAX_MINUTES leaves the bay: LAYOVER up to
# LAYOVER_THRESHOLD, LONG BREAK beyond. Route each status to one space, or leave {} when
# the facility has no layover space. Spaces hold one bus unless OVERFLOW_CAPACITY says more.
OVERFLOW_ROUTING: Dict[str, List[str]] = {
    "layover_bay_A": ["LAYOVER"],
    "layover_bay_B": ["LONG BREAK"],
}
OVERFLOW_CAPACITY: Dict[str, int] = {}

# Other facilities, needed only to tell a layover spent at another facility from a
# deadhead. Each entry: {"name": "Other facility", "stops": ["stop_id", ...]}.
OTHER_CLUSTERS: List[Dict[str, Any]] = []

# --- Occupancy standards --------------------------------------------------------------------------
# Both standards are rendered from the same schedule and never added together. The
# values are Step 1's presets (block_status_timeline_exporter.py).
OCCUPANCY_STANDARDS: Dict[str, Dict[str, int]] = {
    "Direct conflict": {
        "THROUGH_DWELL_MINUTES": 1,
        "PRE_DEPARTURE_MINUTES": 0,
        "POST_ARRIVAL_MINUTES": 0,
        "IN_BAY_LAYOVER_MAX_MINUTES": 10,
        "LAYOVER_THRESHOLD": 20,
    },
    "Likely conflict": {
        "THROUGH_DWELL_MINUTES": 2,
        "PRE_DEPARTURE_MINUTES": 5,
        "POST_ARRIVAL_MINUTES": 2,
        "IN_BAY_LAYOVER_MAX_MINUTES": 10,
        "LAYOVER_THRESHOLD": 20,
    },
}
OPTIMIZER_DIRECT_STANDARD = "Direct conflict"
OPTIMIZER_LIKELY_STANDARD = "Likely conflict"
DEFAULT_HOURS = 26  # rendered service day; extended automatically for later trips
# False: untimed intermediate stops stop the run. True: interpolate them (flagged).
INTERPOLATE_UNTIMED_STOPS = False

# --- When to optimize -----------------------------------------------------------------------------
# "if_conflicts": optimize only when a bay conflict exists under either standard.
# "never": check and list movements only (PuLP is not needed).
# "always": optimize even a conflict-free baseline (e.g. to meet MAX_BAYS_PER_ROUTE).
OPTIMIZE_MODE = "if_conflicts"

# --- Objective ------------------------------------------------------------------------------------
# Starting policy values, not measured probabilities. Direct takes precedence when the
# same bay-minute conflicts under both standards; it is counted only once.
CONFLICT_WEIGHTS: Dict[str, int] = {
    "direct_peak": 10,
    "direct_off_peak": 5,
    "likely_only_peak": 2,
    "likely_only_off_peak": 1,
}
# Local service-day clock; start inclusive, end exclusive. Periods repeat after 24:00,
# and a window may cross midnight, e.g. ["22:00", "02:00"].
PEAK_WINDOWS: List[List[str]] = [["06:00", "10:00"], ["15:00", "19:00"]]
# "min_conflicts": lowest weighted score, then fewest changed routes, then movements.
# "min_changes": fewest changed routes, then movements, then score; needs a score limit.
OPTIMIZER_OBJECTIVE = "min_conflicts"
# Hard ceiling on the weighted score (None: no ceiling). Required by "min_changes".
WEIGHTED_SCORE_LIMIT: Optional[int] = None
# One solve per budget: the most routes a plan may change. "all" allows every route.
# [1, 2, 3, "all"] compares what each extra changed route buys.
ROUTE_CHANGE_LIMITS: List[Any] = ["all"]
# Distinct passenger bays a route may use across all its movements, KEEP visits included.
# 2 forbids a three-bay route; 1 forces one bay per route; None removes the limit.
MAX_BAYS_PER_ROUTE: Optional[int] = 2
# True: each route boards at one bay per passenger direction (GTFS direction_id). Its
# departures from the facility and its through visits in that direction must share a
# bay. Arrivals (last stops) only set down and are not counted; KEEP visits are.
REQUIRE_CONSISTENT_BOARDING_BAYS = True
# "weighted": allow trade-offs; "no_increase": cap total direct minutes at the baseline;
# "no_new": forbid direct conflicts at any bay-minute that is clear today. This restricts
# bay-minutes, not bus pairs: different buses may conflict at an existing location.
DIRECT_CONFLICT_POLICY = "no_new"

# --- Which movements may change bay ---------------------------------------------------------------
# A movement is "<route> arrive", "<route> depart" or "<route> through <direction_id>",
# named as the Route-ends sheet lists it (or by route_id). A bare "<route>" applies to
# all of the route's movements.
# True: every movement may use every bay, subject to the restrictions below.
# False: only movements listed in BAY_OPTIONS may change.
AUTO_BAY_CANDIDATES = True
BAY_OPTIONS: Dict[str, List[str]] = {
    # "101 arrive": ["A", "B"],
}
LOCKED_ROUTES: List[str] = []  # e.g. ["101"]: every movement keeps its scheduled stops
LOCKED_ROUTE_ENDS: List[str] = []  # e.g. ["101 depart"]
SAME_BAY_GROUPS: List[List[str]] = [
    # ["101 arrive", "101 depart"],  # must finish at one common bay
]
# Optional exact movement -> bay proposal from an earlier run, rebuilt and validated on
# this run, then used as the solver's starting point. Not a set of locks; {} for none.
OPTIMIZER_STARTING_ASSIGNMENTS: Dict[str, str] = {}
# Fixed proposals to compare with the baseline and the solver's plans: name -> {movement:
# bay}; unlisted movements keep their stops. Each is rebuilt and scored like any plan.
# Rule breaks, including BAY_OPTIONS and locks, are flagged rather than rejected.
NAMED_PROPOSALS: Dict[str, Dict[str, str]] = {
    # "Consultant": {"101 arrive": "B", "102 depart": "A"},
}

# --- Solver ---------------------------------------------------------------------------------------
SOLVER_TIME_LIMIT_SECONDS = 1800  # per route budget; CBC is stopped 30 seconds later
SOLVER_RELATIVE_GAP = 0.0
SOLVER_THREADS = 1  # CBC runs serially on Windows regardless
MAX_OPTIMIZER_BINARY_VARIABLES = 250000
MAX_OPTIMIZER_CONSTRAINTS = 1000000
WRITE_SOLVER_MODEL = False  # also write each model as an .lp file in solver/

# Every output must be traceable: a failed run-log write aborts the script.
# Set to False only when writing to a genuinely read-only location.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO

# === END CONFIG ===

CONFIG_KEYS: Tuple[str, ...] = tuple(name for name in globals() if name.isupper())

BAY_STATUSES = frozenset({"ARRIVE", "DEPART", "ARRIVE/DEPART", "LOADING", "DWELL"})
OVERFLOW_STATUSES = frozenset({"LAYOVER", "LONG BREAK"})
OCCUPANCY_KEYS: Tuple[str, ...] = (
    "THROUGH_DWELL_MINUTES",
    "PRE_DEPARTURE_MINUTES",
    "POST_ARRIVAL_MINUTES",
    "IN_BAY_LAYOVER_MAX_MINUTES",
    "LAYOVER_THRESHOLD",
)
OPTIMIZE_MODES: Tuple[str, ...] = ("never", "if_conflicts", "always")
OBJECTIVE_MODES: Tuple[str, ...] = ("min_conflicts", "min_changes")
DIRECT_POLICIES: Tuple[str, ...] = ("weighted", "no_increase", "no_new")
CATEGORIES: Tuple[str, ...] = (
    "direct_peak",
    "direct_off_peak",
    "likely_only_peak",
    "likely_only_off_peak",
)

REQUIRED_GTFS_FILES: Tuple[str, ...] = ("trips.txt", "stop_times.txt", "stops.txt", "routes.txt")
OPTIONAL_GTFS_FILES: Tuple[str, ...] = ("calendar.txt", "calendar_dates.txt", "frequencies.txt")
MAX_TRIPS_PER_BLOCK = 150

BIG = 100000  # minute key spacing per bay (bay index * BIG + minute)
TOLERANCE = 1e-5
CBC_TIMEOUT_GRACE_SECONDS = 30
CBC_PROGRESS_INTERVAL_SECONDS = 15

OCCUPANCY_COLUMNS = ["Block", "Minute", "Bay", "Status", "route_end"]
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
MINUTE_COLUMNS = [
    "plan",
    "bay",
    "minute",
    "time",
    "category",
    "weight",
    "direct_buses",
    "direct_blocks",
    "likely_buses",
    "likely_blocks",
]

_PLACEHOLDER_MARKERS: Tuple[str, ...] = ("path\\to\\", "path/to/")


class OptimizerUnavailable(RuntimeError):
    """PuLP or its bundled CBC executable is missing or could not run."""


class RunLogError(RuntimeError):
    """Raised when the required ``_runlog.txt`` sidecar could not be written."""


# ==================================================================================================
# CONFIGURATION CHECKS
# ==================================================================================================


def default_config() -> Dict[str, Any]:
    """Copy the CONFIGURATION constants, including edits made in a notebook session."""
    return {key: copy.deepcopy(globals()[key]) for key in CONFIG_KEYS}


def is_placeholder_path(path: object) -> bool:
    """Return True if *path* still points at a default placeholder location."""
    text = str(path).lower()
    return any(marker in text for marker in _PLACEHOLDER_MARKERS)


def slug(value: str) -> str:
    """Return a Windows-safe file or folder label that cannot traverse paths."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    if not text:
        raise ValueError(f"A label has no usable filename characters: {value!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10))}
    reserved |= {f"LPT{i}" for i in range(1, 10)}
    return "_" + text if text.upper() in reserved else text


def _positive_int(value: object) -> bool:
    """Whether *value* is an int (not a bool) of at least one."""
    return type(value) is int and value >= 1


def clock_minute(text: str) -> int:
    """Parse a peak-window boundary, allowing 24:00 only as the end of the day."""
    if not isinstance(text, str) or not re.fullmatch(r"\d{1,2}:\d{2}", text):
        raise ValueError(f"Invalid peak-window clock time: {text!r}; use HH:MM.")
    hour, minute = map(int, text.split(":"))
    if not (0 <= hour <= 24 and 0 <= minute < 60) or (hour == 24 and minute):
        raise ValueError(f"Invalid peak-window clock time: {text!r}")
    return hour * 60 + minute


def peak_mask(windows: object) -> Tuple[bool, ...]:
    """Return a repeating 24-hour mask of peak minutes from half-open clock windows."""
    mask = [False] * 1440
    if not isinstance(windows, list):
        raise ValueError("PEAK_WINDOWS must be a list of [start, end] clock pairs.")
    for pair in windows:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each PEAK_WINDOWS entry must contain a start and an end.")
        start, end = map(clock_minute, pair)
        if start == 1440 or start == end:
            raise ValueError("Peak windows need distinct boundaries and a start before 24:00.")
        minutes = range(start, end) if start < end else (*range(start, 1440), *range(end))
        for minute in minutes:
            mask[minute] = True
    return tuple(mask)


def validate_configuration(cfg: Dict[str, Any]) -> None:
    """Reject invalid settings before any GTFS is read or any output is written.

    Raises:
        ValueError: Naming the first setting that is invalid.
    """
    unknown = set(cfg) - set(CONFIG_KEYS)
    missing = set(CONFIG_KEYS) - set(cfg)
    if unknown or missing:
        raise ValueError(f"Unknown settings: {sorted(unknown)}; missing: {sorted(missing)}")
    for name in ("GTFS_PATH", "OUTPUT_DIR", "SCENARIO_LABEL", "CLUSTER_NAME"):
        if not isinstance(cfg[name], str) or not cfg[name].strip():
            raise ValueError(f"{name} must be a nonempty string.")
    slug(cfg["SCENARIO_LABEL"])
    if not isinstance(cfg["SERVICE_IDS"], list) or any(
        not isinstance(value, str) or not value.strip() for value in cfg["SERVICE_IDS"]
    ):
        raise ValueError("SERVICE_IDS must be a list of service_id strings.")
    if not isinstance(cfg["SERVICE_DATE"], str):
        raise ValueError("SERVICE_DATE must be a YYYYMMDD string, or '' to use SERVICE_IDS.")
    if bool(cfg["SERVICE_DATE"].strip()) == bool(cfg["SERVICE_IDS"]):
        raise ValueError("Set exactly one of SERVICE_DATE or SERVICE_IDS.")
    if cfg["SERVICE_DATE"].strip():
        try:
            datetime.strptime(cfg["SERVICE_DATE"].strip(), "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"SERVICE_DATE must be YYYYMMDD; got {cfg['SERVICE_DATE']!r}") from exc

    stops = cfg["CLUSTER_STOPS"]
    if not isinstance(stops, dict) or not stops:
        raise ValueError("CLUSTER_STOPS must map at least one GTFS stop_id to a bay label.")
    if any(not isinstance(s, str) or not s.strip() or s != s.strip() for s in stops):
        raise ValueError("Cluster stop IDs must be nonempty strings with no outer whitespace.")
    labels = list(stops.values())
    if any(not isinstance(bay, str) or not bay.strip() or "," in bay for bay in labels):
        raise ValueError("Bay labels must be nonempty strings without commas.")
    if len(labels) != len(set(labels)):
        raise ValueError("Use one stop_id per physical bay; repeated bay labels are ambiguous.")
    for bay, capacity in cfg["CLUSTER_CAPACITY"].items():
        if bay not in labels or type(capacity) is not int or not 1 <= capacity <= 3:
            raise ValueError(f"Invalid CLUSTER_CAPACITY {bay!r}: {capacity!r}; use integers 1-3.")

    routing = cfg["OVERFLOW_ROUTING"]
    if not isinstance(routing, dict):
        raise ValueError("OVERFLOW_ROUTING must map layover-space names to statuses.")
    for space, statuses in routing.items():
        if not isinstance(space, str) or not space.strip() or space in stops or space in labels:
            raise ValueError(f"Invalid layover space name {space!r}: use a name, not a bay.")
        if not isinstance(statuses, list):
            raise ValueError(f"OVERFLOW_ROUTING[{space!r}] must be a list of statuses.")
    routed = [status for statuses in routing.values() for status in statuses]
    if len(routed) != len(set(routed)) or set(routed) - OVERFLOW_STATUSES:
        raise ValueError(
            "OVERFLOW_ROUTING repeats a status or uses one other than LAYOVER/LONG BREAK."
        )
    if routing and set(routed) != OVERFLOW_STATUSES:
        raise ValueError("When routing layovers, assign both LAYOVER and LONG BREAK explicitly.")
    for space, capacity in cfg["OVERFLOW_CAPACITY"].items():
        if space not in routing or not _positive_int(capacity):
            raise ValueError(f"Invalid OVERFLOW_CAPACITY {space!r}: {capacity!r}.")

    if not isinstance(cfg["OTHER_CLUSTERS"], list):
        raise ValueError("OTHER_CLUSTERS must be a list of {'name': ..., 'stops': [...]} entries.")
    seen = set(stops)
    for other in cfg["OTHER_CLUSTERS"]:
        if (
            not isinstance(other, dict)
            or not str(other.get("name", "")).strip()
            or other.get("name") == cfg["CLUSTER_NAME"]
            or not isinstance(other.get("stops"), list)
        ):
            raise ValueError(f"Invalid OTHER_CLUSTERS entry: {other!r}")
        members = {str(stop) for stop in other["stops"]}
        if members & seen:
            raise ValueError("A stop may belong to only one facility (OTHER_CLUSTERS overlap).")
        seen |= members

    standards = cfg["OCCUPANCY_STANDARDS"]
    expected = {cfg["OPTIMIZER_DIRECT_STANDARD"], cfg["OPTIMIZER_LIKELY_STANDARD"]}
    if len(expected) != 2:
        raise ValueError("The direct and likely standards must have distinct names.")
    if not isinstance(standards, dict) or set(standards) != expected:
        raise ValueError(
            "OCCUPANCY_STANDARDS must define exactly the two standards named by "
            f"OPTIMIZER_DIRECT_STANDARD and OPTIMIZER_LIKELY_STANDARD: {sorted(expected)}."
        )
    for name, settings in standards.items():
        if not isinstance(settings, dict) or set(settings) != set(OCCUPANCY_KEYS):
            raise ValueError(f"{name} needs exactly these settings: {list(OCCUPANCY_KEYS)}")
        if any(type(value) is not int or value < 0 for value in settings.values()):
            raise ValueError(f"{name} occupancy values must be nonnegative integer minutes.")
        if settings["THROUGH_DWELL_MINUTES"] < 1 or (
            settings["LAYOVER_THRESHOLD"] < settings["IN_BAY_LAYOVER_MAX_MINUTES"]
        ):
            raise ValueError(f"Invalid dwell or layover thresholds for {name}.")
    if not _positive_int(cfg["DEFAULT_HOURS"]):
        raise ValueError("DEFAULT_HOURS must be a positive integer.")
    for name in (
        "INTERPOLATE_UNTIMED_STOPS",
        "AUTO_BAY_CANDIDATES",
        "WRITE_SOLVER_MODEL",
        "REQUIRE_CONSISTENT_BOARDING_BAYS",
    ):
        if type(cfg[name]) is not bool:
            raise ValueError(f"{name} must be True or False.")
    if type(cfg["REQUIRE_RUN_LOG"]) is not bool:
        raise ValueError("REQUIRE_RUN_LOG must be True or False.")
    if cfg["OPTIMIZE_MODE"] not in OPTIMIZE_MODES:
        raise ValueError(f"OPTIMIZE_MODE must be one of {list(OPTIMIZE_MODES)}.")

    if not isinstance(cfg["BAY_OPTIONS"], dict):
        raise ValueError("BAY_OPTIONS must map movements to lists of bay labels.")
    for selector, bays in cfg["BAY_OPTIONS"].items():
        if not isinstance(selector, str) or not isinstance(bays, list) or set(bays) - set(labels):
            raise ValueError(f"Invalid BAY_OPTIONS entry: {selector!r}: {bays!r}")
    for name in ("LOCKED_ROUTES", "LOCKED_ROUTE_ENDS"):
        if not isinstance(cfg[name], list) or any(not isinstance(v, str) for v in cfg[name]):
            raise ValueError(f"{name} must be a list of route or movement names.")
    groups = cfg["SAME_BAY_GROUPS"]
    if not isinstance(groups, list) or any(
        not isinstance(group, list) or len(group) < 2 or any(not isinstance(v, str) for v in group)
        for group in groups
    ):
        raise ValueError("SAME_BAY_GROUPS must be a list of lists of two or more movements.")

    weights = cfg["CONFLICT_WEIGHTS"]
    if not isinstance(weights, dict) or set(weights) != set(CATEGORIES):
        raise ValueError(f"CONFLICT_WEIGHTS needs exactly {list(CATEGORIES)}.")
    if any(type(v) is not int or v < 0 for v in weights.values()) or not any(weights.values()):
        raise ValueError("Conflict weights must be nonnegative integers, at least one positive.")
    peak_mask(cfg["PEAK_WINDOWS"])
    limits = cfg["ROUTE_CHANGE_LIMITS"]
    if (
        not isinstance(limits, list)
        or not limits
        or any(v != "all" and (type(v) is not int or v < 0) for v in limits)
    ):
        raise ValueError("ROUTE_CHANGE_LIMITS needs nonnegative integers or 'all'.")
    bay_limit = cfg["MAX_BAYS_PER_ROUTE"]
    if bay_limit is not None and not _positive_int(bay_limit):
        raise ValueError("MAX_BAYS_PER_ROUTE must be a positive integer or None.")
    if cfg["DIRECT_CONFLICT_POLICY"] not in DIRECT_POLICIES:
        raise ValueError(f"DIRECT_CONFLICT_POLICY must be one of {list(DIRECT_POLICIES)}.")
    if cfg["OPTIMIZER_OBJECTIVE"] not in OBJECTIVE_MODES:
        raise ValueError(f"OPTIMIZER_OBJECTIVE must be one of {list(OBJECTIVE_MODES)}.")
    score_limit = cfg["WEIGHTED_SCORE_LIMIT"]
    if score_limit is not None and (type(score_limit) is not int or score_limit < 0):
        raise ValueError("WEIGHTED_SCORE_LIMIT must be a nonnegative integer or None.")
    if cfg["OPTIMIZER_OBJECTIVE"] == "min_changes" and score_limit is None:
        raise ValueError("OPTIMIZER_OBJECTIVE = 'min_changes' requires WEIGHTED_SCORE_LIMIT.")
    proposals = cfg["NAMED_PROPOSALS"]
    if not isinstance(proposals, dict):
        raise ValueError("NAMED_PROPOSALS must map proposal names to {movement: bay} mappings.")
    for name, assignments in proposals.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or name.strip().casefold() in {"baseline", "starting plan"}
            or re.fullmatch(r"Up to \d+ route\(s\)", name.strip())
        ):
            raise ValueError(
                f"Invalid proposal name {name!r}: use a nonblank name other than 'Baseline', "
                "'Starting plan' or a route-budget label."
            )
        if (
            not isinstance(assignments, dict)
            or not assignments
            or any(
                not isinstance(end, str) or not end.strip() or bay not in labels
                for end, bay in assignments.items()
            )
        ):
            raise ValueError(f"Proposal {name!r} must map one or more movements to bay labels.")
    starting = cfg["OPTIMIZER_STARTING_ASSIGNMENTS"]
    if not isinstance(starting, dict) or any(
        not isinstance(end, str) or not end.strip() or bay not in labels
        for end, bay in starting.items()
    ):
        raise ValueError("OPTIMIZER_STARTING_ASSIGNMENTS must map movements to bay labels.")
    for name in (
        "SOLVER_TIME_LIMIT_SECONDS",
        "SOLVER_THREADS",
        "MAX_OPTIMIZER_BINARY_VARIABLES",
        "MAX_OPTIMIZER_CONSTRAINTS",
    ):
        if not _positive_int(cfg[name]):
            raise ValueError(f"{name} must be a positive integer.")
    gap = cfg["SOLVER_RELATIVE_GAP"]
    if type(gap) not in (int, float) or not math.isfinite(gap) or not 0 <= gap < 1:
        raise ValueError("SOLVER_RELATIVE_GAP must be at least 0 and below 1.")


# ==================================================================================================
# GTFS SCHEDULE
# ==================================================================================================


# Canonical version lives in utils/time_helpers.py -- keep this copy in sync.
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


# Canonical version lives in utils/time_helpers.py -- keep this copy in sync.
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


def read_gtfs_tables(gtfs_path: str, files: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Read GTFS text files from a folder or ZIP archive, keeping every field as written.

    Adapted from ``load_gtfs_data`` in utils/gtfs_helpers.py, with one deliberate
    difference: pandas' missing-value markers are switched off, so identifiers such
    as route_id ``"NA"`` or ``"None"`` stay text, and a blank field reads as ``""``
    (the schedule checks treat blank and missing fields alike). ZIP members may sit
    at the archive root or one folder deep.

    Raises:
        OSError: If the path or one of *files* is missing.
        ValueError: If the path is not a folder or ZIP archive, a file appears more
            than once inside the archive, a file is empty, or it cannot be parsed.
    """
    if not os.path.exists(gtfs_path):
        raise OSError(f"The path '{gtfs_path}' does not exist.")
    is_zip = os.path.isfile(gtfs_path) and zipfile.is_zipfile(gtfs_path)
    if not is_zip and not os.path.isdir(gtfs_path):
        raise ValueError(f"'{gtfs_path}' is neither a directory nor a .zip file.")
    archive: Optional[zipfile.ZipFile] = zipfile.ZipFile(gtfs_path) if is_zip else None
    try:
        members: dict[str, list[str]] = {}
        if archive is not None:
            for name in archive.namelist():
                members.setdefault(os.path.basename(name), []).append(name)
        missing = [
            name
            for name in files
            if (archive is None and not os.path.exists(os.path.join(gtfs_path, name)))
            or (archive is not None and not members.get(name))
        ]
        ambiguous = [name for name in files if len(members.get(name, [])) > 1]
        if ambiguous:
            raise ValueError(
                f"Ambiguous GTFS files in '{gtfs_path}' (found in multiple "
                f"locations): {', '.join(ambiguous)}"
            )
        if missing:
            raise OSError(f"Missing GTFS files in '{gtfs_path}': {', '.join(missing)}")
        data: dict[str, pd.DataFrame] = {}
        for name in files:
            options: dict[str, Any] = {"dtype": str, "keep_default_na": False, "low_memory": False}
            try:
                if archive is None:
                    frame = pd.read_csv(os.path.join(gtfs_path, name), **options)
                else:
                    with archive.open(members[name][0]) as handle:
                        frame = pd.read_csv(handle, **options)
            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"File '{name}' in '{gtfs_path}' is empty.") from exc
            except pd.errors.ParserError as exc:
                raise ValueError(f"Parser error in '{name}' in '{gtfs_path}': {exc}") from exc
            data[name.replace(".txt", "")] = frame
            logging.info("Loaded %s (%d records).", name, len(frame))
        return data
    finally:
        if archive is not None:
            archive.close()


# Canonical version lives in utils/calendar_helpers.py -- keep this copy in sync.
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


# Canonical version lives in utils/calendar_helpers.py -- keep this copy in sync.
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


def resolve_service_ids(
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame],
    service_date: str,
) -> list[str]:
    """Return every service_id active on ``service_date`` (``YYYYMMDD``).

    Raises:
        ValueError: If *service_date* is not a valid ``YYYYMMDD`` date.
    """
    try:
        target = dt.datetime.strptime(service_date.strip(), "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"SERVICE_DATE must be a YYYYMMDD date; got {service_date!r}") from exc
    active = expand_service_active_dates(calendar_df, calendar_dates_df, today=target)
    return sorted(service_ids_active_on(active, target))


def available_gtfs_files(gtfs_path: str, names: Sequence[str]) -> tuple[str, ...]:
    """Return the *names* present, and not empty, in a GTFS folder or ZIP archive."""
    sizes: dict[str, int] = {}
    if os.path.isdir(gtfs_path):
        for name in names:
            path = os.path.join(gtfs_path, name)
            if os.path.isfile(path):
                sizes[name] = os.path.getsize(path)
    elif zipfile.is_zipfile(gtfs_path):
        with zipfile.ZipFile(gtfs_path) as archive:
            for info in archive.infolist():
                name = os.path.basename(info.filename)
                if name in names:
                    sizes[name] = max(sizes.get(name, 0), info.file_size)
    for name in names:
        if sizes.get(name) == 0:
            logging.warning("Skipping empty optional file %s.", name)
    return tuple(name for name in names if sizes.get(name, 0) > 0)


def _listed(values: Sequence[str], limit: int = 5) -> str:
    """Comma-separated *values*, cut off after *limit* for error messages."""
    return ", ".join(values[:limit]) + (", ..." if len(values) > limit else "")


def _text(value: object) -> str:
    """GTFS field as text, with a missing value as ``""``."""
    return "" if value is None or pd.isna(value) else str(value)


def blank_text(values: pd.Series) -> pd.Series:
    """True where a GTFS text field is missing or only whitespace."""
    return values.isna() | values.astype(str).str.strip().eq("")


def label_unnamed_stops(stops_df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Return a copy of stops.txt whose blank stop_names are replaced by the stop_id."""
    stops_df = stops_df.copy()
    if "stop_name" not in stops_df.columns:
        stops_df["stop_name"] = ""
    unnamed = blank_text(stops_df["stop_name"])
    stops_df.loc[unnamed, "stop_name"] = stops_df.loc[unnamed, "stop_id"]
    return stops_df, set(stops_df.loc[unnamed, "stop_id"].astype(str))


def mark_first_and_last_stops(df_in: pd.DataFrame) -> pd.DataFrame:
    """Mark each stop in the trip as the first or last using boolean columns."""
    df_out = df_in.sort_values(["trip_id", "stop_sequence"]).copy()
    seq_min = df_out.groupby("trip_id")["stop_sequence"].transform("min")
    seq_max = df_out.groupby("trip_id")["stop_sequence"].transform("max")
    df_out["is_first_stop"] = df_out["stop_sequence"] == seq_min
    df_out["is_last_stop"] = df_out["stop_sequence"] == seq_max
    return df_out


def resolve_stop_times(merged_df: pd.DataFrame, interpolate: bool) -> pd.DataFrame:
    """Parse each visit's scheduled times, estimating missing ones when *interpolate* is set.

    Matches Step 1: adds integer ``arrival_min``/``departure_min`` and
    ``estimated_time``. An untimed intermediate stop is placed between the
    timed stops around it, by shape_dist_traveled when every stop of that
    stretch has an increasing distance, otherwise evenly by stop order.

    Raises:
        ValueError: If a time is malformed, a time is missing while
            interpolation is off, or a trip's first or last stop is untimed.
    """
    df = merged_df.sort_values(["trip_id", "stop_sequence"]).copy()
    times: dict[str, pd.Series] = {}
    given: dict[str, pd.Series] = {}
    for column in ("arrival_time", "departure_time"):
        raw = df[column] if column in df.columns else pd.Series(None, index=df.index, dtype=object)
        text = raw.where(raw.notna(), "").astype(str).str.strip()
        minutes = text.map(parse_time_to_minutes)
        invalid = minutes.isna() & text.ne("")
        if invalid.any():
            row = df.loc[invalid].iloc[0]
            raise ValueError(
                f"{int(invalid.sum())} selected stop_times row(s) have an invalid {column}, "
                f"e.g. {text[invalid].iloc[0]!r} in trip {row['trip_id']} at stop_sequence "
                f"{row['stop_sequence']}."
            )
        times[column] = pd.to_numeric(minutes, errors="coerce")
        given[column] = text.ne("")
    arrival = times["arrival_time"].fillna(times["departure_time"])
    departure = times["departure_time"].fillna(times["arrival_time"])
    estimated = ~(given["arrival_time"] & given["departure_time"])
    if estimated.any() and not interpolate:
        trips = sorted(df.loc[estimated, "trip_id"].astype(str).unique())
        raise ValueError(
            f"{int(estimated.sum())} selected stop visit(s) on {len(trips)} trip(s) lack an "
            f"arrival or departure time ({_listed(trips)}). Set INTERPOLATE_UNTIMED_STOPS = "
            "True to estimate them from each trip's timed stops, or add the times upstream."
        )
    untimed = arrival.isna()
    first = ~df["trip_id"].duplicated(keep="first")
    last = ~df["trip_id"].duplicated(keep="last")
    if (untimed & (first | last)).any():
        trips = sorted(df.loc[untimed & (first | last), "trip_id"].astype(str).unique())
        raise ValueError(
            f"{len(trips)} trip(s) have an untimed first or last stop ({_listed(trips)}); "
            "there is no timed stop to estimate a trip's start or end from. Add those times."
        )
    if untimed.any():
        # Positions of the timed stops before and after each untimed one, within its trip.
        timed_at = pd.Series(np.arange(len(df)), index=df.index, dtype=float).where(~untimed)
        rows = np.flatnonzero(untimed.to_numpy())
        before = timed_at.groupby(df["trip_id"], sort=False).ffill().to_numpy()[rows].astype(int)
        after = timed_at.groupby(df["trip_id"], sort=False).bfill().to_numpy()[rows].astype(int)
        fraction = (rows - before) / (after - before)
        if "shape_dist_traveled" in df.columns:
            dist = pd.to_numeric(df["shape_dist_traveled"], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                along = (dist[rows] - dist[before]) / (dist[after] - dist[before])
            usable = pd.Series((dist[after] > dist[before]) & (along >= 0) & (along <= 1))
            stretch = pd.Series(before)
            rising = pd.Series(along).groupby(stretch).diff().fillna(0).ge(0)
            by_distance = (usable & rising).groupby(stretch).transform("all").to_numpy()
            fraction = np.where(by_distance, along, fraction)
        leave = departure.to_numpy(dtype=float)[before]
        reach = arrival.to_numpy(dtype=float)[after]
        estimate = np.round(leave + fraction * (reach - leave))
        arrival.iloc[rows] = estimate
        departure.iloc[rows] = estimate
    df["arrival_min"] = arrival.astype(int)
    df["departure_min"] = departure.astype(int)
    df["estimated_time"] = estimated
    if "timepoint" in df.columns:
        df.loc[estimated & ~(first | last), "timepoint"] = 0
    return df


def select_facility_blocks(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    service_ids: Sequence[str],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Return every stop visit on the selected service's blocks that visit the facility.

    A whole block is kept, not just its visits to the facility: the vehicle's
    other trips decide whether a gap is spent in a bay, a layover space or
    elsewhere. The checks match Step 1's.

    Raises:
        ValueError: If no selected trip serves the facility, a visiting trip has
            no block_id, or the selected stop_times are malformed.
    """
    required = {"trip_id", "route_id", "service_id"}
    if required - set(trips_df.columns):
        raise ValueError(
            f"trips.txt is missing required fields: {sorted(required - set(trips_df.columns))}"
        )
    trips_df = trips_df.copy()
    stop_times_df = stop_times_df.copy()
    for column in ("direction_id", "block_id", "trip_headsign", "route_short_name"):
        trips_df[column] = trips_df[column].fillna("") if column in trips_df.columns else ""
    if "route_long_name" in trips_df.columns:
        trips_df["route_long_name"] = trips_df["route_long_name"].fillna("")
    else:
        trips_df["route_long_name"] = ""
    trips_df = trips_df[trips_df["service_id"].isin(list(service_ids))]
    stop_times_df["stop_sequence"] = pd.to_numeric(stop_times_df["stop_sequence"], errors="coerce")
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]
    stops_df, unnamed = label_unnamed_stops(stops_df)
    if trips_df["trip_id"].duplicated().any() or stops_df["stop_id"].duplicated().any():
        raise ValueError("Trip IDs and stop IDs must be unique in their GTFS tables.")
    absent = sorted(set(cfg["CLUSTER_STOPS"]) - set(stops_df["stop_id"].astype(str)))
    if absent:
        raise ValueError(f"CLUSTER_STOPS are absent from stops.txt: {_listed(absent)}")
    merged = stop_times_df.merge(trips_df, on="trip_id", how="left")
    merged = merged.merge(stops_df[["stop_id", "stop_name"]], on="stop_id", how="left")
    merged = mark_first_and_last_stops(merged)
    if "timepoint" not in merged.columns:
        merged["timepoint"] = 0
    else:
        merged["timepoint"] = pd.to_numeric(merged["timepoint"], errors="coerce").fillna(0)
        merged["timepoint"] = merged["timepoint"].astype(int)
    merged.loc[merged["is_first_stop"] & merged["timepoint"].eq(0), "timepoint"] = 2
    merged.loc[merged["is_last_stop"] & merged["timepoint"].eq(0), "timepoint"] = 2

    at_facility = merged["stop_id"].astype(str).isin(cfg["CLUSTER_STOPS"])
    if not at_facility.any():
        raise ValueError(
            f"No trip in the selected service stops at {cfg['CLUSTER_NAME']}'s CLUSTER_STOPS. "
            "Check the stop IDs and SERVICE_DATE / SERVICE_IDS."
        )
    blockless = blank_text(merged["block_id"]) & at_facility
    if blockless.any():
        trips = sorted(merged.loc[blockless, "trip_id"].astype(str).unique())
        raise ValueError(
            f"{len(trips)} trip(s) serving the facility have no block_id ({_listed(trips)}). "
            "Bay analysis needs the trips each vehicle runs: fill in block_id upstream."
        )
    merged = merged[merged["block_id"].isin(set(merged.loc[at_facility, "block_id"]))].copy()
    logging.info(
        "%d blocks (%d trips) visit %s.",
        merged["block_id"].nunique(),
        merged["trip_id"].nunique(),
        cfg["CLUSTER_NAME"],
    )
    for column in ("stop_id", "route_id"):
        if blank_text(merged[column]).any():
            raise ValueError(
                f"Analyzed trips require a populated {column}; check GTFS joins and source fields."
            )
    absent = sorted(set(merged["stop_id"].astype(str)) - set(stops_df["stop_id"].astype(str)))
    if absent:
        raise ValueError(
            f"stop_times.txt references stop_ids absent from stops.txt: {_listed(absent)}"
        )
    unnamed_used = sorted(unnamed & set(merged["stop_id"].astype(str)))
    if unnamed_used:
        logging.warning(
            "%d stop(s) used by the selected trips have no stop_name; showing the stop_id: %s",
            len(unnamed_used),
            _listed(unnamed_used),
        )
    if merged["stop_sequence"].isna().any():
        raise ValueError("Selected stop_times contain missing or invalid stop_sequence values.")
    if (merged["stop_sequence"] % 1 != 0).any():
        raise ValueError("stop_sequence must be an integer.")
    merged["stop_sequence"] = merged["stop_sequence"].astype(int)
    if merged.duplicated(["trip_id", "stop_sequence"]).any():
        raise ValueError("Duplicate stop_sequence within a trip.")
    merged = resolve_stop_times(merged, cfg["INTERPOLATE_UNTIMED_STOPS"])
    for trip_id, group in merged.groupby("trip_id", sort=False):
        group = group.sort_values("stop_sequence")
        if (group["arrival_min"] > group["departure_min"]).any():
            raise ValueError(f"Arrival follows departure in trip {trip_id}.")
        if (
            group["arrival_min"].iloc[1:].to_numpy() < group["departure_min"].iloc[:-1].to_numpy()
        ).any():
            raise ValueError(f"Stop times run backwards in trip {trip_id}.")
    return merged


def reject_selected_frequency_trips(
    frequencies: Optional[pd.DataFrame], merged_df: pd.DataFrame
) -> None:
    """Stop on frequency-based trips among the analyzed blocks; ignore the rest.

    Raises:
        ValueError: If any analyzed trip appears in *frequencies*.
    """
    if frequencies is None or frequencies.empty or "trip_id" not in frequencies:
        return
    selected = sorted(
        set(frequencies["trip_id"].dropna().astype(str)) & set(merged_df["trip_id"].astype(str))
    )
    if selected:
        raise ValueError(
            f"{len(selected)} analyzed trip(s) are frequency-based ({_listed(selected)}). "
            "Expand them into scheduled trips first."
        )


def check_for_overlapping_trips(block_subset: pd.DataFrame, block_id: str) -> None:
    """Reject positive-duration trip overlaps within one physical vehicle block."""
    spans = block_subset.groupby("trip_id").agg(
        start=("arrival_min", "min"), end=("departure_min", "max")
    )
    previous_end = -1
    previous_id = ""
    for trip_id, row in spans.sort_values("start").iterrows():
        if row["start"] < previous_end:
            raise ValueError(
                f"Overlapping trips in block {block_id}: {previous_id} and {trip_id}. "
                "Check service selection and the source schedule."
            )
        previous_end = int(row["end"])
        previous_id = str(trip_id)


def _create_trips_summary(block_subset: pd.DataFrame) -> list[dict[str, Any]]:
    """Build Step 1's trip summaries for a block: trip fields and sorted stop visits."""
    trips_summary = []
    for trip_id, trip_df in block_subset.groupby("trip_id"):
        trip_df_sorted = trip_df.sort_values("stop_sequence")
        stop_times_sequence = [
            (
                int(row["arrival_min"]),
                int(row["departure_min"]),
                str(row["stop_id"]),
                str(row["stop_name"]),
                str(row["trip_id"]),
                bool(row["is_first_stop"]),
                bool(row["is_last_stop"]),
                int(row["stop_sequence"]),
                int(row.get("timepoint", 0)),
            )
            for _, row in trip_df_sorted.iterrows()
        ]
        first_row = trip_df_sorted.iloc[0]
        last_row = trip_df_sorted.iloc[-1]
        flags = trip_df_sorted.get("estimated_time")
        estimated = [] if flags is None else trip_df_sorted.loc[flags.astype(bool), "stop_sequence"]
        trips_summary.append(
            {
                "trip_id": str(trip_id),
                # Occupancy bounds: the trip holds its vehicle from its first scheduled
                # time (first-stop arrival) to its last (last-stop departure).
                "start": int(trip_df_sorted["arrival_min"].iloc[0]),
                "end": int(trip_df_sorted["departure_min"].iloc[-1]),
                # Scheduled first-stop departure and last-stop arrival, as reported.
                "departure": int(first_row["departure_min"]),
                "arrival": int(last_row["arrival_min"]),
                "stop_times_sequence": stop_times_sequence,
                "route_id": str(first_row["route_id"]),
                "route_short_name": _text(first_row.get("route_short_name")),
                "route_long_name": _text(first_row.get("route_long_name")),
                "trip_headsign": _text(first_row.get("trip_headsign")),
                "direction_id": _text(first_row["direction_id"]),
                "block": str(first_row["block_id"]),
                "estimated_stop_sequences": [int(seq) for seq in estimated],
                "first_stop_id": str(first_row["stop_id"]),
                "first_stop_name": str(first_row["stop_name"]),
                "first_stop_seq": int(first_row["stop_sequence"]),
                "last_stop_id": str(last_row["stop_id"]),
                "last_stop_name": str(last_row["stop_name"]),
                "last_stop_seq": int(last_row["stop_sequence"]),
            }
        )
    trips_summary.sort(key=lambda x: x["start"])
    return trips_summary


class Schedule(NamedTuple):
    """Scheduled trips of every block that visits the facility, as Step 1 summarizes them."""

    trips: List[Dict[str, Any]]
    stop_names: Dict[str, str]
    service_ids: List[str]
    estimated_visits: int


def load_schedule(cfg: Dict[str, Any]) -> Schedule:
    """Read the GTFS feed and summarize every trip of the blocks that visit the facility.

    Raises:
        ValueError: If no service runs on the selected day, no selected trip
            serves the facility, or the selected schedule is malformed.
        OSError: If the feed or one of its required files cannot be read.
    """
    gtfs = cfg["GTFS_PATH"]
    if not os.path.isdir(gtfs) and not (os.path.isfile(gtfs) and zipfile.is_zipfile(gtfs)):
        raise NotADirectoryError(f"GTFS_PATH must be a GTFS folder or ZIP archive: {gtfs}")
    data = read_gtfs_tables(
        gtfs, REQUIRED_GTFS_FILES + available_gtfs_files(gtfs, OPTIONAL_GTFS_FILES)
    )
    # GTFS requires a route's short name or its long name, not both.
    route_names = data["routes"].reindex(
        columns=["route_id", "route_short_name", "route_long_name"]
    )
    trips_df = (
        data["trips"]
        .drop(columns=["route_short_name", "route_long_name"], errors="ignore")
        .merge(route_names, on="route_id", how="left", validate="many_to_one")
    )
    if cfg["SERVICE_DATE"].strip():
        service_ids = resolve_service_ids(
            data.get("calendar"), data.get("calendar_dates"), cfg["SERVICE_DATE"]
        )
        if not service_ids:
            raise ValueError(f"No service_id is active on {cfg['SERVICE_DATE']} in this feed.")
        logging.info("Service on %s: %s", cfg["SERVICE_DATE"], ", ".join(service_ids))
    else:
        service_ids = [str(value) for value in cfg["SERVICE_IDS"]]
    merged = select_facility_blocks(trips_df, data["stop_times"], data["stops"], service_ids, cfg)
    reject_selected_frequency_trips(data.get("frequencies"), merged)
    trips: List[Dict[str, Any]] = []
    for block, group in merged.groupby("block_id", sort=False):
        check_for_overlapping_trips(group, str(block))
        if group["trip_id"].nunique() > MAX_TRIPS_PER_BLOCK:
            raise ValueError(f"Block {block} has more than {MAX_TRIPS_PER_BLOCK} trips.")
        trips.extend(_create_trips_summary(group))
    estimated = int(merged["estimated_time"].sum())
    if estimated:
        logging.warning(
            "%d stop visit(s) have interpolated times; conflicts involving them rest on "
            "estimated times.",
            estimated,
        )
    stops = label_unnamed_stops(data["stops"])[0]
    return Schedule(
        trips=trips,
        stop_names=dict(zip(stops["stop_id"].astype(str), stops["stop_name"].astype(str))),
        service_ids=service_ids,
        estimated_visits=estimated,
    )


# ==================================================================================================
# BLOCK RENDERER
# ==================================================================================================

# The renderer must match block_status_timeline_exporter.py's for results to match Step 1.


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
def find_cluster(stop_id: str, clusters: list[dict[str, Any]]) -> Optional[str]:
    """Return cluster name containing the given stop ID, if any."""
    for cluster_item in clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
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


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
def gap_status(gap: int, same_place: bool, settings: dict[str, int]) -> tuple[str, str]:
    """Classify a between-trip gap using this scenario's occupancy assumptions."""
    if not same_place:
        return "DEADHEAD", ""
    if gap <= settings["IN_BAY_LAYOVER_MAX_MINUTES"]:
        return "DWELL", "in bay"
    if gap <= settings["LAYOVER_THRESHOLD"]:
        return "LAYOVER", "overflow"
    return "LONG BREAK", "overflow"


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
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


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
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


# Canonical version lives in utils/block_timeline_helpers.py -- keep this copy in sync.
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


# ==================================================================================================
# ROUTE NAMES AND MOVEMENTS
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
    """A movement at the facility: a route's arrivals, departures or through visits.

    ``route`` is the route_id. ``role`` is ``"arrive"``, ``"depart"``,
    ``"through"``, or ``""`` for every movement of the route; ``direction`` (a
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
        self.routes: Dict[str, Set[str]] = defaultdict(set)
        for route, name in display.items():
            for alias in (route, name, self.label[route]):
                self.routes[alias].add(route)

    def parse(self, text: str) -> Optional[RouteEnd]:
        """Read ``"<route>"``, ``"<route> arrive|depart"`` or ``"<route> through [<dir>]"``.

        Returns:
            The movement, or ``None`` if no route has that name.

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
                "by the label the Route-ends sheet lists, or by its route_id."
            )
        return found.pop() if found else None


def schedule_route_names(trips: Sequence[Dict[str, Any]]) -> RouteNames:
    """Names of the routes in the scheduled trips."""
    return RouteNames(
        {
            str(trip["route_id"]): route_display_name(
                trip["route_short_name"], trip.get("route_long_name"), trip["route_id"]
            )
            for trip in trips
        }
    )


def stop_role(stop: Sequence[Any]) -> str:
    """The movement role of a scheduled stop visit: first stop departs, last arrives."""
    return "depart" if stop[5] else "arrive" if stop[6] else "through"


def scheduled_bays(
    trips: Sequence[Dict[str, Any]], cluster_stops: Mapping[str, str], names: RouteNames
) -> Dict[RouteEnd, Set[str]]:
    """Find every scheduled bay per movement, including visits hidden at handoffs."""
    result: Dict[RouteEnd, Set[str]] = defaultdict(set)
    for trip in trips:
        route = str(trip["route_id"])
        for stop in trip["stop_times_sequence"]:
            if stop[2] not in cluster_stops:
                continue
            role = stop_role(stop)
            direction = str(trip["direction_id"]) if role == "through" else ""
            result[RouteEnd(route, role, direction, names.label[route])].add(cluster_stops[stop[2]])
    return dict(result)


def scheduled_boarding_bays(
    trips: Sequence[Dict[str, Any]], cluster_stops: Mapping[str, str], names: RouteNames
) -> Dict[RouteEnd, Dict[str, Set[str]]]:
    """Scheduled bays of each boarding movement, split by passenger direction.

    Boarding movements are departures and through visits; arrivals (last stops)
    only set passengers down. The passenger direction is the trip's
    direction_id, so a departure movement whose trips leave in both directions
    is counted in each.
    """
    result: Dict[RouteEnd, Dict[str, Set[str]]] = {}
    for trip in trips:
        route, direction = str(trip["route_id"]), str(trip["direction_id"])
        for stop in trip["stop_times_sequence"]:
            role = stop_role(stop)
            if stop[2] not in cluster_stops or role == "arrive":
                continue
            end = RouteEnd(route, role, direction if role == "through" else "", names.label[route])
            result.setdefault(end, {}).setdefault(direction, set()).add(cluster_stops[stop[2]])
    return result


def boarding_bays(
    inventory: Dict[RouteEnd, Dict[str, Set[str]]], change: Change
) -> Dict[Tuple[str, str], Set[str]]:
    """Boarding bays per (route_id, direction_id) after *change*, from movement assignments.

    A moved movement boards at its new bay in every direction it serves; KEEP
    retains each direction's scheduled bays.
    """
    result: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for end, directions in inventory.items():
        for direction, bays in directions.items():
            result[end.route, direction].update(
                {change.moves[end]} if end in change.moves else bays
            )
    return dict(result)


# ==================================================================================================
# DISCOVERY: TRIPS, BLOCK CHAINS, ROUTE-ENDS
# ==================================================================================================


def trip_table(
    trips: Sequence[Dict[str, Any]], cluster_stops: Mapping[str, str], names: RouteNames
) -> DataFrame:
    """One row per scheduled trip with its block, times and facility bays."""
    rows = []
    for trip in trips:
        sequence = trip["stop_times_sequence"]
        first, last = sequence[0], sequence[-1]
        through = [
            (cluster_stops[stop[2]], int(stop[0]))
            for stop in sequence[1:-1]
            if stop[2] in cluster_stops
        ]
        rows.append(
            {
                "trip_id": trip["trip_id"],
                "block": trip["block"],
                "route": str(trip["route_id"]),
                "route_name": names.label[str(trip["route_id"])],
                "headsign": trip["trip_headsign"],
                "direction": trip["direction_id"],
                "start": int(trip["start"]),
                "end": int(trip["end"]),
                "departure": int(trip["departure"]),
                "arrival": int(trip["arrival"]),
                "first_stop": first[2],
                "first_bay": cluster_stops.get(first[2], ""),
                "last_stop": last[2],
                "last_bay": cluster_stops.get(last[2], ""),
                "through_bays": ",".join(sorted({bay for bay, _ in through})),
                "through_visits": through,
            }
        )
    frame = DataFrame(rows, columns=TRIP_COLUMNS)
    return frame.sort_values(["block", "start"]).reset_index(drop=True)


def route_end_of(trip: pd.Series) -> List[Tuple[RouteEnd, str]]:
    """Which movements this trip contributes at the facility: ``[(movement, bay), ...]``."""
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
    next trip's scheduled departure: the layover a timetable shows.
    ``occupancy_gap_min`` runs from the previous trip's last scheduled time
    (last-stop departure) to the next trip's first (first-stop arrival).
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
                    where = f"facility (Bay {prev['last_bay']} to Bay {t['first_bay']})"
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
    """Inventory of movements at the facility with bays, visits, times and tightest gaps."""
    visits: Dict[RouteEnd, List[Tuple[str, int, str]]] = defaultdict(list)
    for _, t in trips.iterrows():
        for route_end, bay in route_end_of(t):
            if route_end.role == "through":
                for visit_bay, minute in t["through_visits"]:
                    if visit_bay == bay:
                        visits[route_end].append((bay, int(minute), t["trip_id"]))
            else:
                minute = t["departure"] if route_end.role == "depart" else t["arrival"]
                visits[route_end].append((bay, int(minute), t["trip_id"]))
    columns = [
        "route_end",
        "route_name",
        "route_id",
        "bays",
        "visits_per_day",
        "first",
        "last",
        "min_schedule_gap_before",
        "min_schedule_gap_after",
        "interlines_with",
    ]
    rows = []
    for route_end, found in sorted(visits.items()):
        minutes = [minute for _, minute, _ in found]
        trip_ids = {trip_id for _, _, trip_id in found}
        before = chains[chains["to_trip"].isin(trip_ids)]
        after = chains[chains["from_trip"].isin(trip_ids)]
        partners = set(before.loc[before["interline"], "from_route"]) | set(
            after.loc[after["interline"], "to_route"]
        )
        rows.append(
            {
                "route_end": str(route_end),
                "route_name": route_end.label,
                "route_id": route_end.route,
                "bays": ",".join(sorted({bay for bay, _, _ in found})),
                "visits_per_day": len(found),
                "first": minutes_to_hhmm(min(minutes)),
                "last": minutes_to_hhmm(max(minutes)),
                "min_schedule_gap_before": (
                    int(before["schedule_gap_min"].min()) if not before.empty else ""
                ),
                "min_schedule_gap_after": (
                    int(after["schedule_gap_min"].min()) if not after.empty else ""
                ),
                "interlines_with": ",".join(sorted(partners)),
            }
        )
    return DataFrame(rows, columns=columns)


def interline_summary(chains: DataFrame) -> DataFrame:
    """Tightest hand-off per ordered route pair, with the count and the tightest example."""
    columns = [
        "from_route",
        "to_route",
        "hand_offs_per_day",
        "min_schedule_gap_min",
        "max_schedule_gap_min",
        "min_occupancy_gap_min",
        "tightest_example",
    ]
    inter = chains[chains["interline"]] if not chains.empty else chains
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
            }
        )
    frame = DataFrame(rows, columns=columns)
    return frame.sort_values("min_schedule_gap_min") if rows else frame


# ==================================================================================================
# OCCUPANCY STANDARDS
# ==================================================================================================


class Change:
    """Absolute bay assignments; a movement not listed keeps its scheduled stops (KEEP)."""

    def __init__(self, moves: Optional[Dict[RouteEnd, str]] = None) -> None:
        """Copy the assignments so later edits to the caller's mapping cannot leak in."""
        self.moves = dict(moves or {})

    def key(self) -> tuple:
        """Return an order-independent identity for caching."""
        return tuple(sorted(self.moves.items()))

    def label(self) -> str:
        """Describe the assignments in a stable order."""
        return "; ".join(f"{key} to Bay {bay}" for key, bay in sorted(self.moves.items()))


def facility_clusters(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The renderer's stop clusters: this facility first, then OTHER_CLUSTERS."""
    return [
        {"name": cfg["CLUSTER_NAME"], "stops": list(cfg["CLUSTER_STOPS"])},
        *(
            {"name": str(other["name"]), "stops": [str(stop) for stop in other["stops"]]}
            for other in cfg["OTHER_CLUSTERS"]
        ),
    ]


class Standard:
    """One occupancy standard: baseline facility occupancy and exact rebuilds of bay changes.

    ``rows`` holds every vehicle-minute at a facility stop, in a bay (ARRIVE,
    DEPART, ARRIVE/DEPART, LOADING, DWELL) or laying over (LAYOVER, LONG
    BREAK), with the movement each belongs to; ``occ`` is its bay subset.
    """

    def __init__(
        self,
        name: str,
        settings: Dict[str, int],
        schedule: Schedule,
        cfg: Dict[str, Any],
        names: RouteNames,
    ) -> None:
        """Render every block that visits the facility under this standard's settings."""
        self.name = name
        self.settings = dict(settings)
        self.cluster_stops: Dict[str, str] = dict(cfg["CLUSTER_STOPS"])
        self.clusters = facility_clusters(cfg)
        self.names = names
        self.bay_names = sorted(set(self.cluster_stops.values()))
        self.bay_index = {bay: i for i, bay in enumerate(self.bay_names)}
        self.cap = np.array(
            [cfg["CLUSTER_CAPACITY"].get(bay, 1) for bay in self.bay_names], dtype=int
        )
        self.target_stops = {bay: stop for stop, bay in self.cluster_stops.items()}
        self.stop_names = schedule.stop_names
        self.block_trips: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for trip in sorted(schedule.trips, key=lambda trip: (trip["start"], trip["trip_id"])):
            self.block_trips[trip["block"]].append(trip)
        self.timeline_end = max(
            cfg["DEFAULT_HOURS"] * 60,
            max(trip["end"] for trip in schedule.trips) + settings["POST_ARRIVAL_MINUTES"] + 1,
        )
        if self.timeline_end >= BIG:
            raise ValueError("The schedule runs beyond the supported service-day range.")
        self.boarding_inventory = scheduled_boarding_bays(schedule.trips, self.cluster_stops, names)
        self.rows = self.render(self.block_trips)
        self.occ = self.rows[self.rows["Status"].isin(BAY_STATUSES)].reset_index(drop=True)
        self.cache: OrderedDict[tuple, DataFrame] = OrderedDict()

    def render(self, blocks: Mapping[str, List[Dict[str, Any]]]) -> DataFrame:
        """Render *blocks* and keep each vehicle-minute spent at a facility stop."""
        records = []
        statuses = BAY_STATUSES | OVERFLOW_STATUSES
        for block, trips in blocks.items():
            rows = build_schedule_rows(
                trips,
                range(self.timeline_end),
                block,
                self.clusters,
                self.settings,
                occupancy_only=True,
            )
            for row in rows:
                bay = self.cluster_stops.get(row["Stop ID"])
                if bay is None or row["Status"] not in statuses:
                    continue
                role = row["Stop Role"]
                if role not in {"arrive", "depart", "through"}:
                    raise ValueError(f"Facility row without a visit role in block {block}.")
                route = str(row["Route"])
                direction = str(row["Direction"]) if role == "through" else ""
                records.append(
                    (
                        block,
                        timestamp_to_minutes(row["Timestamp"]),
                        bay,
                        row["Status"],
                        RouteEnd(route, role, direction, self.names.label.get(route, route)),
                    )
                )
        return DataFrame(records, columns=OCCUPANCY_COLUMNS).astype({"Minute": "int64"})

    def changed_blocks(self, change: Change) -> Dict[str, List[Dict[str, Any]]]:
        """Copies of the affected blocks' trips with each moved visit sent to its new bay."""
        moved_routes = {route_end.route for route_end in change.moves}
        changed: Dict[str, List[Dict[str, Any]]] = {}
        for block, originals in self.block_trips.items():
            revised = []
            affected = False
            for original in originals:
                if original["route_id"] not in moved_routes:
                    revised.append(original)
                    continue
                trip = copy.deepcopy(original)
                stops = []
                for original_stop in trip["stop_times_sequence"]:
                    stop = list(original_stop)
                    role = stop_role(stop)
                    direction = str(trip["direction_id"]) if role == "through" else ""
                    route_end = RouteEnd(str(trip["route_id"]), role, direction)
                    if stop[2] in self.cluster_stops and route_end in change.moves:
                        target = self.target_stops[change.moves[route_end]]
                        affected |= target != stop[2]
                        stop[2], stop[3] = target, self.stop_names.get(target, target)
                    stops.append(tuple(stop))
                trip["stop_times_sequence"] = stops
                for prefix, stop in (("first", stops[0]), ("last", stops[-1])):
                    trip[f"{prefix}_stop_id"], trip[f"{prefix}_stop_name"] = stop[2], stop[3]
                revised.append(trip)
            if affected:
                changed[block] = revised
        return changed

    def apply(self, change: Change) -> DataFrame:
        """Rebuild every affected block completely and return the facility rows."""
        if not change.moves:
            return self.rows
        key = change.key()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        changed = self.changed_blocks(change)
        pieces = [self.rows[~self.rows["Block"].isin(changed)]]
        if changed:
            pieces.append(self.render(changed))
        result = pd.concat(pieces, ignore_index=True)
        if result.duplicated(["Block", "Minute"]).any():
            raise ValueError("Rebuilt schedule contains duplicate vehicle/minute rows.")
        self.cache[key] = result
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        return result

    def rebuilt_boarding_bays(self, change: Change) -> Dict[Tuple[str, str], Set[str]]:
        """Boarding bays per (route_id, direction_id), read from the re-targeted trips.

        Uses the same edited schedule the renderer rebuilds, independently of the
        movement-level assignments the optimizer reasons with.
        """
        revised = self.changed_blocks(change)
        result: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for block, originals in self.block_trips.items():
            for trip in revised.get(block, originals):
                for stop in trip["stop_times_sequence"]:
                    if stop[2] in self.cluster_stops and stop_role(stop) != "arrive":
                        key = (str(trip["route_id"]), str(trip["direction_id"]))
                        result[key].add(self.cluster_stops[stop[2]])
        return dict(result)

    def score(self, frame: DataFrame) -> Tuple[int, Dict[str, int], np.ndarray]:
        """Count over-capacity bay-minutes; one vehicle has one row per minute."""
        frame = frame[frame["Status"].isin(BAY_STATUSES)]
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

    def key_label(self, key: int) -> Tuple[str, int]:
        """Convert an internal conflict key to its bay and minute."""
        return self.bay_names[int(key // BIG)], int(key % BIG)


def bay_rows(frame: DataFrame) -> DataFrame:
    """The rows of *frame* in which a bus occupies a passenger bay."""
    return frame[frame["Status"].isin(BAY_STATUSES)]


def conflict_summary(standard: Standard, cfg: Dict[str, Any], schedule: Schedule) -> Dict[str, Any]:
    """Count bay, layover-space and facility-wide conflict minutes as Step 2 counts them.

    Bay conflicts are counted a second, independent way (distinct blocks per
    bay and minute) and must match the optimizer's scorer.

    Raises:
        ValueError: If the two bay counts disagree.
    """
    frame = standard.rows
    bays = bay_rows(frame)
    buses = bays.groupby(["Bay", "Minute"])["Block"].nunique()
    capacity = buses.index.get_level_values("Bay").map(
        lambda bay: cfg["CLUSTER_CAPACITY"].get(bay, 1)
    )
    over = buses[buses.to_numpy() > capacity.to_numpy()]
    total, per_bay, keys = standard.score(frame)
    if set(over.index) != {standard.key_label(int(key)) for key in keys} or len(over) != total:
        raise ValueError(
            f"{standard.name}: bay conflict counts disagree between the summary and the "
            "optimizer's scorer. No results were published."
        )
    row: Dict[str, Any] = {
        "standard": standard.name,
        "blocks_at_facility": int(frame["Block"].nunique()),
        "trips_in_analyzed_blocks": len(schedule.trips),
        "bay_conflict_minutes": total,
        "minutes_with_any_bay_conflict": int(over.index.get_level_values("Minute").nunique()),
        **{f"Bay {bay} conflict minutes": per_bay.get(bay, 0) for bay in standard.bay_names},
    }
    layover = frame[frame["Status"].isin(OVERFLOW_STATUSES)]
    by_minute = layover.groupby("Minute")["Block"].nunique()
    row["peak_buses_laying_over"] = int(by_minute.max()) if len(by_minute) else 0
    routing = {
        status: space for space, statuses in cfg["OVERFLOW_ROUTING"].items() for status in statuses
    }
    overflow_minutes = 0
    for space in cfg["OVERFLOW_ROUTING"]:
        space_capacity = cfg["OVERFLOW_CAPACITY"].get(space, 1)
        counts = layover[layover["Status"].map(routing).eq(space)].groupby("Minute")["Block"]
        minutes_over = int((counts.nunique() > space_capacity).sum())
        row[f"{space} conflict minutes"] = minutes_over
        overflow_minutes += minutes_over
    row["layover_space_conflict_minutes"] = overflow_minutes if routing else ""
    facility_capacity = int(standard.cap.sum()) + sum(
        cfg["OVERFLOW_CAPACITY"].get(space, 1) for space in cfg["OVERFLOW_ROUTING"]
    )
    present = frame.groupby("Minute")["Block"].nunique()
    row["facility_capacity"] = facility_capacity
    row["facility_wide_conflict_minutes"] = int((present > facility_capacity).sum())
    row.update(standard.settings)
    return row


# ==================================================================================================
# BAY PERMISSIONS
# ==================================================================================================


def resolved_selector(
    text: str, names: RouteNames, known: Set[RouteEnd], whole_route: bool = True
) -> RouteEnd:
    """Read a CONFIGURATION movement or route name; fail clearly on unknown names.

    Raises:
        ValueError: If *text* names no movement at the facility (or, with
            *whole_route*, no route that has one), or fits several routes.
    """
    value = names.parse(text)
    valid = value is not None and value in known
    if value is not None and whole_route and not value.role:
        valid = any(key.route == value.route for key in known)
    if value is None or not valid:
        raise ValueError(
            f"{text!r} does not identify a movement at the facility; use a name from the "
            "Route-ends sheet, e.g. '101 arrive' or '101 through 0'."
        )
    return value


def resolve_bay_permissions(
    cfg: Dict[str, Any], current: Dict[RouteEnd, Set[str]], names: RouteNames
) -> Tuple[Dict[RouteEnd, Set[str]], List[List[RouteEnd]]]:
    """Resolve the bays each movement may move to, and the same-bay groups.

    Returns:
        ``(allowed, groups)``: the permitted new bays per movement (empty when
        it must keep its scheduled stops) and the resolved SAME_BAY_GROUPS.

    Raises:
        ValueError: If a selector is unknown or ambiguous.
    """
    known = set(current)
    options: Dict[RouteEnd, Set[str]] = {}
    for text, bays in cfg["BAY_OPTIONS"].items():
        key = resolved_selector(text, names, known)
        # An ID and its display name may refer to the same selector. Honor both restrictions.
        options[key] = options.get(key, set(bays)) & set(bays)
    locked_routes = set()
    for text in cfg["LOCKED_ROUTES"]:
        key = resolved_selector(text, names, known)
        if key.role:
            raise ValueError(f"LOCKED_ROUTES expects a route, not a movement: {text!r}")
        locked_routes.add(key.route)
    locked_ends = {
        resolved_selector(text, names, known, False) for text in cfg["LOCKED_ROUTE_ENDS"]
    }
    all_bays = set(cfg["CLUSTER_STOPS"].values())
    allowed: Dict[RouteEnd, Set[str]] = {}
    for end in sorted(known):
        matches = [
            bays
            for key, bays in options.items()
            if key == end or (key.route == end.route and not key.role)
        ]
        targets = set(all_bays) if cfg["AUTO_BAY_CANDIDATES"] or matches else set()
        for bays in matches:
            targets &= bays
        if end.route in locked_routes or end in locked_ends:
            targets = set()
        allowed[end] = targets
    groups = [
        [resolved_selector(text, names, known, False) for text in group]
        for group in cfg["SAME_BAY_GROUPS"]
    ]
    return allowed, groups


def permission_inventory(
    current: Dict[RouteEnd, Set[str]], allowed: Dict[RouteEnd, Set[str]]
) -> DataFrame:
    """Describe permissions while keeping KEEP distinct from a uniform reassignment."""
    return DataFrame(
        [
            {
                "route_end": str(end),
                "route_id": end.route,
                "current_bays": ", ".join(sorted(current[end])),
                "permitted_new_bays": ", ".join(sorted(allowed[end])),
                "may_change": bool(
                    allowed[end] - (current[end] if len(current[end]) == 1 else set())
                ),
                "keep_original_stops": True,
            }
            for end in sorted(current)
        ],
        columns=[
            "route_end",
            "route_id",
            "current_bays",
            "permitted_new_bays",
            "may_change",
            "keep_original_stops",
        ],
    )


# ==================================================================================================
# PLAN EVALUATION
# ==================================================================================================


def conflict_cells(standard: Standard, frame: DataFrame) -> Set[Tuple[str, int]]:
    """Unique over-capacity bay-minutes of *frame*."""
    return {standard.key_label(int(key)) for key in standard.score(frame)[2]}


def category_counts(
    direct: Set[Tuple[str, int]], likely: Set[Tuple[str, int]], mask: Tuple[bool, ...]
) -> Dict[str, int]:
    """Classify disjoint bay-minutes without assuming that the standards are nested."""
    result = dict.fromkeys(CATEGORIES, 0)
    for prefix, cells in (("direct", direct), ("likely_only", likely - direct)):
        for _, minute in cells:
            suffix = "peak" if mask[minute % 1440] else "off_peak"
            result[f"{prefix}_{suffix}"] += 1
    return result


def final_route_bays(current: Dict[RouteEnd, Set[str]], change: Change) -> Dict[str, Set[str]]:
    """Count every scheduled passenger bay across a route's roles and directions.

    KEEP retains every original bay, including visits hidden by the renderer's
    shared-boundary ownership rule. Layover spaces are outside this inventory.
    """
    result: Dict[str, Set[str]] = defaultdict(set)
    for end, original in current.items():
        result[end.route].update({change.moves[end]} if end in change.moves else original)
    return dict(result)


def evaluate_plan(
    cfg: Dict[str, Any],
    standards: Dict[str, Standard],
    current: Dict[RouteEnd, Set[str]],
    change: Change,
) -> Dict[str, Any]:
    """Rebuild occupancy and require exact agreement with the assignment model's premise.

    The model assumes a bay change only relabels the bay of each moved
    movement's vehicle-minutes. Every affected block is re-rendered and its
    facility rows, layovers included, must equal that relabeling exactly.
    Boarding bays are read from the re-targeted trips and must equal the
    movement-level inventory's prediction.

    Raises:
        ValueError: If the rebuilt rows or boarding bays differ from the model's premise.
    """
    frames = {}
    cells = {}
    for name, standard in standards.items():
        predicted = standard.rows.copy()
        predicted["Bay"] = [
            change.moves.get(end, bay) for end, bay in zip(predicted["route_end"], predicted["Bay"])
        ]
        actual = standard.apply(change)
        before = Counter(predicted[OCCUPANCY_COLUMNS].itertuples(index=False, name=None))
        after = Counter(actual[OCCUPANCY_COLUMNS].itertuples(index=False, name=None))
        if before != after:
            raise ValueError(
                f"{name}: rebuilt occupancy differs from the optimizer's minute-level model. "
                "No recommendation is accepted; review the renderer assumptions."
            )
        if actual.duplicated(["Block", "Minute"]).any():
            raise ValueError("A proposed plan counts the same physical bus twice in one minute.")
        frames[name] = bay_rows(actual)
        cells[name] = conflict_cells(standard, frames[name])
    first = next(iter(standards.values()))
    boarding = first.rebuilt_boarding_bays(change)
    if boarding != boarding_bays(first.boarding_inventory, change):
        raise ValueError(
            "Rebuilt boarding bays differ from the movement-level model. "
            "No recommendation is accepted; review the schedule's directions and roles."
        )
    direct = cells[cfg["OPTIMIZER_DIRECT_STANDARD"]]
    likely = cells[cfg["OPTIMIZER_LIKELY_STANDARD"]]
    counts = category_counts(direct, likely, peak_mask(cfg["PEAK_WINDOWS"]))
    score = sum(counts[key] * cfg["CONFLICT_WEIGHTS"][key] for key in CATEGORIES)
    return {
        "counts": counts,
        "weighted_score": score,
        "direct": direct,
        "likely": likely,
        "frames": frames,
        "changed_routes": len({end.route for end in change.moves}),
        "changed_movements": len(change.moves),
        "route_bays": final_route_bays(current, change),
        "boarding_bays": boarding,
    }


def plan_violations(
    cfg: Dict[str, Any],
    current: Dict[RouteEnd, Set[str]],
    groups: List[List[RouteEnd]],
    change: Change,
    evaluation: Dict[str, Any],
    baseline: Dict[str, Any],
    limit: Optional[int] = None,
    allowed: Optional[Dict[RouteEnd, Set[str]]] = None,
) -> List[str]:
    """List the rules a plan breaks.

    The rules are permitted bays (when *allowed* is given), the route budget, the
    score ceiling, bays per route, consistent boarding bays, same-bay groups and
    the direct-conflict policy.
    """
    problems = []
    labels = {end.route: end.label for end in current}
    if allowed is not None:
        outside = sorted(
            str(end) for end, bay in change.moves.items() if bay not in allowed.get(end, set())
        )
        if outside:
            problems.append(f"moves not permitted by BAY_OPTIONS or locks: {', '.join(outside)}")
    if limit is not None and evaluation["changed_routes"] > limit:
        problems.append(f"changes {evaluation['changed_routes']} routes; the budget is {limit}")
    score_limit = cfg["WEIGHTED_SCORE_LIMIT"]
    if score_limit is not None and evaluation["weighted_score"] > score_limit:
        problems.append(
            f"weighted score {evaluation['weighted_score']} exceeds WEIGHTED_SCORE_LIMIT "
            f"{score_limit}"
        )
    bay_limit = cfg["MAX_BAYS_PER_ROUTE"]
    if bay_limit is not None:
        over = sorted(
            labels.get(route, route)
            for route, bays in evaluation["route_bays"].items()
            if len(bays) > bay_limit
        )
        if over:
            problems.append(f"routes using more than {bay_limit} bays: {', '.join(over)}")
    if cfg["REQUIRE_CONSISTENT_BOARDING_BAYS"]:
        split = sorted(
            f"{labels.get(route, route)} direction {direction or '(blank)'} "
            f"({', '.join(sorted(bays))})"
            for (route, direction), bays in evaluation["boarding_bays"].items()
            if len(bays) > 1
        )
        if split:
            problems.append(f"passengers board at more than one bay: {'; '.join(split)}")
    for group in groups:
        final = [({change.moves[end]} if end in change.moves else current[end]) for end in group]
        if any(len(bays) != 1 or bays != final[0] for bays in final):
            problems.append(f"same-bay group {', '.join(map(str, group))} is split")
    policy = cfg["DIRECT_CONFLICT_POLICY"]
    if policy == "no_increase" and len(evaluation["direct"]) > len(baseline["direct"]):
        problems.append("more direct conflict minutes than the baseline")
    new = evaluation["direct"] - baseline["direct"]
    if policy == "no_new" and new:
        problems.append(f"direct conflicts at {len(new)} bay-minute(s) clear in the baseline")
    return problems


def plan_summary(
    label: str,
    limit: Optional[int],
    change: Change,
    evaluation: Dict[str, Any],
    baseline: Dict[str, Any],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Make a spreadsheet row that preserves raw counts beside the weighted score."""
    return {
        "plan": label,
        "route_change_limit": limit,
        "changed_routes": evaluation["changed_routes"],
        "changed_movements": evaluation["changed_movements"],
        "max_bays_used_by_one_route": max(map(len, evaluation["route_bays"].values()), default=0),
        "max_boarding_bays_per_direction": max(
            map(len, evaluation["boarding_bays"].values()), default=0
        ),
        **evaluation["counts"],
        "weighted_score": evaluation["weighted_score"],
        "weighted_improvement": baseline["weighted_score"] - evaluation["weighted_score"],
        "direct_total": len(evaluation["direct"]),
        "likely_total": len(evaluation["likely"]),
        "direct_removed": len(baseline["direct"] - evaluation["direct"]),
        "direct_retained": len(baseline["direct"] & evaluation["direct"]),
        "direct_new": len(evaluation["direct"] - baseline["direct"]),
        "changes": change.label(),
        "validation": "passed",
        **info,
    }


def conflict_rows(label: str, cfg: Dict[str, Any], evaluation: Dict[str, Any]) -> List[Dict]:
    """List mutually exclusive conflict minutes with the actual buses under each standard."""
    vehicles = {}
    for name, frame in evaluation["frames"].items():
        vehicles[name] = frame.groupby(["Bay", "Minute"])["Block"].agg(list).to_dict()
    mask = peak_mask(cfg["PEAK_WINDOWS"])
    rows = []
    for cell in sorted(
        evaluation["direct"] | evaluation["likely"], key=lambda key: (key[1], key[0])
    ):
        bay, minute = cell
        period = "peak" if mask[minute % 1440] else "off_peak"
        prefix = "direct" if cell in evaluation["direct"] else "likely_only"
        row = {
            "plan": label,
            "bay": bay,
            "minute": minute,
            "time": minutes_to_hhmm(minute),
            "category": f"{prefix}_{period}",
            "weight": cfg["CONFLICT_WEIGHTS"][f"{prefix}_{period}"],
        }
        for kind in ("direct", "likely"):
            name = cfg[f"OPTIMIZER_{kind.upper()}_STANDARD"]
            buses = vehicles[name].get(cell, [])
            row[f"{kind}_buses"] = len(buses)
            row[f"{kind}_blocks"] = ", ".join(sorted(map(str, buses)))
        rows.append(row)
    return rows


def assignment_rows(label: str, current: Dict[RouteEnd, Set[str]], change: Change) -> List[Dict]:
    """Export unambiguous route IDs, visit roles, directions, and target bays."""
    return [
        {
            "plan": label,
            "route_id": end.route,
            "route_end": str(end),
            "role": end.role,
            "direction": end.direction,
            "original_bays": ", ".join(sorted(current[end])),
            "target_bay": change.moves.get(end, "KEEP ORIGINAL STOPS"),
            "changed": end in change.moves,
        }
        for end in sorted(current)
    ]


def route_bay_rows(
    label: str, cfg: Dict[str, Any], current: Dict[RouteEnd, Set[str]], change: Change
) -> List[Dict]:
    """Show route-wide bay counts separately from individual movement reassignments."""
    original = final_route_bays(current, Change())
    final = final_route_bays(current, change)
    labels = {end.route: end.label for end in current}
    limit = cfg["MAX_BAYS_PER_ROUTE"]
    return [
        {
            "plan": label,
            "route_id": route,
            "route_name": labels[route],
            "original_bays": ", ".join(sorted(original[route])),
            "final_bays": ", ".join(sorted(bays)),
            "final_bay_count": len(bays),
            "max_bays_allowed": limit,
            "bay_limit_passed": limit is None or len(bays) <= limit,
            "changed_movements": sum(end.route == route for end in change.moves),
        }
        for route, bays in sorted(final.items())
    ]


def boarding_rows(
    label: str,
    cfg: Dict[str, Any],
    inventory: Dict[RouteEnd, Dict[str, Set[str]]],
    evaluation: Dict[str, Any],
) -> List[Dict]:
    """Show where each route boards in each passenger direction, before and after a plan."""
    original = boarding_bays(inventory, Change())
    movements: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    labels = {end.route: end.label for end in inventory}
    for end, directions in inventory.items():
        for direction in directions:
            movements[end.route, direction].append(str(end))
    return [
        {
            "plan": label,
            "route_id": route,
            "route_name": labels[route],
            "direction": direction,
            "boarding_movements": "; ".join(sorted(movements[route, direction])),
            "original_boarding_bays": ", ".join(sorted(original[route, direction])),
            "final_boarding_bays": ", ".join(sorted(bays)),
            "final_bay_count": len(bays),
            "consistent": len(bays) == 1,
            "rule_required": cfg["REQUIRE_CONSISTENT_BOARDING_BAYS"],
        }
        for (route, direction), bays in sorted(evaluation["boarding_bays"].items())
    ]


def objective_description(cfg: Dict[str, Any]) -> str:
    """Describe the actual priority order and hard score restriction."""
    if cfg["OPTIMIZER_OBJECTIVE"] == "min_changes":
        return (
            f"Weighted score <= {cfg['WEIGHTED_SCORE_LIMIT']}; minimize changed routes, "
            "then changed movements, then weighted score."
        )
    return "Weighted disjoint conflict minutes; then fewer routes; then fewer movements."


# ==================================================================================================
# PULP / CBC
# ==================================================================================================


def load_pulp() -> Any:
    """Load PuLP 3.x and check that its bundled CBC solver can run.

    Raises:
        OptimizerUnavailable: If PuLP is missing, too old, or has no CBC.
    """
    try:
        import pulp
    except ImportError as exc:
        raise OptimizerUnavailable(
            "PuLP is missing. In the Python environment running this tool, install "
            "'PuLP==3.2.2', then rerun. The conflict check and movement lists were written."
        ) from exc
    if not str(pulp.__version__).startswith("3."):
        raise OptimizerUnavailable(
            f"This script uses the PuLP 3.x API (tested with PuLP==3.2.2); found "
            f"{pulp.__version__}. Install 'PuLP==3.2.2' in this environment."
        )
    if not pulp.PULP_CBC_CMD(msg=False).available():
        raise OptimizerUnavailable("PuLP is installed, but its bundled CBC solver is unavailable.")
    return pulp


def wait_for_cbc(process: Any, timeout: float, log_path: Path) -> bool:
    """Wait with progress messages; kill only this child on timeout or interruption.

    Args:
        process: The CBC child process owned by this solve.
        timeout: External wall-clock allowance, including the shutdown grace period.
        log_path: Solver log to identify in progress messages.

    Returns:
        True if the external timeout was reached, otherwise False.
    """
    started = time.monotonic()
    timed_out = False
    try:
        while process.poll() is None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                logging.warning(
                    "CBC exceeded its %.0f-second process limit; stopping PID %s. Log: %s",
                    timeout,
                    process.pid,
                    log_path,
                )
                break
            try:
                process.wait(timeout=min(CBC_PROGRESS_INTERVAL_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                logging.info(
                    "CBC is still running: %.0f seconds elapsed; process limit %.0f seconds.",
                    time.monotonic() - started,
                    timeout,
                )
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise OptimizerUnavailable(
                    f"CBC PID {process.pid} did not exit after termination. See {log_path}."
                ) from exc
    return timed_out


def solve_cbc_process(solver: Any, problem: Any, pulp: Any, log_path: Path) -> int:
    """Run CBC as a child process with an external timeout and isolated solver files.

    PuLP writes the MPS model and warm start and reads CBC's result. Owning the
    child process means CBC's internal time limit is not the only protection
    against a hang. A forcibly stopped solve contributes no new solution; the
    caller keeps its previous validated plan. Partial solution files are not
    trusted.
    """
    work_dir = Path(tempfile.mkdtemp(prefix=f"{log_path.stem}_", dir=log_path.parent))
    variables, variable_names, constraint_names, _ = problem.writeMPS(
        str(work_dir / "model.mps"), rename=1
    )
    # CBC sees simple basenames, including the MIP start on Windows. Running the
    # child in work_dir avoids changing Python's working directory or splitting
    # paths containing spaces.
    command = [str(Path(solver.path).resolve()), "model.mps"]
    if problem.sense == pulp.LpMaximize:
        command.append("-max")
    if solver.optionsDict.get("warmStart", False):
        solver.writesol(
            str(work_dir / "start.mst"), problem, variables, variable_names, constraint_names
        )
        command.extend(["-mips", "start.mst"])
    command.extend(["-sec", str(solver.timeLimit)])
    for option in solver.options + solver.getOptions():
        parts = option.split()
        command.extend([f"-{parts[0]}", *parts[1:]])
    command.extend(["-branch", "-printingOptions", "all", "-solution", "solution.sol"])
    problem.assignStatus(pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound)
    for variable in variables:
        variable.varValue = None
    logging.info(
        "Starting CBC: internal limit %s seconds; external limit %.0f seconds. Files: %s",
        solver.timeLimit,
        solver.process_time_limit,
        work_dir,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(work_dir),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            solver.external_timeout = wait_for_cbc(process, solver.process_time_limit, log_path)
    except OSError as exc:
        raise OptimizerUnavailable(f"Cannot run CBC: {exc}. See {log_path}.") from exc
    if solver.external_timeout:
        logging.warning("CBC was forcibly stopped. No new assignment is accepted from this solve.")
        return problem.status
    if process.returncode != 0:
        raise OptimizerUnavailable(f"CBC exited with code {process.returncode}. See {log_path}.")
    solution_path = work_dir / "solution.sol"
    if not solution_path.is_file():
        raise OptimizerUnavailable(f"CBC exited without a solution file. See {log_path}.")
    # PuLP defaults missing values to zero. Require a complete variable listing
    # before parsing so a truncated file cannot become a fabricated assignment.
    records = solution_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
    reported = set()
    for record in records:
        parts = record.split()
        if parts and parts[0] == "**":
            parts = parts[1:]
        if len(parts) >= 4:
            reported.add(parts[1])
    if not set(variable_names.values()).issubset(reported):
        raise OptimizerUnavailable(f"CBC wrote an incomplete solution file. See {log_path}.")
    status, values, reduced_costs, prices, slacks, solution_status = solver.readsol_MPS(
        str(solution_path), problem, variables, variable_names, constraint_names
    )
    problem.assignVarsVals(values)
    problem.assignVarsDj(reduced_costs)
    problem.assignConsPi(prices)
    problem.assignConsSlack(slacks, activity=True)
    problem.assignStatus(status, solution_status)
    return status


def make_cbc_solver(pulp: Any, cfg: Dict[str, Any], log_path: Path, warm_start: bool) -> Any:
    """Build a monitored CBC solver, using serial execution on Windows.

    CBC's serial setting is zero; one requests a worker thread. The normal
    single-CPU setting maps to serial mode, and threaded CBC is avoided on Windows.
    """

    class MonitoredCBC(pulp.PULP_CBC_CMD):
        """PuLP's CBC interface with the child process run by :func:`solve_cbc_process`."""

        def solve_CBC(self, lp: Any, use_mps: bool = True) -> int:
            """Solve *lp* in a monitored child process."""
            if not use_mps:
                raise ValueError("The monitored bay solver requires MPS format.")
            return solve_cbc_process(self, lp, pulp, log_path)

    requested_threads = cfg["SOLVER_THREADS"]
    threads = 0 if sys.platform == "win32" or requested_threads == 1 else requested_threads
    if sys.platform == "win32":
        logging.info("Using CBC serial mode on Windows (threads=0).")
    solver = MonitoredCBC(
        msg=False,
        timeLimit=cfg["SOLVER_TIME_LIMIT_SECONDS"],
        gapRel=cfg["SOLVER_RELATIVE_GAP"],
        gapAbs=0,
        threads=threads,
        warmStart=warm_start,
        keepFiles=True,
        timeMode="elapsed",
        logPath=str(log_path),
        options=["randomSeed 1", "randomCbcSeed 1"],
    )
    solver.external_timeout = False
    solver.process_time_limit = cfg["SOLVER_TIME_LIMIT_SECONDS"] + CBC_TIMEOUT_GRACE_SECONDS
    return solver


class AssignmentModel:
    """Exact discrete assignment model for the direct and likely occupancy standards."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        standards: Dict[str, Standard],
        current: Dict[RouteEnd, Set[str]],
        names: RouteNames,
        allowed: Dict[RouteEnd, Set[str]],
        groups: List[List[RouteEnd]],
        pulp: Any,
    ) -> None:
        """Build choices, vehicle occupancy, exact conflict flags, and policy constraints."""
        self.cfg, self.standards, self.names, self.pulp = cfg, standards, names, pulp
        self.current = current
        self.ends = sorted(self.current)
        self.routes = sorted({end.route for end in self.ends})
        if set(self.ends) != set(allowed):
            raise ValueError("Allowed movements do not match the scheduled facility movements.")
        self.model = pulp.LpProblem("weighted_bay_assignment", pulp.LpMinimize)
        self.variable_count = 0
        self.choices: Dict[RouteEnd, tuple] = {}
        self.x: Dict[tuple, Any] = {}
        self.changed: Dict[RouteEnd, Any] = {}
        self.route_changed: Dict[str, Any] = {}
        self.flags: Dict[str, Dict[tuple, Any]] = {}
        self.likely_only: Dict[tuple, Any] = {}
        self.mask = peak_mask(cfg["PEAK_WINDOWS"])
        self.baseline = evaluate_plan(cfg, standards, current, Change())
        bays = sorted(set(cfg["CLUSTER_STOPS"].values()))
        for end in self.ends:
            targets = [bay for bay in sorted(allowed[end]) if self.current[end] != {bay}]
            self.choices[end] = (None, *targets)
            if not targets:
                self.x[end, None], self.changed[end] = 1, 0
                continue
            for bay in self.choices[end]:
                self.x[end, bay] = self.binary("assign")
            self.add(pulp.lpSum(self.x[end, bay] for bay in self.choices[end]) == 1)
            self.changed[end] = 1 - self.x[end, None]
        for route in self.routes:
            members = [self.changed[end] for end in self.ends if end.route == route]
            if all(type(value) is int and value == 0 for value in members):
                self.route_changed[route] = 0
                continue
            flag = self.binary("route_changed")
            self.route_changed[route] = flag
            for member in members:
                self.add(flag >= member)
            self.add(flag <= pulp.lpSum(members))
        self.route_bay_used: Dict[Tuple[str, str], Any] = {}
        self.bay_limit = cfg["MAX_BAYS_PER_ROUTE"]
        if self.bay_limit is not None:
            for route in self.routes:
                members = [end for end in self.ends if end.route == route]
                for bay in bays:
                    signals = [self.bay_signal(end, bay, self.current[end]) for end in members]
                    self.route_bay_used[route, bay] = self.usage_flag("route_bay_used", signals)
                self.add(
                    pulp.lpSum(
                        used
                        for (member_route, _), used in self.route_bay_used.items()
                        if member_route == route
                    )
                    <= self.bay_limit
                )
        # Consistent boarding: per route and passenger direction, the departures and
        # through visits that board there may use one bay between them.
        self.boarding = next(iter(standards.values())).boarding_inventory
        self.boarding_used: Dict[Tuple[str, str, str], Any] = {}
        if cfg["REQUIRE_CONSISTENT_BOARDING_BAYS"]:
            members_by_pair: Dict[Tuple[str, str], List[Tuple[RouteEnd, Set[str]]]] = defaultdict(
                list
            )
            for end, directions in self.boarding.items():
                for direction, current_bays in directions.items():
                    members_by_pair[end.route, direction].append((end, current_bays))
            for (route, direction), members in sorted(members_by_pair.items()):
                for bay in bays:
                    signals = [self.bay_signal(end, bay, current) for end, current in members]
                    self.boarding_used[route, direction, bay] = self.usage_flag(
                        "boarding_bay_used", signals
                    )
                self.add(pulp.lpSum(self.boarding_used[route, direction, bay] for bay in bays) <= 1)
        self.groups = groups
        for group in self.groups:
            for end in group:
                if end not in self.choices:
                    raise ValueError(f"Unknown same-bay group member: {end}")
                if len(self.current[end]) != 1:
                    self.add(pulp.lpSum([self.x[end, None]]) == 0)
            for end in group[1:]:
                for bay in bays:
                    self.add(
                        self.assignment_signal(group[0], bay) == self.assignment_signal(end, bay)
                    )
        for name, standard in standards.items():
            self.flags[name] = self.build_conflicts(standard)
        direct = self.flags[cfg["OPTIMIZER_DIRECT_STANDARD"]]
        likely = self.flags[cfg["OPTIMIZER_LIKELY_STANDARD"]]
        self.category_expressions: Dict[str, Any] = {
            key: pulp.LpAffineExpression() for key in CATEGORIES
        }
        for cell in sorted(set(direct) | set(likely)):
            direct_flag, likely_flag = direct.get(cell, 0), likely.get(cell, 0)
            if type(direct_flag) is int and type(likely_flag) is int:
                only = int(bool(likely_flag) and not direct_flag)
            elif type(direct_flag) is int:
                only = likely_flag if direct_flag == 0 else 0
            elif type(likely_flag) is int:
                only = 1 - direct_flag if likely_flag else 0
            else:
                only = self.binary("likely_only")
                self.add(only <= likely_flag)
                self.add(only <= 1 - direct_flag)
                self.add(only >= likely_flag - direct_flag)
            self.likely_only[cell] = only
            suffix = "peak" if self.mask[cell[1] % 1440] else "off_peak"
            self.category_expressions[f"direct_{suffix}"] += direct_flag
            self.category_expressions[f"likely_only_{suffix}"] += only
        self.weighted = pulp.lpSum(
            cfg["CONFLICT_WEIGHTS"][key] * value for key, value in self.category_expressions.items()
        )
        policy = cfg["DIRECT_CONFLICT_POLICY"]
        if policy == "no_increase":
            self.add(pulp.lpSum(direct.values()) <= len(self.baseline["direct"]))
        elif policy == "no_new":
            for cell, flag in direct.items():
                if cell not in self.baseline["direct"]:
                    self.add(pulp.lpSum([flag]) == 0)
        self.route_count = pulp.lpSum(self.route_changed.values())
        self.movement_count = pulp.lpSum(self.changed.values())
        self.objective_mode = cfg["OPTIMIZER_OBJECTIVE"]
        self.score_limit = cfg["WEIGHTED_SCORE_LIMIT"]
        if self.score_limit is not None:
            self.add(self.weighted <= self.score_limit)
        # Integer scaling makes each tie breaker strictly subordinate to its predecessor.
        if self.objective_mode == "min_changes":
            self.primary_scale = 1
            self.movement_scale = self.score_limit + 1
            self.route_scale = (len(self.ends) + 1) * self.movement_scale
        else:
            self.movement_scale = 1
            self.route_scale = len(self.ends) + 1
            self.primary_scale = (len(self.routes) + 1) * self.route_scale
        self.objective = (
            self.weighted * self.primary_scale
            + self.route_count * self.route_scale
            + self.movement_count * self.movement_scale
        )
        self.model += self.objective
        self.budget = self.route_count <= 0
        self.add(self.budget)
        self.incumbent_bound: Optional[Any] = None
        logging.info(
            "PuLP model: %d binaries, %d constraints, %d movements.",
            self.variable_count,
            len(self.model.constraints),
            len(self.ends),
        )

    def binary(self, prefix: str) -> Any:
        """Allocate a stable, ID-independent variable name and enforce a size limit."""
        self.variable_count += 1
        if self.variable_count > self.cfg["MAX_OPTIMIZER_BINARY_VARIABLES"]:
            raise ValueError("Optimizer variable limit exceeded; restrict bays or raise the limit.")
        return self.pulp.LpVariable(f"{prefix}_{self.variable_count}", cat="Binary")

    def add(self, constraint: Any) -> None:
        """Add one constraint without silently discarding an oversized model."""
        if len(self.model.constraints) >= self.cfg["MAX_OPTIMIZER_CONSTRAINTS"]:
            raise ValueError(
                "Optimizer constraint limit exceeded; restrict bays or raise the limit."
            )
        self.model += constraint

    def bay_signal(self, end: RouteEnd, bay: str, kept_bays: Set[str]) -> Any:
        """1 when *end* ends up visiting *bay*: moved there, or kept with a visit there.

        *kept_bays* are the scheduled bays that KEEP retains for the visits counted.
        """
        terms = [self.x[end, bay]] if bay in self.choices[end] else []
        if bay in kept_bays:
            terms.append(self.x[end, None])
        return self.pulp.lpSum(terms)

    def usage_flag(self, prefix: str, signals: List[Any]) -> Any:
        """1 when any signal is on: a constant where the signals settle it, else a binary."""
        if any(len(signal) == 0 and signal.constant == 1 for signal in signals):
            return 1
        if all(len(signal) == 0 and signal.constant == 0 for signal in signals):
            return 0
        used = self.binary(prefix)
        for signal in signals:
            self.add(used >= signal)
        self.add(used <= self.pulp.lpSum(signals))
        return used

    def assignment_signal(self, end: RouteEnd, bay: str) -> Any:
        """Indicate a uniform final bay, including an unchanged uniform assignment."""
        terms = [self.x[end, bay]] if bay in self.choices[end] else []
        if self.current[end] == {bay}:
            terms.append(self.x[end, None])
        return self.pulp.lpSum(terms)

    def build_conflicts(self, standard: Standard) -> Dict[tuple, Any]:
        """Constrain both directions of the over-capacity test for each possible bay-minute."""
        coefficients: Dict[tuple, Dict[tuple, int]] = defaultdict(dict)
        fixed: Counter = Counter()
        grouped = standard.occ.groupby(["route_end", "Minute", "Bay"], sort=False).size()
        for (end, minute, original_bay), count in grouped.items():
            if end not in self.choices:
                raise ValueError(f"Occupied movement {end} has no assignment choice.")
            for choice in self.choices[end]:
                cell = (original_bay if choice is None else choice, int(minute))
                variable = self.x[end, choice]
                if type(variable) is int:
                    fixed[cell] += int(count)
                else:
                    key = (end, choice)
                    coefficients[cell][key] = coefficients[cell].get(key, 0) + int(count)
        result = {}
        for cell in sorted(set(fixed) | set(coefficients)):
            terms = coefficients.get(cell, {})
            ends = {end for end, _ in terms}
            maximum = fixed[cell] + sum(
                max(terms.get((end, bay), 0) for bay in self.choices[end]) for end in ends
            )
            minimum = fixed[cell] + sum(
                min(terms.get((end, bay), 0) for bay in self.choices[end]) for end in ends
            )
            capacity = self.cfg["CLUSTER_CAPACITY"].get(cell[0], 1)
            if maximum <= capacity:
                continue
            if minimum > capacity:
                result[cell] = 1
                continue
            load = fixed[cell] + self.pulp.lpSum(
                count * self.x[key] for key, count in terms.items()
            )
            flag = self.binary("conflict")
            self.add(load <= capacity + (maximum - capacity) * flag)
            self.add(load >= minimum + (capacity + 1 - minimum) * flag)
            result[cell] = flag
        return result

    def combined_score(self, evaluation: Dict[str, Any]) -> int:
        """Encode the configured objective priorities with exact integer multipliers."""
        return (
            evaluation["weighted_score"] * self.primary_scale
            + evaluation["changed_routes"] * self.route_scale
            + evaluation["changed_movements"] * self.movement_scale
        )

    def permitted(self, change: Change, limit: int, evaluation: Dict[str, Any]) -> bool:
        """Validate domain, group, route-budget and direct-conflict restrictions independently."""
        if any(
            end not in self.choices or bay not in self.choices[end]
            for end, bay in change.moves.items()
        ):
            return False
        return not plan_violations(
            self.cfg, self.current, self.groups, change, evaluation, self.baseline, limit
        )

    def seed(self, change: Change, evaluation: Dict[str, Any]) -> None:
        """Warm-start all binary decisions from a previously validated plan."""
        for (end, bay), variable in self.x.items():
            if type(variable) is not int:
                variable.setInitialValue(int(change.moves.get(end) == bay))
        for route, variable in self.route_changed.items():
            if type(variable) is not int:
                variable.setInitialValue(int(any(end.route == route for end in change.moves)))
        route_bays = final_route_bays(self.current, change)
        for (route, bay), variable in self.route_bay_used.items():
            if type(variable) is not int:
                variable.setInitialValue(int(bay in route_bays[route]))
        boarding = boarding_bays(self.boarding, change)
        for (route, direction, bay), variable in self.boarding_used.items():
            if type(variable) is not int:
                variable.setInitialValue(int(bay in boarding[route, direction]))
        for name, flags in self.flags.items():
            cells = conflict_cells(self.standards[name], evaluation["frames"][name])
            for cell, variable in flags.items():
                if type(variable) is not int:
                    variable.setInitialValue(int(cell in cells))
        for cell, variable in self.likely_only.items():
            # Some entries are affine expressions (1 - direct), not independent variables.
            if isinstance(variable, self.pulp.LpVariable) and variable.name.startswith(
                "likely_only"
            ):
                variable.setInitialValue(int(cell in evaluation["likely"] - evaluation["direct"]))
        cost = self.combined_score(evaluation)
        if self.incumbent_bound is None:
            self.incumbent_bound = self.objective <= cost
            self.add(self.incumbent_bound)
        else:
            self.incumbent_bound.changeRHS(cost - self.objective.constant)

    def solution_change(self) -> Optional[Change]:
        """Accept only integral variable values that satisfy every encoded constraint."""
        for variable in self.model.variables():
            if variable.name == "__dummy":
                continue
            value = variable.value()
            if value is None or not math.isfinite(value) or abs(value - round(value)) > TOLERANCE:
                return None
            if variable.lowBound is not None and value < variable.lowBound - TOLERANCE:
                return None
            if variable.upBound is not None and value > variable.upBound + TOLERANCE:
                return None
        for constraint in self.model.constraints.values():
            value = constraint.value()
            if value is None or (constraint.sense == 0 and abs(value) > TOLERANCE):
                return None
            if constraint.sense == -1 and value > TOLERANCE:
                return None
            if constraint.sense == 1 and value < -TOLERANCE:
                return None
        moves = {}
        for end, choices in self.choices.items():
            chosen = [bay for bay in choices if self.pulp.value(self.x[end, bay]) > 0.5]
            if len(chosen) != 1:
                return None
            if chosen[0] is not None:
                moves[end] = chosen[0]
        return Change(moves)


def solver_status(pulp: Any, model: Any, log: str) -> Dict[str, Any]:
    """Distinguish CBC's proof from its gap- or time-limited integer incumbents.

    PuLP can label a time-limited incumbent LpStatusOptimal while sol_status is
    only LpSolutionIntegerFeasible. Require both status fields and CBC's proof line.
    """
    proof = (
        model.status == pulp.LpStatusOptimal
        and model.sol_status == pulp.LpSolutionOptimal
        and re.search(r"^Result - Optimal solution found\s*$", log, re.MULTILINE) is not None
    )
    match = re.search(r"^Result - (.+)$", log, re.MULTILINE)
    reason = match.group(1).strip() if match else pulp.LpStatus.get(model.status, "Unknown")
    bound = re.search(r"^Lower bound:\s*([-+\deE.]+)", log, re.MULTILINE)
    lower = float(bound.group(1)) + model.objective.constant if bound else None
    return {
        "proven_optimal": bool(proof),
        "solver_stop": reason,
        "pulp_status": pulp.LpStatus.get(model.status, str(model.status)),
        "pulp_solution_status": pulp.LpSolution.get(model.sol_status, str(model.sol_status)),
        "combined_objective_bound": lower,
    }


def resolve_assignments(
    assignments: Mapping[str, str],
    names: RouteNames,
    current: Dict[RouteEnd, Set[str]],
    setting: str,
) -> Change:
    """Read exact movement -> bay assignments; an unchanged uniform bay becomes KEEP.

    Raises:
        ValueError: If a movement is unknown, incomplete or listed twice.
    """
    moves = {}
    seen = set()
    for selector, bay in assignments.items():
        end = names.parse(selector)
        if end is None or end not in current or end in seen:
            raise ValueError(
                f"{setting}: unknown or duplicate movement {selector!r}; use a name from the "
                "Route-ends sheet, e.g. '101 arrive' or '101 through 0'."
            )
        seen.add(end)
        if current[end] != {bay}:
            moves[end] = bay
    return Change(moves)


def starting_plan(model: AssignmentModel) -> Optional[Tuple[Change, Dict[str, Any]]]:
    """Rebuild a configured proposal before accepting it as the solver's starting point.

    These assignments are optional suggestions, not fixed constraints. An
    unchanged uniform assignment becomes KEEP.

    Raises:
        ValueError: If a suggestion is unknown, prohibited, or breaks a rule on
            the current data.
    """
    configured = model.cfg["OPTIMIZER_STARTING_ASSIGNMENTS"]
    if not configured:
        return None
    change = resolve_assignments(
        configured, model.names, model.current, "OPTIMIZER_STARTING_ASSIGNMENTS"
    )
    for end, bay in change.moves.items():
        if bay not in model.choices[end]:
            raise ValueError(f"Starting-plan bay is not permitted: {end} -> {bay!r}.")
    evaluation = evaluate_plan(model.cfg, model.standards, model.current, change)
    if not model.permitted(change, len(model.routes), evaluation):
        problems = plan_violations(
            model.cfg, model.current, model.groups, change, evaluation, model.baseline
        )
        raise ValueError(
            "OPTIMIZER_STARTING_ASSIGNMENTS breaks a rule on the current data "
            f"({'; '.join(problems)}). Review it, or set it to {{}} to search without it."
        )
    logging.info(
        "Validated starting plan: score %d, %d changed routes, %d changed movements.",
        evaluation["weighted_score"],
        evaluation["changed_routes"],
        evaluation["changed_movements"],
    )
    return change, evaluation


# ==================================================================================================
# REPORT
# ==================================================================================================


READ_ME_NOTES: Tuple[Tuple[str, str], ...] = (
    (
        "Score limit",
        "WEIGHTED_SCORE_LIMIT is a hard ceiling. A baseline above it is reference only.",
    ),
    (
        "Starting plan",
        "Configured assignments are rebuilt and validated, then used as an optional starting "
        "solution. They do not lock routes or bays. Blank route budget = reference row.",
    ),
    ("Standards", "Both model the same scheduled service; never add them together."),
    ("Direct precedence", "A direct conflict is not also charged as likely-only."),
    ("Likely total", "Overlaps direct; do not add direct_total and likely_total."),
    ("Peak periods", "Local clock, start inclusive/end exclusive; repeat after 24:00."),
    ("Schedules", "Fixed. Only the facility bay a movement uses may change."),
    ("KEEP", "Preserves original visits, including movements using multiple bays."),
    ("Groups", "Every member must finish at one common bay, including locked members."),
    (
        "Route bay limit",
        "MAX_BAYS_PER_ROUTE counts distinct passenger bays across ALL roles and directions, "
        "including unchanged visits. It is not a one-bay-per-direction rule.",
    ),
    (
        "New direct conflicts",
        "With no_new, every direct-conflict bay-minute must already exist in the baseline. "
        "This restricts bay-minutes, not bus pairs.",
    ),
    (
        "Implementation burden",
        "changed_routes counts route IDs; changed_movements counts reassigned arrival, "
        "departure and through-direction movements. Plans are alternatives.",
    ),
    (
        "Boarding bays",
        "With REQUIRE_CONSISTENT_BOARDING_BAYS, each route boards at one bay per passenger "
        "direction (GTFS direction_id): its departures and its through visits in that "
        "direction share a bay. Arrivals only set down and are not counted; KEEP visits are. "
        "Checked against the re-targeted schedule for every plan.",
    ),
    (
        "Named proposals",
        "NAMED_PROPOSALS are fixed layouts scored like any plan. Rule breaks, including moves "
        "outside BAY_OPTIONS or locks, are listed in policy_issues rather than rejected.",
    ),
    ("Permitted new bays", "Restrict changes; KEEP may preserve a bay outside that list."),
    (
        "Automatic candidates",
        "AUTO_BAY_CANDIDATES assumes the listed passenger bays are interchangeable unless "
        "BAY_OPTIONS or locks restrict them.",
    ),
    (
        "Layover and facility totals",
        "Reported on Conflict summary and not optimized. Every plan's rebuilt layover rows "
        "equal the baseline's, so bay changes leave these totals unchanged.",
    ),
    ("Proof", "Proven only for this model, the allowed bays, groups and route budget."),
    (
        "External timeout",
        "If CBC exceeds its time limit plus 30 seconds, its process is terminated. Only the "
        "previous validated plan can be retained; no optimality is claimed.",
    ),
    (
        "Validation",
        "Every reported plan matched a full re-render of the affected vehicle blocks.",
    ),
    (
        "Combined objective gap",
        "Uses the configured priority order. In min_changes mode it is not a conflict-score "
        "gap. Objective values from different modes are not comparable.",
    ),
    ("Operational review", "Review bay access, bus suitability and inter-bay travel separately."),
)


class Report:
    """The workbook's contents, republished atomically as the run progresses."""

    def __init__(self, path: Path, cfg: Dict[str, Any]) -> None:
        """Start an empty report for *path*."""
        self.path = path
        self.cfg = cfg
        self.status = "in_progress"
        self.reason = ""
        self.notes: List[Tuple[str, str]] = []
        self.metadata: Dict[str, Any] = {}
        self.summary: List[Dict[str, Any]] = []
        self.discovery: List[Tuple[str, DataFrame]] = []
        self.options = DataFrame()
        self.plans: List[Dict[str, Any]] = []
        self.assignments: List[Dict[str, Any]] = []
        self.route_usage: List[Dict[str, Any]] = []
        self.boarding: List[Dict[str, Any]] = []
        self.boarding_inventory: Dict[RouteEnd, Dict[str, Set[str]]] = {}
        self.minutes: List[Dict[str, Any]] = []

    def add_plan(
        self,
        row: Dict[str, Any],
        current: Dict[RouteEnd, Set[str]],
        change: Optional[Change] = None,
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one plan row and, when it has an assignment, its detail rows."""
        self.plans.append(row)
        if change is None or evaluation is None:
            return
        self.assignments.extend(assignment_rows(row["plan"], current, change))
        self.route_usage.extend(route_bay_rows(row["plan"], self.cfg, current, change))
        self.boarding.extend(
            boarding_rows(row["plan"], self.cfg, self.boarding_inventory, evaluation)
        )
        self.minutes.extend(conflict_rows(row["plan"], self.cfg, evaluation))

    def sheets(self) -> List[Tuple[str, DataFrame]]:
        """Every sheet in workbook order."""
        read_me = [
            ("Status", self.status),
            ("Reason", self.reason),
            ("Facility", self.cfg["CLUSTER_NAME"]),
            ("Objective", objective_description(self.cfg)),
            *self.notes,
            *READ_ME_NOTES,
            *[(str(key), str(value)) for key, value in self.metadata.items()],
        ]
        plans = DataFrame(self.plans)
        if not plans.empty:
            limit = self.cfg["WEIGHTED_SCORE_LIMIT"]
            plans["objective_mode"] = self.cfg["OPTIMIZER_OBJECTIVE"]
            plans["weighted_score_limit"] = limit
            plans["score_limit_passed"] = [
                None if pd.isna(score) else limit is None or score <= limit
                for score in plans.get("weighted_score", pd.Series(None, index=plans.index))
            ]
        return [
            ("Read me", DataFrame(read_me, columns=["Item", "Value"])),
            ("Conflict summary", DataFrame(self.summary)),
            ("Plan comparison", plans),
            ("Assignments", DataFrame(self.assignments)),
            ("Route bay usage", DataFrame(self.route_usage)),
            ("Boarding bays", DataFrame(self.boarding)),
            ("Conflict minutes", DataFrame(self.minutes, columns=MINUTE_COLUMNS)),
            *self.discovery,
            ("Bay options", self.options),
            (
                "Configuration",
                DataFrame(
                    {
                        "setting": list(self.cfg),
                        "value": [json.dumps(v, default=str) for v in self.cfg.values()],
                    }
                ),
            ),
        ]

    def publish(self, final: bool = False) -> Path:
        """Replace the workbook atomically; keep running if Excel holds it open.

        A checkpoint that cannot replace the file (for example because it is open
        in Excel) is skipped with a warning. The final write falls back to a new
        file name so the results are never lost.

        Returns:
            The path written, or the current path when a checkpoint was skipped.
        """
        sheets = self.sheets()
        temporary = self.path.with_name(self.path.stem + ".tmp.xlsx")
        write_workbook(temporary, sheets)
        try:
            os.replace(temporary, self.path)
            return self.path
        except OSError as exc:
            if not final:
                logging.warning("Could not update %s (%s); is it open? Continuing.", self.path, exc)
                temporary.unlink(missing_ok=True)
                return self.path
            fallback = self.path.with_name(
                f"{self.path.stem}_{datetime.now().strftime('%H%M%S')}.xlsx"
            )
            os.replace(temporary, fallback)
            logging.warning(
                "Could not replace %s (%s); wrote %s instead.", self.path, exc, fallback
            )
            self.path = fallback
            return fallback


def write_workbook(path: Path, sheets: Sequence[Tuple[str, DataFrame]]) -> None:
    """Write readable sheets with frozen headers, filters, widths and literal text."""
    with open_report_workbook(str(path)) as writer:
        for name, frame in sheets:
            if len(frame) > 1_048_575:
                raise ValueError(f"Sheet {name!r} exceeds Excel's row limit.")
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            sheet.freeze_panes = "A2"
            if sheet.max_row > 1:
                sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                values = [cell.value for cell in column[:500]]
                width = min(65, max(14, *(len(str(value or "")) + 2 for value in values)))
                sheet.column_dimensions[column[0].column_letter].width = width
                for cell in column:
                    # Route and stop names are data, never Excel formulas.
                    if isinstance(cell.value, str) and cell.value.startswith("="):
                        cell.data_type = "s"


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


# ==================================================================================================
# OPTIMIZATION
# ==================================================================================================


def run_optimizer(
    cfg: Dict[str, Any],
    standards: Dict[str, Standard],
    current: Dict[RouteEnd, Set[str]],
    names: RouteNames,
    allowed: Dict[RouteEnd, Set[str]],
    groups: List[List[RouteEnd]],
    solver_dir: Path,
    report: Report,
) -> None:
    """Solve each route budget, validate every incumbent and checkpoint the report.

    Sets ``report.status`` to ``optimized`` (every budget proven optimal or
    infeasible), ``optimizer_partial`` (validated plans, not all proven) or
    ``optimizer_no_solution``.
    """
    pulp = load_pulp()
    model = AssignmentModel(cfg, standards, current, names, allowed, groups, pulp)
    solver_dir.mkdir(parents=True, exist_ok=True)
    report.metadata.update(
        {
            "PuLP distribution": importlib.metadata.version("PuLP"),
            "binary variables": model.variable_count,
            "constraints": len(model.model.constraints),
            "conflict-score multiplier": model.primary_scale,
            "changed-route multiplier": model.route_scale,
            "changed-movement multiplier": model.movement_scale,
            "discovered routes": len(model.routes),
            "maximum passenger bays per route": cfg["MAX_BAYS_PER_ROUTE"],
        }
    )
    empty = Change()
    baseline = model.baseline
    best: Optional[Tuple[Change, Dict[str, Any]]] = (
        (empty, baseline) if model.permitted(empty, 0, baseline) else None
    )
    starting = starting_plan(model)
    if starting is not None:
        seed_change, seed_evaluation = starting
        info = {"plan_status": "starting_plan", "proven_optimal": False, "policy_compliant": True}
        report.add_plan(
            plan_summary("Starting plan", None, seed_change, seed_evaluation, baseline, info),
            current,
            seed_change,
            seed_evaluation,
        )
    report.publish()
    limits = sorted(
        {len(model.routes) if value == "all" else value for value in cfg["ROUTE_CHANGE_LIMITS"]}
    )
    if best is None:
        logging.info("The baseline is reference only: it breaks the score limit or a policy rule.")
    logging.info("Objective: %s", objective_description(cfg))
    for limit in limits:
        label = f"Up to {limit} route(s)"
        logging.info("Optimizing %s", label)
        model.budget.changeRHS(limit)
        if starting is not None and model.permitted(starting[0], limit, starting[1]):
            if best is None or model.combined_score(starting[1]) < model.combined_score(best[1]):
                best = starting
        if model.variable_count == 0:
            # Exactly one assignment is possible. Check it directly, avoiding CBC's
            # special empty-model output and PuLP's artificial dummy variable.
            if model.permitted(empty, limit, baseline):
                combined = model.combined_score(baseline)
                info = {
                    "plan_status": "optimal",
                    "proven_optimal": True,
                    "solver_stop": "Only one permitted assignment; verified directly.",
                    "combined_objective": combined,
                    "combined_objective_bound": combined,
                    "combined_objective_gap": 0.0,
                    "solve_seconds": 0.0,
                    "policy_compliant": True,
                    "all_routes_eligible": limit >= len(model.routes),
                }
                report.add_plan(
                    plan_summary(label, limit, empty, baseline, baseline, info),
                    current,
                    empty,
                    baseline,
                )
            else:
                report.add_plan(
                    {
                        "plan": label,
                        "route_change_limit": limit,
                        "plan_status": "infeasible",
                        "proven_optimal": False,
                        "validation": "no plan",
                        "solver_stop": "The only assignment breaks the score or policy rules.",
                    },
                    current,
                )
            report.publish()
            continue
        if best is not None:
            model.seed(*best)
        log_path = solver_dir / f"cbc_max_{limit}_routes.log"
        if cfg["WRITE_SOLVER_MODEL"]:
            model.model.writeLP(str(solver_dir / f"model_max_{limit}_routes.lp"))
        solver = make_cbc_solver(pulp, cfg, log_path, warm_start=best is not None)
        start = time.monotonic()
        try:
            model.model.solve(solver)
        except pulp.PulpSolverError as exc:
            raise OptimizerUnavailable(f"CBC could not run: {exc}. See {log_path}") from exc
        elapsed = time.monotonic() - start
        logging.info("CBC returned for %s after %.1f seconds; validating.", label, elapsed)
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        info = solver_status(pulp, model.model, log)
        info.update(
            solve_seconds=round(elapsed, 3),
            solver_log=str(log_path),
            external_timeout=solver.external_timeout,
            solver_process_limit_seconds=solver.process_time_limit,
            all_routes_eligible=limit >= len(model.routes),
        )
        if solver.external_timeout:
            info.update(
                proven_optimal=False,
                combined_objective_bound=None,
                solver_stop=(
                    f"CBC exceeded the external {solver.process_time_limit}-second limit. "
                    "The process was terminated; no new assignment was accepted."
                ),
            )
        change = (
            model.solution_change()
            if model.model.sol_status in {pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible}
            else None
        )
        if model.model.status in {pulp.LpStatusInfeasible, pulp.LpStatusUnbounded}:
            change = None
            if best is not None:
                raise ValueError("CBC rejected a model with a validated feasible incumbent.")
        if change is not None:
            evaluation = evaluate_plan(cfg, standards, current, change)
            if not model.permitted(change, limit, evaluation):
                raise ValueError(
                    "Solver assignment breaks a bay, group, route-budget or policy rule."
                )
            predicted = {
                key: int(round(pulp.value(expr)))
                for key, expr in model.category_expressions.items()
            }
            if predicted != evaluation["counts"]:
                raise ValueError("Rebuilt conflict categories do not match the optimization model.")
            if best is not None and model.combined_score(evaluation) > model.combined_score(
                best[1]
            ):
                raise ValueError("Solver incumbent is worse than its enforced validated bound.")
            best = change, evaluation
            info["plan_status"] = "optimal" if info["proven_optimal"] else "feasible_not_proven"
        elif info["proven_optimal"]:
            raise ValueError("CBC reported optimality without a valid integral assignment.")
        elif best is not None:
            change, evaluation = best
            info.update(plan_status="retained_previous_feasible", proven_optimal=False)
            logging.warning("%s: retaining the previous validated plan.", label)
        else:
            state = (
                "infeasible" if model.model.status == pulp.LpStatusInfeasible else "no_incumbent"
            )
            report.add_plan(
                {
                    "plan": label,
                    "route_change_limit": limit,
                    "plan_status": state,
                    "validation": "no plan",
                    **info,
                },
                current,
            )
            report.publish()
            continue
        combined = model.combined_score(evaluation)
        info["policy_compliant"] = True
        lower = combined if info["proven_optimal"] else info["combined_objective_bound"]
        info["combined_objective_bound"] = lower
        info["combined_objective"] = combined
        info["combined_objective_gap"] = (
            max(0.0, (combined - lower) / max(1, abs(combined))) if lower is not None else None
        )
        report.add_plan(
            plan_summary(label, limit, change, evaluation, baseline, info),
            current,
            change,
            evaluation,
        )
        report.publish()
        logging.info(
            "%s: %s, weighted score %d (baseline %d), %d route(s) changed.",
            label,
            info["plan_status"],
            evaluation["weighted_score"],
            baseline["weighted_score"],
            evaluation["changed_routes"],
        )
    completed = [plan for plan in report.plans if plan.get("route_change_limit") is not None]
    if not any(row.get("validation") == "passed" for row in completed):
        report.status, report.reason = "optimizer_no_solution", "No feasible plan was found."
    elif all(row.get("plan_status") in {"optimal", "infeasible"} for row in completed):
        report.status = "optimized"
        report.reason = "Every route budget was solved to completion."
    else:
        report.status = "optimizer_partial"
        report.reason = "Validated plans are available; some route budgets were not proven optimal."


# ==================================================================================================
# WORKFLOW
# ==================================================================================================


def optimize_decision(cfg: Dict[str, Any], summary: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Decide whether to optimize; layover and facility conflicts alone do not trigger it."""
    if cfg["OPTIMIZE_MODE"] == "never":
        return False, "Check-only run (OPTIMIZE_MODE = 'never'); PuLP was not used."
    if any(row["bay_conflict_minutes"] > 0 for row in summary):
        return True, "Bay conflicts found."
    if cfg["OPTIMIZE_MODE"] == "always":
        return True, "Optimization requested on a conflict-free baseline."
    if any(
        row["layover_space_conflict_minutes"] or row["facility_wide_conflict_minutes"]
        for row in summary
    ):
        return False, (
            "Only layover-space or facility-wide conflicts found. Bay changes cannot reduce "
            "them; review layover capacity separately."
        )
    return False, "No bay conflicts under either standard."


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Check the facility, list its movements and optimize bay assignments when needed.

    Returns:
        ``{"status", "reason", "workbook", "plans"}`` for the completed run.

    Raises:
        ValueError: If the configuration or the schedule is invalid.
        OSError: If the feed cannot be read or outputs cannot be written.
        RunLogError: If the required run log could not be written.
    """
    validate_configuration(cfg)
    label = slug(cfg["SCENARIO_LABEL"])
    stem = Path(cfg["OUTPUT_DIR"]).expanduser() / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir, suffix = stem, 1
    while run_dir.exists():
        suffix += 1
        run_dir = stem.with_name(f"{stem.name}_{suffix}")
    run_dir.mkdir(parents=True)
    report = Report(run_dir / f"{label}_bay_plans.xlsx", cfg)
    logging.info("Writing to %s", run_dir)

    schedule = load_schedule(cfg)
    names = schedule_route_names(schedule.trips)
    current = scheduled_bays(schedule.trips, cfg["CLUSTER_STOPS"], names)
    standards = {}
    for name, settings in cfg["OCCUPANCY_STANDARDS"].items():
        logging.info("Rendering occupancy under %s.", name)
        standards[name] = Standard(name, settings, schedule, cfg, names)
    report.summary = [conflict_summary(standard, cfg, schedule) for standard in standards.values()]
    for row in report.summary:
        logging.info(
            "%s: %d bay conflict minutes; %s layover-space; %d facility-wide.",
            row["standard"],
            row["bay_conflict_minutes"],
            row["layover_space_conflict_minutes"] or "no",
            row["facility_wide_conflict_minutes"],
        )
    trips = trip_table(schedule.trips, cfg["CLUSTER_STOPS"], names)
    chains = build_block_chains(trips)
    report.discovery = [
        ("Route-ends", build_route_ends(trips, chains)),
        ("Interlines", interline_summary(chains)),
        ("Block chains", chains),
    ]
    report.notes = [
        ("Run folder", str(run_dir.resolve())),
        ("Service IDs", ", ".join(schedule.service_ids)),
        ("Stop visits with interpolated times", str(schedule.estimated_visits)),
    ]
    allowed, groups = resolve_bay_permissions(cfg, current, names)
    report.options = permission_inventory(current, allowed)
    report.boarding_inventory = next(iter(standards.values())).boarding_inventory
    baseline = evaluate_plan(cfg, standards, current, Change())
    problems = plan_violations(cfg, current, groups, Change(), baseline, baseline)
    report.add_plan(
        plan_summary(
            "Baseline",
            None,
            Change(),
            baseline,
            baseline,
            {
                "plan_status": "baseline",
                "proven_optimal": False,
                "policy_compliant": not problems,
                "policy_issues": "; ".join(problems),
            },
        ),
        current,
        Change(),
        baseline,
    )
    for name, assignments in cfg["NAMED_PROPOSALS"].items():
        change = resolve_assignments(assignments, names, current, f"NAMED_PROPOSALS[{name!r}]")
        evaluation = evaluate_plan(cfg, standards, current, change)
        problems = plan_violations(
            cfg, current, groups, change, evaluation, baseline, allowed=allowed
        )
        info = {
            "plan_status": "proposal",
            "proven_optimal": False,
            "policy_compliant": not problems,
            "policy_issues": "; ".join(problems),
        }
        report.add_plan(
            plan_summary(name.strip(), None, change, evaluation, baseline, info),
            current,
            change,
            evaluation,
        )
        logging.info(
            "Proposal %s: weighted score %d (baseline %d)%s.",
            name,
            evaluation["weighted_score"],
            baseline["weighted_score"],
            f"; breaks: {info['policy_issues']}" if problems else "",
        )
    optimize, reason = optimize_decision(cfg, report.summary)
    report.status = "optimizer_pending" if optimize else "checked"
    report.reason = reason
    report.publish()
    require_run_log(write_run_log(report.path, cfg), cfg)
    if optimize:
        try:
            run_optimizer(
                cfg, standards, current, names, allowed, groups, run_dir / "solver", report
            )
        except OptimizerUnavailable as exc:
            report.status, report.reason = "optimizer_unavailable", str(exc)
        except Exception as exc:
            # Keep the checked results and every plan validated so far, marked as failed.
            report.status, report.reason = "failed", str(exc)
            report.publish(final=True)
            raise
    path = report.publish(final=True)
    level = logging.INFO if report.status in {"checked", "optimized"} else logging.WARNING
    logging.log(level, "%s: %s", report.status, report.reason)
    logging.info("Workbook written to %s", path)
    return {
        "status": report.status,
        "reason": report.reason,
        "workbook": str(path),
        "plans": report.plans,
    }


# ==================================================================================================
# RUN LOG
# ==================================================================================================


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


def write_run_log(output_file: Path, cfg: Dict[str, Any]) -> bool:
    """Write the ``_runlog.txt`` sidecar for *output_file* (same folder, same stem).

    The log captures this script's CONFIGURATION block verbatim, between the
    ``# === BEGIN CONFIG ===`` / ``# === END CONFIG ===`` markers, and appends
    the effective settings, including notebook edits and command-line overrides.

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
        "BAY ASSIGNMENT OPTIMIZER RUN LOG",
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
        json.dumps(cfg, indent=2, default=str),
        "=" * 72,
    ]
    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to %s", log_path)
    return True


def require_run_log(written: bool, cfg: Dict[str, Any]) -> None:
    """Raise :class:`RunLogError` when a run log failed and ``REQUIRE_RUN_LOG`` is set."""
    if not written and cfg["REQUIRE_RUN_LOG"]:
        raise RunLogError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )


# ==================================================================================================
# MAIN
# ==================================================================================================


# Canonical version lives in utils/cli_helpers.py -- keep this copy in sync.
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


# CLI flag destination -> CONFIGURATION constant.
CLI_SETTINGS: Dict[str, str] = {
    "gtfs_path": "GTFS_PATH",
    "output_dir": "OUTPUT_DIR",
    "scenario_label": "SCENARIO_LABEL",
    "service_date": "SERVICE_DATE",
    "service_ids": "SERVICE_IDS",
    "optimize": "OPTIMIZE_MODE",
    "objective": "OPTIMIZER_OBJECTIVE",
    "score_limit": "WEIGHTED_SCORE_LIMIT",
    "time_limit": "SOLVER_TIME_LIMIT_SECONDS",
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser; every default is the CONFIGURATION value."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gtfs-path", default=GTFS_PATH, help="GTFS folder or .zip.")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for run subfolders.")
    p.add_argument("--scenario-label", default=SCENARIO_LABEL, help="Run and file label.")
    p.add_argument("--service-date", default=SERVICE_DATE, help="Service day as YYYYMMDD.")
    p.add_argument(
        "--service-ids",
        nargs="*",
        default=SERVICE_IDS,
        help="service_ids operating together (when no --service-date).",
    )
    p.add_argument("--optimize", choices=OPTIMIZE_MODES, default=OPTIMIZE_MODE)
    p.add_argument("--objective", choices=OBJECTIVE_MODES, default=OPTIMIZER_OBJECTIVE)
    p.add_argument(
        "--score-limit", type=int, default=WEIGHTED_SCORE_LIMIT, help="WEIGHTED_SCORE_LIMIT."
    )
    p.add_argument(
        "--time-limit",
        type=int,
        default=SOLVER_TIME_LIMIT_SECONDS,
        help="CBC seconds per route budget.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help='JSON file of further CONFIGURATION overrides, e.g. {"BAY_OPTIONS": {...}}. '
        "Flags given on the command line take precedence over it.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run with CONFIGURATION, optional JSON overrides and command-line flags.

    Returns:
        Process exit code: 0 when the run completes (including runs where no
        feasible plan exists, which the workbook explains), 1 if the input,
        configuration or solver fails, 2 if required paths are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    # argparse fills a default only where the namespace lacks the attribute, so a
    # sentinel marks each flag that was not given; a flag repeating its default counts.
    not_given = object()
    args = parser.parse_args(
        notebook_safe_argv(argv),
        namespace=argparse.Namespace(**dict.fromkeys(CLI_SETTINGS, not_given)),
    )
    try:
        cfg = default_config()
        if args.config:
            overrides = json.loads(args.config.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                raise ValueError("--config must contain a JSON object of setting overrides.")
            cfg.update(overrides)
        given = {dest for dest in CLI_SETTINGS if getattr(args, dest) is not not_given}
        for dest in given:
            cfg[CLI_SETTINGS[dest]] = getattr(args, dest)
        # A service day is chosen one way: a date given alone replaces the service_ids.
        if "service_date" in given and "service_ids" not in given:
            cfg["SERVICE_IDS"] = []
        if "service_ids" in given and "service_date" not in given:
            cfg["SERVICE_DATE"] = ""
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    unset = [key for key in ("GTFS_PATH", "OUTPUT_DIR") if is_placeholder_path(cfg[key])]
    if unset:
        logging.warning(
            "Default placeholder paths detected for: %s. Update the CONFIGURATION section "
            "or pass --gtfs-path / --output-dir before running.",
            ", ".join(unset),
        )
        return 2
    try:
        result = run(cfg)
    except (RunLogError, ValueError, OSError, KeyError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    if result["status"] == "optimizer_unavailable":
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
