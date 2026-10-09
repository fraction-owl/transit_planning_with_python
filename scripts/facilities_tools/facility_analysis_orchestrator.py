"""Run route schedules, block timelines and bay optimization for chosen stop clusters.

One run produces three outputs for one or more clusters (for example the bays
of a transit center): print-style schedules for every route that stops at a
cluster, minute-by-minute vehicle-block timelines, and bay-assignment plans per
cluster. The service day is resolved once -- a representative weekday unless
SERVICE_DATE or SERVICE_IDS says otherwise -- and every step receives the same
explicit service_ids, so the three outputs describe the same day.

Each step is the repository's own script, run unchanged in its own Python
process with its CONFIGURATION constants set from this file:
``gtfs_exports/timepoint_schedule_exporter.py`` for the routes found at the
cluster stops; ``gtfs_exports/block_status_timeline_exporter.py`` for every
block touching a cluster stop; and ``facilities_tools/bay_assignment_optimizer.py``
once per cluster, with the other clusters as OTHER_CLUSTERS so that a layover
there is not read as a deadhead. A failed step is logged and the rest still run.

Inputs
------
- A GTFS folder or ``.zip`` with trips, stop_times, stops and routes, plus
  calendar.txt and/or calendar_dates.txt. The schedule step also needs
  agency.txt and calendar.txt, and exports only service_ids in calendar.txt.
- The three step scripts, under ``SCRIPTS_ROOT`` (by default, this repository).

Outputs
-------
A new ``<RUN_LABEL>_<timestamp>`` folder in ``OUTPUT_DIR`` holding:
- ``routes_serving_clusters.csv``: each route and direction stopping at each
  cluster on the service day, its trips there and first and last visit.
- ``1_route_schedules/``, ``2_block_timelines/`` and ``3_bay_optimizer/``: each
  step's own outputs and run logs (one optimizer subfolder per cluster).
- ``step_settings/``: the settings given to each step (JSON) and its console
  output (``.log``).
- ``facility_analysis_orchestrator_runlog.txt``: the verbatim CONFIGURATION
  block, effective settings, service day, and each step's command and result.

Typical usage
-------------
Set the paths and CLUSTERS in the CONFIGURATION section (or pass the matching
CLI flags; ``--config`` takes a JSON file of further overrides) and run from a
shell, ArcGIS Pro's Python window, or a Jupyter notebook. Optimizing needs
PuLP; OPTIMIZE_MODE = "never" reports conflicts and movements without it.

Limitations
-----------
GTFS has no deadhead or garage trips: deadheads, pull-outs and pull-ins in the
step outputs are inferred from block_id, and their times are assumptions.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_PATH = r"Path\To\GTFS_folder_or_zip"
OUTPUT_DIR = r"Path\To\facility_analysis_output"  # each run writes a new subfolder here
RUN_LABEL = "weekday"  # names the run folder

# Service day, resolved once and given to every step as explicit service_ids. Leave
# both empty to use a representative weekday (the most common set of weekday
# service_ids in calendar.txt / calendar_dates.txt), give one date (YYYYMMDD), or
# list the service_ids that operate together. Not both.
SERVICE_DATE = ""
SERVICE_IDS: List[str] = []
SCHEDULE_LABEL = "Weekday"  # used in the schedule exporter's folder and file names

# Stop clusters in block_status_timeline_exporter.py's CLUSTER_DEFINITIONS format
# (the text cluster_stops_from_zones_*.py writes). "stops" are GTFS stop_ids, one per
# physical bay; a stop may belong to one cluster only. Optional: "two_bay_stops" and
# "three_bay_stops" (stops that hold two or three buses at once) and "overflow_bays"
# (names of layover spaces; the optimizer treats them as one pool).
CLUSTERS: Dict[str, Dict[str, List[str]]] = {
    "Pentagon Zone": {
        "stops": [
            "36",  # 1107 | METRO PENTAGON BAY L7
            "2781",  # 2816 | METRO PENTAGON BAY L4
            "3234",  # 6548 | METRO PENTAGON BAY L6
        ],
        "overflow_bays": [],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
    "Mark Center Zone": {
        "stops": [
            "6257",  # 6257 | Mark Center Transit Station
        ],
        "overflow_bays": [],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
}

# Bay labels for the optimizer's reports, by stop_id; unlisted stops are labelled by
# their stop_id. Labels must be unique within a cluster. Example:
#   BAY_LABELS = {"36": "L7", "2781": "L4", "3234": "L6"}
BAY_LABELS: Dict[str, str] = {}

# Steps to run.
RUN_SCHEDULES = True
RUN_TIMELINES = True
RUN_OPTIMIZER = True
# "if_conflicts": optimize only when a bay conflict exists; "never": report conflicts
# and movements only (PuLP not needed); "always": optimize even a conflict-free day.
OPTIMIZE_MODE = "if_conflicts"

# Further CONFIGURATION constants for each step script, by name, applied after the
# values this script sets. Settings this script controls (paths, service day,
# clusters, route and stop filters) cannot be set here. Examples:
#   SCHEDULE_SETTINGS = {"TIME_FORMAT_OPTION": "12"}
#   TIMELINE_SETTINGS = {"PRE_DEPARTURE_MINUTES": 3}
#   OPTIMIZER_SETTINGS = {"SOLVER_TIME_LIMIT_SECONDS": 600}
#   OPTIMIZER_SETTINGS_BY_CLUSTER = {"Pentagon Zone": {"LOCKED_ROUTES": ["101"]}}
SCHEDULE_SETTINGS: Dict[str, Any] = {}
TIMELINE_SETTINGS: Dict[str, Any] = {}
OPTIMIZER_SETTINGS: Dict[str, Any] = {}
OPTIMIZER_SETTINGS_BY_CLUSTER: Dict[str, Dict[str, Any]] = {}

# Folder holding the repository's scripts/ subfolders; "" uses the folder above this
# file. Set it when running from a notebook cell, which has no file location.
SCRIPTS_ROOT = r""
# Python that runs each step; "" uses the interpreter running this script (in ArcGIS
# Pro's Python window, the python.exe of the active environment).
PYTHON_EXECUTABLE = r""

ROUTES_FILENAME = r"routes_serving_clusters.csv"
RUN_LOG_FILENAME = r"facility_analysis_orchestrator_runlog.txt"

# Every output must be traceable: a failed run-log write aborts the script.
# Set to False only when writing to a genuinely read-only location.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

CONFIG_KEYS: Tuple[str, ...] = tuple(name for name in globals() if name.isupper())

# Step scripts, relative to SCRIPTS_ROOT.
STEP_SCRIPTS: Dict[str, Tuple[str, str]] = {
    "schedules": ("gtfs_exports", "timepoint_schedule_exporter.py"),
    "timelines": ("gtfs_exports", "block_status_timeline_exporter.py"),
    "optimizer": ("facilities_tools", "bay_assignment_optimizer.py"),
}
STEP_SWITCHES: Dict[str, str] = {
    "schedules": "RUN_SCHEDULES",
    "timelines": "RUN_TIMELINES",
    "optimizer": "RUN_OPTIMIZER",
}
SCHEDULE_DIRNAME = "1_route_schedules"
TIMELINE_DIRNAME = "2_block_timelines"
OPTIMIZER_DIRNAME = "3_bay_optimizer"
SETTINGS_DIRNAME = "step_settings"

CLUSTER_KEYS: Tuple[str, ...] = ("stops", "overflow_bays", "two_bay_stops", "three_bay_stops")
OPTIMIZE_MODES: Tuple[str, ...] = ("never", "if_conflicts", "always")
# The optimizer's single layover space when a cluster lists overflow_bays.
OVERFLOW_POOL = "Overflow pool"

# Step settings this script sets; the *_SETTINGS dictionaries may not override them.
RESERVED_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "SCHEDULE_SETTINGS": (
        "GTFS_FOLDER_PATH",
        "BASE_OUTPUT_PATH",
        "FILTER_SERVICE_IDS",
        "FILTER_IN_ROUTES",
        "SERVICE_LABEL_OVERRIDES",
    ),
    "TIMELINE_SETTINGS": (
        "GTFS_FOLDER_PATH",
        "BLOCK_OUTPUT_FOLDER",
        "SCENARIO_NAME",
        "CALENDAR_SERVICE_IDS",
        "SERVICE_DATE",
        "CLUSTER_DEFINITIONS",
        "BUS_STOP_CLUSTERS_STEP1",
        "ROUTE_SHORTNAME_FILTER",
        "STOP_ID_FILTER",
        "STOP_CODE_FILTER",
    ),
    "OPTIMIZER_SETTINGS": (
        "GTFS_PATH",
        "OUTPUT_DIR",
        "SCENARIO_LABEL",
        "SERVICE_DATE",
        "SERVICE_IDS",
        "CLUSTER_NAME",
        "CLUSTER_STOPS",
        "OTHER_CLUSTERS",
        "OPTIMIZE_MODE",
    ),
}

# Runs a step script in a child process: load it as a module (its main() guard does
# not fire), replace CONFIGURATION constants from a JSON file, then call main().
# Setting the constants is what a notebook user does; the script itself is unchanged.
STEP_BOOTSTRAP = """\
import importlib.util, json, sys
script, settings_file = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("orchestrated_step", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open(settings_file, encoding="utf-8") as handle:
    settings = json.load(handle)
unknown = sorted(key for key in settings if not key.isupper() or not hasattr(module, key))
if unknown:
    sys.exit("Unknown CONFIGURATION settings for " + script + ": " + ", ".join(unknown))
for key, value in settings.items():
    setattr(module, key, value)
sys.exit(module.main())
"""

# Child log lines start with their own timestamp; it is dropped when echoed here.
CHILD_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ")

REQUIRED_GTFS_FILES: Tuple[str, ...] = ("trips.txt", "stop_times.txt", "stops.txt", "routes.txt")
OPTIONAL_GTFS_FILES: Tuple[str, ...] = ("calendar.txt", "calendar_dates.txt")

# CLI flag destination -> CONFIGURATION constant.
CLI_SETTINGS: Dict[str, str] = {
    "gtfs_path": "GTFS_PATH",
    "output_dir": "OUTPUT_DIR",
    "run_label": "RUN_LABEL",
    "service_date": "SERVICE_DATE",
    "service_ids": "SERVICE_IDS",
    "run_schedules": "RUN_SCHEDULES",
    "run_timelines": "RUN_TIMELINES",
    "run_optimizer": "RUN_OPTIMIZER",
    "optimize": "OPTIMIZE_MODE",
    "scripts_root": "SCRIPTS_ROOT",
    "python_executable": "PYTHON_EXECUTABLE",
}

# =============================================================================
# FUNCTIONS
# =============================================================================


class ConfigError(ValueError):
    """A setting is invalid or does not match the feed; reported with exit code 2."""


class RunLogError(RuntimeError):
    """The required run log could not be written."""


def default_config() -> Dict[str, Any]:
    """Copy the CONFIGURATION constants, including edits made in a notebook session."""
    return {key: copy.deepcopy(globals()[key]) for key in CONFIG_KEYS}


def is_placeholder_path(path: object) -> bool:
    """Return True if *path* still points at a default placeholder location."""
    return "path\\to" in str(path).lower()


def file_label(text: str) -> str:
    """Return a Windows-safe file or folder label for *text*."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")


def normalized_clusters(clusters: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Any]:
    """Return CLUSTERS with every key present and every ID as a string."""
    return {
        str(name): {key: [str(stop) for stop in info.get(key, [])] for key in CLUSTER_KEYS}
        for name, info in clusters.items()
    }


def _string_list(value: object) -> bool:
    """Whether *value* is a list of nonempty strings without outer whitespace."""
    return isinstance(value, list) and all(
        isinstance(item, str) and item.strip() and item == item.strip() for item in value
    )


def validate_clusters(cfg: Mapping[str, Any]) -> None:
    """Check CLUSTERS and BAY_LABELS.

    Raises:
        ConfigError: Naming the first problem found.
    """
    clusters = cfg["CLUSTERS"]
    if not isinstance(clusters, dict) or not clusters:
        raise ConfigError("CLUSTERS must name at least one cluster.")
    owner: Dict[str, str] = {}
    labels_seen = set()
    for name, info in clusters.items():
        if not isinstance(name, str) or not file_label(name):
            raise ConfigError(f"Cluster names need letters or digits: {name!r}")
        label = file_label(name).lower()
        if label in labels_seen:
            raise ConfigError(f"Cluster names too similar for file names: {name!r}")
        labels_seen.add(label)
        if not isinstance(info, dict) or set(info) - set(CLUSTER_KEYS):
            raise ConfigError(
                f"Cluster {name!r} must be a dict with keys from {list(CLUSTER_KEYS)}; "
                f"got {sorted(info) if isinstance(info, dict) else info!r}"
            )
        for key in CLUSTER_KEYS:
            if not _string_list(info.get(key, [])):
                raise ConfigError(f"CLUSTERS[{name!r}][{key!r}] must be a list of strings.")
        stops = info.get("stops", [])
        if not stops:
            raise ConfigError(f"Cluster {name!r} has no stops.")
        if len(stops) != len(set(stops)):
            raise ConfigError(f"Cluster {name!r} lists a stop twice.")
        for stop in stops:
            if stop in owner:
                raise ConfigError(f"Stop {stop} is in both {owner[stop]!r} and {name!r}.")
            owner[stop] = name
        two, three = set(info.get("two_bay_stops", [])), set(info.get("three_bay_stops", []))
        if (two | three) - set(stops) or two & three:
            raise ConfigError(
                f"Cluster {name!r}: two_bay_stops and three_bay_stops must be distinct "
                "members of its stops."
            )
        overflow = info.get("overflow_bays", [])
        if len(overflow) != len(set(overflow)) or set(overflow) & set(stops):
            raise ConfigError(f"Cluster {name!r}: overflow_bays must be unique names, not stops.")

    labels = cfg["BAY_LABELS"]
    if not isinstance(labels, dict):
        raise ConfigError("BAY_LABELS must map stop_ids to bay labels.")
    unknown = sorted(set(labels) - set(owner))
    if unknown:
        raise ConfigError(f"BAY_LABELS names stops that are in no cluster: {unknown}")
    for name, info in clusters.items():
        bays = [labels.get(stop, stop) for stop in info["stops"]]
        if any(not isinstance(bay, str) or not bay.strip() or "," in bay for bay in bays):
            raise ConfigError(f"Cluster {name!r}: bay labels must be nonempty, without commas.")
        if len(bays) != len(set(bays)) or OVERFLOW_POOL in bays:
            raise ConfigError(f"Cluster {name!r}: bay labels must be unique and not a space name.")


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Reject invalid settings before any GTFS is read or any output is written.

    Raises:
        ConfigError: Naming the first setting that is invalid.
    """
    for key in ("GTFS_PATH", "OUTPUT_DIR", "RUN_LABEL", "SCHEDULE_LABEL"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ConfigError(f"{key} must be a nonempty string.")
    if not file_label(cfg["RUN_LABEL"]):
        raise ConfigError("RUN_LABEL needs letters or digits.")
    if not isinstance(cfg["SERVICE_DATE"], str) or not _string_list(cfg["SERVICE_IDS"]):
        raise ConfigError("SERVICE_DATE must be a string and SERVICE_IDS a list of strings.")
    if cfg["SERVICE_DATE"].strip() and cfg["SERVICE_IDS"]:
        raise ConfigError("Set SERVICE_DATE or SERVICE_IDS, not both.")
    if cfg["SERVICE_DATE"].strip():
        try:
            dt.datetime.strptime(cfg["SERVICE_DATE"].strip(), "%Y%m%d")
        except ValueError as exc:
            raise ConfigError(
                f"SERVICE_DATE must be YYYYMMDD; got {cfg['SERVICE_DATE']!r}"
            ) from exc
    switches = list(STEP_SWITCHES.values()) + ["REQUIRE_RUN_LOG"]
    for key in switches:
        if not isinstance(cfg[key], bool):
            raise ConfigError(f"{key} must be True or False.")
    if not any(cfg[key] for key in STEP_SWITCHES.values()):
        raise ConfigError("Every step is switched off; set at least one RUN_* switch to True.")
    if cfg["OPTIMIZE_MODE"] not in OPTIMIZE_MODES:
        raise ConfigError(f"OPTIMIZE_MODE must be one of {list(OPTIMIZE_MODES)}.")
    for key in ("ROUTES_FILENAME", "RUN_LOG_FILENAME"):
        name = cfg[key]
        if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name:
            raise ConfigError(f"{key} must be a file name without folders.")
    validate_clusters(cfg)

    for key, reserved in RESERVED_SETTINGS.items():
        settings = cfg[key]
        if not isinstance(settings, dict) or any(
            not isinstance(name, str) or not name.isupper() for name in settings
        ):
            raise ConfigError(f"{key} must map upper-case setting names to values.")
        taken = sorted(set(settings) & set(reserved))
        if taken:
            raise ConfigError(f"{key} may not set {taken}; this script sets them.")
    by_cluster = cfg["OPTIMIZER_SETTINGS_BY_CLUSTER"]
    if not isinstance(by_cluster, dict) or set(by_cluster) - set(cfg["CLUSTERS"]):
        raise ConfigError("OPTIMIZER_SETTINGS_BY_CLUSTER keys must be cluster names.")
    for name, settings in by_cluster.items():
        reserved = RESERVED_SETTINGS["OPTIMIZER_SETTINGS"]
        if not isinstance(settings, dict) or set(settings) & set(reserved):
            raise ConfigError(
                f"OPTIMIZER_SETTINGS_BY_CLUSTER[{name!r}] must be a dict without {list(reserved)}."
            )


def resolve_python(configured: str) -> str:
    """Return the Python interpreter that runs each step.

    ArcGIS Pro's Python window reports ArcGISPro.exe as ``sys.executable``, so
    the environment's own python.exe is used there instead.

    Raises:
        ConfigError: If no interpreter is found.
    """
    if configured.strip():
        if not Path(configured).is_file():
            raise ConfigError(f"PYTHON_EXECUTABLE does not exist: {configured}")
        return configured
    current = Path(sys.executable) if sys.executable else None
    if current is not None and current.name.lower().startswith("python"):
        return str(current)
    for candidate in (Path(sys.exec_prefix) / "python.exe", Path(sys.exec_prefix) / "bin/python3"):
        if candidate.is_file():
            return str(candidate)
    raise ConfigError(
        f"Could not find a Python interpreter for the steps (sys.executable is "
        f"{sys.executable!r}); set PYTHON_EXECUTABLE."
    )


def resolve_scripts(cfg: Mapping[str, Any]) -> Dict[str, Path]:
    """Return the path of each enabled step's script.

    Raises:
        ConfigError: If SCRIPTS_ROOT cannot be found or a script is missing.
    """
    if cfg["SCRIPTS_ROOT"].strip():
        root = Path(cfg["SCRIPTS_ROOT"])
    elif "__file__" in globals():
        root = Path(__file__).resolve().parent.parent
    else:
        raise ConfigError(
            "SCRIPTS_ROOT is empty and this code has no file location (a notebook cell); "
            "set SCRIPTS_ROOT to the repository's scripts folder."
        )
    scripts = {
        step: (root / folder / name).resolve()
        for step, (folder, name) in STEP_SCRIPTS.items()
        if cfg[STEP_SWITCHES[step]]
    }
    missing = [str(path) for path in scripts.values() if not path.is_file()]
    if missing:
        raise ConfigError(f"Step scripts not found (check SCRIPTS_ROOT): {', '.join(missing)}")
    return scripts


def feed_has_file(gtfs_path: str, file_name: str) -> bool:
    """Whether the GTFS folder or zip holds *file_name* (at the root or one folder down)."""
    if os.path.isdir(gtfs_path):
        return os.path.exists(os.path.join(gtfs_path, file_name))
    with zipfile.ZipFile(gtfs_path) as archive:
        return any(os.path.basename(name) == file_name for name in archive.namelist())


def load_feed(gtfs_path: str) -> Dict[str, pd.DataFrame]:
    """Load the GTFS tables this script reads; calendar tables only when present."""
    files = list(REQUIRED_GTFS_FILES)
    files += [name for name in OPTIONAL_GTFS_FILES if feed_has_file(gtfs_path, name)]
    return load_gtfs_data(gtfs_path, files=files, dtype=str)


def resolve_service(
    cfg: Mapping[str, Any], feed: Mapping[str, pd.DataFrame]
) -> Tuple[str, List[str]]:
    """Return a description of the service day and its service_ids that have trips.

    Raises:
        ConfigError: If the listed service_ids have no trips, or no trips run on
            the chosen day.
    """
    used = set(feed["trips"]["service_id"].dropna().astype(str))
    if cfg["SERVICE_IDS"]:
        unused = sorted(set(cfg["SERVICE_IDS"]) - used)
        if unused:
            raise ConfigError(f"SERVICE_IDS not used by any trip: {unused}")
        return "service_ids listed in SERVICE_IDS", sorted(set(cfg["SERVICE_IDS"]))

    active = expand_service_active_dates(feed.get("calendar"), feed.get("calendar_dates"))
    if not active:
        raise ConfigError("The feed has no calendar.txt or calendar_dates.txt; set SERVICE_IDS.")
    override = None
    if cfg["SERVICE_DATE"].strip():
        override = dt.datetime.strptime(cfg["SERVICE_DATE"].strip(), "%Y%m%d").date()
        if override.weekday() >= 5:
            logging.warning(
                "SERVICE_DATE %s is a %s; outputs are still labelled %r (SCHEDULE_LABEL).",
                override,
                override.strftime("%A"),
                cfg["SCHEDULE_LABEL"],
            )
    chosen, active_ids = representative_service_date(active, "weekday", override_date=override)
    ids = sorted(active_ids & used)
    if not ids:
        raise ConfigError(f"No trips run on {chosen}; choose another SERVICE_DATE.")
    return f"{chosen:%Y-%m-%d} ({chosen:%A})", ids


def routes_serving_clusters(
    feed: Mapping[str, pd.DataFrame],
    service_ids: Sequence[str],
    clusters: Mapping[str, Mapping[str, Sequence[str]]],
) -> pd.DataFrame:
    """List each route and direction that stops at each cluster on the service day.

    Every scheduled visit counts, including drop-off-only and pick-up-only ones.

    Returns:
        One row per cluster, route and direction: the cluster stops used, trips
        that visit the cluster, and the first and last visit (``HH:MM``).
    """
    columns = [
        "cluster",
        "route",
        "route_id",
        "route_short_name",
        "route_long_name",
        "direction_id",
        "cluster_stop_ids",
        "trips",
        "first_visit",
        "last_visit",
    ]
    stop_cluster = {stop: name for name, info in clusters.items() for stop in info["stops"]}
    trips = feed["trips"]
    trips = trips.loc[trips["service_id"].isin(service_ids)].copy()
    if "direction_id" not in trips.columns:
        trips["direction_id"] = ""
    stop_times = feed["stop_times"]
    visits = stop_times.loc[stop_times["stop_id"].isin(stop_cluster)].merge(
        trips[["trip_id", "route_id", "direction_id"]], on="trip_id"
    )
    if visits.empty:
        return pd.DataFrame(columns=columns)
    for name in ("arrival_time", "departure_time"):
        if name not in visits.columns:
            visits[name] = None
    departure = visits["departure_time"].map(parse_time_to_minutes)
    visits["minute"] = departure.where(
        departure.notna(), visits["arrival_time"].map(parse_time_to_minutes)
    )
    visits["cluster"] = visits["stop_id"].map(stop_cluster)
    visits["direction_id"] = visits["direction_id"].fillna("")
    table = (
        visits.groupby(["cluster", "route_id", "direction_id"])
        .agg(
            cluster_stop_ids=("stop_id", lambda s: ", ".join(sorted(set(s)))),
            trips=("trip_id", "nunique"),
            first=("minute", "min"),
            last=("minute", "max"),
        )
        .reset_index()
    )
    routes = feed["routes"].copy()
    for name in ("route_short_name", "route_long_name"):
        if name not in routes.columns:
            routes[name] = ""
    table = table.merge(
        routes[["route_id", "route_short_name", "route_long_name"]], on="route_id", how="left"
    )
    table[["route_short_name", "route_long_name"]] = table[
        ["route_short_name", "route_long_name"]
    ].fillna("")
    table["route"] = [
        route_display_name(short, long, rid)
        for short, long, rid in zip(
            table["route_short_name"], table["route_long_name"], table["route_id"]
        )
    ]
    table["first_visit"] = table["first"].map(minutes_to_hhmm)
    table["last_visit"] = table["last"].map(minutes_to_hhmm)
    order = {name: index for index, name in enumerate(clusters)}
    table["cluster_order"] = table["cluster"].map(order)
    table = table.sort_values(["cluster_order", "route", "route_id", "direction_id"])
    return table[columns].reset_index(drop=True)


def schedule_settings(
    cfg: Mapping[str, Any],
    gtfs_path: str,
    output_dir: Path,
    service_ids: Sequence[str],
    route_names: Sequence[str],
) -> Dict[str, Any]:
    """CONFIGURATION values for ``timepoint_schedule_exporter.py``."""
    settings: Dict[str, Any] = {
        "GTFS_FOLDER_PATH": gtfs_path,
        "BASE_OUTPUT_PATH": str(output_dir),
        "FILTER_SERVICE_IDS": list(service_ids),
        "FILTER_IN_ROUTES": list(route_names),
        "SERVICE_LABEL_OVERRIDES": {sid: cfg["SCHEDULE_LABEL"] for sid in service_ids},
    }
    settings.update(cfg["SCHEDULE_SETTINGS"])
    return settings


def timeline_settings(
    cfg: Mapping[str, Any], gtfs_path: str, output_dir: Path, service_ids: Sequence[str]
) -> Dict[str, Any]:
    """CONFIGURATION values for ``block_status_timeline_exporter.py``."""
    clusters = normalized_clusters(cfg["CLUSTERS"])
    settings: Dict[str, Any] = {
        "GTFS_FOLDER_PATH": gtfs_path,
        "BLOCK_OUTPUT_FOLDER": str(output_dir),
        "SCENARIO_NAME": "",
        "CALENDAR_SERVICE_IDS": list(service_ids),
        "SERVICE_DATE": "",
        "CLUSTER_DEFINITIONS": clusters,
        # Derived from CLUSTER_DEFINITIONS when the script loads, so it is set as well.
        "BUS_STOP_CLUSTERS_STEP1": [
            {"name": name, "stops": info["stops"]} for name, info in clusters.items()
        ],
        "ROUTE_SHORTNAME_FILTER": [],
        "STOP_ID_FILTER": [stop for info in clusters.values() for stop in info["stops"]],
        "STOP_CODE_FILTER": [],
    }
    settings.update(cfg["TIMELINE_SETTINGS"])
    return settings


def optimizer_settings(
    cfg: Mapping[str, Any],
    cluster: str,
    gtfs_path: str,
    output_dir: Path,
    service_ids: Sequence[str],
) -> Dict[str, Any]:
    """CONFIGURATION overrides for ``bay_assignment_optimizer.py`` for one cluster.

    Stops become bays (labelled by BAY_LABELS, else stop_id); two- and three-bay
    stops set CLUSTER_CAPACITY; overflow_bays become one layover space holding
    as many buses as there are names, used for both LAYOVER and LONG BREAK. The
    other clusters are passed as OTHER_CLUSTERS.
    """
    clusters = normalized_clusters(cfg["CLUSTERS"])
    info = clusters[cluster]
    bays = {stop: cfg["BAY_LABELS"].get(stop, stop) for stop in info["stops"]}
    capacity = {bays[stop]: 2 for stop in info["two_bay_stops"]}
    capacity.update({bays[stop]: 3 for stop in info["three_bay_stops"]})
    overflow = info["overflow_bays"]
    settings: Dict[str, Any] = {
        "GTFS_PATH": gtfs_path,
        "OUTPUT_DIR": str(output_dir),
        "SCENARIO_LABEL": file_label(cluster),
        "SERVICE_DATE": "",
        "SERVICE_IDS": list(service_ids),
        "CLUSTER_NAME": cluster,
        "CLUSTER_STOPS": bays,
        "CLUSTER_CAPACITY": capacity,
        "OVERFLOW_ROUTING": {OVERFLOW_POOL: ["LAYOVER", "LONG BREAK"]} if overflow else {},
        "OVERFLOW_CAPACITY": {OVERFLOW_POOL: len(overflow)} if overflow else {},
        "OTHER_CLUSTERS": [
            {"name": name, "stops": other["stops"]}
            for name, other in clusters.items()
            if name != cluster
        ],
        "OPTIMIZE_MODE": cfg["OPTIMIZE_MODE"],
    }
    settings.update(cfg["OPTIMIZER_SETTINGS"])
    settings.update(cfg["OPTIMIZER_SETTINGS_BY_CLUSTER"].get(cluster, {}))
    return settings


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write *value* as indented UTF-8 JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_step(label: str, command: List[str], log_path: Path, cwd: Path) -> int:
    """Run one step in its own Python process, echoing and saving its console output.

    Returns:
        The process exit code (1 if it could not be started).
    """
    logging.info("Starting %s.", label)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logging.error("Could not start %s: %s", label, exc)
            log_file.write(f"Could not start: {exc}\n")
            return 1
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    log_file.write(line)
                    logging.info("  [%s] %s", label, CHILD_TIMESTAMP.sub("", line.rstrip()))
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    level = logging.INFO if code == 0 else logging.ERROR
    logging.log(
        level, "%s finished with exit code %d in %.0f s.", label, code, time.monotonic() - started
    )
    return code


def write_run_log(
    run_dir: Path,
    cfg: Mapping[str, Any],
    details: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> bool:
    """Write the run-log sidecar: verbatim CONFIGURATION, effective settings and steps.

    Returns:
        ``True`` if the log was written, ``False`` otherwise.
    """
    source = Path(__file__).resolve() if "__file__" in globals() else None
    if source is None:
        config_text = "(config block unavailable: interactive session, no __file__ on disk)"
    else:
        try:
            config_text = extract_config_block(source)
        except (OSError, ValueError) as exc:
            logging.error("Could not extract config block for run log: %s", exc)
            return False
    step_lines = [
        f"- {step['step']}: {step['result']}\n    command: {step['command']}\n"
        f"    console log: {step['log']}"
        for step in steps
    ] or ["(no step has run yet)"]
    lines = [
        "=" * 72,
        "FACILITY ANALYSIS ORCHESTRATOR RUN LOG",
        "=" * 72,
        f"Run timestamp:   {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Run folder:      {run_dir}",
        f"Source script:   {source if source is not None else '<interactive>'}",
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "",
        "EFFECTIVE SETTINGS",
        json.dumps(dict(cfg), indent=2, default=str, ensure_ascii=False),
        "",
        "RESOLVED",
        json.dumps(dict(details), indent=2, default=str, ensure_ascii=False),
        "",
        "STEPS",
        *step_lines,
        "=" * 72,
    ]
    try:
        (run_dir / cfg["RUN_LOG_FILENAME"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False
    return True


def require_run_log(written: bool, cfg: Mapping[str, Any]) -> None:
    """Raise :class:`RunLogError` when a run log failed and REQUIRE_RUN_LOG is set."""
    if not written and cfg["REQUIRE_RUN_LOG"]:
        raise RunLogError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )


def run(cfg: Dict[str, Any]) -> int:
    """Resolve the service day and routes, then run each enabled step.

    Returns:
        0 if every step that ran succeeded, otherwise 1.

    Raises:
        ConfigError: If a setting is invalid or does not match the feed.
        RunLogError: If the required run log cannot be written.
    """
    validate_config(cfg)
    gtfs_path = str(Path(cfg["GTFS_PATH"]).resolve())
    if not os.path.isdir(gtfs_path) and not (
        os.path.isfile(gtfs_path) and zipfile.is_zipfile(gtfs_path)
    ):
        raise ConfigError(f"GTFS_PATH must be a GTFS folder or .zip: {gtfs_path}")
    python = resolve_python(cfg["PYTHON_EXECUTABLE"])
    scripts = resolve_scripts(cfg)
    clusters = normalized_clusters(cfg["CLUSTERS"])

    # Read the feed once to resolve the day and the routes, then release it.
    feed = load_feed(gtfs_path)
    all_stops = [stop for info in clusters.values() for stop in info["stops"]]
    absent = sorted(set(all_stops) - set(feed["stops"]["stop_id"].astype(str)))
    if absent:
        raise ConfigError(f"Cluster stops are absent from stops.txt: {absent}")
    service, service_ids = resolve_service(cfg, feed)
    logging.info("Service day: %s; service_ids: %s", service, ", ".join(service_ids))
    routes = routes_serving_clusters(feed, service_ids, clusters)
    calendar = feed.get("calendar")
    in_calendar = set(calendar["service_id"].astype(str)) if calendar is not None else set()
    del feed

    run_dir = Path(cfg["OUTPUT_DIR"]).resolve() / (
        f"{file_label(cfg['RUN_LABEL'])}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    )
    settings_dir = run_dir / SETTINGS_DIRNAME
    settings_dir.mkdir(parents=True)
    routes.to_csv(run_dir / cfg["ROUTES_FILENAME"], index=False, encoding="utf-8-sig")
    logging.info("Routes serving the clusters written to %s", run_dir / cfg["ROUTES_FILENAME"])

    served = set(routes["cluster"])
    for name in clusters:
        count = int((routes["cluster"] == name).sum())
        if name in served:
            logging.info("%s: %d route-direction(s) stop here.", name, count)
        else:
            logging.warning("%s: no trip stops at its stops on this service day.", name)
    unnamed = sorted(set(routes.loc[routes["route_short_name"] == "", "route_id"]))
    if unnamed and cfg["RUN_SCHEDULES"]:
        logging.warning(
            "Routes without a route_short_name get no schedule (the exporter selects routes "
            "by short name): %s",
            ", ".join(unnamed),
        )
    route_names = sorted(set(routes["route_short_name"]) - {""})
    schedule_ids = [sid for sid in service_ids if sid in in_calendar]
    if cfg["RUN_SCHEDULES"] and len(schedule_ids) < len(service_ids):
        logging.warning(
            "Not in calendar.txt, so not in the schedules: service_id(s) %s.",
            ", ".join(sorted(set(service_ids) - set(schedule_ids))),
        )

    details: Dict[str, Any] = {
        "gtfs_path": gtfs_path,
        "service_day": service,
        "service_ids": service_ids,
        "schedule_routes": route_names,
        "python": python,
        "scripts": {step: str(path) for step, path in scripts.items()},
    }
    steps: List[Dict[str, Any]] = []
    require_run_log(write_run_log(run_dir, cfg, details, steps), cfg)

    def launch(step: str, label: str, settings: Dict[str, Any], skip: str = "") -> None:
        """Run (or skip) one step and record its result."""
        settings_path = settings_dir / f"{label}.json"
        log_path = settings_dir / f"{label}.log"
        write_json(settings_path, settings)
        if step == "optimizer":
            command = [python, str(scripts[step]), "--config", str(settings_path)]
        else:
            command = [python, "-c", STEP_BOOTSTRAP, str(scripts[step]), str(settings_path)]
        shown = " ".join(command).replace(STEP_BOOTSTRAP, "<bootstrap>")
        if skip:
            logging.warning("Skipping %s: %s", label, skip)
            steps.append({"step": label, "result": f"skipped: {skip}", "command": "", "log": ""})
            return
        code = run_step(label, command, log_path, run_dir)
        result = "succeeded" if code == 0 else f"FAILED (exit code {code})"
        steps.append({"step": label, "result": result, "command": shown, "log": str(log_path)})

    if cfg["RUN_SCHEDULES"]:
        skip = ""
        if not route_names:
            skip = "no route with a route_short_name stops at the clusters"
        elif not schedule_ids:
            skip = "none of the service_ids is in calendar.txt"
        settings = schedule_settings(
            cfg, gtfs_path, run_dir / SCHEDULE_DIRNAME, schedule_ids, route_names
        )
        launch("schedules", "1_schedules", settings, skip)
    if cfg["RUN_TIMELINES"]:
        skip = "" if served else "no trip stops at any cluster on this service day"
        settings = timeline_settings(cfg, gtfs_path, run_dir / TIMELINE_DIRNAME, service_ids)
        launch("timelines", "2_timelines", settings, skip)
    if cfg["RUN_OPTIMIZER"]:
        for name in clusters:
            skip = "" if name in served else "no trip stops at this cluster on this service day"
            settings = optimizer_settings(
                cfg, name, gtfs_path, run_dir / OPTIMIZER_DIRNAME, service_ids
            )
            launch("optimizer", f"3_optimizer_{file_label(name)}", settings, skip)

    require_run_log(write_run_log(run_dir, cfg, details, steps), cfg)
    logging.info("Run log saved to %s", run_dir / cfg["RUN_LOG_FILENAME"])
    failed = [step["step"] for step in steps if step["result"].startswith("FAILED")]
    if failed:
        logging.error(
            "%d step(s) failed: %s. See their .log files in %s",
            len(failed),
            ", ".join(failed),
            settings_dir,
        )
        return 1
    logging.info("All steps finished. Outputs are in %s", run_dir)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser; every default is the CONFIGURATION value."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gtfs-path", default=GTFS_PATH, help="GTFS folder or .zip.")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for run subfolders.")
    p.add_argument("--run-label", default=RUN_LABEL, help="Names the run folder.")
    p.add_argument("--service-date", default=SERVICE_DATE, help="Service day as YYYYMMDD.")
    p.add_argument(
        "--service-ids",
        nargs="*",
        default=SERVICE_IDS,
        help="service_ids operating together (instead of a service date).",
    )
    for flag, default in (
        ("--run-schedules", RUN_SCHEDULES),
        ("--run-timelines", RUN_TIMELINES),
        ("--run-optimizer", RUN_OPTIMIZER),
    ):
        p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
    p.add_argument("--optimize", choices=OPTIMIZE_MODES, default=OPTIMIZE_MODE)
    p.add_argument("--scripts-root", default=SCRIPTS_ROOT, help="Repository scripts folder.")
    p.add_argument(
        "--python-executable", default=PYTHON_EXECUTABLE, help="Python that runs the steps."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help='JSON file of further CONFIGURATION overrides, e.g. {"CLUSTERS": {...}}. '
        "Flags given on the command line take precedence over it.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run with CONFIGURATION, optional JSON overrides and command-line flags.

    Returns:
        Process exit code: 0 when every step that ran succeeded, 1 if a step,
        the input or the run log failed, 2 if a setting is invalid or a
        required path is still a placeholder.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    # A sentinel marks each flag that was not given, so the CONFIGURATION value and
    # any --config override stand unless a flag replaces them.
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
                raise ConfigError("--config must contain a JSON object of setting overrides.")
            unknown = sorted(set(overrides) - set(CONFIG_KEYS))
            if unknown:
                raise ConfigError(f"--config names unknown settings: {unknown}")
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
        return 2
    unset = [key for key in ("GTFS_PATH", "OUTPUT_DIR") if is_placeholder_path(cfg[key])]
    if unset:
        logging.warning(
            "Default placeholder paths detected for: %s. Update the CONFIGURATION section "
            "or pass --gtfs-path / --output-dir before running.",
            ", ".join(unset),
        )
        return 2
    try:
        code = run(cfg)
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2
    except (RunLogError, OSError, ValueError, KeyError) as exc:
        logging.error("%s", exc)
        return 1
    if code == 0:
        logging.info("Script completed successfully.")
    return code


# =============================================================================
# REUSABLE FUNCTIONS
# =============================================================================


# Canonical versions live in utils/cli_helpers.py -- keep these copies in sync.
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


# Canonical versions live in utils/run_log.py -- keep these copies in sync.
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


# Canonical versions live in utils/gtfs_helpers.py -- keep these copies in sync.
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


# Canonical versions live in utils/calendar_helpers.py -- keep these copies in sync.
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


def representative_service_date(
    active_dates: Mapping[str, set[dt.date]],
    service_day: str,
    override_date: Optional[dt.date] = None,
    exclude_dates: Optional[set[dt.date]] = None,
) -> tuple[dt.date, set[str]]:
    """Pick a typical date for *service_day* and the service_ids active on it.

    Rather than trusting any single date or unioning every service whose
    columns mention a weekday (which double-counts agencies running distinct
    Monday / midweek / Friday schedules), this scans every candidate date of
    the requested day type, groups them by their exact set of active
    service_ids, and returns the median date of the **modal** (most common)
    set. A few miscoded dates therefore cannot steer the result, and per-day
    math (headways, spans, trip counts) reflects one real operating day.

    Warnings are logged when the choice is ambiguous: when the modal set
    covers under half the candidate dates, and — for ``"weekday"`` — when
    Monday-through-Friday do not all share one service pattern, so the user
    knows a single representative day cannot speak for the whole week.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        service_day: ``"weekday"`` or one of ``"monday"`` … ``"sunday"``.
        override_date: Skip selection entirely and use this date (the
            explicit user override). Logged, and a warning is emitted if no
            service is active on it.
        exclude_dates: Dates to skip as candidates — typically observed
            holidays, so a holiday cannot masquerade as a typical day.

    Returns:
        Tuple of (chosen date, set of service_id strings active on it).

    Raises:
        ValueError: If *service_day* is not recognised, or no candidate
            dates exist for it.
    """
    day_names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    key = service_day.strip().lower()
    if key == "weekday":
        allowed = {0, 1, 2, 3, 4}
    elif key in day_names:
        allowed = {day_names.index(key)}
    else:
        raise ValueError(
            f"service_day must be 'weekday' or one of {', '.join(day_names)}; got {service_day!r}"
        )

    if override_date is not None:
        ids = service_ids_active_on(active_dates, override_date)
        if not ids:
            logging.warning("No service is active on override date %s.", override_date)
        else:
            logging.info(
                "Using override date %s (%s): %d service_id(s).",
                override_date,
                day_names[override_date.weekday()],
                len(ids),
            )
        return override_date, ids

    skip = exclude_dates or set()
    candidates = sorted(
        {d for dates in active_dates.values() for d in dates if d.weekday() in allowed} - skip
    )
    if not candidates:
        raise ValueError(f"No active dates found for service_day={service_day!r}.")

    by_set: dict[frozenset[str], list[dt.date]] = {}
    for d in candidates:
        by_set.setdefault(frozenset(service_ids_active_on(active_dates, d)), []).append(d)

    modal_ids = max(by_set, key=lambda ids: (len(by_set[ids]), -min(by_set[ids]).toordinal()))
    modal_dates = by_set[modal_ids]
    chosen = modal_dates[len(modal_dates) // 2]
    share = len(modal_dates) / len(candidates)

    if key == "weekday":
        per_day: dict[str, frozenset[str]] = {}
        for dow in sorted({d.weekday() for d in candidates}):
            day_candidates = [d for d in candidates if d.weekday() == dow]
            day_sets: dict[frozenset[str], int] = {}
            for d in day_candidates:
                s = frozenset(service_ids_active_on(active_dates, d))
                day_sets[s] = day_sets.get(s, 0) + 1
            per_day[day_names[dow]] = max(day_sets, key=lambda s: day_sets[s])
        if len(set(per_day.values())) > 1:
            detail = "; ".join(
                f"{day}={sorted(ids) if ids else '{}'}" for day, ids in per_day.items()
            )
            logging.warning(
                "Weekday service varies by day of week (%s). Using %s (%s) as the "
                "representative weekday; pass an explicit service date to analyse "
                "a different day.",
                detail,
                chosen,
                day_names[chosen.weekday()],
            )
    if share < 0.5:
        logging.warning(
            "The chosen service pattern covers only %.0f%% of candidate %s dates — "
            "this feed's %s service is irregular; consider an explicit service date.",
            share * 100,
            service_day,
            service_day,
        )

    logging.info(
        "Representative %s: %s (%s) with %d service_id(s), matching %.0f%% of candidate dates.",
        service_day,
        chosen,
        day_names[chosen.weekday()],
        len(modal_ids),
        share * 100,
    )
    return chosen, set(modal_ids)


# Canonical versions live in utils/time_helpers.py -- keep these copies in sync.
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


# Canonical versions live in utils/block_timeline_helpers.py -- keep these copies in sync.
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


if __name__ == "__main__":
    raise SystemExit(main())
