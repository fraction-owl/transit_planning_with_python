"""Generates minute-by-minute transit block status timelines from GTFS data.

Processes GTFS files to determine each vehicle block's operational status (e.g.
DWELL, LAYOVER, LOADING, TRAVELING BETWEEN STOPS, LONG BREAK, INACTIVE) at
one-minute intervals across the service day, producing one spreadsheet per
block for further analysis (bay conflict checks, layover planning, cluster
staffing). Named stop clusters (``CLUSTER_DEFINITIONS``) group related stops --
for example the bays of a transit center -- so statuses reflect time spent
anywhere in the cluster.

Inputs
------
- A GTFS folder (or ``.zip``) containing trips.txt, stop_times.txt,
  routes.txt, stops.txt, calendar.txt and calendar_dates.txt. The service day
  to analyze is selected via ``CALENDAR_SERVICE_IDS`` or ``SERVICE_DATE``;
  optional route and stop filters narrow the run.

Occupancy assumptions
---------------------
GTFS gives a single minute for most stop visits. The ``*_MINUTES`` settings in
CONFIGURATION stretch that into the time a bus actually occupies a stop
(through-stop dwell, sitting at the departure bay before a trip, staying at
the arrival bay after one) and decide whether a between-trip gap is spent in
the arrival bay (``DWELL``) or in the cluster's overflow/layover space
(``LAYOVER`` / ``LONG BREAK``). ``BAY_OVERRIDES`` re-assigns a route's visits
to a different stop so alternative bay assignments can be tested without
editing the feed. The assumptions used are written to ``ASSUMPTIONS_FILE``
alongside the output so downstream reports can echo them.

Outputs
-------
- One Excel workbook per vehicle block (``block_<block_id>_<route(s)>.xlsx``)
  in ``BLOCK_OUTPUT_FOLDER``: one row per minute with timestamp, route,
  direction, trip, stop, arrival/departure times, and status.
- Optionally a single ``COMBINED_TIMELINE_FILE`` (CSV) holding every block's
  rows, which Step 2 can read directly instead of re-opening each workbook.
- ``ASSUMPTIONS_FILE`` recording the parameters used for the run.
- ``RUN_LOG_FILENAME``, a run-log sidecar in the same folder capturing the
  source CONFIGURATION block and the effective runtime settings.
- ``SCHEDULE_SNAPSHOT_FILE`` with scheduled visits and occupancy settings for
  exact bay-change rescoring, plus ``timeline_manifest.json`` identifying the
  completed run. A failed rerun invalidates the run for downstream readers;
  old files are retained but cannot be mistaken for current output.

Typical usage
-------------
Update the paths in the CONFIGURATION section and run from a shell, ArcGIS
Pro's Python window, or a Jupyter notebook.

Run this revised exporter once per scenario before using the revised sweep.
The pipeline requires one-minute sampling. The timeline automatically extends
past DEFAULT_HOURS when a scheduled trip or its post-arrival buffer ends later.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
# === BEGIN CONFIG ===

GTFS_FOLDER_PATH = r"your_GTFS_folder_path\here"
BLOCK_OUTPUT_FOLDER = r"your_output_folder_path\here"

# Optional: each run is written to BLOCK_OUTPUT_FOLDER\<SCENARIO_NAME>. Name it
# for the assumptions and bay assignments used, e.g. "direct_conflict",
# "likely_conflict_alt_bays". Step 2 (bay_usage_analyzer.py) reads the same
# subfolder. Leave "" to write to BLOCK_OUTPUT_FOLDER itself.
SCENARIO_NAME = ""

DEFAULT_HOURS = 26
TIME_INTERVAL_MINUTES = 1  # must remain 1 for the downstream minute-based reports

# Occupancy assumptions: how long a bus counts as occupying a stop beyond what
# the GTFS times say, given as THROUGH_DWELL / PRE_DEPARTURE / POST_ARRIVAL /
# IN_BAY_LAYOVER_MAX minutes. Presets that have proved useful:
#   "direct conflict" (same scheduled minute only) .......... 1 / 0 / 0 / 10
#   "likely conflict" (typical observed dwell; the default) . 2 / 5 / 2 / 10
#   "schedule only"   (roughly what earlier versions did) ... 1 / 1 / 0 / 3
# With 0 for the pre/post values a departure or arrival occupies only its
# scheduled minute; each extra minute adds one to what a bus is charged.
# Mid-trip stop (arrival == departure): occupied for this many minutes from the stop time.
THROUGH_DWELL_MINUTES = 2
# Bus sits at its departure stop (LOADING) this many minutes before a trip's first time.
PRE_DEPARTURE_MINUTES = 5
# Bus stays at its arrival stop (ARRIVE) this many minutes after a trip's last time.
POST_ARRIVAL_MINUTES = 2
# Between-trip gap up to this stays in the arrival bay (DWELL); longer gaps move to the
# overflow/layover space (LAYOVER, or LONG BREAK once the gap exceeds LAYOVER_THRESHOLD).
IN_BAY_LAYOVER_MAX_MINUTES = 10
LAYOVER_THRESHOLD = 20  # minutes; overflow gaps longer than this are labelled LONG BREAK
MAX_TRIPS_PER_BLOCK = 150

# Service day. Either list service_ids explicitly, or leave the list empty and
# give a date (YYYYMMDD): every service_id active on that date per calendar.txt
# and calendar_dates.txt is used. Leave both empty to process every trip in
# the feed regardless of service. service_id values are agency-specific —
# REPLACE WITH YOUR VALUES (the placeholder assumes a regular-weekday service),
# or set SERVICE_DATE instead and let the calendar resolve the ids for you.
CALENDAR_SERVICE_IDS: list[str] = ["4"]
SERVICE_DATE = ""

# Scenario testing: move a route's visits from one stop (bay) to another before
# the timeline is built. Each entry matches rows on route_short_name and any of
# from_stop_ids (and direction_id, if given) and rewrites stop_id/stop_name/
# stop_code to to_stop_id. Applied before the stop filters below. Example:
# move route 101's visits from bays 2956 and 2955 to bay 65:
#   {"route_short_name": "101", "from_stop_ids": ["2956", "2955"], "to_stop_id": "65"}
BAY_OVERRIDES: list[dict[str, Any]] = []

# Only blocks that touch these routes / stops are processed (much faster than
# the whole feed when only one facility matters). Leave all three empty to
# process every block.
ROUTE_SHORTNAME_FILTER: list[str] = []
STOP_ID_FILTER: list[str] = []
STOP_CODE_FILTER: list[str] = []

WRITE_PER_BLOCK_FILES = True
COMBINED_TIMELINE_FILE = r"all_blocks_timeline.csv"  # in the run folder; "" to skip
SCHEDULE_SNAPSHOT_FILE = r"schedule_snapshot.json"  # required by the exact sweep
ASSUMPTIONS_FILE = r"timeline_assumptions.txt"  # in the run folder; "" to skip
RUN_LOG_FILENAME = r"block_status_timeline_exporter_runlog.txt"  # in the run folder

CLUSTER_DEFINITIONS = {
    "Metro": {
        "stops": ["2956", "2955", "65", "3295", "2957", "66", "3296", "64", "63"],
        "overflow_bays": ["OverflowA", "OverflowB", "OverflowC"],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
    "Park & Ride": {
        "stops": ["3882", "3881", "1880"],
        "overflow_bays": [],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
    "Metro North": {
        "stops": ["2832", "2373"],
        "overflow_bays": [],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
}

BUS_STOP_CLUSTERS_STEP1 = [
    {"name": name, "stops": info["stops"]} for name, info in CLUSTER_DEFINITIONS.items()
]

# Every output must be traceable: a failed run-log write aborts the script.
# Set to False only when writing to a genuinely read-only location.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================


def occupancy_settings() -> dict[str, int]:
    """Return the effective occupancy settings used by this run."""
    names = (
        "THROUGH_DWELL_MINUTES",
        "PRE_DEPARTURE_MINUTES",
        "POST_ARRIVAL_MINUTES",
        "IN_BAY_LAYOVER_MAX_MINUTES",
        "LAYOVER_THRESHOLD",
    )
    return {name: globals()[name] for name in names}


def validate_configuration() -> None:
    """Reject unsupported intervals, invalid durations and ambiguous output names."""
    if TIME_INTERVAL_MINUTES != 1:
        raise ValueError("The downstream bay reports require TIME_INTERVAL_MINUTES = 1.")
    for name, value in occupancy_settings().items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if THROUGH_DWELL_MINUTES < 1 or LAYOVER_THRESHOLD < IN_BAY_LAYOVER_MAX_MINUTES:
        raise ValueError("Through dwell must be positive and layover thresholds must be ordered.")
    if not isinstance(DEFAULT_HOURS, int) or DEFAULT_HOURS < 1 or MAX_TRIPS_PER_BLOCK < 1:
        raise ValueError("DEFAULT_HOURS and MAX_TRIPS_PER_BLOCK must be positive integers.")
    if not COMBINED_TIMELINE_FILE and not WRITE_PER_BLOCK_FILES:
        raise ValueError("Enable the combined CSV or per-block workbooks.")
    filenames = [
        value
        for value in (
            COMBINED_TIMELINE_FILE,
            ASSUMPTIONS_FILE,
            SCHEDULE_SNAPSHOT_FILE,
            RUN_LOG_FILENAME,
            "timeline_manifest.json",
        )
        if value
    ]
    if not SCHEDULE_SNAPSHOT_FILE or len({name.casefold() for name in filenames}) != len(filenames):
        raise ValueError("Output filenames must be nonempty where required and distinct.")
    for name in filenames:
        if Path(name).name != name or "\\" in name:
            raise ValueError("Output filename settings must be basenames inside the run folder.")
    stops = [str(stop) for info in CLUSTER_DEFINITIONS.values() for stop in info["stops"]]
    if len(stops) != len(set(stops)):
        raise ValueError("A stop may belong to only one cluster.")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Publish JSON atomically so consumers never read a partial manifest or snapshot."""
    temporary = path.with_name(path.name + ".tmp")
    # Pandas normalizes NumPy scalar types to JSON-compatible built-ins.
    payload = json.loads(pd.Series([value]).to_json(orient="values"))[0]
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_output_folder() -> str:
    """Folder this run writes to: BLOCK_OUTPUT_FOLDER, or a SCENARIO_NAME subfolder of it."""
    return (
        os.path.join(BLOCK_OUTPUT_FOLDER, SCENARIO_NAME) if SCENARIO_NAME else BLOCK_OUTPUT_FOLDER
    )


def resolve_service_ids(
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame],
    service_date: str,
) -> list[str]:
    """Return every service_id active on ``service_date`` (``YYYYMMDD``).

    Expands calendar.txt and calendar_dates.txt into each service's real set
    of active dates (see :func:`expand_service_active_dates`) and keeps the
    services that operate on the requested day.

    Args:
        calendar_df: Parsed ``calendar.txt``, or ``None`` if the feed has none.
        calendar_dates_df: Parsed ``calendar_dates.txt``, or ``None``.
        service_date: The day to analyze, as ``YYYYMMDD``.

    Returns:
        Sorted list of service_id strings (possibly empty).

    Raises:
        ValueError: If *service_date* is not a valid ``YYYYMMDD`` date.
    """
    try:
        target = dt.datetime.strptime(service_date.strip(), "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"SERVICE_DATE must be a YYYYMMDD date; got {service_date!r}") from exc
    active = expand_service_active_dates(calendar_df, calendar_dates_df, today=target)
    return sorted(service_ids_active_on(active, target))


def validate_folders(input_path: str, output_path: str) -> None:
    """Check that the input folder exists, and ensure the output folder is created if not."""
    if not os.path.isdir(input_path) and not (
        os.path.isfile(input_path) and zipfile.is_zipfile(input_path)
    ):
        raise ValueError(f"Input must be a GTFS directory or ZIP archive: {input_path}")
    os.makedirs(output_path, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------


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
    return hours * 60 + minutes + (seconds + 30) // 60


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


def mark_first_and_last_stops(df_in: pd.DataFrame) -> pd.DataFrame:
    """Mark each stop in the trip as the first or last using boolean columns."""
    df_out = df_in.sort_values(["trip_id", "stop_sequence"]).copy()
    seq_min = df_out.groupby("trip_id")["stop_sequence"].transform("min")
    seq_max = df_out.groupby("trip_id")["stop_sequence"].transform("max")
    df_out["is_first_stop"] = df_out["stop_sequence"] == seq_min
    df_out["is_last_stop"] = df_out["stop_sequence"] == seq_max
    return df_out


def find_cluster(stop_id: str, bus_stop_clusters: list[dict[str, Any]]) -> Optional[str]:
    """Given a stop_id, return the named cluster (if any) or None if not found."""
    for cluster_item in bus_stop_clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


# --------------------------------------------------------------------------------------------------
# BRIDGING LOGIC REFACTOR
# --------------------------------------------------------------------------------------------------


def _status_for_same_trip(
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


def _gap_status(gap: int, same_place: bool, settings: dict[str, int]) -> tuple[str, str]:
    """Classify a between-trip gap using this scenario's occupancy assumptions."""
    if not same_place:
        return "DEADHEAD", ""
    if gap <= settings["IN_BAY_LAYOVER_MAX_MINUTES"]:
        return "DWELL", "in bay"
    if gap <= settings["LAYOVER_THRESHOLD"]:
        return "LAYOVER", "overflow"
    return "LONG BREAK", "overflow"


# --------------------------------------------------------------------------------------------------
# MAIN BLOCK PROCESSING
# --------------------------------------------------------------------------------------------------


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


def _create_trips_summary(
    block_subset: pd.DataFrame,
) -> list[dict[str, Any]]:  # Added return type annotation
    """Helper to build trip summaries for a block.

    Returns a list of dictionaries with trip info and sorted stop times.
    """
    trips_summary = []
    for trip_id, trip_df in block_subset.groupby("trip_id"):
        trip_df_sorted = trip_df.sort_values("stop_sequence")
        start_time = int(trip_df_sorted["arrival_min"].iloc[0])
        end_time = int(trip_df_sorted["departure_min"].iloc[-1])

        stop_times_sequence = []
        for _, row in trip_df_sorted.iterrows():
            t_val = row.get("timepoint", 0)
            stop_times_sequence.append(
                (
                    row["arrival_min"],
                    row["departure_min"],
                    row["stop_id"],
                    row["stop_name"],
                    row["trip_id"],
                    row["is_first_stop"],
                    row["is_last_stop"],
                    row["stop_sequence"],
                    t_val,
                )
            )

        first_row = trip_df_sorted.iloc[0]
        last_row = trip_df_sorted.iloc[-1]
        trips_summary.append(
            {
                "trip_id": trip_id,
                "start": start_time,
                "end": end_time,
                "stop_times_sequence": stop_times_sequence,
                "route_id": first_row["route_id"],
                "route_short_name": first_row.get("route_short_name", "") or "",
                "trip_headsign": first_row.get("trip_headsign", "") or "",
                "direction_id": first_row["direction_id"],
                "block": str(first_row["block_id"]),
                "first_stop_id": first_row["stop_id"],
                "first_stop_name": first_row["stop_name"],
                "first_stop_seq": first_row["stop_sequence"],
                "last_stop_id": last_row["stop_id"],
                "last_stop_name": last_row["stop_name"],
                "last_stop_seq": last_row["stop_sequence"],
            }
        )

    trips_summary.sort(key=lambda x: x["start"])
    return trips_summary


def _make_row(
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


def _row_for_inactive(
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
        return _make_row(
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
        return _make_row(
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
        status, location = _gap_status(nxt["start"] - prev["end"], same_place, settings)
        if status != "DEADHEAD":
            return _make_row(
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
        return _make_row(minute, block_id, status, prev_trip_id=prev_id, next_trip_id=next_id)
    return _make_row(minute, block_id, "INACTIVE", prev_trip_id=prev_id, next_trip_id=next_id)


def _build_schedule_rows(
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
                status = _status_for_same_trip(minute, stop, settings)
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
                _make_row(
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
                    _make_row(
                        minute, block_id, "TRAVELING BETWEEN STOPS", trip, trip_id=trip["trip_id"]
                    )
                )
        else:
            row = _row_for_inactive(minute, block_id, trips_summary, bus_stop_clusters, settings)
            if not occupancy_only or row["Status"] not in {"INACTIVE", "DEADHEAD"}:
                rows.append(row)
    return rows


def process_block(
    block_subset: pd.DataFrame,
    block_id: str,
    timeline: range,
    bus_stop_clusters: list[dict[str, Any]],
) -> pd.DataFrame:
    """Generate a minute-by-minute schedule DataFrame for a single block."""
    trips_summary = _create_trips_summary(block_subset)
    rows = _build_schedule_rows(
        trips_summary, timeline, block_id, bus_stop_clusters, occupancy_settings()
    )
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------------------------------
# STEP 1: GTFS -> Block Spreadsheets
# --------------------------------------------------------------------------------------------------


def apply_bay_overrides(merged_df: pd.DataFrame, stops_df: pd.DataFrame) -> pd.DataFrame:
    """Re-assign stop visits according to ``BAY_OVERRIDES``.

    Each override matches rows whose ``route_short_name`` and ``stop_id`` (and
    ``direction_id``, if the override gives one) agree, and rewrites
    ``stop_id``, ``stop_name`` and ``stop_code`` to the target stop. Rows that
    match no override are untouched.

    Args:
        merged_df: stop_times joined to trips and stops.
        stops_df: stops.txt, used to look up the target stop's name and code.

    Returns:
        A copy of ``merged_df`` with the overrides applied.
    """
    if not BAY_OVERRIDES:
        return merged_df

    df_out = merged_df.copy()
    lookup = stops_df.drop_duplicates("stop_id").set_index("stop_id")
    for override in BAY_OVERRIDES:
        route = str(override["route_short_name"])
        from_ids = [str(s) for s in override["from_stop_ids"]]
        to_id = str(override["to_stop_id"])

        mask = (df_out["route_short_name"].astype(str) == route) & df_out["stop_id"].astype(
            str
        ).isin(from_ids)
        if override.get("direction_id") is not None:
            mask &= df_out["direction_id"].astype(str) == str(override["direction_id"])

        if to_id not in lookup.index:
            raise ValueError(f"Bay override target {to_id!r} is absent from stops.txt.")
        n_rows = int(mask.sum())
        df_out.loc[mask, "stop_id"] = to_id
        if to_id in lookup.index:
            df_out.loc[mask, "stop_name"] = lookup.loc[to_id, "stop_name"]
            if "stop_code" in lookup.columns:
                df_out.loc[mask, "stop_code"] = lookup.loc[to_id, "stop_code"]
        else:
            logging.warning("Bay override target stop_id %s not found in stops.txt.", to_id)
        logging.info(
            "Bay override: route %s stops %s -> %s (%d stop_times rows).",
            route,
            from_ids,
            to_id,
            n_rows,
        )
    return df_out


def _merge_and_filter_data(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    service_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Merge trips and stops, filter by service_id, route, etc.

    Return a single merged DataFrame with arrival_min/departure_min.
    """
    required = {"trip_id", "route_id", "service_id", "block_id"}
    if required - set(trips_df.columns):
        raise ValueError(
            f"trips.txt is missing required fields: {sorted(required - set(trips_df.columns))}"
        )
    trips_df = trips_df.copy()
    stop_times_df = stop_times_df.copy()
    stops_df = stops_df.copy()
    if "direction_id" not in trips_df:
        trips_df["direction_id"] = ""
    trips_df["direction_id"] = trips_df["direction_id"].fillna("")
    # Filter by service_id if set
    if service_ids:
        trips_df = trips_df[trips_df["service_id"].isin(list(service_ids))]

    # Convert times to minutes; stop_sequence must be numeric so first/last
    # detection and stop ordering compare 10 > 9 rather than "10" < "9"
    stop_times_df["arrival_min"] = stop_times_df["arrival_time"].apply(parse_time_to_minutes)
    stop_times_df["departure_min"] = stop_times_df["departure_time"].apply(parse_time_to_minutes)
    stop_times_df["stop_sequence"] = pd.to_numeric(stop_times_df["stop_sequence"], errors="coerce")
    # Keep only stop_times for the trips that survived
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]

    # Make sure stops_df has stop_code, and trips_df has trip_headsign
    if "stop_code" not in stops_df.columns:
        stops_df["stop_code"] = None
    if "trip_headsign" not in trips_df.columns:
        trips_df = trips_df.assign(trip_headsign="")

    if trips_df["trip_id"].duplicated().any() or stops_df["stop_id"].duplicated().any():
        raise ValueError("Trip IDs and stop IDs must be unique in their GTFS tables.")
    # Merge stop_times + trips
    merged_df = stop_times_df.merge(trips_df, on="trip_id", how="left")

    # Merge with stops to get stop_name, stop_code, timepoint
    stops_merge_cols = ["stop_id", "stop_name", "stop_code"]
    merged_df = merged_df.merge(stops_df[stops_merge_cols], on="stop_id", how="left")

    # Scenario bay re-assignments (before filters, so moved visits still qualify)
    merged_df = apply_bay_overrides(merged_df, stops_df)

    # Mark first/last stops
    merged_df = mark_first_and_last_stops(merged_df)

    # Ensure 'timepoint' is numeric
    if "timepoint" not in merged_df.columns:
        merged_df["timepoint"] = 0
    else:
        merged_df["timepoint"] = (
            pd.to_numeric(merged_df["timepoint"], errors="coerce").fillna(0).astype(int)
        )

    # Force first/last to timepoint=2 if it was 0
    merged_df.loc[(merged_df["is_first_stop"]) & (merged_df["timepoint"] == 0), "timepoint"] = 2
    merged_df.loc[(merged_df["is_last_stop"]) & (merged_df["timepoint"] == 0), "timepoint"] = 2

    # Block-level filtering based on route_short_name, stop_id, stop_code
    if ROUTE_SHORTNAME_FILTER or STOP_ID_FILTER or STOP_CODE_FILTER:
        route_match = (
            merged_df["route_short_name"].isin(ROUTE_SHORTNAME_FILTER)
            if ROUTE_SHORTNAME_FILTER
            else pd.Series(False, index=merged_df.index)
        )
        stopid_match = (
            merged_df["stop_id"].astype(str).isin(STOP_ID_FILTER)
            if STOP_ID_FILTER
            else pd.Series(False, index=merged_df.index)
        )
        stopcode_match = (
            merged_df["stop_code"].astype(str).isin(STOP_CODE_FILTER)
            if STOP_CODE_FILTER
            else pd.Series(False, index=merged_df.index)
        )

        row_match = route_match | stopid_match | stopcode_match
        blocks_that_qualify = merged_df.loc[row_match, "block_id"].unique()
        blocks_that_qualify = [b for b in blocks_that_qualify if pd.notna(b)]

        block_filter_mask = merged_df["block_id"].isin(blocks_that_qualify)
        merged_df = merged_df[block_filter_mask]
        logging.info("After filtering, total rows = %d", len(merged_df))
    else:
        logging.info("No block-level filters applied.")

    if merged_df.empty:
        raise ValueError(
            "No trips remain after service/route/stop filtering; no current timeline was produced."
        )
    for column in ("block_id", "stop_id", "stop_name", "route_short_name"):
        if merged_df[column].isna().any() or merged_df[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Analyzed trips require a populated {column}; check GTFS joins and source fields."
            )
    numeric = ["arrival_min", "departure_min", "stop_sequence"]
    if merged_df[numeric].isna().any().any():
        raise ValueError(
            "Selected stop_times contain missing/invalid times or sequences. "
            "This analysis requires timed visits; interpolate untimed stops upstream."
        )
    if (merged_df["stop_sequence"] % 1 != 0).any():
        raise ValueError("stop_sequence must be an integer.")
    merged_df[numeric] = merged_df[numeric].astype(int)
    if merged_df.duplicated(["trip_id", "stop_sequence"]).any():
        raise ValueError("Duplicate stop_sequence within a trip.")
    for trip_id, group in merged_df.groupby("trip_id", sort=False):
        group = group.sort_values("stop_sequence")
        if (group["arrival_min"] > group["departure_min"]).any():
            raise ValueError(f"Arrival follows departure in trip {trip_id}.")
        if (
            group["arrival_min"].iloc[1:].to_numpy() < group["departure_min"].iloc[:-1].to_numpy()
        ).any():
            raise ValueError(f"Stop times run backwards in trip {trip_id}.")
    return merged_df


def run_step1_gtfs_to_blocks() -> None:
    """Export a complete, manifest-identified timeline and canonical schedule snapshot."""
    out_folder = run_output_folder()
    validate_folders(GTFS_FOLDER_PATH, out_folder)
    manifest_path = Path(out_folder) / "timeline_manifest.json"
    manifest = {"schema_version": 1, "status": "in_progress", "interval_minutes": 1}
    write_json_atomic(manifest_path, manifest)
    try:
        validate_configuration()
        data = load_gtfs_data(GTFS_FOLDER_PATH, dtype=str)
        if "frequencies" in data and not data["frequencies"].empty:
            raise ValueError(
                "Frequency-based trips must be expanded into scheduled trips before analysis."
            )
        trips_df = data["trips"]
        if "route_short_name" not in trips_df:
            trips_df = trips_df.merge(
                data["routes"][["route_id", "route_short_name"]],
                on="route_id",
                how="left",
                validate="many_to_one",
            )
        service_ids = [str(value) for value in CALENDAR_SERVICE_IDS]
        if SERVICE_DATE and service_ids:
            raise ValueError("Set either CALENDAR_SERVICE_IDS or SERVICE_DATE, not both.")
        if SERVICE_DATE:
            service_ids = resolve_service_ids(
                data.get("calendar"), data.get("calendar_dates"), SERVICE_DATE
            )
            if not service_ids:
                raise ValueError(f"No service is active on {SERVICE_DATE}.")
        merged = _merge_and_filter_data(trips_df, data["stop_times"], data["stops"], service_ids)
        summaries = []
        for block, group in merged.groupby("block_id", sort=False):
            check_for_overlapping_trips(group, str(block))
            if group["trip_id"].nunique() > MAX_TRIPS_PER_BLOCK:
                raise ValueError(
                    f"Block {block} exceeds MAX_TRIPS_PER_BLOCK; review service selection."
                )
            summaries.extend(_create_trips_summary(group))
        settings = occupancy_settings()
        end = max(
            DEFAULT_HOURS * 60, max(trip["end"] for trip in summaries) + POST_ARRIVAL_MINUTES + 1
        )
        clusters = [
            {"name": name, "stops": info["stops"]} for name, info in CLUSTER_DEFINITIONS.items()
        ]
        snapshot = {
            "schema_version": 1,
            "interval_minutes": 1,
            "timeline_end": end,
            "settings": settings,
            "clusters": clusters,
            "trips": summaries,
            "service_ids": service_ids,
            "stops": data["stops"][["stop_id", "stop_name"]].to_dict("records"),
        }
        written: list[str] = []
        workbooks: list[str] = []
        frames = []
        used_names: set[str] = set()
        for block, group in merged.groupby("block_id", sort=False):
            block_trips = [trip for trip in summaries if trip["block"] == str(block)]
            frame = pd.DataFrame(
                _build_schedule_rows(block_trips, range(end), str(block), clusters, settings)
            )
            routes = "_".join(sorted(group["route_id"].astype(str).unique()))
            filename = "block_" + re.sub(r'[<>:"/\\|?*]', "_", f"{block}_{routes}") + ".xlsx"
            if filename.lower() in used_names:
                raise ValueError("Block output filenames collide after sanitizing IDs.")
            used_names.add(filename.lower())
            if WRITE_PER_BLOCK_FILES:
                frame.to_excel(Path(out_folder) / filename, index=False)
                written.append(filename)
                workbooks.append(filename)
            if COMBINED_TIMELINE_FILE:
                frames.append(frame.assign(FileName=filename))
        if COMBINED_TIMELINE_FILE:
            pd.concat(frames, ignore_index=True).to_csv(
                Path(out_folder) / COMBINED_TIMELINE_FILE, index=False
            )
            written.append(COMBINED_TIMELINE_FILE)
        write_json_atomic(Path(out_folder) / SCHEDULE_SNAPSHOT_FILE, snapshot)
        written.append(SCHEDULE_SNAPSHOT_FILE)
        if ASSUMPTIONS_FILE:
            (Path(out_folder) / ASSUMPTIONS_FILE).write_text(
                assumptions_text(service_ids), encoding="utf-8"
            )
            written.append(ASSUMPTIONS_FILE)
        require_run_log(write_run_log(Path(out_folder)))
        manifest.update(
            status="complete",
            combined_timeline=COMBINED_TIMELINE_FILE,
            block_workbooks=workbooks,
            schedule_snapshot=SCHEDULE_SNAPSHOT_FILE,
            assumptions_file=ASSUMPTIONS_FILE,
            timeline_end=end,
            files={
                name: hashlib.sha256((Path(out_folder) / name).read_bytes()).hexdigest()
                for name in written
            },
        )
        write_json_atomic(manifest_path, manifest)
        logging.info(
            "Exported %d trips on %d blocks to %s.",
            len(summaries),
            merged["block_id"].nunique(),
            out_folder,
        )
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json_atomic(manifest_path, manifest)
        raise


def assumptions_text(service_ids: Optional[Sequence[str]] = None) -> str:
    """Return the run's occupancy assumptions and scenario settings as text."""
    lines = [
        f"SCENARIO_NAME={SCENARIO_NAME}",
        f"GTFS_FOLDER_PATH={GTFS_FOLDER_PATH}",
        f"SERVICE_DATE={SERVICE_DATE}",
        f"SERVICE_IDS_USED={list(service_ids) if service_ids else CALENDAR_SERVICE_IDS}",
        f"THROUGH_DWELL_MINUTES={THROUGH_DWELL_MINUTES}",
        f"PRE_DEPARTURE_MINUTES={PRE_DEPARTURE_MINUTES}",
        f"POST_ARRIVAL_MINUTES={POST_ARRIVAL_MINUTES}",
        f"IN_BAY_LAYOVER_MAX_MINUTES={IN_BAY_LAYOVER_MAX_MINUTES}",
        f"LAYOVER_THRESHOLD={LAYOVER_THRESHOLD}",
        f"BAY_OVERRIDES={BAY_OVERRIDES}",
        f"ROUTE_SHORTNAME_FILTER={ROUTE_SHORTNAME_FILTER}",
        f"STOP_ID_FILTER={STOP_ID_FILTER}",
        f"STOP_CODE_FILTER={STOP_CODE_FILTER}",
        f"CLUSTER_DEFINITIONS={CLUSTER_DEFINITIONS}",
    ]
    return "\n".join(lines) + "\n"


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


def write_run_log(output_dir: Path) -> bool:
    """Write the ``_runlog.txt`` sidecar for this run into *output_dir*.

    The run writes many files (one workbook per block), so a single log named
    after the script sits alongside them. It captures this script's
    CONFIGURATION block verbatim, between the ``# === BEGIN CONFIG ===`` /
    ``# === END CONFIG ===`` markers, and appends the effective runtime values, including notebook edits.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / RUN_LOG_FILENAME

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
        "BLOCK STATUS TIMELINE RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output folder:    {output_dir}",
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


# --------------------------------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------------------------------------------------------


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
        files: Explicit file names, all required when supplied. With ``None``,
            the standard 13 files are attempted, but only trips, stop_times,
            stops and routes are required; missing optional files are skipped.
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

    explicit_files = files is not None
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
        required = (
            set(files)
            if explicit_files
            else {"trips.txt", "stop_times.txt", "stops.txt", "routes.txt"}
        )
        required_missing = sorted(required.intersection(missing))
        if required_missing:
            raise OSError(f"Missing required GTFS files: {', '.join(required_missing)}")
        files = [name for name in files if name not in missing]

        data: dict[str, pd.DataFrame] = {}
        for file_name in files:
            key = file_name.replace(".txt", "")
            try:
                if archive is None:
                    df = pd.read_csv(
                        os.path.join(gtfs_path, file_name),
                        dtype=dtype,
                        low_memory=False,
                        keep_default_na=False,
                    )
                else:
                    with archive.open(resolved[file_name]) as handle:
                        df = pd.read_csv(
                            handle, dtype=dtype, low_memory=False, keep_default_na=False
                        )
                data[key] = df
                log.info("Loaded %s (%d records).", file_name, len(df))

            except pd.errors.EmptyDataError as exc:
                if file_name not in required:
                    log.warning("Skipping empty optional file %s.", file_name)
                    continue
                raise ValueError(f"File '{file_name}' in '{gtfs_path}' is empty.") from exc

            except pd.errors.ParserError as exc:
                raise ValueError(f"Parser error in '{file_name}' in '{gtfs_path}': {exc}") from exc

        return data
    finally:
        if archive is not None:
            archive.close()


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> int:
    """Master entry point.

    Returns:
        Process exit code: 0 on success, 1 if the required run log could not
        be written, 2 if required CONFIGURATION values are still placeholders
        or ``GTFS_FOLDER_PATH`` does not exist.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if (
        GTFS_FOLDER_PATH == r"your_GTFS_folder_path\here"
        or BLOCK_OUTPUT_FOLDER == r"your_output_folder_path\here"
    ):
        logging.warning(
            "GTFS_FOLDER_PATH and/or BLOCK_OUTPUT_FOLDER are still set to their default "
            "placeholder values. Please update them in the CONFIGURATION section before running."
        )
        return 2
    if not os.path.exists(GTFS_FOLDER_PATH):
        logging.warning(
            "GTFS_FOLDER_PATH does not exist: %s. Update the CONFIGURATION section before running.",
            GTFS_FOLDER_PATH,
        )
        return 2
    try:
        run_step1_gtfs_to_blocks()
    except (RunLogError, ValueError, OSError, KeyError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
