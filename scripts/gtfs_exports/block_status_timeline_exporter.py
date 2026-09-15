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

Typical usage
-------------
Update the paths in the CONFIGURATION section and run from a shell, ArcGIS
Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pandas as pd

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

GTFS_FOLDER_PATH = r"your_GTFS_folder_path\here"
BLOCK_OUTPUT_FOLDER = r"your_output_folder_path\here"

# Optional: each run is written to BLOCK_OUTPUT_FOLDER\<SCENARIO_NAME>. Name it
# for the assumptions and bay assignments used, e.g. "direct_conflict",
# "likely_conflict_alt_bays". Step 2 (bay_usage_analyzer.py) reads the same
# subfolder. Leave "" to write to BLOCK_OUTPUT_FOLDER itself.
SCENARIO_NAME = ""

DEFAULT_HOURS = 26
TIME_INTERVAL_MINUTES = 1

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
# the feed regardless of service.
CALENDAR_SERVICE_IDS: list[str] = ["3"]
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
ASSUMPTIONS_FILE = r"timeline_assumptions.txt"  # in the run folder; "" to skip

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

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================


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
    active = expand_service_active_dates(calendar_df, calendar_dates_df)
    return sorted(service_ids_active_on(active, target))


def validate_folders(input_path: str, output_path: str) -> None:
    """Check that the input folder exists, and ensure the output folder is created if not."""
    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Input path does not exist or is not a directory: {input_path}")
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


def _status_for_same_trip(minute: int, stop_info: tuple) -> Optional[tuple]:
    """Status of a bus at ``minute`` if it is at one of this trip's own stops.

    Through stops (arrival == departure, neither first nor last) are occupied
    for ``THROUGH_DWELL_MINUTES`` starting at the stop time. A first stop with
    a GTFS hold (arrival < departure) is DWELL, then LOADING for the last
    ``PRE_DEPARTURE_MINUTES`` before departure. Sitting *before* a trip's
    first time and staying *after* its last time are handled in
    :func:`_row_for_inactive`, because those minutes fall outside the trip.
    """
    (arr, dep, s_id, s_name, t_id, is_first, is_last, s_seq, t_val) = stop_info
    detail = (s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep), t_id, s_seq, t_val)

    if is_last and minute == arr:
        return ("ARRIVE",) + detail
    if is_first and minute == dep:
        return ("DEPART",) + detail
    if not is_first and not is_last and arr == dep and arr <= minute < arr + THROUGH_DWELL_MINUTES:
        return ("ARRIVE/DEPART",) + detail
    if arr < minute < dep:
        if is_first and minute >= dep - PRE_DEPARTURE_MINUTES:
            return ("LOADING",) + detail
        return ("DWELL",) + detail
    return None


def _gap_status(gap: int, same_place: bool) -> tuple[str, str]:
    """Classify the gap between two trips of one block.

    Returns ``(status, layover_location)``. A gap that starts and ends at the
    same stop or in the same cluster is a layover: in the arrival bay
    (``DWELL``) when the gap is at most ``IN_BAY_LAYOVER_MAX_MINUTES``,
    otherwise in the overflow/layover space (``LAYOVER``, or ``LONG BREAK``
    beyond ``LAYOVER_THRESHOLD``). Anything else is a ``DEADHEAD``.
    """
    if not same_place:
        return ("DEADHEAD", "")
    if gap <= IN_BAY_LAYOVER_MAX_MINUTES:
        return ("DWELL", "in bay")
    if gap <= LAYOVER_THRESHOLD:
        return ("LAYOVER", "overflow")
    return ("LONG BREAK", "overflow")


def get_status_for_minute(
    minute: int, stop_times_sequence: list[tuple]
) -> tuple[
    str,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[int],
    int,
]:
    """Determine the block's status at a specific 'minute' within one trip.

    Returns tuple:
       (status, stop_id, stop_name, arrival_str, departure_str,
        trip_id_for_status, stop_sequence, timepoint_value)
    """
    if not stop_times_sequence:
        return ("EMPTY", None, None, None, None, None, None, 0)

    # Later stops take precedence: when THROUGH_DWELL_MINUTES stretches one
    # stop's window over the next stop's scheduled minute, the bus has moved on.
    for item in reversed(stop_times_sequence):
        same_trip_result = _status_for_same_trip(minute, item)
        if same_trip_result:
            return same_trip_result

    for i, item in enumerate(stop_times_sequence):
        # Between this stop and the next one of the same trip
        if i < len(stop_times_sequence) - 1:
            next_arr = stop_times_sequence[i + 1][0]
            if item[1] < minute < next_arr:
                return ("TRAVELING BETWEEN STOPS", None, None, None, None, item[4], None, 0)

    return ("EMPTY", None, None, None, None, None, None, 0)


# --------------------------------------------------------------------------------------------------
# MAIN BLOCK PROCESSING
# --------------------------------------------------------------------------------------------------


def check_for_overlapping_trips(
    block_subset: pd.DataFrame, block_id: str
) -> None:  # Added return type annotation
    """Print a warning if any trips within this block overlap in time."""
    trip_times = []
    for trip_id, group in block_subset.groupby("trip_id"):
        start_val = group["arrival_min"].min()
        end_val = group["departure_min"].max()
        trip_times.append((trip_id, start_val, end_val))

    trip_times.sort(key=lambda x: x[1])
    for i in range(len(trip_times) - 1):
        trip1_id, start1, end1 = trip_times[i]
        for j in range(i + 1, len(trip_times)):
            trip2_id, start2, end2 = trip_times[j]
            # If they intersect in any way
            if start2 <= end1 and start1 <= end2:
                logging.warning(
                    "WARNING: Overlapping trips in block %s: "
                    "Trip %s (%s–%s) overlaps with "
                    "%s (%s–%s)",
                    block_id,
                    trip1_id,
                    start1,
                    end1,
                    trip2_id,
                    start2,
                    end2,
                )


def fill_stop_ids_for_dwell_layover_loading(
    df_in: pd.DataFrame,
) -> pd.DataFrame:  # Added return type annotation
    """Fills in missing stop information for certain vehicle statuses.

    For rows with a status of 'DWELL', 'LAYOVER', or 'LOADING', where the
    'Stop ID' is typically empty, this function populates the 'Stop ID',
    'Stop Name', 'Stop Sequence', 'Arrival Time', 'Departure Time', and
    'Trip ID' fields. The data used for filling is based on the last known
    stop information from the same vehicle block, enhancing the readability
    of the final output.

    Args:
        df_in (pd.DataFrame): The input DataFrame containing vehicle status and
            stop information.

    Returns:
        pd.DataFrame: A new DataFrame with the missing stop information filled in.
    """
    df_out = df_in.copy()
    last_stop_id = None
    last_stop_name = None
    last_stop_seq = None
    last_arr = None
    last_dep = None
    last_trip_id = None

    for idx in df_out.index:
        status = df_out.loc[idx, "Status"]
        stop_id = df_out.loc[idx, "Stop ID"]
        if stop_id:
            # Update "last known" stop info
            last_stop_id = stop_id
            last_stop_name = df_out.loc[idx, "Stop Name"]
            last_stop_seq = df_out.loc[idx, "Stop Sequence"]
            last_arr = df_out.loc[idx, "Arrival Time"]
            last_dep = df_out.loc[idx, "Departure Time"]
            last_trip_id = df_out.loc[idx, "Trip ID"]
        else:
            if status in ["DWELL", "LAYOVER", "LOADING"]:
                if last_stop_id is not None:
                    df_out.loc[idx, "Stop ID"] = last_stop_id
                if last_stop_name is not None:
                    df_out.loc[idx, "Stop Name"] = last_stop_name
                if last_stop_seq is not None:
                    df_out.loc[idx, "Stop Sequence"] = last_stop_seq
                if last_arr is not None:
                    df_out.loc[idx, "Arrival Time"] = last_arr
                if last_dep is not None:
                    df_out.loc[idx, "Departure Time"] = last_dep
                if last_trip_id is not None:
                    df_out.loc[idx, "Trip ID"] = last_trip_id
    return df_out


def _create_trips_summary(
    block_subset: pd.DataFrame,
) -> list[dict[str, Any]]:  # Added return type annotation
    """Helper to build trip summaries for a block.

    Returns a list of dictionaries with trip info and sorted stop times.
    """
    trips_summary = []
    for trip_id, trip_df in block_subset.groupby("trip_id"):
        trip_df_sorted = trip_df.sort_values("stop_sequence")
        start_time = trip_df_sorted["arrival_min"].min()
        end_time = trip_df_sorted["departure_min"].max()

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


def _status_for_active_trips(
    minute: int,
    active_trips: list[dict[str, Any]],
    bus_stop_clusters: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], tuple]:  # Added return type annotation
    """Among all 'active' trips for the given minute, determine the single chosen status."""
    candidate_info = []
    for trip_obj in active_trips:
        status_tuple = get_status_for_minute(minute, trip_obj["stop_times_sequence"])
        candidate_info.append((trip_obj, status_tuple))

    # Filter out actual "EMPTY" statuses
    valid_candidates = [
        (trip_obj, stat) for (trip_obj, stat) in candidate_info if stat[0] != "EMPTY"
    ]
    if not valid_candidates:
        # All were "EMPTY"
        return None, ("EMPTY", None, None, None, None, None, None, 0)

    if len(valid_candidates) == 1:
        return valid_candidates[0]

    # Tie-break if multiple
    def candidate_sort_key(item: tuple[dict[str, Any], tuple]) -> tuple[bool, int]:
        stat = item[1]
        stop_seq = stat[6] if stat[6] is not None else 999999
        timepoint_val = stat[7] if len(stat) == 8 else 0
        # Prefer timepoints, then lower stop_sequence
        return (timepoint_val == 0, stop_seq)

    valid_candidates.sort(key=candidate_sort_key)
    return valid_candidates[0]


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
) -> dict[str, Any]:
    """Assemble one output row. Route fields come from ``trip`` when given."""
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
    }


def _row_for_inactive(
    minute: int,
    block_id: str,
    all_trips: list[dict[str, Any]],
    bus_stop_clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a row describing the vehicle when **no trip is active**.

    Precedence, using the closest finished trip (``prev``) and the next
    upcoming trip (``next``):

    1. Within ``PRE_DEPARTURE_MINUTES`` of ``next``'s first time: *LOADING*
       at ``next``'s first stop (this also covers a block's pull-out).
    2. Within ``POST_ARRIVAL_MINUTES`` after ``prev``'s last time: *ARRIVE*
       at ``prev``'s last stop (also covers a block's pull-in).
    3. Otherwise, between two trips: *DWELL* (in the arrival bay),
       *LAYOVER* / *LONG BREAK* (in the overflow space) when the next trip
       starts at the same stop or in the same cluster; *DEADHEAD* when it
       starts somewhere else.
    4. No bounding trip on one side: *INACTIVE*.
    """
    prev_trip: Optional[dict[str, Any]] = None
    next_trip: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ locate bounding trips
    for trip_obj in all_trips:
        if trip_obj["end"] < minute:
            if prev_trip is None or trip_obj["end"] > prev_trip["end"]:
                prev_trip = trip_obj
        elif trip_obj["start"] > minute:
            if next_trip is None or trip_obj["start"] < next_trip["start"]:
                next_trip = trip_obj

    prev_id = prev_trip["trip_id"] if prev_trip else ""
    next_id = next_trip["trip_id"] if next_trip else ""
    prev_end_str = minutes_to_hhmm(prev_trip["end"]) if prev_trip else ""
    next_start_str = minutes_to_hhmm(next_trip["start"]) if next_trip else ""

    # ------------------------------------------------------------------ status decision
    if next_trip is not None and minute >= next_trip["start"] - PRE_DEPARTURE_MINUTES:
        return _make_row(
            minute,
            block_id,
            "LOADING",
            trip=next_trip,
            stop_id=next_trip["first_stop_id"],
            stop_name=next_trip["first_stop_name"],
            stop_seq=next_trip["first_stop_seq"],
            arr_str=prev_end_str,
            dep_str=next_start_str,
            trip_id=next_id,
            layover_location="in bay",
            prev_trip_id=prev_id,
            next_trip_id=next_id,
        )

    if prev_trip is not None and minute < prev_trip["end"] + POST_ARRIVAL_MINUTES:
        return _make_row(
            minute,
            block_id,
            "ARRIVE",
            trip=prev_trip,
            stop_id=prev_trip["last_stop_id"],
            stop_name=prev_trip["last_stop_name"],
            stop_seq=prev_trip["last_stop_seq"],
            arr_str=prev_end_str,
            dep_str=next_start_str,
            trip_id=prev_id,
            layover_location="in bay",
            prev_trip_id=prev_id,
            next_trip_id=next_id,
        )

    if prev_trip is not None and next_trip is not None:
        same_stop = prev_trip["last_stop_id"] == next_trip["first_stop_id"]
        prev_cluster = find_cluster(prev_trip["last_stop_id"], bus_stop_clusters)
        next_cluster = find_cluster(next_trip["first_stop_id"], bus_stop_clusters)
        same_cluster = prev_cluster is not None and prev_cluster == next_cluster
        status, location = _gap_status(
            next_trip["start"] - prev_trip["end"], same_stop or same_cluster
        )
        if status == "DEADHEAD":
            return _make_row(minute, block_id, status, prev_trip_id=prev_id, next_trip_id=next_id)
        return _make_row(
            minute,
            block_id,
            status,
            trip=prev_trip,
            stop_id=prev_trip["last_stop_id"],
            stop_name=prev_trip["last_stop_name"],
            stop_seq=prev_trip["last_stop_seq"],
            arr_str=prev_end_str,
            dep_str=next_start_str,
            trip_id=prev_id,
            layover_location=location,
            prev_trip_id=prev_id,
            next_trip_id=next_id,
        )

    return _make_row(minute, block_id, "INACTIVE", prev_trip_id=prev_id, next_trip_id=next_id)


def _build_schedule_rows(
    trips_summary: list[dict[str, Any]],
    timeline: range,
    block_id: str,
    bus_stop_clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the minute-by-minute schedule rows for a single block.

    Args:
        trips_summary: Output of :func:`_create_trips_summary`.
        timeline: Range object representing every minute to be evaluated.
        block_id: Identifier of the vehicle block.
        bus_stop_clusters: Cluster definitions used for dwell/layover logic.

    Returns:
        A list of dictionaries (one per minute) suitable for `pd.DataFrame`.
    """
    rows: list[dict[str, Any]] = []

    for minute in timeline:
        # --------------------------------------------------- identify active trips
        possible_trips = [t for t in trips_summary if t["start"] <= minute <= t["end"]]

        if possible_trips:
            chosen_trip, chosen_status = _status_for_active_trips(
                minute, possible_trips, bus_stop_clusters
            )

            if chosen_trip is not None:
                (
                    status,
                    stop_id,
                    stop_name,
                    arr_str,
                    dep_str,
                    trip_id_for_status,
                    stop_seq,
                    t_val,
                ) = chosen_status

                # Treat “EMPTY” as travelling between stops.
                if status == "EMPTY":
                    status = "TRAVELING BETWEEN STOPS"

                row = _make_row(
                    minute,
                    block_id,
                    status,
                    trip=chosen_trip,
                    stop_id=stop_id,
                    stop_name=stop_name,
                    stop_seq=stop_seq,
                    arr_str=arr_str,
                    dep_str=dep_str,
                    trip_id=trip_id_for_status or "",
                    timepoint=t_val,
                    layover_location="in bay" if status in {"DWELL", "LOADING"} else "",
                )
            else:
                # All active trips returned “EMPTY”.
                row = _make_row(minute, block_id, "TRAVELING BETWEEN STOPS")
        else:
            # No active trips ⇒ bridging or fully inactive.
            row = _row_for_inactive(minute, block_id, trips_summary, bus_stop_clusters)

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
    rows = _build_schedule_rows(trips_summary, timeline, block_id, bus_stop_clusters)
    df = pd.DataFrame(rows)
    return fill_stop_ids_for_dwell_layover_loading(df)


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

    # Merge stop_times + trips
    merged_df = stop_times_df.merge(trips_df, on="trip_id", how="left")

    # Merge with stops to get stop_name, stop_code, timepoint
    stops_merge_cols = ["stop_id", "stop_name", "stop_code"]
    if "timepoint" in stops_df.columns:
        stops_merge_cols.append("timepoint")
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

    return merged_df


def run_step1_gtfs_to_blocks() -> None:
    """Step 1: Generate block-level schedules from GTFS."""
    logging.info("=== Step 1: Reading GTFS and generating block-level schedules ===")
    out_folder = run_output_folder()
    validate_folders(GTFS_FOLDER_PATH, out_folder)

    # Use the standardized loader
    logging.info("Loading GTFS data using standardized function ...")
    gtfs_data = load_gtfs_data(GTFS_FOLDER_PATH, dtype=str)

    # Pull out frames you'll need
    trips_df = gtfs_data["trips"]
    stop_times_df = gtfs_data["stop_times"]
    stops_df = gtfs_data["stops"]

    # Optionally handle routes
    routes_df = gtfs_data.get("routes", None)
    if routes_df is not None and "route_short_name" in routes_df.columns:
        logging.info("Merging routes.txt with trips ...")
        if "route_short_name" not in trips_df.columns:
            trips_df = trips_df.merge(
                routes_df[["route_id", "route_short_name"]],
                on="route_id",
                how="left",
            )
    else:
        logging.warning("WARNING: No routes.txt found or missing route_short_name column.")
        if "route_short_name" not in trips_df.columns:
            trips_df["route_short_name"] = None

    # Service day: explicit service_ids, or everything active on SERVICE_DATE
    service_ids = [str(sid) for sid in CALENDAR_SERVICE_IDS]
    if not service_ids and SERVICE_DATE:
        service_ids = resolve_service_ids(
            gtfs_data.get("calendar"), gtfs_data.get("calendar_dates"), SERVICE_DATE
        )
        logging.info("Service IDs active on %s: %s", SERVICE_DATE, service_ids)
        if not service_ids:
            raise ValueError(f"No service_id is active on {SERVICE_DATE} in this feed.")

    # Now merge & filter to get the final dataset
    merged_df = _merge_and_filter_data(trips_df, stop_times_df, stops_df, service_ids)

    all_blocks = merged_df["block_id"].dropna().unique()
    logging.info("Identified %d block(s) to process.", len(all_blocks))
    logging.info("Occupancy assumptions: %s", assumptions_text(service_ids).replace("\n", "; "))

    max_minutes = DEFAULT_HOURS * 60
    timeline = range(0, max_minutes, TIME_INTERVAL_MINUTES)

    combined_frames: list[pd.DataFrame] = []
    for blk_id in all_blocks:
        logging.info("\nProcessing block %s...", blk_id)
        block_data = merged_df[merged_df["block_id"] == blk_id].copy()
        trip_ids = block_data["trip_id"].unique()

        check_for_overlapping_trips(block_data, blk_id)

        if len(trip_ids) > MAX_TRIPS_PER_BLOCK:
            logging.info(
                "Block %s has %d trips > limit %d. Skipped.",
                blk_id,
                len(trip_ids),
                MAX_TRIPS_PER_BLOCK,
            )
            continue

        block_schedule_df = process_block(block_data, blk_id, timeline, BUS_STOP_CLUSTERS_STEP1)
        block_schedule_df = block_schedule_df.sort_values("Timestamp")

        block_route_ids = block_data["route_id"].dropna().unique()
        if len(block_route_ids) > 0:
            block_route_str = "_".join(str(rte) for rte in block_route_ids)
        else:
            block_route_str = "NA"

        out_name = f"block_{blk_id}_{block_route_str}.xlsx"
        if WRITE_PER_BLOCK_FILES:
            out_path = os.path.join(out_folder, out_name)
            block_schedule_df.to_excel(out_path, index=False)
            logging.info("Finished block %s; saved to %s", blk_id, out_path)
        if COMBINED_TIMELINE_FILE:
            combined_frames.append(block_schedule_df.assign(FileName=out_name))

    if COMBINED_TIMELINE_FILE and combined_frames:
        combined_path = os.path.join(out_folder, COMBINED_TIMELINE_FILE)
        pd.concat(combined_frames, ignore_index=True).to_csv(combined_path, index=False)
        logging.info("Combined timeline written to %s", combined_path)

    if ASSUMPTIONS_FILE:
        assumptions_path = os.path.join(out_folder, ASSUMPTIONS_FILE)
        with open(assumptions_path, "w", encoding="utf-8") as handle:
            handle.write(assumptions_text(service_ids))
        logging.info("Assumptions written to %s", assumptions_path)

    logging.info("\nStep 1 complete: All block-level spreadsheets generated.")


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


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> int:
    """Master entry point.

    Returns:
        Process exit code: 0 on success, 2 if required CONFIGURATION values
        are still placeholders or ``GTFS_FOLDER_PATH`` does not exist.
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
    run_step1_gtfs_to_blocks()
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
