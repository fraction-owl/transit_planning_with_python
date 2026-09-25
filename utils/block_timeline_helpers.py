"""Canonical block-timeline helpers shared by the bay-analysis scripts.

Holds the canonical versions of the minute-by-minute block renderer used by
``block_status_timeline_exporter.py`` and ``bay_change_sweep.py``, and of the
Step 1 timeline readers used by ``bay_usage_analyzer.py`` and
``bay_change_sweep.py``. Per CONTRIBUTING.md, scripts do not import these at
runtime — they carry verbatim copies, and CI's helper-function audit flags any
copy that drifts from this file. The sweep rescores a candidate change by
re-rendering the affected blocks, so its results match a fresh Step 1 run only
while its renderer copy matches the exporter's.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pandas import DataFrame

from utils.time_helpers import minutes_to_hhmm

# -----------------------------------------------------------------------------
# ROUTE NAMES
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# BLOCK RENDERER
# -----------------------------------------------------------------------------


def find_cluster(stop_id: str, clusters: list[dict[str, Any]]) -> Optional[str]:
    """Return cluster name containing the given stop ID, if any."""
    for cluster_item in clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


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


# -----------------------------------------------------------------------------
# STEP 1 TIMELINE READERS
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# REPORT WORKBOOKS
# -----------------------------------------------------------------------------


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
