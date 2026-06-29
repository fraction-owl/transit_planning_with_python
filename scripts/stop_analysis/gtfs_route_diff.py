"""GTFS route comparison (before vs after) with notebook-friendly execution.

A planner-facing companion to ``gtfs_stop_diff.py``. Point it at two GTFS feeds
(an *old*/before feed and a *new*/after feed) and it classifies every route into
the service-change buckets planners actually track:

- eliminated / added routes (after reconciling ``route_id`` rekeys, so a renumbered
  route is reported as a rekey, not a spurious elimination + addition);
- major / minor alignment changes (how much the set of stops a route serves moved);
- major / minor schedule changes (trips/day, span, headway, revenue-hours on a
  representative weekday). "none" means the weekday trips are *identical* (an exact
  fingerprint check, not a tolerance), so a feed that keeps trips/day, span and headway
  but shifts individual trips reads as a minor change rather than unchanged;
- candidate route splits (1 -> 2+) and merges (2+ -> 1), inferred from stop-set
  containment and flagged for planner review;
- reblocking (the set of routes a route interlines with via shared ``block_id`` changed);
- route name changes (``route_short_name`` / ``route_long_name``) and other route
  attribute changes (``route_type``/mode, colors, ``agency_id``, ...);
- calendar-approach changes (a ``service_id`` redefined from, say, weekday to a new
  day pattern or date range) and per-route day-type coverage changes (gained/lost
  Saturday or Sunday service);
- fare changes (GTFS-Fares v1: ``fare_attributes`` prices/attributes and the
  ``fare_rules`` route -> fare mapping);
- each feed's effective date range and how the two windows relate (gap / overlap).

Design decisions (all threshold knobs live in the CONFIG block and are CLI flags):
- Alignment is classified from **stop patterns**, not shape geometry: ``stop_times``
  exists in every feed, and a stop-set comparison is what catches "Route 10 now
  skips the Elm St loop". Shape length is reported as context when ``shapes.txt``
  exists but does not drive the major/minor call.
- Schedule magnitude is measured on a single **representative weekday** picked per
  feed (the in-range weekday with the most active service). Weekend gains/losses are
  still surfaced through the route-level day-type rollup.
- Fares cover GTFS-Fares **v1** only (the files the loader already reads).

Outputs (CSV):
- routes_overview.csv         : one row per route with flags + a plain-English summary
- routes_eliminated.csv       : routes only in the before feed (true eliminations)
- routes_added.csv            : routes only in the after feed (true additions)
- routes_alignment_changes.csv
- routes_schedule_changes.csv
- routes_reblocked.csv
- routes_name_attr_changes.csv
- routes_daytype_changes.csv  : routes that gained/lost weekday/Saturday/Sunday service
- routes_split_merge_candidates.csv : heuristic split (1->2+) / merge (2+->1) candidates
- calendar_changes.csv        : service_ids added/removed/redefined (the "calendar approach")
- fare_attribute_changes.csv  : fare_id prices/attributes added/removed/changed
- route_fare_changes.csv      : routes whose fare_rules mapping changed
- summary.json

Also outputs:
- routes_comparison.xlsx (one sheet per table above + summary)
- gtfs_route_diff.log

No arcpy / geopandas. pandas + numpy only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

BEFORE_GTFS_DIR = Path(r"Path\To\Old\GTFS\Dir")
AFTER_GTFS_DIR = Path(r"Path\To\New\GTFS\Dir")
OUTPUT_DIR = Path(r"Path\To\Output\Dir")

# Weekday used to characterize the schedule (monday..friday). The in-range date of
# this weekday with the most active service is picked independently in each feed.
ANALYSIS_WEEKDAY: str = "tuesday"

# --- Route-identity / rekey matching ---------------------------------------
# When a route_id exists in only one feed, try to match it to an only-in-the-other
# route_id with the same route_short_name and a stop-set Jaccard at least this high,
# and report it as a rekey (route_id change) instead of an elimination + addition.
REKEY_MIN_JACCARD: float = 0.50

# --- Alignment (stop-pattern) thresholds -----------------------------------
# Jaccard of the route's served-stop set (after vs before). Identical sets and
# terminals => unchanged; >= this and not identical => minor; below => major.
ALIGN_MINOR_JACCARD: float = 0.80

# --- Schedule thresholds (representative weekday) ---------------------------
# "Schedule change" is a three-state classification:
#   none  : the route's weekday trips are *identical* (same set of trip start/end
#           times) -- literally nothing changed. This is an exact fingerprint check,
#           not a tolerance, so two feeds that share trips/day, span and headway but
#           differ trip-by-trip are NOT called unchanged.
#   minor : the schedule changed but no metric reaches its *_MAJOR knob below.
#   major : at least one metric change reaches its *_MAJOR knob.
# Only the major boundary is a knob; "minor vs none" is decided by exact equality.
SCHED_TRIPS_PCT_MAJOR: float = 0.30  # fractional change in trips/day
SCHED_SPAN_MIN_MAJOR: float = 60.0  # absolute change in service span (minutes)
SCHED_HEADWAY_PCT_MAJOR: float = 0.50  # fractional change in median headway

# --- Splits / merges --------------------------------------------------------
# Heuristic, stop-set based *candidate* detection (for planner review, not gospel):
# a split is a route whose stops no single successor covers but 2+ routes jointly do;
# a merge is the mirror image. Each contributing route must cover at least
# SPLIT_MERGE_MIN_SHARE of the source's stops, and together at least
# SPLIT_MERGE_MIN_COVERAGE. Set SPLIT_MERGE_ENABLE = False to skip.
SPLIT_MERGE_ENABLE: bool = True
SPLIT_MERGE_MIN_COVERAGE: float = 0.60
SPLIT_MERGE_MIN_SHARE: float = 0.30

# --- Fares ------------------------------------------------------------------
# A fare price change at/above this absolute amount is flagged "major".
FARE_PRICE_MAJOR_DELTA: float = 0.25

# --- Date ranges ------------------------------------------------------------
# NOTE: feed date metadata is unreliable in practice -- feed_info dates are often
# missing or stale and calendar ranges are sometimes placeholders. The windows
# reported here are "as declared by the feed" (the summary notes the source), and
# the script warns when feed_info and the calendar disagree. When you know the real
# service dates, set the overrides below. The schedule comparison itself does NOT
# depend on accurate absolute dates: it only needs each feed's calendar to be
# internally consistent enough to pick a representative weekday with active service.
# Each override is None (auto-detect) or a ("YYYYMMDD", "YYYYMMDD") tuple.
BEFORE_DATE_RANGE: Optional[tuple[str, str]] = None
AFTER_DATE_RANGE: Optional[tuple[str, str]] = None

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

_PLACEHOLDER = "Path\\To"

# Route attribute columns compared for "other" (non-name) changes, when present.
ROUTE_ATTR_COLS: tuple[str, ...] = (
    "route_type",
    "route_color",
    "route_text_color",
    "route_desc",
    "route_url",
    "agency_id",
    "route_sort_order",
)

_WEEKDAY_COLS: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class Summary:
    """Summary metrics for the route comparison."""

    before_route_count: int
    after_route_count: int
    matched_count: int
    rekeyed_count: int
    eliminated_count: int
    added_count: int
    alignment_major_count: int
    alignment_minor_count: int
    schedule_major_count: int
    schedule_minor_count: int
    reblocked_count: int
    name_changed_count: int
    attr_changed_count: int
    daytype_changed_count: int
    fare_changed_count: int
    split_merge_candidate_count: int
    unchanged_count: int
    before_date_range: str
    after_date_range: str
    date_range_relationship: str


# =============================================================================
# Logging
# =============================================================================


def setup_logging(output_dir: Path) -> None:
    """Configure the root logger to write to console + a file in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "gtfs_route_diff.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# =============================================================================
# IO helpers
# =============================================================================

_REQUIRED_FILES: tuple[str, ...] = ("routes.txt", "trips.txt", "stop_times.txt")
_OPTIONAL_FILES: tuple[str, ...] = (
    "calendar.txt",
    "calendar_dates.txt",
    "fare_attributes.txt",
    "fare_rules.txt",
    "feed_info.txt",
    "shapes.txt",
    "frequencies.txt",
    "agency.txt",
)


def _read_csv(path: Path, usecols: Optional[object] = None) -> pd.DataFrame:
    """Read a GTFS table as all-string columns with missing values as empty strings.

    Args:
        path: Path to the ``.txt`` file.
        usecols: Optional value forwarded to ``pandas.read_csv(usecols=...)``; a
            callable is convenient for keeping only columns that exist.

    Returns:
        The parsed table; every column is ``str`` and absent values are ``""``.
    """
    return pd.read_csv(
        path,
        dtype=str,
        usecols=usecols,  # type: ignore[arg-type]
        encoding="utf-8-sig",
        low_memory=False,
        keep_default_na=False,
        na_filter=False,
    )


def load_feed(gtfs_dir: Path, label: str) -> dict[str, pd.DataFrame]:
    """Load the GTFS files needed for a route diff, tolerating absent optional files.

    Args:
        gtfs_dir: Folder containing the GTFS feed.
        label: Human-readable feed label used in log messages (e.g. ``"before"``).

    Returns:
        Mapping of file stem -> DataFrame for every file that was present.

    Raises:
        OSError: ``gtfs_dir`` is missing, a required file is absent, or neither
            ``calendar.txt`` nor ``calendar_dates.txt`` is present.
    """
    if not os.path.exists(gtfs_dir):
        raise OSError(f"{label}: directory '{gtfs_dir}' does not exist.")

    missing_required = [
        name for name in _REQUIRED_FILES if not os.path.exists(os.path.join(gtfs_dir, name))
    ]
    if missing_required:
        raise OSError(f"{label}: missing required GTFS files: {', '.join(missing_required)}")

    feed: dict[str, pd.DataFrame] = {}
    trip_cols = {"route_id", "trip_id", "service_id", "direction_id", "block_id", "shape_id"}
    st_cols = {"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"}

    for name in _REQUIRED_FILES + _OPTIONAL_FILES:
        path = Path(gtfs_dir) / name
        if not path.exists():
            continue
        key = name.replace(".txt", "")
        if name == "trips.txt":
            frame = _read_csv(path, usecols=lambda c: c in trip_cols)
        elif name == "stop_times.txt":
            frame = _read_csv(path, usecols=lambda c: c in st_cols)
        else:
            frame = _read_csv(path)
        feed[key] = frame
        logging.info("%s: loaded %s (%d records).", label, name, len(frame))

    if "calendar" not in feed and "calendar_dates" not in feed:
        raise OSError(f"{label}: feed has neither calendar.txt nor calendar_dates.txt.")

    if "direction_id" not in feed["trips"].columns:
        feed["trips"]["direction_id"] = ""

    return feed


# =============================================================================
# Generic helpers
# =============================================================================


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize a text column for comparisons (fill NA, cast to str, strip)."""
    return series.fillna("").astype(str).str.strip()


def jaccard(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Jaccard similarity of two sets; two empty sets are treated as identical (1.0)."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert GTFS ``HH:MM:SS`` strings (hours may exceed 24) to seconds since midnight."""
    parts = series.astype(str).str.split(":", expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    seconds = pd.to_numeric(parts[2], errors="coerce") if parts.shape[1] >= 3 else 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds


def median_headway_min(start_secs: pd.Series) -> float:
    """Median gap (minutes) between consecutive trip starts; NaN if fewer than 3 trips.

    Coarse and all-day: it mixes directions and ignores gaps of 4 h or more (layovers
    and the overnight break), matching the convention used elsewhere in the repo.
    """
    times = np.sort(start_secs.dropna().to_numpy(dtype=float))
    if times.size < 3:
        return float("nan")
    diffs = np.diff(times)
    diffs = diffs[(diffs > 0) & (diffs < 4 * 3600)]
    return float(np.median(diffs) / 60.0) if diffs.size else float("nan")


def _parse_yyyymmdd(value: str) -> Optional[dt.date]:
    """Parse a ``YYYYMMDD`` string to a date, or return None if it is not parseable."""
    value = str(value).strip()
    if len(value) != 8 or not value.isdigit():
        return None
    return dt.date(int(value[0:4]), int(value[4:6]), int(value[6:8]))


# =============================================================================
# Calendar / service selection + day-type coverage
# =============================================================================


def _weekday_name_to_int(name: str) -> int:
    """Map a weekday name (monday..friday) to a Python weekday index (Monday=0)."""
    key = name.strip().lower()
    if key not in _WEEKDAY_COLS or _WEEKDAY_COLS.index(key) >= 5:
        raise ValueError(f"ANALYSIS_WEEKDAY must be monday..friday, got {name!r}.")
    return _WEEKDAY_COLS.index(key)


def _active_service_ids_for_date(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    day: dt.date,
) -> set[str]:
    """Resolve the service_ids active on ``day`` from calendar + calendar_dates."""
    ds = day.strftime("%Y%m%d")
    active: set[str] = set()

    if calendar is not None and not calendar.empty:
        col = _WEEKDAY_COLS[day.weekday()]
        cal = calendar
        in_range = (cal["start_date"].str.strip() <= ds) & (cal["end_date"].str.strip() >= ds)
        runs = cal[col].str.strip() == "1" if col in cal.columns else False
        active |= set(cal.loc[in_range & runs, "service_id"].astype(str))

    if calendar_dates is not None and not calendar_dates.empty:
        cd = calendar_dates[calendar_dates["date"].str.strip() == ds]
        active |= set(cd.loc[cd["exception_type"].str.strip() == "1", "service_id"].astype(str))
        active -= set(cd.loc[cd["exception_type"].str.strip() == "2", "service_id"].astype(str))

    return active


def _feed_date_bounds(
    calendar: Optional[pd.DataFrame], calendar_dates: Optional[pd.DataFrame]
) -> tuple[Optional[dt.date], Optional[dt.date]]:
    """Earliest and latest service dates implied by calendar + calendar_dates."""
    dates: list[dt.date] = []
    if calendar is not None and not calendar.empty:
        for col in ("start_date", "end_date"):
            if col in calendar.columns:
                dates += [d for d in (_parse_yyyymmdd(x) for x in calendar[col]) if d is not None]
    if calendar_dates is not None and not calendar_dates.empty and "date" in calendar_dates.columns:
        dates += [d for d in (_parse_yyyymmdd(x) for x in calendar_dates["date"]) if d is not None]
    if not dates:
        return None, None
    return min(dates), max(dates)


def pick_representative_weekday_services(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    weekday: str,
    max_search_days: int = 400,
) -> tuple[Optional[dt.date], set[str]]:
    """Pick the in-range date of ``weekday`` with the most active service_ids.

    Scans outward from the midpoint of the feed's date range and returns the date of
    the requested weekday whose active service_id set is largest. Works from
    ``calendar.txt`` and/or a ``calendar_dates.txt``-only feed.

    Returns:
        ``(date, active_service_ids)``; ``(None, set())`` if no date can be resolved.
    """
    target = _weekday_name_to_int(weekday)
    lo, hi = _feed_date_bounds(calendar, calendar_dates)
    if lo is None or hi is None:
        return None, set()

    anchor = lo + (hi - lo) // 2
    best_date: Optional[dt.date] = None
    best_active: set[str] = set()
    for step in range(max_search_days + 1):
        for sign in (1, -1):
            if step == 0 and sign == -1:
                continue
            cand = anchor + dt.timedelta(days=step * sign)
            if cand < lo or cand > hi or cand.weekday() != target:
                continue
            active = _active_service_ids_for_date(calendar, calendar_dates, cand)
            if len(active) > len(best_active):
                best_date, best_active = cand, active
    return best_date, best_active


def service_id_day_types(
    calendar: Optional[pd.DataFrame], calendar_dates: Optional[pd.DataFrame]
) -> dict[str, frozenset[str]]:
    """Map each service_id to the day types it covers: weekday / saturday / sunday.

    Uses ``calendar.txt`` day-of-week flags when present and augments with the actual
    weekday of any ``calendar_dates`` additions, so ``calendar_dates``-only feeds are
    still bucketed. ``exception_type == 2`` removals are ignored (coarse on purpose).
    """
    out: dict[str, set[str]] = {}

    if calendar is not None and not calendar.empty:
        for _, row in calendar.iterrows():
            sid = str(row.get("service_id", "")).strip()
            if not sid:
                continue
            types = out.setdefault(sid, set())
            if any(str(row.get(c, "")).strip() == "1" for c in _WEEKDAY_COLS[:5]):
                types.add("weekday")
            if str(row.get("saturday", "")).strip() == "1":
                types.add("saturday")
            if str(row.get("sunday", "")).strip() == "1":
                types.add("sunday")

    if calendar_dates is not None and not calendar_dates.empty:
        adds = calendar_dates[calendar_dates["exception_type"].str.strip() == "1"]
        for sid, day in zip(adds["service_id"].astype(str), adds["date"].astype(str)):
            parsed = _parse_yyyymmdd(day)
            if parsed is None:
                continue
            bucket = (
                "saturday"
                if parsed.weekday() == 5
                else "sunday"
                if parsed.weekday() == 6
                else "weekday"
            )
            out.setdefault(sid.strip(), set()).add(bucket)

    return {sid: frozenset(types) for sid, types in out.items()}


def route_day_types(
    trips: pd.DataFrame, sid_day_types: dict[str, frozenset[str]]
) -> dict[str, frozenset[str]]:
    """Union of day types served by each route, via its trips' service_ids."""
    out: dict[str, set[str]] = {}
    for route_id, service_id in zip(trips["route_id"].astype(str), trips["service_id"].astype(str)):
        out.setdefault(route_id, set()).update(sid_day_types.get(service_id.strip(), frozenset()))
    return {rid: frozenset(types) for rid, types in out.items()}


# =============================================================================
# Trip endpoints / schedule metrics / alignment
# =============================================================================


def trip_endpoints(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Per trip: first/last stop_id and start/end/runtime seconds.

    Returns a frame keyed on ``trip_id`` with ``first_stop``, ``last_stop``,
    ``start_sec``, ``end_sec`` and ``runtime_sec`` (NaN runtime guards clock anomalies).
    """
    cols = ["trip_id", "first_stop", "last_stop", "start_sec", "end_sec", "runtime_sec"]
    if stop_times.empty:
        return pd.DataFrame(columns=cols)

    st = stop_times.copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"])
    dep = st["departure_time"] if "departure_time" in st.columns else pd.Series("", index=st.index)
    arr = st["arrival_time"] if "arrival_time" in st.columns else pd.Series("", index=st.index)
    dep = dep.where(dep.astype(str).str.len() > 0)
    arr = arr.where(arr.astype(str).str.len() > 0)
    st["t_sec"] = time_to_seconds(dep.fillna(arr))

    idx_first = st.groupby("trip_id")["stop_sequence"].idxmin()
    idx_last = st.groupby("trip_id")["stop_sequence"].idxmax()
    first = st.loc[idx_first, ["trip_id", "stop_id", "t_sec"]].rename(
        columns={"stop_id": "first_stop", "t_sec": "start_sec"}
    )
    last = st.loc[idx_last, ["trip_id", "stop_id", "t_sec"]].rename(
        columns={"stop_id": "last_stop", "t_sec": "end_sec"}
    )
    out = first.merge(last, on="trip_id", how="inner")
    out["runtime_sec"] = out["end_sec"] - out["start_sec"]
    out.loc[out["runtime_sec"] < 0, "runtime_sec"] = np.nan
    return out[cols]


def route_stop_sets(trips: pd.DataFrame, stop_times: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Map each route_id to the set of stop_ids any of its trips serve."""
    if trips.empty or stop_times.empty:
        return {}
    rs = trips[["route_id", "trip_id"]].merge(
        stop_times[["trip_id", "stop_id"]], on="trip_id", how="inner"
    )
    rs = rs[["route_id", "stop_id"]].drop_duplicates()
    return {str(rid): frozenset(grp.astype(str)) for rid, grp in rs.groupby("route_id")["stop_id"]}


def route_terminals(trips: pd.DataFrame, endpoints: pd.DataFrame) -> dict[str, frozenset[tuple]]:
    """Map each route_id to its modal (direction, first_stop, last_stop) terminal triples.

    For each route/direction the most common first/last stop pair across trips is the
    terminal pair; the per-route value is the frozenset of those triples over directions.
    """
    if trips.empty or endpoints.empty:
        return {}
    merged = trips[["route_id", "direction_id", "trip_id"]].merge(
        endpoints[["trip_id", "first_stop", "last_stop"]], on="trip_id", how="inner"
    )
    if merged.empty:
        return {}
    counts = (
        merged.groupby(["route_id", "direction_id", "first_stop", "last_stop"])
        .size()
        .reset_index(name="n")
        .sort_values(["route_id", "direction_id", "n"], ascending=[True, True, False])
    )
    modal = counts.drop_duplicates(["route_id", "direction_id"], keep="first")
    out: dict[str, set[tuple]] = {}
    for rid, d, f, last in zip(
        modal["route_id"].astype(str),
        modal["direction_id"].astype(str),
        modal["first_stop"].astype(str),
        modal["last_stop"].astype(str),
    ):
        out.setdefault(rid, set()).add((d, f, last))
    return {rid: frozenset(triples) for rid, triples in out.items()}


def classify_alignment(
    stops_before: frozenset[str],
    stops_after: frozenset[str],
    terminals_changed: bool,
    minor_jaccard: float,
) -> str:
    """Classify an alignment change as ``"none"``, ``"minor"`` or ``"major"``.

    Identical stop sets with unchanged terminals are ``"none"``; otherwise the
    stop-set Jaccard draws the line (``>= minor_jaccard`` is minor, below is major).
    """
    if stops_before == stops_after and not terminals_changed:
        return "none"
    if jaccard(stops_before, stops_after) >= minor_jaccard:
        return "minor"
    return "major"


def route_schedule_metrics(trips: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Per-route weekday schedule metrics from in-scope trips and their endpoints.

    Returns ``[route_id, trips_per_day, span_hours, median_headway_min, revenue_hours]``.
    ``trips`` should already be filtered to the representative weekday's service_ids.
    """
    cols = ["route_id", "trips_per_day", "span_hours", "median_headway_min", "revenue_hours"]
    if trips.empty:
        return pd.DataFrame(columns=cols)

    te = trips[["route_id", "trip_id"]].merge(endpoints, on="trip_id", how="inner")
    if te.empty:
        return pd.DataFrame(columns=cols)

    grouped = te.groupby("route_id")
    out = grouped.agg(
        trips_per_day=("trip_id", "nunique"),
        revenue_hours=("runtime_sec", lambda s: float(np.nansum(s.to_numpy(dtype=float)) / 3600.0)),
        first_dep_sec=("start_sec", "min"),
        last_arr_sec=("end_sec", "max"),
    )
    out["span_hours"] = (out["last_arr_sec"] - out["first_dep_sec"]) / 3600.0
    out["median_headway_min"] = grouped["start_sec"].apply(median_headway_min)
    out = out.reset_index()
    out["route_id"] = out["route_id"].astype(str)
    return out[cols]


def schedule_fingerprint(trips: pd.DataFrame, endpoints: pd.DataFrame) -> dict[str, tuple]:
    """Per-route exact signature of weekday service: sorted (start_sec, end_sec) pairs.

    Two routes with equal fingerprints run the same trips at the same times -- the basis
    for the "literally nothing changed" check. It is deliberately stricter than the
    summary metrics: a feed can preserve trips/day, span and median headway while
    shifting individual trips, and that should read as a (minor) change, not "none".
    ``trips`` should already be filtered to the representative weekday's service_ids.
    """
    if trips.empty or endpoints.empty:
        return {}
    te = trips[["route_id", "trip_id"]].merge(
        endpoints[["trip_id", "start_sec", "end_sec"]], on="trip_id", how="inner"
    )
    if te.empty:
        return {}
    out: dict[str, tuple] = {}
    for rid, sub in te.groupby("route_id"):
        pairs = sub[["start_sec", "end_sec"]].fillna(-1.0).round(0).to_numpy()
        out[str(rid)] = tuple(sorted(map(tuple, pairs.tolist())))
    return out


def classify_schedule(
    before: Optional[pd.Series],
    after: Optional[pd.Series],
    identical: bool,
    knobs: dict[str, float],
) -> tuple[str, str]:
    """Classify a route's schedule change and describe the deltas.

    The three states are exact: ``"none"`` means the weekday schedule is identical
    (``identical`` is True); a route running on the representative weekday in only one
    feed is ``"major"`` (weekday service added/dropped); otherwise the magnitude of the
    trips/day, span and headway deltas decides ``"minor"`` vs ``"major"`` against the
    ``*_major`` knobs.

    Args:
        before: The route's before-feed metric row (or None if it runs no weekday trips).
        after: The route's after-feed metric row (or None if it runs no weekday trips).
        identical: Whether the two weekday schedule fingerprints are exactly equal.
        knobs: The ``SCHED_*`` thresholds keyed by name.

    Returns:
        ``(tier, description)`` where tier is ``"none"``/``"minor"``/``"major"``.
    """
    if before is None and after is None:
        return "none", ""
    if (before is None) != (after is None):
        return "major", "weekday service added" if before is None else "weekday service dropped"
    if identical:
        return "none", ""

    def _f(row: pd.Series, key: str) -> float:
        return float(row[key]) if pd.notna(row.get(key)) else float("nan")

    trips_b, trips_a = _f(before, "trips_per_day"), _f(after, "trips_per_day")
    span_b, span_a = _f(before, "span_hours"), _f(after, "span_hours")
    hw_b, hw_a = _f(before, "median_headway_min"), _f(after, "median_headway_min")

    trips_pct = abs(trips_a - trips_b) / trips_b if trips_b > 0 else (1.0 if trips_a > 0 else 0.0)
    span_min = abs(span_a - span_b) * 60.0 if np.isfinite(span_a) and np.isfinite(span_b) else 0.0
    hw_pct = (
        abs(hw_a - hw_b) / hw_b if np.isfinite(hw_b) and hw_b > 0 and np.isfinite(hw_a) else 0.0
    )

    is_major = (
        trips_pct >= knobs["trips_major"]
        or span_min >= knobs["span_major"]
        or hw_pct >= knobs["headway_major"]
    )
    tier = "major" if is_major else "minor"

    bits: list[str] = []
    if np.isfinite(trips_b) and np.isfinite(trips_a) and trips_a != trips_b:
        bits.append(f"trips/day {trips_b:.0f}->{trips_a:.0f}")
    if np.isfinite(span_b) and np.isfinite(span_a) and abs(span_a - span_b) >= 1 / 60:
        bits.append(f"span {span_b:.1f}h->{span_a:.1f}h")
    if np.isfinite(hw_b) and np.isfinite(hw_a) and abs(hw_a - hw_b) >= 0.5:
        bits.append(f"headway {hw_b:.0f}->{hw_a:.0f} min")
    if not bits:
        bits.append("trip times shifted")
    return tier, "; ".join(bits)


# =============================================================================
# Reblocking (interline partners)
# =============================================================================


def route_interline_partners(trips: pd.DataFrame) -> Optional[dict[str, frozenset[str]]]:
    """Map each route_id to the set of other routes it shares a ``block_id`` with.

    Returns None if the feed has no usable ``block_id`` column, so callers can skip
    reblocking detection gracefully.
    """
    if "block_id" not in trips.columns:
        return None
    t = trips[["route_id", "block_id"]].copy()
    t["block_id"] = t["block_id"].astype(str).str.strip()
    t = t[t["block_id"].str.len() > 0][["route_id", "block_id"]].drop_duplicates()
    if t.empty:
        return None

    pairs = t.merge(t, on="block_id", suffixes=("_a", "_b"))
    pairs = pairs[pairs["route_id_a"] != pairs["route_id_b"]]
    out: dict[str, set[str]] = {rid: set() for rid in t["route_id"].astype(str).unique()}
    for a, b in zip(pairs["route_id_a"].astype(str), pairs["route_id_b"].astype(str)):
        out[a].add(b)
    return {rid: frozenset(parts) for rid, parts in out.items()}


# =============================================================================
# Route attributes / names
# =============================================================================


def _routes_indexed(routes: pd.DataFrame) -> pd.DataFrame:
    """Return routes keyed on a normalized string route_id, first row wins on dupes."""
    df = routes.copy()
    df["route_id"] = normalize_text(df["route_id"])
    return df.drop_duplicates(subset="route_id", keep="first").set_index("route_id")


def compare_route_names_attrs(
    before_row: pd.Series, after_row: pd.Series, attr_cols: Sequence[str]
) -> tuple[bool, bool, list[str]]:
    """Compare two route rows' names and attributes.

    Returns ``(short_name_changed, long_name_changed, other_changed_fields)``.
    """

    def _v(row: pd.Series, key: str) -> str:
        return str(row.get(key, "") or "").strip()

    short_changed = _v(before_row, "route_short_name") != _v(after_row, "route_short_name")
    long_changed = _v(before_row, "route_long_name") != _v(after_row, "route_long_name")
    changed = [c for c in attr_cols if _v(before_row, c) != _v(after_row, c)]
    return short_changed, long_changed, changed


# =============================================================================
# Route correspondence / rekey matching
# =============================================================================


@dataclass(frozen=True)
class Correspondence:
    """Resolved route correspondence between the two feeds."""

    matched: list[str]  # route_ids present (by id) in both feeds
    rekeyed: dict[str, str]  # before_id -> after_id (same route, new route_id)
    eliminated: list[str]  # before-only route_ids (true eliminations)
    added: list[str]  # after-only route_ids (true additions)


def build_correspondence(
    routes_before: pd.DataFrame,
    routes_after: pd.DataFrame,
    stops_before: dict[str, frozenset[str]],
    stops_after: dict[str, frozenset[str]],
    rekey_min_jaccard: float,
) -> Correspondence:
    """Resolve which routes match, were rekeyed, eliminated, or added.

    Routes match first on ``route_id``. Each remaining before-only id is matched to an
    after-only id with the same ``route_short_name`` and the highest stop-set Jaccard,
    provided that Jaccard is at least ``rekey_min_jaccard``; such a pair is a rekey.
    Whatever stays unmatched is a true elimination or addition.
    """
    before_ids = set(normalize_text(routes_before["route_id"]))
    after_ids = set(normalize_text(routes_after["route_id"]))
    matched = sorted(before_ids & after_ids)

    before_only = before_ids - after_ids
    after_only = after_ids - before_ids

    short_before = _routes_indexed(routes_before).get("route_short_name")
    short_after = _routes_indexed(routes_after).get("route_short_name")

    def _short(table: Optional[pd.Series], rid: str) -> str:
        if table is None or rid not in table.index:
            return ""
        return str(table.loc[rid] or "").strip()

    rekeyed: dict[str, str] = {}
    claimed_after: set[str] = set()
    for b_id in sorted(before_only):
        b_short = _short(short_before, b_id)
        best_id, best_j = None, rekey_min_jaccard
        for a_id in sorted(after_only):
            if a_id in claimed_after or _short(short_after, a_id) != b_short:
                continue
            j = jaccard(stops_before.get(b_id, frozenset()), stops_after.get(a_id, frozenset()))
            if j >= best_j:
                best_id, best_j = a_id, j
        if best_id is not None:
            rekeyed[b_id] = best_id
            claimed_after.add(best_id)

    eliminated = sorted(before_only - set(rekeyed))
    added = sorted(after_only - claimed_after)
    return Correspondence(matched=matched, rekeyed=rekeyed, eliminated=eliminated, added=added)


# =============================================================================
# Calendar approach diff
# =============================================================================


def _day_pattern(row: pd.Series) -> str:
    """Render a calendar row's 7 day flags as an MTWTFSS-style pattern string."""
    letters = "MTWTFSS"
    return "".join(
        letters[i] if str(row.get(col, "")).strip() == "1" else "."
        for i, col in enumerate(_WEEKDAY_COLS)
    )


def compare_calendars(
    calendar_before: Optional[pd.DataFrame], calendar_after: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Diff ``calendar.txt`` between feeds: service_ids added/removed/redefined.

    "Redefined" captures the user's "calendar approach" change — a service_id whose
    day-of-week pattern or active date range moved (e.g. weekday -> a new pattern).
    """
    cols = ["service_id", "status", "detail"]
    if calendar_before is None and calendar_after is None:
        return pd.DataFrame(columns=cols)

    def _index(cal: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
        if cal is None or cal.empty or "service_id" not in cal.columns:
            return {}
        cal = cal.copy()
        cal["service_id"] = normalize_text(cal["service_id"])
        cal = cal.drop_duplicates(subset="service_id", keep="first")
        return {str(r["service_id"]): r for _, r in cal.iterrows()}

    before, after = _index(calendar_before), _index(calendar_after)
    rows: list[dict[str, str]] = []
    for sid in sorted(set(before) | set(after)):
        if sid in before and sid not in after:
            rows.append({"service_id": sid, "status": "removed", "detail": ""})
        elif sid not in before and sid in after:
            rows.append(
                {"service_id": sid, "status": "added", "detail": f"days={_day_pattern(after[sid])}"}
            )
        else:
            b, a = before[sid], after[sid]
            details: list[str] = []
            if _day_pattern(b) != _day_pattern(a):
                details.append(f"days {_day_pattern(b)}->{_day_pattern(a)}")
            b_range = f"{str(b.get('start_date', '')).strip()}-{str(b.get('end_date', '')).strip()}"
            a_range = f"{str(a.get('start_date', '')).strip()}-{str(a.get('end_date', '')).strip()}"
            if b_range != a_range:
                details.append(f"dates {b_range}->{a_range}")
            if details:
                rows.append(
                    {"service_id": sid, "status": "redefined", "detail": "; ".join(details)}
                )
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# Fares diff (GTFS-Fares v1)
# =============================================================================

_FARE_ATTR_FIELDS: tuple[str, ...] = (
    "price",
    "currency_type",
    "payment_method",
    "transfers",
    "transfer_duration",
)


def compare_fare_attributes(
    fa_before: Optional[pd.DataFrame],
    fa_after: Optional[pd.DataFrame],
    price_major_delta: float,
) -> pd.DataFrame:
    """Diff ``fare_attributes.txt``: fare_ids added/removed and field changes.

    Returns rows with ``fare_id``, ``status`` (added/removed/changed), a ``detail``
    string, the signed ``price_delta`` (when both prices parse), and a ``severity``
    flag of ``"major"`` when an absolute price change is at/above ``price_major_delta``.
    """
    cols = ["fare_id", "status", "detail", "price_delta", "severity"]
    if fa_before is None and fa_after is None:
        return pd.DataFrame(columns=cols)

    def _index(fa: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
        if fa is None or fa.empty or "fare_id" not in fa.columns:
            return {}
        fa = fa.copy()
        fa["fare_id"] = normalize_text(fa["fare_id"])
        fa = fa.drop_duplicates(subset="fare_id", keep="first")
        return {str(r["fare_id"]): r for _, r in fa.iterrows()}

    before, after = _index(fa_before), _index(fa_after)
    rows: list[dict[str, object]] = []
    for fid in sorted(set(before) | set(after)):
        if fid in before and fid not in after:
            rows.append(
                {
                    "fare_id": fid,
                    "status": "removed",
                    "detail": "",
                    "price_delta": np.nan,
                    "severity": "",
                }
            )
            continue
        if fid not in before and fid in after:
            rows.append(
                {
                    "fare_id": fid,
                    "status": "added",
                    "detail": "",
                    "price_delta": np.nan,
                    "severity": "",
                }
            )
            continue
        b, a = before[fid], after[fid]
        changed = [
            f"{f} {str(b.get(f, '')).strip()}->{str(a.get(f, '')).strip()}"
            for f in _FARE_ATTR_FIELDS
            if str(b.get(f, "")).strip() != str(a.get(f, "")).strip()
        ]
        if not changed:
            continue
        price_b = pd.to_numeric(str(b.get("price", "")).strip(), errors="coerce")
        price_a = pd.to_numeric(str(a.get("price", "")).strip(), errors="coerce")
        delta = (
            float(price_a - price_b) if pd.notna(price_b) and pd.notna(price_a) else float("nan")
        )
        severity = "major" if np.isfinite(delta) and abs(delta) >= price_major_delta else "minor"
        rows.append(
            {
                "fare_id": fid,
                "status": "changed",
                "detail": "; ".join(changed),
                "price_delta": delta,
                "severity": severity,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def route_fare_map(fare_rules: Optional[pd.DataFrame]) -> dict[str, frozenset[str]]:
    """Map each route_id to the set of fare_ids that reference it in ``fare_rules.txt``."""
    if fare_rules is None or fare_rules.empty:
        return {}
    if "route_id" not in fare_rules.columns or "fare_id" not in fare_rules.columns:
        return {}
    fr = fare_rules[["route_id", "fare_id"]].copy()
    fr["route_id"] = normalize_text(fr["route_id"])
    fr["fare_id"] = normalize_text(fr["fare_id"])
    fr = fr[fr["route_id"].str.len() > 0]
    return {str(rid): frozenset(grp.astype(str)) for rid, grp in fr.groupby("route_id")["fare_id"]}


# =============================================================================
# Date ranges
# =============================================================================


def feed_date_range(
    feed: dict[str, pd.DataFrame], override: Optional[tuple[str, str]]
) -> tuple[str, str, str]:
    """Determine a feed's effective ``(start, end, source)`` date window.

    Prefers an explicit ``override``, then ``feed_info`` feed_start/end dates, then the
    min/max of calendar + calendar_dates. Dates are ``YYYYMMDD`` strings (``""`` if
    unknown); ``source`` is one of ``override``/``feed_info``/``calendar``/``unknown``.
    """
    if override is not None:
        return override[0], override[1], "override"

    feed_info = feed.get("feed_info")
    if feed_info is not None and not feed_info.empty:
        cols = set(feed_info.columns)
        if {"feed_start_date", "feed_end_date"} <= cols:
            starts = [s for s in normalize_text(feed_info["feed_start_date"]) if s]
            ends = [e for e in normalize_text(feed_info["feed_end_date"]) if e]
            if starts and ends:
                return min(starts), max(ends), "feed_info"

    lo, hi = _feed_date_bounds(feed.get("calendar"), feed.get("calendar_dates"))
    if lo is not None and hi is not None:
        return lo.strftime("%Y%m%d"), hi.strftime("%Y%m%d"), "calendar"
    return "", "", "unknown"


def date_range_relationship(before: tuple[str, str, str], after: tuple[str, str, str]) -> str:
    """Describe how the before and after validity windows relate (gap / overlap)."""
    b_end = _parse_yyyymmdd(before[1])
    a_start = _parse_yyyymmdd(after[0])
    if b_end is None or a_start is None:
        return "unknown (a feed has no parseable date range)"
    gap = (a_start - b_end).days
    if gap > 1:
        return f"gap of {gap - 1} day(s) between feeds"
    if gap < 0:
        return f"overlapping by {-gap} day(s)"
    return "contiguous (after starts when before ends)"


def warn_on_date_discrepancy(
    feed: dict[str, pd.DataFrame], label: str, tolerance_days: int = 31
) -> None:
    """Log a warning when a feed's feed_info dates and calendar bounds disagree.

    feed_info dates are frequently stale; surfacing a large divergence from the actual
    calendar service window tells a planner not to trust the declared range.
    """
    feed_info = feed.get("feed_info")
    if feed_info is None or feed_info.empty:
        return
    if not ({"feed_start_date", "feed_end_date"} <= set(feed_info.columns)):
        return
    starts = [s for s in normalize_text(feed_info["feed_start_date"]) if s]
    ends = [e for e in normalize_text(feed_info["feed_end_date"]) if e]
    if not starts or not ends:
        return
    fi_lo, fi_hi = _parse_yyyymmdd(min(starts)), _parse_yyyymmdd(max(ends))
    cal_lo, cal_hi = _feed_date_bounds(feed.get("calendar"), feed.get("calendar_dates"))
    if fi_lo is None or fi_hi is None or cal_lo is None or cal_hi is None:
        return
    if abs((fi_lo - cal_lo).days) > tolerance_days or abs((fi_hi - cal_hi).days) > tolerance_days:
        logging.warning(
            "%s: feed_info dates (%s..%s) disagree with the calendar window (%s..%s) by more "
            "than %d days; declared dates may be stale -- consider an explicit date override.",
            label,
            fi_lo.isoformat(),
            fi_hi.isoformat(),
            cal_lo.isoformat(),
            cal_hi.isoformat(),
            tolerance_days,
        )


# =============================================================================
# Split / merge candidate detection (heuristic, stop-set containment)
# =============================================================================


def _containment_candidates(
    sources: dict[str, frozenset[str]],
    others: dict[str, frozenset[str]],
    kind: str,
    min_coverage: float,
    min_share: float,
) -> list[dict[str, object]]:
    """Find sources whose stops no single ``other`` covers but 2+ jointly do.

    A source qualifies only when its single best coverage by any one ``other`` route is
    below ``min_coverage`` (so clean 1:1 renames/reroutes are excluded), yet 2+ routes
    each covering at least ``min_share`` of it together reach ``min_coverage``.
    """
    rows: list[dict[str, object]] = []
    for s_id, s_stops in sources.items():
        if len(s_stops) < 2:
            continue
        coverages = []
        best_single = 0.0
        for o_id, o_stops in others.items():
            if not o_stops:
                continue
            cov = len(s_stops & o_stops) / len(s_stops)
            best_single = max(best_single, cov)
            if cov >= min_share:
                coverages.append((o_id, cov))
        if best_single >= min_coverage or len(coverages) < 2:
            continue
        coverages.sort(key=lambda pair: pair[1], reverse=True)
        union: set[str] = set()
        for o_id, _ in coverages:
            union |= others[o_id]
        union_cov = len(s_stops & union) / len(s_stops)
        if union_cov < min_coverage:
            continue
        rows.append(
            {
                "kind": kind,
                "source_route_id": s_id,
                "target_route_ids": ";".join(o_id for o_id, _ in coverages),
                "union_coverage": round(union_cov, 3),
                "target_shares": ";".join(f"{o_id}:{cov:.2f}" for o_id, cov in coverages),
            }
        )
    return rows


def detect_splits_merges(
    stops_before: dict[str, frozenset[str]],
    stops_after: dict[str, frozenset[str]],
    min_coverage: float,
    min_share: float,
) -> pd.DataFrame:
    """Heuristic candidate detection of route splits (1->2+) and merges (2+->1).

    A *split* is a before route whose served stops are jointly covered by 2+ after
    routes (no single successor covers it); a *merge* is the mirror image over before
    routes. These are candidates for planner review -- routes sharing a busy corridor
    can trip the heuristic -- not authoritative classifications.

    Returns ``[kind, source_route_id, target_route_ids, union_coverage, target_shares]``.
    """
    rows = _containment_candidates(stops_before, stops_after, "split", min_coverage, min_share)
    rows += _containment_candidates(stops_after, stops_before, "merge", min_coverage, min_share)
    return pd.DataFrame(
        rows,
        columns=["kind", "source_route_id", "target_route_ids", "union_coverage", "target_shares"],
    )


# =============================================================================
# Per-route overview assembly
# =============================================================================

_HEADLINE_PRIORITY: tuple[str, ...] = (
    "eliminated",
    "added",
    "split_merge",
    "major_alignment",
    "major_schedule",
    "rekeyed",
    "minor_alignment",
    "minor_schedule",
    "reblocked",
    "name_change",
    "daytype_change",
    "fare_change",
    "attr_change",
    "unchanged",
)


def summarize_change(flags: dict[str, object]) -> tuple[str, str]:
    """Pick the most severe headline label and build a plain-English summary string.

    Args:
        flags: The per-route change components (status, tiers, booleans, detail text).

    Returns:
        ``(headline_change, change_summary)``.
    """
    present: set[str] = set()
    parts: list[str] = []

    status = flags.get("status")
    if status == "eliminated":
        present.add("eliminated")
        parts.append("Route eliminated")
    elif status == "added":
        present.add("added")
        parts.append("Route added")

    if flags.get("rekeyed"):
        present.add("rekeyed")
        parts.append(
            f"route_id changed {flags.get('route_id_before')}->{flags.get('route_id_after')}"
        )

    def _labeled(label: str, detail: object) -> str:
        text = str(detail or "").strip()
        return f"{label} ({text})" if text else label

    align = str(flags.get("alignment_change") or "none")
    if align == "major":
        present.add("major_alignment")
        parts.append(_labeled("Major alignment change", flags.get("alignment_detail")))
    elif align == "minor":
        present.add("minor_alignment")
        parts.append(_labeled("Minor alignment change", flags.get("alignment_detail")))

    sched = str(flags.get("schedule_change") or "none")
    if sched == "major":
        present.add("major_schedule")
        parts.append(_labeled("Major schedule change", flags.get("schedule_detail")))
    elif sched == "minor":
        present.add("minor_schedule")
        parts.append(_labeled("Minor schedule change", flags.get("schedule_detail")))

    if flags.get("split_merge"):
        present.add("split_merge")
        parts.append(str(flags.get("split_merge")))

    if flags.get("reblocked"):
        present.add("reblocked")
        parts.append("Reblocked (interline partners changed)")

    if flags.get("name_change"):
        present.add("name_change")
        name_detail = str(flags.get("name_detail") or "").strip()
        parts.append(f"Name change: {name_detail}" if name_detail else "Name change")

    if flags.get("daytype_change"):
        present.add("daytype_change")
        parts.append(f"Day-type change: {flags.get('daytype_change')}")

    if flags.get("fare_change"):
        present.add("fare_change")
        parts.append("Fare mapping changed")

    if flags.get("attr_changes"):
        present.add("attr_change")
        parts.append(f"Attributes changed: {flags.get('attr_changes')}")

    if not present:
        return "unchanged", "No change detected"

    headline = next((label for label in _HEADLINE_PRIORITY if label in present), "unchanged")
    return headline, "; ".join(parts)


# =============================================================================
# Comparison driver
# =============================================================================


def _round(value: object, digits: int = 2) -> object:
    """Round a numeric value for output, passing non-numeric/NaN through unchanged."""
    num = pd.to_numeric(value, errors="coerce")
    return round(float(num), digits) if pd.notna(num) else value


def compare_routes(
    before: dict[str, pd.DataFrame],
    after: dict[str, pd.DataFrame],
    weekday: str,
    knobs: dict[str, float],
) -> dict[str, object]:
    """Compare two loaded feeds and return every output table plus a Summary.

    Args:
        before: Loaded before feed (see :func:`load_feed`).
        after: Loaded after feed.
        weekday: Representative weekday name (monday..friday) for schedule metrics.
        knobs: Threshold knobs (rekey, alignment, schedule, fare).

    Returns:
        A dict of named DataFrames plus ``"summary"`` (a :class:`Summary`).
    """
    routes_b, routes_a = before["routes"], after["routes"]
    trips_b, trips_a = before["trips"], after["trips"]
    st_b, st_a = before["stop_times"], after["stop_times"]

    # Alignment inputs.
    stops_b = route_stop_sets(trips_b, st_b)
    stops_a = route_stop_sets(trips_a, st_a)
    ends_b, ends_a = trip_endpoints(st_b), trip_endpoints(st_a)
    terms_b, terms_a = route_terminals(trips_b, ends_b), route_terminals(trips_a, ends_a)

    # Correspondence (matched / rekeyed / eliminated / added).
    corr = build_correspondence(routes_b, routes_a, stops_b, stops_a, knobs["rekey_jaccard"])
    logging.info(
        "Routes: before=%d after=%d | matched=%d rekeyed=%d eliminated=%d added=%d",
        len(stops_b) or len(routes_b),
        len(stops_a) or len(routes_a),
        len(corr.matched),
        len(corr.rekeyed),
        len(corr.eliminated),
        len(corr.added),
    )

    # Schedule metrics on each feed's representative weekday.
    date_b, sids_b = pick_representative_weekday_services(
        before.get("calendar"), before.get("calendar_dates"), weekday
    )
    date_a, sids_a = pick_representative_weekday_services(
        after.get("calendar"), after.get("calendar_dates"), weekday
    )
    logging.info("Representative %s: before=%s after=%s", weekday, date_b, date_a)
    wk_trips_b = trips_b[trips_b["service_id"].astype(str).isin(sids_b)]
    wk_trips_a = trips_a[trips_a["service_id"].astype(str).isin(sids_a)]
    sched_b = route_schedule_metrics(wk_trips_b, ends_b).set_index("route_id")
    sched_a = route_schedule_metrics(wk_trips_a, ends_a).set_index("route_id")
    fp_b = schedule_fingerprint(wk_trips_b, ends_b)
    fp_a = schedule_fingerprint(wk_trips_a, ends_a)

    # Reblocking inputs.
    partners_b = route_interline_partners(trips_b)
    partners_a = route_interline_partners(trips_a)
    reblock_possible = partners_b is not None and partners_a is not None
    if not reblock_possible:
        logging.info("Reblocking detection skipped (no usable block_id in one or both feeds).")

    # Day-type coverage.
    daytypes_b = route_day_types(
        trips_b, service_id_day_types(before.get("calendar"), before.get("calendar_dates"))
    )
    daytypes_a = route_day_types(
        trips_a, service_id_day_types(after.get("calendar"), after.get("calendar_dates"))
    )

    # Fares.
    route_fares_b = route_fare_map(before.get("fare_rules"))
    route_fares_a = route_fare_map(after.get("fare_rules"))

    # Split / merge candidates (a before route -> 2+ after routes, or the mirror).
    if knobs.get("split_merge_enable", 1.0):
        split_merge_df = detect_splits_merges(
            stops_b, stops_a, knobs["split_merge_min_coverage"], knobs["split_merge_min_share"]
        )
    else:
        split_merge_df = pd.DataFrame(
            columns=[
                "kind",
                "source_route_id",
                "target_route_ids",
                "union_coverage",
                "target_shares",
            ]
        )
    split_sources = {
        str(r["source_route_id"]): r
        for _, r in split_merge_df[split_merge_df["kind"] == "split"].iterrows()
    }
    merge_sources = {
        str(r["source_route_id"]): r
        for _, r in split_merge_df[split_merge_df["kind"] == "merge"].iterrows()
    }

    routes_b_idx = _routes_indexed(routes_b)
    routes_a_idx = _routes_indexed(routes_a)

    # Build the per-route overview. Each entity is keyed by its after-feed id, except
    # eliminations which only exist before.
    entities: list[tuple[str, str, str]] = []  # (before_id, after_id, status)
    for rid in corr.matched:
        entities.append((rid, rid, "present"))
    for b_id, a_id in corr.rekeyed.items():
        entities.append((b_id, a_id, "present"))
    for rid in corr.added:
        entities.append(("", rid, "added"))
    for rid in corr.eliminated:
        entities.append((rid, "", "eliminated"))

    rows: list[dict[str, object]] = []
    for before_id, after_id, status in entities:
        b_row = routes_b_idx.loc[before_id] if before_id in routes_b_idx.index else None
        a_row = routes_a_idx.loc[after_id] if after_id in routes_a_idx.index else None
        name_row = a_row if a_row is not None else b_row

        rekeyed = bool(before_id and after_id and before_id != after_id)

        # Names / attributes (only meaningful when present in both feeds).
        short_changed = long_changed = False
        attr_changed: list[str] = []
        name_detail = ""
        if b_row is not None and a_row is not None:
            short_changed, long_changed, attr_changed = compare_route_names_attrs(
                b_row, a_row, ROUTE_ATTR_COLS
            )
            name_bits = []
            if short_changed:
                name_bits.append(
                    f"short {str(b_row.get('route_short_name', '')).strip()}->"
                    f"{str(a_row.get('route_short_name', '')).strip()}"
                )
            if long_changed:
                name_bits.append("long_name changed")
            name_detail = "; ".join(name_bits)

        # Alignment.
        align_tier = "none"
        align_detail = ""
        sb = stops_b.get(before_id, frozenset())
        sa = stops_a.get(after_id, frozenset())
        if status == "present":
            terms_changed = terms_b.get(before_id, frozenset()) != terms_a.get(
                after_id, frozenset()
            )
            align_tier = classify_alignment(sb, sa, terms_changed, knobs["align_minor_jaccard"])
            if align_tier != "none":
                bits = [f"jaccard={jaccard(sb, sa):.2f}", f"stops {len(sb)}->{len(sa)}"]
                if terms_changed:
                    bits.append("terminal changed")
                align_detail = ", ".join(bits)

        # Schedule. "none" means the weekday trips are identical (exact fingerprint),
        # not merely that the summary metrics match.
        sched_tier, sched_detail = "none", ""
        if status == "present":
            b_sched = sched_b.loc[before_id] if before_id in sched_b.index else None
            a_sched = sched_a.loc[after_id] if after_id in sched_a.index else None
            sched_identical = (
                before_id in fp_b and after_id in fp_a and fp_b[before_id] == fp_a[after_id]
            )
            sched_tier, sched_detail = classify_schedule(b_sched, a_sched, sched_identical, knobs)

        # Reblocking.
        reblocked = False
        if status == "present" and reblock_possible:
            reblocked = partners_b.get(before_id, frozenset()) != partners_a.get(
                after_id, frozenset()
            )

        # Day-type coverage.
        daytype_change = ""
        if status == "present":
            db, da = daytypes_b.get(before_id, frozenset()), daytypes_a.get(after_id, frozenset())
            gained = sorted(da - db)
            lost = sorted(db - da)
            daytype_change = "; ".join([f"+{d}" for d in gained] + [f"-{d}" for d in lost])

        # Fares.
        fare_change = False
        if status == "present":
            fare_change = route_fares_b.get(before_id, frozenset()) != route_fares_a.get(
                after_id, frozenset()
            )

        # Split / merge candidate annotation (a before route is a split source; an
        # after route is a merge source).
        split_merge = ""
        if before_id and before_id in split_sources:
            split_merge = f"split candidate -> {split_sources[before_id]['target_route_ids']}"
        elif after_id and after_id in merge_sources:
            split_merge = f"merge candidate <- {merge_sources[after_id]['target_route_ids']}"

        flags: dict[str, object] = {
            "status": "eliminated"
            if status == "eliminated"
            else "added"
            if status == "added"
            else "present",
            "rekeyed": rekeyed,
            "route_id_before": before_id,
            "route_id_after": after_id,
            "alignment_change": align_tier,
            "alignment_detail": align_detail,
            "schedule_change": sched_tier,
            "schedule_detail": sched_detail,
            "reblocked": reblocked,
            "name_change": short_changed or long_changed,
            "name_detail": name_detail,
            "daytype_change": daytype_change,
            "fare_change": fare_change,
            "attr_changes": ";".join(attr_changed),
            "split_merge": split_merge,
        }
        headline, summary = summarize_change(flags)

        def _name(row: Optional[pd.Series], key: str) -> str:
            return str(row.get(key, "")).strip() if row is not None else ""

        rows.append(
            {
                "route_id": after_id or before_id,
                "route_id_before": before_id,
                "route_id_after": after_id,
                "route_short_name": _name(name_row, "route_short_name"),
                "route_long_name": _name(name_row, "route_long_name"),
                "headline_change": headline,
                "change_summary": summary,
                "rekeyed": rekeyed,
                "alignment_change": align_tier,
                "schedule_change": sched_tier,
                "reblocked": reblocked,
                "name_change": short_changed or long_changed,
                "attr_changes": ";".join(attr_changed),
                "daytype_change": daytype_change,
                "fare_change": fare_change,
                "split_merge": split_merge,
                "stop_jaccard": _round(jaccard(sb, sa)) if status == "present" else "",
                "n_stops_before": len(sb) if before_id else "",
                "n_stops_after": len(sa) if after_id else "",
                "alignment_detail": align_detail,
                "schedule_detail": sched_detail,
            }
        )

    overview = pd.DataFrame(rows)
    if not overview.empty:
        overview = overview.sort_values(
            ["headline_change", "route_short_name", "route_id"]
        ).reset_index(drop=True)

    # Derived sub-tables (views onto the overview).
    def _subset(mask: pd.Series) -> pd.DataFrame:
        return overview.loc[mask].reset_index(drop=True) if not overview.empty else overview

    eliminated_df = (
        _subset(overview["headline_change"] == "eliminated") if not overview.empty else overview
    )
    added_df = _subset(overview["headline_change"] == "added") if not overview.empty else overview
    alignment_df = (
        _subset(overview["alignment_change"].isin(["minor", "major"]))
        if not overview.empty
        else overview
    )
    schedule_df = (
        _subset(overview["schedule_change"].isin(["minor", "major"]))
        if not overview.empty
        else overview
    )
    reblock_df = _subset(overview["reblocked"]) if not overview.empty else overview
    name_attr_df = (
        _subset(overview["name_change"] | (overview["attr_changes"].astype(str).str.len() > 0))
        if not overview.empty
        else overview
    )
    daytype_df = (
        _subset(overview["daytype_change"].astype(str).str.len() > 0)
        if not overview.empty
        else overview
    )

    calendar_changes = compare_calendars(before.get("calendar"), after.get("calendar"))
    fare_attr_changes = compare_fare_attributes(
        before.get("fare_attributes"), after.get("fare_attributes"), knobs["fare_price_major"]
    )
    route_fare_changes = _subset(overview["fare_change"]) if not overview.empty else overview

    warn_on_date_discrepancy(before, "before")
    warn_on_date_discrepancy(after, "after")
    range_b = feed_date_range(before, BEFORE_DATE_RANGE)
    range_a = feed_date_range(after, AFTER_DATE_RANGE)
    rel = date_range_relationship(range_b, range_a)
    logging.info("Date range before=%s after=%s | %s", range_b[:2], range_a[:2], rel)

    def _count(df: pd.DataFrame, col: str, value: str) -> int:
        return int((df[col] == value).sum()) if not df.empty and col in df.columns else 0

    unchanged = _count(overview, "headline_change", "unchanged")
    summary = Summary(
        before_route_count=len(routes_b_idx),
        after_route_count=len(routes_a_idx),
        matched_count=len(corr.matched),
        rekeyed_count=len(corr.rekeyed),
        eliminated_count=len(corr.eliminated),
        added_count=len(corr.added),
        alignment_major_count=_count(overview, "alignment_change", "major"),
        alignment_minor_count=_count(overview, "alignment_change", "minor"),
        schedule_major_count=_count(overview, "schedule_change", "major"),
        schedule_minor_count=_count(overview, "schedule_change", "minor"),
        reblocked_count=int(overview["reblocked"].sum()) if not overview.empty else 0,
        name_changed_count=int(overview["name_change"].sum()) if not overview.empty else 0,
        attr_changed_count=(
            int((overview["attr_changes"].astype(str).str.len() > 0).sum())
            if not overview.empty
            else 0
        ),
        daytype_changed_count=len(daytype_df),
        fare_changed_count=len(route_fare_changes),
        split_merge_candidate_count=len(split_merge_df),
        unchanged_count=unchanged,
        before_date_range=f"{range_b[0]}-{range_b[1]} ({range_b[2]})",
        after_date_range=f"{range_a[0]}-{range_a[1]} ({range_a[2]})",
        date_range_relationship=rel,
    )

    return {
        "overview": overview,
        "eliminated": eliminated_df,
        "added": added_df,
        "alignment": alignment_df,
        "schedule": schedule_df,
        "reblocked": reblock_df,
        "name_attr": name_attr_df,
        "daytype": daytype_df,
        "split_merge": split_merge_df,
        "calendar_changes": calendar_changes,
        "fare_attribute_changes": fare_attr_changes,
        "route_fare_changes": route_fare_changes,
        "summary": summary,
    }


# =============================================================================
# Export
# =============================================================================

_CSV_NAMES: dict[str, str] = {
    "overview": "routes_overview.csv",
    "eliminated": "routes_eliminated.csv",
    "added": "routes_added.csv",
    "alignment": "routes_alignment_changes.csv",
    "schedule": "routes_schedule_changes.csv",
    "reblocked": "routes_reblocked.csv",
    "name_attr": "routes_name_attr_changes.csv",
    "daytype": "routes_daytype_changes.csv",
    "split_merge": "routes_split_merge_candidates.csv",
    "calendar_changes": "calendar_changes.csv",
    "fare_attribute_changes": "fare_attribute_changes.csv",
    "route_fare_changes": "route_fare_changes.csv",
}

_SHEET_NAMES: dict[str, str] = {
    "overview": "overview",
    "eliminated": "eliminated",
    "added": "added",
    "alignment": "alignment",
    "schedule": "schedule",
    "reblocked": "reblocked",
    "name_attr": "name_attr",
    "daytype": "daytype",
    "split_merge": "split_merge",
    "calendar_changes": "calendar",
    "fare_attribute_changes": "fare_attrs",
    "route_fare_changes": "route_fares",
}


def write_outputs(output_dir: Path, results: dict[str, object]) -> None:
    """Write all CSVs, the Excel workbook, and ``summary.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Summary = results["summary"]  # type: ignore[assignment]

    for key, filename in _CSV_NAMES.items():
        frame: pd.DataFrame = results[key]  # type: ignore[assignment]
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8")
        logging.info("Wrote: %s (%d rows)", path, len(frame))

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)
    logging.info("Wrote: %s", summary_path)

    xlsx_path = output_dir / "routes_comparison.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for key, sheet in _SHEET_NAMES.items():
            frame = results[key]  # type: ignore[assignment]
            frame.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame([asdict(summary)]).to_excel(writer, sheet_name="summary", index=False)
    logging.info("Wrote: %s", xlsx_path)


# =============================================================================
# Notebook-friendly entry point
# =============================================================================


def _knobs() -> dict[str, float]:
    """Collect the threshold knobs from the CONFIG block into one mapping."""
    return {
        "rekey_jaccard": REKEY_MIN_JACCARD,
        "align_minor_jaccard": ALIGN_MINOR_JACCARD,
        "trips_major": SCHED_TRIPS_PCT_MAJOR,
        "span_major": SCHED_SPAN_MIN_MAJOR,
        "headway_major": SCHED_HEADWAY_PCT_MAJOR,
        "fare_price_major": FARE_PRICE_MAJOR_DELTA,
        "split_merge_enable": 1.0 if SPLIT_MERGE_ENABLE else 0.0,
        "split_merge_min_coverage": SPLIT_MERGE_MIN_COVERAGE,
        "split_merge_min_share": SPLIT_MERGE_MIN_SHARE,
    }


def run_compare(
    before_dir: Path = BEFORE_GTFS_DIR,
    after_dir: Path = AFTER_GTFS_DIR,
    out_dir: Path = OUTPUT_DIR,
    weekday: str = ANALYSIS_WEEKDAY,
    knobs: Optional[dict[str, float]] = None,
) -> Summary:
    """Run the route comparison (notebook-friendly) and write outputs."""
    setup_logging(out_dir)
    logging.info("Before GTFS: %s", before_dir)
    logging.info("After GTFS:  %s", after_dir)
    logging.info("Output dir:  %s", out_dir)

    before = load_feed(Path(before_dir), label="before")
    after = load_feed(Path(after_dir), label="after")

    results = compare_routes(before, after, weekday=weekday, knobs=knobs or _knobs())
    write_outputs(Path(out_dir), results)

    summary: Summary = results["summary"]  # type: ignore[assignment]
    logging.info(
        "Done. eliminated=%d added=%d rekeyed=%d | alignment major/minor=%d/%d | "
        "schedule major/minor=%d/%d | reblocked=%d | unchanged=%d",
        summary.eliminated_count,
        summary.added_count,
        summary.rekeyed_count,
        summary.alignment_major_count,
        summary.alignment_minor_count,
        summary.schedule_major_count,
        summary.schedule_minor_count,
        summary.reblocked_count,
        summary.unchanged_count,
    )
    return summary


# =============================================================================
# CLI (notebook-safe)
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args and return ``(args, unknown_args)``."""
    parser = argparse.ArgumentParser(description="Compare GTFS routes between two feeds.")
    parser.add_argument(
        "--before", type=Path, default=BEFORE_GTFS_DIR, help="Old/before GTFS folder"
    )
    parser.add_argument("--after", type=Path, default=AFTER_GTFS_DIR, help="New/after GTFS folder")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--weekday", default=ANALYSIS_WEEKDAY, help="Representative weekday (monday..friday)"
    )
    parser.add_argument("--rekey-jaccard", type=float, default=REKEY_MIN_JACCARD)
    parser.add_argument("--align-minor-jaccard", type=float, default=ALIGN_MINOR_JACCARD)
    parser.add_argument("--sched-trips-major", type=float, default=SCHED_TRIPS_PCT_MAJOR)
    parser.add_argument("--sched-span-major", type=float, default=SCHED_SPAN_MIN_MAJOR)
    parser.add_argument("--sched-headway-major", type=float, default=SCHED_HEADWAY_PCT_MAJOR)
    parser.add_argument("--fare-price-major", type=float, default=FARE_PRICE_MAJOR_DELTA)
    parser.add_argument(
        "--no-split-merge", action="store_true", help="Disable split/merge candidate detection"
    )
    parser.add_argument("--split-merge-min-coverage", type=float, default=SPLIT_MERGE_MIN_COVERAGE)
    parser.add_argument("--split-merge-min-share", type=float, default=SPLIT_MERGE_MIN_SHARE)
    args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    return args, unknown


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point (notebook-safe)."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if (
        _PLACEHOLDER in str(BEFORE_GTFS_DIR)
        or _PLACEHOLDER in str(AFTER_GTFS_DIR)
        or _PLACEHOLDER in str(OUTPUT_DIR)
    ):
        logging.warning(
            "BEFORE_GTFS_DIR / AFTER_GTFS_DIR / OUTPUT_DIR are still placeholders. Update the "
            "CONFIG block or pass --before/--after/--out before running."
        )
        return

    args, _unknown = parse_args(argv)
    knobs = {
        "rekey_jaccard": args.rekey_jaccard,
        "align_minor_jaccard": args.align_minor_jaccard,
        "trips_major": args.sched_trips_major,
        "span_major": args.sched_span_major,
        "headway_major": args.sched_headway_major,
        "fare_price_major": args.fare_price_major,
        "split_merge_enable": 0.0 if args.no_split_merge else 1.0,
        "split_merge_min_coverage": args.split_merge_min_coverage,
        "split_merge_min_share": args.split_merge_min_share,
    }
    run_compare(
        before_dir=args.before,
        after_dir=args.after,
        out_dir=args.out,
        weekday=args.weekday,
        knobs=knobs,
    )
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
