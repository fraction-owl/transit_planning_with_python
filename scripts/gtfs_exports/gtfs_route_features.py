"""Compute route-level GTFS supply & competition features (PART A feature generator).

This is a feature-generation script for the cross-sectional route boardings model
(engine 1). It reads a GTFS feed directly — no arcpy, no geospatial deps beyond a
haversine — and emits one row per route, keyed so the result joins straight onto the
NTD boardings anchor consumed by ``route_performance_model.py`` (PART B).

Where this sits in the pipeline:

    PART A (this script + others)   ->  route-keyed CSV of non-NTD features
    prep_features.py (orchestrator) ->  bundles the CSV(s) + writes manifest.json
    route_performance_model.py (PART B) ->  joins the bundle onto the NTD anchor, fits OLS

Features produced (one row per route, keyed on ``route_id`` = GTFS route_short_name):

    Supply (the actionable levers + scale):
        trips_per_day         trips on the analysis weekday
        revenue_hours         sum of trip in-service runtimes (hours)
        span_hours            first departure to last arrival (hours)
        median_headway_min    median gap between consecutive trip starts (route-level, all-day)
        pct_day_with_service  percent of the day's 24 one-hour bins with a trip departure
        revenue_miles         daily total revenue miles (sum of trip shape lengths)
        route_length_mi       representative one-way length (longest shape the route uses)
        route_length_modal_mi one-way length of the modal shape variant (shape most trips run)
        avg_speed_mph         revenue_miles / revenue_hours

    Network structure / redundancy:
        n_stops                          distinct stops served by the route
        stops_per_mile                   n_stops / route_length_mi
        shared_stop_share                fraction of the route's stops also served by 2+ routes
        n_competitor_routes              distinct other routes sharing >=1 stop
        competitor_trips_at_shared_stops summed competitor trip pressure at shared stops
        competition_intensity            competitor trips at shared stops / this route's trips

PHASE 2 (deferred to a separate spatial feature script): WMATA cross-agency competition
and Metrorail proximity both need an external feed and spatial stop matching, so they
belong with the geospatial features, not in this pure-GTFS extractor.

Caveats worth knowing before you read the coefficients:
    - median_headway_min is a coarse all-day, all-direction route-level median; it mixes
      directions and counts layover gaps below 4 h. Peak/by-direction headway is a later
      refinement, not this number.
    - pct_day_with_service is the more robust service-span measure: the share of the day's
      24 one-hour bins in which at least one trip *departs*, x100. Counting departures (not
      whole operating intervals) mirrors what a rider waiting at a stop experiences -- how
      often a boardable vehicle shows up, not that some bus is mid-route elsewhere -- so a
      long through-running trip credits only its start bin. Bins are one hour wide on
      purpose: a route running at least hourly counts as serving the whole hour, so this
      stays a span/availability measure and leaves finer frequency (30- vs 60-minute
      service) to median_headway_min rather than docking the route for the empty half of
      each hour. Unlike that headway it is direction-agnostic and counts every
      midday/overnight gap rather than ignoring it. Late trips keep the GTFS extended clock
      (25:xx, 26:xx are not wrapped to the next day), matching span_hours / runtimes; the
      denominator is a nominal 24 h (24 bins).
    - revenue_miles is a daily total (a supply *quantity*); route_length_mi is the
      one-way extent used for stops_per_mile. They are deliberately different units.
    - route_length_mi is the route's longest shape (its full extent); route_length_modal_mi
      is the length of the shape variant the most trips run (the *typical* one-way trip).
      They diverge when a route has short-turns or occasional extensions. stops_per_mile
      still pairs with route_length_mi (the all-shapes distinct-stop count needs the extent).

Inputs:
    A GTFS feed folder containing routes/trips/calendar/stop_times/shapes (.txt).

Outputs:
    OUTPUT_DIR / OUTPUT_CSV_NAME : the route-keyed feature table.
    A run-log sidecar capturing the verbatim config block, feed SHA-256, and the
    analysis date that was actually selected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import sys
from pathlib import Path
from typing import Final, Optional, Sequence

import numpy as np
import pandas as pd

# Sentinel markers bounding the configuration block for the run log. Each must
# appear exactly once below as a stand-alone comment line.
CONFIG_BEGIN_MARKER: Final[str] = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: Final[str] = "# === END CONFIG ==="


# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Root folder for this run's inputs. Standalone, point it at the analysis
# request's input folder; under the orchestrator, this is the directory
# prep_features drops the feed/data into. Phase-2 inputs (WMATA feed, Metro
# shapefile, service-type file) live here as siblings of the GTFS feed.
INPUT_DIR: Final[Path] = Path(r"Path\To\Your\Input_Folder")  # <<< EDIT ME

# GTFS feed folder. Resolved under INPUT_DIR by default; repoint to an absolute
# path (e.g. the shared G:\ data drive) if the feed isn't copied per request.
GTFS_DIR: Final[Path] = INPUT_DIR / "your_gtfs_name"  # <<< EDIT ME

# Weekday to characterize (monday..friday). The script picks the date with the
# most active weekday service_ids so a holiday-skewed date isn't used.
ANALYSIS_WEEKDAY: Final[str] = "tuesday"  # <<< EDIT ME

# Where the feature CSV and run log are written (the bundle dir PART B reads).
OUTPUT_DIR: Final[Path] = Path(r"Path\To\Your\prepped_features")  # <<< EDIT ME
OUTPUT_CSV_NAME: Final[str] = "gtfs_route_features.csv"

# Name of the join-key column written to the output. The NTD anchor in PART B
# keys on "route_id" whose values are public route numbers (e.g. 101), which is
# GTFS route_short_name — so we surface route_short_name under this name.
ROUTE_KEY_OUT: Final[str] = "route_id"

# stop_times.txt is read in chunks to bound memory on large feeds.
STOP_TIMES_CHUNKSIZE: Final[int] = 1_500_000

# Write a run-log sidecar next to the output CSV.
WRITE_RUN_LOG: Final[bool] = True

LOG_LEVEL: int = logging.INFO

# === END CONFIG ===


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def _canonical_key(series: pd.Series) -> pd.Series:
    """Normalize a join-key column so this output matches the NTD anchor reliably.

    Kept BYTE-IDENTICAL to the copy in route_performance_model.py / prep_features.py:
    collapse to a trimmed string and strip a single trailing ``.0`` so an integer
    that survived a float round-trip still matches its string form.
    """
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.fillna("")


def _gtfs_path(gtfs_dir: Path, name: str) -> Path:
    """Return the path to a required GTFS file, raising if it is missing."""
    path = gtfs_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing GTFS file: {path}")
    return path


def _read_gtfs_csv(path: Path, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a GTFS table as all-string columns (IDs are categorical, never numeric)."""
    return pd.read_csv(path, dtype=str, usecols=usecols, encoding="utf-8-sig", low_memory=False)


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# CALENDAR / SERVICE-ID SELECTION
# =============================================================================

_WEEKDAY_COLS: Final[tuple[str, ...]] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _weekday_name_to_int(name: str) -> int:
    """Map a weekday name to a Python weekday index (Monday=0)."""
    key = name.strip().lower()
    if key not in _WEEKDAY_COLS or _WEEKDAY_COLS.index(key) >= 5:
        raise ValueError(f"ANALYSIS_WEEKDAY must be monday..friday, got {name!r}.")
    return _WEEKDAY_COLS.index(key)


def _parse_yyyymmdd(value: str) -> dt.date:
    return dt.date(int(value[0:4]), int(value[4:6]), int(value[6:8]))


def _weekday_only_service_ids(calendar: pd.DataFrame) -> set[str]:
    """Service_ids that run on >=1 weekday and on neither weekend day."""
    cal = calendar.copy()
    for col in _WEEKDAY_COLS:
        cal[col] = cal[col].fillna("0").astype(str).str.strip()
    is_weekday = (
        (cal["monday"] == "1")
        | (cal["tuesday"] == "1")
        | (cal["wednesday"] == "1")
        | (cal["thursday"] == "1")
        | (cal["friday"] == "1")
    )
    is_not_weekend = (cal["saturday"] == "0") & (cal["sunday"] == "0")
    return set(cal.loc[is_weekday & is_not_weekend, "service_id"].dropna().astype(str))


def _active_service_ids_for_date(
    calendar: pd.DataFrame,
    calendar_dates: Optional[pd.DataFrame],
    analysis_date: dt.date,
    weekday_service_ids: set[str],
) -> set[str]:
    """Resolve the service_ids active on a date, applying calendar_dates exceptions."""
    ds = analysis_date.strftime("%Y%m%d")
    col = _WEEKDAY_COLS[analysis_date.weekday()]

    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(str).str.strip()
    cal["end_date"] = cal["end_date"].astype(str).str.strip()
    cal[col] = cal[col].fillna("0").astype(str).str.strip()

    active = set(
        cal[(cal["start_date"] <= ds) & (cal["end_date"] >= ds) & (cal[col] == "1")]["service_id"]
        .dropna()
        .astype(str)
    )

    if calendar_dates is not None and not calendar_dates.empty:
        cd = calendar_dates.copy()
        cd["date"] = cd["date"].astype(str).str.strip()
        cd = cd[cd["date"] == ds]
        if not cd.empty:
            cd["exception_type"] = cd["exception_type"].astype(str).str.strip()
            active |= set(cd[cd["exception_type"] == "1"]["service_id"].dropna().astype(str))
            active -= set(cd[cd["exception_type"] == "2"]["service_id"].dropna().astype(str))

    return active & weekday_service_ids


def choose_analysis_date_and_services(
    calendar: pd.DataFrame,
    calendar_dates: Optional[pd.DataFrame],
    preferred_weekday: str,
    max_search_days: int = 366,
) -> tuple[dt.date, set[str]]:
    """Pick the in-range date of the preferred weekday with the most active service.

    Scans outward from today (clamped to the feed's date range) and returns the
    first date whose active weekday service_ids equal the full weekday set, or the
    best date found otherwise.
    """
    pref_int = _weekday_name_to_int(preferred_weekday)

    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(str).str.strip()
    cal["end_date"] = cal["end_date"].astype(str).str.strip()
    min_start = min(_parse_yyyymmdd(x) for x in cal["start_date"].dropna())
    max_end = max(_parse_yyyymmdd(x) for x in cal["end_date"].dropna())

    anchor = max(min_start, min(dt.date.today(), max_end))
    weekday_sids = _weekday_only_service_ids(cal)
    if not weekday_sids:
        raise ValueError("No weekday-only service_ids found in calendar.txt.")

    best_date: Optional[dt.date] = None
    best_active: set[str] = set()
    for step in range(0, max_search_days + 1):
        for sign in (1, -1):
            if step == 0 and sign == -1:
                continue
            cand = anchor + dt.timedelta(days=step * sign)
            if cand < min_start or cand > max_end or cand.weekday() != pref_int:
                continue
            active = _active_service_ids_for_date(cal, calendar_dates, cand, weekday_sids)
            if len(active) > len(best_active):
                best_date, best_active = cand, active
            if active and len(active) == len(weekday_sids):
                return cand, active

    if best_date is None or not best_active:
        raise ValueError(f"No active {preferred_weekday} service found within the feed date range.")
    return best_date, best_active


# =============================================================================
# GTFS GEOMETRY / TIME
# =============================================================================


def _haversine_m(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized great-circle distance in meters."""
    radius = 6_371_000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert GTFS HH:MM:SS strings (hours may exceed 24) to seconds since midnight."""
    parts = series.astype(str).str.split(":", expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    seconds = pd.to_numeric(parts[2], errors="coerce") if parts.shape[1] >= 3 else 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds


def load_stop_times_filtered(
    stop_times_path: Path, trip_ids: set[str], chunksize: int
) -> pd.DataFrame:
    """Stream stop_times.txt, keeping only rows for the in-scope trips."""
    usecols = ["trip_id", "arrival_time", "departure_time", "stop_sequence", "stop_id"]
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        stop_times_path,
        dtype=str,
        usecols=usecols,
        encoding="utf-8-sig",
        low_memory=False,
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["trip_id"].astype(str).isin(trip_ids)]
        if not chunk.empty:
            kept.append(chunk)
    if not kept:
        return pd.DataFrame(columns=usecols)
    return pd.concat(kept, ignore_index=True)


def compute_shape_lengths(shapes_path: Path, shape_ids: set[str]) -> pd.DataFrame:
    """Haversine length (meters) of each in-scope shape -> [shape_id, shape_len_m]."""
    usecols = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    df = _read_gtfs_csv(shapes_path, usecols=usecols)
    df = df[df["shape_id"].astype(str).isin(shape_ids)].copy()
    if df.empty:
        return pd.DataFrame(columns=["shape_id", "shape_len_m"])

    df["shape_id"] = df["shape_id"].astype(str)
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values(["shape_id", "shape_pt_sequence"]).reset_index(drop=True)

    df["prev_lat"] = df.groupby("shape_id")["shape_pt_lat"].shift(1)
    df["prev_lon"] = df.groupby("shape_id")["shape_pt_lon"].shift(1)
    df = df.dropna(subset=["prev_lat", "prev_lon"])
    df["seg_m"] = _haversine_m(
        df["prev_lat"].to_numpy(),
        df["prev_lon"].to_numpy(),
        df["shape_pt_lat"].to_numpy(),
        df["shape_pt_lon"].to_numpy(),
    )
    return df.groupby("shape_id", dropna=False)["seg_m"].sum().rename("shape_len_m").reset_index()


def compute_trip_start_end(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Per trip: first-stop start, last-stop end, and runtime (seconds)."""
    if stop_times.empty:
        return pd.DataFrame(columns=["trip_id", "start_sec", "end_sec", "runtime_sec"])

    st = stop_times.copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"])
    # Prefer departure_time, fall back to arrival_time for the time at each stop.
    dep = st["departure_time"].where(
        st["departure_time"].notna() & (st["departure_time"].astype(str).str.len() > 0)
    )
    arr = st["arrival_time"].where(
        st["arrival_time"].notna() & (st["arrival_time"].astype(str).str.len() > 0)
    )
    st["time_sec"] = _time_to_seconds(dep.fillna(arr))

    idx_first = st.groupby("trip_id")["stop_sequence"].idxmin()
    idx_last = st.groupby("trip_id")["stop_sequence"].idxmax()
    first = st.loc[idx_first, ["trip_id", "time_sec"]].rename(columns={"time_sec": "start_sec"})
    last = st.loc[idx_last, ["trip_id", "time_sec"]].rename(columns={"time_sec": "end_sec"})

    out = first.merge(last, on="trip_id", how="inner")
    out["runtime_sec"] = out["end_sec"] - out["start_sec"]
    out.loc[out["runtime_sec"] < 0, "runtime_sec"] = np.nan  # guard clock anomalies
    return out[["trip_id", "start_sec", "end_sec", "runtime_sec"]]


# =============================================================================
# ROUTE-LEVEL FEATURES (keyed on GTFS route_id)
# =============================================================================


def _median_headway_min(start_secs: pd.Series) -> float:
    """Median gap (minutes) between consecutive trip starts; NaN if < 3 trips.

    Coarse and all-day: mixes directions and ignores gaps >= 4 h (layovers / the
    overnight break). A by-direction, peak-windowed headway is a later refinement.
    """
    times = np.sort(start_secs.dropna().to_numpy(dtype=float))
    if times.size < 3:
        return float("nan")
    diffs = np.diff(times)
    diffs = diffs[(diffs > 0) & (diffs < 4 * 3600)]
    return float(np.median(diffs) / 60.0) if diffs.size else float("nan")


def _service_day_coverage(trip_times: pd.DataFrame, bins_per_day: int = 24) -> pd.DataFrame:
    """Per route: percent of the day (0-100) with a trip departure.

    The day is split into ``bins_per_day`` equal bins (24 -> one-hour bins). A bin
    counts as served if at least one of the route's trips *departs* within it; served
    bins are unioned across the route's trips and divided by ``bins_per_day``. Counting
    departures -- rather than whole operating intervals -- mirrors the service a rider
    waiting at a stop actually experiences: what matters is how often a boardable vehicle
    shows up, not that some bus is mid-route elsewhere. A long trip therefore credits only
    the bin it starts in. Late-night trips keep the GTFS extended clock (e.g. a 25:30
    departure falls in bin 25, not wrapped back to 01:30), matching how span_hours and
    runtimes treat after-midnight times here.

    Bins are one hour wide on purpose: a route running at least hourly then fills every
    hour it operates, so this stays a span / availability measure. Finer frequency (e.g.
    30- vs 60-minute service) is a headway concern, carried by median_headway_min, not a
    penalty here -- a narrower bin would dock an hourly route for the empty half of each
    hour. A more robust alternative to median_headway_min all the same: direction-agnostic,
    and it explicitly penalizes midday / overnight gaps instead of ignoring gaps above a
    threshold.

    Returns ``[route_id, pct_day_with_service]``; routes with no usable trip start times
    are omitted so callers can left-merge and leave them NaN.
    """
    cols = ["route_id", "pct_day_with_service"]
    df = trip_times.dropna(subset=["start_sec"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    bin_sec = 86_400.0 / bins_per_day
    bin_idx = np.floor(df["start_sec"].to_numpy(dtype=float) / bin_sec).astype(np.int64)

    served: dict[str, set[int]] = {}
    for route_id, b in zip(df["route_id"].to_numpy(), bin_idx):
        served.setdefault(route_id, set()).add(int(b))

    rows = [(rid, 100.0 * len(bins) / bins_per_day) for rid, bins in served.items()]
    return pd.DataFrame(rows, columns=cols)


def compute_route_supply_metrics(
    trips: pd.DataFrame, stop_times: pd.DataFrame, shape_len: pd.DataFrame
) -> pd.DataFrame:
    """Route-level supply metrics keyed on GTFS ``route_id``."""
    if trips.empty:
        raise ValueError("No in-scope trips to summarize.")

    trips_per_day = (
        trips.groupby("route_id", dropna=False)["trip_id"]
        .count()
        .rename("trips_per_day")
        .reset_index()
    )

    tse = compute_trip_start_end(stop_times)
    trips_t = trips.merge(tse, on="trip_id", how="left")

    agg = (
        trips_t.groupby("route_id", dropna=False)
        .agg(
            revenue_hours=(
                "runtime_sec",
                lambda s: float(np.nansum(s.to_numpy(dtype=float)) / 3600.0),
            ),
            min_start_sec=(
                "start_sec",
                lambda s: (
                    float(np.nanmin(s.to_numpy(dtype=float)))
                    if np.isfinite(s.to_numpy(dtype=float)).any()
                    else float("nan")
                ),
            ),
            max_end_sec=(
                "end_sec",
                lambda s: (
                    float(np.nanmax(s.to_numpy(dtype=float)))
                    if np.isfinite(s.to_numpy(dtype=float)).any()
                    else float("nan")
                ),
            ),
            median_headway_min=("start_sec", _median_headway_min),
        )
        .reset_index()
    )
    agg["span_hours"] = (agg["max_end_sec"] - agg["min_start_sec"]) / 3600.0

    out = trips_per_day.merge(agg, on="route_id", how="left")

    coverage = _service_day_coverage(trips_t[["route_id", "start_sec"]])
    out = out.merge(coverage, on="route_id", how="left")

    if not shape_len.empty:
        sl = shape_len.copy()
        sl["shape_id"] = sl["shape_id"].astype(str)
        trips_s = trips.merge(sl, on="shape_id", how="left")
        # Daily total revenue miles = sum over trips of the trip's shape length.
        rev_miles = (
            trips_s.groupby("route_id", dropna=False)["shape_len_m"]
            .sum(min_count=1)
            .rename("revenue_miles_m")
            .reset_index()
        )
        # Representative one-way extent = the longest shape the route operates.
        route_len = (
            trips_s.groupby("route_id", dropna=False)["shape_len_m"]
            .max()
            .rename("route_length_m")
            .reset_index()
        )
        # Modal shape variant = the shape the most trips run (the *typical* one-way trip).
        # Ties break toward the longer shape so the fuller variant wins a 50/50 split; we
        # take the first row per route after sorting (not groupby.first(), which would skip
        # a NaN length and return a *different* shape's length).
        shape_counts = (
            trips_s.groupby(["route_id", "shape_id"], dropna=False)
            .agg(n_trips=("trip_id", "count"), shape_len_m=("shape_len_m", "first"))
            .reset_index()
            .sort_values(
                ["route_id", "n_trips", "shape_len_m"],
                ascending=[True, False, False],
                na_position="last",
            )
        )
        modal_len = shape_counts.drop_duplicates(subset="route_id", keep="first")[
            ["route_id", "shape_len_m"]
        ].rename(columns={"shape_len_m": "route_length_modal_m"})
        out = (
            out.merge(rev_miles, on="route_id", how="left")
            .merge(route_len, on="route_id", how="left")
            .merge(modal_len, on="route_id", how="left")
        )
        out["revenue_miles"] = out["revenue_miles_m"] / 1609.344
        out["route_length_mi"] = out["route_length_m"] / 1609.344
        out["route_length_modal_mi"] = out["route_length_modal_m"] / 1609.344
        out = out.drop(columns=["revenue_miles_m", "route_length_m", "route_length_modal_m"])
        out["avg_speed_mph"] = out["revenue_miles"] / out["revenue_hours"].replace(0, np.nan)
    else:
        out["revenue_miles"] = np.nan
        out["route_length_mi"] = np.nan
        out["route_length_modal_mi"] = np.nan
        out["avg_speed_mph"] = np.nan

    return out


def compute_competition_metrics(
    trips: pd.DataFrame, stop_times: pd.DataFrame, supply: pd.DataFrame
) -> pd.DataFrame:
    """Intra-agency shared-stop & competition metrics keyed on GTFS ``route_id``.

    A stop is "shared" for a route if it is also served by >= 1 other route. Competitor
    pressure at a shared stop is each other route's trips spread across the stops it
    serves, summed over the focal route's shared stops; intensity normalizes by the
    focal route's own trips.
    """
    cols = [
        "n_stops",
        "shared_stop_share",
        "n_competitor_routes",
        "competitor_trips_at_shared_stops",
        "competition_intensity",
    ]
    if stop_times.empty:
        base = supply[["route_id"]].copy()
        for c in cols:
            base[c] = np.nan
        return base

    st = stop_times[["trip_id", "stop_id"]].astype(str)
    tr = trips[["route_id", "trip_id"]].astype(str)
    route_stops = tr.merge(st, on="trip_id", how="inner").dropna(subset=["stop_id"])
    route_stops = route_stops[["route_id", "stop_id"]].drop_duplicates()

    stop_route_counts = (
        route_stops.groupby("stop_id").size().rename("n_routes_serving").reset_index()
    )
    rsm = route_stops.merge(stop_route_counts, on="stop_id", how="left")
    rsm["is_shared"] = rsm["n_routes_serving"] >= 2

    basic = (
        rsm.groupby("route_id", dropna=False)
        .agg(
            n_stops=("stop_id", "count"),
            shared_stop_share=("is_shared", "mean"),
        )
        .reset_index()
    )

    shared = rsm[rsm["is_shared"]][["route_id", "stop_id"]]
    if shared.empty:
        for c in [
            "n_competitor_routes",
            "competitor_trips_at_shared_stops",
            "competition_intensity",
        ]:
            basic[c] = 0.0
        return basic

    pairs = shared.merge(
        route_stops.rename(columns={"route_id": "route_j"}), on="stop_id", how="inner"
    )
    pairs = pairs[pairs["route_id"] != pairs["route_j"]]

    n_competitors = (
        pairs.groupby("route_id", dropna=False)["route_j"]
        .nunique()
        .rename("n_competitor_routes")
        .reset_index()
    )

    sup = supply[["route_id", "trips_per_day"]].copy()
    sup["route_id"] = sup["route_id"].astype(str)
    comp_trips = sup.rename(columns={"route_id": "route_j", "trips_per_day": "comp_trips"})
    comp_stops = (
        route_stops.rename(columns={"route_id": "route_j"})
        .groupby("route_j", dropna=False)
        .size()
        .rename("comp_n_stops")
        .reset_index()
    )
    pairs = pairs.merge(comp_trips, on="route_j", how="left").merge(
        comp_stops, on="route_j", how="left"
    )
    pairs["comp_trips"] = pd.to_numeric(pairs["comp_trips"], errors="coerce").fillna(0.0)
    pairs["comp_n_stops"] = pairs["comp_n_stops"].replace(0, 1).fillna(1)
    pairs["comp_trips_per_stop"] = pairs["comp_trips"] / pairs["comp_n_stops"]

    pairs = pairs.drop_duplicates(subset=["route_id", "stop_id", "route_j"])
    weighted = (
        pairs.groupby("route_id", dropna=False)["comp_trips_per_stop"]
        .sum()
        .rename("competitor_trips_at_shared_stops")
        .reset_index()
    )

    out = basic.merge(n_competitors, on="route_id", how="left").merge(
        weighted, on="route_id", how="left"
    )
    out["n_competitor_routes"] = out["n_competitor_routes"].fillna(0)
    out["competitor_trips_at_shared_stops"] = out["competitor_trips_at_shared_stops"].fillna(0.0)

    focal = supply[["route_id", "trips_per_day"]].copy()
    focal["route_id"] = focal["route_id"].astype(str)
    out = out.merge(focal, on="route_id", how="left")
    out["competition_intensity"] = (
        out["competitor_trips_at_shared_stops"] / out["trips_per_day"].replace(0, np.nan)
    ).fillna(0.0)
    return out.drop(columns=["trips_per_day"])


# =============================================================================
# COLLAPSE GTFS route_id -> public route number (route_short_name)
# =============================================================================


def collapse_to_route_number(metrics: pd.DataFrame, route_col: str) -> pd.DataFrame:
    """Collapse per-GTFS-route_id metrics to one row per public route number.

    GTFS route_id is usually 1:1 with route_short_name for Connector, in which case
    this is a relabel. When a public number spans multiple route_ids, additive
    quantities are summed, extents/spans take the max, and rate-like metrics are
    recomputed from the aggregates (headway becomes a trips-weighted mean and is
    flagged as approximate). Collisions are logged so a real run reveals whether
    the approximation is even in play.
    """
    df = metrics.copy()
    multiplicity = df.groupby(route_col)["route_id"].nunique()
    collided = multiplicity[multiplicity > 1]
    if collided.empty:
        df = df.drop(columns=["route_id", "min_start_sec", "max_end_sec"], errors="ignore")
        return df.rename(columns={route_col: "route_id"})

    logging.warning(
        "%d public route number(s) map to multiple GTFS route_ids; aggregating "
        "(headway/share become weighted and approximate): %s",
        len(collided),
        sorted(collided.index.astype(str)),
    )

    # Weighting numerators, masking NaN sub-route values out of the weighted means.
    hw_mask = df["median_headway_min"].notna()
    df["_hw_num"] = np.where(hw_mask, df["median_headway_min"] * df["trips_per_day"], 0.0)
    df["_hw_w"] = np.where(hw_mask, df["trips_per_day"], 0.0)
    stops = pd.to_numeric(df["n_stops"], errors="coerce").fillna(0.0)
    df["_share_num"] = df["shared_stop_share"].fillna(0.0) * stops
    df["_share_w"] = stops
    # pct_day_with_service and route_length_modal_mi are rate-/typical-like, so they
    # collapse to a trips-weighted mean (approximate, like headway) rather than a sum/max.
    cov_mask = df["pct_day_with_service"].notna()
    df["_cov_num"] = np.where(cov_mask, df["pct_day_with_service"] * df["trips_per_day"], 0.0)
    df["_cov_w"] = np.where(cov_mask, df["trips_per_day"], 0.0)
    rlm_mask = df["route_length_modal_mi"].notna()
    df["_rlm_num"] = np.where(rlm_mask, df["route_length_modal_mi"] * df["trips_per_day"], 0.0)
    df["_rlm_w"] = np.where(rlm_mask, df["trips_per_day"], 0.0)

    g = df.groupby(route_col, dropna=False)
    agg = g.agg(
        trips_per_day=("trips_per_day", "sum"),
        revenue_hours=("revenue_hours", "sum"),
        revenue_miles=("revenue_miles", "sum"),
        route_length_mi=("route_length_mi", "max"),
        n_stops=("n_stops", "sum"),
        n_competitor_routes=("n_competitor_routes", "max"),
        competitor_trips_at_shared_stops=("competitor_trips_at_shared_stops", "sum"),
        min_start_sec=("min_start_sec", "min"),
        max_end_sec=("max_end_sec", "max"),
        _hw_num=("_hw_num", "sum"),
        _hw_w=("_hw_w", "sum"),
        _share_num=("_share_num", "sum"),
        _share_w=("_share_w", "sum"),
        _cov_num=("_cov_num", "sum"),
        _cov_w=("_cov_w", "sum"),
        _rlm_num=("_rlm_num", "sum"),
        _rlm_w=("_rlm_w", "sum"),
    )

    agg["span_hours"] = (agg["max_end_sec"] - agg["min_start_sec"]) / 3600.0
    agg["avg_speed_mph"] = agg["revenue_miles"] / agg["revenue_hours"].replace(0, np.nan)
    agg["median_headway_min"] = agg["_hw_num"] / agg["_hw_w"].replace(0, np.nan)
    agg["pct_day_with_service"] = agg["_cov_num"] / agg["_cov_w"].replace(0, np.nan)
    agg["route_length_modal_mi"] = agg["_rlm_num"] / agg["_rlm_w"].replace(0, np.nan)
    agg["shared_stop_share"] = agg["_share_num"] / agg["_share_w"].replace(0, np.nan)
    agg["competition_intensity"] = (
        agg["competitor_trips_at_shared_stops"] / agg["trips_per_day"].replace(0, np.nan)
    ).fillna(0.0)
    # Recompute from the aggregates like the other rate-like metrics; otherwise the
    # output schema would silently lose stops_per_mile only on collision. n_stops is
    # the summed (deliberately approximate) count, so a stop shared by two sub-routes
    # double-counts here — accepted because collisions are already logged as approximate.
    agg["stops_per_mile"] = agg["n_stops"] / agg["route_length_mi"].replace(0, np.nan)

    agg = agg.drop(
        columns=[
            "min_start_sec",
            "max_end_sec",
            "_hw_num",
            "_hw_w",
            "_share_num",
            "_share_w",
            "_cov_num",
            "_cov_w",
            "_rlm_num",
            "_rlm_w",
        ]
    )
    return agg.reset_index().rename(columns={route_col: "route_id"})


# =============================================================================
# RUN LOG
# =============================================================================


def _extract_config_block() -> str:
    """Return the verbatim text between the config sentinels in this source file."""
    try:
        source = Path(__file__)
    except NameError:  # interactive paste, no __file__
        return "(config block unavailable: not run from a source file)"
    lines = source.read_text(encoding="utf-8").splitlines()
    begin = end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin is None and stripped == CONFIG_BEGIN_MARKER:
            begin = i
        elif begin is not None and stripped == CONFIG_END_MARKER:
            end = i
            break
    if begin is None or end is None:
        return "(config block markers not found)"
    return "\n".join(lines[begin + 1 : end])


def write_run_log(
    output_dir: Path,
    feed_files: dict[str, str],
    analysis_date: dt.date,
    n_routes: int,
    gtfs_dir: Path = GTFS_DIR,
    analysis_weekday: str = ANALYSIS_WEEKDAY,
) -> None:
    """Write a run-log sidecar capturing config, feed provenance, and selected date."""
    log_path = output_dir / "gtfs_route_features_runlog.txt"
    provenance = [f"  {name}  sha256={digest}" for name, digest in sorted(feed_files.items())]
    lines = [
        "=" * 72,
        "GTFS ROUTE FEATURE EXTRACTION RUN LOG (PART A)",
        "=" * 72,
        f"Run timestamp:   {dt.datetime.now().isoformat(timespec='seconds')}",
        f"GTFS feed:       {gtfs_dir}",
        f"Analysis date:   {analysis_date.isoformat()} ({analysis_weekday})",
        f"Routes emitted:  {n_routes}",
        "",
        "-" * 72,
        "FEED FILE PROVENANCE (SHA-256)",
        "-" * 72,
        *provenance,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        _extract_config_block(),
        "=" * 72,
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Run log written to '%s'.", log_path)


# =============================================================================
# ENTRY POINT
# =============================================================================


def run(
    gtfs_dir: Path | None = None,
    output_dir: Path | None = None,
    analysis_weekday: str | None = None,
) -> Optional[pd.DataFrame]:
    """Build the route-level GTFS feature table and write it to ``output_dir``.

    Unset args fall back to the config block at the top of this file, so both
    ``m.GTFS_DIR = ...; m.run()`` after a plain import and the orchestrator's
    ``--gtfs-dir/--output-dir`` invocation work from the same body.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    gtfs_dir = GTFS_DIR if gtfs_dir is None else Path(gtfs_dir)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    analysis_weekday = ANALYSIS_WEEKDAY if analysis_weekday is None else analysis_weekday

    if "Path\\To\\Your" in str(gtfs_dir) or "Path\\To\\Your" in str(output_dir):
        logging.warning("Set GTFS_DIR and OUTPUT_DIR (marked '# <<< EDIT ME') before running.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    routes = _read_gtfs_csv(_gtfs_path(gtfs_dir, "routes.txt"))
    trips = _read_gtfs_csv(_gtfs_path(gtfs_dir, "trips.txt"))
    calendar = _read_gtfs_csv(_gtfs_path(gtfs_dir, "calendar.txt"))
    cd_path = gtfs_dir / "calendar_dates.txt"
    calendar_dates = _read_gtfs_csv(cd_path) if cd_path.exists() else None

    analysis_date, active_sids = choose_analysis_date_and_services(
        calendar, calendar_dates, analysis_weekday
    )
    logging.info("Analysis date %s | %d active service_ids.", analysis_date, len(active_sids))

    for col in ("route_id", "trip_id", "service_id", "shape_id"):
        if col not in trips.columns:
            raise ValueError(f"trips.txt missing required column: {col}")
    trips = trips[trips["service_id"].astype(str).isin(active_sids)].copy()
    for col in ("route_id", "trip_id", "shape_id"):
        trips[col] = trips[col].fillna("").astype(str)
    logging.info("Trips on analysis weekday: %d", len(trips))
    if trips.empty:
        raise ValueError("No trips after the weekday service filter.")

    trip_ids = set(trips["trip_id"])
    shape_ids = set(trips["shape_id"]) - {""}

    stop_times = load_stop_times_filtered(
        _gtfs_path(gtfs_dir, "stop_times.txt"), trip_ids, STOP_TIMES_CHUNKSIZE
    )
    logging.info("stop_times rows in scope: %d", len(stop_times))
    shape_len = compute_shape_lengths(_gtfs_path(gtfs_dir, "shapes.txt"), shape_ids)
    logging.info("Shapes measured: %d", len(shape_len))

    supply = compute_route_supply_metrics(trips, stop_times, shape_len)
    competition = compute_competition_metrics(trips, stop_times, supply)
    metrics = supply.merge(competition, on="route_id", how="left")
    metrics["stops_per_mile"] = pd.to_numeric(metrics["n_stops"], errors="coerce") / pd.to_numeric(
        metrics["route_length_mi"], errors="coerce"
    ).replace(0, np.nan)

    # Attach public route number (route_short_name) and collapse onto it.
    routes["route_id"] = routes["route_id"].astype(str)
    if "route_short_name" not in routes.columns:
        raise ValueError("routes.txt has no route_short_name to key the NTD anchor on.")
    metrics = metrics.merge(routes[["route_id", "route_short_name"]], on="route_id", how="left")
    unmatched = metrics["route_short_name"].isna().sum()
    if unmatched:
        logging.warning("%d route_id(s) had no route_short_name and will be dropped.", unmatched)
        metrics = metrics.dropna(subset=["route_short_name"])

    features = collapse_to_route_number(metrics, route_col="route_short_name")
    features["route_id"] = _canonical_key(features["route_id"])
    if ROUTE_KEY_OUT != "route_id":
        features = features.rename(columns={"route_id": ROUTE_KEY_OUT})

    ordered = [
        ROUTE_KEY_OUT,
        "trips_per_day",
        "revenue_hours",
        "span_hours",
        "median_headway_min",
        "pct_day_with_service",
        "revenue_miles",
        "route_length_mi",
        "route_length_modal_mi",
        "avg_speed_mph",
        "n_stops",
        "stops_per_mile",
        "shared_stop_share",
        "n_competitor_routes",
        "competitor_trips_at_shared_stops",
        "competition_intensity",
    ]
    features = features[[c for c in ordered if c in features.columns]]
    for col in features.columns:
        if col != ROUTE_KEY_OUT:
            features[col] = pd.to_numeric(features[col], errors="coerce").round(4)

    out_csv = output_dir / OUTPUT_CSV_NAME
    features.to_csv(out_csv, index=False)
    logging.info("Wrote %d routes x %d features to '%s'.", *features.shape, out_csv)

    if WRITE_RUN_LOG:
        feed_files = {
            name: _sha256_file(gtfs_dir / name)
            for name in ("routes.txt", "trips.txt", "calendar.txt", "stop_times.txt", "shapes.txt")
            if (gtfs_dir / name).exists()
        }
        write_run_log(
            output_dir, feed_files, analysis_date, len(features), gtfs_dir, analysis_weekday
        )

    logging.info("Done.")
    return features


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the configuration block.

    ``parse_known_args`` is used so a notebook kernel's injected argv (or the
    orchestrator's extra ``--input-dir`` token) does not raise ``SystemExit: 2``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute route-level GTFS supply & competition features (Part A feature "
            "generator). Defaults come from the CONFIGURATION block at the top of this file."
        )
    )
    parser.add_argument(
        "--gtfs-dir", type=Path, default=GTFS_DIR, help="Path to the GTFS feed folder."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory the feature CSV and run log are written to.",
    )
    parser.add_argument(
        "--analysis-weekday",
        default=ANALYSIS_WEEKDAY,
        help="Weekday to characterize (monday..friday).",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. Defaults fall back to the configuration block."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if "Path\\To\\Your" in str(args.gtfs_dir) or "Path\\To\\Your" in str(args.output_dir):
        logging.warning(
            "GTFS_DIR and/or OUTPUT_DIR are still placeholders. Update the CONFIGURATION "
            "block or pass --gtfs-dir/--output-dir before running."
        )
        return
    run(
        gtfs_dir=args.gtfs_dir,
        output_dir=args.output_dir,
        analysis_weekday=args.analysis_weekday,
    )


def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":
    # In a notebook (pasted cell or %run), use the config block instead of
    # argparse, which would otherwise try to parse the kernel's own argv.
    if _in_ipython():
        run()
    else:
        main()
