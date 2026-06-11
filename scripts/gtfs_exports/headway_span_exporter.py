"""Compute route-level headway and span of service from a GTFS feed.

Produces a single CSV with one row per ``route_id``, ready to join to the
ridership-model anchor table on that key.

Metrics
-------
avg_headway_min
    Mean gap (minutes) between consecutive trip departures from the first
    stop of each trip.  Computed per direction then averaged across
    directions, so a two-way route is not credited with half its true
    headway.
span_hrs
    Hours from the first departure to the last departure.  GTFS times
    beyond 24:00 (e.g. ``"25:30:00"``) are handled correctly — they are
    preserved as integers ≥ 1440 minutes before the span is calculated.
trip_count
    Total one-way trips across all directions (diagnostic; not used by
    the model directly).

Weekday service is the default because the ridership model uses weekday
NTD data.  Saturday and Sunday can be selected via ``SERVICE_DAY``.

Notes:
-----
Only ``calendar.txt`` is consulted for service-day classification.
``calendar_dates.txt`` exceptions (added / removed dates) are not
applied here; if your feed relies heavily on calendar_dates.txt, filter
``FILTER_IN_SERVICE_IDS`` manually instead.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Final, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me
OUTPUT_PATH: Path = Path(r"Path\To\Your\Output\headway_span_by_route.csv")  # ←–– change me

# "weekday" (Mon–Fri), "saturday", or "sunday".
SERVICE_DAY: str = "weekday"

# Optional route filters — leave empty to process all routes.
FILTER_IN_ROUTE_SHORT_NAMES: list[str] = []
FILTER_OUT_ROUTE_SHORT_NAMES: list[str] = []

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# CONSTANTS
# =============================================================================

_TIME_RE: re.Pattern[str] = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?$")
REQ_FILES: Final[tuple[str, ...]] = (
    "trips.txt",
    "stop_times.txt",
    "routes.txt",
    "calendar.txt",
)
_DAY_COLS: Final[dict[str, list[str]]] = {
    "weekday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "saturday": ["saturday"],
    "sunday": ["sunday"],
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def hhmmss_to_min(time_str: Optional[str]) -> Optional[int]:
    """Convert ``HH:MM`` or ``HH:MM:SS`` to minutes past midnight.

    GTFS times can exceed 24:00 (e.g. ``"25:30:00"`` for a 1:30 AM trip
    starting on the next calendar day).  Those values are preserved as
    integers ≥ 1440 so that span calculations remain correct.
    """
    if not isinstance(time_str, str):
        return None
    m = _TIME_RE.match(time_str.strip())
    if m is None:
        return None
    return int(m.group("h")) * 60 + int(m.group("m")) + round(int(m.group("s") or 0) / 60)


def load_gtfs(folder: Path) -> dict[str, pd.DataFrame]:
    """Load the required GTFS files from *folder* into a keyed dict."""
    missing = [f for f in REQ_FILES if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing GTFS file(s): {', '.join(missing)}")
    return {f[:-4]: pd.read_csv(folder / f, dtype=str, low_memory=False) for f in REQ_FILES}


def service_ids_for_day(calendar: pd.DataFrame, service_day: str) -> set[str]:
    """Return service_ids that operate on every day covered by *service_day*.

    Args:
        calendar: DataFrame from ``calendar.txt``.
        service_day: One of ``"weekday"``, ``"saturday"``, ``"sunday"``.

    Raises:
        ValueError: If *service_day* is not a recognised key or if a
            required day column is absent from *calendar*.
    """
    cols = _DAY_COLS.get(service_day)
    if cols is None:
        raise ValueError(
            f"SERVICE_DAY must be 'weekday', 'saturday', or 'sunday'; got {service_day!r}"
        )
    cal = calendar.copy()
    for c in cols:
        if c not in cal.columns:
            raise ValueError(f"calendar.txt is missing expected column '{c}'.")
        cal[c] = pd.to_numeric(cal[c], errors="coerce").fillna(0)
    mask = (cal[cols] == 1).all(axis=1)
    ids = set(cal.loc[mask, "service_id"].astype(str))
    if not ids:
        logging.warning(
            "No service_ids found for SERVICE_DAY=%r — check calendar.txt.",
            service_day,
        )
    return ids


def first_departures(stop_times: pd.DataFrame, trip_ids: set[str]) -> pd.DataFrame:
    """Return one row per trip with the departure time of its first stop.

    Args:
        stop_times: Full ``stop_times.txt`` DataFrame.
        trip_ids: Set of trip_id strings to process.

    Returns:
        DataFrame with columns ``trip_id`` and ``departure_min``.
    """
    st = stop_times[stop_times["trip_id"].isin(trip_ids)].copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st["departure_min"] = st["departure_time"].map(hhmmss_to_min)
    st = st.dropna(subset=["stop_sequence", "departure_min"])
    first = st.sort_values("stop_sequence").groupby("trip_id", sort=False).first().reset_index()
    return first[["trip_id", "departure_min"]]


def compute_headway_span(trips_dep: pd.DataFrame) -> pd.DataFrame:
    """Compute avg_headway_min, span_hrs, and trip_count per route.

    Args:
        trips_dep: DataFrame with columns ``route_id``, ``direction_id``,
            and ``departure_min`` (one row per trip).

    Returns:
        One row per ``route_id`` with ``avg_headway_min``, ``span_hrs``,
        and ``trip_count``.
    """
    per_dir: list[dict] = []
    for (route_id, direction_id), grp in trips_dep.groupby(["route_id", "direction_id"]):
        deps = grp["departure_min"].sort_values().to_numpy()
        n = len(deps)
        span_min = float(deps[-1] - deps[0]) if n > 1 else 0.0
        headway = span_min / (n - 1) if n > 1 else float("nan")
        per_dir.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "avg_headway_min": round(headway, 1),
                "span_min": span_min,
                "trip_count": n,
            }
        )

    if not per_dir:
        return pd.DataFrame(columns=["route_id", "avg_headway_min", "span_hrs", "trip_count"])

    df = pd.DataFrame(per_dir)
    out = (
        df.groupby("route_id")
        .agg(
            avg_headway_min=("avg_headway_min", "mean"),
            span_min=("span_min", "max"),  # longest span across directions
            trip_count=("trip_count", "sum"),
        )
        .reset_index()
    )
    out["avg_headway_min"] = out["avg_headway_min"].round(1)
    out["span_hrs"] = (out["span_min"] / 60).round(2)
    return out[["route_id", "avg_headway_min", "span_hrs", "trip_count"]]


def run(
    gtfs_folder: Path | None = None,
    output_path: Path | None = None,
    service_day: str | None = None,
) -> pd.DataFrame:
    """Compute headway and span for one service day and write to CSV.

    Unset args fall back to the config block at the top of this file, so
    ``m.GTFS_FOLDER = ...; m.run()`` works after a plain import.
    """
    gtfs_folder = GTFS_FOLDER if gtfs_folder is None else Path(gtfs_folder)
    output_path = OUTPUT_PATH if output_path is None else Path(output_path)
    service_day = SERVICE_DAY if service_day is None else service_day

    gtfs = load_gtfs(gtfs_folder)

    svc_ids = service_ids_for_day(gtfs["calendar"], service_day)
    logging.info("%d service_id(s) matched for SERVICE_DAY=%r.", len(svc_ids), service_day)

    trips = gtfs["trips"].copy()
    trips = trips[trips["service_id"].isin(svc_ids)]

    routes = gtfs["routes"][["route_id", "route_short_name"]]
    trips = trips.merge(routes, on="route_id", how="left")

    if FILTER_IN_ROUTE_SHORT_NAMES:
        trips = trips[trips["route_short_name"].isin(FILTER_IN_ROUTE_SHORT_NAMES)]
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        trips = trips[~trips["route_short_name"].isin(FILTER_OUT_ROUTE_SHORT_NAMES)]

    if trips.empty:
        logging.error("No trips remain after filtering. Check SERVICE_DAY and route filter lists.")
        sys.exit(1)

    logging.info("%d trips after filtering.", len(trips))

    # direction_id is optional in GTFS; default to "0" when absent.
    if "direction_id" not in trips.columns:
        trips["direction_id"] = "0"
    else:
        trips["direction_id"] = trips["direction_id"].fillna("0")

    trip_ids = set(trips["trip_id"].astype(str))
    dep = first_departures(gtfs["stop_times"], trip_ids)
    trips["trip_id"] = trips["trip_id"].astype(str)

    trips_dep = trips[["trip_id", "route_id", "direction_id"]].merge(dep, on="trip_id", how="inner")
    result = compute_headway_span(trips_dep)
    logging.info("Computed metrics for %d route(s).", len(result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logging.info("Written → %s", output_path)
    return result


def main() -> None:
    """CLI / notebook entry point."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sentinels = {
        Path(r"Path\To\Your\GTFS_Folder"),
        Path(r"Path\To\Your\Output\headway_span_by_route.csv"),
    }
    if GTFS_FOLDER in sentinels or OUTPUT_PATH in sentinels:
        logging.warning(
            "Update GTFS_FOLDER and OUTPUT_PATH in the configuration block before running."
        )
        return
    run()


if __name__ == "__main__":
    main()
