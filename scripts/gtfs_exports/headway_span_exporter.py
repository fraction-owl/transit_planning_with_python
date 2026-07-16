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

Outputs
-------
- ``headway_span_by_route.csv`` (``OUTPUT_FILENAME``) in ``OUTPUT_DIR``: one
  row per ``route_id`` with ``avg_headway_min``, ``span_hrs``, and
  ``trip_count``.

Notes:
-----
Only ``calendar.txt`` is consulted for service-day classification.
``calendar_dates.txt`` exceptions (added / removed dates) are not
applied here; if your feed relies heavily on calendar_dates.txt, filter
``FILTER_IN_SERVICE_IDS`` manually instead.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--gtfs-folder`` /
``--output`` / ``--service-day`` / ``--filter-in`` / ``--filter-out``) and
run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final, List, Optional, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me
OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me
OUTPUT_FILENAME: str = "headway_span_by_route.csv"

# "weekday" (Mon–Fri), "saturday", or "sunday".
SERVICE_DAY: str = "weekday"

# Optional route filters — leave empty to process all routes.
FILTER_IN_ROUTE_SHORT_NAMES: list[str] = []
FILTER_OUT_ROUTE_SHORT_NAMES: list[str] = []

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# CONSTANTS
# =============================================================================

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
    st["departure_min"] = st["departure_time"].map(parse_time_to_minutes)
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
    filter_in_route_short_names: Sequence[str] | None = None,
    filter_out_route_short_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute headway and span for one service day and write to CSV.

    Unset args fall back to the config block at the top of this file, so
    ``m.GTFS_FOLDER = ...; m.run()`` works after a plain import.
    """
    gtfs_folder = GTFS_FOLDER if gtfs_folder is None else Path(gtfs_folder)
    output_path = (OUTPUT_DIR / OUTPUT_FILENAME) if output_path is None else Path(output_path)
    service_day = SERVICE_DAY if service_day is None else service_day
    filter_in = (
        FILTER_IN_ROUTE_SHORT_NAMES
        if filter_in_route_short_names is None
        else list(filter_in_route_short_names)
    )
    filter_out = (
        FILTER_OUT_ROUTE_SHORT_NAMES
        if filter_out_route_short_names is None
        else list(filter_out_route_short_names)
    )

    gtfs = load_gtfs(gtfs_folder)

    svc_ids = service_ids_for_day(gtfs["calendar"], service_day)
    logging.info("%d service_id(s) matched for SERVICE_DAY=%r.", len(svc_ids), service_day)

    trips = gtfs["trips"].copy()
    trips = trips[trips["service_id"].isin(svc_ids)]

    routes = gtfs["routes"][["route_id", "route_short_name"]]
    trips = trips.merge(routes, on="route_id", how="left")

    if filter_in:
        trips = trips[trips["route_short_name"].isin(filter_in)]
    if filter_out:
        trips = trips[~trips["route_short_name"].isin(filter_out)]

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the config block values."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute route-level headway and span of service from a GTFS feed. "
            "Defaults come from the configuration block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-folder", type=Path, default=GTFS_FOLDER, help="Path to the GTFS folder."
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / OUTPUT_FILENAME, help="Output CSV path."
    )
    parser.add_argument(
        "--service-day",
        default=SERVICE_DAY,
        choices=("weekday", "saturday", "sunday"),
        help="Service day to summarize.",
    )
    parser.add_argument(
        "--filter-in",
        nargs="*",
        default=FILTER_IN_ROUTE_SHORT_NAMES,
        metavar="ROUTE_SHORT_NAME",
        help="Only keep these route_short_name values (empty = keep all).",
    )
    parser.add_argument(
        "--filter-out",
        nargs="*",
        default=FILTER_OUT_ROUTE_SHORT_NAMES,
        metavar="ROUTE_SHORT_NAME",
        help="Drop these route_short_name values (empty = drop none).",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(notebook_safe_argv(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the config block.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sentinels = {
        Path(r"Path\To\Your\GTFS_Folder"),
        Path(r"Path\To\Your\Output_Folder") / "headway_span_by_route.csv",
    }
    if args.gtfs_folder in sentinels or args.output in sentinels:
        logging.warning(
            "GTFS_FOLDER and/or OUTPUT_DIR are still placeholders. Update the configuration "
            "block or pass --gtfs-folder/--output before running."
        )
        return 2
    try:
        run(
            gtfs_folder=args.gtfs_folder,
            output_path=args.output,
            service_day=args.service_day,
            filter_in_route_short_names=args.filter_in,
            filter_out_route_short_names=args.filter_out,
        )
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the config block stays in charge.
    raise SystemExit(main())
