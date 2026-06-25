"""Maximum load (peak passenger load) from TIDES ``stop_visits``.

This is the maximum-load counterpart to ``otp_monthly_tides.py``: it derives
each performed trip's peak passenger load directly from raw TIDES
automated-passenger-counter (APC) events rather than consuming a vendor's
pre-aggregated ridecheck export. It is meant for agencies that have TIDES
``stop_visits`` data but no database or vendor software that already computes a
trip's maximum load and peak load point.

The TIDES ``stop_visits`` table carries ``departure_load`` -- the number of
passengers on board as the vehicle leaves each stop. A trip's **maximum load**
is the largest ``departure_load`` over its stop sequence, and the **peak load
point** is the stop where that maximum occurs. This script reduces the
stop-level events to one max-load record per performed trip, joins trip-level
attributes from ``trips_performed``, and rolls the per-trip maxima up across
several aggregation levels:

  * **route_direction**  - one series per ``(route_id, direction_id)``
  * **route**            - one series per ``route_id`` (both directions pooled)
  * **service_type**     - one series per ``route_type_agency`` (e.g. LOCAL,
    EXPRESS)
  * **overall**          - a single system-wide series

Temporal grain
--------------
The CONFIGURATION block produces every temporal grain at once by default, so an
analyst who is not yet sure which cut they need gets all of them. Each grain is
written to its own file. This matters more than for simple ridership counts:
max-load statistics (mean / median / p85 / peak) do not compose across grains,
so every grain is recomputed from the per-trip table rather than rolled up from
a finer one. A trip is assigned to a single time period by its **start time**
(the conventional "this is an AM-peak trip" rule):

  * ``TIME_PERIODS`` - named clock windows (e.g. AM PEAK / MIDDAY / PM PEAK).
    Leave the mapping empty to disable time-of-day splitting entirely.
  * ``EXPORT_GRAINS`` - which grains to write: ``month_and_period`` (finest),
    ``month`` (a monthly trend), ``period`` (a time-of-day profile), and
    ``total`` (one number per group). Period grains are skipped automatically
    when ``TIME_PERIODS`` is empty.

Load factor
-----------
Each trip's load factor is ``max_load / VEHICLE_CAPACITY``; the rolled-up
series report the mean load factor and the share of trips whose maximum load
exceeded capacity. ``VEHICLE_CAPACITY`` is a single agency-wide default here;
for mixed fleets, filter by route or run per service type.

Outputs
-------
  1) ``max_load_by_trip.csv`` - one row per performed trip with its maximum
     load, load factor, peak load point (``stop_id`` and sequence), start time,
     month, and time period. Its column shape echoes the vendor
     ``statistics_by_route_and_trip`` export that ``load_factor_monitor``
     consumes, so it can serve as that tool's input.
  2) One tidy long table per exported grain -- ``max_load_by_month_and_period``
     / ``max_load_by_month`` / ``max_load_by_period`` / ``max_load_total``
     ``.csv`` -- each with one row per (level, group, *temporal keys*) carrying
     the trip count and the mean / median / 85th-percentile / peak maximum load,
     mean load factor, and the share of trips over capacity.
  3) PNG bar charts of mean and peak maximum load per group within each level,
     drawn from the monthly grain when present, otherwise the time-period grain.
  4) A run-log sidecar capturing the verbatim CONFIGURATION block.

A note on APC data
------------------
``departure_load`` is the observed on-board count from whatever vehicles
carried working APC units on each performed trip; it is not an expansion-
factored estimate. The observed trip count is reported alongside every
statistic so thinly observed cells can be spotted.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--stop-visits`` /
``--trips-performed`` / ``--output-dir``) and run from a shell, ArcGIS Pro's
Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Sentinel markers used by extract_config_block / write_run_log to identify the
# configuration block within this file's source. Each string must appear exactly
# once in this file as a stand-alone comment line. Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Seated + standee capacity used to convert a trip's max load into a load
# factor. A single agency-wide value; for mixed fleets, filter by route or run
# per service type.
VEHICLE_CAPACITY: int = 39

# Named clock windows for the time-of-day split, as {name: ("HH:MM", "HH:MM")}
# with the start inclusive and the end exclusive. A window may wrap past
# midnight (start > end), e.g. "NIGHT": ("22:00", "06:00"). Leave the mapping
# empty ({}) to disable the split entirely -- every trip is labeled "ALL DAY".
# A trip is assigned to the window containing its start time.
TIME_PERIODS: Mapping[str, Tuple[str, str]] = {
    "AM PEAK": ("06:00", "09:00"),
    "MIDDAY": ("09:00", "15:00"),
    "PM PEAK": ("15:00", "18:00"),
    "EVENING": ("18:00", "22:00"),
    "NIGHT": ("22:00", "06:00"),
}

# Temporal grains to export. Each grain is written to its own CSV so that
# grains are never mixed as rows in a single file. This matters more here than
# for simple counts: max-load statistics (mean / median / p85 / peak) do not
# compose across grains -- a monthly mean cannot be recovered from the
# month-by-period means -- so every grain is recomputed from the per-trip table.
# Available grains:
#   "month_and_period" - finest: one row per (group, month, time period)
#   "month"            - pooled across time periods (a monthly trend)
#   "period"           - pooled across months (a time-of-day profile)
#   "total"            - pooled across both (a single number per group)
# The two period-bearing grains are skipped automatically when TIME_PERIODS is
# empty. The default exports everything, which is handy when you are not yet
# sure which cut you need.
EXPORT_GRAINS: Sequence[str] = ("month_and_period", "month", "period", "total")

# Optional route filters (matched against route_id as a string). Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

LOG_LEVEL: int = logging.INFO

# Filenames.
BY_TRIP_FILENAME: str = "max_load_by_trip.csv"

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a max-load-from-TIDES run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    vehicle_capacity: int = VEHICLE_CAPACITY
    time_periods: Mapping[str, Tuple[str, str]] = field(default_factory=dict)
    export_grains: Sequence[str] = EXPORT_GRAINS
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()


# Aggregation levels: name -> grouping columns (besides month / time_period).
LEVELS: Dict[str, List[str]] = {
    "route_direction": ["route_id", "direction_id"],
    "route": ["route_id"],
    "service_type": ["route_type_agency"],
    "overall": [],
}

# Temporal grain -> the temporal grouping columns it adds to each level.
GRAIN_TEMPORAL_KEYS: Dict[str, List[str]] = {
    "month_and_period": ["month", "time_period"],
    "month": ["month"],
    "period": ["time_period"],
    "total": [],
}

# Temporal grain -> output filename.
GRAIN_FILENAME: Dict[str, str] = {
    "month_and_period": "max_load_by_month_and_period.csv",
    "month": "max_load_by_month.csv",
    "period": "max_load_by_period.csv",
    "total": "max_load_total.csv",
}

# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read a TIDES ``stop_visits`` CSV and coerce its load/time columns.

    Args:
        path: Path to the ``stop_visits`` CSV export.

    Returns:
        DataFrame with ``service_date`` parsed to a datetime, ``departure_load``
        coerced to numeric, and ``trip_stop_sequence`` coerced to numeric.
    """
    df = pd.read_csv(path, dtype=str)
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["departure_load"] = pd.to_numeric(df["departure_load"], errors="coerce")
    if "trip_stop_sequence" in df.columns:
        df["trip_stop_sequence"] = pd.to_numeric(df["trip_stop_sequence"], errors="coerce")
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read a TIDES ``trips_performed`` CSV and parse its start timestamp.

    Args:
        path: Path to the ``trips_performed`` CSV export.

    Returns:
        DataFrame with string columns and a parsed ``schedule_trip_start``.
    """
    df = pd.read_csv(path, dtype=str)
    if "schedule_trip_start" in df.columns:
        df["schedule_trip_start"] = pd.to_datetime(df["schedule_trip_start"], errors="coerce")
    return df


# Attributes carried over from trips_performed onto each per-trip max-load row.
_TRIP_ATTR_COLS: List[str] = [
    "route_id",
    "direction_id",
    "route_type_agency",
    "ntd_mode",
    "schedule_trip_start",
]


# =============================================================================
# MAX-LOAD DERIVATION
# =============================================================================


def compute_trip_max_load(stop_visits: pd.DataFrame) -> pd.DataFrame:
    """Reduce stop visits to one maximum-load record per performed trip.

    For each ``trip_id_performed`` the row carrying the largest
    ``departure_load`` is kept, yielding the trip's maximum load and the stop
    where it occurs (the peak load point). Trips with no usable load reading are
    dropped.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.

    Returns:
        DataFrame with one row per trip and columns ``trip_id_performed``,
        ``service_date``, ``max_load``, ``peak_stop_id``, and
        ``peak_stop_sequence``.
    """
    usable = stop_visits.loc[stop_visits["departure_load"].notna()].copy()
    if usable.empty:
        return pd.DataFrame(
            columns=[
                "trip_id_performed",
                "service_date",
                "max_load",
                "peak_stop_id",
                "peak_stop_sequence",
            ]
        )

    # Sort so idxmax breaks ties on the earliest stop in the sequence.
    sort_cols = ["trip_id_performed"]
    if "trip_stop_sequence" in usable.columns:
        sort_cols.append("trip_stop_sequence")
    usable = usable.sort_values(sort_cols)

    peak_idx = usable.groupby("trip_id_performed")["departure_load"].idxmax()
    peak = usable.loc[peak_idx]

    out = pd.DataFrame(
        {
            "trip_id_performed": peak["trip_id_performed"].to_numpy(),
            "service_date": peak["service_date"].to_numpy(),
            "max_load": peak["departure_load"].to_numpy(),
            "peak_stop_id": peak["stop_id"].to_numpy() if "stop_id" in peak.columns else np.nan,
            "peak_stop_sequence": peak["trip_stop_sequence"].to_numpy()
            if "trip_stop_sequence" in peak.columns
            else np.nan,
        }
    )
    return out.reset_index(drop=True)


def join_trip_attributes(
    trip_max_load: pd.DataFrame,
    trips_performed: pd.DataFrame,
) -> pd.DataFrame:
    """Attach route/direction/service-type/start-time attributes per trip.

    Trips that were Canceled (or not in revenue service) in ``trips_performed``
    are dropped. The join key is ``trip_id_performed``.

    Args:
        trip_max_load: Output of :func:`compute_trip_max_load`.
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Per-trip max-load rows with the ``_TRIP_ATTR_COLS`` attributes joined on.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [c for c in _TRIP_ATTR_COLS if c in trips.columns]
    trips = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")

    return trip_max_load.merge(trips, on="trip_id_performed", how="inner")


def _hhmm_to_minutes(hhmm: str) -> int:
    """Convert an ``"HH:MM"`` clock string to minutes since midnight."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def assign_time_period(
    df: pd.DataFrame,
    time_periods: Mapping[str, Tuple[str, str]],
    time_col: str = "schedule_trip_start",
) -> pd.DataFrame:
    """Add a ``time_period`` column from each trip's start time.

    Each trip is assigned to the first window in ``time_periods`` that contains
    its start time (start inclusive, end exclusive; windows may wrap midnight).
    Trips that fall in no window are labeled ``"UNCLASSIFIED"``. When
    ``time_periods`` is empty, every trip is labeled ``"ALL DAY"``.

    Args:
        df: Per-trip rows with a parsed start-time column.
        time_periods: Mapping of window name -> (``"HH:MM"`` start, end).
        time_col: Timestamp column to read the start time-of-day from.

    Returns:
        Copy of ``df`` with a string ``time_period`` column.
    """
    df = df.copy()
    if not time_periods:
        df["time_period"] = "ALL DAY"
        return df

    stamp = pd.to_datetime(df[time_col], errors="coerce")
    minutes = stamp.dt.hour * 60 + stamp.dt.minute

    label = pd.Series("UNCLASSIFIED", index=df.index, dtype="object")
    assigned = pd.Series(False, index=df.index)
    for name, (start, end) in time_periods.items():
        start_min = _hhmm_to_minutes(start)
        end_min = _hhmm_to_minutes(end)
        if start_min <= end_min:
            in_window = (minutes >= start_min) & (minutes < end_min)
        else:  # window wraps past midnight
            in_window = (minutes >= start_min) | (minutes < end_min)
        take = in_window & ~assigned & minutes.notna()
        label = label.mask(take, name)
        assigned = assigned | take
    df["time_period"] = label
    return df


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``month`` column in ``YYYY-MM`` form derived from ``service_date``.

    Args:
        df: Per-trip rows with a parsed ``service_date`` column.

    Returns:
        Copy of ``df`` with a string ``month`` column.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["service_date"], errors="coerce").dt.strftime("%Y-%m")
    return df


def resolve_grains(
    export_grains: Sequence[str],
    time_periods: Mapping[str, Tuple[str, str]],
) -> List[str]:
    """Return the grains to actually export, filtered and de-duplicated.

    Unknown grain names are dropped with a warning. When ``time_periods`` is
    empty the period-bearing grains (``month_and_period`` and ``period``) are
    skipped, since they would duplicate ``month`` and ``total`` respectively.

    Args:
        export_grains: Requested grain names (see ``EXPORT_GRAINS``).
        time_periods: The configured time-period windows.

    Returns:
        An ordered list of valid grain names with duplicates removed.
    """
    has_periods = bool(time_periods)
    resolved: List[str] = []
    for grain in export_grains:
        if grain not in GRAIN_TEMPORAL_KEYS:
            logging.warning("Ignoring unknown grain %r.", grain)
            continue
        if not has_periods and "time_period" in GRAIN_TEMPORAL_KEYS[grain]:
            continue
        if grain not in resolved:
            resolved.append(grain)
    return resolved


def add_load_factor(df: pd.DataFrame, vehicle_capacity: int) -> pd.DataFrame:
    """Add a ``load_factor`` column (max load divided by capacity).

    Args:
        df: Per-trip rows with a ``max_load`` column.
        vehicle_capacity: Seated + standee capacity used as the denominator.

    Returns:
        Copy of ``df`` with a float ``load_factor`` column (rounded to 4 dp).
    """
    df = df.copy()
    if vehicle_capacity > 0:
        df["load_factor"] = (df["max_load"] / vehicle_capacity).round(4)
    else:
        df["load_factor"] = np.nan
    return df


# =============================================================================
# AGGREGATION
# =============================================================================


def _p85(series: pd.Series) -> float:
    """Return the 85th percentile of *series* (NaN for an empty series)."""
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 85))


def aggregate_max_load(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    temporal_keys: Sequence[str],
    vehicle_capacity: int,
) -> pd.DataFrame:
    """Aggregate per-trip maxima to summary statistics per group.

    Because max-load statistics do not compose across grains, this always reads
    the per-trip rows directly; coarser grains simply group on fewer columns.

    Args:
        df: Per-trip max-load rows with ``max_load``, ``load_factor``,
            ``month``, and ``time_period`` columns.
        group_cols: Grouping columns in addition to the temporal keys (may be
            empty for a system-wide aggregation).
        temporal_keys: Temporal grouping columns for the chosen grain (e.g.
            ``["month", "time_period"]``, ``["month"]``, or ``[]``).
        vehicle_capacity: Capacity used to flag over-capacity trips.

    Returns:
        Tidy DataFrame with one row per (``*group_cols``, ``*temporal_keys``)
        and columns ``trips``, ``mean_max_load``, ``median_max_load``,
        ``p85_max_load``, ``peak_max_load``, ``mean_load_factor``, and
        ``pct_trips_over_capacity``.
    """
    keys = list(group_cols) + list(temporal_keys)
    work = df.copy()
    work["_over"] = (work["max_load"] > vehicle_capacity).astype(float)

    group_on: List[str] = keys
    if not keys:
        # System-wide single-row aggregation: group on a constant helper column.
        work["_all"] = 0
        group_on = ["_all"]

    out = work.groupby(group_on, dropna=False).agg(
        trips=("max_load", "size"),
        mean_max_load=("max_load", "mean"),
        median_max_load=("max_load", "median"),
        p85_max_load=("max_load", _p85),
        peak_max_load=("max_load", "max"),
        mean_load_factor=("load_factor", "mean"),
        pct_trips_over_capacity=("_over", "mean"),
    )
    out["mean_max_load"] = out["mean_max_load"].round(2)
    out["p85_max_load"] = out["p85_max_load"].round(2)
    out["mean_load_factor"] = out["mean_load_factor"].round(4)
    out["pct_trips_over_capacity"] = (out["pct_trips_over_capacity"] * 100.0).round(1)
    out = out.reset_index()
    if not keys:
        return out.drop(columns="_all")
    return out.sort_values(keys).reset_index(drop=True)


def build_all_levels(
    df: pd.DataFrame,
    temporal_keys: Sequence[str],
    vehicle_capacity: int,
) -> Dict[str, pd.DataFrame]:
    """Compute max-load summaries for every aggregation level at one grain.

    Args:
        df: Per-trip max-load rows ready for aggregation.
        temporal_keys: Temporal grouping columns for the chosen grain.
        vehicle_capacity: Capacity used to flag over-capacity trips.

    Returns:
        Mapping of level name -> aggregated DataFrame. Each frame gains a
        ``level`` column naming its aggregation level.
    """
    results: Dict[str, pd.DataFrame] = {}
    for level, group_cols in LEVELS.items():
        present = [c for c in group_cols if c in df.columns]
        agg = aggregate_max_load(df, present, temporal_keys, vehicle_capacity)
        agg.insert(0, "level", level)
        results[level] = agg
    return results


def make_long_table(
    levels: Mapping[str, pd.DataFrame],
    temporal_keys: Sequence[str],
) -> pd.DataFrame:
    """Concatenate per-level frames into a single tidy long table.

    A ``group`` column is synthesized as a human-readable identifier for each
    series (e.g. ``"101 | 0"``, ``"101"``, ``"LOCAL"``, ``"ALL"``).

    Args:
        levels: Mapping of level name -> aggregated DataFrame.
        temporal_keys: Temporal columns present in the frames, placed up front.

    Returns:
        A single long DataFrame with a leading ``level`` / ``group`` /
        ``*temporal_keys`` column order.
    """
    rows: List[pd.DataFrame] = []
    for level, agg in levels.items():
        frame = agg.copy()
        if level == "route_direction":
            group = frame["route_id"].astype(str) + " | " + frame["direction_id"].astype(str)
        elif level == "route":
            group = frame["route_id"].astype(str)
        elif level == "service_type":
            group = frame["route_type_agency"].astype(str)
        else:  # overall
            group = pd.Series(["ALL"] * len(frame), index=frame.index)
        frame.insert(1, "group", group)
        rows.append(frame)

    combined = pd.concat(rows, ignore_index=True)
    front = ["level", "group", *temporal_keys, "trips", "mean_max_load", "peak_max_load"]
    ordered = front + [c for c in combined.columns if c not in front]
    return combined[ordered]


def build_by_trip_export(df: pd.DataFrame) -> pd.DataFrame:
    """Order the per-trip max-load rows for the trip-level CSV export.

    Args:
        df: Per-trip max-load rows after attributes/time period are attached.

    Returns:
        A column-ordered copy suitable for direct CSV export.
    """
    cols = [
        "service_date",
        "month",
        "time_period",
        "route_id",
        "direction_id",
        "route_type_agency",
        "trip_id_performed",
        "schedule_trip_start",
        "peak_stop_id",
        "peak_stop_sequence",
        "max_load",
        "load_factor",
    ]
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    sort_cols = [c for c in ("route_id", "direction_id", "schedule_trip_start") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


# =============================================================================
# OUTPUT
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _slug(value: object) -> str:
    """Return a filesystem-safe token for a group identifier."""
    txt = str(value).strip()
    for ch in (" ", "|", "/", "\\", ":"):
        txt = txt.replace(ch, "_")
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt.strip("_") or "group"


def export_tables(
    by_trip: pd.DataFrame,
    grain_tables: Mapping[str, pd.DataFrame],
    out_dir: Path,
) -> List[Path]:
    """Write the per-trip max-load table and one CSV per temporal grain.

    Args:
        by_trip: The per-trip max-load table.
        grain_tables: Mapping of grain name -> its tidy long table.
        out_dir: Destination directory.

    Returns:
        Paths of the files written.
    """
    ensure_dir(out_dir)
    written: List[Path] = []

    by_trip_path = out_dir / BY_TRIP_FILENAME
    by_trip.to_csv(by_trip_path, index=False)
    written.append(by_trip_path)

    for grain, table in grain_tables.items():
        grain_path = out_dir / GRAIN_FILENAME[grain]
        table.to_csv(grain_path, index=False)
        written.append(grain_path)

    return written


def plot_levels(long_table: pd.DataFrame, out_dir: Path, x_field: str) -> List[Path]:
    """Render mean/peak max-load bar charts per group within each level.

    Args:
        long_table: The tidy long max-load table for the chart grain.
        out_dir: Output directory; charts are written under ``out_dir/plots``.
        x_field: Column to place on the x-axis (``"month"`` or
            ``"time_period"``).

    Returns:
        Paths of the PNG files written.
    """
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []

    for level in long_table["level"].unique():
        sub = long_table.loc[long_table["level"] == level]
        categories = sorted(sub[x_field].dropna().unique())
        if not categories:
            continue
        for group, g in sub.groupby("group"):
            mean_series = g.groupby(x_field)["mean_max_load"].mean().reindex(categories)
            peak_series = g.groupby(x_field)["peak_max_load"].max().reindex(categories)
            if mean_series.dropna().empty:
                continue
            x = np.arange(len(categories))
            width = 0.4
            plt.figure()
            plt.bar(x - width / 2, mean_series.to_numpy(dtype=float), width, label="Mean max load")
            plt.bar(x + width / 2, peak_series.to_numpy(dtype=float), width, label="Peak max load")
            plt.xticks(x, categories, rotation=45, ha="right")
            plt.xlabel(x_field.replace("_", " ").title())
            plt.ylabel("Passengers")
            plt.title(f"{level}: {group} - maximum load")
            plt.legend()
            plt.tight_layout()
            out_path = plots_dir / f"max_load_{level}_{_slug(group)}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            written.append(out_path)

    return written


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def extract_config_block(source_file: Path) -> str:
    """Return the text between the CONFIG markers in *source_file*.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "max_load_from_tides_runlog.txt"

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
        "MAX LOAD FROM TIDES RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "BUILD SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# PIPELINE
# =============================================================================


def _pick_chart_grain(grains: Sequence[str]) -> str | None:
    """Choose which exported grain to chart (prefer a single-axis trend/profile).

    Args:
        grains: The grains being exported.

    Returns:
        A grain name to chart, or ``None`` if none is suitable.
    """
    for preferred in ("month", "period", "total"):
        if preferred in grains:
            return preferred
    return None


def run(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Execute the full max-load pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        Mapping with key ``by_trip`` (per-trip maxima) plus one key per
        exported grain (the tidy long summary written for it).
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)

    per_trip = compute_trip_max_load(stop_visits)
    joined = join_trip_attributes(per_trip, trips)

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        joined = joined.loc[joined["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        joined = joined.loc[~joined["route_id"].astype(str).isin(drop)]

    prepared = (
        joined.pipe(assign_time_period, cfg.time_periods)
        .pipe(add_month)
        .pipe(add_load_factor, cfg.vehicle_capacity)
    )

    by_trip = build_by_trip_export(prepared)

    grains = resolve_grains(cfg.export_grains, cfg.time_periods)
    if not grains:
        raise RuntimeError("No valid grains to export; check EXPORT_GRAINS.")

    grain_tables: Dict[str, pd.DataFrame] = {}
    for grain in grains:
        temporal_keys = GRAIN_TEMPORAL_KEYS[grain]
        levels = build_all_levels(prepared, temporal_keys, cfg.vehicle_capacity)
        grain_tables[grain] = make_long_table(levels, temporal_keys)

    paths = export_tables(by_trip, grain_tables, cfg.output_dir)
    for p in paths:
        logging.info("Wrote table: %s", p)

    chart_grain = _pick_chart_grain(grains)
    if chart_grain is not None and GRAIN_TEMPORAL_KEYS[chart_grain]:
        x_field = GRAIN_TEMPORAL_KEYS[chart_grain][0]
        plot_paths = plot_levels(grain_tables[chart_grain], cfg.output_dir, x_field)
        logging.info("Wrote %d max-load charts to %s", len(plot_paths), cfg.output_dir / "plots")

    peak = int(prepared["max_load"].max()) if not prepared.empty else 0
    mean_load = round(float(prepared["max_load"].mean()), 2) if not prepared.empty else 0
    summary_lines = [
        f"Trips with max load: {len(prepared)}",
        f"System peak load:    {peak}",
        f"Mean max load:       {mean_load}",
        f"Vehicle capacity:    {cfg.vehicle_capacity}",
        f"Months in panel:     {prepared['month'].nunique()}",
        f"Time periods:        {', '.join(sorted(prepared['time_period'].unique()))}",
        f"Grains exported:     {', '.join(grains)}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return {"by_trip": by_trip, **grain_tables}


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(description="Maximum load from TIDES stop_visits.")
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--vehicle-capacity",
        type=int,
        default=VEHICLE_CAPACITY,
        help="Capacity used for the load factor.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point. Validates placeholder paths before doing any work."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(argv)

    if args.stop_visits == STOP_VISITS_PATH or args.trips_performed == TRIPS_PERFORMED_PATH:
        logging.warning(
            "STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. Update the "
            "CONFIGURATION section or pass --stop-visits/--trips-performed before running."
        )
        return

    cfg = Config(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        vehicle_capacity=args.vehicle_capacity,
        time_periods=TIME_PERIODS,
        export_grains=EXPORT_GRAINS,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
    )

    if not cfg.stop_visits_path.exists():
        logging.warning("stop_visits not found: %s", cfg.stop_visits_path)
        return
    if not cfg.trips_performed_path.exists():
        logging.warning("trips_performed not found: %s", cfg.trips_performed_path)
        return

    run(cfg)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
