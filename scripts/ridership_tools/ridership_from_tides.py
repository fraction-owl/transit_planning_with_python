"""Ridership (boardings/alightings) from TIDES ``stop_visits``.

This is the ridership counterpart to ``otp_monthly_panel.py``: it derives
ridership directly from raw TIDES automated-passenger-counter (APC) events
rather than consuming a vendor's pre-aggregated ridecheck export. It is meant
for agencies that have TIDES ``stop_visits`` data but no database or vendor
software that already rolls boardings and alightings up by stop, route, and
time period.

It reads a TIDES-style ``stop_visits`` table (stop-level events carrying the
per-door ``boarding_*`` / ``alighting_*`` counts and the post-stop
``departure_load``) joined to a ``trips_performed`` table (trip-level
attributes such as ``route_id`` and ``direction_id``), then sums boardings and
alightings for several aggregation levels:

  * **route_stop**       - one row per ``(route_id, direction_id, stop_id)``
    (the stop-level ridership grain that feeds ``stops_ridership_joiner``)
  * **stop**             - one row per ``stop_id`` (all routes pooled)
  * **route_direction**  - one series per ``(route_id, direction_id)``
  * **route**            - one series per ``route_id`` (both directions pooled)
  * **service_type**     - one series per ``route_type_agency`` (e.g. LOCAL,
    EXPRESS)
  * **overall**          - a single system-wide total

Temporal grain
--------------
The CONFIGURATION block produces every temporal grain at once by default, so an
analyst who is not yet sure which cut they need gets all of them. Each grain is
written to its own file -- grains are never mixed as rows in one table, which
would invite double-counting when a column is naively summed:

  * ``TIME_PERIODS`` - named clock windows (e.g. AM PEAK / MIDDAY / PM PEAK).
    Each event is assigned to the window containing its departure time. Leave
    the mapping empty to disable time-of-day splitting entirely.
  * ``EXPORT_GRAINS`` - which grains to write: ``month_and_period`` (finest),
    ``month`` (a monthly trend), ``period`` (a time-of-day profile), and
    ``total`` (one number per group). Period grains are skipped automatically
    when ``TIME_PERIODS`` is empty.

Outputs
-------
  1) One tidy long table per exported grain -- ``ridership_by_month_and_period``
     / ``ridership_by_month`` / ``ridership_by_period`` / ``ridership_total``
     ``.csv`` -- each with one row per (level, group, *temporal keys*) carrying
     boardings, alightings, net boardings, observed stop visits, observed
     trips, and average boardings per observed trip.
  2) ``ridership_by_route_and_stop.csv`` - the ``route_stop`` level (at the
     finest exported grain) reshaped into a vendor-style export (``TIME_PERIOD``
     / ``ROUTE_ID`` / ``STOP_ID`` / ``BOARD_ALL`` / ``ALIGHT_ALL``) so it can
     stand in as the input that ``stops_ridership_joiner`` expects.
  3) PNG bar charts of boardings per group within each level, drawn from the
     monthly grain when present, otherwise the time-period grain.
  4) A run-log sidecar capturing the verbatim CONFIGURATION block.

A note on APC data
------------------
Raw TIDES counts are the observed boardings/alightings from whatever vehicles
carried working APC units on each performed trip; they are *not* expansion-
factored systemwide estimates the way many vendor ridecheck products are. The
observed stop-visit and trip counts are reported alongside the sums so a cell
backed by very little data can be spotted and, if desired, expanded downstream.
Both doors are summed (``boarding_1 + boarding_2``); single-door counters simply
leave the door-2 column at zero.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--stop-visits`` /
``--trips-performed`` / ``--output-dir``) and run from a shell, ArcGIS Pro's
Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
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

# Named clock windows for the time-of-day split, as {name: ("HH:MM", "HH:MM")}
# with the start inclusive and the end exclusive. A window may wrap past
# midnight (start > end), e.g. "NIGHT": ("22:00", "06:00"). Leave the mapping
# empty ({}) to disable the split entirely -- every event is labeled "ALL DAY".
TIME_PERIODS: Mapping[str, Tuple[str, str]] = {
    "AM PEAK": ("06:00", "09:00"),
    "MIDDAY": ("09:00", "15:00"),
    "PM PEAK": ("15:00", "18:00"),
    "EVENING": ("18:00", "22:00"),
    "NIGHT": ("22:00", "06:00"),
}

# Temporal grains to export. Each grain is written to its own CSV so that
# grains are never mixed as rows in a single file (which would invite
# double-counting when an analyst naively sums a column). Available grains:
#   "month_and_period" - finest: one row per (group, month, time period)
#   "month"            - pooled across time periods (a monthly trend)
#   "period"           - pooled across months (a time-of-day profile)
#   "total"            - pooled across both (a single number per group)
# The two period-bearing grains are skipped automatically when TIME_PERIODS is
# empty, since they would duplicate "month" and "total". The default exports
# everything, which is handy when you are not yet sure which cut you need.
EXPORT_GRAINS: Sequence[str] = ("month_and_period", "month", "period", "total")

# Optional route filters (matched against route_id as a string). Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

LOG_LEVEL: int = logging.INFO

# Filenames.
BY_ROUTE_AND_STOP_FILENAME: str = "ridership_by_route_and_stop.csv"

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a ridership-from-TIDES run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    time_periods: Mapping[str, Tuple[str, str]] = field(default_factory=dict)
    export_grains: Sequence[str] = EXPORT_GRAINS
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()


# Aggregation levels: name -> grouping columns (besides month / time_period).
LEVELS: Dict[str, List[str]] = {
    "route_stop": ["route_id", "direction_id", "stop_id"],
    "stop": ["stop_id"],
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
    "month_and_period": "ridership_by_month_and_period.csv",
    "month": "ridership_by_month.csv",
    "period": "ridership_by_period.csv",
    "total": "ridership_total.csv",
}

# Per-door APC count columns summed into total boardings / alightings.
_BOARDING_COLS: List[str] = ["boarding_1", "boarding_2"]
_ALIGHTING_COLS: List[str] = ["alighting_1", "alighting_2"]

# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read a TIDES ``stop_visits`` CSV and coerce its count/time columns.

    Args:
        path: Path to the ``stop_visits`` CSV export.

    Returns:
        DataFrame with ``service_date`` and the departure/arrival timestamp
        columns parsed to datetimes and the per-door boarding/alighting columns
        coerced to numeric (missing counts become 0).
    """
    df = pd.read_csv(path, dtype=str)
    for col in ("actual_departure_time", "schedule_departure_time", "actual_arrival_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    for col in _BOARDING_COLS + _ALIGHTING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read a TIDES ``trips_performed`` CSV (trip-level attributes).

    Args:
        path: Path to the ``trips_performed`` CSV export.

    Returns:
        DataFrame with all columns read as strings; only the attribute columns
        are needed for the join here.
    """
    return pd.read_csv(path, dtype=str)


# Attributes carried over from trips_performed onto each stop visit.
_TRIP_ATTR_COLS: List[str] = [
    "route_id",
    "direction_id",
    "route_type_agency",
    "ntd_mode",
]


def join_trip_attributes(
    stop_visits: pd.DataFrame,
    trips_performed: pd.DataFrame,
) -> pd.DataFrame:
    """Attach route/direction/service-type attributes to each stop visit.

    Trips that were Canceled (or not in revenue service) in ``trips_performed``
    are dropped, since their stop visits carry no meaningful ridership. The join
    key is ``trip_id_performed``, unique per performed trip in TIDES.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Stop visits with the ``_TRIP_ATTR_COLS`` attributes joined on.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [c for c in _TRIP_ATTR_COLS if c in trips.columns]
    trips = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")

    return stop_visits.merge(trips, on="trip_id_performed", how="inner")


# =============================================================================
# RIDERSHIP DERIVATION
# =============================================================================


def filter_for_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """Drop stop visits whose ``schedule_relationship`` is ``Skipped``.

    A Skipped visit is one where the bus passed without opening its doors, so
    no boardings or alightings could occur. Added visits are kept (they carried
    riders); their per-door counts flow through unchanged.

    Args:
        df: Joined stop visits.

    Returns:
        Filtered copy suitable for ridership aggregation.
    """
    out = df
    if "schedule_relationship" in out.columns:
        out = out.loc[out["schedule_relationship"].fillna("Scheduled") != "Skipped"]
    return out.copy()


def add_ridership_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add total ``boardings`` and ``alightings`` columns (both doors summed).

    Args:
        df: Stop visits carrying the per-door count columns.

    Returns:
        Copy of ``df`` with float ``boardings`` and ``alightings`` columns.
    """
    df = df.copy()
    df["boardings"] = df[_BOARDING_COLS].sum(axis=1)
    df["alightings"] = df[_ALIGHTING_COLS].sum(axis=1)
    return df


def _hhmm_to_minutes(hhmm: str) -> int:
    """Convert an ``"HH:MM"`` clock string to minutes since midnight."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def assign_time_period(
    df: pd.DataFrame,
    time_periods: Mapping[str, Tuple[str, str]],
    time_col: str = "actual_departure_time",
    fallback_cols: Sequence[str] = ("schedule_departure_time", "actual_arrival_time"),
) -> pd.DataFrame:
    """Add a ``time_period`` column from each row's time-of-day.

    Each row is assigned to the first window in ``time_periods`` that contains
    its time-of-day (start inclusive, end exclusive; windows may wrap midnight).
    Rows that fall in no window are labeled ``"UNCLASSIFIED"``. When
    ``time_periods`` is empty, every row is labeled ``"ALL DAY"``.

    Args:
        df: Stop visits with parsed timestamp columns.
        time_periods: Mapping of window name -> (``"HH:MM"`` start, end).
        time_col: Preferred timestamp column to read the time-of-day from.
        fallback_cols: Timestamp columns used, in order, where ``time_col`` is
            missing (e.g. terminal stops with no departure timestamp).

    Returns:
        Copy of ``df`` with a string ``time_period`` column.
    """
    df = df.copy()
    if not time_periods:
        df["time_period"] = "ALL DAY"
        return df

    stamp = df[time_col]
    for col in fallback_cols:
        if col in df.columns:
            stamp = stamp.fillna(df[col])
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
        df: Stop visits with a parsed ``service_date`` column.

    Returns:
        Copy of ``df`` with a string ``month`` column.
    """
    df = df.copy()
    df["month"] = df["service_date"].dt.strftime("%Y-%m")
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


# =============================================================================
# AGGREGATION
# =============================================================================


def aggregate_ridership(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    temporal_keys: Sequence[str],
) -> pd.DataFrame:
    """Aggregate ridership to boardings/alightings counts per group.

    Args:
        df: Stop visits with ``boardings``, ``alightings``, ``month``, and
            ``time_period`` columns.
        group_cols: Grouping columns in addition to the temporal keys (may be
            empty for a system-wide aggregation).
        temporal_keys: Temporal grouping columns for the chosen grain (e.g.
            ``["month", "time_period"]``, ``["month"]``, or ``[]``).

    Returns:
        Tidy DataFrame with one row per (``*group_cols``, ``*temporal_keys``)
        and columns ``boardings``, ``alightings``, ``net_boardings``,
        ``stop_visits``, ``trips``, and ``avg_boardings_per_trip``.
    """
    keys = list(group_cols) + list(temporal_keys)
    work = df
    group_on: List[str] = keys
    if not keys:
        # System-wide single-row aggregation: group on a constant helper column.
        work = df.assign(_all=0)
        group_on = ["_all"]

    out = work.groupby(group_on, dropna=False).agg(
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        stop_visits=("boardings", "size"),
        trips=("trip_id_performed", "nunique"),
    )
    out["net_boardings"] = out["boardings"] - out["alightings"]
    out["avg_boardings_per_trip"] = (out["boardings"] / out["trips"].where(out["trips"] > 0)).round(
        2
    )
    out = out[
        [
            "boardings",
            "alightings",
            "net_boardings",
            "stop_visits",
            "trips",
            "avg_boardings_per_trip",
        ]
    ]
    out = out.reset_index()
    if not keys:
        out = out.drop(columns="_all")
        return out
    return out.sort_values(keys).reset_index(drop=True)


def build_all_levels(df: pd.DataFrame, temporal_keys: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Compute ridership for every aggregation level at one temporal grain.

    Args:
        df: Stop visits ready for aggregation (boardings/alightings + temporal
            keys present).
        temporal_keys: Temporal grouping columns for the chosen grain.

    Returns:
        Mapping of level name -> aggregated DataFrame. Each frame gains a
        ``level`` column naming its aggregation level.
    """
    results: Dict[str, pd.DataFrame] = {}
    for level, group_cols in LEVELS.items():
        present = [c for c in group_cols if c in df.columns]
        agg = aggregate_ridership(df, present, temporal_keys)
        agg.insert(0, "level", level)
        results[level] = agg
    return results


def make_long_table(
    levels: Mapping[str, pd.DataFrame],
    temporal_keys: Sequence[str],
) -> pd.DataFrame:
    """Concatenate per-level frames into a single tidy long table.

    A ``group`` column is synthesized as a human-readable identifier for each
    series (e.g. ``"101 | 0 | 1012"``, ``"1012"``, ``"101"``, ``"LOCAL"``,
    ``"ALL"``).

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
        if level == "route_stop":
            group = (
                frame["route_id"].astype(str)
                + " | "
                + frame["direction_id"].astype(str)
                + " | "
                + frame["stop_id"].astype(str)
            )
        elif level == "stop":
            group = frame["stop_id"].astype(str)
        elif level == "route_direction":
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
    front = ["level", "group", *temporal_keys, "boardings", "alightings", "net_boardings"]
    ordered = front + [c for c in combined.columns if c not in front]
    return combined[ordered]


def build_by_route_and_stop(levels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Reshape the ``route_stop`` level into a vendor-style export.

    The result mirrors the column shape ``stops_ridership_joiner`` consumes
    (``TIME_PERIOD`` / ``ROUTE_ID`` / ``STOP_ID`` / ``BOARD_ALL`` /
    ``ALIGHT_ALL``), so the TIDES-derived counts can stand in for a vendor
    ridecheck export.

    Args:
        levels: Mapping returned by :func:`build_all_levels`.

    Returns:
        A DataFrame in the vendor-style layout (empty if the ``route_stop``
        level is absent).
    """
    frame = levels.get("route_stop")
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.rename(
        columns={
            "time_period": "TIME_PERIOD",
            "route_id": "ROUTE_ID",
            "direction_id": "DIRECTION_ID",
            "stop_id": "STOP_ID",
            "month": "MONTH",
            "boardings": "BOARD_ALL",
            "alightings": "ALIGHT_ALL",
        }
    )
    cols = [
        "TIME_PERIOD",
        "MONTH",
        "ROUTE_ID",
        "DIRECTION_ID",
        "STOP_ID",
        "BOARD_ALL",
        "ALIGHT_ALL",
    ]
    return out[[c for c in cols if c in out.columns]]


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
    grain_tables: Mapping[str, pd.DataFrame],
    by_route_and_stop: pd.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """Write one CSV per temporal grain plus the vendor-style stop export.

    Args:
        grain_tables: Mapping of grain name -> its tidy long table.
        by_route_and_stop: The vendor-style stop export (may be empty).
        out_dir: Destination directory.

    Returns:
        Paths of the files written.
    """
    ensure_dir(out_dir)
    written: List[Path] = []

    for grain, table in grain_tables.items():
        grain_path = out_dir / GRAIN_FILENAME[grain]
        table.to_csv(grain_path, index=False)
        written.append(grain_path)

    if not by_route_and_stop.empty:
        brs_path = out_dir / BY_ROUTE_AND_STOP_FILENAME
        by_route_and_stop.to_csv(brs_path, index=False)
        written.append(brs_path)

    return written


def plot_levels(long_table: pd.DataFrame, out_dir: Path, x_field: str) -> List[Path]:
    """Render boardings bar charts, one PNG per group within each level.

    The high-cardinality ``route_stop`` and ``stop`` levels are skipped to keep
    the chart count manageable.

    Args:
        long_table: The tidy long ridership table for the chart grain.
        out_dir: Output directory; charts are written under ``out_dir/plots``.
        x_field: Column to place on the x-axis (``"month"`` or
            ``"time_period"``).

    Returns:
        Paths of the PNG files written.
    """
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []

    chart_levels = ["route_direction", "route", "service_type", "overall"]
    for level in chart_levels:
        sub = long_table.loc[long_table["level"] == level]
        if sub.empty:
            continue
        categories = sorted(sub[x_field].dropna().unique())
        if not categories:
            continue
        for group, g in sub.groupby("group"):
            series = g.groupby(x_field)["boardings"].sum().reindex(categories, fill_value=0)
            if series.sum() == 0:
                continue
            plt.figure()
            plt.bar(range(len(categories)), series.to_numpy(dtype=float))
            plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
            plt.xlabel(x_field.replace("_", " ").title())
            plt.ylabel("Boardings")
            plt.title(f"{level}: {group} - boardings")
            plt.tight_layout()
            out_path = plots_dir / f"ridership_{level}_{_slug(group)}.png"
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
    log_path = output_dir / "ridership_from_tides_runlog.txt"

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
        "RIDERSHIP FROM TIDES RUN LOG",
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

    Charts read best with one categorical axis, so a monthly trend is preferred,
    then a time-of-day profile, then the system total.

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
    """Execute the full ridership pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        Mapping of grain name -> the tidy long ridership table written for it.
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)
    joined = join_trip_attributes(stop_visits, trips)

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        joined = joined.loc[joined["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        joined = joined.loc[~joined["route_id"].astype(str).isin(drop)]

    prepared = (
        joined.pipe(filter_for_ridership)
        .pipe(add_ridership_columns)
        .pipe(assign_time_period, cfg.time_periods)
        .pipe(add_month)
    )

    grains = resolve_grains(cfg.export_grains, cfg.time_periods)
    if not grains:
        raise RuntimeError("No valid grains to export; check EXPORT_GRAINS.")

    grain_tables: Dict[str, pd.DataFrame] = {}
    for grain in grains:
        temporal_keys = GRAIN_TEMPORAL_KEYS[grain]
        levels = build_all_levels(prepared, temporal_keys)
        grain_tables[grain] = make_long_table(levels, temporal_keys)

    # The vendor-style stop export is built at the finest grain available.
    finest = "month_and_period" if "month_and_period" in grains else grains[0]
    finest_levels = build_all_levels(prepared, GRAIN_TEMPORAL_KEYS[finest])
    by_route_and_stop = build_by_route_and_stop(finest_levels)

    paths = export_tables(grain_tables, by_route_and_stop, cfg.output_dir)
    for p in paths:
        logging.info("Wrote table: %s", p)

    chart_grain = _pick_chart_grain(grains)
    if chart_grain is not None and GRAIN_TEMPORAL_KEYS[chart_grain]:
        x_field = GRAIN_TEMPORAL_KEYS[chart_grain][0]
        plot_paths = plot_levels(grain_tables[chart_grain], cfg.output_dir, x_field)
        logging.info("Wrote %d ridership charts to %s", len(plot_paths), cfg.output_dir / "plots")

    summary_lines = [
        f"Total boardings:    {int(prepared['boardings'].sum())}",
        f"Total alightings:   {int(prepared['alightings'].sum())}",
        f"Stop visits used:   {len(prepared)}",
        f"Trips observed:     {prepared['trip_id_performed'].nunique()}",
        f"Months in panel:    {prepared['month'].nunique()}",
        f"Time periods:       {', '.join(sorted(prepared['time_period'].unique()))}",
        f"Grains exported:    {', '.join(grains)}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return grain_tables


# =============================================================================
# CLI / MAIN
# =============================================================================


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Ridership from TIDES stop_visits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    if args.stop_visits == STOP_VISITS_PATH or args.trips_performed == TRIPS_PERFORMED_PATH:
        logging.warning(
            "STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. Update the "
            "CONFIGURATION section or pass --stop-visits/--trips-performed before running."
        )
        return 2

    cfg = Config(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        time_periods=TIME_PERIODS,
        export_grains=EXPORT_GRAINS,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
    )

    if not cfg.stop_visits_path.exists():
        logging.warning("stop_visits not found: %s", cfg.stop_visits_path)
        return 1
    if not cfg.trips_performed_path.exists():
        logging.warning("trips_performed not found: %s", cfg.trips_performed_path)
        return 1

    run(cfg)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
