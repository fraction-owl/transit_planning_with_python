"""Roll trip-level ridership up to average boardings by hour of the service day.

Two input modes cover the common data situations. ``"route_trip_xlsx"`` reads the
pre-aggregated "Route and Trip" workbooks a vendor ridership database exports (one
row per scheduled trip carrying period-average passengers on/off and a trip start
time; one workbook per day type), and assigns each trip's average boardings to the
hour of its start time. ``"tides"`` derives the same table from raw TIDES
``stop_visits`` + ``trips_performed`` events, counting boardings at the hour each
stop visit actually occurred and averaging across the service dates observed for
each day type.

Hours are expressed on the GTFS-style service-day clock, the industry convention
for owl service: trips or events in the small hours (before
``LATE_NIGHT_CUTOVER_HOUR``) are treated as hours 24-27 of the prior service
day, so late-night service does not masquerade as early-morning ridership and
the default service day runs 04:00-27:59.

Inputs:
    - ``route_trip_xlsx`` mode: one workbook per day type (weekday/saturday/
      sunday), column names configurable via ``XLSX_COLUMN_MAP``.
    - ``tides`` mode: ``stop_visits.csv`` and ``trips_performed.csv`` (e.g. from
      ``convert_to_tides_stop_visits.py`` / ``convert_to_tides_trips_performed.py``).

Outputs:
    - ``ridership_by_hour_route.csv``: day type × route × hour average boardings
      (and alightings when available), with observation counts.
    - ``ridership_by_hour_system.csv``: day type × hour system totals.
    - ``ridership_by_trip_<day>.csv`` per day type (``EXPORT_TRIP_TABLE``):
      average daily boardings by ``trip_id`` (and ``stop_id`` in tides mode) —
      the exact ``RIDERSHIP_CSV`` input ``service_cut_impact_gpd.py`` expects;
      pass the file matching that script's ``SERVICE_DAY``.
    - ``low_ridership_trips.csv``: trips whose ``LOW_RIDERSHIP_MEASURE``
      (passengers per revenue hour by default; per revenue mile and raw
      boardings also available) falls below ``LOW_RIDERSHIP_THRESHOLD``, with
      route, start time, service-day hour, and all three measures; each
      flagged day type is also logged as a warning. Written only when trips
      are flagged; set the threshold to 0 to disable.
    - ``charts/``: one bar chart PNG per route per day type, plus a system chart
      per day type (``EXPORT_CHARTS``).
    - A run-log sidecar capturing the verbatim CONFIGURATION block.

Notes:
    The two modes time-locate boardings differently — trip start hour versus
    actual event hour — so expect small shifts between them for long trips.
    Holiday service is not separated: dates are classed weekday/saturday/sunday
    by calendar day alone in ``tides`` mode, and the xlsx export's own day-type
    split is trusted as-is. The trip tables are only as good as their ids: the
    xlsx export's trip-number column must hold GTFS ``trip_id`` values, and
    TIDES trips should carry ``trip_id_scheduled``; otherwise blank the
    ``trip_id`` mapping / expect unmatched ids downstream.

Typical usage:
    Update the paths in the CONFIGURATION section (or pass the matching CLI
    flags) and run from a shell, ArcGIS Pro's Python window, or a Jupyter
    notebook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402

# Sentinel markers used by extract_config_block / write_run_log to identify the
# configuration block within this file's source. Each string must appear exactly
# once in this file as a stand-alone comment line. Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# "route_trip_xlsx" (vendor Route-and-Trip average workbooks, one per day type)
# or "tides" (raw TIDES stop_visits + trips_performed event files).
INPUT_MODE: str = "route_trip_xlsx"

OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# --- route_trip_xlsx mode -----------------------------------------------------
# One workbook per day type. Remove entries you don't have.
XLSX_INPUTS: Mapping[str, str] = {
    "weekday": r"Path\To\RIDERSHIP_BY_ROUTE_AND_TRIP_weekday.xlsx",
    "saturday": r"Path\To\RIDERSHIP_BY_ROUTE_AND_TRIP_saturday.xlsx",
    "sunday": r"Path\To\RIDERSHIP_BY_ROUTE_AND_TRIP_sunday.xlsx",
}
# Sheet to read; None reads the first sheet in each workbook.
XLSX_SHEET: Optional[str] = None
# Source column names. route, trip_start_time, and boardings are required;
# set an optional entry to "" if the export lacks that column. trip_id must
# hold GTFS trip_id values for the engine-ready trip table to be meaningful —
# set it to "" if your export's trip numbers do not match GTFS.
XLSX_COLUMN_MAP: Mapping[str, str] = {
    "route": "ROUTE_NUMBER",
    "route_name": "ROUTE_NAME",
    "trip_id": "TRIP_NUMBER",
    "trip_start_time": "TRIP_START_TIME",
    "boardings": "PASSENGERS_ON",
    "alightings": "PASSENGERS_OFF",
    "days_observed": "TRIPS_COUNT",
    "revenue_hours": "REVENUE_HOURS",
    "revenue_miles": "REVENUE_MILES",
}

# --- tides mode ---------------------------------------------------------------
TIDES_STOP_VISITS_PATH: str = r"Path\To\stop_visits.csv"
TIDES_TRIPS_PERFORMED_PATH: str = r"Path\To\trips_performed.csv"
# First column found (and non-blank per row) provides each stop visit's time.
TIDES_TIME_COLUMNS: Sequence[str] = (
    "departure_time",
    "arrival_time",
    "scheduled_departure_time",
    "scheduled_arrival_time",
)
# Per-door APC count columns summed into total boardings / alightings.
TIDES_BOARDING_COLS: Sequence[str] = ("boarding_1", "boarding_2")
TIDES_ALIGHTING_COLS: Sequence[str] = ("alighting_1", "alighting_2")

# --- shared -------------------------------------------------------------------
# Events/trips starting before this clock hour belong to the prior service day
# and are shifted to hour 24+ (e.g. a 00:30 owl trip becomes hour 24, a 2:15
# trip hour 26). The default 4 gives the industry-standard 04:00-27:59 service
# day; set 3 if your agency's service day starts at 3am (hours run to 26).
LATE_NIGHT_CUTOVER_HOUR: int = 4

# Optional route filters (matched against the route column as strings).
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

ROUTE_OUTPUT_FILENAME: str = r"ridership_by_hour_route.csv"
SYSTEM_OUTPUT_FILENAME: str = r"ridership_by_hour_system.csv"

# Flag individual trips whose LOW_RIDERSHIP_MEASURE falls below this
# threshold. Flagged trips are logged as warnings per day type and written to
# LOW_RIDERSHIP_FILENAME with their route, start time, and service-day hour.
# Set the threshold to 0 to disable the flagging pass entirely.
#
# Measures: "pass_per_hour" (average boardings per revenue hour, the
# productivity measure most agency service standards use), "pass_per_mile"
# (per revenue mile), or "boardings" (raw average boardings per trip, which
# favors longer trips). xlsx mode computes the ratios from the revenue_hours /
# revenue_miles column mappings; tides mode derives hours from the
# trips_performed start/end times and has no mileage, so "pass_per_mile"
# works only in xlsx mode. Trips missing the measure are logged and skipped.
LOW_RIDERSHIP_MEASURE: str = "pass_per_hour"
LOW_RIDERSHIP_THRESHOLD: float = 5.0
LOW_RIDERSHIP_FILENAME: str = r"low_ridership_trips.csv"

# Also write ridership_by_trip_<day>.csv per day type — average daily boardings
# by trip_id (and stop_id in tides mode), in exactly the schema
# service_cut_impact_gpd.py's RIDERSHIP_CSV input expects. Pass the file whose
# day type matches that script's SERVICE_DAY.
EXPORT_TRIP_TABLE: bool = True
TRIP_TABLE_FILENAME_PREFIX: str = r"ridership_by_trip"

# Bar charts: one PNG per route per day type plus a system chart per day type.
EXPORT_CHARTS: bool = True
CHARTS_SUBDIR: str = r"charts"
CHART_MEASURE: str = "boardings"  # "boardings" or "alightings"

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Day types recognised, in output order.
_DAY_ORDER: List[str] = ["weekday", "saturday", "sunday"]

# Per-trip row schema shared by both input modes; feeds flag_low_ridership.
_TRIP_ROW_COLS: List[str] = [
    "day_type",
    "route",
    "route_name",
    "trip_id",
    "start_time",
    "hour",
    "boardings",
    "pass_per_hour",
    "pass_per_mile",
    "days_observed",
]

# Valid LOW_RIDERSHIP_MEASURE values.
_LOW_RIDERSHIP_MEASURES: Sequence[str] = ("boardings", "pass_per_hour", "pass_per_mile")

# Chart styling: series hue and neutral ink drawn from the repo's chart palette.
_BAR_COLOR: str = "#2a78d6"
_INK_COLOR: str = "#52514e"
_GRID_COLOR: str = "#e3e2de"


# =============================================================================
# CANONICAL HELPERS (copied verbatim from utils/ — see CONTRIBUTING.md)
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


# =============================================================================
# SMALL PRIVATE HELPERS
# =============================================================================


def _service_hour(clock_hour: int, cutover_hour: int) -> int:
    """Shift small-hours clock times onto the 24+ end of the service day."""
    if 0 <= clock_hour < cutover_hour:
        return clock_hour + 24
    return clock_hour


def _apply_route_filters(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the ROUTES_TO_INCLUDE / ROUTES_TO_EXCLUDE config filters."""
    if ROUTES_TO_INCLUDE:
        keep = {str(r) for r in ROUTES_TO_INCLUDE}
        table = table[table["route"].isin(keep)]
    if ROUTES_TO_EXCLUDE:
        drop = {str(r) for r in ROUTES_TO_EXCLUDE}
        table = table[~table["route"].isin(drop)]
    return table


def _sanitize_token(name: str) -> str:
    """Reduce a route/day label to a filesystem-safe token."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip()).strip("_")
    return token or "unnamed"


def _day_sort_key(day: str) -> int:
    """Order day types weekday, saturday, sunday, then anything else."""
    try:
        return _DAY_ORDER.index(day)
    except ValueError:
        return len(_DAY_ORDER)


# =============================================================================
# ROUTE-AND-TRIP XLSX MODE
# =============================================================================


def load_route_trip_workbook(path: Path, day_type: str) -> pd.DataFrame:
    """Read one Route-and-Trip workbook into tidy per-trip rows.

    Args:
        path: Workbook path.
        day_type: Day-type label to stamp on every row (e.g. ``"weekday"``).

    Returns:
        DataFrame with columns ``day_type``, ``route``, ``route_name``,
        ``trip_id`` (blank when unmapped), ``start_time``, ``hour``,
        ``boardings``, ``alightings``, ``days_observed``, ``revenue_hours``,
        ``revenue_miles`` — one row per scheduled trip. Unmapped optional
        columns come back as NaN.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a required mapped column is missing from the sheet.
    """
    if not path.exists():
        raise FileNotFoundError(f"Workbook for {day_type!r} not found: {path}")
    sheet: object = XLSX_SHEET if XLSX_SHEET else 0
    raw = pd.read_excel(path, sheet_name=sheet, dtype=str)
    if isinstance(raw, dict):  # future-proof: dict return if a list is ever passed
        raw = next(iter(raw.values()))

    colmap = {key: str(val).strip() for key, val in XLSX_COLUMN_MAP.items() if str(val).strip()}
    required = ("route", "trip_start_time", "boardings")
    missing = [colmap[k] for k in required if k in colmap and colmap[k] not in raw.columns]
    absent = [k for k in required if k not in colmap]
    if absent:
        raise ValueError(f"XLSX_COLUMN_MAP must map required key(s): {absent}")
    if missing:
        raise ValueError(
            f"Workbook '{path}' is missing mapped column(s): {missing}. "
            f"Columns present: {list(raw.columns)[:15]}…"
        )

    out = pd.DataFrame()
    out["route"] = raw[colmap["route"]].astype(str).str.strip()
    name_col = colmap.get("route_name")
    out["route_name"] = (
        raw[name_col].astype(str).str.strip() if name_col and name_col in raw.columns else ""
    )
    trip_col = colmap.get("trip_id")
    out["trip_id"] = (
        raw[trip_col].astype(str).str.strip() if trip_col and trip_col in raw.columns else ""
    )

    start_min = raw[colmap["trip_start_time"]].map(parse_time_to_minutes)
    bad_time = int(start_min.isna().sum())
    if bad_time:
        logging.warning(
            "%s: dropped %d row(s) with unparseable %s.",
            path.name,
            bad_time,
            colmap["trip_start_time"],
        )
    out["hour"] = start_min.map(
        lambda m: None if pd.isna(m) else _service_hour(int(m) // 60, LATE_NIGHT_CUTOVER_HOUR)
    )
    out["start_time"] = start_min.map(minutes_to_hhmm)

    out["boardings"] = pd.to_numeric(raw[colmap["boardings"]], errors="coerce")
    bad_board = int(out["boardings"].isna().sum()) - bad_time
    if bad_board > 0:
        logging.warning("%s: %d row(s) have non-numeric boardings.", path.name, bad_board)
    alight_col = colmap.get("alightings")
    if alight_col and alight_col in raw.columns:
        out["alightings"] = pd.to_numeric(raw[alight_col], errors="coerce")
    else:
        out["alightings"] = float("nan")
    for key in ("days_observed", "revenue_hours", "revenue_miles"):
        col = colmap.get(key)
        if col and col in raw.columns:
            out[key] = pd.to_numeric(raw[col], errors="coerce")
        else:
            out[key] = float("nan")

    out["day_type"] = day_type
    out = out[out["hour"].notna() & out["boardings"].notna()]
    logging.info("%s: %d trip row(s) loaded for %s.", path.name, len(out), day_type)
    return out


def build_hourly_from_xlsx(
    inputs: Mapping[str, str],
) -> "tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]":
    """Build the route × day × hour table from Route-and-Trip workbooks.

    Each trip's period-average boardings are assigned wholly to the hour of the
    trip's start time — a deliberate simplification for trips that span hours.

    Args:
        inputs: Mapping of day-type label to workbook path.

    Returns:
        Tuple of (hourly table, engine trip tables, per-trip rows). The hourly
        table has one row per (day_type, route, hour): summed average
        ``boardings``/``alightings``, scheduled ``trips`` in that hour, and the
        mean ``days_observed`` behind those averages. The trip tables map day
        type to a ``trip_id``/``stop_id``/``avg_daily_boardings`` frame ready
        for ``service_cut_impact_gpd.py`` (empty when ``trip_id`` is unmapped).
        The per-trip rows are one row per scheduled trip (``_TRIP_ROW_COLS``)
        for the low-ridership flagging pass, with ``pass_per_hour``/
        ``pass_per_mile`` computed from the revenue_hours / revenue_miles
        column mappings (NaN when unmapped or zero).

    Raises:
        ValueError: If *inputs* is empty.
    """
    if not inputs:
        raise ValueError("XLSX_INPUTS is empty — map at least one day type to a workbook.")
    frames = [
        load_route_trip_workbook(Path(path), str(day).strip().lower())
        for day, path in inputs.items()
    ]
    trips_rows = _apply_route_filters(pd.concat(frames, ignore_index=True))
    if trips_rows.empty:
        raise ValueError("No usable trip rows remain after parsing/filtering the workbooks.")

    grouped = trips_rows.groupby(["day_type", "route", "hour"], as_index=False).agg(
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        trips=("boardings", "size"),
        days_observed=("days_observed", "mean"),
    )
    names = (
        trips_rows[trips_rows["route_name"] != ""]
        .groupby("route")["route_name"]
        .agg(lambda s: s.mode().iloc[0])
    )
    grouped["route_name"] = grouped["route"].map(names).fillna("")

    trip_tables: dict[str, pd.DataFrame] = {}
    with_ids = trips_rows[trips_rows["trip_id"] != ""]
    if not with_ids.empty:
        for day_type, sub in with_ids.groupby("day_type"):
            dup = int(sub["trip_id"].duplicated().sum())
            if dup:
                logging.warning(
                    "%s workbook: %d duplicate trip_id value(s) — their boardings are summed.",
                    day_type,
                    dup,
                )
            table = sub.groupby("trip_id", as_index=False)["boardings"].sum()
            table = table.rename(columns={"boardings": "avg_daily_boardings"})
            table.insert(1, "stop_id", "")
            trip_tables[str(day_type)] = table
    trips_rows = trips_rows.copy()
    trips_rows["pass_per_hour"] = trips_rows["boardings"] / trips_rows["revenue_hours"].where(
        trips_rows["revenue_hours"] > 0
    )
    trips_rows["pass_per_mile"] = trips_rows["boardings"] / trips_rows["revenue_miles"].where(
        trips_rows["revenue_miles"] > 0
    )
    return grouped, trip_tables, trips_rows[_TRIP_ROW_COLS]


# =============================================================================
# TIDES MODE
# =============================================================================


def _coalesce_time_columns(visits: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Return the first non-blank time value per row across *candidates*."""
    present = [c for c in candidates if c in visits.columns]
    if not present:
        raise ValueError(
            f"stop_visits has none of the configured TIDES_TIME_COLUMNS: {list(candidates)}"
        )
    coalesced = visits[present[0]]
    for col in present[1:]:
        coalesced = coalesced.where(coalesced.notna() & (coalesced.astype(str) != ""), visits[col])
    return coalesced


def _event_service_hour(time_value: object, service_date: object, cutover_hour: int) -> float:
    """Locate one stop-visit event on the service-day hour clock.

    Full timestamps are compared against the service date, so an event logged
    at 00:30 on the calendar day after its service date lands at hour 24. An
    event timestamped in the small hours *of the service date itself* (a vendor
    that stamps owl events with the operating day's date) gets the same
    cutover shift, so hours below the cutover never appear either way.
    Time-only values fall back to the cutover rule.

    Args:
        time_value: Raw time cell (timestamp or ``HH:MM[:SS]`` string).
        service_date: Raw service_date cell for the same row.
        cutover_hour: Clock hours below this are shifted to 24+.

    Returns:
        Service-day hour as a float, or NaN when unparseable.
    """
    stamp = pd.to_datetime(time_value, errors="coerce")
    if not pd.isna(stamp):
        day = pd.to_datetime(service_date, errors="coerce")
        if not pd.isna(day):
            offset_days = (stamp.date() - day.date()).days
            if offset_days == 1:
                return float(stamp.hour + 24)
        return float(_service_hour(stamp.hour, cutover_hour))
    minutes = parse_time_to_minutes(str(time_value))
    if minutes is None:
        return float("nan")
    hour = minutes // 60
    if hour < 24:
        hour = _service_hour(hour, cutover_hour)
    return float(hour)


def _trip_duration_hours(performed: pd.DataFrame) -> pd.Series:
    """Per-row trip duration in hours from trips_performed start/end times.

    Prefers ``actual_trip_start``/``actual_trip_end`` and falls back to
    ``schedule_trip_start``/``schedule_trip_end`` per row. Rows where neither
    pair parses to a positive duration come back NaN.
    """
    duration = pd.Series(float("nan"), index=performed.index)
    for start_col, end_col in (
        ("actual_trip_start", "actual_trip_end"),
        ("schedule_trip_start", "schedule_trip_end"),
    ):
        if start_col not in performed.columns or end_col not in performed.columns:
            continue
        start = pd.to_datetime(performed[start_col], errors="coerce")
        end = pd.to_datetime(performed[end_col], errors="coerce")
        hours = (end - start).dt.total_seconds() / 3600.0
        duration = duration.fillna(hours.where(hours > 0))
    return duration


def build_hourly_from_tides(
    stop_visits_path: Path, trips_performed_path: Path
) -> "tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]":
    """Build the route × day × hour table from TIDES event files.

    Boardings are counted at the hour each stop visit occurred and averaged
    over the distinct service dates observed for each day type, so the result
    is an average *daily* profile comparable to the xlsx mode's.

    Args:
        stop_visits_path: TIDES ``stop_visits`` CSV.
        trips_performed_path: TIDES ``trips_performed`` CSV.

    Returns:
        Tuple of (hourly table, engine trip tables, per-trip rows). The hourly
        table has one row per (day_type, route, hour): average daily
        ``boardings``/``alightings``, average daily distinct ``trips``, and the
        number of service dates observed (``days_observed``). The trip tables
        map day type to a ``trip_id``/``stop_id``/``avg_daily_boardings`` frame
        ready for ``service_cut_impact_gpd.py``, keyed on ``trip_id_scheduled``
        (the GTFS trip_id) when trips_performed carries it, with per-stop
        grain when stop_visits carries ``stop_id``. The per-trip rows are one
        row per trip for the low-ridership flagging pass: average daily
        boardings across the dates each trip was observed, with ``hour`` set
        to the trip's earliest observed stop-visit hour, ``pass_per_hour``
        derived from the trips_performed actual (or scheduled) start/end
        times, ``pass_per_mile`` NaN (no mileage in TIDES event data), and
        ``start_time`` blank (no scheduled start time in the event data).

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If required columns are missing or nothing is parseable.
    """
    for path in (stop_visits_path, trips_performed_path):
        if not path.exists():
            raise FileNotFoundError(f"TIDES file not found: {path}")
    visits = pd.read_csv(stop_visits_path, dtype=str, low_memory=False)
    performed = pd.read_csv(trips_performed_path, dtype=str, low_memory=False)

    for frame, label in ((visits, "stop_visits"), (performed, "trips_performed")):
        for col in ("trip_id_performed", "service_date"):
            if col not in frame.columns:
                raise ValueError(f"{label} is missing required column {col!r}.")
    if "route_id" not in performed.columns:
        raise ValueError("trips_performed is missing required column 'route_id'.")

    board_cols = [c for c in TIDES_BOARDING_COLS if c in visits.columns]
    if not board_cols:
        raise ValueError(
            f"stop_visits has none of the configured TIDES_BOARDING_COLS: "
            f"{list(TIDES_BOARDING_COLS)}"
        )
    alight_cols = [c for c in TIDES_ALIGHTING_COLS if c in visits.columns]
    for col in board_cols + alight_cols:
        visits[col] = pd.to_numeric(visits[col], errors="coerce")
    visits["boardings"] = visits[board_cols].sum(axis=1)
    visits["alightings"] = visits[alight_cols].sum(axis=1) if alight_cols else float("nan")

    time_raw = _coalesce_time_columns(visits, TIDES_TIME_COLUMNS)
    visits["hour"] = [
        _event_service_hour(t, d, LATE_NIGHT_CUTOVER_HOUR)
        for t, d in zip(time_raw, visits["service_date"])
    ]
    bad = int(visits["hour"].isna().sum())
    if bad:
        logging.warning("stop_visits: dropped %d row(s) with unparseable event times.", bad)
    visits = visits[visits["hour"].notna()].copy()
    visits["hour"] = visits["hour"].astype(int)

    routes = performed.drop_duplicates(subset=["service_date", "trip_id_performed"]).copy()
    routes["duration_hours"] = _trip_duration_hours(routes)
    route_cols = ["service_date", "trip_id_performed", "route_id", "duration_hours"]
    if "trip_id_scheduled" in routes.columns:
        route_cols.append("trip_id_scheduled")
    events = visits.merge(routes[route_cols], on=["service_date", "trip_id_performed"], how="left")
    unmatched = int(events["route_id"].isna().sum())
    if unmatched:
        logging.warning(
            "stop_visits: %d visit(s) had no matching trips_performed row; they are excluded.",
            unmatched,
        )
        events = events[events["route_id"].notna()]
    if events.empty:
        raise ValueError("No stop visits joined to trips_performed — check the two files.")

    day = pd.to_datetime(events["service_date"], errors="coerce")
    events = events[day.notna()].copy()
    dow = day[day.notna()].dt.dayofweek
    events["day_type"] = "weekday"
    events.loc[dow == 5, "day_type"] = "saturday"
    events.loc[dow == 6, "day_type"] = "sunday"
    events["route"] = events["route_id"].astype(str).str.strip()
    events = _apply_route_filters(events)

    n_dates = events.groupby("day_type")["service_date"].nunique()
    totals = events.groupby(["day_type", "route", "hour"], as_index=False).agg(
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        trip_visits=("trip_id_performed", "nunique"),
    )
    totals["days_observed"] = totals["day_type"].map(n_dates)
    totals["boardings"] = totals["boardings"] / totals["days_observed"]
    totals["alightings"] = totals["alightings"] / totals["days_observed"]
    totals["trips"] = totals["trip_visits"] / totals["days_observed"]
    totals["route_name"] = ""
    for day_type, count in n_dates.items():
        logging.info("TIDES: %s averaged over %d service date(s).", day_type, count)

    if "trip_id_scheduled" in events.columns:
        sched = events["trip_id_scheduled"].fillna("").astype(str).str.strip()
        events["trip_key"] = sched.where(sched != "", events["trip_id_performed"])
    else:
        events["trip_key"] = events["trip_id_performed"]
    if "stop_id" in events.columns:
        events["_stop"] = events["stop_id"].fillna("").astype(str).str.strip()
    else:
        events["_stop"] = ""
    trip_tables: dict[str, pd.DataFrame] = {}
    trip_row_frames: list[pd.DataFrame] = []
    for day_type, sub in events.groupby("day_type"):
        trip_dates = sub.groupby("trip_key")["service_date"].nunique()
        table = sub.groupby(["trip_key", "_stop"], as_index=False)["boardings"].sum()
        table["avg_daily_boardings"] = table["boardings"] / table["trip_key"].map(trip_dates)
        table = table.rename(columns={"trip_key": "trip_id", "_stop": "stop_id"})
        trip_tables[str(day_type)] = table[["trip_id", "stop_id", "avg_daily_boardings"]]

        detail = sub.groupby("trip_key", as_index=False).agg(
            route=("route", "first"),
            hour=("hour", "min"),
            boardings=("boardings", "sum"),
            days_observed=("service_date", "nunique"),
        )
        detail["boardings"] = detail["boardings"] / detail["days_observed"]
        per_date = sub.drop_duplicates(subset=["trip_key", "service_date"])
        mean_duration = per_date.groupby("trip_key")["duration_hours"].mean()
        duration = detail["trip_key"].map(mean_duration)
        detail["pass_per_hour"] = detail["boardings"] / duration.where(duration > 0)
        detail["pass_per_mile"] = float("nan")  # no mileage in TIDES event data
        detail["day_type"] = str(day_type)
        detail["route_name"] = ""
        detail["start_time"] = ""
        detail = detail.rename(columns={"trip_key": "trip_id"})
        trip_row_frames.append(detail)
    trip_rows = (
        pd.concat(trip_row_frames, ignore_index=True)[_TRIP_ROW_COLS]
        if trip_row_frames
        else pd.DataFrame(columns=_TRIP_ROW_COLS)
    )
    return totals.drop(columns=["trip_visits"]), trip_tables, trip_rows


# =============================================================================
# OUTPUT TABLES & CHARTS
# =============================================================================


def finalize_tables(hourly: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Sort, label, and round the route table; derive the system rollup.

    Args:
        hourly: Route × day × hour table from either input mode.

    Returns:
        Tuple of (route table, system table), both ready to write.
    """
    table = hourly.copy()
    table["hour"] = table["hour"].astype(int)
    table["hour_label"] = table["hour"].map(lambda h: minutes_to_hhmm(h * 60))
    table["_day_order"] = table["day_type"].map(_day_sort_key)
    table = table.sort_values(["_day_order", "route", "hour"]).drop(columns="_day_order")
    for col in ("boardings", "alightings", "trips", "days_observed"):
        table[col] = pd.to_numeric(table[col], errors="coerce").round(2)

    system = table.groupby(["day_type", "hour", "hour_label"], as_index=False).agg(
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        routes=("route", "nunique"),
    )
    system["_day_order"] = system["day_type"].map(_day_sort_key)
    system = system.sort_values(["_day_order", "hour"]).drop(columns="_day_order")
    system[["boardings", "alightings"]] = system[["boardings", "alightings"]].round(2)

    cols = [
        "day_type",
        "route",
        "route_name",
        "hour",
        "hour_label",
        "boardings",
        "alightings",
        "trips",
        "days_observed",
    ]
    return table[cols], system


def flag_low_ridership(
    trip_rows: pd.DataFrame, measure: str, threshold: float, out_path: Path
) -> int:
    """Log and export trips whose *measure* falls below *threshold*.

    Flagged trips are written to *out_path* sorted by day type, route, and
    service-day hour, and a warning is logged per day type so low performers
    surface in the console and run log even when nobody opens the CSV.

    Args:
        trip_rows: Per-trip rows from either input mode (``_TRIP_ROW_COLS``).
        measure: Column to test — ``"boardings"``, ``"pass_per_hour"``, or
            ``"pass_per_mile"``. Trips where the measure is NaN (unmapped
            revenue columns, missing trip times) are skipped with a warning.
        threshold: Trips with *measure* strictly below this are flagged.
            Values <= 0 disable the pass.
        out_path: Destination CSV (only written when trips are flagged).

    Returns:
        Number of trips flagged.

    Raises:
        ValueError: On an unrecognised *measure*.
    """
    if measure not in _LOW_RIDERSHIP_MEASURES:
        raise ValueError(
            f"LOW_RIDERSHIP_MEASURE must be one of {list(_LOW_RIDERSHIP_MEASURES)}; "
            f"got {measure!r}."
        )
    if threshold <= 0 or trip_rows.empty:
        return 0
    values = trip_rows[measure]
    missing = int(values.isna().sum())
    if missing:
        logging.warning(
            "%d trip(s) have no %s value (unmapped revenue columns or missing trip "
            "times) and were not evaluated for the low-ridership flag.",
            missing,
            measure,
        )
    flagged = trip_rows[values < threshold].copy()
    if flagged.empty:
        logging.info(
            "No trips below %.1f %s (%d trips checked).",
            threshold,
            measure,
            len(trip_rows) - missing,
        )
        return 0
    flagged["hour"] = flagged["hour"].astype(int)
    flagged.insert(
        flagged.columns.get_loc("hour") + 1,
        "hour_label",
        flagged["hour"].map(lambda h: minutes_to_hhmm(h * 60)),
    )
    flagged["_day_order"] = flagged["day_type"].map(_day_sort_key)
    flagged = flagged.sort_values(["_day_order", "route", "hour", measure]).drop(
        columns="_day_order"
    )
    round_cols = ["boardings", "pass_per_hour", "pass_per_mile", "days_observed"]
    flagged[round_cols] = flagged[round_cols].round(2)
    flagged.to_csv(out_path, index=False)
    for day_type in sorted(flagged["day_type"].unique(), key=_day_sort_key):
        sub = flagged[flagged["day_type"] == day_type]
        logging.warning(
            "%s: %d of %d trip(s) below %.1f %s, on %d route(s).",
            day_type,
            len(sub),
            int((trip_rows["day_type"] == day_type).sum()),
            threshold,
            measure,
            sub["route"].nunique(),
        )
    logging.info("Wrote: %s (%d rows)", out_path, len(flagged))
    return len(flagged)


def draw_hour_chart(
    sub: pd.DataFrame, title: str, measure: str, out_path: Path, hour_span: Sequence[int]
) -> None:
    """Draw one average-by-hour bar chart PNG.

    Args:
        sub: Rows for one route (or the system) on one day type.
        title: Chart title.
        measure: Column to plot (``"boardings"`` or ``"alightings"``).
        out_path: PNG destination.
        hour_span: Continuous hour axis to draw (hours absent from *sub* get a
            zero bar), so charts of the same day type line up across routes.
    """
    by_hour = sub.set_index("hour")[measure]
    values = [float(by_hour.get(h, 0.0) or 0.0) for h in hour_span]
    labels = [minutes_to_hhmm(h * 60) for h in hour_span]

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(range(len(values)), values, color=_BAR_COLOR, width=0.72)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=_INK_COLOR)
    ax.tick_params(axis="y", labelsize=8, colors=_INK_COLOR)
    ax.set_ylabel(f"Avg daily {measure}", fontsize=9, color=_INK_COLOR)
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_charts(route_table: pd.DataFrame, system: pd.DataFrame, out_dir: Path) -> int:
    """Write one chart per route per day type, plus system charts.

    Args:
        route_table: Finalized route table.
        system: Finalized system table.
        out_dir: Charts folder (created if needed).

    Returns:
        Number of PNGs written.
    """
    measure = CHART_MEASURE.strip().lower()
    if measure not in ("boardings", "alightings"):
        raise ValueError(f"CHART_MEASURE must be 'boardings' or 'alightings'; got {measure!r}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    hour_span = list(range(int(route_table["hour"].min()), int(route_table["hour"].max()) + 1))
    written = 0
    for (day_type, route), sub in route_table.groupby(["day_type", "route"]):
        name = sub["route_name"].iloc[0]
        label = f"Route {route}" + (f" ({name})" if name and name != str(route) else "")
        out_path = out_dir / f"route_{_sanitize_token(route)}_{_sanitize_token(day_type)}.png"
        draw_hour_chart(
            sub, f"{label} — {day_type} {measure} by hour", measure, out_path, hour_span
        )
        written += 1
    for day_type, sub in system.groupby("day_type"):
        out_path = out_dir / f"system_{_sanitize_token(day_type)}.png"
        draw_hour_chart(
            sub, f"All routes — {day_type} {measure} by hour", measure, out_path, hour_span
        )
        written += 1
    logging.info("Wrote %d chart PNG(s) to '%s'.", written, out_dir)
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


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "ridership_by_hour_runlog.txt"

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
        "RIDERSHIP BY HOUR RUN LOG",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "RUN SUMMARY",
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
# PIPELINE / CLI / MAIN
# =============================================================================


def run(
    input_mode: str,
    output_dir: Path,
    xlsx_inputs: Mapping[str, str],
    stop_visits_path: Path,
    trips_performed_path: Path,
    export_charts_flag: bool,
) -> pd.DataFrame:
    """Build the hourly tables, write CSVs and charts, and log a summary.

    Args:
        input_mode: ``"route_trip_xlsx"`` or ``"tides"``.
        output_dir: Destination folder.
        xlsx_inputs: Day-type → workbook mapping (xlsx mode).
        stop_visits_path: TIDES stop_visits CSV (tides mode).
        trips_performed_path: TIDES trips_performed CSV (tides mode).
        export_charts_flag: Write chart PNGs when True.

    Returns:
        The route × day × hour table (also written to disk).

    Raises:
        ValueError: On an unrecognised *input_mode*.
    """
    mode = input_mode.strip().lower()
    if mode == "route_trip_xlsx":
        hourly, trip_tables, trip_rows = build_hourly_from_xlsx(xlsx_inputs)
    elif mode == "tides":
        hourly, trip_tables, trip_rows = build_hourly_from_tides(
            stop_visits_path, trips_performed_path
        )
    else:
        raise ValueError(f"INPUT_MODE must be 'route_trip_xlsx' or 'tides'; got {input_mode!r}.")

    route_table, system = finalize_tables(hourly)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_path = output_dir / ROUTE_OUTPUT_FILENAME
    system_path = output_dir / SYSTEM_OUTPUT_FILENAME
    route_table.to_csv(route_path, index=False)
    system.to_csv(system_path, index=False)
    logging.info("Wrote: %s (%d rows)", route_path, len(route_table))
    logging.info("Wrote: %s (%d rows)", system_path, len(system))

    trip_files = 0
    if EXPORT_TRIP_TABLE:
        if trip_tables:
            for day_type in sorted(trip_tables, key=_day_sort_key):
                table = trip_tables[day_type].copy()
                table["avg_daily_boardings"] = table["avg_daily_boardings"].round(2)
                trip_path = (
                    output_dir / f"{TRIP_TABLE_FILENAME_PREFIX}_{_sanitize_token(day_type)}.csv"
                )
                table.to_csv(trip_path, index=False)
                trip_files += 1
                logging.info(
                    "Wrote: %s (%d rows) — feed to service_cut_impact_gpd.py via "
                    "RIDERSHIP_CSV for a %s analysis.",
                    trip_path,
                    len(table),
                    day_type,
                )
        else:
            logging.warning(
                "EXPORT_TRIP_TABLE is on but no trip ids are available — map "
                "XLSX_COLUMN_MAP['trip_id'] (xlsx mode) to write engine-ready trip tables."
            )

    low_flagged = flag_low_ridership(
        trip_rows,
        LOW_RIDERSHIP_MEASURE.strip().lower(),
        LOW_RIDERSHIP_THRESHOLD,
        output_dir / LOW_RIDERSHIP_FILENAME,
    )

    charts_written = 0
    if export_charts_flag:
        charts_written = export_charts(route_table, system, output_dir / CHARTS_SUBDIR)

    day_types = ", ".join(sorted(route_table["day_type"].unique(), key=_day_sort_key))
    low_summary = (
        f"{low_flagged} trip(s) below {LOW_RIDERSHIP_THRESHOLD:g} "
        f"{LOW_RIDERSHIP_MEASURE.strip().lower()}"
        if LOW_RIDERSHIP_THRESHOLD > 0
        else "disabled (LOW_RIDERSHIP_THRESHOLD <= 0)"
    )
    summary_lines = [
        f"Input mode:        {mode}",
        f"Day types:         {day_types}",
        f"Routes:            {route_table['route'].nunique()}",
        f"Route rows:        {len(route_table)}",
        f"Trip tables:       {trip_files}",
        f"Low ridership:     {low_summary}",
        f"Charts written:    {charts_written}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return route_table


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Average ridership by hour of day from Route-and-Trip xlsx or TIDES events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        default=INPUT_MODE,
        choices=("route_trip_xlsx", "tides"),
        help="Input mode.",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--weekday", default=XLSX_INPUTS.get("weekday", ""), help="Weekday workbook (xlsx mode)."
    )
    p.add_argument(
        "--saturday",
        default=XLSX_INPUTS.get("saturday", ""),
        help="Saturday workbook (xlsx mode).",
    )
    p.add_argument(
        "--sunday", default=XLSX_INPUTS.get("sunday", ""), help="Sunday workbook (xlsx mode)."
    )
    p.add_argument(
        "--stop-visits",
        default=TIDES_STOP_VISITS_PATH,
        help="TIDES stop_visits CSV (tides mode).",
    )
    p.add_argument(
        "--trips-performed",
        default=TIDES_TRIPS_PERFORMED_PATH,
        help="TIDES trips_performed CSV (tides mode).",
    )
    p.add_argument(
        "--charts",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_CHARTS,
        help="Write bar-chart PNGs (--charts) or skip them (--no-charts).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
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

    if args.output_dir == OUTPUT_DIR and OUTPUT_DIR.startswith("Path"):
        logging.warning(
            "OUTPUT_DIR is still a placeholder. Update the CONFIGURATION section or pass "
            "--output-dir before running."
        )
        return 2
    xlsx_inputs = {
        day: path
        for day, path in (
            ("weekday", args.weekday),
            ("saturday", args.saturday),
            ("sunday", args.sunday),
        )
        if str(path).strip() and not str(path).startswith("Path")
    }
    if args.mode == "route_trip_xlsx" and not xlsx_inputs:
        logging.warning(
            "XLSX_INPUTS paths are still placeholders. Update the CONFIGURATION section or "
            "pass --weekday/--saturday/--sunday before running."
        )
        return 2
    if args.mode == "tides" and str(args.stop_visits).startswith("Path"):
        logging.warning(
            "TIDES paths are still placeholders. Update the CONFIGURATION section or pass "
            "--stop-visits/--trips-performed before running."
        )
        return 2

    try:
        run(
            input_mode=args.mode,
            output_dir=Path(args.output_dir).expanduser(),
            xlsx_inputs=xlsx_inputs,
            stop_visits_path=Path(str(args.stop_visits)).expanduser(),
            trips_performed_path=Path(str(args.trips_performed)).expanduser(),
            export_charts_flag=bool(args.charts),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
