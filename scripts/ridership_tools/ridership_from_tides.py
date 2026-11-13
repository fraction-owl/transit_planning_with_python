"""Export TIDES ridership and maximum load at every grain in one run.

Reads TIDES ``stop_visits`` (per-door ``boarding_*`` / ``alighting_*`` counts and
``departure_load``) joined to ``trips_performed``, and writes one CSV per grain. A grain
is an aggregation level crossed with a time cut; ``EXPORT_LEVELS`` and
``EXPORT_TIME_CUTS`` switch each one on or off, and all are on by default.

- Levels: system, service_type, route, route_direction, stop, route_stop,
  route_direction_stop, trip, trip_stop.
- Time cuts: all_day, time_period, hour, month, month_time_period.

Every table is split by day type. ``DAY_TYPES`` maps day-type names to weekdays,
``START_DATE`` / ``END_DATE`` limit the dates, and dates listed in ``HOLIDAYS`` form a
day type of their own. Averages divide by the service dates that have observed data:
per day type (and month) across the system, or per trip for the trip levels. Each row's
``days_observed`` column shows the divisor.

Time periods come from ``TIME_PERIODS``, a ``TIME_PERIODS_FILE`` (see
``time_periods_example.txt``) or ``--time-period`` flags, and must cover the day exactly
once. A visit falls in a period by its clock time, and in an hour of the service day
(04:00-27:59 by default; see ``LATE_NIGHT_CUTOVER_HOUR``).

Inputs
------
- TIDES ``stop_visits.csv`` and ``trips_performed.csv`` (for example from
  ``convert_to_tides_stop_visits.py`` and ``convert_to_tides_trips_performed.py``).
- Optional: a time-periods ``.txt`` file, one ``NAME = HH:MM - HH:MM`` per line.

Outputs
-------
- ``ridership/ridership_by_<level>[_and_<cut>].csv``: average daily and total boardings
  and alightings, with the dates, stop visits and trips observed behind them.
- ``max_load/max_load_by_performed_trip.csv``: each performed trip's maximum
  ``departure_load``, peak load point and load factor, plus
  ``max_load_by_<level>[_and_<cut>].csv`` rollups for the system, service_type, route,
  route_direction and trip levels.
- ``service_cut_inputs/ridership_by_trip_<day type>.csv``: the ``RIDERSHIP_CSV`` input of
  ``service_cut_impact_gpd.py``.
- ``low_ridership_trips.csv``: trips below ``LOW_RIDERSHIP_THRESHOLD``.
- ``charts/``: bar charts by time period, hour and month, and per-trip charts.
- ``ridership_from_tides_runlog.txt``: settings, time periods and the CONFIGURATION block.

Limitations
-----------
- Counts are raw APC observations, not expanded to systemwide estimates; the stop-visit
  and trip counts show how much data backs each row.
- Canceled and non-revenue trips are dropped. Skipped visits are left out of ridership
  but kept for maximum load.
- A trip that crosses a period or hour boundary counts in both; maximum load places
  each trip by its start time instead.

Typical usage
-------------
Update the paths in CONFIGURATION (or pass ``--stop-visits`` / ``--trips-performed`` /
``--output-dir``) and run from a shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# GRAINS ----------------------------------------------------------------------
# Aggregation levels. Each level switched on is written once for every time cut
# switched on below, as its own CSV. Set an entry to False to skip that level.
#   system               - all routes pooled
#   service_type         - per route_type_agency (e.g. LOCAL, EXPRESS)
#   route                - per route_id
#   route_direction      - per route_id x direction_id
#   stop                 - per stop_id, all routes pooled
#   route_stop           - per route_id x stop_id, both directions pooled
#   route_direction_stop - per route_id x direction_id x stop_id
#   trip                 - per scheduled trip (trip_id_scheduled, else trip_id_performed)
#   trip_stop            - per scheduled trip x stop_id
# Maximum-load tables use the system, service_type, route, route_direction and
# trip levels.
EXPORT_LEVELS: Mapping[str, bool] = {
    "system": True,
    "service_type": True,
    "route": True,
    "route_direction": True,
    "stop": True,
    "route_stop": True,
    "route_direction_stop": True,
    "trip": True,
    "trip_stop": True,
}

# Time cuts, each crossed with every level switched on above:
#   all_day           - one row per day type
#   time_period       - per TIME_PERIODS window (by clock time)
#   hour              - per hour of the service day (see LATE_NIGHT_CUTOVER_HOUR)
#   month             - per calendar month of service_date
#   month_time_period - per month x time period
EXPORT_TIME_CUTS: Mapping[str, bool] = {
    "all_day": True,
    "time_period": True,
    "hour": True,
    "month": True,
    "month_time_period": True,
}

# Maximum load (largest departure_load on each performed trip) and its rollups.
EXPORT_MAX_LOAD: bool = True
# Seated + standee capacity used to turn a trip's maximum load into a load
# factor. One agency-wide value; for mixed fleets, filter by route.
VEHICLE_CAPACITY: int = 39

# DAY TYPES -------------------------------------------------------------------
# Day types as {name: weekdays} (full names or three-letter abbreviations).
# Every table is split by day type, in this order. A weekday in no entry is
# left out; a weekday in several entries counts in each. To pool every date,
# use {"All days": ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")}.
DAY_TYPES: Mapping[str, Sequence[str]] = {
    "Weekday": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"),
    "Saturday": ("Saturday",),
    "Sunday": ("Sunday",),
}

# Holiday service dates as "YYYY-MM-DD". They leave the day type of their
# weekday and form a day type of their own, named HOLIDAY_DAY_TYPE. Leave the
# list empty and holidays count as the weekday they fall on.
HOLIDAYS: Sequence[str] = ()
HOLIDAY_DAY_TYPE: str = "Holiday"

# Optional service-date range as "YYYY-MM-DD", both ends inclusive. None = no limit.
START_DATE: Optional[str] = None
END_DATE: Optional[str] = None

# TIME OF DAY -----------------------------------------------------------------
# Time periods as {name: ("HH:MM", "HH:MM")} on the 24-hour clock, start
# inclusive and end exclusive. An end earlier than the start wraps past
# midnight, and "24:00" may be used as an end. Together the periods must cover
# every minute of the day exactly once; the script stops before reading any
# data if there is a gap or an overlap. Output rows follow the order listed.
TIME_PERIODS: Mapping[str, Tuple[str, str]] = {
    "Early AM": ("04:00", "06:00"),
    "AM Peak": ("06:00", "09:00"),
    "Midday": ("09:00", "15:00"),
    "PM Peak": ("15:00", "18:00"),
    "Late PM": ("18:00", "22:00"),
    "Night": ("22:00", "04:00"),
}

# Optional plain-text file of time periods. When set, it replaces TIME_PERIODS.
# One period per line as "NAME = HH:MM - HH:MM"; blank lines and anything after
# "#" are ignored. See time_periods_example.txt next to this script. The same
# checks apply. --time-period flags on the command line replace both.
TIME_PERIODS_FILE: Optional[str] = None  # e.g. r"C:\Data\time_periods.txt"

# stop_visits columns tried in order for each visit's time of day; the first
# readable value in each row is used.
TIME_COLUMNS: Sequence[str] = (
    "actual_departure_time",
    "actual_arrival_time",
    "schedule_departure_time",
    "schedule_arrival_time",
)

# Clock hours before this belong to the previous service day and appear as
# hours 24 and up in the hour cut (a 00:30 visit is hour 24). The default 4
# gives the usual 04:00-27:59 service day.
LATE_NIGHT_CUTOVER_HOUR: int = 4

# FILTERS ---------------------------------------------------------------------
# Optional route filters, matched against route_id as text. Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# LOW-RIDERSHIP TRIPS ---------------------------------------------------------
# Trips whose LOW_RIDERSHIP_MEASURE falls below the threshold are written to
# low_ridership_trips.csv and logged. Measures: "pass_per_hour" (average daily
# boardings per hour of trip running time, from the trips_performed start and
# end times) or "boardings" (average daily boardings per trip). Set the
# threshold to 0 to switch the flag off.
LOW_RIDERSHIP_MEASURE: str = "pass_per_hour"
LOW_RIDERSHIP_THRESHOLD: float = 5.0

# OTHER OUTPUTS ---------------------------------------------------------------
# One trip_id / stop_id / avg_daily_boardings CSV per day type, the RIDERSHIP_CSV
# input service_cut_impact_gpd.py expects. Pass the file whose day type matches
# that script's SERVICE_DAY.
EXPORT_SERVICE_CUT_TRIP_TABLES: bool = True

# Bar charts of average daily boardings (and of mean / peak maximum load) by time
# period, hour and month, one PNG per group per day type, for CHART_LEVELS
# (any of system, service_type, route, route_direction). Also one per-trip chart
# per route, direction and day type on the LOW_RIDERSHIP_MEASURE scale.
EXPORT_CHARTS: bool = True
CHART_LEVELS: Sequence[str] = ("system", "route")

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# CONSTANTS
# =============================================================================

# Level -> its key columns as (column, required). A level whose required column
# is missing from the data is skipped with a warning; optional columns (the
# route and direction beside a trip) are carried when present.
LEVELS: Dict[str, Tuple[Tuple[str, bool], ...]] = {
    "system": (),
    "service_type": (("route_type_agency", True),),
    "route": (("route_id", True),),
    "route_direction": (("route_id", True), ("direction_id", True)),
    "stop": (("stop_id", True),),
    "route_stop": (("route_id", True), ("stop_id", True)),
    "route_direction_stop": (("route_id", True), ("direction_id", True), ("stop_id", True)),
    "trip": (("route_id", False), ("direction_id", False), ("trip_id", True)),
    "trip_stop": (
        ("route_id", False),
        ("direction_id", False),
        ("trip_id", True),
        ("stop_id", True),
    ),
}

# Levels averaged over the dates each trip was observed rather than system-wide.
TRIP_LEVELS: frozenset[str] = frozenset({"trip", "trip_stop"})

# Levels whose time-period tables list every period for every group, with zeros
# where nothing was observed. Stop and trip groups are not filled: a stop or trip
# with no service in a period has no row rather than a misleading zero.
ZERO_FILL_LEVELS: frozenset[str] = frozenset({"system", "service_type", "route", "route_direction"})

# Levels that apply to maximum load (a per-trip measure has no stop level).
MAX_LOAD_LEVELS: Tuple[str, ...] = ("system", "service_type", "route", "route_direction", "trip")

# Levels that can be charted (one chart per group).
CHARTABLE_LEVELS: Tuple[str, ...] = ("system", "service_type", "route", "route_direction")

# Time cut -> the time columns it adds to each level's keys.
TIME_CUTS: Dict[str, Tuple[str, ...]] = {
    "all_day": (),
    "time_period": ("time_period",),
    "hour": ("hour",),
    "month": ("month",),
    "month_time_period": ("month", "time_period"),
}

# Time cuts drawn as charts, with one bar per period, hour or month.
CHART_CUTS: Tuple[str, ...] = ("time_period", "hour", "month")

# LOW_RIDERSHIP_MEASURE value -> the per-trip column it tests.
LOW_RIDERSHIP_MEASURES: Dict[str, str] = {
    "pass_per_hour": "pass_per_hour",
    "boardings": "avg_daily_boardings",
}

WEEKDAY_NAMES: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

MINUTES_PER_DAY: int = 24 * 60
BLANK_LABEL: str = "(blank)"
UNCLASSIFIED: str = "UNCLASSIFIED"

# Per-door APC count columns summed into total boardings / alightings.
BOARDING_COLUMNS: Tuple[str, ...] = ("boarding_1", "boarding_2")
ALIGHTING_COLUMNS: Tuple[str, ...] = ("alighting_1", "alighting_2")

# trips_performed columns carried onto each stop visit (where present).
TRIP_ATTRIBUTE_COLUMNS: Tuple[str, ...] = (
    "route_id",
    "direction_id",
    "route_type_agency",
    "trip_id_scheduled",
    "schedule_trip_start",
    "schedule_trip_end",
    "actual_trip_start",
    "actual_trip_end",
)

# Columns used as group keys; blanks in them are reported as BLANK_LABEL.
GROUP_KEY_COLUMNS: Tuple[str, ...] = (
    "route_id",
    "direction_id",
    "route_type_agency",
    "stop_id",
    "trip_id",
)

RIDERSHIP_SUBDIR: str = "ridership"
MAX_LOAD_SUBDIR: str = "max_load"
SERVICE_CUT_SUBDIR: str = "service_cut_inputs"
CHARTS_SUBDIR: str = "charts"
MAX_LOAD_BY_TRIP_FILENAME: str = "max_load_by_performed_trip.csv"
LOW_RIDERSHIP_FILENAME: str = "low_ridership_trips.csv"
RUN_LOG_FILENAME: str = "ridership_from_tides_runlog.txt"
RUN_LOG_TITLE: str = "RIDERSHIP FROM TIDES RUN LOG"

# Chart styling: series hues and neutral ink from the repo's chart palette.
_SERIES_COLORS: Tuple[str, ...] = ("#2a78d6", "#e08b2c")
_INK_COLOR: str = "#52514e"
_GRID_COLOR: str = "#e3e2de"
# Status red for below-threshold trips and the threshold line; never the only
# encoding (the dashed line and the legend carry the same meaning).
_FLAG_COLOR: str = "#d03b3b"

# =============================================================================
# ERRORS & SETTINGS
# =============================================================================


class TimePeriodError(ValueError):
    """Raised when the time periods are malformed or do not cover the day exactly once."""


class SettingsError(ValueError):
    """Raised when a setting (day types, dates, grain flags, measure) is unusable."""


class InputDataError(ValueError):
    """Raised when the TIDES tables cannot support the requested exports."""


class PeriodSpec(NamedTuple):
    """A time period as written, before its times are checked."""

    name: str
    start: str
    end: str
    where: str  # where it was written, for error messages


class TimePeriod(NamedTuple):
    """A checked time period in minutes since midnight.

    ``start`` runs 0-1439 and ``end`` runs 1-1440. A period whose ``end`` is not
    after its ``start`` wraps past midnight.
    """

    name: str
    start: int
    end: int

    @property
    def minutes(self) -> int:
        """Length of the period in minutes."""
        if self.end > self.start:
            return self.end - self.start
        return self.end + MINUTES_PER_DAY - self.start

    @property
    def span(self) -> str:
        """The period's window as ``HH:MM-HH:MM``."""
        return f"{_clock(self.start)}-{_clock(self.end)}"


class RunSettings(NamedTuple):
    """Resolved settings for one run.

    ``None`` for ``levels``, ``time_cuts`` or ``day_types`` means "take them from
    the CONFIGURATION block" (``EXPORT_LEVELS``, ``EXPORT_TIME_CUTS``, ``DAY_TYPES``).
    """

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    periods: Sequence[TimePeriod]
    periods_source: str = "TIME_PERIODS in the CONFIGURATION block"
    levels: Optional[Sequence[str]] = None
    time_cuts: Optional[Sequence[str]] = None
    export_max_load: bool = EXPORT_MAX_LOAD
    vehicle_capacity: int = VEHICLE_CAPACITY
    day_types: Optional[Mapping[str, Sequence[str]]] = None
    holidays: Sequence[dt.date] = ()
    holiday_day_type: str = HOLIDAY_DAY_TYPE
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    time_columns: Sequence[str] = TIME_COLUMNS
    cutover_hour: int = LATE_NIGHT_CUTOVER_HOUR
    routes_to_include: Sequence[str] = ROUTES_TO_INCLUDE
    routes_to_exclude: Sequence[str] = ROUTES_TO_EXCLUDE
    low_ridership_measure: str = LOW_RIDERSHIP_MEASURE
    low_ridership_threshold: float = LOW_RIDERSHIP_THRESHOLD
    export_service_cut_tables: bool = EXPORT_SERVICE_CUT_TRIP_TABLES
    export_charts: bool = EXPORT_CHARTS
    chart_levels: Sequence[str] = CHART_LEVELS
    require_run_log: bool = REQUIRE_RUN_LOG


class RunResult(NamedTuple):
    """Tables built by one run (each also written to disk)."""

    ridership: Dict[Tuple[str, str], pd.DataFrame]
    max_load: Dict[Tuple[str, str], pd.DataFrame]
    max_load_by_trip: pd.DataFrame
    trip_rows: pd.DataFrame
    service_cut_tables: Dict[str, pd.DataFrame]
    files: List[Path]


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
# TIME PERIODS
# =============================================================================

_CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})")
_SPEC_RE = re.compile(r"(?P<name>[^=]*?)\s*=\s*(?P<start>[^\s\-–]+)\s*[-–]\s*(?P<end>[^\s\-–]+)")


def _clock(minutes: int) -> str:
    """Format minutes since midnight as ``HH:MM`` (1440 becomes ``24:00``)."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _parse_clock(text: str, *, is_end: bool) -> int:
    """Return minutes since midnight for a 24-hour ``HH:MM`` string.

    Args:
        text: The time as written.
        is_end: Whether this is a period's end time (only an end may be
            ``24:00``).

    Returns:
        Minutes since midnight, 0-1440.

    Raises:
        ValueError: With a short reason if the time is malformed or out of range.
    """
    match = _CLOCK_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"{text.strip()!r} is not a 24-hour HH:MM time")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if minutes > 59 or hours > 24 or (hours == 24 and minutes > 0):
        raise ValueError(f"{text.strip()!r} is outside 00:00-24:00")
    if hours == 24 and not is_end:
        raise ValueError("a period cannot start at 24:00 (use 00:00)")
    return hours * 60 + minutes


def minute_mask(period: TimePeriod) -> np.ndarray:
    """Return a 1440-long boolean array that is True for the minutes in *period*."""
    mask = np.zeros(MINUTES_PER_DAY, dtype=bool)
    if period.end > period.start:
        mask[period.start : period.end] = True
    else:
        mask[period.start :] = True
        mask[: period.end] = True
    return mask


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return the ``(start, end)`` minute ranges where *mask* is True.

    The day is treated as a circle, so a run that crosses midnight comes back as
    one range whose end is earlier than its start.
    """
    if mask.all():
        return [(0, MINUTES_PER_DAY)]
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    runs = list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == MINUTES_PER_DAY:
        runs = [(runs[-1][0], runs[0][1]), *runs[1:-1]]
    return runs


def find_coverage_problems(periods: Sequence[TimePeriod]) -> List[str]:
    """Describe every gap and overlap in *periods* across the 24-hour clock.

    Args:
        periods: Time periods whose individual times are already valid.

    Returns:
        One message per uncovered range and per overlapping pair of periods;
        empty when the periods cover every minute exactly once.
    """
    masks = [minute_mask(p) for p in periods]
    covered = np.zeros(MINUTES_PER_DAY, dtype=int)
    for mask in masks:
        covered += mask
    problems = [f"no period covers {_clock(s)}-{_clock(e)}" for s, e in _runs(covered == 0)]
    for (first, first_mask), (second, second_mask) in itertools.combinations(
        zip(periods, masks), 2
    ):
        for s, e in _runs(first_mask & second_mask):
            problems.append(
                f"{first.name!r} ({first.span}) and {second.name!r} ({second.span}) "
                f"overlap at {_clock(s)}-{_clock(e)}"
            )
    return problems


def _problem_report(header: str, problems: Sequence[str]) -> str:
    """Join *problems* under *header* as an indented bullet list."""
    return "\n".join([header, *(f"  - {p}" for p in problems)])


def build_time_periods(specs: Sequence[PeriodSpec], source: str) -> List[TimePeriod]:
    """Check *specs* and return them as time periods that cover the day exactly once.

    Every problem found is reported together: blank or repeated names
    (case-insensitive), malformed or out-of-range times, zero-length periods,
    and then, once every period reads cleanly, each gap and overlap.

    Args:
        specs: The time periods as written.
        source: Where they came from, for the error message.

    Returns:
        The periods in the order given.

    Raises:
        TimePeriodError: If any check fails.
    """
    if not specs:
        raise TimePeriodError(f"No time periods were given ({source}).")
    problems: List[str] = []
    periods: List[TimePeriod] = []
    seen: Dict[str, str] = {}
    for spec in specs:
        name = spec.name.strip()
        if not name:
            problems.append(f"{spec.where}: the period name is blank")
            continue
        if name.casefold() in seen:
            problems.append(
                f"{spec.where}: the name {name!r} is already used ({seen[name.casefold()]})"
            )
            continue
        seen[name.casefold()] = spec.where
        try:
            start = _parse_clock(spec.start, is_end=False)
            end = _parse_clock(spec.end, is_end=True)
        except ValueError as exc:
            problems.append(f"{spec.where}: period {name!r}: {exc}")
            continue
        if start == end:
            problems.append(
                f"{spec.where}: period {name!r} starts and ends at {spec.start.strip()}; "
                "to cover the whole day use 00:00 - 24:00"
            )
            continue
        periods.append(TimePeriod(name, start, end or MINUTES_PER_DAY))
    if not problems:
        problems = find_coverage_problems(periods)
    if problems:
        raise TimePeriodError(
            _problem_report(f"The time periods from {source} are not usable:", problems)
            + "\nTime periods must use 24-hour HH:MM times and, together, cover every "
            "minute of the day exactly once. Fix them and run again."
        )
    return periods


def specs_from_mapping(mapping: Mapping[str, Sequence[str]]) -> List[PeriodSpec]:
    """Read the inline ``TIME_PERIODS`` mapping into period specs.

    Args:
        mapping: ``{name: (start, end)}`` as in the CONFIGURATION block.

    Returns:
        One spec per entry, in mapping order.

    Raises:
        TimePeriodError: If an entry is not a ``(start, end)`` pair.
    """
    specs: List[PeriodSpec] = []
    for name, window in mapping.items():
        where = f"TIME_PERIODS[{name!r}]"
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            raise TimePeriodError(
                f'{where} must be a (start, end) pair such as ("06:00", "09:00"), not {window!r}.'
            )
        specs.append(PeriodSpec(str(name), str(window[0]), str(window[1]), where))
    return specs


def parse_period_spec(text: str, where: str) -> PeriodSpec:
    """Parse one ``NAME = HH:MM - HH:MM`` line or ``NAME=HH:MM-HH:MM`` flag.

    Args:
        text: The period as written.
        where: Where it was written, for error messages.

    Returns:
        The unchecked period spec.

    Raises:
        TimePeriodError: If *text* is not in that form.
    """
    match = _SPEC_RE.fullmatch(text.strip())
    if match is None:
        raise TimePeriodError(f"{where}: expected NAME = HH:MM - HH:MM, got {text.strip()!r}")
    return PeriodSpec(match["name"], match["start"], match["end"], where)


def load_time_periods_file(path: Path) -> List[PeriodSpec]:
    """Read a time-periods text file (one ``NAME = HH:MM - HH:MM`` per line).

    Blank lines and anything after ``#`` are ignored.

    Args:
        path: The text file.

    Returns:
        One spec per period line, in file order.

    Raises:
        TimePeriodError: If the file cannot be read, a line is not in that
            form, or the file lists no periods.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise TimePeriodError(f"Could not read the time-periods file '{path}': {exc}") from exc
    specs: List[PeriodSpec] = []
    problems: List[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            specs.append(parse_period_spec(line, f"line {number}"))
        except TimePeriodError as exc:
            problems.append(str(exc))
    if problems:
        raise TimePeriodError(
            _problem_report(f"Could not read the time-periods file '{path}':", problems)
        )
    if not specs:
        raise TimePeriodError(f"The time-periods file '{path}' lists no periods.")
    return specs


def resolve_time_periods(
    cli_periods: Optional[Sequence[str]],
    periods_file: Optional[str],
    inline: Mapping[str, Sequence[str]],
) -> Tuple[List[TimePeriod], str]:
    """Pick the time periods to use and check them.

    Precedence: ``--time-period`` flags, then the time-periods file, then the
    inline ``TIME_PERIODS`` mapping.

    Args:
        cli_periods: ``NAME=HH:MM-HH:MM`` strings from the command line, if any.
        periods_file: Path to a time-periods text file, if any.
        inline: The ``TIME_PERIODS`` mapping.

    Returns:
        The checked periods and a description of where they came from.

    Raises:
        TimePeriodError: If the chosen periods fail any check.
    """
    if cli_periods:
        if periods_file:
            logging.info("Using --time-period flags; ignoring time-periods file %s.", periods_file)
        source = "the --time-period flags"
        specs = [parse_period_spec(v, f"--time-period {v!r}") for v in cli_periods]
    elif periods_file:
        path = Path(periods_file).expanduser()
        source = f"time-periods file '{path}'"
        specs = load_time_periods_file(path)
    else:
        source = "TIME_PERIODS in the CONFIGURATION block"
        specs = specs_from_mapping(inline)
    return build_time_periods(specs, source), source


def format_period(period: TimePeriod) -> str:
    """Format *period* as a time-periods file line (``NAME = HH:MM - HH:MM``)."""
    return f"{period.name} = {_clock(period.start)} - {_clock(period.end)}"


def period_lookup(periods: Sequence[TimePeriod]) -> np.ndarray:
    """Return a 1440-long array giving the period name for each minute of the day."""
    lookup = np.full(MINUTES_PER_DAY, None, dtype=object)
    for period in periods:
        lookup[minute_mask(period)] = period.name
    return lookup


# =============================================================================
# OTHER SETTINGS
# =============================================================================


def resolve_flags(flags: Mapping[str, bool], known: Sequence[str], setting: str) -> List[str]:
    """Return the names switched on in a ``{name: bool}`` flag mapping, in *known* order.

    Args:
        flags: The flag mapping (an entry left out counts as off).
        known: Every valid name, in output order.
        setting: The setting's name, for error messages.

    Returns:
        The names whose flag is True.

    Raises:
        SettingsError: If the mapping names an unknown entry or switches everything off.
    """
    unknown = [str(k) for k in flags if k not in known]
    if unknown:
        raise SettingsError(
            f"{setting} has unknown entries: {', '.join(unknown)}. Known: {', '.join(known)}."
        )
    chosen = [name for name in known if flags.get(name, False)]
    if not chosen:
        raise SettingsError(f"{setting} switches every entry off; nothing would be exported.")
    return chosen


def check_names(values: Sequence[str], known: Sequence[str], setting: str) -> List[str]:
    """Check a list of names against *known* and return it in *known* order.

    Args:
        values: The names given.
        known: Every valid name, in output order.
        setting: The setting's name, for error messages.

    Returns:
        The given names, de-duplicated and in *known* order.

    Raises:
        SettingsError: If a name is unknown or the list is empty.
    """
    unknown = [str(v) for v in values if v not in known]
    if unknown:
        raise SettingsError(
            f"{setting} has unknown entries: {', '.join(unknown)}. Known: {', '.join(known)}."
        )
    chosen = [name for name in known if name in set(values)]
    if not chosen:
        raise SettingsError(f"{setting} is empty; nothing would be exported.")
    return chosen


def build_day_types(
    mapping: Mapping[str, Sequence[str]], holiday_day_type: str = HOLIDAY_DAY_TYPE
) -> Dict[str, Tuple[int, ...]]:
    """Check ``DAY_TYPES`` and return each day type's weekday numbers (Monday = 0).

    Args:
        mapping: ``{name: weekdays}``; weekdays are full English names or their
            three-letter abbreviations, in any case. A single name may be given
            as a plain string.
        holiday_day_type: The holiday day type's name, which no entry may reuse.

    Returns:
        ``{name: weekday numbers}`` in the order given.

    Raises:
        SettingsError: Listing every problem found (blank or repeated names, an
            unknown weekday, an entry with no weekdays, or an empty mapping).
    """
    problems: List[str] = []
    out: Dict[str, Tuple[int, ...]] = {}
    lookup: Dict[str, int] = {}
    for number, day in enumerate(WEEKDAY_NAMES):
        lookup[day.casefold()] = number
        lookup[day[:3].casefold()] = number
    for raw_name, days in mapping.items():
        name = str(raw_name).strip()
        where = f"DAY_TYPES[{raw_name!r}]"
        if not name:
            problems.append(f"{where}: the day type name is blank")
            continue
        taken = {n.casefold() for n in out} | {holiday_day_type.strip().casefold()}
        if name.casefold() in taken:
            problems.append(f"{where}: the name {name!r} is already used")
            continue
        entries = [days] if isinstance(days, str) else list(days)
        numbers: List[int] = []
        for entry in entries:
            key = str(entry).strip().casefold()
            if key not in lookup:
                problems.append(f"{where}: {entry!r} is not a weekday name")
                continue
            numbers.append(lookup[key])
        if not entries:
            problems.append(f"{where} lists no weekdays")
        out[name] = tuple(sorted(set(numbers)))
    if not mapping:
        problems.append("DAY_TYPES is empty")
    if problems:
        raise SettingsError(_problem_report("DAY_TYPES is not usable:", problems))
    return out


def parse_iso_dates(values: Iterable[str], setting: str) -> List[dt.date]:
    """Parse ``YYYY-MM-DD`` strings, reporting every unreadable one together.

    Args:
        values: The dates as written.
        setting: The setting's name, for error messages.

    Returns:
        The parsed dates, sorted and de-duplicated.

    Raises:
        SettingsError: If any value is not a ``YYYY-MM-DD`` date.
    """
    dates: set[dt.date] = set()
    problems: List[str] = []
    for value in values:
        text = str(value).strip()
        try:
            dates.add(dt.date.fromisoformat(text))
        except ValueError:
            problems.append(f"{text!r} is not a YYYY-MM-DD date")
    if problems:
        raise SettingsError(_problem_report(f"{setting} is not usable:", problems))
    return sorted(dates)


def parse_optional_date(value: Optional[str], setting: str) -> Optional[dt.date]:
    """Parse one optional ``YYYY-MM-DD`` setting (``None`` or blank stays ``None``).

    Args:
        value: The date as written, or ``None``.
        setting: The setting's name, for error messages.

    Returns:
        The date, or ``None``.

    Raises:
        SettingsError: If the value is not a ``YYYY-MM-DD`` date.
    """
    if value is None or not str(value).strip():
        return None
    return parse_iso_dates([str(value)], setting)[0]


# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_tides_csv(path: Path, table: str) -> pd.DataFrame:
    """Read a TIDES CSV with every column as text.

    Args:
        path: The CSV file.
        table: The table's name, for error messages.

    Returns:
        The table, all columns as strings.

    Raises:
        InputDataError: If the file is empty or cannot be parsed as CSV.
    """
    try:
        return pd.read_csv(path, dtype=str, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise InputDataError(f"Could not read {table} from '{path}': {exc}") from exc


def _require_columns(df: pd.DataFrame, columns: Sequence[str], table: str, why: str) -> None:
    """Raise :class:`InputDataError` naming any of *columns* missing from *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InputDataError(f"{table} has no {', '.join(missing)} column(s), which {why}.")


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse a service_date column to midnight timestamps (NaT where unreadable)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce").dt.normalize()


def _casefolded(values: pd.Series, blank: str) -> pd.Series:
    """Return *values* stripped and casefolded, with blanks replaced by *blank*."""
    text = values.fillna("").astype(str).str.strip()
    return text.where(text != "", blank).str.casefold()


def join_trips(stop_visits: pd.DataFrame, trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Attach trip attributes to each stop visit and keep revenue service only.

    The join keys are ``service_date`` and ``trip_id_performed`` (TIDES makes
    the trip ID unique only within a service date); dates are compared parsed,
    since the two exports may format them differently. Trips marked Canceled or
    with a ``trip_type`` other than In service are dropped (compared without
    regard to case). Visits with no same-day trip are counted in a warning.
    Skipped visits are kept here; :func:`drop_skipped` removes them for ridership.

    Args:
        stop_visits: The TIDES ``stop_visits`` table.
        trips_performed: The TIDES ``trips_performed`` table.

    Returns:
        Stop visits with the :data:`TRIP_ATTRIBUTE_COLUMNS` present in
        ``trips_performed`` joined on, and ``service_date`` parsed.

    Raises:
        InputDataError: If a join key is missing or no visit survives.
    """
    keys = ["service_date", "trip_id_performed"]
    _require_columns(stop_visits, keys, "stop_visits", "the join to trips_performed needs")
    _require_columns(trips_performed, keys, "trips_performed", "the join to stop_visits needs")

    visits = stop_visits.assign(service_date=_parse_dates(stop_visits["service_date"]))
    trips = trips_performed.assign(service_date=_parse_dates(trips_performed["service_date"]))
    bad_dates = visits["service_date"].isna()
    if bad_dates.any():
        logging.warning(
            "%d stop visit(s) have an unreadable service_date; they are left out.",
            int(bad_dates.sum()),
        )
        visits = visits.loc[~bad_dates]
    trips = trips.loc[trips["service_date"].notna()]
    duplicated = trips.duplicated(keys)
    if duplicated.any():
        logging.warning(
            "trips_performed repeats %d service_date + trip_id_performed pair(s); "
            "the first row of each is used.",
            int(duplicated.sum()),
        )
        trips = trips.loc[~duplicated]

    trip_index = pd.MultiIndex.from_frame(trips[keys])
    unmatched = ~pd.MultiIndex.from_frame(visits[keys]).isin(trip_index)
    if unmatched.any():
        logging.warning(
            "%d stop visit(s) have no trips_performed row with the same service_date and "
            "trip_id_performed; they are left out.",
            int(unmatched.sum()),
        )

    revenue = pd.Series(True, index=trips.index)
    if "schedule_relationship" in trips.columns:
        status = _casefolded(trips["schedule_relationship"], "scheduled")
        revenue &= ~status.isin(["canceled", "cancelled"])
    if "trip_type" in trips.columns:
        revenue &= _casefolded(trips["trip_type"], "in service") == "in service"
    kept_index = pd.MultiIndex.from_frame(trips.loc[revenue, keys])
    on_dropped = ~unmatched & ~pd.MultiIndex.from_frame(visits[keys]).isin(kept_index)
    if on_dropped.any():
        logging.info(
            "Left out %d stop visit(s) on canceled or non-revenue trips.", int(on_dropped.sum())
        )

    attr_cols = [c for c in TRIP_ATTRIBUTE_COLUMNS if c in trips.columns]
    joined = visits.drop(columns=attr_cols, errors="ignore").merge(
        trips.loc[revenue, [*keys, *attr_cols]], on=keys, how="inner"
    )
    if joined.empty:
        raise InputDataError(
            "No stop visits matched an in-service trip in trips_performed; check that the "
            "two files cover the same service dates and trip_id_performed values."
        )
    return joined.reset_index(drop=True)


def drop_skipped(df: pd.DataFrame) -> pd.DataFrame:
    """Leave out stop visits marked Skipped (the bus passed without serving the stop).

    Args:
        df: Joined stop visits.

    Returns:
        The visits that were served (Added visits are kept).
    """
    if "schedule_relationship" not in df.columns:
        return df
    skipped = _casefolded(df["schedule_relationship"], "scheduled") == "skipped"
    if skipped.any():
        logging.info("Left out %d Skipped stop visit(s) from ridership.", int(skipped.sum()))
    return df.loc[~skipped]


def add_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``boardings`` and ``alightings`` columns (both doors summed; blanks count 0).

    Args:
        df: Stop visits carrying the per-door APC count columns.

    Returns:
        A copy of *df* with float ``boardings`` and ``alightings`` columns.

    Raises:
        InputDataError: If no boarding count column is present.
    """
    boarding_cols = [c for c in BOARDING_COLUMNS if c in df.columns]
    if not boarding_cols:
        raise InputDataError(
            f"stop_visits has none of {', '.join(BOARDING_COLUMNS)}, so there are no "
            "boardings to count."
        )
    alighting_cols = [c for c in ALIGHTING_COLUMNS if c in df.columns]
    if not alighting_cols:
        logging.warning(
            "stop_visits has none of %s; alightings are reported as 0.",
            ", ".join(ALIGHTING_COLUMNS),
        )
    out = df.copy()
    for target, cols in (("boardings", boarding_cols), ("alightings", alighting_cols)):
        total = pd.Series(0.0, index=out.index)
        for col in cols:
            total += pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[target] = total
    return out


def add_trip_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``trip_id``: the scheduled (GTFS) trip, else the performed trip ID.

    Args:
        df: Joined stop visits.

    Returns:
        A copy of *df* with a text ``trip_id`` column.
    """
    out = df.copy()
    performed = out["trip_id_performed"].fillna("").astype(str).str.strip()
    if "trip_id_scheduled" in out.columns:
        scheduled = out["trip_id_scheduled"].fillna("").astype(str).str.strip()
        out["trip_id"] = scheduled.where(scheduled != "", performed)
    else:
        out["trip_id"] = performed
    return out


def filter_dates(
    df: pd.DataFrame, start: Optional[dt.date], end: Optional[dt.date]
) -> pd.DataFrame:
    """Keep service dates from *start* to *end* (inclusive; ``None`` = no limit).

    Args:
        df: Rows with a parsed ``service_date``.
        start: First date kept, or ``None``.
        end: Last date kept, or ``None``.

    Returns:
        The rows in range.

    Raises:
        InputDataError: If no rows are in range.
    """
    if start is None and end is None:
        return df
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df["service_date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= df["service_date"] <= pd.Timestamp(end)
    if (~keep).any():
        logging.info("Left out %d stop visit(s) outside the date range.", int((~keep).sum()))
    out = df.loc[keep]
    if out.empty:
        raise InputDataError(
            f"No stop visits fall between {start or 'the first date'} and "
            f"{end or 'the last date'}; check START_DATE / END_DATE."
        )
    return out


# =============================================================================
# DAY TYPES
# =============================================================================


def build_date_table(
    dates: Iterable[pd.Timestamp],
    day_types: Mapping[str, Sequence[int]],
    holidays: Iterable[dt.date] = (),
    holiday_day_type: str = HOLIDAY_DAY_TYPE,
) -> pd.DataFrame:
    """Assign each service date its day type(s).

    A holiday gets only the holiday day type. Any other date gets every day type
    listing its weekday, so a date can belong to several (or to none, in which
    case it is left out of every table).

    Args:
        dates: Service dates (midnight timestamps).
        day_types: Output of :func:`build_day_types`.
        holidays: Holiday dates.
        holiday_day_type: Name of the holiday day type.

    Returns:
        One row per date and day type: ``service_date``, ``day_type``.
    """
    holiday_set = set(holidays)
    rows: List[Tuple[pd.Timestamp, str]] = []
    no_type: List[str] = []
    seen_holidays: List[str] = []
    for stamp in sorted({pd.Timestamp(d) for d in dates}):
        day = stamp.date()
        if day in holiday_set:
            rows.append((stamp, holiday_day_type))
            seen_holidays.append(day.isoformat())
            continue
        names = [name for name, numbers in day_types.items() if day.weekday() in numbers]
        if not names:
            no_type.append(day.isoformat())
        rows.extend((stamp, name) for name in names)
    if seen_holidays:
        logging.info(
            "%d holiday date(s) in the data form the %s day type: %s",
            len(seen_holidays),
            holiday_day_type,
            ", ".join(seen_holidays),
        )
    if no_type:
        logging.info(
            "%d service date(s) fall on a weekday in no day type and are left out: %s",
            len(no_type),
            ", ".join(no_type[:10]) + (" ..." if len(no_type) > 10 else ""),
        )
    return pd.DataFrame(rows, columns=["service_date", "day_type"])


def day_type_order(
    day_types: Mapping[str, Sequence[int]], holiday_day_type: str = HOLIDAY_DAY_TYPE
) -> List[str]:
    """Return the output order of day types: ``DAY_TYPES`` order, then the holiday type."""
    return [*day_types, holiday_day_type]


def add_day_type(df: pd.DataFrame, date_table: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its date's day type(s); rows on dates in no day type drop out.

    Args:
        df: Rows with a parsed ``service_date``.
        date_table: Output of :func:`build_date_table`.

    Returns:
        *df* with a ``day_type`` column (a row appears once per day type of its date).
    """
    return df.merge(date_table, on="service_date", how="inner")


# =============================================================================
# TIME OF DAY
# =============================================================================

# Clock time inside an ISO 8601 timestamp (2026-01-05T21:03:14-05:00 -> 21:03).
_ISO_CLOCK_RE = r"^\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})"
# Date and hour of an ISO 8601 timestamp, for the service-day hour.
_ISO_DATE_HOUR_RE = r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}):\d{2}"
# Time-only value, hours possibly past 23 as in GTFS (25:10:00 -> 01:10).
_TIME_ONLY_RE = r"^(\d{1,2}):([0-5]\d)(?::[0-5]\d(?:\.\d+)?)?$"


def minute_of_day(values: pd.Series) -> pd.Series:
    """Return each value's clock time as minutes since midnight (0-1439).

    ISO 8601 timestamps are read as written, ignoring any UTC offset, so a
    daylight-saving change within the data does not shift them. Time-only
    values may run past 24:00 (GTFS style) and wrap to the next day's clock.
    Anything else is handed to :func:`pandas.to_datetime`.

    Args:
        values: Timestamps or times as text.

    Returns:
        Float minutes since midnight, NaN where the value is blank or unreadable.
    """
    text = values.astype("string").str.strip()
    text = text.mask(text == "")
    minutes = pd.Series(np.nan, index=values.index, dtype="float64")

    iso = text.str.extract(_ISO_CLOCK_RE)
    hours = pd.to_numeric(iso[0], errors="coerce")
    mins = pd.to_numeric(iso[1], errors="coerce")
    valid = (hours < 24) & (mins < 60)
    minutes = minutes.mask(valid, hours * 60 + mins)

    rest = minutes.isna() & text.notna()
    if rest.any():
        parts = text[rest].str.extract(_TIME_ONLY_RE)
        clock = pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(
            parts[1], errors="coerce"
        )
        minutes.loc[rest] = (clock % MINUTES_PER_DAY).to_numpy(dtype="float64")

    rest = minutes.isna() & text.notna()
    if rest.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stamp = pd.to_datetime(text[rest], errors="coerce")
            except (ValueError, TypeError):
                stamp = None
        if stamp is not None and pd.api.types.is_datetime64_any_dtype(stamp):
            parsed = stamp.dt.hour * 60 + stamp.dt.minute
            minutes.loc[rest] = parsed.to_numpy(dtype="float64", na_value=np.nan)
    return minutes


def _to_service_hour(clock: pd.Series, offset: pd.Series, cutover_hour: int) -> pd.Series:
    """Move clock hours onto the service-day clock.

    A stamp dated the day after its service date gains 24 hours (00:30 -> 24).
    Otherwise an hour before *cutover_hour* gains 24 too, for vendors that stamp
    small-hours events with the operating day's date.

    Args:
        clock: Clock hours 0-23.
        offset: Days between each stamp's date and its service date (NaN unknown).
        cutover_hour: First hour of the service day.

    Returns:
        Service-day hours.
    """
    next_day = offset == 1
    early = clock < cutover_hour
    return clock.where(~(next_day | early), clock + 24)


def service_day_hour(values: pd.Series, service_dates: pd.Series, cutover_hour: int) -> pd.Series:
    """Return each value's hour on the service-day clock (cutover to cutover + 23).

    ISO 8601 timestamps keep the date and hour as written and are compared with
    the service date (a stamp on the next calendar day lands at hour 24 or later).
    Time-only values past 23:59 keep their GTFS hour; earlier ones before the
    cutover move to the 24+ end. Anything else is handed to
    :func:`pandas.to_datetime`.

    Args:
        values: Timestamps or times as text.
        service_dates: Each row's service date.
        cutover_hour: First hour of the service day (``LATE_NIGHT_CUTOVER_HOUR``).

    Returns:
        Float service-day hours, NaN where the value is blank or unreadable.
    """
    text = values.astype("string").str.strip()
    text = text.mask(text == "")
    days = pd.to_datetime(service_dates, errors="coerce").dt.normalize()
    hours = pd.Series(np.nan, index=values.index, dtype="float64")

    iso = text.str.extract(_ISO_DATE_HOUR_RE)
    stamp_day = pd.to_datetime(iso[0], format="%Y-%m-%d", errors="coerce")
    clock = pd.to_numeric(iso[1], errors="coerce")
    offset = (stamp_day - days).dt.days
    valid = clock.notna() & stamp_day.notna() & (clock < 24)
    hours = hours.mask(valid, _to_service_hour(clock, offset, cutover_hour))

    rest = hours.isna() & text.notna()
    if rest.any():
        parts = text[rest].str.extract(_TIME_ONLY_RE)
        hour = pd.to_numeric(parts[0], errors="coerce")
        hour = hour.where(hour >= 24, hour.where(hour >= cutover_hour, hour + 24))
        hours.loc[rest] = hour.to_numpy(dtype="float64")

    rest = hours.isna() & text.notna()
    if rest.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stamp = pd.to_datetime(text[rest], errors="coerce")
            except (ValueError, TypeError):
                stamp = None
        if stamp is not None and pd.api.types.is_datetime64_any_dtype(stamp):
            if stamp.dt.tz is not None:
                stamp = stamp.dt.tz_localize(None)
            later = (stamp.dt.normalize() - days[rest]).dt.days
            shifted = _to_service_hour(stamp.dt.hour.astype("float64"), later, cutover_hour)
            hours.loc[rest] = shifted.to_numpy(dtype="float64", na_value=np.nan)
    return hours


def _first_present(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return the first non-blank value per row across *columns* (NA where none)."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        text = df[col].astype("string").str.strip()
        filled = text.notna() & (text != "").fillna(False)
        out = out.where(out.notna() | ~filled, text.astype(object))
    return out


def add_time_period(
    df: pd.DataFrame,
    periods: Sequence[TimePeriod],
    time_columns: Sequence[str] = TIME_COLUMNS,
) -> pd.DataFrame:
    """Add a ``time_period`` column from each visit's clock time.

    The first readable value among *time_columns* gives the time of day. Visits
    with no readable time are counted in a warning and left out. The value used
    is kept in ``_time_text`` (and its clock minute in ``_minute``) for
    :func:`add_service_hour`.

    Args:
        df: Stop visits.
        periods: Checked time periods covering the whole day.
        time_columns: Columns to try, in order.

    Returns:
        A copy of *df* (minus unreadable rows) with a ``time_period`` column.

    Raises:
        InputDataError: If none of *time_columns* is present, or no visit has a
            readable time.
    """
    present = [c for c in time_columns if c in df.columns]
    if not present:
        raise InputDataError(
            f"stop_visits has none of the TIME_COLUMNS ({', '.join(time_columns)}), so "
            "visits cannot be placed in a time period."
        )
    sample = df[present[0]].dropna().head(1000).astype(str).str.strip()
    if not sample.empty and sample.str.contains(r"(?:Z|[+-]00:?00)$").mean() > 0.5:
        logging.warning(
            "%s looks like UTC time. Time periods use the clock time as written, so "
            "convert the timestamps to local time first if they are UTC.",
            present[0],
        )

    minutes = pd.Series(np.nan, index=df.index, dtype="float64")
    chosen = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in present:
        need = minutes.isna()
        if not need.any():
            break
        found = minute_of_day(df.loc[need, col])
        hit = found.index[found.notna()]
        minutes.loc[hit] = found.loc[hit]
        chosen.loc[hit] = df.loc[hit, col]
    missing = minutes.isna()
    if missing.all():
        raise InputDataError(
            f"No stop visit has a readable time in {', '.join(present)}; check TIME_COLUMNS."
        )
    if missing.any():
        logging.warning(
            "%d stop visit(s) have no readable time in %s; they are left out.",
            int(missing.sum()),
            ", ".join(present),
        )
    out = df.loc[~missing].copy()
    out["_minute"] = minutes.loc[~missing].astype(int)
    out["_time_text"] = chosen.loc[~missing]
    lookup = period_lookup(periods)
    out["time_period"] = lookup[out["_minute"].to_numpy(dtype=int)]
    return out


def add_service_hour(df: pd.DataFrame, cutover_hour: int = LATE_NIGHT_CUTOVER_HOUR) -> pd.DataFrame:
    """Add ``hour``, each visit's hour of the service day.

    Args:
        df: Output of :func:`add_time_period` (needs ``_time_text``, ``_minute``
            and a parsed ``service_date``).
        cutover_hour: First hour of the service day.

    Returns:
        A copy of *df* with an integer ``hour`` column.
    """
    out = df.copy()
    hours = service_day_hour(out["_time_text"], out["service_date"], cutover_hour)
    clock = (out["_minute"] // 60).astype("float64")
    fallback = clock.where(clock >= cutover_hour, clock + 24)
    out["hour"] = hours.fillna(fallback).astype(int)
    return out


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``month`` column in ``YYYY-MM`` form derived from ``service_date``.

    Args:
        df: Rows with a parsed ``service_date`` column.

    Returns:
        Copy of ``df`` with a string ``month`` column.
    """
    out = df.copy()
    out["month"] = out["service_date"].dt.strftime("%Y-%m")
    return out


# =============================================================================
# FILTERING
# =============================================================================


def filter_routes(
    df: pd.DataFrame,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    warn: bool = True,
) -> pd.DataFrame:
    """Keep or drop routes by ``route_id`` (compared as stripped text).

    Filter entries that match no route in the data are named in a warning.

    Args:
        df: Rows with ``route_id``.
        include: Route IDs to keep; empty keeps all.
        exclude: Route IDs to drop.
        warn: Whether to warn about filter entries that match no route.

    Returns:
        The filtered rows.

    Raises:
        InputDataError: If no rows remain.
    """
    if not include and not exclude:
        return df
    route = df["route_id"].fillna("").astype(str).str.strip()
    known = set(route)
    wanted = {str(r).strip() for r in include}
    unwanted = {str(r).strip() for r in exclude}
    if warn:
        for label, values in (("ROUTES_TO_INCLUDE", wanted), ("ROUTES_TO_EXCLUDE", unwanted)):
            absent = sorted(values - known)
            if absent:
                logging.warning("%s entries match no route_id: %s", label, ", ".join(absent))
    keep = pd.Series(True, index=df.index)
    if wanted:
        keep &= route.isin(wanted)
    if unwanted:
        keep &= ~route.isin(unwanted)
    out = df.loc[keep]
    if out.empty:
        raise InputDataError(
            "No stop visits are left after the route filters; check ROUTES_TO_INCLUDE / "
            "ROUTES_TO_EXCLUDE."
        )
    return out


def label_blank_groups(
    df: pd.DataFrame, group_cols: Sequence[str], warn: bool = True
) -> pd.DataFrame:
    """Replace blank grouping values with ``"(blank)"`` so they form their own group.

    Args:
        df: Rows to label.
        group_cols: The grouping columns (any not in *df* are ignored).
        warn: Whether to warn about the blanks found.

    Returns:
        A copy of *df* with the grouping columns as stripped text.
    """
    out = df.copy()
    for col in group_cols:
        if col not in out.columns:
            continue
        text = out[col].fillna("").astype(str).str.strip()
        blank = text == ""
        if blank.any() and warn:
            logging.warning(
                "%d row(s) have a blank %s; they are reported as %s.",
                int(blank.sum()),
                col,
                BLANK_LABEL,
            )
        out[col] = text.where(~blank, BLANK_LABEL)
    return out


# =============================================================================
# RIDERSHIP AGGREGATION
# =============================================================================


def level_columns(level: str, columns: Iterable[str]) -> Optional[List[str]]:
    """Return *level*'s key columns found in *columns*, or ``None`` if a required one is missing.

    Args:
        level: A key of :data:`LEVELS`.
        columns: The columns available.

    Returns:
        The key columns, in order, or ``None``.
    """
    available = set(columns)
    cols: List[str] = []
    for col, required in LEVELS[level]:
        if col in available:
            cols.append(col)
        elif required:
            return None
    return cols


def missing_level_columns(level: str, columns: Iterable[str]) -> List[str]:
    """Return the required key columns of *level* that are not in *columns*."""
    available = set(columns)
    return [col for col, required in LEVELS[level] if required and col not in available]


def add_performed_trip_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``_trip``, one value per performed trip (service date + trip_id_performed)."""
    trip = df["service_date"].astype(str) + "|" + df["trip_id_performed"].astype(str)
    return df.assign(_trip=trip)


def observed_dates(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Count the distinct service dates per group.

    Args:
        df: Rows with ``service_date`` and the *keys* columns.
        keys: Grouping columns.

    Returns:
        One row per group with a ``days_observed`` column.
    """
    counts = df.groupby(list(keys), sort=False, dropna=False)["service_date"].nunique()
    return counts.rename("days_observed").reset_index()


def _sort_table(
    table: pd.DataFrame,
    sort_cols: Sequence[str],
    day_order: Sequence[str],
    periods: Sequence[TimePeriod] = (),
) -> pd.DataFrame:
    """Sort *table* by *sort_cols*, with day types and time periods in configured order."""
    work = table.copy()
    order_cols: List[str] = []
    helpers: List[str] = []
    period_rank = {p.name: i for i, p in enumerate(periods)}
    for col in sort_cols:
        if col == "day_type":
            rank = {d: i for i, d in enumerate(day_order)}
            work["_day_rank"] = work["day_type"].map(rank).fillna(len(rank))
            order_cols.append("_day_rank")
            helpers.append("_day_rank")
        elif col == "time_period" and period_rank:
            work["_period_rank"] = work["time_period"].map(period_rank).fillna(len(period_rank))
            order_cols.append("_period_rank")
            helpers.append("_period_rank")
        else:
            order_cols.append(col)
    work = work.sort_values(order_cols, kind="stable")
    return work.drop(columns=helpers).reset_index(drop=True)


def _cut_keys(cut: str) -> Tuple[List[str], List[str]]:
    """Split a time cut's columns into date-level keys and time-of-day keys.

    Args:
        cut: A key of :data:`TIME_CUTS`.

    Returns:
        ``(date keys, time keys)``: ``["day_type"]`` plus ``"month"`` when the cut
        has one, and ``[]``, ``["time_period"]`` or ``["hour"]``.
    """
    temporal = TIME_CUTS[cut]
    date_keys = ["day_type", *[t for t in temporal if t == "month"]]
    time_keys = [t for t in temporal if t != "month"]
    return date_keys, time_keys


def summarize_ridership(
    df: pd.DataFrame,
    level: str,
    cut: str,
    periods: Sequence[TimePeriod],
    system_days: pd.DataFrame,
    day_order: Sequence[str],
) -> pd.DataFrame:
    """Aggregate stop visits to one ridership table (one level crossed with one time cut).

    Averages divide each group's totals by ``days_observed``: the service dates
    of its day type (and month) with any observed visit system-wide, or, for the
    trip levels, the dates that trip was observed.

    Args:
        df: Prepared stop visits (``day_type``, ``month``, ``time_period``,
            ``hour``, ``boardings``, ``alightings`` and the level's key columns).
        level: A key of :data:`LEVELS`; its required columns must be present.
        cut: A key of :data:`TIME_CUTS`.
        periods: The checked time periods (their order sets the row order).
        system_days: ``days_observed`` per day type (and month, for month cuts).
        day_order: Output order of the day types.

    Returns:
        The table: date keys, level keys, time keys, then the measures.
    """
    level_cols = level_columns(level, df.columns) or []
    date_keys, time_keys = _cut_keys(cut)
    keys = [*date_keys, *level_cols, *time_keys]
    work = df if "_trip" in df.columns else add_performed_trip_key(df)
    agg = (
        work.groupby(keys, sort=False, dropna=False)
        .agg(
            total_boardings=("boardings", "sum"),
            total_alightings=("alightings", "sum"),
            stop_visits=("boardings", "size"),
            trips_observed=("_trip", "nunique"),
        )
        .reset_index()
    )
    if time_keys == ["time_period"] and level in ZERO_FILL_LEVELS:
        names = pd.DataFrame({"time_period": [p.name for p in periods]})
        grid = work[[*date_keys, *level_cols]].drop_duplicates().merge(names, how="cross")
        agg = grid.merge(agg, on=keys, how="left")
        for col in ("total_boardings", "total_alightings", "stop_visits", "trips_observed"):
            agg[col] = agg[col].fillna(0)

    if level in TRIP_LEVELS:
        trip_keys = [*date_keys, *[c for c in level_cols if c != "stop_id"]]
        agg = agg.merge(observed_dates(work, trip_keys), on=trip_keys, how="left")
    else:
        agg = agg.merge(system_days, on=date_keys, how="left")
    agg["stop_visits"] = agg["stop_visits"].astype(int)
    agg["trips_observed"] = agg["trips_observed"].astype(int)
    agg["days_observed"] = agg["days_observed"].fillna(0).astype(int)

    days = agg["days_observed"].where(agg["days_observed"] > 0)
    daily_boardings = agg["total_boardings"] / days
    agg["avg_daily_boardings"] = daily_boardings.round(2)
    agg["avg_daily_alightings"] = (agg["total_alightings"] / days).round(2)
    trips = agg["trips_observed"].where(agg["trips_observed"] > 0)
    agg["avg_boardings_per_trip"] = (agg["total_boardings"] / trips).round(2)

    time_cols = list(time_keys)
    measures = ["days_observed", "avg_daily_boardings", "avg_daily_alightings"]
    if time_keys == ["time_period"]:
        by_name = {p.name: p for p in periods}
        agg["period_start"] = agg["time_period"].map(lambda n: _clock(by_name[n].start))
        agg["period_end"] = agg["time_period"].map(lambda n: _clock(by_name[n].end))
        hours = agg["time_period"].map(lambda n: by_name[n].minutes / 60)
        agg["period_hours"] = hours.round(2)
        agg["avg_daily_boardings_per_period_hour"] = (daily_boardings / hours).round(2)
        time_cols = ["time_period", "period_start", "period_end", "period_hours"]
        measures.append("avg_daily_boardings_per_period_hour")
    elif time_keys == ["hour"]:
        agg["hour"] = agg["hour"].astype(int)
        agg["hour_label"] = agg["hour"].map(lambda h: minutes_to_hhmm(h * 60))
        time_cols = ["hour", "hour_label"]
    measures += [
        "total_boardings",
        "total_alightings",
        "stop_visits",
        "trips_observed",
        "avg_boardings_per_trip",
    ]
    agg["total_boardings"] = agg["total_boardings"].round(2)
    agg["total_alightings"] = agg["total_alightings"].round(2)
    table = _sort_table(agg, keys, day_order, periods)
    return table[[*date_keys, *level_cols, *time_cols, *measures]]


def build_service_cut_tables(visits: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build the per-day-type trip tables ``service_cut_impact_gpd.py`` reads.

    Each trip's boardings at each stop are averaged over the dates that trip
    was observed in the day type.

    Args:
        visits: Prepared stop visits (``day_type``, ``trip_id``, ``boardings``,
            and ``stop_id`` when present).

    Returns:
        ``{day_type: frame}`` with columns ``trip_id``, ``stop_id``,
        ``avg_daily_boardings``.
    """
    work = visits.assign(_stop=visits["stop_id"] if "stop_id" in visits.columns else "")
    tables: Dict[str, pd.DataFrame] = {}
    for day_type, sub in work.groupby("day_type", sort=False):
        trip_dates = sub.groupby("trip_id")["service_date"].nunique()
        table = sub.groupby(["trip_id", "_stop"])["boardings"].sum().reset_index()
        table["avg_daily_boardings"] = (
            table["boardings"] / table["trip_id"].map(trip_dates)
        ).round(2)
        table = table.rename(columns={"_stop": "stop_id"})
        tables[str(day_type)] = table[["trip_id", "stop_id", "avg_daily_boardings"]]
    return tables


# =============================================================================
# MAXIMUM LOAD
# =============================================================================


def compute_trip_max_load(stop_visits: pd.DataFrame) -> pd.DataFrame:
    """Reduce stop visits to one maximum-load record per performed trip.

    A performed trip is a (``service_date``, ``trip_id_performed``) pair, since
    the ID alone may repeat every day. For each one the row carrying the largest
    ``departure_load`` is kept, yielding the trip's maximum load and the stop
    where it occurs (the peak load point); ties go to the earliest stop. Trips
    with no usable load reading are dropped.

    Args:
        stop_visits: Stop visits with ``service_date``, ``trip_id_performed`` and
            ``departure_load`` (text or numeric), plus ``stop_id`` and
            ``trip_stop_sequence`` when available.

    Returns:
        DataFrame with one row per trip and columns ``trip_id_performed``,
        ``service_date``, ``max_load``, ``peak_stop_id``, and
        ``peak_stop_sequence``.
    """
    columns = [
        "trip_id_performed",
        "service_date",
        "max_load",
        "peak_stop_id",
        "peak_stop_sequence",
    ]
    if "departure_load" not in stop_visits.columns:
        return pd.DataFrame(columns=columns)
    loads = pd.to_numeric(stop_visits["departure_load"], errors="coerce")
    usable = stop_visits.assign(departure_load=loads).loc[loads.notna()].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)

    # Sort so idxmax breaks ties on the earliest stop in the sequence.
    trip_keys = ["service_date", "trip_id_performed"]
    sort_cols = list(trip_keys)
    if "trip_stop_sequence" in usable.columns:
        usable["trip_stop_sequence"] = pd.to_numeric(usable["trip_stop_sequence"], errors="coerce")
        sort_cols.append("trip_stop_sequence")
    usable = usable.sort_values(sort_cols, kind="stable")

    peak_idx = usable.groupby(trip_keys, dropna=False)["departure_load"].idxmax()
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


def build_trip_max_load(
    joined: pd.DataFrame,
    periods: Sequence[TimePeriod],
    vehicle_capacity: int,
    cutover_hour: int = LATE_NIGHT_CUTOVER_HOUR,
) -> pd.DataFrame:
    """Build the per-performed-trip maximum-load table with its trip attributes.

    Each trip is placed in a time period and an hour by its start time
    (``schedule_trip_start``, else ``actual_trip_start``); trips with neither
    are labeled ``UNCLASSIFIED`` and have no hour.

    Args:
        joined: Joined stop visits (output of :func:`join_trips` and
            :func:`add_trip_key`), Skipped visits included.
        periods: The checked time periods.
        vehicle_capacity: Capacity used for the load factor.
        cutover_hour: First hour of the service day.

    Returns:
        One row per performed trip with ``max_load``, the peak load point, trip
        attributes, ``time_period``, ``hour``, ``month`` and ``load_factor``.
    """
    per_trip = compute_trip_max_load(joined)
    if per_trip.empty:
        return per_trip
    keys = ["service_date", "trip_id_performed"]
    attrs = [
        c
        for c in (
            "route_id",
            "direction_id",
            "route_type_agency",
            "trip_id",
            "schedule_trip_start",
            "actual_trip_start",
        )
        if c in joined.columns
    ]
    first = joined.drop_duplicates(keys)[[*keys, *attrs]]
    out = per_trip.assign(service_date=pd.to_datetime(per_trip["service_date"])).merge(
        first, on=keys, how="left"
    )
    start_text = _first_present(out, ("schedule_trip_start", "actual_trip_start"))
    minutes = minute_of_day(start_text)
    lookup = period_lookup(periods)
    labels = pd.Series(UNCLASSIFIED, index=out.index, dtype="object")
    known = minutes.notna()
    labels.loc[known] = lookup[minutes.loc[known].to_numpy(dtype=int)]
    out["time_period"] = labels
    out["hour"] = service_day_hour(start_text, out["service_date"], cutover_hour)
    out["month"] = out["service_date"].dt.strftime("%Y-%m")
    out["max_load"] = pd.to_numeric(out["max_load"], errors="coerce")
    if vehicle_capacity > 0:
        out["load_factor"] = (out["max_load"] / vehicle_capacity).round(4)
    else:
        out["load_factor"] = np.nan
    return out


def _p85(series: pd.Series) -> float:
    """Return the 85th percentile of *series* (NaN for an empty series)."""
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 85))


def summarize_max_load(
    per_trip: pd.DataFrame,
    level: str,
    cut: str,
    vehicle_capacity: int,
    day_order: Sequence[str],
    periods: Sequence[TimePeriod],
) -> pd.DataFrame:
    """Aggregate per-trip maxima to summary statistics (one level by one time cut).

    Max-load statistics do not compose across grains, so every table is computed
    from the per-trip rows. Trips with no start time are left out of the hour cut.

    Args:
        per_trip: Output of :func:`build_trip_max_load` with ``day_type`` added.
        level: A key of :data:`MAX_LOAD_LEVELS`; its required columns must be present.
        cut: A key of :data:`TIME_CUTS`.
        vehicle_capacity: Capacity used to flag over-capacity trips.
        day_order: Output order of the day types.
        periods: The checked time periods (their order sets the row order).

    Returns:
        The table: keys, then ``trips``, ``mean_max_load``, ``median_max_load``,
        ``p85_max_load``, ``peak_max_load``, ``mean_load_factor`` and
        ``pct_trips_over_capacity``.
    """
    level_cols = level_columns(level, per_trip.columns) or []
    date_keys, time_keys = _cut_keys(cut)
    keys = [*date_keys, *level_cols, *time_keys]
    work = per_trip
    if time_keys == ["hour"]:
        work = work.loc[work["hour"].notna()].assign(hour=lambda d: d["hour"].astype(int))
    work = work.assign(_over=(work["max_load"] > vehicle_capacity).astype(float))
    agg = (
        work.groupby(keys, sort=False, dropna=False)
        .agg(
            trips=("max_load", "size"),
            mean_max_load=("max_load", "mean"),
            median_max_load=("max_load", "median"),
            p85_max_load=("max_load", _p85),
            peak_max_load=("max_load", "max"),
            mean_load_factor=("load_factor", "mean"),
            pct_trips_over_capacity=("_over", "mean"),
        )
        .reset_index()
    )
    agg["mean_max_load"] = agg["mean_max_load"].round(2)
    agg["median_max_load"] = agg["median_max_load"].round(2)
    agg["p85_max_load"] = agg["p85_max_load"].round(2)
    agg["mean_load_factor"] = agg["mean_load_factor"].round(4)
    agg["pct_trips_over_capacity"] = (agg["pct_trips_over_capacity"] * 100.0).round(1)
    time_cols = list(time_keys)
    if time_keys == ["hour"]:
        agg["hour_label"] = agg["hour"].map(lambda h: minutes_to_hhmm(h * 60))
        time_cols = ["hour", "hour_label"]
    table = _sort_table(agg, keys, day_order, periods)
    measures = [
        "trips",
        "mean_max_load",
        "median_max_load",
        "p85_max_load",
        "peak_max_load",
        "mean_load_factor",
        "pct_trips_over_capacity",
    ]
    return table[[*date_keys, *level_cols, *time_cols, *measures]]


def build_max_load_by_trip_export(
    per_trip: pd.DataFrame, day_order: Sequence[str], periods: Sequence[TimePeriod]
) -> pd.DataFrame:
    """Order the per-trip maximum-load rows for ``max_load_by_performed_trip.csv``.

    Args:
        per_trip: Per-trip rows with ``day_type`` added.
        day_order: Output order of the day types.
        periods: The checked time periods.

    Returns:
        A column-ordered, sorted copy.
    """
    cols = [
        "service_date",
        "day_type",
        "month",
        "time_period",
        "route_id",
        "direction_id",
        "route_type_agency",
        "trip_id",
        "trip_id_performed",
        "schedule_trip_start",
        "peak_stop_id",
        "peak_stop_sequence",
        "max_load",
        "load_factor",
    ]
    out = per_trip[[c for c in cols if c in per_trip.columns]].copy()
    out["service_date"] = pd.to_datetime(out["service_date"]).dt.strftime("%Y-%m-%d")
    sort_cols = [
        c
        for c in ("day_type", "route_id", "direction_id", "schedule_trip_start", "service_date")
        if c in out.columns
    ]
    return _sort_table(out, sort_cols, day_order, periods)


# =============================================================================
# LOW-RIDERSHIP TRIPS
# =============================================================================


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = pd.to_datetime(performed[start_col], errors="coerce", utc=True)
            end = pd.to_datetime(performed[end_col], errors="coerce", utc=True)
        hours = (end - start).dt.total_seconds() / 3600.0
        duration = duration.fillna(hours.where(hours > 0))
    return duration


def build_trip_rows(
    visits: pd.DataFrame, cutover_hour: int = LATE_NIGHT_CUTOVER_HOUR
) -> pd.DataFrame:
    """Summarize each scheduled trip per day type for the low-ridership flag.

    Args:
        visits: Prepared stop visits (``day_type``, ``trip_id``, ``boardings``,
            ``hour``, ``_minute`` and the trips_performed time columns).
        cutover_hour: First hour of the service day.

    Returns:
        One row per day type and trip: ``route_id``, ``direction_id``,
        ``trip_id``, ``start_time`` (service-day clock), ``hour``,
        ``avg_daily_boardings`` (averaged over the dates the trip was observed),
        ``pass_per_hour`` (average boardings per hour of running time),
        ``days_observed`` and ``start_minutes``.
    """
    columns = [
        "day_type",
        "route_id",
        "direction_id",
        "trip_id",
        "start_time",
        "hour",
        "avg_daily_boardings",
        "pass_per_hour",
        "days_observed",
        "start_minutes",
    ]
    if visits.empty:
        return pd.DataFrame(columns=columns)
    work = visits.copy()
    for col in ("route_id", "direction_id"):
        if col not in work.columns:
            work[col] = ""
    work["_duration"] = _trip_duration_hours(work)
    start_text = _first_present(work, ("schedule_trip_start", "actual_trip_start"))
    start_hour = service_day_hour(start_text, work["service_date"], cutover_hour)
    start_minute = minute_of_day(start_text)
    start = start_hour * 60 + start_minute % 60
    work["_start"] = start.fillna(work["hour"] * 60 + work["_minute"] % 60)

    runs = work.groupby(["day_type", "service_date", "trip_id_performed"], sort=False).agg(
        trip_id=("trip_id", "first"),
        route_id=("route_id", "first"),
        direction_id=("direction_id", "first"),
        boardings=("boardings", "sum"),
        start=("_start", "min"),
        duration=("_duration", "first"),
    )
    runs = runs.reset_index()
    rows = runs.groupby(["day_type", "trip_id"], sort=False, as_index=False).agg(
        route_id=("route_id", "first"),
        direction_id=("direction_id", "first"),
        start_minutes=("start", "min"),
        boardings=("boardings", "sum"),
        days_observed=("service_date", "nunique"),
        duration=("duration", "mean"),
    )
    rows["avg_daily_boardings"] = rows["boardings"] / rows["days_observed"]
    rows["pass_per_hour"] = rows["avg_daily_boardings"] / rows["duration"].where(
        rows["duration"] > 0
    )
    rows["hour"] = (rows["start_minutes"] // 60).astype(int)
    rows["start_time"] = rows["start_minutes"].map(minutes_to_hhmm)
    return rows[columns]


def flag_low_ridership(
    trip_rows: pd.DataFrame,
    measure: str,
    threshold: float,
    out_path: Path,
    day_order: Sequence[str],
) -> int:
    """Log and export trips whose *measure* falls below *threshold*.

    Flagged trips are written to *out_path* sorted by day type, route and start
    time, and a warning is logged per day type so low performers surface in the
    console and run log even when nobody opens the CSV.

    Args:
        trip_rows: Output of :func:`build_trip_rows`.
        measure: A key of :data:`LOW_RIDERSHIP_MEASURES`. Trips with no value
            (no running time in trips_performed) are skipped with a warning.
        threshold: Trips with *measure* strictly below this are flagged; values
            <= 0 switch the flag off.
        out_path: Destination CSV (only written when trips are flagged).
        day_order: Output order of the day types.

    Returns:
        Number of trips flagged.
    """
    if threshold <= 0 or trip_rows.empty:
        return 0
    column = LOW_RIDERSHIP_MEASURES[measure]
    values = trip_rows[column]
    missing = int(values.isna().sum())
    if missing:
        logging.warning(
            "%d trip(s) have no %s value (no running time in trips_performed) and were "
            "not checked for the low-ridership flag.",
            missing,
            measure,
        )
    flagged = trip_rows.loc[values < threshold].copy()
    if flagged.empty:
        logging.info(
            "No trips below %.1f %s (%d trips checked).",
            threshold,
            measure,
            len(trip_rows) - missing,
        )
        return 0
    flagged = _sort_table(
        flagged, ["day_type", "route_id", "direction_id", "start_minutes"], day_order
    )
    flagged["avg_daily_boardings"] = flagged["avg_daily_boardings"].round(2)
    flagged["pass_per_hour"] = flagged["pass_per_hour"].round(2)
    flagged = flagged.drop(columns="start_minutes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(out_path, index=False)
    for day_type in [d for d in day_order if d in set(flagged["day_type"])]:
        sub = flagged.loc[flagged["day_type"] == day_type]
        logging.warning(
            "%s: %d of %d trip(s) below %.1f %s, on %d route(s).",
            day_type,
            len(sub),
            int((trip_rows["day_type"] == day_type).sum()),
            threshold,
            measure,
            sub["route_id"].nunique(),
        )
    logging.info("Wrote: %s (%d rows)", out_path, len(flagged))
    return len(flagged)


# =============================================================================
# CHARTS
# =============================================================================


def _sanitize_token(name: object) -> str:
    """Reduce a label to a filesystem-safe token."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip()).strip("_")
    return token or "unnamed"


def _style_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply the shared chart styling (light grid, no top/right spines)."""
    ax.tick_params(axis="y", labelsize=8, colors=_INK_COLOR)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK_COLOR)
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID_COLOR)


def draw_bar_chart(
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Draw one bar chart PNG with one or two series side by side.

    Args:
        labels: Category labels along the x axis.
        series: ``(name, values)`` pairs, one value per label; a legend is drawn
            when there is more than one.
        title: Chart title.
        ylabel: Y-axis label.
        out_path: PNG destination.
    """
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    x = np.arange(len(labels))
    width = 0.72 / max(len(series), 1)
    for i, (name, values) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.bar(x + offset, list(values), width, color=color, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(labels), rotation=45, ha="right", fontsize=7, color=_INK_COLOR)
    _style_axes(ax, title, ylabel)
    if len(series) > 1:
        # Above the plot, beside the title, so the legend never covers a bar.
        ax.legend(
            fontsize=7,
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=len(series),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _group_label(level: str, values: Sequence[object]) -> Tuple[str, str]:
    """Return a chart title fragment and filename token for one group of *level*."""
    parts = [str(v) for v in values]
    if level == "system":
        return "All routes", "system"
    if level == "service_type":
        return f"{parts[0]} service", parts[0]
    if level == "route":
        return f"Route {parts[0]}", parts[0]
    return f"Route {parts[0]} direction {parts[1]}", f"{parts[0]}_dir{parts[1]}"


def export_summary_charts(
    tables: Mapping[Tuple[str, str], pd.DataFrame],
    chart_levels: Sequence[str],
    series_columns: Sequence[Tuple[str, str]],
    ylabel: str,
    out_dir: Path,
    periods: Sequence[TimePeriod],
    fill_missing: float = 0.0,
) -> int:
    """Draw one chart per group and day type for each chartable table.

    Charts cover the levels in *chart_levels* crossed with the time period, hour
    and month cuts (whichever were exported). Hours run continuously across a
    table's range so charts of one table line up.

    Args:
        tables: ``{(level, cut): table}``.
        chart_levels: Levels to chart.
        series_columns: ``(legend name, column)`` pairs to draw.
        ylabel: Y-axis label.
        out_dir: Folder for this family of charts.
        periods: The checked time periods (x order for time-period charts).
        fill_missing: Value drawn for a period, hour or month with no row.

    Returns:
        Number of PNGs written.
    """
    written = 0
    for (level, cut), table in tables.items():
        if level not in chart_levels or cut not in CHART_CUTS or table.empty:
            continue
        x_field = TIME_CUTS[cut][0]
        if x_field == "time_period":
            categories: List[object] = [p.name for p in periods]
            labels = [str(c) for c in categories]
        elif x_field == "hour":
            categories = list(range(int(table["hour"].min()), int(table["hour"].max()) + 1))
            labels = [minutes_to_hhmm(int(h) * 60) for h in categories]
        else:
            categories = sorted(table[x_field].dropna().unique().tolist())
            labels = [str(c) for c in categories]
        level_cols = level_columns(level, table.columns) or []
        folder = out_dir / f"{level}_by_{cut}"
        folder.mkdir(parents=True, exist_ok=True)
        for keys, sub in table.groupby(["day_type", *level_cols], sort=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            day_type, group_values = key_tuple[0], key_tuple[1:]
            label, token = _group_label(level, group_values)
            series = [
                (
                    name,
                    sub.groupby(x_field)[column]
                    .sum()
                    .reindex(categories)
                    .fillna(fill_missing)
                    .to_numpy(dtype=float)
                    .tolist(),
                )
                for name, column in series_columns
            ]
            out_path = folder / f"{_sanitize_token(token)}_{_sanitize_token(day_type)}.png"
            title = f"{label} — {day_type}, by {x_field.replace('_', ' ')}"
            draw_bar_chart(labels, series, title, ylabel, out_path)
            written += 1
    return written


def draw_trip_chart(
    sub: pd.DataFrame,
    title: str,
    column: str,
    ylabel: str,
    threshold: float,
    out_path: Path,
    y_max: Optional[float],
) -> None:
    """Draw one per-trip bar chart PNG for a single route, direction and day type.

    One bar per trip in start-time order. When *threshold* is positive it is
    drawn as a dashed line and bars below it are highlighted, so the flag is
    carried by both position and color; a legend names both.

    Args:
        sub: Per-trip rows, already sorted by start time, with a ``_label`` column.
        title: Chart title.
        column: Per-trip column to plot.
        ylabel: Y-axis label.
        threshold: Low-ridership threshold; <= 0 draws no line or highlight.
        out_path: PNG destination.
        y_max: Shared y-axis limit (None lets matplotlib autoscale).
    """
    values = sub[column].astype(float).tolist()
    colors = [_FLAG_COLOR if threshold > 0 and v < threshold else _SERIES_COLORS[0] for v in values]

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(range(len(values)), values, color=colors, width=0.72)
    # Floor the axis width so a route with few trips keeps bars proportioned.
    ax.set_xlim(-0.5, max(len(values), 12) - 0.5)
    step = max(1, len(values) // 24)  # keep x labels legible on busy routes
    ticks = list(range(0, len(values), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [sub["_label"].iloc[i] for i in ticks],
        rotation=45,
        ha="right",
        fontsize=7,
        color=_INK_COLOR,
    )
    _style_axes(ax, title, ylabel)
    if y_max is not None and y_max > 0:
        ax.set_ylim(0, y_max)
    if threshold > 0:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.2,
            color=_FLAG_COLOR,
            label=f"threshold ({threshold:g})",
        )
        if any(c == _FLAG_COLOR for c in colors):
            ax.bar(0, 0, color=_FLAG_COLOR, label="below threshold")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_trip_charts(
    trip_rows: pd.DataFrame, measure: str, threshold: float, out_dir: Path
) -> int:
    """Write one per-trip chart per route, direction and day type.

    The y-axis is shared across all routes within a day type (and stretched to
    keep the threshold line visible) so charts compare directly. Trips with no
    value for *measure* cannot be placed on the scale and are left off.

    Args:
        trip_rows: Output of :func:`build_trip_rows`.
        measure: A key of :data:`LOW_RIDERSHIP_MEASURES`.
        threshold: Low-ridership threshold; <= 0 draws plain charts.
        out_dir: Charts folder (created if needed).

    Returns:
        Number of PNGs written.
    """
    column = LOW_RIDERSHIP_MEASURES[measure]
    rows = trip_rows.loc[trip_rows[column].notna()].copy()
    if rows.empty:
        logging.warning("No trips have a %s value; skipping per-trip charts.", measure)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = rows["start_time"].astype(str).str.strip()
    rows["_label"] = labels.where(labels != "", rows["trip_id"].astype(str))
    y_by_day = rows.groupby("day_type")[column].max() * 1.05
    if threshold > 0:
        y_by_day = y_by_day.clip(lower=threshold * 1.2)
    ylabel = {
        "pass_per_hour": "Avg boardings / running hour",
        "boardings": "Avg daily boardings",
    }[measure]
    written = 0
    for (day_type, route, direction), sub in rows.groupby(
        ["day_type", "route_id", "direction_id"], sort=False
    ):
        sub = sub.sort_values("start_minutes")
        label = f"Route {route}"
        stem = f"route_{_sanitize_token(route)}"
        if str(direction) not in ("", BLANK_LABEL):
            label = f"{label} direction {direction}"
            stem = f"{stem}_dir{_sanitize_token(direction)}"
        out_path = out_dir / f"{stem}_{_sanitize_token(day_type)}.png"
        draw_trip_chart(
            sub,
            f"{label} — {day_type} {measure} by trip",
            column,
            ylabel,
            threshold,
            out_path,
            float(y_by_day[day_type]),
        )
        written += 1
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


def write_run_log(
    log_path: Path,
    settings_lines: Sequence[str],
    periods: Sequence[TimePeriod],
    periods_source: str,
    summary_lines: Sequence[str],
) -> bool:
    """Write the run-log sidecar into the output folder.

    Args:
        log_path: Where to write the log.
        settings_lines: The settings actually used (after CLI overrides).
        periods: The time periods used.
        periods_source: Where the time periods came from.
        summary_lines: Build summary lines.

    Returns:
        True if the log was written, False otherwise.
    """
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
        RUN_LOG_TITLE,
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {log_path.parent}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "SETTINGS USED",
        "-" * 72,
        *settings_lines,
        "",
        "-" * 72,
        f"TIME PERIODS USED (from {periods_source})",
        "-" * 72,
        "# Same format as a time-periods file; copy these lines into one to reuse them.",
        *(format_period(p) for p in periods),
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
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to '%s'.", log_path)
    return True


# =============================================================================
# PIPELINE
# =============================================================================


def grain_filename(prefix: str, level: str, cut: str) -> str:
    """Return the CSV name for one grain, e.g. ``ridership_by_route_and_time_period.csv``."""
    suffix = "".join(f"_and_{col}" for col in TIME_CUTS[cut])
    return f"{prefix}_by_{level}{suffix}.csv"


def _write_csv(table: pd.DataFrame, path: Path, files: List[Path]) -> None:
    """Write *table* to *path* (creating its folder) and record the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    files.append(path)


def _resolve_run_settings(
    settings: RunSettings,
) -> Tuple[List[str], List[str], Dict[str, Tuple[int, ...]], List[str]]:
    """Check the settings that ``run`` cannot take on trust.

    Args:
        settings: The run settings.

    Returns:
        ``(levels, time cuts, day types, chart levels)``.

    Raises:
        TimePeriodError: If the periods do not cover the day exactly once.
        SettingsError: If any other setting is unusable.
    """
    problems = find_coverage_problems(settings.periods)
    if problems:
        raise TimePeriodError(
            _problem_report("The time periods do not cover the day exactly once:", problems)
        )
    if settings.levels is None:
        levels = resolve_flags(EXPORT_LEVELS, list(LEVELS), "EXPORT_LEVELS")
    else:
        levels = check_names(settings.levels, list(LEVELS), "levels")
    if settings.time_cuts is None:
        cuts = resolve_flags(EXPORT_TIME_CUTS, list(TIME_CUTS), "EXPORT_TIME_CUTS")
    else:
        cuts = check_names(settings.time_cuts, list(TIME_CUTS), "time cuts")
    day_types = build_day_types(
        DAY_TYPES if settings.day_types is None else settings.day_types,
        settings.holiday_day_type,
    )
    unknown_charts = [lv for lv in settings.chart_levels if lv not in CHARTABLE_LEVELS]
    if unknown_charts:
        raise SettingsError(
            f"CHART_LEVELS has entries that cannot be charted: {', '.join(unknown_charts)}. "
            f"Use any of {', '.join(CHARTABLE_LEVELS)}."
        )
    if settings.low_ridership_measure not in LOW_RIDERSHIP_MEASURES:
        raise SettingsError(
            f"LOW_RIDERSHIP_MEASURE must be one of {', '.join(LOW_RIDERSHIP_MEASURES)}, "
            f"not {settings.low_ridership_measure!r}."
        )
    if settings.start_date and settings.end_date and settings.start_date > settings.end_date:
        raise SettingsError(
            f"START_DATE {settings.start_date} is after END_DATE {settings.end_date}."
        )
    return levels, cuts, day_types, list(settings.chart_levels)


def run(settings: RunSettings) -> RunResult:
    """Build every enabled table, write them with the charts, and write the run log.

    Args:
        settings: Resolved settings.

    Returns:
        The tables built (also written under ``settings.output_dir``).

    Raises:
        TimePeriodError: If ``settings.periods`` do not cover the day exactly once.
        SettingsError: If another setting is unusable.
        InputDataError: If the TIDES tables cannot support the exports.
        OSError: If an input cannot be read or an output cannot be written.
        RuntimeError: If the run log cannot be written and it is required.
    """
    levels, cuts, day_types, chart_levels = _resolve_run_settings(settings)
    holiday = settings.holiday_day_type.strip()
    order = day_type_order(day_types, holiday)
    periods = list(settings.periods)

    stop_visits = load_tides_csv(settings.stop_visits_path, "stop_visits")
    trips = load_tides_csv(settings.trips_performed_path, "trips_performed")
    if settings.routes_to_include or settings.routes_to_exclude:
        _require_columns(trips, ["route_id"], "trips_performed", "the route filters need")

    joined = join_trips(stop_visits, trips)
    joined = filter_dates(joined, settings.start_date, settings.end_date)
    joined = add_trip_key(joined)
    date_table = build_date_table(
        joined["service_date"].unique(), day_types, settings.holidays, holiday
    )
    if date_table.empty:
        raise InputDataError("No service date falls in any of the DAY_TYPES; check DAY_TYPES.")

    # --- ridership -------------------------------------------------------------
    visits = (
        drop_skipped(joined)
        .pipe(add_ridership)
        .pipe(add_time_period, periods, settings.time_columns)
        .pipe(add_service_hour, settings.cutover_hour)
        .pipe(add_month)
        .pipe(add_day_type, date_table)
    )
    if visits.empty:
        raise InputDataError("No stop visits are left on a service date in any day type.")
    system_days = {
        False: observed_dates(visits, ["day_type"]),
        True: observed_dates(visits, ["day_type", "month"]),
    }
    day_counts = system_days[False].set_index("day_type")["days_observed"].to_dict()
    for day_type in [d for d in order if d in day_counts]:
        logging.info("%s: averaged over %d service date(s).", day_type, day_counts[day_type])
    visits = filter_routes(visits, settings.routes_to_include, settings.routes_to_exclude)
    visits = label_blank_groups(visits, GROUP_KEY_COLUMNS).pipe(add_performed_trip_key)

    files: List[Path] = []
    out_dir = settings.output_dir
    ridership: Dict[Tuple[str, str], pd.DataFrame] = {}
    for level in levels:
        missing = missing_level_columns(level, visits.columns)
        if missing:
            logging.warning(
                "Skipping the %s level: the data has no %s column.", level, ", ".join(missing)
            )
            continue
        for cut in cuts:
            month_days = system_days["month" in TIME_CUTS[cut]]
            table = summarize_ridership(visits, level, cut, periods, month_days, order)
            ridership[(level, cut)] = table
            _write_csv(
                table, out_dir / RIDERSHIP_SUBDIR / grain_filename("ridership", level, cut), files
            )
    logging.info("Wrote %d ridership table(s) to %s.", len(ridership), out_dir / RIDERSHIP_SUBDIR)

    service_cut_tables: Dict[str, pd.DataFrame] = {}
    if settings.export_service_cut_tables:
        service_cut_tables = build_service_cut_tables(visits)
        for day_type, table in service_cut_tables.items():
            name = f"ridership_by_trip_{_sanitize_token(day_type).lower()}.csv"
            _write_csv(table, out_dir / SERVICE_CUT_SUBDIR / name, files)

    trip_rows = build_trip_rows(visits, settings.cutover_hour)
    low_count = flag_low_ridership(
        trip_rows,
        settings.low_ridership_measure,
        settings.low_ridership_threshold,
        out_dir / LOW_RIDERSHIP_FILENAME,
        order,
    )
    if low_count:
        files.append(out_dir / LOW_RIDERSHIP_FILENAME)

    # --- maximum load ----------------------------------------------------------
    max_load: Dict[Tuple[str, str], pd.DataFrame] = {}
    by_trip = pd.DataFrame()
    if settings.export_max_load:
        per_trip = build_trip_max_load(
            joined, periods, settings.vehicle_capacity, settings.cutover_hour
        )
        if per_trip.empty:
            logging.warning("stop_visits has no departure_load values; maximum load is skipped.")
        else:
            per_trip = add_day_type(per_trip, date_table)
            try:
                per_trip = filter_routes(
                    per_trip, settings.routes_to_include, settings.routes_to_exclude, warn=False
                )
            except InputDataError:
                per_trip = per_trip.iloc[0:0]
            per_trip = label_blank_groups(per_trip, GROUP_KEY_COLUMNS, warn=False)
        if not per_trip.empty:
            by_trip = build_max_load_by_trip_export(per_trip, order, periods)
            _write_csv(by_trip, out_dir / MAX_LOAD_SUBDIR / MAX_LOAD_BY_TRIP_FILENAME, files)
            for level in [lv for lv in levels if lv in MAX_LOAD_LEVELS]:
                if missing_level_columns(level, per_trip.columns):
                    continue
                for cut in cuts:
                    table = summarize_max_load(
                        per_trip, level, cut, settings.vehicle_capacity, order, periods
                    )
                    max_load[(level, cut)] = table
                    path = out_dir / MAX_LOAD_SUBDIR / grain_filename("max_load", level, cut)
                    _write_csv(table, path, files)
            logging.info(
                "Wrote %d maximum-load table(s) to %s.",
                len(max_load) + 1,
                out_dir / MAX_LOAD_SUBDIR,
            )

    # --- charts ----------------------------------------------------------------
    charts = 0
    if settings.export_charts:
        charts_dir = out_dir / CHARTS_SUBDIR
        charts += export_summary_charts(
            ridership,
            chart_levels,
            [("Avg daily boardings", "avg_daily_boardings")],
            "Avg daily boardings",
            charts_dir / "ridership",
            periods,
        )
        charts += export_summary_charts(
            max_load,
            chart_levels,
            [("Mean max load", "mean_max_load"), ("Peak max load", "peak_max_load")],
            "Passengers",
            charts_dir / "max_load",
            periods,
        )
        charts += export_trip_charts(
            trip_rows,
            settings.low_ridership_measure,
            settings.low_ridership_threshold,
            charts_dir / "trips",
        )
        logging.info("Wrote %d chart PNG(s) to %s.", charts, charts_dir)

    # --- run log ---------------------------------------------------------------
    holidays_text = ", ".join(d.isoformat() for d in settings.holidays) or "(none)"
    settings_lines = [
        f"stop_visits:        {settings.stop_visits_path}",
        f"trips_performed:    {settings.trips_performed_path}",
        f"Levels:             {', '.join(levels)}",
        f"Time cuts:          {', '.join(cuts)}",
        f"Maximum load:       {settings.export_max_load} "
        f"(vehicle capacity {settings.vehicle_capacity})",
        "Day types:          "
        + "; ".join(
            f"{name} = {', '.join(WEEKDAY_NAMES[n][:3] for n in nums)}"
            for name, nums in day_types.items()
        ),
        f"Holidays ({holiday}): {holidays_text}",
        f"Date range:         {settings.start_date or '(start)'} to {settings.end_date or '(end)'}",
        f"Time columns:       {', '.join(settings.time_columns)}",
        f"Service day starts: {settings.cutover_hour:02d}:00",
        f"Routes included:    {', '.join(settings.routes_to_include) or '(all)'}",
        f"Routes excluded:    {', '.join(settings.routes_to_exclude) or '(none)'}",
        f"Low ridership:      below {settings.low_ridership_threshold:g} "
        f"{settings.low_ridership_measure}",
    ]
    n_trips = len(visits[["service_date", "trip_id_performed"]].drop_duplicates())
    pooled = visits.drop_duplicates(["service_date", "trip_id_performed", *_visit_keys(visits)])
    summary_lines = [
        f"Ridership tables:   {len(ridership)}",
        f"Max-load tables:    {len(max_load) + (0 if by_trip.empty else 1)}",
        f"Total boardings:    {pooled['boardings'].sum():.0f}",
        f"Total alightings:   {pooled['alightings'].sum():.0f}",
        f"Stop visits used:   {len(pooled)}",
        f"Trips observed:     {n_trips}",
        "Service dates:      "
        + ", ".join(f"{d} {day_counts[d]}" for d in order if d in day_counts),
        f"Low-ridership trips: {low_count}",
        f"Charts written:     {charts}",
    ]
    logged = write_run_log(
        out_dir / RUN_LOG_FILENAME,
        settings_lines,
        periods,
        settings.periods_source,
        summary_lines,
    )
    if not logged and settings.require_run_log:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return RunResult(ridership, max_load, by_trip, trip_rows, service_cut_tables, files)


def _visit_keys(visits: pd.DataFrame) -> List[str]:
    """Columns that identify one stop visit within a performed trip."""
    for col in ("trip_stop_sequence", "stop_id"):
        if col in visits.columns:
            return [col]
    return []


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser (defaults are the CONFIGURATION values)."""
    p = argparse.ArgumentParser(
        description="Export TIDES ridership and maximum load at every grain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="TIDES stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="TIDES trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for the tables and run log.")
    p.add_argument(
        "--levels",
        nargs="+",
        choices=list(LEVELS),
        default=None,
        help="Levels to export; replaces EXPORT_LEVELS.",
    )
    p.add_argument(
        "--time-cuts",
        nargs="+",
        choices=list(TIME_CUTS),
        default=None,
        help="Time cuts to export; replaces EXPORT_TIME_CUTS.",
    )
    p.add_argument(
        "--max-load",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_MAX_LOAD,
        help="Export maximum-load tables.",
    )
    p.add_argument(
        "--vehicle-capacity",
        type=int,
        default=VEHICLE_CAPACITY,
        help="Capacity used for the load factor.",
    )
    periods = p.add_mutually_exclusive_group()
    periods.add_argument(
        "--time-periods-file",
        default=TIME_PERIODS_FILE,
        help="Text file of time periods, one 'NAME = HH:MM - HH:MM' per line; replaces "
        "TIME_PERIODS.",
    )
    periods.add_argument(
        "--time-period",
        action="append",
        metavar="NAME=HH:MM-HH:MM",
        help="One time period; repeat the flag for each. Replaces TIME_PERIODS and any "
        "time-periods file. Leave unset to use those.",
    )
    p.add_argument("--start-date", default=START_DATE, help="First service date (YYYY-MM-DD).")
    p.add_argument("--end-date", default=END_DATE, help="Last service date (YYYY-MM-DD).")
    p.add_argument(
        "--holiday",
        action="append",
        metavar="YYYY-MM-DD",
        help="One holiday date; repeat the flag for each. Replaces HOLIDAYS.",
    )
    p.add_argument(
        "--routes-include",
        nargs="*",
        default=list(ROUTES_TO_INCLUDE),
        metavar="ROUTE_ID",
        help="Keep only these route_ids.",
    )
    p.add_argument(
        "--routes-exclude",
        nargs="*",
        default=list(ROUTES_TO_EXCLUDE),
        metavar="ROUTE_ID",
        help="Drop these route_ids.",
    )
    p.add_argument(
        "--charts",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_CHARTS,
        help="Write chart PNGs (--charts) or skip them (--no-charts).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the settings, then build and write every enabled table.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv`` (or nothing in
            a notebook).

    Returns:
        Process exit code: 0 on success, 1 on a runtime failure, 2 if a setting
        fails its checks or the input paths are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))

    try:
        periods, source = resolve_time_periods(
            args.time_period, args.time_periods_file, TIME_PERIODS
        )
    except TimePeriodError as exc:
        logging.error("%s", exc)
        return 2
    logging.info("Time periods from %s:", source)
    for period in periods:
        logging.info("  %s", format_period(period))

    try:
        holiday_setting = "--holiday" if args.holiday else "HOLIDAYS"
        holidays = parse_iso_dates(args.holiday or HOLIDAYS, holiday_setting)
        start = parse_optional_date(args.start_date, "START_DATE")
        end = parse_optional_date(args.end_date, "END_DATE")
        settings = RunSettings(
            stop_visits_path=Path(args.stop_visits).expanduser(),
            trips_performed_path=Path(args.trips_performed).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            periods=periods,
            periods_source=source,
            levels=args.levels,
            time_cuts=args.time_cuts,
            export_max_load=args.max_load,
            vehicle_capacity=args.vehicle_capacity,
            holidays=holidays,
            start_date=start,
            end_date=end,
            routes_to_include=tuple(args.routes_include),
            routes_to_exclude=tuple(args.routes_exclude),
            export_charts=args.charts,
        )
        _resolve_run_settings(settings)
    except SettingsError as exc:
        logging.error("%s", exc)
        return 2

    if args.stop_visits == STOP_VISITS_PATH or args.trips_performed == TRIPS_PERFORMED_PATH:
        logging.warning(
            "STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. Update the "
            "CONFIGURATION section or pass --stop-visits/--trips-performed before running."
        )
        return 2
    for path in (settings.stop_visits_path, settings.trips_performed_path):
        if not path.is_file():
            logging.error("Input file not found: %s", path)
            return 1

    try:
        run(settings)
    except (InputDataError, SettingsError, OSError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
