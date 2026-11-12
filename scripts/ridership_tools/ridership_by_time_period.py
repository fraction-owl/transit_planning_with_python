"""Summarize TIDES ridership by time period, optionally by route or by route and direction.

Reads TIDES ``stop_visits`` (stop-level APC events) joined to ``trips_performed`` (route
and direction), places each stop visit in a named time period by its clock time, and
reports average daily boardings and alightings per period and day type (weekday, saturday,
sunday). ``GROUP_BY`` picks the grain: ``"system"``, ``"route"``, or ``"route_direction"``.

Time periods are named clock windows from ``TIME_PERIODS`` in CONFIGURATION, a text
file named by ``TIME_PERIODS_FILE`` (see ``time_periods_example.txt``), or
``--time-period`` flags; flags replace a file, which replaces the mapping. A gap,
overlap, malformed time, or repeated name stops the run (exit 2) before data is read.

``ridership_from_tides.py`` also has a time-period grain, but it sums all days and does
not check that the windows cover the day; ``ridership_by_hour.py`` profiles each hour.

Inputs
------
- TIDES ``stop_visits.csv`` and ``trips_performed.csv`` (for example from
  ``convert_to_tides_stop_visits.py`` and ``convert_to_tides_trips_performed.py``).
- Optional: a time-periods ``.txt`` file, one ``NAME = HH:MM - HH:MM`` per line.

Outputs
-------
- ``ridership_by_time_period_<GROUP_BY>.csv``: one row per day type, group, and time
  period, with the window and its length, average daily boardings, alightings, and
  boardings per hour, the totals behind them, and the visits, trips, and dates observed.
- A run-log sidecar with the verbatim CONFIGURATION block, the settings actually
  used, and the time periods in the same format as a time-periods file.

Method
------
- Canceled/non-revenue trips and Skipped visits are dropped; both doors' counts are summed.
- Time of day is the first readable ``TIME_COLUMNS`` value; ISO 8601 timestamps are read
  as written, so UTC offsets and daylight-saving changes do not shift them.
- Day type follows ``service_date``; the time period follows the clock, even after midnight.
- Averages divide by the service dates of that day type with any observed stop
  visit, counted before the route filters.

Limitations
-----------
- Counts are raw APC observations, not expanded to systemwide estimates; the
  stop-visit and trip counts show how much data backs each row.
- Holidays are not separated: a weekday holiday counts as a weekday.
- A trip that crosses a period boundary contributes to both periods.

Typical usage
-------------
Update the paths in CONFIGURATION (or pass ``--stop-visits`` / ``--trips-performed`` /
``--output-dir``) and run from a shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Grain of the output table:
#   "system"          - one row per day type x time period (all routes pooled)
#   "route"           - one row per day type x route_id x time period
#   "route_direction" - one row per day type x route_id x direction_id x time period
# The output is named ridership_by_time_period_<GROUP_BY>.csv, so runs at
# different grains can share OUTPUT_DIR.
GROUP_BY: str = "system"

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

# True splits rows by day type (weekday / saturday / sunday, from service_date)
# and averages over the service dates of that type. False pools every date into
# one "all days" day type.
SPLIT_BY_DAY_TYPE: bool = True

# Optional route filters, matched against route_id as text. Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# stop_visits columns tried in order for each visit's time of day; the first
# readable value in each row is used.
TIME_COLUMNS: Sequence[str] = (
    "actual_departure_time",
    "actual_arrival_time",
    "schedule_departure_time",
    "schedule_arrival_time",
)

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# CONSTANTS
# =============================================================================

# GROUP_BY value -> grouping columns (in addition to day_type and time_period).
GROUP_COLUMNS: Dict[str, List[str]] = {
    "system": [],
    "route": ["route_id"],
    "route_direction": ["route_id", "direction_id"],
}

OUTPUT_FILENAME_TEMPLATE: str = "ridership_by_time_period_{group_by}.csv"
RUN_LOG_TITLE: str = "RIDERSHIP BY TIME PERIOD RUN LOG"

MINUTES_PER_DAY: int = 24 * 60
ALL_DAYS: str = "all days"
DAY_TYPE_ORDER: Tuple[str, ...] = ("weekday", "saturday", "sunday", ALL_DAYS)
BLANK_LABEL: str = "(blank)"

# Per-door APC count columns summed into total boardings / alightings.
BOARDING_COLUMNS: Tuple[str, ...] = ("boarding_1", "boarding_2")
ALIGHTING_COLUMNS: Tuple[str, ...] = ("alighting_1", "alighting_2")

# Measure columns of the output table, after day_type, the group columns and
# time_period.
MEASURE_COLUMNS: List[str] = [
    "period_start",
    "period_end",
    "period_hours",
    "days_observed",
    "avg_daily_boardings",
    "avg_daily_alightings",
    "avg_daily_boardings_per_period_hour",
    "total_boardings",
    "total_alightings",
    "stop_visits",
    "trips_observed",
]

# =============================================================================
# TIME PERIODS
# =============================================================================


class TimePeriodError(ValueError):
    """Raised when the time periods are malformed or do not cover the day exactly once."""


class InputDataError(ValueError):
    """Raised when the TIDES tables cannot support the requested summary."""


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
    """Attach route and direction to each stop visit and keep revenue service only.

    The join keys are ``service_date`` and ``trip_id_performed`` (TIDES makes
    the trip ID unique only within a service date); dates are compared parsed,
    since the two exports may format them differently. Trips marked Canceled or
    with a ``trip_type`` other than In service are dropped, as are stop visits
    marked Skipped. Visits with no same-day trip are counted in a warning.

    Args:
        stop_visits: The TIDES ``stop_visits`` table.
        trips_performed: The TIDES ``trips_performed`` table.

    Returns:
        Stop visits with ``route_id`` / ``direction_id`` (where present) joined
        on and ``service_date`` parsed.

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

    attr_cols = [c for c in ("route_id", "direction_id") if c in trips.columns]
    joined = visits.drop(columns=attr_cols, errors="ignore").merge(
        trips.loc[revenue, [*keys, *attr_cols]], on=keys, how="inner"
    )
    if "schedule_relationship" in joined.columns:
        skipped = _casefolded(joined["schedule_relationship"], "scheduled") == "skipped"
        if skipped.any():
            logging.info("Left out %d Skipped stop visit(s).", int(skipped.sum()))
            joined = joined.loc[~skipped]
    if joined.empty:
        raise InputDataError(
            "No stop visits matched an in-service trip in trips_performed; check that the "
            "two files cover the same service dates and trip_id_performed values."
        )
    return joined.reset_index(drop=True)


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


# =============================================================================
# TIME OF DAY & DAY TYPE
# =============================================================================

# Clock time inside an ISO 8601 timestamp (2026-01-05T21:03:14-05:00 -> 21:03).
_ISO_CLOCK_RE = r"^\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})"
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


def add_time_period(
    df: pd.DataFrame,
    periods: Sequence[TimePeriod],
    time_columns: Sequence[str] = TIME_COLUMNS,
) -> pd.DataFrame:
    """Add a ``time_period`` column from each visit's clock time.

    The first readable value among *time_columns* gives the time of day. Visits
    with no readable time are counted in a warning and left out.

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
    for col in present:
        need = minutes.isna()
        if not need.any():
            break
        minutes.loc[need] = minute_of_day(df.loc[need, col]).to_numpy()
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
    lookup = period_lookup(periods)
    out["time_period"] = lookup[minutes.loc[~missing].to_numpy(dtype=int)]
    return out


def add_day_type(df: pd.DataFrame, split_by_day_type: bool = True) -> pd.DataFrame:
    """Add a ``day_type`` column from the parsed ``service_date``.

    Args:
        df: Stop visits with a parsed ``service_date``.
        split_by_day_type: True for weekday / saturday / sunday; False puts
            every date in one ``"all days"`` type.

    Returns:
        A copy of *df* with a string ``day_type`` column.
    """
    out = df.copy()
    if not split_by_day_type:
        out["day_type"] = ALL_DAYS
        return out
    dow = out["service_date"].dt.dayofweek
    out["day_type"] = np.select([dow == 5, dow == 6], ["saturday", "sunday"], default="weekday")
    return out


# =============================================================================
# FILTERING & AGGREGATION
# =============================================================================


def filter_routes(
    df: pd.DataFrame,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Keep or drop routes by ``route_id`` (compared as stripped text).

    Filter entries that match no route in the data are named in a warning.

    Args:
        df: Stop visits with ``route_id``.
        include: Route IDs to keep; empty keeps all.
        exclude: Route IDs to drop.

    Returns:
        The filtered rows.

    Raises:
        InputDataError: If no visits remain.
    """
    if not include and not exclude:
        return df
    route = df["route_id"].fillna("").astype(str).str.strip()
    known = set(route)
    wanted = {str(r).strip() for r in include}
    unwanted = {str(r).strip() for r in exclude}
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


def label_blank_groups(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Replace blank grouping values with ``"(blank)"`` so they form their own group.

    Args:
        df: Stop visits.
        group_cols: The grouping columns for the chosen grain.

    Returns:
        A copy of *df* with the grouping columns as stripped text.
    """
    out = df.copy()
    for col in group_cols:
        text = out[col].fillna("").astype(str).str.strip()
        blank = text == ""
        if blank.any():
            logging.warning(
                "%d stop visit(s) are on trips with a blank %s; they are reported as %s.",
                int(blank.sum()),
                col,
                BLANK_LABEL,
            )
        out[col] = text.where(~blank, BLANK_LABEL)
    return out


def summarize_by_time_period(
    df: pd.DataFrame,
    periods: Sequence[TimePeriod],
    group_by: str,
    days_observed: Mapping[str, int],
) -> pd.DataFrame:
    """Aggregate stop visits to one row per day type, group, and time period.

    Every group seen for a day type gets a row for every period, with zeros
    where no visit was observed, so a missing row never stands in for zero.

    Args:
        df: Stop visits with ``day_type``, ``time_period``, ``boardings``,
            ``alightings``, and the grouping columns.
        periods: The checked time periods (their order sets the row order).
        group_by: A key of :data:`GROUP_COLUMNS`.
        days_observed: Day type -> number of service dates observed.

    Returns:
        The output table (columns: ``day_type``, the group columns,
        ``time_period``, then :data:`MEASURE_COLUMNS`).
    """
    group_cols = GROUP_COLUMNS[group_by]
    outer = ["day_type", *group_cols]
    keys = [*outer, "time_period"]
    work = df.assign(
        _trip=df["service_date"].astype(str) + "|" + df["trip_id_performed"].astype(str)
    )
    agg = (
        work.groupby(keys, sort=False)
        .agg(
            total_boardings=("boardings", "sum"),
            total_alightings=("alightings", "sum"),
            stop_visits=("boardings", "size"),
            trips_observed=("_trip", "nunique"),
        )
        .reset_index()
    )
    period_frame = pd.DataFrame(
        {
            "time_period": [p.name for p in periods],
            "period_start": [_clock(p.start) for p in periods],
            "period_end": [_clock(p.end) for p in periods],
            "_minutes": [p.minutes for p in periods],
            "_period_order": list(range(len(periods))),
        }
    )
    grid = work[outer].drop_duplicates().merge(period_frame, how="cross")
    out = grid.merge(agg, on=keys, how="left")
    for col in ("total_boardings", "total_alightings", "stop_visits", "trips_observed"):
        out[col] = out[col].fillna(0)
    out["stop_visits"] = out["stop_visits"].astype(int)
    out["trips_observed"] = out["trips_observed"].astype(int)
    out["days_observed"] = out["day_type"].map(days_observed).astype(int)
    out["period_hours"] = (out["_minutes"] / 60).round(2)

    daily_boardings = out["total_boardings"] / out["days_observed"]
    out["avg_daily_boardings"] = daily_boardings.round(2)
    out["avg_daily_alightings"] = (out["total_alightings"] / out["days_observed"]).round(2)
    out["avg_daily_boardings_per_period_hour"] = (daily_boardings / (out["_minutes"] / 60)).round(2)
    out["total_boardings"] = out["total_boardings"].round(2)
    out["total_alightings"] = out["total_alightings"].round(2)

    day_order = {d: i for i, d in enumerate(DAY_TYPE_ORDER)}
    out["_day_order"] = out["day_type"].map(day_order).fillna(len(day_order))
    out = out.sort_values(["_day_order", *group_cols, "_period_order"], kind="stable")
    return out[[*keys, *MEASURE_COLUMNS]].reset_index(drop=True)


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


def write_run_log(
    log_path: Path,
    settings_lines: Sequence[str],
    periods: Sequence[TimePeriod],
    periods_source: str,
    summary_lines: Sequence[str],
) -> bool:
    """Write the run-log sidecar next to the output table.

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
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
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


class RunSettings(NamedTuple):
    """Resolved settings for one run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    periods: Sequence[TimePeriod]
    periods_source: str = "TIME_PERIODS in the CONFIGURATION block"
    group_by: str = GROUP_BY
    split_by_day_type: bool = SPLIT_BY_DAY_TYPE
    routes_to_include: Sequence[str] = ROUTES_TO_INCLUDE
    routes_to_exclude: Sequence[str] = ROUTES_TO_EXCLUDE
    time_columns: Sequence[str] = TIME_COLUMNS
    require_run_log: bool = REQUIRE_RUN_LOG


def run(settings: RunSettings) -> pd.DataFrame:
    """Build the ridership-by-time-period table, write it, and write the run log.

    Args:
        settings: Resolved settings.

    Returns:
        The table written to ``ridership_by_time_period_<group_by>.csv``.

    Raises:
        TimePeriodError: If ``settings.periods`` do not cover the day exactly once.
        InputDataError: If the TIDES tables cannot support the summary.
        OSError: If an input cannot be read or the output cannot be written.
        RuntimeError: If the run log cannot be written and it is required.
    """
    if settings.group_by not in GROUP_COLUMNS:
        raise InputDataError(
            f"GROUP_BY must be one of {', '.join(GROUP_COLUMNS)}, not {settings.group_by!r}."
        )
    problems = find_coverage_problems(settings.periods)
    if problems:
        raise TimePeriodError(
            _problem_report("The time periods do not cover the day exactly once:", problems)
        )

    group_cols = GROUP_COLUMNS[settings.group_by]
    filtering = bool(settings.routes_to_include or settings.routes_to_exclude)
    stop_visits = load_tides_csv(settings.stop_visits_path, "stop_visits")
    trips = load_tides_csv(settings.trips_performed_path, "trips_performed")
    _require_columns(trips, group_cols, "trips_performed", f"GROUP_BY={settings.group_by!r} needs")
    if filtering:
        _require_columns(trips, ["route_id"], "trips_performed", "the route filters need")

    prepared = (
        join_trips(stop_visits, trips)
        .pipe(add_ridership)
        .pipe(add_time_period, settings.periods, settings.time_columns)
        .pipe(add_day_type, settings.split_by_day_type)
    )
    days_observed = prepared.groupby("day_type")["service_date"].nunique().to_dict()
    for day_type in sorted(days_observed, key=DAY_TYPE_ORDER.index):
        logging.info("%s: averaged over %d service date(s).", day_type, days_observed[day_type])

    selected = filter_routes(prepared, settings.routes_to_include, settings.routes_to_exclude)
    selected = label_blank_groups(selected, group_cols)
    table = summarize_by_time_period(selected, settings.periods, settings.group_by, days_observed)

    out_path = settings.output_dir / OUTPUT_FILENAME_TEMPLATE.format(group_by=settings.group_by)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logging.info("Wrote %d row(s) to %s", len(table), out_path)

    n_trips = len(selected[["service_date", "trip_id_performed"]].drop_duplicates())
    settings_lines = [
        f"stop_visits:        {settings.stop_visits_path}",
        f"trips_performed:    {settings.trips_performed_path}",
        f"Group by:           {settings.group_by}",
        f"Split by day type:  {settings.split_by_day_type}",
        f"Routes included:    {', '.join(settings.routes_to_include) or '(all)'}",
        f"Routes excluded:    {', '.join(settings.routes_to_exclude) or '(none)'}",
        f"Time columns:       {', '.join(settings.time_columns)}",
    ]
    summary_lines = [
        f"Output table:       {out_path.name} ({len(table)} rows)",
        f"Total boardings:    {selected['boardings'].sum():.0f}",
        f"Total alightings:   {selected['alightings'].sum():.0f}",
        f"Stop visits used:   {len(selected)}",
        f"Trips observed:     {n_trips}",
        "Service dates:      "
        + ", ".join(
            f"{d} {days_observed[d]}" for d in sorted(days_observed, key=DAY_TYPE_ORDER.index)
        ),
    ]
    log_path = out_path.with_name(f"{out_path.stem}_runlog.txt")
    logged = write_run_log(
        log_path, settings_lines, settings.periods, settings.periods_source, summary_lines
    )
    if not logged and settings.require_run_log:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return table


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
    """Create the command-line argument parser (defaults are the CONFIGURATION values)."""
    p = argparse.ArgumentParser(
        description="Average daily TIDES ridership by time period.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="TIDES stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="TIDES trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder for the table and run log.")
    p.add_argument(
        "--group-by",
        choices=list(GROUP_COLUMNS),
        default=GROUP_BY,
        help="Grain of the output table.",
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
    p.add_argument(
        "--split-by-day-type",
        action=argparse.BooleanOptionalAction,
        default=SPLIT_BY_DAY_TYPE,
        help="Split rows by weekday / saturday / sunday.",
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
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the time periods, then build the table.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv`` (or nothing in
            a notebook).

    Returns:
        Process exit code: 0 on success, 1 on a runtime failure, 2 if the time
        periods fail their checks or the input paths are still placeholders.
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

    if args.stop_visits == STOP_VISITS_PATH or args.trips_performed == TRIPS_PERFORMED_PATH:
        logging.warning(
            "STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. Update the "
            "CONFIGURATION section or pass --stop-visits/--trips-performed before running."
        )
        return 2

    settings = RunSettings(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        periods=periods,
        periods_source=source,
        group_by=args.group_by,
        split_by_day_type=args.split_by_day_type,
        routes_to_include=tuple(args.routes_include),
        routes_to_exclude=tuple(args.routes_exclude),
    )
    for path in (settings.stop_visits_path, settings.trips_performed_path):
        if not path.is_file():
            logging.error("Input file not found: %s", path)
            return 1

    try:
        run(settings)
    except (InputDataError, OSError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
