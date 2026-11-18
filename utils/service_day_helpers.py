"""Canonical time-period, service-day clock and day-type helpers for TIDES summaries.

Holds the canonical versions of the helpers that read time periods (inline,
from a ``NAME = HH:MM - HH:MM`` text file, or from ``--time-period`` flags),
check that they cover the day exactly once, place a timestamp in a period or an
hour of the service day, and turn ``DAY_TYPES`` and date settings into checked
values. ``ridership_from_tides.py`` and ``otp_runtime_from_tides.py`` carry
verbatim copies. Per CONTRIBUTING.md, scripts do not import these at runtime,
and CI's helper-function audit flags any copy that drifts from this file.
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONSTANTS & TYPES
# -----------------------------------------------------------------------------

MINUTES_PER_DAY: int = 24 * 60

WEEKDAY_NAMES: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

# Name of the day type holidays form (the scripts set this in CONFIGURATION).
HOLIDAY_DAY_TYPE: str = "Holiday"


class TimePeriodError(ValueError):
    """Raised when the time periods are malformed or do not cover the day exactly once."""


class SettingsError(ValueError):
    """Raised when a setting (day types, dates, grain flags, measure) is unusable."""


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


# -----------------------------------------------------------------------------
# TIME PERIODS
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# SERVICE-DAY CLOCK
# -----------------------------------------------------------------------------

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
    text = text.mask(text == "", pd.NA)
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
    text = text.mask(text == "", pd.NA)
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


# -----------------------------------------------------------------------------
# DAY TYPES & DATES
# -----------------------------------------------------------------------------


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
