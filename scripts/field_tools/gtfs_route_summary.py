"""Export a one-row-per-route GTFS desk reference spreadsheet for transit planners.

Typical usage
-------------
Adjust the paths and options in the CONFIGURATION section below, then run
the script in ArcGIS Pro, a standalone Python environment, or a notebook.

Key Features
------------
* Reads a GTFS feed and classifies each service_id as Weekday / Saturday /
  Sunday / Holiday based on the real active-date calendar (calendar.txt
  and/or calendar_dates.txt). A day of the week counts when the service
  recurs on it; Holiday marks a service with no recurring day that runs on
  observed U.S. federal holidays or the dates in EXTRA_HOLIDAY_DATES.
* Computes per-route variants, directions, average trip distance, duration,
  and speed from the trips whose service has at least one active date.
* Optionally joins service-type, corridor, last-changed, and ridership
  lookups keyed on route_id.
* Exports a formatted, print-ready XLSX wall chart suitable for field use.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ==== CONFIGURATION ==========================================================

_DEFAULT_GTFS_FOLDER_PATH = r"Path\To\Your\GTFS_Folder"
_DEFAULT_BASE_OUTPUT_PATH = r"Path\To\Your\Output_Folder"

GTFS_FOLDER_PATH = _DEFAULT_GTFS_FOLDER_PATH  # <<< EDIT HERE
BASE_OUTPUT_PATH = _DEFAULT_BASE_OUTPUT_PATH  # <<< EDIT HERE
OUTPUT_FILENAME = "routes_summary.xlsx"
DISTANCE_UNIT = "meters"  # meters | kilometers | feet | miles
OUTPUT_UNITS = "imperial"  # imperial (mi/mph) or metric (km/kmh)
EXCLUDED_ROUTE_SHORT_NAMES = ["9999A", "9999B", "9999C"]

SERVICE_TYPES_PATH = ""
CORRIDORS_PATH = ""
LAST_CHANGED_PATH = ""
RIDERSHIP_PATH = ""

# calendar.txt and calendar_dates.txt are loaded separately: GTFS requires at
# least one of them, and a feed may list every service date in calendar_dates.txt.
REQUIRED_GTFS_FILES = ["routes.txt", "trips.txt", "stop_times.txt"]

# -----------------------------------------------------------------------------
# Classification thresholds
# -----------------------------------------------------------------------------
# Each day of the week is tested separately for every service_id. Between the
# service's first and last active date, the day "recurs" when the service runs
# on at least WEEKDAY_DOW_SHARE of that day's dates and on at least
# MIN_RECURRING_DATES of them. Holiday dates are left out of both counts, so
# holiday cancellations do not count against a regular pattern. A service
# earns "Weekday" if any of Monday-Friday recurs (agencies that split Monday,
# midweek, and Friday into separate service_ids still get the label), and
# "Saturday" / "Sunday" if that day recurs. A Monday-Saturday service
# therefore earns both Weekday and Saturday.
#
# WEEKDAY_DOW_SHARE: raise it (e.g. 0.90) if a service that skips many
# Saturdays still earns "Saturday"; lower it (e.g. 0.60) if a service that
# runs most Saturdays is missing the label.
#
# MIN_RECURRING_DATES keeps one-off specials (a single event Saturday) from
# earning a day label. Keep it below the number of weeks the feed covers, or
# every service will go unlabeled.
#
# A service that recurs on no day of the week but runs on at least one
# holiday date is labeled "Holiday". Holiday dates are the observed U.S.
# federal holidays across the feed's years plus EXTRA_HOLIDAY_DATES — add
# agency holidays such as Christmas Eve or the day after Thanksgiving here
# ("YYYYMMDD" or "YYYY-MM-DD"). A non-recurring service that touches no
# holiday date gets no label; it is logged as a warning. When in doubt,
# inspect the INFO logs this script emits — each service_id prints its label,
# active-date count, and holiday-date count.
WEEKDAY_DOW_SHARE: float = 0.80
MIN_RECURRING_DATES: int = 4
EXTRA_HOLIDAY_DATES: list[str] = []

# ==== FUNCTIONS ==============================================================

# ---- REUSABLE HELPERS (copied from utils/gtfs_helpers.py) ------------------


def load_gtfs_data(
    gtfs_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_path: Absolute or relative path to the folder containing the
            GTFS feed, or to a ``.zip`` archive of it — the form GTFS
            producers and most open-data portals distribute feeds in. Zip
            members may sit at the archive root or nested one level inside
            a single wrapper folder; both layouts are handled.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.
        logger: Logger for progress messages. Defaults to this module's
            logger (``logging.getLogger(__name__)``) rather than the root
            logger, so callers keep control of handler configuration.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Path missing, one of *files* not present in the feed, or
            an OS-level failure while reading a file.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, a requested file matches more than one location inside
            the zip, a file is empty, or the CSV parser fails.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs). Pandas' default NA parsing is
        disabled (``keep_default_na=False``) so identifiers such as ``"NA"``
        or ``"NULL"`` stay literal strings; only empty fields load as NaN.
        This differs from the canonical copy in ``utils/gtfs_helpers.py``.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    if not os.path.exists(gtfs_path):
        raise OSError(f"The path '{gtfs_path}' does not exist.")

    if files is None:
        files = (
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        )

    is_zip = os.path.isfile(gtfs_path) and gtfs_path.lower().endswith(".zip")
    if not is_zip and not os.path.isdir(gtfs_path):
        raise ValueError(f"'{gtfs_path}' is neither a directory nor a .zip file.")

    archive: zipfile.ZipFile | None = None
    members_by_name: dict[str, list[str]] = {}
    if is_zip:
        try:
            archive = zipfile.ZipFile(gtfs_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"'{gtfs_path}' is not a valid zip archive.") from exc
        for name in archive.namelist():
            members_by_name.setdefault(os.path.basename(name), []).append(name)

    try:
        missing: list[str] = []
        ambiguous: list[str] = []
        resolved: dict[str, str] = {}
        for file_name in files:
            if archive is None:
                if not os.path.exists(os.path.join(gtfs_path, file_name)):
                    missing.append(file_name)
                continue
            candidates = members_by_name.get(file_name, [])
            if not candidates:
                missing.append(file_name)
            elif len(candidates) > 1:
                ambiguous.append(file_name)
            else:
                resolved[file_name] = candidates[0]

        if ambiguous:
            raise ValueError(
                f"Ambiguous GTFS files in '{gtfs_path}' (found in multiple "
                f"locations): {', '.join(ambiguous)}"
            )
        if missing:
            raise OSError(f"Missing GTFS files in '{gtfs_path}': {', '.join(missing)}")

        data: dict[str, pd.DataFrame] = {}
        for file_name in files:
            key = file_name.replace(".txt", "")
            try:
                if archive is None:
                    df = pd.read_csv(
                        os.path.join(gtfs_path, file_name),
                        dtype=dtype,
                        keep_default_na=False,
                        na_values=[""],
                        low_memory=False,
                    )
                else:
                    with archive.open(resolved[file_name]) as handle:
                        df = pd.read_csv(
                            handle,
                            dtype=dtype,
                            keep_default_na=False,
                            na_values=[""],
                            low_memory=False,
                        )
                data[key] = df
                log.info("Loaded %s (%d records).", file_name, len(df))

            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"File '{file_name}' in '{gtfs_path}' is empty.") from exc

            except pd.errors.ParserError as exc:
                raise ValueError(f"Parser error in '{file_name}' in '{gtfs_path}': {exc}") from exc

        return data
    finally:
        if archive is not None:
            archive.close()


# ---- REUSABLE HELPERS (copied from utils/calendar_helpers.py) --------------


def expand_service_active_dates(
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame] = None,
    max_days_per_service: int = 1830,
    today: Optional[dt.date] = None,
) -> dict[str, set[dt.date]]:
    """Expand each service_id into its real set of active calendar dates.

    Builds the base date set from each ``calendar.txt`` row (day-of-week
    pattern × ``start_date``–``end_date`` range), then applies
    ``calendar_dates.txt`` exceptions (``exception_type`` 1 adds a date,
    2 removes it). Handles calendar_dates-only feeds (*calendar_df* empty or
    ``None``), redundant additions, and fully negated base patterns — the
    returned sets reflect only the dates a service truly operates.

    Rows with unparseable or reversed dates are skipped with a warning.
    A date range longer than *max_days_per_service* (a common placeholder
    pattern, e.g. 2000–2099) is clamped to a window of that length centred
    on *today* and logged, so expansion stays fast and downstream per-year
    statistics stay meaningful.

    Args:
        calendar_df: Parsed ``calendar.txt``, or ``None`` if the feed has
            none. Expected columns: ``service_id``, the seven day-of-week
            flags, ``start_date``, ``end_date``.
        calendar_dates_df: Parsed ``calendar_dates.txt`` or ``None``.
            Expected columns: ``service_id``, ``date``, ``exception_type``.
        max_days_per_service: Longest date range expanded per service before
            clamping kicks in. The default (1830 ≈ 5 years) is far beyond
            any real service span but well short of placeholder ranges.
        today: Anchor date for clamping oversized ranges. Defaults to the
            current date; pass a fixed date for deterministic tests.

    Returns:
        Mapping of ``service_id`` (as ``str``) to the set of dates the
        service operates. Services whose dates never parse map to an empty
        set rather than being dropped, so callers can report them.

    Raises:
        ValueError: If *calendar_df* is provided but lacks ``service_id``,
            ``start_date``, or ``end_date`` columns.
    """
    day_cols = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    anchor = dt.date.today() if today is None else today
    active: dict[str, set[dt.date]] = {}

    if calendar_df is not None and not calendar_df.empty:
        required = {"service_id", "start_date", "end_date"}
        missing = required - set(calendar_df.columns)
        if missing:
            raise ValueError(f"calendar.txt is missing required column(s): {sorted(missing)}")
        for _, row in calendar_df.iterrows():
            sid = str(row["service_id"]).strip()
            try:
                start = dt.datetime.strptime(str(row["start_date"]).strip(), "%Y%m%d").date()
                end = dt.datetime.strptime(str(row["end_date"]).strip(), "%Y%m%d").date()
            except ValueError:
                logging.warning("Service %s: unparseable start/end date — skipping row.", sid)
                active.setdefault(sid, set())
                continue
            if end < start:
                logging.warning(
                    "Service %s: end_date %s precedes start_date %s — skipping row.",
                    sid,
                    end,
                    start,
                )
                active.setdefault(sid, set())
                continue
            if (end - start).days + 1 > max_days_per_service:
                half = max_days_per_service // 2
                clamped_start = max(start, anchor - dt.timedelta(days=half))
                clamped_end = min(end, anchor + dt.timedelta(days=half))
                logging.warning(
                    "Service %s: date range %s–%s looks like a placeholder; "
                    "clamping expansion to %s–%s.",
                    sid,
                    start,
                    end,
                    clamped_start,
                    clamped_end,
                )
                start, end = clamped_start, clamped_end
            pattern = [str(row.get(c, "0")).strip() == "1" for c in day_cols]
            dates = active.setdefault(sid, set())
            d = start
            while d <= end:
                if pattern[d.weekday()]:
                    dates.add(d)
                d += dt.timedelta(days=1)

    if calendar_dates_df is not None and not calendar_dates_df.empty:
        bad_rows = 0
        for _, row in calendar_dates_df.iterrows():
            sid = str(row["service_id"]).strip()
            try:
                d = dt.datetime.strptime(str(row["date"]).strip(), "%Y%m%d").date()
            except ValueError:
                bad_rows += 1
                continue
            etype = str(row.get("exception_type", "")).strip()
            dates = active.setdefault(sid, set())
            if etype == "1":
                dates.add(d)
            elif etype == "2":
                dates.discard(d)
            else:
                bad_rows += 1
        if bad_rows:
            logging.warning(
                "calendar_dates.txt: skipped %d row(s) with unparseable date/exception_type.",
                bad_rows,
            )

    return active


# ---- REUSABLE HELPERS (copied from utils/time_helpers.py) ------------------


def federal_holidays_observed(year: int) -> set[dt.date]:
    """Return the observed dates of the U.S. federal holidays of *year*.

    Covers the eleven holidays of 5 U.S.C. 6103: New Year's Day, Birthday of
    Martin Luther King Jr. (3rd Monday of January), Washington's Birthday
    (3rd Monday of February), Memorial Day (last Monday of May), Juneteenth
    (June 19, from its 2021 establishment onward), Independence Day, Labor
    Day (1st Monday of September), Columbus Day (2nd Monday of October),
    Veterans Day, Thanksgiving (4th Thursday of November), and Christmas.

    Fixed-date holidays falling on a Saturday are observed on the preceding
    Friday and those falling on a Sunday on the following Monday, so an
    observed date can land in the *previous* calendar year (e.g. New Year's
    Day 2022 was observed on 2021-12-31). Callers classifying a span of dates
    should therefore union this set over ``range(first_year, last_year + 2)``.

    Args:
        year: Calendar year whose holidays are computed.

    Returns:
        The observed dates of *year*'s federal holidays.
    """

    def nth_weekday(month: int, weekday: int, n: int) -> dt.date:
        first = dt.date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + dt.timedelta(days=offset + 7 * (n - 1))

    def last_monday(month: int) -> dt.date:
        next_month = dt.date(year + (month == 12), month % 12 + 1, 1)
        last = next_month - dt.timedelta(days=1)
        return last - dt.timedelta(days=last.weekday())

    def observed(day: dt.date) -> dt.date:
        if day.weekday() == 5:  # Saturday -> preceding Friday
            return day - dt.timedelta(days=1)
        if day.weekday() == 6:  # Sunday -> following Monday
            return day + dt.timedelta(days=1)
        return day

    fixed = [
        dt.date(year, 1, 1),  # New Year's Day
        dt.date(year, 7, 4),  # Independence Day
        dt.date(year, 11, 11),  # Veterans Day
        dt.date(year, 12, 25),  # Christmas Day
    ]
    if year >= 2021:
        fixed.append(dt.date(year, 6, 19))  # Juneteenth
    floating = [
        nth_weekday(1, 0, 3),  # Birthday of Martin Luther King Jr.
        nth_weekday(2, 0, 3),  # Washington's Birthday
        last_monday(5),  # Memorial Day
        nth_weekday(9, 0, 1),  # Labor Day
        nth_weekday(10, 0, 2),  # Columbus Day
        nth_weekday(11, 3, 4),  # Thanksgiving Day
    ]
    return {observed(day) for day in fixed} | set(floating)


# ---- SCRIPT FUNCTIONS -------------------------------------------------------


def hms_to_seconds(time_str: str) -> Optional[int]:
    """Convert a GTFS ``HH:MM:SS`` string to seconds since service start.

    GTFS allows hours >= 24 to represent trips that span midnight; this
    function preserves that overflow rather than taking a modulo.

    Args:
        time_str: Time string in ``HH:MM:SS`` form.

    Returns:
        Integer seconds, or ``None`` if the input is missing/invalid.
    """
    if time_str is None or (isinstance(time_str, float) and pd.isna(time_str)):
        return None
    try:
        h, m, s = str(time_str).strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def parse_holiday_dates(values: Sequence[str]) -> set[dt.date]:
    """Parse configured holiday dates.

    Args:
        values: Dates as ``"YYYYMMDD"`` (the GTFS form) or ``"YYYY-MM-DD"``.

    Returns:
        The parsed dates.

    Raises:
        ValueError: An entry matches neither format.
    """
    dates: set[dt.date] = set()
    for value in values:
        text = str(value).strip()
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                dates.add(dt.datetime.strptime(text, fmt).date())
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Holiday date {value!r} is not YYYYMMDD or YYYY-MM-DD.")
    return dates


def holiday_dates_for(
    active_dates: Mapping[str, set[dt.date]],
    extra_holiday_dates: Sequence[str] = (),
) -> set[dt.date]:
    """Return the holiday dates that apply to a feed.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        extra_holiday_dates: Agency holidays to add, as accepted by
            :func:`parse_holiday_dates`.

    Returns:
        Observed U.S. federal holidays across the feed's active years (plus
        the following year, whose New Year's Day can be observed on December
        31), together with *extra_holiday_dates*.
    """
    holidays = parse_holiday_dates(extra_holiday_dates)
    years = {d.year for dates in active_dates.values() for d in dates}
    if years:
        for year in range(min(years), max(years) + 2):
            holidays |= federal_holidays_observed(year)
    return holidays


def classify_services(
    active_dates: Mapping[str, set[dt.date]],
    holiday_dates: Optional[set[dt.date]] = None,
    weekday_dow_share: float = WEEKDAY_DOW_SHARE,
    min_recurring_dates: int = MIN_RECURRING_DATES,
) -> dict[str, set[str]]:
    """Classify each service_id by the days of the week it recurs on.

    Each day of the week is tested on its own. Between the service's first
    and last active date, the day recurs when the service runs on at least
    *weekday_dow_share* of that day's dates and on at least
    *min_recurring_dates* of them; holiday dates are left out of both
    counts. A service earns ``Weekday`` if any of Monday-Friday recurs, and
    ``Saturday`` / ``Sunday`` if that day recurs. A service with no
    recurring day that runs on at least one holiday date is ``Holiday``.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        holiday_dates: Holiday dates, e.g. from :func:`holiday_dates_for`.
            ``None`` means no holidays.
        weekday_dow_share: Minimum fraction of a day of the week's dates
            the service must run on for that day to recur.
        min_recurring_dates: Minimum number of dates the service must run
            on a day of the week for that day to recur.

    Returns:
        Mapping of ``service_id`` to a set of labels drawn from
        ``{"Weekday", "Saturday", "Sunday", "Holiday"}``. A service with no
        active dates, or with no recurring day and no holiday date, maps to
        an empty set.
    """
    holidays = holiday_dates if holiday_dates is not None else set()
    result: dict[str, set[str]] = {}
    for sid, dates in active_dates.items():
        if not dates:
            result[sid] = set()
            logging.info("Service %s: empty (0 active dates)", sid)
            continue
        available = [0] * 7
        d, last = min(dates), max(dates)
        while d <= last:
            if d not in holidays:
                available[d.weekday()] += 1
            d += dt.timedelta(days=1)
        served = [0] * 7
        for d in dates - holidays:
            served[d.weekday()] += 1
        recurs = [
            served[dow] >= min_recurring_dates and served[dow] >= weekday_dow_share * available[dow]
            for dow in range(7)
        ]
        on_holidays = len(dates & holidays)
        labels: set[str] = set()
        if any(recurs[:5]):
            labels.add("Weekday")
        if recurs[5]:
            labels.add("Saturday")
        if recurs[6]:
            labels.add("Sunday")
        if not labels and on_holidays:
            labels.add("Holiday")
        result[sid] = labels
        if not labels:
            logging.warning(
                "Service %s: %s active date(s) with no recurring day of the week and none on "
                "a holiday date; no day label. Add its dates to EXTRA_HOLIDAY_DATES if it is "
                "holiday service.",
                sid,
                len(dates),
            )
            continue
        logging.info(
            "Service %s -> %s (%s dates, %s on holidays)",
            sid,
            sorted(labels),
            len(dates),
            on_holidays,
        )
    return result


def first_to_last_distance(df: pd.DataFrame, id_col: str, seq_col: str) -> pd.Series:
    """Return the ``shape_dist_traveled`` covered from each group's first row to its last.

    A group's distance is known only when its first and last rows (ordered by
    *seq_col*) both carry a numeric ``shape_dist_traveled`` and the last
    exceeds the first; other groups are left out rather than reported as a
    partial or zero distance.

    Args:
        df: Table with *id_col*, *seq_col*, and ``shape_dist_traveled``.
        id_col: Grouping column, e.g. ``trip_id`` or ``shape_id``.
        seq_col: Ordering column, e.g. ``stop_sequence`` or ``shape_pt_sequence``.

    Returns:
        Series indexed by *id_col* with distances in the feed's units.
    """
    tmp = pd.DataFrame(
        {
            "id": df[id_col],
            "seq": pd.to_numeric(df[seq_col], errors="coerce"),
            "d": pd.to_numeric(df["shape_dist_traveled"], errors="coerce"),
        }
    ).sort_values(["id", "seq"], kind="stable")
    first = tmp.drop_duplicates("id", keep="first").set_index("id")["d"]
    last = tmp.drop_duplicates("id", keep="last").set_index("id")["d"]
    span = (last - first).astype(float).rename_axis(id_col)
    return span[span > 0]


def trip_distances_meters(
    stop_times_df: pd.DataFrame,
    shapes_df: Optional[pd.DataFrame],
    trips_df: pd.DataFrame,
    distance_unit: str,
) -> pd.Series:
    """Return a ``trip_id`` -> meters Series for all trips with a known length.

    Each trip uses ``stop_times.txt`` distances when its first and last stops
    both have them, and otherwise falls back to the full length of its shape
    in ``shapes.txt``. Trips with neither are left out.

    Args:
        stop_times_df: Parsed ``stop_times.txt``.
        shapes_df: Parsed ``shapes.txt`` or ``None``.
        trips_df: Parsed ``trips.txt``.
        distance_unit: Unit of ``shape_dist_traveled`` in the feed.

    Returns:
        Series indexed by ``trip_id`` with distances in metres.
    """
    factors: dict[str, float] = {
        "meters": 1.0,
        "kilometers": 1000.0,
        "feet": 0.3048,
        "miles": 1609.344,
    }
    factor = factors.get(distance_unit, 1.0)

    per_trip = pd.Series(dtype=float)
    if "shape_dist_traveled" in stop_times_df.columns:
        per_trip = first_to_last_distance(stop_times_df, "trip_id", "stop_sequence") * factor

    if (
        shapes_df is not None
        and not shapes_df.empty
        and {"shape_id", "shape_pt_sequence", "shape_dist_traveled"} <= set(shapes_df.columns)
        and "shape_id" in trips_df.columns
    ):
        per_shape = first_to_last_distance(shapes_df, "shape_id", "shape_pt_sequence") * factor
        from_shapes = trips_df.set_index("trip_id")["shape_id"].map(per_shape).dropna()
        from_shapes = from_shapes[~from_shapes.index.isin(per_trip.index)]
        per_trip = pd.concat([per_trip, from_shapes.astype(float)])

    return per_trip


def trip_durations_seconds(stop_times_df: pd.DataFrame) -> pd.Series:
    """Return a ``trip_id`` -> duration-in-seconds Series.

    Args:
        stop_times_df: Parsed ``stop_times.txt``.

    Returns:
        Series indexed by ``trip_id`` with trip durations in seconds.
    """
    df = stop_times_df.copy()
    df["_seq"] = pd.to_numeric(df["stop_sequence"], errors="coerce")
    df = df.sort_values(["trip_id", "_seq"])
    grp = df.groupby("trip_id")
    first_dep = grp["departure_time"].first().map(hms_to_seconds)
    last_arr = grp["arrival_time"].last().map(hms_to_seconds)
    return (last_arr - first_dep).dropna()


def load_optional_lookup(path: str, value_col: str) -> dict[str, str]:
    """Load an optional route_id -> value lookup from CSV or TSV.

    Args:
        path: Filesystem path to a CSV or TSV file.
        value_col: Name of the column to use as the value.

    Returns:
        Mapping of ``route_id`` to the value in *value_col*, or an empty
        dict if the file is absent or cannot be parsed.
    """
    if not path:
        return {}
    if not os.path.isfile(path):
        logging.warning("Optional lookup not found: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            first = fh.readline()
        sep = "\t" if "\t" in first else ","
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logging.warning("Could not read lookup %s: %s", path, exc)
        return {}
    if "route_id" not in df.columns or value_col not in df.columns:
        logging.warning("Lookup %s missing route_id or %s column", path, value_col)
        return {}
    blank_ids = df["route_id"].isna()
    if blank_ids.any():
        logging.warning(
            "Lookup %s: skipping %d row(s) with a blank route_id", path, int(blank_ids.sum())
        )
    df = df[~blank_ids]
    return dict(zip(df["route_id"], df[value_col].fillna("")))


def build_summary(
    routes_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame],
    shapes_df: Optional[pd.DataFrame],
    distance_unit: str,
    extras: Mapping[str, Mapping[str, str]],
    weekday_dow_share: float = WEEKDAY_DOW_SHARE,
    min_recurring_dates: int = MIN_RECURRING_DATES,
    extra_holiday_dates: Sequence[str] = EXTRA_HOLIDAY_DATES,
) -> pd.DataFrame:
    """Assemble the per-route summary DataFrame.

    Args:
        routes_df: Parsed ``routes.txt``.
        trips_df: Parsed ``trips.txt``.
        stop_times_df: Parsed ``stop_times.txt``.
        calendar_df: Parsed ``calendar.txt`` or ``None``.
        calendar_dates_df: Parsed ``calendar_dates.txt`` or ``None``.
        shapes_df: Parsed ``shapes.txt`` or ``None``.
        distance_unit: Unit of ``shape_dist_traveled`` in the feed.
        extras: Optional lookup dicts keyed by category then ``route_id``.
        weekday_dow_share: Forwarded to :func:`classify_services`.
        min_recurring_dates: Forwarded to :func:`classify_services`.
        extra_holiday_dates: Agency holidays added to the observed federal
            holidays; see :func:`holiday_dates_for`.

    Returns:
        One-row-per-route :class:`pandas.DataFrame` ready for export. Trips
        whose service has no active dates are left out of every column.
    """
    active_dates = expand_service_active_dates(calendar_df, calendar_dates_df)
    service_labels = classify_services(
        active_dates,
        holiday_dates_for(active_dates, extra_holiday_dates),
        weekday_dow_share=weekday_dow_share,
        min_recurring_dates=min_recurring_dates,
    )
    dist_m = trip_distances_meters(stop_times_df, shapes_df, trips_df, distance_unit)
    dur_s = trip_durations_seconds(stop_times_df)

    t = trips_df.copy()
    t["_sid"] = t["service_id"].map(lambda v: v.strip() if isinstance(v, str) else None)
    running = t["_sid"].map(lambda sid: bool(active_dates.get(sid)) if sid else False)
    if not running.all():
        logging.info(
            "Skipping %d trip(s) whose service_id has no active dates.", int((~running).sum())
        )
    t = t[running].copy()
    t["_dist_m"] = t["trip_id"].map(dist_m)
    t["_dur_s"] = t["trip_id"].map(dur_s)

    excluded = {str(x) for x in (EXCLUDED_ROUTE_SHORT_NAMES or [])}
    imperial = str(OUTPUT_UNITS).lower() == "imperial"
    dist_col = "avg_distance_mi" if imperial else "avg_distance_km"
    speed_col = "avg_speed_mph" if imperial else "avg_speed_kmh"

    rows = []
    for _, r in routes_df.iterrows():
        rid = r["route_id"]
        if str(r.get("route_short_name", "")) in excluded:
            logging.info("Excluding route %s", r.get("route_short_name", ""))
            continue
        rt = t[t["route_id"] == rid]
        if rt.empty:
            continue

        if "shape_id" in rt.columns and rt["shape_id"].notna().any():
            variants = int(rt["shape_id"].nunique())
        elif "trip_headsign" in rt.columns and rt["trip_headsign"].notna().any():
            variants = int(rt["trip_headsign"].nunique())
        else:
            variants = 1
        variants = max(variants, 1)

        directions = int(rt["direction_id"].nunique()) if "direction_id" in rt.columns else 1
        directions = max(directions, 1)

        day_cats: set[str] = set()
        for sid in rt["_sid"].unique():
            day_cats |= service_labels.get(sid, set())

        avg_dist_m = rt["_dist_m"].dropna().mean()
        avg_dur_s = rt["_dur_s"].dropna().mean()
        # Speed uses only trips with both a distance and a positive duration,
        # so one trip's distance is never divided by another trip's time.
        timed = rt[rt["_dist_m"].notna() & (rt["_dur_s"] > 0)]
        if pd.notna(avg_dist_m):
            avg_distance: Optional[float] = round(
                avg_dist_m / (1609.344 if imperial else 1000.0), 2
            )
        else:
            avg_distance = None
        avg_duration_min: Optional[float] = (
            round(avg_dur_s / 60.0, 1) if pd.notna(avg_dur_s) else None
        )
        if not timed.empty:
            mps = timed["_dist_m"].sum() / timed["_dur_s"].sum()
            avg_speed: Optional[float] = round(mps * (2.23694 if imperial else 3.6), 1)
        else:
            avg_speed = None

        rows.append(
            {
                "route_short_name": r.get("route_short_name", ""),
                "route_long_name": r.get("route_long_name", ""),
                "variants": variants,
                "directions": directions,
                "weekday": "Y" if "Weekday" in day_cats else "",
                "saturday": "Y" if "Saturday" in day_cats else "",
                "sunday": "Y" if "Sunday" in day_cats else "",
                "holiday": "Y" if "Holiday" in day_cats else "",
                dist_col: avg_distance,
                "avg_duration_min": avg_duration_min,
                speed_col: avg_speed,
                "service_type": extras.get("service_types", {}).get(rid, ""),
                "corridor": extras.get("corridors", {}).get(rid, ""),
                "last_changed": extras.get("last_changed", {}).get(rid, ""),
                "ridership": extras.get("ridership", {}).get(rid, ""),
            }
        )

    cols = [
        "route_short_name",
        "route_long_name",
        "variants",
        "directions",
        "weekday",
        "saturday",
        "sunday",
        "holiday",
        dist_col,
        "avg_duration_min",
        speed_col,
        "service_type",
        "corridor",
        "last_changed",
        "ridership",
    ]
    return pd.DataFrame(rows, columns=cols)


def export_to_xlsx(data_frame: pd.DataFrame, output_file: str) -> None:
    """Write the summary to a formatted XLSX wall chart.

    Freezes the header row, bolds headers, widens columns based on
    content, wraps ``route_long_name``, and enables print-friendly
    settings (landscape, fit-to-width, repeat header on every page).

    Args:
        data_frame: Summary built by :func:`build_summary`.
        output_file: Destination ``.xlsx`` path.
    """
    if data_frame is None or data_frame.empty:
        logging.info("No rows to export; skipping write.")
        return
    if not output_file.lower().endswith(".xlsx"):
        output_file = os.path.splitext(output_file)[0] + ".xlsx"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Routes"

    headers = list(data_frame.columns)
    ws.append(headers)
    for row in data_frame.itertuples(index=False):
        ws.append(list(row))

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="305496")
    center = Alignment(horizontal="center", vertical="center")
    wrap = Alignment(horizontal="left", vertical="center", wrap_text=True)

    for col_idx, _name in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center

    widths: dict[str, Union[int, float]] = {
        "route_short_name": 12,
        "route_long_name": 40,
        "variants": 9,
        "directions": 10,
        "weekday": 9,
        "saturday": 9,
        "sunday": 8,
        "holiday": 9,
        "avg_distance_mi": 14,
        "avg_distance_km": 14,
        "avg_duration_min": 15,
        "avg_speed_mph": 13,
        "avg_speed_kmh": 13,
        "service_type": 14,
        "corridor": 16,
        "last_changed": 14,
        "ridership": 12,
    }
    for col_idx, name in enumerate(headers, start=1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = widths.get(name, 12)

    long_col = headers.index("route_long_name") + 1 if "route_long_name" in headers else None
    for row_idx in range(2, ws.max_row + 1):
        if long_col is not None:
            ws.cell(row=row_idx, column=long_col).alignment = wrap
        ws.row_dimensions[row_idx].height = 22

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.page_setup.orientation = ws.ORIENTATION_LANDSCAPE
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True  # ty: ignore[invalid-assignment]
    ws.print_title_rows = "1:1"

    wb.save(output_file)
    logging.info("Wrote %s (%s rows)", output_file, len(data_frame))


# ==== MAIN ===================================================================


def main() -> int:
    """Entry point: load feed, build summary, write XLSX.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders or invalid.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if (
        GTFS_FOLDER_PATH == _DEFAULT_GTFS_FOLDER_PATH
        or BASE_OUTPUT_PATH == _DEFAULT_BASE_OUTPUT_PATH
    ):
        logging.warning(
            "GTFS_FOLDER_PATH and/or BASE_OUTPUT_PATH are still set to their default "
            "placeholder values. Please update them in the CONFIGURATION section before running."
        )
        return 2
    try:
        parse_holiday_dates(EXTRA_HOLIDAY_DATES)
    except ValueError as exc:
        logging.error("EXTRA_HOLIDAY_DATES: %s", exc)
        return 2

    logging.info("==== GTFS Route Summary ====")
    logging.info("GTFS dir      : %s", GTFS_FOLDER_PATH)
    logging.info("Output        : %s", os.path.join(BASE_OUTPUT_PATH, OUTPUT_FILENAME))
    logging.info("Distance unit : %s", DISTANCE_UNIT)
    logging.info("Weekday share : %s", WEEKDAY_DOW_SHARE)
    logging.info("Min recurring : %s", MIN_RECURRING_DATES)
    logging.info("Extra holidays: %s", ", ".join(EXTRA_HOLIDAY_DATES) or "(none)")

    try:
        core = load_gtfs_data(GTFS_FOLDER_PATH, files=REQUIRED_GTFS_FILES)

        calendar_df: Optional[pd.DataFrame] = None
        try:
            ca = load_gtfs_data(GTFS_FOLDER_PATH, files=("calendar.txt",))
            calendar_df = ca.get("calendar")
        except OSError as exc:
            logging.info("calendar.txt unavailable: %s", exc)

        calendar_dates_df: Optional[pd.DataFrame] = None
        try:
            cd = load_gtfs_data(GTFS_FOLDER_PATH, files=("calendar_dates.txt",))
            calendar_dates_df = cd.get("calendar_dates")
        except OSError as exc:
            logging.info("calendar_dates.txt unavailable: %s", exc)

        if calendar_df is None and calendar_dates_df is None:
            raise OSError(
                f"'{GTFS_FOLDER_PATH}' has neither calendar.txt nor calendar_dates.txt; "
                "GTFS requires at least one."
            )

        shapes_df: Optional[pd.DataFrame] = None
        try:
            sh = load_gtfs_data(GTFS_FOLDER_PATH, files=("shapes.txt",))
            shapes_df = sh.get("shapes")
        except OSError as exc:
            logging.warning("shapes.txt unavailable: %s", exc)

        extras: dict[str, dict[str, str]] = {
            "service_types": load_optional_lookup(SERVICE_TYPES_PATH, "service_type"),
            "corridors": load_optional_lookup(CORRIDORS_PATH, "corridor"),
            "last_changed": load_optional_lookup(LAST_CHANGED_PATH, "last_changed"),
            "ridership": load_optional_lookup(RIDERSHIP_PATH, "ridership"),
        }

        summary = build_summary(
            routes_df=core["routes"],
            trips_df=core["trips"],
            stop_times_df=core["stop_times"],
            calendar_df=calendar_df,
            calendar_dates_df=calendar_dates_df,
            shapes_df=shapes_df,
            distance_unit=DISTANCE_UNIT,
            extras=extras,
            weekday_dow_share=WEEKDAY_DOW_SHARE,
            min_recurring_dates=MIN_RECURRING_DATES,
            extra_holiday_dates=EXTRA_HOLIDAY_DATES,
        )

        if summary.empty:
            logging.warning("Summary is empty; nothing to write.")
            return 1

        export_to_xlsx(summary, os.path.join(BASE_OUTPUT_PATH, OUTPUT_FILENAME))
        logging.info("Script completed successfully.")
        return 0

    except (OSError, ValueError) as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1
    except Exception:
        logging.exception("Unexpected error in pipeline")
        return 1
    finally:
        logging.info("Exiting script.")


if __name__ == "__main__":
    raise SystemExit(main())
