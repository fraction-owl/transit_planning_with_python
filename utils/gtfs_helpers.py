"""General-purpose helper functions for GTFS and transit data workflows.

Includes reusable utilities for loading GTFS files and other common tasks used
across transit data processing scripts.
"""

from __future__ import annotations

import logging
import os
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def validate_gtfs_files_exist(
    gtfs_path: str,
    files: Optional[Sequence[str]] = None,
) -> None:
    """Check that specific GTFS text files exist and log a warning if missing.

    Args:
        gtfs_path: Absolute or relative path to the folder containing the
            GTFS feed, or to a ``.zip`` archive of it. Zip members may sit
            at the archive root or nested one level inside a single
            wrapper folder — both layouts are common among GTFS producers
            and open-data portals.
        files: Explicit sequence of file names to check. If ``None``,
            a standard set of GTFS files is checked.
    """
    if not os.path.exists(gtfs_path):
        logging.warning("The path '%s' does not exist.", gtfs_path)
        return

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
    if not is_zip:
        if not os.path.isdir(gtfs_path):
            logging.warning("'%s' is neither a directory nor a .zip file.", gtfs_path)
            return
        for file_name in files:
            if not os.path.exists(os.path.join(gtfs_path, file_name)):
                logging.warning("Missing GTFS file: %s", file_name)
        return

    try:
        with zipfile.ZipFile(gtfs_path) as archive:
            names_by_basename: dict[str, list[str]] = {}
            for name in archive.namelist():
                names_by_basename.setdefault(os.path.basename(name), []).append(name)
    except zipfile.BadZipFile:
        logging.warning("'%s' is not a valid zip archive.", gtfs_path)
        return

    for file_name in files:
        matches = names_by_basename.get(file_name, [])
        if not matches:
            logging.warning("Missing GTFS file: %s", file_name)
        elif len(matches) > 1:
            logging.warning("Ambiguous GTFS file (found in multiple locations): %s", file_name)


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
        pitfalls (e.g. leading zeros in IDs).
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
                        os.path.join(gtfs_path, file_name), dtype=dtype, low_memory=False
                    )
                else:
                    with archive.open(resolved[file_name]) as handle:
                        df = pd.read_csv(handle, dtype=dtype, low_memory=False)
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


def expand_service_dates(
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Expand GTFS calendars into one row per active (service_date, service_id).

    Both ``calendar.txt`` (weekly patterns bounded by a date range) and
    ``calendar_dates.txt`` (per-date add/remove exceptions) are honored, so
    feeds that use either convention -- or both -- expand correctly. Exception
    type 1 rows add a (date, service) pair; type 2 rows remove one, including
    pairs the weekly pattern would otherwise activate.

    Args:
        calendar: Parsed ``calendar.txt`` with string columns, or ``None`` /
            empty when the feed has no such file.
        calendar_dates: Parsed ``calendar_dates.txt`` with string columns, or
            ``None`` / empty when the feed has no such file.
        start_date: First service date (inclusive) of the expansion window.
        end_date: Last service date (inclusive) of the expansion window.

    Returns:
        DataFrame with normalized datetime ``service_date`` and string
        ``service_id`` columns, one row per active pair, de-duplicated and
        sorted. May be empty if no service falls inside the window.

    Raises:
        ValueError: If the window is inverted, both calendar inputs are
            missing/empty, or a required column is absent.
    """
    window_start = start_date.normalize()
    window_end = end_date.normalize()
    if window_start > window_end:
        raise ValueError(f"start_date {window_start.date()} is after end_date {window_end.date()}.")

    has_calendar = calendar is not None and not calendar.empty
    has_dates = calendar_dates is not None and not calendar_dates.empty
    if not has_calendar and not has_dates:
        raise ValueError(
            "Neither calendar.txt nor calendar_dates.txt has rows; cannot expand service dates."
        )

    day_columns = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    frames: list[pd.DataFrame] = []

    if has_calendar:
        required = ["service_id", "start_date", "end_date", *day_columns]
        missing = [col for col in required if col not in calendar.columns]
        if missing:
            raise ValueError(f"calendar.txt is missing required column(s): {', '.join(missing)}")
        cal = calendar.copy()
        cal["_start"] = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
        cal["_end"] = pd.to_datetime(cal["end_date"], format="%Y%m%d", errors="coerce")
        cal_long = cal.melt(
            id_vars=["service_id", "_start", "_end"],
            value_vars=day_columns,
            var_name="_day_name",
            value_name="_runs",
        )
        cal_long["_dow"] = cal_long["_day_name"].map(
            {name: i for i, name in enumerate(day_columns)}
        )
        dates = pd.DataFrame({"service_date": pd.date_range(window_start, window_end, freq="D")})
        dates["_dow"] = dates["service_date"].dt.dayofweek
        merged = dates.merge(cal_long, on="_dow", how="inner")
        keep = (
            (merged["_runs"].astype(str).str.strip() == "1")
            & (merged["service_date"] >= merged["_start"])
            & (merged["service_date"] <= merged["_end"])
        )
        frames.append(merged.loc[keep, ["service_date", "service_id"]])

    removed: Optional[pd.DataFrame] = None
    if has_dates:
        required = ["service_id", "date", "exception_type"]
        missing = [col for col in required if col not in calendar_dates.columns]
        if missing:
            raise ValueError(
                f"calendar_dates.txt is missing required column(s): {', '.join(missing)}"
            )
        cd = calendar_dates.copy()
        cd["service_date"] = pd.to_datetime(cd["date"], format="%Y%m%d", errors="coerce")
        cd = cd.loc[cd["service_date"].between(window_start, window_end)]
        exception = cd["exception_type"].astype(str).str.strip()
        frames.append(cd.loc[exception == "1", ["service_date", "service_id"]])
        removed = cd.loc[exception == "2", ["service_date", "service_id"]]

    active = pd.concat(frames, ignore_index=True)
    active["service_id"] = active["service_id"].astype(str)
    active = active.drop_duplicates()

    if removed is not None and not removed.empty:
        removed = removed.copy()
        removed["service_id"] = removed["service_id"].astype(str)
        active = active.merge(
            removed.drop_duplicates(),
            on=["service_date", "service_id"],
            how="left",
            indicator=True,
        )
        active = active.loc[active["_merge"] == "left_only", ["service_date", "service_id"]]

    return active.sort_values(["service_date", "service_id"], ignore_index=True)


def classify_day_type(dates: pd.Series) -> pd.Series:
    """Label each date ``Weekday`` / ``Saturday`` / ``Sunday`` by day of week.

    Holiday-aware relabeling (e.g. a Monday running Sunday service) is
    deliberately out of scope: callers that want holiday-like dates excluded
    should drop the dates returned by ``find_exception_dates`` first, so both
    sides of a schedule-vs-observed comparison see the same date pool.

    Args:
        dates: Series of datetime-like values.

    Returns:
        String Series aligned to ``dates`` holding ``"Weekday"``,
        ``"Saturday"``, or ``"Sunday"``.
    """
    dow = pd.to_datetime(dates).dt.dayofweek
    return dow.map({5: "Saturday", 6: "Sunday"}).fillna("Weekday").astype(str)


def find_exception_dates(
    calendar_dates: Optional[pd.DataFrame],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> set[pd.Timestamp]:
    """Return dates carrying any ``calendar_dates.txt`` exception row.

    These are the holiday-like dates whose service differs from the weekly
    pattern (service added, removed, or swapped to another day's schedule).
    Coverage comparisons typically exclude them so a Monday running a Sunday
    schedule does not dilute the Weekday expectation -- mirroring how vendor
    ridership and ridecheck exports usually treat holidays. Only meaningful
    for feeds that define a weekly baseline in ``calendar.txt``; for feeds
    built purely from ``calendar_dates.txt``, every service day is an
    "exception" and callers should skip the exclusion instead.

    Args:
        calendar_dates: Parsed ``calendar_dates.txt`` with string columns, or
            ``None`` / empty.
        start_date: Optional inclusive lower bound on the returned dates.
        end_date: Optional inclusive upper bound on the returned dates.

    Returns:
        Set of normalized ``pd.Timestamp`` dates having at least one exception
        row of either type (possibly empty).
    """
    if calendar_dates is None or calendar_dates.empty or "date" not in calendar_dates.columns:
        return set()
    parsed = pd.to_datetime(calendar_dates["date"], format="%Y%m%d", errors="coerce").dropna()
    if start_date is not None:
        parsed = parsed.loc[parsed >= start_date.normalize()]
    if end_date is not None:
        parsed = parsed.loc[parsed <= end_date.normalize()]
    return set(parsed)


def load_id_set(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
    *,
    kind: str = "id",
) -> set[str]:
    """Union an inline list and an optional text file of ids into one set.

    Used to resolve override lists (express routes, express origin stops, …) that
    a caller may supply inline, in an external file, or both — without repeating
    the parsing for each one.

    Args:
        inline_ids: Id values supplied directly (e.g. a config list). ``None`` is
            treated as empty.
        txt_path: Path to a text file with one id per line. Blank lines are
            skipped and ``#`` starts a comment (whole-line or inline). ``None``
            skips the file. A path that is set but missing is logged as a warning
            and skipped — the inline ids are still returned.
        kind: Human-readable noun used only in log messages (e.g.
            ``"express route"``, ``"express origin stop"``).

    Returns:
        The unioned set of id strings (possibly empty). Every id is coerced to a
        trimmed ``str`` so it matches GTFS values, which are read as strings.
    """
    ids: set[str] = set()

    for raw in inline_ids or ():
        text = str(raw).strip()
        if text:
            ids.add(text)

    if txt_path:
        if not os.path.exists(txt_path):
            logging.warning(
                "%s file '%s' not found; using inline ids only.", kind.capitalize(), txt_path
            )
        else:
            with open(txt_path, encoding="utf-8") as handle:
                for line in handle:
                    text = line.split("#", 1)[0].strip()
                    if text:
                        ids.add(text)
            logging.info("Loaded %s ids from '%s'.", kind, txt_path)

    logging.info("Resolved %d %s id(s).", len(ids), kind)
    return ids


def load_express_route_ids(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
) -> set[str]:
    """Resolve the set of express-route ``route_id`` values (see ``load_id_set``).

    Thin wrapper kept for readable call sites and backwards compatibility.
    """
    return load_id_set(inline_ids, txt_path, kind="express route")
