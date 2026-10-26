"""Run all four runtime-analysis stages from one self-contained Python script.

Edit USER SETTINGS below, then run this file with Python 3.9 or later.
Dependencies: pandas, numpy, openpyxl. No other workflow .py files are needed.

Stages:
    1. GTFS scheduled movement, holding, total runtime, and known layover.
    2. CLEVER observed whole-trip runtime compared with its own schedule.
    3. CLEVER observed timepoint intervals compared with scheduled intervals.
    4. Percentile-based segment recommendations for manual schedule drafting.

Select stages with STAGES below, or at the command line:
    python runtime_workflow.py --stages 1
    python runtime_workflow.py --stages 2 3 4

In an ArcGIS Pro notebook:
    %run "C:/path/to/runtime_workflow.py"

Outputs are unchanged: one workbook per route/direction, plus run.log and
settings.json, under OUTPUT_DIR/runtime_workflow/run_<timestamp>/.
Stage 1 needs only GTFS; Stage 4 calculates Stage 3 statistics internally.
Recommendations include intermediate destination dwell/holding. Do not add
Stage 1 holding again. Segment percentiles are not an OTP guarantee.
Planner periods, budget constraints, and revised layovers remain manual.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import re
import sys
import tempfile
import textwrap
import threading
import time
import zipfile
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    cast,
)

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.workbook.properties import CalcProperties

# =============================================================================
# USER SETTINGS
# =============================================================================

PROJECT_DIR = r"Path\To\Your\Project_Folder"
GTFS_PATH = r"Path\To\Your\GTFS_Folder"  # folder of .txt files or a .zip
GTFS_SERVICE_DATE = "2025-11-10"
RUNTIME_PATH = PROJECT_DIR + r"\raw_data\runtime_prepared.csv"
TIMEPOINT_PATH = PROJECT_DIR + r"\output_clean_data\timepoint_prepared.xlsx"
OUTPUT_DIR = PROJECT_DIR + r"\output"
ROUTES = ("798", "983", "952", "RIBS4")  # RIB4 is normalized to RIBS4 in every input.
STAGES = (1, 2, 3, 4)  # Set (1,) to export GTFS only; no CLEVER files are needed.
DATE_START = ""
DATE_END = ""
DAYS_OF_WEEK = (0, 1, 2, 3, 4)  # Monday=0. Different day types are kept separate.
EXCLUDE_FEDERAL_HOLIDAYS = True
EXCLUDE_DATES = ()
TRIM_EACH_TAIL = 0.05  # floor(n * fraction) removed from each tail of a trip sample.
RECOMMEND_PERCENTILE = 0.85
ROUND_UP_TO_MINUTES = 1.0
MIN_OBSERVATIONS = 10  # Below this, show statistics but leave recommendation blank.
GTFS_MATCH_TOLERANCE_MIN = 1.0  # Optional segment schedule enrichment; not an exclusion gate.
CSV_ENCODING = "auto"
PREPARED_SHEET = 0
CACHE_PREPARED_EXCEL = True


@dataclass(frozen=True)
class Settings:
    """Settings passed explicitly to each stage; defaults come from the block above."""

    gtfs_path: Path = Path(GTFS_PATH)
    gtfs_service_date: str = GTFS_SERVICE_DATE
    runtime_path: Path = Path(RUNTIME_PATH)
    timepoint_path: Path = Path(TIMEPOINT_PATH)
    output_dir: Path = Path(OUTPUT_DIR)
    routes: tuple[str, ...] = ROUTES
    stages: tuple[int, ...] = STAGES
    date_start: str = DATE_START
    date_end: str = DATE_END
    days_of_week: tuple[int, ...] = DAYS_OF_WEEK
    exclude_holidays: bool = EXCLUDE_FEDERAL_HOLIDAYS
    exclude_dates: tuple[str, ...] = EXCLUDE_DATES
    trim: float = TRIM_EACH_TAIL
    percentile: float = RECOMMEND_PERCENTILE
    rounding: float = ROUND_UP_TO_MINUTES
    min_observations: int = MIN_OBSERVATIONS
    match_tolerance: float = GTFS_MATCH_TOLERANCE_MIN


# =============================================================================
# INPUT READING, TIME HANDLING, AND STATISTICS
# =============================================================================

PROGRESS_INTERVAL_SECONDS = 20.0
PROGRESS_EVERY_ITEMS = 1000


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


def elapsed_label(seconds: float) -> str:
    """Format a duration for progress messages."""
    total = max(0, int(seconds))
    minutes, second = divmod(total, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:d}:{minute:02d}:{second:02d}" if hours else f"{minute:d}:{second:02d}"


@contextmanager
def log_stage(label: str, heartbeat: bool = False) -> Generator[None, None, None]:
    """Log stage start/end and optionally report a pending operation periodically.

    A heartbeat describes elapsed waiting time, not measured completion. The
    daemon thread performs logging only and is stopped on success or failure.
    """
    started = time.monotonic()
    stop = threading.Event()
    worker = None
    logging.info("%s | starting", label)

    def report_wait() -> None:
        while not stop.wait(max(float(PROGRESS_INTERVAL_SECONDS), 0.01)):
            logging.info(
                "%s | operation still pending; elapsed %s (completion count unavailable)",
                label,
                elapsed_label(time.monotonic() - started),
            )

    if heartbeat:
        worker = threading.Thread(target=report_wait, name="schedule-progress", daemon=True)
        worker.start()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=1.0)
        logging.info(
            "%s | %s after %s",
            label,
            "finished" if succeeded else "stopped before completion",
            elapsed_label(time.monotonic() - started),
        )


def timed_stage(label: str, heartbeat: bool = False) -> Callable:
    """Decorate a calculation with stage timing without changing its return value."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with log_stage(label, heartbeat=heartbeat):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def progress_items(
    items: Iterable, label: str, total: int, detail: Callable[[], str] | None = None
) -> Iterator:
    """Yield items unchanged and report work completed by the consuming loop.

    Counts advance only after control returns from the loop body. Rejected
    records still count as processed; a detail callback can report retained
    versus rejected counts. ETA applies only to this loop and is approximate.
    An interrupted or failed loop never reports successful completion.
    """
    started = last_report = time.monotonic()
    done = last_count = 0
    logging.info("%s | 0/%s processed", label, f"{total:,}")
    for item in items:
        yield item
        done += 1
        now = time.monotonic()
        due = (
            now - last_report >= PROGRESS_INTERVAL_SECONDS
            or done - last_count >= PROGRESS_EVERY_ITEMS
        )
        if due and done < total:
            elapsed = now - started
            remaining = elapsed / done * max(total - done, 0)
            logging.info(
                "%s | %s/%s (%.1f%%) | elapsed %s | this stage ETA ~%s%s",
                label,
                f"{done:,}",
                f"{total:,}",
                done / total * 100 if total else 100,
                elapsed_label(elapsed),
                elapsed_label(remaining),
                f" | {detail()}" if detail else "",
            )
            last_report, last_count = now, done
    logging.info(
        "%s | %s/%s processed | finished in %s%s",
        label,
        f"{done:,}",
        f"{total:,}",
        elapsed_label(time.monotonic() - started),
        f" | {detail()}" if detail else "",
    )


def _check_columns(df: pd.DataFrame, required: Sequence[str], label: str, path: Path) -> None:
    """Raise a readable ValueError when *df* lacks any *required* column."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} '{path}' is missing required column(s): {missing}. "
            f"Expected at least {list(required)}."
        )


def parse_datetimes(values: pd.Series) -> pd.Series:
    """Parse common timestamp formats in bulk, with a cached unusual-value fallback.

    Explicit formats avoid repeated format inference and work with pandas 1.4.
    Mixed US and ISO dates are supported. Unusual values are parsed once per
    unique token. Timezone-aware values remain unsupported and raise an error.
    """
    text = values.fillna("").astype(str).str.strip()
    normalized = text.str.replace("T", " ", regex=False)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    date_formats = (
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{1,2}/\d{1,2}/\d{4}", "%m/%d/%Y"),
    )
    time_formats = (
        ("", ""),
        (r" \d{1,2}:\d{2}", " %H:%M"),
        (r" \d{1,2}:\d{2}:\d{2}", " %H:%M:%S"),
        (r" \d{1,2}:\d{2}:\d{2}\.\d+", " %H:%M:%S.%f"),
        (r" \d{1,2}:\d{2} [APap][Mm]", " %I:%M %p"),
        (r" \d{1,2}:\d{2}:\d{2} [APap][Mm]", " %I:%M:%S %p"),
    )
    for date_pattern, date_format in date_formats:
        for time_pattern, time_format in time_formats:
            mask = normalized.str.fullmatch(date_pattern + time_pattern)
            if mask.any():
                result.loc[mask] = pd.to_datetime(
                    normalized.loc[mask], format=date_format + time_format, errors="coerce"
                )
    missing = result.isna() & text.ne("")
    unusual = text.loc[missing]
    parsed = {}
    unique = unusual.unique()
    tokens = (
        progress_items(unique, "Parsing unusual timestamp formats", len(unique))
        if len(unique) >= PROGRESS_EVERY_ITEMS
        else unique
    )
    for value in tokens:
        stamp = pd.to_datetime(value, errors="coerce")
        if pd.notna(stamp) and stamp.tzinfo is not None:
            raise ValueError("Use local timestamps without timezone offsets in both input files.")
        parsed[value] = stamp
    if len(unusual):
        result.loc[missing] = pd.to_datetime(unusual.map(parsed), errors="coerce")
    return result


@timed_stage("Reading CSV", heartbeat=True)
def read_csv_text(path: Path) -> pd.DataFrame:
    """Read text-valued CSV fields with strict UTF-8/Windows export handling.

    Automatic mode retries Windows-1252 only after a UnicodeDecodeError from
    UTF-8. Parsing, permission and file-not-found errors propagate unchanged.
    Literal IDs such as NA and leading zeroes are preserved in either encoding.
    """
    logging.info("Opening CSV: %s", path)
    encoding = CSV_ENCODING.strip()
    if encoding.lower() != "auto":
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
        except (LookupError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Could not decode CSV '{path}' using CSV_ENCODING={encoding!r}: {exc}"
            ) from exc
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        logging.warning("CSV '%s' is not UTF-8; retrying with Windows-1252.", path)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp1252")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"CSV '{path}' cannot be decoded as UTF-8 or Windows-1252. "
            "Set CSV_ENCODING to the source encoding or re-export the CSV as UTF-8."
        ) from exc
    logging.info("Read CSV '%s' using Windows-1252.", path)
    return frame


def _excel_cache_key(path: Path, sheet: str | int) -> dict[str, Any]:
    """Identify an Excel source by path, sheet, byte count, and modification time."""
    stat = path.stat()
    return {
        "version": 1,
        "path": str(path.resolve()),
        "sheet": sheet,
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _read_excel_with_cache(path: Path, sheet: str | int, cache_dir: Path | None) -> pd.DataFrame:
    """Read Excel once; reuse an integrity-checked, non-executable CSV cache.

    An unreadable or stale cache is ignored. Atomic replacement and a content
    digest prevent partially written cache files from being used. Cache write
    failure is logged but does not discard a successfully read workbook.
    """
    source = _excel_cache_key(path, sheet)
    cache_file = metadata_file = None
    if CACHE_PREPARED_EXCEL and cache_dir is not None:
        identity = json.dumps([source["path"], sheet], ensure_ascii=False)
        stem = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        cache_file = cache_dir / f"{stem}.csv.gz"
        metadata_file = cache_dir / f"{stem}.json"
        if cache_file.is_file() and metadata_file.is_file():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                if metadata.get("source") == source:
                    digest = hashlib.sha256(cache_file.read_bytes()).hexdigest()
                    if digest != metadata.get("sha256"):
                        raise ValueError("Cache content hash does not match its metadata.")
                    with log_stage(f"Reading cached Excel data for {path.name}", heartbeat=True):
                        frame = pd.read_csv(
                            cache_file,
                            dtype=str,
                            keep_default_na=False,
                            encoding="utf-8",
                            compression="gzip",
                        )
                    if list(frame.columns) != metadata.get("columns"):
                        raise ValueError("Cache columns do not match its metadata.")
                    if len(frame) != metadata.get("rows"):
                        raise ValueError("Cache row count does not match its metadata.")
                    # Recheck after the read in case another process changed the workbook.
                    if _excel_cache_key(path, sheet) == source:
                        logging.info("Reused unchanged Excel input: %s", path)
                        return frame
                logging.info("Excel input changed; rebuilding cache for %s.", path.name)
            except (OSError, ValueError, EOFError) as exc:
                logging.warning("Ignoring unusable Excel cache for %s: %s", path.name, exc)
    with log_stage(f"Reading Excel {path.name} (sheet {sheet!r})", heartbeat=True):
        frame = pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False)
    if cache_file is not None and metadata_file is not None and cache_dir is not None:
        temporary: list[Path] = []
        try:
            if _excel_cache_key(path, sheet) != source:
                logging.warning("Excel input changed during reading; cache was not saved: %s", path)
                return frame
            cache_dir.mkdir(parents=True, exist_ok=True)
            for suffix in (".csv.gz", ".json"):
                with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=suffix, delete=False) as tmp:
                    temporary.append(Path(tmp.name))
            frame.to_csv(temporary[0], index=False, encoding="utf-8", compression="gzip")
            metadata = {
                "source": source,
                "rows": len(frame),
                "columns": list(frame.columns),
                "sha256": hashlib.sha256(temporary[0].read_bytes()).hexdigest(),
            }
            temporary[1].write_text(json.dumps(metadata), encoding="utf-8")
            temporary[0].replace(cache_file)
            temporary[1].replace(metadata_file)
            logging.info("Cached %s for faster subsequent runs.", path.name)
        except (OSError, ValueError) as exc:
            logging.warning("Could not save Excel cache for %s: %s", path.name, exc)
        finally:
            for temp_path in temporary:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
    return frame


def _prepared_table(path: Path, sheet: str | int, cache_dir: Path | None = None) -> pd.DataFrame:
    """Read a prepared CSV/XLSX; preserve text IDs and discard empty padding."""
    if path.suffix.lower() == ".csv":
        frame = read_csv_text(path)
    elif path.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            frame = _read_excel_with_cache(path, sheet, cache_dir)
        except ImportError as exc:
            raise ValueError(
                "Excel inputs require openpyxl; alternatively export the sheet to CSV."
            ) from exc
    else:
        raise ValueError(f"Prepared input must be CSV or XLSX/XLSM: {path}")
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate column names after trimming/lowercasing in {path}.")
    frame = frame.apply(lambda col: col.fillna("").astype(str).str.strip())
    empty = frame.eq("").all(axis=1)
    logging.info(
        "%s: %d populated rows; discarded %d blank rows.",
        path.name,
        (~empty).sum(),
        empty.sum(),
    )
    return frame.loc[~empty].copy()


def _prepared_times(
    values: pd.Series,
    service_dates: pd.Series,
    anchors: pd.Series | None = None,
    progress_label: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Read dated timestamps or clocks, including GTFS hours above 24.

    Ordinary clock-only values are aligned to the nearest day of an optional
    dated anchor. Explicit dates and hours >= 24 are never shifted.
    """
    label = progress_label or "Parsing prepared timestamps"
    with log_stage(f"{label} ({len(values):,} rows; bulk parse)", heartbeat=True):
        text = values.fillna("").astype(str).str.strip()
        parts = text.str.extract(r"^(\d{1,3}):(\d{2})(?::(\d{2}(?:\.\d+)?))?$")
        is_clock = parts[0].notna()
        hour = pd.to_numeric(parts[0], errors="coerce")
        minute = pd.to_numeric(parts[1], errors="coerce")
        second = pd.to_numeric(parts[2].fillna("0"), errors="coerce")
        clock_only = (is_clock & (hour < 24)).astype(bool)
        valid = is_clock & (minute < 60) & (second < 60)
        result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        if valid.any():
            seconds = hour.loc[valid] * 3600 + minute.loc[valid] * 60 + second.loc[valid]
            result.loc[valid] = service_dates.loc[valid] + pd.to_timedelta(seconds, unit="s")
        if (~is_clock).any():
            result.loc[~is_clock] = parse_datetimes(text.loc[~is_clock])
        if anchors is not None:
            align = clock_only & result.notna() & anchors.notna()
            difference = (anchors.loc[align] - result.loc[align]).dt.total_seconds() / 86400
            result.loc[align] += pd.to_timedelta(np.rint(difference), unit="D")
        return result, clock_only


def route_label(value: Any) -> str:
    """Normalize text route labels and the RIB4/RIBS4 spelling difference."""
    text = str(value).strip().upper()
    return "RIBS4" if text == "RIB4" else text


def load_clever(path: Path, cfg: Settings, required: Sequence[str]) -> pd.DataFrame:
    """Read a prepared CLEVER export and select routes, dates and day types."""
    frame = _prepared_table(path, PREPARED_SHEET, cfg.output_dir / "input_cache")
    _check_columns(frame, ["date", "route", "direction", "trip_id", *required], "CLEVER", path)
    frame["route"] = frame.route.map(route_label)
    frame = frame.loc[frame.route.isin({route_label(r) for r in cfg.routes})].copy()
    frame["direction"] = frame.direction.str.upper().str.strip()
    frame["date"] = parse_datetimes(frame.date).dt.normalize()
    bad = frame.date.isna() | frame.trip_id.eq("") | frame.direction.eq("")
    if bad.any():
        logging.warning(
            "%s: excluded %d rows with invalid dates or blank trip/direction keys.",
            path.name,
            int(bad.sum()),
        )
    frame = frame.loc[~bad].copy()
    if cfg.date_start:
        frame = frame.loc[frame.date.ge(pd.Timestamp(cfg.date_start))]
    if cfg.date_end:
        frame = frame.loc[frame.date.le(pd.Timestamp(cfg.date_end))]
    frame = frame.loc[frame.date.dt.dayofweek.isin(cfg.days_of_week)].copy()
    holidays = {pd.Timestamp(d).date() for d in cfg.exclude_dates}
    if cfg.exclude_holidays and not frame.empty:
        for year in range(frame.date.dt.year.min(), frame.date.dt.year.max() + 2):
            holidays |= federal_holidays_observed(year)
    frame = frame.loc[~frame.date.dt.date.isin(holidays)].copy()
    frame["day_type"] = np.where(
        frame.date.dt.dayofweek.lt(5),
        "Weekday",
        np.where(frame.date.dt.dayofweek.eq(5), "Saturday", "Sunday"),
    )
    frame = frame.drop_duplicates().reset_index(drop=True)
    logging.info("%s: selected %d rows for %s.", path.name, len(frame), ", ".join(cfg.routes))
    return frame


def parse_field(frame: pd.DataFrame, name: str, anchor: pd.Series | None = None) -> pd.Series:
    """Parse an optional timestamp column; absent events remain missing."""
    values = frame.get(name, pd.Series("", index=frame.index))
    return _prepared_times(values, frame.date, anchor, name)[0]


def minutes_between(later: Any, earlier: Any) -> float:
    """Return elapsed minutes, or NaN if either endpoint is unknown."""
    return (
        (later - earlier).total_seconds() / 60 if pd.notna(later) and pd.notna(earlier) else np.nan
    )


def service_minute(value: Any, date: Any) -> float:
    """Measure a timestamp relative to its service date, retaining hours above 24."""
    return minutes_between(value, date)


def trim_sample(group: pd.DataFrame, column: str, fraction: float) -> pd.DataFrame:
    """Trim whole observations deterministically, rounding each tail count down."""
    valid = group.loc[np.isfinite(group[column]) & group[column].ge(0)]
    ordered = valid.sort_values([column, "date", "trip_id"], kind="mergesort")
    count = math.floor(len(ordered) * fraction)
    return ordered.iloc[count : len(ordered) - count].copy() if count else ordered.copy()


def describe(values: pd.Series, percentile: float = 0.85) -> dict[str, Any]:
    """Summarize numeric observations; empty samples produce blanks, never zero runtimes."""
    values = pd.to_numeric(values, errors="coerce")
    values = values.loc[np.isfinite(values)]
    if values.empty:
        return {
            "n": 0,
            **{key: np.nan for key in ("mean", "median", "p85", "p95", "target", "min", "max")},
        }
    return {
        "n": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "p85": values.quantile(0.85),
        "p95": values.quantile(0.95),
        "target": values.quantile(percentile),
        "min": values.min(),
        "max": values.max(),
    }


def distinct_daily_rows(
    frame: pd.DataFrame, keys: Sequence[str], values: Sequence[str]
) -> pd.DataFrame:
    """Avoid counting duplicated exports; exclude conflicting records on the same daily key."""
    out = frame.drop_duplicates([*keys, *values]).copy()
    conflicts = out.duplicated(list(keys), keep=False)
    if conflicts.any():
        logging.warning(
            "Excluded %d conflicting duplicate rows on keys %s.",
            int(conflicts.sum()),
            ", ".join(keys),
        )
    return out.loc[~conflicts].copy()


# =============================================================================
# EXCEL OUTPUT
# =============================================================================


class Sheet(NamedTuple):
    """A worksheet specification shared by the four independent stages."""

    name: str
    columns: list[str]
    rows: list[list[Any]]


def column_letter(index: int) -> str:
    """Return Excel's one-based column letter for readable total/change formulas."""
    return get_column_letter(index)


def from_frame(name: str, frame: pd.DataFrame) -> Sheet:
    """Convert a summary table to a worksheet specification."""
    return Sheet(
        name, list(frame.columns), [list(row) for row in frame.itertuples(index=False, name=None)]
    )


def write_book(path: Path, sheets: list[Sheet]) -> None:
    """Write formulas as formulas and all source IDs/names as literal text."""
    book = Workbook()
    book.remove(book.active)
    book.calculation = CalcProperties(calcId=0, fullCalcOnLoad=True)
    used = set()
    for spec in sheets:
        base = re.sub(r"[\\/*?:\[\]]", "_", spec.name)[:31] or "Sheet"
        name = base
        suffix = 1
        while name in used:
            suffix += 1
            name = base[:26] + "_" + str(suffix)
        used.add(name)
        ws = book.create_sheet(name)
        ws.append(spec.columns)
        for values in spec.rows:
            ws.append(
                [
                    None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v
                    for v in values
                ]
            )
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[1].height = 60
        ws.freeze_panes = "D2" if len(spec.columns) > 4 else "A2"
        ws.auto_filter.ref = ws.dimensions
        for index, label in enumerate(spec.columns, 1):
            ws.column_dimensions[get_column_letter(index)].width = (
                95
                if label == "Explanation"
                else 30
                if label in ("From", "To", "Basis")
                else 40
                if label == "Review note"
                else 24
                if "\n" in label
                else 18
            )
            for (cell,) in ws.iter_rows(min_row=2, min_col=index, max_col=index):
                if isinstance(cell.value, str):
                    # Only formulas generated by this script use this wrapper.
                    cell.data_type = "s"
                    if cell.value.startswith("__FORMULA__="):
                        cell.value = cell.value[len("__FORMULA__") :]
                        cell.data_type = "f"
                elif isinstance(cell.value, (float, int)):
                    cell.number_format = "0.0" if isinstance(cell.value, float) else "0"
                if label in ("From", "To", "Basis", "Review note"):
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    width = 38 if label == "Review note" else 28
                    lines = max(1, len(textwrap.wrap(str(cell.value or ""), width=width)))
                    ws.row_dimensions[cell.row].height = max(
                        ws.row_dimensions[cell.row].height or 15, 15 * lines
                    )
        if spec.name == "Notes":
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                ws.row_dimensions[row[0].row].height = 48
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    book.close()


def formula(expression: str) -> str:
    """Mark a locally generated formula; source strings never become executable formulas."""
    return "__FORMULA__" + expression


# =============================================================================
# STAGE 1: GTFS SCHEDULED RUNTIMES
# =============================================================================


def read_table(path: Path, name: str, required: bool = True) -> pd.DataFrame:
    """Read GTFS text without losing leading zeroes or the literal identifier NA."""
    if path.is_dir():
        matches = list(path.rglob(name))
        if len(matches) == 1:
            return pd.read_csv(matches[0], dtype=str, keep_default_na=False)
    else:
        with zipfile.ZipFile(path) as archive:
            matches = [n for n in archive.namelist() if Path(n).name == name]
            if len(matches) == 1:
                with archive.open(matches[0]) as stream:
                    return pd.read_csv(
                        stream, dtype=str, keep_default_na=False, encoding="utf-8-sig"
                    )
    if len(matches) > 1:
        raise ValueError(f"Multiple {name} files in {path}.")
    if required:
        raise ValueError(f"Missing {name} in {path}.")
    return pd.DataFrame()


def clock(value: str) -> float:
    """Convert a GTFS clock to minutes, including seconds and hours greater than 24."""
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})", str(value))
    if not match:
        return np.nan
    hour, minute, second = map(int, match.groups())
    return hour * 60 + minute + second / 60 if minute < 60 and second < 60 else np.nan


def load_schedule(cfg: Settings) -> dict[str, pd.DataFrame]:
    """Build the reference timetable and full-system block order for one service date."""
    tables = {
        name: read_table(
            cfg.gtfs_path, name + ".txt", name not in ("calendar", "calendar_dates", "frequencies")
        )
        for name in (
            "routes",
            "trips",
            "stops",
            "stop_times",
            "calendar",
            "calendar_dates",
            "frequencies",
        )
    }
    date = pd.Timestamp(cfg.gtfs_service_date)
    if not isinstance(date, pd.Timestamp):
        raise ValueError("GTFS_SERVICE_DATE must be a valid reference date.")
    token, weekday = date.strftime("%Y%m%d"), date.strftime("%A").lower()
    active = set()
    cal, exceptions = tables["calendar"], tables["calendar_dates"]
    if not cal.empty:
        active.update(
            cal.loc[
                cal[weekday].eq("1") & cal.start_date.le(token) & cal.end_date.ge(token),
                "service_id",
            ]
        )
    if not exceptions.empty:
        ex = exceptions.loc[exceptions.date.eq(token)]
        active.difference_update(ex.loc[ex.exception_type.eq("2"), "service_id"])
        active.update(ex.loc[ex.exception_type.eq("1"), "service_id"])
    trips = tables["trips"].loc[tables["trips"].service_id.isin(active)].copy()
    if trips.trip_id.duplicated().any():
        raise ValueError("Duplicate active GTFS trip IDs.")
    routes = tables["routes"].copy()
    routes["label"] = routes.route_short_name.where(routes.route_short_name.ne(""), routes.route_id)
    routes["label"] = routes.label.map(route_label)
    # Accept either an agency route_id or its public route_short_name.
    requested = {route_label(r) for r in cfg.routes}
    selected_ids = {}
    for label in requested:
        selected = routes.loc[routes.label.eq(label) | routes.route_id.map(route_label).eq(label)]
        if len(selected) != 1:
            raise ValueError(f"Route {label} must identify exactly one GTFS route.")
        selected_ids[selected.route_id.iloc[0]] = label
    trips["route"] = trips.route_id.map(routes.set_index("route_id").label)
    trips.loc[trips.route_id.isin(selected_ids), "route"] = trips.route_id.map(selected_ids)
    trips["direction"] = trips.get("direction_id", pd.Series("", index=trips.index))
    trips["block"] = trips.get("block_id", pd.Series("", index=trips.index))
    trips["headsign"] = trips.get("trip_headsign", pd.Series("", index=trips.index))
    frequency_ids = set(tables["frequencies"].trip_id) if not tables["frequencies"].empty else set()
    if trips.loc[trips.route.isin(requested), "trip_id"].isin(frequency_ids).any():
        raise ValueError("Selected routes use frequencies.txt; expand those trips before export.")
    times = tables["stop_times"].loc[tables["stop_times"].trip_id.isin(trips.trip_id)].copy()
    times["sequence"] = pd.to_numeric(times.stop_sequence, errors="coerce")
    invalid_ids = set(
        times.loc[
            times.sequence.isna() | times.duplicated(["trip_id", "sequence"], keep=False), "trip_id"
        ]
    )
    times = times.sort_values(["trip_id", "sequence"])
    for event in ("arrival", "departure"):
        times[event] = times[event + "_time"].map(clock)
    first, last = times.drop_duplicates("trip_id"), times.drop_duplicates("trip_id", keep="last")
    first, last = first.set_index("trip_id"), last.set_index("trip_id")
    trips["start"] = trips.trip_id.map(first.departure)
    trips["end"] = trips.trip_id.map(last.arrival)
    trips["first_stop"] = trips.trip_id.map(first.stop_id)
    trips["last_stop"] = trips.trip_id.map(last.stop_id)
    trips["runtime"] = trips.end - trips.start
    trips["layover"] = np.nan
    trips["layover_status"] = np.where(trips.block.eq(""), "No block", "Last trip in block")
    trips["bad"] = (
        ~np.isfinite(trips[["start", "end"]]).all(axis=1)
        | trips.runtime.le(0)
        | trips.trip_id.isin(invalid_ids | frequency_ids)
    )
    for _, group in trips.loc[trips.block.ne("")].groupby("block", sort=False):
        group = group.sort_values(["start", "trip_id"])
        if group.bad.any() or group.start.lt(group.end.cummax().shift()).any():
            trips.loc[group.index, "layover_status"] = "Invalid / overlapping / frequency block"
            continue
        records: list[Any] = list(group.itertuples())
        for current, nxt in zip(records, records[1:]):
            if current.last_stop == nxt.first_stop:
                trips.loc[current.Index, "layover"] = nxt.start - current.end
                trips.loc[current.Index, "layover_status"] = "Same-stop scheduled gap"
            else:
                trips.loc[current.Index, "layover_status"] = (
                    "Different stops: deadhead split unknown"
                )
    trips = trips.loc[trips.route.isin(requested)].copy()
    if trips.bad.any():
        raise ValueError("Selected GTFS trips have invalid endpoints or stop sequences.")
    times = times.loc[times.trip_id.isin(trips.trip_id)].copy()
    edges = times.groupby("trip_id").cumcount().eq(0) | times.groupby("trip_id").cumcount(
        ascending=False
    ).eq(0)
    exact = times.get("timepoint", pd.Series("1", index=times.index)).isin(["1", ""])
    # Retain a scheduled dwell stop even if it is not a GTFS timepoint.
    times = times.loc[edges | exact | times.departure.gt(times.arrival)].copy()
    times["holding"] = times.departure - times.arrival
    if (
        (~np.isfinite(times[["arrival", "departure"]]).all(axis=1)).any()
        or times.holding.lt(0).any()
        or times.arrival.lt(times.groupby("trip_id").departure.shift()).any()
    ):
        raise ValueError("GTFS timepoints have missing, negative or nonmonotonic times.")
    times["seq"] = times.groupby("trip_id").cumcount() + 1
    times["count"] = times.groupby("trip_id").stop_id.transform("size")
    stops = tables["stops"].copy()
    stops["stop_code"] = stops.get("stop_code", pd.Series("", index=stops.index))
    times = times.merge(
        stops[["stop_id", "stop_code", "stop_name"]],
        on="stop_id",
        how="left",
        validate="many_to_one",
    )
    if times.stop_name.isna().any():
        raise ValueError("GTFS stop_times references an unknown stop.")
    paths = times.groupby("trip_id").stop_id.agg(tuple)
    trips["point_ids"] = trips.trip_id.map(paths)
    if trips.point_ids.isna().any() or trips.point_ids.map(len).lt(2).any():
        raise ValueError("Every selected trip needs at least two scheduled timepoints.")
    trips["pattern"] = ""
    for _, group in trips.groupby(["route", "direction"]):
        patterns = {path: "P" + str(i + 1) for i, path in enumerate(sorted(set(group.point_ids)))}
        trips.loc[group.index, "pattern"] = group.point_ids.map(patterns.__getitem__)
    logging.info("GTFS loaded %d scheduled trips for %s.", len(trips), cfg.gtfs_service_date)
    return {"trips": trips.reset_index(drop=True), "points": times.reset_index(drop=True)}


def build_gtfs_sheets(schedule: dict[str, pd.DataFrame], route: str) -> list[Sheet]:
    """Place holding immediately after its incoming movement; total and layover are last."""
    trips = schedule["trips"].loc[schedule["trips"].route.eq(route)]
    point_groups = {
        key: group.sort_values("seq") for key, group in schedule["points"].groupby("trip_id")
    }
    sheets = []
    for (direction, pattern), group in trips.groupby(["direction", "pattern"], sort=True):
        example = point_groups[group.trip_id.iloc[0]].to_dict("records")
        hold_sequences = {
            int(sequence)
            for trip_id in group.trip_id
            for sequence, count, holding in point_groups[trip_id][
                ["seq", "count", "holding"]
            ].itertuples(index=False, name=None)
            if 1 < sequence < count and holding > 0
        }
        columns = ["Block", "GTFS trip", "Start"]
        for previous, point in zip(example, example[1:]):
            columns.append(previous["stop_name"].strip() + " →\n" + point["stop_name"].strip())
            if point["seq"] in hold_sequences:
                columns.append("Holding\n" + point["stop_name"].strip())
        total_column = len(columns) + 1
        columns += ["Total runtime min", "Layover min"]
        rows = []
        for excel_row, trip in enumerate(group.sort_values(["start", "trip_id"]).itertuples(), 2):
            points = point_groups[trip.trip_id].to_dict("records")
            values: list[Any] = [trip.block, trip.trip_id, minutes_to_hhmm(trip.start)]
            for previous, point in zip(points, points[1:]):
                values.append(point["arrival"] - previous["departure"])
                if point["seq"] in hold_sequences:
                    values.append(point["holding"])
            if not np.isclose(sum(values[3:]), trip.runtime):
                raise ValueError(f"GTFS movement/holding does not reconcile for {trip.trip_id}.")
            values += [
                formula(f"=SUM(D{excel_row}:{column_letter(total_column - 1)}{excel_row})"),
                trip.layover,
            ]
            rows.append(values)
        sheets.append(Sheet(f"1_GTFS_D{direction or 'NA'}_{pattern}", columns, rows))
    return sheets


def match_points(
    schedule: dict[str, pd.DataFrame] | None,
    route: str,
    scheduled: list[float],
    source_codes: list[str],
    tolerance: float,
    require_terminal: bool = True,
) -> pd.DataFrame | None:
    """Optionally enrich one CLEVER trip using a unique timetable/ordered-stop match.

    Source direction labels and repeated source codes are not global constraints.
    A failed or ambiguous match returns None; CLEVER observations remain usable.
    """
    if schedule is None or len(scheduled) < 2 or not np.isfinite(scheduled).all():
        return None
    roster = schedule["trips"]
    known_codes = set(schedule["points"].stop_id) | set(schedule["points"].stop_code)
    known_codes.discard("")
    candidates = roster.loc[
        roster.route.eq(route) & roster.start.sub(scheduled[0]).abs().le(tolerance)
    ]
    if require_terminal:
        candidates = candidates.loc[candidates.end.sub(scheduled[-1]).abs().le(tolerance)]
    solutions = []
    for trip in candidates.itertuples():
        target = schedule["points"].loc[schedule["points"].trip_id.eq(trip.trip_id)].copy()
        paths = [[]]
        for i, (minute, code) in enumerate(zip(scheduled, source_codes)):
            allowed = target.arrival.sub(minute).abs().le(tolerance) | target.departure.sub(
                minute
            ).abs().le(tolerance)
            direct = target.stop_id.eq(code) | target.stop_code.eq(code)
            if code in known_codes:
                allowed &= direct
            if i == 0:
                allowed &= target.seq.eq(1)
            elif i == len(scheduled) - 1 and require_terminal:
                allowed &= target.seq.eq(target["count"])
            positions = np.flatnonzero(allowed.to_numpy())
            paths = [
                path + [int(j)] for path in paths for j in positions if not path or j > path[-1]
            ]
            if len(paths) > 100 or not paths:
                paths = []
                break
        for path in paths:
            # Preserve GTFS sequence numbers so missing rows cannot create a new pattern.
            matched = target.iloc[path].copy().reset_index(drop=True)
            matched["gtfs_direction"] = trip.direction
            solutions.append(matched)
    return solutions[0] if len(solutions) == 1 else None


# =============================================================================
# STAGE 2: OBSERVED WHOLE-TRIP RUNTIMES
# =============================================================================

TRIP_VARIANT = ["route", "direction", "day_type", "start", "end"]


def load_runtime(cfg: Settings) -> pd.DataFrame:
    """Read runtime exports without requiring a GTFS or timepoint-file join."""
    names = [
        f"trip_{kind}_{end}_time_only"
        for kind in ("scheduled", "actual")
        for end in ("start", "finish")
    ]
    frame = load_clever(cfg.runtime_path, cfg, names)
    frame["scheduled_start"] = parse_field(frame, names[0])
    frame["scheduled_end"] = parse_field(frame, names[1], frame.scheduled_start)
    # A clock-only finish can cross midnight; explicit dated values are not moved.
    clock_only = frame[names[1]].str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?")
    rollover = clock_only & frame.scheduled_start.sub(frame.scheduled_end).gt(
        pd.Timedelta(hours=12)
    )
    frame.loc[rollover, "scheduled_end"] += pd.Timedelta(days=1)
    frame["actual_start"] = parse_field(frame, names[2], frame.scheduled_start)
    frame["actual_end"] = parse_field(frame, names[3], frame.scheduled_end)
    frame["start"] = (frame.scheduled_start - frame.date).dt.total_seconds() / 60
    frame["end"] = (frame.scheduled_end - frame.date).dt.total_seconds() / 60
    frame["scheduled"] = frame.end - frame.start
    frame["actual"] = (frame.actual_end - frame.actual_start).dt.total_seconds() / 60
    valid_schedule = np.isfinite(frame[["start", "end"]]).all(axis=1) & frame.scheduled.gt(0)
    logging.info("CLEVER runtime: %d rows have invalid schedules.", int((~valid_schedule).sum()))
    frame = frame.loc[valid_schedule].copy()
    frame.loc[frame.actual.le(0), "actual"] = np.nan
    return distinct_daily_rows(frame, ["date", *TRIP_VARIANT], ["actual_start", "actual_end"])


def summarize_runtime(frame: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Keep schedule variants separate and trim within each scheduled departure/runtime."""
    records = []
    for key, group in frame.groupby(TRIP_VARIANT, sort=True):
        retained = trim_sample(group, "actual", cfg.trim)
        stats = describe(retained.actual, cfg.percentile)
        current = group.scheduled.iloc[0]
        records.append(
            {
                **dict(zip(TRIP_VARIANT, cast("tuple[Any, ...]", key))),
                "Scheduled min": current,
                "Mean min": stats["mean"],
                "Median min": stats["median"],
                "P85 min": stats["p85"],
                "P95 min": stats["p95"],
                "Median +/- min": stats["median"] - current,
                "P85 +/- min": stats["p85"] - current,
                "N retained": stats["n"],
                "N before trim": int(group.actual.notna().sum()),
                "N trimmed": int(group.actual.notna().sum()) - stats["n"],
                "First date": retained.date.min().strftime("%Y-%m-%d") if stats["n"] else "",
                "Last date": retained.date.max().strftime("%Y-%m-%d") if stats["n"] else "",
                "Source trip IDs": "; ".join(sorted(set(group.trip_id))),
                "Status": "No valid observations"
                if not stats["n"]
                else "Thin sample"
                if stats["n"] < cfg.min_observations
                else "Observed",
            }
        )
    return pd.DataFrame(records)


def display_trip_table(summary: pd.DataFrame, route: str) -> pd.DataFrame:
    """Put the scheduled departure and comparisons first in the user-facing table."""
    out = summary.loc[summary.route.eq(route)].copy()
    out["start"] = out.start.map(minutes_to_hhmm)
    out["end"] = out.end.map(minutes_to_hhmm)
    return out.drop(columns="route").rename(
        columns={
            "direction": "Direction",
            "day_type": "Day type",
            "start": "Start",
            "end": "Scheduled finish",
        }
    )


# =============================================================================
# STAGE 3: OBSERVED SEGMENT RUNTIMES
# =============================================================================


def build_observations(
    cfg: Settings,
    runtime: pd.DataFrame | None = None,
    schedule: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Keep source trips even when GTFS enrichment fails; never invent actual events.

    Intermediate selected events are departures; the terminal selected event is
    arrival. Generic Actual Time is a marked fallback. A uniquely identified
    runtime-export endpoint can fill a missing endpoint only, using date, route,
    direction and scheduled start/end, not assuming the two exports share IDs.
    """
    frame = load_clever(
        cfg.timepoint_path,
        cfg,
        [
            "timepoint_order",
            "timepoint_id",
            "timepoint_scheduled_time_only",
            "timepoint_actual_time_only",
        ],
    )
    frame["order"] = pd.to_numeric(frame.timepoint_order, errors="coerce")
    frame = frame.loc[
        frame.order.ge(1) & frame.order.mod(1).eq(0) & frame.timepoint_id.ne("")
    ].copy()
    raw_generic = (
        "timepoint_actual_time_original"
        if "timepoint_actual_time_original" in frame
        else "timepoint_actual_time_only"
    )
    frame["_scheduled"] = parse_field(frame, "timepoint_scheduled_time_only")
    for dest, source in [
        ("_arrival", "timepoint_arrival_time_raw"),
        ("_departure", "timepoint_departure_time_raw"),
        ("_generic", raw_generic),
        ("_sched_arr", "timepoint_scheduled_arrival_time_only"),
        ("_sched_dep", "timepoint_scheduled_departure_time_only"),
    ]:
        frame[dest] = parse_field(frame, source, frame._scheduled)
    event_cols = ["_scheduled", "_arrival", "_departure", "_generic", "_sched_arr", "_sched_dep"]
    frame = distinct_daily_rows(
        frame, ["date", "route", "direction", "trip_id", "order"], ["timepoint_id", *event_cols]
    )
    runtime_groups = (
        {key: group for key, group in runtime.groupby(["date", "route", "direction"])}
        if runtime is not None and not runtime.empty
        else {}
    )
    rows, matched_count, source_count = [], 0, 0
    groups = frame.groupby(["date", "route", "direction", "day_type", "trip_id"], sort=False)
    for key, group in groups:
        date, route, direction, day_type, trip_id = cast("tuple[Any, ...]", key)
        group = group.sort_values("order").copy()
        if len(group) < 2 or group.order.iloc[0] != 1:
            continue
        data = group.to_dict("records")
        # Clock-only schedules crossing midnight are aligned in sequence.
        scheduled, previous = [], pd.NaT
        for i, row in enumerate(data):
            explicit = row["_sched_arr"] if i == len(data) - 1 else row["_sched_dep"]
            value = explicit if pd.notna(explicit) else row["_scheduled"]
            raw = str(row["timepoint_scheduled_time_only"])
            if pd.notna(previous) and pd.notna(value) and value < previous:
                if ":" in raw and len(raw) <= 8 and previous - value > pd.Timedelta(hours=12):
                    value += pd.Timedelta(days=1)
            scheduled.append(value)
            if pd.notna(value):
                previous = value
        start, end = service_minute(scheduled[0], date), service_minute(scheduled[-1], date)
        if not np.isfinite([start, end]).all() or end <= start:
            continue
        source_codes = [row["timepoint_id"] for row in data]
        raw_minutes = [service_minute(value, date) for value in scheduled]
        enriched = match_points(
            schedule, route, raw_minutes, source_codes, cfg.match_tolerance, require_terminal=False
        )
        schedule_basis = (
            "GTFS" if enriched is not None else "CLEVER passing times (verify event meaning)"
        )
        matched_count += enriched is not None
        source_count += enriched is None
        if enriched is not None and schedule is not None:
            gtfs_trip = enriched.trip_id.iloc[0]
            trip = schedule["trips"].set_index("trip_id").loc[gtfs_trip]
            full = (
                schedule["points"].loc[schedule["points"].trip_id.eq(gtfs_trip)].sort_values("seq")
            )
            event_minutes = [
                float(p.arrival if p.seq == p.count else p.departure) for p in full.itertuples()
            ]
            holding = full.holding.tolist()
            names = full.stop_name.str.strip().tolist()
            positions = (enriched.seq - 1).astype(int).tolist()
            start, end = float(trip.start), float(trip.end)
            pattern, gtfs_direction = trip.pattern, trip.direction
            pattern_key = json.dumps(full.stop_id.tolist())
        else:
            event_minutes = raw_minutes
            holding = [minutes_between(row["_sched_dep"], row["_sched_arr"]) for row in data]
            names = [str(row.get("timepoint", "")).strip() or row["timepoint_id"] for row in data]
            gtfs_trip = ""
            positions = list(range(len(data)))
            pattern, gtfs_direction = "Unmatched", ""
            pattern_key = json.dumps(source_codes)
        fallback = runtime_groups.get((date, route, direction), pd.DataFrame())
        if not fallback.empty:
            fallback = fallback.loc[
                fallback.start.sub(start).abs().le(cfg.match_tolerance)
                & fallback.end.sub(end).abs().le(cfg.match_tolerance)
            ]
        actual = [pd.NaT] * len(event_minutes)
        basis = ["missing"] * len(event_minutes)
        dwell = [np.nan] * len(event_minutes)
        for i, row in enumerate(data):
            position = positions[i]
            final = position == len(event_minutes) - 1
            field = "_arrival" if final else "_departure"
            value, label = row[field], "arrival" if final else "departure"
            if pd.isna(value):
                value, label = row["_generic"], "generic (unverified)"
            if pd.isna(value) and position in (0, len(event_minutes) - 1) and len(fallback) == 1:
                value = fallback.actual_end.iloc[0] if final else fallback.actual_start.iloc[0]
                label = "runtime endpoint"
            # Only clock-only source values are shifted; explicit dated timestamps stay intact.
            source_field = "timepoint_arrival_time_raw" if final else "timepoint_departure_time_raw"
            if label.startswith("generic"):
                source_field = raw_generic
            raw = str(row.get(source_field, ""))
            if pd.notna(value) and ":" in raw and len(raw) <= 8 and pd.notna(scheduled[i]):
                value += pd.Timedelta(days=round((scheduled[i] - value).total_seconds() / 86400))
            actual[position] = value
            basis[position] = label if pd.notna(value) else "missing"
            pair = minutes_between(row["_departure"], row["_arrival"])
            dwell[position] = pair if pair >= 0 else np.nan
        # A unique runtime-export endpoint may also fill an entirely absent endpoint row.
        if enriched is not None and len(fallback) == 1:
            for position, column in [(0, "actual_start"), (len(actual) - 1, "actual_end")]:
                if pd.isna(actual[position]):
                    actual[position] = fallback[column].iloc[0]
                    basis[position] = "runtime endpoint"
        profile = json.dumps([pattern_key, event_minutes, gtfs_trip])
        for i in range(1, len(event_minutes)):
            elapsed = minutes_between(actual[i], actual[i - 1])
            elapsed = elapsed if elapsed >= 0 else np.nan
            scheduled_elapsed = event_minutes[i] - event_minutes[i - 1]
            scheduled_elapsed = scheduled_elapsed if scheduled_elapsed >= 0 else np.nan
            rows.append(
                {
                    "date": date,
                    "route": route,
                    "direction": direction,
                    "day_type": day_type,
                    "trip_id": trip_id,
                    "start": start,
                    "end": end,
                    "profile": profile,
                    "pattern_key": pattern_key,
                    "pattern": pattern,
                    "gtfs_direction": gtfs_direction,
                    "seq": i,
                    "from": names[i - 1],
                    "to": names[i],
                    "actual": elapsed,
                    "scheduled": scheduled_elapsed,
                    "scheduled_holding": holding[i] if i < len(event_minutes) - 1 else 0.0,
                    "observed_dwell": dwell[i] if i < len(event_minutes) - 1 else np.nan,
                    "basis": schedule_basis,
                    "generic": "unverified" in basis[i] or "unverified" in basis[i - 1],
                    "gtfs_trip": gtfs_trip,
                }
            )
    logging.info(
        "CLEVER timepoint trips: %d enriched with GTFS; %d retained on CLEVER schedule.",
        matched_count,
        source_count,
    )
    return pd.DataFrame(rows)


def summarize_segments(observations: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Trim and describe each trip/segment sample separately, keeping partial trips usable."""
    if observations.empty:
        return pd.DataFrame()
    keys = ["route", "direction", "day_type", "start", "end", "profile", "pattern_key", "seq"]
    observations = distinct_daily_rows(
        observations, ["date", *keys], ["actual", "scheduled", "observed_dwell"]
    )
    records = []
    for key, group in observations.groupby(keys, sort=True, dropna=False):
        retained = trim_sample(group, "actual", cfg.trim)
        stats = describe(retained.actual, cfg.percentile)
        first = group.iloc[0]
        records.append(
            {
                **dict(zip(keys, cast("tuple[Any, ...]", key))),
                "from": first["from"],
                "to": first["to"],
                "scheduled": first.scheduled,
                "median": stats["median"],
                "p85": stats["p85"],
                "p95": stats["p95"],
                "target": stats["target"],
                "n": stats["n"],
                "n_before": int(group.actual.notna().sum()),
                "scheduled_holding": first.scheduled_holding,
                "observed_dwell": (
                    retained.observed_dwell.median()
                    if retained.observed_dwell.notna().any()
                    else np.nan
                ),
                "generic_n": int(retained.generic.sum()),
                "basis": first.basis,
                "gtfs_trip": first.gtfs_trip,
                "gtfs_direction": first.gtfs_direction,
                "pattern": first.pattern,
                "source_ids": "; ".join(sorted(set(group.trip_id))),
            }
        )
    result = pd.DataFrame(records)
    return result


def display_segment_table(summary: pd.DataFrame, route: str) -> pd.DataFrame:
    """Show segment intervals and counts without hiding holding inside an added allowance."""
    out = summary.loc[summary.route.eq(route)].copy()
    out["start"] = out.start.map(minutes_to_hhmm)
    out["change"] = out.p85 - out.scheduled
    cols = [
        "direction",
        "day_type",
        "pattern",
        "start",
        "seq",
        "from",
        "to",
        "scheduled",
        "median",
        "p85",
        "p95",
        "change",
        "n",
        "n_before",
        "scheduled_holding",
        "observed_dwell",
        "generic_n",
        "basis",
        "gtfs_trip",
        "source_ids",
    ]
    labels = [
        "Direction",
        "Day type",
        "Pattern",
        "Start",
        "Segment",
        "From",
        "To",
        "Scheduled interval min",
        "Observed median min",
        "Observed P85 min",
        "Observed P95 min",
        "P85 +/- min",
        "N retained",
        "N before trim",
        "Scheduled holding included min",
        "Observed dwell median min",
        "N generic events",
        "Basis",
        "GTFS trip",
        "Source trip IDs",
    ]
    out = out[cols]
    out.columns = labels
    return out


# =============================================================================
# STAGE 4: SEGMENT RUNTIME RECOMMENDATIONS
# =============================================================================


def build_recommendation_sheets(summary: pd.DataFrame, route: str, cfg: Settings) -> list[Sheet]:
    """Recommend each supported segment independently; do not create periods or cap increases.

    Intervals run from departure to departure, except terminal arrival, and already
    include destination dwell. Do not add the Stage 1 holding minutes again.
    Summed segment percentiles are a starting point, not a trip-level OTP guarantee.
    """
    if summary.empty:
        return []
    frame = summary.loc[summary.route.eq(route) & summary.basis.eq("GTFS")].copy()
    # Recommendation sheets use the scheduled direction/pattern, never a missing-row subset.
    frame["direction"] = frame.gtfs_direction
    sheets = []
    for (_direction, day_type, pattern), group in frame.groupby(
        ["direction", "day_type", "pattern"]
    ):
        example = group.drop_duplicates("seq").sort_values("seq")
        columns = ["Source trip IDs", "Start", "Planner period", "Review note"]
        scheduled_cols, recommendation_cols = [], []
        for seg in example.itertuples():
            label = f"{seg.to}\n"
            scheduled_cols.append(len(columns) + 1)
            recommendation_cols.append(len(columns) + 2)
            columns += [label + "Scheduled interval", label + "Recommended interval", label + "+/-"]
        current_total_col = len(columns) + 1
        columns += ["Total scheduled min", "Total recommended min", "Total +/- min"]
        rows = []
        for row_number, (_, trip) in enumerate(
            group.groupby(["start", "end", "profile"], sort=True), 2
        ):
            trip = trip.sort_values("seq")
            notes = []
            if trip.n.lt(cfg.min_observations).any():
                notes.append("Thin/missing samples: blank recommendations require manual entry")
            if trip.generic_n.gt(0).any() or trip.basis.ne("GTFS").any():
                notes.append("Verify source event conventions")
            values = [
                trip.source_ids.iloc[0],
                minutes_to_hhmm(trip.start.iloc[0]),
                "",
                "; ".join(notes),
            ]
            for seg in cast("Iterable[Any]", trip.itertuples()):
                recommended = (
                    math.ceil((seg.target - 1e-10) / cfg.rounding) * cfg.rounding
                    if seg.n >= cfg.min_observations and np.isfinite(seg.target)
                    else np.nan
                )
                current_col, rec_col = len(values) + 1, len(values) + 2
                values += [
                    seg.scheduled,
                    recommended,
                    formula(
                        f"=IF(COUNT({column_letter(current_col)}{row_number}:"
                        f"{column_letter(rec_col)}{row_number})=2,"
                        f'{column_letter(rec_col)}{row_number}-{column_letter(current_col)}{row_number},"")'
                    ),
                ]
            current_refs = ",".join(f"{column_letter(c)}{row_number}" for c in scheduled_cols)
            rec_refs = ",".join(f"{column_letter(c)}{row_number}" for c in recommendation_cols)
            values += [
                formula(f'=IF(COUNT({current_refs})={len(scheduled_cols)},SUM({current_refs}),"")'),
                formula(f'=IF(COUNT({rec_refs})={len(recommendation_cols)},SUM({rec_refs}),"")'),
                formula(
                    f"=IF(COUNT({column_letter(current_total_col)}{row_number}:"
                    f"{column_letter(current_total_col + 1)}{row_number})=2,"
                    f"{column_letter(current_total_col + 1)}{row_number}-"
                    f'{column_letter(current_total_col)}{row_number},"")'
                ),
            ]
            rows.append(values)
        day = {"Weekday": "WK", "Saturday": "SA", "Sunday": "SU"}[day_type]
        sheets.append(Sheet(f"Recommend_{day}_{pattern}", columns, rows))
    return sheets


# =============================================================================
# WORKFLOW AND SCRIPT ENTRY POINT
# =============================================================================


def notes(cfg: Settings, route: str, direction: str, status: dict[int, str]) -> Sheet:
    """Keep short definitions and stage status next to the worksheets."""
    return Sheet(
        "Notes",
        ["Item", "Explanation"],
        [
            ["Route", route],
            [
                "Direction",
                direction + ". D0/D1 are GTFS direction identifiers; consult the stop names "
                "and source Direction columns. Unmatched labels identify records not assigned "
                "uniquely.",
            ],
            ["Stages", "; ".join(f"{k}: {v}" for k, v in sorted(status.items()))],
            [
                "1 — GTFS",
                f"Reference date {cfg.gtfs_service_date}. One row per scheduled trip; sheets "
                "split by GTFS direction and ordered timepoint pattern. Block first; movement, "
                "then holding at its destination; runtime SUM formula and layover last.",
            ],
            [
                "Holding",
                "GTFS scheduled departure minus arrival at an intermediate stop. This is "
                "scheduled dwell / potential holding, not proof that the entire amount is "
                "removable. Origin and terminal dwell are outside the trip runtime.",
            ],
            [
                "Layover",
                "Only consecutive active trips in the full GTFS block meeting at the same "
                "stop_id have a known scheduled gap. Includes interlined routes. Different stops, "
                "last trip, missing block or invalid block order leave layover blank; blank is "
                "not zero.",
            ],
            [
                "2 — Observed trips",
                "Runtime-export actual finish minus actual start, compared with "
                "its own scheduled finish minus start. Independent of GTFS/timepoint matching. "
                "Different scheduled start/end variants and day types remain separate.",
            ],
            [
                "3 — Observed segments",
                "Previous departure to next departure, except arrival at the "
                "terminal. These intervals include destination dwell. Explicit events take "
                "precedence; generic events are flagged. Missing actual values are not filled. "
                "Intervals follow the full GTFS pattern when uniquely matched. Missing rows do "
                "not create new patterns or combine adjacent segments; available segments "
                "remain usable.",
            ],
            [
                "Schedule comparison",
                "Segment scheduled intervals use GTFS when timetable and ordered "
                "points have one unique match. Otherwise CLEVER scheduled passing intervals remain "
                "visible in the observed-segment summary with a verify-event-meaning label. "
                "Unmatched records do not create recommendation sheets.",
            ],
            [
                "Trimming",
                f"{cfg.trim:.0%} per tail, counts rounded down: whole-trip samples in Stage 2; "
                "each trip/segment sample independently in Stages 3–4. Counts are shown. These "
                "sources and retained samples can differ; segment medians/percentiles need not "
                "add to trip statistics.",
            ],
            [
                "4 — Recommendations",
                f"Each segment uses observed P{cfg.percentile * 100:g}, rounded UP "
                f"to {cfg.rounding:g} minute(s), with at least {cfg.min_observations} retained "
                "observations. Otherwise the recommended cell is blank. Positive/negative changes "
                "and totals are Excel formulas; totals remain blank until every segment has a "
                "numeric recommendation.",
            ],
            [
                "Do not double-count holding",
                "Recommended segment intervals already include observed "
                "intermediate dwell. Do not add Stage 1 holding again. Decide manually how to "
                "split movement and holding when drafting the actual timetable.",
            ],
            [
                "Manual work",
                "Create period blocks, refine runtimes and review holding, layover, "
                "frequency/span and available hours. No automatic periods, hour cap or layover "
                "recommendation. Segment percentiles do not guarantee trip-level or "
                "timepoint OTP.",
            ],
            [
                "Planner period",
                "The blank column beside Start is for proposed period labels or boundaries. "
                "Rows remain trip-level while boundaries are being selected. Writing a label "
                "does not pool observations or recalculate a period percentile. Preserve this "
                "workbook when rerunning.",
            ],
            [
                "Date selection",
                f"{cfg.date_start or 'Open start'} through {cfg.date_end or 'open end'}; "
                f"weekdays {cfg.days_of_week} (Monday=0); "
                f"federal holidays excluded={cfg.exclude_holidays}. "
                "Choose dates representative of the timetable and season.",
            ],
            [
                "Excel",
                "Open in Excel to calculate formulas. Recommendation values and Planner period "
                "cells can be edited; change/total formulas update. Every run creates a new "
                "folder.",
            ],
        ],
    )


def direction_partitions(
    route: str,
    schedule: dict[str, pd.DataFrame] | None,
    trips: pd.DataFrame,
    segments: pd.DataFrame,
    tolerance: float,
) -> Iterator[tuple[str, dict[str, pd.DataFrame] | None, pd.DataFrame, pd.DataFrame]]:
    """Split by GTFS direction, retaining unresolved source records for review.

    Source direction names are mapped only when all uniquely identified records
    support one GTFS direction. Shared labels across both GTFS directions never
    force those directions into one workbook.
    """
    roster = (
        schedule["trips"].loc[schedule["trips"].route.eq(route)]
        if schedule is not None
        else pd.DataFrame()
    )
    frames = [
        frame.loc[frame.route.eq(route)].copy() if not frame.empty else pd.DataFrame()
        for frame in (trips, segments)
    ]
    source_directions = {}
    for frame in frames:
        frame["_book_direction"] = ""
        for index, row in frame.iterrows():
            candidates = roster
            if not roster.empty:
                if "gtfs_trip" in row and row.gtfs_trip:
                    candidates = roster.loc[roster.trip_id.eq(row.gtfs_trip)]
                else:
                    candidates = roster.loc[
                        roster.start.sub(row.start).abs().le(tolerance)
                        & roster.end.sub(row.end).abs().le(tolerance)
                    ]
            if len(candidates) == 1:
                direction = "D" + (str(candidates.direction.iloc[0]) or "UNKNOWN")
                frame.loc[index, "_book_direction"] = direction
                source_directions.setdefault(str(row.direction), set()).add(direction)
    for frame in frames:
        for index, row in frame.loc[frame._book_direction.eq("")].iterrows():
            choices = source_directions.get(str(row.direction), set())
            label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(row.direction)).strip("_") or "UNKNOWN"
            frame.loc[index, "_book_direction"] = (
                next(iter(choices)) if len(choices) == 1 else "UNMATCHED_" + label
            )
    directions = (
        set("D" + (str(d) or "UNKNOWN") for d in roster.direction) if not roster.empty else set()
    )
    for frame in frames:
        directions.update(frame._book_direction)
    for direction in sorted(directions or {"UNMATCHED"}):
        sub_schedule = None
        if schedule is not None:
            selected = (
                roster.loc[
                    roster.direction.map(lambda d: "D" + (str(d) or "UNKNOWN")).eq(direction)
                ]
                if not roster.empty
                else roster
            )
            sub_schedule = {"trips": selected, "points": schedule["points"]}
        selected_frames = [
            frame.loc[frame._book_direction.eq(direction)].drop(columns="_book_direction")
            for frame in frames
        ]
        yield direction, sub_schedule, selected_frames[0], selected_frames[1]


def run(cfg: Settings) -> dict[str, object]:
    """Execute requested stages; a failed source does not hide other completed stages."""
    if not 0 <= cfg.trim < 0.5 or not 0 < cfg.percentile <= 1 or cfg.rounding <= 0:
        raise ValueError("Use 0 <= trim < .5, 0 < percentile <= 1, and positive rounding.")
    if cfg.min_observations < 1 or not cfg.stages or set(cfg.stages) - {1, 2, 3, 4}:
        raise ValueError("Use a positive sample minimum and stages from 1 through 4.")
    if not cfg.days_of_week or set(cfg.days_of_week) - set(range(7)):
        raise ValueError("DAYS_OF_WEEK must contain values from 0 through 6.")
    if cfg.date_start and cfg.date_end and cfg.date_start > cfg.date_end:
        raise ValueError("DATE_START must not be later than DATE_END.")
    routes = tuple(dict.fromkeys(route_label(r) for r in cfg.routes))
    if any(not r or not all(c.isalnum() or c in "_-" for c in r) for r in routes):
        raise ValueError("Route labels must be nonblank letters/numbers, underscores or hyphens.")
    cfg = replace(cfg, routes=routes)
    output = (
        cfg.output_dir / "runtime_workflow" / dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger()
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    schedule, runtime, trips, segments = None, None, pd.DataFrame(), pd.DataFrame()
    status = {}
    files = []
    try:
        logging.info(
            "Routes: %s | stages %s | trim %.0f%% each tail.", routes, cfg.stages, cfg.trim * 100
        )
        logging.warning(
            "Use observation dates representative of the timetable, season and day type."
        )
        if set(cfg.stages) & {1, 3, 4}:
            try:
                with log_stage("Read GTFS", heartbeat=True):
                    schedule = load_schedule(cfg)
                if 1 in cfg.stages:
                    status[1] = "Completed"
            except Exception as exc:
                logging.exception("GTFS stage unavailable; CLEVER reports can still run.")
                if 1 in cfg.stages:
                    status[1] = "FAILED: " + str(exc)
        if set(cfg.stages) & {2, 3, 4}:
            try:
                with log_stage("Read CLEVER runtime export", heartbeat=True):
                    runtime = load_runtime(cfg)
                if 2 in cfg.stages:
                    trips = summarize_runtime(runtime, cfg)
                    status[2] = "Completed" if not trips.empty else "No selected observations"
            except Exception as exc:
                logging.exception("Runtime export unavailable; timepoint summaries can still run.")
                if 2 in cfg.stages:
                    status[2] = "FAILED: " + str(exc)
        if set(cfg.stages) & {3, 4}:
            try:
                with log_stage("Summarize CLEVER segments", heartbeat=True):
                    observations = build_observations(cfg, runtime, schedule)
                    segments = summarize_segments(observations, cfg)
                for stage in set(cfg.stages) & {3, 4}:
                    status[stage] = "Completed" if not segments.empty else "No selected segments"
            except Exception as exc:
                logging.exception("Segment calculation failed.")
                for stage in set(cfg.stages) & {3, 4}:
                    status[stage] = "FAILED: " + str(exc)
        for route, direction, selected_schedule, selected_trips, selected_segments in (
            (name, *partition)
            for name in routes
            for partition in direction_partitions(
                name, schedule, trips, segments, cfg.match_tolerance
            )
        ):
            sheets = []
            route_status = dict(status)
            if 4 in cfg.stages and not selected_segments.empty:
                sheets += build_recommendation_sheets(selected_segments, route, cfg)
                if not sheets:
                    route_status[4] = "No uniquely matched GTFS segments"
            elif 4 in cfg.stages and not route_status.get(4, "").startswith("FAILED"):
                route_status[4] = "No selected segments"
            if 1 in cfg.stages and selected_schedule is not None:
                sheets += build_gtfs_sheets(selected_schedule, route)
                if selected_schedule["trips"].empty:
                    route_status[1] = "No scheduled trips on reference date"
            if 2 in cfg.stages and not selected_trips.empty:
                sheets.append(
                    from_frame("2_Observed_trips", display_trip_table(selected_trips, route))
                )
            elif 2 in cfg.stages and not route_status.get(2, "").startswith("FAILED"):
                route_status[2] = "No selected observations"
            if 3 in cfg.stages and not selected_segments.empty:
                sheets.append(
                    from_frame(
                        "3_Observed_segments", display_segment_table(selected_segments, route)
                    )
                )
            elif 3 in cfg.stages and not route_status.get(3, "").startswith("FAILED"):
                route_status[3] = "No selected segments"
            sheets.append(notes(cfg, route, direction, route_status))
            path = output / f"route_{route}_{direction}_runtime.xlsx"
            write_book(path, sheets)
            files.append(path)
            logging.info("Wrote %s", path)
        (output / "settings.json").write_text(
            json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8"
        )
        return {
            "files": files,
            "status": status,
            "schedule": schedule,
            "trips": trips,
            "segments": segments,
            "output": output,
            "failed": any(value.startswith("FAILED") for value in status.values()),
        }
    finally:
        logger.removeHandler(handler)
        handler.close()


def main(stages: Sequence[int] | None = None, argv: Sequence[str] | None = None) -> int:
    """Run the configured stages, supporting ArcGIS notebooks and normal Python execution."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gtfs", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--timepoints", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4),
        help="Stages to run; defaults to STAGES in USER SETTINGS.",
    )
    interactive = "ipykernel" in sys.modules or "IPython" in sys.modules
    args = parser.parse_args([] if argv is None and interactive else argv)
    cfg = Settings()
    selected_stages = args.stages
    del args.stages
    changes = {
        key + "_path"
        if key in ("gtfs", "runtime")
        else "timepoint_path"
        if key == "timepoints"
        else "output_dir": value
        for key, value in vars(args).items()
        if value is not None
    }
    if selected_stages is not None:
        changes["stages"] = tuple(selected_stages)
    if stages is not None:
        changes["stages"] = tuple(stages)
    cfg = replace(cfg, **changes)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        result = run(cfg)
        logging.info("Done. Open the route workbooks in %s", result["output"])
        return 1 if result["failed"] else 0
    except Exception:
        logging.exception("Workflow failed.")
        return 1


if __name__ == "__main__":
    result_code = main()
    if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
        raise SystemExit(result_code)
