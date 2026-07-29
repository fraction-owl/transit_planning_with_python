"""Flag scheduled intervals between consecutive timepoints that are too long.

This script reviews every trip in a GTFS feed — for all schedules
(service_ids) by default — and measures the scheduled running time between
each pair of consecutive timepoints. Any interval longer than
MAX_INTERVAL_MINUTES (default 10) is flagged for review. Long gaps
between timepoints make mid-route on-time performance hard to monitor and
often signal that a timepoint is missing from the schedule data.

Timepoints are the stop_times.txt rows with ``timepoint == 1``. When the
feed carries no ``timepoint`` column (or marks no rows), every stop with a
scheduled arrival or departure time is treated as a timepoint instead, so
the check still works on feeds that only publish times at timepoints — and
degrades to a stop-to-stop interval review on feeds that time every stop.

Inputs
------
- A GTFS feed (folder or .zip) with stops.txt, routes.txt, trips.txt, and
  stop_times.txt. calendar.txt, when present, labels each service_id with a
  human-readable schedule name (Weekday, Saturday, ...).

Outputs
-------
- timepoint_interval_flags.csv — one row per flagged interval (schedule,
  route, trip, bounding timepoints, scheduled times, interval minutes).
- timepoint_interval_summary.csv — flagged intervals rolled up per
  (schedule, route, direction, timepoint pair) with flagged-trip counts and
  min/mean/max interval minutes.
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI
flags) and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"  # folder of .txt files or a .zip
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Maximum scheduled minutes allowed between consecutive timepoints on a trip.
# Intervals STRICTLY greater than this are flagged. Timepoints are commonly
# spaced every 5-10 scheduled minutes of running time, so anything over 10
# merits review; lower this to tighten the review.
MAX_INTERVAL_MINUTES: float = 10.0

# Schedules (service_ids from calendar.txt / trips.txt) to review.
# Empty = review all schedules in the feed.
FILTER_SERVICE_IDS: list[str] = []

# Route filters, matched against route_id OR route_short_name. Empty = all.
FILTER_IN_ROUTES: list[str] = []
FILTER_OUT_ROUTES: list[str] = []

# Output filenames (written inside OUTPUT_DIR).
DETAIL_FILENAME: str = r"timepoint_interval_flags.csv"
SUMMARY_FILENAME: str = r"timepoint_interval_summary.csv"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output is never
# left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Columns required in each loaded GTFS table (headers only; values may be blank).
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "stops": {"stop_id"},
    "routes": {"route_id"},
    "trips": {"trip_id", "route_id", "service_id"},
    "stop_times": {"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"},
}

# Column order of the interval table built by compute_timepoint_intervals.
INTERVAL_COLUMNS: List[str] = [
    "service_id",
    "route_id",
    "route_short_name",
    "direction_id",
    "trip_id",
    "from_stop_id",
    "from_stop_name",
    "from_stop_sequence",
    "from_departure",
    "to_stop_id",
    "to_stop_name",
    "to_stop_sequence",
    "to_arrival",
    "interval_minutes",
]

# =============================================================================
# CORE LOGIC
# =============================================================================


def validate_required_columns(data: Mapping[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` if any required GTFS column is missing.

    Args:
        data: Mapping of table name → DataFrame from :func:`load_gtfs_data`.

    Raises:
        ValueError: Naming every missing table.column so the user can fix
            the feed (or the export that produced it) in one pass.
    """
    problems: list[str] = []
    for table, needed in REQUIRED_COLUMNS.items():
        if table not in data:
            continue
        missing = needed - set(data[table].columns)
        if missing:
            problems.append(f"{table}.txt is missing column(s): {', '.join(sorted(missing))}")
    if problems:
        raise ValueError("GTFS validation failed:\n  " + "\n  ".join(problems))


def filter_trips(
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    service_ids: Sequence[str] = (),
    routes_include: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Apply the schedule and route filters to *trips*.

    Route filters match against both ``route_id`` and ``route_short_name``
    so users can list whichever identifier they know.

    Args:
        trips: Parsed *trips.txt*.
        routes: Parsed *routes.txt* (for the short-name lookup).
        service_ids: Schedules to keep; empty keeps all schedules.
        routes_include: Routes to keep; empty keeps all routes.
        routes_exclude: Routes to drop.

    Returns:
        The filtered trips table.
    """
    kept = trips.copy()
    if service_ids:
        wanted = {str(s) for s in service_ids}
        kept = kept.loc[kept["service_id"].astype(str).isin(wanted)]

    short_names: dict[str, str] = {}
    if "route_short_name" in routes.columns:
        short_names = dict(zip(routes["route_id"], routes["route_short_name"].fillna("")))

    include = {str(r) for r in routes_include}
    exclude = {str(r) for r in routes_exclude}
    if include or exclude:

        def _keys(route_id: str) -> set[str]:
            return {str(route_id), str(short_names.get(route_id, ""))}

        keep_mask = kept["route_id"].map(
            lambda rid: (not include or bool(_keys(rid) & include)) and not (_keys(rid) & exclude)
        )
        kept = kept.loc[keep_mask]

    if kept.empty:
        logging.warning("No trips remain after applying the schedule/route filters.")
    return kept


def select_timepoint_rows(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Return the stop_times rows that count as timepoints.

    Rows with ``timepoint == 1`` are used when the feed marks any. When the
    column is absent — or present but nothing is marked — every row with a
    scheduled arrival or departure time is used instead.

    Args:
        stop_times: Raw *stop_times.txt* DataFrame (string dtypes).

    Returns:
        The timepoint rows, with ``stop_sequence`` coerced to numeric and
        rows lacking a usable sequence dropped.
    """
    out = stop_times.copy()
    out["stop_sequence"] = pd.to_numeric(out["stop_sequence"], errors="coerce")
    n_bad_seq = int(out["stop_sequence"].isna().sum())
    if n_bad_seq:
        logging.warning("Dropped %d stop_times row(s) with non-numeric stop_sequence.", n_bad_seq)
        out = out.loc[out["stop_sequence"].notna()]

    if "timepoint" in out.columns:
        marked = out["timepoint"].fillna("").astype(str).str.strip() == "1"
        if marked.any():
            logging.info("Using %d stop_times rows marked timepoint=1.", int(marked.sum()))
            return out.loc[marked].copy()
        logging.warning(
            "'timepoint' column present but no row is marked 1 — "
            "treating every timed stop as a timepoint."
        )
    else:
        logging.warning(
            "No 'timepoint' column in stop_times.txt — treating every timed stop as a timepoint."
        )

    timed = out["arrival_time"].fillna("").astype(str).str.strip().ne("") | out[
        "departure_time"
    ].fillna("").astype(str).str.strip().ne("")
    logging.info("Using %d stop_times rows with scheduled times as timepoints.", int(timed.sum()))
    return out.loc[timed].copy()


def build_schedule_labels(calendar: Optional[pd.DataFrame]) -> dict[str, str]:
    """Map each service_id to a human-readable schedule label.

    Labels are inferred from the seven day-of-week flags in calendar.txt
    (Weekday, Saturday, Sunday, Weekend, Daily, ...). Callers should fall
    back to the raw service_id for ids not in the returned mapping — e.g.
    feeds that define service only in calendar_dates.txt.

    Args:
        calendar: Parsed *calendar.txt*, or ``None`` when the feed has none.

    Returns:
        Mapping of service_id → schedule label.
    """
    if calendar is None or calendar.empty or "service_id" not in calendar.columns:
        return {}

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    known = {
        frozenset(days[:5]): "Weekday",
        frozenset(days[:4]): "Weekday_except_Friday",
        frozenset(["friday", "saturday"]): "Friday-Saturday",
        frozenset(["saturday"]): "Saturday",
        frozenset(["sunday"]): "Sunday",
        frozenset(["saturday", "sunday"]): "Weekend",
        frozenset(days[:6]): "Mon-Sat",
        frozenset(days): "Daily",
    }
    labels: dict[str, str] = {}
    for _, row in calendar.iterrows():
        served = frozenset(d for d in days if str(row.get(d, "0")).strip() == "1")
        labels[str(row["service_id"])] = known.get(served, "Special")
    return labels


def compute_timepoint_intervals(
    timepoints: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    stops: pd.DataFrame,
) -> pd.DataFrame:
    """Measure the scheduled interval between consecutive timepoints per trip.

    Within each trip, timepoints are ordered by ``stop_sequence`` and each
    interval runs from the upstream timepoint's departure time to the
    downstream timepoint's arrival time (each falling back to the other
    field when blank). Rows with no parseable time are skipped; GTFS times
    past 24:00 are handled, and seconds are rounded to the nearest minute.

    Args:
        timepoints: Output of :func:`select_timepoint_rows`.
        trips: Parsed *trips.txt* (route_id / service_id / direction_id).
        routes: Parsed *routes.txt* (route_short_name lookup).
        stops: Parsed *stops.txt* (stop_name lookup).

    Returns:
        One row per consecutive timepoint pair, with the columns listed in
        :data:`INTERVAL_COLUMNS` (``interval_minutes`` may be negative when
        the feed's times run backwards — see :func:`flag_long_intervals`).
    """
    tp = timepoints.copy()
    if tp.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)

    tp["_depart_min"] = tp["departure_time"].map(parse_time_to_minutes)
    tp["_arrive_min"] = tp["arrival_time"].map(parse_time_to_minutes)
    tp["_depart_min"] = tp["_depart_min"].fillna(tp["_arrive_min"])
    tp["_arrive_min"] = tp["_arrive_min"].fillna(tp["_depart_min"])

    n_untimed = int(tp["_depart_min"].isna().sum())
    if n_untimed:
        logging.info("Skipped %d timepoint row(s) with no parseable scheduled time.", n_untimed)
        tp = tp.loc[tp["_depart_min"].notna()]
    if tp.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)

    tp = tp.sort_values(["trip_id", "stop_sequence"], kind="mergesort")
    nxt = tp.groupby("trip_id")[["stop_id", "stop_sequence", "_arrive_min"]].shift(-1)
    tp["to_stop_id"] = nxt["stop_id"]
    tp["to_stop_sequence"] = nxt["stop_sequence"]
    tp["_to_arrive_min"] = nxt["_arrive_min"]

    pairs = tp.loc[tp["to_stop_id"].notna()].copy()
    if pairs.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)

    pairs["interval_minutes"] = pairs["_to_arrive_min"] - pairs["_depart_min"]
    pairs["from_departure"] = pairs["_depart_min"].map(minutes_to_hhmm)
    pairs["to_arrival"] = pairs["_to_arrive_min"].map(minutes_to_hhmm)
    pairs = pairs.rename(columns={"stop_id": "from_stop_id", "stop_sequence": "from_stop_sequence"})
    pairs["from_stop_sequence"] = pairs["from_stop_sequence"].astype(int)
    pairs["to_stop_sequence"] = pairs["to_stop_sequence"].astype(int)

    trip_attrs = trips.copy()
    if "direction_id" not in trip_attrs.columns:
        trip_attrs["direction_id"] = ""
    trip_attrs = trip_attrs[["trip_id", "route_id", "service_id", "direction_id"]]
    trip_attrs = trip_attrs.drop_duplicates("trip_id")

    route_attrs = routes.copy()
    if "route_short_name" not in route_attrs.columns:
        route_attrs["route_short_name"] = route_attrs["route_id"]
    route_attrs = route_attrs[["route_id", "route_short_name"]].drop_duplicates("route_id")

    stop_names = stops.copy()
    if "stop_name" not in stop_names.columns:
        stop_names["stop_name"] = stop_names["stop_id"]
    stop_names = stop_names[["stop_id", "stop_name"]].drop_duplicates("stop_id")

    pairs = (
        pairs.merge(trip_attrs, on="trip_id", how="left")
        .merge(route_attrs, on="route_id", how="left")
        .merge(
            stop_names.rename(columns={"stop_id": "from_stop_id", "stop_name": "from_stop_name"}),
            on="from_stop_id",
            how="left",
        )
        .merge(
            stop_names.rename(columns={"stop_id": "to_stop_id", "stop_name": "to_stop_name"}),
            on="to_stop_id",
            how="left",
        )
    )
    return pairs[INTERVAL_COLUMNS].reset_index(drop=True)


def flag_long_intervals(intervals: pd.DataFrame, max_minutes: float) -> pd.DataFrame:
    """Return the intervals strictly longer than *max_minutes*.

    Negative intervals (scheduled times running backwards along the trip)
    are a different data problem: they are logged as a warning and excluded
    from the flags rather than being reported as long gaps.

    Args:
        intervals: Output of :func:`compute_timepoint_intervals`.
        max_minutes: Flag intervals strictly greater than this many minutes.

    Returns:
        The flagged rows, sorted by schedule, route, trip, and sequence.
    """
    negative = intervals.loc[intervals["interval_minutes"] < 0]
    if not negative.empty:
        logging.warning(
            "%d timepoint interval(s) run backwards in time (first example: trip '%s') — "
            "check the feed's schedule order. These rows are excluded from the flags.",
            len(negative),
            negative.iloc[0]["trip_id"],
        )

    flags = intervals.loc[intervals["interval_minutes"] > max_minutes].copy()
    sort_cols = [c for c in ("schedule",) if c in flags.columns]
    sort_cols += ["service_id", "route_id", "direction_id", "trip_id", "from_stop_sequence"]
    return flags.sort_values(sort_cols, ignore_index=True)


def summarize_flags(flags: pd.DataFrame) -> pd.DataFrame:
    """Roll the flagged intervals up to one row per timepoint pair.

    Args:
        flags: Output of :func:`flag_long_intervals` (with the ``schedule``
            label column attached).

    Returns:
        One row per (schedule, service_id, route, direction, timepoint pair)
        with the number of flagged trips/instances and the min/mean/max
        interval in minutes.
    """
    group_cols = [
        "schedule",
        "service_id",
        "route_id",
        "route_short_name",
        "direction_id",
        "from_stop_id",
        "from_stop_name",
        "to_stop_id",
        "to_stop_name",
    ]
    agg_cols = [
        "n_trips_flagged",
        "n_intervals_flagged",
        "min_interval_minutes",
        "mean_interval_minutes",
        "max_interval_minutes",
    ]
    if flags.empty:
        return pd.DataFrame(columns=[*group_cols, *agg_cols])

    grouped = (
        flags.groupby(group_cols, dropna=False)
        .agg(
            n_trips_flagged=("trip_id", "nunique"),
            n_intervals_flagged=("trip_id", "size"),
            min_interval_minutes=("interval_minutes", "min"),
            mean_interval_minutes=("interval_minutes", "mean"),
            max_interval_minutes=("interval_minutes", "max"),
        )
        .reset_index()
    )
    grouped["mean_interval_minutes"] = grouped["mean_interval_minutes"].round(1)
    return grouped.sort_values(
        ["schedule", "service_id", "route_id", "direction_id", "from_stop_id"],
        ignore_index=True,
    )


# =============================================================================
# OUTPUT & RUN LOG
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


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


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a review summary into *output_dir*.

    The sidecar is named after :data:`DETAIL_FILENAME` (same stem,
    ``_runlog.txt`` suffix) per CONTRIBUTING.md.

    Args:
        output_dir: Directory that received this run's output CSVs.
        summary_lines: Human-readable lines describing what the run did.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / f"{Path(DETAIL_FILENAME).stem}_runlog.txt"

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
        "TIMEPOINT INTERVAL REVIEW RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "REVIEW SUMMARY",
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
# REUSABLE FUNCTIONS (canonical copies from utils/ — do not edit here)
# =============================================================================


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


# =============================================================================
# PIPELINE
# =============================================================================


def run(
    gtfs_path: str,
    output_dir: Path,
    max_interval_minutes: float = MAX_INTERVAL_MINUTES,
    service_ids: Sequence[str] = (),
    routes_include: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Execute the timepoint-interval review and write all artifacts.

    Args:
        gtfs_path: GTFS feed folder or ``.zip`` archive.
        output_dir: Directory receiving both CSVs and the run log.
        max_interval_minutes: Flag intervals strictly greater than this many
            scheduled minutes.
        service_ids: Schedules to review; empty reviews all schedules.
        routes_include: Routes to keep (route_id or route_short_name).
        routes_exclude: Routes to drop (route_id or route_short_name).

    Returns:
        The flagged intervals (also written to disk).

    Raises:
        OSError: The feed path or a required file is missing.
        ValueError: The feed fails column validation or cannot be parsed.
        RuntimeError: The run log could not be written while
            ``REQUIRE_RUN_LOG`` is True.
    """
    data = load_gtfs_data(
        gtfs_path, files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
    )
    validate_required_columns(data)

    calendar: Optional[pd.DataFrame]
    try:
        calendar = load_gtfs_data(gtfs_path, files=("calendar.txt",))["calendar"]
    except (OSError, ValueError):
        calendar = None
        logging.info("No usable calendar.txt — schedule labels fall back to raw service_ids.")

    trips = filter_trips(data["trips"], data["routes"], service_ids, routes_include, routes_exclude)
    logging.info(
        "Reviewing %d trips across %d schedule(s).", len(trips), trips["service_id"].nunique()
    )

    stop_times = data["stop_times"].loc[data["stop_times"]["trip_id"].isin(trips["trip_id"])]
    timepoints = select_timepoint_rows(stop_times)
    intervals = compute_timepoint_intervals(timepoints, trips, data["routes"], data["stops"])

    labels = build_schedule_labels(calendar)
    intervals.insert(0, "schedule", intervals["service_id"].map(lambda sid: labels.get(sid, sid)))

    flags = flag_long_intervals(intervals, max_interval_minutes)
    summary = summarize_flags(flags)

    ensure_dir(output_dir)
    detail_path = output_dir / DETAIL_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    flags.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    logging.info("Wrote %d flagged interval(s) → %s", len(flags), detail_path)
    logging.info("Wrote %d summary row(s) → %s", len(summary), summary_path)
    if flags.empty:
        logging.info(
            "No timepoint interval exceeds %.1f minutes — nothing to review.",
            max_interval_minutes,
        )

    summary_lines = [
        f"GTFS feed:            {gtfs_path}",
        f"Max interval:         {max_interval_minutes:g} minutes",
        f"Schedules reviewed:   {', '.join(str(s) for s in service_ids) if service_ids else 'all'}",
        f"Trips reviewed:       {trips['trip_id'].nunique()}",
        f"Intervals measured:   {len(intervals)}",
        f"Intervals flagged:    {len(flags)}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return flags


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Flag scheduled intervals between consecutive timepoints that exceed a maximum, "
            "for every trip in every schedule of a GTFS feed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs",
        default=GTFS_PATH,
        help="GTFS feed: a folder of .txt files or a .zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for the two output CSVs and the run log.",
    )
    parser.add_argument(
        "--max-interval-minutes",
        type=float,
        default=MAX_INTERVAL_MINUTES,
        help="Flag intervals STRICTLY greater than this many scheduled minutes.",
    )
    parser.add_argument(
        "--service-ids",
        nargs="*",
        default=FILTER_SERVICE_IDS,
        help="service_ids (schedules) to review; omit to review all schedules.",
    )
    parser.add_argument(
        "--routes-include",
        nargs="*",
        default=FILTER_IN_ROUTES,
        help="Routes to keep, matched on route_id or route_short_name; omit to keep all.",
    )
    parser.add_argument(
        "--routes-exclude",
        nargs="*",
        default=FILTER_OUT_ROUTES,
        help="Routes to drop, matched on route_id or route_short_name.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Args:
        argv: Optional explicit argument list (used by tests); ``None``
            reads ``sys.argv`` outside notebook kernels.

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

    if args.gtfs == r"Path\To\Your\GTFS_Folder" or args.output_dir == r"Path\To\Your\Output_Folder":
        logging.warning(
            "GTFS_PATH and/or OUTPUT_DIR are still set to placeholder values. "
            "Update the CONFIGURATION section (or pass --gtfs / --output-dir) before running."
        )
        return 2

    try:
        run(
            gtfs_path=args.gtfs,
            output_dir=Path(args.output_dir).expanduser(),
            max_interval_minutes=args.max_interval_minutes,
            service_ids=args.service_ids,
            routes_include=args.routes_include,
            routes_exclude=args.routes_exclude,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
