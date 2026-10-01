"""Generate printable Excel schedules for each vehicle block in a GTFS feed.

The script reads the five core GTFS tables—``trips``, ``stop_times``, ``stops``,
``routes`` and ``calendar``—and optionally filters them by *service ID* and/or
*route short name*.  For every vehicle ``block_id`` that survives filtering it
produces a nicely-formatted ``.xlsx`` file ready for field auditing.

Outputs
-------
- ``block_<block_id>_<routes>_<HHMM>-<HHMM>_<calendar>.xlsx`` (one per surviving
  block, written to ``BASE_OUTPUT_PATH``; e.g. ``block_101_10_20_0532-1847_Weekday.xlsx``):
  the block's stop-by-stop schedule with placeholder columns for handwritten
  field notes. Set ``TIMEPOINTS_ONLY`` to list only timepoint stops; otherwise
  all stops are listed and timepoint rows are shaded.

Typical usage
-------------
Run from the command line, an ArcGIS Pro Python toolbox, or a notebook.

Key Features
------------
- Loads GTFS text files into ``pandas`` DataFrames with robust error handling.
- Converts ``HH:MM(:SS)`` time strings to seconds (and back) safely.
- Applies ergonomic Excel formatting via ``openpyxl`` (column widths, wrapping).
- Inserts placeholders for handwritten field notes (actual time, boardings, etc.).
- Widens the Comments column and shades timepoint rows for easier field use.
- Names each file by block, routes, time span and calendar for easy sorting by hand.
"""

from __future__ import annotations

import logging
import math
import os
import re
import zipfile
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Optional, Union

import pandas as pd
from openpyxl.styles import Alignment, PatternFill
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

_DEFAULT_GTFS_FOLDER_PATH = r"Path\To\Your\Input\Folder"
_DEFAULT_BASE_OUTPUT_PATH = r"Path\To\Your\Output\Folder"

GTFS_FOLDER_PATH = _DEFAULT_GTFS_FOLDER_PATH  # <<< EDIT HERE
BASE_OUTPUT_PATH = _DEFAULT_BASE_OUTPUT_PATH  # <<< EDIT HERE

REQUIRED_GTFS_FILES = [
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
    "routes.txt",
    "calendar.txt",
]

# If you only want certain service IDs or route short names, specify them here:
FILTER_SERVICE_IDS: list[str] = []  # e.g. ["WKD", "SAT"]
FILTER_ROUTE_SHORT_NAMES: list[str] = []  # e.g. ["101", "202"]

# Calendar label used in output file names, by service_id. Unlisted service_ids
# are labeled from their calendar.txt days (e.g. "Weekday", "Saturday", "MonTueWedThu").
SERVICE_LABEL_OVERRIDES: dict[str, str] = {}  # e.g. {"4": "Weekday", "5": "Saturday"}

# Stops to list on each block sheet:
TIMEPOINTS_ONLY: bool = False  # True → timepoint stops only; False → all stops

# Placeholder values for printing:
MISSING_TIME = "________"
MISSING_VALUE = "_____"
COMMENTS_PLACEHOLDER = "_" * 50  # Roughly fills COMMENTS_COLUMN_WIDTH; adjust together

# Maximum column width for neat Excel formatting:
MAX_COLUMN_WIDTH = 35

# Width of the Comments column (not capped by MAX_COLUMN_WIDTH), for handwritten notes:
COMMENTS_COLUMN_WIDTH: int = 60

# Fill color (hex RGB) for timepoint rows when listing all stops; "" = no highlight.
# Light gray stays visible on black-and-white printers.
TIMEPOINT_HIGHLIGHT_COLOR: str = "D9D9D9"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def time_to_seconds(time_str: str) -> Union[float, int]:
    """Convert a ``HH:MM`` or ``HH:MM:SS`` string to total seconds.

    Args:
        time_str: Time string *or* ``NaN``; may exceed 24 h (e.g.,
            ``'25:10:00'`` → 1:10 a.m. next day).

    Returns:
        Non-negative number of seconds, or :pydata:`math.nan` on failure.
    """
    if pd.isna(time_str):
        return math.nan

    parts = time_str.strip().split(":")
    if len(parts) < 2:
        return math.nan

    try:
        hours = int(parts[0]) % 24  # Roll over hours >= 24
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return math.nan

    return hours * 3600 + minutes * 60 + seconds


def format_hhmm(total_seconds: Union[int, float]) -> str:
    """Render seconds since midnight as a ``HH:MM`` string.

    Args:
        total_seconds: Seconds since 00:00.  Negative or ``NaN`` returns
            an empty string.

    Returns:
        Two-digit hour and minute representation (24-hour clock).
    """
    if pd.isna(total_seconds) or total_seconds < 0:
        return ""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------


def export_to_excel(data_frame: pd.DataFrame, output_file: str) -> None:
    """Write *data_frame* to an Excel file with basic styling.

    The sheet is named **Schedule** and receives:

    * Left-aligned cells.
    * Word-wrapped headers.
    * Column widths sized to longest cell (capped by ``MAX_COLUMN_WIDTH``),
      except **Comments**, which is set to ``COMMENTS_COLUMN_WIDTH``.
    * Rows with ``Timepoint == 1`` filled with ``TIMEPOINT_HIGHLIGHT_COLOR``
      (skipped when ``TIMEPOINTS_ONLY`` is set, since every row qualifies).

    Args:
        data_frame: Tidy table to export; must be non-empty.
        output_file: Full path of the ``.xlsx`` file to create.

    Notes:
        ``os.makedirs`` is called with *exist_ok=True* so nested output
        folders are created automatically.
    """
    if data_frame.empty:
        logging.info("No data to export to %s", output_file)
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write DataFrame to Excel and access the worksheet object for formatting
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        data_frame.to_excel(writer, index=False, sheet_name="Schedule")
        worksheet = writer.sheets["Schedule"]

        # Adjust columns
        for col_i, col_name in enumerate(data_frame.columns, 1):
            col_letter = get_column_letter(col_i)

            # Header alignment and text wrap
            header_cell = worksheet[f"{col_letter}1"]
            header_cell.alignment = Alignment(horizontal="left", wrap_text=True)

            # Data alignment
            for row_i in range(2, worksheet.max_row + 1):
                cell = worksheet[f"{col_letter}{row_i}"]
                cell.alignment = Alignment(horizontal="left")

            # Set column width based on max content length, capped at MAX_COLUMN_WIDTH
            max_len = max(len(str(col_name)), 10)  # Minimum width
            for row_i in range(2, worksheet.max_row + 1):
                val = worksheet[f"{col_letter}{row_i}"].value
                if val is not None:
                    max_len = max(max_len, len(str(val)))
            width = min(max_len + 2, MAX_COLUMN_WIDTH)
            if col_name == "Comments":
                width = COMMENTS_COLUMN_WIDTH
            worksheet.column_dimensions[col_letter].width = width

        # Shade timepoint rows (row 1 is the header, so data row N is at N + 2)
        if TIMEPOINT_HIGHLIGHT_COLOR and not TIMEPOINTS_ONLY and "Timepoint" in data_frame.columns:
            timepoint_fill = PatternFill("solid", fgColor=TIMEPOINT_HIGHLIGHT_COLOR)
            is_timepoint = (data_frame["Timepoint"] == 1).tolist()
            for pos, flag in enumerate(is_timepoint):
                if flag:
                    for cell in worksheet[pos + 2]:
                        cell.fill = timepoint_fill

    logging.info("Exported: %s", output_file)


def filter_data(
    trips_df: pd.DataFrame, stop_times_df: pd.DataFrame, routes_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply route and service filters, propagating them to stop times.

    Args:
        trips_df: Parsed **trips.txt** table.
        stop_times_df: Parsed **stop_times.txt** table.
        routes_df: Parsed **routes.txt** table (for ``route_short_name``).

    Returns:
        ``(filtered_trips, filtered_stop_times)``.

    Raises:
        KeyError: If required columns are missing.

    Warning:
        If the global constants ``FILTER_ROUTE_SHORT_NAMES`` or
        ``FILTER_SERVICE_IDS`` remove every trip, the function returns
        two **empty** DataFrames.
    """
    # Merge route_short_name into trips
    routes_subset = routes_df[["route_id", "route_short_name"]]
    trips_df = trips_df.merge(routes_subset, on="route_id", how="left")

    # Apply Route Filtering
    if FILTER_ROUTE_SHORT_NAMES:
        blocks_for_selected_routes = (
            trips_df[trips_df["route_short_name"].isin(FILTER_ROUTE_SHORT_NAMES)]["block_id"]
            .dropna()
            .unique()
        )
        if len(blocks_for_selected_routes) == 0:
            logging.info("No blocks found with the specified route short names.")
            return pd.DataFrame(), pd.DataFrame()

        trips_df = trips_df[trips_df["block_id"].isin(blocks_for_selected_routes)]

    # Apply Service ID Filtering
    if FILTER_SERVICE_IDS:
        trips_df = trips_df[trips_df["service_id"].isin(FILTER_SERVICE_IDS)]

    # Filter stop_times to only include relevant trips
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]

    return trips_df, stop_times_df


def prepare_stop_times(
    trips_df: pd.DataFrame, stop_times_df: pd.DataFrame, stops_df: pd.DataFrame
) -> pd.DataFrame:
    """Enrich and tidy ``stop_times`` for Excel export.

    Steps
    -----
    1. Ensure a numeric ``timepoint`` column (create if absent).
    2. Attach ``block_id``, ``route_short_name``, ``direction_id`` and ``service_id``.
    3. Convert arrival/departure times → seconds → ``HH:MM`` format.
    4. Map ``stop_id`` → human-readable stop names.
    5. Sort by ``block_id``, ``trip_id``, ``stop_sequence``.

    Args:
        trips_df: Output of :pyfunc:`filter_data`.
        stop_times_df: Ditto.
        stops_df: Parsed **stops.txt** table.

    Returns:
        Cleaned ``stop_times`` DataFrame ready for grouping by block.
    """
    # If 'timepoint' does not exist, create a new column with 0.
    if "timepoint" not in stop_times_df.columns:
        stop_times_df["timepoint"] = 0
    else:
        # Convert to numeric, fill NaN with 0
        stop_times_df["timepoint"] = (
            pd.to_numeric(stop_times_df["timepoint"], errors="coerce").fillna(0).astype(int)
        )

    # Merge essential trip columns into stop_times
    needed_trip_cols = ["trip_id", "block_id", "route_short_name", "direction_id", "service_id"]
    stop_times_df = stop_times_df.merge(trips_df[needed_trip_cols], on="trip_id", how="left")

    # Convert arrival/departure times to seconds and format
    stop_times_df["arrival_seconds"] = stop_times_df["arrival_time"].apply(time_to_seconds)
    stop_times_df["departure_seconds"] = stop_times_df["departure_time"].apply(time_to_seconds)
    stop_times_df["scheduled_time_hhmm"] = stop_times_df["departure_seconds"].apply(format_hhmm)

    # Merge in stop names
    stop_name_map = stops_df.set_index("stop_id")["stop_name"].to_dict()
    stop_times_df["stop_name"] = stop_times_df["stop_id"].map(stop_name_map).fillna("Unknown Stop")

    # Sort by block, trip, and stop_sequence
    stop_times_df = stop_times_df.dropna(subset=["block_id"])
    stop_times_df["stop_sequence"] = pd.to_numeric(stop_times_df["stop_sequence"], errors="coerce")
    stop_times_df = stop_times_df.dropna(subset=["stop_sequence"])
    stop_times_df = stop_times_df.sort_values(["block_id", "trip_id", "stop_sequence"])

    return stop_times_df


def service_day_label(calendar_row: Mapping[Hashable, Any]) -> str:
    """Summarize the day-of-week flags of one **calendar.txt** row.

    Args:
        calendar_row: One calendar record with ``monday`` … ``sunday`` flags.

    Returns:
        ``"Weekday"``, ``"Saturday"``, ``"Sunday"``, ``"Weekend"`` or ``"Daily"``
        for those common patterns; otherwise the served days abbreviated and
        joined (e.g. ``"MonTueWedThu"``), or ``""`` if no day is flagged.
    """
    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    served = tuple(day for day in days if str(calendar_row.get(day, "")).strip() == "1")
    named = {
        days[:5]: "Weekday",
        ("saturday",): "Saturday",
        ("sunday",): "Sunday",
        ("saturday", "sunday"): "Weekend",
        days: "Daily",
    }
    return named.get(served, "".join(day[:3].capitalize() for day in served))


def build_service_labels(calendar_df: pd.DataFrame) -> dict[str, str]:
    """Map each ``service_id`` to the calendar label used in file names.

    ``SERVICE_LABEL_OVERRIDES`` wins; otherwise the label comes from
    :pyfunc:`service_day_label`, falling back to the ``service_id`` itself
    when no day is flagged.

    Args:
        calendar_df: Parsed **calendar.txt** table.

    Returns:
        Mapping of ``service_id`` → label, e.g. ``{"4": "Weekday"}``.
    """
    labels = {
        str(row["service_id"]): service_day_label(row) or str(row["service_id"])
        for row in calendar_df.to_dict("records")
    }
    labels.update(SERVICE_LABEL_OVERRIDES)
    return labels


def build_block_filename(block_df: pd.DataFrame, service_labels: Mapping[str, str]) -> str:
    """Build a human-readable ``.xlsx`` file name for one block.

    Format: ``block_<block_id>_<route>_<route>…_<HHMM>-<HHMM>_<calendar>.xlsx``,
    e.g. ``block_101_10_20_0532-1847_Weekday.xlsx``.

    * Routes are listed in the order the block first serves them.
    * Times span the block's earliest and latest scheduled times; hours of
      24 or more are kept for after-midnight service, GTFS-style (``2530``).
    * Several calendars on one block are joined with ``+``.
    * Characters not allowed in Windows file names, and spaces, become ``-``.

    Args:
        block_df: All stop times of one block (before any timepoint filter),
            with ``block_id``, ``route_short_name``, ``service_id``,
            ``arrival_time`` and ``departure_time`` columns.
        service_labels: Output of :pyfunc:`build_service_labels`; service_ids
            missing from it are shown as-is.

    Returns:
        The file name (no directory).
    """
    departures = pd.to_numeric(block_df["departure_time"].map(parse_time_to_minutes))
    arrivals = pd.to_numeric(block_df["arrival_time"].map(parse_time_to_minutes))
    all_minutes = pd.concat([departures, arrivals])
    start = minutes_to_hhmm(all_minutes.min()).replace(":", "")
    end = minutes_to_hhmm(all_minutes.max()).replace(":", "")

    routes = departures.groupby(block_df["route_short_name"]).min().sort_values().index
    service_ids = sorted(block_df["service_id"].dropna().unique())
    calendars = dict.fromkeys(service_labels.get(sid, sid) for sid in service_ids)

    parts = [
        "block",
        str(block_df["block_id"].iloc[0]),
        *(str(route) for route in routes),
        f"{start}-{end}" if start else "",
        "+".join(calendars),
    ]
    safe_parts = [re.sub(r'[\\/:*?"<>|\s]+', "-", part.strip()) for part in parts if part.strip()]
    return "_".join(safe_parts) + ".xlsx"


def export_blocks(
    stop_times_df: pd.DataFrame, service_labels: Optional[Mapping[str, str]] = None
) -> None:
    """Generate one Excel schedule per vehicle block.

    Args:
        stop_times_df: Prepared stop times (see
            :pyfunc:`prepare_stop_times`).  Must include the columns
            produced earlier (``block_id``, ``scheduled_time_hhmm``,
            etc.).
        service_labels: ``service_id`` → calendar label for file names (see
            :pyfunc:`build_service_labels`). ``None`` shows raw service_ids.

    Side Effects:
        Writes one file per block, named by :pyfunc:`build_block_filename`, to
        ``BASE_OUTPUT_PATH``; creates the folder tree if needed. When
        ``TIMEPOINTS_ONLY`` is set, only rows with ``timepoint == 1`` are
        written, and blocks with no timepoint stops are skipped.
    """
    all_blocks = stop_times_df["block_id"].unique()
    logging.info("Found %d blocks to export.\n", len(all_blocks))

    for block_id in all_blocks:
        block_subset = stop_times_df[stop_times_df["block_id"] == block_id].copy()
        if block_subset.empty:
            continue

        # For each trip_id within this block, find earliest departure
        first_departures = (
            block_subset.groupby("trip_id")["departure_seconds"]
            .min()
            .reset_index(name="trip_start_seconds")
        )
        first_departures["trip_start_hhmm"] = first_departures["trip_start_seconds"].apply(
            format_hhmm
        )
        block_subset = block_subset.merge(first_departures, on="trip_id", how="left")

        block_subset["Trip Start Time"] = block_subset["trip_start_hhmm"]
        filename = build_block_filename(block_subset, service_labels or {})

        # Filter after the trip start times and file name so they still cover every stop
        if TIMEPOINTS_ONLY:
            block_subset = block_subset[block_subset["timepoint"] == 1]
            if block_subset.empty:
                logging.info("Block %s has no timepoint stops – skipped.", block_id)
                continue

        # Select and rename columns for clarity
        out_cols = [
            "block_id",
            "route_short_name",
            "direction_id",
            "trip_id",
            "Trip Start Time",
            "stop_sequence",
            "timepoint",
            "stop_id",
            "stop_name",
            "scheduled_time_hhmm",
        ]
        final_df = block_subset[out_cols].copy()
        final_df = final_df.rename(
            columns={
                "block_id": "Block ID",
                "route_short_name": "Route",
                "direction_id": "Direction",
                "trip_id": "Trip ID",
                "stop_sequence": "Stop Sequence",
                "timepoint": "Timepoint",
                "stop_id": "Stop ID",
                "stop_name": "Stop Name",
                "scheduled_time_hhmm": "Scheduled Time",
            },
        )

        # Insert placeholders
        final_df["Actual Time"] = MISSING_TIME
        final_df["Boardings"] = MISSING_VALUE
        final_df["Alightings"] = MISSING_VALUE
        final_df["Comments"] = COMMENTS_PLACEHOLDER

        # Reorder columns to place 'Timepoint' after 'Stop Sequence'
        final_df = final_df[
            [
                "Block ID",
                "Route",
                "Direction",
                "Trip ID",
                "Trip Start Time",
                "Stop Sequence",
                "Timepoint",
                "Stop ID",
                "Stop Name",
                "Scheduled Time",
                "Actual Time",
                "Boardings",
                "Alightings",
                "Comments",
            ]
        ]

        final_df = final_df.sort_values(by=["Trip Start Time", "Trip ID", "Stop Sequence"])

        output_path = os.path.join(BASE_OUTPUT_PATH, filename)
        export_to_excel(final_df, output_path)


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


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


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Command-line entry point.

    Orchestrates:

    * Logging configuration.
    * Data ingestion via :pyfunc:`load_gtfs_data`.
    * Optional filtering (:pyfunc:`filter_data`).
    * Data preparation (:pyfunc:`prepare_stop_times`).
    * Per-block Excel export (:pyfunc:`export_blocks`).

    The function traps anticipated exceptions and logs them with useful
    context before exiting with a non-zero status.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    using_defaults = False
    if GTFS_FOLDER_PATH == _DEFAULT_GTFS_FOLDER_PATH:
        logging.warning(
            "GTFS_FOLDER_PATH is still the default placeholder – update it before running: %s",
            _DEFAULT_GTFS_FOLDER_PATH,
        )
        using_defaults = True
    if BASE_OUTPUT_PATH == _DEFAULT_BASE_OUTPUT_PATH:
        logging.warning(
            "BASE_OUTPUT_PATH is still the default placeholder – update it before running: %s",
            _DEFAULT_BASE_OUTPUT_PATH,
        )
        using_defaults = True
    if using_defaults:
        logging.info("No processing performed. Update the placeholder paths above and re-run.")
        return 2

    logging.info("========================================================")
    logging.info("GTFS Block Schedule Printable Generator")
    logging.info("Input GTFS Folder: %s", GTFS_FOLDER_PATH)
    logging.info("Output Folder:     %s", BASE_OUTPUT_PATH)
    if FILTER_ROUTE_SHORT_NAMES:
        logging.info("Filtering for Routes: %s", FILTER_ROUTE_SHORT_NAMES)
    if FILTER_SERVICE_IDS:
        logging.info("Filtering for Service IDs: %s", FILTER_SERVICE_IDS)
    if TIMEPOINTS_ONLY:
        logging.info("Listing timepoint stops only.")

    try:
        gtfs_data = load_gtfs_data(
            gtfs_path=GTFS_FOLDER_PATH,
            files=REQUIRED_GTFS_FILES,
            dtype=str,
        )

        trips_df = gtfs_data["trips"]
        stop_times_df = gtfs_data["stop_times"]
        stops_df = gtfs_data["stops"]
        routes_df = gtfs_data["routes"]
        service_labels = build_service_labels(gtfs_data["calendar"])

        trips_df, stop_times_df = filter_data(trips_df, stop_times_df, routes_df)
        if trips_df.empty or stop_times_df.empty:
            logging.warning("No data remains after filtering – no files generated.")
            return 1

        prepared = prepare_stop_times(trips_df, stop_times_df, stops_df)
        if prepared.empty:
            logging.warning("No data remains after preparation – no files generated.")
            return 1
        if TIMEPOINTS_ONLY and not (prepared["timepoint"] == 1).any():
            logging.warning(
                "TIMEPOINTS_ONLY is True but no stops have timepoint = 1 in stop_times.txt "
                "– no files generated. Set TIMEPOINTS_ONLY = False to list all stops."
            )
            return 1

        export_blocks(prepared, service_labels)
        logging.info("Script finished successfully.")
        logging.info("Script completed successfully.")
        return 0

    except (OSError, ValueError) as err:
        logging.error("%s", err)
        return 1
    except Exception as err:  # catch-all for unforeseen issues
        logging.exception("Unexpected error: %s", err)
        return 1
    finally:
        logging.info("Exiting script.")


if __name__ == "__main__":
    raise SystemExit(main())
