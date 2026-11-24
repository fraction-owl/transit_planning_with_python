"""Generate printable Excel schedules for each vehicle block in a GTFS feed.

The script reads the five core GTFS tables—``trips``, ``stop_times``, ``stops``,
``routes`` and ``calendar``—and optionally filters them by *service ID* (or the
service IDs active on one *service date*) and/or *route short name*.  For every
vehicle ``block_id`` that survives filtering it produces a nicely-formatted
``.xlsx`` file ready for field auditing. A frequency-based trip
(``frequencies.txt``) on a selected block stops the export, since a block sheet
cannot list each of its runs.

Outputs
-------
- ``block_<block_id>_<routes>_<HHMM>-<HHMM>_<calendar>.xlsx`` (one per surviving
  block/service ID, written to ``BASE_OUTPUT_PATH``):
  the block's stop-by-stop schedule with placeholder columns for handwritten
  field notes. Set ``TIMEPOINTS_ONLY`` to list only timepoint stops; otherwise
  all stops are listed and timepoint rows are shaded.
  Blocks shared by service IDs are exported separately, with a service-ID
  suffix on their file names. With ``SERVICE_DATE`` set, each block's trips on
  that date share one file, named with the date, and layovers connect trips of
  different service IDs. Optional layover rows are enabled by default.

Typical usage
-------------
Run from the command line, an ArcGIS Pro Python toolbox, or a notebook.

Key Features
------------
- Loads GTFS text files into ``pandas`` DataFrames with robust error handling.
- Converts ``HH:MM(:SS)`` time strings to seconds (and back) safely.
- Applies ergonomic Excel formatting via ``openpyxl`` (column widths, wrapping)
  and a print layout that fits every column on one page width (11 x 17
  landscape by default).
- Lists scheduled arrival and departure separately and provides separate blanks
  for actual arrival and departure, along with boardings, alightings and notes.
- Widens the Comments column and shades timepoint rows for easier field use.
- Fills the Stop ID column from ``stop_code`` (default) or ``stop_id``.
- Names each file by block, routes, time span and calendar for easy sorting by hand.
- Marks boardings at the last stop and alightings at the first stop of each trip.
- Prints collector Name/Date fields and the generated workbook's full file path
  in Excel headers/footers, outside the data table and repeated on every page.
- Adds a writable row for positive gaps between trips, with scheduled times,
  duration, actual times, and location/activity notes. Different endpoint stops
  are labeled "Between trips" because the gap may include travel.
- With ``TODS_PATH`` (TODS 2.1 supplement files for the same feed, e.g. from
  ``tsde_to_tods.py``), applies their edits and deletions (a corrected
  departure prints at its new time; trips on a deleted route are dropped),
  lists each block's scheduled pull-out, pull-in and deadheads with the waits
  around them, and splits a gap that a deadhead covers into the deadhead and
  the layover before or after it, so "Between trips" is left only for gaps the
  TODS files do not explain. File names span the pull-out to the pull-in.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import re
import zipfile
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Optional, Union

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# REQUIRED: Paste your paths between the quotes on these two lines.
# Keep the r prefix and omit any trailing backslash.
GTFS_FOLDER_PATH = r""  # Input folder containing GTFS .txt files, or a GTFS .zip file.
BASE_OUTPUT_PATH = r""  # Output folder for the Excel files; created if needed.

REQUIRED_GTFS_FILES = [
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
    "routes.txt",
    "calendar.txt",
]

# If you only want certain service IDs or route short names, specify them here.
# A route filter keeps each block (per service ID, or on SERVICE_DATE) that serves
# one of the routes on the selected service, including its interlined trips on
# other routes.
FILTER_SERVICE_IDS: list[str] = []  # e.g. ["WKD", "SAT"]
FILTER_ROUTE_SHORT_NAMES: list[str] = []  # e.g. ["101", "202"]

# Optional service date (YYYYMMDD), used instead of FILTER_SERVICE_IDS. Every
# service_id active on that date per calendar.txt and calendar_dates.txt is
# selected, and each block's trips from all of them go on one sheet: the
# vehicle's whole day, with layovers between trips of different service IDs.
# Leave "" to export each block once per service ID.
SERVICE_DATE: str = ""  # e.g. "20260105"

# Calendar label used in output file names, by service_id. Unlisted service_ids
# are labeled from their calendar.txt days (e.g. "Weekday", "Saturday", "MonTueWedThu").
SERVICE_LABEL_OVERRIDES: dict[str, str] = {}  # e.g. {"4": "Weekday", "5": "Saturday"}

# Stops to list on each block sheet:
TIMEPOINTS_ONLY: bool = False  # True → timepoint stops only; False → all stops

# stops.txt field printed in the "Stop ID" column: "stop_code" or "stop_id".
# Falls back to stop_id if stops.txt has no stop_code column.
STOP_ID_FIELD: str = "stop_code"

# Add observation rows for positive scheduled gaps between trips in a block.
INCLUDE_LAYOVER_ROWS: bool = True  # False → stop events only.

# Optional: TODS 2.1 supplement files for the same GTFS feed (folder or .zip).
# Their edits, additions and deletions to trips, stop times, stops and routes are
# applied first; then each block's pull-out, pull-in and deadhead rows (trips with
# a TODS_trip_type) are listed and deadhead-covered gaps split into deadhead +
# layover. TODS v1 deadhead tables stop the run. Blank: GTFS only.
TODS_PATH = r""

# Placeholder values for printing:
MISSING_TIME = "________"
MISSING_VALUE = "_____"
BLOCKED_COUNT_MARKER: str = "X"  # Use "-" instead if preferred; means not applicable.
COMMENTS_PLACEHOLDER = "_" * 50  # Roughly fills COMMENTS_COLUMN_WIDTH; adjust together

# Maximum column width for neat Excel formatting:
MAX_COLUMN_WIDTH = 35

# Width of the Comments column (not capped by MAX_COLUMN_WIDTH), for handwritten notes:
COMMENTS_COLUMN_WIDTH: int = 60

# Fill color (hex RGB) for timepoint rows when listing all stops; "" = no highlight.
# Light gray stays visible on black-and-white printers.
TIMEPOINT_HIGHLIGHT_COLOR: str = "D9D9D9"

# Layover/between-trip rows have a border and extra writing space.
LAYOVER_HIGHLIGHT_COLOR: str = "E7E6E6"

# Print layout saved in each workbook; all columns are scaled to fit one page wide.
# Excel paper size codes: 1 = Letter, 3 = Tabloid (11 x 17), 5 = Legal.
PRINT_PAPER_SIZE: int = 3
PRINT_LANDSCAPE: bool = True

# Native print header/footer fields; these do not add rows or columns to Schedule.
# Visible in Excel's Page Layout view, Print Preview, and printed/PDF output.
COLLECTOR_NAME_HEADER: str = "Name: ________________________________________"
COLLECTION_DATE_HEADER: str = "Date: ____________________"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

# GTFS files loaded only when the feed has them: calendar_dates.txt for SERVICE_DATE,
# frequencies.txt to stop on frequency-based trips.
OPTIONAL_GTFS_FILES: tuple[str, ...] = ("calendar_dates.txt", "frequencies.txt")

# TODS_trip_type -> Event Type label; other types are shown capitalized.
TODS_EVENT_LABELS: dict[str, str] = {
    "pull-out": "Pull-out",
    "pull-back": "Pull-in",
    "pull-in": "Pull-in",
    "deadhead": "Deadhead",
}

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
    if not isinstance(time_str, str):
        return math.nan

    parts = time_str.strip().split(":")
    if len(parts) not in (2, 3):
        return math.nan

    try:
        hours = int(parts[0])  # Preserve the GTFS service day, including hours >= 24.
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return math.nan

    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return math.nan

    return hours * 3600 + minutes * 60 + seconds


def format_hhmm(total_seconds: Union[int, float]) -> str:
    """Render seconds since midnight as a ``HH:MM`` string.

    Args:
        total_seconds: Seconds since 00:00.  Negative or ``NaN`` returns
            an empty string.

    Returns:
        Zero-padded hour and minute representation; GTFS hours >= 24 are preserved.
    """
    if pd.isna(total_seconds) or total_seconds < 0:
        return ""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------


def header_width(header: str) -> int:
    """Return the shortest line length that fits *header* on at most two lines.

    Args:
        header: Column header text; it may wrap only between words.

    Returns:
        Characters in the longer line of the best two-line split (e.g. 10 for
        ``"Trip Start Time"`` → ``"Trip Start"`` / ``"Time"``), or the full
        length for a one-word header.
    """
    words = str(header).split()
    if len(words) < 2:
        return len(str(header))
    return min(
        max(len(" ".join(words[:i])), len(" ".join(words[i:]))) for i in range(1, len(words))
    )


def export_to_excel(data_frame: pd.DataFrame, output_file: str) -> None:
    """Write *data_frame* to an Excel file with basic styling.

    The sheet is named **Schedule** and receives:

    * Left-aligned cells.
    * Headers wrapped onto up to two lines, so columns are as narrow as their
      data allows (see :pyfunc:`header_width`).
    * Column widths sized to the longest value (capped by ``MAX_COLUMN_WIDTH``),
      except **Comments**, which is set to ``COMMENTS_COLUMN_WIDTH``.
    * Rows with ``Timepoint == 1`` filled with ``TIMEPOINT_HIGHLIGHT_COLOR``
      (skipped when ``TIMEPOINTS_ONLY`` is set, since every row qualifies).
    * A print layout of ``PRINT_PAPER_SIZE`` / ``PRINT_LANDSCAPE`` with all
      columns fitted to one page wide and the header row repeated on every page.
    * Collector Name/Date lines in the print header and the absolute output
      workbook path in the print footer on every page. These are stored outside
      worksheet cells, leaving row 1 and the data-table schema unchanged.
    * Bold, centered ``BLOCKED_COUNT_MARKER`` cells for inapplicable counts.
    * Taller, outlined layover/between-trip rows, with wrapped location and
      activity prompts. ``Event Type`` distinguishes these from stop records.

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

    output_file = os.path.abspath(output_file)
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
                if col_name in ("Boardings", "Alightings") and cell.value == BLOCKED_COUNT_MARKER:
                    cell.alignment = Alignment(horizontal="center")
                    cell.font = Font(bold=True)

            # Size to the longest value (header may wrap), capped at MAX_COLUMN_WIDTH
            max_len = header_width(col_name)
            for row_i in range(2, worksheet.max_row + 1):
                val = worksheet[f"{col_letter}{row_i}"].value
                if val is not None:
                    max_len = max(max_len, *(len(line) for line in str(val).split("\n")))
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

        # Extra writing room for between-trip observations; no merged table cells.
        if "Event Type" in data_frame.columns:
            gap_fill = (
                PatternFill("solid", fgColor=LAYOVER_HIGHLIGHT_COLOR)
                if LAYOVER_HIGHLIGHT_COLOR
                else None
            )
            gap_border = Border(top=Side(style="thin"), bottom=Side(style="thin"))
            for pos, event_type in enumerate(data_frame["Event Type"]):
                if event_type == "Stop":
                    continue
                row_i = pos + 2
                worksheet.row_dimensions[row_i].height = 60
                for cell in worksheet[row_i]:
                    cell.border = gap_border
                    if gap_fill is not None:
                        cell.fill = gap_fill
                    cell.alignment = Alignment(
                        horizontal=cell.alignment.horizontal, vertical="top", wrap_text=True
                    )
                event_column = data_frame.columns.tolist().index("Event Type") + 1
                worksheet.cell(row_i, event_column).font = Font(bold=True)

        # Two-line header row (15 pt per line of the default 11 pt font)
        worksheet.row_dimensions[1].height = 30

        # Print layout: one page wide, as many pages tall as needed, header on every page
        worksheet.page_setup.paperSize = PRINT_PAPER_SIZE
        worksheet.page_setup.orientation = "landscape" if PRINT_LANDSCAPE else "portrait"
        worksheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        worksheet.page_setup.fitToWidth = 1
        worksheet.page_setup.fitToHeight = 0
        worksheet.print_title_rows = "1:1"

        # Native headers/footers stay outside the table ingested by pandas/Excel.
        # Odd settings apply to all pages when separate first/even pages are disabled.
        worksheet.HeaderFooter.differentFirst = False
        worksheet.HeaderFooter.differentOddEven = False
        worksheet.HeaderFooter.scaleWithDoc = False
        worksheet.HeaderFooter.alignWithMargins = True
        worksheet.oddHeader.left.text = COLLECTOR_NAME_HEADER.replace("&", "&&")
        worksheet.oddHeader.right.text = COLLECTION_DATE_HEADER.replace("&", "&&")
        for header_part in (worksheet.oddHeader.left, worksheet.oddHeader.right):
            header_part.size = 11
            header_part.font = "Calibri,Regular"

        # Save the actual generated path, rather than a field that changes if a
        # copy is moved. Escape ampersands because Excel uses them as control codes.
        worksheet.oddFooter.left.text = f"File: {output_file}".replace("&", "&&")
        worksheet.oddFooter.left.size = 8
        worksheet.oddFooter.left.font = "Calibri,Regular"
        worksheet.page_margins.header = 0.3
        worksheet.page_margins.footer = 0.3

    logging.info("Exported: %s", output_file)


def filter_data(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    date_service_ids: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply route and service filters, propagating them to stop times.

    Service IDs are filtered first: *date_service_ids* when given, otherwise
    ``FILTER_SERVICE_IDS``. The route filter then keeps every trip of each
    ``(block_id, service_id)`` schedule with at least one trip on a selected
    route, so interlined trips stay on the sheet. Blocks are matched per
    service ID because each is exported separately, and a ``block_id`` reused
    on another service may not serve the route there. With
    *date_service_ids*, every kept trip runs on that date, so blocks are
    matched on ``block_id`` alone.

    Args:
        trips_df: Parsed **trips.txt** table.
        stop_times_df: Parsed **stop_times.txt** table.
        routes_df: Parsed **routes.txt** table (for ``route_short_name``).
        date_service_ids: The service_ids active on ``SERVICE_DATE``, or
            ``None`` to select by ``FILTER_SERVICE_IDS``.

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

    # Apply Service ID Filtering
    if date_service_ids is not None:
        trips_df = trips_df[trips_df["service_id"].isin(list(date_service_ids))]
    elif FILTER_SERVICE_IDS:
        trips_df = trips_df[trips_df["service_id"].isin(FILTER_SERVICE_IDS)]

    # Apply Route Filtering to whole block/service (or block/date) schedules
    if FILTER_ROUTE_SHORT_NAMES:
        keys = ["block_id"] if date_service_ids is not None else ["block_id", "service_id"]
        serving = (
            trips_df.loc[trips_df["route_short_name"].isin(FILTER_ROUTE_SHORT_NAMES), keys]
            .dropna()
            .drop_duplicates()
        )
        if serving.empty:
            logging.info(
                "No blocks found with the specified route short names%s.",
                " on the selected service" if FILTER_SERVICE_IDS or date_service_ids else "",
            )
            return pd.DataFrame(), pd.DataFrame()

        on_serving_block = pd.MultiIndex.from_frame(trips_df[keys]).isin(
            pd.MultiIndex.from_frame(serving)
        )
        trips_df = trips_df[on_serving_block]

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
       ``direction_id`` is optional in GTFS; without it, Direction prints blank.
    3. Convert arrival/departure times → seconds → ``HH:MM`` format.
    4. Map ``stop_id`` → human-readable stop names and the ``stop_label``
       printed as Stop ID (per ``STOP_ID_FIELD``).
    5. Sort by ``block_id``, ``trip_id``, ``stop_sequence``.

    Args:
        trips_df: Output of :pyfunc:`filter_data`.
        stop_times_df: Ditto.
        stops_df: Parsed **stops.txt** table.

    Returns:
        Cleaned ``stop_times`` DataFrame ready for grouping by block.

    Raises:
        ValueError: If ``STOP_ID_FIELD`` is not ``"stop_code"`` or ``"stop_id"``.
    """
    if STOP_ID_FIELD not in ("stop_code", "stop_id"):
        raise ValueError(f"STOP_ID_FIELD must be 'stop_code' or 'stop_id', not {STOP_ID_FIELD!r}.")

    # If 'timepoint' does not exist, create a new column with 0.
    if "timepoint" not in stop_times_df.columns:
        stop_times_df["timepoint"] = 0
    else:
        # Convert to numeric, fill NaN with 0
        stop_times_df["timepoint"] = (
            pd.to_numeric(stop_times_df["timepoint"], errors="coerce").fillna(0).astype(int)
        )

    # Merge essential trip columns into stop_times
    if "direction_id" not in trips_df.columns:
        trips_df = trips_df.assign(direction_id="")
    needed_trip_cols = ["trip_id", "block_id", "route_short_name", "direction_id", "service_id"]
    stop_times_df = stop_times_df.merge(trips_df[needed_trip_cols], on="trip_id", how="left")

    # Convert arrival/departure times to seconds and format
    stop_times_df["arrival_seconds"] = stop_times_df["arrival_time"].apply(time_to_seconds)
    stop_times_df["departure_seconds"] = stop_times_df["departure_time"].apply(time_to_seconds)
    stop_times_df["scheduled_arrival_hhmm"] = stop_times_df["arrival_seconds"].apply(format_hhmm)
    stop_times_df["scheduled_departure_hhmm"] = stop_times_df["departure_seconds"].apply(
        format_hhmm
    )

    # Merge in stop names
    stop_name_map = stops_df.set_index("stop_id")["stop_name"].to_dict()
    stop_times_df["stop_name"] = stop_times_df["stop_id"].map(stop_name_map).fillna("Unknown Stop")

    # Stop identifier printed in the "Stop ID" column
    if STOP_ID_FIELD == "stop_code" and "stop_code" in stops_df.columns:
        stop_code_map = stops_df.set_index("stop_id")["stop_code"].to_dict()
        stop_times_df["stop_label"] = stop_times_df["stop_id"].map(stop_code_map).fillna("")
    else:
        if STOP_ID_FIELD == "stop_code":
            logging.warning("stops.txt has no stop_code column; printing stop_id as Stop ID.")
        stop_times_df["stop_label"] = stop_times_df["stop_id"]

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


def build_block_filename(
    block_df: pd.DataFrame,
    service_labels: Mapping[str, str],
    movements: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Build a human-readable ``.xlsx`` file name for one block.

    Format: ``block_<block_id>_<route>_<route>…_<HHMM>-<HHMM>_<calendar>.xlsx``,
    e.g. ``block_101_10_20_0532-1847_Weekday.xlsx``.

    * Routes are listed in the order the block first serves them.
    * Times span the block's earliest and latest scheduled times, including
      its TODS pull-out, pull-in and deadheads; hours of 24 or more are kept
      for after-midnight service, GTFS-style (``2530``).
    * Several calendars on one block are joined with ``+``.
    * Characters not allowed in Windows file names, and spaces, become ``-``.

    Args:
        block_df: All stop times of one block (before any timepoint filter),
            with ``block_id``, ``route_short_name``, ``service_id``,
            ``arrival_time`` and ``departure_time`` columns.
        service_labels: Output of :pyfunc:`build_service_labels`; service_ids
            missing from it are shown as-is.
        movements: This block's TODS movements (see
            :pyfunc:`build_tods_movements`), with ``start``/``end`` seconds.

    Returns:
        The file name (no directory).
    """
    departures = pd.to_numeric(block_df["departure_time"].map(parse_time_to_minutes))
    arrivals = pd.to_numeric(block_df["arrival_time"].map(parse_time_to_minutes))
    moves = pd.Series([m[k] / 60 for m in movements for k in ("start", "end")], dtype=float)
    all_minutes = pd.concat([departures, arrivals, moves])
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


TODS_V1_TABLES = ("deadheads.txt", "deadhead_times.txt")
TODS_SUPPLEMENT_FILES = (
    "trips_supplement.txt",
    "stop_times_supplement.txt",
    "stops_supplement.txt",
    "routes_supplement.txt",
)


def feed_members(path: str) -> set[str]:
    """Return the file names present in a feed folder or ``.zip`` (basenames).

    Raises:
        OSError: If *path* is neither a folder nor a readable ``.zip``.
    """
    if os.path.isdir(path):
        return {name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))}
    try:
        with zipfile.ZipFile(path) as archive:
            return {os.path.basename(name) for name in archive.namelist() if not name.endswith("/")}
    except (OSError, zipfile.BadZipFile) as exc:
        raise OSError(f"TODS_PATH '{path}' is neither a folder nor a readable .zip.") from exc


def load_tods_supplements(tods_path: str) -> dict[str, pd.DataFrame]:
    """Load the TODS 2.1 supplement files present in *tods_path*.

    Args:
        tods_path: Folder or ``.zip`` holding the ``*_supplement.txt`` files.

    Returns:
        File stem (e.g. ``"trips_supplement"``) to an all-text DataFrame.

    Raises:
        OSError: If *tods_path* cannot be read.
        ValueError: If it holds TODS v1 deadhead tables, or no supplement file.
    """
    members = feed_members(tods_path)
    found_v1 = sorted(set(TODS_V1_TABLES) & members)
    if found_v1:
        raise ValueError(
            f"'{tods_path}' contains TODS v1 deadhead tables ({', '.join(found_v1)}). This "
            "script expects TODS 2.1 supplements (trips_supplement.txt, "
            "stop_times_supplement.txt); convert the files to TODS 2.1 first."
        )
    present = [name for name in TODS_SUPPLEMENT_FILES if name in members]
    if not present:
        raise ValueError(f"No TODS supplement files found in '{tods_path}'.")
    tables = load_gtfs_data(tods_path, files=present, dtype=str)
    return {name: df.fillna("") for name, df in tables.items()}


def apply_tods_supplements(
    gtfs: Mapping[str, pd.DataFrame], supplements: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Apply TODS 2.1 supplements to the GTFS trips, stop times, stops and routes.

    Rows are deleted, updated or added by :func:`apply_supplement`, so a TODS
    edit to a public trip (a corrected departure, say) prints at its new time.
    Trips on a route deleted in ``routes_supplement.txt`` are then dropped
    unless ``trips_supplement.txt`` moved them to another route.

    Args:
        gtfs: ``trips``, ``stop_times``, ``stops`` and ``routes`` tables.
        supplements: Output of :func:`load_tods_supplements`.

    Returns:
        The four tables after the supplements, with blank text as missing
        values (as when GTFS is read without TODS). ``trips`` keeps a
        ``TODS_trip_type`` column, blank for public trips.
    """
    routes_before = gtfs["routes"].fillna("")
    out = {
        "trips": apply_supplement(gtfs["trips"], supplements.get("trips_supplement"), ["trip_id"]),
        "stop_times": apply_supplement(
            gtfs["stop_times"],
            supplements.get("stop_times_supplement"),
            ["trip_id", "stop_sequence"],
        ),
        "stops": apply_supplement(gtfs["stops"], supplements.get("stops_supplement"), ["stop_id"]),
        "routes": apply_supplement(
            routes_before, supplements.get("routes_supplement"), ["route_id"]
        ),
    }
    out["trips"] = drop_trips_on_deleted_routes(out["trips"], routes_before, out["routes"])
    if "TODS_trip_type" not in out["trips"].columns:
        out["trips"]["TODS_trip_type"] = ""
    return {name: df.mask(df == "") for name, df in out.items()}


def build_tods_movements(
    trips: pd.DataFrame, stop_times: pd.DataFrame, stops: pd.DataFrame
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Collect TODS nonrevenue trips, keyed by ``(block_id, service_id)``.

    Args:
        trips: Supplemented trips (see :func:`apply_tods_supplements`); only
            rows with a ``TODS_trip_type`` count.
        stop_times: Supplemented stop times.
        stops: Supplemented stops, used to name and label the end stops.

    Returns:
        Per block and service, the movements sorted by start: ``trip_id``,
        ``label`` (Event Type), ``start``/``end`` seconds and the ``from_*`` /
        ``to_*`` stop IDs, Stop ID labels (``from_label`` / ``to_label``, per
        ``STOP_ID_FIELD``) and names.

    Raises:
        ValueError: If a nonrevenue trip has no stop times or no end times.
    """
    stops = stops.fillna("")
    names = dict(zip(stops["stop_id"].astype(str), stops.get("stop_name", stops["stop_id"])))
    codes = dict(zip(stops["stop_id"].astype(str), stops.get("stop_code", "")))
    # Stop ID labels follow prepare_stop_times: stop_code if stops.txt has that column.
    use_codes = STOP_ID_FIELD == "stop_code" and "stop_code" in stops.columns

    def stop_label(stop_id: str) -> str:
        return codes.get(stop_id, "") if use_codes else stop_id

    trips = trips.fillna("")
    nonrevenue = trips.loc[trips["TODS_trip_type"].astype(str).str.strip() != ""]
    stop_times = stop_times.loc[stop_times["trip_id"].isin(set(nonrevenue["trip_id"]))].fillna("")
    stop_times = stop_times.assign(_seq=pd.to_numeric(stop_times["stop_sequence"]))
    calls = {trip: g.sort_values("_seq") for trip, g in stop_times.groupby("trip_id")}

    movements: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trip in nonrevenue.to_dict("records"):
        rows = calls.get(trip["trip_id"])
        if rows is None or rows.empty:
            raise ValueError(f"TODS trip {trip['trip_id']} has no stop times.")
        first, last = rows.iloc[0], rows.iloc[-1]
        start = time_to_seconds(first["departure_time"] or first["arrival_time"])
        end = time_to_seconds(last["arrival_time"] or last["departure_time"])
        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"TODS trip {trip['trip_id']} lacks a first or last stop time.")
        kind = str(trip["TODS_trip_type"]).strip()
        movements.setdefault((trip.get("block_id", ""), trip["service_id"]), []).append(
            {
                "trip_id": trip["trip_id"],
                "label": TODS_EVENT_LABELS.get(kind.lower(), kind[:1].upper() + kind[1:]),
                "start": start,
                "end": end,
                "from_stop": first["stop_id"],
                "to_stop": last["stop_id"],
                "from_label": stop_label(first["stop_id"]),
                "to_label": stop_label(last["stop_id"]),
                "from_name": names.get(first["stop_id"], first["stop_id"]),
                "to_name": names.get(last["stop_id"], last["stop_id"]),
            }
        )
    for legs in movements.values():
        legs.sort(key=lambda m: (m["start"], m["end"]))
    logging.info("Loaded %d TODS nonrevenue trip(s).", len(nonrevenue))
    return movements


def movements_on_date(
    movements: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    service_ids: Sequence[str],
    service_date: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Regroup TODS movements for a ``SERVICE_DATE`` export, keyed by ``(block_id, date)``.

    Args:
        movements: Output of :pyfunc:`load_tods_movements`.
        service_ids: The service_ids active on *service_date*; movements on
            other services are dropped.
        service_date: The date (``YYYYMMDD``) used as each key's second part.

    Returns:
        Each block's movements on the active services, sorted by start.
    """
    merged: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (block_id, service_id), legs in movements.items():
        if service_id in service_ids:
            merged.setdefault((block_id, service_date), []).extend(legs)
    for legs in merged.values():
        legs.sort(key=lambda m: (m["start"], m["end"]))
    return merged


def reject_frequency_trips(frequencies_df: Optional[pd.DataFrame], trip_ids: pd.Series) -> None:
    """Stop when a trip to be printed is frequency-based.

    frequencies.txt repeats a trip at a headway (with or without
    ``exact_times``), and its stop_times describe one template run, so a block
    sheet would list a single run and silently omit the rest. Frequency-based
    trips outside the selected blocks are ignored.

    Args:
        frequencies_df: Parsed ``frequencies.txt``, or ``None`` if the feed has none.
        trip_ids: The trip_id of every stop event to be printed.

    Raises:
        ValueError: If any of *trip_ids* appears in *frequencies_df*.
    """
    if frequencies_df is None or frequencies_df.empty or "trip_id" not in frequencies_df:
        return
    frequency_trips = set(frequencies_df["trip_id"].dropna().astype(str))
    selected = sorted(frequency_trips & set(trip_ids.astype(str)))
    if selected:
        listed = ", ".join(selected[:5]) + (", ..." if len(selected) > 5 else "")
        raise ValueError(
            f"{len(selected)} selected trip(s) are frequency-based ({listed}); a block sheet "
            "can list only one run of each. Expand them into scheduled trips, or exclude "
            "their blocks with FILTER_SERVICE_IDS, SERVICE_DATE or FILTER_ROUTE_SHORT_NAMES."
        )


def _duration_text(seconds: float) -> str:
    """Format a duration as ``"10 min"`` or ``"17m 30s"``."""
    if seconds % 60:
        return f"{int(seconds) // 60}m {int(seconds) % 60}s"
    return f"{seconds / 60:g} min"


def add_layover_rows(
    schedule_df: pd.DataFrame,
    block_df: pd.DataFrame,
    movements: Sequence[Mapping[str, Any]] = (),
    include_gaps: bool = True,
    service_date: str = "",
) -> pd.DataFrame:
    """Insert observation rows between adjacent trips in one block/service ID.

    Args:
        schedule_df: Formatted stop rows, possibly filtered to timepoints.
        block_df: All prepared stop events for this block and service ID,
            including ``trip_start_seconds`` before the timepoint filter.
        movements: This block's TODS nonrevenue trips (see
            :pyfunc:`load_tods_movements`), placed before the first trip,
            between trips or after the last trip by their scheduled times.
        include_gaps: Add rows for waiting time (``INCLUDE_LAYOVER_ROWS``).
        service_date: The date (``YYYYMMDD``) every trip in *block_df* runs
            on, which lets them span several service IDs; ``""`` otherwise.

    Returns:
        Stop rows with their positive scheduled gaps and TODS movements.
        ``Layover`` means the vehicle waits at one stop. ``Between trips`` means
        the endpoint stops differ and no TODS movement covers the gap, so it may
        include repositioning as well as waiting. A TODS movement row
        (``Pull-out``, ``Deadhead``, ``Pull-in`` ...) gives its scheduled
        departure from the first stop and arrival at the last; a gap it covers
        becomes its movements plus the waits before, between and after them.
        Each wait is placed where the preceding trip or movement left the
        vehicle, and is a ``Layover`` only if the next trip or movement starts
        at that stop; otherwise it is ``Between trips``.

    Notes:
        Gap limits are the preceding trip's last arrival and the following
        trip's first departure, with the other time at that same event used
        only if the preferred time is missing. Actual time fields record the
        observed waiting period; Comments records its location and activity.
        A gap row carries the preceding trip ID (the first trip's, for a wait
        between a pull-out and the first trip), with no stop sequence or
        timepoint flag. Without TODS there are no rows before the first or
        after the last trip, and no rows for zero or negative gaps. Endpoints
        come from full trips even when only timepoints are printed. Separate
        service IDs are connected only through *service_date*, the date on
        which all of them run.
    """
    if not service_date and len(block_df["service_id"].unique()) != 1:
        raise ValueError(
            "Layover rows require one service ID per block schedule, or a SERVICE_DATE "
            "on which all of its trips run."
        )
    schedule = (
        f"service date {service_date}"
        if service_date
        else f"service {block_df['service_id'].iloc[0]}"
    )

    ordered = block_df.sort_values(["trip_start_seconds", "trip_id", "stop_sequence"])
    trips = [
        (trip_id, group.iloc[0], group.iloc[-1])
        for trip_id, group in ordered.groupby("trip_id", sort=False)
    ]
    if any(pd.isna(first["trip_start_seconds"]) for _, first, _ in trips):
        logging.warning(
            "Block %s / %s has a trip with no start time; layover rows skipped.",
            block_df["block_id"].iloc[0],
            schedule,
        )
        return schedule_df

    stop_rows = {
        trip_id: group.to_dict("records")
        for trip_id, group in schedule_df.groupby("Trip ID", sort=False)
    }
    block_id = trips[0][1]["block_id"]

    def observation_row(fields: Mapping[str, Any], note: str) -> dict[str, Any]:
        row = dict.fromkeys(schedule_df.columns, "")
        row.update(
            {
                "Block ID": block_id,
                "Actual Arrival Time": MISSING_TIME,
                "Actual Departure Time": MISSING_TIME,
                "Boardings": BLOCKED_COUNT_MARKER,
                "Alightings": BLOCKED_COUNT_MARKER,
                "Comments": (
                    f"{note}\nActual location: __________________________"
                    "\nActivity / reason: _________________________"
                ),
                **fields,
            }
        )
        return row

    def movement_row(move: Mapping[str, Any]) -> dict[str, Any]:
        label = str(move["label"])
        return observation_row(
            {
                "Event Type": label,
                "Trip ID": move["trip_id"],
                "Trip Start Time": format_hhmm(move["start"]),
                "Stop Name": f"From: {move['from_name']}\nTo: {move['to_name']}",
                "Scheduled Departure Time": format_hhmm(move["start"]),
                "Scheduled Arrival Time": format_hhmm(move["end"]),
            },
            f"Scheduled {label.lower()} (TODS): {_duration_text(move['end'] - move['start'])}",
        )

    def wait_row(trip: tuple, begin: float, finish: float, stop: Any) -> dict[str, Any]:
        trip_id, first, last = trip
        return observation_row(
            {
                "Event Type": "Layover",
                "Route": last["route_short_name"],
                "Direction": last["direction_id"],
                "Trip ID": trip_id,
                "Trip Start Time": format_hhmm(first["trip_start_seconds"]),
                "Stop ID": stop["stop_label"],
                "Stop Name": stop["stop_name"],
                "Scheduled Arrival Time": format_hhmm(begin),
                "Scheduled Departure Time": format_hhmm(finish),
            },
            f"Scheduled gap: {_duration_text(finish - begin)}",
        )

    def between_row(trip: tuple, begin: float, finish: float, origin: str, dest: str) -> dict:
        trip_id, first, last = trip
        return observation_row(
            {
                "Event Type": "Between trips",
                "Route": last["route_short_name"],
                "Direction": last["direction_id"],
                "Trip ID": trip_id,
                "Trip Start Time": format_hhmm(first["trip_start_seconds"]),
                "Stop Name": f"From: {origin}\nTo: {dest}",
                "Scheduled Arrival Time": format_hhmm(begin),
                "Scheduled Departure Time": format_hhmm(finish),
            },
            f"Scheduled gap: {_duration_text(finish - begin)} (may include travel)",
        )

    def gap_row(trip: tuple, begin: float, finish: float, here: Any, there: Any) -> dict:
        # here/there: stop_id, stop_label, stop_name where the vehicle is / goes next.
        same_stop = (
            pd.notna(here["stop_id"])
            and pd.notna(there["stop_id"])
            and str(here["stop_id"]).strip() != ""
            and str(here["stop_id"]).strip() == str(there["stop_id"]).strip()
        )
        if same_stop:
            return wait_row(trip, begin, finish, here)
        return between_row(trip, begin, finish, here["stop_name"], there["stop_name"])

    def move_stop(move: Mapping[str, Any], end: str) -> dict[str, Any]:
        # end is "from" or "to"; same keys as a prepared stop row.
        return {
            "stop_id": move[f"{end}_stop"],
            "stop_label": move[f"{end}_label"],
            "stop_name": move[f"{end}_name"],
        }

    def trip_end(last: Any) -> float:
        end = last["arrival_seconds"]
        return last["departure_seconds"] if pd.isna(end) else end

    placed: set[str] = set()
    rows: list[dict[str, Any]] = []
    first_start = trips[0][1]["trip_start_seconds"]
    # Before the first trip, follow the vehicle from movement to movement so a
    # wait between a pull-out's arrival and the first departure gets its row.
    cursor: Optional[float] = None
    here: Any = None
    for move in movements:
        if move["end"] <= first_start:
            if include_gaps and cursor is not None and move["start"] > cursor:
                rows.append(gap_row(trips[0], cursor, move["start"], here, move_stop(move, "from")))
            rows.append(movement_row(move))
            placed.add(move["trip_id"])
            cursor, here = move["end"], move_stop(move, "to")
    if include_gaps and cursor is not None and first_start > cursor:
        rows.append(gap_row(trips[0], cursor, first_start, here, trips[0][1]))

    latest_end = -math.inf
    for pos, (trip_id, first, last) in enumerate(trips):
        rows.extend(stop_rows.get(trip_id, []))
        if pos + 1 == len(trips):
            end_seconds = trip_end(last)
            if pd.isna(end_seconds):
                continue
            cursor, here = end_seconds, last
            for move in movements:
                if move["trip_id"] not in placed and move["start"] >= end_seconds:
                    if include_gaps and move["start"] > cursor:
                        rows.append(
                            gap_row(
                                trips[pos], cursor, move["start"], here, move_stop(move, "from")
                            )
                        )
                    rows.append(movement_row(move))
                    placed.add(move["trip_id"])
                    cursor, here = move["end"], move_stop(move, "to")
            continue

        next_trip_id, next_first, _ = trips[pos + 1]
        end_seconds = trip_end(last)
        start_seconds = next_first["trip_start_seconds"]
        if pd.isna(end_seconds):
            logging.warning("No terminal time for trip %s; its following gap is skipped.", trip_id)
            continue

        gap_seconds = start_seconds - end_seconds
        overlaps_earlier_trip = end_seconds < latest_end
        latest_end = max(latest_end, end_seconds)
        if gap_seconds < 0 or overlaps_earlier_trip:
            logging.warning(
                "Overlapping trips affect the gap after %s before %s in block %s / "
                "%s; no gap row added.",
                trip_id,
                next_trip_id,
                first["block_id"],
                schedule,
            )
            continue

        inside = [
            m
            for m in movements
            if m["trip_id"] not in placed
            and m["start"] >= end_seconds
            and m["end"] <= start_seconds
        ]
        if inside:
            # TODS explains the gap. Follow the vehicle: each wait is where the
            # previous trip or movement left it, checked against the next origin.
            cursor, here = end_seconds, last
            for move in inside:
                if include_gaps and move["start"] > cursor:
                    rows.append(
                        gap_row(trips[pos], cursor, move["start"], here, move_stop(move, "from"))
                    )
                rows.append(movement_row(move))
                placed.add(move["trip_id"])
                cursor, here = move["end"], move_stop(move, "to")
            if include_gaps and start_seconds > cursor:
                rows.append(gap_row(trips[pos], cursor, start_seconds, here, next_first))
            continue
        if gap_seconds == 0 or not include_gaps:
            continue
        rows.append(gap_row(trips[pos], end_seconds, start_seconds, last, next_first))

    unplaced = [m["trip_id"] for m in movements if m["trip_id"] not in placed]
    if unplaced:
        logging.warning(
            "Block %s: TODS trip(s) %s overlap revenue trips; not listed.",
            block_id,
            ", ".join(unplaced),
        )
    return pd.DataFrame(rows, columns=schedule_df.columns)


def export_blocks(
    stop_times_df: pd.DataFrame,
    service_labels: Optional[Mapping[str, str]] = None,
    tods_movements: Optional[Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]] = None,
) -> None:
    """Generate one Excel schedule per vehicle block and service ID.

    Args:
        stop_times_df: Prepared stop times (see
            :pyfunc:`prepare_stop_times`).  Must include the columns
            produced earlier (``block_id``, ``scheduled_arrival_hhmm``,
            ``scheduled_departure_hhmm``, etc.).
        service_labels: ``service_id`` → calendar label for file names (see
            :pyfunc:`build_service_labels`). ``None`` shows raw service_ids.
        tods_movements: TODS nonrevenue trips by ``(block_id, service_id)``
            (see :pyfunc:`load_tods_movements`), or by ``(block_id,
            SERVICE_DATE)`` when that is set (see :pyfunc:`movements_on_date`);
            ``None`` lists GTFS trips only.

    Side Effects:
        Writes one file per block/service, named by :pyfunc:`build_block_filename`, to
        ``BASE_OUTPUT_PATH``; creates the folder tree if needed. When
        ``TIMEPOINTS_ONLY`` is set, only rows with ``timepoint == 1`` are
        written, and blocks with no timepoint stops are skipped. Boardings at
        each trip's final stop and alightings at its first stop are marked with
        ``BLOCKED_COUNT_MARKER``. Endpoints come from the full stop sequence,
        before the timepoint filter; a one-event trip has both counts marked.
        Scheduled arrival and departure come from their respective GTFS fields;
        actual arrival and departure have independent handwritten-entry blanks.
        With ``INCLUDE_LAYOVER_ROWS``, positive gaps receive observation rows.
        ``Event Type`` is always present so stop records can be selected during
        ingestion. A block reused across service IDs is exported separately
        for each service, with its service ID appended to prevent name collisions.
        With ``SERVICE_DATE`` set, the stop times must all run on that date
        (see :pyfunc:`filter_data`): each block is exported once, with every
        trip of every service ID, and the date is appended to its file name.
    """
    service_date = SERVICE_DATE.strip()
    # One schedule per block and service ID, or per block on the service date.
    schedule = (
        pd.Series(service_date, index=stop_times_df.index, name="service_id")
        if service_date
        else stop_times_df["service_id"]
    )
    groups = stop_times_df.groupby([stop_times_df["block_id"], schedule], dropna=False, sort=True)
    service_counts = stop_times_df.groupby("block_id")["service_id"].nunique(dropna=False)
    logging.info("Found %d block/service schedules to export.\n", groups.ngroups)

    for (block_id, service_id), group in groups:
        block_subset = group.copy()
        if block_subset.empty:
            continue

        # Use the actual first event, not a minimum taken across all trip stops.
        first_events = block_subset.sort_values(["trip_id", "stop_sequence"]).drop_duplicates(
            "trip_id", keep="first"
        )
        first_departures = first_events[["trip_id"]].copy()
        first_departures["trip_start_seconds"] = first_events["departure_seconds"].fillna(
            first_events["arrival_seconds"]
        )
        first_departures["trip_start_hhmm"] = first_departures["trip_start_seconds"].apply(
            format_hhmm
        )
        block_subset = block_subset.merge(first_departures, on="trip_id", how="left")

        block_subset["Trip Start Time"] = block_subset["trip_start_hhmm"]
        filename = build_block_filename(block_subset, service_labels or {})
        if service_date:
            safe_date = re.sub(r'[\\/:*?"<>|\s]+', "-", service_date)
            filename = f"{os.path.splitext(filename)[0]}_{safe_date}.xlsx"
        elif service_counts.loc[block_id] > 1:
            safe_service = re.sub(r'[\\/:*?"<>|\s]+', "-", str(service_id))
            filename = f"{os.path.splitext(filename)[0]}_service-{safe_service}.xlsx"

        # Identify actual endpoints before filtering the displayed stop events.
        # Use sequence numbers, not stop IDs: loop trips may revisit the same stop.
        trip_sequences = block_subset.groupby("trip_id")["stop_sequence"]
        block_subset["is_first_trip_stop"] = block_subset["stop_sequence"].eq(
            trip_sequences.transform("min")
        )
        block_subset["is_last_trip_stop"] = block_subset["stop_sequence"].eq(
            trip_sequences.transform("max")
        )
        full_block = block_subset.copy()

        # Filter after trip starts, file naming, and endpoint flags cover every stop.
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
            "stop_label",
            "stop_name",
            "scheduled_arrival_hhmm",
            "scheduled_departure_hhmm",
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
                "stop_label": "Stop ID",
                "stop_name": "Stop Name",
                "scheduled_arrival_hhmm": "Scheduled Arrival Time",
                "scheduled_departure_hhmm": "Scheduled Departure Time",
            },
        )

        # Insert placeholders
        final_df["Actual Arrival Time"] = MISSING_TIME
        final_df["Actual Departure Time"] = MISSING_TIME
        final_df["Event Type"] = "Stop"
        final_df["Boardings"] = MISSING_VALUE
        final_df["Alightings"] = MISSING_VALUE
        final_df["Comments"] = COMMENTS_PLACEHOLDER
        final_df.loc[block_subset["is_last_trip_stop"], "Boardings"] = BLOCKED_COUNT_MARKER
        final_df.loc[block_subset["is_first_trip_stop"], "Alightings"] = BLOCKED_COUNT_MARKER

        # Reorder columns to place 'Timepoint' after 'Stop Sequence'
        final_df = final_df[
            [
                "Block ID",
                "Event Type",
                "Route",
                "Direction",
                "Trip ID",
                "Trip Start Time",
                "Stop Sequence",
                "Timepoint",
                "Stop ID",
                "Stop Name",
                "Scheduled Arrival Time",
                "Scheduled Departure Time",
                "Actual Arrival Time",
                "Actual Departure Time",
                "Boardings",
                "Alightings",
                "Comments",
            ]
        ]

        final_df["_trip_start_seconds"] = block_subset["trip_start_seconds"]
        final_df = final_df.sort_values(by=["_trip_start_seconds", "Trip ID", "Stop Sequence"])
        final_df = final_df.drop(columns="_trip_start_seconds")
        if INCLUDE_LAYOVER_ROWS or movements:
            final_df = add_layover_rows(
                final_df,
                full_block,
                movements,
                include_gaps=INCLUDE_LAYOVER_ROWS,
                service_date=service_date,
            )

        output_path = os.path.join(BASE_OUTPUT_PATH, filename)
        export_to_excel(final_df, output_path)


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def available_gtfs_files(gtfs_path: str, names: Sequence[str]) -> tuple[str, ...]:
    """Return the *names* present, and not empty, in a GTFS folder or ZIP archive.

    ZIP members are matched by base name, as :func:`load_gtfs_data` matches
    them, so a feed nested one folder deep inside the archive is found too.
    Zero-byte files are skipped with a warning because the loader rejects them.
    """
    sizes: dict[str, int] = {}
    if os.path.isdir(gtfs_path):
        for name in names:
            path = os.path.join(gtfs_path, name)
            if os.path.isfile(path):
                sizes[name] = os.path.getsize(path)
    else:
        with zipfile.ZipFile(gtfs_path) as archive:
            for info in archive.infolist():
                name = os.path.basename(info.filename)
                if name in names:
                    sizes[name] = max(sizes.get(name, 0), info.file_size)
    for name in names:
        if sizes.get(name) == 0:
            logging.warning("Skipping empty optional file %s.", name)
    return tuple(name for name in names if sizes.get(name, 0) > 0)


def resolve_service_ids(
    calendar_df: Optional[pd.DataFrame],
    calendar_dates_df: Optional[pd.DataFrame],
    service_date: str,
) -> list[str]:
    """Return every service_id active on ``service_date`` (``YYYYMMDD``).

    Expands calendar.txt and calendar_dates.txt into each service's real set
    of active dates (see :func:`expand_service_active_dates`) and keeps the
    services that operate on the requested day.

    Args:
        calendar_df: Parsed ``calendar.txt``, or ``None`` if the feed has none.
        calendar_dates_df: Parsed ``calendar_dates.txt``, or ``None``.
        service_date: The day to analyze, as ``YYYYMMDD``.

    Returns:
        Sorted list of service_id strings (possibly empty).

    Raises:
        ValueError: If *service_date* is not a valid ``YYYYMMDD`` date.
    """
    try:
        target = dt.datetime.strptime(service_date.strip(), "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"SERVICE_DATE must be a YYYYMMDD date; got {service_date!r}") from exc
    active = expand_service_active_dates(calendar_df, calendar_dates_df, today=target)
    return sorted(service_ids_active_on(active, target))


# Canonical version lives in utils/calendar_helpers.py -- keep this copy in sync.
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


# Canonical version lives in utils/calendar_helpers.py -- keep this copy in sync.
def service_ids_active_on(
    active_dates: Mapping[str, set[dt.date]],
    target_date: dt.date,
) -> set[str]:
    """Return the service_ids operating on *target_date*.

    Args:
        active_dates: Output of :func:`expand_service_active_dates`.
        target_date: The calendar date to query.

    Returns:
        Set of service_id strings active on that date (possibly empty).
    """
    return {sid for sid, dates in active_dates.items() if target_date in dates}


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


def apply_supplement(
    base: Optional[pd.DataFrame], supplement: Optional[pd.DataFrame], keys: Sequence[str]
) -> pd.DataFrame:
    """Apply one TODS supplement file to its GTFS table (TODS 2.1 evaluation rules).

    A row whose primary key matches the GTFS table deletes that row when
    ``TODS_delete`` is ``1`` and otherwise overwrites it with the row's
    non-empty values; a row with a new key is added whole.

    Args:
        base: The GTFS table (``None`` or empty if the feed lacks it).
        supplement: The ``*_supplement.txt`` table, or ``None``.
        keys: The GTFS file's primary-key columns.

    Returns:
        The supplemented table, all values as text.

    Raises:
        ValueError: If the supplement lacks a key column or both deletes and
            re-adds the same primary key.
    """
    base = pd.DataFrame(columns=list(keys)) if base is None else base.fillna("")
    if supplement is None or supplement.empty:
        return base.copy()
    supplement = supplement.fillna("")
    missing = [k for k in keys if k not in supplement.columns]
    if missing:
        raise ValueError(f"Supplement file is missing primary-key column(s): {missing}")
    rows: dict[tuple, dict[str, Any]] = {
        tuple(r[k] for k in keys): r for r in base.to_dict("records")
    }
    order = list(rows)
    deleted: set[tuple] = set()
    written: set[tuple] = set()
    for record in supplement.to_dict("records"):
        key = tuple(record[k] for k in keys)
        values = {c: v for c, v in record.items() if c != "TODS_delete"}
        if str(record.get("TODS_delete", "")).strip() == "1":
            if key in rows:
                deleted.add(key)
            else:
                logging.warning("TODS_delete for %s matches no GTFS row; ignored.", key)
            continue
        written.add(key)
        if key in rows:
            rows[key].update({c: v for c, v in values.items() if v != ""})
        else:
            rows[key] = values
            order.append(key)
    if deleted & written:
        raise ValueError(f"Supplement both deletes and re-adds key(s): {sorted(deleted & written)}")
    out = pd.DataFrame([rows[k] for k in order if k not in deleted])
    return out.reindex(columns=list(dict.fromkeys([*base.columns, *out.columns]))).fillna("")


def drop_trips_on_deleted_routes(
    trips: pd.DataFrame, routes_before: pd.DataFrame, routes_after: pd.DataFrame
) -> pd.DataFrame:
    """Drop trips whose route a TODS ``routes_supplement.txt`` deletes.

    TODS ignores trips that reference a deleted route unless they are
    reassigned, so pass *trips* after ``trips_supplement.txt`` has been
    applied: a trip moved to a surviving route keeps its new ``route_id`` and
    stays. Each dropped trip is counted in a warning.

    Args:
        trips: Trips table with a ``route_id`` column, already supplemented.
        routes_before: ``routes.txt`` before ``routes_supplement.txt``.
        routes_after: The same table after :func:`apply_supplement`.

    Returns:
        *trips* without the rows on deleted routes.
    """
    deleted = set(routes_before.get("route_id", pd.Series(dtype=str)).astype(str)) - set(
        routes_after.get("route_id", pd.Series(dtype=str)).astype(str)
    )
    if not deleted:
        return trips
    on_deleted = trips["route_id"].astype(str).isin(deleted)
    if on_deleted.any():
        logging.warning(
            "Ignoring %d trip(s) on route(s) deleted in routes_supplement.txt (%s); "
            "reassign them to another route in trips_supplement.txt to keep them.",
            int(on_deleted.sum()),
            ", ".join(sorted(deleted)),
        )
    return trips.loc[~on_deleted]


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
        CONFIGURATION paths are blank.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    missing_paths = False
    if not GTFS_FOLDER_PATH.strip():
        logging.warning(
            "GTFS_FOLDER_PATH is blank. Paste the input GTFS folder or .zip path "
            "between its quotes in CONFIGURATION."
        )
        missing_paths = True
    if not BASE_OUTPUT_PATH.strip():
        logging.warning(
            "BASE_OUTPUT_PATH is blank. Paste the output folder path "
            "between its quotes in CONFIGURATION."
        )
        missing_paths = True
    if missing_paths:
        logging.info("No processing performed. Fill in both paths above and re-run.")
        return 2

    logging.info("========================================================")
    logging.info("GTFS Block Schedule Printable Generator")
    logging.info("Input GTFS Folder: %s", GTFS_FOLDER_PATH)
    logging.info("Output Folder:     %s", BASE_OUTPUT_PATH)
    if FILTER_ROUTE_SHORT_NAMES:
        logging.info("Filtering for Routes: %s", FILTER_ROUTE_SHORT_NAMES)
    if FILTER_SERVICE_IDS:
        logging.info("Filtering for Service IDs: %s", FILTER_SERVICE_IDS)
    if SERVICE_DATE.strip():
        logging.info("Service date: %s", SERVICE_DATE.strip())
    if TIMEPOINTS_ONLY:
        logging.info("Listing timepoint stops only.")

    try:
        gtfs_data = load_gtfs_data(
            gtfs_path=GTFS_FOLDER_PATH,
            files=REQUIRED_GTFS_FILES,
            dtype=str,
        )
        optional_files = available_gtfs_files(GTFS_FOLDER_PATH, OPTIONAL_GTFS_FILES)
        if optional_files:
            gtfs_data.update(load_gtfs_data(GTFS_FOLDER_PATH, files=optional_files, dtype=str))

        trips_df = gtfs_data["trips"]
        stop_times_df = gtfs_data["stop_times"]
        stops_df = gtfs_data["stops"]
        routes_df = gtfs_data["routes"]
        service_labels = build_service_labels(gtfs_data["calendar"])

        date_service_ids: Optional[list[str]] = None
        if SERVICE_DATE.strip():
            if FILTER_SERVICE_IDS:
                raise ValueError("Set either FILTER_SERVICE_IDS or SERVICE_DATE, not both.")
            date_service_ids = resolve_service_ids(
                gtfs_data["calendar"], gtfs_data.get("calendar_dates"), SERVICE_DATE
            )
            if not date_service_ids:
                raise ValueError(f"No service_id is active on {SERVICE_DATE.strip()} in this feed.")
            logging.info("Service IDs active on that date: %s", date_service_ids)

        trips_df, stop_times_df = filter_data(trips_df, stop_times_df, routes_df, date_service_ids)
        if trips_df.empty or stop_times_df.empty:
            logging.warning("No data remains after filtering – no files generated.")
            return 1

        prepared = prepare_stop_times(trips_df, stop_times_df, stops_df)
        if prepared.empty:
            logging.warning("No data remains after preparation – no files generated.")
            return 1
        reject_frequency_trips(gtfs_data.get("frequencies"), prepared["trip_id"])
        if TIMEPOINTS_ONLY and not (prepared["timepoint"] == 1).any():
            logging.warning(
                "TIMEPOINTS_ONLY is True but no stops have timepoint = 1 in stop_times.txt "
                "– no files generated. Set TIMEPOINTS_ONLY = False to list all stops."
            )
            return 1

        tods_movements = None
        if TODS_PATH.strip():
            logging.info("TODS supplement files: %s", TODS_PATH)
            tods_movements = load_tods_movements(TODS_PATH, stops_df)
            if date_service_ids is not None:
                tods_movements = movements_on_date(
                    tods_movements, date_service_ids, SERVICE_DATE.strip()
                )
        export_blocks(prepared, service_labels, tods_movements)
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
