"""Generate printable Excel schedules for each vehicle block in a GTFS feed.

The script reads the five core GTFS tables—``trips``, ``stop_times``, ``stops``,
``routes`` and ``calendar``—and optionally filters them by *service ID* and/or
*route short name*.  For every vehicle ``block_id`` that survives filtering it
produces a nicely-formatted ``.xlsx`` file ready for field auditing.

Outputs
-------
- ``block_<block_id>_<routes>_<HHMM>-<HHMM>_<calendar>.xlsx`` (one per surviving
  block/service ID, written to ``BASE_OUTPUT_PATH``):
  the block's stop-by-stop schedule with placeholder columns for handwritten
  field notes. Set ``TIMEPOINTS_ONLY`` to list only timepoint stops; otherwise
  all stops are listed and timepoint rows are shaded.
  Blocks shared by service IDs are exported separately, with a service-ID
  suffix on their file names. Optional layover rows are enabled by default.

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
- Names each file by block, routes, time span and calendar for easy sorting by hand.
- Marks boardings at the last stop and alightings at the first stop of each trip.
- Prints collector Name/Date fields and the generated workbook's full file path
  in Excel headers/footers, outside the data table and repeated on every page.
- Adds a writable row for positive gaps between trips, with scheduled times,
  duration, actual times, and location/activity notes. Different endpoint stops
  are labeled "Between trips" because the gap may include travel.
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
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

# =============================================================================
# CONFIGURATION
# =============================================================================

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

# If you only want certain service IDs or route short names, specify them here:
FILTER_SERVICE_IDS: list[str] = []  # e.g. ["WKD", "SAT"]
FILTER_ROUTE_SHORT_NAMES: list[str] = []  # e.g. ["101", "202"]

# Calendar label used in output file names, by service_id. Unlisted service_ids
# are labeled from their calendar.txt days (e.g. "Weekday", "Saturday", "MonTueWedThu").
SERVICE_LABEL_OVERRIDES: dict[str, str] = {}  # e.g. {"4": "Weekday", "5": "Saturday"}

# Stops to list on each block sheet:
TIMEPOINTS_ONLY: bool = False  # True → timepoint stops only; False → all stops

# Add observation rows for positive scheduled gaps between trips in a block.
INCLUDE_LAYOVER_ROWS: bool = True  # False → stop events only.

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
    stop_times_df["scheduled_arrival_hhmm"] = stop_times_df["arrival_seconds"].apply(format_hhmm)
    stop_times_df["scheduled_departure_hhmm"] = stop_times_df["departure_seconds"].apply(
        format_hhmm
    )

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


def add_layover_rows(schedule_df: pd.DataFrame, block_df: pd.DataFrame) -> pd.DataFrame:
    """Insert observation rows between adjacent trips in one block/service ID.

    Args:
        schedule_df: Formatted stop rows, possibly filtered to timepoints.
        block_df: All prepared stop events for this block and service ID,
            including ``trip_start_seconds`` before the timepoint filter.

    Returns:
        Stop rows followed by their positive scheduled gap, where applicable.
        ``Layover`` means the endpoint stop IDs match. ``Between trips`` means
        they differ, so the gap may include repositioning as well as waiting.

    Notes:
        Gap limits are the preceding trip's last arrival and the following
        trip's first departure, with the other time at that same event used
        only if the preferred time is missing. Actual time fields record the
        observed waiting period; Comments records its location and activity.
        A gap row carries the preceding trip ID, with no stop sequence or
        timepoint flag. There are no rows before the first or after the last
        trip, and no rows for zero or negative gaps. Endpoints come from full
        trips even when only timepoints are printed. Separate service IDs are
        never connected; selecting an actual date is outside this exporter.
    """
    if len(block_df["service_id"].unique()) != 1:
        raise ValueError("Layover rows require one service ID per block schedule.")

    ordered = block_df.sort_values(["trip_start_seconds", "trip_id", "stop_sequence"])
    trips = [
        (trip_id, group.iloc[0], group.iloc[-1])
        for trip_id, group in ordered.groupby("trip_id", sort=False)
    ]
    if any(pd.isna(first["trip_start_seconds"]) for _, first, _ in trips):
        logging.warning(
            "Block %s / service %s has a trip with no start time; layover rows skipped.",
            block_df["block_id"].iloc[0],
            block_df["service_id"].iloc[0],
        )
        return schedule_df

    stop_rows = {
        trip_id: group.to_dict("records")
        for trip_id, group in schedule_df.groupby("Trip ID", sort=False)
    }
    rows: list[dict[str, Any]] = []
    latest_end = -math.inf
    for pos, (trip_id, first, last) in enumerate(trips):
        rows.extend(stop_rows.get(trip_id, []))
        if pos + 1 == len(trips):
            continue

        next_trip_id, next_first, _ = trips[pos + 1]
        end_seconds = last["arrival_seconds"]
        if pd.isna(end_seconds):
            end_seconds = last["departure_seconds"]
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
                "service %s; no gap row added.",
                trip_id,
                next_trip_id,
                first["block_id"],
                first["service_id"],
            )
            continue
        if gap_seconds == 0:
            continue

        same_stop = (
            pd.notna(last["stop_id"])
            and pd.notna(next_first["stop_id"])
            and str(last["stop_id"]).strip() != ""
            and last["stop_id"] == next_first["stop_id"]
        )
        duration = f"{gap_seconds / 60:g} min"
        if gap_seconds % 60:
            duration = f"{int(gap_seconds) // 60}m {int(gap_seconds) % 60}s"
        gap_note = f"Scheduled gap: {duration}"
        if not same_stop:
            gap_note += " (may include travel)"

        row = dict.fromkeys(schedule_df.columns, "")
        row.update(
            {
                "Block ID": first["block_id"],
                "Event Type": "Layover" if same_stop else "Between trips",
                "Route": last["route_short_name"],
                "Direction": last["direction_id"],
                "Trip ID": trip_id,
                "Trip Start Time": format_hhmm(first["trip_start_seconds"]),
                "Stop ID": last["stop_id"] if same_stop else "",
                "Stop Name": (
                    last["stop_name"]
                    if same_stop
                    else f"From: {last['stop_name']}\nTo: {next_first['stop_name']}"
                ),
                "Scheduled Arrival Time": format_hhmm(end_seconds),
                "Scheduled Departure Time": format_hhmm(start_seconds),
                "Actual Arrival Time": MISSING_TIME,
                "Actual Departure Time": MISSING_TIME,
                "Boardings": BLOCKED_COUNT_MARKER,
                "Alightings": BLOCKED_COUNT_MARKER,
                "Comments": (
                    f"{gap_note}\nActual location: __________________________"
                    "\nActivity / reason: _________________________"
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=schedule_df.columns)


def export_blocks(
    stop_times_df: pd.DataFrame, service_labels: Optional[Mapping[str, str]] = None
) -> None:
    """Generate one Excel schedule per vehicle block and service ID.

    Args:
        stop_times_df: Prepared stop times (see
            :pyfunc:`prepare_stop_times`).  Must include the columns
            produced earlier (``block_id``, ``scheduled_arrival_hhmm``,
            ``scheduled_departure_hhmm``, etc.).
        service_labels: ``service_id`` → calendar label for file names (see
            :pyfunc:`build_service_labels`). ``None`` shows raw service_ids.

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
    """
    groups = stop_times_df.groupby(["block_id", "service_id"], dropna=False, sort=True)
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
        if service_counts.loc[block_id] > 1:
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
            "stop_id",
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
                "stop_id": "Stop ID",
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
        if INCLUDE_LAYOVER_ROWS:
            final_df = add_layover_rows(final_df, full_block)

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
