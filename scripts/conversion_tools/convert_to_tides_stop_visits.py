"""Convert an AVL stop visit export, or a folder of them, into TIDES stop_visits records.

This module reads AVL "Stop Visit Events" reports (CSV) and produces a
`stop_visits.csv` file aligned with the TIDES `stop_visits` schema. It
normalizes common real-world export issues (inconsistent whitespace, AM/PM
timestamps, mixed null tokens) and applies pragmatic, schema-aware rules so
the output can be ingested by downstream validation and analytics pipelines.

The Stop Visit Events report is timepoint-based rather than true stop-ordered.
This converter therefore preserves source `Timepoint Order` values as both
`trip_stop_sequence` and `scheduled_stop_sequence`. While this may violate
strict stop-order expectations in the TIDES schema, it reflects the best
available ordering in the source data. Potential sequencing issues are logged
for visibility.

Key behaviors:
- Parses scheduled and actual stop timestamps robustly (tolerates AM/PM,
  inconsistent whitespace, and partial availability). Rows are not dropped
  solely due to incomplete timing data.
- Uses Scheduled Passing Time for scheduled arrival and departure when
  present. Uses Arrival Time and Departure Time for actual stop times, with
  fallback to Actual Time when needed.
- Computes `dwell` in seconds when actual arrival and departure are present
  and non-negative; invalid dwell values are left blank and logged.
- Derives `service_date` from the Date field and sets `trip_id_performed` by
  `TRIP_ID_MODE`, the same rule `convert_to_tides_trips_performed.py` applies:
  the Trip token, except that a token run by more than one vehicle on a date
  gets a stable hash of (service_date, trip token, vehicle_id) per vehicle.
- Shifts a trip whose Timepoint Order starts at 0 up by one as a whole, and
  warns when a trip repeats a sequence value.
- Reads every input column as text, so identifiers keep leading zeros.
- Accepts a folder as INPUT_PATH: its CSVs are combined in name order before
  conversion; rows repeated across files are dropped, unusable files skipped.
- Extracts `stop_id` from Timepoint ID, marks all records as
  `timepoint = True`, and synthesizes `pattern_id` from available route,
  direction, and variation fields.
- Emits schema-required fields using schema-valid types and enum values, and
  leaves unsupported stop-level attributes (APC, door events, ramp activity,
  revenue) blank with explicit warnings.

The conversion is deterministic: the same inputs and configuration give the same output.

Column names are configured via COLUMN_MAP in the CONFIGURATION section.
Defaults are pre-set for CLEVER "Stop Visit Events" exports; update COLUMN_MAP
values to adapt to other AVL sources.

Outputs
-------
- `stop_visits.csv` (`OUTPUT_FILENAME`, written to `OUTPUT_DIR`) - one TIDES
  `stop_visits` record per source stop visit event, with columns in the exact
  order of the TIDES template (`TIDES_COLS`).
- Optional combined input (`COMBINED_INPUT_FILENAME`, folder input only).
- `stop_visits_runlog.txt` - run log with row counts and the verbatim CONFIGURATION.

Typical usage
-------------
Update the paths in the CONFIGURATION section and run from a shell, ArcGIS
Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# A single Stop Visit Events CSV, or a folder of them. A folder's matching CSVs
# are combined into one table before conversion, so a trip or service date
# split across two exports is converted as a whole.
INPUT_PATH: Path = Path(r"Path\To\Stop Visit Events.csv")

# Folder input only (ignored when INPUT_PATH is a file):
INPUT_GLOB: str = "*.csv"  # file name pattern, e.g. "Stop Visit Events*.csv"
INPUT_RECURSIVE: bool = False  # also search subfolders
# Drop rows that exactly repeat a row from an earlier file (sorted by name), as
# overlapping date-range exports do. Repeats within one file are kept.
DROP_DUPLICATE_ROWS: bool = True
# Also write the combined input, with a source_file column naming each row's
# file, to OUTPUT_DIR under this name. None skips it.
COMBINED_INPUT_FILENAME: str | None = None

OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")
OUTPUT_FILENAME: str = "stop_visits.csv"
OUTPUT_CSV: Path = OUTPUT_DIR / OUTPUT_FILENAME

# Trip ID strategy. Set the same value as TRIP_ID_MODE in
# convert_to_tides_trips_performed.py so the two outputs join:
# - "token": trip_id_performed = second token in Trip field, except that a token
#   run by more than one vehicle on a service_date gets a stable hash of
#   (service_date, trip_token, vehicle_id) for each vehicle (default)
# - "hashed": trip_id_performed = stable hash of (service_date, trip_token, vehicle_id)
#   for every trip
TRIP_ID_MODE: str = "token"  # "token" | "hashed"

# Map generic field roles to actual column names in your input file.
# Defaults are set for CLEVER "Stop Visit Events" exports.
# Change values here to adapt to other AVL sources; do not rename the keys.
COLUMN_MAP: dict[str, str | None] = {
    "date": "Date",
    "trip": "Trip",
    "route": "Route",
    "direction": "Direction",
    "vehicle": "Vehicle",
    "variation": "Variation",
    "timepoint_order": "Timepoint Order",
    "timepoint_id": "Timepoint ID",
    "actual_time": "Actual Time",
    "arrival_time": "Arrival Time",
    "departure_time": "Departure Time",
    "scheduled_passing": "Scheduled Passing Time",
}

# TIDES stop_visits output columns in the exact order of the template.
TIDES_COLS: list[str] = [
    "service_date",
    "trip_id_performed",
    "trip_stop_sequence",
    "scheduled_stop_sequence",
    "pattern_id",
    "vehicle_id",
    "dwell",
    "stop_id",
    "timepoint",
    "schedule_arrival_time",
    "schedule_departure_time",
    "actual_arrival_time",
    "actual_departure_time",
    "distance",
    "boarding_1",
    "alighting_1",
    "boarding_2",
    "alighting_2",
    "departure_load",
    "door_open",
    "door_close",
    "door_status",
    "ramp_deployed_time",
    "ramp_failure",
    "kneel_deployed_time",
    "lift_deployed_time",
    "bike_rack_deployed",
    "bike_load",
    "revenue",
    "number_of_transactions",
    "schedule_relationship",
]

# Required input columns to produce a useful stop_visits file.
REQ_COLS: list[str] = [
    v
    for k in ("date", "trip", "timepoint_order", "timepoint_id")
    if (v := COLUMN_MAP[k]) is not None
]

# Optional input columns (used when present).
OPT_COLS: list[str] = [
    v
    for k in (
        "route",
        "direction",
        "vehicle",
        "variation",
        "actual_time",
        "arrival_time",
        "departure_time",
        "scheduled_passing",
    )
    if (v := COLUMN_MAP[k]) is not None
]

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write makes the script exit with code 1 so the
# analyst is never left with an output that lacks a matching configuration
# record. Set to False only when writing to a location where a sidecar file
# cannot be created (e.g. a read-only share).
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# HELPERS
# =============================================================================


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text values: strip, collapse whitespace, coerce common blanks to NA."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "N/A": pd.NA})
    )


def parse_service_date(date_series: pd.Series) -> pd.Series:
    """Parse Date (m/d/YYYY) into ISO service_date (YYYY-MM-DD)."""
    s = normalize_text(date_series)
    dt = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")
    bad = int(dt.isna().sum())
    if bad:
        logging.warning("Failed to parse %d Date values; emitting blanks for those rows.", bad)
    return dt.dt.strftime("%Y-%m-%d").astype("string")


def split_trip_token(trip_series: pd.Series) -> pd.Series:
    """Extract the trip token from the Trip field.

    # CLEVER Trip field format: "HH:MM <trip_token>" (e.g. "04:02 1550064")
    If parsing fails, emits NA and logs a warning.
    """
    s = normalize_text(trip_series).fillna("")
    parts = s.str.split(r"\s+", n=1, expand=True)

    trip_token = (
        parts[1].astype("string").fillna("").str.strip()
        if parts.shape[1] > 1
        else pd.Series("", index=s.index, dtype="string")
    )

    if (trip_token == "").any():
        logging.warning(
            "Some Trip values did not include a second token (trip token). "
            "trip_id_performed may be blank for those rows."
        )

    return trip_token.replace({"": pd.NA}).astype("string")


def stable_id(*parts: str) -> str:
    """Short deterministic ID from multiple fields (sha1 truncated)."""
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def parse_route_short(route_text: pd.Series) -> pd.Series:
    """Route like '101 - Fort Hunt - Mount Vernon' -> '101' (best-effort).

    # CLEVER default format
    """
    s = normalize_text(route_text).fillna("")
    left = s.str.split("-", n=1, expand=True)[0].astype("string").str.strip()
    digits = left.str.extract(r"^\s*([0-9A-Za-z]+)\s*$")[0]
    return digits.fillna(left).replace({"": pd.NA}).astype("string")


def normalize_vehicle_id(vehicle_series: pd.Series) -> pd.Series:
    """Normalize vehicle id values (trim whitespace, strip float suffix).

    # CLEVER sometimes exports vehicle IDs as floats (e.g. "7906.0")
    """
    s = normalize_text(vehicle_series)
    return s.str.replace(r"\.0$", "", regex=True).replace({"<NA>": pd.NA}).astype("string")


def normalize_dt_series(series: pd.Series) -> pd.Series:
    """Normalize datetime-like strings prior to pandas parsing."""
    return normalize_text(series)


def parse_dt(series: pd.Series, *, col_name: str) -> pd.Series:
    """Parse datetime series robustly; logs count of unparseable values."""
    dt = pd.to_datetime(
        normalize_dt_series(series),
        errors="coerce",
    )
    bad = int(dt.isna().sum())
    if bad:
        logging.warning("Failed to parse %d values in %r.", bad, col_name)
    return dt


def dt_to_iso(series: pd.Series) -> pd.Series:
    """Datetime -> ISO string (YYYY-MM-DDTHH:MM:SS); NaT -> NA."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S").astype("string")


def warn_missing_columns(df: pd.DataFrame) -> None:
    """Check for missing columns and log warnings or raise errors."""
    missing_req = [c for c in REQ_COLS if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required input columns: {missing_req}")

    missing_opt = [c for c in OPT_COLS if c not in df.columns]
    if missing_opt:
        logging.warning("Missing optional input columns (will continue): %s", missing_opt)


# Column read_input_folder adds to name the file each row came from.
SOURCE_FILE_COL: str = "source_file"


def read_input_folder(
    folder: Path,
    *,
    pattern: str = "*.csv",
    recursive: bool = False,
    drop_duplicates: bool = True,
    exclude: tuple[Path, ...] = (),
) -> pd.DataFrame:
    """Read every CSV in *folder* matching *pattern* into one DataFrame.

    Files are read as text in sorted path order, so the combined rows come out
    in the same order on every run. A file that cannot be read or lacks any of
    ``REQ_COLS`` is skipped with a warning. A file whose columns differ from
    the first file's is kept with a warning; columns it lacks are blank for its
    rows.

    Args:
        folder: Folder to search.
        pattern: Glob pattern for file names, e.g. ``"*.csv"``.
        recursive: Also search subfolders.
        drop_duplicates: Drop rows that exactly repeat a row from an earlier
            file, keeping the earliest file's copy. Repeats within a single
            file are kept, as they are for single-file input.
        exclude: Paths to leave out even when they match, such as this
            script's own output files.

    Returns:
        The combined rows plus a ``SOURCE_FILE_COL`` column holding each row's
        path relative to *folder*. Empty when no file was usable.
    """
    excluded = {p.resolve() for p in exclude}
    matches = folder.rglob(pattern) if recursive else folder.glob(pattern)
    paths = sorted(p for p in matches if p.is_file() and p.resolve() not in excluded)
    if not paths:
        logging.warning("No files matching %r in %s.", pattern, folder)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    first_cols: set[str] | None = None
    for p in paths:
        name = p.relative_to(folder).as_posix()
        try:
            frame = pd.read_csv(p, dtype=str, low_memory=False).dropna(how="all")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            logging.warning("Skipped %s: could not read it (%s).", name, e)
            continue
        missing = [c for c in REQ_COLS if c not in frame.columns]
        if missing:
            logging.warning("Skipped %s: missing required columns %s.", name, missing)
            continue
        cols = set(frame.columns)
        if first_cols is None:
            first_cols = cols
        elif cols != first_cols:
            logging.warning(
                "%s columns differ from the first file's (missing %s, extra %s); "
                "columns a file lacks are blank for its rows.",
                name,
                sorted(first_cols - cols),
                sorted(cols - first_cols),
            )
        logging.info("Read %d rows from %s", len(frame), name)
        frames.append(frame.assign(**{SOURCE_FILE_COL: name}))

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    if drop_duplicates and len(frames) > 1:
        row_cols = [c for c in combined.columns if c != SOURCE_FILE_COL]
        first_source = combined.groupby(row_cols, dropna=False, sort=False)[
            SOURCE_FILE_COL
        ].transform("first")
        repeat = combined[SOURCE_FILE_COL] != first_source
        if repeat.any():
            logging.warning(
                "Dropped %d rows that repeat a row from an earlier file "
                "(overlapping exports); kept the earliest file's copy.",
                int(repeat.sum()),
            )
            combined = combined.loc[~repeat].reset_index(drop=True)

    logging.info(
        "Combined %d of %d matching files: %d rows.", len(frames), len(paths), len(combined)
    )
    return combined


def normalize_timepoint_order(series: pd.Series, trip_key: pd.DataFrame | None = None) -> pd.Series:
    """Parse Timepoint Order as integer-ish sequence.

    Policy:
    - If values parse as numeric:
        - if any are < 0, we drop them to NA and warn.
        - a trip whose smallest value is 0 is zero-based, so its whole
          sequence is shifted by +1 and we warn (schema expects min 1).
          Shifting only the zero would collide with the trip's existing 1
          ([0, 1, 2] -> [1, 1, 2]).
    - If unparseable, NA with warning elsewhere.

    Args:
        series: Raw Timepoint Order values.
        trip_key: Columns identifying each trip (e.g. ``service_date`` and
            ``trip_id_performed``), aligned with *series*. When omitted, the
            whole series is treated as one trip.
    """
    seq = pd.to_numeric(series, errors="coerce")

    neg = seq.notna() & (seq < 0)
    if neg.any():
        logging.warning(
            "Found %d rows with negative %r; setting those sequences to blank.",
            int(neg.sum()),
            COLUMN_MAP["timepoint_order"],
        )
        seq = seq.mask(neg)

    if trip_key is None:
        trip_key = pd.DataFrame({"_trip": 0}, index=seq.index)
    groups = [trip_key[c] for c in trip_key.columns]
    zero_based = seq.groupby(groups, dropna=False).transform("min").eq(0)
    if zero_based.any():
        logging.warning(
            "Found %d trip(s) whose %r starts at 0; shifting each such trip's "
            "sequence by +1 (%d rows).",
            len(trip_key.loc[zero_based].drop_duplicates()),
            COLUMN_MAP["timepoint_order"],
            int(zero_based.sum()),
        )
        seq = seq.mask(zero_based, seq + 1)

    # Keep as nullable Int64
    return seq.round(0).astype("Int64")


def warn_nonconsecutive_sequences(
    df_out: pd.DataFrame,
    *,
    service_date_col: str = "service_date",
    trip_id_col: str = "trip_id_performed",
    seq_col: str = "trip_stop_sequence",
) -> None:
    """Warn if sequences within (service_date, trip_id_performed) are not consecutive starting at 1.

    We do NOT fix them (user asked to accept timepoint order), but we do make it visible.
    """
    if df_out.empty:
        return

    key_cols = [service_date_col, trip_id_col]
    missing_keys = [c for c in key_cols + [seq_col] if c not in df_out.columns]
    if missing_keys:
        return

    tmp = df_out[key_cols + [seq_col]].copy()
    tmp = tmp.dropna(subset=key_cols, how="any")

    def _is_consecutive(s: pd.Series) -> bool:
        vals = pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()
        if not vals:
            return True
        vals_sorted = sorted(set(vals))
        return vals_sorted[0] == 1 and vals_sorted == list(range(1, vals_sorted[-1] + 1))

    bad_groups = 0
    for _, g in tmp.groupby(key_cols, dropna=False):
        if not _is_consecutive(g[seq_col]):
            bad_groups += 1

    if bad_groups:
        logging.warning(
            "Timepoint-based sequencing: %d trip(s) have non-consecutive %r values. "
            "This may violate strict TIDES schema expectations for stop order.",
            bad_groups,
            seq_col,
        )


def warn_duplicate_sequences(
    df_out: pd.DataFrame,
    *,
    service_date_col: str = "service_date",
    trip_id_col: str = "trip_id_performed",
    seq_col: str = "trip_stop_sequence",
) -> None:
    """Warn if a sequence value repeats within (service_date, trip_id_performed).

    The consecutiveness check above looks at distinct values only, so it cannot
    see a repeated value; a repeat makes stop order within the trip ambiguous.
    """
    key_cols = [service_date_col, trip_id_col, seq_col]
    if df_out.empty or any(c not in df_out.columns for c in key_cols):
        return

    tmp = df_out[key_cols].dropna(subset=[seq_col])
    dup = tmp.duplicated(keep=False)
    if dup.any():
        logging.warning(
            "%d trip(s) repeat a %r value (%d rows); stop order within those trips is ambiguous.",
            len(tmp.loc[dup, [service_date_col, trip_id_col]].drop_duplicates()),
            seq_col,
            int(dup.sum()),
        )


# =============================================================================
# CORE CONVERSION
# =============================================================================


def convert_to_tides(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an AVL stop visit export to TIDES stop_visits.csv."""
    df = df.dropna(how="all").copy()
    warn_missing_columns(df)

    out = pd.DataFrame(index=df.index)

    # service_date
    out["service_date"] = parse_service_date(df[COLUMN_MAP["date"]])

    # Trip token + vehicle_id (both feed trip_id_performed below)
    trip_token = split_trip_token(df[COLUMN_MAP["trip"]])

    if COLUMN_MAP["vehicle"] in df.columns:
        vehicle_id = normalize_vehicle_id(df[COLUMN_MAP["vehicle"]])
    else:
        vehicle_id = pd.Series(pd.NA, index=df.index, dtype="string")

    # trip_id_performed
    trip_id_mode = str(TRIP_ID_MODE).strip().lower()
    if trip_id_mode not in {"token", "hashed"}:
        logging.warning("TRIP_ID_MODE=%r is invalid; defaulting to 'token'.", TRIP_ID_MODE)
        trip_id_mode = "token"

    # Hash inputs match convert_to_tides_trips_performed.choose_trip_id_performed
    # exactly; change both together or the two outputs stop joining.
    hashed = pd.Series(
        [
            f"perf_{stable_id(str(sd), str(tt), str(vv))}"
            for sd, tt, vv in zip(
                out["service_date"].fillna(""), trip_token.fillna(""), vehicle_id.fillna("")
            )
        ],
        index=df.index,
        dtype="string",
    ).replace({"perf_" + stable_id("", "", ""): pd.NA})

    if trip_id_mode == "token":
        # A token run by more than one vehicle on a date is a split or doubled
        # trip; hash each vehicle's instance, as the trips_performed converter does.
        n_vehicles = vehicle_id.groupby([out["service_date"], trip_token], dropna=False).transform(
            "nunique"
        )
        multi = (n_vehicles > 1) & trip_token.notna()
        if multi.any():
            logging.warning(
                "%d rows belong to trip tokens run by more than one vehicle on a "
                "service_date; used hashed trip_id_performed (service_date, trip token, "
                "vehicle_id) for those rows.",
                int(multi.sum()),
            )
        out["trip_id_performed"] = trip_token.where(~multi, hashed)
    else:
        out["trip_id_performed"] = hashed

    # sequences + stop_id
    seq = normalize_timepoint_order(
        df[COLUMN_MAP["timepoint_order"]], out[["service_date", "trip_id_performed"]]
    )
    bad_seq = int(seq.isna().sum())
    if bad_seq:
        logging.warning(
            "Failed to parse %d %r values as integers; emitting blanks for those rows.",
            bad_seq,
            COLUMN_MAP["timepoint_order"],
        )

    out["trip_stop_sequence"] = seq
    out["scheduled_stop_sequence"] = seq

    out["stop_id"] = normalize_text(df[COLUMN_MAP["timepoint_id"]])

    # vehicle_id
    out["vehicle_id"] = vehicle_id
    if out["vehicle_id"].isna().any():
        logging.warning("vehicle_id is missing on %d rows.", int(out["vehicle_id"].isna().sum()))

    # pattern_id (synthetic)
    route_short = (
        parse_route_short(df[COLUMN_MAP["route"]])
        if COLUMN_MAP["route"] in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    direction = (
        normalize_text(df[COLUMN_MAP["direction"]])
        if COLUMN_MAP["direction"] in df.columns
        else pd.Series(pd.NA, index=df.index)
    )

    if COLUMN_MAP["variation"] in df.columns:
        var = (
            pd.to_numeric(df[COLUMN_MAP["variation"]], errors="coerce")
            .astype("Int64")
            .astype("string")
        )
        var = var.replace({"<NA>": pd.NA})
    else:
        var = pd.Series(pd.NA, index=df.index, dtype="string")

    pattern = (
        route_short.fillna("").astype("string").str.strip()
        + "|"
        + direction.fillna("").astype("string").str.strip()
        + "|"
        + var.fillna("").astype("string").str.strip()
    ).str.strip("|")
    out["pattern_id"] = pattern.replace({"": pd.NA}).astype("string")

    # timepoint flag (boolean per schema)
    out["timepoint"] = True

    # --- Times: scheduled + actual
    # Scheduled Passing Time -> schedule_arrival_time + schedule_departure_time
    if COLUMN_MAP["scheduled_passing"] in df.columns:
        sched_dt = parse_dt(
            df[COLUMN_MAP["scheduled_passing"]], col_name=COLUMN_MAP["scheduled_passing"]
        )
        sched_iso = dt_to_iso(sched_dt).where(sched_dt.notna(), pd.NA)
        out["schedule_arrival_time"] = sched_iso
        out["schedule_departure_time"] = sched_iso
    else:
        out["schedule_arrival_time"] = pd.NA
        out["schedule_departure_time"] = pd.NA

    # Arrival/Departure -> actual arrival/departure; fallback to Actual Time
    arrival_dt = (
        parse_dt(df[COLUMN_MAP["arrival_time"]], col_name=COLUMN_MAP["arrival_time"])
        if COLUMN_MAP["arrival_time"] in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    depart_dt = (
        parse_dt(df[COLUMN_MAP["departure_time"]], col_name=COLUMN_MAP["departure_time"])
        if COLUMN_MAP["departure_time"] in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    actual_dt = (
        parse_dt(df[COLUMN_MAP["actual_time"]], col_name=COLUMN_MAP["actual_time"])
        if COLUMN_MAP["actual_time"] in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )

    arrival_best = arrival_dt.where(arrival_dt.notna(), actual_dt)
    depart_best = depart_dt.where(depart_dt.notna(), actual_dt)

    out["actual_arrival_time"] = dt_to_iso(arrival_best).where(arrival_best.notna(), pd.NA)
    out["actual_departure_time"] = dt_to_iso(depart_best).where(depart_best.notna(), pd.NA)

    # dwell seconds (only when both present and non-negative)
    dwell_sec = (depart_best - arrival_best).dt.total_seconds()
    dwell_ok = dwell_sec.notna() & (dwell_sec >= 0)
    if (~dwell_ok & dwell_sec.notna()).any():
        logging.warning(
            "Found %d rows with negative dwell (departure before arrival); "
            "leaving dwell blank for those rows.",
            int((~dwell_ok & dwell_sec.notna()).sum()),
        )
    out["dwell"] = dwell_sec.where(dwell_ok, pd.NA).round(0).astype("Int64")

    # Fields CLEVER Stop Visit Events doesn't provide: leave blank
    blanks = [
        "distance",
        "boarding_1",
        "alighting_1",
        "boarding_2",
        "alighting_2",
        "departure_load",
        "door_open",
        "door_close",
        "door_status",
        "ramp_deployed_time",
        "ramp_failure",
        "kneel_deployed_time",
        "lift_deployed_time",
        "bike_rack_deployed",
        "bike_load",
        "revenue",
        "number_of_transactions",
    ]
    for c in blanks:
        out[c] = pd.NA

    # schedule_relationship: schema enum value
    out["schedule_relationship"] = "Scheduled"

    # Warn about strict-schema expectations we are knowingly violating / risking
    warn_nonconsecutive_sequences(out)
    warn_duplicate_sequences(out)

    # Minimal sanity warnings
    if out["service_date"].isna().any():
        logging.warning(
            "service_date is missing on %d rows.", int(out["service_date"].isna().sum())
        )
    if out["stop_id"].isna().any():
        logging.warning("stop_id is missing on %d rows.", int(out["stop_id"].isna().sum()))
    if out["trip_id_performed"].isna().any():
        logging.warning(
            "trip_id_performed is missing on %d rows.", int(out["trip_id_performed"].isna().sum())
        )
    if out["trip_stop_sequence"].isna().any():
        logging.warning(
            "trip_stop_sequence is missing on %d rows (unparseable %r).",
            int(out["trip_stop_sequence"].isna().sum()),
            COLUMN_MAP["timepoint_order"],
        )

    # Final order + stable dtypes for CSV output
    out = out.reindex(columns=TIDES_COLS)

    # Convert nullable Int64 columns to plain strings/blank for cleaner CSV, if desired.
    # Keep as numeric if you prefer strict typing in downstream ingestion.
    # Here we keep numeric for seq + dwell, which is generally what schemas expect.
    return out


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


# Canonical version lives in utils/run_log.py — keep this copy in sync.
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


def run_log_path(output_file: Path) -> Path:
    """Return the run-log sidecar path for *output_file* (same folder and stem)."""
    return output_file.with_name(f"{output_file.stem}_runlog.txt")


def write_run_log(output_file: Path, summary_lines: list[str]) -> bool:
    """Write the run-log sidecar for *output_file*.

    The log records the run timestamp, the source script, *summary_lines*, and
    the CONFIGURATION block verbatim, so it cannot drift from the settings that
    produced the output.

    Args:
        output_file: The stop_visits CSV this run wrote.
        summary_lines: What the run read and wrote, such as the rows kept from
            each input file. The CONFIGURATION block records only a folder's
            path, and the folder's contents can change after the run.

    Returns:
        ``True`` if the log was written, ``False`` otherwise.
    """
    log_path = run_log_path(output_file)

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

    lines: list[str] = [
        "=" * 72,
        "TIDES STOP VISITS CONVERSION RUN LOG",
        "=" * 72,
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Output file:   {output_file}",
        f"Source script: {source_display}",
        "",
        "-" * 72,
        "RUN SUMMARY",
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
# MAIN
# =============================================================================


def main() -> int:
    """Run the conversion pipeline.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if INPUT_PATH == Path(r"Path\To\Stop Visit Events.csv") or OUTPUT_DIR == Path(
        r"Path\To\Your\Output_Folder"
    ):
        logging.warning(
            "INPUT_PATH and/or OUTPUT_DIR are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2

    if not INPUT_PATH.exists():
        logging.warning(
            "INPUT_PATH does not exist: %s — update INPUT_PATH in the CONFIGURATION "
            "section to your actual Stop Visit Events export (or a folder of them) "
            "before running.",
            INPUT_PATH,
        )
        logging.info("Completed (no data processed — update INPUT_PATH to proceed).")
        return 1

    # Read as text: type inference would turn IDs like "00123" into 123 (or
    # "123.0" in a column with blanks) and break joins to trips_performed.
    summary: list[str] = []
    if INPUT_PATH.is_dir():
        combined_csv = OUTPUT_DIR / COMBINED_INPUT_FILENAME if COMBINED_INPUT_FILENAME else None
        own_outputs = (OUTPUT_CSV, run_log_path(OUTPUT_CSV), combined_csv)
        combined = read_input_folder(
            INPUT_PATH,
            pattern=INPUT_GLOB,
            recursive=INPUT_RECURSIVE,
            drop_duplicates=DROP_DUPLICATE_ROWS,
            # A rerun must not read back this script's own outputs.
            exclude=tuple(p for p in own_outputs if p is not None),
        )
        if combined.empty:
            logging.warning("No usable stop visit rows in %s; nothing written.", INPUT_PATH)
            return 1
        kept = combined.groupby(SOURCE_FILE_COL, sort=False).size()
        summary += [
            f"Input folder: {INPUT_PATH}",
            "Rows kept per input file (after dropping rows repeated across files):",
            *(f"  {name}: {n}" for name, n in kept.items()),
        ]
        if combined_csv is not None:
            combined_csv.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(combined_csv, index=False)
            logging.info("Wrote combined input (%d rows) -> %s", len(combined), combined_csv)
            summary.append(f"Combined input written: {combined_csv}")
        df = combined.drop(columns=SOURCE_FILE_COL)
    else:
        df = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)
        summary.append(f"Input file: {INPUT_PATH}")
    logging.info("Read %d rows, %d columns", len(df), df.shape[1])

    out = convert_to_tides(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logging.info("Wrote %d rows -> %s", len(out), OUTPUT_CSV)

    summary += [f"Input rows:  {len(df)}", f"Output rows: {len(out)}"]
    # Sidecar run log — required by default so outputs are always traceable.
    if not write_run_log(OUTPUT_CSV, summary) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        return 1

    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
