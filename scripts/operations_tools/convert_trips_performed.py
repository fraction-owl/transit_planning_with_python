"""Convert an AVL runtime export into TIDES-compliant trips_performed records.

This module reads an AVL "Event Runtime Analysis" report (CSV) and produces a
`trips_performed.csv` file that conforms to the TIDES `trips_performed` schema.
It normalizes common real-world export issues (inconsistent whitespace, AM/PM
timestamps, mixed null tokens) and applies schema-aligned data quality rules so
the output can be ingested by downstream validation and analytics pipelines.

Key behaviors:
- Parses scheduled and actual timestamps robustly (tolerates AM/PM and extra
  whitespace). Missing start/end times are permitted; rows are not dropped solely
  for partial timing data.
- Derives `service_date` from Scheduled Start Time when available, otherwise
  falls back to Actual Start Time. Rows that cannot be dated are excluded.
- Requires `vehicle_id` (as per schema). Rows with missing or unusable vehicle
  identifiers are excluded and counted in logs.
- Extracts `route_id` from the human-readable Route field by taking the token to
  the left of "-" and trimming whitespace (e.g., "301 - Telegraph Rd" -> "301").
- Preserves the source TripID as `trip_id_scheduled` when it matches the GTFS
  `trip_id`. `trip_id_performed` is chosen to remain unique within `service_date`
  (using the scheduled trip id when safe, otherwise a stable derived identifier).
- Optionally filters to a single Trip Type (e.g., "Revenue"). When enabled, the
  module logs how many rows were removed by each other trip type.
- Maps source Trip Type values into the schema-constrained TIDES `trip_type`
  enumeration (e.g., "Revenue" -> "In service", "Pull-In" -> "Pullin").

The conversion is designed to be deterministic: given the same input file and
configuration, the output identifiers and records are stable across runs.

Column names are configured via COLUMN_MAP in the CONFIGURATION section.
Defaults are pre-set for CLEVER "Event Runtime Analysis" exports; update
COLUMN_MAP values to adapt to other AVL sources.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV: Path = Path(r"Path\To\Event Runtime Analysis.csv")
OUTPUT_CSV: Path = Path(r"Path\To\trips_performed.csv")

# If set (e.g., "Revenue"), keeps only Trip Type == this value.
# If None/blank, keeps everything and logs nothing about filtering.
KEEP_TRIP_TYPE: str | None = "Revenue"

# Map generic field roles to actual column names in your input file.
# Defaults are set for CLEVER "Event Runtime Analysis" exports.
# Change values here to adapt to other AVL sources; do not rename the keys.
COLUMN_MAP: dict[str, str | None] = {
    "vehicle": "Vehicle",
    "route": "Route",
    "direction": "Direction",
    "block": "Block",
    "trip_id": "TripID",
    "trip_type": "Trip Type",
    "sched_start": "Scheduled Start Time",
    "sched_end": "Scheduled Finish Time",
    "actual_start": "Actual Start Time",
    "actual_end": "Actual Finish Time",
    "operator": "Operator",  # set to None if not present
    "trip_start_stop": None,  # e.g. "First Stop"; None = omit
    "trip_end_stop": "Last Stop",  # e.g. "Last Stop";  None = omit
}

# Direction mapping. Must be 0/1 if present; unmapped values -> NA.
DIRECTION_TEXT_TO_ID: dict[str, int] = {
    "NORTHBOUND": 0,
    "WESTBOUND": 0,
    "SOUTHBOUND": 1,
    "EASTBOUND": 1,
}

# Source trip_type values -> TIDES schema trip_type enum
# Allowed enum values include: "In service", "Deadhead", "Pullout", "Pullin", ...
# (see schema for full list)
CLEVER_TO_TIDES_TRIP_TYPE: dict[str, str] = {
    "REVENUE": "In service",
    "DEADHEAD": "Deadhead",
    "LAYOVER": "Layover",
    "PULL-OUT": "Pullout",
    "PULL OUT": "Pullout",
    "PULL OUT ": "Pullout",
    "PULLOUT": "Pullout",
    "PULL-IN": "Pullin",
    "PULL IN": "Pullin",
    "PULLIN": "Pullin",
    # If you have agency-specific labels, map them here.
}

# Output columns per schema (do NOT include "date"—schema does not define it).
TIDES_COLS: list[str] = [
    "service_date",
    "trip_id_performed",
    "vehicle_id",
    "trip_id_scheduled",
    "route_id",
    "route_type",
    "shape_id",
    "pattern_id",
    "direction_id",
    "operator_id",
    "block_id",
    "trip_start_stop_id",
    "trip_end_stop_id",
    "schedule_trip_start",
    "schedule_trip_end",
    "actual_trip_start",
    "actual_trip_end",
    "trip_type",
    "schedule_relationship",
    "ntd_mode",
    "route_type_agency",
]

# Required input columns for this converter (excluding optional operator/stops).
REQ_COLS: list[str] = [
    COLUMN_MAP[k]
    for k in (
        "vehicle",
        "route",
        "direction",
        "block",
        "trip_id",
        "trip_type",
        "sched_start",
        "sched_end",
        "actual_start",
        "actual_end",
    )
]

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# HELPERS
# =============================================================================


def normalize_dt_series(series: pd.Series) -> pd.Series:
    """Normalize datetime-like strings prior to pandas parsing."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "N/A": pd.NA})
    )


def parse_route_id_from_route_text(route_series: pd.Series) -> pd.Series:
    """Parse route_id from Route text left of '-' (trimmed).

    # CLEVER default format: "301 - Telegraph Rd" -> "301"
    """
    route_text = route_series.astype("string").fillna("").str.strip()
    left = route_text.str.split("-", n=1, expand=True)[0].str.strip()

    # Prefer a 1–4 digit token when present; otherwise keep the left token.
    digits = left.str.extract(r"^\s*([0-9]{1,4})\s*$")[0]
    return digits.fillna(left).replace({"": pd.NA})


def normalize_vehicle_id(vehicle_series: pd.Series) -> pd.Series:
    """Normalize vehicle_id values (trim whitespace, strip float suffix).

    # CLEVER sometimes exports vehicle IDs as floats (e.g. "7785.0")
    """
    raw = vehicle_series.astype("string").str.strip()
    return (
        raw.str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
    )


def stable_id(*parts: str) -> str:
    """Short deterministic ID from multiple fields (sha1 truncated)."""
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def direction_id_from_text(series: pd.Series) -> pd.Series:
    """Map Direction text to direction_id (nullable Int64)."""
    s = series.astype("string").str.strip().str.upper()
    mapped = s.map(DIRECTION_TEXT_TO_ID)
    return mapped.astype("Int64")


def dt_to_iso(series: pd.Series) -> pd.Series:
    """Datetime -> ISO string (YYYY-MM-DDTHH:MM:SS); NaT -> NaN."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S")


def summarize_trip_type_drops(
    df: pd.DataFrame,
    trip_type_col: str,
    *,
    keep_trip_type: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Optionally filter to a single trip type and return drop counts by type."""
    if not keep_trip_type:
        return df, {}

    keep_norm = str(keep_trip_type).strip()
    if not keep_norm:
        return df, {}

    s = df[trip_type_col].astype("string").str.strip()
    mask_keep = s.eq(keep_norm)

    dropped = s.loc[~mask_keep].fillna("<NA>")
    dropped_counts = dropped.value_counts(dropna=False).to_dict()
    return df.loc[mask_keep].copy(), {str(k): int(v) for k, v in dropped_counts.items()}


def map_clever_trip_type_to_tides(series: pd.Series) -> pd.Series:
    """Map source Trip Type labels to schema-conformant TIDES trip_type values."""
    s = series.astype("string").str.strip()
    upper = s.str.upper()

    mapped = upper.map(CLEVER_TO_TIDES_TRIP_TYPE)

    # If not mapped:
    # - If original is missing -> NA
    # - Else default to "Other not in service" (schema enum)
    out = mapped.where(mapped.notna(), other="Other not in service")
    out = out.where(s.notna(), other=pd.NA)
    return out


def choose_trip_id_performed(
    service_date: pd.Series,
    trip_id_scheduled: pd.Series,
    vehicle_id: pd.Series,
    best_start_dt: pd.Series,
) -> tuple[pd.Series, int]:
    """Use GTFS trip_id as trip_id_performed when unique within service_date; else hash.

    Returns:
        (trip_id_performed_series, n_dupe_rows)
    """
    base = pd.DataFrame(
        {
            "service_date": service_date.astype("string"),
            "trip_id_scheduled": trip_id_scheduled.astype("string"),
        }
    )

    dup = base.duplicated(keep=False) & trip_id_scheduled.notna()

    best_start_str = best_start_dt.astype("datetime64[ns]").astype("string").fillna("")
    hashed = pd.Series(
        [
            f"perf_{stable_id(str(sd), str(tid), str(veh), str(bs))}"
            for sd, tid, veh, bs in zip(
                service_date.fillna(""),
                trip_id_scheduled.fillna(""),
                vehicle_id.fillna(""),
                best_start_str,
            )
        ],
        index=service_date.index,
        dtype="string",
    )

    perf = trip_id_scheduled.where(~dup, other=hashed)
    return perf, int(dup.sum())


# =============================================================================
# CORE CONVERSION
# =============================================================================


def convert_to_tides(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an AVL runtime export to TIDES trips_performed.csv."""
    df = df.dropna(how="all").copy()

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    # Optional trip type filter (keep only Revenue, etc.)
    df, dropped_counts = summarize_trip_type_drops(
        df,
        COLUMN_MAP["trip_type"],
        keep_trip_type=KEEP_TRIP_TYPE,
    )
    if dropped_counts:
        total_dropped = sum(dropped_counts.values())
        logging.warning(
            "Trip Type filter enabled (keep=%r): dropped %d rows of other types: %s",
            KEEP_TRIP_TYPE,
            total_dropped,
            dropped_counts,
        )

    if df.empty:
        logging.warning("No rows remain after Trip Type filtering.")
        return pd.DataFrame(columns=TIDES_COLS)

    # Parse datetimes robustly
    sched_start_dt = pd.to_datetime(
        normalize_dt_series(df[COLUMN_MAP["sched_start"]]),
        errors="coerce",
        infer_datetime_format=True,
    )
    sched_end_dt = pd.to_datetime(
        normalize_dt_series(df[COLUMN_MAP["sched_end"]]),
        errors="coerce",
        infer_datetime_format=True,
    )
    actual_start_dt = pd.to_datetime(
        normalize_dt_series(df[COLUMN_MAP["actual_start"]]),
        errors="coerce",
        infer_datetime_format=True,
    )
    actual_end_dt = pd.to_datetime(
        normalize_dt_series(df[COLUMN_MAP["actual_end"]]),
        errors="coerce",
        infer_datetime_format=True,
    )

    # Date the trip: prefer scheduled start; fallback to actual start
    best_start_dt = sched_start_dt.where(sched_start_dt.notna(), actual_start_dt)

    # Drop rows that cannot be dated at all
    mask_undated = best_start_dt.isna()
    if mask_undated.any():
        n_drop = int(mask_undated.sum())
        logging.warning(
            "Dropping %d rows: neither Scheduled Start Time nor Actual Start Time is parseable.",
            n_drop,
        )
        df = df.loc[~mask_undated].copy()
        sched_start_dt = sched_start_dt.loc[~mask_undated]
        sched_end_dt = sched_end_dt.loc[~mask_undated]
        actual_start_dt = actual_start_dt.loc[~mask_undated]
        actual_end_dt = actual_end_dt.loc[~mask_undated]
        best_start_dt = best_start_dt.loc[~mask_undated]

    # vehicle_id is required by schema
    vehicle_id = normalize_vehicle_id(df[COLUMN_MAP["vehicle"]])
    mask_no_vehicle = vehicle_id.isna()
    if mask_no_vehicle.any():
        n_drop = int(mask_no_vehicle.sum())
        logging.warning("Dropping %d rows: missing Vehicle / vehicle_id.", n_drop)
        df = df.loc[~mask_no_vehicle].copy()
        vehicle_id = vehicle_id.loc[~mask_no_vehicle]
        sched_start_dt = sched_start_dt.loc[~mask_no_vehicle]
        sched_end_dt = sched_end_dt.loc[~mask_no_vehicle]
        actual_start_dt = actual_start_dt.loc[~mask_no_vehicle]
        actual_end_dt = actual_end_dt.loc[~mask_no_vehicle]
        best_start_dt = best_start_dt.loc[~mask_no_vehicle]

    if df.empty:
        logging.warning("No rows remain after dropping missing vehicle_id.")
        return pd.DataFrame(columns=TIDES_COLS)

    out = pd.DataFrame(index=df.index)

    # Required schema fields
    out["service_date"] = best_start_dt.dt.date.astype("string")
    out["vehicle_id"] = vehicle_id

    # GTFS trip_id from source trip_id field
    out["trip_id_scheduled"] = (
        df[COLUMN_MAP["trip_id"]].astype("string").str.strip().replace({"": pd.NA})
    )

    # Per schema: trip_id_performed must be unique within service_date
    out["trip_id_performed"], n_dupe_rows = choose_trip_id_performed(
        out["service_date"],
        out["trip_id_scheduled"],
        out["vehicle_id"],
        best_start_dt,
    )
    if n_dupe_rows:
        logging.warning(
            "trip_id_scheduled duplicates within service_date for %d rows; "
            "used hashed trip_id_performed for those rows.",
            n_dupe_rows,
        )

    # Optional schema fields we can populate
    out["route_id"] = parse_route_id_from_route_text(df[COLUMN_MAP["route"]])
    out["direction_id"] = direction_id_from_text(df[COLUMN_MAP["direction"]])
    out["block_id"] = df[COLUMN_MAP["block"]].astype("string").str.strip().replace({"": pd.NA})

    if COLUMN_MAP["operator"] and COLUMN_MAP["operator"] in df.columns:
        out["operator_id"] = (
            df[COLUMN_MAP["operator"]].astype("string").str.strip().replace({"": pd.NA})
        )
    else:
        out["operator_id"] = pd.NA

    # Stops (text IDs are fine)
    if COLUMN_MAP["trip_start_stop"] and COLUMN_MAP["trip_start_stop"] in df.columns:
        out["trip_start_stop_id"] = (
            df[COLUMN_MAP["trip_start_stop"]].astype("string").str.strip().replace({"": pd.NA})
        )
    else:
        out["trip_start_stop_id"] = pd.NA

    if COLUMN_MAP["trip_end_stop"] and COLUMN_MAP["trip_end_stop"] in df.columns:
        out["trip_end_stop_id"] = (
            df[COLUMN_MAP["trip_end_stop"]].astype("string").str.strip().replace({"": pd.NA})
        )
    else:
        out["trip_end_stop_id"] = pd.NA

    # Times (allowed to be missing)
    out["schedule_trip_start"] = dt_to_iso(sched_start_dt)
    out["schedule_trip_end"] = dt_to_iso(sched_end_dt)
    out["actual_trip_start"] = dt_to_iso(actual_start_dt)
    out["actual_trip_end"] = dt_to_iso(actual_end_dt)

    # trip_type must match schema enum; map from source Trip Type
    out["trip_type"] = map_clever_trip_type_to_tides(df[COLUMN_MAP["trip_type"]])

    # Schedule relationship enum
    out["schedule_relationship"] = "Scheduled"

    # Fields not available from this export (leave blank)
    out["route_type"] = pd.NA
    out["shape_id"] = pd.NA
    out["pattern_id"] = pd.NA
    out["ntd_mode"] = pd.NA
    out["route_type_agency"] = pd.NA

    # Final order (schema-defined columns; extras left out)
    out = out.reindex(columns=TIDES_COLS)

    # Log missing actual times (expected; do not drop)
    n_missing_actual_start = int(out["actual_trip_start"].isna().sum())
    n_missing_actual_end = int(out["actual_trip_end"].isna().sum())
    if n_missing_actual_start or n_missing_actual_end:
        logging.warning(
            "Kept rows with missing actual times: %d missing actual start, %d missing actual end.",
            n_missing_actual_start,
            n_missing_actual_end,
        )

    # Required field checks per schema (service_date, trip_id_performed, vehicle_id)
    for req in ("service_date", "trip_id_performed", "vehicle_id"):
        if out[req].isna().any():
            bad = out.loc[out[req].isna()].head(10)
            raise ValueError(f"Nulls found in required column '{req}'. Sample:\n{bad}")

    return out


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the AVL -> TIDES trips_performed conversion pipeline."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if INPUT_CSV == Path(r"Path\To\Event Runtime Analysis.csv") or OUTPUT_CSV == Path(
        r"Path\To\trips_performed.csv"
    ):
        logging.warning(
            "INPUT_CSV and/or OUTPUT_CSV are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return

    if not INPUT_CSV.exists():
        logging.warning(
            "INPUT_CSV path does not exist: %s — update INPUT_CSV in the CONFIGURATION "
            "section to your actual Event Runtime Analysis export before running.",
            INPUT_CSV,
        )
        logging.info("Completed (no data processed — update INPUT_CSV to proceed).")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    logging.info("Read %d rows, %d columns", len(df), df.shape[1])

    out = convert_to_tides(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logging.info("Wrote %d rows -> %s", len(out), OUTPUT_CSV)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
