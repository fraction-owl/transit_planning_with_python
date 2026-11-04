"""Flag stops needing amenity upgrades using ridership thresholds and amenity data.

This script reads stop-level ridership and amenity information from one or
(optionally) two Excel workbooks. It normalizes and (optionally) aggregates
duplicate STOP_IDs, then computes boolean "FLAG_*" columns to highlight where
usage warrants an upgrade but the amenity is missing.

Amenity values are read as Y/N (Yes/No and True/False are also accepted, as
are counts: above zero means present, zero means absent). Where both workbooks
carry an amenity, the amenity workbook wins and the ridership workbook fills
its blanks. Values neither source supplies are reported as UNKNOWN; a stop at
or above a threshold with an UNKNOWN amenity gets a "REVIEW_*" column instead
of a flag. Stop IDs are trimmed and a trailing ".0" is removed ("1257.0"
matches "1257"); rows with a blank stop ID are dropped, and duplicate amenity
records for one stop are collapsed (Y if any record says Y) before joining.
The log reports how many stops in each workbook have no match in the other.
Thousands separators in ridership ("1,250") are removed; other non-numeric
values are logged and counted as zero. Set ``AMENITIES_XLSX = None`` to use
the ridership workbook alone.

Outputs
-------
Both files are written to ``OUTPUT_FOLDER``:

    stops_needing_improvement.xlsx:
        Raw Data – Unmodified import of the ridership workbook.
        All Flags – Every stop plus boolean flag and review columns for each
        amenity and NEEDS_IMPROVEMENT / NEEDS_REVIEW summary flags.
        Shelter / Bench / TrashCan / Pad – One sheet per amenity, listing only
        the stops that require that specific upgrade.
        Needs Review – Stops at or above a threshold whose amenity is UNKNOWN.
    stops_needing_improvement.txt: A concise summary of flagged and review
    stops by category, followed by a detailed, human-readable list of each.

Typical use-cases include batch reviews of bus-stop needs based on ArcGIS
outputs, planning reports, or independently maintained amenity inventories.

Typical usage
-------------
Update the paths in the CONFIGURATION section and run from a shell or a
Jupyter notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ridership source workbook
RIDERSHIP_XLSX: Path = Path(r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).xlsx")
RIDERSHIP_SHEET: int | str = 0
# Output folder (Excel + TXT will be written here)
OUTPUT_FOLDER: Path = Path(r"Your\Folder\Path\To\Output")

# Fields in ridership workbook
RIDERSHIP_FIELD: str = "XBOARDINGS"
STOP_ID_FIELD: str = "STOP_ID"

# Amenity thresholds and fields (must match column names after standardisation)
AMENITIES: Dict[str, Dict[str, Any]] = {
    "Shelter": {"field": "SHELTER", "thresh": 25},
    "Bench": {"field": "BENCH", "thresh": 10},
    "TrashCan": {"field": "TRASHCAN", "thresh": 10},
    "Pad": {"field": "PAD", "thresh": 1},
}

# Aggregation behaviour: True | False | "auto"
AGGREGATE_BY_STOP: bool | str = "auto"

# -----------------------------------------------------------------------------
# OPTIONAL SECOND WORKBOOK – AMENITY DETAILS
# -----------------------------------------------------------------------------

# Set to None to skip the amenity workbook and use only the ridership workbook.
AMENITIES_XLSX: Path | None = Path(r"Your\File\Path\To\bus_stop_amenities.xlsx")
AMENITIES_SHEET: int | str = 0
AMENITY_JOIN_FIELD: str = "stop_code"
TXT_LOG_PATH: Path = OUTPUT_FOLDER / "stops_needing_improvement.txt"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# -----------------------------------------------------------------------------
# AMENITY FIELD MAPPINGS
# -----------------------------------------------------------------------------

_AMENITY_ALIASES: Dict[str, str] = {
    "bus_shelte": "SHELTER",
    "bus_shelter": "SHELTER",
    "pad": "PAD",
    "bench": "BENCH",
    "trash_can": "TRASHCAN",
    "trashcan": "TRASHCAN",
}

# Recognised amenity values (compared after strip + uppercase). Counts are also
# accepted: above zero is present, zero is absent. Blank cells and anything
# unrecognised (including negative counts) are treated as unknown, not absent.
_PRESENT_VALUES = frozenset({"Y", "YES", "TRUE", "T", "1", "1.0"})
_ABSENT_VALUES = frozenset({"N", "NO", "FALSE", "F", "0", "0.0"})
UNKNOWN_VALUE: str = "UNKNOWN"

# =============================================================================
# FUNCTIONS
# =============================================================================


def _standardise_yn(series: pd.Series) -> pd.Series:
    """Normalise an amenity column to 'Y', 'N', or missing (unknown).

    Blank cells stay missing so that a later source can fill them; they are
    not treated as a confirmed absence.
    """
    text = series.astype("string").str.strip().str.upper()
    counts = pd.to_numeric(text, errors="coerce")
    is_count = ((counts >= 0) & (counts < float("inf"))).fillna(False)
    present = text.isin(_PRESENT_VALUES).fillna(False) | (is_count & (counts > 0))
    absent = text.isin(_ABSENT_VALUES).fillna(False) | (is_count & (counts == 0))
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result[present.fillna(False)] = "Y"
    result[absent.fillna(False)] = "N"

    known = _PRESENT_VALUES | _ABSENT_VALUES | {"", UNKNOWN_VALUE}
    unrecognised = text.notna() & ~(text.isin(known).fillna(False) | is_count)
    if unrecognised.any():
        logging.warning(
            "Column '%s': unrecognised amenity value(s) %s treated as unknown.",
            series.name,
            sorted(series[unrecognised].astype(str).unique()),
        )
    return result


def _apply_amenity_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and rename known amenity aliases to standard fields."""
    df.columns = [str(c).strip() for c in df.columns]
    rename: Dict[str, str] = {}
    for alias, field in _AMENITY_ALIASES.items():
        if alias in df.columns and field not in df.columns and field not in rename.values():
            rename[alias] = field
    return df.rename(columns=rename)


def _drop_missing_ids(df: pd.DataFrame, id_field: str, source: str) -> pd.DataFrame:
    """Strip stop IDs and drop rows whose ID is blank, so they cannot match each other.

    A trailing ".0" left by a numeric export is removed ("1257.0" -> "1257");
    leading zeros are kept.
    """
    ids = df[id_field].astype("string").str.strip().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    missing = ids.isna() | (ids == "")
    if missing.any():
        logging.warning(
            "Dropped %d %s row(s) with a blank '%s'.", int(missing.sum()), source, id_field
        )
    df = df.loc[~missing].copy()
    df[id_field] = ids[~missing].astype(str)
    return df


def _load_ridership_data(path: Path, sheet: int | str) -> pd.DataFrame:
    """Load ridership workbook, coercing all columns to strings."""
    return pd.read_excel(path, sheet_name=sheet, dtype=str)


def _prepare_amenity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every expected amenity column exists as 'Y', 'N', or UNKNOWN."""
    for cfg in AMENITIES.values():
        col = cfg["field"]
        if col not in df.columns:
            logging.warning(
                "Amenity column '%s' not found; recording it as %s.", col, UNKNOWN_VALUE
            )
            df[col] = pd.NA
        df[col] = _standardise_yn(df[col]).fillna(UNKNOWN_VALUE)
    return df


def _convert_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the ridership column to numeric, keeping decimals; non-numerics become zero.

    Thousands separators ("1,250") are removed first. Values that are still not
    numbers are logged before being set to zero; blanks become zero silently.
    """
    text = df[RIDERSHIP_FIELD].astype("string").str.strip().str.replace(",", "", regex=False)
    values = pd.to_numeric(text, errors="coerce").astype("float64")
    invalid = values.isna() & text.fillna("").ne("")
    if invalid.any():
        logging.warning(
            "Column '%s': %d non-numeric value(s) %s treated as zero.",
            RIDERSHIP_FIELD,
            int(invalid.sum()),
            sorted(df.loc[invalid, RIDERSHIP_FIELD].astype(str).unique()),
        )
    df[RIDERSHIP_FIELD] = values.fillna(0.0)
    return df


def _needs_aggregation(df: pd.DataFrame) -> bool:
    """Decide if STOP_ID-level aggregation should be performed."""
    decision_map = {
        True: True,
        False: False,
        "auto": df[STOP_ID_FIELD].duplicated().any(),
    }
    return decision_map[AGGREGATE_BY_STOP]


def _combine_amenity_values(values: pd.Series) -> Any:
    """Collapse one stop's amenity values: 'Y' if any 'Y', else 'N' if any 'N', else missing."""
    if (values == "Y").any():
        return "Y"
    if (values == "N").any():
        return "N"
    return pd.NA


def _aggregate_by_stop(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate STOP_ID rows: sum ridership, OR amenities."""
    agg_map: Dict[str, Any] = {RIDERSHIP_FIELD: "sum"}
    amenity_cols = [cfg["field"] for cfg in AMENITIES.values()]
    for col in amenity_cols:
        agg_map[col] = _combine_amenity_values
    out = df.groupby(STOP_ID_FIELD, as_index=False).agg(agg_map)
    out[amenity_cols] = out[amenity_cols].fillna(UNKNOWN_VALUE)
    return out


def _compute_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add FLAG_* / REVIEW_* columns and NEEDS_IMPROVEMENT / NEEDS_REVIEW summaries.

    A stop at or above a threshold is flagged when the amenity is recorded as
    absent ('N') and marked for review when it is unknown (neither 'Y' nor 'N').
    """
    review_cols: List[str] = []
    for name, cfg in AMENITIES.items():
        col = cfg["field"]
        meets_thresh = df[RIDERSHIP_FIELD] >= cfg["thresh"]
        df[f"FLAG_{name.upper()}"] = meets_thresh & (df[col] == "N")
        df[f"REVIEW_{name.upper()}"] = meets_thresh & ~df[col].isin(["Y", "N"])
        review_cols.append(f"REVIEW_{name.upper()}")
    flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
    df["NEEDS_IMPROVEMENT"] = df[flag_cols].any(axis=1)
    df["NEEDS_REVIEW"] = df[review_cols].any(axis=1)
    return df, flag_cols


def _write_workbook(raw_df: pd.DataFrame, processed_df: pd.DataFrame, out_path: Path) -> None:
    """Write multi-sheet Excel: Raw Data, All Flags, plus one per amenity."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        processed_df.to_excel(writer, sheet_name="All Flags", index=False)
        for name in AMENITIES:
            flag_col = f"FLAG_{name.upper()}"
            processed_df[processed_df[flag_col]].to_excel(writer, sheet_name=name, index=False)
        processed_df[processed_df["NEEDS_REVIEW"]].to_excel(
            writer, sheet_name="Needs Review", index=False
        )


def _load_amenity_data(path: Path, sheet: int | str) -> pd.DataFrame:
    """Read and sanitise the separate amenities workbook."""
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    # Normalise column names and apply known aliases
    df = _apply_amenity_aliases(df)
    # Standardise any amenity columns present (blanks stay missing)
    for cfg in AMENITIES.values():
        col = cfg["field"]
        if col in df.columns:
            df[col] = _standardise_yn(df[col])
    return df


def _log_match_rates(rider_ids: Set[str], amen_ids: Set[str]) -> None:
    """Log how many unique stop IDs in each workbook have no match in the other."""
    for source, ids, other_ids, consequence in (
        (
            "Ridership",
            rider_ids,
            amen_ids,
            "their amenities are UNKNOWN unless the ridership workbook has them",
        ),
        ("Amenity", amen_ids, rider_ids, "they are not in the output"),
    ):
        unmatched = len(ids - other_ids)
        share = 100 * unmatched / len(ids) if ids else 0.0
        logging.info(
            "%s stops without a match in the other workbook: %d of %d (%.1f%%); %s.",
            source,
            unmatched,
            len(ids),
            share,
            consequence,
        )


def _merge_ridership_and_amenities(rider_df: pd.DataFrame, amen_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join amenity info onto ridership on STOP_ID_FIELD ↔ AMENITY_JOIN_FIELD."""
    if AMENITY_JOIN_FIELD not in amen_df.columns:
        raise ValueError(f"Column '{AMENITY_JOIN_FIELD}' not found in amenity workbook.")
    rider_df = _drop_missing_ids(rider_df, STOP_ID_FIELD, "ridership")
    amen_df = _drop_missing_ids(amen_df, AMENITY_JOIN_FIELD, "amenity")
    _log_match_rates(set(rider_df[STOP_ID_FIELD]), set(amen_df[AMENITY_JOIN_FIELD]))

    fields = [cfg["field"] for cfg in AMENITIES.values() if cfg["field"] in amen_df.columns]
    amen_subset = amen_df[[AMENITY_JOIN_FIELD, *fields]].rename(
        columns={AMENITY_JOIN_FIELD: STOP_ID_FIELD}
    )

    # One inventory row per stop, or the join would duplicate ridership rows
    dup_ct = int(amen_subset[STOP_ID_FIELD].duplicated().sum())
    if dup_ct:
        logging.warning(
            "Amenity workbook has %d duplicate '%s' row(s); combining them (Y if any record is Y).",
            dup_ct,
            AMENITY_JOIN_FIELD,
        )
        amen_subset = amen_subset.groupby(STOP_ID_FIELD, as_index=False).agg(
            dict.fromkeys(fields, _combine_amenity_values)
        )

    merged = rider_df.merge(
        amen_subset,
        how="left",
        on=STOP_ID_FIELD,
        suffixes=("", "_amen"),
        validate="many_to_one",
    )

    # Coalesce: prefer explicit amenity file, fall back to ridership data
    for col in fields:
        alt = f"{col}_amen"
        if alt in merged.columns:
            merged[col] = merged[alt].fillna(merged[col])
            merged = merged.drop(columns=[alt])
    return merged


def _write_txt_log(processed_df: pd.DataFrame, flag_cols: List[str], out_path: Path) -> None:
    """Write a plain-text summary of flagged and review stops and counts."""
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Run date: {pd.Timestamp.now():%Y-%m-%d %H:%M}\n\n")
        f.write("Stops needing improvement by category\n")
        f.write("──────────────────────────────────────\n")
        for name in AMENITIES:
            col = f"FLAG_{name.upper()}"
            f.write(f"{name:10s}: {processed_df[col].sum():>6}\n")
        f.write(f"\nTotal flagged stops: {processed_df['NEEDS_IMPROVEMENT'].sum()}\n\n")

        f.write("Stops needing review (amenity UNKNOWN) by category\n")
        f.write("───────────────────────────────────────────────────\n")
        review_cols = [f"REVIEW_{name.upper()}" for name in AMENITIES]
        for name, col in zip(AMENITIES, review_cols):
            f.write(f"{name:10s}: {processed_df[col].sum():>6}\n")
        f.write(f"\nTotal stops needing review: {processed_df['NEEDS_REVIEW'].sum()}\n\n")

        f.write("Detailed list (one row per stop)\n")
        f.write("─────────────────────────────────\n")
        base_cols = [STOP_ID_FIELD, RIDERSHIP_FIELD] + [cfg["field"] for cfg in AMENITIES.values()]
        cols = base_cols + flag_cols
        f.write(processed_df[processed_df["NEEDS_IMPROVEMENT"]][cols].to_string(index=False))

        f.write("\n\nStops needing review (one row per stop)\n")
        f.write("────────────────────────────────────────\n")
        review_rows = processed_df[processed_df["NEEDS_REVIEW"]]
        f.write(review_rows[base_cols + review_cols].to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================


_PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "your\\file\\path",
    "your\\folder\\path",
    "path\\to\\your",
    "your/file/path",
    "your/folder/path",
    "path/to/your",
)


def _is_placeholder_path(p: Path) -> bool:
    """Return True if *p* still points at a default placeholder location."""
    s = str(p).lower()
    return any(marker in s for marker in _PLACEHOLDER_MARKERS)


def main() -> int:
    """Run the ETL pipeline and produce both Excel and text outputs.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    placeholders = {
        "RIDERSHIP_XLSX": RIDERSHIP_XLSX,
        "AMENITIES_XLSX": AMENITIES_XLSX,
        "OUTPUT_FOLDER": OUTPUT_FOLDER,
    }
    unset = [name for name, p in placeholders.items() if p is not None and _is_placeholder_path(p)]
    if unset:
        logging.warning(
            "Default placeholder filepaths detected for: %s. "
            "Update the CONFIGURATION section of this script with real paths "
            "before running. Exiting without processing.",
            ", ".join(unset),
        )
        return 2

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # 1–2. LOAD SOURCE FILES (keep an untouched copy for the Raw Data sheet)
    df_source = _load_ridership_data(RIDERSHIP_XLSX, RIDERSHIP_SHEET)
    df_rider = _apply_amenity_aliases(df_source.copy())
    for field in (STOP_ID_FIELD, RIDERSHIP_FIELD):
        if field not in df_rider.columns:
            raise ValueError(f"Column '{field}' not found in workbook.")
    df_rider = _drop_missing_ids(df_rider, STOP_ID_FIELD, "ridership")

    # 3. MERGE (optional amenity workbook)
    if AMENITIES_XLSX is None:
        df_merged = df_rider
    else:
        df_amen = _load_amenity_data(AMENITIES_XLSX, AMENITIES_SHEET)
        df_merged = _merge_ridership_and_amenities(df_rider, df_amen)

    # 4–5. CLEAN & AGGREGATE
    df_merged = _prepare_amenity_columns(df_merged)
    df_merged = _convert_ridership(df_merged)

    need_agg = _needs_aggregation(df_merged)
    df_processed = _aggregate_by_stop(df_merged) if need_agg else df_merged.copy()

    # 6. COMPUTE FLAGS
    df_processed, flag_cols = _compute_flags(df_processed)

    # 7. WRITE OUTPUTS
    out_xlsx = OUTPUT_FOLDER / "stops_needing_improvement.xlsx"
    _write_workbook(df_source, df_processed, out_xlsx)
    _write_txt_log(df_processed, flag_cols, TXT_LOG_PATH)

    # 8. CONSOLE SUMMARY
    logging.info("\n✓ Workbook created: %s", out_xlsx)
    logging.info("✓ Text log created: %s", TXT_LOG_PATH)
    if need_agg:
        dup_ct = df_merged.shape[0] - df_processed.shape[0]
        logging.info("  (Aggregated %d duplicate STOP_ID rows.)", dup_ct)

    logging.info("flag_stop_upgrades.py completed successfully.")
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
