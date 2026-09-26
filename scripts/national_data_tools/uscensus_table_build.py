"""Join Census block- and tract-level data into a unified DataFrame.

This module discovers, reads, and merges demographic and employment datasets
from the U.S. Census and LODES, based on GEO_ID alignment. Output includes
population, household counts, job totals, income brackets, ethnicity, language
proficiency, vehicle availability, age group, and commuting (journey-to-work)
statistics.

Supports input as CSV, GZ, or ZIP files (containing '-Data.csv'), and can filter
by county FIPS codes. Output may be saved as a flat CSV. Tract counts are split
across each tract's blocks by block population (P1) or households (H9), the same
count allocation uscensus_tiger_join_gpd uses, so every count column can be
summed over blocks; tract percentages ride along unchanged.

Outputs:
    - joined_blocks.csv (OUTPUT_CSV_NAME, written into OUTPUT_DIR): the joined
      block + tract attribute table, one row per block. Set OUTPUT_DIR to None
      to skip writing and use build_joined_table in memory instead.

Helpful links:
    https://data.census.gov/table
    https://lehd.ces.census.gov/data/

Typical usage:
    Update ROOT_DATA_DIR and OUTPUT_DIR in the CONFIGURATION section and run
    from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Hashable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

#: Folder that holds every Census download (plain CSV, *.csv.gz*, or ZIPs).
#: Sub-directories are searched automatically.
ROOT_DATA_DIR: str | Path = r"Path\To\Your\Census_Table_Data_Files"  # <<< EDIT ME

#: Optional output folder for the joined CSV (set OUTPUT_DIR to None to skip writing).
OUTPUT_DIR: str | Path | None = r"Path\To\Your\Output_Folder"  # <<< EDIT ME
OUTPUT_CSV_NAME: str = "joined_blocks.csv"
CSV_OUTPUT_PATH: str | None = (
    None if OUTPUT_DIR is None else str(Path(OUTPUT_DIR) / OUTPUT_CSV_NAME)
)

#: Optional county FIPS filter (5-digit codes, e.g. ["11001", "51059"])
COUNTY_FIPS_FILTER: list[str] = [
    "11001",
    "24031",
    "24033",
    "51683",
    "51685",
    "51059",
    "51013",
    "51510",
    "51600",
    "51610",
    "51107",
    "51153",
]

# -----------------------------------------------------------------------------
# Signatures that map a file name to a *topic* variable.
# ALL tokens listed for a topic must appear in the file name (case-insensitive), each
# as a whole code: "B19001" matches "ACSDT5Y2024.B19001-Data.csv" but not the race
# iteration "B19001A", and "P1" does not match "P12". Pin a vintage by adding its
# product code, e.g. ("ACSDT5Y2024", "B19001"). Files that disagree on the same
# geography (two vintages, say) abort the run rather than one being silently kept.
TOPIC_SIGNATURES: dict[str, Sequence[str] | str] = {
    "POP_FILES": ("P1",),
    "HH_FILES": ("H9",),
    "JOBS_FILES": ("_S000_JT00_",),  # LODES WAC
    "INCOME_FILES": ("B19001",),
    "ETHNICITY_FILES": ("P9",),
    "LANGUAGE_FILES": ("C16001",),
    "VEHICLE_FILES": ("B08201",),
    "AGE_FILES": ("B01001",),
    "COMMUTE_FILES": ("S0801",),
}

# Tract tables (income, ethnicity, language, vehicles, age, commuting) are split across
# their blocks rather than repeated on each one: household-universe counts listed here
# by block households (H9), everything else by block population (P1).
HOUSEHOLD_UNIVERSE_COUNTS: frozenset[str] = frozenset(
    {
        # B19001 household income
        "sub_10k",
        "10k_15k",
        "15k_20k",
        "20k_25k",
        "25k_30k",
        "30k_35k",
        "35k_40k",
        "40k_45k",
        "45k_50k",
        "50k_60k",
        "low_income",
        # B08201 household size by vehicles available
        "all_hhs",
        "veh_0_all_hh",
        "veh_1_all_hh",
        "veh_0_hh_1",
        "veh_1_hh_1",
        "veh_0_hh_2",
        "veh_1_hh_2",
        "veh_0_hh_3",
        "veh_1_hh_3",
        "veh_2_hh_3",
        "veh_0_hh_4p",
        "veh_1_hh_4p",
        "veh_2_hh_4p",
        "all_lo_veh_hh",
        "all_lo_veh_hh_mod",
    }
)

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# Sentinel values — detect un-edited placeholder paths
_DEFAULT_ROOT_DATA_DIR: str = r"Path\To\Your\Census_Table_Data_Files"
_DEFAULT_OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# =============================================================================
# FUNCTIONS
# =============================================================================


def _token_match(name: str, tokens: Sequence[str] | str) -> bool:
    """Return True if *all* tokens occur in *name* as whole codes (case-insensitive).

    A token that starts or ends with a letter or digit must sit against a
    non-alphanumeric boundary there, so a table code matches only itself:
    ``B19001`` does not match the race-iteration table ``B19001A``, and ``P1`` does
    not match ``P12`` or ``DP1``. Tokens that begin and end with a separator (the
    LODES ``_S000_JT00_``) still match anywhere.
    """
    if isinstance(tokens, str):
        tokens = (tokens,)
    low = name.lower()
    for tok in tokens:
        tok_low = tok.lower()
        pattern = re.escape(tok_low)
        if tok_low[:1].isalnum():
            pattern = r"(?<![a-z0-9])" + pattern
        if tok_low[-1:].isalnum():
            pattern += r"(?![a-z0-9])"
        if re.search(pattern, low) is None:
            return False
    return True


def _zip_data_members(zip_path: str | Path) -> dict[str, list[tuple[str, int]]]:
    """Map each '*-Data.csv' member's base name to its (member name, size) entries.

    Reads only the ZIP central directory (no extraction). Returns an empty map
    when the archive cannot be read, so a corrupt ZIP never suppresses a loose CSV.
    """
    members: dict[str, list[tuple[str, int]]] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith("-data.csv"):
                    base = Path(info.filename).name.lower()
                    members.setdefault(base, []).append((info.filename, info.file_size))
    except (zipfile.BadZipFile, OSError):
        return {}
    return members


def _sha256(stream: Any) -> str:
    """Return the SHA-256 hex digest of a binary stream, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _same_as_zip_member(loose: Path, zip_path: str, member: str) -> bool:
    """Return True if *loose* has exactly the bytes of *member* inside *zip_path*."""
    try:
        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as packed, loose.open("rb") as fh:
            return _sha256(packed) == _sha256(fh)
    except (zipfile.BadZipFile, OSError):
        return False


def _dedupe_extracted_zip_members(paths: Sequence[str]) -> list[str]:
    """Drop loose '*-Data.csv' files that duplicate a '-Data.csv' member of a ZIP.

    ``_read_csv_any`` reads a Census ZIP's '-Data.csv' member directly, so when
    the same archive has also been unzipped in place — which the feature-prep
    orchestrator does by default, and a human may do manually — the scanned root
    holds both the ``*.zip`` and the extracted ``*-Data.csv``. Bucketing both
    would concatenate the identical table twice (duplicate GEO_ID rows). A loose
    CSV is treated as redundant only when its base name matches a member of a ZIP
    in the same bucket AND its bytes are identical to that member (size first,
    then a content hash), so the ZIP is kept and the extracted copy dropped.
    Anything else — another geography, a re-saved or edited copy — is kept, and a
    real conflict is then caught by the duplicate check in ``_load_and_concat``.
    """
    members: dict[str, list[tuple[str, str, int]]] = {}
    for p in paths:
        if p.lower().endswith(".zip"):
            for base, entries in _zip_data_members(p).items():
                members.setdefault(base, []).extend((p, name, size) for name, size in entries)
    if not members:
        return list(paths)

    kept: list[str] = []
    for p in paths:
        base = Path(p).name.lower()
        if base.endswith("-data.csv") and base in members:
            try:
                size = Path(p).stat().st_size
            except OSError:
                size = -1
            if any(
                member_size == size and _same_as_zip_member(Path(p), zip_path, member)
                for zip_path, member, member_size in members[base]
            ):
                logging.info(
                    "Skipping '%s'; byte-identical to a ZIP member already in this bucket.",
                    p,
                )
                continue
        kept.append(p)
    return kept


def discover_census_files(
    root_dir: str | Path,
    signatures: Mapping[str, Sequence[str] | str] = TOPIC_SIGNATURES,
) -> dict[str, list[str]]:
    """Recursively locate Census “data” files and bucket them by topic.

    * Accepts plain **CSV**, **CSV.GZ**, or **ZIP** archives.
    * ZIPs are returned as the ZIP path itself – content is handled later.
    * A loose ``*-Data.csv`` that merely duplicates a ZIP member already in the
      same bucket (e.g. an in-place unzip) is dropped, so the table is read once.
    * File order is sorted for determinism.

    Returns:
    -------
    dict[str, list[str]]
        Keys mirror *signatures* and align with downstream variable names.
    """
    buckets: dict[str, list[str]] = {k: [] for k in signatures}
    root = Path(root_dir).expanduser().resolve()

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not path.name.lower().endswith(("-data.csv", ".csv.gz", ".zip")):
            continue

        for var, sig in signatures.items():
            if _token_match(path.name, sig):
                buckets[var].append(str(path))
                break

    for var, lst in buckets.items():
        buckets[var] = sorted(_dedupe_extracted_zip_members(lst))
    return buckets


def _read_csv_any(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Read a CSV/CSV.GZ directly *or* the first “-Data.csv” member in a ZIP."""
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".zip":
        with zipfile.ZipFile(p) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith("-data.csv")]
            if not members:
                raise FileNotFoundError(f"No '*-Data.csv' inside {p}")
            with zf.open(members[0]) as fh, io.TextIOWrapper(fh, encoding="utf-8") as txt:
                return pd.read_csv(txt, **read_kwargs)

    return pd.read_csv(p, **read_kwargs)


# -----------------------------------------------------------------------------
# DATA-PROCESSING CONSTANTS & REGEXES
# -----------------------------------------------------------------------------

GEO_ID_COL = "GEO_ID"
_UNFRIENDLY_COL_RE = re.compile(r"^[A-Z]{2,}\d{3,}.*")
#: Temporary per-row record of the input file, used to name files in duplicate errors.
_SOURCE_COL: Final[str] = "_source_file"


def _fill_numeric_only(df: pd.DataFrame, value: int | float = 0) -> pd.DataFrame:
    """Replace *only* numeric NaNs with *value*; leave object columns untouched."""
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(value)
    return df


def _clean_name_cols(df: pd.DataFrame) -> None:
    """Sanitise NAME‑like columns in place (remove CR/LF/TAB)."""
    for col in df.filter(regex=r"^NAME").columns:
        df[col] = (
            df[col]  # Series
            .astype(str)  # ensure string dtype
            .str.replace(r"[\r\n\t]+", " ", regex=True)  # collapse control chars
            .str.strip()  # trim leading/trailing spaces
        )


def _dedupe_topic_rows(df: pd.DataFrame, key: Hashable) -> pd.DataFrame:
    """Collapse rows that repeat *key* with identical data; reject conflicting repeats.

    A topic bucket can gather more than one input file for the same geography — a
    copy of the same download, or a second ACS vintage of the table. Concatenated,
    those files repeat every ``GEO_ID``, and because the later GEO_ID merges and the
    one-to-many block<->tract join both fan out on the key, each repeat becomes a
    *multiplicative* row explosion.

    Rows that repeat a key with the same data (``NAME`` labels aside) are collapsed
    to one. A key that repeats with *different* data is an ambiguous input — keeping
    either row would silently pick a vintage — so it raises, naming the files. The
    temporary ``_SOURCE_COL`` added by ``_load_and_concat`` is dropped on return.

    Raises:
        ValueError: If any key repeats with conflicting values.
    """
    if key not in df.columns:
        return df.drop(columns=_SOURCE_COL, errors="ignore")
    data_cols = [c for c in df.columns if c != _SOURCE_COL and not str(c).startswith("NAME")]
    unique = df.drop_duplicates(subset=data_cols)
    conflicts = unique[unique.duplicated(subset=[key], keep=False)]
    if not conflicts.empty:
        keys = conflicts[key].astype(str).unique().tolist()
        files = (
            sorted(conflicts[_SOURCE_COL].astype(str).unique())
            if _SOURCE_COL in conflicts.columns
            else []
        )
        raise ValueError(
            f"{len(keys)} geography key(s) in '{key}' carry conflicting values across input "
            f"rows (e.g. {', '.join(keys[:3])}); files involved: {files}. Keep one table and "
            "vintage per topic: remove the extra file, or pin the vintage in TOPIC_SIGNATURES "
            "(e.g. ('ACSDT5Y2024', 'B19001'))."
        )
    dropped = len(df) - len(unique)
    if dropped:
        logging.info(
            "Collapsed %d duplicate row(s) on '%s' carrying identical data (same geography "
            "supplied more than once).",
            dropped,
            key,
        )
    return unique.drop(columns=_SOURCE_COL, errors="ignore").reset_index(drop=True)


def _load_and_concat(
    files: Sequence[str],
    *,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    dtype: Mapping[Hashable, str | np.dtype[Any]] | None = None,
    usecols: Sequence[Hashable] | None = None,
    rename: Mapping[str, str] | None = None,
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] | None = None,
    dedupe_key: Hashable | None = GEO_ID_COL,
) -> pd.DataFrame:
    """Read multiple Census CSV / CSV‑GZ / ZIP files and concatenate the results.

    Embedded control characters in *NAME* columns are stripped immediately to
    guarantee that every logical record remains on a single physical line when
    the final DataFrame is exported.

    Parameters
    ----------
    files :
        Paths to source files.
    skiprows, dtype, usecols, rename, compression :
        Passed straight through to :func:`pandas.read_csv`; see pandas docs.
    dedupe_key :
        Key checked for repeats (default ``GEO_ID``): rows repeating it with
        identical data are collapsed, conflicting repeats raise ``ValueError``
        (see ``_dedupe_topic_rows``). ``None`` disables the check.

    Returns:
    -------
    pd.DataFrame
        Concatenated frame (empty if *files* is empty).

    Notes:
    -----
    * ZIP archives are handled transparently via ``_read_csv_any``.
    * Column renaming occurs **before** we prune columns via *usecols*
      (unless *usecols* is explicitly supplied).
    """
    frames: list[pd.DataFrame] = []

    for path in files:
        # Let pandas decide the right decompression unless the caller
        # explicitly overrides it.
        read_kwargs: dict[str, Any] = {}
        if compression is not None:
            read_kwargs["compression"] = compression
        if skiprows is not None:
            read_kwargs["skiprows"] = skiprows
        if dtype is not None:
            read_kwargs["dtype"] = dtype
        if usecols is not None:
            read_kwargs["usecols"] = usecols

        # --- read, then sanitise ---
        df = _read_csv_any(path, **read_kwargs)
        _clean_name_cols(df)

        # --- optional column renaming & pruning ---
        if rename:
            df = df.rename(columns=rename)
            if usecols is None:
                keep = {GEO_ID_COL, "NAME", *rename.values()}
                df = df.loc[:, df.columns.intersection(keep)]

        df[_SOURCE_COL] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if dedupe_key is not None:
        combined = _dedupe_topic_rows(combined, dedupe_key)
    return combined.drop(columns=_SOURCE_COL, errors="ignore")


def _merge_on_geo_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge two frames on GEO_ID (one row per key each), dropping duplicate columns."""
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()

    dup = (set(left.columns) & set(right.columns)) - {GEO_ID_COL}
    return left.merge(right.drop(columns=dup), on=GEO_ID_COL, how="outer", validate="1:1")


def _drop_unfriendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any column that still looks like a raw Census code."""
    to_drop = [c for c in df.columns if _UNFRIENDLY_COL_RE.match(c)]
    return df.drop(columns=to_drop, errors="ignore")


# -----------------------------------------------------------------------------
# BLOCK-LEVEL BUILD
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class _BlockInputs:
    pop_files: list[str]
    hh_files: list[str]
    jobs_files: list[str]


def _build_block_df(inp: _BlockInputs) -> pd.DataFrame:
    """Return a block-level DataFrame with population, households, and jobs."""
    pop = _load_and_concat(
        inp.pop_files,
        skiprows=[1],
        rename={"P1_001N": "total_pop"},
        usecols=[GEO_ID_COL, "NAME", "P1_001N"],
    )
    hh = _load_and_concat(
        inp.hh_files,
        skiprows=[1],
        rename={"H9_001N": "total_hh"},
        usecols=[GEO_ID_COL, "H9_001N"],
        dtype={"H9_001N": "Int64"},
    )
    jobs = _load_and_concat(
        inp.jobs_files,
        rename={
            "C000": "tot_empl",
            "CE01": "low_wage",
            "CE02": "mid_wage",
            "CE03": "high_wage",
        },
        usecols=["w_geocode", "C000", "CE01", "CE02", "CE03"],
        # LODES is keyed on the block geocode, not GEO_ID; check repeats there.
        dedupe_key="w_geocode",
    )
    if not jobs.empty:
        jobs[GEO_ID_COL] = "1000000US" + jobs["w_geocode"].astype(str)
        jobs = jobs.drop(columns="w_geocode")

    df = _merge_on_geo_id(pop, hh)
    df = _merge_on_geo_id(df, jobs)
    df["tract_id_synth"] = df[GEO_ID_COL].str[9:20]
    df["block_id_synth"] = df[GEO_ID_COL].str[9:24]

    _fill_numeric_only(df)
    return df


# -----------------------------------------------------------------------------
# TRACT-LEVEL BUILD & DERIVATIONS
# -----------------------------------------------------------------------------
def _derive_income(df: pd.DataFrame) -> pd.DataFrame:
    bands = [
        "sub_10k",
        "10k_15k",
        "15k_20k",
        "20k_25k",
        "25k_30k",
        "30k_35k",
        "35k_40k",
        "40k_45k",
        "45k_50k",
        "50k_60k",
    ]
    df["low_income"] = df[bands].sum(axis=1)
    df["perc_low_income"] = (df["low_income"] / df["total_hh"]).fillna(0)
    df = df.drop(columns="total_hh")
    return df


def _derive_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    # Minority population, Title VI/EJ convention: total minus Not-Hispanic White
    # alone. This necessarily includes Hispanic/Latino residents of any race (a
    # separate P9 branch from the race-alone rows, so no double-count) — summing
    # only the non-white race-alone categories would silently drop them.
    df["minority"] = df["total_pop"] - df["white"]
    df["perc_minority"] = (df["minority"] / df["total_pop"]).fillna(0)
    df = df.drop(columns="total_pop")
    return df


def _derive_language(df: pd.DataFrame) -> pd.DataFrame:
    lep_cols = [c for c in df.columns if c.endswith("_engnwell")]
    df[lep_cols] = df[lep_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["all_nwell"] = df[lep_cols].sum(axis=1)
    df["perc_lep"] = (df["all_nwell"] / df["total_lang_pop"]).fillna(0).round(3)
    return df


def _derive_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    # Low-vehicle households from Census table B08201 (Household Size by
    # Vehicles Available). Two definitions are emitted side by side so each
    # consumer can choose the one it wants; neither is suppressed.
    #
    # Standard ("all_lo_veh_hh" / "perc_lo_veh"): every household with 0 or 1
    # vehicle. This is the common transit-equity convention (e.g. FTA Title VI
    # analyses) -- transparent and comparable across agencies -- but it flags a
    # 1-person/1-vehicle household as low-vehicle even though it is fully mobile.
    #
    # Modified ("all_lo_veh_hh_mod" / "perc_lo_veh_mod"): the standard count
    # minus 1-person households that already own a vehicle. The implied model is
    # "a household needs 1 vehicle when it has 1 person and 2 vehicles
    # otherwise" -- i.e. assume at most ~2 drivers (2 adults) per household and
    # treat any additional members as non-drivers (children). This keeps a 3- or
    # 4-person household with 2 vehicles from being counted as constrained.
    # Caveat: it undercounts deficiency in multi-adult households
    # (multigenerational, adult children, shared housing), which skew toward
    # transit-dependent populations. Measuring vehicles against *workers*
    # (ACS B08203 / B08141) would capture the commute constraint directly.
    df["all_lo_veh_hh"] = df[["veh_0_all_hh", "veh_1_all_hh"]].sum(axis=1)
    df["all_lo_veh_hh_mod"] = df["all_lo_veh_hh"] - df["veh_1_hh_1"]
    df["perc_lo_veh"] = (df["all_lo_veh_hh"] / df["all_hhs"]).fillna(0).round(3)
    df["perc_0_veh"] = (df["veh_0_all_hh"] / df["all_hhs"]).fillna(0).round(3)
    df["perc_1_veh"] = (df["veh_1_all_hh"] / df["all_hhs"]).fillna(0).round(3)
    df["perc_veh_1_hh_1"] = (df["veh_1_hh_1"] / df["all_hhs"]).fillna(0).round(3)
    df["perc_lo_veh_mod"] = (df["all_lo_veh_hh_mod"] / df["all_hhs"]).fillna(0).round(3)
    return df


def _derive_age(df: pd.DataFrame) -> pd.DataFrame:
    youth = [
        "m_15_17",
        "f_15_17",
        "m_18_19",
        "f_18_19",
        "m_20",
        "f_20",
        "m_21",
        "f_21",
    ]
    elderly = [
        "m_65_66",
        "f_65_66",
        "m_67_69",
        "f_67_69",
        "m_70_74",
        "f_70_74",
        "m_75_79",
        "f_75_79",
        "m_80_84",
        "f_80_84",
        "m_a_85",
        "f_a_85",
    ]
    df["all_youth"] = df[[c for c in youth if c in df]].sum(axis=1)
    df["all_elderly"] = df[[c for c in elderly if c in df]].sum(axis=1)
    if "total_pop" in df.columns:
        df["perc_youth"] = (df["all_youth"] / df["total_pop"]).fillna(0).round(3)
        df["perc_elderly"] = (df["all_elderly"] / df["total_pop"]).fillna(0).round(3)
        df = df.drop(columns="total_pop")
    return df


def _derive_commute(df: pd.DataFrame) -> pd.DataFrame:
    """Derive commuting measures from ACS S0801 (Commuting Characteristics).

    Unlike the count-based detailed tables, S0801's means-of-transportation rows
    arrive as *percentages* of workers 16+ (and travel time as a mean in minutes).
    We keep those percentages for readability, and also materialize the matching
    additive worker *counts* (``workers * pct / 100``) plus person-minutes. Only
    those counts are legal to area-weight in the tract->block disaggregation and the
    service-area clip downstream — percentages and means are never additive.

    The mean travel time (S0801_C01_046E) is over workers who did NOT work from
    home, so person-minutes are ``(workers - wfh) * mean``. Their denominator,
    ``commute_timed``, holds those same commuters, but only where the tract
    publishes both the mean and the work-from-home share: a suppressed value drops
    the tract from numerator and denominator alike instead of counting as zero
    minutes. A catchment mean travel time is recoverable later as
    ``sum(commute_person_min) / sum(commute_timed)``.
    """
    perc_cols = ["perc_drove_alone", "perc_carpool", "perc_transit", "perc_wfh"]
    for col in ["commute_workers", "mean_travel_time", *perc_cols]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    workers = df["commute_workers"].fillna(0)
    df["commute_transit"] = workers * df["perc_transit"] / 100.0
    df["commute_drove"] = workers * df["perc_drove_alone"] / 100.0
    df["commute_carpool"] = workers * df["perc_carpool"] / 100.0
    df["commute_wfh"] = workers * df["perc_wfh"] / 100.0
    commuters = (workers - df["commute_wfh"]).clip(lower=0)
    timed = commuters.notna() & df["mean_travel_time"].notna()
    df["commute_timed"] = commuters.where(timed, 0.0)
    df["commute_person_min"] = (commuters * df["mean_travel_time"]).where(timed, 0.0)
    return df


@dataclass(slots=True)
class _TractInputs:
    income_files: list[str]
    ethnicity_files: list[str]
    language_files: list[str]
    vehicle_files: list[str]
    age_files: list[str]
    commute_files: list[str]


def _build_tract_df(inp: _TractInputs) -> pd.DataFrame:
    """Return a tract-level DataFrame of optional socio-economic measures."""
    dfs: list[pd.DataFrame] = []

    if inp.income_files:
        income = _load_and_concat(
            inp.income_files,
            skiprows=[1],
            rename={
                "B19001_001E": "total_hh",
                "B19001_002E": "sub_10k",
                "B19001_003E": "10k_15k",
                "B19001_004E": "15k_20k",
                "B19001_005E": "20k_25k",
                "B19001_006E": "25k_30k",
                "B19001_007E": "30k_35k",
                "B19001_008E": "35k_40k",
                "B19001_009E": "40k_45k",
                "B19001_010E": "45k_50k",
                "B19001_011E": "50k_60k",
            },
        )
        dfs.append(_derive_income(income))

    if inp.ethnicity_files:
        ethnicity = _load_and_concat(
            inp.ethnicity_files,
            skiprows=[1],
            rename={
                "P9_001N": "total_pop",
                "P9_002N": "all_hisp",
                "P9_005N": "white",
                "P9_006N": "black",
                "P9_007N": "native",
                "P9_008N": "asian",
                "P9_009N": "pac_isl",
                "P9_010N": "other",
                "P9_011N": "multi",
            },
        )
        dfs.append(_derive_ethnicity(ethnicity))

    if inp.language_files:
        language = _load_and_concat(
            inp.language_files,
            skiprows=[1],
            rename={
                "C16001_001E": "total_lang_pop",
                "C16001_005E": "spanish_engnwell",
                "C16001_008E": "frenchetc_engnwell",
                "C16001_011E": "germanetc_engnwell",
                "C16001_014E": "slavicetc_engnwell",
                "C16001_017E": "indoeuroetc_engnwell",
                "C16001_020E": "korean_engnwell",
                "C16001_023E": "chineseetc_engnwell",
                "C16001_026E": "vietnamese_engnwell",
                "C16001_029E": "tagalog_engnwell",
                "C16001_032E": "asiapacetc_engnwell",
                "C16001_035E": "arabic_engnwell",
                "C16001_038E": "otheretc_engnwell",
            },
        )
        dfs.append(_derive_language(language))

    if inp.vehicle_files:
        vehicle = _load_and_concat(
            inp.vehicle_files,
            skiprows=[1],
            rename={
                "B08201_001E": "all_hhs",
                "B08201_002E": "veh_0_all_hh",
                "B08201_003E": "veh_1_all_hh",
                "B08201_008E": "veh_0_hh_1",
                "B08201_009E": "veh_1_hh_1",
                "B08201_014E": "veh_0_hh_2",
                "B08201_015E": "veh_1_hh_2",
                "B08201_020E": "veh_0_hh_3",
                "B08201_021E": "veh_1_hh_3",
                "B08201_022E": "veh_2_hh_3",
                "B08201_026E": "veh_0_hh_4p",
                "B08201_027E": "veh_1_hh_4p",
                "B08201_028E": "veh_2_hh_4p",
            },
        )
        dfs.append(_derive_vehicle(vehicle))

    if inp.age_files:
        age = _load_and_concat(
            inp.age_files,
            skiprows=[1],
            rename={
                "B01001_001E": "total_pop",
                "B01001_006E": "m_15_17",
                "B01001_007E": "m_18_19",
                "B01001_008E": "m_20",
                "B01001_009E": "m_21",
                "B01001_020E": "m_65_66",
                "B01001_021E": "m_67_69",
                "B01001_022E": "m_70_74",
                "B01001_023E": "m_75_79",
                "B01001_024E": "m_80_84",
                "B01001_025E": "m_a_85",
                "B01001_030E": "f_15_17",
                "B01001_031E": "f_18_19",
                "B01001_032E": "f_20",
                "B01001_033E": "f_21",
                "B01001_044E": "f_65_66",
                "B01001_045E": "f_67_69",
                "B01001_046E": "f_70_74",
                "B01001_047E": "f_75_79",
                "B01001_048E": "f_80_84",
                "B01001_049E": "f_a_85",
            },
        )
        dfs.append(_derive_age(age))

    if inp.commute_files:
        commute = _load_and_concat(
            inp.commute_files,
            skiprows=[1],
            rename={
                "S0801_C01_001E": "commute_workers",
                "S0801_C01_003E": "perc_drove_alone",
                "S0801_C01_004E": "perc_carpool",
                "S0801_C01_009E": "perc_transit",
                "S0801_C01_013E": "perc_wfh",
                "S0801_C01_046E": "mean_travel_time",
            },
        )
        dfs.append(_derive_commute(commute))

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for optional in dfs[1:]:
        merged = _merge_on_geo_id(merged, optional)

    _fill_numeric_only(merged)
    merged["tract_id_clean"] = merged[GEO_ID_COL].str[9:]
    return merged


# -----------------------------------------------------------------------------
# FIPS HELPERS
# -----------------------------------------------------------------------------
def _ensure_fips_column(
    df: pd.DataFrame,
    *,
    dst: str = "FIPS",
    geo_candidates: tuple[str, ...] = ("GEO_ID", "GEO_ID_blk", "GEO_ID_trt"),
    start: int = 9,
    end: int = 14,
) -> None:
    """Create a 5-digit county FIPS column *in-place* from the first GEO_ID."""
    if dst in df.columns:
        return
    source = next((c for c in geo_candidates if c in df.columns), None)
    if source is None:
        raise KeyError(f"No GEO_ID column found among {geo_candidates}")
    df[dst] = df[source].astype(str).str[start:end]


def _apply_fips_filter(
    df: pd.DataFrame,
    *,
    fips: Iterable[str] | None = None,
    dst_col: str = "FIPS",
) -> pd.DataFrame:
    """Return a copy filtered to *fips* (or unchanged if *fips* is empty/None)."""
    if not fips:
        return df
    _ensure_fips_column(df, dst=dst_col)
    wanted = {str(code).zfill(5) for code in fips}
    return df[df[dst_col].isin(wanted)].copy()


def _tract_count_columns(tract_df: pd.DataFrame) -> list[str]:
    """Return the additive count columns of a tract table (numeric, not ``perc_*`` or ids)."""
    ids = {GEO_ID_COL, "tract_id_clean", "FIPS"}
    return [
        str(c)
        for c in tract_df.columns
        if c not in ids
        and not str(c).startswith("perc_")
        and pd.api.types.is_numeric_dtype(tract_df[c])
    ]


def allocate_tract_counts_to_blocks(
    df: pd.DataFrame,
    count_cols: Iterable[str],
    *,
    tract_key: str = "tract_id_synth",
) -> pd.DataFrame:
    """Split each tract count across the tract's blocks in proportion to a block weight.

    After the block<->tract merge every block carries its tract's totals verbatim, so
    summing a count over blocks multiplies it by the number of blocks. Each count is
    rewritten in place to ``tract_total * block_weight / sum(block_weight over the
    tract)`` — the count allocation uscensus_tiger_join_gpd uses — so the parts sum
    back to the tract total. Household-universe counts (``HOUSEHOLD_UNIVERSE_COUNTS``)
    are weighted by block households (``total_hh``, from H9) and all others by block
    population (``total_pop``, from P1); without H9, household counts are weighted by
    population instead, with a warning. Tracts whose weights sum to zero get zero.

    Args:
        df: The merged block+tract frame, one row per block.
        count_cols: Tract count columns to split (never ``perc_*`` ratios).
        tract_key: Column grouping blocks by their tract.

    Returns:
        ``df`` with each available count column split in place.
    """
    if tract_key not in df.columns:
        return df
    fallback_cols: list[str] = []
    for col in count_cols:
        if col not in df.columns:
            continue
        weight = "total_hh" if col in HOUSEHOLD_UNIVERSE_COUNTS else "total_pop"
        if weight not in df.columns and weight == "total_hh" and "total_pop" in df.columns:
            fallback_cols.append(col)
            weight = "total_pop"
        if weight not in df.columns:
            logging.warning(
                "Cannot split tract count '%s' to blocks: no '%s' weight column; it is left "
                "as a whole-tract figure on every block.",
                col,
                weight,
            )
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        weights = pd.to_numeric(df[weight], errors="coerce").fillna(0.0)
        tract_weight = weights.groupby(df[tract_key]).transform("sum")
        df[col] = values * np.where(tract_weight > 0, weights / tract_weight, 0.0)
    if fallback_cols:
        logging.warning(
            "No block household counts (H9) were supplied, so household counts %s were "
            "split across blocks by population instead of households.",
            fallback_cols,
        )
    return df


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
def build_joined_table(
    *,
    pop_files: list[str],
    hh_files: list[str],
    jobs_files: list[str],
    income_files: list[str] | None = None,
    ethnicity_files: list[str] | None = None,
    language_files: list[str] | None = None,
    vehicle_files: list[str] | None = None,
    age_files: list[str] | None = None,
    commute_files: list[str] | None = None,
    county_fips_filter: Iterable[str] | None = None,
    _clean_columns: bool = True,
) -> pd.DataFrame:
    """Return a fully joined block + tract DataFrame with optional FIPS filter.

    Tract percentages (``perc_*``) ride along per block unchanged; tract counts are
    split across each tract's blocks by block population (P1) or households (H9), so
    every count column sums over blocks to the tract total.
    """
    block_df = _build_block_df(_BlockInputs(pop_files, hh_files, jobs_files))
    tract_df = _build_tract_df(
        _TractInputs(
            income_files or [],
            ethnicity_files or [],
            language_files or [],
            vehicle_files or [],
            age_files or [],
            commute_files or [],
        )
    )

    combined = (
        block_df
        if tract_df.empty
        else block_df.merge(
            tract_df,
            left_on="tract_id_synth",
            right_on="tract_id_clean",
            how="outer",
            suffixes=("_blk", "_trt"),
            validate="m:1",
        )
    )

    if _clean_columns:
        combined = _drop_unfriendly_cols(combined)

    combined = _apply_fips_filter(combined, fips=county_fips_filter)
    _fill_numeric_only(combined)
    # Count allocation (as in uscensus_tiger_join_gpd): split every tract count across
    # the tract's blocks, so block rows can be summed without multiplying tract totals.
    return allocate_tract_counts_to_blocks(combined, _tract_count_columns(tract_df))


__all__ = ["build_joined_table"]

# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Orchestrate discovery, join, and optional CSV export.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    defaults_found = False
    if str(ROOT_DATA_DIR) == _DEFAULT_ROOT_DATA_DIR:
        logging.warning("ROOT_DATA_DIR is still the placeholder value — update it before running.")
        defaults_found = True
    if OUTPUT_DIR is not None and str(OUTPUT_DIR) == _DEFAULT_OUTPUT_DIR:
        logging.warning("OUTPUT_DIR is still the placeholder value — update it before running.")
        defaults_found = True
    if defaults_found:
        logging.info("No processing performed. Update the configuration paths and re-run.")
        return 2

    try:
        logging.info("Discovering Census datasets under %s …", ROOT_DATA_DIR)
        discovered = discover_census_files(ROOT_DATA_DIR)

        df_joined = build_joined_table(
            pop_files=discovered["POP_FILES"],
            hh_files=discovered["HH_FILES"],
            jobs_files=discovered["JOBS_FILES"],
            income_files=discovered["INCOME_FILES"],
            ethnicity_files=discovered["ETHNICITY_FILES"],
            language_files=discovered["LANGUAGE_FILES"],
            vehicle_files=discovered["VEHICLE_FILES"],
            age_files=discovered["AGE_FILES"],
            commute_files=discovered["COMMUTE_FILES"],
            county_fips_filter=COUNTY_FIPS_FILTER,
        )
        logging.info("Created DataFrame with shape %s", df_joined.shape)

        if CSV_OUTPUT_PATH:
            out_path = Path(CSV_OUTPUT_PATH).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_joined.to_csv(out_path, index=False)
            logging.info("CSV written to %s", out_path)

        logging.info("Script completed successfully.")
    except Exception:  # noqa: BLE001
        logging.exception("Processing failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
