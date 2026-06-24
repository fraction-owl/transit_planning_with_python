"""End-to-end open-source pipeline: merge Census CSVs, merge TIGER shapefiles, join to output.

GeoPandas equivalent of the ArcPy pipeline.  Runs three stages in one process,
passing DataFrames between stages in memory — no intermediate disk writes are
required, though they can be enabled via configuration.

Stages
------
1) CSV stage (pandas):
        Discover and merge Census + LODES CSVs into a single block-level
        attribute table, optionally filtered by county FIPS.
2) TIGER stage (GeoPandas):
        Discover and merge TIGER/Line tabblock shapefiles into a single
        GeoDataFrame, with the same optional FIPS filter applied.
3) Join stage (GeoPandas):
        Merge attributes onto block geometry on the 15-digit block FIPS
        identifier and write the final output.

Configuration
-------------
At minimum, set INPUT_CSV_DIR, INPUT_SHP_DIR, and FINAL_JOINED_FEATURES.
Set INTERMEDIATE_COMBINED_CSV or INTERMEDIATE_MERGED_SHP to an empty string
to skip writing those intermediate artifacts; the pipeline will still run
in memory.

Notes:
-----
* All TIGER input layers must share the same CRS.
* Shapefile outputs truncate column names to 10 chars; use a GeoPackage
  (``.gpkg``) to preserve full names.
* Zipped TIGER archives are read transparently via the ``zip://`` VFS prefix.

Helpful links
-------------
    Demographic data: https://data.census.gov/table
    Jobs data:        https://lehd.ces.census.gov/data/
    Geographic data:  https://www.census.gov/cgi-bin/geo/shapefiles/index.php
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Hashable,
    Iterable,
    List,
    Literal,
    Mapping,
    Sequence,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Stage 1: CSV inputs ----------------------------------------------------
#: Root folder holding every Census download (plain CSV, *.csv.gz*, or ZIPs).
#: Sub-directories are searched automatically.
INPUT_CSV_DIR: str | Path = r"Path\To\Your\Census_Table_Data_Files"  # <<< EDIT ME

# ---- Stage 2: TIGER inputs --------------------------------------------------
#: Root folder containing one or more TIGER/Line shapefiles. Sub-directories
#: are searched automatically; zipped archives are read transparently.
INPUT_SHP_DIR: str | Path = r"Path\To\Your\TIGER_Shapefiles"  # <<< EDIT ME

#: Glob pattern matched against TIGER shapefile (and matching .zip) basenames.
#: Typical: "tl_*_*_tabblock20.shp" for blocks, "tl_*_*_bg.shp" for block groups.
TIGER_INPUT_GLOB: str = "tl_*_*_tabblock20.shp"

# ---- Shared: optional FIPS filter ------------------------------------------
#: 5-digit county FIPS codes. Applied in both Stage 1 and Stage 2.
#: Leave empty ([]) to export everything.
FIPS_TO_FILTER: List[str] = [
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

# ---- Outputs ----------------------------------------------------------------
#: Intermediate combined CSV from Stage 1. Set to "" or None to skip writing.
INTERMEDIATE_COMBINED_CSV: str | None = r"Path\To\Your\Output_Folder\joined_blocks.csv"

#: Intermediate merged geometry from Stage 2 (Shapefile *.shp or GeoPackage
#: *.gpkg). Set to "" or None to skip writing.
INTERMEDIATE_MERGED_SHP: str | None = r"Path\To\Your\Output_Folder\va_md_dc_blocks_fips_merge.shp"

#: Final joined geometry + attributes (Stage 3 output).
FINAL_JOINED_FEATURES: str = r"Path\To\Your\Output_Folder\va_md_dc_blocks_plus_data.shp"

# ---- Join settings ----------------------------------------------------------
LEFT_KEY: Final[str] = "GEOID20"  # 15-digit block ID in geometry
RIGHT_KEY: Final[str] = "GEO_ID"  # 24-char Census ID in CSV (last 15 chars used)
DERIVATION_SRC: Final[str] = "GEO_ID_blk"  # fallback CSV column post-merge
FORCE_FLOAT: Final[bool] = True  # cast nullable Int64 → float64 for shapefile safety
MAX_FIELD_LEN: Final[int] = 10  # DBF column-name limit

# ---- CSV topic signatures ---------------------------------------------------
#: File-name token signatures used to bucket CSV/GZ/ZIP inputs by topic.
#: ALL tokens listed for a topic must appear in the file name (case-insensitive).
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

# ---- Tract -> block disaggregation ------------------------------------------
#: Income, ethnicity, language, vehicle and age tables arrive at the tract level, so
#: after the block<->tract join every block in a tract carries that tract's totals
#: verbatim. For an *additive count* that over-counts — each block claims the whole
#: tract figure — so each configured count is split across the tract's blocks in
#: proportion to a block-level weight (population for person counts, households for
#: household counts): ``tract_total * block_weight / sum(block_weight over the tract)``.
#: The parts then sum back to the tract total, and a partial-area service-area clip
#: downstream keeps a proportional slice. Each entry maps the derived tract column to
#: ``(weight_column, output_column)``; output names stay <=10 chars so the Shapefile
#: writer does not truncate them. ``perc_*`` ratio columns are intentionally excluded —
#: percentages are not additive and must never be area-weighted.
TRACT_COUNT_DISAGG: dict[str, tuple[str, str]] = {
    "low_income": ("total_hh", "low_income"),  # households under the low-income bands
    "minority": ("total_pop", "minority"),  # non-white-alone residents
    "all_nwell": ("total_pop", "lep"),  # limited-English-proficiency residents
    "all_lo_veh_hh": ("total_hh", "lo_veh_hh"),  # households with 0-1 vehicles
    "all_lo_veh_hh_mod": ("total_hh", "lo_veh_mod"),  # low-vehicle, excl. 1-person/1-vehicle
    "all_youth": ("total_pop", "youth"),  # residents age 15-21
    "all_elderly": ("total_pop", "elderly"),  # residents age 65+
    # Commuting (ACS S0801) worker counts, reconstructed from percentages in
    # _derive_commute. Weighted by population (blocks carry no worker count); the
    # percentages/mean themselves are intentionally absent here — never area-weighted.
    "commute_workers": ("total_pop", "cmt_wrkrs"),  # workers 16+
    "commute_transit": ("total_pop", "cmt_trnst"),  # commute by public transit
    "commute_drove": ("total_pop", "cmt_drove"),  # drove alone
    "commute_carpool": ("total_pop", "cmt_carpl"),  # carpooled
    "commute_wfh": ("total_pop", "cmt_wfh"),  # worked from home
    "commute_person_min": ("total_pop", "cmt_pmin"),  # worker-minutes (mean = /workers)
}

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# ---- Sentinel defaults — detect un-edited placeholder paths ----------------
_DEFAULT_INPUT_CSV_DIR: str = r"Path\To\Your\Census_Table_Data_Files"
_DEFAULT_INPUT_SHP_DIR: str = r"Path\To\Your\TIGER_Shapefiles"
_DEFAULT_INTERMEDIATE_COMBINED_CSV: str = r"Path\To\Your\Output_Folder\joined_blocks.csv"
_DEFAULT_INTERMEDIATE_MERGED_SHP: str = r"Path\To\Your\Output_Folder\va_md_dc_blocks_fips_merge.shp"
_DEFAULT_FINAL_JOINED_FEATURES: str = r"Path\To\Your\Output_Folder\va_md_dc_blocks_plus_data.shp"


class _Unset:
    """Sentinel type for _check_placeholders args whose valid values include None/""."""


#: Sentinel instance: "argument not supplied, fall back to the module constant".
_UNSET: Final[_Unset] = _Unset()

# =============================================================================
# STAGE 1: CSV DISCOVERY & MERGE  (pandas)
# =============================================================================

GEO_ID_COL: Final[str] = "GEO_ID"
_UNFRIENDLY_COL_RE = re.compile(r"^[A-Z]{2,}\d{3,}.*")


def _token_match(name: str, tokens: Sequence[str] | str) -> bool:
    """Return True if *all* tokens occur in *name* (case-insensitive)."""
    if isinstance(tokens, str):
        tokens = (tokens,)
    low = name.lower()
    return all(tok.lower() in low for tok in tokens)


def _zip_data_member_sizes(zip_path: str | Path) -> dict[str, int]:
    """Map each '*-Data.csv' member's base name to its uncompressed size in bytes.

    Reads only the ZIP central directory (no extraction). Returns an empty map
    when the archive cannot be read, so a corrupt ZIP never suppresses a loose CSV.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return {
                Path(info.filename).name.lower(): info.file_size
                for info in zf.infolist()
                if info.filename.lower().endswith("-data.csv")
            }
    except (zipfile.BadZipFile, OSError):
        return {}


def _dedupe_extracted_zip_members(paths: Sequence[str]) -> list[str]:
    """Drop loose '*-Data.csv' files that duplicate a '-Data.csv' member of a ZIP.

    ``_read_csv_any`` reads a Census ZIP's '-Data.csv' member directly, so when
    the same archive has also been unzipped in place — which the feature-prep
    orchestrator does by default, and a human may do manually — the scanned root
    holds both the ``*.zip`` and the extracted ``*-Data.csv``. Bucketing both
    would concatenate the identical table twice (duplicate GEO_ID rows). A loose
    CSV is treated as redundant only when its base name AND byte size match a
    member of a ZIP in the same bucket, so the ZIP is kept and the extracted copy
    dropped; genuinely distinct downloads (e.g. other geographies, which differ
    in size) are never removed.
    """
    member_sizes: dict[str, set[int]] = {}
    for p in paths:
        if p.lower().endswith(".zip"):
            for name, size in _zip_data_member_sizes(p).items():
                member_sizes.setdefault(name, set()).add(size)
    if not member_sizes:
        return list(paths)

    kept: list[str] = []
    for p in paths:
        base = Path(p).name.lower()
        if base.endswith("-data.csv") and base in member_sizes:
            try:
                if Path(p).stat().st_size in member_sizes[base]:
                    logging.info(
                        "Skipping '%s'; byte-identical to a ZIP member already in this bucket.",
                        p,
                    )
                    continue
            except OSError:
                pass
        kept.append(p)
    return kept


def discover_census_files(
    root_dir: str | Path,
    signatures: Mapping[str, Sequence[str] | str] = TOPIC_SIGNATURES,
) -> dict[str, list[str]]:
    """Recursively locate Census data files and bucket them by topic.

    Accepts plain CSV, CSV.GZ, or ZIP archives. ZIPs are returned as the
    ZIP path itself; content is handled later by ``_read_csv_any``. A loose
    ``*-Data.csv`` that merely duplicates a ZIP member already in the same
    bucket (e.g. an in-place unzip) is dropped, so the table is read once.
    File order is sorted for determinism.
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
    """Read a CSV/CSV.GZ directly, or the first '-Data.csv' member in a ZIP."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith("-data.csv")]
            if not members:
                raise FileNotFoundError(f"No '*-Data.csv' inside {p}")
            with zf.open(members[0]) as fh, io.TextIOWrapper(fh, encoding="utf-8") as txt:
                return pd.read_csv(txt, **read_kwargs)
    return pd.read_csv(p, **read_kwargs)


def _fill_numeric_only(df: pd.DataFrame, value: int | float = 0) -> pd.DataFrame:
    """Replace only numeric NaNs with *value*; leave object columns untouched."""
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(value)
    return df


def _clean_name_cols(df: pd.DataFrame) -> None:
    """Sanitize NAME-like columns in place (remove CR/LF/TAB)."""
    for col in df.filter(regex=r"^NAME").columns:
        df[col] = df[col].astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.strip()


def _dedupe_topic_rows(df: pd.DataFrame, key: Hashable, *, source: str) -> pd.DataFrame:
    """Collapse rows that repeat *key* to the first occurrence, logging any drops.

    A topic bucket can legitimately gather more than one input file for the same
    geography: multiple ACS vintages of a table, or race/ethnicity iteration
    tables (e.g. ``B19001A``..``B19001I``, ``B01001A``..``B01001I``) whose codes
    contain the base topic's token and so match the same signature. Concatenated,
    those files repeat every ``GEO_ID``, and because the later GEO_ID merges and
    the one-to-many block<->tract join both fan out on the key, each repeat becomes
    a *multiplicative* row explosion — enough to violate the Stage 3 ``1:1`` join
    and abort the whole pipeline.

    Collapsing here, at load time and before any merge, keeps each geography to a
    single row. Files are read in sorted order, so ``keep="first"`` deterministically
    prefers the base table over its race iterations (``B19001`` sorts before
    ``B19001A``) and, for true vintage duplicates, the earliest file.
    """
    if key not in df.columns:
        return df
    before = len(df)
    deduped = df.drop_duplicates(subset=[key], keep="first")
    dropped = before - len(deduped)
    if dropped:
        logging.warning(
            "Dropped %d row(s) repeating '%s' while loading %s (kept the first of each). "
            "More than one input file covered the same geography — typically multiple ACS "
            "vintages of a table, or race-iteration tables sharing the topic's token. Keep "
            "one file per topic per geography to avoid this.",
            dropped,
            key,
            source,
        )
    return deduped.reset_index(drop=True)


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
    """Read multiple Census CSV / CSV-GZ / ZIP files and concatenate the results.

    Embedded control characters in NAME columns are stripped immediately to
    guarantee that every logical record remains on a single physical line
    when the final DataFrame is exported.

    Column renaming occurs *before* column pruning via ``usecols`` (unless
    ``usecols`` is explicitly supplied).

    When ``dedupe_key`` is set (default ``GEO_ID``) and present in the combined
    frame, rows repeating that key are collapsed to the first occurrence so a
    topic that gathered several files for the same geography (multiple ACS
    vintages, or race-iteration tables matched by the same signature) cannot fan
    out through the downstream merges. Pass ``dedupe_key=None`` to disable.
    """
    frames: list[pd.DataFrame] = []
    for path in files:
        read_kwargs: dict[str, Any] = {}
        if compression is not None:
            read_kwargs["compression"] = compression
        if skiprows is not None:
            read_kwargs["skiprows"] = skiprows
        if dtype is not None:
            read_kwargs["dtype"] = dtype
        if usecols is not None:
            read_kwargs["usecols"] = usecols

        df = _read_csv_any(path, **read_kwargs)
        _clean_name_cols(df)

        if rename:
            df = df.rename(columns=rename)
            if usecols is None:
                keep = {GEO_ID_COL, "NAME", *rename.values()}
                df = df.loc[:, df.columns.intersection(keep)]

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if dedupe_key is not None:
        combined = _dedupe_topic_rows(combined, dedupe_key, source=f"{len(files)} file(s)")
    return combined


def _merge_on_geo_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge two frames on GEO_ID, dropping duplicate columns."""
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    dup = (set(left.columns) & set(right.columns)) - {GEO_ID_COL}
    return left.merge(right.drop(columns=dup), on=GEO_ID_COL, how="outer")


def _drop_unfriendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any column that still looks like a raw Census code."""
    to_drop = [c for c in df.columns if _UNFRIENDLY_COL_RE.match(c)]
    return df.drop(columns=to_drop, errors="ignore")


# ----- Block-level build -----------------------------------------------------


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
        # LODES is keyed on the block geocode, not GEO_ID; dedupe on it so several
        # WAC vintages do not repeat a block and fan out in the merge below.
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


# ----- Tract-level derivations -----------------------------------------------


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
    df["perc_low_income"] = df["low_income"] / df["total_hh"]
    return df.drop(columns="total_hh")


def _derive_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    minority = ["black", "native", "asian", "pac_isl", "other", "multi"]
    df["minority"] = df[minority].sum(axis=1)
    df["perc_minority"] = df["minority"] / df["total_pop"]
    return df.drop(columns="total_pop")


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
    youth = ["m_15_17", "f_15_17", "m_18_19", "f_18_19", "m_20", "f_20", "m_21", "f_21"]
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
        df["perc_youth"] = (df["all_youth"] / df["total_pop"]).round(3)
        df["perc_elderly"] = (df["all_elderly"] / df["total_pop"]).round(3)
        df = df.drop(columns="total_pop")
    return df


def _derive_commute(df: pd.DataFrame) -> pd.DataFrame:
    """Derive commuting measures from ACS S0801 (Commuting Characteristics).

    S0801's means-of-transportation rows are *percentages* of workers 16+ (and
    travel time is a mean in minutes), so they can ride along verbatim per tract
    but must never be area-weighted. The additive worker *counts* derived here
    (``workers * pct / 100``) plus person-minutes are what TRACT_COUNT_DISAGG
    splits down to blocks; a catchment mean travel time is then recoverable as
    ``sum(commute_person_min) / sum(commute_workers)``.
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
    df["commute_person_min"] = workers * df["mean_travel_time"]
    # ``mean_travel_time`` is a non-additive mean: drop it from this block-bound path
    # (build_tract_attributes sums count columns) — it is recoverable downstream as
    # commute_person_min / commute_workers. The flat uscensus_table_build keeps it.
    df = df.drop(columns="mean_travel_time")
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
                "C16001_032E": "asiapacetc_engnwell",
                "C16001_035E": "arabic_engnwell",
                "C16001_037E": "otheretc_engnwell",
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


# ----- CSV-stage FIPS helpers ------------------------------------------------


def _ensure_fips_column_df(
    df: pd.DataFrame,
    *,
    dst: str = "FIPS",
    geo_candidates: tuple[str, ...] = ("GEO_ID", "GEO_ID_blk", "GEO_ID_trt"),
    start: int = 9,
    end: int = 14,
) -> None:
    """Create a 5-digit county FIPS column in place from the first GEO_ID."""
    if dst in df.columns:
        return
    source = next((c for c in geo_candidates if c in df.columns), None)
    if source is None:
        raise KeyError(f"No GEO_ID column found among {geo_candidates}")
    df[dst] = df[source].astype(str).str[start:end]


def _apply_fips_filter_df(
    df: pd.DataFrame,
    *,
    fips: Iterable[str] | None = None,
    dst_col: str = "FIPS",
) -> pd.DataFrame:
    """Return a copy filtered to *fips* (or unchanged if *fips* is empty/None)."""
    if not fips:
        return df
    _ensure_fips_column_df(df, dst=dst_col)
    wanted = {str(code).zfill(5) for code in fips}
    return df[df[dst_col].isin(wanted)].copy()


# ----- Tract -> block disaggregation -----------------------------------------


def disaggregate_tract_counts_to_blocks(
    df: pd.DataFrame,
    *,
    tract_key: str = "tract_id_synth",
    field_weights: Mapping[str, tuple[str, str]] = TRACT_COUNT_DISAGG,
) -> pd.DataFrame:
    """Split each tract-level count across its blocks in proportion to a block weight.

    After the block<->tract merge, every block in a tract carries that tract's totals
    verbatim. For an additive count (households below an income threshold, minority
    residents, ...) that copy-down over-counts: each block claims the whole tract
    figure. This rewrites each configured count to the block's share —
    ``tract_total * block_weight / sum(block_weight over the tract)`` — so the parts sum
    back to the tract total and a partial-area clip downstream keeps a proportional
    slice. Tracts whose weight sums to zero receive zero (no basis to apportion). The
    result is written under the configured output column (the source column is dropped
    when the name changes); ``perc_*`` ratio columns are never touched.

    Args:
        df: The merged block+tract frame, keyed per block, with ``tract_key`` present.
        tract_key: Column grouping blocks by their tract (block GEO_ID's tract slice).
        field_weights: ``{source_count: (weight_column, output_column)}``.

    Returns:
        ``df`` with each available count column disaggregated and renamed in place.
    """
    if tract_key not in df.columns:
        return df
    for src, (weight, out) in field_weights.items():
        if src not in df.columns or weight not in df.columns:
            continue
        values = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
        weights = pd.to_numeric(df[weight], errors="coerce").fillna(0.0)
        tract_weight = weights.groupby(df[tract_key]).transform("sum")
        share = np.where(tract_weight > 0, weights / tract_weight, 0.0)
        df[out] = values * share
        if out != src:
            df = df.drop(columns=src)
    return df


# ----- Stage 1 public entry point --------------------------------------------


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

    Legacy path that assumes *block-level* P1/H9 census tables. ``run()`` now builds the
    block layer with ``build_tract_attributes`` + ``build_block_jobs`` +
    ``attach_demographics_to_blocks``, which sources block population/households from the
    TIGER blocks' POP20/HOUSING20 and so works with tract-level Census downloads. Kept
    for callers that genuinely have block-level P1/H9 tables.
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

    # Filter each frame to the requested counties BEFORE the block<->tract join. Both
    # carry a county FIPS inside their GEO_ID, so each filters on its own key; the
    # temporary column is dropped so it cannot collide under the merge suffixes. This
    # is output-equivalent to filtering after the merge but keeps the one-to-many join
    # (and the disaggregation below) off every out-of-area block in a full-state input.
    if county_fips_filter:
        block_df = _apply_fips_filter_df(block_df, fips=county_fips_filter).drop(
            columns="FIPS", errors="ignore"
        )
        if not tract_df.empty:
            tract_df = _apply_fips_filter_df(tract_df, fips=county_fips_filter).drop(
                columns="FIPS", errors="ignore"
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
        )
    )

    if _clean_columns:
        combined = _drop_unfriendly_cols(combined)

    # Re-attach the single 'FIPS' column (the per-frame ones were dropped above). With
    # both inputs already filtered this is a cheap no-op filter, but it keeps one 'FIPS'
    # on the output and drops any tract row the outer merge left unmatched to a block.
    combined = _apply_fips_filter_df(combined, fips=county_fips_filter)
    _fill_numeric_only(combined)
    combined = disaggregate_tract_counts_to_blocks(combined)
    return combined


def build_tract_attributes(
    *,
    income_files: list[str] | None = None,
    ethnicity_files: list[str] | None = None,
    language_files: list[str] | None = None,
    vehicle_files: list[str] | None = None,
    age_files: list[str] | None = None,
    commute_files: list[str] | None = None,
    county_fips_filter: Iterable[str] | None = None,
    _clean_columns: bool = True,
) -> pd.DataFrame:
    """Build a tract-keyed table of additive demographic counts.

    The income / ethnicity / language / vehicle / age / commute tables are tract-level
    (or coarser), so this collapses them to one row per 11-digit tract (``tract_fips``)
    with the additive count columns summed. Block-level population and households are NOT
    sourced here: they come from the TIGER blocks' POP20/HOUSING20 in
    ``attach_demographics_to_blocks``, so a tract-level Census download is sufficient and
    no block-level P1/H9 table is required. Percent (``perc_*``) ratio columns are
    dropped — they are not additive and would be meaningless once summed.
    """
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
    if tract_df.empty:
        return tract_df
    if _clean_columns:
        tract_df = _drop_unfriendly_cols(tract_df)
    tract_df = _apply_fips_filter_df(tract_df, fips=county_fips_filter)
    tract_df["tract_fips"] = tract_df["tract_id_clean"].astype(str).str[:11]
    count_cols = [
        str(c)
        for c in tract_df.columns
        if c != "tract_fips"
        and not str(c).startswith("perc_")
        and pd.api.types.is_numeric_dtype(tract_df[c])
    ]
    return tract_df.groupby("tract_fips", as_index=False)[count_cols].sum()


def build_block_jobs(
    jobs_files: list[str],
    *,
    county_fips_filter: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a block-keyed LEHD WAC jobs table, keyed on the 15-digit ``block_fips``.

    LEHD WAC is natively block-level (``w_geocode``), so it stays at the block level and
    joins straight onto the TIGER blocks — no disaggregation needed. Several WAC vintages
    are collapsed to one row per block.
    """
    jobs = _load_and_concat(
        jobs_files,
        rename={"C000": "tot_empl", "CE01": "low_wage", "CE02": "mid_wage", "CE03": "high_wage"},
        usecols=["w_geocode", "C000", "CE01", "CE02", "CE03"],
        dedupe_key="w_geocode",
    )
    if jobs.empty:
        return jobs
    jobs["block_fips"] = jobs["w_geocode"].astype(str).str.zfill(15)
    jobs = jobs.drop(columns="w_geocode")
    if county_fips_filter:
        wanted = {str(code).zfill(5) for code in county_fips_filter}
        jobs = jobs[jobs["block_fips"].str[:5].isin(wanted)].copy()
    _fill_numeric_only(jobs)
    return jobs


# =============================================================================
# STAGE 2: TIGER DISCOVERY & MERGE  (GeoPandas)
# =============================================================================


def discover_tiger_datasets(
    root_dir: str | Path,
    pattern: str = "tl_*_*_*.shp",
    *,
    prefer: str = "shp",  # "shp" → use plain files when both exist, "zip" → reverse
) -> List[str]:
    """Return absolute paths to TIGER shapefiles, plain or zipped.

    Search is recursive. ``pattern`` is applied to both *.shp* and *.zip*
    names. Zipped archives are returned with the ``zip://`` VFS prefix so
    GeoPandas can open them directly.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a valid directory")

    shp_paths = list(root.rglob(pattern))
    zip_glob = re.sub(r"\.shp$", ".zip", pattern, flags=re.IGNORECASE)
    zip_paths = list(root.rglob(zip_glob))

    chosen: "OrderedDict[str, Path]" = OrderedDict()
    for p in sorted(shp_paths + zip_paths):
        stem = p.stem
        ext = p.suffix.lower()
        if stem in chosen:
            keep_zip = prefer == "zip"
            if keep_zip and ext == ".zip":
                chosen[stem] = p
            elif not keep_zip and ext == ".shp":
                chosen[stem] = p
        else:
            chosen[stem] = p

    if not chosen:
        raise FileNotFoundError(f"No datasets matching '{pattern}' were found under {root}")

    logging.info(
        "Discovered %d TIGER dataset(s) (%d plain, %d zipped)",
        len(chosen),
        sum(p.suffix == ".shp" for p in chosen.values()),
        sum(p.suffix == ".zip" for p in chosen.values()),
    )

    return sorted(f"zip://{p}" if p.suffix.lower() == ".zip" else str(p) for p in chosen.values())


def _read_shapefile(path: str) -> GeoDataFrame:
    """Read a single shapefile as a GeoDataFrame and normalize ID columns."""
    logging.info("Reading %s", path)
    gdf = gpd.read_file(path)
    for col in ("STATEFP20", "STATEFP", "COUNTYFP20", "COUNTYFP"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str).str.zfill(2 if "STATE" in col else 3)
    return gdf


def merge_shapefiles(shp_paths: Sequence[str]) -> GeoDataFrame:
    """Load and concatenate multiple shapefiles. All inputs must share a CRS."""
    gdfs: list[GeoDataFrame] = [_read_shapefile(p) for p in shp_paths]

    crs_set = {str(gdf.crs) for gdf in gdfs}
    if len(crs_set) != 1:
        raise RuntimeError(
            "CRS mismatch between input layers: %s. Re-project first." % ", ".join(crs_set)
        )

    merged = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs,
        geometry="geometry",
    )
    logging.info("Merged %d input files → %d features", len(shp_paths), len(merged))
    return merged


def ensure_fips_column(
    gdf: GeoDataFrame,
    *,
    fips_col: str = "FIPS",
    state_candidates: tuple[str, ...] = ("STATEFP20", "STATEFP"),
    county_candidates: tuple[str, ...] = ("COUNTYFP20", "COUNTYFP"),
) -> GeoDataFrame:
    """Add a 5-digit county FIPS column to *gdf* if it does not already exist."""
    if fips_col in gdf.columns:
        logging.info("Field %s already present — skipping creation", fips_col)
        return gdf

    state_field = next((c for c in state_candidates if c in gdf.columns), None)
    county_field = next((c for c in county_candidates if c in gdf.columns), None)
    if state_field is None or county_field is None:
        raise KeyError(
            "Required columns not found. Expected one of %s and one of %s."
            % (state_candidates, county_candidates)
        )

    gdf[fips_col] = gdf[state_field].astype(str).str.zfill(2) + gdf[county_field].astype(
        str
    ).str.zfill(3)
    logging.info("Populated new column %s", fips_col)
    return gdf


def filter_by_fips(
    gdf: GeoDataFrame,
    fips_values: Sequence[str],
    *,
    fips_col: str = "FIPS",
) -> GeoDataFrame:
    """Return a view of *gdf* containing only requested FIPS codes."""
    if not fips_values:
        logging.info("FIPS filter empty — exporting full dataset")
        return gdf

    mask = gdf[fips_col].isin(fips_values)
    selected = gdf.loc[mask].copy()
    if selected.empty:
        logging.warning("No features matched the FIPS list — output will be empty")
    else:
        logging.info("Selected %d of %d features", len(selected), len(gdf))
    return selected


# =============================================================================
# STAGE 3: JOIN ATTRIBUTES → GEOMETRY  (GeoPandas)
# =============================================================================


def normalize_attribute_keys(
    df: DataFrame,
    *,
    key: str = RIGHT_KEY,
    derivation_src: str = DERIVATION_SRC,
) -> DataFrame:
    """Ensure *df* has a 15-digit join key column suitable for block geometry.

    The Census 24-char ``GEO_ID`` (e.g. ``1000000US110010001011000``) is
    normalized to its rightmost 15 chars to match TIGER ``GEOID20``.
    """
    if key in df.columns:
        df[key] = df[key].astype(str).str[-15:]
    elif derivation_src in df.columns:
        df[key] = df[derivation_src].astype(str).str[-15:]
    else:
        raise KeyError(f"Neither '{key}' nor '{derivation_src}' found in attribute frame.")
    return df


def join_blocks_to_attributes(
    blocks: GeoDataFrame,
    attrs: DataFrame,
    *,
    left_key: str = LEFT_KEY,
    right_key: str = RIGHT_KEY,
    how: Literal["left", "right", "outer", "inner", "cross"] = "left",
) -> GeoDataFrame:
    """Merge *attrs* onto *blocks* on the specified keys.

    Raises:
        ValueError: If duplicates in either key violate a 1:1 expectation.
    """
    blocks = blocks.copy()
    blocks[left_key] = blocks[left_key].astype(str)

    logging.info("Merging geometry (%d) with table (%d) …", len(blocks), len(attrs))
    merged: GeoDataFrame = blocks.merge(
        attrs,
        left_on=left_key,
        right_on=right_key,
        how=how,
        validate="1:1",
    )
    logging.info("Merged result → %d rows, %d columns", *merged.shape)

    if FORCE_FLOAT:
        _cast_int64_to_float(merged)

    return merged


def attach_demographics_to_blocks(
    blocks: GeoDataFrame,
    tract_attrs: DataFrame,
    block_jobs: DataFrame | None = None,
    *,
    block_key: str = LEFT_KEY,
    pop_col: str = "POP20",
    hh_col: str = "HOUSING20",
    field_weights: Mapping[str, tuple[str, str]] = TRACT_COUNT_DISAGG,
) -> GeoDataFrame:
    """Attach tract demographics to every block and split them down by block weight.

    The TIGER block layer is the spine. Each block is matched to its tract
    (``GEOID20[:11]``) and receives that tract's additive counts, which are then
    apportioned to the block in proportion to its share of the tract's TIGER ``POP20``
    (person counts) or ``HOUSING20`` (household counts) — the block-level population and
    housing the 2020 Census already records in the block layer. ``total_pop`` /
    ``total_hh`` are taken straight from POP20/HOUSING20. LEHD jobs, natively block-level,
    are joined on the block id. This means tract-level Census tables are sufficient; no
    block-level P1/H9 download is required.

    Args:
        blocks: TIGER block geometry, carrying ``block_key`` plus ``pop_col``/``hh_col``.
        tract_attrs: One row per ``tract_fips`` with the additive count columns.
        block_jobs: Optional block-keyed LEHD jobs (``block_fips`` + job columns).
        block_key: 15-digit block id column on ``blocks`` (``GEOID20``).
        pop_col: Block population column (``POP20``) used as ``total_pop`` and person weight.
        hh_col: Block housing column (``HOUSING20``) used as ``total_hh`` and household weight.
        field_weights: ``{count: (weight_column, output_column)}`` for the split.

    Returns:
        ``blocks`` with ``total_pop``/``total_hh``, the disaggregated counts, and jobs.
    """
    merged = blocks.copy()
    merged[block_key] = merged[block_key].astype(str)
    merged["tract_fips"] = merged[block_key].str[:11]

    logging.info(
        "Attaching demographics: %d block(s), %d tract row(s), %d job row(s).",
        len(merged),
        0 if tract_attrs is None else len(tract_attrs),
        0 if block_jobs is None else len(block_jobs),
    )

    if tract_attrs is not None and not tract_attrs.empty:
        merged = merged.merge(tract_attrs, on="tract_fips", how="left", validate="m:1")

    # Block population & households come straight from TIGER; they double as the weights
    # that split each tract count back down to its blocks.
    for out_col, src_col in (("total_pop", pop_col), ("total_hh", hh_col)):
        if src_col in merged.columns:
            merged[out_col] = pd.to_numeric(merged[src_col], errors="coerce").fillna(0.0)
        else:
            logging.warning("TIGER blocks lack '%s'; '%s' set to 0.", src_col, out_col)
            merged[out_col] = 0.0

    if block_jobs is not None and not block_jobs.empty:
        merged = merged.merge(
            block_jobs, left_on=block_key, right_on="block_fips", how="left", validate="m:1"
        ).drop(columns="block_fips", errors="ignore")

    _fill_numeric_only(merged)
    merged = disaggregate_tract_counts_to_blocks(
        merged, tract_key="tract_fips", field_weights=field_weights
    )
    if FORCE_FLOAT:
        _cast_int64_to_float(merged)
    return merged


# =============================================================================
# SHARED OUTPUT HELPERS
# =============================================================================


def _cast_int64_to_float(gdf: GeoDataFrame) -> None:
    """Convert nullable Int64 columns to float64 in place for shapefile safety.

    Shapefile drivers cannot store pandas' nullable integer extension type.
    """
    int_cols: list[str] = [
        str(col) for col, dtype in gdf.dtypes.items() if pd.api.types.is_integer_dtype(dtype)
    ]
    if int_cols:
        logging.debug("Casting %d Int64 column(s) → float64: %s", len(int_cols), int_cols)
        gdf[int_cols] = gdf[int_cols].astype("float64")


def _truncate_field_names(gdf: GeoDataFrame, max_len: int = MAX_FIELD_LEN) -> GeoDataFrame:
    """Truncate attribute names to fit the Shapefile 10-char DBF limit.

    When a truncated name collides with one already used, a numeric suffix
    is appended to make it unique. Returns the (possibly renamed) GeoDataFrame.
    """
    renames: dict[str, str] = {}
    seen: set[str] = set()
    geom_name = gdf.geometry.name

    for col in list(gdf.columns):
        if col == geom_name:
            continue
        new = col[:max_len]
        counter = 1
        while new in seen:
            new = f"{col[: max_len - len(str(counter))]}{counter}"
            counter += 1
        if new != col:
            renames[col] = new
        seen.add(new)

    if renames:
        logging.warning(
            "Truncated %d column name(s) to %d chars: %s",
            len(renames),
            max_len,
            renames,
        )
        gdf = gdf.rename(columns=renames)
    return gdf


def _shp_schema(gdf: GeoDataFrame) -> dict:
    """Build a fiona write schema capping float fields to 1 decimal place.

    Fiona defaults to float:24.15 for float64, producing 15 trailing zeros for
    integer-valued fields and excessive precision for estimates. Using float:24.1
    limits all float columns to 1 decimal place in the DBF output.
    """
    geom_type = gdf.geom_type.mode().iloc[0] if not gdf.empty else "Unknown"
    props: dict[str, str] = {}
    for col, dtype in gdf.drop(columns=gdf.geometry.name).dtypes.items():
        col_str = str(col)
        if pd.api.types.is_float_dtype(dtype):
            props[col_str] = "float:24.1"
        elif pd.api.types.is_integer_dtype(dtype):
            props[col_str] = "int:18"
        elif pd.api.types.is_bool_dtype(dtype):
            props[col_str] = "int:1"
        else:
            # An object column that is entirely NaN (e.g. a tract attribute that
            # matched no block in a left join) yields a NaN max length -- and in
            # pandas 3.x ``astype(str)`` keeps NaN rather than writing "nan", so this
            # is common. Fall back to width 1 instead of ``int(NaN)``, which raises.
            width = 1
            if not gdf.empty:
                str_len = gdf[col].astype(str).str.len().max()
                if pd.notna(str_len):
                    width = int(str_len)
            props[col_str] = f"str:{max(width, 1)}"
    return {"geometry": geom_type, "properties": props}


def write_geo(gdf: GeoDataFrame, out_path: str) -> None:
    """Write *gdf* to disk, creating parent dirs if needed.

    Shapefile outputs have field names truncated to 10 chars and the implicit
    pandas index suppressed; other drivers are inferred from the extension.
    Float columns in shapefiles are written with 1 decimal place when fiona is
    available (see ``_write_shapefile``).
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing %d features → %s", len(gdf), path.resolve())
    if out_path.lower().endswith(".shp"):
        gdf = _truncate_field_names(gdf)
        _write_shapefile(gdf, out_path)
    else:
        gdf.to_file(out_path, index=False)


def _write_shapefile(gdf: GeoDataFrame, out_path: str) -> None:
    """Write *gdf* as an ESRI Shapefile, preferring fiona's float-precision schema.

    The fiona engine accepts a per-field ``schema`` (``_shp_schema``) that caps float
    columns to one decimal place in the DBF. fiona is awkward to install on newer
    Python versions (it needs a matching GDAL build and often lacks wheels), so when it
    is not importable we fall back to GeoPandas' default engine (pyogrio). The fallback
    drops only the cosmetic 1-decimal cap — geometry and attribute values are identical,
    and pyogrio is what GeoPandas already uses to read the inputs here.
    """
    try:
        import fiona  # noqa: F401  (presence check only)
    except ImportError:
        logging.info(
            "fiona is not installed; writing the Shapefile with GeoPandas' default "
            "engine (pyogrio), without the cosmetic 1-decimal float cap."
        )
        gdf.to_file(Path(out_path), driver="ESRI Shapefile", index=False)
        return
    gdf.to_file(
        out_path, driver="ESRI Shapefile", schema=_shp_schema(gdf), engine="fiona", index=False
    )


def write_csv(df: DataFrame, out_path: str) -> None:
    """Write *df* to disk as CSV, creating parent dirs if needed."""
    p = Path(out_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logging.info("CSV written → %s (rows=%d, cols=%d)", p, *df.shape)


# =============================================================================
# MAIN
# =============================================================================


def _is_blank(p: str | None) -> bool:
    """Treat None and empty/whitespace-only strings as 'skip this output'."""
    return p is None or not str(p).strip()


def _skip_placeholder_output(path: str | None, placeholder: str, flag: str) -> str | None:
    """Treat an optional output left at its template placeholder as 'skip'.

    The two intermediate artifacts are optional, so leaving their path un-edited
    must not abort the whole run -- only the required input/output paths do that.
    This is exactly the case the prep_features_public orchestrator hits: it wires
    ``--input-csv-dir``/``--input-shp-dir``/``--output`` but not the
    intermediates, so they fall back to the placeholder constants.

    Args:
        path: The configured (or defaulted) intermediate output path.
        placeholder: The template placeholder value for this output.
        flag: The CLI flag name, used only in the warning message.

    Returns:
        ``""`` (skip this artifact) when ``path`` is still the placeholder,
        otherwise ``path`` unchanged.
    """
    if path is not None and str(path) == placeholder:
        logging.warning(
            "%s left at its placeholder; skipping that optional artifact. "
            "Pass a real path to write it, or '' to silence this warning.",
            flag,
        )
        return ""
    return path


def _check_placeholders(
    input_csv_dir: str | Path | None = None,
    input_shp_dir: str | Path | None = None,
    final_joined_features: str | None = None,
    intermediate_combined_csv: str | None | _Unset = _UNSET,
    intermediate_merged_shp: str | None | _Unset = _UNSET,
) -> bool:
    """Warn about un-edited placeholder paths. Return True if any are still set.

    Unset args fall back to the module-level CONFIGURATION constants at call
    time, so a plain ``_check_placeholders()`` (and monkeypatching those
    constants) behaves as before, while ``run()`` can pass resolved CLI values.
    The two intermediate paths use a sentinel because ``None``/``""`` are valid
    explicit values (they mean "skip writing that artifact").
    """
    input_csv_dir = INPUT_CSV_DIR if input_csv_dir is None else input_csv_dir
    input_shp_dir = INPUT_SHP_DIR if input_shp_dir is None else input_shp_dir
    final_joined_features = (
        FINAL_JOINED_FEATURES if final_joined_features is None else final_joined_features
    )
    if intermediate_combined_csv is _UNSET:
        intermediate_combined_csv = INTERMEDIATE_COMBINED_CSV
    if intermediate_merged_shp is _UNSET:
        intermediate_merged_shp = INTERMEDIATE_MERGED_SHP

    found = False
    if str(input_csv_dir) == _DEFAULT_INPUT_CSV_DIR:
        logging.warning(
            "INPUT_CSV_DIR is still the placeholder value — update it or pass "
            "--input-csv-dir before running."
        )
        found = True
    if str(input_shp_dir) == _DEFAULT_INPUT_SHP_DIR:
        logging.warning(
            "INPUT_SHP_DIR is still the placeholder value — update it or pass "
            "--input-shp-dir before running."
        )
        found = True
    if final_joined_features == _DEFAULT_FINAL_JOINED_FEATURES:
        logging.warning(
            "FINAL_JOINED_FEATURES is still the placeholder value — update it or pass "
            "--output before running."
        )
        found = True
    # The two intermediate paths are optional — only flag them if non-empty AND still the default
    if (
        not _is_blank(intermediate_combined_csv)
        and intermediate_combined_csv == _DEFAULT_INTERMEDIATE_COMBINED_CSV
    ):
        logging.warning(
            "INTERMEDIATE_COMBINED_CSV is still the placeholder value — "
            "update it, or set it to '' to skip writing it."
        )
        found = True
    if (
        not _is_blank(intermediate_merged_shp)
        and intermediate_merged_shp == _DEFAULT_INTERMEDIATE_MERGED_SHP
    ):
        logging.warning(
            "INTERMEDIATE_MERGED_SHP is still the placeholder value — "
            "update it, or set it to '' to skip writing it."
        )
        found = True
    return found


def run(
    input_csv_dir: str | Path | None = None,
    input_shp_dir: str | Path | None = None,
    final_joined_features: str | None = None,
    tiger_input_glob: str | None = None,
    fips_to_filter: Sequence[str] | None = None,
    intermediate_combined_csv: str | None = None,
    intermediate_merged_shp: str | None = None,
) -> None:
    """Run the full three-stage pipeline.

    Unset args fall back to the CONFIGURATION block at the top of this file, so
    ``m.INPUT_CSV_DIR = ...; m.run()`` works after a plain import. Pass an empty
    string for an intermediate path to skip writing that artifact.
    """
    input_csv_dir = INPUT_CSV_DIR if input_csv_dir is None else input_csv_dir
    input_shp_dir = INPUT_SHP_DIR if input_shp_dir is None else input_shp_dir
    final_joined_features = (
        FINAL_JOINED_FEATURES if final_joined_features is None else final_joined_features
    )
    tiger_input_glob = TIGER_INPUT_GLOB if tiger_input_glob is None else tiger_input_glob
    fips_to_filter = list(FIPS_TO_FILTER if fips_to_filter is None else fips_to_filter)
    intermediate_combined_csv = (
        INTERMEDIATE_COMBINED_CSV
        if intermediate_combined_csv is None
        else intermediate_combined_csv
    )
    intermediate_merged_shp = (
        INTERMEDIATE_MERGED_SHP if intermediate_merged_shp is None else intermediate_merged_shp
    )

    # Optional intermediates left at their placeholder mean "don't write them",
    # not "abort": coerce to blank so a run wired with only --output (the
    # orchestrator's pattern) proceeds instead of bailing with "No processing".
    intermediate_combined_csv = _skip_placeholder_output(
        intermediate_combined_csv, _DEFAULT_INTERMEDIATE_COMBINED_CSV, "--intermediate-csv"
    )
    intermediate_merged_shp = _skip_placeholder_output(
        intermediate_merged_shp, _DEFAULT_INTERMEDIATE_MERGED_SHP, "--intermediate-shp"
    )

    if _check_placeholders(
        input_csv_dir,
        input_shp_dir,
        final_joined_features,
        intermediate_combined_csv,
        intermediate_merged_shp,
    ):
        logging.info("No processing performed. Update the configuration paths and re-run.")
        return

    try:
        # -------- Stage 1: CSV merge --------
        logging.info("Stage 1/3: discovering & merging Census CSVs under %s", input_csv_dir)
        discovered = discover_census_files(input_csv_dir)
        if discovered["POP_FILES"] or discovered["HH_FILES"]:
            logging.info(
                "Population and households are taken from the TIGER blocks' POP20/HOUSING20; "
                "any P1/H9 tables found are not required and are ignored."
            )
        tract_attrs = build_tract_attributes(
            income_files=discovered["INCOME_FILES"],
            ethnicity_files=discovered["ETHNICITY_FILES"],
            language_files=discovered["LANGUAGE_FILES"],
            vehicle_files=discovered["VEHICLE_FILES"],
            age_files=discovered["AGE_FILES"],
            commute_files=discovered["COMMUTE_FILES"],
            county_fips_filter=fips_to_filter,
        )
        block_jobs = build_block_jobs(discovered["JOBS_FILES"], county_fips_filter=fips_to_filter)
        logging.info(
            "Stage 1 produced %d tract attribute row(s) and %d block-job row(s).",
            len(tract_attrs),
            len(block_jobs),
        )

        if not _is_blank(intermediate_combined_csv):
            write_csv(tract_attrs, intermediate_combined_csv)

        # -------- Stage 2: TIGER merge + FIPS filter --------
        logging.info("Stage 2/3: discovering & merging TIGER shapefiles under %s", input_shp_dir)
        shp_paths = discover_tiger_datasets(input_shp_dir, tiger_input_glob, prefer="shp")
        blocks_gdf = merge_shapefiles(shp_paths)
        ensure_fips_column(blocks_gdf)
        blocks_gdf = filter_by_fips(blocks_gdf, fips_to_filter)

        if not _is_blank(intermediate_merged_shp):
            write_geo(blocks_gdf, intermediate_merged_shp)

        # -------- Stage 3: attach demographics onto geometry --------
        logging.info("Stage 3/3: attaching demographics onto block geometry")
        joined = attach_demographics_to_blocks(blocks_gdf, tract_attrs, block_jobs)
        write_geo(joined, final_joined_features)

        logging.info("Pipeline completed successfully.")
    except Exception:  # noqa: BLE001
        logging.exception("Pipeline failed")
        sys.exit(1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge Census CSVs and TIGER shapefiles and join them into a single "
            "block-level layer. Defaults come from the CONFIGURATION block at the "
            "top of this file; the CSV topic signatures and join keys stay in the "
            "config block."
        )
    )
    parser.add_argument(
        "--input-csv-dir", default=INPUT_CSV_DIR, help="Root folder of Census CSV downloads."
    )
    parser.add_argument(
        "--input-shp-dir", default=INPUT_SHP_DIR, help="Root folder of TIGER/Line shapefiles."
    )
    parser.add_argument(
        "--tiger-glob",
        default=TIGER_INPUT_GLOB,
        help="Glob matched against TIGER shapefile basenames.",
    )
    parser.add_argument(
        "--fips",
        nargs="*",
        default=FIPS_TO_FILTER,
        metavar="FIPS",
        help="5-digit county FIPS codes to keep (default: config; empty = all).",
    )
    parser.add_argument(
        "--output",
        default=FINAL_JOINED_FEATURES,
        help="Final joined geometry + attributes output path.",
    )
    parser.add_argument(
        "--intermediate-csv",
        default=INTERMEDIATE_COMBINED_CSV,
        help="Stage 1 combined CSV path (empty string to skip).",
    )
    parser.add_argument(
        "--intermediate-shp",
        default=INTERMEDIATE_MERGED_SHP,
        help="Stage 2 merged geometry path (empty string to skip).",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. Defaults fall back to the CONFIGURATION block."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run(
        input_csv_dir=args.input_csv_dir,
        input_shp_dir=args.input_shp_dir,
        final_joined_features=args.output,
        tiger_input_glob=args.tiger_glob,
        fips_to_filter=args.fips,
        intermediate_combined_csv=args.intermediate_csv,
        intermediate_merged_shp=args.intermediate_shp,
    )


def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":
    # In a notebook (pasted cell or %run), use the CONFIGURATION block instead
    # of argparse, which would otherwise try to parse the kernel's own argv.
    if _in_ipython():
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        run()
    else:
        main()
