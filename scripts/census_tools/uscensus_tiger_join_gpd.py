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
}

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# ---- Sentinel defaults — detect un-edited placeholder paths ----------------
_DEFAULT_INPUT_CSV_DIR: str = r"Path\To\Your\Census_Table_Data_Files"
_DEFAULT_INPUT_SHP_DIR: str = r"Path\To\Your\TIGER_Shapefiles"
_DEFAULT_INTERMEDIATE_COMBINED_CSV: str = r"Path\To\Your\Output_Folder\joined_blocks.csv"
_DEFAULT_INTERMEDIATE_MERGED_SHP: str = r"Path\To\Your\Output_Folder\va_md_dc_blocks_fips_merge.shp"
_DEFAULT_FINAL_JOINED_FEATURES: str = r"Path\To\Your\Output_Folder\va_md_dc_blocks_plus_data.shp"

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


def discover_census_files(
    root_dir: str | Path,
    signatures: Mapping[str, Sequence[str] | str] = TOPIC_SIGNATURES,
) -> dict[str, list[str]]:
    """Recursively locate Census data files and bucket them by topic.

    Accepts plain CSV, CSV.GZ, or ZIP archives. ZIPs are returned as the
    ZIP path itself; content is handled later by ``_read_csv_any``.
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

    for lst in buckets.values():
        lst.sort()
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


def _load_and_concat(
    files: Sequence[str],
    *,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    dtype: Mapping[Hashable, str | np.dtype[Any]] | None = None,
    usecols: Sequence[Hashable] | None = None,
    rename: Mapping[str, str] | None = None,
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] | None = None,
) -> pd.DataFrame:
    """Read multiple Census CSV / CSV-GZ / ZIP files and concatenate the results.

    Embedded control characters in NAME columns are stripped immediately to
    guarantee that every logical record remains on a single physical line
    when the final DataFrame is exported.

    Column renaming occurs *before* column pruning via ``usecols`` (unless
    ``usecols`` is explicitly supplied).
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

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    df["all_lo_veh_hh"] = df[["veh_0_all_hh", "veh_1_all_hh"]].sum(axis=1)
    df["perc_lo_veh"] = df["all_lo_veh_hh"] / df["all_hhs"]
    df["perc_0_veh"] = df["veh_0_all_hh"] / df["all_hhs"]
    df["perc_1_veh"] = df["veh_1_all_hh"] / df["all_hhs"]
    df["perc_veh_1_hh_1"] = df["veh_1_hh_1"] / df["all_hhs"]
    df["perc_lo_veh_mod"] = (df["perc_lo_veh"] - df["perc_veh_1_hh_1"]).round(3)
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


@dataclass(slots=True)
class _TractInputs:
    income_files: list[str]
    ethnicity_files: list[str]
    language_files: list[str]
    vehicle_files: list[str]
    age_files: list[str]


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
    county_fips_filter: Iterable[str] | None = None,
    _clean_columns: bool = True,
) -> pd.DataFrame:
    """Return a fully joined block + tract DataFrame with optional FIPS filter."""
    block_df = _build_block_df(_BlockInputs(pop_files, hh_files, jobs_files))
    tract_df = _build_tract_df(
        _TractInputs(
            income_files or [],
            ethnicity_files or [],
            language_files or [],
            vehicle_files or [],
            age_files or [],
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
        )
    )

    if _clean_columns:
        combined = _drop_unfriendly_cols(combined)

    combined = _apply_fips_filter_df(combined, fips=county_fips_filter)
    _fill_numeric_only(combined)
    return combined


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
        gdf[int_cols] = gdf[int_cols].astype("float64").round(1)


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


def write_geo(gdf: GeoDataFrame, out_path: str) -> None:
    """Write *gdf* to disk, creating parent dirs if needed.

    Shapefile outputs have field names truncated to 10 chars and the implicit
    pandas index suppressed; other drivers are inferred from the extension.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.lower().endswith(".shp"):
        gdf = _truncate_field_names(gdf)
        driver: str | None = "ESRI Shapefile"
    else:
        driver = None

    logging.info("Writing %d features → %s", len(gdf), path.resolve())
    gdf.to_file(out_path, driver=driver, index=False)


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


def _check_placeholders() -> bool:
    """Warn about un-edited placeholder paths. Return True if any are still set."""
    found = False
    if str(INPUT_CSV_DIR) == _DEFAULT_INPUT_CSV_DIR:
        logging.warning("INPUT_CSV_DIR is still the placeholder value — update it before running.")
        found = True
    if str(INPUT_SHP_DIR) == _DEFAULT_INPUT_SHP_DIR:
        logging.warning("INPUT_SHP_DIR is still the placeholder value — update it before running.")
        found = True
    if FINAL_JOINED_FEATURES == _DEFAULT_FINAL_JOINED_FEATURES:
        logging.warning(
            "FINAL_JOINED_FEATURES is still the placeholder value — update it before running."
        )
        found = True
    # The two intermediate paths are optional — only flag them if non-empty AND still the default
    if (
        not _is_blank(INTERMEDIATE_COMBINED_CSV)
        and INTERMEDIATE_COMBINED_CSV == _DEFAULT_INTERMEDIATE_COMBINED_CSV
    ):
        logging.warning(
            "INTERMEDIATE_COMBINED_CSV is still the placeholder value — "
            "update it, or set it to '' to skip writing it."
        )
        found = True
    if (
        not _is_blank(INTERMEDIATE_MERGED_SHP)
        and INTERMEDIATE_MERGED_SHP == _DEFAULT_INTERMEDIATE_MERGED_SHP
    ):
        logging.warning(
            "INTERMEDIATE_MERGED_SHP is still the placeholder value — "
            "update it, or set it to '' to skip writing it."
        )
        found = True
    return found


def main() -> None:
    """Run the full three-stage pipeline."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if _check_placeholders():
        logging.info("No processing performed. Update the configuration paths and re-run.")
        return

    try:
        # -------- Stage 1: CSV merge --------
        logging.info("Stage 1/3: discovering & merging Census CSVs under %s", INPUT_CSV_DIR)
        discovered = discover_census_files(INPUT_CSV_DIR)
        attrs_df = build_joined_table(
            pop_files=discovered["POP_FILES"],
            hh_files=discovered["HH_FILES"],
            jobs_files=discovered["JOBS_FILES"],
            income_files=discovered["INCOME_FILES"],
            ethnicity_files=discovered["ETHNICITY_FILES"],
            language_files=discovered["LANGUAGE_FILES"],
            vehicle_files=discovered["VEHICLE_FILES"],
            age_files=discovered["AGE_FILES"],
            county_fips_filter=FIPS_TO_FILTER,
        )
        logging.info("Stage 1 produced attribute table with shape %s", attrs_df.shape)

        if not _is_blank(INTERMEDIATE_COMBINED_CSV):
            write_csv(attrs_df, INTERMEDIATE_COMBINED_CSV)

        # -------- Stage 2: TIGER merge + FIPS filter --------
        logging.info("Stage 2/3: discovering & merging TIGER shapefiles under %s", INPUT_SHP_DIR)
        shp_paths = discover_tiger_datasets(INPUT_SHP_DIR, TIGER_INPUT_GLOB, prefer="shp")
        blocks_gdf = merge_shapefiles(shp_paths)
        ensure_fips_column(blocks_gdf)
        blocks_gdf = filter_by_fips(blocks_gdf, FIPS_TO_FILTER)

        if not _is_blank(INTERMEDIATE_MERGED_SHP):
            write_geo(blocks_gdf, INTERMEDIATE_MERGED_SHP)

        # -------- Stage 3: join attributes onto geometry --------
        logging.info("Stage 3/3: joining attributes onto block geometry")
        attrs_df = normalize_attribute_keys(attrs_df)
        joined = join_blocks_to_attributes(blocks_gdf, attrs_df)
        write_geo(joined, FINAL_JOINED_FEATURES)

        logging.info("Pipeline completed successfully.")
    except Exception:  # noqa: BLE001
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
