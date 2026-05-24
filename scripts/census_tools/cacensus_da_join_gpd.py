"""End-to-end open-source pipeline: pivot StatCan Census Profile CSVs, load DA boundaries, join.

Canadian Census equivalent of ``uscensus_tiger_join_gpd.py``.  Runs three stages
in one process, passing DataFrames between stages in memory — no intermediate
disk writes are required, though they can be enabled via configuration.

Stages
------
1) CSV stage (pandas):
        Discover StatCan Census Profile 2021 CSV(s) anywhere under
        ``INPUT_CSV_DIR``.  All three StatCan packaging types are supported:
        plain per-province ``.csv`` files, per-province ``.zip`` archives, and
        bundle zips that contain multiple per-province CSVs as members (e.g.
        ``98-401-X2021006_eng_CSV.zip``).  Only Dissemination Area rows are
        kept; the data is pivoted from long format to wide format using the
        ``CHARACTERISTIC_MAP``, and derived transit-planning indicators
        (youth, elderly, low income, visible minority, allophone) are
        computed.
2) DA stage (GeoPandas):
        Discover and load the StatCan DA cartographic boundary shapefile
        (``lda_000b21a_e.shp`` or its zipped form) from ``INPUT_SHP_DIR``.
        An optional census-division (CDUID) filter is applied.
3) Join stage (GeoPandas):
        Merge DA-level attributes onto DA geometry on DAUID and write the
        final output.

Configuration
-------------
At minimum, set ``INPUT_CSV_DIR``, ``INPUT_SHP_DIR``, and
``FINAL_JOINED_FEATURES``.  Set ``INTERMEDIATE_COMBINED_CSV`` or
``INTERMEDIATE_DA_SHP`` to an empty string to skip writing those intermediate
artifacts.

Notes:
    The Census Profile is long-format (one row per DA × characteristic),
    unlike US Census wide-format tables.  Stage 1 pivots so each DAUID
    becomes one row.

    ``C1_COUNT_TOTAL`` cells suppressed by StatCan (quality symbols ``x``,
    ``F``, ``...``) are coerced to ``NaN`` and filled with ``0`` after the
    pivot.

    Census Profile CSVs use Latin-1 encoding; accented French place names
    survive end-to-end.

    Shapefile outputs truncate column names to 10 chars; use a GeoPackage
    (``.gpkg``) to preserve full names.

Helpful links
-------------
    Census Profile 2021:  https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/
    DA boundary file:     https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/
"""

from __future__ import annotations

import io
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, List, Literal, Sequence

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Stage 1: CSV inputs ----------------------------------------------------
#: Root folder holding Census Profile 2021 CSVs, per-province zips, or a
#: bundle zip.  Sub-directories are searched automatically.
INPUT_CSV_DIR: str | Path = r"Path\To\Your\CensusProfile_CSV_Dir"  # <<< EDIT ME

# ---- Stage 2: DA boundary inputs --------------------------------------------
#: Root folder containing the StatCan DA cartographic boundary file
#: (``lda_000b21a_e.shp`` or ``.zip``).
INPUT_SHP_DIR: str | Path = r"Path\To\Your\DA_Boundary_Dir"  # <<< EDIT ME

#: Glob pattern matched against the DA shapefile (and matching .zip) basenames.
DA_SHP_GLOB: str = "lda_*.shp"

# ---- Shared: optional CDUID filter ------------------------------------------
#: 4-digit census-division UIDs (Canadian analog of county FIPS).  Applied in
#: both Stage 1 and Stage 2.  Leave empty ([]) to export everything.
CDUIDS_TO_FILTER: List[str] = [
    "3506",  # Ottawa (Ontario)
    "2481",  # Gatineau (Quebec)
    "2482",  # Les Collines-de-l'Outaouais (Quebec)
]

# ---- Outputs ----------------------------------------------------------------
#: Intermediate combined CSV from Stage 1. Set to "" or None to skip.
INTERMEDIATE_COMBINED_CSV: str | None = r"Path\To\Your\Output\da_attributes.csv"

#: Intermediate DA geometry from Stage 2. Set to "" or None to skip.
INTERMEDIATE_DA_SHP: str | None = None

#: Final joined geometry + attributes (Stage 3 output).
FINAL_JOINED_FEATURES: str = r"Path\To\Your\Output\da_joined.gpkg"

# ---- Join settings ----------------------------------------------------------
JOIN_KEY: Final[str] = "DAUID"
FORCE_FLOAT: Final[bool] = True
MAX_FIELD_LEN: Final[int] = 10

# ---- Characteristic ID → column name map ------------------------------------
#: Maps ``CHARACTERISTIC_ID`` strings from the Census Profile long-format CSV
#: to friendly column names in the pivoted wide-format output.
CHARACTERISTIC_MAP: dict[str, str] = {
    # ---- Population & housing (100% data) ----
    "1": "total_pop",
    "4": "total_dwell",
    "5": "occ_dwell",
    # ---- Age groups (100% data) ----
    "8": "age_total",
    "9": "age_0_14",
    "13": "age_15_64",
    "14": "age_15_19",
    "15": "age_20_24",
    "24": "age_65_plus",
    "25": "age_65_69",
    "26": "age_70_74",
    "27": "age_75_79",
    "28": "age_80_84",
    "29": "age_85_plus",
    # ---- Official language knowledge (100% data) ----
    "383": "lang_total",
    "384": "lang_eng_only",
    "385": "lang_fr_only",
    "386": "lang_eng_fr",
    "387": "lang_neither",
    # ---- Household total income groups, 2020 (100% data) ----
    "260": "hh_inc_total",
    "261": "hh_u5k",
    "262": "hh_5_10k",
    "263": "hh_10_15k",
    "264": "hh_15_20k",
    "265": "hh_20_25k",
    "266": "hh_25_30k",
    "267": "hh_30_35k",
    "268": "hh_35_40k",
    "269": "hh_40_45k",
    "270": "hh_45_50k",
    "271": "hh_50_60k",
    # ---- LIM-AT low-income status, 2020 (100% data) ----
    "335": "lim_total",
    "340": "lim_count",
    "345": "lim_pct",
    # ---- Visible minority (25% sample data) ----
    "1683": "vm_denom",
    "1684": "vm_count",
    "1697": "vm_not",
}

LOG_LEVEL: int = logging.INFO

# ---- Sentinel defaults — detect un-edited placeholder paths ----------------
_DEFAULT_INPUT_CSV_DIR: str = r"Path\To\Your\CensusProfile_CSV_Dir"
_DEFAULT_INPUT_SHP_DIR: str = r"Path\To\Your\DA_Boundary_Dir"
_DEFAULT_FINAL_JOINED_FEATURES: str = r"Path\To\Your\Output\da_joined.gpkg"
_DEFAULT_INTERMEDIATE_COMBINED_CSV: str = r"Path\To\Your\Output\da_attributes.csv"

# =============================================================================
# STAGE 1: CSV DISCOVERY & PIVOT  (pandas)
# =============================================================================

DA_GEO_LEVEL: Final[str] = "Dissemination area"
CENSUS_ENCODING: Final[str] = "latin-1"
DAUID_COL: Final[str] = "DAUID"

#: Regex that matches StatCan Census Profile 2021 per-province data filenames.
#: Handles both English and French variants and is case-insensitive for the
#: extension so ``.CSV`` and ``.csv`` both match.
_CENSUS_CSV_RE: Final[re.Pattern[str]] = re.compile(
    r"^98-401-X\d{7}_(English|French)_CSV_data_(.+)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _CsvSource:
    """Reference to one Census Profile data CSV.

    The underlying file may be a direct ``.csv`` on disk (``inner`` is
    ``None``) or a member inside a zip archive (``inner`` holds the member
    name within ``container``).
    """

    container: Path
    inner: str | None = None

    @property
    def name(self) -> str:
        """Filename of the underlying CSV."""
        return Path(self.inner).name if self.inner is not None else self.container.name

    @property
    def display(self) -> str:
        """Human-readable label for log output."""
        if self.inner is not None:
            return f"{self.container.name}!{Path(self.inner).name}"
        return self.container.name


def discover_census_profile_csvs(root_dir: str | Path) -> list[_CsvSource]:
    """Recursively find StatCan Census Profile 2021 CSVs under *root_dir*.

    Handles three StatCan packaging types:

    * Direct ``.csv`` files whose name matches ``_CENSUS_CSV_RE``.
    * Per-province ``.zip`` archives whose filename (after stripping ``.zip``)
      matches ``_CENSUS_CSV_RE``.
    * Bundle zips (e.g. ``98-401-X2021006_eng_CSV.zip``) whose filename does
      *not* match the pattern but which contain per-province CSV members.

    If the same data-CSV name appears as both a direct file on disk and a zip
    member, the direct file wins (no extraction needed at read time).
    File order within each pass is sorted for determinism.
    """
    root = Path(root_dir).expanduser().resolve()
    sources_by_name: dict[str, _CsvSource] = {}

    # Pass 1: plain .csv files.
    for path in sorted(root.rglob("*.csv")):
        if _CENSUS_CSV_RE.match(path.name):
            sources_by_name[path.name] = _CsvSource(container=path)

    # Pass 2: members inside zip archives (both per-province and bundle zips).
    for path in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(path) as zf:
                members = zf.namelist()
        except zipfile.BadZipFile:
            logging.warning("Skipping bad zip: %s", path.name)
            continue
        for member in members:
            if member.endswith("/"):
                continue
            member_name = Path(member).name
            if not _CENSUS_CSV_RE.match(member_name):
                continue
            if member_name in sources_by_name:
                # Direct file already discovered; prefer it.
                continue
            sources_by_name[member_name] = _CsvSource(container=path, inner=member)

    sources = sorted(sources_by_name.values(), key=lambda s: s.name)
    if sources:
        logging.info("Discovered %d Census Profile CSV source(s)", len(sources))
    else:
        logging.warning("No Census Profile CSVs found under %s", root)
    return sources


def _read_profile_csv(source: _CsvSource) -> DataFrame:
    """Read one Census Profile CSV, returning only Dissemination Area rows.

    Columns retained: ``ALT_GEO_CODE`` (the DAUID), ``GEO_NAME``,
    ``CHARACTERISTIC_ID``, and ``C1_COUNT_TOTAL``.  All columns are read as
    strings; numeric coercion happens downstream in ``_load_and_pivot``.
    """
    logging.info("Reading %s", source.display)

    read_kwargs: dict = dict(
        dtype=str,
        encoding=CENSUS_ENCODING,
        usecols=["ALT_GEO_CODE", "GEO_LEVEL", "GEO_NAME", "CHARACTERISTIC_ID", "C1_COUNT_TOTAL"],
        low_memory=False,
    )

    if source.inner is None:
        df = pd.read_csv(source.container, **read_kwargs)
    else:
        zip_kwargs = {k: v for k, v in read_kwargs.items() if k != "encoding"}
        with zipfile.ZipFile(source.container) as zf:
            with zf.open(source.inner) as fh:
                with io.TextIOWrapper(fh, encoding=CENSUS_ENCODING) as txt:
                    df = pd.read_csv(txt, **zip_kwargs)

    da_rows = df[df["GEO_LEVEL"] == DA_GEO_LEVEL].copy()
    logging.info("  → %d DA rows from %s", len(da_rows), source.display)
    return da_rows.drop(columns=["GEO_LEVEL"])


def _fill_numeric_only(df: DataFrame, value: int | float = 0) -> DataFrame:
    """Replace only numeric NaNs with *value*; leave object columns untouched."""
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(value)
    return df


def _load_and_pivot(
    sources: Sequence[_CsvSource],
    characteristic_map: dict[str, str] = CHARACTERISTIC_MAP,
) -> DataFrame:
    """Read all *sources*, concatenate DA rows, and pivot to wide format.

    Steps:
    1. Concatenate all DA-level rows from every source.
    2. De-duplicate on ``(ALT_GEO_CODE, CHARACTERISTIC_ID)`` — keeps the
       first occurrence so bundle + per-province overlap is handled cleanly.
    3. Filter to only the ``CHARACTERISTIC_ID`` values in *characteristic_map*.
    4. Coerce ``C1_COUNT_TOTAL`` to float64 (suppression symbols → NaN).
    5. Pivot: rows = DAUID, columns = CHARACTERISTIC_ID.
    6. Rename columns using *characteristic_map*.
    7. Fill numeric NaNs with 0; preserve ``GEO_NAME`` as a string label.

    Returns an empty DataFrame if *sources* is empty.
    """
    if not sources:
        return DataFrame()

    frames: list[DataFrame] = []
    for src in sources:
        try:
            frames.append(_read_profile_csv(src))
        except Exception:  # noqa: BLE001
            logging.exception("Could not read %s — skipping", src.display)

    if not frames:
        return DataFrame()

    long_df = pd.concat(frames, ignore_index=True)

    # De-duplicate: keep first occurrence per (DAUID, characteristic).
    long_df = long_df.drop_duplicates(subset=["ALT_GEO_CODE", "CHARACTERISTIC_ID"], keep="first")

    # Retain GEO_NAME for the first row of each DAUID (carried into pivot separately).
    geo_name = long_df.groupby("ALT_GEO_CODE", sort=False)["GEO_NAME"].first()

    # Filter to the characteristics we care about.
    wanted = set(characteristic_map.keys())
    long_df = long_df[long_df["CHARACTERISTIC_ID"].isin(wanted)].copy()

    # Coerce values (suppression symbols like "x", "F", "..." → NaN).
    long_df["value"] = pd.to_numeric(long_df["C1_COUNT_TOTAL"], errors="coerce")

    # Pivot to wide format. dropna=False keeps columns where all DAs are
    # suppressed so _fill_numeric_only can later replace those NaNs with 0.
    wide = long_df.pivot_table(
        index="ALT_GEO_CODE",
        columns="CHARACTERISTIC_ID",
        values="value",
        aggfunc="first",
        dropna=False,
    )
    wide.index.name = DAUID_COL
    wide = wide.reset_index()

    # Rename characteristic columns.
    char_rename = {k: v for k, v in characteristic_map.items() if k in wide.columns}
    wide = wide.rename(columns=char_rename)

    # Re-attach GEO_NAME.
    name_df = geo_name.rename("GEO_NAME").reset_index().rename(columns={"ALT_GEO_CODE": DAUID_COL})
    wide = wide.merge(name_df, on=DAUID_COL, how="left")

    _fill_numeric_only(wide)
    n_rows, n_cols = wide.shape
    logging.info("Pivoted %d DA rows → %d features, %d columns", len(long_df), n_rows, n_cols)
    return wide


# ---- Derivation functions ---------------------------------------------------


def _derive_income(df: DataFrame) -> DataFrame:
    """Add low-income household count and percentage columns.

    Low-income households are defined as household total income under $35,000
    (sum of income bands ``hh_u5k`` through ``hh_30_35k``), as a Canadian
    analog to the US B19001 low-income threshold.  ``lim_pct`` from the
    Census Profile (LIM-AT prevalence) is preserved alongside as the official
    Statistics Canada low-income measure.
    """
    low_bands = [
        "hh_u5k",
        "hh_5_10k",
        "hh_10_15k",
        "hh_15_20k",
        "hh_20_25k",
        "hh_25_30k",
        "hh_30_35k",
    ]
    present = [c for c in low_bands if c in df.columns]
    df["low_inc_hh"] = df[present].sum(axis=1)
    if "hh_inc_total" in df.columns:
        df["perc_low_inc"] = (df["low_inc_hh"] / df["hh_inc_total"]).round(3)
    return df


def _derive_visible_minority(df: DataFrame) -> DataFrame:
    """Add visible-minority percentage (Canadian analog to US perc_minority)."""
    if "vm_count" in df.columns and "vm_denom" in df.columns:
        df["perc_vm"] = (df["vm_count"] / df["vm_denom"]).round(3).fillna(0)
    return df


def _derive_language(df: DataFrame) -> DataFrame:
    """Add percentage with no knowledge of official languages (allophone proxy).

    This is the Canadian analog of LEP (Limited English Proficiency) from the
    US pipeline: ``perc_allophone`` captures the share of the DA population
    that reported knowing neither English nor French.
    """
    if "lang_neither" in df.columns and "lang_total" in df.columns:
        df["perc_allophone"] = (df["lang_neither"] / df["lang_total"]).round(3).fillna(0)
    return df


def _derive_age(df: DataFrame) -> DataFrame:
    """Add youth (15–24) and elderly (65+) aggregates and percentages."""
    youth_cols = [c for c in ("age_15_19", "age_20_24") if c in df.columns]
    df["all_youth"] = df[youth_cols].sum(axis=1)

    if "age_65_plus" in df.columns:
        df["all_elderly"] = df["age_65_plus"]
    else:
        _all_elderly_bands = ("age_65_69", "age_70_74", "age_75_79", "age_80_84", "age_85_plus")
        elderly_cols = [c for c in _all_elderly_bands if c in df.columns]
        df["all_elderly"] = df[elderly_cols].sum(axis=1)

    if "total_pop" in df.columns:
        df["perc_youth"] = (df["all_youth"] / df["total_pop"]).round(3).fillna(0)
        df["perc_elderly"] = (df["all_elderly"] / df["total_pop"]).round(3).fillna(0)
    return df


# ---- CDUID filter helpers ---------------------------------------------------


def _ensure_cduid_column_df(
    df: DataFrame,
    *,
    dst: str = "CDUID",
    src: str = DAUID_COL,
) -> None:
    """Create a 4-digit census-division (CDUID) column in place from DAUID."""
    if dst in df.columns:
        return
    if src not in df.columns:
        raise KeyError(f"Source column '{src}' not found; cannot derive '{dst}'.")
    df[dst] = df[src].astype(str).str[:4]


def _apply_cduid_filter_df(
    df: DataFrame,
    *,
    cduids: Iterable[str] | None = None,
    dst_col: str = "CDUID",
) -> DataFrame:
    """Return a copy filtered to *cduids* (or unchanged if *cduids* is empty/None)."""
    if not cduids:
        return df
    _ensure_cduid_column_df(df, dst=dst_col)
    wanted = {str(c).zfill(4) for c in cduids}
    return df[df[dst_col].isin(wanted)].copy()


# ---- Stage 1 public entry point ---------------------------------------------


def build_da_table(
    sources: Sequence[_CsvSource],
    *,
    characteristic_map: dict[str, str] = CHARACTERISTIC_MAP,
    cduid_filter: Iterable[str] | None = None,
) -> DataFrame:
    """Return a wide-format DA attribute DataFrame, ready for joining.

    Applies the full pipeline: pivot → derive indicators → optional CDUID
    filter → fill remaining NaNs.
    """
    df = _load_and_pivot(sources, characteristic_map)
    if df.empty:
        return df

    df = _derive_income(df)
    df = _derive_visible_minority(df)
    df = _derive_language(df)
    df = _derive_age(df)
    df = _apply_cduid_filter_df(df, cduids=cduid_filter)
    _fill_numeric_only(df)
    return df


# =============================================================================
# STAGE 2: DA BOUNDARY DISCOVERY & LOAD  (GeoPandas)
# =============================================================================


def discover_da_shapefile(
    root_dir: str | Path,
    pattern: str = DA_SHP_GLOB,
    *,
    prefer: str = "shp",
) -> str:
    """Return the path to the StatCan DA boundary file (plain or zipped).

    Search is recursive. When both a ``.shp`` and a ``.zip`` with the same
    stem exist, *prefer* controls which is returned (``"shp"`` keeps the
    plain file; ``"zip"`` keeps the archive). Zipped files are returned with
    the ``zip://`` VFS prefix so GeoPandas can open them directly.

    Raises:
        NotADirectoryError: If *root_dir* does not exist.
        FileNotFoundError: If no matching file is found.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a valid directory")

    shp_paths = list(root.rglob(pattern))
    zip_glob = re.sub(r"\.shp$", ".zip", pattern, flags=re.IGNORECASE)
    zip_paths = list(root.rglob(zip_glob))

    candidates: dict[str, Path] = {}
    for p in sorted(shp_paths + zip_paths):
        stem = p.stem
        ext = p.suffix.lower()
        if stem in candidates:
            keep_zip = prefer == "zip"
            if keep_zip and ext == ".zip":
                candidates[stem] = p
            elif not keep_zip and ext == ".shp":
                candidates[stem] = p
        else:
            candidates[stem] = p

    if not candidates:
        raise FileNotFoundError(f"No DA boundary matching '{pattern}' found under {root}")

    # Return a single file (the first by stem alphabetically).
    chosen = sorted(candidates.values())[0]
    logging.info("Discovered DA boundary: %s", chosen.name)
    path_str = f"zip://{chosen}" if chosen.suffix.lower() == ".zip" else str(chosen)
    return path_str


def load_da_shapefile(path: str) -> GeoDataFrame:
    """Load the DA boundary shapefile and normalize key columns.

    Coerces ``DAUID``, ``DGUID``, and ``PRUID`` to string; ensures
    ``LANDAREA`` is float; derives ``CDUID`` from the first four characters
    of ``DAUID`` for downstream filtering.
    """
    logging.info("Reading DA boundary: %s", path)
    gdf = gpd.read_file(path)

    for col in ("DAUID", "DGUID", "PRUID"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str)

    if "LANDAREA" in gdf.columns:
        gdf["LANDAREA"] = pd.to_numeric(gdf["LANDAREA"], errors="coerce")

    if "CDUID" not in gdf.columns and "DAUID" in gdf.columns:
        gdf["CDUID"] = gdf["DAUID"].str[:4]

    logging.info("Loaded %d DA features", len(gdf))
    return gdf


def filter_da_by_cduid(
    gdf: GeoDataFrame,
    cduids: Sequence[str],
    *,
    cduid_col: str = "CDUID",
) -> GeoDataFrame:
    """Return a view of *gdf* containing only requested CDUIDs.

    Canadian analog of ``filter_by_fips`` from the US pipeline.
    """
    if not cduids:
        logging.info("CDUID filter empty — exporting full DA layer")
        return gdf

    if cduid_col not in gdf.columns:
        if "DAUID" in gdf.columns:
            gdf = gdf.copy()
            gdf[cduid_col] = gdf["DAUID"].str[:4]
        else:
            raise KeyError(f"Column '{cduid_col}' not found and DAUID unavailable.")

    wanted = {str(c).zfill(4) for c in cduids}
    selected = gdf.loc[gdf[cduid_col].isin(wanted)].copy()
    if selected.empty:
        logging.warning("No DA features matched the CDUID list — output will be empty")
    else:
        logging.info("Selected %d of %d DA features", len(selected), len(gdf))
    return selected


# =============================================================================
# STAGE 3: JOIN ATTRIBUTES → GEOMETRY  (GeoPandas)
# =============================================================================


def join_das_to_attributes(
    da_gdf: GeoDataFrame,
    attrs: DataFrame,
    *,
    left_key: str = JOIN_KEY,
    right_key: str = JOIN_KEY,
    how: Literal["left", "right", "outer", "inner", "cross"] = "left",
) -> GeoDataFrame:
    """Merge *attrs* onto *da_gdf* on DAUID.

    Both key columns are coerced to ``str`` before joining to prevent type
    mismatches.  Raises ``ValueError`` if duplicate keys would violate a
    1:1 expectation.
    """
    da_gdf = da_gdf.copy()
    da_gdf[left_key] = da_gdf[left_key].astype(str)
    attrs = attrs.copy()
    attrs[right_key] = attrs[right_key].astype(str)

    logging.info("Merging DA geometry (%d) with attributes (%d) …", len(da_gdf), len(attrs))
    merged: GeoDataFrame = da_gdf.merge(
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
    """Convert nullable Int64 columns to float64 in place for shapefile safety."""
    int_cols: list[str] = [
        str(col) for col, dtype in gdf.dtypes.items() if pd.api.types.is_integer_dtype(dtype)
    ]
    if int_cols:
        logging.debug("Casting %d Int64 column(s) → float64", len(int_cols))
        gdf[int_cols] = gdf[int_cols].astype("float64")


def _truncate_field_names(gdf: GeoDataFrame, max_len: int = MAX_FIELD_LEN) -> GeoDataFrame:
    """Truncate attribute names to fit the Shapefile 10-char DBF limit.

    Collisions after truncation are resolved by appending a numeric suffix.
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
            "Truncated %d column name(s) to %d chars: %s", len(renames), max_len, renames
        )
        gdf = gdf.rename(columns=renames)
    return gdf


def _shp_schema(gdf: GeoDataFrame) -> dict:
    """Build a fiona write schema capping float fields to 1 decimal place."""
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
            max_len = int(gdf[col].astype(str).str.len().max()) if not gdf.empty else 1
            props[col_str] = f"str:{max(max_len, 1)}"
    return {"geometry": geom_type, "properties": props}


def write_geo(gdf: GeoDataFrame, out_path: str) -> None:
    """Write *gdf* to disk, creating parent dirs if needed.

    Shapefile outputs have field names truncated to 10 chars and float
    columns written with 1 decimal place.  GeoPackage and other drivers are
    inferred from the extension.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing %d features → %s", len(gdf), path.resolve())
    if out_path.lower().endswith(".shp"):
        gdf = _truncate_field_names(gdf)
        gdf.to_file(
            out_path, driver="ESRI Shapefile", schema=_shp_schema(gdf), engine="fiona", index=False
        )
    else:
        gdf.to_file(out_path, index=False)


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
    """Treat None and empty/whitespace strings as 'skip this output'."""
    return p is None or not str(p).strip()


def _check_placeholders() -> bool:
    """Warn about un-edited placeholder paths. Return True if any remain."""
    found = False
    if str(INPUT_CSV_DIR) == _DEFAULT_INPUT_CSV_DIR:
        logging.warning("INPUT_CSV_DIR is still the placeholder value — update before running.")
        found = True
    if str(INPUT_SHP_DIR) == _DEFAULT_INPUT_SHP_DIR:
        logging.warning("INPUT_SHP_DIR is still the placeholder value — update before running.")
        found = True
    if FINAL_JOINED_FEATURES == _DEFAULT_FINAL_JOINED_FEATURES:
        logging.warning(
            "FINAL_JOINED_FEATURES is still the placeholder value — update before running."
        )
        found = True
    if (
        not _is_blank(INTERMEDIATE_COMBINED_CSV)
        and INTERMEDIATE_COMBINED_CSV == _DEFAULT_INTERMEDIATE_COMBINED_CSV
    ):
        logging.warning(
            "INTERMEDIATE_COMBINED_CSV is still the placeholder value — "
            "update it, or set it to '' to skip writing it."
        )
        found = True
    return found


def main() -> None:
    """Run the full three-stage Canadian Census DA pipeline."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if _check_placeholders():
        logging.info("No processing performed. Update the configuration paths and re-run.")
        return

    try:
        # -------- Stage 1: discover CSVs, pivot, derive indicators --------
        logging.info(
            "Stage 1/3: discovering & pivoting Census Profile CSVs under %s", INPUT_CSV_DIR
        )
        sources = discover_census_profile_csvs(INPUT_CSV_DIR)
        if not sources:
            logging.error("No Census Profile CSVs found — aborting.")
            sys.exit(1)

        attrs_df = build_da_table(sources, cduid_filter=CDUIDS_TO_FILTER or None)
        logging.info("Stage 1 produced attribute table with shape %s", attrs_df.shape)

        if not _is_blank(INTERMEDIATE_COMBINED_CSV):
            write_csv(attrs_df, INTERMEDIATE_COMBINED_CSV)

        # -------- Stage 2: load DA boundary + CDUID filter --------
        logging.info("Stage 2/3: loading DA boundary shapefile from %s", INPUT_SHP_DIR)
        da_path = discover_da_shapefile(INPUT_SHP_DIR, DA_SHP_GLOB)
        da_gdf = load_da_shapefile(da_path)
        da_gdf = filter_da_by_cduid(da_gdf, CDUIDS_TO_FILTER)

        if not _is_blank(INTERMEDIATE_DA_SHP):
            write_geo(da_gdf, INTERMEDIATE_DA_SHP)

        # -------- Stage 3: join attributes onto DA geometry --------
        logging.info("Stage 3/3: joining attributes onto DA geometry")
        joined = join_das_to_attributes(da_gdf, attrs_df)
        write_geo(joined, FINAL_JOINED_FEATURES)

        logging.info("Pipeline completed successfully.")
    except Exception:  # noqa: BLE001
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
