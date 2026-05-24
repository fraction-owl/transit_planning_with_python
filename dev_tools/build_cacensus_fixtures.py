"""Generate the Canadian Census fixture set: StatCan DA boundary + Census Profile tables.

Edit the paths in the CONFIG block below and run from the repo root.

Workflow
--------

1. Discovers the StatCan dissemination area boundary shapefile
   (``lda_000b21a_e.shp`` or its zipped form) and the Census Profile 2021
   per-province CSV bundles (``98-401-X2021006_English_CSV_data_<Province>.csv``)
   anywhere under ``INPUT_DIR`` (sub-directories searched recursively).

2. Scans each Census Profile CSV (latin-1 encoded) to build a
   ``DAUID -> population_2021`` lookup. Only the population rows
   (``GEO_LEVEL == 'Dissemination area'`` and ``CHARACTERISTIC_ID == '1'``)
   are loaded, so memory cost is bounded even on the full national release.

3. Performs a stratified sample of DAs per (province, census division):

   * highest-population DA (residential density edge case)
   * zero-population DA, if any (uninhabited / industrial / vacant edge case)
   * highest-LANDAREA DA (rural mega-DA edge case -- captures the long tail
     where DA areas span orders of magnitude even though their populations
     are bounded to roughly 400-700)
   * remainder filled with a proportional random draw, clamped per CD to
     ``[MIN_PER_CD, MAX_PER_CD]``.

4. Writes a single zipped sample shapefile bundling all selected DAs, plus
   ``sample_dauids.csv`` (the manifest, which carries DAUID, PRUID, CDUID,
   selection reason, population, and land area for every picked DA).

5. Filters every Census Profile CSV under ``INPUT_DIR`` down to the selected
   DAUIDs. Country-level rows are preserved as a sanity-check anchor. CSD and
   higher rows are dropped (a DA's parent CSD is not derivable from its DAUID
   without an external relationship file -- documented as an extension point).

Modes
-----

* First run (no manifest in ``OUTPUT_DIR``): generate everything from scratch.
* Replay (manifest present): reuse the manifest's DAUID list, skip the CSV
  scan and sampling, just write the shapefile and re-filter the CSVs. Delete
  the manifest to re-sample.

``OUTPUT_DIR`` may be nested inside ``INPUT_DIR``; the script skips anything
under ``OUTPUT_DIR`` during input discovery so re-runs are idempotent.

Notes on extensions
-------------------

* No LEHD-equivalent jobs stratum: Statistics Canada does not publish an open
  small-area employment-by-workplace dataset comparable to LEHD WAC. The
  ``max_jobs`` stratum from the US fixture builder is intentionally omitted;
  a future version could derive a proxy from CHARACTERISTIC_ID rows for
  employed labour force or place-of-work counts.
* CSD/CD rows are dropped from filtered CSVs. To preserve them, you'd need the
  StatCan Dissemination Geographies Relationship File to map DAUIDs to their
  parent CSDUIDs (the relationship is not encoded in the DAUID itself).
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd
import polars as pl

# =============================================================================
# CONFIG
# =============================================================================

#: Root directory holding all fixture inputs: the DA boundary shapefile (.shp
#: or .zip) and Census Profile per-province CSVs. Sub-directories are searched
#: automatically; anything under ``OUTPUT_DIR`` is skipped so re-runs don't
#: reprocess their own output.
INPUT_DIR: Final[Path] = Path(r"PATH\TO\CANADA\FIXTURE\INPUTS")  # <<< EDIT ME
 
#: Directory where the sample shapefile, the manifest, and all filtered
#: fixtures are written. May be nested inside ``INPUT_DIR``.
OUTPUT_DIR: Final[Path] = Path(r"PATH\TO\CANADA\FIXTURE\OUTPUT")  # <<< EDIT ME

#: Map from 2-char province ID (PRUID) to the list of 2-char census division
#: codes (the last two chars of CDUID) to sample. The default selects the
#: core Toronto-area CDs; comment in/out lines to retarget. CDs are the
#: closest Canadian analog to US counties for sampling purposes.
CDS_BY_PROVINCE: Final[dict[str, list[str]]] = {
    "35": [  # Ontario
        "06",  # Ottawa (single-tier city CD, CDUID 3506)
    ],
    "24": [  # Quebec  
        "81",  # Gatineau (CDUID 2481)
        "82",  # Les Collines-de-l'Outaouais (CDUID 2482)
    ],
}

# ---- File patterns ----------------------------------------------------------

#: Regex for discovering the StatCan DA boundary file. The standard filenames
#: are ``lda_000b21a_e`` (English cartographic) and ``lda_000a21a_e`` (English
#: digital); accept either, plus zipped variants. The "lda" prefix means
#: "limits / dissemination areas" (note: "ldb_" is the dissemination-block
#: equivalent if you ever want finer geography).
DA_SHAPEFILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^lda_\d{3}[ab]\d{2}[a-z]_[ef]\.(shp|zip)$"
)

#: Regex for discovering Census Profile 2021 data files. Filename convention
#: is ``98-401-X2021006_(English|French)_CSV_data_<Province>.csv``; StatCan
#: also ships these as zip archives with the same naming and a ``.zip``
#: suffix. Extension match is case-insensitive because StatCan distributions
#: mix ``.csv`` / ``.CSV``.
CENSUS_CSV_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^98-401-X2021006_(English|French)_CSV_data_(.+)\.(?:csv|zip)$",
    re.IGNORECASE,
)

#: Map from the province-token in a Census Profile CSV filename to the set of
#: PRUIDs the file covers. Some StatCan bundles span multiple provinces, so
#: this is not a one-to-one mapping. Used to skip irrelevant CSVs at discovery
#: time when ``CDS_BY_PROVINCE`` only targets a subset of the country.
#: Keys correspond to English filename tokens; add French equivalents
#: (``ColombieBritannique``, etc.) if working with the French CSV variants.
CENSUS_CSV_TO_PRUIDS: Final[dict[str, frozenset[str]]] = {
    "Atlantic": frozenset({"10", "11", "12", "13"}),  # NL, PE, NS, NB
    "Quebec": frozenset({"24"}),
    "Ontario": frozenset({"35"}),
    "Prairies": frozenset({"46", "47", "48"}),  # MB, SK, AB
    "BritishColumbia": frozenset({"59"}),
    "Territories": frozenset({"60", "61", "62"}),  # YT, NT, NU
}

OUTPUT_DA_SHAPEFILE: Final[str] = "lda_000b21a_e_sample.shp"
MANIFEST_FILENAME: Final[str] = "sample_dauids.csv"

# ---- Sampling parameters ----------------------------------------------------

TARGET_TOTAL: Final[int] = 120
MIN_PER_CD: Final[int] = 5
MAX_PER_CD: Final[int] = 15
RANDOM_SEED: Final[int] = 42

# ---- CSV / characteristic identifiers --------------------------------------

#: Statistics Canada Census Profile CSVs are Windows-1252 / Latin-1, NOT UTF-8.
#: French place names with accents (and the bilingual variable labels) will
#: silently mangle if you read them as UTF-8 -- the file may even fail to open
#: depending on which byte trips the decoder first.
CENSUS_ENCODING: Final[str] = "latin-1"

#: CHARACTERISTIC_ID for "Population, 2021" -- the single characteristic we
#: need for sampling stratification.
POPULATION_CHARACTERISTIC_ID: Final[str] = "1"

#: GEO_LEVEL values used for filtering. The bundle 98-401-X2021006 mixes
#: Country, CSD, and DA rows in one file; we sample on DA rows and keep
#: Country rows as a sanity-check anchor in filtered output.
DA_GEO_LEVEL: Final[str] = "Dissemination area"
COUNTRY_GEO_LEVEL: Final[str] = "Country"

# =============================================================================
# HELPERS
# =============================================================================


def _is_under(child: Path, parent: Path) -> bool:
    """Return True if *child* resolves under *parent* (or equals *parent*)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _cduid_from_dauid(dauid: str) -> str:
    """Return the 4-char CDUID embedded in the first 4 chars of a DAUID."""
    return dauid[:4]


@dataclass(frozen=True)
class CensusCsvSource:
    """Reference to a Census Profile data CSV.

    A source is either an on-disk CSV file (``inner`` is ``None``) or a member
    inside a zip archive (``inner`` is the member name within ``container``).
    The distinction lets ``discover_census_csvs`` enumerate both per-province
    CSVs and CSVs nested inside a StatCan bundle zip (e.g. the all-province
    ``98-401-X2021006_eng_CSV.zip``) uniformly downstream.
    """

    container: Path
    inner: str | None = None

    @property
    def name(self) -> str:
        """Filename of the underlying CSV (always the data CSV's own name)."""
        return Path(self.inner).name if self.inner is not None else self.container.name

    @property
    def display(self) -> str:
        """Human-readable label for log output. For zip members, formats as
        ``<archive>!<inner>`` so logs disambiguate same-named CSVs in different zips."""
        if self.inner is not None:
            return f"{self.container.name}!{Path(self.inner).name}"
        return self.container.name

    def output_path(self, input_root: Path, output_root: Path) -> Path:
        """Compute where filtered output should be written.

        Outputs always carry a ``.csv`` extension regardless of input form;
        re-zipping the small filtered result isn't worth the complexity.
        For zip members the output lives next to the source archive, named
        after the inner CSV.
        """
        if self.inner is None:
            rel = self.container.relative_to(input_root)
        else:
            rel = self.container.parent.relative_to(input_root) / self.name
        if rel.suffix.lower() != ".csv":
            rel = rel.with_suffix(".csv")
        return output_root / rel


def _csv_pruids(source: CensusCsvSource) -> frozenset[str]:
    """Return the set of PRUIDs covered by a Census Profile CSV, by filename.

    Returns an empty set for filenames that don't match the known StatCan
    bundle layout; callers should treat that as "unknown coverage" and
    typically skip those files rather than guessing.
    """
    match = CENSUS_CSV_PATTERN.match(source.name)
    if not match:
        return frozenset()
    province_token = match.group(2)
    return CENSUS_CSV_TO_PRUIDS.get(province_token, frozenset())


@contextmanager
def _open_census_csv(source: CensusCsvSource):
    """Yield a real CSV path, extracting from a zip transparently if needed.

    On context exit, the temp directory used for extraction (if any) is
    cleaned up; the in-memory data read during the ``with`` block survives
    independently as long as it was eagerly collected.
    """
    if source.inner is None and source.container.suffix.lower() != ".zip":
        yield source.container
        return

    with tempfile.TemporaryDirectory(prefix="census_extract_") as tmpdir:
        with zipfile.ZipFile(source.container) as zf:
            target = source.inner
            if target is None:
                # Per-province zip discovered by filename only -- locate the
                # data CSV inside.
                members = [m for m in zf.namelist() if not m.endswith("/")]
                data_members = [
                    m
                    for m in members
                    if Path(m).suffix.lower() == ".csv"
                    and CENSUS_CSV_PATTERN.match(Path(m).name)
                ]
                if not data_members:
                    msg = (
                        f"no Census Profile data CSV inside {source.container.name} "
                        f"(members: {members})"
                    )
                    raise FileNotFoundError(msg)
                target = data_members[0]
            extracted = Path(zf.extract(target, tmpdir))
        yield extracted


# =============================================================================
# INPUT DISCOVERY
# =============================================================================


def discover_da_shapefile(input_dir: Path, output_dir: Path) -> Path | None:
    """Find the StatCan DA boundary file under *input_dir*.

    Accepts either an unzipped ``.shp`` or a ``.zip`` archive. If both exist,
    the unzipped ``.shp`` is preferred (the zip would just need to be
    extracted again).
    """
    candidates: dict[str, Path] = {}
    for ext in (".zip", ".shp"):
        for path in sorted(input_dir.rglob(f"lda_*{ext}")):
            if _is_under(path, output_dir):
                continue
            if DA_SHAPEFILE_PATTERN.match(path.name):
                # Stem (sans extension) keys the dedup so .shp wins over .zip.
                candidates[path.stem] = path
    if not candidates:
        return None
    # Prefer .shp variants when both exist for the same stem.
    for path in candidates.values():
        if path.suffix.lower() == ".shp":
            return path
    return next(iter(candidates.values()))


def discover_census_csvs(input_dir: Path, output_dir: Path) -> list[CensusCsvSource]:
    """Find all Census Profile data CSVs under *input_dir*.

    Handles three packagings StatCan ships:

    * Direct ``.csv`` files matching the per-province pattern.
    * Per-province zips (e.g. ``..._Quebec.zip``) whose filename matches.
    * Bundle zips (e.g. ``98-401-X2021006_eng_CSV.zip``) whose filename does
      *not* match the per-province pattern but which contain multiple
      per-province data CSVs inside.

    Every zip is opened just to read its directory (cheap); inner CSVs whose
    *member* name matches ``CENSUS_CSV_PATTERN`` become individual sources.
    De-dup: if the same data-CSV filename appears as both a direct file on
    disk and as a zip member, the direct file wins (no extraction needed at
    read time).
    """
    sources_by_name: dict[str, CensusCsvSource] = {}

    # Pass 1: direct .csv files matching the per-province pattern.
    for path in sorted(input_dir.rglob("*.csv")):
        if _is_under(path, output_dir):
            continue
        if not CENSUS_CSV_PATTERN.match(path.name):
            continue
        sources_by_name[path.name] = CensusCsvSource(container=path, inner=None)

    # Pass 2: inner CSVs nested in zip archives (whether per-province or bundle).
    for path in sorted(input_dir.rglob("*.zip")):
        if _is_under(path, output_dir):
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                members = zf.namelist()
        except zipfile.BadZipFile:
            print(f"  [skip] bad zip: {path.name}")
            continue

        for member in members:
            if member.endswith("/"):
                continue
            member_path = Path(member)
            if member_path.suffix.lower() != ".csv":
                continue
            if not CENSUS_CSV_PATTERN.match(member_path.name):
                continue
            if member_path.name in sources_by_name:
                # Already discovered as a direct file; prefer that.
                continue
            sources_by_name[member_path.name] = CensusCsvSource(
                container=path, inner=member
            )

    return sorted(sources_by_name.values(), key=lambda s: s.name)


def load_da_shapefile(path: Path) -> gpd.GeoDataFrame:
    """Load the DA boundary file. Handles either .shp or .zip transparently.

    For ``.zip`` inputs the archive is extracted into a temp dir, read into
    memory, and the temp dir is cleaned up at function exit (the GeoDataFrame
    survives independently).
    """
    if path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="da_extract_") as tmpdir:
            extract_dir = Path(tmpdir)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(extract_dir)
            shp_candidates = list(extract_dir.rglob("*.shp"))
            if not shp_candidates:
                msg = f"no .shp inside {path.name}"
                raise FileNotFoundError(msg)
            print(f"  extracted from {path.name}")
            gdf = gpd.read_file(shp_candidates[0])
    else:
        print(f"  reading {path.name}")
        gdf = gpd.read_file(path)

    # Coerce string identifiers; LANDAREA stays numeric.
    for col in ("DAUID", "DGUID", "PRUID"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str)
    if "LANDAREA" in gdf.columns:
        gdf["LANDAREA"] = pd.to_numeric(gdf["LANDAREA"], errors="coerce")

    # Derive CDUID for downstream grouping.
    gdf["CDUID"] = gdf["DAUID"].str[:4]
    return gdf


def load_da_population(sources: list[CensusCsvSource]) -> dict[str, int]:
    """Build a ``DAUID -> population_2021`` lookup from province CSV sources.

    Uses the polars lazy API (``scan_csv``) so column projection and the
    DA/CHARACTERISTIC_ID filters push down into the CSV reader, keeping peak
    memory bounded regardless of input size. Typical speedup over the
    equivalent pandas pipeline is 5-10x on full-province releases.

    Encoding: polars only supports ``utf8`` or ``utf8-lossy``, not latin-1.
    Used here on the safe assumption that the columns we touch (DGUID,
    ALT_GEO_CODE, GEO_LEVEL, CHARACTERISTIC_ID, C1_COUNT_TOTAL) are pure
    ASCII -- the StatCan French characters live in GEO_NAME, which this
    function does not read. ``filter_census_csv`` handles GEO_NAME via
    pandas latin-1 to preserve accents end-to-end.

    Rows with missing identifiers (which appear in malformed sample files)
    are silently dropped via the ``ALT_GEO_CODE.is_not_null()`` predicate.
    """
    pop_lookup: dict[str, int] = {}
    columns = [
        "DGUID",
        "ALT_GEO_CODE",
        "GEO_LEVEL",
        "CHARACTERISTIC_ID",
        "C1_COUNT_TOTAL",
    ]
    # Force all relevant columns to Utf8; C1_COUNT_TOTAL in particular has
    # mixed content (integers and quality symbols like 'x', 'F') that would
    # otherwise trip dtype inference.
    schema_overrides = {col: pl.Utf8 for col in columns}

    for source in sources:
        t0 = time.monotonic()
        print(f"  scanning {source.display} ...", flush=True)
        try:
            with _open_census_csv(source) as real_csv:
                result = (
                    pl.scan_csv(
                        real_csv,
                        encoding="utf8-lossy",
                        has_header=True,
                        schema_overrides=schema_overrides,
                    )
                    .select(columns)
                    .filter(
                        (pl.col("GEO_LEVEL") == DA_GEO_LEVEL)
                        & (pl.col("CHARACTERISTIC_ID") == POPULATION_CHARACTERISTIC_ID)
                        & pl.col("ALT_GEO_CODE").is_not_null()
                    )
                    .select(
                        [
                            pl.col("ALT_GEO_CODE"),
                            # Non-numeric quality symbols become null then 0.
                            pl.col("C1_COUNT_TOTAL")
                            .cast(pl.Int64, strict=False)
                            .fill_null(0)
                            .alias("population"),
                        ]
                    )
                    .collect()
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] couldn't read {source.display}: {exc}")
            continue

        added = dict(zip(result["ALT_GEO_CODE"].to_list(), result["population"].to_list()))
        pop_lookup.update(added)
        elapsed = time.monotonic() - t0
        print(f"  {source.display}: loaded {len(added):,} DA populations ({elapsed:.1f}s)")

    return pop_lookup


# =============================================================================
# SAMPLING (stratified)
# =============================================================================


def _stratified_picks(
    sub: gpd.GeoDataFrame,
    pop_lookup: dict[str, int],
    target_n: int,
) -> list[tuple[str, str]]:
    """Return up to *target_n* (DAUID, reason) tuples from one CD.

    Strata are applied in priority order. Each stratum picks at most one DA
    and never re-picks one already chosen. The remainder is filled randomly.
    """
    selected: dict[str, str] = {}

    sub_pop = sub["DAUID"].map(pop_lookup).fillna(0).astype(int)

    # Stratum 1: highest population (residential density edge case). Skipped
    # if no population data was loaded for any DA in this CD (e.g., the
    # province's CSV wasn't supplied).
    if (sub_pop > 0).any():
        top_pop_dauid = sub.loc[sub_pop.idxmax(), "DAUID"]
        selected[top_pop_dauid] = "max_pop"

    # Stratum 2: zero-population DA (uninhabited / industrial / vacant).
    zero_pool = sub.loc[(sub_pop == 0) & (~sub["DAUID"].isin(selected))]
    if not zero_pool.empty:
        pick = zero_pool.sample(n=1, random_state=RANDOM_SEED).iloc[0]
        selected[pick["DAUID"]] = "zero_pop"

    # Stratum 3: highest LANDAREA (rural mega-DA edge case). DA land areas
    # span orders of magnitude even though populations are bounded.
    water_pool = sub.loc[~sub["DAUID"].isin(selected)]
    if not water_pool.empty and "LANDAREA" in water_pool.columns:
        top_area = water_pool.nlargest(1, "LANDAREA").iloc[0]
        if pd.notna(top_area["LANDAREA"]) and top_area["LANDAREA"] > 0:
            selected[top_area["DAUID"]] = "max_area"

    # Stratum 4: random fill.
    remaining = target_n - len(selected)
    if remaining > 0:
        pool = sub.loc[~sub["DAUID"].isin(selected)]
        if not pool.empty:
            picks = pool.sample(n=min(remaining, len(pool)), random_state=RANDOM_SEED)
            for dauid in picks["DAUID"]:
                selected[dauid] = "random"

    return list(selected.items())


def _allocate_targets(gdf: gpd.GeoDataFrame) -> dict[tuple[str, str], int]:
    """Compute per-CD sample sizes, proportional to CD DA count."""
    counts: dict[tuple[str, str], int] = {}
    for pruid, cd_codes in CDS_BY_PROVINCE.items():
        for cd_code in cd_codes:
            cduid = f"{pruid}{cd_code}"
            counts[(pruid, cd_code)] = int((gdf["CDUID"] == cduid).sum())

    total = sum(counts.values()) or 1
    targets: dict[tuple[str, str], int] = {}
    for key, count in counts.items():
        raw = TARGET_TOTAL * count / total
        targets[key] = max(MIN_PER_CD, min(MAX_PER_CD, round(raw)))
    return targets


def sample_das(
    gdf: gpd.GeoDataFrame,
    pop_lookup: dict[str, int],
) -> pd.DataFrame:
    """Stratified-sample DAs across all configured CDs; return manifest DataFrame."""
    targets = _allocate_targets(gdf)
    rows: list[dict[str, object]] = []

    for pruid, cd_codes in CDS_BY_PROVINCE.items():
        for cd_code in cd_codes:
            cduid = f"{pruid}{cd_code}"
            sub = gdf[gdf["CDUID"] == cduid]
            if sub.empty:
                print(f"  [warn] no DAs for PRUID={pruid} CDUID={cduid}")
                continue
            for dauid, reason in _stratified_picks(
                sub, pop_lookup, targets[(pruid, cd_code)]
            ):
                rows.append(
                    {
                        "DAUID": dauid,
                        "DGUID": sub.loc[sub["DAUID"] == dauid, "DGUID"].iloc[0],
                        "PRUID": pruid,
                        "CDUID": cduid,
                        "selection_reason": reason,
                        "population_2021": int(pop_lookup.get(dauid, 0)),
                        "land_area_km2": float(
                            sub.loc[sub["DAUID"] == dauid, "LANDAREA"].iloc[0]
                        ),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(["PRUID", "CDUID", "DAUID"])
        .reset_index(drop=True)
    )


# =============================================================================
# OUTPUT: SAMPLE SHAPEFILE
# =============================================================================


def write_da_sample(
    gdf: gpd.GeoDataFrame,
    manifest: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Filter the DA shapefile to manifest DAUIDs; write a single zipped shapefile.

    Output naming mirrors the StatCan convention with a ``_sample`` suffix.
    All shapefile components (.shp/.shx/.dbf/.prj/.cpg) are bundled into one
    zip at the output root.
    """
    wanted = set(manifest["DAUID"])
    out_gdf = gdf[gdf["DAUID"].isin(wanted)].copy()
    missing = wanted - set(out_gdf["DAUID"])
    if missing:
        print(f"  [warn] {len(missing)} manifest DAUIDs not found in source shapefile")

    # The derived CDUID isn't part of the StatCan schema; drop before writing
    # so the output round-trips through downstream tools that expect the
    # original column set.
    if "CDUID" in out_gdf.columns:
        out_gdf = out_gdf.drop(columns=["CDUID"])

    with tempfile.TemporaryDirectory(prefix="da_write_") as tmpdir:
        tmp_root = Path(tmpdir)
        out_gdf.to_file(tmp_root / OUTPUT_DA_SHAPEFILE, driver="ESRI Shapefile", index=False)

        zip_filename = Path(OUTPUT_DA_SHAPEFILE).with_suffix(".zip").name
        out_zip = output_dir / zip_filename
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for component in sorted(tmp_root.iterdir()):
                zout.write(component, component.name)

    print(f"  wrote {len(out_gdf):,} DAs -> {out_zip}")


# =============================================================================
# CSV FILTERING
# =============================================================================


def filter_census_csv(
    source: CensusCsvSource,
    dauids: set[str],
    input_root: Path,
    output_root: Path,
) -> bool:
    """Filter one Census Profile CSV down to manifest DAUIDs + Country rows.

    Reads the source in chunks to keep memory bounded on full-province files.
    Country rows are preserved as a sanity-check anchor; CSD and other parent
    rows are dropped (see module docstring for rationale).
    """
    out_path = source.output_path(input_root, output_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_read = 0
    header_written = False
    t0 = time.monotonic()
    print(f"  filtering {source.display} ...", flush=True)

    with _open_census_csv(source) as real_csv, open(
        out_path, "w", encoding=CENSUS_ENCODING, newline=""
    ) as out_f:
        try:
            chunks = pd.read_csv(
                real_csv,
                dtype=str,
                encoding=CENSUS_ENCODING,
                chunksize=200_000,
                low_memory=False,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            print(f"  [skip] couldn't read {source.display}: {exc}")
            return False

        for chunk_idx, chunk in enumerate(chunks, start=1):
            total_read += len(chunk)

            # Drop the malformed phantom rows seen in some sample files
            # (rows with empty DGUID / ALT_GEO_CODE / GEO_LEVEL).
            chunk = chunk.dropna(subset=["DGUID", "GEO_LEVEL"])

            keep_da = (
                chunk["GEO_LEVEL"].eq(DA_GEO_LEVEL)
                & chunk["ALT_GEO_CODE"].isin(dauids)
            )
            keep_country = chunk["GEO_LEVEL"].eq(COUNTRY_GEO_LEVEL)
            filtered = chunk[keep_da | keep_country]

            if not filtered.empty:
                filtered.to_csv(out_f, index=False, header=not header_written, lineterminator="\r\n")
                header_written = True
                total_kept += len(filtered)

            # Light progress trace every ~5M rows so long scans show signs of life.
            if chunk_idx % 25 == 0:
                print(
                    f"    ... {total_read:,} rows scanned, {total_kept:,} kept "
                    f"({time.monotonic() - t0:.1f}s)",
                    flush=True,
                )

    if total_kept == 0:
        # Nothing matched; remove the empty file we just created.
        out_path.unlink(missing_ok=True)
        print(f"  [skip] {source.display}: no matching rows")
        return False

    elapsed = time.monotonic() - t0
    print(f"  {source.display}: kept {total_kept:,} / {total_read:,} rows ({elapsed:.1f}s)")
    return True


# =============================================================================
# ENTRYPOINT
# =============================================================================


def main() -> int:
    """Entry point."""
    manifest_path = OUTPUT_DIR / MANIFEST_FILENAME

    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}", file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: discover and load the DA shapefile -----------------------
    print("Discovering DA boundary file (.shp or .zip)...")
    da_path = discover_da_shapefile(INPUT_DIR, OUTPUT_DIR)
    if da_path is None:
        print(f"No DA boundary file found in {INPUT_DIR}.", file=sys.stderr)
        return 1
    print(f"  found {da_path.name}")

    print("Reading DA geometries...")
    gdf = load_da_shapefile(da_path)
    print(f"  loaded {len(gdf):,} DAs nationally")

    # ---- Phase 2: discover Census Profile CSVs -----------------------------
    print("\nDiscovering Census Profile CSVs...")
    all_sources = discover_census_csvs(INPUT_DIR, OUTPUT_DIR)
    target_pruids = frozenset(CDS_BY_PROVINCE.keys())
    sources = [s for s in all_sources if _csv_pruids(s) & target_pruids]
    skipped = [s for s in all_sources if s not in sources]

    if not all_sources:
        print(f"  [warn] no Census Profile CSVs found in {INPUT_DIR}")
    else:
        for s in sources:
            print(f"  using {s.display} (PRUIDs {sorted(_csv_pruids(s))})")
        for s in skipped:
            reason = (
                f"covers {sorted(_csv_pruids(s))}, none in target"
                if _csv_pruids(s)
                else "unknown PRUID coverage (filename token not in map)"
            )
            print(f"  skipping {s.display} ({reason})")

    covered = frozenset().union(*(_csv_pruids(s) for s in sources)) if sources else frozenset()
    missing_pruids = target_pruids - covered
    if missing_pruids:
        print(
            f"  [warn] no CSV covers PRUID(s) {sorted(missing_pruids)}; "
            f"max_pop and zero_pop strata will fall back to random for those provinces"
        )

    # ---- Phase 3: sample (first run) or replay (manifest exists) -----------
    if manifest_path.exists():
        print(f"\nManifest found ({manifest_path.name}) - replay mode")
        manifest = pd.read_csv(manifest_path, dtype=str)
    else:
        print("\nLoading DA populations for stratification...")
        pop_lookup = load_da_population(sources)
        if pop_lookup:
            print(f"  loaded {len(pop_lookup):,} DA->population entries")
        else:
            print("  no population data found; max_pop and zero_pop strata will fall back to random")

        print("\nSampling DAs (first run, stratified)...")
        manifest = sample_das(gdf, pop_lookup)
        manifest.to_csv(manifest_path, index=False)
        print(f"  wrote manifest -> {manifest_path}")
        reason_counts = manifest["selection_reason"].value_counts().to_dict()
        print(f"  selection breakdown: {reason_counts}")

    # ---- Phase 4: write the sample shapefile -------------------------------
    print("\nWriting DA sample shapefile...")
    write_da_sample(gdf, manifest, OUTPUT_DIR)

    # ---- Phase 5: filter Census Profile CSVs -------------------------------
    dauids = set(manifest["DAUID"].astype(str))
    print(f"\nFiltering Census Profile CSVs to {len(dauids):,} DAUIDs...")
    written = 0
    for source in sources:
        if filter_census_csv(source, dauids, INPUT_DIR, OUTPUT_DIR):
            written += 1

    print(f"\nSummary: 1 shapefile, {written} census CSV(s), {len(manifest):,} DAs")
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code != 0:
        raise SystemExit(_exit_code)
