"""Generate the full Census fixture set: TIGER blocks + LEHD WAC + Census tables.

Edit the paths in the CONFIG block below and run from the repo root.

Workflow
--------

1. Discovers TIGER tabblock20 shapefiles (one per state), LEHD WAC files, and
   data.census.gov table bundles (zipped or unzipped) anywhere under
   ``INPUT_DIR`` (sub-directories searched recursively).

2. Performs a jobs-aware stratified sample of blocks per county:

   * highest-POP20 block (residential density edge case)
   * highest-C000 WAC block (employment center edge case)
   * zero-POP20 block, if any (uninhabited / commercial / vacant edge case)
   * highest-AWATER20 block, if any (water-heavy edge case)
   * remainder filled with a proportional random draw, clamped per county to
     ``[MIN_PER_COUNTY, MAX_PER_COUNTY]``.

3. Writes per-state zipped sample shapefiles (one ``.zip`` per state, with the
   shapefile components at the zip's top level) plus ``sample_block_geoids.csv``
   (the manifest, which carries the geoid, county, selection reason, and total
   jobs for every picked block).

4. Filters every Census table bundle (zipped or unzipped folder) and every
   LEHD file (.csv or .csv.gz) in ``INPUT_DIR`` down to the geographies
   referenced in the manifest. Census bundle geography level is auto-detected
   from the ``GEO_ID`` prefix:

   * ``1000000US`` → Census Block (15-char ID)
   * ``1500000US`` → Block Group  (12-char ID)
   * ``1400000US`` → Census Tract (11-char ID)
   * ``0500000US`` → County       (5-char ID)

   LEHD files (WAC or RAC) are filtered by 15-char block ID in ``w_geocode``
   or ``h_geocode``.

Modes
-----

* First run (no manifest in ``OUTPUT_DIR``): generate everything from scratch.
* Replay (manifest present): reuse the manifest's GEOID20 list, skip WAC
  loading and sampling, just write shapefiles and re-filter tables / LEHD.
  Reproduces fixtures byte-for-byte across machines and package versions.
  Delete the manifest to re-sample.

``OUTPUT_DIR`` may be nested inside ``INPUT_DIR``; the script skips anything
under ``OUTPUT_DIR`` during input discovery so re-runs are idempotent.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

#: Root directory holding all fixture inputs: TIGER tabblock20 folders, LEHD
#: WAC files (.csv or .csv.gz), and data.census.gov bundles (zipped or
#: unzipped). Sub-directories are searched automatically; anything under
#: ``OUTPUT_DIR`` is skipped so re-runs don't reprocess their own output.
INPUT_DIR: Final[Path] = Path(r"PATH\TO\CENSUS\FIXTURE\INPUTS")  # <<< EDIT ME

#: Directory where sample shapefiles, the manifest, and all filtered fixtures
#: are written. May be nested inside ``INPUT_DIR``.
OUTPUT_DIR: Final[Path] = Path(r"PATH\TO\CENSUS\FIXTURE\OUTPUT")  # <<< EDIT ME

#: Map from 2-char state FIPS to the list of 3-char county FIPS to sample.
COUNTIES_BY_STATE: Final[dict[str, list[str]]] = {
    "11": ["001"],  # Washington, DC
    "24": [  # Maryland
        "017",  # Charles
        "027",  # Howard
        "031",  # Montgomery
        "033",  # Prince George's
    ],
    "51": [  # Virginia
        "013",  # Arlington
        "059",  # Fairfax County
        "107",  # Loudoun
        "153",  # Prince William
        "510",  # Alexandria City
        "600",  # Fairfax City
        "610",  # Falls Church City
    ],
}

# ---- File patterns ----------------------------------------------------------

#: Regex for discovering TIGER tabblock20 inputs (either unzipped .shp or
#: zipped .zip). State FIPS in group 1.
TIGER_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^tl_\d{4}_(\d{2})_tabblock20\.(shp|zip)$"
)

OUTPUT_TIGER_TEMPLATE: Final[str] = "tl_2025_{state_fp}_tabblock20_sample.shp"
MANIFEST_FILENAME: Final[str] = "sample_block_geoids.csv"

# ---- Sampling parameters ----------------------------------------------------

TARGET_TOTAL: Final[int] = 120
MIN_PER_COUNTY: Final[int] = 5
MAX_PER_COUNTY: Final[int] = 15
RANDOM_SEED: Final[int] = 42

# ---- Filtering identifiers --------------------------------------------------

GEO_ID_COL: Final[str] = "GEO_ID"
LEHD_GEO_COLS: Final[tuple[str, ...]] = ("w_geocode", "h_geocode")
WAC_JOBS_COL: Final[str] = "C000"  # LEHD WAC: total jobs by workplace block

#: Census summary-level prefix -> (geo_keys lookup name, ID length after prefix).
GEO_PREFIXES: Final[dict[str, tuple[str, int]]] = {
    "1000000US": ("block", 15),
    "1500000US": ("block_group", 12),
    "1400000US": ("tract", 11),
    "0500000US": ("county", 5),
}


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


# =============================================================================
# INPUT DISCOVERY
# =============================================================================


def discover_tiger_inputs(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Find TIGER tabblock20 inputs, keyed by 2-char state FIPS.

    Accepts either unzipped ``.shp`` files or zipped ``.zip`` archives. If both
    variants exist for a given state, the unzipped ``.shp`` is preferred (the
    zip would just need to be extracted again).
    """
    inputs: dict[str, Path] = {}
    # Search zips first so .shp variants override.
    for ext in (".zip", ".shp"):
        for path in sorted(input_dir.rglob(f"tl_*_tabblock20{ext}")):
            if _is_under(path, output_dir):
                continue
            m = TIGER_FILENAME_PATTERN.match(path.name)
            if m:
                inputs[m.group(1)] = path
    return inputs


def load_tiger_geometries(
    state_to_path: dict[str, Path],
) -> dict[str, gpd.GeoDataFrame]:
    """Read each TIGER input and coerce ``GEOID20`` to string.

    For ``.zip`` inputs, the archive is extracted into a temporary directory,
    the shapefile is read into memory, and the temp dir is cleaned up at the
    end of the function (the in-memory GeoDataFrame survives independently).
    """
    gdfs: dict[str, gpd.GeoDataFrame] = {}
    with tempfile.TemporaryDirectory(prefix="tiger_extract_") as tmpdir:
        tmp_root = Path(tmpdir)
        for state_fp, path in state_to_path.items():
            if path.suffix.lower() == ".zip":
                extract_dir = tmp_root / state_fp
                extract_dir.mkdir()
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(extract_dir)
                shp_candidates = list(extract_dir.rglob("*.shp"))
                if not shp_candidates:
                    print(f"  [warn] no .shp inside {path.name}, skipping")
                    continue
                shp = shp_candidates[0]
                print(f"  state {state_fp}: extracted from {path.name}")
            else:
                shp = path
                print(f"  state {state_fp}: reading {path.name}")
            gdf = gpd.read_file(shp)
            gdf["GEOID20"] = gdf["GEOID20"].astype(str)
            gdfs[state_fp] = gdf
    return gdfs


def load_wac_jobs(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Build a block-GEOID -> total-jobs (C000) lookup from LEHD WAC files.

    Reads any ``.csv`` or ``.csv.gz`` under *input_dir* that has both
    ``w_geocode`` and ``C000`` columns. Returns an empty dict if no WAC files
    are present, in which case the ``max_jobs`` stratum is skipped during
    sampling.
    """
    jobs: dict[str, int] = {}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".csv", ".gz"):
            continue
        if _is_under(path, output_dir):
            continue
        try:
            head = pd.read_csv(path, nrows=0, compression="infer")
        except Exception:  # noqa: BLE001
            continue
        if "w_geocode" not in head.columns or WAC_JOBS_COL not in head.columns:
            continue
        df = pd.read_csv(
            path,
            usecols=["w_geocode", WAC_JOBS_COL],
            dtype={"w_geocode": str, WAC_JOBS_COL: int},
            compression="infer",
        )
        jobs.update(dict(zip(df["w_geocode"], df[WAC_JOBS_COL])))
    return jobs


# =============================================================================
# SAMPLING (jobs-aware stratification)
# =============================================================================


def _stratified_picks(
    sub: gpd.GeoDataFrame,
    jobs_lookup: dict[str, int],
    target_n: int,
) -> list[tuple[str, str]]:
    """Return up to *target_n* (GEOID20, reason) tuples from one county.

    Strata are applied in priority order. Each stratum picks at most one block
    and never re-picks one already chosen. The remainder is filled randomly.
    """
    selected: dict[str, str] = {}

    # Stratum 1: highest POP20 (residential density edge case).
    top_pop = sub.nlargest(1, "POP20").iloc[0]
    selected[top_pop["GEOID20"]] = "max_pop"

    # Stratum 2: highest C000 from WAC (employment center). Skipped if no WAC
    # data was loaded or no block in this county appears in WAC with jobs > 0.
    if jobs_lookup:
        sub_jobs = sub["GEOID20"].map(jobs_lookup).fillna(0).astype(int)
        if (sub_jobs > 0).any():
            top_jobs_geoid = sub.loc[sub_jobs.idxmax(), "GEOID20"]
            if top_jobs_geoid not in selected:
                selected[top_jobs_geoid] = "max_jobs"

    # Stratum 3: a zero-POP20 block (uninhabited / commercial / vacant).
    zero_pool = sub[(sub["POP20"] == 0) & (~sub["GEOID20"].isin(selected))]
    if not zero_pool.empty:
        pick = zero_pool.sample(n=1, random_state=RANDOM_SEED).iloc[0]
        selected[pick["GEOID20"]] = "zero_pop"

    # Stratum 4: highest AWATER20 (only if it actually has water).
    water_pool = sub[~sub["GEOID20"].isin(selected)]
    if not water_pool.empty:
        top_water = water_pool.nlargest(1, "AWATER20").iloc[0]
        if top_water["AWATER20"] > 0:
            selected[top_water["GEOID20"]] = "max_water"

    # Stratum 5: random fill.
    remaining = target_n - len(selected)
    if remaining > 0:
        pool = sub[~sub["GEOID20"].isin(selected)]
        if not pool.empty:
            picks = pool.sample(n=min(remaining, len(pool)), random_state=RANDOM_SEED)
            for geoid in picks["GEOID20"]:
                selected[geoid] = "random"

    return list(selected.items())


def _allocate_targets(
    gdfs: dict[str, gpd.GeoDataFrame],
) -> dict[tuple[str, str], int]:
    """Compute per-county sample sizes, proportional to county block count."""
    counts: dict[tuple[str, str], int] = {}
    for state_fp, gdf in gdfs.items():
        for county_fp in COUNTIES_BY_STATE[state_fp]:
            counts[(state_fp, county_fp)] = int((gdf["COUNTYFP20"] == county_fp).sum())

    total = sum(counts.values()) or 1
    targets: dict[tuple[str, str], int] = {}
    for key, count in counts.items():
        raw = TARGET_TOTAL * count / total
        targets[key] = max(MIN_PER_COUNTY, min(MAX_PER_COUNTY, round(raw)))
    return targets


def sample_blocks(
    gdfs: dict[str, gpd.GeoDataFrame],
    jobs_lookup: dict[str, int],
) -> pd.DataFrame:
    """Stratified-sample blocks across all states; return manifest dataframe."""
    targets = _allocate_targets(gdfs)
    rows: list[dict[str, object]] = []

    for state_fp, gdf in gdfs.items():
        for county_fp in COUNTIES_BY_STATE[state_fp]:
            sub = gdf[gdf["COUNTYFP20"] == county_fp]
            if sub.empty:
                print(f"  [warn] no blocks for state={state_fp} county={county_fp}")
                continue
            for geoid, reason in _stratified_picks(
                sub, jobs_lookup, targets[(state_fp, county_fp)]
            ):
                rows.append(
                    {
                        "GEOID20": geoid,
                        "state_fp": state_fp,
                        "county_fp": county_fp,
                        "selection_reason": reason,
                        "total_jobs": int(jobs_lookup.get(geoid, 0)),
                    }
                )

    return (
        pd.DataFrame(rows).sort_values(["state_fp", "county_fp", "GEOID20"]).reset_index(drop=True)
    )


def write_tiger_samples(
    gdfs: dict[str, gpd.GeoDataFrame],
    manifest: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Filter each state's blocks to the manifest GEOIDs; write as a zipped shapefile.

    The shapefile is written into a temp dir, then all of its components
    (.shp/.shx/.dbf/.prj/.cpg) are bundled into a single ``.zip`` at the output
    root. Output naming mirrors the standard TIGER zip convention: one zip per
    state, with the shapefile components at the zip's top level.
    """
    with tempfile.TemporaryDirectory(prefix="tiger_write_") as tmpdir:
        tmp_root = Path(tmpdir)
        for state_fp, gdf in gdfs.items():
            wanted = set(manifest.loc[manifest["state_fp"] == state_fp, "GEOID20"])
            out_gdf = gdf[gdf["GEOID20"].isin(wanted)]
            missing = wanted - set(out_gdf["GEOID20"])
            if missing:
                print(
                    f"  [warn] state {state_fp}: {len(missing)} manifest GEOIDs not found in source"
                )

            # Write the shapefile into its own per-state temp subdir to keep
            # component listings unambiguous.
            shp_filename = OUTPUT_TIGER_TEMPLATE.format(state_fp=state_fp)
            state_tmp = tmp_root / state_fp
            state_tmp.mkdir()
            out_gdf.to_file(state_tmp / shp_filename, driver="ESRI Shapefile", index=False)

            # Bundle every component into a single zip at the output root.
            zip_filename = Path(shp_filename).with_suffix(".zip").name
            out_zip = output_dir / zip_filename
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
                for component in sorted(state_tmp.iterdir()):
                    zout.write(component, component.name)

            print(f"  state {state_fp}: wrote {len(out_gdf):,} blocks -> {out_zip}")


# =============================================================================
# GEO KEY DERIVATION (for downstream filtering)
# =============================================================================


def derive_geo_keys(manifest: pd.DataFrame) -> dict[str, set[str]]:
    """Build sets of GEOIDs at each geography level from the manifest."""
    blocks = set(manifest["GEOID20"].astype(str))
    state_county = manifest["state_fp"].astype(str) + manifest["county_fp"].astype(str)
    return {
        "block": blocks,
        "block_group": {b[:12] for b in blocks},
        "tract": {b[:11] for b in blocks},
        "county": set(state_county.tolist()),
    }


# =============================================================================
# CENSUS TABLE FILTERING (folder + zip)
# =============================================================================


def _filter_data_csv(
    data_file: Path, geo_keys: dict[str, set[str]]
) -> tuple[pd.DataFrame, str] | None:
    """Filter a data.census.gov *-Data.csv. Returns (DataFrame, geo_level) or None.

    The returned DataFrame includes the human-readable label row at position 0,
    matching the format consumers expect (i.e., ``skiprows=[1]`` reads cleanly).
    """
    df = pd.read_csv(data_file, dtype=str, encoding="utf-8-sig")
    if GEO_ID_COL not in df.columns:
        print(f"  [skip] no GEO_ID column in {data_file.name}")
        return None

    label_row, data_rows = df.iloc[0:1], df.iloc[1:]
    if data_rows.empty:
        print(f"  [skip] no data rows in {data_file.name}")
        return None

    sample_geo = data_rows[GEO_ID_COL].iloc[0]
    match = next((p for p in GEO_PREFIXES if sample_geo.startswith(p)), None)
    if match is None:
        print(f"  [skip] unrecognized GEO_ID prefix in {data_file.name}: {sample_geo!r}")
        return None
    keys_name, id_len = GEO_PREFIXES[match]
    wanted = geo_keys[keys_name]

    filtered = data_rows[data_rows[GEO_ID_COL].str[-id_len:].isin(wanted)]
    if filtered.empty:
        print(f"  [skip] {data_file.name}: no matching rows ({keys_name})")
        return None

    return pd.concat([label_row, filtered], ignore_index=True), keys_name


def filter_census_folder(
    bundle_dir: Path,
    geo_keys: dict[str, set[str]],
    input_root: Path,
    output_root: Path,
) -> bool:
    """Filter an unzipped data.census.gov folder. Returns True if written."""
    data_files = list(bundle_dir.glob("*-Data.csv"))
    if not data_files:
        return False
    data_file = data_files[0]

    result = _filter_data_csv(data_file, geo_keys)
    if result is None:
        return False
    out_df, keys_name = result

    sidecars = [f for f in bundle_dir.iterdir() if f.is_file() and f.name != data_file.name]
    out_bundle = output_root / bundle_dir.relative_to(input_root)
    out_bundle.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_bundle / data_file.name, index=False, encoding="utf-8-sig")
    for sidecar in sidecars:
        shutil.copy2(sidecar, out_bundle / sidecar.name)

    print(f"  [census] {bundle_dir.name} ({keys_name}): {len(out_df) - 1:,} rows kept")
    return True


def filter_census_zip(
    zip_path: Path,
    geo_keys: dict[str, set[str]],
    input_root: Path,
    output_root: Path,
) -> bool:
    """Filter a zipped data.census.gov download. Returns True if written."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as zin:
            zin.extractall(tmp)

        data_files = list(tmp.rglob("*-Data.csv"))
        if not data_files:
            return False
        data_file = data_files[0]
        bundle_dir = data_file.parent

        # Snapshot sidecars before adding any new files to the temp dir,
        # otherwise the filtered CSV would loop back in as a sidecar.
        sidecars = [f for f in bundle_dir.iterdir() if f.is_file() and f.name != data_file.name]

        result = _filter_data_csv(data_file, geo_keys)
        if result is None:
            return False
        out_df, keys_name = result

        # Render filtered CSV to disk first to guarantee the BOM is written
        # correctly, then add it to the output zip.
        filtered_path = tmp / f"_filtered_{data_file.name}"
        out_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")

        out_zip = output_root / zip_path.relative_to(input_root)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.write(filtered_path, data_file.name)
            for sidecar in sidecars:
                zout.write(sidecar, sidecar.name)

    print(f"  [census] {zip_path.name} ({keys_name}, zipped): {len(out_df) - 1:,} rows kept")
    return True


def _zip_contains_census_bundle(zip_path: Path) -> bool:
    """Return True if *zip_path* contains a data.census.gov *-Data.csv member."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return any(n.endswith("-Data.csv") for n in zf.namelist())
    except zipfile.BadZipFile:
        print(f"  [skip] bad zip: {zip_path.name}")
        return False


# =============================================================================
# LEHD LODES FILTERING (block-level)
# =============================================================================


def _lehd_geo_column(path: Path) -> str | None:
    """Return the LEHD geo column name if this CSV/CSV.GZ looks like LEHD."""
    try:
        head = pd.read_csv(path, nrows=0, compression="infer")
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] couldn't read header of {path.name}: {exc}")
        return None
    for col in LEHD_GEO_COLS:
        if col in head.columns:
            return col
    return None


def filter_lehd_file(
    path: Path,
    blocks: set[str],
    input_root: Path,
    output_root: Path,
    geo_col: str,
) -> bool:
    """Filter a LEHD WAC/RAC CSV (.csv or .csv.gz) by block geocode."""
    df = pd.read_csv(path, dtype={geo_col: str}, compression="infer")
    filtered = df[df[geo_col].isin(blocks)]
    if filtered.empty:
        print(f"  [skip] {path.name}: no matching rows")
        return False

    out_path = output_root / path.relative_to(input_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    filtered.to_csv(out_path, index=False, compression=compression)
    print(f"  [lehd]   {path.name} ({geo_col}): {len(filtered):,}/{len(df):,} rows kept")
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

    # ---- Phase 1: discover and load TIGER ----------------------------------
    print("Discovering TIGER block inputs (.shp or .zip)...")
    state_to_shp = discover_tiger_inputs(INPUT_DIR, OUTPUT_DIR)
    for state_fp in COUNTIES_BY_STATE:
        if state_fp not in state_to_shp:
            print(f"  [warn] missing TIGER input for state {state_fp}")
    if not state_to_shp:
        print(f"No TIGER inputs found in {INPUT_DIR}.", file=sys.stderr)
        return 1
    print(f"  found {len(state_to_shp)} state(s): {sorted(state_to_shp)}")

    print("Reading TIGER geometries...")
    gdfs = load_tiger_geometries(state_to_shp)

    # ---- Phase 2: sample (first run) or replay (manifest exists) -----------
    if manifest_path.exists():
        print(f"\nManifest found ({manifest_path.name}) - replay mode")
        manifest = pd.read_csv(manifest_path, dtype=str)
    else:
        print("\nLoading WAC jobs data for jobs-aware sampling...")
        jobs_lookup = load_wac_jobs(INPUT_DIR, OUTPUT_DIR)
        if jobs_lookup:
            print(f"  loaded {len(jobs_lookup):,} block->jobs entries")
        else:
            print("  no WAC data found; max_jobs stratum will be skipped")

        print("\nSampling blocks (first run, stratified)...")
        manifest = sample_blocks(gdfs, jobs_lookup)
        manifest.to_csv(manifest_path, index=False)
        print(f"  wrote manifest -> {manifest_path}")
        reason_counts = manifest["selection_reason"].value_counts().to_dict()
        print(f"  selection breakdown: {reason_counts}")

    # ---- Phase 3: write TIGER sample shapefiles ----------------------------
    print("\nWriting TIGER sample shapefiles...")
    write_tiger_samples(gdfs, manifest, OUTPUT_DIR)

    # ---- Phase 4: derive geo keys, filter tables and LEHD ------------------
    geo_keys = derive_geo_keys(manifest)
    print(
        f"\nDerived geo keys: "
        f"{len(geo_keys['block']):,} blocks, "
        f"{len(geo_keys['block_group']):,} block groups, "
        f"{len(geo_keys['tract']):,} tracts, "
        f"{len(geo_keys['county']):,} counties"
    )

    folder_bundles = sorted(
        {f.parent for f in INPUT_DIR.rglob("*-Data.csv") if not _is_under(f, OUTPUT_DIR)}
    )
    zip_bundles = sorted(
        p
        for p in INPUT_DIR.rglob("*.zip")
        if not _is_under(p, OUTPUT_DIR) and _zip_contains_census_bundle(p)
    )

    print(f"\nCensus table bundles ({len(folder_bundles)} folder, {len(zip_bundles)} zip):")
    written_bundles = 0
    for bdir in folder_bundles:
        if filter_census_folder(bdir, geo_keys, INPUT_DIR, OUTPUT_DIR):
            written_bundles += 1
    for zpath in zip_bundles:
        if filter_census_zip(zpath, geo_keys, INPUT_DIR, OUTPUT_DIR):
            written_bundles += 1

    print("\nLEHD files:")
    written_lehd = 0
    for path in sorted(INPUT_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".csv", ".gz"):
            continue
        if path.parent in folder_bundles:
            continue
        if _is_under(path, OUTPUT_DIR):
            continue
        geo_col = _lehd_geo_column(path)
        if geo_col is None:
            continue
        if filter_lehd_file(path, geo_keys["block"], INPUT_DIR, OUTPUT_DIR, geo_col):
            written_lehd += 1

    print(f"\nSummary: {written_bundles} census bundle(s), {written_lehd} LEHD file(s)")
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code != 0:
        raise SystemExit(_exit_code)
