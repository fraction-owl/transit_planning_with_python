"""Build slim sample TIGER block shapefiles for DC, MD, and VA.

Two modes:

* First run (no manifest present): stratified sample of blocks per county,
  written as ``tl_2025_{state}_tabblock20_sample.shp`` plus
  ``sample_block_geoids.csv`` (the manifest).

* Replay (manifest present): filter inputs by the manifest's GEOID20 list.
  Reproduces the exact same fixtures byte-for-byte across machines and
  package versions. Delete the manifest to re-sample.

Per-county stratification on the first run:

* highest-POP20 block (urban density edge case)
* zero-POP20 block, if any (uninhabited / commercial edge case)
* highest-AWATER20 block (water-heavy edge case)
* remainder filled with a proportional random draw, clamped to [5, 15].
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

COUNTIES_BY_STATE: dict[str, list[str]] = {
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

INPUTS: dict[str, str] = {
    "11": r"path\to\your\tl_2025_11_tabblock20\tl_2025_11_tabblock20.shp",
    "24": r"path\to\your\tl_2025_24_tabblock20\tl_2025_24_tabblock20.shp",
    "51": r"path\to\your\tl_2025_51_tabblock20\tl_2025_51_tabblock20.shp",
}

OUTPUT_DIR = Path(r"path\to\output\folder")
MANIFEST_PATH = OUTPUT_DIR / "sample_block_geoids.csv"

TARGET_TOTAL = 120
MIN_PER_COUNTY = 5
MAX_PER_COUNTY = 15
RANDOM_SEED = 42


# =============================================================================
# FIRST-RUN SAMPLING
# =============================================================================


def _stratified_picks(sub: gpd.GeoDataFrame, target_n: int) -> list[tuple[str, str]]:
    """Return up to *target_n* (GEOID20, reason) tuples from one county."""
    selected: dict[str, str] = {}

    # Stratum 1: highest POP20.
    top_pop = sub.nlargest(1, "POP20").iloc[0]
    selected[top_pop["GEOID20"]] = "max_pop"

    # Stratum 2: a zero-POP20 block if any exist.
    zero_pool = sub[(sub["POP20"] == 0) & (~sub["GEOID20"].isin(selected))]
    if not zero_pool.empty:
        pick = zero_pool.sample(n=1, random_state=RANDOM_SEED).iloc[0]
        selected[pick["GEOID20"]] = "zero_pop"

    # Stratum 3: highest AWATER20 (only if it actually has water).
    water_pool = sub[~sub["GEOID20"].isin(selected)]
    if not water_pool.empty:
        top_water = water_pool.nlargest(1, "AWATER20").iloc[0]
        if top_water["AWATER20"] > 0:
            selected[top_water["GEOID20"]] = "max_water"

    # Stratum 4: random fill.
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


def first_run(gdfs: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    """Sample blocks, write the per-state shapefiles, return the manifest."""
    targets = _allocate_targets(gdfs)
    rows: list[dict[str, str]] = []

    for state_fp, gdf in gdfs.items():
        picked: list[str] = []
        for county_fp in COUNTIES_BY_STATE[state_fp]:
            sub = gdf[gdf["COUNTYFP20"] == county_fp]
            if sub.empty:
                print(f"  [warn] no blocks for state={state_fp} county={county_fp}")
                continue
            for geoid, reason in _stratified_picks(sub, targets[(state_fp, county_fp)]):
                rows.append(
                    {
                        "GEOID20": geoid,
                        "state_fp": state_fp,
                        "county_fp": county_fp,
                        "selection_reason": reason,
                    }
                )
                picked.append(geoid)

        out_gdf = gdf[gdf["GEOID20"].isin(picked)]
        out_path = OUTPUT_DIR / f"tl_2025_{state_fp}_tabblock20_sample.shp"
        out_gdf.to_file(out_path, driver="ESRI Shapefile", index=False)
        print(f"  state {state_fp}: wrote {len(out_gdf):,} blocks -> {out_path}")

    return (
        pd.DataFrame(rows).sort_values(["state_fp", "county_fp", "GEOID20"]).reset_index(drop=True)
    )


# =============================================================================
# REPLAY
# =============================================================================


def replay(gdfs: dict[str, gpd.GeoDataFrame], manifest: pd.DataFrame) -> None:
    """Filter each state file to the manifest's GEOIDs and write outputs."""
    for state_fp, gdf in gdfs.items():
        wanted = set(manifest.loc[manifest["state_fp"] == state_fp, "GEOID20"])
        out_gdf = gdf[gdf["GEOID20"].isin(wanted)]
        missing = wanted - set(out_gdf["GEOID20"])
        if missing:
            print(f"  [warn] state {state_fp}: {len(missing)} manifest GEOIDs not found in source")
        out_path = OUTPUT_DIR / f"tl_2025_{state_fp}_tabblock20_sample.shp"
        out_gdf.to_file(out_path, driver="ESRI Shapefile", index=False)
        print(f"  state {state_fp}: wrote {len(out_gdf):,} blocks -> {out_path}")


# =============================================================================
# ENTRYPOINT
# =============================================================================


def main() -> None:
    """Read input shapefiles, then sample or replay according to manifest state."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading state shapefiles...")
    gdfs: dict[str, gpd.GeoDataFrame] = {}
    for state_fp, shp in INPUTS.items():
        gdf = gpd.read_file(shp)
        gdf["GEOID20"] = gdf["GEOID20"].astype(str)
        gdfs[state_fp] = gdf

    if MANIFEST_PATH.exists():
        print(f"Manifest found ({MANIFEST_PATH.name}) - replay mode")
        manifest = pd.read_csv(MANIFEST_PATH, dtype=str)
        replay(gdfs, manifest)
    else:
        print("No manifest - first-run stratified sampling")
        manifest = first_run(gdfs)
        manifest.to_csv(MANIFEST_PATH, index=False)
        print(f"Wrote manifest -> {MANIFEST_PATH}")
        reason_counts = manifest["selection_reason"].value_counts().to_dict()
        print(f"Selection breakdown: {reason_counts}")


if __name__ == "__main__":
    main()
