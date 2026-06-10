"""Build small, ID-aligned schools fixtures from full EDGE + ELSI (+ CCD) inputs.

Edit the paths in the CONFIG block below and run from the repo root. For each
requested school type the builder:

1. Reads the EDGE geocode points (``.zip``, ``.shp``, or a bare ``.dbf`` whose
   geometry is rebuilt from LAT/LON) and filters to the target states.
2. Reads the matching ELSI table-generator export (public vs private is
   auto-detected from the preamble) and the optional CCD ``ccd_sch_052`` zip.
3. Deterministically selects a few schools present in BOTH the geocode and the
   ELSI export (``matched``), a few geocode-only points (``unmatched_point``),
   and a few ELSI-only rows (``orphan_enrollment``) -- so the downstream join is
   exercised on all three paths.
4. Writes trimmed fixtures into ``OUTPUT_DIR`` with the exact filenames the test
   suite expects, plus ``schools_fixture_manifest.csv`` documenting every pick.

Selection is sorted-deterministic (no RNG): re-running with the same inputs
reproduces equivalent fixtures. This builder pre-filters geocodes to the target
states; the committed ``tests/fixtures`` samples follow the same format and ID
alignment but deliberately retain one out-of-region row so the state filter has
something to drop.
"""

from __future__ import annotations

import csv
import tempfile
import zipfile
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.exogenous_tools import schools_prep_join as sj

# =============================================================================
# CONFIG
# =============================================================================

#: Folder holding the full EDGE geocodes, ELSI exports, and (optional) CCD zip.
INPUT_DIR: Final[Path] = Path(r"PATH\TO\SCHOOLS\FIXTURE\INPUTS")  # <<< EDIT ME

#: Where to write the trimmed fixtures (the repo's test fixtures directory).
OUTPUT_DIR: Final[Path] = Path("tests/fixtures")  # <<< EDIT ME if needed

SCHOOL_TYPES_TO_BUILD: Final[tuple[str, ...]] = ("public", "private", "postsec")
STATES: Final[set[str]] = {"VA", "MD", "DC"}

N_MATCHED: Final[int] = 3  # schools in both geocode and ELSI
N_UNMATCHED_POINTS: Final[int] = 1  # geocode points with no enrollment row
N_ORPHAN_ENROLLMENT: Final[int] = 1  # ELSI rows with no matching point

MANIFEST_FILENAME: Final[str] = "schools_fixture_manifest.csv"

#: Output filenames per school type (matches what test_schools_prep_join expects).
GEOCODE_OUT: Final[dict[str, str]] = {
    "public": "EDGE_GEOCODE_PUBLICSCH_1920_sample.zip",
    "private": "EDGE_GEOCODE_PRIVATESCH_1920_sample.zip",
    "postsec": "EDGE_GEOCODE_POSTSEC_1920_sample.zip",
}
ELSI_OUT: Final[dict[str, str]] = {
    "public": "ELSI_csv_export_public_1920_sample.csv",
    "private": "ELSI_csv_export_private_1920_sample.csv",
}
CCD_OUT: Final[str] = "ccd_sch_052_1920_sample.zip"
IPEDS_OUT: Final[str] = "effy2019_sample.csv"


# =============================================================================
# READERS
# =============================================================================


def read_geocode_points(input_dir: Path, st: sj.SchoolType) -> gpd.GeoDataFrame:
    """Read EDGE points for ``st`` from a zip, a shapefile, or a bare .dbf."""
    zips = sorted(input_dir.glob(st.geocode_glob))
    if zips:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zips[0]) as zf:
                zf.extractall(tmp)
            shp = next(Path(tmp).rglob("*.shp"))
            gdf = gpd.read_file(shp)
    else:
        glob_stem = st.geocode_glob.replace("_*.zip", "*")
        shps = sorted(input_dir.glob(f"{glob_stem}.shp"))
        dbfs = sorted(input_dir.glob(f"{glob_stem}.dbf"))
        if shps:
            gdf = gpd.read_file(shps[0])
        elif dbfs:
            gdf = gpd.read_file(dbfs[0])
            geom = [Point(xy) for xy in zip(gdf["LON"].astype(float), gdf["LAT"].astype(float))]
            gdf = gpd.GeoDataFrame(
                gdf.drop(columns="geometry", errors="ignore"), geometry=geom, crs="EPSG:4269"
            )
        else:
            raise FileNotFoundError(f"No EDGE geocode for {st.name} in {input_dir}")

    gdf[st.id_col] = gdf[st.id_col].astype(str).str.strip().str.zfill(st.id_width)
    gdf = gdf[gdf[sj.STATE_COL].isin(STATES)].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4269)
    return gdf


def read_elsi_rows(input_dir: Path, st: sj.SchoolType) -> tuple[list[list[str]], int, int]:
    """Return (all_csv_rows, header_index, key_column_index) for the ELSI export."""
    path = sj._find_elsi_csv(input_dir, st)
    if path is None:
        raise FileNotFoundError(f"No {st.elsi_kind} ELSI export in {input_dir}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    header_labels = {"School Name", "Private School Name"}
    header_idx = next(i for i, r in enumerate(rows) if r and r[0].strip() in header_labels)
    key_idx = next(i for i, c in enumerate(rows[header_idx]) if st.elsi_id_substr in c)
    return rows, header_idx, key_idx


# =============================================================================
# WRITERS
# =============================================================================


def write_geocode_zip(gdf: gpd.GeoDataFrame, out_zip: Path) -> None:
    """Write a GeoDataFrame as a zipped point shapefile at the zip's top level."""
    stem = out_zip.with_suffix("").name
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gdf.to_file(tmp / f"{stem}.shp", driver="ESRI Shapefile", index=False)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for comp in sorted(tmp.glob(f"{stem}.*")):
                zf.write(comp, comp.name)


def write_elsi_csv(
    rows: list[list[str]],
    header_idx: int,
    key_idx: int,
    id_width: int,
    keep_keys: set[str],
    out_csv: Path,
) -> None:
    """Re-emit an ELSI export keeping its preamble/footer and only ``keep_keys`` rows.

    Original key cells are preserved verbatim; selection compares the zero-padded
    form (``keep_keys`` is zero-padded) so leading-zero IDs still match.
    """
    preamble, header = rows[:header_idx], rows[header_idx]
    body = rows[header_idx + 1 :]

    def is_data(row: list[str]) -> bool:
        return len(row) > key_idx and row[key_idx].strip() not in {"", "nan"}

    kept_data = [r for r in body if is_data(r) and r[key_idx].strip().zfill(id_width) in keep_keys]
    footer = [r for r in body if not is_data(r)]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([*preamble, header, *kept_data, *footer])


def read_ipeds(input_dir: Path) -> pd.DataFrame:
    """Read the IPEDS EFFY file (csv or xlsx) as strings, restricted to used levels."""
    path = sj._find_ipeds_file(input_dir)
    if path is None:
        raise FileNotFoundError(f"No IPEDS EFFY file (effy*.csv/.xlsx) in {input_dir}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df[sj.IPEDS_ID_COL] = df[sj.IPEDS_ID_COL].astype(str).str.strip().str.zfill(6)
    return df[df[sj.IPEDS_LEVEL_COL].isin(sj.IPEDS_LEVEL_MAP)].copy()


def build_postsec(st: sj.SchoolType, manifest: list[dict[str, str]]) -> None:
    """Build college fixtures (EDGE POSTSEC geocode + trimmed IPEDS EFFY)."""
    print(f"\n=== {st.name} ===")
    points = read_geocode_points(INPUT_DIR, st)
    effy = read_ipeds(INPUT_DIR)

    point_ids = set(points[st.id_col])
    effy_ids = set(effy[sj.IPEDS_ID_COL])

    matched = sorted(point_ids & effy_ids)[:N_MATCHED]
    unmatched_points = sorted(point_ids - effy_ids)[:N_UNMATCHED_POINTS]
    orphans = sorted(effy_ids - point_ids)[:N_ORPHAN_ENROLLMENT]
    print(f"  matched={len(matched)} unmatched_point={len(unmatched_points)} orphan={len(orphans)}")

    out_points = points[points[st.id_col].isin(set(matched) | set(unmatched_points))].copy()
    write_geocode_zip(out_points, OUTPUT_DIR / GEOCODE_OUT["postsec"])

    keep_ids = set(matched) | set(orphans)
    out_effy = effy[effy[sj.IPEDS_ID_COL].isin(keep_ids)].sort_values(
        [sj.IPEDS_ID_COL, sj.IPEDS_LEVEL_COL]
    )
    out_effy.to_csv(OUTPUT_DIR / IPEDS_OUT, index=False)
    print(f"  wrote {GEOCODE_OUT['postsec']} ({len(out_points)} pts) and {IPEDS_OUT}")

    for key in matched:
        manifest.append({"school_type": st.name, "id": key, "role": "matched"})
    for key in unmatched_points:
        manifest.append({"school_type": st.name, "id": key, "role": "unmatched_point"})
    for key in orphans:
        manifest.append({"school_type": st.name, "id": key, "role": "orphan_enrollment"})


def trim_ccd_zip(input_dir: Path, keep_ids: set[str], out_zip: Path) -> bool:
    """Filter the CCD ``ccd_sch_052`` membership zip to ``keep_ids``. Returns written?"""
    ccd = sorted(input_dir.glob("ccd_sch_052_*.zip"))
    if not ccd:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(ccd[0]) as zf:
            zf.extractall(tmp)
        src = next(Path(tmp).rglob("*.csv"))
        df = pd.read_csv(src, dtype=str)
        df["NCESSCH"] = df["NCESSCH"].astype(str).str.zfill(12)
        kept = df[df["NCESSCH"].isin(keep_ids)]
        out_name = "ccd_sch_052_1920_l_1a_sample.csv"
        out_path = Path(tmp) / out_name
        kept.to_csv(out_path, index=False)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, out_name)
    print(f"  [ccd] kept {len(kept):,} rows for {kept['NCESSCH'].nunique()} schools")
    return True


# =============================================================================
# DRIVER
# =============================================================================


def build_school_type(st: sj.SchoolType, manifest: list[dict[str, str]]) -> set[str]:
    """Build fixtures for one school type; return the geocode IDs kept (for CCD)."""
    print(f"\n=== {st.name} ===")
    points = read_geocode_points(INPUT_DIR, st)
    rows, header_idx, key_idx = read_elsi_rows(INPUT_DIR, st)

    point_ids = set(points[st.id_col])
    elsi_ids = [
        r[key_idx].strip().zfill(st.id_width)
        for r in rows[header_idx + 1 :]
        if len(r) > key_idx and r[key_idx].strip() not in {"", "nan"}
    ]
    elsi_set = set(elsi_ids)

    matched = sorted(point_ids & elsi_set)[:N_MATCHED]
    unmatched_points = sorted(point_ids - elsi_set)[:N_UNMATCHED_POINTS]
    orphans = sorted(elsi_set - point_ids)[:N_ORPHAN_ENROLLMENT]
    print(f"  matched={len(matched)} unmatched_point={len(unmatched_points)} orphan={len(orphans)}")

    keep_point_ids = set(matched) | set(unmatched_points)
    keep_elsi_keys = set(matched) | set(orphans)

    out_points = points[points[st.id_col].isin(keep_point_ids)].copy()
    write_geocode_zip(out_points, OUTPUT_DIR / GEOCODE_OUT[st.name])
    write_elsi_csv(
        rows, header_idx, key_idx, st.id_width, keep_elsi_keys, OUTPUT_DIR / ELSI_OUT[st.name]
    )
    print(f"  wrote {GEOCODE_OUT[st.name]} ({len(out_points)} pts) and {ELSI_OUT[st.name]}")

    for key in matched:
        manifest.append({"school_type": st.name, "id": key, "role": "matched"})
    for key in unmatched_points:
        manifest.append({"school_type": st.name, "id": key, "role": "unmatched_point"})
    for key in orphans:
        manifest.append({"school_type": st.name, "id": key, "role": "orphan_enrollment"})
    return keep_point_ids


def main() -> int:
    """Entry point."""
    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}")
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    public_keep: set[str] = set()
    for name in SCHOOL_TYPES_TO_BUILD:
        st = sj.SCHOOL_TYPES[name]
        if name == "postsec":
            build_postsec(st, manifest)
            continue
        keep = build_school_type(st, manifest)
        if name == "public":
            public_keep = keep

    if public_keep:
        print("\n=== CCD (public) ===")
        if not trim_ccd_zip(INPUT_DIR, public_keep, OUTPUT_DIR / CCD_OUT):
            print("  no ccd_sch_052_*.zip found; skipping CCD fixture")

    pd.DataFrame(manifest).to_csv(OUTPUT_DIR / MANIFEST_FILENAME, index=False)
    print(f"\nWrote manifest -> {OUTPUT_DIR / MANIFEST_FILENAME} ({len(manifest)} picks)")
    return 0


if __name__ == "__main__":
    _code = main()
    if _code != 0:
        raise SystemExit(_code)
