"""Wrangle CCD school membership and join enrollment to EDGE public-school points.

Standalone script: edit the config block and call ``run()`` (notebook) or run as
``python schools_prep_join.py [--input-dir ...] [--output-dir ...] [--states VA MD DC]``
(CLI). Produces a point GeoPackage (plus an attribute-only CSV companion) for the
target jurisdictions -- Virginia, Maryland, and the District of Columbia by
default -- with total enrollment plus a per-grade breakout joined to each NCES
school point.

Inputs (placed in the input directory, read straight from their distribution zips):
    - ccd_sch_052_<year>_l_1a_<date>.zip   CCD school membership (long format)
    - EDGE_GEOCODE_PUBLICSCH_<year>.zip     EDGE public-school point shapefile

Sources:
    - CCD school membership (fiscal files):
      https://nces.ed.gov/ccd/files.asp#Fiscal:2,LevelId:7,SchoolYearId:39,Page:1
    - EDGE public-school geocodes:
      https://nces.ed.gov/programs/edge/geographic/schoollocations

IMPORTANT: this expects the SCHOOL-level membership file (``ccd_sch_052``). The
district-level file (``ccd_lea_052``) is keyed on LEAID and cannot join to school
points by NCESSCH; the loader will stop with a clear message if only the LEA file
is present. Keep the geocode and membership years on the same vintage, or the join
will silently drop schools that opened or closed between the two collections.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# --- <<< EDIT ME -------------------------------------------------------------
# Set these to use them directly. Leave as None to be prompted (notebook) or to
# pass --input-dir / --output-dir on the command line.
INPUT_DIR: Path | None = None  # folder holding the two distribution zips
OUTPUT_DIR: Path | None = None  # folder for the GeoPackage + CSV outputs

OUTPUT_GPKG_NAME = "va_md_dc_schools_enrollment.gpkg"
OUTPUT_LAYER = "schools"
OUTPUT_CSV_NAME = "va_md_dc_schools_enrollment.csv"  # attribute-only companion

STATE_ABBRS = {"VA", "MD", "DC"}  # jurisdictions to keep
STATE_COL = "STATE"  # postal-abbrev column in the EDGE geocode file
OUTPUT_CRS = 6487  # NAD83(2011) / DC-MD-VA region (meters)
# --- EDIT ME >>> -------------------------------------------------------------

logger = logging.getLogger(__name__)

# CCD membership TOTAL_INDICATOR labels (whitespace-stripped before matching).
EDU_TOTAL = "Education Unit Total"
GRADE_SUBTOTAL = "Subtotal 4 - By Grade"


def _in_ipython_kernel() -> bool:
    """Return True inside a Jupyter/IPython kernel.

    There ``sys.argv`` holds the kernel launcher args (e.g. ``-f
    ...kernel.json``) rather than user CLI args.
    """
    return "ipykernel" in sys.modules or Path(sys.argv[0]).name == "ipykernel_launcher.py"


def _ensure_logging(level: int = logging.INFO) -> None:
    """Make INFO visible in both CLI and notebook sessions."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


def _prompt_path(label: str, *, must_exist: bool) -> Path:
    """Interactively ask for a path; works in a terminal and in a notebook."""
    while True:
        raw = input(f"Enter {label}: ").strip().strip('"').strip("'")
        if not raw:
            print("  a path is required")
            continue
        path = Path(raw).expanduser()
        if must_exist and not path.exists():
            print(f"  not found: {path}")
            continue
        return path


def _find_one(directory: Path, pattern: str) -> Path:
    """Return the single path in ``directory`` matching ``pattern``.

    Args:
        directory: Folder to search (non-recursive).
        pattern: Glob pattern, e.g. ``"ccd_sch_052_*.zip"``.

    Returns:
        The matching path.

    Raises:
        FileNotFoundError: If zero matches are found.
        ValueError: If more than one match is found.
    """
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files match {pattern!r}: {[m.name for m in matches]}")
    return matches[0]


def _slug(label: object) -> str:
    """Coerce a grade label into a column-safe suffix (e.g. 'Grade 1' -> 'grade_1')."""
    text = re.sub(r"[^0-9a-z]+", "_", str(label).strip().lower())
    return text.strip("_") or "unknown"


def load_school_points(
    input_dir: Path,
    *,
    state_abbrs: set[str] = STATE_ABBRS,
    state_col: str = STATE_COL,
    output_crs: int = OUTPUT_CRS,
) -> gpd.GeoDataFrame:
    """Load EDGE public-school points, filter to the target states, reproject.

    Args:
        input_dir: Folder holding the ``EDGE_GEOCODE_PUBLICSCH_*.zip`` file.
        state_abbrs: Postal abbreviations to keep.
        state_col: Postal-abbrev column in the EDGE geocode file.
        output_crs: EPSG code to reproject the points into.

    Returns:
        Point GeoDataFrame in ``output_crs`` with a string ``NCESSCH`` key.
    """
    zip_path = _find_one(input_dir, "EDGE_GEOCODE_PUBLICSCH_*.zip")
    logger.info("Reading school points from %s", zip_path.name)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        shp = _find_one(Path(tmp), "*.shp")
        gdf = gpd.read_file(shp)

    if state_col not in gdf.columns:
        raise KeyError(
            f"State column {state_col!r} not in geocode file; "
            f"available columns: {list(gdf.columns)}"
        )

    gdf["NCESSCH"] = gdf["NCESSCH"].astype(str)  # IDs are strings; preserve leading zeros
    gdf = gdf[gdf[state_col].isin(state_abbrs)].copy()
    if gdf.empty:
        raise ValueError(f"No school points matched states {sorted(state_abbrs)}")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4269)  # EDGE geocodes are NAD83 lat/lon
    gdf = gdf.to_crs(epsg=output_crs)

    logger.info("Kept %d school points across %s", len(gdf), sorted(state_abbrs))
    return gdf


def load_membership_wide(input_dir: Path) -> pd.DataFrame:
    """Load CCD school membership and reshape to one row per school.

    Output columns: ``NCESSCH``, ``enroll_total``, and one ``g_<grade>`` column per
    grade. Letter flags and negative NCES sentinels (-1/-2/-9) become NaN.

    Args:
        input_dir: Folder holding the ``ccd_sch_052_*.zip`` file.

    Returns:
        Wide enrollment DataFrame keyed on a string ``NCESSCH``.

    Raises:
        FileNotFoundError: If the school-level membership zip is absent. The
            district-level ``ccd_lea_052`` file does not satisfy this loader.
    """
    try:
        zip_path = _find_one(input_dir, "ccd_sch_052_*.zip")
    except FileNotFoundError as exc:
        lea = list(input_dir.glob("ccd_lea_052_*.zip"))
        hint = (
            " Found a district-level file (ccd_lea_052) instead; that one is keyed on "
            "LEAID and cannot join to school points. Download ccd_sch_052 for the same "
            "year."
            if lea
            else ""
        )
        raise FileNotFoundError(str(exc) + hint) from exc

    logger.info("Reading membership from %s", zip_path.name)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        csv = _find_one(Path(tmp), "*.csv")
        mem = pd.read_csv(csv, dtype=str)

    mem["NCESSCH"] = mem["NCESSCH"].astype(str)
    mem["TOTAL_INDICATOR"] = mem["TOTAL_INDICATOR"].str.strip()
    mem["count"] = pd.to_numeric(mem["STUDENT_COUNT"], errors="coerce")
    mem.loc[mem["count"] < 0, "count"] = pd.NA  # null -1/-2/-9 sentinels

    totals = (
        mem.loc[mem["TOTAL_INDICATOR"] == EDU_TOTAL]
        .groupby("NCESSCH", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "enroll_total"})
    )

    by_grade = mem.loc[mem["TOTAL_INDICATOR"] == GRADE_SUBTOTAL]
    wide = (
        by_grade.pivot_table(index="NCESSCH", columns="GRADE", values="count", aggfunc="sum")
        .rename(columns=lambda c: f"g_{_slug(c)}")
        .reset_index()
    )

    enroll = totals.merge(wide, on="NCESSCH", how="outer")
    if enroll.empty:
        raise ValueError(
            "Membership file produced no enrollment rows; check TOTAL_INDICATOR labels"
        )

    logger.info(
        "Built enrollment for %d schools (%d grade columns)", len(enroll), len(wide.columns) - 1
    )
    return enroll


def join_and_validate(points: gpd.GeoDataFrame, enroll: pd.DataFrame) -> gpd.GeoDataFrame:
    """Left-join enrollment onto points and log match diagnostics."""
    out = points.merge(enroll, on="NCESSCH", how="left")

    matched = out["enroll_total"].notna().sum()
    unmatched_pts = len(out) - matched
    orphan_enroll = (~enroll["NCESSCH"].isin(points["NCESSCH"])).sum()

    logger.info("Join: %d/%d points matched enrollment", matched, len(out))
    if unmatched_pts:
        logger.warning("%d points have no enrollment row (closed/new or ID drift)", unmatched_pts)
    if orphan_enroll:
        logger.warning("%d enrollment rows have no matching point in target states", orphan_enroll)
    return out


def run(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    states: set[str] | None = None,
    output_crs: int | None = None,
) -> gpd.GeoDataFrame:
    """Notebook entry point: wrangle, join, and write GeoPackage + CSV outputs.

    Unset args fall back to the config block, resolved at call time -- so
    ``m.INPUT_DIR = ...; m.run()`` works as expected after a plain import.

    Args:
        input_dir: Folder holding the two distribution zips.
        output_dir: Folder for the GeoPackage + CSV outputs.
        states: Postal abbreviations to keep.
        output_crs: EPSG code to reproject the points into.

    Returns:
        The joined point GeoDataFrame that was written to disk.
    """
    _ensure_logging()
    input_dir = INPUT_DIR if input_dir is None else Path(input_dir)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    states = STATE_ABBRS if states is None else states
    output_crs = OUTPUT_CRS if output_crs is None else output_crs

    # Anything still unset after arg + config block falls to an interactive prompt.
    if input_dir is None:
        input_dir = _prompt_path("input directory (holds the distribution zips)", must_exist=True)
    if output_dir is None:
        output_dir = _prompt_path("output directory", must_exist=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg = output_dir / OUTPUT_GPKG_NAME
    output_csv = output_dir / OUTPUT_CSV_NAME

    points = load_school_points(input_dir, state_abbrs=states, output_crs=output_crs)
    enroll = load_membership_wide(input_dir)
    out = join_and_validate(points, enroll)

    out.to_file(output_gpkg, layer=OUTPUT_LAYER, driver="GPKG")
    out.drop(columns="geometry").to_csv(output_csv, index=False)
    logger.info("Wrote %s (layer %r) and %s", output_gpkg.name, OUTPUT_LAYER, output_csv.name)
    return out


def main(argv: list[str] | None = None) -> None:
    """Entry point for both notebook and CLI.

    Path resolution is the same everywhere: explicit value -> config block ->
    interactive prompt. In a Jupyter/IPython kernel the launcher injects its own
    argv (``-f kernel.json``), which argparse would reject, so we skip parsing
    and let ``run()`` resolve from the config block or prompt. On the command
    line, ``--input-dir`` / ``--output-dir`` override the config; omit them (with
    config left as None) to be prompted.
    """
    _ensure_logging()
    if argv is None and _in_ipython_kernel():
        logger.info("kernel detected; resolving paths from config block or prompt")
        run()
        return

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR, help="folder holding the distribution zips"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="folder for the GeoPackage + CSV"
    )
    parser.add_argument(
        "--states",
        nargs="+",
        metavar="ABBR",
        default=sorted(STATE_ABBRS),
        help="postal abbreviations to keep (default: VA MD DC)",
    )
    parser.add_argument(
        "--crs",
        type=int,
        default=OUTPUT_CRS,
        help="EPSG code to reproject school points into (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    run(
        args.input_dir,
        args.output_dir,
        states={s.upper() for s in args.states},
        output_crs=args.crs,
    )


if __name__ == "__main__":
    main()
