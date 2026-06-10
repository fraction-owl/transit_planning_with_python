"""Join school enrollment to EDGE school-location points (public or private).

Standalone script: edit the config block and call ``run()`` (notebook) or run as
``python schools_prep_join.py [--input-dir ...] [--output-dir ...]
[--school-type public|private|both] [--enrollment-source auto|ccd|elsi]
[--states VA MD DC]`` (CLI). Produces a point GeoPackage (plus an attribute-only
CSV companion) for the target jurisdictions -- Virginia, Maryland, and the
District of Columbia by default -- with total enrollment plus a per-grade
breakout joined to each EDGE school point.

Two enrollment sources are supported and auto-detected (``--enrollment-source``):
    - CCD school membership (``ccd_sch_052_<year>_l_1a_<date>.zip``), long format,
      keyed on NCESSCH. Public schools only. Richer per-grade detail.
    - ELSI table-generator exports (``ELSI_csv_export_*.csv``), wide format, with
      a preamble/footer and total + Grades 1-8 / Grades 9-12 columns. Available
      for both public and private schools. Use this when a full CCD download is
      impractical (the ELSI generator lets you pre-filter to a few states).

Inputs (placed in the input directory, read straight from their distribution form):
    - EDGE_GEOCODE_PUBLICSCH_<year>.zip      EDGE public-school point shapefile
    - EDGE_GEOCODE_PRIVATESCH_<year>.zip     EDGE private-school point shapefile
    - ccd_sch_052_<year>_l_1a_<date>.zip     CCD school membership (public, long)
    - ELSI_csv_export_*.csv                  ELSI export (public and/or private)

Sources:
    - CCD school membership (fiscal files):
      https://nces.ed.gov/ccd/files.asp#Fiscal:2,LevelId:7,SchoolYearId:39,Page:1
    - ELSI table generator: https://nces.ed.gov/ccd/elsi/tableGenerator.aspx
    - EDGE school geocodes:
      https://nces.ed.gov/programs/edge/geographic/schoollocations

Notes:
    - The public CCD path expects the SCHOOL-level membership file (``ccd_sch_052``).
      The district-level file (``ccd_lea_052``) is keyed on LEAID and cannot join to
      school points by NCESSCH; the loader stops with a clear message if only the LEA
      file is present.
    - Keep the geocode and enrollment years on the same vintage, or the join will
      silently drop schools that opened or closed between the two collections.
    - Postsecondary (college) geocodes carry no enrollment field, so they are out of
      scope here; the EDGE_GEOCODE_POSTSEC file would need an IPEDS join for counts.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd

# --- <<< EDIT ME -------------------------------------------------------------
# Set these to use them directly. Leave as None to be prompted (notebook) or to
# pass --input-dir / --output-dir on the command line.
INPUT_DIR: Path | None = None  # folder holding the distribution files
OUTPUT_DIR: Path | None = None  # folder for the GeoPackage + CSV outputs

# Output names are templated on the school type ("public" / "private").
OUTPUT_GPKG_TEMPLATE = "va_md_dc_{school_type}_schools_enrollment.gpkg"
OUTPUT_CSV_TEMPLATE = "va_md_dc_{school_type}_schools_enrollment.csv"  # attribute-only
OUTPUT_LAYER = "schools"

DEFAULT_SCHOOL_TYPE = "public"  # "public" or "private"
DEFAULT_ENROLLMENT_SOURCE = "auto"  # "auto" | "ccd" | "elsi"

STATE_ABBRS = {"VA", "MD", "DC"}  # jurisdictions to keep
STATE_COL = "STATE"  # postal-abbrev column in the EDGE geocode file
OUTPUT_CRS = 6487  # NAD83(2011) / DC-MD-VA region (meters)
# --- EDIT ME >>> -------------------------------------------------------------

logger = logging.getLogger(__name__)

# CCD membership TOTAL_INDICATOR labels (whitespace-stripped before matching).
EDU_TOTAL = "Education Unit Total"
GRADE_SUBTOTAL = "Subtotal 4 - By Grade"


@dataclass(frozen=True)
class SchoolType:
    """Per-school-type wiring for geocodes and the two enrollment sources.

    Attributes:
        name: Short key (``"public"`` / ``"private"``) used in CLI args and outputs.
        geocode_glob: Glob for the EDGE point-shapefile zip in the input directory.
        id_col: Join key column in the EDGE point file (and the enrollment output).
        id_width: Zero-pad width for the ID, guarding against leading-zero loss in
            CSV round-trips. Alphanumeric IDs at this width are left unchanged.
        supports_ccd: Whether a CCD ``ccd_sch_052`` membership file applies (public
            only; the CCD fiscal collection has no private-school equivalent).
        elsi_kind: Phrase identifying the ELSI export ("Public School" /
            "Private School"), matched against the export's preamble.
        elsi_id_substr: Substring identifying the ELSI key column for this type.
        elsi_total_substrs: Candidate substrings for the ELSI total-enrollment
            column, tried in order (first match wins).
    """

    name: str
    geocode_glob: str
    id_col: str
    id_width: int
    supports_ccd: bool
    elsi_kind: str
    elsi_id_substr: str
    elsi_total_substrs: tuple[str, ...] = field(default_factory=tuple)


SCHOOL_TYPES: dict[str, SchoolType] = {
    "public": SchoolType(
        name="public",
        geocode_glob="EDGE_GEOCODE_PUBLICSCH_*.zip",
        id_col="NCESSCH",
        id_width=12,
        supports_ccd=True,
        elsi_kind="Public School",
        elsi_id_substr="School ID (12-digit)",
        elsi_total_substrs=(
            "Total Students All Grades (Excludes AE)",
            "Total Students All Grades (Includes AE)",
        ),
    ),
    "private": SchoolType(
        name="private",
        geocode_glob="EDGE_GEOCODE_PRIVATESCH_*.zip",
        id_col="PPIN",
        id_width=8,
        supports_ccd=False,
        elsi_kind="Private School",
        elsi_id_substr="School ID - NCES Assigned",
        elsi_total_substrs=(
            "Total Students (Ungraded & PK-12)",
            "Total Students (Ungraded & K-12)",
        ),
    ),
}


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


def resolve_school_type(school_type: str | SchoolType) -> SchoolType:
    """Return the :class:`SchoolType` for a name (or pass one through).

    Args:
        school_type: Either a registry key (``"public"``/``"private"``) or an
            already-resolved :class:`SchoolType`.

    Returns:
        The matching :class:`SchoolType`.

    Raises:
        KeyError: If ``school_type`` is an unknown name.
    """
    if isinstance(school_type, SchoolType):
        return school_type
    try:
        return SCHOOL_TYPES[school_type]
    except KeyError as exc:
        raise KeyError(
            f"Unknown school type {school_type!r}; choose from {sorted(SCHOOL_TYPES)}"
        ) from exc


def load_school_points(
    input_dir: Path,
    school_type: str | SchoolType = DEFAULT_SCHOOL_TYPE,
    *,
    state_abbrs: set[str] = STATE_ABBRS,
    state_col: str = STATE_COL,
    output_crs: int = OUTPUT_CRS,
) -> gpd.GeoDataFrame:
    """Load EDGE school points, filter to the target states, reproject.

    Args:
        input_dir: Folder holding the EDGE geocode zip for ``school_type``.
        school_type: ``"public"``/``"private"`` (or a :class:`SchoolType`).
        state_abbrs: Postal abbreviations to keep.
        state_col: Postal-abbrev column in the EDGE geocode file.
        output_crs: EPSG code to reproject the points into.

    Returns:
        Point GeoDataFrame in ``output_crs`` with a string ID key.
    """
    st = resolve_school_type(school_type)
    zip_path = _find_one(input_dir, st.geocode_glob)
    logger.info("Reading %s school points from %s", st.name, zip_path.name)

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
    if st.id_col not in gdf.columns:
        raise KeyError(
            f"ID column {st.id_col!r} not in geocode file; "
            f"available columns: {list(gdf.columns)}"
        )

    # IDs are strings; preserve leading zeros and pad to the expected width.
    gdf[st.id_col] = gdf[st.id_col].astype(str).str.strip().str.zfill(st.id_width)
    gdf = gdf[gdf[state_col].isin(state_abbrs)].copy()
    if gdf.empty:
        raise ValueError(f"No school points matched states {sorted(state_abbrs)}")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4269)  # EDGE geocodes are NAD83 lat/lon
    gdf = gdf.to_crs(epsg=output_crs)

    logger.info("Kept %d %s school points across %s", len(gdf), st.name, sorted(state_abbrs))
    return gdf


def _load_ccd_long(zip_path: Path, id_col: str) -> pd.DataFrame:
    """Reshape a CCD ``ccd_sch_052`` long-format membership zip to one row per school.

    Output columns: ``id_col``, ``enroll_total``, and one ``g_<grade>`` column per
    grade. Letter flags and negative NCES sentinels (-1/-2/-9) become NaN.
    """
    logger.info("Reading CCD membership from %s", zip_path.name)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        csv = _find_one(Path(tmp), "*.csv")
        mem = pd.read_csv(csv, dtype=str)

    mem[id_col] = mem[id_col].astype(str)
    mem["TOTAL_INDICATOR"] = mem["TOTAL_INDICATOR"].str.strip()
    mem["count"] = pd.to_numeric(mem["STUDENT_COUNT"], errors="coerce")
    mem.loc[mem["count"] < 0, "count"] = pd.NA  # null -1/-2/-9 sentinels

    totals = (
        mem.loc[mem["TOTAL_INDICATOR"] == EDU_TOTAL]
        .groupby(id_col, as_index=False)["count"]
        .sum()
        .rename(columns={"count": "enroll_total"})
    )

    by_grade = mem.loc[mem["TOTAL_INDICATOR"] == GRADE_SUBTOTAL]
    wide = (
        by_grade.pivot_table(index=id_col, columns="GRADE", values="count", aggfunc="sum")
        .rename(columns=lambda c: f"g_{_slug(c)}")
        .reset_index()
    )

    enroll = totals.merge(wide, on=id_col, how="outer")
    if enroll.empty:
        raise ValueError(
            "Membership file produced no enrollment rows; check TOTAL_INDICATOR labels"
        )
    logger.info(
        "Built CCD enrollment for %d schools (%d grade columns)",
        len(enroll),
        len(wide.columns) - 1,
    )
    return enroll


def _read_elsi_table(path: Path) -> pd.DataFrame:
    """Read an ELSI export, skipping its preamble and trailing footnote rows.

    ELSI table-generator CSVs wrap the data in a banner/preamble (the title, the
    source URL, the applied filters) and a footer (data-source citation and the
    ``† – ‡`` legend). The real header row is the first whose first cell is
    ``"School Name"`` or ``"Private School Name"``; footer/blank rows are left in
    place here and dropped downstream once the key column is known.

    Args:
        path: Path to the ELSI ``*.csv`` export.

    Returns:
        DataFrame with the export's columns and every data + footer row.

    Raises:
        ValueError: If no recognizable header row is found.
    """
    # The preamble/footer rows are ragged (often a single cell), so scan with the
    # csv module first -- pd.read_csv(header=None) would error on the row width
    # mismatch -- then re-read from the located header with the fast C engine.
    header_idx: int | None = None
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if row and row[0].strip() in {"School Name", "Private School Name"}:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"No ELSI header row ('School Name'...) found in {path.name}")
    return pd.read_csv(path, skiprows=header_idx, dtype=str, encoding="utf-8-sig")


def _match_col(columns: list[str], substrs: tuple[str, ...] | str) -> str:
    """Return the first column whose name contains one of ``substrs`` (in order).

    Args:
        columns: Column names to search.
        substrs: One substring or a tuple of candidates, tried in order.

    Returns:
        The matching column name.

    Raises:
        KeyError: If no column matches any candidate.
    """
    candidates = (substrs,) if isinstance(substrs, str) else substrs
    for needle in candidates:
        for col in columns:
            if needle in col:
                return col
    raise KeyError(f"No column matching any of {candidates!r} in {columns}")


def _elsi_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce an ELSI count column to numeric, nulling the ``† – ‡`` symbols.

    Thousands separators are stripped; any non-numeric token (including the ELSI
    not-applicable/missing/quality symbols) becomes NaN.
    """
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _load_elsi_wide(path: Path, school_type: SchoolType) -> pd.DataFrame:
    """Parse an ELSI export into the shared wide enrollment schema.

    Output columns: ``school_type.id_col``, ``enroll_total``, and one ``g_<band>``
    column per ELSI grade band (e.g. ``g_grades_1_8``, ``g_grades_9_12``). Column
    names are matched by substring so the same code reads any vintage (the year
    suffix on each ELSI header is ignored).

    Args:
        path: Path to the ELSI ``*.csv`` export.
        school_type: The resolved :class:`SchoolType` (drives column matching).

    Returns:
        Wide enrollment DataFrame keyed on a zero-padded string ID.
    """
    logger.info("Reading %s ELSI export from %s", school_type.name, path.name)
    df = _read_elsi_table(path)
    cols = list(df.columns)

    id_src = _match_col(cols, school_type.elsi_id_substr)
    total_src = _match_col(cols, school_type.elsi_total_substrs)
    grade_srcs = [c for c in cols if "Grades" in c and "Students" in c and "Total" not in c]

    ids = df[id_src].astype(str).str.strip()
    keep = ids.ne("") & ids.str.lower().ne("nan")  # drop blank + footnote rows

    out = pd.DataFrame({school_type.id_col: ids[keep].str.zfill(school_type.id_width)})
    out["enroll_total"] = _elsi_to_numeric(df[total_src][keep])
    for col in grade_srcs:
        label = re.sub(r"\s*\[.*$", "", col)  # drop the "[Public School] 2019-20" tail
        label = re.sub(r"\s*students?$", "", label, flags=re.IGNORECASE).strip()
        out[f"g_{_slug(label)}"] = _elsi_to_numeric(df[col][keep])

    out = (
        out.groupby(school_type.id_col, as_index=False).first()
        if out[school_type.id_col].duplicated().any()
        else out.reset_index(drop=True)
    )
    if out.empty:
        raise ValueError(f"ELSI export {path.name} produced no enrollment rows")
    logger.info(
        "Built ELSI enrollment for %d %s schools (%d grade columns)",
        len(out),
        school_type.name,
        len(grade_srcs),
    )
    return out


def _find_elsi_csv(input_dir: Path, school_type: SchoolType) -> Path | None:
    """Return the ELSI export CSV for ``school_type``, or None if absent.

    Disambiguates public vs private exports by the "This is a ... based table"
    line in the preamble, so both can sit in the same folder.
    """
    marker = f"This is a {school_type.elsi_kind} based table"
    for csv_path in sorted(input_dir.glob("*.csv")):
        try:
            head = "".join(csv_path.open(encoding="utf-8-sig").readlines()[:10])
        except OSError:  # pragma: no cover - unreadable file
            continue
        if head.startswith("ELSI Export") and marker in head:
            return csv_path
    return None


def load_enrollment_wide(
    input_dir: Path,
    school_type: str | SchoolType = DEFAULT_SCHOOL_TYPE,
    *,
    source: str = DEFAULT_ENROLLMENT_SOURCE,
) -> pd.DataFrame:
    """Load enrollment from a CCD membership zip or an ELSI export, auto-detected.

    With ``source="auto"`` a CCD ``ccd_sch_052`` zip wins when present (it carries
    the finer per-grade detail and is public-only); otherwise the matching ELSI
    export is used. ``source="ccd"`` or ``"elsi"`` forces one path.

    Args:
        input_dir: Folder holding the enrollment file(s).
        school_type: ``"public"``/``"private"`` (or a :class:`SchoolType`).
        source: ``"auto"`` | ``"ccd"`` | ``"elsi"``.

    Returns:
        Wide enrollment DataFrame keyed on ``school_type.id_col``.

    Raises:
        ValueError: If ``source`` is not one of the accepted values, or ``"ccd"``
            is requested for a school type with no CCD collection.
        FileNotFoundError: If the requested (or any) enrollment source is missing.
    """
    st = resolve_school_type(school_type)
    if source not in {"auto", "ccd", "elsi"}:
        raise ValueError(f"source must be 'auto', 'ccd', or 'elsi'; got {source!r}")
    if source == "ccd" and not st.supports_ccd:
        raise ValueError(f"No CCD membership collection exists for {st.name} schools")

    ccd_zips = sorted(input_dir.glob("ccd_sch_052_*.zip")) if st.supports_ccd else []
    elsi_csv = _find_elsi_csv(input_dir, st)

    chosen = source
    if source == "auto":
        chosen = "ccd" if ccd_zips else "elsi"

    if chosen == "ccd":
        if not ccd_zips:
            lea = list(input_dir.glob("ccd_lea_052_*.zip"))
            hint = (
                " Found a district-level file (ccd_lea_052) instead; that one is keyed on "
                "LEAID and cannot join to school points. Download ccd_sch_052 for the same "
                "year."
                if lea
                else ""
            )
            raise FileNotFoundError(
                f"No CCD membership file (ccd_sch_052_*.zip) in {input_dir}." + hint
            )
        return _load_ccd_long(_find_one(input_dir, "ccd_sch_052_*.zip"), st.id_col)

    if elsi_csv is None:
        raise FileNotFoundError(
            f"No enrollment source for {st.name} schools in {input_dir}: expected a "
            f"{st.elsi_kind} ELSI export (ELSI_csv_export_*.csv)"
            + (" or ccd_sch_052_*.zip" if st.supports_ccd else "")
        )
    return _load_elsi_wide(elsi_csv, st)


def join_and_validate(
    points: gpd.GeoDataFrame,
    enroll: pd.DataFrame,
    id_col: str = "NCESSCH",
) -> gpd.GeoDataFrame:
    """Left-join enrollment onto points and log match diagnostics."""
    out = points.merge(enroll, on=id_col, how="left")

    matched = out["enroll_total"].notna().sum()
    unmatched_pts = len(out) - matched
    orphan_enroll = (~enroll[id_col].isin(points[id_col])).sum()

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
    school_type: str | SchoolType = DEFAULT_SCHOOL_TYPE,
    enrollment_source: str = DEFAULT_ENROLLMENT_SOURCE,
    states: set[str] | None = None,
    output_crs: int | None = None,
) -> gpd.GeoDataFrame:
    """Notebook entry point: wrangle, join, and write GeoPackage + CSV outputs.

    Unset path args fall back to the config block, resolved at call time -- so
    ``m.INPUT_DIR = ...; m.run()`` works as expected after a plain import.

    Args:
        input_dir: Folder holding the distribution files.
        output_dir: Folder for the GeoPackage + CSV outputs.
        school_type: ``"public"``/``"private"`` (or a :class:`SchoolType`).
        enrollment_source: ``"auto"`` | ``"ccd"`` | ``"elsi"``.
        states: Postal abbreviations to keep.
        output_crs: EPSG code to reproject the points into.

    Returns:
        The joined point GeoDataFrame that was written to disk.
    """
    _ensure_logging()
    st = resolve_school_type(school_type)
    input_dir = INPUT_DIR if input_dir is None else Path(input_dir)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    states = STATE_ABBRS if states is None else states
    output_crs = OUTPUT_CRS if output_crs is None else output_crs

    # Anything still unset after arg + config block falls to an interactive prompt.
    if input_dir is None:
        input_dir = _prompt_path("input directory (holds the distribution files)", must_exist=True)
    if output_dir is None:
        output_dir = _prompt_path("output directory", must_exist=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg = output_dir / OUTPUT_GPKG_TEMPLATE.format(school_type=st.name)
    output_csv = output_dir / OUTPUT_CSV_TEMPLATE.format(school_type=st.name)

    points = load_school_points(input_dir, st, state_abbrs=states, output_crs=output_crs)
    enroll = load_enrollment_wide(input_dir, st, source=enrollment_source)
    out = join_and_validate(points, enroll, id_col=st.id_col)

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
    line, flags override the config; omit them (with config left as None) to be
    prompted. ``--school-type both`` processes public then private.
    """
    _ensure_logging()
    if argv is None and _in_ipython_kernel():
        logger.info("kernel detected; resolving paths from config block or prompt")
        run(school_type=DEFAULT_SCHOOL_TYPE, enrollment_source=DEFAULT_ENROLLMENT_SOURCE)
        return

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR, help="folder holding the distribution files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="folder for the GeoPackage + CSV"
    )
    parser.add_argument(
        "--school-type",
        choices=[*SCHOOL_TYPES, "both"],
        default=DEFAULT_SCHOOL_TYPE,
        help="school type to process (default: %(default)s)",
    )
    parser.add_argument(
        "--enrollment-source",
        choices=["auto", "ccd", "elsi"],
        default=DEFAULT_ENROLLMENT_SOURCE,
        help="enrollment source; auto prefers CCD when present (default: %(default)s)",
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

    types = list(SCHOOL_TYPES) if args.school_type == "both" else [args.school_type]
    for type_name in types:
        run(
            args.input_dir,
            args.output_dir,
            school_type=type_name,
            enrollment_source=args.enrollment_source,
            states={s.upper() for s in args.states},
            output_crs=args.crs,
        )


if __name__ == "__main__":
    main()
