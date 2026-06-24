"""Tests for scripts/national_data_tools/schools_prep_join_gpd.py.

Fixtures (in ``tests/fixtures``) are small, ID-aligned samples of the 2019-20
NCES collections this script consumes:

    - ``EDGE_GEOCODE_PUBLICSCH_1920_sample.zip``   EDGE public-school points (5)
    - ``EDGE_GEOCODE_PRIVATESCH_1920_sample.zip``  EDGE private-school points (5)
    - ``EDGE_GEOCODE_POSTSEC_1920_sample.zip``     EDGE college points (5)
    - ``ELSI_csv_export_public_1920_sample.csv``   ELSI public enrollment export
    - ``ELSI_csv_export_private_1920_sample.csv``  ELSI private enrollment export
    - ``ccd_sch_052_1920_sample.zip``              CCD public membership (long)
    - ``effy2019_sample.csv``                      IPEDS 12-month enrollment (colleges)

Enrollment keys are aligned to the EDGE point IDs so the join exercises matched,
unmatched-point, and orphan-enrollment paths. The geocode zips were rebuilt from
the EDGE ``.dbf`` distribution samples (geometry reconstructed from LAT/LON); the
ELSI/CCD values are illustrative, while the IPEDS rows are real 2019 EFFY records
for the colleges that overlap the geocode. Regenerate larger fixtures from full
downloads with ``dev_tools/build_schools_fixtures.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from scripts.national_data_tools import schools_prep_join_gpd as mod

FIXTURE_DIR = Path("tests/fixtures")
SCHOOL_FIXTURES = (
    "EDGE_GEOCODE_PUBLICSCH_1920_sample.zip",
    "EDGE_GEOCODE_PRIVATESCH_1920_sample.zip",
    "EDGE_GEOCODE_POSTSEC_1920_sample.zip",
    "ELSI_csv_export_public_1920_sample.csv",
    "ELSI_csv_export_private_1920_sample.csv",
    "ccd_sch_052_1920_sample.zip",
    "effy2019_sample.csv",
)

# Points kept after the VA/MD/DC filter (each geocode sample also has one AL row).
PUBLIC_POINT_IDS = {"510267002843", "240051001208", "240051001211", "110000500377"}
PRIVATE_POINT_IDS = {"00253029", "BB181325", "BB181326", "00735363"}
POSTSEC_POINT_IDS = {"131283", "232186", "163347", "163426"}

# Public enrollment overlap: ELSI omits the DC point and adds one orphan.
PUBLIC_ELSI_MATCHED = {"510267002843", "240051001208", "240051001211"}
PUBLIC_ELSI_ORPHAN = "519999000099"
PRIVATE_ELSI_MATCHED = {"00253029", "BB181325", "BB181326"}
PRIVATE_ELSI_ORPHAN = "BB999999"
# IPEDS overlap: EFFY omits the MD point 163347 and adds orphan 100663.
POSTSEC_IPEDS_MATCHED = {"131283", "232186", "163426"}
POSTSEC_IPEDS_ORPHAN = "100663"


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture()
def staged_dir(tmp_path: Path) -> Path:
    """Stage every school fixture in an isolated input directory."""
    staged = tmp_path / "schools_in"
    staged.mkdir()
    for name in SCHOOL_FIXTURES:
        shutil.copy(FIXTURE_DIR / name, staged / name)
    return staged


@pytest.fixture()
def elsi_only_dir(tmp_path: Path) -> Path:
    """Stage geocodes + ELSI exports but no CCD zip (the user's real workflow)."""
    staged = tmp_path / "elsi_in"
    staged.mkdir()
    for name in SCHOOL_FIXTURES:
        if name.startswith("ccd_"):
            continue
        shutil.copy(FIXTURE_DIR / name, staged / name)
    return staged


# =============================================================================
# resolve_school_type
# =============================================================================


def test_resolve_school_type_by_name() -> None:
    assert mod.resolve_school_type("public").id_col == "NCESSCH"
    assert mod.resolve_school_type("private").id_col == "PPIN"
    assert mod.resolve_school_type("postsec").id_col == "UNITID"


def test_resolve_school_type_passes_through_instance() -> None:
    st = mod.SCHOOL_TYPES["public"]
    assert mod.resolve_school_type(st) is st


def test_resolve_school_type_unknown_raises() -> None:
    with pytest.raises(KeyError):
        mod.resolve_school_type("charter")


# =============================================================================
# _slug
# =============================================================================


def test_slug_normalizes_grade_label() -> None:
    assert mod._slug("Grade 1") == "grade_1"
    assert mod._slug("Grades 9-12") == "grades_9_12"


def test_slug_empty_falls_back_to_unknown() -> None:
    assert mod._slug("   ") == "unknown"


# =============================================================================
# _match_col
# =============================================================================


def test_match_col_returns_first_candidate_in_order() -> None:
    cols = ["Total Students All Grades (Includes AE) 2019-20"]
    assert mod._match_col(cols, ("Excludes AE", "Includes AE")) == cols[0]


def test_match_col_accepts_single_string() -> None:
    assert mod._match_col(["School ID (12-digit) X"], "School ID (12-digit)").endswith("X")


def test_match_col_raises_when_no_match() -> None:
    with pytest.raises(KeyError):
        mod._match_col(["a", "b"], ("zzz",))


# =============================================================================
# _elsi_to_numeric
# =============================================================================


def test_elsi_to_numeric_nulls_symbols_and_strips_commas() -> None:
    s = pd.Series(["100", "1,234", "†", "–", "‡", ""])
    out = mod._elsi_to_numeric(s)
    assert out.iloc[0] == 100
    assert out.iloc[1] == 1234
    assert out.iloc[2:].isna().all()


# =============================================================================
# _read_elsi_table
# =============================================================================


def test_read_elsi_table_locates_header_below_preamble(elsi_only_dir: Path) -> None:
    df = mod._read_elsi_table(elsi_only_dir / "ELSI_csv_export_public_1920_sample.csv")
    assert df.columns[0] == "School Name"
    assert any("School ID (12-digit)" in c for c in df.columns)


def test_read_elsi_table_raises_without_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("ELSI Export\n\nno header here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        mod._read_elsi_table(bad)


# =============================================================================
# _load_elsi_wide
# =============================================================================


def test_load_elsi_wide_public_schema_and_values(elsi_only_dir: Path) -> None:
    st = mod.SCHOOL_TYPES["public"]
    df = mod._load_elsi_wide(elsi_only_dir / "ELSI_csv_export_public_1920_sample.csv", st)
    assert set(df.columns) == {"NCESSCH", "enroll_total", "g_grades_1_8", "g_grades_9_12"}
    row = df.set_index("NCESSCH").loc["510267002843"]
    assert row["enroll_total"] == 612
    assert row["g_grades_1_8"] == 420
    assert pd.isna(row["g_grades_9_12"])  # "†" -> NaN


def test_load_elsi_wide_drops_footer_keeps_only_real_rows(elsi_only_dir: Path) -> None:
    st = mod.SCHOOL_TYPES["public"]
    df = mod._load_elsi_wide(elsi_only_dir / "ELSI_csv_export_public_1920_sample.csv", st)
    assert set(df["NCESSCH"]) == PUBLIC_ELSI_MATCHED | {PUBLIC_ELSI_ORPHAN}


def test_load_elsi_wide_missing_symbol_becomes_nan(elsi_only_dir: Path) -> None:
    st = mod.SCHOOL_TYPES["public"]
    df = mod._load_elsi_wide(elsi_only_dir / "ELSI_csv_export_public_1920_sample.csv", st)
    orphan = df.set_index("NCESSCH").loc[PUBLIC_ELSI_ORPHAN]
    assert pd.isna(orphan["g_grades_1_8"])  # "–" (missing) -> NaN


def test_load_elsi_wide_private_uses_ppin_key(elsi_only_dir: Path) -> None:
    st = mod.SCHOOL_TYPES["private"]
    df = mod._load_elsi_wide(elsi_only_dir / "ELSI_csv_export_private_1920_sample.csv", st)
    assert "PPIN" in df.columns
    assert set(df["PPIN"]) == PRIVATE_ELSI_MATCHED | {PRIVATE_ELSI_ORPHAN}
    assert df.set_index("PPIN").loc["00253029", "enroll_total"] == 240


# =============================================================================
# _find_elsi_csv
# =============================================================================


def test_find_elsi_csv_disambiguates_public_and_private(staged_dir: Path) -> None:
    pub = mod._find_elsi_csv(staged_dir, mod.SCHOOL_TYPES["public"])
    prv = mod._find_elsi_csv(staged_dir, mod.SCHOOL_TYPES["private"])
    assert pub is not None and "public" in pub.name
    assert prv is not None and "private" in prv.name


def test_find_elsi_csv_returns_none_when_absent(tmp_path: Path) -> None:
    assert mod._find_elsi_csv(tmp_path, mod.SCHOOL_TYPES["public"]) is None


def test_find_elsi_csv_recurses_into_subfolders(tmp_path: Path) -> None:
    # The prep_features_public orchestrator unpacks each ELSI export zip into its own
    # subfolder, so the CSV sits one level below the input dir. Confirm the
    # finder recurses to it instead of only scanning the top level.
    nested = tmp_path / "ELSI_csv_export_6391668083015904102675"
    nested.mkdir()
    shutil.copy(
        FIXTURE_DIR / "ELSI_csv_export_public_1920_sample.csv",
        nested / "ELSI_csv_export_public_1920_sample.csv",
    )
    found = mod._find_elsi_csv(tmp_path, mod.SCHOOL_TYPES["public"])
    assert found is not None and found.parent == nested


def test_find_ipeds_file_recurses_into_subfolders(tmp_path: Path) -> None:
    # Same nesting story for the IPEDS EFFY enrollment file.
    nested = tmp_path / "EFFY2019"
    nested.mkdir()
    shutil.copy(FIXTURE_DIR / "effy2019_sample.csv", nested / "effy2019_sample.csv")
    found = mod._find_ipeds_file(tmp_path)
    assert found is not None and found.parent == nested


# =============================================================================
# load_enrollment_wide  (dispatch)
# =============================================================================


def test_enrollment_auto_prefers_ccd_when_present(staged_dir: Path) -> None:
    df = mod.load_enrollment_wide(staged_dir, "public", source="auto")
    # CCD totals differ from ELSI (700 vs 612) and CCD covers the DC point.
    indexed = df.set_index("NCESSCH")
    assert indexed.loc["510267002843", "enroll_total"] == 700
    assert "110000500377" in indexed.index


def test_enrollment_elsi_forced_ignores_ccd(staged_dir: Path) -> None:
    df = mod.load_enrollment_wide(staged_dir, "public", source="elsi")
    indexed = df.set_index("NCESSCH")
    assert indexed.loc["510267002843", "enroll_total"] == 612
    assert "110000500377" not in indexed.index  # ELSI sample omits the DC school


def test_enrollment_ccd_nulls_negative_sentinel(staged_dir: Path) -> None:
    # The CCD fixture has a Grade 5 row of -2 for 510267002843; it must not count.
    df = mod.load_enrollment_wide(staged_dir, "public", source="ccd")
    val = df.set_index("NCESSCH").loc["510267002843", "g_grade_5"]
    assert pd.isna(val) or val == 0  # sentinel nulled, not summed as -2


def test_enrollment_ccd_for_private_raises(staged_dir: Path) -> None:
    # Private declares only the ELSI source, so forcing CCD is rejected.
    with pytest.raises(ValueError, match="private"):
        mod.load_enrollment_wide(staged_dir, "private", source="ccd")


def test_enrollment_private_auto_falls_back_to_elsi(staged_dir: Path) -> None:
    df = mod.load_enrollment_wide(staged_dir, "private", source="auto")
    assert set(df["PPIN"]) == PRIVATE_ELSI_MATCHED | {PRIVATE_ELSI_ORPHAN}


def test_enrollment_invalid_source_raises(staged_dir: Path) -> None:
    with pytest.raises(ValueError, match="source"):
        mod.load_enrollment_wide(staged_dir, "public", source="xml")


def test_enrollment_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_enrollment_wide(tmp_path, "public")


# =============================================================================
# IPEDS / postsec  (third enrollment source)
# =============================================================================


def test_find_ipeds_file_locates_effy(staged_dir: Path) -> None:
    found = mod._find_ipeds_file(staged_dir)
    assert found is not None and found.name == "effy2019_sample.csv"


def test_find_ipeds_file_returns_none_when_absent(tmp_path: Path) -> None:
    assert mod._find_ipeds_file(tmp_path) is None


def test_load_ipeds_wide_schema_and_levels(staged_dir: Path) -> None:
    st = mod.SCHOOL_TYPES["postsec"]
    df = mod._load_ipeds_wide(staged_dir / "effy2019_sample.csv", st)
    assert set(df.columns) == {"UNITID", "enroll_total", "g_undergrad", "g_graduate"}
    row = df.set_index("UNITID").loc["232186"]
    assert row["enroll_total"] == 48678  # EFFYLEV 1 (all students)
    assert row["g_undergrad"] == 34722  # EFFYLEV 2
    assert row["g_graduate"] == 13956  # EFFYLEV 4


def test_load_ipeds_wide_missing_graduate_level_is_nan(staged_dir: Path) -> None:
    st = mod.SCHOOL_TYPES["postsec"]
    df = mod._load_ipeds_wide(staged_dir / "effy2019_sample.csv", st)
    # 163426 reports no EFFYLEV-4 (graduate) row in the source.
    assert pd.isna(df.set_index("UNITID").loc["163426", "g_graduate"])


def test_enrollment_postsec_auto_uses_ipeds(staged_dir: Path) -> None:
    df = mod.load_enrollment_wide(staged_dir, "postsec", source="auto")
    assert set(df["UNITID"]) == POSTSEC_IPEDS_MATCHED | {POSTSEC_IPEDS_ORPHAN}


def test_enrollment_postsec_rejects_non_ipeds_source(staged_dir: Path) -> None:
    with pytest.raises(ValueError, match="postsec"):
        mod.load_enrollment_wide(staged_dir, "postsec", source="ccd")


def test_load_school_points_postsec_uses_unitid(staged_dir: Path) -> None:
    gdf = mod.load_school_points(staged_dir, "postsec")
    assert set(gdf["UNITID"]) == POSTSEC_POINT_IDS  # AL row dropped


def test_join_postsec_ipeds_overlap(staged_dir: Path) -> None:
    points = mod.load_school_points(staged_dir, "postsec")
    enroll = mod.load_enrollment_wide(staged_dir, "postsec")
    out = mod.join_and_validate(points, enroll, id_col="UNITID")
    assert len(out) == len(POSTSEC_POINT_IDS)
    assert out["enroll_total"].notna().sum() == len(POSTSEC_IPEDS_MATCHED)


def test_run_postsec_writes_outputs(staged_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    mod.run(staged_dir, out_dir, school_type="postsec")
    assert (out_dir / "va_md_dc_postsec_schools_enrollment.gpkg").exists()


# =============================================================================
# load_school_points
# =============================================================================


def test_load_school_points_public_filters_to_target_states(staged_dir: Path) -> None:
    gdf = mod.load_school_points(staged_dir, "public")
    assert set(gdf["NCESSCH"]) == PUBLIC_POINT_IDS  # AL row dropped
    assert set(gdf["STATE"]) <= {"VA", "MD", "DC"}


def test_load_school_points_reprojects_to_output_crs(staged_dir: Path) -> None:
    gdf = mod.load_school_points(staged_dir, "public", output_crs=6487)
    assert gdf.crs.to_epsg() == 6487


def test_load_school_points_private_uses_ppin(staged_dir: Path) -> None:
    gdf = mod.load_school_points(staged_dir, "private")
    assert set(gdf["PPIN"]) == PRIVATE_POINT_IDS


def test_load_school_points_handles_nested_geocode_zip(tmp_path: Path) -> None:
    # Real NCES EDGE downloads unpack into a nested folder named after the
    # archive, so the .shp sits one level below the extraction root. Repack the
    # public geocode that way and confirm the (recursive) shp search still finds
    # it instead of raising FileNotFoundError.
    import zipfile

    flat = FIXTURE_DIR / "EDGE_GEOCODE_PUBLICSCH_1920_sample.zip"
    staged = tmp_path / "schools_in"
    staged.mkdir()
    nested = staged / "EDGE_GEOCODE_PUBLICSCH_1920_sample.zip"
    with zipfile.ZipFile(flat) as src, zipfile.ZipFile(nested, "w") as dst:
        for name in src.namelist():
            dst.writestr(f"EDGE_GEOCODE_PUBLICSCH_1920_sample/{name}", src.read(name))

    gdf = mod.load_school_points(staged, "public")
    assert set(gdf["NCESSCH"]) == PUBLIC_POINT_IDS


def test_load_school_points_no_matching_states_raises(staged_dir: Path) -> None:
    with pytest.raises(ValueError, match="No school points"):
        mod.load_school_points(staged_dir, "public", state_abbrs={"CA"})


def test_load_school_points_bad_state_col_raises(staged_dir: Path) -> None:
    with pytest.raises(KeyError):
        mod.load_school_points(staged_dir, "public", state_col="PROVINCE")


# =============================================================================
# join_and_validate
# =============================================================================


def _toy_points(ids: list[str], id_col: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {id_col: ids, "geometry": [Point(i, i) for i in range(len(ids))]},
        crs="EPSG:4269",
    )


def test_join_left_keeps_all_points_and_flags_matches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    points = _toy_points(["A", "B", "C"], "NCESSCH")
    enroll = pd.DataFrame({"NCESSCH": ["A", "B", "Z"], "enroll_total": [10, 20, 30]})
    with caplog.at_level("WARNING"):
        out = mod.join_and_validate(points, enroll, id_col="NCESSCH")
    assert len(out) == 3
    assert out["enroll_total"].notna().sum() == 2  # A, B matched; C not
    assert any("no enrollment row" in r.getMessage() for r in caplog.records)
    assert any("no matching point" in r.getMessage() for r in caplog.records)


def test_join_real_public_elsi_overlap(staged_dir: Path) -> None:
    points = mod.load_school_points(staged_dir, "public")
    enroll = mod.load_enrollment_wide(staged_dir, "public", source="elsi")
    out = mod.join_and_validate(points, enroll, id_col="NCESSCH")
    assert len(out) == len(PUBLIC_POINT_IDS)
    assert out["enroll_total"].notna().sum() == len(PUBLIC_ELSI_MATCHED)


# =============================================================================
# run  (integration)
# =============================================================================


def test_run_public_elsi_writes_outputs(staged_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    gdf = mod.run(staged_dir, out_dir, school_type="public", enrollment_source="elsi")
    gpkg = out_dir / "va_md_dc_public_schools_enrollment.gpkg"
    csv_path = out_dir / "va_md_dc_public_schools_enrollment.csv"
    assert gpkg.exists() and csv_path.exists()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gpd.read_file(gpkg)) == len(PUBLIC_POINT_IDS)
    assert "geometry" not in pd.read_csv(csv_path).columns


def test_run_private_writes_separate_layer(staged_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    mod.run(staged_dir, out_dir, school_type="private")
    assert (out_dir / "va_md_dc_private_schools_enrollment.gpkg").exists()


# =============================================================================
# main  (CLI)
# =============================================================================


def test_main_both_processes_public_and_private(staged_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    mod.main(
        [
            "--input-dir",
            str(staged_dir),
            "--output-dir",
            str(out_dir),
            "--school-type",
            "both",
            "--enrollment-source",
            "elsi",
        ]
    )
    produced = {p.name for p in out_dir.iterdir()}
    assert "va_md_dc_public_schools_enrollment.gpkg" in produced
    assert "va_md_dc_private_schools_enrollment.gpkg" in produced


def test_main_defaults_to_public(staged_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    mod.main(["--input-dir", str(staged_dir), "--output-dir", str(out_dir)])
    assert (out_dir / "va_md_dc_public_schools_enrollment.gpkg").exists()


def test_main_all_processes_every_school_type(staged_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    mod.main(["--input-dir", str(staged_dir), "--output-dir", str(out_dir), "--school-type", "all"])
    produced = {p.name for p in out_dir.iterdir()}
    for kind in ("public", "private", "postsec"):
        assert f"va_md_dc_{kind}_schools_enrollment.gpkg" in produced
