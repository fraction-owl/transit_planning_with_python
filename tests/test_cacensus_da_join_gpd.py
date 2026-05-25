"""Tests for scripts/census_tools/cacensus_da_join_gpd.py."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from scripts.census_tools import cacensus_da_join_gpd as mod

FIXTURE_DIR = Path("tests/fixtures")
FIXTURE_BUNDLE_ZIP = "98-401-X2021006_eng_CSV.zip"
FIXTURE_DA_ZIP = "lda_000b21a_e_sample.zip"

# Known counts from lda_000b21a_e_sample.zip
_FIXTURE_DA_TOTAL = 35
_FIXTURE_DA_ON_COUNT = 15   # CDUID 3506 (Ottawa)
_FIXTURE_DA_2481_COUNT = 15  # CDUID 2481 (Gatineau)
_FIXTURE_DA_2482_COUNT = 5   # CDUID 2482 (Les Collines-de-l'Outaouais)

# Representative DAUID from the Ontario fixture file.
_ON_DAUID = "35060207"
# Representative DAUID from the Quebec fixture file.
_QC_DAUID = "24810032"
_ON_CDUID = _ON_DAUID[:4]  # "3506"
_QC_CDUID = _QC_DAUID[:4]  # "2481"


# =============================================================================
# Helpers
# =============================================================================


def _make_census_csv(dauids: list[str], char_rows: list[tuple[str, str, str]]) -> str:
    """Build a minimal Census Profile CSV string (properly quoted, latin-1 safe).

    ``char_rows`` is a list of ``(CHARACTERISTIC_ID, CHARACTERISTIC_NAME,
    C1_COUNT_TOTAL)`` tuples applied to every DAUID.  Uses ``csv.writer`` so
    fields containing commas are quoted correctly.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ALT_GEO_CODE",
            "GEO_LEVEL",
            "GEO_NAME",
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            "C1_COUNT_TOTAL",
        ]
    )
    for dauid in dauids:
        for char_id, char_name, value in char_rows:
            writer.writerow([dauid, mod.DA_GEO_LEVEL, f"DA {dauid}", char_id, char_name, value])
    return buf.getvalue()


def _write_plain_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding=mod.CENSUS_ENCODING)


def _write_bundle_zip(zip_path: Path, csv_name: str, content: str) -> None:
    """Write a bundle zip containing one Census Profile CSV."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(csv_name, content.encode(mod.CENSUS_ENCODING))


def _make_da_gdf(dauids: list[str], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Create a minimal DA GeoDataFrame with box geometries for testing."""
    geoms = [box(i * 0.1, 0.0, i * 0.1 + 0.09, 0.09) for i in range(len(dauids))]
    return gpd.GeoDataFrame(
        {
            "DAUID": dauids,
            "PRUID": [d[:2] for d in dauids],
            "LANDAREA": [1.0] * len(dauids),
            "GEO_NAME": [f"DA {d}" for d in dauids],
        },
        geometry=geoms,
        crs=crs,
    )


_MINIMAL_CHAR_ROWS: list[tuple[str, str, str]] = [
    ("1", "Population, 2021", "500"),
    ("5", "Private dwellings occupied", "200"),
    ("8", "Total - Age groups", "500"),
    ("14", "15 to 19 years", "25"),
    ("15", "20 to 24 years", "20"),
    ("24", "65 years and over", "80"),
    ("260", "Total - Household total income groups", "195"),
    ("261", "Under $5,000", "5"),
    ("262", "$5,000 to $9,999", "5"),
    ("263", "$10,000 to $14,999", "10"),
    ("264", "$15,000 to $19,999", "10"),
    ("265", "$20,000 to $24,999", "10"),
    ("266", "$25,000 to $29,999", "10"),
    ("267", "$30,000 to $34,999", "10"),
    ("335", "Total - LIM low-income status", "500"),
    ("340", "In low income (LIM-AT)", "50"),
    ("345", "Prevalence of low income (LIM-AT) (%)", "10.0"),
    ("383", "Total - Knowledge of official languages", "490"),
    ("387", "Neither English nor French", "15"),
    ("1683", "Total - Visible minority", "520"),
    ("1684", "Total visible minority population", "90"),
    ("1697", "Not a visible minority", "430"),
    ("2603", "Total - Main mode of commuting", "200"),
    ("2604", "Car, truck or van", "140"),
    ("2605", "Car, truck or van - as a driver", "120"),
    ("2606", "Car, truck or van - as a passenger", "20"),
    ("2607", "Public transit", "40"),
    ("2608", "Walked", "15"),
    ("2609", "Bicycle", "5"),
    ("2610", "Other method", "0"),
]


@pytest.fixture()
def bundle_zip_dir(tmp_path: Path) -> Path:
    """Stage a bundle zip with one per-province Census Profile CSV."""
    csv_content = _make_census_csv([_ON_DAUID, "35060337"], _MINIMAL_CHAR_ROWS)
    _write_bundle_zip(
        tmp_path / "98-401-X2021006_eng_CSV.zip",
        "98-401-X2021006_English_CSV_data_Ontario.csv",
        csv_content,
    )
    return tmp_path


@pytest.fixture()
def plain_csv_dir(tmp_path: Path) -> Path:
    """Stage two plain per-province Census Profile CSVs."""
    on_content = _make_census_csv([_ON_DAUID], _MINIMAL_CHAR_ROWS)
    qc_content = _make_census_csv([_QC_DAUID], _MINIMAL_CHAR_ROWS)
    _write_plain_csv(tmp_path / "98-401-X2021006_English_CSV_data_Ontario.csv", on_content)
    _write_plain_csv(tmp_path / "98-401-X2021006_English_CSV_data_Quebec.csv", qc_content)
    return tmp_path


@pytest.fixture()
def da_shp_dir(tmp_path: Path) -> Path:
    """Write a small DA shapefile fixture."""
    dauids = [_ON_DAUID, "35060337", _QC_DAUID, "24810038"]
    gdf = _make_da_gdf(dauids)
    out = tmp_path / "lda_000b21a_e.shp"
    gdf.to_file(out, driver="ESRI Shapefile", index=False)
    return tmp_path


@pytest.fixture()
def da_zip_dir(tmp_path: Path, da_shp_dir: Path) -> Path:
    """Bundle the DA shapefile into a zip alongside the .shp directory."""
    shp_dir = da_shp_dir
    zip_path = tmp_path / "lda_000b21a_e.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            candidate = shp_dir / f"lda_000b21a_e{ext}"
            if candidate.exists():
                zf.write(candidate, candidate.name)
    # Remove the plain .shp so only the zip remains.
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        candidate = shp_dir / f"lda_000b21a_e{ext}"
        if candidate.exists():
            candidate.unlink()
    return tmp_path


# =============================================================================
# discover_census_profile_csvs
# =============================================================================


def test_discover_finds_bundle_zip(bundle_zip_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(bundle_zip_dir)
    assert len(sources) == 1
    assert sources[0].inner is not None
    assert "Ontario" in sources[0].name


def test_discover_finds_plain_csvs(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    assert len(sources) == 2
    names = {s.name for s in sources}
    assert any("Ontario" in n for n in names)
    assert any("Quebec" in n for n in names)


def test_discover_plain_csv_has_no_inner(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    assert all(s.inner is None for s in sources)


def test_discover_direct_file_wins_over_zip_member(tmp_path: Path) -> None:
    csv_name = "98-401-X2021006_English_CSV_data_Ontario.csv"
    csv_content = _make_census_csv([_ON_DAUID], _MINIMAL_CHAR_ROWS)

    # Write both a direct file and a bundle zip containing the same CSV.
    _write_plain_csv(tmp_path / csv_name, csv_content)
    _write_bundle_zip(tmp_path / "bundle.zip", csv_name, csv_content)

    sources = mod.discover_census_profile_csvs(tmp_path)
    assert len(sources) == 1
    assert sources[0].inner is None  # direct file preferred


def test_discover_recurses_into_subdirs(tmp_path: Path) -> None:
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    csv_content = _make_census_csv([_ON_DAUID], _MINIMAL_CHAR_ROWS)
    _write_plain_csv(sub / "98-401-X2021006_English_CSV_data_Ontario.csv", csv_content)
    sources = mod.discover_census_profile_csvs(tmp_path)
    assert len(sources) == 1


def test_discover_returns_empty_when_none_found(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("nothing here")
    assert mod.discover_census_profile_csvs(tmp_path) == []


def test_discover_results_sorted(tmp_path: Path) -> None:
    for prov in ["Quebec", "Ontario", "Atlantic"]:
        _write_plain_csv(
            tmp_path / f"98-401-X2021006_English_CSV_data_{prov}.csv",
            _make_census_csv(["12340001"], _MINIMAL_CHAR_ROWS),
        )
    sources = mod.discover_census_profile_csvs(tmp_path)
    names = [s.name for s in sources]
    assert names == sorted(names)


def test_discover_ignores_bad_zip(tmp_path: Path) -> None:
    (tmp_path / "corrupt.zip").write_bytes(b"not a zip")
    sources = mod.discover_census_profile_csvs(tmp_path)
    assert sources == []


# =============================================================================
# _read_profile_csv
# =============================================================================


def test_read_profile_csv_returns_da_rows_only(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod._read_profile_csv(sources[0])
    assert (df["ALT_GEO_CODE"] == _ON_DAUID).all() or (df["ALT_GEO_CODE"] == _QC_DAUID).all()


def test_read_profile_csv_from_bundle_zip(bundle_zip_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(bundle_zip_dir)
    df = mod._read_profile_csv(sources[0])
    assert len(df) > 0
    assert "GEO_LEVEL" not in df.columns


def test_read_profile_csv_columns_present(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod._read_profile_csv(sources[0])
    assert {"ALT_GEO_CODE", "GEO_NAME", "CHARACTERISTIC_ID", "C1_COUNT_TOTAL"}.issubset(df.columns)


# =============================================================================
# _fill_numeric_only
# =============================================================================


def test_fill_numeric_only_fills_numeric_nans() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": [float("nan"), 3.0]})
    result = mod._fill_numeric_only(df)
    assert result["a"].iloc[1] == 0
    assert result["b"].iloc[0] == 0


def test_fill_numeric_only_leaves_strings_untouched() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")], "name": ["Alice", None]})
    result = mod._fill_numeric_only(df)
    assert pd.isna(result["name"].iloc[1])


# =============================================================================
# _load_and_pivot
# =============================================================================


def test_load_and_pivot_one_row_per_dauid(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod._load_and_pivot(sources, mod.CHARACTERISTIC_MAP)
    assert len(df) == 2  # one row per DAUID
    assert df[mod.DAUID_COL].is_unique


def test_load_and_pivot_renames_columns(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod._load_and_pivot(sources, mod.CHARACTERISTIC_MAP)
    assert "total_pop" in df.columns
    assert "occ_dwell" in df.columns


def test_load_and_pivot_dauid_column_present(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod._load_and_pivot(sources, mod.CHARACTERISTIC_MAP)
    assert mod.DAUID_COL in df.columns


def test_load_and_pivot_handles_suppressed_values(tmp_path: Path) -> None:
    char_rows_with_suppression: list[tuple[str, str, str]] = [
        ("1", "Population, 2021", "x"),
        ("5", "Private dwellings", "F"),
        ("8", "Age total", "500"),
    ]
    csv_content = _make_census_csv([_ON_DAUID], char_rows_with_suppression)
    _write_plain_csv(tmp_path / "98-401-X2021006_English_CSV_data_Ontario.csv", csv_content)
    sources = mod.discover_census_profile_csvs(tmp_path)
    custom_map = {"1": "total_pop", "5": "occ_dwell", "8": "age_total"}
    df = mod._load_and_pivot(sources, custom_map)
    # Suppressed values become 0 after fill.
    assert df["total_pop"].iloc[0] == 0
    assert df["occ_dwell"].iloc[0] == 0
    assert df["age_total"].iloc[0] == 500


def test_load_and_pivot_deduplicates_overlapping_sources(tmp_path: Path) -> None:
    csv_name = "98-401-X2021006_English_CSV_data_Ontario.csv"
    csv_content = _make_census_csv([_ON_DAUID], _MINIMAL_CHAR_ROWS)

    # Both a direct file and a bundle zip containing the same CSV.
    _write_plain_csv(tmp_path / csv_name, csv_content)
    _write_bundle_zip(tmp_path / "bundle.zip", csv_name, csv_content)

    sources = mod.discover_census_profile_csvs(tmp_path)
    df = mod._load_and_pivot(sources, mod.CHARACTERISTIC_MAP)
    # Should still have exactly one row per DAUID, not duplicates.
    assert len(df) == 1


def test_load_and_pivot_returns_empty_for_no_sources() -> None:
    df = mod._load_and_pivot([])
    assert df.empty


def test_load_and_pivot_geo_name_preserved(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod._load_and_pivot(sources, mod.CHARACTERISTIC_MAP)
    assert "GEO_NAME" in df.columns


# =============================================================================
# Derivation functions
# =============================================================================


def test_derive_income_adds_low_inc_hh() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "hh_inc_total": [200.0],
            "hh_u5k": [10.0],
            "hh_5_10k": [10.0],
            "hh_10_15k": [10.0],
            "hh_15_20k": [10.0],
            "hh_20_25k": [10.0],
            "hh_25_30k": [10.0],
            "hh_30_35k": [10.0],
        }
    )
    result = mod._derive_income(df)
    assert result["low_inc_hh"].iloc[0] == 70
    assert result["perc_low_inc"].iloc[0] == pytest.approx(0.35)


def test_derive_income_without_hh_inc_total_skips_percentage() -> None:
    df = pd.DataFrame({mod.DAUID_COL: [_ON_DAUID], "hh_u5k": [5.0], "hh_5_10k": [5.0]})
    result = mod._derive_income(df)
    assert result["low_inc_hh"].iloc[0] == 10
    assert "perc_low_inc" not in result.columns


def test_derive_visible_minority_computes_perc_vm() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "vm_denom": [500.0],
            "vm_count": [100.0],
            "vm_not": [400.0],
        }
    )
    result = mod._derive_visible_minority(df)
    assert result["perc_vm"].iloc[0] == pytest.approx(0.2)


def test_derive_visible_minority_zero_denom_yields_zero() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "vm_denom": [0.0],
            "vm_count": [0.0],
            "vm_not": [0.0],
        }
    )
    result = mod._derive_visible_minority(df)
    assert result["perc_vm"].iloc[0] == 0.0


def test_derive_language_computes_perc_allophone() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "lang_total": [400.0],
            "lang_neither": [40.0],
        }
    )
    result = mod._derive_language(df)
    assert result["perc_allophone"].iloc[0] == pytest.approx(0.1)


def test_derive_language_zero_lang_total_yields_zero() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "lang_total": [0.0],
            "lang_neither": [0.0],
        }
    )
    result = mod._derive_language(df)
    assert result["perc_allophone"].iloc[0] == 0.0


def test_derive_age_computes_youth_from_age_bands() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "total_pop": [1000.0],
            "age_15_19": [50.0],
            "age_20_24": [40.0],
            "age_65_plus": [120.0],
        }
    )
    result = mod._derive_age(df)
    assert result["all_youth"].iloc[0] == 90
    assert result["all_elderly"].iloc[0] == 120
    assert result["perc_youth"].iloc[0] == pytest.approx(0.09)
    assert result["perc_elderly"].iloc[0] == pytest.approx(0.12)


def test_derive_age_builds_elderly_from_sub_bands_if_65_plus_missing() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "age_65_69": [30.0],
            "age_70_74": [20.0],
            "age_75_79": [10.0],
        }
    )
    result = mod._derive_age(df)
    assert result["all_elderly"].iloc[0] == 60


def test_derive_age_without_total_pop_skips_percentages() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "age_15_19": [30.0],
            "age_20_24": [20.0],
        }
    )
    result = mod._derive_age(df)
    assert result["all_youth"].iloc[0] == 50
    assert "perc_youth" not in result.columns


def test_derive_commute_mode_computes_percentages() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "commute_total": [200.0],
            "commute_car": [140.0],
            "commute_car_driver": [120.0],
            "commute_car_pass": [20.0],
            "commute_transit": [40.0],
            "commute_walk": [15.0],
            "commute_bike": [5.0],
            "commute_other": [0.0],
        }
    )
    result = mod._derive_commute_mode(df)
    assert result["perc_car"].iloc[0] == pytest.approx(0.7)
    assert result["perc_car_driver"].iloc[0] == pytest.approx(0.6)
    assert result["perc_car_pass"].iloc[0] == pytest.approx(0.1)
    assert result["perc_transit"].iloc[0] == pytest.approx(0.2)
    assert result["perc_walk"].iloc[0] == pytest.approx(0.075)
    assert result["perc_bike"].iloc[0] == pytest.approx(0.025)
    assert result["perc_cm_other"].iloc[0] == 0.0


def test_derive_commute_mode_zero_denom_yields_zero() -> None:
    df = pd.DataFrame(
        {
            mod.DAUID_COL: [_ON_DAUID],
            "commute_total": [0.0],
            "commute_car_driver": [0.0],
            "commute_transit": [0.0],
        }
    )
    result = mod._derive_commute_mode(df)
    assert result["perc_car_driver"].iloc[0] == 0.0
    assert result["perc_transit"].iloc[0] == 0.0


def test_derive_commute_mode_missing_denom_is_noop() -> None:
    df = pd.DataFrame({mod.DAUID_COL: [_ON_DAUID], "commute_car_driver": [100.0]})
    result = mod._derive_commute_mode(df)
    assert "perc_car_driver" not in result.columns


# =============================================================================
# CDUID filter helpers
# =============================================================================


def test_ensure_cduid_column_df_creates_four_char_cduid() -> None:
    df = pd.DataFrame({mod.DAUID_COL: [_ON_DAUID]})
    mod._ensure_cduid_column_df(df)
    assert "CDUID" in df.columns
    assert df["CDUID"].iloc[0] == _ON_CDUID


def test_ensure_cduid_column_df_is_idempotent() -> None:
    df = pd.DataFrame({mod.DAUID_COL: [_ON_DAUID], "CDUID": ["9999"]})
    mod._ensure_cduid_column_df(df)
    assert df["CDUID"].iloc[0] == "9999"


def test_ensure_cduid_column_df_raises_without_dauid() -> None:
    df = pd.DataFrame({"other": [1]})
    with pytest.raises(KeyError):
        mod._ensure_cduid_column_df(df)


def test_apply_cduid_filter_df_keeps_matching_rows() -> None:
    df = pd.DataFrame({mod.DAUID_COL: [_ON_DAUID, _QC_DAUID], "pop": [100, 200]})
    result = mod._apply_cduid_filter_df(df, cduids=[_ON_CDUID])
    assert len(result) == 1
    assert result[mod.DAUID_COL].iloc[0] == _ON_DAUID


def test_apply_cduid_filter_df_empty_cduids_returns_unchanged() -> None:
    df = pd.DataFrame({mod.DAUID_COL: [_ON_DAUID, _QC_DAUID]})
    assert len(mod._apply_cduid_filter_df(df, cduids=[])) == 2


def test_apply_cduid_filter_df_none_cduids_returns_unchanged() -> None:
    df = pd.DataFrame({mod.DAUID_COL: [_ON_DAUID]})
    assert len(mod._apply_cduid_filter_df(df, cduids=None)) == 1


# =============================================================================
# build_da_table
# =============================================================================


def test_build_da_table_returns_one_row_per_dauid(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod.build_da_table(sources)
    assert len(df) == 2
    assert df[mod.DAUID_COL].is_unique


def test_build_da_table_has_derived_columns(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod.build_da_table(sources)
    expected_derived = (
        "low_inc_hh",
        "perc_low_inc",
        "perc_vm",
        "perc_allophone",
        "all_youth",
        "all_elderly",
        "perc_car",
        "perc_car_driver",
        "perc_car_pass",
        "perc_transit",
        "perc_walk",
        "perc_bike",
        "perc_cm_other",
    )
    for col in expected_derived:
        assert col in df.columns, f"Expected derived column '{col}' missing"


def test_build_da_table_with_cduid_filter(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod.build_da_table(sources, cduid_filter=[_ON_CDUID])
    assert (df["CDUID"] == _ON_CDUID).all()
    assert len(df) == 1


def test_build_da_table_no_nans_in_numeric_columns(plain_csv_dir: Path) -> None:
    sources = mod.discover_census_profile_csvs(plain_csv_dir)
    df = mod.build_da_table(sources)
    numeric = df.select_dtypes(include="number")
    assert not numeric.isna().any().any(), "NaN found in numeric columns after build_da_table"


# =============================================================================
# discover_da_shapefile
# =============================================================================


def test_discover_da_shapefile_finds_plain_shp(da_shp_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    assert "zip://" not in path
    assert path.endswith(".shp")


def test_discover_da_shapefile_finds_zip(da_zip_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_zip_dir)
    assert path.startswith("zip://")


def test_discover_da_shapefile_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        mod.discover_da_shapefile(tmp_path / "nonexistent")


def test_discover_da_shapefile_raises_when_no_match(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.discover_da_shapefile(tmp_path)


def test_discover_da_shapefile_prefers_shp_by_default(tmp_path: Path) -> None:
    """When both .shp and .zip exist, the plain .shp is preferred."""
    dauids = [_ON_DAUID]
    gdf = _make_da_gdf(dauids)
    shp = tmp_path / "lda_000b21a_e.shp"
    gdf.to_file(shp, driver="ESRI Shapefile", index=False)

    zip_path = tmp_path / "lda_000b21a_e.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            candidate = tmp_path / f"lda_000b21a_e{ext}"
            if candidate.exists():
                zf.write(candidate, candidate.name)

    path = mod.discover_da_shapefile(tmp_path)
    assert "zip://" not in path


# =============================================================================
# load_da_shapefile
# =============================================================================


def test_load_da_shapefile_returns_geodataframe(da_shp_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_load_da_shapefile_has_dauid_and_cduid(da_shp_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    assert "DAUID" in gdf.columns
    assert "CDUID" in gdf.columns


def test_load_da_shapefile_dauid_is_string(da_shp_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    assert pd.api.types.is_string_dtype(gdf["DAUID"])


def test_load_da_shapefile_cduid_is_four_chars(da_shp_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    assert (gdf["CDUID"].str.len() == 4).all()


# =============================================================================
# filter_da_by_cduid
# =============================================================================


def test_filter_da_by_cduid_keeps_matching_das(da_shp_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    result = mod.filter_da_by_cduid(gdf, [_ON_CDUID])
    assert set(result["CDUID"].unique()) == {_ON_CDUID}
    assert len(result) < len(gdf)


def test_filter_da_by_cduid_empty_list_returns_all(da_shp_dir: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    assert len(mod.filter_da_by_cduid(gdf, [])) == len(gdf)


def test_filter_da_by_cduid_derives_cduid_if_missing() -> None:
    dauids = [_ON_DAUID, _QC_DAUID]
    gdf = _make_da_gdf(dauids)
    # Drop CDUID so the function must derive it.
    gdf_no_cduid = gdf.copy()
    result = mod.filter_da_by_cduid(gdf_no_cduid, [_ON_CDUID])
    assert len(result) == 1
    assert result["DAUID"].iloc[0] == _ON_DAUID


# =============================================================================
# join_das_to_attributes
# =============================================================================


@pytest.fixture()
def da_and_attrs() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    dauids = [_ON_DAUID, "35060337", _QC_DAUID]
    gdf = _make_da_gdf(dauids)
    attrs = pd.DataFrame(
        {
            mod.JOIN_KEY: dauids[:2],
            "total_pop": [500.0, 300.0],
        }
    )
    return gdf, attrs


def test_join_left_preserves_all_da_rows(
    da_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    gdf, attrs = da_and_attrs
    result = mod.join_das_to_attributes(gdf, attrs)
    assert len(result) == len(gdf)


def test_join_inner_keeps_only_matched_rows(
    da_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    gdf, attrs = da_and_attrs
    result = mod.join_das_to_attributes(gdf, attrs, how="inner")
    assert len(result) == 2  # only two DAUIDs in attrs


def test_join_returns_geodataframe_with_geometry(
    da_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    gdf, attrs = da_and_attrs
    result = mod.join_das_to_attributes(gdf, attrs)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.geometry.name in result.columns


def test_join_attribute_columns_present_in_result(
    da_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    gdf, attrs = da_and_attrs
    assert "total_pop" in mod.join_das_to_attributes(gdf, attrs).columns


def test_join_raises_on_duplicate_attribute_keys() -> None:
    gdf = _make_da_gdf([_ON_DAUID])
    dup_attrs = pd.DataFrame({mod.JOIN_KEY: [_ON_DAUID, _ON_DAUID], "pop": [1, 2]})
    with pytest.raises(ValueError):
        mod.join_das_to_attributes(gdf, dup_attrs)


def test_join_casts_int64_to_float(da_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame]) -> None:
    gdf, attrs = da_and_attrs
    attrs["int_col"] = pd.array([1, 2], dtype="Int64")
    result = mod.join_das_to_attributes(gdf, attrs, how="inner")
    assert result["int_col"].dtype == "float64"


# =============================================================================
# _cast_int64_to_float
# =============================================================================


def test_cast_int64_to_float_converts() -> None:
    _two_da_geoms = _make_da_gdf([_ON_DAUID, _QC_DAUID]).geometry
    gdf = gpd.GeoDataFrame({"val": pd.array([1, 2], dtype="Int64"), "geometry": _two_da_geoms})
    mod._cast_int64_to_float(gdf)
    assert gdf["val"].dtype == "float64"


def test_cast_int64_to_float_leaves_strings() -> None:
    _two_da_geoms = _make_da_gdf([_ON_DAUID, _QC_DAUID]).geometry
    gdf = gpd.GeoDataFrame({"label": ["a", "b"], "geometry": _two_da_geoms})
    mod._cast_int64_to_float(gdf)
    assert not pd.api.types.is_float_dtype(gdf["label"])


# =============================================================================
# _truncate_field_names
# =============================================================================


def test_truncate_field_names_shortens_long_names() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "a_very_long_column_name": [1],
            "geometry": _make_da_gdf([_ON_DAUID]).geometry,
        }
    )
    result = mod._truncate_field_names(gdf, max_len=10)
    non_geom = [c for c in result.columns if c != "geometry"]
    assert all(len(c) <= 10 for c in non_geom)


def test_truncate_field_names_resolves_collisions() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "total_population": [1],
            "total_pop_count": [2],
            "geometry": _make_da_gdf([_ON_DAUID]).geometry,
        }
    )
    result = mod._truncate_field_names(gdf, max_len=10)
    non_geom = [c for c in result.columns if c != "geometry"]
    assert len(non_geom) == len(set(non_geom))


def test_truncate_field_names_never_renames_geometry() -> None:
    gdf = gpd.GeoDataFrame({"geometry": _make_da_gdf([_ON_DAUID]).geometry})
    result = mod._truncate_field_names(gdf, max_len=5)
    assert "geometry" in result.columns


# =============================================================================
# write_geo / write_csv
# =============================================================================


def test_write_geo_gpkg_roundtrip(da_shp_dir: Path, tmp_path: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    out = tmp_path / "out" / "da.gpkg"
    mod.write_geo(gdf, str(out))
    assert out.exists()
    assert len(gpd.read_file(out)) == len(gdf)


def test_write_geo_shapefile_roundtrip(da_shp_dir: Path, tmp_path: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    out = tmp_path / "out" / "da.shp"
    mod.write_geo(gdf, str(out))
    assert out.exists()
    assert len(gpd.read_file(out)) == len(gdf)


def test_write_geo_creates_parent_dirs(da_shp_dir: Path, tmp_path: Path) -> None:
    path = mod.discover_da_shapefile(da_shp_dir)
    gdf = mod.load_da_shapefile(path)
    out = tmp_path / "a" / "b" / "c" / "da.gpkg"
    mod.write_geo(gdf, str(out))
    assert out.exists()


def test_write_geo_shapefile_truncates_column_names(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "a_very_long_column_name_here": [1],
            "geometry": gpd.GeoSeries(_make_da_gdf([_ON_DAUID]).geometry, crs="EPSG:4326"),
        }
    )
    out = tmp_path / "out.shp"
    mod.write_geo(gdf, str(out))
    roundtripped = gpd.read_file(out)
    non_geom = [c for c in roundtripped.columns if c != "geometry"]
    assert all(len(c) <= 10 for c in non_geom)


def test_write_csv_creates_correct_file(tmp_path: Path) -> None:
    df = pd.DataFrame({"DAUID": [_ON_DAUID], "pop": [500]})
    out = tmp_path / "out.csv"
    mod.write_csv(df, str(out))
    assert out.exists()
    result = pd.read_csv(out)
    assert list(result.columns) == ["DAUID", "pop"]
    assert len(result) == 1


def test_write_csv_creates_missing_parent_dirs(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1]})
    out = tmp_path / "nested" / "dir" / "out.csv"
    mod.write_csv(df, str(out))
    assert out.exists()


# =============================================================================
# _is_blank / _check_placeholders
# =============================================================================


def test_is_blank_none() -> None:
    assert mod._is_blank(None) is True


def test_is_blank_empty_string() -> None:
    assert mod._is_blank("") is True


def test_is_blank_whitespace_only() -> None:
    assert mod._is_blank("   ") is True


def test_is_blank_real_path() -> None:
    assert mod._is_blank("/some/path/da.gpkg") is False


def test_check_placeholders_true_when_defaults_unchanged() -> None:
    assert mod._check_placeholders() is True


def test_check_placeholders_false_when_overridden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mod, "INPUT_CSV_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(mod, "INPUT_SHP_DIR", str(tmp_path / "shp"))
    monkeypatch.setattr(mod, "FINAL_JOINED_FEATURES", str(tmp_path / "out.gpkg"))
    monkeypatch.setattr(mod, "INTERMEDIATE_COMBINED_CSV", "")
    assert mod._check_placeholders() is False


# =============================================================================
# Integration tests using the real fixture bundle zip
# =============================================================================


@pytest.mark.skipif(
    not (FIXTURE_DIR / FIXTURE_BUNDLE_ZIP).exists(),
    reason=f"Fixture {FIXTURE_BUNDLE_ZIP} not present",
)
def test_integration_discover_fixture_bundle() -> None:
    sources = mod.discover_census_profile_csvs(FIXTURE_DIR)
    fixture_sources = [s for s in sources if FIXTURE_BUNDLE_ZIP in str(s.container)]
    assert len(fixture_sources) >= 1


@pytest.mark.skipif(
    not (FIXTURE_DIR / FIXTURE_BUNDLE_ZIP).exists(),
    reason=f"Fixture {FIXTURE_BUNDLE_ZIP} not present",
)
def test_integration_build_da_table_from_fixture() -> None:
    sources = mod.discover_census_profile_csvs(FIXTURE_DIR)
    fixture_sources = [s for s in sources if FIXTURE_BUNDLE_ZIP in str(s.container)]
    df = mod.build_da_table(fixture_sources)

    assert not df.empty
    assert mod.DAUID_COL in df.columns
    assert df[mod.DAUID_COL].is_unique
    assert "total_pop" in df.columns
    assert "perc_vm" in df.columns
    assert "perc_allophone" in df.columns
    assert "perc_low_inc" in df.columns
    assert "all_youth" in df.columns
    assert "all_elderly" in df.columns
    assert "perc_car" in df.columns
    assert "perc_car_driver" in df.columns
    assert "perc_transit" in df.columns

    # Spot-check a known DA value (Ontario DA 35060207, total_pop = 503).
    row = df[df[mod.DAUID_COL] == _ON_DAUID]
    assert len(row) == 1
    assert row["total_pop"].iloc[0] == pytest.approx(503.0)
    # Commute: total=170, car_driver=95, transit=40 → perc_car_driver≈0.559
    assert row["commute_total"].iloc[0] == pytest.approx(170.0)
    assert row["perc_car_driver"].iloc[0] == pytest.approx(95 / 170, rel=1e-2)
    assert row["perc_transit"].iloc[0] == pytest.approx(40 / 170, rel=1e-2)


@pytest.mark.skipif(
    not (FIXTURE_DIR / FIXTURE_BUNDLE_ZIP).exists(),
    reason=f"Fixture {FIXTURE_BUNDLE_ZIP} not present",
)
def test_integration_cduid_filter_from_fixture() -> None:
    sources = mod.discover_census_profile_csvs(FIXTURE_DIR)
    fixture_sources = [s for s in sources if FIXTURE_BUNDLE_ZIP in str(s.container)]
    df = mod.build_da_table(fixture_sources, cduid_filter=[_ON_CDUID])

    # All returned DAUIDs must belong to the requested CDUID.
    assert (df["CDUID"] == _ON_CDUID).all()
    # No Quebec DAs should appear.
    assert not df[mod.DAUID_COL].str.startswith("24").any()


@pytest.mark.skipif(
    not (FIXTURE_DIR / FIXTURE_BUNDLE_ZIP).exists(),
    reason=f"Fixture {FIXTURE_BUNDLE_ZIP} not present",
)
def test_integration_full_pipeline_with_mock_shapefile(tmp_path: Path) -> None:  # noqa: D103
    """End-to-end test: fixture CSVs → pivot → join onto mock DA geometry → write GeoPackage."""
    sources = mod.discover_census_profile_csvs(FIXTURE_DIR)
    fixture_sources = [s for s in sources if FIXTURE_BUNDLE_ZIP in str(s.container)]
    attrs_df = mod.build_da_table(fixture_sources)

    # Build a mock DA GeoDataFrame covering the DAUIDs present in the fixture.
    dauids = attrs_df[mod.DAUID_COL].tolist()
    da_gdf = _make_da_gdf(dauids, crs="EPSG:4326")
    # Patch in CDUID column (load_da_shapefile would add this from a real file).
    da_gdf["CDUID"] = da_gdf["DAUID"].str[:4]

    joined = mod.join_das_to_attributes(da_gdf, attrs_df)

    assert isinstance(joined, gpd.GeoDataFrame)
    assert len(joined) == len(da_gdf)
    assert "total_pop" in joined.columns
    assert "perc_vm" in joined.columns

    # Round-trip to GeoPackage.
    out = tmp_path / "da_joined.gpkg"
    mod.write_geo(joined, str(out))
    assert out.exists()
    roundtripped = gpd.read_file(out)
    assert len(roundtripped) == len(joined)


# =============================================================================
# Integration tests using the real fixture DA shapefile (lda_000b21a_e_sample.zip)
# =============================================================================

_da_shp_present = (FIXTURE_DIR / FIXTURE_DA_ZIP).exists()
_da_shp_skipif = pytest.mark.skipif(
    not _da_shp_present,
    reason=f"Fixture {FIXTURE_DA_ZIP} not present",
)


@_da_shp_skipif
def test_fixture_da_zip_is_discovered() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    assert path.startswith("zip://")
    assert FIXTURE_DA_ZIP in path


@_da_shp_skipif
def test_fixture_da_zip_loads_correct_row_count() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    gdf = mod.load_da_shapefile(path)
    assert len(gdf) == _FIXTURE_DA_TOTAL


@_da_shp_skipif
def test_fixture_da_zip_dauid_is_string() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    gdf = mod.load_da_shapefile(path)
    assert pd.api.types.is_string_dtype(gdf["DAUID"])


@_da_shp_skipif
def test_fixture_da_zip_cduid_derived_and_four_chars() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    gdf = mod.load_da_shapefile(path)
    assert "CDUID" in gdf.columns
    assert (gdf["CDUID"].str.len() == 4).all()


@_da_shp_skipif
def test_fixture_da_zip_known_dauids_present() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    gdf = mod.load_da_shapefile(path)
    assert _ON_DAUID in gdf["DAUID"].values
    assert _QC_DAUID in gdf["DAUID"].values


@_da_shp_skipif
def test_fixture_da_zip_filter_by_ontario_cduid() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    gdf = mod.load_da_shapefile(path)
    filtered = mod.filter_da_by_cduid(gdf, [_ON_CDUID])
    assert len(filtered) == _FIXTURE_DA_ON_COUNT
    assert (filtered["CDUID"] == _ON_CDUID).all()


@_da_shp_skipif
def test_fixture_da_zip_filter_by_qc_cduid_2481() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    gdf = mod.load_da_shapefile(path)
    filtered = mod.filter_da_by_cduid(gdf, [_QC_CDUID])
    assert len(filtered) == _FIXTURE_DA_2481_COUNT
    assert (filtered["CDUID"] == _QC_CDUID).all()


@_da_shp_skipif
def test_fixture_da_zip_filter_multi_cduid() -> None:
    path = mod.discover_da_shapefile(FIXTURE_DIR)
    gdf = mod.load_da_shapefile(path)
    filtered = mod.filter_da_by_cduid(gdf, [_ON_CDUID, _QC_CDUID])
    assert len(filtered) == _FIXTURE_DA_ON_COUNT + _FIXTURE_DA_2481_COUNT
    assert set(filtered["CDUID"].unique()) == {_ON_CDUID, _QC_CDUID}


@pytest.mark.skipif(
    not _da_shp_present or not (FIXTURE_DIR / FIXTURE_BUNDLE_ZIP).exists(),
    reason=f"Fixtures {FIXTURE_DA_ZIP} and/or {FIXTURE_BUNDLE_ZIP} not present",
)
def test_integration_census_and_shapefile_fixtures(tmp_path: Path) -> None:
    """Full pipeline: fixture census CSVs → pivot → join onto fixture DA shapefile → write."""
    sources = mod.discover_census_profile_csvs(FIXTURE_DIR)
    fixture_sources = [s for s in sources if FIXTURE_BUNDLE_ZIP in str(s.container)]
    attrs_df = mod.build_da_table(fixture_sources)

    assert len(attrs_df) == _FIXTURE_DA_TOTAL
    assert mod.DAUID_COL in attrs_df.columns

    da_path = mod.discover_da_shapefile(FIXTURE_DIR)
    da_gdf = mod.load_da_shapefile(da_path)
    assert len(da_gdf) == _FIXTURE_DA_TOTAL

    joined = mod.join_das_to_attributes(da_gdf, attrs_df)

    assert isinstance(joined, gpd.GeoDataFrame)
    assert len(joined) == _FIXTURE_DA_TOTAL
    assert "total_pop" in joined.columns
    assert "perc_transit" in joined.columns
    assert "perc_vm" in joined.columns

    # Spot-check the known Ontario DA (35060207).
    row = joined[joined["DAUID"] == _ON_DAUID]
    assert len(row) == 1
    assert row["total_pop"].iloc[0] == pytest.approx(503.0)

    out = tmp_path / "da_joined_fixture.gpkg"
    mod.write_geo(joined, str(out))
    assert out.exists()
    roundtripped = gpd.read_file(out)
    assert len(roundtripped) == _FIXTURE_DA_TOTAL
