from __future__ import annotations

import logging
import shutil
from pathlib import Path

import geopandas as gpd
import pytest

from scripts.census_tools import uscensus_blocks_merge_gpd as merge_mod

FIXTURE_DIR = Path("tests/fixtures")
FIXTURE_ZIPS = (
    "tl_2025_11_tabblock20_sample.zip",
    "tl_2025_24_tabblock20_sample.zip",
    "tl_2025_51_tabblock20_sample.zip",
)
# Feature counts in each fixture, in the same order as FIXTURE_ZIPS.
EXPECTED_COUNTS = (14, 40, 59)
EXPECTED_TOTAL = sum(EXPECTED_COUNTS)
# All county FIPS codes present across the three sample fixtures.
ALL_FIXTURE_FIPS = frozenset(
    {
        "11001",  # DC
        "24017",
        "24027",
        "24031",
        "24033",  # MD counties
        "51013",
        "51059",
        "51107",
        "51153",
        "51510",
        "51600",
        "51610",  # VA
    }
)


@pytest.fixture
def tiger_dir(tmp_path: Path) -> Path:
    """Stage the three TIGER zip fixtures in an isolated directory."""
    staged = tmp_path / "tiger"
    staged.mkdir()
    for name in FIXTURE_ZIPS:
        shutil.copy(FIXTURE_DIR / name, staged / name)
    return staged


def test_discover_tiger_datasets_finds_all_zips(tiger_dir: Path) -> None:
    """Recursive discovery picks up every zipped TIGER fixture with the VFS prefix."""
    paths = merge_mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")

    assert len(paths) == len(FIXTURE_ZIPS)
    assert all(p.startswith("zip://") for p in paths)
    stems = {Path(p.removeprefix("zip://")).stem for p in paths}
    assert stems == {Path(name).stem for name in FIXTURE_ZIPS}


def test_discover_tiger_datasets_raises_on_empty_dir(tmp_path: Path) -> None:
    """An empty input directory is a hard error, not a silent no-op."""
    with pytest.raises(FileNotFoundError):
        merge_mod.discover_tiger_datasets(tmp_path, "tl_*_*_*.shp")


def test_read_shapefile_zfills_state_and_county_codes(tiger_dir: Path) -> None:
    """STATEFP20 stays 2 digits and COUNTYFP20 stays 3 digits after read."""
    gdf = merge_mod.read_shapefile(f"zip://{tiger_dir / FIXTURE_ZIPS[0]}")

    assert len(gdf) == EXPECTED_COUNTS[0]
    assert (gdf["STATEFP20"].str.len() == 2).all()
    assert (gdf["COUNTYFP20"].str.len() == 3).all()
    assert set(gdf["STATEFP20"].unique()) == {"11"}


def test_merge_shapefiles_concatenates_features_and_preserves_crs(tiger_dir: Path) -> None:
    """All three fixtures merge into one frame; CRS is unchanged (EPSG:4269)."""
    paths = merge_mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = merge_mod.merge_shapefiles(paths)

    assert len(merged) == EXPECTED_TOTAL
    assert str(merged.crs) == "EPSG:4269"
    assert set(merged["STATEFP20"].unique()) == {"11", "24", "51"}


def test_ensure_fips_column_builds_five_digit_code(tiger_dir: Path) -> None:
    """FIPS column equals STATEFP20 + COUNTYFP20 with correct padding."""
    paths = merge_mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = merge_mod.merge_shapefiles(paths)

    result = merge_mod.ensure_fips_column(merged)

    assert "FIPS" in result.columns
    assert (result["FIPS"].str.len() == 5).all()
    assert set(result["FIPS"].unique()) == ALL_FIXTURE_FIPS


def test_ensure_fips_column_is_idempotent(
    tiger_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A second call leaves an existing FIPS column untouched and logs a skip."""
    gdf = merge_mod.read_shapefile(f"zip://{tiger_dir / FIXTURE_ZIPS[0]}")
    merge_mod.ensure_fips_column(gdf)
    original = gdf["FIPS"].copy()

    with caplog.at_level(logging.INFO):
        merge_mod.ensure_fips_column(gdf)

    assert (gdf["FIPS"] == original).all()
    assert any("already present" in r.getMessage() for r in caplog.records)


def test_filter_by_fips_keeps_only_requested_counties(tiger_dir: Path) -> None:
    """Filtering returns just the rows whose FIPS appears in the requested list."""
    paths = merge_mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = merge_mod.ensure_fips_column(merge_mod.merge_shapefiles(paths))

    wanted = ["11001", "24031", "51510"]
    selected = merge_mod.filter_by_fips(merged, wanted)

    assert set(selected["FIPS"].unique()) == set(wanted)
    assert len(selected) < len(merged)


def test_filter_by_fips_empty_list_returns_everything(tiger_dir: Path) -> None:
    """An empty FIPS filter is a passthrough."""
    paths = merge_mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = merge_mod.ensure_fips_column(merge_mod.merge_shapefiles(paths))

    selected = merge_mod.filter_by_fips(merged, [])

    assert len(selected) == len(merged)


def test_write_output_roundtrip_to_geopackage(tiger_dir: Path, tmp_path: Path) -> None:
    """Full pipeline writes a GeoPackage that reads back with the same feature count."""
    paths = merge_mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = merge_mod.ensure_fips_column(merge_mod.merge_shapefiles(paths))
    selected = merge_mod.filter_by_fips(merged, ["11001", "24031"])

    out_path = tmp_path / "out" / "blocks.gpkg"
    merge_mod.write_output(selected, str(out_path))

    assert out_path.exists()
    roundtripped = gpd.read_file(out_path)
    assert len(roundtripped) == len(selected)
    assert "FIPS" in roundtripped.columns
    assert set(roundtripped["FIPS"].unique()) == {"11001", "24031"}
