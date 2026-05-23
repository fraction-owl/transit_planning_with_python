from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from scripts.census_tools import uscensus_blocks_table_join_gpd as join_mod

FIXTURE_DIR = Path("tests/fixtures")
DC_TIGER_ZIP = "tl_2025_11_tabblock20_sample.zip"
DC_BLOCK_COUNT = 14


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def blocks_gdf(tmp_path: Path) -> gpd.GeoDataFrame:
    """DC block geometry loaded via load_blocks from the sample TIGER zip."""
    staged = tmp_path / "tiger"
    staged.mkdir()
    shutil.copy(FIXTURE_DIR / DC_TIGER_ZIP, staged / DC_TIGER_ZIP)
    return join_mod.load_blocks(f"zip://{staged / DC_TIGER_ZIP}")


@pytest.fixture
def attrs_df(blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attribute DataFrame whose GEO_IDs match the first 5 blocks in blocks_gdf.

    GEO_ID values are left as 15-char GEOID20 strings so they join directly
    to blocks_gdf without needing load_attributes normalization.
    """
    geoids = blocks_gdf["GEOID20"].tolist()[:5]
    return pd.DataFrame(
        {
            "GEO_ID": geoids,
            "total_pop": list(range(5)),
        }
    )


# =============================================================================
# load_blocks
# =============================================================================


def test_load_blocks_returns_geodataframe_with_correct_row_count(tmp_path: Path) -> None:
    staged = tmp_path / "tiger"
    staged.mkdir()
    shutil.copy(FIXTURE_DIR / DC_TIGER_ZIP, staged / DC_TIGER_ZIP)

    gdf = join_mod.load_blocks(f"zip://{staged / DC_TIGER_ZIP}")

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == DC_BLOCK_COUNT


def test_load_blocks_coerces_key_to_str_15_chars(tmp_path: Path) -> None:
    staged = tmp_path / "tiger"
    staged.mkdir()
    shutil.copy(FIXTURE_DIR / DC_TIGER_ZIP, staged / DC_TIGER_ZIP)

    gdf = join_mod.load_blocks(f"zip://{staged / DC_TIGER_ZIP}")

    assert gdf["GEOID20"].dtype == object
    assert (gdf["GEOID20"].str.len() == 15).all()


def test_load_blocks_raises_key_error_on_missing_column(tmp_path: Path) -> None:
    staged = tmp_path / "tiger"
    staged.mkdir()
    shutil.copy(FIXTURE_DIR / DC_TIGER_ZIP, staged / DC_TIGER_ZIP)

    with pytest.raises(KeyError, match="NOT_A_COL"):
        join_mod.load_blocks(f"zip://{staged / DC_TIGER_ZIP}", key="NOT_A_COL")


# =============================================================================
# load_attributes
# =============================================================================


def test_load_attributes_trims_24char_geo_id_to_15(tmp_path: Path) -> None:
    csv_path = tmp_path / "attrs.csv"
    pd.DataFrame(
        {
            "GEO_ID": ["1000000US110010001001000", "1000000US110010001002000"],
            "pop": [100, 200],
        }
    ).to_csv(csv_path, index=False)

    result = join_mod.load_attributes(str(csv_path))

    assert (result["GEO_ID"].str.len() == 15).all()
    assert result["GEO_ID"].iloc[0] == "110010001001000"


def test_load_attributes_leaves_15char_geo_id_unchanged(tmp_path: Path) -> None:
    csv_path = tmp_path / "attrs.csv"
    pd.DataFrame({"GEO_ID": ["110010001001000", "110010001002000"]}).to_csv(
        csv_path, index=False
    )

    result = join_mod.load_attributes(str(csv_path))

    assert result["GEO_ID"].iloc[0] == "110010001001000"


def test_load_attributes_derives_key_from_geo_id_blk_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "attrs.csv"
    pd.DataFrame(
        {"GEO_ID_blk": ["1000000US110010001001000"], "pop": [10]}
    ).to_csv(csv_path, index=False)

    result = join_mod.load_attributes(str(csv_path))

    assert "GEO_ID" in result.columns
    assert result["GEO_ID"].iloc[0] == "110010001001000"


def test_load_attributes_raises_when_neither_key_present(tmp_path: Path) -> None:
    csv_path = tmp_path / "attrs.csv"
    pd.DataFrame({"unrelated_col": [1, 2]}).to_csv(csv_path, index=False)

    with pytest.raises(KeyError, match="Neither"):
        join_mod.load_attributes(str(csv_path))


# =============================================================================
# join_blocks_to_attributes
# =============================================================================


def test_join_left_preserves_all_block_rows(
    blocks_gdf: gpd.GeoDataFrame,
    attrs_df: pd.DataFrame,
) -> None:
    result = join_mod.join_blocks_to_attributes(blocks_gdf, attrs_df)

    assert len(result) == len(blocks_gdf)


def test_join_returns_geodataframe_with_geometry(
    blocks_gdf: gpd.GeoDataFrame,
    attrs_df: pd.DataFrame,
) -> None:
    result = join_mod.join_blocks_to_attributes(blocks_gdf, attrs_df)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.geometry.name in result.columns


def test_join_inner_keeps_only_matched_rows(
    blocks_gdf: gpd.GeoDataFrame,
    attrs_df: pd.DataFrame,
) -> None:
    result = join_mod.join_blocks_to_attributes(blocks_gdf, attrs_df, how="inner")

    assert len(result) == len(attrs_df)


def test_join_attribute_columns_present_in_result(
    blocks_gdf: gpd.GeoDataFrame,
    attrs_df: pd.DataFrame,
) -> None:
    result = join_mod.join_blocks_to_attributes(blocks_gdf, attrs_df)

    assert "total_pop" in result.columns


def test_join_raises_on_duplicate_attribute_keys(
    blocks_gdf: gpd.GeoDataFrame,
) -> None:
    geoid = blocks_gdf["GEOID20"].iloc[0]
    dup_attrs = pd.DataFrame({"GEO_ID": [geoid, geoid], "total_pop": [1, 2]})

    with pytest.raises(ValueError):  # pandas.errors.MergeError is a ValueError subclass
        join_mod.join_blocks_to_attributes(blocks_gdf, dup_attrs)


def test_join_casts_int64_attribute_column_to_float(
    blocks_gdf: gpd.GeoDataFrame,
) -> None:
    block = blocks_gdf.iloc[:1].copy()
    geoid = block["GEOID20"].iloc[0]
    attrs = pd.DataFrame(
        {"GEO_ID": [geoid], "count": pd.array([5], dtype="Int64")}
    )

    result = join_mod.join_blocks_to_attributes(block, attrs, how="inner")

    assert result["count"].dtype == "float64"


# =============================================================================
# save_output
# =============================================================================


def test_save_output_geopackage_roundtrip(
    blocks_gdf: gpd.GeoDataFrame,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "out" / "blocks.gpkg"
    join_mod.save_output(blocks_gdf, str(out_path))

    assert out_path.exists()
    roundtripped = gpd.read_file(out_path)
    assert len(roundtripped) == len(blocks_gdf)


def test_save_output_shapefile_roundtrip(
    blocks_gdf: gpd.GeoDataFrame,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "out" / "blocks.shp"
    join_mod.save_output(blocks_gdf, str(out_path))

    assert out_path.exists()
    roundtripped = gpd.read_file(out_path)
    assert len(roundtripped) == len(blocks_gdf)


def test_save_output_creates_missing_parent_dirs(
    blocks_gdf: gpd.GeoDataFrame,
    tmp_path: Path,
) -> None:
    deeply_nested = tmp_path / "a" / "b" / "c" / "out.gpkg"
    join_mod.save_output(blocks_gdf, str(deeply_nested))

    assert deeply_nested.exists()


# =============================================================================
# _cast_int64_to_float
# =============================================================================


def test_cast_int64_to_float_converts_integer_column() -> None:
    gdf = gpd.GeoDataFrame(
        {"val": pd.array([1, 2, 3], dtype="Int64"), "geometry": [Point(0, 0)] * 3}
    )

    join_mod._cast_int64_to_float(gdf)

    assert gdf["val"].dtype == "float64"


def test_cast_int64_to_float_leaves_string_column_unchanged() -> None:
    gdf = gpd.GeoDataFrame(
        {"label": ["a", "b", "c"], "geometry": [Point(0, 0)] * 3}
    )

    join_mod._cast_int64_to_float(gdf)

    assert gdf["label"].dtype == object


def test_cast_int64_to_float_noop_when_no_numeric_columns() -> None:
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})

    join_mod._cast_int64_to_float(gdf)  # must not raise
