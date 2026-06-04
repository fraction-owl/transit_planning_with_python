from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from scripts.gtfs_exports.gtfs_to_shapefile_gpd import (
    GTFS_CRS,
    export_gdf,
    gtfs_to_shapefiles,
    read_shapes,
    read_stops,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    extracted = [p for p in dest.iterdir() if p.is_dir()]
    assert len(extracted) == 1, f"Expected one top-level dir, got: {extracted}"
    return extracted[0]


@pytest.fixture()
def dc_gtfs_dir(tmp_path: Path) -> Path:
    return _extract_zip(FIXTURES / "mock_gtfs_dc.zip", tmp_path)


@pytest.fixture()
def ottawa_gtfs_dir(tmp_path: Path) -> Path:
    return _extract_zip(FIXTURES / "mock_gtfs_ottawa.zip", tmp_path)


# ---------------------------------------------------------------------------
# read_stops — happy path
# ---------------------------------------------------------------------------


def test_read_stops_dc_row_count(dc_gtfs_dir: Path) -> None:
    """DC fixture loads the expected number of stops."""
    gdf = read_stops(dc_gtfs_dir)
    assert len(gdf) == 395


def test_read_stops_ottawa_row_count(ottawa_gtfs_dir: Path) -> None:
    """Ottawa fixture loads the expected number of stops."""
    gdf = read_stops(ottawa_gtfs_dir)
    assert len(gdf) == 619


def test_read_stops_returns_geodataframe(dc_gtfs_dir: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_read_stops_crs_is_wgs84(dc_gtfs_dir: Path) -> None:
    """Stops GeoDataFrame uses EPSG:4326."""
    gdf = read_stops(dc_gtfs_dir)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_read_stops_required_columns_present(dc_gtfs_dir: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    for col in ("stop_id", "stop_name", "stop_lat", "stop_lon", "geometry"):
        assert col in gdf.columns, f"Missing column: {col}"


def test_read_stops_geometry_is_point(dc_gtfs_dir: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    assert all(isinstance(g, Point) for g in gdf.geometry)


def test_read_stops_dc_coordinates_in_range(dc_gtfs_dir: Path) -> None:
    """DC stops fall within the Washington DC bounding box."""
    gdf = read_stops(dc_gtfs_dir)
    assert gdf["stop_lat"].between(38.0, 40.0).all()
    assert gdf["stop_lon"].between(-78.0, -76.0).all()


def test_read_stops_ottawa_coordinates_in_range(ottawa_gtfs_dir: Path) -> None:
    """Ottawa stops fall within the Ottawa bounding box."""
    gdf = read_stops(ottawa_gtfs_dir)
    assert gdf["stop_lat"].between(44.0, 46.0).all()
    assert gdf["stop_lon"].between(-77.0, -74.0).all()


# ---------------------------------------------------------------------------
# read_stops — error cases
# ---------------------------------------------------------------------------


def test_read_stops_missing_stops_txt_raises(tmp_path: Path) -> None:
    """FileNotFoundError when stops.txt is absent from the GTFS directory."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    with pytest.raises(FileNotFoundError):
        read_stops(gtfs)


def test_read_stops_missing_columns_raises(tmp_path: Path) -> None:
    """ValueError when stops.txt lacks required columns."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "stops.txt").write_text("stop_id,stop_name\n001,Main\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required columns"):
        read_stops(gtfs)


# ---------------------------------------------------------------------------
# read_shapes — happy path
# ---------------------------------------------------------------------------


def test_read_shapes_dc_shape_count(dc_gtfs_dir: Path) -> None:
    """DC fixture produces 5 distinct LineString features."""
    gdf = read_shapes(dc_gtfs_dir)
    assert len(gdf) == 5


def test_read_shapes_ottawa_shape_count(ottawa_gtfs_dir: Path) -> None:
    """Ottawa fixture produces 5 distinct LineString features."""
    gdf = read_shapes(ottawa_gtfs_dir)
    assert len(gdf) == 5


def test_read_shapes_returns_geodataframe(dc_gtfs_dir: Path) -> None:
    gdf = read_shapes(dc_gtfs_dir)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_read_shapes_crs_is_wgs84(dc_gtfs_dir: Path) -> None:
    """Shapes GeoDataFrame uses EPSG:4326."""
    gdf = read_shapes(dc_gtfs_dir)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_read_shapes_geometry_is_linestring(dc_gtfs_dir: Path) -> None:
    gdf = read_shapes(dc_gtfs_dir)
    assert all(isinstance(g, LineString) for g in gdf.geometry)


def test_read_shapes_has_shape_id_column(dc_gtfs_dir: Path) -> None:
    gdf = read_shapes(dc_gtfs_dir)
    assert "shape_id" in gdf.columns


def test_read_shapes_all_lines_have_at_least_two_points(dc_gtfs_dir: Path) -> None:
    """Every LineString contains at least 2 coordinate pairs."""
    gdf = read_shapes(dc_gtfs_dir)
    assert all(len(g.coords) >= 2 for g in gdf.geometry)


# ---------------------------------------------------------------------------
# read_shapes — missing / invalid file
# ---------------------------------------------------------------------------


def test_read_shapes_missing_file_returns_empty_gdf(tmp_path: Path) -> None:
    """Missing shapes.txt returns an empty GeoDataFrame instead of raising."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    gdf = read_shapes(gtfs)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.empty


def test_read_shapes_missing_columns_raises(tmp_path: Path) -> None:
    """ValueError when shapes.txt exists but lacks required columns."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "shapes.txt").write_text("shape_id,shape_pt_lat\nS1,38.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required columns"):
        read_shapes(gtfs)


# ---------------------------------------------------------------------------
# export_gdf
# ---------------------------------------------------------------------------


def test_export_gdf_writes_shapefile(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    out = tmp_path / "output" / "stops.shp"
    export_gdf(gdf, out)
    assert out.exists()


def test_export_gdf_creates_output_directory(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """export_gdf creates any missing parent directories automatically."""
    gdf = read_stops(dc_gtfs_dir)
    out = tmp_path / "deeply" / "nested" / "stops.shp"
    export_gdf(gdf, out)
    assert out.exists()


def test_export_gdf_skips_empty_geodataframe(tmp_path: Path) -> None:
    """No file is written when the GeoDataFrame is empty."""
    empty = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs=GTFS_CRS)
    out = tmp_path / "output" / "empty.shp"
    export_gdf(empty, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# gtfs_to_shapefiles — full pipeline
# ---------------------------------------------------------------------------


def test_gtfs_to_shapefiles_both_dc(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='both' creates stops and lines shapefiles for the DC feed."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="both")
    assert (out / "gtfs_stops.shp").exists()
    assert (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_both_ottawa(ottawa_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='both' creates stops and lines shapefiles for the Ottawa feed."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(ottawa_gtfs_dir, out, kind="both")
    assert (out / "gtfs_stops.shp").exists()
    assert (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_kind_stops_only(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='stops' writes only the stops shapefile."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="stops")
    assert (out / "gtfs_stops.shp").exists()
    assert not (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_kind_lines_only(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='lines' writes only the lines shapefile."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="lines")
    assert not (out / "gtfs_stops.shp").exists()
    assert (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_creates_output_dir(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """Output directory is created if it does not already exist."""
    out = tmp_path / "new_dir"
    assert not out.exists()
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="stops")
    assert out.is_dir()


# ---------------------------------------------------------------------------
# gtfs_to_shapefiles — error cases
# ---------------------------------------------------------------------------


def test_gtfs_to_shapefiles_nonexistent_gtfs_dir_raises(tmp_path: Path) -> None:
    """NotADirectoryError when the GTFS directory does not exist."""
    with pytest.raises(NotADirectoryError):
        gtfs_to_shapefiles(tmp_path / "no_such_dir", tmp_path / "out")


def test_gtfs_to_shapefiles_none_gtfs_dir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ValueError when gtfs_dir is None and the module default is also None."""
    import scripts.gtfs_exports.gtfs_to_shapefile_gpd as mod

    monkeypatch.setattr(mod, "DEFAULT_GTFS_DIR", None)
    with pytest.raises(ValueError, match="GTFS input directory"):
        gtfs_to_shapefiles(None, tmp_path / "out")


def test_gtfs_to_shapefiles_none_output_dir_raises(
    dc_gtfs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ValueError when output_dir is None and the module default is also None."""
    import scripts.gtfs_exports.gtfs_to_shapefile_gpd as mod

    monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", None)
    with pytest.raises(ValueError, match="Output directory"):
        gtfs_to_shapefiles(dc_gtfs_dir, None)
