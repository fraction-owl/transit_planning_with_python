from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from utils.shapefile_helpers import load_shapefile


def _mk_point_shp(directory: Path, stem: str, name_value: str) -> Path:
    """Write a one-feature point shapefile named *stem*.shp under *directory*."""
    gdf = gpd.GeoDataFrame({"NAME": [name_value]}, geometry=[Point(-77.0, 38.9)], crs="EPSG:4326")
    path = directory / f"{stem}.shp"
    gdf.to_file(path)
    return path


def _zip_shp(shp_path: Path, zip_path: Path, arcname_stem: str | None = None) -> Path:
    """Zip a shapefile's sibling components (.shp/.shx/.dbf/...) into *zip_path*."""
    stem = arcname_stem or shp_path.stem
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for component in sorted(shp_path.parent.glob(f"{shp_path.stem}.*")):
            zf.write(component, f"{stem}{component.suffix}")
    return zip_path


def test_load_shapefile_direct_shp(tmp_path: Path) -> None:
    """A direct .shp path loads via gpd.read_file, unchanged from prior behavior."""
    shp_path = _mk_point_shp(tmp_path, "stops", "Main St")

    gdf = load_shapefile(str(shp_path))

    assert len(gdf) == 1
    assert gdf.loc[0, "NAME"] == "Main St"


def test_load_shapefile_zip_single_member_auto_detected(tmp_path: Path) -> None:
    """A zip containing exactly one shapefile loads without specifying `member`."""
    shp_dir = tmp_path / "src"
    shp_dir.mkdir()
    shp_path = _mk_point_shp(shp_dir, "stops", "Main St")
    zip_path = _zip_shp(shp_path, tmp_path / "stops.zip")

    gdf = load_shapefile(str(zip_path))

    assert len(gdf) == 1
    assert gdf.loc[0, "NAME"] == "Main St"


def test_load_shapefile_zip_multiple_members_requires_member(tmp_path: Path) -> None:
    """A zip with more than one shapefile raises unless `member` disambiguates."""
    shp_dir = tmp_path / "src"
    shp_dir.mkdir()
    stops_shp = _mk_point_shp(shp_dir, "stops", "Main St")
    hospitals_shp = _mk_point_shp(shp_dir, "hospitals", "General Hospital")
    zip_path = tmp_path / "layers.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for shp in (stops_shp, hospitals_shp):
            for component in sorted(shp.parent.glob(f"{shp.stem}.*")):
                zf.write(component, component.name)

    with pytest.raises(ValueError) as excinfo:
        load_shapefile(str(zip_path))
    msg = str(excinfo.value)
    assert "contains 2 shapefiles" in msg

    gdf = load_shapefile(str(zip_path), member="Hospitals.shp")
    assert gdf.loc[0, "NAME"] == "General Hospital"


def test_load_shapefile_zip_member_not_found_raises(tmp_path: Path) -> None:
    """An explicit `member` that isn't in the zip raises a clear ValueError."""
    shp_dir = tmp_path / "src"
    shp_dir.mkdir()
    shp_path = _mk_point_shp(shp_dir, "stops", "Main St")
    zip_path = _zip_shp(shp_path, tmp_path / "stops.zip")

    with pytest.raises(ValueError) as excinfo:
        load_shapefile(str(zip_path), member="Nonexistent.shp")
    assert "not found inside" in str(excinfo.value)


def test_load_shapefile_zip_with_no_shp_raises(tmp_path: Path) -> None:
    """A zip containing no .shp member raises a clear ValueError."""
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", "no shapefile here")

    with pytest.raises(ValueError) as excinfo:
        load_shapefile(str(zip_path))
    assert "No .shp file found" in str(excinfo.value)


def test_load_shapefile_bad_zip_raises(tmp_path: Path) -> None:
    """A .zip path that isn't actually a valid archive raises ValueError."""
    bad_zip = tmp_path / "not_really_a.zip"
    bad_zip.write_text("this is not a zip file", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_shapefile(str(bad_zip))
    assert "not a valid zip archive" in str(excinfo.value)


def test_load_shapefile_missing_path_raises(tmp_path: Path) -> None:
    """A path that does not exist at all raises OSError."""
    missing = tmp_path / "no_such_file.shp"

    with pytest.raises(OSError) as excinfo:
        load_shapefile(str(missing))
    assert str(missing) in str(excinfo.value)


def test_load_shapefile_unsupported_extension_raises(tmp_path: Path) -> None:
    """A path that's neither .shp nor .zip raises ValueError."""
    geojson_path = tmp_path / "stops.geojson"
    geojson_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_shapefile(str(geojson_path))
    assert "neither a .shp nor a .zip file" in str(excinfo.value)
