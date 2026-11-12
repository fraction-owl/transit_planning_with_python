"""Zipped-shapefile inputs for the GeoPandas scripts that copy ``load_shapefile``.

Each loader must read a ``.zip`` whose shapefile sits in a subfolder (which
``gpd.read_file`` alone cannot open) and ignore macOS ``__MACOSX`` metadata,
while ``.shp`` paths keep reading as before.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon


def _zip_layer(gdf: gpd.GeoDataFrame, tmp_path: Path, stem: str) -> tuple[Path, Path]:
    """Write *gdf* as ``<stem>.shp`` and as ``<stem>.zip`` (inside ``data/``); return both."""
    shp_dir = tmp_path / f"{stem}_shp"
    shp_dir.mkdir()
    shp = shp_dir / f"{stem}.shp"
    gdf.to_file(shp)
    zipped = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        for part in shp_dir.iterdir():
            zf.write(part, f"data/{part.name}")
        zf.writestr(f"__MACOSX/data/._{stem}.shp", b"")
    return shp, zipped


@pytest.fixture
def points(tmp_path: Path) -> tuple[Path, Path]:
    gdf = gpd.GeoDataFrame(
        {"NAME": ["A", "B"], "StopId": ["1", "2"], "StopNum": ["10", "20"]},
        geometry=[Point(-77.03, 38.90), Point(-77.02, 38.91)],
        crs="EPSG:4326",
    )
    return _zip_layer(gdf, tmp_path, "points")


@pytest.fixture
def lines(tmp_path: Path) -> tuple[Path, Path]:
    gdf = gpd.GeoDataFrame(
        {"FULLNAME": ["Main St", "Oak Ave"]},
        geometry=[
            LineString([(-77.03, 38.90), (-77.02, 38.90)]),
            LineString([(-77.03, 38.91), (-77.02, 38.91)]),
        ],
        crs="EPSG:4326",
    )
    return _zip_layer(gdf, tmp_path, "lines")


@pytest.fixture
def polygons(tmp_path: Path) -> tuple[Path, Path]:
    gdf = gpd.GeoDataFrame(
        {"DISTRICT": ["North", "South"], "NAME": ["North", "South"]},
        geometry=[
            Polygon([(-77.1, 38.9), (-77.0, 38.9), (-77.0, 39.0), (-77.1, 39.0)]),
            Polygon([(-77.1, 38.8), (-77.0, 38.8), (-77.0, 38.9), (-77.1, 38.9)]),
        ],
        crs="EPSG:4326",
    )
    return _zip_layer(gdf, tmp_path, "polygons")


def _same(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> None:
    pd.testing.assert_frame_equal(
        pd.DataFrame(a.drop(columns="geometry")), b.drop(columns="geometry")
    )
    assert list(a.geometry.to_wkt()) == list(b.geometry.to_wkt())


def test_plain_read_file_cannot_open_nested_zip(points: tuple[Path, Path]) -> None:
    """Guards the premise: without load_shapefile, a nested zip fails."""
    _, zipped = points
    with pytest.raises(Exception):  # noqa: B017 - pyogrio's DataSourceError
        gpd.read_file(zipped)


def test_stop_vs_roadname_load_roadways(lines: tuple[Path, Path]) -> None:
    from scripts.gtfs_data_quality.stop_vs_roadname_checker_gpd import load_roadways

    shp, zipped = lines
    _same(load_roadways(str(zipped)), load_roadways(str(shp)))


def test_stop_v_conflict_load_context(polygons: tuple[Path, Path]) -> None:
    from scripts.gtfs_data_quality.stop_v_conflict_checker_gpd import _load_context

    shp, zipped = polygons
    from_zip = _load_context(str(zipped), "EPSG:2248")
    from_shp = _load_context(str(shp), "EPSG:2248")
    assert from_zip is not None and from_shp is not None
    assert from_zip.crs == from_shp.crs
    assert len(from_zip) == len(from_shp) == 2


def test_cluster_stops_load_zones(polygons: tuple[Path, Path]) -> None:
    from scripts.facilities_tools.cluster_stops_from_zones_gpd import load_zones

    shp, zipped = polygons
    assert len(load_zones(str(zipped), "NAME")) == len(load_zones(str(shp), "NAME")) == 2


def test_district_ridership_share_load_districts(polygons: tuple[Path, Path]) -> None:
    from scripts.ridership_tools.district_ridership_share_gpd import load_districts

    shp, zipped = polygons
    from_zip = load_districts(str(zipped), "DISTRICT")
    from_shp = load_districts(str(shp), "DISTRICT")
    assert sorted(from_zip["DISTRICT"]) == sorted(from_shp["DISTRICT"]) == ["North", "South"]


def test_stop_removal_load_centerlines(lines: tuple[Path, Path]) -> None:
    from scripts.stop_analysis.stop_removal_impact_gpd import load_centerlines

    shp, zipped = lines
    _same(load_centerlines(zipped, 2248), load_centerlines(shp, 2248))


def test_site_route_proximity_load_locations(points: tuple[Path, Path]) -> None:
    from scripts.service_coverage.site_route_proximity_gpd import _load_locations

    shp, zipped = points
    from_zip = _load_locations("shapefile", shp_path=str(zipped), name_field="NAME")
    from_shp = _load_locations("shapefile", shp_path=str(shp), name_field="NAME")
    _same(from_zip, from_shp)


def test_stops_ridership_joiner_reads_zipped_stop_shapefile(
    points: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .zip without stops.txt is a zipped stop layer, not a GTFS feed."""
    import scripts.ridership_tools.stops_ridership_joiner_gpd as joiner

    _, zipped = points
    monkeypatch.setattr(joiner, "BUS_STOPS_INPUT", zipped)
    gdf, key_field = joiner.load_bus_stops()
    assert key_field == joiner.SHAPE_KEY_FIELD
    assert sorted(gdf["StopId"]) == ["1", "2"]
