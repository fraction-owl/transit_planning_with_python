"""Checks for the mock CBD fixture written by dev_tools/generate_mock_cbd.py.

The fixture exists so a "serves the CBD" route flag gets both answers on the
mock DC feed (tests/fixtures/mock_gtfs_dc.zip). These tests pin which mock
routes stop in each of its two polygons, and that route 40 crosses the annex
without stopping in it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from dev_tools.generate_mock_cbd import build_cbd_gdf
from utils.shapefile_helpers import load_shapefile

FIXTURES = Path(__file__).parent / "fixtures"
CBD_ZIP = FIXTURES / "mock_cbd_dc.zip"
GTFS_ZIP = FIXTURES / "mock_gtfs_dc.zip"

DC_CRS_FT = "EPSG:2248"
QUARTER_MILE_FT = 1320.0


def _read_gtfs(name: str) -> pd.DataFrame:
    with zipfile.ZipFile(GTFS_ZIP) as zf:
        return pd.read_csv(zf.open(f"dc/{name}"), dtype=str)


def _route_geometry() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (route shapes, route stops) from the mock DC feed in DC State Plane feet."""
    trips = _read_gtfs("trips.txt")
    shapes = _read_gtfs("shapes.txt")
    shapes["shape_pt_sequence"] = shapes["shape_pt_sequence"].astype(int)
    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])
    lines = {
        shape_id: LineString(zip(g["shape_pt_lon"].astype(float), g["shape_pt_lat"].astype(float)))
        for shape_id, g in shapes.groupby("shape_id")
    }
    route_shapes = trips[["route_id", "shape_id"]].drop_duplicates()
    shape_gdf = gpd.GeoDataFrame(
        route_shapes, geometry=[lines[s] for s in route_shapes["shape_id"]], crs="EPSG:4326"
    )

    stop_times = _read_gtfs("stop_times.txt")[["trip_id", "stop_id"]]
    route_stops = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id")
    route_stops = route_stops[["route_id", "stop_id"]].drop_duplicates()
    route_stops = route_stops.merge(_read_gtfs("stops.txt"), on="stop_id")
    stop_gdf = gpd.GeoDataFrame(
        route_stops[["route_id", "stop_id"]],
        geometry=gpd.points_from_xy(
            route_stops["stop_lon"].astype(float), route_stops["stop_lat"].astype(float)
        ),
        crs="EPSG:4326",
    )
    return shape_gdf.to_crs(DC_CRS_FT), stop_gdf.to_crs(DC_CRS_FT)


def _load_cbd_ft() -> dict[str, BaseGeometry]:
    """Return the fixture's polygons by GIS_ID, in DC State Plane feet."""
    cbd = load_shapefile(str(CBD_ZIP)).to_crs(DC_CRS_FT)
    return dict(zip(cbd["GIS_ID"], cbd.geometry))


def test_fixture_is_two_valid_polygons() -> None:
    """The fixture loads from its zip as two valid WGS84 polygons."""
    cbd = load_shapefile(str(CBD_ZIP))

    assert cbd.crs.to_epsg() == 4326
    assert list(cbd.columns) == ["GIS_ID", "NAME", "geometry"]
    assert cbd["GIS_ID"].tolist() == ["MOCK_CBD_1", "MOCK_CBD_2"]
    assert (cbd.geometry.geom_type == "Polygon").all()
    assert cbd.geometry.is_valid.all()


def test_fixture_matches_generator() -> None:
    """The committed zip is what the generator currently builds (regenerate if this fails)."""
    committed = load_shapefile(str(CBD_ZIP)).geometry
    built = build_cbd_gdf().geometry

    assert len(committed) == len(built)
    for kept, fresh in zip(committed, built):
        assert kept.normalize().equals_exact(fresh.normalize(), tolerance=1e-9)


def test_four_routes_stop_downtown_and_none_stop_in_the_annex() -> None:
    """Routes 10, 20, 30, and 50H stop in downtown only; no route stops in the annex."""
    polygons = _load_cbd_ft()
    _, stops = _route_geometry()

    downtown = stops[stops.intersects(polygons["MOCK_CBD_1"])].groupby("route_id").size()
    assert set(downtown.index) == {"DC_R10", "DC_R20", "DC_R30", "DC_R50H"}
    assert (downtown >= 2).all()
    assert not stops.intersects(polygons["MOCK_CBD_2"]).any()


def test_route_40_crosses_the_annex_without_stopping_in_any_polygon() -> None:
    """Route 40's line cuts the annex, but every route 40 stop is outside both polygons."""
    polygons = _load_cbd_ft()
    shapes, stops = _route_geometry()
    r40_shapes = shapes[shapes["route_id"] == "DC_R40"]
    r40_stops = stops[stops["route_id"] == "DC_R40"]

    assert r40_shapes.intersects(polygons["MOCK_CBD_2"]).any()
    assert r40_stops.distance(polygons["MOCK_CBD_2"]).min() > 50.0
    assert r40_shapes.distance(polygons["MOCK_CBD_1"]).min() > QUARTER_MILE_FT
