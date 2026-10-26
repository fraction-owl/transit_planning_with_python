"""Checks for the mock CBD fixture written by dev_tools/generate_mock_cbd.py.

The fixture exists so a "serves the CBD" route flag gets both answers on the
mock DC feed (tests/fixtures/mock_gtfs_dc.zip). These tests pin which mock
routes cross it and keep route 40 more than a quarter mile away.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

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


def test_fixture_is_one_valid_polygon() -> None:
    """The fixture loads from its zip as a single valid WGS84 polygon."""
    cbd = load_shapefile(str(CBD_ZIP))

    assert len(cbd) == 1
    assert cbd.crs.to_epsg() == 4326
    assert list(cbd.columns) == ["GIS_ID", "NAME", "geometry"]
    assert cbd.geometry.iloc[0].geom_type == "Polygon"
    assert cbd.geometry.iloc[0].is_valid


def test_fixture_matches_generator() -> None:
    """The committed zip is what the generator currently builds (regenerate if this fails)."""
    committed = load_shapefile(str(CBD_ZIP)).geometry.iloc[0]
    built = build_cbd_gdf().geometry.iloc[0]

    assert committed.normalize().equals_exact(built.normalize(), tolerance=1e-9)


def test_route_40_misses_the_cbd_and_the_other_routes_serve_it() -> None:
    """Routes 10, 20, 30, and 50H have stops inside the CBD; 40 is over a quarter mile out."""
    cbd = load_shapefile(str(CBD_ZIP)).to_crs(DC_CRS_FT).geometry.iloc[0]
    shapes, stops = _route_geometry()

    stops_inside = stops[stops.within(cbd)].groupby("route_id").size()
    assert set(stops_inside.index) == {"DC_R10", "DC_R20", "DC_R30", "DC_R50H"}
    assert (stops_inside >= 2).all()

    assert shapes.loc[shapes["route_id"] == "DC_R40"].distance(cbd).min() > QUARTER_MILE_FT
    assert stops.loc[stops["route_id"] == "DC_R40"].distance(cbd).min() > QUARTER_MILE_FT
