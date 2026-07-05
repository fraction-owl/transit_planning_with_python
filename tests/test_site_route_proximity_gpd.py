from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

import scripts.service_coverage.site_route_proximity_gpd as target

# ---------------------------------------------------------------------------
# _check_gtfs / _load_gtfs
# ---------------------------------------------------------------------------

GTFS_FILES = ("stops.txt", "stop_times.txt", "trips.txt", "routes.txt")


def _write_minimal_gtfs(gtfs_dir: Path) -> None:
    (gtfs_dir / "stops.txt").write_text("stop_id,stop_lat,stop_lon\nS1,38.9,-77.0\n")
    (gtfs_dir / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\nT1,S1,1\n")
    (gtfs_dir / "trips.txt").write_text("trip_id,route_id,direction_id\nT1,R1,0\n")
    (gtfs_dir / "routes.txt").write_text("route_id,route_short_name\nR1,101\n")


def test_check_gtfs_passes_when_all_files_present(tmp_path: Path) -> None:
    _write_minimal_gtfs(tmp_path)
    target._check_gtfs(str(tmp_path))  # should not raise


def test_check_gtfs_missing_file_raises(tmp_path: Path) -> None:
    _write_minimal_gtfs(tmp_path)
    (tmp_path / "routes.txt").unlink()
    with pytest.raises(FileNotFoundError, match="routes.txt"):
        target._check_gtfs(str(tmp_path))


def test_load_gtfs_returns_string_dataframes(tmp_path: Path) -> None:
    _write_minimal_gtfs(tmp_path)
    gtfs = target._load_gtfs(str(tmp_path))
    assert set(gtfs) == {"stops", "stop_times", "trips", "routes"}
    assert gtfs["stops"]["stop_id"].iloc[0] == "S1"
    assert gtfs["stops"]["stop_lat"].dtype == object  # loaded as str


# ---------------------------------------------------------------------------
# _load_locations
# ---------------------------------------------------------------------------


def test_load_locations_manual_builds_wgs84_points() -> None:
    manual = [{"name": "Braddock", "latitude": 38.81, "longitude": -77.05}]
    gdf = target._load_locations("manual", manual_list=manual)
    assert gdf.crs.to_string() == "EPSG:4326"
    assert gdf["name"].iloc[0] == "Braddock"
    assert gdf.geometry.iloc[0].x == pytest.approx(-77.05)
    assert gdf.geometry.iloc[0].y == pytest.approx(38.81)


def test_load_locations_manual_without_list_raises() -> None:
    with pytest.raises(ValueError, match="manual_list"):
        target._load_locations("manual", manual_list=None)


def test_load_locations_shapefile_renames_name_field(tmp_path: Path) -> None:
    shp = tmp_path / "points.shp"
    gpd.GeoDataFrame(
        {"SITE": ["Depot"]},
        geometry=[Point(-77.0, 38.9)],
        crs="EPSG:4326",
    ).to_file(shp)
    gdf = target._load_locations("shapefile", shp_path=str(shp), name_field="SITE")
    assert gdf["name"].iloc[0] == "Depot"


def test_load_locations_shapefile_without_path_raises() -> None:
    with pytest.raises(ValueError, match="shp_path"):
        target._load_locations("shapefile", shp_path=None)


def test_load_locations_invalid_source_raises() -> None:
    with pytest.raises(ValueError, match="LOCATION_SOURCE"):
        target._load_locations("database")


# ---------------------------------------------------------------------------
# _stops_to_gdf
# ---------------------------------------------------------------------------


def test_stops_to_gdf_builds_points_from_lat_lon_strings() -> None:
    stops = pd.DataFrame({"stop_id": ["S1"], "stop_lat": ["38.9"], "stop_lon": ["-77.0"]})
    gdf = target._stops_to_gdf(stops)
    assert gdf.crs.to_string() == "EPSG:4326"
    assert gdf.geometry.iloc[0].x == pytest.approx(-77.0)


# ---------------------------------------------------------------------------
# _distance_ft
# ---------------------------------------------------------------------------


def test_distance_ft_converts_miles() -> None:
    assert target._distance_ft(0.25, "miles") == 1320.0
    assert target._distance_ft(1, "Miles") == 5280.0


def test_distance_ft_feet_passthrough() -> None:
    assert target._distance_ft(500, "feet") == 500


# ---------------------------------------------------------------------------
# _apply_route_filters
# ---------------------------------------------------------------------------


def _routes_df() -> pd.DataFrame:
    return pd.DataFrame({"route_short_name": ["101", "202", "9999A"]})


def test_apply_route_filters_out_drops_listed_routes() -> None:
    with (
        patch.object(target, "ROUTE_FILTER_IN", []),
        patch.object(target, "ROUTE_FILTER_OUT", ["9999A"]),
    ):
        out = target._apply_route_filters(_routes_df())
    assert set(out["route_short_name"]) == {"101", "202"}


def test_apply_route_filters_in_keeps_only_listed_routes() -> None:
    with (
        patch.object(target, "ROUTE_FILTER_IN", ["101"]),
        patch.object(target, "ROUTE_FILTER_OUT", []),
    ):
        out = target._apply_route_filters(_routes_df())
    assert set(out["route_short_name"]) == {"101"}


def test_apply_route_filters_no_filters_is_noop() -> None:
    with (
        patch.object(target, "ROUTE_FILTER_IN", []),
        patch.object(target, "ROUTE_FILTER_OUT", []),
    ):
        out = target._apply_route_filters(_routes_df())
    assert len(out) == 3


# ---------------------------------------------------------------------------
# _nearby_routes
# ---------------------------------------------------------------------------


def _projected_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Planar-CRS location/stop layers: one stop 100 ft away, one 10,000 ft away."""
    crs = "EPSG:2232"
    locations = gpd.GeoDataFrame(
        {"name": ["Site A", "Site B"]},
        geometry=[Point(0, 0), Point(100000, 100000)],
        crs=crs,
    )
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1", "S2"]},
        geometry=[Point(100, 0), Point(10000, 0)],
        crs=crs,
    )
    st_trips_routes = pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "route_short_name": ["101", "202"],
            "direction_id": ["0", "1"],
        }
    )
    return locations, stops, st_trips_routes


def test_nearby_routes_lists_routes_within_buffer() -> None:
    locations, stops, st_trips_routes = _projected_inputs()
    rows = target._nearby_routes(locations, stops, st_trips_routes, buf_ft=1320.0, extra_cols=[])
    site_a = next(r for r in rows if r["Location"] == "Site A")
    assert site_a["Routes"] == "101 (dir 0)"
    assert site_a["Stops"] == "S1"


def test_nearby_routes_no_stops_in_buffer() -> None:
    locations, stops, st_trips_routes = _projected_inputs()
    rows = target._nearby_routes(locations, stops, st_trips_routes, buf_ft=1320.0, extra_cols=[])
    site_b = next(r for r in rows if r["Location"] == "Site B")
    assert site_b["Routes"] == "No routes"
    assert site_b["Stops"] == "No stops"


def test_nearby_routes_includes_extra_columns() -> None:
    locations, stops, st_trips_routes = _projected_inputs()
    locations["TYPE"] = ["School", None]
    rows = target._nearby_routes(
        locations, stops, st_trips_routes, buf_ft=1320.0, extra_cols=["TYPE"]
    )
    site_a = next(r for r in rows if r["Location"] == "Site A")
    site_b = next(r for r in rows if r["Location"] == "Site B")
    assert site_a["TYPE"] == "School"
    assert site_b["TYPE"] == ""  # NaN rendered as empty string


def test_nearby_routes_reports_nearest_stop_per_route_direction() -> None:
    crs = "EPSG:2232"
    locations = gpd.GeoDataFrame({"name": ["Site A"]}, geometry=[Point(0, 0)], crs=crs)
    # Two stops on the same route/direction; only the closest is reported.
    stops = gpd.GeoDataFrame(
        {"stop_id": ["NEAR", "FAR"]},
        geometry=[Point(100, 0), Point(900, 0)],
        crs=crs,
    )
    st_trips_routes = pd.DataFrame(
        {
            "stop_id": ["NEAR", "FAR"],
            "route_short_name": ["101", "101"],
            "direction_id": ["0", "0"],
        }
    )
    rows = target._nearby_routes(locations, stops, st_trips_routes, buf_ft=1320.0, extra_cols=[])
    assert rows[0]["Stops"] == "NEAR"
    assert rows[0]["Routes"] == "101 (dir 0)"
