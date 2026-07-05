from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

import scripts.service_coverage.gtfs_service_by_district_gpd as target

TARGET_EPSG = 2248  # matches the script default (Maryland State Plane, ftUS)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _stops_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_lat": ["38.90", "38.95"],
            "stop_lon": ["-77.00", "-77.05"],
        }
    )


def _gtfs_data() -> dict[str, pd.DataFrame]:
    return {
        "routes": pd.DataFrame({"route_id": ["R1", "R2"], "route_short_name": ["101", "202"]}),
        "trips": pd.DataFrame({"trip_id": ["T1", "T2"], "route_id": ["R1", "R2"]}),
        "stop_times": pd.DataFrame({"trip_id": ["T1", "T1", "T2"], "stop_id": ["S1", "S2", "S2"]}),
    }


def _intersect_gdf(pairs: list[tuple[str, str]]) -> gpd.GeoDataFrame:
    """Build a minimal (stop_id, DISTRICT) intersection layer."""
    return gpd.GeoDataFrame(
        {
            "stop_id": [p[0] for p in pairs],
            "DISTRICT": [p[1] for p in pairs],
        },
        geometry=[Point(0, 0)] * len(pairs),
        crs=f"EPSG:{TARGET_EPSG}",
    )


# ---------------------------------------------------------------------------
# create_projected_stops_gdf
# ---------------------------------------------------------------------------


def test_create_projected_stops_gdf_projects_to_target_epsg() -> None:
    gdf = target.create_projected_stops_gdf(_stops_df(), TARGET_EPSG)
    assert gdf.crs.to_epsg() == TARGET_EPSG
    assert len(gdf) == 2
    assert (gdf.geometry.type == "Point").all()


def test_create_projected_stops_gdf_coerces_string_coordinates() -> None:
    gdf = target.create_projected_stops_gdf(_stops_df(), TARGET_EPSG)
    assert gdf["stop_lat"].dtype == float
    assert gdf["stop_lon"].dtype == float


# ---------------------------------------------------------------------------
# buffer_stops_gdf
# ---------------------------------------------------------------------------


def test_buffer_stops_gdf_produces_polygons_of_expected_size() -> None:
    stops = target.create_projected_stops_gdf(_stops_df(), TARGET_EPSG)
    buffered = target.buffer_stops_gdf(stops, 1000.0)
    assert (buffered.geometry.type == "Polygon").all()
    # Buffer area approximates a circle of radius 1000.
    assert buffered.geometry.area.iloc[0] == pytest.approx(3.14159 * 1000**2, rel=0.01)


def test_buffer_stops_gdf_does_not_mutate_input() -> None:
    stops = target.create_projected_stops_gdf(_stops_df(), TARGET_EPSG)
    target.buffer_stops_gdf(stops, 1000.0)
    assert (stops.geometry.type == "Point").all()


# ---------------------------------------------------------------------------
# intersect_districts_gdf
# ---------------------------------------------------------------------------


def test_intersect_districts_gdf_keeps_overlapping_stops_only() -> None:
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1", "S2"]},
        geometry=[Point(100, 100), Point(9000, 9000)],
        crs=f"EPSG:{TARGET_EPSG}",
    )
    buffered = target.buffer_stops_gdf(stops, 50.0)
    district = gpd.GeoDataFrame(
        {"DISTRICT": ["D1"]},
        geometry=[Polygon([(0, 0), (0, 500), (500, 500), (500, 0)])],
        crs=f"EPSG:{TARGET_EPSG}",
    )
    out = target.intersect_districts_gdf(buffered, district)
    assert set(out["stop_id"]) == {"S1"}
    assert set(out["DISTRICT"]) == {"D1"}


# ---------------------------------------------------------------------------
# build_route_district_matrix
# ---------------------------------------------------------------------------


def test_build_route_district_matrix_marks_coverage() -> None:
    intersect = _intersect_gdf([("S1", "D1"), ("S2", "D2")])
    matrix = target.build_route_district_matrix(_gtfs_data(), intersect, "DISTRICT")
    # R1 (101) serves S1+S2 → both districts; R2 (202) serves S2 only → D2.
    r1 = matrix[matrix["route_short_name"] == "101"].iloc[0]
    r2 = matrix[matrix["route_short_name"] == "202"].iloc[0]
    assert r1["D1"] == "y" and r1["D2"] == "y"
    assert r2["D1"] == "n" and r2["D2"] == "y"


def test_build_route_district_matrix_excludes_routes_without_district_stops() -> None:
    intersect = _intersect_gdf([("S1", "D1")])  # only S1 falls in a district
    matrix = target.build_route_district_matrix(_gtfs_data(), intersect, "DISTRICT")
    # R2 serves only S2, which is in no district → not in the matrix.
    assert set(matrix["route_short_name"]) == {"101"}


def test_build_route_district_matrix_stop_in_multiple_districts() -> None:
    intersect = _intersect_gdf([("S2", "D1"), ("S2", "D2")])
    matrix = target.build_route_district_matrix(_gtfs_data(), intersect, "DISTRICT")
    r2 = matrix[matrix["route_short_name"] == "202"].iloc[0]
    assert r2["D1"] == "y" and r2["D2"] == "y"


# ---------------------------------------------------------------------------
# write_dataframe_to_excel
# ---------------------------------------------------------------------------


def test_write_dataframe_to_excel_round_trip(tmp_path: Path) -> None:
    df = pd.DataFrame({"route_short_name": ["101"], "D1": ["y"]})
    out = tmp_path / "matrix.xlsx"
    target.write_dataframe_to_excel(df, str(out))
    assert out.exists()
    read_back = pd.read_excel(out, sheet_name="districts_vs_routes")
    assert (
        read_back["route_short_name"].iloc[0] == 101
        or str(read_back["route_short_name"].iloc[0]) == "101"
    )
    assert read_back["D1"].iloc[0] == "y"


# ---------------------------------------------------------------------------
# load_gtfs_data
# ---------------------------------------------------------------------------


def test_load_gtfs_data_missing_directory_raises() -> None:
    with pytest.raises(OSError, match="does not exist"):
        target.load_gtfs_data("/nonexistent/gtfs")


def test_load_gtfs_data_missing_file_raises(tmp_path: Path) -> None:
    (tmp_path / "routes.txt").write_text("route_id\nR1\n")
    with pytest.raises(OSError, match="stops.txt"):
        target.load_gtfs_data(str(tmp_path), files=["routes.txt", "stops.txt"])


def test_load_gtfs_data_loads_requested_files_as_str(tmp_path: Path) -> None:
    (tmp_path / "routes.txt").write_text("route_id,route_short_name\nR1,007\n")
    data = target.load_gtfs_data(str(tmp_path), files=["routes.txt"])
    assert data["routes"]["route_short_name"].iloc[0] == "007"  # leading zeros kept
