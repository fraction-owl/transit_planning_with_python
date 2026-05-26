from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import LineString

import scripts.network_analysis.route_direction_classifier as mod
from scripts.network_analysis.route_direction_classifier import (
    classify_direction,
    create_lines_from_shapes,
    determine_dominant_shapes,
    flag_suspicious_data,
)

# ---------------------------------------------------------------------------
# classify_direction
# ---------------------------------------------------------------------------

# Helper: a projected line with endpoints far apart (> default loop_threshold=200)
_FAR = 500_000.0


def test_classify_direction_northbound() -> None:
    line_4326 = LineString([(0.0, 0.0), (0.0, 1.0)])       # lon, lat → going north
    line_proj = LineString([(0.0, 0.0), (0.0, _FAR)])
    assert classify_direction(line_4326, line_proj) == "NB"


def test_classify_direction_southbound() -> None:
    line_4326 = LineString([(0.0, 1.0), (0.0, 0.0)])       # going south
    line_proj = LineString([(0.0, _FAR), (0.0, 0.0)])
    assert classify_direction(line_4326, line_proj) == "SB"


def test_classify_direction_eastbound() -> None:
    line_4326 = LineString([(0.0, 0.5), (1.0, 0.5)])       # going east
    line_proj = LineString([(0.0, 0.0), (_FAR, 0.0)])
    assert classify_direction(line_4326, line_proj) == "EB"


def test_classify_direction_westbound() -> None:
    line_4326 = LineString([(1.0, 0.5), (0.0, 0.5)])       # going west
    line_proj = LineString([(_FAR, 0.0), (0.0, 0.0)])
    assert classify_direction(line_4326, line_proj) == "WB"


def test_classify_direction_counter_clockwise_loop() -> None:
    # start and end are within loop_threshold distance in projected coords
    line_4326 = LineString([(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)])
    # CCW square: (0,0)→(100,0)→(100,100)→(0,100); end dist to start = 100 < 200
    line_proj = LineString([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert classify_direction(line_4326, line_proj) == "CCW"


def test_classify_direction_clockwise_loop() -> None:
    line_4326 = LineString([(0.0, 0.0), (0.0, 0.001), (0.001, 0.001), (0.001, 0.0)])
    # CW square: (0,0)→(0,100)→(100,100)→(100,0); end dist to start = 100 < 200
    line_proj = LineString([(0, 0), (0, 100), (100, 100), (100, 0)])
    assert classify_direction(line_4326, line_proj) == "CW"


def test_classify_direction_custom_loop_threshold() -> None:
    # end is 150 units from start; with threshold=200 it's a loop, threshold=100 it's not
    line_4326 = LineString([(0.0, 0.0), (0.001, 0.0)])
    line_proj = LineString([(0, 0), (150, 0)])
    assert classify_direction(line_4326, line_proj, loop_threshold=200) != classify_direction(
        line_4326, line_proj, loop_threshold=100
    )


# ---------------------------------------------------------------------------
# create_lines_from_shapes
# ---------------------------------------------------------------------------


@pytest.fixture()
def shapes_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shape_id": ["S1", "S1", "S2", "S2"],
            "shape_pt_sequence": ["1", "2", "1", "2"],
            "shape_pt_lat": ["38.7", "38.8", "38.9", "39.0"],
            "shape_pt_lon": ["-77.0", "-77.0", "-77.1", "-77.1"],
        }
    )


def test_create_lines_from_shapes_returns_geodataframe(shapes_df: pd.DataFrame) -> None:
    import geopandas as gpd

    gdf = create_lines_from_shapes(shapes_df)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_create_lines_from_shapes_correct_row_count(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert len(gdf) == 2


def test_create_lines_from_shapes_geometry_is_linestring(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert all(isinstance(g, LineString) for g in gdf.geometry)


def test_create_lines_from_shapes_has_shape_id_column(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert "shape_id" in gdf.columns


def test_create_lines_from_shapes_crs_is_wgs84(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# determine_dominant_shapes
# ---------------------------------------------------------------------------


@pytest.fixture()
def final_data_df() -> pd.DataFrame:
    # SHP1 has 3 trips, SHP2 has 2 trips → SHP1 is dominant
    return pd.DataFrame(
        {
            "route_short_name": ["R1"] * 5,
            "direction_id": ["0"] * 5,
            "shape_id": ["SHP1", "SHP1", "SHP1", "SHP2", "SHP2"],
            "other_col": range(5),
        }
    )


def test_determine_dominant_shapes_flags_most_common(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    shp1_dominant = result.loc[result.shape_id == "SHP1", "is_dominant"]
    assert shp1_dominant.all()


def test_determine_dominant_shapes_non_dominant_gets_nan(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    shp2_dominant = result.loc[result.shape_id == "SHP2", "is_dominant"]
    assert shp2_dominant.isna().all()


def test_determine_dominant_shapes_preserves_row_count(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    assert len(result) == len(final_data_df)


def test_determine_dominant_shapes_has_is_dominant_column(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    assert "is_dominant" in result.columns


# ---------------------------------------------------------------------------
# flag_suspicious_data
# ---------------------------------------------------------------------------


def test_flag_suspicious_data_clean_data_creates_no_file(tmp_path: pytest.MonkeyPatch) -> None:
    summary = pd.DataFrame(
        {
            "route_short_name": ["R1", "R2"],
            "direction_id": ["0", "1"],
            "shape_direction": ["NB", "SB"],
            "shape_id": ["SHP1", "SHP2"],
        }
    )
    # No flags expected – function should not attempt to write a file
    flag_suspicious_data(summary)  # must not raise


def test_flag_suspicious_data_writes_file_on_suspicious_routes(
    tmp_path: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "OUTPUT_FOLDER", str(tmp_path))
    # Two direction_ids share the same shape_direction → suspicious
    summary = pd.DataFrame(
        {
            "route_short_name": ["R1", "R1"],
            "direction_id": ["0", "1"],
            "shape_direction": ["NB", "NB"],
            "shape_id": ["SHP1", "SHP2"],
        }
    )
    flag_suspicious_data(summary)
    assert (tmp_path / "Suspicious_RouteDirections.xlsx").exists()
