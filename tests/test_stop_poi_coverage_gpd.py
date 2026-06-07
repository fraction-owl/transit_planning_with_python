from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from scripts.service_coverage.stop_poi_coverage_gpd import (
    BUFFER_DIST_FT,
    build_stop_buffers,
    count_pois_within_buffers,
    extract_config_block,
    load_gtfs_stops,
)

# A projected CRS in US feet, so buffer distances are applied directly in feet.
FEET_CRS = "EPSG:2248"


# ---------------------------------------------------------------------------
# build_stop_buffers
# ---------------------------------------------------------------------------


def _single_stop(x: float, y: float, stop_id: str = "A") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"stop_id": [stop_id]},
        geometry=[Point(x, y)],
        crs=FEET_CRS,
    )


def test_build_stop_buffers_area_matches_radius() -> None:
    buffers = build_stop_buffers(_single_stop(0.0, 0.0), BUFFER_DIST_FT, FEET_CRS)
    expected_area = math.pi * BUFFER_DIST_FT**2
    # Buffer polygon slightly under-approximates the true circle; allow 1%.
    assert buffers.geometry.iloc[0].area == pytest.approx(expected_area, rel=0.01)


def test_build_stop_buffers_preserves_stop_id_and_crs() -> None:
    buffers = build_stop_buffers(_single_stop(0.0, 0.0, "XYZ"), BUFFER_DIST_FT, FEET_CRS)
    assert list(buffers["stop_id"]) == ["XYZ"]
    assert buffers.crs == FEET_CRS


def test_build_stop_buffers_returns_polygons() -> None:
    buffers = build_stop_buffers(_single_stop(0.0, 0.0), BUFFER_DIST_FT, FEET_CRS)
    assert buffers.geometry.iloc[0].geom_type == "Polygon"


# ---------------------------------------------------------------------------
# count_pois_within_buffers
# ---------------------------------------------------------------------------


def _two_stop_buffers() -> gpd.GeoDataFrame:
    # Two stops 5000 ft apart, each buffered by 1320 ft (no overlap).
    return gpd.GeoDataFrame(
        {"stop_id": ["A", "B"]},
        geometry=[Point(0.0, 0.0).buffer(1320.0), Point(5000.0, 0.0).buffer(1320.0)],
        crs=FEET_CRS,
    )


def _points(*coords: tuple[float, float]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs=FEET_CRS)


def test_count_assigns_pois_to_correct_stop() -> None:
    layers = {"grocery": _points((100.0, 0.0), (5000.0, 0.0), (10000.0, 0.0))}
    result = count_pois_within_buffers(_two_stop_buffers(), layers, ["grocery"])

    by_stop = result.set_index("stop_id")
    assert by_stop.loc["A", "grocery"] == 1  # (100, 0) inside A
    assert by_stop.loc["B", "grocery"] == 1  # (5000, 0) inside B; (10000, 0) outside both


def test_count_missing_category_is_zero_column() -> None:
    result = count_pois_within_buffers(_two_stop_buffers(), {}, ["libraries"])
    assert list(result["libraries"]) == [0, 0]


def test_count_empty_layer_is_zero_column() -> None:
    layers = {"libraries": _points()}
    result = count_pois_within_buffers(_two_stop_buffers(), layers, ["libraries"])
    assert list(result["libraries"]) == [0, 0]


def test_count_poi_total_sums_categories() -> None:
    layers = {
        "grocery": _points((100.0, 0.0), (5000.0, 0.0)),
        "library": _points((50.0, 50.0)),
    }
    result = count_pois_within_buffers(_two_stop_buffers(), layers, ["grocery", "library"])
    by_stop = result.set_index("stop_id")
    assert by_stop.loc["A", "poi_total"] == 2  # 1 grocery + 1 library
    assert by_stop.loc["B", "poi_total"] == 1  # 1 grocery


def test_count_preserves_category_order() -> None:
    result = count_pois_within_buffers(_two_stop_buffers(), {}, ["b", "a", "c"])
    assert list(result.columns) == ["stop_id", "b", "a", "c", "poi_total"]


# ---------------------------------------------------------------------------
# load_gtfs_stops
# ---------------------------------------------------------------------------


def test_load_gtfs_stops_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gtfs_stops(tmp_path, "stop_id", "EPSG:4326")


def test_load_gtfs_stops_missing_columns_raises(tmp_path: Path) -> None:
    pd.DataFrame({"stop_id": ["A"]}).to_csv(tmp_path / "stops.txt", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_gtfs_stops(tmp_path, "stop_id", "EPSG:4326")


def test_load_gtfs_stops_builds_points(tmp_path: Path) -> None:
    pd.DataFrame(
        {"stop_id": ["A"], "stop_name": ["Main & 1st"], "stop_lat": [38.74], "stop_lon": [-77.08]}
    ).to_csv(tmp_path / "stops.txt", index=False)

    gdf = load_gtfs_stops(tmp_path, "stop_id", "EPSG:4326")
    assert gdf.crs == "EPSG:4326"
    assert gdf.geometry.iloc[0].geom_type == "Point"
    assert gdf.loc[0, "stop_id"] == "A"


# ---------------------------------------------------------------------------
# extract_config_block
# ---------------------------------------------------------------------------


def test_extract_config_block_returns_config_text() -> None:
    source = Path("scripts/service_coverage/stop_poi_coverage_gpd.py")
    block = extract_config_block(source)
    assert "BUFFER_DIST_FT" in block
    assert "LAYER_SPECS" in block
