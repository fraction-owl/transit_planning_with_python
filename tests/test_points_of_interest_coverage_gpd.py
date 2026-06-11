from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
from shapely.geometry import Point

matplotlib.use("Agg")

from scripts.service_coverage.points_of_interest_coverage_gpd import (
    _count_features,
    _load_gtfs_tables,
    _load_layers,
    _prepare_route_buffers,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_gtfs_files(gtfs_dir: Path) -> None:
    """Write a minimal single-route GTFS feed (R1 / T1 / SH1) into *gtfs_dir*."""
    (gtfs_dir / "routes.txt").write_text("route_id\nR1\n")
    (gtfs_dir / "trips.txt").write_text("route_id,trip_id,shape_id\nR1,T1,SH1\n")
    (gtfs_dir / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\n")
    (gtfs_dir / "stops.txt").write_text("stop_id,stop_lat,stop_lon\nS1,0.0,0.0\nS2,0.01,0.01\n")
    (gtfs_dir / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\nSH1,0.0,0.0,1\nSH1,0.01,0.01,2\n"
    )


def _minimal_tables() -> dict[str, pd.DataFrame]:
    """Return in-memory GTFS DataFrames for a single route R1 (shape SH1)."""
    return {
        "routes": pd.DataFrame({"route_id": ["R1"]}),
        "trips": pd.DataFrame({"route_id": ["R1"], "trip_id": ["T1"], "shape_id": ["SH1"]}),
        "stop_times": pd.DataFrame(
            {"trip_id": ["T1", "T1"], "stop_id": ["S1", "S2"], "stop_sequence": [1, 2]}
        ),
        "stops": pd.DataFrame(
            {"stop_id": ["S1", "S2"], "stop_lat": [0.0, 0.01], "stop_lon": [0.0, 0.01]}
        ),
        "shapes": pd.DataFrame(
            {
                "shape_id": ["SH1", "SH1"],
                "shape_pt_lat": [0.0, 0.01],
                "shape_pt_lon": [0.0, 0.01],
                "shape_pt_sequence": [1, 2],
            }
        ),
    }


def _write_point_shp(directory: Path, shp_name: str, lon: float, lat: float) -> Path:
    """Write a one-feature Point shapefile to *directory*/*shp_name* and return its path."""
    gdf = gpd.GeoDataFrame(
        {"NAME": ["Feature"]},
        geometry=[Point(lon, lat)],
        crs="EPSG:4326",
    )
    path = directory / shp_name
    gdf.to_file(path)
    return path


def _layers_at(lon: float, lat: float) -> dict[str, gpd.GeoDataFrame]:
    """Return a single-point layers dict already reprojected to EPSG:3857."""
    gdf = gpd.GeoDataFrame(
        {"NAME": ["Feature"]},
        geometry=[Point(lon, lat)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    return {"POI.shp": gdf}


# ---------------------------------------------------------------------------
# _load_gtfs_tables
# ---------------------------------------------------------------------------


def test_load_gtfs_tables_returns_all_five_keys(tmp_path: Path) -> None:
    """Happy path: all five required CSVs present → dict has every table name."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    tables = _load_gtfs_tables(gtfs_dir)
    assert set(tables) == {"routes", "trips", "stop_times", "stops", "shapes"}


def test_load_gtfs_tables_missing_file_raises(tmp_path: Path) -> None:
    """A missing GTFS file should raise FileNotFoundError."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    (gtfs_dir / "stops.txt").unlink()
    with pytest.raises(FileNotFoundError):
        _load_gtfs_tables(gtfs_dir)


# ---------------------------------------------------------------------------
# _prepare_route_buffers
# ---------------------------------------------------------------------------


def test_prepare_route_buffers_shape_mode_produces_non_empty_geometry() -> None:
    """Shape-buffer mode should produce one non-empty polygon per route."""
    result = _prepare_route_buffers(_minimal_tables(), use_shape_buffer=True, buffer_dist_ft=1320.0)
    assert len(result) == 1
    assert result.iloc[0]["route_id"] == "R1"
    assert not result.geometry.is_empty.any()


def test_prepare_route_buffers_stop_mode_produces_non_empty_geometry() -> None:
    """Stop-buffer mode should produce the same number of rows as shape mode."""
    result = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=False, buffer_dist_ft=1320.0
    )
    assert len(result) == 1
    assert not result.geometry.is_empty.any()


def test_prepare_route_buffers_stop_mode_buffers_only_each_routes_own_stops() -> None:
    """Stop mode must resolve each route to its own stops (single up-front merge)."""
    tables = _minimal_tables()
    # Add route R2 with its own trip T2 and a far-away stop S3 at (10, 10).
    tables["trips"] = pd.concat(
        [
            tables["trips"],
            pd.DataFrame({"route_id": ["R2"], "trip_id": ["T2"], "shape_id": ["SH2"]}),
        ],
        ignore_index=True,
    )
    tables["stop_times"] = pd.concat(
        [
            tables["stop_times"],
            pd.DataFrame({"trip_id": ["T2"], "stop_id": ["S3"], "stop_sequence": [1]}),
        ],
        ignore_index=True,
    )
    tables["stops"] = pd.concat(
        [
            tables["stops"],
            pd.DataFrame({"stop_id": ["S3"], "stop_lat": [10.0], "stop_lon": [10.0]}),
        ],
        ignore_index=True,
    )

    result = _prepare_route_buffers(tables, use_shape_buffer=False, buffer_dist_ft=1320.0)
    by_route = result.set_index("route_id").geometry
    assert set(by_route.index) == {"R1", "R2"}

    near = gpd.GeoSeries([Point(0.0, 0.0)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    far = gpd.GeoSeries([Point(10.0, 10.0)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    # R1 covers its stops near the origin but not R2's far stop, and vice versa.
    assert by_route.loc["R1"].contains(near)
    assert not by_route.loc["R1"].contains(far)
    assert by_route.loc["R2"].contains(far)
    assert not by_route.loc["R2"].contains(near)


def test_prepare_route_buffers_route_filter_excludes_unspecified_routes() -> None:
    """When route_filter is set, routes not in the list should be omitted."""
    tables = _minimal_tables()
    tables["trips"] = pd.concat(
        [
            tables["trips"],
            pd.DataFrame({"route_id": ["R2"], "trip_id": ["T2"], "shape_id": ["SH2"]}),
        ],
        ignore_index=True,
    )
    tables["shapes"] = pd.concat(
        [
            tables["shapes"],
            pd.DataFrame(
                {
                    "shape_id": ["SH2", "SH2"],
                    "shape_pt_lat": [1.0, 1.01],
                    "shape_pt_lon": [1.0, 1.01],
                    "shape_pt_sequence": [1, 2],
                }
            ),
        ],
        ignore_index=True,
    )
    result = _prepare_route_buffers(
        tables, use_shape_buffer=True, buffer_dist_ft=1320.0, route_filter=["R1"]
    )
    assert list(result["route_id"]) == ["R1"]


def test_prepare_route_buffers_simplify_reduces_buffer_vertices() -> None:
    """A positive tolerance collapses collinear points into a coarser buffer.

    Fewer vertices result, but the buffered area is essentially unchanged.
    """
    n = 100
    step = 0.05 / (n - 1)
    diag = [i * step for i in range(n)]  # 100 collinear points along y = x
    tables = _minimal_tables()
    tables["shapes"] = pd.DataFrame(
        {
            "shape_id": ["SH1"] * n,
            "shape_pt_lat": diag,
            "shape_pt_lon": diag,
            "shape_pt_sequence": list(range(1, n + 1)),
        }
    )
    fine = _prepare_route_buffers(
        tables, use_shape_buffer=True, buffer_dist_ft=1320.0, simplify_tolerance_m=0.0
    )
    coarse = _prepare_route_buffers(
        tables, use_shape_buffer=True, buffer_dist_ft=1320.0, simplify_tolerance_m=50.0
    )
    n_fine = len(fine.geometry.iloc[0].exterior.coords)
    n_coarse = len(coarse.geometry.iloc[0].exterior.coords)
    assert n_coarse < n_fine
    # Coverage is preserved: simplifying within 50 m barely changes a ~400 m buffer.
    assert coarse.geometry.iloc[0].area == pytest.approx(fine.geometry.iloc[0].area, rel=0.02)


def test_prepare_route_buffers_missing_shape_columns_raises() -> None:
    """Shapes table missing required columns should raise ValueError."""
    tables = _minimal_tables()
    tables["shapes"] = pd.DataFrame({"shape_id": ["SH1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        _prepare_route_buffers(tables, use_shape_buffer=True, buffer_dist_ft=1320.0)


# ---------------------------------------------------------------------------
# _load_layers
# ---------------------------------------------------------------------------


def test_load_layers_finds_shapefile_and_reprojects(tmp_path: Path) -> None:
    """A present shapefile with the correct id column should be loaded and reprojected."""
    _write_point_shp(tmp_path, "Station.shp", lon=0.001, lat=0.001)
    layers = _load_layers([("Station.shp", "NAME")], tmp_path)
    assert "Station.shp" in layers
    assert layers["Station.shp"].crs.to_epsg() == 3857


def test_load_layers_missing_file_excluded_from_result(tmp_path: Path) -> None:
    """A shapefile that does not exist should be silently excluded from the result."""
    layers = _load_layers([("Missing.shp", "NAME")], tmp_path)
    assert "Missing.shp" not in layers


def test_load_layers_wrong_id_column_excluded_from_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A shapefile that lacks the expected id column should be excluded from the result."""
    _write_point_shp(tmp_path, "Station.shp", lon=0.001, lat=0.001)
    with caplog.at_level(logging.WARNING):
        layers = _load_layers([("Station.shp", "WRONG_COL")], tmp_path)
    assert "Station.shp" not in layers
    # The warning should surface the actual attribute columns (here, NAME) so the
    # configured id_col can be corrected without inspecting the shapefile by hand.
    assert "NAME" in caplog.text
    assert "geometry" not in caplog.text


def test_load_layers_finds_shapefile_in_subdirectory(tmp_path: Path) -> None:
    """Recursive search should discover shapefiles nested in subdirectories."""
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    _write_point_shp(sub, "Station.shp", lon=0.001, lat=0.001)
    layers = _load_layers([("Station.shp", "NAME")], tmp_path)
    assert "Station.shp" in layers


def test_load_layers_respects_projected_crs_parameter(tmp_path: Path) -> None:
    """Passing a custom projected_crs should reproject each layer to that CRS."""
    _write_point_shp(tmp_path, "Station.shp", lon=0.001, lat=0.001)
    layers = _load_layers([("Station.shp", "NAME")], tmp_path, projected_crs="EPSG:32618")
    assert layers["Station.shp"].crs.to_epsg() == 32618


# ---------------------------------------------------------------------------
# _count_features
# ---------------------------------------------------------------------------


def test_count_features_counts_intersecting_point(tmp_path: Path) -> None:
    """A POI inside the route buffer should contribute a count of 1."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=True, buffer_dist_ft=1320.0
    )
    # Point at ~111 m north of origin — well within the 402 m (1 320 ft) buffer.
    layers = _layers_at(lon=0.0, lat=0.001)
    summary = _count_features(buffers, layers, [("POI.shp", "NAME")], tmp_path)
    assert summary.loc["R1", "POI.shp"] == 1


def test_count_features_non_intersecting_point_gives_zero(tmp_path: Path) -> None:
    """A POI outside the route buffer should produce a count of 0."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=True, buffer_dist_ft=1320.0
    )
    layers = _layers_at(lon=100.0, lat=100.0)
    summary = _count_features(buffers, layers, [("POI.shp", "NAME")], tmp_path)
    assert summary.loc["R1", "POI.shp"] == 0


def test_count_features_writes_per_route_csv(tmp_path: Path) -> None:
    """A per-route feature summary CSV should be written for each processed route."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=True, buffer_dist_ft=1320.0
    )
    _count_features(buffers, _layers_at(0.0, 0.001), [("POI.shp", "NAME")], tmp_path)
    assert (tmp_path / "R1_feature_summary.csv").exists()


def test_count_features_writes_per_route_png_when_enabled(tmp_path: Path) -> None:
    """With make_plots=True a per-route map PNG is written for each route."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=True, buffer_dist_ft=1320.0
    )
    _count_features(
        buffers, _layers_at(0.0, 0.001), [("POI.shp", "NAME")], tmp_path, make_plots=True
    )
    assert (tmp_path / "R1_buffer_plot.png").exists()


def test_count_features_skips_png_by_default(tmp_path: Path) -> None:
    """No PNG is rendered by default (the plot is opt-in to avoid headless hangs)."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=True, buffer_dist_ft=1320.0
    )
    _count_features(buffers, _layers_at(0.0, 0.001), [("POI.shp", "NAME")], tmp_path)
    assert not (tmp_path / "R1_buffer_plot.png").exists()
