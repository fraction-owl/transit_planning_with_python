from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
from shapely.geometry import Point

matplotlib.use("Agg")

import scripts.service_coverage.points_of_interest_coverage_gpd as poi_mod
from scripts.service_coverage.points_of_interest_coverage_gpd import (
    LAYER_SPECS,
    _count_features,
    _load_gtfs_tables,
    _load_layers,
    _prepare_route_buffers,
    run,
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


def _write_point_shp_zip(
    directory: Path,
    zip_name: str,
    shp_name: str,
    lon: float,
    lat: float,
    id_col: str = "NAME",
) -> Path:
    """Write a one-feature point shapefile zipped into *directory*/*zip_name*.

    The shapefile components are stored at the archive's top level (matching what
    dev_tools/generate_mock_points_of_interest.py emits) and the path is returned.
    """
    gdf = gpd.GeoDataFrame({id_col: ["Feature"]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    zip_path = directory / zip_name
    stem = Path(shp_name).stem
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        gdf.to_file(tmp_dir / f"{stem}.shp")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for comp in sorted(tmp_dir.glob(f"{stem}.*")):
                zf.write(comp, comp.name)
    return zip_path


# ---------------------------------------------------------------------------
# _load_gtfs_tables
# ---------------------------------------------------------------------------


def test_load_gtfs_tables_returns_all_five_keys(tmp_path: Path) -> None:
    """Happy path: all five required CSVs present → dict has every table name."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    tables = _load_gtfs_tables(gtfs_dir, need_shapes=True)
    assert set(tables) == {"routes", "trips", "stop_times", "stops", "shapes"}


def test_load_gtfs_tables_missing_file_raises(tmp_path: Path) -> None:
    """A missing GTFS file should raise FileNotFoundError."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    (gtfs_dir / "stops.txt").unlink()
    with pytest.raises(FileNotFoundError):
        _load_gtfs_tables(gtfs_dir, need_shapes=True)


def test_load_gtfs_tables_stop_mode_omits_shapes(tmp_path: Path) -> None:
    """Stop-buffer mode loads four tables and does not require shapes.txt."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    (gtfs_dir / "shapes.txt").unlink()
    tables = _load_gtfs_tables(gtfs_dir, need_shapes=False)
    assert set(tables) == {"routes", "trips", "stop_times", "stops"}


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


def test_prepare_route_buffers_stop_mode_without_shape_id_column() -> None:
    """shape_id is optional in GTFS; stop-buffer mode must not require it."""
    tables = _minimal_tables()
    tables["trips"] = tables["trips"].drop(columns=["shape_id"])
    del tables["shapes"]
    result = _prepare_route_buffers(tables, use_shape_buffer=False, buffer_dist_ft=1320.0)
    assert len(result) == 1
    assert not result.geometry.is_empty.any()


def test_prepare_route_buffers_shape_mode_missing_shape_id_raises() -> None:
    """Shape-buffer mode without a trips.txt shape_id column fails clearly."""
    tables = _minimal_tables()
    tables["trips"] = tables["trips"].drop(columns=["shape_id"])
    with pytest.raises(ValueError, match="shape_id"):
        _prepare_route_buffers(tables, use_shape_buffer=True, buffer_dist_ft=1320.0)


# ---------------------------------------------------------------------------
# main (placeholder guard)
# ---------------------------------------------------------------------------


def test_main_blocks_unedited_placeholder_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CONFIG untouched and no flags, main() warns and does not run."""
    calls: list[dict] = []
    monkeypatch.setattr(poi_mod, "run", lambda **kw: calls.append(kw))
    poi_mod.main([])
    assert calls == []


def test_main_runs_after_config_edit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The documented edit-CONFIG-then-run workflow must reach run()."""
    calls: list[dict] = []
    monkeypatch.setattr(poi_mod, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(poi_mod, "GTFS_DIR", tmp_path / "gtfs")
    monkeypatch.setattr(poi_mod, "SHP_INPUT_DIR", tmp_path / "shp")
    poi_mod.main([])
    assert len(calls) == 1


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


def test_load_layers_reads_zipped_shapefile(tmp_path: Path) -> None:
    """A shapefile packaged inside a .zip should be discovered, loaded, reprojected."""
    _write_point_shp_zip(tmp_path, "Libraries.zip", "Libraries.shp", lon=0.001, lat=0.001)
    layers = _load_layers([("Libraries.shp", "NAME")], tmp_path)
    assert "Libraries.shp" in layers
    assert layers["Libraries.shp"].crs.to_epsg() == 3857
    assert len(layers["Libraries.shp"]) == 1


def test_load_layers_finds_zipped_shapefile_in_subdirectory(tmp_path: Path) -> None:
    """Recursive search should discover zipped shapefiles nested in subdirectories."""
    sub = tmp_path / "downloads"
    sub.mkdir()
    _write_point_shp_zip(sub, "Libraries.zip", "Libraries.shp", lon=0.001, lat=0.001)
    layers = _load_layers([("Libraries.shp", "NAME")], tmp_path)
    assert "Libraries.shp" in layers


def test_load_layers_zipped_wrong_id_column_excluded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A zipped shapefile lacking the expected id column should be excluded with a warning."""
    _write_point_shp_zip(tmp_path, "Libraries.zip", "Libraries.shp", lon=0.001, lat=0.001)
    with caplog.at_level(logging.WARNING):
        layers = _load_layers([("Libraries.shp", "WRONG_COL")], tmp_path)
    assert "Libraries.shp" not in layers
    # The warning should still surface the layer's actual attribute columns.
    assert "NAME" in caplog.text


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


# ---------------------------------------------------------------------------
# Integration tests against the committed POI fixtures
#
# tests/fixtures/points_of_interest/*.zip are produced by
# dev_tools/generate_mock_points_of_interest.py and mirror LAYER_SPECS exactly
# (one zipped point shapefile per layer). These tests exercise the real loader
# (including zip discovery), LAYER_SPECS, and the coverage pipeline end to end,
# locking in the generator-output <-> consumer contract.
# ---------------------------------------------------------------------------

POI_FIXTURE_DIR = Path("tests/fixtures/points_of_interest")

# Shared DC bounding box used by the mock generators (WGS84).
_DC_BBOX = (-77.120, 38.790, -76.910, 39.000)


def test_fixtures_present_for_every_layer_spec() -> None:
    """Every LAYER_SPECS layer ships a matching fixture zip in the fixtures dir."""
    available = {p.name for p in POI_FIXTURE_DIR.glob("*.zip")}
    expected = {f"{Path(fn).stem}.zip" for fn, _ in LAYER_SPECS}
    assert expected <= available


def test_fixture_layers_all_load_with_expected_schema() -> None:
    """All LAYER_SPECS layers load from the zipped fixtures with their id column."""
    layers = _load_layers(LAYER_SPECS, POI_FIXTURE_DIR)
    assert set(layers) == {fn for fn, _ in LAYER_SPECS}
    for filename, id_col in LAYER_SPECS:
        gdf = layers[filename]
        assert id_col in gdf.columns
        assert len(gdf) > 0
        assert (gdf.geometry.geom_type == "Point").all()
        assert gdf.crs.to_epsg() == 3857


def test_fixture_points_lie_within_the_dc_bbox() -> None:
    """Fixture points fall inside the shared DC bbox (sanity-checks the region)."""
    min_lon, min_lat, max_lon, max_lat = _DC_BBOX
    layers = _load_layers(LAYER_SPECS, POI_FIXTURE_DIR, projected_crs="EPSG:4326")
    for filename, _ in LAYER_SPECS:
        pts = layers[filename].geometry
        assert pts.x.between(min_lon, max_lon).all(), filename
        assert pts.y.between(min_lat, max_lat).all(), filename


def test_coverage_counts_a_fixture_poi_for_a_colocated_stop(tmp_path: Path) -> None:
    """A route with a stop on a fixture point counts that point within its buffer."""
    layers = _load_layers([("Metrorail_Stations.shp", "NAME")], POI_FIXTURE_DIR)
    seed = (
        gpd.GeoSeries([layers["Metrorail_Stations.shp"].geometry.iloc[0]], crs="EPSG:3857")
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    tables = {
        "routes": pd.DataFrame({"route_id": ["RX"]}),
        "trips": pd.DataFrame({"route_id": ["RX"], "trip_id": ["TX"], "shape_id": ["SX"]}),
        "stop_times": pd.DataFrame({"trip_id": ["TX"], "stop_id": ["SX1"], "stop_sequence": [1]}),
        "stops": pd.DataFrame({"stop_id": ["SX1"], "stop_lat": [seed.y], "stop_lon": [seed.x]}),
        "shapes": pd.DataFrame(
            {
                "shape_id": ["SX", "SX"],
                "shape_pt_lat": [seed.y, seed.y],
                "shape_pt_lon": [seed.x, seed.x],
                "shape_pt_sequence": [1, 2],
            }
        ),
    }
    buffers = _prepare_route_buffers(tables, use_shape_buffer=False, buffer_dist_ft=1320.0)
    summary = _count_features(buffers, layers, [("Metrorail_Stations.shp", "NAME")], tmp_path)
    assert summary.loc["RX", "Metrorail_Stations.shp"] >= 1


def test_run_end_to_end_against_fixtures(tmp_path: Path) -> None:
    """run() loads the zipped fixtures and writes a summary spanning every layer."""
    layers = _load_layers(LAYER_SPECS, POI_FIXTURE_DIR, projected_crs="EPSG:4326")
    seed = layers["Metrorail_Stations.shp"].geometry.iloc[0]

    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "routes.txt").write_text("route_id\nRA\n")
    (gtfs / "trips.txt").write_text("route_id,trip_id,shape_id\nRA,TA,SA\n")
    (gtfs / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\nTA,SA1,1\n")
    (gtfs / "stops.txt").write_text(f"stop_id,stop_lat,stop_lon\nSA1,{seed.y},{seed.x}\n")
    (gtfs / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        f"SA,{seed.y},{seed.x},1\nSA,{seed.y},{seed.x},2\n"
    )

    out = tmp_path / "out"
    run(gtfs_dir=gtfs, shp_input_dir=POI_FIXTURE_DIR, output_dir=out)

    summary = pd.read_csv(out / "all_routes_feature_summary.csv", index_col="route_id")
    assert set(summary.columns) == {fn for fn, _ in LAYER_SPECS}
    assert summary.loc["RA", "Metrorail_Stations.shp"] >= 1
