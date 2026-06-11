from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from scripts.service_coverage.school_coverage_by_route_gpd import (
    _load_gtfs_tables,
    _prepare_route_buffers,
    load_schools_layer,
    run,
    summarize_schools_by_route,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_gtfs_files(gtfs_dir: Path) -> None:
    """Write a minimal single-route GTFS feed (R1 / T1 / SH1) into *gtfs_dir*."""
    (gtfs_dir / "routes.txt").write_text("route_id,route_short_name\nR1,1\n")
    (gtfs_dir / "trips.txt").write_text("route_id,trip_id,shape_id\nR1,T1,SH1\n")
    (gtfs_dir / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\n")
    (gtfs_dir / "stops.txt").write_text("stop_id,stop_lat,stop_lon\nS1,0.0,0.0\nS2,0.01,0.01\n")
    (gtfs_dir / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\nSH1,0.0,0.0,1\nSH1,0.01,0.01,2\n"
    )


def _minimal_tables() -> dict[str, pd.DataFrame]:
    """Return in-memory GTFS DataFrames for a single route R1 (shape SH1)."""
    return {
        "routes": pd.DataFrame({"route_id": ["R1"], "route_short_name": ["1"]}),
        "trips": pd.DataFrame({"route_id": ["R1"], "trip_id": ["T1"], "shape_id": ["SH1"]}),
        "stop_times": pd.DataFrame(
            {"trip_id": ["T1", "T1"], "stop_id": ["S1", "S2"], "stop_sequence": [1, 2]}
        ),
        "stops": pd.DataFrame(
            {"stop_id": ["S1", "S2"], "stop_lat": [0.0, 0.01], "stop_lon": [0.0, 0.01]}
        ),
    }


def _schools_layer(*points: tuple[float, float, float]) -> gpd.GeoDataFrame:
    """Build an EPSG:3857 schools layer from (lon, lat, enroll_total) triples."""
    gdf = gpd.GeoDataFrame(
        {"enroll_total": [p[2] for p in points]},
        geometry=[Point(p[0], p[1]) for p in points],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    return gdf


def _write_schools_gpkg(directory: Path, name: str, *points: tuple[float, float, float]) -> Path:
    """Write a schools point GeoPackage to *directory*/*name* and return its path."""
    gdf = gpd.GeoDataFrame(
        {"enroll_total": [p[2] for p in points]},
        geometry=[Point(p[0], p[1]) for p in points],
        crs="EPSG:4326",
    )
    path = directory / name
    gdf.to_file(path, driver="GPKG")
    return path


# ---------------------------------------------------------------------------
# _load_gtfs_tables
# ---------------------------------------------------------------------------


def test_load_gtfs_tables_stop_mode_omits_shapes(tmp_path: Path) -> None:
    """Stop-buffer mode loads four tables and does not require shapes.txt."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    (gtfs_dir / "shapes.txt").unlink()
    tables = _load_gtfs_tables(gtfs_dir, need_shapes=False)
    assert set(tables) == {"routes", "trips", "stop_times", "stops"}


def test_load_gtfs_tables_shape_mode_requires_shapes(tmp_path: Path) -> None:
    """Shape-buffer mode requires shapes.txt and raises when it is absent."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    (gtfs_dir / "shapes.txt").unlink()
    with pytest.raises(FileNotFoundError):
        _load_gtfs_tables(gtfs_dir, need_shapes=True)


# ---------------------------------------------------------------------------
# _prepare_route_buffers
# ---------------------------------------------------------------------------


def test_prepare_route_buffers_stop_mode_produces_one_polygon() -> None:
    """Stop-buffer mode yields one non-empty catchment for the single route."""
    result = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=False, buffer_dist_ft=1320.0
    )
    assert list(result["route_id"]) == ["R1"]
    assert not result.geometry.is_empty.any()


# ---------------------------------------------------------------------------
# load_schools_layer
# ---------------------------------------------------------------------------


def test_load_schools_layer_reads_single_file(tmp_path: Path) -> None:
    """A single GeoPackage is loaded and reprojected to the projected CRS."""
    path = _write_schools_gpkg(
        tmp_path, "va_md_dc_public_schools_enrollment.gpkg", (0.0, 0.001, 500)
    )
    gdf = load_schools_layer(path)
    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == 3857


def test_load_schools_layer_combines_folder(tmp_path: Path) -> None:
    """A folder of *schools_enrollment* layers is combined into one GeoDataFrame."""
    _write_schools_gpkg(tmp_path, "va_md_dc_public_schools_enrollment.gpkg", (0.0, 0.001, 500))
    _write_schools_gpkg(tmp_path, "va_md_dc_private_schools_enrollment.gpkg", (0.0, 0.002, 250))
    gdf = load_schools_layer(tmp_path)
    assert len(gdf) == 2


def test_load_schools_layer_missing_dir_raises(tmp_path: Path) -> None:
    """A folder with no matching layers raises FileNotFoundError."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_schools_layer(empty)


# ---------------------------------------------------------------------------
# summarize_schools_by_route
# ---------------------------------------------------------------------------


def test_summarize_counts_and_sums_enrollment() -> None:
    """Schools inside the catchment contribute to both count and enrollment."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=False, buffer_dist_ft=1320.0
    )
    # Two schools ~111 m / ~222 m north of origin — both within the 402 m buffer.
    schools = _schools_layer((0.0, 0.001, 500), (0.0, 0.002, 250))
    summary = summarize_schools_by_route(buffers, schools)
    row = summary.set_index("route_id").loc["R1"]
    assert row["schools_served"] == 2
    assert row["enrollment_served"] == 750


def test_summarize_far_school_gives_zero() -> None:
    """A school far outside the catchment yields zero served counts."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=False, buffer_dist_ft=1320.0
    )
    schools = _schools_layer((100.0, 100.0, 999))
    summary = summarize_schools_by_route(buffers, schools)
    row = summary.set_index("route_id").loc["R1"]
    assert row["schools_served"] == 0
    assert row["enrollment_served"] == 0


def test_summarize_nan_enrollment_counted_as_zero_enrollment() -> None:
    """An unmatched (NaN) enrollment still counts as a school but adds 0 enrollment."""
    buffers = _prepare_route_buffers(
        _minimal_tables(), use_shape_buffer=False, buffer_dist_ft=1320.0
    )
    schools = _schools_layer((0.0, 0.001, float("nan")))
    summary = summarize_schools_by_route(buffers, schools)
    row = summary.set_index("route_id").loc["R1"]
    assert row["schools_served"] == 1
    assert row["enrollment_served"] == 0


# ---------------------------------------------------------------------------
# run  (integration)
# ---------------------------------------------------------------------------


def test_run_writes_route_keyed_csv(tmp_path: Path) -> None:
    """run() writes school_coverage_by_route.csv with the expected columns."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    schools_dir = tmp_path / "schools"
    schools_dir.mkdir()
    _write_schools_gpkg(
        schools_dir, "va_md_dc_public_schools_enrollment.gpkg", (0.0, 0.001, 500)
    )
    out_dir = tmp_path / "out"

    summary = run(gtfs_dir=gtfs_dir, schools_path=schools_dir, output_dir=out_dir)

    out_csv = out_dir / "school_coverage_by_route.csv"
    assert out_csv.exists()
    written = pd.read_csv(out_csv)
    assert {"route_id", "schools_served", "enrollment_served"} <= set(written.columns)
    assert written.set_index("route_id").loc["R1", "enrollment_served"] == 500
    assert "route_short_name" in summary.columns
