from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from scripts.gtfs_exports.gtfs_to_shapefile_gpd import (
    GTFS_CRS,
    PER_ROUTE_SUBDIR,
    build_export_basenames,
    export_gdf,
    export_lines_per_route,
    gtfs_to_shapefiles,
    map_shapes_to_routes,
    read_shapes,
    read_stops,
    sanitize_filename_component,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    extracted = [p for p in dest.iterdir() if p.is_dir()]
    assert len(extracted) == 1, f"Expected one top-level dir, got: {extracted}"
    return extracted[0]


@pytest.fixture()
def dc_gtfs_dir(tmp_path: Path) -> Path:
    return _extract_zip(FIXTURES / "mock_gtfs_dc.zip", tmp_path)


@pytest.fixture()
def ottawa_gtfs_dir(tmp_path: Path) -> Path:
    return _extract_zip(FIXTURES / "mock_gtfs_ottawa.zip", tmp_path)


# ---------------------------------------------------------------------------
# read_stops — happy path
# ---------------------------------------------------------------------------


def test_read_stops_dc_row_count(dc_gtfs_dir: Path) -> None:
    """DC fixture loads the expected number of stops."""
    gdf = read_stops(dc_gtfs_dir)
    assert len(gdf) == 395


def test_read_stops_ottawa_row_count(ottawa_gtfs_dir: Path) -> None:
    """Ottawa fixture loads the expected number of stops."""
    gdf = read_stops(ottawa_gtfs_dir)
    assert len(gdf) == 619


def test_read_stops_returns_geodataframe(dc_gtfs_dir: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_read_stops_crs_is_wgs84(dc_gtfs_dir: Path) -> None:
    """Stops GeoDataFrame uses EPSG:4326."""
    gdf = read_stops(dc_gtfs_dir)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_read_stops_required_columns_present(dc_gtfs_dir: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    for col in ("stop_id", "stop_name", "stop_lat", "stop_lon", "geometry"):
        assert col in gdf.columns, f"Missing column: {col}"


def test_read_stops_geometry_is_point(dc_gtfs_dir: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    assert all(isinstance(g, Point) for g in gdf.geometry)


def test_read_stops_dc_coordinates_in_range(dc_gtfs_dir: Path) -> None:
    """DC stops fall within the Washington DC bounding box."""
    gdf = read_stops(dc_gtfs_dir)
    assert gdf["stop_lat"].between(38.0, 40.0).all()
    assert gdf["stop_lon"].between(-78.0, -76.0).all()


def test_read_stops_ottawa_coordinates_in_range(ottawa_gtfs_dir: Path) -> None:
    """Ottawa stops fall within the Ottawa bounding box."""
    gdf = read_stops(ottawa_gtfs_dir)
    assert gdf["stop_lat"].between(44.0, 46.0).all()
    assert gdf["stop_lon"].between(-77.0, -74.0).all()


# ---------------------------------------------------------------------------
# read_stops — error cases
# ---------------------------------------------------------------------------


def test_read_stops_missing_stops_txt_raises(tmp_path: Path) -> None:
    """FileNotFoundError when stops.txt is absent from the GTFS directory."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    with pytest.raises(FileNotFoundError):
        read_stops(gtfs)


def test_read_stops_missing_columns_raises(tmp_path: Path) -> None:
    """ValueError when stops.txt lacks required columns."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "stops.txt").write_text("stop_id,stop_name\n001,Main\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required columns"):
        read_stops(gtfs)


# ---------------------------------------------------------------------------
# read_shapes — happy path
# ---------------------------------------------------------------------------


def test_read_shapes_dc_shape_count(dc_gtfs_dir: Path) -> None:
    """DC fixture produces 5 distinct LineString features."""
    gdf = read_shapes(dc_gtfs_dir)
    assert len(gdf) == 5


def test_read_shapes_ottawa_shape_count(ottawa_gtfs_dir: Path) -> None:
    """Ottawa fixture produces 5 distinct LineString features."""
    gdf = read_shapes(ottawa_gtfs_dir)
    assert len(gdf) == 5


def test_read_shapes_returns_geodataframe(dc_gtfs_dir: Path) -> None:
    gdf = read_shapes(dc_gtfs_dir)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_read_shapes_crs_is_wgs84(dc_gtfs_dir: Path) -> None:
    """Shapes GeoDataFrame uses EPSG:4326."""
    gdf = read_shapes(dc_gtfs_dir)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_read_shapes_geometry_is_linestring(dc_gtfs_dir: Path) -> None:
    gdf = read_shapes(dc_gtfs_dir)
    assert all(isinstance(g, LineString) for g in gdf.geometry)


def test_read_shapes_has_shape_id_column(dc_gtfs_dir: Path) -> None:
    gdf = read_shapes(dc_gtfs_dir)
    assert "shape_id" in gdf.columns


def test_read_shapes_all_lines_have_at_least_two_points(dc_gtfs_dir: Path) -> None:
    """Every LineString contains at least 2 coordinate pairs."""
    gdf = read_shapes(dc_gtfs_dir)
    assert all(len(g.coords) >= 2 for g in gdf.geometry)


# ---------------------------------------------------------------------------
# read_shapes — missing / invalid file
# ---------------------------------------------------------------------------


def test_read_shapes_missing_file_returns_empty_gdf(tmp_path: Path) -> None:
    """Missing shapes.txt returns an empty GeoDataFrame instead of raising."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    gdf = read_shapes(gtfs)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.empty


def test_read_shapes_missing_columns_raises(tmp_path: Path) -> None:
    """ValueError when shapes.txt exists but lacks required columns."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "shapes.txt").write_text("shape_id,shape_pt_lat\nS1,38.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required columns"):
        read_shapes(gtfs)


# ---------------------------------------------------------------------------
# export_gdf
# ---------------------------------------------------------------------------


def test_export_gdf_writes_shapefile(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    gdf = read_stops(dc_gtfs_dir)
    out = tmp_path / "output" / "stops.shp"
    export_gdf(gdf, out)
    assert out.exists()


def test_export_gdf_creates_output_directory(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """export_gdf creates any missing parent directories automatically."""
    gdf = read_stops(dc_gtfs_dir)
    out = tmp_path / "deeply" / "nested" / "stops.shp"
    export_gdf(gdf, out)
    assert out.exists()


def test_export_gdf_skips_empty_geodataframe(tmp_path: Path) -> None:
    """No file is written when the GeoDataFrame is empty."""
    empty = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs=GTFS_CRS)
    out = tmp_path / "output" / "empty.shp"
    export_gdf(empty, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# gtfs_to_shapefiles — full pipeline
# ---------------------------------------------------------------------------


def test_gtfs_to_shapefiles_both_dc(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='both' creates stops and lines shapefiles for the DC feed."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="both")
    assert (out / "gtfs_stops.shp").exists()
    assert (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_both_ottawa(ottawa_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='both' creates stops and lines shapefiles for the Ottawa feed."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(ottawa_gtfs_dir, out, kind="both")
    assert (out / "gtfs_stops.shp").exists()
    assert (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_kind_stops_only(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='stops' writes only the stops shapefile."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="stops")
    assert (out / "gtfs_stops.shp").exists()
    assert not (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_kind_lines_only(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """kind='lines' writes only the lines shapefile."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="lines")
    assert not (out / "gtfs_stops.shp").exists()
    assert (out / "gtfs_lines.shp").exists()


def test_gtfs_to_shapefiles_creates_output_dir(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """Output directory is created if it does not already exist."""
    out = tmp_path / "new_dir"
    assert not out.exists()
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="stops")
    assert out.is_dir()


# ---------------------------------------------------------------------------
# sanitize_filename_component / build_export_basenames
# ---------------------------------------------------------------------------


def test_sanitize_replaces_unsafe_characters() -> None:
    assert sanitize_filename_component("10/A B:C") == "10_A_B_C"


def test_sanitize_empty_falls_back_to_unnamed() -> None:
    assert sanitize_filename_component("///") == "unnamed"


def test_sanitize_truncates_to_max_len() -> None:
    assert sanitize_filename_component("x" * 100, max_len=10) == "x" * 10


def test_build_export_basenames_prefers_label_over_key() -> None:
    names = build_export_basenames([("route_1", "10")], prefix="route")
    assert names == {"route_1": "route_10"}


def test_build_export_basenames_falls_back_to_key() -> None:
    names = build_export_basenames([("route_1", None), ("route_2", "  ")], prefix="route")
    assert names == {"route_1": "route_route_1", "route_2": "route_route_2"}


def test_build_export_basenames_dedupes_case_insensitively() -> None:
    """Case-only collisions get numeric suffixes (safe on Windows filesystems)."""
    names = build_export_basenames([("a", "X1"), ("b", "x1"), ("c", "X1")], prefix="route")
    assert names["a"] == "route_X1"
    assert names["b"] == "route_x1_2"
    assert names["c"] == "route_X1_3"


def test_build_export_basenames_dedupes_after_sanitizing() -> None:
    """Distinct labels that sanitize to the same string still get unique names."""
    names = build_export_basenames([("a", "10/A"), ("b", "10 A")], prefix="route")
    assert len(set(names.values())) == 2


# ---------------------------------------------------------------------------
# map_shapes_to_routes
# ---------------------------------------------------------------------------


def test_map_shapes_to_routes_dc(dc_gtfs_dir: Path) -> None:
    """DC fixture maps each of the 5 shapes to its route with a short name."""
    mapping = map_shapes_to_routes(dc_gtfs_dir)
    assert mapping is not None
    assert set(mapping.columns) == {"shape_id", "route_id", "route_short"}
    assert len(mapping) == 5
    row = mapping[mapping["route_id"] == "DC_R10"].iloc[0]
    assert row["shape_id"] == "DC_R10_shp"
    assert row["route_short"] == "10"


def test_map_shapes_to_routes_missing_trips_returns_none(tmp_path: Path) -> None:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    assert map_shapes_to_routes(gtfs) is None


def test_map_shapes_to_routes_without_routes_txt(tmp_path: Path) -> None:
    """Mapping still works without routes.txt; route_short is empty."""
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "trips.txt").write_text(
        "route_id,service_id,trip_id,shape_id\nR1,wk,T1,S1\n", encoding="utf-8"
    )
    mapping = map_shapes_to_routes(gtfs)
    assert mapping is not None
    assert mapping.iloc[0]["route_id"] == "R1"
    assert pd.isna(mapping.iloc[0]["route_short"])


# ---------------------------------------------------------------------------
# export_lines_per_route
# ---------------------------------------------------------------------------


def _write_two_shape_gtfs(gtfs: Path) -> None:
    """Write a minimal shapes.txt with two 2-point shapes (S1, S2)."""
    gtfs.mkdir(parents=True, exist_ok=True)
    (gtfs / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        "S1,38.90,-77.03,1\nS1,38.91,-77.02,2\n"
        "S2,38.92,-77.01,1\nS2,38.93,-77.00,2\n",
        encoding="utf-8",
    )


def test_export_lines_per_route_dc(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """DC fixture yields one shapefile per route, named after route_short_name."""
    lines_gdf = read_shapes(dc_gtfs_dir)
    out = tmp_path / "out"
    export_lines_per_route(lines_gdf, dc_gtfs_dir, out)
    subdir = out / PER_ROUTE_SUBDIR
    for short in ("10", "20", "30", "40", "50H"):
        assert (subdir / f"route_{short}.shp").exists()
    assert len(list(subdir.glob("*.shp"))) == 5


def test_export_lines_per_route_file_contents(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """Each per-route shapefile carries route_id, rshort, and shape_id fields."""
    lines_gdf = read_shapes(dc_gtfs_dir)
    out = tmp_path / "out"
    export_lines_per_route(lines_gdf, dc_gtfs_dir, out)
    gdf = gpd.read_file(out / PER_ROUTE_SUBDIR / "route_10.shp")
    assert len(gdf) == 1
    assert gdf.iloc[0]["route_id"] == "DC_R10"
    assert gdf.iloc[0]["rshort"] == "10"
    assert gdf.iloc[0]["shape_id"] == "DC_R10_shp"
    assert isinstance(gdf.iloc[0].geometry, LineString)


def test_export_lines_per_route_fallback_per_shape(tmp_path: Path) -> None:
    """Without trips.txt, the split degrades to one shapefile per shape_id."""
    gtfs = tmp_path / "gtfs"
    _write_two_shape_gtfs(gtfs)
    lines_gdf = read_shapes(gtfs)
    out = tmp_path / "out"
    export_lines_per_route(lines_gdf, gtfs, out)
    subdir = out / PER_ROUTE_SUBDIR
    assert (subdir / "shape_S1.shp").exists()
    assert (subdir / "shape_S2.shp").exists()


def test_export_lines_per_route_unassigned_shapes(tmp_path: Path) -> None:
    """Shapes not referenced by any trip land in unassigned_shapes.shp."""
    gtfs = tmp_path / "gtfs"
    _write_two_shape_gtfs(gtfs)
    (gtfs / "trips.txt").write_text(
        "route_id,service_id,trip_id,shape_id\nR1,wk,T1,S1\n", encoding="utf-8"
    )
    lines_gdf = read_shapes(gtfs)
    out = tmp_path / "out"
    export_lines_per_route(lines_gdf, gtfs, out)
    subdir = out / PER_ROUTE_SUBDIR
    assert (subdir / "route_R1.shp").exists()
    assert (subdir / "unassigned_shapes.shp").exists()
    unassigned = gpd.read_file(subdir / "unassigned_shapes.shp")
    assert list(unassigned["shape_id"]) == ["S2"]


def test_export_lines_per_route_empty_gdf_writes_nothing(tmp_path: Path) -> None:
    empty = gpd.GeoDataFrame(columns=["shape_id", "geometry"], geometry=[], crs=GTFS_CRS)
    out = tmp_path / "out"
    export_lines_per_route(empty, tmp_path, out)
    assert not (out / PER_ROUTE_SUBDIR).exists()


# ---------------------------------------------------------------------------
# gtfs_to_shapefiles — per-route split option
# ---------------------------------------------------------------------------


def test_gtfs_to_shapefiles_split_by_route(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """split_by_route=True writes the combined file plus the per-route folder."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="both", split_by_route=True)
    assert (out / "gtfs_lines.shp").exists()
    assert len(list((out / PER_ROUTE_SUBDIR).glob("*.shp"))) == 5


def test_gtfs_to_shapefiles_split_off_by_default(dc_gtfs_dir: Path, tmp_path: Path) -> None:
    """Default behavior is unchanged: no per-route subfolder appears."""
    out = tmp_path / "out"
    gtfs_to_shapefiles(dc_gtfs_dir, out, kind="both")
    assert (out / "gtfs_lines.shp").exists()
    assert not (out / PER_ROUTE_SUBDIR).exists()


# ---------------------------------------------------------------------------
# gtfs_to_shapefiles — error cases
# ---------------------------------------------------------------------------


def test_gtfs_to_shapefiles_nonexistent_gtfs_dir_raises(tmp_path: Path) -> None:
    """NotADirectoryError when the GTFS directory does not exist."""
    with pytest.raises(NotADirectoryError):
        gtfs_to_shapefiles(tmp_path / "no_such_dir", tmp_path / "out")


def test_gtfs_to_shapefiles_none_gtfs_dir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ValueError when gtfs_dir is None and the module default is also None."""
    import scripts.gtfs_exports.gtfs_to_shapefile_gpd as mod

    monkeypatch.setattr(mod, "DEFAULT_GTFS_DIR", None)
    with pytest.raises(ValueError, match="GTFS input directory"):
        gtfs_to_shapefiles(None, tmp_path / "out")


def test_gtfs_to_shapefiles_none_output_dir_raises(
    dc_gtfs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ValueError when output_dir is None and the module default is also None."""
    import scripts.gtfs_exports.gtfs_to_shapefile_gpd as mod

    monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", None)
    with pytest.raises(ValueError, match="Output directory"):
        gtfs_to_shapefiles(dc_gtfs_dir, None)
