"""Tests for scripts/data_quality/stop_v_conflict_checker_gpd.py."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

script_dir = Path("scripts/data_quality").resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import stop_v_conflict_checker_gpd as target  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ANALYSIS_CRS = "EPSG:32618"
WGS84_CRS = "EPSG:4326"


def _make_stops_txt(tmp_path: Path, rows: list[dict]) -> Path:  # type: ignore[type-arg]
    df = pd.DataFrame(rows)
    p = tmp_path / "stops.txt"
    df.to_csv(p, index=False)
    return p


def _dc_stops_txt(tmp_path: Path) -> Path:
    with zipfile.ZipFile(FIXTURES / "mock_gtfs_dc.zip") as zf:
        zf.extractall(tmp_path)
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    return dirs[0] / "stops.txt"


def _dc_road_shp(tmp_path: Path) -> Path:
    road_dir = tmp_path / "roads"
    road_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(FIXTURES / "output_road_shps_dc.zip") as zf:
        zf.extractall(road_dir)
    return road_dir / "dc_road_centerlines.shp"


# ---------------------------------------------------------------------------
# _deg_tolerance_for_meters
# ---------------------------------------------------------------------------


def test_deg_tolerance_zero_tol_returns_zeros() -> None:
    lon, lat = target._deg_tolerance_for_meters(38.9, 0.0)
    assert lon == 0.0 and lat == 0.0


def test_deg_tolerance_returns_positive_values() -> None:
    lon, lat = target._deg_tolerance_for_meters(38.9, 1.0)
    assert lon > 0 and lat > 0


# ---------------------------------------------------------------------------
# _pandas_dedupe_stops
# ---------------------------------------------------------------------------


def test_pandas_dedupe_stops_removes_exact_duplicates(tmp_path: Path) -> None:
    txt = _make_stops_txt(
        tmp_path,
        [
            {
                "stop_id": "S1",
                "stop_code": "001",
                "stop_name": "Main",
                "stop_lat": 38.9,
                "stop_lon": -77.0,
            },
            {
                "stop_id": "S1",
                "stop_code": "001",
                "stop_name": "Main",
                "stop_lat": 38.9,
                "stop_lon": -77.0,
            },
        ],
    )
    df = target._pandas_dedupe_stops(
        str(txt), keys=["stop_id", "stop_code", "stop_name"], xy_tol_m=0.0
    )
    assert len(df) == 1


def test_pandas_dedupe_stops_keeps_distinct(tmp_path: Path) -> None:
    txt = _make_stops_txt(
        tmp_path,
        [
            {"stop_id": "S1", "stop_name": "Main", "stop_lat": 38.9, "stop_lon": -77.0},
            {"stop_id": "S2", "stop_name": "Oak", "stop_lat": 38.91, "stop_lon": -77.0},
        ],
    )
    df = target._pandas_dedupe_stops(str(txt), keys=["stop_id"], xy_tol_m=0.0)
    assert len(df) == 2


def test_pandas_dedupe_stops_raises_on_missing_lat_lon(tmp_path: Path) -> None:
    txt = tmp_path / "stops.txt"
    txt.write_text("stop_id,stop_name\nS1,Main\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stop_lat"):
        target._pandas_dedupe_stops(str(txt), keys=["stop_id"], xy_tol_m=0.0)


def test_pandas_dedupe_stops_xy_tolerance_merges_close(tmp_path: Path) -> None:
    # Two stops ~0.7 m apart (0.000005° offset) fall in the same 10 m grid bin → merged
    txt = _make_stops_txt(
        tmp_path,
        [
            {"stop_id": "S1", "stop_name": "Main", "stop_lat": 38.900000, "stop_lon": -77.000000},
            {"stop_id": "S1", "stop_name": "Main", "stop_lat": 38.900005, "stop_lon": -77.000005},
        ],
    )
    df = target._pandas_dedupe_stops(str(txt), keys=["stop_id", "stop_name"], xy_tol_m=10.0)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# _stops_to_gdf
# ---------------------------------------------------------------------------


def test_stops_to_gdf_returns_geodataframe() -> None:
    df = pd.DataFrame({"stop_id": ["S1"], "stop_lat": [38.9], "stop_lon": [-77.0]})
    gdf = target._stops_to_gdf(df, ANALYSIS_CRS)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_stops_to_gdf_projects_to_analysis_crs() -> None:
    df = pd.DataFrame({"stop_id": ["S1"], "stop_lat": [38.9], "stop_lon": [-77.0]})
    gdf = target._stops_to_gdf(df, ANALYSIS_CRS)
    assert gdf.crs.to_epsg() == 32618


def test_stops_to_gdf_geometry_is_point() -> None:
    df = pd.DataFrame(
        {"stop_id": ["S1", "S2"], "stop_lat": [38.9, 38.91], "stop_lon": [-77.0, -77.01]}
    )
    gdf = target._stops_to_gdf(df, ANALYSIS_CRS)
    assert all(isinstance(g, Point) for g in gdf.geometry)


# ---------------------------------------------------------------------------
# _load_context
# ---------------------------------------------------------------------------


def test_load_context_returns_none_for_empty_path() -> None:
    result = target._load_context("", ANALYSIS_CRS)
    assert result is None


def test_load_context_returns_none_for_missing_file() -> None:
    result = target._load_context("/nonexistent/path/roads.shp", ANALYSIS_CRS)
    assert result is None


def test_load_context_loads_real_shapefile(tmp_path: Path) -> None:
    shp = _dc_road_shp(tmp_path)
    gdf = target._load_context(str(shp), ANALYSIS_CRS)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0


def test_load_context_reprojects_to_analysis_crs(tmp_path: Path) -> None:
    shp = _dc_road_shp(tmp_path)
    gdf = target._load_context(str(shp), ANALYSIS_CRS)
    assert gdf is not None
    assert gdf.crs.to_epsg() == 32618


# ---------------------------------------------------------------------------
# _flag_intersections
# ---------------------------------------------------------------------------


def _make_stops_gdf(coords_wgs84: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    geom = [Point(lon, lat) for lat, lon in coords_wgs84]
    gdf = gpd.GeoDataFrame(
        {"stop_id": [f"S{i}" for i in range(len(geom))]},
        geometry=geom,
        crs=WGS84_CRS,
    ).to_crs(ANALYSIS_CRS)
    return gdf


def _make_polygon_layer(
    wgs84_coords: list[tuple[float, float]], buffer_deg: float = 0.001
) -> gpd.GeoDataFrame:
    polys = [Point(lon, lat).buffer(buffer_deg) for lat, lon in wgs84_coords]
    gdf = gpd.GeoDataFrame(geometry=polys, crs=WGS84_CRS).to_crs(ANALYSIS_CRS)
    return gdf


def test_flag_intersections_no_context_sets_zero() -> None:
    stops = _make_stops_gdf([(38.9, -77.0)])
    result = target._flag_intersections(stops.copy(), None, "in_roadway")
    assert result["in_roadway"].iloc[0] == 0


def test_flag_intersections_empty_context_sets_zero() -> None:
    stops = _make_stops_gdf([(38.9, -77.0)])
    empty_ctx = gpd.GeoDataFrame(geometry=[], crs=ANALYSIS_CRS)
    result = target._flag_intersections(stops.copy(), empty_ctx, "in_roadway")
    assert result["in_roadway"].iloc[0] == 0


def test_flag_intersections_overlapping_stop_flagged() -> None:
    # Stop placed inside a polygon layer
    stops = _make_stops_gdf([(38.9, -77.0)])
    ctx = _make_polygon_layer([(38.9, -77.0)], buffer_deg=0.01)
    result = target._flag_intersections(stops.copy(), ctx, "in_roadway")
    assert result["in_roadway"].iloc[0] == 1


def test_flag_intersections_non_overlapping_stop_not_flagged() -> None:
    stops = _make_stops_gdf([(38.9, -77.0)])
    # Context polygon far away
    ctx = _make_polygon_layer([(39.9, -78.0)], buffer_deg=0.001)
    result = target._flag_intersections(stops.copy(), ctx, "in_roadway")
    assert result["in_roadway"].iloc[0] == 0


# ---------------------------------------------------------------------------
# _add_conflict_summary
# ---------------------------------------------------------------------------


def test_add_conflict_summary_no_flags() -> None:
    gdf = _make_stops_gdf([(38.9, -77.0)])
    gdf["in_roadway"] = 0
    gdf["in_driveway"] = 0
    gdf["in_building"] = 0
    result = target._add_conflict_summary(gdf, ["in_roadway", "in_driveway", "in_building"])
    assert result["has_conflict"].iloc[0] == 0
    assert result["conflict_types"].iloc[0] == ""


def test_add_conflict_summary_single_flag() -> None:
    gdf = _make_stops_gdf([(38.9, -77.0)])
    gdf["in_roadway"] = 1
    gdf["in_driveway"] = 0
    gdf["in_building"] = 0
    result = target._add_conflict_summary(gdf, ["in_roadway", "in_driveway", "in_building"])
    assert result["has_conflict"].iloc[0] == 1
    assert "in_roadway" in result["conflict_types"].iloc[0]


def test_add_conflict_summary_multiple_flags() -> None:
    gdf = _make_stops_gdf([(38.9, -77.0)])
    gdf["in_roadway"] = 1
    gdf["in_driveway"] = 1
    gdf["in_building"] = 0
    result = target._add_conflict_summary(gdf, ["in_roadway", "in_driveway", "in_building"])
    assert result["has_conflict"].iloc[0] == 1
    assert "in_roadway" in result["conflict_types"].iloc[0]
    assert "in_driveway" in result["conflict_types"].iloc[0]


# ---------------------------------------------------------------------------
# _export_conflicts
# ---------------------------------------------------------------------------


def test_export_conflicts_csv(tmp_path: Path) -> None:
    gdf = _make_stops_gdf([(38.9, -77.0)])
    gdf["has_conflict"] = 1
    csv_path = str(tmp_path / "out.csv")
    target._export_conflicts(
        gdf,
        csv_path=csv_path,
        xlsx_path=None,
        shp_path=None,
        gpkg_path=None,
        layer_name=None,
        overwrite=True,
    )
    assert Path(csv_path).exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 1


def test_export_conflicts_gpkg(tmp_path: Path) -> None:
    gdf = _make_stops_gdf([(38.9, -77.0)])
    gdf["has_conflict"] = 1
    gpkg_path = str(tmp_path / "out.gpkg")
    target._export_conflicts(
        gdf,
        csv_path=None,
        xlsx_path=None,
        shp_path=None,
        gpkg_path=gpkg_path,
        layer_name="stops",
        overwrite=True,
    )
    assert Path(gpkg_path).exists()


def test_export_conflicts_overwrite_replaces_existing(tmp_path: Path) -> None:
    gdf = _make_stops_gdf([(38.9, -77.0)])
    gdf["has_conflict"] = 1
    csv_path = str(tmp_path / "out.csv")
    call_kwargs = {
        "csv_path": csv_path,
        "xlsx_path": None,
        "shp_path": None,
        "gpkg_path": None,
        "layer_name": None,
        "overwrite": True,
    }
    target._export_conflicts(gdf, **call_kwargs)
    target._export_conflicts(gdf, **call_kwargs)
    df = pd.read_csv(csv_path)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Integration: DC fixtures
# ---------------------------------------------------------------------------


def test_integration_dc_no_crash(tmp_path: Path) -> None:
    stops_txt = _dc_stops_txt(tmp_path / "gtfs")
    road_shp = _dc_road_shp(tmp_path)

    df = target._pandas_dedupe_stops(str(stops_txt), keys=["stop_id", "stop_name"], xy_tol_m=0.5)
    stops_gdf = target._stops_to_gdf(df, ANALYSIS_CRS)
    road_gdf = target._load_context(str(road_shp), ANALYSIS_CRS)

    work = stops_gdf.copy()
    work = target._flag_intersections(work, road_gdf, "in_roadway")
    work = target._flag_intersections(work, None, "in_driveway")
    work = target._flag_intersections(work, None, "in_building")
    work = target._add_conflict_summary(work, ["in_roadway", "in_driveway", "in_building"])

    assert "has_conflict" in work.columns
    assert "conflict_types" in work.columns
    assert len(work) > 0
