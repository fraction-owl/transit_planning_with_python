from __future__ import annotations

import logging
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

matplotlib.use("Agg")  # headless backend; the module imports matplotlib.pyplot

from scripts.service_coverage.gtfs_service_demographics_gpd import (
    CRS_EPSG_CODE,
    METERS_PER_MILE,
    _present_synthetic_cols,
    _stops_to_points_gdf,
    apply_fips_filter,
    build_pedestrian_time_network,
    build_route_shapes_gdf,
    build_service_area_polygon,
    build_walk_isochrone,
    clip_and_calculate_synthetic_fields,
    export_summary_to_excel,
    filter_weekday_service,
    get_included_routes,
    get_included_stops,
    load_express_route_ids,
    load_gtfs_data,
    pick_buffer_distance,
    quantize_node,
    run,
)

FIXTURES = Path(__file__).parent / "fixtures"

# GTFS text files this script relies on (plus shapes.txt for route geometry).
_REQUIRED_GTFS = ["trips.txt", "stop_times.txt", "routes.txt", "stops.txt", "calendar.txt"]


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract *zip_path* into *dest* and return its single top-level folder."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    extracted = [p for p in dest.iterdir() if p.is_dir()]
    assert len(extracted) == 1, f"Expected one top-level dir, got: {extracted}"
    return extracted[0]


@pytest.fixture(scope="module")
def dc_gtfs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to the extracted mock DC GTFS feed (shared across the module)."""
    dest = tmp_path_factory.mktemp("mock_gtfs_dc")
    return _extract_zip(FIXTURES / "mock_gtfs_dc.zip", dest)


@pytest.fixture(scope="module")
def dc_gtfs(dc_gtfs_dir: Path) -> dict[str, pd.DataFrame]:
    """The DC feed's required tables loaded once (all columns as strings)."""
    return load_gtfs_data(str(dc_gtfs_dir), files=_REQUIRED_GTFS)


@pytest.fixture(scope="module")
def dc_shapes(dc_gtfs_dir: Path) -> pd.DataFrame:
    """The DC feed's shapes.txt table (route geometry)."""
    return pd.read_csv(dc_gtfs_dir / "shapes.txt", dtype=str, low_memory=False)


# ---------------------------------------------------------------------------
# filter_weekday_service
# ---------------------------------------------------------------------------


def _calendar(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_filter_weekday_service_keeps_full_week() -> None:
    cal = _calendar(
        [
            {
                "service_id": "WK",
                "monday": 1,
                "tuesday": 1,
                "wednesday": 1,
                "thursday": 1,
                "friday": 1,
                "saturday": 0,
                "sunday": 0,
            },
            {
                "service_id": "SAT",
                "monday": 0,
                "tuesday": 0,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 1,
                "sunday": 0,
            },
        ]
    )
    result = filter_weekday_service(cal)
    assert list(result) == ["WK"]


def test_filter_weekday_service_drops_partial_week() -> None:
    # Runs every weekday except Wednesday → should be excluded.
    cal = _calendar(
        [
            {
                "service_id": "PARTIAL",
                "monday": 1,
                "tuesday": 1,
                "wednesday": 0,
                "thursday": 1,
                "friday": 1,
                "saturday": 0,
                "sunday": 0,
            },
        ]
    )
    assert filter_weekday_service(cal).empty


def test_filter_weekday_service_handles_string_flags() -> None:
    # Real feeds load calendar.txt with every column as a string. Service "2" runs the
    # full Mon–Fri week; "1" skips Thursday; "3" is Saturday — only "2" should qualify.
    cal = _calendar(
        [
            {
                k: v
                for k, v in zip(
                    [
                        "service_id",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ],
                    ["1", "1", "1", "1", "0", "1", "0", "0"],
                )
            },
            {
                k: v
                for k, v in zip(
                    [
                        "service_id",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ],
                    ["2", "1", "1", "1", "1", "1", "0", "0"],
                )
            },
            {
                k: v
                for k, v in zip(
                    [
                        "service_id",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ],
                    ["3", "0", "0", "0", "0", "0", "1", "0"],
                )
            },
        ]
    )
    assert list(filter_weekday_service(cal)) == ["2"]


# ---------------------------------------------------------------------------
# _present_synthetic_cols
# ---------------------------------------------------------------------------


def test_present_synthetic_cols_filters_to_existing() -> None:
    clipped = gpd.GeoDataFrame(
        {"synthetic_total_pop": [1.0], "synthetic_minority": [2.0], "geometry": [Point(0, 0)]}
    )
    cols = _present_synthetic_cols(clipped, ["total_pop", "minority", "tot_empl", "youth"])
    assert cols == ["synthetic_total_pop", "synthetic_minority"]


def test_present_synthetic_cols_empty_when_none_present() -> None:
    clipped = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
    assert _present_synthetic_cols(clipped, ["total_pop", "minority"]) == []


# ---------------------------------------------------------------------------
# get_included_stops
# ---------------------------------------------------------------------------


def _stops_df() -> pd.DataFrame:
    return pd.DataFrame({"stop_id": ["S1", "S2", "S3"], "stop_name": ["a", "b", "c"]})


def test_get_included_stops_no_filters_keeps_all() -> None:
    result = get_included_stops(_stops_df(), [], [])
    assert list(result["stop_id"]) == ["S1", "S2", "S3"]


def test_get_included_stops_include_only() -> None:
    result = get_included_stops(_stops_df(), ["S1", "S3"], [])
    assert set(result["stop_id"]) == {"S1", "S3"}


def test_get_included_stops_exclude() -> None:
    result = get_included_stops(_stops_df(), [], ["S2"])
    assert set(result["stop_id"]) == {"S1", "S3"}


def test_get_included_stops_coerces_int_ids_to_str() -> None:
    # Filter values are ints but stop_id column is str; should still match.
    result = get_included_stops(_stops_df(), [1], [])
    assert list(result["stop_id"]) == []  # "1" != "S1"
    df = pd.DataFrame({"stop_id": [1, 2, 3]})
    result = get_included_stops(df, [1, 3], [])
    assert set(result["stop_id"]) == {"1", "3"}


# ---------------------------------------------------------------------------
# get_included_routes
# ---------------------------------------------------------------------------


def _routes_df() -> pd.DataFrame:
    return pd.DataFrame({"route_id": ["R1", "R2", "R3"], "route_short_name": ["101", "202", "303"]})


def test_get_included_routes_no_filters_keeps_all() -> None:
    result = get_included_routes(_routes_df(), [], [])
    assert len(result) == 3


def test_get_included_routes_include_and_exclude() -> None:
    result = get_included_routes(_routes_df(), ["101", "202"], ["202"])
    assert list(result["route_short_name"]) == ["101"]


def test_get_included_routes_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        get_included_routes(pd.DataFrame({"route_id": ["R1"]}), [], [])


# ---------------------------------------------------------------------------
# pick_buffer_distance
# ---------------------------------------------------------------------------


def test_pick_buffer_distance_normal() -> None:
    assert pick_buffer_distance("S1", 0.25, 2.0, []) == 0.25


def test_pick_buffer_distance_large() -> None:
    assert pick_buffer_distance("S1", 0.25, 2.0, ["S1"]) == 2.0


def test_pick_buffer_distance_coerces_types() -> None:
    # stop_id is int, large-buffer list holds strings.
    assert pick_buffer_distance(5, 0.25, 2.0, ["5"]) == 2.0


# ---------------------------------------------------------------------------
# quantize_node
# ---------------------------------------------------------------------------


def test_quantize_node_snaps_to_grid() -> None:
    assert quantize_node(12.3, 47.8, step=5.0) == (10.0, 50.0)


def test_quantize_node_exact_multiple_unchanged() -> None:
    assert quantize_node(100.0, 200.0, step=5.0) == (100.0, 200.0)


# ---------------------------------------------------------------------------
# build_pedestrian_time_network
# ---------------------------------------------------------------------------


def _centerlines(geoms: list[LineString], crs: str = "EPSG:3395") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def test_build_pedestrian_time_network_basic() -> None:
    # Two connected segments sharing the (100, 0) node.
    lines = [LineString([(0, 0), (100, 0)]), LineString([(100, 0), (100, 100)])]
    graph, edge_endpoints = build_pedestrian_time_network(
        _centerlines(lines), walk_speed=10.0, node_grid=5.0
    )
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert len(edge_endpoints) == 2
    # First edge is 100 units at 10 units/s → 10 s.
    times = sorted(d["time_s"] for _, _, d in graph.edges(data=True))
    assert times == pytest.approx([10.0, 10.0])


def test_build_pedestrian_time_network_skips_zero_and_degenerate() -> None:
    # A zero-length point-line and a loop that collapses after snapping.
    lines = [LineString([(0, 0), (1, 0)])]  # 1 unit, both endpoints snap to (0,0)
    graph, _ = build_pedestrian_time_network(_centerlines(lines), walk_speed=10.0, node_grid=5.0)
    assert graph.number_of_edges() == 0


def test_build_pedestrian_time_network_requires_crs() -> None:
    gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
    with pytest.raises(ValueError, match="no CRS"):
        build_pedestrian_time_network(gdf)


def test_build_pedestrian_time_network_requires_positive_speed() -> None:
    with pytest.raises(ValueError, match="walk_speed"):
        build_pedestrian_time_network(_centerlines([LineString([(0, 0), (1, 1)])]), walk_speed=0.0)


# ---------------------------------------------------------------------------
# build_route_shapes_gdf
# ---------------------------------------------------------------------------


def test_build_route_shapes_gdf_builds_one_line_per_route(
    dc_gtfs: dict[str, pd.DataFrame], dc_shapes: pd.DataFrame
) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], [], [])
    result = build_route_shapes_gdf(dc_shapes, dc_gtfs["trips"], final_routes, CRS_EPSG_CODE)
    assert set(result.columns) == {"route_short_name", "geometry"}
    # One dissolved (multi)line per route_short_name in the DC feed.
    assert set(result["route_short_name"]) == {"10", "20", "30", "40", "50H"}
    assert result.crs.to_epsg() == CRS_EPSG_CODE
    assert (result.geometry.length > 0).all()


def test_build_route_shapes_gdf_respects_route_filter(
    dc_gtfs: dict[str, pd.DataFrame], dc_shapes: pd.DataFrame
) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], ["10", "20"], [])
    result = build_route_shapes_gdf(dc_shapes, dc_gtfs["trips"], final_routes, CRS_EPSG_CODE)
    assert set(result["route_short_name"]) == {"10", "20"}


def test_build_route_shapes_gdf_none_shapes_returns_empty(dc_gtfs: dict[str, pd.DataFrame]) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], [], [])
    result = build_route_shapes_gdf(None, dc_gtfs["trips"], final_routes, CRS_EPSG_CODE)
    assert result.empty
    assert "route_short_name" in result.columns


def test_build_route_shapes_gdf_missing_columns_returns_empty(
    dc_gtfs: dict[str, pd.DataFrame],
) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], [], [])
    bad = pd.DataFrame({"shape_id": ["SH1"]})  # lacks lat/lon/sequence columns
    result = build_route_shapes_gdf(bad, dc_gtfs["trips"], final_routes, CRS_EPSG_CODE)
    assert result.empty


# ---------------------------------------------------------------------------
# build_walk_isochrone
# ---------------------------------------------------------------------------


def _ped_graph() -> nx.MultiGraph:
    lines = [
        LineString([(0, 0), (100, 0)]),
        LineString([(100, 0), (200, 0)]),
        LineString([(100, 0), (100, 100)]),
    ]
    graph, _ = build_pedestrian_time_network(_centerlines(lines), walk_speed=10.0, node_grid=5.0)
    return graph


def test_build_walk_isochrone_returns_polygon() -> None:
    stops = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3395")
    iso = build_walk_isochrone(stops, _ped_graph(), walk_time_min=1.0, walk_speed_units_per_s=10.0)
    assert iso is not None
    assert len(iso) == 1
    assert iso.geometry.iloc[0].area > 0
    assert iso.crs == stops.crs


def test_build_walk_isochrone_empty_graph_returns_none() -> None:
    stops = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3395")
    iso = build_walk_isochrone(
        stops, nx.MultiGraph(), walk_time_min=10.0, walk_speed_units_per_s=10.0
    )
    assert iso is None


# ---------------------------------------------------------------------------
# build_service_area_polygon
# ---------------------------------------------------------------------------


def _stop_points(stop_ids: list[str], coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"stop_id": stop_ids},
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:3395",
    )


def test_build_service_area_polygon_stop_buffer_area() -> None:
    pts = _stop_points(["S1"], [(0.0, 0.0)])
    result = build_service_area_polygon(
        pts,
        method="stop_buffer",
        buffer_distance_mi=0.25,
        large_buffer_distance_mi=2.0,
        stop_ids_large_buffer=[],
    )
    assert result is not None
    radius_m = 0.25 * METERS_PER_MILE
    expected = math.pi * radius_m**2
    # shapely's faceted buffer slightly underestimates the true circle area.
    assert result.geometry.iloc[0].area == pytest.approx(expected, rel=0.02)


def test_build_service_area_polygon_large_buffer_applied() -> None:
    pts = _stop_points(["S1"], [(0.0, 0.0)])
    normal = build_service_area_polygon(
        pts,
        method="stop_buffer",
        buffer_distance_mi=0.25,
        large_buffer_distance_mi=2.0,
        stop_ids_large_buffer=[],
    )
    large = build_service_area_polygon(
        pts,
        method="stop_buffer",
        buffer_distance_mi=0.25,
        large_buffer_distance_mi=2.0,
        stop_ids_large_buffer=["S1"],
    )
    ratio = large.geometry.iloc[0].area / normal.geometry.iloc[0].area
    assert ratio == pytest.approx((2.0 / 0.25) ** 2, rel=0.01)


def test_build_service_area_polygon_empty_stops_returns_none() -> None:
    empty = gpd.GeoDataFrame(
        {"stop_id": pd.Series(dtype=str)}, geometry=gpd.GeoSeries([], crs="EPSG:3395")
    )
    result = build_service_area_polygon(
        empty,
        method="stop_buffer",
        buffer_distance_mi=0.25,
        large_buffer_distance_mi=2.0,
        stop_ids_large_buffer=[],
    )
    assert result is None


def test_build_service_area_polygon_route_buffer() -> None:
    pts = _stop_points(["S1"], [(0.0, 0.0)])
    route_shapes = gpd.GeoDataFrame(
        {"route_short_name": ["101"]},
        geometry=[LineString([(0, 0), (1000, 0)])],
        crs="EPSG:3395",
    )
    result = build_service_area_polygon(
        pts,
        method="route_buffer",
        buffer_distance_mi=0.25,
        large_buffer_distance_mi=2.0,
        stop_ids_large_buffer=[],
        route_shapes_gdf=route_shapes,
    )
    assert result is not None
    # Area should exceed the buffered straight-line capsule's rectangle portion.
    radius_m = 0.25 * METERS_PER_MILE
    assert result.geometry.iloc[0].area > 1000 * 2 * radius_m * 0.9


def test_build_service_area_polygon_route_buffer_falls_back_to_stops() -> None:
    pts = _stop_points(["S1"], [(0.0, 0.0)])
    result = build_service_area_polygon(
        pts,
        method="route_buffer",
        buffer_distance_mi=0.25,
        large_buffer_distance_mi=2.0,
        stop_ids_large_buffer=[],
        route_shapes_gdf=None,
    )
    # No route geometry → fall back to a per-stop buffer.
    assert result is not None
    radius_m = 0.25 * METERS_PER_MILE
    assert result.geometry.iloc[0].area == pytest.approx(math.pi * radius_m**2, rel=0.02)


def test_build_service_area_polygon_isochrone_falls_back_without_graph() -> None:
    pts = _stop_points(["S1"], [(0.0, 0.0)])
    result = build_service_area_polygon(
        pts,
        method="isochrone",
        buffer_distance_mi=0.25,
        large_buffer_distance_mi=2.0,
        stop_ids_large_buffer=[],
        ped_graph=None,
    )
    assert result is not None  # falls back to stop buffers


# ---------------------------------------------------------------------------
# clip_and_calculate_synthetic_fields
# ---------------------------------------------------------------------------


def _demographics() -> gpd.GeoDataFrame:
    # A single 1000 x 1000 m square holding 100 people.
    square = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    return gpd.GeoDataFrame({"total_pop": [100]}, geometry=[square], crs="EPSG:3395")


def _half_buffer() -> gpd.GeoDataFrame:
    # Covers the left half of the demographics square.
    half = Polygon([(0, 0), (500, 0), (500, 1000), (0, 1000)])
    return gpd.GeoDataFrame(geometry=[half], crs="EPSG:3395")


def test_clip_and_calculate_synthetic_fields_half_overlap() -> None:
    result = clip_and_calculate_synthetic_fields(_demographics(), _half_buffer(), ["total_pop"])
    assert len(result) == 1
    row = result.iloc[0]
    assert row["area_perc"] == pytest.approx(0.5, rel=1e-6)
    assert row["synthetic_total_pop"] == pytest.approx(50.0, rel=1e-6)
    assert "area_ac_og" in result.columns
    assert "area_ac_cl" in result.columns


def test_clip_and_calculate_synthetic_fields_missing_field_skipped() -> None:
    result = clip_and_calculate_synthetic_fields(
        _demographics(), _half_buffer(), ["total_pop", "does_not_exist"]
    )
    assert "synthetic_total_pop" in result.columns
    assert "synthetic_does_not_exist" not in result.columns


def test_clip_and_calculate_synthetic_fields_preserves_existing_area_ac_og() -> None:
    demo = _demographics()
    demo["area_ac_og"] = 999.0  # pre-existing column should not be recomputed
    result = clip_and_calculate_synthetic_fields(demo, _half_buffer(), ["total_pop"])
    assert result["area_ac_og"].iloc[0] == 999.0


# ---------------------------------------------------------------------------
# export_summary_to_excel
# ---------------------------------------------------------------------------


def test_export_summary_to_excel_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "summary.xlsx"
    export_summary_to_excel({"synthetic_total_pop": 50, "synthetic_total_hh": 20}, str(out))
    assert out.is_file()
    df = pd.read_excel(out)
    assert len(df) == 1
    assert df["synthetic_total_pop"].iloc[0] == 50
    assert df["synthetic_total_hh"].iloc[0] == 20


# ---------------------------------------------------------------------------
# apply_fips_filter
# ---------------------------------------------------------------------------


def test_apply_fips_filter_empty_returns_unchanged() -> None:
    gdf = gpd.GeoDataFrame({"FIPS": ["11001", "51059"]}, geometry=[Point(0, 0), Point(1, 1)])
    result = apply_fips_filter(gdf, [])
    assert len(result) == 2


def test_apply_fips_filter_by_fips_column() -> None:
    gdf = gpd.GeoDataFrame({"FIPS": ["11001", "51059"]}, geometry=[Point(0, 0), Point(1, 1)])
    result = apply_fips_filter(gdf, ["11001"])
    assert list(result["FIPS"]) == ["11001"]


def test_apply_fips_filter_derives_from_geoid() -> None:
    gdf = gpd.GeoDataFrame(
        {"GEOID20": ["110010001001", "510590001001"]},
        geometry=[Point(0, 0), Point(1, 1)],
    )
    result = apply_fips_filter(gdf, ["11001"])
    assert len(result) == 1
    assert result["FIPS"].iloc[0] == "11001"


def test_apply_fips_filter_no_column_skips() -> None:
    gdf = gpd.GeoDataFrame({"other": ["a", "b"]}, geometry=[Point(0, 0), Point(1, 1)])
    result = apply_fips_filter(gdf, ["11001"])
    assert len(result) == 2  # filter skipped, nothing dropped


# ---------------------------------------------------------------------------
# load_gtfs_data
# ---------------------------------------------------------------------------


def test_load_gtfs_data_loads_required_files(dc_gtfs: dict[str, pd.DataFrame]) -> None:
    assert set(dc_gtfs) == {"trips", "stop_times", "routes", "stops", "calendar"}
    assert set(dc_gtfs["routes"]["route_short_name"]) == {"10", "20", "30", "40", "50H"}
    # dtype=str by default → IDs are read as strings rather than coerced numbers.
    assert all(isinstance(v, str) for v in dc_gtfs["stops"]["stop_id"])


def test_load_gtfs_data_missing_folder_raises() -> None:
    with pytest.raises(OSError, match="does not exist"):
        load_gtfs_data("/no/such/folder", files=["stops.txt"])


def test_load_gtfs_data_missing_file_raises(dc_gtfs_dir: Path) -> None:
    with pytest.raises(OSError, match="Missing GTFS files"):
        load_gtfs_data(str(dc_gtfs_dir), files=["nonexistent.txt"])


def test_run_raises_on_missing_demographics(tmp_path: Path, dc_gtfs_dir: Path) -> None:
    # run() used to catch every error and return, so a missing demographics
    # input exited 0 and looked identical to "produced nothing" under the
    # prep_features orchestrator. It must now surface the failure instead.
    with pytest.raises(FileNotFoundError, match="Demographics shapefile not found"):
        run(
            analysis_mode="route",
            service_area_method="stop_buffer",
            gtfs_data_path=str(dc_gtfs_dir),
            demographics_shp_path=str(tmp_path / "does_not_exist.shp"),
            output_directory=str(tmp_path / "out"),
        )


# ---------------------------------------------------------------------------
# _stops_to_points_gdf
# ---------------------------------------------------------------------------


def test_stops_to_points_gdf_builds_points(dc_gtfs: dict[str, pd.DataFrame]) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], [], [])
    result = _stops_to_points_gdf(
        dc_gtfs["trips"], dc_gtfs["stop_times"], dc_gtfs["stops"], final_routes, [], []
    )
    assert result is not None
    assert "stop_id" in result.columns
    assert "route_short_name" in result.columns
    assert result.crs.to_epsg() == CRS_EPSG_CODE
    assert (result.geometry.geom_type == "Point").all()


def test_stops_to_points_gdf_respects_route_filter(dc_gtfs: dict[str, pd.DataFrame]) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], ["10"], [])
    result = _stops_to_points_gdf(
        dc_gtfs["trips"], dc_gtfs["stop_times"], dc_gtfs["stops"], final_routes, [], []
    )
    assert result is not None
    assert set(result["route_short_name"]) == {"10"}


def test_stops_to_points_gdf_no_matching_stops_returns_none(
    dc_gtfs: dict[str, pd.DataFrame],
) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], [], [])
    result = _stops_to_points_gdf(
        dc_gtfs["trips"],
        dc_gtfs["stop_times"],
        dc_gtfs["stops"],
        final_routes,
        ["DOES_NOT_EXIST"],
        [],
    )
    assert result is None


# ---------------------------------------------------------------------------
# load_express_route_ids
# ---------------------------------------------------------------------------


def test_load_express_route_ids_inline_only() -> None:
    assert load_express_route_ids(["101", "303"], None) == {"101", "303"}


def test_load_express_route_ids_none_returns_empty() -> None:
    assert load_express_route_ids(None, None) == set()


def test_load_express_route_ids_trims_dedups_and_coerces() -> None:
    # Whitespace trimmed, duplicates collapsed, non-str ids coerced to str.
    assert load_express_route_ids([" 101 ", "101", 303], None) == {"101", "303"}


def test_load_express_route_ids_unions_inline_and_file(tmp_path: Path) -> None:
    f = tmp_path / "express.txt"
    f.write_text("101\n# whole-line comment\n202  # inline comment\n\n303\n", encoding="utf-8")
    assert load_express_route_ids(["404"], str(f)) == {"101", "202", "303", "404"}


def test_load_express_route_ids_missing_file_warns_and_keeps_inline(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = load_express_route_ids(["101"], str(tmp_path / "nope.txt"))
    assert result == {"101"}
    assert "not found" in caplog.text


def test_load_express_route_ids_reads_repo_fixture() -> None:
    # The shipped example fixture demonstrates the on-disk format.
    result = load_express_route_ids(None, str(FIXTURES / "express_routes.txt"))
    assert result == {"101", "303"}


# ---------------------------------------------------------------------------
# run() route mode — service_type labeling (Phase 1)
# ---------------------------------------------------------------------------


def _covering_demographics(stops_gdf: gpd.GeoDataFrame, tmp_path: Path) -> Path:
    """Write a demographics shapefile whose single square covers every stop."""
    minx, miny, maxx, maxy = stops_gdf.total_bounds
    pad = 5_000.0  # metres, comfortably larger than the 0.25-mile stop buffer
    demo = gpd.GeoDataFrame(
        {"total_pop": [1_000]},
        geometry=[box(minx - pad, miny - pad, maxx + pad, maxy + pad)],
        crs=f"EPSG:{CRS_EPSG_CODE}",
    )
    path = tmp_path / "demo.shp"
    demo.to_file(path)
    return path


def test_run_route_mode_labels_express_routes(
    tmp_path: Path, dc_gtfs_dir: Path, dc_gtfs: dict[str, pd.DataFrame]
) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], [], [])
    stops_gdf = _stops_to_points_gdf(
        dc_gtfs["trips"], dc_gtfs["stop_times"], dc_gtfs["stops"], final_routes, [], []
    )
    assert stops_gdf is not None
    demo_path = _covering_demographics(stops_gdf, tmp_path)

    express_id = str(dc_gtfs["routes"]["route_id"].iloc[0])
    out_dir = tmp_path / "out"
    run(
        analysis_mode="route",
        service_area_method="stop_buffer",
        gtfs_data_path=str(dc_gtfs_dir),
        demographics_shp_path=str(demo_path),
        output_directory=str(out_dir),
        express_route_ids=[express_id],
    )

    summary = pd.read_csv(out_dir / "service_demographics_by_route.csv")
    assert "service_type" in summary.columns
    assert set(summary["service_type"]) <= {"express", "local"}

    route_ids = summary["route_id"].astype(str)
    assert (summary.loc[route_ids == express_id, "service_type"] == "express").all()
    assert (summary.loc[route_ids != express_id, "service_type"] == "local").all()
    # Exactly the one named route is flagged express.
    assert (summary["service_type"] == "express").sum() == 1


def test_run_route_mode_all_local_without_express_list(
    tmp_path: Path, dc_gtfs_dir: Path, dc_gtfs: dict[str, pd.DataFrame]
) -> None:
    final_routes = get_included_routes(dc_gtfs["routes"], [], [])
    stops_gdf = _stops_to_points_gdf(
        dc_gtfs["trips"], dc_gtfs["stop_times"], dc_gtfs["stops"], final_routes, [], []
    )
    assert stops_gdf is not None
    demo_path = _covering_demographics(stops_gdf, tmp_path)

    out_dir = tmp_path / "out"
    run(
        analysis_mode="route",
        service_area_method="stop_buffer",
        gtfs_data_path=str(dc_gtfs_dir),
        demographics_shp_path=str(demo_path),
        output_directory=str(out_dir),
        express_route_ids=[],
    )

    summary = pd.read_csv(out_dir / "service_demographics_by_route.csv")
    assert (summary["service_type"] == "local").all()
