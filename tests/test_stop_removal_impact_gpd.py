from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

import scripts.stop_analysis.stop_removal_impact_gpd as sri
from scripts.stop_analysis.stop_removal_impact_gpd import (
    NODE_GRID_FT,
    build_graph,
    build_segment_index,
    coverage_polygon,
    explode_segments,
    linestring_length,
    linestring_substring,
    quantize_node,
    resolve_deleted_stop_ids,
    snap_stops_to_segments,
    stop_to_stop_network,
)

# ---------------------------------------------------------------------------
# quantize_node
# ---------------------------------------------------------------------------


def test_quantize_node_snaps_to_grid() -> None:
    # 12.3 / 5 = 2.46 → rounds to 2 → 2 * 5 = 10.0
    # 47.8 / 5 = 9.56 → rounds to 10 → 10 * 5 = 50.0
    result = quantize_node(12.3, 47.8, step_ft=5.0)
    assert result == (10.0, 50.0)


def test_quantize_node_exact_multiple_unchanged() -> None:
    result = quantize_node(100.0, 200.0, step_ft=5.0)
    assert result == (100.0, 200.0)


def test_quantize_node_zero_zero() -> None:
    result = quantize_node(0.0, 0.0, step_ft=5.0)
    assert result == (0.0, 0.0)


def test_quantize_node_returns_tuple_of_floats() -> None:
    x, y = quantize_node(7.3, 9.9, step_ft=5.0)
    assert isinstance(x, float)
    assert isinstance(y, float)


# ---------------------------------------------------------------------------
# linestring_length
# ---------------------------------------------------------------------------


def test_linestring_length_horizontal_segment() -> None:
    line = LineString([(0, 0), (100, 0)])
    assert linestring_length(line) == pytest.approx(100.0)


def test_linestring_length_pythagorean_triple() -> None:
    # 3-4-5 triangle
    line = LineString([(0, 0), (3, 4)])
    assert linestring_length(line) == pytest.approx(5.0)


def test_linestring_length_returns_float() -> None:
    line = LineString([(0, 0), (1, 0)])
    assert isinstance(linestring_length(line), float)


# ---------------------------------------------------------------------------
# linestring_substring
# ---------------------------------------------------------------------------


def test_linestring_substring_middle_portion() -> None:
    line = LineString([(0, 0), (100, 0)])
    sub = linestring_substring(line, 25.0, 75.0)
    assert sub.length == pytest.approx(50.0, abs=0.01)


def test_linestring_substring_full_line() -> None:
    line = LineString([(0, 0), (100, 0)])
    sub = linestring_substring(line, 0.0, 100.0)
    assert sub.length == pytest.approx(100.0, abs=0.01)


def test_linestring_substring_returns_linestring() -> None:
    line = LineString([(0, 0), (100, 0)])
    sub = linestring_substring(line, 10.0, 90.0)
    assert isinstance(sub, LineString)


def test_linestring_substring_clamps_to_line_bounds() -> None:
    line = LineString([(0, 0), (100, 0)])
    # start_m beyond line length should be clamped
    sub = linestring_substring(line, 0.0, 999.0)
    assert sub.length == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# resolve_deleted_stop_ids
# ---------------------------------------------------------------------------


@pytest.fixture()
def stops_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "stop_code": ["100", "200", "300"],
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)],
        },
        crs="EPSG:3857",
    )


def test_resolve_by_stop_code(stops_gdf: gpd.GeoDataFrame) -> None:
    resolved, match_map = resolve_deleted_stop_ids(stops_gdf, ["100"], prefer_stop_code=True)
    assert "S1" in resolved
    assert match_map["100"] == ["S1"]


def test_resolve_by_stop_id(stops_gdf: gpd.GeoDataFrame) -> None:
    resolved, match_map = resolve_deleted_stop_ids(stops_gdf, ["S2"], prefer_stop_code=False)
    assert "S2" in resolved


def test_resolve_unmatched_identifier_returns_empty(stops_gdf: gpd.GeoDataFrame) -> None:
    resolved, match_map = resolve_deleted_stop_ids(stops_gdf, ["UNKNOWN"])
    assert match_map["UNKNOWN"] == []
    assert "UNKNOWN" not in resolved


def test_resolve_multiple_identifiers(stops_gdf: gpd.GeoDataFrame) -> None:
    resolved, _ = resolve_deleted_stop_ids(stops_gdf, ["100", "200"], prefer_stop_code=True)
    assert "S1" in resolved
    assert "S2" in resolved


def test_resolve_no_stop_code_column_falls_back_to_stop_id() -> None:
    gdf = gpd.GeoDataFrame(
        {"stop_id": ["S1", "S2"], "geometry": [Point(0, 0), Point(1, 0)]},
        crs="EPSG:3857",
    )
    resolved, _ = resolve_deleted_stop_ids(gdf, ["S1"], prefer_stop_code=True)
    assert "S1" in resolved


# ---------------------------------------------------------------------------
# explode_segments
# ---------------------------------------------------------------------------


@pytest.fixture()
def centerlines_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
            ]
        },
        crs="EPSG:3857",
    )


def test_explode_segments_has_edge_id_column(centerlines_gdf: gpd.GeoDataFrame) -> None:
    result = explode_segments(centerlines_gdf)
    assert "edge_id" in result.columns


def test_explode_segments_row_count(centerlines_gdf: gpd.GeoDataFrame) -> None:
    result = explode_segments(centerlines_gdf)
    assert len(result) == 2


def test_explode_segments_all_linestrings(centerlines_gdf: gpd.GeoDataFrame) -> None:
    result = explode_segments(centerlines_gdf)
    assert (result.geom_type == "LineString").all()


def test_explode_segments_edge_ids_are_unique(centerlines_gdf: gpd.GeoDataFrame) -> None:
    result = explode_segments(centerlines_gdf)
    assert result["edge_id"].nunique() == len(result)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


@pytest.fixture()
def segments_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "edge_id": [0, 1],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
            ],
        },
        crs="EPSG:3857",
    )


def test_build_graph_returns_multigraph(segments_gdf: gpd.GeoDataFrame) -> None:
    G, _ = build_graph(segments_gdf, node_grid_ft=NODE_GRID_FT)
    assert isinstance(G, nx.MultiGraph)


def test_build_graph_edge_count(segments_gdf: gpd.GeoDataFrame) -> None:
    G, _ = build_graph(segments_gdf, node_grid_ft=NODE_GRID_FT)
    assert G.number_of_edges() == 2


def test_build_graph_node_count(segments_gdf: gpd.GeoDataFrame) -> None:
    # Three distinct endpoints: (0,0), (100,0), (200,0)
    G, _ = build_graph(segments_gdf, node_grid_ft=NODE_GRID_FT)
    assert G.number_of_nodes() == 3


def test_build_graph_edge_endpoints_dict(segments_gdf: gpd.GeoDataFrame) -> None:
    _, edge_endpoints = build_graph(segments_gdf, node_grid_ft=NODE_GRID_FT)
    assert 0 in edge_endpoints
    assert 1 in edge_endpoints


# ---------------------------------------------------------------------------
# coverage_polygon
# ---------------------------------------------------------------------------


def test_coverage_polygon_returns_geodataframe() -> None:
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1", "S2"], "geometry": [Point(0, 0), Point(500, 0)]},
        crs="EPSG:3857",
    )
    result = coverage_polygon(stops, buffer_miles=0.25)
    assert isinstance(result, gpd.GeoDataFrame)


def test_coverage_polygon_is_non_empty() -> None:
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1"], "geometry": [Point(0, 0)]},
        crs="EPSG:3857",
    )
    result = coverage_polygon(stops, buffer_miles=0.25)
    assert not result.empty
    assert result.area.sum() > 0


def test_coverage_polygon_larger_buffer_gives_larger_area() -> None:
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1"], "geometry": [Point(0, 0)]},
        crs="EPSG:3857",
    )
    small = coverage_polygon(stops, buffer_miles=0.1).area.sum()
    large = coverage_polygon(stops, buffer_miles=0.5).area.sum()
    assert large > small


# ---------------------------------------------------------------------------
# stop_to_stop_network
# ---------------------------------------------------------------------------


def _network_between(lines: list[LineString], stops: dict[str, Point]) -> tuple[float, LineString]:
    """Snap *stops* onto a network of *lines* and route stop A to stop B."""
    sri._node_dist_cache.clear()  # node keys repeat across tests' synthetic networks
    segments = explode_segments(gpd.GeoDataFrame(geometry=lines, crs="EPSG:6447"))
    G, edge_endpoints = build_graph(segments)
    seg_index, _, wkb_to_eid = build_segment_index(segments)
    stops_gdf = gpd.GeoDataFrame(
        {"stop_id": list(stops)}, geometry=list(stops.values()), crs="EPSG:6447"
    )
    snap_map = snap_stops_to_segments(stops_gdf, seg_index, wkb_to_eid, segments, edge_endpoints)
    dist_ft, geom = stop_to_stop_network(G, segments, edge_endpoints, snap_map, "A", "B")
    assert geom is not None
    return dist_ft, geom


def test_stop_to_stop_network_same_segment_walks_directly() -> None:
    # Stops 400 ft and 600 ft along one 1,000-ft segment are 200 ft apart, not
    # 1,000 ft via either endpoint.
    dist_ft, geom = _network_between(
        [LineString([(0, 0), (1000, 0)])], {"A": Point(400, 10), "B": Point(600, 10)}
    )
    assert dist_ft == pytest.approx(200.0)
    assert geom.length == pytest.approx(200.0)
    assert geom.coords[0] == pytest.approx((400.0, 0.0))
    assert geom.coords[-1] == pytest.approx((600.0, 0.0))


def test_stop_to_stop_network_same_segment_keeps_shorter_network_route() -> None:
    # A 1,100-ft U-shaped segment whose ends are also joined by a 100-ft segment:
    # stops 10 ft from either end walk 10 + 100 + 10 ft, not 1,080 ft along the U.
    u_shape = LineString([(0, 0), (0, 500), (100, 500), (100, 0)])
    shortcut = LineString([(0, 0), (100, 0)])
    dist_ft, geom = _network_between([u_shape, shortcut], {"A": Point(-5, 10), "B": Point(105, 10)})
    assert dist_ft == pytest.approx(120.0)
    assert geom.bounds[3] < 500  # the path takes the shortcut, not the U
