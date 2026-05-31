from __future__ import annotations

import logging
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

from utils.network_helpers import (
    DEFAULT_NODE_GRID_FT,
    DEFAULT_WALK_SPEED_FT_PER_S,
    _parse_gtfs_time,
    build_gtfs_transit_time_network,
    build_pedestrian_time_network,
    quantize_node,
)

FIXTURES = Path(__file__).parent / "fixtures"

# A projected CRS in US feet (matches the module's foot-based default walk speed);
# used when the geometry values themselves don't need to be meaningful.
PROJECTED_FT_CRS = "EPSG:6447"
# UTM 18N (metres) covers Washington, DC; used to reproject the geographic
# centerlines fixture into a metric CRS before measuring lengths.
DC_UTM_CRS = "EPSG:32618"


# ---------------------------------------------------------------------------
# quantize_node
# ---------------------------------------------------------------------------


def test_quantize_node_snaps_to_grid() -> None:
    """Coordinates round to the nearest multiple of the grid step."""
    assert quantize_node(12.0, 7.0, step=5.0) == (10.0, 5.0)


def test_quantize_node_merges_near_coincident_points() -> None:
    """Two slightly-different points within a grid cell collapse to one key."""
    assert quantize_node(10.0, 0.0, step=5.0) == quantize_node(11.4, 1.2, step=5.0)


def test_quantize_node_default_step() -> None:
    """Omitting step uses the module default grid."""
    assert quantize_node(3.0, 3.0) == quantize_node(3.0, 3.0, step=DEFAULT_NODE_GRID_FT)


def test_quantize_node_returns_float_tuple() -> None:
    """Accepts numeric-like input and always returns a float (x, y) tuple."""
    key = quantize_node("10", "20", step=5.0)  # type: ignore[arg-type]
    assert key == (10.0, 20.0)
    assert all(isinstance(c, float) for c in key)


# ---------------------------------------------------------------------------
# _parse_gtfs_time
# ---------------------------------------------------------------------------


def test_parse_gtfs_time_normal() -> None:
    assert _parse_gtfs_time("05:01:09") == 5 * 3600 + 1 * 60 + 9


def test_parse_gtfs_time_past_midnight_no_wrap() -> None:
    """Hours >= 24 are kept (GTFS allows them); they must not wrap at 24:00."""
    assert _parse_gtfs_time("25:14:00") == 25 * 3600 + 14 * 60


def test_parse_gtfs_time_none_is_nan() -> None:
    assert math.isnan(_parse_gtfs_time(None))


def test_parse_gtfs_time_blank_is_nan() -> None:
    assert math.isnan(_parse_gtfs_time("   "))


def test_parse_gtfs_time_nan_float_is_nan() -> None:
    assert math.isnan(_parse_gtfs_time(float("nan")))


def test_parse_gtfs_time_wrong_field_count_is_nan() -> None:
    assert math.isnan(_parse_gtfs_time("05:01"))


def test_parse_gtfs_time_non_numeric_is_nan() -> None:
    assert math.isnan(_parse_gtfs_time("aa:bb:cc"))


# ---------------------------------------------------------------------------
# build_pedestrian_time_network
# ---------------------------------------------------------------------------


def _centerlines(geoms: list, crs: str | None = PROJECTED_FT_CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs=crs)


def test_pedestrian_happy_path_nodes_edges_and_time() -> None:
    """Two end-to-end segments share a node; time_s = length / walk_speed."""
    gdf = _centerlines(
        [
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
        ]
    )
    graph, edge_endpoints = build_pedestrian_time_network(gdf, walk_speed=10.0, node_grid=5.0)

    assert isinstance(graph, nx.MultiGraph)
    assert graph.number_of_edges() == 2
    # (0,0), (10,0), (20,0) — the shared (10,0) endpoint collapses to one node.
    assert graph.number_of_nodes() == 3
    assert len(edge_endpoints) == 2

    for _u, _v, data in graph.edges(data=True):
        assert data["length"] == pytest.approx(10.0)
        assert data["time_s"] == pytest.approx(1.0)  # 10 units / 10 units-per-s

    # Endpoint mapping is consistent with the graph's actual nodes.
    for u, v in edge_endpoints.values():
        assert graph.has_edge(u, v)


def test_pedestrian_node_attributes_present() -> None:
    """Each node carries x/y attributes equal to its quantized key."""
    gdf = _centerlines([LineString([(0, 0), (10, 0)])])
    graph, _ = build_pedestrian_time_network(gdf, walk_speed=10.0, node_grid=5.0)
    for node, data in graph.nodes(data=True):
        assert (data["x"], data["y"]) == node


def test_pedestrian_near_coincident_endpoints_merge() -> None:
    """Endpoints within one grid cell snap together into a single shared node."""
    gdf = _centerlines(
        [
            LineString([(0, 0), (10, 0)]),
            LineString([(11, 1), (20, 0)]),  # start ~(10,0) after snapping to grid 5
        ]
    )
    graph, _ = build_pedestrian_time_network(gdf, walk_speed=10.0, node_grid=5.0)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_pedestrian_zero_length_segment_skipped() -> None:
    """A zero-length segment contributes no edge."""
    gdf = _centerlines([LineString([(0, 0), (0, 0)])])
    graph, edge_endpoints = build_pedestrian_time_network(gdf, walk_speed=10.0)
    assert graph.number_of_edges() == 0
    assert edge_endpoints == {}


def test_pedestrian_degenerate_loop_after_snapping_skipped() -> None:
    """A short segment whose endpoints snap to the same node is dropped."""
    # (0,0)->(2,0) both quantize to (0,0) at grid 5 -> u == v -> skipped.
    gdf = _centerlines([LineString([(0, 0), (2, 0)])])
    graph, _ = build_pedestrian_time_network(gdf, walk_speed=10.0, node_grid=5.0)
    assert graph.number_of_edges() == 0


def test_pedestrian_multilinestring_exploded() -> None:
    """MultiLineString inputs are exploded into one edge per simple part."""
    multi = MultiLineString([[(0, 0), (10, 0)], [(0, 10), (10, 10)]])
    gdf = _centerlines([multi])
    graph, _ = build_pedestrian_time_network(gdf, walk_speed=10.0, node_grid=5.0)
    assert graph.number_of_edges() == 2


def test_pedestrian_no_crs_raises() -> None:
    gdf = _centerlines([LineString([(0, 0), (10, 0)])], crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        build_pedestrian_time_network(gdf)


@pytest.mark.parametrize("bad_speed", [0.0, -1.0])
def test_pedestrian_non_positive_speed_raises(bad_speed: float) -> None:
    gdf = _centerlines([LineString([(0, 0), (10, 0)])])
    with pytest.raises(ValueError, match="walk_speed must be positive"):
        build_pedestrian_time_network(gdf, walk_speed=bad_speed)


def test_pedestrian_geographic_crs_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A geographic CRS still builds, but warns that lengths are meaningless."""
    gdf = _centerlines([LineString([(0, 0), (0.01, 0)])], crs="EPSG:4326")
    with caplog.at_level(logging.WARNING):
        build_pedestrian_time_network(gdf, walk_speed=10.0)
    assert any("geographic" in rec.message for rec in caplog.records)


def test_pedestrian_default_walk_speed_used() -> None:
    """Omitting walk_speed applies the foot-per-second default."""
    gdf = _centerlines([LineString([(0, 0), (DEFAULT_WALK_SPEED_FT_PER_S, 0)])])
    graph, _ = build_pedestrian_time_network(gdf)
    times = [d["time_s"] for *_e, d in graph.edges(data=True)]
    assert times == [pytest.approx(1.0)]


# ---------------------------------------------------------------------------
# build_gtfs_transit_time_network
# ---------------------------------------------------------------------------


def _stop_times(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "trip_id",
            "stop_id",
            "stop_sequence",
            "arrival_time",
            "departure_time",
        ],
    )


def test_transit_happy_path_edges_and_times() -> None:
    """Consecutive stops become directed edges weighted by scheduled ride time."""
    st = _stop_times(
        [
            {
                "trip_id": "T1",
                "stop_id": "A",
                "stop_sequence": 1,
                "arrival_time": "05:00:00",
                "departure_time": "05:00:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "B",
                "stop_sequence": 2,
                "arrival_time": "05:01:00",
                "departure_time": "05:01:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "C",
                "stop_sequence": 3,
                "arrival_time": "05:03:00",
                "departure_time": "05:03:00",
            },
        ]
    )
    graph = build_gtfs_transit_time_network(st)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph["A"]["B"][0]["time_s"] == pytest.approx(60.0)
    assert graph["B"]["C"][0]["time_s"] == pytest.approx(120.0)
    assert graph["A"]["B"][0]["trip_id"] == "T1"
    assert "route_id" not in graph["A"]["B"][0]


def test_transit_orders_by_stop_sequence_not_row_order() -> None:
    """Edges follow stop_sequence even when rows are shuffled."""
    st = _stop_times(
        [
            {
                "trip_id": "T1",
                "stop_id": "C",
                "stop_sequence": 3,
                "arrival_time": "05:03:00",
                "departure_time": "05:03:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "A",
                "stop_sequence": 1,
                "arrival_time": "05:00:00",
                "departure_time": "05:00:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "B",
                "stop_sequence": 2,
                "arrival_time": "05:01:00",
                "departure_time": "05:01:00",
            },
        ]
    )
    graph = build_gtfs_transit_time_network(st)
    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")
    assert not graph.has_edge("C", "A")


def test_transit_route_id_tagged_from_trips() -> None:
    """When a trips table is supplied, edges carry the trip's route_id."""
    st = _stop_times(
        [
            {
                "trip_id": "T1",
                "stop_id": "A",
                "stop_sequence": 1,
                "arrival_time": "05:00:00",
                "departure_time": "05:00:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "B",
                "stop_sequence": 2,
                "arrival_time": "05:02:00",
                "departure_time": "05:02:00",
            },
        ]
    )
    trips = pd.DataFrame({"trip_id": ["T1"], "route_id": ["R10"]})
    graph = build_gtfs_transit_time_network(st, trips=trips)
    assert graph["A"]["B"][0]["route_id"] == "R10"


def test_transit_skips_negative_and_missing_times() -> None:
    """Segments with negative or unparseable ride times are dropped."""
    st = _stop_times(
        [
            # Negative: arrival at B precedes departure from A.
            {
                "trip_id": "T1",
                "stop_id": "A",
                "stop_sequence": 1,
                "arrival_time": "05:05:00",
                "departure_time": "05:05:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "B",
                "stop_sequence": 2,
                "arrival_time": "05:00:00",
                "departure_time": "05:00:00",
            },
            # Missing arrival time for the next hop -> NaN -> skipped.
            {
                "trip_id": "T1",
                "stop_id": "C",
                "stop_sequence": 3,
                "arrival_time": "",
                "departure_time": "",
            },
        ]
    )
    graph = build_gtfs_transit_time_network(st)
    assert graph.number_of_edges() == 0


def test_transit_past_midnight_segment_positive() -> None:
    """Times past 24:00 are handled without wrapping and yield positive durations."""
    st = _stop_times(
        [
            {
                "trip_id": "T1",
                "stop_id": "A",
                "stop_sequence": 1,
                "arrival_time": "23:59:00",
                "departure_time": "23:59:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "B",
                "stop_sequence": 2,
                "arrival_time": "24:02:00",
                "departure_time": "24:02:00",
            },
        ]
    )
    graph = build_gtfs_transit_time_network(st)
    assert graph["A"]["B"][0]["time_s"] == pytest.approx(180.0)


def test_transit_parallel_edges_preserved_across_trips() -> None:
    """Two trips serving the same stop pair keep both segments (MultiDiGraph)."""
    st = _stop_times(
        [
            {
                "trip_id": "T1",
                "stop_id": "A",
                "stop_sequence": 1,
                "arrival_time": "05:00:00",
                "departure_time": "05:00:00",
            },
            {
                "trip_id": "T1",
                "stop_id": "B",
                "stop_sequence": 2,
                "arrival_time": "05:05:00",
                "departure_time": "05:05:00",
            },
            {
                "trip_id": "T2",
                "stop_id": "A",
                "stop_sequence": 1,
                "arrival_time": "06:00:00",
                "departure_time": "06:00:00",
            },
            {
                "trip_id": "T2",
                "stop_id": "B",
                "stop_sequence": 2,
                "arrival_time": "06:03:00",
                "departure_time": "06:03:00",
            },
        ]
    )
    graph = build_gtfs_transit_time_network(st)
    assert graph.number_of_edges("A", "B") == 2
    times = sorted(d["time_s"] for d in graph["A"]["B"].values())
    assert times == [pytest.approx(180.0), pytest.approx(300.0)]


def test_transit_missing_required_columns_raises() -> None:
    st = pd.DataFrame({"trip_id": ["T1"], "stop_id": ["A"]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_gtfs_transit_time_network(st)


# ---------------------------------------------------------------------------
# Integration: real DC fixtures
# ---------------------------------------------------------------------------


def _extract_dc_gtfs(tmp_path: Path) -> Path:
    with zipfile.ZipFile(FIXTURES / "mock_gtfs_dc.zip") as zf:
        zf.extractall(tmp_path)
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    return dirs[0]


def _extract_dc_centerlines(tmp_path: Path) -> Path:
    road_dir = tmp_path / "roads"
    road_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(FIXTURES / "output_road_shps_dc.zip") as zf:
        zf.extractall(road_dir)
    return road_dir / "dc_road_centerlines.shp"


def test_transit_network_from_dc_gtfs_fixture(tmp_path: Path) -> None:
    """Builds a sane transit graph from the bundled DC GTFS feed."""
    gtfs_dir = _extract_dc_gtfs(tmp_path)
    stop_times = pd.read_csv(gtfs_dir / "stop_times.txt", dtype=str)
    trips = pd.read_csv(gtfs_dir / "trips.txt", dtype=str)

    graph = build_gtfs_transit_time_network(stop_times, trips=trips)

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    for _u, _v, data in graph.edges(data=True):
        assert data["time_s"] >= 0
        assert not math.isnan(data["time_s"])
        assert "trip_id" in data
        assert "route_id" in data  # trips table was supplied


def test_pedestrian_network_from_dc_centerlines_fixture(tmp_path: Path) -> None:
    """Builds a sane walking graph from the DC centerlines, reprojected to metres."""
    shp = _extract_dc_centerlines(tmp_path)
    centerlines = gpd.read_file(shp)
    # Fixture ships in geographic WGS84; reproject so lengths/times are metric.
    centerlines = centerlines.to_crs(DC_UTM_CRS)

    # ~1.34 m/s is a typical 3 mph walk in metres-per-second.
    graph, edge_endpoints = build_pedestrian_time_network(centerlines, walk_speed=1.34)

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert len(edge_endpoints) == graph.number_of_edges()
    for _u, _v, data in graph.edges(data=True):
        assert data["length"] > 0
        assert data["time_s"] > 0


def test_pedestrian_network_geographic_fixture_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The unprojected (geographic) centerlines fixture triggers the CRS warning."""
    shp = _extract_dc_centerlines(tmp_path)
    centerlines = gpd.read_file(shp)
    assert centerlines.crs is not None and centerlines.crs.is_geographic
    with caplog.at_level(logging.WARNING):
        build_pedestrian_time_network(centerlines, walk_speed=1.34)
    assert any("geographic" in rec.message for rec in caplog.records)
