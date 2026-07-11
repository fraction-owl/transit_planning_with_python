from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

script_dir = Path("scripts/ridership_tools").resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import stops_ridership_joiner_gpd as target  # noqa: E402

# ---------------------------------------------------------------------------
# is_gtfs_txt
# ---------------------------------------------------------------------------


def test_is_gtfs_txt_true_for_txt_suffix() -> None:
    assert target.is_gtfs_txt(Path("stops.txt")) is True


def test_is_gtfs_txt_false_for_shp_suffix() -> None:
    assert target.is_gtfs_txt(Path("stops.shp")) is False


def test_is_gtfs_txt_false_for_gpkg_suffix() -> None:
    assert target.is_gtfs_txt(Path("stops.gpkg")) is False


def test_is_gtfs_txt_case_insensitive() -> None:
    assert target.is_gtfs_txt(Path("STOPS.TXT")) is True


# ---------------------------------------------------------------------------
# _safe_to_str
# ---------------------------------------------------------------------------


def test_safe_to_str_converts_int_to_string() -> None:
    s = pd.Series([1001, 2002])
    result = target._safe_to_str(s)
    assert result.iloc[0] == "1001"
    assert result.iloc[1] == "2002"


def test_safe_to_str_preserves_nan() -> None:
    s = pd.Series([1001, None])
    result = target._safe_to_str(s)
    assert pd.isna(result.iloc[1])


# ---------------------------------------------------------------------------
# _require_columns
# ---------------------------------------------------------------------------


def test_require_columns_all_present_does_not_raise() -> None:
    df = pd.DataFrame({"A": [1], "B": [2]})
    target._require_columns(df, ["A", "B"], context="test")  # no exception


def test_require_columns_missing_field_raises_value_error() -> None:
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        target._require_columns(df, ["A", "MISSING"], context="test")


def test_require_columns_error_message_names_the_missing_field() -> None:
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError, match="MISSING"):
        target._require_columns(df, ["MISSING"], context="test")


# ---------------------------------------------------------------------------
# _to_common_crs
# ---------------------------------------------------------------------------


def _points(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs=crs)


def _polygons(crs: str) -> gpd.GeoDataFrame:
    ring = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    return gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon(ring)], crs=crs)


def test_to_common_crs_same_crs_leaves_points_unchanged() -> None:
    pts, polys = target._to_common_crs(_points("EPSG:4326"), _polygons("EPSG:4326"))
    assert pts.crs == polys.crs


def test_to_common_crs_reprojects_points_to_polygon_crs() -> None:
    pts_out, polys_out = target._to_common_crs(_points("EPSG:4326"), _polygons("EPSG:3857"))
    assert pts_out.crs == polys_out.crs


def test_to_common_crs_no_points_crs_raises() -> None:
    pts = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])  # no CRS
    with pytest.raises(ValueError, match="Stop layer has no CRS"):
        target._to_common_crs(pts, _polygons("EPSG:4326"))


def test_to_common_crs_no_polygon_crs_raises() -> None:
    polys = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])  # no CRS
    with pytest.raises(ValueError, match="Polygon layer has no CRS"):
        target._to_common_crs(_points("EPSG:4326"), polys)


# ---------------------------------------------------------------------------
# output_path
# ---------------------------------------------------------------------------


def test_output_path_gpkg_without_route(tmp_path: Path) -> None:
    with (
        patch("stops_ridership_joiner_gpd.OUT_FORMAT", "gpkg"),
        patch("stops_ridership_joiner_gpd.VECTOR_DIR", tmp_path),
    ):
        result = target.output_path("bus_stops")
    assert result == tmp_path / "bus_stops.gpkg"


def test_output_path_shp_without_route(tmp_path: Path) -> None:
    with (
        patch("stops_ridership_joiner_gpd.OUT_FORMAT", "shp"),
        patch("stops_ridership_joiner_gpd.VECTOR_DIR", tmp_path),
    ):
        result = target.output_path("bus_stops")
    assert result == tmp_path / "bus_stops.shp"


def test_output_path_gpkg_with_route(tmp_path: Path) -> None:
    with (
        patch("stops_ridership_joiner_gpd.OUT_FORMAT", "gpkg"),
        patch("stops_ridership_joiner_gpd.VECTOR_DIR", tmp_path),
    ):
        result = target.output_path("bus_stops", route="10A")
    assert result == tmp_path / "bus_stops_10A.gpkg"


# ---------------------------------------------------------------------------
# write_vector
# ---------------------------------------------------------------------------


def _simple_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")


def test_write_vector_gpkg_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out.gpkg"
    target.write_vector(_simple_gdf(), path)
    assert path.exists()


def test_write_vector_shp_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out.shp"
    target.write_vector(_simple_gdf(), path)
    assert path.exists()


def test_write_vector_unsupported_extension_raises() -> None:
    path = Path("/tmp/out.csv")
    with pytest.raises(ValueError, match="Unsupported output format"):
        target.write_vector(_simple_gdf(), path)


# ---------------------------------------------------------------------------
# aggregate_excel_per_stop
# ---------------------------------------------------------------------------


def _excel_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STOP_ID": ["1001", "1001", "2002"],
            "ROUTE_NAME": ["10A", "10B", "20"],
            "XBOARDINGS": [12.0, 18.0, 5.0],
            "XALIGHTINGS": [3.0, 7.0, 2.0],
            "TOTAL": [15.0, 25.0, 7.0],
        }
    )


def test_aggregate_excel_per_stop_sums_boardings() -> None:
    result = target.aggregate_excel_per_stop(_excel_df())
    row = result[result["STOP_ID"] == "1001"].iloc[0]
    assert row["XBOARDINGS"] == pytest.approx(30.0)


def test_aggregate_excel_per_stop_sums_alightings() -> None:
    result = target.aggregate_excel_per_stop(_excel_df())
    row = result[result["STOP_ID"] == "1001"].iloc[0]
    assert row["XALIGHTINGS"] == pytest.approx(10.0)


def test_aggregate_excel_per_stop_sums_total() -> None:
    result = target.aggregate_excel_per_stop(_excel_df())
    row = result[result["STOP_ID"] == "1001"].iloc[0]
    assert row["TOTAL"] == pytest.approx(40.0)


def test_aggregate_excel_per_stop_one_row_per_stop() -> None:
    result = target.aggregate_excel_per_stop(_excel_df())
    assert result["STOP_ID"].nunique() == len(result)


# ---------------------------------------------------------------------------
# merge_ridership
# ---------------------------------------------------------------------------


def _stops_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "stop_code": ["1001", "2002", "3003"],
            "stop_name": ["Oak St", "Elm Ave", "Pine Rd"],
        },
        geometry=[Point(-77.1, 38.9), Point(-77.2, 38.8), Point(-77.3, 38.7)],
        crs="EPSG:4326",
    )


def _agg_excel_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STOP_ID": ["1001", "2002"],
            "XBOARDINGS": [30.0, 5.0],
            "XALIGHTINGS": [10.0, 2.0],
            "TOTAL": [40.0, 7.0],
        }
    )


def test_merge_ridership_returns_geodataframe() -> None:
    result = target.merge_ridership(_stops_gdf(), _agg_excel_df(), "stop_code")
    assert isinstance(result, gpd.GeoDataFrame)


def test_merge_ridership_inner_join_excludes_unmatched_stops() -> None:
    # stop 3003 has no ridership row → excluded
    result = target.merge_ridership(_stops_gdf(), _agg_excel_df(), "stop_code")
    assert len(result) == 2


def test_merge_ridership_preserves_stops_crs() -> None:
    result = target.merge_ridership(_stops_gdf(), _agg_excel_df(), "stop_code")
    assert result.crs == _stops_gdf().crs


def test_merge_ridership_missing_key_field_raises() -> None:
    with pytest.raises(ValueError, match="not found in stops layer"):
        target.merge_ridership(_stops_gdf(), _agg_excel_df(), "nonexistent_key")


# ---------------------------------------------------------------------------
# add_output_ridership_fields
# ---------------------------------------------------------------------------


def _matched_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "stop_code": ["1001", "2002"],
            "XBOARDINGS": [30.0, 5.0],
            "XALIGHTINGS": [10.0, 2.0],
            "TOTAL": [40.0, 7.0],
        },
        geometry=[Point(-77.1, 38.9), Point(-77.2, 38.8)],
        crs="EPSG:4326",
    )


def test_add_output_ridership_fields_creates_xboard_column() -> None:
    result = target.add_output_ridership_fields(_matched_gdf())
    assert "XBOARD" in result.columns


def test_add_output_ridership_fields_creates_xalight_column() -> None:
    result = target.add_output_ridership_fields(_matched_gdf())
    assert "XALIGHT" in result.columns


def test_add_output_ridership_fields_creates_xtotal_column() -> None:
    result = target.add_output_ridership_fields(_matched_gdf())
    assert "XTOTAL" in result.columns


def test_add_output_ridership_fields_values_are_float() -> None:
    result = target.add_output_ridership_fields(_matched_gdf())
    assert result["XBOARD"].dtype == float
    assert result["XALIGHT"].dtype == float
    assert result["XTOTAL"].dtype == float


def test_add_output_ridership_fields_correct_values() -> None:
    result = target.add_output_ridership_fields(_matched_gdf())
    assert result["XBOARD"].iloc[0] == pytest.approx(30.0)
    assert result["XALIGHT"].iloc[0] == pytest.approx(10.0)
    assert result["XTOTAL"].iloc[0] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# aggregate_by_polygon
# ---------------------------------------------------------------------------


def _matched_with_geoid() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "stop_code": ["1001", "2002", "3003"],
            "GEOID": ["P1", "P1", "P2"],
            "XBOARDINGS": [10.0, 20.0, 5.0],
            "XALIGHTINGS": [3.0, 7.0, 2.0],
            "TOTAL": [13.0, 27.0, 7.0],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs="EPSG:4326",
    )


def _polygon_layer() -> gpd.GeoDataFrame:
    p1 = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    p2 = Polygon([(2, 0), (4, 0), (4, 1), (2, 1)])
    return gpd.GeoDataFrame({"GEOID": ["P1", "P2"]}, geometry=[p1, p2], crs="EPSG:4326")


def test_aggregate_by_polygon_returns_geodataframe() -> None:
    result = target.aggregate_by_polygon(_matched_with_geoid(), _polygon_layer())
    assert isinstance(result, gpd.GeoDataFrame)


def test_aggregate_by_polygon_sums_boardings_per_polygon() -> None:
    result = target.aggregate_by_polygon(_matched_with_geoid(), _polygon_layer())
    p1 = result[result["GEOID"] == "P1"].iloc[0]
    assert p1["XBOARD_SUM"] == pytest.approx(30.0)  # 10 + 20


def test_aggregate_by_polygon_unmatched_polygon_gets_zero() -> None:
    p3 = Polygon([(4, 0), (6, 0), (6, 1), (4, 1)])
    extra_row = gpd.GeoDataFrame({"GEOID": ["P3"]}, geometry=[p3], crs="EPSG:4326")
    extended = gpd.GeoDataFrame(
        pd.concat([_polygon_layer(), extra_row], ignore_index=True), crs="EPSG:4326"
    )
    result = target.aggregate_by_polygon(_matched_with_geoid(), extended)
    p3_row = result[result["GEOID"] == "P3"].iloc[0]
    assert p3_row["XBOARD_SUM"] == pytest.approx(0.0)


def test_aggregate_by_polygon_missing_geoid_in_stops_raises() -> None:
    matched_no_geoid = _matched_with_geoid().drop(columns=["GEOID"])
    with pytest.raises(ValueError, match="polygon join field"):
        target.aggregate_by_polygon(matched_no_geoid, _polygon_layer())


# ---------------------------------------------------------------------------
# extract_config_block
# ---------------------------------------------------------------------------


def test_extract_config_block_returns_inner_content(tmp_path: Path) -> None:
    src = tmp_path / "script.py"
    src.write_text(
        "# preamble\n"
        "# === BEGIN CONFIG ===\n"
        "KEY = 1\n"
        "OTHER = 2\n"
        "# === END CONFIG ===\n"
        "# epilogue\n",
        encoding="utf-8",
    )
    block = target.extract_config_block(src)
    assert "KEY = 1" in block
    assert "OTHER = 2" in block


def test_extract_config_block_excludes_markers_and_surrounding_text(tmp_path: Path) -> None:
    src = tmp_path / "script.py"
    src.write_text(
        "# preamble\n# === BEGIN CONFIG ===\nKEY = 1\n# === END CONFIG ===\n# epilogue\n",
        encoding="utf-8",
    )
    block = target.extract_config_block(src)
    assert "preamble" not in block
    assert "epilogue" not in block
    assert "BEGIN CONFIG" not in block
    assert "END CONFIG" not in block


def test_extract_config_block_missing_markers_raises(tmp_path: Path) -> None:
    src = tmp_path / "script.py"
    src.write_text("KEY = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Config markers not found"):
        target.extract_config_block(src)


# ---------------------------------------------------------------------------
# write_run_log
# ---------------------------------------------------------------------------


def test_write_run_log_creates_runlog_file(tmp_path: Path) -> None:
    result = target.write_run_log(tmp_path)
    assert result is True
    assert (tmp_path / "stops_ridership_joiner_gpd_runlog.txt").exists()


def test_write_run_log_file_contains_config_values(tmp_path: Path) -> None:
    target.write_run_log(tmp_path)
    content = (tmp_path / "stops_ridership_joiner_gpd_runlog.txt").read_text(encoding="utf-8")
    # The config block from the real script must appear in the log
    assert "BEGIN CONFIG" not in content  # markers stripped
    assert "CONFIGURATION" in content


def test_write_run_log_returns_false_on_write_error(tmp_path: Path) -> None:
    fake_source = "# === BEGIN CONFIG ===\nKEY = 1\n# === END CONFIG ===\n"
    with (
        patch.object(Path, "read_text", return_value=fake_source),
        patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
    ):
        result = target.write_run_log(tmp_path)
    assert result is False


def test_write_run_log_live_snapshot_when_no_source(tmp_path: Path) -> None:
    # No SOURCE_FILE_OVERRIDE and no SELF_PATH → falls back to a live snapshot,
    # which must still produce a log naming a config global rather than failing.
    with (
        patch("stops_ridership_joiner_gpd.SOURCE_FILE_OVERRIDE", ""),
        patch("stops_ridership_joiner_gpd.SELF_PATH", None),
    ):
        result = target.write_run_log(tmp_path)
    assert result is True
    content = (tmp_path / "stops_ridership_joiner_gpd_runlog.txt").read_text(encoding="utf-8")
    assert "live snapshot" in content
    assert "OUT_FORMAT" in content  # a config global captured by the snapshot


# ---------------------------------------------------------------------------
# is_gtfs_input / resolve_gtfs_dir / resolve_stops_table
# ---------------------------------------------------------------------------


def test_is_gtfs_input_true_for_directory(tmp_path: Path) -> None:
    assert target.is_gtfs_input(tmp_path) is True


def test_is_gtfs_input_true_for_txt() -> None:
    assert target.is_gtfs_input(Path("stops.txt")) is True


def test_is_gtfs_input_false_for_shapefile() -> None:
    assert target.is_gtfs_input(Path("stops.shp")) is False


def test_resolve_gtfs_dir_returns_directory_as_is(tmp_path: Path) -> None:
    assert target.resolve_gtfs_dir(tmp_path) == tmp_path


def test_resolve_gtfs_dir_returns_parent_for_stops_txt() -> None:
    assert target.resolve_gtfs_dir(Path("/feed/stops.txt")) == Path("/feed")


def test_resolve_gtfs_dir_none_for_vector_file() -> None:
    assert target.resolve_gtfs_dir(Path("stops.shp")) is None


def test_resolve_stops_table_appends_stops_txt_for_dir(tmp_path: Path) -> None:
    assert target.resolve_stops_table(tmp_path) == tmp_path / "stops.txt"


def test_resolve_stops_table_returns_file_unchanged() -> None:
    p = Path("/feed/stops.txt")
    assert target.resolve_stops_table(p) == p


# ---------------------------------------------------------------------------
# write_vector creates parent directories
# ---------------------------------------------------------------------------


def test_write_vector_creates_missing_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "vector" / "sub"
    path = nested / "out.gpkg"
    target.write_vector(_simple_gdf(), path)
    assert path.exists()


def test_write_vector_drops_case_insensitive_duplicate_field(tmp_path: Path) -> None:
    # GTFS 'stop_id' + Excel 'STOP_ID' collide case-insensitively in OGR; the
    # write must succeed and keep only one of them in the file.
    gdf = gpd.GeoDataFrame(
        {"stop_id": ["a"], "STOP_ID": ["a"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    path = tmp_path / "out.gpkg"
    target.write_vector(gdf, path)  # must not raise
    written = gpd.read_file(path)
    assert [c.lower() for c in written.columns].count("stop_id") == 1


def test_drop_case_insensitive_duplicate_fields_keeps_first() -> None:
    gdf = gpd.GeoDataFrame(
        {"stop_id": ["a"], "STOP_ID": ["b"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    result = target._drop_case_insensitive_duplicate_fields(gdf)
    assert "stop_id" in result.columns
    assert "STOP_ID" not in result.columns


# ---------------------------------------------------------------------------
# normalize_route_name
# ---------------------------------------------------------------------------


def test_normalize_route_name_uppercases_and_strips() -> None:
    assert target.normalize_route_name("  10a ") == "10A"


def test_normalize_route_name_none_returns_empty() -> None:
    assert target.normalize_route_name(None) == ""


def test_normalize_route_name_nan_returns_empty() -> None:
    assert target.normalize_route_name(float("nan")) == ""


# ---------------------------------------------------------------------------
# _polyline_length
# ---------------------------------------------------------------------------


def test_polyline_length_of_two_unit_steps() -> None:
    pl = pd.DataFrame({"lon": [0.0, 3.0, 3.0], "lat": [0.0, 0.0, 4.0]})
    assert target._polyline_length(pl) == pytest.approx(7.0)  # 3 + 4


# ---------------------------------------------------------------------------
# color_for_value
# ---------------------------------------------------------------------------


def test_color_for_value_low_bin() -> None:
    assert target.color_for_value(0.0) == "green"


def test_color_for_value_mid_bin() -> None:
    assert target.color_for_value(10.0) == "yellow"


def test_color_for_value_top_open_bin() -> None:
    assert target.color_for_value(9999.0) == "red"


def test_color_for_value_bin_boundary_is_lower_inclusive() -> None:
    # 5.0 is the lower edge of the middle bin (half-open [5, 25))
    assert target.color_for_value(5.0) == "yellow"


# ---------------------------------------------------------------------------
# _normalize_hex_color
# ---------------------------------------------------------------------------


def test_normalize_hex_color_bare_six_digits() -> None:
    assert target._normalize_hex_color("FF8800") == "#FF8800"


def test_normalize_hex_color_strips_leading_hash() -> None:
    assert target._normalize_hex_color("#00ff00") == "#00ff00"


def test_normalize_hex_color_wrong_length_returns_none() -> None:
    assert target._normalize_hex_color("FFF") is None


def test_normalize_hex_color_non_hex_returns_none() -> None:
    assert target._normalize_hex_color("ZZZZZZ") is None


def test_normalize_hex_color_none_returns_none() -> None:
    assert target._normalize_hex_color(None) is None


# ---------------------------------------------------------------------------
# _roads_in_extent
# ---------------------------------------------------------------------------


def _roads_index() -> list:
    inside = pd.DataFrame({"lon": [0.0, 1.0], "lat": [0.0, 1.0]})
    outside = pd.DataFrame({"lon": [10.0, 11.0], "lat": [10.0, 11.0]})
    return [
        (0.0, 1.0, 0.0, 1.0, inside),
        (10.0, 11.0, 10.0, 11.0, outside),
    ]


def test_roads_in_extent_returns_only_overlapping() -> None:
    result = target._roads_in_extent(_roads_index(), -1.0, 2.0, -1.0, 2.0)
    assert len(result) == 1


def test_roads_in_extent_excludes_disjoint() -> None:
    result = target._roads_in_extent(_roads_index(), 100.0, 200.0, 100.0, 200.0)
    assert result == []


# ---------------------------------------------------------------------------
# GTFS shape/stop lookups (build_route_shape_lookup, load_stop_coords)
# ---------------------------------------------------------------------------


def _write_min_gtfs(gtfs_dir: Path) -> None:
    """Write a tiny but valid GTFS feed (routes/trips/shapes/stops) into gtfs_dir."""
    gtfs_dir.mkdir(parents=True, exist_ok=True)
    (gtfs_dir / "routes.txt").write_text(
        "route_id,route_short_name,route_color\nR1,10A,FF0000\nR2,20,\n",
        encoding="utf-8",
    )
    (gtfs_dir / "trips.txt").write_text(
        "route_id,shape_id\nR1,S1\nR1,S1\nR1,S2\nR2,S3\n",
        encoding="utf-8",
    )
    (gtfs_dir / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        "S1,38.90,-77.10,1\nS1,38.95,-77.05,2\nS1,39.00,-77.00,3\n"
        "S2,38.80,-77.20,1\nS2,38.85,-77.15,2\n"
        "S3,38.70,-77.05,1\nS3,38.72,-77.00,2\n",
        encoding="utf-8",
    )
    (gtfs_dir / "stops.txt").write_text(
        "stop_code,stop_id,stop_name,stop_lat,stop_lon\n"
        "1001,A,Oak St,38.90,-77.10\n2002,B,Elm Ave,38.80,-77.20\n",
        encoding="utf-8",
    )


def test_build_route_shape_lookup_all_shapes(tmp_path: Path) -> None:
    _write_min_gtfs(tmp_path)
    with patch("stops_ridership_joiner_gpd.PLOT_ALL_SHAPES_PER_ROUTE", True):
        polylines, colors = target.build_route_shape_lookup(tmp_path)
    # Route 10A has two distinct shapes (S1 deduped + S2)
    assert len(polylines["10A"]) == 2
    assert len(polylines["20"]) == 1
    assert colors["10A"] == "#FF0000"


def test_build_route_shape_lookup_single_longest(tmp_path: Path) -> None:
    _write_min_gtfs(tmp_path)
    with patch("stops_ridership_joiner_gpd.PLOT_ALL_SHAPES_PER_ROUTE", False):
        polylines, _colors = target.build_route_shape_lookup(tmp_path)
    # Only the longest shape (S1, three points) is kept for 10A
    assert len(polylines["10A"]) == 1
    assert len(polylines["10A"][0]) == 3


def test_load_stop_coords_reads_stops(tmp_path: Path) -> None:
    _write_min_gtfs(tmp_path)
    coords = target.load_stop_coords(tmp_path)
    assert set(coords.columns) == {"stop_code", "stop_lon", "stop_lat"}
    assert len(coords) == 2
    row = coords[coords["stop_code"] == "1001"].iloc[0]
    assert row["stop_lon"] == pytest.approx(-77.10)


# ---------------------------------------------------------------------------
# load_road_polylines
# ---------------------------------------------------------------------------


def test_load_road_polylines_reads_line_segments(tmp_path: Path) -> None:
    roads = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            LineString([(-77.1, 38.9), (-77.0, 38.95)]),
            LineString([(-77.2, 38.8), (-77.1, 38.85)]),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "roads.gpkg"
    roads.to_file(path, driver="GPKG")
    indexed = target.load_road_polylines(path)
    assert len(indexed) == 2
    xmin, xmax, ymin, ymax, df = indexed[0]
    assert xmin <= xmax and ymin <= ymax
    assert set(df.columns) == {"lon", "lat"}


def test_load_road_polylines_reprojects_to_wgs84(tmp_path: Path) -> None:
    # A line defined in Web Mercator must come back in lon/lat degrees.
    roads = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(-8000000, 4700000), (-8000100, 4700100)])],
        crs="EPSG:3857",
    )
    path = tmp_path / "roads_3857.gpkg"
    roads.to_file(path, driver="GPKG")
    indexed = target.load_road_polylines(path)
    _xmin, xmax, _ymin, ymax, _df = indexed[0]
    assert -180.0 <= xmax <= 180.0
    assert -90.0 <= ymax <= 90.0


def test_load_road_polylines_handles_multilinestring(tmp_path: Path) -> None:
    roads = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[
            MultiLineString(
                [
                    [(-77.1, 38.9), (-77.0, 38.95)],
                    [(-77.2, 38.8), (-77.1, 38.85)],
                ]
            )
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "roads_multi.gpkg"
    roads.to_file(path, driver="GPKG")
    indexed = target.load_road_polylines(path)
    assert len(indexed) == 2  # one segment per part


# ---------------------------------------------------------------------------
# generate_route_plots (end-to-end, headless)
# ---------------------------------------------------------------------------


def test_generate_route_plots_writes_pngs(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    _write_min_gtfs(gtfs_dir)
    plot_dir = tmp_path / "plots"

    ridership = pd.DataFrame(
        {
            "ROUTE_NAME": ["10A", "10A", "20"],
            "STOP_ID": ["1001", "2002", "1001"],
            "XBOARDINGS": [10.0, 30.0, 5.0],
            "XALIGHTINGS": [3.0, 7.0, 2.0],
        }
    )

    with (
        patch("stops_ridership_joiner_gpd.BUS_STOPS_INPUT", gtfs_dir),
        patch("stops_ridership_joiner_gpd.PLOT_DIR", plot_dir),
        patch("stops_ridership_joiner_gpd.ROADS_SHAPEFILE", None),
        patch("stops_ridership_joiner_gpd.read_and_filter_excel", return_value=ridership),
    ):
        target.generate_route_plots()

    expected = [
        "route_10A_boardings.png",
        "route_10A_alightings.png",
        "route_20_boardings.png",
        "route_20_alightings.png",
    ]
    for name in expected:
        assert (plot_dir / name).exists(), f"missing {name}"


def test_generate_route_plots_exits_for_non_gtfs_input(tmp_path: Path) -> None:
    shp_input = tmp_path / "stops.shp"  # not a folder or stops.txt
    with (
        patch("stops_ridership_joiner_gpd.BUS_STOPS_INPUT", shp_input),
        pytest.raises(SystemExit),
    ):
        target.generate_route_plots()
