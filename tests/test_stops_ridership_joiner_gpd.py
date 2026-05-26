from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

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
        patch("stops_ridership_joiner_gpd.OUTPUT_FOLDER", tmp_path),
    ):
        result = target.output_path("bus_stops")
    assert result == tmp_path / "bus_stops.gpkg"


def test_output_path_shp_without_route(tmp_path: Path) -> None:
    with (
        patch("stops_ridership_joiner_gpd.OUT_FORMAT", "shp"),
        patch("stops_ridership_joiner_gpd.OUTPUT_FOLDER", tmp_path),
    ):
        result = target.output_path("bus_stops")
    assert result == tmp_path / "bus_stops.shp"


def test_output_path_gpkg_with_route(tmp_path: Path) -> None:
    with (
        patch("stops_ridership_joiner_gpd.OUT_FORMAT", "gpkg"),
        patch("stops_ridership_joiner_gpd.OUTPUT_FOLDER", tmp_path),
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
