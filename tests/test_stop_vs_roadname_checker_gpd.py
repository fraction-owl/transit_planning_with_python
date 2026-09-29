"""Tests for scripts/gtfs_data_quality/stop_vs_roadname_checker_gpd.py."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

script_dir = Path("scripts/gtfs_data_quality").resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import stop_vs_roadname_checker_gpd as target  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TARGET_CRS = "EPSG:32618"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stops_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _extract_dc_gtfs(tmp_path: Path) -> Path:
    with zipfile.ZipFile(FIXTURES / "mock_gtfs_dc.zip") as zf:
        zf.extractall(tmp_path)
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    return dirs[0]


def _extract_dc_roads(tmp_path: Path) -> Path:
    road_dir = tmp_path / "roads"
    road_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(FIXTURES / "output_road_shps_dc.zip") as zf:
        zf.extractall(road_dir)
    return road_dir / "dc_road_centerlines.shp"


def _make_roads_gdf(names: list[str], crs: str = TARGET_CRS) -> gpd.GeoDataFrame:
    """Build a minimal roads GeoDataFrame with FULLNAME and related columns."""
    lines = [LineString([(0, i * 100), (1000, i * 100)]) for i in range(len(names))]
    return gpd.GeoDataFrame(
        {
            "FULLNAME": names,
            "RW_PREFIX": [""] * len(names),
            "RW_TYPE_US": ["St"] * len(names),
            "RW_SUFFIX": [""] * len(names),
            "RW_SUFFIX_": [""] * len(names),
        },
        geometry=lines,
        crs=crs,
    )


# ---------------------------------------------------------------------------
# get_crs_unit
# ---------------------------------------------------------------------------


def test_get_crs_unit_returns_metre_for_utm() -> None:
    unit = target.get_crs_unit("EPSG:32618")
    assert unit is not None
    assert "metre" in unit.lower() or "meter" in unit.lower()


def test_get_crs_unit_returns_string_for_geographic_crs() -> None:
    unit = target.get_crs_unit("EPSG:4326")
    assert unit is not None
    assert isinstance(unit, str)


# ---------------------------------------------------------------------------
# convert_buffer_distance
# ---------------------------------------------------------------------------


def test_convert_feet_to_utm_metres() -> None:
    # PyProj reports UTM units as "metre"; the conversion must still work.
    result = target.convert_buffer_distance(1.0, "feet", "EPSG:32618")
    assert result == pytest.approx(0.3048, rel=1e-9)


def test_convert_meters_to_utm_metres_is_identity() -> None:
    result = target.convert_buffer_distance(15.0, "meters", "EPSG:32618")
    assert result == pytest.approx(15.0, rel=1e-12)


def test_convert_meters_to_us_survey_feet() -> None:
    # EPSG:2248 (NAD83 / Maryland (ftUS)) uses US survey feet.
    result = target.convert_buffer_distance(1.0, "meters", "EPSG:2248")
    assert result == pytest.approx(3937 / 1200, rel=1e-9)


def test_convert_feet_to_us_survey_feet_is_near_identity() -> None:
    result = target.convert_buffer_distance(100.0, "feet", "EPSG:2248")
    assert result == pytest.approx(100.0, rel=1e-5)


def test_convert_unsupported_unit_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        target.convert_buffer_distance(1.0, "miles", "EPSG:32618")


def test_convert_geographic_crs_raises() -> None:
    with pytest.raises(ValueError, match="not a projected CRS"):
        target.convert_buffer_distance(1.0, "meters", "EPSG:4326")


# ---------------------------------------------------------------------------
# load_stops
# ---------------------------------------------------------------------------


def test_load_stops_returns_geodataframe() -> None:
    df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main St", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    gdf = target.load_stops(df)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_load_stops_geometry_is_point() -> None:
    df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main St", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    gdf = target.load_stops(df)
    assert all(isinstance(g, Point) for g in gdf.geometry)


def test_load_stops_raises_on_missing_stop_id() -> None:
    df = _make_stops_df([{"stop_name": "Main", "stop_lat": "38.9", "stop_lon": "-77.0"}])
    with pytest.raises(ValueError, match="stop_id"):
        target.load_stops(df)


def test_load_stops_raises_on_missing_stop_name() -> None:
    df = _make_stops_df([{"stop_id": "S1", "stop_lat": "38.9", "stop_lon": "-77.0"}])
    with pytest.raises(ValueError, match="stop_name"):
        target.load_stops(df)


def test_load_stops_raises_on_missing_lat_lon() -> None:
    df = _make_stops_df([{"stop_id": "S1", "stop_name": "Main"}])
    with pytest.raises(ValueError, match="stop_lat"):
        target.load_stops(df)


def test_load_stops_drops_blank_coordinates() -> None:
    df = _make_stops_df(
        [
            {"stop_id": "S1", "stop_name": "Main", "stop_lat": "38.9", "stop_lon": "-77.0"},
            {"stop_id": "P1", "stop_name": "Station", "stop_lat": "", "stop_lon": ""},
        ]
    )
    gdf = target.load_stops(df)
    assert list(gdf["stop_id"]) == ["S1"]


def test_load_stops_raises_on_non_numeric_coordinates() -> None:
    df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main", "stop_lat": "abc", "stop_lon": "-77.0"}]
    )
    with pytest.raises(ValueError):
        target.load_stops(df)


def test_load_stops_sets_correct_crs() -> None:
    df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main St", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    gdf = target.load_stops(df, crs="EPSG:4326")
    assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# normalize_street_name
# ---------------------------------------------------------------------------


def test_normalize_street_name_lowercases() -> None:
    result = target.normalize_street_name("Main Street", set())
    assert result == "main street"


def test_normalize_street_name_removes_modifiers() -> None:
    result = target.normalize_street_name("Main St", {"st"})
    assert "st" not in result


def test_normalize_street_name_handles_nan() -> None:
    result = target.normalize_street_name(float("nan"), set())
    assert result == ""


def test_normalize_street_name_strips_punctuation() -> None:
    result = target.normalize_street_name("Main St.", set())
    assert "." not in result


def test_normalize_street_name_collapses_spaces() -> None:
    result = target.normalize_street_name("Main   Street", set())
    assert "  " not in result


# ---------------------------------------------------------------------------
# extract_modifiers
# ---------------------------------------------------------------------------


def test_extract_modifiers_returns_lowercase_set() -> None:
    roads = _make_roads_gdf(["Main St"])
    mapping = {"RW_TYPE_US": "RW_TYPE_US"}
    modifiers = target.extract_modifiers(roads, mapping)
    assert "st" in modifiers


def test_extract_modifiers_skips_missing_column() -> None:
    roads = _make_roads_gdf(["Main St"])
    # Map to a column that doesn't exist
    mapping: dict[str, str] = {}
    modifiers = target.extract_modifiers(roads, mapping)
    assert isinstance(modifiers, set)


# ---------------------------------------------------------------------------
# apply_roadway_column_mapping
# ---------------------------------------------------------------------------


def test_apply_roadway_column_mapping_exposes_expected_names() -> None:
    roads = gpd.GeoDataFrame(
        {"STREET_NAME": ["Main St"], "ST_TYPE": ["St"]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs=TARGET_CRS,
    )
    mapping = {"FULLNAME": "STREET_NAME", "RW_TYPE_US": "ST_TYPE"}  # expected -> actual
    mapped = target.apply_roadway_column_mapping(roads, mapping)
    assert list(mapped["FULLNAME"]) == ["Main St"]
    modifiers = target.extract_modifiers(mapped, {col: col for col in mapping})
    assert modifiers == {"st"}


# ---------------------------------------------------------------------------
# extract_street_names
# ---------------------------------------------------------------------------


def test_extract_street_names_splits_on_at() -> None:
    names = target.extract_street_names("Main St @ Oak Ave", set())
    assert len(names) == 2


def test_extract_street_names_splits_on_ampersand() -> None:
    names = target.extract_street_names("Main St & Oak Ave", set())
    assert len(names) == 2


def test_extract_street_names_single_name() -> None:
    names = target.extract_street_names("Main Street", set())
    assert len(names) == 1


def test_extract_street_names_handles_nan() -> None:
    names = target.extract_street_names(float("nan"), set())
    assert names == []


def test_extract_street_names_normalizes_parts() -> None:
    names = target.extract_street_names("Main St @ Oak Ave", {"st", "ave"})
    # Each part should have modifiers stripped
    assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# create_buffered_stops
# ---------------------------------------------------------------------------


def test_create_buffered_stops_adds_buffered_geometry() -> None:
    df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main St", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    gdf = target.load_stops(df).to_crs(TARGET_CRS)
    buffered = target.create_buffered_stops(gdf, buffer_distance=15.0)
    assert "buffered_geometry" in buffered.columns


def test_create_buffered_stops_buffer_larger_than_point() -> None:
    df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main St", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    gdf = target.load_stops(df).to_crs(TARGET_CRS)
    buffered = target.create_buffered_stops(gdf, buffer_distance=15.0)
    # Area should be > 0 (polygon, not a point)
    assert buffered.geometry.iloc[0].area > 0


# ---------------------------------------------------------------------------
# compare_stop_to_roads
# ---------------------------------------------------------------------------


def test_compare_stop_to_roads_detects_typo() -> None:
    road_names = {"washington": {"Washington Blvd"}}
    results = target.compare_stop_to_roads(
        "S1",
        "Washingtn Blvd @ Oak",
        ["washingtn"],
        road_names,
        threshold=80,
    )
    assert [r["similar_road_name_original"] for r in results] == ["Washington Blvd"]


def test_compare_stop_to_roads_exact_match_skipped() -> None:
    road_names = {"main street": {"Main Street"}}
    results = target.compare_stop_to_roads(
        "S1",
        "Main Street @ Oak",
        ["main street"],
        road_names,
        threshold=80,
    )
    # Exact matches should be skipped (no typo)
    assert results == []


def test_compare_stop_to_roads_flags_unequal_names_scoring_100() -> None:
    # token_set_ratio scores a subset as 100; the names still differ.
    road_names = {"old mill": {"Old Mill Rd"}}
    results = target.compare_stop_to_roads("S1", "Mill @ Oak", ["mill"], road_names, 80)
    assert len(results) == 1
    assert results[0]["similarity_score"] == 100


def test_compare_stop_to_roads_returns_list() -> None:
    results = target.compare_stop_to_roads(
        "S1", "Oak Ave", ["oak ave"], {"main street": {"Main Street"}}, 80
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# process_typos
# ---------------------------------------------------------------------------


def test_process_typos_returns_dataframe() -> None:
    stops_df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main @ Oak", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    stops_gdf = target.load_stops(stops_df).to_crs(TARGET_CRS)
    roads = _make_roads_gdf(["Main Street", "Oak Avenue"])
    roads["FULLNAME_clean"] = roads["FULLNAME"].apply(
        lambda x: target.normalize_street_name(x, set())
    )
    # Both roads fall inside this stop's buffer.
    join_gdf = pd.DataFrame(
        {
            "stop_id": ["S1", "S1"],
            "FULLNAME": list(roads["FULLNAME"]),
            "FULLNAME_clean": list(roads["FULLNAME_clean"]),
        }
    )
    result = target.process_typos(stops_gdf, set(), join_gdf, 80)
    assert isinstance(result, pd.DataFrame)


def test_process_typos_empty_when_no_nearby_roads() -> None:
    stops_df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main @ Oak", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    stops_gdf = target.load_stops(stops_df).to_crs(TARGET_CRS)
    roads = _make_roads_gdf(["Main Street"])
    roads["FULLNAME_clean"] = "main street"

    # Pass an empty join DataFrame so that each stop gets no local roads
    # process_typos only calls dropna/groupby on join_gdf, no geometry ops needed
    empty_join = pd.DataFrame(columns=["stop_id", "FULLNAME", "FULLNAME_clean"])
    result = target.process_typos(stops_gdf, set(), empty_join, 80)
    assert result.empty


def test_process_typos_respects_buffer_scoping() -> None:
    """A stop is only matched against roads inside its own buffer.

    Regression guard: the fuzzy match must run against the per-stop spatial-join
    set, not the global roster of road names. Previously a stop could be flagged
    as a typo of a similarly-named road on the far side of the region because
    the buffer was used only for logging, not for the matching itself.
    """
    # Two roads ~10 km apart. Both normalize to something fuzzy-similar to the
    # stop's street token, so name similarity alone cannot distinguish them --
    # only the spatial buffer can.
    roads = gpd.GeoDataFrame(
        {
            "FULLNAME": ["Washington Blvd", "Washingten Blvd"],
            "RW_TYPE_US": ["Blvd", "Blvd"],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),  # near the stop
            LineString([(10_000, 0), (10_100, 0)]),  # far away
        ],
        crs=TARGET_CRS,
    )
    roads["FULLNAME_clean"] = roads["FULLNAME"].apply(
        lambda x: target.normalize_street_name(x, {"blvd"})
    )

    # Stop sits on the near road; its street token is a typo of both road names.
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1"], "stop_name": ["Washingtn Blvd @ Oak"]},
        geometry=[Point(50, 0)],
        crs=TARGET_CRS,
    )

    buffered = target.create_buffered_stops(stops, buffer_distance=100.0)
    join_gdf = target.spatial_join_stops_roadways(buffered, roads)
    result = target.process_typos(stops, {"blvd"}, join_gdf, threshold=80)

    matched = set(result["similar_road_name_original"])
    assert "Washington Blvd" in matched  # near road IS a candidate
    assert "Washingten Blvd" not in matched  # far road is OUT of the buffer


def test_process_typos_reports_only_originals_inside_buffer() -> None:
    """Distant roads sharing a normalized name must not appear as suggestions."""
    roads = gpd.GeoDataFrame(
        {"FULLNAME": ["Main Rd", "Main St"], "RW_TYPE_US": ["Rd", "St"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),  # near the stop
            LineString([(10_000, 0), (10_100, 0)]),  # 10 km away
        ],
        crs=TARGET_CRS,
    )
    modifiers = {"rd", "st"}
    roads["FULLNAME_clean"] = roads["FULLNAME"].apply(
        lambda x: target.normalize_street_name(x, modifiers)
    )
    stops = gpd.GeoDataFrame(
        {"stop_id": ["S1"], "stop_name": ["Mainn Rd"]},
        geometry=[Point(50, 0)],
        crs=TARGET_CRS,
    )
    buffered = target.create_buffered_stops(stops, buffer_distance=100.0)
    join_gdf = target.spatial_join_stops_roadways(buffered, roads)
    result = target.process_typos(stops, modifiers, join_gdf, threshold=80)
    assert list(result["similar_road_name_original"]) == ["Main Rd"]


# ---------------------------------------------------------------------------
# load_gtfs_data
# ---------------------------------------------------------------------------


def test_load_gtfs_data_loads_stops_txt(tmp_path: Path) -> None:
    (tmp_path / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nS1,Main,38.9,-77.0\n",
        encoding="utf-8",
    )
    data = target.load_gtfs_data(str(tmp_path), files=["stops.txt"])
    assert "stops" in data
    assert len(data["stops"]) == 1


def test_load_gtfs_data_raises_on_missing_directory() -> None:
    with pytest.raises(OSError, match="does not exist"):
        target.load_gtfs_data("/nonexistent/path")


def test_load_gtfs_data_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Missing"):
        target.load_gtfs_data(str(tmp_path), files=["stops.txt"])


def test_load_gtfs_data_raises_on_empty_file(tmp_path: Path) -> None:
    (tmp_path / "stops.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        target.load_gtfs_data(str(tmp_path), files=["stops.txt"])


# ---------------------------------------------------------------------------
# Integration: DC fixtures
# ---------------------------------------------------------------------------


def test_integration_dc_load_and_process(tmp_path: Path) -> None:
    gtfs_dir = _extract_dc_gtfs(tmp_path / "gtfs")
    road_shp = _extract_dc_roads(tmp_path)

    gtfs_data = target.load_gtfs_data(str(gtfs_dir), files=["stops.txt"])
    stops_gdf = target.load_stops(gtfs_data["stops"])
    roads_gdf = target.load_roadways(str(road_shp))

    stops_gdf = stops_gdf.to_crs(TARGET_CRS)
    roads_gdf = roads_gdf.to_crs(TARGET_CRS)

    column_mapping = {c: c for c in target.REQUIRED_COLUMNS_ROADWAY if c in roads_gdf.columns}
    assert "FULLNAME" in column_mapping, "FULLNAME expected in DC roads fixture"

    modifiers = target.extract_modifiers(roads_gdf, column_mapping)
    roads_gdf["FULLNAME_clean"] = roads_gdf["FULLNAME"].apply(
        lambda x: target.normalize_street_name(x, modifiers)
    )

    buffered = target.create_buffered_stops(stops_gdf, buffer_distance=50.0)
    join_gdf = target.spatial_join_stops_roadways(buffered, roads_gdf)
    result = target.process_typos(stops_gdf, modifiers, join_gdf, threshold=80)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# main (end to end)
# ---------------------------------------------------------------------------


def _run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_name: str) -> pd.DataFrame:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir(exist_ok=True)
    # Stop sits on the road below, ~50 m north of the UTM 18N origin line.
    stops = gpd.GeoDataFrame(
        {"stop_id": ["NA"], "stop_name": [stop_name]},
        geometry=[Point(500_000, 4_300_000)],
        crs=TARGET_CRS,
    ).to_crs("EPSG:4326")
    pd.DataFrame(
        {
            "stop_id": stops["stop_id"],
            "stop_name": stops["stop_name"],
            "stop_lat": stops.geometry.y,
            "stop_lon": stops.geometry.x,
        }
    ).to_csv(gtfs / "stops.txt", index=False)

    roads = gpd.GeoDataFrame(
        {"STREET_NAME": ["Main Rd"], "ST_TYPE": ["Rd"]},
        geometry=[LineString([(499_900, 4_300_000), (500_100, 4_300_000)])],
        crs=TARGET_CRS,
    )
    roads_path = tmp_path / "roads.gpkg"
    roads.to_file(roads_path, driver="GPKG")

    answers = {"FULLNAME": "STREET_NAME", "RW_TYPE_US": "ST_TYPE"}

    def fake_input(prompt: str) -> str:
        return next((v for k, v in answers.items() if f"'{k}'" in prompt), "")

    out_dir = tmp_path / "out"
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(target, "GTFS_FOLDER", str(gtfs))
    monkeypatch.setattr(target, "ROADWAYS_PATH", str(roads_path))
    monkeypatch.setattr(target, "OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(target, "TARGET_CRS", TARGET_CRS)
    monkeypatch.setattr(target, "BUFFER_DISTANCE_VALUE", 50)
    monkeypatch.setattr(target, "BUFFER_DISTANCE_UNIT", "feet")
    assert target.main() == 0
    return pd.read_csv(out_dir / target.OUTPUT_CSV_NAME, dtype=str, keep_default_na=False)


def test_main_custom_mapping_utm_na_id_and_clean_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Custom column mapping, a "metre" CRS, and a literal "NA" stop_id.
    first = _run_main(tmp_path, monkeypatch, "Mainn Rd")
    assert list(first["stop_id"]) == ["NA"]
    assert list(first["similar_road_name_original"]) == ["Main Rd"]

    # Fixing the name and rerunning must replace the stale finding.
    second = _run_main(tmp_path, monkeypatch, "Main Rd")
    assert second.empty
    assert "stop_id" in second.columns


# ---------------------------------------------------------------------------
# load_gtfs_data: literal NA identifiers
# ---------------------------------------------------------------------------


def test_load_gtfs_data_keeps_literal_na(tmp_path: Path) -> None:
    (tmp_path / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nNA,Main,38.9,-77.0\n",
        encoding="utf-8",
    )
    data = target.load_gtfs_data(str(tmp_path), files=["stops.txt"])
    assert data["stops"]["stop_id"].tolist() == ["NA"]
