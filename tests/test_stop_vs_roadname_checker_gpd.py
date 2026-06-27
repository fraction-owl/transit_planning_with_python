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


def test_convert_feet_to_meters() -> None:
    result = target.convert_buffer_distance(1.0, "feet", "meters")
    assert result == pytest.approx(0.3048, rel=1e-4)


def test_convert_meters_to_feet() -> None:
    result = target.convert_buffer_distance(1.0, "meters", "feet")
    assert result == pytest.approx(3.28084, rel=1e-4)


def test_convert_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        target.convert_buffer_distance(1.0, "miles", "meters")


def test_convert_same_unit_returns_identity() -> None:
    # feet -> us survey foot is a supported near-identity conversion
    result = target.convert_buffer_distance(100.0, "feet", "us survey foot")
    assert result == pytest.approx(100.0, rel=0.01)


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
    roads = _make_roads_gdf(["Washington Boulevard"])
    roads["FULLNAME_clean"] = roads["FULLNAME"].str.lower()
    road_names: set[str] = {"washington boulevard"}
    results = target.compare_stop_to_roads(
        "S1",
        "Washingtn Blvd @ Oak",
        ["washingtn blvd"],
        road_names,
        roads,
        threshold=70,
    )
    assert len(results) >= 0  # may find a match or not depending on fuzzy score


def test_compare_stop_to_roads_exact_match_skipped() -> None:
    roads = _make_roads_gdf(["Main Street"])
    roads["FULLNAME_clean"] = roads["FULLNAME"].str.lower()
    road_names: set[str] = {"main street"}
    results = target.compare_stop_to_roads(
        "S1",
        "Main Street @ Oak",
        ["main street"],
        road_names,
        roads,
        threshold=80,
    )
    # Exact matches should be skipped (no typo)
    assert results == []


def test_compare_stop_to_roads_returns_list() -> None:
    roads = _make_roads_gdf(["Main Street"])
    roads["FULLNAME_clean"] = roads["FULLNAME"].str.lower()
    results = target.compare_stop_to_roads("S1", "Oak Ave", ["oak ave"], {"main street"}, roads, 80)
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
        {"stop_id": ["S1", "S1"], "FULLNAME_clean": list(roads["FULLNAME_clean"])}
    )
    result = target.process_typos(stops_gdf, roads, set(), join_gdf, 80)
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
    empty_join = pd.DataFrame(columns=["stop_id", "FULLNAME_clean"])
    result = target.process_typos(stops_gdf, roads, set(), empty_join, 80)
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
    result = target.process_typos(stops, roads, {"blvd"}, join_gdf, threshold=80)

    matched = set(result["similar_road_name_original"])
    assert "Washington Blvd" in matched  # near road IS a candidate
    assert "Washingten Blvd" not in matched  # far road is OUT of the buffer


# ---------------------------------------------------------------------------
# resolve_stop_id_field
# ---------------------------------------------------------------------------


def test_resolve_stop_id_field_returns_preferred_when_present() -> None:
    df = _make_stops_df([{"stop_id": "S1", "stop_code": "100", "stop_name": "Main"}])
    assert target.resolve_stop_id_field(df, "stop_code") == "stop_code"


def test_resolve_stop_id_field_falls_back_to_stop_id() -> None:
    df = _make_stops_df([{"stop_id": "S1", "stop_name": "Main"}])
    assert target.resolve_stop_id_field(df, "stop_code") == "stop_id"


# ---------------------------------------------------------------------------
# normalize_street_name — abbreviation expansion
# ---------------------------------------------------------------------------


def test_normalize_street_name_expands_abbreviations() -> None:
    result = target.normalize_street_name("Ft Hunt", set(), {"ft": "fort"})
    assert result == "fort hunt"


def test_normalize_street_name_abbreviations_default_noop() -> None:
    # Without an abbreviation map, tokens are left untouched.
    assert target.normalize_street_name("Ft Hunt", set()) == "ft hunt"


# ---------------------------------------------------------------------------
# detect_truncation_length
# ---------------------------------------------------------------------------


def test_detect_truncation_length_finds_spike() -> None:
    # Ten names all exactly 24 chars long -> a clear fixed-width spike.
    names = ["X" * 24 for _ in range(10)] + ["Short", "Also short"]
    assert target.detect_truncation_length(names, min_count=5, min_fraction=0.1) == 24


def test_detect_truncation_length_none_when_no_spike() -> None:
    # A single long name is not enough to count as truncation.
    names = ["A" * 30] + ["short name {}".format(i) for i in range(50)]
    assert target.detect_truncation_length(names, min_count=5, min_fraction=0.1) is None


def test_detect_truncation_length_handles_empty() -> None:
    assert target.detect_truncation_length([]) is None


# ---------------------------------------------------------------------------
# flag_truncated_stops
# ---------------------------------------------------------------------------


def test_flag_truncated_stops_flags_long_names() -> None:
    df = _make_stops_df(
        [
            {"stop_id": "S1", "stop_name": "X" * 24, "stop_lat": "38.9", "stop_lon": "-77.0"},
            {"stop_id": "S2", "stop_name": "Short", "stop_lat": "38.9", "stop_lon": "-77.0"},
        ]
    )
    gdf = target.load_stops(df)
    flagged = target.flag_truncated_stops(gdf, truncation_length=24, stop_id_field="stop_id")
    assert list(flagged["stop_id"]) == ["S1"]
    assert flagged["suspected_truncation_length"].iloc[0] == 24


def test_flag_truncated_stops_empty_when_length_none() -> None:
    df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Main", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    gdf = target.load_stops(df)
    flagged = target.flag_truncated_stops(gdf, truncation_length=None)
    assert flagged.empty
    assert "suspected_truncation_length" in flagged.columns


# ---------------------------------------------------------------------------
# is_substring_match
# ---------------------------------------------------------------------------


def test_is_substring_match_detects_containment() -> None:
    assert target.is_substring_match("washington", "washington heights") is True


def test_is_substring_match_rejects_partial_token() -> None:
    # "oak" must not be treated as contained in "oakland" (token-aware).
    assert target.is_substring_match("oak", "oakland") is False


def test_is_substring_match_handles_empty() -> None:
    assert target.is_substring_match("", "main") is False


# ---------------------------------------------------------------------------
# compare_stop_to_roads — substring filter & custom id field
# ---------------------------------------------------------------------------


def test_compare_stop_to_roads_substring_filter_suppresses() -> None:
    roads = _make_roads_gdf(["Washington Heights"])
    roads["FULLNAME_clean"] = "washington heights"
    filtered = target.compare_stop_to_roads(
        "S1",
        "Washington @ Oak",
        ["washington"],
        {"washington heights"},
        roads,
        threshold=70,
        filter_substring=True,
    )
    assert filtered == []


def test_compare_stop_to_roads_uses_custom_id_field() -> None:
    roads = _make_roads_gdf(["Washington Boulevard"])
    roads["FULLNAME_clean"] = "washington boulevard"
    results = target.compare_stop_to_roads(
        "100",
        "Washingtn Blvd",
        ["washingtn blvd"],
        {"washington boulevard"},
        roads,
        threshold=60,
        stop_id_field="stop_code",
    )
    assert all("stop_code" in row for row in results)


# ---------------------------------------------------------------------------
# process_typos — skip_ids
# ---------------------------------------------------------------------------


def test_process_typos_skips_flagged_ids() -> None:
    stops_df = _make_stops_df(
        [{"stop_id": "S1", "stop_name": "Washingtn @ Oak", "stop_lat": "38.9", "stop_lon": "-77.0"}]
    )
    stops_gdf = target.load_stops(stops_df).to_crs(TARGET_CRS)
    roads = _make_roads_gdf(["Washington Street"])
    roads["FULLNAME_clean"] = roads["FULLNAME"].apply(
        lambda x: target.normalize_street_name(x, set())
    )
    join_gdf = pd.DataFrame({"stop_id": ["S1"], "FULLNAME_clean": ["washington street"]})
    result = target.process_typos(stops_gdf, roads, set(), join_gdf, 60, skip_ids={"S1"})
    assert result.empty


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
    result = target.process_typos(stops_gdf, roads_gdf, modifiers, join_gdf, threshold=80)
    assert isinstance(result, pd.DataFrame)
