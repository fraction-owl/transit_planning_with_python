from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

import scripts.ridership_tools.district_ridership_share_gpd as target

# Two adjacent unit squares sharing the lon = -76.95 edge (WGS84).
_EDGE_LON = -76.95


def _districts_gdf() -> gpd.GeoDataFrame:
    square_a = Polygon([(-77.0, 38.8), (_EDGE_LON, 38.8), (_EDGE_LON, 39.0), (-77.0, 39.0)])
    square_b = Polygon([(_EDGE_LON, 38.8), (-76.9, 38.8), (-76.9, 39.0), (_EDGE_LON, 39.0)])
    return gpd.GeoDataFrame(
        {"DISTRICT": ["District A", "District B"]},
        geometry=[square_a, square_b],
        crs="EPSG:4326",
    )


def _stops_txt_df() -> pd.DataFrame:
    # 1001 inside A, 1002 inside B, 1003 exactly on the shared edge, 1004 is a
    # station (filtered out), one blank stop_code, one row missing coords.
    return pd.DataFrame(
        {
            "stop_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
            "stop_code": ["1001", "1002", "1003", "1004", "", "1006"],
            "stop_lat": ["38.90", "38.90", "38.90", "38.90", "38.90", None],
            "stop_lon": ["-76.98", "-76.92", str(_EDGE_LON), "-76.97", "-76.96", "-76.99"],
            "location_type": ["0", "", "0", "1", "0", "0"],
        }
    )


def _ridership_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STOP_ID": [1001.0, 1001.0, 1002.0, 1002.0, 1003.0, 9999.0, 1002.0],
            "ROUTE_NAME": ["R1", "R1", "R1", "R2", "R1", "R1", "DROPME"],
            "TIME_PERIOD": [
                "AM Peak",
                "PM Peak",
                "AM Peak",
                "PM Peak",
                "AM Peak",
                "AM Peak",
                "AM Peak",
            ],
            "BOARD_ALL": [10.0, 5.0, 20.0, 4.0, 8.0, 7.0, 100.0],
            "ALIGHT_ALL": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
        }
    )


def _write_inputs(tmp_path: Path) -> dict[str, str]:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _stops_txt_df().to_csv(gtfs_dir / "stops.txt", index=False)

    excel = tmp_path / "ridership.xlsx"
    _ridership_df().to_excel(excel, index=False)

    districts = tmp_path / "districts.geojson"
    _districts_gdf().to_file(districts, driver="GeoJSON")

    out_dir = tmp_path / "out"
    return {
        "gtfs": str(gtfs_dir),
        "excel": str(excel),
        "districts": str(districts),
        "out": str(out_dir),
        "logs": str(out_dir / "logs"),
    }


def _cli(paths: dict[str, str], *extra: str) -> list[str]:
    return [
        "--gtfs-input",
        paths["gtfs"],
        "--excel-file",
        paths["excel"],
        "--districts-fc",
        paths["districts"],
        "--output-dir",
        paths["out"],
        "--log-dir",
        paths["logs"],
        "--routes-exclude",
        "DROPME",
        *extra,
    ]


# ---------------------------------------------------------------------------
# _clean_id
# ---------------------------------------------------------------------------


def test_clean_id_renders_integral_floats_without_decimal() -> None:
    assert target._clean_id(1234.0) == "1234"


def test_clean_id_strips_strings_and_handles_nan() -> None:
    assert target._clean_id("  A12 ") == "A12"
    assert target._clean_id(float("nan")) == ""
    assert target._clean_id(12.5) == "12.5"


# ---------------------------------------------------------------------------
# load_gtfs_stops
# ---------------------------------------------------------------------------


def test_load_gtfs_stops_reads_folder(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    df = target.load_gtfs_stops(paths["gtfs"])
    assert len(df) == 6
    # Read as strings: numeric-looking codes must not become numbers.
    assert all(isinstance(v, str) for v in df["stop_code"].dropna())


def test_load_gtfs_stops_reads_nested_zip(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    feed = tmp_path / "feed.zip"
    with zipfile.ZipFile(feed, "w") as z:
        z.write(Path(paths["gtfs"]) / "stops.txt", "wrapper/stops.txt")
    df = target.load_gtfs_stops(str(feed))
    assert len(df) == 6


def test_load_gtfs_stops_ambiguous_zip_raises(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    feed = tmp_path / "feed.zip"
    with zipfile.ZipFile(feed, "w") as z:
        z.write(Path(paths["gtfs"]) / "stops.txt", "a/stops.txt")
        z.write(Path(paths["gtfs"]) / "stops.txt", "b/stops.txt")
    with pytest.raises(ValueError, match="Ambiguous"):
        target.load_gtfs_stops(str(feed))


def test_load_gtfs_stops_missing_path_raises() -> None:
    with pytest.raises(OSError, match="does not exist"):
        target.load_gtfs_stops("nope/never")


# ---------------------------------------------------------------------------
# filter_stops
# ---------------------------------------------------------------------------


def test_filter_stops_keeps_boarding_locations_with_key_and_coords() -> None:
    out = target.filter_stops(_stops_txt_df(), "stop_code")
    # i4 is a station, i5 has a blank code, i6 has no latitude.
    assert sorted(out["stop_key"]) == ["1001", "1002", "1003"]
    assert out["stop_lat"].dtype == float


def test_filter_stops_invalid_join_key_raises() -> None:
    with pytest.raises(ValueError, match="GTFS_JOIN_KEY"):
        target.filter_stops(_stops_txt_df(), "stop_name")


def test_filter_stops_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing column"):
        target.filter_stops(_stops_txt_df().drop(columns=["stop_lat"]), "stop_code")


def test_filter_stops_empty_result_raises() -> None:
    df = _stops_txt_df()
    df["stop_code"] = ""
    with pytest.raises(ValueError, match="No usable boarding stops"):
        target.filter_stops(df, "stop_code")


# ---------------------------------------------------------------------------
# spatial join
# ---------------------------------------------------------------------------


def test_join_assigns_interior_stops_to_their_district() -> None:
    stops = target.stops_to_gdf(target.filter_stops(_stops_txt_df(), "stop_code"))
    mapping = target.join_stops_to_districts(stops, _districts_gdf(), "DISTRICT")
    assert mapping["1001"] == {"District A"}
    assert mapping["1002"] == {"District B"}


def test_join_boundary_stop_lands_in_both_districts() -> None:
    stops = target.stops_to_gdf(target.filter_stops(_stops_txt_df(), "stop_code"))
    mapping = target.join_stops_to_districts(stops, _districts_gdf(), "DISTRICT")
    assert mapping["1003"] == {"District A", "District B"}


def test_join_zero_pairs_raises() -> None:
    far_away = pd.DataFrame({"stop_key": ["x"], "stop_lon": [10.0], "stop_lat": [10.0]})
    stops = target.stops_to_gdf(far_away)
    with pytest.raises(ValueError, match="zero stop-district pairs"):
        target.join_stops_to_districts(stops, _districts_gdf(), "DISTRICT")


def test_load_districts_missing_field_raises(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="WRONG_FIELD"):
        target.load_districts(paths["districts"], "WRONG_FIELD")


def test_load_districts_missing_crs_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_inputs(tmp_path)
    naked = _districts_gdf().set_crs(None, allow_override=True)
    monkeypatch.setattr(target.gpd, "read_file", lambda p: naked)
    with pytest.raises(ValueError, match="no CRS"):
        target.load_districts(paths["districts"], "DISTRICT")


# ---------------------------------------------------------------------------
# load_stop_ridership
# ---------------------------------------------------------------------------


def test_load_stop_ridership_collapses_and_cleans_ids(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    per_stop = target.load_stop_ridership(paths["excel"], routes_exclude=["DROPME"])
    assert per_stop["1001"] == {"boardings": 15.0, "alightings": 3.0, "total": 18.0}
    assert per_stop["1002"] == {"boardings": 24.0, "alightings": 7.0, "total": 31.0}
    assert "9999" in per_stop  # unmatched stop still loads


def test_load_stop_ridership_route_keep_filter(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    per_stop = target.load_stop_ridership(paths["excel"], routes=["R2"])
    assert set(per_stop) == {"1002"}
    assert per_stop["1002"]["boardings"] == 4.0


def test_load_stop_ridership_time_period_filter(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    per_stop = target.load_stop_ridership(
        paths["excel"], routes_exclude=["DROPME"], time_periods=["am peak"]
    )
    assert per_stop["1001"]["boardings"] == 10.0  # PM row excluded


def test_load_stop_ridership_unknown_time_period_raises(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="MIDNIGHT"):
        target.load_stop_ridership(paths["excel"], time_periods=["MIDNIGHT"])


def test_load_stop_ridership_missing_column_raises(tmp_path: Path) -> None:
    excel = tmp_path / "bad.xlsx"
    _ridership_df().drop(columns=["BOARD_ALL"]).to_excel(excel, index=False)
    with pytest.raises(ValueError, match="BOARD_ALL"):
        target.load_stop_ridership(str(excel))


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------


def _alloc_fixtures() -> tuple[dict[str, set[str]], dict[str, dict[str, float]], set[str]]:
    stop_to_districts = {
        "1001": {"District A"},
        "1002": {"District B"},
        "1003": {"District A", "District B"},
    }
    ridership = {
        "1001": {"boardings": 15.0, "alightings": 3.0, "total": 18.0},
        "1002": {"boardings": 24.0, "alightings": 7.0, "total": 31.0},
        "1003": {"boardings": 8.0, "alightings": 5.0, "total": 13.0},
        "9999": {"boardings": 7.0, "alightings": 6.0, "total": 13.0},
    }
    geocoded = {"1001", "1002", "1003", "1006"}
    return stop_to_districts, ridership, geocoded


def test_allocate_split_reconciles_exactly() -> None:
    mapping, ridership, geocoded = _alloc_fixtures()
    district_df, alloc_df, diag = target.allocate_ridership_to_districts(
        mapping, ridership, geocoded, boundary_allocation="split"
    )
    by_district = district_df.set_index("district")
    assert by_district.loc["District A", "boardings"] == 19.0  # 15 + 8/2
    assert by_district.loc["District B", "boardings"] == 28.0  # 24 + 8/2
    assert by_district["pct_boardings"].sum() == pytest.approx(100.0, abs=0.2)
    assert diag["matched"]["boardings"] == 47.0
    assert diag["boundary_extra"]["boardings"] == 0.0
    assert diag["unmatched_to_gtfs"]["boardings"] == 7.0
    # The boundary stop appears once per touching district in the detail.
    assert len(alloc_df[alloc_df["stop_key"] == "1003"]) == 2


def test_allocate_full_double_counts_boundary_stop() -> None:
    mapping, ridership, geocoded = _alloc_fixtures()
    district_df, _, diag = target.allocate_ridership_to_districts(
        mapping, ridership, geocoded, boundary_allocation="full"
    )
    by_district = district_df.set_index("district")
    assert by_district.loc["District A", "boardings"] == 23.0  # 15 + 8
    assert by_district.loc["District B", "boardings"] == 32.0  # 24 + 8
    assert diag["boundary_extra"]["boardings"] == 8.0
    # Percentages stay based on the once-counted total, so they sum above 100.
    assert by_district["pct_boardings"].sum() > 100.0


def test_allocate_geocoded_stop_outside_all_districts_is_diagnosed() -> None:
    mapping, ridership, geocoded = _alloc_fixtures()
    ridership["1006"] = {"boardings": 3.0, "alightings": 1.0, "total": 4.0}
    _, _, diag = target.allocate_ridership_to_districts(mapping, ridership, geocoded)
    assert diag["no_district"]["boardings"] == 3.0


def test_allocate_invalid_mode_raises() -> None:
    mapping, ridership, geocoded = _alloc_fixtures()
    with pytest.raises(ValueError, match="BOUNDARY_ALLOCATION"):
        target.allocate_ridership_to_districts(
            mapping, ridership, geocoded, boundary_allocation="fractional"
        )


def test_allocate_nothing_matched_raises() -> None:
    with pytest.raises(ValueError, match="join-key mismatch"):
        target.allocate_ridership_to_districts(
            {}, {"1": {"boardings": 5.0, "alightings": 0.0, "total": 5.0}}, set()
        )


def test_reconciliation_grand_total_excludes_double_count() -> None:
    mapping, ridership, geocoded = _alloc_fixtures()
    _, _, diag = target.allocate_ridership_to_districts(
        mapping, ridership, geocoded, boundary_allocation="full"
    )
    recon = target.build_reconciliation_frame(diag).set_index("category")
    assert recon.loc["boundary_double_count", "boardings"] == 8.0
    assert recon.loc["grand_total", "boardings"] == 54.0  # 47 allocated + 7 unmatched


# ---------------------------------------------------------------------------
# main / end-to-end
# ---------------------------------------------------------------------------


def test_main_placeholders_exit_2() -> None:
    assert target.main([]) == 2


def test_main_missing_input_exits_1(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    argv = _cli(paths)
    argv[argv.index(paths["gtfs"])] = str(tmp_path / "nope")
    assert target.main(argv) == 1


def test_main_end_to_end_writes_workbook_and_runlog(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    assert target.main(_cli(paths)) == 0

    xlsx = Path(paths["out"]) / "district_ridership_share.xlsx"
    summary = pd.read_excel(xlsx, sheet_name="district_ridership").set_index("district")
    recon = pd.read_excel(xlsx, sheet_name="reconciliation").set_index("category")
    assert summary.loc["District A", "boardings"] == 19.0
    assert summary.loc["District B", "boardings"] == 28.0
    assert recon.loc["allocated_to_districts", "boardings"] == 47.0
    assert recon.loc["unmatched_to_gtfs", "boardings"] == 7.0
    assert recon.loc["grand_total", "boardings"] == 54.0

    runlog = (Path(paths["out"]) / "district_ridership_share_runlog.txt").read_text()
    assert 'BOUNDARY_ALLOCATION: str = "split"' in runlog  # verbatim config block
    assert "EFFECTIVE SETTINGS" in runlog
    assert "sha256:" in runlog


def test_main_end_to_end_full_mode(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    assert target.main(_cli(paths, "--boundary-allocation", "full")) == 0

    xlsx = Path(paths["out"]) / "district_ridership_share.xlsx"
    recon = pd.read_excel(xlsx, sheet_name="reconciliation").set_index("category")
    assert recon.loc["boundary_double_count", "boardings"] == 8.0


def test_stops_to_gdf_is_wgs84_points() -> None:
    gdf = target.stops_to_gdf(target.filter_stops(_stops_txt_df(), "stop_code"))
    assert gdf.crs.to_epsg() == 4326
    assert isinstance(gdf.geometry.iloc[0], Point)
