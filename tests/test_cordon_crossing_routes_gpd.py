from __future__ import annotations

import logging
import math
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for savefig

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

import scripts.service_coverage.cordon_crossing_routes_gpd as target

FIXTURES = Path(__file__).parent / "fixtures"

# Geometry tests work in plain feet (as in EPSG:2248) with a 500 ft threshold.
THRESHOLD = 500.0
KW = {"step": 25.0, "delta": 0.5, "tol": 0.01}

# A straight north-south screenline 10,000 ft long, centered on the origin.
SCREENLINE = LineString([(0, -5000), (0, 5000)])
SQUARE = Polygon([(0, 0), (4000, 0), (4000, 4000), (0, 4000)])


def _assess(path: list[tuple[float, float]], cordon=SCREENLINE) -> target.PathResult:
    return target.assess_path(LineString(path), cordon, THRESHOLD, **KW)


# ---------------------------------------------------------------------------
# assess_path — screenlines
# ---------------------------------------------------------------------------


def test_clean_crossing_counts_once_with_full_margin() -> None:
    result = _assess([(-3000, 0), (3000, 0)])
    assert result.status == target.STATUS_CROSSES
    assert result.n_crossings == 1
    assert result.margin == pytest.approx(3000, abs=1)
    assert result.margin_reason == "depth"
    assert len(result.crossing_xy) == 1
    assert result.crossing_xy[0] == pytest.approx((0.0, 0.0))


def test_shallow_dip_past_the_line_is_marginal() -> None:
    # Runs 1,000 ft west of the line, pokes 100 ft east of it, and comes back.
    result = _assess([(-1000, -3000), (-1000, -100), (100, 0), (-1000, 100), (-1000, 3000)])
    assert result.status == target.STATUS_MARGINAL
    assert result.n_crossings == 2
    assert result.margin == pytest.approx(100, abs=1)
    assert result.margin_reason == "depth"


def test_crossing_near_the_end_of_a_screenline_is_marginal() -> None:
    result = _assess([(-3000, 4900), (3000, 4900)])
    assert result.status == target.STATUS_MARGINAL
    assert result.margin == pytest.approx(100, abs=1)
    assert result.margin_reason == "cordon_end"


def test_parallel_route_within_threshold_is_a_near_miss() -> None:
    result = _assess([(300, -3000), (300, 3000)])
    assert result.status == target.STATUS_NEAR_MISS
    assert result.n_crossings == 0
    assert result.min_distance == pytest.approx(300)
    (px, _), (cx, _) = result.nearest_xy
    assert px == pytest.approx(300)
    assert cx == pytest.approx(0)


def test_far_route_has_no_status() -> None:
    assert _assess([(2000, -3000), (2000, 3000)]).status == target.STATUS_NONE


def test_route_ending_on_the_line_does_not_cross() -> None:
    result = _assess([(-3000, 0), (0, 0)])
    assert result.n_crossings == 0
    assert result.status == target.STATUS_NEAR_MISS
    assert result.min_distance == 0


def test_tangent_touch_does_not_cross() -> None:
    # A V whose point touches the line from the west.
    result = _assess([(-1000, -1000), (0, 0), (-1000, 1000)])
    assert result.n_crossings == 0
    assert result.status == target.STATUS_NEAR_MISS


def test_out_and_back_on_the_same_street_crosses_twice() -> None:
    result = _assess([(-3000, 0), (3000, 0), (-3000, 0)])
    assert result.status == target.STATUS_CROSSES
    assert result.n_crossings == 2


def test_running_along_the_line_then_crossing_counts_one_crossing() -> None:
    result = _assess([(-2000, -3000), (0, -1000), (0, 1000), (2000, 3000)])
    assert result.status == target.STATUS_CROSSES
    assert result.n_crossings == 1
    assert result.overlap == pytest.approx(2000)


def test_running_along_the_line_and_returning_is_a_near_miss() -> None:
    result = _assess([(-2000, -3000), (0, -1000), (0, 1000), (-2000, 3000)])
    assert result.n_crossings == 0
    assert result.status == target.STATUS_NEAR_MISS
    assert result.overlap == pytest.approx(2000)


def test_zero_threshold_turns_flags_off() -> None:
    dip = LineString([(-1000, -3000), (-1000, -100), (100, 0), (-1000, 100), (-1000, 3000)])
    assert target.assess_path(dip, SCREENLINE, 0.0, **KW).status == target.STATUS_CROSSES
    parallel = LineString([(300, -3000), (300, 3000)])
    assert target.assess_path(parallel, SCREENLINE, 0.0, **KW).status == target.STATUS_NONE


# ---------------------------------------------------------------------------
# assess_path / assess_cordon — polygon cordons
# ---------------------------------------------------------------------------


def test_route_through_polygon_crosses_its_outline_twice() -> None:
    result = _assess([(-2000, 2000), (6000, 2000)], cordon=SQUARE)
    assert result.status == target.STATUS_CROSSES
    assert result.n_crossings == 2
    assert result.margin == pytest.approx(2000, abs=1)


def test_route_clipping_a_polygon_corner_is_marginal() -> None:
    # Cuts the lower-right corner, never more than ~71 ft inside.
    result = _assess([(3700, -200), (4300, 400)], cordon=SQUARE)
    assert result.status == target.STATUS_MARGINAL
    assert result.n_crossings == 2
    assert result.margin < 100


def test_route_inside_polygon_near_its_edge_is_a_near_miss() -> None:
    result = _assess([(100, 1000), (100, 3000)], cordon=SQUARE)
    assert result.status == target.STATUS_NEAR_MISS
    assert result.inside


def test_route_deep_inside_polygon_is_inside() -> None:
    result = _assess([(1500, 2000), (2500, 2000)], cordon=SQUARE)
    assert result.status == target.STATUS_INSIDE
    assert result.inside


def test_assess_cordon_skips_far_paths_and_keeps_inside_ones() -> None:
    paths = gpd.GeoSeries(
        [
            LineString([(-2000, 2000), (6000, 2000)]),
            LineString([(1500, 2000), (2500, 2000)]),
            LineString([(9000, 0), (9000, 4000)]),
        ],
        index=["through", "inside", "far"],
    )
    results = target.assess_cordon(SQUARE, paths, THRESHOLD, unit_to_ft=1.0)
    assert set(results) == {"through", "inside"}
    assert results["through"].status == target.STATUS_CROSSES
    assert results["inside"].status == target.STATUS_INSIDE


# ---------------------------------------------------------------------------
# cordon layer
# ---------------------------------------------------------------------------


def test_cordon_names_detected_numbered_and_deduplicated() -> None:
    layer = gpd.GeoDataFrame(
        {"NAME": ["Bridge", None, "bridge"]}, geometry=[SCREENLINE] * 3, crs="EPSG:2248"
    )
    assert target.resolve_cordon_names(layer, None) == ["Bridge", "cordon_2", "bridge_2"]


def test_cordon_names_numbered_when_no_name_field() -> None:
    layer = gpd.GeoDataFrame({"id": [7, 8]}, geometry=[SCREENLINE] * 2, crs="EPSG:2248")
    assert target.resolve_cordon_names(layer, None) == ["cordon_1", "cordon_2"]


def test_missing_cordon_name_field_raises() -> None:
    layer = gpd.GeoDataFrame({"NAME": ["A"]}, geometry=[SCREENLINE], crs="EPSG:2248")
    with pytest.raises(ValueError, match="CORDON_NAME_FIELD"):
        target.resolve_cordon_names(layer, "label")


def test_load_cordons_rejects_layer_without_crs(tmp_path: Path) -> None:
    path = tmp_path / "no_crs.shp"
    gpd.GeoDataFrame({"NAME": ["A"]}, geometry=[SCREENLINE]).to_file(path)
    with pytest.raises(ValueError, match="no coordinate reference system"):
        target.load_cordons(path, target.CRS.from_epsg(2248))


def test_load_cordons_skips_points_and_types_features(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "cordons.gpkg"
    gpd.GeoDataFrame(
        {"name": ["Line", "Area", "Dot"]},
        geometry=[SCREENLINE, SQUARE, Point(0, 0)],
        crs="EPSG:2248",
    ).to_file(path)
    with caplog.at_level(logging.WARNING):
        cordons = target.load_cordons(path, target.CRS.from_epsg(2248))
    assert list(cordons["cordon"]) == ["Line", "Area"]
    assert list(cordons["cordon_type"]) == ["line", "polygon"]
    assert "Dot" in caplog.text


def test_load_cordons_repairs_self_crossing_polygon(tmp_path: Path) -> None:
    path = tmp_path / "bowtie.gpkg"
    bowtie = Polygon([(0, 0), (2000, 2000), (2000, 0), (0, 2000)])
    gpd.GeoDataFrame({"name": ["Bowtie"]}, geometry=[bowtie], crs="EPSG:2248").to_file(path)
    cordons = target.load_cordons(path, target.CRS.from_epsg(2248))
    assert cordons.geometry.iloc[0].is_valid
    assert cordons["cordon_type"].iloc[0] == "polygon"


# ---------------------------------------------------------------------------
# run() on a small synthetic feed
# ---------------------------------------------------------------------------

HUB = (-77.015, 38.895)


def _offsets_to_lonlat(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convert feet east/north of HUB (EPSG:2248) to lon/lat."""
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:2248", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:2248", "EPSG:4326", always_xy=True)
    hx, hy = to_ft.transform(*HUB)
    return [to_ll.transform(hx + dx, hy + dy) for dx, dy in points]


def _write_feed(gtfs_dir: Path) -> None:
    """Write a feed where route 1 crosses the test screenline.

    Route 2 runs 300 ft east of the line, and route 3 (no shape) crosses it stop to stop.
    """
    gtfs_dir.mkdir()
    (gtfs_dir / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name,route_type\n"
        "R1,1,Crosstown,3\nR2,2,Parallel,3\nR3,3,No Shape,3\n"
    )
    (gtfs_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id,shape_id\n"
        "R1,wk,T1,0,SH1\nR1,wk,T1b,1,SH1\nR2,wk,T2,0,SH2\nR3,wk,T3,0,\n"
    )
    shape_rows = ["shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence"]
    for shape_id, pts in (
        ("SH1", [(-3000, 0), (3000, 0)]),
        ("SH2", [(300, -3000), (300, 3000)]),
    ):
        for seq, (lon, lat) in enumerate(_offsets_to_lonlat(pts), start=1):
            shape_rows.append(f"{shape_id},{lat:.7f},{lon:.7f},{seq}")
    (gtfs_dir / "shapes.txt").write_text("\n".join(shape_rows) + "\n")
    (s1, s2) = _offsets_to_lonlat([(-2000, 2000), (2000, 2000)])
    (gtfs_dir / "stops.txt").write_text(
        f"stop_id,stop_name,stop_lat,stop_lon\nS1,West,{s1[1]:.7f},{s1[0]:.7f}\n"
        f"S2,East,{s2[1]:.7f},{s2[0]:.7f}\n"
    )
    (gtfs_dir / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T3,08:00:00,08:00:00,S1,1\nT3,08:10:00,08:10:00,S2,2\n"
    )


def _write_zipped_screenline(tmp_path: Path) -> Path:
    """Write the 10,000 ft north-south screenline through HUB as a zipped shapefile."""
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    line = LineString(_offsets_to_lonlat([(0, -5000), (0, 5000)]))
    gpd.GeoDataFrame({"NAME": ["Test line"]}, geometry=[line], crs="EPSG:4326").to_file(
        shp_dir / "screenline.shp"
    )
    zip_path = tmp_path / "screenline.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for part in shp_dir.iterdir():
            archive.write(part, part.name)
    return zip_path


def test_run_writes_tables_maps_and_run_log(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    _write_feed(gtfs_dir)
    cordons = _write_zipped_screenline(tmp_path)
    out = tmp_path / "out"

    routes = target.run(
        gtfs_path=gtfs_dir, cordons_path=cordons, output_dir=out, write_maps=True, dpi=40
    )

    by_route = routes.set_index("route")
    assert by_route.loc["1", "status"] == target.STATUS_CROSSES
    assert by_route.loc["1", "trips_crossing"] == 2
    assert by_route.loc["1", "directions_crossing"] == "0; 1"
    assert by_route.loc["2", "status"] == target.STATUS_NEAR_MISS
    assert by_route.loc["2", "closest_approach_ft"] == pytest.approx(300, abs=2)
    assert re.search(r"within 30[01] ft", by_route.loc["2", "note"])
    assert by_route.loc["3", "status"] == target.STATUS_CROSSES

    summary = pd.read_csv(out / target.SUMMARY_FILENAME, dtype=str, keep_default_na=False)
    assert summary.loc[0, "routes_crossing"] == "1; 3"
    assert summary.loc[0, "routes_near_miss"] == "2"
    shapes = pd.read_csv(out / target.SHAPES_FILENAME, dtype=str, keep_default_na=False)
    assert not set(target.INTERNAL_COLUMNS) & set(shapes.columns)
    assert set(shapes["geometry_source"]) == {target.SOURCE_SHAPES, target.SOURCE_STOPS}
    lat, lon = map(float, shapes.set_index("route").loc["1", "crossing_points"].split())
    assert lon == pytest.approx(HUB[0], abs=1e-4)
    assert lat == pytest.approx(HUB[1], abs=1e-4)

    png = out / target.MAPS_SUBDIR / "cordon_Test_line.png"
    assert png.exists() and png.stat().st_size > 0
    run_log = (out / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert target.CONFIG_BEGIN_MARKER in run_log
    assert "NEAR_MISS_FT" in run_log


def test_run_with_service_filter_uses_only_those_trips(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    _write_feed(gtfs_dir)
    cordons = _write_zipped_screenline(tmp_path)
    routes = target.run(
        gtfs_path=gtfs_dir,
        cordons_path=cordons,
        output_dir=tmp_path / "out",
        service_ids=["wk"],
        write_maps=False,
    )
    assert routes["trips_total"].sum() == 4


def test_mock_dc_cbd_fixture_matches_its_documented_layout(tmp_path: Path) -> None:
    """Check the layout documented in generate_mock_cbd.py.

    Routes 10, 20, 30 and 50H run through downtown; route 40 stays 1,500 ft from it and
    cuts across the annex's southeast corner.
    """
    routes = target.run(
        gtfs_path=FIXTURES / "mock_gtfs_dc.zip",
        cordons_path=FIXTURES / "mock_cbd_dc.zip",
        output_dir=tmp_path / "out",
        write_maps=False,
    )
    downtown = routes[routes["cordon"] == "Mock Central Business District"]
    assert set(downtown["route"]) == {"10", "20", "30", "50H"}
    assert set(downtown["status"]) == {target.STATUS_CROSSES}
    annex = routes[routes["cordon"] == "Mock Central Business District Annex"].set_index("route")
    assert annex.loc["40", "status"] == target.STATUS_MARGINAL

    wider = target.run(
        gtfs_path=FIXTURES / "mock_gtfs_dc.zip",
        cordons_path=FIXTURES / "mock_cbd_dc.zip",
        output_dir=tmp_path / "out_wide",
        near_miss_ft=2000,
        write_maps=False,
    )
    route_40 = wider[
        (wider["cordon"] == "Mock Central Business District") & (wider["route"] == "40")
    ]
    assert route_40["status"].tolist() == [target.STATUS_NEAR_MISS]
    assert math.isclose(route_40["closest_approach_ft"].iloc[0], 1500, abs_tol=5)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_blocks_unedited_placeholder_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(target, "run", lambda **kw: calls.append(kw))
    assert target.main([]) == 2
    assert calls == []


def test_main_unknown_service_id_exits_2(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    _write_feed(gtfs_dir)
    cordons = _write_zipped_screenline(tmp_path)
    argv = [
        "--gtfs-path",
        str(gtfs_dir),
        "--cordons-path",
        str(cordons),
        "--output-dir",
        str(tmp_path / "out"),
        "--service-ids",
        "saturday",
        "--no-write-maps",
    ]
    assert target.main(argv) == 2


def test_run_without_any_usable_geometry_raises(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    _write_feed(gtfs_dir)
    (gtfs_dir / "trips.txt").write_text("route_id,service_id,trip_id,shape_id\nR1,wk,T9,NOPE\n")
    cordons = _write_zipped_screenline(tmp_path)
    with pytest.raises(ValueError, match="usable geometry"):
        target.run(gtfs_path=gtfs_dir, cordons_path=cordons, output_dir=tmp_path / "out")


def test_main_missing_cordon_layer_exits_1(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    _write_feed(gtfs_dir)
    argv = [
        "--gtfs-path",
        str(gtfs_dir),
        "--cordons-path",
        str(tmp_path / "missing.shp"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert target.main(argv) == 1
