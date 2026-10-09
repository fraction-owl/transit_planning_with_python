from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

import scripts.facilities_tools.walking_network_qa_gpd as target

# A street scene in Maryland State Plane feet (EPSG:2248) inside Washington, DC.
X0, Y0 = 1_300_000.0, 450_000.0
CRS_FT = "EPSG:2248"
TO_WGS84 = Transformer.from_crs(CRS_FT, "EPSG:4326", always_xy=True)

# A 40 ft road runs east-west; sidewalks sit 30 ft either side of its centerline.
SIDEWALKS = {
    # North sidewalk, west piece: interior vertices where the stub and path meet it.
    "N1": [(X0 - 1500, Y0 + 30), (X0 - 500, Y0 + 30), (X0, Y0 + 30), (X0 + 200, Y0 + 30)],
    # North sidewalk, east piece: starts 0.5 ft past N1's end (a tiny gap).
    "N2": [(X0 + 200.5, Y0 + 30), (X0 + 1500, Y0 + 30)],
    # South sidewalk runs past the context area (component touches the context edge).
    "S1": [(X0 - 3000, Y0 - 30), (X0 + 1500, Y0 - 30)],
    # North path shares N1's vertex; south path ends on S1 without a shared vertex.
    "P1": [(X0, Y0 + 30), (X0, Y0 + 400)],
    "P2": [(X0, Y0 - 30), (X0, Y0 - 400)],
    # Curb-ramp stub ending 2 ft from the road edge, 52 ft across from S1.
    "R1": [(X0 - 500, Y0 + 30), (X0 - 500, Y0 + 22)],
    # A 50 ft isolated piece.
    "I1": [(X0 + 600, Y0 + 300), (X0 + 650, Y0 + 300)],
}
ROAD = Polygon(
    [(X0 - 4000, Y0 - 20), (X0 + 4000, Y0 - 20), (X0 + 4000, Y0 + 20), (X0 - 4000, Y0 + 20)]
)


def _lonlat(x: float, y: float) -> tuple[float, float]:
    lon, lat = TO_WGS84.transform(x, y)
    return float(lon), float(lat)


def _write_gtfs(folder: Path) -> Path:
    """Two platform stops (one with the literal ID "NA") and a parent station."""
    folder.mkdir(parents=True, exist_ok=True)
    near_lon, near_lat = _lonlat(X0 + 50, Y0 + 35)
    far_lon, far_lat = _lonlat(X0 + 900, Y0 + 900)
    pd.DataFrame(
        {
            "stop_id": ["A", "NA", "STATION"],
            "stop_name": ["Near stop", "Far stop", "Station"],
            "stop_lat": [near_lat, far_lat, near_lat],
            "stop_lon": [near_lon, far_lon, near_lon],
            "location_type": ["0", "", "1"],
        }
    ).to_csv(folder / "stops.txt", index=False)
    pd.DataFrame({"route_id": ["r1", "r2"], "route_short_name": ["10", "20"]}).to_csv(
        folder / "routes.txt", index=False
    )
    pd.DataFrame({"trip_id": ["t1", "t2"], "route_id": ["r1", "r2"]}).to_csv(
        folder / "trips.txt", index=False
    )
    pd.DataFrame({"trip_id": ["t1", "t2"], "stop_id": ["A", "NA"], "stop_sequence": [1, 1]}).to_csv(
        folder / "stop_times.txt", index=False
    )
    return folder


def _write_scene(tmp_path: Path, extra: dict[str, list[tuple[float, float]]] | None = None) -> dict:
    lines = {**SIDEWALKS, **(extra or {})}
    sidewalks = gpd.GeoDataFrame(
        {"SW_ID": list(lines), "LEVEL": [0] * len(lines)},
        geometry=[LineString(coords) for coords in lines.values()],
        crs=CRS_FT,
    )
    sidewalk_path = tmp_path / "sidewalks.gpkg"
    sidewalks.to_file(sidewalk_path, layer="sidewalks", driver="GPKG")
    # Roads are stored in WGS84 so the run exercises reprojection.
    road_path = tmp_path / "roads.gpkg"
    gpd.GeoDataFrame({"NAME": ["Main St"]}, geometry=[ROAD], crs=CRS_FT).to_crs(
        "EPSG:4326"
    ).to_file(road_path, driver="GPKG")
    return {
        "sidewalks": sidewalk_path,
        "roads": road_path,
        "gtfs": _write_gtfs(tmp_path / "gtfs"),
        "out": tmp_path / "out",
    }


def _args(scene: dict, *extra: str) -> list[str]:
    return [
        "--network",
        str(scene["sidewalks"]),
        "--gtfs",
        str(scene["gtfs"]),
        "--roads",
        str(scene["roads"]),
        "--output-dir",
        str(scene["out"]),
        *extra,
    ]


def _issues(scene: dict) -> pd.DataFrame:
    return pd.read_csv(scene["out"] / target.ISSUES_FILENAME, dtype={"ISSUE_ID": str})


def _settings(**overrides: object) -> target.Settings:
    base = target.Settings(network_sources=(), gtfs_path="", output_dir=Path("."))
    return base._replace(**overrides)


def _network(
    lines: list[list[tuple[float, float]]], levels: list[str], policy: str
) -> target.Network:
    parts = [
        target.Part(i, f"s:{i}", 0, tuple(points), level)
        for i, (points, level) in enumerate(zip(lines, levels))
    ]
    return target.Network(parts, 0.001, policy)


# --------------------------------------------------------------------------
# Pure geometry helpers
# --------------------------------------------------------------------------


def test_segment_intersection_classifies_contacts() -> None:
    crossing = target.segment_intersection((0, 0), (10, 0), (5, -5), (5, 5), 1e-6)
    assert crossing == ("POINT", [(5.0, 0.0)])
    overlap = target.segment_intersection((0, 0), (10, 0), (4, 0), (20, 0), 1e-6)
    assert overlap == ("OVERLAP", [(4.0, 0.0), (10.0, 0.0)])
    assert target.segment_intersection((0, 0), (10, 0), (0, 1), (10, 1), 1e-6)[0] == "NONE"
    assert target.segment_intersection((0, 0), (10, 0), (11, -1), (11, 1), 1e-6)[0] == "NONE"


def test_check_line_geometry_flags_problem_shapes() -> None:
    assert target.check_line_geometry(LineString([(0, 0), (1, 1)])) == []
    assert target.check_line_geometry(LineString([(0, 0), (0, 0), (1, 1)])) == ["DUPLICATE_VERTEX"]
    assert target.check_line_geometry(Point(0, 0)) == ["NOT_A_LINE: Point"]
    assert target.check_line_geometry(None) == ["NULL_GEOMETRY"]


def test_normalize_level_treats_numeric_zero_alike() -> None:
    assert target.normalize_level(0) == target.normalize_level(0.0) == "0"
    assert target.normalize_level(" B1 ") == "B1"
    assert target.normalize_level(float("nan")) == ""
    assert target.normalize_level(None) == ""


# --------------------------------------------------------------------------
# Network connectivity rules
# --------------------------------------------------------------------------


def test_t_junction_connects_only_under_any_vertex_policy() -> None:
    lines = [[(0, 0), (5, 0), (10, 0)], [(5, 0), (5, 8)]]
    any_vertex = _network(lines, ["", ""], "ANY_VERTEX")
    assert any_vertex.component_count() == 1
    assert target.scan_intersections(any_vertex, 1.0) == []

    endpoint = _network(lines, ["", ""], "ENDPOINT")
    assert endpoint.component_count() == 2
    kinds = [i.kind for i in target.scan_intersections(endpoint, 1.0)]
    assert kinds == ["JUNCTION_POLICY_MISMATCH"]


def test_different_levels_do_not_connect() -> None:
    network = _network([[(0, 0), (10, 0)], [(10, 0), (20, 0)]], ["0", "1"], "ANY_VERTEX")
    assert network.component_count() == 2
    assert len(network.dangles()) == 4
    [issue] = target.scan_intersections(network, 1.0)
    assert issue.kind == "JUNCTION_LEVEL_MISMATCH"
    assert issue.confidence == "LIKELY_INTENTIONAL"


def test_crossing_without_shared_vertex_is_reported_not_joined() -> None:
    network = _network([[(0, 0), (10, 0)], [(5, -5), (5, 5)]], ["", ""], "ANY_VERTEX")
    assert network.component_count() == 2
    assert [i.kind for i in target.scan_intersections(network, 1.0)] == [
        "CROSSING_NO_SHARED_VERTEX"
    ]


def test_gap_scan_flags_near_endpoint_gap() -> None:
    # Lines long enough that their far ends sit well beyond the 10 ft review distance.
    network = _network([[(0, 0), (30, 0)], [(30.5, 0), (60, 0)]], ["", ""], "ANY_VERTEX")
    roads = target.RoadContext([], [], 1.0)
    issues = target.scan_gaps(network, roads, 1.0, _settings())
    gaps = [i for i in issues if i.kind == "NEAR_ENDPOINT_GAP"]
    assert len(gaps) == 1
    assert gaps[0].value_ft == pytest.approx(0.5)


# --------------------------------------------------------------------------
# GTFS stop selection
# --------------------------------------------------------------------------


def test_select_gtfs_stops_keeps_literal_na_and_skips_stations(tmp_path: Path) -> None:
    gtfs = _write_gtfs(tmp_path / "gtfs")
    stops = target.select_gtfs_stops(_settings(gtfs_path=str(gtfs)))
    assert stops["stop_id"].tolist() == ["A", "NA"]

    by_route = target.select_gtfs_stops(
        _settings(gtfs_path=str(gtfs), route_short_names=frozenset({"20"}))
    )
    assert by_route["stop_id"].tolist() == ["NA"]

    with pytest.raises(ValueError, match="absent from GTFS"):
        target.select_gtfs_stops(
            _settings(gtfs_path=str(gtfs), route_short_names=frozenset({"99"}))
        )


def test_select_gtfs_stops_rejects_bad_coordinates(tmp_path: Path) -> None:
    gtfs = _write_gtfs(tmp_path / "gtfs")
    stops = pd.read_csv(gtfs / "stops.txt", dtype=str, keep_default_na=False)
    stops.loc[0, "stop_lat"] = "not a number"
    stops.to_csv(gtfs / "stops.txt", index=False)
    with pytest.raises(ValueError, match="stop A"):
        target.select_gtfs_stops(_settings(gtfs_path=str(gtfs)))


# --------------------------------------------------------------------------
# End-to-end runs
# --------------------------------------------------------------------------


def test_main_returns_2_for_placeholder_paths() -> None:
    assert target.main([]) == 2


def test_main_returns_2_for_inconsistent_thresholds(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path)
    # The approved-snap limit (1 ft) may not exceed the gap review distance.
    assert target.main(_args(scene, "--gap-review-ft", "0.5")) == 2


def test_main_writes_review_outputs(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path)
    assert target.main(_args(scene)) == 0
    out = scene["out"]

    layers = set(pyogrio.list_layers(out / target.REVIEW_GPKG_FILENAME)[:, 0])
    assert {
        "gtfs_stops",
        "stop_buffers",
        "study_area",
        "context_area",
        "road_context",
        "network_original",
        "network_working",
        "network_components",
        "issues_points",
        "issues_lines",
        "network_clip_0p25",
    } <= layers

    kinds = set(_issues(scene)["ISSUE_TYPE"])
    assert {
        "CROSSING_NO_SHARED_VERTEX",
        "POSSIBLE_MISSING_CROSSING",
        "END_NEAR_ROAD",
        "NEAR_ENDPOINT_GAP",
        "SMALL_DISCONNECTED_GROUP",
        "STOP_FAR_FROM_NETWORK",
    } <= kinds

    stops = pd.read_csv(out / target.STOP_SUMMARY_FILENAME, dtype={"STOP_ID": str}, na_filter=False)
    assert stops["STOP_ID"].tolist() == ["A", "NA"]
    assert float(stops.loc[0, "NEAREST_LINE_FT"]) == pytest.approx(5.0, abs=0.01)

    components = pd.read_csv(out / target.COMPONENTS_FILENAME)
    assert components["CONTEXT_EDGE"].sum() >= 1  # S1 runs past the context area.

    candidates = pd.read_csv(out / target.REPAIR_CANDIDATES_FILENAME, dtype=str)
    assert len(candidates) == 1
    assert {candidates.loc[0, "SOURCE_KEY"], candidates.loc[0, "TARGET_KEY"]} == {
        "sidewalks:1",
        "sidewalks:2",
    }

    summary = json.loads((out / target.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["stops"] == 2
    assert summary["selected_features"] == len(SIDEWALKS)
    assert summary["road_polygons_in_context"] == 1

    run_log = (out / "walking_network_qa_runlog.txt").read_text(encoding="utf-8")
    assert "CONFIGURATION (verbatim from source)" in run_log
    assert "NETWORK_SOURCES" in run_log
    assert "# === BEGIN CONFIG ===" not in run_log


def test_approved_snap_closes_gap_and_stale_approval_fails(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path)
    assert target.main(_args(scene)) == 0
    candidates = pd.read_csv(
        scene["out"] / target.REPAIR_CANDIDATES_FILENAME, dtype=str, keep_default_na=False
    )
    candidates["APPROVE"] = "YES"
    approved = tmp_path / "approved_snaps.csv"
    candidates.to_csv(approved, index=False)

    assert target.main(_args(scene, "--approved-snaps", str(approved))) == 0
    log = pd.read_csv(scene["out"] / target.REPAIR_LOG_FILENAME)
    assert log["ACTION"].tolist() == ["APPROVED_ENDPOINT_SNAP"]
    # EPSG:2248 is in US survey feet; outputs are international feet.
    assert float(log.loc[0, "MOVE_FT"]) == pytest.approx(0.5, abs=1e-4)
    assert "NEAR_ENDPOINT_GAP" not in set(_issues(scene)["ISSUE_TYPE"])
    working = gpd.read_file(scene["out"] / target.REVIEW_GPKG_FILENAME, layer="network_working")
    original = gpd.read_file(scene["out"] / target.REVIEW_GPKG_FILENAME, layer="network_original")
    assert (working.geometry != original.geometry).sum() == 1

    stale = candidates.copy()
    stale["SOURCE_HASH"] = "0" * 64
    stale.to_csv(approved, index=False)
    assert target.main(_args(scene, "--approved-snaps", str(approved))) == 1


def test_duplicate_vertex_is_excluded_unless_cleanup_is_on(tmp_path: Path) -> None:
    extra = {"D1": [(X0 + 300, Y0 + 500), (X0 + 300, Y0 + 500), (X0 + 400, Y0 + 500)]}
    scene = _write_scene(tmp_path, extra)
    assert target.main(_args(scene)) == 0
    issues = _issues(scene)
    excluded = issues.loc[issues["ISSUE_TYPE"] == "EXCLUDED_GEOMETRY", "DETAIL"].tolist()
    assert excluded == ["GEOMETRY_CHECK: DUPLICATE_VERTEX"]

    assert target.main(_args(scene, "--remove-duplicate-vertices")) == 0
    assert "EXCLUDED_GEOMETRY" not in set(_issues(scene)["ISSUE_TYPE"])
    log = pd.read_csv(scene["out"] / target.REPAIR_LOG_FILENAME)
    assert log["ACTION"].tolist() == ["REMOVE_REPEATED_VERTICES"]
