from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

import scripts.gtfs_data_quality.route_direction_classifier as mod
from scripts.gtfs_data_quality.route_direction_classifier import (
    add_route_labels,
    classify_direction,
    create_lines_from_shapes,
    determine_dominant_shapes,
    export_jpegs,
    flag_suspicious_data,
    identify_first_last_stops,
    split_trip_exceptions,
)

# ---------------------------------------------------------------------------
# classify_direction
# ---------------------------------------------------------------------------

# Helper: a projected line with endpoints far apart (> default loop_threshold=200)
_FAR = 500_000.0


def test_classify_direction_northbound() -> None:
    line_proj = LineString([(0.0, 0.0), (0.0, _FAR)])
    assert classify_direction(line_proj) == "NB"


def test_classify_direction_southbound() -> None:
    line_proj = LineString([(0.0, _FAR), (0.0, 0.0)])
    assert classify_direction(line_proj) == "SB"


def test_classify_direction_eastbound() -> None:
    line_proj = LineString([(0.0, 0.0), (_FAR, 0.0)])
    assert classify_direction(line_proj) == "EB"


def test_classify_direction_westbound() -> None:
    line_proj = LineString([(_FAR, 0.0), (0.0, 0.0)])
    assert classify_direction(line_proj) == "WB"


def test_classify_direction_counter_clockwise_loop() -> None:
    # start and end are within loop_threshold distance in projected coords
    # CCW square: (0,0)→(100,0)→(100,100)→(0,100); end dist to start = 100 < 200
    line_proj = LineString([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert classify_direction(line_proj) == "CCW"


def test_classify_direction_clockwise_loop() -> None:
    # CW square: (0,0)→(0,100)→(100,100)→(100,0); end dist to start = 100 < 200
    line_proj = LineString([(0, 0), (0, 100), (100, 100), (100, 0)])
    assert classify_direction(line_proj) == "CW"


def test_classify_direction_custom_loop_threshold() -> None:
    # end is 150 units from start; with threshold=200 it's a loop, threshold=100 it's not
    line_proj = LineString([(0, 0), (150, 0)])
    assert classify_direction(line_proj, loop_threshold=200) != classify_direction(
        line_proj, loop_threshold=100
    )


def test_classify_direction_compares_ground_distance_not_degrees() -> None:
    # ~5.0 km north and ~4.3 km east near 38.9 N: 0.045 deg of latitude but 0.05 deg
    # of longitude, so a degree comparison would call it EB.
    line = gpd.GeoSeries([LineString([(-77.0, 38.9), (-76.95, 38.945)])], crs="EPSG:4326").to_crs(
        mod.PROJECTED_CRS
    )
    assert classify_direction(line.iloc[0]) == "NB"


# ---------------------------------------------------------------------------
# create_lines_from_shapes
# ---------------------------------------------------------------------------


@pytest.fixture()
def shapes_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shape_id": ["S1", "S1", "S2", "S2"],
            "shape_pt_sequence": ["1", "2", "1", "2"],
            "shape_pt_lat": ["38.7", "38.8", "38.9", "39.0"],
            "shape_pt_lon": ["-77.0", "-77.0", "-77.1", "-77.1"],
        }
    )


def test_create_lines_from_shapes_returns_geodataframe(shapes_df: pd.DataFrame) -> None:
    import geopandas as gpd

    gdf = create_lines_from_shapes(shapes_df)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_create_lines_from_shapes_correct_row_count(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert len(gdf) == 2


def test_create_lines_from_shapes_geometry_is_linestring(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert all(isinstance(g, LineString) for g in gdf.geometry)


def test_create_lines_from_shapes_has_shape_id_column(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert "shape_id" in gdf.columns


def test_create_lines_from_shapes_crs_is_wgs84(shapes_df: pd.DataFrame) -> None:
    gdf = create_lines_from_shapes(shapes_df)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_create_lines_from_shapes_orders_points_numerically() -> None:
    # Read as text, sequences 1, 10, 2 would put the last point second.
    shapes = pd.DataFrame(
        {
            "shape_id": ["S1"] * 3,
            "shape_pt_sequence": [10, 1, 2],
            "shape_pt_lat": [39.0, 38.0, 38.5],
            "shape_pt_lon": [-77.0, -77.0, -77.0],
        }
    )
    line = create_lines_from_shapes(shapes).geometry.iloc[0]
    assert [lat for _, lat in line.coords] == [38.0, 38.5, 39.0]


def test_create_lines_from_shapes_skips_single_point_shape() -> None:
    shapes = pd.DataFrame(
        {
            "shape_id": ["S1", "S1", "S2"],
            "shape_pt_sequence": [1, 2, 1],
            "shape_pt_lat": [38.0, 38.1, 38.2],
            "shape_pt_lon": [-77.0, -77.0, -77.0],
        }
    )
    assert create_lines_from_shapes(shapes)["shape_id"].tolist() == ["S1"]


# ---------------------------------------------------------------------------
# add_route_labels
# ---------------------------------------------------------------------------


def test_add_route_labels_keeps_routes_sharing_a_short_name_apart() -> None:
    routes = pd.DataFrame({"route_id": ["A", "B", "C"], "route_short_name": ["10", "10", ""]})
    assert add_route_labels(routes)["route_label"].tolist() == ["10_A", "10_B", "C"]


# ---------------------------------------------------------------------------
# identify_first_last_stops
# ---------------------------------------------------------------------------


def test_identify_first_last_stops_uses_lowest_sequence_whatever_its_value() -> None:
    # The first stop_sequence is 5, not 1, and a blank departure_time on that row
    # must not be filled from a later stop.
    trips = pd.DataFrame({"trip_id": ["T1", "T2"]})
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1", "T2", "T2"],
            "stop_sequence": [20, 5, 6, 1, 2],
            "stop_id": ["C", "A", "B", "A", "B"],
            "departure_time": ["7:30:00", "7:05:00", "7:10:00", None, "8:10:00"],
        }
    )
    stops = pd.DataFrame({"stop_id": ["A", "B", "C"], "stop_name": ["Alpha", "Bravo", "Charlie"]})
    out = identify_first_last_stops(trips, stop_times, stops).set_index("trip_id")
    assert out.loc["T1", "first_stop_name"] == "Alpha"
    assert out.loc["T1", "last_stop_name"] == "Charlie"
    assert out.loc["T1", "departure_time"] == "7:05:00"
    assert out.loc["T1", "departure_minutes"] == 425
    assert pd.isna(out.loc["T2", "departure_time"])


def test_identify_first_last_stops_keeps_trips_without_stop_times() -> None:
    trips = pd.DataFrame({"trip_id": ["T1", "T2"]})
    stop_times = pd.DataFrame(
        {"trip_id": ["T1"], "stop_sequence": [1], "stop_id": ["A"], "departure_time": ["7:00"]}
    )
    stops = pd.DataFrame({"stop_id": ["A"], "stop_name": ["Alpha"]})
    out = identify_first_last_stops(trips, stop_times, stops)
    assert out["trip_id"].tolist() == ["T1", "T2"]


# ---------------------------------------------------------------------------
# split_trip_exceptions
# ---------------------------------------------------------------------------


def test_split_trip_exceptions_lists_broken_references() -> None:
    final_data = pd.DataFrame(
        {
            "route_id": ["R1"] * 4,
            "route_short_name": ["1"] * 4,
            "direction_id": ["0"] * 4,
            "trip_id": ["OK", "NOSHAPE", "BADSHAPE", "BADSTOP"],
            "shape_id": ["S1", "", "S9", "S1"],
            "shape_direction": ["NB", None, None, "NB"],
            "first_stop_id": ["A", "A", "A", "Z"],
            "last_stop_id": ["B", "B", "B", "B"],
            "departure_minutes": [420, 420, 420, 420],
        }
    )
    trips = pd.DataFrame(
        {
            "route_id": ["R1", "GONE"],
            "trip_id": ["OK", "ORPHAN"],
            "direction_id": ["0", "0"],
            "shape_id": ["S1", "S1"],
        }
    )
    routes = pd.DataFrame({"route_id": ["R1"]})
    stops = pd.DataFrame({"stop_id": ["A", "B"]})

    kept, exceptions = split_trip_exceptions(final_data, trips, routes, stops)

    assert kept["trip_id"].tolist() == ["OK", "BADSTOP"]
    listed = dict(zip(exceptions["trip_id"], exceptions["excluded"]))
    assert listed == {"ORPHAN": True, "NOSHAPE": True, "BADSHAPE": True, "BADSTOP": False}


# ---------------------------------------------------------------------------
# determine_dominant_shapes
# ---------------------------------------------------------------------------


@pytest.fixture()
def final_data_df() -> pd.DataFrame:
    # SHP1 has 3 trips, SHP2 has 2 trips → SHP1 is dominant
    return pd.DataFrame(
        {
            "route_id": ["R1"] * 5,
            "route_short_name": ["1"] * 5,
            "direction_id": ["0"] * 5,
            "shape_id": ["SHP1", "SHP1", "SHP1", "SHP2", "SHP2"],
            "other_col": range(5),
        }
    )


def test_determine_dominant_shapes_flags_most_common(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    shp1_dominant = result.loc[result.shape_id == "SHP1", "is_dominant"]
    assert shp1_dominant.all()


def test_determine_dominant_shapes_non_dominant_is_false(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    shp2_dominant = result.loc[result.shape_id == "SHP2", "is_dominant"]
    assert (shp2_dominant == False).all()  # noqa: E712 - checks the value, not truthiness


def test_determine_dominant_shapes_preserves_row_count(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    assert len(result) == len(final_data_df)


def test_determine_dominant_shapes_has_is_dominant_column(final_data_df: pd.DataFrame) -> None:
    result = determine_dominant_shapes(final_data_df)
    assert "is_dominant" in result.columns


def test_determine_dominant_shapes_by_route_id_not_short_name() -> None:
    # Two routes share short name "10"; each keeps its own dominant shape.
    data = pd.DataFrame(
        {
            "route_id": ["A", "A", "A", "B"],
            "route_short_name": ["10"] * 4,
            "direction_id": ["0"] * 4,
            "shape_id": ["SA", "SA", "SA", "SB"],
        }
    )
    result = determine_dominant_shapes(data)
    assert result["is_dominant"].all()


def test_determine_dominant_shapes_keeps_blank_direction() -> None:
    data = pd.DataFrame(
        {"route_id": ["A"], "route_short_name": ["1"], "direction_id": [""], "shape_id": ["S"]}
    )
    assert determine_dominant_shapes(data)["is_dominant"].tolist() == [True]


# ---------------------------------------------------------------------------
# flag_suspicious_data
# ---------------------------------------------------------------------------


def _summary(**columns: list[Any]) -> pd.DataFrame:
    n = len(columns["shape_direction"])
    base: dict[str, list[Any]] = {"route_id": ["R1"] * n, "route_short_name": ["1"] * n}
    return pd.DataFrame({**base, **columns})


def test_flag_suspicious_data_clean_data_returns_empty() -> None:
    summary = _summary(direction_id=["0", "1"], shape_direction=["NB", "SB"])
    flags = flag_suspicious_data(summary)
    assert flags.empty
    assert list(flags.columns) == ["route_id", "route_short_name", "direction_id", "problem"]


def test_flag_suspicious_data_flags_one_shape_direction_for_two_direction_ids() -> None:
    summary = _summary(direction_id=["0", "1"], shape_direction=["NB", "NB"])
    assert len(flag_suspicious_data(summary)) == 1


def test_flag_suspicious_data_flags_minority_shape_running_the_other_way() -> None:
    # Five NB trips and one SB trip share direction_id 0.
    summary = _summary(direction_id=["0", "0"], shape_direction=["NB", "SB"])
    flags = flag_suspicious_data(summary)
    assert flags["problem"].str.contains("Conflicting cardinal").tolist() == [True]


# ---------------------------------------------------------------------------
# export_jpegs
# ---------------------------------------------------------------------------


def _record_plots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    names: list[str] = []
    monkeypatch.setattr(mod, "OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(mod, "plot_route_shape", lambda *args: names.append(Path(args[3]).name))
    return names


def test_export_jpegs_maps_shared_shape_for_each_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    names = _record_plots(monkeypatch, tmp_path)
    summary = pd.DataFrame(
        {
            "route_id": ["A", "B"],
            "route_label": ["1", "2"],
            "direction_id": ["0", "0"],
            "shape_id": ["S", "S"],
            "is_dominant": [True, True],
        }
    )
    export_jpegs(summary, gpd.GeoDataFrame({"shape_id": []}), dominant_only=True)
    assert names == ["Route_1_Dir_0_DominantShape.jpeg", "Route_2_Dir_0_DominantShape.jpeg"]


def test_export_jpegs_all_shapes_get_distinct_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    names = _record_plots(monkeypatch, tmp_path)
    summary = pd.DataFrame(
        {
            "route_id": ["A", "A"],
            "route_label": ["1/X", "1/X"],
            "direction_id": ["0", "0"],
            "shape_id": ["S1", "S2"],
            "is_dominant": [True, False],
        }
    )
    export_jpegs(summary, gpd.GeoDataFrame({"shape_id": []}), dominant_only=False)
    assert names == ["Route_1-X_Dir_0_Shape_S1.jpeg", "Route_1-X_Dir_0_Shape_S2.jpeg"]


# ---------------------------------------------------------------------------
# main (end to end on a tiny feed, maps off)
# ---------------------------------------------------------------------------


def _write_feed(folder: Path, sequence_column: str = "1,2,10") -> None:
    folder.mkdir()
    seqs = sequence_column.split(",")
    (folder / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name\nR1,1,Main\n", encoding="utf-8"
    )
    (folder / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nA,Alpha,38.9,-77.0\nB,Bravo,39.0,-77.0\n",
        encoding="utf-8",
    )
    (folder / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id,shape_id\n"
        "R1,WK,T1,0,S1\nR1,WK,T2,0,S1\nR1,WK,T3,1,\n",
        encoding="utf-8",
    )
    (folder / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        f"T1,10:00:00,10:00:00,A,{seqs[0]}\nT1,10:20:00,10:20:00,B,{seqs[2]}\n"
        f"T2,9:00:00,9:00:00,A,{seqs[0]}\nT2,9:20:00,9:20:00,B,{seqs[2]}\n"
        f"T3,9:00:00,9:00:00,B,{seqs[0]}\nT3,9:20:00,9:20:00,A,{seqs[2]}\n",
        encoding="utf-8",
    )
    (folder / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        f"S1,38.9,-77.0,{seqs[0]}\nS1,38.95,-77.0,{seqs[1]}\nS1,39.0,-77.0,{seqs[2]}\n",
        encoding="utf-8",
    )


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    _write_feed(tmp_path / "gtfs")
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "GTFS_FOLDER", str(tmp_path / "gtfs"))
    monkeypatch.setattr(mod, "OUTPUT_FOLDER", str(out))
    monkeypatch.setattr(mod, "ROUTE_FILTER_OUT", [])
    monkeypatch.setattr(mod, "EXPORT_JPEG", False)
    return out


def test_main_end_to_end(configured: Path) -> None:
    assert mod.main() == 0
    summary = pd.read_excel(configured / "Directions_Summary.xlsx", dtype=str)
    assert summary[["shape_id", "shape_direction", "trip_count"]].to_numpy().tolist() == [
        ["S1", "NB", "2"]
    ]
    departures = pd.read_excel(configured / "Route_1_Dir_0_departures.xlsx", dtype=str)
    assert departures["departure_time"].tolist() == ["9:00:00", "10:00:00"]
    exceptions = pd.read_excel(configured / "Trip_Exceptions.xlsx", dtype=str)
    assert exceptions["trip_id"].tolist() == ["T3"]


def test_main_rewrites_suspicious_workbook_on_clean_run(configured: Path) -> None:
    configured.mkdir()
    stale = pd.DataFrame({"problem": ["from an earlier run"]})
    stale.to_excel(configured / "Suspicious_RouteDirections.xlsx", index=False)
    assert mod.main() == 0
    assert pd.read_excel(configured / "Suspicious_RouteDirections.xlsx").empty


def test_main_skips_workbooks_when_xlsx_export_off(
    configured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "EXPORT_XLSX", False)
    assert mod.main() == 0
    assert list(configured.iterdir()) == []


def test_main_reports_non_numeric_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_feed(tmp_path / "bad", sequence_column="1,2,x")
    monkeypatch.setattr(mod, "GTFS_FOLDER", str(tmp_path / "bad"))
    monkeypatch.setattr(mod, "OUTPUT_FOLDER", str(tmp_path / "out"))
    assert mod.main() == 1
    assert "non-numeric 'stop_sequence'" in caplog.text
