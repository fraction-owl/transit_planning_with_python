"""Tests for scripts/gtfs_data_quality/route_direction_classifier_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a stand-in module supplies the
geometry calls the script uses (SpatialReference, Point, Array, Polyline and
``projectAs``). Its "projection" is a flat meters-per-degree scaling near 38.9 N,
close enough to check the pipeline; no real coordinate transformation runs.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

MODULE = "scripts.gtfs_data_quality.route_direction_classifier_arcpy"

_M_PER_DEG_LAT = 110_574.0
_M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(38.9))


class _SpatialReference:
    def __init__(self, wkid: int) -> None:
        self.type = "Geographic" if wkid == 4326 else "Projected"


class _Point:
    def __init__(self, x: float, y: float) -> None:
        self.X = x
        self.Y = y


class _Polyline:
    def __init__(self, points: list[_Point], sr: Any) -> None:
        # Like ArcPy, a line with no length has no points.
        distinct = {(p.X, p.Y) for p in points}
        self._points = points if len(distinct) > 1 else []
        self.pointCount = len(self._points)  # noqa: N815 - arcpy API name

    def projectAs(self, sr: Any) -> _Polyline:  # noqa: N802 - arcpy API name
        projected = [_Point(p.X * _M_PER_DEG_LON, p.Y * _M_PER_DEG_LAT) for p in self._points]
        return _Polyline(projected, sr)

    def getPart(self, index: int) -> list[_Point]:  # noqa: N802 - arcpy API name
        return self._points


@pytest.fixture()
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the script against a stand-in ``arcpy`` without leaking either module."""
    stub = types.ModuleType("arcpy")
    stub.SpatialReference = _SpatialReference  # type: ignore[attr-defined]
    stub.Point = _Point  # type: ignore[attr-defined]
    stub.Array = list  # type: ignore[attr-defined]
    stub.Polyline = _Polyline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arcpy", stub)
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_build_shape_coords_orders_points_numerically(mod: types.ModuleType) -> None:
    shapes = pd.DataFrame(
        {
            "shape_id": ["S1"] * 3 + ["S2"],
            "shape_pt_sequence": [10, 1, 2, 1],
            "shape_pt_lat": [39.0, 38.0, 38.5, 38.0],
            "shape_pt_lon": [-77.0, -77.0, -77.0, -77.0],
        }
    )
    coords = mod.build_shape_coords(shapes)
    assert list(coords) == ["S1"]  # S2 has a single point
    assert [lat for _, lat in coords["S1"]] == [38.0, 38.5, 39.0]


def test_project_shapes_skips_zero_length_shape(mod: types.ModuleType) -> None:
    coords = {"S1": [(-77.0, 38.9), (-77.0, 39.0)], "S2": [(-77.0, 38.9), (-77.0, 38.9)]}
    assert list(mod.project_shapes(coords, 26985)) == ["S1"]


def test_classify_direction_compares_ground_distance_not_degrees(mod: types.ModuleType) -> None:
    # ~5.0 km north and ~4.3 km east: more degrees of longitude than latitude.
    projected = mod.project_shapes({"S": [(-77.0, 38.9), (-76.95, 38.945)]}, 26985)
    assert mod.classify_direction(projected["S"]) == "NB"


def test_project_shapes_rejects_geographic_crs(mod: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="not a projected coordinate system"):
        mod.project_shapes({"S": [(-77.0, 38.9), (-77.0, 39.0)]}, 4326)


def test_plot_route_shape_leaves_session_backend_and_figures_alone(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    original = matplotlib.get_backend()
    plt.switch_backend("svg")  # stands in for a notebook or ArcGIS Pro backend
    try:
        open_figure = plt.figure()
        mod.plot_route_shape([(-77.0, 38.8), (-76.9, 39.0)], "1", "0", tmp_path / "map.jpeg")
        assert matplotlib.get_backend() == "svg"
        assert plt.fignum_exists(open_figure.number)
        assert (tmp_path / "map.jpeg").stat().st_size > 0
    finally:
        plt.close("all")
        plt.switch_backend(original)


def test_plot_route_shape_keeps_ground_proportions(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    saved: list[Any] = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: saved.append(self))
    mod.plot_route_shape([(-77.0, 38.8), (-76.9, 39.0)], "1", "0", tmp_path / "map.jpeg")
    aspect = saved[0].axes[0].get_aspect()
    assert aspect == pytest.approx(1 / math.cos(math.radians(38.9)))


@pytest.mark.parametrize(
    ("coords", "expected"),
    [
        ([(0, 0), (100, 0), (100, 100), (0, 100)], "CCW"),
        ([(0, 0), (0, 100), (100, 100), (100, 0)], "CW"),
        ([(0, 500_000), (0, 0)], "SB"),
        ([(500_000, 0), (0, 0)], "WB"),
    ],
)
def test_classify_direction(
    mod: types.ModuleType, coords: list[tuple[float, float]], expected: str
) -> None:
    assert mod.classify_direction(coords) == expected


# ---------------------------------------------------------------------------
# Table logic shared with the GeoPandas script
# ---------------------------------------------------------------------------


def test_determine_dominant_shapes_by_route_id_not_short_name(mod: types.ModuleType) -> None:
    data = pd.DataFrame(
        {
            "route_id": ["A", "A", "A", "B"],
            "route_short_name": ["10"] * 4,
            "direction_id": ["0"] * 4,
            "shape_id": ["SA", "SA", "SA", "SB"],
        }
    )
    assert mod.determine_dominant_shapes(data)["is_dominant"].all()


def test_flag_suspicious_data_flags_minority_shape(mod: types.ModuleType) -> None:
    summary = pd.DataFrame(
        {
            "route_id": ["R1", "R1"],
            "route_short_name": ["1", "1"],
            "direction_id": ["0", "0"],
            "shape_direction": ["NB", "SB"],
        }
    )
    assert len(mod.flag_suspicious_data(summary)) == 1


def test_flag_suspicious_data_flags_opposite_loops(mod: types.ModuleType) -> None:
    summary = pd.DataFrame(
        {
            "route_id": ["R1", "R1"],
            "route_short_name": ["1", "1"],
            "direction_id": ["0", "0"],
            "shape_direction": ["CW", "CCW"],
        }
    )
    assert mod.flag_suspicious_data(summary)["problem"].tolist() == [
        "Conflicting loop directions ['CCW', 'CW']"
    ]


def test_output_file_name_replaces_unsafe_characters(mod: types.ModuleType) -> None:
    assert mod.output_file_name("1/X", "", "Shape_a:b.jpeg") == "Route_1-X_Dir_none_Shape_a-b.jpeg"


def test_add_route_labels_separates_labels_that_share_a_file_name(mod: types.ModuleType) -> None:
    routes = pd.DataFrame(
        {"route_id": ["R1", "R2", "R3"], "route_short_name": ["A/B", "A:B", "a/b"]}
    )
    assert mod.add_route_labels(routes)["route_label"].tolist() == ["A/B_R1", "A:B_R2", "a/b_R3"]


def test_filter_routes_compares_values_as_text(mod: types.ModuleType) -> None:
    routes = pd.DataFrame({"route_id": ["R1", "R2"], "route_short_name": ["10", "20"]})
    assert mod.filter_routes(routes, routes_include=[10])["route_id"].tolist() == ["R1"]
    with pytest.raises(ValueError, match="No routes are left"):
        mod.filter_routes(routes, routes_include=["99"])


def test_check_unique_file_names_ignores_case(mod: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="2 output file name"):
        mod.check_unique_file_names(
            [("A.xlsx", "one"), ("a.xlsx", "two"), ("B.jpeg", "three"), ("b.JPEG", "four")]
        )


def test_split_trip_exceptions_tells_blank_first_stop_from_missing_stop_times(
    mod: types.ModuleType,
) -> None:
    final_data = pd.DataFrame(
        {
            "route_id": ["R1", "R1"],
            "route_short_name": ["1", "1"],
            "direction_id": ["0", "0"],
            "trip_id": ["BLANK", "NOROWS"],
            "shape_id": ["S1", "S1"],
            "shape_direction": ["NB", "NB"],
            "first_stop_id": [None, None],
            "last_stop_id": ["B", None],
            "departure_minutes": [420, None],
        }
    )
    kept, exceptions = mod.split_trip_exceptions(
        final_data,
        final_data,
        pd.DataFrame({"route_id": ["R1"]}),
        pd.DataFrame({"stop_id": ["B"]}),
        pd.DataFrame({"trip_id": ["BLANK"]}),
    )
    assert kept["trip_id"].tolist() == ["BLANK"]
    assert exceptions["problem"].tolist() == ["no rows in stop_times.txt", "first stop_id is blank"]


# ---------------------------------------------------------------------------
# main (end to end on a tiny feed, maps off)
# ---------------------------------------------------------------------------


def _write_feed(folder: Path) -> None:
    folder.mkdir()
    files = {
        "routes.txt": "route_id,route_short_name,route_long_name\nR1,1,Main\nR2,1,Main Exp\n",
        "stops.txt": (
            "stop_id,stop_name,stop_lat,stop_lon\nA,Alpha,38.9,-77.0\nB,Bravo,39.0,-77.0\n"
            "A2,Alpha,38.9,-77.0\n"
        ),
        "trips.txt": (
            "route_id,service_id,trip_id,direction_id,shape_id\n"
            "R1,WK,T1,0,S1\nR1,WK,T2,0,S1\nR1,WK,T3,0,NA\nR2,WK,T4,0,S1\nR1,WK,T5,1,\n"
            "R1,WK,T6,0,S1\n"
        ),
        "stop_times.txt": (
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "T1,10:00:00,10:00:00,A,1\nT1,10:20:00,10:20:00,B,10\n"
            "T2,9:00:01,9:00:01,A,1\nT2,9:20:00,9:20:00,B,10\n"
            "T3,8:00:00,8:00:00,B,1\nT3,8:20:00,8:20:00,A,10\n"
            "T4,8:00:00,8:00:00,A,1\nT4,8:20:00,8:20:00,B,10\n"
            "T5,8:00:00,8:00:00,B,1\nT5,8:20:00,8:20:00,A,10\n"
            "T6,9:00:29,9:00:29,A2,1\nT6,9:20:00,9:20:00,B,10\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
            "S1,38.9,-77.0,1\nS1,38.95,-77.0,2\nS1,39.0,-77.0,10\n"
            "NA,39.0,-77.0,1\nNA,38.95,-77.0,2\nNA,38.9,-77.0,10\n"
        ),
    }
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")


def test_main_end_to_end(mod: types.ModuleType, tmp_path: Path) -> None:
    _write_feed(tmp_path / "gtfs")
    out = tmp_path / "out"
    argv = ["--gtfs", str(tmp_path / "gtfs"), "--output-dir", str(out), "--no-export-jpeg"]
    assert mod.main(argv) == 0

    summary = pd.read_excel(out / "Directions_Summary.xlsx", dtype=str, keep_default_na=False)
    columns = ["route_id", "shape_id", "shape_direction", "is_dominant", "first_stop_id"]
    rows = summary[[*columns, "trip_count"]]
    assert rows.to_numpy().tolist() == [
        ["R1", "NA", "SB", "False", "B", "1"],
        ["R1", "S1", "NB", "True", "A", "2"],
        ["R1", "S1", "NB", "True", "A2", "1"],
        ["R2", "S1", "NB", "True", "A", "1"],
    ]
    flagged = pd.read_excel(out / "Suspicious_RouteDirections.xlsx", dtype=str)
    assert flagged["route_id"].tolist() == ["R1"]
    exceptions = pd.read_excel(out / "Trip_Exceptions.xlsx", dtype=str)
    assert exceptions["trip_id"].tolist() == ["T5"]
    departures = pd.read_excel(out / "Route_1_R1_Dir_0_departures.xlsx", dtype=str)
    assert departures["trip_id"].tolist() == ["T3", "T2", "T6", "T1"]
    run_log = (out / "Directions_Summary_runlog.txt").read_text(encoding="utf-8")
    assert "GTFS_PATH = " in run_log
    assert "Trips excluded:          1" in run_log


def test_main_survives_arcpy_rebinding_datetime(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In ArcGIS Pro, arcpy can replace a script's `datetime` name with the datetime
    # module, so `datetime.now()` fails. The run log must not depend on that name.
    import datetime

    monkeypatch.setattr(mod, "datetime", datetime, raising=False)
    _write_feed(tmp_path / "gtfs")
    out = tmp_path / "out"
    argv = ["--gtfs", str(tmp_path / "gtfs"), "--output-dir", str(out), "--no-export-jpeg"]
    assert mod.main(argv) == 0
    assert (out / "Directions_Summary_runlog.txt").exists()


def test_main_writes_header_only_review_workbooks_for_clean_feed(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    _write_feed(tmp_path / "gtfs")
    trips = tmp_path / "gtfs" / "trips.txt"
    trips.write_text(
        "route_id,service_id,trip_id,direction_id,shape_id\nR1,WK,T1,0,S1\nR1,WK,T3,1,NA\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    argv = ["--gtfs", str(tmp_path / "gtfs"), "--output-dir", str(out), "--no-export-jpeg"]
    assert mod.main(argv) == 0
    assert pd.read_excel(out / "Suspicious_RouteDirections.xlsx").empty
    assert pd.read_excel(out / "Trip_Exceptions.xlsx").empty


def test_main_maps_every_shape_when_asked(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_feed(tmp_path / "gtfs")
    out = tmp_path / "out"
    names: list[str] = []
    monkeypatch.setattr(mod, "plot_route_shape", lambda *args: names.append(args[3].name))
    argv = [
        "--gtfs",
        str(tmp_path / "gtfs"),
        "--output-dir",
        str(out),
        "--no-export-xlsx",
        "--no-map-only-dominant-shape",
    ]
    assert mod.main(argv) == 0
    assert names == [
        "Route_1_R1_Dir_0_Shape_NA.jpeg",
        "Route_1_R1_Dir_0_Shape_S1.jpeg",
        "Route_1_R2_Dir_0_Shape_S1.jpeg",
    ]


def test_main_returns_2_for_placeholder_paths(mod: types.ModuleType) -> None:
    assert mod.main([]) == 2


def test_main_rejects_repeated_stop_id(
    mod: types.ModuleType, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_feed(tmp_path / "gtfs")
    with (tmp_path / "gtfs" / "stops.txt").open("a", encoding="utf-8") as handle:
        handle.write("B,Bravo again,39.0,-77.0\n")
    argv = ["--gtfs", str(tmp_path / "gtfs"), "--output-dir", str(tmp_path / "out")]
    assert mod.main(argv) == 1
    assert "repeated 'stop_id'" in caplog.text


def test_main_warns_about_route_files_from_an_earlier_run(
    mod: types.ModuleType, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_feed(tmp_path / "gtfs")
    out = tmp_path / "out"
    out.mkdir()
    (out / "Route_9_Dir_0_Shape_X.jpeg").write_bytes(b"")
    argv = ["--gtfs", str(tmp_path / "gtfs"), "--output-dir", str(out), "--no-export-jpeg"]
    assert mod.main(argv) == 0
    assert "1 per-route file(s)" in caplog.text
    assert (out / "Route_9_Dir_0_Shape_X.jpeg").exists()


def test_main_reports_missing_required_column(
    mod: types.ModuleType, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_feed(tmp_path / "gtfs")
    (tmp_path / "gtfs" / "trips.txt").write_text(
        "service_id,trip_id,direction_id,shape_id\nWK,T1,0,S1\n", encoding="utf-8"
    )
    argv = ["--gtfs", str(tmp_path / "gtfs"), "--output-dir", str(tmp_path / "out")]
    assert mod.main(argv) == 1
    assert "trips.txt is missing required column(s): route_id" in caplog.text


def test_main_reports_missing_feed(mod: types.ModuleType, tmp_path: Path) -> None:
    argv = ["--gtfs", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")]
    assert mod.main(argv) == 1
