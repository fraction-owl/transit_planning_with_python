"""Tests for scripts/gtfs_exports/gtfs_to_shapefile_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a stand-in module supplies the geometry
calls used to read a feed and pick route patterns (SpatialReference, Point, Array,
Polyline and ``getLength``). No shapefile is written.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

MODULE = "scripts.gtfs_exports.gtfs_to_shapefile_arcpy"


class _SpatialReference:
    def __init__(self, wkid: int) -> None:
        self.name = "GCS_WGS_1984"


class _Array(list):
    def add(self, item: Any) -> None:
        self.append(item)

    @property
    def count(self) -> int:
        return len(self)


class _Polyline:
    length_units: list[str] = []

    def __init__(self, points: _Array, sr: Any) -> None:
        self.points = list(points)

    def getLength(self, measurement: str, units: str) -> float:  # noqa: N802 - arcpy API name
        _Polyline.length_units.append(units)
        return float(len(self.points))


@pytest.fixture()
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the script against a stand-in ``arcpy`` without leaking either module."""
    stub = types.ModuleType("arcpy")
    stub.SpatialReference = _SpatialReference  # type: ignore[attr-defined]
    stub.Point = lambda x, y: (x, y)  # type: ignore[attr-defined]
    stub.Array = _Array  # type: ignore[attr-defined]
    stub.Polyline = _Polyline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arcpy", stub)
    sys.modules.pop(MODULE, None)
    _Polyline.length_units = []
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


def _write_feed(gtfs_dir: Path) -> None:
    """Write a feed with numeric-looking ids, a leading zero and one blank shape_id."""
    gtfs_dir.mkdir()
    (gtfs_dir / "routes.txt").write_text("route_id,route_short_name\n01,1\n")
    (gtfs_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id,shape_id,direction_id\n"
        "01,WK,1,12345,0\n01,WK,2,,0\n01,WK,3,12345,1\n"
    )
    (gtfs_dir / "stop_times.txt").write_text(
        "trip_id,stop_id,stop_sequence\n1,001,1\n1,002,2\n3,002,1\n3,001,2\n"
    )
    (gtfs_dir / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n001,A,38.90,-77.03\n002,B,38.91,-77.02\n"
    )
    (gtfs_dir / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        "12345,38.90,-77.03,1\n12345,38.91,-77.02,2\n"
    )


def test_read_gtfs_tables_keeps_ids_as_text(mod: types.ModuleType, tmp_path: Path) -> None:
    """Ids keep leading zeros, and a blank shape_id doesn't turn shape ids into floats."""
    _write_feed(tmp_path / "gtfs")
    tables = mod._read_gtfs_tables(tmp_path / "gtfs")
    assert tables["trips"]["route_id"].tolist() == ["01", "01", "01"]
    assert tables["trips"]["shape_id"].dropna().tolist() == ["12345", "12345"]
    assert tables["stops"]["stop_id"].tolist() == ["001", "002"]


@pytest.mark.parametrize("pattern_mode", ["most_common", "most_stops", "longest"])
def test_build_route_patterns_matches_shapes_with_blank_shape_id(
    mod: types.ModuleType, tmp_path: Path, pattern_mode: str
) -> None:
    """Every direction gets its shape geometry, and the route filter matches "01"."""
    _write_feed(tmp_path / "gtfs")
    tables = mod._read_gtfs_tables(tmp_path / "gtfs")
    shape_geoms = mod._build_shape_geometries(tables["shapes"])

    records = mod._build_route_patterns(
        tables["trips"],
        tables["stop_times"],
        tables["routes"],
        shape_geoms,
        pattern_mode,
        route_filter_in={"01"},
    )

    assert [(r["route_id"], r["direction_id"], r["shape_id"]) for r in records] == [
        ("01", 0, "12345"),
        ("01", 1, "12345"),
    ]
    assert records[0]["route_short"] == "1"
    if pattern_mode == "longest":
        assert set(_Polyline.length_units) == {"MILES"}
