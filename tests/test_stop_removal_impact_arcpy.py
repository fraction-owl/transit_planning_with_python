"""Walking routes in stop_removal_impact_arcpy, with arcpy stubbed by shapely-backed lines."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pandas as pd
import pytest
from shapely.geometry import LineString, Point
from shapely.ops import substring

# A 1,000 ft sidewalk, and a 2,100 ft U-shaped one whose ends a 100 ft connector joins.
STRAIGHT = [[(0, 0), (1000, 0)]]
U_LOOP = [[(0, 0), (0, 1000), (100, 1000), (100, 0)], [(100, 0), (0, 0)]]


class FakePolyline:
    """The parts of arcpy.Polyline the routing code uses, backed by a shapely LineString."""

    def __init__(self, line: LineString) -> None:
        self.line = line
        self.length = float(line.length)
        self.pointCount = len(line.coords)
        first, last = line.coords[0], line.coords[-1]
        self.firstPoint = SimpleNamespace(X=first[0], Y=first[1])
        self.lastPoint = SimpleNamespace(X=last[0], Y=last[1])

    def __iter__(self) -> Iterator[list[SimpleNamespace]]:
        yield [SimpleNamespace(X=x, Y=y) for x, y in self.line.coords]

    def segmentAlongLine(  # noqa: N802 - arcpy's name
        self, start: float, end: float, use_percentage: bool = False
    ) -> FakePolyline:
        return FakePolyline(substring(self.line, start, end, normalized=use_percentage))


@pytest.fixture()
def target(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    name = "scripts.stop_analysis.stop_removal_impact_arcpy"
    arcpy = mock.MagicMock()
    arcpy.Array = list
    arcpy.Polyline = lambda points, _sr=None: FakePolyline(LineString([(p.X, p.Y) for p in points]))
    monkeypatch.setitem(sys.modules, "arcpy", arcpy)
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.delitem(sys.modules, name)
    return module


def walk(
    target: ModuleType,
    lines: list[list[tuple[float, float]]],
    stops: dict[str, tuple[float, float]],
    a: str,
    b: str,
) -> tuple[float, Any]:
    """Snap *stops* to the nearest of *lines* (feet) and route stop *a* to stop *b*."""
    segments = [FakePolyline(LineString(c)) for c in lines]
    graph, endpoints = target.build_graph(segments, 5.0)
    rows = []
    for sid, xy in stops.items():
        point = Point(xy)
        eid = min(range(len(segments)), key=lambda i: segments[i].line.distance(point))
        s_u = float(segments[eid].line.project(point))
        rows.append(
            {
                "stop_id": sid,
                "edge_id": eid,
                "s_u": s_u,
                "to_u_u": s_u,
                "to_v_u": segments[eid].length - s_u,
            }
        )
    return target.stop_to_stop_network(
        graph, segments, endpoints, pd.DataFrame(rows), a, b, sr=None
    )


def ends(path: FakePolyline) -> tuple[tuple[float, float], tuple[float, float]]:
    return path.line.coords[0], path.line.coords[-1]


@pytest.mark.parametrize(("a", "b"), [("A", "B"), ("B", "A")])
def test_stops_on_one_segment_walk_straight_between(target: ModuleType, a: str, b: str) -> None:
    stops = {"A": (400, 0), "B": (600, 0)}
    distance, path = walk(target, STRAIGHT, stops, a, b)
    assert distance == pytest.approx(200.0)  # Not 1,000 ft out to an end and back.
    assert path.length == pytest.approx(distance)
    assert ends(path) == (stops[a], stops[b])


def test_coincident_stops_are_zero_feet_apart(target: ModuleType) -> None:
    distance, path = walk(target, STRAIGHT, {"A": (400, 0), "B": (400, 0)}, "A", "B")
    assert distance == 0.0
    assert path is None  # No line to draw.


@pytest.mark.parametrize(("a", "b"), [("A", "B"), ("B", "A")])
def test_a_shorter_network_path_beats_walking_along_the_segment(
    target: ModuleType, a: str, b: str
) -> None:
    # Both stops are on the U, 1,900 ft apart along it, but 300 ft via the connector.
    stops = {"A": (0, 100), "B": (100, 100)}
    distance, path = walk(target, U_LOOP, stops, a, b)
    assert distance == pytest.approx(300.0)
    assert path.length == pytest.approx(distance)
    assert ends(path) == (stops[a], stops[b])
