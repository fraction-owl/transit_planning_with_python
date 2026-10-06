"""Tests for scripts/stop_analysis/stop_spacing_flagger_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a stand-in module backed by shapely
and pyproj answers the geometry calls the script makes (projectAs,
measureOnLine, segmentAlongLine, distanceTo, ...) and keeps feature classes in
memory. The tests run the whole pipeline on small synthetic feeds.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import LineString
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import substring

MODULE = "scripts.stop_analysis.stop_spacing_flagger_arcpy"

# ---------------------------------------------------------------------------
# Stand-in arcpy
# ---------------------------------------------------------------------------


def _fake_arcpy() -> types.ModuleType:
    features: dict[str, dict[str, Any]] = {}

    class SpatialReference:
        def __init__(self, wkid: int) -> None:
            try:
                self._crs = CRS.from_epsg(int(wkid))
            except Exception as err:  # arcpy raises RuntimeError for unknown codes
                raise RuntimeError(f"invalid wkid {wkid}") from err
            self.name = self._crs.name.replace(" ", "_")
            self.type = "Projected" if self._crs.is_projected else "Geographic"
            unit = self._crs.axis_info[0]
            self.metersPerUnit = unit.unit_conversion_factor if self._crs.is_projected else 0.0

    class Point:
        def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
            self.X, self.Y = float(x), float(y)

    class Array(list):
        def add(self, pt: Point) -> None:
            self.append(pt)

        @property
        def count(self) -> int:
            return len(self)

    def _project(geom: Any, src: SpatialReference, dst: SpatialReference) -> Any:
        t = Transformer.from_crs(src._crs, dst._crs, always_xy=True)
        return ops.transform(t.transform, geom)

    class _Geometry:
        def __init__(self, geom: Any, sr: SpatialReference) -> None:
            self._g, self.spatialReference = geom, sr

        @property
        def length(self) -> float:
            return self._g.length

        @property
        def extent(self) -> types.SimpleNamespace:
            xmin, ymin, xmax, ymax = self._g.bounds
            return types.SimpleNamespace(XMin=xmin, YMin=ymin, XMax=xmax, YMax=ymax)

        @property
        def firstPoint(self) -> Point | None:
            if self._g.is_empty:
                return None
            x, y = self._g.coords[0] if hasattr(self._g, "coords") else self._g.geoms[0].coords[0]
            return Point(x, y)

        def distanceTo(self, other: _Geometry) -> float:
            return self._g.distance(other._g)

    class PointGeometry(_Geometry):
        def __init__(self, pt: Point, sr: SpatialReference) -> None:
            super().__init__(ShapelyPoint(pt.X, pt.Y), sr)

        def projectAs(self, sr: SpatialReference) -> PointGeometry:
            out = PointGeometry(Point(), sr)
            out._g = _project(self._g, self.spatialReference, sr)
            return out

    class Polyline(_Geometry):
        def __init__(self, array: Sequence[Point] | None, sr: SpatialReference) -> None:
            super().__init__(LineString([(p.X, p.Y) for p in array or []]), sr)

        @classmethod
        def _wrap(cls, geom: Any, sr: SpatialReference) -> Polyline:
            out = cls(None, sr)
            out._g = geom
            return out

        @property
        def pointCount(self) -> int:
            parts = getattr(self._g, "geoms", [self._g])
            return sum(len(p.coords) for p in parts if not p.is_empty)

        def getPart(self, index: int) -> Array:
            part = getattr(self._g, "geoms", [self._g])[index]
            return Array(Point(x, y) for x, y in part.coords)

        def projectAs(self, sr: SpatialReference) -> Polyline:
            return Polyline._wrap(_project(self._g, self.spatialReference, sr), sr)

        def measureOnLine(self, geom: _Geometry, use_percentage: bool = False) -> float:
            return self._g.project(geom._g, normalized=use_percentage)

        def segmentAlongLine(
            self, start: float, end: float, use_percentage: bool = False
        ) -> Polyline:
            seg = substring(self._g, start, end, normalized=use_percentage)
            return Polyline._wrap(seg, self.spatialReference)

        def union(self, other: Polyline) -> Polyline:
            return Polyline._wrap(self._g.union(other._g), self.spatialReference)

    class InsertCursor:
        def __init__(self, path: str, fields: list[str]) -> None:
            self.rows = features[path]["rows"]
            self.fields = fields

        def __enter__(self) -> InsertCursor:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def insertRow(self, row: list[Any]) -> None:
            self.rows.append(dict(zip(self.fields, row)))

    def create_feature_class(folder: str, name: str, *_: Any, **__: Any) -> None:
        features[os.path.join(folder, name + ".shp")] = {"rows": []}

    arcpy = types.ModuleType("arcpy")
    members = {
        "SpatialReference": SpatialReference,
        "Point": Point,
        "Array": Array,
        "PointGeometry": PointGeometry,
        "Polyline": Polyline,
        "env": types.SimpleNamespace(overwriteOutput=False),
        "Exists": lambda path: path in features,
        "management": types.SimpleNamespace(
            Delete=lambda path: features.pop(path, None),
            CreateFeatureclass=create_feature_class,
            AddField=lambda *a, **k: None,
        ),
        "da": types.SimpleNamespace(InsertCursor=InsertCursor),
        "FEATURES": features,
    }
    for name, member in members.items():
        setattr(arcpy, name, member)
    return arcpy


@pytest.fixture()
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the script against the stand-in ``arcpy`` without leaking either module."""
    monkeypatch.setitem(sys.modules, "arcpy", _fake_arcpy())
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


# ---------------------------------------------------------------------------
# Synthetic feeds (EPSG:2248 feet around a DC origin)
# ---------------------------------------------------------------------------

_X0, _Y0 = 1_300_000.0, 450_000.0
_TO_LL = Transformer.from_crs("EPSG:2248", "EPSG:4326", always_xy=True)


def _write_feed(
    root: Path,
    stops_xy: Mapping[str, tuple[float, float]],
    trips: list[tuple[str, str, str, str, list[str]]],
    shapes: Mapping[str, Sequence[tuple[float, float]]],
    stop_dists: Mapping[str, Sequence[float]] | None = None,
) -> Path:
    """Write a feed; *trips* holds (trip_id, route_id, direction_id, shape_id, stop_ids)."""
    root.mkdir()

    def lat_lon(x: float, y: float) -> str:
        lon, lat = _TO_LL.transform(_X0 + x, _Y0 + y)
        return f"{lat:.9f},{lon:.9f}"

    (root / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        + "".join(f"{s},Stop {s},{lat_lon(*xy)}\n" for s, xy in stops_xy.items()),
        encoding="utf-8",
    )
    (root / "routes.txt").write_text(
        "route_id,route_short_name\n"
        + "".join(f"{r},{r}\n" for r in sorted({t[1] for t in trips})),
        encoding="utf-8",
    )
    (root / "trips.txt").write_text(
        "trip_id,route_id,direction_id,shape_id\n"
        + "".join(f"{t},{r},{d},{sh}\n" for t, r, d, sh, _ in trips),
        encoding="utf-8",
    )
    sdt_head = ",shape_dist_traveled" if stop_dists else ""
    st_rows = []
    for t, _, _, _, seq in trips:
        for i, s in enumerate(seq):
            dist = f",{stop_dists[t][i]}" if stop_dists else ""
            st_rows.append(f"{t},{s},{i + 1}{dist}\n")
    (root / "stop_times.txt").write_text(
        f"trip_id,stop_id,stop_sequence{sdt_head}\n" + "".join(st_rows), encoding="utf-8"
    )
    sh_rows = []
    for sid, pts in shapes.items():
        along = 0.0
        for i, xy in enumerate(pts):
            if i:
                along += LineString([pts[i - 1], xy]).length
            dist = f",{along:.3f}" if stop_dists else ""
            sh_rows.append(f"{sid},{i + 1},{lat_lon(*xy)}{dist}\n")
    (root / "shapes.txt").write_text(
        f"shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon{sdt_head}\n" + "".join(sh_rows),
        encoding="utf-8",
    )
    return root


def _run(
    mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    gtfs: Path,
    out: Path,
    **config: Any,
) -> int:
    settings = {
        "GTFS_PATH": str(gtfs),
        "OUTPUT_FOLDER": str(out),
        "INCLUDE_ROUTE_IDS": [],
        "FILTER_OUT_LIST": [],
        "PROJECTED_WKID": 2248,
        **config,
    }
    for name, value in settings.items():
        monkeypatch.setattr(mod, name, value)
    return mod.main()


def _short_pairs(out: Path) -> list[tuple[str, str, int]]:
    log = pd.read_csv(out / "short_spacing_segments.txt", sep="\t", dtype=str)
    return [
        (b, e, round(float(d)))
        for b, e, d in zip(log["begin_stop_id"], log["end_stop_id"], log["spacing_ft"])
    ]


def _long_rows(out: Path) -> list[list[str]]:
    flagged = pd.read_csv(out / "long_spacing_segments.csv", dtype=str, keep_default_na=False)
    cols = ["route_id", "direction_id", "start_stop_id", "end_stop_id", "flagged_stop_id"]
    return flagged[cols].to_numpy().tolist()


# ---------------------------------------------------------------------------
# Reading, filtering and spatial reference
# ---------------------------------------------------------------------------


def test_numeric_ids_keep_their_text_and_shape_geometry(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    # A blank shape_id once turned the column to floats, so shape "10" became
    # "10.0" in trips and lost its geometry.
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"007": (0, 30), "NA": (500, 30)},
        [("1", "101", "0", "10", ["007", "NA"]), ("2", "101", "1", "", ["NA", "007"])],
        {"10": [(0, 0), (2_000, 0)]},
    )
    dfs = mod._read_gtfs_tables(gtfs)
    assert dfs["stops"]["stop_id"].tolist() == ["007", "NA"]
    assert dfs["trips"]["shape_id"].tolist() == ["10", ""]
    mod._prepare_tables(dfs)
    routes, trips = mod._filter_routes(dfs["routes"], dfs["trips"], [101], [])
    assert len(routes) == 1 and len(trips) == 2

    shape_geoms, _ = mod._build_shape_geometries(dfs["shapes"], mod.arcpy.SpatialReference(2248))
    records = mod._build_routes_from_shapes(trips, routes, shape_geoms, False)
    assert [(r["route_id"], r["direction_id"]) for r in records] == [("101", 0)]


def test_prepare_tables_keeps_trips_with_blank_or_missing_direction(
    mod: types.ModuleType,
) -> None:
    trips = pd.DataFrame({"trip_id": ["T1", "T2", "T3"], "direction_id": ["0", "", "2"]})
    dfs = {
        "stops": pd.DataFrame({"stop_lat": ["38.9"], "stop_lon": [""]}),
        "stop_times": pd.DataFrame({"stop_sequence": ["1"]}),
        "shapes": pd.DataFrame({"shape_pt_sequence": ["1"]}),
        "trips": trips,
    }
    mod._prepare_tables(dfs)
    unknown = mod.UNKNOWN_DIRECTION_ID
    assert dfs["trips"]["direction_id"].tolist() == [0, unknown, unknown]
    assert dfs["stops"]["stop_lon"].isna().all()

    dfs["trips"] = pd.DataFrame({"trip_id": ["T1"]})
    mod._prepare_tables(dfs)
    assert dfs["trips"]["direction_id"].tolist() == [unknown]


@pytest.mark.parametrize(("wkid", "feet"), [(2248, 1.000002), (26918, 3.280840)])
def test_feet_factor_reads_the_linear_unit(mod: types.ModuleType, wkid: int, feet: float) -> None:
    assert mod._feet_factor(mod._get_projected_sr(wkid)) == pytest.approx(feet, abs=1e-6)


@pytest.mark.parametrize(("wkid", "match"), [(4326, "not projected"), (999_999, "not recognized")])
def test_get_projected_sr_rejects_unusable_wkid(
    mod: types.ModuleType, wkid: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        mod._get_projected_sr(wkid)


def test_main_returns_config_error_for_geographic_sr_or_empty_filter(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"S0": (0, 30), "S1": (2_000, 30)},
        [("T", "A", "0", "S", ["S0", "S1"])],
        {"S": [(0, 0), (3_000, 0)]},
    )
    assert _run(mod, monkeypatch, gtfs, tmp_path / "out", PROJECTED_WKID=4326) == 2
    assert _run(mod, monkeypatch, gtfs, tmp_path / "out", INCLUDE_ROUTE_IDS=["B"]) == 2


# ---------------------------------------------------------------------------
# Spacing along stopping patterns
# ---------------------------------------------------------------------------


def test_main_measures_each_stopping_pattern(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Route E: an express (P0-P2) and a local (P0-P1-P2) share shape SE.
    # Route A/0's 2,000 ft A0-A1 gap holds M, served by A/1 and B/0 only.
    # Route L runs a square loop that ends where it started. Route F's stop R
    # sits in the express gap. Every trip of route L has a blank direction.
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {
            "P0": (0, 30),
            "P1": (1_000, 30),
            "P2": (2_000, 30),
            "R": (1_500, 40),
            "R9": (1_500, 900),
            "A0": (0, 5_030),
            "A1": (2_000, 5_030),
            "M": (1_000, 5_030),
            "N0": (-400, 5_090),
            "Q": (1_000, 7_000),
            "LA": (-30, 9_970),
            "LB": (2_030, 9_970),
            "LC": (2_030, 12_030),
            "LD": (-30, 12_030),
        },
        [
            ("EXP", "E", "0", "SE", ["P0", "P2"]),
            ("LOC", "E", "0", "SE", ["P0", "P1", "P2"]),
            ("TF", "F", "0", "SF", ["R", "R9"]),
            ("TA0", "A", "0", "SA0", ["A0", "A1"]),
            ("TA1", "A", "1", "SA1", ["M", "N0"]),
            ("TB0", "B", "0", "SB0", ["M", "Q"]),
            ("TL", "L", "", "LOOP", ["LA", "LB", "LC", "LD", "LA"]),
        ],
        {
            "SE": [(0, 0), (3_000, 0)],
            "SF": [(1_500, -500), (1_500, 1_000)],
            "SA0": [(0, 5_000), (3_000, 5_000)],
            "SA1": [(3_000, 5_060), (-1_000, 5_060)],
            "SB0": [(1_000, 2_000), (1_000, 8_000)],
            "LOOP": [(0, 10_000), (2_000, 10_000), (2_000, 12_000), (0, 12_000), (0, 10_000)],
        },
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "long_spacing_segments.csv").write_text("stale\n", encoding="utf-8")
    assert (
        _run(
            mod,
            monkeypatch,
            gtfs,
            out,
            INCLUDE_ROUTE_IDS=["A", "E", "L"],
            MIN_SPACING_FT=2_500.0,
            LONG_SPACING_FT=1_500.0,
        )
        == 0
    )

    unknown = str(mod.UNKNOWN_DIRECTION_ID)
    assert sorted(_short_pairs(out)) == [
        ("A0", "A1", 2_000),
        ("LA", "LB", 2_000),
        ("LB", "LC", 2_000),
        ("LC", "LD", 2_000),
        ("LD", "LA", 2_000),
        ("M", "N0", 1_400),
        ("P0", "P1", 1_000),
        ("P0", "P2", 2_000),
        ("P1", "P2", 1_000),
    ]
    assert sorted(_long_rows(out)) == [
        ["A", "0", "A0", "A1", "M"],
        ["E", "0", "P0", "P2", "R"],
    ]
    summary = pd.read_csv(out / "long_spacing_segments_summary.txt", sep="\t", dtype=str)
    assert summary.to_numpy().tolist() == [["A", "0"], ["E", "0"]]

    segments = mod.arcpy.FEATURES[os.path.join(out.as_posix(), "segments.shp")]["rows"]
    loop = [round(r["len_ft"]) for r in segments if r["route_id"] == "L"]
    assert loop == [2_000] * 4
    assert {r["dir"] for r in segments if r["route_id"] == "L"} == {int(unknown)}

    # A rerun with nothing long enough leaves header-only results
    assert _run(mod, monkeypatch, gtfs, out, LONG_SPACING_FT=50_000.0) == 0
    assert _long_rows(out) == []
    assert pd.read_csv(out / "long_spacing_segments_summary.txt", sep="\t").empty


def test_routes_shapefile_keeps_every_route_on_a_shared_shape(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"S0": (0, 30), "S1": (2_000, 30)},
        [("TA", "A", "0", "S", ["S0", "S1"]), ("TB", "B", "0", "S", ["S0", "S1"])],
        {"S": [(0, 0), (3_000, 0)]},
    )
    for union in (False, True):
        out = tmp_path / f"out_{union}"
        assert _run(mod, monkeypatch, gtfs, out, ROUTE_UNION=union) == 0
        rows = mod.arcpy.FEATURES[os.path.join(out.as_posix(), "routes.shp")]["rows"]
        assert sorted(r["route_id"] for r in rows) == ["A", "B"]


def test_retraced_street_uses_shape_dist_traveled(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Out along y=0 and back along y=20; every stop sits at y=25, nearer the
    # return leg, so O3 fits either pass without shape_dist_traveled.
    gtfs = _write_feed(
        tmp_path / "gtfs",
        {"O1": (500, 25), "O2": (1_500, 25), "O3": (2_500, 25), "R1": (2_000, 25)},
        [("T", "A", "0", "RT", ["O1", "O2", "O3", "R1"])],
        {"RT": [(0, 0), (3_000, 0), (3_000, 20), (0, 20)]},
        stop_dists={"T": [500, 1_500, 2_500, 4_020]},
    )
    out = tmp_path / "out"
    assert _run(mod, monkeypatch, gtfs, out, MIN_SPACING_FT=5_000.0) == 0
    assert _short_pairs(out) == [("O1", "O2", 1_000), ("O2", "O3", 1_000), ("O3", "R1", 1_520)]
