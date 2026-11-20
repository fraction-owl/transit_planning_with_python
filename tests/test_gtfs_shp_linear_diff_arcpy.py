"""Tests for scripts/stop_analysis/gtfs_shp_linear_diff_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a stand-in module backed by shapely,
pyproj and GeoPandas answers the calls the script makes (polyline overlay,
projection, Describe/ListFields/SearchCursor on a real shapefile) and keeps
output feature classes in memory. The end-to-end tests also run the GeoPandas
twin on the same inputs and check that the two agree.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry import Point as ShapelyPoint

MODULE = "scripts.stop_analysis.gtfs_shp_linear_diff_arcpy"
GPD_MODULE = "scripts.stop_analysis.gtfs_shp_linear_diff_gpd"

# ---------------------------------------------------------------------------
# Stand-in arcpy
# ---------------------------------------------------------------------------


def _lines_only(geom: Any) -> Any:
    """Keep the linear parts of an overlay result, as arcpy's polyline ops do."""
    if geom.is_empty:
        return MultiLineString()
    if isinstance(geom, (LineString, MultiLineString)):
        return geom
    if isinstance(geom, GeometryCollection):
        parts: list[LineString] = []
        for part in geom.geoms:
            sub = _lines_only(part)
            parts.extend(getattr(sub, "geoms", [sub]) if not sub.is_empty else [])
        return MultiLineString(parts)
    return MultiLineString()


def _fake_arcpy() -> types.ModuleType:
    features: dict[str, dict[str, Any]] = {}

    class ExecuteError(Exception):
        pass

    class SpatialReference:
        def __init__(self, wkid: int | None = None, crs: CRS | None = None) -> None:
            try:
                self._crs = crs if crs is not None else CRS.from_epsg(int(wkid or 0))
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
        pass

    def _project(geom: Any, src: SpatialReference, dst: SpatialReference) -> Any:
        t = Transformer.from_crs(src._crs, dst._crs, always_xy=True)
        return ops.transform(t.transform, geom)

    class PointGeometry:
        def __init__(self, pt: Point, sr: SpatialReference) -> None:
            self._g, self.spatialReference = ShapelyPoint(pt.X, pt.Y), sr

        def projectAs(self, sr: SpatialReference) -> PointGeometry:
            out = PointGeometry(Point(), sr)
            out._g = _project(self._g, self.spatialReference, sr)
            return out

        @property
        def firstPoint(self) -> Point:
            return Point(self._g.x, self._g.y)

    class Polygon:
        def __init__(self, geom: Any, sr: SpatialReference) -> None:
            self._g, self.spatialReference = geom, sr

        def union(self, other: Polygon) -> Polygon:
            return Polygon(self._g.union(other._g), self.spatialReference)

    class Polyline:
        def __init__(self, array: Sequence[Any] | None, sr: SpatialReference) -> None:
            self.spatialReference = sr
            items = list(array or [])
            if items and isinstance(items[0], Array):
                self._g = MultiLineString([[(p.X, p.Y) for p in part] for part in items])
            else:
                self._g = LineString([(p.X, p.Y) for p in items])

        @classmethod
        def _wrap(cls, geom: Any, sr: SpatialReference) -> Polyline:
            out = cls(None, sr)
            out._g = _lines_only(geom)
            return out

        def _parts(self) -> list[LineString]:
            if self._g.is_empty:
                return []
            return list(getattr(self._g, "geoms", [self._g]))

        @property
        def length(self) -> float:
            return self._g.length

        @property
        def extent(self) -> types.SimpleNamespace:
            xmin, ymin, xmax, ymax = self._g.bounds
            return types.SimpleNamespace(XMin=xmin, YMin=ymin, XMax=xmax, YMax=ymax)

        @property
        def partCount(self) -> int:
            return len(self._parts())

        def getPart(self, index: int) -> Array:
            return Array(Point(x, y) for x, y in self._parts()[index].coords)

        def projectAs(self, sr: SpatialReference) -> Polyline:
            return Polyline._wrap(_project(self._g, self.spatialReference, sr), sr)

        def buffer(self, distance: float) -> Polygon:
            return Polygon(self._g.buffer(distance), self.spatialReference)

        def difference(self, other: Any) -> Polyline:
            return Polyline._wrap(self._g.difference(other._g), self.spatialReference)

        def intersect(self, other: Any, dimension: int) -> Polyline:
            assert dimension == 2
            return Polyline._wrap(self._g.intersection(other._g), self.spatialReference)

        def union(self, other: Polyline) -> Polyline:
            return Polyline._wrap(self._g.union(other._g), self.spatialReference)

    def _read(path: str) -> gpd.GeoDataFrame:
        return gpd.read_file(path)

    def describe(path: str) -> types.SimpleNamespace:
        layer = _read(path)
        kinds = set(layer.geom_type.dropna())
        shape_type = "Polyline" if kinds <= {"LineString", "MultiLineString"} else "Polygon"
        sr = SpatialReference(crs=CRS(layer.crs)) if layer.crs is not None else None
        return types.SimpleNamespace(shapeType=shape_type, spatialReference=sr)

    def list_fields(path: str) -> list[types.SimpleNamespace]:
        layer = _read(path)
        names = ["FID", "Shape", *[c for c in layer.columns if c != layer.geometry.name]]
        return [types.SimpleNamespace(name=n) for n in names]

    class SearchCursor:
        def __init__(
            self, path: str, fields: list[str], spatial_reference: SpatialReference
        ) -> None:
            layer = _read(path)
            src = SpatialReference(crs=CRS(layer.crs))
            self._rows = []
            for _, rec in layer.iterrows():
                values: list[Any] = []
                for field in fields:
                    if field == "SHAPE@":
                        geom = rec.geometry
                        values.append(
                            None
                            if geom is None
                            else Polyline._wrap(
                                _project(geom, src, spatial_reference), spatial_reference
                            )
                        )
                    else:
                        values.append(rec[field])
                self._rows.append(tuple(values))

        def __enter__(self) -> SearchCursor:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def __iter__(self) -> Iterator[tuple[Any, ...]]:
            return iter(self._rows)

    class InsertCursor:
        def __init__(self, path: str, fields: list[str]) -> None:
            self.rows = features[path]["rows"]
            self.fields = fields

        def __enter__(self) -> InsertCursor:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def insertRow(self, row: Sequence[Any]) -> None:
            self.rows.append(dict(zip(self.fields, row)))

    def create_feature_class(folder: str, name: str, *_: Any, **kwargs: Any) -> None:
        features[os.path.join(folder, name)] = {"rows": [], "sr": kwargs.get("spatial_reference")}

    arcpy = types.ModuleType("arcpy")
    members = {
        "ExecuteError": ExecuteError,
        "SpatialReference": SpatialReference,
        "Point": Point,
        "Array": Array,
        "Polyline": Polyline,
        "PointGeometry": PointGeometry,
        "Polygon": Polygon,
        "Describe": describe,
        "ListFields": list_fields,
        "env": types.SimpleNamespace(overwriteOutput=False),
        "Exists": lambda path: path in features or os.path.exists(path),
        "management": types.SimpleNamespace(
            Delete=lambda path: features.pop(path, None),
            CreateFeatureclass=create_feature_class,
            AddField=lambda *a, **k: None,
        ),
        "da": types.SimpleNamespace(InsertCursor=InsertCursor, SearchCursor=SearchCursor),
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
# Synthetic network (EPSG:2248 feet around a DC origin)
# ---------------------------------------------------------------------------

_X0, _Y0 = 1_300_000.0, 450_000.0
_TO_LL = Transformer.from_crs("EPSG:2248", "EPSG:4326", always_xy=True)

# Route 10's westbound trips run a one-way couplet 600 ft north; the short turn
# and Saturday detour should not be picked on a weekday. Route 7 has no
# direction_id. Route 30 has no shape; 40 is only in GTFS; 99 only in the layer.
SHAPES = {
    "S10_0": [(0, 0), (10_000, 0)],
    "S10_0_SHORT": [(0, 0), (5_000, 0)],
    "S10_1": [(10_000, 20), (7_000, 20), (7_000, 600), (4_000, 600), (4_000, 20), (0, 20)],
    "S10_SAT": [(0, 0), (0, -3_000), (10_000, -3_000)],
    "S7_E": [(0, 5_000), (8_000, 5_000)],
    "S7_W": [(8_000, 5_015), (0, 5_015)],
    "S40": [(0, -8_000), (6_000, -8_000)],
}
TRIP_SPECS = [
    ("R10", "WK", "0", "East", "S10_0", 20),
    ("R10", "WK", "0", "East", "S10_0_SHORT", 5),
    ("R10", "WK", "1", "West", "S10_1", 18),
    ("R10", "SAT", "0", "East", "S10_SAT", 30),
    ("R10", "SAT", "1", "West", "S10_1", 30),
    ("R7", "WK", "", "East", "S7_E", 12),
    ("R7", "WK", "", "West", "S7_W", 12),
    ("R30", "WK", "0", "North", "", 8),
    ("R40", "WK", "0", "South", "S40", 6),
]
ROUTES = [("R10", "10"), ("R7", "7"), ("R30", "30"), ("R40", "40")]
REFERENCE = [
    ("10", [(0, 0), (10_000, 0)]),
    ("10", [(7_000, 0), (7_000, 600), (4_000, 600), (4_000, 0)]),
    (
        "007",
        [
            (0, 5_000),
            (3_000, 5_000),
            (3_000, 6_000),
            (6_000, 6_000),
            (6_000, 5_000),
            (8_000, 5_000),
        ],
    ),
    ("30", [(0, 9_000), (5_000, 9_000)]),
    ("99", [(0, 20_000), (5_000, 20_000)]),
]


def _abs(points: Sequence[tuple[float, float]]) -> LineString:
    return LineString([(_X0 + x, _Y0 + y) for x, y in points])


def _lat_lon(x: float, y: float) -> str:
    lon, lat = _TO_LL.transform(_X0 + x, _Y0 + y)
    return f"{lat:.9f},{lon:.9f}"


def write_feed(root: Path, shapes: Mapping[str, Sequence[tuple[float, float]]] = SHAPES) -> Path:
    root.mkdir(parents=True)
    (root / "routes.txt").write_text(
        "route_id,route_short_name\n" + "".join(f"{r},{s}\n" for r, s in ROUTES), encoding="utf-8"
    )
    rows = [
        f"{rid}_{sid}_{shape}_{i},{rid},{sid},{d},{h},{shape}\n"
        for rid, sid, d, h, shape, n in TRIP_SPECS
        for i in range(n)
    ]
    (root / "trips.txt").write_text(
        "trip_id,route_id,service_id,direction_id,trip_headsign,shape_id\n" + "".join(rows),
        encoding="utf-8",
    )
    (root / "shapes.txt").write_text(
        "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
        + "".join(
            f"{sid},{i + 1},{_lat_lon(x, y)}\n"
            for sid, pts in shapes.items()
            for i, (x, y) in enumerate(pts)
        ),
        encoding="utf-8",
    )
    (root / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "WK,1,1,1,1,1,0,0,20260105,20260327\n"
        "SAT,0,0,0,0,0,1,0,20260105,20260327\n",
        encoding="utf-8",
    )
    return root


def write_reference(
    path: Path, features: Sequence[tuple[Any, Sequence[tuple[float, float]]]], field: str = "ROUTE"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {field: [name for name, _ in features]},
        geometry=[_abs(pts) for _, pts in features],
        crs="EPSG:2248",
    ).to_file(path)
    return path


def route_table(out: Path) -> pd.DataFrame:
    return pd.read_csv(
        out / "gtfs_shp_route_comparison.csv",
        dtype={"route_key": str, "route_label": str},
        keep_default_na=False,
    ).set_index("route_key")


def run_main(mod: types.ModuleType, gtfs: Path, shapes: Path, out: Path, *extra: str) -> int:
    return mod.main(
        ["--gtfs", str(gtfs), "--route-shapes", str(shapes), "--output-dir", str(out), *extra]
    )


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


def test_merge_line_parts_joins_end_to_end_in_either_direction(mod: types.ModuleType) -> None:
    parts = [[(0, 0), (1, 0)], [(2, 0), (1, 0)], [(2, 0), (3, 0)]]
    assert mod.merge_line_parts(parts) == [[(0, 0), (1, 0), (2, 0), (3, 0)]]


def test_merge_line_parts_stops_at_junctions(mod: types.ModuleType) -> None:
    # Three parts meet at (1, 0), so none is joined there.
    parts = [[(0, 0), (1, 0)], [(1, 0), (2, 0)], [(1, 0), (1, 1)]]
    assert len(mod.merge_line_parts(parts)) == 3
    assert mod.merge_line_parts([[(0, 0)], []]) == []


def test_extract_lines_merges_split_overlay_parts(mod: types.ModuleType) -> None:
    sr = mod.arcpy.SpatialReference(2248)
    split = mod.arcpy.Polyline(
        mod.arcpy.Array(
            [
                mod.arcpy.Array([mod.arcpy.Point(0, 0), mod.arcpy.Point(100, 0)]),
                mod.arcpy.Array([mod.arcpy.Point(100, 0), mod.arcpy.Point(200, 0)]),
            ]
        ),
        sr,
    )
    lines = mod.extract_lines(split)
    assert len(lines) == 1 and lines[0].length == pytest.approx(200.0)
    assert mod.extract_lines(None) == []
    assert mod.filter_short_lines(lines, 250.0) == []


def test_measure_lines_counts_a_two_way_street_once(mod: types.ModuleType) -> None:
    sr = mod.arcpy.SpatialReference(2248)
    out = mod._polyline([(0, 0), (10_000, 0)], sr)
    back = mod._polyline([(10_000, 20), (0, 20)], sr)
    kept = mod.measure_lines([out, back], 80.0, 150.0)
    assert sum(line.length for line in kept) == pytest.approx(10_000.0)
    assert mod.measure_lines([], 80.0) == []


def test_chain_lines_joins_segments(mod: types.ModuleType) -> None:
    sr = mod.arcpy.SpatialReference(2248)
    segments = [mod._polyline([(x, 0), (x + 100, 0)], sr) for x in range(0, 1_000, 100)]
    chained = mod.chain_lines(segments[::-1])
    assert len(chained) == 1 and chained[0].length == pytest.approx(1_000.0)


def test_divided_road_lengths_reconcile(mod: types.ModuleType) -> None:
    sr = mod.arcpy.SpatialReference(2248)
    weave = [(x * 120, 60 if i % 2 == 0 else 100) for i, x in enumerate(range(51))]
    gtfs = [mod._polyline([(0, 0), (6_000, 0)], sr), mod._polyline(weave[::-1], sr)]
    median = [mod._polyline([(0, 40), (6_000, 40)], sr)]
    lengths = {
        name: sum(line.length for line in lines)
        for name, lines in mod.compare_alignments(gtfs, median, 80.0, 150.0).items()
    }
    total = lengths["matched"] + lengths["gtfs_only"] + lengths["gtfs_short"]
    assert total == pytest.approx(6_000, rel=0.02) and lengths["matched"] <= total


def test_spatial_reference_rejects_unusable_wkids(mod: types.ModuleType) -> None:
    assert mod.crs_feet_per_unit(mod._spatial_reference(2248)) == pytest.approx(1.000002, abs=1e-6)
    assert mod.crs_feet_per_unit(mod._spatial_reference(32618)) == pytest.approx(3.28084, abs=1e-5)
    with pytest.raises(mod.ConfigError, match="projected"):
        mod._spatial_reference(4326)
    with pytest.raises(mod.ConfigError, match="arcpy knows"):
        mod._spatial_reference(999_999)


def test_read_reference_layers_rejects_zip_and_gdb(mod: types.ModuleType, tmp_path: Path) -> None:
    with pytest.raises(mod.ConfigError, match="unzip"):
        mod.read_reference_layers(tmp_path / "routes.zip")
    with pytest.raises(mod.ConfigError, match="feature class"):
        mod.read_reference_layers(tmp_path / "data.gdb")
    with pytest.raises(OSError, match="does not exist"):
        mod.read_reference_layers(tmp_path / "missing.shp")


# ---------------------------------------------------------------------------
# End to end, checked against the GeoPandas twin
# ---------------------------------------------------------------------------


def test_run_matches_the_geopandas_twin(mod: types.ModuleType, tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    shp = write_reference(tmp_path / "ref" / "routes.shp", REFERENCE)
    out = tmp_path / "out"
    assert run_main(mod, gtfs, shp, out, "--shp-route-field", "route") == 0

    table = route_table(out)
    assert table["status"].to_dict() == {
        "7": "mismatch",
        "40": "missing_from_shp",
        "99": "missing_from_gtfs",
        "30": "no_gtfs_geometry",
        "10": "match",
    }
    assert table.loc["10", "shape_ids"] == "S10_0;S10_1"
    assert table.loc["10", "gtfs_len_ft"] == pytest.approx(14_000, rel=0.02)

    gpd_mod = importlib.import_module(GPD_MODULE)
    twin_out = tmp_path / "twin"
    twin_args = ["--shp-route-field", "route", "--target-crs", "EPSG:2248"]
    assert run_main(gpd_mod, gtfs, shp, twin_out, *twin_args) == 0
    twin = route_table(twin_out)
    assert list(table.index) == list(twin.index)
    assert table["status"].to_dict() == twin["status"].to_dict()
    assert table["shape_ids"].to_dict() == twin["shape_ids"].to_dict()
    for col in (
        "gtfs_len_ft",
        "matched_len_ft",
        "gtfs_only_len_ft",
        "gtfs_short_len_ft",
        "shp_len_ft",
        "shp_matched_len_ft",
        "shp_only_len_ft",
        "shp_short_len_ft",
    ):
        assert table[col].to_numpy() == pytest.approx(twin[col].to_numpy(), rel=0.01, abs=5.0)

    patterns = pd.read_csv(out / "gtfs_shp_selected_patterns.csv", dtype=str, keep_default_na=False)
    twin_patterns = pd.read_csv(
        twin_out / "gtfs_shp_selected_patterns.csv", dtype=str, keep_default_na=False
    )
    cols = ["route_id", "direction", "shape_id", "trips", "direction_trips", "day_basis"]
    assert patterns[cols].to_numpy().tolist() == twin_patterns[cols].to_numpy().tolist()

    features = mod.arcpy.FEATURES
    gtfs_only = features[str(out / "gtfs_shp_segments_gtfs_only.shp")]
    assert gtfs_only["sr"].name == "WGS_84"
    assert {row["rt_status"] for row in gtfs_only["rows"]} == {"mismatch", "missing_from_shp"}
    shp_only = features[str(out / "gtfs_shp_segments_shp_only.shp")]["rows"]
    assert {row["rt_key"] for row in shp_only} == {"7", "99", "30"}  # 30 has no GTFS shape
    combined = features[str(out / "gtfs_shp_combined_routes.shp")]["rows"]
    assert sorted(row["rt_key"] for row in combined) == ["10", "40", "7"]
    assert all(isinstance(row["n_patterns"], int) for row in combined)
    selected = features[str(out / "gtfs_shp_selected_patterns.shp")]["rows"]
    assert len(selected) == len(patterns)

    summary = (out / "gtfs_shp_summary.json").read_text(encoding="utf-8")
    assert "WKID 2248" in summary
    log = (out / "gtfs_shp_linear_diff_runlog.txt").read_text(encoding="utf-8")
    assert "PROJECTED_CRS_WKID: int = 2248" in log


def test_per_route_folder_named_by_file(mod: types.ModuleType, tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    folder = tmp_path / "per_route"
    write_reference(folder / "Route_10_a.shp", [("x", REFERENCE[0][1])], field="NOTE")
    write_reference(folder / "Route_10_b.shp", [("x", REFERENCE[1][1])], field="NOTE")
    write_reference(folder / "Route_7.shp", [("x", SHAPES["S7_E"])], field="NOTE")
    out = tmp_path / "out"
    assert run_main(mod, gtfs, folder, out, "--shp-route-pattern", r"(?i)route_([0-9]+)") == 0
    table = route_table(out)
    assert table.loc["10", "status"] == "match"
    assert table.loc["10", "shp_sources"] == "Route_10_a;Route_10_b"
    assert table.loc["7", "status"] == "match"


def test_main_returns_2_for_configuration_problems(mod: types.ModuleType, tmp_path: Path) -> None:
    assert mod.main([]) == 2  # placeholders
    gtfs = write_feed(tmp_path / "gtfs")
    shp = write_reference(tmp_path / "ref" / "routes.shp", REFERENCE)
    out = tmp_path / "out"
    assert run_main(mod, gtfs, shp, out, "--shp-route-field", "LINE") == 2
    assert (
        run_main(mod, gtfs, shp, out, "--shp-route-field", "ROUTE", "--projected-wkid", "4326") == 2
    )
    assert run_main(mod, gtfs, tmp_path / "ref.zip", out, "--shp-route-field", "ROUTE") == 2
    assert run_main(mod, gtfs, shp, tmp_path / "out.gdb", "--shp-route-field", "ROUTE") == 2


def test_main_returns_1_for_missing_inputs(mod: types.ModuleType, tmp_path: Path) -> None:
    shp = write_reference(tmp_path / "ref" / "routes.shp", REFERENCE)
    out = tmp_path / "out"
    assert run_main(mod, tmp_path / "missing", shp, out, "--shp-route-field", "ROUTE") == 1


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


def _chop(points: Sequence[tuple[float, float]], step: float) -> list[list[tuple[float, float]]]:
    """Cut a line into pieces of *step* length (the last one shorter)."""
    line = LineString(points)
    marks = [*range(0, int(line.length), int(step)), line.length]
    return [
        [line.interpolate(a).coords[0], line.interpolate(b).coords[0]]
        for a, b in zip(marks, marks[1:])
    ]


def test_compare_alignments_same_linework_never_differs(mod: types.ModuleType) -> None:
    sr = mod.arcpy.SpatialReference(2248)
    out = [(0, 0), (3_000, 0), (3_000, 2_000), (6_000, 2_000)]
    back = [(6_000, 30), (3_030, 30), (3_030, -500), (2_000, -500), (2_000, 300), (0, 300)]
    gtfs = [mod._polyline(out, sr), mod._polyline(back, sr)]
    pieces = [mod._polyline(p, sr) for p in _chop(back, 100) + _chop(out, 100)]
    classes = mod.compare_alignments(gtfs, pieces, 80.0, 150.0)
    assert classes["gtfs_only"] == [] and classes["shp_only"] == []

    low, mid, high = (mod._polyline([(0, y), (5_000, y)], sr) for y in (0, 70, 140))
    for reference in ([mid, high], [high, mid]):
        classes = mod.compare_alignments([low, mid], reference, 80.0, 150.0)
        assert classes["gtfs_only"] == [] and classes["shp_only"] == []


def test_rerun_clears_old_layers_and_logs_settings(mod: types.ModuleType, tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    out = tmp_path / "out"
    first = write_reference(tmp_path / "ref1" / "routes.shp", REFERENCE)
    assert run_main(mod, gtfs, first, out, "--shp-route-field", "ROUTE") == 0
    features = mod.arcpy.FEATURES
    gtfs_only = str(out / "gtfs_shp_segments_gtfs_only.shp")
    shp_only = str(out / "gtfs_shp_segments_shp_only.shp")
    assert gtfs_only in features and shp_only in features

    same = [
        (name, piece)
        for name, shape_ids in (
            ("10", ("S10_0", "S10_1")),
            ("7", ("S7_E", "S7_W")),
            ("40", ("S40",)),
        )
        for sid in shape_ids
        for piece in _chop(SHAPES[sid], 300)
    ]
    second = write_reference(tmp_path / "ref2" / "routes.shp", same)
    assert (
        run_main(mod, gtfs, second, out, "--shp-route-field", "ROUTE", "--buffer-feet", "90") == 0
    )
    table = route_table(out)
    assert table["status"].to_dict() == {
        "30": "no_gtfs_geometry",
        "10": "match",
        "7": "match",
        "40": "match",
    }
    assert (table["gtfs_only_len_ft"] == 0).all() and (table["shp_only_len_ft"] == 0).all()
    assert gtfs_only not in features and shp_only not in features
    log = (out / "gtfs_shp_linear_diff_runlog.txt").read_text(encoding="utf-8")
    assert "BUFFER_TOLERANCE_FEET = 90.0" in log and "SHP_ROUTE_FIELD = 'ROUTE'" in log
    assert "PROJECTED_CRS_WKID = 2248" in log


def test_aliases_and_km_fields(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    features = [REFERENCE[0], REFERENCE[1], ("7 AM", SHAPES["S7_E"]), ("7 PM", SHAPES["S7_W"])]
    shp = write_reference(tmp_path / "ref" / "routes.shp", features)
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "DISTANCE_OUTPUT_UNIT", "km")
    args = [
        "--shp-route-field",
        "ROUTE",
        "--shp-route-alias",
        "7 AM=7",
        "--shp-route-alias",
        "7 PM=7",
    ]
    assert run_main(mod, gtfs, shp, out, *args) == 0
    table = route_table(out)
    assert table.loc["7", "status"] == "match" and table.loc["7", "shp_route_names"] == "7 AM;7 PM"
    assert "matched_len_km" in table.columns
    rows = mod.arcpy.FEATURES[str(out / "gtfs_shp_combined_routes.shp")]["rows"]
    assert "len_km" in rows[0] and "len_mi" not in rows[0]


def test_maps_match_the_geopandas_twin(mod: types.ModuleType, tmp_path: Path) -> None:
    gtfs = write_feed(tmp_path / "gtfs")
    shp = write_reference(tmp_path / "ref" / "routes.shp", REFERENCE)
    out, twin_out = tmp_path / "out", tmp_path / "twin"
    args = ["--shp-route-field", "ROUTE", "--write-pngs", "--plot-dpi", "72"]
    assert run_main(mod, gtfs, shp, out, *args) == 0
    gpd_mod = importlib.import_module(GPD_MODULE)
    assert run_main(gpd_mod, gtfs, shp, twin_out, *args, "--target-crs", "EPSG:2248") == 0
    manifests = [
        pd.read_csv(folder / "comparison_pngs" / "plots_manifest.csv", dtype=str)
        for folder in (out, twin_out)
    ]
    assert manifests[0]["png_file"].tolist() == manifests[1]["png_file"].tolist()
    assert manifests[0]["route_key"].tolist() == ["30", "40", "7", "99"]  # key order
    for name in manifests[0]["png_file"]:
        assert (out / "comparison_pngs" / name).read_bytes()[:4] == b"\x89PNG"
    arc = manifests[0].set_index("route_key")["detail_centers_lat_lon"]
    twin = manifests[1].set_index("route_key")["detail_centers_lat_lon"]
    for key in arc.index:
        a = [tuple(map(float, c.split(","))) for c in arc[key].split("; ")]
        b = [tuple(map(float, c.split(","))) for c in twin[key].split("; ")]
        assert len(a) == len(b)
        assert all(abs(x[0] - y[0]) < 0.002 and abs(x[1] - y[1]) < 0.002 for x, y in zip(a, b))
