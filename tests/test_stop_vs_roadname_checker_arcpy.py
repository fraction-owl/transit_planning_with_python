"""Tests for scripts/gtfs_data_quality/stop_vs_roadname_checker_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a minimal stand-in module is installed
for the import. The pandas and matching logic is exercised directly; ``main()``
runs with its geoprocessing steps replaced by fakes.
"""

from __future__ import annotations

import importlib
import sys
import types
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

MODULE = "scripts.gtfs_data_quality.stop_vs_roadname_checker_arcpy"


@pytest.fixture()
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the script against a stand-in ``arcpy`` without leaking either module."""
    stub = types.ModuleType("arcpy")
    stub.env = types.SimpleNamespace(overwriteOutput=False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arcpy", stub)
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


def _stops(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["stop_id", "stop_name"])


# ---------------------------------------------------------------------------
# load_gtfs_stops
# ---------------------------------------------------------------------------


def test_load_gtfs_stops_keeps_literal_na_and_drops_blank_coords(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\nNA,NA,38.9,-77.0\nS2,Station, , \nS3,Main,,\n",
        encoding="utf-8",
    )
    df = mod.load_gtfs_stops(str(tmp_path))
    assert df["stop_id"].tolist() == ["NA"]
    assert df["stop_name"].tolist() == ["NA"]
    assert df["stop_lat"].tolist() == [38.9]


def test_load_gtfs_stops_reads_zipped_feed(mod: types.ModuleType, tmp_path: Path) -> None:
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("feed/stops.txt", "stop_id,stop_name,stop_lat,stop_lon\nNA,NA,38.9,-77.0\n")
    df = mod.load_gtfs_stops(str(zipped))
    assert df["stop_id"].tolist() == ["NA"]
    assert df["stop_lon"].tolist() == [-77.0]


def test_load_gtfs_stops_missing_stops_raises(mod: types.ModuleType, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="stops.txt"):
        mod.load_gtfs_stops(str(tmp_path))


# ---------------------------------------------------------------------------
# normalize_street
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("St Johns Rd", "st johns"),
        ("Johns Rd", "johns"),
        ("St Paul St", "st paul"),
        ("Mount St Albans Rd", "mount st albans"),
        ("N St Johns Rd", "n st johns"),
        ("N St NW", "n nw"),
        ("St", "st"),
    ],
)
def test_normalize_street_strips_only_last_non_leading_modifier(
    mod: types.ModuleType, name: str, expected: str
) -> None:
    assert mod.normalize_street(name, {"st", "rd"}) == expected


# ---------------------------------------------------------------------------
# split_stop_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop_name",
    ["Mainn St & Elm St", "Mainn St&Elm St", "Mainn St@Elm St", "Mainn St + Elm St"],
)
def test_split_stop_name_splits_symbols_with_or_without_spaces(
    mod: types.ModuleType, stop_name: str
) -> None:
    assert mod.split_stop_name(stop_name, {"st"}) == ["mainn", "elm"]


def test_split_stop_name_keeps_and_inside_words(mod: types.ModuleType) -> None:
    assert mod.split_stop_name("Anderson Rd and Elm St", set()) == ["anderson rd", "elm st"]


@pytest.mark.parametrize(
    "stop_name",
    [
        "Richmond Hwy at Lockhed Blvd",
        "RICHMOND HWY AT LOCKHED BLVD",
        "Intersection of Richmond Hwy and Lockhed Blvd",
        "Intersection of Richmond Hwy & Lockhed Blvd",
    ],
)
def test_split_stop_name_splits_at_and_leading_intersection_of(
    mod: types.ModuleType, stop_name: str
) -> None:
    assert mod.split_stop_name(stop_name, {"hwy", "blvd"}) == ["richmond", "lockhed"]


def test_split_stop_name_keeps_at_inside_words(mod: types.ModuleType) -> None:
    assert mod.split_stop_name("Battery Ln", {"ln"}) == ["battery"]


# ---------------------------------------------------------------------------
# detect_typos
# ---------------------------------------------------------------------------


def test_detect_typos_skips_stop_without_nearby_roads(mod: types.ModuleType) -> None:
    # Stop 9 has no roads in its buffer; it must not be compared region-wide.
    stop2roads = {"1": {"main": {"Main St"}}}
    out = mod.detect_typos(_stops(("9", "Mainn St")), stop2roads, {"st"}, 80)
    assert out.empty


def test_detect_typos_reports_only_originals_inside_buffer(mod: types.ModuleType) -> None:
    # "Main Rd" elsewhere also normalizes to "main" but is not in this buffer.
    stop2roads = {"1": {"main": {"Main St"}}}
    out = mod.detect_typos(_stops(("1", "Mainn St")), stop2roads, {"st", "rd"}, 80)
    assert out["similar_road_name_orig"].tolist() == ["Main St"]


def test_detect_typos_flags_typo_after_at(mod: types.ModuleType) -> None:
    # Unsplit, "richmond at lockhed" scored 51.9 against "lockheed" and was dropped.
    stop2roads = {"1": {"richmond": {"Richmond Hwy"}, "lockheed": {"Lockheed Blvd"}}}
    stops = _stops(("1", "Richmond Hwy at Lockhed Blvd"))
    out = mod.detect_typos(stops, stop2roads, {"hwy", "blvd"}, 80)
    assert out["street_in_stop_name"].tolist() == ["lockhed"]
    assert out["similar_road_name_orig"].tolist() == ["Lockheed Blvd"]


def test_detect_typos_flags_dropped_saint(mod: types.ModuleType) -> None:
    # Stripping every "St" made both names "mount albans", so the exact-match skip hid them.
    mods = {"st", "rd"}
    stop2roads = {"1": {mod.normalize_street("Mount Albans Rd", mods): {"Mount Albans Rd"}}}
    out = mod.detect_typos(_stops(("1", "Mount St Albans Rd")), stop2roads, mods, 80)
    assert out["street_in_stop_name"].tolist() == ["mount st albans"]
    assert out["similar_road_name_orig"].tolist() == ["Mount Albans Rd"]


def test_detect_typos_empty_result_keeps_columns(mod: types.ModuleType) -> None:
    out = mod.detect_typos(_stops(("1", "Main St")), {"1": {"main": {"Main St"}}}, {"st"}, 80)
    assert out.empty
    assert "similar_road_name_orig" in out.columns


# ---------------------------------------------------------------------------
# make_stops_fc
# ---------------------------------------------------------------------------


def test_make_stops_fc_sets_text_field_lengths(mod: types.ModuleType) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _Cursor:
        def __enter__(self) -> _Cursor:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def insertRow(self, row: list[Any]) -> None:  # noqa: N802 - arcpy API name
            pass

    arcpy = mod.arcpy
    arcpy.Exists = lambda path: False
    arcpy.SpatialReference = lambda wkid: wkid
    arcpy.management = types.SimpleNamespace(
        CreateFeatureclass=lambda *a, **k: None,
        AddField=lambda *a, **k: calls.append((a, k)),
    )
    arcpy.da = types.SimpleNamespace(InsertCursor=lambda *a, **k: _Cursor())

    long_id = "x" * 60
    df = pd.DataFrame(
        {"stop_id": ["1", long_id], "stop_name": ["A", "B"], "stop_lat": [0, 0], "stop_lon": [0, 0]}
    )
    mod.make_stops_fc(df, "work.gdb/stops_raw", 4326)

    lengths = {a[1]: k["field_length"] for a, k in calls}
    assert lengths == {"stop_id": 60, "stop_name": 255}
    assert all(len(a) == 3 for a, _ in calls)  # nothing passed as field_precision


# ---------------------------------------------------------------------------
# safe_project_or_copy
# ---------------------------------------------------------------------------


def _fake_projection(
    mod: types.ModuleType, src_name: str, src_wkid: int, project_fails: bool = False
) -> list[str]:
    """Fake the geoprocessing ``safe_project_or_copy`` uses; return the tools it runs."""
    calls: list[str] = []
    created: set[str] = set()

    class ExecuteError(Exception):
        pass

    def project(in_fc: str, out_fc: str, sr: Any) -> None:
        calls.append("Project")
        if project_fails:
            raise ExecuteError("ERROR 999999: simulated failure")
        created.add(out_fc)

    def copy(in_fc: str, out_fc: str) -> None:
        calls.append("CopyFeatures")
        created.add(out_fc)

    arcpy = mod.arcpy
    arcpy.ExecuteError = ExecuteError
    arcpy.Exists = lambda path: path in created
    arcpy.Describe = lambda path: types.SimpleNamespace(
        spatialReference=types.SimpleNamespace(name=src_name, factoryCode=src_wkid)
    )
    arcpy.SpatialReference = lambda wkid: types.SimpleNamespace(name="Target", factoryCode=wkid)
    arcpy.management = types.SimpleNamespace(
        Project=project, CopyFeatures=copy, Delete=created.discard
    )
    return calls


def test_safe_project_or_copy_projects_other_sr(mod: types.ModuleType) -> None:
    calls = _fake_projection(mod, "GCS_WGS_1984", 4326)
    mod.safe_project_or_copy("roads.shp", "work.gdb/roads_proj", 2248)
    assert calls == ["Project"]


def test_safe_project_or_copy_copies_when_already_in_target_sr(mod: types.ModuleType) -> None:
    calls = _fake_projection(mod, "NAD_1983_StatePlane_Maryland_FIPS_1900_Feet", 2248)
    mod.safe_project_or_copy("roads.shp", "work.gdb/roads_proj", 2248)
    assert calls == ["CopyFeatures"]


def test_safe_project_or_copy_failed_projection_stops(mod: types.ModuleType) -> None:
    # The old fallback copied the input unprojected and carried on.
    calls = _fake_projection(mod, "GCS_WGS_1984", 4326, project_fails=True)
    with pytest.raises(RuntimeError, match="Could not project roads.shp to WKID 2248"):
        mod.safe_project_or_copy("roads.shp", "work.gdb/roads_proj", 2248)
    assert calls == ["Project"]


def test_safe_project_or_copy_unknown_sr_stops_before_projecting(mod: types.ModuleType) -> None:
    calls = _fake_projection(mod, "Unknown", 0)
    with pytest.raises(RuntimeError, match="no defined coordinate system"):
        mod.safe_project_or_copy("roads.shp", "work.gdb/roads_proj", 2248)
    assert calls == []


# ---------------------------------------------------------------------------
# main (geoprocessing faked)
# ---------------------------------------------------------------------------


def _run_main(
    mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stop_name: str,
    col_map: dict[str, str],
) -> pd.DataFrame:
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir(exist_ok=True)
    (gtfs / "stops.txt").write_text(
        f"stop_id,stop_name,stop_lat,stop_lon\n1,{stop_name},38.9,-77.0\n", encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)

    road_names = ["Main St"]

    def fake_modifiers(fc: str, fld: str) -> set[str]:
        # Mirrors modifiers_from_roads: unique lower-cased values of ``fld``.
        values = {"FULLNAME": road_names, "RW_TYPE_US": ["St"]}[fld]
        return {v.lower() for v in values}

    def fake_candidates(join_fc: str, fullname: str, mods: set[str]) -> dict[str, Any]:
        return {"1": {mod.normalize_street(n, mods): {n} for n in road_names}}

    for name in ("make_stops_fc", "safe_project_or_copy", "buffer_fc", "spatial_join_fc"):
        monkeypatch.setattr(mod, name, lambda *a, **k: None)
    monkeypatch.setattr(mod, "create_work_gdb", lambda base: str(tmp_path / "work.gdb"))
    monkeypatch.setattr(mod, "map_road_fields", lambda fc: col_map)
    monkeypatch.setattr(mod, "modifiers_from_roads", fake_modifiers)
    monkeypatch.setattr(mod, "stop_to_candidate_roads", fake_candidates)
    monkeypatch.setattr(mod, "GTFS_FOLDER", str(gtfs))
    monkeypatch.setattr(mod, "OUTPUT_DIR", str(out_dir))

    assert mod.main() == 0
    return pd.read_csv(out_dir / mod.OUTPUT_CSV, dtype=str, keep_default_na=False)


def test_main_without_street_type_field_still_flags_typo(
    mod: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # RW_TYPE_US unmapped: modifiers used to come from FULLNAME, blanking "Main St".
    out = _run_main(mod, monkeypatch, tmp_path, "Mainn St", {"FULLNAME": "FULLNAME"})
    assert out["similar_road_name_orig"].tolist() == ["Main St"]


def test_main_clean_rerun_replaces_stale_csv(
    mod: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    col_map = {"FULLNAME": "FULLNAME", "RW_TYPE_US": "RW_TYPE_US"}
    first = _run_main(mod, monkeypatch, tmp_path, "Mainn St", col_map)
    assert first["street_in_stop_name"].tolist() == ["mainn"]

    second = _run_main(mod, monkeypatch, tmp_path, "Main St", col_map)
    assert second.empty
    assert "stop_id" in second.columns
