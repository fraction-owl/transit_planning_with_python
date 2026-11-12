"""Zipped-GTFS input for route_length_by_district_{gpd,arcpy} (arcpy stubbed).

Both scripts read GTFS_FOLDER through ``read_gtfs``; a .zip of the feed (files
in a wrapper folder) must read exactly like the extracted folder, and main()'s
input check must accept a zip that has the three files it needs.
"""

from __future__ import annotations

import importlib
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest import mock

import pandas as pd
import pytest

FIXTURE_ZIP = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"
GPD = "scripts.service_coverage.route_length_by_district_gpd"
ARCPY = "scripts.service_coverage.route_length_by_district_arcpy"


@pytest.fixture
def extracted(tmp_path: Path) -> Path:
    with zipfile.ZipFile(FIXTURE_ZIP) as zf:
        zf.extractall(tmp_path)
    return tmp_path / "dc"


@pytest.fixture
def arcpy_mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setitem(sys.modules, "arcpy", mock.MagicMock())
    sys.modules.pop(ARCPY, None)
    yield importlib.import_module(ARCPY)
    sys.modules.pop(ARCPY, None)


def test_gpd_read_gtfs_zip_matches_folder(extracted: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.import_module(GPD)
    required = {"route_id", "route_short_name"}
    monkeypatch.setattr(mod, "GTFS_FOLDER", extracted)
    from_folder = mod.read_gtfs("routes.txt", required)
    monkeypatch.setattr(mod, "GTFS_FOLDER", FIXTURE_ZIP)
    from_zip = mod.read_gtfs("routes.txt", required)
    pd.testing.assert_frame_equal(from_zip, from_folder)


def test_gpd_main_reports_zip_missing_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module(GPD)
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("feed/routes.txt", "route_id\nR1\n")
        zf.writestr("feed/trips.txt", "route_id,trip_id,shape_id\nR1,T1,S1\n")
    districts = tmp_path / "districts.shp"
    districts.write_bytes(b"")
    monkeypatch.setattr(mod, "GTFS_FOLDER", zipped)
    monkeypatch.setattr(mod, "DISTRICTS_PATH", districts)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")
    assert mod.main() == 2
    assert not (tmp_path / "out").exists()


def test_arcpy_read_gtfs_zip_matches_folder(
    arcpy_mod: ModuleType, extracted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = {"route_id", "trip_id", "shape_id"}
    monkeypatch.setattr(arcpy_mod, "GTFS_FOLDER", extracted)
    from_folder = list(arcpy_mod.read_gtfs("trips.txt", required))
    monkeypatch.setattr(arcpy_mod, "GTFS_FOLDER", FIXTURE_ZIP)
    from_zip = list(arcpy_mod.read_gtfs("trips.txt", required))
    assert from_zip == from_folder
    assert from_zip and set(from_zip[0]) >= required


def test_arcpy_read_gtfs_zip_missing_file_raises(
    arcpy_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("routes.txt", "route_id\nR1\n")
    monkeypatch.setattr(arcpy_mod, "GTFS_FOLDER", zipped)
    with pytest.raises(FileNotFoundError, match="shapes.txt"):
        list(arcpy_mod.read_gtfs("shapes.txt", {"shape_id"}))
