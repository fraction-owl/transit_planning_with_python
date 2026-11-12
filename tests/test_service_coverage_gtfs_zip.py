"""Zipped-GTFS input for the service_coverage scripts' ``_load_gtfs_tables``.

The five route-catchment scripts share one loader shape; each must read a
zipped feed (files nested in a wrapper folder, as in ``mock_gtfs_dc.zip``)
exactly as it reads the extracted folder. The arcpy twins are imported against
a stubbed ``arcpy``.
"""

from __future__ import annotations

import importlib
import sys
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from unittest import mock

import pandas as pd
import pytest

FIXTURE_ZIP = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"

MODULES = [
    "scripts.service_coverage.cabi_coverage_by_route_gpd",
    "scripts.service_coverage.school_coverage_by_route_gpd",
    "scripts.service_coverage.school_coverage_by_route_arcpy",
    "scripts.service_coverage.private_shuttle_coverage_by_route_gpd",
    "scripts.service_coverage.private_shuttle_coverage_by_route_arcpy",
]


@pytest.fixture(params=MODULES, ids=lambda name: name.rsplit(".", 1)[-1])
def target(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    name: str = request.param
    if name.endswith("_arcpy"):
        monkeypatch.setitem(sys.modules, "arcpy", mock.MagicMock())
        sys.modules.pop(name, None)
        yield importlib.import_module(name)
        sys.modules.pop(name, None)
    else:
        yield importlib.import_module(name)


@pytest.fixture
def extracted(tmp_path: Path) -> Path:
    with zipfile.ZipFile(FIXTURE_ZIP) as zf:
        zf.extractall(tmp_path)
    return tmp_path / "dc"


@pytest.mark.parametrize("need_shapes", [False, True])
def test_load_gtfs_tables_zip_matches_folder(
    target: ModuleType, extracted: Path, need_shapes: bool
) -> None:
    from_zip = target._load_gtfs_tables(FIXTURE_ZIP, need_shapes=need_shapes)
    from_folder = target._load_gtfs_tables(extracted, need_shapes=need_shapes)
    assert from_zip.keys() == from_folder.keys()
    assert ("shapes" in from_zip) is need_shapes
    for key, frame in from_folder.items():
        pd.testing.assert_frame_equal(from_zip[key], frame)


def test_load_gtfs_tables_zip_missing_file_raises(target: ModuleType, tmp_path: Path) -> None:
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("routes.txt", "route_id\nR1\n")
    with pytest.raises(FileNotFoundError, match="trips.txt"):
        target._load_gtfs_tables(zipped, need_shapes=False)


POI_MODULES = [
    "scripts.service_coverage.points_of_interest_coverage_gpd",
    "scripts.service_coverage.points_of_interest_coverage_arcpy",
]


@pytest.mark.parametrize("name", POI_MODULES, ids=lambda name: name.rsplit(".", 1)[-1])
def test_poi_load_gtfs_tables_zip_matches_folder(
    name: str, extracted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if name.endswith("_arcpy"):
        monkeypatch.setitem(sys.modules, "arcpy", mock.MagicMock())
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        monkeypatch.delitem(sys.modules, name)

        def load(path: Path) -> Mapping[str, pd.DataFrame]:
            return module._load_gtfs_tables(path)
    else:
        module = importlib.import_module(name)

        def load(path: Path) -> Mapping[str, pd.DataFrame]:
            return module._load_gtfs_tables(path, need_shapes=True)

    from_zip = load(FIXTURE_ZIP)
    from_folder = load(extracted)
    assert set(from_zip) == {"routes", "trips", "stop_times", "stops", "shapes"}
    for key, frame in from_folder.items():
        pd.testing.assert_frame_equal(from_zip[key], frame)
