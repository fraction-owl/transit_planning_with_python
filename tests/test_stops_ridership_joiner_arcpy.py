"""Tests for scripts/ridership_tools/stops_ridership_joiner_arcpy.py (arcpy stubbed)."""

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

FIXTURES = Path(__file__).parent / "fixtures"
MODULE = "scripts.ridership_tools.stops_ridership_joiner_arcpy"


@pytest.fixture
def target(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setitem(sys.modules, "arcpy", mock.MagicMock())
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


def test_resolve_gtfs_dir_returns_zip_as_is(target: ModuleType) -> None:
    assert target.resolve_gtfs_dir(r"C:\data\feed.zip") == r"C:\data\feed.zip"
    assert target.resolve_gtfs_dir(r"C:\data\stops.shp") is None


def test_resolve_stops_table_copies_stops_out_of_zip(
    target: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zipped = FIXTURES / "mock_gtfs_dc.zip"
    monkeypatch.setattr(target, "OUTPUT_FOLDER", str(tmp_path / "out"))

    table = Path(target.resolve_stops_table(str(zipped)))

    assert table == tmp_path / "out" / "stops_from_zip_tmp.txt"
    with zipfile.ZipFile(zipped) as zf:
        assert table.read_bytes() == zf.read("dc/stops.txt")


def test_read_gtfs_table_reads_zipped_feed(target: ModuleType) -> None:
    zipped = str(FIXTURES / "mock_gtfs_dc.zip")
    routes = target._read_gtfs_table(zipped, "routes.txt", ["route_id", "route_short_name"])
    assert isinstance(routes, pd.DataFrame) and not routes.empty


def test_read_gtfs_table_missing_file_exits(target: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        target._read_gtfs_table(str(tmp_path), "shapes.txt", ["shape_id"])
