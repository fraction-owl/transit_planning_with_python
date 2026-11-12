"""Tests for scripts/gtfs_data_quality/stop_v_conflict_checker_arcpy.py (arcpy stubbed)."""

from __future__ import annotations

import importlib
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
MODULE = "scripts.gtfs_data_quality.stop_v_conflict_checker_arcpy"


@pytest.fixture
def target(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setitem(sys.modules, "arcpy", mock.MagicMock())
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


def test_resolve_stops_path_copies_stops_out_of_zipped_feed(
    target: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zipped = FIXTURES / "mock_gtfs_dc.zip"
    monkeypatch.setattr(target, "STOPS_TXT_PATH", "")
    monkeypatch.setattr(target, "GTFS_DIR", str(zipped))
    out_dir = tmp_path / "out"

    resolved = Path(target._resolve_stops_path(out_dir))

    assert resolved == out_dir / "stops_from_zip_tmp.txt"
    with zipfile.ZipFile(zipped) as zf:
        assert resolved.read_bytes() == zf.read("dc/stops.txt")


def test_resolve_stops_path_folder_returns_stops_txt(
    target: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stops.txt").write_text("stop_id,stop_lat,stop_lon\n1,38.9,-77.0\n")
    monkeypatch.setattr(target, "STOPS_TXT_PATH", "")
    monkeypatch.setattr(target, "GTFS_DIR", str(tmp_path))
    assert target._resolve_stops_path(tmp_path / "out") == str(tmp_path / "stops.txt")


def test_resolve_stops_path_zip_without_stops_raises(
    target: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("routes.txt", "route_id\n1\n")
    monkeypatch.setattr(target, "STOPS_TXT_PATH", "")
    monkeypatch.setattr(target, "GTFS_DIR", str(zipped))
    with pytest.raises(FileNotFoundError, match="stops.txt"):
        target._resolve_stops_path(tmp_path / "out")
