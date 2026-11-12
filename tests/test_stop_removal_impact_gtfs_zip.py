"""Zipped-GTFS input for stop_removal_impact_{gpd,arcpy}.load_gtfs_stops (arcpy stubbed)."""

from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

FIXTURE_ZIP = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"


def test_gpd_load_gtfs_stops_zip_matches_folder(tmp_path: Path) -> None:
    from scripts.stop_analysis.stop_removal_impact_gpd import load_gtfs_stops

    with zipfile.ZipFile(FIXTURE_ZIP) as zf:
        zf.extractall(tmp_path)
    from_folder = load_gtfs_stops(tmp_path / "dc", 2248)
    from_zip = load_gtfs_stops(FIXTURE_ZIP, 2248)
    pd.testing.assert_frame_equal(pd.DataFrame(from_zip), pd.DataFrame(from_folder))
    assert from_zip.crs == from_folder.crs


def test_arcpy_load_gtfs_stops_missing_stops_in_zip_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "scripts.stop_analysis.stop_removal_impact_arcpy"
    monkeypatch.setitem(sys.modules, "arcpy", mock.MagicMock())
    monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module(name)
    monkeypatch.delitem(sys.modules, name)

    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("routes.txt", "route_id\nR1\n")
    with pytest.raises(OSError, match="stops.txt not found"):
        mod.load_gtfs_stops(str(zipped), mock.MagicMock())
