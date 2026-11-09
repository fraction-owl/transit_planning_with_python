"""Tests for scripts/service_coverage/points_of_interest_coverage_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a stand-in module answers the message
calls the GTFS loader makes. No geoprocessing runs.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

MODULE = "scripts.service_coverage.points_of_interest_coverage_arcpy"


@pytest.fixture()
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the script against a stand-in ``arcpy`` without leaking either module."""
    stub = types.ModuleType("arcpy")
    for name in ("AddMessage", "AddWarning", "AddError"):
        setattr(stub, name, lambda msg: None)
    monkeypatch.setitem(sys.modules, "arcpy", stub)
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


def test_load_gtfs_tables_reads_ids_as_text(mod: types.ModuleType, tmp_path: Path) -> None:
    """Ids keep leading zeros, and stop_id has one type in stops.txt and stop_times.txt."""
    (tmp_path / "routes.txt").write_text("route_id\n10\n")
    (tmp_path / "trips.txt").write_text("route_id,trip_id,shape_id\n10,1,12345\n10,2,\n")
    (tmp_path / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\n1,001,1\n1,002,2\n")
    (tmp_path / "stops.txt").write_text(
        "stop_id,stop_lat,stop_lon,parent_station\n"
        "001,38.90,-77.03,\n002,38.91,-77.02,STN1\nSTN1,38.91,-77.02,\n"
    )
    (tmp_path / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        "12345,38.90,-77.03,1\n12345,38.91,-77.02,2\n"
    )

    tables = mod._load_gtfs_tables(tmp_path)

    assert tables["trips"]["route_id"].tolist() == ["10", "10"]
    assert tables["stop_times"]["stop_id"].tolist() == ["001", "002"]
    assert set(tables["stop_times"]["stop_id"]) <= set(tables["stops"]["stop_id"])
    assert pd.api.types.is_numeric_dtype(tables["shapes"]["shape_pt_sequence"])
