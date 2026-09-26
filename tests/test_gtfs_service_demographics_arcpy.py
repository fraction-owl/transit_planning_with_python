"""Tests for the schema detection in scripts/service_coverage/gtfs_service_demographics_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a stand-in module answers
``arcpy.ListFields`` from an in-memory field list; no geoprocessing runs.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Iterator

import pytest

MODULE = "scripts.service_coverage.gtfs_service_demographics_arcpy"

_BASE_FIELDS = [
    "POP_TOT",
    "HH_TOT",
    "HH_LOWINC",
    "PCT_LOWINC",
    "MINOR_CNT",
    "PCT_MINOR",
    "EMP_LO",
    "EMP_TOT",
]
_LAYERS = {
    "allocated.shp": [*_BASE_FIELDS, "CNT_ALLOC"],
    "legacy.shp": _BASE_FIELDS,
}


@pytest.fixture()
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the script against a stand-in ``arcpy`` without leaking either module."""
    stub = types.ModuleType("arcpy")
    stub.ListFields = lambda dataset: [  # type: ignore[attr-defined]
        types.SimpleNamespace(name=name) for name in _LAYERS[dataset]
    ]
    monkeypatch.setitem(sys.modules, "arcpy", stub)
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


def test_allocated_layer_uses_block_counts(mod: types.ModuleType) -> None:
    # Block-allocated counts are area-weighted directly, matching the GeoPandas
    # pipeline's count allocation.
    schema = mod.detect_demog_schema("allocated.shp")
    assert schema.counts_allocated
    assert schema.strategies["loinc_hh"] == ("count", "HH_LOWINC")
    assert schema.strategies["minor_pop"] == ("count", "MINOR_CNT")


def test_legacy_layer_falls_back_to_rate_times_total(
    mod: types.ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    # A layer without the marker repeats whole-tract counts on each block, so the
    # counts must not be area-weighted; the tract rate x block total is used instead.
    with caplog.at_level(logging.WARNING):
        schema = mod.detect_demog_schema("legacy.shp")
        mod.detect_demog_schema("legacy.shp")
    assert not schema.counts_allocated
    assert schema.strategies["loinc_hh"] == ("derived", ("PCT_LOWINC", "HH_TOT"))
    assert schema.strategies["all_jobs"] == ("count", "EMP_TOT")  # no rate: count
    assert caplog.text.count("predates the block count split") == 1
