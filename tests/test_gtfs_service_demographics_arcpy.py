"""Tests for scripts/service_coverage/gtfs_service_demographics_arcpy.py.

Covers the demographics schema detection and the service-day selection. ArcPy is
unavailable outside ArcGIS Pro, so a stand-in module answers ``arcpy.ListFields``
from an in-memory field list and ``arcpy.Exists`` with True; no geoprocessing runs.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
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
    stub.Exists = lambda path: True  # type: ignore[attr-defined]
    stub.env = types.SimpleNamespace()  # type: ignore[attr-defined]
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


# ---------------------------------------------------------------------------
# Service-day selection
# ---------------------------------------------------------------------------

# Route 101 runs on weekday service "4" and Saturday service "5"; route 202 only on "5".
_FEED = {
    "routes.txt": "route_id,route_short_name\nR1,101\nR2,202\n",
    "trips.txt": "route_id,service_id,trip_id\nR1,4,T1\nR1,5,T2\nR2,5,T3\n",
    "stop_times.txt": "trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\nT2,S3,1\nT3,S4,1\n",
    "stops.txt": (
        "stop_id,stop_name,stop_lat,stop_lon\n"
        "S1,A,38.90,-77.03\nS2,B,38.91,-77.03\nS3,C,38.92,-77.03\nS4,D,38.93,-77.03\n"
    ),
    "calendar.txt": (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "4,1,1,1,1,1,0,0,20260101,20261231\n"
        "5,0,0,0,0,0,1,0,20260101,20261231\n"
    ),
}


@pytest.fixture()
def feed(tmp_path: Path) -> Path:
    for name, content in _FEED.items():
        (tmp_path / name).write_text(content)
    return tmp_path


def _stop_ids(
    mod: types.ModuleType, feed: Path, routes: list[str], service_ids: list[str]
) -> list[str]:
    gtfs = mod.load_gtfs_data(
        str(feed), files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
    )
    stops = mod._filter_gtfs_stops_by_route_short_name(gtfs, routes, service_ids)
    return sorted(stops["stop_id"])


def test_stop_filter_uses_service_day(mod: types.ModuleType, feed: Path) -> None:
    assert _stop_ids(mod, feed, ["101"], ["4"]) == ["S1", "S2"]
    assert _stop_ids(mod, feed, ["101"], ["5"]) == ["S3"]
    assert _stop_ids(mod, feed, [], ["5"]) == ["S3", "S4"]  # every route, Saturday
    assert _stop_ids(mod, feed, ["101"], []) == ["S1", "S2", "S3"]  # every day
    with pytest.raises(mod.RoutesNotInServiceError, match=r"\['202'\].*\['4'\]"):
        _stop_ids(mod, feed, ["202"], ["4"])


def test_check_service_selection_logs_calendar_and_default(
    mod: types.ModuleType, feed: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        mod._check_service_selection(str(feed), ["4"])
    assert "calendar.txt (2 service_id row(s))" in caplog.text
    assert "still the default ['4']" in caplog.text

    caplog.clear()
    (feed / "calendar.txt").unlink()  # calendar_dates.txt-only feeds have none
    with caplog.at_level(logging.INFO):
        mod._check_service_selection(str(feed), ["5"])
    assert "calendar.txt is absent or empty" in caplog.text
    assert "still the default" not in caplog.text


def test_check_service_selection_rejects_missing_id(mod: types.ModuleType, feed: Path) -> None:
    with pytest.raises(mod.ServiceSelectionError, match=r"4 \(1 trips\), 5 \(2 trips\)"):
        mod._check_service_selection(str(feed), ["9"])


def _configure_main(
    mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    feed: Path,
    routes: list[str],
    service_ids: list[str],
) -> None:
    monkeypatch.setattr(mod, "GTFS_FOLDER", str(feed))
    monkeypatch.setattr(mod, "GTFS_ROUTE_SHORT_NAMES", routes)
    monkeypatch.setattr(mod, "SERVICE_IDS_TO_INCLUDE", service_ids)


def test_main_exits_2_when_service_id_missing(
    mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    feed: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure_main(mod, monkeypatch, feed, ["101"], ["9"])
    with caplog.at_level(logging.INFO):
        assert mod.main() == 2
    assert "calendar.txt (2 service_id row(s))" in caplog.text
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is None  # reported without a traceback


def test_main_exits_2_when_no_selected_route_runs_that_day(
    mod: types.ModuleType, monkeypatch: pytest.MonkeyPatch, feed: Path
) -> None:
    _configure_main(mod, monkeypatch, feed, ["202"], ["4"])
    assert mod.main() == 2


def test_run_by_route_skips_route_without_service(
    mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    feed: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    built: list[list[str]] = []

    def fake_points_layer(stops_df: pd.DataFrame, layer_name: str = "") -> str:
        built.append(sorted(stops_df["stop_id"]))
        return layer_name

    monkeypatch.setattr(mod, "_points_layer_from_stops_df", fake_points_layer)
    monkeypatch.setattr(mod, "_process_service_area_from_stops_layer", lambda **kwargs: ({}, None))
    monkeypatch.setattr(mod, "resolved_clipped_fields", lambda fc: [])
    monkeypatch.setattr(mod, "BY_ROUTE_WRITE_CSV", False)

    with caplog.at_level(logging.WARNING):
        result = mod._run_by_route(str(feed), ["101", "202"], ["4"])
    assert list(result["route_short_name"]) == ["101"]
    assert built == [["S1", "S2"]]
    assert "Skipping route 202" in caplog.text
