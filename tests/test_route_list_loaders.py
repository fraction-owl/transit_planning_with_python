"""Tests for the route-list loaders and their sample_data examples."""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path

import pytest

from scripts.ridership_tools.trip_cut_savings import read_route_list
from utils.gtfs_helpers import (
    load_event_route_ids,
    load_express_route_ids,
    load_non_revenue_route_ids,
)

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"
NON_REVENUE_EXAMPLE = SAMPLE_DATA / "non_revenue_routes_example.txt"
EVENT_EXAMPLE = SAMPLE_DATA / "event_routes_example.txt"
EXPRESS_EXAMPLE = SAMPLE_DATA / "express_routes_example.txt"


def test_load_express_route_ids_reads_sample() -> None:
    assert load_express_route_ids(None, str(EXPRESS_EXAMPLE)) == {"DC_R20"}


def test_express_sample_routes_are_in_sample_gtfs() -> None:
    # gtfs_service_demographics_gpd.py matches express routes on route_id, so every
    # listed id must be a route_id in the sample feed for the example to label anything.
    with zipfile.ZipFile(SAMPLE_DATA / "gtfs_current.zip") as feed:
        with feed.open("dc/routes.txt") as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig"))
            route_ids = {row["route_id"] for row in reader}
    assert load_express_route_ids(None, str(EXPRESS_EXAMPLE)) <= route_ids


def test_load_non_revenue_route_ids_reads_sample() -> None:
    assert load_non_revenue_route_ids(None, str(NON_REVENUE_EXAMPLE)) == {"990", "991", "992"}


def test_load_event_route_ids_reads_sample() -> None:
    assert load_event_route_ids(None, str(EVENT_EXAMPLE)) == {"60E"}


def test_loaders_union_inline_and_file() -> None:
    assert load_event_route_ids([" 70E ", "60E"], str(EVENT_EXAMPLE)) == {"60E", "70E"}


@pytest.mark.parametrize("loader", [load_non_revenue_route_ids, load_event_route_ids])
def test_loaders_missing_file_warns_and_keeps_inline(
    loader, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = loader(["990"], str(tmp_path / "nope.txt"))
    assert result == {"990"}
    assert "not found" in caplog.text


@pytest.mark.parametrize(
    ("path", "expected"),
    [(NON_REVENUE_EXAMPLE, ("990", "991", "992")), (EVENT_EXAMPLE, ("60E",))],
)
def test_trip_cut_savings_reads_samples(path: Path, expected: tuple[str, ...]) -> None:
    # sample_data/README.md points trip_cut_savings.py at these files. Its reader keeps
    # text after a route on the same line, so an inline comment would break the match.
    routes, fingerprint = read_route_list(str(path), "SAMPLE", path.parent)
    assert routes == expected
    assert fingerprint is not None
