"""Tests that the TIDES fixtures mesh with the mock GTFS feed and its TODS supplements.

tests/fixtures/stop_visits.csv and trips_performed.csv are written by
dev_tools/generate_mock_tides.py from tests/fixtures/mock_gtfs_dc.zip and
mock_tods_dc.zip, so every TIDES identifier must resolve against them.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from dev_tools import generate_mock_tides as gen

FIXTURES = Path(__file__).parent / "fixtures"
TRIPS_PERFORMED = FIXTURES / "trips_performed.csv"
STOP_VISITS = FIXTURES / "stop_visits.csv"


def read_zip(path: Path, folder: str) -> dict[str, list[dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        return {
            Path(name).name: list(
                csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8-sig"))
            )
            for name in archive.namelist()
            if name.startswith(f"{folder}/") and name.endswith(".txt")
        }


@pytest.fixture(scope="module")
def gtfs() -> dict[str, list[dict[str, str]]]:
    return read_zip(FIXTURES / "mock_gtfs_dc.zip", "dc")


@pytest.fixture(scope="module")
def tods() -> dict[str, list[dict[str, str]]]:
    return read_zip(FIXTURES / "mock_tods_dc.zip", "dc_tods")


@pytest.fixture(scope="module")
def trips() -> list[dict[str, str]]:
    return list(csv.DictReader(TRIPS_PERFORMED.open(encoding="utf-8")))


@pytest.fixture(scope="module")
def visits() -> list[dict[str, str]]:
    return list(csv.DictReader(STOP_VISITS.open(encoding="utf-8")))


def test_revenue_trips_resolve_against_gtfs(gtfs: dict, trips: list) -> None:
    scheduled = {t["trip_id"]: t for t in gtfs["trips.txt"]}
    revenue = [t for t in trips if t["trip_type"] == "In service"]
    assert revenue
    for row in revenue:
        gtfs_trip = scheduled[row["trip_id_scheduled"]]
        for field in ("route_id", "shape_id", "block_id", "direction_id"):
            assert row[field] == gtfs_trip[field], (row["trip_id_performed"], field)


def test_service_dates_are_days_the_gtfs_service_runs(gtfs: dict, trips: list) -> None:
    active = gen.expand_service_active_dates(
        pd.DataFrame(gtfs["calendar.txt"]), pd.DataFrame(gtfs["calendar_dates.txt"])
    )
    service = {t["trip_id"]: t["service_id"] for t in gtfs["trips.txt"]}
    service.update({t["trip_id"]: t["service_id"] for t in read_tods_trips()})
    for row in trips:
        day = dt.date.fromisoformat(row["service_date"])
        assert day in active[service[row["trip_id_scheduled"]]], row["trip_id_performed"]


def read_tods_trips() -> list[dict[str, str]]:
    return read_zip(FIXTURES / "mock_tods_dc.zip", "dc_tods")["trips_supplement.txt"]


def test_nonrevenue_trips_resolve_against_tods(tods: dict, trips: list) -> None:
    planned = {t["trip_id"]: t for t in tods["trips_supplement.txt"]}
    nonrevenue = [t for t in trips if t["trip_type"] != "In service"]
    assert {t["trip_type"] for t in nonrevenue} == {"Pullout", "Pullin", "Deadhead"}
    for row in nonrevenue:
        plan = planned[row["trip_id_scheduled"]]
        assert row["block_id"] == plan["block_id"]
        assert gen.TODS_TO_TIDES_TRIP_TYPE[plan["TODS_trip_type"]] == row["trip_type"]


def test_stop_visits_follow_the_gtfs_stop_times(gtfs: dict, trips: list, visits: list) -> None:
    calls = {
        (r["trip_id"], r["stop_sequence"]): r
        for r in gtfs["stop_times.txt"]
        if r["trip_id"] in {t["trip_id_scheduled"] for t in trips}
    }
    scheduled_trip = {t["trip_id_performed"]: t["trip_id_scheduled"] for t in trips}
    for row in visits:
        if row["schedule_relationship"] == "Added":
            continue
        call = calls[scheduled_trip[row["trip_id_performed"]], row["scheduled_stop_sequence"]]
        assert row["stop_id"] == call["stop_id"]
        assert row["schedule_arrival_time"][11:] == gen._at(
            dt.date(2026, 1, 1), gen._gtfs_seconds(call["arrival_time"])
        ).strftime("%H:%M:%S")


def test_every_observed_revenue_trip_has_stop_visits(trips: list, visits: list) -> None:
    visited = {v["trip_id_performed"] for v in visits}
    for row in trips:
        observed = row["trip_type"] == "In service" and row["schedule_relationship"] != "Canceled"
        assert (row["trip_id_performed"] in visited) == observed, row["trip_id_performed"]


def test_one_vehicle_runs_each_block_without_overlap(trips: list) -> None:
    by_block: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in trips:
        by_block[row["service_date"], row["block_id"]].append(row)
    for key, rows in by_block.items():
        assert len({r["vehicle_id"] for r in rows}) == 1, key
        run = sorted(
            (r for r in rows if r["actual_trip_start"]), key=lambda r: r["schedule_trip_start"]
        )
        assert run[0]["trip_type"] == "Pullout" and run[-1]["trip_type"] == "Pullin", key
        for before, after in zip(run, run[1:]):
            assert after["actual_trip_start"] >= before["actual_trip_end"], key


def test_fixture_keeps_its_edge_cases(trips: list, visits: list) -> None:
    revenue = [t for t in trips if t["trip_type"] == "In service"]
    # Owl trips end on the next calendar day while service_date stays put.
    assert any(t["schedule_trip_end"][:10] > t["service_date"] for t in revenue)
    assert any(t["schedule_relationship"] == "Canceled" for t in revenue)
    assert {"Skipped", "Added"} <= {v["schedule_relationship"] for v in visits}
    # Every month covers every sampled route and direction.
    cells = {
        (t["service_date"][:7], t["route_id"], t["direction_id"])
        for t in revenue
        if t["route_id"] in gen.ROUTES
    }
    assert len(cells) == 3 * len(gen.ROUTES) * 2
    # stop_visits keeps the M/D/YYYY service_date format of the real exports.
    assert visits[0]["service_date"].count("/") == 2


def test_holiday_pass_adds_the_holiday_service_date(trips: list) -> None:
    """2026-01-01 runs only the holiday service, every block of it, and nothing else."""
    holiday = [t for t in trips if t["service_date"] == "2026-01-01"]
    revenue = [t for t in holiday if t["trip_type"] == "In service"]
    assert {t["route_id"] for t in revenue} == {"DC_R50H"}
    assert all("_holiday_" in t["trip_id_scheduled"] for t in holiday)
    schedule = gen.load_schedule(gen.GTFS_PATH, gen.TODS_PATH)
    blocks = schedule.blocks["holiday", "DC_R50H"]
    assert {t["block_id"] for t in holiday} == set(blocks)
    # The holiday pass sits after the main sample, so its rows are appended.
    assert trips[-1]["service_date"] == "2026-01-01"
    assert all(t["route_id"] != "DC_R50H" for t in trips if t["service_date"] != "2026-01-01")


def test_fixtures_match_the_generator_output() -> None:
    import random

    schedule = gen.load_schedule(gen.GTFS_PATH, gen.TODS_PATH)
    trips_df, stops_df = gen.generate(schedule, random.Random(gen.RANDOM_SEED))
    for frame, path in ((trips_df, TRIPS_PERFORMED), (stops_df, STOP_VISITS)):
        expected = pd.read_csv(path, dtype=str, keep_default_na=False)
        actual = pd.read_csv(
            io.StringIO(frame.to_csv(index=False)), dtype=str, keep_default_na=False
        )
        pd.testing.assert_frame_equal(actual, expected, obj=path.name)
