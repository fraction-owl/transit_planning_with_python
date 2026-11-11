"""Tests for tests/fixtures/mock_tods_dc.zip and the generator logic behind it.

dev_tools/generate_mock_gtfs.py writes the mock TODS supplements (WRITE_TODS)
from the same blocks as tests/fixtures/mock_gtfs_dc.zip, so the two fixtures
must mesh: every TODS row is new, refers to GTFS services and blocks, and fills
each block's timeline from garage to garage without gaps, overlaps or jumps.
"""

from __future__ import annotations

import csv
import io
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import pytest

from dev_tools.generate_mock_gtfs import (
    BLOCK_LAYOVER_MINUTES,
    DEADHEAD_SPEED_MPH,
    Region,
    TripPlan,
    _assign_blocks,
    build_tods_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"
GTFS_ZIP = FIXTURES / "mock_gtfs_dc.zip"
TODS_ZIP = FIXTURES / "mock_tods_dc.zip"


def read_tables(path: Path, folder: str) -> dict[str, list[dict[str, str]]]:
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
    return read_tables(GTFS_ZIP, "dc")


@pytest.fixture(scope="module")
def tods() -> dict[str, list[dict[str, str]]]:
    return read_tables(TODS_ZIP, "dc_tods")


def seconds(value: str) -> int:
    h, m, s = map(int, value.split(":"))
    return h * 3600 + m * 60 + s


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    v = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 3958.8 * 2 * math.asin(math.sqrt(v))


def movements(tables: dict[str, list[dict[str, str]]], types: dict[str, str]) -> dict:
    """Return (service_id, block_id) -> [(start, end, from_stop, to_stop, type, trip_id)]."""
    calls: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tables.get("stop_times.txt", []) + tables.get("stop_times_supplement.txt", []):
        calls[row["trip_id"]].append(row)
    result: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for trip in tables.get("trips.txt", []) + tables.get("trips_supplement.txt", []):
        rows = sorted(calls[trip["trip_id"]], key=lambda r: int(r["stop_sequence"]))
        result[trip["service_id"], trip["block_id"]].append(
            (
                seconds(rows[0]["departure_time"]),
                seconds(rows[-1]["arrival_time"]),
                rows[0]["stop_id"],
                rows[-1]["stop_id"],
                types.get(trip["trip_id"], "revenue"),
                trip["trip_id"],
            )
        )
    return result


@pytest.fixture(scope="module")
def merged(gtfs: dict, tods: dict) -> dict:
    types = {t["trip_id"]: t["TODS_trip_type"] for t in tods["trips_supplement.txt"]}
    combined = {
        "trips.txt": gtfs["trips.txt"] + tods["trips_supplement.txt"],
        "stop_times.txt": gtfs["stop_times.txt"] + tods["stop_times_supplement.txt"],
    }
    return movements(combined, types)


def test_supplements_only_add_rows(gtfs: dict, tods: dict) -> None:
    # A supplement row whose primary key matches GTFS would update that row instead.
    for supplement, base, key in (
        ("routes_supplement.txt", "routes.txt", "route_id"),
        ("stops_supplement.txt", "stops.txt", "stop_id"),
        ("trips_supplement.txt", "trips.txt", "trip_id"),
    ):
        new = [r[key] for r in tods[supplement]]
        assert new and len(new) == len(set(new))
        assert not set(new) & {r[key] for r in gtfs[base]}


def test_added_rows_define_required_gtfs_fields(gtfs: dict, tods: dict) -> None:
    agencies = {r["agency_id"] for r in gtfs["agency.txt"]}
    routes = {r["route_id"] for r in tods["routes_supplement.txt"]}
    for route in tods["routes_supplement.txt"]:
        assert route["agency_id"] in agencies and route["route_type"] == "3"
        assert route["route_long_name"]
    for stop in tods["stops_supplement.txt"]:
        assert stop["stop_name"] and stop["TODS_location_type"] == "garage"
        assert 38.79 <= float(stop["stop_lat"]) <= 39.0
        assert -77.12 <= float(stop["stop_lon"]) <= -76.91
    services = {r["service_id"] for r in gtfs["calendar.txt"]}
    for trip in tods["trips_supplement.txt"]:
        assert trip["route_id"] in routes and trip["service_id"] in services
    stop_ids = {r["stop_id"] for r in gtfs["stops.txt"] + tods["stops_supplement.txt"]}
    for row in tods["stop_times_supplement.txt"]:
        assert row["stop_id"] in stop_ids
        assert (row["pickup_type"], row["drop_off_type"]) == ("1", "1")


def test_every_gtfs_block_has_one_pull_out_and_one_pull_back(gtfs: dict, tods: dict) -> None:
    gtfs_blocks = {(t["service_id"], t["block_id"]) for t in gtfs["trips.txt"]}
    counts: dict[tuple[str, str], list[str]] = defaultdict(list)
    for trip in tods["trips_supplement.txt"]:
        counts[trip["service_id"], trip["block_id"]].append(trip["TODS_trip_type"])
    assert set(counts) == gtfs_blocks
    for kinds in counts.values():
        assert kinds.count("pull-out") == 1 and kinds.count("pull-back") == 1
        assert set(kinds) <= {"pull-out", "pull-back", "deadhead"}


def test_block_timelines_run_garage_to_garage_without_jumps(
    merged: dict, gtfs: dict, tods: dict
) -> None:
    garages = {s["stop_id"] for s in tods["stops_supplement.txt"]}
    # Compare places, not stop_ids: route 50H's loop ends at a second stop_id
    # with the same coordinates as its first stop, so no deadhead is needed there.
    place = {
        s["stop_id"]: (s["stop_lat"], s["stop_lon"])
        for s in gtfs["stops.txt"] + tods["stops_supplement.txt"]
    }
    layover = BLOCK_LAYOVER_MINUTES * 60
    for block, legs in merged.items():
        legs.sort(key=lambda leg: (leg[0], leg[1]))
        assert legs[0][4] == "pull-out" and legs[-1][4] == "pull-back", block
        assert legs[0][2] in garages and legs[-1][3] == legs[0][2], block
        for before, after in zip(legs, legs[1:]):
            # No location jump between consecutive movements.
            assert place[after[2]] == place[before[3]], (block, before[5], after[5])
            assert after[0] >= before[1], (block, before[5], after[5])  # No overlap.
            if after[4] == "revenue" and before[4] != "pull-out":
                # Deadhead or revenue arrival still leaves the full layover.
                assert after[0] - before[1] >= layover, (block, before[5], after[5])


def test_deadheads_are_drivable(gtfs: dict, tods: dict) -> None:
    coords = {
        s["stop_id"]: (float(s["stop_lat"]), float(s["stop_lon"]))
        for s in gtfs["stops.txt"] + tods["stops_supplement.txt"]
    }
    calls: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tods["stop_times_supplement.txt"]:
        calls[row["trip_id"]].append(row)
    for trip_id, (first, last) in calls.items():
        hours = (seconds(last["arrival_time"]) - seconds(first["departure_time"])) / 3600
        miles = haversine_miles(coords[first["stop_id"]], coords[last["stop_id"]])
        assert hours > 0 and first["stop_id"] != last["stop_id"], trip_id
        assert miles / hours <= DEADHEAD_SPEED_MPH, trip_id  # Straight-line speed.


def test_fixture_covers_interlining_and_after_midnight(tods: dict) -> None:
    kinds = [t["TODS_trip_type"] for t in tods["trips_supplement.txt"]]
    assert kinds.count("deadhead") >= 10
    assert max(r["arrival_time"] for r in tods["stop_times_supplement.txt"]) >= "24:00:00"
    assert len(tods["stops_supplement.txt"]) == 2  # Both garages are used.


REGION = Region(
    key="t",
    agency_id="A",
    agency_name="Test",
    agency_url="https://example.org",
    agency_timezone="America/New_York",
    fixture_path=None,
    default_bbox=(-77.2, 38.8, -77.0, 39.0),
    garages=(("N", "North", 38.95, -77.1), ("S", "South", 38.85, -77.1)),
)
COORDS = {"A": (38.94, -77.1), "B": (38.86, -77.1)}  # About 5.5 miles apart.


def trip(trip_id: str, start: int, end: int, stops: tuple[str, str], block: str = "") -> TripPlan:
    return TripPlan(
        trip_id=trip_id,
        route_id="R",
        service_id="wk",
        shape_id="S",
        direction_id=0,
        block_id=block,
        headsign="",
        start_seconds=start,
        end_seconds=end,
        stop_visits=[(stops[0], start, 0.0), (stops[1], end, 0.0)],
    )


def test_assign_blocks_leaves_time_to_deadhead() -> None:
    # A->B ends 06:30. B->A at 06:40 can follow; A->B at 06:40 cannot, because the
    # bus would need ~22 min to get back to A plus the layover.
    first = trip("T1", 6 * 3600, 6 * 3600 + 1800, ("A", "B"))
    back = trip("T2", 6 * 3600 + 2400, 7 * 3600, ("B", "A"))
    same_way = trip("T3", 6 * 3600 + 2400, 7 * 3600, ("A", "B"))
    _assign_blocks([first, back, same_way], COORDS)
    assert first.block_id == back.block_id != same_way.block_id


def test_build_tods_rows_adds_pull_out_deadhead_and_pull_back() -> None:
    trips = [
        trip("T1", 6 * 3600, 6 * 3600 + 1800, ("A", "B"), "BLK1"),
        trip("T2", 8 * 3600, 8 * 3600 + 1800, ("A", "B"), "BLK1"),
    ]
    tables = build_tods_rows(REGION, trips, COORDS)
    kinds = [(t["trip_id"], t["TODS_trip_type"]) for t in tables["trips_supplement.txt"]]
    assert kinds == [("BLK1_PO", "pull-out"), ("BLK1_DH01", "deadhead"), ("BLK1_PB", "pull-back")]
    stops = [(r["trip_id"], r["stop_id"]) for r in tables["stop_times_supplement.txt"]]
    assert stops == [
        ("BLK1_PO", "T_GARAGE_N"),  # Nearest garage to the first stop, A.
        ("BLK1_PO", "A"),
        ("BLK1_DH01", "B"),
        ("BLK1_DH01", "A"),
        ("BLK1_PB", "B"),
        ("BLK1_PB", "T_GARAGE_N"),  # Back to the garage it left.
    ]
    times = [r["arrival_time"] for r in tables["stop_times_supplement.txt"]]
    assert times[1] == "06:00:00" and times[2] == "06:30:00"  # Pull-out ends as T1 starts.
    assert [s["stop_id"] for s in tables["stops_supplement.txt"]] == ["T_GARAGE_N"]
