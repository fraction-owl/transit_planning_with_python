"""Tests for scheduled_service_hours (GTFS + TODS revenue and nonrevenue hours)."""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from openpyxl import load_workbook

import scripts.gtfs_exports.scheduled_service_hours as target

FIXTURES = Path(__file__).parent / "fixtures"
GTFS_ZIP = FIXTURES / "mock_gtfs_dc.zip"
TODS_ZIP = FIXTURES / "mock_tods_dc.zip"


def write_feed(folder: Path, tables: dict[str, list[dict[str, str]]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (folder / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return folder


def stop_time(trip: str, seq: int, stop: str, time: str) -> dict[str, str]:
    return {
        "trip_id": trip,
        "stop_sequence": str(seq),
        "stop_id": stop,
        "arrival_time": time,
        "departure_time": time,
    }


@pytest.fixture()
def small_feed(tmp_path: Path) -> tuple[Path, Path]:
    """One weekday block: two revenue trips A->B, so B->A must be deadheaded."""
    stops = [
        {"stop_id": s, "stop_name": s, "stop_lat": lat, "stop_lon": "-77.0"}
        for s, lat in (("A", "38.90"), ("B", "38.95"), ("A2", "38.90"))
    ]
    gtfs = write_feed(
        tmp_path / "gtfs",
        {
            "trips.txt": [
                {"route_id": "R", "service_id": "wk", "trip_id": t, "block_id": "B1"}
                for t in ("T1", "T2", "T3")
            ],
            "stop_times.txt": [
                stop_time("T1", 1, "A", "06:00:00"),
                stop_time("T1", 2, "B", "07:00:00"),
                stop_time("T2", 1, "A", "08:00:00"),
                stop_time("T2", 2, "B", "09:00:00"),
                # T3 starts at A2, a second stop_id at A's coordinates: no move.
                stop_time("T3", 1, "A2", "10:00:00"),
                stop_time("T3", 2, "B", "11:00:00"),
            ],
            "stops.txt": stops,
            "calendar.txt": [
                {
                    "service_id": "wk",
                    **{d: "1" for d in ("monday", "tuesday", "wednesday", "thursday", "friday")},
                    "saturday": "0",
                    "sunday": "0",
                    "start_date": "20261101",
                    "end_date": "20261130",
                }
            ],
        },
    )
    tods = write_feed(
        tmp_path / "tods",
        {
            "trips_supplement.txt": [
                {
                    "route_id": "NR",
                    "service_id": "wk",
                    "trip_id": t,
                    "block_id": "B1",
                    "TODS_trip_type": k,
                }
                for t, k in (("PO", "pull-out"), ("DH", "deadhead"), ("PB", "pull-back"))
            ]
            # T3 is deleted: its stop_times are then ignored, per the spec.
            + [{"trip_id": "T3", "TODS_delete": "1"}],
            "stop_times_supplement.txt": [
                stop_time("PO", 1, "G", "05:30:00"),
                stop_time("PO", 2, "A", "06:00:00"),
                stop_time("DH", 1, "B", "07:00:00"),
                stop_time("DH", 2, "A", "07:30:00"),
                stop_time("PB", 1, "B", "09:00:00"),
                stop_time("PB", 2, "G", "09:45:00"),
            ],
            "stops_supplement.txt": [
                {"stop_id": "G", "stop_name": "Garage", "stop_lat": "38.80", "stop_lon": "-77.0"}
            ],
        },
    )
    return gtfs, tods


# =============================================================================
# TODS SUPPLEMENTS
# =============================================================================


def test_apply_supplement_deletes_updates_and_adds() -> None:
    base = pd.DataFrame(
        {
            "stop_id": ["1", "2", "3"],
            "stop_name": ["One", "Two", "Three"],
            "stop_url": ["a", "b", "c"],
        }
    )
    supplement = pd.DataFrame(
        {
            "stop_id": ["2", "3", "4"],
            "stop_name": ["", "", "Four"],
            "stop_desc": ["", "Modified", "New"],
            "TODS_delete": ["1", "", ""],
        }
    )
    out = target.apply_supplement(base, supplement, ["stop_id"]).set_index("stop_id")
    # The spec's own example: 2 deleted, 3 updated (blank name ignored), 4 added.
    assert list(out.index) == ["1", "3", "4"]
    assert out.loc["3", "stop_name"] == "Three" and out.loc["3", "stop_desc"] == "Modified"
    assert out.loc["3", "stop_url"] == "c"
    assert out.loc["4", "stop_name"] == "Four" and out.loc["4", "stop_url"] == ""


def test_apply_supplement_rejects_delete_and_re_add() -> None:
    base = pd.DataFrame({"trip_id": ["T1"]})
    supplement = pd.DataFrame({"trip_id": ["T1", "T1"], "TODS_delete": ["1", ""]})
    with pytest.raises(ValueError, match="deletes and re-adds"):
        target.apply_supplement(base, supplement, ["trip_id"])


# =============================================================================
# HOURS
# =============================================================================


def test_small_feed_splits_the_block_day(small_feed: tuple[Path, Path]) -> None:
    gtfs, tods = small_feed
    schedule = target.load_schedule(gtfs, tods)
    assert "T3" not in set(schedule.movements["trip_id"])  # Deleted by TODS.
    movements = target.expand_by_service_set(schedule.movements, [("wk",)])
    (row,) = target.block_hours(movements, schedule.stop_places).to_dict("records")
    assert row["revenue_trip_hours"] == 2.0  # T1 + T2
    assert row["pull_out_hours"] == 0.5 and row["pull_in_hours"] == 0.75
    assert row["deadhead_hours"] == 0.5
    assert row["layover_hours"] == 0.5  # Deadhead arrives 07:30, T2 leaves 08:00.
    assert row["unresolved_gap_hours"] == 0.0 and row["location_jumps"] == 0
    assert (row["start"], row["end"]) == ("05:30", "09:45")


def test_moving_without_a_scheduled_movement_is_unresolved(small_feed: tuple[Path, Path]) -> None:
    gtfs, _ = small_feed
    # Without TODS, T1 ends at B and T2 starts at A an hour later: an unresolved gap.
    schedule = target.load_schedule(gtfs, None)
    movements = target.expand_by_service_set(schedule.movements, [("wk",)])
    (row,) = target.block_hours(movements, schedule.stop_places).to_dict("records")
    assert row["unresolved_gap_hours"] == 2.0  # T1->T2 and T2->T3 both change place.
    assert row["layover_hours"] == 0.0
    assert row["location_jumps"] == 2
    assert row["revenue_trip_hours"] == 3.0


def test_same_coordinates_count_as_one_place() -> None:
    places = {"A": ("38.9", "-77.0"), "A2": ("38.9", "-77.0"), "X": ("", "")}
    assert target.same_place("A", "A2", places)
    assert not target.same_place("X", "Y", places)


def test_same_place_compares_coordinates_as_numbers() -> None:
    places = {
        "B": ("38.8", "-77.2"),
        "B_padded": ("38.800000", "-77.200000"),
        "B_bay": ("38.8001", "-77.2"),  # A nearby bay, not the same place.
        "X": ("", ""),
        "X2": ("", ""),
        "bad": ("n/a", "-77.2"),
        "bad2": ("n/a", "-77.2"),
    }
    assert target.same_place("B", "B_padded", places)
    assert not target.same_place("B", "B_bay", places)
    # Missing or unreadable coordinates never make two stops one place.
    assert not target.same_place("X", "X2", places)
    assert not target.same_place("bad", "bad2", places)


def test_layover_at_reformatted_coordinates_counts_as_revenue(tmp_path: Path) -> None:
    # T1 ends at B at 06:30 and T2 leaves B2 at 07:00. B and B2 are the same
    # point, written with different precision.
    gtfs = write_feed(
        tmp_path / "gtfs",
        {
            "trips.txt": [
                {"route_id": "R", "service_id": "wk", "trip_id": t, "block_id": "B1"}
                for t in ("T1", "T2")
            ],
            "stop_times.txt": [
                stop_time("T1", 1, "A", "06:00:00"),
                stop_time("T1", 2, "B", "06:30:00"),
                stop_time("T2", 1, "B2", "07:00:00"),
                stop_time("T2", 2, "A", "07:30:00"),
            ],
            "stops.txt": [
                {"stop_id": "A", "stop_lat": "38.9", "stop_lon": "-77.2"},
                {"stop_id": "B", "stop_lat": "38.8", "stop_lon": "-77.2"},
                {"stop_id": "B2", "stop_lat": "38.800000", "stop_lon": "-77.200000"},
            ],
            "calendar.txt": [calendar_row("wk", DAYS[:5])],
        },
    )
    cfg = target.Config(
        gtfs_path=gtfs,
        tods_path=None,
        output_dir=tmp_path / "out",
        periods=[("Nov", "20261101", "20261130")],
        layover_in_revenue_hours=True,
    )
    (block,) = target.run(cfg)["block"].to_dict("records")
    assert block["layover_hours"] == 0.5
    assert block["unresolved_gap_hours"] == 0.0 and block["location_jumps"] == 0
    assert block["vehicle_revenue_hours"] == 1.5


def test_add_derived_follows_the_layover_setting() -> None:
    df = pd.DataFrame({c: [1.0] for c in target.COMPONENTS})
    counted = target.add_derived(df, layover_in_revenue=True, movements_known=True).iloc[0]
    assert counted["vehicle_revenue_hours"] == 2.0 and counted["nonrevenue_hours"] == 4.0
    separate = target.add_derived(df, layover_in_revenue=False, movements_known=True).iloc[0]
    assert separate["vehicle_revenue_hours"] == 1.0 and separate["nonrevenue_hours"] == 5.0
    assert counted["vehicle_hours"] == separate["vehicle_hours"] == 7.0
    blank = target.add_derived(df, layover_in_revenue=True, movements_known=False).iloc[0]
    assert math.isnan(blank["deadhead_hours"]) and math.isnan(blank["vehicle_hours"])


def test_service_day_types_explicit_mapping() -> None:
    types = target.service_day_types({}, {"wk": "Weekday", "end": "saturday, Sunday"})
    assert types == {"wk": {"Weekday"}, "end": {"Saturday", "Sunday"}}
    with pytest.raises(ValueError, match="unknown day type"):
        target.service_day_types({}, {"x": "Fiesta"})


# =============================================================================
# SERVICE SETS (services that run together on a date)
# =============================================================================

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def calendar_row(service_id: str, days: tuple[str, ...]) -> dict[str, str]:
    return {
        "service_id": service_id,
        **{d: "1" if d in days else "0" for d in DAYS},
        "start_date": "20261101",
        "end_date": "20261130",
    }


def run_feed(
    folder: Path,
    trips: list[tuple[str, str, str]],
    times: dict[str, tuple[str, str, str, str]],
    calendar: list[dict[str, str]],
) -> dict[str, pd.DataFrame]:
    """Run November 2026 on trips (trip_id, service_id, block_id) with (from, to, start, end)."""
    gtfs = write_feed(
        folder / "gtfs",
        {
            "trips.txt": [
                {"route_id": "R", "service_id": s, "trip_id": t, "block_id": b} for t, s, b in trips
            ],
            "stop_times.txt": [
                row
                for t, (a, b, start, end) in times.items()
                for row in (stop_time(t, 1, a, start), stop_time(t, 2, b, end))
            ],
            "stops.txt": [
                {"stop_id": s, "stop_lat": lat, "stop_lon": "-77.0"}
                for s, lat in (("A", "38.90"), ("B", "38.95"))
            ],
            "calendar.txt": calendar,
        },
    )
    cfg = target.Config(
        gtfs_path=gtfs,
        tods_path=None,
        output_dir=folder / "out",
        periods=[("Nov", "20261101", "20261130")],
    )
    return target.run(cfg)


TWO_HOUR_BLOCK = {
    "T1": ("A", "B", "06:00:00", "07:00:00"),
    "T2": ("B", "A", "08:00:00", "09:00:00"),  # One hour of layover at B before it.
}


OVERLAPPING_BLOCK = {
    "T1": ("A", "A", "08:00:00", "10:00:00"),
    "T2": ("A", "A", "08:15:00", "08:30:00"),  # Starts while T1 is still running.
    "T3": ("A", "A", "09:00:00", "09:15:00"),  # So does this one.
}


def overlapping_block(tmp_path: Path) -> dict[str, pd.DataFrame]:
    return run_feed(
        tmp_path,
        [(t, "wk", "B1") for t in OVERLAPPING_BLOCK],
        OVERLAPPING_BLOCK,
        [calendar_row("wk", DAYS[:5])],
    )


def test_time_an_earlier_trip_covers_is_not_layover(tmp_path: Path) -> None:
    (block,) = overlapping_block(tmp_path)["block"].to_dict("records")
    # T2 ends at 08:30 and T3 starts at 09:00, but T1 runs until 10:00.
    assert block["layover_hours"] == 0.0
    assert block["overlaps"] == 2
    assert block["revenue_trip_hours"] == 2.5


def test_overlapping_block_withholds_vehicle_hours(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        tables = overlapping_block(tmp_path)
    for key in ("block", "service", "day_type", "period"):
        row = tables[key].iloc[0]
        assert row["overlaps"] > 0, key
        assert all(math.isnan(row[c]) for c in target.DERIVED), key
    assert tables["period"].iloc[0]["revenue_trip_hours"] == 21 * 2.5
    assert "B1 (wk)" in caplog.text


def test_gap_after_an_overlap_starts_where_the_latest_trip_ends(tmp_path: Path) -> None:
    # T2 runs inside T1 and ends at A; the vehicle is free at 09:00 at B, where T1
    # ends, so T3 from B at 09:30 follows half an hour of layover, not an unresolved
    # gap from A starting at 07:00.
    tables = run_feed(
        tmp_path,
        [(t, "wk", "B1") for t in ("T1", "T2", "T3")],
        {
            "T1": ("A", "B", "06:00:00", "09:00:00"),
            "T2": ("A", "A", "06:30:00", "07:00:00"),
            "T3": ("B", "A", "09:30:00", "10:00:00"),
        },
        [calendar_row("wk", DAYS[:5])],
    )
    (block,) = tables["block"].to_dict("records")
    assert block["layover_hours"] == 0.5
    assert block["unresolved_gap_hours"] == 0.0
    assert block["overlaps"] == 1


@pytest.mark.parametrize("services", [("wk", "wk"), ("wkA", "wkB")])
def test_block_across_services_with_the_same_days_keeps_its_layover(
    tmp_path: Path, services: tuple[str, str]
) -> None:
    # GTFS defines a block by block_id and shared service days, not by service_id.
    weekdays = DAYS[:5]
    tables = run_feed(
        tmp_path,
        [("T1", services[0], "B1"), ("T2", services[1], "B1")],
        TWO_HOUR_BLOCK,
        [calendar_row(s, weekdays) for s in sorted(set(services))],
    )
    (block,) = tables["block"].to_dict("records")
    assert block["service_ids"] == "|".join(sorted(set(services)))
    assert block["layover_hours"] == 1.0 and block["location_jumps"] == 0
    nov = tables["period"].iloc[0]
    # 21 weekdays x (2 revenue trip hours + 1 layover hour).
    assert nov["layover_hours"] == 21.0
    assert nov["vehicle_revenue_hours"] == 63.0


def everyday_plus_weekday_trip(tmp_path: Path) -> dict[str, pd.DataFrame]:
    """Block B1: T1 and T3 run every day; T2, between them, runs on weekdays only."""
    return run_feed(
        tmp_path,
        [("T1", "all", "B1"), ("T2", "wk", "B1"), ("T3", "all", "B1")],
        {**TWO_HOUR_BLOCK, "T3": ("A", "B", "10:00:00", "11:00:00")},
        [calendar_row("all", DAYS), calendar_row("wk", DAYS[:5])],
    )


def test_block_is_rebuilt_for_each_set_of_services_that_runs(tmp_path: Path) -> None:
    tables = everyday_plus_weekday_trip(tmp_path)
    blocks = tables["block"].set_index("service_ids")
    # Weekdays: T1, T2, T3 with an hour of layover before T2 and before T3.
    assert blocks.loc["all|wk", "revenue_trip_hours"] == 3.0
    assert blocks.loc["all|wk", "layover_hours"] == 2.0
    # Weekends: T1 ends at B and T3 starts at A three hours later.
    assert blocks.loc["all", "revenue_trip_hours"] == 2.0
    assert blocks.loc["all", "unresolved_gap_hours"] == 3.0
    nov = tables["period"].iloc[0]
    # Veterans Day (11/11) and Thanksgiving (11/26) are holidays by date.
    assert (nov["weekdays"], nov["saturdays"] + nov["sundays"]) == (19, 9)
    assert nov["revenue_trip_hours"] == 21 * 3.0 + 9 * 2.0
    assert nov["layover_hours"] == 21 * 2.0
    assert nov["unresolved_gap_hours"] == 9 * 3.0
    day_types = tables["day_type"].set_index("day_type")
    assert day_types.loc["Weekday", "service_ids"] == "all|wk"
    assert day_types.loc["Saturday", "service_ids"] == "all"


def test_typical_weekday_is_the_most_common_operating_day(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # One service_id per weekday: Monday-Thursday one hour, Friday two hours.
    times = {f"T{d}": ("A", "B", "06:00:00", "07:00:00") for d in DAYS[:4]}
    times["Tfriday"] = ("A", "B", "06:00:00", "08:00:00")
    with caplog.at_level("WARNING"):
        tables = run_feed(
            tmp_path / "split",
            [(f"T{d}", d, f"B{d}") for d in DAYS[:5]],
            times,
            [calendar_row(d, (d,)) for d in DAYS[:5]],
        )
    weekday = tables["day_type"].set_index("day_type").loc["Weekday"]
    # November 2026 has five Mondays and four of each other weekday; Veterans Day
    # (Wednesday) and Thanksgiving (Thursday) are holidays, not weekdays.
    assert weekday["service_ids"] == "monday"
    assert (weekday["days"], weekday["typical_days"]) == (19, 5)
    assert weekday["revenue_trip_hours"] == 1.0  # Not 6.0, the five services summed.
    assert "5 service sets run on Weekday dates" in caplog.text
    # Period totals count each weekday's own service.
    assert tables["period"].iloc[0]["revenue_trip_hours"] == 17 * 1.0 + 4 * 2.0

    single = run_feed(
        tmp_path / "single",
        [("T", "wk", "B")],
        {"T": ("A", "B", "06:00:00", "07:00:00")},
        [calendar_row("wk", DAYS[:5])],
    )
    weekday = single["day_type"].set_index("day_type").loc["Weekday"]
    assert (weekday["days"], weekday["typical_days"]) == (19, 19)
    assert weekday["revenue_trip_hours"] == 1.0


def christmas_on_sunday_service(folder: Path) -> dict[str, pd.DataFrame]:
    """Weekday, Saturday and Sunday service; Friday 2026-12-25 runs Sunday service."""
    weekday, saturday, sunday = DAYS[:5], ("saturday",), ("sunday",)
    calendar = [
        {**calendar_row(sid, days), "end_date": "20261231"}
        for sid, days in (("wk", weekday), ("sat", saturday), ("sun", sunday))
    ]
    gtfs = write_feed(
        folder / "gtfs",
        {
            "trips.txt": [
                {"route_id": "R", "service_id": sid, "trip_id": f"T{sid}", "block_id": f"B{sid}"}
                for sid in ("wk", "sat", "sun")
            ],
            "stop_times.txt": [
                row
                for sid, end in (("wk", "08:00:00"), ("sat", "07:30:00"), ("sun", "07:00:00"))
                for row in (
                    stop_time(f"T{sid}", 1, "A", "06:00:00"),
                    stop_time(f"T{sid}", 2, "B", end),
                )
            ],
            "stops.txt": [
                {"stop_id": "A", "stop_lat": "38.90", "stop_lon": "-77.0"},
                {"stop_id": "B", "stop_lat": "38.95", "stop_lon": "-77.0"},
            ],
            "calendar.txt": calendar,
            "calendar_dates.txt": [
                {"service_id": "wk", "date": "20261225", "exception_type": "2"},
                {"service_id": "sun", "date": "20261225", "exception_type": "1"},
            ],
        },
    )
    cfg = target.Config(
        gtfs_path=gtfs,
        tods_path=None,
        output_dir=folder / "out",
        periods=[("Dec", "20261201", "20261231")],
    )
    return target.run(cfg)


def test_period_counts_days_by_the_schedule_they_run(tmp_path: Path) -> None:
    dec = christmas_on_sunday_service(tmp_path)["period"].iloc[0]
    # Calendar days: Christmas is a holiday (on a Friday).
    assert (dec["weekdays"], dec["saturdays"], dec["sundays"], dec["holidays"]) == (22, 4, 4, 1)
    # Operating schedules: Christmas runs the Sunday schedule.
    schedules = ("weekday", "saturday", "sunday", "holiday")
    assert tuple(dec[f"{d}_schedule_days"] for d in schedules) == (22, 4, 5, 0)
    assert dec["revenue_trip_hours"] == 22 * 2.0 + 4 * 1.5 + 5 * 1.0


def test_schedule_day_types_follow_the_typical_sets() -> None:
    reference = pd.DataFrame(
        {
            "day_type": ["Weekday"] * 5 + ["Saturday", "Sunday"] * 3 + ["Weekday"],
            "service_set": ["wk"] * 4 + ["wk|school"] + ["end"] * 6 + [""],
        }
    )
    calendar = pd.DataFrame(
        {
            "day_type": ["Weekday", "Weekday", "Saturday", "Sunday", "Weekday", "Weekday"],
            "service_set": ["wk", "wk|school", "end", "end", "end", ""],
        }
    )
    # "end" is typical of both weekend days: a weekend date keeps its own type,
    # a weekday running it takes the first (Saturday). A set typical of nothing
    # (wk|school) keeps the date's day type; no service is blank.
    assert target.schedule_day_types(calendar, reference) == [
        "Weekday",
        "Weekday",
        "Saturday",
        "Sunday",
        "Saturday",
        "",
    ]


def split_weekday_feed(folder: Path) -> Path:
    """WMATA-shaped feed: four weekday service_ids that never run on the same date.

    Each runs its own date range and reuses blocks B1 and B2 (two one-hour trips
    each), as successive schedules do. Service 3 replaces the weekday service on
    Friday 2026-02-13 only, by calendar_dates exceptions: the feed says which trips
    run that Friday, not that it is a holiday.
    """
    weekday = DAYS[:5]
    ranges = (("1", "20251215", "20260109"), ("10", "20260112", "20260206"))
    ranges += (("9", "20260209", "20260403"), ("6", "20260406", "20260612"))
    calendar = [
        {**calendar_row(sid, weekday), "start_date": start, "end_date": end}
        for sid, start, end in ranges
    ]
    calendar += [
        {**calendar_row(sid, (day,)), "start_date": "20251214", "end_date": "20260613"}
        for sid, day in (("S", "saturday"), ("U", "sunday"))
    ]
    trips, stop_times = [], []
    for sid in ("1", "10", "9", "6", "3", "S", "U"):
        for block in ("B1", "B2"):
            for k, (start, end) in enumerate((("06:00:00", "07:00:00"), ("08:00:00", "09:00:00"))):
                trip = f"{sid}_{block}_{k}"
                trips.append(
                    {"route_id": "R", "service_id": sid, "trip_id": trip, "block_id": block}
                )
                stop_times += [stop_time(trip, 1, "A", start), stop_time(trip, 2, "B", end)]
    return write_feed(
        folder,
        {
            "trips.txt": trips,
            "stop_times.txt": stop_times,
            "stops.txt": [
                {"stop_id": "A", "stop_lat": "38.90", "stop_lon": "-77.0"},
                {"stop_id": "B", "stop_lat": "38.95", "stop_lon": "-77.0"},
            ],
            "calendar.txt": calendar,
            "calendar_dates.txt": [
                {"service_id": "9", "date": "20260213", "exception_type": "2"},
                {"service_id": "3", "date": "20260213", "exception_type": "1"},
            ],
        },
    )


def split_weekday_run(tmp_path: Path, **settings: Any) -> dict[str, pd.DataFrame]:
    cfg = target.Config(
        gtfs_path=split_weekday_feed(tmp_path / "gtfs"),
        tods_path=None,
        output_dir=tmp_path / "out",
        periods=[("Mon", "20260209", "20260209"), ("Week", "20260209", "20260215")],
        **settings,
    )
    return target.run(cfg)


def test_config_calendar_runs_one_observed_weekday_set(tmp_path: Path) -> None:
    # Adding up services 1, 10, 9 and 6 counted four weekdays' trips on every date,
    # with every block overlapping itself.
    tables = split_weekday_run(tmp_path, period_calendar="config")
    periods = tables["period"].set_index("period")
    assert periods.loc["Mon", "revenue_trip_hours"] == 4.0
    assert periods.loc["Week", "revenue_trip_hours"] == 7 * 4.0
    assert tables["block"]["overlaps"].sum() == 0
    assert "|" not in tables["day_type"].set_index("day_type").loc["Weekday", "service_ids"]


def test_config_calendar_rejects_services_that_never_run_together(tmp_path: Path) -> None:
    types = {"1": "Weekday", "9": "Weekday", "S": "Saturday", "U": "Sunday"}
    with pytest.raises(ValueError, match=r"lists 1, 9 as Weekday, but they never all run"):
        split_weekday_run(tmp_path, period_calendar="config", service_day_types=types)
    types = {"9": "Weekday", "S": "Saturday", "U": "Sunday"}
    tables = split_weekday_run(tmp_path, period_calendar="config", service_day_types=types)
    assert tables["period"].set_index("period").loc["Mon", "revenue_trip_hours"] == 4.0


def test_exception_only_friday_is_a_weekday_not_a_holiday(tmp_path: Path) -> None:
    week = split_weekday_run(tmp_path)["period"].set_index("period").loc["Week"]
    assert (week["weekdays"], week["holidays"], week["saturdays"], week["sundays"]) == (5, 0, 1, 1)
    assert week["holiday_schedule_days"] == 0
    # The Friday's own schedule stays visible in the period's service sets.
    assert week["service_sets"] == "9 (4 days); 3 (1 day); S (1 day); U (1 day)"
    assert week["revenue_trip_hours"] == 7 * 4.0


def test_listed_holiday_dates_are_holidays_in_the_gtfs_calendar(tmp_path: Path) -> None:
    # Wednesday 2026-02-11 runs the regular weekday service, but is listed as a holiday.
    tables = split_weekday_run(tmp_path, holiday_dates=["20260211"])
    week = tables["period"].set_index("period").loc["Week"]
    assert (week["weekdays"], week["holidays"]) == (4, 1)
    assert (week["weekday_schedule_days"], week["holiday_schedule_days"]) == (5, 0)


def test_typical_holiday_set_runs_mostly_on_holidays() -> None:
    calendar = pd.DataFrame(
        {
            "day_type": ["Weekday"] * 6 + ["Holiday"] * 3 + ["Sunday"] * 2,
            "service_set": ["wk"] * 6 + ["wk", "hol", "sun"] + ["sun"] * 2,
        }
    )
    # A holiday running the weekday or Sunday schedule does not make it a holiday
    # schedule; "hol" runs only on holidays.
    typical = target.typical_service_sets(calendar)
    assert typical["Holiday"] == [("hol", 1)]
    assert typical["Weekday"][0] == ("wk", 6)
    no_holiday_schedule = calendar.assign(service_set=["wk"] * 7 + ["sun"] * 4)
    assert "Holiday" not in target.typical_service_sets(no_holiday_schedule)


# =============================================================================
# FREQUENCY-BASED TRIPS
# =============================================================================

FREQUENCIES_HEADER = "trip_id,start_time,end_time,headway_secs,exact_times\n"


def frequency_feed(folder: Path, frequencies: str) -> Path:
    """One 30-minute trip, repeated as *frequencies* (the frequencies.txt text) says."""
    gtfs = write_feed(
        folder,
        {
            "trips.txt": [{"route_id": "R", "service_id": "wk", "trip_id": "T1"}],
            "stop_times.txt": [
                stop_time("T1", 1, "A", "06:00:00"),
                stop_time("T1", 2, "B", "06:30:00"),
            ],
        },
    )
    (gtfs / "frequencies.txt").write_text(frequencies, encoding="utf-8")
    return gtfs


def test_frequency_based_trips_stop_the_run(tmp_path: Path) -> None:
    # Every 30 minutes from 06:00 to 18:00 is 24 trips and 12 revenue trip hours;
    # counting T1 once would report 1 trip and 0.5 hours.
    rows = FREQUENCIES_HEADER + "T1,06:00:00,18:00:00,1800,1\n"
    gtfs = frequency_feed(tmp_path / "gtfs", rows)
    with pytest.raises(ValueError, match=r"frequency-based trips \(T1\)"):
        target.load_schedule(gtfs, None)
    out = tmp_path / "out"
    argv = ["--gtfs", str(gtfs), "--no-tods", "--output-dir", str(out)]
    assert target.main([*argv, "--period", "Nov", "20261101", "20261130"]) == 1
    assert not out.exists()  # No CSVs, workbook or run log.


def test_header_only_frequencies_file_is_ignored(tmp_path: Path) -> None:
    gtfs = frequency_feed(tmp_path / "gtfs", FREQUENCIES_HEADER)
    assert list(target.load_schedule(gtfs, None).movements["trip_id"]) == ["T1"]


# =============================================================================
# MOCK FIXTURES (GTFS + TODS)
# =============================================================================


def read_zip(path: Path, folder: str) -> dict[str, list[dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        return {
            Path(n).name: list(csv.DictReader(io.TextIOWrapper(archive.open(n), encoding="utf-8")))
            for n in archive.namelist()
            if n.startswith(f"{folder}/") and n.endswith(".txt")
        }


def trip_hours(stop_times: list[dict[str, str]]) -> dict[str, float]:
    calls: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stop_times:
        calls[row["trip_id"]].append(row)
    out = {}
    for trip, rows in calls.items():
        rows.sort(key=lambda r: int(r["stop_sequence"]))
        start, end = (
            target._seconds(rows[0]["departure_time"]),
            target._seconds(rows[-1]["arrival_time"]),
        )
        out[trip] = (end - start) / 3600
    return out


@pytest.fixture(scope="module")
def mock_tables(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=TODS_ZIP,
        output_dir=tmp_path_factory.mktemp("hours"),
        periods=[("Nov", "20261101", "20261130"), ("Jan", "20260101", "20260131")],
        tods_movements_complete=True,
    )
    return target.run(cfg)


def test_mock_hours_match_the_feeds(mock_tables: dict[str, pd.DataFrame]) -> None:
    gtfs, tods = read_zip(GTFS_ZIP, "dc"), read_zip(TODS_ZIP, "dc_tods")
    revenue, nonrevenue = (
        trip_hours(gtfs["stop_times.txt"]),
        trip_hours(tods["stop_times_supplement.txt"]),
    )
    services = mock_tables["service"].set_index("service_ids")
    for sid in services.index:
        rev = sum(revenue[t["trip_id"]] for t in gtfs["trips.txt"] if t["service_id"] == sid)
        non = sum(
            nonrevenue[t["trip_id"]] for t in tods["trips_supplement.txt"] if t["service_id"] == sid
        )
        row = services.loc[sid]
        assert row["revenue_trip_hours"] == pytest.approx(rev, abs=0.01)
        tods_hours = row[["pull_out_hours", "pull_in_hours", "deadhead_hours"]].sum()
        assert tods_hours == pytest.approx(non, abs=0.02)
        # Mock blocks are complete: no gaps without a movement, no overlaps.
        assert row["unresolved_gap_hours"] == 0 and row["overlaps"] == 0
        assert row["blocks_without_tods"] == 0


def test_day_types_are_classified_from_the_calendar(mock_tables: dict[str, pd.DataFrame]) -> None:
    day_types = mock_tables["day_type"].set_index("day_type")
    assert list(day_types.index) == ["Weekday", "Saturday", "Sunday", "Holiday"]
    assert day_types.loc["Weekday", "service_ids"] == "weekday"
    assert day_types.loc["Holiday", "service_ids"] == "holiday"
    # Only the weekday express (route 20) interlines with deadheads.
    assert day_types.loc["Weekday", "deadhead_hours"] > 0
    assert day_types.loc["Saturday", "deadhead_hours"] == 0


def test_gtfs_calendar_periods_follow_the_feed(mock_tables: dict[str, pd.DataFrame]) -> None:
    services = mock_tables["service"].set_index("service_ids")
    periods = mock_tables["period"].set_index("period")
    nov = periods.loc["Nov"]
    # calendar_dates swaps Veterans Day (Wed 11/11) and Thanksgiving (Thu 11/26)
    # from weekday to holiday service.
    assert (nov["weekdays"], nov["saturdays"], nov["sundays"], nov["holidays"]) == (19, 4, 5, 2)
    schedules = ("weekday", "saturday", "sunday", "holiday")
    assert tuple(nov[f"{d}_schedule_days"] for d in schedules) == (19, 4, 5, 2)
    expected = (
        19 * services.loc["weekday", "vehicle_hours"]
        + 4 * services.loc["saturday", "vehicle_hours"]
        + 5 * services.loc["sunday", "vehicle_hours"]
        + 2 * services.loc["holiday", "vehicle_hours"]
    )
    assert nov["vehicle_hours"] == pytest.approx(expected, abs=0.1)
    # 2026-01-01 runs holiday service only (calendar_dates removes weekday service).
    # MLK Day (1/19) is a holiday by date but runs the weekday schedule.
    jan = periods.loc["Jan"]
    assert (jan["weekdays"], jan["holidays"]) == (20, 2)
    assert (jan["weekday_schedule_days"], jan["holiday_schedule_days"]) == (21, 1)


def test_config_calendar_uses_federal_holidays(tmp_path: Path) -> None:
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=TODS_ZIP,
        output_dir=tmp_path,
        periods=[("Nov", "20261101", "20261130")],
        period_calendar="config",
        holiday_service_day_type="Sunday",
    )
    tables = target.run(cfg)
    services = tables["service"].set_index("service_ids")
    nov = tables["period"].iloc[0]
    # Veterans Day (Wed 11/11) and Thanksgiving (Thu 11/26) run the Sunday schedule.
    assert (nov["weekdays"], nov["holidays"]) == (19, 2)
    assert (nov["weekday_schedule_days"], nov["sunday_schedule_days"]) == (19, 7)
    assert nov["holiday_schedule_days"] == 0
    expected = (
        19 * services.loc["weekday", "revenue_trip_hours"]
        + 7 * services.loc["sunday", "revenue_trip_hours"]
        + 4 * services.loc["saturday", "revenue_trip_hours"]
    )
    assert nov["revenue_trip_hours"] == pytest.approx(expected, abs=0.1)


# =============================================================================
# CLI
# =============================================================================


def test_main_returns_2_for_placeholders() -> None:
    assert target.main([]) == 2


def test_main_writes_outputs_and_run_log(tmp_path: Path) -> None:
    argv = ["--gtfs", str(GTFS_ZIP), "--tods", str(TODS_ZIP), "--output-dir", str(tmp_path)]
    assert target.main([*argv, "--period", "Nov", "20261101", "20261130"]) == 0
    for name in (
        target.BLOCK_FILENAME,
        target.SERVICE_FILENAME,
        target.DAY_TYPE_FILENAME,
        target.PERIOD_FILENAME,
    ):
        assert (tmp_path / name).is_file()
    log = (tmp_path / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "LAYOVER_IN_REVENUE_HOURS: bool =" in log  # Verbatim CONFIGURATION block.


def test_main_without_tods_leaves_nonrevenue_blank(tmp_path: Path) -> None:
    argv = ["--gtfs", str(GTFS_ZIP), "--no-tods", "--output-dir", str(tmp_path)]
    assert target.main([*argv, "--period", "Nov", "20261101", "20261130"]) == 0
    period = pd.read_csv(tmp_path / target.PERIOD_FILENAME).iloc[0]
    assert period["revenue_trip_hours"] > 0
    assert pd.isna(period["deadhead_hours"]) and pd.isna(period["vehicle_hours"])


def stops_only_tods(folder: Path) -> Path:
    """A TODS folder that only edits one stop's description: no movements."""
    with zipfile.ZipFile(GTFS_ZIP) as archive:
        name = next(n for n in archive.namelist() if n.endswith("stops.txt"))
        stop_id = next(csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8")))
    return write_feed(
        folder, {"stops_supplement.txt": [{"stop_id": stop_id["stop_id"], "stop_desc": "Edited"}]}
    )


@pytest.mark.parametrize("confirmed", [False, True])
def test_tods_without_movements_leaves_vehicle_hours_blank(
    tmp_path: Path, confirmed: bool, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=stops_only_tods(tmp_path / "tods"),
        output_dir=tmp_path / "out",
        periods=[("Nov", "20261101", "20261130")],
        tods_movements_complete=confirmed,
    )
    with caplog.at_level("WARNING"):
        nov = target.run(cfg)["period"].iloc[0]
    assert nov["revenue_trip_hours"] > 0 and nov["unresolved_gap_hours"] > 0
    # Not 0.00 nonrevenue hours and a complete-looking vehicle-hours total.
    assert pd.isna(nov["nonrevenue_hours"]) and pd.isna(nov["vehicle_hours"])
    assert pd.isna(nov["pull_out_hours"])
    assert ("no nonrevenue trips" if confirmed else "0 of") in caplog.text
    total = period_totals(load_workbook(tmp_path / "out" / target.WORKBOOK_FILENAME))["Nov"]
    assert total["Vehicle h"] == total["Nonrevenue h"] == "n/a"


def test_tods_movements_need_confirmation(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    argv = [
        "--gtfs",
        str(GTFS_ZIP),
        "--tods",
        str(TODS_ZIP),
        "--period",
        "Nov",
        "20261101",
        "20261130",
    ]
    with caplog.at_level("WARNING"):
        assert target.main([*argv, "--output-dir", str(tmp_path / "unconfirmed")]) == 0
    assert "TODS_MOVEMENTS_COMPLETE is False" in caplog.text
    unconfirmed = pd.read_csv(tmp_path / "unconfirmed" / target.PERIOD_FILENAME).iloc[0]
    assert pd.isna(unconfirmed["deadhead_hours"]) and pd.isna(unconfirmed["vehicle_hours"])

    confirmed_dir = tmp_path / "confirmed"
    assert (
        target.main([*argv, "--tods-movements-complete", "--output-dir", str(confirmed_dir)]) == 0
    )
    confirmed = pd.read_csv(confirmed_dir / target.PERIOD_FILENAME).iloc[0]
    assert confirmed["deadhead_hours"] > 0 and confirmed["vehicle_hours"] > 0
    # TODS deletes and adds apply to revenue hours either way.
    assert confirmed["revenue_trip_hours"] == unconfirmed["revenue_trip_hours"]
    log = (confirmed_dir / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "TODS movements:    confirmed complete" in log


def test_confirmed_partial_tods_warns_about_blocks_without_movements(
    small_feed: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs, tods = small_feed
    # A second block with no TODS movements.
    with (gtfs / "trips.txt").open("a", encoding="utf-8") as fh:
        fh.write("R,wk,T9,B2\n")
    with (gtfs / "stop_times.txt").open("a", encoding="utf-8") as fh:
        fh.write("T9,1,A,12:00:00,12:00:00\nT9,2,B,13:00:00,13:00:00\n")
    cfg = target.Config(
        gtfs_path=gtfs,
        tods_path=tods,
        output_dir=tmp_path / "out",
        periods=[("Nov", "20261101", "20261130")],
        tods_movements_complete=True,
    )
    with caplog.at_level("WARNING"):
        nov = target.run(cfg)["period"].iloc[0]
    assert "1 of 2 block(s) have TODS nonrevenue trips" in caplog.text
    assert nov["vehicle_hours"] > 0


# =============================================================================
# FORMULA WORKBOOK
# =============================================================================

SOFFICE = shutil.which("soffice")
needs_libreoffice = pytest.mark.skipif(SOFFICE is None, reason="LibreOffice not installed")


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, pd.DataFrame]]:
    out = tmp_path_factory.mktemp("workbook")
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=TODS_ZIP,
        output_dir=out,
        periods=[("November 2026", "20261101", "20261130"), ("Jan 2026", "20260101", "20260131")],
        tods_movements_complete=True,
    )
    return out, target.run(cfg)


def recalculate(path: Path, work: Path) -> Any:
    """Recalculate *path* in LibreOffice and return it with formula results."""
    out = work / "recalculated"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(SOFFICE),
            "--headless",
            "--calc",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(out),
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=180,
        env={**os.environ, "HOME": str(work / "lo-profile")},
    )
    return load_workbook(out / path.name, data_only=True)


def sheet_rows(ws: Any) -> list[dict[str, Any]]:
    header = [c.value for c in ws[1]]
    return [dict(zip(header, [c.value for c in row])) for row in ws.iter_rows(min_row=2)]


def period_totals(wb: Any) -> dict[str, dict[str, Any]]:
    return {
        r["Period"]: r for r in sheet_rows(wb["Periods"]) if r["Services"] == "All service sets"
    }


def test_workbook_formulas_trace_back_to_trip_times(mock_run: tuple) -> None:
    out, _ = mock_run
    wb = load_workbook(out / target.WORKBOOK_FILENAME)
    assert wb.sheetnames == [
        "Periods",
        "Settings",
        "Period days",
        "Calendar",
        "Services",
        "Blocks",
        "Trips",
    ]
    assert wb.calculation.fullCalcOnLoad
    assert wb["Settings"]["B2"].value is True  # LAYOVER_IN_REVENUE_HOURS default.
    periods = sheet_rows(wb["Periods"])
    service_row = next(r for r in periods if r["Services"] == "weekday")
    assert service_row["Service days"].startswith("=SUMIFS(Calendar!")
    assert (
        service_row["Revenue trip h"].startswith("=$E")
        and "Services!" in service_row["Revenue trip h"]
    )
    # The total sums the service rows, or is "n/a" while any of their blocks overlap.
    vehicle_total = period_totals(wb)["November 2026"]["Vehicle h"]
    assert vehicle_total.startswith("=IF(") and ',"n/a",SUM(' in vehicle_total
    assert period_totals(wb)["November 2026"]["Overlaps"].startswith("=SUM(")
    block = sheet_rows(wb["Blocks"])[0]
    assert block["Revenue trip h"].startswith("=SUMIFS(Trips!$K$")
    trip = sheet_rows(wb["Trips"])[1]
    assert trip["Duration h"] == "=(J3-I3)*24"
    assert wb["Trips"]["I2"].number_format == "[h]:mm:ss"  # GTFS times past 24:00.


@needs_libreoffice
def test_workbook_totals_match_the_csv(mock_run: tuple, tmp_path: Path) -> None:
    out, tables = mock_run
    wb = recalculate(out / target.WORKBOOK_FILENAME, tmp_path)
    labels = {**target.COMPONENT_LABELS, **target.DERIVED_LABELS}
    csv_periods = tables["period"].set_index("period")
    for name, row in period_totals(wb).items():
        for column, label in labels.items():
            assert row[label] == pytest.approx(csv_periods.loc[name, column], abs=0.01), label
    days = {r["Period"]: r for r in sheet_rows(wb["Period days"])}
    assert (days["Jan 2026"]["Weekdays"], days["Jan 2026"]["Holidays"]) == (20, 2)
    errors = [
        (ws.title, c.coordinate)
        for ws in wb
        for row in ws.iter_rows()
        for c in row
        if isinstance(c.value, str) and c.value.startswith("#")
    ]
    assert not errors


@needs_libreoffice
def test_workbook_schedule_day_counts_match_the_csv(tmp_path: Path) -> None:
    tables = christmas_on_sunday_service(tmp_path)
    wb = recalculate(tmp_path / "out" / target.WORKBOOK_FILENAME, tmp_path)
    christmas = next(r for r in sheet_rows(wb["Calendar"]) if r["Date"] and r["Date"].day == 25)
    assert (christmas["Day type"], christmas["Schedule day type"]) == ("Holiday", "Sunday")
    (days,) = sheet_rows(wb["Period days"])
    dec = tables["period"].iloc[0]
    for label in target.DAY_TYPES:
        column = f"{label.lower()}_schedule_days"
        assert days[f"{label} schedule days"] == dec[column], label
    assert days["Weekdays"] == 22 and days["Days without service"] == 0
    total = period_totals(wb)["Dec"]
    assert total["Revenue trip h"] == pytest.approx(dec["revenue_trip_hours"], abs=0.01)


@needs_libreoffice
def test_workbook_layover_switch_and_calendar_edits_recalculate(
    mock_run: tuple, tmp_path: Path
) -> None:
    out, tables = mock_run
    edited = tmp_path / "edited.xlsx"
    wb = load_workbook(out / target.WORKBOOK_FILENAME)
    wb["Settings"]["B2"] = False
    # Drop weekday service on the first November weekday (2026-11-02).
    cal = wb["Calendar"]
    header = [c.value for c in cal[1]]
    weekday_col = header.index("weekday") + 1
    row = next(
        r
        for r in range(2, cal.max_row + 1)
        if cal.cell(r, 1).value == "November 2026" and cal.cell(r, 2).value.day == 2
    )
    cal.cell(row, weekday_col).value = 0
    wb.save(edited)

    totals = period_totals(recalculate(edited, tmp_path))["November 2026"]
    weekday = tables["service"].set_index("service_ids").loc["weekday"]
    nov = tables["period"].set_index("period").loc["November 2026"]
    assert totals["Revenue trip h"] == pytest.approx(
        nov["revenue_trip_hours"] - weekday["revenue_trip_hours"], abs=0.01
    )
    # Layover now counts as nonrevenue time.
    assert totals["Vehicle revenue h"] == pytest.approx(totals["Revenue trip h"], abs=1e-6)
    moves = sum(totals[target.COMPONENT_LABELS[c]] for c in target.TODS_COMPONENTS)
    assert totals["Nonrevenue h"] == pytest.approx(moves + totals["Layover h"], abs=1e-6)


@needs_libreoffice
def test_workbook_matches_the_csv_for_blocks_across_services(tmp_path: Path) -> None:
    tables = everyday_plus_weekday_trip(tmp_path)
    out = tmp_path / "out"
    wb = recalculate(out / target.WORKBOOK_FILENAME, tmp_path)
    blocks = {r["Services"]: r for r in sheet_rows(wb["Blocks"])}
    assert blocks["all|wk"]["Layover h"] == pytest.approx(2.0)
    assert blocks["all"]["Unresolved gap h"] == pytest.approx(3.0)
    total = period_totals(wb)["Nov"]
    nov = tables["period"].iloc[0]
    for column in ("revenue_trip_hours", "layover_hours", "unresolved_gap_hours"):
        label = target.COMPONENT_LABELS[column]
        assert total[label] == pytest.approx(nov[column], abs=0.01), label
    assert total["Vehicle revenue h"] == pytest.approx(nov["vehicle_revenue_hours"], abs=0.01)


@needs_libreoffice
def test_workbook_matches_the_csv_for_overlapping_trips(tmp_path: Path) -> None:
    tables = overlapping_block(tmp_path)
    wb = recalculate(tmp_path / "out" / target.WORKBOOK_FILENAME, tmp_path)
    trips = {r["Trip ID"]: r for r in sheet_rows(wb["Trips"])}
    # Gaps run from the latest end so far (T1's 10:00), not the previous row's end.
    assert trips["T3"]["Gap before h"] == pytest.approx(-1.0)
    (block,) = sheet_rows(wb["Blocks"])
    assert block["Layover h"] == 0 and block["Overlaps"] == 2
    total = period_totals(wb)["Nov"]
    nov = tables["period"].iloc[0]
    assert total["Revenue trip h"] == pytest.approx(nov["revenue_trip_hours"], abs=0.01)
    assert total["Overlaps"] == nov["overlaps"]
    for label in target.DERIVED_LABELS.values():
        assert block[label] == total[label] == "n/a", label


def test_workbook_without_tods_marks_unknowns(tmp_path: Path) -> None:
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=None,
        output_dir=tmp_path,
        periods=[("Nov", "20261101", "20261130")],
    )
    target.run(cfg)
    wb = load_workbook(tmp_path / target.WORKBOOK_FILENAME)
    total = period_totals(wb)["Nov"]
    assert total["Deadhead h"] == total["Vehicle h"] == "n/a"
    assert total["Revenue trip h"].startswith("=SUM(")
