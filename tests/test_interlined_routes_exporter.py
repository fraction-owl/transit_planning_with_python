from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import scripts.gtfs_exports.interlined_routes_exporter as target

# (trip_id, route_id, service_id, block_id, [(stop_id, HH:MM), ...])
TRIPS = [
    # Weekday block B1: 10 -> 20 (same stop B) -> 10 (same stop C) -> 30 (deadhead A to D).
    ("T1", "R10", "WK", "B1", [("A", "07:00"), ("B", "07:30")]),
    ("T2", "R20", "WK", "B1", [("B", "07:40"), ("C", "08:10")]),
    ("T3", "R10", "WK", "B1", [("C", "08:30"), ("A", "09:00")]),
    ("T4", "R30", "WK", "B1", [("D", "09:30"), ("E", "10:00")]),
    # Weekday block B2: route 10 only.
    ("T5", "R10", "WK", "B2", [("A", "07:15"), ("B", "07:45")]),
    ("T6", "R10", "WK", "B2", [("B", "08:00"), ("A", "08:30")]),
    # Weekday block B3: 20 -> 10 at stop B.
    ("T7", "R20", "WK", "B3", [("C", "06:00"), ("B", "06:30")]),
    ("T8", "R10", "WK", "B3", [("B", "06:35"), ("A", "07:05")]),
    # Weekday route 40 trip with no block_id.
    ("T9", "R40", "WK", "", [("E", "12:00"), ("D", "12:30")]),
    # Saturday reuses block_id B1 for a different vehicle day: 40 -> 30 at stop D.
    ("T10", "R40", "SA", "B1", [("E", "09:00"), ("D", "09:30")]),
    ("T11", "R30", "SA", "B1", [("D", "09:45"), ("E", "10:15")]),
]


def _write_feed(parent: Path, name: str = "feed", with_blocks: bool = True) -> Path:
    feed_dir = parent / name
    feed_dir.mkdir(parents=True)
    (feed_dir / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name,route_type\n"
        "R10,10,Main Street,3\n"
        "R20,20,Crosstown,3\n"
        "R30,30,Airport,3\n"
        "R40,40,Riverside,3\n",
        encoding="utf-8",
    )
    (feed_dir / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "WK,1,1,1,1,1,0,0,20260105,20260130\n"
        "SA,0,0,0,0,0,1,0,20260105,20260130\n",
        encoding="utf-8",
    )
    (feed_dir / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        + "".join(f"{s},Stop {s},38.9,-77.0{i}\n" for i, s in enumerate("ABCDE")),
        encoding="utf-8",
    )
    trip_rows = ["route_id,service_id,trip_id,trip_headsign" + (",block_id" if with_blocks else "")]
    time_rows = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    for trip_id, route_id, service_id, block_id, stops in TRIPS:
        row = f"{route_id},{service_id},{trip_id},To {stops[-1][0]}"
        trip_rows.append(row + (f",{block_id}" if with_blocks else ""))
        for seq, (stop_id, hhmm) in enumerate(stops, start=1):
            time_rows.append(f"{trip_id},{hhmm}:00,{hhmm}:00,{stop_id},{seq}")
    (feed_dir / "trips.txt").write_text("\n".join(trip_rows) + "\n", encoding="utf-8")
    (feed_dir / "stop_times.txt").write_text("\n".join(time_rows) + "\n", encoding="utf-8")
    return feed_dir


def _zip_with_wrapper(feed_dir: Path) -> Path:
    zip_path = feed_dir.parent / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in feed_dir.iterdir():
            archive.write(path, f"wrapper/{path.name}")
    shutil.rmtree(feed_dir)
    return zip_path


@pytest.fixture
def tables(tmp_path: Path) -> dict[str, pd.DataFrame]:
    feed = _write_feed(tmp_path)
    return target.run(
        gtfs_path=feed, output_dir=tmp_path / "out", day_types=["weekday", "saturday"]
    )


def _row(df: pd.DataFrame, **match: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, value in match.items():
        mask &= df[col] == value
    assert mask.sum() == 1, f"expected one row for {match}, got {int(mask.sum())}"
    return df[mask].iloc[0]


def test_pairs_count_shared_blocks_and_hand_offs(tables: dict[str, pd.DataFrame]) -> None:
    weekday = tables["pairs"][tables["pairs"]["day_type"] == "weekday"]
    assert list(zip(weekday["route_a"], weekday["route_b"])) == [
        ("10", "20"),
        ("10", "30"),
        ("20", "30"),
    ]

    p = _row(weekday, route_a="10", route_b="20")
    assert p["shared_blocks"] == 2
    assert p["block_ids"] == "B1; B3"
    assert (p["hand_offs"], p["hand_offs_a_to_b"], p["hand_offs_b_to_a"]) == (3, 1, 2)
    assert p["hand_offs_same_stop"] == 3
    assert (p["min_layover_min"], p["max_layover_min"]) == (5, 20)

    p = _row(weekday, route_a="10", route_b="30")
    assert (p["shared_blocks"], p["hand_offs"], p["hand_offs_same_stop"]) == (1, 1, 0)
    assert p["min_layover_min"] == 30

    # 20 and 30 share block B1 but never follow each other on it.
    p = _row(weekday, route_a="20", route_b="30")
    assert (p["shared_blocks"], p["hand_offs"]) == (1, 0)
    assert pd.isna(p["min_layover_min"])


def test_block_ids_are_scoped_to_the_service_date(tables: dict[str, pd.DataFrame]) -> None:
    saturday = tables["pairs"][tables["pairs"]["day_type"] == "saturday"]
    assert list(zip(saturday["route_a"], saturday["route_b"])) == [("30", "40")]
    p = saturday.iloc[0]
    assert (p["hand_offs_a_to_b"], p["hand_offs_b_to_a"], p["min_layover_min"]) == (0, 1, 15)


def test_route_table_lists_partners_and_unblocked_trips(tables: dict[str, pd.DataFrame]) -> None:
    weekday = tables["routes"][tables["routes"]["day_type"] == "weekday"]
    assert list(weekday["route"]) == ["10", "20", "30", "40"]

    r10 = _row(weekday, route="10")
    assert (r10["trips"], r10["blocks"], r10["interlined_blocks"]) == (5, 3, 2)
    assert r10["interlines_with"] == "20, 30"
    assert r10["route_long_name"] == "Main Street"

    r40 = _row(weekday, route="40")
    assert (r40["trips"], r40["trips_without_block_id"], r40["blocks"]) == (1, 1, 0)
    assert r40["interlines_with"] == ""


def test_hand_off_detail(tables: dict[str, pd.DataFrame]) -> None:
    weekday = tables["hand_offs"][tables["hand_offs"]["day_type"] == "weekday"]
    assert list(zip(weekday["block_id"], weekday["from_trip_id"], weekday["to_trip_id"])) == [
        ("B1", "T1", "T2"),
        ("B1", "T2", "T3"),
        ("B1", "T3", "T4"),
        ("B3", "T7", "T8"),
    ]
    deadhead = _row(weekday, from_trip_id="T3")
    assert (deadhead["arrives"], deadhead["departs"], deadhead["layover_min"]) == (
        "09:00",
        "09:30",
        30,
    )
    assert (deadhead["from_end_stop_name"], deadhead["to_start_stop_name"]) == ("Stop A", "Stop D")
    assert not deadhead["same_stop"]
    assert deadhead["from_headsign"] == "To A"


def test_outputs_and_run_log_written(tmp_path: Path) -> None:
    feed = _zip_with_wrapper(_write_feed(tmp_path))
    out = tmp_path / "out"
    assert target.main(["--gtfs-path", str(feed), "--output-dir", str(out)]) == 0

    for name in (
        target.ROUTES_FILENAME,
        target.PAIRS_FILENAME,
        target.HAND_OFFS_FILENAME,
        target.RUN_LOG_FILENAME,
    ):
        assert (out / name).is_file(), name
    pairs = pd.read_csv(out / target.PAIRS_FILENAME, dtype=str)
    assert set(pairs["day_type"]) == {"weekday", "saturday"}  # no Sunday service in the feed
    log = (out / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "# === BEGIN CONFIG ===" in log and "DAY_TYPES" in log
    assert "weekday: 2026-01-" in log


def test_service_date_replaces_day_types(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    result = target.run(gtfs_path=feed, output_dir=tmp_path / "out", service_date="20260110")
    assert set(result["pairs"]["day_type"]) == {target.SERVICE_DATE_LABEL}
    assert set(result["pairs"]["service_date"]) == {"2026-01-10"}
    assert list(zip(result["pairs"]["route_a"], result["pairs"]["route_b"])) == [("30", "40")]


def test_overlapping_trips_on_a_block_are_counted() -> None:
    block_trips = pd.DataFrame(
        {
            "trip_id": ["X1", "X2"],
            "route_id": ["R1", "R2"],
            "block_id": ["BX", "BX"],
            "trip_headsign": ["", ""],
            "start_stop_id": ["A", "B"],
            "end_stop_id": ["B", "A"],
            "start_min": [600.0, 620.0],
            "end_min": [630.0, 650.0],
        }
    )
    hand_offs, overlapping = target.find_hand_offs(block_trips)
    assert overlapping == 1
    assert hand_offs["layover_min"].tolist() == [-10.0]


def test_feed_without_block_id_fails(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path, with_blocks=False)
    assert target.main(["--gtfs-path", str(feed), "--output-dir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out" / target.PAIRS_FILENAME).exists()


@pytest.mark.parametrize(
    "extra",
    [["--day-types", "funday"], ["--service-date", "2026-13-01"]],
)
def test_bad_settings_exit_2(tmp_path: Path, extra: list[str]) -> None:
    feed = _write_feed(tmp_path)
    argv = ["--gtfs-path", str(feed), "--output-dir", str(tmp_path / "out"), *extra]
    assert target.main(argv) == 2


def test_placeholder_paths_exit_2() -> None:
    assert target.main([]) == 2
