from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import scripts.gtfs_exports.route_stops_exporter as target


def _write_feed(parent: Path, name: str = "feed", zipped: bool = False) -> Path:
    """Write a tiny two-route feed.

    Route 10 (route_id R10, weekday WKD + Saturday SAT) runs A-B-C outbound
    (direction 0) and C-B-A inbound (direction 1); one weekday short-turn runs
    A-B only. Route 20 (route_id R20, weekday only) runs B-D-STN, where STN is
    a station (location_type 1). Route 30 (R30) never runs on weekdays and
    serves only stop D.
    """
    feed_dir = parent / name
    feed_dir.mkdir(parents=True)
    (feed_dir / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name,route_type\n"
        "R10,10,Main Street,3\n"
        "R20,20,Crosstown,3\n"
        "R30,30,Sunday Loop,3\n",
        encoding="utf-8",
    )
    (feed_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id\n"
        "R10,WKD,T1,0\n"
        "R10,WKD,T2,0\n"
        "R10,WKD,T3,1\n"
        "R10,SAT,T4,0\n"
        "R20,WKD,T5,0\n"
        "R30,SUN,T6,0\n",
        encoding="utf-8",
    )
    (feed_dir / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T1,08:00:00,08:00:00,A,1\n"
        "T1,08:05:00,08:05:00,B,2\n"
        "T1,08:10:00,08:10:00,C,3\n"
        "T2,09:00:00,09:00:00,A,1\n"
        "T2,09:05:00,09:05:00,B,2\n"
        "T3,10:00:00,10:00:00,C,1\n"
        "T3,10:05:00,10:05:00,B,2\n"
        "T3,10:10:00,10:10:00,A,3\n"
        "T4,11:00:00,11:00:00,A,1\n"
        "T4,11:05:00,11:05:00,B,2\n"
        "T4,11:10:00,11:10:00,C,3\n"
        "T5,12:00:00,12:00:00,B,1\n"
        "T5,12:05:00,12:05:00,D,2\n"
        "T5,12:10:00,12:10:00,STN,3\n"
        "T6,13:00:00,13:00:00,D,1\n",
        encoding="utf-8",
    )
    (feed_dir / "stops.txt").write_text(
        "stop_id,stop_code,stop_name,stop_lat,stop_lon,location_type\n"
        "A,1001,Alpha St,38.90,-77.03,0\n"
        "B,1002,Bravo Ave,38.91,-77.02,\n"
        "C,1003,Charlie Rd,38.92,-77.01,0\n"
        "D,1004,Delta Blvd,38.93,-77.00,0\n"
        "STN,1005,Echo Station,38.94,-76.99,1\n",
        encoding="utf-8",
    )
    if zipped:
        zip_path = parent / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for member in sorted(feed_dir.iterdir()):
                archive.write(member, arcname=member.name)
        shutil.rmtree(feed_dir)
        return zip_path
    return feed_dir


def test_detail_and_unique_tables_for_one_route(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    out = tmp_path / "out"
    detail, unique = target.run(gtfs_dir=feed, output_dir=out, route_names=["10"])

    # One row per (direction, stop): 3 outbound + 3 inbound.
    assert len(detail) == 6
    assert set(detail["route_id"]) == {"R10"}
    outbound = detail[detail["direction_id"] == "0"]
    assert list(outbound["stop_id"]) == ["A", "B", "C"]
    assert list(outbound["typical_stop_sequence"]) == [1.0, 2.0, 3.0]
    # A and B are served by all three outbound trips; C only by the two full-length ones.
    assert dict(zip(outbound["stop_id"], outbound["n_trips"])) == {"A": 3, "B": 3, "C": 2}
    inbound = detail[detail["direction_id"] == "1"]
    assert list(inbound["stop_id"]) == ["C", "B", "A"]
    assert set(detail.columns) >= {
        "route_short_name",
        "route_long_name",
        "stop_code",
        "stop_name",
        "stop_lat",
        "stop_lon",
    }

    assert list(unique["stop_id"]) == ["A", "B", "C"]  # sorted by stop_name
    by_stop = unique.set_index("stop_id")
    assert by_stop.loc["B", "other_routes"] == "20"
    assert by_stop.loc["A", "other_routes"] == ""
    assert (by_stop["selected_routes"] == "10").all()
    assert (by_stop["n_selected_routes"] == 1).all()

    assert (out / target.DETAIL_FILENAME).exists()
    assert (out / target.UNIQUE_FILENAME).exists()
    runlog = out / target.RUN_LOG_FILENAME
    assert runlog.exists()
    text = runlog.read_text(encoding="utf-8")
    assert "# === BEGIN CONFIG ===" in text
    assert "Route names:        10" in text


def test_multiple_routes_platform_filter_and_route_id_names(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    detail, unique = target.run(
        gtfs_dir=feed, output_dir=tmp_path / "out", route_names=["10", "R20"]
    )
    # Station STN is dropped by the platform filter.
    assert "STN" not in set(detail["stop_id"])
    assert set(unique["stop_id"]) == {"A", "B", "C", "D"}
    by_stop = unique.set_index("stop_id")
    assert by_stop.loc["B", "selected_routes"] == "10, 20"
    assert by_stop.loc["B", "n_selected_routes"] == 2
    assert by_stop.loc["B", "other_routes"] == ""
    assert by_stop.loc["D", "other_routes"] == "30"


def test_non_platform_stops_kept_when_filter_off(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    detail, _ = target.run(
        gtfs_dir=feed,
        output_dir=tmp_path / "out",
        route_names=["20"],
        platform_stops_only=False,
    )
    assert list(detail["stop_id"]) == ["B", "D", "STN"]


def test_service_id_filter_limits_trips_and_other_routes(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    detail, unique = target.run(
        gtfs_dir=feed, output_dir=tmp_path / "out", route_names=["20"], service_ids=["WKD"]
    )
    assert set(detail["stop_id"]) == {"B", "D"}
    by_stop = unique.set_index("stop_id")
    # Route 30 only runs on SUN, so it no longer counts as another route at D.
    assert by_stop.loc["D", "other_routes"] == ""
    assert by_stop.loc["B", "other_routes"] == "10"

    with pytest.raises(ValueError, match="SERVICE_IDS"):
        target.run(
            gtfs_dir=feed,
            output_dir=tmp_path / "out2",
            route_names=["10"],
            service_ids=["NOPE"],
        )


def test_routes_file_and_unmatched_names_warning(tmp_path: Path, caplog) -> None:
    feed = _write_feed(tmp_path, zipped=True)
    names_file = tmp_path / "routes.txt"
    names_file.write_text("# my routes\n10  # main street\n\n999\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        detail, _ = target.run(
            gtfs_dir=feed,
            output_dir=tmp_path / "out",
            route_names=[],
            route_names_file=str(names_file),
        )
    assert set(detail["route_id"]) == {"R10"}
    assert "999" in caplog.text


def test_no_matching_route_raises(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    with pytest.raises(ValueError, match="matched"):
        target.run(gtfs_dir=feed, output_dir=tmp_path / "out", route_names=["999"])


def test_identify_target_route_ids_matches_short_name_then_id() -> None:
    routes = pd.DataFrame(
        {
            "route_id": ["R10-WKD", "R10-SAT", "R20"],
            "route_short_name": ["10", "10", "20"],
        }
    )
    matched, unmatched = target.identify_target_route_ids(routes, {"10", "R20", "77"})
    assert matched == {"R10-WKD", "R10-SAT", "R20"}
    assert unmatched == {"77"}


def test_main_returns_2_on_placeholder_paths() -> None:
    assert target.main([]) == 2


def test_main_returns_2_when_no_routes_selected(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    rc = target.main(["--gtfs-dir", str(feed), "--output-dir", str(tmp_path / "out")])
    assert rc == 2
    assert not (tmp_path / "out").exists()


def test_main_cli_flags_run_end_to_end(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    out = tmp_path / "out"
    rc = target.main(
        [
            "--gtfs-dir",
            str(feed),
            "--output-dir",
            str(out),
            "--routes",
            "10",
            "20",
            "--no-platform-stops-only",
        ]
    )
    assert rc == 0
    unique = pd.read_csv(out / target.UNIQUE_FILENAME, dtype=str)
    assert set(unique["stop_id"]) == {"A", "B", "C", "D", "STN"}


def test_main_returns_1_on_bad_feed(tmp_path: Path) -> None:
    rc = target.main(
        [
            "--gtfs-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "out"),
            "--routes",
            "10",
        ]
    )
    assert rc == 1
