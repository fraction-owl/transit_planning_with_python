"""Behavioral tests for conservative matching and nonrevenue export."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import pytest

import scripts.operations_tools.tsde_to_tods as converter


def xml_bytes(tag: str, rows: list[dict]) -> bytes:
    root = Element("file")
    for row in rows:
        record = SubElement(root, tag)
        for key, value in row.items():
            SubElement(record, key).text = str(value)
    return tostring(root, encoding="utf-8", xml_declaration=True)


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def fixtures(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    # Identical numeric point IDs across types must remain distinct; leading zeros
    # in stop codes must survive. Direction IDs deliberately disagree across feeds.
    points = [
        {
            "pid": "00100",
            "ptype": "ga",
            "sname": "G",
            "lname": "Garage",
            "lat": 38700000,
            "lon": -77100000,
        },
        {
            "pid": "00100",
            "ptype": "no",
            "sname": "A",
            "lname": "Node A",
            "lat": 38710000,
            "lon": -77100000,
        },
        {
            "pid": "00100",
            "ptype": "st",
            "sname": "A",
            "lname": "Stop A",
            "lat": 38710000,
            "lon": -77100000,
        },
        {
            "pid": "200",
            "ptype": "no",
            "sname": "B",
            "lname": "Node B",
            "lat": 38720000,
            "lon": -77100000,
        },
        {
            "pid": "200",
            "ptype": "st",
            "sname": "B",
            "lname": "Stop B",
            "lat": 38720000,
            "lon": -77100000,
        },
        {
            "pid": "150",
            "ptype": "st",
            "sname": "M",
            "lname": "Middle",
            "lat": 38715000,
            "lon": -77100000,
        },
    ]
    trip_rows = []
    for tid, times in (("T1", (86400, 87300, 88200)), ("T2", (89400, 90000, 90600))):
        for sequence, pid, ptype, time in (
            (1, "00100", "no", times[0]),
            (1, "00100", "st", times[0]),
            (2, "150", "st", times[1]),
            (3, "200", "no", times[2]),
            (3, "200", "st", times[2]),
        ):
            trip_rows.append(
                dict(
                    verid="V",
                    sid="S",
                    tid=tid,
                    lid="L",
                    did="0",
                    ecid="0",
                    wktplt="0111110",
                    lpsnum=sequence,
                    pid=pid,
                    ptype=ptype,
                    dtime=time,
                )
            )
    events = [
        (1, 85800, "ga", "00100", "T1", "T1"),
        (15, 86400, "no", "00100", "T1", "T1"),
        (3, 86400, "no", "00100", "T1", "0"),
        (17, 88200, "no", "200", "T1", "T2"),
        (18, 88800, "no", "00100", "T1", "T2"),
        (3, 89400, "no", "00100", "T2", "0"),
        (16, 90600, "no", "200", "T2", "0"),
        (2, 91200, "ga", "00100", "T2", "0"),
    ]
    block_rows = [
        dict(
            verid="V",
            sid="S",
            bid="B",
            bnum="10",
            lgabbr="RIBS 1",
            did="D",
            snum=n,
            etype=code,
            etime=time,
            ptype=ptype,
            pid=pid,
            tid=tid,
            ttid=ttid,
            ecid="0",
            ecin="N",
        )
        for n, (code, time, ptype, pid, tid, ttid) in enumerate(events, 1)
    ]
    source = {
        "ver": [dict(verid="V", name="Test", fdate="20261109", tdate="20261115", prod="Y")],
        "pnt": points,
        "svc": [
            dict(sid="S", sname="Weekday", wktplt="0111110"),
            dict(sid="H", sname="Holiday", wktplt="0000000"),
        ],
        "cal": [dict(cdate="20261111", sid="H", ecid="0")],
        "lin": [dict(lid="L", sname="RIBS 1")],
        "trp": trip_rows,
        "blk": block_rows,
        "deh": [
            dict(sid="S", fpid="200", fptype="no", tpid="00100", tptype="ga", dhtime=-1, dist=0)
        ],
    }
    feed = {
        "agency.txt": [
            dict(
                agency_id="A",
                agency_name="Agency",
                agency_url="https://example.org",
                agency_timezone="America/New_York",
            )
        ],
        "routes.txt": [dict(route_id="R", route_short_name="RIBS1", agency_id="A", route_type="3")],
        "stops.txt": [
            dict(stop_id=key, stop_code=code, stop_lat=lat, stop_lon="-77.1", stop_name=key)
            for key, code, lat in (
                ("GA", "00100", "38.71"),
                ("GM", "150", "38.715"),
                ("GB", "200", "38.72"),
            )
        ],
        "trips.txt": [
            dict(trip_id=tid, route_id="R", service_id="GS", block_id="GBLOCK", direction_id="1")
            for tid in ("GT1", "GT2")
        ],
        "calendar.txt": [
            dict(
                service_id="GS",
                monday="1",
                tuesday="1",
                wednesday="1",
                thursday="1",
                friday="1",
                saturday="0",
                sunday="0",
                start_date="20261109",
                end_date="20261115",
            )
        ],
        "calendar_dates.txt": [dict(service_id="GS", date="20261111", exception_type="2")],
        "stop_times.txt": [],
    }
    for tid, times in (("GT1", (86400, 87305, 88200)), ("GT2", (89400, 90005, 90600))):
        for sequence, (stop, time) in enumerate(zip(("GA", "GM", "GB"), times), 1):
            feed["stop_times.txt"].append(
                dict(
                    trip_id=tid,
                    stop_id=stop,
                    stop_sequence=str(sequence),
                    arrival_time=converter.clock(time),
                    departure_time=converter.clock(time),
                    timepoint="0" if stop == "GM" else "1",
                )
            )
    tsde, gtfs = tmp_path / "tsde.zip", tmp_path / "gtfs.zip"
    save_inputs(tsde, gtfs, source, feed)
    return tsde, gtfs, source, feed


def save_inputs(tsde: Path, gtfs: Path, source: dict, feed: dict) -> None:
    with zipfile.ZipFile(tsde, "w") as archive:
        for tag, rows in source.items():
            archive.writestr(f"20261101_{tag}.xml", xml_bytes(tag, rows))
    with zipfile.ZipFile(gtfs, "w") as archive:
        for name, rows in feed.items():
            archive.writestr(name, csv_text(rows))


def execute(tmp_path: Path, tsde: Path, gtfs: Path | None) -> tuple[Path, dict]:
    out = converter.run(converter.Config(tsde=tsde, gtfs=gtfs, output=tmp_path / "results"))
    return out, json.loads((out / "manifest.json").read_text())["summary"]


def test_valid_overnight_feed_and_sentinel_lookup(tmp_path: Path) -> None:
    tsde, gtfs, _, _ = fixtures(tmp_path)
    out, summary = execute(tmp_path, tsde, gtfs)
    assert summary["exported_blocks"] == 1
    assert summary["exported_deadheads"] == 3
    assert summary["matched_trips"] == 2
    with zipfile.ZipFile(out / "draft_tods.zip") as archive:
        rows = list(
            csv.DictReader(
                io.TextIOWrapper(archive.open("stop_times_supplement.txt"), encoding="utf-8-sig")
            )
        )
    assert max(r["arrival_time"] for r in rows) == "25:20:00"
    assert all(r["pickup_type"] == r["drop_off_type"] == "1" for r in rows)
    lookup = list(csv.DictReader((out / "deadhead_lookup.csv").open(encoding="utf-8-sig")))
    assert lookup[0]["dhtime"] == "-1"
    assert lookup[0]["duration_defined"] == "False"
    assert summary["qa_counts"].get("SERVICE_UNMATCHED") == 1  # Unused holiday service.


@pytest.mark.parametrize(
    "change,expected",
    [
        ("zero_deadhead", "INVALID_MOVEMENT_TIME"),
        ("unknown_event", "UNKNOWN_EVENT_CODE"),
        ("bad_pair", "PAIR_TRIP_CONFLICT"),
        ("unmapped_trip", "BLOCK_UNMATCHED_TRIPS"),
        ("extra_gtfs_trip", "INCOMPLETE_GTFS_BLOCK"),
        ("wrong_calendar", "BLOCK_UNMATCHED_TRIPS"),
        ("wrong_timepoint", "BLOCK_UNMATCHED_TRIPS"),
        ("stop_order", "BLOCK_UNMATCHED_TRIPS"),
        ("overlap", "MOVEMENT_OVERLAP"),
    ],
)
def test_unsafe_blocks_are_not_exported(tmp_path: Path, change: str, expected: str) -> None:
    tsde, gtfs, source, feed = fixtures(tmp_path)
    if change == "zero_deadhead":
        source["blk"][4]["etime"] = 88200
    elif change == "unknown_event":
        source["blk"][0]["etype"] = 99
    elif change == "bad_pair":
        source["blk"][4]["ttid"] = "WRONG"
    elif change == "unmapped_trip":
        feed["trips.txt"][1]["route_id"] = "R2"
        feed["routes.txt"].append(
            dict(route_id="R2", route_short_name="OTHER", agency_id="A", route_type="3")
        )
    elif change == "extra_gtfs_trip":
        feed["trips.txt"].append(
            dict(
                trip_id="EXTRA", route_id="R", service_id="GS", block_id="GBLOCK", direction_id="1"
            )
        )
        feed["stop_times.txt"].extend(
            [
                dict(
                    trip_id="EXTRA",
                    stop_id="GA",
                    stop_sequence="1",
                    arrival_time="26:00:00",
                    departure_time="26:00:00",
                    timepoint="1",
                ),
                dict(
                    trip_id="EXTRA",
                    stop_id="GB",
                    stop_sequence="2",
                    arrival_time="26:30:00",
                    departure_time="26:30:00",
                    timepoint="1",
                ),
            ]
        )
    elif change == "wrong_calendar":
        feed["calendar_dates.txt"][0]["exception_type"] = "1"
    elif change == "wrong_timepoint":
        feed["stop_times.txt"][1]["timepoint"] = "1"
    elif change == "stop_order":
        feed["stop_times.txt"][0]["stop_id"] = "GM"
        feed["stop_times.txt"][1]["stop_id"] = "GA"
    elif change == "overlap":
        source["blk"][3]["etime"] = 88100
    save_inputs(tsde, gtfs, source, feed)
    out, summary = execute(tmp_path, tsde, gtfs)
    assert summary["exported_blocks"] == 0
    assert expected in summary["qa_counts"]
    assert not (out / "draft_tods.zip").exists()


def test_audit_only_without_gtfs(tmp_path: Path) -> None:
    tsde, _, _, _ = fixtures(tmp_path)
    out, summary = execute(tmp_path, tsde, None)
    assert summary["tsde_trips"] == 2
    assert summary["qa_counts"]["NO_GTFS"] == 1
    assert (out / "block_movements.csv").is_file()
    assert not (out / "draft_tods.zip").exists()


def test_multiple_exports_require_prefix(tmp_path: Path) -> None:
    tsde, _, source, _ = fixtures(tmp_path)
    with zipfile.ZipFile(tsde, "a") as archive:
        archive.writestr("20261102_ver.xml", xml_bytes("ver", source["ver"]))
    with pytest.raises(ValueError, match="Choose --prefix"):
        execute(tmp_path, tsde, None)


def test_folder_input(tmp_path: Path) -> None:
    tsde, gtfs, _, _ = fixtures(tmp_path)
    folder = tmp_path / "source_folder"
    with zipfile.ZipFile(tsde) as archive:
        archive.extractall(folder)
    _, summary = execute(tmp_path, folder, gtfs)
    assert summary["exported_deadheads"] == 3


def test_nonchronological_times_are_not_wrapped_to_next_day() -> None:
    assert converter.seconds("25:20:00") == 91200
    assert converter.clock(91200) == "25:20:00"
    with pytest.raises(ValueError):
        converter.clock(-1)
    with pytest.raises(ValueError):
        converter.seconds("25:61:00")


def test_require_all_blocks_suppresses_partial_feed(tmp_path: Path) -> None:
    tsde, gtfs, source, feed = fixtures(tmp_path)
    # Keep the valid block, but introduce another block that cannot be exported.
    source["blk"].append({**source["blk"][0], "bid": "BAD", "etype": 99})
    save_inputs(tsde, gtfs, source, feed)
    out = converter.run(
        converter.Config(tsde=tsde, gtfs=gtfs, output=tmp_path / "all", require_all_blocks=True)
    )
    summary = json.loads((out / "manifest.json").read_text())["summary"]
    assert summary["exported_blocks"] == 0
    assert summary["qa_counts"]["EXPORT_SUPPRESSED"] == 1
    assert not (out / "draft_tods.zip").exists()


def test_intermediate_point_identifier_types_remain_distinct(tmp_path: Path) -> None:
    tsde, _, _, _ = fixtures(tmp_path)
    source = converter.Source(tsde)
    try:
        data = converter.read_tsde(source, converter.Config(), converter.QA())
    finally:
        source.close()
    assert ("st", "00100") in data.points
    assert ("no", "00100") in data.points
    assert ("ga", "00100") in data.points
    assert data.points["st", "00100"]["stop_code"] == "00100"
