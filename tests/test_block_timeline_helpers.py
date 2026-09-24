"""Tests for utils/block_timeline_helpers.py and the script copies that must match it."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from utils.block_timeline_helpers import (
    build_schedule_rows,
    find_cluster,
    gap_status,
    read_run_manifest,
    read_timeline_files,
    row_for_inactive,
    status_for_same_trip,
    timestamp_to_minutes,
    verified_run_file,
)

REPO = Path(__file__).resolve().parents[1]
SETTINGS = {
    "THROUGH_DWELL_MINUTES": 2,
    "PRE_DEPARTURE_MINUTES": 5,
    "POST_ARRIVAL_MINUTES": 2,
    "IN_BAY_LAYOVER_MAX_MINUTES": 10,
    "LAYOVER_THRESHOLD": 20,
}

# ---------------------------------------------------------------------------
# Script copies must match utils/ (the sweep's rescoring is exact only while its
# renderer matches the exporter's)
# ---------------------------------------------------------------------------

RENDERER = {
    "find_cluster",
    "status_for_same_trip",
    "gap_status",
    "make_timeline_row",
    "row_for_inactive",
    "build_schedule_rows",
}
READERS = {"timestamp_to_minutes", "read_run_manifest", "verified_run_file", "read_timeline_files"}
COPIES = {
    "scripts/gtfs_exports/block_status_timeline_exporter.py": RENDERER,
    "scripts/facilities_tools/bay_usage_analyzer.py": READERS,
    "scripts/facilities_tools/bay_change_sweep.py": RENDERER | READERS,
}


def _functions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}


def _canonical() -> dict[str, tuple[str, str]]:
    """Public functions defined in utils/, collected as the helper audit collects them."""
    found: dict[str, tuple[str, str]] = {}
    for path in sorted((REPO / "utils").glob("*.py")):
        for name, dump in _functions(path).items():
            if not name.startswith("_"):
                found.setdefault(name, (path.name, dump))
    return found


@pytest.mark.parametrize("script", sorted(COPIES))
def test_script_helper_copies_match_utils(script: str) -> None:
    canonical = _canonical()
    local = _functions(REPO / script)
    missing = sorted(COPIES[script] - set(local))
    assert not missing, f"{script} no longer defines shared helpers: {missing}"
    drifted = sorted(
        f"{name} (utils/{canonical[name][0]})"
        for name in set(local) & set(canonical)
        if local[name] != canonical[name][1]
    )
    assert not drifted, f"{script} differs from utils/ in: {', '.join(drifted)}"


# ---------------------------------------------------------------------------
# Block renderer
# ---------------------------------------------------------------------------


def _stop(
    arr: int, dep: int, stop: str, first: bool, last: bool, seq: int, trip: str = "T1"
) -> tuple:
    return (arr, dep, stop, f"Stop {stop}", trip, first, last, seq, 0)


def _trip(trip_id: str, visits: list[tuple]) -> dict[str, Any]:
    return {
        "trip_id": trip_id,
        "start": visits[0][0],
        "end": visits[-1][1],
        "stop_times_sequence": visits,
        "route_id": "R1",
        "route_short_name": "1",
        "trip_headsign": "Downtown",
        "direction_id": "0",
        "first_stop_id": visits[0][2],
        "first_stop_name": visits[0][3],
        "first_stop_seq": visits[0][7],
        "last_stop_id": visits[-1][2],
        "last_stop_name": visits[-1][3],
        "last_stop_seq": visits[-1][7],
    }


def test_find_cluster() -> None:
    clusters = [{"name": "Hub", "stops": ["S1", "S2"]}]
    assert find_cluster("S2", clusters) == "Hub"
    assert find_cluster("S9", clusters) is None


def test_status_for_same_trip_hold_occupies_both_boundaries() -> None:
    hold = _stop(430, 431, "S2", first=False, last=False, seq=2)
    statuses = [status_for_same_trip(m, hold, SETTINGS) for m in (429, 430, 431, 432)]
    assert [s[0] if s else None for s in statuses] == [None, "DWELL", "DWELL", None]


def test_status_for_same_trip_first_stop_and_through_stop() -> None:
    origin = _stop(420, 430, "S1", first=True, last=False, seq=1)
    assert [status_for_same_trip(m, origin, SETTINGS)[0] for m in (420, 425, 430)] == [
        "DWELL",
        "LOADING",
        "DEPART",
    ]
    through = _stop(440, 440, "S2", first=False, last=False, seq=2)
    statuses = [status_for_same_trip(m, through, SETTINGS) for m in (440, 441, 442)]
    assert [s[0] if s else None for s in statuses] == ["ARRIVE/DEPART", "ARRIVE/DEPART", None]


@pytest.mark.parametrize(
    ("gap", "same_place", "expected"),
    [
        (10, True, ("DWELL", "in bay")),
        (11, True, ("LAYOVER", "overflow")),
        (21, True, ("LONG BREAK", "overflow")),
        (5, False, ("DEADHEAD", "")),
    ],
)
def test_gap_status(gap: int, same_place: bool, expected: tuple[str, str]) -> None:
    assert gap_status(gap, same_place, SETTINGS) == expected


def test_row_for_inactive_arrival_buffer_then_loading() -> None:
    trips = [
        _trip("T1", [_stop(400, 400, "S9", True, False, 1), _stop(420, 420, "S1", False, True, 2)]),
        _trip("T2", [_stop(430, 430, "S1", True, False, 1), _stop(450, 450, "S9", False, True, 2)]),
    ]
    statuses = [row_for_inactive(m, "B1", trips, [], SETTINGS)["Status"] for m in range(421, 430)]
    # Two minutes of ARRIVE after the 420 arrival, in-bay DWELL, then five of LOADING.
    assert statuses == ["ARRIVE"] * 2 + ["DWELL"] * 2 + ["LOADING"] * 5


def test_build_schedule_rows_through_dwell_stops_at_next_arrival() -> None:
    trip = _trip(
        "T1",
        [
            _stop(420, 420, "S1", True, False, 1),
            _stop(430, 430, "S2", False, False, 2),
            _stop(431, 433, "S3", False, False, 3),
            _stop(440, 440, "S4", False, True, 4),
        ],
    )
    rows = build_schedule_rows([trip], range(429, 435), "B1", [], SETTINGS)
    assert [(r["Status"], r["Stop ID"]) for r in rows] == [
        ("TRAVELING BETWEEN STOPS", ""),
        ("ARRIVE/DEPART", "S2"),
        ("DWELL", "S3"),  # S2's two-minute dwell ends when the bus reaches S3
        ("DWELL", "S3"),
        ("DWELL", "S3"),
        ("TRAVELING BETWEEN STOPS", ""),
    ]


def test_build_schedule_rows_occupancy_only_and_zero_gap_handoff() -> None:
    arriving = _trip(
        "T1", [_stop(400, 400, "S9", True, False, 1), _stop(420, 420, "S1", False, True, 2)]
    )
    departing = _trip(
        "T2",
        [_stop(420, 420, "S1", True, False, 1, "T2"), _stop(440, 440, "S9", False, True, 2, "T2")],
    )
    rows = build_schedule_rows([arriving, departing], range(380, 460), "B1", [], SETTINGS)
    assert len(rows) == 80  # one row per minute
    handoff = next(r for r in rows if r["Timestamp"] == "07:00")
    assert (handoff["Status"], handoff["Trip ID"]) == ("DEPART", "T2")
    occupancy = build_schedule_rows(
        [arriving, departing], range(380, 460), "B1", [], SETTINGS, True
    )
    assert {r["Status"] for r in occupancy} <= {"LOADING", "DEPART", "ARRIVE"}
    assert occupancy[0]["Timestamp"] == "06:35"  # pull-out LOADING, five minutes before 06:40


def test_build_schedule_rows_requires_one_minute_steps() -> None:
    trip = _trip(
        "T1", [_stop(400, 400, "S9", True, False, 1), _stop(420, 420, "S1", False, True, 2)]
    )
    with pytest.raises(ValueError, match="TIME_INTERVAL_MINUTES = 1"):
        build_schedule_rows([trip], range(0, 60, 5), "B1", [], SETTINGS)


# ---------------------------------------------------------------------------
# Step 1 timeline readers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        ("08:05", 485),
        ("26:10", 1570),
        (" 7:30 ", 450),
        ("08:05:00", None),
        ("08:60", None),
        ("", None),
    ],
)
def test_timestamp_to_minutes(ts: str, expected: int | None) -> None:
    assert timestamp_to_minutes(ts) == expected


def _rows(block: str, minutes: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": [f"{m // 60:02d}:{m % 60:02d}" for m in minutes],
            "Block": block,
            "Trip ID": "T1",
            "Stop ID": "S1",
            "Status": "ARRIVE",
        }
    )


def _write_run(folder: Path, frame: pd.DataFrame, **manifest: Any) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "timeline.csv", index=False)
    digest = hashlib.sha256((folder / "timeline.csv").read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "complete",
        "interval_minutes": 1,
        "combined_timeline": "timeline.csv",
        "block_workbooks": [],
        "files": {"timeline.csv": digest},
        **manifest,
    }
    (folder / "timeline_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_read_run_manifest_absent_incomplete_and_wrong_interval(tmp_path: Path) -> None:
    assert read_run_manifest(str(tmp_path)) is None
    _write_run(tmp_path, _rows("B1", range(480, 482)), status="in_progress")
    with pytest.raises(ValueError, match="not complete"):
        read_run_manifest(str(tmp_path))
    _write_run(tmp_path, _rows("B1", range(480, 482)), interval_minutes=5)
    with pytest.raises(ValueError, match="one-minute"):
        read_run_manifest(str(tmp_path))


def test_verified_run_file_detects_edits_and_foreign_names(tmp_path: Path) -> None:
    _write_run(tmp_path, _rows("B1", range(480, 482)))
    manifest = read_run_manifest(str(tmp_path))
    assert manifest is not None
    assert verified_run_file(str(tmp_path), "timeline.csv", manifest).name == "timeline.csv"
    with pytest.raises(ValueError, match="filename inside the run folder"):
        verified_run_file(str(tmp_path), "../timeline.csv", manifest)
    with pytest.raises(ValueError, match="Missing current-run file"):
        verified_run_file(str(tmp_path), "other.csv", manifest)
    with (tmp_path / "timeline.csv").open("a", encoding="utf-8") as handle:
        handle.write("B1,08:02,T1,S1,ARRIVE\n")
    with pytest.raises(ValueError, match="has changed"):
        verified_run_file(str(tmp_path), "timeline.csv", manifest)


def test_read_timeline_files_ignores_stale_files_from_earlier_runs(tmp_path: Path) -> None:
    _rows("B9", range(600, 602)).to_excel(tmp_path / "block_B9_R1.xlsx", index=False)  # stale
    _write_run(tmp_path, _rows("B1", range(480, 483)))
    frame = read_timeline_files(str(tmp_path), "timeline.csv")
    assert set(frame["Block"]) == {"B1"}
    with pytest.raises(ValueError, match="does not match the exporter manifest"):
        read_timeline_files(str(tmp_path), "all_blocks_timeline.csv")


def test_read_timeline_files_legacy_folder_is_read_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _rows("B1", range(480, 482)).to_csv(tmp_path / "timeline.csv", index=False)
    assert len(read_timeline_files(str(tmp_path), "timeline.csv")) == 2
    assert "Legacy timeline" in caplog.text


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.concat([_rows("B1", range(480, 482)), _rows("B1", range(481, 482))]), "Duplicate"),
        (pd.concat([_rows("B1", range(480, 482)), _rows("B1", range(485, 486))]), "consecutive"),
        (_rows("B1", range(480, 482)).drop(columns="Status"), "missing columns"),
    ],
)
def test_read_timeline_files_rejects_malformed_timelines(
    tmp_path: Path, frame: pd.DataFrame, message: str
) -> None:
    frame.to_csv(tmp_path / "timeline.csv", index=False)
    with pytest.raises(ValueError, match=message):
        read_timeline_files(str(tmp_path), "timeline.csv")
