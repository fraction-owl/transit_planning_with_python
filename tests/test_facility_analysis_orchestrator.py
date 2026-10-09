from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import scripts.facilities_tools.facility_analysis_orchestrator as target

# ---------------------------------------------------------------------------
# Synthetic feed: route 10 ends at bay S1 and starts again there; route 20 runs
# through bay S3 of a second cluster; route 30 never visits a cluster; a
# Saturday trip of route 10 must not count on the weekday.
# ---------------------------------------------------------------------------
STOPS = ["S1", "S2", "S3", "S9"]
ROUTES = [("R10", "10", "Ten Local"), ("R20", "", "Twenty Express"), ("R30", "30", "Thirty")]
# trip_id, route_id, service_id, direction_id, [(stop, time)]
TRIPS: list[tuple[str, str, str, str, list[tuple[str, str]]]] = [
    ("T1", "R10", "WK", "0", [("S9", "07:50:00"), ("S1", "08:00:00")]),
    ("T2", "R10", "WK", "1", [("S1", "08:10:00"), ("S9", "08:20:00")]),
    ("T3", "R20", "WK", "0", [("S9", "09:00:00"), ("S3", "09:05:00"), ("S2", "09:10:00")]),
    ("T4", "R30", "WK", "0", [("S9", "10:00:00"), ("S2", "10:10:00")]),
    ("T5", "R10", "SA", "0", [("S9", "11:50:00"), ("S1", "12:00:00")]),
]
CLUSTERS = {"North Bays": {"stops": ["S1", "S2"]}, "South Bay": {"stops": ["S3"]}}


def write_gtfs(folder: Path) -> Path:
    """Write the synthetic feed (weekday service WK, Saturday service SA) to *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stop_id": STOPS, "stop_name": STOPS}).to_csv(folder / "stops.txt", index=False)
    pd.DataFrame(ROUTES, columns=["route_id", "route_short_name", "route_long_name"]).to_csv(
        folder / "routes.txt", index=False
    )
    trips = [
        {"trip_id": t, "route_id": r, "service_id": s, "direction_id": d, "block_id": "B" + t}
        for t, r, s, d, _ in TRIPS
    ]
    pd.DataFrame(trips).to_csv(folder / "trips.txt", index=False)
    rows = [
        {
            "trip_id": t,
            "stop_id": stop,
            "stop_sequence": seq,
            "arrival_time": clock,
            "departure_time": clock,
        }
        for t, _, _, _, visits in TRIPS
        for seq, (stop, clock) in enumerate(visits, start=1)
    ]
    pd.DataFrame(rows).to_csv(folder / "stop_times.txt", index=False)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    calendar = [
        {"service_id": "WK", **{d: int(i < 5) for i, d in enumerate(days)}},
        {"service_id": "SA", **{d: int(i == 5) for i, d in enumerate(days)}},
    ]
    frame = pd.DataFrame(calendar)
    frame["start_date"], frame["end_date"] = "20260101", "20261231"
    frame.to_csv(folder / "calendar.txt", index=False)
    return folder


def config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """CONFIGURATION with real paths, the synthetic clusters and *overrides*."""
    cfg = target.default_config()
    cfg.update(
        GTFS_PATH=str(tmp_path / "gtfs"),
        OUTPUT_DIR=str(tmp_path / "out"),
        CLUSTERS=json.loads(json.dumps(CLUSTERS)),
    )
    cfg.update(overrides)
    return cfg


def feed(tmp_path: Path) -> dict[str, pd.DataFrame]:
    return target.load_feed(str(write_gtfs(tmp_path / "gtfs")))


# ---------------------------------------------------------------------------
# Configuration checks
# ---------------------------------------------------------------------------


def test_default_clusters_are_valid(tmp_path: Path) -> None:
    cfg = target.default_config()
    cfg.update(GTFS_PATH=str(tmp_path), OUTPUT_DIR=str(tmp_path))
    target.validate_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"CLUSTERS": {"A": {"stops": ["S1"]}, "B": {"stops": ["S1"]}}}, "in both"),
        ({"CLUSTERS": {"A": {"stop": ["S1"]}}}, "keys from"),
        ({"CLUSTERS": {"A": {"stops": ["S1"], "two_bay_stops": ["S2"]}}}, "distinct members"),
        ({"CLUSTERS": {"A": {"stops": []}}}, "has no stops"),
        ({"CLUSTERS": {"A B": {"stops": ["S1"]}, "A_B": {"stops": ["S2"]}}}, "too similar"),
        ({"BAY_LABELS": {"S1": "X", "S2": "X"}}, "unique"),
        ({"BAY_LABELS": {"S8": "X"}}, "in no cluster"),
        ({"TIMELINE_SETTINGS": {"STOP_ID_FILTER": []}}, "may not set"),
        ({"OPTIMIZER_SETTINGS_BY_CLUSTER": {"Nowhere": {}}}, "cluster names"),
        ({"SERVICE_DATE": "20260702", "SERVICE_IDS": ["WK"]}, "not both"),
        ({"SERVICE_DATE": "2026-07-02"}, "YYYYMMDD"),
        ({"OPTIMIZE_MODE": "sometimes"}, "OPTIMIZE_MODE"),
        ({"RUN_SCHEDULES": False, "RUN_TIMELINES": False, "RUN_OPTIMIZER": False}, "RUN_"),
    ],
)
def test_validate_config_rejects(tmp_path: Path, overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(target.ConfigError, match=message):
        target.validate_config(config(tmp_path, **overrides))


# ---------------------------------------------------------------------------
# Service day and routes
# ---------------------------------------------------------------------------


def test_resolve_service_picks_representative_weekday(tmp_path: Path) -> None:
    day, ids = target.resolve_service(config(tmp_path), feed(tmp_path))
    assert ids == ["WK"]
    assert "(" in day


def test_resolve_service_uses_listed_ids_and_rejects_unused(tmp_path: Path) -> None:
    gtfs = feed(tmp_path)
    assert target.resolve_service(config(tmp_path, SERVICE_IDS=["SA"]), gtfs)[1] == ["SA"]
    with pytest.raises(target.ConfigError, match="not used"):
        target.resolve_service(config(tmp_path, SERVICE_IDS=["XX"]), gtfs)


def test_routes_serving_clusters_counts_weekday_visits(tmp_path: Path) -> None:
    table = target.routes_serving_clusters(feed(tmp_path), ["WK"], CLUSTERS)
    rows = table.to_dict("records")
    assert [(r["cluster"], r["route"], r["direction_id"], r["trips"]) for r in rows] == [
        ("North Bays", "10", "0", 1),
        ("North Bays", "10", "1", 1),
        ("North Bays", "30", "0", 1),
        ("North Bays", "Twenty Express", "0", 1),
        ("South Bay", "Twenty Express", "0", 1),
    ]
    first = rows[0]
    assert (first["first_visit"], first["last_visit"]) == ("08:00", "08:00")
    assert rows[3]["route_short_name"] == ""


# ---------------------------------------------------------------------------
# Step settings
# ---------------------------------------------------------------------------


def test_optimizer_settings_map_bays_capacity_and_overflow(tmp_path: Path) -> None:
    clusters = {
        "North Bays": {
            "stops": ["S1", "S2"],
            "two_bay_stops": ["S2"],
            "overflow_bays": ["L1", "L2"],
        },
        "South Bay": {"stops": ["S3"]},
    }
    cfg = config(
        tmp_path,
        CLUSTERS=clusters,
        BAY_LABELS={"S1": "A"},
        OPTIMIZER_SETTINGS={"SOLVER_TIME_LIMIT_SECONDS": 60, "LOCKED_ROUTES": ["10"]},
        OPTIMIZER_SETTINGS_BY_CLUSTER={"North Bays": {"LOCKED_ROUTES": ["30"]}},
    )
    settings = target.optimizer_settings(cfg, "North Bays", "feed", tmp_path, ["WK"])
    assert settings["CLUSTER_STOPS"] == {"S1": "A", "S2": "S2"}
    assert settings["CLUSTER_CAPACITY"] == {"S2": 2}
    assert settings["OVERFLOW_ROUTING"] == {target.OVERFLOW_POOL: ["LAYOVER", "LONG BREAK"]}
    assert settings["OVERFLOW_CAPACITY"] == {target.OVERFLOW_POOL: 2}
    assert settings["OTHER_CLUSTERS"] == [{"name": "South Bay", "stops": ["S3"]}]
    assert settings["SERVICE_IDS"] == ["WK"] and settings["SERVICE_DATE"] == ""
    assert settings["LOCKED_ROUTES"] == ["30"]
    assert settings["SOLVER_TIME_LIMIT_SECONDS"] == 60
    south = target.optimizer_settings(cfg, "South Bay", "feed", tmp_path, ["WK"])
    assert south["OVERFLOW_ROUTING"] == {} and south["LOCKED_ROUTES"] == ["10"]


def test_timeline_settings_define_every_cluster(tmp_path: Path) -> None:
    settings = target.timeline_settings(config(tmp_path), "feed", tmp_path, ["WK"])
    assert settings["STOP_ID_FILTER"] == ["S1", "S2", "S3"]
    assert settings["BUS_STOP_CLUSTERS_STEP1"] == [
        {"name": "North Bays", "stops": ["S1", "S2"]},
        {"name": "South Bay", "stops": ["S3"]},
    ]
    assert settings["CLUSTER_DEFINITIONS"]["South Bay"]["overflow_bays"] == []
    assert settings["CALENDAR_SERVICE_IDS"] == ["WK"] and settings["SERVICE_DATE"] == ""


# ---------------------------------------------------------------------------
# Running steps (stub scripts stand in for the real ones)
# ---------------------------------------------------------------------------

STUB_MODULE = """\
import json, sys
from pathlib import Path
OUTPUT = r"unset"
GTFS_FOLDER_PATH = BASE_OUTPUT_PATH = BLOCK_OUTPUT_FOLDER = SCENARIO_NAME = SERVICE_DATE = ""
FILTER_SERVICE_IDS = FILTER_IN_ROUTES = CALENDAR_SERVICE_IDS = STOP_ID_FILTER = []
ROUTE_SHORTNAME_FILTER = STOP_CODE_FILTER = BUS_STOP_CLUSTERS_STEP1 = []
SERVICE_LABEL_OVERRIDES = CLUSTER_DEFINITIONS = {}
FAIL = False
def main():
    folder = Path(BASE_OUTPUT_PATH or BLOCK_OUTPUT_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    values = {k: v for k, v in globals().items() if k.isupper()}
    (folder / "seen.json").write_text(json.dumps(values))
    return 3 if FAIL else 0
if __name__ == "__main__":
    raise SystemExit(main())
"""
STUB_OPTIMIZER = """\
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
folder = Path(cfg["OUTPUT_DIR"]) / cfg["SCENARIO_LABEL"]
folder.mkdir(parents=True)
(folder / "seen.json").write_text(json.dumps(cfg))
"""


def stub_scripts(root: Path) -> Path:
    for folder, name in target.STEP_SCRIPTS.values():
        path = root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(STUB_OPTIMIZER if "optimizer" in name else STUB_MODULE)
    return root


def test_run_hands_each_step_its_settings(tmp_path: Path) -> None:
    write_gtfs(tmp_path / "gtfs")
    cfg = config(
        tmp_path,
        SCRIPTS_ROOT=str(stub_scripts(tmp_path / "scripts")),
        PYTHON_EXECUTABLE=sys.executable,
    )
    assert target.run(cfg) == 0
    (run_dir,) = (tmp_path / "out").iterdir()
    schedules = json.loads((run_dir / target.SCHEDULE_DIRNAME / "seen.json").read_text())
    assert schedules["FILTER_IN_ROUTES"] == ["10", "30"]  # route 20 has no short name
    assert schedules["SERVICE_LABEL_OVERRIDES"] == {"WK": "Weekday"}
    timelines = json.loads((run_dir / target.TIMELINE_DIRNAME / "seen.json").read_text())
    assert timelines["STOP_ID_FILTER"] == ["S1", "S2", "S3"]
    optimizer = run_dir / target.OPTIMIZER_DIRNAME
    assert sorted(p.name for p in optimizer.iterdir()) == ["North_Bays", "South_Bay"]
    assert (run_dir / target.ROUTES_FILENAME).is_file()
    log = (run_dir / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "=== BEGIN CONFIG ===" not in log and "CLUSTERS" in log
    assert log.count(": succeeded") == 4


def test_run_reports_failed_and_unknown_settings(tmp_path: Path) -> None:
    write_gtfs(tmp_path / "gtfs")
    cfg = config(
        tmp_path,
        SCRIPTS_ROOT=str(stub_scripts(tmp_path / "scripts")),
        PYTHON_EXECUTABLE=sys.executable,
        RUN_OPTIMIZER=False,
        SCHEDULE_SETTINGS={"FAIL": True},
        TIMELINE_SETTINGS={"NOT_A_SETTING": 1},
    )
    assert target.run(cfg) == 1
    (run_dir,) = (tmp_path / "out").iterdir()
    log = (run_dir / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "1_schedules: FAILED (exit code 3)" in log
    assert "2_timelines: FAILED" in log
    console = (run_dir / target.SETTINGS_DIRNAME / "2_timelines.log").read_text()
    assert "Unknown CONFIGURATION settings" in console and "NOT_A_SETTING" in console


def test_run_rejects_stops_missing_from_feed(tmp_path: Path) -> None:
    write_gtfs(tmp_path / "gtfs")
    cfg = config(
        tmp_path,
        CLUSTERS={"Nowhere": {"stops": ["S404"]}},
        SCRIPTS_ROOT=str(stub_scripts(tmp_path / "scripts")),
    )
    with pytest.raises(target.ConfigError, match="absent from stops.txt"):
        target.run(cfg)
    assert not (tmp_path / "out").exists()


def test_main_stops_on_placeholder_paths() -> None:
    assert target.main([]) == 2
