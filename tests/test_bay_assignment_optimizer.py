from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import scripts.facilities_tools.bay_assignment_optimizer as target
import scripts.gtfs_exports.block_status_timeline_exporter as step1

# ---------------------------------------------------------------------------
# Synthetic feed: two blocks arriving at bay A of a two-bay facility at 08:00
# ---------------------------------------------------------------------------
# B1: T1 (route 10) ends at S1 08:00; T2 (route 10) departs S1 08:10 (gap 10, in bay).
# B2: T3 (route 20) ends at S1 08:00; T4 (route 21) departs S1 08:20 (gap 20, layover).
# Bay A therefore holds two buses at 08:00: one direct conflict minute, and three
# likely ones (08:00-08:02, from the two-minute arrival buffer).
# A visit is (stop, time) or (stop, arrival, departure).
TRIPS: list[tuple[str, str, str, list[tuple[str, ...]]]] = [
    ("T1", "10", "B1", [("S9", "07:50:00"), ("S1", "08:00:00")]),
    ("T2", "10", "B1", [("S1", "08:10:00"), ("S9", "08:20:00")]),
    ("T3", "20", "B2", [("S8", "07:45:00"), ("S1", "08:00:00")]),
    ("T4", "21", "B2", [("S1", "08:20:00"), ("S8", "08:35:00")]),
    # A block that never visits the facility is left out entirely.
    ("T5", "30", "B3", [("S8", "09:00:00"), ("S9", "09:10:00")]),
]
CLUSTER = {"S1": "A", "S2": "B"}


def write_gtfs(
    folder: Path,
    trips: list[tuple[str, str, str, list[tuple[str, ...]]]],
    directions: dict[str, str] | None = None,
    route_ids: dict[str, str] | None = None,
) -> Path:
    """Write *trips* as a feed; trips run in direction 0 and route "10" is R10 by default."""
    folder.mkdir(parents=True)
    stops = sorted(
        {"S1", "S2", "S8", "S9"} | {visit[0] for *_, visits in trips for visit in visits}
    )
    ids = {route: (route_ids or {}).get(route, f"R{route}") for _, route, _, _ in trips}
    pd.DataFrame({"stop_id": stops, "stop_name": [f"Stop {s}" for s in stops]}).to_csv(
        folder / "stops.txt", index=False
    )
    routes = sorted({route for _, route, _, _ in trips})
    pd.DataFrame(
        {
            "route_id": [ids[r] for r in routes],
            "route_short_name": routes,
            "route_long_name": "",
            "route_type": 3,
        }
    ).to_csv(folder / "routes.txt", index=False)
    pd.DataFrame(
        [
            {
                "route_id": ids[route],
                "service_id": "WKDY",
                "trip_id": trip_id,
                "direction_id": (directions or {}).get(trip_id, "0"),
                "block_id": block,
                "trip_headsign": f"Route {route} headsign",
            }
            for trip_id, route, block, _ in trips
        ]
    ).to_csv(folder / "trips.txt", index=False)
    pd.DataFrame(
        [
            {
                "trip_id": trip_id,
                "arrival_time": times[0],
                "departure_time": times[-1],
                "stop_id": stop,
                "stop_sequence": seq,
            }
            for trip_id, _, _, visits in trips
            for seq, (stop, *times) in enumerate(visits, start=1)
        ]
    ).to_csv(folder / "stop_times.txt", index=False)
    pd.DataFrame(
        {
            "service_id": ["WKDY"],
            **{
                day: [1 if day not in {"saturday", "sunday"} else 0]
                for day in (
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                )
            },
            "start_date": ["20260101"],
            "end_date": ["20261231"],
        }
    ).to_csv(folder / "calendar.txt", index=False)
    return folder


@pytest.fixture
def gtfs(tmp_path: Path) -> Path:
    return write_gtfs(tmp_path / "gtfs", TRIPS)


def make_cfg(gtfs: Path, out: Path, **overrides: Any) -> dict[str, Any]:
    cfg = target.default_config()
    cfg.update(
        GTFS_PATH=str(gtfs),
        OUTPUT_DIR=str(out),
        SCENARIO_LABEL="test",
        SERVICE_DATE="",
        SERVICE_IDS=["WKDY"],
        CLUSTER_NAME="TC",
        CLUSTER_STOPS=dict(CLUSTER),
        CLUSTER_CAPACITY={},
        OVERFLOW_ROUTING={"lay_A": ["LAYOVER"], "lay_B": ["LONG BREAK"]},
        OVERFLOW_CAPACITY={},
        OTHER_CLUSTERS=[],
        OPTIMIZE_MODE="if_conflicts",
        BAY_OPTIONS={},
        AUTO_BAY_CANDIDATES=True,
        LOCKED_ROUTES=[],
        LOCKED_ROUTE_ENDS=[],
        SAME_BAY_GROUPS=[],
        OPTIMIZER_STARTING_ASSIGNMENTS={},
        OPTIMIZER_OBJECTIVE="min_conflicts",
        WEIGHTED_SCORE_LIMIT=None,
        ROUTE_CHANGE_LIMITS=["all"],
        MAX_BAYS_PER_ROUTE=2,
        DIRECT_CONFLICT_POLICY="no_new",
        REQUIRE_CONSISTENT_BOARDING_BAYS=True,
        NAMED_PROPOSALS={},
        SOLVER_TIME_LIMIT_SECONDS=60,
    )
    cfg.update(overrides)
    return cfg


def study(cfg: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Schedule, route names, movements and rendered standards for *cfg*."""
    schedule = target.load_schedule(cfg)
    names = target.schedule_route_names(schedule.trips)
    current = target.scheduled_bays(schedule.trips, cfg["CLUSTER_STOPS"], names)
    standards = {
        name: target.Standard(name, settings, schedule, cfg, names)
        for name, settings in cfg["OCCUPANCY_STANDARDS"].items()
    }
    return schedule, names, current, standards


def sheet(workbook: Path, name: str) -> pd.DataFrame:
    return pd.read_excel(workbook, sheet_name=name)


# ---------------------------------------------------------------------------
# Schedule and occupancy
# ---------------------------------------------------------------------------


def test_load_schedule_keeps_whole_blocks_that_visit_the_facility(
    gtfs: Path, tmp_path: Path
) -> None:
    schedule = target.load_schedule(make_cfg(gtfs, tmp_path / "out"))
    assert sorted(trip["trip_id"] for trip in schedule.trips) == ["T1", "T2", "T3", "T4"]
    assert schedule.service_ids == ["WKDY"]


def test_service_date_selects_calendar_service(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(gtfs, tmp_path / "out", SERVICE_DATE="20260915", SERVICE_IDS=[])
    assert target.load_schedule(cfg).service_ids == ["WKDY"]
    with pytest.raises(ValueError, match="No service_id is active"):
        target.load_schedule({**cfg, "SERVICE_DATE": "20260919"})  # a Saturday


def test_no_trip_at_facility_fails_clearly(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(gtfs, tmp_path / "out", CLUSTER_STOPS={"S2": "B"})
    with pytest.raises(ValueError, match="No trip in the selected service stops"):
        target.load_schedule(cfg)


@pytest.mark.parametrize("standard", ["Direct conflict", "Likely conflict"])
def test_occupancy_matches_step1_export(
    gtfs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, standard: str
) -> None:
    """The in-memory render equals what Step 1 writes, row for row, at the facility."""
    cfg = make_cfg(gtfs, tmp_path / "out")
    _, _, _, standards = study(cfg)
    out = tmp_path / "step1"
    settings = {
        "GTFS_FOLDER_PATH": str(gtfs),
        "BLOCK_OUTPUT_FOLDER": str(out),
        "SCENARIO_NAME": "",
        "CALENDAR_SERVICE_IDS": ["WKDY"],
        "SERVICE_DATE": "",
        "ROUTE_SHORTNAME_FILTER": [],
        "STOP_ID_FILTER": list(CLUSTER),
        "STOP_CODE_FILTER": [],
        "BAY_OVERRIDES": [],
        "WRITE_PER_BLOCK_FILES": False,
        "BUS_STOP_CLUSTERS_STEP1": [{"name": "TC", "stops": list(CLUSTER)}],
        "INTERPOLATE_UNTIMED_STOPS": False,
        "TRIP_ONLY_WITHOUT_BLOCK_ID": False,
        **cfg["OCCUPANCY_STANDARDS"][standard],
    }
    for name, value in settings.items():
        monkeypatch.setattr(step1, name, value)
    step1.run_step1_gtfs_to_blocks()
    exported = pd.read_csv(out / "all_blocks_timeline.csv", dtype=str, keep_default_na=False)
    exported = exported[exported["Stop ID"].isin(CLUSTER)]
    exported = exported[exported["Status"].isin(target.BAY_STATUSES | target.OVERFLOW_STATUSES)]
    expected = Counter(
        (
            row["Block"],
            target.timestamp_to_minutes(row["Timestamp"]),
            CLUSTER[row["Stop ID"]],
            row["Status"],
        )
        for _, row in exported.iterrows()
    )
    rows = standards[standard].rows
    actual = Counter(zip(rows["Block"], rows["Minute"].astype(int), rows["Bay"], rows["Status"]))
    assert actual == expected


def test_conflict_summary_counts_each_standard(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(gtfs, tmp_path / "out")
    schedule, _, _, standards = study(cfg)
    direct = target.conflict_summary(standards["Direct conflict"], cfg, schedule)
    likely = target.conflict_summary(standards["Likely conflict"], cfg, schedule)
    assert (direct["bay_conflict_minutes"], likely["bay_conflict_minutes"]) == (1, 3)
    assert likely["Bay A conflict minutes"] == 3 and likely["Bay B conflict minutes"] == 0
    assert likely["peak_buses_laying_over"] == 1
    assert likely["layover_space_conflict_minutes"] == 0
    assert likely["facility_capacity"] == 4  # two bays plus two layover spaces


def test_bay_change_rebuild_matches_relabelled_model(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(gtfs, tmp_path / "out")
    _, names, current, standards = study(cfg)
    move = target.Change({names.parse("20 arrive"): "B"})
    evaluation = target.evaluate_plan(cfg, standards, current, move)
    assert (len(evaluation["direct"]), len(evaluation["likely"])) == (0, 0)
    assert evaluation["changed_routes"] == evaluation["changed_movements"] == 1
    # The layover after T3 now hangs off bay B; layover rows are otherwise unchanged.
    rows = standards["Likely conflict"].apply(move)
    assert set(rows.loc[rows["Status"].eq("LAYOVER"), "Bay"]) == {"B"}


# ---------------------------------------------------------------------------
# Permissions and rules
# ---------------------------------------------------------------------------


def test_permissions_honor_options_and_locks(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(
        gtfs,
        tmp_path / "out",
        AUTO_BAY_CANDIDATES=False,
        BAY_OPTIONS={"10": ["A", "B"], "10 arrive": ["B"]},
        LOCKED_ROUTE_ENDS=["10 depart"],
    )
    _, names, current, _ = study(cfg)
    allowed, groups = target.resolve_bay_permissions(cfg, current, names)
    by_name = {str(end): bays for end, bays in allowed.items()}
    assert by_name == {
        "10 arrive": {"B"},
        "10 depart": set(),
        "20 arrive": set(),
        "21 depart": set(),
    }
    assert groups == []


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("BAY_OPTIONS", {"99 arrive": ["A"]}),
        ("LOCKED_ROUTES", ["10 arrive"]),
        ("LOCKED_ROUTE_ENDS", ["10"]),
        ("SAME_BAY_GROUPS", [["10 arrive", "10 through 0"]]),
    ],
)
def test_unknown_or_wrong_selectors_fail(
    gtfs: Path, tmp_path: Path, setting: str, value: Any
) -> None:
    cfg = make_cfg(gtfs, tmp_path / "out", **{setting: value})
    _, names, current, _ = study(cfg)
    with pytest.raises(ValueError):
        target.resolve_bay_permissions(cfg, current, names)


def test_plan_violations_names_each_rule(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(gtfs, tmp_path / "out", MAX_BAYS_PER_ROUTE=1, WEIGHTED_SCORE_LIMIT=0)
    _, names, current, standards = study(cfg)
    baseline = target.evaluate_plan(cfg, standards, current, target.Change())
    change = target.Change({names.parse("10 arrive"): "B"})
    evaluation = target.evaluate_plan(cfg, standards, current, change)
    group = [[names.parse("10 arrive"), names.parse("10 depart")]]
    problems = target.plan_violations(cfg, current, group, change, evaluation, baseline, limit=0)
    assert any("budget is 0" in p for p in problems)
    assert any("more than 1 bays: 10" in p for p in problems)
    assert any("is split" in p for p in problems)
    assert target.plan_violations(cfg, current, [], target.Change(), baseline, baseline) == [
        f"weighted score {baseline['weighted_score']} exceeds WEIGHTED_SCORE_LIMIT 0"
    ]


def test_new_direct_conflicts_are_reported(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(gtfs, tmp_path / "out")
    _, names, current, standards = study(cfg)
    baseline = target.evaluate_plan(cfg, standards, current, target.Change())
    # Sending both arrivals to B moves the 08:00 conflict to a bay-minute that was clear.
    change = target.Change({names.parse("10 arrive"): "B", names.parse("20 arrive"): "B"})
    evaluation = target.evaluate_plan(cfg, standards, current, change)
    problems = target.plan_violations(cfg, current, [], change, evaluation, baseline)
    assert problems == ["direct conflicts at 1 bay-minute(s) clear in the baseline"]


@pytest.mark.parametrize(
    ("windows", "peak", "off_peak"),
    [([["06:00", "10:00"]], [360, 599], [359, 600]), ([["22:00", "02:00"]], [1320, 0, 119], [120])],
)
def test_peak_mask_half_open_and_overnight(windows: list, peak: list, off_peak: list) -> None:
    mask = target.peak_mask(windows)
    assert all(mask[m] for m in peak) and not any(mask[m] for m in off_peak)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"SERVICE_DATE": "20260915"}, "exactly one of SERVICE_DATE"),
        ({"CLUSTER_STOPS": {"S1": "A", "S2": "A"}}, "one stop_id per physical bay"),
        ({"CLUSTER_CAPACITY": {"A": 4}}, "CLUSTER_CAPACITY"),
        ({"OVERFLOW_ROUTING": {"lay": ["LAYOVER"]}}, "both LAYOVER and LONG BREAK"),
        ({"OPTIMIZER_OBJECTIVE": "min_changes"}, "requires WEIGHTED_SCORE_LIMIT"),
        ({"OPTIMIZER_DIRECT_STANDARD": "Likely conflict"}, "distinct names"),
        ({"PEAK_WINDOWS": [["06:00", "06:00"]]}, "distinct boundaries"),
        ({"ROUTE_CHANGE_LIMITS": [-1]}, "ROUTE_CHANGE_LIMITS"),
        ({"OPTIMIZER_STARTING_ASSIGNMENTS": {"10 arrive": "Z"}}, "STARTING_ASSIGNMENTS"),
        ({"NOT_A_SETTING": 1}, "Unknown settings"),
        ({"REQUIRE_CONSISTENT_BOARDING_BAYS": "yes"}, "True or False"),
        ({"NAMED_PROPOSALS": {"Baseline": {"10 arrive": "B"}}}, "Invalid proposal name"),
        ({"NAMED_PROPOSALS": {"Up to 3 route(s)": {"10 arrive": "B"}}}, "Invalid proposal"),
        ({"NAMED_PROPOSALS": {"Mine": {"10 arrive": "Z"}}}, "must map one or more"),
        ({"NAMED_PROPOSALS": {"Mine": {}}}, "must map one or more"),
    ],
)
def test_validate_configuration_rejects(
    gtfs: Path, tmp_path: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        target.validate_configuration(make_cfg(gtfs, tmp_path / "out", **overrides))


# ---------------------------------------------------------------------------
# Whole runs
# ---------------------------------------------------------------------------


def test_check_only_run_writes_workbook_and_run_log(gtfs: Path, tmp_path: Path) -> None:
    result = target.run(make_cfg(gtfs, tmp_path / "out", OPTIMIZE_MODE="never"))
    assert result["status"] == "checked"
    workbook = Path(result["workbook"])
    names = openpyxl.load_workbook(workbook, read_only=True).sheetnames
    assert names[:7] == [
        "Read me",
        "Conflict summary",
        "Plan comparison",
        "Assignments",
        "Route bay usage",
        "Boarding bays",
        "Conflict minutes",
    ]
    assert {"Route-ends", "Interlines", "Block chains", "Bay options", "Configuration"} <= set(
        names
    )
    plans = sheet(workbook, "Plan comparison")
    assert list(plans["plan"]) == ["Baseline"]
    # Direct 08:00 at peak (10) plus likely-only 08:01 and 08:02 at peak (2 each).
    assert plans.loc[0, "weighted_score"] == 14
    route_ends = sheet(workbook, "Route-ends")
    assert set(route_ends["route_end"]) == {"10 arrive", "10 depart", "20 arrive", "21 depart"}
    log = workbook.with_name(f"{workbook.stem}_runlog.txt").read_text(encoding="utf-8")
    assert "BAY ASSIGNMENT OPTIMIZER RUN LOG" in log and '"OPTIMIZE_MODE": "never"' in log
    assert "# === BEGIN CONFIG ===" not in log and "OPTIMIZE_MODE =" in log


def test_conflict_free_baseline_is_not_optimized(tmp_path: Path) -> None:
    trips = [trip for trip in TRIPS if trip[0] not in {"T3", "T4"}]
    gtfs = write_gtfs(tmp_path / "gtfs", trips)
    result = target.run(make_cfg(gtfs, tmp_path / "out"))
    assert (result["status"], result["reason"]) == (
        "checked",
        "No bay conflicts under either standard.",
    )


def test_optimizer_removes_conflicts_with_one_move(gtfs: Path, tmp_path: Path) -> None:
    pytest.importorskip("pulp")
    result = target.run(make_cfg(gtfs, tmp_path / "out"))
    assert result["status"] == "optimized"
    plans = sheet(Path(result["workbook"]), "Plan comparison").set_index("plan")
    best = plans.loc["Up to 3 route(s)"]
    assert (best["weighted_score"], best["changed_routes"], best["changed_movements"]) == (0, 1, 1)
    assert best["validation"] == "passed" and bool(best["proven_optimal"])
    assert best["changes"] in {"10 arrive to Bay B", "20 arrive to Bay B"}
    assert (Path(result["workbook"]).parent / "solver" / "cbc_max_3_routes.log").is_file()


def test_route_bay_limit_steers_the_choice(gtfs: Path, tmp_path: Path) -> None:
    pytest.importorskip("pulp")
    # Route 10 already uses bay A for its departures, so moving its arrivals to B would
    # give it two bays; route 20 only arrives here.
    result = target.run(make_cfg(gtfs, tmp_path / "out", MAX_BAYS_PER_ROUTE=1))
    plans = sheet(Path(result["workbook"]), "Plan comparison").set_index("plan")
    assert plans.loc["Up to 3 route(s)", "changes"] == "20 arrive to Bay B"
    usage = sheet(Path(result["workbook"]), "Route bay usage")
    assert usage["bay_limit_passed"].all()


def test_min_changes_with_score_limit_and_budgets(gtfs: Path, tmp_path: Path) -> None:
    pytest.importorskip("pulp")
    cfg = make_cfg(
        gtfs,
        tmp_path / "out",
        OPTIMIZER_OBJECTIVE="min_changes",
        WEIGHTED_SCORE_LIMIT=0,
        ROUTE_CHANGE_LIMITS=[0, "all"],
    )
    result = target.run(cfg)
    plans = sheet(Path(result["workbook"]), "Plan comparison").set_index("plan")
    assert plans.loc["Up to 0 route(s)", "plan_status"] == "infeasible"
    assert plans.loc["Up to 3 route(s)", "weighted_score"] == 0
    assert plans.loc["Up to 3 route(s)", "changed_movements"] == 1
    assert result["status"] == "optimized"


def test_everything_locked_returns_the_baseline(gtfs: Path, tmp_path: Path) -> None:
    pytest.importorskip("pulp")
    cfg = make_cfg(gtfs, tmp_path / "out", LOCKED_ROUTES=["10", "20", "21"])
    result = target.run(cfg)
    plans = sheet(Path(result["workbook"]), "Plan comparison").set_index("plan")
    assert plans.loc["Up to 3 route(s)", "solver_stop"].startswith("Only one permitted")
    assert plans.loc["Up to 3 route(s)", "weighted_score"] == 14


def test_starting_assignment_is_validated(gtfs: Path, tmp_path: Path) -> None:
    pytest.importorskip("pulp")
    ok = make_cfg(gtfs, tmp_path / "ok", OPTIMIZER_STARTING_ASSIGNMENTS={"20 arrive": "B"})
    plans = sheet(Path(target.run(ok)["workbook"]), "Plan comparison").set_index("plan")
    assert plans.loc["Starting plan", "weighted_score"] == 0
    bad = make_cfg(
        gtfs,
        tmp_path / "bad",
        OPTIMIZER_STARTING_ASSIGNMENTS={"10 arrive": "B", "20 arrive": "B"},
    )
    with pytest.raises(ValueError, match="clear in the baseline"):
        target.run(bad)
    workbooks = list((tmp_path / "bad").glob("*/*_bay_plans.xlsx"))
    assert sheet(workbooks[0], "Read me").set_index("Item").loc["Status", "Value"] == "failed"


def test_missing_pulp_keeps_checked_results(
    gtfs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> Any:
        raise target.OptimizerUnavailable("PuLP is missing.")

    monkeypatch.setattr(target, "load_pulp", unavailable)
    result = target.run(make_cfg(gtfs, tmp_path / "out"))
    assert result["status"] == "optimizer_unavailable"
    assert list(sheet(Path(result["workbook"]), "Plan comparison")["plan"]) == ["Baseline"]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_placeholder_paths_return_2() -> None:
    assert target.main([]) == 2


def test_main_flags_and_json_overrides(gtfs: Path, tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "CLUSTER_NAME": "TC",
                "CLUSTER_STOPS": CLUSTER,
                "CLUSTER_CAPACITY": {},
                "OVERFLOW_ROUTING": {},
                "SERVICE_IDS": ["NOT_USED"],
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "--gtfs-path",
        str(gtfs),
        "--output-dir",
        str(tmp_path / "out"),
        "--service-date",
        "20260915",
        "--optimize",
        "never",
        "--config",
        str(overrides),
    ]
    assert target.main(argv) == 0
    log = next((tmp_path / "out").glob("*/*_runlog.txt")).read_text(encoding="utf-8")
    # A date given on the command line replaces the service_ids from the JSON file.
    assert '"SERVICE_DATE": "20260915"' in log and '"SERVICE_IDS": []' in log


# ---------------------------------------------------------------------------
# Consistent boarding bays
# ---------------------------------------------------------------------------
# Route 10 departs bay A at 08:00 (T1) and passes through bay A at 08:40 (T6), both in
# direction 0; route 20 departs bay A at 08:00 too (T3). With route 20 locked, freeing
# bay A at 08:00 means moving route 10's departures, and boarding then splits unless its
# through visits follow.
BOARDING_TRIPS: list[tuple[str, str, str, list[tuple[str, ...]]]] = [
    ("T1", "10", "B1", [("S1", "08:00:00"), ("S9", "08:10:00")]),
    ("T3", "20", "B2", [("S1", "08:00:00"), ("S8", "08:15:00")]),
    ("T6", "10", "B3", [("S9", "08:30:00"), ("S1", "08:40:00"), ("S8", "08:50:00")]),
]


@pytest.fixture
def boarding_gtfs(tmp_path: Path) -> Path:
    return write_gtfs(tmp_path / "boarding_gtfs", BOARDING_TRIPS)


def test_boarding_inventory_splits_departures_by_direction(tmp_path: Path) -> None:
    trips = [
        ("T1", "10", "B1", [("S1", "08:00:00"), ("S9", "08:10:00")]),
        ("T2", "10", "B2", [("S2", "09:00:00"), ("S8", "09:10:00")]),
        ("T3", "10", "B3", [("S8", "07:00:00"), ("S1", "07:10:00")]),
    ]
    gtfs = write_gtfs(tmp_path / "gtfs", trips, directions={"T2": "1"})
    cfg = make_cfg(gtfs, tmp_path / "out")
    _, names, current, standards = study(cfg)
    inventory = standards["Likely conflict"].boarding_inventory
    # Arrivals set down only; the departure movement boards at A outbound and B inbound.
    assert {str(end): bays for end, bays in inventory.items()} == {
        "10 depart": {"0": {"A"}, "1": {"B"}}
    }
    baseline = target.evaluate_plan(cfg, standards, current, target.Change())
    assert baseline["boarding_bays"] == {("R10", "0"): {"A"}, ("R10", "1"): {"B"}}
    assert target.plan_violations(cfg, current, [], target.Change(), baseline, baseline) == []
    moved = target.Change({names.parse("10 depart"): "A"})
    evaluation = target.evaluate_plan(cfg, standards, current, moved)
    assert evaluation["boarding_bays"] == {("R10", "0"): {"A"}, ("R10", "1"): {"A"}}


def test_split_boarding_is_a_rule_break(boarding_gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(boarding_gtfs, tmp_path / "out")
    _, names, current, standards = study(cfg)
    baseline = target.evaluate_plan(cfg, standards, current, target.Change())
    change = target.Change({names.parse("10 depart"): "B"})
    evaluation = target.evaluate_plan(cfg, standards, current, change)
    assert evaluation["boarding_bays"][("R10", "0")] == {"A", "B"}
    problems = target.plan_violations(cfg, current, [], change, evaluation, baseline)
    assert problems == ["passengers board at more than one bay: 10 direction 0 (A, B)"]
    off = {**cfg, "REQUIRE_CONSISTENT_BOARDING_BAYS": False}
    assert target.plan_violations(off, current, [], change, evaluation, baseline) == []


def test_rebuilt_boarding_bays_must_match_the_model(
    boarding_gtfs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(boarding_gtfs, tmp_path / "out")
    _, _, current, standards = study(cfg)
    monkeypatch.setattr(target.Standard, "rebuilt_boarding_bays", lambda self, change: {})
    with pytest.raises(ValueError, match="Rebuilt boarding bays differ"):
        target.evaluate_plan(cfg, standards, current, target.Change())


@pytest.mark.parametrize(
    ("required", "expected"),
    [(True, "10 depart to Bay B; 10 through 0 to Bay B"), (False, "10 depart to Bay B")],
)
def test_optimizer_keeps_boarding_at_one_bay(
    boarding_gtfs: Path, tmp_path: Path, required: bool, expected: str
) -> None:
    pytest.importorskip("pulp")
    cfg = make_cfg(
        boarding_gtfs,
        tmp_path / "out",
        LOCKED_ROUTES=["20"],
        REQUIRE_CONSISTENT_BOARDING_BAYS=required,
    )
    result = target.run(cfg)
    assert result["status"] == "optimized"
    workbook = Path(result["workbook"])
    plans = sheet(workbook, "Plan comparison").set_index("plan")
    best = plans.loc["Up to 2 route(s)"]
    assert (best["changes"], best["weighted_score"]) == (expected, 0)
    assert best["max_boarding_bays_per_direction"] == (1 if required else 2)
    boarding = sheet(workbook, "Boarding bays").set_index(["plan", "route_id"])
    row = boarding.loc[("Up to 2 route(s)", "R10")]
    assert (bool(row["consistent"]), bool(row["rule_required"])) == (required, required)
    assert row["boarding_movements"] == "10 depart; 10 through 0"


# ---------------------------------------------------------------------------
# GTFS text identifiers, command-line precedence, named proposals
# ---------------------------------------------------------------------------


def test_text_identifiers_are_not_read_as_missing(tmp_path: Path) -> None:
    trips = [
        ("T1", "NA", "null", [("S9", "07:50:00"), ("None", "08:00:00")]),
        ("T2", "NA", "null", [("None", "08:10:00"), ("S9", "08:20:00")]),
    ]
    gtfs = write_gtfs(tmp_path / "gtfs", trips, route_ids={"NA": "NA"})
    cfg = make_cfg(gtfs, tmp_path / "out", CLUSTER_STOPS={"None": "A", "S2": "B"})
    schedule, names, current, _ = study(cfg)
    assert {trip["route_id"] for trip in schedule.trips} == {"NA"}
    assert {trip["block"] for trip in schedule.trips} == {"null"}
    assert sorted(map(str, current)) == ["NA arrive", "NA depart"]


def test_flag_equal_to_default_still_overrides_json(gtfs: Path, tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "CLUSTER_NAME": "TC",
                "CLUSTER_STOPS": CLUSTER,
                "CLUSTER_CAPACITY": {},
                "OVERFLOW_ROUTING": {},
                "SERVICE_IDS": ["WKDY"],
                "SOLVER_TIME_LIMIT_SECONDS": 5,
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "--gtfs-path",
        str(gtfs),
        "--output-dir",
        str(tmp_path / "out"),
        "--optimize",
        "never",
        "--time-limit",
        str(target.SOLVER_TIME_LIMIT_SECONDS),  # the CONFIGURATION default
        "--config",
        str(overrides),
    ]
    assert target.main(argv) == 0
    log = next((tmp_path / "out").glob("*/*_runlog.txt")).read_text(encoding="utf-8")
    assert f'"SOLVER_TIME_LIMIT_SECONDS": {target.SOLVER_TIME_LIMIT_SECONDS}' in log


def test_named_proposals_are_scored_and_flagged(gtfs: Path, tmp_path: Path) -> None:
    pytest.importorskip("pulp")
    proposals = {
        "Consultant": {"20 arrive": "B"},  # route 20 is locked below
        "Internal": {"10 arrive": "B", "20 arrive": "B"},  # moves the conflict to bay B
        "Tidy": {"10 arrive": "B", "21 depart": "A"},  # "21 depart" already uses A: KEEP
    }
    cfg = make_cfg(gtfs, tmp_path / "out", LOCKED_ROUTES=["20"], NAMED_PROPOSALS=proposals)
    result = target.run(cfg)
    assert result["status"] == "optimized"
    workbook = Path(result["workbook"])
    plans = sheet(workbook, "Plan comparison").set_index("plan")
    assert list(plans.index) == ["Baseline", "Consultant", "Internal", "Tidy", "Up to 3 route(s)"]
    consultant, internal, tidy = (plans.loc[name] for name in ("Consultant", "Internal", "Tidy"))
    assert consultant["plan_status"] == "proposal" and consultant["weighted_score"] == 0
    assert not consultant["policy_compliant"]
    assert consultant["policy_issues"] == "moves not permitted by BAY_OPTIONS or locks: 20 arrive"
    assert internal["policy_issues"].endswith(
        "direct conflicts at 1 bay-minute(s) clear in the baseline"
    )
    assert bool(tidy["policy_compliant"]) and tidy["changes"] == "10 arrive to Bay B"
    assignments = sheet(workbook, "Assignments")
    assert set(assignments["plan"]) == {
        "Baseline",
        "Consultant",
        "Internal",
        "Tidy",
        "Up to 3 route(s)",
    }


def test_named_proposal_with_unknown_movement_fails(gtfs: Path, tmp_path: Path) -> None:
    cfg = make_cfg(
        gtfs, tmp_path / "out", OPTIMIZE_MODE="never", NAMED_PROPOSALS={"X": {"99 arrive": "B"}}
    )
    with pytest.raises(ValueError, match=r"NAMED_PROPOSALS\['X'\]: unknown or duplicate"):
        target.run(cfg)


# ---------------------------------------------------------------------------
# Boarding directions must be known when the boarding rule is on
# ---------------------------------------------------------------------------
# Route 10 departs bay A (T1) and passes through bay B (T6); T7 ends at bay A.
DIRECTION_TRIPS: list[tuple[str, str, str, list[tuple[str, ...]]]] = [
    ("T1", "10", "B1", [("S1", "08:00:00"), ("S9", "08:10:00")]),
    ("T6", "10", "B3", [("S9", "08:30:00"), ("S2", "08:40:00"), ("S8", "08:50:00")]),
    ("T7", "10", "B4", [("S8", "09:00:00"), ("S1", "09:10:00")]),
]


@pytest.mark.parametrize("value", ["", "2", "north"])
def test_unknown_boarding_direction_stops_the_run(tmp_path: Path, value: str) -> None:
    gtfs = write_gtfs(tmp_path / "gtfs", DIRECTION_TRIPS, directions={"T1": value})
    cfg = make_cfg(gtfs, tmp_path / "out", OPTIMIZE_MODE="never")
    label = re.escape(value or "(blank)")
    with pytest.raises(ValueError, match=rf"Unknown directions: 10 direction {label}: 1 trip"):
        target.run(cfg)


def test_unknown_direction_is_reported_when_the_rule_is_off(tmp_path: Path) -> None:
    gtfs = write_gtfs(tmp_path / "gtfs", DIRECTION_TRIPS, directions={"T1": ""})
    cfg = make_cfg(
        gtfs, tmp_path / "out", OPTIMIZE_MODE="never", REQUIRE_CONSISTENT_BOARDING_BAYS=False
    )
    workbook = Path(target.run(cfg)["workbook"])
    boarding = sheet(workbook, "Boarding bays").fillna({"direction": ""})
    rows = boarding.set_index("direction")
    assert (rows.loc["", "final_boarding_bays"], rows.loc[0, "final_boarding_bays"]) == ("A", "B")
    assert not boarding["rule_required"].any()


def test_arrivals_need_no_direction_and_directions_are_normalized(tmp_path: Path) -> None:
    # T7 only terminates here, so its blank direction is exempt; " 1 " and "1.0" mean 1.
    directions = {"T1": " 1 ", "T6": "1.0", "T7": ""}
    gtfs = write_gtfs(tmp_path / "gtfs", DIRECTION_TRIPS, directions=directions)
    cfg = make_cfg(gtfs, tmp_path / "out")
    schedule, names, current, standards = study(cfg)
    assert {trip["trip_id"]: trip["direction_id"] for trip in schedule.trips} == {
        "T1": "1",
        "T6": "1",
        "T7": "",
    }
    assert sorted(map(str, current)) == ["10 arrive", "10 depart", "10 through 1"]
    baseline = target.evaluate_plan(cfg, standards, current, target.Change())
    assert baseline["boarding_bays"] == {("R10", "1"): {"A", "B"}}
    assert target.plan_violations(cfg, current, [], target.Change(), baseline, baseline) == [
        "passengers board at more than one bay: 10 direction 1 (A, B)"
    ]
