"""Tests for trip_cut_savings: block arithmetic, ridership matching, sessions, TODS, outputs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.ridership_tools import trip_cut_savings as tcs
from scripts.ridership_tools.trip_cut_savings import (
    BlockRules,
    Config,
    RidershipSpec,
    block_metrics,
    build_blocks,
    build_route_dir_index,
    build_trip_lookup,
    cycle_time_table,
    enumerate_runs,
    load_ridership,
    main,
    parse_periods,
    run,
    service_effects,
)

MOCK_FEED = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"
MOCK_TODS = Path(__file__).parent / "fixtures" / "mock_tods_dc.zip"
RULES = BlockRules(min_release_gap_min=90, pull_deadhead_min=30)

# Four 60-minute trips with 10-minute layovers: 0-60, 70-130, 140-200, 210-270.
STARTS = [0, 70, 140, 210]
ENDS = [60, 130, 200, 270]

# A narrower window search than the shipped default keeps each analysis run short.
FAST_SEARCH = {
    "start": "06:00",
    "end": "22:00",
    "durations_min": [240],
    "step_min": 120,
    "headways": [[15, 30], [30, 60]],
    "routes": [],
    "days": [],
}
CSV_SPEC = RidershipSpec(
    match_mode="trip_id",
    boardings_col="avg_daily_boardings",
    direction_col="",
    block_col="",
    obs_col="",
)


def make_trip_table(rows: list[tuple[str, str, str, str, int, int]]) -> pd.DataFrame:
    """Build a trip table shaped like build_trip_table's output.

    Each row is (trip_id, route_id, direction_id, block_id, start_min, end_min).
    """
    table = pd.DataFrame(
        rows, columns=["trip_id", "route_id", "direction_id", "block_id", "start_min", "end_min"]
    ).set_index("trip_id")
    table["runtime_min"] = table["end_min"] - table["start_min"]
    table["route_label"] = table["route_id"]
    return table


def mock_weekday_trips() -> pd.DataFrame:
    """The mock feed's weekday trips with route names and first departures."""
    with zipfile.ZipFile(MOCK_FEED) as archive:
        trips = pd.read_csv(archive.open("dc/trips.txt"), dtype=str)
        routes = pd.read_csv(archive.open("dc/routes.txt"), dtype=str)
        stop_times = pd.read_csv(archive.open("dc/stop_times.txt"), dtype=str)
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)
    first = stop_times.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id")
    weekday = trips[trips["service_id"] == "weekday"].merge(
        routes[["route_id", "route_short_name"]], on="route_id"
    )
    weekday["first_departure"] = weekday["trip_id"].map(first["departure_time"].first())
    return weekday.reset_index(drop=True)


def fast_config(tmp: Path, ridership_file: Path, **overrides: object) -> Config:
    """A weekday-only Config pointed at the mock feed with a narrow session search."""
    settings: dict[str, object] = {
        "gtfs_path": str(MOCK_FEED),
        "ridership_paths": {"weekday": str(ridership_file)},
        "output_dir": tmp,
        "ridership": CSV_SPEC,
        "headway_search": FAST_SEARCH,
        "workbook_limit_per_type": 2,
    }
    settings.update(overrides)
    return Config(**settings)


# ---------------------------------------------------------------------------
# Block arithmetic
# ---------------------------------------------------------------------------


def test_block_metrics_counts_layover_and_one_deadhead_per_piece() -> None:
    assert block_metrics(STARTS, ENDS, RULES) == {
        "trips": 4,
        "pieces": 1,
        "revenue_min": 240,
        "layover_min": 30,
        "deadhead_min": 30,
        "vehicle_min": 300,
    }


def test_block_metrics_treats_long_gap_as_release() -> None:
    # 0-60 then 200-260: a 140-minute gap pulls the vehicle in (two pieces).
    metrics = block_metrics([0, 200], [60, 260], RULES)
    assert metrics["pieces"] == 2
    assert metrics["layover_min"] == 0
    assert metrics["vehicle_min"] == 120 + 2 * 30


def test_enumerate_runs_classifies_and_scores_each_cut() -> None:
    runs = {(r.i, r.j): r for r in enumerate_runs(STARTS, ENDS, RULES)}

    assert runs[(0, 3)].cut_type == "whole_block"
    assert runs[(0, 3)].vehicle_min_saved == 300
    assert runs[(0, 0)].cut_type == "block_start_trim"
    assert runs[(0, 0)].vehicle_min_saved == 70  # 60 running + 10 layover
    assert runs[(3, 3)].cut_type == "block_end_trim"
    assert runs[(3, 3)].vehicle_min_saved == 70

    # One mid-block trip leaves an 80-minute gap: below release, so pure layover.
    solo = runs[(1, 1)]
    assert solo.cut_type == "mid_block_layover_only"
    assert solo.vehicle_min_saved == 0
    assert solo.gap_left_min == 80

    # Two mid-block trips leave 150 minutes: the vehicle is released.
    pair = runs[(1, 2)]
    assert pair.cut_type == "mid_block_release"
    assert pair.vehicle_min_saved == 120 + 30 - 30


def test_enumerate_runs_matches_recount_for_every_run() -> None:
    starts = [0, 70, 300, 370, 440]  # a released gap already sits between trips 1 and 2
    ends = [60, 130, 360, 430, 500]
    total = block_metrics(starts, ends, RULES)["vehicle_min"]
    for cut in enumerate_runs(starts, ends, RULES):
        keep = [k for k in range(len(starts)) if not cut.i <= k <= cut.j]
        after = block_metrics([starts[k] for k in keep], [ends[k] for k in keep], RULES)
        assert cut.vehicle_min_saved == total - after["vehicle_min"], cut


def test_enumerate_runs_skips_runs_through_ineligible_trips() -> None:
    runs = enumerate_runs(STARTS, ENDS, RULES, eligible=[True, False, True, True])
    spans = {(r.i, r.j) for r in runs}
    assert (0, 0) in spans and (2, 3) in spans
    assert not any(i <= 1 <= j for i, j in spans)


def test_service_effects_reports_new_gap_and_span() -> None:
    table = make_trip_table([(f"t{k}", "R", "0", "X", 30 * k, 30 * k + 20) for k in range(4)])
    lookup, index = build_trip_lookup(table), build_route_dir_index(table)
    assert service_effects(["t1"], lookup, index) == (False, 60)
    assert service_effects(["t1", "t2"], lookup, index) == (False, 90)
    assert service_effects(["t0"], lookup, index) == (True, None)


def test_cycle_time_table_headway_spans_midnight_window() -> None:
    # A 22:00-06:00 window holding 22:00, 22:30, 23:00 and 05:00, 05:30 departures:
    # the median gap is 30 minutes, not the overnight hole.
    starts = [1320, 1350, 1380, 300, 330]
    table = make_trip_table([(f"t{k}", "R", "0", f"B{k}", s, s + 20) for k, s in enumerate(starts)])
    periods = parse_periods({"NIGHT": ("22:00", "06:00")})
    out = cycle_time_table(table, build_blocks(table), RULES, periods, 15.0)
    assert out["headway_min"].tolist() == [30.0]


def test_cycle_time_table_reports_recovery_and_freeable_vehicles() -> None:
    # One route, 30-minute trips each way, 20-minute layovers: cycle 100, running 60.
    rows = []
    for v in range(4):  # four vehicles, 25-minute headway
        for leg in range(4):
            start = 25 * v + 50 * leg
            rows.append((f"v{v}l{leg}", "R", str(leg % 2), f"B{v}", start, start + 30))
    table = make_trip_table(rows)
    row = cycle_time_table(table, build_blocks(table), RULES, [], 15.0).iloc[0]
    assert row["cycle_min"] == 100.0
    assert row["round_trip_running_min"] == 60.0
    assert row["vehicles_at_cycle"] == 4
    assert row["vehicles_at_target_recovery"] == 3  # ceil(69 / 25)
    assert row["vehicles_freeable_by_runtime_trim"] == 1


# ---------------------------------------------------------------------------
# Ridership intake
# ---------------------------------------------------------------------------


def test_load_ridership_matches_route_and_time_and_averages_days(tmp_path: Path) -> None:
    table = make_trip_table(
        [("t1", "r10", "0", "B1", 443, 500), ("t2", "r10", "0", "B1", 503, 560)]
    )
    routes = pd.DataFrame({"route_id": ["r10"], "route_short_name": ["10"]})
    path = tmp_path / "ridership.csv"
    pd.DataFrame(
        {
            "ROUTE_NUMBER": ["10", "10", "10", "99"],
            "TRIP_START_TIME": ["07:23", "7:23:00 AM", "08:23", "08:23"],
            "PASSENGERS_ON": ["10", "20", "7", "100"],
            "SURVEY_DATE": ["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-02"],
        }
    ).to_csv(path, index=False)
    spec = RidershipSpec(
        match_mode="route_start_time",
        route_col="ROUTE_NUMBER",
        boardings_col="PASSENGERS_ON",
        date_col="SURVEY_DATE",
        direction_col="",
        block_col="",
        obs_col="",
    )
    result = load_ridership(str(path), spec, table, routes)
    assert result.boardings.to_dict() == {"t1": 15.0, "t2": 7.0}
    assert result.rows_in == 4


def test_load_ridership_raises_when_nothing_matches(tmp_path: Path) -> None:
    table = make_trip_table([("t1", "r10", "0", "B1", 443, 500)])
    routes = pd.DataFrame({"route_id": ["r10"]})
    path = tmp_path / "ridership.csv"
    pd.DataFrame({"trip_id": ["other"], "avg_daily_boardings": [5]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_ridership(str(path), CSV_SPEC, table, routes)


def test_shipped_defaults_are_placeholders_and_empty_route_lists() -> None:
    assert tcs.GTFS_PATH.startswith("Path\\To\\")
    assert tcs.OUTPUT_DIR.startswith("Path\\To\\")
    assert all(p.startswith("Path\\To\\") for p in tcs.RIDERSHIP_PATHS.values())
    assert set(tcs.ROUTE_EXCLUSION_GROUPS) == {"coverage", "policy", "grant", "other"}
    assert not any(tcs.ROUTE_EXCLUSION_GROUPS.values())
    assert not tcs.NON_REVENUE_ROUTES


# ---------------------------------------------------------------------------
# Full runs (one per scenario, shared across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flat_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """GTFS + trip_id ridership CSV, flat deadhead allowance."""
    root = tmp_path_factory.mktemp("flat")
    weekday = mock_weekday_trips()
    ridership = root / "ridership_by_trip_weekday.csv"
    pd.DataFrame(
        {
            "trip_id": weekday["trip_id"],
            "stop_id": "",
            "avg_daily_boardings": [(7 * k) % 53 + 1 for k in range(len(weekday))],
        }
    ).to_csv(ridership, index=False)
    out = root / "out"
    run(fast_config(out, ridership))
    return out


@pytest.fixture(scope="module")
def tods_vendor_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """GTFS + TODS deadheads + a vendor workbook matched on route, time, direction, block."""
    root = tmp_path_factory.mktemp("tods")
    with zipfile.ZipFile(MOCK_TODS) as archive:
        archive.extractall(root / "tods")
    tods_dir = next(p.parent for p in (root / "tods").rglob("trips_supplement.txt"))
    weekday = mock_weekday_trips()
    rows = []
    for k, trip in enumerate(weekday.itertuples()):
        hh, mm, _ = str(trip.first_departure).split(":")
        hour = int(hh) % 24
        rows.append(
            {
                "ROUTE_NAME": trip.route_short_name,
                "DIRECTION_NAME": "OUTBOUND" if trip.direction_id == "0" else "INBOUND",
                "BLOCK": trip.block_id,
                "TRIP_START_TIME": f"{hour % 12 or 12}:{mm} {'AM' if hour < 12 else 'PM'}",
                "PASSENGERS_ON": (7 * k) % 53 + 1,
                "TRIPS_COUNT": 2 if k % 17 == 0 else 20,
            }
        )
    ridership = root / "RIDERSHIP_BY_ROUTE_AND_TRIP_weekday.xlsx"
    pd.DataFrame(rows).to_excel(ridership, index=False)
    out = root / "out"
    run(fast_config(out, ridership, ridership=RidershipSpec(), tods_path=str(tods_dir)))
    return out


def read(out: Path, table: str) -> pd.DataFrame:
    """Read one per-day output table."""
    return pd.read_csv(out / f"trip_cut_{table}_weekday.csv")


def summary(out: Path) -> pd.Series:
    """The weekday column of trip_cut_summary.csv, keyed by metric."""
    return pd.read_csv(out / "trip_cut_summary.csv").set_index("metric")["weekday"]


def test_run_writes_screening_outputs_and_no_selection_tables(flat_run: Path) -> None:
    for table in ("options", "shortlist", "grouped_shortlist", "sessions", "review", "trips"):
        assert (flat_run / f"trip_cut_{table}_weekday.csv").exists(), table
    assert not list(flat_run.glob("trip_cut_groups*")) and not list(flat_run.glob("*candidates*"))
    assert "TARGET_CUT_PERCENT" not in (flat_run / "trip_cut_savings_runlog.txt").read_text()

    stats = summary(flat_run)
    assert stats["option_savings_are_additive"] == "False"
    assert "peak_vehicles" in stats and "vehicle_hours_saved" not in stats
    assert list(read(flat_run, "vehicle_profile").columns) == ["time", "vehicles"]
    assert not [c for c in read(flat_run, "blocks").columns if c.endswith("_after")]
    assert "selected" not in read(flat_run, "trips").columns


def test_ratings_follow_the_configured_thresholds(flat_run: Path) -> None:
    options = read(flat_run, "options")
    assert set(options["option_type"]) == {"single_trip", "frequency_session"}
    rated = options[options["candidate_classification"] != "ineligible"]
    assert not rated.empty
    ratio = rated["boardings_per_vehicle_hour_saved"]
    label = rated["candidate_classification"]
    assert (ratio[label == "excellent"] <= 5.0).all()
    assert ((ratio[label == "good"] > 5.0) & (ratio[label == "good"] <= 10.0)).all()
    assert (ratio[label == "poor"] > 10.0).all()
    assert (rated["vehicle_min_saved"] > 0).all()


def test_sessions_keep_departures_and_daily_endpoints(flat_run: Path) -> None:
    sessions = read(flat_run, "sessions")
    session_trips = read(flat_run, "session_trips")
    assert not sessions.empty
    assert sessions["preserves_service_day_endpoints"].all()
    endpoints = session_trips[session_trips["service_day_endpoint"]]
    assert (endpoints["action"] == "retain").all()
    eligible = sessions[sessions["candidate_classification"] != "ineligible"]
    assert not eligible.empty
    for option in eligible.itertuples():
        removed = session_trips[
            (session_trips["option_id"] == option.option_id) & (session_trips["action"] == "remove")
        ]
        assert set(removed["trip_id"]) == set(json.loads(str(option.trip_ids)))


def test_tods_supplies_scheduled_pull_outs_and_pull_backs(
    tods_vendor_run: Path, flat_run: Path
) -> None:
    stats = summary(tods_vendor_run)
    assert stats["tods_enabled"] == "True"
    assert int(stats["deadhead_audit_fallback_or_unverified_rows"]) == 0
    assert float(stats["baseline_deadhead_hours"]) != float(
        summary(flat_run)["baseline_deadhead_hours"]
    )
    sources = read(tods_vendor_run, "deadhead_audit")["source"].astype(str)
    assert sources.str.endswith("_PO").any() and sources.str.endswith("_PB").any()


def test_vendor_rows_match_on_direction_and_block(tods_vendor_run: Path) -> None:
    matches = read(tods_vendor_run, "ridership_match")
    matched = matches[matches["match_status"] == "matched"]
    assert len(matched) >= len(matches) - 1  # a 12:00 AM start is rejected as unreadable
    assert matched["matched_on_block"].all()
    assert (matched["mapped_direction_id"] == matched["gtfs_direction_id"]).all()


def test_low_observation_trips_are_protected(tods_vendor_run: Path) -> None:
    trips = read(tods_vendor_run, "trips")
    low = trips[trips["low_observations"]]
    assert len(low) >= 10
    assert (low["ineligible_reason"] == "low observation count").all()


def test_workbooks_export_and_reexport_from_saved_tables(flat_run: Path, tmp_path: Path) -> None:
    index = read(flat_run, "workbook_index")
    books = list((flat_run / "proposal_workbooks").glob("*.xlsx"))
    assert len(books) == len(index) > 0
    reexport = tmp_path / "again"
    args = ["--workbooks-from", str(flat_run), "--output-dir", str(reexport)]
    assert main([*args, "--workbook-limit", "2"]) == 0
    assert len(list((reexport / "proposal_workbooks").glob("*.xlsx"))) == len(books)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_main_returns_2_for_placeholders_and_bad_settings(tmp_path: Path) -> None:
    assert main([]) == 2
    args = ["--gtfs", str(MOCK_FEED), "--ridership", "weekday=x.csv", "--output-dir", str(tmp_path)]
    assert main([*args, "--pull-deadhead", "120", "--min-release-gap", "90"]) == 2


def test_main_returns_1_without_blocks(tmp_path: Path) -> None:
    feed = tmp_path / "feed"
    with zipfile.ZipFile(MOCK_FEED) as archive:
        archive.extractall(feed)
    trips_path = feed / "dc" / "trips.txt"
    pd.read_csv(trips_path, dtype=str).drop(columns=["block_id"]).to_csv(trips_path, index=False)
    ridership = tmp_path / "r.csv"
    pd.DataFrame({"trip_id": ["x"], "avg_daily_boardings": [1]}).to_csv(ridership, index=False)
    args = [
        "--gtfs",
        str(feed / "dc"),
        "--ridership",
        f"weekday={ridership}",
        "--output-dir",
        str(tmp_path / "out"),
        "--match-mode",
        "trip_id",
        "--boardings-col",
        "avg_daily_boardings",
    ]
    assert main(args) == 1
