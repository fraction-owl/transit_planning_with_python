"""Tests for trip_cut_savings: block savings arithmetic, greedy selection, and the pipeline."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.ridership_tools.trip_cut_savings import (
    BlockRules,
    Guards,
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
    select_cuts,
    service_effects,
)

MOCK_FEED = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"
RULES = BlockRules(min_release_gap_min=90, pull_deadhead_min=30)

# Four 60-minute trips with 10-minute layovers: 0-60, 70-130, 140-200, 210-270.
STARTS = [0, 70, 140, 210]
ENDS = [60, 130, 200, 270]


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


# ---------------------------------------------------------------------------
# Block arithmetic
# ---------------------------------------------------------------------------


def test_block_metrics_counts_layover_and_one_deadhead_per_piece() -> None:
    metrics = block_metrics(STARTS, ENDS, RULES)
    assert metrics == {
        "trips": 4,
        "pieces": 1,
        "revenue_min": 240,
        "layover_min": 30,
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
    assert solo.revenue_min == 60
    assert solo.gap_left_min == 80

    # Two mid-block trips leave 150 minutes: the vehicle is released.
    pair = runs[(1, 2)]
    assert pair.cut_type == "mid_block_release"
    assert pair.vehicle_min_saved == 120 + 30 - 30
    assert pair.gap_left_min == 150


def test_enumerate_runs_matches_recount_for_every_run() -> None:
    starts = [0, 70, 300, 370, 440]  # a released gap already sits between trips 1 and 2
    ends = [60, 130, 360, 430, 500]
    total = block_metrics(starts, ends, RULES)["vehicle_min"]
    for run in enumerate_runs(starts, ends, RULES):
        keep = [k for k in range(len(starts)) if not run.i <= k <= run.j]
        after = block_metrics([starts[k] for k in keep], [ends[k] for k in keep], RULES)
        assert run.vehicle_min_saved == total - after["vehicle_min"], run


def test_enumerate_runs_skips_runs_through_ineligible_trips() -> None:
    runs = enumerate_runs(STARTS, ENDS, RULES, eligible=[True, False, True, True])
    spans = {(r.i, r.j) for r in runs}
    assert (0, 0) in spans and (2, 3) in spans
    assert not any(i <= 1 <= j for i, j in spans)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@pytest.fixture
def two_blocks() -> pd.DataFrame:
    return make_trip_table(
        [
            ("a1", "R1", "0", "A", 0, 60),
            ("a2", "R1", "1", "A", 70, 130),
            ("a3", "R1", "0", "A", 140, 200),
            ("a4", "R1", "1", "A", 210, 270),
            ("b1", "R1", "0", "B", 30, 90),
            ("b2", "R1", "1", "B", 100, 160),
        ]
    )


BOARDINGS = {"a1": 50.0, "a2": 1.0, "a3": 2.0, "a4": 40.0, "b1": 30.0, "b2": 3.0}


def test_select_cuts_takes_fewest_boardings_per_hour_first(two_blocks: pd.DataFrame) -> None:
    blocks = build_blocks(two_blocks)
    result = select_cuts(
        blocks, two_blocks, BOARDINGS, set(BOARDINGS), RULES, Guards(), "vehicle_hours", 100
    )
    # a2 alone only adds layover; a2 + a3 releases the vehicle for 3 boardings / 2 hours.
    assert result.cut_trip_ids == {"a2", "a3"}
    assert result.groups["cut_type"].tolist() == ["mid_block_release"]
    assert blocks["A"].trip_ids == ["a1", "a4"]


def test_select_cuts_rescores_a_block_after_each_cut(two_blocks: pd.DataFrame) -> None:
    blocks = build_blocks(two_blocks)
    result = select_cuts(
        blocks, two_blocks, BOARDINGS, set(BOARDINGS), RULES, Guards(), "vehicle_hours", 150
    )
    assert result.groups["trip_ids"].tolist() == ["a2;a3", "b2"]
    assert result.groups["cumulative_hours"].iloc[-1] == pytest.approx((120 + 70) / 60, abs=0.01)
    assert result.group_of_trip == {"a2": 1, "a3": 1, "b2": 2}


def test_select_cuts_zero_target_selects_nothing(two_blocks: pd.DataFrame) -> None:
    blocks = build_blocks(two_blocks)
    result = select_cuts(
        blocks, two_blocks, BOARDINGS, set(BOARDINGS), RULES, Guards(), "vehicle_hours", 0
    )
    assert result.groups.empty and not result.cut_trip_ids


def test_select_cuts_span_guard_rejects_first_and_last_trips(two_blocks: pd.DataFrame) -> None:
    blocks = build_blocks(two_blocks)
    guards = Guards(protect_route_span=True)
    result = select_cuts(
        blocks, two_blocks, BOARDINGS, set(BOARDINGS), RULES, guards, "vehicle_hours", 60
    )
    # a2 is direction 1's first trip and a3 direction 0's last, so their pair is refused.
    assert result.rejected["route_span"] >= 1
    assert result.cut_trip_ids == {"b2"}


def test_service_effects_reports_new_gap_and_span() -> None:
    table = make_trip_table([(f"t{k}", "R", "0", "X", 30 * k, 30 * k + 20) for k in range(4)])
    lookup, index = build_trip_lookup(table), build_route_dir_index(table)
    assert service_effects(["t1"], lookup, index) == (False, 60)
    assert service_effects(["t1", "t2"], lookup, index) == (False, 90)
    assert service_effects(["t0"], lookup, index) == (True, None)


# ---------------------------------------------------------------------------
# Ridership and cycle times
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
            "TRIP_START_TIME": ["07:23", "7:24:00 AM", "08:23", "08:23"],
            "PASSENGERS_ON": ["10", "20", "7", "100"],
            "SURVEY_DATE": ["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-02"],
        }
    ).to_csv(path, index=False)
    spec = RidershipSpec(
        match_mode="route_start_time", boardings_col="PASSENGERS_ON", date_col="SURVEY_DATE"
    )
    boardings = load_ridership(str(path), spec, table, routes)
    assert boardings.to_dict() == {"t1": 15.0, "t2": 7.0}


def test_load_ridership_raises_when_nothing_matches(tmp_path: Path) -> None:
    table = make_trip_table([("t1", "r10", "0", "B1", 443, 500)])
    routes = pd.DataFrame({"route_id": ["r10"]})
    path = tmp_path / "ridership.csv"
    pd.DataFrame({"trip_id": ["other"], "avg_daily_boardings": [5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="matched a trip"):
        load_ridership(str(path), RidershipSpec(), table, routes)


def test_cycle_time_table_headway_spans_midnight_window() -> None:
    # A night window 22:00-06:00 holding departures 22:00, 22:30, 23:00 and 05:00, 05:30:
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
        base = 25 * v
        for leg in range(4):
            start = base + 50 * leg
            rows.append((f"v{v}l{leg}", "R", str(leg % 2), f"B{v}", start, start + 30))
    table = make_trip_table(rows)
    out = cycle_time_table(table, build_blocks(table), RULES, [], 15.0)
    row = out.iloc[0]
    assert row["cycle_min"] == 100.0
    assert row["round_trip_running_min"] == 60.0
    assert row["recovery_pct"] == pytest.approx(66.7, abs=0.1)
    assert row["vehicles_at_cycle"] == 4
    assert row["vehicles_at_target_recovery"] == 3  # ceil(69 / 25)
    assert row["vehicles_freeable_by_runtime_trim"] == 1


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ridership(tmp_path: Path) -> Path:
    with zipfile.ZipFile(MOCK_FEED) as archive:
        with archive.open("dc/trips.txt") as handle:
            trips = pd.read_csv(handle, dtype=str)
    weekday = trips[trips["service_id"] == "weekday"].reset_index(drop=True)
    path = tmp_path / "ridership_by_trip_weekday.csv"
    pd.DataFrame(
        {
            "trip_id": weekday["trip_id"],
            "stop_id": "",
            "avg_daily_boardings": [(7 * k) % 53 + 1 for k in range(len(weekday))],
        }
    ).to_csv(path, index=False)
    return path


def test_main_writes_every_output(tmp_path: Path, mock_ridership: Path) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "--gtfs",
            str(MOCK_FEED),
            "--ridership",
            str(mock_ridership),
            "--output-dir",
            str(out),
            "--target-percent",
            "5",
        ]
    )
    assert code == 0
    for name in (
        "trip_cut_groups.csv",
        "trip_cut_trips.csv",
        "trip_cut_candidates.csv",
        "trip_cut_blocks.csv",
        "trip_cut_cycle_times.csv",
        "trip_cut_vehicle_profile.csv",
        "trip_cut_summary.csv",
        "trip_cut_savings_runlog.txt",
    ):
        assert (out / name).exists(), name

    groups = pd.read_csv(out / "trip_cut_groups.csv")
    assert not groups.empty
    assert (groups["vehicle_hours_saved"] > 0).all()
    assert "mid_block_layover_only" not in set(groups["cut_type"])
    assert groups["cumulative_pct_of_baseline"].iloc[-1] >= 5.0

    trips = pd.read_csv(out / "trip_cut_trips.csv")
    selected = set(trips.loc[trips["selected"], "trip_id"])
    listed = {t for ids in groups["trip_ids"] for t in ids.split(";")}
    assert selected == listed

    blocks = pd.read_csv(out / "trip_cut_blocks.csv")
    summary = pd.read_csv(out / "trip_cut_summary.csv").set_index("metric")["value"]
    assert float(summary["vehicle_hours_saved"]) == pytest.approx(
        blocks["vehicle_hours_saved"].sum(), abs=0.1
    )
    assert "TARGET_CUT_PERCENT" in (out / "trip_cut_savings_runlog.txt").read_text()


def test_main_returns_2_for_placeholders_and_bad_settings(tmp_path: Path) -> None:
    assert main([]) == 2
    args = ["--gtfs", str(MOCK_FEED), "--ridership", "x.csv", "--output-dir", str(tmp_path)]
    assert main([*args, "--pull-deadhead", "120", "--min-release-gap", "90"]) == 2


def test_main_returns_1_without_blocks(tmp_path: Path, mock_ridership: Path) -> None:
    feed = tmp_path / "feed"
    with zipfile.ZipFile(MOCK_FEED) as archive:
        archive.extractall(feed)
    trips_path = feed / "dc" / "trips.txt"
    trips = pd.read_csv(trips_path, dtype=str).drop(columns=["block_id"])
    trips.to_csv(trips_path, index=False)
    code = main(
        [
            "--gtfs",
            str(feed / "dc"),
            "--ridership",
            str(mock_ridership),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
