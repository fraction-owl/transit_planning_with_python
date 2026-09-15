from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import openpyxl
import pandas as pd
import pytest

import scripts.facilities_tools.bay_change_sweep as target

# ---------------------------------------------------------------------------
# Synthetic Step 1 timeline: two blocks meeting at a two-bay cluster
# ---------------------------------------------------------------------------

CLUSTER = {"S1": "A", "S2": "B"}


def _row(
    block: str,
    ts: str,
    status: str,
    trip: str,
    route: str,
    stop: str = "",
    arr: str = "",
    dep: str = "",
) -> dict[str, Any]:
    return {
        "Timestamp": ts,
        "Block": block,
        "Route": f"R{route}",
        "Route Short Name": route,
        "Direction": "0",
        "Trip Headsign": f"Route {route} headsign",
        "Trip ID": trip,
        "Stop ID": stop,
        "Stop Name": f"Stop {stop}" if stop else "",
        "Stop Sequence": "1" if stop else "",
        "Arrival Time": arr,
        "Departure Time": dep,
        "Status": status,
        "Layover Location": "",
        "Prev Trip ID": "",
        "Next Trip ID": "",
        "Timepoint": "0",
    }


def _timeline() -> pd.DataFrame:
    """Block B1 (route 10) and block B2 (routes 20 then 21) both arrive at bay A at 08:00.

    B1: T1 arrives S1 08:00, sits, departs as T2 from S1 at 08:10 (gap 10).
    B2: T3 arrives S1 08:00, sits, departs as T4 (route 21) from S1 at 08:20 (interline, gap 20).
    Bay A therefore has two buses at 08:00 and 08:01: two conflict minutes.
    """
    rows = [
        # B1 / T1: route 10, starts at S9 07:50, ends at S1 08:00
        _row("B1", "07:50", "DEPART", "T1", "10", "S9", "07:50", "07:50"),
        *[_row("B1", f"07:5{m}", "TRAVELING BETWEEN STOPS", "T1", "10") for m in range(1, 10)],
        _row("B1", "08:00", "ARRIVE", "T1", "10", "S1", "08:00", "08:00"),
        _row("B1", "08:01", "ARRIVE", "T1", "10", "S1", "08:00", "08:10"),
        *[_row("B1", f"08:0{m}", "DWELL", "T1", "10", "S1", "08:00", "08:10") for m in range(2, 5)],
        *[
            _row("B1", f"08:0{m}", "LOADING", "T2", "10", "S1", "08:00", "08:10")
            for m in range(5, 10)
        ],
        # B1 / T2: route 10, starts at S1 08:10, ends at S9 08:20
        _row("B1", "08:10", "DEPART", "T2", "10", "S1", "08:10", "08:10"),
        *[_row("B1", f"08:1{m}", "TRAVELING BETWEEN STOPS", "T2", "10") for m in range(1, 10)],
        _row("B1", "08:20", "ARRIVE", "T2", "10", "S9", "08:20", "08:20"),
        # B2 / T3: route 20, starts at S8 07:45, ends at S1 08:00
        _row("B2", "07:45", "DEPART", "T3", "20", "S8", "07:45", "07:45"),
        *[_row("B2", f"07:{m}", "TRAVELING BETWEEN STOPS", "T3", "20") for m in range(46, 60)],
        _row("B2", "08:00", "ARRIVE", "T3", "20", "S1", "08:00", "08:00"),
        _row("B2", "08:01", "ARRIVE", "T3", "20", "S1", "08:00", "08:20"),
        *[
            _row("B2", f"08:{m:02d}", "LAYOVER", "T3", "20", "S1", "08:00", "08:20")
            for m in range(2, 15)
        ],
        *[
            _row("B2", f"08:{m}", "LOADING", "T4", "21", "S1", "08:00", "08:20")
            for m in range(15, 20)
        ],
        # B2 / T4: route 21, starts at S1 08:20, ends at S8 08:35
        _row("B2", "08:20", "DEPART", "T4", "21", "S1", "08:20", "08:20"),
        *[_row("B2", f"08:{m}", "TRAVELING BETWEEN STOPS", "T4", "21") for m in range(21, 35)],
        _row("B2", "08:35", "ARRIVE", "T4", "21", "S8", "08:35", "08:35"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def step1_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "step1"
    folder.mkdir()
    _timeline().to_csv(folder / "all_blocks_timeline.csv", index=False)
    return folder


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target, "CLUSTER_STOPS", CLUSTER)
    monkeypatch.setattr(target, "CLUSTER_CAPACITY", {})
    monkeypatch.setattr(target, "SAME_BAY_GROUPS", [])
    monkeypatch.setattr(target, "SHIFT_TOGETHER", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ts", "expected"), [("08:05", 485), ("25:10", 1510), ("", None), ("x", None), (None, None)]
)
def test_timestamp_to_minutes(ts: str | None, expected: int | None) -> None:
    assert target.timestamp_to_minutes(ts) == expected


def test_minutes_to_hhmm_past_midnight() -> None:
    assert target.minutes_to_hhmm(1590) == "26:30"
    assert target.minutes_to_hhmm(None) == ""


# ---------------------------------------------------------------------------
# Trips, chains, route-ends
# ---------------------------------------------------------------------------


def test_load_timeline_reads_csv_and_labels_bays(step1_folder: Path, cluster: None) -> None:
    df = target.load_timeline(str(step1_folder))
    assert set(df["Bay"]) == {"A", ""}
    assert df["Minute"].dtype.kind == "i"


def test_build_trips_scheduled_ends(step1_folder: Path, cluster: None) -> None:
    trips = target.build_trips(target.load_timeline(str(step1_folder))).set_index("trip_id")
    assert list(trips.index) == ["T1", "T2", "T3", "T4"]
    assert (trips.loc["T1", "start"], trips.loc["T1", "end"]) == (470, 480)
    assert (trips.loc["T1", "first_bay"], trips.loc["T1", "last_bay"]) == ("", "A")
    assert (trips.loc["T2", "start"], trips.loc["T2", "first_bay"]) == (490, "A")
    assert trips.loc["T4", "route"] == "21"


def test_build_trips_empty_input_has_columns() -> None:
    empty = pd.DataFrame(columns=list(_row("B", "08:00", "DEPART", "T", "1")) + ["Minute", "Bay"])
    trips = target.build_trips(empty)
    assert trips.empty
    assert list(trips.columns) == target.TRIP_COLUMNS


def test_block_chains_gaps_and_interlines(step1_folder: Path, cluster: None) -> None:
    trips = target.build_trips(target.load_timeline(str(step1_folder)))
    chains = target.build_block_chains(trips).set_index("block")
    assert chains.loc["B1", "gap_min"] == 10
    assert chains.loc["B1", "where"] == "same stop (Bay A)"
    assert not chains.loc["B1", "interline"]
    assert chains.loc["B2", "gap_min"] == 20
    assert chains.loc["B2", "interline"]
    assert (chains.loc["B2", "from_route_end"], chains.loc["B2", "to_route_end"]) == (
        "20 arrive",
        "21 depart",
    )


def test_route_ends_and_interline_summary(step1_folder: Path, cluster: None) -> None:
    trips = target.build_trips(target.load_timeline(str(step1_folder)))
    chains = target.build_block_chains(trips)
    ends = target.build_route_ends(trips, chains).set_index("route_end")
    assert sorted(ends.index) == ["10 arrive", "10 depart", "20 arrive", "21 depart"]
    assert ends.loc["10 arrive", "bays"] == "A"
    assert ends.loc["10 arrive", "min_gap_after"] == 10
    assert ends.loc["21 depart", "interlines_with"] == "20"
    inter = target.interline_summary(chains)
    assert len(inter) == 1
    assert (inter.iloc[0]["from_route"], inter.iloc[0]["to_route"]) == ("20", "21")
    assert not inter.iloc[0]["shift_together_needed"]


def test_route_ends_with_no_chains() -> None:
    trips = pd.DataFrame(
        [
            {
                "trip_id": "T1",
                "block": "B1",
                "route": "10",
                "headsign": "h",
                "direction": "0",
                "start": 470,
                "end": 480,
                "first_stop": "S9",
                "first_bay": "",
                "last_stop": "S1",
                "last_bay": "A",
                "through_bays": "",
            }
        ]
    )
    chains = target.build_block_chains(trips)
    assert chains.empty
    ends = target.build_route_ends(trips, chains)
    assert ends.loc[0, "min_gap_before"] == ""
    assert target.interline_summary(chains).empty


# ---------------------------------------------------------------------------
# Change / expand_groups / Standard / Feasibility
# ---------------------------------------------------------------------------


def test_change_combine_and_compatible() -> None:
    a = target.Change(moves={"10 arrive": "B"})
    b = target.Change(shifts={"20": 2})
    c = target.Change(shifts={"20": -2})
    assert a.compatible(b)
    assert not b.compatible(c)
    combined = a.combine(b)
    assert combined.label() == "10 arrive to Bay B; 20 +2 min"
    assert b.combine(c).shifts == {}  # shifts on one key add up; zero drops out


def test_expand_groups_moves_members_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target, "SAME_BAY_GROUPS", [["10 arrive", "10 depart"]])
    monkeypatch.setattr(target, "SHIFT_TOGETHER", [["20 arrive", "21 depart"]])
    out = target.expand_groups(target.Change(moves={"10 arrive": "B"}, shifts={"20 arrive": 3}))
    assert out.moves == {"10 arrive": "B", "10 depart": "B"}
    assert out.shifts == {"20 arrive": 3, "21 depart": 3}


def test_standard_baseline_conflicts(step1_folder: Path, cluster: None) -> None:
    df = target.load_timeline(str(step1_folder))
    std = target.Standard("x", df, target.build_trips(df))
    total, per_bay, conflict = std.score(std.bays0, std.minutes0)
    assert (total, per_bay) == (2, {"A": 2})
    who, kinds = std.detail(std.bays0, std.minutes0, conflict)
    assert kinds == {"boarding/boarding": 2}
    assert set(who.values()) == {("10 arrive", "20 arrive")}


def test_standard_bay_move_clears_conflict(step1_folder: Path, cluster: None) -> None:
    df = target.load_timeline(str(step1_folder))
    std = target.Standard("x", df, target.build_trips(df))
    bays, minutes = std.apply(target.Change(moves={"20 arrive": "B"}))
    assert std.score(bays, minutes)[0] == 0


def test_standard_shift_drops_own_overlapping_rows(step1_folder: Path, cluster: None) -> None:
    df = target.load_timeline(str(step1_folder))
    std = target.Standard("x", df, target.build_trips(df))
    # Moving route 10's departure 5 minutes earlier slides its LOADING rows over
    # T1's arrival rows on the same block; the departing rows win, so bay A still
    # holds one route-10 bus and one route-20 bus at 08:00-08:01.
    bays, minutes = std.apply(target.Change(shifts={"10 depart": -5}))
    assert std.score(bays, minutes)[0] == 2
    assert (bays == -1).sum() > 0


def test_feasibility_rejects_gap_below_minimum(step1_folder: Path, cluster: None) -> None:
    trips = target.build_trips(target.load_timeline(str(step1_folder)))
    feas = target.Feasibility(trips, target.build_block_chains(trips))
    ok, reason, tightest = feas.check(target.Change(shifts={"10 depart": -8}))
    assert not ok
    assert "goes 10 -> 2 min, below 3" in reason
    ok, reason, tightest = feas.check(target.Change(shifts={"10 depart": -5}))
    assert (ok, reason, tightest) == (True, "", 5)
    # Interline hand-offs use the lower minimum.
    ok, _, tightest = feas.check(target.Change(shifts={"21 depart": -18}))
    assert (ok, tightest) == (True, 2)
    assert not feas.check(target.Change(shifts={"21 depart": -19}))[0]


def test_feasibility_off_clockface_note(step1_folder: Path, cluster: None) -> None:
    trips = target.build_trips(target.load_timeline(str(step1_folder)))
    feas = target.Feasibility(trips, target.build_block_chains(trips))
    # T4 departs the cluster at 08:20 (not on :00/:30), so a shift is not flagged...
    assert feas.off_clockface(target.Change(shifts={"21 depart": 1})) == ""
    # ...while a whole-route shift of route 20 moves nothing that departs the cluster.
    assert feas.off_clockface(target.Change(shifts={"20": 1})) == ""


def test_is_win_and_rank_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target, "REQUIRE_NO_WORSE_ON", "all")
    assert target.is_win({"a": 2, "b": 0})
    assert not target.is_win({"a": 2, "b": -1})
    assert not target.is_win({"a": 0, "b": 0})
    assert not target.is_win(None)
    monkeypatch.setattr(target, "REQUIRE_NO_WORSE_ON", "any")
    assert target.is_win({"a": 2, "b": -1})
    assert target.rank_key({"a": 1, "b": 5}) == (-5, -1)


# ---------------------------------------------------------------------------
# Discover and sweep end to end
# ---------------------------------------------------------------------------


def test_run_discover_writes_workbook_and_stub(
    step1_folder: Path, tmp_path: Path, cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target, "TIMELINES", {"Direct": str(step1_folder)})
    monkeypatch.setattr(target, "OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(target, "SCENARIO_LABEL", "test")
    out = target.run_discover()
    wb = openpyxl.load_workbook(out, read_only=True)
    assert wb.sheetnames == [
        "Route-ends Direct",
        "Block chains Direct",
        "Interlines Direct",
        "Config stub",
    ]
    stub = "\n".join(str(v) for v in pd.read_excel(out, sheet_name="Config stub")["config_stub"])
    assert '"route_end": "20 arrive"' in stub
    assert '"21": [-3, -2, -1, 1, 2, 3],  # whole route' in stub


def test_run_sweep_finds_bay_move(
    step1_folder: Path, tmp_path: Path, cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        target, "TIMELINES", {"Direct": str(step1_folder), "Likely": str(step1_folder)}
    )
    monkeypatch.setattr(target, "OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(target, "SCENARIO_LABEL", "test")
    monkeypatch.setattr(target, "BAY_CANDIDATES", [{"route_end": "20 arrive", "bays": ["A", "B"]}])
    monkeypatch.setattr(target, "SHIFT_CANDIDATES", {"10 depart": [-8, -5]})
    out = target.run_sweep()

    sheets = openpyxl.load_workbook(out, read_only=True).sheetnames
    assert sheets == [
        "Baseline",
        "Easy wins",
        "Pairs",
        "Packages",
        "Package finalists",
        "Rejected",
        "Config used",
    ]
    baseline = pd.read_excel(out, sheet_name="Baseline")
    assert list(baseline["standard"]) == ["Direct", "Likely"]
    assert list(baseline["total"]) == [2, 2]
    assert list(baseline["Bay A"]) == [2, 2]

    wins = pd.read_excel(out, sheet_name="Easy wins")
    assert list(wins["change"]) == ["20 arrive to Bay B"]
    assert (wins.loc[0, "Direct before"], wins.loc[0, "Direct after"]) == (2, 0)
    assert wins.loc[0, "Direct removed"] == "A 08:00-08:01 10/20"

    rejected = pd.read_excel(out, sheet_name="Rejected").set_index("change")
    assert "below 3" in rejected.loc["10 depart -8 min", "reason"]
    assert rejected.loc["10 depart -5 min", "reason"] == "no improvement"

    pairs = pd.read_excel(out, sheet_name="Pairs")
    assert list(pairs["change"]) == ["20 arrive to Bay B; 10 depart -5 min"]

    packages = pd.read_excel(out, sheet_name="Packages")
    assert list(packages["package"]) == ["20 arrive to Bay B"]
    assert packages.loc[0, "Direct conflict minutes"] == 0
    finalists = pd.read_excel(out, sheet_name="Package finalists")
    assert finalists.loc[0, "change"] == "20 arrive to Bay B"
    assert finalists.loc[0, "Direct before"] == 2  # scored against the untouched baseline


def test_main_returns_2_on_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target, "TIMELINES", {"Direct": r"Path\To\block_status_output\x"})
    assert target.main() == 2


def test_main_returns_2_when_folder_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target, "TIMELINES", {"Direct": str(tmp_path / "missing")})
    monkeypatch.setattr(target, "OUTPUT_FOLDER", str(tmp_path / "out"))
    assert target.main() == 2


def test_main_runs_discover(
    step1_folder: Path, tmp_path: Path, cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target, "TIMELINES", {"Direct": str(step1_folder)})
    monkeypatch.setattr(target, "OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(target, "DISCOVER_ONLY", True)
    with patch.object(target, "run_sweep") as sweep:
        assert target.main() == 0
    sweep.assert_not_called()
    assert (tmp_path / "out" / f"{target.SCENARIO_LABEL}_discover.xlsx").is_file()
