"""Regression tests on a one-block TODS bus example, across the hours report and field workbook.

The fixture (``tests/fixtures/tods_bus_example``, see its README) holds one block
whose minutes can be checked by hand: 200 revenue-trip minutes, 95 stationary
minutes and 20 pull-out/pull-in minutes, 315 vehicle minutes from 09:45 to 15:00.
Each test applies one TODS change and checks that ``scheduled_service_hours.py``
and ``printable_block_schedules.py`` tell the same story.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.field_tools.printable_block_schedules as pbs
import scripts.gtfs_exports.scheduled_service_hours as ssh
import scripts.ridership_tools.trip_cut_savings as tcs

FIXTURE = Path(__file__).parent / "fixtures" / "tods_bus_example"


@pytest.fixture
def feed(tmp_path: Path) -> tuple[Path, Path]:
    """A writable copy of the example: (gtfs folder, tods folder)."""
    shutil.copytree(FIXTURE, tmp_path / "example")
    return tmp_path / "example" / "gtfs", tmp_path / "example" / "tods"


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _blocks(gtfs: Path, tods: Path, out: Path) -> pd.DataFrame:
    cfg = ssh.Config(
        gtfs_path=gtfs,
        tods_path=tods,
        output_dir=out,
        periods=[("Nov", "20261101", "20261130")],
        tods_movements_complete=True,
    )
    return ssh.run(cfg)["block"].set_index("block_id")


def _workbooks(gtfs: Path, tods: Path, out: Path) -> tuple[int, dict[str, pd.DataFrame]]:
    with (
        patch.object(pbs, "GTFS_FOLDER_PATH", str(gtfs)),
        patch.object(pbs, "BASE_OUTPUT_PATH", str(out)),
        patch.object(pbs, "TODS_PATH", str(tods)),
        patch.object(pbs, "FILTER_ROUTE_SHORT_NAMES", []),
        patch.object(pbs, "FILTER_SERVICE_IDS", []),
        patch.object(pbs, "TIMEPOINTS_ONLY", False),
        patch.object(pbs, "INCLUDE_LAYOVER_ROWS", True),
    ):
        code = pbs.main()
    sheets = {
        path.name: pd.read_excel(path, dtype=str, keep_default_na=False)
        for path in sorted(out.glob("*.xlsx"))
    }
    return code, sheets


def _minutes(text: pd.Series) -> pd.Series:
    return pd.to_timedelta(text + ":00").dt.total_seconds() / 60


def _wait_minutes(sheet: pd.DataFrame) -> float:
    waits = sheet[sheet["Event Type"].isin(["Layover", "Between trips"])]
    span = _minutes(waits["Scheduled Departure Time"]) - _minutes(waits["Scheduled Arrival Time"])
    return float(span.sum())


def test_baseline_block_accounts_for_all_315_minutes(
    feed: tuple[Path, Path], tmp_path: Path
) -> None:
    gtfs, tods = feed
    block = _blocks(gtfs, tods, tmp_path / "hours").loc["BLOCK-A"]
    assert block["revenue_trips"] == 4
    assert block["revenue_trip_hours"] * 60 == pytest.approx(200, abs=0.5)
    assert block["layover_hours"] * 60 == pytest.approx(95, abs=0.5)
    assert (block["pull_out_hours"] + block["pull_in_hours"]) * 60 == pytest.approx(20, abs=0.5)
    assert block["vehicle_hours"] * 60 == pytest.approx(315, abs=0.5)
    assert (block["start"], block["end"]) == ("09:45", "15:00")

    code, sheets = _workbooks(gtfs, tods, tmp_path / "workbooks")
    assert code == 0
    # The file name spans the pull-out to the pull-in, as the hours report does.
    (name,) = sheets
    assert name == "block_BLOCK-A_12_0945-1500_Daily.xlsx"
    sheet = sheets[name]
    # Every stationary minute has a row, including the 5 minutes between the
    # pull-out's 09:55 arrival and trip 101's 10:00 departure.
    assert _wait_minutes(sheet) == 95
    first_wait = sheet.iloc[1]
    assert first_wait["Event Type"] == "Layover"
    assert (first_wait["Scheduled Arrival Time"], first_wait["Scheduled Departure Time"]) == (
        "09:55",
        "10:00",
    )
    assert sheet["Event Type"].iloc[0] == "Pull-out"
    assert sheet["Event Type"].iloc[-1] == "Pull-in"


def test_departure_edit_reaches_both_outputs(feed: tuple[Path, Path], tmp_path: Path) -> None:
    gtfs, tods = feed
    # TODS corrects trip 101's first departure from 10:00 to 10:05.
    _append(tods / "stop_times_supplement.txt", "101,10:05:00,10:05:00,A,1,1\n")

    block = _blocks(gtfs, tods, tmp_path / "hours").loc["BLOCK-A"]
    assert block["revenue_trip_hours"] * 60 == pytest.approx(195, abs=0.5)
    assert block["layover_hours"] * 60 == pytest.approx(100, abs=0.5)

    _, sheets = _workbooks(gtfs, tods, tmp_path / "workbooks")
    (sheet,) = sheets.values()
    stops = sheet[(sheet["Trip ID"] == "101") & (sheet["Event Type"] == "Stop")]
    assert stops["Scheduled Departure Time"].iloc[0] == "10:05"
    assert _wait_minutes(sheet) == 100


def test_route_deletion_leaves_nothing_to_count(
    feed: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs, tods = feed
    (tods / "routes_supplement.txt").write_text("route_id,TODS_delete\n12,1\n", encoding="utf-8")

    # Route 12 was the only route: neither output may count its trips.
    with pytest.raises(ValueError, match="No trips remain after applying the TODS supplements"):
        _blocks(gtfs, tods, tmp_path / "hours")
    with caplog.at_level(logging.WARNING):
        code, sheets = _workbooks(gtfs, tods, tmp_path / "workbooks")
    assert code == 1 and not sheets
    assert "Ignoring 6 trip(s) on route(s) deleted in routes_supplement.txt (12)" in caplog.text


def test_route_deletion_drops_only_that_routes_block(
    feed: tuple[Path, Path], tmp_path: Path
) -> None:
    gtfs, tods = feed
    (tods / "routes_supplement.txt").write_text("route_id,TODS_delete\n12,1\n", encoding="utf-8")
    _append(gtfs / "routes.txt", "13,EX,13,Cross Town,3\n")
    _append(gtfs / "trips.txt", "13,DAILY,201,0,BLOCK-B\n")
    _append(gtfs / "stop_times.txt", "201,08:00:00,08:00:00,A,1,1\n201,08:30:00,08:30:00,B,2,1\n")

    blocks = _blocks(gtfs, tods, tmp_path / "hours")
    assert blocks.index.tolist() == ["BLOCK-B"]
    assert blocks.loc["BLOCK-B", "revenue_trips"] == 1

    code, sheets = _workbooks(gtfs, tods, tmp_path / "workbooks")
    assert code == 0
    assert list(sheets) == ["block_BLOCK-B_13_0800-0830_Daily.xlsx"]


def test_trip_reassigned_off_a_deleted_route_is_kept(
    feed: tuple[Path, Path], tmp_path: Path
) -> None:
    gtfs, tods = feed
    # Route 12 is deleted, but TODS moves trip 101 to a new route 13 first.
    (tods / "routes_supplement.txt").write_text(
        "route_id,agency_id,route_short_name,route_long_name,route_type,TODS_delete\n"
        "12,,,,,1\n13,EX,13,Shuttle,3,\n",
        encoding="utf-8",
    )
    _append(tods / "trips_supplement.txt", "13,DAILY,101,0,BLOCK-A,\n")

    block = _blocks(gtfs, tods, tmp_path / "hours").loc["BLOCK-A"]
    assert (block["revenue_trips"], block["routes"]) == (1, "13")

    _, sheets = _workbooks(gtfs, tods, tmp_path / "workbooks")
    (sheet,) = sheets.values()
    assert set(sheet.loc[sheet["Event Type"] == "Stop", "Trip ID"]) == {"101"}


def test_tods_v1_deadhead_tables_stop_every_reader(
    feed: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gtfs, tods = feed
    (tods / "deadheads.txt").write_text("deadhead_id\nDH1\n", encoding="utf-8")
    message = "contains TODS v1 deadhead tables (deadheads.txt)"
    pattern = re.escape(message)

    with pytest.raises(ValueError, match=pattern):
        _blocks(gtfs, tods, tmp_path / "hours")
    with caplog.at_level(logging.ERROR):
        code, sheets = _workbooks(gtfs, tods, tmp_path / "workbooks")
    assert code == 1 and not sheets
    assert message in caplog.text
    no_trips = pd.DataFrame(columns=["block_id", "start_min", "end_min", "service_id"])
    with pytest.raises(ValueError, match=pattern):
        tcs.DeadheadModel(str(tods), {}, no_trips, tcs.BlockRules())


def test_trip_cut_savings_refuses_route_deletions(feed: tuple[Path, Path]) -> None:
    gtfs, tods = feed
    (tods / "routes_supplement.txt").write_text("route_id,TODS_delete\n12,1\n", encoding="utf-8")
    tables = {
        name: pd.read_csv(gtfs / f"{name}.txt", dtype=str, keep_default_na=False)
        for name in ("trips", "stops", "routes")
    }
    trips = tables["trips"].set_index("trip_id")
    trips["start_min"] = [600, 660, 780, 840]
    trips["end_min"] = trips["start_min"] + 50
    with pytest.raises(ValueError, match="Apply TODS route deletions upstream"):
        tcs.DeadheadModel(str(tods), tables, trips, tcs.BlockRules())
