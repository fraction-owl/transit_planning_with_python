"""Tests for scheduled_service_hours (GTFS + TODS revenue and nonrevenue hours)."""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from openpyxl import load_workbook

import scripts.gtfs_exports.scheduled_service_hours as target

FIXTURES = Path(__file__).parent / "fixtures"
GTFS_ZIP = FIXTURES / "mock_gtfs_dc.zip"
TODS_ZIP = FIXTURES / "mock_tods_dc.zip"


def write_feed(folder: Path, tables: dict[str, list[dict[str, str]]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (folder / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return folder


def stop_time(trip: str, seq: int, stop: str, time: str) -> dict[str, str]:
    return {
        "trip_id": trip,
        "stop_sequence": str(seq),
        "stop_id": stop,
        "arrival_time": time,
        "departure_time": time,
    }


@pytest.fixture()
def small_feed(tmp_path: Path) -> tuple[Path, Path]:
    """One weekday block: two revenue trips A->B, so B->A must be deadheaded."""
    stops = [
        {"stop_id": s, "stop_name": s, "stop_lat": lat, "stop_lon": "-77.0"}
        for s, lat in (("A", "38.90"), ("B", "38.95"), ("A2", "38.90"))
    ]
    gtfs = write_feed(
        tmp_path / "gtfs",
        {
            "trips.txt": [
                {"route_id": "R", "service_id": "wk", "trip_id": t, "block_id": "B1"}
                for t in ("T1", "T2", "T3")
            ],
            "stop_times.txt": [
                stop_time("T1", 1, "A", "06:00:00"),
                stop_time("T1", 2, "B", "07:00:00"),
                stop_time("T2", 1, "A", "08:00:00"),
                stop_time("T2", 2, "B", "09:00:00"),
                # T3 starts at A2, a second stop_id at A's coordinates: no move.
                stop_time("T3", 1, "A2", "10:00:00"),
                stop_time("T3", 2, "B", "11:00:00"),
            ],
            "stops.txt": stops,
            "calendar.txt": [
                {
                    "service_id": "wk",
                    **{d: "1" for d in ("monday", "tuesday", "wednesday", "thursday", "friday")},
                    "saturday": "0",
                    "sunday": "0",
                    "start_date": "20261101",
                    "end_date": "20261130",
                }
            ],
        },
    )
    tods = write_feed(
        tmp_path / "tods",
        {
            "trips_supplement.txt": [
                {
                    "route_id": "NR",
                    "service_id": "wk",
                    "trip_id": t,
                    "block_id": "B1",
                    "TODS_trip_type": k,
                }
                for t, k in (("PO", "pull-out"), ("DH", "deadhead"), ("PB", "pull-back"))
            ]
            # T3 is deleted: its stop_times are then ignored, per the spec.
            + [{"trip_id": "T3", "TODS_delete": "1"}],
            "stop_times_supplement.txt": [
                stop_time("PO", 1, "G", "05:30:00"),
                stop_time("PO", 2, "A", "06:00:00"),
                stop_time("DH", 1, "B", "07:00:00"),
                stop_time("DH", 2, "A", "07:30:00"),
                stop_time("PB", 1, "B", "09:00:00"),
                stop_time("PB", 2, "G", "09:45:00"),
            ],
            "stops_supplement.txt": [
                {"stop_id": "G", "stop_name": "Garage", "stop_lat": "38.80", "stop_lon": "-77.0"}
            ],
        },
    )
    return gtfs, tods


# =============================================================================
# TODS SUPPLEMENTS
# =============================================================================


def test_apply_supplement_deletes_updates_and_adds() -> None:
    base = pd.DataFrame(
        {
            "stop_id": ["1", "2", "3"],
            "stop_name": ["One", "Two", "Three"],
            "stop_url": ["a", "b", "c"],
        }
    )
    supplement = pd.DataFrame(
        {
            "stop_id": ["2", "3", "4"],
            "stop_name": ["", "", "Four"],
            "stop_desc": ["", "Modified", "New"],
            "TODS_delete": ["1", "", ""],
        }
    )
    out = target.apply_supplement(base, supplement, ["stop_id"]).set_index("stop_id")
    # The spec's own example: 2 deleted, 3 updated (blank name ignored), 4 added.
    assert list(out.index) == ["1", "3", "4"]
    assert out.loc["3", "stop_name"] == "Three" and out.loc["3", "stop_desc"] == "Modified"
    assert out.loc["3", "stop_url"] == "c"
    assert out.loc["4", "stop_name"] == "Four" and out.loc["4", "stop_url"] == ""


def test_apply_supplement_rejects_delete_and_re_add() -> None:
    base = pd.DataFrame({"trip_id": ["T1"]})
    supplement = pd.DataFrame({"trip_id": ["T1", "T1"], "TODS_delete": ["1", ""]})
    with pytest.raises(ValueError, match="deletes and re-adds"):
        target.apply_supplement(base, supplement, ["trip_id"])


# =============================================================================
# HOURS
# =============================================================================


def test_small_feed_splits_the_block_day(small_feed: tuple[Path, Path]) -> None:
    gtfs, tods = small_feed
    schedule = target.load_schedule(gtfs, tods)
    assert "T3" not in set(schedule.movements["trip_id"])  # Deleted by TODS.
    (row,) = target.block_hours(schedule.movements, schedule.stop_places).to_dict("records")
    assert row["revenue_trip_hours"] == 2.0  # T1 + T2
    assert row["pull_out_hours"] == 0.5 and row["pull_in_hours"] == 0.75
    assert row["deadhead_hours"] == 0.5
    assert row["layover_hours"] == 0.5  # Deadhead arrives 07:30, T2 leaves 08:00.
    assert row["unresolved_gap_hours"] == 0.0 and row["location_jumps"] == 0
    assert (row["start"], row["end"]) == ("05:30", "09:45")


def test_moving_without_a_scheduled_movement_is_unresolved(small_feed: tuple[Path, Path]) -> None:
    gtfs, _ = small_feed
    # Without TODS, T1 ends at B and T2 starts at A an hour later: an unresolved gap.
    schedule = target.load_schedule(gtfs, None)
    (row,) = target.block_hours(schedule.movements, schedule.stop_places).to_dict("records")
    assert row["unresolved_gap_hours"] == 2.0  # T1->T2 and T2->T3 both change place.
    assert row["layover_hours"] == 0.0
    assert row["location_jumps"] == 2
    assert row["revenue_trip_hours"] == 3.0


def test_same_coordinates_count_as_one_place() -> None:
    places = {"A": ("38.9", "-77.0"), "A2": ("38.9", "-77.0"), "X": ("", "")}
    assert target.same_place("A", "A2", places)
    assert not target.same_place("X", "Y", places)


def test_add_derived_follows_the_layover_setting() -> None:
    df = pd.DataFrame({c: [1.0] for c in target.COMPONENTS})
    counted = target.add_derived(df, layover_in_revenue=True, tods_supplied=True).iloc[0]
    assert counted["vehicle_revenue_hours"] == 2.0 and counted["nonrevenue_hours"] == 4.0
    separate = target.add_derived(df, layover_in_revenue=False, tods_supplied=True).iloc[0]
    assert separate["vehicle_revenue_hours"] == 1.0 and separate["nonrevenue_hours"] == 5.0
    assert counted["vehicle_hours"] == separate["vehicle_hours"] == 7.0
    blank = target.add_derived(df, layover_in_revenue=True, tods_supplied=False).iloc[0]
    assert math.isnan(blank["deadhead_hours"]) and math.isnan(blank["vehicle_hours"])


def test_service_day_types_explicit_mapping() -> None:
    types = target.service_day_types({}, {"wk": "Weekday", "end": "saturday, Sunday"})
    assert types == {"wk": {"Weekday"}, "end": {"Saturday", "Sunday"}}
    with pytest.raises(ValueError, match="unknown day type"):
        target.service_day_types({}, {"x": "Fiesta"})


# =============================================================================
# MOCK FIXTURES (GTFS + TODS)
# =============================================================================


def read_zip(path: Path, folder: str) -> dict[str, list[dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        return {
            Path(n).name: list(csv.DictReader(io.TextIOWrapper(archive.open(n), encoding="utf-8")))
            for n in archive.namelist()
            if n.startswith(f"{folder}/") and n.endswith(".txt")
        }


def trip_hours(stop_times: list[dict[str, str]]) -> dict[str, float]:
    calls: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stop_times:
        calls[row["trip_id"]].append(row)
    out = {}
    for trip, rows in calls.items():
        rows.sort(key=lambda r: int(r["stop_sequence"]))
        start, end = (
            target._seconds(rows[0]["departure_time"]),
            target._seconds(rows[-1]["arrival_time"]),
        )
        out[trip] = (end - start) / 3600
    return out


@pytest.fixture(scope="module")
def mock_tables(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=TODS_ZIP,
        output_dir=tmp_path_factory.mktemp("hours"),
        periods=[("Nov", "20261101", "20261130"), ("Jan", "20260101", "20260131")],
    )
    return target.run(cfg)


def test_mock_hours_match_the_feeds(mock_tables: dict[str, pd.DataFrame]) -> None:
    gtfs, tods = read_zip(GTFS_ZIP, "dc"), read_zip(TODS_ZIP, "dc_tods")
    revenue, nonrevenue = (
        trip_hours(gtfs["stop_times.txt"]),
        trip_hours(tods["stop_times_supplement.txt"]),
    )
    services = mock_tables["service"].set_index("service_id")
    for sid in services.index:
        rev = sum(revenue[t["trip_id"]] for t in gtfs["trips.txt"] if t["service_id"] == sid)
        non = sum(
            nonrevenue[t["trip_id"]] for t in tods["trips_supplement.txt"] if t["service_id"] == sid
        )
        row = services.loc[sid]
        assert row["revenue_trip_hours"] == pytest.approx(rev, abs=0.01)
        tods_hours = row[["pull_out_hours", "pull_in_hours", "deadhead_hours"]].sum()
        assert tods_hours == pytest.approx(non, abs=0.02)
        # Mock blocks are complete: no gaps without a movement, no overlaps.
        assert row["unresolved_gap_hours"] == 0 and row["overlaps"] == 0
        assert row["blocks_without_tods"] == 0


def test_day_types_are_classified_from_the_calendar(mock_tables: dict[str, pd.DataFrame]) -> None:
    day_types = mock_tables["day_type"].set_index("day_type")
    assert list(day_types.index) == ["Weekday", "Saturday", "Sunday", "Holiday"]
    assert day_types.loc["Weekday", "service_ids"] == "weekday"
    assert day_types.loc["Holiday", "service_ids"] == "holiday"
    # Only the weekday express (route 20) interlines with deadheads.
    assert day_types.loc["Weekday", "deadhead_hours"] > 0
    assert day_types.loc["Saturday", "deadhead_hours"] == 0


def test_gtfs_calendar_periods_follow_the_feed(mock_tables: dict[str, pd.DataFrame]) -> None:
    services = mock_tables["service"].set_index("service_id")
    periods = mock_tables["period"].set_index("period")
    nov = periods.loc["Nov"]
    # calendar_dates swaps Veterans Day (Wed 11/11) and Thanksgiving (Thu 11/26)
    # from weekday to holiday service.
    assert (nov["weekdays"], nov["saturdays"], nov["sundays"], nov["holidays"]) == (19, 4, 5, 2)
    expected = (
        19 * services.loc["weekday", "vehicle_hours"]
        + 4 * services.loc["saturday", "vehicle_hours"]
        + 5 * services.loc["sunday", "vehicle_hours"]
        + 2 * services.loc["holiday", "vehicle_hours"]
    )
    assert nov["vehicle_hours"] == pytest.approx(expected, abs=0.1)
    # 2026-01-01 runs holiday service only (calendar_dates removes weekday service).
    jan = periods.loc["Jan"]
    assert (jan["weekdays"], jan["holidays"]) == (21, 1)


def test_config_calendar_uses_federal_holidays(tmp_path: Path) -> None:
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=TODS_ZIP,
        output_dir=tmp_path,
        periods=[("Nov", "20261101", "20261130")],
        period_calendar="config",
        holiday_service_day_type="Sunday",
    )
    tables = target.run(cfg)
    services = tables["service"].set_index("service_id")
    nov = tables["period"].iloc[0]
    # Veterans Day (Wed 11/11) and Thanksgiving (Thu 11/26) run the Sunday schedule.
    assert (nov["weekdays"], nov["holidays"]) == (19, 2)
    expected = (
        19 * services.loc["weekday", "revenue_trip_hours"]
        + 7 * services.loc["sunday", "revenue_trip_hours"]
        + 4 * services.loc["saturday", "revenue_trip_hours"]
    )
    assert nov["revenue_trip_hours"] == pytest.approx(expected, abs=0.1)


# =============================================================================
# CLI
# =============================================================================


def test_main_returns_2_for_placeholders() -> None:
    assert target.main([]) == 2


def test_main_writes_outputs_and_run_log(tmp_path: Path) -> None:
    argv = ["--gtfs", str(GTFS_ZIP), "--tods", str(TODS_ZIP), "--output-dir", str(tmp_path)]
    assert target.main([*argv, "--period", "Nov", "20261101", "20261130"]) == 0
    for name in (
        target.BLOCK_FILENAME,
        target.SERVICE_FILENAME,
        target.DAY_TYPE_FILENAME,
        target.PERIOD_FILENAME,
    ):
        assert (tmp_path / name).is_file()
    log = (tmp_path / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "LAYOVER_IN_REVENUE_HOURS: bool =" in log  # Verbatim CONFIGURATION block.


def test_main_without_tods_leaves_nonrevenue_blank(tmp_path: Path) -> None:
    argv = ["--gtfs", str(GTFS_ZIP), "--no-tods", "--output-dir", str(tmp_path)]
    assert target.main([*argv, "--period", "Nov", "20261101", "20261130"]) == 0
    period = pd.read_csv(tmp_path / target.PERIOD_FILENAME).iloc[0]
    assert period["revenue_trip_hours"] > 0
    assert pd.isna(period["deadhead_hours"]) and pd.isna(period["vehicle_hours"])


# =============================================================================
# FORMULA WORKBOOK
# =============================================================================

SOFFICE = shutil.which("soffice")
needs_libreoffice = pytest.mark.skipif(SOFFICE is None, reason="LibreOffice not installed")


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, pd.DataFrame]]:
    out = tmp_path_factory.mktemp("workbook")
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=TODS_ZIP,
        output_dir=out,
        periods=[("November 2026", "20261101", "20261130"), ("Jan 2026", "20260101", "20260131")],
    )
    return out, target.run(cfg)


def recalculate(path: Path, work: Path) -> Any:
    """Recalculate *path* in LibreOffice and return it with formula results."""
    out = work / "recalculated"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(SOFFICE),
            "--headless",
            "--calc",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(out),
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=180,
        env={**os.environ, "HOME": str(work / "lo-profile")},
    )
    return load_workbook(out / path.name, data_only=True)


def sheet_rows(ws: Any) -> list[dict[str, Any]]:
    header = [c.value for c in ws[1]]
    return [dict(zip(header, [c.value for c in row])) for row in ws.iter_rows(min_row=2)]


def period_totals(wb: Any) -> dict[str, dict[str, Any]]:
    return {r["Period"]: r for r in sheet_rows(wb["Periods"]) if r["Service"] == "All services"}


def test_workbook_formulas_trace_back_to_trip_times(mock_run: tuple) -> None:
    out, _ = mock_run
    wb = load_workbook(out / target.WORKBOOK_FILENAME)
    assert wb.sheetnames == [
        "Periods",
        "Settings",
        "Period days",
        "Calendar",
        "Services",
        "Blocks",
        "Trips",
    ]
    assert wb.calculation.fullCalcOnLoad
    assert wb["Settings"]["B2"].value is True  # LAYOVER_IN_REVENUE_HOURS default.
    periods = sheet_rows(wb["Periods"])
    service_row = next(r for r in periods if r["Service"] == "weekday")
    assert service_row["Service days"].startswith("=SUMIFS(Calendar!")
    assert (
        service_row["Revenue trip h"].startswith("=$E")
        and "Services!" in service_row["Revenue trip h"]
    )
    assert period_totals(wb)["November 2026"]["Vehicle h"].startswith("=SUM(")
    block = sheet_rows(wb["Blocks"])[0]
    assert block["Revenue trip h"].startswith("=SUMIFS(Trips!$K$")
    trip = sheet_rows(wb["Trips"])[1]
    assert trip["Duration h"] == "=(J3-I3)*24"
    assert wb["Trips"]["I2"].number_format == "[h]:mm:ss"  # GTFS times past 24:00.


@needs_libreoffice
def test_workbook_totals_match_the_csv(mock_run: tuple, tmp_path: Path) -> None:
    out, tables = mock_run
    wb = recalculate(out / target.WORKBOOK_FILENAME, tmp_path)
    labels = {**target.COMPONENT_LABELS, **target.DERIVED_LABELS}
    csv_periods = tables["period"].set_index("period")
    for name, row in period_totals(wb).items():
        for column, label in labels.items():
            assert row[label] == pytest.approx(csv_periods.loc[name, column], abs=0.01), label
    days = {r["Period"]: r for r in sheet_rows(wb["Period days"])}
    assert (days["Jan 2026"]["Weekdays"], days["Jan 2026"]["Holidays"]) == (21, 1)
    errors = [
        (ws.title, c.coordinate)
        for ws in wb
        for row in ws.iter_rows()
        for c in row
        if isinstance(c.value, str) and c.value.startswith("#")
    ]
    assert not errors


@needs_libreoffice
def test_workbook_layover_switch_and_calendar_edits_recalculate(
    mock_run: tuple, tmp_path: Path
) -> None:
    out, tables = mock_run
    edited = tmp_path / "edited.xlsx"
    wb = load_workbook(out / target.WORKBOOK_FILENAME)
    wb["Settings"]["B2"] = False
    # Drop weekday service on the first November weekday (2026-11-02).
    cal = wb["Calendar"]
    header = [c.value for c in cal[1]]
    weekday_col = header.index("weekday") + 1
    row = next(
        r
        for r in range(2, cal.max_row + 1)
        if cal.cell(r, 1).value == "November 2026" and cal.cell(r, 2).value.day == 2
    )
    cal.cell(row, weekday_col).value = 0
    wb.save(edited)

    totals = period_totals(recalculate(edited, tmp_path))["November 2026"]
    weekday = tables["service"].set_index("service_id").loc["weekday"]
    nov = tables["period"].set_index("period").loc["November 2026"]
    assert totals["Revenue trip h"] == pytest.approx(
        nov["revenue_trip_hours"] - weekday["revenue_trip_hours"], abs=0.01
    )
    # Layover now counts as nonrevenue time.
    assert totals["Vehicle revenue h"] == pytest.approx(totals["Revenue trip h"], abs=1e-6)
    moves = sum(totals[target.COMPONENT_LABELS[c]] for c in target.TODS_COMPONENTS)
    assert totals["Nonrevenue h"] == pytest.approx(moves + totals["Layover h"], abs=1e-6)


def test_workbook_without_tods_marks_unknowns(tmp_path: Path) -> None:
    cfg = target.Config(
        gtfs_path=GTFS_ZIP,
        tods_path=None,
        output_dir=tmp_path,
        periods=[("Nov", "20261101", "20261130")],
    )
    target.run(cfg)
    wb = load_workbook(tmp_path / target.WORKBOOK_FILENAME)
    total = period_totals(wb)["Nov"]
    assert total["Deadhead h"] == total["Vehicle h"] == "n/a"
    assert total["Revenue trip h"].startswith("=SUM(")
