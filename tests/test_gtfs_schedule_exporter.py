"""Behavioral regression tests for the four consolidated exporters."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import pytest
from openpyxl import Workbook, load_workbook

import scripts.gtfs_exports.gtfs_schedule_exporter as g


@dataclass(frozen=True)
class Spec:
    """Small synthetic trip with independently chosen times and distances."""

    trip_id: str = "001"
    stops: tuple[str, ...] = ("A", "B", "C")
    times: tuple[str, ...] = ("06:00:00", "06:06:00", "06:09:00")
    distances: tuple[str, ...] = ("0", "1609.344", "3218.688")
    route_id: str = "r1"
    direction: str = "0"


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a fixture without pandas' missing-value or identifier conversions."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(tmp_path: Path, specs: tuple[Spec, ...] = (Spec(),)) -> g.Config:
    folder = tmp_path / "input"
    folder.mkdir()
    write_table(
        folder / "routes.txt",
        [
            {"route_id": rid, "route_short_name": "101"}
            for rid in sorted({s.route_id for s in specs})
        ],
    )
    write_table(
        folder / "stops.txt",
        [
            {"stop_id": sid, "stop_code": f"0{sid}", "stop_name": f"Stop {sid}"}
            for sid in sorted({sid for spec in specs for sid in spec.stops})
        ],
    )
    write_table(
        folder / "trips.txt",
        [
            {
                "trip_id": spec.trip_id,
                "route_id": spec.route_id,
                "service_id": "svc",
                "direction_id": spec.direction,
                "block_id": "00100",
            }
            for spec in specs
        ],
    )
    write_table(
        folder / "stop_times.txt",
        [
            {
                "trip_id": spec.trip_id,
                "stop_id": sid,
                "stop_sequence": str(i + 1),
                "arrival_time": time,
                "departure_time": time,
                "shape_dist_traveled": dist,
                "timepoint": "1",
            }
            for spec in specs
            for i, (sid, time, dist) in enumerate(zip(spec.stops, spec.times, spec.distances))
        ],
    )
    write_table(
        folder / "calendar.txt",
        [{"service_id": "svc", **{day: "1" if i < 5 else "0" for i, day in enumerate(g.WEEKDAYS)}}],
    )
    return g.Config(folder, tmp_path / "output", run_subfolder=False)


def calculate(config: g.Config) -> tuple[g.Feed, list[g.Trip], g.Audit]:
    audit = g.Audit()
    feed = g.load_feed(config, audit)
    trips = [
        g.build_trip(meta, feed.stop_times.get(tid, []), feed, config, audit)
        for tid, meta in feed.trips.items()
    ]
    return feed, trips, audit


def test_optional_files_and_fields(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    for name, column in [
        ("trips", "direction_id"),
        ("stops", "stop_code"),
        ("stop_times", "timepoint"),
    ]:
        path = config.gtfs_path / f"{name}.txt"
        rows = read_table(path)
        for row in rows:
            del row[column]
        write_table(path, rows)
    _, trips, _ = calculate(config)
    assert len(trips) == 1 and trips[0].direction == ""
    assert g.main(config) == 0
    book = load_workbook(next(config.output_dir.rglob("*.xlsx")))
    assert any(name.startswith("DX_Timetable") for name in book.sheetnames)
    assert book["Trips"].max_row == 2


def test_literal_identifiers(tmp_path: Path) -> None:
    config = prepare(tmp_path, (Spec(stops=("NA", "NULL", "00123")),))
    _, trips, _ = calculate(config)
    assert trips[0].pattern == ("NA", "NULL", "00123")
    assert trips[0].trip_id == "001" and trips[0].block_id == "00100"


@pytest.mark.parametrize("flag", ["", "1"])
def test_all_blank_timepoints_equal_all_exact(tmp_path: Path, flag: str) -> None:
    config = prepare(tmp_path)
    path = config.gtfs_path / "stop_times.txt"
    rows = read_table(path)
    for row in rows:
        row["timepoint"] = flag
    write_table(path, rows)
    _, trips, _ = calculate(config)
    assert trips[0].pattern == ("A", "B", "C")


def test_partial_blank_timepoints_retained_with_warning(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    path = config.gtfs_path / "stop_times.txt"
    rows = read_table(path)
    rows[1]["timepoint"] = ""
    write_table(path, rows)
    _, trips, audit = calculate(config)
    assert trips[0].pattern == ("A", "B", "C")
    assert audit.counts["PARTIAL_TIMEPOINTS"] == 1


def test_arrival_fallback_does_not_drop_trip_or_invent_holding(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    path = config.gtfs_path / "stop_times.txt"
    rows = read_table(path)
    rows[0]["departure_time"] = ""
    write_table(path, rows)
    _, trips, audit = calculate(config)
    assert trips[0].start == 21600
    assert trips[0].running == (None, 360, 180)
    assert trips[0].holding[0] is None
    assert audit.counts["TIME_FALLBACK"] == 1


def test_missing_time_keeps_stop_in_pattern(tmp_path: Path) -> None:
    config = prepare(tmp_path, (Spec(times=("06:00:00", "", "06:09:00")),))
    _, trips, _ = calculate(config)
    assert trips[0].pattern == ("A", "B", "C")
    assert trips[0].running == (None, None, None)
    assert g.mean_speed(trips[0], config) is None


def test_seconds_and_midnight_preserved(tmp_path: Path) -> None:
    config = prepare(
        tmp_path,
        (Spec(stops=("A", "B"), times=("25:00:31", "25:01:29"), distances=("0", "160.9344")),),
    )
    _, trips, _ = calculate(config)
    assert trips[0].running == (None, 58)
    assert g.mean_speed(trips[0], config) == pytest.approx(0.1 * 3600 / 58)
    assert g.format_time(trips[0].start, config) == "25:01"
    assert g.format_time(trips[0].start, replace(config, time_format="12")) == "1:01 AM (+1d)"
    assert g.parse_seconds("06:60:00") is None
    assert g.parse_seconds("06:00:60") is None


def test_holding_and_runtime_conventions(tmp_path: Path) -> None:
    config = prepare(tmp_path, (Spec(times=("06:00:00", "06:06:00", "06:14:00")),))
    path = config.gtfs_path / "stop_times.txt"
    rows = read_table(path)
    rows[1]["departure_time"] = "06:11:00"
    rows[2]["departure_time"] = "06:20:00"
    write_table(path, rows)
    _, trips, _ = calculate(config)
    trip = trips[0]
    assert trip.running == (None, 360, 180)
    assert trip.elapsed == (None, 660, 540)
    assert trip.holding == (0, 300, 360)
    assert trip.start is not None and trip.end is not None
    assert trip.end - trip.start == 840
    assert trip.runtimes(replace(config, runtime_basis="departure_to_departure")) == trip.elapsed
    assert g.main(config) == 0
    book = load_workbook(next(config.output_dir.rglob("*.xlsx")))
    table = next(book[name] for name in book.sheetnames if "Timetable" in name)
    assert table.cell(2, table.max_column).value == "06:14"


def test_identical_segment_speeds_have_independent_mean_speeds(tmp_path: Path) -> None:
    config = prepare(
        tmp_path,
        (
            Spec(),
            Spec(
                "002",
                ("A", "X", "C"),
                ("07:00:00", "07:12:00", "07:15:00"),
                ("0", "3218.688", "4828.032"),
            ),
        ),
    )
    _, trips, _ = calculate(config)
    assert [g.mean_speed(t, config) for t in trips] == pytest.approx([40 / 3, 12])


@pytest.mark.parametrize("distances", [("", "", ""), ("0", "", "3218.688")])
def test_missing_distance_never_becomes_zero_or_bridges_a_retained_stop(
    tmp_path: Path, distances: tuple[str, ...]
) -> None:
    config = prepare(tmp_path, (Spec(distances=distances),))
    _, trips, _ = calculate(config)
    assert trips[0].distances == (None, None, None)
    assert g.complete_sum(trips[0].distances[1:]) is None
    assert g.mean_speed(trips[0], config) is None


def test_filtered_distance_and_original_endpoints(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    path = config.gtfs_path / "stop_times.txt"
    rows = read_table(path)
    rows[0]["timepoint"] = rows[2]["timepoint"] = "0"
    write_table(path, rows)
    _, trips, audit = calculate(config)
    assert len(trips[0].events) == 3 and trips[0].full_distance == pytest.approx(3218.688)
    assert not audit.counts["PARTIAL_TRIP_COVERAGE"]
    _, trips, audit = calculate(replace(config, keep_endpoints=False))
    assert len(trips[0].events) == 1
    assert trips[0].start == 21600  # True origin, not the first retained timepoint.
    assert audit.counts["PARTIAL_TRIP_COVERAGE"] == 1


@pytest.mark.parametrize("kind", ["time", "distance"])
def test_reversed_inputs_are_flagged_and_metrics_invalidated(tmp_path: Path, kind: str) -> None:
    spec = (
        Spec(times=("06:00:00", "05:59:00", "06:09:00"))
        if kind == "time"
        else Spec(distances=("0", "2000", "1000"))
    )
    config = prepare(tmp_path, (spec,))
    _, trips, audit = calculate(config)
    assert g.mean_speed(trips[0], config) is None
    assert audit.counts["TIME_REVERSAL" if kind == "time" else "DISTANCE_REVERSAL"] == 1


def test_zero_runtime_has_no_speed(tmp_path: Path) -> None:
    config = prepare(tmp_path, (Spec(times=("06:00:00", "06:00:00", "06:09:00")),))
    _, trips, audit = calculate(config)
    assert g.mean_speed(trips[0], config) is None
    assert g.speed_mph(1, -1, config) is None
    assert audit.counts["ZERO_RUNTIME"] == 1


def test_nonconsecutive_profiles_split_bands(tmp_path: Path) -> None:
    config = prepare(
        tmp_path,
        (
            Spec(),
            Spec("002", times=("09:00:00", "09:12:00", "09:18:00")),
            Spec("003", times=("16:00:00", "16:06:00", "16:09:00")),
        ),
    )
    _, trips, _ = calculate(config)
    assert [len(b) for b in g.make_bands(trips, replace(config, max_band_gap=None))] == [1, 1, 1]
    assert [len(b) for b in g.make_bands(trips, replace(config, band_mode="profile_summary"))] == [
        2,
        1,
    ]


def test_gap_split_is_configurable(tmp_path: Path) -> None:
    config = prepare(tmp_path, (Spec(), Spec("002", times=("16:00:00", "16:06:00", "16:09:00"))))
    _, trips, _ = calculate(config)
    assert len(g.make_bands(trips, config)) == 2
    assert len(g.make_bands(trips, replace(config, max_band_gap=None))) == 1


def test_branches_share_timetable_but_keep_correct_metric_headers(tmp_path: Path) -> None:
    config = prepare(
        tmp_path,
        (Spec(), Spec("002", stops=("A", "X", "C"), times=("07:00:00", "07:06:00", "07:09:00"))),
    )
    assert g.main(config) == 0
    book = load_workbook(next(config.output_dir.rglob("*.xlsx")))
    tables = [book[n] for n in book.sheetnames if "Timetable" in n]
    assert len(tables) == 1 and tables[0].max_column == 9
    metric_tables = [book[n] for n in book.sheetnames if n.endswith("_Run")]
    assert len(metric_tables) == 2
    headers = [[str(c.value) for c in s[1]] for s in metric_tables]
    assert any(
        any("Stop X" in h for h in header) and not any("Stop B" in h for h in header)
        for header in headers
    )
    for sheet in metric_tables:
        assert sheet.max_row == 2
    assert len(read_table(config.output_dir / "trips.csv")) == 2
    assert len(read_table(config.output_dir / "segments.csv")) == 4


def test_shared_stop_sort_handles_short_turns() -> None:
    order, consistent = g.chronological_order([[100, 300, 500], [None, 200, 400]])
    assert consistent and order == [1, 0]
    order, consistent = g.chronological_order([[100, 400], [200, 300]])
    assert not consistent and order == [0, 1]


def test_overtaking_is_retained_and_flagged(tmp_path: Path) -> None:
    config = prepare(
        tmp_path,
        (
            Spec(times=("06:00:00", "06:10:00", "06:20:00")),
            Spec("002", times=("06:05:00", "06:08:00", "06:15:00")),
        ),
    )
    assert g.main(config) == 0
    assert any(
        r["Code"] == "OVERTAKING_OR_ORDER_CONFLICT"
        for r in read_table(config.output_dir / "issues.csv")
    )
    assert len(read_table(config.output_dir / "trips.csv")) == 2


def test_loops_preserve_visits_and_do_not_guess_alignment(tmp_path: Path) -> None:
    config = prepare(
        tmp_path,
        (
            Spec(stops=("A", "B", "A")),
            Spec("002", stops=("B", "A"), times=("07:00:00", "07:03:00"), distances=("0", "1000")),
        ),
    )
    feed, trips, _ = calculate(config)
    layouts, _ = g.build_layouts(trips)
    assert len(layouts) == 2
    assert g.embed_pattern(("A", "B", "A"), ("A", "B", "A")) == [0, 1, 2]
    headers = g.stop_headers(("A", "B", "A"), feed.stops, config)
    assert len(set(headers)) == 3 and "visit 2" in headers[2]


def test_duplicate_short_names_do_not_overwrite_workbooks(tmp_path: Path) -> None:
    config = prepare(tmp_path, (Spec(route_id="r/1"), Spec("002", route_id="r:1")))
    assert g.main(config) == 0
    files = list(config.output_dir.rglob("*.xlsx"))
    assert len(files) == 2 and files[0].name != files[1].name


def test_zip_wrapper_and_ambiguous_members(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    zipped = tmp_path / "feed.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        for path in config.gtfs_path.iterdir():
            archive.write(path, "wrapper/" + path.name)
    assert len(calculate(replace(config, gtfs_path=zipped))[1]) == 1
    with zipfile.ZipFile(zipped, "a") as archive:
        archive.write(config.gtfs_path / "trips.txt", "another/trips.txt")
    with pytest.raises(ValueError, match="Ambiguous"):
        calculate(replace(config, gtfs_path=zipped))


def test_frequency_templates_are_never_counted_as_departures(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    write_table(config.gtfs_path / "frequencies.txt", [{"trip_id": "001"}])
    with pytest.raises(ValueError, match="frequencies"):
        calculate(config)
    feed, trips, audit = calculate(replace(config, frequency_policy="skip"))
    assert trips == [] and feed.frequency_skipped == 1
    assert audit.counts["FREQUENCY_SKIPPED"] == 1


def test_calendar_dates_only_and_unknown_service_still_export(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    (config.gtfs_path / "calendar.txt").unlink()
    write_table(
        config.gtfs_path / "calendar_dates.txt",
        [{"service_id": "svc", "date": "20261116", "exception_type": "1"}],
    )
    feed, _, _ = calculate(config)
    assert feed.service_labels["svc"] == "Dates only (mon)"
    (config.gtfs_path / "calendar_dates.txt").unlink()
    feed, trips, audit = calculate(config)
    assert len(trips) == 1 and audit.counts["UNDEFINED_SERVICE"] == 1


def test_bad_trip_reconciles_in_run_log_and_returns_failure(tmp_path: Path) -> None:
    config = prepare(tmp_path, (Spec(), Spec("002")))
    path = config.gtfs_path / "stop_times.txt"
    rows = read_table(path)
    rows[-1]["stop_sequence"] = "2"
    write_table(path, rows)
    assert g.main(config) == 1
    log = (config.output_dir / "gtfs_schedule_exporter_runlog.txt").read_text()
    assert '"valid_trips": 1' in log and '"rejected_trips": 1' in log
    assert len(read_table(config.output_dir / "trips.csv")) == 1


def test_save_failure_is_not_success_and_preserves_existing_file(tmp_path: Path) -> None:
    config = prepare(tmp_path)
    assert g.main(config) == 0
    workbook_path = next(config.output_dir.rglob("*.xlsx"))
    original = workbook_path.read_bytes()
    with patch.object(g.Workbook, "save", side_effect=PermissionError("Workbook open")):
        assert g.main(config) == 1
    assert workbook_path.read_bytes() == original
    assert any(r["Code"] == "EXPORT_FAILED" for r in read_table(config.output_dir / "issues.csv"))


def test_excel_ids_are_literal_text() -> None:
    book = Workbook()
    sheet = g.new_sheet(book, "Test", ["ID", "Leading zero", "Number"])
    g.append_row(sheet, ["=1+1", "001", 2.0])
    assert sheet["A2"].data_type == "s" and sheet["A2"].value == "=1+1"
    assert sheet["B2"].data_type == "s" and sheet["B2"].value == "001"
    assert sheet["C2"].data_type == "n"


def test_export_switches_and_consistent_pattern_ids(tmp_path: Path) -> None:
    config = replace(prepare(tmp_path), distances=False, runtimes=False, speeds=False)
    assert g.main(config) == 0
    book = load_workbook(next(config.output_dir.rglob("*.xlsx")))
    assert not any(n.endswith(("_Dist", "_Run", "_Speed")) for n in book.sheetnames)
    assert g.stable_id(("A", "B")) == g.stable_id(["A", "B"])
    assert g.stable_id(("A,B",)) != g.stable_id(("A", "B"))
    # Effective settings can always be recovered from the JSON block in the log.
    log = (config.output_dir / "gtfs_schedule_exporter_runlog.txt").read_text()
    settings = log.split("EFFECTIVE CONFIGURATION\n", 1)[1].split("\n\nISSUE COUNTS", 1)[0]
    assert json.loads(settings)["speeds"] is False
