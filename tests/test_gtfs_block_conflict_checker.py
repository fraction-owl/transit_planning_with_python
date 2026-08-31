"""Tests for gtfs_block_conflict_checker: same-stop same-time block conflict QA."""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/gtfs_data_quality").resolve()
sys.path.append(str(script_dir))

import gtfs_block_conflict_checker as target  # noqa: E402


def _write_feed(root: Path, *, with_calendar: bool = True) -> Path:
    """Write a small synthetic GTFS folder and return its path.

    Calendar (Jun 1 – Oct 31, 2026): WKDY runs every weekday, SCHOOL runs
    weekdays from Sep 1, SAT runs Saturdays. So Jun–Aug weekdays are
    {WKDY} (66 dates), Sep–Oct weekdays are {WKDY, SCHOOL} (44 dates), and
    Saturdays are {SAT} (22 dates).

    Built-in situations:
      * BLK1/BLK2 (WKDY) dwell together at C1 10:03–10:05 (3 shared min).
      * BLK3 (SCHOOL) touches C1 at 10:04 — conflicts with BLK1/BLK2 only
        when co-active services are combined.
      * BLK6/BLK7 (WKDY) hand off at S1 at exactly 09:00 (1 shared min).
      * BLK8 (WKDY) has trips T8a/T8b whose spans overlap by 11 minutes —
        a block self-overlap, with no same-stop pair conflict.
      * T9/T10 (WKDY) have no block_id and overlap at S2 11:01–11:02.
      * BLK11/BLK12 (WKDY) sit at sibling bays B1/B2 of station P1 —
        a conflict only when grouping by parent_station.
      * BLK13/BLK14 (WKDY) overlap at S1 past midnight (24:33–24:35).
      * BLK4/BLK5 (SAT) never overlap — Saturday stays clean.
    """
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "C1", "B1", "B2", "P1"],
            "stop_name": [
                "First & Main",
                "Elm & Oak",
                "Metro Center",
                "Central Station Bay 1",
                "Central Station Bay 2",
                "Central Station",
            ],
            "stop_lat": [38.90, 38.91, 38.92, 38.93, 38.93, 38.93],
            "stop_lon": [-77.03, -77.03, -77.04, -77.05, -77.05, -77.05],
            "location_type": ["0", "0", "0", "0", "0", "1"],
            "parent_station": ["", "", "", "P1", "P1", ""],
        }
    ).to_csv(root / "stops.txt", index=False)
    pd.DataFrame(
        {
            "route_id": ["R1", "R2"],
            "route_short_name": ["10", "20"],
            "route_long_name": ["Main Street", "Elm Street"],
            "route_type": [3, 3],
        }
    ).to_csv(root / "routes.txt", index=False)
    if with_calendar:
        pd.DataFrame(
            {
                "service_id": ["WKDY", "SCHOOL", "SAT"],
                "monday": [1, 1, 0],
                "tuesday": [1, 1, 0],
                "wednesday": [1, 1, 0],
                "thursday": [1, 1, 0],
                "friday": [1, 1, 0],
                "saturday": [0, 0, 1],
                "sunday": [0, 0, 0],
                "start_date": [20260601, 20260901, 20260601],
                "end_date": [20261031, 20261031, 20261031],
            }
        ).to_csv(root / "calendar.txt", index=False)
    pd.DataFrame(
        {
            "trip_id": [
                "T1",
                "T2",
                "T3",
                "T4",
                "T5",
                "T6",
                "T7",
                "T8a",
                "T8b",
                "T9",
                "T10",
                "T11",
                "T12",
                "T13",
                "T14",
            ],
            "route_id": [
                "R1",
                "R1",
                "R2",
                "R1",
                "R1",
                "R1",
                "R2",
                "R1",
                "R1",
                "R1",
                "R1",
                "R1",
                "R2",
                "R1",
                "R2",
            ],
            "service_id": [
                "WKDY",
                "WKDY",
                "SCHOOL",
                "SAT",
                "SAT",
                "WKDY",
                "WKDY",
                "WKDY",
                "WKDY",
                "WKDY",
                "WKDY",
                "WKDY",
                "WKDY",
                "WKDY",
                "WKDY",
            ],
            "block_id": [
                "BLK1",
                "BLK2",
                "BLK3",
                "BLK4",
                "BLK5",
                "BLK6",
                "BLK7",
                "BLK8",
                "BLK8",
                "",
                "",
                "BLK11",
                "BLK12",
                "BLK13",
                "BLK14",
            ],
        }
    ).to_csv(root / "trips.txt", index=False)
    rows = [
        ("T1", "S1", 1, "09:50:00", "09:50:00"),
        ("T1", "C1", 2, "10:00:00", "10:05:00"),
        ("T2", "S1", 1, "09:53:00", "09:53:00"),
        ("T2", "C1", 2, "10:03:00", "10:08:00"),
        ("T3", "S2", 1, "09:55:00", "09:55:00"),
        ("T3", "C1", 2, "10:04:00", "10:04:00"),
        ("T4", "S1", 1, "09:50:00", "09:50:00"),
        ("T4", "C1", 2, "10:00:00", "10:01:00"),
        ("T5", "S1", 1, "10:20:00", "10:20:00"),
        ("T5", "C1", 2, "10:30:00", "10:31:00"),
        ("T6", "S1", 1, "09:00:00", "09:00:00"),
        ("T6", "C1", 2, "09:10:00", "09:10:00"),
        ("T7", "S2", 1, "08:50:00", "08:50:00"),
        ("T7", "S1", 2, "09:00:00", "09:00:00"),
        ("T8a", "S2", 1, "13:00:00", "13:00:00"),
        ("T8a", "S1", 2, "13:30:00", "13:30:00"),
        ("T8b", "S2", 1, "13:20:00", "13:20:00"),
        ("T8b", "S1", 2, "13:50:00", "13:50:00"),
        ("T9", "S1", 1, "10:50:00", "10:50:00"),
        ("T9", "S2", 2, "11:00:00", "11:02:00"),
        ("T10", "S1", 1, "10:52:00", "10:52:00"),
        ("T10", "S2", 2, "11:01:00", "11:03:00"),
        ("T11", "S1", 1, "11:55:00", "11:55:00"),
        ("T11", "B1", 2, "12:00:00", "12:05:00"),
        ("T12", "S2", 1, "11:57:00", "11:57:00"),
        ("T12", "B2", 2, "12:02:00", "12:06:00"),
        ("T13", "S2", 1, "24:25:00", "24:25:00"),
        ("T13", "S1", 2, "24:30:00", "24:35:00"),
        ("T14", "C1", 1, "24:20:00", "24:20:00"),
        ("T14", "S1", 2, "24:33:00", "24:37:00"),
    ]
    pd.DataFrame(
        rows, columns=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]
    ).to_csv(root / "stop_times.txt", index=False)
    return root


def _pair(detail: pd.DataFrame, block_a: str, block_b: str) -> pd.DataFrame:
    return detail.loc[(detail["block_a"] == block_a) & (detail["block_b"] == block_b)]


def test_label_service_group() -> None:
    monday = dt.date(2026, 6, 1)
    assert target.label_service_group({monday, dt.date(2026, 6, 3)}) == "Weekday"
    assert target.label_service_group({dt.date(2026, 6, 6)}) == "Saturday"
    assert target.label_service_group({dt.date(2026, 6, 7)}) == "Sunday"
    assert target.label_service_group({dt.date(2026, 6, 6), dt.date(2026, 6, 7)}) == "Weekend"
    assert target.label_service_group({monday + dt.timedelta(days=n) for n in range(7)}) == "Daily"
    assert target.label_service_group({monday, dt.date(2026, 6, 6)}) == "Mixed"
    assert target.label_service_group(set()) == "Empty"


def test_build_service_groups_combines_co_active_services(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    data = target.load_gtfs_data(str(feed), files=("trips.txt", "calendar.txt"))
    groups = target.build_service_groups(data["trips"], data["calendar"], None)

    described = [(g["label"], set(g["service_ids"]), g["n_dates"]) for g in groups]
    assert described == [
        ("Weekday", {"WKDY"}, 66),  # Jun–Aug weekdays, before SCHOOL starts
        ("Saturday", {"SAT"}, 22),
        ("Weekday", {"WKDY", "SCHOOL"}, 44),  # Sep–Oct weekdays
    ]


def test_build_service_groups_explicit_ids_form_one_schedule(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    data = target.load_gtfs_data(str(feed), files=("trips.txt", "calendar.txt"))
    groups = target.build_service_groups(
        data["trips"], data["calendar"], None, explicit_service_ids=["WKDY", "SCHOOL"]
    )
    assert len(groups) == 1
    assert groups[0]["service_ids"] == frozenset({"WKDY", "SCHOOL"})
    assert groups[0]["n_dates"] is None


def test_merge_presence_intervals_joins_touching_same_block_visits() -> None:
    events = pd.DataFrame(
        [
            # Block ends trip T-in at the stop 10:00, next trip T-out departs 10:00.
            ("T-in", "BLK9", True, "WKDY", "10", "S9", "S9", "Ninth & Oak", 600, 600),
            ("T-out", "BLK9", True, "WKDY", "10", "S9", "S9", "Ninth & Oak", 600, 600),
            ("T-late", "BLK9", True, "WKDY", "10", "S9", "S9", "Ninth & Oak", 700, 702),
        ],
        columns=target.EVENT_COLUMNS,
    )
    intervals = target.merge_presence_intervals(events)
    assert len(intervals) == 2  # touching visits merge; the 700 visit stays separate
    merged = intervals.iloc[0]
    assert (merged["arr_min"], merged["dep_min"]) == (600, 600)
    assert merged["trip_ids"] == "T-in+T-out"

    pairs, _stats = target.find_conflict_pairs(intervals, 1)
    assert pairs.empty  # one vehicle never conflicts with itself


def test_find_conflict_pairs_handoff_counts_one_shared_minute() -> None:
    intervals = pd.DataFrame(
        [
            ("S9", "Ninth & Oak", "BLK-A", 600, 605, "TA", "S9", "10", "WKDY"),
            ("S9", "Ninth & Oak", "BLK-B", 605, 610, "TB", "S9", "20", "WKDY"),
        ],
        columns=target.INTERVAL_COLUMNS,
    )
    flagged, stats = target.find_conflict_pairs(intervals, 1)
    assert len(flagged) == 1
    assert flagged.iloc[0]["overlap_minutes"] == 1
    assert stats.iloc[0]["max_simultaneous"] == 2

    strict, _stats = target.find_conflict_pairs(intervals, 2)
    assert strict.empty


def test_run_end_to_end_default(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    out_dir = tmp_path / "out"
    detail = target.run(str(feed), out_dir)

    assert len(detail) == 6
    assert not detail["schedule"].str.contains("Saturday").any()

    main_pair = _pair(detail, "BLK1", "BLK2").iloc[0]
    assert main_pair["place_id"] == "C1"
    assert main_pair["schedule"] == "Weekday"  # found in both weekday schedules, collapsed
    assert main_pair["service_group"] == "WKDY; SCHOOL+WKDY"
    assert main_pair["n_dates"] == 110  # 66 pre-school + 44 school weekdays
    assert (main_pair["overlap_start"], main_pair["overlap_end"]) == ("10:03", "10:05")
    assert main_pair["overlap_minutes"] == 3
    assert (main_pair["routes_a"], main_pair["routes_b"]) == ("10", "10")

    school_pair = _pair(detail, "BLK1", "BLK3").iloc[0]
    assert school_pair["n_dates"] == 44  # school weekdays only
    assert school_pair["overlap_minutes"] == 1
    assert not _pair(detail, "BLK2", "BLK3").empty

    night_pair = _pair(detail, "BLK13", "BLK14").iloc[0]
    assert (night_pair["overlap_start"], night_pair["overlap_end"]) == ("24:33", "24:35")
    assert night_pair["overlap_minutes"] == 3

    summary = pd.read_csv(out_dir / target.SUMMARY_FILENAME)
    c1 = summary.loc[summary["place_id"] == "C1"].iloc[0]
    assert c1["n_conflict_pairs"] == 3
    assert c1["n_blocks"] == 3
    assert c1["max_simultaneous_buses"] == 3
    assert c1["peak_time"] == "10:04"

    self_overlaps = pd.read_csv(out_dir / target.SELF_OVERLAP_FILENAME)
    assert len(self_overlaps) == 1
    blk8 = self_overlaps.iloc[0]
    assert blk8["block_id"] == "BLK8"
    assert (blk8["trip_a"], blk8["trip_b"]) == ("T8a", "T8b")
    assert blk8["overlap_minutes"] == 11

    runlog = out_dir / f"{Path(target.DETAIL_FILENAME).stem}_runlog.txt"
    assert runlog.exists()
    text = runlog.read_text(encoding="utf-8")
    assert "MIN_OVERLAP_MINUTES" in text  # verbatim config block captured
    assert "Conflict pairs:       6" in text


def test_run_min_overlap_threshold_drops_handoffs(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    detail = target.run(str(feed), tmp_path / "out2", min_overlap_minutes=2)

    # The 09:00 hand-off and both 1-minute BLK3 touches fall below the bar.
    # Pairs order lexicographically by vehicle, so "trip:T10" precedes "trip:T9".
    kept = set(zip(detail["block_a"], detail["block_b"]))
    assert kept == {("BLK1", "BLK2"), ("BLK13", "BLK14"), ("trip:T10", "trip:T9")}


def test_run_without_combining_misses_cross_service_conflicts(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    detail = target.run(str(feed), tmp_path / "out3", combine_same_day_services=False)

    assert len(detail) == 4
    assert not detail["block_a"].str.contains("BLK3").any()
    assert not detail["block_b"].str.contains("BLK3").any()
    assert detail["n_dates"].isna().all()


def test_run_explicit_service_ids(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    detail = target.run(str(feed), tmp_path / "out4", service_ids=["WKDY", "SCHOOL"])

    assert set(detail["schedule"]) == {"Selected services"}
    assert len(detail) == 6  # same conflicts as the combined weekday schedule
    assert not _pair(detail, "BLK1", "BLK3").empty


def test_run_group_by_parent_station_finds_bay_siblings(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    base = target.run(str(feed), tmp_path / "out5a")
    assert _pair(base, "BLK11", "BLK12").empty  # different bays, no conflict by default

    detail = target.run(str(feed), tmp_path / "out5b", group_by_parent_station=True)
    station = _pair(detail, "BLK11", "BLK12").iloc[0]
    assert station["place_id"] == "P1"
    assert station["place_name"] == "Central Station"
    assert (station["stops_a"], station["stops_b"]) == ("B1", "B2")
    assert station["overlap_minutes"] == 4  # 12:02–12:05 inclusive


def test_run_blockless_trips_become_pseudo_vehicles(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    detail = target.run(str(feed), tmp_path / "out6")

    pair = _pair(detail, "trip:T10", "trip:T9").iloc[0]  # lexicographic vehicle order
    assert pair["place_id"] == "S2"
    assert pair["overlap_minutes"] == 2  # 11:01–11:02 inclusive


def test_run_self_overlap_toggle(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    target.run(str(feed), tmp_path / "out7", check_block_self_overlaps=False)
    self_overlaps = pd.read_csv(tmp_path / "out7" / target.SELF_OVERLAP_FILENAME)
    assert self_overlaps.empty


def test_run_without_calendar_checks_each_service_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    feed = _write_feed(tmp_path / "gtfs", with_calendar=False)
    with caplog.at_level("WARNING"):
        detail = target.run(str(feed), tmp_path / "out8")

    assert "cannot tell which service_ids run together" in caplog.text
    assert len(detail) == 4  # WKDY-internal conflicts only, labelled by raw id
    assert set(detail["schedule"]) == {"WKDY"}


def test_validate_required_columns_raises() -> None:
    data = {
        "stops": pd.DataFrame({"stop_id": []}),
        "trips": pd.DataFrame({"trip_id": [], "route_id": []}),  # service_id missing
    }
    with pytest.raises(ValueError, match="trips.txt is missing column"):
        target.validate_required_columns(data)


def test_main_placeholder_paths_return_2() -> None:
    assert target.main([]) == 2
