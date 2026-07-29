"""Tests for gtfs_timepoint_interval_flagger: scheduled timepoint-gap QA."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/gtfs_data_quality").resolve()
sys.path.append(str(script_dir))

import gtfs_timepoint_interval_flagger as target  # noqa: E402


def _write_feed(root: Path, *, with_timepoint_col: bool = True) -> Path:
    """Write a small synthetic GTFS folder and return its path.

    Trips (all on route R1 unless noted):
      * T1 (WKDY): timepoints S1 07:00 -> S3 07:12 (12 min gap; S2 is not a
        timepoint and must be skipped over).
      * T2 (SAT):  timepoints S1 08:00 -> S3 08:09 (9 min gap).
      * T3 (WKDY, route R2): timepoints S4 23:55 -> S5 24:07 (12 min gap
        crossing midnight, exercising GTFS times past 24:00).
    """
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3", "S4", "S5"],
            "stop_name": ["First & Main", "Mid & Main", "Last & Main", "Elm & Oak", "Elm & Pine"],
            "stop_lat": [38.90, 38.91, 38.92, 38.93, 38.94],
            "stop_lon": [-77.03, -77.03, -77.03, -77.04, -77.04],
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
    pd.DataFrame(
        {
            "service_id": ["WKDY", "SAT"],
            "monday": [1, 0],
            "tuesday": [1, 0],
            "wednesday": [1, 0],
            "thursday": [1, 0],
            "friday": [1, 0],
            "saturday": [0, 1],
            "sunday": [0, 0],
            "start_date": [20260101, 20260101],
            "end_date": [20261231, 20261231],
        }
    ).to_csv(root / "calendar.txt", index=False)
    pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1", "R1", "R2"],
            "service_id": ["WKDY", "SAT", "WKDY"],
            "direction_id": [0, 0, 1],
        }
    ).to_csv(root / "trips.txt", index=False)
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1", "T2", "T2", "T3", "T3"],
            "arrival_time": [
                "07:00:00",
                "07:04:00",
                "07:12:00",
                "08:00:00",
                "08:09:00",
                "23:55:00",
                "24:07:00",
            ],
            "departure_time": [
                "07:00:00",
                "07:04:00",
                "07:12:00",
                "08:00:00",
                "08:09:00",
                "23:55:00",
                "24:07:00",
            ],
            "stop_id": ["S1", "S2", "S3", "S1", "S2", "S4", "S5"],
            "stop_sequence": [1, 2, 3, 1, 2, 1, 2],
            "timepoint": ["1", "0", "1", "1", "1", "1", "1"],
        }
    )
    if not with_timepoint_col:
        stop_times = stop_times.drop(columns=["timepoint"])
    stop_times.to_csv(root / "stop_times.txt", index=False)
    return root


def test_select_timepoint_rows_uses_timepoint_column(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    data = target.load_gtfs_data(str(feed), files=("stop_times.txt",))
    rows = target.select_timepoint_rows(data["stop_times"])
    assert len(rows) == 6  # T1's S2 (timepoint=0) is excluded
    assert not ((rows["trip_id"] == "T1") & (rows["stop_id"] == "S2")).any()


def test_select_timepoint_rows_falls_back_to_timed_stops(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs", with_timepoint_col=False)
    data = target.load_gtfs_data(str(feed), files=("stop_times.txt",))
    st = data["stop_times"].copy()
    untimed = (st["trip_id"] == "T1") & (st["stop_id"] == "S2")
    st.loc[untimed, ["arrival_time", "departure_time"]] = None
    rows = target.select_timepoint_rows(st)
    assert len(rows) == 6  # untimed S2 row dropped, everything else kept


def test_compute_intervals_pairs_and_math(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    data = target.load_gtfs_data(
        str(feed), files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")
    )
    tp = target.select_timepoint_rows(data["stop_times"])
    intervals = target.compute_timepoint_intervals(tp, data["trips"], data["routes"], data["stops"])

    assert list(intervals.columns) == target.INTERVAL_COLUMNS
    assert len(intervals) == 3  # one consecutive pair per trip

    t1 = intervals.loc[intervals["trip_id"] == "T1"].iloc[0]
    assert (t1["from_stop_id"], t1["to_stop_id"]) == ("S1", "S3")  # skips non-timepoint S2
    assert t1["interval_minutes"] == 12
    assert (t1["from_departure"], t1["to_arrival"]) == ("07:00", "07:12")

    t3 = intervals.loc[intervals["trip_id"] == "T3"].iloc[0]
    assert t3["interval_minutes"] == 12  # 23:55 -> 24:07 crosses midnight
    assert t3["to_arrival"] == "24:07"
    assert t3["route_short_name"] == "20"
    assert t3["from_stop_name"] == "Elm & Oak"


def test_flag_long_intervals_boundary_and_negatives(caplog: pytest.LogCaptureFixture) -> None:
    intervals = pd.DataFrame(
        {
            "service_id": ["A", "A", "A"],
            "route_id": ["R1", "R1", "R1"],
            "direction_id": ["0", "0", "0"],
            "trip_id": ["T1", "T2", "T3"],
            "from_stop_sequence": [1, 1, 1],
            "interval_minutes": [10.0, 10.5, -3.0],
        }
    )
    with caplog.at_level("WARNING"):
        flags = target.flag_long_intervals(intervals, 10.0)
    assert flags["trip_id"].tolist() == ["T2"]  # exactly 10 is allowed; negative excluded
    assert "run backwards" in caplog.text


def test_run_end_to_end_writes_outputs(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    out_dir = tmp_path / "out"
    flags = target.run(str(feed), out_dir, max_interval_minutes=10.0)

    assert flags["trip_id"].tolist() == ["T1", "T3"]  # 12-min gaps; T2's 9-min gap passes
    assert set(flags["schedule"]) == {"Weekday"}

    detail = pd.read_csv(out_dir / target.DETAIL_FILENAME)
    assert len(detail) == 2
    summary = pd.read_csv(out_dir / target.SUMMARY_FILENAME)
    assert len(summary) == 2
    assert summary["n_trips_flagged"].tolist() == [1, 1]

    runlog = out_dir / f"{Path(target.DETAIL_FILENAME).stem}_runlog.txt"
    assert runlog.exists()
    text = runlog.read_text(encoding="utf-8")
    assert "MAX_INTERVAL_MINUTES" in text  # verbatim config block captured
    assert "Intervals flagged:    2" in text


def test_run_stricter_threshold_catches_more(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    flags = target.run(str(feed), tmp_path / "out8", max_interval_minutes=8.0)
    assert sorted(flags["trip_id"]) == ["T1", "T2", "T3"]
    assert set(flags["schedule"]) == {"Weekday", "Saturday"}


def test_run_service_and_route_filters(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "gtfs")
    flags = target.run(str(feed), tmp_path / "out_f", 8.0, service_ids=["WKDY"])
    assert sorted(flags["trip_id"]) == ["T1", "T3"]

    # Short-name matching: "20" keeps only route R2.
    flags = target.run(str(feed), tmp_path / "out_r", 8.0, routes_include=["20"])
    assert flags["trip_id"].tolist() == ["T3"]

    flags = target.run(str(feed), tmp_path / "out_x", 8.0, routes_exclude=["R2"])
    assert sorted(flags["trip_id"]) == ["T1", "T2"]


def test_build_schedule_labels() -> None:
    calendar = pd.DataFrame(
        {
            "service_id": ["W", "S", "U", "X"],
            "monday": ["1", "0", "0", "1"],
            "tuesday": ["1", "0", "0", "0"],
            "wednesday": ["1", "0", "0", "0"],
            "thursday": ["1", "0", "0", "0"],
            "friday": ["1", "0", "0", "0"],
            "saturday": ["0", "1", "0", "0"],
            "sunday": ["0", "0", "1", "0"],
        }
    )
    labels = target.build_schedule_labels(calendar)
    assert labels == {"W": "Weekday", "S": "Saturday", "U": "Sunday", "X": "Special"}
    assert target.build_schedule_labels(None) == {}


def test_validate_required_columns_raises() -> None:
    data = {
        "stops": pd.DataFrame({"stop_id": []}),
        "trips": pd.DataFrame({"trip_id": [], "route_id": []}),  # service_id missing
    }
    with pytest.raises(ValueError, match="trips.txt is missing column"):
        target.validate_required_columns(data)


def test_main_placeholder_paths_return_2() -> None:
    assert target.main([]) == 2
