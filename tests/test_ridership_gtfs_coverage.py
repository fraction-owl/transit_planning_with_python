"""Tests for ridership_gtfs_coverage: stop-level ridership data vs GTFS service."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import ridership_gtfs_coverage as target  # noqa: E402


def _write_gtfs(tmp_path: Path) -> Path:
    """Write a minimal feed with weekday and Saturday service.

    Weekday: route "1" serves S1,S2,S3 and route "2" serves S3,S1.
    Saturday: route "2" serves S2 only.
    """
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "WKDY,1,1,1,1,1,0,0,20250106,20250119\n"
        "SAT,0,0,0,0,0,1,0,20250106,20250119\n",
        encoding="utf-8",
    )
    (gtfs / "stops.txt").write_text(
        "stop_id,stop_code,stop_name,stop_lat,stop_lon\n"
        "S1,101,First & Main,38.90,-77.03\n"
        "S2,102,Second & Main,38.91,-77.03\n"
        "S3,103,Third & Main,38.92,-77.03\n",
        encoding="utf-8",
    )
    (gtfs / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name,route_type\n"
        "R1,1,Main Street,3\n"
        "R2,2,Crosstown,3\n",
        encoding="utf-8",
    )
    (gtfs / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id\n"
        "R1,WKDY,T1,0\n"
        "R1,WKDY,T2,0\n"
        "R2,WKDY,T4,0\n"
        "R2,SAT,T3,0\n",
        encoding="utf-8",
    )
    (gtfs / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T1,07:00:00,07:00:00,S1,1\n"
        "T1,07:05:00,07:05:00,S2,2\n"
        "T1,07:10:00,07:10:00,S3,3\n"
        "T2,08:00:00,08:00:00,S1,1\n"
        "T2,08:05:00,08:05:00,S2,2\n"
        "T4,09:00:00,09:00:00,S3,1\n"
        "T4,09:05:00,09:05:00,S1,2\n"
        "T3,10:00:00,10:00:00,S2,1\n",
        encoding="utf-8",
    )
    return gtfs


def _write_ridership(tmp_path: Path) -> dict:
    """Write ridership exports exercising both disagreement directions.

    Weekday data covers S1 and S2 under route "1" only; S3 is absent and
    phantom stop S9 appears. Saturday data has S2 (expected) and S3 (not
    scheduled on Saturdays).
    """
    weekday = tmp_path / "weekday.csv"
    pd.DataFrame(
        {
            "STOP_ID": ["S1", "S2", "S9"],
            "ROUTE_NAME": ["1", "1", "1"],
            "BOARD_ALL": ["10.5", "7.25", "1.0"],
        }
    ).to_csv(weekday, index=False)
    saturday = tmp_path / "saturday.csv"
    pd.DataFrame(
        {
            "STOP_ID": ["S2", "S3"],
            "ROUTE_NAME": ["2", "2"],
            "BOARD_ALL": ["5.0", "2.0"],
        }
    ).to_csv(saturday, index=False)
    return {"Weekday": (str(weekday),), "Saturday": (str(saturday),)}


def _cfg(tmp_path: Path, **overrides: object) -> "target.Config":
    base = {
        "gtfs_path": _write_gtfs(tmp_path),
        "output_dir": tmp_path / "out",
        "ridership_files": _write_ridership(tmp_path),
    }
    base.update(overrides)
    return target.Config(**base)


def test_normalize_key_series_strips_excel_artifacts() -> None:
    """Whitespace and the trailing '.0' Excel adds to numeric IDs are removed."""
    raw = pd.Series([" 1001.0", "S1 ", "10.5", "1001"])
    assert list(target.normalize_key_series(raw)) == ["1001", "S1", "10.5", "1001"]


def test_load_ridership_records_validates_day_types(tmp_path: Path) -> None:
    """An unknown day-type key is an actionable error."""
    with pytest.raises(ValueError, match="unknown day type"):
        target.load_ridership_records({"Wednesday": ("x.csv",)})


def test_load_ridership_records_missing_column_raises(tmp_path: Path) -> None:
    """A file lacking the configured stop column fails loudly."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"WRONG": ["S1"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="STOP_ID_COLUMN"):
        target.load_ridership_records({"Weekday": (str(bad),)})


def test_summarize_expected_service_share_threshold() -> None:
    """A group under MIN_SHARE_OF_DAYS of the day type's dates is not expected."""
    expected_keyed = pd.DataFrame(
        {
            "service_date": pd.to_datetime(
                ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-06"]
            ),
            "day_type": ["Weekday"] * 4,
            "stop_key": ["A", "A", "A", "B"],
            "stop_id": ["A", "A", "A", "B"],
            "route_id": ["R1"] * 4,
        }
    )
    n_days = pd.Series({"Weekday": 4})
    out = target.summarize_expected_service(
        expected_keyed, n_days, ["day_type", "stop_key"], min_share_of_days=0.5
    ).set_index("stop_key")
    assert bool(out.loc["A", "expected"])  # 3 of 4 days
    assert not bool(out.loc["B", "expected"])  # 1 of 4 days
    assert out.loc["A", "avg_daily_sched_visits"] == 0.8  # 3 visits / 4 days


def test_run_end_to_end_missing_and_unexpected(tmp_path: Path) -> None:
    """The pipeline reports both directions of disagreement at both grains."""
    cfg = _cfg(tmp_path)
    tables = target.run(cfg)

    missing = tables[target.MISSING_STOPS_FILENAME]
    assert list(missing["stop_key"]) == ["S3"]
    row = missing.iloc[0]
    assert row["day_type"] == "Weekday"
    assert row["stop_name"] == "Third & Main"
    assert row["avg_daily_sched_visits"] == 2.0  # T1 + T4 daily, 10 days
    assert row["routes_serving"] == "1, 2"

    unexpected = tables[target.UNEXPECTED_STOPS_FILENAME].set_index("stop_key")
    assert unexpected.loc["S9", "reason"] == "stop not in GTFS"
    assert unexpected.loc["S3", "reason"] == "no service scheduled this day type"
    assert unexpected.loc["S3", "day_type"] == "Saturday"

    # S1 appears in weekday data for route 1 only; route 2 also serves it.
    missing_rs = tables[target.MISSING_ROUTE_STOPS_FILENAME]
    pairs = set(zip(missing_rs["route_key"], missing_rs["stop_key"]))
    assert ("2", "S1") in pairs
    # Wholly missing stops (S3) stay out of the route-level view.
    assert "S3" not in set(missing_rs["stop_key"])

    unexpected_rs = tables[target.UNEXPECTED_ROUTE_STOPS_FILENAME]
    sat_s3 = unexpected_rs.loc[
        (unexpected_rs["stop_key"] == "S3") & (unexpected_rs["day_type"] == "Saturday")
    ]
    assert len(sat_s3) == 1

    out = cfg.output_dir
    for filename in tables:
        assert (out / filename).exists()
    runlog = (out / "ridership_gtfs_coverage_runlog.txt").read_text(encoding="utf-8")
    assert "CONFIGURATION (verbatim from source)" in runlog
    assert "Day types checked" in runlog


def test_run_stop_code_matching(tmp_path: Path) -> None:
    """STOP_MATCH_FIELD='stop_code' compares against stops.txt stop_code."""
    weekday = tmp_path / "wk_codes.csv"
    pd.DataFrame(
        {"STOP_ID": ["101.0", "102"], "ROUTE_NAME": ["1", "1"], "BOARD_ALL": ["1", "1"]}
    ).to_csv(weekday, index=False)
    cfg = _cfg(
        tmp_path,
        ridership_files={"Weekday": (str(weekday),)},
        stop_match_field="stop_code",
    )
    tables = target.run(cfg)
    missing = tables[target.MISSING_STOPS_FILENAME]
    assert list(missing["stop_key"]) == ["103"]
    assert tables[target.UNEXPECTED_STOPS_FILENAME].empty


def test_run_without_route_column_skips_route_outputs(tmp_path: Path) -> None:
    """ROUTE_COLUMN='' produces only the two stop-level tables."""
    weekday = tmp_path / "wk_nr.csv"
    pd.DataFrame({"STOP_ID": ["S1", "S2", "S3"]}).to_csv(weekday, index=False)
    cfg = _cfg(
        tmp_path,
        ridership_files={"Weekday": (str(weekday),)},
        route_column="",
        boardings_column="",
    )
    tables = target.run(cfg)
    assert set(tables) == {target.MISSING_STOPS_FILENAME, target.UNEXPECTED_STOPS_FILENAME}
    assert tables[target.MISSING_STOPS_FILENAME].empty
