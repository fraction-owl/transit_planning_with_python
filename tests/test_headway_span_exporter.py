from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import scripts.gtfs_exports.headway_span_exporter as target

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _calendar_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_id": ["WK", "SAT", "MF_ONLY"],
            "monday": ["1", "0", "1"],
            "tuesday": ["1", "0", "1"],
            "wednesday": ["1", "0", "1"],
            "thursday": ["1", "0", "1"],
            "friday": ["1", "0", "0"],
            "saturday": ["0", "1", "0"],
            "sunday": ["0", "0", "0"],
        }
    )


def _write_gtfs(gtfs_dir: Path) -> None:
    """Minimal weekday feed: route R1 with 3 trips at 06:00/06:30/07:00."""
    (gtfs_dir / "routes.txt").write_text("route_id,route_short_name\nR1,101\nR2,202\n")
    (gtfs_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id\n"
        "R1,WK,T1,0\nR1,WK,T2,0\nR1,WK,T3,0\nR2,SAT,T4,0\n"
    )
    (gtfs_dir / "stop_times.txt").write_text(
        "trip_id,stop_id,stop_sequence,departure_time\n"
        "T1,S1,1,06:00:00\nT1,S2,2,06:10:00\n"
        "T2,S1,1,06:30:00\nT2,S2,2,06:40:00\n"
        "T3,S1,1,07:00:00\nT3,S2,2,07:10:00\n"
        "T4,S1,1,09:00:00\n"
    )
    (gtfs_dir / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday\n"
        "WK,1,1,1,1,1,0,0\nSAT,0,0,0,0,0,1,0\n"
    )


# ---------------------------------------------------------------------------
# parse_time_to_minutes
# ---------------------------------------------------------------------------


def test_parse_time_to_minutes_basic_conversion() -> None:
    assert target.parse_time_to_minutes("06:30:00") == 390
    assert target.parse_time_to_minutes("06:30") == 390


def test_parse_time_to_minutes_preserves_times_past_midnight() -> None:
    assert target.parse_time_to_minutes("25:30:00") == 1530


def test_parse_time_to_minutes_rounds_seconds() -> None:
    assert target.parse_time_to_minutes("06:00:31") == 361
    assert target.parse_time_to_minutes("06:00:29") == 360
    # Python banker's rounding: exactly 30 s rounds to the even value (0).
    assert target.parse_time_to_minutes("06:00:30") == 360


def test_parse_time_to_minutes_invalid_values_return_none() -> None:
    assert target.parse_time_to_minutes("not a time") is None
    assert target.parse_time_to_minutes(None) is None
    assert target.parse_time_to_minutes(360) is None


# ---------------------------------------------------------------------------
# service_ids_for_day
# ---------------------------------------------------------------------------


def test_service_ids_for_day_weekday_requires_all_five_days() -> None:
    ids = target.service_ids_for_day(_calendar_df(), "weekday")
    assert ids == {"WK"}  # MF_ONLY misses Friday


def test_service_ids_for_day_saturday() -> None:
    assert target.service_ids_for_day(_calendar_df(), "saturday") == {"SAT"}


def test_service_ids_for_day_invalid_day_raises() -> None:
    with pytest.raises(ValueError, match="SERVICE_DAY"):
        target.service_ids_for_day(_calendar_df(), "holiday")


def test_service_ids_for_day_missing_column_raises() -> None:
    cal = _calendar_df().drop(columns=["friday"])
    with pytest.raises(ValueError, match="friday"):
        target.service_ids_for_day(cal, "weekday")


def test_service_ids_for_day_no_match_returns_empty(caplog) -> None:
    cal = _calendar_df()
    with caplog.at_level("WARNING"):
        ids = target.service_ids_for_day(cal, "sunday")
    assert ids == set()
    assert "No service_ids" in caplog.text


# ---------------------------------------------------------------------------
# first_departures
# ---------------------------------------------------------------------------


def test_first_departures_takes_lowest_stop_sequence() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2"],
            "stop_sequence": ["2", "1", "1"],
            "departure_time": ["06:10:00", "06:00:00", "07:00:00"],
        }
    )
    out = target.first_departures(stop_times, {"T1", "T2"})
    lookup = dict(zip(out["trip_id"], out["departure_min"]))
    assert lookup["T1"] == 360
    assert lookup["T2"] == 420


def test_first_departures_filters_to_requested_trips() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T9"],
            "stop_sequence": ["1", "1"],
            "departure_time": ["06:00:00", "08:00:00"],
        }
    )
    out = target.first_departures(stop_times, {"T1"})
    assert set(out["trip_id"]) == {"T1"}


# ---------------------------------------------------------------------------
# compute_headway_span
# ---------------------------------------------------------------------------


def test_compute_headway_span_single_direction() -> None:
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"] * 3,
            "direction_id": ["0"] * 3,
            "departure_min": [360, 390, 420],
        }
    )
    out = target.compute_headway_span(trips_dep)
    row = out.iloc[0]
    assert row["avg_headway_min"] == 30.0
    assert row["span_hrs"] == 1.0
    assert row["trip_count"] == 3


def test_compute_headway_span_averages_across_directions() -> None:
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"] * 4,
            "direction_id": ["0", "0", "1", "1"],
            "departure_min": [360, 380, 360, 420],  # 20-min and 60-min headways
        }
    )
    out = target.compute_headway_span(trips_dep)
    row = out.iloc[0]
    assert row["avg_headway_min"] == 40.0  # mean of 20 and 60
    assert row["span_hrs"] == 1.0  # longest span across directions
    assert row["trip_count"] == 4


def test_compute_headway_span_single_trip_gives_nan_headway() -> None:
    trips_dep = pd.DataFrame(
        {
            "route_id": ["R1"],
            "direction_id": ["0"],
            "departure_min": [360],
        }
    )
    out = target.compute_headway_span(trips_dep)
    row = out.iloc[0]
    assert math.isnan(row["avg_headway_min"])
    assert row["span_hrs"] == 0.0
    assert row["trip_count"] == 1


def test_compute_headway_span_empty_input() -> None:
    trips_dep = pd.DataFrame(columns=["route_id", "direction_id", "departure_min"])
    out = target.compute_headway_span(trips_dep)
    assert out.empty
    assert list(out.columns) == ["route_id", "avg_headway_min", "span_hrs", "trip_count"]


# ---------------------------------------------------------------------------
# load_gtfs / run
# ---------------------------------------------------------------------------


def test_load_gtfs_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="trips.txt"):
        target.load_gtfs(tmp_path)


def test_run_end_to_end_writes_csv(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    out_csv = tmp_path / "out" / "headway_span_by_route.csv"

    result = target.run(gtfs_folder=gtfs_dir, output_path=out_csv, service_day="weekday")

    assert out_csv.exists()
    assert list(result["route_id"]) == ["R1"]  # R2 is Saturday-only
    row = result.iloc[0]
    assert row["avg_headway_min"] == 30.0
    assert row["span_hrs"] == 1.0
    assert row["trip_count"] == 3
    written = pd.read_csv(out_csv)
    assert len(written) == 1


def test_run_route_filters_can_exclude_everything(tmp_path: Path) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs(gtfs_dir)
    with pytest.raises(SystemExit):
        target.run(
            gtfs_folder=gtfs_dir,
            output_path=tmp_path / "out.csv",
            service_day="weekday",
            filter_out_route_short_names=["101"],
        )
