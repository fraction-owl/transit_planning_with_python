from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.field_tools.gtfs_route_summary import (
    build_summary,
    classify_services,
    hms_to_seconds,
    load_optional_lookup,
    trip_distances_meters,
    trip_durations_seconds,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gtfs_basic"


# ---------------------------------------------------------------------------
# hms_to_seconds
# ---------------------------------------------------------------------------


def test_hms_to_seconds_normal() -> None:
    assert hms_to_seconds("07:25:00") == 7 * 3600 + 25 * 60


def test_hms_to_seconds_past_midnight() -> None:
    assert hms_to_seconds("25:00:00") == 25 * 3600


def test_hms_to_seconds_none() -> None:
    assert hms_to_seconds(None) is None  # type: ignore[arg-type]


def test_hms_to_seconds_invalid() -> None:
    assert hms_to_seconds("not-a-time") is None


# ---------------------------------------------------------------------------
# classify_services
# ---------------------------------------------------------------------------


def _make_calendar(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_classify_services_weekday_only() -> None:
    cal = _make_calendar(
        [
            {
                "service_id": "WKD",
                "monday": "1",
                "tuesday": "1",
                "wednesday": "1",
                "thursday": "1",
                "friday": "1",
                "saturday": "0",
                "sunday": "0",
                "start_date": "20260101",
                "end_date": "20261231",
            }
        ]
    )
    result = classify_services(cal, None)
    assert result["WKD"] == {"Weekday"}


def test_classify_services_saturday_only() -> None:
    cal = _make_calendar(
        [
            {
                "service_id": "SAT",
                "monday": "0",
                "tuesday": "0",
                "wednesday": "0",
                "thursday": "0",
                "friday": "0",
                "saturday": "1",
                "sunday": "0",
                "start_date": "20260101",
                "end_date": "20261231",
            }
        ]
    )
    result = classify_services(cal, None)
    assert result["SAT"] == {"Saturday"}


def test_classify_services_holiday_threshold() -> None:
    """A service active only ~10 days/year is classified Holiday."""
    cal = _make_calendar(
        [
            {
                "service_id": "HOL",
                "monday": "1",
                "tuesday": "0",
                "wednesday": "0",
                "thursday": "0",
                "friday": "0",
                "saturday": "0",
                "sunday": "0",
                # One Monday in January only → ~1 day active / tiny span
                "start_date": "20260105",
                "end_date": "20260105",
            }
        ]
    )
    result = classify_services(cal, None, holiday_max_days_per_year=25)
    assert result["HOL"] == {"Holiday"}


def test_classify_services_calendar_dates_exception_adds_date() -> None:
    """exception_type=1 adds a date that wasn't in the base schedule."""
    cal = _make_calendar(
        [
            {
                "service_id": "WKD",
                "monday": "1",
                "tuesday": "1",
                "wednesday": "1",
                "thursday": "1",
                "friday": "1",
                "saturday": "0",
                "sunday": "0",
                "start_date": "20260101",
                "end_date": "20261231",
            }
        ]
    )
    cal_dates = pd.DataFrame([{"service_id": "WKD", "date": "20260103", "exception_type": "1"}])
    result = classify_services(cal, cal_dates)
    assert "Weekday" in result["WKD"]


# ---------------------------------------------------------------------------
# trip_distances_meters
# ---------------------------------------------------------------------------


def test_trip_distances_meters_from_stop_times() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "stop_sequence": ["1", "2", "3"],
            "shape_dist_traveled": ["0", "500", "1000"],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1"]})
    result = trip_distances_meters(stop_times, None, trips, "meters")
    assert result["T1"] == pytest.approx(1000.0)


def test_trip_distances_meters_unit_conversion_feet() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "stop_sequence": ["1", "2"],
            "shape_dist_traveled": ["0", "5280"],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1"]})
    result = trip_distances_meters(stop_times, None, trips, "feet")
    assert result["T1"] == pytest.approx(5280 * 0.3048)


def test_trip_distances_meters_no_dist_returns_empty() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "stop_sequence": ["1", "2"],
            "arrival_time": ["07:00:00", "07:05:00"],
            "departure_time": ["07:00:00", "07:05:00"],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1"]})
    result = trip_distances_meters(stop_times, None, trips, "meters")
    assert result.empty


# ---------------------------------------------------------------------------
# trip_durations_seconds
# ---------------------------------------------------------------------------


def test_trip_durations_seconds_basic() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "stop_sequence": ["1", "2"],
            "departure_time": ["07:00:00", "07:25:00"],
            "arrival_time": ["07:00:00", "07:25:00"],
        }
    )
    result = trip_durations_seconds(stop_times)
    assert result["T1"] == pytest.approx(25 * 60)


def test_trip_durations_seconds_past_midnight() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "stop_sequence": ["1", "2"],
            "departure_time": ["23:50:00", "24:10:00"],
            "arrival_time": ["23:50:00", "24:10:00"],
        }
    )
    result = trip_durations_seconds(stop_times)
    assert result["T1"] == pytest.approx(20 * 60)


# ---------------------------------------------------------------------------
# load_optional_lookup
# ---------------------------------------------------------------------------


def test_load_optional_lookup_csv(tmp_path: Path) -> None:
    f = tmp_path / "svc.csv"
    f.write_text("route_id,service_type\nR1,Local\nR2,Express\n", encoding="utf-8")
    result = load_optional_lookup(str(f), "service_type")
    assert result == {"R1": "Local", "R2": "Express"}


def test_load_optional_lookup_tsv(tmp_path: Path) -> None:
    f = tmp_path / "svc.tsv"
    f.write_text("route_id\tservice_type\nR1\tLocal\n", encoding="utf-8")
    result = load_optional_lookup(str(f), "service_type")
    assert result == {"R1": "Local"}


def test_load_optional_lookup_missing_file() -> None:
    result = load_optional_lookup("/nonexistent/path.csv", "service_type")
    assert result == {}


def test_load_optional_lookup_empty_path() -> None:
    assert load_optional_lookup("", "service_type") == {}


def test_load_optional_lookup_missing_column(tmp_path: Path) -> None:
    f = tmp_path / "bad.csv"
    f.write_text("route_id,other_col\nR1,X\n", encoding="utf-8")
    result = load_optional_lookup(str(f), "service_type")
    assert result == {}


# ---------------------------------------------------------------------------
# build_summary (integration using gtfs_basic fixture)
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / name, dtype=str)


@pytest.fixture()
def gtfs_basic() -> dict[str, pd.DataFrame]:
    return {
        "routes": _load_fixture("routes.txt"),
        "trips": _load_fixture("trips.txt"),
        "stop_times": _load_fixture("stop_times.txt"),
        "calendar": _load_fixture("calendar.txt"),
    }


def test_build_summary_returns_one_row_per_route(
    gtfs_basic: dict[str, pd.DataFrame],
) -> None:
    summary = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras={},
    )
    assert len(summary) == 3
    assert set(summary["route_short_name"]) == {"R1", "R2", "R3"}


def test_build_summary_weekday_flag_set(gtfs_basic: dict[str, pd.DataFrame]) -> None:
    summary = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras={},
    )
    assert (summary["weekday"] == "Y").all()
    assert (summary["saturday"] == "").all()
    assert (summary["sunday"] == "").all()


def test_build_summary_duration_computed(gtfs_basic: dict[str, pd.DataFrame]) -> None:
    summary = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras={},
    )
    r1 = summary[summary["route_short_name"] == "R1"].iloc[0]
    assert r1["avg_duration_min"] == pytest.approx(25.0)


def test_build_summary_excluded_route_omitted(gtfs_basic: dict[str, pd.DataFrame]) -> None:
    import scripts.field_tools.gtfs_route_summary as mod

    original = mod.EXCLUDED_ROUTE_SHORT_NAMES
    mod.EXCLUDED_ROUTE_SHORT_NAMES = ["R1"]
    try:
        summary = build_summary(
            routes_df=gtfs_basic["routes"],
            trips_df=gtfs_basic["trips"],
            stop_times_df=gtfs_basic["stop_times"],
            calendar_df=gtfs_basic["calendar"],
            calendar_dates_df=None,
            shapes_df=None,
            distance_unit="meters",
            extras={},
        )
        assert "R1" not in summary["route_short_name"].to_numpy()
        assert len(summary) == 2
    finally:
        mod.EXCLUDED_ROUTE_SHORT_NAMES = original


def test_build_summary_threshold_params_passed_through(
    gtfs_basic: dict[str, pd.DataFrame],
) -> None:
    """Overriding holiday_max_days_per_year via param should take effect."""
    summary_normal = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras={},
        holiday_max_days_per_year=25,
    )
    summary_high = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras={},
        holiday_max_days_per_year=400,
    )
    assert (summary_normal["weekday"] == "Y").all()
    assert (summary_high["holiday"] == "Y").all()


def test_build_summary_extras_joined(gtfs_basic: dict[str, pd.DataFrame]) -> None:
    extras = {
        "service_types": {"R1": "Local", "R2": "Express"},
        "corridors": {},
        "last_changed": {},
        "ridership": {},
    }
    summary = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras=extras,
    )
    assert summary[summary["route_short_name"] == "R1"].iloc[0]["service_type"] == "Local"
    assert summary[summary["route_short_name"] == "R2"].iloc[0]["service_type"] == "Express"
    assert summary[summary["route_short_name"] == "R3"].iloc[0]["service_type"] == ""


def test_build_summary_export_to_xlsx(gtfs_basic: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    from scripts.field_tools.gtfs_route_summary import export_to_xlsx

    summary = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras={},
    )
    out = str(tmp_path / "summary.xlsx")
    export_to_xlsx(summary, out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0
