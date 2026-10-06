from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import scripts.field_tools.gtfs_route_summary as mod
from scripts.field_tools.gtfs_route_summary import (
    build_summary,
    classify_services,
    expand_service_active_dates,
    hms_to_seconds,
    holiday_dates_for,
    load_gtfs_data,
    load_optional_lookup,
    parse_holiday_dates,
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


def _classify(
    cal: pd.DataFrame | None,
    cal_dates: pd.DataFrame | None = None,
    extra_holiday_dates: tuple[str, ...] = (),
    weekday_dow_share: float = mod.WEEKDAY_DOW_SHARE,
) -> dict[str, set[str]]:
    active = expand_service_active_dates(cal, cal_dates)
    return classify_services(
        active,
        holiday_dates_for(active, extra_holiday_dates),
        weekday_dow_share=weekday_dow_share,
    )


_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _cal_row(
    sid: str, days: tuple[str, ...], start: str = "20260101", end: str = "20261231"
) -> dict[str, str]:
    row = {"service_id": sid, "start_date": start, "end_date": end}
    row.update({day: "1" if day in days else "0" for day in _DAYS})
    return row


def _added_dates(sid: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"service_id": sid, "date": d, "exception_type": "1"} for d in dates])


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
    result = _classify(cal)
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
    result = _classify(cal)
    assert result["SAT"] == {"Saturday"}


def test_classify_services_mon_sat_gets_weekday_and_saturday() -> None:
    """Saturday is tested on its own, not against the larger weekday share."""
    cal = _make_calendar([_cal_row("MS", _DAYS[:6])])
    assert _classify(cal)["MS"] == {"Weekday", "Saturday"}


def test_classify_services_split_weekday_schedule_gets_weekday() -> None:
    """A Friday-only service_id still recurs on a weekday."""
    cal = _make_calendar([_cal_row("FRI", ("friday",))])
    assert _classify(cal)["FRI"] == {"Weekday"}


def test_classify_services_occasional_saturday_special_not_saturday() -> None:
    cal = _make_calendar([_cal_row("WKD", _DAYS[:5])])
    cal_dates = _added_dates("WKD", ["20260314", "20260606"])
    assert _classify(cal, cal_dates)["WKD"] == {"Weekday"}


def test_classify_services_holiday_cancellations_keep_weekday() -> None:
    """Removing every federal holiday does not count against the weekday pattern."""
    cal = _make_calendar([_cal_row("WKD", _DAYS[:5])])
    holidays = sorted(d.strftime("%Y%m%d") for d in mod.federal_holidays_observed(2026))
    cal_dates = pd.DataFrame(
        [{"service_id": "WKD", "date": d, "exception_type": "2"} for d in holidays]
    )
    assert _classify(cal, cal_dates, weekday_dow_share=1.0)["WKD"] == {"Weekday"}


@pytest.mark.parametrize(
    "dates",
    [["20261224", "20261225"], ["20261224", "20261225", "20261226"]],
)
def test_classify_services_holiday_stable_when_adjacent_day_added(dates: list[str]) -> None:
    """Christmas service stays Holiday whether or not Dec 26 is added."""
    cal = _make_calendar([_cal_row("WKD", _DAYS[:5])])
    assert _classify(cal, _added_dates("HOL", dates))["HOL"] == {"Holiday"}


def test_classify_services_federal_holidays_only_is_holiday() -> None:
    holidays = sorted(d.strftime("%Y%m%d") for d in mod.federal_holidays_observed(2026))
    assert _classify(None, _added_dates("HOL", holidays))["HOL"] == {"Holiday"}


def test_classify_services_one_off_non_holiday_gets_no_label() -> None:
    """A single event Saturday is neither recurring nor a holiday."""
    assert _classify(None, _added_dates("EVT", ["20260613"]))["EVT"] == set()


def test_classify_services_extra_holiday_dates() -> None:
    cal_dates = _added_dates("EVE", ["20261224"])
    assert _classify(None, cal_dates)["EVE"] == set()
    assert _classify(None, cal_dates, extra_holiday_dates=("2026-12-24",))["EVE"] == {"Holiday"}


def test_classify_services_empty_service() -> None:
    cal = _make_calendar([_cal_row("GONE", ("monday",), "20260105", "20260105")])
    cal_dates = pd.DataFrame([{"service_id": "GONE", "date": "20260105", "exception_type": "2"}])
    assert _classify(cal, cal_dates)["GONE"] == set()


def test_holiday_dates_for_includes_next_year_observed_new_year() -> None:
    """New Year's Day 2022 (a Saturday) was observed on 2021-12-31."""
    import datetime as dt

    active = {"S": {dt.date(2021, 6, 1)}}
    assert dt.date(2021, 12, 31) in holiday_dates_for(active)


def test_parse_holiday_dates_rejects_bad_value() -> None:
    with pytest.raises(ValueError, match="not YYYYMMDD"):
        parse_holiday_dates(["12/24/2026"])


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
    result = _classify(cal, cal_dates)
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


def _shapes(lengths: dict[str, str]) -> pd.DataFrame:
    rows = []
    for shape_id, length in lengths.items():
        rows.append({"shape_id": shape_id, "shape_pt_sequence": "1", "shape_dist_traveled": "0"})
        rows.append({"shape_id": shape_id, "shape_pt_sequence": "2", "shape_dist_traveled": length})
    return pd.DataFrame(rows)


def test_trip_distances_meters_shape_fallback_per_trip() -> None:
    """A trip without stop_times distances still gets its shape's length."""
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2"],
            "stop_sequence": ["1", "2", "1", "2"],
            "shape_dist_traveled": ["0", "1000", "", ""],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1", "T2"], "shape_id": ["S1", "S2"]})
    result = trip_distances_meters(
        stop_times, _shapes({"S1": "999", "S2": "2500"}), trips, "meters"
    )
    assert result.to_dict() == pytest.approx({"T1": 1000.0, "T2": 2500.0})


def test_trip_distances_meters_blank_last_stop_is_unknown_not_zero() -> None:
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "stop_sequence": ["1", "2", "3"],
            "shape_dist_traveled": ["0", "", ""],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1"], "shape_id": ["S1"]})
    assert trip_distances_meters(stop_times, None, trips, "meters").empty
    with_shape = trip_distances_meters(stop_times, _shapes({"S1": "4000"}), trips, "meters")
    assert with_shape["T1"] == pytest.approx(4000.0)


def test_trip_distances_meters_orders_by_stop_sequence() -> None:
    """Rows out of file order are sorted numerically by stop_sequence."""
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "stop_sequence": ["10", "2", "1"],
            "shape_dist_traveled": ["1500", "", "0"],
        }
    )
    trips = pd.DataFrame({"trip_id": ["T1"]})
    assert trip_distances_meters(stop_times, None, trips, "meters")["T1"] == pytest.approx(1500)


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
    """Overriding min_recurring_dates via param should take effect.

    No weekday can recur 400 times in one year, and the fixture's service runs
    on New Year's Day, so every route falls through to Holiday.
    """
    summary_normal = build_summary(
        routes_df=gtfs_basic["routes"],
        trips_df=gtfs_basic["trips"],
        stop_times_df=gtfs_basic["stop_times"],
        calendar_df=gtfs_basic["calendar"],
        calendar_dates_df=None,
        shapes_df=None,
        distance_unit="meters",
        extras={},
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
        min_recurring_dates=400,
    )
    assert (summary_normal["weekday"] == "Y").all()
    assert (summary_high["weekday"] == "").all()
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


# ---------------------------------------------------------------------------
# build_summary: speed pairing, inactive services, calendar_dates-only feeds
# ---------------------------------------------------------------------------


def _two_trip_feed(service_ids: tuple[str, str]) -> dict[str, pd.DataFrame]:
    """Trip A: 10 km in 10 min. Trip B: unknown distance, 100 min."""
    return {
        "routes": pd.DataFrame(
            {"route_id": ["R1"], "route_short_name": ["1"], "route_long_name": ["One"]}
        ),
        "trips": pd.DataFrame(
            {
                "route_id": ["R1", "R1"],
                "service_id": list(service_ids),
                "trip_id": ["A", "B"],
                "direction_id": ["0", "1"],
                "shape_id": ["SA", "SB"],
            }
        ),
        "stop_times": pd.DataFrame(
            {
                "trip_id": ["A", "A", "B", "B"],
                "stop_sequence": ["1", "2", "1", "2"],
                "arrival_time": ["07:00:00", "07:10:00", "08:00:00", "09:40:00"],
                "departure_time": ["07:00:00", "07:10:00", "08:00:00", "09:40:00"],
                "shape_dist_traveled": ["0", "10000", "", ""],
            }
        ),
    }


def _summary(
    feed: dict[str, pd.DataFrame],
    calendar_df: pd.DataFrame | None = None,
    calendar_dates_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_summary(
        routes_df=feed["routes"],
        trips_df=feed["trips"],
        stop_times_df=feed["stop_times"],
        calendar_df=calendar_df,
        calendar_dates_df=calendar_dates_df,
        shapes_df=None,
        distance_unit="meters",
        extras={},
    )


def test_build_summary_speed_uses_trips_with_both_measures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "OUTPUT_UNITS", "metric")
    feed = _two_trip_feed(("WK", "WK"))
    summary = _summary(feed, calendar_df=_make_calendar([_cal_row("WK", _DAYS[:5])]))
    row = summary.iloc[0]
    assert row["avg_speed_kmh"] == pytest.approx(60.0)
    assert row["avg_distance_km"] == pytest.approx(10.0)
    assert row["avg_duration_min"] == pytest.approx(55.0)


def test_build_summary_skips_trips_with_no_active_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "OUTPUT_UNITS", "metric")
    feed = _two_trip_feed(("WK", "GONE"))
    cal = _make_calendar(
        [_cal_row("WK", _DAYS[:5]), _cal_row("GONE", ("monday",), "20260105", "20260105")]
    )
    cal_dates = pd.DataFrame([{"service_id": "GONE", "date": "20260105", "exception_type": "2"}])
    row = _summary(feed, calendar_df=cal, calendar_dates_df=cal_dates).iloc[0]
    assert row["variants"] == 1
    assert row["directions"] == 1
    assert row["avg_duration_min"] == pytest.approx(10.0)
    assert row["avg_speed_kmh"] == pytest.approx(60.0)


def test_build_summary_route_with_only_inactive_service_omitted() -> None:
    feed = _two_trip_feed(("GONE", "GONE"))
    assert _summary(feed, calendar_df=_make_calendar([])).empty


def test_build_summary_calendar_dates_only() -> None:
    feed = _two_trip_feed(("WK", "WK"))
    mondays = pd.date_range("2026-01-05", "2026-03-30", freq="W-MON").strftime("%Y%m%d")
    summary = _summary(feed, calendar_dates_df=_added_dates("WK", list(mondays)))
    assert summary.iloc[0]["weekday"] == "Y"


# ---------------------------------------------------------------------------
# Literal identifiers such as "NA" and feeds without calendar.txt
# ---------------------------------------------------------------------------


def _write_feed(folder: Path, route_id: str = "NA") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "routes.txt").write_text(
        f"route_id,route_short_name,route_long_name\n{route_id},{route_id},Null Ave\n",
        encoding="utf-8",
    )
    (folder / "trips.txt").write_text(
        f"route_id,service_id,trip_id,shape_id\n{route_id},WK,T1,\n", encoding="utf-8"
    )
    (folder / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T1,07:00:00,07:00:00,S1,1\n"
        "T1,07:20:00,07:20:00,S2,2\n",
        encoding="utf-8",
    )
    (folder / "calendar_dates.txt").write_text(
        "service_id,date,exception_type\nWK,20260105,1\n", encoding="utf-8"
    )


def test_load_gtfs_data_keeps_literal_na_and_blanks_missing(tmp_path: Path) -> None:
    _write_feed(tmp_path)
    data = load_gtfs_data(str(tmp_path), files=("routes.txt", "trips.txt"))
    assert data["routes"]["route_id"].tolist() == ["NA"]
    assert data["trips"]["route_id"].tolist() == ["NA"]
    assert data["trips"]["shape_id"].isna().all()


def test_load_optional_lookup_keeps_literal_na(tmp_path: Path) -> None:
    f = tmp_path / "svc.csv"
    f.write_text("route_id,service_type\nNA,Local\nNULL,\n,Orphan\n", encoding="utf-8")
    assert load_optional_lookup(str(f), "service_type") == {"NA": "Local", "NULL": ""}


def test_main_runs_on_calendar_dates_only_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed_dir = tmp_path / "feed"
    out_dir = tmp_path / "out"
    _write_feed(feed_dir)
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(feed_dir))
    monkeypatch.setattr(mod, "BASE_OUTPUT_PATH", str(out_dir))
    assert mod.main() == 0
    written = pd.read_excel(out_dir / mod.OUTPUT_FILENAME, dtype=str, keep_default_na=False)
    assert written["route_short_name"].tolist() == ["NA"]


def test_main_fails_without_any_calendar_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_feed(tmp_path)
    (tmp_path / "calendar_dates.txt").unlink()
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(tmp_path))
    monkeypatch.setattr(mod, "BASE_OUTPUT_PATH", str(tmp_path / "out"))
    assert mod.main() == 1


def test_main_rejects_bad_extra_holiday_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "GTFS_FOLDER_PATH", str(tmp_path))
    monkeypatch.setattr(mod, "BASE_OUTPUT_PATH", str(tmp_path / "out"))
    monkeypatch.setattr(mod, "EXTRA_HOLIDAY_DATES", ["Dec 24"])
    assert mod.main() == 2
