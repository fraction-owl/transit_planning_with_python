"""Tests for otp_runtime_from_tides: OTP, trip and segment running time at every grain."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.operations_tools.otp_runtime_from_tides as target

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")
PERIODS_FIXTURE = Path("tests/fixtures/time_periods.txt")
EXPRESS_ROUTES = Path("tests/fixtures/express_routes.txt")
MOCK_GTFS = Path("tests/fixtures/mock_gtfs_dc.zip")
HOLIDAY = dt.date(2026, 1, 1)  # the fixtures' holiday-service date, a Thursday

DEFAULT_PERIODS = target.build_time_periods(
    target.specs_from_mapping(target.TIME_PERIODS), "test defaults"
)
# The mock TIDES export is sampled, so most GTFS trips have no record on any date.
SAMPLED_EXPORT = {"max_unrecorded_share": 1.0, "max_trip_unrecorded_share": 1.0}


def _settings(tmp_path: Path, **overrides: object) -> target.RunSettings:
    base: dict = {
        "stop_visits_path": STOP_VISITS,
        "trips_performed_path": TRIPS_PERFORMED,
        "output_dir": tmp_path,
        "periods": DEFAULT_PERIODS,
        "export_charts": False,
        **SAMPLED_EXPORT,
    }
    base.update(overrides)
    return target.RunSettings(**base)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> target.RunResult:
    """Every measure, level and time cut on the fixtures, day types from the mock GTFS feed."""
    out_dir = tmp_path_factory.mktemp("full_run")
    settings = target.RunSettings(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=out_dir,
        periods=DEFAULT_PERIODS,
        measures=list(target.MEASURE_LEVELS),
        levels=list(target.LEVELS),
        time_cuts=list(target.TIME_CUTS),
        gtfs_paths=[str(MOCK_GTFS)],
        export_charts=False,
        **SAMPLED_EXPORT,
    )
    return target.run(settings)


def _ts(date: str, hhmm: str | None) -> str | None:
    """Return an ISO timestamp on *date* (``None`` stays ``None``)."""
    return None if hhmm is None else f"{date}T{hhmm}:00"


def _visits(rows: list[dict]) -> pd.DataFrame:
    """Joined stop visits with parsed timestamps; times are given as HH:MM on the row's date."""
    base = {
        "service_date": "2026-01-05",
        "trip_id_performed": "T1",
        "trip_stop_sequence": 1,
        "stop_id": "A",
        "timepoint": "TRUE",
        "schedule_relationship": "Scheduled",
        "route_id": "R1",
        "direction_id": "0",
        "trip_id": "S1",
    }
    records = []
    for row in rows:
        rec = {**base, **row}
        for col in target.TIMESTAMP_COLUMNS:
            rec[col] = _ts(rec["service_date"], rec.get(col))
        records.append(rec)
    df = pd.DataFrame(records)
    df["service_date"] = pd.to_datetime(df["service_date"])
    return target.add_parsed_timestamps(df)


def _write_tides(tmp_path: Path, visits: list[dict], trips: list[dict]) -> tuple[Path, Path]:
    """Write a synthetic stop_visits / trips_performed pair and return their paths."""
    sv, tp = tmp_path / "stop_visits.csv", tmp_path / "trips_performed.csv"
    pd.DataFrame(visits).to_csv(sv, index=False)
    pd.DataFrame(trips).to_csv(tp, index=False)
    return sv, tp


def _basis_tides(tmp_path: Path) -> tuple[Path, Path]:
    """One trip scheduled to leave at 08:55 (AM Peak) that left at 09:05 (Midday)."""
    day = "2026-01-05"  # a Monday
    visits = [
        {
            "service_date": day,
            "trip_id_performed": "T1",
            "trip_stop_sequence": 1,
            "stop_id": "A",
            "timepoint": "TRUE",
            "schedule_relationship": "Scheduled",
            "schedule_arrival_time": _ts(day, "08:55"),
            "schedule_departure_time": _ts(day, "08:55"),
            "actual_arrival_time": _ts(day, "09:04"),
            "actual_departure_time": _ts(day, "09:05"),
        },
        {
            "service_date": day,
            "trip_id_performed": "T1",
            "trip_stop_sequence": 2,
            "stop_id": "B",
            "timepoint": "TRUE",
            "schedule_relationship": "Scheduled",
            "schedule_arrival_time": _ts(day, "09:25"),
            "schedule_departure_time": _ts(day, "09:25"),
            "actual_arrival_time": _ts(day, "09:40"),
            "actual_departure_time": _ts(day, "09:41"),
        },
    ]
    trips = [
        {
            "service_date": day,
            "trip_id_performed": "T1",
            "trip_id_scheduled": "S1",
            "route_id": "R1",
            "direction_id": "0",
            "route_type_agency": "LOCAL",
            "block_id": "B1",
            "schedule_trip_start": _ts(day, "08:55"),
            "schedule_trip_end": _ts(day, "09:25"),
            "actual_trip_start": _ts(day, "09:05"),
            "actual_trip_end": _ts(day, "09:40"),
            "trip_type": "In service",
            "schedule_relationship": "Scheduled",
        }
    ]
    return _write_tides(tmp_path, visits, trips)


# --- timestamps and time keys ------------------------------------------------------


def test_parse_timestamps_reads_clock_time_as_written() -> None:
    """Offsets are dropped (so a DST change does not shift values); other formats still read."""
    values = pd.Series(
        [
            "2026-03-08T01:30:00-05:00",
            "2026-03-08T03:30:00-04:00",
            "2026-01-05 06:00:00",
            "1/5/2026 9:03 PM",
            "",
            None,
            "junk",
        ],
        dtype=object,
    )
    stamps = target.parse_timestamps(values)
    assert (stamps.iloc[1] - stamps.iloc[0]) == pd.Timedelta(hours=2)
    assert stamps.iloc[2] == pd.Timestamp("2026-01-05 06:00")
    assert stamps.iloc[3] == pd.Timestamp("2026-01-05 21:03")
    assert stamps.iloc[4:].isna().all()


def test_add_time_keys_uses_the_given_time_and_the_service_date() -> None:
    """Period and hour come from the text; month and weekday from the service date."""
    rows = pd.DataFrame(
        {"service_date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-05"])}
    )
    text = pd.Series(["2026-01-05T08:59:00", "2026-01-06T00:30:00", None])
    out = target.add_time_keys(rows, text, DEFAULT_PERIODS, 4, "rows")
    assert out["time_period"].iloc[:2].tolist() == ["AM Peak", "Night"]
    assert out["hour"].iloc[:2].tolist() == [8, 24]
    assert pd.isna(out["time_period"].iloc[2]) and pd.isna(out["hour"].iloc[2])
    assert out["month"].tolist() == ["2026-01"] * 3
    assert out["day_of_week"].tolist() == ["Monday"] * 3


@pytest.mark.parametrize(
    ("basis", "period"), [("scheduled_start", "AM Peak"), ("actual_start", "Midday")]
)
def test_trip_time_basis_places_the_trip(tmp_path: Path, basis: str, period: str) -> None:
    """TRIP_TIME_BASIS chooses the scheduled or the actual start for a trip's period."""
    sv, tp = _basis_tides(tmp_path)
    result = target.run(
        _settings(
            tmp_path,
            stop_visits_path=sv,
            trips_performed_path=tp,
            measures=["trip_runtime"],
            levels=["route"],
            time_cuts=["time_period"],
            trip_time_basis=basis,
        )
    )
    table = result.trip_runtime[("route", "time_period")]
    assert table["time_period"].tolist() == [period]
    row = table.iloc[0]
    assert (row["actual_median_min"], row["scheduled_min"], row["diff_min"]) == (35.0, 30.0, 5.0)


@pytest.mark.parametrize(
    ("basis", "otp_periods", "segment_period"),
    [("scheduled", ["AM Peak", "Midday"], "AM Peak"), ("actual", ["Midday"], "Midday")],
)
def test_stop_time_basis_places_visits_and_segments(
    tmp_path: Path, basis: str, otp_periods: list[str], segment_period: str
) -> None:
    """STOP_TIME_BASIS places stop visits by their own time and segments by their first stop."""
    sv, tp = _basis_tides(tmp_path)
    result = target.run(
        _settings(
            tmp_path,
            stop_visits_path=sv,
            trips_performed_path=tp,
            measures=["otp", "segment_runtime"],
            levels=["route", "segment"],
            time_cuts=["time_period"],
            stop_time_basis=basis,
        )
    )
    otp = result.otp[("route", "time_period")]
    assert otp["time_period"].tolist() == otp_periods
    assert otp["late"].sum() == 2  # 10 and 15 minutes late
    segments = result.segment_runtime[("segment", "time_period")]
    assert segments["time_period"].tolist() == [segment_period]
    assert segments["segment"].tolist() == ["A -> B"]


# --- on-time performance ---------------------------------------------------------------


def test_score_otp_window_is_inclusive() -> None:
    """Deviations on the window's edges are on time."""
    minutes = ["06:55", "06:59", "07:00", "07:05", "07:07"]
    visits = _visits(
        [
            {
                "trip_stop_sequence": i,
                "schedule_departure_time": "07:00",
                "actual_departure_time": actual,
            }
            for i, actual in enumerate(minutes, start=1)
        ]
    )
    scored, counts = target.score_otp(visits, -1.0, 5.0, True)
    assert scored["otp_class"].tolist() == ["early", "on_time", "on_time", "on_time", "late"]
    assert counts["scored"] == 5


def test_score_otp_filters_and_explains_unscored_visits() -> None:
    """Only Scheduled timepoints score; unscored visits are counted by cause."""
    visits = _visits(
        [
            {"schedule_departure_time": "07:00", "actual_departure_time": "07:02"},
            {
                "timepoint": "FALSE",
                "schedule_departure_time": "07:00",
                "actual_departure_time": "07:02",
            },
            {"schedule_relationship": "Skipped", "schedule_departure_time": "07:00"},
            {"schedule_relationship": "Added", "actual_departure_time": "07:02"},
            {"schedule_departure_time": "07:00"},
            {"actual_departure_time": "07:02"},
            {"schedule_arrival_time": "07:10", "actual_arrival_time": "07:20"},
        ]
    )
    scored, counts = target.score_otp(visits, -1.0, 5.0, True)
    assert counts == {
        "candidates": 4,
        "scored": 2,
        "missing_schedule_time": 1,
        "missing_actual_time": 1,
        "last_stop_excluded": 0,
        "early_ok_at_last_stop": 0,
    }
    assert scored["dev_min"].tolist() == [2.0, 10.0]
    everything, _ = target.score_otp(visits, -1.0, 5.0, False)
    assert len(everything) == 3  # the non-timepoint visit now scores too


def _loop_trip(route_id: str, end_stop: str = "A") -> list[dict]:
    """A loop trip A -> B -> A on *route_id* that is 3 minutes early at every stop."""
    return [
        {
            "trip_id_performed": f"T_{route_id}",
            "trip_stop_sequence": seq,
            "stop_id": stop,
            "route_id": route_id,
            "trip_end_stop_id": end_stop,
            "schedule_arrival_time": "07:10",
            "schedule_departure_time": "07:10",
            "actual_arrival_time": "07:07",
            "actual_departure_time": "07:07",
        }
        for seq, stop in ((1, "A"), (2, "B"), (3, "A"))
    ]


def _classes(scored: pd.DataFrame) -> dict:
    """Each performed trip's OTP classes in stop order."""
    return scored.groupby("trip_id_performed")["otp_class"].apply(list).to_dict()


def test_score_otp_last_stop_treatments() -> None:
    """Each treatment touches only a treated route's last visit, never a loop's first one."""
    visits = _visits([*_loop_trip("X1"), *_loop_trip("L1")])
    express = {"X1"}
    scored, counts = target.score_otp(visits, -1.0, 5.0, True, "score", express)
    assert set(scored["otp_class"]) == {"early"}
    assert (counts["last_stop_excluded"], counts["early_ok_at_last_stop"]) == (0, 0)

    scored, counts = target.score_otp(visits, -1.0, 5.0, True, "early_ok", express)
    assert _classes(scored) == {
        "T_X1": ["early", "early", "on_time"],
        "T_L1": ["early", "early", "early"],
    }
    assert counts["early_ok_at_last_stop"] == 1
    assert scored["dev_min"].eq(-3.0).all()  # the measured deviation is kept

    scored, counts = target.score_otp(visits, -1.0, 5.0, True, "exclude", express)
    assert _classes(scored) == {"T_X1": ["early", "early"], "T_L1": ["early", "early", "early"]}
    assert (counts["last_stop_excluded"], counts["candidates"]) == (1, 5)

    _, counts = target.score_otp(visits, -1.0, 5.0, True, "early_ok", None)
    assert counts["early_ok_at_last_stop"] == 2  # None = every route


def test_score_otp_scores_the_last_stop_on_arrival() -> None:
    """The last stop is judged and placed in time by arrival; other stops by departure."""
    rows = [
        {
            "trip_stop_sequence": 1,
            "stop_id": "A",
            "trip_end_stop_id": "B",
            "schedule_arrival_time": "08:00",
            "schedule_departure_time": "08:00",
            "actual_arrival_time": "07:58",
            "actual_departure_time": "08:03",
        },
        {
            "trip_stop_sequence": 2,
            "stop_id": "B",
            "trip_end_stop_id": "B",
            "schedule_arrival_time": "08:30",
            "schedule_departure_time": "08:30",
            "actual_arrival_time": "08:31",
            "actual_departure_time": "08:40",  # sat at the terminal before pulling out
        },
    ]
    scored, _ = target.score_otp(_visits(rows), -1.0, 5.0, True)
    assert scored["dev_min"].tolist() == [3.0, 1.0]
    assert scored["otp_class"].tolist() == ["on_time", "on_time"]
    assert scored["_actual_text"].iloc[1] == "2026-01-05T08:31:00"
    for row in rows:
        row["trip_end_stop_id"] = ""  # no known last stop: departure throughout
    scored, _ = target.score_otp(_visits(rows), -1.0, 5.0, True)
    assert scored["dev_min"].tolist() == [3.0, 10.0]


def test_score_otp_last_stop_needs_the_last_stop_visit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A trip missing its last-stop visit, or its trip_end_stop_id, is not treated."""
    missing_last = _loop_trip("X1")[:2]  # AVL lost the visit back at A
    no_end = _loop_trip("X1", end_stop="")
    for row in no_end:
        row["trip_id_performed"] = "T_no_end"
    visits = _visits([*missing_last, *no_end])
    scored, counts = target.score_otp(visits, -1.0, 5.0, True, "early_ok", {"X1"})
    assert set(scored["otp_class"]) == {"early"}
    assert counts["early_ok_at_last_stop"] == 0
    assert "1 performed trip(s) have no trip_end_stop_id" in caplog.text


def test_score_otp_late_last_stop_still_late() -> None:
    """early_ok never turns a late visit on time."""
    rows = _loop_trip("X1")
    rows[-1]["actual_arrival_time"] = "07:20"
    scored, _ = target.score_otp(_visits(rows), -1.0, 5.0, True, "early_ok", {"X1"})
    assert scored["otp_class"].tolist() == ["early", "early", "late"]


def test_resolve_express_routes_sources(caplog: pytest.LogCaptureFixture) -> None:
    """File and inline ids are unioned; with neither, route_type_agency decides."""
    visits = pd.DataFrame(
        {
            "route_id": ["DC_R10", "DC_R20", "DC_R30"],
            "route_type_agency": ["LOCAL", "Express", "LOCAL"],
        }
    )
    routes, source = target.resolve_express_routes(visits, str(EXPRESS_ROUTES), ["DC_R99"], ())
    assert routes == {"DC_R20", "DC_R99"}
    assert "EXPRESS_ROUTES" in source and "express_routes.txt" in source
    assert "Express route(s) not in the data: DC_R99" in caplog.text
    routes, source = target.resolve_express_routes(visits, None, (), ("EXPRESS",))
    assert routes == {"DC_R20"}
    assert "route_type_agency" in source


def _express_tides(tmp_path: Path, route_type_agency: str) -> tuple[Path, Path]:
    """One DC_R20 trip, on time at A and 6 minutes early arriving at its last stop B."""
    day = "2026-01-05"  # a Monday
    visits = [
        {
            "service_date": day,
            "trip_id_performed": "T1",
            "trip_stop_sequence": seq,
            "stop_id": stop,
            "timepoint": "TRUE",
            "schedule_relationship": "Scheduled",
            "schedule_arrival_time": _ts(day, sched),
            "schedule_departure_time": _ts(day, sched),
            "actual_arrival_time": _ts(day, actual),
            "actual_departure_time": _ts(day, actual),
        }
        for seq, stop, sched, actual in ((1, "A", "08:00", "08:00"), (2, "B", "08:30", "08:24"))
    ]
    trips = [
        {
            "service_date": day,
            "trip_id_performed": "T1",
            "trip_id_scheduled": "S1",
            "route_id": "DC_R20",
            "direction_id": "0",
            "route_type_agency": route_type_agency,
            "trip_end_stop_id": "B",
            "trip_type": "In service",
            "schedule_relationship": "Scheduled",
        }
    ]
    return _write_tides(tmp_path, visits, trips)


@pytest.mark.parametrize(
    ("overrides", "early", "on_time", "evaluated"),
    [
        ({}, 1, 1, 2),
        ({"last_stop_treatment": "early_ok", "express_routes_file": str(EXPRESS_ROUTES)}, 0, 2, 2),
        ({"last_stop_treatment": "exclude", "express_routes_file": str(EXPRESS_ROUTES)}, 0, 1, 1),
        ({"last_stop_treatment": "early_ok"}, 1, 1, 2),  # LOCAL by route_type_agency, no file
        ({"last_stop_treatment": "early_ok", "last_stop_routes": "all"}, 0, 2, 2),
    ],
)
def test_run_last_stop_treatment_reaches_the_route_tables(
    tmp_path: Path, overrides: dict, early: int, on_time: int, evaluated: int
) -> None:
    """The express-routes fixture, not route_type_agency, marks DC_R20 express here."""
    sv, tp = _express_tides(tmp_path, route_type_agency="LOCAL")
    settings = _settings(
        tmp_path / "out",
        stop_visits_path=sv,
        trips_performed_path=tp,
        measures=["otp"],
        levels=["route"],
        time_cuts=["all_day"],
        **overrides,
    )
    table = target.run(settings).otp[("route", "all_day")]
    assert table[["early", "on_time", "evaluated"]].iloc[0].tolist() == [early, on_time, evaluated]


@pytest.mark.parametrize(
    ("field", "value"), [("last_stop_treatment", "forgive"), ("last_stop_routes", "local")]
)
def test_run_rejects_unknown_last_stop_settings(tmp_path: Path, field: str, value: str) -> None:
    """A misspelled treatment or route choice stops the run before any work."""
    with pytest.raises(target.SettingsError, match=field.upper()):
        target.run(_settings(tmp_path, **{field: value}))


def test_route_month_otp_matches_a_direct_join(full_run: target.RunResult) -> None:
    """Route x month counts (all day types) equal an independent scoring of the fixtures."""
    sv = pd.read_csv(STOP_VISITS, dtype=str)
    tp = pd.read_csv(TRIPS_PERFORMED, dtype=str)
    sv["service_date"] = pd.to_datetime(sv["service_date"])
    tp["service_date"] = pd.to_datetime(tp["service_date"])
    tp = tp[(tp["trip_type"] == "In service") & (tp["schedule_relationship"] != "Canceled")]
    joined = sv.merge(tp[["service_date", "trip_id_performed", "route_id", "trip_end_stop_id"]])
    seq = pd.to_numeric(joined["trip_stop_sequence"])
    last_seq = seq.groupby([joined["service_date"], joined["trip_id_performed"]]).transform("max")
    at_last = seq.eq(last_seq) & joined["stop_id"].eq(joined["trip_end_stop_id"])
    joined = joined[
        (joined["timepoint"] == "TRUE") & (joined["schedule_relationship"] == "Scheduled")
    ]
    at_last = at_last[joined.index]

    def first(a: str, b: str) -> pd.Series:
        return pd.to_datetime(joined[a]).fillna(pd.to_datetime(joined[b]))

    # Departure, else arrival -- except arrival, else departure, at a trip's last stop.
    sched = first("schedule_departure_time", "schedule_arrival_time").where(
        ~at_last, first("schedule_arrival_time", "schedule_departure_time")
    )
    actual = first("actual_departure_time", "actual_arrival_time").where(
        ~at_last, first("actual_arrival_time", "actual_departure_time")
    )
    dev = (actual - sched).dt.total_seconds() / 60
    joined = joined.assign(
        month=joined["service_date"].dt.strftime("%Y-%m"),
        on_time=dev.between(-1, 5),
        evaluated=dev.notna(),
    )
    expected = joined.groupby(["route_id", "month"])[["on_time", "evaluated"]].sum()
    table = full_run.otp[("route", "month")]
    got = table.groupby(["route_id", "month"])[["on_time", "evaluated"]].sum()
    pd.testing.assert_frame_equal(got.astype(int), expected.astype(int))


def test_every_otp_table_reconciles(full_run: target.RunResult) -> None:
    """Class counts add up to evaluated and the three shares to 100 in every table."""
    for (level, cut), table in full_run.otp.items():
        assert not table.empty, (level, cut)
        assert (table["early"] + table["on_time"] + table["late"] == table["evaluated"]).all()
        shares = table["pct_on_time"] + table["pct_early"] + table["pct_late"]
        assert shares.between(99.8, 100.2).all(), (level, cut)


def test_holiday_from_gtfs_is_its_own_day_type(full_run: target.RunResult) -> None:
    """Jan 1 runs holiday-only service, so it leaves Thursday's Weekday day type."""
    system = full_run.otp[("system", "all_day")].set_index("day_type")
    assert list(system.index) == ["Weekday", "Saturday", "Sunday", "Holiday"]
    assert system.loc["Holiday", "days_observed"] == 1
    route = full_run.otp[("route", "all_day")]
    assert set(route.loc[route["route_id"] == "DC_R50H", "day_type"]) == {"Holiday"}


def test_trip_otp_rows_carry_their_scheduled_start(full_run: target.RunResult) -> None:
    """Trip rows show a service-day start (owl trips past 24:00) and run in start order."""
    trips = full_run.otp[("trip", "all_day")]
    assert trips["scheduled_start"].str.match(r"^\d{2}:\d{2}$").all()
    assert (trips["scheduled_start"] >= "24:00").any()
    for _, group in trips.groupby(["day_type", "route_id", "direction_id"]):
        assert group["scheduled_start"].is_monotonic_increasing


def test_hours_and_weekdays_use_their_own_order(full_run: target.RunResult) -> None:
    """Hours use the service-day clock; weekday rows follow Monday ... Sunday."""
    hours = full_run.otp[("system", "hour")]
    assert hours["hour"].max() >= 24
    assert hours["hour_label"].str.match(r"^\d{2}:00$").all()
    weekdays = full_run.otp[("system", "day_of_week")]
    weekday_rows = weekdays.loc[weekdays["day_type"] == "Weekday", "day_of_week"].tolist()
    order = list(target.WEEKDAY_NAMES)
    assert weekday_rows == sorted(weekday_rows, key=order.index)
    assert set(weekday_rows) <= set(order[:5])


def test_time_period_rows_carry_their_window(full_run: target.RunResult) -> None:
    """Time-period rows follow the configured order and show each window."""
    table = full_run.otp[("system", "time_period")]
    weekday = table.loc[table["day_type"] == "Weekday"]
    names = [p.name for p in DEFAULT_PERIODS]
    assert weekday["time_period"].tolist() == sorted(weekday["time_period"], key=names.index)
    night = weekday.loc[weekday["time_period"] == "Night"].iloc[0]
    assert (night["period_start"], night["period_end"], night["period_hours"]) == (
        "22:00",
        "04:00",
        6.0,
    )


# --- stop flags ----------------------------------------------------------------------


def _make_detail(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal evaluated detail frame from per-row dicts."""
    defaults = {
        "visits_emitted": 0,
        "visits_skipped": 0,
        "visits_missing_actual": 0,
        "visits_missing_schedule": 0,
        "early": 0,
        "late": 0,
    }
    detail = pd.DataFrame([{**defaults, **row} for row in rows])
    detail["pct_trips_observed"] = detail["observed_trips"] / detail["expected_trips"] * 100
    detail["pct_on_time"] = detail["on_time"] / detail["evaluated"] * 100
    return detail


def test_add_pattern_key_prefers_pattern_id() -> None:
    """pattern_id is used where filled; blanks and no column fall back to route | direction."""
    trips = pd.DataFrame(
        {"route_id": ["101", "101"], "direction_id": ["0", "1"], "pattern_id": ["P1", None]}
    )
    assert target.add_pattern_key(trips)["pattern_key"].tolist() == ["P1", "101 | 1"]
    no_column = pd.DataFrame({"route_id": ["101"], "direction_id": ["0"]})
    assert target.add_pattern_key(no_column)["pattern_key"].tolist() == ["101 | 0"]


def test_expected_trips_sums_patterns_serving_the_stop() -> None:
    """A stop's expected trips are the trips of every pattern that serves it."""
    candidates = pd.DataFrame(
        {
            "pattern_key": ["P1", "P1", "P2"],
            "route_id": ["A", "A", "B"],
            "stop_id": ["S1", "S2", "S1"],
        }
    )
    pattern_trips = pd.DataFrame(
        {"pattern_key": ["P1", "P2"], "route_id": ["A", "B"], "n_trips": [10, 5]}
    )
    lookup = target.build_expected_trips(candidates, pattern_trips).set_index(
        ["stop_id", "route_id"]
    )["expected_trips"]
    assert lookup[("S1", "A")] == 10
    assert lookup[("S1", "B")] == 5
    assert lookup[("S2", "A")] == 10
    assert ("S2", "B") not in lookup.index


def test_stop_detail_splits_visit_failure_causes() -> None:
    """Skipped, missing-actual and missing-schedule visits are told apart."""
    visits = _visits(
        [
            {"schedule_departure_time": "06:00", "actual_departure_time": "06:00"},
            {"schedule_departure_time": "06:00"},
            {"actual_departure_time": "06:00"},
            {"schedule_relationship": "Skipped", "schedule_departure_time": "06:00"},
        ]
    ).assign(pattern_key="P1")
    candidates = target.stop_flag_candidates(visits, True).assign(pattern_key="P1")
    scored, _ = target.score_otp(visits, -1.0, 5.0, True)
    trips = pd.DataFrame(
        {
            "service_date": pd.to_datetime(["2026-01-05"]),
            "trip_id_performed": ["T1"],
            "route_id": ["R1"],
            "pattern_key": ["P1"],
        }
    )
    row = target.build_stop_route_detail(candidates, scored, trips).iloc[0]
    assert (row["visits_emitted"], row["visits_skipped"]) == (4, 1)
    assert (row["visits_missing_actual"], row["visits_missing_schedule"]) == (1, 1)
    assert (row["evaluated"], row["expected_trips"], row["observed_trips"]) == (1, 1, 1)


def test_coverage_counts_each_date_of_a_reused_trip_id() -> None:
    """One of two scheduled dates observed is 50% coverage, not 100%."""
    visits = _visits(
        [
            {"stop_id": "A", "schedule_departure_time": "06:00", "actual_departure_time": "06:01"},
            {
                "stop_id": "B",
                "trip_stop_sequence": 2,
                "schedule_departure_time": "06:20",
                "actual_departure_time": "06:22",
            },
        ]
    )
    trips = pd.DataFrame(
        {
            "service_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "trip_id_performed": ["T1", "T1"],
            "route_id": ["R1", "R1"],
            "direction_id": ["0", "0"],
        }
    )
    settings = _settings(Path("."), min_expected_trips=1, min_scored_visits=1)
    scored, _ = target.score_otp(visits, -1.0, 5.0, True)
    _, detail = target.build_stop_flags(visits, scored, trips, settings)
    assert detail["expected_trips"].tolist() == [2, 2]
    assert detail["observed_trips"].tolist() == [1, 1]
    assert detail["pct_trips_observed"].tolist() == [50.0, 50.0]


def test_route_baselines_pool_the_route() -> None:
    """Baselines are route-pooled ratios and gaps are stop minus baseline."""
    detail = _make_detail(
        [
            {"stop_id": "S1", "route_id": "A", "expected_trips": 100, "observed_trips": 100,
             "evaluated": 100, "on_time": 90},
            {"stop_id": "S2", "route_id": "A", "expected_trips": 100, "observed_trips": 50,
             "evaluated": 50, "on_time": 30},
        ]
    )  # fmt: skip
    out = target.attach_route_baselines(detail)
    assert out["route_pct_trips_observed"].eq(75.0).all()  # 150 / 200
    assert out["route_pct_on_time"].eq(80.0).all()  # 120 / 150
    s2 = out.loc[out["stop_id"] == "S2"].iloc[0]
    assert (s2["coverage_gap"], s2["otp_gap"]) == (-25.0, -20.0)


def test_stop_flags_require_multiple_routes() -> None:
    """A stop bad on two routes is flagged; bad on one route is not."""
    rows = [
        {"stop_id": stop, "route_id": route, "expected_trips": 100, "observed_trips": 98,
         "evaluated": 98, "on_time": 95}
        for route, stop in (("A", "S1"), ("B", "S2"), ("C", "S3"))
    ]  # fmt: skip
    bad = {"expected_trips": 100, "observed_trips": 40, "evaluated": 40, "on_time": 10}
    rows += [{"stop_id": "SX", "route_id": r, **bad} for r in ("A", "B")]
    rows += [{"stop_id": "SY", "route_id": "C", **bad}]
    evaluated = target.evaluate_route_level_flags(
        target.attach_route_baselines(_make_detail(rows)), 10, 10
    )
    summary = target.build_stop_summary(evaluated, 2).set_index("stop_id")
    assert summary.loc["SX", "flag_reason"] == "low_coverage+poor_otp"
    assert summary.loc["SY", "flag_reason"] == ""  # only one route agrees
    assert summary.loc["S1", "flag_reason"] == ""
    assert summary.index[0] == "SX"  # flagged stops sort first


def test_absolute_floor_and_minimum_samples_protect_stops() -> None:
    """A big gap alone does not flag a stop above the floor, and small cells are never judged."""
    good = {"route_id": "A", "expected_trips": 100, "observed_trips": 100, "evaluated": 100}
    rows = [{"stop_id": s, **good, "on_time": 100} for s in ("S1", "S3", "S4")]
    rows.append({"stop_id": "S2", **good, "on_time": 80})  # 15 points under, still 80%
    detail = target.attach_route_baselines(_make_detail(rows))
    evaluated = target.evaluate_route_level_flags(detail, 10, 10, otp_abs_flag_pct=75.0)
    s2 = evaluated.loc[evaluated["stop_id"] == "S2"].iloc[0]
    assert s2["otp_gap"] <= -15.0
    assert not s2["poor_otp_route"]
    small = target.evaluate_route_level_flags(detail, 1000, 1000)
    assert not small["coverage_evaluable"].any()
    assert not small["poor_otp_route"].any()


def test_stop_detail_is_bounded_on_the_fixtures(full_run: target.RunResult) -> None:
    """Pattern-inferred expected trips bound the observed trips at every stop."""
    detail = full_run.stop_route_detail
    assert not detail.empty
    assert (detail["observed_trips"] <= detail["expected_trips"]).all()
    assert detail["pct_trips_observed"].dropna().between(0, 100).all()
    assert detail["pct_on_time"].dropna().between(0, 100).all()
    assert not full_run.stop_flags.empty


# --- trip running time -------------------------------------------------------------------


def test_runtimes_are_per_service_date_for_a_reused_trip_id() -> None:
    """Two daily 20-minute runs of T1 are two running times, not one 1,460-minute trip."""
    visits = _visits(
        [
            {"service_date": d, "trip_stop_sequence": seq, "actual_arrival_time": t,
             "actual_departure_time": t}
            for d in ("2026-01-05", "2026-01-06")
            for seq, t in ((1, "06:00"), (2, "06:20"))
        ]
    )  # fmt: skip
    runtimes = target.compute_trip_runtimes(visits)
    assert runtimes["actual_runtime_min"].tolist() == [20.0, 20.0]


def test_single_observed_stop_is_not_a_runtime() -> None:
    """One stop with dwell has no distinct endpoints, so no running time is recorded."""
    visits = _visits([{"actual_arrival_time": "06:00", "actual_departure_time": "06:02"}])
    assert target.compute_trip_runtimes(visits).empty


def test_trip_runtime_compares_the_same_stops() -> None:
    """The scheduled running time covers the same first and last served stops as the actual one."""
    visits = _visits(
        [
            {"trip_stop_sequence": 1, "schedule_departure_time": "07:00",
             "actual_departure_time": "07:02"},
            {"trip_stop_sequence": 2, "schedule_arrival_time": "07:20",
             "actual_arrival_time": "07:25"},
            {"trip_stop_sequence": 3, "schedule_relationship": "Skipped",
             "schedule_arrival_time": "07:30", "actual_arrival_time": "07:31"},
        ]
    )  # fmt: skip
    row = target.compute_trip_runtimes(visits).iloc[0]
    assert (row["actual_runtime_min"], row["scheduled_runtime_min"], row["deviation_min"]) == (
        23.0,
        20.0,
        3.0,
    )


def test_trim_outliers_partitions_rows() -> None:
    """Trimming removes the extreme tails, conserves rows, and frac 0 keeps everything."""
    df = pd.DataFrame({"trip_id": ["T"] * 100, "actual_runtime_min": list(range(100))})
    retained, outliers = target.trim_outliers(df, frac=0.05)
    assert len(retained) + len(outliers) == 100
    assert 0 not in retained["actual_runtime_min"].to_numpy()
    assert 99 not in retained["actual_runtime_min"].to_numpy()
    kept, none = target.trim_outliers(df, frac=0.0)
    assert len(kept) == 100 and none.empty


def test_trip_runtime_tables_match_the_observations(full_run: target.RunResult) -> None:
    """System rows count every kept trip of their day type; trip rows count each trip's dates."""
    obs = full_run.trip_observations
    assert not obs.empty and not full_run.trip_outliers.empty
    system = full_run.trip_runtime[("system", "all_day")]
    assert system["n_obs"].sum() == len(obs)  # day types do not overlap here
    trips = full_run.trip_runtime[("trip", "all_day")]
    per_trip = obs.groupby("trip_id").size()
    assert trips.set_index("trip_id")["n_obs"].sort_index().equals(per_trip.sort_index())
    assert (trips["actual_min_min"] <= trips["actual_median_min"]).all()
    assert (trips["actual_median_min"] <= trips["actual_max_min"]).all()


def test_trip_flags_mark_variation_and_gaps() -> None:
    """High coefficient of variation and thin trips are flagged within their day type."""
    rows = pd.DataFrame(
        {
            "day_type": ["Weekday"] * 12,
            "route_id": ["R1"] * 12,
            "direction_id": ["0"] * 12,
            "trip_id": ["STEADY"] * 5 + ["WILD"] * 6 + ["THIN"],
            "_start_min": [420.0] * 5 + [480.0] * 6 + [540.0],
            "actual_runtime_min": [30, 30, 31, 30, 29, 20, 40, 25, 45, 30, 50, 30],
        }
    )
    flags = target.build_trip_flags(rows, 0.15, 5, 0.3, ["Weekday"]).set_index("trip_id")
    assert bool(flags.loc["WILD", "high_variation"]) and not flags.loc["STEADY", "high_variation"]
    assert bool(flags.loc["THIN", "data_gap"]) and not flags.loc["WILD", "data_gap"]
    assert flags.loc["WILD", "scheduled_start"] == "08:00"


def test_dow_flags_mark_a_thin_weekday() -> None:
    """A weekday observed far less than the trip's other weekdays is low_count."""
    days = ["Monday"] * 5 + ["Tuesday"] * 5 + ["Wednesday"]
    rows = pd.DataFrame(
        {
            "day_type": "Weekday",
            "route_id": "R1",
            "direction_id": "0",
            "trip_id": "S1",
            "_start_min": 420.0,
            "day_of_week": days,
            "actual_runtime_min": [30.0] * 10 + [40.0],
        }
    )
    flags = target.build_dow_flags(rows, 0.4, 0.1, ["Weekday"]).set_index("day_of_week")
    assert flags.index.tolist() == ["Monday", "Tuesday", "Wednesday"]
    assert flags["low_count"].tolist() == [False, False, True]
    assert bool(flags.loc["Wednesday", "runtime_anomaly"])


# --- segment running time ------------------------------------------------------------------


def test_build_segments_bridges_skipped_timepoints() -> None:
    """Segments join consecutive served timepoints; a Skipped timepoint is bridged."""
    visits = _visits(
        [
            {"trip_stop_sequence": 1, "stop_id": "A", "schedule_departure_time": "07:00",
             "actual_departure_time": "07:01"},
            {"trip_stop_sequence": 2, "stop_id": "X", "timepoint": "FALSE",
             "schedule_arrival_time": "07:05", "actual_arrival_time": "07:06",
             "schedule_departure_time": "07:05", "actual_departure_time": "07:06"},
            {"trip_stop_sequence": 3, "stop_id": "B", "schedule_relationship": "Skipped",
             "schedule_arrival_time": "07:10", "schedule_departure_time": "07:10"},
            {"trip_stop_sequence": 4, "stop_id": "C", "schedule_arrival_time": "07:20",
             "actual_arrival_time": "07:24", "schedule_departure_time": "07:20",
             "actual_departure_time": "07:25"},
        ]
    )  # fmt: skip
    segments = target.build_segments(visits, timepoints_only=True)
    assert segments["segment"].tolist() == ["A -> C"]
    row = segments.iloc[0]
    assert (row["actual_runtime_min"], row["scheduled_runtime_min"], row["deviation_min"]) == (
        23.0,
        20.0,
        3.0,
    )
    every_stop = target.build_segments(visits, timepoints_only=False)
    assert every_stop["segment"].tolist() == ["A -> X", "X -> C"]
    assert every_stop["seq"].tolist() == [1, 2]


def test_block_recovery_is_the_gap_to_the_next_trip() -> None:
    """Recovery is the next trip's scheduled start minus this one's end, within one date."""
    trips = pd.DataFrame(
        {
            "service_date": ["2026-01-05", "2026-01-05", "2026-01-05", "2026-01-06"],
            "trip_id_performed": ["A", "B", "C", "A"],
            "block_id": ["BLK"] * 4,
            "schedule_relationship": ["Scheduled", "Scheduled", "Canceled", "Scheduled"],
            "schedule_trip_start": ["2026-01-05T06:00", "2026-01-05T07:00", "2026-01-05T06:50",
                                    "2026-01-06T06:00"],
            "schedule_trip_end": ["2026-01-05T06:45", "2026-01-05T07:45", "2026-01-05T06:55",
                                  "2026-01-06T06:45"],
        }
    )  # fmt: skip
    rec = target.compute_block_recovery(trips)
    lookup = rec.set_index(["service_date", "trip_id_performed"])["recovery_after_min"]
    assert lookup[(pd.Timestamp("2026-01-05"), "A")] == 15.0  # the canceled trip is skipped
    assert pd.isna(lookup[(pd.Timestamp("2026-01-05"), "B")])
    assert pd.isna(lookup[(pd.Timestamp("2026-01-06"), "A")])  # no overnight recovery
    assert target.compute_block_recovery(trips.drop(columns="block_id")).empty


def test_runtime_percentiles_match_pandas() -> None:
    """Percentile, median, mean and modal-schedule columns follow pandas arithmetic."""
    values = [10.0, 12.0, 14.0, 16.0, 30.0]
    rows = pd.DataFrame(
        {
            "day_type": "Weekday",
            "route_id": "R1",
            "direction_id": "0",
            "from_stop_id": "A",
            "to_stop_id": "B",
            "segment": "A -> B",
            "seq": 1,
            "service_date": pd.to_datetime("2026-01-05"),
            "actual_runtime_min": values,
            "scheduled_runtime_min": [12.0, 12.0, 12.0, 13.0, 13.0],
            "deviation_min": np.array(values) - 12.0,
            "recovery_after_min": [5.0] * 5,
        }
    )
    row = target.summarize_runtime(
        rows, "segment", "all_day", DEFAULT_PERIODS, ["Weekday"], (85,)
    ).iloc[0]
    series = pd.Series(values)
    assert row["actual_mean_min"] == pytest.approx(series.mean(), abs=0.01)
    assert row["actual_median_min"] == pytest.approx(series.median(), abs=0.01)
    assert row["actual_p85_min"] == pytest.approx(series.quantile(0.85), abs=0.01)
    assert row["scheduled_min"] == 12.0  # the mode
    assert row["diff_min"] == pytest.approx(series.median() - 12.0)
    assert row["recovery_after_min"] == 5.0


def test_percentile_column_labels() -> None:
    """Column names zero-pad single digits and encode decimals with underscores."""
    assert target.percentile_column(1) == "actual_p01_min"
    assert target.percentile_column(85) == "actual_p85_min"
    assert target.percentile_column(99.5) == "actual_p99_5_min"


def test_segment_tables_and_route_files(full_run: target.RunResult) -> None:
    """Segment rows count each traversal, and each route, direction and day type gets a table."""
    table = full_run.segment_runtime[("segment", "all_day")]
    assert table["n_obs"].sum() == len(full_run.segment_observations)
    assert (table["segment"] == table["from_stop_id"] + " -> " + table["to_stop_id"]).all()
    route_files = list(full_run.files[0].parent.glob("segment_runtime/by_route/*.csv"))
    assert len(route_files) == len(
        table[["day_type", "route_id", "direction_id"]].drop_duplicates()
    )


# --- grains, settings and files ------------------------------------------------------------


def test_every_grain_is_written(full_run: target.RunResult) -> None:
    """Each measure writes one file per applicable level and time cut."""
    out_dir = full_run.files[0].parent
    for measure, tables in (
        ("otp", full_run.otp),
        ("trip_runtime", full_run.trip_runtime),
        ("segment_runtime", full_run.segment_runtime),
    ):
        expected = {(lv, cut) for lv in target.MEASURE_LEVELS[measure] for cut in target.TIME_CUTS}
        assert set(tables) == expected
        for level, cut in expected:
            assert (out_dir / measure / target.grain_filename(measure, level, cut)).exists()
    assert target.grain_filename("otp", "route", "month_time_period") == (
        "otp_by_route_and_month_and_time_period.csv"
    )


def test_selected_grains_limit_what_is_written(tmp_path: Path) -> None:
    """Only the chosen measure, level and cut are written; stop flags can be switched off."""
    result = target.run(
        _settings(
            tmp_path,
            measures=["otp"],
            levels=["route", "segment"],
            time_cuts=["month"],
            export_stop_flags=False,
        )
    )
    assert set(result.otp) == {("route", "month")}
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.csv"))
    assert written == ["date_classification.csv", "otp/otp_by_route_and_month.csv"]
    assert result.stop_flags.empty


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"trip_time_basis": "midpoint"}, "TRIP_TIME_BASIS"),
        ({"stop_time_basis": "arrival"}, "STOP_TIME_BASIS"),
        ({"early_min": 3.0, "late_min": 1.0}, "EARLY_MIN"),
        ({"trim_frac": 0.6}, "TRIM_FRAC"),
        ({"percentiles": (50, 101)}, "RUNTIME_PERCENTILES"),
        ({"measures": ["segment_runtime"], "levels": ["route"]}, "None of the levels"),
        ({"measures": ["speed"]}, "unknown entries"),
        ({"chart_levels": ("trip",)}, "cannot be charted"),
    ],
)
def test_unusable_settings_are_reported(tmp_path: Path, overrides: dict, fragment: str) -> None:
    with pytest.raises(target.SettingsError, match=fragment):
        target.run(_settings(tmp_path, **overrides))


def test_route_filter_narrows_every_table(tmp_path: Path) -> None:
    result = target.run(
        _settings(tmp_path, levels=["route", "trip", "segment"], time_cuts=["all_day"],
                  routes_to_include=("DC_R10",))
    )  # fmt: skip
    for tables in (result.otp, result.trip_runtime, result.segment_runtime):
        for table in tables.values():
            assert set(table["route_id"]) == {"DC_R10"}


def test_charts_are_written(tmp_path: Path) -> None:
    """Summary, per-trip OTP and box-plot charts are drawn for the charted levels."""
    target.run(
        _settings(
            tmp_path,
            measures=["otp", "trip_runtime"],
            levels=["system", "trip"],
            time_cuts=["all_day", "time_period"],
            export_charts=True,
        )
    )
    charts = tmp_path / "charts"
    assert list((charts / "otp" / "system_by_time_period").glob("*.png"))
    assert list((charts / "trip_runtime" / "system_by_time_period").glob("*.png"))
    assert list((charts / "otp_trips").glob("*.png"))
    assert list((charts / "trip_runtime_boxplots").glob("*.png"))


def test_main_guards_placeholders() -> None:
    assert target.main([]) == 2


def test_main_end_to_end_with_a_periods_file(tmp_path: Path) -> None:
    """A CLI run with the periods fixture and basis flags writes a full run log."""
    code = target.main(
        [
            "--stop-visits",
            str(STOP_VISITS),
            "--trips-performed",
            str(TRIPS_PERFORMED),
            "--output-dir",
            str(tmp_path),
            "--time-periods-file",
            str(PERIODS_FIXTURE),
            "--trip-time-basis",
            "actual_start",
            "--measures",
            "otp",
            "trip_runtime",
            "--levels",
            "route",
            "--time-cuts",
            "time_period",
            "--holiday",
            HOLIDAY.isoformat(),
            "--percentiles",
            "50,90",
            "--no-charts",
        ]
    )
    assert code == 0
    table = pd.read_csv(tmp_path / "trip_runtime" / "trip_runtime_by_route_and_time_period.csv")
    assert {"actual_p50_min", "actual_p90_min"} <= set(table.columns)
    assert "Holiday" in set(table["day_type"])
    log = (tmp_path / target.RUN_LOG_FILENAME).read_text("utf-8")
    assert "time_periods.txt" in log
    assert "AM Peak = 06:00 - 09:00" in log
    assert "Trip time basis:    actual_start" in log
    assert "Measures:           otp, trip_runtime" in log
    assert "# === BEGIN CONFIG ===" not in log
    assert "TRIP_TIME_BASIS: str" in log
