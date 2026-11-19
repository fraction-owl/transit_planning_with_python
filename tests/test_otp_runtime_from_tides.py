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
    }
    assert scored["dev_min"].tolist() == [2.0, 10.0]
    everything, _ = target.score_otp(visits, -1.0, 5.0, False)
    assert len(everything) == 3  # the non-timepoint visit now scores too


def test_join_drops_visits_from_the_canceled_date_only() -> None:
    """A trip ID canceled on one date keeps only the other date's visits."""
    trips = pd.DataFrame(
        {
            "service_date": ["2025-01-06", "2025-01-07"],
            "trip_id_performed": ["T1", "T1"],
            "route_id": ["101", "101"],
            "schedule_relationship": ["Scheduled", "Canceled"],
        }
    )
    visits = pd.DataFrame(
        {"service_date": ["2025-01-06", "2025-01-07"], "trip_id_performed": ["T1", "T1"]}
    )
    joined = target.join_trips(visits, trips)
    assert joined["service_date"].tolist() == [pd.Timestamp("2025-01-06")]


def test_route_month_otp_matches_a_direct_join(full_run: target.RunResult) -> None:
    """Route x month counts (all day types) equal an independent scoring of the fixtures."""
    sv = pd.read_csv(STOP_VISITS, dtype=str)
    tp = pd.read_csv(TRIPS_PERFORMED, dtype=str)
    sv["service_date"] = pd.to_datetime(sv["service_date"])
    tp["service_date"] = pd.to_datetime(tp["service_date"])
    tp = tp[(tp["trip_type"] == "In service") & (tp["schedule_relationship"] != "Canceled")]
    joined = sv.merge(tp[["service_date", "trip_id_performed", "route_id"]])
    joined = joined[
        (joined["timepoint"] == "TRUE") & (joined["schedule_relationship"] == "Scheduled")
    ]
    sched = pd.to_datetime(joined["schedule_departure_time"]).fillna(
        pd.to_datetime(joined["schedule_arrival_time"])
    )
    actual = pd.to_datetime(joined["actual_departure_time"]).fillna(
        pd.to_datetime(joined["actual_arrival_time"])
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
        ({"export_monthly_reports": True, "measures": ["trip_runtime"]}, "require the otp"),
        ({"monthly_coverage_warn_pct": 120.0}, "MONTHLY_COVERAGE_WARN_PCT"),
        ({"corridors": {"North": ()}}, "nonempty sequence"),
        ({"corridors": {"North": "DC_R10"}}, "nonempty sequence"),
        ({"corridors": {" ": ("DC_R10",)}}, "nonblank and unique"),
        ({"corridors": {"North": ("DC_R10",), "North ": ("DC_R20",)}}, "nonblank and unique"),
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


# --- monthly OTP reports ------------------------------------------------------------

MONTHLY_DIR = Path(target.OTP_SUBDIR) / target.MONTHLY_SUBDIR
COUNTS = ["early", "on_time", "late", "evaluated"]
CORRIDORS = {"North": ("DC_R10", "DC_R20"), "Crosstown": ("DC_R20", "DC_R30")}


def _read_processed(monthly_dir: Path) -> pd.DataFrame:
    return pd.read_csv(monthly_dir / target.MONTHLY_PROCESSED_FILENAME, dtype={"direction_id": str})


@pytest.fixture(scope="module")
def monthly_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[target.RunResult, Path]:
    """Monthly reports at the four chartable levels with two overlapping corridors."""
    out_dir = tmp_path_factory.mktemp("monthly_run")
    result = target.run(
        _settings(
            out_dir,
            measures=["otp"],
            levels=list(target.CHARTABLE_LEVELS),
            time_cuts=["month"],
            export_stop_flags=False,
            export_monthly_reports=True,
            corridors=CORRIDORS,
            export_charts=True,
            chart_levels=(),
        )
    )
    return result, out_dir / MONTHLY_DIR


def test_monthly_tables_pool_the_day_types(monthly_run: tuple[target.RunResult, Path]) -> None:
    """Route x month counts equal the standard route x month table summed over day types."""
    result, monthly_dir = monthly_run
    standard = result.otp[("route", "month")].groupby(["route_id", "month"])[COUNTS].sum()
    processed = _read_processed(monthly_dir)
    route = processed.loc[processed["level"] == "route"].set_index(["route_id", "month"])[COUNTS]
    pd.testing.assert_frame_equal(route.sort_index(), standard.sort_index(), check_dtype=False)
    assert (processed[COUNTS[:3]].sum(axis=1) == processed["evaluated"]).all()


def test_monthly_reports_count_each_visit_once(tmp_path: Path) -> None:
    """With overlapping day types, monthly counts match one day type, not their sum."""
    weekdays = target.WEEKDAY_NAMES[:5]
    result = target.run(
        _settings(
            tmp_path,
            measures=["otp"],
            levels=["system"],
            time_cuts=["month"],
            export_stop_flags=False,
            day_types={"Weekday": weekdays, "Every day": target.WEEKDAY_NAMES},
            export_monthly_reports=True,
        )
    )
    standard = result.otp[("system", "month")]
    every_day = standard.loc[standard["day_type"] == "Every day"].set_index("month")["evaluated"]
    monthly = _read_processed(tmp_path / MONTHLY_DIR).set_index("month")["evaluated"]
    pd.testing.assert_series_equal(monthly, every_day, check_names=False)
    assert standard["evaluated"].sum() > monthly.sum()


def test_corridors_pool_their_routes(monthly_run: tuple[target.RunResult, Path]) -> None:
    """A corridor sums its routes; a route may sit in two corridors; system totals stay put."""
    _, monthly_dir = monthly_run
    processed = _read_processed(monthly_dir)
    route = processed.loc[processed["level"] == "route"]
    corridor = processed.loc[processed["level"] == "corridor"]
    for name, routes in CORRIDORS.items():
        got = corridor.loc[corridor["corridor"] == name].set_index("month")["evaluated"]
        expected = route.loc[route["route_id"].isin(routes)].groupby("month")["evaluated"].sum()
        pd.testing.assert_series_equal(got, expected, check_names=False)
    system = processed.loc[processed["level"] == "system"].set_index("month")["evaluated"]
    pd.testing.assert_series_equal(
        system, route.groupby("month")["evaluated"].sum(), check_names=False
    )


def test_monthly_tables_lead_with_the_group_keys(
    monthly_run: tuple[target.RunResult, Path],
) -> None:
    """The long table starts with level, keys and month; wide tables have a column per month."""
    _, monthly_dir = monthly_run
    header = pd.read_csv(monthly_dir / target.MONTHLY_PROCESSED_FILENAME, nrows=0).columns
    assert list(header[:6]) == [
        "level",
        "route_type_agency",
        "route_id",
        "direction_id",
        "corridor",
        "month",
    ]
    wide = pd.read_csv(monthly_dir / "otp_monthly_route_direction.csv", dtype=str)
    assert list(wide.columns) == ["route_id", "direction_id", "2026-01", "2026-02", "2026-03"]
    system = pd.read_csv(monthly_dir / "otp_monthly_system.csv")
    assert system["group"].tolist() == ["ALL"]


def test_monthly_line_charts_are_drawn(monthly_run: tuple[target.RunResult, Path]) -> None:
    """One chart per group at each level, corridors included."""
    _, monthly_dir = monthly_run
    plots = monthly_dir / "plots"
    counts = {level: len(list((plots / level).glob("*.png"))) for level in ("system", "corridor")}
    assert counts == {"system": 1, "corridor": 2}
    assert len(list((plots / "route").glob("*.png"))) == 4


def test_monthly_pivot_leaves_unobserved_months_blank() -> None:
    """A month with no visits is blank, not 0% on time; duplicate keys are refused."""
    table = pd.DataFrame(
        {"route_id": ["R1", "R1"], "month": ["2026-01", "2026-03"], "pct_on_time": [80.0, 90.0]}
    )
    wide = target.monthly_otp_pivot(table, "route", ["2026-01", "2026-02", "2026-03"])
    assert list(wide.columns) == ["route_id", "2026-01", "2026-02", "2026-03"]
    assert wide["2026-01"].tolist() == [80.0]
    assert wide["2026-02"].isna().all()
    with pytest.raises(ValueError, match="duplicate"):
        target.monthly_otp_pivot(pd.concat([table, table]), "route", ["2026-01"])


def test_monthly_trip_coverage_counts_unscored_trips() -> None:
    """Listed in-service trips without a scored visit stay in the denominator."""
    trips = pd.DataFrame(
        {
            "trip_id_performed": ["T1", "T2", "T3", "T4", "T1"],
            "route_id": ["101"] * 5,
            "service_date": ["2025-01-02"] * 4 + ["2025-01-03"],
            "schedule_relationship": [
                "Scheduled",
                "Scheduled",
                "Scheduled",
                "Canceled",
                "Scheduled",
            ],
            "trip_type": ["In service"] * 5,
        }
    )
    scored = pd.DataFrame(
        {
            "service_date": pd.to_datetime(["2025-01-02"] * 3),
            "trip_id_performed": ["T1", "T1", "T2"],
        }
    )
    coverage = target.monthly_trip_coverage(target.in_service_trips(trips), scored)
    route = coverage.loc[coverage["level"] == "route"]
    assert len(route) == 1
    row = route.iloc[0]
    assert row["month"] == "2025-01"
    assert row["trips_listed"] == 4  # T4 is Canceled; T1 counts once per date
    assert row["trips_scored"] == 2  # T3 and T1's second date have no scored visit
    assert row["pct_trips_scored"] == pytest.approx(50.0)
    assert row["evaluated_visits"] == 3
    assert row["visits_per_scored_trip"] == pytest.approx(1.5)
    system = coverage.loc[coverage["level"] == "system"].iloc[0]
    assert (system["route_id"], system["trips_listed"]) == ("ALL", 4)


def test_low_coverage_is_one_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Rows below the threshold are summed up in a single warning naming the lowest."""
    coverage = pd.DataFrame(
        {
            "level": ["system", "route", "route", "route"],
            "route_id": ["ALL", "101", "102", "103"],
            "month": ["2026-01"] * 4,
            "trips_listed": [40, 10, 10, 20],
            "trips_scored": [30, 9, 5, 16],
            "pct_trips_scored": [75.0, 90.0, 50.0, 80.0],
            "evaluated_visits": [60, 18, 10, 32],
            "visits_per_scored_trip": [2.0, 2.0, 2.0, 2.0],
        }
    )
    with caplog.at_level("WARNING"):
        target.write_monthly_reports({}, coverage, ["2026-01"], tmp_path, [], 90.0)
    warnings_logged = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings_logged) == 1
    assert "3 of 4" in warnings_logged[0]
    assert "route 102 in 2026-01, 50.0% of 10 trips" in warnings_logged[0]


def test_monthly_reports_do_not_hide_an_empty_run(tmp_path: Path) -> None:
    """With nothing scorable, the coverage table alone does not count as a result."""
    visits = pd.read_csv(STOP_VISITS, dtype=str)
    visits[["actual_arrival_time", "actual_departure_time"]] = None
    blank_path = tmp_path / "stop_visits.csv"
    visits.to_csv(blank_path, index=False)
    settings = _settings(
        tmp_path / "out", stop_visits_path=blank_path, measures=["otp"], export_monthly_reports=True
    )
    with pytest.raises(target.InputDataError, match="No table could be built"):
        target.run(settings)


def test_rerun_with_fewer_corridors_removes_their_outputs(tmp_path: Path) -> None:
    """Tables and charts of corridors dropped from the settings do not linger."""
    base = {
        "measures": ["otp"],
        "levels": ["system"],
        "time_cuts": ["all_day"],
        "export_stop_flags": False,
        "export_monthly_reports": True,
        "export_charts": True,
        "chart_levels": (),
    }
    corridor_plots = tmp_path / MONTHLY_DIR / "plots" / "corridor"
    target.run(_settings(tmp_path, corridors=CORRIDORS, **base))
    assert len(list(corridor_plots.glob("*.png"))) == 2
    target.run(_settings(tmp_path, corridors={"North": CORRIDORS["North"]}, **base))
    assert [p.name for p in corridor_plots.glob("*.png")] == ["0001_North.png"]
    wide = pd.read_csv(tmp_path / MONTHLY_DIR / "otp_monthly_corridor.csv")
    assert wide["corridor"].tolist() == ["North"]
    target.run(_settings(tmp_path, corridors={}, **base))
    assert not list(corridor_plots.glob("*.png"))
    assert not (tmp_path / MONTHLY_DIR / "otp_monthly_corridor.csv").exists()


def test_parse_corridors_reads_repeated_flags() -> None:
    assert target.parse_corridors(None) == target.CORRIDORS
    assert target.parse_corridors(["North = DC_R10, DC_R20", "South=DC_R30"]) == {
        "North": ["DC_R10", "DC_R20"],
        "South": ["DC_R30"],
    }
    for bad in (["North"], ["North="], ["=DC_R10"], ["North=DC_R10,,DC_R20"], ["A=1", "A=2"]):
        with pytest.raises(target.SettingsError, match="--corridor"):
            target.parse_corridors(bad)


def test_main_writes_monthly_reports(tmp_path: Path) -> None:
    """The CLI flags reach the run, and the run log records them in its aligned block."""
    code = target.main(
        [
            "--stop-visits",
            str(STOP_VISITS),
            "--trips-performed",
            str(TRIPS_PERFORMED),
            "--output-dir",
            str(tmp_path),
            "--measures",
            "otp",
            "--levels",
            "route",
            "--time-cuts",
            "all_day",
            "--monthly-reports",
            "--corridor",
            "North=DC_R10,DC_R20",
            "--monthly-coverage-warn-pct",
            "80",
            "--no-charts",
        ]
    )
    assert code == 0
    monthly_dir = tmp_path / MONTHLY_DIR
    assert (monthly_dir / "otp_monthly_route.csv").exists()
    assert (monthly_dir / target.MONTHLY_COVERAGE_FILENAME).exists()
    corridor = pd.read_csv(monthly_dir / "otp_monthly_corridor.csv")
    assert corridor["corridor"].tolist() == ["North"]
    log = (tmp_path / target.RUN_LOG_FILENAME).read_text("utf-8")
    assert "Monthly reports:    True; pooled included dates; coverage warning below 80%" in log
    assert "Monthly corridors:  {'North': ['DC_R10', 'DC_R20']}" in log
    assert "Monthly coverage:   denominator" in log
