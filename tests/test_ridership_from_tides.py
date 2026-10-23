"""Tests for ridership_from_tides using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import ridership_from_tides as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")

FINE_KEYS = ["month", "time_period"]


@pytest.fixture()
def prepared() -> pd.DataFrame:
    """Joined, filtered, ridership-tagged, period/month-tagged stop visits."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(sv, tp)
    return (
        joined.pipe(target.filter_for_ridership)
        .pipe(target.add_ridership_columns)
        .pipe(target.assign_time_period, target.TIME_PERIODS)
        .pipe(target.add_month)
    )


def test_join_brings_route_attributes() -> None:
    """The join attaches route/direction/service-type onto stop visits."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(sv, tp)
    for col in ("route_id", "direction_id", "route_type_agency"):
        assert col in joined.columns
    assert joined["route_id"].notna().all()
    assert set(joined["route_id"].unique()) <= {"101", "202", "303"}


def test_boardings_sum_both_doors() -> None:
    """Total boardings/alightings sum the per-door columns."""
    df = pd.DataFrame(
        {
            "boarding_1": [3.0, 0.0],
            "boarding_2": [1.0, 0.0],
            "alighting_1": [0.0, 2.0],
            "alighting_2": [0.0, 1.0],
        }
    )
    out = target.add_ridership_columns(df)
    assert out["boardings"].tolist() == [4.0, 0.0]
    assert out["alightings"].tolist() == [0.0, 3.0]


def test_filter_drops_skipped() -> None:
    """Skipped stop visits (no doors opened) are removed."""
    df = pd.DataFrame({"schedule_relationship": ["Scheduled", "Skipped", "Added"]})
    out = target.filter_for_ridership(df)
    assert "Skipped" not in set(out["schedule_relationship"])
    assert len(out) == 2


def test_assign_time_period_windows() -> None:
    """Each event lands in the window containing its time-of-day; midnight wraps."""
    df = pd.DataFrame(
        {
            "actual_departure_time": pd.to_datetime(
                [
                    "2025-01-02T07:00:00",  # AM PEAK
                    "2025-01-02T12:00:00",  # MIDDAY
                    "2025-01-02T16:00:00",  # PM PEAK
                    "2025-01-02T23:30:00",  # NIGHT (wraps)
                ]
            )
        }
    )
    out = target.assign_time_period(df, target.TIME_PERIODS)
    assert out["time_period"].tolist() == ["AM PEAK", "MIDDAY", "PM PEAK", "NIGHT"]


def test_assign_time_period_empty_is_all_day() -> None:
    """An empty TIME_PERIODS mapping labels everything ALL DAY."""
    df = pd.DataFrame({"actual_departure_time": pd.to_datetime(["2025-01-02T07:00:00"])})
    out = target.assign_time_period(df, {})
    assert out["time_period"].tolist() == ["ALL DAY"]


def test_add_month_is_year_month() -> None:
    """Month is derived as YYYY-MM from the service date."""
    df = pd.DataFrame({"service_date": pd.to_datetime(["2025-02-15"])})
    assert target.add_month(df)["month"].iloc[0] == "2025-02"


def test_resolve_grains_drops_period_when_no_windows() -> None:
    """Period-bearing grains are skipped when there are no time windows."""
    all_grains = ("month_and_period", "month", "period", "total")
    assert target.resolve_grains(all_grains, target.TIME_PERIODS) == list(all_grains)
    assert target.resolve_grains(all_grains, {}) == ["month", "total"]
    # Unknown grains are ignored.
    assert target.resolve_grains(("month", "bogus"), {}) == ["month"]


def test_aggregate_reconciles_with_raw_totals(prepared: pd.DataFrame) -> None:
    """Aggregated boardings equal the raw boardings sum (no double counting)."""
    agg = target.aggregate_ridership(prepared, ["route_id"], FINE_KEYS)
    assert agg["boardings"].sum() == pytest.approx(prepared["boardings"].sum())
    assert (agg["net_boardings"] == agg["boardings"] - agg["alightings"]).all()


def test_total_grain_is_single_row_per_group(prepared: pd.DataFrame) -> None:
    """The total grain collapses to one row per group with no temporal columns."""
    agg = target.aggregate_ridership(prepared, ["route_id"], [])
    assert "month" not in agg.columns
    assert "time_period" not in agg.columns
    assert agg["route_id"].is_unique


def test_coarser_grains_match_finest_totals(prepared: pd.DataFrame) -> None:
    """Each coarser grain sums to the same boardings as the finest grain."""
    fine = target.aggregate_ridership(prepared, ["route_id"], FINE_KEYS)
    by_month = target.aggregate_ridership(prepared, ["route_id"], ["month"])
    total = target.aggregate_ridership(prepared, ["route_id"], [])
    assert by_month["boardings"].sum() == pytest.approx(fine["boardings"].sum())
    assert total["boardings"].sum() == pytest.approx(fine["boardings"].sum())


def test_build_all_levels_and_long_table(prepared: pd.DataFrame) -> None:
    """All standard levels are produced and concatenate into a long table."""
    levels = target.build_all_levels(prepared, FINE_KEYS)
    assert set(levels) == {
        "route_stop",
        "stop",
        "route_direction",
        "route",
        "service_type",
        "overall",
    }
    long_table = target.make_long_table(levels, FINE_KEYS)
    assert {"level", "group", "month", "time_period", "boardings"} <= set(long_table.columns)
    overall = long_table.loc[long_table["level"] == "overall"]
    assert set(overall["group"].unique()) == {"ALL"}
    assert overall["boardings"].sum() == pytest.approx(prepared["boardings"].sum())


def test_by_route_and_stop_export_shape(prepared: pd.DataFrame) -> None:
    """The vendor-style export carries BOARD_ALL/ALIGHT_ALL keyed by stop."""
    levels = target.build_all_levels(prepared, FINE_KEYS)
    brs = target.build_by_route_and_stop(levels)
    for col in ("TIME_PERIOD", "ROUTE_ID", "STOP_ID", "BOARD_ALL", "ALIGHT_ALL"):
        assert col in brs.columns
    assert brs["BOARD_ALL"].sum() == pytest.approx(prepared["boardings"].sum())


def test_run_writes_all_grain_files(tmp_path: Path) -> None:
    """The end-to-end run writes one file per grain, the stop export, run log."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
        time_periods=target.TIME_PERIODS,
        export_grains=target.EXPORT_GRAINS,
    )
    grain_tables = target.run(cfg)
    assert set(grain_tables) == {"month_and_period", "month", "period", "total"}
    for grain in grain_tables:
        assert (tmp_path / target.GRAIN_FILENAME[grain]).exists()
    assert (tmp_path / target.BY_ROUTE_AND_STOP_FILENAME).exists()
    assert (tmp_path / "ridership_from_tides_runlog.txt").exists()


# ---------------------------------------------------------------------------
# Unknown APC counts and (service_date, trip_id_performed) trip identity
# ---------------------------------------------------------------------------


def _write_tides(
    tmp_path: Path, visits: list[dict[str, str]], trips: list[dict[str, str]]
) -> tuple[Path, Path]:
    """Write minimal stop_visits / trips_performed CSVs and return their paths."""
    sv_path = tmp_path / "stop_visits.csv"
    tp_path = tmp_path / "trips_performed.csv"
    pd.DataFrame(visits).to_csv(sv_path, index=False)
    pd.DataFrame(trips).to_csv(tp_path, index=False)
    return sv_path, tp_path


def _visit(
    date: str, trip: str, stop: str, b1: str, b2: str = "0", a1: str = "0", a2: str = "0"
) -> dict[str, str]:
    return {
        "service_date": date,
        "trip_id_performed": trip,
        "stop_id": stop,
        "actual_departure_time": f"{date}T07:00:00",
        "boarding_1": b1,
        "boarding_2": b2,
        "alighting_1": a1,
        "alighting_2": a2,
    }


def _trip(date: str, trip: str, route: str = "101") -> dict[str, str]:
    return {
        "service_date": date,
        "trip_id_performed": trip,
        "route_id": route,
        "direction_id": "0",
        "route_type_agency": "LOCAL",
    }


def _prepare(sv_path: Path, tp_path: Path, unused_doors: tuple[str, ...] = ()) -> pd.DataFrame:
    joined = target.join_trip_attributes(
        target.load_stop_visits(sv_path), target.load_trips_performed(tp_path)
    )
    return (
        joined.pipe(target.filter_for_ridership)
        .pipe(target.add_ridership_columns, unused_doors)
        .pipe(target.assign_time_period, target.TIME_PERIODS)
        .pipe(target.add_month)
    )


def test_load_stop_visits_keeps_missing_and_invalid_counts_unknown(tmp_path: Path) -> None:
    """Blank, non-numeric, and negative counts load as NaN, not 0; real zeros stay 0."""
    sv_path, _ = _write_tides(
        tmp_path,
        [
            _visit("2025-01-02", "T1", "S1", b1=""),
            _visit("2025-01-02", "T1", "S2", b1="n/a"),
            _visit("2025-01-02", "T1", "S3", b1="-3"),
            _visit("2025-01-02", "T1", "S4", b1="0"),
        ],
        [_trip("2025-01-02", "T1")],
    )
    sv = target.load_stop_visits(sv_path)
    assert sv["boarding_1"].isna().tolist() == [True, True, True, False]
    assert sv["boarding_1"].iloc[3] == 0.0


def test_entirely_blank_apc_is_unknown_not_observed_zero(tmp_path: Path) -> None:
    """With no APC counts at all, ridership is reported unknown, never as zero."""
    blank = {"b1": "", "b2": "", "a1": "", "a2": ""}
    sv_path, tp_path = _write_tides(
        tmp_path,
        [_visit("2025-01-02", "T1", "S1", **blank), _visit("2025-01-02", "T1", "S2", **blank)],
        [_trip("2025-01-02", "T1")],
    )
    overall = target.aggregate_ridership(_prepare(sv_path, tp_path), [], [])
    row = overall.iloc[0]
    assert pd.isna(row["boardings"])
    assert pd.isna(row["alightings"])
    assert pd.isna(row["net_boardings"])
    assert pd.isna(row["avg_boardings_per_trip"])
    assert (row["stop_visits"], row["unknown_stop_visits"], row["trips"]) == (0, 2, 0)

    out_dir = tmp_path / "out"
    target.run(
        target.Config(
            stop_visits_path=sv_path,
            trips_performed_path=tp_path,
            output_dir=out_dir,
            time_periods=target.TIME_PERIODS,
        )
    )
    total = pd.read_csv(out_dir / target.GRAIN_FILENAME["total"])
    assert total.loc[total["level"] == "overall", "boardings"].isna().all()
    run_log = (out_dir / "ridership_from_tides_runlog.txt").read_text(encoding="utf-8")
    assert "Total boardings:    unknown" in run_log


def test_observed_zero_is_kept_apart_from_unknown(tmp_path: Path) -> None:
    """An observed zero counts as data; an unknown visit stays out of sums and denominators."""
    sv_path, tp_path = _write_tides(
        tmp_path,
        [
            _visit("2025-01-02", "T1", "S1", b1="0"),
            _visit("2025-01-02", "T2", "S1", b1="", b2=""),
            _visit("2025-01-02", "T2", "S2", b1="6"),
        ],
        [_trip("2025-01-02", "T1"), _trip("2025-01-02", "T2")],
    )
    by_stop = target.aggregate_ridership(_prepare(sv_path, tp_path), ["stop_id"], [])
    s1 = by_stop.set_index("stop_id").loc["S1"]
    # S1: one observed zero (T1) plus one unknown visit (T2) -> 0 boardings over 1 trip.
    assert s1["boardings"] == 0.0
    assert (s1["stop_visits"], s1["unknown_stop_visits"], s1["trips"]) == (1, 1, 1)
    assert s1["avg_boardings_per_trip"] == 0.0


def test_unused_door_counters_are_skipped_explicitly(tmp_path: Path) -> None:
    """A door declared unused is left out; an undeclared blank door makes counts unknown."""
    sv_path, tp_path = _write_tides(
        tmp_path,
        [_visit("2025-01-02", "T1", "S1", b1="4", b2="", a1="1", a2="")],
        [_trip("2025-01-02", "T1")],
    )
    undeclared = _prepare(sv_path, tp_path)
    assert undeclared["boardings"].isna().all()

    declared = _prepare(sv_path, tp_path, ("boarding_2", "alighting_2"))
    assert declared["boardings"].tolist() == [4.0]
    assert declared["alightings"].tolist() == [1.0]

    with pytest.raises(ValueError):
        target.counted_door_columns(("boarding_3",))
    with pytest.raises(ValueError):
        target.counted_door_columns(("boarding_1", "boarding_2"))


def test_same_trip_id_on_two_dates_is_two_trips(tmp_path: Path) -> None:
    """A trip_id_performed reused on two dates counts as two trips, each joined to its date."""
    sv_path, tp_path = _write_tides(
        tmp_path,
        [_visit("2025-01-02", "T1", "S1", b1="10"), _visit("2025-01-03", "T1", "S1", b1="10")],
        [_trip("2025-01-02", "T1", route="101"), _trip("2025-01-03", "T1", route="202")],
    )
    prepared = _prepare(sv_path, tp_path)
    overall = target.aggregate_ridership(prepared, [], []).iloc[0]
    assert overall["boardings"] == 20.0
    assert overall["trips"] == 2
    assert overall["avg_boardings_per_trip"] == 10.0

    # Each date's visit takes that date's trip attributes.
    by_route = target.aggregate_ridership(prepared, ["route_id"], []).set_index("route_id")
    assert by_route["boardings"].to_dict() == {"101": 10.0, "202": 10.0}
    assert by_route["trips"].to_dict() == {"101": 1, "202": 1}
