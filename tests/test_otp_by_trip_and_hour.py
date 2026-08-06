"""Tests for otp_by_trip_and_hour using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import otp_by_trip_and_hour as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


@pytest.fixture()
def scored() -> pd.DataFrame:
    """Joined, deviated, filtered, classified, hour- and day-typed stop visits."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    joined = target.join_trip_attributes(sv, tp)
    prepared = (
        joined.pipe(target.compute_stop_deviations)
        .pipe(target.filter_for_otp, True)
        .pipe(target.classify_otp, target.EARLY_MIN, target.LATE_MIN)
        .pipe(target.add_service_hour)
        .pipe(target.assign_trip_id)
    )
    return target.add_day_type(prepared, target.normalize_day_types(target.DAY_TYPES))


# =============================================================================
# DAY-TYPE BUCKETS
# =============================================================================


def test_parse_day_types_spec_roundtrips_default() -> None:
    """The serialized DAY_TYPES default parses back to Mon-Fri / Sat / Sun."""
    spec = target.serialize_day_types(target.DAY_TYPES)
    buckets = target.parse_day_types_spec(spec)
    assert buckets == {"Weekday": [0, 1, 2, 3, 4], "Saturday": [5], "Sunday": [6]}


def test_parse_day_types_spec_accepts_abbreviations() -> None:
    """3-letter tokens (any case) resolve to the same weekday indices."""
    buckets = target.parse_day_types_spec("Weekend=SAT,sun;Midweek=Tue,Wed,Thu")
    assert buckets == {"Weekend": [5, 6], "Midweek": [1, 2, 3]}


def test_parse_day_types_spec_rejects_bad_input() -> None:
    """Unknown weekday tokens, malformed entries, and empty specs fail loudly."""
    with pytest.raises(ValueError, match="Unknown weekday"):
        target.parse_day_types_spec("Weekday=Mon,Funday")
    with pytest.raises(ValueError, match="expected 'Name="):
        target.parse_day_types_spec("Weekday")
    with pytest.raises(ValueError, match="at least one"):
        target.parse_day_types_spec("")
    with pytest.raises(ValueError, match="Duplicate"):
        target.parse_day_types_spec("A=Mon;A=Tue")


def test_add_day_type_buckets_overlap_and_drop() -> None:
    """Overlapping buckets duplicate rows; uncovered weekdays are dropped."""
    df = pd.DataFrame(
        {
            # 2025-01-04 = Saturday, 2025-01-05 = Sunday, 2025-01-06 = Monday
            "service_date": pd.to_datetime(["2025-01-04", "2025-01-05", "2025-01-06"]),
            "value": [1, 2, 3],
        }
    )
    buckets = target.normalize_day_types(
        {"Saturday": ("Saturday",), "Sunday": ("Sunday",), "Weekend": ("Sat", "Sun")}
    )
    out = target.add_day_type(df, buckets)
    # Sat and Sun rows each land in their own bucket plus the pooled Weekend;
    # the Monday row is in no bucket and disappears.
    assert len(out) == 4
    assert out.loc[out["day_type"] == "Weekend", "value"].tolist() == [1, 2]
    assert 3 not in out["value"].tolist()


# =============================================================================
# SCORING & SERVICE-DAY HOUR
# =============================================================================


def test_classify_otp_buckets() -> None:
    """Classification respects the inclusive on-time window."""
    df = pd.DataFrame({"dev_min": [-5.0, -1.0, 0.0, 5.0, 7.0]})
    out = target.classify_otp(df, early_min=-1.0, late_min=5.0)
    assert out["otp_class"].tolist() == ["early", "on_time", "on_time", "on_time", "late"]


def test_add_service_hour_handles_post_midnight() -> None:
    """Hours count from service-date midnight, so owl trips land at 24+."""
    service = pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-03"])
    sched = pd.to_datetime(["2025-01-03T04:15:00", "2025-01-04T01:30:00", "2025-01-05T09:00:00"])
    df = pd.DataFrame({"service_date": service, "sched_used_time": sched})
    out = target.add_service_hour(df)
    assert out["hour"].tolist()[:2] == [4, 25]
    # 57 hours past service_date is out of range: blanked, not misbucketed.
    assert pd.isna(out["hour"].iloc[2])


def test_assign_trip_id_prefers_scheduled_id() -> None:
    """trip_id_scheduled wins where present; performed IDs fill the blanks."""
    df = pd.DataFrame(
        {
            "trip_id_performed": ["P1", "P2"],
            "trip_id_scheduled": ["S1", pd.NA],
        }
    )
    out = target.assign_trip_id(df)
    assert out["trip_id"].tolist() == ["S1", "P2"]


# =============================================================================
# AGGREGATION
# =============================================================================


def test_build_trip_table_counts_reconcile(scored: pd.DataFrame) -> None:
    """Per-trip class counts sum to evaluated and pool across service dates."""
    table = target.build_trip_table(scored, list(target.DAY_TYPES))
    assert not table.empty
    assert (table["early"] + table["on_time"] + table["late"] == table["evaluated"]).all()
    # Fixture trips repeat across dates via trip_id_scheduled.
    assert table["n_dates"].max() > 1
    assert table["trip_id"].str.startswith("TRIP_").all()
    # Scheduled starts render as (possibly 24+) service-day clock strings.
    assert table["scheduled_start"].str.match(r"^\d{2}:\d{2}$").all()
    # Sorted by scheduled start within each day type.
    for _, g in table.groupby("day_type"):
        assert g["scheduled_start_minutes"].is_monotonic_increasing


def test_build_hourly_table_percentages_sum_to_100(scored: pd.DataFrame) -> None:
    """Per-hour early/on-time/late percentages add to 100 and hours are sane."""
    hourly = target.build_hourly_table(scored, list(target.DAY_TYPES))
    assert not hourly.empty
    pct_sum = hourly["pct_on_time"] + hourly["pct_early"] + hourly["pct_late"]
    assert pct_sum.round(6).eq(100.0).all()
    assert hourly["hour"].between(0, 35).all()
    # The fixtures include owl trips scheduled past midnight (service-day 24+).
    assert hourly["hour"].max() >= 24


def test_build_day_type_table_summarizes(scored: pd.DataFrame) -> None:
    """The day-type summary carries dates/trips observed and OTP splits."""
    table = target.build_day_type_table(scored, list(target.DAY_TYPES))
    # The fixtures contain weekday service only.
    assert table["day_type"].tolist() == ["Weekday"]
    row = table.iloc[0]
    assert row["n_dates"] > 1
    assert row["n_trips"] > 1
    assert row["early"] + row["on_time"] + row["late"] == row["evaluated"]


# =============================================================================
# END-TO-END
# =============================================================================


def test_run_respects_route_filter(tmp_path: Path) -> None:
    """A route include-filter narrows every output to that route's trips."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        output_dir=tmp_path,
        trips_performed_path=TRIPS_PERFORMED,
        day_types=target.normalize_day_types(target.DAY_TYPES),
        routes_to_include=("101",),
        make_plots=False,
    )
    tables = target.run(cfg)
    assert set(tables["trip"]["route_id"].unique()) == {"101"}


def test_run_writes_tables_charts_and_runlog(tmp_path: Path) -> None:
    """End-to-end run produces the three CSVs, PNG charts, and a run log."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        output_dir=tmp_path,
        trips_performed_path=TRIPS_PERFORMED,
        day_types=target.normalize_day_types(target.DAY_TYPES),
    )
    tables = target.run(cfg)
    assert set(tables) == {"trip", "hourly", "day_type"}

    assert (tmp_path / target.TRIP_FILENAME).exists()
    assert (tmp_path / target.HOURLY_FILENAME).exists()
    assert (tmp_path / target.DAY_TYPE_FILENAME).exists()
    pngs = list((tmp_path / "plots").glob("*.png"))
    assert pngs, "expected at least one OTP chart"

    runlog = tmp_path / "otp_by_trip_and_hour_runlog.txt"
    assert runlog.exists()
    text = runlog.read_text(encoding="utf-8")
    assert "OTP BY TRIP AND HOUR RUN LOG" in text
    assert "DAY_TYPES" in text  # verbatim CONFIGURATION block is embedded


def test_main_guards_placeholders_and_route_filter(tmp_path: Path) -> None:
    """main() exits 2 on placeholder paths or a route filter without trips."""
    assert target.main([]) == 2  # STOP_VISITS_PATH still a placeholder
    assert (
        target.main(
            [
                "--stop-visits",
                str(STOP_VISITS),
                "--output-dir",
                str(tmp_path),
                "--routes",
                "101",
            ]
        )
        == 2
    )


def test_main_runs_stop_visits_only(tmp_path: Path) -> None:
    """Without trips_performed the pipeline still runs on stop_visits alone."""
    rc = target.main(
        [
            "--stop-visits",
            str(STOP_VISITS),
            "--output-dir",
            str(tmp_path),
            "--no-plots",
        ]
    )
    assert rc == 0
    trip = pd.read_csv(tmp_path / target.TRIP_FILENAME)
    # No trips_performed join: trip identity falls back to trip_id_performed
    # and no route column is available.
    assert "route_id" not in trip.columns
    assert trip["trip_id"].str.startswith("TP").all()
