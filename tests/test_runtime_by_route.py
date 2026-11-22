"""Tests for runtime_by_route using the repo TIDES fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

import runtime_by_route as target  # noqa: E402

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")


@pytest.fixture()
def joined() -> pd.DataFrame:
    """Per-trip runtimes joined with route context."""
    sv = target.load_stop_visits(STOP_VISITS)
    tp = target.load_trips_performed(TRIPS_PERFORMED)
    runtimes = target.compute_trip_runtimes(sv)
    return target.join_route_attributes(runtimes, tp)


def test_join_adds_route_and_trip_key(joined: pd.DataFrame) -> None:
    """The join attaches route_id and a trip key, with positive runtimes."""
    assert not joined.empty
    for col in ("route_id", "trip_key", "actual_runtime_min"):
        assert col in joined.columns
    assert (joined["actual_runtime_min"] > 0).all()
    assert set(joined["route_id"].unique()) <= {"DC_R10", "DC_R20", "DC_R30", "DC_R50H"}


def test_runtimes_are_per_service_date_for_a_reused_trip_id() -> None:
    """Two daily 20-minute runs of T1 are two runtimes; a lone stop is none."""
    dates = ["2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07", "2025-01-08"]
    times = ["06:00", "06:20", "06:00", "06:20", "06:00"]
    sv = pd.DataFrame(
        {
            "service_date": pd.to_datetime(dates),
            "trip_id_performed": ["T1"] * 5,
            "trip_stop_sequence": [1, 2, 1, 2, 1],
            "actual_arrival_time": pd.to_datetime([f"{d} {t}" for d, t in zip(dates, times)]),
            "actual_departure_time": pd.to_datetime([f"{d} {t}" for d, t in zip(dates, times)]),
        }
    )
    trips = pd.DataFrame(
        {
            "service_date": ["2025-01-06", "2025-01-07", "2025-01-08"],
            "trip_id_performed": ["T1"] * 3,
            "route_id": ["101"] * 3,
        }
    )
    joined = target.join_route_attributes(target.compute_trip_runtimes(sv), trips)
    assert joined["actual_runtime_min"].tolist() == [20.0, 20.0]


def test_aggregate_route_month_schema(joined: pd.DataFrame) -> None:
    """Panel is one row per (route, month) with the runtime statistics."""
    panel = target.aggregate_route_month(target.add_month(joined))
    assert list(panel.columns) == [
        "route_id",
        "month",
        "n_obs",
        "runtime_mean_min",
        "runtime_median_min",
        "runtime_std_min",
    ]
    # No duplicate (route, month) cells.
    assert not panel.duplicated(subset=["route_id", "month"]).any()


def test_aggregate_empty_returns_schema() -> None:
    """An empty input still yields the panel schema (no crash downstream)."""
    panel = target.aggregate_route_month(pd.DataFrame())
    assert panel.empty
    assert "runtime_mean_min" in panel.columns


def test_select_window_trailing_and_end_month() -> None:
    """The window keeps the most recent N months, respecting END_MONTH."""
    months = ["2025-01", "2025-02", "2025-03", "2025-01"]
    assert target.select_window(months, "", 2) == ["2025-02", "2025-03"]
    assert target.select_window(months, "2025-02", 2) == ["2025-01", "2025-02"]
    assert target.select_window(months, "", 0) == ["2025-01", "2025-02", "2025-03"]


def _toy_panel() -> pd.DataFrame:
    """Route A: a light month (mean 10, 1 obs) and a heavy month (mean 20, 3 obs)."""
    return pd.DataFrame(
        {
            "route_id": ["A", "A"],
            "month": ["2025-01", "2025-02"],
            "n_obs": [1, 3],
            "runtime_mean_min": [10.0, 20.0],
            "runtime_median_min": [10.0, 20.0],
            "runtime_std_min": [0.0, 0.0],
        }
    )


def test_reduce_normalized_weights_months_equally() -> None:
    """Normalized rollup averages the monthly means: (10 + 20) / 2 = 15."""
    rollup = target.reduce_to_route(_toy_panel(), ["2025-01", "2025-02"], normalize_by_month=True)
    row = rollup.iloc[0]
    assert row["runtime_mean_min"] == pytest.approx(15.0)
    assert row["n_months"] == 2
    assert row["n_obs"] == 4
    assert bool(row["normalized"]) is True


def test_reduce_naive_weights_by_observations() -> None:
    """Naive rollup pools observations: (10*1 + 20*3) / 4 = 17.5."""
    rollup = target.reduce_to_route(_toy_panel(), ["2025-01", "2025-02"], normalize_by_month=False)
    assert rollup.iloc[0]["runtime_mean_min"] == pytest.approx(17.5)


def test_reduce_min_obs_filter_drops_sparse_cells() -> None:
    """Cells below min_obs_per_month are excluded before the rollup."""
    rollup = target.reduce_to_route(
        _toy_panel(), ["2025-01", "2025-02"], normalize_by_month=True, min_obs_per_month=2
    )
    # Only the 3-obs month survives -> mean is that month's mean.
    assert rollup.iloc[0]["runtime_mean_min"] == pytest.approx(20.0)
    assert rollup.iloc[0]["n_months"] == 1


def test_run_writes_panel_rollup_and_runlog(tmp_path: Path) -> None:
    """End-to-end run writes the panel, the rollup, and the run-log sidecar."""
    cfg = target.Config(
        stop_visits_path=STOP_VISITS,
        trips_performed_path=TRIPS_PERFORMED,
        output_dir=tmp_path,
        window_months=0,
    )
    results = target.run(cfg)
    assert set(results) == {"panel", "rollup"}

    assert (tmp_path / target.PANEL_FILENAME).exists()
    assert (tmp_path / target.ROLLUP_FILENAME).exists()
    assert (tmp_path / "runtime_by_route_runlog.txt").exists()

    rollup = pd.read_csv(tmp_path / target.ROLLUP_FILENAME)
    assert "runtime_mean_min" in rollup.columns
    assert not rollup.empty


def _write_runtime_inputs(folder: Path, scheduled_ids: dict[str, str]) -> tuple[Path, Path]:
    """One observed trip per route on 2026-01-05: 101 runs 10 min, 202 60 min, 303 30 min."""
    minutes = {"101": 10, "202": 60, "303": 30}
    visits = [
        "service_date,trip_id_performed,trip_stop_sequence,actual_arrival_time,"
        "actual_departure_time"
    ]
    trips = ["service_date,trip_id_performed,trip_id_scheduled,route_id,direction_id"]
    for route, runtime in minutes.items():
        end = f"2026-01-05T08:{runtime:02d}:00" if runtime < 60 else "2026-01-05T09:00:00"
        visits.append(f"2026-01-05,P{route},1,2026-01-05T08:00:00,2026-01-05T08:00:00")
        visits.append(f"2026-01-05,P{route},2,{end},{end}")
        trips.append(f"2026-01-05,P{route},{scheduled_ids[route]},{route},0")
    folder.mkdir()
    (folder / "stop_visits.csv").write_text("\n".join(visits) + "\n")
    (folder / "trips_performed.csv").write_text("\n".join(trips) + "\n")
    return folder / "stop_visits.csv", folder / "trips_performed.csv"


def test_blank_scheduled_trip_ids_fall_back_row_by_row(tmp_path: Path) -> None:
    """Blank scheduled ids get their own route/direction/start keys, not one shared key."""
    sv_path, tp_path = _write_runtime_inputs(tmp_path / "in", {"101": "", "202": " ", "303": "S3"})
    joined = target.join_route_attributes(
        target.compute_trip_runtimes(target.load_stop_visits(sv_path)),
        target.load_trips_performed(tp_path),
    ).set_index("route_id")
    assert joined["trip_key"].to_dict() == {"101": "101_0_08:00", "202": "202_0_08:00", "303": "S3"}


def test_run_keeps_routes_whose_scheduled_trip_ids_are_blank(tmp_path: Path) -> None:
    """Two unrelated blank-id trips are not trimmed together out of the panel."""
    sv_path, tp_path = _write_runtime_inputs(tmp_path / "in", {"101": "", "202": "", "303": "S3"})
    cfg = target.Config(
        stop_visits_path=sv_path,
        trips_performed_path=tp_path,
        output_dir=tmp_path / "out",
        window_months=0,
        min_obs_per_month=1,
    )
    panel = target.run(cfg)["panel"].set_index("route_id")
    assert sorted(panel.index) == ["101", "202", "303"]
    assert panel.loc["101", "runtime_mean_min"] == pytest.approx(10.0)
    assert panel.loc["202", "runtime_mean_min"] == pytest.approx(60.0)
