"""Tests for scripts.ridership_tools.bikeshare_ridership_trends."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless: no display needed for savefig

from scripts.gbfs_tools import bikeshare_ridership_trends as mod

FIXTURE_ZIP = Path("tests/fixtures/capitalbikeshare_fixtures_24mo.zip")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trips() -> pd.DataFrame:
    return mod.load_trips(FIXTURE_ZIP)


# ---------------------------------------------------------------------------
# load_trips
# ---------------------------------------------------------------------------


def test_load_trips_concatenates_all_months(trips: pd.DataFrame) -> None:
    assert len(trips) == 3000
    assert trips["source_file"].nunique() == 24


def test_load_trips_adds_month_column(trips: pd.DataFrame) -> None:
    assert "month" in trips.columns
    assert trips["month"].min() == "2024-05"
    assert trips["month"].max() == "2026-04"


def test_load_trips_keeps_blank_stations_as_empty_strings(trips: pd.DataFrame) -> None:
    # Dockless trips have a blank station; they must be "" not NaN.
    assert (trips["start_station_id"] == "").any()
    assert not trips["start_station_id"].isna().any()


def test_load_trips_is_sorted_by_start(trips: pd.DataFrame) -> None:
    assert trips["started_at"].is_monotonic_increasing


def test_load_trips_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_trips(tmp_path / "nope.zip")


def test_load_trips_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No"):
        mod.load_trips(tmp_path)


def test_load_trips_from_directory(tmp_path: Path, trips: pd.DataFrame) -> None:
    # Round-trip one month through a directory and confirm it loads.
    one_month = trips[trips["month"] == "2024-05"]
    cols = [c for c in trips.columns if c not in ("month", "source_file")]
    csv_path = tmp_path / "202405-capitalbikeshare-tripdata.csv"
    one_month[cols].to_csv(csv_path, index=False)
    loaded = mod.load_trips(tmp_path)
    assert len(loaded) == len(one_month)


def test_load_trips_from_nested_directories(tmp_path: Path, trips: pd.DataFrame) -> None:
    # The prep_features.py orchestrator unzips each monthly archive into its own
    # subfolder, so the CSVs land one level below the directory handed to
    # load_trips. Confirm the directory loader recurses into those subfolders.
    cols = [c for c in trips.columns if c not in ("month", "source_file")]
    months = ["2024-05", "2024-06"]
    for month in months:
        stem = f"{month.replace('-', '')}-capitalbikeshare-tripdata"
        sub = tmp_path / stem
        sub.mkdir()
        trips[trips["month"] == month][cols].to_csv(sub / f"{stem}.csv", index=False)
    loaded = mod.load_trips(tmp_path)
    assert loaded["source_file"].nunique() == len(months)
    assert len(loaded) == len(trips[trips["month"].isin(months)])


# ---------------------------------------------------------------------------
# build_system_monthly
# ---------------------------------------------------------------------------


def test_system_monthly_one_row_per_month(trips: pd.DataFrame) -> None:
    system = mod.build_system_monthly(trips)
    assert len(system) == 24
    assert system["month"].is_monotonic_increasing


def test_system_monthly_totals_reconcile(trips: pd.DataFrame) -> None:
    system = mod.build_system_monthly(trips)
    assert system["total_trips"].sum() == len(trips)
    assert (system["member_trips"] + system["casual_trips"]).equals(system["total_trips"])
    assert (system["electric_trips"] + system["classic_trips"]).equals(system["total_trips"])


# ---------------------------------------------------------------------------
# build_station_monthly
# ---------------------------------------------------------------------------


def test_station_monthly_spans_full_grid(trips: pd.DataFrame) -> None:
    station = mod.build_station_monthly(trips)
    n_months = trips["month"].nunique()
    n_stations = station["station_id"].nunique()
    assert len(station) == n_months * n_stations


def test_station_monthly_total_is_departures_plus_arrivals(trips: pd.DataFrame) -> None:
    station = mod.build_station_monthly(trips)
    assert (station["departures"] + station["arrivals"]).equals(station["total"])


def test_station_monthly_excludes_dockless(trips: pd.DataFrame) -> None:
    station = mod.build_station_monthly(trips)
    assert (station["station_id"].str.len() > 0).all()
    # Docked departures only -> per-station departures sum <= all trips.
    docked_starts = (trips["start_station_id"].str.len() > 0).sum()
    assert station["departures"].sum() == docked_starts


# ---------------------------------------------------------------------------
# generate_and_write (end to end)
# ---------------------------------------------------------------------------


def test_generate_and_write_produces_tables_and_charts(tmp_path: Path) -> None:
    result = mod.generate_and_write(
        input_path=str(FIXTURE_ZIP),
        output_dir=str(tmp_path),
        max_station_plots=0,
    )
    assert (tmp_path / "trips_concatenated.csv").exists()
    assert (tmp_path / "monthly_system_ridership.csv").exists()
    assert (tmp_path / "monthly_station_ridership.csv").exists()
    assert (tmp_path / "plots" / "system_ridership_trend.png").exists()
    station_pngs = list((tmp_path / "plots" / "stations").glob("*.png"))
    assert len(station_pngs) == result["station"]["station_id"].nunique()


def test_max_station_plots_caps_chart_count(tmp_path: Path) -> None:
    mod.generate_and_write(
        input_path=str(FIXTURE_ZIP),
        output_dir=str(tmp_path),
        max_station_plots=3,
    )
    station_pngs = list((tmp_path / "plots" / "stations").glob("*.png"))
    assert len(station_pngs) == 3
