"""Tests for scripts.ridership_tools.bikeshare_ridership_trends."""

from __future__ import annotations

import datetime as dt
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
    # The prep_features_public.py orchestrator unzips each monthly archive into its own
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


def test_load_trips_tolerates_non_utf8_extract(tmp_path: Path, trips: pd.DataFrame) -> None:
    # Some vendor months carry a stray Windows-1252 byte (e.g. 0x9c) that a
    # strict UTF-8 read would choke on, aborting the whole run. Write one month
    # with a cp1252-only station name and confirm it still loads.
    one_month = trips[trips["month"] == "2024-05"].copy()
    cols = [c for c in trips.columns if c not in ("month", "source_file")]
    odd_name = "Cœur Plaza"  # 'œ' encodes to the lone byte 0x9c in cp1252
    one_month.loc[one_month.index[0], "start_station_name"] = odd_name
    csv_path = tmp_path / "202405-capitalbikeshare-tripdata.csv"
    csv_path.write_bytes(one_month[cols].to_csv(index=False).encode("cp1252"))

    # A strict UTF-8 read of these bytes must fail -- proving the fixture is the
    # problematic case the loader has to survive.
    with pytest.raises(UnicodeDecodeError):
        csv_path.read_bytes().decode("utf-8")

    loaded = mod.load_trips(tmp_path)
    assert len(loaded) == len(one_month)
    assert odd_name in set(loaded["start_station_name"])


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


# ---------------------------------------------------------------------------
# build_station_daytype_averages
# ---------------------------------------------------------------------------

# The 24-month fixture spans 2024-05-01 .. 2026-04-30: 731 - 1 = 730 calendar
# days. Of those, 104 are Saturdays (none holiday-shifted), 104 are true
# Sundays, and 22 are observed federal holidays falling on a weekday, which
# the holiday rule reclassifies as Sunday-equivalent: 500 / 104 / 126.
_EXPECTED_DAYS = {"weekday": 500, "saturday": 104, "sunday": 126}


def test_daytype_day_counts_match_covered_calendar(trips: pd.DataFrame) -> None:
    daytype = mod.build_station_daytype_averages(trips)
    assert daytype["weekday_days"].eq(_EXPECTED_DAYS["weekday"]).all()
    assert daytype["saturday_days"].eq(_EXPECTED_DAYS["saturday"]).all()
    assert daytype["sunday_days"].eq(_EXPECTED_DAYS["sunday"]).all()


def test_daytype_one_row_per_station(trips: pd.DataFrame) -> None:
    daytype = mod.build_station_daytype_averages(trips)
    station = mod.build_station_monthly(trips)
    assert sorted(daytype["station_id"]) == sorted(station["station_id"].unique())
    assert daytype["station_id"].is_monotonic_increasing


def test_daytype_averages_reconcile_with_total_activity(trips: pd.DataFrame) -> None:
    # Un-averaging (avg x day count) must recover every docked departure and
    # arrival, up to the 4-decimal rounding of the averages.
    daytype = mod.build_station_daytype_averages(trips)
    recovered = sum(
        (daytype[f"avg_{day_type}_riders"] * daytype[f"{day_type}_days"]).sum()
        for day_type in mod.DAY_TYPES
    )
    docked = (trips["start_station_id"].str.len() > 0).sum() + (
        trips["end_station_id"].str.len() > 0
    ).sum()
    assert abs(recovered - docked) < 0.5


def test_daytype_holiday_weekday_counts_as_sunday(trips: pd.DataFrame) -> None:
    # July 4, 2025 was a Friday. Trips starting that day must land in the
    # Sunday bucket, not the weekday bucket.
    started = pd.to_datetime(trips["started_at"])
    holiday_trips = trips[started.dt.date == dt.date(2025, 7, 4)]
    docked = holiday_trips[holiday_trips["start_station_id"].str.len() > 0]
    assert not docked.empty, "fixture unexpectedly has no docked trips on 2025-07-04"

    daytype = mod.build_station_daytype_averages(trips)
    weekday_trips = trips[
        started.dt.date.map(
            lambda day: mod._day_type(
                day,
                set().union(*(mod.federal_holidays_observed(y) for y in range(2024, 2028))),
            )
        )
        == "weekday"
    ]
    docked_weekday = (weekday_trips["start_station_id"].str.len() > 0).sum() + (
        weekday_trips["end_station_id"].str.len() > 0
    ).sum()
    recovered_weekday = (daytype["avg_weekday_riders"] * daytype["weekday_days"]).sum()
    assert abs(recovered_weekday - docked_weekday) < 0.5


def test_daytype_excludes_dockless(tmp_path: Path, trips: pd.DataFrame) -> None:
    daytype = mod.build_station_daytype_averages(trips)
    assert (daytype["station_id"].str.len() > 0).all()


def test_generate_and_write_produces_daytype_table(tmp_path: Path) -> None:
    result = mod.generate_and_write(
        input_path=str(FIXTURE_ZIP),
        output_dir=str(tmp_path),
        max_station_plots=1,
    )
    out_csv = tmp_path / "station_daytype_ridership.csv"
    assert out_csv.exists()
    written = pd.read_csv(out_csv, dtype={"station_id": str})
    assert list(written.columns) == [
        "station_id",
        "station_name",
        "avg_weekday_riders",
        "avg_saturday_riders",
        "avg_sunday_riders",
        "weekday_days",
        "saturday_days",
        "sunday_days",
    ]
    assert len(written) == result["station_daytype"]["station_id"].nunique()
