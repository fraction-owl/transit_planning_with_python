"""Tests for scripts.gbfs_tools.bikeshare_ridership_trends_polars."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")  # headless: no display needed for savefig

from scripts.gbfs_tools import bikeshare_ridership_trends_polars as mod

FIXTURE_ZIP = Path("tests/fixtures/capitalbikeshare_fixtures_24mo.zip")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trips() -> pl.DataFrame:
    return mod.load_trips(FIXTURE_ZIP)


def _raw_columns(trips: pl.DataFrame) -> list[str]:
    """Columns of the original vendor extract (no derived month/source_file)."""
    return [c for c in trips.columns if c not in ("month", "source_file")]


# ---------------------------------------------------------------------------
# load_trips
# ---------------------------------------------------------------------------


def test_load_trips_concatenates_all_months(trips: pl.DataFrame) -> None:
    assert len(trips) == 3000
    assert trips.get_column("source_file").n_unique() == 24


def test_load_trips_adds_month_column(trips: pl.DataFrame) -> None:
    assert "month" in trips.columns
    assert trips.get_column("month").min() == "2024-05"
    assert trips.get_column("month").max() == "2026-04"


def test_load_trips_keeps_blank_stations_as_empty_strings(trips: pl.DataFrame) -> None:
    # Dockless trips have a blank station; they must be "" not null.
    assert (trips.get_column("start_station_id") == "").any()
    assert not trips.get_column("start_station_id").is_null().any()


def test_load_trips_is_sorted_by_start(trips: pl.DataFrame) -> None:
    assert trips.get_column("started_at").is_sorted()


def test_load_trips_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_trips(tmp_path / "nope.zip")


def test_load_trips_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No"):
        mod.load_trips(tmp_path)


def test_load_trips_from_directory(tmp_path: Path, trips: pl.DataFrame) -> None:
    # Round-trip one month through a directory and confirm it loads.
    one_month = trips.filter(pl.col("month") == "2024-05")
    csv_path = tmp_path / "202405-capitalbikeshare-tripdata.csv"
    one_month.select(_raw_columns(trips)).write_csv(csv_path)
    loaded = mod.load_trips(tmp_path)
    assert len(loaded) == len(one_month)


def test_load_trips_from_nested_directories(tmp_path: Path, trips: pl.DataFrame) -> None:
    # The prep_features_public.py orchestrator unzips each monthly archive into its own
    # subfolder, so the CSVs land one level below the directory handed to
    # load_trips. Confirm the directory loader recurses into those subfolders.
    months = ["2024-05", "2024-06"]
    for month in months:
        stem = f"{month.replace('-', '')}-capitalbikeshare-tripdata"
        sub = tmp_path / stem
        sub.mkdir()
        trips.filter(pl.col("month") == month).select(_raw_columns(trips)).write_csv(
            sub / f"{stem}.csv"
        )
    loaded = mod.load_trips(tmp_path)
    assert loaded.get_column("source_file").n_unique() == len(months)
    assert len(loaded) == len(trips.filter(pl.col("month").is_in(months)))


def test_load_trips_tolerates_non_utf8_extract(tmp_path: Path, trips: pl.DataFrame) -> None:
    # Some vendor months carry a stray Windows-1252 byte (e.g. 0x9c) that a
    # strict UTF-8 read would choke on, aborting the whole run. Write one month
    # with a cp1252-only station name and confirm it still loads.
    odd_name = "Cœur Plaza"  # 'œ' encodes to the lone byte 0x9c in cp1252
    one_month = trips.filter(pl.col("month") == "2024-05").with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(odd_name))
        .otherwise(pl.col("start_station_name"))
        .alias("start_station_name")
    )
    csv_path = tmp_path / "202405-capitalbikeshare-tripdata.csv"
    csv_path.write_bytes(one_month.select(_raw_columns(trips)).write_csv().encode("cp1252"))

    # A strict UTF-8 read of these bytes must fail -- proving the fixture is the
    # problematic case the loader has to survive.
    with pytest.raises(UnicodeDecodeError):
        csv_path.read_bytes().decode("utf-8")

    loaded = mod.load_trips(tmp_path)
    assert len(loaded) == len(one_month)
    assert odd_name in loaded.get_column("start_station_name").to_list()


# ---------------------------------------------------------------------------
# build_system_monthly
# ---------------------------------------------------------------------------


def test_system_monthly_one_row_per_month(trips: pl.DataFrame) -> None:
    system = mod.build_system_monthly(trips)
    assert len(system) == 24
    assert system.get_column("month").is_sorted()


def test_system_monthly_totals_reconcile(trips: pl.DataFrame) -> None:
    system = mod.build_system_monthly(trips)
    assert system.get_column("total_trips").sum() == len(trips)
    assert (
        system.get_column("member_trips") + system.get_column("casual_trips")
        == system.get_column("total_trips")
    ).all()
    assert (
        system.get_column("electric_trips") + system.get_column("classic_trips")
        == system.get_column("total_trips")
    ).all()


# ---------------------------------------------------------------------------
# build_station_monthly
# ---------------------------------------------------------------------------


def test_station_monthly_spans_full_grid(trips: pl.DataFrame) -> None:
    station = mod.build_station_monthly(trips)
    n_months = trips.get_column("month").n_unique()
    n_stations = station.get_column("station_id").n_unique()
    assert len(station) == n_months * n_stations


def test_station_monthly_total_is_departures_plus_arrivals(trips: pl.DataFrame) -> None:
    station = mod.build_station_monthly(trips)
    assert (
        station.get_column("departures") + station.get_column("arrivals")
        == station.get_column("total")
    ).all()


def test_station_monthly_excludes_dockless(trips: pl.DataFrame) -> None:
    station = mod.build_station_monthly(trips)
    assert (station.get_column("station_id").str.len_chars() > 0).all()
    # Docked departures only -> per-station departures sum <= all trips.
    docked_starts = (trips.get_column("start_station_id").str.len_chars() > 0).sum()
    assert station.get_column("departures").sum() == docked_starts


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
    assert len(station_pngs) == result["station"].get_column("station_id").n_unique()


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

# The 24-month fixture spans 2024-05-01 .. 2026-04-30 (730 calendar days):
# 500 non-holiday weekdays, 104 Saturdays, and 126 Sunday-equivalents (104
# true Sundays + 22 observed federal holidays falling on a weekday).
_EXPECTED_DAYS = {"weekday": 500, "saturday": 104, "sunday": 126}


def test_daytype_day_counts_match_covered_calendar(trips: pl.DataFrame) -> None:
    daytype = mod.build_station_daytype_averages(trips)
    for day_type, expected in _EXPECTED_DAYS.items():
        assert (daytype.get_column(f"{day_type}_days") == expected).all()


def test_daytype_one_row_per_station(trips: pl.DataFrame) -> None:
    daytype = mod.build_station_daytype_averages(trips)
    station = mod.build_station_monthly(trips)
    assert (
        daytype.get_column("station_id").sort().to_list()
        == station.get_column("station_id").unique().sort().to_list()
    )


def test_daytype_averages_reconcile_with_total_activity(trips: pl.DataFrame) -> None:
    # Un-averaging (avg x day count) must recover every docked departure and
    # arrival, up to the 4-decimal rounding of the averages.
    daytype = mod.build_station_daytype_averages(trips)
    recovered = sum(
        (
            daytype.get_column(f"avg_{day_type}_riders") * daytype.get_column(f"{day_type}_days")
        ).sum()
        for day_type in mod.DAY_TYPES
    )
    docked = (trips.get_column("start_station_id").str.len_chars() > 0).sum() + (
        trips.get_column("end_station_id").str.len_chars() > 0
    ).sum()
    assert abs(recovered - docked) < 0.5


def test_daytype_table_matches_pandas_twin_byte_for_byte(
    trips: pl.DataFrame, tmp_path: Path
) -> None:
    # The module docstring promises byte-identical aggregate tables; hold the
    # new day-type table to that for the shared fixture.
    from scripts.gbfs_tools import bikeshare_ridership_trends as pandas_mod

    pandas_trips = pandas_mod.load_trips(FIXTURE_ZIP)
    pandas_mod.build_station_daytype_averages(pandas_trips).to_csv(
        tmp_path / "pandas.csv", index=False
    )
    mod.build_station_daytype_averages(trips).write_csv(tmp_path / "polars.csv")
    assert (tmp_path / "pandas.csv").read_bytes() == (tmp_path / "polars.csv").read_bytes()


def test_generate_and_write_produces_daytype_table(tmp_path: Path) -> None:
    result = mod.generate_and_write(
        input_path=str(FIXTURE_ZIP),
        output_dir=str(tmp_path),
        max_station_plots=1,
    )
    out_csv = tmp_path / "station_daytype_ridership.csv"
    assert out_csv.exists()
    written = pl.read_csv(out_csv, schema_overrides={"station_id": pl.Utf8})
    assert written.columns == [
        "station_id",
        "station_name",
        "avg_weekday_riders",
        "avg_saturday_riders",
        "avg_sunday_riders",
        "weekday_days",
        "saturday_days",
        "sunday_days",
    ]
    assert len(written) == result["station_daytype"].get_column("station_id").n_unique()
