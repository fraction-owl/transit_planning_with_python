r"""Summarize Capital Bikeshare trip extracts into ridership-over-time outputs.

Reads a set of per-month Capital Bikeshare trip CSVs -- the vendor extracts
named ``YYYYMM-capitalbikeshare-tripdata.csv``, as produced by
``dev_tools/generate_mock_bikeshare_trips.py`` -- from either a directory or a
``.zip`` archive, and exports tables and charts describing how ridership
changes over time, both system-wide and for each individual station.

This is the polars twin of ``bikeshare_ridership_trends.py``: same inputs,
same outputs (the aggregate tables are byte-identical), but built on polars,
which keeps the load/aggregate steps fast on multi-year, full-size vendor
extracts. polars belongs to the open-source stack (see requirements.txt) and
is not available in ArcGIS Pro's bundled Python -- inside ArcGIS Pro, use the
pandas original instead.

------------------------------------------------------------------------------
RUNNING IT
------------------------------------------------------------------------------
Notebook / manual: edit the CONFIG block below, then run the file (or call
``run()``). No command-line arguments are needed.

Command line: every CONFIG value has a matching flag that overrides it, e.g.
    python scripts/gbfs_tools/bikeshare_ridership_trends_polars.py \\
        --input tests/fixtures/capitalbikeshare_fixtures_24mo.zip \\
        --output-dir out/bikeshare_trends

------------------------------------------------------------------------------
WHAT IT PRODUCES
------------------------------------------------------------------------------
In OUTPUT_DIR:
  * ``trips_concatenated.csv`` -- every trip from every monthly file stacked
    into one table, with an added ``month`` column (``YYYY-MM``) and a
    ``source_file`` column recording which extract each row came from.
  * ``monthly_system_ridership.csv`` -- one row per month: total trips plus
    member/casual and electric/classic splits and the dockless (blank-station)
    start count.
  * ``monthly_station_ridership.csv`` -- one row per (month, station): trips
    departing, arriving, and total activity. Dockless trips with a blank
    station are excluded from the per-station table but still counted in the
    system totals.
  * ``station_daytype_ridership.csv`` -- one row per station: average daily
    ridership (departures + arrivals) on weekdays, Saturdays, and Sundays,
    plus the day counts used as denominators.
  * ``plots/system_ridership_trend.png`` -- system-wide trips per month.
  * ``plots/stations/station_<id>.png`` -- one trend chart per station.

Station ridership counts both departures (trips whose start station is the
station) and arrivals (trips whose end station is the station); ``total`` is
their sum. The per-station table and charts span every month in the data so
trends include months with zero activity rather than skipping them.

Day-type averages divide each station's total activity on weekday / Saturday /
Sunday dates by the number of such dates in the covered calendar (every day of
every month present in the data, so zero-activity days count). HOLIDAY RULE:
observed U.S. federal holidays are classified as Sunday-equivalent -- the
standard transit convention (holidays run Sunday service) -- so
``avg_weekday_riders`` covers non-holiday weekdays only, matching the
weekday posture of the GTFS-side feature scripts. A trip is attributed to the
day type of its start date.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import logging
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

# ===========================================================================
# CONFIG  --  notebook users edit these; CLI flags override them
# ===========================================================================

# Input: a directory of ``*-capitalbikeshare-tripdata.csv`` files, or a single
# ``.zip`` archive containing them. Raw string (r"...") so Windows paths paste
# in safely.
INPUT = r"tests/fixtures/capitalbikeshare_fixtures_24mo.zip"

# Directory the tables and charts are written to.
OUTPUT_DIR = r"out/bikeshare_trends"

# Cap on how many per-station charts to draw, ordered by total ridership
# (largest first). Set to 0 (or pass --max-station-plots 0) to draw every
# station. The per-station CSV always covers every station regardless.
MAX_STATION_PLOTS = 0

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# ===========================================================================
# Constants
# ===========================================================================

# Trip files follow the vendor naming convention ``YYYYMM-capitalbikeshare-...``.
TRIP_FILE_GLOB = "*-capitalbikeshare-tripdata.csv"

# Columns whose blanks are meaningful (dockless trips) and whose values must
# not be coerced to floats, so they are forced to Utf8 with blanks kept as
# empty strings rather than nulls.
_STRING_COLUMNS = (
    "ride_id",
    "rideable_type",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "member_casual",
)

logger = logging.getLogger(__name__)

# matplotlib emits chatty INFO records (e.g. categorical-units notices) that
# would otherwise drown out this script's own logging; keep it to warnings.
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# Trip extracts are normally UTF-8, but some vendor months ship a stray
# Windows-1252 byte (e.g. 0x9c) that makes a strict UTF-8 read abort the whole
# run. Decode order: UTF-8 (BOM-aware) first, then cp1252, then latin-1 -- the
# last accepts every byte, so one oddly encoded file no longer sinks the rest.
# (polars itself only decodes utf8/utf8-lossy, and lossy would silently mangle
# the cp1252 characters instead of preserving them.)
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _decode_csv(data: bytes, source_name: str) -> str:
    """Decode trip-CSV bytes, tolerating the occasional non-UTF-8 extract.

    A file that needs a fallback is logged so the odd encoding is surfaced
    rather than silently reinterpreted.
    """
    for encoding in _ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != _ENCODINGS[0]:
            logger.warning(
                "Decoded %s as %s after UTF-8 failed; verify that file's encoding.",
                source_name,
                encoding,
            )
        return text
    # latin-1 above never raises, so this is unreachable; kept for safety.
    return data.decode("latin-1", errors="replace")


def _read_one(data: bytes, source_name: str) -> pl.DataFrame:
    """Read a single trip CSV from raw bytes, keeping blanks as empty strings.

    Args:
        data: Raw bytes of one trip extract (a directory file or a zip member).
        source_name: File name recorded in the returned ``source_file`` column.

    Returns:
        The trips in one extract, with a ``source_file`` column added.
    """
    frame = pl.read_csv(
        _decode_csv(data, source_name).encode("utf-8"),
        schema_overrides={col: pl.Utf8 for col in _STRING_COLUMNS},
        missing_utf8_is_empty_string=True,
    )
    return frame.with_columns(pl.lit(source_name).alias("source_file"))


def load_trips(input_path: Path) -> pl.DataFrame:
    """Concatenate every monthly trip extract under ``input_path``.

    Args:
        input_path: A directory of ``*-capitalbikeshare-tripdata.csv`` files,
            or a ``.zip`` archive containing them.

    Returns:
        All trips stacked into one frame with added ``month`` (``YYYY-MM``) and
        ``source_file`` columns, sorted by start time.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If no matching trip files are found.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"INPUT not found: {input_path}")

    frames: list[pl.DataFrame] = []
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith("-capitalbikeshare-tripdata.csv")
            )
            for member in members:
                frames.append(_read_one(archive.read(member), Path(member).name))
    else:
        # rglob (not glob) so a directory whose monthly extracts each sit in
        # their own subfolder still resolves -- this is exactly the layout the
        # prep_features_public.py orchestrator produces when it unzips each
        # ``YYYYMM-capitalbikeshare-tripdata.zip`` into a sibling folder.
        for path in sorted(input_path.rglob(TRIP_FILE_GLOB)):
            frames.append(_read_one(path.read_bytes(), path.name))

    if not frames:
        raise ValueError(f"No '{TRIP_FILE_GLOB}' files found under {input_path}")

    # ``diagonal`` unions columns across extracts (like pandas.concat) in case
    # a vendor month adds or drops a field.
    trips = pl.concat(frames, how="diagonal")
    trips = trips.with_columns(
        pl.col("started_at").str.to_datetime().dt.strftime("%Y-%m").alias("month")
    )
    trips = trips.select("month", pl.exclude("month")).sort("started_at", maintain_order=True)
    logger.info("Loaded %d trips from %d file(s).", len(trips), len(frames))
    return trips


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_system_monthly(trips: pl.DataFrame) -> pl.DataFrame:
    """Aggregate trips to one row per month with split-out counts.

    Args:
        trips: Concatenated trips, as returned by :func:`load_trips`.

    Returns:
        One row per month (sorted) with ``total_trips``, member/casual and
        electric/classic splits, and the dockless (blank start station) count.
    """
    return (
        trips.group_by("month")
        .agg(
            pl.len().cast(pl.Int64).alias("total_trips"),
            (pl.col("member_casual") == "member").sum().cast(pl.Int64).alias("member_trips"),
            (pl.col("rideable_type") == "electric_bike")
            .sum()
            .cast(pl.Int64)
            .alias("electric_trips"),
            (pl.col("start_station_id").str.len_chars() == 0)
            .sum()
            .cast(pl.Int64)
            .alias("dockless_start_trips"),
        )
        .with_columns(
            (pl.col("total_trips") - pl.col("member_trips")).alias("casual_trips"),
            (pl.col("total_trips") - pl.col("electric_trips")).alias("classic_trips"),
        )
        .select(
            "month",
            "total_trips",
            "member_trips",
            "casual_trips",
            "electric_trips",
            "classic_trips",
            "dockless_start_trips",
        )
        .sort("month")
    )


def build_station_monthly(trips: pl.DataFrame) -> pl.DataFrame:
    """Aggregate trips to one row per (month, station) with activity counts.

    Departures count trips leaving a station; arrivals count trips ending at a
    station; ``total`` is their sum. Trips with a blank station (dockless) are
    excluded. The result spans every month x station combination so trends
    include months with zero activity.

    Args:
        trips: Concatenated trips, as returned by :func:`load_trips`.

    Returns:
        One row per (month, station_id), sorted, with ``station_name``,
        ``departures``, ``arrivals``, and ``total`` columns.
    """
    months = trips.get_column("month").unique().sort()

    def _counts(id_col: str, name_col: str, label: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        docked = trips.filter(pl.col(id_col).str.len_chars() > 0)
        counts = (
            docked.group_by("month", id_col)
            .agg(pl.len().cast(pl.Int64).alias(label))
            .rename({id_col: "station_id"})
        )
        names = docked.select(
            pl.col(id_col).alias("station_id"),
            pl.col(name_col).alias("station_name"),
        )
        return counts, names

    departures, dep_names = _counts("start_station_id", "start_station_name", "departures")
    arrivals, arr_names = _counts("end_station_id", "end_station_name", "arrivals")

    # One name per id (first seen), preserving any trailing whitespace.
    names = pl.concat([dep_names, arr_names]).unique(
        subset="station_id", keep="first", maintain_order=True
    )

    grid = pl.DataFrame({"month": months}).join(
        names.select("station_id").sort("station_id"), how="cross"
    )
    return (
        grid.join(departures, on=["month", "station_id"], how="left", coalesce=True)
        .join(arrivals, on=["month", "station_id"], how="left", coalesce=True)
        .join(names, on="station_id", how="left", coalesce=True)
        .with_columns(
            pl.col("departures").fill_null(0),
            pl.col("arrivals").fill_null(0),
        )
        .with_columns((pl.col("departures") + pl.col("arrivals")).alias("total"))
        .select("month", "station_id", "station_name", "departures", "arrivals", "total")
        .sort("station_id", "month")
    )


def federal_holidays_observed(year: int) -> set[dt.date]:
    """Return the observed dates of the U.S. federal holidays of *year*.

    Covers the eleven holidays of 5 U.S.C. 6103: New Year's Day, Birthday of
    Martin Luther King Jr. (3rd Monday of January), Washington's Birthday
    (3rd Monday of February), Memorial Day (last Monday of May), Juneteenth
    (June 19, from its 2021 establishment onward), Independence Day, Labor
    Day (1st Monday of September), Columbus Day (2nd Monday of October),
    Veterans Day, Thanksgiving (4th Thursday of November), and Christmas.

    Fixed-date holidays falling on a Saturday are observed on the preceding
    Friday and those falling on a Sunday on the following Monday, so an
    observed date can land in the *previous* calendar year (e.g. New Year's
    Day 2022 was observed on 2021-12-31). Callers classifying a span of dates
    should therefore union this set over ``range(first_year, last_year + 2)``.

    Args:
        year: Calendar year whose holidays are computed.

    Returns:
        The observed dates of *year*'s federal holidays.
    """

    def nth_weekday(month: int, weekday: int, n: int) -> dt.date:
        first = dt.date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + dt.timedelta(days=offset + 7 * (n - 1))

    def last_monday(month: int) -> dt.date:
        next_month = dt.date(year + (month == 12), month % 12 + 1, 1)
        last = next_month - dt.timedelta(days=1)
        return last - dt.timedelta(days=last.weekday())

    def observed(day: dt.date) -> dt.date:
        if day.weekday() == 5:  # Saturday -> preceding Friday
            return day - dt.timedelta(days=1)
        if day.weekday() == 6:  # Sunday -> following Monday
            return day + dt.timedelta(days=1)
        return day

    fixed = [
        dt.date(year, 1, 1),  # New Year's Day
        dt.date(year, 7, 4),  # Independence Day
        dt.date(year, 11, 11),  # Veterans Day
        dt.date(year, 12, 25),  # Christmas Day
    ]
    if year >= 2021:
        fixed.append(dt.date(year, 6, 19))  # Juneteenth
    floating = [
        nth_weekday(1, 0, 3),  # Birthday of Martin Luther King Jr.
        nth_weekday(2, 0, 3),  # Washington's Birthday
        last_monday(5),  # Memorial Day
        nth_weekday(9, 0, 1),  # Labor Day
        nth_weekday(10, 0, 2),  # Columbus Day
        nth_weekday(11, 3, 4),  # Thanksgiving Day
    ]
    return {observed(day) for day in fixed} | set(floating)


#: Day-type labels, in output order.
DAY_TYPES: tuple[str, ...] = ("weekday", "saturday", "sunday")


def _calendar_dates(months: list[str]) -> list[dt.date]:
    """Return every calendar date of the given ``YYYY-MM`` months, sorted."""
    dates: list[dt.date] = []
    for month in sorted(months):
        year, month_num = int(month[:4]), int(month[5:7])
        n_days = calendar.monthrange(year, month_num)[1]
        dates.extend(dt.date(year, month_num, day) for day in range(1, n_days + 1))
    return dates


def _day_type(day: dt.date, holidays: set[dt.date]) -> str:
    """Classify a date as weekday / saturday / sunday (holidays -> sunday)."""
    if day in holidays or day.weekday() == 6:
        return "sunday"
    if day.weekday() == 5:
        return "saturday"
    return "weekday"


def build_station_daytype_averages(trips: pl.DataFrame) -> pl.DataFrame:
    """Compute average daily ridership per station by day type.

    Every calendar date of every month present in the data is classified as
    weekday, Saturday, or Sunday, with observed U.S. federal holidays counted
    as Sunday-equivalent (see the module docstring for the full holiday rule).
    Each station's activity -- departures plus arrivals, with a trip
    attributed to the day type of its *start* date -- is summed per day type
    and divided by the number of such dates in the covered calendar, so days
    with zero activity pull the average down rather than being skipped.
    Dockless trips with a blank station are excluded, as in
    :func:`build_station_monthly`.

    Args:
        trips: Concatenated trips, as returned by :func:`load_trips`.

    Returns:
        One row per station_id (sorted) with ``station_name``,
        ``avg_weekday_riders``, ``avg_saturday_riders``, ``avg_sunday_riders``
        (rounded to 4 decimals), and the ``weekday_days`` / ``saturday_days``
        / ``sunday_days`` denominators (identical on every row).
    """
    dates = _calendar_dates(trips.get_column("month").unique().sort().to_list())
    years = range(dates[0].year, dates[-1].year + 2)
    holidays = set().union(*(federal_holidays_observed(year) for year in years))
    day_counts = Counter(_day_type(day, holidays) for day in dates)

    type_by_date = {day: _day_type(day, holidays) for day in dates}
    mapping = pl.DataFrame({"_date": list(type_by_date), "day_type": list(type_by_date.values())})
    trips = trips.with_columns(
        pl.col("started_at").str.to_datetime().dt.date().alias("_date")
    ).join(mapping, on="_date", how="left", coalesce=True)

    def _counts(id_col: str, name_col: str, label: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        docked = trips.filter(pl.col(id_col).str.len_chars() > 0)
        counts = (
            docked.group_by("day_type", id_col)
            .agg(pl.len().cast(pl.Int64).alias(label))
            .rename({id_col: "station_id"})
        )
        names = docked.select(
            pl.col(id_col).alias("station_id"),
            pl.col(name_col).alias("station_name"),
        )
        return counts, names

    departures, dep_names = _counts("start_station_id", "start_station_name", "departures")
    arrivals, arr_names = _counts("end_station_id", "end_station_name", "arrivals")

    # One name per id (first seen), preserving any trailing whitespace.
    names = pl.concat([dep_names, arr_names]).unique(
        subset="station_id", keep="first", maintain_order=True
    )

    grid = pl.DataFrame({"day_type": list(DAY_TYPES)}).join(
        names.select("station_id").sort("station_id"), how="cross"
    )
    activity = (
        grid.join(departures, on=["day_type", "station_id"], how="left", coalesce=True)
        .join(arrivals, on=["day_type", "station_id"], how="left", coalesce=True)
        .with_columns(
            pl.col("departures").fill_null(0),
            pl.col("arrivals").fill_null(0),
        )
        .with_columns((pl.col("departures") + pl.col("arrivals")).alias("total"))
    )

    wide = activity.pivot(values="total", index="station_id", columns="day_type")
    return (
        wide.join(names, on="station_id", how="left", coalesce=True)
        .with_columns(
            (pl.col(day_type) / day_counts[day_type]).round(4).alias(f"avg_{day_type}_riders")
            for day_type in DAY_TYPES
        )
        .with_columns(
            pl.lit(day_counts[day_type]).cast(pl.Int64).alias(f"{day_type}_days")
            for day_type in DAY_TYPES
        )
        .select(
            "station_id",
            "station_name",
            *(f"avg_{day_type}_riders" for day_type in DAY_TYPES),
            *(f"{day_type}_days" for day_type in DAY_TYPES),
        )
        .sort("station_id")
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def plot_system_trend(system_monthly: pl.DataFrame, out_path: Path) -> None:
    """Draw and save the system-wide monthly ridership trend.

    Args:
        system_monthly: Output of :func:`build_system_monthly`.
        out_path: PNG file path to write.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = _month_datetimes(system_monthly.get_column("month"))
    ax.plot(x, system_monthly.get_column("total_trips"), marker="o", label="All trips")
    ax.plot(x, system_monthly.get_column("member_trips"), marker=".", label="Member")
    ax.plot(x, system_monthly.get_column("casual_trips"), marker=".", label="Casual")
    ax.set_title("System ridership over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Trips")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _format_month_axis(fig, ax, len(system_monthly))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_station_trends(
    station_monthly: pl.DataFrame, out_dir: Path, max_plots: int = 0
) -> list[Path]:
    """Draw and save a monthly ridership trend chart per station.

    Args:
        station_monthly: Output of :func:`build_station_monthly`.
        out_dir: Directory to write per-station PNGs into.
        max_plots: Draw only the busiest ``max_plots`` stations; ``0`` draws
            all of them.

    Returns:
        The chart paths written, busiest station first.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = (
        station_monthly.group_by("station_id")
        .agg(pl.col("total").sum())
        .sort("total", descending=True)
    )
    if max_plots and max_plots > 0:
        totals = totals.head(max_plots)

    written: list[Path] = []
    for station_id in totals.get_column("station_id"):
        rows = station_monthly.filter(pl.col("station_id") == station_id)
        name = str(rows.get_column("station_name")[0]).strip()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = _month_datetimes(rows.get_column("month"))
        ax.plot(x, rows.get_column("total"), marker="o", label="Total")
        ax.plot(x, rows.get_column("departures"), marker=".", label="Departures")
        ax.plot(x, rows.get_column("arrivals"), marker=".", label="Arrivals")
        ax.set_title(f"Ridership over time -- {name} ({station_id})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Trips")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        _format_month_axis(fig, ax, len(rows))
        fig.tight_layout()
        path = out_dir / f"station_{_safe_name(station_id)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def _month_datetimes(months: pl.Series) -> list[datetime]:
    """Convert ``YYYY-MM`` month labels to datetimes for a matplotlib date axis."""
    return [datetime.strptime(month, "%Y-%m") for month in months]


def _format_month_axis(fig: plt.Figure, ax: plt.Axes, n_months: int) -> None:
    """Label the x-axis by month, thinning ticks to at most ~12 for legibility."""
    interval = max(1, n_months // 12)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    fig.autofmt_xdate(rotation=45)


def _safe_name(value: str) -> str:
    """Reduce a value to a filesystem-safe slug for use in a file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate_and_write(
    *, input_path: str, output_dir: str, max_station_plots: int
) -> dict[str, object]:
    """Load the extracts, build the tables and charts, and write everything.

    Args:
        input_path: Directory or ``.zip`` of monthly trip extracts.
        output_dir: Directory to write tables and charts into.
        max_station_plots: Cap on per-station charts (``0`` = all).

    Returns:
        Mapping describing what was written: ``trips``, ``system``, ``station``
        frames and the chart paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trips = load_trips(Path(input_path))
    system_monthly = build_system_monthly(trips)
    station_monthly = build_station_monthly(trips)
    station_daytype = build_station_daytype_averages(trips)

    trips.write_csv(out_dir / "trips_concatenated.csv")
    system_monthly.write_csv(out_dir / "monthly_system_ridership.csv")
    station_monthly.write_csv(out_dir / "monthly_station_ridership.csv")
    station_daytype.write_csv(out_dir / "station_daytype_ridership.csv")

    plots_dir = out_dir / "plots"
    system_plot = plots_dir / "system_ridership_trend.png"
    plot_system_trend(system_monthly, system_plot)
    station_plots = plot_station_trends(station_monthly, plots_dir / "stations", max_station_plots)

    logger.info("Wrote 4 tables and %d charts to %s", 1 + len(station_plots), out_dir)
    return {
        "trips": trips,
        "system": system_monthly,
        "station": station_monthly,
        "station_daytype": station_daytype,
        "system_plot": system_plot,
        "station_plots": station_plots,
    }


def run() -> dict[str, object]:
    """Run using the CONFIG block above. Intended for notebook / manual use."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s %(message)s")
    return generate_and_write(
        input_path=INPUT,
        output_dir=OUTPUT_DIR,
        max_station_plots=MAX_STATION_PLOTS,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIG block values."""
    parser = argparse.ArgumentParser(
        description="Summarize Capital Bikeshare extracts into ridership-over-time "
        "tables and charts. Defaults come from the CONFIG block.",
    )
    parser.add_argument(
        "--input", default=INPUT, help="Directory or .zip of monthly trip extracts."
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR, help="Directory for the tables and charts."
    )
    parser.add_argument(
        "--max-station-plots",
        type=int,
        default=MAX_STATION_PLOTS,
        help="Cap on per-station charts (0 = all, busiest first).",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    try:
        generate_and_write(
            input_path=args.input,
            output_dir=args.output_dir,
            max_station_plots=args.max_station_plots,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Script completed successfully.")
    return 0


def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":
    # In a notebook (pasted cell or %run), use the CONFIG block instead of
    # argparse, which would otherwise try to parse the kernel's own argv.
    if _in_ipython():
        run()
    else:
        raise SystemExit(main())
