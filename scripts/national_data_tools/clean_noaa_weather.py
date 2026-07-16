"""Clean a NOAA CDO daily-summaries export into an analysis-ready frame.

Standalone script: edit the config block and call ``run()`` (notebook) or run as
``python clean_noaa_daily.py [--input ...] [--output ...] [--long-names]`` (CLI).
Cleaning here is deliberately *faithful and non-opinionated*: it repairs
structural and semantic defects only (blank padding rows, WT event-flag
semantics, dtypes, date axis) and adds calendar columns. Feature engineering and
feature selection -- including which weather variables matter for ridership --
live in separate downstream steps, so this frame stays reusable across consumers.

Source: https://www.ncei.noaa.gov/cdo-web/search

Outputs
-------
- The cleaned daily frame at ``OUTPUT_PATH`` (``.parquet`` or ``.csv``): parsed
  dates plus calendar columns, zero-filled ``WT*`` event flags, and numeric
  measurement columns.
- A CSV of monthly aggregates at ``MONTHLY_OUTPUT_PATH`` (``avg_temp_f``,
  ``max_daily_precip_in``, ``days_with_precip``, ``total_snow_in``,
  ``max_daily_snow_in``), written only when that path is set.
- ``<output stem>_processing_log.txt`` (or ``LOG_PATH``): a .txt processing log
  written next to the output when ``WRITE_LOG`` is on (the default).

Typical usage
-------------
Set ``INPUT_PATH`` and ``OUTPUT_PATH`` in the config block (or pass ``--input``
/ ``--output``; ``--long-names`` and ``--log`` are also available) and run from
a shell or a Jupyter notebook. Paths left unset are prompted for interactively.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# --- <<< EDIT ME -------------------------------------------------------------
# Set these to use them directly. Leave as None to be prompted (notebook) or to
# pass --input / --output on the command line.
INPUT_PATH: Path | None = None
OUTPUT_PATH: Path | None = None
MONTHLY_OUTPUT_PATH: Path | None = (
    None  # set to write the monthly aggregates alongside the daily output
)
DATE_FORMAT: str | None = None  # None = infer (handles ISO 2021-01-01 and 1/1/2021)
USE_LONG_NAMES = False  # rename code headers (TMAX) to long names; logged either way
WRITE_LOG = True  # also write a .txt processing log next to the output
LOG_PATH: Path | None = None  # None = derive from output path
WT_PREFIX = "WT"  # NOAA weather-type occurrence flags: blank == did not occur.
ID_COLS = ["STATION", "NAME", "DATE"]
# --- EDIT ME >>> -------------------------------------------------------------

logger = logging.getLogger(__name__)

# NOAA daily-summary element codes -> human-readable names.
COLUMN_LONG_NAMES = {
    "AWND": "Average wind speed",
    "PGTM": "Peak gust time",
    "PRCP": "Precipitation",
    "SNOW": "Snowfall",
    "SNWD": "Snow depth",
    "TAVG": "Average temperature",
    "TMAX": "Maximum temperature",
    "TMIN": "Minimum temperature",
    "WDF2": "Direction of fastest 2-minute wind",
    "WDF5": "Direction of fastest 5-second wind",
    "WSF2": "Fastest 2-minute wind speed",
    "WSF5": "Fastest 5-second wind speed",
    "WT01": "Fog, ice fog, or freezing fog (may include heavy fog)",
    "WT02": "Heavy fog or heavy freezing fog (not always distinguished from fog)",
    "WT03": "Thunder",
    "WT04": "Ice pellets, sleet, snow pellets, or small hail",
    "WT05": "Hail (may include small hail)",
    "WT06": "Glaze or rime",
    "WT08": "Smoke or haze",
    "WT09": "Blowing or drifting snow",
    "WT18": "Snow, snow pellets, snow grains, or ice crystals",
}


def _add_file_log(path: Path) -> logging.FileHandler:
    """Attach a .txt file handler to this module's logger.

    Replaces any stale handler from a prior run so repeated notebook calls
    don't duplicate lines.
    """
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return fh


def _prompt_path(label: str, *, must_exist: bool) -> Path:
    """Interactively ask for a path; works in a terminal and in a notebook."""
    while True:
        raw = input(f"Enter {label}: ").strip().strip('"').strip("'")
        if not raw:
            print("  a path is required")
            continue
        path = Path(raw).expanduser()
        if must_exist and not path.exists():
            print(f"  not found: {path}")
            continue
        return path


def _in_ipython_kernel() -> bool:
    """Return True inside a Jupyter/IPython kernel.

    There ``sys.argv`` holds the kernel launcher args (e.g. ``-f
    ...kernel.json``) rather than user CLI args.
    """
    return "ipykernel" in sys.modules or Path(sys.argv[0]).name == "ipykernel_launcher.py"


def _ensure_logging(level: int = logging.INFO) -> None:
    """Make INFO visible in both CLI and notebook sessions."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


def _check_missing_days(dates: pd.Series) -> None:
    """Warn about calendar days with no row inside the observed date range.

    A day or two missing is usually benign; the point is visibility, so the
    headline is a WARNING with count and percentage of the full span.
    """
    days = pd.to_datetime(dates).dt.normalize()
    start, end = days.min(), days.max()
    full = pd.date_range(start, end, freq="D")
    missing = full.difference(pd.DatetimeIndex(days.unique()))
    total = len(full)
    if len(missing) == 0:
        logger.info("no missing days in %s..%s (%d days)", start.date(), end.date(), total)
        return
    logger.warning(
        "%d of %d days missing in %s..%s (%.1f%% of range)",
        len(missing),
        total,
        start.date(),
        end.date(),
        100 * len(missing) / total,
    )
    listed = [d.date().isoformat() for d in missing]
    if len(listed) <= 20:
        logger.warning("missing days: %s", ", ".join(listed))
    else:
        logger.warning("missing days (first 20 of %d): %s ...", len(listed), ", ".join(listed[:20]))


def clean_weather(df: pd.DataFrame, *, use_long_names: bool = USE_LONG_NAMES) -> pd.DataFrame:
    """Return a faithful, analysis-ready copy of a NOAA CDO daily export."""
    df = df.copy()

    # 1. Drop export padding -- NOAA appends fully-blank trailing rows.
    df = df.dropna(how="all")

    # 2. Parse and order the date axis, then add calendar columns next to it.
    df["DATE"] = pd.to_datetime(df["DATE"], format=DATE_FORMAT)
    df = df.sort_values("DATE").reset_index(drop=True)
    date_pos = int(df.columns.get_loc("DATE"))
    df.insert(date_pos + 1, "YEAR", df["DATE"].dt.year.astype("int16"))
    df.insert(date_pos + 2, "MONTH", df["DATE"].dt.month.astype("int8"))
    df.insert(date_pos + 3, "DAY_OF_WEEK", df["DATE"].dt.day_name())
    _check_missing_days(df["DATE"])

    # 3. WT* flags: blank means the event did not occur, NOT "missing". Fill to
    #    0 and keep every flag -- rare high-impact events (ice, hail, heavy fog)
    #    are precisely the ridership-relevant days. Pruning happens downstream.
    wt_cols = [c for c in df.columns if c.startswith(WT_PREFIX)]
    df[wt_cols] = df[wt_cols].fillna(0).astype("int8")

    # 4. Drop columns with no observations at all (e.g. PGTM, TAVG for this
    #    station). WT flags are exempt -- they were just filled to 0 above, so a
    #    never-fired flag is all-zero, not all-null, and is kept on purpose.
    empty = [c for c in df.columns if c not in wt_cols and df[c].isna().all()]
    if empty:
        logger.info("dropping all-empty columns: %s", empty)
    df = df.drop(columns=empty)

    # 5. Coerce the continuous measurements to numeric (no-op if already clean).
    #    WDF2/WDF5 stay raw degrees here; circular sin/cos encoding is a
    #    downstream feature transform, not a cleaning concern.
    derived = ["YEAR", "MONTH", "DAY_OF_WEEK"]
    num_cols = df.columns.difference([*ID_COLS, *derived, *wt_cols])
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # 6. Log the code -> long-name mapping for the columns actually present,
    #    then optionally apply it. Mapping is logged regardless of the choice.
    present = {c: COLUMN_LONG_NAMES[c] for c in df.columns if c in COLUMN_LONG_NAMES}
    logger.info("NOAA code -> long-name mapping (%d columns):", len(present))
    for code, long_name in present.items():
        logger.info("  %s -> %s", code, long_name)
    if use_long_names:
        logger.info("applying long-name headers")
        df = df.rename(columns=present)
    else:
        logger.info("keeping short code headers (USE_LONG_NAMES is off)")

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a cleaned daily NOAA frame into monthly ridership-model inputs.

    Returns one row per calendar month with a ``period`` key (``"YYYY-MM"``),
    plus daily-derived aggregates that capture both the intensity and the
    frequency of weather events within a month — more informative for ridership
    models than a simple monthly total.

    Columns produced (whichever source columns are present):
        period            — ``"YYYY-MM"`` join key for the exogenous feature table
        avg_temp_f        — mean daily temperature (TAVG, or midpoint of TMAX/TMIN)
        max_daily_precip_in — peak single-day precipitation (captures one very wet day)
        days_with_precip  — count of days with measurable precipitation (PRCP > 0)
        total_snow_in     — sum of daily snowfall
        max_daily_snow_in — peak single-day snowfall
    """
    df = df.copy()
    df["period"] = df["DATE"].dt.to_period("M").astype(str)

    agg: dict[str, Any] = {}

    if "TAVG" in df.columns:
        agg["avg_temp_f"] = pd.NamedAgg(column="TAVG", aggfunc="mean")
    elif "TMAX" in df.columns and "TMIN" in df.columns:
        df["_tmid"] = (df["TMAX"] + df["TMIN"]) / 2.0
        agg["avg_temp_f"] = pd.NamedAgg(column="_tmid", aggfunc="mean")

    if "PRCP" in df.columns:
        agg["max_daily_precip_in"] = pd.NamedAgg(column="PRCP", aggfunc="max")
        agg["days_with_precip"] = pd.NamedAgg(column="PRCP", aggfunc=lambda x: int((x > 0).sum()))

    if "SNOW" in df.columns:
        agg["total_snow_in"] = pd.NamedAgg(column="SNOW", aggfunc="sum")
        agg["max_daily_snow_in"] = pd.NamedAgg(column="SNOW", aggfunc="max")

    if not agg:
        raise ValueError(
            "No recognised weather columns (TAVG, TMAX/TMIN, PRCP, SNOW) found in frame."
        )

    monthly = df.groupby("period").agg(**agg).reset_index()

    rounding = {
        "avg_temp_f": 1,
        "max_daily_precip_in": 2,
        "total_snow_in": 1,
        "max_daily_snow_in": 1,
    }
    monthly = monthly.round({k: v for k, v in rounding.items() if k in monthly.columns})
    logger.info(
        "Monthly aggregates: %d months, columns: %s",
        len(monthly),
        list(monthly.columns),
    )
    return monthly


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
    monthly_output_path: Path | None = None,
    *,
    use_long_names: bool | None = None,
    write_log: bool | None = None,
) -> pd.DataFrame:
    """Notebook entry point: clean ``input_path`` and write ``output_path``.

    If ``monthly_output_path`` (or the module-level ``MONTHLY_OUTPUT_PATH``) is
    set, a second CSV of monthly aggregates is written alongside the daily
    output.  Those monthly columns (``avg_temp_f``, ``max_daily_precip_in``,
    ``days_with_precip``, ``total_snow_in``, ``max_daily_snow_in``) are what the
    ridership models expect in the exogenous feature table.

    Unset args fall back to the config block, resolved at call time -- so
    ``m.INPUT_PATH = ...; m.run()`` works as expected after a plain import.
    """
    _ensure_logging()
    input_path = INPUT_PATH if input_path is None else Path(input_path)
    output_path = OUTPUT_PATH if output_path is None else Path(output_path)
    monthly_output_path = (
        MONTHLY_OUTPUT_PATH if monthly_output_path is None else Path(monthly_output_path)
    )
    use_long_names = USE_LONG_NAMES if use_long_names is None else use_long_names
    write_log = WRITE_LOG if write_log is None else write_log

    # Anything still unset after arg + config block falls to an interactive prompt.
    if input_path is None:
        input_path = _prompt_path("input CSV path", must_exist=True)
    if output_path is None:
        output_path = _prompt_path("output path (.parquet or .csv)", must_exist=False)

    fh = None
    if write_log:
        log_path = LOG_PATH or output_path.with_name(f"{output_path.stem}_processing_log.txt")
        fh = _add_file_log(log_path)
        logger.info("processing %s -> %s", input_path, output_path)
        logger.info("processing log -> %s", log_path)
    try:
        # dtype on STATION guards against numeric-looking IDs being mangled by
        # the bare-read inference path.
        raw = pd.read_csv(input_path, dtype={"STATION": "string"})
        clean = clean_weather(raw, use_long_names=use_long_names)

        if output_path.suffix == ".parquet":
            clean.to_parquet(output_path, index=False)
        else:
            clean.to_csv(output_path, index=False)

        logger.info("%d rows x %d cols -> %s", len(clean), clean.shape[1], output_path)

        if monthly_output_path is not None:
            monthly = aggregate_monthly(clean)
            monthly.to_csv(monthly_output_path, index=False)
            logger.info("%d monthly rows -> %s", len(monthly), monthly_output_path)
    finally:
        if fh is not None:
            logger.removeHandler(fh)
            fh.close()
    return clean


def main(argv: list[str] | None = None) -> None:
    """Entry point for both notebook and CLI.

    Path resolution is the same everywhere: explicit value -> config block ->
    interactive prompt. In a Jupyter/IPython kernel the launcher injects its own
    argv (``-f kernel.json``), which argparse would reject, so we skip parsing
    and let ``run()`` resolve from the config block or prompt. On the command
    line, ``--input`` / ``--output`` override the config; omit them (with config
    left as None) to be prompted.
    """
    _ensure_logging()
    if argv is None and _in_ipython_kernel():
        logger.info("kernel detected; resolving paths from config block or prompt")
        run()
        return

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="input CSV path")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="output path (.parquet/.csv)"
    )
    parser.add_argument(
        "--long-names",
        action=argparse.BooleanOptionalAction,
        default=USE_LONG_NAMES,
        help="rename code headers to long names",
    )
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=WRITE_LOG,
        help="write a .txt processing log next to the output",
    )
    args = parser.parse_args(argv)
    run(args.input, args.output, use_long_names=args.long_names, write_log=args.log)


if __name__ == "__main__":
    main()
