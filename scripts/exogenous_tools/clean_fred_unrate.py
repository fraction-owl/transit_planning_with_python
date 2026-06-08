"""Clean a FRED CSV export into an analysis-ready frame.

Standalone script: edit the config block and call ``run()`` (notebook) or run as
``python clean_fred_series.py [--input ...] [--output ...] [--long-names]`` (CLI).
Cleaning here is deliberately *faithful and non-opinionated*: it repairs
structural and semantic defects only (blank padding rows, the FRED ``.`` missing
marker, dtypes, date axis) and adds calendar columns. It does NOT resample, fill,
or interpolate -- a genuinely missing observation (e.g. a not-yet-released month)
stays NaN. Feature engineering -- lags, YoY deltas, rolling means, alignment to a
ridership calendar -- lives in separate downstream steps, so this frame stays
reusable across consumers and across series (UNRATE, PAYEMS, CPIAUCSL, ...).

The script is generic over a single FRED series export and over the multi-series
``fredgraph`` shape (one date column, N value columns); it is not specific to
UNRATE. The value column(s) are detected as everything that is not the date axis.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# --- <<< EDIT ME -------------------------------------------------------------
# Set these to use them directly. Leave as None to be prompted (notebook) or to
# pass --input / --output on the command line.
INPUT_PATH: Path | None = None
OUTPUT_PATH: Path | None = None
DATE_FORMAT: str | None = None  # None = infer (handles ISO 2021-01-01 and 1/1/2021)
USE_LONG_NAMES = False  # rename series-id headers (UNRATE) to titles; logged either way
WRITE_LOG = True  # also write a .txt processing log next to the output
LOG_PATH: Path | None = None  # None = derive from output path
NA_VALUES = ["."]  # FRED's legacy missing marker; blank cells are NaN already.
DATE_COL_CANDIDATES = ("observation_date", "DATE")  # modern web download, then legacy
# --- EDIT ME >>> -------------------------------------------------------------

logger = logging.getLogger(__name__)

# FRED series id -> title, for the columns we actually touch. This is a starter
# set, not exhaustive: unknown ids pass through unchanged (and are logged), so
# extend it as you pull new series rather than worrying about coverage.
SERIES_LONG_NAMES = {
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "All Employees, Total Nonfarm",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
    "DGS10": "Market Yield on 10-Year Treasury Constant Maturity",
    "GASREGW": "US Regular All Formulations Gas Price",
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


def _resolve_date_col(columns: pd.Index) -> str:
    """Pick the date axis: a known FRED header if present, else the first column.

    FRED's modern web download labels it ``observation_date``; legacy fredgraph
    and ALFRED exports use ``DATE``. We fall back to the leftmost column rather
    than fail, since FRED always puts the date first.
    """
    for cand in DATE_COL_CANDIDATES:
        if cand in columns:
            return cand
    first = columns[0]
    logger.warning(
        "no %s column found; treating leftmost column %r as the date axis",
        " / ".join(DATE_COL_CANDIDATES),
        first,
    )
    return first


def _infer_frequency(dates: pd.Series) -> str | None:
    """Infer a pandas offset alias from the modal spacing of observations.

    Returns a calendar-anchored alias (D/B/W-XXX/MS/QS/YS) usable to rebuild the
    expected period index, or None if the cadence is irregular and a gap check
    would be meaningless. Daily classification is best-effort: a business-day
    series (no weekend observations) is reported as ``B``, so market holidays
    will read as small gaps -- visibility, not an assertion of data loss.
    """
    days = dates.sort_values().diff().dropna().dt.days
    if days.empty:
        return None
    med = days.median()
    if med <= 1.5:
        return "B" if not dates.dt.dayofweek.isin([5, 6]).any() else "D"
    if 6 <= med <= 8:
        anchor = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][dates.min().weekday()]
        return f"W-{anchor}"
    if 26 <= med <= 32:
        return "MS"
    if 85 <= med <= 95:
        return "QS"
    if 350 <= med <= 380:
        return "YS"
    return None


def _check_missing_periods(dates: pd.Series) -> None:
    """Warn about cadence periods with no row inside the observed date range.

    This is about *absent rows*, not present rows with a NaN value -- the latter
    is reported separately as missing observations. The headline is a WARNING
    with count and percentage of the full span, mirroring the daily check but in
    the series' own native cadence (months for UNRATE, quarters for GDP, ...).
    """
    days = pd.to_datetime(dates).dt.normalize()
    freq = _infer_frequency(days)
    start, end = days.min(), days.max()
    if freq is None:
        logger.info(
            "irregular cadence in %s..%s; skipping missing-period check", start.date(), end.date()
        )
        return
    full = pd.date_range(start, end, freq=freq)
    missing = full.difference(pd.DatetimeIndex(days.unique()))
    total = len(full)
    logger.info(
        "inferred cadence %s over %s..%s (%d periods)", freq, start.date(), end.date(), total
    )
    if len(missing) == 0:
        logger.info("no missing periods")
        return
    logger.warning(
        "%d of %d periods missing in %s..%s (%.1f%% of range)",
        len(missing),
        total,
        start.date(),
        end.date(),
        100 * len(missing) / total,
    )
    listed = [d.date().isoformat() for d in missing]
    if len(listed) <= 20:
        logger.warning("missing periods: %s", ", ".join(listed))
    else:
        logger.warning(
            "missing periods (first 20 of %d): %s ...", len(listed), ", ".join(listed[:20])
        )


def clean_series(df: pd.DataFrame, *, use_long_names: bool = USE_LONG_NAMES) -> pd.DataFrame:
    """Return a faithful, analysis-ready copy of a FRED CSV export."""
    df = df.copy()

    # 1. Drop export padding -- a stray fully-blank trailing row is common.
    df = df.dropna(how="all")

    # 2. Resolve and parse the date axis, then add calendar columns next to it.
    #    DAY_OF_WEEK is intentionally omitted: FRED dates anchor at the period
    #    start (day 1 for monthly/quarterly/annual), so weekday is degenerate
    #    for the common cases and a feature concern for daily series anyway.
    date_col = _resolve_date_col(df.columns)
    df[date_col] = pd.to_datetime(df[date_col], format=DATE_FORMAT)
    df = df.sort_values(date_col).reset_index(drop=True)
    date_pos = df.columns.get_loc(date_col)
    df.insert(date_pos + 1, "YEAR", df[date_col].dt.year.astype("int16"))
    df.insert(date_pos + 2, "MONTH", df[date_col].dt.month.astype("int8"))
    df.insert(date_pos + 3, "QUARTER", df[date_col].dt.quarter.astype("int8"))
    _check_missing_periods(df[date_col])

    # 3. Everything that is not the date axis or a derived calendar column is a
    #    value series. Coerce to numeric: the FRED "." marker was already mapped
    #    to NaN at read time (NA_VALUES); errors="coerce" is a backstop so one
    #    stray non-numeric cell can't poison the whole column to object dtype.
    derived = ["YEAR", "MONTH", "QUARTER"]
    value_cols = list(df.columns.difference([date_col, *derived], sort=False))
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

    # 4. Report missing *observations* -- rows that exist but carry no value
    #    (FRED publishes a row before the figure is released, or revises late).
    #    Distinct from missing periods (absent rows) above. Not filled: a NaN
    #    here is a real hole, and how to handle it is a downstream decision.
    for col in value_cols:
        n_na = int(df[col].isna().sum())
        if n_na:
            logger.warning("%s: %d of %d observations missing (NaN)", col, n_na, len(df))
        else:
            logger.info("%s: no missing observations (%d rows)", col, len(df))

    # 5. Drop value columns with no observations at all -- relevant only to the
    #    multi-series fredgraph shape, where one requested series may be empty
    #    over the window. A single-series export keeps its one value column.
    empty = [c for c in value_cols if df[c].isna().all()]
    if empty:
        logger.info("dropping all-empty columns: %s", empty)
        df = df.drop(columns=empty)
        value_cols = [c for c in value_cols if c not in empty]

    # 6. Log the series-id -> title mapping for the columns actually present,
    #    then optionally apply it. Mapping is logged regardless of the choice.
    present = {c: SERIES_LONG_NAMES[c] for c in value_cols if c in SERIES_LONG_NAMES}
    unknown = [c for c in value_cols if c not in SERIES_LONG_NAMES]
    logger.info(
        "FRED series-id -> title mapping (%d of %d value columns known):",
        len(present),
        len(value_cols),
    )
    for code, title in present.items():
        logger.info("  %s -> %s", code, title)
    if unknown:
        logger.info("  no title on file for: %s (passed through unchanged)", ", ".join(unknown))
    if use_long_names:
        logger.info("applying long-name headers")
        df = df.rename(columns=present)
    else:
        logger.info("keeping series-id headers (USE_LONG_NAMES is off)")

    return df


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
    *,
    use_long_names: bool | None = None,
    write_log: bool | None = None,
) -> pd.DataFrame:
    """Notebook entry point: clean ``input_path`` and write ``output_path``.

    Unset args fall back to the config block, resolved at call time -- so
    ``m.INPUT_PATH = ...; m.run()`` works as expected after a plain import.
    """
    _ensure_logging()
    input_path = INPUT_PATH if input_path is None else Path(input_path)
    output_path = OUTPUT_PATH if output_path is None else Path(output_path)
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
        # NA_VALUES maps FRED's "." missing marker to NaN at read time; without
        # it a single "." turns the whole value column to object dtype.
        raw = pd.read_csv(input_path, na_values=NA_VALUES)
        clean = clean_series(raw, use_long_names=use_long_names)

        if output_path.suffix == ".parquet":
            clean.to_parquet(output_path, index=False)
        else:
            clean.to_csv(output_path, index=False)

        logger.info("%d rows x %d cols -> %s", len(clean), clean.shape[1], output_path)
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
        help="rename series-id headers to titles",
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
