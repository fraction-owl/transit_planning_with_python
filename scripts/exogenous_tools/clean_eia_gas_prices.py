"""Clean an EIA weekly retail gasoline/diesel export into a monthly price frame.

Standalone script: edit the config block and call ``run()`` (notebook) or run as
``python clean_eia_gas_prices.py [--input ...] [--output ...] [--long-names]``
(CLI). Input is the EIA "Data 1: U.S. Gasoline and Diesel Retail Prices" CSV
(series ``PET_PRI_GND_DCUS_NUS_W``): a three-row header (a banner row, a
``Sourcekey`` row, a long-name row) over weekly Monday observations in
``DD-Mon-YY`` format.

Cleaning here is deliberately *faithful and non-opinionated*: it repairs
structural defects (the EIA header block, dtypes, date axis), renames the
opaque sourcekeys to compact tokens, reports per-series coverage, and rolls the
weekly survey up to a monthly mean keyed for a route/system ridership panel.
What it does NOT do, on purpose, lives downstream: CPI deflation to real
dollars, lag/rolling features, and choosing which price series matters for
ridership. The output stays reusable across consumers.

Monthly rule: each weekly observation is assigned to the month its date falls
in, and the month value is the mean of those weeks (skipping series with no
reading that week). Weeks straddling a month boundary are NOT prorated -- this
matches EIA's own monthly derivation closely and avoids inventing precision.
``N_WEEKS`` is emitted per month so thin partial months (series start/end) can
be dropped downstream.

Note: this is a single national series. It broadcasts identically across every
route and system -- it is a time effect, not a cross-sectional one.
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
OUTPUT_PATH: Path | None = None  # monthly frame; .parquet or .csv
DATE_FORMAT: str | None = None  # None = auto (EIA varies: 20-Aug-90, "Aug 20, 1990", ISO)
USE_LONG_NAMES = False  # False = compact tokens (gas_regular_all); True = EIA labels
WRITE_LOG = True  # also write a .txt processing log next to the output
LOG_PATH: Path | None = None  # None = derive from output path
WRITE_WEEKLY = False  # also dump the cleaned weekly frame next to the output
ROUND_DECIMALS: int | None = 4  # round monthly price means to N decimals; None = off
ID_COLS = ["MONTH_START", "YEAR", "MONTH", "N_WEEKS"]  # non-price columns in output
# --- EDIT ME >>> -------------------------------------------------------------

logger = logging.getLogger(__name__)

# EIA sourcekey -> compact column name. Keyed on the stable sourcekey rather
# than the verbose long label. Grade {allgrade,regular,midgrade,premium} x
# formulation {all,conv(entional),rfg=reformulated}, plus No 2 diesel grades.
SERIES = {
    "EMM_EPM0_PTE_NUS_DPG": "gas_allgrade_all",
    "EMM_EPM0U_PTE_NUS_DPG": "gas_allgrade_conv",
    "EMM_EPM0R_PTE_NUS_DPG": "gas_allgrade_rfg",
    "EMM_EPMR_PTE_NUS_DPG": "gas_regular_all",
    "EMM_EPMRU_PTE_NUS_DPG": "gas_regular_conv",
    "EMM_EPMRR_PTE_NUS_DPG": "gas_regular_rfg",
    "EMM_EPMM_PTE_NUS_DPG": "gas_midgrade_all",
    "EMM_EPMMU_PTE_NUS_DPG": "gas_midgrade_conv",
    "EMM_EPMMR_PTE_NUS_DPG": "gas_midgrade_rfg",
    "EMM_EPMP_PTE_NUS_DPG": "gas_premium_all",
    "EMM_EPMPU_PTE_NUS_DPG": "gas_premium_conv",
    "EMM_EPMPR_PTE_NUS_DPG": "gas_premium_rfg",
    "EMD_EPD2D_PTE_NUS_DPG": "diesel_no2_all",
    "EMD_EPD2DXL0_PTE_NUS_DPG": "diesel_no2_ulsd",
    "EMD_EPD2DM10_PTE_NUS_DPG": "diesel_no2_lsd",
}

# EIA's CSV export date format is not stable: the web "Download Data" gives
# "Aug 20, 1990" (quoted, comma inside), older/Excel copies give 20-Aug-90, and
# some give ISO. All carry a month name or 4-digit year, so they're unambiguous;
# we try these explicitly (fast, deterministic) before falling back to inference.
EIA_DATE_FORMATS = ("%d-%b-%y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y")


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


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse the EIA date column robustly across export formats.

    If DATE_FORMAT is set, honor it. Otherwise try each known EIA format in turn
    and accept the first that parses >=99% of non-blank values, then fall back to
    pandas inference. Blank cells (NOAA-style trailing padding) become NaT and are
    dropped by the caller.
    """
    raw = s.astype("string").str.strip()
    nonblank = raw.notna() & (raw != "")
    n = int(nonblank.sum())
    if n == 0:
        raise ValueError("date column is empty")
    candidate = raw.where(nonblank)

    if DATE_FORMAT is not None:
        return pd.to_datetime(candidate, format=DATE_FORMAT, errors="coerce")

    for fmt in EIA_DATE_FORMATS:
        parsed = pd.to_datetime(candidate, format=fmt, errors="coerce")
        if int(parsed.notna().sum()) / n >= 0.99:
            logger.info("parsed dates with format %r", fmt)
            return parsed

    parsed = pd.to_datetime(candidate, errors="coerce")  # inference fallback
    ok = int(parsed.notna().sum())
    logger.info("parsed dates by inference (%.1f%% of non-blank values)", 100 * ok / n)
    if ok / n < 0.99:
        logger.warning("%d date value(s) failed to parse (set to NaT)", n - ok)
    return parsed


def _check_missing_weeks(dates: pd.Series) -> None:
    """Warn about missing weekly surveys.

    EIA posts every Monday, so the date axis should be a contiguous run of
    Mondays; gaps mean a skipped survey.
    """
    days = pd.to_datetime(dates).dt.normalize()
    start, end = days.min(), days.max()
    full = pd.date_range(start, end, freq="W-MON")
    missing = full.difference(pd.DatetimeIndex(days.unique()))
    total = len(full)
    if len(missing) == 0:
        logger.info("no missing weeks in %s..%s (%d Mondays)", start.date(), end.date(), total)
        return
    logger.warning(
        "%d of %d weekly surveys missing in %s..%s (%.1f%% of range)",
        len(missing),
        total,
        start.date(),
        end.date(),
        100 * len(missing) / total,
    )
    listed = [d.date().isoformat() for d in missing]
    if len(listed) <= 20:
        logger.warning("missing weeks: %s", ", ".join(listed))
    else:
        logger.warning(
            "missing weeks (first 20 of %d): %s ...", len(listed), ", ".join(listed[:20])
        )


def _coverage_report(df: pd.DataFrame, value_cols: list[str]) -> None:
    """Log per-series coverage: first/last reading and density within that span.

    Series start at different times (RFG post-1995, ULSD ~2007), so a column can
    be mostly empty over the full file yet dense over a recent model window. The
    point is to make 'which series are usable over my window' visible up front.
    """
    dates = df["DATE"]
    logger.info("per-series coverage (first reading .. last reading | n | density in span):")
    for col in value_cols:
        present = df[col].notna()
        n = int(present.sum())
        if n == 0:
            logger.info("  %-18s no readings", col)
            continue
        first, last = dates[present].min().date(), dates[present].max().date()
        span = df[(dates >= dates[present].min()) & (dates <= dates[present].max())]
        density = 100 * n / len(span) if len(span) else 0.0
        logger.info("  %-18s %s .. %s | %5d | %5.1f%%", col, first, last, n, density)


def load_eia_weekly(path: Path, *, use_long_names: bool = USE_LONG_NAMES) -> pd.DataFrame:
    """Read the EIA Data 1 CSV, stripping the header block and renaming columns.

    The file has three header lines: a banner, a ``Sourcekey`` row, and a
    long-name row. We read the long-name row as the header (row 3) and pull the
    sourcekey row separately; the two align column-for-column by position, which
    is how each long label is mapped back to its stable sourcekey -> token.
    """
    keys = pd.read_csv(path, skiprows=1, nrows=1, header=None).iloc[0].tolist()
    df = pd.read_csv(path, skiprows=2)

    rename = {df.columns[0]: "DATE"}  # first column is the date axis
    keep = [df.columns[0]]
    mapping: list[tuple[str, str, str]] = []  # (sourcekey, token, long_name)
    unknown: list[str] = []
    dropped: list[str] = []
    for i in range(1, len(df.columns)):
        long_name = str(df.columns[i]).strip()
        # A trailing comma in the export adds a phantom column whose sourcekey is
        # blank/NaN (and header "Unnamed: N"). Skip anything without a real key,
        # and guard the case where the sourcekey row is shorter than the data row.
        key_raw = keys[i] if i < len(keys) else None
        key = "" if key_raw is None or pd.isna(key_raw) else str(key_raw).strip()
        if key in ("", "nan", "NaN"):
            dropped.append(long_name)
            continue
        token = SERIES.get(key)
        if token is None:
            unknown.append(key)
            token = key  # keep the raw sourcekey rather than drop a series silently
        mapping.append((key, token, long_name))
        rename[df.columns[i]] = long_name if use_long_names else token
        keep.append(df.columns[i])

    if dropped:
        logger.info(
            "dropping %d header-artifact column(s) with no sourcekey: %s", len(dropped), dropped
        )
    if unknown:
        logger.warning("unmapped EIA sourcekeys (kept as raw keys): %s", unknown)
    logger.info("EIA series mapping (%d price columns):", len(mapping))
    for key, token, long_name in mapping:
        logger.info("  %s -> %s | %s", key, token, long_name)
    logger.info("units: US dollars per gallon (all price columns)")
    if use_long_names:
        logger.info("applying EIA long-name headers")
    else:
        logger.info("using compact token headers (USE_LONG_NAMES is off)")

    return df[keep].rename(columns=rename)


def clean_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Return a faithful, analysis-ready *weekly* copy.

    Dates are parsed, prices coerced to numeric, coverage logged, and
    all-empty series dropped.
    """
    df = df.copy()

    # 1. Parse and order the date axis.
    df["DATE"] = _parse_dates(df["DATE"])
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    _check_missing_weeks(df["DATE"])

    # 2. Coerce every price series to numeric (blanks -> NaN; no-op if clean).
    value_cols = [c for c in df.columns if c != "DATE"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

    # 3. Report coverage before pruning, so dropped series are explained.
    _coverage_report(df, value_cols)

    # 4. Drop series with zero readings in this file (e.g. a grade EIA never
    #    surveyed nationally). A NaN here is genuinely missing, not "did not
    #    occur" -- unlike the NOAA WT flags, there is nothing to fill it with.
    empty = [c for c in value_cols if df[c].isna().all()]
    if empty:
        logger.info("dropping all-empty series: %s", empty)
        df = df.drop(columns=empty)

    return df


def to_monthly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Roll the cleaned weekly frame up to one row per calendar month.

    Month value = mean of that month's weekly readings (per series, skipping
    weeks with no reading). ``N_WEEKS`` is the count of survey weeks landing in
    the month -- use it to drop thin partial months at the series boundaries.
    """
    m = weekly.copy()
    m["YEAR"] = m["DATE"].dt.year.astype("int16")
    m["MONTH"] = m["DATE"].dt.month.astype("int8")
    value_cols = [c for c in m.columns if c not in ("DATE", "YEAR", "MONTH")]

    grouped = m.groupby(["YEAR", "MONTH"], sort=True)
    monthly = grouped[value_cols].mean().reset_index()
    monthly["N_WEEKS"] = grouped.size().to_numpy()

    monthly.insert(
        0,
        "MONTH_START",
        pd.to_datetime(dict(year=monthly["YEAR"], month=monthly["MONTH"], day=1)),
    )
    monthly["YEAR"] = monthly["YEAR"].astype("int16")
    monthly["MONTH"] = monthly["MONTH"].astype("int8")
    monthly["N_WEEKS"] = monthly["N_WEEKS"].astype("int8")

    ordered = ["MONTH_START", "YEAR", "MONTH", "N_WEEKS", *value_cols]
    monthly = monthly[ordered].sort_values("MONTH_START").reset_index(drop=True)

    # Round the price means only. The mean of k weekly 3-decimal prices has a real
    # resolution of ~0.001/k, so the 4th decimal carries genuine averaging signal
    # but anything beyond is float dust. Rounding keeps output honest and makes the
    # serialized CSV/parquet deterministic and diff-stable. Ints are left alone.
    if ROUND_DECIMALS is not None:
        monthly[value_cols] = monthly[value_cols].round(ROUND_DECIMALS)
        logger.info("rounded price means to %d decimals", ROUND_DECIMALS)

    thin = monthly[monthly["N_WEEKS"] < 4]
    if not thin.empty:
        months = ", ".join(d.strftime("%Y-%m") for d in thin["MONTH_START"])
        logger.info("%d month(s) with <4 survey weeks (partial): %s", len(thin), months)
    logger.info("rolled %d weeks -> %d months", len(weekly), len(monthly))
    return monthly


def _write(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
    *,
    use_long_names: bool | None = None,
    write_log: bool | None = None,
    write_weekly: bool | None = None,
) -> pd.DataFrame:
    """Notebook entry point: clean ``input_path`` and write the monthly frame.

    Unset args fall back to the config block, resolved at call time -- so
    ``m.INPUT_PATH = ...; m.run()`` works as expected after a plain import.
    Returns the monthly frame.
    """
    _ensure_logging()
    input_path = INPUT_PATH if input_path is None else Path(input_path)
    output_path = OUTPUT_PATH if output_path is None else Path(output_path)
    use_long_names = USE_LONG_NAMES if use_long_names is None else use_long_names
    write_log = WRITE_LOG if write_log is None else write_log
    write_weekly = WRITE_WEEKLY if write_weekly is None else write_weekly

    if input_path is None:
        input_path = _prompt_path("input EIA CSV path", must_exist=True)
    if output_path is None:
        output_path = _prompt_path("output path (.parquet or .csv)", must_exist=False)

    fh = None
    if write_log:
        log_path = LOG_PATH or output_path.with_name(f"{output_path.stem}_processing_log.txt")
        fh = _add_file_log(log_path)
        logger.info("processing %s -> %s", input_path, output_path)
        logger.info("processing log -> %s", log_path)
    try:
        weekly = clean_weekly(load_eia_weekly(input_path, use_long_names=use_long_names))
        monthly = to_monthly(weekly)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write(monthly, output_path)
        logger.info("%d rows x %d cols -> %s", len(monthly), monthly.shape[1], output_path)

        if write_weekly:
            weekly_path = output_path.with_name(f"{output_path.stem}_weekly{output_path.suffix}")
            _write(weekly, weekly_path)
            logger.info("weekly frame -> %s", weekly_path)
    finally:
        if fh is not None:
            logger.removeHandler(fh)
            fh.close()
    return monthly


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
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="input EIA CSV path")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="output path (.parquet/.csv)"
    )
    parser.add_argument(
        "--long-names",
        action=argparse.BooleanOptionalAction,
        default=USE_LONG_NAMES,
        help="use EIA long labels instead of compact tokens",
    )
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=WRITE_LOG,
        help="write a .txt processing log next to the output",
    )
    parser.add_argument(
        "--weekly",
        action=argparse.BooleanOptionalAction,
        default=WRITE_WEEKLY,
        help="also write the cleaned weekly frame",
    )
    args = parser.parse_args(argv)
    run(
        args.input,
        args.output,
        use_long_names=args.long_names,
        write_log=args.log,
        write_weekly=args.weekly,
    )


if __name__ == "__main__":
    main()
