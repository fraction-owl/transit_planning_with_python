"""Compute route-level on-time performance (OTP), windowed for ridership modeling.

This is the windowed-rollup stage that turns the monthly route OTP table of
``otp_runtime_from_tides.py`` (``otp/otp_by_route_and_month.csv``) into one OTP value
per route, so it can join the route-level modeling table on the secured box (see
``scripts/modeling/prep_features_private.py``).

It deliberately does NOT re-score OTP from raw TIDES events. ``otp_runtime_from_tides``
already owns the OTP definition (the early/late window, timepoint-only filter,
``Scheduled``-only rule, departure-with-arrival fallback, arrival at a trip's last
stop, and last-stop treatment) and writes one row per
``(day type, route, month)`` carrying the early / on-time / late counts. Keeping the
scoring in one place means "on-time" cannot quietly come to mean two different
things; this script's only job is the temporal reduction. Point it at any CSV that
carries ``day_type`` / ``route_id`` / ``month`` / ``on_time`` / ``evaluated`` columns.

The day types in ``DAY_TYPES_TO_INCLUDE`` (all of them by default) are pooled first.
The reduction then takes the route rows over a configurable trailing window
(e.g. last 12, 24, or 36 months -- 1, 2, or 3 years) and collapses them two
possible ways:

    * naive       - sum on-time and evaluated counts across the window, then
                    divide (a month with more evaluated visits weighs more).
    * normalized  - average the monthly on-time percentages (each month weighted
                    equally, so a single heavy month cannot dominate).

Inputs:
    - ``otp_by_route_and_month.csv`` from ``otp_runtime_from_tides.py`` (or any
      CSV with the columns listed above).

Outputs:
    - ``otp_by_route.csv`` - one row per route with the windowed % on-time.
    - A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage:
    Update the CONFIGURATION paths (or pass the matching CLI flags) and run from
    a shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

# Sentinel markers used by extract_config_block / write_run_log to identify the
# configuration block within this file's source. Each string must appear exactly
# once in this file as a stand-alone comment line. Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# The route x month OTP table written by otp_runtime_from_tides.py
# (otp/otp_by_route_and_month.csv in its output folder).
OTP_MONTHLY_PATH: str = r"Path\To\Your\otp_by_route_and_month.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Day types (the table's day_type column) pooled into the rollup; empty = all of
# them. With otp_runtime_from_tides.py's default DAY_TYPES no date falls in two day
# types, so pooling all of them gives all-days OTP. If you set overlapping day types
# there, list the ones to pool here so no visit counts twice.
DAY_TYPES_TO_INCLUDE: Sequence[str] = ()

# Trailing window length, in months, for the rollup. The most recent
# WINDOW_MONTHS months present (up to END_MONTH, if set) are used. Any positive
# integer works; common cadences are 12, 24, or 36 (1, 2, or 3 years).
# 0 = all months.
WINDOW_MONTHS: int = 12

# Rollup aggregation across the window:
#   True  -> normalized: average the monthly % on-time (each month weighted
#            equally regardless of how many visits it evaluated).
#   False -> naive: sum on-time and evaluated counts across the window, then
#            divide (a month with more evaluated visits weighs more).
NORMALIZE_BY_MONTH: bool = True

# Right edge of the trailing window, as "YYYY-MM". Leave "" to anchor at the
# latest month present. Months after END_MONTH are ignored.
END_MONTH: str = ""

# Drop (route, month) cells with fewer than this many evaluated visits before the
# rollup, so a thinly-observed month cannot skew a route's average. 0 = keep all.
MIN_EVAL_PER_MONTH: int = 1

# Optional route filters (matched against route_id as a string). Empty = all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

LOG_LEVEL: int = logging.INFO

# Output filename.
ROLLUP_FILENAME: str = "otp_by_route.csv"

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Columns required from the input table (post-load).
REQUIRED_COLS: List[str] = ["day_type", "route_id", "month", "on_time", "evaluated"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for an OTP-by-route rollup run."""

    otp_monthly_path: Path
    output_dir: Path
    day_types: Sequence[str] = DAY_TYPES_TO_INCLUDE
    window_months: int = WINDOW_MONTHS
    normalize_by_month: bool = NORMALIZE_BY_MONTH
    end_month: str = END_MONTH
    min_eval_per_month: int = MIN_EVAL_PER_MONTH
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()


# =============================================================================
# LOADING
# =============================================================================


def load_otp_table(path: Path, day_types: Sequence[str] = DAY_TYPES_TO_INCLUDE) -> pd.DataFrame:
    """Read the route x month OTP table and pool the chosen day types.

    Args:
        path: Path to ``otp_by_route_and_month.csv`` (or a compatible CSV).
        day_types: Day types to pool; empty pools every day type in the table.

    Returns:
        DataFrame with columns ``route_id``, ``month``, ``on_time`` and
        ``evaluated``, one row per (route, month), with counts coerced to numeric.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required column is missing or no row has a chosen day type.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"OTP table not found: {path}. Run otp_runtime_from_tides.py first (it writes "
            "otp/otp_by_route_and_month.csv), or point --otp-monthly at the table."
        )

    df = pd.read_csv(path, dtype={"route_id": str, "day_type": str})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"OTP table '{path}' is missing required column(s): {missing}. "
            f"Expected at least {REQUIRED_COLS}."
        )

    out = df[REQUIRED_COLS].copy()
    out["day_type"] = out["day_type"].astype(str).str.strip()
    if day_types:
        wanted = {str(d).strip() for d in day_types}
        absent = sorted(wanted - set(out["day_type"]))
        if absent:
            logging.warning("DAY_TYPES_TO_INCLUDE entries not in the table: %s", ", ".join(absent))
        out = out.loc[out["day_type"].isin(wanted)]
        if out.empty:
            raise ValueError(
                f"No row of '{path}' has a day type in {sorted(wanted)}; the table holds "
                f"{sorted(set(df['day_type'].astype(str)))}."
            )
    out["route_id"] = out["route_id"].astype(str).str.strip()
    out["month"] = out["month"].astype(str).str.strip()
    out["on_time"] = pd.to_numeric(out["on_time"], errors="coerce").fillna(0.0)
    out["evaluated"] = pd.to_numeric(out["evaluated"], errors="coerce").fillna(0.0)
    pooled = out.groupby(["route_id", "month"], as_index=False)[["on_time", "evaluated"]].sum()
    return pooled.reset_index(drop=True)


# =============================================================================
# WINDOWED ROLLUP
# =============================================================================


def select_window(months: Sequence[str], end_month: str, window_months: int) -> List[str]:
    """Return the trailing window of ``YYYY-MM`` months to include in the rollup.

    Args:
        months: All month labels present in the panel (any order, may repeat).
        end_month: Right edge as ``YYYY-MM``; ``""`` anchors at the latest month.
        window_months: Number of most-recent months to keep; ``0`` keeps all.

    Returns:
        The selected month labels, chronologically. ``YYYY-MM`` sorts correctly
        as plain strings, so no date parsing is needed.
    """
    uniq = sorted({m for m in months if isinstance(m, str) and m})
    if end_month:
        uniq = [m for m in uniq if m <= end_month]
    if window_months and window_months > 0:
        uniq = uniq[-window_months:]
    return uniq


def reduce_to_route(
    panel: pd.DataFrame,
    window: Sequence[str],
    *,
    normalize_by_month: bool,
    min_eval_per_month: int = 0,
) -> pd.DataFrame:
    """Reduce the route x month OTP panel to one windowed row per route.

    Args:
        panel: Output of :func:`load_otp_panel`.
        window: Month labels to include (from :func:`select_window`).
        normalize_by_month: When True, average the monthly % on-time (equal
            weight per month); when False, pool the counts (weight by evaluated).
        min_eval_per_month: Drop (route, month) cells with fewer evaluated visits
            than this before reducing.

    Returns:
        Rollup with columns ``route_id``, ``pct_on_time``, ``on_time``,
        ``evaluated``, ``n_months``, ``window_start``, ``window_end``,
        ``normalized``.
    """
    cols = [
        "route_id",
        "pct_on_time",
        "on_time",
        "evaluated",
        "n_months",
        "window_start",
        "window_end",
        "normalized",
    ]
    win = sorted(set(window))
    sub = panel.loc[panel["month"].isin(win)].copy()
    if min_eval_per_month > 0:
        sub = sub.loc[sub["evaluated"] >= min_eval_per_month]
    sub = sub.loc[sub["evaluated"] > 0]
    if sub.empty:
        return pd.DataFrame(columns=cols)

    sub["_pct"] = sub["on_time"] / sub["evaluated"] * 100.0
    grouped = sub.groupby("route_id", dropna=False).agg(
        on_time=("on_time", "sum"),
        evaluated=("evaluated", "sum"),
        n_months=("month", "nunique"),
        _simple=("_pct", "mean"),
    )
    if normalize_by_month:
        grouped["pct_on_time"] = grouped["_simple"]
    else:
        grouped["pct_on_time"] = grouped["on_time"] / grouped["evaluated"] * 100.0

    out = grouped.reset_index()
    out["window_start"] = win[0] if win else ""
    out["window_end"] = win[-1] if win else ""
    out["normalized"] = bool(normalize_by_month)
    out["on_time"] = out["on_time"].astype(int)
    out["evaluated"] = out["evaluated"].astype(int)
    out["n_months"] = out["n_months"].astype(int)
    out["pct_on_time"] = out["pct_on_time"].round(1)
    return out[cols].sort_values("route_id", ignore_index=True)


# =============================================================================
# OUTPUT
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


def export_rollup(rollup: pd.DataFrame, out_dir: Path) -> Path:
    """Write the windowed route-level OTP rollup CSV and return its path."""
    ensure_dir(out_dir)
    rollup_path = out_dir / ROLLUP_FILENAME
    rollup.to_csv(rollup_path, index=False)
    return rollup_path


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def extract_config_block(source_file: Path) -> str:
    """Return the text between the CONFIG markers in *source_file*.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "otp_by_route_runlog.txt"

    source_file = resolve_source_file()
    if source_file is None:
        config_text = "(config block unavailable: interactive session, no __file__ on disk)"
        source_display = "<interactive>"
    else:
        try:
            config_text = extract_config_block(source_file)
        except (OSError, ValueError) as exc:
            logging.error("Could not extract config block for run log: %s", exc)
            return False
        source_display = str(source_file)

    lines: List[str] = [
        "=" * 72,
        "OTP BY ROUTE RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "BUILD SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# PIPELINE
# =============================================================================


def run(cfg: Config) -> pd.DataFrame:
    """Execute the OTP-by-route rollup and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        The windowed route-level rollup (also written to disk).
    """
    panel = load_otp_table(cfg.otp_monthly_path, cfg.day_types)

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        panel = panel.loc[panel["route_id"].isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        panel = panel.loc[~panel["route_id"].isin(drop)]

    window = select_window(panel["month"].tolist(), cfg.end_month, cfg.window_months)
    rollup = reduce_to_route(
        panel,
        window,
        normalize_by_month=cfg.normalize_by_month,
        min_eval_per_month=cfg.min_eval_per_month,
    )

    rollup_path = export_rollup(rollup, cfg.output_dir)
    logging.info("Wrote: %s", rollup_path)

    agg_label = (
        "normalized (equal month weight)" if cfg.normalize_by_month else "naive (pooled counts)"
    )
    summary_lines = [
        f"Day types pooled:   {', '.join(cfg.day_types) or '(all)'}",
        f"Routes in rollup:   {len(rollup)}",
        f"Window applied:     {window[0] if window else 'none'}.."
        f"{window[-1] if window else 'none'} ({len(window)} month(s))",
        f"Aggregation:        {agg_label}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return rollup


# =============================================================================
# CLI / MAIN
# =============================================================================


def notebook_safe_argv(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Return the argv to parse, shielding notebook kernels from stray flags.

    When a script's ``main()`` runs with no explicit ``argv`` inside a
    Jupyter/IPython kernel, ``sys.argv`` holds kernel plumbing (for example
    ``-f /path/kernel.json``) rather than flags meant for the script, and
    strict ``argparse.parse_args`` would reject it and abort.  This helper
    detects the notebook case and substitutes an empty argument list so the
    CONFIGURATION constants stay in charge, while shell runs keep strict
    parsing (a typo in a flag fails loudly instead of being silently ignored).

    Canonical implementation: ``utils/cli_helpers.py``.

    Args:
        argv: Explicit argument list passed to ``main()``, or ``None`` to
            fall back to ``sys.argv``.

    Returns:
        ``list(argv)`` when *argv* was provided; ``[]`` when running inside a
        notebook kernel; otherwise ``None`` so argparse reads ``sys.argv[1:]``.
    """
    if argv is not None:
        return list(argv)
    if "ipykernel" in sys.modules:
        return []
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Windowed route-level OTP rollup from otp_runtime_from_tides's output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--otp-monthly",
        default=OTP_MONTHLY_PATH,
        help="Path to otp_by_route_and_month.csv.",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--day-types",
        nargs="*",
        default=list(DAY_TYPES_TO_INCLUDE),
        metavar="DAY_TYPE",
        help="Day types to pool (none = all of them).",
    )
    p.add_argument(
        "--window-months",
        type=int,
        default=WINDOW_MONTHS,
        help="Trailing window length in months, e.g. 12, 24, 36 (0 = all months).",
    )
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=NORMALIZE_BY_MONTH,
        help="Average monthly %% on-time (--normalize) vs pool counts (--no-normalize).",
    )
    p.add_argument(
        "--end-month",
        default=END_MONTH,
        help="Right edge of the window as YYYY-MM; empty anchors at the latest month present.",
    )
    p.add_argument(
        "--min-eval-per-month",
        type=int,
        default=MIN_EVAL_PER_MONTH,
        help="Drop (route, month) cells with fewer evaluated visits than this.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    if args.otp_monthly == r"Path\To\Your\otp_by_route_and_month.csv":
        logging.warning(
            "OTP_MONTHLY_PATH is still a placeholder. Update the CONFIGURATION section "
            "or pass --otp-monthly before running."
        )
        return 2

    cfg = Config(
        otp_monthly_path=Path(args.otp_monthly).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        day_types=tuple(args.day_types),
        window_months=args.window_months,
        normalize_by_month=args.normalize,
        end_month=args.end_month,
        min_eval_per_month=args.min_eval_per_month,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
    )

    try:
        run(cfg)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
