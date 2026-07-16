"""Route-level running-time features from TIDES, windowed for ridership modeling.

This is the *modeling-feature* counterpart to ``runtime_by_trip.py``. Where
that script diagnoses individual scheduled trips (per-trip stats, variation
flags, day-of-week anomalies), this one rolls observed running time up to one
value per route so it can join the route-level modeling table on the secured
box (see ``scripts/modeling/prep_features_private.py``).

It runs in two stages, mirroring the OTP pipeline (``otp_monthly_panel.py`` ->
``otp_by_route.py``) but packaged in a single file because no monthly-runtime
panel exists elsewhere to consume:

    derive -> panel        Observed trip running times (last served arrival minus
                           first served departure) are derived from a TIDES
                           ``stop_visits`` table, joined to ``route_id`` from
                           ``trips_performed``, optionally outlier-trimmed per
                           trip, and aggregated to a ``route_id`` x ``month``
                           panel (mean / median / std / observation count).
    panel  -> rollup       The panel is reduced to one row per route over a
                           configurable trailing window (e.g. last 12, 24, or
                           36 months -- 1, 2, or 3 years) using one of two
                           aggregations:
                             * naive       - pool every observation in the window
                                             (months weighted by how much data
                                             they carry).
                             * normalized  - average the monthly means (each
                                             month weighted equally, so a single
                                             heavy month cannot dominate).

The route x month panel is itself a deliverable: it feeds the future
"monthly change in runtime" model, while the rollup feeds the main route-level
ridership model.

Inputs:
    - A TIDES ``stop_visits`` CSV (stop-level actual arrival/departure events).
    - A TIDES ``trips_performed`` CSV (trip-level attributes incl. ``route_id``).

Outputs:
    - ``route_runtime_monthly.csv`` - the route x month panel.
    - ``route_runtime_by_route.csv`` - the windowed route-level rollup.
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
from typing import Dict, List, Optional, Sequence

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

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Outlier trimming: drop the shortest and longest TRIM_FRAC of runtimes per
# scheduled trip before aggregating. 0.01 = 1%, 0.05 = 5%. Set 0 to disable.
TRIM_FRAC: float = 0.01

# Trailing window length, in months, used to build the route-level rollup. The
# most recent WINDOW_MONTHS months present (up to END_MONTH, if set) are pooled.
# Any positive integer works; common cadences are 12, 24, or 36 (1, 2, or 3
# years). Set 0 to use every month available (no windowing).
WINDOW_MONTHS: int = 12

# Rollup aggregation across the window:
#   True  -> normalized: average the monthly means, so each month is weighted
#            equally regardless of how many trips it observed.
#   False -> naive: pool every observation in the window (a month with more
#            observed trips contributes proportionally more).
NORMALIZE_BY_MONTH: bool = True

# Right edge of the trailing window, as "YYYY-MM". Leave "" to anchor the window
# at the latest month present in the data. Months after END_MONTH are ignored.
END_MONTH: str = ""

# Drop (route, month) cells with fewer than this many observed trips before the
# rollup, so a month with a single stray trip cannot skew a route's average.
# Set 0 to keep every month.
MIN_OBS_PER_MONTH: int = 1

# Optional route filters (matched against route_id as a string). Empty = all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

LOG_LEVEL: int = logging.INFO

# Output filenames.
PANEL_FILENAME: str = "route_runtime_monthly.csv"
ROLLUP_FILENAME: str = "route_runtime_by_route.csv"

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a route-runtime rollup run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    trim_frac: float = TRIM_FRAC
    window_months: int = WINDOW_MONTHS
    normalize_by_month: bool = NORMALIZE_BY_MONTH
    end_month: str = END_MONTH
    min_obs_per_month: int = MIN_OBS_PER_MONTH
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()


# =============================================================================
# LOADING & RUNTIME DERIVATION
#   (mechanics mirror runtime_by_trip.py; kept self-contained on purpose)
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read ``stop_visits`` and parse the actual timestamps + stop sequence."""
    df = pd.read_csv(path, dtype=str)
    for col in ("actual_arrival_time", "actual_departure_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["trip_stop_sequence"] = pd.to_numeric(df["trip_stop_sequence"], errors="coerce")
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read ``trips_performed`` (trip-level attributes; timestamps left as str)."""
    return pd.read_csv(path, dtype=str)


def compute_trip_runtimes(stop_visits: pd.DataFrame) -> pd.DataFrame:
    """Derive observed running time for each performed trip.

    The running time is the last usable actual arrival minus the first usable
    actual departure on the trip. Skipped stop visits (no actual time) are
    excluded before picking the endpoints, so a trip's runtime spans its first
    to last *served* stop.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.

    Returns:
        DataFrame with one row per ``trip_id_performed`` and columns
        ``service_date``, ``start_time``, and ``actual_runtime_min`` (NaN when
        fewer than two usable stops exist).
    """
    work = stop_visits
    if "schedule_relationship" in work.columns:
        work = work.loc[work["schedule_relationship"].fillna("Scheduled") != "Skipped"]
    work = work.dropna(subset=["actual_arrival_time", "actual_departure_time"], how="all")
    work = work.sort_values(["trip_id_performed", "trip_stop_sequence"])

    rows: List[Dict[str, object]] = []
    for trip_id, g in work.groupby("trip_id_performed", sort=False):
        deps = g["actual_departure_time"].dropna()
        arrs = g["actual_arrival_time"].dropna()
        if deps.empty or arrs.empty:
            continue
        start = deps.iloc[0]
        end = arrs.iloc[-1]
        runtime = (end - start).total_seconds() / 60.0
        rows.append(
            {
                "trip_id_performed": trip_id,
                "service_date": g["service_date"].iloc[0],
                "start_time": start,
                "actual_runtime_min": runtime,
            }
        )

    return pd.DataFrame(rows)


def join_route_attributes(
    trip_runtimes: pd.DataFrame, trips_performed: pd.DataFrame
) -> pd.DataFrame:
    """Attach ``route_id`` / ``direction_id`` and a stable trip key.

    Canceled / non-in-service trips are dropped via the inner join. A
    ``trip_key`` is added for per-trip outlier trimming: the scheduled trip id
    when present, else a route/direction/start-label fallback so the same
    recurring trip is trimmed as one group.

    Args:
        trip_runtimes: Output of :func:`compute_trip_runtimes`.
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Per-trip-per-date runtimes with route context and ``trip_key``.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [
        c
        for c in ("route_id", "direction_id", "trip_id_scheduled", "schedule_trip_start")
        if c in trips.columns
    ]
    trips_small = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")

    merged = trip_runtimes.merge(trips_small, on="trip_id_performed", how="inner")

    if "trip_id_scheduled" in merged.columns:
        merged["trip_key"] = merged["trip_id_scheduled"].astype(str)
    else:
        start_label = merged["start_time"].dt.strftime("%H:%M")
        direction = merged["direction_id"].astype(str) if "direction_id" in merged.columns else ""
        merged["trip_key"] = merged["route_id"].astype(str) + "_" + direction + "_" + start_label
    return merged


def trim_outliers(
    df: pd.DataFrame,
    frac: float = TRIM_FRAC,
    group_col: str = "trip_key",
    value_col: str = "actual_runtime_min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations into retained and trimmed-outlier frames per trip.

    Within each ``group_col`` group, observations below the ``frac`` quantile or
    above the ``1 - frac`` quantile of ``value_col`` are treated as outliers.

    Args:
        df: Per-trip-per-date observations.
        frac: Tail fraction to trim from each end (0 disables trimming).
        group_col: Column identifying each recurring trip.
        value_col: Numeric column to trim on.

    Returns:
        ``(retained, outliers)`` DataFrames. ``outliers`` is empty when
        ``frac <= 0`` or no rows fall in the tails.
    """
    if frac <= 0 or df.empty:
        return df.copy(), df.iloc[0:0].copy()

    keep_mask = pd.Series(True, index=df.index)
    for _, sub in df.groupby(group_col, sort=False):
        vals = sub[value_col]
        if vals.notna().sum() < 2:
            continue
        lo = vals.quantile(frac)
        hi = vals.quantile(1 - frac)
        out = (vals < lo) | (vals > hi)
        keep_mask.loc[sub.index[out.to_numpy()]] = False

    return df.loc[keep_mask].copy(), df.loc[~keep_mask].copy()


# =============================================================================
# AGGREGATION: PANEL + WINDOWED ROLLUP
# =============================================================================


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``month`` column in ``YYYY-MM`` form derived from ``service_date``."""
    df = df.copy()
    df["month"] = df["service_date"].dt.strftime("%Y-%m")
    return df


def aggregate_route_month(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed runtimes to one row per ``(route_id, month)``.

    Args:
        df: Retained per-trip-per-date observations with ``route_id``, ``month``,
            and ``actual_runtime_min`` columns.

    Returns:
        Panel with columns ``route_id``, ``month``, ``n_obs``,
        ``runtime_mean_min``, ``runtime_median_min``, ``runtime_std_min``.
    """
    cols = [
        "route_id",
        "month",
        "n_obs",
        "runtime_mean_min",
        "runtime_median_min",
        "runtime_std_min",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    panel = (
        df.groupby(["route_id", "month"], dropna=False)
        .agg(
            n_obs=("actual_runtime_min", "count"),
            runtime_mean_min=("actual_runtime_min", "mean"),
            runtime_median_min=("actual_runtime_min", "median"),
            runtime_std_min=("actual_runtime_min", "std"),
        )
        .reset_index()
        .sort_values(["route_id", "month"], ignore_index=True)
    )
    return panel.round({"runtime_mean_min": 2, "runtime_median_min": 2, "runtime_std_min": 2})


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
    min_obs_per_month: int = 0,
) -> pd.DataFrame:
    """Reduce the route x month panel to one windowed row per route.

    Args:
        panel: Output of :func:`aggregate_route_month`.
        window: Month labels to include (from :func:`select_window`).
        normalize_by_month: When True, average the monthly means (equal weight
            per month); when False, pool observations (weight by ``n_obs``).
        min_obs_per_month: Drop (route, month) cells with fewer observations than
            this before reducing.

    Returns:
        Rollup with columns ``route_id``, ``runtime_mean_min``, ``n_obs``,
        ``n_months``, ``window_start``, ``window_end``, ``normalized``.
    """
    cols = [
        "route_id",
        "runtime_mean_min",
        "n_obs",
        "n_months",
        "window_start",
        "window_end",
        "normalized",
    ]
    win = sorted(set(window))
    sub = panel.loc[panel["month"].isin(win)].copy()
    if min_obs_per_month > 0:
        sub = sub.loc[sub["n_obs"] >= min_obs_per_month]
    if sub.empty:
        return pd.DataFrame(columns=cols)

    sub["_wsum"] = sub["runtime_mean_min"] * sub["n_obs"]
    grouped = sub.groupby("route_id", dropna=False).agg(
        _wsum=("_wsum", "sum"),
        n_obs=("n_obs", "sum"),
        n_months=("month", "nunique"),
        _simple=("runtime_mean_min", "mean"),
    )
    if normalize_by_month:
        grouped["runtime_mean_min"] = grouped["_simple"]
    else:
        grouped["runtime_mean_min"] = grouped["_wsum"] / grouped["n_obs"]

    out = grouped.reset_index()
    out["window_start"] = win[0] if win else ""
    out["window_end"] = win[-1] if win else ""
    out["normalized"] = bool(normalize_by_month)
    out["n_obs"] = out["n_obs"].astype(int)
    out["n_months"] = out["n_months"].astype(int)
    out["runtime_mean_min"] = out["runtime_mean_min"].round(2)
    return out[cols].sort_values("route_id", ignore_index=True)


# =============================================================================
# OUTPUT
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


def export_outputs(panel: pd.DataFrame, rollup: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Write the route x month panel and the windowed route rollup CSVs."""
    ensure_dir(out_dir)
    written: List[Path] = []

    panel_path = out_dir / PANEL_FILENAME
    panel.to_csv(panel_path, index=False)
    written.append(panel_path)

    rollup_path = out_dir / ROLLUP_FILENAME
    rollup.to_csv(rollup_path, index=False)
    written.append(rollup_path)

    return written


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
    log_path = output_dir / "runtime_by_route_runlog.txt"

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
        "ROUTE RUNTIME ROLLUP RUN LOG",
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


def run(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Execute the full route-runtime pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        Mapping with keys ``panel`` (route x month) and ``rollup`` (route level).
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)

    runtimes = compute_trip_runtimes(stop_visits)
    joined = join_route_attributes(runtimes, trips)

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        joined = joined.loc[joined["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        joined = joined.loc[~joined["route_id"].astype(str).isin(drop)]

    retained, _outliers = trim_outliers(joined, cfg.trim_frac)
    retained = add_month(retained)
    panel = aggregate_route_month(retained)

    window = select_window(panel["month"].tolist(), cfg.end_month, cfg.window_months)
    rollup = reduce_to_route(
        panel,
        window,
        normalize_by_month=cfg.normalize_by_month,
        min_obs_per_month=cfg.min_obs_per_month,
    )

    paths = export_outputs(panel, rollup, cfg.output_dir)
    for p in paths:
        logging.info("Wrote: %s", p)

    agg_label = (
        "normalized (equal month weight)"
        if cfg.normalize_by_month
        else "naive (pooled observations)"
    )
    summary_lines = [
        f"Routes in rollup:   {len(rollup)}",
        f"Months in panel:    {panel['month'].nunique() if not panel.empty else 0}",
        f"Window applied:     {window[0] if window else 'none'}.."
        f"{window[-1] if window else 'none'} ({len(window)} month(s))",
        f"Aggregation:        {agg_label}",
        f"Panel rows:         {len(panel)}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    return {"panel": panel, "rollup": rollup}


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
        description="Route-level windowed running-time features from TIDES stop_visits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--trim-frac",
        type=float,
        default=TRIM_FRAC,
        help="Tail fraction trimmed per trip (e.g. 0.01 for 1%%). 0 disables.",
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
        help="Average monthly means (--normalize) vs pool observations (--no-normalize).",
    )
    p.add_argument(
        "--end-month",
        default=END_MONTH,
        help="Right edge of the window as YYYY-MM (empty = latest month present).",
    )
    p.add_argument(
        "--min-obs-per-month",
        type=int,
        default=MIN_OBS_PER_MONTH,
        help="Drop (route, month) cells with fewer observed trips than this.",
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

    if args.stop_visits == STOP_VISITS_PATH or args.trips_performed == TRIPS_PERFORMED_PATH:
        logging.warning(
            "STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. Update the "
            "CONFIGURATION section or pass --stop-visits/--trips-performed before running."
        )
        return 2

    cfg = Config(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        trim_frac=args.trim_frac,
        window_months=args.window_months,
        normalize_by_month=args.normalize,
        end_month=args.end_month,
        min_obs_per_month=args.min_obs_per_month,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
    )

    if not cfg.stop_visits_path.exists():
        logging.warning("stop_visits not found: %s", cfg.stop_visits_path)
        return 1
    if not cfg.trips_performed_path.exists():
        logging.warning("trips_performed not found: %s", cfg.trips_performed_path)
        return 1

    run(cfg)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
