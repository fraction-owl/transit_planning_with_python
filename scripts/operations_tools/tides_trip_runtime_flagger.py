"""Per-trip running-time statistics, variation flags, and data diagnostics.

This script derives observed trip running times from a TIDES ``stop_visits``
table, joins route/direction context from ``trips_performed``, and summarizes
each recurring scheduled trip (``trip_id_scheduled``) across all the dates it
ran. For every such trip it reports the mean, median, standard deviation, and
coefficient of variation of the running time, and flags trips whose runtime is
unusually variable.

Two data-quality diagnostics are produced as well:

  * **Apparent data gaps** - scheduled trips with far fewer observations than
    their peers (a sign of missing AVL/feed coverage), flagged relative to the
    median observation count.
  * **Day-of-week anomalies** - per trip, day-of-week buckets that either carry
    very little data (e.g. only a couple of Mondays observed) or whose mean
    running time departs noticeably from the trip's overall mean (e.g. Monday
    trips running materially longer/shorter than Tue-Fri). Both are common
    artifacts of pooling weekdays that actually behave differently.

Outlier trimming is applied per trip before statistics are computed: the
shortest and longest ``TRIM_FRAC`` of observations are dropped (default 1%; set
to 0.05 for 5%, etc.). The trimmed rows are written to their own CSV so nothing
is silently discarded.

Outputs:
  * ``trip_runtime_observations.csv`` - retained per-trip-per-date runtimes.
  * ``trip_runtime_outliers.csv``     - rows removed by trimming.
  * ``trip_runtime_stats.csv``        - per-trip summary statistics + flags.
  * ``trip_runtime_dow.csv``          - per-trip-per-DOW counts/means + flags.
  * PNG charts per route: runtime boxplot by trip start, and mean runtime by
    day of week.

Typical usage
-------------
Update the CONFIGURATION paths (or pass the matching CLI flags) and run from a
shell, ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; charts are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Outlier trimming: drop the shortest and longest TRIM_FRAC of runtimes per trip
# before computing statistics. 0.01 = 1%, 0.05 = 5%. Set 0 to disable.
TRIM_FRAC: float = 0.01

# A trip is flagged "high variation" when its coefficient of variation
# (std / mean of runtime) exceeds this threshold and it has enough observations.
HIGH_CV_THRESHOLD: float = 0.15
MIN_OBS_FOR_CV: int = 5

# Apparent data gap: a trip is flagged when its observation count is below
# GAP_FRAC * median(observation counts) across all trips.
GAP_FRAC: float = 0.30

# Day-of-week anomaly thresholds.
#   * A (trip, DOW) bucket is "low count" when its observations are below
#     DOW_LOW_COUNT_FRAC * the trip's median per-DOW count.
#   * A bucket's runtime is "anomalous" when its mean differs from the trip's
#     overall mean by more than DOW_RUNTIME_PCT (fraction).
DOW_LOW_COUNT_FRAC: float = 0.40
DOW_RUNTIME_PCT: float = 0.10

# Optional route filters (matched against route_id as a string). Empty = all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

LOG_LEVEL: int = logging.INFO

DOW_ORDER: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a trip-runtime flagging run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    trim_frac: float = TRIM_FRAC
    high_cv_threshold: float = HIGH_CV_THRESHOLD
    min_obs_for_cv: int = MIN_OBS_FOR_CV
    gap_frac: float = GAP_FRAC
    dow_low_count_frac: float = DOW_LOW_COUNT_FRAC
    dow_runtime_pct: float = DOW_RUNTIME_PCT
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()


# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read ``stop_visits`` and parse timestamps + numeric sequence."""
    df = pd.read_csv(path, dtype=str)
    for col in ("actual_arrival_time", "actual_departure_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["trip_stop_sequence"] = pd.to_numeric(df["trip_stop_sequence"], errors="coerce")
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read ``trips_performed`` and parse the scheduled start timestamp."""
    df = pd.read_csv(path, dtype=str)
    if "schedule_trip_start" in df.columns:
        df["schedule_trip_start"] = pd.to_datetime(df["schedule_trip_start"], errors="coerce")
    return df


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
        ``service_date``, ``start_time`` (first actual departure), and
        ``actual_runtime_min`` (NaN when fewer than two usable stops exist).
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


def join_trip_attributes(
    trip_runtimes: pd.DataFrame, trips_performed: pd.DataFrame
) -> pd.DataFrame:
    """Attach route/direction/scheduled-trip context and a day-of-week column.

    Canceled / non-in-service trips are dropped via the join. Adds:
    ``route_id``, ``direction_id``, ``trip_id_scheduled``, ``start_hhmm`` (the
    *scheduled* start label, stable across dates, used to order trips), and
    ``dow`` (day name).

    The start label is taken from the scheduled trip start rather than the
    observed departure, so the same recurring trip carries one consistent label
    on every date it ran (the actual departure jitters by seconds day to day).
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [
        c
        for c in (
            "route_id",
            "direction_id",
            "trip_id_scheduled",
            "route_type_agency",
            "schedule_trip_start",
        )
        if c in trips.columns
    ]
    trips_small = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")

    merged = trip_runtimes.merge(trips_small, on="trip_id_performed", how="inner")
    merged["dow"] = merged["service_date"].dt.day_name()

    # Prefer the scheduled start for the (stable) label; fall back to actual.
    if "schedule_trip_start" in merged.columns:
        sched_hhmm = merged["schedule_trip_start"].dt.strftime("%H:%M")
    else:
        sched_hhmm = pd.Series(pd.NA, index=merged.index)
    merged["start_hhmm"] = sched_hhmm.fillna(merged["start_time"].dt.strftime("%H:%M"))

    # Trip key falls back to a route/direction/start label when no scheduled id.
    if "trip_id_scheduled" in merged.columns:
        merged["trip_key"] = merged["trip_id_scheduled"].astype(str)
    else:
        merged["trip_key"] = (
            merged["route_id"].astype(str)
            + "_"
            + merged["direction_id"].astype(str)
            + "_"
            + merged["start_hhmm"].astype(str)
        )
    return merged


# =============================================================================
# OUTLIER TRIMMING
# =============================================================================


def trim_outliers(
    df: pd.DataFrame,
    frac: float = TRIM_FRAC,
    group_col: str = "trip_key",
    value_col: str = "actual_runtime_min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations into retained and trimmed-outlier frames.

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

    retained = df.loc[keep_mask].copy()
    outliers = df.loc[~keep_mask].copy()
    return retained, outliers


# =============================================================================
# STATISTICS & FLAGS
# =============================================================================


def _representative_label(series: pd.Series) -> str:
    """Return the most common start label, breaking ties by the earliest time."""
    s = series.dropna().astype(str)
    if s.empty:
        return ""
    counts = s.value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def compute_trip_stats(
    df: pd.DataFrame,
    high_cv_threshold: float = HIGH_CV_THRESHOLD,
    min_obs_for_cv: int = MIN_OBS_FOR_CV,
    gap_frac: float = GAP_FRAC,
) -> pd.DataFrame:
    """Compute per-trip runtime statistics and quality flags.

    Args:
        df: Retained observations (after trimming) with a ``trip_key`` column.
        high_cv_threshold: Coefficient-of-variation cutoff for the
            ``high_variation`` flag.
        min_obs_for_cv: Minimum observations before a CV flag is trusted.
        gap_frac: Fraction of the median observation count below which a trip is
            flagged as a likely data gap.

    Returns:
        DataFrame with one row per ``trip_key`` and statistics + boolean flags
        ``high_variation`` and ``data_gap``.
    """
    if df.empty:
        return df.copy()

    grouped = df.groupby(["route_id", "direction_id", "trip_key"], dropna=False)
    stats = grouped.agg(
        start_hhmm=("start_hhmm", _representative_label),
        n_obs=("actual_runtime_min", "count"),
        runtime_mean_min=("actual_runtime_min", "mean"),
        runtime_median_min=("actual_runtime_min", "median"),
        runtime_std_min=("actual_runtime_min", "std"),
        runtime_min_min=("actual_runtime_min", "min"),
        runtime_max_min=("actual_runtime_min", "max"),
    ).reset_index()

    with np.errstate(divide="ignore", invalid="ignore"):
        stats["cv"] = stats["runtime_std_min"] / stats["runtime_mean_min"]

    stats["high_variation"] = (stats["cv"] > high_cv_threshold) & (stats["n_obs"] >= min_obs_for_cv)

    median_obs = stats["n_obs"].median()
    cutoff = median_obs * gap_frac if pd.notna(median_obs) else 0.0
    stats["data_gap"] = stats["n_obs"] < cutoff

    stats = stats.sort_values(["route_id", "direction_id", "start_hhmm"]).reset_index(drop=True)
    return stats.round(
        {
            "runtime_mean_min": 2,
            "runtime_median_min": 2,
            "runtime_std_min": 2,
            "cv": 3,
        }
    )


def compute_dow_anomalies(
    df: pd.DataFrame,
    low_count_frac: float = DOW_LOW_COUNT_FRAC,
    runtime_pct: float = DOW_RUNTIME_PCT,
) -> pd.DataFrame:
    """Flag day-of-week buckets with sparse data or anomalous runtimes.

    For each ``trip_key`` the overall mean runtime is compared against the mean
    within each day-of-week. A bucket is flagged ``low_count`` when its
    observation count is far below the trip's typical per-DOW count, and
    ``runtime_anomaly`` when its mean runtime differs from the trip's overall
    mean by more than ``runtime_pct``.

    Args:
        df: Retained observations with ``trip_key`` and ``dow`` columns.
        low_count_frac: Threshold (fraction of the trip's median per-DOW count).
        runtime_pct: Relative runtime-deviation threshold (fraction).

    Returns:
        DataFrame with one row per ``(trip_key, dow)`` and the diagnostic
        columns ``n_obs``, ``dow_mean_min``, ``trip_mean_min``, ``pct_diff``,
        ``low_count``, ``runtime_anomaly``.
    """
    if df.empty:
        return df.copy()

    trip_mean = df.groupby("trip_key")["actual_runtime_min"].transform("mean")
    work = df.assign(_trip_mean=trip_mean)

    grouped = work.groupby(["route_id", "direction_id", "trip_key", "dow"], dropna=False)
    out = grouped.agg(
        n_obs=("actual_runtime_min", "count"),
        dow_mean_min=("actual_runtime_min", "mean"),
        trip_mean_min=("_trip_mean", "first"),
    ).reset_index()

    median_dow_count = out.groupby("trip_key")["n_obs"].transform("median")
    out["low_count"] = out["n_obs"] < (median_dow_count * low_count_frac)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["pct_diff"] = (out["dow_mean_min"] - out["trip_mean_min"]) / out["trip_mean_min"]
    out["runtime_anomaly"] = out["pct_diff"].abs() > runtime_pct

    # Order day names naturally where possible.
    out["dow"] = pd.Categorical(out["dow"], categories=DOW_ORDER, ordered=True)
    out = out.sort_values(["route_id", "direction_id", "trip_key", "dow"]).reset_index(drop=True)
    return out.round({"dow_mean_min": 2, "trip_mean_min": 2, "pct_diff": 3})


# =============================================================================
# PLOTS
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


def plot_route_runtime_box(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Boxplot of trip runtime, one box per scheduled trip, per route/direction.

    Each box pools all observed runtimes for one ``trip_key``; boxes are ordered
    and labeled by the trip's representative scheduled start time.
    """
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []

    for (route, direction), g in df.groupby(["route_id", "direction_id"], dropna=False):
        # One box per scheduled trip, ordered by its representative start time.
        trip_boxes: List[tuple[str, object]] = []
        for _trip_key, tg in g.groupby("trip_key", dropna=False):
            vals = tg["actual_runtime_min"].dropna().to_numpy()
            if len(vals):
                label = _representative_label(tg["start_hhmm"])
                trip_boxes.append((label, vals))
        if not trip_boxes:
            continue
        trip_boxes.sort(key=lambda t: t[0])
        labels = [lbl for lbl, _ in trip_boxes]
        data = [vals for _, vals in trip_boxes]
        plt.figure(figsize=(max(8, len(labels) * 0.5), 5))
        positions = range(1, len(data) + 1)
        plt.boxplot(data, positions=positions)
        plt.xticks(list(positions), labels, rotation=90)
        plt.title(f"Route {route} dir {direction} - runtime by trip start")
        plt.ylabel("Runtime (minutes)")
        plt.xlabel("Scheduled start (HH:MM)")
        plt.tight_layout()
        p = plots_dir / f"runtime_box_{route}_dir{direction}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        written.append(p)

    return written


def plot_dow_runtime(dow_table: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Bar chart of mean runtime by day of week, one PNG per route/direction."""
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []
    if dow_table.empty:
        return written

    for (route, direction), g in dow_table.groupby(["route_id", "direction_id"], dropna=False):
        means = g.groupby("dow", observed=True)["dow_mean_min"].mean().reindex(DOW_ORDER).dropna()
        if means.empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.bar(means.index.astype(str), means.to_numpy())
        plt.title(f"Route {route} dir {direction} - mean runtime by day of week")
        plt.ylabel("Mean runtime (minutes)")
        plt.xlabel("Day of week")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        p = plots_dir / f"runtime_dow_{route}_dir{direction}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        written.append(p)

    return written


# =============================================================================
# OUTPUT
# =============================================================================


def export_outputs(
    retained: pd.DataFrame,
    outliers: pd.DataFrame,
    stats: pd.DataFrame,
    dow_table: pd.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """Write the observation, outlier, stats, and DOW-anomaly CSVs."""
    ensure_dir(out_dir)
    written: List[Path] = []

    obs_path = out_dir / "trip_runtime_observations.csv"
    retained.round({"actual_runtime_min": 2}).to_csv(obs_path, index=False)
    written.append(obs_path)

    out_path = out_dir / "trip_runtime_outliers.csv"
    outliers.round({"actual_runtime_min": 2}).to_csv(out_path, index=False)
    written.append(out_path)

    stats_path = out_dir / "trip_runtime_stats.csv"
    stats.to_csv(stats_path, index=False)
    written.append(stats_path)

    dow_path = out_dir / "trip_runtime_dow.csv"
    dow_table.to_csv(dow_path, index=False)
    written.append(dow_path)

    return written


# =============================================================================
# PIPELINE
# =============================================================================


def run(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Execute the full trip-runtime pipeline and write all artifacts.

    Returns:
        Mapping with keys ``retained``, ``outliers``, ``stats``, ``dow``.
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)

    runtimes = compute_trip_runtimes(stop_visits)
    joined = join_trip_attributes(runtimes, trips)

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        joined = joined.loc[joined["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        joined = joined.loc[~joined["route_id"].astype(str).isin(drop)]

    retained, outliers = trim_outliers(joined, cfg.trim_frac)
    stats = compute_trip_stats(retained, cfg.high_cv_threshold, cfg.min_obs_for_cv, cfg.gap_frac)
    dow_table = compute_dow_anomalies(retained, cfg.dow_low_count_frac, cfg.dow_runtime_pct)

    paths = export_outputs(retained, outliers, stats, dow_table, cfg.output_dir)
    for p in paths:
        logging.info("Wrote: %s", p)

    box_paths = plot_route_runtime_box(retained, cfg.output_dir)
    dow_paths = plot_dow_runtime(dow_table, cfg.output_dir)
    logging.info("Wrote %d runtime charts.", len(box_paths) + len(dow_paths))

    return {"retained": retained, "outliers": outliers, "stats": stats, "dow": dow_table}


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(description="Per-trip runtime statistics and flags.")
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--trim-frac",
        type=float,
        default=TRIM_FRAC,
        help="Tail fraction trimmed per trip (e.g. 0.01 for 1%%, 0.05 for 5%%).",
    )
    p.add_argument(
        "--high-cv", type=float, default=HIGH_CV_THRESHOLD, help="CV cutoff for high variation."
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point. Validates placeholder paths before doing any work."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(argv)

    if args.stop_visits == STOP_VISITS_PATH or args.trips_performed == TRIPS_PERFORMED_PATH:
        logging.warning(
            "STOP_VISITS_PATH/TRIPS_PERFORMED_PATH are still placeholders. Update the "
            "CONFIGURATION section or pass --stop-visits/--trips-performed before running."
        )
        return

    cfg = Config(
        stop_visits_path=Path(args.stop_visits).expanduser(),
        trips_performed_path=Path(args.trips_performed).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        trim_frac=args.trim_frac,
        high_cv_threshold=args.high_cv,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
    )

    if not cfg.stop_visits_path.exists():
        logging.warning("stop_visits not found: %s", cfg.stop_visits_path)
        return
    if not cfg.trips_performed_path.exists():
        logging.warning("trips_performed not found: %s", cfg.trips_performed_path)
        return

    run(cfg)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
