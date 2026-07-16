"""Segment-level running times from TIDES ``stop_visits``.

A *segment* is the leg between two consecutive timepoint stops on a trip. For
each segment this script measures the **actual** running time (from the
departure of the upstream timepoint to the arrival at the downstream timepoint)
and the **scheduled** running time, and reports their difference. Segments are
keyed by ``route_id``/``direction_id`` (joined from ``trips_performed``) and the
ordered stop pair, so the same physical leg lines up across trips and dates.

Two views are exported:

  1) **Long table (for computers)** - one row per
     ``(route, direction, segment, service_date, trip)`` with actual, scheduled,
     and difference minutes. Easy to filter, pivot, or load into a database.
  2) **Pivot table (for humans)** - one wide CSV per ``(route, direction)`` with
     segments as rows and a small set of summary columns:
       * ``actual_median_min``     - median observed running time
       * ``actual_avg_min``        - mean observed running time
       * ``actual_pNN_min``        - observed running-time percentiles (one
         column per entry in ``PERCENTILES``, e.g. ``actual_p85_min``)
       * ``scheduled_min``         - modal scheduled running time
       * ``diff_min``              - actual median minus scheduled
       * ``recovery_after_min``    - median scheduled recovery (layover) time in
         the block immediately *after* the trips that traverse this segment,
         i.e. how much slack exists before the next trip on the same block.

Recovery time is derived from ``trips_performed`` block chaining: for each trip
the gap between its scheduled end and the next trip's scheduled start on the
same ``block_id`` is the scheduled recovery. It is attached to every segment of
the trip so planners can see, per corridor leg, how much downstream slack is
available to absorb overruns.

Outputs
-------
All files land in ``OUTPUT_DIR`` (``--output-dir``):

- ``segment_runtime_long.csv`` - the long table: one row per
  (route, direction, segment, service_date, trip) with actual, scheduled, and
  difference minutes.
- ``segment_runtime_summary.csv`` - one row per (route, direction, segment)
  with the summary columns described above (median/mean/percentile actuals,
  modal schedule, difference, and downstream recovery).
- ``pivots/segment_runtime_<route>_dir<direction>.csv`` - one human-readable
  pivot per (route, direction) with segments as rows in stop order.

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
from typing import Callable, Dict, List, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

STOP_VISITS_PATH: str = r"Path\To\Your\stop_visits.csv"
TRIPS_PERFORMED_PATH: str = r"Path\To\Your\trips_performed.csv"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Build segments between consecutive timepoint stops only (the usual schedule
# building blocks). Set False to build segments between every consecutive stop.
TIMEPOINTS_ONLY: bool = True

# Optional route filters (matched against route_id as a string). Empty = all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# Minimum observations required for a segment to appear in the human pivot.
MIN_OBS_FOR_PIVOT: int = 1

# Percentiles (0-100) of the actual running time reported per segment, in
# addition to the median and mean. Each value becomes an ``actual_pNN_min``
# column. Override here or with the --percentiles CLI flag.
PERCENTILES: Sequence[float] = (1, 5, 85, 95, 99)

LOG_LEVEL: int = logging.INFO

LONG_FILENAME: str = "segment_runtime_long.csv"

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a segment-runtime run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    timepoints_only: bool = TIMEPOINTS_ONLY
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()
    min_obs_for_pivot: int = MIN_OBS_FOR_PIVOT
    percentiles: Sequence[float] = PERCENTILES


# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read ``stop_visits`` and parse timestamps + numeric sequence.

    Args:
        path: Path to the ``stop_visits`` CSV.

    Returns:
        DataFrame with parsed timestamp columns, a parsed ``service_date``, and
        an integer-friendly ``trip_stop_sequence``.
    """
    df = pd.read_csv(path, dtype=str)
    for col in (
        "schedule_arrival_time",
        "schedule_departure_time",
        "actual_arrival_time",
        "actual_departure_time",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["trip_stop_sequence"] = pd.to_numeric(df["trip_stop_sequence"], errors="coerce")
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read ``trips_performed`` and parse the scheduled trip endpoints."""
    df = pd.read_csv(path, dtype=str)
    for col in ("schedule_trip_start", "schedule_trip_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    return df


def join_trip_attributes(stop_visits: pd.DataFrame, trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Attach route/direction/block attributes to each stop visit.

    Canceled / non-in-service trips are dropped. Joins on ``trip_id_performed``.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [
        c
        for c in ("route_id", "direction_id", "route_type_agency", "block_id")
        if c in trips.columns
    ]
    trips_small = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")
    return stop_visits.merge(trips_small, on="trip_id_performed", how="inner")


# =============================================================================
# SEGMENT CONSTRUCTION
# =============================================================================


def build_segments(df: pd.DataFrame, timepoints_only: bool = TIMEPOINTS_ONLY) -> pd.DataFrame:
    """Construct per-trip segments between consecutive (timepoint) stops.

    For each trip, stops are ordered by ``trip_stop_sequence`` and consecutive
    pairs become segments. The actual running time is the downstream arrival
    minus the upstream departure; the scheduled running time is computed the
    same way on the scheduled timestamps.

    Skipped stop visits (``schedule_relationship == 'Skipped'``) are excluded
    before pairing because they have no usable actual time; this means the
    segment chain "bridges" a skipped stop to the next available one.

    Args:
        df: Stop visits with joined trip attributes and parsed timestamps.
        timepoints_only: When True, only ``timepoint == TRUE`` stops anchor
            segment endpoints.

    Returns:
        Long DataFrame with one row per segment observation and columns:
        ``route_id``, ``direction_id``, ``route_type_agency``, ``block_id``,
        ``service_date``, ``trip_id_performed``, ``segment``, ``from_stop_id``,
        ``to_stop_id``, ``seq``, ``actual_runtime_min``, ``scheduled_runtime_min``,
        ``diff_min``.
    """
    work = df
    if "schedule_relationship" in work.columns:
        work = work.loc[work["schedule_relationship"].fillna("Scheduled") != "Skipped"]
    if timepoints_only and "timepoint" in work.columns:
        work = work.loc[work["timepoint"].astype(str).str.upper() == "TRUE"]

    work = work.sort_values(["trip_id_performed", "trip_stop_sequence"])

    rows: List[Dict[str, object]] = []
    for trip_id, g in work.groupby("trip_id_performed", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        first = g.iloc[0]
        for i in range(len(g) - 1):
            up = g.iloc[i]
            dn = g.iloc[i + 1]
            actual = (
                dn["actual_arrival_time"] - up["actual_departure_time"]
            ).total_seconds() / 60.0
            scheduled = (
                dn["schedule_arrival_time"] - up["schedule_departure_time"]
            ).total_seconds() / 60.0
            rows.append(
                {
                    "route_id": first.get("route_id"),
                    "direction_id": first.get("direction_id"),
                    "route_type_agency": first.get("route_type_agency"),
                    "block_id": first.get("block_id"),
                    "service_date": up["service_date"],
                    "trip_id_performed": trip_id,
                    "segment": f"{up['stop_id']} -> {dn['stop_id']}",
                    "from_stop_id": up["stop_id"],
                    "to_stop_id": dn["stop_id"],
                    "seq": i + 1,
                    "actual_runtime_min": actual,
                    "scheduled_runtime_min": scheduled,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["diff_min"] = out["actual_runtime_min"] - out["scheduled_runtime_min"]
    return out


# =============================================================================
# BLOCK RECOVERY
# =============================================================================


def compute_block_recovery(trips_performed: pd.DataFrame) -> pd.DataFrame:
    """Compute scheduled recovery time after each trip within its block.

    For each ``block_id`` the trips are ordered by scheduled start; the recovery
    after a trip is the next trip's scheduled start minus this trip's scheduled
    end. The final trip in a block has no successor and gets NaN.

    Args:
        trips_performed: Output of :func:`load_trips_performed` (Canceled rows
            are dropped here).

    Returns:
        DataFrame with ``trip_id_performed`` and ``recovery_after_min``.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]

    trips = trips.dropna(subset=["schedule_trip_start", "schedule_trip_end"])
    trips = trips.sort_values(["block_id", "schedule_trip_start"])

    next_start = trips.groupby("block_id")["schedule_trip_start"].shift(-1)
    recovery = (next_start - trips["schedule_trip_end"]).dt.total_seconds() / 60.0

    return pd.DataFrame(
        {
            "trip_id_performed": trips["trip_id_performed"].to_numpy(),
            "recovery_after_min": recovery.to_numpy(),
        }
    )


# =============================================================================
# SUMMARIES & PIVOTS
# =============================================================================


def _mode_or_median(series: pd.Series) -> float:
    """Return the modal value of a numeric series, breaking ties by median."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    modes = s.mode()
    if len(modes) == 1:
        return float(modes.iloc[0])
    return float(s.median())


def percentile_column(pct: float) -> str:
    """Return the summary column name for a percentile, e.g. 85 -> ``actual_p85_min``."""
    label = f"{pct:g}".replace(".", "_")
    if float(pct) < 10 and "_" not in label:
        label = f"0{label}"
    return f"actual_p{label}_min"


def _quantile_agg(pct: float) -> Callable[[pd.Series], float]:
    """Build an aggregation callable for the given percentile (0-100)."""

    def agg(series: pd.Series) -> float:
        return series.quantile(pct / 100.0)

    return agg


def summarize_segments(
    segments: pd.DataFrame,
    recovery: pd.DataFrame,
    min_obs: int = MIN_OBS_FOR_PIVOT,
    percentiles: Sequence[float] = PERCENTILES,
) -> pd.DataFrame:
    """Summarize each segment within each ``(route, direction)``.

    Args:
        segments: Long segment table from :func:`build_segments`.
        recovery: Per-trip recovery table from :func:`compute_block_recovery`.
        min_obs: Drop segments with fewer than this many observations.
        percentiles: Percentiles (0-100) of the actual running time to report,
            each as an ``actual_pNN_min`` column.

    Returns:
        DataFrame with one row per ``(route_id, direction_id, segment)`` and the
        summary columns ``n_obs``, ``actual_median_min``, ``actual_avg_min``,
        one ``actual_pNN_min`` column per requested percentile,
        ``scheduled_min``, ``diff_min``, ``recovery_after_min``.
    """
    if segments.empty:
        return segments

    seg = segments.merge(recovery, on="trip_id_performed", how="left")

    pct_cols = [percentile_column(p) for p in percentiles]
    agg_specs = {
        "n_obs": ("actual_runtime_min", "count"),
        "actual_median_min": ("actual_runtime_min", "median"),
        "actual_avg_min": ("actual_runtime_min", "mean"),
        **{col: ("actual_runtime_min", _quantile_agg(p)) for col, p in zip(pct_cols, percentiles)},
        "scheduled_min": ("scheduled_runtime_min", _mode_or_median),
        "recovery_after_min": ("recovery_after_min", "median"),
    }

    grouped = seg.groupby(["route_id", "direction_id", "segment", "seq"], dropna=False)
    summary = grouped.agg(**agg_specs).reset_index()

    summary["diff_min"] = summary["actual_median_min"] - summary["scheduled_min"]
    summary = summary.loc[summary["n_obs"] >= min_obs]

    cols = [
        "route_id",
        "direction_id",
        "seq",
        "segment",
        "n_obs",
        "actual_median_min",
        "actual_avg_min",
        *pct_cols,
        "scheduled_min",
        "diff_min",
        "recovery_after_min",
    ]
    summary = summary[cols].sort_values(["route_id", "direction_id", "seq"])
    return summary.round(2).reset_index(drop=True)


def build_human_pivots(summary: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Reshape the segment summary into one human-readable frame per route/dir.

    Returns:
        Mapping of ``"<route>_dir<direction>"`` -> a frame indexed by segment
        (in travel order) with the summary metric columns.
    """
    pivots: Dict[str, pd.DataFrame] = {}
    if summary.empty:
        return pivots
    metric_cols = [
        c for c in summary.columns if c not in ("route_id", "direction_id", "seq", "segment")
    ]
    for (route, direction), g in summary.groupby(["route_id", "direction_id"], dropna=False):
        ordered = g.sort_values("seq").set_index("segment")
        pivots[f"{route}_dir{direction}"] = ordered[metric_cols]
    return pivots


# =============================================================================
# OUTPUT
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


def export_outputs(
    segments: pd.DataFrame,
    summary: pd.DataFrame,
    pivots: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> List[Path]:
    """Write the long table, the tidy summary, and per-route human pivots."""
    ensure_dir(out_dir)
    written: List[Path] = []

    long_path = out_dir / LONG_FILENAME
    long_out = segments.copy()
    for col in ("actual_runtime_min", "scheduled_runtime_min", "diff_min"):
        if col in long_out.columns:
            long_out[col] = long_out[col].round(2)
    long_out.to_csv(long_path, index=False)
    written.append(long_path)

    summary_path = out_dir / "segment_runtime_summary.csv"
    summary.to_csv(summary_path, index=False)
    written.append(summary_path)

    pivot_dir = out_dir / "pivots"
    ensure_dir(pivot_dir)
    for name, frame in pivots.items():
        p = pivot_dir / f"segment_runtime_{name}.csv"
        frame.to_csv(p)
        written.append(p)

    return written


# =============================================================================
# PIPELINE
# =============================================================================


def run(cfg: Config) -> pd.DataFrame:
    """Execute the full segment-runtime pipeline and write all artifacts.

    Returns:
        The long segment table (also written to disk).
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)
    joined = join_trip_attributes(stop_visits, trips)

    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        joined = joined.loc[joined["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        joined = joined.loc[~joined["route_id"].astype(str).isin(drop)]

    segments = build_segments(joined, cfg.timepoints_only)
    recovery = compute_block_recovery(trips)
    summary = summarize_segments(segments, recovery, cfg.min_obs_for_pivot, cfg.percentiles)
    pivots = build_human_pivots(summary)

    paths = export_outputs(segments, summary, pivots, cfg.output_dir)
    for p in paths:
        logging.info("Wrote: %s", p)

    return segments


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(description="Segment runtimes from TIDES stop_visits.")
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--min-obs", type=int, default=MIN_OBS_FOR_PIVOT, help="Min observations per segment."
    )
    p.add_argument(
        "--percentiles",
        default=",".join(f"{p:g}" for p in PERCENTILES),
        help="Comma-separated actual-runtime percentiles (0-100) to report, e.g. '1,5,85,95,99'.",
    )
    return p


def parse_percentiles(raw: str) -> Sequence[float]:
    """Parse a comma-separated percentile list, validating each is in (0, 100)."""
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0 < value < 100:
            raise ValueError(f"Percentile out of range (0, 100): {token}")
        values.append(value)
    return tuple(values)


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
        timepoints_only=TIMEPOINTS_ONLY,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
        min_obs_for_pivot=args.min_obs,
        percentiles=parse_percentiles(args.percentiles),
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
