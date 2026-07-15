"""On-time performance (OTP) over time from TIDES ``stop_visits``.

This script computes monthly on-time performance from a TIDES-style
``stop_visits`` table (stop-level arrival/departure events) joined to a
``trips_performed`` table (trip-level attributes such as ``route_id`` and
``direction_id``).  OTP is evaluated at each *timepoint* stop visit by
comparing the actual departure time to the scheduled departure time and
classifying the visit as Early, On-time, or Late using a configurable window.

It produces, for several aggregation levels:

  * **route/direction** - one series per ``(route_id, direction_id)``
  * **route**           - one series per ``route_id`` (both directions pooled)
  * **service type**    - one series per ``route_type_agency`` (e.g. LOCAL,
    EXPRESS) -- a stand-in for "service type" lists
  * **corridor**        - optional, one series per user-defined corridor (a
    named group of routes; see ``CORRIDORS``)
  * **overall**         - a single system-wide series

Outputs:
  1) ``otp_monthly_processed.csv`` - tidy long table with one row per
     (level, group, month) carrying the on-time / early / late counts and the
     derived percentages.
  2) ``otp_monthly_<level>.csv`` - one wide pivot per level (rows = group,
     columns = month, values = % on-time) for quick human scanning.
  3) PNG line charts of % on-time over time, one per group within each level,
     with a dashed reference line at the configured OTP standard.
  4) ``otp_coverage_monthly.csv`` - trip-level AVL coverage per route and month
     (see "AVL coverage" below), so differential coverage can be spotted before
     the OTP numbers are trusted.

Normalization
-------------
OTP is reported as a **percentage of evaluated timepoint visits** within each
(group, month) cell: ``pct_on_time = on_time / evaluated * 100``.  This makes
groups with very different service volumes directly comparable.  Only visits
that can actually be scored are counted in the denominator: ``timepoint`` is
TRUE (when ``TIMEPOINTS_ONLY`` is set), ``schedule_relationship`` is
``Scheduled`` (Skipped/Added visits have no comparable actual/used time), and a
finite scheduled-vs-actual deviation could be computed.  Raw counts are kept
alongside the percentages so cells backed by very little data can be spotted
and, if desired, re-weighted downstream.

AVL coverage (observed-only exports)
------------------------------------
Most AVL-derived ``stop_visits`` exports are *observed-only*: a row exists only
when the vehicle actually reported at the stop, so the file itself cannot show
what is missing.  A trip whose AVL unit was dead (or whose block never matched)
simply contributes no rows, and if such gaps correlate with route or time of
day, the naive visit-pooled OTP above is silently biased.  ``trips_performed``
closes that gap from the schedule side -- it lists every scheduled trip on the
service day (that is what makes ``schedule_relationship = Canceled`` possible),
so scheduled in-service trips with zero scorable timepoint visits are
recoverable even though ``stop_visits`` never mentions them.

This script therefore also writes ``otp_coverage_monthly.csv``: per route and
month (plus an overall series), the number of scheduled in-service trips, how
many of them produced at least one scored timepoint visit, the resulting
percent-of-trips-observed, and the mean scored visits per observed trip.  Cells
whose coverage falls below ``COVERAGE_WARN_PCT`` are logged as warnings.  The
OTP estimator itself stays naive (visit-pooled within each month); the coverage
table is the evidence for deciding whether that is safe -- flat, high coverage
means yes, while coverage that varies by route or month means the affected
cells deserve caution or reweighting downstream.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--stop-visits`` /
``--trips-performed`` / ``--output-dir``) and run from a shell, ArcGIS Pro's
Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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

# OTP window (minutes). A timepoint departure is "on time" when its deviation
# (actual - scheduled) falls within [EARLY_MIN, LATE_MIN], inclusive. The
# common transit convention is up to 1 minute early through 5 minutes late.
EARLY_MIN: float = -1.0
LATE_MIN: float = 5.0

# Only evaluate OTP at timepoint stops (timepoint == TRUE). Set False to score
# every Scheduled stop visit.
TIMEPOINTS_ONLY: bool = True

# Optional route filters (matched against route_id as a string). Empty = keep all.
ROUTES_TO_INCLUDE: Sequence[str] = ()
ROUTES_TO_EXCLUDE: Sequence[str] = ()

# Optional corridor definitions: corridor name -> list of route_ids. When
# non-empty, an extra "corridor" level is produced. Routes may appear in more
# than one corridor.
CORRIDORS: Mapping[str, Sequence[str]] = {}

# Agency OTP standard (percent) -- drawn as a dashed reference line on charts.
OTP_STANDARD: float = 85.0

# Use departure times for the deviation; fall back to arrival when the
# departure timestamp is missing (e.g. terminal stops on some feeds).
LOG_LEVEL: int = logging.INFO

# Warn when, within a (route, month) cell, the share of scheduled in-service
# trips that produced at least one scored timepoint visit falls below this
# percentage. Purely a logging threshold; no rows are dropped.
COVERAGE_WARN_PCT: float = 90.0

# Filenames
PROCESSED_FILENAME: str = "otp_monthly_processed.csv"
COVERAGE_FILENAME: str = "otp_coverage_monthly.csv"

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for an OTP monthly run."""

    stop_visits_path: Path
    trips_performed_path: Path
    output_dir: Path
    early_min: float = EARLY_MIN
    late_min: float = LATE_MIN
    timepoints_only: bool = TIMEPOINTS_ONLY
    routes_to_include: Sequence[str] = ()
    routes_to_exclude: Sequence[str] = ()
    corridors: Mapping[str, Sequence[str]] = field(default_factory=dict)
    otp_standard: float = OTP_STANDARD


# Aggregation levels: name -> grouping columns (besides 'month').
LEVELS: Dict[str, List[str]] = {
    "route_direction": ["route_id", "direction_id"],
    "route": ["route_id"],
    "service_type": ["route_type_agency"],
    "overall": [],
}

# =============================================================================
# LOADING & JOINING
# =============================================================================


def load_stop_visits(path: Path) -> pd.DataFrame:
    """Read a TIDES ``stop_visits`` CSV and parse its timestamp columns.

    Args:
        path: Path to the ``stop_visits`` CSV export.

    Returns:
        DataFrame with the four schedule/actual timestamp columns parsed to
        datetimes and ``service_date`` parsed to a date.
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
    return df


def load_trips_performed(path: Path) -> pd.DataFrame:
    """Read a TIDES ``trips_performed`` CSV (trip-level attributes).

    Args:
        path: Path to the ``trips_performed`` CSV export.

    Returns:
        DataFrame with string columns (timestamps left as strings; only the
        attribute columns are needed for the join here).
    """
    return pd.read_csv(path, dtype=str)


# Attributes carried over from trips_performed onto each stop visit.
_TRIP_ATTR_COLS: List[str] = [
    "route_id",
    "direction_id",
    "route_type_agency",
    "ntd_mode",
    "block_id",
]


def join_trip_attributes(
    stop_visits: pd.DataFrame,
    trips_performed: pd.DataFrame,
) -> pd.DataFrame:
    """Attach route/direction/service-type attributes to each stop visit.

    Trips that were Canceled in ``trips_performed`` are dropped (their stop
    visits are not meaningful for OTP). The join key is ``trip_id_performed``,
    which is unique per performed trip in TIDES.

    Args:
        stop_visits: Output of :func:`load_stop_visits`.
        trips_performed: Output of :func:`load_trips_performed`.

    Returns:
        Stop visits with the ``_TRIP_ATTR_COLS`` attributes joined on.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]

    attr_cols = [c for c in _TRIP_ATTR_COLS if c in trips.columns]
    trips = trips[["trip_id_performed", *attr_cols]].drop_duplicates("trip_id_performed")

    merged = stop_visits.merge(trips, on="trip_id_performed", how="inner")
    return merged


# =============================================================================
# DEVIATION & OTP SCORING
# =============================================================================


def compute_stop_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``dev_min`` column: actual minus scheduled departure, in minutes.

    Departure is the standard reference for OTP. Where a departure timestamp is
    missing (scheduled or actual), the corresponding arrival timestamp is used
    as a fallback so terminal/first stops are still scored when possible.

    Args:
        df: Stop visits with parsed timestamp columns.

    Returns:
        Copy of ``df`` with a float ``dev_min`` column (NaN where neither pair
        of timestamps is available).
    """
    df = df.copy()
    sched = df["schedule_departure_time"].fillna(df["schedule_arrival_time"])
    actual = df["actual_departure_time"].fillna(df["actual_arrival_time"])
    df["dev_min"] = (actual - sched).dt.total_seconds() / 60.0
    return df


def filter_for_otp(df: pd.DataFrame, timepoints_only: bool = TIMEPOINTS_ONLY) -> pd.DataFrame:
    """Keep only stop visits that can be scored for OTP.

    Drops non-timepoint visits (when ``timepoints_only``), visits whose
    ``schedule_relationship`` is not ``Scheduled`` (Skipped/Added carry no
    comparable used time), and visits with a missing deviation.

    Args:
        df: Stop visits with a ``dev_min`` column.
        timepoints_only: When True, retain only ``timepoint == TRUE`` rows.

    Returns:
        Filtered copy suitable for OTP aggregation.
    """
    out = df
    if timepoints_only and "timepoint" in out.columns:
        out = out.loc[out["timepoint"].astype(str).str.upper() == "TRUE"]
    if "schedule_relationship" in out.columns:
        out = out.loc[out["schedule_relationship"].fillna("Scheduled") == "Scheduled"]
    out = out.loc[out["dev_min"].notna()]
    return out.copy()


def summarize_unscorable(
    df: pd.DataFrame,
    timepoints_only: bool = TIMEPOINTS_ONLY,
) -> Dict[str, int]:
    """Break down why otherwise-eligible visits could not be scored.

    ``dev_min`` going NaN has two very different causes: no actual timestamp
    (an AVL dropout on a row the export chose to emit anyway) versus no
    schedule timestamp (a data-quality defect in the export's schedule join).
    In an observed-only export the first should be near zero, so a nonzero
    ``missing_schedule_time`` is the number to chase with the AVL vendor.

    Args:
        df: Stop visits with a ``dev_min`` column, before :func:`filter_for_otp`.
        timepoints_only: Apply the same timepoint filter as scoring, so the
            counts describe the same candidate pool.

    Returns:
        Dict with ``candidates`` (eligible visits), ``scored`` (finite
        deviation), ``missing_actual_time``, and ``missing_schedule_time``.
        Rows missing both timestamps count under ``missing_schedule_time``.
    """
    sub = df
    if timepoints_only and "timepoint" in sub.columns:
        sub = sub.loc[sub["timepoint"].astype(str).str.upper() == "TRUE"]
    if "schedule_relationship" in sub.columns:
        sub = sub.loc[sub["schedule_relationship"].fillna("Scheduled") == "Scheduled"]

    unscorable = sub.loc[sub["dev_min"].isna()]
    if "schedule_departure_time" in unscorable.columns:
        sched = unscorable["schedule_departure_time"]
        if "schedule_arrival_time" in unscorable.columns:
            sched = sched.fillna(unscorable["schedule_arrival_time"])
    else:
        sched = pd.Series(pd.NaT, index=unscorable.index)
    missing_schedule = int(sched.isna().sum())

    return {
        "candidates": int(len(sub)),
        "scored": int(len(sub) - len(unscorable)),
        "missing_actual_time": int(len(unscorable) - missing_schedule),
        "missing_schedule_time": missing_schedule,
    }


def classify_otp(
    df: pd.DataFrame,
    early_min: float = EARLY_MIN,
    late_min: float = LATE_MIN,
) -> pd.DataFrame:
    """Classify each scored visit as ``early``/``on_time``/``late``.

    Args:
        df: Stop visits with a ``dev_min`` column.
        early_min: Lower (inclusive) bound of the on-time window, in minutes
            (typically negative, e.g. -1.0).
        late_min: Upper (inclusive) bound of the on-time window, in minutes.

    Returns:
        Copy of ``df`` with a string ``otp_class`` column.
    """
    df = df.copy()
    dev = df["dev_min"]
    conditions = [dev < early_min, dev > late_min]
    df["otp_class"] = np.select(conditions, ["early", "late"], default="on_time")
    return df


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``month`` column in ``YYYY-MM`` form derived from ``service_date``."""
    df = df.copy()
    df["month"] = df["service_date"].dt.strftime("%Y-%m")
    return df


# =============================================================================
# AGGREGATION
# =============================================================================


def aggregate_otp(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate scored visits to monthly OTP counts and percentages.

    Args:
        df: Classified stop visits (must contain ``otp_class`` and ``month``).
        group_cols: Grouping columns in addition to ``month`` (may be empty for
            a system-wide aggregation).

    Returns:
        Tidy DataFrame with one row per (``*group_cols``, ``month``) and columns
        ``early``, ``on_time``, ``late``, ``evaluated``, ``pct_on_time``,
        ``pct_early``, ``pct_late``.
    """
    keys = list(group_cols) + ["month"]
    counts = df.assign(_n=1).pivot_table(
        index=keys,
        columns="otp_class",
        values="_n",
        aggfunc="sum",
        fill_value=0,
        dropna=False,
    )
    for cls in ("early", "on_time", "late"):
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[["early", "on_time", "late"]]
    counts["evaluated"] = counts.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        counts["pct_on_time"] = counts["on_time"] / counts["evaluated"] * 100.0
        counts["pct_early"] = counts["early"] / counts["evaluated"] * 100.0
        counts["pct_late"] = counts["late"] / counts["evaluated"] * 100.0

    return counts.reset_index().sort_values(keys).reset_index(drop=True)


def build_all_levels(
    df: pd.DataFrame,
    corridors: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Compute monthly OTP for every aggregation level.

    Args:
        df: Classified stop visits with a ``month`` column.
        corridors: Optional ``{corridor_name: [route_id, ...]}`` mapping. When
            provided and non-empty, a ``corridor`` level is added.

    Returns:
        Mapping of level name -> aggregated DataFrame. Each frame gains a
        ``level`` column naming its aggregation level.
    """
    results: Dict[str, pd.DataFrame] = {}
    for level, group_cols in LEVELS.items():
        agg = aggregate_otp(df, group_cols)
        agg.insert(0, "level", level)
        results[level] = agg

    if corridors:
        frames: List[pd.DataFrame] = []
        for name, routes in corridors.items():
            route_set = {str(r) for r in routes}
            sub = df.loc[df["route_id"].astype(str).isin(route_set)]
            if sub.empty:
                logging.warning("Corridor %r matched no rows; skipping.", name)
                continue
            agg = aggregate_otp(sub, [])
            agg.insert(0, "corridor", name)
            frames.append(agg)
        if frames:
            corridor_df = pd.concat(frames, ignore_index=True)
            corridor_df.insert(0, "level", "corridor")
            results["corridor"] = corridor_df

    return results


def compute_trip_coverage(
    trips_performed: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Trip-level AVL coverage per route and month, from the schedule side.

    Observed-only ``stop_visits`` exports emit rows only when the vehicle
    reported, so a trip with a dead AVL unit leaves no trace there. This builds
    the denominator from ``trips_performed`` instead: every scheduled
    in-service trip (the same Canceled / non-revenue filter as
    :func:`join_trip_attributes`, so the pools match) is checked for at least
    one scored timepoint visit.

    Args:
        trips_performed: Output of :func:`load_trips_performed`, after any
            route include/exclude filtering.
        scored: Fully scored visits (post :func:`filter_for_otp`), whose
            ``trip_id_performed`` values mark a trip as observed.

    Returns:
        Tidy DataFrame with one row per (``level``, ``route_id``, ``month``)
        for levels ``route`` and ``overall`` (``route_id`` = ``"ALL"``),
        carrying ``trips_scheduled``, ``trips_observed``,
        ``pct_trips_observed``, ``evaluated_visits``, and
        ``visits_per_observed_trip``.
    """
    trips = trips_performed.copy()
    if "schedule_relationship" in trips.columns:
        trips = trips.loc[trips["schedule_relationship"].fillna("Scheduled") != "Canceled"]
    if "trip_type" in trips.columns:
        trips = trips.loc[trips["trip_type"].fillna("In service") == "In service"]
    trips = trips.drop_duplicates("trip_id_performed").copy()
    trips["month"] = pd.to_datetime(trips["service_date"], errors="coerce").dt.strftime("%Y-%m")

    if scored.empty:
        visit_counts = pd.Series(dtype="int64")
    else:
        visit_counts = scored.groupby("trip_id_performed").size()
    trips["evaluated_visits"] = trips["trip_id_performed"].map(visit_counts).fillna(0).astype(int)
    trips["_observed"] = trips["evaluated_visits"] > 0

    def _reduce(frame: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        out = (
            frame.groupby(keys, dropna=False)
            .agg(
                trips_scheduled=("trip_id_performed", "size"),
                trips_observed=("_observed", "sum"),
                evaluated_visits=("evaluated_visits", "sum"),
            )
            .reset_index()
        )
        out["trips_observed"] = out["trips_observed"].astype(int)
        return out

    by_route = _reduce(trips, ["route_id", "month"])
    by_route.insert(0, "level", "route")
    overall = _reduce(trips, ["month"])
    overall.insert(0, "level", "overall")
    overall.insert(1, "route_id", "ALL")

    cov = pd.concat([by_route, overall], ignore_index=True)
    cov["pct_trips_observed"] = (cov["trips_observed"] / cov["trips_scheduled"] * 100.0).round(1)
    cov["visits_per_observed_trip"] = np.where(
        cov["trips_observed"] > 0,
        cov["evaluated_visits"] / cov["trips_observed"].replace(0, np.nan),
        np.nan,
    ).round(1)
    cols = [
        "level",
        "route_id",
        "month",
        "trips_scheduled",
        "trips_observed",
        "pct_trips_observed",
        "evaluated_visits",
        "visits_per_observed_trip",
    ]
    return cov[cols].sort_values(["level", "route_id", "month"], ignore_index=True)


def make_long_table(levels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-level frames into a single tidy long table.

    A ``group`` column is synthesized as a human-readable identifier for each
    series (e.g. ``"101 | 0"``, ``"101"``, ``"LOCAL"``, ``"ALL"``).
    """
    rows: List[pd.DataFrame] = []
    for level, agg in levels.items():
        frame = agg.copy()
        if level == "route_direction":
            group = frame["route_id"].astype(str) + " | " + frame["direction_id"].astype(str)
        elif level == "route":
            group = frame["route_id"].astype(str)
        elif level == "service_type":
            group = frame["route_type_agency"].astype(str)
        elif level == "corridor":
            group = frame["corridor"].astype(str)
        else:  # overall
            group = pd.Series(["ALL"] * len(frame), index=frame.index)
        frame.insert(1, "group", group)
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    front = ["level", "group", "month", "early", "on_time", "late", "evaluated"]
    ordered = front + [c for c in combined.columns if c not in front]
    return combined[ordered]


def pivot_pct_on_time(long_table: pd.DataFrame, level: str) -> pd.DataFrame:
    """Pivot one level into rows=group, columns=month, values=% on-time."""
    sub = long_table.loc[long_table["level"] == level]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="group", columns="month", values="pct_on_time", aggfunc="mean"
    ).round(1)


# =============================================================================
# OUTPUT
# =============================================================================


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _slug(value: object) -> str:
    """Return a filesystem-safe token for a group identifier."""
    txt = str(value).strip()
    for ch in (" ", "|", "/", "\\", ":"):
        txt = txt.replace(ch, "_")
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt.strip("_") or "group"


def export_coverage(coverage: pd.DataFrame, out_dir: Path) -> Path:
    """Write the trip-level AVL coverage table and return its path."""
    ensure_dir(out_dir)
    coverage_path = out_dir / COVERAGE_FILENAME
    coverage.to_csv(coverage_path, index=False)
    return coverage_path


def export_tables(long_table: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Write the long processed table plus one % on-time pivot per level."""
    ensure_dir(out_dir)
    written: List[Path] = []

    long_path = out_dir / PROCESSED_FILENAME
    long_table.to_csv(long_path, index=False)
    written.append(long_path)

    for level in long_table["level"].unique():
        pivot = pivot_pct_on_time(long_table, level)
        if pivot.empty:
            continue
        pivot_path = out_dir / f"otp_monthly_{level}.csv"
        pivot.to_csv(pivot_path)
        written.append(pivot_path)

    return written


def plot_levels(long_table: pd.DataFrame, out_dir: Path, otp_standard: float) -> List[Path]:
    """Render % on-time over time, one PNG per group within each level."""
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    written: List[Path] = []

    for level in long_table["level"].unique():
        sub = long_table.loc[long_table["level"] == level]
        months = sorted(sub["month"].dropna().unique())
        if not months:
            continue
        x = np.arange(len(months))
        for group, g in sub.groupby("group"):
            series = g.set_index("month")["pct_on_time"].reindex(months)
            if series.dropna().empty:
                continue
            plt.figure()
            plt.plot(x, series.to_numpy(dtype=float), marker="o", label="% On-time")
            plt.axhline(
                y=otp_standard,
                linestyle="--",
                color="red",
                linewidth=1,
                label=f"OTP Standard ({otp_standard:.0f}%)",
            )
            plt.xticks(ticks=x, labels=months, rotation=45, ha="right")
            plt.ylim(0, 100)
            plt.xlabel("Month")
            plt.ylabel("% On-time")
            plt.title(f"{level}: {group} - OTP over time")
            plt.legend()
            plt.tight_layout()
            fname = f"otp_{level}_{_slug(group)}.png"
            out_path = plots_dir / fname
            plt.savefig(out_path, dpi=150)
            plt.close()
            written.append(out_path)

    return written


# =============================================================================
# PIPELINE
# =============================================================================


def run(cfg: Config) -> pd.DataFrame:
    """Execute the full OTP pipeline and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        The tidy long OTP table (also written to disk).
    """
    stop_visits = load_stop_visits(cfg.stop_visits_path)
    trips = load_trips_performed(cfg.trips_performed_path)

    # Route filters are applied to trips_performed (route_id's source of truth)
    # so the OTP join and the coverage denominator see the same trip pool.
    if cfg.routes_to_include:
        keep = {str(r) for r in cfg.routes_to_include}
        trips = trips.loc[trips["route_id"].astype(str).isin(keep)]
    if cfg.routes_to_exclude:
        drop = {str(r) for r in cfg.routes_to_exclude}
        trips = trips.loc[~trips["route_id"].astype(str).isin(drop)]

    joined = join_trip_attributes(stop_visits, trips)

    deviated = compute_stop_deviations(joined)
    unscorable = summarize_unscorable(deviated, cfg.timepoints_only)
    if unscorable["missing_schedule_time"]:
        logging.warning(
            "%d of %d eligible timepoint visits lack a schedule timestamp and cannot "
            "be scored -- a data-quality defect in the export's schedule join, not an "
            "AVL gap.",
            unscorable["missing_schedule_time"],
            unscorable["candidates"],
        )
    if unscorable["missing_actual_time"]:
        logging.info(
            "%d of %d eligible timepoint visits have a schedule but no actual "
            "timestamp (within-row AVL dropouts).",
            unscorable["missing_actual_time"],
            unscorable["candidates"],
        )

    scored = (
        deviated.pipe(filter_for_otp, cfg.timepoints_only)
        .pipe(classify_otp, cfg.early_min, cfg.late_min)
        .pipe(add_month)
    )

    levels = build_all_levels(scored, cfg.corridors)
    long_table = make_long_table(levels)

    coverage = compute_trip_coverage(trips, scored)
    coverage_path = export_coverage(coverage, cfg.output_dir)
    logging.info("Wrote trip-coverage table: %s", coverage_path)
    low = coverage.loc[
        (coverage["level"] == "route") & (coverage["pct_trips_observed"] < COVERAGE_WARN_PCT)
    ]
    if not low.empty:
        logging.warning(
            "%d (route, month) cell(s) have < %.0f%% of scheduled in-service trips "
            "observed (worst: route %s in %s at %.1f%%). OTP for these cells rests on "
            "a nonrandom subset of trips -- inspect %s before trusting rollups.",
            len(low),
            COVERAGE_WARN_PCT,
            low.loc[low["pct_trips_observed"].idxmin(), "route_id"],
            low.loc[low["pct_trips_observed"].idxmin(), "month"],
            low["pct_trips_observed"].min(),
            COVERAGE_FILENAME,
        )

    paths = export_tables(long_table, cfg.output_dir)
    for p in paths:
        logging.info("Wrote table: %s", p)
    plot_paths = plot_levels(long_table, cfg.output_dir, cfg.otp_standard)
    logging.info("Wrote %d OTP charts to %s", len(plot_paths), cfg.output_dir / "plots")

    return long_table


# =============================================================================
# CLI / MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(description="Monthly OTP from TIDES stop_visits.")
    p.add_argument("--stop-visits", default=STOP_VISITS_PATH, help="Path to stop_visits CSV.")
    p.add_argument(
        "--trips-performed", default=TRIPS_PERFORMED_PATH, help="Path to trips_performed CSV."
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument("--early-min", type=float, default=EARLY_MIN, help="On-time window lower bound.")
    p.add_argument("--late-min", type=float, default=LATE_MIN, help="On-time window upper bound.")
    p.add_argument("--otp-standard", type=float, default=OTP_STANDARD, help="OTP standard (%%).")
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
        early_min=args.early_min,
        late_min=args.late_min,
        timepoints_only=TIMEPOINTS_ONLY,
        routes_to_include=ROUTES_TO_INCLUDE,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
        corridors=CORRIDORS,
        otp_standard=args.otp_standard,
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
