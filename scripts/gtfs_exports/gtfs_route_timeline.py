"""GTFS route timeline across many feeds (when routes begin and end over time).

The longitudinal companion to ``gtfs_route_diff.py``. That script answers "what
changed between these *two* feeds?" in depth; this one takes a whole *sequence*
of GTFS feeds in chronological order and answers "which routes existed when?"
across the entire span:

- Point ``GTFS_FEEDS`` at your feed folders, oldest first. Optionally provide
  ``CHANGE_DATES`` (the service-change date each feed took effect); otherwise
  each feed's start date is inferred from ``feed_info.txt`` / the calendar.
- Route identity is chained across consecutive feeds: a route continues through
  a ``route_id`` match, or through a *rekey* (same ``route_short_name`` and a
  stop-set Jaccard at least ``REKEY_MIN_JACCARD``, mirroring gtfs_route_diff),
  so a renumbered route stays one row instead of ending and "starting" again.
  A route_id that disappears and later returns rejoins its old row with a gap.

Outputs (CSV):
- route_timeline.csv        : one row per route, one column per feed holding the
                              route_id served in that feed ("" = not in service),
                              plus first/last active dates and gap flags
- route_timeline_events.csv : dated added / removed / reappeared / rekeyed events
- feed_periods.csv          : each feed's label, folder, effective start/end and
                              where the start date came from

Also outputs:
- route_timeline.png  : Gantt-style chart -- routes down the left side, year/month
                        along the bottom, a bar wherever the route was in service
- route_timeline.xlsx : the three tables above as sheets
- route_timeline_runlog.txt (configuration sidecar)

No arcpy / geopandas. pandas + numpy + matplotlib only.

Typical usage:
    Update ``GTFS_FEEDS``, ``CHANGE_DATES``, and ``OUTPUT_DIR`` in the
    CONFIGURATION section and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# CONFIGURATION
# =============================================================================

# === BEGIN CONFIG ===

# GTFS feed folders in chronological order, OLDEST FIRST. Each must contain
# routes.txt; trips.txt + stop_times.txt are used (when present) to chain route
# identity across renumberings. Two feeds minimum.
GTFS_FEEDS: list[str] = [
    r"Path\To\Oldest\GTFS_Folder",
    r"Path\To\Next\GTFS_Folder",
    # r"Path\To\Newest\GTFS_Folder",
]

# Optional labels for each feed, parallel to GTFS_FEEDS (e.g. "Aug 2023 pick").
# None = label each feed by its folder name (duplicates are suffixed _2, _3, ...).
FEED_LABELS: Optional[list[str]] = None

# Optional service-change dates, parallel to GTFS_FEEDS: the date each feed took
# effect, as "YYYY-MM-DD" or "YYYYMMDD". None = infer every start from the feed
# itself (feed_info.txt, then the calendar); individual entries may also be None
# to infer just that feed. Dates must be strictly increasing.
CHANGE_DATES: Optional[list[Optional[str]]] = None

# End date for the final feed's bar on the chart. None = the last feed's own
# declared end (feed_info/calendar); if that too is unavailable, the final period
# is drawn FINAL_PERIOD_FALLBACK_DAYS long with a warning.
FINAL_END_DATE: Optional[str] = None
FINAL_PERIOD_FALLBACK_DAYS: int = 90

# Folder where all outputs are written.
OUTPUT_DIR: Path = Path(r"Path\To\Output_Folder")

TIMELINE_FILENAME = "route_timeline.csv"
EVENTS_FILENAME = "route_timeline_events.csv"
PERIODS_FILENAME = "feed_periods.csv"
CHART_FILENAME = "route_timeline.png"
XLSX_FILENAME = "route_timeline.xlsx"

# When a route_id exists in one feed but not the next, try to match it to a
# not-yet-claimed route in the other feed with the same route_short_name and a
# stop-set Jaccard at least this high, and treat it as the same route renumbered
# (identical to gtfs_route_diff's rekey rule). Requires trips + stop_times.
REKEY_MIN_JACCARD: float = 0.50

# When True (the default), a failed run-log write aborts the script so analysts
# are never left with an untraced output.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

_PLACEHOLDER = "Path\\To"

# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class FeedSummary:
    """The slice of one loaded GTFS feed the timeline needs.

    ``stop_sets`` maps route_id -> the stop_ids its trips serve (empty when
    trips/stop_times are absent); ``date_bounds`` is the feed's declared
    ``(start, end, source)`` window as ``YYYYMMDD`` strings (``""`` = unknown).
    """

    routes: pd.DataFrame
    stop_sets: dict[str, frozenset[str]]
    date_bounds: tuple[str, str, str]


@dataclass(frozen=True)
class FeedPeriod:
    """One feed's slot on the timeline: its label and effective date window."""

    index: int
    label: str
    gtfs_dir: str
    start: dt.date
    end: dt.date
    start_source: str  # "change_dates" | "feed_info" | "calendar"
    route_count: int


@dataclass(eq=False)  # identity equality: two distinct empty lineages are not "equal"
class Lineage:
    """One route tracked across feeds; ``ids`` maps feed index -> route_id."""

    ids: dict[int, str] = field(default_factory=dict)
    short_name: str = ""
    long_name: str = ""

    def first_index(self) -> int:
        """Index of the first feed this route appears in."""
        return min(self.ids)

    def last_index(self) -> int:
        """Index of the last feed this route appears in."""
        return max(self.ids)

    def display_name(self) -> str:
        """Planner-facing name: short name, else long name, else latest route_id."""
        return self.short_name or self.long_name or self.ids[self.last_index()]


# =============================================================================
# Logging
# =============================================================================


def setup_logging(output_dir: Path) -> None:
    """Configure the root logger to write to console + a file in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "gtfs_route_timeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# =============================================================================
# IO helpers
# =============================================================================


def _read_csv(path: Path, usecols: Optional[object] = None) -> pd.DataFrame:
    """Read a GTFS table as all-string columns with missing values as empty strings.

    Args:
        path: Path to the ``.txt`` file.
        usecols: Optional value forwarded to ``pandas.read_csv(usecols=...)``; a
            callable is convenient for keeping only columns that exist.

    Returns:
        The parsed table; every column is ``str`` and absent values are ``""``.
    """
    return pd.read_csv(
        path,
        dtype=str,
        usecols=usecols,  # type: ignore[arg-type]
        encoding="utf-8-sig",
        low_memory=False,
        keep_default_na=False,
        na_filter=False,
    )


def load_feed_summary(gtfs_dir: Path, label: str) -> FeedSummary:
    """Load just enough of one GTFS feed for the timeline, then let the rest go.

    Only ``routes.txt`` is required. ``trips.txt`` + ``stop_times.txt`` (when both
    exist) are reduced to per-route stop sets for rekey matching; ``feed_info.txt``
    and the calendar files are reduced to declared date bounds.

    Args:
        gtfs_dir: Folder containing the GTFS feed.
        label: Human-readable feed label used in log messages.

    Returns:
        The feed's :class:`FeedSummary`.

    Raises:
        OSError: ``gtfs_dir`` or its ``routes.txt`` is missing.
    """
    if not os.path.exists(gtfs_dir):
        raise OSError(f"{label}: directory '{gtfs_dir}' does not exist.")
    routes_path = Path(gtfs_dir) / "routes.txt"
    if not routes_path.exists():
        raise OSError(f"{label}: missing required GTFS file routes.txt in '{gtfs_dir}'.")

    routes = _read_csv(routes_path)
    logging.info("%s: loaded routes.txt (%d records).", label, len(routes))

    stop_sets: dict[str, frozenset[str]] = {}
    trips_path = Path(gtfs_dir) / "trips.txt"
    st_path = Path(gtfs_dir) / "stop_times.txt"
    if trips_path.exists() and st_path.exists():
        trips = _read_csv(trips_path, usecols=lambda c: c in {"route_id", "trip_id"})
        stop_times = _read_csv(st_path, usecols=lambda c: c in {"trip_id", "stop_id"})
        stop_sets = route_stop_sets(trips, stop_times)
        logging.info(
            "%s: loaded trips.txt/stop_times.txt (%d trips, %d stop_time rows).",
            label,
            len(trips),
            len(stop_times),
        )
    else:
        logging.warning(
            "%s: trips.txt/stop_times.txt not both present; rekey matching (renumbered "
            "routes) is disabled for this feed -- only exact route_id continuity is used.",
            label,
        )

    feed_info = None
    calendar = None
    calendar_dates = None
    if (Path(gtfs_dir) / "feed_info.txt").exists():
        feed_info = _read_csv(Path(gtfs_dir) / "feed_info.txt")
    if (Path(gtfs_dir) / "calendar.txt").exists():
        calendar = _read_csv(Path(gtfs_dir) / "calendar.txt")
    if (Path(gtfs_dir) / "calendar_dates.txt").exists():
        calendar_dates = _read_csv(Path(gtfs_dir) / "calendar_dates.txt")

    return FeedSummary(
        routes=routes,
        stop_sets=stop_sets,
        date_bounds=feed_date_bounds(feed_info, calendar, calendar_dates),
    )


# =============================================================================
# Generic helpers (copied from scripts/stop_analysis/gtfs_route_diff.py)
# =============================================================================


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize a text column for comparisons (fill NA, cast to str, strip)."""
    return series.fillna("").astype(str).str.strip()


def jaccard(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Jaccard similarity of two sets; two empty sets are treated as identical (1.0)."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def route_stop_sets(trips: pd.DataFrame, stop_times: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Map each route_id to the set of stop_ids any of its trips serve."""
    if trips.empty or stop_times.empty:
        return {}
    rs = trips[["route_id", "trip_id"]].merge(
        stop_times[["trip_id", "stop_id"]], on="trip_id", how="inner"
    )
    rs = rs[["route_id", "stop_id"]].drop_duplicates()
    return {str(rid): frozenset(grp.astype(str)) for rid, grp in rs.groupby("route_id")["stop_id"]}


def parse_date(value: object) -> Optional[dt.date]:
    """Parse a ``YYYYMMDD`` or ``YYYY-MM-DD`` string to a date; None if unparseable."""
    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return dt.date(int(text[0:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def natural_key(text: str) -> tuple[object, ...]:
    """Sort key that orders embedded numbers numerically ("2" before "10")."""
    parts = re.split(r"(\d+)", str(text))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts if p != "")


# =============================================================================
# Feed date bounds / periods
# =============================================================================


def feed_date_bounds(
    feed_info: Optional[pd.DataFrame],
    calendar: Optional[pd.DataFrame],
    calendar_dates: Optional[pd.DataFrame],
) -> tuple[str, str, str]:
    """A feed's declared ``(start, end, source)`` window as ``YYYYMMDD`` strings.

    Prefers ``feed_info`` start/end dates, then the min/max over calendar +
    calendar_dates. Empty strings (source ``"unknown"``) when nothing parses.
    """
    if feed_info is not None and not feed_info.empty:
        cols = set(feed_info.columns)
        if {"feed_start_date", "feed_end_date"} <= cols:
            starts = [s for s in normalize_text(feed_info["feed_start_date"]) if parse_date(s)]
            ends = [e for e in normalize_text(feed_info["feed_end_date"]) if parse_date(e)]
            if starts and ends:
                return min(starts), max(ends), "feed_info"

    dates: list[dt.date] = []
    if calendar is not None and not calendar.empty:
        for col in ("start_date", "end_date"):
            if col in calendar.columns:
                dates += [d for d in (parse_date(x) for x in calendar[col]) if d is not None]
    if calendar_dates is not None and not calendar_dates.empty and "date" in calendar_dates.columns:
        dates += [d for d in (parse_date(x) for x in calendar_dates["date"]) if d is not None]
    if dates:
        return min(dates).strftime("%Y%m%d"), max(dates).strftime("%Y%m%d"), "calendar"
    return "", "", "unknown"


def dedupe_labels(labels: list[str]) -> list[str]:
    """Suffix repeated labels with _2, _3, ... so every feed label is unique."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label}_{seen[label]}")
    return out


def resolve_periods(
    feed_dirs: list[str],
    labels: list[str],
    summaries: list[FeedSummary],
    change_dates: Optional[list[Optional[str]]],
    final_end_date: Optional[str],
    final_fallback_days: int,
) -> list[FeedPeriod]:
    """Turn the chronological feed list into dated timeline periods.

    Each feed's start is the user's change date when given, else the feed's
    declared start (feed_info, then calendar). Each period ends where the next
    one starts; the final period ends at ``final_end_date``, else the last feed's
    declared end, else ``final_fallback_days`` after its start (with a warning).

    Args:
        feed_dirs: GTFS folders, oldest first.
        labels: Unique label per feed, parallel to ``feed_dirs``.
        summaries: Loaded feed summaries (see :func:`load_feed_summary`).
        change_dates: Optional per-feed effective dates (entries may be None).
        final_end_date: Optional explicit end for the last period.
        final_fallback_days: Final-period length when no end date is available.

    Returns:
        One :class:`FeedPeriod` per feed, in order.

    Raises:
        ValueError: Fewer than two feeds, a start date that can neither be read
            from the config nor inferred from the feed, or starts that are not
            strictly increasing (the fix in both cases is to fill CHANGE_DATES).
    """
    if len(feed_dirs) < 2:
        raise ValueError(
            "GTFS_FEEDS needs at least two feeds in chronological order; for a "
            "single before/after pair use gtfs_route_diff.py instead."
        )
    if change_dates is not None and len(change_dates) != len(feed_dirs):
        raise ValueError(
            f"CHANGE_DATES has {len(change_dates)} entries but GTFS_FEEDS has "
            f"{len(feed_dirs)}; provide one date (or None) per feed."
        )

    starts: list[dt.date] = []
    sources: list[str] = []
    for i, (feed_dir, summary) in enumerate(zip(feed_dirs, summaries)):
        override = change_dates[i] if change_dates is not None else None
        if override is not None:
            parsed = parse_date(override)
            if parsed is None:
                raise ValueError(
                    f"CHANGE_DATES[{i}] = {override!r} is not a YYYY-MM-DD or YYYYMMDD date."
                )
            starts.append(parsed)
            sources.append("change_dates")
            continue
        inferred = parse_date(summary.date_bounds[0])
        if inferred is None:
            raise ValueError(
                f"Feed {i + 1} ('{feed_dir}') declares no usable start date "
                "(no feed_info.txt dates and no calendar); set CHANGE_DATES to give "
                "each feed its effective date."
            )
        starts.append(inferred)
        sources.append(summary.date_bounds[2])

    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            raise ValueError(
                f"Feed start dates must be strictly increasing, but feed {i + 1} "
                f"('{labels[i]}') starts {starts[i]} which is not after feed {i} "
                f"('{labels[i - 1]}') at {starts[i - 1]}. Check that GTFS_FEEDS is "
                "oldest-first, or set CHANGE_DATES explicitly."
            )

    if final_end_date is not None:
        final_end = parse_date(final_end_date)
        if final_end is None:
            raise ValueError(
                f"FINAL_END_DATE = {final_end_date!r} is not a YYYY-MM-DD or YYYYMMDD date."
            )
    else:
        final_end = parse_date(summaries[-1].date_bounds[1])
    if final_end is None or final_end <= starts[-1]:
        final_end = starts[-1] + dt.timedelta(days=final_fallback_days)
        logging.warning(
            "No usable end date for the final feed; drawing its period as %d days "
            "(set FINAL_END_DATE to control this).",
            final_fallback_days,
        )

    periods: list[FeedPeriod] = []
    for i, (feed_dir, label, summary) in enumerate(zip(feed_dirs, labels, summaries)):
        end = starts[i + 1] if i + 1 < len(starts) else final_end
        periods.append(
            FeedPeriod(
                index=i,
                label=label,
                gtfs_dir=str(feed_dir),
                start=starts[i],
                end=end,
                start_source=sources[i],
                route_count=int(normalize_text(summary.routes["route_id"]).nunique()),
            )
        )
    return periods


# =============================================================================
# Route identity chaining
# =============================================================================


def match_consecutive(
    routes_prev: pd.DataFrame,
    routes_cur: pd.DataFrame,
    stops_prev: dict[str, frozenset[str]],
    stops_cur: dict[str, frozenset[str]],
    rekey_min_jaccard: float,
) -> dict[str, str]:
    """Map each continuing route's previous route_id to its current route_id.

    Routes match first on ``route_id``. Each remaining previous-only id is then
    matched to a current-only id with the same ``route_short_name`` and the
    highest stop-set Jaccard at or above ``rekey_min_jaccard`` (a *rekey*, the
    same rule as gtfs_route_diff). Rekey matching is skipped when either feed has
    no stop sets (trips/stop_times absent).

    Returns:
        Dict of previous route_id -> current route_id for continuing routes.
    """
    prev_ids = set(normalize_text(routes_prev["route_id"]))
    cur_ids = set(normalize_text(routes_cur["route_id"]))
    mapping = {rid: rid for rid in prev_ids & cur_ids}

    if not stops_prev or not stops_cur:
        return mapping

    def _shorts(routes: pd.DataFrame) -> dict[str, str]:
        if "route_short_name" not in routes.columns:
            return {}
        ids = normalize_text(routes["route_id"])
        shorts = normalize_text(routes["route_short_name"])
        return dict(zip(ids, shorts))

    short_prev, short_cur = _shorts(routes_prev), _shorts(routes_cur)
    prev_only = sorted(prev_ids - cur_ids)
    cur_only = sorted(cur_ids - prev_ids)
    claimed: set[str] = set()
    for p_id in prev_only:
        best_id, best_j = None, rekey_min_jaccard
        for c_id in cur_only:
            if c_id in claimed or short_prev.get(p_id, "") != short_cur.get(c_id, ""):
                continue
            j = jaccard(stops_prev.get(p_id, frozenset()), stops_cur.get(c_id, frozenset()))
            if j >= best_j:
                best_id, best_j = c_id, j
        if best_id is not None:
            mapping[p_id] = best_id
            claimed.add(best_id)
    return mapping


def build_lineages(summaries: list[FeedSummary], rekey_min_jaccard: float) -> list[Lineage]:
    """Chain route identity across the feed sequence into one lineage per route.

    A lineage extends feed-to-feed through :func:`match_consecutive`. A route_id
    that vanishes and later returns rejoins its dormant lineage (a service gap)
    rather than starting a new row; names are refreshed from the newest feed the
    route appears in.

    Returns:
        Lineages ordered by first appearance, then input order.
    """
    lineages: list[Lineage] = []
    by_last_id: dict[str, Lineage] = {}  # last route_id seen -> lineage (incl. dormant)
    prev_present: dict[str, Lineage] = {}  # route_id in previous feed -> lineage

    def _names(routes: pd.DataFrame) -> dict[str, tuple[str, str]]:
        ids = normalize_text(routes["route_id"])
        shorts = (
            normalize_text(routes["route_short_name"])
            if "route_short_name" in routes.columns
            else pd.Series("", index=routes.index)
        )
        longs = (
            normalize_text(routes["route_long_name"])
            if "route_long_name" in routes.columns
            else pd.Series("", index=routes.index)
        )
        return {rid: (s, lo) for rid, s, lo in zip(ids, shorts, longs)}

    for i, summary in enumerate(summaries):
        names = _names(summary.routes)
        cur_ids = sorted(names)

        continuing: dict[str, Lineage] = {}
        if i > 0:
            prev_summary = summaries[i - 1]
            mapping = match_consecutive(
                prev_summary.routes,
                summary.routes,
                prev_summary.stop_sets,
                summary.stop_sets,
                rekey_min_jaccard,
            )
            for p_id, c_id in mapping.items():
                continuing[c_id] = prev_present[p_id]

        for c_id in cur_ids:
            lineage = continuing.get(c_id)
            if lineage is None:
                dormant = by_last_id.get(c_id)
                if dormant is not None and dormant not in continuing.values():
                    lineage = dormant  # same route_id returning after a gap
                else:
                    lineage = Lineage()
                    lineages.append(lineage)
            lineage.ids[i] = c_id
            short, long_ = names[c_id]
            lineage.short_name = short or lineage.short_name
            lineage.long_name = long_ or lineage.long_name
            by_last_id[c_id] = lineage

        prev_present = {lineage.ids[i]: lineage for lineage in lineages if i in lineage.ids}

    lineages.sort(key=lambda ln: (ln.first_index(), natural_key(ln.display_name())))
    return lineages


def presence_spans(lineage: Lineage, periods: list[FeedPeriod]) -> list[tuple[dt.date, dt.date]]:
    """Merge a lineage's consecutive in-service periods into (start, end) spans."""
    spans: list[tuple[dt.date, dt.date]] = []
    for period in periods:
        if period.index not in lineage.ids:
            continue
        if spans and spans[-1][1] == period.start:
            spans[-1] = (spans[-1][0], period.end)
        else:
            spans.append((period.start, period.end))
    return spans


# =============================================================================
# Output tables
# =============================================================================


def build_timeline_table(lineages: list[Lineage], periods: list[FeedPeriod]) -> pd.DataFrame:
    """One row per route lineage, one column per feed with the route_id served.

    Fixed columns: ``route`` (display name), short/long name, ``first_active`` /
    ``last_active`` (dates the route's presence begins/ends on the timeline),
    ``feeds_present`` and ``gap_feeds`` (feeds absent between first and last).
    """
    rows: list[dict[str, object]] = []
    for lineage in lineages:
        spans = presence_spans(lineage, periods)
        row: dict[str, object] = {
            "route": lineage.display_name(),
            "route_short_name": lineage.short_name,
            "route_long_name": lineage.long_name,
            "first_active": spans[0][0].isoformat(),
            "last_active": spans[-1][1].isoformat(),
            "feeds_present": len(lineage.ids),
            "gap_feeds": lineage.last_index() - lineage.first_index() + 1 - len(lineage.ids),
        }
        for period in periods:
            row[period.label] = lineage.ids.get(period.index, "")
        rows.append(row)
    return pd.DataFrame(rows)


def build_events_table(lineages: list[Lineage], periods: list[FeedPeriod]) -> pd.DataFrame:
    """Dated change events between consecutive feeds, for every route lineage.

    Events: ``added`` (first appearance after the first feed), ``removed`` (in
    the previous feed, not this one), ``reappeared`` (returns after a gap) and
    ``rekeyed`` (continues under a new route_id). The event date is the start of
    the feed period in which the change is first visible.
    """
    rows: list[dict[str, object]] = []
    for lineage in lineages:
        for period in periods[1:]:
            i = period.index
            prev_id = lineage.ids.get(i - 1, "")
            cur_id = lineage.ids.get(i, "")
            if not prev_id and not cur_id:
                continue
            event = detail = ""
            if prev_id and not cur_id:
                returns = next((p.label for p in periods[i:] if p.index in lineage.ids), "")
                event = "removed"
                detail = f"returns in {returns}" if returns else ""
            elif cur_id and not prev_id:
                event = "added" if i == lineage.first_index() else "reappeared"
            elif prev_id != cur_id:
                event = "rekeyed"
                detail = f"{prev_id} -> {cur_id}"
            if event:
                rows.append(
                    {
                        "date": period.start.isoformat(),
                        "feed_label": period.label,
                        "event": event,
                        "route": lineage.display_name(),
                        "route_id_before": prev_id,
                        "route_id_after": cur_id,
                        "detail": detail,
                    }
                )
    frame = pd.DataFrame(
        rows,
        columns=[
            "date",
            "feed_label",
            "event",
            "route",
            "route_id_before",
            "route_id_after",
            "detail",
        ],
    )
    return frame.sort_values(["date", "event", "route"]).reset_index(drop=True)


def build_periods_table(periods: list[FeedPeriod]) -> pd.DataFrame:
    """One row per feed: label, folder, effective window, and date provenance."""
    return pd.DataFrame(
        [
            {
                "feed_label": p.label,
                "gtfs_dir": p.gtfs_dir,
                "start": p.start.isoformat(),
                "end": p.end.isoformat(),
                "start_source": p.start_source,
                "route_count": p.route_count,
            }
            for p in periods
        ]
    )


# =============================================================================
# Chart
# =============================================================================

_INK = "#0b0b0b"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_BAR = "#2a78d6"
_SURFACE = "#fcfcfb"


def render_timeline_chart(
    lineages: list[Lineage],
    periods: list[FeedPeriod],
    out_path: Path,
) -> None:
    """Draw the Gantt-style route timeline PNG.

    Routes run down the left side (first-appearing routes on top); the bottom
    axis is calendar time in year/month. A bar spans every window a route was in
    service; feed effective dates are marked with vertical hairlines, and a
    small diamond marks a feed where the route continued under a new route_id.
    """
    n = len(lineages)
    if n == 0:
        logging.warning("No routes found in any feed; skipping the timeline chart.")
        return
    fig_height = min(30.0, max(3.0, 1.8 + 0.28 * n))
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    fig.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    rekey_dates: list[dt.date] = []
    rekey_rows: list[int] = []
    for row, lineage in enumerate(lineages):
        for start, end in presence_spans(lineage, periods):
            ax.barh(
                y=row,
                width=mdates.date2num(end) - mdates.date2num(start),
                left=mdates.date2num(start),
                height=0.55,
                color=_BAR,
                linewidth=0,
                zorder=3,
            )
        for period in periods[1:]:
            prev_id = lineage.ids.get(period.index - 1)
            cur_id = lineage.ids.get(period.index)
            if prev_id and cur_id and prev_id != cur_id:
                rekey_dates.append(period.start)
                rekey_rows.append(row)
    if rekey_dates:
        ax.plot(
            mdates.date2num(rekey_dates),
            rekey_rows,
            linestyle="none",
            marker="D",
            markersize=4.5,
            markerfacecolor=_SURFACE,
            markeredgecolor=_BAR,
            markeredgewidth=1.2,
            zorder=4,
        )

    # Feed boundaries: dashed verticals (distinct from the month grid) with the
    # feed label along the top.
    for period in periods:
        x = mdates.date2num(period.start)
        ax.axvline(x, color=_BASELINE, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            period.label,
            xy=(x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, 4),
            textcoords="offset points",
            rotation=30,
            ha="left",
            va="bottom",
            fontsize=7,
            color=_INK_MUTED,
            annotation_clip=False,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels([lineage.display_name() for lineage in lineages], fontsize=8, color=_INK)
    ax.set_ylim(n - 0.5, -0.5)  # first-appearing routes at the top
    ax.set_xlim(
        mdates.date2num(periods[0].start) - 5,
        mdates.date2num(periods[-1].end) + 5,
    )

    locator = mdates.AutoDateLocator(minticks=4, maxticks=14)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", colors=_INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=_GRIDLINE, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)

    ax.set_title("Route service timeline", loc="left", fontsize=11, color=_INK, pad=30)
    if rekey_dates:
        # In the figure's title band, clear of the bars and the feed labels.
        fig.legend(
            handles=[
                Patch(color=_BAR, label="In service"),
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="D",
                    markersize=4.5,
                    markerfacecolor=_SURFACE,
                    markeredgecolor=_BAR,
                    label="route_id changed",
                ),
            ],
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=8,
            labelcolor=_INK,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("Wrote: %s (%d routes x %d feeds)", out_path, n, len(periods))


# =============================================================================
# Run log
# =============================================================================


# Canonical version lives in utils/run_log.py — keep this copy in sync.
def extract_config_block(source_file: Path) -> str:
    r"""Return the text between the CONFIG markers in *source_file*.

    Reads ``source_file`` as UTF-8 text and slices out the lines strictly
    *between* the first occurrence of ``# === BEGIN CONFIG ===`` and the first
    subsequent occurrence of ``# === END CONFIG ===``.  The marker lines
    themselves are excluded; whitespace and inline comments inside the block
    are preserved verbatim.

    Args:
        source_file: Path to the Python source file to scan (typically
            ``Path(__file__)`` from the calling script).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    _BEGIN = "# === BEGIN CONFIG ==="
    _END = "# === END CONFIG ==="

    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == _BEGIN:
            begin_idx = i
        elif begin_idx is not None and stripped == _END:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. Expected '{_BEGIN}' and '{_END}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def write_run_log(output_dir: Path, periods: list[FeedPeriod]) -> bool:
    """Write the configuration run-log sidecar into *output_dir*.

    The resolved feed periods are appended so the log records exactly which
    folder covered which dates in this run.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "route_timeline_runlog.txt"

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

    lines: list[str] = [
        "=" * 72,
        "GTFS ROUTE TIMELINE RUN LOG",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "",
        "-" * 72,
        "RESOLVED FEED PERIODS",
        "-" * 72,
        build_periods_table(periods).to_string(index=False),
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
# Orchestration (notebook-friendly)
# =============================================================================


def run_timeline(
    feed_dirs: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    change_dates: Optional[list[Optional[str]]] = None,
    out_dir: Path = OUTPUT_DIR,
    rekey_min_jaccard: float = REKEY_MIN_JACCARD,
) -> pd.DataFrame:
    """Run the timeline end-to-end and write all outputs (notebook-friendly).

    Args:
        feed_dirs: GTFS folders, oldest first (defaults to ``GTFS_FEEDS``).
        labels: Optional per-feed labels (defaults to ``FEED_LABELS`` / folder names).
        change_dates: Optional per-feed effective dates (defaults to ``CHANGE_DATES``).
        out_dir: Output folder (defaults to ``OUTPUT_DIR``).
        rekey_min_jaccard: Rekey-matching threshold (defaults to the config knob).

    Returns:
        The route timeline table (also written to ``route_timeline.csv``).

    Raises:
        OSError: The run log could not be written and ``REQUIRE_RUN_LOG`` is True.
    """
    feed_dirs = list(feed_dirs if feed_dirs is not None else GTFS_FEEDS)
    if change_dates is None:
        change_dates = CHANGE_DATES
    raw_labels = list(labels if labels is not None else (FEED_LABELS or []))
    if raw_labels and len(raw_labels) != len(feed_dirs):
        raise ValueError(
            f"FEED_LABELS has {len(raw_labels)} entries but GTFS_FEEDS has "
            f"{len(feed_dirs)}; provide one label per feed or None."
        )
    if not raw_labels:
        raw_labels = [Path(d).name for d in feed_dirs]
    feed_labels = dedupe_labels(raw_labels)

    setup_logging(Path(out_dir))
    logging.info("Feeds (%d, oldest first): %s", len(feed_dirs), ", ".join(feed_labels))
    logging.info("Output dir: %s", out_dir)

    summaries = [load_feed_summary(Path(d), label) for d, label in zip(feed_dirs, feed_labels)]
    periods = resolve_periods(
        feed_dirs,
        feed_labels,
        summaries,
        change_dates,
        FINAL_END_DATE,
        FINAL_PERIOD_FALLBACK_DAYS,
    )
    for p in periods:
        logging.info(
            "Period %d: %s | %s .. %s (start from %s) | %d routes",
            p.index + 1,
            p.label,
            p.start,
            p.end,
            p.start_source,
            p.route_count,
        )

    lineages = build_lineages(summaries, rekey_min_jaccard)
    timeline = build_timeline_table(lineages, periods)
    events = build_events_table(lineages, periods)
    periods_df = build_periods_table(periods)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for frame, filename in (
        (timeline, TIMELINE_FILENAME),
        (events, EVENTS_FILENAME),
        (periods_df, PERIODS_FILENAME),
    ):
        path = out / filename
        frame.to_csv(path, index=False, encoding="utf-8")
        logging.info("Wrote: %s (%d rows)", path, len(frame))

    xlsx_path = out / XLSX_FILENAME
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        timeline.to_excel(writer, sheet_name="timeline", index=False)
        events.to_excel(writer, sheet_name="events", index=False)
        periods_df.to_excel(writer, sheet_name="feed_periods", index=False)
    logging.info("Wrote: %s", xlsx_path)

    render_timeline_chart(lineages, periods, out / CHART_FILENAME)

    if not write_run_log(out, periods) and REQUIRE_RUN_LOG:
        raise OSError(
            "Run log could not be written and REQUIRE_RUN_LOG is True; outputs "
            "would be untraceable. Fix the output location or set REQUIRE_RUN_LOG "
            "to False for read-only destinations."
        )

    logging.info(
        "Done. %d routes across %d feeds | events: %d added, %d removed, %d reappeared, %d rekeyed",
        len(lineages),
        len(periods),
        int((events["event"] == "added").sum()),
        int((events["event"] == "removed").sum()),
        int((events["event"] == "reappeared").sum()),
        int((events["event"] == "rekeyed").sum()),
    )
    return timeline


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if any(_PLACEHOLDER in str(d) for d in GTFS_FEEDS) or _PLACEHOLDER in str(OUTPUT_DIR):
        logging.warning(
            "GTFS_FEEDS / OUTPUT_DIR are still placeholders. Update the CONFIG "
            "block before running."
        )
        return

    run_timeline()
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
