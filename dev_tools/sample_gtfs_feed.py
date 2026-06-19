"""Sample a small, self-consistent subset of a full GTFS feed.

Given a complete agency GTFS feed, this tool extracts one or a few routes and
writes an *abbreviated* feed — every file trimmed to just the rows that the
selected routes reference — to an output folder. The result is a valid GTFS
package small enough to commit, inspect by hand, or feed to downstream tools
(e.g. a feed combiner) as test data.

Why a generic cascade instead of per-agency logic
--------------------------------------------------
Agencies populate GTFS differently: some omit ``shapes.txt`` or
``direction_id``, some lean on ``parent_station`` hierarchies (rail), some ship
GTFS-Fares, ``pathways.txt``, ``levels.txt``, ``attributions.txt``, or files
this script has never heard of. To stay portable the sampler makes no
assumptions about which files or columns exist. It:

    * loads every ``*.txt`` table present, preserving all columns verbatim;
    * selects routes, then follows the foreign keys that are *actually present*
      in the data (routes -> trips -> stop_times -> stops, plus the lookup
      tables those rows reference);
    * resolves ``parent_station`` chains so station rows are never dangling;
    * copies any unrecognized table through untouched, with a warning.

Only rows are dropped; columns, ordering, and cell values are left as-is.

Inputs:
    - One or more :class:`SampleJob` entries in the CONFIGURATION block, each
      naming an input feed folder, an output folder, and a route selection.

Outputs:
    - An abbreviated GTFS feed per job, plus a ``sample_manifest.json``
      describing the selection and per-file row counts.

Typical usage:
    Edit the JOBS list below to point at real feeds (WMATA, Ride On, Fairfax
    Connector, ART, DASH, ...), choose routes, then run as a standalone script.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import pandas as pd

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

# Each job samples one input feed into one output folder. Pick routes by, in
# priority order: explicit ``route_ids``; else ``route_short_names``; else the
# first ``sample_n_routes`` routes as they appear in routes.txt.


@dataclass(frozen=True)
class SampleJob:
    """One feed-sampling task.

    Attributes:
        input_dir: Folder holding the full source GTFS feed (the ``*.txt``
            files, unzipped).
        output_dir: Folder to write the abbreviated feed into. Created if
            missing; existing same-named files are overwritten.
        route_ids: Exact ``route_id`` values to keep. Highest priority.
        route_short_names: ``route_short_name`` values to keep, used only when
            ``route_ids`` is empty. All matching route_ids are kept (handles
            short names shared by route variants).
        sample_n_routes: Fallback count of routes to keep (in file order) when
            neither selection list is given.
        max_trips_per_route: Cap on trips kept per
            (route_id, direction_id, service_id) group. ``0`` keeps them all.
            A small cap (e.g. 4) yields a tiny but structurally complete feed.
    """

    input_dir: Path
    output_dir: Path
    route_ids: Tuple[str, ...] = ()
    route_short_names: Tuple[str, ...] = ()
    sample_n_routes: int = 2
    max_trips_per_route: int = 0


# Replace these placeholders with real feeds. The DC-region agencies below are
# the intended targets; fill in the local paths where each feed is unzipped.
JOBS: Tuple[SampleJob, ...] = (
    SampleJob(
        input_dir=Path(r"Path\To\WMATA_GTFS"),
        output_dir=Path(r"Path\To\Output\wmata_sample"),
        route_short_names=("70", "33"),
    ),
    SampleJob(
        input_dir=Path(r"Path\To\RideOn_GTFS"),  # Montgomery County Ride On
        output_dir=Path(r"Path\To\Output\rideon_sample"),
        sample_n_routes=2,
    ),
    SampleJob(
        input_dir=Path(r"Path\To\FairfaxConnector_GTFS"),
        output_dir=Path(r"Path\To\Output\fairfax_sample"),
        sample_n_routes=2,
    ),
    SampleJob(
        input_dir=Path(r"Path\To\ART_GTFS"),  # Arlington Transit
        output_dir=Path(r"Path\To\Output\art_sample"),
        sample_n_routes=2,
    ),
    SampleJob(
        input_dir=Path(r"Path\To\DASH_GTFS"),  # Alexandria DASH
        output_dir=Path(r"Path\To\Output\dash_sample"),
        sample_n_routes=2,
    ),
)

# Logging verbosity.
LOG_LEVEL = logging.INFO

# --------------------------------------------------------------------------------------------------
# Internal constants
# --------------------------------------------------------------------------------------------------

# Tables handled by the referential cascade. Any present ``*.txt`` not listed
# here is copied through unchanged (with a warning) so nothing is silently lost.
_KNOWN_TABLES: Tuple[str, ...] = (
    "agency",
    "stops",
    "routes",
    "trips",
    "stop_times",
    "calendar",
    "calendar_dates",
    "shapes",
    "frequencies",
    "transfers",
    "fare_attributes",
    "fare_rules",
    "pathways",
    "levels",
    "attributions",
    "feed_info",
)

# ==================================================================================================
# IO HELPERS
# ==================================================================================================


def _read_table(path: Path) -> pd.DataFrame:
    """Read one GTFS ``*.txt`` table as all-string columns.

    Uses ``utf-8-sig`` to absorb a UTF-8 BOM (common in some feeds) and keeps
    blank cells as empty strings rather than NaN, so values round-trip exactly
    and ``== ""`` blank tests work.

    Args:
        path: Path to a GTFS text (CSV) file.

    Returns:
        The parsed table. An empty (header-only) file yields an empty frame.
    """
    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            encoding="utf-8-sig",
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        logging.warning("  %s is empty (no header); skipping.", path.name)
        return pd.DataFrame()


def _load_feed(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load every ``*.txt`` table in a feed folder, keyed by file stem.

    Args:
        input_dir: Folder containing an unzipped GTFS feed.

    Returns:
        Mapping of table name (e.g. ``"stop_times"``) to DataFrame.

    Raises:
        OSError: If the folder is missing or contains no ``*.txt`` files.
    """
    if not input_dir.is_dir():
        raise OSError(f"Input feed folder does not exist: {input_dir}")

    feed: Dict[str, pd.DataFrame] = {}
    for path in sorted(input_dir.glob("*.txt")):
        feed[path.stem] = _read_table(path)
        logging.info("  loaded %s (%d rows)", path.name, len(feed[path.stem]))

    if not feed:
        raise OSError(f"No GTFS *.txt files found in: {input_dir}")
    return feed


def _write_feed(feed: Dict[str, pd.DataFrame], output_dir: Path) -> Dict[str, int]:
    """Write each table back out as ``<name>.txt``, preserving column order.

    Args:
        feed: Mapping of table name to (already filtered) DataFrame.
        output_dir: Destination folder, created if needed.

    Returns:
        Mapping of table name to the number of rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for name, df in feed.items():
        out_path = output_dir / f"{name}.txt"
        df.to_csv(out_path, index=False)
        counts[name] = len(df)
    return counts


# ==================================================================================================
# FILTER HELPERS
# ==================================================================================================


def _column_values(df: Optional[pd.DataFrame], column: str) -> Set[str]:
    """Return the set of non-blank values in ``column``, or empty if absent.

    Args:
        df: Table to read, or ``None``.
        column: Column name to collect.

    Returns:
        Distinct non-empty string values found in the column.
    """
    if df is None or df.empty or column not in df.columns:
        return set()
    return {v for v in df[column].unique() if v != ""}


def _keep_rows(
    df: Optional[pd.DataFrame], column: str, allowed: Set[str], *, keep_blank: bool = False
) -> Optional[pd.DataFrame]:
    """Filter ``df`` to rows whose ``column`` value is in ``allowed``.

    A missing table or missing column is returned unchanged — the cascade only
    constrains keys that actually exist in the feed.

    Args:
        df: Table to filter, or ``None``.
        column: Foreign-key column to test.
        allowed: Permitted values.
        keep_blank: If True, also keep rows whose value is blank (used for
            optional/global references such as a fare rule with no route_id).

    Returns:
        The filtered table (a copy), or the input unchanged when the column is
        absent.
    """
    if df is None or df.empty or column not in df.columns:
        return df
    mask = df[column].isin(allowed)
    if keep_blank:
        mask = mask | (df[column] == "")
    return df[mask].copy()


def _resolve_route_ids(routes: pd.DataFrame, job: SampleJob) -> Set[str]:
    """Determine which route_ids to keep for a job.

    Args:
        routes: The feed's routes table.
        job: The sampling job describing the selection.

    Returns:
        The set of route_ids to retain.

    Raises:
        ValueError: If the selection resolves to no routes.
    """
    available = list(dict.fromkeys(routes["route_id"].tolist()))

    if job.route_ids:
        wanted = set(job.route_ids)
        selected = {r for r in available if r in wanted}
        missing = wanted - selected
        if missing:
            logging.warning("  requested route_ids not in feed: %s", ", ".join(sorted(missing)))
    elif job.route_short_names:
        if "route_short_name" not in routes.columns:
            raise ValueError("Feed has no route_short_name column; select by route_ids instead.")
        wanted = set(job.route_short_names)
        selected = set(routes.loc[routes["route_short_name"].isin(wanted), "route_id"])
        found_names = set(routes.loc[routes["route_id"].isin(selected), "route_short_name"])
        for name in sorted(wanted - found_names):
            logging.warning("  requested route_short_name not in feed: %s", name)
    else:
        selected = set(available[: max(0, job.sample_n_routes)])

    if not selected:
        raise ValueError("Route selection matched no routes; nothing to sample.")
    logging.info("  selected %d route(s): %s", len(selected), ", ".join(sorted(selected)))
    return selected


def _cap_trips(trips: pd.DataFrame, max_per_group: int) -> pd.DataFrame:
    """Limit trips to ``max_per_group`` per route/direction/service group.

    Grouping columns are intersected with those present, so feeds without
    ``direction_id`` still cap sensibly.

    Args:
        trips: Trips already filtered to the selected routes.
        max_per_group: Maximum trips per group; ``0`` disables capping.

    Returns:
        The capped trips table.
    """
    if max_per_group <= 0 or trips.empty:
        return trips
    group_cols = [c for c in ("route_id", "direction_id", "service_id") if c in trips.columns]
    if not group_cols:
        return trips.head(max_per_group).copy()
    return trips.groupby(group_cols, sort=False, group_keys=False).head(max_per_group).copy()


def _expand_parent_stations(stops: pd.DataFrame, stop_ids: Set[str]) -> Set[str]:
    """Grow a stop_id set to include all ancestor ``parent_station`` rows.

    Walks the parent chain to a fixed point so station/entrance rows referenced
    by kept platforms are never left dangling.

    Args:
        stops: The feed's stops table.
        stop_ids: Stop ids referenced directly by kept stop_times.

    Returns:
        The closure of ``stop_ids`` under the parent_station relation.
    """
    if "parent_station" not in stops.columns or "stop_id" not in stops.columns:
        return set(stop_ids)

    parent_of = dict(zip(stops["stop_id"], stops["parent_station"]))
    keep = set(stop_ids)
    frontier = set(stop_ids)
    while frontier:
        parents = {parent_of.get(sid, "") for sid in frontier}
        parents = {p for p in parents if p and p not in keep}
        keep |= parents
        frontier = parents
    return keep


# ==================================================================================================
# CASCADE
# ==================================================================================================


def sample_feed(feed: Dict[str, pd.DataFrame], job: SampleJob) -> Dict[str, pd.DataFrame]:
    """Trim a full feed down to the rows referenced by the selected routes.

    Args:
        feed: All tables loaded from the source feed.
        job: The sampling job (selection + trip cap).

    Returns:
        A new mapping of table name to filtered DataFrame, ready to write.

    Raises:
        ValueError: If the feed lacks a routes/trips/stop_times table, or the
            selection matches no routes.
    """
    for required in ("routes", "trips", "stop_times"):
        if required not in feed:
            raise ValueError(f"Feed is missing required table: {required}.txt")

    out = dict(feed)  # shallow copy; replaced entries below are filtered copies

    # --- routes -> route_ids, agency_ids ---------------------------------------------------------
    route_ids = _resolve_route_ids(feed["routes"], job)
    out["routes"] = feed["routes"][feed["routes"]["route_id"].isin(route_ids)].copy()
    agency_ids = _column_values(out["routes"], "agency_id")

    # --- trips (filtered + capped) -> trip_ids, service_ids, shape_ids ---------------------------
    trips = feed["trips"][feed["trips"]["route_id"].isin(route_ids)].copy()
    trips = _cap_trips(trips, job.max_trips_per_route)
    out["trips"] = trips
    trip_ids = _column_values(trips, "trip_id")
    service_ids = _column_values(trips, "service_id")
    shape_ids = _column_values(trips, "shape_id")

    # --- stop_times -> stop_ids ------------------------------------------------------------------
    out["stop_times"] = _keep_rows(feed.get("stop_times"), "trip_id", trip_ids)
    stop_ids = _column_values(out["stop_times"], "stop_id")

    # --- stops (+ parent stations) -> level_ids --------------------------------------------------
    if "stops" in feed:
        stop_ids = _expand_parent_stations(feed["stops"], stop_ids)
        out["stops"] = _keep_rows(feed["stops"], "stop_id", stop_ids)
    level_ids = _column_values(out.get("stops"), "level_id")

    # --- calendars, shapes, agency, frequencies --------------------------------------------------
    out["calendar"] = _keep_rows(feed.get("calendar"), "service_id", service_ids)
    out["calendar_dates"] = _keep_rows(feed.get("calendar_dates"), "service_id", service_ids)
    out["shapes"] = _keep_rows(feed.get("shapes"), "shape_id", shape_ids)
    # Keep referenced agencies; if routes carried no agency_id, leave agency.txt as-is.
    if agency_ids:
        out["agency"] = _keep_rows(feed.get("agency"), "agency_id", agency_ids, keep_blank=True)
    out["frequencies"] = _keep_rows(feed.get("frequencies"), "trip_id", trip_ids)

    # --- transfers: every present id column must reference a kept entity -------------------------
    out["transfers"] = _filter_transfers(feed.get("transfers"), stop_ids, route_ids, trip_ids)

    # --- fares: route-scoped rules -> fare_ids -> attributes -------------------------------------
    out["fare_rules"], fare_ids = _filter_fare_rules(feed.get("fare_rules"), route_ids)
    if fare_ids is not None:
        out["fare_attributes"] = _keep_rows(feed.get("fare_attributes"), "fare_id", fare_ids)

    # --- pathways / levels (station internals) ---------------------------------------------------
    out["pathways"] = _filter_pathways(feed.get("pathways"), stop_ids)
    out["levels"] = _keep_rows(feed.get("levels"), "level_id", level_ids)

    # --- attributions: keep global rows + those referencing kept ids -----------------------------
    out["attributions"] = _filter_attributions(
        feed.get("attributions"), agency_ids, route_ids, trip_ids
    )

    _warn_unknown_tables(feed)
    # Drop keys whose value is None (table absent in the source feed).
    return {name: df for name, df in out.items() if df is not None}


def _filter_transfers(
    transfers: Optional[pd.DataFrame],
    stop_ids: Set[str],
    route_ids: Set[str],
    trip_ids: Set[str],
) -> Optional[pd.DataFrame]:
    """Keep transfer rows whose every populated id column references a kept entity.

    Handles base ``transfers.txt`` (stop ids) and the optional route/trip id
    columns used by GTFS linked-trip and fares extensions.

    Args:
        transfers: The transfers table, or ``None``.
        stop_ids: Kept stop ids.
        route_ids: Kept route ids.
        trip_ids: Kept trip ids.

    Returns:
        The filtered transfers table, or the input unchanged/``None``.
    """
    if transfers is None or transfers.empty:
        return transfers
    id_columns = (
        ("from_stop_id", stop_ids),
        ("to_stop_id", stop_ids),
        ("from_route_id", route_ids),
        ("to_route_id", route_ids),
        ("from_trip_id", trip_ids),
        ("to_trip_id", trip_ids),
    )
    mask = pd.Series(True, index=transfers.index)
    for column, allowed in id_columns:
        if column in transfers.columns:
            mask &= (transfers[column] == "") | transfers[column].isin(allowed)
    return transfers[mask].copy()


def _filter_fare_rules(
    fare_rules: Optional[pd.DataFrame], route_ids: Set[str]
) -> Tuple[Optional[pd.DataFrame], Optional[Set[str]]]:
    """Filter fare_rules to the kept routes and report the fare_ids they use.

    Zone-based scoping (origin_id/destination_id/contains_id vs. stop zone_id)
    is intentionally not applied — kept rules may reference zones beyond the
    sampled stops. Tighten here if zone-exact fares matter for your test.

    Args:
        fare_rules: The fare_rules table, or ``None``.
        route_ids: Kept route ids.

    Returns:
        A tuple of (filtered fare_rules, set of referenced fare_ids). The
        fare_ids set is ``None`` when the table is absent, signalling that
        fare_attributes should be left untouched.
    """
    if fare_rules is None:
        return None, None
    filtered = _keep_rows(fare_rules, "route_id", route_ids, keep_blank=True)
    fare_ids = _column_values(filtered, "fare_id")
    return filtered, fare_ids


def _filter_pathways(
    pathways: Optional[pd.DataFrame], stop_ids: Set[str]
) -> Optional[pd.DataFrame]:
    """Keep pathway rows whose endpoints are both kept stops.

    Args:
        pathways: The pathways table, or ``None``.
        stop_ids: Kept stop ids.

    Returns:
        The filtered pathways table, or the input unchanged/``None``.
    """
    if pathways is None or pathways.empty:
        return pathways
    mask = pd.Series(True, index=pathways.index)
    for column in ("from_stop_id", "to_stop_id"):
        if column in pathways.columns:
            mask &= pathways[column].isin(stop_ids)
    return pathways[mask].copy()


def _filter_attributions(
    attributions: Optional[pd.DataFrame],
    agency_ids: Set[str],
    route_ids: Set[str],
    trip_ids: Set[str],
) -> Optional[pd.DataFrame]:
    """Keep attribution rows that are global or reference a kept entity.

    Each row attributes at most one of agency/route/trip. A row is kept if all
    of its populated id columns point at kept entities (blank = feed-wide).

    Args:
        attributions: The attributions table, or ``None``.
        agency_ids: Kept agency ids.
        route_ids: Kept route ids.
        trip_ids: Kept trip ids.

    Returns:
        The filtered attributions table, or the input unchanged/``None``.
    """
    if attributions is None or attributions.empty:
        return attributions
    id_columns = (
        ("agency_id", agency_ids),
        ("route_id", route_ids),
        ("trip_id", trip_ids),
    )
    mask = pd.Series(True, index=attributions.index)
    for column, allowed in id_columns:
        if column in attributions.columns:
            mask &= (attributions[column] == "") | attributions[column].isin(allowed)
    return attributions[mask].copy()


def _warn_unknown_tables(feed: Dict[str, pd.DataFrame]) -> None:
    """Log a warning for present tables the cascade copies through verbatim.

    Args:
        feed: The source feed mapping.
    """
    unknown = sorted(set(feed) - set(_KNOWN_TABLES))
    if unknown:
        logging.warning(
            "  copying unrecognized table(s) through unfiltered: %s "
            "(rows may reference dropped ids — review before relying on them)",
            ", ".join(f"{name}.txt" for name in unknown),
        )


# ==================================================================================================
# MANIFEST
# ==================================================================================================


def _write_manifest(
    output_dir: Path,
    job: SampleJob,
    route_ids: Set[str],
    source_counts: Dict[str, int],
    written_counts: Dict[str, int],
) -> None:
    """Write a JSON summary of what was sampled.

    Args:
        output_dir: Folder the abbreviated feed was written to.
        job: The sampling job.
        route_ids: The route_ids that were kept.
        source_counts: Row counts per table in the source feed.
        written_counts: Row counts per table in the abbreviated feed.
    """
    manifest = {
        "source_feed": str(job.input_dir),
        "selected_route_ids": sorted(route_ids),
        "max_trips_per_route": job.max_trips_per_route,
        "tables": {
            name: {"source_rows": source_counts.get(name, 0), "sample_rows": written}
            for name, written in sorted(written_counts.items())
        },
    }
    (output_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))


# ==================================================================================================
# DRIVER
# ==================================================================================================


def run_job(job: SampleJob) -> None:
    """Execute one sampling job end to end.

    Args:
        job: The job to run.
    """
    logging.info("Sampling %s -> %s", job.input_dir, job.output_dir)
    feed = _load_feed(job.input_dir)
    source_counts = {name: len(df) for name, df in feed.items()}

    sampled = sample_feed(feed, job)
    route_ids = set(sampled["routes"]["route_id"])

    written_counts = _write_feed(sampled, job.output_dir)
    _write_manifest(job.output_dir, job, route_ids, source_counts, written_counts)

    logging.info(
        "  wrote %d tables to %s (%d trips, %d stops)",
        len(written_counts),
        job.output_dir,
        written_counts.get("trips", 0),
        written_counts.get("stops", 0),
    )


def _is_placeholder(job: SampleJob) -> bool:
    r"""Return True if a job still points at the placeholder paths.

    Args:
        job: The job to inspect.

    Returns:
        Whether either path still begins with the ``Path\\To`` sentinel.
    """
    return str(job.input_dir).startswith("Path\\To") or str(job.output_dir).startswith("Path\\To")


def main(jobs: Sequence[SampleJob] = JOBS) -> None:
    """Run every configured sampling job, skipping unedited placeholders.

    Args:
        jobs: The jobs to run. Defaults to the module-level ``JOBS``.
    """
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")

    runnable = [job for job in jobs if not _is_placeholder(job)]
    if not runnable:
        logging.warning(
            "All jobs still use placeholder paths. Edit the JOBS list in the "
            "CONFIGURATION block to point at real feed folders, then re-run."
        )
        return

    for job in runnable:
        try:
            run_job(job)
        except (OSError, ValueError) as err:
            logging.error("Skipping job for %s: %s", job.input_dir, err)

    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
