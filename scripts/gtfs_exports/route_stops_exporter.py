"""Export the stops served by a chosen list of routes from a GTFS feed.

Give the script a list of routes — by ``route_short_name`` or ``route_id`` —
and it walks stop_times → trips → routes to list every stop those routes
serve, in two shapes: one row per (route, direction, stop) with an approximate
stop order and the number of trips that call there, and one row per distinct
stop across the selected routes, with the selected routes and any *other*
routes in the feed that also serve it. It answers the everyday data-request
question "which stops does route X serve?" without running a heavier
analysis: ``stop_impacts_target_routes.py`` produces a similar stop set only
as a by-product of a removal-impact analysis, and ``stop_pattern_exporter.py``
exports full per-pattern stop sequences as Excel workbooks.

Stop order within a route and direction is the *median* ``stop_sequence``
across that direction's trips, so routes with short-turns or branches get a
single approximate ordering rather than one row per pattern. Use
``stop_pattern_exporter.py`` when the exact sequence of each pattern matters.

Inputs
------
- A GTFS feed folder or ``.zip`` containing ``routes.txt``, ``trips.txt``,
  ``stop_times.txt``, and ``stops.txt``.
- ``ROUTE_NAMES`` and/or ``ROUTE_NAMES_FILE`` (one name per line, ``#``
  comments allowed): the routes to export. Each name is matched against
  ``route_short_name`` first and ``route_id`` second.
- Optional ``SERVICE_IDS`` to restrict the trips considered to particular
  calendars (e.g. weekday only).

Outputs
-------
- ``route_stops_by_direction.csv``: one row per (route, direction, stop) with
  stop attributes, ``typical_stop_sequence``, and ``n_trips``.
- ``route_stops_unique.csv``: one row per distinct stop served by the selected
  routes, with ``selected_routes`` and ``other_routes`` (routes outside the
  selection that also serve the stop under the same service filter).
- ``route_stops_exporter_runlog.txt``: run-log sidecar capturing the verbatim
  CONFIGURATION block plus a run summary.

Typical usage
-------------
Update the paths and ``ROUTE_NAMES`` in the CONFIGURATION section (or pass
``--gtfs-dir`` / ``--output-dir`` / ``--routes``) and run from a shell,
ArcGIS Pro's Python window, or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_DIR: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me
OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me

# Routes to export, e.g. ["101", "102"]. Each name is matched against
# route_short_name first and route_id second, so either form works.
ROUTE_NAMES: List[str] = []  # ←–– change me

# Optional text file with one route name per line ('#' starts a comment).
# Its names are unioned with ROUTE_NAMES. None or "" to skip.
ROUTE_NAMES_FILE: Optional[str] = None

# Optional service_id filter (values from trips.txt). Leave empty to consider
# every calendar; set e.g. ["WKD"] to list only the stops served on weekdays.
SERVICE_IDS: List[str] = []

# Keep only boarding platforms (stops.txt location_type blank or 0). Set False
# to also keep stations, entrances, and other non-platform locations that
# appear in stop_times.
PLATFORM_STOPS_ONLY: bool = True

# Output filenames (written inside OUTPUT_DIR).
DETAIL_FILENAME: str = r"route_stops_by_direction.csv"
UNIQUE_FILENAME: str = r"route_stops_unique.csv"
RUN_LOG_FILENAME: str = r"route_stops_exporter_runlog.txt"

# When True, a failed run-log write aborts the script so no output is left
# without its configuration record. Set False only for read-only destinations.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

REQUIRED_GTFS_FILES: List[str] = ["routes.txt", "trips.txt", "stop_times.txt", "stops.txt"]

# Optional stops.txt columns carried through to the outputs when present.
OPTIONAL_STOP_COLUMNS: List[str] = ["stop_code", "stop_lat", "stop_lon", "parent_station"]

# =============================================================================
# HELPERS (canonical copies — see utils/)
# =============================================================================


def load_gtfs_data(
    gtfs_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_path: Absolute or relative path to the folder containing the
            GTFS feed, or to a ``.zip`` archive of it — the form GTFS
            producers and most open-data portals distribute feeds in. Zip
            members may sit at the archive root or nested one level inside
            a single wrapper folder; both layouts are handled.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.
        logger: Logger for progress messages. Defaults to this module's
            logger (``logging.getLogger(__name__)``) rather than the root
            logger, so callers keep control of handler configuration.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Path missing, one of *files* not present in the feed, or
            an OS-level failure while reading a file.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, a requested file matches more than one location inside
            the zip, a file is empty, or the CSV parser fails.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    if not os.path.exists(gtfs_path):
        raise OSError(f"The path '{gtfs_path}' does not exist.")

    if files is None:
        files = (
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        )

    is_zip = os.path.isfile(gtfs_path) and gtfs_path.lower().endswith(".zip")
    if not is_zip and not os.path.isdir(gtfs_path):
        raise ValueError(f"'{gtfs_path}' is neither a directory nor a .zip file.")

    archive: zipfile.ZipFile | None = None
    members_by_name: dict[str, list[str]] = {}
    if is_zip:
        try:
            archive = zipfile.ZipFile(gtfs_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"'{gtfs_path}' is not a valid zip archive.") from exc
        for name in archive.namelist():
            members_by_name.setdefault(os.path.basename(name), []).append(name)

    try:
        missing: list[str] = []
        ambiguous: list[str] = []
        resolved: dict[str, str] = {}
        for file_name in files:
            if archive is None:
                if not os.path.exists(os.path.join(gtfs_path, file_name)):
                    missing.append(file_name)
                continue
            candidates = members_by_name.get(file_name, [])
            if not candidates:
                missing.append(file_name)
            elif len(candidates) > 1:
                ambiguous.append(file_name)
            else:
                resolved[file_name] = candidates[0]

        if ambiguous:
            raise ValueError(
                f"Ambiguous GTFS files in '{gtfs_path}' (found in multiple "
                f"locations): {', '.join(ambiguous)}"
            )
        if missing:
            raise OSError(f"Missing GTFS files in '{gtfs_path}': {', '.join(missing)}")

        data: dict[str, pd.DataFrame] = {}
        for file_name in files:
            key = file_name.replace(".txt", "")
            try:
                if archive is None:
                    df = pd.read_csv(
                        os.path.join(gtfs_path, file_name), dtype=dtype, low_memory=False
                    )
                else:
                    with archive.open(resolved[file_name]) as handle:
                        df = pd.read_csv(handle, dtype=dtype, low_memory=False)
                data[key] = df
                log.info("Loaded %s (%d records).", file_name, len(df))

            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"File '{file_name}' in '{gtfs_path}' is empty.") from exc

            except pd.errors.ParserError as exc:
                raise ValueError(f"Parser error in '{file_name}' in '{gtfs_path}': {exc}") from exc

        return data
    finally:
        if archive is not None:
            archive.close()


def load_id_set(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
    *,
    kind: str = "id",
) -> set[str]:
    """Union an inline list and an optional text file of ids into one set.

    Used to resolve override lists (express routes, express origin stops, …) that
    a caller may supply inline, in an external file, or both — without repeating
    the parsing for each one.

    Args:
        inline_ids: Id values supplied directly (e.g. a config list). ``None`` is
            treated as empty.
        txt_path: Path to a text file with one id per line. Blank lines are
            skipped and ``#`` starts a comment (whole-line or inline). ``None``
            skips the file. A path that is set but missing is logged as a warning
            and skipped — the inline ids are still returned.
        kind: Human-readable noun used only in log messages (e.g.
            ``"express route"``, ``"express origin stop"``).

    Returns:
        The unioned set of id strings (possibly empty). Every id is coerced to a
        trimmed ``str`` so it matches GTFS values, which are read as strings.
    """
    ids: set[str] = set()

    for raw in inline_ids or ():
        text = str(raw).strip()
        if text:
            ids.add(text)

    if txt_path:
        if not os.path.exists(txt_path):
            logging.warning(
                "%s file '%s' not found; using inline ids only.", kind.capitalize(), txt_path
            )
        else:
            with open(txt_path, encoding="utf-8") as handle:
                for line in handle:
                    text = line.split("#", 1)[0].strip()
                    if text:
                        ids.add(text)
            logging.info("Loaded %s ids from '%s'.", kind, txt_path)

    logging.info("Resolved %d %s id(s).", len(ids), kind)
    return ids


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


# =============================================================================
# FUNCTIONS
# =============================================================================


def route_label(route_short_name: str, route_id: str) -> str:
    """Return the display label for a route: its short name, else its route_id."""
    short = str(route_short_name).strip()
    return short if short and short.lower() != "nan" else str(route_id).strip()


def identify_target_route_ids(routes: pd.DataFrame, names: set[str]) -> tuple[set[str], set[str]]:
    """Resolve route names to route_id values.

    Each name is compared (trimmed) against ``route_short_name`` first and
    ``route_id`` second, so a name that names a short name shared by several
    ``route_id`` values (a common pattern for per-calendar route ids) selects
    all of them.

    Args:
        routes: The routes.txt table (string columns).
        names: Route names to resolve.

    Returns:
        A tuple ``(matched_route_ids, unmatched_names)``.
    """
    route_ids = routes["route_id"].astype(str).str.strip()
    if "route_short_name" in routes.columns:
        short_names = routes["route_short_name"].fillna("").astype(str).str.strip()
    else:
        short_names = pd.Series("", index=routes.index, dtype=str)

    matched: set[str] = set()
    unmatched: set[str] = set()
    for name in sorted(names):
        hits = route_ids[(short_names == name) | (route_ids == name)]
        if hits.empty:
            unmatched.add(name)
        else:
            matched.update(hits.tolist())
    return matched, unmatched


def filter_platform_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """Keep boarding platforms only (``location_type`` blank or ``0``)."""
    if "location_type" not in stops.columns:
        return stops
    loc = stops["location_type"].fillna("").astype(str).str.strip()
    return stops[(loc == "") | (loc == "0")]


def build_stop_visits(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    service_ids: set[str],
) -> pd.DataFrame:
    """Join stop_times to trips and apply the optional service_id filter.

    Args:
        trips: The trips.txt table.
        stop_times: The stop_times.txt table.
        service_ids: Calendars to keep; an empty set keeps every trip.

    Returns:
        One row per stop-time with ``trip_id``, ``route_id``, ``direction_id``
        (blank when the feed omits it), ``service_id``, ``stop_id`` and a
        numeric ``stop_sequence`` (NaN when unparseable).

    Raises:
        ValueError: If no trips survive the service_id filter.
    """
    trip_cols = ["trip_id", "route_id", "service_id"]
    if "direction_id" in trips.columns:
        trip_cols.append("direction_id")
    trips_slim = trips[trip_cols].copy()
    if "direction_id" not in trips_slim.columns:
        trips_slim["direction_id"] = ""
    trips_slim["direction_id"] = trips_slim["direction_id"].fillna("").astype(str).str.strip()

    if service_ids:
        trips_slim = trips_slim[trips_slim["service_id"].astype(str).isin(service_ids)]
        if trips_slim.empty:
            raise ValueError(
                "No trips match SERVICE_IDS "
                f"{sorted(service_ids)}; check the values against trips.txt."
            )

    visits = stop_times[["trip_id", "stop_id", "stop_sequence"]].merge(
        trips_slim, on="trip_id", how="inner"
    )
    visits["stop_sequence"] = pd.to_numeric(visits["stop_sequence"], errors="coerce")
    return visits


def build_route_stop_detail(
    visits: pd.DataFrame,
    routes: pd.DataFrame,
    stops: pd.DataFrame,
    target_route_ids: set[str],
) -> pd.DataFrame:
    """Build the (route, direction, stop) table for the selected routes.

    Args:
        visits: Output of :func:`build_stop_visits`.
        routes: The routes.txt table.
        stops: The stops.txt table, already filtered to the stops to keep.
        target_route_ids: ``route_id`` values to export.

    Returns:
        One row per (route_id, direction_id, stop_id) with route and stop
        attributes, ``typical_stop_sequence`` (median stop_sequence across the
        direction's trips) and ``n_trips`` (distinct trips calling there),
        sorted by route, direction, and typical stop order.
    """
    target = visits[visits["route_id"].isin(target_route_ids)]
    grouped = (
        target.groupby(["route_id", "direction_id", "stop_id"], sort=False)
        .agg(
            typical_stop_sequence=("stop_sequence", "median"),
            n_trips=("trip_id", "nunique"),
        )
        .reset_index()
    )

    route_cols = ["route_id"] + [
        c for c in ("route_short_name", "route_long_name") if c in routes.columns
    ]
    routes_slim = routes[route_cols].drop_duplicates("route_id").copy()
    for col in ("route_short_name", "route_long_name"):
        if col not in routes_slim.columns:
            routes_slim[col] = ""
    routes_slim["route_short_name"] = routes_slim["route_short_name"].fillna("")
    routes_slim["route_long_name"] = routes_slim["route_long_name"].fillna("")

    stop_cols = ["stop_id", "stop_name"] + [c for c in OPTIONAL_STOP_COLUMNS if c in stops.columns]
    stops_slim = stops[stop_cols].drop_duplicates("stop_id")

    detail = grouped.merge(routes_slim, on="route_id", how="left").merge(
        stops_slim, on="stop_id", how="inner"
    )

    detail["route_label"] = [
        route_label(s, r) for s, r in zip(detail["route_short_name"], detail["route_id"])
    ]
    detail = detail.sort_values(
        by=["route_label", "route_id", "direction_id", "typical_stop_sequence", "stop_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    ordered = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "direction_id",
        "typical_stop_sequence",
        "stop_id",
        "stop_code",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "parent_station",
        "n_trips",
    ]
    cols = [c for c in ordered if c in detail.columns]
    return detail[cols]


def build_unique_stop_table(
    detail: pd.DataFrame,
    visits: pd.DataFrame,
    routes: pd.DataFrame,
    target_route_ids: set[str],
) -> pd.DataFrame:
    """Collapse the detail table to one row per distinct stop.

    Args:
        detail: Output of :func:`build_route_stop_detail`.
        visits: Output of :func:`build_stop_visits` (all routes, same service
            filter), used to find the *other* routes serving each stop.
        routes: The routes.txt table (for route labels).
        target_route_ids: ``route_id`` values that were exported.

    Returns:
        One row per stop with stop attributes, ``n_selected_routes``,
        ``selected_routes`` (comma-joined labels of the selected routes that
        serve it) and ``other_routes`` (labels of every non-selected route
        that also serves it), sorted by stop name and id.
    """
    if "route_short_name" in routes.columns:
        labels = {
            rid: route_label(short, rid)
            for rid, short in zip(routes["route_id"], routes["route_short_name"].fillna(""))
        }
    else:
        labels = {rid: str(rid) for rid in routes["route_id"]}

    def _join_labels(route_ids: pd.Series) -> str:
        return ", ".join(sorted({labels.get(r, str(r)) for r in route_ids}))

    stop_attr_cols = ["stop_id", "stop_name"] + [
        c for c in OPTIONAL_STOP_COLUMNS if c in detail.columns
    ]
    attrs = detail[stop_attr_cols].drop_duplicates("stop_id")

    selected = (
        detail.groupby("stop_id")["route_id"]
        .agg(n_selected_routes="nunique", selected_routes=_join_labels)
        .reset_index()
    )

    others = visits[
        visits["stop_id"].isin(attrs["stop_id"]) & ~visits["route_id"].isin(target_route_ids)
    ]
    other_routes = (
        others.groupby("stop_id")["route_id"].agg(_join_labels).rename("other_routes")
    ).reset_index()

    unique = attrs.merge(selected, on="stop_id", how="left").merge(
        other_routes, on="stop_id", how="left"
    )
    unique["other_routes"] = unique["other_routes"].fillna("")
    unique = unique.sort_values(by=["stop_name", "stop_id"], kind="mergesort").reset_index(
        drop=True
    )
    return unique


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / RUN_LOG_FILENAME
    try:
        config_text = extract_config_block(Path(__file__))
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: List[str] = [
        "=" * 72,
        "ROUTE STOPS EXPORTER RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {Path(__file__).resolve()}",
        "",
        "-" * 72,
        "RUN SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim)",
        "-" * 72,
        "# === BEGIN CONFIG ===",
        config_text,
        "# === END CONFIG ===",
        "",
    ]
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logging.error("Could not write run log '%s': %s", log_path, exc)
        return False
    logging.info("Run log written → %s", log_path)
    return True


# =============================================================================
# ORCHESTRATION
# =============================================================================


def run(
    gtfs_dir: Path | None = None,
    output_dir: Path | None = None,
    route_names: Optional[Sequence[str]] = None,
    route_names_file: Optional[str] = None,
    service_ids: Optional[Sequence[str]] = None,
    platform_stops_only: Optional[bool] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the feed, resolve the routes, and write both stop tables.

    Unset args fall back to the CONFIGURATION block, so ``m.GTFS_DIR = ...;
    m.run()`` works after a plain import.

    Args:
        gtfs_dir: GTFS feed folder or ``.zip``.
        output_dir: Folder for the CSVs and run-log sidecar.
        route_names: Route short names and/or ids to export.
        route_names_file: Optional text file of route names (one per line).
        service_ids: Calendars to keep; empty/None per config = all.
        platform_stops_only: Keep only ``location_type`` 0/blank stops.

    Returns:
        Tuple of (detail table, unique-stop table).

    Raises:
        OSError: If the feed is missing files, or the run log is required but
            cannot be written.
        ValueError: If no route name matches the feed, no trips match the
            service filter, or the selected routes serve no kept stops.
    """
    gtfs_dir = GTFS_DIR if gtfs_dir is None else Path(gtfs_dir)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    route_names = ROUTE_NAMES if route_names is None else route_names
    route_names_file = ROUTE_NAMES_FILE if route_names_file is None else route_names_file
    service_ids = SERVICE_IDS if service_ids is None else service_ids
    platform_stops_only = (
        PLATFORM_STOPS_ONLY if platform_stops_only is None else platform_stops_only
    )

    names = load_id_set(route_names, route_names_file or None, kind="route")
    if not names:
        raise ValueError("No route names given — set ROUTE_NAMES / ROUTE_NAMES_FILE.")
    service_id_set = {str(s).strip() for s in service_ids if str(s).strip()}

    gtfs = load_gtfs_data(str(gtfs_dir), files=REQUIRED_GTFS_FILES)
    routes, trips, stop_times, stops = (
        gtfs["routes"],
        gtfs["trips"],
        gtfs["stop_times"],
        gtfs["stops"],
    )

    target_route_ids, unmatched = identify_target_route_ids(routes, names)
    if unmatched:
        logging.warning(
            "%d route name(s) matched no route_short_name or route_id and were skipped: %s",
            len(unmatched),
            ", ".join(sorted(unmatched)),
        )
    if not target_route_ids:
        raise ValueError(
            f"None of the route names {sorted(names)} matched a route_short_name or "
            "route_id in routes.txt."
        )
    logging.info(
        "Resolved %d name(s) to %d route_id(s): %s",
        len(names) - len(unmatched),
        len(target_route_ids),
        ", ".join(sorted(target_route_ids)),
    )

    if platform_stops_only:
        before = len(stops)
        stops = filter_platform_stops(stops)
        logging.info("Kept %d of %d stops as boarding platforms.", len(stops), before)

    visits = build_stop_visits(trips, stop_times, service_id_set)
    detail = build_route_stop_detail(visits, routes, stops, target_route_ids)
    if detail.empty:
        raise ValueError(
            "The selected routes have no stop_times rows for the kept stops "
            "(check SERVICE_IDS and PLATFORM_STOPS_ONLY)."
        )
    unique = build_unique_stop_table(detail, visits, routes, target_route_ids)

    for (rid, short), grp in detail.groupby(["route_id", "route_short_name"], sort=True):
        logging.info(
            "Route %s (%s): %d direction(s), %d distinct stop(s).",
            route_label(short, rid),
            rid,
            grp["direction_id"].nunique(),
            grp["stop_id"].nunique(),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / DETAIL_FILENAME
    unique_path = output_dir / UNIQUE_FILENAME
    detail.to_csv(detail_path, index=False)
    unique.to_csv(unique_path, index=False)
    logging.info("Wrote %d detail rows → %s", len(detail), detail_path)
    logging.info("Wrote %d unique stops → %s", len(unique), unique_path)

    summary_lines = [
        f"GTFS feed:          {gtfs_dir}",
        f"Route names:        {', '.join(sorted(names))}",
        f"Unmatched names:    {', '.join(sorted(unmatched)) or '-'}",
        f"Resolved route_ids: {', '.join(sorted(target_route_ids))}",
        f"Service filter:     {', '.join(sorted(service_id_set)) or 'all service_ids'}",
        f"Platform stops only: {platform_stops_only}",
        f"Detail rows:        {len(detail)}",
        f"Unique stops:       {len(unique)}",
        f"Detail CSV:         {detail_path}",
        f"Unique-stop CSV:    {unique_path}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise OSError(
            f"Run log could not be written to '{output_dir}' and REQUIRE_RUN_LOG is True."
        )

    logging.info(
        "Route stops export complete — %d unique stop(s) across %d route_id(s).",
        len(unique),
        len(target_route_ids),
    )
    return detail, unique


# =============================================================================
# MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; every flag defaults to its CONFIGURATION constant."""
    parser = argparse.ArgumentParser(
        description=(
            "List the stops served by a chosen set of GTFS routes: one CSV per "
            "(route, direction, stop) and one CSV of distinct stops. Defaults come "
            "from the configuration block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-dir", type=Path, default=GTFS_DIR, help="GTFS feed folder or .zip archive."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for the CSVs and run log."
    )
    parser.add_argument(
        "--routes",
        nargs="*",
        default=ROUTE_NAMES,
        metavar="ROUTE",
        help="Route short names and/or route_ids to export.",
    )
    parser.add_argument(
        "--routes-file",
        default=ROUTE_NAMES_FILE,
        help="Text file of route names, one per line ('#' comments allowed).",
    )
    parser.add_argument(
        "--service-ids",
        nargs="*",
        default=SERVICE_IDS,
        metavar="SERVICE_ID",
        help="Restrict to these service_ids from trips.txt (empty = all).",
    )
    parser.add_argument(
        "--platform-stops-only",
        action=argparse.BooleanOptionalAction,
        default=PLATFORM_STOPS_ONLY,
        help="Keep only stops with location_type blank or 0.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the config block.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sentinels = {Path(r"Path\To\Your\GTFS_Folder"), Path(r"Path\To\Your\Output_Folder")}
    if args.gtfs_dir in sentinels or args.output_dir in sentinels:
        logging.warning(
            "GTFS_DIR and/or OUTPUT_DIR are still placeholders. Update the configuration "
            "block or pass --gtfs-dir/--output-dir before running."
        )
        return 2
    if not args.routes and not args.routes_file:
        logging.warning(
            "No routes selected. Fill in ROUTE_NAMES (or ROUTE_NAMES_FILE) in the "
            "configuration block, or pass --routes/--routes-file."
        )
        return 2
    try:
        run(
            gtfs_dir=args.gtfs_dir,
            output_dir=args.output_dir,
            route_names=args.routes,
            route_names_file=args.routes_file,
            service_ids=args.service_ids,
            platform_stops_only=args.platform_stops_only,
        )
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


# Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
# injected argv away from argparse so the config block stays in charge.
if __name__ == "__main__":
    raise SystemExit(main())
