"""Resolve a hand-kept disruption-events file into route-month exposure tables.

Reads ``disruptions.csv`` (one row per event x affected target; schema in
``sample_data/README.md``), resolves each row's scope onto routes, and expresses
every event as an exposure fraction per month: affected days / days in month.
``stop`` rows roll up to every route serving the stop (GTFS stop_times -> trips
-> routes), ``district`` rows go through a stop -> district crosswalk first, and
``system`` rows stay system-wide. A ``step`` with no end date is 0 before its
start and 1 after; STEP_TRANSITION sets its value in the start month. Exposure
carries no sign: the model's coefficient says whether an event lifts or cuts
ridership. Only dated, identifiable shocks belong in the events file; routine
weather is already a continuous covariate (``clean_noaa_weather.py``).

System-scope events only vary by month, so they go to a ``period``-keyed table
instead of being copied to every route; that table joins every anchor row,
including routes missing from the current GTFS feed. If a model adds full
year-month fixed effects, those effects absorb the system columns.

Inputs
------
- ``EVENTS_CSV``: the hand-maintained events file.
- ``GTFS_PATH``: GTFS feed folder or ``.zip`` (routes, trips, stop_times).
- ``STOP_DISTRICTS_CSV`` (optional; needed only for ``district`` rows): a
  ``stop_id,district`` crosswalk, e.g. from a GIS point-in-polygon join.

Outputs
-------
- ``disruption_exposure_by_route.csv``: ``route_id`` x ``period`` grid,
  zero-filled, one column per route/stop/district event (``ev_<event_id>``).
- ``disruption_exposure_system.csv``: one row per ``period``, one column per
  system event (``sys_ev_<event_id>``).
- ``disruption_exposure_long.csv``: audit table, one row per event x route x
  month with nonzero exposure. Its keys are named ``route_label`` /
  ``year_month`` so ``prep_features_public.py`` does not try to bundle it.
- ``disruption_exposure_builder_runlog.txt``: the CONFIGURATION block, input
  SHA-256s, and every row's resolved routes.

Limitations
-----------
- One GTFS feed maps stops to routes for the whole window. A stop or route
  that changed during the window resolves as it is in that feed.
- Columns that are constant over the window (no overlap with the window, or a
  step that started before it) can't be estimated and are dropped with a warning.

Typical usage
-------------
Set the paths and START_MONTH / END_MONTH in the CONFIGURATION section (or
pass the matching flags), match the months to the NTD anchor's period range,
and run from a shell or a notebook. Feed the outputs to the public orchestrator.

See Also:
    - ``prep_features_public.py``
    - ``monthly_ridership_model.py``
    - ``ntd_anchor_builder.py``
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, List, NamedTuple, Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Hand-maintained events file: one row per (event x affected target). Columns:
# event_id, start_date, end_date, shape, scope, target, category, severity,
# source, notes. Dates are YYYY-MM-DD; a blank end_date means ongoing.
#   shape: "step" (permanent level shift; end_date must be blank) or "pulse".
#   scope: "system" (target ALL), "route" (route label), "stop" (GTFS stop_id),
#          or "district" (a name from STOP_DISTRICTS_CSV).
EVENTS_CSV: str = r"Path\To\Your\disruptions.csv"

# GTFS feed folder or .zip. Route labels come from route_short_name (route_id
# when it is blank), folded the same way as the NTD anchor's route_id.
GTFS_PATH: str = r"Path\To\Your\GTFS"

# Optional stop_id -> district crosswalk (CSV with columns stop_id, district).
# Required only when the events file has district-scope rows.
STOP_DISTRICTS_CSV: str = ""

# Inclusive month range of the output grid, as YYYY-MM. Match the NTD anchor's
# period range: anchor months outside it get no exposure values and drop out
# of the model.
START_MONTH: str = ""
END_MONTH: str = ""

# Exposure of a step event in the month it starts:
#   "fractional"  -> days from start_date to month end / days in month
#   "start_month" -> 1 for the whole start month
#   "next_month"  -> 0 in the start month, 1 from the next month
# Pulses always use exact days.
STEP_TRANSITION: str = "fractional"

# "event"    -> one column per event_id (ev_<event_id>, sys_ev_<event_id>).
# "category" -> one column per category (evcat_<category>, sys_evcat_<category>);
#               overlapping events in a category count each day once.
PIVOT_BY: str = "event"

# When True, a stop or district row counts toward a route in proportion to the
# share of that route's stops it covers; when False, any affected stop puts the
# whole route under exposure. The long table reports stop_share either way.
WEIGHT_BY_STOP_SHARE: bool = False

# Route labels to add to the route grid even though the GTFS feed lacks them
# (e.g. routes discontinued during the window that are still in the anchor).
# Route-scope targets missing from GTFS are added automatically.
EXTRA_ROUTES: Sequence[str] = ()

OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"
ROUTE_FILENAME: str = "disruption_exposure_by_route.csv"
SYSTEM_FILENAME: str = "disruption_exposure_system.csv"
LONG_FILENAME: str = "disruption_exposure_long.csv"
RUN_LOG_FILENAME: str = "disruption_exposure_builder_runlog.txt"

LOG_LEVEL: int = logging.INFO

# When True, a failed run-log write aborts the script so an output is never
# left without its configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

PLACEHOLDER: str = "Path\\To\\Your"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "start_date",
    "end_date",
    "shape",
    "scope",
    "target",
    "category",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("severity", "source", "notes")
SHAPES: frozenset[str] = frozenset({"step", "pulse"})
SCOPES: frozenset[str] = frozenset({"system", "route", "stop", "district"})
LOCAL_STOP_SCOPES: frozenset[str] = frozenset({"stop", "district"})
STEP_TRANSITIONS: tuple[str, ...] = ("fractional", "start_month", "next_month")
PIVOT_OPTIONS: tuple[str, ...] = ("event", "category")
SYSTEM_TARGET: str = "ALL"
SYSTEM_ROUTE_LABEL: str = "ALL"

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ExposureTables(NamedTuple):
    """Everything one run produces, before it is written to disk.

    Attributes:
        by_route: ``route_id`` x ``period`` grid of local-scope exposure columns.
        system: ``period`` rows of system-scope exposure columns.
        long: Event x route x month audit rows with nonzero exposure.
        resolution: One human-readable line per events-file row, naming the
            routes it resolved to (written to the run log).
        dropped_columns: Exposure columns dropped for being constant.
        warnings: Resolution warnings (unknown routes / stops / districts).
    """

    by_route: pd.DataFrame
    system: pd.DataFrame
    long: pd.DataFrame
    resolution: list[str]
    dropped_columns: list[str]
    warnings: list[str]


# =============================================================================
# COPIED HELPERS (canonical versions named in each docstring; keep in sync)
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


def _canonical_key(series: pd.Series) -> pd.Series:
    """Normalize a join-key column so the anchor and a bundle match reliably.

    The anchor and bundle are produced on different machines from different
    files, so a key can arrive as ``101`` on one side and ``"101"`` (or
    ``"101.0"`` from a float round-trip) on the other. This collapses every
    key to a trimmed, upper-cased, space-free string and strips a single trailing
    ``.0`` — the same folding as ntd_anchor_builder's normalise_route, so an
    anchor keyed ``RT5`` matches a bundle keyed ``"Rt 5"``. The same helper
    is copied into monthly_ridership_model.py and MUST stay byte-identical between the two.
    """
    out = series.astype("string").str.strip().str.upper()
    out = out.str.replace(" ", "", regex=False)
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.fillna("")


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# EVENTS FILE
# =============================================================================


def load_events(path: Path) -> pd.DataFrame:
    """Read the events CSV with every value as text (blank cells stay "").

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def column_suffix(value: str) -> str:
    """Turn an event_id or category into a lowercase column-name suffix."""
    return re.sub(r"[^0-9a-z]+", "_", value.lower()).strip("_")


def validate_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the events table and return it with parsed, normalized columns.

    Every problem is collected and reported together, so one pass through the
    file fixes them all.

    Returns:
        A copy with ``start_date`` / ``end_date`` as Timestamps (NaT for a blank
        end), ``shape`` / ``scope`` lowercased, ``severity`` as float, and a
        ``row_number`` column giving each row's line in the CSV.

    Raises:
        ValueError: Listing every row-level problem found.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Events file is missing required column(s): {missing}")

    df = raw.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS):
        df[col] = df[col].astype(str).str.strip()
    df["shape"] = df["shape"].str.lower()
    df["scope"] = df["scope"].str.lower()
    df["row_number"] = np.arange(len(df)) + 2  # line 1 is the header

    problems: list[str] = []

    def flag(mask: pd.Series, message: str) -> None:
        for line in df.loc[mask, "row_number"]:
            problems.append(f"line {line}: {message}")

    for col in ("event_id", "start_date", "shape", "scope", "target", "category"):
        flag(df[col] == "", f"blank {col}")

    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y-%m-%d", errors="coerce")
    end_text = df["end_date"]
    df["end_date"] = pd.to_datetime(end_text, format="%Y-%m-%d", errors="coerce")
    flag(
        df["start_date"].isna() & (raw["start_date"].astype(str).str.strip() != ""),
        "start_date is not a YYYY-MM-DD date",
    )
    flag(df["end_date"].isna() & (end_text != ""), "end_date is not a YYYY-MM-DD date")
    flag(df["end_date"] < df["start_date"], "end_date is before start_date")

    flag(~df["shape"].isin(SHAPES) & (df["shape"] != ""), "shape must be 'step' or 'pulse'")
    flag(
        ~df["scope"].isin(SCOPES) & (df["scope"] != ""),
        "scope must be 'system', 'route', 'stop', or 'district'",
    )
    flag(
        (df["shape"] == "step") & df["end_date"].notna(),
        "a step is permanent, so its end_date must be blank (use a pulse for a bounded event)",
    )
    is_system = df["scope"] == "system"
    target_is_all = df["target"].str.upper() == SYSTEM_TARGET
    flag(is_system & ~target_is_all, f"system scope needs target {SYSTEM_TARGET}")
    flag(
        ~is_system & target_is_all & (df["scope"] != ""),
        f"target {SYSTEM_TARGET} is only valid with system scope",
    )

    severity = pd.to_numeric(df["severity"], errors="coerce")
    flag(severity.isna() & (df["severity"] != ""), "severity must be a number or blank")
    df["severity"] = severity

    dup = df.duplicated(subset=["event_id", "scope", "target"], keep="first")
    flag(dup & (df["event_id"] != ""), "duplicate (event_id, scope, target) row")

    for event_id, rows in df[df["event_id"] != ""].groupby("event_id", sort=False):
        if len(set(rows["shape"])) > 1:
            problems.append(f"event {event_id}: rows disagree on shape")
        if len(set(rows["category"])) > 1:
            problems.append(f"event {event_id}: rows disagree on category")
        scopes = set(rows["scope"])
        if "system" in scopes and len(scopes) > 1:
            problems.append(
                f"event {event_id}: mixes system scope with {sorted(scopes - {'system'})}; "
                "a system row already covers every route"
            )

    for col in ("event_id", "category"):
        suffixes = df.loc[df[col] != "", [col]].drop_duplicates()
        suffixes["suffix"] = suffixes[col].map(column_suffix)
        flag_empty = suffixes["suffix"] == ""
        for value in suffixes.loc[flag_empty, col]:
            problems.append(f"{col} '{value}' has no letters or digits to name a column with")
        clashes = suffixes[~flag_empty].groupby("suffix")[col].unique()
        for suffix, values in clashes.items():
            if len(values) > 1:
                problems.append(
                    f"{col} values {sorted(values)} all become column suffix '{suffix}'; rename one"
                )

    if problems:
        raise ValueError("Events file failed validation:\n  - " + "\n  - ".join(problems))
    return df


# =============================================================================
# GTFS / CROSSWALK RESOLUTION
# =============================================================================


def route_label_map(routes: pd.DataFrame) -> pd.DataFrame:
    """Map GTFS route_id to the label the NTD anchor uses (route_short_name).

    A blank route_short_name falls back to the route_id. Labels are folded with
    ``_canonical_key`` so they match the anchor.

    Returns:
        Columns ``route_id`` (GTFS) and ``route_label``.
    """
    if "route_id" not in routes.columns:
        raise ValueError("routes.txt has no route_id column.")
    short = (
        routes["route_short_name"].fillna("").astype(str).str.strip()
        if "route_short_name" in routes.columns
        else pd.Series("", index=routes.index)
    )
    n_fallback = int((short == "").sum())
    if n_fallback:
        logging.warning(
            "%d route(s) have no route_short_name; using their route_id as the label.",
            n_fallback,
        )
    label = short.where(short != "", routes["route_id"].astype(str))
    out = pd.DataFrame({"route_id": routes["route_id"].astype(str), "route_label": label})
    out["route_label"] = _canonical_key(out["route_label"])
    return out


def stop_route_crosswalk(
    stop_times: pd.DataFrame, trips: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Return the distinct (stop_id, route_label) pairs served in the feed."""
    pairs = stop_times[["stop_id", "trip_id"]].drop_duplicates()
    pairs = pairs.merge(trips[["trip_id", "route_id"]], on="trip_id", how="inner")
    pairs = pairs.merge(labels, on="route_id", how="inner")
    out = pairs[["stop_id", "route_label"]].drop_duplicates()
    out["stop_id"] = out["stop_id"].astype(str).str.strip()
    return out.reset_index(drop=True)


def load_stop_districts(path: Path) -> pd.DataFrame:
    """Read a ``stop_id,district`` crosswalk, adding a case-folded match key.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If either column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Stop-district crosswalk not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("stop_id", "district") if c not in df.columns]
    if missing:
        raise ValueError(f"Stop-district crosswalk '{path}' is missing column(s): {missing}")
    df["stop_id"] = df["stop_id"].str.strip()
    df["district_key"] = df["district"].str.strip().str.upper()
    return df


def resolve_targets(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    stop_routes: pd.DataFrame | None,
    stop_districts: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Resolve every events row onto routes.

    Args:
        events: Output of :func:`validate_events`.
        labels: Output of :func:`route_label_map`.
        stop_routes: Output of :func:`stop_route_crosswalk`; required when any
            row has stop or district scope.
        stop_districts: Output of :func:`load_stop_districts`; required when any
            row has district scope.

    Returns:
        ``(resolved, resolution_lines, warnings)``. ``resolved`` has one row per
        (events row, route) with columns ``row_number``, ``event_id``,
        ``route_label`` (``ALL`` for system rows), ``stops_hit`` and
        ``stop_share`` (1.0 for system and route rows).

    Raises:
        ValueError: If stop/district rows exist but the needed crosswalk is missing.
    """
    scopes = set(events["scope"])
    if scopes & LOCAL_STOP_SCOPES and stop_routes is None:
        raise ValueError("Stop or district rows need GTFS stop_times.txt and trips.txt.")
    if "district" in scopes and stop_districts is None:
        raise ValueError(
            "The events file has district-scope rows, but STOP_DISTRICTS_CSV is not set. "
            "Point it at a stop_id,district crosswalk (e.g. a GIS point-in-polygon join "
            "of stops.txt to the district layer)."
        )

    gtfs_labels = set(labels["route_label"])
    stops_per_route = (
        stop_routes.groupby("route_label")["stop_id"].nunique()
        if stop_routes is not None
        else pd.Series(dtype=float)
    )

    records: list[dict[str, Any]] = []
    lines: list[str] = []
    warnings: list[str] = []

    for row in events.itertuples(index=False):
        head = f"line {row.row_number}: {row.event_id} | {row.scope} | {row.target}"
        if row.scope == "system":
            records.append(
                {
                    "row_number": row.row_number,
                    "event_id": row.event_id,
                    "route_label": SYSTEM_ROUTE_LABEL,
                    "stops_hit": np.nan,
                    "stop_share": 1.0,
                }
            )
            lines.append(f"{head} -> ALL (system table)")
            continue

        if row.scope == "route":
            label = str(_canonical_key(pd.Series([row.target])).iloc[0])
            if label not in gtfs_labels:
                msg = (
                    f"{head}: route {label} is not in the GTFS feed; kept in the grid "
                    "(it may be a route the anchor has and the feed no longer does)"
                )
                warnings.append(msg)
                logging.warning(msg)
            records.append(
                {
                    "row_number": row.row_number,
                    "event_id": row.event_id,
                    "route_label": label,
                    "stops_hit": np.nan,
                    "stop_share": 1.0,
                }
            )
            lines.append(f"{head} -> route {label}")
            continue

        assert stop_routes is not None  # checked above
        if row.scope == "stop":
            stop_ids = {row.target}
        else:
            assert stop_districts is not None  # checked above
            hits = stop_districts["district_key"] == row.target.upper()
            stop_ids = set(stop_districts.loc[hits, "stop_id"])
            if not stop_ids:
                msg = f"{head}: district not found in STOP_DISTRICTS_CSV; resolves to no routes"
                warnings.append(msg)
                logging.warning(msg)
                lines.append(f"{head} -> (no match)")
                continue

        matched = stop_routes[stop_routes["stop_id"].isin(stop_ids)]
        if matched.empty:
            what = "this stop" if row.scope == "stop" else "these stops"
            msg = f"{head}: no GTFS route serves {what}; resolves to no routes"
            warnings.append(msg)
            logging.warning(msg)
            lines.append(f"{head} -> (no match)")
            continue

        per_route = matched.groupby("route_label")["stop_id"].nunique()
        parts: list[str] = []
        for label, n_hit in per_route.items():
            share = float(n_hit) / float(stops_per_route[label])
            records.append(
                {
                    "row_number": row.row_number,
                    "event_id": row.event_id,
                    "route_label": str(label),
                    "stops_hit": int(n_hit),
                    "stop_share": share,
                }
            )
            parts.append(f"{label} ({n_hit}/{int(stops_per_route[label])} stops)")
        prefix = f"{len(stop_ids)} stop(s); " if row.scope == "district" else ""
        lines.append(f"{head} -> {prefix}routes {', '.join(parts)}")

    resolved = pd.DataFrame.from_records(
        records, columns=["row_number", "event_id", "route_label", "stops_hit", "stop_share"]
    )
    return resolved, lines, warnings


# =============================================================================
# EXPOSURE
# =============================================================================


def parse_month(value: str, name: str) -> pd.Period:
    """Parse a ``YYYY-MM`` setting into a monthly Period.

    Raises:
        ValueError: If ``value`` is not a ``YYYY-MM`` month.
    """
    if not re.fullmatch(r"\d{4}-\d{2}", str(value).strip()):
        raise ValueError(f"{name} must be a YYYY-MM month, got '{value}'.")
    return pd.Period(str(value).strip(), freq="M")


def active_span(
    start: pd.Timestamp,
    end: pd.Timestamp,
    shape: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    step_transition: str,
) -> tuple[int, int] | None:
    """Return the first and last affected day as offsets into the window.

    A missing ``end`` (NaT) means the event runs through the window's end. A
    step's start moves per ``step_transition``. Returns None when the event
    doesn't overlap the window.
    """
    if shape == "step":
        if step_transition == "start_month":
            start = start.to_period("M").to_timestamp()
        elif step_transition == "next_month":
            start = (start.to_period("M") + 1).to_timestamp()
    last = window_end if pd.isna(end) else min(end, window_end)
    first = max(start, window_start)
    if first > last:
        return None
    return (first - window_start).days, (last - window_start).days


def monthly_exposure(
    resolved: pd.DataFrame,
    events: pd.DataFrame,
    group_col: str,
    months: pd.PeriodIndex,
    step_transition: str,
    weight_by_stop_share: bool,
) -> pd.DataFrame:
    """Compute exposure per (group, route, month) from resolved rows.

    Each resolved row adds its weight (1, or its stop share when weighting) to
    every day it is active; per day the total is capped at 1, so overlapping
    rows of one group count each day once. Exposure is the month's summed
    daily weight over its number of days.

    Args:
        resolved: Output of :func:`resolve_targets`.
        events: Output of :func:`validate_events`.
        group_col: ``"event_id"`` or ``"category"``: what one column measures.
        months: The output grid's months.
        step_transition: See STEP_TRANSITION.
        weight_by_stop_share: See WEIGHT_BY_STOP_SHARE.

    Returns:
        Columns ``group``, ``route_label``, ``period``, ``affected_days``,
        ``days_in_month``, ``exposure_frac``; only months with exposure.
    """
    columns = ["group", "route_label", "period", "affected_days", "days_in_month", "exposure_frac"]
    if resolved.empty:
        return pd.DataFrame(columns=columns)

    window_start = months[0].to_timestamp()
    window_end = months[-1].to_timestamp(how="end").normalize()
    days = pd.date_range(window_start, window_end, freq="D")
    month_idx = (days.year - window_start.year) * 12 + (days.month - window_start.month)
    days_in_month = np.bincount(month_idx, minlength=len(months)).astype(float)
    period_labels = months.strftime("%Y-%m")

    attrs = events.set_index("row_number")[["start_date", "end_date", "shape", "scope", "category"]]
    rows = resolved.join(attrs, on="row_number")

    out: list[pd.DataFrame] = []
    for (group, label), chunk in rows.groupby([group_col, "route_label"], sort=True):
        weight = np.zeros(len(days))
        for r in chunk.itertuples(index=False):
            span = active_span(
                r.start_date, r.end_date, r.shape, window_start, window_end, step_transition
            )
            if span is None:
                continue
            w = r.stop_share if (weight_by_stop_share and r.scope in LOCAL_STOP_SCOPES) else 1.0
            weight[span[0] : span[1] + 1] += w
        weight = np.minimum(weight, 1.0)
        affected = np.bincount(month_idx, weights=(weight > 0).astype(float), minlength=len(months))
        summed = np.bincount(month_idx, weights=weight, minlength=len(months))
        keep = affected > 0
        if not keep.any():
            continue
        out.append(
            pd.DataFrame(
                {
                    "group": group,
                    "route_label": label,
                    "period": period_labels[keep],
                    "affected_days": affected[keep].astype(int),
                    "days_in_month": days_in_month[keep].astype(int),
                    "exposure_frac": summed[keep] / days_in_month[keep],
                }
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


def _drop_constant(df: pd.DataFrame, key_cols: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop value columns with a single value across every row."""
    value_cols = [c for c in df.columns if c not in key_cols]
    constant = [c for c in value_cols if len(df) == 0 or bool((df[c] == df[c].iloc[0]).all())]
    return df.drop(columns=constant), constant


def build_exposure_tables(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    stop_routes: pd.DataFrame | None,
    stop_districts: pd.DataFrame | None,
    *,
    start_month: str,
    end_month: str,
    step_transition: str = STEP_TRANSITION,
    pivot_by: str = PIVOT_BY,
    weight_by_stop_share: bool = WEIGHT_BY_STOP_SHARE,
    extra_routes: Sequence[str] = EXTRA_ROUTES,
) -> ExposureTables:
    """Resolve validated events into the wide and long exposure tables.

    Args:
        events: Output of :func:`validate_events`.
        labels: Output of :func:`route_label_map` (defines the route grid).
        stop_routes: Output of :func:`stop_route_crosswalk`, or None when no
            row has stop or district scope.
        stop_districts: Output of :func:`load_stop_districts`, or None.
        start_month: First grid month, ``YYYY-MM``.
        end_month: Last grid month, ``YYYY-MM``.
        step_transition: See STEP_TRANSITION.
        pivot_by: See PIVOT_BY.
        weight_by_stop_share: See WEIGHT_BY_STOP_SHARE.
        extra_routes: See EXTRA_ROUTES.

    Returns:
        An :class:`ExposureTables`.

    Raises:
        ValueError: On a bad month range or option value, or a missing crosswalk.
    """
    if step_transition not in STEP_TRANSITIONS:
        raise ValueError(f"STEP_TRANSITION must be one of {STEP_TRANSITIONS}.")
    if pivot_by not in PIVOT_OPTIONS:
        raise ValueError(f"PIVOT_BY must be one of {PIVOT_OPTIONS}.")
    first = parse_month(start_month, "START_MONTH")
    last = parse_month(end_month, "END_MONTH")
    if last.ordinal < first.ordinal:
        raise ValueError(f"END_MONTH ({end_month}) is before START_MONTH ({start_month}).")
    months = pd.period_range(first, last, freq="M")
    period_labels = list(months.strftime("%Y-%m"))

    resolved, resolution, warnings = resolve_targets(events, labels, stop_routes, stop_districts)
    # --- Long audit table (always per event) ----------------------------------
    per_event = monthly_exposure(
        resolved, events, "event_id", months, step_transition, weight_by_stop_share
    )
    event_attrs = events.groupby("event_id", sort=False).agg(
        category=("category", "first"), shape=("shape", "first")
    )
    pair_attrs = resolved.join(
        events.set_index("row_number")[["scope", "severity"]], on="row_number"
    )
    pair_attrs = pair_attrs.groupby(["event_id", "route_label"]).agg(
        scope=("scope", lambda s: "+".join(sorted(set(s)))),
        severity=("severity", "max"),
        stop_share=("stop_share", lambda s: min(1.0, float(s.sum()))),
    )
    long = per_event.rename(
        columns={"group": "event_id", "route_label": "route_label", "period": "year_month"}
    )
    long = long.join(event_attrs, on="event_id").join(pair_attrs, on=["event_id", "route_label"])
    long = long[
        [
            "event_id",
            "category",
            "scope",
            "shape",
            "severity",
            "route_label",
            "year_month",
            "affected_days",
            "days_in_month",
            "stop_share",
            "exposure_frac",
        ]
    ].sort_values(["event_id", "route_label", "year_month"], ignore_index=True)

    # --- Wide tables ----------------------------------------------------------
    if pivot_by == "event":
        wide_src = per_event
        local_prefix, system_prefix = "ev_", "sys_ev_"
    else:
        wide_src = monthly_exposure(
            resolved, events, "category", months, step_transition, weight_by_stop_share
        )
        local_prefix, system_prefix = "evcat_", "sys_evcat_"
    wide_src = wide_src.assign(column=wide_src["group"].astype(str).map(column_suffix))

    is_system = wide_src["route_label"] == SYSTEM_ROUTE_LABEL
    system_src = wide_src[is_system]
    system = (
        system_src.pivot_table(
            index="period", columns="column", values="exposure_frac", aggfunc="sum"
        )
        .reindex(period_labels)
        .fillna(0.0)
        .add_prefix(system_prefix)
        .rename_axis(index="period", columns=None)
        .reset_index()
    )

    grid_routes = set(labels["route_label"])
    grid_routes |= set(resolved.loc[resolved["route_label"] != SYSTEM_ROUTE_LABEL, "route_label"])
    if extra_routes:
        grid_routes |= set(_canonical_key(pd.Series(list(extra_routes), dtype="string")))
    grid_routes.discard("")
    grid = pd.MultiIndex.from_product(
        [sorted(grid_routes), period_labels], names=["route_id", "period"]
    )
    local_src = wide_src[~is_system].rename(columns={"route_label": "route_id"})
    by_route = (
        local_src.pivot_table(
            index=["route_id", "period"], columns="column", values="exposure_frac", aggfunc="sum"
        )
        .reindex(grid)
        .fillna(0.0)
        .add_prefix(local_prefix)
        .rename_axis(columns=None)
        .reset_index()
    )

    # Events that resolved to nothing never reach the pivot; name them.
    produced = set(wide_src["group"])
    expected = set(events["event_id"] if pivot_by == "event" else events["category"])
    for name in sorted(expected - produced):
        msg = f"'{name}' has no exposure in {start_month}..{end_month}; no column written"
        warnings.append(msg)
        logging.warning(msg)

    system, dropped_sys = _drop_constant(system, ["period"])
    by_route, dropped_local = _drop_constant(by_route, ["route_id", "period"])
    dropped = dropped_sys + dropped_local
    for col in dropped:
        logging.warning(
            "Dropped '%s': it is constant over %s..%s (e.g. a step that began before the "
            "window), so it is collinear with the intercept.",
            col,
            start_month,
            end_month,
        )

    return ExposureTables(
        by_route=by_route,
        system=system,
        long=long,
        resolution=resolution,
        dropped_columns=dropped,
        warnings=warnings,
    )


# =============================================================================
# RUN LOG
# =============================================================================


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


def write_run_log(
    output_dir: Path,
    tables: ExposureTables,
    inputs: Mapping[str, Path | None],
    settings: Mapping[str, object],
    written: Sequence[Path],
    source_path: Path,
) -> bool:
    """Write the run log: inputs + hashes, settings, resolved targets, config.

    Returns:
        True if the log was written, False otherwise.
    """
    log_path = output_dir / RUN_LOG_FILENAME
    route_cols = [c for c in tables.by_route.columns if c not in ("route_id", "period")]
    try:
        config_text = extract_config_block(source_path)
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    input_lines: list[str] = []
    for name, path in inputs.items():
        if path is None:
            input_lines.append(f"  {name}: (not used)")
        elif path.is_file():
            input_lines.append(f"  {name}: {path}  sha256={_sha256_file(path)}")
        else:
            input_lines.append(f"  {name}: {path}")

    lines = [
        "=" * 72,
        "DISRUPTION EXPOSURE BUILDER RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_path.resolve() if source_path.exists() else source_path}",
        "",
        "-" * 72,
        "INPUTS",
        "-" * 72,
        *input_lines,
        "",
        "-" * 72,
        "EFFECTIVE SETTINGS",
        "-" * 72,
        *[f"  {k}: {v}" for k, v in settings.items()],
        "",
        "-" * 72,
        "RESOLVED TARGETS (one line per events-file row)",
        "-" * 72,
        *[f"  {line}" for line in tables.resolution],
        "",
        "-" * 72,
        "OUTPUTS",
        "-" * 72,
        *[f"  {p.name}" for p in written],
        f"  route columns:  {route_cols}",
        f"  system columns: {[c for c in tables.system.columns if c != 'period']}",
        f"  dropped as constant: {tables.dropped_columns or 'none'}",
        "",
        "-" * 72,
        "WARNINGS",
        "-" * 72,
        *([f"  {w}" for w in tables.warnings] or ["  (none)"]),
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]
    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to '%s'.", log_path)
    return True


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
    """Create the command-line argument parser (defaults = CONFIGURATION)."""
    p = argparse.ArgumentParser(
        description="Resolve a disruption-events CSV into route-month exposure tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--events-csv", default=EVENTS_CSV, help="Disruption events CSV.")
    p.add_argument("--gtfs-path", default=GTFS_PATH, help="GTFS feed folder or .zip.")
    p.add_argument(
        "--stop-districts-csv",
        default=STOP_DISTRICTS_CSV,
        help="stop_id,district crosswalk (needed only for district-scope rows).",
    )
    p.add_argument("--start-month", default=START_MONTH, help="First grid month, YYYY-MM.")
    p.add_argument("--end-month", default=END_MONTH, help="Last grid month, YYYY-MM.")
    p.add_argument(
        "--step-transition",
        default=STEP_TRANSITION,
        choices=STEP_TRANSITIONS,
        help="Exposure of a step event in its start month.",
    )
    p.add_argument(
        "--pivot-by", default=PIVOT_BY, choices=PIVOT_OPTIONS, help="One column per what."
    )
    p.add_argument(
        "--weight-by-stop-share",
        action=argparse.BooleanOptionalAction,
        default=WEIGHT_BY_STOP_SHARE,
        help="Scale stop/district exposure by the share of the route's stops affected.",
    )
    p.add_argument(
        "--extra-routes",
        nargs="*",
        default=list(EXTRA_ROUTES),
        help="Route labels to add to the grid even if the GTFS feed lacks them.",
    )
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--log-level", default=logging.getLevelName(LOG_LEVEL), help="DEBUG / INFO / WARNING."
    )
    return p


def _source_path() -> Path:
    """Path to this file for run-log extraction, with a notebook-safe fallback."""
    return Path(__file__) if "__file__" in globals() else Path("disruption_exposure_builder.py")


def main(argv: Sequence[str] | None = None) -> int:
    """Build the exposure tables and write them with a run log.

    Returns:
        Process exit code: 0 on success, 1 on a runtime failure, 2 if required
        CONFIGURATION values are still placeholders or invalid.
    """
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if any(PLACEHOLDER in str(v) for v in (args.events_csv, args.gtfs_path, args.output_dir)):
        logging.warning(
            "EVENTS_CSV / GTFS_PATH / OUTPUT_DIR are still placeholders. Update the "
            "CONFIGURATION section or pass --events-csv / --gtfs-path / --output-dir."
        )
        return 2
    if not args.start_month or not args.end_month:
        logging.warning(
            "Set START_MONTH and END_MONTH (YYYY-MM) to the NTD anchor's period range, "
            "or pass --start-month / --end-month."
        )
        return 2
    try:
        parse_month(args.start_month, "START_MONTH")
        parse_month(args.end_month, "END_MONTH")
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    events_path = Path(args.events_csv).expanduser()
    gtfs_path = Path(args.gtfs_path).expanduser()
    districts_path = Path(args.stop_districts_csv).expanduser() if args.stop_districts_csv else None
    output_dir = Path(args.output_dir).expanduser()

    try:
        events = validate_events(load_events(events_path))
        logging.info(
            "Loaded %d events-file row(s) for %d event(s).",
            len(events),
            events["event_id"].nunique(),
        )
        needs_stops = bool(set(events["scope"]) & LOCAL_STOP_SCOPES)
        files = ["routes.txt", "trips.txt", "stop_times.txt"] if needs_stops else ["routes.txt"]
        gtfs = load_gtfs_data(str(gtfs_path), files=files)
        labels = route_label_map(gtfs["routes"])
        stop_routes = (
            stop_route_crosswalk(gtfs["stop_times"], gtfs["trips"], labels) if needs_stops else None
        )
        stop_districts = (
            load_stop_districts(districts_path)
            if districts_path is not None and "district" in set(events["scope"])
            else None
        )
        tables = build_exposure_tables(
            events,
            labels,
            stop_routes,
            stop_districts,
            start_month=args.start_month,
            end_month=args.end_month,
            step_transition=args.step_transition,
            pivot_by=args.pivot_by,
            weight_by_stop_share=args.weight_by_stop_share,
            extra_routes=args.extra_routes,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for frame, filename, keys, kind in (
        (tables.by_route, ROUTE_FILENAME, ("route_id", "period"), "route/stop/district"),
        (tables.system, SYSTEM_FILENAME, ("period",), "system"),
    ):
        path = output_dir / filename
        if len(frame.columns) > len(keys):
            frame.to_csv(path, index=False, encoding="utf-8")
            written.append(path)
            logging.info(
                "Wrote '%s': %d rows, %d exposure column(s).",
                path,
                len(frame),
                len(frame.columns) - len(keys),
            )
        else:
            logging.info("No %s exposure columns; '%s' not written.", kind, filename)
            if path.exists():
                logging.warning(
                    "'%s' from an earlier run is still in %s; delete it so the "
                    "orchestrator doesn't bundle stale columns.",
                    filename,
                    output_dir,
                )
    long_path = output_dir / LONG_FILENAME
    tables.long.to_csv(long_path, index=False, encoding="utf-8")
    written.append(long_path)
    logging.info("Wrote '%s': %d event x route x month row(s).", long_path, len(tables.long))

    settings = {
        "window": f"{args.start_month}..{args.end_month}",
        "step_transition": args.step_transition,
        "pivot_by": args.pivot_by,
        "weight_by_stop_share": args.weight_by_stop_share,
        "extra_routes": list(args.extra_routes),
    }
    inputs = {
        "events_csv": events_path,
        "gtfs_path": gtfs_path,
        "stop_districts_csv": districts_path,
    }
    if (
        not write_run_log(output_dir, tables, inputs, settings, written, _source_path())
        and REQUIRE_RUN_LOG
    ):
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
        return 1

    logging.info("All processing complete. Disruption exposure tables written to '%s'.", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
