"""Compare public-service and concept GTFS stops without conflating feed scope.

Python 3.9+; requires pandas, numpy, openpyxl. No ArcPy or SciPy required.
Edit the CONFIGURATION block and run in an IDE/notebook, or use CLI options.
Paths may be GTFS ZIP archives or extracted folders. Relative paths are resolved
against this script's folder (the working directory when pasted in a notebook).

DEFAULT: SCOPE_MODE='after_routes' for a partial Remix/concept export.
PRIMARY_STOP_KEY='stop_code' compares public stop codes across feeds. Choose
'stop_id' explicitly for native-ID comparisons. The selected field must be
populated and unique in both full inventories; missing/duplicate keys stop the
run. GTFS joins still use native stop_id. Codes and IDs remain strings, preserving
leading zeros. Same-code stops with changed stop_id values remain matched; those
secondary ID changes are flagged separately and do not count as attribute changes.
- Match concept routes to current routes by route label, not generated route ID.
- Compare stops referenced by stop_times on those routes, across ALL feed trips.
- Stops outside these routes are excluded, not reported as deletions.
- Use BEFORE_ROUTE_NAMES to include an affected route absent from the concept.
- Use AFTER_TO_BEFORE_ROUTE_NAMES when a concept route has been renamed.
- Unmatched concept routes prompt for current counterparts or an explicit 'none'.
- Use --route-pair AFTER BEFORE (repeatable), or --no-route-match AFTER.
- A concept route can map to several current routes; a current route can map to
  several concept routes. Scope counts use distinct stops, not summed pair counts.
- 'none' means no current counterpart; the concept route remains in after scope.
- --unmatched-routes error disables prompts; warn explicitly allows provisional
  scope counts. Headless prompt mode stops and requests explicit choices.
- route_pairings.json saves resolved pairs and no-match decisions for reuse with
  --route-pairs-file. CLI choices override file/config for each named route.
- Use 'all_stops' for an intentional inventory comparison of two complete feeds.

Outputs in a new timestamped run folder:
  stops_comparison.xlsx, summary.json, run_settings.json, gtfs_stop_diff.log,
  plus a CSV for each workbook sheet. Primary sheets: summary, route_scope,
  modified, removed_from_scope, new_stop_codes, existing_stops_new_to_scope,
  needs_review, attribute_summary. ADDITIONS_OUTPUT_MODE controls addition tables:
  separate (default), combined, new_only, or existing_only. Summary totals always
  retain both categories. "New" means the selected primary key is absent from
  the full current inventory; candidate ID matches still require review before claiming new poles.
  Existing stops new to route scope are separate from route gains/losses at stops
  already shared by both scopes (see route_changes). With PRIMARY_STOP_KEY='stop_id',
  the new-key output is named new_stop_ids instead. stop_id_code_changes exposes
  native IDs associated with different public codes across the feeds.
  'removed_from_scope' means no stop_times reference in the selected after scope;
  it does NOT establish a physical stop closure or loss of all transit service.
  Unchanged means no compared attribute or above-threshold coordinate change.
  Route-context changes are reported separately and do not drive classification.
  reused_stop_comparison checks existing stops newly used by the concept routes.
  Coordinates changed is not proof that a bus pole physically moved.

Primary-key matches are exact. Candidate ID changes use stop_code and nearby coordinates;
these are review suggestions only and never silently revise the counts. Blank
GTFS defaults are normalized for location_type and wheelchair_boarding. Other
attribute losses in a concept export are retained and explained, not suppressed.
No service-date or pickup/drop-off filtering is performed. In route scope,
parent stations and other unreferenced locations remain in the full inventory.

Examples:
    python gtfs_stop_diff.py --before current.zip --after concept.zip --out output
    python gtfs_stop_diff.py --scope all_stops --before a.zip --after b.zip
    python gtfs_stop_diff.py --before-routes 301 305 310
    python gtfs_stop_diff.py --additions new_only
    python gtfs_stop_diff.py --primary-stop-key stop_id
    python gtfs_stop_diff.py --route-pair "353New" "353" --no-route-match "400X"
    python gtfs_stop_diff.py --route-pairs-file "previous_run/route_pairings.json"
Notebook:
    summary = run_compare()  # or main(["--before", ..., "--after", ...])
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# CONFIGURATION -- EDIT HERE
# =============================================================================
# === BEGIN CONFIG ===
BEFORE_GTFS_DIR = Path(r"Path\To\Current_GTFS\current_gtfs.zip")
AFTER_GTFS_DIR = Path(r"Path\To\Concept_GTFS\concept_gtfs.zip")
OUTPUT_DIR = Path(r"Path\To\Output_Folder")

# 'after_routes': partial concept vs full current feed. 'all_stops': inventories.
SCOPE_MODE = "after_routes"
# Cross-feed comparison key: 'stop_code' (default) or 'stop_id'.
# Must be populated and unique in BOTH full stop inventories; never mixed/fallback.
# GTFS stop_times/route joins always retain the native stop_id relationship.
PRIMARY_STOP_KEY = "stop_code"
# Optional EXTRA current route names, in addition to automatic concept matches.
# Include eliminated routes here; an omitted route cannot be inferred as removed.
BEFORE_ROUTE_NAMES: list[str] = []
# Examples: {"Renamed": "101"} or {"Merged": ["101", "202"]}. Verify each pair.
AFTER_TO_BEFORE_ROUTE_NAMES: dict[str, str | list[str]] = {}
# Explicitly confirm that a concept route has no current counterpart.
# This retains its stops in the concept scope; it does NOT exclude the route.
AFTER_ROUTES_WITH_NO_MATCH: list[str] = []
# 'prompt' asks in a terminal/notebook; headless runs require explicit decisions.
# 'error' never prompts. 'warn' permits unresolved routes with provisional counts.
UNMATCHED_ROUTE_POLICY = "prompt"
# Optional previous run's route_pairings.json (or supply --route-pairs-file).
ROUTE_PAIRS_FILE: Path | None = None

# Controls addition CSVs/workbook sheets, not comparison scope or summary counts.
# separate: new_stop_codes (or new_stop_ids) + existing_stops_new_to_scope.
# combined: one added_to_scope table with an addition_category column.
# new_only / existing_only: export only the corresponding addition-detail table.
ADDITIONS_OUTPUT_MODE = "separate"

RELOCATE_THRESHOLD_FEET = 25.0
NEAREST_MATCHES_MAX_FEET = 500.0
NEAREST_CANDIDATES_PER_STOP = 3
ENABLE_NEAREST_MATCHES = True
ENABLE_ROUTE_CONTEXT = True  # Required in after_routes mode; optional in all_stops.
# Attributes explicitly ignored here still appear in the full inventories.
IGNORE_ATTRIBUTES: list[str] = []
LOG_LEVEL = logging.INFO

# === END CONFIG ===

# =============================================================================
# Models and shared helpers
# =============================================================================
LOGGER = logging.getLogger("gtfs_stop_diff")
VERSION = "2.3.0"
ATTRIBUTES = (
    "stop_name",
    "stop_code",
    "stop_desc",
    "zone_id",
    "location_type",
    "parent_station",
    "stop_timezone",
    "wheelchair_boarding",
    "platform_code",
    "stop_url",
    "tts_stop_name",
    "level_id",
    "stop_access",
)
DEFAULTS = {"location_type": "0", "wheelchair_boarding": "0"}


@dataclass
class Feed:
    """Stop inventory with optional validated route references."""

    stops: pd.DataFrame
    routes: pd.DataFrame | None
    pairs: pd.DataFrame | None
    source: Path


@dataclass
class Scope:
    """Scope uses native GTFS stop_id within each feed; matching uses the selected key."""

    before_ids: set[str]
    after_ids: set[str]
    before_route_ids: set[str]
    after_route_ids: set[str]
    route_table: pd.DataFrame
    warnings: list[str]
    resolved_route_pairs: dict[str, list[str]]
    confirmed_no_match_routes: list[str]
    unresolved_routes: list[str]


def normalize_text(series: pd.Series) -> pd.Series:
    """Preserve literal IDs such as NA, NULL, and leading zeros."""
    return series.fillna("").astype(str).str.strip()


def text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a normalized string column, or blanks for an absent attribute."""
    if column in frame:
        return normalize_text(frame[column])
    return pd.Series("", index=frame.index, dtype=object)


def normalized_attribute(frame: pd.DataFrame, column: str) -> pd.Series:
    """Apply only explicit GTFS equivalent defaults; do not lowercase names."""
    values = text_column(frame, column)
    return values.replace("", DEFAULTS[column]) if column in DEFAULTS else values


def route_sort_key(label: str) -> tuple[int, int, str]:
    """Sort numeric route names numerically, then other names alphabetically."""
    return (0, int(label), label) if label.isdigit() else (1, 0, label.casefold())


def join_labels(values: Sequence[str] | set[str]) -> str:
    """Return a readable, deterministic route list."""
    return "; ".join(sorted(set(values), key=route_sort_key))


def resolve_path(value: Path) -> Path:
    """Keep relative paths independent of the shell's working directory."""
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    return value.expanduser() if value.is_absolute() else base / value.expanduser()


def setup_logging(output_dir: Path) -> None:
    """Configure only this script's logger; leave notebook logging alone."""
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()
    LOGGER.setLevel(LOG_LEVEL)
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "gtfs_stop_diff.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


# =============================================================================
# Read and validate -- ZIP and folder inputs use the same parsing path
# =============================================================================
def read_gtfs_table(source: Path, filename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read strings faithfully, including from one unambiguous nested ZIP path."""
    if not source.exists():
        raise FileNotFoundError(f"GTFS source does not exist: {source}")

    requested = set(usecols or [])

    def read(handle: Any) -> pd.DataFrame:
        return pd.read_csv(
            handle,
            dtype="str",
            keep_default_na=False,
            encoding="utf-8-sig",
            usecols=(lambda column: column in requested) if usecols is not None else None,
            low_memory=False,
        )

    try:
        if source.is_dir():
            frame = read(source / filename)
        elif source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                matches = [
                    name
                    for name in archive.namelist()
                    if not name.endswith("/")
                    and Path(name).name == filename
                    and "__MACOSX" not in Path(name).parts
                ]
                if not matches:
                    raise FileNotFoundError(f"{source}: missing {filename}")
                if len(matches) != 1:
                    raise ValueError(f"{source}: ambiguous {filename}: {matches}")
                with archive.open(matches[0]) as handle:
                    frame = read(handle)
        else:
            raise ValueError(f"Expected a GTFS folder or ZIP: {source}")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not parse {filename} in {source}: {exc}") from exc
    if usecols is not None:
        require_columns(frame, requested, f"{source.name}/{filename}")
    LOGGER.info("Loaded %s (%d records) from %s", filename, len(frame), source.name)
    return frame


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Fail before producing results when required schema is absent."""
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def validate_key(frame: pd.DataFrame, key: str, label: str) -> None:
    """Reject blank or duplicate identifiers; never discard conflicting records."""
    require_columns(frame, {key}, label)
    frame[key] = normalize_text(frame[key])
    if frame[key].eq("").any():
        raise ValueError(f"{label}: {key} contains blank IDs.")
    duplicates = frame.loc[frame[key].duplicated(keep=False), key].unique().tolist()
    if duplicates:
        raise ValueError(f"{label}: duplicate {key} values (sample): {duplicates[:20]}")


PRIMARY_STOP_KEYS = ("stop_id", "stop_code")


def comparison_inventory(
    stops: pd.DataFrame,
    primary_stop_key: str,
    label: str,
) -> pd.DataFrame:
    """Validate the selected cross-feed identifier without changing GTFS keys.

    The selected field must be populated and unique across the full inventory.
    Do not substitute stop_id for missing codes: this mixes two ID namespaces.
    GTFS stop_times and route joins continue using the original stop_id column.
    """
    if primary_stop_key not in PRIMARY_STOP_KEYS:
        raise ValueError(f"primary stop key must be one of {PRIMARY_STOP_KEYS}.")
    require_columns(stops, {primary_stop_key}, f"{label} primary key")
    out = stops.copy()
    values = normalize_text(out[primary_stop_key])
    blank = values.eq("")
    if blank.any():
        sample = text_column(out.loc[blank], "stop_id").head(10).tolist()
        raise ValueError(
            f"{label}: primary key {primary_stop_key!r} is blank for {int(blank.sum())} "
            f"records (GTFS stop_id sample: {sample}). Populate the selected field or "
            "choose PRIMARY_STOP_KEY='stop_id'; no fallback was applied."
        )
    duplicates = values[values.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(
            f"{label}: primary key {primary_stop_key!r} must be unique; duplicate "
            f"values include {duplicates[:10]}. Correct the keys or select stop_id."
        )
    out[primary_stop_key] = values
    out["comparison_key"] = values
    out["primary_stop_key"] = primary_stop_key
    return out


def stop_id_code_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Expose reused native IDs with different public codes; do not infer identity."""
    columns = [
        "stop_id",
        "stop_code_before",
        "stop_code_after",
        "stop_name_before",
        "stop_name_after",
        "stop_lat_before",
        "stop_lon_before",
        "stop_lat_after",
        "stop_lon_after",
        "in_comparison_scope_before",
        "in_comparison_scope_after",
        "review_note",
    ]
    if "stop_code" not in before or "stop_code" not in after:
        return pd.DataFrame(columns=pd.Index(columns))
    b = before.set_index("stop_id")
    a = after.set_index("stop_id")
    rows: list[dict[str, Any]] = []
    for stop_id in sorted(set(b.index) & set(a.index)):
        br, ar = b.loc[stop_id], a.loc[stop_id]
        bc, ac = (
            str(br.get("stop_code", "") or "").strip(),
            str(ar.get("stop_code", "") or "").strip(),
        )
        if bc == ac:
            continue
        row: dict[str, Any] = {"stop_id": stop_id}
        for side, record in (("before", br), ("after", ar)):
            for column in (
                "stop_code",
                "stop_name",
                "stop_lat",
                "stop_lon",
                "in_comparison_scope",
            ):
                row[f"{column}_{side}"] = record.get(column, "")
        row["review_note"] = (
            "Same GTFS stop_id has a different public code: code edit or reused ID. "
            "Cross-feed matching follows PRIMARY_STOP_KEY; review names and coordinates."
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=pd.Index(columns))


def load_stops(source: Path, label: str) -> pd.DataFrame:
    """Load the full inventory and expose coordinate-quality problems."""
    stops = read_gtfs_table(source, "stops.txt")
    require_columns(stops, {"stop_id", "stop_lat", "stop_lon"}, f"{label} stops")
    validate_key(stops, "stop_id", f"{label} stops")
    for column in stops.columns:
        stops[column] = normalize_text(stops[column])
    lat_raw = stops["stop_lat"].copy()
    lon_raw = stops["stop_lon"].copy()
    lat = pd.to_numeric(lat_raw, errors="coerce").astype(float)
    lon = pd.to_numeric(lon_raw, errors="coerce").astype(float)
    valid = np.isfinite(lat) & np.isfinite(lon) & lat.between(-90, 90) & lon.between(-180, 180)
    missing = lat_raw.eq("") | lon_raw.eq("")
    optional_missing = (
        normalized_attribute(stops, "location_type").isin(["3", "4"])
        & lat_raw.eq("")
        & lon_raw.eq("")
    )
    stops["raw_stop_lat"] = lat_raw
    stops["raw_stop_lon"] = lon_raw
    stops["stop_lat"] = lat
    stops["stop_lon"] = lon
    stops["coordinate_status"] = np.select(
        [valid, optional_missing, missing],
        ["valid", "not_provided_optional", "missing_required_or_partial"],
        default="invalid_or_out_of_range",
    )
    problems = int((~valid & ~optional_missing).sum())
    if problems:
        LOGGER.warning("%s: %d stops have missing/invalid coordinates.", label, problems)
    if stops.empty:
        raise ValueError(f"{label}: stops.txt contains no stops.")
    return stops


def load_route_context(source: Path, stops: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate IDs and build unique stop/route references across all trips."""
    routes = read_gtfs_table(source, "routes.txt")
    trips = read_gtfs_table(source, "trips.txt")
    stop_times = read_gtfs_table(source, "stop_times.txt", ["trip_id", "stop_id"])
    validate_key(routes, "route_id", "routes.txt")
    validate_key(trips, "trip_id", "trips.txt")
    require_columns(trips, {"route_id"}, "trips.txt")
    for frame, columns in ((trips, ["route_id"]), (stop_times, ["trip_id", "stop_id"])):
        for column in columns:
            frame[column] = normalize_text(frame[column])
            if frame[column].eq("").any():
                raise ValueError(f"Route context: blank {column}; fixed-stop references required.")
    for values, allowed, description in (
        (
            trips["route_id"],
            set(routes["route_id"]),
            "trips reference unknown route_id",
        ),
        (
            stop_times["trip_id"],
            set(trips["trip_id"]),
            "stop_times reference unknown trip_id",
        ),
        (
            stop_times["stop_id"],
            set(stops["stop_id"]),
            "stop_times reference unknown stop_id",
        ),
    ):
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"{description}: {unknown[:20]}")
    labels = text_column(routes, "route_short_name")
    labels = labels.mask(labels.eq(""), text_column(routes, "route_long_name"))
    routes["route_label"] = labels.mask(labels.eq(""), routes["route_id"])
    pairs = (
        stop_times.drop_duplicates()
        .merge(
            trips[["trip_id", "route_id"]],
            on="trip_id",
            how="left",
            validate="many_to_one",
        )[["stop_id", "route_id"]]
        .drop_duplicates()
    )
    pairs = pairs.merge(
        routes[["route_id", "route_label"]],
        on="route_id",
        how="left",
        validate="many_to_one",
    )
    LOGGER.info("Built %d stop/route pairs for %s", len(pairs), source.name)
    return routes, pairs


def load_feed(source: Path, label: str, context: bool, required: bool) -> Feed:
    """Require route context for scoped comparisons; warn if optional context fails."""
    stops = load_stops(source, label)
    routes, pairs = None, None
    if context or required:
        try:
            routes, pairs = load_route_context(source, stops)
        except (OSError, ValueError) as exc:
            if required:
                raise ValueError(f"{label}: route scope cannot be established: {exc}") from exc
            LOGGER.warning("%s: route context unavailable: %s", label, exc)
    return Feed(stops, routes, pairs, source)


# =============================================================================
# Scope -- make partial concept exports comparable to the public feed
# =============================================================================
def route_lookup(routes: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Resolve case-insensitive route labels; reject ambiguity across agencies."""
    result: dict[str, tuple[str, str]] = {}
    for route_id, label in routes[["route_id", "route_label"]].itertuples(index=False, name=None):
        key = str(label).strip().casefold()
        if key in result:
            raise ValueError(f"Ambiguous route label {label!r}; route names must be unique.")
        result[key] = (str(route_id), str(label))
    return result


def normalize_route_pairs(
    mapping: Mapping[str, str | Sequence[str]],
) -> dict[str, list[str]]:
    """Normalize pair keys; accept a single before name or a list for mergers."""
    result: dict[str, list[str]] = {}
    for after_name, targets in mapping.items():
        if not isinstance(after_name, str) or not after_name.strip():
            raise ValueError("Concept route names in pairings must be nonblank strings.")
        key = after_name.strip().casefold()
        if key in result:
            raise ValueError(f"Repeated concept route in pairing configuration: {after_name!r}")
        if isinstance(targets, str):
            target_list = [targets]
        elif isinstance(targets, (list, tuple)):
            target_list = list(targets)
        else:
            raise TypeError(f"Before routes for {after_name!r} must be a name or list of names.")
        if not target_list or any(not isinstance(t, str) or not t.strip() for t in target_list):
            raise ValueError(
                f"Empty pairing for {after_name!r}; "
                "use AFTER_ROUTES_WITH_NO_MATCH or --no-route-match."
            )
        result[key] = list(dict.fromkeys(t.strip().casefold() for t in target_list))
    return result


def normalize_route_names(names: Sequence[str], label: str) -> set[str]:
    """Validate explicit route decisions rather than treating a string as a list."""
    if isinstance(names, str) or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError(f"{label} must be a list of nonblank route names.")
    return {name.strip().casefold() for name in names}


def prompt_route_pair(
    after_label: str,
    before_lookup: Mapping[str, tuple[str, str]],
) -> list[str] | None:
    """Ask for current counterparts; None is a deliberate no-counterpart choice."""
    available = join_labels({label for _, label in before_lookup.values()})
    print(f"\nConcept route {after_label!r} has no exact current-route name match.")  # noqa: T201
    print(f"Available current routes: {available}")  # noqa: T201
    while True:
        try:
            answer = input(
                "Enter current route name(s), separated by commas; 'none' = no current "
                "counterpart; '?' = list; 'cancel' = stop: "
            ).strip()
        except EOFError as exc:
            raise ValueError(
                f"No input available for {after_label!r}. Use --route-pair AFTER BEFORE "
                "or --no-route-match AFTER, or configure the choices at the top."
            ) from exc
        if answer.casefold() == "cancel":
            raise ValueError("Route pairing cancelled; no comparison results were written.")
        if answer.casefold() == "none":
            return None
        if answer == "?":
            print(f"Available current routes: {available}")  # noqa: T201
            continue
        if not answer:
            print("Enter a route name or 'none'; blank input leaves the decision unresolved.")  # noqa: T201
            continue
        keys = list(dict.fromkeys(part.strip().casefold() for part in answer.split(",")))
        unknown = [key for key in keys if key not in before_lookup]
        if unknown:
            print(f"Unknown current route name(s): {unknown}. Try again.")  # noqa: T201
            continue
        return keys


def make_scope(
    before: Feed,
    after: Feed,
    mode: str,
    before_names: Sequence[str],
    route_name_map: Mapping[str, str | Sequence[str]],
    no_match_routes: Sequence[str] = (),
    unmatched_policy: str = "error",
) -> Scope:
    """Resolve counterparts before deriving scope; 'none' is distinct from unknown.

    Explicit pairs override automatic same-name matches. One before route can map
    to multiple concept routes, and one concept route can map to multiple before
    routes. Scope membership is a set union, so these relationships never double
    count stops. Confirmed unmatched concept routes remain in the after scope.
    """
    if mode not in {"after_routes", "all_stops"}:
        raise ValueError("scope must be 'after_routes' or 'all_stops'.")
    if unmatched_policy not in {"prompt", "error", "warn"}:
        raise ValueError("unmatched route policy must be prompt, error, or warn.")
    if mode == "all_stops":
        if before_names or route_name_map or no_match_routes:
            raise ValueError("Route selections/mappings require scope='after_routes'.")
        return Scope(
            set(before.stops["stop_id"]),
            set(after.stops["stop_id"]),
            set(before.routes["route_id"]) if before.routes is not None else set(),
            set(after.routes["route_id"]) if after.routes is not None else set(),
            pd.DataFrame(columns=pd.Index(["before_route", "after_route", "status"])),
            [],
            {},
            [],
            [],
        )
    if before.routes is None or after.routes is None or before.pairs is None or after.pairs is None:
        raise ValueError("Route context is required for after_routes scope.")
    before_lookup = route_lookup(before.routes)
    after_lookup = route_lookup(after.routes)
    mapping = normalize_route_pairs(route_name_map)
    no_match = normalize_route_names(no_match_routes, "AFTER_ROUTES_WITH_NO_MATCH")
    conflict = set(mapping) & no_match
    if conflict:
        raise ValueError(
            f"Routes cannot be both paired and confirmed unmatched: {sorted(conflict)}"
        )
    for key in set(mapping) | no_match:
        if key not in after_lookup:
            raise ValueError(f"Unknown concept route in pairing decisions: {key!r}")
    for key, targets in mapping.items():
        unknown = [target for target in targets if target not in before_lookup]
        if unknown:
            raise ValueError(
                f"Unknown before route(s) paired with {after_lookup[key][1]!r}: {unknown}"
            )
    explicit = normalize_route_names(before_names, "BEFORE_ROUTE_NAMES")
    for key in explicit:
        if key not in before_lookup:
            raise ValueError(f"Unknown before route name: {key!r}")

    unresolved = sorted(
        set(after_lookup) - set(mapping) - no_match - set(before_lookup),
        key=lambda key: route_sort_key(after_lookup[key][1]),
    )
    if unresolved and unmatched_policy == "prompt":
        interactive = "ipykernel" in sys.modules or bool(sys.stdin and sys.stdin.isatty())
        if not interactive:
            labels = [after_lookup[key][1] for key in unresolved]
            raise ValueError(
                f"Unresolved concept routes: {labels}. Interactive input is unavailable. "
                "Supply --route-pair AFTER BEFORE and/or --no-route-match AFTER, "
                "load --route-pairs-file, or configure the choices at the top. "
                "Use --unmatched-routes warn only for an explicitly provisional comparison."
            )
        for key in unresolved:
            targets = prompt_route_pair(after_lookup[key][1], before_lookup)
            if targets is None:
                no_match.add(key)
            else:
                mapping[key] = targets
        unresolved = []
    if unresolved and unmatched_policy == "error":
        raise ValueError(
            f"Unresolved concept routes: {[after_lookup[key][1] for key in unresolved]}. "
            "Use --route-pair AFTER BEFORE or --no-route-match AFTER; "
            "or use --unmatched-routes prompt for interactive choices."
        )

    selected = {before_lookup[key][0] for key in explicit}
    matched: set[str] = set()
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    resolved_map: dict[str, list[str]] = {}
    for after_key, (after_id, after_label) in after_lookup.items():
        if after_key in no_match:
            targets, status = [], "confirmed_no_before_match"
        elif after_key in mapping:
            targets, status = mapping[after_key], "explicit_pair"
        elif after_key in before_lookup:
            targets, status = [after_key], "same_name"
        else:
            targets, status = [], "unresolved_concept_route_review"
            warnings.append(
                f"Unresolved concept route {after_label!r}; scope counts are provisional."
            )
        if targets:
            resolved_map[after_label] = [before_lookup[key][1] for key in targets]
        for target in targets or [None]:
            before_id, before_label = before_lookup[target] if target is not None else ("", "")
            if before_id:
                selected.add(before_id)
                matched.add(before_id)
            rows.append(
                {
                    "before_route_id": before_id,
                    "before_route": before_label,
                    "after_route_id": after_id,
                    "after_route": after_label,
                    "status": status,
                }
            )
    for before_id, before_label in before_lookup.values():
        if before_id not in matched:
            rows.append(
                {
                    "before_route_id": before_id,
                    "before_route": before_label,
                    "after_route_id": "",
                    "after_route": "",
                    "status": "explicit_before_only" if before_id in selected else "outside_scope",
                }
            )
    if not selected and unresolved:
        raise ValueError(
            "No current routes resolved; an entirely unresolved baseline cannot be compared."
        )
    after_routes = set(after.routes["route_id"])
    before_ids = set(before.pairs.loc[before.pairs["route_id"].isin(selected), "stop_id"])
    after_ids = set(after.pairs.loc[after.pairs["route_id"].isin(after_routes), "stop_id"])
    if (selected and not before_ids) or not after_ids:
        raise ValueError("Selected route scope contains no referenced stops; check stop_times.")
    if not selected:
        warnings.append(
            "All concept routes were explicitly marked as having no current counterpart. "
            "The before route scope is empty; existing inventory stops are checked separately."
        )
    warnings.append(
        "Scope uses all trips, without service-date filtering. Current routes absent from the "
        "concept are outside scope unless explicitly included in BEFORE_ROUTE_NAMES."
    )
    return Scope(
        before_ids,
        after_ids,
        selected,
        after_routes,
        pd.DataFrame(rows),
        warnings,
        resolved_map,
        sorted((after_lookup[key][1] for key in no_match), key=route_sort_key),
        [after_lookup[key][1] for key in unresolved],
    )


def annotate_stops(feed: Feed, stop_ids: set[str], route_ids: set[str]) -> pd.DataFrame:
    """Retain all stops and show which current routes fall outside the concept."""
    out = feed.stops.copy()
    out["in_comparison_scope"] = out["stop_id"].isin(stop_ids)
    out["route_context_available"] = feed.pairs is not None
    for column in ("routes", "scope_routes", "other_routes"):
        out[column] = ""
    out["route_count"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if feed.pairs is not None:
        pairs = feed.pairs
        for column, subset in (
            ("routes", pairs),
            ("scope_routes", pairs.loc[pairs["route_id"].isin(route_ids)]),
            ("other_routes", pairs.loc[~pairs["route_id"].isin(route_ids)]),
        ):
            grouped = subset.groupby("stop_id")["route_label"].agg(lambda x: join_labels(list(x)))
            out[column] = out["stop_id"].map(grouped).fillna("")
        counts = pairs.groupby("stop_id")["route_id"].nunique()
        out["route_count"] = out["stop_id"].map(counts).fillna(0).astype("Int64")
    return out.sort_values("stop_id").reset_index(drop=True)


# =============================================================================
# Compare shared IDs, diagnose attributes, and retain uncertain coordinate cases
# =============================================================================
def haversine_feet(
    lat1: NDArray[np.float64],
    lon1: NDArray[np.float64],
    lat2: NDArray[np.float64],
    lon2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute great-circle distances, including scalar-array broadcasting."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6_371_000.0 * 3.280839895013123 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compare_shared(
    before: pd.DataFrame,
    after: pd.DataFrame,
    threshold: float,
    ignored: Sequence[str],
    primary_stop_key: str = "stop_code",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return shared-stop results, field counts, field details, and schema changes."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Relocation threshold must be finite and nonnegative.")
    unknown = set(ignored) - set(ATTRIBUTES)
    if unknown:
        raise ValueError(f"Unknown IGNORE_ATTRIBUTES entries: {sorted(unknown)}")
    b = comparison_inventory(before, primary_stop_key, "before").set_index(
        "comparison_key", drop=False
    )
    a = comparison_inventory(after, primary_stop_key, "after").set_index(
        "comparison_key", drop=False
    )
    ids = sorted(set(b.index) & set(a.index))
    b, a = b.loc[ids], a.loc[ids]
    result = pd.DataFrame({primary_stop_key: ids, "comparison_key": ids})
    result["primary_stop_key"] = primary_stop_key
    secondary_key = "stop_id" if primary_stop_key == "stop_code" else "stop_code"
    result["secondary_id_changed"] = (
        text_column(b, secondary_key).ne(text_column(a, secondary_key)).to_numpy(dtype=bool)
    )
    valid = (
        b["coordinate_status"].eq("valid").to_numpy()
        & a["coordinate_status"].eq("valid").to_numpy()
    )
    distance = np.full(len(ids), np.nan)
    if valid.any():
        distance[valid] = haversine_feet(
            b["stop_lat"].to_numpy(dtype=float)[valid],
            b["stop_lon"].to_numpy(dtype=float)[valid],
            a["stop_lat"].to_numpy(dtype=float)[valid],
            a["stop_lon"].to_numpy(dtype=float)[valid],
        )
    optional_both = (
        b["coordinate_status"].eq("not_provided_optional").to_numpy()
        & a["coordinate_status"].eq("not_provided_optional").to_numpy()
    )
    review = ~valid & ~optional_both
    result["distance_ft"] = distance
    result["coordinate_changed_over_threshold"] = distance > threshold
    result["location_assessment"] = np.where(
        valid, "assessed", np.where(optional_both, "not_applicable", "unable_to_assess")
    )
    changed: list[list[str]] = [[] for _ in ids]
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    schema: list[dict[str, Any]] = []
    for column in ATTRIBUTES:
        present_b, present_a = column in before.columns, column in after.columns
        if not present_b and not present_a:
            continue
        rb, ra = text_column(b, column), text_column(a, column)
        nb, na = normalized_attribute(b, column), normalized_attribute(a, column)
        mask = nb.ne(na)
        raw_mask = rb.ne(ra)
        schema.append(
            {
                "field": column,
                "present_before": present_b,
                "present_after": present_a,
                "schema_change": "same"
                if present_b == present_a
                else ("added_column" if present_a else "removed_column"),
                "included_in_classification": column not in ignored,
            }
        )
        if column in ignored:
            continue
        positions = np.flatnonzero(mask.to_numpy())
        for pos in positions:
            changed[pos].append(column)
            details.append(
                {
                    primary_stop_key: ids[pos],
                    "comparison_key": ids[pos],
                    "primary_stop_key": primary_stop_key,
                    "field": column,
                    "value_before": rb.iloc[pos],
                    "value_after": ra.iloc[pos],
                    "normalized_before": nb.iloc[pos],
                    "normalized_after": na.iloc[pos],
                    "column_present_before": present_b,
                    "column_present_after": present_a,
                }
            )
        examples = [f"{rb.iloc[p]!r} -> {ra.iloc[p]!r}" for p in positions[:3]]
        summaries.append(
            {
                "field": column,
                "changed_count": int(mask.sum()),
                "shared_stop_count": len(ids),
                "changed_pct": round(100 * int(mask.sum()) / len(ids), 2) if ids else 0.0,
                "populated_to_blank": int((mask & rb.ne("") & ra.eq("")).sum()),
                "blank_to_populated": int((mask & rb.eq("") & ra.ne("")).sum()),
                "equivalent_raw_changes": int((raw_mask & ~mask).sum()),
                "column_present_before": present_b,
                "column_present_after": present_a,
                "examples": " | ".join(examples),
            }
        )
    result["changed_fields"] = ["; ".join(fields) for fields in changed]
    result["attributes_changed"] = pd.Series([bool(fields) for fields in changed], dtype=bool)
    result["needs_review"] = review
    modified = result["coordinate_changed_over_threshold"] | result["attributes_changed"]
    result["change_type"] = np.select(
        [
            modified & review,
            review,
            result["coordinate_changed_over_threshold"] & result["attributes_changed"],
            result["coordinate_changed_over_threshold"],
            result["attributes_changed"],
        ],
        [
            "modified_needs_review",
            "needs_review",
            "coordinates+attributes",
            "coordinates",
            "attributes",
        ],
        default="unchanged",
    )
    # Explicit suffixes retain attributes that exist on only one side.
    for label, frame in (("before", b), ("after", a)):
        payload = (
            frame.drop(
                columns=[primary_stop_key, "comparison_key", "primary_stop_key"],
                errors="ignore",
            )
            .reset_index(drop=True)
            .add_suffix(f"_{label}")
        )
        result = pd.concat([result, payload], axis=1)
    summary_columns = [
        "field",
        "changed_count",
        "shared_stop_count",
        "changed_pct",
        "populated_to_blank",
        "blank_to_populated",
        "equivalent_raw_changes",
        "column_present_before",
        "column_present_after",
        "examples",
    ]
    summary_frame = pd.DataFrame(summaries, columns=pd.Index(summary_columns)).sort_values(
        ["changed_count", "field"], ascending=[False, True]
    )
    details_frame = pd.DataFrame(
        details,
        columns=pd.Index(
            [
                primary_stop_key,
                "comparison_key",
                "primary_stop_key",
                "field",
                "value_before",
                "value_after",
                "normalized_before",
                "normalized_after",
                "column_present_before",
                "column_present_after",
            ]
        ),
    )
    schema_frame = pd.DataFrame(
        schema,
        columns=pd.Index(
            [
                "field",
                "present_before",
                "present_after",
                "schema_change",
                "included_in_classification",
            ]
        ),
    )
    return result, summary_frame, details_frame, schema_frame


# =============================================================================
# Candidate ID changes -- proposals for review, never automatic equivalence
# =============================================================================
def candidate_matches(
    before: pd.DataFrame,
    after: pd.DataFrame,
    max_feet: float,
    top_n: int,
    primary_stop_key: str = "stop_code",
) -> pd.DataFrame:
    """Match unmatched inventories using spherical distance and nonblank stop_code.

    At most top_n nearby candidates plus any exact stop-code candidates per after
    stop. Same-code candidates can be distant or lack coordinates. No matches are
    made across different location types. Many-to-one possibilities are exposed.
    """
    if primary_stop_key not in PRIMARY_STOP_KEYS:
        raise ValueError(f"primary stop key must be one of {PRIMARY_STOP_KEYS}.")
    columns = [
        "primary_stop_key",
        "after_comparison_key",
        "before_comparison_key",
        "after_stop_id",
        "before_stop_id",
        "after_stop_name",
        "before_stop_name",
        "after_stop_code",
        "before_stop_code",
        "distance_ft",
        "same_stop_code",
        "within_distance",
        "after_routes",
        "before_routes",
        "rank",
        "review_note",
        "after_candidate_count",
        "before_used_by_after_count",
    ]
    if not math.isfinite(max_feet) or max_feet < 0 or top_n < 1:
        raise ValueError(
            "Candidate distance must be finite/nonnegative and candidate count positive."
        )
    b = before.reset_index(drop=True)
    a = after.reset_index(drop=True)
    bkeys = text_column(b, primary_stop_key).to_numpy()
    akeys = text_column(a, primary_stop_key).to_numpy()
    rows: list[dict[str, Any]] = []
    btypes = normalized_attribute(b, "location_type").to_numpy()
    atypes = normalized_attribute(a, "location_type").to_numpy()
    bcodes = text_column(b, "stop_code").to_numpy()
    acodes = text_column(a, "stop_code").to_numpy()
    bnames = text_column(b, "stop_name").to_numpy()
    anames = text_column(a, "stop_name").to_numpy()
    for ai in range(len(a)):
        same_type = btypes == atypes[ai]
        valid = b["coordinate_status"].eq("valid").to_numpy() & same_type
        distances = np.full(len(b), np.nan)
        if a.iloc[ai]["coordinate_status"] == "valid" and valid.any():
            distances[valid] = haversine_feet(
                b.loc[valid, "stop_lat"].to_numpy(dtype=float),
                b.loc[valid, "stop_lon"].to_numpy(dtype=float),
                np.asarray(a.iloc[ai]["stop_lat"], dtype=float),
                np.asarray(a.iloc[ai]["stop_lon"], dtype=float),
            )
        code_match = (bcodes == acodes[ai]) & (acodes[ai] != "") & same_type
        nearby = np.flatnonzero(distances <= max_feet)
        nearest = sorted(nearby, key=lambda i: (distances[i], str(b.iloc[i]["stop_id"])))[:top_n]
        candidates = set(nearest) | set(np.flatnonzero(code_match))
        ordered = sorted(
            candidates,
            key=lambda i: (
                not bool(code_match[i]),
                distances[i] if np.isfinite(distances[i]) else math.inf,
                str(b.iloc[i]["stop_id"]),
            ),
        )
        for rank, bi in enumerate(ordered, start=1):
            rows.append(
                {
                    "primary_stop_key": primary_stop_key,
                    "after_comparison_key": akeys[ai],
                    "before_comparison_key": bkeys[bi],
                    "after_stop_id": a.iloc[ai]["stop_id"],
                    "before_stop_id": b.iloc[bi]["stop_id"],
                    "after_stop_name": anames[ai],
                    "before_stop_name": bnames[bi],
                    "after_stop_code": acodes[ai],
                    "before_stop_code": bcodes[bi],
                    "distance_ft": distances[bi],
                    "same_stop_code": bool(code_match[bi]),
                    "within_distance": bool(distances[bi] <= max_feet),
                    "after_routes": a.iloc[ai].get("routes", ""),
                    "before_routes": b.iloc[bi].get("routes", ""),
                    "rank": rank,
                    "review_note": "Candidate only; counts have not been reclassified.",
                }
            )
    out = pd.DataFrame(rows, columns=pd.Index(columns))
    if not out.empty:
        out["after_candidate_count"] = out.groupby("after_comparison_key")[
            "before_comparison_key"
        ].transform("nunique")
        out["before_used_by_after_count"] = out.groupby("before_comparison_key")[
            "after_comparison_key"
        ].transform("nunique")
    return out


# =============================================================================
# Assemble reports -- scope loss is distinct from a missing inventory record
# =============================================================================
ADDITIONS_OUTPUT_MODES = ("separate", "combined", "new_only", "existing_only")


def addition_output_tables(
    added: pd.DataFrame,
    output_mode: str,
    primary_stop_key: str = "stop_code",
) -> dict[str, pd.DataFrame]:
    """Separate inventory-new IDs from existing stops newly entering route scope.

    This partitions the additions-to-scope set, not all changes in routes serving
    existing stops. Route changes at already-shared stops remain in route_changes.
    A new ID may still be a renamed/rekeyed stop; retain candidate-match flags.
    The selection controls addition detail reports only, not comparison scope,
    summary totals, or supporting comparison/diagnostic tables.
    """
    if primary_stop_key not in PRIMARY_STOP_KEYS:
        raise ValueError(f"primary stop key must be one of {PRIMARY_STOP_KEYS}.")
    new_table = "new_stop_codes" if primary_stop_key == "stop_code" else "new_stop_ids"
    if output_mode not in ADDITIONS_OUTPUT_MODES:
        raise ValueError(f"additions output mode must be one of {ADDITIONS_OUTPUT_MODES}.")
    require_columns(added, {"present_in_before_inventory"}, "Addition report")
    out = added.copy()
    # Accept booleans produced by build_reports or their CSV text representation.
    # Do not use astype(bool): the text 'False' would otherwise become True.
    flags = normalize_text(out["present_in_before_inventory"]).str.casefold()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Addition report has missing/invalid inventory-presence flags.")
    existing_mask = flags.eq("true")
    out["present_in_before_inventory"] = existing_mask
    out["primary_stop_key"] = primary_stop_key
    out["addition_category"] = np.where(
        existing_mask, "existing_stop_new_to_scope", f"new_{primary_stop_key}"
    )
    new_ids = out.loc[~existing_mask].reset_index(drop=True)
    existing = out.loc[existing_mask].reset_index(drop=True)
    if output_mode == "combined":
        return {"added_to_scope": out.reset_index(drop=True)}
    if output_mode == "new_only":
        return {new_table: new_ids}
    if output_mode == "existing_only":
        return {"existing_stops_new_to_scope": existing}
    return {new_table: new_ids, "existing_stops_new_to_scope": existing}


def build_reports(
    before: Feed,
    after: Feed,
    scope: Scope,
    mode: str,
    threshold: float,
    ignored: Sequence[str],
    nearest_enabled: bool,
    nearest_feet: float,
    top_n: int,
    additions_output_mode: str = "separate",
    primary_stop_key: str = "stop_code",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build auditable tables without labelling partial-feed omissions closures."""
    full_b = comparison_inventory(
        annotate_stops(before, scope.before_ids, scope.before_route_ids),
        primary_stop_key,
        "before",
    )
    full_a = comparison_inventory(
        annotate_stops(after, scope.after_ids, scope.after_route_ids),
        primary_stop_key,
        "after",
    )
    b = full_b.loc[full_b["in_comparison_scope"]].copy()
    a = full_a.loc[full_a["in_comparison_scope"]].copy()
    common, attr_summary, attr_details, schema = compare_shared(
        b, a, threshold, ignored, primary_stop_key
    )
    before_scope_keys, after_scope_keys = (
        set(b["comparison_key"]),
        set(a["comparison_key"]),
    )
    removed = b.loc[~b["comparison_key"].isin(after_scope_keys)].copy()
    added = a.loc[~a["comparison_key"].isin(before_scope_keys)].copy()
    removed["present_in_after_inventory"] = removed["comparison_key"].isin(
        set(full_a["comparison_key"])
    )
    added["present_in_before_inventory"] = added["comparison_key"].isin(
        set(full_b["comparison_key"])
    )
    removed["current_other_routes"] = removed["other_routes"]
    added["current_routes_at_same_key"] = (
        added["comparison_key"].map(full_b.set_index("comparison_key")["routes"]).fillna("")
    )
    removed["interpretation"] = (
        "Not referenced by selected after routes; physical removal is unconfirmed."
        if mode == "after_routes"
        else f"{primary_stop_key} absent from after inventory; physical removal is unconfirmed."
    )
    added["interpretation"] = np.where(
        added["present_in_before_inventory"],
        f"Existing {primary_stop_key} newly used in the compared route scope.",
        f"{primary_stop_key} absent from before inventory; review candidate identity matches.",
    )
    # A concept can newly use a stop currently served only by unaffected routes.
    # It remains an addition to route scope, but its location/attributes still
    # need a comparison against the full current inventory.
    reused_ids = set(added.loc[added["present_in_before_inventory"], "comparison_key"])
    reused, _, _, _ = compare_shared(
        full_b.loc[full_b["comparison_key"].isin(reused_ids)],
        added.loc[added["comparison_key"].isin(reused_ids)],
        threshold,
        ignored,
        primary_stop_key,
    )
    reused_modified = reused["coordinate_changed_over_threshold"] | reused["attributes_changed"]
    changed = common["coordinate_changed_over_threshold"] | common["attributes_changed"]
    review = common["needs_review"]
    # Find rekeys among full-feed unmatched IDs, but only retain candidates
    # involving an addition/removal in the comparison scope.
    unmatched_b = full_b.loc[~full_b["comparison_key"].isin(set(full_a["comparison_key"]))]
    unmatched_a = full_a.loc[~full_a["comparison_key"].isin(set(full_b["comparison_key"]))]
    candidates = candidate_matches(
        unmatched_b if nearest_enabled else unmatched_b.iloc[:0],
        unmatched_a if nearest_enabled else unmatched_a.iloc[:0],
        nearest_feet,
        top_n,
        primary_stop_key,
    )
    candidates = candidates.loc[
        candidates["before_comparison_key"].isin(set(removed["comparison_key"]))
        | candidates["after_comparison_key"].isin(set(added["comparison_key"]))
    ].reset_index(drop=True)
    if not candidates.empty:
        candidates["after_candidate_count"] = candidates.groupby("after_comparison_key")[
            "before_comparison_key"
        ].transform("nunique")
        candidates["before_used_by_after_count"] = candidates.groupby("before_comparison_key")[
            "after_comparison_key"
        ].transform("nunique")
    removed["has_id_match_candidate"] = removed["comparison_key"].isin(
        set(candidates["before_comparison_key"])
    )
    added["has_id_match_candidate"] = added["comparison_key"].isin(
        set(candidates["after_comparison_key"])
    )
    addition_tables = addition_output_tables(added, additions_output_mode, primary_stop_key)
    exported_addition_count = sum(len(frame) for frame in addition_tables.values())
    route_change_mask = (
        common["scope_routes_before"].ne(common["scope_routes_after"])
        & common["route_context_available_before"]
        & common["route_context_available_after"]
    )
    route_changes = common.loc[
        route_change_mask,
        [
            primary_stop_key,
            "comparison_key",
            "primary_stop_key",
            "scope_routes_before",
            "scope_routes_after",
            "routes_before",
            "routes_after",
        ],
    ].copy()
    quality = pd.concat(
        [
            full_b.loc[full_b["coordinate_status"].ne("valid")].assign(feed="before"),
            full_a.loc[full_a["coordinate_status"].ne("valid")].assign(feed="after"),
        ],
        ignore_index=True,
    )
    identifier_changes = stop_id_code_changes(full_b, full_a)
    warnings = list(scope.warnings)
    raw_ids_b, raw_ids_a = set(full_b["comparison_key"]), set(full_a["comparison_key"])
    if (
        mode == "all_stops"
        and before.routes is not None
        and after.routes is not None
        and len(before.routes) != len(after.routes)
    ):
        warnings.append(
            "Route counts differ. all_stops omissions may reflect different feed scope."
        )
    if not common.empty and int(common["attributes_changed"].sum()) == len(common):
        warnings.append(
            "Every shared stop has attribute changes. "
            "Inspect attribute_summary for exporter-wide differences."
        )
    if before.pairs is None or after.pairs is None:
        warnings.append(
            "Route context unavailable on at least one side; "
            "blank route lists do not establish no service."
        )
    summary: dict[str, Any] = {
        "version": VERSION,
        "scope_mode": mode,
        "primary_stop_key": primary_stop_key,
        "route_pairing_complete": not scope.unresolved_routes,
        "confirmed_no_counterpart_routes": scope.confirmed_no_match_routes,
        "unresolved_concept_routes": scope.unresolved_routes,
        "before_inventory_stops": len(full_b),
        "after_inventory_stops": len(full_a),
        "before_inventory_routes": len(before.routes) if before.routes is not None else None,
        "after_inventory_routes": len(after.routes) if after.routes is not None else None,
        "raw_shared_keys": len(raw_ids_b & raw_ids_a),
        "raw_keys_absent_after": len(raw_ids_b - raw_ids_a),
        "raw_keys_absent_before": len(raw_ids_a - raw_ids_b),
        "before_scope_routes": len(scope.before_route_ids),
        "after_scope_routes": len(scope.after_route_ids),
        "before_scope_stops": len(b),
        "after_scope_stops": len(a),
        "before_outside_scope": len(full_b) - len(b),
        "after_outside_scope": len(full_a) - len(a),
        "shared_scope_keys": len(common),
        "modified_count": int(changed.sum()),
        "unchanged_count": int((~changed & ~review).sum()),
        "review_only_count": int((~changed & review).sum()),
        "needs_review_count": int(review.sum()),
        "modified_and_needs_review_count": int((changed & review).sum()),
        "coordinate_changed_count": int(common["coordinate_changed_over_threshold"].sum()),
        "attr_changed_count": int(common["attributes_changed"].sum()),
        "removed_from_scope_count": len(removed),
        "added_to_scope_count": len(added),
        "added_new_key_count": int((~added["present_in_before_inventory"]).sum()),
        "additions_output_mode": additions_output_mode,
        "exported_addition_count": exported_addition_count,
        "additions_not_in_primary_output_count": len(added) - exported_addition_count,
        "added_existing_inventory_count": int(added["present_in_before_inventory"].sum()),
        "reused_stops_modified_count": int(reused_modified.sum()),
        "reused_stops_needs_review_count": int(reused["needs_review"].sum()),
        "inventory_modified_in_after_scope_count": int(changed.sum() + reused_modified.sum()),
        "removed_with_current_other_routes": int(removed["current_other_routes"].ne("").sum()),
        "candidate_id_pairs": len(candidates),
        "shared_secondary_id_changes": int(common["secondary_id_changed"].sum()),
        "stop_id_code_change_count": len(identifier_changes),
        "route_context_changed_count": len(route_changes),
        "relocation_threshold_feet": threshold,
        "count_reconciliation": (
            "Shared = modified + unchanged + review_only. Needs_review overlaps modified."
        ),
        "interpretation": "Scope/ID changes do not prove physical stop openings or closures.",
        "warnings": warnings,
    }
    if summary["added_to_scope_count"] != (
        summary["added_new_key_count"] + summary["added_existing_inventory_count"]
    ):
        raise RuntimeError("Addition categories failed count reconciliation.")
    # Runtime reconciliation guards against lost or multiply classified records.
    if len(b) != len(common) + len(removed) or len(a) != len(common) + len(added):
        raise RuntimeError("Stop comparison failed count reconciliation.")
    if (
        len(common)
        != summary["modified_count"] + summary["unchanged_count"] + summary["review_only_count"]
    ):
        raise RuntimeError("Shared-stop categories failed reconciliation.")
    table_summary = pd.DataFrame(
        [
            {
                "metric": key,
                "value": " | ".join(value) if isinstance(value, list) else value,
            }
            for key, value in summary.items()
        ]
    )
    tables = {
        "summary": table_summary,
        "route_scope": scope.route_table,
        "modified": common.loc[changed].reset_index(drop=True),
        "removed_from_scope": removed.reset_index(drop=True),
        **addition_tables,
        "needs_review": common.loc[review].reset_index(drop=True),
        "reused_stop_comparison": reused,
        "unchanged": common.loc[~changed & ~review].reset_index(drop=True),
        "attribute_summary": attr_summary,
        "attribute_changes": attr_details,
        "schema_changes": schema,
        "nearest_id_matches": candidates,
        "stop_id_code_changes": identifier_changes,
        "route_changes": route_changes,
        "coordinate_issues": quality,
        "before": full_b,
        "after": full_a,
        "outside_scope_before": full_b.loc[~full_b["in_comparison_scope"]].reset_index(drop=True),
        "outside_scope_after": full_a.loc[~full_a["in_comparison_scope"]].reset_index(drop=True),
    }
    return tables, summary


# =============================================================================
# Export and orchestration
# =============================================================================
def write_outputs(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    summary: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> None:
    """Write plain sheets and CSVs, including empty candidate tables every run."""
    # A fresh run directory avoids mixing results with any previous version.
    with pd.ExcelWriter(output_dir / "stops_comparison.xlsx", engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_csv(output_dir / f"{sheet}.csv", index=False, encoding="utf-8-sig")
            frame.to_excel(writer, sheet_name=sheet, index=False)
        # Preserve GTFS text beginning '=' as text in Excel, including identifiers.
        for worksheet in writer.book.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"
    for filename, payload in (
        ("summary.json", summary),
        ("run_settings.json", settings),
    ):
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    pairings = {
        "schema_version": 1,
        "route_name_map": settings["route_name_map"],
        "after_routes_with_no_match": settings["after_routes_with_no_match"],
        "before_route_names": settings["before_route_names"],
        "unresolved_routes": settings["unresolved_routes"],
    }
    (output_dir / "route_pairings.json").write_text(
        json.dumps(pairings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Wrote workbook, %d CSV tables, and run metadata to %s", len(tables), output_dir)


def run_compare(
    before_dir: Path | None = None,
    after_dir: Path | None = None,
    out_dir: Path | None = None,
    threshold_feet: float | None = None,
    include_route_context: bool | None = None,
    scope_mode: str | None = None,
    before_route_names: Sequence[str] | None = None,
    route_name_map: Mapping[str, str | Sequence[str]] | None = None,
    no_match_routes: Sequence[str] | None = None,
    unmatched_policy: str | None = None,
    route_pairs_file: Path | None = None,
    additions_output_mode: str | None = None,
    primary_stop_key: str | None = None,
) -> dict[str, Any]:
    """Notebook-friendly entry point. Defaults are read at call time, not import."""
    before_path = resolve_path(Path(before_dir if before_dir is not None else BEFORE_GTFS_DIR))
    after_path = resolve_path(Path(after_dir if after_dir is not None else AFTER_GTFS_DIR))
    output_root = resolve_path(Path(out_dir if out_dir is not None else OUTPUT_DIR))
    threshold = float(threshold_feet if threshold_feet is not None else RELOCATE_THRESHOLD_FEET)
    context = include_route_context if include_route_context is not None else ENABLE_ROUTE_CONTEXT
    mode = scope_mode if scope_mode is not None else SCOPE_MODE
    stop_key = primary_stop_key if primary_stop_key is not None else PRIMARY_STOP_KEY
    if stop_key not in PRIMARY_STOP_KEYS:
        raise ValueError(f"primary stop key must be one of {PRIMARY_STOP_KEYS}.")
    additions_mode = (
        additions_output_mode if additions_output_mode is not None else ADDITIONS_OUTPUT_MODE
    )
    if additions_mode not in ADDITIONS_OUTPUT_MODES:
        raise ValueError(f"additions output mode must be one of {ADDITIONS_OUTPUT_MODES}.")
    default_mapping = AFTER_TO_BEFORE_ROUTE_NAMES
    default_no_match = AFTER_ROUTES_WITH_NO_MATCH
    default_names = BEFORE_ROUTE_NAMES
    pairing_path = route_pairs_file if route_pairs_file is not None else ROUTE_PAIRS_FILE
    if pairing_path is not None and (
        route_name_map is None or no_match_routes is None or before_route_names is None
    ):
        default_mapping, default_no_match, default_names = read_pairing_file(pairing_path)
    raw_names = before_route_names if before_route_names is not None else default_names
    raw_no_match = no_match_routes if no_match_routes is not None else default_no_match
    normalize_route_names(raw_names, "BEFORE_ROUTE_NAMES")
    normalize_route_names(raw_no_match, "AFTER_ROUTES_WITH_NO_MATCH")
    names = list(raw_names)
    no_match = list(raw_no_match)
    mapping = dict(route_name_map if route_name_map is not None else default_mapping)
    policy = unmatched_policy if unmatched_policy is not None else UNMATCHED_ROUTE_POLICY
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Relocation threshold must be finite and nonnegative.")
    if mode not in {"after_routes", "all_stops"}:
        raise ValueError("scope must be after_routes or all_stops.")
    if mode == "after_routes" and not context:
        raise ValueError(
            "after_routes scope requires route context; enable it or choose all_stops."
        )
    for path in (before_path, after_path):
        if not path.exists():
            raise FileNotFoundError(f"GTFS source not found: {path}")
    output_dir = output_root / datetime.now().strftime("stop_diff_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(output_dir)
    LOGGER.info("Before GTFS: %s", before_path)
    LOGGER.info("After GTFS: %s", after_path)
    LOGGER.info(
        "Scope: %s; primary stop key: %s; coordinate threshold: %.1f ft",
        mode,
        stop_key,
        threshold,
    )
    before = load_feed(before_path, "before", context, mode == "after_routes")
    after = load_feed(after_path, "after", context, mode == "after_routes")
    # Validate before prompting, so invalid stop codes do not waste route-pairing work.
    comparison_inventory(before.stops, stop_key, "before")
    comparison_inventory(after.stops, stop_key, "after")
    scope = make_scope(before, after, mode, names, mapping, no_match, policy)
    LOGGER.info(
        "Scoped stops: before=%d of %d; after=%d of %d. Scoped routes: %d -> %d.",
        len(scope.before_ids),
        len(before.stops),
        len(scope.after_ids),
        len(after.stops),
        len(scope.before_route_ids),
        len(scope.after_route_ids),
    )
    tables, summary = build_reports(
        before,
        after,
        scope,
        mode,
        threshold,
        IGNORE_ATTRIBUTES,
        ENABLE_NEAREST_MATCHES,
        NEAREST_MATCHES_MAX_FEET,
        NEAREST_CANDIDATES_PER_STOP,
        additions_output_mode=additions_mode,
        primary_stop_key=stop_key,
    )
    summary["output_directory"] = str(output_dir)
    tables["summary"] = pd.concat(
        [
            tables["summary"],
            pd.DataFrame([{"metric": "output_directory", "value": str(output_dir)}]),
        ],
        ignore_index=True,
    )
    for warning in summary["warnings"]:
        LOGGER.warning("%s", warning)
    for _, row in tables["attribute_summary"].head(10).iterrows():
        if row["changed_count"]:
            LOGGER.info(
                "Attribute %s: %d changed (%.1f%%); populated->blank=%d; examples: %s",
                row["field"],
                row["changed_count"],
                row["changed_pct"],
                row["populated_to_blank"],
                row["examples"],
            )
    settings = {
        "version": VERSION,
        "before": str(before_path),
        "after": str(after_path),
        "scope_mode": mode,
        "before_route_names": names,
        "route_name_map": scope.resolved_route_pairs,
        "after_routes_with_no_match": scope.confirmed_no_match_routes,
        "unresolved_routes": scope.unresolved_routes,
        "unmatched_route_policy": policy,
        "additions_output_mode": additions_mode,
        "threshold_feet": threshold,
        "primary_stop_key": stop_key,
        "include_route_context": context,
        "ignore_attributes": IGNORE_ATTRIBUTES,
        "nearest_matches_enabled": ENABLE_NEAREST_MATCHES,
        "nearest_max_feet": NEAREST_MATCHES_MAX_FEET,
        "nearest_candidates_per_stop": NEAREST_CANDIDATES_PER_STOP,
        "service_date_filter": None,
    }
    write_outputs(output_dir, tables, summary, settings)
    LOGGER.info(
        "Done. Modified=%d; unchanged=%d; review-only=%d; removed from scope=%d; "
        "added to scope=%d (new primary keys=%d; existing stops=%d).",
        summary["modified_count"],
        summary["unchanged_count"],
        summary["review_only_count"],
        summary["removed_from_scope_count"],
        summary["added_to_scope_count"],
        summary["added_new_key_count"],
        summary["added_existing_inventory_count"],
    )
    LOGGER.info(
        "Addition detail mode=%s; exported %d of %d additions. Summary retains both categories.",
        additions_mode,
        summary["exported_addition_count"],
        summary["added_to_scope_count"],
    )
    return summary


# =============================================================================
# Strict CLI parsing; notebook-safe arguments and exit handling
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read current config defaults while permitting explicit overrides."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--primary-stop-key",
        choices=PRIMARY_STOP_KEYS,
        default=PRIMARY_STOP_KEY,
        help="Cross-feed stop identifier; GTFS joins always use stop_id",
    )
    parser.add_argument("--before", type=Path, default=BEFORE_GTFS_DIR)
    parser.add_argument("--after", type=Path, default=AFTER_GTFS_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threshold-feet", type=float, default=RELOCATE_THRESHOLD_FEET)
    parser.add_argument("--scope", choices=["after_routes", "all_stops"], default=SCOPE_MODE)
    parser.add_argument(
        "--before-routes",
        nargs="*",
        default=None,
        help="Extra current route names, in addition to all resolved pairs",
    )
    parser.add_argument(
        "--route-pair",
        nargs=2,
        action="append",
        metavar=("AFTER", "BEFORE"),
        help="Explicit concept/current route pair; repeat for splits or mergers",
    )
    parser.add_argument(
        "--no-route-match",
        action="append",
        metavar="AFTER",
        help="Confirm a concept route has no current counterpart; keep it in scope",
    )
    parser.add_argument(
        "--unmatched-routes",
        choices=["prompt", "error", "warn"],
        default=UNMATCHED_ROUTE_POLICY,
        help="How to handle concept routes without a resolved counterpart",
    )
    parser.add_argument(
        "--route-pairs-file",
        type=Path,
        default=ROUTE_PAIRS_FILE,
        help="Reuse a saved route_pairings.json or run_settings.json",
    )
    parser.add_argument(
        "--additions",
        choices=ADDITIONS_OUTPUT_MODES,
        default=ADDITIONS_OUTPUT_MODE,
        help=(
            "Addition detail output: separate categories, combined, new IDs only, "
            "or existing stops only"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--route-context", dest="route_context", action="store_true")
    group.add_argument("--no-route-context", dest="route_context", action="store_false")
    parser.set_defaults(route_context=ENABLE_ROUTE_CONTEXT)
    return parser.parse_args(notebook_safe_argv(argv))


def read_pairing_file(path: Path) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Read saved decisions without applying paths or unrelated run settings."""
    path = resolve_path(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("route_name_map"), dict):
        raise TypeError("Pairing file must be a JSON object containing route_name_map.")
    if "schema_version" in payload and payload["schema_version"] != 1:
        raise ValueError("Unsupported route-pairing file schema_version.")
    mapping = normalize_route_pairs(payload["route_name_map"])
    raw_no_match = payload.get("after_routes_with_no_match", [])
    raw_extras = payload.get("before_route_names", [])
    if not isinstance(raw_no_match, list) or not isinstance(raw_extras, list):
        raise TypeError("Pairing-file no-match choices and before_route_names must be lists.")
    no_match = normalize_route_names(raw_no_match, "after_routes_with_no_match")
    normalize_route_names(raw_extras, "before_route_names")
    extra_names = list(raw_extras)
    return mapping, sorted(no_match), extra_names


def cli_pairing_options(
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Load saved choices, then apply per-concept-route CLI overrides.

    A pairing file replaces configured choices. CLI choices then override the
    file/config for each named concept route. Multiple --route-pair entries for
    the same concept route form a merger. Contradictory CLI decisions are errors.
    """
    mapping = normalize_route_pairs(AFTER_TO_BEFORE_ROUTE_NAMES)
    no_match = normalize_route_names(AFTER_ROUTES_WITH_NO_MATCH, "AFTER_ROUTES_WITH_NO_MATCH")
    extra_names = list(BEFORE_ROUTE_NAMES)
    if args.route_pairs_file is not None:
        mapping, saved_no_match, extra_names = read_pairing_file(args.route_pairs_file)
        no_match = set(saved_no_match)
    command_pairs: dict[str, list[str]] = {}
    for after_name, before_name in args.route_pair or []:
        if not after_name.strip() or not before_name.strip():
            raise ValueError("--route-pair requires two nonblank route names: AFTER BEFORE.")
        key = after_name.strip().casefold()
        command_pairs.setdefault(key, []).append(before_name.strip())
    command_no_match = normalize_route_names(args.no_route_match or [], "--no-route-match")
    conflict = set(command_pairs) & command_no_match
    if conflict:
        raise ValueError(
            f"Conflicting --route-pair and --no-route-match choices: {sorted(conflict)}"
        )
    for key, targets in command_pairs.items():
        mapping[key] = targets
        no_match.discard(key)
    for key in command_no_match:
        mapping.pop(key, None)
        no_match.add(key)
    if args.before_routes is not None:
        extra_names = list(args.before_routes)
    return mapping, sorted(no_match), extra_names


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured comparison, logging actionable failures."""
    args = parse_args(argv)
    try:
        mapping, no_match, names = cli_pairing_options(args)
        run_compare(
            before_dir=args.before,
            after_dir=args.after,
            out_dir=args.out,
            threshold_feet=args.threshold_feet,
            include_route_context=args.route_context,
            scope_mode=args.scope,
            before_route_names=names,
            route_name_map=mapping,
            no_match_routes=no_match,
            unmatched_policy=args.unmatched_routes,
            additions_output_mode=args.additions,
            primary_stop_key=args.primary_stop_key,
        )
    except KeyboardInterrupt:
        print("Route comparison cancelled.", file=sys.stderr)  # noqa: T201
        return 130
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        if LOGGER.handlers:
            LOGGER.exception("Comparison failed")
        else:
            print(f"Comparison failed: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    if "ipykernel" not in sys.modules:
        raise SystemExit(exit_code)
    if exit_code:
        print("Comparison failed; see the error above.")  # noqa: T201
