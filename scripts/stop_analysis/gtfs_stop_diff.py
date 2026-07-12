"""GTFS stop comparison (before vs after) with notebook-friendly execution.

Outputs (CSV):
- stops_before.csv     : all stops from before feed
- stops_after.csv      : all stops from after feed
- stops_modified.csv   : overlap stop_id where relocated > threshold and/or attributes changed
- stops_deleted.csv    : stop_id present only in before feed
- stops_new.csv        : stop_id present only in after feed
- summary.json

Also outputs:
- stops_comparison.xlsx (sheets: before, after, modified, deleted, new, summary,
  optional nearest_id_matches)
- gtfs_stop_diff.log

When route context is enabled (the default), each stop is annotated with the
routes that serve it (via stop_times -> trips -> routes). This is informational
only and never affects the modified/unchanged classification.

No arcpy / geopandas. pandas + numpy + scipy only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =============================================================================
# Config
# =============================================================================

BEFORE_GTFS_DIR = Path(r"Path\To\GTFS\Dir")
AFTER_GTFS_DIR = Path(r"Path\to\GTFS\Dir")
OUTPUT_DIR = Path(r"Path\To\Output\Dir")

RELOCATE_THRESHOLD_FEET = 25.0
OVERLAP_WARN_THRESHOLD = 0.10  # warn if overlap fraction < 10%

ENABLE_NEAREST_MATCHES_WHEN_LOW_OVERLAP = True
NEAREST_MATCHES_MAX_FEET = 500.0  # only report nearest matches within this distance

# Route context: annotate each stop with the routes that serve it (via
# stop_times -> trips -> routes). This is informational only; route changes
# do NOT affect the modified/unchanged classification.
ENABLE_ROUTE_CONTEXT = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class Summary:
    """Summary metrics for the comparison."""

    before_stop_count: int
    after_stop_count: int
    overlap_stop_count: int
    overlap_fraction_of_before: float
    overlap_fraction_of_after: float
    modified_count: int
    unchanged_count: int
    new_count: int
    deleted_count: int
    relocated_count: int
    attr_changed_count: int


# =============================================================================
# Logging
# =============================================================================


def setup_logging(output_dir: Path) -> None:
    """Configure root logger to write to console + a file in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "gtfs_stop_diff.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# =============================================================================
# IO helpers
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


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text for comparisons."""
    return series.fillna("").astype(str).str.strip()


def coerce_float(series: pd.Series) -> pd.Series:
    """Convert a string series to float; invalid values become NaN."""
    return pd.to_numeric(series, errors="coerce").astype(float)


def validate_stop_ids_unique(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Ensure stop_id is unique; if not, warn and keep the first occurrence per stop_id."""
    if "stop_id" not in df.columns:
        raise ValueError(f"{label}: stops.txt is missing required column 'stop_id'.")

    dup_mask = df["stop_id"].duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        dup_ids = df.loc[dup_mask, "stop_id"].head(20).tolist()
        logging.warning(
            "%s: found %s duplicate stop_id values; keeping first occurrence. Sample: %s",
            label,
            dup_count,
            dup_ids,
        )
        df = df.loc[~dup_mask].copy()

    return df


def load_stops(gtfs_path: Path, label: str) -> pd.DataFrame:
    """Load and standardize GTFS stops using the canonical helper."""
    # load_gtfs_data expects a str path
    data = load_gtfs_data(str(gtfs_path), files=["stops.txt"])
    df = data["stops"]

    required = {"stop_id", "stop_lat", "stop_lon"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label}: stops.txt missing required columns: {missing}")

    df = df.copy()
    df["stop_id"] = normalize_text(df["stop_id"])

    if "stop_name" in df.columns:
        df["stop_name"] = normalize_text(df["stop_name"])

    df["stop_lat"] = coerce_float(df["stop_lat"])
    df["stop_lon"] = coerce_float(df["stop_lon"])

    df = validate_stop_ids_unique(df, label=label)

    missing_xy = int(df["stop_lat"].isna().sum() + df["stop_lon"].isna().sum())
    if missing_xy > 0:
        logging.warning("%s: %s rows have missing/invalid stop_lat/stop_lon.", label, missing_xy)

    return df


# =============================================================================
# Route context
# =============================================================================


def _route_display_label(row: pd.Series) -> str:
    """Best human-readable label for a route: short name, else long name, else ID."""
    short = str(row.get("route_short_name") or "").strip()
    if short and short.lower() != "nan":
        return short
    long_name = str(row.get("route_long_name") or "").strip()
    if long_name and long_name.lower() != "nan":
        return long_name
    return str(row.get("route_id") or "").strip()


def _route_sort_key(label: str) -> tuple[int, int, str]:
    """Sort numeric route labels numerically, then everything else alphabetically."""
    if label.isdigit():
        return (0, int(label), label)
    return (1, 0, label)


def load_stop_route_pairs(gtfs_path: Path, label: str) -> pd.DataFrame | None:
    """Build unique (stop_id, route_id, route_label) pairs for a feed.

    Uses stop_times → trips → routes joins over the entire feed (all service
    days). Returns ``None`` (with a warning) if the required files or columns
    are unavailable, so downstream steps can degrade gracefully.
    """
    try:
        data = load_gtfs_data(str(gtfs_path), files=["routes.txt", "trips.txt"])
    except (OSError, ValueError) as exc:
        logging.warning(
            "%s: could not load routes.txt/trips.txt for route context (%s). "
            "Route context will be skipped.",
            label,
            exc,
        )
        return None

    routes = data["routes"].copy()
    trips = data["trips"].copy()

    if "route_id" not in routes.columns:
        logging.warning("%s: routes.txt missing 'route_id'; skipping route context.", label)
        return None
    if not {"trip_id", "route_id"}.issubset(trips.columns):
        logging.warning(
            "%s: trips.txt missing 'trip_id'/'route_id'; skipping route context.", label
        )
        return None

    # stop_times can be large, so read only the two columns we need rather
    # than going through load_gtfs_data (which loads every column).
    st_path = os.path.join(str(gtfs_path), "stop_times.txt")
    if not os.path.exists(st_path):
        logging.warning("%s: stop_times.txt not found; skipping route context.", label)
        return None
    try:
        stop_times = pd.read_csv(st_path, dtype=str, usecols=["trip_id", "stop_id"])
    except ValueError as exc:
        logging.warning(
            "%s: stop_times.txt missing 'trip_id'/'stop_id' (%s); skipping route context.",
            label,
            exc,
        )
        return None

    stop_times["trip_id"] = normalize_text(stop_times["trip_id"])
    stop_times["stop_id"] = normalize_text(stop_times["stop_id"])
    trips["trip_id"] = normalize_text(trips["trip_id"])
    trips["route_id"] = normalize_text(trips["route_id"])
    routes["route_id"] = normalize_text(routes["route_id"])

    routes["route_label"] = routes.apply(_route_display_label, axis=1)
    label_by_route_id = dict(zip(routes["route_id"], routes["route_label"]))

    pairs = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id", how="inner")[
        ["stop_id", "route_id"]
    ].drop_duplicates()
    if pairs.empty:
        logging.warning("%s: no stop/route pairs found; skipping route context.", label)
        return None

    # Fall back to the raw route_id if it has no entry in routes.txt.
    pairs["route_label"] = pairs["route_id"].map(label_by_route_id)
    pairs["route_label"] = pairs["route_label"].fillna(pairs["route_id"])

    logging.info(
        "%s: built %s stop/route pairs (%s routes, %s stops with service).",
        label,
        len(pairs),
        pairs["route_id"].nunique(),
        pairs["stop_id"].nunique(),
    )
    return pairs.reset_index(drop=True)


def build_stop_routes_table(pairs: pd.DataFrame | None) -> pd.DataFrame | None:
    """Collapse stop/route pairs into stop_id → routes (semicolon list) + route_count."""
    if pairs is None:
        return None

    grouped = pairs.groupby("stop_id").agg(
        routes=("route_label", lambda s: "; ".join(sorted(set(s), key=_route_sort_key))),
        route_count=("route_id", "nunique"),
    )
    return grouped.reset_index()


def attach_route_context(
    stops_df: pd.DataFrame, stop_routes: pd.DataFrame | None, label: str
) -> pd.DataFrame:
    """Left-join route context onto a stops table; no-op if context unavailable."""
    if stop_routes is None:
        return stops_df

    out = stops_df.merge(stop_routes, on="stop_id", how="left")
    out["routes"] = out["routes"].fillna("")
    out["route_count"] = out["route_count"].fillna(0).astype(int)

    unserved = int((out["route_count"] == 0).sum())
    if unserved > 0:
        logging.info(
            "%s: %s stops have no trips in stop_times.txt (routes column left blank).",
            label,
            unserved,
        )
    return out


# =============================================================================
# Distance
# =============================================================================


def haversine_meters(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (meters) between arrays of lat/lon in degrees."""
    r = 6_371_000.0
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def meters_to_feet(meters: np.ndarray) -> np.ndarray:
    """Convert meters to feet."""
    return meters * 3.280839895013123


# =============================================================================
# Comparison logic
# =============================================================================


def pick_attribute_columns(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    """Columns to compare for attribute changes (only those present in both feeds).

    Note: 'routes' / 'route_count' are deliberately excluded — route context is
    informational and must not drive the modified/unchanged classification.
    """
    candidates = [
        "stop_name",
        "stop_code",
        "stop_desc",
        "zone_id",
        "location_type",
        "parent_station",
        "stop_timezone",
        "wheelchair_boarding",
        "platform_code",
    ]
    return [c for c in candidates if c in before.columns and c in after.columns]


def build_modified_description(
    relocated: bool, changed_fields: list[str], distance_ft: float | None
) -> str:
    """Build a compact description of what changed for a modified stop."""
    parts: list[str] = []
    if relocated:
        if distance_ft is None or not np.isfinite(distance_ft):
            parts.append("Relocated (> threshold), distance unavailable.")
        else:
            parts.append(f"Relocated {distance_ft:.1f} ft (> threshold).")
    if changed_fields:
        parts.append(f"Fields changed: {';'.join(changed_fields)}")
    return " ".join(parts).strip()


def compare_stops(
    before: pd.DataFrame,
    after: pd.DataFrame,
    relocate_threshold_ft: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Summary, pd.DataFrame | None]:
    """Compare stops from two GTFS feeds.

    If the input frames carry a ``routes`` column (see attach_route_context),
    it is passed through to every output — including ``routes_before`` /
    ``routes_after`` on the modified sheet — but never affects classification.

    Returns:
        modified_df, deleted_df, new_df, unchanged_df, summary, nearest_matches(optional)
    """
    before_ids = set(before["stop_id"].tolist())
    after_ids = set(after["stop_id"].tolist())
    overlap_ids = before_ids & after_ids

    overlap_fraction_of_before = (len(overlap_ids) / len(before_ids)) if before_ids else 0.0
    overlap_fraction_of_after = (len(overlap_ids) / len(after_ids)) if after_ids else 0.0

    logging.info("Before stops: %s", len(before_ids))
    logging.info("After stops:  %s", len(after_ids))
    logging.info("Overlap IDs:  %s", len(overlap_ids))
    logging.info("Overlap as %% of before: %.1f%%", 100.0 * overlap_fraction_of_before)
    logging.info("Overlap as %% of after:  %.1f%%", 100.0 * overlap_fraction_of_after)

    if min(overlap_fraction_of_before, overlap_fraction_of_after) < OVERLAP_WARN_THRESHOLD:
        logging.warning(
            "Stop_id overlap is under %.0f%%. This often means either a major system overhaul "
            "or a stop_id renumbering/rekeying.",
            100.0 * OVERLAP_WARN_THRESHOLD,
        )

    deleted_ids = before_ids - after_ids
    new_ids = after_ids - before_ids

    deleted_df = before.loc[before["stop_id"].isin(deleted_ids)].copy()
    new_df = after.loc[after["stop_id"].isin(new_ids)].copy()

    # Compare overlap stops
    before_o = before.loc[before["stop_id"].isin(overlap_ids)].copy()
    after_o = after.loc[after["stop_id"].isin(overlap_ids)].copy()

    merged = before_o.merge(after_o, on="stop_id", how="inner", suffixes=("_before", "_after"))

    # Distance
    lat_b = merged["stop_lat_before"].to_numpy(dtype=float)
    lon_b = merged["stop_lon_before"].to_numpy(dtype=float)
    lat_a = merged["stop_lat_after"].to_numpy(dtype=float)
    lon_a = merged["stop_lon_after"].to_numpy(dtype=float)

    valid_xy = ~(np.isnan(lat_b) | np.isnan(lon_b) | np.isnan(lat_a) | np.isnan(lon_a))
    dist_ft = np.full(shape=(len(merged),), fill_value=np.nan, dtype=float)
    if int(valid_xy.sum()) > 0:
        meters = haversine_meters(
            lat_b[valid_xy], lon_b[valid_xy], lat_a[valid_xy], lon_a[valid_xy]
        )
        dist_ft[valid_xy] = meters_to_feet(meters)

    merged["distance_ft"] = dist_ft
    merged["delta_lat"] = merged["stop_lat_after"] - merged["stop_lat_before"]
    merged["delta_lon"] = merged["stop_lon_after"] - merged["stop_lon_before"]

    relocated_mask = merged["distance_ft"] > relocate_threshold_ft

    # Attribute changes
    attr_cols = pick_attribute_columns(before, after)
    changed_fields_col: list[str] = []

    for _, row in merged.iterrows():
        changed_fields: list[str] = []
        for c in attr_cols:
            b = str(row.get(f"{c}_before", "") or "").strip()
            a = str(row.get(f"{c}_after", "") or "").strip()
            if b != a:
                changed_fields.append(c)
        changed_fields_col.append(";".join(changed_fields))

    merged["changed_fields"] = changed_fields_col
    attr_changed_mask = merged["changed_fields"].astype(str).str.len() > 0

    modified_mask = relocated_mask | attr_changed_mask
    unchanged_mask = ~modified_mask

    def classify_type(relocated: bool, attr_changed: bool) -> str:
        if relocated and attr_changed:
            return "relocated+attrs"
        if relocated:
            return "relocated"
        if attr_changed:
            return "attrs"
        return "unchanged"

    relocated_arr = relocated_mask.to_numpy()
    attr_changed_arr = attr_changed_mask.to_numpy()
    merged["change_type"] = [
        classify_type(bool(r), bool(a)) for r, a in zip(relocated_arr, attr_changed_arr)
    ]

    # Friendly description
    descs: list[str] = []
    for _, row in merged.iterrows():
        relocated = row["change_type"] in {"relocated", "relocated+attrs"}
        changed_fields = [f for f in str(row.get("changed_fields", "")).split(";") if f]
        dist = row.get("distance_ft")
        dist_val = float(dist) if pd.notna(dist) else None
        descs.append(
            build_modified_description(
                relocated=relocated, changed_fields=changed_fields, distance_ft=dist_val
            )
        )

    merged["change_description"] = descs

    modified_df = merged.loc[modified_mask].copy()
    unchanged_df = merged.loc[unchanged_mask].copy()

    # Order columns for modified output (keep it readable)
    key_cols = [
        "stop_id",
        "change_type",
        "change_description",
        "distance_ft",
        "delta_lat",
        "delta_lon",
        "changed_fields",
    ]
    before_cols = [c for c in modified_df.columns if c.endswith("_before")]
    after_cols = [c for c in modified_df.columns if c.endswith("_after")]

    # Prefer to show stop_name/lat/lon/routes early
    def sort_cols(cols: list[str]) -> list[str]:
        priority = {"stop_name": 0, "stop_lat": 1, "stop_lon": 2, "routes": 3, "route_count": 4}
        return sorted(
            cols,
            key=lambda x: (
                priority.get(x.replace("_before", "").replace("_after", ""), 99),
                x,
            ),
        )

    before_cols = sort_cols(before_cols)
    after_cols = sort_cols(after_cols)

    keep_cols = [c for c in key_cols if c in modified_df.columns] + before_cols + after_cols
    modified_df = (
        modified_df[keep_cols].sort_values(["change_type", "stop_id"]).reset_index(drop=True)
    )

    # Sort other outputs
    deleted_df = deleted_df.sort_values("stop_id").reset_index(drop=True)
    new_df = new_df.sort_values("stop_id").reset_index(drop=True)
    unchanged_df = unchanged_df.sort_values("stop_id").reset_index(drop=True)

    summary = Summary(
        before_stop_count=len(before_ids),
        after_stop_count=len(after_ids),
        overlap_stop_count=len(overlap_ids),
        overlap_fraction_of_before=float(overlap_fraction_of_before),
        overlap_fraction_of_after=float(overlap_fraction_of_after),
        modified_count=int(modified_mask.sum()),
        unchanged_count=int(unchanged_mask.sum()),
        new_count=len(new_df),
        deleted_count=len(deleted_df),
        relocated_count=int(relocated_mask.sum()),
        attr_changed_count=int(attr_changed_mask.sum()),
    )

    nearest_matches = None
    if (
        ENABLE_NEAREST_MATCHES_WHEN_LOW_OVERLAP
        and min(overlap_fraction_of_before, overlap_fraction_of_after) < OVERLAP_WARN_THRESHOLD
    ):
        nearest_matches = try_build_nearest_matches(
            before=before,
            after=after,
            max_feet=NEAREST_MATCHES_MAX_FEET,
        )

    return modified_df, deleted_df, new_df, unchanged_df, summary, nearest_matches


def try_build_nearest_matches(
    before: pd.DataFrame,
    after: pd.DataFrame,
    max_feet: float,
) -> pd.DataFrame | None:
    """Optional helper when stop_id overlap is very low.

    For each AFTER stop, finds nearest BEFORE stop by coordinates (within max_feet).
    If route context is present on the inputs, the routes serving each stop are
    included — helpful for confirming that a candidate ID rekeying is plausible.
    """
    has_routes = "routes" in before.columns and "routes" in after.columns

    b_cols = ["stop_id", "stop_lat", "stop_lon"] + (["routes"] if has_routes else [])
    a_cols = ["stop_id", "stop_lat", "stop_lon"] + (["routes"] if has_routes else [])

    b = before[b_cols].dropna(subset=["stop_lat", "stop_lon"]).copy()
    a = after[a_cols].dropna(subset=["stop_lat", "stop_lon"]).copy()
    if b.empty or a.empty:
        logging.info("Insufficient valid coordinates for nearest-match output.")
        return None

    lat0 = float(pd.concat([b["stop_lat"], a["stop_lat"]], ignore_index=True).mean())
    meters_per_degree = 111_320.0
    cos0 = math.cos(math.radians(lat0))

    bx = b["stop_lon"].to_numpy() * cos0 * meters_per_degree
    by = b["stop_lat"].to_numpy() * meters_per_degree
    ax = a["stop_lon"].to_numpy() * cos0 * meters_per_degree
    ay = a["stop_lat"].to_numpy() * meters_per_degree

    tree = cKDTree(np.column_stack([bx, by]))
    _, idx = tree.query(np.column_stack([ax, ay]), k=1)

    nearest_before_ids = b["stop_id"].to_numpy()[idx]
    meters = haversine_meters(
        a["stop_lat"].to_numpy(),
        a["stop_lon"].to_numpy(),
        b["stop_lat"].to_numpy()[idx],
        b["stop_lon"].to_numpy()[idx],
    )
    dist_ft = meters_to_feet(meters)

    data: dict[str, Any] = {
        "after_stop_id": a["stop_id"].to_numpy(),
        "nearest_before_stop_id": nearest_before_ids,
        "nearest_distance_ft": dist_ft,
    }
    if has_routes:
        data["after_routes"] = a["routes"].to_numpy()
        data["nearest_before_routes"] = b["routes"].to_numpy()[idx]

    out = pd.DataFrame(data)
    out = (
        out.loc[out["nearest_distance_ft"] <= max_feet]
        .sort_values("nearest_distance_ft")
        .reset_index(drop=True)
    )

    logging.info("Nearest-match output created (%s rows within %.0f ft).", len(out), max_feet)
    return out


# =============================================================================
# Export
# =============================================================================


def write_outputs(
    output_dir: Path,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    modified_df: pd.DataFrame,
    deleted_df: pd.DataFrame,
    new_df: pd.DataFrame,
    summary: Summary,
    nearest_matches: pd.DataFrame | None,
) -> None:
    """Write CSVs + Excel workbook + summary json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    before_csv = output_dir / "stops_before.csv"
    after_csv = output_dir / "stops_after.csv"
    modified_csv = output_dir / "stops_modified.csv"
    deleted_csv = output_dir / "stops_deleted.csv"
    new_csv = output_dir / "stops_new.csv"
    summary_json = output_dir / "summary.json"
    xlsx_path = output_dir / "stops_comparison.xlsx"

    before_df.to_csv(before_csv, index=False, encoding="utf-8")
    after_df.to_csv(after_csv, index=False, encoding="utf-8")
    modified_df.to_csv(modified_csv, index=False, encoding="utf-8")
    deleted_df.to_csv(deleted_csv, index=False, encoding="utf-8")
    new_df.to_csv(new_csv, index=False, encoding="utf-8")

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    # Use openpyxl engine explicitly
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        before_df.to_excel(writer, sheet_name="before", index=False)
        after_df.to_excel(writer, sheet_name="after", index=False)
        modified_df.to_excel(writer, sheet_name="modified", index=False)
        deleted_df.to_excel(writer, sheet_name="deleted", index=False)
        new_df.to_excel(writer, sheet_name="new", index=False)
        pd.DataFrame([asdict(summary)]).to_excel(writer, sheet_name="summary", index=False)

        if nearest_matches is not None and not nearest_matches.empty:
            nearest_matches.to_excel(writer, sheet_name="nearest_id_matches", index=False)

    logging.info("Wrote: %s", before_csv)
    logging.info("Wrote: %s", after_csv)
    logging.info("Wrote: %s", modified_csv)
    logging.info("Wrote: %s", deleted_csv)
    logging.info("Wrote: %s", new_csv)
    logging.info("Wrote: %s", summary_json)
    logging.info("Wrote: %s", xlsx_path)

    if nearest_matches is not None:
        nm_csv = output_dir / "nearest_id_matches.csv"
        nearest_matches.to_csv(nm_csv, index=False, encoding="utf-8")
        logging.info("Wrote: %s", nm_csv)


# =============================================================================
# Notebook-friendly entry point
# =============================================================================


def run_compare(
    before_dir: Path = BEFORE_GTFS_DIR,
    after_dir: Path = AFTER_GTFS_DIR,
    out_dir: Path = OUTPUT_DIR,
    threshold_feet: float = RELOCATE_THRESHOLD_FEET,
    include_route_context: bool = ENABLE_ROUTE_CONTEXT,
) -> Summary:
    """Run the comparison (notebook-friendly) and write outputs."""
    setup_logging(out_dir)

    logging.info("Before GTFS: %s", before_dir)
    logging.info("After GTFS:  %s", after_dir)
    logging.info("Output dir:  %s", out_dir)
    logging.info("Relocation threshold: %.1f ft", threshold_feet)
    logging.info("Route context: %s", "enabled" if include_route_context else "disabled")

    before_df = load_stops(before_dir, label="before")
    after_df = load_stops(after_dir, label="after")

    if include_route_context:
        before_df = attach_route_context(
            before_df,
            build_stop_routes_table(load_stop_route_pairs(before_dir, label="before")),
            label="before",
        )
        after_df = attach_route_context(
            after_df,
            build_stop_routes_table(load_stop_route_pairs(after_dir, label="after")),
            label="after",
        )

    modified_df, deleted_df, new_df, _unchanged_df, summary, nearest_matches = compare_stops(
        before=before_df,
        after=after_df,
        relocate_threshold_ft=float(threshold_feet),
    )

    write_outputs(
        output_dir=out_dir,
        before_df=before_df,
        after_df=after_df,
        modified_df=modified_df,
        deleted_df=deleted_df,
        new_df=new_df,
        summary=summary,
        nearest_matches=nearest_matches,
    )

    logging.info(
        "Done. Modified=%s (relocated=%s, attr_changed=%s). Deleted=%s. New=%s. Unchanged=%s.",
        summary.modified_count,
        summary.relocated_count,
        summary.attr_changed_count,
        summary.deleted_count,
        summary.new_count,
        summary.unchanged_count,
    )
    return summary


# =============================================================================
# CLI (still supported; notebook ignores injected args)
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args and return (args, unknown_args)."""
    parser = argparse.ArgumentParser(description="Compare GTFS stops between two feeds.")
    parser.add_argument("--before", type=Path, default=BEFORE_GTFS_DIR, help="Before GTFS folder")
    parser.add_argument("--after", type=Path, default=AFTER_GTFS_DIR, help="After GTFS folder")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--threshold-feet",
        type=float,
        default=RELOCATE_THRESHOLD_FEET,
        help="Relocation threshold in feet",
    )
    parser.add_argument(
        "--no-route-context",
        action="store_true",
        help="Skip attaching routes-serving-stop context columns to outputs",
    )
    args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    return args, unknown


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (notebook-safe)."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if (
        BEFORE_GTFS_DIR == Path(r"Path\To\GTFS\Dir")
        or AFTER_GTFS_DIR == Path(r"Path\to\GTFS\Dir")
        or OUTPUT_DIR == Path(r"Path\To\Output\Dir")
    ):
        logging.warning(
            "BEFORE_GTFS_DIR and/or AFTER_GTFS_DIR and/or OUTPUT_DIR are still set to "
            "placeholder values. Please update them in the CONFIGURATION section before running."
        )
        return
    args, _unknown = parse_args(argv)
    run_compare(
        before_dir=args.before,
        after_dir=args.after,
        out_dir=args.out,
        threshold_feet=args.threshold_feet,
        include_route_context=not args.no_route_context,
    )
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
