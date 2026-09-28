"""GTFS stop proximity checker.

Reads GTFS stops.txt (and optionally stop_times.txt and trips.txt when the
direction-based filter is enabled) and flags stops closer than a configured
distance threshold (default 50 feet). Only stops/platforms (location_type blank
or 0) are checked; stations, entrances, generic nodes and boarding areas are
skipped.

Optionally excludes ("passes") stops whose stop_name contains any configured safe
words (e.g., "bay", "metro"). When enabled, any close pair where either stop is
"safe" is ignored entirely (i.e., not emitted to outputs).

Optionally excludes close stop pairs that appear to be "across the street" stops:
pairs that share at least one route and, on every shared route, are served in
opposite directions (e.g., one stop only served by dir 0 trips for route X, the
other only served by dir 1 trips for route X). Incomplete service data never
justifies suppression: a stop served by a trip missing from trips.txt or with a
blank route_id, or a shared route with a blank or invalid direction_id, keeps
the pair.

Outputs:
- CSV of close stop pairs (with distance + safe flags)
- CSV of per-stop summary (count of close neighbors)

Typical usage:
    Update the paths in the CONFIGURATION section and run from a shell or a
    Jupyter notebook.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, NamedTuple, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_DIR = Path(r"path\to\gtfs")  # folder containing GTFS .txt files
OUT_DIR = Path(r"path\to\output")  # output folder for CSVs

THRESHOLD_FEET = 50.0

# Safe-word handling: stops with these words/phrases in stop_name are exempted from pair output.
# Add/remove terms here. Matching is case-insensitive.
SAFE_WORDS: list[str] = [
    "bay",
    # "metro",
    # "station",
]
SAFE_WORD_MATCH_WHOLE_WORD = True  # True -> \bterm\b, False -> substring match
PASS_SAFE_STOPS = True  # if True, skip any close pair where either stop is "safe"

# Across-the-street handling (directional pairs on same route):
# If True, exclude pairs served exclusively by opposite directions on every route they share.
EXCLUDE_OPPOSITE_DIRECTION_SAME_ROUTE_PAIRS = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

EARTH_RADIUS_M = 6_371_000.0
FEET_PER_M = 3.280839895


def _meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters * FEET_PER_M


def _approx_xy_meters(
    lat_deg: pd.Series,
    lon_deg: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Project lat/lon to local x/y in meters using equirectangular approximation."""
    lat0_rad = math.radians(float(lat_deg.mean()))
    lon0_rad = math.radians(float(lon_deg.mean()))

    lat_rad = lat_deg.astype(float).map(math.radians)
    lon_rad = lon_deg.astype(float).map(math.radians)

    x_m = (lon_rad - lon0_rad) * math.cos(lat0_rad) * EARTH_RADIUS_M
    y_m = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x_m, y_m


def _euclid_feet(dx_m: float, dy_m: float) -> float:
    """Euclidean distance (feet) from delta meters."""
    return _meters_to_feet(math.hypot(dx_m, dy_m))


# =============================================================================
# GRID INDEXING (FAST NEIGHBOR SEARCH)
# =============================================================================


@dataclass(frozen=True)
class GridParams:
    """Parameters defining a grid used for near-neighbor lookup."""

    cell_size_m: float


def _grid_cell(x_m: float, y_m: float, cell_size_m: float) -> tuple[int, int]:
    """Return integer grid cell coordinates for a point."""
    return (int(math.floor(x_m / cell_size_m)), int(math.floor(y_m / cell_size_m)))


def _neighbor_cells(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    """Yield the 3x3 neighborhood (cell and its 8 adjacent cells)."""
    cx, cy = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (cx + dx, cy + dy)


# =============================================================================
# SAFE WORD HELPERS
# =============================================================================


def compile_safe_words_regex(words: list[str], whole_word: bool) -> re.Pattern[str]:
    """Compile a case-insensitive regex that matches any provided words/phrases."""
    cleaned = [w.strip() for w in words if w and w.strip()]
    if not cleaned:
        return re.compile(r"(?!)", flags=re.IGNORECASE)  # match nothing

    parts: list[str] = []
    for w in cleaned:
        esc = re.escape(w)
        parts.append(rf"\b{esc}\b" if whole_word else esc)

    return re.compile("|".join(parts), flags=re.IGNORECASE)


def add_safe_flag(df: pd.DataFrame, safe_words: list[str], whole_word: bool) -> pd.DataFrame:
    """Add is_safe_stop flag based on stop_name matching safe words/phrases."""
    rx = compile_safe_words_regex(safe_words, whole_word=whole_word)
    out = df.copy()
    out["is_safe_stop"] = out["stop_name"].astype(str).str.contains(rx, na=False)
    return out


# =============================================================================
# GTFS LOADING
# =============================================================================


def validate_gtfs_files_exist(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
) -> None:
    """Check that specific GTFS text files exist and log a warning if missing.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to check. If ``None``,
            a standard set of GTFS files is checked.
    """
    if not os.path.exists(gtfs_folder_path):
        logging.warning("The directory '%s' does not exist.", gtfs_folder_path)
        return

    if files is None:
        # Only check files this script actually reads. stop_times.txt and
        # trips.txt are consumed only when the direction filter is enabled,
        # but a missing stops.txt is always fatal.
        files = (
            "stops.txt",
            "stop_times.txt",
            "trips.txt",
        )

    for file_name in files:
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name)):
            logging.warning("Missing GTFS file: %s", file_name)


def load_stops(stops_txt: Path) -> pd.DataFrame:
    """Load the stops/platforms in GTFS stops.txt with minimal validation.

    Keeps location_type 0 or blank (an absent column counts as blank), the only
    kind stop_times.txt can reference. Other rows are dropped before coordinates
    are checked, because generic nodes and boarding areas may omit them. Only
    empty fields count as missing, so literal IDs such as "NA" are kept.
    """
    df = pd.read_csv(
        stops_txt,
        dtype={"stop_id": "string", "stop_name": "string", "location_type": "string"},
        keep_default_na=False,
        na_values=[""],
    )
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"stops.txt missing required columns: {missing}")

    if "location_type" in df.columns:
        is_stop = df["location_type"].fillna("").str.strip().isin(["", "0"])
        skipped = int((~is_stop).sum())
        if skipped:
            logging.info(
                "Skipped %s stops.txt rows that are not stops/platforms "
                "(location_type other than 0 or blank).",
                skipped,
            )
        df = df[is_stop]

    df = df[list(required)].copy()
    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")

    # between() is False for NaN and +/-inf, so this also rejects non-finite values.
    usable = (
        df["stop_id"].notna() & df["stop_lat"].between(-90, 90) & df["stop_lon"].between(-180, 180)
    )
    dropped = int((~usable).sum())
    if dropped:
        logging.warning("Dropped %s stops with missing/invalid lat/lon/stop_id.", dropped)
    df = df[usable].reset_index(drop=True)

    df["stop_name"] = df["stop_name"].fillna("")
    return df


class DirectionIndex(NamedTuple):
    """Per-stop route/direction evidence from stop_times.txt joined to trips.txt.

    The defaults are immutable empties, so default-built indexes never share
    mutable state.

    Attributes:
        directions: stop_id -> route_id -> valid direction_ids (0/1) observed.
            Every known route association is present, including one whose
            trips all lack a usable direction_id (its set is then empty), so
            such a route still counts as shared.
        unknown_direction: stop_id -> route_ids on which at least one trip
            serving the stop has a blank or invalid direction_id. Kept apart
            from ``directions`` so {0} plus an unknown trip never reads as
            exclusively {0}.
        unresolved_stops: stop_ids served by a trip whose route is unknown
            (trip_id missing from trips.txt, or a blank route_id).
    """

    directions: Mapping[str, Mapping[str, set[int]]] = MappingProxyType({})
    unknown_direction: Mapping[str, set[str]] = MappingProxyType({})
    unresolved_stops: frozenset[str] = frozenset()


def build_stop_route_direction_index(gtfs_dir: Path) -> DirectionIndex:
    """Build the per-stop route/direction evidence used by the direction filter.

    Uses stop_times.txt left-joined to trips.txt, so stop_times rows whose trip
    cannot be resolved are kept as evidence against suppression rather than
    dropped. This is used to suppress "opposite direction on same route" pairs.

    Returns:
        A DirectionIndex; empty (feature disabled) when stop_times.txt or
        trips.txt is missing, or trips.txt has no direction_id of 0 or 1.

    Behavior:
        - A blank direction_id (valid GTFS) or an invalid one (anything but 0 or
          1, which also logs a warning) records the trip's route as having an
          unknown direction at the stops it serves.
        - A stop_times.txt trip_id missing from trips.txt, or a trip with a
          blank route_id, marks the stops it serves as unresolved.
        - Only empty fields count as missing, so literal IDs such as "NA" are kept.
    """
    stop_times_path = gtfs_dir / "stop_times.txt"
    trips_path = gtfs_dir / "trips.txt"

    if not stop_times_path.exists() or not trips_path.exists():
        logging.warning("Missing stop_times.txt or trips.txt; direction-based filtering disabled.")
        return DirectionIndex()

    stop_times = pd.read_csv(
        stop_times_path,
        usecols=["trip_id", "stop_id"],
        dtype={"trip_id": "string", "stop_id": "string"},
        keep_default_na=False,
        na_values=[""],
    )

    trips = pd.read_csv(
        trips_path,
        dtype={"trip_id": "string", "route_id": "string"},
        keep_default_na=False,
        na_values=[""],
    )

    if "direction_id" not in trips.columns:
        logging.warning("trips.txt has no direction_id; direction-based filtering disabled.")
        return DirectionIndex()

    # A blank trip_id can't be referenced; dropping it also keeps the merge
    # below from pairing it with blank stop_times.trip_id values.
    trips = trips.loc[trips["trip_id"].notna(), ["trip_id", "route_id", "direction_id"]]
    direction = pd.to_numeric(trips["direction_id"], errors="coerce")
    has_direction = direction.isin([0, 1])
    invalid = trips["direction_id"].notna() & ~has_direction
    if invalid.any():
        logging.warning(
            "trips.txt has %s direction_id value(s) other than 0 or 1 (e.g. %s); "
            "treating them as unknown.",
            int(invalid.sum()),
            trips.loc[invalid, "direction_id"].drop_duplicates().head(3).tolist(),
        )
    if not has_direction.any():
        logging.warning(
            "trips.txt has no direction_id of 0 or 1; direction-based filtering disabled."
        )
        return DirectionIndex()
    trips = trips.assign(direction_id=direction.where(has_direction))

    st = stop_times.merge(trips, on="trip_id", how="left", indicator=True)
    st = st[st["stop_id"].notna()]
    trip_missing = st["_merge"].eq("left_only")
    route_blank = ~trip_missing & st["route_id"].isna()
    unresolved = trip_missing | route_blank
    unknown = ~unresolved & st["direction_id"].isna()

    for mask, level, message in (
        (
            trip_missing,
            logging.WARNING,
            "%s stop_times.txt trip_id(s) not found in trips.txt; direction-based "
            "suppression disabled for pairs involving the %s stop(s) they serve.",
        ),
        (
            route_blank,
            logging.WARNING,
            "%s trip(s) with a blank route_id; direction-based suppression disabled "
            "for pairs involving the %s stop(s) they serve.",
        ),
        (
            unknown,
            logging.INFO,
            "%s trip(s) with a blank or invalid direction_id; their routes cannot "
            "justify suppression at the %s stop(s) they serve.",
        ),
    ):
        if mask.any():
            logging.log(
                level,
                message,
                st.loc[mask, "trip_id"].nunique(dropna=False),
                st.loc[mask, "stop_id"].nunique(),
            )

    directions: dict[str, dict[str, set[int]]] = {}
    unknown_direction: dict[str, set[str]] = {}
    # Unique combos only
    resolved = st.loc[~unresolved, ["stop_id", "route_id", "direction_id"]].drop_duplicates()
    for stop_id, route_id, direction_id in resolved.itertuples(index=False, name=None):
        # Record every known route association, even one with no usable
        # direction, so it still counts as a shared route.
        dirs = directions.setdefault(str(stop_id), {}).setdefault(str(route_id), set())
        if pd.isna(direction_id):
            unknown_direction.setdefault(str(stop_id), set()).add(str(route_id))
        else:
            dirs.add(int(direction_id))

    unresolved_stops = frozenset(str(stop_id) for stop_id in st.loc[unresolved, "stop_id"].unique())
    return DirectionIndex(directions, unknown_direction, unresolved_stops)


def is_opposite_direction_pair_same_route(
    stop_a: str,
    stop_b: str,
    index: DirectionIndex,
) -> bool:
    """Return True if stops look like opposite-direction-only on every shared route.

    Heuristic (all must hold):
      1. Neither stop is served by a trip whose route is unresolved.
      2. The stops share at least one route.
      3. Every shared route has complete direction information at both stops.
      4. On every shared route, one stop is served only by {0} and the other
         only by {1}.

    If either stop lacks route/dir info, returns False.
    """
    # A trip whose route is unknown could run on any route, in either direction.
    if stop_a in index.unresolved_stops or stop_b in index.unresolved_stops:
        return False

    a = index.directions.get(stop_a)
    b = index.directions.get(stop_b)
    if not a or not b:
        return False

    # Includes routes whose trips all lack a usable direction_id (empty sets).
    shared_routes = set(a).intersection(b)
    if not shared_routes:
        return False

    a_unknown = index.unknown_direction.get(stop_a, set())
    b_unknown = index.unknown_direction.get(stop_b, set())
    for r in shared_routes:
        # Intentionally conservative: only suppress a pair when each stop is
        # served *exclusively* by the opposite direction on every shared route.
        # Stops that appear in both directions (terminals, loops, transfer
        # points), or have an unknown direction on a shared route, are NOT
        # filtered out, so legitimate missing-stop flags at those locations
        # are preserved.
        if r in a_unknown or r in b_unknown:
            return False
        if not ((a[r] == {0} and b[r] == {1}) or (a[r] == {1} and b[r] == {0})):
            return False

    return True


# =============================================================================
# CORE LOGIC
# =============================================================================

# Column order of the close-pairs output; also its header when no pair is found.
PAIR_COLUMNS: list[str] = [
    "stop_id_a",
    "stop_name_a",
    "is_safe_a",
    "stop_id_b",
    "stop_name_b",
    "is_safe_b",
    "distance_feet",
    "threshold_feet",
]


def find_close_stop_pairs(
    stops: pd.DataFrame,
    threshold_feet: float,
    pass_safe_stops: bool,
    exclude_opposite_direction_same_route_pairs: bool,
    stop_route_dir_index: DirectionIndex,
) -> pd.DataFrame:
    """Find stop pairs within the distance threshold, with optional filters."""
    if "is_safe_stop" not in stops.columns:
        raise ValueError("Expected stops to include 'is_safe_stop'. Call add_safe_flag() first.")
    if not math.isfinite(threshold_feet) or threshold_feet <= 0:
        raise ValueError(f"threshold_feet must be a finite number above 0; got {threshold_feet}.")

    threshold_m = threshold_feet / FEET_PER_M
    x_m, y_m = _approx_xy_meters(stops["stop_lat"], stops["stop_lon"])

    # Reset to a clean 0-based integer index so that the positional indices
    # stored in the grid correspond exactly to array positions below.
    work = stops.reset_index(drop=True).copy()
    work["x_m"] = x_m.to_numpy()
    work["y_m"] = y_m.to_numpy()

    # Pull columns into numpy arrays for O(1) positional access in the hot loop,
    # avoiding the per-lookup overhead of pandas .loc and the silent
    # label-vs-position mismatch that would occur on non-default-indexed input.
    x_arr = work["x_m"].to_numpy()
    y_arr = work["y_m"].to_numpy()
    is_safe_arr = work["is_safe_stop"].to_numpy()
    stop_id_arr = work["stop_id"].to_numpy()
    stop_name_arr = work["stop_name"].to_numpy()

    params = GridParams(cell_size_m=threshold_m)

    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (xv, yv) in enumerate(zip(x_arr, y_arr)):
        cell = _grid_cell(float(xv), float(yv), params.cell_size_m)
        grid.setdefault(cell, []).append(idx)

    rows: list[dict[str, object]] = []
    n = len(work)

    for i in range(n):
        xi = float(x_arr[i])
        yi = float(y_arr[i])
        cell_i = _grid_cell(xi, yi, params.cell_size_m)

        for nc in _neighbor_cells(cell_i):
            for j in grid.get(nc, []):
                if j <= i:
                    continue

                i_safe = bool(is_safe_arr[i])
                j_safe = bool(is_safe_arr[j])
                if pass_safe_stops and (i_safe or j_safe):
                    continue

                stop_id_a = str(stop_id_arr[i])
                stop_id_b = str(stop_id_arr[j])

                if exclude_opposite_direction_same_route_pairs and stop_route_dir_index.directions:
                    if is_opposite_direction_pair_same_route(
                        stop_id_a,
                        stop_id_b,
                        stop_route_dir_index,
                    ):
                        continue

                dx = float(x_arr[j]) - xi
                dy = float(y_arr[j]) - yi

                if abs(dx) > threshold_m or abs(dy) > threshold_m:
                    continue

                dist_ft = _euclid_feet(dx, dy)
                if dist_ft > threshold_feet:
                    continue

                rows.append(
                    {
                        "stop_id_a": stop_id_a,
                        "stop_name_a": stop_name_arr[i],
                        "is_safe_a": i_safe,
                        "stop_id_b": stop_id_b,
                        "stop_name_b": stop_name_arr[j],
                        "is_safe_b": j_safe,
                        "distance_feet": round(dist_ft, 2),
                        "threshold_feet": threshold_feet,
                    }
                )

    pairs = pd.DataFrame.from_records(rows, columns=PAIR_COLUMNS)
    if pairs.empty:
        return pairs

    return pairs.sort_values(["distance_feet"], ascending=[True]).reset_index(drop=True)


def summarize_by_stop(pairs: pd.DataFrame) -> pd.DataFrame:
    """Create per-stop counts of close neighbors (based on emitted pairs only)."""
    if pairs.empty:
        return pd.DataFrame(columns=["stop_id", "close_neighbor_pairs"])

    a = pairs[["stop_id_a"]].rename(columns={"stop_id_a": "stop_id"})
    b = pairs[["stop_id_b"]].rename(columns={"stop_id_b": "stop_id"})
    both = pd.concat([a, b], ignore_index=True)

    out = (
        both.groupby("stop_id", as_index=False)
        .size()
        .rename(columns={"size": "close_neighbor_pairs"})
        .sort_values(["close_neighbor_pairs"], ascending=[False])
        .reset_index(drop=True)
    )
    return out


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Run the stop proximity QC.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if GTFS_DIR == Path(r"path\to\gtfs") or OUT_DIR == Path(r"path\to\output"):
        logging.warning(
            "GTFS_DIR and/or OUT_DIR are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2

    validate_gtfs_files_exist(str(GTFS_DIR))

    stops_txt = GTFS_DIR / "stops.txt"
    if not stops_txt.exists():
        raise FileNotFoundError(f"Could not find stops.txt at: {stops_txt}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stops = load_stops(stops_txt)
    stops = add_safe_flag(stops, safe_words=SAFE_WORDS, whole_word=SAFE_WORD_MATCH_WHOLE_WORD)

    stop_route_dir_index = DirectionIndex()
    if EXCLUDE_OPPOSITE_DIRECTION_SAME_ROUTE_PAIRS:
        stop_route_dir_index = build_stop_route_direction_index(GTFS_DIR)

    pairs = find_close_stop_pairs(
        stops=stops,
        threshold_feet=float(THRESHOLD_FEET),
        pass_safe_stops=bool(PASS_SAFE_STOPS),
        exclude_opposite_direction_same_route_pairs=bool(
            EXCLUDE_OPPOSITE_DIRECTION_SAME_ROUTE_PAIRS
        ),
        stop_route_dir_index=stop_route_dir_index,
    )
    summary = summarize_by_stop(pairs)

    pairs_path = OUT_DIR / "close_stop_pairs.csv"
    summary_path = OUT_DIR / "close_stop_summary_by_stop.csv"

    pairs.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)

    logging.info("Stops loaded: %s", len(stops))
    logging.info("Close pairs found (after filtering): %s", 0 if pairs.empty else len(pairs))
    logging.info("Wrote: %s", pairs_path)
    logging.info("Wrote: %s", summary_path)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
