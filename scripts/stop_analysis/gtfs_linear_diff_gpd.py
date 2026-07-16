"""Map where a service change adds, keeps, or removes street coverage (linear GTFS diff).

A geometry-first companion to ``gtfs_stop_diff.py`` (points) and
``gtfs_route_diff.py`` (tables). Point it at two GTFS feeds (a *before* feed and
an *after* feed) and it overlays each route's alignment geometry to produce
shapefiles of **new**, **retained**, and **eliminated** running-way segments —
the layers behind a "where does the network gain/keep/lose service" map — and a
per-route log of which routes actually changed alignment. Routes whose alignment
is unchanged (schedule-only changes: trips added or dropped, times shifted,
headways revised) are explicitly excluded from the change log and reported as
such, because their geometry diffs to nothing.

Method
------
- Geometry never compares ``shape_id`` values (they are routinely rekeyed on
  every feed export) and never compares vertices exactly (shapes get
  re-digitized). Instead both feeds are projected to a metric CRS (auto-picked
  UTM unless ``TARGET_CRS`` is set) and each side is compared against a
  *buffered* copy of the other: an after-segment farther than
  ``BUFFER_TOLERANCE_FEET`` from the before alignment is **new**, an
  after-segment inside the buffer is **retained**, and a before-segment outside
  the buffered after alignment is **eliminated**.
- Overlay slivers shorter than ``MIN_SEGMENT_LENGTH_FEET`` are dropped, and a
  route is logged as having a linear change only when its new + eliminated
  length reaches ``LINEAR_CHANGE_MIN_FEET`` — so digitization jitter and a stop
  relocation nudging a shape do not read as reroutes.
- Renumbered routes are reconciled before diffing (same ``route_short_name``
  plus a served-stop-set Jaccard of at least ``REKEY_MIN_JACCARD``), matching
  ``gtfs_route_diff.py``, so a rekeyed route is diffed against its successor
  instead of being reported as an elimination plus an addition.
- Routes with no usable ``shapes.txt`` geometry fall back to stop-to-stop
  chords built from ``stop_times`` + ``stops`` (flagged in a ``geom_src``
  attribute) unless ``ALLOW_CHORD_FALLBACK`` is disabled.

Inputs
------
- Two GTFS folders, each with ``routes.txt``, ``trips.txt``, ``stop_times.txt``
  and, ideally, ``shapes.txt`` (``stops.txt`` is used for the chord fallback).

Outputs
-------
- ``linear_segments_new.shp`` / ``_retained.shp`` / ``_eliminated.shp``:
  per-route segment features (route_id, rt_short, chg_class, geom_src, lengths).
- ``linear_network_new.shp`` / ``_retained.shp`` / ``_eliminated.shp``
  (optional): the same three classes computed on the merged system network, so
  a street kept by *any* route counts as retained system-wide.
- ``routes_linear_changes.csv``: one row per route — status, lengths, changed
  share, and whether the route has a linear change or was excluded as
  schedule-only/unchanged.
- ``linear_diff_summary.json``: run-level counts and mileage totals.
- A run-log sidecar capturing the verbatim CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass ``--before``,
``--after``, ``--output-dir`` and the threshold flags) and run from a shell or
a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

# Sentinel markers used by extract_config_block / write_run_log to identify the
# configuration block within this file's source. Each string must appear exactly
# once in this file as a stand-alone comment line. Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# GTFS folders to compare (each contains routes.txt, trips.txt, stop_times.txt, ...).
BEFORE_GTFS_DIR: str = r"Path\To\Your\GTFS_Before_Folder"
AFTER_GTFS_DIR: str = r"Path\To\Your\GTFS_After_Folder"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

# Match tolerance between the two alignments. Geometry within this distance of
# the other feed's alignment counts as the same street; beyond it counts as a
# reroute. ~80 ft (about 25 m) absorbs re-digitized shapes and one-way couplet
# jitter without swallowing a move to a parallel street.
BUFFER_TOLERANCE_FEET: float = 80.0

# Overlay slivers shorter than this are dropped from every segment layer
# (crossing artifacts where an old and new alignment intersect).
MIN_SEGMENT_LENGTH_FEET: float = 150.0

# A route is logged as having a linear change only when its new + eliminated
# length reaches this. Below it, the route is reported as unchanged-alignment
# (i.e. schedule-only, from this tool's point of view).
LINEAR_CHANGE_MIN_FEET: float = 500.0

# Route-rekey reconciliation: a before-only route_id is matched to an
# after-only route_id with the same route_short_name when their served-stop-set
# Jaccard is at least this (mirrors gtfs_route_diff.py).
REKEY_MIN_JACCARD: float = 0.50

# When a route has no usable shapes.txt geometry, build stop-to-stop chord
# lines from stop_times + stops instead of skipping the route. Chord-derived
# features carry geom_src = "stops_chord" so map readers can discount them.
ALLOW_CHORD_FALLBACK: bool = True

# Also write the three system-level layers (before/after networks merged across
# routes). A street is only "eliminated" system-wide when no after route runs
# within the buffer tolerance — per-route layers cannot tell you that.
WRITE_NETWORK_LAYERS: bool = True

# Metric CRS for buffering/length math, e.g. "EPSG:26918". Empty = auto-pick a
# UTM zone from the feeds' extent. Output shapefiles are always written in
# WGS 84 (EPSG:4326), the GTFS-native CRS.
TARGET_CRS: str = ""

# Unit for the *_mi length columns ("miles" or "km").
DISTANCE_OUTPUT_UNIT: Literal["miles", "km"] = "miles"

# Output filenames (all written inside OUTPUT_DIR).
SEGMENTS_NEW_FILENAME: str = r"linear_segments_new.shp"
SEGMENTS_RETAINED_FILENAME: str = r"linear_segments_retained.shp"
SEGMENTS_ELIMINATED_FILENAME: str = r"linear_segments_eliminated.shp"
NETWORK_NEW_FILENAME: str = r"linear_network_new.shp"
NETWORK_RETAINED_FILENAME: str = r"linear_network_retained.shp"
NETWORK_ELIMINATED_FILENAME: str = r"linear_network_eliminated.shp"
ROUTE_LOG_FILENAME: str = r"routes_linear_changes.csv"
SUMMARY_FILENAME: str = r"linear_diff_summary.json"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output is never left
# without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

FEET_PER_METER: float = 3.280839895013123

_REQUIRED_FILES: tuple[str, ...] = ("routes.txt", "trips.txt", "stop_times.txt")
_OPTIONAL_FILES: tuple[str, ...] = ("shapes.txt", "stops.txt")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Runtime configuration for one linear-diff run."""

    before_dir: Path
    after_dir: Path
    output_dir: Path
    buffer_tolerance_feet: float = BUFFER_TOLERANCE_FEET
    min_segment_length_feet: float = MIN_SEGMENT_LENGTH_FEET
    linear_change_min_feet: float = LINEAR_CHANGE_MIN_FEET
    rekey_min_jaccard: float = REKEY_MIN_JACCARD
    allow_chord_fallback: bool = ALLOW_CHORD_FALLBACK
    write_network_layers: bool = WRITE_NETWORK_LAYERS
    target_crs: str = TARGET_CRS
    distance_output_unit: Literal["miles", "km"] = DISTANCE_OUTPUT_UNIT


@dataclass(frozen=True)
class RouteGeometry:
    """One route's alignment geometry within a single feed."""

    route_id: str
    geometry: Optional[BaseGeometry]  # metric CRS once projected; None = no geometry
    source: str  # "shapes" / "stops_chord" / "none"


@dataclass(frozen=True)
class Correspondence:
    """Resolved route correspondence between the two feeds."""

    matched: list[str]  # route_ids present (by id) in both feeds
    rekeyed: dict[str, str]  # before_id -> after_id (same route, new route_id)
    eliminated: list[str]  # before-only route_ids (true eliminations)
    added: list[str]  # after-only route_ids (true additions)


@dataclass
class RouteLinearResult:
    """Per-route linear-diff outcome (one row of routes_linear_changes.csv)."""

    route_id_before: str
    route_id_after: str
    route_label: str
    status: str  # matched / rekeyed / added / eliminated
    geom_source_before: str
    geom_source_after: str
    before_len_ft: float
    after_len_ft: float
    new_len_ft: float
    retained_len_ft: float
    eliminated_len_ft: float
    changed_len_ft: float
    changed_share: float
    change_kind: str  # realigned / added / eliminated / unchanged_alignment / no_geometry
    has_linear_change: bool
    note: str = ""


@dataclass
class SegmentRecord:
    """One exportable segment feature."""

    route_id: str
    route_label: str
    change_class: str  # new / retained / eliminated
    geom_source: str
    geometry: BaseGeometry


@dataclass(frozen=True)
class Summary:
    """Run-level metrics for the linear diff."""

    before_route_count: int
    after_route_count: int
    matched_count: int
    rekeyed_count: int
    added_count: int
    eliminated_count: int
    realigned_count: int
    unchanged_alignment_count: int
    no_geometry_count: int
    chord_fallback_route_count: int
    new_len_mi: float
    retained_len_mi: float
    eliminated_len_mi: float
    buffer_tolerance_feet: float
    min_segment_length_feet: float
    linear_change_min_feet: float
    metric_crs: str


# =============================================================================
# IO HELPERS
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


def load_feed(gtfs_dir: Path, label: str) -> dict[str, pd.DataFrame]:
    """Load the GTFS files needed for a linear diff, tolerating absent optional files.

    Args:
        gtfs_dir: Folder containing the GTFS feed.
        label: Human-readable feed label used in log messages (e.g. ``"before"``).

    Returns:
        Mapping of file stem -> DataFrame for every file that was present.

    Raises:
        OSError: ``gtfs_dir`` is missing or a required file is absent.
    """
    if not os.path.isdir(gtfs_dir):
        raise OSError(f"{label}: directory '{gtfs_dir}' does not exist.")

    missing_required = [
        name for name in _REQUIRED_FILES if not os.path.exists(os.path.join(gtfs_dir, name))
    ]
    if missing_required:
        raise OSError(f"{label}: missing required GTFS files: {', '.join(missing_required)}")

    feed: dict[str, pd.DataFrame] = {}
    trip_cols = {"route_id", "trip_id", "shape_id"}
    st_cols = {"trip_id", "stop_id", "stop_sequence"}
    shape_cols = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    stop_cols = {"stop_id", "stop_lat", "stop_lon"}

    for name in _REQUIRED_FILES + _OPTIONAL_FILES:
        path = Path(gtfs_dir) / name
        if not path.exists():
            continue
        key = name.replace(".txt", "")
        if name == "trips.txt":
            frame = _read_csv(path, usecols=lambda c: c in trip_cols)
        elif name == "stop_times.txt":
            frame = _read_csv(path, usecols=lambda c: c in st_cols)
        elif name == "shapes.txt":
            frame = _read_csv(path, usecols=lambda c: c in shape_cols)
        elif name == "stops.txt":
            frame = _read_csv(path, usecols=lambda c: c in stop_cols)
        else:
            frame = _read_csv(path)
        feed[key] = frame
        logging.info("%s: loaded %s (%d records).", label, name, len(frame))

    if "shape_id" not in feed["trips"].columns:
        feed["trips"]["shape_id"] = ""

    if "shapes" not in feed:
        logging.warning(
            "%s: shapes.txt not found; route geometry will rely on the stop-chord fallback.",
            label,
        )
    return feed


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize a text column for comparisons (fill NA, cast to str, strip)."""
    return series.fillna("").astype(str).str.strip()


def _route_display_label(row: pd.Series) -> str:
    """Best human-readable label for a route: short name, else long name, else ID."""
    short = str(row.get("route_short_name") or "").strip()
    if short and short.lower() != "nan":
        return short
    long_name = str(row.get("route_long_name") or "").strip()
    if long_name and long_name.lower() != "nan":
        return long_name
    return str(row.get("route_id") or "").strip()


# =============================================================================
# ROUTE GEOMETRY (shapes.txt, with stop-chord fallback)
# =============================================================================


def build_shape_geometries(shapes: Optional[pd.DataFrame], label: str) -> dict[str, LineString]:
    """Build one LineString per shape_id from a ``shapes.txt`` table.

    Points are ordered by numeric ``shape_pt_sequence``; rows with unparseable
    coordinates or sequence values are dropped, and shapes left with fewer than
    two points are skipped (with a warning).

    Args:
        shapes: Parsed ``shapes.txt`` (or None when the file is absent).
        label: Feed label for log messages.

    Returns:
        Mapping of shape_id -> LineString.
    """
    if shapes is None or shapes.empty:
        return {}
    required = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    missing = sorted(required - set(shapes.columns))
    if missing:
        logging.warning("%s: shapes.txt missing columns %s; ignoring shapes.", label, missing)
        return {}

    df = shapes.copy()
    df["shape_id"] = normalize_text(df["shape_id"])
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
    df = df.sort_values(["shape_id", "shape_pt_sequence"])

    out: dict[str, LineString] = {}
    skipped = 0
    for shape_id, grp in df.groupby("shape_id"):
        coords = list(zip(grp["shape_pt_lon"].astype(float), grp["shape_pt_lat"].astype(float)))
        if len(coords) < 2:
            skipped += 1
            continue
        out[str(shape_id)] = LineString(coords)
    if skipped:
        logging.warning("%s: skipped %d shape(s) with fewer than 2 valid points.", label, skipped)
    return out


def build_chord_lines(
    trip_ids: frozenset[str],
    stop_times: pd.DataFrame,
    stop_xy: dict[str, tuple[float, float]],
) -> list[LineString]:
    """Build straight stop-to-stop chord lines for a route's distinct trip patterns.

    Each unique ordered stop_id sequence among ``trip_ids`` becomes one
    LineString through the stop coordinates. This is the fallback used when a
    route has no usable ``shapes.txt`` geometry.

    Args:
        trip_ids: The route's trip_ids.
        stop_times: ``stop_times`` rows (trip_id, stop_id, stop_sequence as str).
        stop_xy: Mapping of stop_id -> (lon, lat).

    Returns:
        One LineString per distinct stop pattern with at least two locatable stops.
    """
    st = stop_times.loc[stop_times["trip_id"].isin(trip_ids)].copy()
    if st.empty:
        return []
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])

    patterns: set[tuple[str, ...]] = set()
    for _, grp in st.groupby("trip_id"):
        patterns.add(tuple(grp["stop_id"].astype(str)))

    lines: list[LineString] = []
    for pattern in sorted(patterns):
        coords = [stop_xy[sid] for sid in pattern if sid in stop_xy]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def build_route_geometries(
    feed: dict[str, pd.DataFrame],
    label: str,
    allow_chord_fallback: bool,
) -> dict[str, RouteGeometry]:
    """Assemble each route's alignment geometry (WGS 84) for one feed.

    Every trip's shape contributes, so branches, short-turns, and both
    directions are all part of a route's geometry. Routes with no usable shape
    fall back to stop-to-stop chords when ``allow_chord_fallback`` is True;
    otherwise (or when stops are also unusable) the route carries no geometry.

    Args:
        feed: Loaded feed tables (see :func:`load_feed`).
        label: Feed label for log messages.
        allow_chord_fallback: Whether to build chord lines for shapeless routes.

    Returns:
        Mapping of route_id -> :class:`RouteGeometry` (geometry in EPSG:4326).
    """
    trips = feed["trips"].copy()
    trips["route_id"] = normalize_text(trips["route_id"])
    trips["trip_id"] = normalize_text(trips["trip_id"])
    trips["shape_id"] = normalize_text(trips["shape_id"])

    shape_geoms = build_shape_geometries(feed.get("shapes"), label)

    stop_xy: dict[str, tuple[float, float]] = {}
    stops = feed.get("stops")
    if stops is not None and {"stop_id", "stop_lat", "stop_lon"} <= set(stops.columns):
        lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
        lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
        for sid, x, y in zip(normalize_text(stops["stop_id"]), lon, lat):
            if pd.notna(x) and pd.notna(y):
                stop_xy[str(sid)] = (float(x), float(y))

    out: dict[str, RouteGeometry] = {}
    chord_count = 0
    for route_id, grp in trips.groupby("route_id"):
        rid = str(route_id)
        shape_ids = sorted({s for s in grp["shape_id"] if s})
        lines: list[LineString] = [shape_geoms[s] for s in shape_ids if s in shape_geoms]
        source = "shapes"
        if not lines and allow_chord_fallback:
            lines = build_chord_lines(frozenset(grp["trip_id"]), feed["stop_times"], stop_xy)
            source = "stops_chord"
            if lines:
                chord_count += 1
        if not lines:
            out[rid] = RouteGeometry(route_id=rid, geometry=None, source="none")
            continue
        out[rid] = RouteGeometry(route_id=rid, geometry=unary_union(lines), source=source)

    no_geom = sum(1 for rg in out.values() if rg.geometry is None)
    logging.info(
        "%s: built geometry for %d route(s) (%d via stop-chord fallback, %d without geometry).",
        label,
        len(out) - no_geom,
        chord_count,
        no_geom,
    )
    return out


# =============================================================================
# ROUTE CORRESPONDENCE / REKEY MATCHING (mirrors gtfs_route_diff.py)
# =============================================================================


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


def _routes_indexed(routes: pd.DataFrame) -> pd.DataFrame:
    """Return routes keyed on a normalized string route_id, first row wins on dupes."""
    df = routes.copy()
    df["route_id"] = normalize_text(df["route_id"])
    return df.drop_duplicates(subset="route_id", keep="first").set_index("route_id")


def build_correspondence(
    routes_before: pd.DataFrame,
    routes_after: pd.DataFrame,
    stops_before: dict[str, frozenset[str]],
    stops_after: dict[str, frozenset[str]],
    rekey_min_jaccard: float,
) -> Correspondence:
    """Resolve which routes match, were rekeyed, eliminated, or added.

    Routes match first on ``route_id``. Each remaining before-only id is matched to an
    after-only id with the same ``route_short_name`` and the highest stop-set Jaccard,
    provided that Jaccard is at least ``rekey_min_jaccard``; such a pair is a rekey.
    Whatever stays unmatched is a true elimination or addition.
    """
    before_ids = set(normalize_text(routes_before["route_id"]))
    after_ids = set(normalize_text(routes_after["route_id"]))
    matched = sorted(before_ids & after_ids)

    before_only = before_ids - after_ids
    after_only = after_ids - before_ids

    short_before = _routes_indexed(routes_before).get("route_short_name")
    short_after = _routes_indexed(routes_after).get("route_short_name")

    def _short(table: Optional[pd.Series], rid: str) -> str:
        if table is None or rid not in table.index:
            return ""
        return str(table.loc[rid] or "").strip()

    rekeyed: dict[str, str] = {}
    claimed_after: set[str] = set()
    for b_id in sorted(before_only):
        b_short = _short(short_before, b_id)
        best_id, best_j = None, rekey_min_jaccard
        for a_id in sorted(after_only):
            if a_id in claimed_after or _short(short_after, a_id) != b_short:
                continue
            j = jaccard(stops_before.get(b_id, frozenset()), stops_after.get(a_id, frozenset()))
            if j >= best_j:
                best_id, best_j = a_id, j
        if best_id is not None:
            rekeyed[b_id] = best_id
            claimed_after.add(best_id)

    eliminated = sorted(before_only - set(rekeyed))
    added = sorted(after_only - claimed_after)
    return Correspondence(matched=matched, rekeyed=rekeyed, eliminated=eliminated, added=added)


# =============================================================================
# LINEAR OVERLAY
# =============================================================================


def extract_lines(geom: BaseGeometry) -> list[LineString]:
    """Flatten a geometry to its LineString parts, merging touching pieces.

    Points and polygons that an intersection/difference can emit (e.g. where an
    old and a new alignment cross at a single vertex) are discarded — only
    linear geometry is meaningful in these layers.

    Args:
        geom: Any shapely geometry.

    Returns:
        A list of LineStrings (possibly empty).
    """
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return [merged]
        return list(merged.geoms)
    if isinstance(geom, GeometryCollection):
        parts: list[LineString] = []
        for part in geom.geoms:
            parts.extend(extract_lines(part))
        if len(parts) > 1:
            return extract_lines(MultiLineString(parts))
        return parts
    return []


def filter_short_lines(lines: Sequence[LineString], min_length_m: float) -> list[LineString]:
    """Drop overlay slivers: lines shorter than ``min_length_m`` (metric CRS)."""
    return [line for line in lines if line.length >= min_length_m]


def diff_alignments(
    before_geom: BaseGeometry,
    after_geom: BaseGeometry,
    buffer_m: float,
    min_segment_m: float,
) -> dict[str, list[LineString]]:
    """Overlay two alignments (metric CRS) into new / retained / eliminated lines.

    ``after`` geometry outside a ``buffer_m`` buffer of ``before`` is **new**;
    ``after`` geometry inside it is **retained**; ``before`` geometry outside a
    buffered ``after`` is **eliminated**. Slivers shorter than
    ``min_segment_m`` are dropped from every class.

    Args:
        before_geom: Before alignment (LineString/MultiLineString, metric CRS).
        after_geom: After alignment (same CRS).
        buffer_m: Match tolerance in meters.
        min_segment_m: Minimum surviving segment length in meters.

    Returns:
        ``{"new": [...], "retained": [...], "eliminated": [...]}``.
    """
    before_buffer = before_geom.buffer(buffer_m)
    after_buffer = after_geom.buffer(buffer_m)
    return {
        "new": filter_short_lines(
            extract_lines(after_geom.difference(before_buffer)), min_segment_m
        ),
        "retained": filter_short_lines(
            extract_lines(after_geom.intersection(before_buffer)), min_segment_m
        ),
        "eliminated": filter_short_lines(
            extract_lines(before_geom.difference(after_buffer)), min_segment_m
        ),
    }


def total_length_ft(lines: Sequence[LineString]) -> float:
    """Sum the length of metric-CRS lines, in feet."""
    return float(sum(line.length for line in lines)) * FEET_PER_METER


def resolve_metric_crs(
    geometries: Sequence[BaseGeometry],
    target_crs: str,
) -> Any:
    """Pick the projected CRS used for buffering and length math.

    Args:
        geometries: WGS 84 geometries from both feeds (used for auto-UTM).
        target_crs: Explicit CRS string from the CONFIGURATION block, or ""
            to estimate a UTM zone from the geometries' extent.

    Returns:
        A pyproj CRS accepted by ``GeoSeries.to_crs``.

    Raises:
        ValueError: No geometry is available to estimate a CRS from.
    """
    if target_crs.strip():
        return target_crs.strip()
    if not geometries:
        raise ValueError("No route geometry available to estimate a metric CRS from.")
    return gpd.GeoSeries(list(geometries), crs="EPSG:4326").estimate_utm_crs()


# =============================================================================
# DIFF DRIVER
# =============================================================================


def _project_route_geometries(
    geoms: dict[str, RouteGeometry], metric_crs: Any
) -> dict[str, RouteGeometry]:
    """Reproject every route geometry from WGS 84 to the metric CRS."""
    ids = [rid for rid, rg in geoms.items() if rg.geometry is not None]
    if not ids:
        return dict(geoms)
    series = gpd.GeoSeries([geoms[rid].geometry for rid in ids], crs="EPSG:4326")
    projected = series.to_crs(metric_crs)
    out = dict(geoms)
    for rid, geom in zip(ids, projected):
        out[rid] = RouteGeometry(route_id=rid, geometry=geom, source=geoms[rid].source)
    return out


def _route_labels(routes: pd.DataFrame) -> dict[str, str]:
    """Map each normalized route_id to its display label."""
    indexed = _routes_indexed(routes)
    return {str(rid): _route_display_label(row) for rid, row in indexed.iterrows()}


def _whole_route_result(
    rid: str,
    label: str,
    geom: RouteGeometry,
    status: str,
    change_min_ft: float,
) -> tuple[RouteLinearResult, list[SegmentRecord]]:
    """Build the result and segments for a wholly added or eliminated route.

    The full alignment goes into a single class (new for additions, eliminated
    for eliminations); the sliver filter is not applied because whole-route
    geometry contains no overlay artifacts.
    """
    is_added = status == "added"
    lines = extract_lines(geom.geometry) if geom.geometry is not None else []
    length_ft = total_length_ft(lines)
    change_class = "new" if is_added else "eliminated"
    segments = [
        SegmentRecord(
            route_id=rid,
            route_label=label,
            change_class=change_class,
            geom_source=geom.source,
            geometry=line,
        )
        for line in lines
    ]
    if not lines:
        kind, has_change, note = "no_geometry", False, "route has no usable geometry"
    else:
        kind = "added" if is_added else "eliminated"
        has_change = length_ft >= change_min_ft
        note = "" if has_change else "below linear-change threshold"
    result = RouteLinearResult(
        route_id_before="" if is_added else rid,
        route_id_after=rid if is_added else "",
        route_label=label,
        status=status,
        geom_source_before="" if is_added else geom.source,
        geom_source_after=geom.source if is_added else "",
        before_len_ft=0.0 if is_added else length_ft,
        after_len_ft=length_ft if is_added else 0.0,
        new_len_ft=length_ft if is_added else 0.0,
        retained_len_ft=0.0,
        eliminated_len_ft=0.0 if is_added else length_ft,
        changed_len_ft=length_ft,
        changed_share=1.0 if lines else 0.0,
        change_kind=kind,
        has_linear_change=has_change,
        note=note,
    )
    return result, segments


def _paired_route_result(
    b_id: str,
    a_id: str,
    label: str,
    status: str,
    geom_b: RouteGeometry,
    geom_a: RouteGeometry,
    cfg: Config,
) -> tuple[RouteLinearResult, list[SegmentRecord]]:
    """Diff one corresponding route pair and build its result + segment records."""
    if geom_b.geometry is None or geom_a.geometry is None:
        side = "before" if geom_b.geometry is None else "after"
        result = RouteLinearResult(
            route_id_before=b_id,
            route_id_after=a_id,
            route_label=label,
            status=status,
            geom_source_before=geom_b.source,
            geom_source_after=geom_a.source,
            before_len_ft=0.0,
            after_len_ft=0.0,
            new_len_ft=0.0,
            retained_len_ft=0.0,
            eliminated_len_ft=0.0,
            changed_len_ft=0.0,
            changed_share=0.0,
            change_kind="no_geometry",
            has_linear_change=False,
            note=f"no usable geometry in {side} feed; route skipped",
        )
        return result, []

    buffer_m = cfg.buffer_tolerance_feet / FEET_PER_METER
    min_segment_m = cfg.min_segment_length_feet / FEET_PER_METER
    classes = diff_alignments(geom_b.geometry, geom_a.geometry, buffer_m, min_segment_m)

    before_len = total_length_ft(extract_lines(geom_b.geometry))
    after_len = total_length_ft(extract_lines(geom_a.geometry))
    new_len = total_length_ft(classes["new"])
    retained_len = total_length_ft(classes["retained"])
    eliminated_len = total_length_ft(classes["eliminated"])
    changed_len = new_len + eliminated_len
    denom = max(before_len, after_len)
    changed_share = (changed_len / denom) if denom > 0 else 0.0

    has_change = changed_len >= cfg.linear_change_min_feet
    kind = "realigned" if has_change else "unchanged_alignment"
    note = "" if has_change else "alignment unchanged (schedule-only or no change)"

    segments = [
        SegmentRecord(
            route_id=a_id,
            route_label=label,
            change_class=change_class,
            geom_source=geom_a.source if change_class != "eliminated" else geom_b.source,
            geometry=line,
        )
        for change_class, lines in classes.items()
        for line in lines
    ]
    result = RouteLinearResult(
        route_id_before=b_id,
        route_id_after=a_id,
        route_label=label,
        status=status,
        geom_source_before=geom_b.source,
        geom_source_after=geom_a.source,
        before_len_ft=before_len,
        after_len_ft=after_len,
        new_len_ft=new_len,
        retained_len_ft=retained_len,
        eliminated_len_ft=eliminated_len,
        changed_len_ft=changed_len,
        changed_share=changed_share,
        change_kind=kind,
        has_linear_change=has_change,
        note=note,
    )
    return result, segments


def diff_network(
    geoms_before: dict[str, RouteGeometry],
    geoms_after: dict[str, RouteGeometry],
    cfg: Config,
) -> dict[str, list[LineString]]:
    """Overlay the merged before/after networks (all routes unioned per feed).

    Unlike the per-route layers, a street only lands in the network-level
    ``eliminated`` class when *no* after route runs within the buffer tolerance
    — the system-wide "where did service disappear" view.

    Args:
        geoms_before: Projected before-feed route geometries.
        geoms_after: Projected after-feed route geometries.
        cfg: Run configuration (tolerances).

    Returns:
        ``{"new": [...], "retained": [...], "eliminated": [...]}`` (metric CRS).
    """
    before_parts = [rg.geometry for rg in geoms_before.values() if rg.geometry is not None]
    after_parts = [rg.geometry for rg in geoms_after.values() if rg.geometry is not None]
    if not before_parts or not after_parts:
        return {"new": [], "retained": [], "eliminated": []}
    return diff_alignments(
        unary_union(before_parts),
        unary_union(after_parts),
        cfg.buffer_tolerance_feet / FEET_PER_METER,
        cfg.min_segment_length_feet / FEET_PER_METER,
    )


# =============================================================================
# EXPORT
# =============================================================================


def convert_distance(
    value: Any,
    input_unit: str,
    output_unit: Literal["miles", "km"] = "miles",
) -> Optional[float]:
    """Convert a distance value between transit-planning units.

    Args:
        value: Distance as a number or numeric string. ``None``, NaN, and
            empty/whitespace strings yield ``None``.
        input_unit: Unit of *value*: ``"feet"``, ``"meters"``, ``"km"``, or
            ``"miles"`` (case-insensitive).
        output_unit: Unit to convert to: ``"miles"`` or ``"km"``.

    Returns:
        The converted distance as a float, or ``None`` when *value* is
        missing or cannot be interpreted as a number.

    Raises:
        ValueError: If *input_unit* or *output_unit* is not a supported unit.
    """
    meters_per_input_unit = {"feet": 0.3048, "meters": 1.0, "km": 1000.0, "miles": 1609.344}
    meters_per_output_unit = {"miles": 1609.344, "km": 1000.0}

    input_factor = meters_per_input_unit.get(str(input_unit).strip().lower())
    if input_factor is None:
        raise ValueError(
            f"Unsupported input_unit {input_unit!r}; "
            f"expected one of {sorted(meters_per_input_unit)}."
        )
    output_factor = meters_per_output_unit.get(str(output_unit).strip().lower())
    if output_factor is None:
        raise ValueError(
            f"Unsupported output_unit {output_unit!r}; "
            f"expected one of {sorted(meters_per_output_unit)}."
        )

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric * input_factor / output_factor


def write_segment_shapefile(
    segments: Sequence[SegmentRecord],
    path: Path,
    metric_crs: Any,
    output_unit: Literal["miles", "km"],
) -> bool:
    """Write per-route segment records to a shapefile (WGS 84).

    Field names are kept within the shapefile 10-character limit:
    ``route_id``, ``rt_label``, ``chg_class``, ``geom_src``, ``len_ft``,
    ``len_mi`` (which holds km when ``output_unit="km"``).

    Args:
        segments: Segment records in the metric CRS.
        path: Destination ``.shp`` path.
        metric_crs: CRS the segment geometries are currently in.
        output_unit: Unit for the ``len_mi`` field.

    Returns:
        True when a file was written; False when there was nothing to write.
    """
    if not segments:
        logging.info("No features for %s; file not written.", path.name)
        return False
    gdf = gpd.GeoDataFrame(
        {
            "route_id": [s.route_id for s in segments],
            "rt_label": [s.route_label for s in segments],
            "chg_class": [s.change_class for s in segments],
            "geom_src": [s.geom_source for s in segments],
            "len_ft": [round(s.geometry.length * FEET_PER_METER, 1) for s in segments],
        },
        geometry=[s.geometry for s in segments],
        crs=metric_crs,
    )
    gdf["len_mi"] = [
        round(v, 3) if (v := convert_distance(ft, "feet", output_unit)) is not None else None
        for ft in gdf["len_ft"]
    ]
    gdf.to_crs("EPSG:4326").to_file(path)
    logging.info("Wrote %d feature(s): %s", len(gdf), path)
    return True


def write_network_shapefiles(
    network_classes: dict[str, list[LineString]],
    cfg: Config,
    metric_crs: Any,
) -> None:
    """Write the three system-level network layers (new/retained/eliminated)."""
    filenames = {
        "new": NETWORK_NEW_FILENAME,
        "retained": NETWORK_RETAINED_FILENAME,
        "eliminated": NETWORK_ELIMINATED_FILENAME,
    }
    for change_class, lines in network_classes.items():
        records = [
            SegmentRecord(
                route_id="(network)",
                route_label="(network)",
                change_class=change_class,
                geom_source="network",
                geometry=line,
            )
            for line in lines
        ]
        write_segment_shapefile(
            records,
            cfg.output_dir / filenames[change_class],
            metric_crs,
            cfg.distance_output_unit,
        )


def route_results_frame(
    results: Sequence[RouteLinearResult],
    output_unit: Literal["miles", "km"],
) -> pd.DataFrame:
    """Assemble the per-route results into the routes_linear_changes table."""
    df = pd.DataFrame([asdict(r) for r in results])
    if df.empty:
        return df
    for col in (
        "before_len_ft",
        "after_len_ft",
        "new_len_ft",
        "retained_len_ft",
        "eliminated_len_ft",
        "changed_len_ft",
    ):
        unit_col = col.replace("_ft", "_mi")
        df[unit_col] = [convert_distance(v, "feet", output_unit) for v in df[col]]
        df[col] = df[col].round(1)
        df[unit_col] = df[unit_col].round(3)
    df["changed_share"] = df["changed_share"].round(4)
    order = {
        "added": 0,
        "eliminated": 1,
        "realigned": 2,
        "unchanged_alignment": 3,
        "no_geometry": 4,
    }
    df["_order"] = df["change_kind"].map(order).fillna(5)
    df = df.sort_values(["_order", "route_label"]).drop(columns="_order")
    return df.reset_index(drop=True)


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


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


def write_run_log(output_dir: Path, summary_lines: List[str]) -> bool:
    """Write the verbatim config block plus a build summary into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "gtfs_linear_diff_runlog.txt"

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

    lines: List[str] = [
        "=" * 72,
        "GTFS LINEAR DIFF RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "BUILD SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
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
# PIPELINE
# =============================================================================


def run(cfg: Config) -> Summary:
    """Execute the linear diff end-to-end and write all artifacts.

    Args:
        cfg: Resolved configuration.

    Returns:
        Run-level :class:`Summary` (also written to ``linear_diff_summary.json``).

    Raises:
        RuntimeError: The run-log sidecar could not be written and
            ``REQUIRE_RUN_LOG`` is True.
    """
    before = load_feed(cfg.before_dir, label="before")
    after = load_feed(cfg.after_dir, label="after")

    geoms_before = build_route_geometries(before, "before", cfg.allow_chord_fallback)
    geoms_after = build_route_geometries(after, "after", cfg.allow_chord_fallback)

    corr = build_correspondence(
        before["routes"],
        after["routes"],
        route_stop_sets(before["trips"], before["stop_times"]),
        route_stop_sets(after["trips"], after["stop_times"]),
        cfg.rekey_min_jaccard,
    )
    logging.info(
        "Routes: before=%d after=%d | matched=%d rekeyed=%d eliminated=%d added=%d",
        len(geoms_before),
        len(geoms_after),
        len(corr.matched),
        len(corr.rekeyed),
        len(corr.eliminated),
        len(corr.added),
    )
    for b_id, a_id in corr.rekeyed.items():
        logging.info("Rekeyed route matched across feeds: %s -> %s", b_id, a_id)

    all_geoms = [
        rg.geometry
        for rg in list(geoms_before.values()) + list(geoms_after.values())
        if rg.geometry is not None
    ]
    metric_crs = resolve_metric_crs(all_geoms, cfg.target_crs)
    logging.info("Metric CRS for overlay/lengths: %s", metric_crs)
    geoms_before = _project_route_geometries(geoms_before, metric_crs)
    geoms_after = _project_route_geometries(geoms_after, metric_crs)

    labels_before = _route_labels(before["routes"])
    labels_after = _route_labels(after["routes"])
    no_geom = RouteGeometry(route_id="", geometry=None, source="none")

    results: list[RouteLinearResult] = []
    segments: list[SegmentRecord] = []
    pairs = [(rid, rid, "matched") for rid in corr.matched] + [
        (b, a, "rekeyed") for b, a in corr.rekeyed.items()
    ]
    for b_id, a_id, status in pairs:
        label = labels_after.get(a_id) or labels_before.get(b_id) or a_id
        result, segs = _paired_route_result(
            b_id,
            a_id,
            label,
            status,
            geoms_before.get(b_id, no_geom),
            geoms_after.get(a_id, no_geom),
            cfg,
        )
        results.append(result)
        segments.extend(segs)
    for rid in corr.added:
        result, segs = _whole_route_result(
            rid,
            labels_after.get(rid, rid),
            geoms_after.get(rid, no_geom),
            "added",
            cfg.linear_change_min_feet,
        )
        results.append(result)
        segments.extend(segs)
    for rid in corr.eliminated:
        result, segs = _whole_route_result(
            rid,
            labels_before.get(rid, rid),
            geoms_before.get(rid, no_geom),
            "eliminated",
            cfg.linear_change_min_feet,
        )
        results.append(result)
        segments.extend(segs)

    changed = [r for r in results if r.has_linear_change]
    unchanged = [r for r in results if r.change_kind == "unchanged_alignment"]
    missing = [r for r in results if r.change_kind == "no_geometry"]
    if changed:
        logging.info(
            "%d route(s) have a linear change: %s",
            len(changed),
            "; ".join(f"{r.route_label} ({r.change_kind})" for r in changed),
        )
    else:
        logging.info("No routes have a linear change at the current thresholds.")
    logging.info(
        "%d route(s) excluded as alignment-unchanged (schedule-only or identical).",
        len(unchanged),
    )
    if missing:
        logging.warning(
            "%d route(s) skipped for missing geometry: %s",
            len(missing),
            "; ".join(r.route_label for r in missing),
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for change_class, filename in (
        ("new", SEGMENTS_NEW_FILENAME),
        ("retained", SEGMENTS_RETAINED_FILENAME),
        ("eliminated", SEGMENTS_ELIMINATED_FILENAME),
    ):
        write_segment_shapefile(
            [s for s in segments if s.change_class == change_class],
            cfg.output_dir / filename,
            metric_crs,
            cfg.distance_output_unit,
        )
    if cfg.write_network_layers:
        write_network_shapefiles(diff_network(geoms_before, geoms_after, cfg), cfg, metric_crs)

    route_table = route_results_frame(results, cfg.distance_output_unit)
    route_log_path = cfg.output_dir / ROUTE_LOG_FILENAME
    route_table.to_csv(route_log_path, index=False, encoding="utf-8")
    logging.info("Wrote: %s", route_log_path)

    summary = Summary(
        before_route_count=len(geoms_before),
        after_route_count=len(geoms_after),
        matched_count=len(corr.matched),
        rekeyed_count=len(corr.rekeyed),
        added_count=len(corr.added),
        eliminated_count=len(corr.eliminated),
        realigned_count=sum(1 for r in results if r.change_kind == "realigned"),
        unchanged_alignment_count=len(unchanged),
        no_geometry_count=len(missing),
        chord_fallback_route_count=sum(
            1
            for rg in list(geoms_before.values()) + list(geoms_after.values())
            if rg.source == "stops_chord"
        ),
        new_len_mi=round(sum(r.new_len_ft for r in results) / 5280.0, 2),
        retained_len_mi=round(sum(r.retained_len_ft for r in results) / 5280.0, 2),
        eliminated_len_mi=round(sum(r.eliminated_len_ft for r in results) / 5280.0, 2),
        buffer_tolerance_feet=cfg.buffer_tolerance_feet,
        min_segment_length_feet=cfg.min_segment_length_feet,
        linear_change_min_feet=cfg.linear_change_min_feet,
        metric_crs=str(metric_crs),
    )
    summary_path = cfg.output_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    logging.info("Wrote: %s", summary_path)

    summary_lines = [
        f"Before / after routes:   {summary.before_route_count} / {summary.after_route_count}",
        f"Matched / rekeyed:       {summary.matched_count} / {summary.rekeyed_count}",
        f"Added / eliminated:      {summary.added_count} / {summary.eliminated_count}",
        f"Realigned:               {summary.realigned_count}",
        f"Alignment unchanged:     {summary.unchanged_alignment_count} (excluded)",
        f"No geometry:             {summary.no_geometry_count}",
        f"New / eliminated miles:  {summary.new_len_mi} / {summary.eliminated_len_mi}",
        f"Metric CRS:              {summary.metric_crs}",
    ]
    if not write_run_log(cfg.output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
    return summary


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
    """Create the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Diff two GTFS feeds into new/retained/eliminated alignment segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--before", default=BEFORE_GTFS_DIR, help="Before GTFS folder.")
    p.add_argument("--after", default=AFTER_GTFS_DIR, help="After GTFS folder.")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for outputs.")
    p.add_argument(
        "--buffer-feet",
        type=float,
        default=BUFFER_TOLERANCE_FEET,
        help="Match tolerance between alignments, in feet.",
    )
    p.add_argument(
        "--min-segment-feet",
        type=float,
        default=MIN_SEGMENT_LENGTH_FEET,
        help="Drop overlay slivers shorter than this, in feet.",
    )
    p.add_argument(
        "--change-min-feet",
        type=float,
        default=LINEAR_CHANGE_MIN_FEET,
        help="Minimum new+eliminated length for a route to count as a linear change.",
    )
    p.add_argument(
        "--rekey-jaccard",
        type=float,
        default=REKEY_MIN_JACCARD,
        help="Minimum stop-set Jaccard to treat a renumbered route_id as the same route.",
    )
    p.add_argument(
        "--chord-fallback",
        action=argparse.BooleanOptionalAction,
        default=ALLOW_CHORD_FALLBACK,
        help="Build stop-to-stop chord geometry for routes without usable shapes.txt.",
    )
    p.add_argument(
        "--network-layers",
        action=argparse.BooleanOptionalAction,
        default=WRITE_NETWORK_LAYERS,
        help="Also write the three system-level (all routes merged) layers.",
    )
    p.add_argument(
        "--target-crs",
        default=TARGET_CRS,
        help='Metric CRS for the overlay, e.g. "EPSG:26918"; empty auto-picks a UTM zone.',
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(notebook_safe_argv(argv))

    placeholders = (
        ("--before", args.before, r"Path\To\Your\GTFS_Before_Folder"),
        ("--after", args.after, r"Path\To\Your\GTFS_After_Folder"),
        ("--output-dir", args.output_dir, r"Path\To\Your\Output_Folder"),
    )
    unset = [flag for flag, value, placeholder in placeholders if value == placeholder]
    if unset:
        logging.warning(
            "These paths are still placeholders: %s. Update the CONFIGURATION section "
            "or pass the matching flags before running.",
            ", ".join(unset),
        )
        return 2

    cfg = Config(
        before_dir=Path(args.before).expanduser(),
        after_dir=Path(args.after).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        buffer_tolerance_feet=args.buffer_feet,
        min_segment_length_feet=args.min_segment_feet,
        linear_change_min_feet=args.change_min_feet,
        rekey_min_jaccard=args.rekey_jaccard,
        allow_chord_fallback=args.chord_fallback,
        write_network_layers=args.network_layers,
        target_crs=args.target_crs,
        distance_output_unit=DISTANCE_OUTPUT_UNIT,
    )
    try:
        run(cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
