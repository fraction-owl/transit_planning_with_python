"""Review current GTFS stop spacing and proposed stop removals.

EDIT HERE: GTFS_PATH and OUTPUT_FOLDER below. Add RemovalScenario entries to
REMOVAL_SCENARIOS to compare proposed removals; leave it [] for a current-only
report, and REVIEW_STOP_KEYS=[] for full selected routes. Inputs can be a GTFS
folder or a zip (nested folder supported). Relative paths are resolved beside
this script, or against the working directory when pasted into a notebook.

Dependencies: pandas, numpy, geopandas, shapely, pyproj; openpyxl for Excel;
matplotlib only when WRITE_REVIEW_MAPS=True. Python 3.9+ syntax.

Outputs under OUTPUT_FOLDER:
  scenario_review/stop_spacing_review.xlsx: Changes, Spacing, Removed stops.
  scenario_review/*.csv: the same three plain tables (optional).
  scenario_review/maps/*.png: paired before/after map panels (optional).
  current_gtfs_gis/: the original baseline GIS / nearby-stop QA outputs.

Each scenario independently removes visits from the original GTFS. compare_to
chooses a reporting baseline; it does not inherit removals. Inputs and GTFS
are never edited. Identical results across patterns are combined; distinct
shape distances and repeated stop visits stay separate. direction_id values
are GTFS identifiers and do not imply north/south. Thresholds are review flags,
not automatic closure decisions. Distances follow bus shapes, not sidewalks.
Ridership joins are descriptive and are not summed across trips or patterns.

X,Y additions, relocations, terminal changes, new route alignments, inferred
opposite-stop pairs and two-feed comparisons are outside this draft's scope.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sys
import zipfile
from collections.abc import Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Point
from shapely.ops import substring


@dataclass(frozen=True)
class RemovalScenario:
    """Independent removals from GTFS, with optional route/direction scope.

    compare_to chooses the report baseline; it does NOT inherit removals.
    Empty route_ids/direction_ids mean all selected routes/directions.
    """

    name: str
    remove_stops: tuple[str, ...]
    route_ids: tuple[str, ...] = ()
    direction_ids: tuple[int, ...] = ()
    compare_to: str = "Current GTFS"


# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_PATH: str = r"Path\To\Your\GTFS_Data_Folder"  # folder or .zip
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

FILTER_OUT_LIST: list[str] = ["9999A", "9999B", "9999C"]
INCLUDE_ROUTE_IDS: list[str] = []  # [] = all routes except FILTER_OUT_LIST

# Union each route/direction's shapes in routes.shp only; spacing is always
# measured along each stopping pattern's own shape.
ROUTE_UNION: bool = False
PROJECTED_CRS: str = "EPSG:2248"  # NAD83 / Maryland (ftUS); any projected CRS in feet or metres

# A pattern's stop farther than this from its place on the shape is skipped;
# 100 m (328 ft) is the GTFS Best Practices stop-to-shape limit.
SERVED_STOP_MAX_OFFSET_FT: float = 328.0

# direction_id is optional in GTFS; trips without a 0/1 value are analyzed
# under this direction instead of being dropped.
UNKNOWN_DIRECTION_ID: int = -1

MIN_SPACING_FT: float = 400.0  # < this distance between served stops
SPACING_LOG_FILE: str = "short_spacing_segments.txt"

# Sets standards for route segments that are too long
# Best applied to local routes, use on express routes sparingly
LONG_SPACING_FT: float = 1_500.0  # > this distance between served stops …
NEAR_BUFFER_FT: float = 99.0  # … and a “missed” stop must lie ≤ this
LONG_SPACING_LOG_FILE: str = "long_spacing_segments.txt"
LONG_SPACING_CSV_FILE: str = "long_spacing_segments.csv"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === BEFORE / AFTER REVIEW ===
# IDs remain text. Choose stop_code OR stop_id; there is no silent fallback.
STOP_KEY: str = "stop_code"
REMOVAL_SCENARIOS: list[RemovalScenario] = []
# Optional corridor focus. [] reports full selected routes. Context endpoints
# outside this list remain in the analysis; this is NOT a stop-service filter.
REVIEW_STOP_KEYS: list[str] = []
WRITE_REVIEW_XLSX: bool = True
WRITE_REVIEW_CSV: bool = True
WRITE_REVIEW_MAPS: bool = False  # Optional matplotlib PNGs, no basemap download
WRITE_LEGACY_GIS: bool = True  # Existing GTFS-only GIS and missed-stop outputs

# Optional one-row-per-stop CSV/XLSX; leave blank to omit this join.
# Attributes are reported, never used to automatically choose removals.
STOP_ATTRIBUTES_PATH: str = ""
STOP_ATTRIBUTES_SHEET: str | int = 0
STOP_ATTRIBUTES_KEY: str = "Stop code"
STOP_ATTRIBUTES_COLUMNS: list[str] = [
    "Avg weekday boardings",
    "Avg weekday alightings",
    "Shelter",
    "Bench",
    "Trash can",
    "Boarding pad",
]
# Optional supplied pair/candidate column, using STOP_KEY values. Blank = off.
STOP_PAIR_COLUMN: str = "Opposite stop candidate"

# Workbook sheets: Changes, Spacing, Removed stops (plain cells, one header row).
# Maps show one scenario/pattern at a time on identical before/after extents.
# Distances are along the unchanged bus shape, NOT pedestrian travel distances.
# Scenario removals never modify the source GTFS. Terminal removals are rejected.

# === END CONFIG ===

# =============================================================================
# FUNCTIONS
# =============================================================================

LONG_SPACING_COLUMNS: list[str] = [
    "route_id",
    "route_short",
    "direction_id",
    "seg_len_ft",
    "start_stop_id",
    "start_stop_name",
    "end_stop_id",
    "end_stop_name",
    "flagged_stop_id",
    "flagged_stop_name",
    "dist_to_route_ft",
]


def _ensure_output_folder(folder: str | Path) -> Path:
    """Create (if necessary) and return the output folder as a ``Path``."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _served_mask(df: pd.DataFrame, rid: str, drn: int) -> pd.Series:
    """Return boolean mask for stops whose ``route_dirs`` pairs include (rid, drn)."""
    return df["route_dirs"].apply(lambda pairs, key=(rid, drn): key in pairs)


def _locate_stops(
    line: LineString,
    points: Sequence[Point | None],
    dists: Sequence[float],
    max_offset: float,
) -> np.ndarray:
    """Return each stop's distance along *line*, in trip order.

    *dists* are positions from ``shape_dist_traveled`` (CRS units, NaN where
    missing). They are used when every stop has one, they never decrease, and
    each lies within *max_offset* of its stop. Otherwise the stops are matched
    to the line's segments in trip order (see :func:`_match_in_order`), so a
    loop or a retraced street keeps the trip's sequence. A stop farther than
    *max_offset* from its position, or without a point, is returned as NaN.
    """
    hint = np.asarray(dists, dtype=float)
    if (
        len(hint) == len(points)
        and np.isfinite(hint).all()
        and (np.diff(hint) >= 0).all()
        and (hint >= 0).all()
        and (hint <= line.length).all()
        and all(
            pt is not None and pt.distance(line.interpolate(d)) <= max_offset
            for pt, d in zip(points, hint)
        )
    ):
        return hint

    out = np.full(len(points), np.nan)
    usable = [
        (i, pt) for i, pt in enumerate(points) if pt is not None and pt.distance(line) <= max_offset
    ]
    if usable:
        xy = np.array([[pt.x, pt.y] for _, pt in usable])
        along, offset = _match_in_order(np.asarray(line.coords)[:, :2], xy)
        ok = offset <= max_offset
        positions = np.maximum.accumulate(along[ok])
        indexes = np.array([i for i, _ in usable])[ok]
        # Recheck offsets after enforcing monotonicity; moving a projection
        # forward must not silently turn a bad location into a valid stop.
        for index, position in zip(indexes, positions):
            point = points[index]
            if point is not None and point.distance(line.interpolate(position)) <= max_offset:
                out[index] = position
    return out


def _match_in_order(vertices: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place points on a polyline in their given order.

    Each point goes to its nearest spot on one segment of the polyline. The
    segments are chosen so they never go backwards from one point to the
    next, and so that the total of each point's distance from the line, plus
    any backtrack between two points on the same segment, is as small as
    possible (a dynamic programme over segments). Unlike projecting each
    point on its own, a stop on a street the route uses twice is placed on
    the pass that fits the trip's order.

    Args:
        vertices: ``(k + 1, 2)`` polyline vertices.
        xy: ``(n, 2)`` points in order.

    Returns:
        ``(along, offset)``: each point's distance along the polyline and its
        distance from that spot.
    """
    start = vertices[:-1]
    vec = np.diff(vertices, axis=0)
    seg_len = np.hypot(vec[:, 0], vec[:, 1])
    seg_from = np.concatenate(([0.0], np.cumsum(seg_len)[:-1]))
    len_sq = np.where(seg_len > 0, seg_len**2, 1.0)

    def _on_segments(pt: np.ndarray, segs: np.ndarray | slice) -> tuple[np.ndarray, np.ndarray]:
        rel = pt - start[segs]
        t = np.clip((rel * vec[segs]).sum(axis=-1) / len_sq[segs], 0.0, 1.0)
        gap = rel - t[..., None] * vec[segs]
        return seg_from[segs] + t * seg_len[segs], np.hypot(gap[..., 0], gap[..., 1])

    # cost[j]: least total so far with the latest point on segment j
    idx = np.arange(len(seg_len))
    prev_along, cost = _on_segments(xy[0], slice(None))
    back: list[np.ndarray] = []
    for pt in xy[1:]:
        along, offset = _on_segments(pt, slice(None))
        # best over strictly earlier segments, and which segment gave it
        best = np.minimum.accumulate(cost)
        improves = np.concatenate(([True], cost[1:] < best[:-1]))
        best_arg = np.maximum.accumulate(np.where(improves, idx, 0))
        earlier = np.concatenate(([np.inf], best[:-1]))
        earlier_arg = np.concatenate(([0], best_arg[:-1]))
        # or the same segment, paying for any step backwards along it; ties
        # go to the earlier segment, i.e. the first pass that fits
        same = cost + np.maximum(prev_along - along, 0.0)
        back.append(np.where(same < earlier, idx, earlier_arg))
        cost = offset + np.minimum(same, earlier)
        prev_along = along

    segs = [int(np.argmin(cost))]
    for prev_best in reversed(back):
        segs.append(int(prev_best[segs[-1]]))
    segs.reverse()
    return _on_segments(xy, np.asarray(segs))


def _pattern_stops(
    pattern: pd.Series,
    stops_by_id: pd.DataFrame,
    max_offset: float,
) -> pd.DataFrame:
    """Return a pattern's located stops in trip order.

    Columns are ``stop_id``, ``stop_name`` and ``dist_along`` (CRS units along
    the pattern's shape). Stops missing from *stops_by_id* (indexed by
    ``stop_id``) or farther than *max_offset* from the shape are left out.
    """
    stop_ids = list(pattern.stop_ids)
    stops = stops_by_id.reindex(stop_ids)
    points = [pt if isinstance(pt, Point) else None for pt in stops.geometry]
    hint = pattern.get("stop_dists")
    dists = _locate_stops(
        pattern.geometry,
        points,
        [np.nan] * len(stop_ids) if hint is None else hint,
        max_offset,
    )
    located = pd.DataFrame(
        {
            "stop_id": stop_ids,
            "stop_name": stops["stop_name"].to_numpy(),
            "dist_along": dists,
        }
    )
    return located.dropna(subset=["dist_along"]).reset_index(drop=True)


def _consecutive_stops(
    patterns_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    max_offset: float,
) -> Iterator[tuple[pd.Series, pd.Series, pd.Series]]:
    """Yield ``(pattern, stop, next_stop)`` for consecutive stops of each pattern.

    A pair repeated by another pattern of the same route, direction and shape
    (e.g. a local and an express sharing a stretch) is yielded once.
    """
    stops_by_id = stops_gdf.drop_duplicates("stop_id").set_index("stop_id")
    seen: set[tuple[Any, ...]] = set()
    for _, pattern in patterns_gdf.iterrows():
        located = _pattern_stops(pattern, stops_by_id, max_offset)
        for i in range(len(located) - 1):
            s0, s1 = located.iloc[i], located.iloc[i + 1]
            key = (
                str(pattern.route_id),
                int(pattern.direction_id),
                pattern.shape_id,
                s0.stop_id,
                s1.stop_id,
                s0.dist_along,
                s1.dist_along,
            )
            if key not in seen:
                seen.add(key)
                yield pattern, s0, s1


def _feet_factor(crs: CRS | str | None) -> float:
    """Return factor to convert from CRS linear units to feet.

    The unit is read from the CRS definition, so any projected CRS works:
    about 1.000002 for US survey feet (e.g. EPSG:2248) and 3.28084 for
    metres (e.g. EPSG:26918).

    Raises:
        ValueError: If *crs* is missing or not projected (degrees are not a
            distance unit).
    """
    if crs is None:
        raise ValueError("No CRS set; cannot convert distances to feet.")
    proj_crs = CRS.from_user_input(crs)
    if not proj_crs.is_projected:
        raise ValueError(
            f"CRS {proj_crs.to_string()} is not projected; set PROJECTED_CRS to a "
            "projected CRS in feet or metres (e.g. EPSG:2248)."
        )
    return proj_crs.axis_info[0].unit_conversion_factor / 0.3048


def _flag_long_spacing_csv(
    patterns_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    threshold_ft: float,
    near_buffer_ft: float,
    csv_path: Path,
    summary: bool = True,
    max_offset_ft: float = SERVED_STOP_MAX_OFFSET_FT,
) -> None:
    """Export a CSV of “missed” stops that fill unusually long gaps.

    A *long gap* is any consecutive pair of stops in a stopping pattern
    whose spacing exceeds *threshold_ft*. For every stop not served by the
    pattern's (route_id, direction_id) that falls **inside** the gap and
    within *near_buffer_ft* of it, a row is written containing:

    | route_id | route_short | direction_id | seg_len_ft | start_stop_id |
    | start_stop_name | end_stop_id | end_stop_name | flagged_stop_id |
    | flagged_stop_name | dist_to_route_ft |

    Parameters
    ----------
    patterns_gdf, stops_gdf
        Projected GeoDataFrames created by :func:`_build_patterns_gdf` and
        :func:`_build_stops_gdf`.
    threshold_ft
        Minimum gap length to examine.
    near_buffer_ft
        Maximum perpendicular distance from the route to consider a stop
        “near” the line.
    csv_path
        Destination for the detailed CSV.
    summary
        If *True*, also write ``<stem>_summary.txt`` listing each
        (route_id, direction_id) that triggered at least one flag.
    max_offset_ft
        A pattern's stops farther than this from the shape are skipped.

    Notes:
    -----
    • The CSV and summary are rewritten on every run, header only when
      nothing is flagged, so results from an earlier run never linger.
    • CRS units are read from ``stops_gdf.crs`` and converted to feet;
      *near_buffer_ft* is converted to CRS units before the spatial search.
    """
    ft_factor: float = _feet_factor(stops_gdf.crs)
    near_buffer_crs: float = near_buffer_ft / ft_factor
    max_offset_crs: float = max_offset_ft / ft_factor
    sindex = stops_gdf.sindex

    records: list[dict[str, Any]] = []

    for pattern, s0, s1 in _consecutive_stops(patterns_gdf, stops_gdf, max_offset_crs):
        seg_len_ft: float = (s1.dist_along - s0.dist_along) * ft_factor
        if seg_len_ft <= threshold_ft:
            continue

        rid: str = str(pattern.route_id)
        drn: int = int(pattern.direction_id)
        # The gap itself, so a loop's other passes and curves are handled
        gap = substring(pattern.geometry, s0.dist_along, s1.dist_along)
        minx, miny, maxx, maxy = gap.bounds
        box = (
            minx - near_buffer_crs,
            miny - near_buffer_crs,
            maxx + near_buffer_crs,
            maxy + near_buffer_crs,
        )

        # candidate “missed” stops not served by this route/direction
        maybe = stops_gdf.iloc[list(sindex.intersection(box))]
        maybe = maybe[~_served_mask(maybe, rid, drn)]

        for _, st in maybe.iterrows():
            offset = st.geometry.distance(gap)
            if 0 < gap.project(st.geometry) < gap.length and offset <= near_buffer_crs:
                records.append(
                    {
                        "route_id": rid,
                        "route_short": pattern.get("route_short_name"),
                        "direction_id": drn,
                        "seg_len_ft": round(seg_len_ft, 1),
                        "start_stop_id": s0.stop_id,
                        "start_stop_name": s0.stop_name,
                        "end_stop_id": s1.stop_id,
                        "end_stop_name": s1.stop_name,
                        "flagged_stop_id": st.stop_id,
                        "flagged_stop_name": st.stop_name,
                        "dist_to_route_ft": round(offset * ft_factor, 1),
                    }
                )

    # —— export (always, so a clean run replaces old findings) ————————
    pd.DataFrame.from_records(records, columns=LONG_SPACING_COLUMNS).to_csv(csv_path, index=False)
    if records:
        logging.info("Wrote long-spacing CSV → %s (%d rows)", csv_path.name, len(records))
    else:
        logging.info("No nearby unserved-stop candidates; see scenario review for all gaps.")

    # —— optional one-line summary ————————————————————————————
    if summary:
        flagged: set[tuple[str, int]] = {(rec["route_id"], rec["direction_id"]) for rec in records}
        summ_path = csv_path.with_name(f"{csv_path.stem}_summary.txt")
        with summ_path.open("w", encoding="utf-8") as fh:
            fh.write("route_id\tdirection_id\n")
            for rid, drn in sorted(flagged):
                fh.write(f"{rid}\t{drn}\n")
        logging.info("Wrote summary → %s", summ_path.name)


def _read_gtfs_tables(gtfs_path: Path) -> dict[str, pd.DataFrame]:
    """Load the five core GTFS tables into DataFrames.

    Parameters
    ----------
    gtfs_path
        Path to either a directory containing ``*.txt`` files or a ``.zip`` GTFS.

    Returns:
    -------
    dict
        Keys ``stops, routes, trips, stop_times, shapes`` → dataframes.

    Notes:
    -----
    Every column is read as text and blanks stay empty strings, so IDs keep
    leading zeros and values such as ``"NA"``; :func:`_prepare_tables`
    converts the numeric fields afterwards.
    """
    filenames: dict[str, str] = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
    }

    def _read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    if gtfs_path.is_dir():
        return {k: _read(gtfs_path / v) for k, v in filenames.items()}

    if gtfs_path.is_file() and gtfs_path.suffix.lower() == ".zip":
        tables: dict[str, pd.DataFrame] = {}
        with zipfile.ZipFile(gtfs_path) as archive:
            for key, filename in filenames.items():
                matches = [
                    name
                    for name in archive.namelist()
                    if Path(name).name.lower() == filename and "__MACOSX" not in name
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {filename} in GTFS zip; found {len(matches)}.")
                with archive.open(matches[0]) as stream:
                    tables[key] = pd.read_csv(stream, dtype=str, keep_default_na=False)
        return tables

    raise ValueError("GTFS_PATH must be a folder or a .zip file.")


def _validate_columns(dfs: dict[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` if any required GTFS column is missing."""
    required: dict[str, set[str]] = {
        "stops": {"stop_id", "stop_lat", "stop_lon", "stop_name"},
        "routes": {"route_id", "route_short_name"},
        "trips": {"trip_id", "route_id", "shape_id"},
        "stop_times": {"trip_id", "stop_id", "stop_sequence"},
        "shapes": {
            "shape_id",
            "shape_pt_sequence",
            "shape_pt_lat",
            "shape_pt_lon",
        },
    }

    missing_msgs: list[str] = []
    for tbl, needed in required.items():
        present = set(dfs[tbl].columns)
        missing = needed - present
        if missing:
            missing_msgs.append(f"{tbl}.txt → missing {', '.join(sorted(missing))}")

    if missing_msgs:
        joined = "\n".join(" • " + msg for msg in missing_msgs)
        raise ValueError(f"GTFS validation failed – required columns not found:\n{joined}")


def _prepare_tables(
    dfs: dict[str, pd.DataFrame],
    unknown_direction: int = UNKNOWN_DIRECTION_ID,
) -> None:
    """Convert the numeric GTFS fields in place; IDs stay text.

    Unparseable numbers become NaN. ``trips.direction_id`` becomes an int,
    with *unknown_direction* for trips whose value is blank, invalid or
    absent (the field is optional in GTFS).
    """
    numeric: dict[str, list[str]] = {
        "stops": ["stop_lat", "stop_lon"],
        "stop_times": ["stop_sequence", "shape_dist_traveled"],
        "shapes": [
            "shape_pt_sequence",
            "shape_pt_lat",
            "shape_pt_lon",
            "shape_dist_traveled",
        ],
    }
    for tbl, cols in numeric.items():
        for col in cols:
            if col in dfs[tbl].columns:
                dfs[tbl][col] = pd.to_numeric(dfs[tbl][col], errors="coerce")

    trips = dfs["trips"]
    raw = trips["direction_id"] if "direction_id" in trips.columns else pd.Series("", trips.index)
    direction = pd.to_numeric(raw, errors="coerce")
    unknown = ~direction.isin([0, 1])
    if unknown.any():
        logging.info(
            "%d of %d trips lack direction_id; analyzing them as direction %d.",
            unknown.sum(),
            len(trips),
            unknown_direction,
        )
    trips["direction_id"] = direction.where(~unknown, unknown_direction).astype(int)
    events = dfs["stop_times"]
    for column in ("pickup_type", "drop_off_type"):
        if column not in events:
            events[column] = "0"
    no_service = (events["pickup_type"] == "1") & (events["drop_off_type"] == "1")
    if no_service.any():
        logging.info("Excluded %d events forbidding both pickup and drop-off.", no_service.sum())
    dfs["stop_times"] = events.loc[~no_service].copy()


def _filter_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    include_ids: Sequence[str | int],
    exclude_ids: Sequence[str | int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply include/exclude lists and return filtered ``routes`` and ``trips``."""
    exclude = {str(rid) for rid in exclude_ids}
    include = {str(rid) for rid in include_ids}
    routes_ok = routes.loc[~routes["route_id"].isin(exclude)].copy()
    if include:
        routes_ok = routes_ok.loc[routes_ok["route_id"].isin(include)].copy()
    trips_ok = trips.loc[trips["route_id"].isin(routes_ok["route_id"])].copy()
    return routes_ok, trips_ok


def _build_stops_gdf(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of **served** stops with list fields for routes/directions.

    ``route_id``, ``direction_id`` and ``route_short_name`` list the values
    seen at each stop; ``route_dirs`` lists the actual (route_id,
    direction_id) pairs, which the separate lists cannot reconstruct.
    """
    served = stop_times.loc[stop_times["trip_id"].isin(trips["trip_id"])]
    stops = stops.loc[stops["stop_id"].isin(served["stop_id"])].copy()
    no_coords = stops["stop_lat"].isna() | stops["stop_lon"].isna()
    if no_coords.any():
        logging.warning("Skipping %d served stops without valid coordinates.", no_coords.sum())
        stops = stops.loc[~no_coords]

    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    ).to_crs(crs)

    trip_attrs = trips[["trip_id", "route_id", "direction_id"]].merge(
        routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )
    merged = (
        served[["trip_id", "stop_id"]]
        .merge(trip_attrs, on="trip_id", how="left")
        .drop_duplicates(["stop_id", "route_id", "direction_id"])
    )
    merged["route_dir"] = list(zip(merged["route_id"], merged["direction_id"]))

    agg = (
        merged.groupby("stop_id")[["route_id", "direction_id", "route_short_name", "route_dir"]]
        .agg(lambda s: sorted(set(s.dropna())))
        .rename(columns={"route_dir": "route_dirs"})
        .reset_index()
    )
    gdf = gdf.merge(agg, on="stop_id", how="left")

    logging.info("Stops GDF – kept %d served stops.", len(gdf))
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)


def _build_shape_lines(
    shapes: pd.DataFrame,
    shape_ids: Iterable[str],
    crs: str,
) -> gpd.GeoDataFrame:
    """Return one projected polyline per shape in *shape_ids*.

    ``vertex_dists`` holds each vertex's ``shape_dist_traveled`` as an array,
    or None when the shape's values are missing or decrease.
    """
    pts = shapes.loc[shapes["shape_id"].isin(set(shape_ids))]
    pts = pts.dropna(subset=["shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"])
    pts = pts.sort_values(["shape_id", "shape_pt_sequence"])
    has_dists = "shape_dist_traveled" in pts.columns

    records: list[dict[str, Any]] = []
    for shape_id, grp in pts.groupby("shape_id", sort=False):
        if len(grp) < 2:
            logging.warning("Shape %s has fewer than 2 valid points; skipping.", shape_id)
            continue
        vertex_dists = grp["shape_dist_traveled"].to_numpy(float) if has_dists else None
        if vertex_dists is not None and not (
            np.isfinite(vertex_dists).all() and (np.diff(vertex_dists) >= 0).all()
        ):
            vertex_dists = None
        records.append(
            {
                "shape_id": shape_id,
                "vertex_dists": vertex_dists,
                "geometry": LineString(zip(grp.shape_pt_lon, grp.shape_pt_lat)),
            }
        )

    return gpd.GeoDataFrame(
        records,
        columns=["shape_id", "vertex_dists", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(crs)


def _build_routes_gdf(
    shape_lines: gpd.GeoDataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    union_shapes: bool,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame of polylines keyed by ``(route_id, direction_id)``.

    One row per (route_id, direction_id, shape_id), so a shape shared by two
    routes appears under each; with *union_shapes* each route/direction's
    shapes are dissolved together. Used for ``routes.shp`` only.
    """
    keys = trips[["route_id", "direction_id", "shape_id"]].drop_duplicates()
    gdf = (
        shape_lines[["shape_id", "geometry"]]
        .merge(keys, on="shape_id", how="inner")
        .merge(routes, on="route_id", how="left")
    )

    if union_shapes:
        gdf = gdf.dissolve(
            by=["route_id", "direction_id"],
            as_index=False,
            aggfunc={
                col: "first"
                for col in ("route_short_name", "route_long_name")
                if col in gdf.columns
            },
        ).explode(ignore_index=True)

    logging.info("Routes GDF – built %d shapes.", len(gdf))
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=shape_lines.crs)


def _dists_along_line(
    line: LineString,
    vertex_dists: np.ndarray | None,
    stop_dists: Sequence[float],
) -> tuple[float, ...]:
    """Convert stops' ``shape_dist_traveled`` to CRS distances along *line*.

    Interpolates between the shape's vertices, whose own values are
    *vertex_dists*. Returns NaNs when either side lacks the values.
    """
    stop_along = np.asarray(stop_dists, dtype=float)
    coords = np.asarray(line.coords)
    if (
        vertex_dists is None
        or len(vertex_dists) != len(coords)
        or not np.isfinite(stop_along).all()
        or not np.isfinite(vertex_dists).all()
        or (np.diff(vertex_dists) < 0).any()
        or (np.diff(stop_along) < 0).any()
        or (stop_along < vertex_dists[0]).any()
        or (stop_along > vertex_dists[-1]).any()
    ):
        return tuple(np.full(len(stop_along), np.nan))
    steps = np.diff(coords[:, :2], axis=0)
    vertex_along = np.concatenate(([0.0], np.cumsum(np.hypot(steps[:, 0], steps[:, 1]))))
    if (np.diff(vertex_along)[np.diff(vertex_dists) == 0] > 1e-6).any():
        return tuple(np.full(len(stop_along), np.nan))
    return tuple(np.interp(stop_along, vertex_dists, vertex_along))


def _build_patterns_gdf(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    shape_lines: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return one row per distinct stopping pattern, with its shape's polyline.

    A pattern is a (route_id, direction_id, shape_id) and the trip's stop IDs
    in ``stop_sequence`` order, repeats included. ``stop_dists`` holds each
    stop's distance along the shape from ``shape_dist_traveled`` (CRS units),
    or NaNs when the feed does not provide it.
    """
    st = stop_times.loc[stop_times["trip_id"].isin(trips["trip_id"])]
    st = st.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])
    if "shape_dist_traveled" not in st.columns:
        st = st.assign(shape_dist_traveled=np.nan)

    per_trip = (
        st.groupby("trip_id", sort=False)
        .agg(stop_ids=("stop_id", tuple), stop_dists=("shape_dist_traveled", tuple))
        .reset_index()
        .merge(trips[["trip_id", "route_id", "direction_id", "shape_id"]], on="trip_id")
    )
    no_shape = ~per_trip["shape_id"].isin(shape_lines["shape_id"])
    if no_shape.any():
        logging.warning(
            "Skipping %d of %d trips whose shape_id is blank or has no usable shape.",
            no_shape.sum(),
            len(per_trip),
        )
    # Different passes along a loop can share the same stop IDs.
    patterns = per_trip.drop_duplicates(
        ["route_id", "direction_id", "shape_id", "stop_ids", "stop_dists"]
    )

    gdf = (
        shape_lines.merge(patterns.drop(columns="trip_id"), on="shape_id", how="inner")
        .merge(routes[["route_id", "route_short_name"]], on="route_id", how="left")
        .sort_values(["route_id", "direction_id", "shape_id"], kind="stable")
        .reset_index(drop=True)
    )
    gdf["stop_dists"] = [
        _dists_along_line(line, vertex_dists, stop_dists)
        for line, vertex_dists, stop_dists in zip(
            gdf.geometry, gdf["vertex_dists"], gdf["stop_dists"]
        )
    ]
    gdf = gdf.drop(columns="vertex_dists")
    gdf["pattern_id"] = [f"P{i + 1:04d}" for i in range(len(gdf))]

    logging.info("Patterns GDF – %d stopping patterns from %d trips.", len(gdf), len(per_trip))
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=shape_lines.crs)


def _split_into_segments(
    patterns_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    crs: str,
    max_offset_ft: float = SERVED_STOP_MAX_OFFSET_FT,
) -> gpd.GeoDataFrame:
    """Split each pattern's shape at the pattern's stops and return segment GDF.

    The pieces before the first stop and after the last stop are kept. A
    pattern's stops farther than *max_offset_ft* from the shape are skipped,
    and a piece repeated by another pattern on the same shape is kept once.
    """
    seg_records: list[dict[str, object]] = []
    ft_factor = _feet_factor(crs)
    max_offset_crs = max_offset_ft / ft_factor
    stops_by_id = stops_gdf.drop_duplicates("stop_id").set_index("stop_id")
    seen: set[tuple[Any, ...]] = set()

    for _, r in patterns_gdf.iterrows():
        line: LineString = r.geometry
        rid: str = str(r.route_id)
        drn: int = int(r.direction_id)

        located = _pattern_stops(r, stops_by_id, max_offset_crs)
        if located.empty:
            continue

        # Cut at each stop's distance along the line. Unlike shapely's split,
        # substring does not need the cut points to lie exactly on the line.
        cuts = np.unique(np.concatenate(([0.0], located["dist_along"], [line.length])))

        for start_d, end_d in zip(cuts[:-1], cuts[1:]):
            key = (rid, drn, r.shape_id, start_d, end_d)
            if key in seen:
                continue
            seen.add(key)
            seg = substring(line, start_d, end_d)
            if isinstance(seg, LineString) and seg.length > 0:
                seg_records.append(
                    {
                        "route_id": rid,
                        "direction_id": drn,
                        "route_short": r.get("route_short_name"),
                        "geometry": seg,
                    }
                )

    seg_gdf = gpd.GeoDataFrame(
        seg_records,
        columns=["route_id", "direction_id", "route_short", "geometry"],
        geometry="geometry",
        crs=crs,
    )
    seg_gdf["length_ft"] = seg_gdf.length * ft_factor
    logging.info("Segments GDF – generated %d pieces.", len(seg_gdf))
    return seg_gdf


def _export(gdf: gpd.GeoDataFrame, out_dir: Path, name: str) -> None:
    """Write *gdf* to ESRI Shapefile ``<out_dir>/<name>.shp``.

    Shapefile fields hold single values, so list values (a stop's routes and
    directions) are written as comma-separated text.
    """
    out = gdf.copy()
    for col in out.columns.drop(out.geometry.name):
        is_list = out[col].map(lambda v: isinstance(v, (list, tuple, set)))
        if is_list.any():
            out[col] = out[col].map(
                lambda v: ",".join(map(str, v)) if isinstance(v, (list, tuple, set)) else v
            )
    path = out_dir / f"{name}.shp"
    out.to_file(path)
    logging.info("Wrote %s", path.name)


def _export_segments_by_route_dir(seg_gdf: gpd.GeoDataFrame, out_dir: Path) -> None:
    """Write one shapefile per ``(route_id, direction_id)``."""
    for key, grp in seg_gdf.groupby(["route_id", "direction_id"]):
        rid, drn = cast("tuple[str, int]", key)
        suffix = f"dir{drn}"
        fname = f"{rid}_{suffix}.shp"
        grp_gdf = gpd.GeoDataFrame(grp, geometry="geometry", crs=seg_gdf.crs)
        grp_gdf.to_file(out_dir / fname)
        logging.info("Wrote %s", fname)


def _flag_short_spacing(
    patterns_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    threshold_ft: float,
    log_path: Path,
    max_offset_ft: float = SERVED_STOP_MAX_OFFSET_FT,
) -> None:
    """Write a log of consecutive stops spaced closer than *threshold_ft*.

    Stops are evaluated in trip order along each stopping pattern's shape.
    A pattern's stops farther than *max_offset_ft* from the shape are skipped.
    """
    factor_ft: float = _feet_factor(stops_gdf.crs)
    max_offset_crs: float = max_offset_ft / factor_ft

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "route_id\tdirection_id\tbegin_stop_id\tbegin_stop_name\t"
            "end_stop_id\tend_stop_name\tspacing_ft\n"
        )

        for pattern, s0, s1 in _consecutive_stops(patterns_gdf, stops_gdf, max_offset_crs):
            spacing_ft = (s1.dist_along - s0.dist_along) * factor_ft
            if spacing_ft < threshold_ft:
                fh.write(
                    f"{pattern.route_id}\t{int(pattern.direction_id)}\t"
                    f"{s0.stop_id}\t{s0.stop_name}\t"
                    f"{s1.stop_id}\t{s1.stop_name}\t"
                    f"{spacing_ft:.1f}\n"
                )

    logging.info("Wrote short-spacing log → %s", log_path.name)


def _build_stop_layers(
    dfs: dict[str, pd.DataFrame],
    trips_selected: pd.DataFrame,
    routes_selected: pd.DataFrame,
    crs: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return stop layers for *all* routes and for the filtered subset.

    Parameters
    ----------
    dfs
        Dictionary of raw GTFS tables as DataFrames (output of
        ``_read_gtfs_tables``).
    trips_selected
        Trips that survived the include/exclude filter.
    routes_selected
        Routes that survived the include/exclude filter.
    crs
        Target projected CRS (feet or metres).

    Returns:
    -------
    tuple
        ``(all_stops_gdf, selected_stops_gdf)`` where:

        * **all_stops_gdf** – every served stop in the feed (no filters),
        * **selected_stops_gdf** – only the stops used by the filtered
          ``routes_selected``/``trips_selected`` set.

    Notes:
    -----
    This helper lets the long-spacing check see *all* active stops, while the
    segment-splitting logic still works with the leaner, filtered layer.
    """
    all_stops_gdf = _build_stops_gdf(
        dfs["stops"],
        dfs["stop_times"],
        dfs["trips"],  # unfiltered
        dfs["routes"],  # unfiltered
        crs,
    )

    selected_stops_gdf = _build_stops_gdf(
        dfs["stops"],
        dfs["stop_times"],
        trips_selected,  # filtered
        routes_selected,  # filtered
        crs,
    )

    return all_stops_gdf, selected_stops_gdf


@dataclass
class LocatedPattern:
    """One stopping pattern located once, before applying any removals."""

    pattern_id: str
    route_id: str
    route_short: str
    direction_id: int
    shape_id: str
    geometry: LineString
    stops: pd.DataFrame


def _scoped_removals(pattern: LocatedPattern, scenario: RemovalScenario) -> set[str]:
    """Resolve a scenario's applicability without altering other routes."""
    if scenario.route_ids and pattern.route_id not in scenario.route_ids:
        return set()
    if scenario.direction_ids and pattern.direction_id not in scenario.direction_ids:
        return set()
    return set(scenario.remove_stops)


def _locate_review_patterns(
    patterns: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    stop_key: str,
) -> list[LocatedPattern]:
    """Locate full patterns; fail rather than bridge across an unlocated stop."""
    by_id = stops.set_index("stop_id")
    factor = _feet_factor(stops.crs)
    result: list[LocatedPattern] = []
    for _, pattern in patterns.iterrows():
        ids = list(pattern.stop_ids)
        rows = by_id.reindex(ids)
        points = [p if isinstance(p, Point) else None for p in rows.geometry]
        distances = _locate_stops(
            pattern.geometry,
            points,
            pattern.stop_dists,
            SERVED_STOP_MAX_OFFSET_FT / factor,
        )
        bad = [ids[i] for i, distance in enumerate(distances) if not math.isfinite(distance)]
        if bad:
            raise ValueError(
                f"Pattern {pattern.pattern_id} / shape {pattern.shape_id}: "
                f"could not locate stop_id(s) {bad}. "
                "Review aborted; these stops were not treated as removals."
            )
        located = pd.DataFrame(
            {
                "stop_id": ids,
                "key": rows[stop_key].astype(str).to_numpy(),
                "name": rows["stop_name"].to_numpy(),
                "along_ft": distances * factor,
                "x": rows.geometry.x.to_numpy(),
                "y": rows.geometry.y.to_numpy(),
            }
        )
        located["visit"] = located.groupby("stop_id", sort=False).cumcount() + 1
        result.append(
            LocatedPattern(
                str(pattern.pattern_id),
                str(pattern.route_id),
                str(pattern.route_short_name),
                int(pattern.direction_id),
                str(pattern.shape_id),
                pattern.geometry,
                located,
            )
        )
    return result


def _validate_review(
    dfs: dict[str, pd.DataFrame],
    trips: pd.DataFrame,
    shape_lines: gpd.GeoDataFrame,
    stop_key: str,
    scenarios: list[RemovalScenario],
    focus: list[str],
) -> None:
    """Reject ambiguous IDs, unusable source events and misleading comparisons."""
    if stop_key not in {"stop_id", "stop_code"}:
        raise ValueError("STOP_KEY must be 'stop_code' or 'stop_id'.")
    stops = dfs["stops"]
    if stop_key not in stops:
        raise ValueError(f"stops.txt has no {stop_key}; explicitly choose STOP_KEY='stop_id'.")
    if stops["stop_id"].duplicated().any():
        raise ValueError("Duplicate stop_id values in stops.txt.")
    events = dfs["stop_times"].loc[dfs["stop_times"]["trip_id"].isin(trips["trip_id"])]
    missing = set(events["stop_id"]) - set(stops["stop_id"])
    if missing:
        raise ValueError(f"Selected stop_times reference missing stop_id(s): {sorted(missing)}")
    selected = stops.loc[stops["stop_id"].isin(events["stop_id"])]
    if selected[stop_key].eq("").any() or selected[stop_key].duplicated().any():
        raise ValueError(f"Blank or duplicate {stop_key} in selected stops; use a unique STOP_KEY.")
    if (
        events["stop_sequence"].isna().any()
        or events.duplicated(["trip_id", "stop_sequence"]).any()
    ):
        raise ValueError("Selected stop_times contain invalid or duplicate stop_sequence values.")
    if not set(trips["trip_id"]).issubset(set(events["trip_id"])):
        raise ValueError("A selected trip has no boardable/alightable stop events.")
    if not set(trips["shape_id"]).issubset(set(shape_lines["shape_id"])):
        raise ValueError("A selected trip has no usable shape; review would be incomplete.")
    if (
        not selected["stop_lat"].between(-90, 90).all()
        or not selected["stop_lon"].between(-180, 180).all()
    ):
        raise ValueError("Invalid coordinates in selected stops.")
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("Scenario names must be nonempty and unique.")
    if "Current GTFS" in names:
        raise ValueError("'Current GTFS' is reserved for the automatic baseline.")
    known_names = {"Current GTFS", *names}
    for scenario in scenarios:
        if scenario.compare_to not in known_names or scenario.compare_to == scenario.name:
            raise ValueError(f"Invalid compare_to for scenario {scenario.name!r}.")
        if any(not isinstance(key, str) or not key.strip() for key in scenario.remove_stops):
            raise ValueError("Removal IDs must be nonempty strings, e.g. '1001'.")
        if len(scenario.remove_stops) != len(set(scenario.remove_stops)):
            raise ValueError(f"Duplicate removal ID in {scenario.name!r}.")
        if set(scenario.route_ids) - set(trips["route_id"]):
            raise ValueError(f"Scenario {scenario.name!r} references an unselected route_id.")
        scoped = trips
        if scenario.route_ids:
            scoped = scoped.loc[scoped["route_id"].isin(scenario.route_ids)]
        if scenario.direction_ids:
            scoped = scoped.loc[scoped["direction_id"].isin(scenario.direction_ids)]
        scoped_ids = events.loc[events["trip_id"].isin(scoped["trip_id"]), "stop_id"]
        keys = set(stops.loc[stops["stop_id"].isin(scoped_ids), stop_key])
        unknown = set(scenario.remove_stops) - keys
        if unknown:
            raise ValueError(
                f"Scenario {scenario.name!r}: IDs not served in its scope: {sorted(unknown)}"
            )
    unknown_focus = set(focus) - set(selected[stop_key])
    if unknown_focus:
        raise ValueError(f"Review focus IDs not served in selected routes: {sorted(unknown_focus)}")
    if not 0 <= MIN_SPACING_FT < LONG_SPACING_FT:
        raise ValueError("Require 0 <= MIN_SPACING_FT < LONG_SPACING_FT.")


def _read_stop_attributes() -> dict[str, dict[str, Any]]:
    """Read an optional one-to-one join; never silently aggregate ridership."""
    if not STOP_ATTRIBUTES_PATH:
        return {}
    path = Path(STOP_ATTRIBUTES_PATH).expanduser()
    if not path.is_absolute():
        script = Path(globals().get("__file__", Path.cwd() / "gtfs_stop_spacing.py"))
        path = script.resolve().parent / path
    text_columns: dict[Hashable, Any] = {STOP_ATTRIBUTES_KEY: str}
    if STOP_PAIR_COLUMN:
        text_columns[STOP_PAIR_COLUMN] = str
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=text_columns, keep_default_na=False)
    elif path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(
            path,
            sheet_name=STOP_ATTRIBUTES_SHEET,
            dtype=text_columns,
            keep_default_na=False,
        )
    else:
        raise ValueError("STOP_ATTRIBUTES_PATH must be a CSV or XLSX.")
    columns = list(dict.fromkeys([STOP_ATTRIBUTES_KEY, *STOP_ATTRIBUTES_COLUMNS]))
    if STOP_PAIR_COLUMN:
        columns = list(dict.fromkeys([*columns, STOP_PAIR_COLUMN]))
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Stop attributes missing columns: {sorted(missing)}")
    frame[STOP_ATTRIBUTES_KEY] = frame[STOP_ATTRIBUTES_KEY].str.strip()
    if frame[STOP_ATTRIBUTES_KEY].eq("").any() or frame[STOP_ATTRIBUTES_KEY].duplicated().any():
        raise ValueError("Stop attributes must contain exactly one row per nonblank stop key.")
    return {
        str(row[STOP_ATTRIBUTES_KEY]): {
            key: row[key] for key in columns if key != STOP_ATTRIBUTES_KEY
        }
        for row in frame.to_dict("records")
    }


def _kept_indexes(pattern: LocatedPattern, scenario: RemovalScenario) -> list[int]:
    """Remove visits on their original measures; never rematch the reduced sequence."""
    removals = _scoped_removals(pattern, scenario)
    keys = pattern.stops["key"].tolist()
    if keys and (keys[0] in removals or keys[-1] in removals):
        raise ValueError(
            f"{scenario.name}: removal affects an endpoint of {pattern.pattern_id} "
            f"({keys[0]} to {keys[-1]}). Terminal changes require a separate review."
        )
    return [i for i, key in enumerate(keys) if key not in removals]


def _pattern_info(pattern: LocatedPattern) -> dict[str, Any]:
    """Shared, public report identity fields."""
    return {
        "Route": pattern.route_short,
        "Route ID": pattern.route_id,
        "Direction ID": pattern.direction_id,
        "Pattern IDs": pattern.pattern_id,
        "Shape IDs": pattern.shape_id,
    }


def _focus_bounds(pattern: LocatedPattern, focus: list[str]) -> tuple[int, int] | None:
    """Include a corridor and its immediately bounding retained stops."""
    if not focus:
        return 0, len(pattern.stops) - 1
    indexes = [i for i, key in enumerate(pattern.stops["key"]) if key in set(focus)]
    return (min(indexes), max(indexes)) if indexes else None


def _gap_rows(
    pattern: LocatedPattern,
    scenario: RemovalScenario,
    baseline: RemovalScenario,
    focus: list[str],
) -> list[dict[str, Any]]:
    """Compare each after-gap to the original chain occupying the same interval."""
    before = _kept_indexes(pattern, baseline)
    after = _kept_indexes(pattern, scenario)
    if not set(after).issubset(before):
        raise ValueError(f"{scenario.name!r} would add service relative to {baseline.name!r}.")
    bounds = _focus_bounds(pattern, focus)
    if bounds is None:
        return []
    stops = pattern.stops
    result: list[dict[str, Any]] = []
    for start, end in zip(after[:-1], after[1:]):
        if end < bounds[0] or start > bounds[1]:
            continue
        chain = [i for i in before if start <= i <= end]
        old_gaps = [
            float(stops.loc[j, "along_ft"] - stops.loc[i, "along_ft"])
            for i, j in zip(chain[:-1], chain[1:])
        ]
        spacing = float(stops.loc[end, "along_ft"] - stops.loc[start, "along_ft"])
        if not math.isclose(sum(old_gaps), spacing, abs_tol=1e-6):
            raise ValueError("Before/after shape-distance reconciliation failed.")
        removed = chain[1:-1]
        long_status = ""
        if spacing > LONG_SPACING_FT:
            long_status = "Unchanged long gap"
            if removed:
                long_status = (
                    "Lengthened existing long gap"
                    if max(old_gaps) > LONG_SPACING_FT
                    else "New long gap"
                )
            elif scenario.name == "Current GTFS":
                long_status = "Existing long gap"
        result.append(
            {
                "Scenario": scenario.name,
                "Compared with": baseline.name,
                **_pattern_info(pattern),
                "From stop": stops.loc[start, "key"],
                "To stop": stops.loc[end, "key"],
                "From visit": int(stops.loc[start, "visit"]),
                "To visit": int(stops.loc[end, "visit"]),
                "Spacing (ft)": spacing,
                "Spacing (mi)": spacing / 5280,
                "Change": "Consolidated" if removed else "Unchanged",
                "Stops removed vs baseline": ", ".join(stops.loc[i, "key"] for i in removed),
                "Baseline stop chain": " > ".join(stops.loc[i, "key"] for i in chain),
                "Baseline gaps (ft)": " + ".join(f"{gap:.1f}" for gap in old_gaps),
                "Below short threshold": spacing < MIN_SPACING_FT,
                "Above long threshold": spacing > LONG_SPACING_FT,
                "Short threshold (ft)": MIN_SPACING_FT,
                "Long threshold (ft)": LONG_SPACING_FT,
                "Long gap status": long_status,
                "From stop name": stops.loc[start, "name"],
                "To stop name": stops.loc[end, "name"],
                "_start": start,
                "_end": end,
            }
        )
    return result


def _pair_status(
    key: str,
    patterns: list[LocatedPattern],
    scenario: RemovalScenario,
) -> str:
    """Describe supplied pair service across the selected patterns."""
    states: set[bool] = set()
    for pattern in patterns:
        if key in set(pattern.stops["key"]):
            states.add(key not in _scoped_removals(pattern, scenario))
    if not states:
        return "Not identified in selected service"
    if len(states) == 2:
        return "Partly retained in selected service"
    return "Retained in selected service" if True in states else "Removed from selected service"


def _removed_rows(
    pattern: LocatedPattern,
    scenario: RemovalScenario,
    baseline: RemovalScenario,
    focus: list[str],
    attributes: dict[str, dict[str, Any]],
    all_patterns: list[LocatedPattern],
) -> list[dict[str, Any]]:
    """Report alternatives for each newly removed visit in its travel direction."""
    before, after = _kept_indexes(pattern, baseline), _kept_indexes(pattern, scenario)
    bounds = _focus_bounds(pattern, focus)
    if bounds is None:
        return []
    stops = pattern.stops
    result: list[dict[str, Any]] = []
    for index in sorted(set(before) - set(after)):
        if not bounds[0] <= index <= bounds[1]:
            continue
        previous = max(i for i in after if i < index)
        following = min(i for i in after if i > index)
        backward = float(stops.loc[index, "along_ft"] - stops.loc[previous, "along_ft"])
        forward = float(stops.loc[following, "along_ft"] - stops.loc[index, "along_ft"])
        nearest = previous if backward <= forward else following
        key = str(stops.loc[index, "key"])
        attrs = attributes.get(key, {})
        row = {
            "Scenario": scenario.name,
            "Compared with": baseline.name,
            **_pattern_info(pattern),
            "Removed stop": key,
            "Visit": int(stops.loc[index, "visit"]),
            "Stop name": stops.loc[index, "name"],
            "Previous retained stop": stops.loc[previous, "key"],
            "Distance back (ft)": backward,
            "Next retained stop": stops.loc[following, "key"],
            "Distance forward (ft)": forward,
            "Nearest retained stop": stops.loc[nearest, "key"],
            "Nearest distance (ft)": min(backward, forward),
        }
        if attributes:
            row["Attributes matched"] = key in attributes
            row.update({column: attrs.get(column, "") for column in STOP_ATTRIBUTES_COLUMNS})
            if STOP_PAIR_COLUMN:
                pair = str(attrs.get(STOP_PAIR_COLUMN, "")).strip()
                row[STOP_PAIR_COLUMN] = pair
                row["Pair service after"] = (
                    _pair_status(pair, all_patterns, scenario) if pair else ""
                )
        result.append(row)
    return result


def _collapse_report_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Combine identical results across patterns, preserving distinct stop visits."""
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        public = {key: value for key, value in record.items() if not key.startswith("_")}
        key = tuple(
            (column, round(value, 6) if isinstance(value, float) else value)
            for column, value in public.items()
            if column not in {"Pattern IDs", "Shape IDs"}
        )
        if key not in groups:
            groups[key] = {**public, "Pattern IDs": set(), "Shape IDs": set()}
        groups[key]["Pattern IDs"].add(public["Pattern IDs"])
        groups[key]["Shape IDs"].add(public["Shape IDs"])
    rows = []
    for record in groups.values():
        rows.append(
            {
                **record,
                "Pattern IDs": ", ".join(sorted(record["Pattern IDs"])),
                "Shape IDs": ", ".join(sorted(record["Shape IDs"])),
            }
        )
    return pd.DataFrame(rows)


def _write_review_tables(
    spacing: pd.DataFrame,
    removed: pd.DataFrame,
    output: Path,
) -> None:
    """Write ordinary cells only; no banners, colors, notes, styled tables or charts."""
    changes = spacing.loc[spacing["Change"] == "Consolidated"]
    if removed.empty and not len(removed.columns):
        removed = pd.DataFrame(
            columns=[
                "Scenario",
                "Removed stop",
                "Nearest retained stop",
                "Nearest distance (ft)",
            ]
        )
    tables = {"Changes": changes, "Spacing": spacing, "Removed stops": removed}
    if WRITE_REVIEW_CSV:
        for name, frame in tables.items():
            frame.to_csv(output / f"{name.lower().replace(' ', '_')}.csv", index=False)
    if WRITE_REVIEW_XLSX:
        # A native Python exporter keeps the downloaded script self-contained.
        from openpyxl import Workbook
        from openpyxl.cell.cell import Cell
        from openpyxl.utils import get_column_letter

        workbook = Workbook()
        workbook.remove(workbook.active)
        for name, frame in tables.items():
            sheet = workbook.create_sheet(name)
            sheet.append(frame.columns.tolist())
            for values in frame.itertuples(index=False, name=None):
                cells = []
                for value in values:
                    if pd.isna(value):
                        value = None
                    if isinstance(value, str):
                        cell = Cell(sheet, value=value)
                        cell.data_type = "s"  # IDs/names beginning '=' are literal text.
                        cells.append(cell)
                    else:
                        cells.append(value)
                sheet.append(cells)
            sheet.freeze_panes = "A2"
            for col_index, column in enumerate(frame.columns, start=1):
                width = max(len(str(column)), *(len(str(v)) for v in frame[column].head(200)), 8)
                sheet.column_dimensions[get_column_letter(col_index)].width = min(width + 2, 55)
                number_format = (
                    "0.0"
                    if column.endswith("(ft)")
                    else "0.000"
                    if column.endswith("(mi)")
                    else None
                )
                if number_format:
                    for cell in list(sheet.columns)[col_index - 1][1:]:
                        cell.number_format = number_format
        workbook.save(output / "stop_spacing_review.xlsx")
    logging.info(
        "Review tables: %d gaps, %d changed gaps, %d removed-stop rows.",
        len(spacing),
        len(changes),
        len(removed),
    )


def _plot_comparison(
    pattern: LocatedPattern,
    scenario: RemovalScenario,
    baseline: RemovalScenario,
    rows: list[dict[str, Any]],
    output: Path,
    factor: float,
) -> None:
    """Render matching before/after extents from the same records used by the tables."""
    if not any(row["Change"] == "Consolidated" for row in rows):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = min(row["_start"] for row in rows)
    last = max(row["_end"] for row in rows)
    stops = pattern.stops
    begin, finish = (
        float(stops.loc[first, "along_ft"]),
        float(stops.loc[last, "along_ft"]),
    )
    line = substring(pattern.geometry, begin / factor, finish / factor)
    if not isinstance(line, LineString):
        return
    minx, miny, maxx, maxy = line.bounds
    # Include physical stop coordinates as well as their projected positions.
    visible = stops.iloc[first : last + 1]
    minx, maxx = min(minx, visible.x.min()), max(maxx, visible.x.max())
    miny, maxy = min(miny, visible.y.min()), max(maxy, visible.y.max())
    padding = max(maxx - minx, maxy - miny, 200 / factor) * 0.12
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    before, after = _kept_indexes(pattern, baseline), _kept_indexes(pattern, scenario)
    removed = set(before) - set(after)
    for axis, title, kept in zip(axes, [baseline.name, scenario.name], [before, after]):
        x, y = line.xy
        axis.plot(x, y, color="0.65", linewidth=2, zorder=1)
        indexes = [i for i in kept if first <= i <= last]
        selected = stops.iloc[indexes]
        axis.scatter(selected.x, selected.y, color="#176087", s=30, label="Served stop", zorder=3)
        for index in indexes:
            row = stops.iloc[index]
            axis.annotate(
                str(row.key),
                (row.x, row.y),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=8,
            )
        if title == scenario.name:
            deleted = stops.iloc[sorted(i for i in removed if first <= i <= last)]
            axis.scatter(
                deleted.x,
                deleted.y,
                marker="x",
                color="#a33a2b",
                s=45,
                label="Removed stop",
                zorder=4,
            )
            for row in deleted.itertuples():
                axis.annotate(
                    str(row.key),
                    (row.x, row.y),
                    xytext=(-7, 7),
                    ha="right",
                    textcoords="offset points",
                    fontsize=8,
                    color="#a33a2b",
                )
            for row in rows:
                if row["Change"] != "Consolidated":
                    continue
                start = float(stops.loc[row["_start"], "along_ft"]) / factor
                end = float(stops.loc[row["_end"], "along_ft"]) / factor
                segment = substring(pattern.geometry, start, end)
                if isinstance(segment, LineString):
                    x, y = segment.xy
                    axis.plot(x, y, color="#a33a2b", linewidth=2.5, zorder=2)
                    midpoint = segment.interpolate(0.5, normalized=True)
                    axis.annotate(
                        f"{row['Spacing (ft)']:,.0f} ft",
                        (midpoint.x, midpoint.y),
                        xytext=(12, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color="#a33a2b",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                    )
        axis.set_xlim(minx - padding, maxx + padding)
        axis.set_ylim(miny - padding, maxy + padding)
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=11)
        axis.set_axis_off()
        axis.legend(loc="lower left", fontsize=8)
        axis.annotate(
            "N",
            xy=(0.95, 0.95),
            xytext=(0.95, 0.86),
            xycoords="axes fraction",
            ha="center",
            arrowprops={"arrowstyle": "->"},
        )
    fig.suptitle(
        f"Route {pattern.route_short} | direction_id {pattern.direction_id} | {pattern.shape_id}",
        fontsize=12,
    )
    fig.supxlabel(
        "Distances follow the bus route; pedestrian access is not evaluated.",
        fontsize=9,
    )
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", scenario.name).strip("_")[:60]
    digest = hashlib.sha256(scenario.name.encode()).hexdigest()[:6]
    filename = f"{slug}_{digest}_{pattern.pattern_id}.png"
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def _run_spacing_review(
    patterns: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    output: Path,
) -> None:
    """Apply independent removal scenarios to a common set of located patterns."""
    located = _locate_review_patterns(patterns, stops, STOP_KEY)
    current = RemovalScenario("Current GTFS", ())
    scenarios = [current, *REMOVAL_SCENARIOS]
    by_name = {scenario.name: scenario for scenario in scenarios}
    attributes = _read_stop_attributes()
    gaps: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    plot_jobs: list[
        tuple[LocatedPattern, RemovalScenario, RemovalScenario, list[dict[str, Any]]]
    ] = []
    for scenario in scenarios:
        baseline = by_name[scenario.compare_to]
        for pattern in located:
            records = _gap_rows(pattern, scenario, baseline, REVIEW_STOP_KEYS)
            gaps.extend(records)
            removed.extend(
                _removed_rows(pattern, scenario, baseline, REVIEW_STOP_KEYS, attributes, located)
            )
            if WRITE_REVIEW_MAPS:
                plot_jobs.append((pattern, scenario, baseline, records))
    if not gaps:
        raise ValueError("No gaps found in the selected review scope.")
    folder = _ensure_output_folder(output / "scenario_review")
    _write_review_tables(_collapse_report_rows(gaps), _collapse_report_rows(removed), folder)
    if plot_jobs:
        map_folder = _ensure_output_folder(folder / "maps")
        for pattern, scenario, baseline, records in plot_jobs:
            _plot_comparison(
                pattern,
                scenario,
                baseline,
                records,
                map_folder,
                _feet_factor(stops.crs),
            )
        logging.info("Wrote before/after maps to %s", map_folder)


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Run baseline QA plus optional removal comparisons; work from any folder."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if GTFS_PATH.startswith("Path\\To\\") or OUTPUT_FOLDER.startswith("Path\\To\\"):
        logging.error("Edit GTFS_PATH and OUTPUT_FOLDER in the configuration section.")
        return 2
    script_folder = (
        Path(globals().get("__file__", Path.cwd() / "gtfs_stop_spacing.py")).resolve().parent
    )
    gtfs_path, output_path = (
        Path(GTFS_PATH).expanduser(),
        Path(OUTPUT_FOLDER).expanduser(),
    )
    if not gtfs_path.is_absolute():
        gtfs_path = script_folder / gtfs_path
    if not output_path.is_absolute():
        output_path = script_folder / output_path
    _feet_factor(PROJECTED_CRS)
    logging.info("STEP 1: Reading GTFS tables.")
    dfs = _read_gtfs_tables(gtfs_path)
    _validate_columns(dfs)
    _prepare_tables(dfs)
    routes, trips = _filter_routes(dfs["routes"], dfs["trips"], INCLUDE_ROUTE_IDS, FILTER_OUT_LIST)
    if trips.empty:
        raise ValueError("No selected trips; check route_id include/exclude settings.")
    logging.info("STEP 2: Building shapes and validating scenario inputs.")
    lines = _build_shape_lines(dfs["shapes"], trips["shape_id"], PROJECTED_CRS)
    _validate_review(dfs, trips, lines, STOP_KEY, REMOVAL_SCENARIOS, REVIEW_STOP_KEYS)
    selected_stops = _build_stops_gdf(dfs["stops"], dfs["stop_times"], trips, routes, PROJECTED_CRS)
    patterns = _build_patterns_gdf(dfs["stop_times"], trips, routes, lines)
    output = _ensure_output_folder(output_path)
    logging.info("STEP 3: Calculating current and proposed spacing.")
    _run_spacing_review(patterns, selected_stops, output)
    if WRITE_LEGACY_GIS:
        logging.info("STEP 4: Writing existing GTFS GIS and nearby-stop QA outputs.")
        legacy = _ensure_output_folder(output / "current_gtfs_gis")
        all_stops = _build_stops_gdf(
            dfs["stops"],
            dfs["stop_times"],
            dfs["trips"],
            dfs["routes"],
            PROJECTED_CRS,
        )
        _export(
            gpd.GeoDataFrame(
                selected_stops.drop(columns="route_dirs"),
                geometry="geometry",
                crs=selected_stops.crs,
            ),
            legacy,
            "stops",
        )
        route_lines = _build_routes_gdf(lines, trips, routes, ROUTE_UNION)
        _export(route_lines, legacy, "routes")
        segments = _split_into_segments(patterns, selected_stops, PROJECTED_CRS)
        _export(segments, legacy, "segments")
        _export_segments_by_route_dir(segments, legacy)
        _flag_short_spacing(patterns, selected_stops, MIN_SPACING_FT, legacy / SPACING_LOG_FILE)
        _flag_long_spacing_csv(
            patterns,
            all_stops,
            LONG_SPACING_FT,
            NEAR_BUFFER_FT,
            legacy / LONG_SPACING_CSV_FILE,
        )
    logging.info("Finished. Outputs: %s", output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logging.error("\nUNEXPECTED ERROR: %s", exc)
        sys.exit(1)
