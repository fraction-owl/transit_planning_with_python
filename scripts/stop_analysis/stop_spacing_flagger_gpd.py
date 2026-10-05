"""GTFS to GIS pipeline for stop-spacing QA and segment analysis.

This module converts a General Transit Feed Specification (GTFS) package
(directory or .zip) into projected ESRI Shapefiles suitable for spatial
analysis and provides quality assurance (QA) checks on stop spacing.

Outputs:
• GeoDataFrames for served stops, route polylines, and stop-to-stop segments
• Shapefiles for use in GIS
• Logs flagging consecutive served stops that are spaced too closely
• CSVs identifying potential missed stops located between long stop-to-stop gaps

The long-spacing check examines whether stops from other routes fall within
a specified buffer distance of unusually long segments and may merit further
review as possible missed service opportunities.

Spacing is measured along each distinct stopping pattern: the ordered stops
of a trip in ``stop_times.txt``, on the shape that trip uses. Express and
local trips sharing a shape are measured separately, and a stop visited twice
(a loop) keeps both visits. Each stop is placed along the shape from
``shape_dist_traveled`` when both files provide it; otherwise the stops are
matched to the shape in trip order.

Typical usage:
Update the paths in the CONFIGURATION section and run from a shell or a
Jupyter notebook.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Point
from shapely.ops import substring

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_PATH: str = r"Path\To\Your\GTFS_Data_Folder"  # folder or .zip
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

FILTER_OUT_LIST: list[str] = ["9999A", "9999B", "9999C"]
INCLUDE_ROUTE_IDS: list[str] = ["101", "202"]

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
        out[np.array([i for i, _ in usable])[ok]] = np.maximum.accumulate(along[ok])
    return out


def _match_in_order(vertices: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def _on_segments(pt: np.ndarray, segs: np.ndarray | slice) -> Tuple[np.ndarray, np.ndarray]:
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
    stops_by_id: gpd.GeoDataFrame,
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
        {"stop_id": stop_ids, "stop_name": stops["stop_name"].to_numpy(), "dist_along": dists}
    )
    return located.dropna(subset=["dist_along"]).reset_index(drop=True)


def _consecutive_stops(
    patterns_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    max_offset: float,
) -> Iterator[Tuple[pd.Series, pd.Series, pd.Series]]:
    """Yield ``(pattern, stop, next_stop)`` for consecutive stops of each pattern.

    A pair repeated by another pattern of the same route, direction and shape
    (e.g. a local and an express sharing a stretch) is yielded once.
    """
    stops_by_id = stops_gdf.drop_duplicates("stop_id").set_index("stop_id")
    seen: Set[Tuple[Any, ...]] = set()
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

    records: List[Dict[str, Any]] = []

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
        logging.info("No long-spacing issues found; wrote empty %s.", csv_path.name)

    # —— optional one-line summary ————————————————————————————
    if summary:
        flagged: Set[Tuple[str, int]] = {(rec["route_id"], rec["direction_id"]) for rec in records}
        summ_path = csv_path.with_name(f"{csv_path.stem}_summary.txt")
        with summ_path.open("w", encoding="utf-8") as fh:
            fh.write("route_id\tdirection_id\n")
            for rid, drn in sorted(flagged):
                fh.write(f"{rid}\t{drn}\n")
        logging.info("Wrote summary → %s", summ_path.name)


def _read_gtfs_tables(gtfs_path: Path) -> Dict[str, pd.DataFrame]:
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
    filenames: Dict[str, str] = {
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
        logging.info("Detected GTFS zip – extracting to temporary directory …")
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs_path, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        return {k: _read(root / v) for k, v in filenames.items()}

    raise ValueError("GTFS_PATH must be a folder or a .zip file.")


def _validate_columns(dfs: Dict[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` if any required GTFS column is missing."""
    required: Dict[str, set[str]] = {
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
    dfs: Dict[str, pd.DataFrame],
    unknown_direction: int = UNKNOWN_DIRECTION_ID,
) -> None:
    """Convert the numeric GTFS fields in place; IDs stay text.

    Unparseable numbers become NaN. ``trips.direction_id`` becomes an int,
    with *unknown_direction* for trips whose value is blank, invalid or
    absent (the field is optional in GTFS).
    """
    numeric: Dict[str, list[str]] = {
        "stops": ["stop_lat", "stop_lon"],
        "stop_times": ["stop_sequence", "shape_dist_traveled"],
        "shapes": ["shape_pt_sequence", "shape_pt_lat", "shape_pt_lon", "shape_dist_traveled"],
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
            "%d of %d trips have no valid direction_id; analyzing them as direction %d.",
            unknown.sum(),
            len(trips),
            unknown_direction,
        )
    trips["direction_id"] = direction.where(~unknown, unknown_direction).astype(int)


def _filter_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    include_ids: Sequence[str | int],
    exclude_ids: Sequence[str | int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    return gdf


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
    return gdf


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
    ):
        return tuple(np.full(len(stop_along), np.nan))
    steps = np.diff(coords[:, :2], axis=0)
    vertex_along = np.concatenate(([0.0], np.cumsum(np.hypot(steps[:, 0], steps[:, 1]))))
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
    patterns = per_trip.drop_duplicates(["route_id", "direction_id", "shape_id", "stop_ids"])

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

    logging.info("Patterns GDF – %d stopping patterns from %d trips.", len(gdf), len(per_trip))
    return gdf


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
    seen: Set[Tuple[Any, ...]] = set()

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
    for (rid, drn), grp in seg_gdf.groupby(["route_id", "direction_id"]):
        suffix = f"dir{drn}"
        fname = f"{rid}_{suffix}.shp"
        grp_gdf: gpd.GeoDataFrame = grp  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
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
    dfs: Dict[str, pd.DataFrame],
    trips_selected: pd.DataFrame,
    routes_selected: pd.DataFrame,
    crs: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:  # noqa: D401
    """Run the entire GTFS-to-GIS pipeline with both spacing QA checks.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if (
        GTFS_PATH == r"Path\To\Your\GTFS_Data_Folder"
        or OUTPUT_FOLDER == r"Path\To\Your\Output_Folder"
    ):
        logging.warning(
            "GTFS_PATH and/or OUTPUT_FOLDER are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2
    try:
        feet_factor = _feet_factor(PROJECTED_CRS)
    except ValueError as err:
        logging.error("%s", err)
        return 2
    logging.info("Using CRS: %s (1 unit ≈ %.3f ft)", PROJECTED_CRS, feet_factor)
    # -----------------------------------------------------------------
    # STEP 0  Read GTFS tables and validate
    # -----------------------------------------------------------------
    logging.info("STEP 0  Reading GTFS tables …")
    gtfs_path = Path(GTFS_PATH)
    dfs = _read_gtfs_tables(gtfs_path)

    try:
        _validate_columns(dfs)
    except ValueError as err:
        logging.error("\nERROR – invalid GTFS feed:\n%s", err)
        return 1
    _prepare_tables(dfs)

    # -----------------------------------------------------------------
    # 0·1  Route / trip filtering
    # -----------------------------------------------------------------
    routes_df, trips_df = _filter_routes(
        dfs["routes"], dfs["trips"], INCLUDE_ROUTE_IDS, FILTER_OUT_LIST
    )
    if trips_df.empty:
        logging.error(
            "No trips left after applying INCLUDE_ROUTE_IDS / FILTER_OUT_LIST; "
            "check that the IDs match route_id values in routes.txt."
        )
        return 2

    out_dir = _ensure_output_folder(OUTPUT_FOLDER)

    # -----------------------------------------------------------------
    # STEP 1  Build stop layers
    # -----------------------------------------------------------------
    logging.info("STEP 1  Building stop layers …")
    all_stops_gdf, stops_gdf = _build_stop_layers(dfs, trips_df, routes_df, PROJECTED_CRS)
    # export only the filtered set; route_dirs is an internal lookup
    _export(stops_gdf.drop(columns="route_dirs"), out_dir, "stops")

    # -----------------------------------------------------------------
    # STEP 2  Build route polylines and stopping patterns
    # -----------------------------------------------------------------
    logging.info("STEP 2  Building routes shapefile and stopping patterns …")
    shape_lines = _build_shape_lines(dfs["shapes"], trips_df["shape_id"], PROJECTED_CRS)
    routes_gdf = _build_routes_gdf(shape_lines, trips_df, routes_df, ROUTE_UNION)
    _export(routes_gdf, out_dir, "routes")
    patterns_gdf = _build_patterns_gdf(dfs["stop_times"], trips_df, routes_df, shape_lines)

    # -----------------------------------------------------------------
    # STEP 3  Split polylines into stop-to-stop segments
    # -----------------------------------------------------------------
    logging.info("STEP 3  Splitting routes into stop-to-stop segments …")
    segs_gdf = _split_into_segments(
        patterns_gdf, stops_gdf, PROJECTED_CRS, max_offset_ft=SERVED_STOP_MAX_OFFSET_FT
    )
    _export(segs_gdf, out_dir, "segments")  # master file
    _export_segments_by_route_dir(segs_gdf, out_dir)  # per-route files

    # -----------------------------------------------------------------
    # STEP 4  Short-spacing QA
    # -----------------------------------------------------------------
    logging.info("STEP 4  Flagging closely-spaced stops …")
    _flag_short_spacing(
        patterns_gdf,
        stops_gdf,  # filtered layer
        MIN_SPACING_FT,
        out_dir / SPACING_LOG_FILE,
        max_offset_ft=SERVED_STOP_MAX_OFFSET_FT,
    )

    # -----------------------------------------------------------------
    # STEP 5  Long-spacing QA (needs *all* stops) – CSV export
    # -----------------------------------------------------------------
    logging.info("STEP 5  Flagging long-spacing segments …")
    _flag_long_spacing_csv(
        patterns_gdf,
        all_stops_gdf,  # unfiltered layer
        LONG_SPACING_FT,
        NEAR_BUFFER_FT,
        out_dir / LONG_SPACING_CSV_FILE,
        max_offset_ft=SERVED_STOP_MAX_OFFSET_FT,
    )

    logging.info("\nAll done! Outputs in: %s", out_dir)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logging.error("\nUNEXPECTED ERROR: %s", exc)
        sys.exit(1)
