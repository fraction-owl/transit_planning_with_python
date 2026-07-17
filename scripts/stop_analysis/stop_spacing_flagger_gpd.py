"""GTFS to GIS pipeline for stop-spacing QA and segment analysis.

This module converts a General Transit Feed Specification (GTFS) package
(directory or .zip) into projected ESRI Shapefiles suitable for spatial
analysis and provides quality assurance (QA) checks on stop spacing.

Outputs:
• GeoDataFrames for served stops, route polylines, and stop-to-stop segments
• Shapefiles for use in GIS
• Logs flagging consecutive served stops that are spaced too closely
• CSVs identifying potential missed stops located between long stop-to-stop gaps
• Optional what-if QA for proposed stop relocations: a CSV of recomputed
  along-route spacing around each relocated stop with a compliance verdict,
  plus a PNG map per relocated stop (``proposed_maps/<stop_id>.png``)

The long-spacing check examines whether stops from other routes fall within
a specified buffer distance of unusually long segments and may merit further
review as possible missed service opportunities.

The proposed-relocation check accepts new coordinates for existing stops —
either as an in-line list or a .txt/.csv file — and reports whether the
spacing to each adjacent served stop, measured along the route polyline,
would comply with the short/long spacing thresholds after the move.

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
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split as split_line

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_PATH: str = r"Path\To\Your\GTFS_Data_Folder"  # folder or .zip
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

FILTER_OUT_LIST: list[str] = ["9999A", "9999B", "9999C"]
INCLUDE_ROUTE_IDS: list[str] = ["101", "202"]

ROUTE_UNION: bool = False
PROJECTED_CRS: str = "EPSG:2263"  # feet-based CRS

MIN_SPACING_FT: float = 400.0  # < this distance between served stops
SPACING_LOG_FILE: str = "short_spacing_segments.txt"

# Sets standards for route segments that are too long
# Best applied to local routes, use on express routes sparingly
LONG_SPACING_FT: float = 1_500.0  # > this distance between served stops …
NEAR_BUFFER_FT: float = 99.0  # … and a “missed” stop must lie ≤ this
LONG_SPACING_LOG_FILE: str = "long_spacing_segments.txt"
LONG_SPACING_CSV_FILE: str = "long_spacing_segments.csv"

# Proposed stop relocations (optional what-if QA)
# Provide new coordinates for existing stops either as an in-line list of
# (stop identifier, new_lat, new_lon) tuples or as the path to a .txt/.csv
# file with a header row, e.g. ``stop_id,new_lat,new_lon`` (comma- or
# tab-separated). Identifiers are matched against stop_id first, then
# stop_code (when stops.txt has one). Leave as None (or empty) to skip.
PROPOSED_STOPS: list[tuple[str, float, float]] | str | None = None
# PROPOSED_STOPS = [("1001", 38.8895, -77.0353)]
# PROPOSED_STOPS = r"Path\To\Your\proposed_stops.txt"
PROPOSED_SPACING_CSV_FILE: str = "proposed_spacing_compliance.csv"
PROPOSED_MAPS: bool = True  # write a PNG map per relocated stop

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def _ensure_output_folder(folder: str | Path) -> Path:
    """Create (if necessary) and return the output folder as a ``Path``."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _served_mask(df: pd.DataFrame, rid: str, drn: int) -> pd.Series:
    """Return boolean mask for rows whose list fields include rid/drn."""
    return df["route_id"].apply(lambda xs, rid=rid: rid in xs) & df["direction_id"].apply(
        lambda xs, drn=drn: drn in xs
    )


def _flag_long_spacing_csv(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    threshold_ft: float,
    near_buffer_ft: float,
    csv_path: Path,
    summary: bool = True,
) -> None:
    """Export a CSV of “missed” stops that fill unusually long gaps.

    A *long gap* is any consecutive pair of served stops on a given
    (route_id, direction_id) whose spacing exceeds *threshold_ft*.
    For every other-route stop that falls **inside** the gap and within
    *near_buffer_ft* of the polyline, a row is written containing:

    | route_id | route_short | direction_id | seg_len_ft | start_stop_id |
    | start_stop_name | end_stop_id | end_stop_name | flagged_stop_id |
    | flagged_stop_name | dist_to_route_ft |

    Parameters
    ----------
    routes_gdf, stops_gdf
        Projected GeoDataFrames created by :func:`_build_routes_gdf` and
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

    Notes:
    -----
    • The function silently skips shapes that have fewer than two served
      stops (nothing to measure).
    • CRS units are assumed feet if the EPSG contains *2263*, otherwise
      they are interpreted as metres and converted to feet.
    """
    crs_str: str = str(stops_gdf.crs) if stops_gdf.crs else ""
    ft_factor: float = 1.0 if "2263" in crs_str else 3.28084
    sindex = stops_gdf.sindex

    records: List[Dict[str, Any]] = []

    for _, row in routes_gdf.iterrows():
        rid: str = str(row.route_id)
        drn: int = int(row.direction_id)
        rshort: str | None = row.get("route_short_name")
        line: LineString = row.geometry

        # —— served stops on this route/direction ————————————————
        cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
        served = cand[_served_mask(cand, rid, drn)].copy()

        if len(served) < 2:
            continue

        served["dist_along"] = served.geometry.apply(line.project)
        served = (
            served.drop_duplicates("dist_along").sort_values("dist_along").reset_index(drop=True)
        )

        # —— check each consecutive pair ————————————————
        for i in range(len(served) - 1):
            s0, s1 = served.iloc[i], served.iloc[i + 1]
            seg_len_ft: float = (s1.dist_along - s0.dist_along) * ft_factor
            if seg_len_ft <= threshold_ft:
                continue

            # bounding envelope for spatial filter
            start_d, end_d = s0.dist_along, s1.dist_along
            sub_bounds = line.interpolate(start_d).bounds + line.interpolate(end_d).bounds
            minx, miny, maxx, maxy = (
                min(sub_bounds[0], sub_bounds[2]) - near_buffer_ft,
                min(sub_bounds[1], sub_bounds[3]) - near_buffer_ft,
                max(sub_bounds[0], sub_bounds[2]) + near_buffer_ft,
                max(sub_bounds[1], sub_bounds[3]) + near_buffer_ft,
            )

            # candidate “missed” stops from *other* routes
            maybe = stops_gdf.iloc[list(sindex.intersection((minx, miny, maxx, maxy)))]
            maybe = maybe[~_served_mask(maybe, rid, drn)]

            for _, st in maybe.iterrows():
                proj = line.project(st.geometry)
                if start_d < proj < end_d and st.geometry.distance(line) <= near_buffer_ft:
                    records.append(
                        {
                            "route_id": rid,
                            "route_short": rshort,
                            "direction_id": drn,
                            "seg_len_ft": round(seg_len_ft, 1),
                            "start_stop_id": s0.stop_id,
                            "start_stop_name": s0.stop_name,
                            "end_stop_id": s1.stop_id,
                            "end_stop_name": s1.stop_name,
                            "flagged_stop_id": st.stop_id,
                            "flagged_stop_name": st.stop_name,
                            "dist_to_route_ft": round(st.geometry.distance(line) * ft_factor, 1),
                        }
                    )

    # —— export ————————————————————————————————————————————————
    if not records:
        logging.info("No long-spacing issues found.")
        return

    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
    logging.info("Wrote long-spacing CSV → %s", csv_path.name)

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
    """
    filenames: Dict[str, str] = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
    }

    if gtfs_path.is_dir():
        return {k: pd.read_csv(gtfs_path / v) for k, v in filenames.items()}

    if gtfs_path.is_file() and gtfs_path.suffix.lower() == ".zip":
        logging.info("Detected GTFS zip – extracting to temporary directory …")
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs_path, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        return {k: pd.read_csv(root / v) for k, v in filenames.items()}

    raise ValueError("GTFS_PATH must be a folder or a .zip file.")


def _validate_columns(dfs: Dict[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` if any required GTFS column is missing."""
    required: Dict[str, set[str]] = {
        "stops": {"stop_id", "stop_lat", "stop_lon", "stop_name"},
        "routes": {"route_id", "route_short_name"},
        "trips": {"trip_id", "route_id", "shape_id", "direction_id"},
        "stop_times": {"trip_id", "stop_id"},
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


def _filter_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    include_ids: Sequence[str],
    exclude_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply include/exclude lists and return filtered ``routes`` and ``trips``."""
    routes_ok = routes.loc[~routes["route_id"].isin(exclude_ids)].copy()
    if include_ids:
        routes_ok = routes_ok.loc[routes_ok["route_id"].isin(include_ids)].copy()
    trips_ok = trips.loc[trips["route_id"].isin(routes_ok["route_id"])].copy()
    return routes_ok, trips_ok


def _build_stops_gdf(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of **served** stops with list fields for routes/directions."""
    served = stop_times.loc[stop_times["trip_id"].isin(trips["trip_id"])]
    stops = stops.loc[stops["stop_id"].isin(served["stop_id"])].copy()

    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    ).to_crs(crs)

    trip_attrs = trips[["trip_id", "route_id", "direction_id"]].merge(
        routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )
    merged = served[["trip_id", "stop_id"]].merge(trip_attrs, on="trip_id", how="left")

    agg = (
        merged.groupby("stop_id")[["route_id", "direction_id", "route_short_name"]]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    gdf = gdf.merge(agg, on="stop_id", how="left")

    logging.info("Stops GDF – kept %d served stops.", len(gdf))
    return gdf


def _build_routes_gdf(
    shapes: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
    union_shapes: bool,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame of polylines keyed by ``(route_id, direction_id)``."""
    shape_cols: list[str] = [
        "shape_id",
        "shape_pt_sequence",
        "shape_pt_lat",
        "shape_pt_lon",
    ]

    lines = (
        shapes[shape_cols]
        .sort_values(["shape_id", "shape_pt_sequence"])
        .groupby("shape_id")
        .apply(lambda g: LineString(zip(g.shape_pt_lon, g.shape_pt_lat)))
        .to_frame("geometry")
        .reset_index()
    )

    gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(crs)

    gdf = gdf.merge(
        trips.drop_duplicates("shape_id")[["shape_id", "route_id", "direction_id"]],
        on="shape_id",
        how="left",
    ).merge(routes, on="route_id", how="left")

    # ---- NEW ---------------------------------------------------------------
    before = len(gdf)
    gdf = gdf[gdf["direction_id"].notna()].copy()
    dropped = before - len(gdf)
    if dropped:
        logging.info(
            "Routes GDF – %d of %d shapes were missing `direction_id` and were skipped.",
            dropped,
            before,
        )
    # ------------------------------------------------------------------------

    if union_shapes:
        gdf = gdf.dissolve(
            by=["route_id", "direction_id"],
            as_index=False,
            aggfunc={"route_short_name": "first", "route_long_name": "first"},
        ).explode(ignore_index=True)

    logging.info("Routes GDF – built %d shapes.", len(gdf))
    return gdf


def _split_into_segments(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """Split each route polyline at its own stops and return segment GDF."""
    seg_records: list[dict[str, object]] = []
    sindex = stops_gdf.sindex

    for _, r in routes_gdf.iterrows():
        # -------------------------------------------------------------------
        if pd.isna(r.direction_id):  # extra safety – should not occur
            continue
        # -------------------------------------------------------------------

        line: LineString = r.geometry
        rid: str = str(r.route_id)
        drn: int = int(r.direction_id)

        cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
        cand = cand[_served_mask(cand, rid, drn)]
        if cand.empty:
            continue

        dists = np.array([line.project(pt) for pt in cand.geometry if isinstance(pt, Point)])
        uniq_dists = np.unique(dists)
        snap_pts: list[Point] = [line.interpolate(d) for d in uniq_dists]

        pieces = split_line(line, MultiPoint(snap_pts))
        geoms: Iterable[LineString]
        if isinstance(pieces, LineString):
            geoms = [pieces]
        else:
            geoms = (g for g in pieces.geoms if isinstance(g, LineString))

        for seg in geoms:
            if seg.length > 0:
                seg_records.append(
                    {
                        "route_id": rid,
                        "direction_id": drn,
                        "route_short": r.get("route_short_name"),
                        "geometry": seg,
                    }
                )

    seg_gdf = gpd.GeoDataFrame(seg_records, crs=crs)
    seg_gdf["length_ft"] = seg_gdf.length * (1.0 if "2263" in crs else 3.28084)
    logging.info("Segments GDF – generated %d pieces.", len(seg_gdf))
    return seg_gdf


def _export(gdf: gpd.GeoDataFrame, out_dir: Path, name: str) -> None:
    """Write *gdf* to ESRI Shapefile ``<out_dir>/<name>.shp``."""
    path = out_dir / f"{name}.shp"
    gdf.to_file(path)
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
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    threshold_ft: float,
    log_path: Path,
) -> None:
    """Write a log of consecutive stops spaced closer than *threshold_ft*.

    Stops are evaluated along each route polyline.
    """
    crs_str = str(stops_gdf.crs) if stops_gdf.crs is not None else ""
    factor_ft: float = 1.0 if "2263" in crs_str else 3.28084
    sindex = stops_gdf.sindex

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "route_id\tdirection_id\tbegin_stop_id\tbegin_stop_name\t"
            "end_stop_id\tend_stop_name\tspacing_ft\n"
        )

        for _, row in routes_gdf.iterrows():
            rid = str(row.route_id)
            drn = int(row.direction_id)
            line: LineString = row.geometry

            cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
            cand = cand[_served_mask(cand, rid, drn)].copy()

            if len(cand) < 2:
                continue

            cand["dist_along"] = cand.geometry.apply(line.project)
            cand = cand.drop_duplicates("dist_along").sort_values("dist_along")

            for i in range(len(cand) - 1):
                s0, s1 = cand.iloc[i], cand.iloc[i + 1]
                spacing_ft = (s1.dist_along - s0.dist_along) * factor_ft
                if spacing_ft < threshold_ft:
                    fh.write(
                        f"{rid}\t{drn}\t"
                        f"{s0.stop_id}\t{s0.stop_name}\t"
                        f"{s1.stop_id}\t{s1.stop_name}\t"
                        f"{spacing_ft:.1f}\n"
                    )

    logging.info("Wrote short-spacing log → %s", log_path.name)


# =============================================================================
# PROPOSED STOP RELOCATIONS (WHAT-IF QA)
# =============================================================================


def _load_proposed_stops(
    source: Sequence[tuple[str, float, float]] | str | Path | None,
) -> List[Tuple[str, float, float]]:
    """Normalize ``PROPOSED_STOPS`` into a list of (identifier, lat, lon).

    Args:
        source: Either an in-line sequence of ``(identifier, new_lat, new_lon)``
            tuples, a path to a delimited text file (comma- or tab-separated)
            with a header row naming an identifier column (``stop_id``,
            ``stop_code``, or ``stop``) plus latitude/longitude columns
            (``new_lat``/``new_lon``, ``stop_lat``/``stop_lon``, or
            ``lat``/``lon``), or None.

    Returns:
        List of ``(identifier, lat, lon)`` tuples. Empty when *source* is
        None or empty.

    Raises:
        ValueError: If the file lacks the required columns, an entry does not
            have exactly three fields, coordinates are not numeric, or a
            coordinate is outside the valid WGS84 range.
    """
    if source is None or (not isinstance(source, (str, Path)) and len(source) == 0):
        return []

    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, sep=None, engine="python", dtype=str, skipinitialspace=True)
        cols = {str(c).strip().lower(): c for c in df.columns}
        id_col = next((cols[k] for k in ("stop_id", "stop_code", "stop") if k in cols), None)
        lat_col = next((cols[k] for k in ("new_lat", "stop_lat", "lat") if k in cols), None)
        lon_col = next((cols[k] for k in ("new_lon", "stop_lon", "lon") if k in cols), None)
        if id_col is None or lat_col is None or lon_col is None:
            raise ValueError(
                f"Proposed-stops file {source!s} must have a header row with an "
                "identifier column (stop_id, stop_code, or stop) and coordinate "
                "columns (new_lat/new_lon, stop_lat/stop_lon, or lat/lon)."
            )
        raw_rows: Iterable[tuple[Any, Any, Any]] = zip(df[id_col], df[lat_col], df[lon_col])
    else:
        raw_rows = source  # type: ignore[assignment]

    out: List[Tuple[str, float, float]] = []
    for entry in raw_rows:
        try:
            ident, lat_raw, lon_raw = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Proposed-stop entry {entry!r} must have exactly three fields: "
                "(identifier, new_lat, new_lon)."
            ) from exc
        try:
            lat, lon = float(lat_raw), float(lon_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Proposed-stop entry for {ident!r} has non-numeric coordinates: "
                f"({lat_raw!r}, {lon_raw!r})."
            ) from exc
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(
                f"Proposed-stop entry for {ident!r} has out-of-range WGS84 "
                f"coordinates: ({lat}, {lon})."
            )
        out.append((str(ident).strip(), lat, lon))
    return out


def _resolve_proposed_stops(
    proposals: Sequence[Tuple[str, float, float]],
    stops_master: pd.DataFrame,
) -> Dict[str, Tuple[float, float]]:
    """Map proposal identifiers to canonical ``stop_id`` values.

    Identifiers are matched against ``stop_id`` first, then ``stop_code``
    (when that column exists in *stops_master*). Unmatched identifiers are
    logged and dropped; a repeated identifier overrides the earlier entry.

    Args:
        proposals: Output of :func:`_load_proposed_stops`.
        stops_master: The raw ``stops.txt`` DataFrame.

    Returns:
        Mapping ``stop_id → (new_lat, new_lon)``.
    """
    stop_ids: Set[str] = set(stops_master["stop_id"].astype(str))
    code_to_id: Dict[str, str] = {}
    if "stop_code" in stops_master.columns:
        codes = stops_master.loc[stops_master["stop_code"].notna(), ["stop_code", "stop_id"]]
        for code, sid in zip(codes["stop_code"].astype(str), codes["stop_id"].astype(str)):
            code_to_id.setdefault(code, sid)

    resolved: Dict[str, Tuple[float, float]] = {}
    for ident, lat, lon in proposals:
        sid = ident if ident in stop_ids else code_to_id.get(ident)
        if sid is None:
            logging.warning(
                "Proposed stop %r not found in stops.txt (checked stop_id and stop_code); "
                "skipping.",
                ident,
            )
            continue
        if sid in resolved:
            logging.warning("Proposed stop %s appears more than once; keeping the last entry.", sid)
        resolved[sid] = (lat, lon)
    return resolved


def _apply_proposed_coords(
    stops_gdf: gpd.GeoDataFrame,
    moves: Dict[str, Tuple[float, float]],
    crs: str,
) -> gpd.GeoDataFrame:
    """Return a copy of *stops_gdf* with geometry replaced for relocated stops.

    Args:
        stops_gdf: Projected served-stops layer.
        moves: Mapping ``stop_id → (new_lat, new_lon)`` in WGS84.
        crs: Target projected CRS of *stops_gdf*.

    Returns:
        A new GeoDataFrame; stops absent from *moves* keep their geometry.
    """
    out = stops_gdf.copy()
    if not moves:
        return out

    sids = out["stop_id"].astype(str)
    for sid, (lat, lon) in moves.items():
        mask = sids == sid
        if not mask.any():
            logging.warning(
                "Proposed stop %s is not served by any selected route/direction; "
                "its relocation cannot be evaluated here.",
                sid,
            )
            continue
        new_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
        out.loc[mask, "geometry"] = new_geom
    return out


def _ordered_served_stops(
    line: LineString,
    stops_gdf: gpd.GeoDataFrame,
    rid: str,
    drn: int,
) -> pd.DataFrame:
    """Return stops served by (*rid*, *drn*) ordered along *line*.

    The result carries a ``dist_along`` column (projected line measure) and
    drops stops that share an identical measure, mirroring the behaviour of
    the short-spacing QA.
    """
    served = stops_gdf[_served_mask(stops_gdf, rid, drn)].copy()
    if served.empty:
        served["dist_along"] = pd.Series(dtype=float)
        return served
    served["dist_along"] = served.geometry.apply(line.project)
    return served.drop_duplicates("dist_along").sort_values("dist_along").reset_index(drop=True)


def _evaluate_proposed_spacing(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    proposed_gdf: gpd.GeoDataFrame,
    moved_ids: Set[str],
    min_spacing_ft: float,
    long_spacing_ft: float,
) -> pd.DataFrame:
    """Recompute along-route spacing around relocated stops.

    For every (route_id, direction_id) polyline that serves a relocated stop,
    each consecutive served-stop pair touching a relocated stop is measured
    along the polyline under both the original and the proposed coordinates.

    Args:
        routes_gdf: Projected route polylines.
        stops_gdf: Served stops with original geometry.
        proposed_gdf: Served stops with relocated geometry
            (output of :func:`_apply_proposed_coords`).
        moved_ids: ``stop_id`` values that were relocated.
        min_spacing_ft: Spacing below this is flagged ``too short``.
        long_spacing_ft: Spacing above this is flagged ``too long``.

    Returns:
        One row per affected consecutive pair with columns: route_id,
        route_short, direction_id, moved_stop_id, begin/end stop id + name,
        spacing_ft_before, spacing_ft_after, verdict, compliant.
    """
    columns = [
        "route_id",
        "route_short",
        "direction_id",
        "moved_stop_id",
        "begin_stop_id",
        "begin_stop_name",
        "end_stop_id",
        "end_stop_name",
        "spacing_ft_before",
        "spacing_ft_after",
        "verdict",
        "compliant",
    ]
    crs_str: str = str(stops_gdf.crs) if stops_gdf.crs else ""
    ft_factor: float = 1.0 if "2263" in crs_str else 3.28084

    records: List[Dict[str, Any]] = []
    for _, row in routes_gdf.iterrows():
        rid: str = str(row.route_id)
        drn: int = int(row.direction_id)
        line: LineString = row.geometry

        after = _ordered_served_stops(line, proposed_gdf, rid, drn)
        if len(after) < 2:
            continue
        moved_here = set(after["stop_id"].astype(str)) & moved_ids
        if not moved_here:
            continue

        before = _ordered_served_stops(line, stops_gdf, rid, drn)
        before_pos: Dict[str, float] = dict(
            zip(before["stop_id"].astype(str), before["dist_along"])
        )

        for i in range(len(after) - 1):
            s0, s1 = after.iloc[i], after.iloc[i + 1]
            pair_moved = [sid for sid in (str(s0.stop_id), str(s1.stop_id)) if sid in moved_here]
            if not pair_moved:
                continue

            spacing_after = (s1.dist_along - s0.dist_along) * ft_factor
            b0 = before_pos.get(str(s0.stop_id))
            b1 = before_pos.get(str(s1.stop_id))
            if b0 is not None and b1 is not None:
                spacing_before = abs(b1 - b0) * ft_factor
            else:
                spacing_before = np.nan

            if spacing_after < min_spacing_ft:
                verdict = "too short"
            elif spacing_after > long_spacing_ft:
                verdict = "too long"
            else:
                verdict = "OK"

            records.append(
                {
                    "route_id": rid,
                    "route_short": row.get("route_short_name"),
                    "direction_id": drn,
                    "moved_stop_id": ",".join(pair_moved),
                    "begin_stop_id": s0.stop_id,
                    "begin_stop_name": s0.stop_name,
                    "end_stop_id": s1.stop_id,
                    "end_stop_name": s1.stop_name,
                    "spacing_ft_before": round(spacing_before, 1),
                    "spacing_ft_after": round(spacing_after, 1),
                    "verdict": verdict,
                    "compliant": verdict == "OK",
                }
            )

    return pd.DataFrame.from_records(records, columns=columns)


def _plot_proposed_stops(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    proposed_gdf: gpd.GeoDataFrame,
    moves: Dict[str, Tuple[float, float]],
    report: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write one PNG map per relocated stop to ``<out_dir>/proposed_maps``.

    Each map shows the serving route polylines, nearby served stops, the
    original (red) and proposed (green) stop positions joined by a dashed
    arrow, and the recomputed spacing to each adjacent stop annotated at the
    segment midpoint (green when compliant, red otherwise).
    """
    map_dir = out_dir / "proposed_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    orig_by_id: Dict[str, Point] = dict(zip(stops_gdf["stop_id"].astype(str), stops_gdf.geometry))
    prop_by_id: Dict[str, Point] = dict(
        zip(proposed_gdf["stop_id"].astype(str), proposed_gdf.geometry)
    )
    name_by_id: Dict[str, str] = dict(
        zip(stops_gdf["stop_id"].astype(str), stops_gdf["stop_name"].astype(str))
    )

    for sid in moves:
        old_pt = orig_by_id.get(sid)
        new_pt = prop_by_id.get(sid)
        if old_pt is None or new_pt is None:
            continue

        stop_row = stops_gdf.loc[stops_gdf["stop_id"].astype(str) == sid].iloc[0]
        serving = routes_gdf[
            routes_gdf.apply(
                lambda r, xs=stop_row: (
                    str(r.route_id) in [str(x) for x in xs.route_id]
                    and int(r.direction_id) in [int(x) for x in xs.direction_id]
                ),
                axis=1,
            )
        ]

        rows = report[
            report["moved_stop_id"].astype(str).str.split(",").apply(lambda xs, s=sid: s in xs)
        ]

        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

        if not serving.empty:
            serving.plot(ax=ax, color="0.6", linewidth=1.0, zorder=1)
            for rid, drn in serving[["route_id", "direction_id"]].itertuples(index=False):
                neighbors = proposed_gdf[_served_mask(proposed_gdf, str(rid), int(drn))]
                if not neighbors.empty:
                    neighbors.plot(ax=ax, color="0.4", markersize=8, zorder=2)

        ax.plot([old_pt.x], [old_pt.y], "o", color="red", markersize=7, zorder=4)
        ax.plot([new_pt.x], [new_pt.y], "o", color="green", markersize=7, zorder=4)
        ax.annotate(
            "",
            xy=(new_pt.x, new_pt.y),
            xytext=(old_pt.x, old_pt.y),
            arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "black"},
            zorder=3,
        )

        focus_xs: List[float] = [old_pt.x, new_pt.x]
        focus_ys: List[float] = [old_pt.y, new_pt.y]
        for rec in rows.itertuples(index=False):
            p0 = prop_by_id.get(str(rec.begin_stop_id))
            p1 = prop_by_id.get(str(rec.end_stop_id))
            if p0 is None or p1 is None:
                continue
            focus_xs.extend([p0.x, p1.x])
            focus_ys.extend([p0.y, p1.y])
            ax.annotate(
                f"{rec.spacing_ft_after:,.0f} ft ({rec.verdict})",
                xy=((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="green" if rec.compliant else "red",
                zorder=5,
            )

        n_bad = int((~rows["compliant"]).sum()) if not rows.empty else 0
        status = "all spacing OK" if n_bad == 0 else f"{n_bad} non-compliant segment(s)"
        ax.set_title(
            f"Proposed move: {name_by_id.get(sid, '')} ({sid}) — {status}",
            fontsize=8,
        )
        # Zoom to the relocation and its adjacent segments; without this the
        # view spans the whole route and the move is unreadable.
        span = max(max(focus_xs) - min(focus_xs), max(focus_ys) - min(focus_ys))
        pad = max(0.25 * span, 200.0)
        ax.set_xlim(min(focus_xs) - pad, max(focus_xs) + pad)
        ax.set_ylim(min(focus_ys) - pad, max(focus_ys) + pad)
        ax.set_aspect("equal")
        ax.set_axis_off()

        fig.tight_layout()
        fig.savefig(map_dir / f"{sid}.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        logging.info("Wrote proposed-stop map → proposed_maps/%s.png", sid)


def _run_proposed_stops_qa(
    proposed_source: Sequence[tuple[str, float, float]] | str | None,
    stops_master: pd.DataFrame,
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    crs: str,
    min_spacing_ft: float,
    long_spacing_ft: float,
    csv_path: Path,
    export_maps: bool,
) -> None:
    """Load, evaluate, and report proposed stop relocations (no-op when unset).

    Args:
        proposed_source: The ``PROPOSED_STOPS`` configuration value.
        stops_master: Raw ``stops.txt`` DataFrame (for stop_id/stop_code lookup).
        routes_gdf: Projected route polylines.
        stops_gdf: Projected served-stops layer (filtered set).
        crs: Projected CRS of the layers above.
        min_spacing_ft: Short-spacing threshold.
        long_spacing_ft: Long-spacing threshold.
        csv_path: Destination for the compliance CSV.
        export_maps: If True, also write a PNG map per relocated stop.
    """
    proposals = _load_proposed_stops(proposed_source)
    if not proposals:
        return

    moves = _resolve_proposed_stops(proposals, stops_master)
    if not moves:
        logging.warning("No proposed stops could be matched; skipping what-if QA.")
        return

    proposed_gdf = _apply_proposed_coords(stops_gdf, moves, crs)
    report = _evaluate_proposed_spacing(
        routes_gdf,
        stops_gdf,
        proposed_gdf,
        set(moves),
        min_spacing_ft,
        long_spacing_ft,
    )

    if report.empty:
        logging.warning(
            "Proposed stops matched no served route/direction in the selected set; "
            "no compliance rows to report."
        )
        return

    report.to_csv(csv_path, index=False)
    n_bad = int((~report["compliant"]).sum())
    logging.info(
        "Wrote proposed-spacing CSV → %s (%d segment(s), %d non-compliant).",
        csv_path.name,
        len(report),
        n_bad,
    )

    if export_maps:
        _plot_proposed_stops(routes_gdf, stops_gdf, proposed_gdf, moves, report, csv_path.parent)


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
        Target projected CRS (feet-based).

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

    # -----------------------------------------------------------------
    # 0·1  Route / trip filtering
    # -----------------------------------------------------------------
    routes_df, trips_df = _filter_routes(
        dfs["routes"], dfs["trips"], INCLUDE_ROUTE_IDS, FILTER_OUT_LIST
    )

    out_dir = _ensure_output_folder(OUTPUT_FOLDER)

    # -----------------------------------------------------------------
    # STEP 1  Build stop layers
    # -----------------------------------------------------------------
    logging.info("STEP 1  Building stop layers …")
    all_stops_gdf, stops_gdf = _build_stop_layers(dfs, trips_df, routes_df, PROJECTED_CRS)
    _export(stops_gdf, out_dir, "stops")  # export only the filtered set

    # -----------------------------------------------------------------
    # STEP 2  Build route polylines
    # -----------------------------------------------------------------
    logging.info("STEP 2  Building routes shapefile …")
    routes_gdf = _build_routes_gdf(dfs["shapes"], trips_df, routes_df, PROJECTED_CRS, ROUTE_UNION)
    _export(routes_gdf, out_dir, "routes")

    # -----------------------------------------------------------------
    # STEP 3  Split polylines into stop-to-stop segments
    # -----------------------------------------------------------------
    logging.info("STEP 3  Splitting routes into stop-to-stop segments …")
    segs_gdf = _split_into_segments(routes_gdf, stops_gdf, PROJECTED_CRS)
    _export(segs_gdf, out_dir, "segments")  # master file
    _export_segments_by_route_dir(segs_gdf, out_dir)  # per-route files

    # -----------------------------------------------------------------
    # STEP 4  Short-spacing QA
    # -----------------------------------------------------------------
    logging.info("STEP 4  Flagging closely-spaced stops …")
    _flag_short_spacing(
        routes_gdf,
        stops_gdf,  # filtered layer
        MIN_SPACING_FT,
        out_dir / SPACING_LOG_FILE,
    )

    # -----------------------------------------------------------------
    # STEP 5  Long-spacing QA (needs *all* stops) – CSV export
    # -----------------------------------------------------------------
    logging.info("STEP 5  Flagging long-spacing segments …")
    _flag_long_spacing_csv(
        routes_gdf,
        all_stops_gdf,  # unfiltered layer
        LONG_SPACING_FT,
        NEAR_BUFFER_FT,
        out_dir / LONG_SPACING_CSV_FILE,
    )

    # -----------------------------------------------------------------
    # STEP 6  Proposed stop relocations – what-if QA (optional)
    # -----------------------------------------------------------------
    if PROPOSED_STOPS:
        logging.info("STEP 6  Evaluating proposed stop relocations …")
        _run_proposed_stops_qa(
            PROPOSED_STOPS,
            dfs["stops"],
            routes_gdf,
            stops_gdf,
            PROJECTED_CRS,
            MIN_SPACING_FT,
            LONG_SPACING_FT,
            out_dir / PROPOSED_SPACING_CSV_FILE,
            export_maps=PROPOSED_MAPS,
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
