"""GTFS to GIS pipeline for stop-spacing QA and segment analysis (ArcPy version).

This module converts a General Transit Feed Specification (GTFS) package
(directory or .zip) into projected ESRI Shapefiles suitable for spatial analysis
and provides quality assurance (QA) checks on stop spacing.

Outputs:
* Shapefiles for served stops, route polylines, and stop-to-stop segments
* Logs flagging consecutive served stops that are spaced too closely
* CSVs identifying potential “missed” stops located between long stop-to-stop gaps
* Optional what-if QA for proposed stop relocations: a CSV of recomputed
  along-route spacing around each relocated stop with a compliance verdict,
  plus a PNG map per relocated stop (``proposed_maps/<stop_id>.png``)

The long-spacing check examines whether stops from other routes fall within a
specified buffer distance of unusually long segments and may merit further
review as possible missed service opportunities.

The proposed-relocation check accepts new coordinates for existing stops —
either as an in-line list or a .txt/.csv file — and reports whether the
spacing to each adjacent served stop, measured along the route polyline,
would comply with the short/long spacing thresholds after the move.

Typical usage:
Update the paths in the CONFIGURATION section and run from ArcGIS Pro's Python
window or a shell whose environment provides ``arcpy`` (an ArcGIS Pro install).
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence, Set, Tuple

import arcpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# GTFS source – folder containing *.txt or a .zip GTFS package.
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"

# Output folder for shapefiles and QA logs (NOT a geodatabase).
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

# Route filtering
FILTER_OUT_LIST: list[str] = []
INCLUDE_ROUTE_IDS: list[str] = ["101", "202", "303"]  # empty list → all routes except filtered-out

# Route geometry options
ROUTE_UNION: bool = False

# Projected CRS – should be feet-based if you want spacing_ft directly in feet.
# Example: 2240 = NAD83 / Maryland (ftUS)
PROJECTED_WKID: int = 2240

# Short-spacing QA – “too close” consecutive served stops along a route
MIN_SPACING_FT: float = 400.0
SPACING_LOG_FILE: str = "short_spacing_segments.txt"

# Long-spacing QA – “too long” gaps and potential missed stops
LONG_SPACING_FT: float = 1_500.0
NEAR_BUFFER_FT: float = 99.0
LONG_SPACING_LOG_FILE: str = "long_spacing_segments.txt"  # currently unused (CSV + summary)
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
# HELPERS – I/O AND BASIC GTFS HANDLING
# =============================================================================


def _ensure_output_folder(folder: str | Path) -> Path:
    """Create (if necessary) and return the output folder as a Path."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_gtfs_tables(gtfs_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Load the five core GTFS tables into DataFrames.

    Args:
        gtfs_path: Path to either a directory containing *.txt files or a .zip GTFS.

    Returns:
        Mapping of table name to DataFrame with keys:
        "stops", "routes", "trips", "stop_times", "shapes".

    Raises:
        ValueError: If the path is neither a folder nor a .zip GTFS.
    """
    gtfs = Path(gtfs_path)
    filenames: Dict[str, str] = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
    }

    if gtfs.is_dir():
        logging.info("Detected GTFS directory at %s", gtfs)
        return {k: pd.read_csv(gtfs / v) for k, v in filenames.items()}

    if gtfs.is_file() and gtfs.suffix.lower() == ".zip":
        logging.info("Detected GTFS zip at %s – extracting to temporary directory …", gtfs)
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        tables = {k: pd.read_csv(root / v) for k, v in filenames.items()}
        return tables

    raise ValueError("GTFS_PATH must be a folder or a .zip file.")


def _validate_columns(dfs: Dict[str, pd.DataFrame]) -> None:
    """Raise ValueError if any required GTFS column is missing."""
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
    """Apply include/exclude lists and return filtered routes and trips.

    Args:
        routes: routes.txt DataFrame.
        trips: trips.txt DataFrame.
        include_ids: Route IDs to include. Empty means include all.
        exclude_ids: Route IDs to drop.

    Returns:
        (routes_filtered, trips_filtered)
    """
    routes_ok = routes.loc[~routes["route_id"].isin(exclude_ids)].copy()
    if include_ids:
        routes_ok = routes_ok.loc[routes_ok["route_id"].isin(include_ids)].copy()

    trips_ok = trips.loc[trips["route_id"].isin(routes_ok["route_id"])].copy()
    return routes_ok, trips_ok


def _get_projected_sr(wkid: int) -> arcpy.SpatialReference:
    """Return the projected spatial reference for the given WKID."""
    sr = arcpy.SpatialReference(wkid)
    if sr.name == "Unknown":
        raise ValueError(f"Spatial reference WKID {wkid} is not recognized.")
    return sr


def _feet_factor(sr: arcpy.SpatialReference) -> float:
    """Return factor to convert from SR linear units to feet."""
    name = (sr.linearUnitName or "").lower()
    if "foot" in name:
        return 1.0
    return 3.28084


def _is_empty_polyline(geom: arcpy.Polyline | None) -> bool:
    """Return True if a Polyline is None, has no points, or has zero length."""
    if geom is None:
        return True
    if getattr(geom, "pointCount", 0) == 0:
        return True
    if getattr(geom, "length", 0.0) == 0.0:
        return True
    return False


# =============================================================================
# GEOMETRY BUILDERS (ARCPY)
# =============================================================================


def _build_stop_geometries(
    stops_df: pd.DataFrame,
    projected_sr: arcpy.SpatialReference,
) -> Dict[str, arcpy.PointGeometry]:
    """Build projected PointGeometry objects keyed by stop_id.

    Args:
        stops_df: stops.txt DataFrame (must contain stop_id, stop_lat, stop_lon).
        projected_sr: Target projected spatial reference.

    Returns:
        Mapping stop_id → PointGeometry (projected).
    """
    required = {"stop_id", "stop_lat", "stop_lon"}
    missing = required - set(stops_df.columns)
    if missing:
        raise ValueError(f"stops.txt missing columns: {', '.join(sorted(missing))}")

    wgs84 = arcpy.SpatialReference(4326)
    out: Dict[str, arcpy.PointGeometry] = {}

    for _, row in stops_df.iterrows():
        stop_id = str(row["stop_id"])
        try:
            lon = float(row["stop_lon"])
            lat = float(row["stop_lat"])
        except (TypeError, ValueError):
            logging.warning("Skipping stop %s due to invalid coords.", stop_id)
            continue

        pt = arcpy.Point(lon, lat)
        pt_geom = arcpy.PointGeometry(pt, wgs84).projectAs(projected_sr)
        out[stop_id] = pt_geom

    logging.info("Built %d stop point geometries.", len(out))
    return out


def _build_shape_geometries(
    shapes_df: pd.DataFrame,
    projected_sr: arcpy.SpatialReference,
) -> Dict[str, arcpy.Polyline]:
    """Build projected Polyline geometries keyed by shape_id.

    Args:
        shapes_df: shapes.txt DataFrame.
        projected_sr: Target projected spatial reference.

    Returns:
        Mapping shape_id → Polyline in projected_sr.
    """
    required = {
        "shape_id",
        "shape_pt_sequence",
        "shape_pt_lat",
        "shape_pt_lon",
    }
    missing = required - set(shapes_df.columns)
    if missing:
        raise ValueError(f"shapes.txt missing columns: {', '.join(sorted(missing))}")

    wgs84 = arcpy.SpatialReference(4326)
    out: Dict[str, arcpy.Polyline] = {}

    shapes = shapes_df.copy()
    shapes["shape_pt_sequence"] = pd.to_numeric(
        shapes["shape_pt_sequence"],
        errors="coerce",
    )

    shapes_sorted = shapes.sort_values(["shape_id", "shape_pt_sequence"])

    for shape_id, group in shapes_sorted.groupby("shape_id"):
        array = arcpy.Array()
        for _, row in group.iterrows():
            try:
                lon = float(row["shape_pt_lon"])
                lat = float(row["shape_pt_lat"])
            except (TypeError, ValueError):
                logging.warning("Skipping bad shape point in shape_id=%s", shape_id)
                continue
            pt = arcpy.Point(lon, lat)
            array.add(pt)

        if array.count < 2:
            logging.debug("Shape %s has fewer than 2 points; skipping.", shape_id)
            continue

        line_wgs = arcpy.Polyline(array, wgs84)
        line_proj = line_wgs.projectAs(projected_sr)
        out[str(shape_id)] = line_proj

    logging.info("Built %d shape polylines.", len(out))
    return out


def _build_routes_from_shapes(
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    shape_geoms: Dict[str, arcpy.Polyline],
    union_shapes: bool,
) -> List[Dict[str, Any]]:
    """Build route polylines keyed by (route_id, direction_id).

    Args:
        trips_df: Filtered trips.txt DataFrame.
        routes_df: Filtered routes.txt DataFrame.
        shape_geoms: Mapping shape_id → Polyline (projected).
        union_shapes: If True, union all shapes for (route_id, direction_id)
            into a single polyline; otherwise keep individual shapes.

    Returns:
        List of route records:
        {
            "route_id": str,
            "direction_id": int,
            "route_short": str | None,
            "geometry": arcpy.Polyline,
        }
    """
    trips = trips_df.copy()
    trips = trips.dropna(subset=["direction_id", "shape_id"]).copy()

    # NEW: collapse to unique combinations so we do not duplicate per trip.
    trips = trips.drop_duplicates(subset=["route_id", "direction_id", "shape_id"]).copy()
    logging.info(
        "Routes – using %d unique (route_id, direction_id, shape_id) combinations.",
        len(trips),
    )

    # Route short name lookup
    route_short_lookup = (
        routes_df[["route_id", "route_short_name"]]
        .assign(route_id_str=lambda df: df["route_id"].astype(str))
        .set_index("route_id_str")["route_short_name"]
        .to_dict()
    )

    records: List[Dict[str, Any]] = []

    if not union_shapes:
        for _, row in trips.iterrows():
            shape_id = str(row["shape_id"])
            line = shape_geoms.get(shape_id)
            if line is None:
                logging.debug("Missing geometry for shape_id=%s; skipping trip.", shape_id)
                continue

            rid = str(row["route_id"])
            try:
                drn = int(row["direction_id"])
            except (TypeError, ValueError):
                logging.warning("Bad direction_id for route_id=%s; skipping.", rid)
                continue

            rshort = route_short_lookup.get(rid)
            records.append(
                {
                    "route_id": rid,
                    "direction_id": drn,
                    "route_short": rshort,
                    "geometry": line,
                }
            )
        logging.info("Routes – built %d route-shape records.", len(records))
        return records

    # Union shapes per (route_id, direction_id)
    grouped = trips.groupby(["route_id", "direction_id"])["shape_id"].apply(
        lambda s: sorted(set(str(x) for x in s))
    )

    for (rid, drn_val), shape_ids in grouped.items():
        lines: List[arcpy.Polyline] = []
        for sid in shape_ids:
            line = shape_geoms.get(sid)
            if line is not None:
                lines.append(line)

        if not lines:
            logging.debug(
                "No geometries found for route_id=%s, direction_id=%s; skipping.", rid, drn_val
            )
            continue

        geom_union = lines[0]
        for line in lines[1:]:
            geom_union = geom_union.union(line)

        try:
            drn = int(drn_val)
        except (TypeError, ValueError):
            logging.warning("Bad direction_id=%s for route_id=%s; skipping union.", drn_val, rid)
            continue

        rshort = route_short_lookup.get(str(rid))
        records.append(
            {
                "route_id": str(rid),
                "direction_id": drn,
                "route_short": rshort,
                "geometry": geom_union,
            }
        )

    logging.info("Routes – built %d unioned route polylines.", len(records))
    return records


# =============================================================================
# STOP AGGREGATION (ROUTE/DIRECTION LISTS)
# =============================================================================


def _build_stop_aggregates(
    dfs: Dict[str, pd.DataFrame],
    trips_selected: pd.DataFrame,
    routes_selected: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return stop layers for all routes and for the filtered subset.

    Args:
        dfs: Dictionary of raw GTFS tables.
        trips_selected: Trips that survived the include/exclude filter.
        routes_selected: Routes that survived the include/exclude filter.

    Returns:
        (all_stops_df, selected_stops_df) where each DataFrame contains:
        stop_id, stop_name, stop_lat, stop_lon, route_id, direction_id,
        route_short_name. The three *_id/name columns are Python lists with
        normalized types (route_id → str, direction_id → int, route_short_name → str).
    """
    stops = dfs["stops"]
    stop_times = dfs["stop_times"]

    def _agg_for(trips_df: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
        served = stop_times.loc[stop_times["trip_id"].isin(trips_df["trip_id"])]

        trip_attrs = trips_df[["trip_id", "route_id", "direction_id"]].merge(
            routes_df[["route_id", "route_short_name"]],
            on="route_id",
            how="left",
        )
        merged = served[["trip_id", "stop_id"]].merge(
            trip_attrs,
            on="trip_id",
            how="left",
        )

        merged["direction_id"] = pd.to_numeric(
            merged["direction_id"],
            errors="coerce",
        )

        agg = (
            merged.groupby("stop_id")[["route_id", "direction_id", "route_short_name"]]
            .agg(lambda s: sorted(set(s.dropna())))
            .reset_index()
        )

        base = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
        out_df = base.merge(agg, on="stop_id", how="inner")

        # Normalize types once here; downstream functions rely on this.
        out_df["route_id"] = out_df["route_id"].apply(lambda vals: [str(v) for v in vals])
        out_df["direction_id"] = out_df["direction_id"].apply(lambda vals: [int(v) for v in vals])
        out_df["route_short_name"] = out_df["route_short_name"].apply(
            lambda vals: [str(v) for v in vals]
        )
        return out_df

    all_trips = dfs["trips"]
    all_routes = dfs["routes"]

    all_stops_df = _agg_for(all_trips, all_routes)
    selected_stops_df = _agg_for(trips_selected, routes_selected)

    logging.info(
        "Stops – all served stops: %d; filtered served stops: %d",
        len(all_stops_df),
        len(selected_stops_df),
    )
    return all_stops_df, selected_stops_df


# =============================================================================
# ROUTE/STOP INDEX + ORDERED STOPS
# =============================================================================


def _build_route_stop_index(
    stops_df: pd.DataFrame,
) -> Dict[Tuple[str, int], np.ndarray]:
    """Map (route_id, direction_id) to row indices in stops_df.

    This is a one-time, global index that replaces repeated per-route
    `.apply(lambda ids: rid in ids)` scans in the QA and segment exporter.
    """
    index: dict[tuple[str, int], list[int]] = defaultdict(list)

    for idx, row in stops_df.iterrows():
        route_ids: list[str] = row.route_id
        dir_ids: list[int] = row.direction_id
        for rid in route_ids:
            for drn in dir_ids:
                key = (str(rid), int(drn))
                index[key].append(int(idx))

    return {key: np.asarray(vals, dtype=int) for key, vals in index.items()}


class RouteStop(NamedTuple):
    """Representation of a stop ordered along a route polyline."""

    stop_id: str
    stop_name: str | None
    measure: float


def _ordered_route_stops(
    rec: Dict[str, Any],
    stops_df: pd.DataFrame,
    route_index: Dict[Tuple[str, int], np.ndarray],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    line: arcpy.Polyline,
) -> List[RouteStop]:
    """Return unique, ordered stops along a route polyline.

    The result is sorted by measureOnLine and de-duplicates equal measures.
    """
    rid = rec["route_id"]
    drn = int(rec["direction_id"])

    idxs = route_index.get((rid, drn))
    if idxs is None or len(idxs) < 2:
        return []

    served = stops_df.iloc[idxs]

    dists: list[float] = []
    stop_ids: list[str] = []
    stop_names: list[str] = []

    for _, row in served.iterrows():
        sid = str(row.stop_id)
        pt_geom = stop_geoms.get(sid)
        if pt_geom is None:
            continue

        m = line.measureOnLine(pt_geom, use_percentage=False)
        if not np.isfinite(m):
            continue

        dists.append(float(m))
        stop_ids.append(sid)
        stop_names.append(str(row.stop_name))

    if len(dists) < 2:
        return []

    order = np.argsort(dists)
    d_sorted = [dists[i] for i in order]
    id_sorted = [stop_ids[i] for i in order]
    name_sorted = [stop_names[i] for i in order]

    result: list[RouteStop] = []
    last_d: float | None = None
    for d_val, sid, sname in zip(d_sorted, id_sorted, name_sorted):
        if last_d is None or d_val > last_d:
            result.append(RouteStop(stop_id=sid, stop_name=sname, measure=d_val))
            last_d = d_val

    return result


# =============================================================================
# EXPORT SHAPEFILES (STOPS, ROUTES, SEGMENTS)
# =============================================================================


def _export_stops_shapefile(
    stops_df: pd.DataFrame,
    stop_geoms: Dict[str, arcpy.PointGeometry],
    sr: arcpy.SpatialReference,
    out_folder: Path,
) -> None:
    """Write filtered served stops to stops.shp."""
    arcpy.env.overwriteOutput = True

    out_name = "stops"
    fc_path = os.path.join(out_folder.as_posix(), out_name + ".shp")

    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        out_folder.as_posix(),
        out_name,
        "POINT",
        spatial_reference=sr,
    )

    # Shapefile field names must be <= 10 characters.
    fields = [
        ("stop_id", "TEXT", 64),
        ("stop_name", "TEXT", 128),
        ("routes", "TEXT", 254),  # comma-separated route_ids
        ("dirs", "TEXT", 254),  # comma-separated direction_ids
        ("rshorts", "TEXT", 254),  # comma-separated route_short_names
    ]
    for name, ftype, length in fields:
        arcpy.management.AddField(
            fc_path,
            name,
            ftype,
            field_length=length if length is not None else None,
        )

    insert_fields = ["stop_id", "stop_name", "routes", "dirs", "rshorts", "SHAPE@"]

    rows_written = 0
    with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
        for row in stops_df.itertuples(index=False):
            sid = str(row.stop_id)
            geom = stop_geoms.get(sid)
            if geom is None:
                logging.debug("No geometry for stop_id=%s; skipping.", sid)
                continue

            routes_str = ",".join(str(r) for r in row.route_id)
            dirs_str = ",".join(str(d) for d in row.direction_id)
            shorts_str = ",".join(str(s) for s in row.route_short_name)

            cursor.insertRow(
                [
                    sid,
                    str(row.stop_name),
                    routes_str,
                    dirs_str,
                    shorts_str,
                    geom,
                ]
            )
            rows_written += 1

    logging.info("Wrote %s (%d features).", fc_path, rows_written)


def _export_routes_shapefile(
    routes: List[Dict[str, Any]],
    sr: arcpy.SpatialReference,
    out_folder: Path,
) -> None:
    """Write route polylines to routes.shp."""
    arcpy.env.overwriteOutput = True

    out_name = "routes"
    fc_path = os.path.join(out_folder.as_posix(), out_name + ".shp")

    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        out_folder.as_posix(),
        out_name,
        "POLYLINE",
        spatial_reference=sr,
    )

    # Field names <= 10 chars.
    fields = [
        ("route_id", "TEXT", 64),
        ("dir", "SHORT", None),
        ("rshort", "TEXT", 64),
    ]
    for name, ftype, length in fields:
        arcpy.management.AddField(
            fc_path,
            name,
            ftype,
            field_length=length if length is not None else None,
        )

    insert_fields = ["route_id", "dir", "rshort", "SHAPE@"]

    rows_written = 0
    with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
        for rec in routes:
            geom: arcpy.Polyline | None = rec.get("geometry")
            if _is_empty_polyline(geom):
                logging.debug(
                    "Skipping empty or null route geometry for route_id=%s, dir=%s",
                    rec.get("route_id"),
                    rec.get("direction_id"),
                )
                continue

            cursor.insertRow(
                [
                    rec["route_id"],
                    int(rec["direction_id"]),
                    rec.get("route_short"),
                    geom,
                ]
            )
            rows_written += 1

    logging.info("Wrote %s (%d features).", fc_path, rows_written)


def _export_segments_shapefile(
    routes: List[Dict[str, Any]],
    stops_df: pd.DataFrame,
    route_index: Dict[Tuple[str, int], np.ndarray],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    sr: arcpy.SpatialReference,
    out_folder: Path,
) -> None:
    """Split each route polyline at its own stops and write segments.shp."""
    arcpy.env.overwriteOutput = True

    out_name = "segments"
    fc_path = os.path.join(out_folder.as_posix(), out_name + ".shp")

    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        out_folder.as_posix(),
        out_name,
        "POLYLINE",
        spatial_reference=sr,
    )

    # Field names <= 10 chars.
    fields = [
        ("route_id", "TEXT", 64),
        ("dir", "SHORT", None),
        ("rshort", "TEXT", 64),
        ("len_ft", "DOUBLE", None),
    ]
    for name, ftype, length in fields:
        arcpy.management.AddField(
            fc_path,
            name,
            ftype,
            field_length=length if length is not None else None,
        )

    insert_fields = ["route_id", "dir", "rshort", "len_ft", "SHAPE@"]

    ft_factor = _feet_factor(sr)
    rows_written = 0

    with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
        for rec in routes:
            line: arcpy.Polyline = rec["geometry"]
            if _is_empty_polyline(line):
                continue

            rid = rec["route_id"]
            drn = int(rec["direction_id"])
            rshort = rec.get("route_short")

            stops = _ordered_route_stops(rec, stops_df, route_index, stop_geoms, line)
            if len(stops) < 2:
                logging.debug(
                    "Route %s dir=%s has fewer than 2 ordered stops; skipping segments.",
                    rid,
                    drn,
                )
                continue

            for j in range(len(stops) - 1):
                start_m = stops[j].measure
                end_m = stops[j + 1].measure
                if end_m <= start_m:
                    continue

                seg_geom = line.segmentAlongLine(start_m, end_m, use_percentage=False)
                if _is_empty_polyline(seg_geom):
                    continue

                length_ft = seg_geom.length * ft_factor
                cursor.insertRow([rid, drn, rshort, float(length_ft), seg_geom])
                rows_written += 1

    logging.info("Wrote %s (%d features).", fc_path, rows_written)


# =============================================================================
# QA – SHORT AND LONG SPACING (ARCPY GEOMETRY)
# =============================================================================


def _flag_short_spacing(
    routes: List[Dict[str, Any]],
    stops_df: pd.DataFrame,
    route_index: Dict[Tuple[str, int], np.ndarray],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    sr: arcpy.SpatialReference,
    threshold_ft: float,
    log_path: Path,
) -> None:
    """Write a log of consecutive stops spaced closer than threshold_ft."""
    ft_factor = _feet_factor(sr)
    count = 0

    with log_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "route_id\tdirection_id\tbegin_stop_id\tbegin_stop_name\t"
            "end_stop_id\tend_stop_name\tspacing_ft\n"
        )

        for rec in routes:
            line: arcpy.Polyline = rec["geometry"]
            if _is_empty_polyline(line):
                continue

            rid = rec["route_id"]
            drn = int(rec["direction_id"])

            stops = _ordered_route_stops(rec, stops_df, route_index, stop_geoms, line)
            if len(stops) < 2:
                continue

            for i in range(len(stops) - 1):
                spacing_ft = (stops[i + 1].measure - stops[i].measure) * ft_factor
                if spacing_ft < threshold_ft:
                    fh.write(
                        f"{rid}\t{drn}\t"
                        f"{stops[i].stop_id}\t{stops[i].stop_name}\t"
                        f"{stops[i + 1].stop_id}\t{stops[i + 1].stop_name}\t"
                        f"{spacing_ft:.1f}\n"
                    )
                    count += 1

    logging.info(
        "Wrote short-spacing log → %s (%d flagged segments).",
        log_path.name,
        count,
    )


def _flag_long_spacing_csv(
    routes: List[Dict[str, Any]],
    all_stops_df: pd.DataFrame,
    all_route_index: Dict[Tuple[str, int], np.ndarray],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    sr: arcpy.SpatialReference,
    threshold_ft: float,
    near_buffer_ft: float,
    csv_path: Path,
    summary: bool = True,
) -> None:
    """Export a CSV of “missed” stops that fill unusually long gaps.

    A long gap is any consecutive pair of served stops on a given
    (route_id, direction_id) whose spacing exceeds threshold_ft. For every
    other-route stop that lies inside the gap and within near_buffer_ft of
    the route polyline, a row is written to the CSV.
    """
    ft_factor = _feet_factor(sr)
    records: List[Dict[str, Any]] = []

    # Optional precomputation of stop coordinates (not strictly required, but cheap).
    stop_coords: dict[str, Tuple[float, float]] = {}
    for sid, geom in stop_geoms.items():
        if geom is not None and geom.firstPoint is not None:
            stop_coords[sid] = (geom.firstPoint.X, geom.firstPoint.Y)

    for rec in routes:
        line: arcpy.Polyline = rec["geometry"]
        if _is_empty_polyline(line):
            continue

        rid = rec["route_id"]
        drn = int(rec["direction_id"])
        rshort = rec.get("route_short")

        stops = _ordered_route_stops(rec, all_stops_df, all_route_index, stop_geoms, line)
        if len(stops) < 2:
            continue

        for i in range(len(stops) - 1):
            start_m = float(stops[i].measure)
            end_m = float(stops[i + 1].measure)
            seg_len_ft = (end_m - start_m) * ft_factor
            if seg_len_ft <= threshold_ft:
                continue

            # Endpoints of the long segment
            start_pt = line.positionAlongLine(start_m, use_percentage=False)
            end_pt = line.positionAlongLine(end_m, use_percentage=False)

            minx = min(start_pt.firstPoint.X, end_pt.firstPoint.X) - near_buffer_ft
            miny = min(start_pt.firstPoint.Y, end_pt.firstPoint.Y) - near_buffer_ft
            maxx = max(start_pt.firstPoint.X, end_pt.firstPoint.X) + near_buffer_ft
            maxy = max(start_pt.firstPoint.Y, end_pt.firstPoint.Y) + near_buffer_ft

            start_sid = stops[i].stop_id
            end_sid = stops[i + 1].stop_id
            start_name = stops[i].stop_name or ""
            end_name = stops[i + 1].stop_name or ""

            # Scan all stops (could be optimized further with spatial index if needed).
            for _, st_row in all_stops_df.iterrows():
                # Skip stops on the same route/direction.
                if rid in st_row.route_id and drn in st_row.direction_id:
                    continue

                sid = str(st_row.stop_id)
                pt_geom = stop_geoms.get(sid)
                if pt_geom is None or pt_geom.firstPoint is None:
                    continue

                x = pt_geom.firstPoint.X
                y = pt_geom.firstPoint.Y
                if not (minx <= x <= maxx and miny <= y <= maxy):
                    continue

                proj_m = line.measureOnLine(pt_geom, use_percentage=False)
                if not (np.isfinite(proj_m) and start_m < proj_m < end_m):
                    continue

                dist_to_route_ft = line.distanceTo(pt_geom) * ft_factor
                if dist_to_route_ft <= near_buffer_ft:
                    records.append(
                        {
                            "route_id": rid,
                            "route_short": rshort,
                            "direction_id": drn,
                            "seg_len_ft": round(seg_len_ft, 1),
                            "start_stop_id": start_sid,
                            "start_stop_name": start_name,
                            "end_stop_id": end_sid,
                            "end_stop_name": end_name,
                            "flagged_stop_id": sid,
                            "flagged_stop_name": str(st_row.stop_name),
                            "dist_to_route_ft": round(dist_to_route_ft, 1),
                        }
                    )

    if not records:
        logging.info("No long-spacing issues found.")
        return

    fieldnames = [
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
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    logging.info("Wrote long-spacing CSV → %s (%d rows).", csv_path.name, len(records))

    if summary:
        flagged_pairs = {(rec["route_id"], rec["direction_id"]) for rec in records}
        summ_path = csv_path.with_name(f"{csv_path.stem}_summary.txt")
        with summ_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("route_id\tdirection_id\n")
            for rid, drn in sorted(flagged_pairs):
                fh.write(f"{rid}\t{drn}\n")
        logging.info(
            "Wrote summary → %s (%d route/direction pairs).",
            summ_path.name,
            len(flagged_pairs),
        )


# =============================================================================
# PROPOSED STOP RELOCATIONS (WHAT-IF QA)
# =============================================================================


def _load_proposed_stops(
    source: Sequence[Tuple[str, float, float]] | str | Path | None,
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
        raw_rows: Iterable[Tuple[Any, Any, Any]] = zip(df[id_col], df[lat_col], df[lon_col])
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


def _apply_proposed_stop_geoms(
    stop_geoms: Dict[str, arcpy.PointGeometry],
    moves: Dict[str, Tuple[float, float]],
    projected_sr: arcpy.SpatialReference,
) -> Dict[str, arcpy.PointGeometry]:
    """Return a copy of *stop_geoms* with relocated stops re-projected.

    Args:
        stop_geoms: Mapping stop_id → PointGeometry (projected).
        moves: Mapping ``stop_id → (new_lat, new_lon)`` in WGS84.
        projected_sr: Target projected spatial reference.

    Returns:
        New mapping; stops absent from *moves* keep their geometry.
    """
    out = dict(stop_geoms)
    if not moves:
        return out

    wgs84 = arcpy.SpatialReference(4326)
    for sid, (lat, lon) in moves.items():
        pt = arcpy.Point(lon, lat)
        out[sid] = arcpy.PointGeometry(pt, wgs84).projectAs(projected_sr)
    return out


def _evaluate_proposed_spacing(
    routes: List[Dict[str, Any]],
    stops_df: pd.DataFrame,
    route_index: Dict[Tuple[str, int], np.ndarray],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    proposed_geoms: Dict[str, arcpy.PointGeometry],
    moved_ids: Set[str],
    sr: arcpy.SpatialReference,
    min_spacing_ft: float,
    long_spacing_ft: float,
) -> pd.DataFrame:
    """Recompute along-route spacing around relocated stops.

    For every (route_id, direction_id) polyline that serves a relocated stop,
    each consecutive served-stop pair touching a relocated stop is measured
    along the polyline under both the original and the proposed coordinates.

    Args:
        routes: Route records built by :func:`_build_routes_from_shapes`.
        stops_df: Served stops (filtered set) with route/direction lists.
        route_index: Output of :func:`_build_route_stop_index` for *stops_df*.
        stop_geoms: Original stop geometries.
        proposed_geoms: Output of :func:`_apply_proposed_stop_geoms`.
        moved_ids: ``stop_id`` values that were relocated.
        sr: Projected spatial reference (for the feet factor).
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
    ft_factor = _feet_factor(sr)
    records: List[Dict[str, Any]] = []

    for rec in routes:
        line: arcpy.Polyline = rec["geometry"]
        if _is_empty_polyline(line):
            continue

        rid = rec["route_id"]
        drn = int(rec["direction_id"])

        after = _ordered_route_stops(rec, stops_df, route_index, proposed_geoms, line)
        if len(after) < 2:
            continue
        moved_here = {s.stop_id for s in after} & moved_ids
        if not moved_here:
            continue

        before = _ordered_route_stops(rec, stops_df, route_index, stop_geoms, line)
        before_pos: Dict[str, float] = {s.stop_id: s.measure for s in before}

        for i in range(len(after) - 1):
            s0, s1 = after[i], after[i + 1]
            pair_moved = [sid for sid in (s0.stop_id, s1.stop_id) if sid in moved_here]
            if not pair_moved:
                continue

            spacing_after = (s1.measure - s0.measure) * ft_factor
            b0 = before_pos.get(s0.stop_id)
            b1 = before_pos.get(s1.stop_id)
            if b0 is not None and b1 is not None:
                spacing_before = abs(b1 - b0) * ft_factor
            else:
                spacing_before = float("nan")

            if spacing_after < min_spacing_ft:
                verdict = "too short"
            elif spacing_after > long_spacing_ft:
                verdict = "too long"
            else:
                verdict = "OK"

            records.append(
                {
                    "route_id": rid,
                    "route_short": rec.get("route_short"),
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


def _plot_route_polyline(ax: Any, line: arcpy.Polyline) -> None:
    """Draw every part of an arcpy Polyline on a matplotlib axis."""
    for part in line:
        xs = [pt.X for pt in part if pt]
        ys = [pt.Y for pt in part if pt]
        if xs:
            ax.plot(xs, ys, color="0.6", linewidth=1.0, zorder=1)


def _plot_proposed_stops(
    routes: List[Dict[str, Any]],
    stops_df: pd.DataFrame,
    route_index: Dict[Tuple[str, int], np.ndarray],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    proposed_geoms: Dict[str, arcpy.PointGeometry],
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

    name_by_id: Dict[str, str] = dict(
        zip(stops_df["stop_id"].astype(str), stops_df["stop_name"].astype(str))
    )

    def _xy(geom: arcpy.PointGeometry | None) -> Tuple[float, float] | None:
        if geom is None or geom.firstPoint is None:
            return None
        return (geom.firstPoint.X, geom.firstPoint.Y)

    for sid in moves:
        old_xy = _xy(stop_geoms.get(sid))
        new_xy = _xy(proposed_geoms.get(sid))
        if old_xy is None or new_xy is None:
            continue

        stop_rows = stops_df.loc[stops_df["stop_id"].astype(str) == sid]
        if stop_rows.empty:
            continue
        row = stop_rows.iloc[0]
        serving = [
            rec
            for rec in routes
            if not _is_empty_polyline(rec["geometry"])
            and str(rec["route_id"]) in [str(x) for x in row.route_id]
            and int(rec["direction_id"]) in [int(x) for x in row.direction_id]
        ]

        rows = report[
            report["moved_stop_id"].astype(str).str.split(",").apply(lambda xs, s=sid: s in xs)
        ]

        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

        for rec in serving:
            _plot_route_polyline(ax, rec["geometry"])
            idxs = route_index.get((str(rec["route_id"]), int(rec["direction_id"])))
            if idxs is None:
                continue
            for nsid in stops_df.iloc[idxs]["stop_id"].astype(str):
                n_xy = _xy(proposed_geoms.get(nsid))
                if n_xy is not None:
                    ax.plot([n_xy[0]], [n_xy[1]], "o", color="0.4", markersize=3, zorder=2)

        ax.plot([old_xy[0]], [old_xy[1]], "o", color="red", markersize=7, zorder=4)
        ax.plot([new_xy[0]], [new_xy[1]], "o", color="green", markersize=7, zorder=4)
        ax.annotate(
            "",
            xy=new_xy,
            xytext=old_xy,
            arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "black"},
            zorder=3,
        )

        focus_xs: List[float] = [old_xy[0], new_xy[0]]
        focus_ys: List[float] = [old_xy[1], new_xy[1]]
        for rec_row in rows.itertuples(index=False):
            p0 = _xy(proposed_geoms.get(str(rec_row.begin_stop_id)))
            p1 = _xy(proposed_geoms.get(str(rec_row.end_stop_id)))
            if p0 is None or p1 is None:
                continue
            focus_xs.extend([p0[0], p1[0]])
            focus_ys.extend([p0[1], p1[1]])
            ax.annotate(
                f"{rec_row.spacing_ft_after:,.0f} ft ({rec_row.verdict})",
                xy=((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="green" if rec_row.compliant else "red",
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
    proposed_source: Sequence[Tuple[str, float, float]] | str | None,
    stops_master: pd.DataFrame,
    routes: List[Dict[str, Any]],
    stops_df: pd.DataFrame,
    route_index: Dict[Tuple[str, int], np.ndarray],
    stop_geoms: Dict[str, arcpy.PointGeometry],
    sr: arcpy.SpatialReference,
    min_spacing_ft: float,
    long_spacing_ft: float,
    csv_path: Path,
    export_maps: bool,
) -> None:
    """Load, evaluate, and report proposed stop relocations (no-op when unset).

    Args:
        proposed_source: The ``PROPOSED_STOPS`` configuration value.
        stops_master: Raw ``stops.txt`` DataFrame (for stop_id/stop_code lookup).
        routes: Route records built by :func:`_build_routes_from_shapes`.
        stops_df: Served stops (filtered set) with route/direction lists.
        route_index: Output of :func:`_build_route_stop_index` for *stops_df*.
        stop_geoms: Original projected stop geometries.
        sr: Projected spatial reference.
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

    served_ids = set(stops_df["stop_id"].astype(str))
    for sid in moves:
        if sid not in served_ids:
            logging.warning(
                "Proposed stop %s is not served by any selected route/direction; "
                "its relocation cannot be evaluated here.",
                sid,
            )

    proposed_geoms = _apply_proposed_stop_geoms(stop_geoms, moves, sr)
    report = _evaluate_proposed_spacing(
        routes,
        stops_df,
        route_index,
        stop_geoms,
        proposed_geoms,
        set(moves),
        sr,
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
        _plot_proposed_stops(
            routes,
            stops_df,
            route_index,
            stop_geoms,
            proposed_geoms,
            moves,
            report,
            csv_path.parent,
        )


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
    if GTFS_PATH == r"Path\To\Your\GTFS_Folder" or OUTPUT_FOLDER == r"Path\To\Your\Output_Folder":
        logging.warning(
            "GTFS_PATH and/or OUTPUT_FOLDER are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2
    arcpy.env.overwriteOutput = True

    out_dir = _ensure_output_folder(OUTPUT_FOLDER)
    logging.info("STEP 0  Reading GTFS tables …")
    dfs = _read_gtfs_tables(GTFS_PATH)

    try:
        _validate_columns(dfs)
    except ValueError as err:
        logging.error("ERROR – invalid GTFS feed:\n%s", err)
        return 1

    logging.info("STEP 0·1  Filtering routes and trips …")
    routes_df, trips_df = _filter_routes(
        dfs["routes"],
        dfs["trips"],
        INCLUDE_ROUTE_IDS,
        FILTER_OUT_LIST,
    )
    logging.info(
        "Routes kept: %d; Trips kept: %d (after include/exclude).",
        len(routes_df),
        len(trips_df),
    )

    sr = _get_projected_sr(PROJECTED_WKID)
    feet_factor = _feet_factor(sr)
    logging.info("Using SR: %s (1 unit ≈ %.3f ft)", sr.name, feet_factor)

    logging.info("STEP 1  Building stop aggregates …")
    all_stops_df, sel_stops_df = _build_stop_aggregates(dfs, trips_df, routes_df)

    # Build route/stop index (all stops and selected stops) once.
    sel_route_index = _build_route_stop_index(sel_stops_df)
    all_route_index = _build_route_stop_index(all_stops_df)

    logging.info("STEP 2  Building geometries for shapes and stops …")
    shape_geoms = _build_shape_geometries(dfs["shapes"], sr)
    stop_geoms = _build_stop_geometries(dfs["stops"], sr)
    logging.info(
        "Built %d shape polylines and %d stop points.",
        len(shape_geoms),
        len(stop_geoms),
    )

    logging.info("STEP 3  Building route polylines …")
    routes = _build_routes_from_shapes(trips_df, routes_df, shape_geoms, ROUTE_UNION)

    logging.info("STEP 4  Exporting stops and routes shapefiles …")
    _export_stops_shapefile(sel_stops_df, stop_geoms, sr, out_dir)
    _export_routes_shapefile(routes, sr, out_dir)

    logging.info("STEP 5  Building stop-to-stop segment shapefile …")
    _export_segments_shapefile(routes, sel_stops_df, sel_route_index, stop_geoms, sr, out_dir)

    logging.info("STEP 6  Short-spacing QA …")
    _flag_short_spacing(
        routes,
        sel_stops_df,
        sel_route_index,
        stop_geoms,
        sr,
        MIN_SPACING_FT,
        out_dir / SPACING_LOG_FILE,
    )

    logging.info("STEP 7  Long-spacing QA …")
    _flag_long_spacing_csv(
        routes,
        all_stops_df,
        all_route_index,
        stop_geoms,
        sr,
        LONG_SPACING_FT,
        NEAR_BUFFER_FT,
        out_dir / LONG_SPACING_CSV_FILE,
    )

    if PROPOSED_STOPS:
        logging.info("STEP 8  Evaluating proposed stop relocations …")
        _run_proposed_stops_qa(
            PROPOSED_STOPS,
            dfs["stops"],
            routes,
            sel_stops_df,
            sel_route_index,
            stop_geoms,
            sr,
            MIN_SPACING_FT,
            LONG_SPACING_FT,
            out_dir / PROPOSED_SPACING_CSV_FILE,
            export_maps=PROPOSED_MAPS,
        )

    logging.info("All done! Outputs in: %s", out_dir)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        logging.exception("UNEXPECTED ERROR")
        sys.exit(1)
