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

An optional stop-deletion scenario (``STOPS_TO_DELETE`` and/or
``STOPS_TO_DELETE_BY_ROUTE``) runs the pipeline twice – once on the original
feed and once with the listed stops removed from service, either everywhere
or from a single route – writing each run to its own subfolder plus a CSV
summarizing the spacing gap each deletion opens up.

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
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Set, Tuple

import geopandas as gpd
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

# Optional stop-deletion scenario. Leave both settings empty for a single
# normal run. When either is non-empty the pipeline runs twice – the original
# feed goes to OUTPUT_FOLDER/BASELINE_SUBFOLDER and a feed with the listed
# stops removed from service goes to OUTPUT_FOLDER/SCENARIO_SUBFOLDER – and a
# CSV describing the gap each deletion opens is written to OUTPUT_FOLDER.
# Entries may be stop_id or stop_code values; stop_code wins when an entry
# matches both (same convention as stop_removal_impact_gpd.py).
# STOPS_TO_DELETE removes stops from every route and from stops.txt entirely.
# STOPS_TO_DELETE_BY_ROUTE (route_id → entries) removes them from that one
# route's trips only; the stop keeps existing for other routes, so the
# scenario's long-spacing QA may legitimately re-suggest it as a nearby stop.
STOPS_TO_DELETE: list[str] = []  # e.g. ["1001", "1002"]
STOPS_TO_DELETE_BY_ROUTE: dict[str, list[str]] = {}  # e.g. {"101": ["1001"]}
BASELINE_SUBFOLDER: str = r"baseline"
SCENARIO_SUBFOLDER: str = r"stops_removed"
DELETION_IMPACT_CSV_FILE: str = r"stop_deletion_impact.csv"

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
# OPTIONAL STOP-DELETION SCENARIO
# =============================================================================


class RouteStop(NamedTuple):
    """Representation of a stop ordered along a route polyline."""

    stop_id: str
    stop_name: str | None
    measure: float


def _resolve_stops_to_delete(
    stops_df: pd.DataFrame,
    identifiers: Sequence[str],
    label: str = "STOPS_TO_DELETE",
) -> set[str]:
    """Resolve user-entered identifiers to canonical GTFS stop_ids.

    Mirrors the convention in ``stop_removal_impact_gpd.py``: each entry is
    matched against stop_code first (when stops.txt provides one) and then
    stop_id; the first field that matches wins. A stop_code shared by several
    stops resolves to all of them.

    Unmatched entries are logged as warnings; when nothing matches at all, an
    error-level hint shows sample identifiers from this feed so the user can
    spot format mismatches (missing stop_code column, leading zeros, etc.).

    This helper is deliberately kept as a verbatim copy in both spacing
    flaggers (arcpy and gpd); see tests/test_pipeline_contracts.py.

    Args:
        stops_df: stops.txt DataFrame.
        identifiers: Raw stop_id / stop_code values from the deletion config.
        label: Config-setting name used in log messages.

    Returns:
        Set of matched stop_id strings (possibly empty).
    """
    sid_series = stops_df["stop_id"].astype(str)
    id_by_stop_id: Dict[str, List[str]] = (
        sid_series.to_frame(name="stop_id").groupby("stop_id")["stop_id"].apply(list).to_dict()
    )
    id_by_stop_code: Dict[str, List[str]] = {}
    if "stop_code" in stops_df.columns:
        id_by_stop_code = (
            pd.DataFrame({"stop_code": stops_df["stop_code"].astype(str), "stop_id": sid_series})
            .groupby("stop_code")["stop_id"]
            .apply(list)
            .to_dict()
        )

    resolved: set[str] = set()
    unmatched: list[str] = []
    for raw in identifiers:
        key = str(raw).strip()
        if not key:
            continue
        matches = id_by_stop_code.get(key) or id_by_stop_id.get(key) or []
        if matches:
            resolved.update(matches)
        else:
            unmatched.append(key)

    logging.info(
        "%s resolved: %d identifier(s) → %d unique stop_id(s), %d unmatched.",
        label,
        len(identifiers),
        len(resolved),
        len(unmatched),
    )
    if unmatched:
        logging.warning(
            "Unmatched identifiers (neither stop_code nor stop_id): %s",
            ", ".join(unmatched[:20]) + ("…" if len(unmatched) > 20 else ""),
        )
    if unmatched and not resolved:
        sid_examples = ", ".join(sid_series.head(3))
        if id_by_stop_code:
            code_examples = ", ".join(stops_df["stop_code"].astype(str).head(3))
            logging.error(
                "None of the %s entries matched this feed, though both fields were "
                "checked. Compare your entries against the feed's values – stop_ids "
                "look like [%s] and stop_codes like [%s]. Watch for quoting, "
                "prefixes, or leading zeros lost to numeric parsing.",
                label,
                sid_examples,
                code_examples,
            )
        else:
            logging.error(
                "None of the %s entries matched this feed, and its stops.txt has "
                "no stop_code column, so entries can only match stop_id. This "
                "feed's stop_ids look like [%s].",
                label,
                sid_examples,
            )
    return resolved


def _build_deletion_plan(
    stops_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    global_identifiers: Sequence[str],
    by_route_identifiers: Mapping[str, Sequence[str]],
) -> Tuple[set[str], Dict[str, set[str]]]:
    """Resolve the global and per-route deletion configs against the feed.

    This helper is deliberately kept as a verbatim copy in both spacing
    flaggers (arcpy and gpd); see tests/test_pipeline_contracts.py.

    Args:
        stops_df: stops.txt DataFrame.
        routes_df: routes.txt DataFrame, used to validate the route keys.
        global_identifiers: STOPS_TO_DELETE entries, deleted from every route.
        by_route_identifiers: STOPS_TO_DELETE_BY_ROUTE mapping of route_id to
            entries deleted from that route only.

    Returns:
        (global_ids, by_route_ids); route keys that match no route_id in
        routes.txt are warned about and skipped.
    """
    global_ids: set[str] = set()
    if global_identifiers:
        global_ids = _resolve_stops_to_delete(stops_df, global_identifiers)

    by_route_ids: Dict[str, set[str]] = {}
    known_routes = set(routes_df["route_id"].astype(str))
    for route_key, idents in by_route_identifiers.items():
        rid = str(route_key).strip()
        if rid not in known_routes:
            logging.warning(
                "STOPS_TO_DELETE_BY_ROUTE key %r does not match any route_id in "
                "routes.txt; skipping its %d entries.",
                route_key,
                len(idents),
            )
            continue
        ids = _resolve_stops_to_delete(
            stops_df,
            idents,
            label=f"STOPS_TO_DELETE_BY_ROUTE[{route_key!r}]",
        )
        if ids:
            by_route_ids[rid] = ids
    return global_ids, by_route_ids


def _drop_stops_from_feed(
    dfs: Dict[str, pd.DataFrame],
    stop_ids: set[str],
    by_route: Dict[str, set[str]] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Return a copy of the GTFS tables with the given stops removed.

    Globally deleted stops (stop_ids) are dropped from stop_times entirely
    and from stops itself, so the long-spacing QA does not suggest a deleted
    stop as a candidate fill for the gap it left behind. Route-scoped
    deletions (by_route) drop only the stop_times rows belonging to that
    route's trips: the stop keeps existing – and stays served by any other
    routes – so the scenario's long-spacing QA may legitimately re-suggest
    it as a nearby stop for the route it was removed from.

    This helper is deliberately kept as a verbatim copy in both spacing
    flaggers (arcpy and gpd); see tests/test_pipeline_contracts.py.

    Args:
        dfs: Raw GTFS tables keyed by name.
        stop_ids: Canonical stop_ids to remove from every route.
        by_route: Mapping route_id → stop_ids to remove from that route only.

    Returns:
        New table mapping; untouched tables are shared, not copied.
    """
    by_route = by_route or {}
    stop_times = dfs["stop_times"]
    stops = dfs["stops"]

    st_stop_ids = stop_times["stop_id"].astype(str)
    global_mask = st_stop_ids.isin(stop_ids)
    drop_mask = global_mask.copy()
    if by_route:
        trips = dfs["trips"]
        trip_to_route = dict(zip(trips["trip_id"].astype(str), trips["route_id"].astype(str)))
        st_routes = stop_times["trip_id"].astype(str).map(trip_to_route)
        for rid, ids in by_route.items():
            drop_mask |= (st_routes == rid) & st_stop_ids.isin(ids)

    out = dict(dfs)
    out["stop_times"] = stop_times.loc[~drop_mask].copy()
    out["stops"] = stops.loc[~stops["stop_id"].astype(str).isin(stop_ids)].copy()

    logging.info(
        "Scenario feed: dropped %d stop_times row(s) (%d everywhere, %d route-scoped) "
        "and removed %d stop(s) from stops.txt.",
        int(drop_mask.sum()),
        int(global_mask.sum()),
        int((drop_mask & ~global_mask).sum()),
        len(stops) - len(out["stops"]),
    )
    return out


def _deletion_impact_rows_for_route(
    stops: List[RouteStop],
    deleted_ids: set[str],
    global_ids: set[str],
    ft_factor: float,
    long_threshold_ft: float,
    route_id: str,
    route_short: Any,
    direction_id: int,
) -> List[Dict[str, Any]]:
    """Return deletion-impact CSV rows for one route/direction stop sequence.

    Walks the ordered baseline stops and groups each run of consecutive
    deleted stops between two surviving stops into one row: the worst
    sub-gap before deletion, the merged survivor-to-survivor gap after
    deletion, and a flag when that new gap exceeds long_threshold_ft.
    Deletions at a pattern end (which shorten the route rather than open a
    gap) get an explanatory note. The deletion_scope column says whether the
    run's stops were deleted everywhere or only from this route.

    This helper is deliberately kept as a verbatim copy in both spacing
    flaggers (arcpy and gpd); see tests/test_pipeline_contracts.py.

    Args:
        stops: Baseline stops ordered along the route polyline.
        deleted_ids: Stop_ids deleted from this route (global + route-scoped).
        global_ids: Stop_ids deleted from every route, for scope labeling.
        ft_factor: Multiplier converting measure units to feet.
        long_threshold_ft: Gap length that triggers the "exceeds" flag.
        route_id: Route identifier for the output rows.
        route_short: Route short name for the output rows.
        direction_id: Direction identifier for the output rows.

    Returns:
        One dict per deleted run, ready for the impact CSV.
    """
    rows: List[Dict[str, Any]] = []
    i = 0
    n = len(stops)
    while i < n:
        if stops[i].stop_id not in deleted_ids:
            i += 1
            continue

        j = i
        while j < n and stops[j].stop_id in deleted_ids:
            j += 1
        removed = stops[i:j]

        prev_stop = stops[i - 1] if i > 0 else None
        next_stop = stops[j] if j < n else None

        chain = [s for s in [prev_stop, *removed, next_stop] if s is not None]
        old_max_ft: float | None = None
        if len(chain) >= 2:
            old_max_ft = max(
                (b.measure - a.measure) * ft_factor for a, b in zip(chain[:-1], chain[1:])
            )

        new_spacing_ft: float | None = None
        note = ""
        if prev_stop is not None and next_stop is not None:
            new_spacing_ft = (next_stop.measure - prev_stop.measure) * ft_factor
        elif prev_stop is None and next_stop is None:
            note = "all served stops on this route/direction deleted"
        else:
            end = "start" if prev_stop is None else "end"
            note = f"deleted at route {end}; pattern shortened, no new gap"

        scopes = {"all routes" if s.stop_id in global_ids else "this route" for s in removed}
        scope = scopes.pop() if len(scopes) == 1 else "mixed"

        rows.append(
            {
                "route_id": route_id,
                "route_short": route_short,
                "direction_id": direction_id,
                "prev_stop_id": prev_stop.stop_id if prev_stop else "",
                "prev_stop_name": (prev_stop.stop_name or "") if prev_stop else "",
                "next_stop_id": next_stop.stop_id if next_stop else "",
                "next_stop_name": (next_stop.stop_name or "") if next_stop else "",
                "deleted_stop_ids": ",".join(s.stop_id for s in removed),
                "deleted_stop_names": ";".join(str(s.stop_name) for s in removed),
                "n_deleted": len(removed),
                "deletion_scope": scope,
                "old_max_spacing_ft": round(old_max_ft, 1) if old_max_ft is not None else "",
                "new_spacing_ft": round(new_spacing_ft, 1) if new_spacing_ft is not None else "",
                "exceeds_long_ft": (
                    ""
                    if new_spacing_ft is None
                    else ("yes" if new_spacing_ft > long_threshold_ft else "no")
                ),
                "note": note,
            }
        )
        i = j

    return rows


def _log_unserved_deletions(
    global_ids: set[str],
    by_route: Dict[str, set[str]],
    served_deleted: set[str],
    seen_by_route: Dict[str, set[str]],
) -> None:
    """Log deletion-list stops that never appeared on an analyzed route.

    This helper is deliberately kept as a verbatim copy in both spacing
    flaggers (arcpy and gpd); see tests/test_pipeline_contracts.py.

    Args:
        global_ids: Stop_ids slated for deletion from every route.
        by_route: Mapping route_id → stop_ids slated for per-route deletion.
        served_deleted: Deletion-list stop_ids seen on any analyzed route.
        seen_by_route: Mapping route_id → deletion-list stop_ids seen there.
    """
    unserved = sorted(global_ids - served_deleted)
    if unserved:
        logging.info(
            "%d deleted stop(s) are not served by the analyzed routes: %s",
            len(unserved),
            ", ".join(unserved[:20]) + ("…" if len(unserved) > 20 else ""),
        )
    for rid in sorted(by_route):
        missing = sorted(by_route[rid] - seen_by_route.get(rid, set()))
        if missing:
            logging.info(
                "Route %s: %d stop(s) slated for per-route deletion are not served "
                "by that route in the baseline: %s",
                rid,
                len(missing),
                ", ".join(missing[:20]) + ("…" if len(missing) > 20 else ""),
            )


class PipelineArtifacts(NamedTuple):
    """Layers from one pipeline run, kept for cross-run reporting."""

    routes_gdf: gpd.GeoDataFrame
    stops_gdf: gpd.GeoDataFrame


def _ordered_route_stops(
    stops_gdf: gpd.GeoDataFrame,
    line: LineString,
    rid: str,
    drn: int,
) -> List[RouteStop]:
    """Return unique, ordered served stops along a route polyline.

    Args:
        stops_gdf: Served-stop layer with route/direction list fields.
        line: The route polyline to order stops along.
        rid: Route identifier.
        drn: Direction identifier.

    Returns:
        Stops sorted by distance along the line, de-duplicated by measure.
    """
    sindex = stops_gdf.sindex
    cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
    cand = cand[_served_mask(cand, rid, drn)].copy()
    if cand.empty:
        return []

    cand["dist_along"] = cand.geometry.apply(line.project)
    cand = cand.drop_duplicates("dist_along").sort_values("dist_along")
    return [
        RouteStop(stop_id=str(r.stop_id), stop_name=str(r.stop_name), measure=float(r.dist_along))
        for r in cand.itertuples(index=False)
    ]


def _report_stop_deletion_impact(
    baseline: PipelineArtifacts,
    global_ids: set[str],
    by_route: Dict[str, set[str]],
    long_threshold_ft: float,
    csv_path: Path,
) -> None:
    """Write a CSV describing the spacing gap each stop deletion opens.

    For every (route_id, direction_id) in the baseline run, the ordered stop
    sequence is re-walked; see _deletion_impact_rows_for_route for the row
    semantics. Per-route deletions only affect the rows of their own route.

    Args:
        baseline: Layers from the baseline (pre-deletion) pipeline run.
        global_ids: Canonical stop_ids removed from every route.
        by_route: Mapping route_id → stop_ids removed from that route only.
        long_threshold_ft: Gap length that triggers the "exceeds" flag.
        csv_path: Destination CSV path.
    """
    stops_gdf = baseline.stops_gdf
    crs_str = str(stops_gdf.crs) if stops_gdf.crs else ""
    ft_factor = 1.0 if "2263" in crs_str else 3.28084

    rows: List[Dict[str, Any]] = []
    served_deleted: set[str] = set()
    seen_by_route: Dict[str, set[str]] = {}

    for _, row in baseline.routes_gdf.iterrows():
        line: LineString = row.geometry
        rid = str(row.route_id)
        drn = int(row.direction_id)

        stops = _ordered_route_stops(stops_gdf, line, rid, drn)
        if not stops:
            continue

        effective_ids = global_ids | by_route.get(rid, set())
        deleted_here = [s.stop_id for s in stops if s.stop_id in effective_ids]
        served_deleted.update(deleted_here)
        seen_by_route.setdefault(rid, set()).update(deleted_here)

        rows.extend(
            _deletion_impact_rows_for_route(
                stops,
                effective_ids,
                global_ids,
                ft_factor,
                long_threshold_ft,
                rid,
                row.get("route_short_name"),
                drn,
            )
        )

    _log_unserved_deletions(global_ids, by_route, served_deleted, seen_by_route)

    if not rows:
        logging.info("Deletion impact: no deleted stop appears on the analyzed routes.")
        return

    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)
    n_flagged = sum(1 for r in rows if r["exceeds_long_ft"] == "yes")
    logging.info(
        "Wrote deletion-impact CSV → %s (%d gap rows, %d exceeding %.0f ft).",
        csv_path.name,
        len(rows),
        n_flagged,
        long_threshold_ft,
    )


# =============================================================================
# MAIN
# =============================================================================


def _run_pipeline(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
    label: str = "",
) -> PipelineArtifacts:
    """Run filtering, layer building, exports, and both QA checks once.

    Args:
        dfs: Validated GTFS tables keyed by name.
        out_dir: Folder that receives the shapefiles, logs, and CSVs.
        label: Short tag added to log lines when running multiple scenarios.

    Returns:
        The layers needed for cross-run reporting.
    """
    tag = f" [{label}]" if label else ""

    # -----------------------------------------------------------------
    # 0·1  Route / trip filtering
    # -----------------------------------------------------------------
    routes_df, trips_df = _filter_routes(
        dfs["routes"], dfs["trips"], INCLUDE_ROUTE_IDS, FILTER_OUT_LIST
    )

    # -----------------------------------------------------------------
    # STEP 1  Build stop layers
    # -----------------------------------------------------------------
    logging.info("STEP 1%s  Building stop layers …", tag)
    all_stops_gdf, stops_gdf = _build_stop_layers(dfs, trips_df, routes_df, PROJECTED_CRS)
    _export(stops_gdf, out_dir, "stops")  # export only the filtered set

    # -----------------------------------------------------------------
    # STEP 2  Build route polylines
    # -----------------------------------------------------------------
    logging.info("STEP 2%s  Building routes shapefile …", tag)
    routes_gdf = _build_routes_gdf(dfs["shapes"], trips_df, routes_df, PROJECTED_CRS, ROUTE_UNION)
    _export(routes_gdf, out_dir, "routes")

    # -----------------------------------------------------------------
    # STEP 3  Split polylines into stop-to-stop segments
    # -----------------------------------------------------------------
    logging.info("STEP 3%s  Splitting routes into stop-to-stop segments …", tag)
    segs_gdf = _split_into_segments(routes_gdf, stops_gdf, PROJECTED_CRS)
    _export(segs_gdf, out_dir, "segments")  # master file
    _export_segments_by_route_dir(segs_gdf, out_dir)  # per-route files

    # -----------------------------------------------------------------
    # STEP 4  Short-spacing QA
    # -----------------------------------------------------------------
    logging.info("STEP 4%s  Flagging closely-spaced stops …", tag)
    _flag_short_spacing(
        routes_gdf,
        stops_gdf,  # filtered layer
        MIN_SPACING_FT,
        out_dir / SPACING_LOG_FILE,
    )

    # -----------------------------------------------------------------
    # STEP 5  Long-spacing QA (needs *all* stops) – CSV export
    # -----------------------------------------------------------------
    logging.info("STEP 5%s  Flagging long-spacing segments …", tag)
    _flag_long_spacing_csv(
        routes_gdf,
        all_stops_gdf,  # unfiltered layer
        LONG_SPACING_FT,
        NEAR_BUFFER_FT,
        out_dir / LONG_SPACING_CSV_FILE,
    )

    return PipelineArtifacts(routes_gdf=routes_gdf, stops_gdf=stops_gdf)


def main() -> int:  # noqa: D401
    """Run the entire GTFS-to-GIS pipeline with both spacing QA checks.

    When STOPS_TO_DELETE and/or STOPS_TO_DELETE_BY_ROUTE is non-empty the
    pipeline runs twice – the original feed into BASELINE_SUBFOLDER and the
    feed minus the listed stops into SCENARIO_SUBFOLDER – followed by a
    stop-deletion impact CSV in OUTPUT_FOLDER comparing spacing before and
    after the removals.

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

    out_dir = _ensure_output_folder(OUTPUT_FOLDER)

    if not (STOPS_TO_DELETE or STOPS_TO_DELETE_BY_ROUTE):
        _run_pipeline(dfs, out_dir)
        logging.info("\nAll done! Outputs in: %s", out_dir)
        logging.info("Script completed successfully.")
        return 0

    global_ids, by_route_ids = _build_deletion_plan(
        dfs["stops"],
        dfs["routes"],
        STOPS_TO_DELETE,
        STOPS_TO_DELETE_BY_ROUTE,
    )
    if not (global_ids or by_route_ids):
        logging.error(
            "A stop-deletion scenario is configured but no entry matched the feed – "
            "nothing to simulate."
        )
        return 1

    baseline_dir = _ensure_output_folder(out_dir / BASELINE_SUBFOLDER)
    scenario_dir = _ensure_output_folder(out_dir / SCENARIO_SUBFOLDER)

    logging.info("=== Baseline run (original feed) → %s ===", baseline_dir)
    baseline = _run_pipeline(dfs, baseline_dir, label="baseline")

    n_unique = len(global_ids.union(*by_route_ids.values()) if by_route_ids else global_ids)
    logging.info("=== Scenario run (%d stop(s) removed) → %s ===", n_unique, scenario_dir)
    scenario_dfs = _drop_stops_from_feed(dfs, global_ids, by_route_ids)
    _run_pipeline(scenario_dfs, scenario_dir, label="stops removed")

    # -----------------------------------------------------------------
    # STEP 6  Stop-deletion impact report
    # -----------------------------------------------------------------
    logging.info("STEP 6  Stop-deletion impact report …")
    _report_stop_deletion_impact(
        baseline,
        global_ids,
        by_route_ids,
        LONG_SPACING_FT,
        out_dir / DELETION_IMPACT_CSV_FILE,
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
