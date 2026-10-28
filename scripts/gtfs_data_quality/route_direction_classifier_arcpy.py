"""Classify GTFS route directions and find each route's dominant shape, using ArcPy.

ArcPy twin of ``route_direction_classifier_pd.py``. ArcPy only projects the shapes,
so the script runs in ArcGIS Pro's stock Python environment without GeoPandas or
Shapely; the classification, flags, and outputs match the GeoPandas script.

Each shape is labeled cardinal (NB, SB, EB, WB) from the projected distance
between its first and last points, or looped (CW, CCW, LOOP) when those points
are closer than LOOP_THRESHOLD. Every shape is analyzed and flagged; the
dominant (most-trips) shape per route_id and direction_id is marked, and only
affects which maps are drawn.

Inputs
------
- GTFS feed (folder or .zip): routes.txt, trips.txt, stop_times.txt,
  shapes.txt, stops.txt.

Outputs
-------
- Directions_Summary.xlsx: one row per route/direction/shape/terminal pattern.
- Route_<route>_Dir_<direction_id>_departures.xlsx: every trip, sorted by
  departure. <route> is route_short_name (route_id when blank), with route_id
  appended when several routes share a short name.
- Route_<route>_Dir_<direction_id>_DominantShape.jpeg, or one
  ..._Shape_<shape_id>.jpeg per shape when MAP_ONLY_DOMINANT_SHAPE is False.
- Suspicious_RouteDirections.xlsx and Trip_Exceptions.xlsx: review lists,
  rewritten every run (header-only when clean).
- Directions_Summary_runlog.txt: run-log sidecar with the verbatim
  CONFIGURATION block and the effective settings.

Limitations
-----------
- Trip counts span the whole feed, mixing service calendars.
- NB/SB/EB/WB is the straight-line bearing from first to last shape point,
  and CW/CCW assumes a shape whose ends meet is a simple loop. Neither is the
  agency's own direction label.

Requires
--------
ArcGIS Pro (``arcpy``). pandas, openpyxl, and matplotlib ship with Pro.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI
flags) and run from ArcGIS Pro's Python window, an ArcPy-enabled shell, or a
Jupyter notebook.
"""

from __future__ import annotations

import argparse
import datetime as datetime_module  # arcpy can rebind a from-imported `datetime` to the module
import logging
import math
import os
import re
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, List, Optional

import arcpy
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_PATH = r"Path\To\Your\GTFS_Folder"  # folder of .txt files or a .zip
OUTPUT_DIR = r"Path\To\Your\Output_Folder"

# Optional: Include only these route_short_name values
ROUTE_FILTER_IN: list[str] = []
# Routes to exclude
ROUTE_FILTER_OUT: list[str] = ["9999A", "9999B", "9999C"]

# WKID of the projected coordinate system used for distances.
# 26985 = NAD83 / Maryland (meters).
PROJECTED_CRS: int = 26985
# Shapes whose first and last points are closer than this are treated as loops.
# Units are those of PROJECTED_CRS (meters for 26985).
LOOP_THRESHOLD: float = 200

# Toggles
EXPORT_XLSX = True
EXPORT_JPEG = True

# True: map only the dominant (most-trips) shape per route and direction.
# False: map every shape. Summaries, departures, and flags always use every shape.
MAP_ONLY_DOMINANT_SHAPE = True

# Fixed output filenames (written inside OUTPUT_DIR).
SUMMARY_FILENAME = "Directions_Summary.xlsx"
SUSPICIOUS_FILENAME = "Suspicious_RouteDirections.xlsx"
EXCEPTIONS_FILENAME = "Trip_Exceptions.xlsx"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output is never
# left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

_PLACEHOLDER_GTFS = r"Path\To\Your\GTFS_Folder"
_PLACEHOLDER_OUTPUT = r"Path\To\Your\Output_Folder"

# =============================================================================
# FUNCTIONS
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


def to_numeric_column(df: pd.DataFrame, column: str, table: str) -> pd.Series:
    """Return *column* of *df* as numbers.

    Raises:
        ValueError: The column is missing, or holds a blank or non-numeric value.
    """
    if column not in df.columns:
        raise ValueError(f"{table} is missing required column '{column}'.")
    values = pd.to_numeric(df[column], errors="coerce")
    bad = df.loc[values.isna(), column]
    if not bad.empty:
        raise ValueError(
            f"{table} has {len(bad)} blank or non-numeric '{column}' value(s), "
            f"e.g. {bad.iloc[0]!r}."
        )
    return values


def read_gtfs_data(
    gtfs_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the GTFS files and return routes, trips, stop_times, shapes, and stops.

    Sequence and coordinate columns become numbers, so sequences sort as
    1, 2, 10 rather than 1, 10, 2; IDs stay text. Blank or absent optional
    text columns become empty strings so that groupings keep their rows.
    """
    files = ["routes.txt", "trips.txt", "stop_times.txt", "shapes.txt", "stops.txt"]
    gtfs_data = load_gtfs_data(gtfs_path, files=files, dtype=str)

    routes = gtfs_data["routes"]
    trips = gtfs_data["trips"]
    stop_times = gtfs_data["stop_times"]
    shapes = gtfs_data["shapes"]
    stops = gtfs_data["stops"]

    stop_times["stop_sequence"] = to_numeric_column(stop_times, "stop_sequence", "stop_times.txt")
    for col in ("shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"):
        shapes[col] = to_numeric_column(shapes, col, "shapes.txt")
    for df, columns in (
        (routes, ("route_short_name", "route_long_name")),
        (trips, ("direction_id", "shape_id")),
    ):
        for col in columns:
            df[col] = df[col].fillna("") if col in df.columns else ""

    return routes, trips, stop_times, shapes, stops


def filter_routes(
    routes: pd.DataFrame,
    routes_include: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Apply the route_short_name include/exclude lists and return the kept routes."""
    routes_filtered = routes.copy()
    if routes_include:
        routes_filtered = routes_filtered[routes_filtered["route_short_name"].isin(routes_include)]
    if routes_exclude:
        routes_filtered = routes_filtered[~routes_filtered["route_short_name"].isin(routes_exclude)]
    return routes_filtered


def add_route_labels(routes: pd.DataFrame) -> pd.DataFrame:
    """Add a ``route_label`` column used in map titles and file names.

    The label is route_short_name (route_id when that is blank), with route_id
    appended when several routes share a short name, so each route keeps its own
    files.
    """
    routes = routes.copy()
    label = routes["route_short_name"].where(routes["route_short_name"] != "", routes["route_id"])
    shared = label.duplicated(keep=False)
    routes["route_label"] = label.where(~shared, label + "_" + routes["route_id"])
    return routes


def build_shape_coords(shapes: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    """Return each shape's (lon, lat) points in shape_pt_sequence order.

    Shapes with fewer than two points cannot form a line and are skipped; their
    trips are listed in the exceptions report.
    """
    shape_coords: dict[str, list[tuple[float, float]]] = {}
    for shape_id, group in shapes.sort_values(["shape_id", "shape_pt_sequence"]).groupby(
        "shape_id"
    ):
        if len(group) < 2:
            logging.warning("Shape %s has fewer than two points; skipped.", shape_id)
            continue
        shape_coords[str(shape_id)] = list(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
    return shape_coords


def project_shapes(
    shape_coords: Mapping[str, Sequence[tuple[float, float]]], projected_crs: int
) -> dict[str, list[tuple[float, float]]]:
    """Project each shape's (lon, lat) points from WGS 84 to *projected_crs* with ArcPy.

    Shapes that ArcPy reduces to fewer than two points (for example, every point
    identical) are skipped; their trips are listed in the exceptions report.

    Raises:
        RuntimeError: ArcPy does not recognize *projected_crs*.
    """
    sr_wgs84 = arcpy.SpatialReference(4326)
    sr_projected = arcpy.SpatialReference(projected_crs)

    projected: dict[str, list[tuple[float, float]]] = {}
    for shape_id, coords in shape_coords.items():
        points = arcpy.Array([arcpy.Point(lon, lat) for lon, lat in coords])
        line = arcpy.Polyline(points, sr_wgs84).projectAs(sr_projected)
        if line.pointCount < 2:
            logging.warning("Shape %s has no length after projection; skipped.", shape_id)
            continue
        projected[shape_id] = [(pt.X, pt.Y) for pt in line.getPart(0)]
    return projected


def classify_direction(
    coords: Sequence[tuple[float, float]],
    loop_threshold: float = LOOP_THRESHOLD,
) -> str:
    """Classify the direction of a GTFS shape between its start and end points.

    If the start/end points are close (less than loop_threshold), the shape
    is treated as a loop and classified based on the polygon's signed area.
    Otherwise, the projected east-west and north-south distances between the
    endpoints decide NB, SB, EB, or WB.

    Args:
        coords: The shape's points in a projected coordinate system, in order.
        loop_threshold: Loop cutoff, in the coordinate system's units.
    """
    start_p = coords[0]
    end_p = coords[-1]

    dist_start_end = math.sqrt((start_p[0] - end_p[0]) ** 2 + (start_p[1] - end_p[1]) ** 2)

    if dist_start_end < loop_threshold:
        ring = list(coords)
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        area = sum((x1 * y2 - x2 * y1) for (x1, y1), (x2, y2) in zip(ring, ring[1:]))

        if area > 0:
            return "CCW"
        if area < 0:
            return "CW"
        return "LOOP"

    dx = end_p[0] - start_p[0]
    dy = end_p[1] - start_p[1]
    if abs(dy) > abs(dx):
        return "NB" if dy > 0 else "SB"
    return "EB" if dx > 0 else "WB"


def merge_and_classify_shapes(
    trips: pd.DataFrame, routes_filtered: pd.DataFrame, directions: Mapping[str, str]
) -> pd.DataFrame:
    """Filter trips by routes and attach each trip's shape_direction.

    Returns trips joined to route metadata (including `route_label` from
    :func:`add_route_labels`). Trips whose shape is blank or missing from
    shapes.txt are kept with a blank `shape_direction`.
    """
    trips_filtered = trips[trips["route_id"].isin(routes_filtered["route_id"])]
    trips_merged = trips_filtered.merge(
        routes_filtered[["route_id", "route_short_name", "route_long_name", "route_label"]],
        on="route_id",
    )
    direction_df = pd.DataFrame(
        {"shape_id": list(directions.keys()), "shape_direction": list(directions.values())}
    )
    return trips_merged.merge(direction_df, on="shape_id", how="left")


def identify_first_last_stops(
    trips_merged: pd.DataFrame, stop_times: pd.DataFrame, stops: pd.DataFrame
) -> pd.DataFrame:
    """Identify first/last stops (with names) and merge into trips.

    Returns the augmented DataFrame with first/last stop IDs and names,
    plus `departure_time` from the first stop and `departure_minutes` for
    sorting. The first stop is the row with the lowest numeric stop_sequence,
    whatever its value. Trips without stop_times, or whose stops are missing
    from stops.txt, are kept with blank values.
    """
    stop_times_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])
    # drop_duplicates keeps whole rows; groupby().first() would take the first
    # non-blank value per column, borrowing a later stop's departure_time.
    first_stops = stop_times_sorted.drop_duplicates("trip_id", keep="first")
    last_stops = stop_times_sorted.drop_duplicates("trip_id", keep="last")

    first_stops = first_stops.rename(columns={"stop_id": "first_stop_id"})
    last_stops = last_stops.rename(columns={"stop_id": "last_stop_id"})

    trips_merged = trips_merged.merge(
        first_stops[["trip_id", "first_stop_id", "departure_time"]], on="trip_id", how="left"
    ).merge(last_stops[["trip_id", "last_stop_id"]], on="trip_id", how="left")

    stops_ren_first = stops[["stop_id", "stop_name"]].rename(
        columns={"stop_id": "first_stop_id", "stop_name": "first_stop_name"}
    )
    stops_ren_last = stops[["stop_id", "stop_name"]].rename(
        columns={"stop_id": "last_stop_id", "stop_name": "last_stop_name"}
    )

    trips_merged = trips_merged.merge(stops_ren_first, on="first_stop_id", how="left")
    trips_merged = trips_merged.merge(stops_ren_last, on="last_stop_id", how="left")

    trips_merged["departure_minutes"] = trips_merged["departure_time"].map(parse_time_to_minutes)
    return trips_merged


def parse_time_to_minutes(time_value: Optional[str]) -> Optional[int]:
    """Convert an ``HH:MM[:SS]`` time string to integer minutes past midnight.

    GTFS times may exceed 24:00 (e.g. ``"25:30:00"`` for a 1:30 AM trip on
    the following calendar day); those values are preserved as integers
    greater than or equal to 1440. Seconds, when present, are rounded to the
    nearest minute.

    Args:
        time_value: Time string such as ``"7:05"``, ``"07:05:00"``, or
            ``"26:30:00"``. Leading/trailing whitespace is ignored.
            Non-string or malformed values yield ``None``.

    Returns:
        Minutes since midnight, or ``None`` if the value cannot be parsed.
    """
    if not isinstance(time_value, str):
        return None
    parts = time_value.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return hours * 60 + minutes + round(seconds / 60)


def split_trip_exceptions(
    final_data: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    stops: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """List trips with broken references and remove those that cannot be classified.

    A trip is removed from the analysis when its shape is blank or unusable, or
    when it has no stop_times rows. Trips whose terminal stops are missing from
    stops.txt, or whose first stop has no departure_time, stay in the analysis
    but are still listed. Trips whose route_id is not in routes.txt are listed
    regardless of the route filters, since their route is unknown.

    Args:
        final_data: Output of :func:`identify_first_last_stops`.
        trips: The full trips table.
        routes: The full (unfiltered) routes table.
        stops: The stops table.

    Returns:
        ``(kept, exceptions)``: the trips that remain in the analysis, and one
        row per (trip, problem) with an ``excluded`` column.
    """
    known_stops = stops["stop_id"]
    has_stop_times = final_data["first_stop_id"].notna()
    checks = [
        (final_data["shape_id"] == "", "shape_id is blank", True),
        (
            (final_data["shape_id"] != "") & final_data["shape_direction"].isna(),
            "shape_id not in shapes.txt, or its shape has fewer than two points",
            True,
        ),
        (~has_stop_times, "no rows in stop_times.txt", True),
        (
            has_stop_times & ~final_data["first_stop_id"].isin(known_stops),
            "first stop_id not in stops.txt",
            False,
        ),
        (
            has_stop_times & ~final_data["last_stop_id"].isin(known_stops),
            "last stop_id not in stops.txt",
            False,
        ),
        (
            has_stop_times & final_data["departure_minutes"].isna(),
            "first stop has a blank or unreadable departure_time",
            False,
        ),
    ]

    columns = ["route_id", "route_short_name", "direction_id", "trip_id", "shape_id"]
    orphans = trips.loc[
        ~trips["route_id"].isin(routes["route_id"]),
        ["route_id", "direction_id", "trip_id", "shape_id"],
    ]
    frames = [orphans.assign(problem="route_id not in routes.txt", excluded=True)]
    excluded = pd.Series(False, index=final_data.index)
    for mask, problem, excludes in checks:
        frames.append(final_data.loc[mask, columns].assign(problem=problem, excluded=excludes))
        if excludes:
            excluded |= mask

    exceptions = pd.concat(frames, ignore_index=True).reindex(
        columns=[*columns, "problem", "excluded"]
    )
    return final_data[~excluded].copy(), exceptions


def determine_dominant_shapes(final_data: pd.DataFrame) -> pd.DataFrame:
    """Determines the dominant shape per route and direction.

    A dominant shape is the one associated with the most trips for a given
    (route_id, direction_id) pair; ties go to the first shape_id in sort
    order. This function flags these shapes with a boolean column
    'is_dominant' in the returned DataFrame.

    Args:
        final_data: Merged trip-shape-stop DataFrame.

    Returns:
        DataFrame with 'is_dominant' column added: True for dominant shapes,
        False otherwise.
    """
    keys = ["route_id", "direction_id"]
    shape_counts = final_data.groupby([*keys, "shape_id"]).size().reset_index(name="trip_count")

    idx_max = shape_counts.groupby(keys)["trip_count"].idxmax()
    dominant = shape_counts.loc[idx_max, [*keys, "shape_id"]]

    merged = final_data.merge(dominant, on=[*keys, "shape_id"], how="left", indicator=True)
    merged["is_dominant"] = merged.pop("_merge") == "both"
    return merged


def summarize_patterns(final_data: pd.DataFrame) -> pd.DataFrame:
    """Count trips per route, direction, shape, and first/last stop pattern."""
    return (
        final_data.groupby(
            [
                "route_id",
                "route_short_name",
                "route_label",
                "direction_id",
                "shape_id",
                "shape_direction",
                "is_dominant",
                "first_stop_name",
                "last_stop_name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="trip_count")
    )


def flag_suspicious_data(summary: pd.DataFrame) -> pd.DataFrame:
    """Flag inconsistent shape_direction vs. direction_id combinations.

    Uses every shape in *summary*, not only dominant ones, so a minority shape
    running the opposite way is still caught.

    Returns:
        One row per flagged route or route/direction, with columns route_id,
        route_short_name, direction_id, and problem. Empty when nothing is flagged.
    """
    summary_simplified = summary[
        ["route_id", "route_short_name", "direction_id", "shape_direction"]
    ].drop_duplicates()

    flags = []
    route_groups = summary_simplified.groupby(["route_id", "route_short_name"])
    for (route_id, route_name), grp in route_groups:
        unique_dirs = sorted(grp["direction_id"].unique())
        unique_shape_dirs = grp["shape_direction"].unique()
        if len(unique_dirs) > 1 and len(unique_shape_dirs) == 1:
            flags.append(
                {
                    "route_id": route_id,
                    "route_short_name": route_name,
                    "direction_id": ", ".join(unique_dirs),
                    "problem": (
                        f"Multiple direction_ids {unique_dirs} but only one "
                        f"shape_direction '{unique_shape_dirs[0]}'"
                    ),
                }
            )

    def is_cardinal_direction(direct: str) -> bool:
        return direct in ("NB", "SB", "EB", "WB")

    rd_group = summary_simplified.groupby(["route_id", "route_short_name", "direction_id"])
    for (route_id, rname, did), subgrp in rd_group:
        cardinal_dirs = sorted({d for d in subgrp["shape_direction"] if is_cardinal_direction(d)})
        if len(cardinal_dirs) > 1:
            flags.append(
                {
                    "route_id": route_id,
                    "route_short_name": rname,
                    "direction_id": did,
                    "problem": f"Conflicting cardinal directions {cardinal_dirs}",
                }
            )

    return pd.DataFrame(flags, columns=["route_id", "route_short_name", "direction_id", "problem"])


# -----------------------------------------------------------------------------
# OUTPUTS
# -----------------------------------------------------------------------------


def output_file_name(route_label: str, direction_id: str, suffix: str) -> str:
    """Return ``Route_<route>_Dir_<direction>_<suffix>`` safe to use as a file name.

    A blank direction_id appears as ``none``. Characters that Windows does not
    allow in file names become "-".
    """
    name = f"Route_{route_label}_Dir_{direction_id or 'none'}_{suffix}"
    return re.sub(r'[\\/:*?"<>|]+', "-", name)


def export_excel_summaries(
    summary: pd.DataFrame,
    final_data: pd.DataFrame,
    flagged: pd.DataFrame,
    exceptions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the summary, review workbooks, and per-route/direction departure files.

    The review workbooks are rewritten on every run, header-only when there is
    nothing to report, so a clean run never leaves an earlier run's findings in
    place. Departure files hold every trip of one route_id and direction_id,
    sorted by first departure.
    """
    for frame, name in (
        (summary, SUMMARY_FILENAME),
        (flagged, SUSPICIOUS_FILENAME),
        (exceptions, EXCEPTIONS_FILENAME),
    ):
        frame.to_excel(output_dir / name, index=False)

    for (_route_id, direction), group in final_data.groupby(["route_id", "direction_id"]):
        output_name = output_file_name(group["route_label"].iloc[0], direction, "departures.xlsx")
        group.sort_values(["departure_minutes", "trip_id"]).to_excel(
            output_dir / output_name, index=False
        )


def plot_route_shape(
    coords: Sequence[tuple[float, float]],
    route: str,
    direction: str,
    output_path: Path,
    shape_label: str = "Most Common Shape",
) -> None:
    """Plot one shape from its (lon, lat) points and save it as a .jpeg.

    Highlights the start and end of the route shape, and includes a simple
    'N' arrow for orientation in the top-left corner of the map.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)  # file output; leaves an active GUI backend alone
    import matplotlib.pyplot as plt

    lons = [lon for lon, _ in coords]
    lats = [lat for _, lat in coords]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lons, lats, color="blue", linewidth=2, alpha=0.8)

    ax.plot(lons[0], lats[0], "go", label="Start")
    ax.plot(lons[-1], lats[-1], "ro", label="End")

    # "N" sits just above the arrow's tip so the two never overlap.
    ax.text(
        0.05,
        0.94,
        "N",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="center",
        rotation=0,
    )
    ax.annotate(
        "",
        xy=(0.05, 0.93),
        xytext=(0.05, 0.85),
        xycoords="axes fraction",
        arrowprops=dict(facecolor="black", width=1, headwidth=6),
    )

    ax.set_title(f"Route {route}, Direction {direction} - {shape_label}")
    ax.legend()
    ax.axis("equal")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_jpegs(
    summary: pd.DataFrame,
    shape_coords: Mapping[str, Sequence[tuple[float, float]]],
    output_dir: Path,
    dominant_only: bool,
) -> None:
    """Exports JPEG maps of route shapes, one per (route, direction, shape).

    With *dominant_only*, only the dominant shape of each route and direction is
    mapped, as ``..._DominantShape.jpeg``. Otherwise each shape gets its own
    ``..._Shape_<shape_id>.jpeg``. A shape used by several routes is mapped once
    per route.
    """
    maps = summary.drop_duplicates(["route_id", "direction_id", "shape_id"])
    if dominant_only:
        maps = maps[maps["is_dominant"]]

    for row in maps.to_dict("records"):
        shape_id, route, direction = row["shape_id"], row["route_label"], row["direction_id"]
        if dominant_only:
            suffix, shape_label = "DominantShape.jpeg", "Most Common Shape"
        else:
            suffix = f"Shape_{shape_id}.jpeg"
            shape_label = f"Shape {shape_id}" + (" (most common)" if row["is_dominant"] else "")

        output_path = output_dir / output_file_name(route, direction, suffix)
        plot_route_shape(shape_coords[shape_id], route, direction, output_path, shape_label)


# -----------------------------------------------------------------------------
# RUN LOG
# -----------------------------------------------------------------------------


def resolve_source_file() -> Optional[Path]:
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
    """Write the verbatim config block plus a run summary into *output_dir*.

    The sidecar is named after :data:`SUMMARY_FILENAME` (same stem,
    ``_runlog.txt`` suffix) per CONTRIBUTING.md.

    Args:
        output_dir: Directory that received this run's outputs.
        summary_lines: Human-readable lines describing what the run did.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / f"{Path(SUMMARY_FILENAME).stem}_runlog.txt"

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
        "ROUTE DIRECTION CLASSIFIER (ARCPY) RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime_module.datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "RUN SUMMARY",
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
    except OSError as exc:
        logging.error("Could not write run log '%s': %s", log_path, exc)
        return False
    logging.info("Run log written to '%s'.", log_path)
    return True


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


# =============================================================================
# PIPELINE
# =============================================================================


def run(
    gtfs_path: str,
    output_dir: Path,
    routes_include: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
    projected_crs: int = PROJECTED_CRS,
    loop_threshold: float = LOOP_THRESHOLD,
    export_xlsx: bool = True,
    export_jpeg: bool = True,
    map_only_dominant_shape: bool = True,
) -> pd.DataFrame:
    """Classify every shape, flag inconsistencies, and write the requested outputs.

    1. Reads GTFS files (routes, trips, stop_times, shapes, stops).
    2. Filters the routes according to the include/exclude lists.
    3. Projects each shape with ArcPy and classifies its direction.
    4. Identifies first/last stops for each trip, merges stop names, and sets
       aside trips with broken references.
    5. Determines dominant shapes within each route/direction based on trip counts.
    6. Flags data where shape_direction and direction_id seem inconsistent.
    7. Exports summaries, per-route files, and review workbooks to Excel, if enabled.
    8. Exports JPEG maps of dominant (or all) shapes, if enabled.

    Returns:
        The pattern summary (one row per route/direction/shape/terminal pattern).

    Raises:
        OSError: The feed cannot be read or an output cannot be written.
        ValueError: The feed is malformed.
        RuntimeError: ArcPy cannot project the shapes, or the run log could
            not be written while ``REQUIRE_RUN_LOG`` is True.
    """
    # Step 1: Read GTFS
    routes, trips, stop_times, shapes, stops = read_gtfs_data(gtfs_path)

    # Step 2: Filter routes
    routes_filtered = add_route_labels(filter_routes(routes, routes_include, routes_exclude))

    # Step 3: Build, project, and classify shapes
    shape_coords = build_shape_coords(shapes)
    projected = project_shapes(shape_coords, projected_crs)
    directions = {sid: classify_direction(xy, loop_threshold) for sid, xy in projected.items()}
    trips_merged = merge_and_classify_shapes(trips, routes_filtered, directions)

    # Step 4: Identify first and last stops; set aside trips with broken references
    final_data = identify_first_last_stops(trips_merged, stop_times, stops)
    final_data, exceptions = split_trip_exceptions(final_data, trips, routes, stops)
    excluded_trips = exceptions.loc[exceptions["excluded"], "trip_id"].nunique()
    if not exceptions.empty:
        logging.warning(
            "%d trip problem(s) found; %d trip(s) excluded from the analysis.",
            len(exceptions),
            excluded_trips,
        )

    # Step 5: Determine dominant shapes (every shape stays in the analysis)
    final_data = determine_dominant_shapes(final_data)
    summary = summarize_patterns(final_data)

    # Step 6: Flag suspicious data
    flagged = flag_suspicious_data(summary)
    if flagged.empty:
        logging.info("No suspicious route/direction combos found.")
    else:
        logging.warning("%d suspicious route/direction combination(s) flagged.", len(flagged))

    if not (export_xlsx or export_jpeg):
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 7: Export to Excel
    if export_xlsx:
        export_excel_summaries(summary, final_data, flagged, exceptions, output_dir)

    # Step 8: Export JPEGs
    if export_jpeg:
        export_jpegs(summary, shape_coords, output_dir, map_only_dominant_shape)

    summary_lines = [
        f"GTFS feed:               {gtfs_path}",
        f"Route include/exclude:   {list(routes_include)} / {list(routes_exclude)}",
        f"Projected CRS (WKID):    {projected_crs}",
        f"Loop threshold:          {loop_threshold}",
        f"Excel / JPEG export:     {export_xlsx} / {export_jpeg}",
        f"Map only dominant shape: {map_only_dominant_shape}",
        f"Routes analyzed:         {final_data['route_id'].nunique()}",
        f"Trips analyzed:          {len(final_data)}",
        f"Trips excluded:          {excluded_trips}",
        f"Patterns:                {len(summary)}",
        f"Suspicious combinations: {len(flagged)}",
    ]
    if not write_run_log(output_dir, summary_lines) and REQUIRE_RUN_LOG:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "check when writing to a read-only location."
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify GTFS shape directions with ArcPy, mark each route's dominant shape, "
            "and flag direction inconsistencies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs", default=GTFS_PATH, help="GTFS feed: a folder of .txt files or a .zip archive."
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for all outputs.")
    parser.add_argument(
        "--routes-include",
        nargs="*",
        default=ROUTE_FILTER_IN,
        help="route_short_name values to keep; omit to keep all.",
    )
    parser.add_argument(
        "--routes-exclude",
        nargs="*",
        default=ROUTE_FILTER_OUT,
        help="route_short_name values to drop.",
    )
    parser.add_argument(
        "--projected-crs",
        type=int,
        default=PROJECTED_CRS,
        help="WKID of the projected coordinate system used for distances.",
    )
    parser.add_argument(
        "--loop-threshold",
        type=float,
        default=LOOP_THRESHOLD,
        help="Shapes whose ends are closer than this (projected units) are loops.",
    )
    parser.add_argument(
        "--export-xlsx",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_XLSX,
        help="Write the Excel summaries, review workbooks, and departure files.",
    )
    parser.add_argument(
        "--export-jpeg",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_JPEG,
        help="Write JPEG maps of route shapes.",
    )
    parser.add_argument(
        "--map-only-dominant-shape",
        action=argparse.BooleanOptionalAction,
        default=MAP_ONLY_DOMINANT_SHAPE,
        help="Map only each route/direction's dominant shape instead of every shape.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Validates placeholder paths before doing any work.

    Args:
        argv: Optional explicit argument list (used by tests); ``None``
            reads ``sys.argv`` outside notebook kernels.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))

    if args.gtfs == _PLACEHOLDER_GTFS or args.output_dir == _PLACEHOLDER_OUTPUT:
        logging.warning(
            "GTFS_PATH and/or OUTPUT_DIR are still set to placeholder values. "
            "Update the CONFIGURATION section (or pass --gtfs / --output-dir) before running."
        )
        return 2

    output_dir = Path(args.output_dir).expanduser()
    try:
        run(
            gtfs_path=args.gtfs,
            output_dir=output_dir,
            routes_include=args.routes_include,
            routes_exclude=args.routes_exclude,
            projected_crs=args.projected_crs,
            loop_threshold=args.loop_threshold,
            export_xlsx=args.export_xlsx,
            export_jpeg=args.export_jpeg,
            map_only_dominant_shape=args.map_only_dominant_shape,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Script completed successfully. Outputs are in %s.", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
