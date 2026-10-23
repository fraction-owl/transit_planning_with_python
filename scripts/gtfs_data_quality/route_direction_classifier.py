"""Classifies GTFS route directions and identifies dominant shapes per route and direction.

Determines whether GTFS shapes are cardinal (NB, SB, EB, WB) or looped (CW, CCW, LOOP),
flags suspicious direction inconsistencies, and optionally exports Excel summaries and
JPEG maps of dominant route shapes. Every shape is analyzed and flagged; the dominant
(most-trips) shape per route_id and direction_id is marked, and only affects which maps
are drawn. An ArcPy twin, ``route_direction_classifier_arcpy.py``, runs without GeoPandas.

Inputs:
    - GTFS feed (folder or .zip): routes.txt, trips.txt, stop_times.txt, shapes.txt,
      stops.txt

Outputs:
    - Directions_Summary.xlsx (one row per route/direction/shape/terminal pattern)
    - Route_<route>_Dir_<direction_id>_departures.xlsx (every trip, sorted by departure)
    - Route_<route>_Dir_<direction_id>_DominantShape.jpeg, or one
      ..._Shape_<shape_id>.jpeg per shape when MAP_ONLY_DOMINANT_SHAPE is False
    - Suspicious_RouteDirections.xlsx (rewritten every run; header-only when clean)
    - Trip_Exceptions.xlsx (trips with broken references; rewritten every run)

    <route> is route_short_name, or route_id when the short name is blank; route_id is
    appended when several routes share a short name.

Limitations:
    - Trip counts span the whole feed, mixing service calendars.
    - NB/SB/EB/WB is the straight-line bearing from first to last shape point, and
      CW/CCW assumes a shape whose ends meet is a simple loop. Neither is the agency's
      own direction label.

Typical usage:
    Update the paths in the CONFIGURATION section and run from a shell or a Jupyter
    notebook.
"""

from __future__ import annotations

import logging
import math
import os
import re
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER = r"/path/to/your/gtfs_folder"
OUTPUT_FOLDER = r"/path/to/your/output_folder"

# Optional: Include only these route_short_name values
ROUTE_FILTER_IN: list[str] = []
# Routes to exclude
ROUTE_FILTER_OUT = ["9999A", "9999B", "9999C"]

# NAD83 / Maryland (meters)
PROJECTED_CRS = "EPSG:26985"
# Shapes whose first and last points are closer than this are treated as loops.
# Units are those of PROJECTED_CRS (meters for EPSG:26985).
LOOP_THRESHOLD = 200

# Toggles
EXPORT_XLSX = True
EXPORT_JPEG = True

# True: map only the dominant (most-trips) shape per route and direction.
# False: map every shape. Summaries, departures, and flags always use every shape.
MAP_ONLY_DOMINANT_SHAPE = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

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


def classify_direction(
    line_projected: LineString,
    loop_threshold: float = LOOP_THRESHOLD,
) -> str:
    """Classify the direction of a GTFS shape between its start and end points.

    If the start/end points are close (less than loop_threshold), the shape
    is treated as a loop and classified based on the polygon's signed area.
    Otherwise, the projected east-west and north-south distances between the
    endpoints decide NB, SB, EB, or WB. Projected units are used because a
    degree of longitude covers less ground than a degree of latitude.
    """
    start_p = line_projected.coords[0]
    end_p = line_projected.coords[-1]

    dist_start_end = math.sqrt((start_p[0] - end_p[0]) ** 2 + (start_p[1] - end_p[1]) ** 2)

    if dist_start_end < loop_threshold:
        coords = list(line_projected.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        area = sum((x1 * y2 - x2 * y1) for (x1, y1), (x2, y2) in zip(coords, coords[1:]))

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


def plot_route_shape(
    gdf_shape: gpd.GeoDataFrame,
    route: str,
    direction: str,
    output_path: str,
    shape_label: str = "Most Common Shape",
) -> None:
    """Plot the given shape geometry and save as a .jpeg.

    Highlights the start and end of the route shape, and includes a simple
    'N' arrow for orientation in the top-left corner of the map.
    """
    if not isinstance(gdf_shape, gpd.GeoDataFrame):
        gdf_shape = gpd.GeoDataFrame([gdf_shape], columns=gdf_shape.index, crs="EPSG:4326")

    gdf_shape_4326 = gdf_shape.to_crs(epsg=4326)
    shape_line = gdf_shape_4326.iloc[0].geometry
    start_lon, start_lat = shape_line.coords[0]
    end_lon, end_lat = shape_line.coords[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    gdf_shape_4326.plot(ax=ax, color="blue", linewidth=2, alpha=0.8)

    ax.plot(start_lon, start_lat, "go", label="Start")
    ax.plot(end_lon, end_lat, "ro", label="End")

    ax.text(
        0.05,
        0.95,
        "N",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="center",
        rotation=0,
    )
    ax.annotate(
        "",
        xy=(0.05, 0.94),
        xytext=(0.05, 0.90),
        xycoords="axes fraction",
        arrowprops=dict(facecolor="black", width=1, headwidth=6),
    )

    ax.set_title(f"Route {route}, Direction {direction} - {shape_label}")
    ax.legend()
    plt.axis("equal")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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


def read_gtfs_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads GTFS files and returns the dataframes as a tuple.

    Sequence and coordinate columns become numbers, so sequences sort as
    1, 2, 10 rather than 1, 10, 2; IDs stay text. Blank or absent optional
    text columns become empty strings so that groupings keep their rows.
    """
    files = ["routes.txt", "trips.txt", "stop_times.txt", "shapes.txt", "stops.txt"]
    gtfs_data = load_gtfs_data(GTFS_FOLDER, files=files, dtype=str)

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


def filter_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """Applies the route filters to the routes DataFrame and returns the filtered routes."""
    if ROUTE_FILTER_IN:
        routes_filtered = routes[routes["route_short_name"].isin(ROUTE_FILTER_IN)]
    else:
        routes_filtered = routes.copy()

    if ROUTE_FILTER_OUT:
        routes_filtered = routes_filtered[
            ~routes_filtered["route_short_name"].isin(ROUTE_FILTER_OUT)
        ]
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


def create_lines_from_shapes(shapes: pd.DataFrame) -> gpd.GeoDataFrame:
    """Sorts shapes, builds LineString geometries, and returns a GeoDataFrame (EPSG:4326).

    Shapes with fewer than two points cannot form a line and are skipped; their
    trips are listed in the exceptions report.
    """
    shapes_grouped = shapes.sort_values(["shape_id", "shape_pt_sequence"]).groupby("shape_id")

    lines = []
    for sid, group in shapes_grouped:
        if len(group) < 2:
            logging.warning("Shape %s has fewer than two points; skipped.", sid)
            continue
        line = LineString(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
        lines.append((sid, line))

    gdf = gpd.GeoDataFrame(lines, columns=["shape_id", "geometry"], crs="EPSG:4326")
    return gdf


def merge_and_classify_shapes(
    trips: pd.DataFrame, routes_filtered: pd.DataFrame, gdf_shapes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Filter trips by routes and merge with shape direction classification.

    Returns a DataFrame of trips joined to route metadata (including
    `route_label` from :func:`add_route_labels`) and a derived
    `shape_direction` for each `shape_id`. Trips whose shape is blank or
    missing from shapes.txt are kept with a blank `shape_direction`.
    """
    trips_filtered = trips[trips["route_id"].isin(routes_filtered["route_id"])]
    trips_merged = trips_filtered.merge(
        routes_filtered[["route_id", "route_short_name", "route_long_name", "route_label"]],
        on="route_id",
    )

    gdf_shapes_proj = gdf_shapes.to_crs(PROJECTED_CRS)

    directions: list[tuple[str, str]] = []
    for row_proj in gdf_shapes_proj.itertuples(index=False):
        directions.append((row_proj.shape_id, classify_direction(row_proj.geometry)))

    direction_df = pd.DataFrame(directions, columns=["shape_id", "shape_direction"])
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


def output_file_name(route_label: str, direction_id: str, suffix: str) -> str:
    """Return ``Route_<route>_Dir_<direction>_<suffix>`` safe to use as a file name.

    A blank direction_id appears as ``none``. Characters that Windows does not
    allow in file names become "-".
    """
    name = f"Route_{route_label}_Dir_{direction_id or 'none'}_{suffix}"
    return re.sub(r'[\\/:*?"<>|]+', "-", name)


def export_excel_summaries(summary: pd.DataFrame, final_data: pd.DataFrame) -> None:
    """Export Excel summary and per-route/direction departure files.

    Writes `Directions_Summary.xlsx` and `Route_<route>_Dir_<direction_id>_departures.xlsx`,
    one per route_id and direction_id, with every trip sorted by first departure.
    """
    summary_path = os.path.join(OUTPUT_FOLDER, "Directions_Summary.xlsx")
    summary.to_excel(summary_path, index=False)

    grouped_fd = final_data.groupby(["route_id", "direction_id"])
    for (_route_id, direction), group in grouped_fd:
        output_name = output_file_name(group["route_label"].iloc[0], direction, "departures.xlsx")
        output_file = os.path.join(OUTPUT_FOLDER, output_name)
        group.sort_values(["departure_minutes", "trip_id"]).to_excel(output_file, index=False)


def export_jpegs(
    summary: pd.DataFrame,
    gdf_shapes: gpd.GeoDataFrame,
    dominant_only: bool = MAP_ONLY_DOMINANT_SHAPE,
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
        route_shape_gdf = gdf_shapes[gdf_shapes["shape_id"] == shape_id]
        if dominant_only:
            suffix, shape_label = "DominantShape.jpeg", "Most Common Shape"
        else:
            suffix = f"Shape_{shape_id}.jpeg"
            shape_label = f"Shape {shape_id}" + (" (most common)" if row["is_dominant"] else "")

        output_name = output_file_name(route, direction, suffix)
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        plot_route_shape(route_shape_gdf, route, direction, output_path, shape_label)


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


def write_review_workbooks(flagged: pd.DataFrame, exceptions: pd.DataFrame) -> None:
    """Write Suspicious_RouteDirections.xlsx and Trip_Exceptions.xlsx.

    Both are rewritten on every run, header-only when there is nothing to
    report, so a clean run never leaves an earlier run's findings in place.
    """
    for frame, name in (
        (flagged, "Suspicious_RouteDirections.xlsx"),
        (exceptions, "Trip_Exceptions.xlsx"),
    ):
        frame.to_excel(os.path.join(OUTPUT_FOLDER, name), index=False)


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Primary entry point for GTFS direction classification.

    1. Reads GTFS files (routes, trips, stop_times, shapes, stops).
    2. Filters the routes according to user-defined inclusion/exclusion lists.
    3. Merges shape data with trip data, creating LineString geometries and classifying them.
    4. Identifies first/last stops for each trip, merges stop names, and sets aside
       trips with broken references.
    5. Determines dominant shapes within each route/direction based on trip counts.
    6. Flags suspicious data where shape_direction and direction_id seem inconsistent.
    7. Exports summarized results, per-route files, and review workbooks to Excel, if enabled.
    8. Exports JPEG maps of dominant (or all) shapes, if enabled.

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
        GTFS_FOLDER == r"/path/to/your/gtfs_folder"
        or OUTPUT_FOLDER == r"/path/to/your/output_folder"
    ):
        logging.warning(
            "GTFS_FOLDER and/or OUTPUT_FOLDER are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Step 1: Read GTFS
        routes, trips, stop_times, shapes, stops = read_gtfs_data()

        # Step 2: Filter routes
        routes_filtered = add_route_labels(filter_routes(routes))

        # Step 3: Build shape geometries and classify directions
        gdf_shapes = create_lines_from_shapes(shapes)
        trips_merged = merge_and_classify_shapes(trips, routes_filtered, gdf_shapes)

        # Step 4: Identify first and last stops; set aside trips with broken references
        final_data = identify_first_last_stops(trips_merged, stop_times, stops)
        final_data, exceptions = split_trip_exceptions(final_data, trips, routes, stops)
        if not exceptions.empty:
            logging.warning(
                "%d trip problem(s) found; %d trip(s) excluded from the analysis.",
                len(exceptions),
                exceptions.loc[exceptions["excluded"], "trip_id"].nunique(),
            )

        # Step 5: Determine dominant shapes (every shape stays in the analysis)
        final_data = determine_dominant_shapes(final_data)

        summary = (
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

        # Step 6: Flag suspicious data
        flagged = flag_suspicious_data(summary)
        if flagged.empty:
            logging.info("No suspicious route/direction combos found.")
        else:
            logging.warning("%d suspicious route/direction combination(s) flagged.", len(flagged))

        # Step 7: Export to Excel
        if EXPORT_XLSX:
            export_excel_summaries(summary, final_data)
            write_review_workbooks(flagged, exceptions)

        # Step 8: Export JPEGs
        if EXPORT_JPEG:
            export_jpegs(summary, gdf_shapes, dominant_only=MAP_ONLY_DOMINANT_SHAPE)
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Script completed successfully. Outputs are in %s.", OUTPUT_FOLDER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
