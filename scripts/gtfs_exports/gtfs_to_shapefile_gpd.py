"""Converts GTFS `stops.txt` and `shapes.txt` files into ESRI Shapefiles.

Exports GTFS stops as point features and routes as LineStrings using
standard WGS 84 coordinates. Designed for notebook workflows.
Supports configurable default input/output paths and selective export.

Inputs:
    - GTFS directory with `stops.txt` (required) and `shapes.txt` (optional)
    - Optional export type: "stops", "lines", or "both"
    - Optional per-route split (uses `trips.txt`/`routes.txt` when present)

Outputs:
    - `gtfs_stops.shp`: Shapefile of transit stop points
    - `gtfs_lines.shp`: Shapefile of transit route line geometries
    - `gtfs_lines_by_route/`: One shapefile per route (only when the
      per-route split is enabled)

Typical usage:
    Update the default paths in the CONFIGURATION section and run from a
    shell or a Jupyter notebook, or import and call `gtfs_to_shapefiles()`
    with explicit paths.
"""

import logging
import os
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

# ===========================================================================
# CONFIGURATION
# ===========================================================================

GTFS_CRS = "EPSG:4326"  # Standard CRS for GTFS (WGS 84)
# Type alias for export choices for clarity
ExportKind = Literal["stops", "lines", "both"]

# REQUIRED: Default path to the directory containing GTFS .txt files
DEFAULT_GTFS_DIR: Optional[Path] = Path(r"/path/to/your/default_gtfs_folder")  # <-- EDIT ME

# REQUIRED: Default path to the directory where Shapefiles will be saved
DEFAULT_OUTPUT_DIR: Optional[Path] = Path(r"/path/to/your/default_output_folder")  # <-- EDIT ME
# Set to None if you always want to provide paths as arguments
# DEFAULT_GTFS_DIR = None
# DEFAULT_OUTPUT_DIR = None

# If True, additionally write one shapefile per route (grouped through the
# shape_id → route_id mapping in trips.txt) into PER_ROUTE_SUBDIR alongside
# the combined gtfs_lines.shp. Off by default to keep output folders small.
SPLIT_BY_ROUTE: bool = False

# Subdirectory (inside the output directory) that receives the per-route
# shapefiles when SPLIT_BY_ROUTE is enabled.
PER_ROUTE_SUBDIR: str = "gtfs_lines_by_route"

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# ===========================================================================
# FUNCTIONS
# ===========================================================================


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


def read_stops(gtfs_dir: Path) -> gpd.GeoDataFrame:
    """Reads GTFS 'stops.txt' file into a Point GeoDataFrame.

    Args:
        gtfs_dir: Path to the directory containing the GTFS files.

    Returns:
        A GeoDataFrame containing stop locations as Points.

    Raises:
        FileNotFoundError: If 'stops.txt' is not found in gtfs_dir.
        ValueError: If required columns are missing or lat/lon are invalid.
    """
    try:
        data = load_gtfs_data(str(gtfs_dir), files=["stops.txt"], dtype={"stop_id": str})
        df = data["stops"]
    except Exception as e:
        # Map helper errors to script's expected errors for backward compatibility
        if "Missing GTFS files" in str(e):
            raise FileNotFoundError(f"Required file not found: {gtfs_dir / 'stops.txt'}") from e
        raise ValueError(f"Could not read stops.txt: {e}") from e

    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(df.columns)))
        raise ValueError(f"Missing required columns in stops.txt: {', '.join(missing)}")

    # Validate and clean coordinate columns
    for col in ["stop_lat", "stop_lon"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logging.warning(
                "Warning: Non-numeric values found in '%s'. Attempting conversion.", col
            )
            original_count = len(df)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=[col])
            if len(df) < original_count:
                logging.warning(
                    "Warning: Dropped %d stops due to invalid values in '%s'.",
                    original_count - len(df),
                    col,
                )

    if df.empty:
        logging.warning("Warning: No valid stop data found after cleaning.")
        return gpd.GeoDataFrame(columns=list(required) + ["geometry"], geometry=[], crs=GTFS_CRS)

    try:
        geometry = [Point(xy) for xy in zip(df["stop_lon"], df["stop_lat"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=GTFS_CRS)
    except Exception as e:
        raise ValueError(f"Stop geometry creation failed: {e}") from e

    # Keep essential columns
    essential_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon", "geometry"]
    cols_to_keep = [col for col in essential_cols if col in gdf.columns]
    gdf = gdf[cols_to_keep]

    return gdf


def read_shapes(gtfs_dir: Path) -> gpd.GeoDataFrame:
    """Reads GTFS 'shapes.txt' file into a LineString GeoDataFrame.

    If 'shapes.txt' is missing, an empty GeoDataFrame is returned.

    Args:
        gtfs_dir: Path to the directory containing the GTFS files.

    Returns:
        A GeoDataFrame containing shape geometries as LineStrings. Returns an
        empty GeoDataFrame if 'shapes.txt' is missing or invalid.

    Raises:
        ValueError: If 'shapes.txt' exists but is missing required columns
                    or contains invalid coordinate/sequence data.
    """
    if not (gtfs_dir / "shapes.txt").exists():
        logging.info("Info: Optional file 'shapes.txt' not found. Skipping shapes.")
        return gpd.GeoDataFrame(columns=["shape_id", "geometry"], geometry=[], crs=GTFS_CRS)

    try:
        data = load_gtfs_data(str(gtfs_dir), files=["shapes.txt"], dtype={"shape_id": str})
        df = data["shapes"]
    except Exception as e:
        raise ValueError(f"Could not read shapes.txt: {e}") from e

    required = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(df.columns)))
        raise ValueError(f"Missing required columns in shapes.txt: {', '.join(missing)}")

    # Validate and clean coordinate and sequence columns
    coord_cols = ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    for col in coord_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logging.warning(
                "Warning: Non-numeric values found in '%s'. Attempting conversion.", col
            )
            original_count = len(df)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=[col])
            if len(df) < original_count:
                logging.warning(
                    "Warning: Dropped %d shape points due to invalid values in '%s'.",
                    original_count - len(df),
                    col,
                )

    if df.empty:
        logging.warning("Warning: No valid shape point data found after cleaning.")
        return gpd.GeoDataFrame(columns=["shape_id", "geometry"], geometry=[], crs=GTFS_CRS)

    # Ensure sequence is integer and sort points correctly
    df["shape_pt_sequence"] = df["shape_pt_sequence"].astype(int)
    df = df.sort_values(by=["shape_id", "shape_pt_sequence"])

    # Create LineString geometries
    records: list[dict] = []
    try:
        for shape_id, group in df.groupby("shape_id", sort=False):
            coordinates = list(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
            if len(coordinates) < 2:
                logging.warning(
                    "Warning: Shape ID %s skipped: has fewer than 2 valid points.", shape_id
                )
                continue
            line = LineString(coordinates)
            records.append({"shape_id": shape_id, "geometry": line})
    except Exception as e:
        raise ValueError(f"Shape geometry creation failed: {e}") from e

    if not records:
        logging.warning("Warning: No valid line geometries constructed from shapes.txt.")
        return gpd.GeoDataFrame(columns=["shape_id", "geometry"], geometry=[], crs=GTFS_CRS)

    gdf = gpd.GeoDataFrame(records, crs=GTFS_CRS)
    return gdf


def export_gdf(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Exports a GeoDataFrame to an ESRI Shapefile.

    Creates the output directory if needed. Skips export if GDF is empty.

    Args:
        gdf: The GeoDataFrame to export.
        out_path: Full path for the output Shapefile (e.g., /path/to/output.shp).

    Raises:
        IOError: If the file cannot be written.
    """
    if gdf.empty:
        logging.info("Info: Skipping export for %s: No data.", out_path.name)
        return

    try:
        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Export to Shapefile
        gdf.to_file(out_path, driver="ESRI Shapefile", index=False)
        logging.info("Successfully exported %d features to: %s", len(gdf), out_path)
    except Exception as e:
        # Raise as an IOError for clearer upstream handling
        raise IOError(f"Could not write shapefile {out_path}: {e}") from e


def sanitize_filename_component(value: str, max_len: int = 40) -> str:
    """Make a string safe to use as (part of) a shapefile base name.

    Collapses every character outside ``[A-Za-z0-9_]`` to an underscore,
    trims leading/trailing underscores, and truncates to *max_len*
    characters. GTFS route identifiers may contain slashes, spaces, or
    unicode, none of which are safe in file names (or, for ArcGIS,
    in feature class names).

    Args:
        value: Raw string (e.g. a route_short_name or route_id).
        max_len: Maximum length of the returned string.

    Returns:
        A non-empty, filesystem-safe string ("unnamed" if nothing survives).
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return cleaned[:max_len].rstrip("_") or "unnamed"


def build_export_basenames(
    items: Sequence[tuple[str, Optional[str]]],
    prefix: str,
) -> dict[str, str]:
    """Map each key to a unique, filesystem-safe shapefile base name.

    Args:
        items: ``(key, label)`` pairs. The label (e.g. route_short_name) is
            preferred for the visible name; it falls back to the key itself
            (e.g. route_id) when the label is missing or blank.
        prefix: Prepended to every name, e.g. ``"route"`` → ``route_30``.

    Returns:
        Mapping of key → base name (without extension). Names are
        deduplicated case-insensitively so they remain unique on
        case-insensitive filesystems such as Windows.
    """
    used: set[str] = set()
    out: dict[str, str] = {}
    for key, label in items:
        raw = label if label is not None and label.strip() else key
        base = f"{prefix}_{sanitize_filename_component(raw)}"
        candidate = base
        suffix = 2
        while candidate.lower() in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate.lower())
        out[key] = candidate
    return out


def map_shapes_to_routes(gtfs_dir: Path) -> Optional[pd.DataFrame]:
    """Build a shape_id → route mapping from `trips.txt` (and `routes.txt`).

    Args:
        gtfs_dir: Path to the directory containing the GTFS files.

    Returns:
        DataFrame with columns ``shape_id``, ``route_id``, and
        ``route_short`` (one row per distinct shape/route pair), or ``None``
        when `trips.txt` is missing, unreadable, or lacks the needed columns.
    """
    if not (gtfs_dir / "trips.txt").exists():
        logging.warning("Warning: trips.txt not found; cannot map shapes to routes.")
        return None

    try:
        trips = load_gtfs_data(str(gtfs_dir), files=["trips.txt"])["trips"]
    except (OSError, ValueError) as e:
        logging.warning("Warning: Could not read trips.txt (%s); cannot map shapes to routes.", e)
        return None

    if not {"route_id", "shape_id"}.issubset(trips.columns):
        logging.warning("Warning: trips.txt lacks route_id/shape_id; cannot map shapes to routes.")
        return None

    pairs = trips[["route_id", "shape_id"]].dropna().drop_duplicates()
    pairs = pairs[(pairs["shape_id"].str.strip() != "") & (pairs["route_id"].str.strip() != "")]
    if pairs.empty:
        logging.warning("Warning: trips.txt contains no usable shape/route pairs.")
        return None

    short_lookup: dict[str, str] = {}
    if (gtfs_dir / "routes.txt").exists():
        try:
            routes = load_gtfs_data(str(gtfs_dir), files=["routes.txt"])["routes"]
            if {"route_id", "route_short_name"}.issubset(routes.columns):
                lookup_df = routes[["route_id", "route_short_name"]].dropna()
                short_lookup = dict(zip(lookup_df["route_id"], lookup_df["route_short_name"]))
        except (OSError, ValueError) as e:
            logging.warning("Warning: Could not read routes.txt (%s); using route_id for names.", e)

    pairs = pairs.copy()
    pairs["route_short"] = pairs["route_id"].map(short_lookup)
    return pairs.reset_index(drop=True)


def export_lines_per_route(
    lines_gdf: gpd.GeoDataFrame,
    gtfs_dir: Path,
    output_dir: Path,
) -> None:
    """Write one shapefile per route into the PER_ROUTE_SUBDIR subfolder.

    Route membership comes from the shape_id → route_id mapping in
    `trips.txt`; a shape serving several routes appears in each of their
    files. File names derive from route_short_name when available (falling
    back to route_id), sanitized and deduplicated case-insensitively.

    Shapes not referenced by any trip are written to
    ``unassigned_shapes.shp`` so the split output never silently drops
    geometry. When `trips.txt` is unavailable the function degrades to one
    shapefile per shape_id.

    Args:
        lines_gdf: LineString GeoDataFrame as returned by :func:`read_shapes`.
        gtfs_dir: Path to the directory containing the GTFS files.
        output_dir: Base output directory (the subfolder is created inside).

    Raises:
        IOError: If a shapefile cannot be written.
    """
    if lines_gdf.empty:
        logging.info("Info: Skipping per-route export: no line data.")
        return

    out_dir = output_dir / PER_ROUTE_SUBDIR
    mapping = map_shapes_to_routes(gtfs_dir)

    if mapping is None or mapping.empty:
        logging.warning(
            "Warning: No shape-to-route mapping available; "
            "exporting one shapefile per shape_id instead."
        )
        shape_ids = sorted(lines_gdf["shape_id"].astype(str).unique())
        names = build_export_basenames([(sid, None) for sid in shape_ids], prefix="shape")
        for shape_id, group in lines_gdf.groupby("shape_id"):
            export_gdf(group, out_dir / f"{names[str(shape_id)]}.shp")
        logging.info("Per-shape export complete: %d shapefile(s) in %s", len(names), out_dir)
        return

    merged = lines_gdf.merge(mapping, on="shape_id", how="left")

    unassigned = merged[merged["route_id"].isna()]
    if not unassigned.empty:
        logging.warning(
            "Warning: %d shape(s) are not referenced by any trip; "
            "writing them to unassigned_shapes.shp.",
            unassigned["shape_id"].nunique(),
        )
        export_gdf(unassigned[["shape_id", "geometry"]], out_dir / "unassigned_shapes.shp")

    assigned = merged.dropna(subset=["route_id"])
    if assigned.empty:
        logging.warning("Warning: No shapes could be matched to routes; nothing to split.")
        return

    route_labels = (
        assigned[["route_id", "route_short"]]
        .drop_duplicates(subset="route_id")
        .sort_values("route_id")
    )
    names = build_export_basenames(
        [
            (row.route_id, row.route_short if isinstance(row.route_short, str) else None)
            for row in route_labels.itertuples(index=False)
        ],
        prefix="route",
    )

    for route_id, group in assigned.groupby("route_id"):
        out_gdf = group.rename(columns={"route_short": "rshort"})[
            ["route_id", "rshort", "shape_id", "geometry"]
        ].copy()
        out_gdf["rshort"] = out_gdf["rshort"].fillna("")
        export_gdf(out_gdf, out_dir / f"{names[str(route_id)]}.shp")

    logging.info("Per-route export complete: %d route shapefile(s) in %s", len(names), out_dir)


# --- Main Orchestration Function (Core Logic) ---


def gtfs_to_shapefiles(
    gtfs_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    kind: ExportKind = "both",
    split_by_route: Optional[bool] = None,
) -> None:
    """Converts GTFS stops and/or shapes files to ESRI Shapefiles.

    Reads data from GTFS directory and writes Shapefiles to output directory.
    Uses default paths from the User Configuration section if arguments
    are not provided.

    Args:
        gtfs_dir: Path to the GTFS directory. If None, uses
                  DEFAULT_GTFS_DIR from module configuration.
        output_dir: Path to the output directory. If None, uses
                    DEFAULT_OUTPUT_DIR from module configuration.
        kind: Specifies elements to export ("stops", "lines", "both").
              Defaults to "both".
        split_by_route: If True, additionally write one shapefile per route
                        into the PER_ROUTE_SUBDIR subfolder (applies when
                        lines are exported). If None, uses SPLIT_BY_ROUTE
                        from module configuration.

    Raises:
        ValueError: If required path arguments are None and defaults are also None.
        NotADirectoryError: If resolved gtfs_dir does not exist or is not a directory.
        FileNotFoundError: If 'stops.txt' is required but not found.
        ValueError: If GTFS files have missing columns or invalid data.
        IOError: If shapefiles cannot be written to the output directory.
    """
    # Resolve paths using defaults if arguments are None
    resolved_gtfs_dir = gtfs_dir if gtfs_dir is not None else DEFAULT_GTFS_DIR
    resolved_output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    resolved_split = split_by_route if split_by_route is not None else SPLIT_BY_ROUTE

    # Validate that paths are set either via args or defaults
    if resolved_gtfs_dir is None:
        raise ValueError("GTFS input directory is not specified and no default is set.")
    if resolved_output_dir is None:
        raise ValueError("Output directory is not specified and no default is set.")

    logging.info("-" * 50)
    logging.info("Starting GTFS to Shapefile conversion...")
    logging.info("Input GTFS Directory: %s", resolved_gtfs_dir)
    logging.info("Output Directory: %s", resolved_output_dir)
    logging.info("Export Type: %s", kind)
    logging.info("Split lines by route: %s", resolved_split)
    logging.info("-" * 50)

    if not resolved_gtfs_dir.is_dir():
        raise NotADirectoryError(
            f"Input GTFS directory not found or is not a directory: {resolved_gtfs_dir}"
        )

    # Ensure output directory exists before processing files
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise IOError(f"Could not create output directory {resolved_output_dir}: {e}") from e

    # --- Process Stops ---
    if kind in ("stops", "both"):
        logging.info("\nProcessing Stops...")
        try:
            stops_gdf = read_stops(resolved_gtfs_dir)
            export_gdf(stops_gdf, resolved_output_dir / "gtfs_stops.shp")
        except (FileNotFoundError, ValueError, IOError, NotADirectoryError) as e:
            logging.error("ERROR processing stops: %s", e)
            # Decide if you want to stop or continue if stops fail
            # raise # Uncomment to stop execution on error
        except Exception as e:
            logging.error("An unexpected error occurred during stops processing: %s", e)
            # raise # Uncomment to stop execution on error

    # --- Process Shapes (Lines) ---
    if kind in ("lines", "both"):
        logging.info("\nProcessing Shapes (Lines)...")
        try:
            lines_gdf = read_shapes(resolved_gtfs_dir)
            export_gdf(lines_gdf, resolved_output_dir / "gtfs_lines.shp")
            if resolved_split:
                logging.info("\nProcessing per-route split...")
                export_lines_per_route(lines_gdf, resolved_gtfs_dir, resolved_output_dir)
        except (ValueError, IOError) as e:
            logging.error("ERROR processing shapes: %s", e)
            # raise # Uncomment to stop execution on error
        except Exception as e:
            logging.error("An unexpected error occurred during shapes processing: %s", e)
            # raise # Uncomment to stop execution on error

    logging.info("-" * 50)
    logging.info("Conversion finished.")
    # Provide context requested
    logging.info(
        "Current time: %s", pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    logging.info("-" * 50)


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> int:
    """Run GTFS to Shapefile conversion using the configured default paths.

    Returns:
        Process exit code: 0 on success, 2 if required CONFIGURATION values
        are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if DEFAULT_GTFS_DIR == Path(r"/path/to/your/default_gtfs_folder") or DEFAULT_OUTPUT_DIR == Path(
        r"/path/to/your/default_output_folder"
    ):
        logging.warning(
            "DEFAULT_GTFS_DIR and/or DEFAULT_OUTPUT_DIR are still set to their default "
            "placeholder values. Please update them in the CONFIGURATION section before running."
        )
        return 2
    gtfs_to_shapefiles()
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
