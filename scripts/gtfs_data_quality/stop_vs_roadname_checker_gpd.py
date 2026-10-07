"""Detects potential typos in GTFS stop names using spatial and fuzzy matching.

This script buffers GTFS stops, spatially joins them with nearby roadway
centerlines, and uses fuzzy string comparison to flag discrepancies between
stop names and adjacent road names.

Inputs:
    - GTFS 'stops.txt' file
    - Roadway centerline shapefile
    - Configuration parameters (paths, CRS, buffer distance, similarity threshold)
    - Optional user input for mapping non-standard roadway field names

Outputs:
    - CSV listing potential stop name typos and similarity scores. The CSV is
      rewritten on every run; a run with no findings writes a header-only CSV.

Typical usage:
    Update the paths in the CONFIGURATION section and run from a shell or a
    Jupyter notebook.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Set

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from rapidfuzz import fuzz, process

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to input files
GTFS_FOLDER = r"path\to\your\GTFS\folder"  # Replace with your GTFS folder path

ROADWAYS_PATH = r"path\to\your\roadways.shp"  # Replace with your roadways centerline shapefile path

# Output settings
OUTPUT_DIR = r"path\to\output\directory"  # Replace with your desired output directory
OUTPUT_CSV_NAME = "potential_typos.csv"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)

# Coordinate Reference Systems
STOPS_CRS = "EPSG:4326"  # WGS84 Latitude/Longitude. Typically standard for GTFS stops.
TARGET_CRS = "EPSG:2248"  # Projected CRS for spatial analysis (adjust as needed).

# Processing parameters
SIMILARITY_THRESHOLD = 80  # 0-100, higher number yields fewer results

# Buffer distance configuration
BUFFER_DISTANCE_VALUE = 50
BUFFER_DISTANCE_UNIT = "feet"  # 'feet', 'meters', or 'us survey foot'

# Roadway Shapefile Column Configuration
REQUIRED_COLUMNS_ROADWAY = [
    "RW_PREFIX",
    "RW_TYPE_US",
    "RW_SUFFIX",
    "RW_SUFFIX_",
    "FULLNAME",
]

DESCRIPTIONS_ROADWAY = {
    "RW_PREFIX": "Directional prefix (e.g., 'N' in 'N Washington St')",
    "RW_TYPE_US": "Street type (e.g., 'St' in 'N Washington St')",
    "RW_SUFFIX": "Directional suffix (e.g., 'SE' in 'Park St SE')",
    "RW_SUFFIX_": "Additional suffix (e.g., 'EB' in 'RT267 EB')",
    "FULLNAME": "Full street name",
}

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def get_crs_unit(crs_code: str) -> Optional[str]:
    """Determine the linear unit of a CRS.

    Args:
        crs_code: The CRS code (e.g., "EPSG:4326").

    Returns:
        str or None: The unit name if found, otherwise None.
    """
    try:
        crs = CRS.from_user_input(crs_code)
        if crs.axis_info:
            return crs.axis_info[0].unit_name
        logging.error("CRS has no axis information.")
        return None
    except ValueError as err:
        logging.error("Error determining CRS unit: %s", err)
        return None


# Metres per unit for the buffer units a user may configure. Keys are
# lower-case aliases, including the spellings PyProj reports (e.g. "metre").
UNIT_TO_METRES: Dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "us survey foot": 1200 / 3937,
    "us survey feet": 1200 / 3937,
    "ftus": 1200 / 3937,
}


def convert_buffer_distance(value: float, from_unit: str, crs_code: str) -> float:
    """Convert a buffer distance into the linear units of a projected CRS.

    Uses the CRS's own numeric unit-conversion factor (metres per unit), so
    any projected CRS works regardless of how PyProj spells its unit name.

    Args:
        value (float): The distance value to convert.
        from_unit (str): The unit of the input value; a key of
            :data:`UNIT_TO_METRES` (e.g., "feet", "meters", "metre").
        crs_code (str): The target CRS (e.g., "EPSG:2248").

    Returns:
        float: The distance expressed in the CRS's linear units.

    Raises:
        ValueError: If ``from_unit`` is unknown or the CRS is not projected.
    """
    from_factor = UNIT_TO_METRES.get(from_unit.strip().lower())
    if from_factor is None:
        raise ValueError(
            f"Buffer unit '{from_unit}' not supported; use one of {sorted(UNIT_TO_METRES)}."
        )
    crs = CRS.from_user_input(crs_code)
    if not crs.is_projected or not crs.axis_info:
        raise ValueError(f"CRS {crs_code} is not a projected CRS with linear units.")
    crs_factor = crs.axis_info[0].unit_conversion_factor  # metres per CRS unit
    return value * from_factor / crs_factor


# -----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------


def load_stops(stops_df: pd.DataFrame, crs: str = STOPS_CRS) -> gpd.GeoDataFrame:
    """Validate an in-memory GTFS stops DataFrame and return a GeoDataFrame.

    Args:
        stops_df (pandas.DataFrame): Frame created by `load_gtfs_data(..., files=["stops.txt"])`.
        crs (str, optional): CRS to assign to the resulting GeoDataFrame.
            Defaults to STOPS_CRS.

    Returns:
        geopandas.GeoDataFrame: Stops with point geometries in the requested CRS.

    Raises:
        ValueError: If required columns are missing or a non-blank lat/lon
            cannot be cast to float. Rows with blank coordinates (allowed by
            GTFS for some location types) are dropped with a warning.
    """
    required_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    missing = [c for c in required_cols if c not in stops_df.columns]
    if missing:
        raise ValueError(f"Required columns missing from stops.txt: {', '.join(missing)}")

    # Ensure numeric latitude / longitude
    stops_df = stops_df.copy()
    blank = pd.Series(False, index=stops_df.index)
    for col in ("stop_lat", "stop_lon"):
        text = stops_df[col].astype("string").str.strip()
        blank |= text.isna() | (text == "")
    if blank.any():
        logging.warning("Dropping %d stop(s) with blank coordinates.", int(blank.sum()))
        stops_df = stops_df.loc[~blank].copy()
    stops_df["stop_lat"] = stops_df["stop_lat"].astype(float)
    stops_df["stop_lon"] = stops_df["stop_lon"].astype(float)

    gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs=crs,
    )
    return gdf


def load_roadways(roadways_path: str) -> gpd.GeoDataFrame:
    """Load the roadway shapefile and return a GeoDataFrame.

    Args:
        roadways_path (str): The file path to the roadway shapefile.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the roadway data.
    """
    return gpd.read_file(roadways_path)


# -----------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------


def map_roadway_columns(roadways_gdf: gpd.GeoDataFrame) -> Dict[str, str]:
    """Map the required roadway columns.

    Prompts the user to input the correct column names if missing.

    Args:
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame containing roadway data.

    Returns:
        dict: A dictionary mapping required column names to their actual names in the
        GeoDataFrame (expected -> actual). Use :func:`apply_roadway_column_mapping`
        to give the data the expected names.
    """
    column_mapping = {}
    for col in REQUIRED_COLUMNS_ROADWAY:
        if col in roadways_gdf.columns:
            column_mapping[col] = col
        else:
            logging.warning("The column '%s' is missing from the roadway shapefile.", col)
            logging.info("Description: %s", DESCRIPTIONS_ROADWAY[col])
            logging.info("Available columns: %s", roadways_gdf.columns.tolist())
            new_col = input(
                f"Please enter the correct column name for '{col}' (or leave blank to skip): "
            ).strip()
            while new_col and new_col not in roadways_gdf.columns:
                logging.warning(
                    "'%s' is not among the available columns: %s",
                    new_col,
                    roadways_gdf.columns.tolist(),
                )
                new_col = input(
                    f"Please enter the correct column name for '{col}' (or leave blank to skip): "
                ).strip()
            if new_col:
                column_mapping[col] = new_col
                logging.info("Mapped '%s' to '%s'", col, new_col)
            else:
                logging.info("Skipped mapping for '%s'", col)
    return {k: v for k, v in column_mapping.items() if v is not None}


def apply_roadway_column_mapping(
    roadways_gdf: gpd.GeoDataFrame, column_mapping: Dict[str, str]
) -> gpd.GeoDataFrame:
    """Expose each mapped column under its expected name.

    Args:
        roadways_gdf (gpd.GeoDataFrame): The roadway data.
        column_mapping (dict): Expected name -> actual name, as returned by
            :func:`map_roadway_columns`.

    Returns:
        gpd.GeoDataFrame: A copy in which every expected name is a column.
        Columns are copied rather than renamed, so one source column may
        serve several expected names and existing columns are not clobbered.
    """
    roadways_gdf = roadways_gdf.copy()
    for expected, actual in column_mapping.items():
        if expected != actual:
            roadways_gdf[expected] = roadways_gdf[actual]
    return roadways_gdf


def extract_modifiers(
    roadways_gdf: gpd.GeoDataFrame, column_mapping_roadway: Dict[str, str]
) -> Set[str]:
    """Extract unique modifier values (e.g., street types) from the roadway GeoDataFrame.

    Args:
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame containing roadway data.
        column_mapping_roadway (dict): A dictionary mapping required column names to
            their actual names.

    Returns:
        set: A set of unique, normalized modifier strings.
    """
    modifiers_fields = ["RW_TYPE_US"]
    modifiers = set()
    for field in modifiers_fields:
        mapped_field = column_mapping_roadway.get(field)
        if mapped_field and mapped_field in roadways_gdf.columns:
            unique_vals = roadways_gdf[mapped_field].dropna().unique()
            modifiers.update(unique_vals)
    modifiers = set(
        str(mod).lower().strip() for mod in modifiers if pd.notna(mod) and str(mod).strip()
    )
    return modifiers


def normalize_street_name(name: str, modifiers_set: Set[str]) -> str:
    """Normalize a street name by removing known modifiers, punctuation, and extra spaces.

    Args:
        name (str): The street name to normalize.
        modifiers_set (set): A set of known modifiers to remove from the name.

    Returns:
        str: The normalized street name.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    if modifiers_set:
        pattern = r"\b(" + "|".join(re.escape(m) for m in modifiers_set) + r")\b"
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def create_buffered_stops(stops_gdf: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
    """Create a buffered geometry for each stop.

    Args:
        stops_gdf (gpd.GeoDataFrame): The GeoDataFrame of stops.
        buffer_distance (float): The distance to buffer the stops by.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with a new 'buffered_geometry' column.
    """
    stops_gdf["buffered_geometry"] = stops_gdf.geometry.buffer(buffer_distance)
    return stops_gdf.set_geometry("buffered_geometry")  # type: ignore[no-any-return]


def spatial_join_stops_roadways(
    stops_buffered_gdf: gpd.GeoDataFrame, roadways_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatially join the buffered stops with the roadways.

    Args:
        stops_buffered_gdf (gpd.GeoDataFrame): The GeoDataFrame of buffered stops.
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame of roadways.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame resulting from the spatial join.
    """
    return gpd.sjoin(
        stops_buffered_gdf[["stop_id", "stop_name", "buffered_geometry"]],
        roadways_gdf[["FULLNAME", "FULLNAME_clean", "geometry"]],
        how="left",
        predicate="intersects",
    )


def extract_street_names(stop_name: str, modifiers: Set[str]) -> List[str]:
    """Extract potential street names from a stop name using common separators.

    Args:
        stop_name (str): The name of the stop.
        modifiers (set): A set of known modifiers to assist in normalization.

    Returns:
        list: A list of normalized street names extracted from the stop name.
    """
    if pd.isna(stop_name) or not isinstance(stop_name, str):
        return []
    # Symbol separators split with or without surrounding spaces ("A&B", "A @ B").
    pattern = r"\s*[@&/+]\s*| and | intersection of "
    streets = re.split(pattern, stop_name, flags=re.IGNORECASE)
    return [normalize_street_name(street, modifiers) for street in streets if street]


def compare_stop_to_roads(
    stop_id: str,
    stop_name: str,
    stop_streets: List[str],
    road_names: Mapping[str, Set[str]],
    threshold: int,
) -> List[Dict[str, Any]]:
    """Compare each portion of the stop name to known road names via fuzzy matching.

    A street that exactly equals a nearby normalized road name is not
    reported. Any other street scoring at or above ``threshold`` is reported,
    including a score of 100: ``token_set_ratio`` scores subsets (e.g.
    "mill" vs. "old mill"), reordered words, and duplicated words as 100,
    and those unequal names are discrepancies worth flagging.

    Args:
        stop_id (str): The ID of the stop.
        stop_name (str): The original name of the stop.
        stop_streets (list): A list of potential street names extracted from the stop
            name.
        road_names (Mapping[str, set]): Normalized name of each road inside the
            stop's buffer -> the original names of those same road features.
        threshold (int): The similarity score threshold (0-100) for considering a
            match.

    Returns:
        list[dict]: A list of dictionaries, each representing a potential typo.
    """
    potential_typos_list = []
    candidates = list(road_names)
    for street in stop_streets:
        if not street or street in road_names:
            continue
        match_tuples = process.extract(street, candidates, scorer=fuzz.token_set_ratio, limit=3)
        for match_clean, score, _ in match_tuples:
            if score >= threshold:
                for original_match in sorted(road_names[match_clean]):
                    potential_typos_list.append(
                        {
                            "stop_id": stop_id,
                            "stop_name": stop_name,
                            "street_in_stop_name": street,
                            "similar_road_name_clean": match_clean,
                            "similar_road_name_original": original_match,
                            "similarity_score": score,
                        }
                    )
    return potential_typos_list


def process_typos(
    stops_gdf: gpd.GeoDataFrame,
    modifiers: Set[str],
    join_gdf: gpd.GeoDataFrame,
    threshold: int,
) -> pd.DataFrame:
    """Process each stop and perform fuzzy matching to identify potential typos.

    Fuzzy comparison is restricted to the roads that intersect each stop's
    buffer (the per-stop local set), as determined by ``join_gdf``. A stop is
    therefore never compared against a similarly-named road elsewhere in the
    region, and reported original road names come only from the road
    features that actually intersect the buffer.

    Args:
        stops_gdf (gpd.GeoDataFrame): The GeoDataFrame of stops.
        modifiers (set): A set of known street name modifiers.
        join_gdf (gpd.GeoDataFrame): Output of
            :func:`spatial_join_stops_roadways`. Each stop is compared only
            against roads inside its own buffer.
        threshold (int): The similarity score threshold for fuzzy matching.

    Returns:
        pd.DataFrame: A deduplicated DataFrame of potential typos, sorted by
        similarity score. Empty DataFrame if no candidates are found.
    """
    # Build per-stop {normalized name -> original names} from the spatial join,
    # so both the comparison set and the reported names are buffer-local.
    local = join_gdf.dropna(subset=["FULLNAME_clean", "FULLNAME"])
    nearby_by_stop: Dict[str, Dict[str, Set[str]]] = {}
    for s_id, clean, original in zip(local["stop_id"], local["FULLNAME_clean"], local["FULLNAME"]):
        nearby_by_stop.setdefault(s_id, {}).setdefault(clean, set()).add(original)

    potential_typos: List[Dict[str, Any]] = []
    for _, stop in stops_gdf.iterrows():
        s_id = stop["stop_id"]
        s_name = stop["stop_name"]
        s_streets = extract_street_names(s_name, modifiers)

        local_road_names = nearby_by_stop.get(s_id, {})
        if not local_road_names:
            # No roads within this stop's buffer -- nothing to compare against.
            continue

        typos = compare_stop_to_roads(s_id, s_name, s_streets, local_road_names, threshold)
        potential_typos.extend(typos)

    logging.info("Total potential typos found before deduplication: %d", len(potential_typos))
    if not potential_typos:
        return pd.DataFrame(
            columns=[
                "stop_id",
                "stop_name",
                "street_in_stop_name",
                "similar_road_name_clean",
                "similar_road_name_original",
                "similarity_score",
            ]
        )
    typos_df = pd.DataFrame(potential_typos)
    typos_df_sorted = typos_df.sort_values(by="similarity_score", ascending=False).drop_duplicates()
    return typos_df_sorted


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


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
            Pandas' default NA parsing is disabled (``keep_default_na=False``),
            so values such as ``"NA"`` stay literal strings and empty fields
            load as ``""``.
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
                        os.path.join(gtfs_path, file_name),
                        dtype=dtype,
                        keep_default_na=False,
                        low_memory=False,
                    )
                else:
                    with archive.open(resolved[file_name]) as handle:
                        df = pd.read_csv(
                            handle, dtype=dtype, keep_default_na=False, low_memory=False
                        )
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


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Entry point for the GTFS stop-vs-road typo-checker script.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    # ------------------------------------------------------------------
    # 1. Configure logging *inside* main so importing this module is silent
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if GTFS_FOLDER == r"path\to\your\GTFS\folder" or OUTPUT_DIR == r"path\to\output\directory":
        logging.warning(
            "GTFS_FOLDER and/or OUTPUT_DIR are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2
    logging.info("Starting processing …")

    # ------------------------------------------------------------------
    # 2. Ensure the output directory exists
    # ------------------------------------------------------------------
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info("Created output directory %s", OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 3. Load GTFS data (only stops.txt is required for this task)
    # ------------------------------------------------------------------
    gtfs_data = load_gtfs_data(GTFS_FOLDER, files=["stops.txt"])
    stops_df = gtfs_data["stops"]  # key name = file name w/o ".txt"
    stops_gdf = load_stops(stops_df)  # validate and convert to GDF

    # 4. Load roadway shapefile
    roadways_gdf = load_roadways(ROADWAYS_PATH)

    # 5. Re-project both layers to TARGET_CRS
    stops_gdf = stops_gdf.to_crs(TARGET_CRS)
    roadways_gdf = roadways_gdf.to_crs(TARGET_CRS)

    # ------------------------------------------------------------------
    # 6. Map roadway columns (prompting user if needed)
    # ------------------------------------------------------------------
    column_mapping = map_roadway_columns(roadways_gdf)
    if not column_mapping.get("FULLNAME"):
        raise ValueError("The 'FULLNAME' column is required in the roadway data.")
    roadways_gdf = apply_roadway_column_mapping(roadways_gdf, column_mapping)

    # 7. Extract modifiers and normalise roadway names. Mapped columns now
    #    carry their expected names, so look them up by those names.
    modifiers = extract_modifiers(roadways_gdf, {col: col for col in column_mapping})
    logging.info("Extracted modifiers (%d): %s", len(modifiers), modifiers)
    roadways_gdf["FULLNAME_clean"] = roadways_gdf["FULLNAME"].apply(
        lambda x: normalize_street_name(x, modifiers)
    )

    # ------------------------------------------------------------------
    # 8. Compute buffer distance in target CRS units
    # ------------------------------------------------------------------
    buffer_distance = convert_buffer_distance(
        BUFFER_DISTANCE_VALUE, BUFFER_DISTANCE_UNIT, TARGET_CRS
    )
    logging.info(
        "Buffer distance: %s %s = %.4f %s",
        BUFFER_DISTANCE_VALUE,
        BUFFER_DISTANCE_UNIT,
        buffer_distance,
        get_crs_unit(TARGET_CRS),
    )

    # 9. Buffer stops, spatial-join with roadways
    stops_buffered = create_buffered_stops(stops_gdf, buffer_distance)
    join_gdf = spatial_join_stops_roadways(stops_buffered, roadways_gdf)
    logging.info("Spatial join produced %d candidate matches", join_gdf.shape[0])

    # ------------------------------------------------------------------
    # 10. Fuzzy-match street names to find potential typos
    # ------------------------------------------------------------------
    typos_df = process_typos(
        stops_gdf,
        modifiers,
        join_gdf,
        SIMILARITY_THRESHOLD,
    )

    # 11. Save results. Always write, so a clean run replaces stale findings
    #     from an earlier run with a header-only CSV.
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    typos_df.to_csv(out_path, index=False)
    if typos_df.empty:
        logging.info("No potential typos found; wrote header-only CSV to %s", out_path)
    else:
        logging.info("%d potential typos saved to %s", len(typos_df), out_path)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
