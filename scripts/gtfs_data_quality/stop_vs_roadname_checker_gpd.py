"""Detects potential typos in GTFS stop names using spatial and fuzzy matching.

This script buffers GTFS stops, spatially joins them with nearby roadway
centerlines, and uses fuzzy string comparison to flag discrepancies between
stop names and adjacent road names.

To reduce false positives, suspected fixed-width-truncated stop names are
flagged in a separate report and excluded from matching (keeping a planner in
the loop), common abbreviations are expanded before comparison, and substring
containment matches (abbreviations/partials) are suppressed.

Inputs:
    - GTFS 'stops.txt' file
    - Roadway centerline shapefile
    - Configuration parameters (paths, CRS, buffer distance, similarity threshold,
      stop-identifier field, truncation detection, false-positive filters)
    - Optional user input for mapping non-standard roadway field names

Outputs:
    - CSV listing potential stop name typos and similarity scores
    - CSV listing suspected-truncated stop names for manual review
"""

import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
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

# Stop identifier field used to label rows in the output. "stop_id" is required
# by the GTFS spec and always present; "stop_code" is the optional public-facing
# code. If STOP_ID_FIELD is set to a column that is absent from stops.txt, the
# script falls back to "stop_id" with a warning.
STOP_ID_FIELD = "stop_id"  # "stop_id" or "stop_code"

# Coordinate Reference Systems
STOPS_CRS = "EPSG:4326"  # WGS84 Latitude/Longitude. Typically standard for GTFS stops.
TARGET_CRS = "EPSG:2248"  # Projected CRS for spatial analysis (adjust as needed).

# Processing parameters
SIMILARITY_THRESHOLD = 80  # 0-100, higher number yields fewer results

# -----------------------------------------------------------------------------
# Length-truncation detection
# -----------------------------------------------------------------------------
# Legacy AVL / scheduling systems frequently truncate ``stop_name`` to a fixed
# character width. Truncated names ("Martin Luther King Jr Av" -> "Martin Luther
# King Jr") produce noisy, low-confidence fuzzy matches, so they are flagged in a
# separate report and excluded from typo matching rather than silently dropped.
# This keeps a planner in the loop: the truncated stops are surfaced for manual
# review instead of polluting the typo output.
DETECT_TRUNCATION = True
# Manual override. When set to an int, any stop whose name length is >= this value
# is treated as suspected-truncated. Leave as None to auto-detect the truncation
# width from the stop-name length distribution.
TRUNCATION_LENGTH: Optional[int] = None
# Auto-detection guardrails: the longest observed name length is treated as a
# truncation artifact only if at least TRUNCATION_MIN_COUNT stops *and* at least
# TRUNCATION_MIN_FRACTION of all stops share that exact length (i.e. a spike at
# the maximum width, which is the signature of fixed-width truncation).
TRUNCATION_MIN_COUNT = 5
TRUNCATION_MIN_FRACTION = 0.02
OUTPUT_TRUNCATED_CSV_NAME = "suspected_truncated_stops.csv"

# -----------------------------------------------------------------------------
# False-positive filters
# -----------------------------------------------------------------------------
# When True, a stop/road pair is not flagged as a typo if one normalized name
# fully contains the other (e.g. "washington" vs "washington heights"). These are
# abbreviations or partial names, not misspellings.
FILTER_SUBSTRING_CONTAINMENT = True
# When True, common abbreviations are expanded before comparison so that, e.g.,
# "Ft Hunt" matches "Fort Hunt" instead of being flagged as a typo. Street-type
# modifiers (St, Ave, Rd, ...) are already removed during normalization, so this
# map deliberately targets name-body abbreviations rather than street types.
EXPAND_ABBREVIATIONS = True
ABBREVIATIONS: Dict[str, str] = {
    "ft": "fort",
    "mt": "mount",
    "mtn": "mountain",
    "jct": "junction",
    "spgs": "springs",
    "hts": "heights",
    "pt": "point",
    "ctr": "center",
}

# Buffer distance configuration
BUFFER_DISTANCE_VALUE = 50
BUFFER_DISTANCE_UNIT = "feet"  # 'feet' or 'meters'

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


def convert_buffer_distance(value: float, from_unit: str, to_unit: str) -> float:
    """Convert buffer distance from `from_unit` to `to_unit` using known conversion factors.

    Args:
        value (float): The distance value to convert.
        from_unit (str): The unit of the input value (e.g., "feet", "meters").
        to_unit (str): The desired unit for the output value (e.g., "feet", "meters").

    Returns:
        float: The converted distance value.

    Raises:
        ValueError: If the conversion from `from_unit` to `to_unit` is not supported.
    """
    conversion_factors = {
        ("feet", "meters"): 0.3048,
        ("meters", "feet"): 3.28084,
        ("metre", "feet"): 3.28084,
        ("us survey foot", "meters"): 0.3048006096012192,
        ("meters", "us survey foot"): 3.280833333333333,
        ("feet", "us survey foot"): 0.999998,
        ("us survey foot", "feet"): 1.000002,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversion_factors:
        return value * conversion_factors[key]
    raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported.")


# -----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------


def resolve_stop_id_field(stops_df: pd.DataFrame, preferred: str = STOP_ID_FIELD) -> str:
    """Return the stop-identifier column to use, falling back to ``stop_id``.

    ``stop_id`` is mandatory in GTFS, whereas ``stop_code`` is optional. If the
    preferred field is absent, the caller is warned and ``stop_id`` is used.

    Args:
        stops_df (pandas.DataFrame): Parsed ``stops.txt``.
        preferred (str, optional): Configured identifier field. Defaults to
            STOP_ID_FIELD.

    Returns:
        str: ``preferred`` if present, otherwise ``"stop_id"``.
    """
    if preferred in stops_df.columns:
        return preferred
    logging.warning(
        "STOP_ID_FIELD '%s' is not present in stops.txt; falling back to 'stop_id'.",
        preferred,
    )
    return "stop_id"


def load_stops(stops_df: pd.DataFrame, crs: str = STOPS_CRS) -> gpd.GeoDataFrame:
    """Validate an in-memory GTFS stops DataFrame and return a GeoDataFrame.

    Args:
        stops_df (pandas.DataFrame): Frame created by `load_gtfs_data(..., files=["stops.txt"])`.
        crs (str, optional): CRS to assign to the resulting GeoDataFrame.
            Defaults to STOPS_CRS.

    Returns:
        geopandas.GeoDataFrame: Stops with point geometries in the requested CRS.

    Raises:
        ValueError: If required columns are missing or lat/lon cannot be cast to float.
    """
    required_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    missing = [c for c in required_cols if c not in stops_df.columns]
    if missing:
        raise ValueError(f"Required columns missing from stops.txt: {', '.join(missing)}")

    # Ensure numeric latitude / longitude
    stops_df = stops_df.copy()
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
        dict: A dictionary mapping required column names to their actual names in the GeoDataFrame.
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


def normalize_street_name(
    name: str,
    modifiers_set: Set[str],
    abbreviations: Optional[Mapping[str, str]] = None,
) -> str:
    """Normalize a street name by removing known modifiers, punctuation, and extra spaces.

    Args:
        name (str): The street name to normalize.
        modifiers_set (set): A set of known modifiers to remove from the name.
        abbreviations (Mapping[str, str], optional): Token-level abbreviation map
            applied after cleaning so that, e.g., ``"ft" -> "fort"``. When ``None``
            (the default) no expansion is performed.

    Returns:
        str: The normalized street name.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    if modifiers_set:
        pattern = r"\b(" + "|".join(re.escape(m) for m in modifiers_set) + r")\b"
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    if abbreviations:
        name = " ".join(abbreviations.get(token, token) for token in name.split())
    return name


def detect_truncation_length(
    stop_names: Iterable[Any],
    min_count: int = TRUNCATION_MIN_COUNT,
    min_fraction: float = TRUNCATION_MIN_FRACTION,
) -> Optional[int]:
    """Infer a fixed-width truncation length from a collection of stop names.

    Fixed-width truncation leaves a tell-tale spike of names at the maximum
    observed length. The longest length is reported as the truncation width only
    if enough stops cluster there (both an absolute count and a fraction of all
    stops), which avoids mistaking a single long name for truncation.

    Args:
        stop_names (Iterable): Stop-name values (non-strings are ignored).
        min_count (int, optional): Minimum number of stops at the maximum length.
            Defaults to TRUNCATION_MIN_COUNT.
        min_fraction (float, optional): Minimum share of all stops at the maximum
            length. Defaults to TRUNCATION_MIN_FRACTION.

    Returns:
        int or None: The suspected truncation width, or None if no spike is found.
    """
    lengths = [len(name) for name in stop_names if isinstance(name, str) and name]
    if not lengths:
        return None
    max_len = max(lengths)
    count_at_max = sum(1 for length in lengths if length == max_len)
    if count_at_max >= min_count and (count_at_max / len(lengths)) >= min_fraction:
        return max_len
    return None


def flag_truncated_stops(
    stops_gdf: gpd.GeoDataFrame,
    truncation_length: Optional[int],
    stop_id_field: str = "stop_id",
) -> pd.DataFrame:
    """Return stops whose names are at/above the suspected truncation width.

    Args:
        stops_gdf (gpd.GeoDataFrame): The GeoDataFrame of stops.
        truncation_length (int or None): Suspected truncation width. When None, no
            stop is flagged and an empty (but correctly-columned) frame is returned.
        stop_id_field (str, optional): Identifier column to include. Defaults to
            ``"stop_id"``.

    Returns:
        pd.DataFrame: Columns ``[stop_id_field, "stop_name", "stop_name_length",
        "suspected_truncation_length"]`` for each flagged stop.
    """
    columns = [stop_id_field, "stop_name", "stop_name_length", "suspected_truncation_length"]
    if truncation_length is None:
        return pd.DataFrame(columns=columns)

    name_lengths = stops_gdf["stop_name"].apply(lambda s: len(s) if isinstance(s, str) else 0)
    mask = name_lengths >= truncation_length
    flagged = stops_gdf.loc[mask, [stop_id_field, "stop_name"]].copy()
    flagged["stop_name_length"] = name_lengths[mask].to_numpy()
    flagged["suspected_truncation_length"] = truncation_length
    return flagged.reset_index(drop=True)


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
    stops_buffered_gdf: gpd.GeoDataFrame,
    roadways_gdf: gpd.GeoDataFrame,
    stop_id_field: str = "stop_id",
) -> gpd.GeoDataFrame:
    """Spatially join the buffered stops with the roadways.

    Args:
        stops_buffered_gdf (gpd.GeoDataFrame): The GeoDataFrame of buffered stops.
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame of roadways.
        stop_id_field (str, optional): Stop-identifier column to carry through the
            join. Defaults to ``"stop_id"``.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame resulting from the spatial join.
    """
    return gpd.sjoin(
        stops_buffered_gdf[[stop_id_field, "stop_name", "buffered_geometry"]],
        roadways_gdf[["FULLNAME", "FULLNAME_clean", "geometry"]],
        how="left",
        predicate="intersects",
    )


def extract_street_names(
    stop_name: str,
    modifiers: Set[str],
    abbreviations: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Extract potential street names from a stop name using common separators.

    Args:
        stop_name (str): The name of the stop.
        modifiers (set): A set of known modifiers to assist in normalization.
        abbreviations (Mapping[str, str], optional): Abbreviation map forwarded to
            :func:`normalize_street_name`.

    Returns:
        list: A list of normalized street names extracted from the stop name.
    """
    if pd.isna(stop_name) or not isinstance(stop_name, str):
        return []
    separators = [" @ ", " and ", " & ", "/", " intersection of "]
    pattern = "|".join(map(re.escape, separators))
    streets = re.split(pattern, stop_name, flags=re.IGNORECASE)
    return [normalize_street_name(street, modifiers, abbreviations) for street in streets if street]


def is_substring_match(street: str, road_name: str) -> bool:
    """Return True if one normalized name fully contains the other (token-aware).

    Used to suppress false positives where a stop fragment is an abbreviation or
    partial form of a road name (e.g. ``"washington"`` vs ``"washington heights"``)
    rather than a misspelling. Whole-token containment is required so that, e.g.,
    ``"oak"`` does not match ``"oakland"``.

    Args:
        street (str): Normalized street fragment from the stop name.
        road_name (str): Normalized road name.

    Returns:
        bool: True if either name's tokens are a contiguous subsequence of the other.
    """
    if not street or not road_name:
        return False
    street_tokens = street.split()
    road_tokens = road_name.split()
    shorter, longer = (
        (street_tokens, road_tokens)
        if len(street_tokens) <= len(road_tokens)
        else (road_tokens, street_tokens)
    )
    span = len(shorter)
    return any(longer[i : i + span] == shorter for i in range(len(longer) - span + 1))


def compare_stop_to_roads(
    stop_id: str,
    stop_name: str,
    stop_streets: List[str],
    road_names: Set[str],
    roads_gdf: gpd.GeoDataFrame,
    threshold: int,
    stop_id_field: str = "stop_id",
    filter_substring: bool = False,
) -> List[Dict[str, Any]]:
    """Compare each portion of the stop name to known road names via fuzzy matching.

    Args:
        stop_id (str): The ID of the stop.
        stop_name (str): The original name of the stop.
        stop_streets (list): A list of potential street names extracted from the stop
            name.
        road_names (list): A list of normalized road names for comparison.
        roads_gdf (gpd.GeoDataFrame): The GeoDataFrame of roadways, used to retrieve
            original road names.
        threshold (int): The similarity score threshold (0-100) for considering a
            match.
        stop_id_field (str, optional): Key used for the identifier column in each
            output row. Defaults to ``"stop_id"``.
        filter_substring (bool, optional): When True, skip pairs where one
            normalized name fully contains the other (likely abbreviation, not a
            typo). Defaults to False.

    Returns:
        list[dict]: A list of dictionaries, each representing a potential typo.
    """
    potential_typos_list = []
    for street in stop_streets:
        if street in road_names:
            continue
        match_tuples = process.extract(street, road_names, scorer=fuzz.token_set_ratio, limit=3)
        for match_clean, score, _ in match_tuples:
            if threshold <= score < 100:
                if filter_substring and is_substring_match(street, match_clean):
                    continue
                original_matches = roads_gdf.loc[
                    roads_gdf["FULLNAME_clean"] == match_clean, "FULLNAME"
                ].unique()
                for original_match in original_matches:
                    potential_typos_list.append(
                        {
                            stop_id_field: stop_id,
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
    roadways_gdf: gpd.GeoDataFrame,
    modifiers: Set[str],
    join_gdf: gpd.GeoDataFrame,
    threshold: int,
    stop_id_field: str = "stop_id",
    skip_ids: Optional[Set[str]] = None,
    abbreviations: Optional[Mapping[str, str]] = None,
    filter_substring: bool = False,
) -> pd.DataFrame:
    """Process each stop and perform fuzzy matching to identify potential typos.

    Fuzzy comparison is restricted to the roads that intersect each stop's
    buffer (the per-stop local set), as determined by ``join_gdf``. A stop is
    therefore never compared against a similarly-named road elsewhere in the
    region.

    Args:
        stops_gdf (gpd.GeoDataFrame): The GeoDataFrame of stops.
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame of roadways.
        modifiers (set): A set of known street name modifiers.
        join_gdf (gpd.GeoDataFrame): Output of
            :func:`spatial_join_stops_roadways`. Each stop is compared only
            against roads inside its own buffer.
        threshold (int): The similarity score threshold for fuzzy matching.
        stop_id_field (str, optional): Stop-identifier column. Defaults to
            ``"stop_id"``.
        skip_ids (set, optional): Stop identifiers to exclude from matching (e.g.
            suspected-truncated stops reported separately). Defaults to None.
        abbreviations (Mapping[str, str], optional): Abbreviation map applied when
            normalizing stop-name fragments. Defaults to None.
        filter_substring (bool, optional): Forwarded to
            :func:`compare_stop_to_roads` to drop substring-containment matches.
            Defaults to False.

    Returns:
        pd.DataFrame: A deduplicated DataFrame of potential typos, sorted by
        similarity score. Empty DataFrame if no candidates are found.
    """
    skip_ids = skip_ids or set()

    # Build per-stop nearby-road sets from the spatial join.
    local = join_gdf.dropna(subset=["FULLNAME_clean"])
    nearby_clean_by_stop: Dict[str, Set[str]] = (
        local.groupby(stop_id_field)["FULLNAME_clean"].apply(lambda s: set(s.unique())).to_dict()
    )

    potential_typos: List[Dict[str, Any]] = []
    for _, stop in stops_gdf.iterrows():
        s_id = stop[stop_id_field]
        if s_id in skip_ids:
            continue
        s_name = stop["stop_name"]
        s_streets = extract_street_names(s_name, modifiers, abbreviations)

        local_road_names = nearby_clean_by_stop.get(s_id, set())
        if not local_road_names:
            # No roads within this stop's buffer -- nothing to compare against.
            continue
        local_roads_gdf = roadways_gdf[roadways_gdf["FULLNAME_clean"].isin(local_road_names)]

        typos = compare_stop_to_roads(
            s_id,
            s_name,
            s_streets,
            local_road_names,
            local_roads_gdf,
            threshold,
            stop_id_field=stop_id_field,
            filter_substring=filter_substring,
        )
        potential_typos.extend(typos)

    logging.info("Total potential typos found before deduplication: %d", len(potential_typos))
    if not potential_typos:
        return pd.DataFrame(
            columns=[
                stop_id_field,
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
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

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

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry point for the GTFS stop-vs-road typo-checker script."""
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
        return
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
    stop_id_field = resolve_stop_id_field(stops_df, STOP_ID_FIELD)
    logging.info("Using '%s' as the stop identifier field.", stop_id_field)
    stops_gdf = load_stops(stops_df)  # validate and convert to GDF

    # Abbreviation map applied during normalization (None disables expansion).
    abbreviations = ABBREVIATIONS if EXPAND_ABBREVIATIONS else None

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
    roadways_gdf = roadways_gdf.rename(columns=column_mapping)

    # 7. Extract modifiers and normalise roadway names
    modifiers = extract_modifiers(roadways_gdf, column_mapping)
    logging.info("Extracted modifiers (%d): %s", len(modifiers), modifiers)
    roadways_gdf["FULLNAME_clean"] = roadways_gdf["FULLNAME"].apply(
        lambda x: normalize_street_name(x, modifiers, abbreviations)
    )

    # ------------------------------------------------------------------
    # 8. Compute buffer distance in target CRS units
    # ------------------------------------------------------------------
    crs_unit = get_crs_unit(TARGET_CRS)
    if crs_unit is None:
        raise ValueError("Unable to determine the unit for TARGET_CRS.")
    buffer_distance = (
        convert_buffer_distance(BUFFER_DISTANCE_VALUE, BUFFER_DISTANCE_UNIT, crs_unit)
        if BUFFER_DISTANCE_UNIT.lower() != crs_unit.lower()
        else BUFFER_DISTANCE_VALUE
    )

    # ------------------------------------------------------------------
    # 9. Flag suspected-truncated stop names (planner-in-the-loop)
    #    These are reported separately and excluded from fuzzy matching so
    #    they don't generate noisy, low-confidence "typos".
    # ------------------------------------------------------------------
    skip_ids: Set[str] = set()
    if DETECT_TRUNCATION:
        truncation_length = (
            TRUNCATION_LENGTH
            if TRUNCATION_LENGTH is not None
            else detect_truncation_length(stops_gdf["stop_name"])
        )
        if truncation_length is not None:
            logging.info("Suspected truncation width: %d characters.", truncation_length)
            truncated_df = flag_truncated_stops(stops_gdf, truncation_length, stop_id_field)
            if not truncated_df.empty:
                skip_ids = set(truncated_df[stop_id_field])
                trunc_path = os.path.join(OUTPUT_DIR, OUTPUT_TRUNCATED_CSV_NAME)
                truncated_df.to_csv(trunc_path, index=False)
                logging.info(
                    "Flagged %d suspected-truncated stop(s) -> %s (excluded from matching).",
                    len(truncated_df),
                    trunc_path,
                )
        else:
            logging.info("No fixed-width truncation pattern detected.")

    # 10. Buffer stops, spatial-join with roadways
    stops_buffered = create_buffered_stops(stops_gdf, buffer_distance)
    join_gdf = spatial_join_stops_roadways(stops_buffered, roadways_gdf, stop_id_field)
    logging.info("Spatial join produced %d candidate matches", join_gdf.shape[0])

    # ------------------------------------------------------------------
    # 11. Fuzzy-match street names to find potential typos
    # ------------------------------------------------------------------
    typos_df = process_typos(
        stops_gdf,
        roadways_gdf,
        modifiers,
        join_gdf,
        SIMILARITY_THRESHOLD,
        stop_id_field=stop_id_field,
        skip_ids=skip_ids,
        abbreviations=abbreviations,
        filter_substring=FILTER_SUBSTRING_CONTAINMENT,
    )

    # 12. Save or report results
    if typos_df.empty:
        logging.info("No potential typos found.")
    else:
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
        typos_df.to_csv(out_path, index=False)
        logging.info("Potential typos saved to %s", out_path)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
