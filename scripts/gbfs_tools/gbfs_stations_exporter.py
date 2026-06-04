"""Exports GBFS static station information to ESRI Shapefile and GeoJSON.

Reads a GBFS feed's ``station_information.json`` (the static docked-station
inventory) and writes the stations as WGS 84 point features. The source may
be a GBFS auto-discovery URL (``gbfs.json``), a direct
``station_information.json`` URL, or a path to a local JSON file already
downloaded from a feed.

Both GBFS 2.x (where ``name`` is a plain string) and GBFS 3.x (where ``name``
is an array of localized ``{"text", "language"}`` objects) are supported.

Inputs:
    - GBFS source: ``gbfs.json`` URL, ``station_information.json`` URL, or a
      local JSON file path
    - Optional export formats: "shapefile", "geojson", or "both"

Outputs:
    - `gbfs_stations.shp`: Shapefile of static station points
    - `gbfs_stations.geojson`: GeoJSON of static station points
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

# ===========================================================================
# CONFIGURATION
# ===========================================================================

GBFS_CRS = "EPSG:4326"  # GBFS coordinates are always WGS 84
# Type alias for export choices for clarity
ExportKind = Literal["shapefile", "geojson", "both"]

# REQUIRED: Default GBFS source. May be a gbfs.json auto-discovery URL, a
# direct station_information.json URL, or a local JSON file path.
DEFAULT_GBFS_SOURCE: Optional[str] = r"https://example.com/gbfs/gbfs.json"  # <-- EDIT ME

# REQUIRED: Default path to the directory where outputs will be saved
DEFAULT_OUTPUT_DIR: Optional[Path] = Path(r"/path/to/your/default_output_folder")  # <-- EDIT ME
# Set to None if you always want to provide paths as arguments
# DEFAULT_GBFS_SOURCE = None
# DEFAULT_OUTPUT_DIR = None

# Preferred language code for localized station names (GBFS 3.x). Falls back
# to the first available language if this code is not present.
PREFERRED_LANGUAGE: str = "en"

# Network request timeout, in seconds
REQUEST_TIMEOUT: int = 30

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# ===========================================================================
# FUNCTIONS
# ===========================================================================


def _is_url(source: str) -> bool:
    """Returns True if ``source`` looks like an http(s) URL."""
    return urlparse(source).scheme in ("http", "https")


def fetch_json(source: str) -> dict[str, Any]:
    """Loads a JSON document from a URL or local file path.

    Args:
        source: An http(s) URL or a local filesystem path to a JSON file.

    Returns:
        The parsed JSON document as a dictionary.

    Raises:
        IOError: If the URL cannot be retrieved or the local file is missing.
        ValueError: If the content is not valid JSON.
    """
    if _is_url(source):
        try:
            response = requests.get(source, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise IOError(f"Could not fetch GBFS source '{source}': {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise ValueError(f"Response from '{source}' is not valid JSON: {exc}") from exc

    path = Path(source)
    if not path.is_file():
        raise IOError(f"GBFS source file not found: {source}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"File '{source}' is not valid JSON: {exc}") from exc


def resolve_station_information_url(source: str) -> str:
    """Resolves a GBFS source to a ``station_information`` URL or file path.

    If ``source`` is a GBFS auto-discovery document (``gbfs.json``), the feed
    list is searched for the ``station_information`` feed. Otherwise the source
    is assumed to already point at ``station_information.json`` (URL or file)
    and is returned unchanged.

    Args:
        source: A GBFS auto-discovery URL/path, or a direct
            ``station_information.json`` URL/path.

    Returns:
        A URL or file path to the ``station_information`` feed.

    Raises:
        IOError: If the auto-discovery document cannot be retrieved.
        ValueError: If a discovery document is supplied but contains no
            ``station_information`` feed.
    """
    # Direct station_information references are used as-is.
    if "station_information" in source:
        return source

    doc = fetch_json(source)

    # Only auto-discovery documents expose a feed catalog under "data".
    data = doc.get("data")
    if not isinstance(data, dict):
        # Not a discovery document; assume the caller already has the right URL.
        return source

    # GBFS 2.x nests feeds per language: data[<lang>]["feeds"].
    # GBFS 3.x flattens them: data["feeds"].
    feeds: list[dict[str, Any]] = []
    if "feeds" in data and isinstance(data["feeds"], list):
        feeds = data["feeds"]
    else:
        for lang_block in data.values():
            if isinstance(lang_block, dict) and isinstance(lang_block.get("feeds"), list):
                feeds = lang_block["feeds"]
                break

    for feed in feeds:
        if feed.get("name") == "station_information" and feed.get("url"):
            logging.info("Resolved station_information feed: %s", feed["url"])
            return str(feed["url"])

    raise ValueError(f"No 'station_information' feed found in GBFS discovery document '{source}'.")


def _extract_name(raw_name: Any) -> Optional[str]:
    """Normalizes a GBFS station name to a plain string.

    Handles GBFS 2.x string names and GBFS 3.x localized arrays of
    ``{"text", "language"}`` objects.

    Args:
        raw_name: The ``name`` value from a station record.

    Returns:
        A station name string, or None if no usable value is present.
    """
    if isinstance(raw_name, str):
        return raw_name

    if isinstance(raw_name, list) and raw_name:
        preferred = [
            entry.get("text")
            for entry in raw_name
            if isinstance(entry, dict) and entry.get("language") == PREFERRED_LANGUAGE
        ]
        if preferred and preferred[0]:
            return str(preferred[0])
        # Fall back to the first entry that carries text.
        for entry in raw_name:
            if isinstance(entry, dict) and entry.get("text"):
                return str(entry["text"])

    return None


def build_stations_gdf(station_info: dict[str, Any]) -> gpd.GeoDataFrame:
    """Builds a point GeoDataFrame from a ``station_information`` document.

    Args:
        station_info: Parsed ``station_information.json`` document.

    Returns:
        A GeoDataFrame of station points in WGS 84. Returns an empty
        GeoDataFrame if no valid stations are present.

    Raises:
        ValueError: If the document does not contain a station list.
    """
    stations = station_info.get("data", {}).get("stations")
    if not isinstance(stations, list):
        raise ValueError("GBFS document does not contain 'data.stations' list.")

    df = pd.DataFrame(stations)
    if df.empty:
        logging.warning("Warning: station_information contains no stations.")
        return gpd.GeoDataFrame(
            columns=["station_id", "name", "lat", "lon", "geometry"],
            geometry=[],
            crs=GBFS_CRS,
        )

    required = {"station_id", "lat", "lon"}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"Missing required station fields: {', '.join(missing)}")

    # Normalize localized names (GBFS 3.x) to plain strings.
    if "name" in df.columns:
        df["name"] = df["name"].apply(_extract_name)

    # Validate and clean coordinates.
    original_count = len(df)
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    if len(df) < original_count:
        logging.warning(
            "Warning: Dropped %d stations due to invalid coordinates.",
            original_count - len(df),
        )

    if df.empty:
        logging.warning("Warning: No valid station data found after cleaning.")
        return gpd.GeoDataFrame(
            columns=["station_id", "name", "lat", "lon", "geometry"],
            geometry=[],
            crs=GBFS_CRS,
        )

    # Drop nested/complex columns that cannot be written to flat formats
    # (e.g. rental_uris, rental_methods, vehicle_type_capacity).
    keep_cols: list[str] = []
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
            logging.info("Info: Dropping nested column '%s' (unsupported in flat output).", col)
            continue
        keep_cols.append(col)
    df = df[keep_cols]

    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=GBFS_CRS)


def export_geojson(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Exports a GeoDataFrame to a GeoJSON file.

    Args:
        gdf: The GeoDataFrame to export.
        out_path: Full path for the output ``.geojson`` file.

    Raises:
        IOError: If the file cannot be written.
    """
    if gdf.empty:
        logging.info("Info: Skipping GeoJSON export for %s: No data.", out_path.name)
        return
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GeoJSON", index=False)
        logging.info("Successfully exported %d features to: %s", len(gdf), out_path)
    except Exception as exc:
        raise IOError(f"Could not write GeoJSON {out_path}: {exc}") from exc


def export_shapefile(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Exports a GeoDataFrame to an ESRI Shapefile.

    Args:
        gdf: The GeoDataFrame to export.
        out_path: Full path for the output ``.shp`` file.

    Raises:
        IOError: If the file cannot be written.
    """
    if gdf.empty:
        logging.info("Info: Skipping Shapefile export for %s: No data.", out_path.name)
        return
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="ESRI Shapefile", index=False)
        logging.info("Successfully exported %d features to: %s", len(gdf), out_path)
    except Exception as exc:
        raise IOError(f"Could not write shapefile {out_path}: {exc}") from exc


# --- Main Orchestration Function (Core Logic) ---


def gbfs_stations_to_files(
    gbfs_source: Optional[str] = None,
    output_dir: Optional[Path] = None,
    kind: ExportKind = "both",
) -> None:
    """Exports GBFS static stations to Shapefile and/or GeoJSON.

    Uses default paths from the CONFIGURATION section if arguments are not
    provided.

    Args:
        gbfs_source: GBFS auto-discovery URL, ``station_information.json`` URL,
            or local JSON file path. If None, uses ``DEFAULT_GBFS_SOURCE``.
        output_dir: Output directory. If None, uses ``DEFAULT_OUTPUT_DIR``.
        kind: Output format(s): "shapefile", "geojson", or "both".

    Raises:
        ValueError: If required arguments are None and no defaults are set,
            or if the feed contains no usable station data.
        IOError: If the feed cannot be retrieved or outputs cannot be written.
    """
    resolved_source = gbfs_source if gbfs_source is not None else DEFAULT_GBFS_SOURCE
    resolved_output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR

    if resolved_source is None:
        raise ValueError("GBFS source is not specified and no default is set.")
    if resolved_output_dir is None:
        raise ValueError("Output directory is not specified and no default is set.")

    logging.info("-" * 50)
    logging.info("Starting GBFS station export...")
    logging.info("GBFS Source: %s", resolved_source)
    logging.info("Output Directory: %s", resolved_output_dir)
    logging.info("Export Type: %s", kind)
    logging.info("-" * 50)

    station_info_ref = resolve_station_information_url(resolved_source)
    station_info = fetch_json(station_info_ref)
    stations_gdf = build_stations_gdf(station_info)

    if stations_gdf.empty:
        logging.warning("No stations to export; nothing written.")
        return

    if kind in ("shapefile", "both"):
        export_shapefile(stations_gdf, resolved_output_dir / "gbfs_stations.shp")
    if kind in ("geojson", "both"):
        export_geojson(stations_gdf, resolved_output_dir / "gbfs_stations.geojson")

    logging.info("-" * 50)
    logging.info("Export finished.")
    logging.info("-" * 50)


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    """Run GBFS station export using the configured default paths."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if DEFAULT_GBFS_SOURCE == r"https://example.com/gbfs/gbfs.json" or DEFAULT_OUTPUT_DIR == Path(
        r"/path/to/your/default_output_folder"
    ):
        logging.warning(
            "DEFAULT_GBFS_SOURCE and/or DEFAULT_OUTPUT_DIR are still set to their default "
            "placeholder values. Please update them in the CONFIGURATION section before running."
        )
        return
    gbfs_stations_to_files()
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
