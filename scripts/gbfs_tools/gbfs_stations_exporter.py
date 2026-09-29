"""Exports GBFS static station information to ESRI Shapefile and GeoJSON.

Reads a GBFS feed's ``station_information.json`` (the static docked-station
inventory) and writes the stations as WGS 84 point features. The source may
be a GBFS auto-discovery URL (``gbfs.json``), a direct
``station_information.json`` URL, or a path to a local JSON file already
downloaded from a feed.

Both GBFS 2.x (where ``name`` and ``short_name`` are plain strings) and GBFS
3.x (where they are arrays of localized ``{"text", "language"}`` objects) are
supported. ``short_name`` is kept because it carries the station number used
in Capital Bikeshare trip histories (see ``gbfs_ridership_join``).

Inputs:
    - GBFS source: ``gbfs.json`` URL, ``station_information.json`` URL, or a
      local JSON file path
    - Optional export formats: "shapefile", "geojson", or "both"

Outputs:
    - `gbfs_stations.shp`: Shapefile of static station points
    - `gbfs_stations.geojson`: GeoJSON of static station points

Typical usage:
    Update the paths in the CONFIGURATION section (or pass the matching CLI
    flags, e.g. ``--source``, ``--output-dir``, ``--format``) and run from a
    shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence
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


def _is_station_document(doc: Any) -> bool:
    """Returns True if ``doc`` is a ``station_information`` document.

    Recognition is by content (a ``data.stations`` list), not by file name,
    so a station file saved as e.g. ``stations.json`` is still accepted.
    """
    if not isinstance(doc, dict):
        return False
    data = doc.get("data")
    return isinstance(data, dict) and isinstance(data.get("stations"), list)


def _discovery_station_feed(source: str, doc: dict[str, Any]) -> Optional[str]:
    """Returns the ``station_information`` URL listed in a discovery document.

    Args:
        source: Where ``doc`` came from, used in error messages.
        doc: A parsed JSON document.

    Returns:
        The feed URL, or None if ``doc`` is not a discovery document (it has no
        feed catalog under ``data``).

    Raises:
        ValueError: If ``doc`` is a discovery document without a
            ``station_information`` feed.
    """
    data = doc.get("data")
    if not isinstance(data, dict):
        return None

    # GBFS 2.x nests feeds per language: data[<lang>]["feeds"].
    # GBFS 3.x flattens them: data["feeds"].
    feeds: Optional[list[dict[str, Any]]] = None
    if isinstance(data.get("feeds"), list):
        feeds = data["feeds"]
    else:
        for lang_block in data.values():
            if isinstance(lang_block, dict) and isinstance(lang_block.get("feeds"), list):
                feeds = lang_block["feeds"]
                break
    if feeds is None:
        return None

    for feed in feeds:
        if feed.get("name") == "station_information" and feed.get("url"):
            logging.info("Resolved station_information feed: %s", feed["url"])
            return str(feed["url"])

    raise ValueError(f"No 'station_information' feed found in GBFS discovery document '{source}'.")


def resolve_station_information_url(source: str) -> str:
    """Resolves a GBFS source to a ``station_information`` URL or file path.

    The source is fetched and inspected. A document with a ``data.stations``
    list is already station information and is returned unchanged, whatever
    its name. A GBFS auto-discovery document (``gbfs.json``) is searched for
    its ``station_information`` feed. Anything else is returned unchanged.

    Args:
        source: A GBFS auto-discovery URL/path, or a direct station
            information URL/path.

    Returns:
        A URL or file path to the ``station_information`` feed.

    Raises:
        IOError: If the source cannot be retrieved.
        ValueError: If a discovery document is supplied but contains no
            ``station_information`` feed.
    """
    doc = fetch_json(source)
    if _is_station_document(doc):
        return source
    return _discovery_station_feed(source, doc) or source


def load_station_information(source: str) -> dict[str, Any]:
    """Loads the ``station_information`` document for a GBFS source.

    Like :func:`resolve_station_information_url`, but returns the parsed
    document and fetches a direct station file only once.

    Args:
        source: A GBFS auto-discovery URL/path, or a direct station
            information URL/path.

    Returns:
        The parsed ``station_information`` document (or the source document
        itself if it is neither, for :func:`build_stations_gdf` to reject).

    Raises:
        IOError: If a document cannot be retrieved.
        ValueError: If a discovery document has no ``station_information``
            feed, or a document is not valid JSON.
    """
    doc = fetch_json(source)
    if _is_station_document(doc):
        return doc
    feed_url = _discovery_station_feed(source, doc)
    return doc if feed_url is None else fetch_json(feed_url)


#: Station fields that GBFS 3.x localizes as ``[{"text", "language"}]`` arrays.
LOCALIZED_TEXT_FIELDS: tuple[str, ...] = ("name", "short_name")


def _extract_text(raw_value: Any) -> Optional[str]:
    """Normalizes a GBFS text field (e.g. ``name``, ``short_name``) to a string.

    Handles GBFS 2.x plain values and GBFS 3.x localized arrays of
    ``{"text", "language"}`` objects. Integer values (some feeds publish
    ``short_name`` as a JSON number) are converted to their string form so the
    column stays textual.

    Args:
        raw_value: The field value from a station record.

    Returns:
        A string, or None if no usable value is present.
    """
    if isinstance(raw_value, str):
        return raw_value

    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        return str(raw_value)

    if isinstance(raw_value, list) and raw_value:
        preferred = [
            entry.get("text")
            for entry in raw_value
            if isinstance(entry, dict) and entry.get("language") == PREFERRED_LANGUAGE
        ]
        if preferred and preferred[0]:
            return str(preferred[0])
        # Fall back to the first entry that carries text.
        for entry in raw_value:
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

    # Normalize localized text (GBFS 3.x) to plain strings before building the
    # frame, so numeric short_names cannot turn into floats alongside gaps.
    stations = [
        {
            key: _extract_text(value) if key in LOCALIZED_TEXT_FIELDS else value
            for key, value in station.items()
        }
        if isinstance(station, dict)
        else station
        for station in stations
    ]
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

    # Validate and clean coordinates.
    original_count = len(df)
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    in_range = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    df = df[in_range]
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

    station_info = load_station_information(resolved_source)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block."""
    parser = argparse.ArgumentParser(
        description=(
            "Export GBFS static station information to Shapefile and/or GeoJSON. "
            "Defaults come from the CONFIGURATION section at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_GBFS_SOURCE,
        help="GBFS gbfs.json URL, station_information.json URL, or local JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--format",
        dest="kind",
        choices=("shapefile", "geojson", "both"),
        default="both",
        help="Output format(s) to write.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(notebook_safe_argv(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the CONFIGURATION block.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.source == r"https://example.com/gbfs/gbfs.json" or args.output_dir == Path(
        r"/path/to/your/default_output_folder"
    ):
        logging.warning(
            "DEFAULT_GBFS_SOURCE and/or DEFAULT_OUTPUT_DIR are still set to their default "
            "placeholder values. Update them in the CONFIGURATION section or pass "
            "--source/--output-dir before running."
        )
        return 2
    try:
        gbfs_stations_to_files(gbfs_source=args.source, output_dir=args.output_dir, kind=args.kind)
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the CONFIGURATION block stays in charge.
    raise SystemExit(main())
