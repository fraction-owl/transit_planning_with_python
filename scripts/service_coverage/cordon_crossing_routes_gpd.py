"""List the GTFS routes that cross each cordon or screenline, and flag near misses.

Corridor studies often start by drawing cordons around an area (a polygon
around downtown) or screenlines across parallel streets (a line along a river),
then asking which routes cross each one so that ridership and service can be
summed by corridor. This script overlays every GTFS shape on every cordon,
counts how many times each shape crosses it, and rolls the result up to routes,
with the share of each route's trips whose shape crosses.

Lines drawn by hand rarely meet routes cleanly, so the script also flags the
cases to check before trusting the list: near misses (routes that come within
``NEAR_MISS_FT`` of a cordon without crossing it) and marginal crossings
(routes that cross it by less than that distance, e.g. a route that turns back
or ends just past the line, or crosses near the end of a screenline). In both
cases, moving either line by less than ``NEAR_MISS_FT`` would change the answer.

Inputs
------
- A GTFS feed folder or ``.zip`` with routes.txt, trips.txt and shapes.txt.
  Trips without a usable shape are traced stop to stop (stop_times, stops).
- A cordon layer (``.shp``, zipped shapefile, ``.gpkg`` or ``.geojson``). Each
  line feature is a screenline; each polygon is a cordon whose outline is the
  line crossed. Names come from ``CORDON_NAME_FIELD``.

Outputs
-------
- ``cordon_summary.csv``: one row per cordon listing the routes that cross it,
  the marginal crossings among them, and the near misses.
- ``cordon_route_crossings.csv``: one row per cordon and route within
  ``NEAR_MISS_FT``: status, crossings per trip, share of trips crossing, margin
  or closest approach in feet, and a note on each flagged row.
- ``cordon_shape_crossings.csv``: the same per GTFS shape, with the crossing
  locations (lat/lon) for checking in GIS.
- ``cordon_maps/cordon_<name>.png`` (optional): each cordon with the nearby
  routes colored by result.
- ``cordon_crossing_routes_runlog.txt``: run-log sidecar with the verbatim
  CONFIGURATION block.

Limitations
-----------
- Results are only as good as the GTFS shapes and the cordon drawing; the
  review flags exist for this reason.
- Trip counts take each trips.txt row once (frequencies.txt is not expanded)
  and combine every service day unless ``SERVICE_IDS_TO_INCLUDE`` is set.
- Crossing direction (inbound or outbound) is not reported.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags)
and run from a shell or a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import textwrap
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, List, NamedTuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, substring

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# === BEGIN CONFIG ===

# GTFS feed: a folder of .txt files or the published .zip.
GTFS_PATH: Path = Path(r"Path\To\Your\GTFS_Folder")  # ←–– change me

# Cordon / screenline layer: a .shp, a zipped shapefile, a .gpkg or a .geojson.
# Each line feature is a screenline; each polygon is a closed cordon whose
# outline is the line crossed. Multi-part features count as one cordon.
CORDONS_PATH: Path = Path(r"Path\To\Your\Cordons.shp")  # ←–– change me

OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")  # ←–– change me

# When CORDONS_PATH is a .zip holding several shapefiles, the one to use,
# e.g. "screenlines.shp". None when the zip holds a single shapefile.
CORDONS_ZIP_MEMBER: Optional[str] = None

# Attribute holding each cordon's name. None = the first of name / cordon_name /
# cordon / screenline / label found (any capitalization); features without a
# name are called cordon_1, cordon_2, … in file order.
CORDON_NAME_FIELD: Optional[str] = None

# Review threshold, in feet. A route that comes within this distance of a
# cordon without crossing it is a near miss; a route that crosses it by less
# than this distance is a marginal crossing. 0 turns both flags off.
NEAR_MISS_FT: float = 500.0

# Optional service_id filter (values from trips.txt). Leave empty to use every
# trip in the feed; trip counts then combine all service days.
SERVICE_IDS_TO_INCLUDE: List[str] = []

# Projected CRS used for distances (feet or metre units both work).
# Default: Maryland State Plane, US feet (DC region). None = the feed's UTM zone.
PROJECTED_CRS: Optional[str] = "EPSG:2248"

# Write one PNG map per cordon, and their resolution.
WRITE_MAPS: bool = True
FIGURE_DPI: int = 150

# Output names (written inside OUTPUT_DIR).
MAPS_SUBDIR: str = r"cordon_maps"
SUMMARY_FILENAME: str = r"cordon_summary.csv"
ROUTES_FILENAME: str = r"cordon_route_crossings.csv"
SHAPES_FILENAME: str = r"cordon_shape_crossings.csv"
RUN_LOG_FILENAME: str = r"cordon_crossing_routes_runlog.txt"

# When True, a failed run-log write aborts the script so no output is left
# without its configuration record. Set False only for read-only destinations.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# === END CONFIG ===

NAME_FIELD_CANDIDATES: tuple[str, ...] = ("name", "cordon_name", "cordon", "screenline", "label")

# Result labels, best first. A route takes the best label among its shapes.
STATUS_CROSSES: str = "crosses"
STATUS_MARGINAL: str = "crosses_marginal"
STATUS_NEAR_MISS: str = "near_miss"
STATUS_INSIDE: str = "inside"
STATUS_NONE: str = "none"
STATUS_RANK: dict[str, int] = {
    STATUS_CROSSES: 4,
    STATUS_MARGINAL: 3,
    STATUS_NEAR_MISS: 2,
    STATUS_INSIDE: 1,
    STATUS_NONE: 0,
}
CROSSING_STATUSES: frozenset[str] = frozenset({STATUS_CROSSES, STATUS_MARGINAL})

# Shape-table columns used to draw the maps; dropped before the table is written.
INTERNAL_COLUMNS: tuple[str, ...] = ("margin_reason_code", "crossing_xy", "nearest_xy")

MARGIN_REASON_TEXT: dict[str, str] = {
    "depth": "the route gets no more than {ft} ft past the cordon before crossing back or ending",
    "cordon_end": "only {ft} ft from the end of the cordon line",
}

SOURCE_SHAPES: str = "shapes.txt"
SOURCE_STOPS: str = "stop-to-stop"

# Drawing style. Status colors are validated together for color-vision
# deficiency; line style and labels carry the same information.
SURFACE_COLOR: str = "#fcfcfb"
INK_COLOR: str = "#0b0b0b"
SECONDARY_INK: str = "#52514e"
MUTED_INK: str = "#898781"
CONTEXT_COLOR: str = "#c3c2b7"
CORDON_FILL: str = "#f0efec"
HAIRLINE_COLOR: str = "#e1e0d9"
STATUS_COLORS: dict[str, str] = {
    STATUS_CROSSES: "#2a78d6",
    STATUS_MARGINAL: "#ec835a",
    STATUS_NEAR_MISS: "#d03b3b",
}
STATUS_DASHES: dict[str, Any] = {
    STATUS_CROSSES: "solid",
    STATUS_MARGINAL: (0, (5, 2)),
    STATUS_NEAR_MISS: (0, (1.5, 1.5)),
}
MAP_WIDTH_IN: float = 8.0

FEET_PER_MILE: float = 5280.0

# =============================================================================
# HELPERS (canonical copies — see utils/)
# =============================================================================


class ServiceSelectionError(ValueError):
    """A requested GTFS service_id is not used by any trip in the feed.

    A configuration problem rather than a crash: callers report it without a
    traceback and exit with code 2.
    """


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


def load_shapefile(path: str, member: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load a shapefile given a direct ``.shp`` path or a ``.zip`` archive.

    Args:
        path: Path to a ``.shp`` file, or to a ``.zip`` archive containing
            one. A zipped shapefile is opened directly via GDAL's ``zip://``
            virtual filesystem, so the archive is never extracted to disk.
        member: For a ``.zip`` containing more than one ``.shp``, the name
            of the one to load (matched case-insensitively against its
            base filename, e.g. ``"Hospitals.shp"``). Ignored when *path*
            is a direct ``.shp`` file. Required when the archive contains
            more than one shapefile. macOS metadata entries (anything under
            ``__MACOSX/`` and ``._*`` files) are not counted as shapefiles.

    Returns:
        The loaded :class:`geopandas.GeoDataFrame`.

    Raises:
        OSError: *path* does not exist.
        ValueError: *path* is neither a ``.shp`` nor a ``.zip`` file; the
            zip is unreadable/corrupt; the zip contains no shapefile; the
            zip contains more than one shapefile and *member* was not
            supplied; or *member* does not match any shapefile in the zip.
    """
    if not os.path.exists(path):
        raise OSError(f"The path '{path}' does not exist.")

    lower_path = path.lower()
    if lower_path.endswith(".shp"):
        logging.info("Reading %s", path)
        return gpd.read_file(path)

    if not lower_path.endswith(".zip"):
        raise ValueError(f"'{path}' is neither a .shp nor a .zip file.")

    try:
        with zipfile.ZipFile(path) as archive:
            # Zips made with macOS Finder add __MACOSX/._<name>.shp metadata
            # files next to the real ones; they are not shapefiles.
            shp_members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".shp")
                and not name.startswith("__MACOSX/")
                and not os.path.basename(name).startswith("._")
            ]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc

    if not shp_members:
        raise ValueError(f"No .shp file found inside '{path}'.")

    if member is not None:
        matches = [name for name in shp_members if os.path.basename(name).lower() == member.lower()]
        if not matches:
            raise ValueError(f"'{member}' not found inside '{path}'. Available: {shp_members}")
        chosen = matches[0]
    elif len(shp_members) == 1:
        chosen = shp_members[0]
    else:
        raise ValueError(
            f"'{path}' contains {len(shp_members)} shapefiles; pass `member=` to "
            f"choose one of: {shp_members}"
        )

    source = f"zip://{path}!{chosen}"
    logging.info("Reading %s", source)
    return gpd.read_file(source)


def stop_pattern_id(stop_ids: Iterable[object]) -> str:
    """Return a short text ID for an ordered stop pattern that is stable across runs.

    Python's built-in ``hash()`` is salted per process, so it gives the same
    pattern a different value on every run. This SHA-1 digest gives the same
    ID for the same stop sequence on every run, in every feed, and in every
    script that copies this helper, so outputs from different runs can be
    matched up by pattern.

    Args:
        stop_ids: Ordered ``stop_id`` values for one trip pattern. Order
            matters: the reversed sequence gets a different ID. Values are
            coerced to ``str``, so ``1`` and ``"1"`` give the same ID.

    Returns:
        The first 8 lowercase hex characters of the digest, e.g.
        ``"3f9a1c2e"``. Write it to Excel as text, not as a number.
    """
    # JSON-encode the list so a stop_id containing a separator can't collide
    # with a different split of the same characters (["a,b"] vs ["a", "b"]).
    payload = json.dumps([str(stop_id) for stop_id in stop_ids])
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def select_trips_by_service(
    trips_df: pd.DataFrame,
    service_ids: Sequence[str],
    *,
    default_service_ids: Sequence[str] = (),
) -> pd.DataFrame:
    """Return the trips that run on *service_ids*; every trip when it is empty.

    Warns when *service_ids* still equals *default_service_ids* (the calling
    script's shipped default), so an unedited default is noticed, and warns that an
    empty selection combines every service day.

    Args:
        trips_df: Parsed ``trips.txt`` (needs a ``service_id`` column).
        service_ids: The service_id values to keep. Empty keeps every trip.
        default_service_ids: The calling script's default, used only for the warning.

    Returns:
        The trips whose ``service_id`` is in *service_ids*.

    Raises:
        ServiceSelectionError: If a requested service_id is used by no trip. The
            message lists the feed's service_ids with their trip counts.
    """
    wanted = [str(s).strip() for s in service_ids if str(s).strip()]
    if not wanted:
        logging.warning(
            "No service_id selected: using every trip in the feed, all service days "
            "combined. Set SERVICE_IDS_TO_INCLUDE to analyze a single service day."
        )
        return trips_df
    defaults = [str(s).strip() for s in default_service_ids]
    if defaults and wanted == defaults:
        logging.warning(
            "SERVICE_IDS_TO_INCLUDE is still the default %s. Check it against calendar.txt "
            "and change it if this feed uses other service_ids or you want another day.",
            wanted,
        )
    trip_service = trips_df["service_id"].astype(str).str.strip()
    counts = trip_service.value_counts().sort_index()
    missing = [s for s in wanted if s not in counts.index]
    if missing:
        available = ", ".join(f"{sid} ({n} trips)" for sid, n in counts.items())
        raise ServiceSelectionError(
            f"service_id(s) {missing} are not used by any trip in trips.txt. This feed's "
            f"service_ids: {available}. Set SERVICE_IDS_TO_INCLUDE to the service day to "
            "analyze (calendar.txt shows which days each service_id runs)."
        )
    kept = trips_df[trip_service.isin(wanted)]
    logging.info("Service filter %s: trips %d -> %d.", wanted, len(trips_df), len(kept))
    return kept


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
# FUNCTIONS — inputs
# =============================================================================


def route_label(route_short_name: object, route_id: object) -> str:
    """Return the display label for a route: its short name, else its route_id."""
    short = "" if pd.isna(route_short_name) else str(route_short_name).strip()
    return short if short else str(route_id).strip()


def natural_sort_key(label: str) -> str:
    """Return a sort key that orders embedded numbers numerically ("2" < "10")."""
    return re.sub(r"\d+", lambda m: m.group().zfill(8), label)


def sanitize_filename_component(value: str, max_len: int = 40) -> str:
    """Make a string safe to use as (part of) a file name.

    Collapses every character outside ``[A-Za-z0-9_]`` to an underscore,
    trims leading/trailing underscores, and truncates to *max_len*
    characters.

    Args:
        value: Raw string (e.g. a cordon name).
        max_len: Maximum length of the returned string.

    Returns:
        A non-empty, filesystem-safe string ("unnamed" if nothing survives).
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return cleaned[:max_len].rstrip("_") or "unnamed"


def resolve_projected_crs(crs: Optional[str], sample: gpd.GeoSeries) -> CRS:
    """Return the projected CRS for distances.

    Args:
        crs: A projected CRS string, or ``None`` to use the UTM zone of *sample*.
        sample: WGS 84 geometries of the feed, used only when *crs* is ``None``.

    Raises:
        ValueError: If the CRS is geographic (degrees cannot measure feet).
    """
    target = CRS.from_user_input(crs) if crs else CRS.from_user_input(sample.estimate_utm_crs())
    if not target.is_projected:
        raise ValueError(f"PROJECTED_CRS {crs!r} is not a projected CRS; distances need one.")
    return target


def crs_unit_to_feet(crs: CRS) -> float:
    """Return how many feet one unit of *crs*'s x axis spans."""
    return crs.axis_info[0].unit_conversion_factor / 0.3048


def resolve_cordon_names(cordons: gpd.GeoDataFrame, name_field: Optional[str]) -> list[str]:
    """Return one unique, non-empty display name per cordon feature.

    Args:
        cordons: The cordon layer.
        name_field: Attribute to read names from, or ``None`` to use the first
            of ``NAME_FIELD_CANDIDATES`` present (case-insensitive).

    Returns:
        Names in feature order. Missing names become ``cordon_<n>``; repeats
        get ``_2``, ``_3`` … appended.

    Raises:
        ValueError: If *name_field* is given but is not in the layer.
    """
    attributes = [c for c in cordons.columns if c != cordons.geometry.name]
    field: Optional[str] = None
    if name_field:
        if name_field not in attributes:
            raise ValueError(
                f"CORDON_NAME_FIELD {name_field!r} is not in the cordon layer. "
                f"Its fields are: {', '.join(map(str, attributes)) or '(none)'}."
            )
        field = name_field
    else:
        by_lower = {str(c).lower(): str(c) for c in attributes}
        field = next((by_lower[c] for c in NAME_FIELD_CANDIDATES if c in by_lower), None)
        if field is None:
            logging.info("No name field found in the cordon layer; numbering cordons instead.")

    values: list[object] = list(cordons[field]) if field else [None] * len(cordons)
    names: list[str] = []
    used: set[str] = set()
    for number, value in enumerate(values, start=1):
        text = "" if value is None or pd.isna(value) else str(value).strip()
        base = text or f"cordon_{number}"
        candidate, suffix = base, 2
        while candidate.lower() in used:
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        used.add(candidate.lower())
        names.append(candidate)
    return names


def repair_polygon(geom: BaseGeometry, name: str) -> BaseGeometry:
    """Return *geom* made valid, keeping only its polygon parts.

    Hand-drawn cordons sometimes have an outline that crosses itself, which
    makes inside/outside tests unreliable. The repaired polygon covers the
    same area; a warning names the cordon so the drawing can be fixed.
    """
    if geom.is_valid:
        return geom
    fixed = shapely.make_valid(geom)
    polygons = [p for p in shapely.get_parts(fixed) if p.geom_type in ("Polygon", "MultiPolygon")]
    logging.warning(
        "Cordon %r is an invalid polygon (e.g. its outline crosses itself); repaired it. "
        "Check the drawing in GIS.",
        name,
    )
    return shapely.union_all(polygons) if polygons else geom


def load_cordons(
    path: Path,
    crs: CRS,
    name_field: Optional[str] = None,
    zip_member: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Read the cordon layer and project it to *crs*.

    Args:
        path: A .shp, a zipped shapefile, a .gpkg or a .geojson.
        crs: The projected CRS distances are measured in.
        name_field: See :func:`resolve_cordon_names`.
        zip_member: Shapefile to use when *path* is a zip holding several.

    Returns:
        GeoDataFrame with columns ``cordon`` (name), ``cordon_type``
        (``"line"`` or ``"polygon"``) and ``geometry`` (projected).

    Raises:
        OSError: If *path* does not exist.
        ValueError: If the layer cannot be read, has no CRS, or has no line or
            polygon features.
    """
    text_path = str(path)
    if not os.path.exists(text_path):
        raise OSError(f"The cordon layer '{text_path}' does not exist.")
    if text_path.lower().endswith((".shp", ".zip")):
        layer = load_shapefile(text_path, member=zip_member or None)
    else:
        try:
            layer = gpd.read_file(text_path)
        except Exception as exc:  # noqa: BLE001 - GDAL raises several unrelated types
            raise ValueError(f"Could not read the cordon layer '{text_path}': {exc}") from exc

    if layer.crs is None:
        raise ValueError(
            f"The cordon layer '{text_path}' has no coordinate reference system (a shapefile "
            "needs its .prj file). Define the projection in GIS and export it again."
        )
    layer = layer.reset_index(drop=True)
    layer["cordon"] = resolve_cordon_names(layer, name_field)

    kinds = {
        "LineString": "line",
        "MultiLineString": "line",
        "LinearRing": "line",
        "Polygon": "polygon",
        "MultiPolygon": "polygon",
    }
    geometry = layer.geometry
    usable = geometry.notna() & ~geometry.is_empty & geometry.geom_type.isin(list(kinds))
    for name, geom_type in zip(layer.loc[~usable, "cordon"], geometry[~usable].geom_type):
        logging.warning(
            "Cordon %r skipped: %s geometry is not a line or polygon.", name, geom_type or "empty"
        )
    layer = layer[usable]
    if layer.empty:
        raise ValueError(f"The cordon layer '{text_path}' has no line or polygon features.")

    projected = layer.to_crs(crs)
    cordon_types = projected.geometry.geom_type.map(kinds).to_numpy()
    geoms = [
        repair_polygon(geom, name) if kind == "polygon" else geom
        for geom, kind, name in zip(projected.geometry, cordon_types, projected["cordon"])
    ]
    out = gpd.GeoDataFrame(
        {"cordon": projected["cordon"].to_numpy(), "cordon_type": cordon_types},
        geometry=geoms,
        crs=crs,
    )
    logging.info(
        "Loaded %d cordon(s): %d line, %d polygon.",
        len(out),
        int((out["cordon_type"] == "line").sum()),
        int((out["cordon_type"] == "polygon").sum()),
    )
    return out


def build_shape_lines(shapes: Optional[pd.DataFrame]) -> dict[str, LineString]:
    """Build one WGS 84 LineString per shape_id from shapes.txt.

    Args:
        shapes: Parsed shapes.txt, or ``None`` when the feed has none.

    Returns:
        Mapping of shape_id to LineString (lon/lat). Shapes with fewer than
        two distinct valid points are skipped with a warning. Empty when
        *shapes* is missing or lacks the required columns.
    """
    if shapes is None or shapes.empty:
        return {}
    required = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    missing = [c for c in required if c not in shapes.columns]
    if missing:
        logging.warning("shapes.txt lacks %s; tracing routes stop to stop.", missing)
        return {}

    points = shapes[required].copy()
    points["shape_id"] = points["shape_id"].astype(str).str.strip()
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        points[col] = pd.to_numeric(points[col], errors="coerce")
    numeric = ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    n_bad = int(points[numeric].isna().any(axis=1).sum())
    if n_bad:
        logging.warning("Dropped %d shapes.txt row(s) with unparseable values.", n_bad)
    points = points.dropna().sort_values(["shape_id", "shape_pt_sequence"], kind="mergesort")

    lines: dict[str, LineString] = {}
    for shape_id, grp in points.groupby("shape_id", sort=False):
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        if len(set(coords)) < 2:
            logging.warning("Shape %s has fewer than two distinct points; skipped.", shape_id)
            continue
        lines[str(shape_id)] = LineString(coords)
    return lines


def build_stop_pattern_lines(
    trip_ids: Iterable[str], stop_times: pd.DataFrame, stops: pd.DataFrame
) -> tuple[dict[str, LineString], dict[str, str]]:
    """Trace stop-to-stop WGS 84 lines for trips that have no usable shape.

    Args:
        trip_ids: Trips to trace.
        stop_times: Parsed stop_times.txt (string columns).
        stops: Parsed stops.txt (string columns).

    Returns:
        ``(lines, trip_to_path)``: one LineString per distinct stop sequence,
        keyed ``stops_<pattern id>``, and each traceable trip's key. Trips
        whose stops give fewer than two distinct locations are left out.
    """
    wanted = set(trip_ids)
    times = stop_times.loc[
        stop_times["trip_id"].astype(str).str.strip().isin(wanted),
        ["trip_id", "stop_id", "stop_sequence"],
    ].copy()
    times["trip_id"] = times["trip_id"].astype(str).str.strip()
    times["stop_id"] = times["stop_id"].astype(str).str.strip()
    times["stop_sequence"] = pd.to_numeric(times["stop_sequence"], errors="coerce")
    times = times.dropna(subset=["stop_sequence"]).sort_values(
        ["trip_id", "stop_sequence"], kind="mergesort"
    )

    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    valid = lat.notna() & lon.notna()
    stop_xy = dict(
        zip(stops.loc[valid, "stop_id"].astype(str).str.strip(), zip(lon[valid], lat[valid]))
    )

    lines: dict[str, LineString] = {}
    trip_to_path: dict[str, str] = {}
    unusable: set[str] = set()
    for trip_id, seq in times.groupby("trip_id", sort=False)["stop_id"]:
        stop_ids = tuple(seq)
        path_id = f"stops_{stop_pattern_id(stop_ids)}"
        if path_id not in lines and path_id not in unusable:
            coords = [stop_xy[s] for s in stop_ids if s in stop_xy]
            deduped = [c for i, c in enumerate(coords) if i == 0 or c != coords[i - 1]]
            if len(set(deduped)) >= 2:
                lines[path_id] = LineString(deduped)
            else:
                unusable.add(path_id)
        if path_id in lines:
            trip_to_path[str(trip_id)] = path_id
    return lines, trip_to_path


# =============================================================================
# FUNCTIONS — crossing geometry
# =============================================================================


class PathResult(NamedTuple):
    """How one route path (a GTFS shape) relates to one cordon.

    Distances are in the projected CRS's units.
    """

    status: str
    n_crossings: int
    margin: float  # how far a line must move to remove every crossing; NaN if none
    margin_reason: str  # "depth" or "cordon_end" for crossings, else ""
    min_distance: float
    overlap: float  # length the path runs exactly along the cordon line
    crossing_xy: tuple[tuple[float, float], ...]
    nearest_xy: Optional[tuple[tuple[float, float], tuple[float, float]]]  # (path, cordon)
    inside: bool  # polygon cordons only: the path never leaves the cordon area


def cordon_line_of(geom: BaseGeometry) -> BaseGeometry:
    """Return the line a route must cross: a polygon's outline, or the line itself."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom.boundary
    return geom


def find_contact_events(
    path: LineString, cordon_line: BaseGeometry, tol: float
) -> tuple[list[tuple[float, float]], float]:
    """Return where *path* meets *cordon_line*, as spans of distance along *path*.

    Works segment by segment, so a path that passes the same spot twice (an
    out-and-back shape) reports both passes.

    Args:
        path: The route path (projected).
        cordon_line: The cordon line (projected).
        tol: Contacts closer together than this along the path are merged.

    Returns:
        ``(events, overlap)``: sorted ``(start, end)`` spans (``start == end``
        for a point contact; longer where the path runs along the cordon), and
        the total length the path runs along the cordon.
    """
    coords = shapely.get_coordinates(path)
    keep = np.r_[True, np.any(np.diff(coords, axis=0) != 0, axis=1)]
    coords = coords[keep]
    if len(coords) < 2:
        return [], 0.0
    starts, ends = coords[:-1], coords[1:]
    cumulative = np.r_[0.0, np.cumsum(np.hypot(*(ends - starts).T))]
    segments = shapely.linestrings(np.stack([starts, ends], axis=1))
    hits = np.flatnonzero(shapely.intersects(segments, cordon_line))

    raw: list[tuple[float, float]] = []
    overlap = 0.0
    for i in hits:
        contact = shapely.intersection(segments[i], cordon_line)
        for part in shapely.get_parts(contact):
            if part.is_empty:
                continue
            part_xy = shapely.get_coordinates(part)
            along = cumulative[i] + np.hypot(*(part_xy - starts[i]).T)
            raw.append((float(along.min()), float(along.max())))
            if part.geom_type == "LineString":
                overlap += part.length

    raw.sort()
    merged: list[tuple[float, float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged, overlap


def max_distance_from(line: BaseGeometry, target: BaseGeometry, step: float) -> float:
    """Return the largest distance from any point of *line* to *target*.

    *line* is densified to points no more than *step* apart, so the result
    can fall short of the true maximum by at most about *step*.
    """
    if line.is_empty or line.length == 0:
        return float(line.distance(target)) if not line.is_empty else 0.0
    dense = shapely.segmentize(line, step)
    points = shapely.points(shapely.get_coordinates(dense))
    return float(shapely.distance(points, target).max())


def assess_path(
    path: LineString,
    cordon_geom: BaseGeometry,
    threshold: float,
    *,
    step: float,
    delta: float,
    tol: float,
) -> PathResult:
    """Classify how *path* relates to one cordon.

    A crossing is a contact where the path passes from one side of the cordon
    to the other; a path that only touches the cordon, or ends on it, does not
    cross. Each crossing's margin is how far either line would have to move to
    remove it: the smaller of how far the path gets from the cordon on each
    side before crossing again or ending, and (for open lines) the distance
    from the crossing to the cordon's nearer end. The path's margin is the
    largest margin among its crossings.

    Args:
        path: The route path (projected).
        cordon_geom: The cordon as drawn (line or polygon, projected).
        threshold: The near-miss threshold, in CRS units.
        step: Densification step for depth measurements, in CRS units.
        delta: Distance along the path used to test which side it is on.
        tol: Contacts closer together than this along the path are merged.

    Returns:
        A :class:`PathResult`. Status is ``crosses`` (margin of at least
        *threshold*), ``crosses_marginal``, ``near_miss`` (no crossing, within
        *threshold*), ``inside`` (polygon only; no crossing, inside and farther
        than *threshold* from its edge) or ``none``.
    """
    polygonal = cordon_geom.geom_type in ("Polygon", "MultiPolygon")
    cordon_line = cordon_line_of(cordon_geom)
    length = path.length
    min_distance = float(path.distance(cordon_line))

    events: list[tuple[float, float]] = []
    overlap = 0.0
    if min_distance <= tol:
        events, overlap = find_contact_events(path, cordon_line, tol)

    crossings: list[tuple[float, float]] = []
    for k, (start, end) in enumerate(events):
        if start <= tol or end >= length - tol:
            continue  # the path ends on the cordon rather than passing through it
        gap_before = start - (events[k - 1][1] if k else 0.0)
        gap_after = (events[k + 1][0] if k + 1 < len(events) else length) - end
        d = min(delta, 0.5 * gap_before, 0.5 * gap_after)
        if d <= 0:
            continue
        before = path.interpolate(start - d)
        after = path.interpolate(end + d)
        if polygonal:
            changes_side = bool(cordon_geom.contains(before)) != bool(cordon_geom.contains(after))
        else:
            changes_side = bool(LineString([before, after]).intersects(cordon_line))
        if changes_side:
            crossings.append((start, end))

    if crossings:
        bounds = [0.0, *[v for span in crossings for v in span], length]
        depths = [
            max_distance_from(substring(path, bounds[2 * j], bounds[2 * j + 1]), cordon_line, step)
            if bounds[2 * j + 1] - bounds[2 * j] > tol
            else 0.0
            for j in range(len(crossings) + 1)
        ]
        cordon_ends = cordon_line.boundary
        best_margin, best_reason = -1.0, ""
        crossing_xy: list[tuple[float, float]] = []
        for k, (start, end) in enumerate(crossings):
            point = path.interpolate(0.5 * (start + end))
            crossing_xy.append((point.x, point.y))
            depth_margin = min(depths[k], depths[k + 1])
            end_margin = math.inf if cordon_ends.is_empty else float(point.distance(cordon_ends))
            margin = min(depth_margin, end_margin)
            if margin > best_margin:
                best_margin = margin
                best_reason = "cordon_end" if end_margin < depth_margin else "depth"
        status = STATUS_CROSSES if best_margin >= threshold else STATUS_MARGINAL
        return PathResult(
            status=status,
            n_crossings=len(crossings),
            margin=best_margin,
            margin_reason=best_reason,
            min_distance=0.0,
            overlap=overlap,
            crossing_xy=tuple(crossing_xy),
            nearest_xy=None,
            inside=False,
        )

    inside = polygonal and bool(cordon_geom.contains(path))
    nearest: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    if min_distance <= threshold:
        status = STATUS_NEAR_MISS
        on_path, on_cordon = nearest_points(path, cordon_line)
        nearest = ((on_path.x, on_path.y), (on_cordon.x, on_cordon.y))
    else:
        status = STATUS_INSIDE if inside else STATUS_NONE
    return PathResult(
        status=status,
        n_crossings=0,
        margin=math.nan,
        margin_reason="",
        min_distance=min_distance,
        overlap=overlap,
        crossing_xy=(),
        nearest_xy=nearest,
        inside=inside,
    )


def assess_cordon(
    cordon_geom: BaseGeometry, paths: gpd.GeoSeries, threshold: float, unit_to_ft: float
) -> dict[str, PathResult]:
    """Assess every path against one cordon, skipping paths that are clearly far away.

    Args:
        cordon_geom: The cordon as drawn (projected).
        paths: Route paths (projected), indexed by path id.
        threshold: The near-miss threshold, in CRS units.
        unit_to_ft: Feet per CRS unit.

    Returns:
        Mapping of path id to :class:`PathResult`, for paths whose status is not
        ``none``.
    """
    foot = 1.0 / unit_to_ft
    tol = 0.01 * foot
    step = max(threshold, 100.0 * foot) / 10.0
    delta = 0.5 * foot
    cordon_line = cordon_line_of(cordon_geom)
    shapely.prepare(cordon_line)
    geoms = paths.to_numpy()
    distances = shapely.distance(geoms, cordon_line)
    near = distances <= threshold + tol
    inside = np.zeros(len(geoms), dtype=bool)
    if cordon_geom.geom_type in ("Polygon", "MultiPolygon"):
        inside = shapely.contains(cordon_geom, geoms)

    results: dict[str, PathResult] = {}
    for path_id, geom, is_near, is_inside, distance in zip(
        paths.index, geoms, near, inside, distances
    ):
        if is_near:
            result = assess_path(geom, cordon_geom, threshold, step=step, delta=delta, tol=tol)
        elif is_inside:
            result = PathResult(
                STATUS_INSIDE, 0, math.nan, "", float(distance), 0.0, (), None, True
            )
        else:
            continue
        if result.status != STATUS_NONE:
            results[str(path_id)] = result
    return results


# =============================================================================
# FUNCTIONS — tables
# =============================================================================


def _feet(value: float, unit_to_ft: float) -> Optional[float]:
    """Convert a CRS distance to whole feet; NaN and infinity become ``None``."""
    if value is None or not math.isfinite(value):
        return None
    return float(round(value * unit_to_ft))


def describe_route_result(
    status: str,
    margin_ft: Optional[float],
    margin_reason: str,
    closest_ft: Optional[float],
    overlap_ft: float,
    inside: bool,
) -> str:
    """Return the plain-English note for a flagged cordon/route row ('' otherwise)."""
    if status == STATUS_MARGINAL:
        template = MARGIN_REASON_TEXT.get(margin_reason, MARGIN_REASON_TEXT["depth"])
        reason = template.format(ft=f"{margin_ft or 0:,.0f}")
        return (
            f"Crosses, but {reason}; a small change to either line would remove the crossing. "
            "Check whether this route belongs to the corridor."
        )
    if status == STATUS_NEAR_MISS:
        if overlap_ft > 0:
            where = f"Runs along the cordon for {overlap_ft:,.0f} ft without crossing it"
        elif closest_ft is not None and closest_ft < 1:
            where = "Touches the cordon without crossing it"
        else:
            where = f"Comes within {closest_ft or 0:,.0f} ft of the cordon without crossing it"
        if inside:
            where += " (it stays inside the cordon area)"
        return f"{where}. Check whether it should cross."
    if status == STATUS_INSIDE:
        return "Runs entirely inside the cordon area without crossing it."
    return ""


def build_shape_table(
    cordons: gpd.GeoDataFrame,
    results: Mapping[str, Mapping[str, PathResult]],
    usage: pd.DataFrame,
    path_sources: Mapping[str, str],
    to_lonlat: Transformer,
    unit_to_ft: float,
) -> pd.DataFrame:
    """Build the one-row-per-cordon-route-shape table (SHAPES_FILENAME).

    The ``INTERNAL_COLUMNS`` carry projected coordinates for the maps and are
    dropped before the table is written.
    """
    rows: list[dict[str, Any]] = []
    for cordon_name, cordon_type in zip(cordons["cordon"], cordons["cordon_type"]):
        cordon_results = results.get(cordon_name, {})
        if not cordon_results:
            continue
        hits = usage[usage["path_id"].isin(list(cordon_results))]
        for use in hits.itertuples(index=False):
            result = cordon_results[str(use.path_id)]
            points = [to_lonlat.transform(x, y) for x, y in result.crossing_xy]
            reason = result.margin_reason
            rows.append(
                {
                    "cordon": cordon_name,
                    "cordon_type": cordon_type,
                    "route_id": use.route_id,
                    "route": use.route,
                    "shape_id": use.path_id,
                    "geometry_source": path_sources.get(str(use.path_id), ""),
                    "direction_ids": use.direction_ids,
                    "n_trips": int(use.n_trips),
                    "status": result.status,
                    "crossings": result.n_crossings,
                    "crossing_margin_ft": _feet(result.margin, unit_to_ft),
                    "margin_limited_by": {
                        "depth": "how far the route gets past the cordon",
                        "cordon_end": "distance from the crossing to the cordon's end",
                    }.get(reason, "")
                    if result.status in CROSSING_STATUSES
                    else "",
                    "closest_approach_ft": None
                    if result.status in CROSSING_STATUSES
                    else _feet(result.min_distance, unit_to_ft),
                    "runs_along_cordon_ft": _feet(result.overlap, unit_to_ft) or 0.0,
                    "inside_cordon_area": result.inside,
                    "crossing_points": "; ".join(f"{lat:.6f} {lon:.6f}" for lon, lat in points),
                    "margin_reason_code": reason,
                    "crossing_xy": result.crossing_xy,
                    "nearest_xy": result.nearest_xy,
                }
            )
    columns = [
        "cordon",
        "cordon_type",
        "route_id",
        "route",
        "shape_id",
        "geometry_source",
        "direction_ids",
        "n_trips",
        "status",
        "crossings",
        "crossing_margin_ft",
        "margin_limited_by",
        "closest_approach_ft",
        "runs_along_cordon_ft",
        "inside_cordon_area",
        "crossing_points",
        "margin_reason_code",
        "crossing_xy",
        "nearest_xy",
    ]
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table
    order = table["route"].map(natural_sort_key)
    return (
        table.assign(_order=order)
        .sort_values(["cordon", "_order", "shape_id"], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def build_route_table(
    shape_table: pd.DataFrame, routes_info: pd.DataFrame, route_trip_totals: Mapping[str, int]
) -> pd.DataFrame:
    """Roll the shape table up to one row per cordon and route (ROUTES_FILENAME).

    A route takes the best status among its shapes; trip counts and directions
    count only shapes that cross.
    """
    columns = [
        "cordon",
        "cordon_type",
        "route_id",
        "route",
        "route_long_name",
        "route_type",
        "status",
        "crossings_per_trip",
        "crossing_margin_ft",
        "closest_approach_ft",
        "trips_crossing",
        "trips_total",
        "pct_trips_crossing",
        "directions_crossing",
        "shape_ids_crossing",
        "note",
    ]
    if shape_table.empty:
        return pd.DataFrame(columns=columns)

    info = routes_info.set_index("route_id")
    rows: list[dict[str, Any]] = []
    for (cordon, route_id), grp in shape_table.groupby(["cordon", "route_id"], sort=False):
        ranks = grp["status"].map(STATUS_RANK)
        status = str(grp.loc[ranks.idxmax(), "status"])
        crossing = grp[grp["status"].isin(CROSSING_STATUSES)]
        trips_crossing = int(crossing["n_trips"].sum())
        trips_total = int(route_trip_totals.get(str(route_id), 0))
        margin_ft: Optional[float] = None
        margin_reason = ""
        if not crossing.empty:
            best = crossing["crossing_margin_ft"].astype(float).fillna(-1.0).idxmax()
            margin_ft = float(crossing.loc[best, "crossing_margin_ft"])
            margin_reason = str(crossing.loc[best, "margin_reason_code"])
        closest_ft: Optional[float] = None
        if status not in CROSSING_STATUSES:
            closest_ft = grp["closest_approach_ft"].astype(float).min()
        flagged = grp[grp["status"] == status]
        directions = sorted(
            {d.strip() for v in crossing["direction_ids"] for d in str(v).split(";") if d.strip()}
        )
        rows.append(
            {
                "cordon": cordon,
                "cordon_type": grp["cordon_type"].iloc[0],
                "route_id": route_id,
                "route": grp["route"].iloc[0],
                "route_long_name": info["route_long_name"].get(route_id, ""),
                "route_type": info["route_type"].get(route_id, ""),
                "status": status,
                "crossings_per_trip": int(crossing["crossings"].max()) if len(crossing) else 0,
                "crossing_margin_ft": margin_ft,
                "closest_approach_ft": closest_ft,
                "trips_crossing": trips_crossing,
                "trips_total": trips_total,
                "pct_trips_crossing": round(100.0 * trips_crossing / trips_total, 1)
                if trips_total
                else 0.0,
                "directions_crossing": "; ".join(directions),
                "shape_ids_crossing": "; ".join(crossing["shape_id"].astype(str)),
                "note": describe_route_result(
                    status,
                    margin_ft,
                    margin_reason,
                    closest_ft,
                    float(flagged["runs_along_cordon_ft"].max() or 0.0),
                    bool(flagged["inside_cordon_area"].all()),
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_summary_table(
    cordons: gpd.GeoDataFrame, route_table: pd.DataFrame, map_files: Mapping[str, str]
) -> pd.DataFrame:
    """Build the one-row-per-cordon summary (SUMMARY_FILENAME)."""

    def listed(names: pd.Series) -> str:
        """Join route labels in natural order."""
        return "; ".join(sorted(set(names.astype(str)), key=natural_sort_key))

    rows: list[dict[str, Any]] = []
    for name, cordon_type in zip(cordons["cordon"], cordons["cordon_type"]):
        sub = route_table[route_table["cordon"] == name]
        crossing = sub[sub["status"].isin(CROSSING_STATUSES)]
        marginal = sub[sub["status"] == STATUS_MARGINAL]
        near = sub[sub["status"] == STATUS_NEAR_MISS]
        inside = sub[sub["status"] == STATUS_INSIDE]
        rows.append(
            {
                "cordon": name,
                "cordon_type": cordon_type,
                "n_routes_crossing": crossing["route"].nunique(),
                "routes_crossing": listed(crossing["route"]),
                "n_routes_crossing_marginal": marginal["route"].nunique(),
                "routes_crossing_marginal": listed(marginal["route"]),
                "n_routes_near_miss": near["route"].nunique(),
                "routes_near_miss": listed(near["route"]),
                "n_routes_inside": inside["route"].nunique(),
                "routes_inside": listed(inside["route"]),
                "map_file": map_files.get(name, ""),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# FUNCTIONS — maps
# =============================================================================


def map_extent(
    geom: BaseGeometry, pad_floor: float, min_aspect: float = 0.6, max_aspect: float = 1.3
) -> tuple[float, float, float, float]:
    """Return a padded map extent around *geom* with a height/width ratio in range."""
    xmin, ymin, xmax, ymax = geom.bounds
    pad = max(0.12 * max(xmax - xmin, ymax - ymin), pad_floor)
    xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad
    width, height = xmax - xmin, ymax - ymin
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    if height / width < min_aspect:
        height = width * min_aspect
    elif height / width > max_aspect:
        width = height / max_aspect
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def line_segments(geoms: Iterable[BaseGeometry]) -> list[np.ndarray]:
    """Return the coordinate arrays of every line part, for a LineCollection."""
    parts: list[np.ndarray] = []
    for geom in geoms:
        for part in shapely.get_parts(geom):
            if part.geom_type in ("LineString", "LinearRing") and not part.is_empty:
                parts.append(shapely.get_coordinates(part))
    return parts


def cluster_labels(
    anchors: Sequence[tuple[float, float, str]], radius: float
) -> list[tuple[float, float, str]]:
    """Merge labels whose anchors lie within *radius* of each other.

    Several routes often cross a cordon on the same street; one label listing
    them all stays legible where separate labels would overlap.
    """
    clusters: list[tuple[float, float, list[str]]] = []
    for x, y, text in anchors:
        for cx, cy, texts in clusters:
            if math.hypot(cx - x, cy - y) <= radius:
                if text not in texts:
                    texts.append(text)
                break
        else:
            clusters.append((x, y, [text]))
    return [(x, y, textwrap.fill(", ".join(texts), width=26)) for x, y, texts in clusters]


def add_scale_bar(
    ax: Axes, extent: tuple[float, float, float, float], unit_to_ft: float, halo: list[Any]
) -> None:
    """Draw a scale bar of a round length in the lower-left corner of *ax*."""
    xmin, ymin, xmax, ymax = extent
    target_ft = (xmax - xmin) * unit_to_ft / 5.0
    options_ft = [100.0, 250.0, 500.0, 1000.0, 2000.0]
    options_ft += [m * FEET_PER_MILE for m in (0.5, 1, 2, 5, 10, 20, 50, 100)]
    length_ft = max([o for o in options_ft if o <= target_ft], default=options_ft[0])
    if length_ft >= FEET_PER_MILE / 2:
        label = f"{length_ft / FEET_PER_MILE:g} mi"
    else:
        label = f"{length_ft:,.0f} ft"
    x0 = xmin + 0.03 * (xmax - xmin)
    y0 = ymin + 0.04 * (ymax - ymin)
    x1 = x0 + length_ft / unit_to_ft
    ax.plot(
        [x0, x1],
        [y0, y0],
        color=INK_COLOR,
        lw=2,
        solid_capstyle="butt",
        zorder=9,
        path_effects=halo,
    )
    ax.text(
        0.5 * (x0 + x1),
        y0 + 0.012 * (ymax - ymin),
        label,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK_COLOR,
        path_effects=halo,
        zorder=9,
    )


def plot_cordon_map(
    cordon_name: str,
    cordon_geom: BaseGeometry,
    paths: gpd.GeoSeries,
    shape_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    threshold_ft: float,
    unit_to_ft: float,
    out_path: Path,
    dpi: int,
    any_stop_traced: bool,
) -> None:
    """Write one PNG: the cordon, nearby routes colored by result, and labels.

    Args:
        cordon_name: Title of the map.
        cordon_geom: The cordon as drawn (projected).
        paths: Every route path (projected), indexed by path id; drawn in gray.
        shape_rows: This cordon's rows of the shape table, with ``INTERNAL_COLUMNS``.
        route_rows: This cordon's rows of the route table.
        threshold_ft: The near-miss threshold, in feet.
        unit_to_ft: Feet per CRS unit.
        out_path: PNG to write.
        dpi: PNG resolution.
        any_stop_traced: Whether any path was traced stop to stop.
    """
    polygonal = cordon_geom.geom_type in ("Polygon", "MultiPolygon")
    threshold = threshold_ft / unit_to_ft
    extent = map_extent(cordon_geom, max(3.0 * threshold, 1500.0 / unit_to_ft))
    xmin, ymin, xmax, ymax = extent
    aspect = (ymax - ymin) / (xmax - xmin)
    halo = [withStroke(linewidth=3, foreground=SURFACE_COLOR)]

    present = [s for s in STATUS_COLORS if (shape_rows["status"] == s).any()]
    handles: list[Line2D] = []
    for status in present:
        label = {
            STATUS_CROSSES: "Crosses",
            STATUS_MARGINAL: f"Crosses by less than {threshold_ft:,.0f} ft (review)",
            STATUS_NEAR_MISS: f"Within {threshold_ft:,.0f} ft, no crossing (review)",
        }[status]
        handles.append(
            Line2D(
                [],
                [],
                color=STATUS_COLORS[status],
                lw=2,
                linestyle=STATUS_DASHES[status],
                marker="o" if status in CROSSING_STATUSES else None,
                markersize=5,
                markeredgecolor=SURFACE_COLOR,
                label=label,
            )
        )
    handles.append(
        Line2D([], [], color=INK_COLOR, lw=2.6, label="Cordon outline" if polygonal else "Cordon")
    )
    handles.append(Line2D([], [], color=CONTEXT_COLOR, lw=1, label="Other routes and variants"))

    counts = route_rows["status"].value_counts()
    n_cross = int(counts.get(STATUS_CROSSES, 0) + counts.get(STATUS_MARGINAL, 0))
    subtitle = f"{n_cross} route crosses" if n_cross == 1 else f"{n_cross} routes cross"
    if counts.get(STATUS_MARGINAL, 0):
        subtitle += f" ({int(counts[STATUS_MARGINAL])} by less than {threshold_ft:,.0f} ft)"
    n_near = int(counts.get(STATUS_NEAR_MISS, 0))
    subtitle += f" · {n_near} near miss{'es' if n_near != 1 else ''}"
    n_inside = int(counts.get(STATUS_INSIDE, 0))
    if n_inside:
        subtitle += f" · {n_inside} inside only"
    note = (
        f"Review flags: a route that comes within {threshold_ft:,.0f} ft of the cordon without "
        f"crossing it, or crosses it by less than {threshold_ft:,.0f} ft. Line colors show each "
        "GTFS shape's result; routes are counted once, by their best result."
    )
    if any_stop_traced:
        note += " Routes without shapes.txt geometry are traced stop to stop."

    width_in = MAP_WIDTH_IN + 0.4
    map_h = MAP_WIDTH_IN * aspect
    header_in = 0.75
    legend_cols = len(handles) if len(handles) <= 3 else math.ceil(len(handles) / 2)
    legend_in = 0.24 * math.ceil(len(handles) / legend_cols)
    note = textwrap.fill(note, width=118)
    note_in = 0.15 * (note.count("\n") + 1)
    footer_in = 0.15 + legend_in + note_in + 0.1
    height_in = header_in + map_h + footer_in

    fig = Figure(figsize=(width_in, height_in), facecolor=SURFACE_COLOR)
    ax = fig.add_axes(
        (0.2 / width_in, footer_in / height_in, MAP_WIDTH_IN / width_in, map_h / height_in)
    )
    ax.set_facecolor(SURFACE_COLOR)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(HAIRLINE_COLOR)

    if polygonal:
        for polygon in shapely.get_parts(cordon_geom):
            ax.fill(*polygon.exterior.xy, color=CORDON_FILL, lw=0, zorder=1)
            for ring in polygon.interiors:
                ax.fill(*ring.xy, color=SURFACE_COLOR, lw=0, zorder=1)

    window = box(*extent)
    in_view = paths[paths.intersects(window)]
    ax.add_collection(
        LineCollection(line_segments(in_view), colors=CONTEXT_COLOR, linewidths=0.7, zorder=2)
    )

    z_order = {STATUS_CROSSES: 4, STATUS_MARGINAL: 5, STATUS_NEAR_MISS: 5}
    for status in present:
        ids = shape_rows.loc[shape_rows["status"] == status, "shape_id"].astype(str).unique()
        ax.add_collection(
            LineCollection(
                line_segments(paths.loc[list(ids)]),
                colors=STATUS_COLORS[status],
                linewidths=1.8,
                linestyles=STATUS_DASHES[status],
                capstyle="round",
                zorder=z_order[status],
            )
        )

    ax.add_collection(
        LineCollection(
            line_segments([cordon_line_of(cordon_geom)]),
            colors=INK_COLOR,
            linewidths=2.6,
            zorder=6,
        )
    )

    anchors: list[tuple[float, float, str]] = []
    for row in shape_rows.sort_values("route", key=lambda s: s.map(natural_sort_key)).itertuples():
        status = str(row.status)
        if status in CROSSING_STATUSES:
            xy = np.asarray(row.crossing_xy)
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=30,
                color=STATUS_COLORS[status],
                edgecolors=SURFACE_COLOR,
                linewidths=1.2,
                zorder=7,
            )
            anchors.extend((float(x), float(y), str(row.route)) for x, y in xy)
        elif status == STATUS_NEAR_MISS and row.nearest_xy is not None:
            (px, py), (cx, cy) = row.nearest_xy
            ax.plot([px, cx], [py, cy], color=STATUS_COLORS[status], lw=1.0, zorder=6)
            ax.scatter([px], [py], s=18, color=STATUS_COLORS[status], zorder=7)
            anchors.append((px, py, f"{row.route} ({row.closest_approach_ft:,.0f} ft)"))
    for x, y, text in cluster_labels(anchors, radius=0.04 * (xmax - xmin)):
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=INK_COLOR,
            path_effects=halo,
            zorder=8,
        )

    add_scale_bar(ax, extent, unit_to_ft, halo)
    fig.text(
        0.2 / width_in,
        1 - 0.12 / height_in,
        cordon_name,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )
    fig.text(
        0.2 / width_in,
        1 - 0.45 / height_in,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=SECONDARY_INK,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, (footer_in - 0.08) / height_in),
        ncol=legend_cols,
        fontsize=8,
        frameon=False,
    )
    fig.text(0.5, 0.08 / height_in, note, ha="center", va="bottom", fontsize=7, color=MUTED_INK)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE_COLOR)


def unique_map_names(names: Sequence[str]) -> dict[str, str]:
    """Map each cordon name to a unique, filesystem-safe PNG file name.

    Names are deduplicated case-insensitively so they stay unique on Windows.
    """
    used: set[str] = set()
    out: dict[str, str] = {}
    for name in names:
        stem = f"cordon_{sanitize_filename_component(name)}"
        candidate, n = stem, 2
        while candidate.lower() in used:
            candidate, n = f"{stem}_{n}", n + 1
        used.add(candidate.lower())
        out[name] = f"{candidate}.png"
    return out


# =============================================================================
# FUNCTIONS — run log
# =============================================================================


def _extract_config_from_text(source_text: str, source_label: str) -> str:
    """Return the text between the CONFIG markers in *source_text*.

    Same slicing as :func:`extract_config_block`, for source that is already
    in memory (a notebook cell) rather than on disk.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
    """
    lines = source_text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_label}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )
    return "\n".join(lines[begin_idx + 1 : end_idx])


def _notebook_path(ip: Any) -> Optional[str]:
    """Return the notebook file path from the IPython kernel namespace, or None.

    Tries the known frontend-specific variables in order:
      * ``__vsc_ipynb_file__``  – VS Code / Pylance Jupyter extension
      * ``__session__``         – JupyterLab ≥ 4 kernel session path

    Returns None when running in an environment that doesn't expose a path
    (ArcGIS Pro notebooks, classic Notebook, plain IPython console, etc.).
    """
    for var in ("__vsc_ipynb_file__", "__session__"):
        val = ip.user_ns.get(var)
        if val:
            return str(val)
    return None


def resolve_config_source() -> tuple[str, str]:
    """Return ``(config_text, source_label)`` for the running configuration.

    Resolution order:

    1. **Jupyter / IPython kernel** – walks the ``In`` cell history in reverse
       and uses the most recent cell that contains both CONFIG markers, so
       pasting this whole script into a cell still yields a faithful run log.
       ``__file__`` does not exist in a kernel, which is why this comes first.
    2. **Plain script / imported module** – reads ``__file__``.

    Raises:
        RuntimeError: If neither source can be located.
        ValueError: If the located source has no CONFIG markers.
        OSError: If the source file cannot be read.
    """
    # IPython registers itself in sys.modules when a kernel starts; avoid a
    # direct import so the dependency stays optional. get_ipython() returns
    # None outside a live kernel.
    _ipython = sys.modules.get("IPython")
    ip = _ipython.get_ipython() if _ipython is not None else None  # type: ignore[attr-defined]
    if ip is not None:
        history: List[str] = ip.user_ns.get("In", [])
        for cell in reversed(history):
            if CONFIG_BEGIN_MARKER in cell and CONFIG_END_MARKER in cell:
                label = _notebook_path(ip) or "<Jupyter cell>"
                return _extract_config_from_text(cell, label), label

    file_attr: Optional[str] = globals().get("__file__")
    if file_attr is not None:
        source_path = Path(file_attr).resolve()
        return extract_config_block(source_path), str(source_path)

    raise RuntimeError(
        "Cannot locate the script source for the run log: __file__ is not defined "
        "and no notebook cell containing the CONFIG markers was found. Run the cell "
        "that holds the CONFIGURATION block (or the whole script) before writing output."
    )


def write_run_log(
    output_dir: Path, summary_lines: List[str], config_text: str, source_label: str
) -> bool:
    """Write the verbatim config block plus a run summary into *output_dir*.

    Args:
        output_dir: Folder that receives ``RUN_LOG_FILENAME``.
        summary_lines: Human-readable lines describing this run.
        config_text: The CONFIG block, from :func:`resolve_config_source`.
        source_label: Where the config came from (file path or notebook).

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / RUN_LOG_FILENAME
    lines: List[str] = [
        "=" * 72,
        "CORDON CROSSING ROUTES RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_label}",
        "",
        "-" * 72,
        "RUN SUMMARY",
        "-" * 72,
        *summary_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim)",
        "-" * 72,
        CONFIG_BEGIN_MARKER,
        config_text,
        CONFIG_END_MARKER,
        "",
    ]
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logging.error("Could not write run log '%s': %s", log_path, exc)
        return False
    logging.info("Run log written → %s", log_path)
    return True


# =============================================================================
# ORCHESTRATION
# =============================================================================


def load_route_paths(
    gtfs_path: Path, service_ids: Sequence[str]
) -> tuple[dict[str, LineString], dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Load the feed and resolve one path (GTFS shape or stop trace) per trip.

    Args:
        gtfs_path: GTFS feed folder or .zip.
        service_ids: Optional service_id filter (empty = every trip).

    Returns:
        ``(lines, sources, trips, routes)``: WGS 84 lines keyed by path id, the
        source of each path (``shapes.txt`` or ``stop-to-stop``), the selected
        trips with a ``path_id`` column (NaN where no geometry could be built),
        and routes.txt.

    Raises:
        OSError: If routes.txt or trips.txt is missing, or stop_times.txt and
            stops.txt are needed but missing.
        ValueError: If no trips remain after the service filter.
        ServiceSelectionError: If a requested service_id is not in trips.txt.
    """
    gtfs = load_gtfs_data(str(gtfs_path), files=["routes.txt", "trips.txt"])
    routes, trips = gtfs["routes"], gtfs["trips"]
    for table in (routes, trips):
        for col in ("route_id", "trip_id", "shape_id", "direction_id"):
            if col in table.columns:
                table[col] = table[col].fillna("").astype(str).str.strip()
    if service_ids:
        trips = select_trips_by_service(trips, service_ids)
    if trips.empty:
        raise ValueError("No trips remain after the service filter.")

    try:
        shapes: Optional[pd.DataFrame] = load_gtfs_data(str(gtfs_path), files=["shapes.txt"])[
            "shapes"
        ]
    except (OSError, ValueError) as exc:
        logging.warning("shapes.txt not usable (%s); tracing routes stop to stop.", exc)
        shapes = None
    lines = build_shape_lines(shapes)
    sources = {path_id: SOURCE_SHAPES for path_id in lines}

    trips = trips.copy()
    shape_ids = trips["shape_id"] if "shape_id" in trips.columns else pd.Series("", trips.index)
    trips["path_id"] = shape_ids.where(shape_ids.isin(set(lines)))
    missing = trips.loc[trips["path_id"].isna(), "trip_id"]
    if not missing.empty:
        logging.warning(
            "%d trip(s) have no usable shape in shapes.txt; tracing them stop to stop, which "
            "follows straight lines between stops and is less exact.",
            len(missing),
        )
        extra = load_gtfs_data(str(gtfs_path), files=["stop_times.txt", "stops.txt"])
        stop_lines, trip_to_path = build_stop_pattern_lines(
            missing, extra["stop_times"], extra["stops"]
        )
        lines.update(stop_lines)
        sources.update({path_id: SOURCE_STOPS for path_id in stop_lines})
        traced = trips["trip_id"].map(trip_to_path)
        trips["path_id"] = trips["path_id"].fillna(traced)
        n_lost = int(trips["path_id"].isna().sum())
        if n_lost:
            logging.warning(
                "%d trip(s) have neither a shape nor two located stops; left out.", n_lost
            )
    return lines, sources, trips, routes


def run(
    gtfs_path: Path | None = None,
    cordons_path: Path | None = None,
    output_dir: Path | None = None,
    cordons_zip_member: Optional[str] = None,
    cordon_name_field: Optional[str] = None,
    near_miss_ft: Optional[float] = None,
    service_ids: Optional[Sequence[str]] = None,
    projected_crs: Optional[str] = None,
    write_maps: Optional[bool] = None,
    dpi: Optional[int] = None,
) -> pd.DataFrame:
    """Find the routes that cross each cordon, flag near misses, and write the outputs.

    Unset args fall back to the CONFIGURATION block, so ``m.GTFS_PATH = ...;
    m.run()`` works after a plain import.

    Returns:
        The cordon/route table (as written to ROUTES_FILENAME).

    Raises:
        OSError: If an input is missing, or the run log is required but cannot
            be written.
        ValueError: If a setting or input is invalid.
        ServiceSelectionError: If a requested service_id is not in trips.txt.
    """
    gtfs_path = GTFS_PATH if gtfs_path is None else Path(gtfs_path)
    cordons_path = CORDONS_PATH if cordons_path is None else Path(cordons_path)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    cordons_zip_member = (
        CORDONS_ZIP_MEMBER if cordons_zip_member is None else (cordons_zip_member or None)
    )
    cordon_name_field = (
        CORDON_NAME_FIELD if cordon_name_field is None else (cordon_name_field or None)
    )
    near_miss_ft = NEAR_MISS_FT if near_miss_ft is None else float(near_miss_ft)
    service_ids = SERVICE_IDS_TO_INCLUDE if service_ids is None else service_ids
    projected_crs = PROJECTED_CRS if projected_crs is None else (projected_crs or None)
    write_maps = WRITE_MAPS if write_maps is None else write_maps
    dpi = FIGURE_DPI if dpi is None else dpi

    if not math.isfinite(near_miss_ft) or near_miss_ft < 0:
        raise ValueError(f"NEAR_MISS_FT must be zero or a positive number, not {near_miss_ft}.")
    if dpi < 30:
        raise ValueError(f"FIGURE_DPI must be at least 30, not {dpi}.")

    # Locate the CONFIG block first so an untraceable run aborts before any
    # output is written (REQUIRE_RUN_LOG), not after.
    try:
        config_text, source_label = resolve_config_source()
    except (OSError, ValueError, RuntimeError) as exc:
        if REQUIRE_RUN_LOG:
            raise OSError(f"Run log source unavailable and REQUIRE_RUN_LOG is True: {exc}") from exc
        logging.warning("Run log will not record the configuration: %s", exc)
        config_text = "(config block unavailable — see warning in the console log)"
        source_label = "<unknown>"

    lines, sources, trips, routes = load_route_paths(gtfs_path, service_ids)
    used_ids = set(trips["path_id"].dropna())
    if not used_ids:
        raise ValueError(
            "None of the selected trips has usable geometry: no shape in shapes.txt and no "
            "two located stops in stop_times.txt/stops.txt."
        )

    lonlat = gpd.GeoSeries(list(lines.values()), index=list(lines), crs="EPSG:4326")
    lonlat = lonlat[lonlat.index.isin(used_ids)]
    crs = resolve_projected_crs(projected_crs, lonlat)
    logging.info("Measuring distances in %s (%s).", crs.name, crs.to_string())
    unit_to_ft = crs_unit_to_feet(crs)
    threshold = near_miss_ft / unit_to_ft
    paths = lonlat.to_crs(crs)

    cordons = load_cordons(cordons_path, crs, cordon_name_field, cordons_zip_member)

    routes_info = pd.DataFrame(
        {
            "route_id": routes["route_id"],
            "route": [
                route_label(s, r)
                for s, r in zip(
                    routes.get("route_short_name", pd.Series("", routes.index)), routes["route_id"]
                )
            ],
            "route_long_name": routes.get("route_long_name", pd.Series("", routes.index)).fillna(
                ""
            ),
            "route_type": routes.get("route_type", pd.Series("", routes.index)).fillna(""),
        }
    ).drop_duplicates("route_id")
    with_path = trips.dropna(subset=["path_id"])
    directions = (
        with_path["direction_id"]
        if "direction_id" in with_path.columns
        else pd.Series("", with_path.index)
    )
    usage = (
        with_path.assign(direction_id=directions)
        .groupby(["route_id", "path_id"], sort=False)
        .agg(
            n_trips=("trip_id", "size"),
            direction_ids=("direction_id", lambda s: "; ".join(sorted({v for v in s if v}))),
        )
        .reset_index()
    )
    labels = dict(zip(routes_info["route_id"], routes_info["route"]))
    usage["route"] = usage["route_id"].map(labels).fillna(usage["route_id"])
    route_trip_totals = trips.groupby("route_id")["trip_id"].size().to_dict()

    network_box = box(*paths.total_bounds).buffer(threshold)
    results: dict[str, dict[str, PathResult]] = {}
    for name, geom in zip(cordons["cordon"], cordons.geometry):
        if not geom.intersects(network_box):
            logging.warning(
                "Cordon %r lies outside the transit network's extent. Check that the layer's "
                "coordinate reference system is right.",
                name,
            )
        results[name] = assess_cordon(geom, paths, threshold, unit_to_ft)

    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    shape_table = build_shape_table(cordons, results, usage, sources, to_lonlat, unit_to_ft)
    route_table = build_route_table(shape_table, routes_info, route_trip_totals)

    for row in route_table.itertuples(index=False):
        if row.status in (STATUS_MARGINAL, STATUS_NEAR_MISS):
            logging.warning("Cordon %r, route %s: %s", row.cordon, row.route, row.note)

    output_dir.mkdir(parents=True, exist_ok=True)
    map_files: dict[str, str] = {}
    if write_maps:
        maps_dir = output_dir / MAPS_SUBDIR
        maps_dir.mkdir(parents=True, exist_ok=True)
        file_names = unique_map_names(list(cordons["cordon"]))
        any_stop_traced = SOURCE_STOPS in set(sources[p] for p in paths.index)
        for name, geom in zip(cordons["cordon"], cordons.geometry):
            out_path = maps_dir / file_names[name]
            plot_cordon_map(
                name,
                geom,
                paths,
                shape_table[shape_table["cordon"] == name],
                route_table[route_table["cordon"] == name],
                near_miss_ft,
                unit_to_ft,
                out_path,
                dpi,
                any_stop_traced,
            )
            map_files[name] = str(Path(MAPS_SUBDIR) / file_names[name])
        logging.info("Wrote %d map(s) → %s", len(map_files), maps_dir)

    summary = build_summary_table(cordons, route_table, map_files)
    summary_path = output_dir / SUMMARY_FILENAME
    routes_path = output_dir / ROUTES_FILENAME
    shapes_path = output_dir / SHAPES_FILENAME
    summary.to_csv(summary_path, index=False)
    route_table.to_csv(routes_path, index=False)
    shape_table.drop(columns=list(INTERNAL_COLUMNS)).to_csv(shapes_path, index=False)
    logging.info("Wrote %d cordon(s) → %s", len(summary), summary_path)
    logging.info("Wrote %d cordon/route row(s) → %s", len(route_table), routes_path)
    logging.info("Wrote %d cordon/shape row(s) → %s", len(shape_table), shapes_path)

    n_stop_paths = sum(1 for p in paths.index if sources[p] == SOURCE_STOPS)
    cordon_lines = [
        f"  {row.cordon} ({row.cordon_type}): {row.n_routes_crossing} crossing "
        f"({row.n_routes_crossing_marginal} marginal), {row.n_routes_near_miss} near miss"
        for row in summary.itertuples(index=False)
    ]
    summary_lines = [
        f"GTFS feed:            {gtfs_path}",
        f"Cordon layer:         {cordons_path}"
        + (f" ({cordons_zip_member})" if cordons_zip_member else ""),
        f"Name field:           {cordon_name_field or 'auto'}",
        f"Near-miss threshold:  {near_miss_ft:,.0f} ft",
        f"Service filter:       {', '.join(service_ids) or 'all service_ids'}",
        f"Projected CRS:        {crs.to_string()} ({crs.name})",
        f"Route paths:          {len(paths)} ({len(paths) - n_stop_paths} from shapes.txt, "
        f"{n_stop_paths} traced stop to stop)",
        f"Trips used:           {len(with_path)} of {len(trips)}",
        f"Summary CSV:          {summary_path}",
        f"Route CSV:            {routes_path}",
        f"Shape CSV:            {shapes_path}",
        f"Maps:                 {len(map_files)}"
        + (f" in {output_dir / MAPS_SUBDIR}" if map_files else ""),
        "Cordons:",
        *cordon_lines,
    ]
    if not write_run_log(output_dir, summary_lines, config_text, source_label) and REQUIRE_RUN_LOG:
        raise OSError(
            f"Run log could not be written to '{output_dir}' and REQUIRE_RUN_LOG is True."
        )

    n_flagged = int(route_table["status"].isin([STATUS_MARGINAL, STATUS_NEAR_MISS]).sum())
    logging.info(
        "Cordon crossing check complete — %d cordon(s), %d route crossing(s), %d flagged for "
        "review.",
        len(cordons),
        int(route_table["status"].isin(CROSSING_STATUSES).sum()),
        n_flagged,
    )
    return route_table


# =============================================================================
# MAIN
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; every flag defaults to its CONFIGURATION constant."""
    parser = argparse.ArgumentParser(
        description=(
            "List the GTFS routes that cross each cordon or screenline and flag near misses. "
            "Defaults come from the configuration block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gtfs-path", type=Path, default=GTFS_PATH, help="GTFS feed folder or .zip archive."
    )
    parser.add_argument(
        "--cordons-path",
        type=Path,
        default=CORDONS_PATH,
        help="Cordon layer: .shp, zipped shapefile, .gpkg or .geojson.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for CSVs, maps and run log."
    )
    parser.add_argument(
        "--cordons-zip-member",
        default=CORDONS_ZIP_MEMBER,
        help="Shapefile to use when --cordons-path is a zip holding several.",
    )
    parser.add_argument(
        "--cordon-name-field",
        default=CORDON_NAME_FIELD,
        help="Attribute holding cordon names (default: detect a name field).",
    )
    parser.add_argument(
        "--near-miss-ft",
        type=float,
        default=NEAR_MISS_FT,
        help="Flag routes within this many feet of a cordon, or crossing it by less.",
    )
    parser.add_argument(
        "--service-ids",
        nargs="*",
        default=SERVICE_IDS_TO_INCLUDE,
        metavar="SERVICE_ID",
        help="Restrict to these service_ids from trips.txt (empty = all).",
    )
    parser.add_argument(
        "--projected-crs",
        default=PROJECTED_CRS,
        help="Projected CRS for distances; pass '' to use the feed's UTM zone.",
    )
    parser.add_argument(
        "--write-maps",
        action=argparse.BooleanOptionalAction,
        default=WRITE_MAPS,
        help="Write one PNG map per cordon.",
    )
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="PNG resolution.")
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the config block.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders or a service_id is unknown.
    """
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sentinels = {
        Path(r"Path\To\Your\GTFS_Folder"),
        Path(r"Path\To\Your\Cordons.shp"),
        Path(r"Path\To\Your\Output_Folder"),
    }
    if {args.gtfs_path, args.cordons_path, args.output_dir} & sentinels:
        logging.warning(
            "GTFS_PATH, CORDONS_PATH and/or OUTPUT_DIR are still placeholders. Update the "
            "configuration block or pass --gtfs-path/--cordons-path/--output-dir before running."
        )
        return 2
    try:
        run(
            gtfs_path=args.gtfs_path,
            cordons_path=args.cordons_path,
            output_dir=args.output_dir,
            cordons_zip_member=args.cordons_zip_member,
            cordon_name_field=args.cordon_name_field,
            near_miss_ft=args.near_miss_ft,
            service_ids=args.service_ids,
            projected_crs=args.projected_crs,
            write_maps=args.write_maps,
            dpi=args.dpi,
        )
    except ServiceSelectionError as exc:
        logging.error("%s", exc)
        return 2
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


# Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
# injected argv away from argparse so the config block stays in charge.
if __name__ == "__main__":
    raise SystemExit(main())
