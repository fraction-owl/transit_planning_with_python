"""Clean a private-shuttle operator registry and roll shuttle presence up to GTFS routes.

ArcPy port of ``private_shuttle_coverage_by_route_gpd.py`` for environments
running ArcGIS Pro's bundled Python (no geopandas/shapely). Private employer,
university, hospital, and residential shuttles overlap public transit markets,
but agencies rarely hold them in analysis-ready form — the typical source is a
hand-maintained spreadsheet of operators with partial addresses, partial
coordinates, blank rows, and free-text notes. This script turns that registry
into three products so planners can see where private shuttles run and the
modeling pipeline can test whether they move ridership:

1. **Clean registry** (``private_shuttles_clean.csv``) — deduplicated,
   whitespace-trimmed rows with usable WGS84 coordinates, plus a ``category``
   column derived from the notes: ``transit_feeder`` (the shuttle connects to
   rail/metro/transit, i.e. likely *complements* fixed-route service),
   ``shuttle`` (a shuttle with no stated transit connection, i.e. a potential
   *competitor*), or ``unspecified``. Service ``start_date`` / ``end_date``
   columns are carried through and normalized to ISO dates (blank = unknown /
   still operating), so dated openings and closures survive cleaning — the
   raw material for panel (route × month) modeling later.
2. **Geocoding worklist** (``private_shuttles_needs_geocoding.csv``) — rows
   whose coordinates are missing or invalid (out of range, or the 0,0
   "null island" geocoder artifact). No geocoding is attempted here — the
   script is offline by design; fill these in and re-run.
3. **POI layer** (``Private_Shuttle_Stops.shp``) — the clean rows as a point
   shapefile whose name and id column (``NAME``) match the
   ``("Private_Shuttle_Stops.shp", "NAME")`` entry listed in
   ``points_of_interest_coverage_arcpy.py``'s ``LAYER_SPECS``, so copying it
   (with its sidecar files) anywhere under that script's ``SHP_INPUT_DIR``
   wires private shuttles into the strategic-site coverage counts with no
   further configuration. The GeoPandas coverage tool discovers loose
   shapefiles too, so the same files satisfy its matching LAYER_SPECS entry;
   only the zipped delivery of the GeoPandas twin differs.

Optionally (when a GTFS folder is supplied), it also writes
``private_shuttle_coverage_by_route.csv`` — one row per ``route_id`` counting
the shuttle sites inside each route's catchment (``shuttle_sites_served``) and
the transit-feeder subset (``shuttle_feeder_sites_served``). That table is the
modeling hook: registered in ``scripts/modeling/orchestrator_jobs_public.json``,
it joins the ridership anchor by ``route_id`` so the OLS / ML models
(``monthly_ridership_model.py``, ``ridership_ml_model.py``) can estimate
whether private-shuttle presence helps explain route-level ridership. Set
``ACTIVE_AS_OF`` (or ``--as-of``) to count only the shuttles active on a given
date — match it to the ridership anchor's period so a shuttle that shut down
years ago doesn't inflate today's counts. A site with an unknown start or end
date is treated as active.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags,
e.g. ``--shuttles-csv``, ``--gtfs-dir``, ``--output-dir``) and run from a
shell, ArcGIS Pro's Python window, or a Jupyter notebook using ArcGIS Pro's
bundled Python. Without ``--gtfs-dir`` the script runs in prep-only mode
(clean registry + worklist + POI layer, no route rollup). Running the script
completely unedited also works: in an interactive session (terminal or
notebook) it prompts for the registry CSV, the output folder, and the
optional GTFS folder instead of exiting, so a new user can copy, paste, and
hit run.

Assumptions
-----------
- The registry CSV carries the operator name, street address fields, WGS84
  coordinates (X = longitude, Y = latitude), and a free-text notes column;
  column names are configurable and matched case-insensitively.
- The analysis spatial reference is projected; the buffer distance is given in
  feet and converted to the spatial reference's linear unit.

Requires
--------
ArcGIS Pro (arcpy) and pandas (bundled with Pro).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Final, List, Mapping, NamedTuple, Optional, Sequence

import arcpy
import pandas as pd

# Path to this file, used to extract the config block for the run log. ``__file__``
# is undefined when the code is pasted into a notebook cell, so a configured
# fallback keeps the run log working there too.
SELF_PATH: Final[Path] = (
    Path(__file__)
    if "__file__" in globals()
    else Path("private_shuttle_coverage_by_route_arcpy.py")
)

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Top-level paths
SHUTTLES_CSV = Path(r"Path\To\Your\private_shuttles.csv")  # the operator registry CSV
GTFS_DIR: Path | None = None  # GTFS folder; None → skip the route rollup
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")  # where all outputs are written

# Registry column names, matched case-insensitively against the CSV header.
# X/Y are WGS84 longitude/latitude; the coordinate and date columns may be
# absent entirely (rows without coordinates land on the geocoding worklist;
# absent dates mean every site is treated as currently operating).
COMPANY_COL = "Company"
ADDRESS_COL = "Address"
CITY_COL = "City"
STATE_COL = "State"
ZIP_COL = "Zip"
X_COL = "X"
Y_COL = "Y"
NOTES_COL = "Notes"
# Service dates: when the shuttle began / ceased operating. Any format pandas
# can parse is accepted per-value; blank = unknown (start) / still running (end).
START_DATE_COL = "Start Date"
END_DATE_COL = "End Date"

# Case-insensitive regex applied to the notes column. A match marks the site a
# ``transit_feeder`` (its shuttle connects to the regional transit network);
# otherwise any mention of "shuttle" yields ``shuttle`` and the rest are
# ``unspecified``.
FEEDER_NOTES_PATTERN = r"metro|rail|station|transit|train"

# POI layer emitted for points_of_interest_coverage_arcpy.py. The filename and
# id column must stay in sync with that script's LAYER_SPECS entry
# ("Private_Shuttle_Stops.shp", "NAME"). Set WRITE_POI_LAYER = False to skip it.
WRITE_POI_LAYER = True
POI_LAYER_FILENAME = "Private_Shuttle_Stops.shp"
POI_ID_COLUMN = "NAME"

# Output filenames. The coverage name matches the orchestrator registry's
# private_shuttle_coverage_by_route.csv entry.
CLEAN_CSV_NAME = "private_shuttles_clean.csv"
NEEDS_GEOCODING_CSV_NAME = "private_shuttles_needs_geocoding.csv"
COVERAGE_CSV_NAME = "private_shuttle_coverage_by_route.csv"
RUN_LOG_NAME = "private_shuttles_runlog.txt"

# Route rollup options (only used when a GTFS folder is supplied).
ROUTE_FILTER: list[str] = []  # only these route_id values; empty = all
USE_SHAPE_BUFFER = False  # False → buffer stops (simple catchment); True → route geometry
BUFFER_DIST_FT = 1320.0  # ¼ mile in feet
# Count only shuttles active on this date (e.g. "2026-06-01" — match it to the
# ridership anchor's period). None counts every site regardless of its dates;
# sites with an unknown start or end date are always treated as active.
ACTIVE_AS_OF: str | None = None

# Projected spatial reference (WKID) used for buffering and the point-in-polygon
# tests. EPSG:3857 (Web Mercator) works globally; its latitude-dependent scale
# distortion is corrected automatically when buffering (see
# _buffer_distance_in_sr_units). Swap for a local CRS (e.g. 2283 for northern
# Virginia in feet) when higher spatial accuracy is needed.
PROJECTED_CRS_WKID = 3857

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so an output directory is
# never left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# Category values written to the ``category`` column.
CATEGORY_TRANSIT_FEEDER = "transit_feeder"
CATEGORY_SHUTTLE = "shuttle"
CATEGORY_UNSPECIFIED = "unspecified"

# Reason values written to the geocoding worklist.
REASON_MISSING_COORDS = "missing_coordinates"
REASON_INVALID_COORDS = "invalid_coordinates"

# Canonical (lower-case) registry column order used in the outputs.
_TEXT_COLUMNS: tuple[str, ...] = ("company", "address", "city", "state", "zip", "notes")


class PrepResult(NamedTuple):
    """The three tables a run produces.

    Attributes:
        clean: Deduplicated registry rows with usable coordinates.
        needs_geocoding: Rows lacking usable coordinates (the manual worklist).
        coverage: Route-level rollup, or None when no GTFS folder was supplied.
    """

    clean: pd.DataFrame
    needs_geocoding: pd.DataFrame
    coverage: pd.DataFrame | None


# =============================================================================
# REGISTRY CLEANING
# =============================================================================


def load_registry_csv(
    shuttles_csv: Path,
    company_col: str = COMPANY_COL,
    address_col: str = ADDRESS_COL,
    city_col: str = CITY_COL,
    state_col: str = STATE_COL,
    zip_col: str = ZIP_COL,
    x_col: str = X_COL,
    y_col: str = Y_COL,
    notes_col: str = NOTES_COL,
    start_date_col: str = START_DATE_COL,
    end_date_col: str = END_DATE_COL,
) -> pd.DataFrame:
    """Read the operator registry CSV into canonically named string columns.

    Every configured column is matched case-insensitively against the file's
    header. Missing optional columns (everything except the company column)
    are created empty, so a registry that was never geocoded or dated still
    loads.

    Args:
        shuttles_csv: Path to the registry CSV.
        company_col: Header holding the operator/site name (required).
        address_col: Street address header.
        city_col: City header.
        state_col: State header.
        zip_col: ZIP code header (read as text to keep leading zeros).
        x_col: WGS84 longitude header.
        y_col: WGS84 latitude header.
        notes_col: Free-text notes header.
        start_date_col: Service start-date header (blank value = unknown).
        end_date_col: Service end-date header (blank value = still running).

    Returns:
        DataFrame with string columns ``company``, ``address``, ``city``,
        ``state``, ``zip``, ``notes``, ``lon_raw``, ``lat_raw``,
        ``start_date_raw``, ``end_date_raw``.

    Raises:
        FileNotFoundError: If ``shuttles_csv`` does not exist.
        ValueError: If the company column cannot be found in the header.
    """
    if not shuttles_csv.exists():
        raise FileNotFoundError(f"Shuttle registry not found: {shuttles_csv}")

    raw = pd.read_csv(shuttles_csv, dtype=str)
    header = {str(col).strip().lower(): col for col in raw.columns}

    def _resolve(name: str) -> str | None:
        return header.get(name.strip().lower())

    company_src = _resolve(company_col)
    if company_src is None:
        raise ValueError(
            f"Column '{company_col}' not found in {shuttles_csv.name}. "
            f"Available columns: {list(raw.columns)}. Adjust COMPANY_COL if the "
            "registry names it differently."
        )

    wanted: dict[str, str | None] = {
        "company": company_src,
        "address": _resolve(address_col),
        "city": _resolve(city_col),
        "state": _resolve(state_col),
        "zip": _resolve(zip_col),
        "notes": _resolve(notes_col),
        "lon_raw": _resolve(x_col),
        "lat_raw": _resolve(y_col),
        "start_date_raw": _resolve(start_date_col),
        "end_date_raw": _resolve(end_date_col),
    }
    # Coordinate and date columns are genuinely optional in the wild, so their
    # absence is not worth a warning; the other text columns are expected.
    quiet_when_absent = {"lon_raw", "lat_raw", "start_date_raw", "end_date_raw"}
    out = pd.DataFrame(index=raw.index)
    for target, source in wanted.items():
        out[target] = raw[source] if source is not None else ""
        if source is None and target not in quiet_when_absent:
            logging.warning(
                "Column '%s' not found in %s; '%s' left empty.", target, shuttles_csv.name, target
            )
    logging.info("Loaded %d registry row(s) from %s", len(out), shuttles_csv.name)
    return out.fillna("")


def categorize_notes(notes: pd.Series, feeder_pattern: str = FEEDER_NOTES_PATTERN) -> pd.Series:
    """Classify each free-text note into a shuttle category.

    Args:
        notes: The registry's notes column (strings).
        feeder_pattern: Case-insensitive regex marking a transit connection.

    Returns:
        Series of ``transit_feeder`` (note matches *feeder_pattern*),
        ``shuttle`` (note mentions a shuttle but no transit connection), or
        ``unspecified`` (anything else, including blank).
    """
    text = notes.fillna("").astype(str)
    categories = pd.Series(CATEGORY_UNSPECIFIED, index=notes.index)
    categories[text.str.contains(r"shuttle", case=False, regex=True)] = CATEGORY_SHUTTLE
    categories[text.str.contains(feeder_pattern, case=False, regex=True)] = CATEGORY_TRANSIT_FEEDER
    return categories


def _parse_registry_date(value: str) -> str:
    """Parse one raw date string into ISO ``YYYY-MM-DD``, or ``""`` when unparseable."""
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _normalize_dates(raw: pd.Series, label: str) -> pd.Series:
    """Parse a raw service-date column into ISO ``YYYY-MM-DD`` strings.

    Registries collect dates in whatever format each editor typed, so each
    value is parsed individually (``2021-06-15``, ``6/15/2021``, ``June 2021``
    all work); scalar parsing also sidesteps pandas-version differences in
    mixed-format Series parsing (older ArcGIS Pro releases bundle pandas 1.x).
    Blank stays blank (unknown / still running); a non-blank value that cannot
    be parsed (e.g. ``TBD``) is blanked with a warning so it is treated as
    unknown rather than silently mis-dated.

    Args:
        raw: The trimmed raw date strings.
        label: Column name used in the warning message.

    Returns:
        Series of ISO date strings, ``""`` where the date is unknown.
    """
    iso = raw.astype(str).apply(_parse_registry_date)
    unparseable = (iso == "") & (raw != "")
    if unparseable.any():
        logging.warning(
            "%d unparseable %s value(s) treated as unknown: %s",
            int(unparseable.sum()),
            label,
            sorted(raw[unparseable].unique()),
        )
    return iso


def clean_registry(
    registry: pd.DataFrame,
    feeder_pattern: str = FEEDER_NOTES_PATTERN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the raw registry into clean (geolocated) rows and a geocoding worklist.

    Cleaning steps, in order:
        1. Trim whitespace on every text column; upper-case the state.
        2. Drop rows with neither a company nor an address (blank filler rows).
        3. Drop exact duplicate rows.
        4. Parse coordinates and validate them: both present, longitude within
           ±180, latitude within ±90, and not the (0, 0) geocoder artifact.
        5. Normalize the service dates to ISO (see :func:`_normalize_dates`).
        6. Categorize the notes (see :func:`categorize_notes`).

    Args:
        registry: Output of :func:`load_registry_csv`.
        feeder_pattern: Passed through to :func:`categorize_notes`.

    Returns:
        ``(clean, needs_geocoding)``. ``clean`` carries the text columns plus
        numeric ``lon`` / ``lat``, ISO ``start_date`` / ``end_date``, and
        ``category``; ``needs_geocoding`` carries the text columns, the dates,
        ``category``, and a ``reason`` column (``missing_coordinates`` or
        ``invalid_coordinates``).
    """
    df = registry.copy()
    for col in (*_TEXT_COLUMNS, "lon_raw", "lat_raw", "start_date_raw", "end_date_raw"):
        df[col] = df[col].astype(str).str.strip()
    df["state"] = df["state"].str.upper()

    blank = (df["company"] == "") & (df["address"] == "")
    if blank.any():
        logging.info("Dropped %d blank row(s) (no company and no address).", int(blank.sum()))
    df = df[~blank]

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logging.info("Dropped %d exact duplicate row(s).", before - len(df))

    lon = pd.to_numeric(df["lon_raw"], errors="coerce")
    lat = pd.to_numeric(df["lat_raw"], errors="coerce")
    missing = lon.isna() | lat.isna()
    out_of_range = ~missing & ((lon.abs() > 180.0) | (lat.abs() > 90.0))
    null_island = ~missing & (lon == 0.0) & (lat == 0.0)
    invalid = out_of_range | null_island
    usable = ~missing & ~invalid

    df = df.assign(
        start_date=_normalize_dates(df["start_date_raw"], "start date"),
        end_date=_normalize_dates(df["end_date_raw"], "end date"),
        category=categorize_notes(df["notes"], feeder_pattern),
    )

    kept_cols = [*_TEXT_COLUMNS, "start_date", "end_date", "category"]
    clean = df.loc[usable, kept_cols].copy()
    clean.insert(_TEXT_COLUMNS.index("notes"), "lon", lon[usable])
    clean.insert(_TEXT_COLUMNS.index("notes") + 1, "lat", lat[usable])

    needs = df.loc[~usable, kept_cols].copy()
    reason = pd.Series(REASON_INVALID_COORDS, index=needs.index)
    reason[missing.loc[needs.index]] = REASON_MISSING_COORDS
    needs["reason"] = reason

    logging.info(
        "Cleaned registry: %d row(s) with usable coordinates, %d for the geocoding "
        "worklist (%d missing, %d invalid).",
        len(clean),
        len(needs),
        int((needs["reason"] == REASON_MISSING_COORDS).sum()),
        int((needs["reason"] == REASON_INVALID_COORDS).sum()),
    )
    return clean.reset_index(drop=True), needs.reset_index(drop=True)


# =============================================================================
# POI LAYER EXPORT
# =============================================================================


def write_poi_shapefile(
    clean: pd.DataFrame,
    output_dir: Path,
    filename: str = POI_LAYER_FILENAME,
    id_column: str = POI_ID_COLUMN,
) -> Path:
    """Write the WGS84 point shapefile for the strategic-site coverage tool.

    Besides the id column the layer carries the service dates as ISO strings
    (``START_DATE`` / ``END_DATE``, blank = unknown) for mapping use; the
    coverage tool itself reads only the id column. An existing shapefile of
    the same name is overwritten.

    Args:
        clean: The clean registry rows (must carry ``company``/``lon``/``lat``
            and the normalized ``start_date``/``end_date``).
        output_dir: Folder the shapefile (and its sidecar files) is written to.
        filename: Shapefile name expected by the coverage tool's LAYER_SPECS
            (``Private_Shuttle_Stops.shp``).
        id_column: Attribute column name expected by the coverage tool's
            LAYER_SPECS (``NAME`` for Private_Shuttle_Stops.shp).

    Returns:
        Path of the shapefile written.
    """
    out_path = output_dir / filename
    if arcpy.Exists(str(out_path)):
        arcpy.management.Delete(str(out_path))
    arcpy.management.CreateFeatureclass(
        str(output_dir), filename, "POINT", spatial_reference=arcpy.SpatialReference(4326)
    )
    # Shapefile attribute names cap at 10 characters; NAME / START_DATE /
    # END_DATE all fit, and the ISO dates fit a 10-character text field.
    arcpy.management.AddField(str(out_path), id_column, "TEXT", field_length=254)
    arcpy.management.AddField(str(out_path), "START_DATE", "TEXT", field_length=10)
    arcpy.management.AddField(str(out_path), "END_DATE", "TEXT", field_length=10)
    with arcpy.da.InsertCursor(
        str(out_path), [id_column, "START_DATE", "END_DATE", "SHAPE@XY"]
    ) as cursor:
        for row in clean.itertuples(index=False):
            cursor.insertRow(
                (row.company, row.start_date, row.end_date, (float(row.lon), float(row.lat)))
            )
    return out_path


# =============================================================================
# ROUTE CATCHMENTS (mirrors school_coverage_by_route_arcpy.py)
# =============================================================================


def _load_gtfs_tables(gtfs_dir: Path, *, need_shapes: bool) -> Mapping[str, pd.DataFrame]:
    """Load the GTFS text files needed to build route catchments.

    Args:
        gtfs_dir: Directory containing GTFS .txt files.
        need_shapes: When True, shapes.txt is also required (shape-buffer mode).

    Returns:
        Mapping keyed by table name (without .txt) to DataFrame.

    Raises:
        FileNotFoundError: If a required file is missing.
    """
    names = ["routes", "trips", "stop_times", "stops"]
    if need_shapes:
        names.append("shapes")

    tables: Dict[str, pd.DataFrame] = {}
    for name in names:
        path = gtfs_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[name] = pd.read_csv(path, dtype={"route_id": str})
        logging.debug("Loaded %s (%d rows)", name, len(tables[name]))
    return tables


def _buffer_distance_in_sr_units(
    target_sr: arcpy.SpatialReference,
    buffer_dist_ft: float,
    mean_latitude: Optional[float] = None,
) -> float:
    """Convert a buffer distance in feet to the linear units of *target_sr*.

    Web Mercator (WKID 3857 / 102100) inflates distances by roughly
    ``1/cos(latitude)``: at 39°N a true quarter mile spans ~1.29x as many map
    "meters", so a buffer drawn in raw map units under-covers the ground by
    the same factor. When *target_sr* is Web Mercator and *mean_latitude* is
    supplied, the distance is scaled up accordingly so it spans a true ground
    distance at that latitude.

    Args:
        target_sr: Projected spatial reference used for geometry operations.
        buffer_dist_ft: Buffer distance in feet.
        mean_latitude: Mean WGS 84 latitude of the analysis features, used to
            correct Web Mercator's scale distortion. Ignored for other CRSs.

    Returns:
        Buffer distance expressed in the units of *target_sr*.

    Raises:
        ValueError: If *target_sr* is not a projected spatial reference.
    """
    if target_sr.type != "Projected":
        raise ValueError(
            f"PROJECTED_CRS_WKID must reference a projected spatial reference; "
            f"got {target_sr.name!r} ({target_sr.type})."
        )
    meters_per_unit = target_sr.metersPerUnit or 1.0
    buffer_units = buffer_dist_ft * 0.3048 / meters_per_unit
    if (
        target_sr.factoryCode in (3857, 102100)
        and mean_latitude is not None
        and -89.0 < mean_latitude < 89.0
    ):
        scale = 1.0 / math.cos(math.radians(mean_latitude))
        buffer_units *= scale
        logging.info(
            "Web Mercator inflates distances by %.4f at latitude %.3f; scaling "
            "the buffer to preserve ground distance.",
            scale,
            mean_latitude,
        )
    logging.debug(
        "Buffer distance: %.2f ft -> %.2f %s",
        buffer_dist_ft,
        buffer_units,
        target_sr.linearUnitName,
    )
    return buffer_units


def _prepare_route_buffers(
    tables: Mapping[str, pd.DataFrame],
    use_shape_buffer: bool,
    buffer_dist_ft: float,
    target_sr: arcpy.SpatialReference,
    route_filter: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Return one buffered catchment geometry per route_id.

    Depending on *use_shape_buffer*, the buffer is built around the union of
    (a) the route's shape(s) or (b) all of its stops. The buffer distance is
    given in feet and converted to the linear units of *target_sr*.

    Args:
        tables: GTFS tables from :func:`_load_gtfs_tables`.
        use_shape_buffer: Buffer route geometry when True, else buffer stops.
        buffer_dist_ft: Catchment radius in feet.
        target_sr: Projected spatial reference used for buffering.
        route_filter: Optional list of route_id values to keep (empty = all).

    Returns:
        List of ``{"route_id": str, "geometry": arcpy polygon}`` records.

    Raises:
        ValueError: If shape-buffer mode is requested but trips.txt or
            shapes.txt is malformed.
    """
    wgs84_sr = arcpy.SpatialReference(4326)

    trips = tables["trips"].copy()
    trips["route_id"] = trips["route_id"].astype(str)

    # stop_id -> (lon, lat), and route_id -> unique stop_ids (the default
    # catchment buffers stops).
    stops = tables["stops"][["stop_id", "stop_lat", "stop_lon"]]

    lat_values = pd.to_numeric(stops["stop_lat"], errors="coerce").dropna()
    mean_latitude = float(lat_values.mean()) if not lat_values.empty else None
    buff_dist = _buffer_distance_in_sr_units(target_sr, buffer_dist_ft, mean_latitude)
    stop_coords = {
        row.stop_id: (row.stop_lon, row.stop_lat)
        for row in stops.itertuples(index=False)
        if pd.notna(row.stop_lat) and pd.notna(row.stop_lon)
    }
    route_stop_ids = (
        tables["stop_times"][["trip_id", "stop_id"]]
        .merge(trips[["trip_id", "route_id"]], on="trip_id", how="inner")
        .drop_duplicates(subset=["route_id", "stop_id"])
        .groupby("route_id")["stop_id"]
        .apply(list)
    )

    shape_lines: Dict[str, arcpy.Polyline] = {}
    route_shapes: Optional[pd.Series] = None
    if use_shape_buffer:
        if "shape_id" not in trips.columns:
            raise ValueError("trips.txt missing shape_id column (required for shape-buffer mode)")
        shapes_df = tables["shapes"]
        if {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.difference(
            shapes_df.columns
        ):
            raise ValueError("shapes.txt missing required columns")
        shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
        for shape_id, grp in shapes_df.groupby("shape_id"):
            pts = [
                arcpy.Point(lon, lat) for lon, lat in zip(grp["shape_pt_lon"], grp["shape_pt_lat"])
            ]
            if len(pts) < 2:
                continue
            line = arcpy.Polyline(arcpy.Array(pts), wgs84_sr)
            shape_lines[str(shape_id)] = line.projectAs(target_sr)
        route_shapes = (
            trips.dropna(subset=["shape_id"])
            .drop_duplicates(subset=["route_id", "shape_id"])
            .groupby("route_id")["shape_id"]
            .apply(list)
        )

    route_ids = route_shapes.index if route_shapes is not None else trips["route_id"].unique()

    buffers: List[Dict[str, object]] = []
    for route_id in route_ids:
        route_id = str(route_id)
        if route_filter and route_id not in route_filter:
            continue

        geom: Optional[arcpy.Geometry] = None
        if use_shape_buffer and route_shapes is not None:
            lines = [
                shape_lines[str(s)] for s in route_shapes.loc[route_id] if str(s) in shape_lines
            ]
            if lines:
                geom = lines[0]
                for line in lines[1:]:
                    geom = geom.union(line)
        else:
            coords = [
                stop_coords[sid] for sid in route_stop_ids.get(route_id, []) if sid in stop_coords
            ]
            if coords:
                multi = arcpy.Multipoint(
                    arcpy.Array([arcpy.Point(lon, lat) for lon, lat in coords]), wgs84_sr
                )
                geom = multi.projectAs(target_sr)

        if geom is None:
            logging.warning("No geometry for route %s – skipped", route_id)
            continue

        buffers.append({"route_id": route_id, "geometry": geom.buffer(buff_dist)})

    return buffers


# =============================================================================
# ROUTE ROLLUP
# =============================================================================


def _parse_as_of(value: str) -> pd.Timestamp:
    """Parse the as-of date, failing with an actionable message.

    Raises:
        ValueError: If *value* cannot be parsed as a date.
    """
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"ACTIVE_AS_OF / --as-of value {value!r} is not a parseable date (use e.g. 2026-06-01)."
        ) from exc


def filter_active_sites(clean: pd.DataFrame, active_as_of: str) -> pd.DataFrame:
    """Return only the clean rows whose shuttle was active on *active_as_of*.

    A site counts as active when its start date is unknown or on/before the
    as-of date AND its end date is unknown or on/after it — i.e. missing dates
    never exclude a site, they only fail to.

    Args:
        clean: The clean registry rows carrying ISO ``start_date``/``end_date``.
        active_as_of: The as-of date (any format ``pd.Timestamp`` accepts).

    Returns:
        The active subset of *clean*.

    Raises:
        ValueError: If *active_as_of* cannot be parsed as a date.
    """
    as_of = _parse_as_of(active_as_of)

    start = pd.to_datetime(clean["start_date"].replace("", pd.NA), errors="coerce")
    end = pd.to_datetime(clean["end_date"].replace("", pd.NA), errors="coerce")
    active = (start.isna() | (start <= as_of)) & (end.isna() | (end >= as_of))
    logging.info(
        "As-of filter %s: %d of %d shuttle site(s) active.",
        as_of.date(),
        int(active.sum()),
        len(clean),
    )
    return clean[active]


def _shuttle_points_in_sr(sites: pd.DataFrame, target_sr: arcpy.SpatialReference) -> pd.DataFrame:
    """Project the clean shuttle sites into *target_sr* as arcpy point geometries.

    Args:
        sites: Clean registry rows carrying ``company``, ``category``, and
            WGS84 ``lon``/``lat``.
        target_sr: Projected spatial reference used for the catchment tests.

    Returns:
        DataFrame with columns ``company``, ``category``, ``x``/``y`` (in
        *target_sr*), and an arcpy ``geometry`` column.
    """
    wgs84_sr = arcpy.SpatialReference(4326)
    records: List[Dict[str, object]] = []
    for row in sites.itertuples(index=False):
        point = arcpy.PointGeometry(
            arcpy.Point(float(row.lon), float(row.lat)), wgs84_sr
        ).projectAs(target_sr)
        records.append(
            {
                "company": row.company,
                "category": row.category,
                "x": point.firstPoint.X,
                "y": point.firstPoint.Y,
                "geometry": point,
            }
        )
    return pd.DataFrame(records, columns=["company", "category", "x", "y", "geometry"])


def summarize_shuttles_by_route(
    route_buffers: List[Dict[str, object]],
    shuttles: pd.DataFrame,
) -> pd.DataFrame:
    """Count shuttle sites (total and transit feeders) per route catchment.

    Args:
        route_buffers: One buffered catchment per route_id (from
            :func:`_prepare_route_buffers`).
        shuttles: Clean shuttle points carrying a ``category`` column (from
            :func:`_shuttle_points_in_sr`), in the same spatial reference as
            *route_buffers*.

    Returns:
        DataFrame with one row per route_id and integer columns
        ``shuttle_sites_served`` and ``shuttle_feeder_sites_served``. Routes
        with no shuttle sites nearby report zeros.
    """
    xs = shuttles["x"].to_numpy(dtype=float) if not shuttles.empty else None
    ys = shuttles["y"].to_numpy(dtype=float) if not shuttles.empty else None

    records: List[Dict[str, object]] = []
    seen: set = set()
    for buffer_rec in route_buffers:
        route_id = str(buffer_rec["route_id"])
        if route_id in seen:
            continue
        seen.add(route_id)

        record: Dict[str, object] = {
            "route_id": route_id,
            "shuttle_sites_served": 0,
            "shuttle_feeder_sites_served": 0,
        }

        if not shuttles.empty:
            geom = buffer_rec["geometry"]
            extent = geom.extent
            candidates = shuttles[
                (xs >= extent.XMin)
                & (xs <= extent.XMax)
                & (ys >= extent.YMin)
                & (ys <= extent.YMax)
            ]
            if not candidates.empty:
                inside = candidates[[not geom.disjoint(pt) for pt in candidates["geometry"]]]
                record["shuttle_sites_served"] = len(inside)
                record["shuttle_feeder_sites_served"] = int(
                    (inside["category"] == CATEGORY_TRANSIT_FEEDER).sum()
                )

        records.append(record)

    summary = pd.DataFrame(
        records, columns=["route_id", "shuttle_sites_served", "shuttle_feeder_sites_served"]
    )
    for col in ("shuttle_sites_served", "shuttle_feeder_sites_served"):
        summary[col] = summary[col].fillna(0).astype(int)
    return summary


def _attach_route_short_name(summary: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable ``route_short_name`` column when routes.txt carries one."""
    if "route_short_name" not in routes_df.columns:
        return summary
    lookup = routes_df.assign(route_id=routes_df["route_id"].astype(str))[
        ["route_id", "route_short_name"]
    ].drop_duplicates(subset="route_id")
    merged = summary.merge(lookup, on="route_id", how="left")
    cols = ["route_id", "route_short_name", "shuttle_sites_served", "shuttle_feeder_sites_served"]
    return merged[cols]


# =============================================================================
# RUN LOG
# =============================================================================


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


def write_run_log(
    output_dir: Path,
    summary_lines: Sequence[str],
    source_path: Path = SELF_PATH,
) -> bool:
    """Write the run-log sidecar: run summary plus the CONFIG block verbatim.

    Args:
        output_dir: Folder the outputs were written to.
        summary_lines: Human-readable lines describing what was produced.
        source_path: Path to this script's source (for config extraction).

    Returns:
        True when the log was written, False on any extraction/write failure.
    """
    log_path = output_dir / RUN_LOG_NAME
    try:
        config_text: str = extract_config_block(source_path)
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: list[str] = [
        "=" * 72,
        "PRIVATE SHUTTLE REGISTRY PREP + ROUTE COVERAGE RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_path.resolve() if source_path.exists() else source_path}",
        "",
        "-" * 72,
        "OUTPUTS",
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
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# MAIN
# =============================================================================


def run(
    shuttles_csv: str | Path | None = None,
    gtfs_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_shape_buffer: bool | None = None,
    buffer_dist_ft: float | None = None,
    route_filter: Sequence[str] | None = None,
    projected_wkid: int | None = None,
    feeder_pattern: str | None = None,
    write_poi_layer: bool | None = None,
    active_as_of: str | None = None,
    require_run_log: bool | None = None,
) -> PrepResult:
    """Clean the registry, export the POI layer, and (optionally) roll up by route.

    Unset args fall back to the CONFIGURATION block, so ``m.SHUTTLES_CSV = ...;
    m.run()`` works after a plain import. The route rollup only runs when a
    GTFS folder is configured (``GTFS_DIR`` or ``--gtfs-dir``); without one the
    script is a pure registry-prep step. When *active_as_of* (or
    ``ACTIVE_AS_OF``) is set, the rollup counts only shuttles active on that
    date; the clean registry, worklist, and POI layer always keep every site.

    Returns:
        A :class:`PrepResult` with the clean registry, the geocoding worklist,
        and the route coverage table (None in prep-only mode).

    Raises:
        ValueError: If *active_as_of* is not a parseable date.
        RuntimeError: If the run-log sidecar cannot be written while
            ``REQUIRE_RUN_LOG`` is enabled.
    """
    shuttles_csv = Path(SHUTTLES_CSV if shuttles_csv is None else shuttles_csv)
    gtfs_dir = GTFS_DIR if gtfs_dir is None else Path(gtfs_dir)
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    use_shape_buffer = USE_SHAPE_BUFFER if use_shape_buffer is None else use_shape_buffer
    buffer_dist_ft = BUFFER_DIST_FT if buffer_dist_ft is None else buffer_dist_ft
    route_filter = list(ROUTE_FILTER if route_filter is None else route_filter)
    projected_wkid = PROJECTED_CRS_WKID if projected_wkid is None else projected_wkid
    feeder_pattern = FEEDER_NOTES_PATTERN if feeder_pattern is None else feeder_pattern
    write_poi_layer_flag = WRITE_POI_LAYER if write_poi_layer is None else write_poi_layer
    active_as_of = ACTIVE_AS_OF if active_as_of is None else active_as_of
    require_run_log = REQUIRE_RUN_LOG if require_run_log is None else require_run_log

    # Validate eagerly so a typo'd as-of date fails before any output is
    # written — even in prep-only mode, where the filter itself never runs.
    if active_as_of is not None:
        _parse_as_of(active_as_of)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []

    logging.info("Loading shuttle registry from %s", shuttles_csv)
    registry = load_registry_csv(shuttles_csv)
    clean, needs = clean_registry(registry, feeder_pattern)

    clean_path = output_dir / CLEAN_CSV_NAME
    clean.to_csv(clean_path, index=False)
    summary_lines.append(f"  {CLEAN_CSV_NAME}  rows={len(clean)}")
    needs_path = output_dir / NEEDS_GEOCODING_CSV_NAME
    needs.to_csv(needs_path, index=False)
    summary_lines.append(f"  {NEEDS_GEOCODING_CSV_NAME}  rows={len(needs)}")
    if not needs.empty:
        logging.warning(
            "%d registry row(s) need geocoding before they can count toward coverage — see %s.",
            len(needs),
            needs_path,
        )

    if write_poi_layer_flag and not clean.empty:
        poi_path = write_poi_shapefile(clean, output_dir)
        summary_lines.append(f"  {poi_path.name}  features={len(clean)}")
        logging.info(
            "Wrote %s (%d feature(s)) — copy it (with its sidecar files) under "
            "points_of_interest_coverage_arcpy.py's SHP_INPUT_DIR to include private "
            "shuttles in strategic-site coverage.",
            poi_path,
            len(clean),
        )
    elif write_poi_layer_flag:
        logging.warning("No clean rows with coordinates; POI layer not written.")

    coverage: pd.DataFrame | None = None
    if gtfs_dir is None:
        logging.info("No GTFS folder configured; skipping the route coverage rollup.")
    else:
        target_sr = arcpy.SpatialReference(projected_wkid)

        logging.info("Loading GTFS from %s", gtfs_dir)
        tables = _load_gtfs_tables(Path(gtfs_dir), need_shapes=use_shape_buffer)

        logging.info("Building route catchments (use_shape_buffer=%s)", use_shape_buffer)
        route_buffers = _prepare_route_buffers(
            tables,
            use_shape_buffer,
            buffer_dist_ft,
            target_sr,
            route_filter=route_filter,
        )
        if not route_buffers:
            logging.error("No route catchments produced – coverage rollup skipped")
        else:
            sites = clean if active_as_of is None else filter_active_sites(clean, active_as_of)
            shuttle_points = _shuttle_points_in_sr(sites, target_sr)
            coverage = summarize_shuttles_by_route(route_buffers, shuttle_points)
            coverage = _attach_route_short_name(coverage, tables["routes"])
            coverage_path = output_dir / COVERAGE_CSV_NAME
            coverage.to_csv(coverage_path, index=False)
            summary_lines.append(f"  {COVERAGE_CSV_NAME}  rows={len(coverage)}")
            logging.info(
                "Wrote %s (%d routes, %d shuttle sites served, %d feeder sites served)",
                coverage_path,
                len(coverage),
                int(coverage["shuttle_sites_served"].sum()),
                int(coverage["shuttle_feeder_sites_served"].sum()),
            )

    if not write_run_log(output_dir, summary_lines) and require_run_log:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    logging.info("Script completed successfully.")
    return PrepResult(clean=clean, needs_geocoding=needs, coverage=coverage)


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


def stdin_is_interactive() -> bool:
    """Return True when ``input()`` can reach a live user.

    True inside a Jupyter/IPython kernel (ipykernel routes ``input()`` to a
    notebook prompt widget) or when stdin is a real terminal. False under
    captured or redirected stdin — CI runners, orchestrator pipelines, cron —
    where an ``input()`` call would hang or crash rather than guide anyone.
    Scripts use this to decide between prompting for missing configuration
    (guided setup) and failing fast with an exit code.

    Canonical implementation: ``utils/cli_helpers.py``.

    Returns:
        True when prompting a user is possible, False otherwise.
    """
    if "ipykernel" in sys.modules:
        return True
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def prompt_for_path(
    prompt: str,
    *,
    must_exist: bool = True,
    default: Optional[Path] = None,
    allow_skip: bool = False,
) -> Optional[Path]:
    """Ask for one path on stdin, re-asking until the answer is usable.

    Surrounding quotes are stripped so values pasted from Windows Explorer's
    "Copy as path" (which wraps the path in double quotes) work as-is. Blank
    input returns *default* when one is set, returns None when *allow_skip*
    is True, and otherwise re-asks. Only call this after
    :func:`stdin_is_interactive` has confirmed a user is present.

    Canonical implementation: ``utils/cli_helpers.py``.

    Args:
        prompt: Text shown to the user; include any default/skip hint.
        must_exist: Re-ask until the entered path exists on disk (applies to
            typed answers only, never to *default*).
        default: Returned on blank input.
        allow_skip: Blank input returns None instead of re-asking (ignored
            when *default* is set).

    Returns:
        The entered path, or *default* / None per the blank-input rules.

    Raises:
        KeyboardInterrupt: The user cancelled with Ctrl+C.
        EOFError: Stdin closed mid-prompt. Callers should catch both and
            treat them as "user aborted the guided setup".
    """
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            if default is not None:
                return default
            if allow_skip:
                return None
            logging.warning("A path is required here — enter one, or press Ctrl+C to abort.")
            continue
        path = Path(raw)
        if must_exist and not path.exists():
            logging.warning("Path not found: %s — check it and try again.", path)
            continue
        return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block."""
    parser = argparse.ArgumentParser(
        description=(
            "Clean a private-shuttle operator registry, export it as a POI layer, and "
            "(optionally) roll shuttle presence up to GTFS routes. Defaults come from "
            "the CONFIGURATION block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shuttles-csv",
        type=Path,
        default=SHUTTLES_CSV,
        help="The operator registry CSV (company / address / X / Y / notes).",
    )
    parser.add_argument(
        "--gtfs-dir",
        type=Path,
        default=GTFS_DIR,
        help="Folder containing GTFS .txt files; omit to skip the route rollup.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Where the outputs are written."
    )
    parser.add_argument(
        "--buffer-ft", type=float, default=BUFFER_DIST_FT, help="Catchment radius in feet."
    )
    parser.add_argument(
        "--shape-buffer",
        dest="use_shape_buffer",
        action="store_true",
        default=USE_SHAPE_BUFFER,
        help="Buffer route geometry instead of stops (requires shapes.txt).",
    )
    parser.add_argument(
        "--routes",
        nargs="*",
        default=ROUTE_FILTER,
        metavar="ROUTE_ID",
        help="Only analyze these route_id values (empty = all).",
    )
    parser.add_argument(
        "--feeder-pattern",
        default=FEEDER_NOTES_PATTERN,
        help="Case-insensitive regex marking a note as a transit-feeder shuttle.",
    )
    parser.add_argument(
        "--as-of",
        dest="active_as_of",
        default=ACTIVE_AS_OF,
        metavar="DATE",
        help="Count only shuttles active on this date in the route rollup "
        "(e.g. 2026-06-01); omit to count every site.",
    )
    parser.add_argument(
        "--no-poi-layer",
        dest="write_poi_layer",
        action="store_false",
        default=WRITE_POI_LAYER,
        help="Skip writing the Private_Shuttle_Stops.shp POI layer.",
    )
    parser.add_argument(
        "--projected-wkid",
        type=int,
        default=PROJECTED_CRS_WKID,
        help="Projected spatial reference WKID for buffering/joins.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(notebook_safe_argv(argv))


# Literal placeholder input paths shipped in the CONFIGURATION block, frozen
# here (do not edit) so main() can tell an unedited config from a real one. An
# input equal to its placeholder in BOTH the CONFIG constant and the CLI arg
# was customized nowhere. Comparing args against the live CONFIG constants
# instead would always match whenever a flag is omitted (argparse defaults to
# those constants), wrongly blocking the edit-CONFIG-then-run workflow.
_PLACEHOLDER_SHUTTLES_CSV = Path(r"Path\To\Your\private_shuttles.csv")
_PLACEHOLDER_OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")


def _shuttles_csv_is_placeholder(args: argparse.Namespace) -> bool:
    """Return True when the registry CSV was customized nowhere."""
    return (
        Path(args.shuttles_csv) == _PLACEHOLDER_SHUTTLES_CSV
        and Path(SHUTTLES_CSV) == _PLACEHOLDER_SHUTTLES_CSV
    )


def _output_dir_is_placeholder(args: argparse.Namespace) -> bool:
    """Return True when the output folder was customized nowhere."""
    return (
        Path(args.output_dir) == _PLACEHOLDER_OUTPUT_DIR
        and Path(OUTPUT_DIR) == _PLACEHOLDER_OUTPUT_DIR
    )


def _guided_path_setup(args: argparse.Namespace) -> bool:
    """Interactively collect the paths a brand-new user has not configured yet.

    Runs only when :func:`main` detects an unedited configuration in an
    interactive session, so someone who copies the script and hits run is
    walked through the required inputs instead of being handed an exit-code
    warning. Only the values still at their placeholders are asked for.
    Answers apply to this run only; the CONFIGURATION block and CLI flags
    remain the ways to set them permanently.

    Args:
        args: Parsed CLI namespace, updated in place.

    Returns:
        True when every required path was collected; False when the user
        aborted (Ctrl+C, or stdin closed mid-prompt).
    """
    logging.info(
        "Required paths have not been configured — starting guided setup. "
        "Answers apply to this run only; set them permanently in the "
        "CONFIGURATION block or via CLI flags. Press Ctrl+C to abort."
    )
    try:
        if _shuttles_csv_is_placeholder(args):
            args.shuttles_csv = prompt_for_path(
                "Path to the private-shuttle registry CSV (the operator spreadsheet): "
            )
        if _output_dir_is_placeholder(args):
            args.output_dir = prompt_for_path(
                "Output folder (press Enter for 'output' in the current directory): ",
                must_exist=False,
                default=Path("output"),
            )
        if args.gtfs_dir is None:
            args.gtfs_dir = prompt_for_path(
                "GTFS folder for the route rollup (press Enter to skip it): ",
                allow_skip=True,
            )
    except (EOFError, KeyboardInterrupt):
        logging.warning("Guided setup aborted — nothing was run.")
        return False
    logging.info("Outputs will be written to '%s'.", args.output_dir)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the CONFIGURATION block.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders and the run is
        non-interactive (an interactive run prompts for them instead — see
        :func:`_guided_path_setup`) or the user aborted the prompts.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if _shuttles_csv_is_placeholder(args) or _output_dir_is_placeholder(args):
        if not stdin_is_interactive():
            logging.warning(
                "SHUTTLES_CSV and/or OUTPUT_DIR still point at the placeholder paths "
                "from the CONFIGURATION block. Update the CONFIGURATION section, pass "
                "--shuttles-csv / --output-dir, or run interactively (terminal or "
                "notebook) to be prompted for the paths."
            )
            return 2
        if not _guided_path_setup(args):
            return 2
    try:
        run(
            shuttles_csv=args.shuttles_csv,
            gtfs_dir=args.gtfs_dir,
            output_dir=args.output_dir,
            use_shape_buffer=args.use_shape_buffer,
            buffer_dist_ft=args.buffer_ft,
            route_filter=args.routes,
            projected_wkid=args.projected_wkid,
            feeder_pattern=args.feeder_pattern,
            write_poi_layer=args.write_poi_layer,
            active_as_of=args.active_as_of,
        )
    except (FileNotFoundError, ValueError, RuntimeError, arcpy.ExecuteError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the CONFIG block stays in charge.
    raise SystemExit(main())
