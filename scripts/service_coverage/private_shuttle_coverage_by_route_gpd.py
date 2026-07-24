"""Clean a private-shuttle operator registry and roll shuttle presence up to GTFS routes.

Private employer, university, hospital, and residential shuttles overlap public
transit markets, but agencies rarely hold them in analysis-ready form — the
typical source is a hand-maintained spreadsheet of operators with partial
addresses, partial coordinates, blank rows, and free-text notes. This script
turns that registry into three products so planners can see where private
shuttles run and the modeling pipeline can test whether they move ridership:

1. **Clean registry** (``private_shuttles_clean.csv``) — deduplicated,
   whitespace-trimmed rows with usable WGS84 coordinates, plus a ``category``
   column derived from the notes: ``transit_feeder`` (the shuttle connects to
   rail/metro/transit, i.e. likely *complements* fixed-route service),
   ``shuttle`` (a shuttle with no stated transit connection, i.e. a potential
   *competitor*), or ``unspecified``.
2. **Geocoding worklist** (``private_shuttles_needs_geocoding.csv``) — rows
   whose coordinates are missing or invalid (out of range, or the 0,0
   "null island" geocoder artifact). No geocoding is attempted here — the
   script is offline by design; fill these in and re-run.
3. **POI layer** (``Private_Shuttle_Stops.zip``) — the clean rows as a zipped
   point shapefile whose name and id column (``NAME``) match the
   ``("Private_Shuttle_Stops.shp", "NAME")`` entry already listed in
   ``points_of_interest_coverage_gpd.py``'s ``LAYER_SPECS``, so dropping this
   zip anywhere under that script's ``SHP_INPUT_DIR`` wires private shuttles
   into the strategic-site coverage counts with no further configuration.

Optionally (when a GTFS folder is supplied), it also writes
``private_shuttle_coverage_by_route.csv`` — one row per ``route_id`` counting
the shuttle sites inside each route's catchment (``shuttle_sites_served``) and
the transit-feeder subset (``shuttle_feeder_sites_served``). That table is the
modeling hook: registered in ``scripts/modeling/orchestrator_jobs_public.json``,
it joins the ridership anchor by ``route_id`` so the OLS / ML models
(``monthly_ridership_model.py``, ``ridership_ml_model.py``) can estimate
whether private-shuttle presence helps explain route-level ridership.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI flags,
e.g. ``--shuttles-csv``, ``--gtfs-dir``, ``--output-dir``) and run from a shell
or a Jupyter notebook. Without ``--gtfs-dir`` the script runs in prep-only mode
(clean registry + worklist + POI layer, no route rollup).

Assumptions
-----------
- The registry CSV carries the operator name, street address fields, WGS84
  coordinates (X = longitude, Y = latitude), and a free-text notes column;
  column names are configurable and matched case-insensitively.
- The buffer distance is given in feet and converted to meters when the
  projected CRS is metric.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Final, List, Mapping, NamedTuple, Optional, Sequence

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import LineString
from shapely.ops import unary_union

# Path to this file, used to extract the config block for the run log. ``__file__``
# is undefined when the code is pasted into a notebook cell, so a configured
# fallback keeps the run log working there too.
SELF_PATH: Final[Path] = (
    Path(__file__) if "__file__" in globals() else Path("private_shuttle_coverage_by_route_gpd.py")
)

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Top-level paths
SHUTTLES_CSV = Path(r"data/private_shuttles.csv")  # the operator registry CSV
GTFS_DIR: Path | None = None  # GTFS folder; None → skip the route rollup
OUTPUT_DIR = Path(r"output")  # where all outputs are written

# Registry column names, matched case-insensitively against the CSV header.
# X/Y are WGS84 longitude/latitude; both may be absent (all rows then land on
# the geocoding worklist).
COMPANY_COL = "Company"
ADDRESS_COL = "Address"
CITY_COL = "City"
STATE_COL = "State"
ZIP_COL = "Zip"
X_COL = "X"
Y_COL = "Y"
NOTES_COL = "Notes"

# Case-insensitive regex applied to the notes column. A match marks the site a
# ``transit_feeder`` (its shuttle connects to the regional transit network);
# otherwise any mention of "shuttle" yields ``shuttle`` and the rest are
# ``unspecified``.
FEEDER_NOTES_PATTERN = r"metro|rail|station|transit|train"

# POI layer emitted for points_of_interest_coverage_gpd.py. The filename and id
# column must stay in sync with that script's LAYER_SPECS entry
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

# Projected CRS used for buffering and the spatial join.
# EPSG:3857 (Web Mercator) works globally; its latitude-dependent scale
# distortion is corrected automatically when buffering (see
# _web_mercator_ground_scale). Swap for a local CRS (e.g. "EPSG:2283" for
# northern Virginia in feet) when higher spatial accuracy is needed.
PROJECTED_CRS = "EPSG:3857"

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
) -> pd.DataFrame:
    """Read the operator registry CSV into canonically named string columns.

    Every configured column is matched case-insensitively against the file's
    header. Missing optional columns (everything except the company column)
    are created empty, so a registry that was never geocoded still loads.

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

    Returns:
        DataFrame with string columns ``company``, ``address``, ``city``,
        ``state``, ``zip``, ``notes``, ``lon_raw``, ``lat_raw``.

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
    }
    out = pd.DataFrame(index=raw.index)
    for target, source in wanted.items():
        out[target] = raw[source] if source is not None else ""
        if source is None and target not in {"lon_raw", "lat_raw"}:
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
        5. Categorize the notes (see :func:`categorize_notes`).

    Args:
        registry: Output of :func:`load_registry_csv`.
        feeder_pattern: Passed through to :func:`categorize_notes`.

    Returns:
        ``(clean, needs_geocoding)``. ``clean`` carries the text columns plus
        numeric ``lon`` / ``lat`` and ``category``; ``needs_geocoding`` carries
        the text columns, ``category``, and a ``reason`` column
        (``missing_coordinates`` or ``invalid_coordinates``).
    """
    df = registry.copy()
    for col in (*_TEXT_COLUMNS, "lon_raw", "lat_raw"):
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

    df = df.assign(category=categorize_notes(df["notes"], feeder_pattern))

    clean = df.loc[usable, [*_TEXT_COLUMNS, "category"]].copy()
    clean.insert(_TEXT_COLUMNS.index("notes"), "lon", lon[usable])
    clean.insert(_TEXT_COLUMNS.index("notes") + 1, "lat", lat[usable])

    needs = df.loc[~usable, [*_TEXT_COLUMNS, "category"]].copy()
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


def build_poi_layer(clean: pd.DataFrame, id_column: str = POI_ID_COLUMN) -> gpd.GeoDataFrame:
    """Build the WGS84 point layer for the strategic-site coverage tool.

    Args:
        clean: The clean registry rows (must carry ``company``/``lon``/``lat``).
        id_column: Attribute column name expected by the coverage tool's
            LAYER_SPECS (``NAME`` for Private_Shuttle_Stops.shp).

    Returns:
        GeoDataFrame with columns ``[id_column, "geometry"]`` in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {id_column: clean["company"].to_numpy()},
        geometry=gpd.points_from_xy(clean["lon"], clean["lat"]),
        crs="EPSG:4326",
    )


def write_layer_zip(gdf: gpd.GeoDataFrame, out_zip: Path) -> None:
    """Write *gdf* as a zipped shapefile with components at the archive's top level.

    The zipped form is what government open-data portals deliver and what
    ``points_of_interest_coverage_gpd.py`` discovers via its ``zip://`` search,
    so one file can be dropped straight into that script's ``SHP_INPUT_DIR``.

    Args:
        gdf: The point layer to write.
        out_zip: Destination ``.zip`` path; the shapefile inside reuses its stem.
    """
    stem = out_zip.with_suffix("").name
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        gdf.to_file(tmp_dir / f"{stem}.shp", driver="ESRI Shapefile", index=False)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for comp in sorted(tmp_dir.glob(f"{stem}.*")):
                zf.write(comp, comp.name)


# =============================================================================
# ROUTE CATCHMENTS (mirrors school_coverage_by_route_gpd.py)
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

    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        path = gtfs_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[name] = pd.read_csv(path, dtype={"route_id": str})
        logging.debug("Loaded %s (%d rows)", name, len(tables[name]))
    return tables


def _prepare_route_buffers(
    tables: Mapping[str, pd.DataFrame],
    use_shape_buffer: bool,
    buffer_dist_ft: float,
    route_filter: list[str] | None = None,
    projected_crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with one buffered catchment geometry per route_id.

    Depending on *use_shape_buffer*, the buffer is built around the union of
    (a) the route's shape(s) or (b) all of its stops. The buffer distance is
    given in feet and converted to meters when *projected_crs* is metric.

    Args:
        tables: GTFS tables from :func:`_load_gtfs_tables`.
        use_shape_buffer: Buffer route geometry when True, else buffer stops.
        buffer_dist_ft: Catchment radius in feet.
        route_filter: Optional list of route_id values to keep (empty = all).
        projected_crs: Projected CRS used for buffering.

    Raises:
        ValueError: If shape-buffer mode is requested but trips.txt lacks a
            ``shape_id`` column or shapes.txt is malformed.
    """
    # shape_id is optional in GTFS and only needed to buffer route geometry,
    # so stop-buffer mode must not require the column.
    trip_cols = ["route_id", "trip_id"]
    if use_shape_buffer:
        if "shape_id" not in tables["trips"].columns:
            raise ValueError("trips.txt missing shape_id column (required for shape-buffer mode)")
        trip_cols.append("shape_id")
    trips = tables["trips"][trip_cols].copy()
    trips["route_id"] = trips["route_id"].astype(str)

    # Stops GeoDataFrame (always built; the default catchment buffers stops).
    stops = tables["stops"][["stop_id", "stop_lat", "stop_lon"]].copy()
    stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    ).to_crs(projected_crs)

    shapes_gdf: gpd.GeoDataFrame | None = None
    route_shapes: pd.Series | None = None
    if use_shape_buffer:
        shapes_df = tables["shapes"]
        if {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.difference(
            shapes_df.columns
        ):
            raise ValueError("shapes.txt missing required columns")
        shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
        lines = (
            shapes_df.groupby("shape_id")
            .apply(
                lambda grp: LineString(grp[["shape_pt_lon", "shape_pt_lat"]].to_numpy(dtype=float))
            )
            .to_frame(name="geometry")
        )
        shapes_gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(
            projected_crs
        )
        route_shapes = (
            trips.drop_duplicates(subset=["route_id", "shape_id"])
            .groupby("route_id")["shape_id"]
            .apply(list)
        )

    # ¼ mile etc. is configured in feet; reproject-aware conversion to meters.
    buff_dist = buffer_dist_ft
    if _crs_is_metric(stops.crs):
        buff_dist = buffer_dist_ft * 0.3048
    buff_dist *= _web_mercator_ground_scale(stops.crs, tables["stops"]["stop_lat"])

    route_ids = route_shapes.index if route_shapes is not None else trips["route_id"].unique()

    buffers: List[dict[str, object]] = []
    for route_id in route_ids:
        route_id = str(route_id)
        if route_filter and route_id not in route_filter:
            continue

        if use_shape_buffer and shapes_gdf is not None and route_shapes is not None:
            shp_ids = [s for s in route_shapes.loc[route_id] if s in shapes_gdf.index]
            geoms = shapes_gdf.loc[shp_ids, "geometry"]
        else:
            trip_stops = (
                tables["stop_times"]
                .merge(trips[trips.route_id == route_id][["trip_id"]], on="trip_id", how="inner")[
                    "stop_id"
                ]
                .unique()
            )
            geoms = stops[stops.stop_id.isin(trip_stops)].geometry

        if geoms.empty:
            logging.warning("No geometry for route %s – skipped", route_id)
            continue

        buf = unary_union(list(geoms)).buffer(buff_dist)
        buffers.append({"route_id": route_id, "geometry": buf})

    return gpd.GeoDataFrame(buffers, geometry="geometry", crs=projected_crs)


def _crs_is_metric(crs: object) -> bool:
    """Return True when *crs* measures distance in meters (best-effort)."""
    try:
        unit = crs.axis_info[0].unit_name.lower()  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        return True  # default GTFS reprojection target (3857) is metric
    return "metre" in unit or "meter" in unit


def _web_mercator_ground_scale(crs: object, latitudes: pd.Series) -> float:
    """Return the Web Mercator map-per-ground distance factor, or 1.0 elsewhere.

    Web Mercator (EPSG:3857) inflates distances by roughly ``1/cos(latitude)``:
    at 39°N a true quarter mile spans ~1.29x as many map "meters", so a buffer
    drawn in raw map units under-covers the ground by the same factor. Buffer
    distances are multiplied by this factor so they span a true ground distance
    at the analysis area's mean latitude. Any other CRS returns 1.0 (projected
    local CRSs are treated as true-scale).

    Args:
        crs: The projected CRS in use (anything pyproj can parse).
        latitudes: WGS 84 latitudes of the analysis features; their mean sets
            the correction.

    Returns:
        The multiplier to apply to a ground distance before buffering.
    """
    try:
        epsg = CRS.from_user_input(crs).to_epsg()
    except CRSError:
        return 1.0
    if epsg != 3857:
        return 1.0
    lat = pd.to_numeric(latitudes, errors="coerce").dropna()
    if lat.empty:
        return 1.0
    mean_lat = float(lat.mean())
    if not -89.0 < mean_lat < 89.0:
        return 1.0
    scale = 1.0 / math.cos(math.radians(mean_lat))
    logging.info(
        "Web Mercator inflates distances by %.4f at latitude %.3f; scaling the "
        "buffer to preserve ground distance.",
        scale,
        mean_lat,
    )
    return scale


# =============================================================================
# ROUTE ROLLUP
# =============================================================================


def summarize_shuttles_by_route(
    route_buffers: gpd.GeoDataFrame,
    shuttles_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Count shuttle sites (total and transit feeders) per route catchment.

    Args:
        route_buffers: One buffered catchment per route_id (from
            :func:`_prepare_route_buffers`).
        shuttles_gdf: Clean shuttle points carrying a ``category`` column, in
            the same CRS as *route_buffers*.

    Returns:
        DataFrame with one row per route_id and integer columns
        ``shuttle_sites_served`` and ``shuttle_feeder_sites_served``. Routes
        with no shuttle sites nearby report zeros.
    """
    routes = route_buffers[["route_id"]].drop_duplicates().reset_index(drop=True)

    if not shuttles_gdf.empty:
        joined = gpd.sjoin(
            shuttles_gdf,
            route_buffers[["route_id", "geometry"]],
            predicate="intersects",
            how="inner",
        )
        joined["is_feeder"] = joined["category"] == CATEGORY_TRANSIT_FEEDER
        grouped = joined.groupby("route_id").agg(
            shuttle_sites_served=("is_feeder", "size"),
            shuttle_feeder_sites_served=("is_feeder", "sum"),
        )
        summary = routes.merge(grouped, on="route_id", how="left")
    else:
        summary = routes.assign(shuttle_sites_served=0, shuttle_feeder_sites_served=0)

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
    projected_crs: str | None = None,
    feeder_pattern: str | None = None,
    write_poi_layer: bool | None = None,
    require_run_log: bool | None = None,
) -> PrepResult:
    """Clean the registry, export the POI layer, and (optionally) roll up by route.

    Unset args fall back to the CONFIGURATION block, so ``m.SHUTTLES_CSV = ...;
    m.run()`` works after a plain import. The route rollup only runs when a
    GTFS folder is configured (``GTFS_DIR`` or ``--gtfs-dir``); without one the
    script is a pure registry-prep step.

    Returns:
        A :class:`PrepResult` with the clean registry, the geocoding worklist,
        and the route coverage table (None in prep-only mode).

    Raises:
        RuntimeError: If the run-log sidecar cannot be written while
            ``REQUIRE_RUN_LOG`` is enabled.
    """
    shuttles_csv = Path(SHUTTLES_CSV if shuttles_csv is None else shuttles_csv)
    gtfs_dir = GTFS_DIR if gtfs_dir is None else Path(gtfs_dir)
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    use_shape_buffer = USE_SHAPE_BUFFER if use_shape_buffer is None else use_shape_buffer
    buffer_dist_ft = BUFFER_DIST_FT if buffer_dist_ft is None else buffer_dist_ft
    route_filter = list(ROUTE_FILTER if route_filter is None else route_filter)
    projected_crs = PROJECTED_CRS if projected_crs is None else projected_crs
    feeder_pattern = FEEDER_NOTES_PATTERN if feeder_pattern is None else feeder_pattern
    write_poi_layer_flag = WRITE_POI_LAYER if write_poi_layer is None else write_poi_layer
    require_run_log = REQUIRE_RUN_LOG if require_run_log is None else require_run_log

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
        poi_zip = output_dir / f"{Path(POI_LAYER_FILENAME).stem}.zip"
        write_layer_zip(build_poi_layer(clean), poi_zip)
        summary_lines.append(f"  {poi_zip.name}  features={len(clean)}")
        logging.info(
            "Wrote %s (%d feature(s)) — drop it under points_of_interest_coverage_gpd.py's "
            "SHP_INPUT_DIR to include private shuttles in strategic-site coverage.",
            poi_zip,
            len(clean),
        )
    elif write_poi_layer_flag:
        logging.warning("No clean rows with coordinates; POI layer not written.")

    coverage: pd.DataFrame | None = None
    if gtfs_dir is None:
        logging.info("No GTFS folder configured; skipping the route coverage rollup.")
    else:
        logging.info("Loading GTFS from %s", gtfs_dir)
        tables = _load_gtfs_tables(Path(gtfs_dir), need_shapes=use_shape_buffer)

        logging.info("Building route catchments (use_shape_buffer=%s)", use_shape_buffer)
        route_buffers = _prepare_route_buffers(
            tables,
            use_shape_buffer,
            buffer_dist_ft,
            route_filter=route_filter,
            projected_crs=projected_crs,
        )
        if route_buffers.empty:
            logging.error("No route catchments produced – coverage rollup skipped")
        else:
            shuttles_gdf = gpd.GeoDataFrame(
                clean[["company", "category"]].copy(),
                geometry=gpd.points_from_xy(clean["lon"], clean["lat"]),
                crs="EPSG:4326",
            ).to_crs(projected_crs)
            coverage = summarize_shuttles_by_route(route_buffers, shuttles_gdf)
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
        "--no-poi-layer",
        dest="write_poi_layer",
        action="store_false",
        default=WRITE_POI_LAYER,
        help="Skip writing the Private_Shuttle_Stops.zip POI layer.",
    )
    parser.add_argument(
        "--projected-crs", default=PROJECTED_CRS, help="Projected CRS for buffering/joins."
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    return parser.parse_args(notebook_safe_argv(argv))


# Literal placeholder input path shipped in the CONFIGURATION block, frozen
# here (do not edit) so main() can tell an unedited config from a real one. An
# input equal to its placeholder in BOTH the CONFIG constant and the CLI arg
# was customized nowhere. Comparing args against the live CONFIG constants
# instead would always match whenever a flag is omitted (argparse defaults to
# those constants), wrongly blocking the edit-CONFIG-then-run workflow.
_PLACEHOLDER_SHUTTLES_CSV = Path(r"data/private_shuttles.csv")


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
    if (
        Path(args.shuttles_csv) == _PLACEHOLDER_SHUTTLES_CSV
        and Path(SHUTTLES_CSV) == _PLACEHOLDER_SHUTTLES_CSV
    ):
        logging.warning(
            "SHUTTLES_CSV still points at the placeholder path from the CONFIGURATION "
            "block. Update the CONFIGURATION section or pass --shuttles-csv before running."
        )
        return 2
    try:
        run(
            shuttles_csv=args.shuttles_csv,
            gtfs_dir=args.gtfs_dir,
            output_dir=args.output_dir,
            use_shape_buffer=args.use_shape_buffer,
            buffer_dist_ft=args.buffer_ft,
            route_filter=args.routes,
            projected_crs=args.projected_crs,
            feeder_pattern=args.feeder_pattern,
            write_poi_layer=args.write_poi_layer,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the CONFIG block stays in charge.
    raise SystemExit(main())
