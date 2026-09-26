"""Compute service-area demographics from transit stops.

Builds stop buffers (from GTFS or a point FC), dissolves to service areas,
clips a demographics layer, and produces area-weighted counts for equity and
employment metrics. Supports whole-network and per-route runs and optional CSV
and feature exports.

Pipeline:
  1) Create stops layer (GTFS filter by service_id and route_short_name, or
     shapefile).
  2) Buffer and dissolve (optional processing CRS for stability).
  3) Clip demographics (SR alignment, PairwiseClip with fallbacks).
  4) Add areas (`area_ac_cl`, `area_perc`) and synthetic counts: area-weighted
     block counts (`HH_LOWINC`, `MINOR_CNT`, ...) that uscensus_tiger_join_arcpy
     split from tract totals by block population/households (marked by its
     `CNT_ALLOC` field). A layer without that marker predates the split — its
     `*_CNT` fields repeat whole-tract totals — so it falls back, with a warning,
     to `PCT_* × *_TOT` (tract rate × block total).
  5) Summarize totals and export.

GTFS trips are limited to the service day(s) in `SERVICE_IDS_TO_INCLUDE` (GTFS
`service_id`; default `"4"`). Each run logs `calendar.txt` in full so the
right id can be checked or picked, warns while the default is unchanged, and
stops with exit code 2 if a listed service_id has no trips in the feed. A
per-route run skips, with a warning, a route with no trips on that day.

Inputs:
  - GTFS: stops.txt, routes.txt, trips.txt, stop_times.txt (calendar.txt, if
    present, is logged)
  - Demographics FC with HH_LOWINC/PCT_LOWINC/HH_TOT, MINOR_CNT/PCT_MINOR/POP_TOT,
    EMP_LO/EMP_TOT; LEP/YOUTH/ELDER optional.

Outputs:
  - Clipped polygons with area fields and synthetic metrics:
    loinc_hh, total_hh, minor_pop, total_pop, loinc_jobs, all_jobs (+ extras).
  - Final FC/shapefile; optional per-route CSV.

Typical usage:
  Update the paths in the CONFIGURATION section and run from a shell, ArcGIS
  Pro's Python window, or a Jupyter notebook (requires ArcGIS Pro's `arcpy`).
"""

from __future__ import annotations

import logging
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import arcpy
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# Overwrite behavior
OVERWRITE_OUTPUTS: bool = True

# --- Stops input mode ---
# Choose one: "shapefile" or "gtfs"
STOPS_INPUT_MODE: str = "gtfs"

# For "shapefile" mode
STOPS_FEATURE_CLASS: str = r""

# For "gtfs" mode
GTFS_FOLDER: str = r"Path\To\Your\GTFS_Folder"

# Optional: filter GTFS to the following route_short_name values.
# Example: ["101", "202"]. Leave as [] or None to include all routes (network run).
GTFS_ROUTE_SHORT_NAMES: Optional[Sequence[str]] = ["101", "202"]

# Service day(s) to analyze, by GTFS service_id ("gtfs" mode only). The default "4" is
# the typical weekday service_id in the source agency's GTFS; other feeds use other
# ids. Each run logs calendar.txt so you can check or pick, warns while this is still
# the default, and stops (exit code 2) if a listed service_id has no trips. For
# Saturday or Sunday, rerun with that day's service_id(s) and its own RUN_TAG.
# Set [] to use every trip (all days combined).
SERVICE_IDS_TO_INCLUDE: Sequence[str] = ["4"]

# Input demographics (from the census-join pipeline).
# This may be a FileGDB feature class or a shapefile; both work.
DEMOGRAPHICS_FC: str = (
    # "File/Path/To/Your/output_final/scratch_join.gdb/joined_blocks"
    "File/Path/To/Your/output_final/blocks_with_attrs.shp"
)

# Optional disk outputs for intermediates (used only for the "network" run).
# If left as empty strings, intermediates are kept in_memory.
BUFFERED_STOPS_OUT: str = r""
DISSOLVED_BUFFERS_OUT: str = r""
CLIPPED_DEMOGRAPHICS_OUT: str = r""

# Final export target:
#   If this ends with ".gdb", results are written as a feature class into that GDB.
#   Otherwise it's treated as a directory and a shapefile is written there.
FINAL_EXPORT_TARGET: str = r"File\Path\To\Your\output_final\output.gdb"

# Processing parameters
BUFFER_DISTANCE_MILES: float = 0.25

# Optional: project just the *processing* into a local planar CRS (WKID),
# while area computations remain geodesic. Leave as None to skip reprojection.
PROCESS_CRS_WKID: Optional[int] = None

# Run label used for final export naming and console prints
RUN_TAG: str = "tsp_service"  # e.g., "weekday_2024q3"

# -----------------------------------------------------------------------------
# OUTPUT MODE
# -----------------------------------------------------------------------------
# One of: "network" | "by_route" | "both"
OUTPUT_MODE: str = "by_route"

# Per-route runs are supported only when STOPS_INPUT_MODE == "gtfs".
# When True, also write a CSV of per-route results to FINAL_EXPORT_TARGET (folder or peer to gdb).
BY_ROUTE_WRITE_CSV: bool = True

# Optional: export per-route clipped polygons to FINAL_EXPORT_TARGET (gdb or folder).
BY_ROUTE_EXPORT_FEATURES: bool = False

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# PREFERRED DEMOGRAPHIC FIELD NAMES (from the census-join pipeline)
# =============================================================================


class _Pref(NamedTuple):
    """Preference tuple for sourcing a metric: count, percent, total."""

    count: str | None
    pct: str | None
    total: str | None


# What we want to produce in the clipped layer → how to source it: the block count
# when the layer's counts were split to blocks (COUNT_ALLOCATION_FIELD present),
# otherwise percent * total.
_PREFS: dict[str, _Pref] = {
    # Households
    "loinc_hh": _Pref(count="HH_LOWINC", pct="PCT_LOWINC", total="HH_TOT"),
    "total_hh": _Pref(count="HH_TOT", pct=None, total=None),
    # Population
    "minor_pop": _Pref(count="MINOR_CNT", pct="PCT_MINOR", total="POP_TOT"),
    "total_pop": _Pref(count="POP_TOT", pct=None, total=None),
    # Jobs (LODES)
    "loinc_jobs": _Pref(count="EMP_LO", pct=None, total=None),
    "all_jobs": _Pref(count="EMP_TOT", pct=None, total=None),
    # Optional extras (auto-included if present/derivable)
    "lep_cnt": _Pref(count="LEP_CNT", pct="PCT_LEP", total="POP_TOT"),
    "youth_cnt": _Pref(count="YOUTH_CNT", pct="PCT_YOUTH", total="POP_TOT"),
    "elder_cnt": _Pref(count="ELDER_CNT", pct="PCT_ELDER", total="POP_TOT"),
}


# Marker field written by uscensus_tiger_join_arcpy (value 1) when its *_CNT fields
# hold counts split from tract totals to blocks, rather than whole-tract figures.
COUNT_ALLOCATION_FIELD: str = "CNT_ALLOC"
_LEGACY_LAYERS_WARNED: set[str] = set()  # warn once per layer, not once per route


@dataclass(frozen=True)
class DemogSchema:
    """Resolved strategies for computing metrics from a demographics layer."""

    outputs: tuple[str, ...]
    # strategy:
    #   ("count", "FIELD") or ("derived", ("PCT_FIELD","TOTAL_FIELD"))
    strategies: dict[str, tuple[str, str | tuple[str, str]]]
    resolved_inputs: Dict[str, str]  # mapping of canonical→resolved name (for diagnostics)
    # True when the layer's *_CNT fields are block-allocated counts (marker present).
    counts_allocated: bool = False


# =============================================================================
# UTILITIES
# =============================================================================
def _as_distance_miles(miles: float) -> str:
    """Return an ArcPy-friendly distance string in miles."""
    return f"{miles} Miles"


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def field_exists(feature_class: str, field_name: str) -> bool:
    """Return True if a field exists in the feature class."""
    return any(f.name.lower() == field_name.lower() for f in arcpy.ListFields(feature_class))


def safe_add_field(
    feature_class: str,
    field_name: str,
    field_type: str,
    **kwargs: Any,
) -> None:
    """Add a field if it does not already exist."""
    if not field_exists(feature_class, field_name):
        arcpy.management.AddField(feature_class, field_name, field_type, **kwargs)


def calc_field(feature_class: str, field_name: str, expression: str) -> None:
    """Calculate a field using PYTHON3 expression type."""
    arcpy.management.CalculateField(
        feature_class,
        field_name,
        expression,
        expression_type="PYTHON3",
    )


def _sanitize_name(token: str) -> str:
    """Return a safe token for in_memory names and filenames."""
    return "".join(ch for ch in str(token) if ch.isalnum() or ch in ("_", "-"))[:40]


def _route_scoped_temp(name: str, scope: str) -> str:
    """Build an in_memory path unique per scope (e.g., route short name)."""
    safe = _sanitize_name(scope)
    return rf"in_memory\{name}_{safe}"


def _resolve_field(dataset: str, desired: str) -> Optional[str]:
    """Resolve a logical field name to the actual field on *dataset*.

    Strategies (in order):
      1) Exact match (case-insensitive).
      2) Endswith match (case-insensitive), e.g., 'attrs_csv_POP_TOT' for 'POP_TOT'.
      3) Endswith match ignoring prefixes before the last underscore in the dataset.
    """
    desired_l = desired.lower()
    fields = arcpy.ListFields(dataset)
    # 1) Exact (case-insensitive)
    for f in fields:
        if f.name.lower() == desired_l:
            return f.name
    # 2) Endswith w/ case-insensitive
    for f in fields:
        if f.name.lower().endswith(desired_l):
            return f.name
    # 3) Try to strip to tail in dataset names and compare
    for f in fields:
        tail = f.name.split("_")[-1].lower()
        if tail == desired_l:
            return f.name
    return None


def _resolve_many(dataset: str, names: Iterable[str]) -> Dict[str, Optional[str]]:
    """Resolve many desired names → actual names on dataset."""
    out: Dict[str, Optional[str]] = {}
    for n in names:
        out[n] = _resolve_field(dataset, n)
    return out


# =============================================================================
# GTFS LOADING
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


# -----------------------------------------------------------------------------
# SERVICE-DAY SELECTION
#
# Copied verbatim from utils/calendar_helpers.py (the canonical versions) so this
# script stays self-contained. Keep the copies in sync when updating either.
# -----------------------------------------------------------------------------
class ServiceSelectionError(ValueError):
    """A requested GTFS service_id is not used by any trip in the feed.

    A configuration problem rather than a crash: callers report it without a
    traceback and exit with code 2.
    """


def log_service_calendar(calendar_df: Optional[pd.DataFrame]) -> None:
    """Log calendar.txt in full, so the user can see which service_id runs on which days.

    calendar.txt is usually a handful of rows; printing it lets the user check or
    pick the service_id(s) to analyze without opening the feed.

    Args:
        calendar_df: Parsed ``calendar.txt``, or ``None`` when the feed has none.
    """
    if calendar_df is None or calendar_df.empty:
        logging.info("calendar.txt is absent or empty; see trips.txt for the feed's service_ids.")
        return
    logging.info(
        "calendar.txt (%d service_id row(s)):\n%s",
        len(calendar_df),
        calendar_df.to_string(index=False),
    )


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


# Shipped default for SERVICE_IDS_TO_INCLUDE, so an unedited value can be flagged.
_DEFAULT_SERVICE_IDS: Sequence[str] = ["4"]


class RoutesNotInServiceError(ServiceSelectionError):
    """The selected routes run no trips on the selected service day(s).

    A per-route run skips such a route with a warning; for the whole run it is a
    configuration error, like its parent.
    """


def _check_service_selection(gtfs_folder: str, service_ids: Sequence[str]) -> None:
    """Log calendar.txt and confirm that every requested service_id has trips.

    Runs once, before any geoprocessing, so a wrong service_id stops the run early.

    Raises:
        ServiceSelectionError: A requested service_id is used by no trip.
    """
    trips = load_gtfs_data(gtfs_folder, files=("trips.txt",), dtype=str)["trips"]
    calendar: Optional[pd.DataFrame] = None
    try:
        calendar = load_gtfs_data(gtfs_folder, files=("calendar.txt",), dtype=str)["calendar"]
    except (OSError, ValueError):
        pass  # optional: calendar_dates.txt alone can define every service day
    log_service_calendar(calendar)
    select_trips_by_service(trips, service_ids, default_service_ids=_DEFAULT_SERVICE_IDS)


def _filter_gtfs_stops_by_route_short_name(
    gtfs: dict[str, pd.DataFrame],
    route_short_names: Optional[Sequence[str]],
    service_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return stops DataFrame filtered to those used by routes with given short names.

    Only trips on *service_ids* count; empty or ``None`` keeps every trip. With
    neither filter set, every stop in stops.txt is returned.

    Raises:
        RoutesNotInServiceError: The selected routes run no trips on *service_ids*.
    """
    stops = gtfs["stops"]
    target = {str(x) for x in route_short_names or ()}
    service = [str(s).strip() for s in service_ids or () if str(s).strip()]
    if not target and not service:
        return stops.copy()

    routes = gtfs["routes"]
    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]

    if target:
        routes_sel = routes[routes["route_short_name"].astype(str).isin(target)]
        if routes_sel.empty:
            raise ValueError(
                f"No routes matched the provided route_short_name filter: {sorted(target)}"
            )

        route_ids = set(routes_sel["route_id"].astype(str))
        trips = trips[trips["route_id"].astype(str).isin(route_ids)]
        if trips.empty:
            raise ValueError("Routes matched, but no trips found for the selected routes.")

    if service:
        trips = trips[trips["service_id"].astype(str).str.strip().isin(service)]
        if trips.empty:
            raise RoutesNotInServiceError(
                f"Route(s) {sorted(target)} run no trips on service_id(s) {service}."
            )

    trip_ids = set(trips["trip_id"].astype(str))
    st_sel = stop_times[stop_times["trip_id"].astype(str).isin(trip_ids)]
    if st_sel.empty:
        raise ValueError("Trips matched, but no stop_times rows found.")

    stop_ids = set(st_sel["stop_id"].astype(str))
    out = stops[stops["stop_id"].astype(str).isin(stop_ids)].copy()
    if out.empty:
        raise ValueError("Filtering produced zero stops. Verify stop_times and stops tables.")
    return out


def _points_layer_from_stops_df(stops_df: pd.DataFrame, layer_name: str = "gtfs_stops") -> str:
    """Create an in-memory point feature layer from a GTFS stops DataFrame.

    Expects columns: stop_id, stop_name, stop_lat, stop_lon.
    Uses a strict NumPy structured array (no 'object' dtypes).
    """
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required.difference(stops_df.columns)
    if missing:
        raise ValueError(f"GTFS stops missing required columns: {sorted(missing)}")

    df = stops_df.loc[:, ["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()

    # Coerce coordinates to float; drop rows with invalid coords.
    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"])

    # Normalize text columns to strings and clamp to safe lengths.
    max_id_len = 64
    max_name_len = 255
    df["stop_id"] = df["stop_id"].astype(str).str.slice(0, max_id_len)
    df["stop_name"] = df["stop_name"].astype(str).str.slice(0, max_name_len)

    # Structured array (no 'object' dtype)
    dtype = np.dtype(
        [
            ("stop_id", f"<U{max_id_len}"),
            ("stop_name", f"<U{max_name_len}"),
            ("stop_lat", "<f8"),
            ("stop_lon", "<f8"),
        ]
    )
    rec = np.empty(len(df), dtype=dtype)
    rec["stop_id"] = df["stop_id"].to_numpy()
    rec["stop_name"] = df["stop_name"].to_numpy()
    rec["stop_lat"] = df["stop_lat"].to_numpy(dtype=np.float64)
    rec["stop_lon"] = df["stop_lon"].to_numpy(dtype=np.float64)

    # Write to in_memory, then convert XY to points.
    tbl_path = r"in_memory\gtfs_stops_tbl"
    pt_path = r"in_memory\gtfs_stops_points"

    if arcpy.Exists(tbl_path):
        arcpy.management.Delete(tbl_path)
    if arcpy.Exists(pt_path):
        arcpy.management.Delete(pt_path)

    arcpy.da.NumPyArrayToTable(rec, tbl_path)

    wgs84 = arcpy.SpatialReference(4326)
    arcpy.management.XYTableToPoint(
        tbl_path,
        pt_path,
        x_field="stop_lon",
        y_field="stop_lat",
        coordinate_system=wgs84,
    )

    lyr_result = arcpy.management.MakeFeatureLayer(pt_path, layer_name)
    return str(lyr_result)


def make_stops_layer(
    mode: str,
    shapefile_fc: Optional[str] = None,
    gtfs_folder: Optional[str] = None,
    route_short_names: Optional[Sequence[str]] = None,
    service_ids: Optional[Sequence[str]] = None,
    layer_name: str = "stops",
) -> str:
    """Create a feature layer of stops for downstream processing.

    In "gtfs" mode, keeps the stops served by *route_short_names* on *service_ids*
    (either empty or ``None``: no filter on it).
    """
    m = mode.strip().lower()
    if m not in {"shapefile", "gtfs"}:
        raise ValueError("STOPS_INPUT_MODE must be 'shapefile' or 'gtfs'.")

    if m == "shapefile":
        if not shapefile_fc or not arcpy.Exists(shapefile_fc):
            raise FileNotFoundError(f"Stops feature class not found: {shapefile_fc}")
        return str(arcpy.management.MakeFeatureLayer(shapefile_fc, layer_name))

    # GTFS mode
    if not gtfs_folder or not os.path.exists(gtfs_folder):
        raise FileNotFoundError(f"GTFS folder not found: {gtfs_folder}")

    gtfs = load_gtfs_data(
        gtfs_folder,
        files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt"),
        dtype=str,
    )
    stops_df = _filter_gtfs_stops_by_route_short_name(gtfs, route_short_names, service_ids)
    return _points_layer_from_stops_df(stops_df, layer_name=layer_name)


# =============================================================================
# DEMOGRAPHIC SCHEMA DETECTION & AREA-WEIGHTED METRICS
# =============================================================================
def _has(dataset: str, field: Optional[str]) -> bool:
    return bool(field) and field_exists(dataset, field or "")


def detect_demog_schema(demographics_fc: str) -> DemogSchema:
    """Inspect the demographics layer and decide how to compute each metric.

    Layers from the current uscensus_tiger_join_arcpy carry ``COUNT_ALLOCATION_FIELD``:
    their ``*_CNT`` fields are tract counts split to blocks by block population or
    households, the same count allocation uscensus_tiger_join_gpd uses. For those:
      1) Use the direct count field if present (e.g., MINOR_CNT).
      2) Otherwise derive from percent + total (e.g., PCT_MINOR × POP_TOT).

    Older layers lack the marker; their ``*_CNT`` fields repeat each tract's whole
    total on every block, so the order flips (derive first, count as a last resort)
    and a warning asks for the join to be re-run. Metrics with neither are skipped.
    """
    strategies: dict[str, tuple[str, str | tuple[str, str]]] = {}
    outputs: list[str] = []
    resolved_inputs: Dict[str, str] = {}
    counts_allocated = _resolve_field(demographics_fc, COUNT_ALLOCATION_FIELD) is not None
    if not counts_allocated and demographics_fc not in _LEGACY_LAYERS_WARNED:
        _LEGACY_LAYERS_WARNED.add(demographics_fc)
        logging.warning(
            "Demographics layer '%s' has no %s field: it predates the block count split in "
            "uscensus_tiger_join_arcpy, so its *_CNT fields are whole-tract totals. Using "
            "PCT_* x block totals instead (tract rate x 2020 block population/households). "
            "Re-run uscensus_tiger_join_arcpy to use block-allocated counts.",
            demographics_fc,
            COUNT_ALLOCATION_FIELD,
        )

    # Pre-resolve all canonical names that might be referenced
    all_needed: set[str] = set()
    for pref in _PREFS.values():
        for cand in (pref.count, pref.pct, pref.total):
            if cand:
                all_needed.add(cand)
    resolved = _resolve_many(demographics_fc, sorted(all_needed))

    for out_name, pref in _PREFS.items():
        rcount = resolved.get(pref.count) if pref.count else None
        rpct = resolved.get(pref.pct) if pref.pct else None
        rtot = resolved.get(pref.total) if pref.total else None
        count_ok = bool(rcount) and _has(demographics_fc, rcount)
        derived_ok = (
            bool(rpct and rtot) and _has(demographics_fc, rpct) and _has(demographics_fc, rtot)
        )
        # Allocated layers: count first. Legacy layers: derived first, count last.
        if count_ok and (counts_allocated or not derived_ok):
            strategies[out_name] = ("count", str(rcount))
            resolved_inputs[str(pref.count)] = str(rcount)
        elif derived_ok:
            strategies[out_name] = ("derived", (str(rpct), str(rtot)))
            resolved_inputs[str(pref.pct)] = str(rpct)
            resolved_inputs[str(pref.total)] = str(rtot)
        else:
            continue
        outputs.append(out_name)

    return DemogSchema(
        outputs=tuple(outputs),
        strategies=strategies,
        resolved_inputs=resolved_inputs,
        counts_allocated=counts_allocated,
    )


def add_original_area_acres(demographics_fc: str, field_name: str = "area_ac_og") -> None:
    """Add original area (acres) to demographic polygons (idempotent)."""
    safe_add_field(demographics_fc, field_name, "DOUBLE")
    calc_field(demographics_fc, field_name, "!shape.area@SQUAREMETERS! / 4046.86")


def add_clipped_area_and_percentage(
    clipped_fc: str,
    area_field: str = "area_ac_cl",
    pct_field: str = "area_perc",
    original_area_field: str = "area_ac_og",
) -> None:
    """Add clipped area (acres) and area percentage vs. original."""
    safe_add_field(clipped_fc, area_field, "DOUBLE")
    calc_field(clipped_fc, area_field, "!shape.area@SQUAREMETERS! / 4046.86")

    safe_add_field(clipped_fc, pct_field, "DOUBLE")
    calc_field(clipped_fc, pct_field, f"!{area_field}! / !{original_area_field}!")


def _normalize_pct(x: float) -> float:
    """Normalize a percentage that may be expressed 0–1 or 0–100."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (ValueError, TypeError):
        return 0.0
    if v < 0:
        return 0.0
    if v > 1.0:
        # Heuristic: treat as 0–100
        return v / 100.0
    return v


def _geom_area_m2(fc: str) -> Optional[float]:
    """Compute total geodesic area (m^2) of a feature class, safely."""
    total = 0.0
    try:
        with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
            for (geom,) in cur:
                if geom:
                    total += geom.getArea("GEODESIC", "SQUAREMETERS")
        return total
    except Exception as exc:  # noqa: BLE001
        logging.warning("Diagnostics: could not compute geodesic area (%s)", exc)
        return None


def add_synthetic_fields(
    clipped_fc: str,
    area_pct_field: str = "area_perc",
    demographics_fc_for_schema: Optional[str] = None,
) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """Create area-weighted counts on the clipped layer using auto-detected schema.

    If a direct count field exists, we scale it by area_perc.
    If only a percent exists (with its appropriate total), we scale (pct * total) by area_perc.

    Safety (legacy layers only, see ``detect_demog_schema``):
      - When a metric on a layer without block-allocated counts resolves to a direct
        count but the pct+total pair is also present on the clipped data, use the
        derived value if the raw count would exceed the implied total for that row.
        Layers with allocated counts use the counts as-is, so every row follows the
        same method.
    """
    if not field_exists(clipped_fc, area_pct_field):
        raise ValueError(
            f"Missing '{area_pct_field}' on {clipped_fc}. "
            "Run add_clipped_area_and_percentage() first."
        )

    probe_fc = demographics_fc_for_schema or clipped_fc
    schema = detect_demog_schema(probe_fc)
    outputs = schema.outputs
    strategies = schema.strategies

    if not outputs:
        logging.info("Synthetic outputs: <none>")
        return tuple(), {}

    # Ensure output fields exist
    for out_name in outputs:
        safe_add_field(clipped_fc, out_name, "DOUBLE")

    # Determine needed inputs (resolved names from schema)
    needed_inputs: set[str] = set()
    for mode, spec in strategies.values():
        if mode == "count":
            needed_inputs.add(str(spec))  # spec is the count field
        else:
            pct_field, tot_field = spec  # type: ignore[misc]
            needed_inputs.add(str(pct_field))
            needed_inputs.add(str(tot_field))

    # Also try to fetch pct/total for count-based metrics (for the legacy sanity
    # fallback; allocated counts are used as-is)
    derived_helpers: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for out_name, (mode, _spec) in strategies.items():
        if mode == "count" and not schema.counts_allocated:
            # Look up matching pct/total names from _PREFS (canonical), then resolve on clipped FC
            pref = _PREFS.get(out_name)
            if pref and pref.pct and pref.total:
                derived_helpers[out_name] = (pref.pct, pref.total)
                needed_inputs.add(pref.pct)
                needed_inputs.add(pref.total)

    # Re-resolve against the *clipped* layer (it inherits prefixes from source)
    desired_again = sorted(needed_inputs)
    remap = _resolve_many(clipped_fc, desired_again)

    # Build final cursor field list
    # area fraction + present inputs + outputs
    def _f(x: Any) -> float:
        if x in (None, "", " "):
            return 0.0
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    present_inputs = [v for v in remap.values() if v is not None]
    cursor_fields = [area_pct_field] + present_inputs + list(outputs)
    idx = {name: i for i, name in enumerate(cursor_fields)}

    # Build helper maps for quick lookups
    input_on_clip: Dict[str, str] = {k: v for k, v in remap.items() if v is not None}

    with arcpy.da.UpdateCursor(clipped_fc, cursor_fields) as cur:
        for row in cur:
            a = _f(row[idx[area_pct_field]])

            # Read all input values present on this row
            vals: dict[str, float] = {}
            for fname in present_inputs:
                vals[fname] = _f(row[idx[fname]])

            for out_name in outputs:
                mode, spec = strategies[out_name]
                if mode == "count":
                    src = str(spec)
                    src_clip = input_on_clip.get(src)
                    raw = a * (vals.get(src_clip or "", 0.0))

                    # Optional derived fallback if pct+total are available
                    dspec = derived_helpers.get(out_name)
                    if dspec:
                        pct_req, tot_req = dspec
                        pct_clip = input_on_clip.get(pct_req or "")
                        tot_clip = input_on_clip.get(tot_req or "")
                        if pct_clip and tot_clip:
                            pct_val = vals.get(pct_clip, 0.0)
                            pct_val = pct_val / 100.0 if pct_val > 1.0 else pct_val
                            tot_val = vals.get(tot_clip, 0.0)
                            derived = a * (pct_val * tot_val)
                            # If raw would exceed the derived-total expectation,
                            # use the safer derived value.
                            if derived > 0.0 and raw > derived:
                                row[idx[out_name]] = derived
                            else:
                                row[idx[out_name]] = raw
                        else:
                            row[idx[out_name]] = raw
                    else:
                        row[idx[out_name]] = raw

                else:
                    pct_field, tot_field = spec  # type: ignore[misc]
                    pct_clip = input_on_clip.get(str(pct_field))
                    tot_clip = input_on_clip.get(str(tot_field))
                    pct_val = vals.get(pct_clip or "", 0.0)
                    pct_val = pct_val / 100.0 if pct_val > 1.0 else pct_val
                    tot_val = vals.get(tot_clip or "", 0.0)
                    row[idx[out_name]] = a * (pct_val * tot_val)

            cur.updateRow(row)

    pretty_map: Dict[str, str] = {}
    for out_name, (mode, spec) in strategies.items():
        if mode == "count":
            pretty_map[out_name] = f"{out_name} := area_perc * {spec}"
        else:
            pct_field, tot_field = spec  # type: ignore[misc]
            pretty_map[out_name] = f"{out_name} := area_perc * ({pct_field} * {tot_field})"

    logging.info("Synthetic outputs: %s", ", ".join(outputs))
    for k, v in pretty_map.items():
        logging.info("  %s -> %s", k, v)

    # Quick diagnostics on area-perc distribution
    try:
        vals = []
        with arcpy.da.SearchCursor(clipped_fc, [area_pct_field]) as cur2:
            for (v,) in cur2:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if vals:
            logging.info(
                "Diagnostics: area_perc per-source: max=%.3f, mean=%.3f (N=%d)",
                max(vals),
                sum(vals) / max(len(vals), 1),
                len(vals),
            )
    except Exception:
        pass

    return outputs, pretty_map


def resolved_clipped_fields(demographics_fc: str) -> List[str]:
    """Return the list of outputs we can produce given the source schema.

    Keeps the original six at the front if present; appends extras in a stable order.
    """
    sch = detect_demog_schema(demographics_fc)
    base = ["loinc_hh", "total_hh", "minor_pop", "total_pop", "loinc_jobs", "all_jobs"]
    extras = [x for x in sch.outputs if x not in base]
    return [x for x in base if x in sch.outputs] + sorted(extras)


# =============================================================================
# GEOPROCESSING STEPS
# =============================================================================
def _maybe_project_for_processing(in_fc: str, name_hint: str) -> str:
    """Optionally project to PROCESS_CRS_WKID for buffering/clipping stability."""
    if PROCESS_CRS_WKID is None:
        return in_fc
    spref = arcpy.SpatialReference(PROCESS_CRS_WKID)
    out_fc = _route_scoped_temp(f"proc_{name_hint}", "proj")
    arcpy.management.Project(in_fc, out_fc, spref)
    return out_fc


def buffer_stops(stops_layer: str, out_path: str, miles: float) -> str:
    """Buffer stops by the given distance in miles (optionally after projection)."""
    src = _maybe_project_for_processing(stops_layer, "stops")
    distance = _as_distance_miles(miles)
    result = arcpy.analysis.Buffer(src, out_path, distance)
    return str(result)


def dissolve_buffers(
    buffer_fc: str,
    out_path: str,
) -> str:
    """Dissolve buffered polygons (all features into one or a few multipart features)."""
    src = _maybe_project_for_processing(buffer_fc, "buffer")
    result = arcpy.management.Dissolve(src, out_path, dissolve_field="")
    return str(result)


def _sr_name(obj_path: str) -> str:
    try:
        return arcpy.Describe(obj_path).spatialReference.name
    except Exception:
        return "<unknown>"


def _count(obj_path: str) -> int:
    try:
        return int(arcpy.management.GetCount(obj_path).getOutput(0))
    except Exception:
        return -1


def _log_env() -> None:
    env = arcpy.env
    logging.info(
        "Env: extent=%s mask=%s snapRaster=%s",
        getattr(env, "extent", None),
        getattr(env, "mask", None),
        getattr(env, "snapRaster", None),
    )


def _project_to_match_sr(in_fc: str, like_fc: str, name_hint: str) -> str:
    """Project *in_fc* to the spatial reference of *like_fc* if they differ.

    Writes to a persistent shapefile in the inspection folder returned by
    _intermediate_buffer_folder(). This avoids scratch.gdb visibility issues
    and keeps the intermediate available for manual inspection.

    Args:
        in_fc: Input feature class or layer to (maybe) project.
        like_fc: Dataset whose spatial reference we want to match.
        name_hint: Short token used in naming the output.

    Returns:
        Path to a shapefile in the same spatial reference as *like_fc*.
        Returns *in_fc* unchanged when spatial references already match.
    """
    try:
        in_sr = arcpy.Describe(in_fc).spatialReference
        like_sr = arcpy.Describe(like_fc).spatialReference
    except Exception as exc:  # noqa: BLE001
        logging.warning("SR inspection failed; proceeding without projection (%s)", exc)
        return in_fc

    def _sr_eq(a: arcpy.SpatialReference, b: arcpy.SpatialReference) -> bool:
        try:
            if a.factoryCode and b.factoryCode:
                return int(a.factoryCode) == int(b.factoryCode)
        except Exception:
            pass
        try:
            return (a.name or "").strip().lower() == (b.name or "").strip().lower()
        except Exception:
            return False

    if _sr_eq(in_sr, like_sr):
        return in_fc

    # Persistent shapefile path for inspection
    folder = _intermediate_buffer_folder()
    base = f"proj_{_sanitize_name(name_hint)}.shp"
    out_fc = arcpy.CreateUniqueName(base, folder)

    # Best-effort: choose a geographic transformation if recommended
    transform = None
    try:
        cands = arcpy.ListTransformations(in_sr, like_sr) or []
        if cands:
            transform = cands[0]
    except Exception:
        pass

    logging.info(
        "Projecting '%s' SR [%s] → match '%s' SR [%s] (transform=%s) → '%s' …",
        in_fc,
        getattr(in_sr, "name", "<unknown>"),
        like_fc,
        getattr(like_sr, "name", "<unknown>"),
        transform or "<none>",
        out_fc,
    )
    arcpy.management.Project(in_fc, out_fc, like_sr, transform)
    return out_fc


def _intermediate_buffer_folder() -> str:
    """Return a stable folder for projected buffer intermediates (shapefiles).

    Uses the user-specified inspection directory so outputs persist for review.

    Returns:
        Absolute path to the inspection folder, created if missing.
    """
    folder = r"projects\dot\zkrohmal\census_merge_test_2025_10_31\output_buffer_intermediate"
    ensure_dir(folder)
    return folder


def _scratch_gdb() -> str:
    """Return a writable scratch file geodatabase path, creating it if needed.

    Prefers arcpy.env.scratchGDB. Falls back to <scratchFolder>/sr_temp.gdb,
    or OS temp if scratchFolder is unavailable.
    """
    gdb = getattr(arcpy.env, "scratchGDB", None)
    if gdb and arcpy.Exists(gdb):
        return gdb

    folder = getattr(arcpy.env, "scratchFolder", None) or os.environ.get("TEMP") or os.getcwd()
    ensure_dir(folder)
    gdb = os.path.join(folder, "sr_temp.gdb")
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(folder, "sr_temp.gdb")
    return gdb


def clip_demographics_to_buffers(
    demographics_fc: str,
    buffers_fc: str,
    out_path: str,
) -> str:
    """Clip demographics by buffers with SR-alignment, diagnostics, and fallbacks.

    Steps:
      1) Ensure the buffer geometry is in the demographics SR (project if needed).
      2) Add a spatial index to demographics (idempotent).
      3) Prefer PairwiseClip; fall back to Clip; then Intersect on failure.

    Args:
        demographics_fc: Polygon feature class with demographic attributes.
        buffers_fc: Dissolved service-area polygons (any SR).
        out_path: Destination path for clipped output.

    Returns:
        Path to the clipped feature class.
    """
    _log_env()

    # Align SRs for robust and predictable overlay
    buffers_for_clip = _project_to_match_sr(
        in_fc=buffers_fc,
        like_fc=demographics_fc,
        name_hint="buffers_for_clip",
    )

    # Pre-flight diagnostics
    logging.info(
        "Clip preflight: DEMO SR=%s, count=%s",
        _sr_name(demographics_fc),
        _count(demographics_fc),
    )
    logging.info(
        "Clip preflight: BUFF SR=%s, count=%s",
        _sr_name(buffers_for_clip),
        _count(buffers_for_clip),
    )

    # Make sure large FCs have a spatial index (idempotent)
    try:
        arcpy.management.AddSpatialIndex(demographics_fc)
    except arcpy.ExecuteError:
        pass

    # Ensure output container exists and is writable
    out_dir = os.path.dirname(out_path)
    if out_dir and out_dir.lower().endswith(".gdb") and not arcpy.Exists(out_dir):
        dir_, name = os.path.split(out_dir)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        arcpy.management.CreateFileGDB(dir_, name)

    # Primary attempt: PairwiseClip (when available) or Clip
    result_path = out_path
    try:
        if hasattr(arcpy.analysis, "PairwiseClip"):
            logging.info("Running PairwiseClip …")
            arcpy.analysis.PairwiseClip(demographics_fc, buffers_for_clip, result_path)
        else:
            logging.info("Running Clip …")
            arcpy.analysis.Clip(demographics_fc, buffers_for_clip, result_path)
    except arcpy.ExecuteError as exc1:
        logging.warning("Primary clip failed: %s", exc1)
        # Fallback: Intersect polygons with buffer
        try:
            logging.info("Fallback: Intersect (polygons ∩ buffer) …")
            tmp = result_path + "_ix_tmp"
            arcpy.analysis.Intersect([demographics_fc, buffers_for_clip], tmp, "ALL", "", "INPUT")
            if arcpy.Exists(result_path):
                arcpy.management.Delete(result_path)
            arcpy.management.CopyFeatures(tmp, result_path)
            arcpy.management.Delete(tmp)
        except Exception:
            logging.error("Fallback Intersect also failed.\nGP Messages:\n%s", arcpy.GetMessages(2))
            raise

    n = _count(result_path)
    logging.info(
        "Clip result: %s features at %s (SR=%s)",
        n,
        result_path,
        _sr_name(result_path),
    )
    if n == 0:
        logging.warning(
            "Clip returned ZERO features. Likely causes:\n"
            "  • Extent/Mask env excludes overlaps (current env logged above).\n"
            "  • No actual overlap (check buffer distance and SR).\n"
            "  • Demographics is multipart with tiny slivers and buffer in a different CRS.\n"
            "  • Selection on inputs (layer with empty selection)."
        )
        logging.warning("GP Messages:\n%s", arcpy.GetMessages(2))

    return str(result_path)


def summarize_fields(feature_class: str, fields: Iterable[str]) -> Dict[str, int]:
    """Sum a set of numeric fields and return rounded integers."""
    fields_list = list(fields)
    if not fields_list:
        return {}

    totals = {f: 0.0 for f in fields_list}
    with arcpy.da.SearchCursor(feature_class, fields_list) as cur:
        for row in cur:
            for i, f in enumerate(fields_list):
                val = row[i]
                if val is not None:
                    totals[f] += float(val)
    return {k: int(round(v)) for k, v in totals.items()}


def export_final_copy(
    in_feature_class: str,
    out_target: str,
    run_tag: str,
) -> str:
    """Copy features to final destination.

    If out_target ends with '.gdb', create a feature class '{run_tag}_service_buffer_data'.
    Otherwise treat out_target as a directory and write a shapefile with that name.
    """
    if out_target.lower().endswith(".gdb"):
        gdb = out_target
        if not arcpy.Exists(gdb):
            dir_, name = os.path.split(gdb)
            ensure_dir(dir_)
            arcpy.management.CreateFileGDB(dir_, name)
        out_name = f"{run_tag}_service_buffer_data"
        out_fc = os.path.join(gdb, out_name)
        if arcpy.Exists(out_fc):
            arcpy.management.Delete(out_fc)
        arcpy.management.CopyFeatures(in_feature_class, out_fc)
        return out_fc

    # shapefile directory
    ensure_dir(out_target)
    out_fc = os.path.join(out_target, f"{run_tag}_service_buffer_data.shp")
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.management.CopyFeatures(in_feature_class, out_fc)
    return out_fc


# =============================================================================
# CONSOLIDATED EXECUTION HELPERS
# =============================================================================


def calc_original_area_for_intersecting(
    demographics_fc: str,
    buffers_fc: str,
    *,
    insurance_distance_ft: float = 100.0,
    field_name: str = "area_ac_og",
) -> None:
    """Populate original area (acres) only for demo polygons near the buffer.

    Creates/ensures the `area_ac_og` field on the source demographics feature
    class, then calculates it *only* for features that intersect the buffer
    expanded by `insurance_distance_ft`. Non-intersecting features are left
    untouched, avoiding a full-table CalculateField over ~40k rows.

    Call this before clipping: the clip copies `area_ac_og` from the source,
    so values written afterwards never reach the clipped features.

    Args:
        demographics_fc: Input demographics polygon feature class to update.
        buffers_fc: Dissolved service-area polygons.
        insurance_distance_ft: Extra distance (feet) to expand the buffer
            before selecting intersecting demographics.
        field_name: Name of the target field holding original area in acres.
    """
    # 1) Ensure target field exists on the *source* FC (idempotent schema change).
    safe_add_field(demographics_fc, field_name, "DOUBLE")

    # 2) Create an expanded selection buffer (temporary).
    sel_buf = _route_scoped_temp("sel_buffer_expanded", "intersect_area")
    try:
        arcpy.analysis.Buffer(
            in_features=buffers_fc,
            out_feature_class=sel_buf,
            buffer_distance_or_field=f"{float(insurance_distance_ft)} Feet",
            line_side="FULL",
            line_end_type="ROUND",
            dissolve_option="ALL",
        )

        # 3) Make a selectable layer from the demographics and select by location.
        lyr = arcpy.management.MakeFeatureLayer(demographics_fc, "demog_for_area_sel")
        arcpy.management.SelectLayerByLocation(
            in_layer=lyr,
            overlap_type="INTERSECT",
            select_features=sel_buf,
            selection_type="NEW_SELECTION",
        )

        # 4) If nothing selected, bail early.
        sel_count = int(arcpy.management.GetCount(lyr).getOutput(0))
        logging.info(
            "Area-prep selection: %d demographics features within %.1f ft of buffer.",
            sel_count,
            insurance_distance_ft,
        )
        if sel_count == 0:
            # Clear selection to be tidy and return.
            arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")
            return

        # 5) Calculate area only for the selected rows.
        arcpy.management.CalculateField(
            in_table=lyr,
            field=field_name,
            expression="!shape.area@SQUAREMETERS! / 4046.86",
            expression_type="PYTHON3",
        )

        # 6) Clear selection.
        arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")

    finally:
        # Best-effort cleanup of the temp selection buffer.
        try:
            if arcpy.Exists(sel_buf):
                arcpy.management.Delete(sel_buf)
        except arcpy.ExecuteError:
            pass


def _process_service_area_from_stops_layer(
    stops_layer: str,
    run_tag: str,
    *,
    export_final: bool,
    final_export_target: str,
    buffered_path: Optional[str] = None,
    dissolved_path: Optional[str] = None,
    clipped_path: Optional[str] = None,
) -> Tuple[Dict[str, int], Optional[str]]:
    """Run the buffer→dissolve→clip→synthetics→summaries pipeline for any stops layer.

    When disk paths are provided for buffered/dissolved/clipped, they are used.
    Otherwise, unique in_memory paths are allocated (per-route use case).
    """
    # Allocate paths if not given
    buffered_path = buffered_path or _route_scoped_temp("buffered_stops", run_tag)
    dissolved_path = dissolved_path or _route_scoped_temp("dissolved_buffers", run_tag)
    clipped_path = clipped_path or _route_scoped_temp("clipped_demog", run_tag)

    # 1) Buffer
    logging.info("Buffering stops (%s)…", run_tag)
    buffered = buffer_stops(stops_layer, buffered_path, BUFFER_DISTANCE_MILES)

    # 2) Dissolve
    logging.info("Dissolving buffers (%s)…", run_tag)
    dissolved = dissolve_buffers(buffered, dissolved_path)

    # 2b) Diagnostics: compute area of dissolved
    area_m2 = _geom_area_m2(dissolved)
    if area_m2 is not None:
        logging.info("Diagnostics: dissolved geodesic area = %.2f sq.m", area_m2)

    # 3) Precompute original area ONLY for demographics that matter (near the buffer).
    # This must run BEFORE the clip: the clip copies area_ac_og from the source
    # rows, and a later update to the source never reaches the clipped copy (it
    # would carry a missing value on a fresh input, or a stale one from a prior run).
    calc_original_area_for_intersecting(
        demographics_fc=DEMOGRAPHICS_FC,
        buffers_fc=dissolved,
        insurance_distance_ft=100.0,
        field_name="area_ac_og",
    )

    # 4) Clip demographics (inherits the fresh area_ac_og computed above)
    logging.info("Clipping demographics (%s)…", run_tag)
    clipped = clip_demographics_to_buffers(DEMOGRAPHICS_FC, dissolved, clipped_path)

    # 5) Areas and percentages on the clipped output
    add_clipped_area_and_percentage(clipped)

    # 6) Synthetic metrics (auto-detected schema from original FC; resolved again on clip)
    outputs, _strategy_map = add_synthetic_fields(
        clipped,
        area_pct_field="area_perc",
        demographics_fc_for_schema=DEMOGRAPHICS_FC,
    )

    # 7) Summaries
    fields_for_summary = resolved_clipped_fields(DEMOGRAPHICS_FC) if outputs else []
    totals = summarize_fields(clipped, fields_for_summary)

    # Optional export to disk/GDB
    exported_path: Optional[str] = None
    if export_final:
        exported_path = export_final_copy(clipped, final_export_target, run_tag)

    return totals, exported_path


def _run_network_total(stops_layer: str) -> None:
    """Whole-network summary using the already-prepared stops layer.

    Intermediates use explicit disk outputs if provided in config; otherwise in_memory.
    """
    logging.info(
        "Buffering stops to %s mi -> %s (intermediates)",
        BUFFER_DISTANCE_MILES,
        "disk" if BUFFERED_STOPS_OUT else "in_memory",
    )

    svc_totals, exported = _process_service_area_from_stops_layer(
        stops_layer=stops_layer,
        run_tag=RUN_TAG,
        export_final=True,
        final_export_target=FINAL_EXPORT_TARGET,
        buffered_path=BUFFERED_STOPS_OUT or None,
        dissolved_path=DISSOLVED_BUFFERS_OUT or None,
        clipped_path=CLIPPED_DEMOGRAPHICS_OUT or None,
    )

    logging.info("[%s] Service buffer totals (area-weighted, rounded):", RUN_TAG)
    for k in resolved_clipped_fields(DEMOGRAPHICS_FC):
        logging.info("  %s: %s", k, f"{svc_totals.get(k, 0):,}")

    if exported:
        logging.info("Final export: %s", exported)


def _run_by_route(
    gtfs_folder: str,
    route_short_names: Sequence[str],
    service_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-route summaries (GTFS only). Returns a DataFrame of results.

    For each route_short_name, rebuilds its own stops layer from GTFS (trips on
    *service_ids* only), runs the service-area pipeline (in_memory by default),
    and collects totals. A route with no trips on *service_ids* is skipped with a
    warning.
    """
    results: List[Dict[str, int | str]] = []
    for route_sn in route_short_names:
        tag = f"{RUN_TAG}_route_{_sanitize_name(route_sn)}"
        logging.info("Processing route %s", route_sn)

        # Build a route-scoped stops layer directly from GTFS
        try:
            stops_layer = make_stops_layer(
                mode="gtfs",
                gtfs_folder=gtfs_folder,
                route_short_names=[route_sn],
                service_ids=service_ids,
                layer_name=f"stops_{_sanitize_name(route_sn)}",
            )
        except RoutesNotInServiceError as exc:
            logging.warning("Skipping route %s: %s", route_sn, exc)
            continue

        totals, exported = _process_service_area_from_stops_layer(
            stops_layer=stops_layer,
            run_tag=tag,
            export_final=BY_ROUTE_EXPORT_FEATURES,
            final_export_target=FINAL_EXPORT_TARGET,
        )

        row: Dict[str, int | str] = {"route_short_name": str(route_sn)}
        for k in resolved_clipped_fields(DEMOGRAPHICS_FC):
            row[k] = int(totals.get(k, 0))
        results.append(row)

        if BY_ROUTE_EXPORT_FEATURES and exported:
            logging.info("  Exported per-route features: %s", exported)

    df = pd.DataFrame(results).sort_values("route_short_name").reset_index(drop=True)

    # CSV output adjacent to the target (folder or alongside GDB)
    if BY_ROUTE_WRITE_CSV and not df.empty:
        out_dir = (
            FINAL_EXPORT_TARGET
            if not FINAL_EXPORT_TARGET.lower().endswith(".gdb")
            else os.path.dirname(FINAL_EXPORT_TARGET)
        )
        ensure_dir(out_dir)
        csv_path = os.path.join(out_dir, f"{RUN_TAG}_by_route_summary.csv")
        df.to_csv(csv_path, index=False)
        logging.info("Per-route summary written: %s", csv_path)

    # Console view
    if not df.empty:
        cols = ["route_short_name"] + resolved_clipped_fields(DEMOGRAPHICS_FC)
        logging.info(
            "Per-route totals (area-weighted, rounded):\n%s", df[cols].to_string(index=False)
        )

    return df


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    """Run the end-to-end pipeline according to OUTPUT_MODE.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders, a requested service_id has
        no trips in the feed, or the selected routes have none on that day.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if GTFS_FOLDER == r"Path\To\Your\GTFS_Folder":
        logging.warning(
            "GTFS_FOLDER is still set to a placeholder value. "
            "Please update it in the CONFIGURATION section before running."
        )
        return 2

    arcpy.env.overwriteOutput = OVERWRITE_OUTPUTS

    # Basic input checks (mode-specific)
    m = STOPS_INPUT_MODE.lower()
    if m == "shapefile":
        if not arcpy.Exists(STOPS_FEATURE_CLASS):
            raise FileNotFoundError(f"Stops feature class not found: {STOPS_FEATURE_CLASS}")
    elif m == "gtfs":
        if not os.path.exists(GTFS_FOLDER):
            raise FileNotFoundError(f"GTFS folder not found: {GTFS_FOLDER}")
    else:
        raise ValueError("STOPS_INPUT_MODE must be 'shapefile' or 'gtfs'.")

    if not arcpy.Exists(DEMOGRAPHICS_FC):
        raise FileNotFoundError(f"Input not found: {DEMOGRAPHICS_FC}")

    # Prepare stops layer (network scope) once
    logging.info("Preparing stops layer...")
    try:
        if m == "gtfs":
            # calendar.txt is logged in full so the user can confirm (or pick) the
            # service_id; a requested service_id with no trips stops the run here.
            _check_service_selection(GTFS_FOLDER, SERVICE_IDS_TO_INCLUDE)
        stops_layer = make_stops_layer(
            mode=STOPS_INPUT_MODE,
            shapefile_fc=STOPS_FEATURE_CLASS,
            gtfs_folder=GTFS_FOLDER,
            route_short_names=GTFS_ROUTE_SHORT_NAMES,
            service_ids=SERVICE_IDS_TO_INCLUDE,
            layer_name="stops_for_buffering",
        )
    except ServiceSelectionError as exc:
        # A configuration problem, not a crash: report the fix without a traceback.
        logging.error("%s", exc)
        return 2

    mode = OUTPUT_MODE.lower().strip()
    if mode not in {"network", "by_route", "both"}:
        raise ValueError("OUTPUT_MODE must be one of {'network','by_route','both'}.")

    run_by_route = mode in {"by_route", "both"}
    run_network = mode in {"network", "both"}

    if run_network:
        _run_network_total(stops_layer)

    if run_by_route:
        if STOPS_INPUT_MODE.lower() != "gtfs":
            raise RuntimeError(
                "Per-route output requires STOPS_INPUT_MODE='gtfs' so that "
                "stops can be reconstructed by route_short_name."
            )
        route_list = list(GTFS_ROUTE_SHORT_NAMES or [])
        if not route_list:
            # Guardrail: avoid accidental 'all routes' on very large feeds.
            raise ValueError(
                "GTFS_ROUTE_SHORT_NAMES is empty; populate it to enable per-route processing."
            )
        _ = _run_by_route(GTFS_FOLDER, route_list, SERVICE_IDS_TO_INCLUDE)

    logging.info("Processing completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
