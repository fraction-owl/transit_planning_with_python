"""Identify potential typos in GTFS stop names by comparing them to nearby roadway names.

Creates a spatial buffer around each GTFS stop, joins to intersecting roadway segments,
and uses fuzzy string matching to flag stop names that are similar—but not identical—
to adjacent street names. Intended for QA of stop name consistency in GIS-based transit data.

Outputs:
    - CSV of potential stop name typos and similarity scores. The CSV is
      rewritten on every run; a run with no findings writes a header-only CSV.
    - File geodatabase with intermediate feature classes for inspection.

Typical usage:
    Run in ArcGIS Pro's Python environment or any ArcPy-enabled session
    after configuring input paths and parameters at the top of the script.
"""

from __future__ import annotations

import datetime as _dt  # aliased: notebooks often rebind 'datetime' to the module
import difflib
import logging
import os
import re
import sys
import zipfile
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

import arcpy  # type: ignore
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

GTFS_FOLDER = r"path\to\your\GTFS"  # GTFS feed folder or .zip
ROADWAYS_PATH = r"path\to\your\roadways.shp"
OUTPUT_DIR = r"path\to\output"  # any writable folder
OUTPUT_CSV = "potential_typos.csv"

# Spatial references
STOPS_CRS = 4326  # GTFS lat/lon – WGS-84
TARGET_CRS = 2248  # example: NAD83 / Maryland (US ft). change if needed

# Processing parameters
BUFFER_DISTANCE = 50
BUFFER_DISTANCE_UNIT = "feet"  # 'feet' or 'meters'
SIMILARITY_THRESHOLD = 80  # 0-100

# Roadway field requirements
REQUIRED_COLUMNS_ROADWAY = [
    "RW_PREFIX",
    "RW_TYPE_US",
    "RW_SUFFIX",
    "RW_SUFFIX_",
    "FULLNAME",
]

DESCRIPTIONS_ROADWAY = {
    "RW_PREFIX": "Directional prefix (e.g. 'N' in 'N Washington St')",
    "RW_TYPE_US": "Street type (e.g. 'St' in 'N Washington St')",
    "RW_SUFFIX": "Directional suffix (e.g. 'SE' in 'Park St SE')",
    "RW_SUFFIX_": "Additional suffix (e.g. 'EB' in 'I-66 EB')",
    "FULLNAME": "Full street name",
}

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

arcpy.env.overwriteOutput = True

# === END CONFIG ===

# =============================================================================
# FUNCTIONS
# =============================================================================


def create_work_gdb(base_dir: str) -> str:
    """Create <base_dir>/typo_work_<timestamp>.gdb and return its path.

    Re-use if it already exists in this session.
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"typo_work_{ts}.gdb"
    gdb = os.path.join(base_dir, name)
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(base_dir, name)
        logging.info("Created workspace %s", gdb)
    else:
        logging.info("Using existing workspace %s", gdb)
    return gdb


def fgdb_path(gdb: str, fc_name: str) -> str:
    """Return full path inside the work GDB."""
    return os.path.join(gdb, fc_name)


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------


@contextmanager
def open_gtfs_file(
    gtfs_path: str | os.PathLike[str], file_name: str
) -> Generator[IO[bytes], None, None]:
    """Open one GTFS text file from a feed folder or a ``.zip`` of the feed.

    The counterpart of :func:`load_gtfs_data` for scripts that read a file with
    their own options (``pandas.read_csv`` with ``usecols`` or
    ``keep_default_na=False``, or the :mod:`csv` module), so they accept a
    zipped feed without extracting it. Zip members are found the way
    :func:`load_gtfs_data` finds them: at the archive root or inside a single
    wrapper folder.

    Args:
        gtfs_path: Folder containing the GTFS feed, or a ``.zip`` archive of it.
        file_name: GTFS file to open, e.g. ``"stops.txt"``.

    Yields:
        A binary file handle. ``pandas.read_csv`` reads it directly; wrap it in
        :class:`io.TextIOWrapper` for the :mod:`csv` module.

    Raises:
        FileNotFoundError: *gtfs_path* does not exist, or the feed has no
            *file_name*.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, or *file_name* is found in more than one place in the zip.
    """
    path = os.fspath(gtfs_path)
    if os.path.isdir(path):
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        with open(file_path, "rb") as handle:
            yield handle
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    if not (os.path.isfile(path) and path.lower().endswith(".zip")):
        raise ValueError(f"'{path}' is neither a directory nor a .zip file.")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc
    with archive:
        matches = [name for name in archive.namelist() if os.path.basename(name) == file_name]
        if not matches:
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous GTFS file in '{path}' (found in multiple locations): {file_name}"
            )
        with archive.open(matches[0]) as handle:
            yield handle


def load_gtfs_stops(folder: str) -> pd.DataFrame:
    """Loads GTFS stops from stops.txt into a pandas DataFrame.

    Args:
        folder: The GTFS feed folder containing `stops.txt`, or a ``.zip`` of
            the feed.

    Returns:
        A pandas DataFrame with stop data. Pandas' default NA parsing is
        disabled, so literal values such as ``"NA"`` stay strings. Rows with
        blank coordinates (allowed by GTFS for some location types) are
        dropped with a warning.

    Raises:
        FileNotFoundError: If `stops.txt` is not found in the feed.
        ValueError: If the `stops.txt` file is missing required columns, a
            non-blank lat/lon cannot be cast to float, or *folder* is neither
            a folder nor a valid ``.zip``.
    """
    with open_gtfs_file(folder, "stops.txt") as handle:
        df = pd.read_csv(handle, dtype=str, keep_default_na=False, low_memory=False)
    need = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    if not need.issubset(df.columns):
        raise ValueError(f"stops.txt missing columns: {', '.join(need - set(df.columns))}")
    blank = pd.Series(False, index=df.index)
    for col in ("stop_lat", "stop_lon"):
        text = df[col].astype("string").str.strip()
        blank |= text.isna() | (text == "")
    if blank.any():
        logging.warning("Dropping %d stop(s) with blank coordinates.", int(blank.sum()))
        df = df.loc[~blank].copy()
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)
    return df


def normalize_street(name: str, mods: set[str]) -> str:
    """Cleans and standardizes a street name string.

    Removes one modifier (e.g., 'St', 'Ave'), punctuation, and extra whitespace,
    and converts the string to lowercase. The modifier removed is the last one
    that is not the first word: the street type follows the name ("Johns Rd",
    "Park St SE"), so "St Johns Rd" becomes "st johns" rather than "johns",
    which would equal "Johns Rd".

    Empty modifiers are ignored: one in the pattern matches at every word
    boundary, so no street type would ever be removed.

    Apostrophes and periods are dropped, so "Buckley's" equals "Buckleys" and
    "U.S.G.S" equals "USGS"; other punctuation becomes a space, so
    "Franconia-Springfield" equals "Franconia Springfield".

    Args:
        name: The street name to normalize.
        mods: A set of modifier words (like 'rd', 'st', 'blvd') to remove.

    Returns:
        The normalized street name string.
    """
    if not isinstance(name, str):
        return ""
    mods = {m for m in mods if m}
    if mods:
        pattern = r"\b(" + "|".join(map(re.escape, mods)) + r")\b"
        matches = [
            m
            for m in re.finditer(pattern, name, flags=re.IGNORECASE)
            if re.search(r"\w", name[: m.start()])
        ]
        if matches:
            name = name[: matches[-1].start()] + " " + name[matches[-1].end() :]
    name = re.sub(r"['’.]", "", name)
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def split_stop_name(stop_name: str, mods: set[str]) -> list[str]:
    """Splits a GTFS stop name into normalized street name components.

    Uses common intersection separators (e.g., '@', '&', '/', '+', 'and', 'at',
    'intersection of') to divide the stop name and then normalizes each
    resulting part.

    Args:
        stop_name: The full stop name string.
        mods: A set of modifier words to be removed during normalization.

    Returns:
        A list of normalized street name fragments from the stop name.
    """
    if not isinstance(stop_name, str):
        return []
    # Symbol separators split with or without surrounding spaces ("A&B", "A @ B");
    # word separators ("and", "at") need them. "Intersection of" also splits at
    # the start of the name, so it is not left attached to the first street.
    pattern = r"\s*[@&/+]\s*| and | at |(?:^|\s)intersection of "
    parts = re.split(pattern, stop_name, flags=re.IGNORECASE)
    return [normalize_street(p, mods) for p in parts if p.strip()]


def dl_score(a: str, b: str) -> float:
    """Calculates difflib's SequenceMatcher similarity ratio between two strings.

    The ratio is 2*M/T, where M is the number of characters in the matching
    blocks SequenceMatcher finds and T is the combined length of both strings.
    It is not an edit distance (Damerau-Levenshtein or otherwise), and swapping
    the arguments can change the result, so score and filter in the same order.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        A similarity score between 0.0 and 100.0.
    """
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


def make_stops_fc(df: pd.DataFrame, out_fc: str, sr: int) -> None:
    """Creates a point feature class from a DataFrame of GTFS stops.

    Args:
        df: DataFrame containing stop_id, stop_name, stop_lon, stop_lat.
        out_fc: The full path for the output feature class.
        sr: The spatial reference ID (WKID) for the output feature class.
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.management.CreateFeatureclass(
        os.path.dirname(out_fc),
        os.path.basename(out_fc),
        "POINT",
        spatial_reference=arcpy.SpatialReference(sr),
    )
    # Widen past the defaults if needed: an over-long value fails on insert.
    id_len = int(max([50, *df["stop_id"].str.len()]))
    name_len = int(max([255, *df["stop_name"].str.len()]))
    arcpy.management.AddField(out_fc, "stop_id", "TEXT", field_length=id_len)
    arcpy.management.AddField(out_fc, "stop_name", "TEXT", field_length=name_len)
    with arcpy.da.InsertCursor(out_fc, ["SHAPE@XY", "stop_id", "stop_name"]) as cur:
        for r in df.itertuples(index=False):
            cur.insertRow([(r.stop_lon, r.stop_lat), r.stop_id, r.stop_name])


def safe_project_or_copy(in_fc: str, out_fc: str, out_sr: int) -> None:
    """Project `in_fc` to `out_fc`, or copy it if it already uses `out_sr`.

    Ensures `out_fc` exists, in `out_sr`, on return. A failed projection stops
    the run: copying the input unprojected would let the buffer and spatial
    join run with layers in different coordinate systems.

    Raises:
        RuntimeError: If `in_fc` has no defined coordinate system, Project
            fails, or `out_fc` is not created.
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)

    desc = arcpy.Describe(in_fc)
    src_sr = desc.spatialReference
    tgt_sr = arcpy.SpatialReference(out_sr)

    if not src_sr.name or src_sr.name == "Unknown":
        raise RuntimeError(
            f"{in_fc} has no defined coordinate system (a shapefile may be missing "
            "its .prj file). Set it with ArcGIS Pro's Define Projection tool, then rerun."
        )

    try:
        if src_sr.name and src_sr.factoryCode == tgt_sr.factoryCode:
            # Already in target SR
            arcpy.management.CopyFeatures(in_fc, out_fc)
        else:
            arcpy.management.Project(in_fc, out_fc, tgt_sr)
    except arcpy.ExecuteError as exc:
        raise RuntimeError(f"Could not project {in_fc} to WKID {out_sr}: {exc}") from exc

    if not arcpy.Exists(out_fc):
        raise RuntimeError(f"Failed to create {out_fc}")


def buffer_fc(in_fc: str, out_fc: str, dist: float, unit: str) -> None:
    """Creates a buffer around features in a feature class.

    Args:
        in_fc: The input feature class.
        out_fc: The path for the output buffer feature class.
        dist: The buffer distance.
        unit: The units for the buffer distance (e.g., 'feet', 'meters').
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.analysis.Buffer(in_fc, out_fc, f"{dist} {unit}", dissolve_option="NONE")


def spatial_join_fc(target: str, join: str, out_fc: str) -> None:
    """Performs a one-to-many spatial join.

    Finds all join features that intersect with each target feature.

    Args:
        target: The target feature class.
        join: The feature class to join to the target.
        out_fc: The path for the output joined feature class.
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.analysis.SpatialJoin(
        target,
        join,
        out_fc,
        join_operation="JOIN_ONE_TO_MANY",
        match_option="INTERSECT",
    )


def field_set(fc: str) -> set[str]:
    """Gets a set of field names from a feature class or table."""
    return {f.name for f in arcpy.ListFields(fc)}


def map_road_fields(fc: str) -> dict[str, str]:
    """Prompts user to map required roadway fields if they are not found.

    Args:
        fc: The roadway feature class to check.

    Returns:
        A dictionary mapping required field names to the actual field names
        found in the feature class.

    Raises:
        ValueError: If a user-provided alternative field does not exist, or
            if a mapping for the 'FULLNAME' field is not provided.
    """
    exists = field_set(fc)
    mapping: dict[str, str] = {}
    for col in REQUIRED_COLUMNS_ROADWAY:
        if col in exists:
            mapping[col] = col
        else:
            logging.warning("Field '%s' missing.", col)
            logging.info("Description: %s", DESCRIPTIONS_ROADWAY[col])
            logging.info("Available: %s", ", ".join(sorted(exists)))
            alt = input(f"Enter field name for '{col}' or blank to skip: ").strip()
            if alt:
                if alt in exists:
                    mapping[col] = alt
                else:
                    raise ValueError(f"Field '{alt}' not present.")
    if "FULLNAME" not in mapping:
        raise ValueError("You must supply a field for FULLNAME.")
    return mapping


def modifiers_from_roads(fc: str, fld: str) -> set[str]:
    """Extracts a set of unique string values from a feature class field.

    Used to build a set of street modifiers (e.g., 'St', 'Rd', 'N') for
    normalization.

    Args:
        fc: The feature class to query.
        fld: The field from which to extract values.

    Returns:
        A set of unique, lowercase string values from the specified field.
        Blank values are skipped, including whitespace-only ones: ArcPy reads
        an empty shapefile text field as spaces (e.g. ``"    "``), which
        would otherwise add ``""`` to the set.
    """
    mods = set()
    with arcpy.da.SearchCursor(fc, [fld]) as cur:
        for (v,) in cur:
            text = str(v).strip().lower() if v is not None else ""
            if text:
                mods.add(text)
    return mods


def stop_to_candidate_roads(
    join_fc: str, fullname: str, mods: set[str]
) -> dict[str, dict[str, set[str]]]:
    """Maps each stop ID to the roads inside its buffer.

    Args:
        join_fc: The feature class from the stop-to-road spatial join.
        fullname: The field containing the full roadway name.
        mods: A set of modifiers to remove during road name normalization.

    Returns:
        A dictionary where keys are 'stop_id's and values map each normalized
        name of a road spatially joined to that stop to the original names of
        those same road features. Stops with no roads in their buffer are absent.
    """
    sc: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with arcpy.da.SearchCursor(join_fc, ["stop_id", fullname]) as cur:
        for sid, full in cur:
            if full:
                sc[sid][normalize_street(full, mods)].add(full)
    return sc


def detect_typos(
    stops_df: pd.DataFrame,
    stop2roads: dict[str, dict[str, set[str]]],
    mods: set[str],
    thresh: int,
) -> pd.DataFrame:
    """Compares stop name parts to nearby road names to find likely typos.

    For each stop, it splits the stop name into parts. Each part is then
    compared against the set of nearby road names for that stop. If a part
    is very similar (but not identical) to a nearby road name, it's flagged
    as a potential typo: each nearby road is scored once with
    :func:`dl_score`, and up to three scoring at least ``thresh`` are
    reported. Stops with no roads in their buffer are skipped and counted as
    unassessed.

    Args:
        stops_df: DataFrame of all GTFS stops.
        stop2roads: A mapping from stop_id to {normalized road name -> original
            names} for the roads inside that stop's buffer.
        mods: A set of modifiers to remove during name normalization.
        thresh: The similarity score (0-100) threshold for flagging a typo.

    Returns:
        A pandas DataFrame containing details of each potential typo found.
        Has the same columns, with no rows, when nothing is found.
    """
    out_rows = []
    unassessed = 0

    for rec in stops_df.itertuples(index=False):
        # Ensure sid and sname are strings to satisfy the type checker
        sid, sname = str(rec.stop_id), str(rec.stop_name)
        pieces = split_stop_name(sname, mods)
        candidates = stop2roads.get(sid)
        if not candidates:
            # No roads within this stop's buffer -- nothing to compare against.
            unassessed += 1
            continue

        for frag in pieces:
            if frag in candidates:
                continue
            # Score each road once, in the order reported. get_close_matches
            # scored (road, fragment) instead, which can differ, so it dropped
            # some pairs whose reported score met the threshold.
            scored = sorted(((dl_score(frag, m), m) for m in candidates), reverse=True)
            for score, match in scored[:3]:
                if thresh <= score < 100:
                    for orig in sorted(candidates[match]):
                        out_rows.append(
                            {
                                "stop_id": sid,
                                "stop_name": sname,
                                "street_in_stop_name": frag,
                                "similar_road_name_clean": match,
                                "similar_road_name_orig": orig,
                                "similarity_score": round(score, 1),
                            }
                        )

    if unassessed:
        logging.info("%d stop(s) had no roads within the buffer and were not assessed.", unassessed)

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "stop_id",
                "stop_name",
                "street_in_stop_name",
                "similar_road_name_clean",
                "similar_road_name_orig",
                "similarity_score",
            ]
        )

    return pd.DataFrame(out_rows).sort_values("similarity_score", ascending=False).drop_duplicates()


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Main script execution function.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if GTFS_FOLDER == r"path\to\your\GTFS" or OUTPUT_DIR == r"path\to\output":
        logging.warning(
            "GTFS_FOLDER and/or OUTPUT_DIR are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return 2
    # workspace
    WORK_GDB = create_work_gdb(OUTPUT_DIR)

    # GTFS stops
    logging.info("Loading GTFS stops …")
    stops_df = load_gtfs_stops(GTFS_FOLDER)
    stops_raw = fgdb_path(WORK_GDB, "stops_raw")
    make_stops_fc(stops_df, stops_raw, STOPS_CRS)

    # Project stops
    stops_proj = fgdb_path(WORK_GDB, "stops_proj")
    logging.info("Projecting stops → %s …", TARGET_CRS)
    safe_project_or_copy(stops_raw, stops_proj, TARGET_CRS)

    # Project roads
    roads_proj = fgdb_path(WORK_GDB, "roads_proj")
    logging.info("Projecting roads …")
    safe_project_or_copy(ROADWAYS_PATH, roads_proj, TARGET_CRS)

    # Roadway schema
    logging.info("Mapping roadway fields …")
    col_map = map_road_fields(roads_proj)
    # Without a street-type field, strip no modifiers: falling back to FULLNAME
    # would treat whole road names as modifiers and blank them out.
    type_field = col_map.get("RW_TYPE_US")
    mods = modifiers_from_roads(roads_proj, type_field) if type_field else set()
    logging.info("Found %d modifiers.", len(mods))

    # Buffer stops
    stops_buf = fgdb_path(WORK_GDB, "stops_buf")
    logging.info("Buffering stops (%s %s) …", BUFFER_DISTANCE, BUFFER_DISTANCE_UNIT)
    buffer_fc(stops_proj, stops_buf, BUFFER_DISTANCE, BUFFER_DISTANCE_UNIT)

    # Spatial join
    join_fc = fgdb_path(WORK_GDB, "stops_roads_join")
    logging.info("SpatialJoin buffers ↔ roads …")
    spatial_join_fc(stops_buf, roads_proj, join_fc)

    # Build lookup dictionary
    stop2rd = stop_to_candidate_roads(join_fc, col_map["FULLNAME"], mods)

    # Detect typos
    logging.info("Running difflib matching …")
    typos = detect_typos(stops_df, stop2rd, mods, SIMILARITY_THRESHOLD)

    # Output. Always write, so a clean run replaces stale findings from an
    # earlier run with a header-only CSV.
    out_csv = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    typos.to_csv(out_csv, index=False)
    if typos.empty:
        logging.info("No potential typos found; wrote header-only CSV → %s", out_csv)
    else:
        logging.info("Wrote %d rows → %s", len(typos), out_csv)

    logging.info("All done. Workspace retained at %s for inspection.", WORK_GDB)
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Processing failed")
        sys.exit(1)
