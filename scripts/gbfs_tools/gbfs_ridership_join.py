"""Join Capital Bikeshare ridership totals onto station geometries for mapping.

This script bridges the two other tools in ``gbfs_tools``:

* :mod:`gbfs_stations_exporter` writes station point geometries
  (``gbfs_stations.geojson`` and ``gbfs_stations.shp``).
* :mod:`bikeshare_ridership_trends` writes ridership summaries, including
  ``monthly_station_ridership.csv`` (one row per month and station).

It aggregates the per-month ridership to per-station totals and joins them onto
the station geometries, writing *new* GeoJSON and/or Shapefile versions that
carry departures, arrivals, and total trip attributes. Those enriched layers
are ready to drop into a GIS for ridership maps (proportional symbols,
choropleths, and so on).

Both inputs and outputs use EPSG:4326 (WGS 84), inherited from the source
geometries.

Join keys: Capital Bikeshare trip histories identify stations by the number
that GBFS publishes as ``short_name`` (e.g. ``31000``); the GBFS
``station_id`` is a different, opaque identifier. The geometry side therefore
joins on ``GEOMETRY_ID_FIELD`` (default ``short_name``) and the ridership side
on ``RIDERSHIP_ID_FIELD`` (default ``station_id``, the column written by
``bikeshare_ridership_trends``). Identifiers are compared as strings exactly as
written (only surrounding whitespace is trimmed), unmatched ids are logged
before stations are zero-filled, and a join that matches no ids at all is an
error rather than an all-zero output.

Outputs:
    - ``<stem>_ridership.geojson`` / ``<stem>_ridership.shp`` in ``OUTPUT_DIR``
      (e.g. ``gbfs_stations_ridership.geojson``): one enriched copy of each
      geometry input, carrying departures, arrivals, and total trip columns.

Typical usage (edit the CONFIG block, then run):

    python gbfs_ridership_join.py

Every CONFIG value also has a matching command-line flag that overrides it, e.g.

    python gbfs_ridership_join.py \
        --ridership-input output/monthly_station_ridership.csv \
        --geojson-input output/gbfs_stations.geojson --output-dir output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import geopandas as gpd
import pandas as pd

# === BEGIN CONFIG ===
RIDERSHIP_INPUT: str = "output/monthly_station_ridership.csv"
GEOJSON_INPUT: Optional[str] = "output/gbfs_stations.geojson"
SHAPEFILE_INPUT: Optional[str] = "output/gbfs_stations.shp"
OUTPUT_DIR: str = "output"
# Station key on the geometry layer. Capital Bikeshare trip station numbers
# match the GBFS ``short_name``, not the GBFS ``station_id``.
GEOMETRY_ID_FIELD: str = "short_name"
# Station key in the ridership CSV (``bikeshare_ridership_trends`` output).
RIDERSHIP_ID_FIELD: str = "station_id"
# === END CONFIG ===

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Ridership measures summed when collapsing months to per-station totals.
#: These match the count columns of ``monthly_station_ridership.csv``.
RIDERSHIP_MEASURES: tuple[str, ...] = ("departures", "arrivals", "total")


#: How many unmatched ids to list in a log message before truncating.
_MAX_IDS_LOGGED = 20


def _normalize_id(value: object) -> str | None:
    """Return the string join key for one identifier value.

    Strings are kept exactly as written apart from surrounding whitespace, so
    ``"001"``, ``"NA"`` and ``"1.0"`` all stay distinct. Integers (and
    integral floats, which is how a numeric attribute column with gaps comes
    back from a GIS file) are rendered without a decimal part. Missing and
    blank values return ``None`` and never match.

    Args:
        value: Any identifier-like value.

    Returns:
        The normalized key, or ``None`` for a missing/blank id.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _join_keys(values: pd.Series, label: str) -> pd.Series:
    """Normalize a column of ids and verify distinct ids stay distinct.

    Args:
        values: Raw identifier column.
        label: Description of the column, used in error messages.

    Returns:
        The normalized keys (``None`` for missing ids), aligned to *values*.

    Raises:
        ValueError: If two different raw ids normalize to the same key.
    """
    keys = values.map(_normalize_id)
    pairs = pd.DataFrame({"raw": values.astype(str), "key": keys}).dropna(subset=["key"])
    raw_per_key = pairs.drop_duplicates().groupby("key")["raw"].agg(list)
    collisions = raw_per_key[raw_per_key.map(len) > 1]
    if not collisions.empty:
        detail = "; ".join(f"{key!r} <- {raws}" for key, raws in collisions.head(5).items())
        raise ValueError(f"Distinct ids in {label} normalize to the same join key: {detail}")
    return keys


def _format_ids(ids: Iterable[str]) -> str:
    """Return a short, sorted, comma-separated listing of ids for logging."""
    ordered = sorted(ids)
    listing = ", ".join(ordered[:_MAX_IDS_LOGGED])
    if len(ordered) > _MAX_IDS_LOGGED:
        listing += f", ... ({len(ordered) - _MAX_IDS_LOGGED} more)"
    return listing


def aggregate_station_totals(
    ridership: pd.DataFrame, id_field: str = RIDERSHIP_ID_FIELD
) -> pd.DataFrame:
    """Collapse per-month station ridership to per-station totals.

    Rows are grouped by ``id_field`` and the ridership measures
    (:data:`RIDERSHIP_MEASURES`) are summed. A ``station_name`` column, if
    present, is carried through using its first nonblank value per station.
    Input that is already aggregated (no ``month`` column) passes through
    unchanged in shape. Ids are grouped exactly as they appear, so ids should
    be read as strings (see :func:`load_station_ridership`).

    Args:
        ridership: Station ridership table, e.g. ``monthly_station_ridership``.
        id_field: Column identifying the station.

    Returns:
        A DataFrame with one row per station and summed ridership measures.
    """
    if id_field not in ridership.columns:
        raise KeyError(
            f"Ridership table is missing the join column {id_field!r}; "
            f"found columns: {list(ridership.columns)}"
        )
    measures = [c for c in RIDERSHIP_MEASURES if c in ridership.columns]
    ridership = ridership.copy()
    for measure in measures:
        ridership[measure] = pd.to_numeric(ridership[measure], errors="raise")
    aggregations: dict[str, str] = {measure: "sum" for measure in measures}
    if "station_name" in ridership.columns:
        # "first" skips missing values only, so treat blank names as missing.
        names = ridership["station_name"].astype("string").str.strip()
        ridership["station_name"] = names.mask(names == "")
        aggregations["station_name"] = "first"
    # dropna=False so rows with a missing id are not silently discarded here;
    # join_ridership reports them as unjoinable instead.
    totals = ridership.groupby(id_field, as_index=False, dropna=False).agg(aggregations)
    for measure in measures:
        totals[measure] = totals[measure].fillna(0).astype(int)
    return totals


def load_station_ridership(path: str | Path, id_field: str = RIDERSHIP_ID_FIELD) -> pd.DataFrame:
    """Load a station ridership CSV and aggregate it to per-station totals.

    Every column is read as text with NA detection off, so ids such as
    ``"001"`` or ``"NA"`` survive intact; ridership measures are converted to
    numbers afterwards.

    Args:
        path: Path to ``monthly_station_ridership.csv`` (or an already
            aggregated station ridership CSV).
        id_field: Column identifying the station.

    Returns:
        A DataFrame with one row per station, ready to join onto geometries.
    """
    ridership = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for measure in RIDERSHIP_MEASURES:
        if measure in ridership.columns:
            ridership[measure] = pd.to_numeric(ridership[measure].replace("", "0"))
    return aggregate_station_totals(ridership, id_field)


def join_ridership(
    stations: gpd.GeoDataFrame,
    ridership: pd.DataFrame,
    geometry_id_field: str = GEOMETRY_ID_FIELD,
    ridership_id_field: str = RIDERSHIP_ID_FIELD,
) -> gpd.GeoDataFrame:
    """Join per-station ridership totals onto station geometries.

    Keys on both sides are normalized with :func:`_normalize_id`, which keeps
    string ids as written and refuses normalizations that would merge distinct
    ids. The merge is a left join validated as many-to-one against unique
    ridership keys, so a station feature can never be duplicated. Every station
    geometry is kept; unmatched ids on either side are logged, and stations
    with no ridership get zero-filled measures.

    Args:
        stations: Station point geometries (from ``gbfs_stations_exporter``).
        ridership: Per-station ridership totals.
        geometry_id_field: Station key column on the geometry layer.
        ridership_id_field: Station key column in the ridership table.

    Returns:
        A GeoDataFrame of stations enriched with ridership attributes.

    Raises:
        KeyError: If either side lacks its join column.
        ValueError: If ids collide after normalization, geometry keys are
            duplicated, or no ridership id matches any station.
    """
    if geometry_id_field not in stations.columns:
        raise KeyError(
            f"Station layer is missing the join column {geometry_id_field!r}; "
            f"found columns: {list(stations.columns)}"
        )
    if ridership_id_field not in ridership.columns:
        raise KeyError(
            f"Ridership table is missing the join column {ridership_id_field!r}; "
            f"found columns: {list(ridership.columns)}"
        )
    stations = stations.copy()
    ridership = ridership.copy()
    stations["_join_key"] = _join_keys(
        stations[geometry_id_field], f"station layer column {geometry_id_field!r}"
    )
    ridership["_join_key"] = _join_keys(
        ridership[ridership_id_field], f"ridership column {ridership_id_field!r}"
    )

    duplicated_geometry = stations["_join_key"].dropna()
    duplicated_geometry = duplicated_geometry[duplicated_geometry.duplicated()]
    if not duplicated_geometry.empty:
        raise ValueError(
            f"Station layer has duplicate {geometry_id_field!r} values, so ridership "
            f"would be counted on several features: {_format_ids(set(duplicated_geometry))}"
        )
    missing_geometry_keys = int(stations["_join_key"].isna().sum())
    if missing_geometry_keys:
        logger.warning(
            "%d station feature(s) have a blank %r and cannot receive ridership.",
            missing_geometry_keys,
            geometry_id_field,
        )
    ridership = ridership.dropna(subset=["_join_key"])
    if ridership["_join_key"].duplicated().any():
        raise ValueError(
            f"Ridership table has more than one row per {ridership_id_field!r}; "
            "aggregate it to per-station totals before joining."
        )

    geometry_keys = set(stations["_join_key"].dropna())
    ridership_keys = set(ridership["_join_key"])
    matched = geometry_keys & ridership_keys
    unmatched_ridership = ridership_keys - geometry_keys
    if ridership_keys and not matched:
        raise ValueError(
            f"No ridership {ridership_id_field!r} value matches any station "
            f"{geometry_id_field!r} value. For Capital Bikeshare, trip station numbers "
            "match the GBFS 'short_name', not 'station_id'. Ridership ids: "
            f"{_format_ids(ridership_keys)}; station ids: {_format_ids(geometry_keys)}"
        )
    if unmatched_ridership:
        lost = ridership["_join_key"].isin(unmatched_ridership)
        lost_trips = int(ridership.loc[lost, "total"].sum()) if "total" in ridership else 0
        logger.warning(
            "%d ridership station id(s) (%d total trips) have no matching station "
            "feature and are left off the map: %s",
            len(unmatched_ridership),
            lost_trips,
            _format_ids(unmatched_ridership),
        )
    unmatched_geometry = geometry_keys - ridership_keys
    if unmatched_geometry:
        logger.info(
            "%d station feature(s) have no ridership and are zero-filled: %s",
            len(unmatched_geometry),
            _format_ids(unmatched_geometry),
        )

    # Drop the id column from the right side; keep the geometry's.
    ridership = ridership.drop(columns=[ridership_id_field])
    # Let ridership win for any other overlapping non-key column (e.g. name).
    overlap = [c for c in ridership.columns if c != "_join_key" and c in stations.columns]
    stations = stations.drop(columns=overlap)
    merged = stations.merge(ridership, on="_join_key", how="left", validate="many_to_one")
    merged = merged.drop(columns=["_join_key"])
    measures = [c for c in RIDERSHIP_MEASURES if c in merged.columns]
    for measure in measures:
        merged[measure] = merged[measure].fillna(0).astype(int)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=stations.crs)


def export_layer(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write a GeoDataFrame, choosing the driver from the file extension.

    Args:
        gdf: The enriched station GeoDataFrame.
        output_path: Destination ``.geojson`` or ``.shp`` path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON", index=False)
    else:
        gdf.to_file(output_path, driver="ESRI Shapefile", index=False)


def _joined_output_path(input_path: str, output_dir: Path) -> Path:
    """Return the ``*_ridership`` output path for a given geometry input.

    Args:
        input_path: Path to a source geometry file.
        output_dir: Directory for the joined output.

    Returns:
        ``<output_dir>/<stem>_ridership<suffix>``.
    """
    source = Path(input_path)
    return output_dir / f"{source.stem}_ridership{source.suffix}"


def run(
    ridership_input: str | None = None,
    geojson_input: str | None = None,
    shapefile_input: str | None = None,
    output_dir: str | Path | None = None,
    geometry_id_field: str | None = None,
    ridership_id_field: str | None = None,
) -> None:
    """Run the ridership-to-geometry join end to end.

    Unset args fall back to the CONFIG block at the top of this file, so
    ``m.RIDERSHIP_INPUT = ...; m.run()`` works after a plain import. Pass an
    empty string for ``geojson_input`` or ``shapefile_input`` to skip that
    output.
    """
    ridership_input = RIDERSHIP_INPUT if ridership_input is None else ridership_input
    geojson_input = GEOJSON_INPUT if geojson_input is None else geojson_input
    shapefile_input = SHAPEFILE_INPUT if shapefile_input is None else shapefile_input
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    geometry_id_field = GEOMETRY_ID_FIELD if geometry_id_field is None else geometry_id_field
    ridership_id_field = RIDERSHIP_ID_FIELD if ridership_id_field is None else ridership_id_field

    if not geojson_input and not shapefile_input:
        raise ValueError("Set GEOJSON_INPUT and/or SHAPEFILE_INPUT in the CONFIG block.")
    ridership = load_station_ridership(ridership_input, ridership_id_field)
    output_dir = Path(output_dir)
    for geometry_input in (geojson_input, shapefile_input):
        if not geometry_input:
            continue
        stations = gpd.read_file(geometry_input)
        joined = join_ridership(stations, ridership, geometry_id_field, ridership_id_field)
        output_path = _joined_output_path(geometry_input, output_dir)
        export_layer(joined, output_path)
        logger.info("Joined ridership onto %d stations -> %s", len(joined), output_path)
    logger.info("Script completed successfully.")


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
    """Parse command-line arguments, defaulting to the CONFIG block values."""
    parser = argparse.ArgumentParser(
        description=(
            "Join Capital Bikeshare ridership totals onto station geometries. "
            "Defaults come from the CONFIG block at the top of this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ridership-input", default=RIDERSHIP_INPUT, help="Per-station ridership CSV."
    )
    parser.add_argument(
        "--geojson-input",
        default=GEOJSON_INPUT,
        help="Station GeoJSON to enrich (empty string to skip).",
    )
    parser.add_argument(
        "--shapefile-input",
        default=SHAPEFILE_INPUT,
        help="Station Shapefile to enrich (empty string to skip).",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for joined outputs.")
    parser.add_argument(
        "--geometry-id-field",
        default=GEOMETRY_ID_FIELD,
        help="Station key on the geometry layer (GBFS short_name for Capital Bikeshare).",
    )
    parser.add_argument(
        "--ridership-id-field",
        default=RIDERSHIP_ID_FIELD,
        help="Station key column in the ridership CSV.",
    )
    return parser.parse_args(notebook_safe_argv(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Defaults fall back to the CONFIG block.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)
    try:
        run(
            ridership_input=args.ridership_input,
            geojson_input=args.geojson_input,
            shapefile_input=args.shapefile_input,
            output_dir=args.output_dir,
            geometry_id_field=args.geometry_id_field,
            ridership_id_field=args.ridership_id_field,
        )
    except (OSError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    # Strict parsing; in a notebook, notebook_safe_argv() keeps the kernel's
    # injected argv away from argparse so the CONFIG block stays in charge.
    raise SystemExit(main())
