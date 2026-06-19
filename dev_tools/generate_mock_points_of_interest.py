"""Generate mock points-of-interest layers (zipped shapefiles) for the DC area.

This is the destination-side companion to the mock GTFS / road fixtures. It
produces one zipped point shapefile per strategic-site category so the service
coverage tools can be exercised end to end against realistic-looking DC data
without shipping any real (and licensing-encumbered) open-data extracts:

    - ``scripts/service_coverage/points_of_interest_coverage_gpd.py`` counts the
      POIs each GTFS route reaches (its ``LAYER_SPECS`` lists exactly the
      filenames and id columns written here).
    - ``scripts/service_coverage/school_coverage_by_route_gpd.py`` consumes the
      same kind of school point layer.

Coherence with the GTFS frame
-----------------------------
Coverage is only interesting when some POIs fall inside a route's ¼-mile buffer
and some fall outside it. To guarantee that spread, point ``GTFS_DIR`` at a mock
feed (e.g. the ``dc/`` folder from ``generate_mock_gtfs.py``): each layer's
features are then placed at controlled offsets from that feed's stops -- a
deterministic ``coverage_fraction`` of them within the buffer distance and the
rest one to three buffer-widths out. With no feed available the points fall back
to a regular grid inside ``DEFAULT_BBOX_DC`` (the same bbox the GTFS and road
generators use), so the output still lands in the right place geographically.

Determinism
-----------
Placement is hashed from the layer name and feature index (no RNG), so repeated
runs reproduce identical geometry.

Schema
------
Each layer carries a single attribute -- its id column -- plus point geometry,
which is all the coverage tools read. Id columns reuse the source data's own
field names where one is established (``DESCRIPTIO`` for medical/library
facilities, ``SCHOOL_NAM`` for schools, ``NAME`` for rail) and default to
``NAME`` for the newer categories.

Outputs
-------
For each layer, ``<OUTPUT_DIR>/<Layer_Name>.zip`` containing the shapefile
components at the archive's top level, plus a ``points_of_interest_manifest.csv``
summarizing every layer's feature count and coverage split.

Configuration:
    - OUTPUT_DIR: Folder where the zipped layers and manifest are written.
    - GTFS_DIR: Optional folder holding a mock GTFS feed (a ``stops.txt``). When
      set, POIs are anchored to its stops; otherwise a bbox grid is used.
"""

from __future__ import annotations

import hashlib
import logging
import math
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

# Optional path to a mock GTFS feed folder (containing stops.txt). When readable,
# POIs are placed relative to the feed's stops so route buffers actually capture a
# meaningful subset of them. Leave as None to scatter POIs on a bbox grid instead.
GTFS_DIR: Path | None = None

# Shared DC frame (WGS84), kept in sync with generate_mock_gtfs.py / generate_mock_roads.py.
DEFAULT_BBOX_DC = (-77.120, 38.790, -76.910, 39.000)

# Inset the grid-fallback drawing area inside the bbox so points avoid the edges.
BBOX_INSET_FRACTION = 0.05

# Output CRS for every layer. WGS84 matches the other mock generators; the coverage
# tools reproject to a projected CRS themselves.
OUTPUT_CRS = "EPSG:4326"

# Route buffer the coverage tool applies (¼ mile). Used here only to decide which
# POIs land "inside" vs "outside" a buffer, so the demo shows a real spread.
BUFFER_DIST_FT = 1320.0

# Earth model + unit conversion (matched to the other generators' spherical model).
EARTH_RADIUS_M = 6_371_000.0
FEET_PER_METER = 3.28084

# Side length of the square grid used when no GTFS feed is available for anchoring.
FALLBACK_GRID_SIZE = 8

# Manifest documenting the generated layers.
WRITE_MANIFEST = True
MANIFEST_FILENAME = "points_of_interest_manifest.csv"

LOG_LEVEL = logging.INFO

#: bbox as (min_lon, min_lat, max_lon, max_lat).
Bbox = tuple[float, float, float, float]

# ==================================================================================================
# POI LAYER REGISTRY
# ==================================================================================================


@dataclass(frozen=True)
class PoiLayer:
    """One mock POI layer.

    Attributes:
        filename: Output shapefile name (also the zip stem), e.g. ``"Libraries.shp"``.
        id_col: The single attribute column carried by the layer.
        label: Human-readable prefix for each feature's id value.
        count: Number of point features to generate.
        coverage_fraction: Fraction of features placed within the route buffer of
            their anchor stop (the remainder land one to three buffer-widths out).
    """

    filename: str
    id_col: str
    label: str
    count: int
    coverage_fraction: float = 0.6


# Order/filenames/id columns mirror points_of_interest_coverage_gpd.LAYER_SPECS.
POI_LAYERS: tuple[PoiLayer, ...] = (
    PoiLayer("Hospitals.shp", "DESCRIPTIO", "MOCK HOSPITAL", 12, 0.60),
    PoiLayer("Urgent_Care_Facilities.shp", "DESCRIPTIO", "MOCK URGENT CARE", 20, 0.55),
    PoiLayer("School_Facilities.shp", "SCHOOL_NAM", "Mock School", 40, 0.70),
    PoiLayer("Colleges_and_Universities.shp", "NAME", "Mock College", 10, 0.60),
    PoiLayer("Libraries.shp", "DESCRIPTIO", "MOCK LIBRARY", 15, 0.65),
    PoiLayer("Metrorail_Stations.shp", "NAME", "Mock Metro Station", 25, 0.80),
    PoiLayer("Commuter_Rail_Stations.shp", "NAME", "Mock Commuter Rail Station", 8, 0.70),
    PoiLayer("Park_and_Rides.shp", "NAME", "Mock Park and Ride", 12, 0.75),
    PoiLayer("Bus_Stations.shp", "NAME", "Mock Bus Station", 6, 0.85),
    PoiLayer("Private_Shuttle_Stops.shp", "NAME", "Mock Shuttle Stop", 18, 0.70),
)

# ==================================================================================================
# GEOMETRY HELPERS
# ==================================================================================================


def _stable_unit(*parts: object) -> float:
    """Return a deterministic float in ``[0, 1)`` derived from *parts* via SHA-256."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _inset_bbox(bbox: Bbox, fraction: float) -> Bbox:
    """Shrink *bbox* toward its center by *fraction* on each side."""
    min_lon, min_lat, max_lon, max_lat = bbox
    dx = (max_lon - min_lon) * fraction
    dy = (max_lat - min_lat) * fraction
    return (min_lon + dx, min_lat + dy, max_lon - dx, max_lat - dy)


def _offset_point(lon: float, lat: float, dist_m: float, bearing_rad: float) -> tuple[float, float]:
    """Return the ``(lon, lat)`` *dist_m* meters from ``(lon, lat)`` at *bearing_rad*.

    Bearing is measured clockwise from north. Uses a local equirectangular
    approximation, which is accurate to well within a meter at city scale.
    """
    dlat = (dist_m * math.cos(bearing_rad)) / EARTH_RADIUS_M
    dlon = (dist_m * math.sin(bearing_rad)) / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    return lon + math.degrees(dlon), lat + math.degrees(dlat)


def _grid_anchors(bbox: Bbox, n: int) -> list[tuple[float, float]]:
    """Return an ``n × n`` grid of cell-center ``(lon, lat)`` anchors within *bbox*."""
    min_lon, min_lat, max_lon, max_lat = bbox
    anchors: list[tuple[float, float]] = []
    for row in range(n):
        for col in range(n):
            fx = (col + 0.5) / n
            fy = (row + 0.5) / n
            anchors.append((min_lon + fx * (max_lon - min_lon), min_lat + fy * (max_lat - min_lat)))
    return anchors


def _read_stop_anchors(stops_path: Path) -> list[tuple[float, float]]:
    """Return ``(lon, lat)`` anchors from a GTFS ``stops.txt``, or ``[]`` if unusable."""
    if not stops_path.exists():
        return []
    stops = pd.read_csv(stops_path)
    if not {"stop_lat", "stop_lon"}.issubset(stops.columns):
        return []
    if "stop_id" in stops.columns:
        stops = stops.sort_values("stop_id")
    return [(float(lon), float(lat)) for lon, lat in zip(stops["stop_lon"], stops["stop_lat"])]


def _load_anchor_points(
    gtfs_dir: Path | None,
    bbox: Bbox,
    inset_fraction: float,
    grid_size: int,
) -> list[tuple[float, float]]:
    """Return anchor points: mock GTFS stops when available, else a bbox grid.

    Args:
        gtfs_dir: Optional folder holding a GTFS ``stops.txt``.
        bbox: Region bbox used for the grid fallback.
        inset_fraction: Inset applied to *bbox* before building the grid.
        grid_size: Side length of the fallback grid.

    Returns:
        A non-empty list of ``(lon, lat)`` anchor points.
    """
    if gtfs_dir is not None:
        pts = _read_stop_anchors(Path(gtfs_dir) / "stops.txt")
        if pts:
            logging.info("Anchoring POIs to %d mock GTFS stop(s) from %s", len(pts), gtfs_dir)
            return pts
        logging.warning("No usable stops.txt under %s; falling back to a bbox grid.", gtfs_dir)

    anchors = _grid_anchors(_inset_bbox(bbox, inset_fraction), grid_size)
    logging.info("Anchoring POIs to a %d×%d grid within the DC bbox.", grid_size, grid_size)
    return anchors


# ==================================================================================================
# LAYER BUILD / WRITE
# ==================================================================================================


def build_layer_gdf(
    layer: PoiLayer,
    anchors: Sequence[tuple[float, float]],
    buffer_m: float,
    crs: str = OUTPUT_CRS,
) -> gpd.GeoDataFrame:
    """Build the point GeoDataFrame for *layer*, placing features near *anchors*.

    The first ``round(count * coverage_fraction)`` features are offset within
    *buffer_m* of a (hashed) anchor stop, so a route buffer captures them; the
    rest land between 1.5 and 3 buffer-widths out. Anchor choice, bearing, and
    distance are all hash-derived, so the layer is reproducible.

    Args:
        layer: The layer definition.
        anchors: Candidate anchor ``(lon, lat)`` points (must be non-empty).
        buffer_m: Route buffer distance in meters.
        crs: Output CRS for the returned GeoDataFrame.

    Returns:
        A GeoDataFrame with columns ``[layer.id_col, "geometry"]``.
    """
    n_inside = round(layer.count * layer.coverage_fraction)
    records: list[dict[str, object]] = []
    for i in range(layer.count):
        idx = int(_stable_unit(layer.filename, "anchor", i) * len(anchors))
        anchor_lon, anchor_lat = anchors[min(idx, len(anchors) - 1)]
        bearing = _stable_unit(layer.filename, "bearing", i) * 2.0 * math.pi
        unit = _stable_unit(layer.filename, "dist", i)
        if i < n_inside:
            dist_m = 15.0 + unit * (buffer_m * 0.8 - 15.0)
        else:
            dist_m = buffer_m * 1.5 + unit * (buffer_m * 1.5)
        lon, lat = _offset_point(anchor_lon, anchor_lat, dist_m, bearing)
        records.append({layer.id_col: f"{layer.label} {i + 1:02d}", "geometry": Point(lon, lat)})

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    if crs != "EPSG:4326":
        gdf = gdf.to_crs(crs)
    return gdf


def write_layer_zip(gdf: gpd.GeoDataFrame, out_zip: Path) -> None:
    """Write *gdf* as a zipped shapefile with components at the archive's top level."""
    stem = out_zip.with_suffix("").name
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        gdf.to_file(tmp_dir / f"{stem}.shp", driver="ESRI Shapefile", index=False)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for comp in sorted(tmp_dir.glob(f"{stem}.*")):
                zf.write(comp, comp.name)


# ==================================================================================================
# DRIVER
# ==================================================================================================


def generate(
    output_dir: str | Path,
    gtfs_dir: Path | None = GTFS_DIR,
    bbox: Bbox = DEFAULT_BBOX_DC,
    layers: Sequence[PoiLayer] = POI_LAYERS,
    buffer_dist_ft: float = BUFFER_DIST_FT,
    output_crs: str = OUTPUT_CRS,
    inset_fraction: float = BBOX_INSET_FRACTION,
    grid_size: int = FALLBACK_GRID_SIZE,
    write_manifest: bool = WRITE_MANIFEST,
) -> list[dict[str, object]]:
    """Generate every mock POI layer as a zipped shapefile under *output_dir*.

    Returns:
        The manifest rows (one per layer), also written to CSV when
        *write_manifest* is True.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_m = buffer_dist_ft / FEET_PER_METER
    anchors = _load_anchor_points(gtfs_dir, bbox, inset_fraction, grid_size)

    manifest: list[dict[str, object]] = []
    for layer in layers:
        gdf = build_layer_gdf(layer, anchors, buffer_m, output_crs)
        out_zip = output_dir / f"{Path(layer.filename).stem}.zip"
        write_layer_zip(gdf, out_zip)
        n_inside = round(layer.count * layer.coverage_fraction)
        logging.info(
            "Wrote %s (%d feature(s), ~%d within buffer)", out_zip.name, layer.count, n_inside
        )
        manifest.append(
            {
                "layer": layer.filename,
                "id_col": layer.id_col,
                "features": layer.count,
                "within_buffer": n_inside,
                "beyond_buffer": layer.count - n_inside,
                "zip": out_zip.name,
            }
        )

    if write_manifest:
        manifest_path = output_dir / MANIFEST_FILENAME
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
        logging.info("Wrote manifest -> %s", manifest_path)

    return manifest


def main() -> int:
    """Command-line entry point. Reads paths from the CONFIGURATION block."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if "Path\\To\\Your" in str(OUTPUT_DIR):
        logging.error("Set OUTPUT_DIR in the CONFIGURATION block before running.")
        return 1
    generate(OUTPUT_DIR)
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
