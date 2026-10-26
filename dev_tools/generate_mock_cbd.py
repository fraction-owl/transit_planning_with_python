"""Generate mock central business district (CBD) polygons (zipped shapefile) for DC.

This is a companion to the mock GTFS / road / points-of-interest fixtures. It
writes one shapefile holding two polygons, so route-profiling tools can flag
whether each route serves the CBD without the repo shipping a real agency's
boundary:

    - Downtown: a rectangle with its southeast corner cut off on a diagonal,
      roughly the size of DC's CBD (about 2.4 by 1.9 miles).
    - Annex: a 2,000 ft square northeast of downtown.

Both outlines are hand-drawn; neither is traced from any published layer.

Coherence with the GTFS frame
-----------------------------
The mock DC feed from ``generate_mock_gtfs.py`` is hub-and-spoke: routes 10,
20, 30, and 40 all cross at the center of ``DEFAULT_BBOX_DC``, and the 50H
holiday loop's east leg runs through the same point. Every route would serve a
CBD that contains the hub, so a "serves the CBD" flag would never be False.
The downtown rectangle instead sits mostly northwest of the hub, and its cut
corner runs parallel to route 40 (NE-SW diagonal), 1,500 ft from it:

    - Routes 10, 20, 30, and 50H run through downtown (4 to 7 stops inside
      each) and stay more than a mile from the annex.
    - Route 40 stays 1,500 ft from downtown and cuts across the annex's
      southeast corner between two of its stops, so it passes through a CBD
      polygon without stopping in one.

The layout is built for a stops-inside test: routes 10, 20, 30, and 50H have a
stop in a CBD polygon and route 40 does not. A test that buffers stops or
counts route lines crossing a polygon would count route 40 as well.

The outlines' vertices are feet east/north of the hub. They are laid out in DC
State Plane (EPSG:2248) and written in WGS84, like the other mock layers.

Schema
------
``GIS_ID`` (the id field name DC's open-data CBD layer uses) and ``NAME``,
plus polygon geometry; one row per polygon.

Inputs:
    - None. The hub location and outlines are set in the CONFIGURATION block.

Outputs:
    - ``<OUTPUT_DIR>/mock_cbd_dc.zip`` containing the shapefile components at
      the archive's top level. The committed copy is
      ``tests/fixtures/mock_cbd_dc.zip``.

Configuration:
    - OUTPUT_DIR: Folder where the zipped layer is written.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, Polygon

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")
OUTPUT_FILENAME = "mock_cbd_dc.zip"

# Hub of the mock DC network: the center of DEFAULT_BBOX_DC in generate_mock_gtfs.py
# (-77.120, 38.790, -76.910, 39.000), where routes 10, 20, 30, and 40 cross.
HUB_LON = -77.015
HUB_LAT = 38.895

# CRS the outline is laid out in (DC State Plane, US feet) and the CRS written to disk.
# WGS84 output matches the other mock generators.
PROJECTED_CRS = "EPSG:2248"
OUTPUT_CRS = "EPSG:4326"

# CBD polygons as (GIS_ID, NAME, outline). Each outline is (feet east, feet north) of the
# hub, walked clockwise from the SW corner.
#   Downtown: routes 10 and 50H (x = 0) and 20 (y = 0) cross the rectangle and route 30
#   (y = -1.28 * x) cuts its northwest part. The diagonal edge runs parallel to route 40
#   (y = 1.28 * x), 1,500 ft northwest of it, so the hub itself is just outside the polygon.
#   Annex: route 40 (stops every 500 ft) cuts across the square's SE corner for about
#   240 ft, midway between two of its stops, which stay about 110 ft outside the square.
CBD_POLYGONS: tuple[tuple[str, str, tuple[tuple[float, float], ...]], ...] = (
    (
        "MOCK_CBD_1",
        "Mock Central Business District",
        (
            (-9000.0, -3000.0),  # SW corner
            (-9000.0, 7000.0),  # NW corner
            (3550.0, 7000.0),  # NE corner, then the diagonal down past the hub
            (-4250.0, -3000.0),  # back to the south edge
        ),
    ),
    (
        "MOCK_CBD_2",
        "Mock Central Business District Annex",
        (
            (5750.0, 9730.0),  # SW corner
            (5750.0, 11730.0),  # NW corner
            (7750.0, 11730.0),  # NE corner
            (7750.0, 9730.0),  # SE corner, just across route 40
        ),
    ),
)

LOG_LEVEL = logging.INFO

# ==================================================================================================
# BUILD / WRITE
# ==================================================================================================


def build_cbd_gdf(
    hub_lon: float = HUB_LON,
    hub_lat: float = HUB_LAT,
    polygons: Sequence[tuple[str, str, Sequence[tuple[float, float]]]] = CBD_POLYGONS,
    projected_crs: str = PROJECTED_CRS,
    output_crs: str = OUTPUT_CRS,
) -> gpd.GeoDataFrame:
    """Build the CBD GeoDataFrame, one row per polygon, from offsets around the hub.

    Args:
        hub_lon: Hub longitude (WGS84) the offsets are measured from.
        hub_lat: Hub latitude (WGS84) the offsets are measured from.
        polygons: ``(GIS_ID, NAME, outline)`` per polygon, each outline a sequence of
            ``(feet east, feet north)`` vertices relative to the hub.
        projected_crs: Foot-based projected CRS the offsets are applied in.
        output_crs: CRS of the returned GeoDataFrame.

    Returns:
        A GeoDataFrame with columns ``["GIS_ID", "NAME", "geometry"]``.

    Raises:
        ValueError: An outline does not form a valid polygon.
    """
    hub = gpd.GeoSeries([Point(hub_lon, hub_lat)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]
    records: list[dict[str, object]] = []
    for gis_id, name, outline_ft in polygons:
        polygon = Polygon([(hub.x + dx, hub.y + dy) for dx, dy in outline_ft])
        if not polygon.is_valid:
            raise ValueError(f"The {gis_id} outline does not form a valid polygon.")
        records.append({"GIS_ID": gis_id, "NAME": name, "geometry": polygon})

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=projected_crs)
    return gdf.to_crs(output_crs)


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


def generate(output_dir: str | Path, output_filename: str = OUTPUT_FILENAME) -> Path:
    """Generate the mock CBD layer as a zipped shapefile under *output_dir*.

    Returns:
        Path to the written zip.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = build_cbd_gdf()
    out_zip = output_dir / output_filename
    write_layer_zip(gdf, out_zip)
    area_sq_mi = gdf.to_crs(PROJECTED_CRS).geometry.area.sum() / 5280.0**2
    logging.info("Wrote %s (%d polygons, %.2f sq mi)", out_zip, len(gdf), area_sq_mi)
    return out_zip


def main() -> int:
    """Command-line entry point. Reads paths from the CONFIGURATION block."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if "Path\\To\\Your" in str(OUTPUT_DIR):
        logging.error("Set OUTPUT_DIR in the CONFIGURATION block before running.")
        return 2
    generate(OUTPUT_DIR)
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
