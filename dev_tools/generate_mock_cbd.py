"""Generate a mock central business district (CBD) polygon (zipped shapefile) for DC.

This is a companion to the mock GTFS / road / points-of-interest fixtures. It
writes a single polygon shaped like a downtown boundary drawn along streets: a
straight west edge, an avenue cutting diagonally across the north side, and a
stepped south edge. Route-profiling tools can use it to flag whether each route
serves the CBD without the repo shipping a real agency's boundary. The outline
is hand-drawn and roughly the size of DC's CBD (about 2.3 by 1.9 miles); it is
not traced from any published layer.

Coherence with the GTFS frame
-----------------------------
The mock DC feed from ``generate_mock_gtfs.py`` is hub-and-spoke: routes 10,
20, 30, and 40 all cross at the center of ``DEFAULT_BBOX_DC``, and the 50H
holiday loop's east leg runs through the same point. Every route would serve a
CBD drawn around that hub, so a "serves the CBD" flag would never be False.
The polygon is drawn in the hub's northwest quadrant instead, where:

    - Route 30 (NW-SE crosstown) runs diagonally through it.
    - Route 50H (holiday loop) runs through it on its west leg.
    - Routes 10, 20, and 40 stay more than half a mile away (about 0.6 mile at
      the closest), so they are not counted even with a half-mile buffer.

The outline's vertices are feet east/north of the hub. They are laid out in DC
State Plane (EPSG:2248) and written in WGS84, like the other mock layers.

Schema
------
``GIS_ID`` (the id field name DC's open-data CBD layer uses) and ``NAME``,
plus polygon geometry.

Inputs:
    - None. The hub location and outline are set in the CONFIGURATION block.

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

# CBD outline as (feet east, feet north) of the hub, walked clockwise from the SW corner.
# Route 30 runs along y = -1.28 * x through this quadrant and route 50H's west leg along
# x = -13,440 ft; routes 10 (x = 0) and 20 (y = 0) pass 3,300 ft from the nearest edges.
CBD_OUTLINE_FT: tuple[tuple[float, float], ...] = (
    (-15400.0, 6200.0),  # SW corner
    (-15400.0, 13400.0),  # NW corner (straight west edge)
    (-12800.0, 13400.0),  # short north edge, then the diagonal avenue
    (-7200.0, 11000.0),  # bend in the avenue
    (-3300.0, 8600.0),  # NE corner
    (-3300.0, 3300.0),  # SE corner (straight east edge)
    (-7500.0, 3300.0),  # south edge, eastern section
    (-7500.0, 4000.0),  # jog
    (-9600.0, 4000.0),
    (-9600.0, 6200.0),  # step up to the western section of the south edge
)

CBD_GIS_ID = "MOCK_CBD_1"
CBD_NAME = "Mock Central Business District"

LOG_LEVEL = logging.INFO

# ==================================================================================================
# BUILD / WRITE
# ==================================================================================================


def build_cbd_gdf(
    hub_lon: float = HUB_LON,
    hub_lat: float = HUB_LAT,
    outline_ft: Sequence[tuple[float, float]] = CBD_OUTLINE_FT,
    projected_crs: str = PROJECTED_CRS,
    output_crs: str = OUTPUT_CRS,
) -> gpd.GeoDataFrame:
    """Build the one-row CBD GeoDataFrame from *outline_ft* offsets around the hub.

    Args:
        hub_lon: Hub longitude (WGS84) the offsets are measured from.
        hub_lat: Hub latitude (WGS84) the offsets are measured from.
        outline_ft: Polygon vertices as ``(feet east, feet north)`` of the hub.
        projected_crs: Foot-based projected CRS the offsets are applied in.
        output_crs: CRS of the returned GeoDataFrame.

    Returns:
        A GeoDataFrame with columns ``["GIS_ID", "NAME", "geometry"]``.

    Raises:
        ValueError: The outline does not form a valid polygon.
    """
    hub = gpd.GeoSeries([Point(hub_lon, hub_lat)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]
    polygon = Polygon([(hub.x + dx, hub.y + dy) for dx, dy in outline_ft])
    if not polygon.is_valid:
        raise ValueError("CBD_OUTLINE_FT does not form a valid polygon (check vertex order).")

    gdf = gpd.GeoDataFrame(
        {"GIS_ID": [CBD_GIS_ID], "NAME": [CBD_NAME]}, geometry=[polygon], crs=projected_crs
    )
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
    area_sq_mi = gdf.to_crs(PROJECTED_CRS).geometry.iloc[0].area / 5280.0**2
    logging.info("Wrote %s (1 polygon, %.2f sq mi)", out_zip, area_sq_mi)
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
