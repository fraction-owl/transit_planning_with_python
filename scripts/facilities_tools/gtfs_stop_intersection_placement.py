"""Classify GTFS stop placement relative to inferred road intersections.

Inputs:
    GTFS directory/ZIP with shapes, stops, trips and stop_times; road SHP/ZIP.
    Optional --footprints-path SHP/ZIP with unique INT_ID, optional STREETS and
    GRADE_SEP (N = verified at grade, Y = grade separation present, U/blank = unknown).
    Digitize the intersection extent up to its approach/departure boundaries.
    Footprint mode measures to the next entry and from the previous exit.
    Stops inside a traversal are within_footprint. Nearby unmapped road junctions
    trigger coverage review. Without --footprints-complete, midblock suggestions
    require review even where road topology shows no missing junction.
    Edit CONFIGURATION or pass CLI options. No network services are used.

Outputs:
    Stop summary and shape-specific detail CSVs, plain XLSX, GIS layers,
    overview PNG and reproducible run logs. Distances are feet.

Method:
    Infer junctions from road endpoints (never arbitrary line crossings),
    requiring at least three distinct arms and two street names. Project
    junctions onto directed shapes, checking approach/departure arm headings.
    Measure along shapes to previous/next junction centerline locations.
    Multiple passes, large offsets, bridge nodes, missing context and
    terminal-name matches are explicitly flagged. This is a screening draft,
    not a curb/crosswalk survey or a safety score. No full road map matching.

Limitations:
    Requires roads split at real junctions; un-noded T intersections are missed.
    Endpoint topology cannot independently establish grade separation.
    Named driveways count unless excluded by road class; unnamed ones do not.
    Divided junction nodes are merged only when their street-name sets match.
    GTFS service dates are not filtered; all supplied trips contribute patterns.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List, Optional

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from openpyxl import Workbook
from pyproj import CRS
from shapely import get_coordinates
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===
GTFS_PATH: str = r"Path/To/GTFS.zip"  # Directory or ZIP, including wrapper folder.
ROADS_PATH: str = r"Path/To/Roadways.zip"  # SHP or ZIP.
OUTPUT_DIR: str = r"Path/To/Output"
BBOX: list[float] | None = None  # west, south, east, north; WGS84. None = served extent.
PROJECTED_CRS: str = "EPSG:2283"
ROAD_NAME_FIELD: str = "FULLNAME"
ROAD_CLASS_FIELD: str = "ROAD_CLASS"
EXCLUDE_ROAD_CLASSES: list[str] = ["PRK", "DRV"]
BRIDGE_FIELD: str = "BRIDGE"  # Empty disables bridge flag; any value other than N/0 flags.
FOOTPRINTS_PATH: str | None = None  # Optional polygon SHP or ZIP.
FOOTPRINTS_MEMBER: str | None = None
FOOTPRINTS_COMPLETE: bool = False  # Set True only for a surveyed, complete study area.
ROADS_MEMBER: str | None = None
NEAR_DISTANCE_FT: float = 150.0  # Exploratory threshold, not an agency standard.
INTERSECTION_ZONE_FT: float = 30.0  # Uncertain relative to junction centerline.
CONTEXT_BUFFER_FT: float = 1500.0
NODE_TOLERANCE_FT: float = 3.0
JUNCTION_MERGE_FT: float = 120.0
MAX_OFFSET_FT: float = 80.0
REVIEW_OFFSET_FT: float = 40.0
HEADING_SAMPLE_FT: float = 60.0
HEADING_TOLERANCE_DEG: float = 40.0
TERMINAL_PATTERN: str = r"\bBAY\b|PARK AND RIDE|PARK & RIDE"
REQUIRE_RUN_LOG: bool = True
# === END CONFIG ===


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


def read_gtfs(path: Path, filename: str) -> pd.DataFrame:
    """Read identifiers as strings and reject ambiguous wrapper directories."""
    if path.is_dir():
        matches = list(path.rglob(filename))
        if len(matches) != 1:
            raise ValueError(f"Expected one {filename}; found {len(matches)} in {path}")
        return pd.read_csv(matches[0], dtype=str, keep_default_na=False)
    with zipfile.ZipFile(path) as archive:
        matches = [
            n
            for n in archive.namelist()
            if Path(n).name == filename and not n.startswith("__MACOSX/")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {filename}; found {len(matches)} in {path}")
        with archive.open(matches[0]) as stream:
            return pd.read_csv(stream, dtype=str, keep_default_na=False)


def bearing(a: Point, b: Point) -> float:
    """Return a mathematical angle in degrees, measured counterclockwise from east."""
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 360


def angle_difference(a: float, b: float) -> float:
    """Return the unsigned shortest angular difference."""
    return abs((a - b + 180) % 360 - 180)


def projection_candidates(line: LineString, point: Point, tolerance: float) -> list[float]:
    """Find distinct local passes near a point without silently picking a loop pass."""
    coords = np.asarray(line.coords, dtype=float)[:, :2]
    vectors = coords[1:] - coords[:-1]
    squared = np.sum(vectors * vectors, axis=1)
    lengths = np.sqrt(squared)
    fractions = np.divide(
        np.sum((np.array([point.x, point.y]) - coords[:-1]) * vectors, axis=1),
        squared,
        out=np.zeros_like(squared),
        where=squared > 0,
    )
    fractions = np.clip(fractions, 0, 1)
    projected = coords[:-1] + vectors * fractions[:, None]
    distances = np.sqrt(np.sum((projected - np.array([point.x, point.y])) ** 2, axis=1))
    cumulative = np.r_[0, np.cumsum(lengths[:-1])]
    limit = min(tolerance, float(distances.min()) + 3)
    candidates = (cumulative + lengths * fractions)[(distances <= limit) & (lengths > 0)]
    groups: list[list[float]] = []
    for station in sorted(candidates):
        if groups and station - groups[-1][-1] <= 20:
            groups[-1].append(station)
        else:
            groups.append([station])
    return [min(group, key=lambda s: line.interpolate(s).distance(point)) for group in groups]


def infer_junctions(roads: gpd.GeoDataFrame, args: argparse.Namespace) -> gpd.GeoDataFrame:
    """Cluster existing endpoints and require three distinct arms and two street names."""
    endpoints: list[Point] = []
    endpoint_roads: list[int] = []
    for i, line in enumerate(roads.geometry):
        endpoints.extend([Point(line.coords[0]), Point(line.coords[-1])])
        endpoint_roads.extend([i, i])
    if not endpoints:
        raise ValueError("No usable roads within the analysis context.")
    tree = STRtree(endpoints)
    parent = list(range(len(endpoints)))

    def root(i: int) -> int:
        """Find a disjoint-set root with path compression."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(endpoints):
        for j in tree.query(p, predicate="dwithin", distance=args.node_tolerance_ft):
            parent[root(int(j))] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(endpoints)):
        groups.setdefault(root(i), []).append(i)
    records = []
    for members in groups.values():
        center = Point(
            np.mean([endpoints[i].x for i in members]),
            np.mean([endpoints[i].y for i in members]),
        )
        names: set[str] = set()
        arms: list[float] = []
        bridge = False
        for i in members:
            road = roads.iloc[endpoint_roads[i]]
            line = road.geometry
            name = str(road[args.road_name_field]).strip().upper()
            if name and name != "NAN":
                names.add(name)
            distance = min(args.heading_sample_ft, line.length / 2)
            sample = line.interpolate(distance if i % 2 == 0 else line.length - distance)
            heading = bearing(endpoints[i], sample)
            if all(angle_difference(heading, old) > 20 for old in arms):
                arms.append(heading)
            if args.bridge_field:
                bridge |= str(road[args.bridge_field]).strip().upper() not in (
                    "N",
                    "0",
                    "",
                    "NONE",
                )
        if len(arms) >= 3 and len(names) >= 2:
            records.append(
                {
                    "junction_id": f"J{len(records) + 1:05d}",
                    "street_names": " | ".join(sorted(names)),
                    "arms": json.dumps(arms),
                    "bridge_review": bridge,
                    "geometry": center,
                }
            )
    return gpd.GeoDataFrame(
        records,
        columns=["junction_id", "street_names", "arms", "bridge_review", "geometry"],
        geometry="geometry",
        crs=roads.crs,
    )


def shape_events(
    line: LineString, junctions: gpd.GeoDataFrame, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Find directed junction encounters supported by incoming and outgoing road arms."""
    events: list[dict[str, Any]] = []
    for index in junctions.sindex.query(line.buffer(args.max_offset_ft)):
        node = junctions.iloc[int(index)]
        for station in projection_candidates(line, node.geometry, args.max_offset_ft):
            if station < args.heading_sample_ft or line.length - station < args.heading_sample_ft:
                continue  # A truncated shape cannot establish both arms.
            center = line.interpolate(station)
            before = line.interpolate(station - args.heading_sample_ft)
            after = line.interpolate(station + args.heading_sample_ft)
            incoming = bearing(center, before)
            outgoing = bearing(center, after)
            arms = json.loads(node.arms)
            matches = [
                (i, j)
                for i, a in enumerate(arms)
                for j, b in enumerate(arms)
                if i != j
                and angle_difference(a, incoming) <= args.heading_tolerance_deg
                and angle_difference(b, outgoing) <= args.heading_tolerance_deg
            ]
            if not matches:
                continue
            events.append(
                {
                    "start": station,
                    "end": station,
                    "ids": node.junction_id,
                    "names": node.street_names,
                    "offset": center.distance(node.geometry),
                    "bridge": bool(node.bridge_review),
                }
            )
    merged: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda e: e["start"]):
        if (
            merged
            and event["start"] - merged[-1]["start"] <= args.junction_merge_ft
            and event["names"] == merged[-1]["names"]
        ):
            merged[-1]["end"] = event["end"]
            merged[-1]["ids"] += " | " + event["ids"]
            merged[-1]["offset"] = max(merged[-1]["offset"], event["offset"])
            merged[-1]["bridge"] |= event["bridge"]
        else:
            merged.append(event.copy())
    return merged


def load_footprints(args: argparse.Namespace, crs: CRS) -> gpd.GeoDataFrame:
    """Read whole polygons so clipping cannot manufacture a crossing boundary."""
    layer = load_shapefile(args.footprints_path, args.footprints_member)
    if layer.crs is None or "INT_ID" not in layer:
        raise ValueError("Footprints require a defined CRS and an INT_ID field.")
    ids = layer["INT_ID"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("INT_ID values must be nonblank and unique.")
    layer = layer.to_crs(crs).reset_index(drop=True)
    layer["INT_ID"] = ids.to_numpy()
    if not (
        layer.geom_type.isin(["Polygon", "MultiPolygon"]) & layer.is_valid & ~layer.is_empty
    ).all():
        raise ValueError("Footprints must contain valid, nonempty polygons.")
    layer["GRADE_SEP"] = (
        layer.get("GRADE_SEP", pd.Series("U", index=layer.index))
        .fillna("U")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", "U")
    )
    if not layer["GRADE_SEP"].isin(["Y", "N", "U"]).all():
        raise ValueError("GRADE_SEP accepts Y, N, U or blank.")
    overlaps = set()
    for i, polygon in enumerate(layer.geometry):
        for j in layer.sindex.query(polygon, predicate="intersects"):
            if j > i and polygon.intersection(layer.geometry.iloc[j]).area > 0:
                overlaps.update((i, int(j)))
    layer["overlap"] = layer.index.isin(overlaps)
    return gpd.GeoDataFrame(layer, geometry="geometry", crs=crs)


def footprint_events(line: LineString, footprints: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    """Split individual directed segments at boundaries to preserve retraced passes."""
    coords = list(line.coords)
    segments = [LineString([a, b]) for a, b in itertools.pairwise(coords) if a != b]
    tree = STRtree(segments)
    starts = np.cumsum([0.0] + [s.length for s in segments])
    events = []
    for idx in footprints.sindex.query(line, predicate="intersects"):
        row = footprints.iloc[idx]
        polygon = row.geometry
        boundary = polygon.boundary
        spans: list[tuple[float, float, str]] = []
        contacts: set[float] = set()
        for j in tree.query(polygon, predicate="intersects"):
            segment = segments[j]
            cuts = {0.0, segment.length}
            for xy in get_coordinates(segment.intersection(boundary)):
                local = segment.project(Point(xy))
                cuts.add(local)
                contacts.add(float(starts[j] + local))
            cuts_sorted = sorted(cuts)
            for a, b in itertools.pairwise(cuts_sorted):
                mid = segment.interpolate((a + b) / 2)
                if polygon.covers(mid):
                    kind = "crossing" if polygon.contains(mid) else "touch"
                    spans.append((float(starts[j] + a), float(starts[j] + b), kind))
        merged: list[tuple[float, float, str]] = []
        for a, b, kind in sorted(spans):
            if merged and merged[-1][2] == kind and abs(merged[-1][1] - a) < 1e-7:
                merged[-1] = (merged[-1][0], b, kind)
            else:
                merged.append((a, b, kind))
        isolated = [
            (x, x, "touch")
            for x in contacts
            if not any(a - 1e-7 <= x <= b + 1e-7 for a, b, _ in merged)
        ]
        repeated = sum(kind == "crossing" for _, _, kind in merged) > 1
        for a, b, kind in merged + isolated:
            flags = []
            if row["overlap"]:
                flags.append("overlapping_footprints")
            if row.get("bridge_road", False):
                flags.append("bridge_road_at_footprint")
            if repeated:
                flags.append("repeated_footprint_pass")
            if row["GRADE_SEP"] != "N":
                flags.append(
                    "grade_separation_review" if row["GRADE_SEP"] == "Y" else "grade_unknown"
                )
            if kind == "touch":
                flags.append("footprint_boundary_touch")
            if a == 0 or abs(line.length - b) < 1e-7:
                flags.append("shape_end_limits_footprint")
            events.append(
                {
                    "start": a,
                    "end": b,
                    "ids": row["INT_ID"],
                    "names": row.get("STREETS", "") or "",
                    "offset": 0.0,
                    "bridge": False,
                    "flags": flags,
                    "kind": kind,
                }
            )
    return events


def placement_at(
    station: float,
    events: list[dict[str, Any]],
    length: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Classify along-shape relationships while preserving previous and next events."""
    polygons = bool(args.footprints_path)
    contacts = [
        e
        for e in events
        if e.get("kind") == "touch"
        and e["start"] - args.near_distance_ft <= station <= e["end"] + args.near_distance_ft
    ]
    events = [e for e in events if e.get("kind") != "touch"]
    zone = 1e-7 if polygons else args.intersection_zone_ft
    previous = [e for e in events if e["end"] < station]
    following = [e for e in events if e["start"] > station]
    prev = max(previous, key=lambda e: e["end"]) if previous else None
    nex = min(following, key=lambda e: e["start"]) if following else None
    behind = station - prev["end"] if prev else None
    ahead = nex["start"] - station if nex else None
    reasons = []
    touching = [e for e in events if e["start"] - zone <= station <= e["end"] + zone]
    near = ahead is not None and ahead <= args.near_distance_ft
    far = behind is not None and behind <= args.near_distance_ft
    proposed = (
        "near_and_far_side"
        if near and far
        else ("near_side" if near else ("far_side" if far else "midblock"))
    )
    if touching:
        if polygons:
            proposed = "within_footprint"
            if any(
                min(abs(station - e["start"]), abs(station - e["end"])) <= zone for e in touching
            ):
                reasons.append("stop_at_footprint_boundary")
        else:
            reasons.append("within_junction_centerline_zone")
    if polygons and proposed == "midblock" and not args.footprints_complete:
        reasons.append("footprint_coverage_unverified")
    if station < args.near_distance_ft or length - station < args.near_distance_ft:
        reasons.append("shape_end_limits_context")
    relevant = contacts + touching + ([prev] if far else []) + ([nex] if near else [])
    for event in relevant:
        if event is None:
            continue
        reasons.extend(event.get("flags", []))
        if event["offset"] > args.review_offset_ft:
            reasons.append("junction_shape_offset")
        if event["bridge"]:
            reasons.append("bridge_at_junction")
    return {
        "suggested_placement": proposed,
        **(
            {
                "containing_footprints": " | ".join(e["ids"] for e in touching),
                "boundary_touch_ids": " | ".join(sorted({e["ids"] for e in contacts})),
            }
            if polygons
            else {}
        ),
        "previous_junction": prev["ids"] if prev else "",
        "previous_streets": prev["names"] if prev else "",
        "previous_distance_ft": round(behind, 1) if behind is not None else None,
        "next_junction": nex["ids"] if nex else "",
        "next_streets": nex["names"] if nex else "",
        "next_distance_ft": round(ahead, 1) if ahead is not None else None,
        "review_reason": ";".join(sorted(set(reasons))),
    }


def unmatched_nearby_junctions(
    line: LineString,
    station: float,
    events: list[dict[str, Any]],
    junctions: gpd.GeoDataFrame,
    args: argparse.Namespace,
) -> list[str]:
    """Identify nearby nodes rejected by heading checks or the shape-offset limit."""
    accepted = {node_id for event in events for node_id in event["ids"].split(" | ")}
    center = line.interpolate(station)
    radius = args.near_distance_ft + args.max_offset_ft
    rejected = []
    for index in junctions.sindex.query(center.buffer(radius)):
        node = junctions.iloc[int(index)]
        if node.junction_id in accepted or center.distance(node.geometry) > radius:
            continue
        positions = projection_candidates(line, node.geometry, radius)
        if any(abs(position - station) <= args.near_distance_ft for position in positions):
            rejected.append(node.junction_id)
    return sorted(rejected)


def classify_stops(
    stops: gpd.GeoDataFrame,
    visits: pd.DataFrame,
    lines: dict[str, LineString],
    junctions: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    context: Any,
    args: argparse.Namespace,
    footprints: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Return one record per stop, shape, route and direction, retaining uncertainty."""
    events_cache = {
        key: (
            footprint_events(line, footprints)
            if footprints is not None
            else shape_events(line, junctions, args)
        )
        for key, line in lines.items()
    }
    if footprints is not None:
        covered, _ = footprints.sindex.query(junctions.geometry, predicate="intersects")
        junctions = junctions.iloc[~np.isin(np.arange(len(junctions)), covered)]
    records = []
    for _, stop in stops.iterrows():
        rows = visits[visits.stop_id == stop.stop_id]
        if rows.empty:
            rows = pd.DataFrame([{"shape_id": "", "route_id": "", "direction_id": ""}])
        for _, visit in rows.iterrows():
            record = {
                "intersection_source": "footprints" if footprints is not None else "roads",
                "stop_id": stop.stop_id,
                "stop_code": stop.get("stop_code", ""),
                "stop_name": stop.stop_name,
                "stop_lon": float(stop.stop_lon),
                "stop_lat": float(stop.stop_lat),
                "route_id": visit.route_id,
                "direction_id": visit.direction_id,
                "shape_id": visit.shape_id,
                "suggested_placement": "",
                "placement": "unresolved",
                "review_reason": "",
            }
            reasons = []
            line = lines.get(visit.shape_id)
            if re.search(args.terminal_pattern, stop.stop_name, re.IGNORECASE):
                record["placement"] = "terminal_or_offstreet_review"
                reasons.append("terminal_name_match_verify")
            elif line is None:
                reasons.append("missing_or_invalid_shape")
            else:
                offset = line.distance(stop.geometry)
                record["stop_shape_offset_ft"] = round(offset, 1)
                candidates = projection_candidates(line, stop.geometry, args.max_offset_ft)
                if not candidates:
                    reasons.append("stop_too_far_from_shape")
                elif len(candidates) != 1:
                    reasons.append("multiple_shape_passes")
                else:
                    station = candidates[0]
                    rejected = unmatched_nearby_junctions(
                        line,
                        station,
                        [] if footprints is not None else events_cache[visit.shape_id],
                        junctions,
                        args,
                    )
                    record["unmatched_nearby_junctions"] = " | ".join(rejected)
                    if rejected:
                        reasons.append(
                            "incomplete_footprint_coverage"
                            if footprints is not None
                            else "nearby_junction_not_matched"
                        )
                    record.update(
                        placement_at(station, events_cache[visit.shape_id], line.length, args)
                    )
                    reasons.extend(filter(None, record["review_reason"].split(";")))
                    if offset > args.review_offset_ft:
                        reasons.append("stop_shape_offset")
                    # A midblock label needs road coverage and context on both sides.
                    sample = line.interpolate(station)
                    nearest = int(roads.sindex.nearest(sample)[1][0])
                    road = roads.iloc[nearest]
                    record["nearest_road"] = road[args.road_name_field]
                    record["shape_road_offset_ft"] = round(sample.distance(road.geometry), 1)
                    if sample.distance(road.geometry) > args.review_offset_ft:
                        reasons.append("shape_road_offset")
                    if (
                        footprints is not None
                        and args.bridge_field
                        and str(road[args.bridge_field]).strip().upper()
                        not in ("N", "0", "", "NONE")
                    ):
                        reasons.append("bridge_on_nearest_road")
                    if context.boundary.distance(stop.geometry) < args.near_distance_ft:
                        reasons.append("road_clip_edge")
                    record["shape_station_ft"] = round(station, 1)
                    if not reasons:
                        record["placement"] = record["suggested_placement"]
            record["review_reason"] = ";".join(sorted(set(reasons)))
            records.append(record)
    return pd.DataFrame(records)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Collapse only agreeing patterns; keep disagreement and unresolved patterns visible."""
    records = []
    for _, group in detail.groupby("stop_id", sort=False):
        row = group.iloc[0]
        labels = sorted(set(group.placement))
        label = labels[0] if len(labels) == 1 else "pattern_disagreement"
        records.append(
            {key: row[key] for key in ["stop_id", "stop_code", "stop_name", "stop_lon", "stop_lat"]}
            | {
                "placement": label,
                "routes": " | ".join(sorted(set(group.route_id) - {""})),
                "shape_count": len(set(group.shape_id) - {""}),
                "pattern_labels": " | ".join(labels),
                "review_reason": ";".join(
                    sorted(
                        {
                            reason
                            for text in group.review_reason
                            for reason in text.split(";")
                            if reason
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(records)


def build_arg_parser() -> argparse.ArgumentParser:
    """Mirror editable settings with command-line flags and optional run-log replay."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", help="Replay effective settings from a saved run log.")
    for name, default in [
        ("gtfs-path", GTFS_PATH),
        ("roads-path", ROADS_PATH),
        ("output-dir", OUTPUT_DIR),
        ("projected-crs", PROJECTED_CRS),
        ("road-name-field", ROAD_NAME_FIELD),
        ("road-class-field", ROAD_CLASS_FIELD),
        ("bridge-field", BRIDGE_FIELD),
        ("footprints-path", FOOTPRINTS_PATH),
        ("footprints-member", FOOTPRINTS_MEMBER),
        ("roads-member", ROADS_MEMBER),
        ("terminal-pattern", TERMINAL_PATTERN),
    ]:
        parser.add_argument("--" + name, default=default)
    for name, default in [
        ("near-distance-ft", NEAR_DISTANCE_FT),
        ("intersection-zone-ft", INTERSECTION_ZONE_FT),
        ("context-buffer-ft", CONTEXT_BUFFER_FT),
        ("node-tolerance-ft", NODE_TOLERANCE_FT),
        ("junction-merge-ft", JUNCTION_MERGE_FT),
        ("max-offset-ft", MAX_OFFSET_FT),
        ("review-offset-ft", REVIEW_OFFSET_FT),
        ("heading-sample-ft", HEADING_SAMPLE_FT),
        ("heading-tolerance-deg", HEADING_TOLERANCE_DEG),
    ]:
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=BBOX,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    parser.add_argument(
        "--footprints-complete",
        action=argparse.BooleanOptionalAction,
        default=FOOTPRINTS_COMPLETE,
    )
    parser.add_argument("--exclude-road-classes", nargs="*", default=EXCLUDE_ROAD_CLASSES)
    parser.add_argument(
        "--require-run-log",
        action=argparse.BooleanOptionalAction,
        default=REQUIRE_RUN_LOG,
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Clip roads, classify served stops and export reproducible review products."""
    crs = CRS(args.projected_crs)
    if not crs.is_projected:
        raise ValueError("Use a projected CRS with feet or meters as its horizontal unit.")
    # All algorithms use feet internally; CRS conversion is explicit.
    feet_per_unit = crs.axis_info[0].unit_conversion_factor / 0.3048
    if not math.isclose(feet_per_unit, 1.0, rel_tol=1e-4):
        raise ValueError("This draft requires a feet-based CRS (default EPSG:2283).")
    if args.near_distance_ft <= args.intersection_zone_ft or args.intersection_zone_ft < 0:
        raise ValueError("near-distance-ft must exceed intersection-zone-ft >= 0.")
    for name in [
        "node_tolerance_ft",
        "junction_merge_ft",
        "max_offset_ft",
        "review_offset_ft",
        "heading_sample_ft",
        "context_buffer_ft",
    ]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.review_offset_ft > args.max_offset_ft:
        raise ValueError("review-offset-ft cannot exceed max-offset-ft.")
    if args.context_buffer_ft < args.near_distance_ft + args.heading_sample_ft + args.max_offset_ft:
        raise ValueError("Increase context-buffer-ft to cover the classification search window.")
    if not 0 < args.heading_tolerance_deg < 90:
        raise ValueError("heading-tolerance-deg must be between 0 and 90.")
    re.compile(args.terminal_pattern)
    gtfs = Path(args.gtfs_path)
    stops_data = read_gtfs(gtfs, "stops.txt")
    stop_times = read_gtfs(gtfs, "stop_times.txt")
    trips = read_gtfs(gtfs, "trips.txt")
    shapes = read_gtfs(gtfs, "shapes.txt")
    for frame, required in [
        (stops_data, ["stop_id", "stop_name", "stop_lon", "stop_lat"]),
        (stop_times, ["trip_id", "stop_id"]),
        (trips, ["trip_id", "route_id", "shape_id"]),
        (shapes, ["shape_id", "shape_pt_lon", "shape_pt_lat", "shape_pt_sequence"]),
    ]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing GTFS columns: {sorted(missing)}")
    if stops_data.stop_id.duplicated().any() or trips.trip_id.duplicated().any():
        raise ValueError("GTFS stop_id and trip_id must be unique in their parent files.")
    stops_data = stops_data[stops_data.stop_id.isin(stop_times.stop_id)].copy()
    for column in ["stop_lon", "stop_lat"]:
        stops_data[column] = pd.to_numeric(stops_data[column], errors="raise")
    stops = gpd.GeoDataFrame(
        stops_data,
        geometry=gpd.points_from_xy(stops_data.stop_lon, stops_data.stop_lat),
        crs=4326,
    ).to_crs(crs)
    if args.bbox is not None:
        west, south, east, north = args.bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox must be west south east north in WGS84.")
        area = gpd.GeoSeries([box(west, south, east, north)], crs=4326).to_crs(crs).iloc[0]
        stops = stops[stops.intersects(area)].copy()
    else:
        area = box(*stops.total_bounds)
    if stops.empty:
        raise ValueError("No served GTFS stops in the requested area.")
    context = area.buffer(args.context_buffer_ft)
    roads = load_shapefile(args.roads_path, args.roads_member)
    if roads.crs is None:
        raise ValueError("Roads have no CRS; define it before running.")
    for field in [args.road_name_field, args.road_class_field, args.bridge_field]:
        if field and field not in roads.columns:
            raise ValueError(f"Road field {field!r} not found; available: {list(roads.columns)}")
    roads = roads.to_crs(crs)
    roads = roads[roads.intersects(context)].copy()
    if args.road_class_field:
        roads = roads[~roads[args.road_class_field].isin(args.exclude_road_classes)].copy()
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    roads = roads[roads.geom_type == "LineString"].copy()
    roads[args.road_name_field] = roads[args.road_name_field].fillna("")
    roads = roads[roads.geometry.length > 0].reset_index(drop=True)
    # Infer before geometrically clipping: clip-created endpoints are not junctions.
    junctions = infer_junctions(roads, args)
    junctions = junctions[junctions.intersects(context)].reset_index(drop=True)
    roads_clip = gpd.clip(roads, context, keep_geom_type=True)
    trips["direction_id"] = trips.get("direction_id", "")
    visits = (
        stop_times[stop_times.stop_id.isin(stops.stop_id)][["stop_id", "trip_id"]]
        .merge(
            trips[["trip_id", "shape_id", "route_id", "direction_id"]],
            on="trip_id",
            how="left",
            validate="many_to_one",
        )
        .fillna("")
    )
    visits = visits[["stop_id", "shape_id", "route_id", "direction_id"]].drop_duplicates()
    shapes = shapes[shapes.shape_id.isin(visits.shape_id)].copy()
    for column in ["shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]:
        shapes[column] = pd.to_numeric(shapes[column], errors="raise")
    lines: dict[str, LineString] = {}
    for shape_id, group in shapes.groupby("shape_id", sort=False):
        group = group.sort_values("shape_pt_sequence")
        if len(group) < 2 or group.shape_pt_sequence.duplicated().any():
            continue
        line = (
            gpd.GeoSeries([LineString(zip(group.shape_pt_lon, group.shape_pt_lat))], crs=4326)
            .to_crs(crs)
            .iloc[0]
        )
        if line.length > 0:
            lines[str(shape_id)] = line
    logger.info(
        "Classifying %s stops, %s shapes, %s inferred junctions.",
        len(stops),
        len(lines),
        len(junctions),
    )
    footprints = load_footprints(args, crs) if args.footprints_path else None
    if footprints is not None and args.bridge_field:
        bridge_values = roads[args.bridge_field].fillna("").astype(str).str.strip().str.upper()
        bridge_roads = roads.loc[~bridge_values.isin(["N", "0", "0.0", "", "NONE"])]
        affected, _ = bridge_roads.sindex.query(footprints.geometry, predicate="intersects")
        footprints["bridge_road"] = footprints.index.isin(affected)
    detail = classify_stops(stops, visits, lines, junctions, roads, context, args, footprints)
    summary = summarize(detail)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name, frame in [("stop_placement", summary), ("stop_placement_detail", detail)]:
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts.append(path)
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, frame in [
        ("Stops", summary),
        ("Pattern detail", detail),
        (
            "Counts",
            summary.placement.value_counts().rename_axis("placement").reset_index(name="stops"),
        ),
    ]:
        sheet = workbook.create_sheet(name)
        sheet.append(list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            sheet.append([None if pd.isna(value) else value for value in row])
    path = out / "stop_placement.xlsx"
    workbook.save(path)
    artifacts.append(path)
    points = stops[["stop_id", "geometry"]].merge(summary, on="stop_id", validate="one_to_one")
    path = out / "placement_review.gpkg"
    if path.exists():
        path.unlink()
    roads_clip.to_file(path, layer="roads_clipped", driver="GPKG")
    if not junctions.empty:
        junctions.to_file(path, layer="inferred_junctions", driver="GPKG")
    if footprints is not None and not footprints.empty:
        footprints.to_file(path, layer="intersection_footprints", driver="GPKG")
    points.to_file(path, layer="classified_stops", driver="GPKG")
    if lines:
        shape_layer = gpd.GeoDataFrame(
            {"shape_id": list(lines), "geometry": list(lines.values())}, crs=crs
        )
        gpd.clip(shape_layer, context, keep_geom_type=True).to_file(
            path, layer="gtfs_shapes_clipped", driver="GPKG"
        )
    gpd.GeoDataFrame({"geometry": [area]}, crs=crs).to_file(path, layer="study_area", driver="GPKG")
    artifacts.append(path)
    figure, ax = plt.subplots(figsize=(12, 12))
    roads_clip.plot(ax=ax, color="#c8c8c8", linewidth=0.6)
    if footprints is not None and not footprints.empty:
        footprints.boundary.plot(ax=ax, color="#413870", linewidth=0.9)
    colors = {
        "within_footprint": "#0e7490",
        "near_side": "#d97706",
        "far_side": "#2563eb",
        "midblock": "#16a34a",
        "near_and_far_side": "#7c3aed",
        "unresolved": "#dc2626",
        "pattern_disagreement": "#a21caf",
        "terminal_or_offstreet_review": "#555555",
    }
    for label, group in points.groupby("placement"):
        group.plot(
            ax=ax,
            color=colors.get(str(label), "#000000"),
            markersize=30,
            label=f"{label} ({len(group)})",
        )
    if not junctions.empty:
        junctions.plot(ax=ax, marker="+", color="#666666", markersize=20, label="Inferred junction")
    for _, stop in points.iterrows():
        if stop.placement != "terminal_or_offstreet_review":
            ax.annotate(
                str(stop.stop_id),
                (stop.geometry.x, stop.geometry.y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=5,
            )
    ax.set_xlim(area.bounds[0], area.bounds[2])
    ax.set_ylim(area.bounds[1], area.bounds[3])
    ax.set_title(
        "Stop placement screening — unverified draft\n"
        + (
            "Distances to footprint boundaries"
            if footprints is not None
            else "Distances to road centerline junctions"
        )
    )
    ax.legend(loc="best", fontsize=8)
    ax.set_axis_off()
    figure.tight_layout()
    path = out / "stop_placement_map.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    artifacts.append(path)
    source = Path(__file__).read_text(encoding="utf-8")
    config = source.split("# === BEGIN CONFIG ===", 1)[1].split("# === END CONFIG ===", 1)[0]
    effective = vars(args).copy()
    effective.pop("config", None)
    log = (
        f"UTC: {datetime.now(UTC).isoformat()}\nSource: {Path(__file__).resolve()}\n"
        f"# === BEGIN CONFIG ==={config}# === END CONFIG ===\n"
        f"Effective settings JSON:\n{json.dumps(effective, indent=2)}\n"
        f"Counts:\n{summary.placement.value_counts().to_string()}\n"
    )
    for artifact in artifacts:
        try:
            artifact.with_name(artifact.stem + "_runlog.txt").write_text(log, encoding="utf-8")
        except OSError:
            if args.require_run_log:
                raise
            logger.warning("Could not write run log for %s", artifact)
    logger.info("Finished: %s\n%s", out, summary.placement.value_counts().to_string())


def main(argv: Sequence[str] | None = None) -> int:
    """Run from edited constants, strict CLI arguments, or a replayed run log."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = build_arg_parser()
    cli = notebook_safe_argv(argv)
    initial = parser.parse_args(cli)
    try:
        if initial.config:
            text = Path(initial.config).read_text(encoding="utf-8")
            settings = json.JSONDecoder().raw_decode(
                text.split("Effective settings JSON:\n", 1)[1]
            )[0]
            allowed = vars(initial).keys() - {"config"}
            if not isinstance(settings, dict) or set(settings) - allowed:
                raise ValueError("Run log has unknown settings.")
            parser.set_defaults(**settings)
        args = parser.parse_args(cli)
        if not Path(args.gtfs_path).exists() or not Path(args.roads_path).is_file():
            logger.error("Set gtfs-path and roads-path to existing inputs.")
            return 2
        run(args)
        return 0
    except (ValueError, KeyError, IndexError, OSError, zipfile.BadZipFile) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
