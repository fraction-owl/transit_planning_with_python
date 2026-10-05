from __future__ import annotations

import ast
import logging
import zipfile
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

import scripts.facilities_tools.cluster_stops_from_zones_gpd as target

# Five transit-center zones traced in ArcGIS, moved onto mock_gtfs_dc.zip. Centers 01-03
# hold stops; 01 and the empty 04 have stops just outside; 05 has no stop within 1,000 ft.
# A zipped shapefile in Maryland State Plane (US feet, EPSG:2248) with one NAME field.
STOP_CLUSTERS_ZIP = Path(__file__).parent / "fixtures" / "stop_clusters_sample.zip"
MOCK_GTFS_DC_ZIP = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"

# Metro and Park & Ride share the lon = -76.95 edge; Empty Lot holds no stop (WGS84).
_EDGE_LON = -76.95


def _square(west: float, south: float, east: float, north: float) -> Polygon:
    return Polygon([(west, south), (east, south), (east, north), (west, north)])


METRO = _square(-77.00, 38.80, _EDGE_LON, 39.00)
PARK_AND_RIDE = _square(_EDGE_LON, 38.80, -76.90, 39.00)
EMPTY_LOT = _square(-76.80, 38.80, -76.75, 38.85)
# Reaches 0.01 degrees into Metro.
OVERLAPS_METRO = _square(-76.96, 38.80, -76.90, 39.00)

ZONE_NAMES = ["Metro", "Park & Ride", "Empty Lot"]
ZONE_SHAPES = [METRO, PARK_AND_RIDE, EMPTY_LOT]


def _layer(
    names: Optional[list[Any]] = None,
    geometries: Optional[list[Any]] = None,
    crs: Optional[str] = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """A zone layer as it might be drawn in GIS, with a NAME field."""
    return gpd.GeoDataFrame(
        {"NAME": ZONE_NAMES if names is None else names},
        geometry=ZONE_SHAPES if geometries is None else geometries,
        crs=crs,
    )


def _loaded(names: list[str], geometries: list[Any], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Zones shaped as load_zones returns them."""
    return gpd.GeoDataFrame(
        {"polygon": list(range(1, len(names) + 1)), "zone": names}, geometry=geometries, crs=crs
    )


def _stops_txt() -> pd.DataFrame:
    # 2956, 65 and 7 are in Metro (out of natural order), 3881 and WEEKEND in
    # Park & Ride. STN is a station inside Metro, OUT lies outside every zone
    # and NOCOORD has no latitude.
    return pd.DataFrame(
        {
            "stop_id": ["2956", "65", "7", "3881", "WEEKEND", "STN", "OUT", "NOCOORD"],
            "stop_code": ["", "1065", "1007", "3881", "5000", "", "9000", "9001"],
            "stop_name": [
                'Metro "North" Bay',
                "Metro Center Bay A",
                "Metro Center  Bay B",
                "Park & Ride Bay 1",
                "Park & Ride Weekend Bay",
                "Metro Center Station",
                "Main St & 1st Ave",
                "Nowhere",
            ],
            "stop_lat": ["38.90", "38.90", "38.95", "38.90", "38.85", "38.90", "38.90", None],
            "stop_lon": [
                "-76.97",
                "-76.98",
                "-76.99",
                "-76.92",
                "-76.93",
                "-76.97",
                "-76.50",
                "-76.98",
            ],
            "location_type": ["0", "", "0", "0", "0", "1", "0", "0"],
        }
    )


def _write_inputs(
    tmp_path: Path,
    layer: Optional[gpd.GeoDataFrame] = None,
    zones_file: str = "zones.geojson",
) -> dict[str, str]:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _stops_txt().to_csv(gtfs_dir / "stops.txt", index=False)
    zones = tmp_path / zones_file
    (_layer() if layer is None else layer).to_file(zones)
    return {"gtfs": str(gtfs_dir), "zones": str(zones)}


def _cli(paths: dict[str, str], *extra: str) -> list[str]:
    return [
        "--gtfs-path",
        paths["gtfs"],
        "--zones-path",
        paths["zones"],
        "--zone-name-field",
        "NAME",
        *extra,
    ]


def _sample_assignments() -> pd.DataFrame:
    return target.assign_stops_to_zones(
        target.prepare_stops(_stops_txt()), _loaded(ZONE_NAMES, ZONE_SHAPES)
    )


def _values(text: str) -> list[tuple[str, Any]]:
    """Every top-level assignment in *text* as (name, literal value), in order."""
    values = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            name = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            name = node.target
        else:
            continue
        assert isinstance(name, ast.Name)
        assert node.value is not None
        values.append((name.id, ast.literal_eval(node.value)))
    return values


# ---------------------------------------------------------------------------
# prepare_stops
# ---------------------------------------------------------------------------


def test_prepare_stops_keeps_stops_with_coordinates(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        stops = target.prepare_stops(_stops_txt())
    # STN is a station; NOCOORD has no latitude and is reported.
    assert stops["stop_id"].tolist() == ["2956", "65", "7", "3881", "WEEKEND", "OUT"]
    assert stops["stop_lat"].dtype == float
    assert stops.loc[2, "stop_name"] == "Metro Center Bay B"  # whitespace collapsed
    assert "NOCOORD" in caplog.text


def test_prepare_stops_fills_missing_optional_columns() -> None:
    stops = target.prepare_stops(
        _stops_txt().drop(columns=["stop_code", "stop_name", "location_type"])
    )
    assert set(stops["stop_code"]) == {""}
    assert set(stops["stop_name"]) == {""}
    assert "STN" in set(stops["stop_id"])  # without location_type every row is a stop


def test_prepare_stops_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="stop_lon"):
        target.prepare_stops(_stops_txt().drop(columns=["stop_lon"]))


def test_prepare_stops_repeated_stop_id_raises() -> None:
    stops = _stops_txt()
    stops.loc[1, "stop_id"] = "2956"
    with pytest.raises(ValueError, match="repeats stop_id 2956"):
        target.prepare_stops(stops)


# ---------------------------------------------------------------------------
# load_zones
# ---------------------------------------------------------------------------


def test_load_zones_reads_names_from_the_field(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    zones = target.load_zones(paths["zones"], "NAME")
    assert zones["zone"].tolist() == ZONE_NAMES
    assert zones["polygon"].tolist() == [1, 2, 3]


def test_load_zones_numbers_zones_without_a_name_field(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    zones = target.load_zones(paths["zones"])
    assert zones["zone"].tolist() == ["Zone 1", "Zone 2", "Zone 3"]


def test_zone_names_number_blank_names_and_tidy_the_rest() -> None:
    names = target._zone_names(
        ["Metro", None, float("nan"), 12.0, "  Park   & Ride "], [1, 2, 3, 4, 5]
    )
    assert names == ["Metro", "Zone 2", "Zone 3", "12", "Park & Ride"]


def test_zone_names_number_that_is_another_name_raises() -> None:
    with pytest.raises(ValueError, match="Zone 2"):
        target._zone_names(["Zone 2", None], [1, 2])


def test_load_zones_missing_field_lists_available_fields(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="Available fields: NAME"):
        target.load_zones(paths["zones"], "ZONE")


def test_load_zones_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="not found"):
        target.load_zones(str(tmp_path / "nope.shp"))


def test_load_zones_missing_crs_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_inputs(tmp_path)
    naked = _layer().set_crs(None, allow_override=True)
    monkeypatch.setattr(target.gpd, "read_file", lambda path: naked)
    with pytest.raises(ValueError, match="no CRS"):
        target.load_zones(paths["zones"])


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (LineString([(-77.0, 38.8), (-76.9, 38.9)]), "must be polygons"),
        # A bowtie: its edges cross.
        (
            Polygon([(-77.0, 38.8), (-76.9, 38.9), (-76.9, 38.8), (-77.0, 38.9)]),
            "Self-intersection",
        ),
        (None, "no geometry"),
    ],
)
def test_load_zones_rejects_bad_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, geometry: Any, message: str
) -> None:
    paths = _write_inputs(tmp_path)
    layer = _layer(["Metro", "Bad"], [METRO, geometry])
    monkeypatch.setattr(target.gpd, "read_file", lambda path: layer)
    with pytest.raises(ValueError, match=message):
        target.load_zones(paths["zones"], "NAME")


def test_load_zones_rejects_projected_coordinates_labelled_lon_lat(tmp_path: Path) -> None:
    mislabelled = _layer().to_crs("EPSG:2248").set_crs("EPSG:4326", allow_override=True)
    paths = _write_inputs(tmp_path, mislabelled)
    with pytest.raises(ValueError, match="out of range"):
        target.load_zones(paths["zones"])


# ---------------------------------------------------------------------------
# check_zone_overlaps
# ---------------------------------------------------------------------------


def test_check_zone_overlaps_rejects_zones_that_share_area() -> None:
    zones = _loaded(["Metro", "Park & Ride"], [METRO, OVERLAPS_METRO])
    with pytest.raises(ValueError, match="'Metro' and 'Park & Ride'"):
        target.check_zone_overlaps(zones)


def test_check_zone_overlaps_allows_touching_zones() -> None:
    target.check_zone_overlaps(_loaded(ZONE_NAMES, ZONE_SHAPES))


def test_check_zone_overlaps_allows_overlap_within_one_zone() -> None:
    target.check_zone_overlaps(_loaded(["Metro", "Metro"], [METRO, OVERLAPS_METRO]))


# ---------------------------------------------------------------------------
# assign_stops_to_zones
# ---------------------------------------------------------------------------


def test_assign_lists_stops_in_zone_order_then_natural_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assignments = _sample_assignments()
    assert list(zip(assignments["zone"], assignments["stop_id"])) == [
        ("Metro", "7"),
        ("Metro", "65"),
        ("Metro", "2956"),
        ("Park & Ride", "3881"),
        ("Park & Ride", "WEEKEND"),
    ]
    assert "Empty Lot" in caplog.text  # a zone without stops is reported


def test_assign_projects_stops_into_the_zone_crs() -> None:
    zones = _loaded(ZONE_NAMES, ZONE_SHAPES).to_crs("EPSG:2248")
    assignments = target.assign_stops_to_zones(target.prepare_stops(_stops_txt()), zones)
    assert assignments["stop_id"].tolist() == ["7", "65", "2956", "3881", "WEEKEND"]


def _edge_stop() -> pd.DataFrame:
    return target.prepare_stops(
        pd.DataFrame({"stop_id": ["EDGE"], "stop_lat": ["38.90"], "stop_lon": [str(_EDGE_LON)]})
    )


def test_assign_stop_on_the_edge_of_two_zones_raises() -> None:
    zones = _loaded(["Metro", "Park & Ride"], [METRO, PARK_AND_RIDE])
    with pytest.raises(ValueError, match=r"EDGE \(Metro \| Park & Ride\)"):
        target.assign_stops_to_zones(_edge_stop(), zones)


def test_assign_stop_between_polygons_of_one_zone_is_listed_once() -> None:
    zones = _loaded(["Metro", "Metro"], [METRO, PARK_AND_RIDE])
    assignments = target.assign_stops_to_zones(_edge_stop(), zones)
    assert assignments["stop_id"].tolist() == ["EDGE"]


def test_assign_no_stop_in_any_zone_raises() -> None:
    with pytest.raises(ValueError, match="No stop falls inside any zone"):
        target.assign_stops_to_zones(
            target.prepare_stops(_stops_txt()), _loaded(["Empty Lot"], [EMPTY_LOT])
        )


# ---------------------------------------------------------------------------
# find_near_misses
# ---------------------------------------------------------------------------


def _near_miss_stops() -> pd.DataFrame:
    # Park & Ride's east edge is lon -76.90: NEAR is 0.0005 degrees (about 141 ft) east
    # of it and FAR 0.002 degrees (about 568 ft). INSIDE is in Metro, 0.0005 degrees
    # from Park & Ride. NORTH is 0.0005 degrees above Metro's top edge (about 182 ft)
    # and about 338 ft from Park & Ride's corner.
    return target.prepare_stops(
        pd.DataFrame(
            {
                "stop_id": ["NEAR", "FAR", "INSIDE", "NORTH"],
                "stop_code": ["", "", "", "4000"],
                "stop_name": ["Park & Ride East", "", "", "Metro North Lot"],
                "stop_lat": ["38.90", "38.90", "38.90", "39.0005"],
                "stop_lon": ["-76.8995", "-76.898", "-76.9505", "-76.951"],
            }
        )
    )


def _near_misses(max_feet: float, zones: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
    zones = _loaded(ZONE_NAMES, ZONE_SHAPES) if zones is None else zones
    stops = _near_miss_stops()
    return target.find_near_misses(
        stops, zones, target.assign_stops_to_zones(stops, zones), max_feet
    )


def test_find_near_misses_lists_stops_just_outside_a_zone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        near = _near_misses(400)
    # INSIDE is in Metro, so it is no near miss of Park & Ride; FAR is beyond 400 ft;
    # NORTH is listed once, against Metro, the nearer zone.
    assert list(zip(near["zone"], near["stop_id"])) == [("Metro", "NORTH"), ("Park & Ride", "NEAR")]
    assert near["feet"].tolist() == pytest.approx([182, 141], abs=1)
    assert "2 stop(s) are outside every zone but within 400 ft" in caplog.text
    assert "NEAR (141 ft from Park & Ride)" in caplog.text


def test_find_near_misses_leaves_out_stops_beyond_the_distance() -> None:
    assert _near_misses(150)["stop_id"].tolist() == ["NEAR"]


def test_find_near_misses_measures_a_projected_layer_in_its_own_units() -> None:
    projected = _loaded(ZONE_NAMES, ZONE_SHAPES).to_crs("EPSG:2248")  # US survey feet
    near = _near_misses(400, projected)
    assert near["stop_id"].tolist() == ["NORTH", "NEAR"]
    assert near["feet"].tolist() == pytest.approx(_near_misses(400)["feet"].tolist(), abs=1)


def test_find_near_misses_zero_turns_the_check_off(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        near = _near_misses(0)
    assert near.empty
    assert list(near.columns) == target.NEAR_MISS_COLUMNS
    assert "outside every zone" not in caplog.text


# ---------------------------------------------------------------------------
# config text
# ---------------------------------------------------------------------------


def test_step1_text_lists_stops_and_filters() -> None:
    values = dict(_values(target.step1_text(_sample_assignments())))
    assert values["CLUSTER_DEFINITIONS"] == {
        "Metro": {"stops": ["7", "65", "2956"]},
        "Park & Ride": {"stops": ["3881", "WEEKEND"]},
    }
    assert values["STOP_ID_FILTER"] == ["7", "65", "2956", "3881", "WEEKEND"]
    assert values["STOP_CODE_FILTER"] == ["1007", "1065", "3881", "5000"]  # 2956 has no code


def test_step1_text_without_stop_codes_leaves_the_code_filter_empty() -> None:
    assignments = _sample_assignments().assign(stop_code="")
    assert dict(_values(target.step1_text(assignments)))["STOP_CODE_FILTER"] == []


def test_step2_text_starts_every_stop_as_a_single_bay() -> None:
    values = dict(_values(target.step2_text(_sample_assignments())))
    assert values["CLUSTER_DEFINITIONS"]["Metro"] == {
        "single_bay_stops": ["7", "65", "2956"],
        "double_bay_stops": [],
        "triple_bay_stops": [],
        "overflow_bays": [],
    }


def test_step3_text_gives_one_block_per_zone() -> None:
    assert _values(target.step3_text(_sample_assignments())) == [
        ("CLUSTER_NAME", "Metro"),
        ("CLUSTER_STOPS", {"7": "7", "65": "65", "2956": "2956"}),
        ("CLUSTER_CAPACITY", {}),
        ("CLUSTER_NAME", "Park & Ride"),
        ("CLUSTER_STOPS", {"3881": "3881", "WEEKEND": "WEEKEND"}),
        ("CLUSTER_CAPACITY", {}),
    ]


def test_stop_lines_carry_the_code_and_name_as_a_comment() -> None:
    text = target.step2_text(_sample_assignments())
    assert '"65",  # 1065 | Metro Center Bay A' in text
    assert '"2956",  # Metro "North" Bay' in text  # no stop_code: the name alone
    assert '"3881",  # Park & Ride Bay 1' in text  # a stop_code equal to the stop_id is left out


def test_config_text_parses_and_summarizes_zones() -> None:
    text = target.build_config_text(_sample_assignments(), ZONE_NAMES, "zones.shp", "gtfs_folder")
    ast.parse(text)
    assert "#   Metro: 3 stop(s)" in text
    assert "#   No stops, left out: Empty Lot" in text
    assert all(len(line) <= target.MAX_LINE_LENGTH for line in text.splitlines())


def test_config_text_lists_near_misses_in_the_header() -> None:
    near = pd.DataFrame(
        [
            ["Park & Ride", "NEAR", "", "Park & Ride East", 142.4],
            ["Park & Ride", "LONG", "9100", "x" * 300, 1234.6],
        ],
        columns=target.NEAR_MISS_COLUMNS,
    )
    text = target.build_config_text(
        _sample_assignments(), ZONE_NAMES, "zones.shp", "gtfs_folder", near, 250
    )
    ast.parse(text)
    lines = text.splitlines()
    assert (
        "# Near misses, outside every zone but within 250 ft. Redraw a zone to add any that belong:"
        in lines
    )
    assert "#   Park & Ride: NEAR, 142 ft | Park & Ride East" in lines
    long_line = next(line for line in lines if "LONG" in line)
    assert long_line.startswith("#   Park & Ride: LONG, 1,235 ft | 9100 | xxx")
    assert long_line.endswith("...")
    assert all(len(line) <= target.MAX_LINE_LENGTH for line in lines)


def test_near_miss_lines_say_when_there_are_none_or_the_check_is_off() -> None:
    no_rows = pd.DataFrame(columns=target.NEAR_MISS_COLUMNS)
    assert target._near_miss_lines(no_rows, 250) == [
        "# Near misses, outside every zone but within 250 ft: none."
    ]
    assert target._near_miss_lines(None, 0) == ["# Near-miss check off (NEAR_MISS_FEET = 0)."]


def test_long_comments_are_shortened_to_the_line_length() -> None:
    line = target._with_comment('            "65",', "x" * 300)
    assert len(line) == target.MAX_LINE_LENGTH
    assert line.endswith("...")


def test_literal_round_trips_quotes_backslashes_and_accents() -> None:
    for text in ['Metro "North"', "C:\\bays", "Café Bay"]:
        assert ast.literal_eval(target._literal(text)) == text


# ---------------------------------------------------------------------------
# input fingerprints
# ---------------------------------------------------------------------------


def test_input_fingerprints_cover_stops_and_shapefile_sidecars(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, zones_file="zones.shp")
    text = "\n".join(target.input_fingerprints(paths["gtfs"], paths["zones"]))
    for label in ("GTFS stops.txt", "Zone layer:", "Zone layer .dbf", "Zone layer .prj"):
        assert label in text
    assert "not fingerprinted" not in text


# ---------------------------------------------------------------------------
# main / end-to-end
# ---------------------------------------------------------------------------


def test_main_placeholders_exit_2() -> None:
    assert target.main([]) == 2


def test_main_logs_the_text_and_writes_no_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = _write_inputs(tmp_path)
    before = set(tmp_path.rglob("*"))
    with caplog.at_level(logging.INFO):
        assert target.main(_cli(paths)) == 0
    assert '"Park & Ride": {' in caplog.text
    assert set(tmp_path.rglob("*")) == before


def test_main_writes_the_text_and_a_run_log(tmp_path: Path) -> None:
    # A projected shapefile, as zones are often drawn.
    paths = _write_inputs(tmp_path, _layer().to_crs("EPSG:2248"), "zones.shp")
    out_dir = tmp_path / "out"
    assert target.main(_cli(paths, "--output-dir", str(out_dir))) == 0

    text = (out_dir / "cluster_stops_from_zones.txt").read_text(encoding="utf-8")
    values = dict(_values(text.split("# Step 2:")[0]))
    assert values["CLUSTER_DEFINITIONS"]["Park & Ride"] == {"stops": ["3881", "WEEKEND"]}

    runlog = (out_dir / "cluster_stops_from_zones_runlog.txt").read_text(encoding="utf-8")
    assert 'ZONE_NAME_FIELD: str = ""' in runlog  # verbatim config block
    assert "Zone name field:  NAME" in runlog  # the setting actually used
    assert "sha256:" in runlog


def test_main_reads_a_zipped_feed(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    feed = tmp_path / "feed.zip"
    with zipfile.ZipFile(feed, "w") as archive:
        archive.write(Path(paths["gtfs"]) / "stops.txt", "feed/stops.txt")
    assert target.main(_cli({**paths, "gtfs": str(feed)})) == 0


def test_main_clusters_the_mock_dc_feed_with_a_zipped_state_plane_shapefile(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = {"gtfs": str(MOCK_GTFS_DC_ZIP), "zones": str(STOP_CLUSTERS_ZIP)}
    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        assert target.main(_cli(paths, "--output-dir", str(out_dir))) == 0

    text = (out_dir / "cluster_stops_from_zones.txt").read_text(encoding="utf-8")
    values = dict(_values(text.split("# Step 2:")[0]))
    assert values["CLUSTER_DEFINITIONS"] == {
        "Mock Transit Center 01": {"stops": ["DC_NS_034", "DC_R30_044", "DC_R40_088"]},
        "Mock Transit Center 02": {"stops": ["DC_NS_052", "DC_R50H_L1_007"]},
        "Mock Transit Center 03": {"stops": ["DC_R40_131", "DC_R50H_L0_000", "DC_R50H_L3_007"]},
    }
    assert "#   No stops, left out: Mock Transit Center 04, Mock Transit Center 05" in text
    assert "No stops in zone(s) Mock Transit Center 04, Mock Transit Center 05" in caplog.text
    # Stops within the default 250 ft of a zone; the next closest are 414 ft (Center 01),
    # 432 ft (03), 643 ft (02), 690 ft (04) and 1,047 ft (05) away.
    near_miss_lines = [line for line in text.splitlines() if " ft | " in line]
    assert near_miss_lines == [
        "#   Mock Transit Center 01: DC_R40_087, 24 ft | MASSACHUSETTS AVE NW @ CAPITOL ST",
        "#   Mock Transit Center 01: DC_R40_089, 113 ft | MASSACHUSEUTS AVE NW",
        "#   Mock Transit Center 03: DC_R40_132, 149 ft | MASSACHUSETTS AVE NW",
        "#   Mock Transit Center 04: DC_NS_017, 34 ft | CAPITOL ST @ VIRGINIA RD",
        "#   Mock Transit Center 04: DC_R50H_L3_000, 35 ft | VIRGINIA RD @ CAPITOL ST",
    ]
    assert not {"DC_R40_087", "DC_NS_017", "DC_R50H_L3_000"} & set(values["STOP_ID_FILTER"])
    assert "5 stop(s) are outside every zone but within 250 ft" in caplog.text
    runlog = (out_dir / "cluster_stops_from_zones_runlog.txt").read_text(encoding="utf-8")
    assert "Near-miss feet:   250" in runlog
    assert f"Zone layer: {STOP_CLUSTERS_ZIP}" in runlog
    assert "not fingerprinted" not in runlog  # the archive itself is fingerprinted


@pytest.mark.parametrize("near_miss_feet", ["-1", "nan"])
def test_main_bad_near_miss_feet_exits_1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, near_miss_feet: str
) -> None:
    paths = _write_inputs(tmp_path)
    assert target.main(_cli(paths, "--near-miss-feet", near_miss_feet)) == 1
    assert "NEAR_MISS_FEET must be 0 (off) or a positive number of feet" in caplog.text


def test_main_overlapping_zones_exit_1(tmp_path: Path) -> None:
    layer = _layer(["Metro", "Park & Ride"], [METRO, OVERLAPS_METRO])
    assert target.main(_cli(_write_inputs(tmp_path, layer))) == 1


def test_main_output_filename_with_a_folder_exits_1(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    argv = _cli(paths, "--output-dir", str(out_dir), "--output-filename", "sub/stops.txt")
    assert target.main(argv) == 1
    assert not out_dir.exists()
