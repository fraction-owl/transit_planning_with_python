from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pyproj import Transformer
from shapely.geometry import box

import scripts.facilities_tools.bus_bay_cluster_mapper as target

FIXTURES = Path(__file__).parent / "fixtures"
# Five transit-center zones in Maryland State Plane feet (EPSG:2248) over the mock DC
# feed; the same pair test_cluster_stops_from_zones_gpd.py checks.
STOP_CLUSTERS_ZIP = FIXTURES / "stop_clusters_sample.zip"
MOCK_GTFS_DC_ZIP = FIXTURES / "mock_gtfs_dc.zip"

# Synthetic stops sit at metre offsets in UTM zone 18N, the zone local_crs picks for
# Washington, DC, so review distances are exact. The "Center" cluster is a 100 m square
# around (E0, N0), drawn as two halves that share the name.
UTM = "EPSG:32618"
E0, N0 = Transformer.from_crs(4326, UTM, always_xy=True).transform(-77.0, 38.9)
TO_LONLAT = Transformer.from_crs(UTM, 4326, always_xy=True)
CENTER_HALVES = [box(E0 - 50, N0 - 50, E0, N0 + 50), box(E0, N0 - 50, E0 + 50, N0 + 50)]
EMPTY_LOT = box(E0 + 1000, N0 - 50, E0 + 1100, N0 + 50)
FEET = 1 / target.METERS_PER_FOOT


def _stop(stop_id: str, name: str, dx: float, dy: float, location_type: str = "") -> dict:
    lon, lat = TO_LONLAT.transform(E0 + dx, N0 + dy)
    return {
        "stop_id": stop_id,
        "stop_code": "",
        "stop_name": name,
        "stop_lat": f"{lat:.8f}",
        "stop_lon": f"{lon:.8f}",
        "location_type": location_type,
    }


def _visits(trip_id: str, *stops: str | tuple[str, str]) -> list[dict]:
    """stop_times rows; a (stop_id, "1") tuple is a stop with no pickup and no drop-off."""
    rows = []
    for sequence, stop in enumerate(stops, 1):
        stop_id, no_service = (stop, "") if isinstance(stop, str) else stop
        rows.append(
            {
                "trip_id": trip_id,
                "arrival_time": f"08:0{sequence}:00",
                "departure_time": f"08:0{sequence}:00",
                "stop_id": stop_id,
                "stop_sequence": str(sequence),
                "pickup_type": no_service,
                "drop_off_type": no_service,
            }
        )
    return rows


def _calendar(service_id: str, days: str) -> dict:
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    return {
        "service_id": service_id,
        **dict(zip(weekdays, days, strict=True)),
        "start_date": "20260101",
        "end_date": "20261231",
    }


# Feed A: Bays A and B are served; Bay C is not. Route 2 runs only on Saturdays and on
# Thursday 26 Nov 2026, when weekday service is removed; it neither picks up nor drops
# off at Bay A. Route R3 is a tram with only a long name. NEAR is 30 m (98.4 ft) east
# of the cluster, FAR 60 m (196.9 ft); STN is a station inside it.
FEED_A: dict[str, list[dict]] = {
    "agency.txt": [{"agency_id": "A1", "agency_name": "Agency A"}],
    "routes.txt": [
        {"route_id": "R1", "agency_id": "A1", "route_short_name": "1", "route_type": "3"},
        {"route_id": "R2", "agency_id": "A1", "route_short_name": "2", "route_type": "3"},
        {
            "route_id": "R3",
            "agency_id": "A1",
            "route_short_name": "",
            "route_long_name": "Tram Line",
            "route_type": "0",
        },
    ],
    "trips.txt": [
        {
            "route_id": "R1",
            "service_id": "WK",
            "trip_id": "T1",
            "direction_id": "0",
            "trip_headsign": "North",
        },
        {
            "route_id": "R1",
            "service_id": "WK",
            "trip_id": "T2",
            "direction_id": "1",
            "trip_headsign": "South",
        },
        {"route_id": "R2", "service_id": "SA", "trip_id": "T3", "direction_id": "0"},
        {"route_id": "R3", "service_id": "WK", "trip_id": "T4", "direction_id": "0"},
        {
            "route_id": "R1",
            "service_id": "WK",
            "trip_id": "T5",
            "direction_id": "0",
            "trip_headsign": "North Express",
        },
    ],
    "stop_times.txt": [
        *_visits("T1", "007", "010", "NEAR"),
        *_visits("T2", "010"),
        *_visits("T3", ("007", "1"), "010"),
        *_visits("T4", "010"),
        *_visits("T5", "007"),
    ],
    "stops.txt": [
        _stop("007", "Center Bay a", -20, 0),
        _stop("010", "Center Bay B", 20, 0),
        _stop("011", "Center Bay C", 0, -30),
        _stop("STN", "Center Station", 0, 0, location_type="1"),
        _stop("NEAR", "Main St", 80, 0),
        _stop("FAR", "Far St", 110, 0),
    ],
    "calendar.txt": [_calendar("WK", "1111100"), _calendar("SA", "0000010")],
    "calendar_dates.txt": [
        {"service_id": "WK", "date": "20261126", "exception_type": "2"},
        {"service_id": "SA", "date": "20261126", "exception_type": "1"},
    ],
    "feed_info.txt": [{"feed_start_date": "20260101", "feed_end_date": "20261231"}],
}

# Feed B reuses stop_id 007 for its own bay 5 m (16.4 ft) from feed A's 007; its trip
# has no direction_id.
FEED_B: dict[str, list[dict]] = {
    "agency.txt": [{"agency_id": "B1", "agency_name": "Agency B"}],
    "routes.txt": [{"route_id": "RX", "agency_id": "B1", "route_short_name": "X"}],
    "trips.txt": [{"route_id": "RX", "service_id": "DAILY", "trip_id": "TX"}],
    "stop_times.txt": _visits("TX", "007", "B2"),
    "stops.txt": [
        _stop("007", "Partner Bay 1", -20, 5),
        _stop("B2", "Partner Bay 2", 0, 40),
    ],
    "calendar.txt": [_calendar("DAILY", "1111111")],
}


def _write_feed(folder: Path, tables: dict[str, list[dict]]) -> Path:
    folder.mkdir(parents=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(folder / name, index=False)
    return folder


def _zip_folder(folder: Path, wrapper: str) -> Path:
    archive_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file in folder.iterdir():
            archive.write(file, f"{wrapper}/{file.name}")
    return archive_path


def _write_clusters(path: Path) -> Path:
    gpd.GeoDataFrame(
        {"NAME": ["Center", "Center", "Empty Lot"]},
        geometry=[*CENTER_HALVES, EMPTY_LOT],
        crs=UTM,
    ).to_crs("EPSG:2248").to_file(path)
    return path


def _write_inputs(folder: Path, **settings: Any) -> Path:
    """Feed A as a folder, feed B as a zip with a wrapper folder, relative JSON paths."""
    inputs = folder / "in"
    _write_feed(inputs / "feed_a", FEED_A)
    _zip_folder(_write_feed(folder / "build" / "feed_b", FEED_B), "feed_b")
    (folder / "build" / "feed_b.zip").rename(inputs / "feed_b.zip")
    _write_clusters(inputs / "clusters.shp")
    config = {
        "gtfs_sources": [
            {"feed_id": "A", "path": "feed_a"},
            {"feed_id": "B", "path": "feed_b.zip"},
        ],
        "cluster_sources": [{"source_id": "Bays", "path": "clusters.shp", "name_field": "NAME"}],
        "output_dir": "../out",
        "figure_dpi": 72,
        **settings,
    }
    path = inputs / "settings.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _run(config: Path, *extra: str) -> Path:
    """Run main and return the run folder it created."""
    output_dir = json.loads(config.read_text(encoding="utf-8"))["output_dir"]
    output = (config.parent / output_dir).resolve()
    before = set(output.glob("bus_bay_maps_*"))
    assert target.main(["--config", str(config), *extra]) == 0
    (run_folder,) = set(output.glob("bus_bay_maps_*")) - before
    return run_folder


def _csv(run_folder: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(run_folder / name, dtype=str, keep_default_na=False)


def _routes_by_stop(run_folder: Path) -> dict[str, str]:
    legend = _csv(run_folder, "cluster_stop_assignments.csv")
    return dict(zip(legend.feed_id + "::" + legend.stop_id, legend.route_directions, strict=True))


def _settings(tmp_path: Path, **overrides: Any) -> target.Settings:
    return target.Settings(
        gtfs_sources=[{"feed_id": "A", "path": str(tmp_path / "feed_a")}],
        cluster_sources=[{"source_id": "Bays", "path": str(tmp_path / "clusters.shp")}],
        output_dir=str(tmp_path / "out"),
        **overrides,
    )


@pytest.fixture(scope="module")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One all-service run of the synthetic inputs, shared by the output tests."""
    return _run(_write_inputs(tmp_path_factory.mktemp("default")))


# ---------------------------------------------------------------------------
# GTFS reading and service selection
# ---------------------------------------------------------------------------


def test_gtfs_reader_reads_a_wrapper_folder_zip_and_keeps_ids_as_text(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path / "feed", {"stops.txt": [_stop("007", "NA", 0, 0)]})
    stops = target.GTFSReader(str(_zip_folder(feed, "wrapper"))).table("stops.txt")
    assert stops.loc[0, "stop_id"] == "007"
    assert stops.loc[0, "stop_name"] == "NA"


def test_gtfs_reader_rejects_a_zip_holding_several_feeds(tmp_path: Path) -> None:
    archive_path = tmp_path / "feeds.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("2025/stops.txt", "stop_id\n1\n")
        archive.writestr("2026/stops.txt", "stop_id\n1\n")
    with pytest.raises(ValueError, match="multiple stops.txt"):
        target.GTFSReader(str(archive_path)).table("stops.txt")


def test_load_stops_drops_stations_and_namespaces_stop_ids(tmp_path: Path) -> None:
    rows = [_stop("007", "", 0, 0), _stop("STN", "Station", 0, 0, location_type="1")]
    _write_feed(tmp_path / "feed", {"stops.txt": rows})
    stops = target.load_stops({"feed_id": "A", "path": str(tmp_path / "feed")})
    assert stops.stop_key.tolist() == ["A::007"]
    assert stops.stop_name.tolist() == ["007"]  # a blank name falls back to the ID
    assert stops.crs.to_epsg() == 4326


def test_load_stops_rejects_out_of_range_coordinates(tmp_path: Path) -> None:
    row = {**_stop("BAD", "Bad", 0, 0), "stop_lat": "138.9"}
    _write_feed(tmp_path / "feed", {"stops.txt": [row]})
    with pytest.raises(ValueError, match="invalid stop coordinates"):
        target.load_stops({"feed_id": "A", "path": str(tmp_path / "feed")})


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("", None),
        ("2026-11-25", {"WK"}),  # Wednesday
        ("20261126", {"SA"}),  # Thursday: weekday service removed, Saturday added
        ("2026-11-28", {"SA"}),  # Saturday
        ("2027-01-04", set()),  # after the calendar ends
    ],
)
def test_active_service_ids_applies_weekdays_and_exceptions(
    tmp_path: Path, date: str, expected: set[str] | None
) -> None:
    reader = target.GTFSReader(str(_write_feed(tmp_path / "feed", FEED_A)))
    assert target.active_service_ids(reader, date) == expected


def test_active_service_ids_without_a_calendar_raises(tmp_path: Path) -> None:
    reader = target.GTFSReader(str(_write_feed(tmp_path / "feed", {"stops.txt": []})))
    with pytest.raises(ValueError, match="requires a service calendar"):
        target.active_service_ids(reader, "2026-11-25")


# ---------------------------------------------------------------------------
# Route/direction inventory
# ---------------------------------------------------------------------------


def _inventory(tmp_path: Path, **overrides: Any) -> pd.DataFrame:
    _write_feed(tmp_path / "feed_a", FEED_A)
    settings = _settings(tmp_path, **overrides)
    inventory, _ = target.route_direction_inventory(
        settings.gtfs_sources[0], {"007", "010", "011"}, settings
    )
    return inventory


def test_route_direction_inventory_counts_trips_by_stop_route_and_direction(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    route_1 = inventory[inventory.stop_id.eq("007") & inventory.route_id.eq("R1")]
    assert route_1[["direction_id", "trip_count", "trip_headsigns"]].to_dict("records") == [
        {"direction_id": "0", "trip_count": 2, "trip_headsigns": "North | North Express"}
    ]
    assert set(inventory.loc[inventory.stop_id.eq("010"), "route_direction_label"]) == {
        "1 (0)",
        "1 (1)",
        "2 (0)",
        "Tram Line (0)",  # long name when there is no short name
    }
    assert "011" not in set(inventory.stop_id)  # no trips stop there


@pytest.mark.parametrize(("include", "expected"), [(False, {"R1"}), (True, {"R1", "R2"})])
def test_route_direction_inventory_skips_non_passenger_visits_unless_asked(
    tmp_path: Path, include: bool, expected: set[str]
) -> None:
    inventory = _inventory(tmp_path, include_non_passenger_visits=include)
    assert set(inventory.loc[inventory.stop_id.eq("007"), "route_id"]) == expected


def test_route_direction_inventory_filters_route_types_and_relabels_directions(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, route_types=["3"], direction_labels={"A|R1|0": "Northbound"})
    assert "R3" not in set(inventory.route_id)
    labels = inventory.loc[inventory.route_id.eq("R1"), "route_direction_label"]
    assert set(labels) == {"1 (Northbound)", "1 (1)"}


def test_route_direction_inventory_rejects_trips_on_unknown_routes(tmp_path: Path) -> None:
    feed = {**FEED_A, "routes.txt": FEED_A["routes.txt"][:2]}  # drops R3
    _write_feed(tmp_path / "feed_a", feed)
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="trips reference missing routes"):
        target.route_direction_inventory(settings.gtfs_sources[0], {"010"}, settings)


# ---------------------------------------------------------------------------
# Cluster layers
# ---------------------------------------------------------------------------


def test_load_clusters_dissolves_names_within_a_layer_not_across_layers(tmp_path: Path) -> None:
    first = _write_clusters(tmp_path / "first.shp")
    second = _write_clusters(tmp_path / "second.gpkg")
    clusters = target.load_clusters(
        [
            {"source_id": "First", "path": str(first), "name_field": "NAME"},
            {"source_id": "Second", "path": str(second), "name_field": "NAME"},
        ],
        [],
    )
    assert clusters.cluster_id.tolist() == [
        "First::Center",
        "First::Empty Lot",
        "Second::Center",
        "Second::Empty Lot",
    ]
    center = gpd.GeoSeries([clusters.geometry[0]], crs=4326).to_crs(UTM).iloc[0]
    assert center.area == pytest.approx(100 * 100, rel=1e-4)  # both halves, one polygon


def test_load_clusters_numbers_unnamed_features_and_rejects_unknown_names(
    tmp_path: Path,
) -> None:
    source = {"source_id": "Bays", "path": str(_write_clusters(tmp_path / "c.shp"))}
    clusters = target.load_clusters([{**source, "name_field": ""}], [])
    assert clusters.cluster_name.tolist() == ["Cluster 1", "Cluster 2", "Cluster 3"]
    with pytest.raises(ValueError, match="Unknown cluster names"):
        target.load_clusters([{**source, "name_field": "NAME"}], ["Centre"])


@pytest.mark.filterwarnings("ignore:'crs' was not provided")
def test_load_clusters_without_a_crs_raises(tmp_path: Path) -> None:
    path = tmp_path / "no_crs.shp"
    gpd.GeoDataFrame({"NAME": ["Center"]}, geometry=[CENTER_HALVES[0]]).to_file(path)
    with pytest.raises(ValueError, match="no CRS"):
        target.load_clusters([{"source_id": "Bays", "path": str(path)}], [])


def test_vector_uri_needs_zip_member_when_a_zip_holds_several_shapefiles(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "layers.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a/one.shp", b"")
        archive.writestr("b/two.shp", b"")
    with pytest.raises(ValueError, match="set zip_member"):
        target.vector_uri(str(archive_path))
    assert target.vector_uri(str(archive_path), "b/two.shp").endswith("layers.zip!b/two.shp")


# ---------------------------------------------------------------------------
# Labels, names and settings
# ---------------------------------------------------------------------------


def test_short_stop_name_shortens_bay_names_only_when_asked() -> None:
    assert target.short_stop_name("Transit Center Bay c", True) == "Bay C"
    assert target.short_stop_name("Transit Center Bay c", False) == "Transit Center Bay c"
    assert target.short_stop_name("Main St & 1st Ave", True) == "Main St & 1st Ave"


def test_natural_key_sorts_numbers_numerically() -> None:
    assert sorted(["Bay 10", "Bay 2", "Bay 1"], key=target.natural_key) == [
        "Bay 1",
        "Bay 2",
        "Bay 10",
    ]


def test_map_filename_is_safe_and_distinguishes_sources() -> None:
    first = target.map_filename("Bays::Center / North")
    assert first.startswith("Bays_Center_North_") and first.endswith(".png")
    assert first != target.map_filename("Other::Center / North")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"Path\To\Output", True),
        ("path/to/output", True),
        (r"C:\work\Path\To", True),
        (r"C:\work\bus_bays", False),
        ("pathology/today", False),
    ],
)
def test_is_placeholder_matches_the_template_path_only(path: str, expected: bool) -> None:
    assert target.is_placeholder(path) is expected


@pytest.mark.parametrize(
    ("feed_ids", "message"),
    [(["A", "A"], "distinct feed_id"), (["A B"], "letters, numbers")],
)
def test_validate_settings_rejects_repeated_or_unsafe_feed_ids(
    tmp_path: Path, feed_ids: list[str], message: str
) -> None:
    settings = _settings(tmp_path)
    settings.gtfs_sources = [{"feed_id": f, "path": str(tmp_path)} for f in feed_ids]
    with pytest.raises(ValueError, match=message):
        target.validate_settings(settings)


def test_render_map_has_no_axis_labels_and_a_feet_scale_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(figure: Figure, *_args: Any, **_kwargs: Any) -> None:
        ax = figure.axes[0]
        seen.update(
            xlabel=ax.get_xlabel(),
            ylabel=ax.get_ylabel(),
            texts=[text.get_text() for text in ax.texts],
        )

    monkeypatch.setattr(Figure, "savefig", capture)
    center = gpd.GeoSeries([box(E0 - 50, N0 - 50, E0 + 50, N0 + 50)], crs=UTM).to_crs(4326)
    point = gpd.GeoSeries.from_xy(*TO_LONLAT.transform([E0], [N0]), crs=4326)
    stop = {
        "feed_id": ["A"],
        "map_ref": ["01"],
        "stop_code": [""],
        "stop_id": ["007"],
        "stop_name": ["Center Bay A"],
        "route_directions": ["1 (0)"],
        "map_label": ["01 Bay A"],
    }
    target.render_map(
        SimpleNamespace(cluster_name="Center", geometry=center.iloc[0]),
        gpd.GeoDataFrame(stop, geometry=point, crs=4326),
        _settings(tmp_path, figure_dpi=72),
        tmp_path / "map.png",
        [{"feed_id": "A", "feed_start": "20260101", "feed_end": "20261231"}],
        None,
    )
    assert seen["xlabel"] == seen["ylabel"] == ""
    # A 100 m square is drawn 144 m wide; the bar is the round length nearest 23% of that.
    assert "100 ft" in seen["texts"]
    assert not [text for text in seen["texts"] if text.endswith(" m")]


# ---------------------------------------------------------------------------
# main / end-to-end
# ---------------------------------------------------------------------------


def test_main_placeholders_exit_2_and_write_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    assert target.main([]) == 2
    assert list(tmp_path.iterdir()) == []
    assert r"GTFS_SOURCES 'Connector': Path\To\connector_gtfs.zip" in caplog.text
    assert r"OUTPUT_DIR: Path\To\Output" in caplog.text


def test_main_placeholder_output_dir_exits_2_before_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert target.main(["--config", str(config), "--output", r"Path\To\Output"]) == 2
    assert not list(tmp_path.glob("Path*"))
    assert not (tmp_path / "out").exists()


def test_main_lists_every_stop_in_the_cluster_with_its_routes(default_run: Path) -> None:
    legend = _csv(default_run, "cluster_stop_assignments.csv")
    assert set(legend.cluster_id) == {"Bays::Center"}  # Empty Lot holds no stop
    assert legend[["map_ref", "feed_id", "stop_id", "map_label"]].to_dict("records") == [
        {"map_ref": "01", "feed_id": "A", "stop_id": "007", "map_label": "01 Bay A"},
        {"map_ref": "02", "feed_id": "A", "stop_id": "010", "map_label": "02 Bay B"},
        {"map_ref": "03", "feed_id": "A", "stop_id": "011", "map_label": "03 Bay C"},
        {"map_ref": "04", "feed_id": "B", "stop_id": "007", "map_label": "04 Bay 1"},
        {"map_ref": "05", "feed_id": "B", "stop_id": "B2", "map_label": "05 Bay 2"},
    ]
    assert _routes_by_stop(default_run) == {
        "A::007": "1 (0)",
        "A::010": "1 (0); 1 (1); 2 (0); Tram Line (0)",
        "A::011": "No qualifying trips",
        "B::007": "X (?)",
        "B::B2": "X (?)",
    }
    assert set(legend.service_date) == {"ALL_SERVICES"}


def test_main_route_detail_matches_the_legend(default_run: Path) -> None:
    legend = _csv(default_run, "cluster_stop_assignments.csv")
    detail = _csv(default_run, "route_direction_assignments.csv")
    for row in legend.itertuples():
        labels = detail.loc[
            detail.cluster_id.eq(row.cluster_id) & detail.map_ref.eq(row.map_ref),
            "route_direction_label",
        ].tolist()
        if row.route_directions == "No qualifying trips":
            assert labels == []
        else:
            assert sorted(labels) == sorted(row.route_directions.split("; "))


def test_main_reports_near_misses_and_cross_feed_pairs_in_feet(default_run: Path) -> None:
    near = _csv(default_run, "near_miss_stops.csv")
    assert near[["cluster_name", "feed_id", "stop_id"]].to_dict("records") == [
        {"cluster_name": "Center", "feed_id": "A", "stop_id": "NEAR"}
    ]  # FAR, at 196.9 ft, is beyond the default 130 ft
    assert float(near.distance_ft[0]) == pytest.approx(30 * FEET, abs=0.2)

    shared = _csv(default_run, "cross_feed_nearby_stops.csv")
    assert shared[["feed_id_a", "stop_id_a", "feed_id_b", "stop_id_b"]].to_dict("records") == [
        {"feed_id_a": "A", "stop_id_a": "007", "feed_id_b": "B", "stop_id_b": "007"}
    ]
    assert float(shared.distance_ft[0]) == pytest.approx(5 * FEET, abs=0.2)


def test_main_draws_one_map_per_cluster(default_run: Path) -> None:
    index = _csv(default_run, "cluster_index.csv")
    assert index[["cluster_name", "stops"]].to_dict("records") == [
        {"cluster_name": "Center", "stops": "5"},
        {"cluster_name": "Empty Lot", "stops": "0"},
    ]
    for name in index.map_file:
        assert (default_run / "maps" / name).stat().st_size > 0


def test_main_writes_the_run_log(default_run: Path) -> None:
    runlog = (default_run / "bus_bay_cluster_mapper_runlog.txt").read_text(encoding="utf-8")
    assert "NEAR_MISS_FEET = 130.0" in runlog  # verbatim CONFIGURATION block
    assert '"figure_dpi": 72' in runlog  # the setting used, from the JSON config
    assert str((default_run.parent.parent / "in" / "feed_b.zip").resolve()) in runlog
    assert runlog.count("sha256:") >= 2


def test_main_service_date_selects_that_days_trips(tmp_path: Path) -> None:
    run_folder = _run(_write_inputs(tmp_path), "--service-date", "20261126")
    routes = _routes_by_stop(run_folder)
    # Only Saturday route 2 runs, and it neither picks up nor drops off at Bay A.
    assert routes["A::007"] == "No qualifying trips"
    assert routes["A::010"] == "2 (0)"
    assert routes["B::007"] == "X (?)"
    legend = _csv(run_folder, "cluster_stop_assignments.csv")
    assert set(legend.service_date) == {"2026-11-26"}


def test_main_can_leave_out_unserved_stops(tmp_path: Path) -> None:
    run_folder = _run(_write_inputs(tmp_path, include_unserved_stops=False))
    assert "A::011" not in _routes_by_stop(run_folder)


def test_rerun_config_repeats_the_run(tmp_path: Path) -> None:
    first = _run(_write_inputs(tmp_path, near_miss_feet=200))
    second = _run(first / "rerun_config.json")
    for name in ("cluster_stop_assignments.csv", "near_miss_stops.csv"):
        pd.testing.assert_frame_equal(_csv(first, name), _csv(second, name))
    assert set(_csv(second, "near_miss_stops.csv").stop_id) == {"NEAR", "FAR"}


def test_main_run_log_failure_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(_source_file: Path) -> str:
        raise OSError("source unreadable")

    monkeypatch.setattr(target, "extract_config_block", unreadable)
    assert target.main(["--config", str(_write_inputs(tmp_path))]) == 1


def test_main_run_log_failure_is_allowed_when_not_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(_source_file: Path) -> str:
        raise OSError("source unreadable")

    monkeypatch.setattr(target, "extract_config_block", unreadable)
    run_folder = _run(_write_inputs(tmp_path, require_run_log=False))
    assert not (run_folder / "bus_bay_cluster_mapper_runlog.txt").exists()


def test_main_maps_the_mock_dc_feed_with_the_zipped_state_plane_clusters(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = tmp_path / "in" / "settings.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "gtfs_sources": [{"feed_id": "DC", "path": str(MOCK_GTFS_DC_ZIP)}],
                "cluster_sources": [
                    {"source_id": "Centers", "path": str(STOP_CLUSTERS_ZIP), "name_field": "NAME"}
                ],
                "output_dir": "../out",
                "near_miss_feet": 250,
                "figure_dpi": 72,
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        run_folder = _run(config)

    legend = _csv(run_folder, "cluster_stop_assignments.csv")
    members = legend.groupby("cluster_name").stop_id.apply(set).to_dict()
    # The same memberships test_cluster_stops_from_zones_gpd.py expects; 04 and 05 are empty.
    assert members == {
        "Mock Transit Center 01": {"DC_NS_034", "DC_R30_044", "DC_R40_088"},
        "Mock Transit Center 02": {"DC_NS_052", "DC_R50H_L1_007"},
        "Mock Transit Center 03": {"DC_R40_131", "DC_R50H_L0_000", "DC_R50H_L3_007"},
    }
    assert not legend.route_directions.eq("No qualifying trips").any()

    near = _csv(run_folder, "near_miss_stops.csv")
    distances = dict(zip(near.stop_id, near.distance_ft.astype(float), strict=True))
    # That test's State Plane distances, to the nearest foot; these are measured in UTM.
    expected = {
        "DC_R40_087": 24,
        "DC_R40_089": 113,
        "DC_R40_132": 149,
        "DC_NS_017": 34,
        "DC_R50H_L3_000": 35,
    }
    assert distances.keys() == expected.keys()
    for stop_id, feet in expected.items():
        assert distances[stop_id] == pytest.approx(feet, abs=1)
    assert "5 nearby stop/cluster pairs fall outside the polygons" in caplog.text
    assert len(_csv(run_folder, "cluster_index.csv")) == 5
