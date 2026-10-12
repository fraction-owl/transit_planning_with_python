from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.gtfs_exports.gtfs_feed_merger as gfm
from scripts.gtfs_exports.gtfs_feed_merger import (
    MergeError,
    MergeSettings,
    allocate_ids,
    chord_ft_to_ground_ft,
    default_feed_labels,
    find_broken_references,
    flag_bay_stops,
    ground_ft_to_chord_ft,
    heading_difference_deg,
    initial_bearing_deg,
    load_feed,
    load_stop_overrides,
    merge_feeds,
    read_gtfs_tables,
)

# One foot of latitude in degrees (1 degree of latitude is ~364,000 ft).
FT = 1.0 / 364_000.0

# ---------------------------------------------------------------------------
# Synthetic two-agency network along an east-west Main St at latitude 38.9.
#
# Agency A (dominant): eastbound route R1 at stops 1 -> 2 -> 3, and route R2
# from transit-center bay 9 to stop 1.
# Agency B: eastbound route R1 (same route_id as A's) whose stop 21 sits 40 ft
# north of A's stop 2 (a duplicate of it); westbound route R5 whose stops 31
# and 32 sit 45 ft south of A's stops 3 and 2 (across the street); and a bay
# 99 at the same transit center, 30 ft from A's bay 9.
# ---------------------------------------------------------------------------

FEED_A = {
    "agency": [
        "agency_id,agency_name,agency_url,agency_timezone",
        "1,Agency A,https://a.example,America/New_York",
    ],
    "stops": [
        "stop_id,stop_name,stop_lat,stop_lon",
        "1,Main St & 1st St,38.9000000,-77.0000",
        "2,Main St & 2nd St,38.9000000,-76.9950",
        "3,Main St & 3rd St,38.9000000,-76.9900",
        "9,Transit Center Bay A,38.9100000,-77.0100",
    ],
    "routes": [
        "route_id,agency_id,route_short_name,route_long_name,route_type",
        "R1,1,10,Main Street,3",
        "R2,1,20,Transit Center,3",
    ],
    "trips": [
        "route_id,service_id,trip_id,direction_id,block_id",
        "R1,WKDY,T1,0,B1",
        "R2,WKDY,T2,0,B2",
    ],
    "stop_times": [
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
        "T1,08:00:00,08:00:00,1,1",
        "T1,08:05:00,08:05:00,2,2",
        "T1,08:10:00,08:10:00,3,3",
        "T2,07:40:00,07:40:00,9,1",
        "T2,07:50:00,07:50:00,1,2",
    ],
    "calendar": [
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date",
        "WKDY,1,1,1,1,1,0,0,20260101,20261231",
    ],
    "transfers": [
        "from_stop_id,to_stop_id,transfer_type,min_transfer_time",
        "2,2,2,120",
    ],
    "feed_info": [
        "feed_publisher_name,feed_publisher_url,feed_lang,feed_version",
        "Agency A,https://a.example,en,A-2026",
    ],
}


def _lat(offset_ft: float) -> str:
    return f"{38.9 + offset_ft * FT:.7f}"


FEED_B = {
    "agency": [
        "agency_name,agency_url,agency_timezone",
        "Agency B,https://b.example,America/New_York",
    ],
    "stops": [
        "stop_id,stop_name,stop_lat,stop_lon",
        "1,Far Away Rd,38.9500000,-77.0500",
        f"20,Main St & Oak St,{_lat(40)},-77.0050",
        f"21,Main St & 2nd St,{_lat(40)},-76.9950",
        f"22,Main St & Elm St,{_lat(40)},-76.9850",
        f"30,Main St & Elm St WB,{_lat(-45)},-76.9850",
        f"31,Main St & 3rd St WB,{_lat(-45)},-76.9900",
        f"32,Main St & 2nd St WB,{_lat(-45)},-76.9950",
        f"99,Transit Center Bay 3,{38.91 + 30 * FT:.7f},-77.0100",
    ],
    "routes": [
        "route_id,route_short_name,route_long_name,route_type",
        "R1,10,Main Street,3",
        "R5,50,Main Street WB,3",
    ],
    "trips": [
        "route_id,service_id,trip_id,direction_id,block_id",
        "R1,WKDY,T1,0,B1",
        "R5,WKDY,T5,1,B5",
        "R5,WKDY,T6,1,B6",
    ],
    "stop_times": [
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
        "T1,08:01:00,08:01:00,20,1",
        "T1,08:06:00,08:06:00,21,2",
        "T1,08:11:00,08:11:00,22,3",
        "T5,09:00:00,09:00:00,30,1",
        "T5,09:05:00,09:05:00,31,2",
        "T5,09:10:00,09:10:00,32,3",
        "T6,10:00:00,10:00:00,99,1",
        "T6,10:10:00,10:10:00,1,2",
    ],
    "calendar": [
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date",
        "WKDY,1,1,1,1,1,0,0,20260301,20260630",
    ],
    "transfers": [
        "from_stop_id,to_stop_id,transfer_type,min_transfer_time",
        "21,21,2,120",
    ],
}


def _write_feed(folder: Path, files: dict[str, list[str]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        (folder / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def _feeds(tmp_path: Path, b_files: dict[str, list[str]] | None = None) -> list[gfm.Feed]:
    a = _write_feed(tmp_path / "a", FEED_A)
    b = _write_feed(tmp_path / "b", b_files or FEED_B)
    return [load_feed(str(a), "A"), load_feed(str(b), "B")]


def _outcome(result: gfm.MergeResult, stop_id: str) -> list[str]:
    rows = result.stop_matches
    return rows.loc[rows["stop_id"] == stop_id, "outcome"].tolist()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_initial_bearing_points_east_and_north() -> None:
    bearing = initial_bearing_deg(
        np.array([38.9, 38.9]),
        np.array([-77.0, -77.0]),
        np.array([38.9, 39.0]),
        np.array([-76.9, -77.0]),
    )
    assert bearing[0] == pytest.approx(90.0, abs=0.1)
    assert bearing[1] == pytest.approx(0.0, abs=0.1)


def test_initial_bearing_same_point_is_nan() -> None:
    bearing = initial_bearing_deg(
        np.array([38.9]), np.array([-77.0]), np.array([38.9]), np.array([-77.0])
    )
    assert np.isnan(bearing[0])


def test_heading_difference_wraps_around_north() -> None:
    assert heading_difference_deg(np.array([350.0]), np.array([10.0])) == pytest.approx(20.0)
    assert heading_difference_deg(np.array([90.0]), np.array([270.0, 100.0])) == pytest.approx(10.0)
    assert heading_difference_deg(np.array([]), np.array([10.0])) is None


def test_chord_and_ground_distance_round_trip() -> None:
    assert chord_ft_to_ground_ft(ground_ft_to_chord_ft(100.0)) == pytest.approx(100.0)
    one_ft = gfm.to_xyz_ft(np.array([38.9, 38.9 + FT * 100]), np.array([-77.0, -77.0]))
    chord = float(np.linalg.norm(one_ft[0] - one_ft[1]))
    assert chord_ft_to_ground_ft(chord) == pytest.approx(100.0, rel=0.01)


# ---------------------------------------------------------------------------
# IDs, labels, bays, overrides
# ---------------------------------------------------------------------------


def test_allocate_ids_prefixes_only_collisions() -> None:
    used = {"1", "B_2"}
    mapping = allocate_ids(
        ["1", "2", "5"], used, "B", dominant=False, always_prefix=False, separator="_"
    )
    assert mapping == {"1": "B_1", "2": "2", "5": "5"}


def test_allocate_ids_always_mode_and_counter() -> None:
    used = {"B_1"}
    mapping = allocate_ids(["1", "2"], used, "B", dominant=False, always_prefix=True, separator="_")
    assert mapping == {"1": "B_1_2", "2": "B_2"}


def test_allocate_ids_dominant_keeps_ids() -> None:
    used: set[str] = set()
    assert allocate_ids(["1"], used, "A", dominant=True, always_prefix=True, separator="_") == {
        "1": "1"
    }
    assert used == {"1"}


def test_default_feed_labels_uses_names_and_dedupes() -> None:
    assert default_feed_labels(["x/google_transit.zip", "y/google_transit.zip", "z/wmata"]) == [
        "google_transit",
        "google_transit_2",
        "wmata",
    ]


def test_flag_bay_stops_by_parent_name_and_zone() -> None:
    stops = pd.DataFrame(
        {
            "stop_id": ["a", "b", "c", "d"],
            "stop_name": ["Main St", "Center Bay 4", "Main St", "Baywood Rd"],
            "parent_station": ["STA", "", "", ""],
        }
    )
    assert flag_bay_stops(stops, [r"\bbay\b"], {"c"}).tolist() == [True, True, True, False]


def test_load_stop_overrides_validates_priority(tmp_path: Path) -> None:
    path = tmp_path / "overrides.csv"
    path.write_text("feed,stop_id,target_feed,target_stop_id\nB,21,,\nB,31,A,3\n")
    assert load_stop_overrides(str(path), ["A", "B"]) == {
        ("B", "21"): None,
        ("B", "31"): ("A", "3"),
    }

    path.write_text("feed,stop_id,target_feed,target_stop_id\nA,2,B,21\n")
    with pytest.raises(MergeError, match="dominant"):
        load_stop_overrides(str(path), ["A", "B"])


# ---------------------------------------------------------------------------
# merge_feeds
# ---------------------------------------------------------------------------


def test_merge_combines_duplicate_stop_into_dominant_stop(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings())

    stops = result.tables["stops"]
    assert "21" not in set(stops["stop_id"])  # B's duplicate is gone
    stop_times = result.tables["stop_times"]
    b_trip = stop_times[stop_times["trip_id"] == "B_T1"].sort_values("stop_sequence")
    assert b_trip["stop_id"].tolist() == ["20", "2", "22"]  # B's trip now serves A's stop 2
    assert _outcome(result, "21") == ["merged"]

    crosswalk = result.crosswalk
    row = crosswalk[(crosswalk["id_type"] == "stop") & (crosswalk["original_id"] == "21")]
    assert row[["merged_id", "change"]].to_numpy().tolist() == [["2", "merged"]]


def test_merge_keeps_across_the_street_stops_separate(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings())
    stop_ids = set(result.tables["stops"]["stop_id"])
    assert {"31", "32"} <= stop_ids
    assert _outcome(result, "31") == ["kept_heading"]
    assert _outcome(result, "32") == ["kept_heading"]


def test_merge_keeps_bays_separate_by_default(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings())
    assert "99" in set(result.tables["stops"]["stop_id"])
    assert _outcome(result, "99") == ["kept_bay"]
    # Without a polygon layer the "Bay" names make 9 and 99 bays already, so they
    # are not reported as a possible cluster.
    assert result.possible_bay_clusters.empty


def test_merge_bays_when_bay_distance_allows(tmp_path: Path) -> None:
    # The two bays' buses leave in opposite directions; bays ignore the heading check.
    result = merge_feeds(_feeds(tmp_path), MergeSettings(bay_match_distance_ft=50.0))
    assert "99" not in set(result.tables["stops"]["stop_id"])
    assert _outcome(result, "99") == ["merged"]


def test_merge_without_heading_check_takes_closest_pair(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings(heading_tolerance_deg=180.0))
    # 21 (40 ft) and 32 (45 ft) both fit A's stop 2; only the closer one merges.
    assert _outcome(result, "21") == ["merged"]
    assert _outcome(result, "32") == ["kept_target_taken"]
    assert _outcome(result, "31") == ["merged"]


def test_merge_zero_distance_merges_nothing(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings(stop_match_distance_ft=0.0))
    assert result.stop_matches.empty
    assert len(result.tables["stops"]) == len(FEED_A["stops"]) - 1 + len(FEED_B["stops"]) - 1


def test_merge_renames_colliding_ids_and_fills_agency(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings())
    tables = result.tables

    assert set(tables["stops"]["stop_id"]) >= {"1", "B_1", "20"}  # only the collision renamed
    assert set(tables["routes"]["route_id"]) == {"R1", "R2", "B_R1", "R5"}
    assert tables["agency"]["agency_id"].tolist() == ["1", "B"]
    b_route = tables["routes"].set_index("route_id").loc["B_R1"]
    assert b_route["agency_id"] == "B"
    trips = tables["trips"].set_index("trip_id")
    assert trips.loc["B_T1", ["route_id", "service_id", "block_id"]].tolist() == [
        "B_R1",
        "B_WKDY",
        "B_B1",
    ]
    assert trips.loc["T1", ["route_id", "service_id"]].tolist() == ["R1", "WKDY"]
    # B's stop 1 collided; its trip now points at the renamed stop.
    t6 = tables["stop_times"][tables["stop_times"]["trip_id"] == "T6"]
    assert t6["stop_id"].tolist() == ["99", "B_1"]
    assert find_broken_references(tables) == []


def test_merge_always_prefix_mode(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings(id_prefix_mode="always"))
    stops = set(result.tables["stops"]["stop_id"])
    assert {"1", "2", "B_1", "B_20", "B_22"} <= stops
    assert "20" not in stops
    assert result.tables["agency"]["agency_id"].tolist() == ["1", "B"]  # agency never prefixed


def test_merge_clips_calendars_to_common_window(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings())
    assert result.window == (dt.date(2026, 3, 2), dt.date(2026, 6, 30))
    calendar = result.tables["calendar"].set_index("service_id")
    assert calendar.loc["WKDY", ["start_date", "end_date"]].tolist() == ["20260302", "20260630"]
    assert calendar.loc["B_WKDY", ["start_date", "end_date"]].tolist() == ["20260302", "20260630"]
    info = result.tables["feed_info"].iloc[0]
    assert (info["feed_start_date"], info["feed_end_date"]) == ("20260302", "20260630")
    assert info["feed_version"] == "merged: A=A-2026; B=unversioned"


def test_merge_without_clipping_keeps_calendars(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings(clip_to_common_window=False))
    assert result.window is None
    calendar = result.tables["calendar"].set_index("service_id")
    assert calendar.loc["WKDY", "start_date"] == "20260101"
    assert "feed_start_date" not in result.tables["feed_info"].columns


def test_merge_drops_trips_outside_window(tmp_path: Path) -> None:
    b_files = dict(FEED_B)
    b_files["calendar"] = [
        *FEED_B["calendar"],
        "OLD,1,1,1,1,1,0,0,20250101,20251231",
    ]
    b_files["trips"] = [*FEED_B["trips"], "R5,OLD,T7,1,B7"]
    b_files["stop_times"] = [*FEED_B["stop_times"], "T7,11:00:00,11:00:00,30,1"]
    result = merge_feeds(_feeds(tmp_path, b_files), MergeSettings())
    assert "T7" not in set(result.tables["trips"]["trip_id"])
    assert "T7" not in set(result.tables["stop_times"]["trip_id"])
    assert "B_OLD" not in set(result.tables["calendar"]["service_id"])
    summary = result.feed_summary.set_index("feed")
    assert summary.loc["B", "trips_dropped_outside_window"] == 1


def test_merge_rejects_non_overlapping_calendars(tmp_path: Path) -> None:
    b_files = dict(FEED_B)
    b_files["calendar"] = [FEED_B["calendar"][0], "WKDY,1,1,1,1,1,0,0,20270101,20270301"]
    with pytest.raises(MergeError, match="empty"):
        merge_feeds(_feeds(tmp_path, b_files), MergeSettings())


def test_merge_rejects_mixed_timezones(tmp_path: Path) -> None:
    b_files = dict(FEED_B)
    b_files["agency"] = [FEED_B["agency"][0], "Agency B,https://b.example,America/Chicago"]
    with pytest.raises(MergeError, match="timezone"):
        merge_feeds(_feeds(tmp_path, b_files), MergeSettings())


def test_merge_dedupes_transfers_on_merged_stops(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings())
    transfers = result.tables["transfers"]
    assert transfers[["from_stop_id", "to_stop_id"]].to_numpy().tolist() == [["2", "2"]]


def test_merge_applies_overrides(tmp_path: Path) -> None:
    overrides = {("B", "21"): None, ("B", "31"): ("A", "3")}
    result = merge_feeds(_feeds(tmp_path), MergeSettings(), overrides)
    stops = set(result.tables["stops"]["stop_id"])
    assert "21" in stops and "31" not in stops
    assert _outcome(result, "21") == ["kept_override"]
    assert _outcome(result, "31") == ["merged_override"]


def test_merge_three_feeds_matches_against_all_earlier_feeds(tmp_path: Path) -> None:
    feed_c = {
        "agency": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "C,Agency C,https://c.example,America/New_York",
        ],
        "stops": [
            "stop_id,stop_name,stop_lat,stop_lon",
            f"20,Main St & 2nd St,{_lat(-20)},-76.9950",
            f"22,Main St & Elm,{_lat(30)},-76.9850",
        ],
        "routes": ["route_id,agency_id,route_short_name,route_type", "C1,C,C1,3"],
        "trips": ["route_id,service_id,trip_id", "C1,DAILY,C_T1"],
        "stop_times": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "C_T1,08:00:00,08:00:00,20,1",
            "C_T1,08:10:00,08:10:00,22,2",
        ],
        "calendar_dates": ["service_id,date,exception_type", "DAILY,20260401,1"],
    }
    feeds = _feeds(tmp_path)
    feeds.append(load_feed(str(_write_feed(tmp_path / "c", feed_c)), "C"))
    result = merge_feeds(feeds, MergeSettings())

    rows = result.stop_matches[result.stop_matches["feed"] == "C"]
    pairs = set(zip(rows["stop_id"], rows["target_feed"], rows["merged_stop_id"], rows["outcome"]))
    # C's stop 20 merges into A's stop 2 (which already absorbed B's 21) and C's 22
    # into B's 22; B's westbound stops across the street stay separate.
    assert pairs == {
        ("20", "A", "2", "merged"),
        ("20", "B", "32", "kept_heading"),
        ("22", "B", "22", "merged"),
    }
    c_stops = result.tables["stop_times"][result.tables["stop_times"]["trip_id"] == "C_T1"]
    assert c_stops["stop_id"].tolist() == ["2", "22"]
    assert result.window == (dt.date(2026, 4, 1), dt.date(2026, 4, 1))
    assert find_broken_references(result.tables) == []


# ---------------------------------------------------------------------------
# Reading feeds and main()
# ---------------------------------------------------------------------------


def test_read_gtfs_tables_from_zip_with_wrapper_folder(tmp_path: Path) -> None:
    folder = _write_feed(tmp_path / "b", FEED_B)
    (folder / "fare_attributes.txt").write_text("fare_id,price\nF,2.00\n")
    zip_path = tmp_path / "b.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in folder.iterdir():
            archive.write(file, arcname=f"wrapper/{file.name}")
        archive.writestr("__MACOSX/wrapper/._stops.txt", "junk")
    tables, other = read_gtfs_tables(str(zip_path))
    assert len(tables["stops"]) == len(FEED_B["stops"]) - 1
    assert other == ["fare_attributes.txt"]


def test_read_gtfs_tables_keeps_text_values(tmp_path: Path) -> None:
    folder = _write_feed(
        tmp_path / "f", {"routes": ["route_id,route_short_name,route_color", "001,NA,"]}
    )
    tables, _ = read_gtfs_tables(str(folder))
    assert tables["routes"].iloc[0].tolist() == ["001", "NA", ""]


def test_main_placeholder_returns_2() -> None:
    assert gfm.main([]) == 2


def test_main_end_to_end(tmp_path: Path) -> None:
    a = _write_feed(tmp_path / "in" / "a", FEED_A)
    b = _write_feed(tmp_path / "in" / "b", FEED_B)
    out = tmp_path / "out"
    code = gfm.main(
        [
            "--gtfs-feeds",
            str(a),
            str(b),
            "--feed-labels",
            "A",
            "B",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    for name in (
        "merge_stop_matches.csv",
        "merge_id_crosswalk.csv",
        "merge_feed_summary.csv",
        "merge_possible_bay_clusters.csv",
    ):
        assert (out / name).exists()
    runlog = (out / "merged_gtfs_runlog.txt").read_text(encoding="utf-8")
    assert "STOP_MATCH_DISTANCE_FT" in runlog

    merged, other = read_gtfs_tables(str(out / "merged_gtfs.zip"))
    assert other == []
    assert set(merged) >= {"agency", "stops", "routes", "trips", "stop_times", "feed_info"}
    assert "21" not in set(merged["stops"]["stop_id"])
    assert find_broken_references(merged) == []


def test_main_rejects_mismatched_labels(tmp_path: Path) -> None:
    a = _write_feed(tmp_path / "a", FEED_A)
    b = _write_feed(tmp_path / "b", FEED_B)
    code = gfm.main(
        ["--gtfs-feeds", str(a), str(b), "--feed-labels", "A", "--output-dir", str(tmp_path)]
    )
    assert code == 2


def test_merge_treats_stops_in_bay_zones_as_bays(tmp_path: Path) -> None:
    # A square around A's stop 2 and B's duplicate 21 makes both bays, so the
    # street-stop merge no longer applies.
    zone = tmp_path / "zones.geojson"
    zone.write_text(
        '{"type": "FeatureCollection", "crs": {"type": "name", "properties": '
        '{"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": [{"type": "Feature", '
        '"properties": {"name": "Hub"}, "geometry": {"type": "Polygon", "coordinates": '
        "[[[-76.996, 38.8995], [-76.994, 38.8995], [-76.994, 38.9005], [-76.996, 38.9005], "
        "[-76.996, 38.8995]]]}}]}"
    )
    zones = gfm.load_bay_zones(str(zone))
    result = merge_feeds(_feeds(tmp_path), MergeSettings(), zones=zones)
    assert _outcome(result, "21") == ["kept_bay"]
    assert "21" in set(result.tables["stops"]["stop_id"])

    # With polygons, names no longer make bays: the transit-center bays 9 and 99
    # (outside the polygon) follow the street-stop rules (their buses leave in
    # opposite directions) and are reported as a possible undrawn bay cluster.
    assert _outcome(result, "99") == ["kept_heading"]
    clusters = result.possible_bay_clusters
    assert set(zip(clusters["feed"], clusters["stop_id"])) == {("A", "9"), ("B", "99")}
    assert set(clusters["reason"]) == {"bay names"}


def _cluster_candidates(names: list[str], terminals: list[bool]) -> pd.DataFrame:
    # Stops 80 ft apart in a row along a curb.
    n = len(names)
    return pd.DataFrame(
        {
            "feed": ["A"] * n,
            "stop_id": [f"s{i}" for i in range(n)],
            "stop_name": names,
            "stop_lat": [_lat(80 * i) for i in range(n)],
            "stop_lon": ["-77.0"] * n,
            "starts_or_ends_trips": terminals,
        }
    )


def test_find_possible_bay_clusters_by_name() -> None:
    candidates = _cluster_candidates(["Center Bay 1", "Center Bay 2", "Elm St"], [False] * 3)
    far = _cluster_candidates(["Far Bay 9"], [False]).assign(stop_id="far", stop_lat="39.5")
    clusters = gfm.find_possible_bay_clusters(
        pd.concat([candidates, far], ignore_index=True), [r"\bbay\b"], 150.0, 4
    )
    # The chained group is reported whole; the lone far stop is not a cluster.
    assert clusters["stop_id"].tolist() == ["s0", "s1", "s2"]
    assert clusters["bay_name"].tolist() == [True, True, False]
    assert set(clusters["reason"]) == {"bay names"}


def test_find_possible_bay_clusters_by_trip_starts_and_ends() -> None:
    names = ["Hub A", "Hub B", "Hub C", "Hub D"]
    clusters = gfm.find_possible_bay_clusters(
        _cluster_candidates(names, [True, True, False, False]), [r"\bbay\b"], 150.0, 4
    )
    assert clusters["reason"].unique().tolist() == ["trips start or end here"]
    assert len(clusters) == 4

    # One terminal of four (e.g. a busy street corner) is not reported, and
    # neither is anything once the gap is shorter than the stop spacing.
    one = _cluster_candidates(names, [True, False, False, False])
    assert gfm.find_possible_bay_clusters(one, [r"\bbay\b"], 150.0, 4).empty
    two = _cluster_candidates(names, [True, True, False, False])
    assert gfm.find_possible_bay_clusters(two, [r"\bbay\b"], 50.0, 4).empty
    assert gfm.find_possible_bay_clusters(two, [r"\bbay\b"], 0.0, 4).empty


def test_merge_keeps_unserved_street_stops_separate(tmp_path: Path) -> None:
    # B's stop 40 sits 30 ft from A's stop 3 on the same curb but has no trips,
    # so there is no direction to compare: it is listed, not merged.
    b_files = dict(FEED_B)
    b_files["stops"] = [*FEED_B["stops"], f"40,Main St & 3rd St,{_lat(30)},-76.9900"]
    result = merge_feeds(_feeds(tmp_path, b_files), MergeSettings())
    assert _outcome(result, "40") == ["kept_unserved"]

    no_check = merge_feeds(
        _feeds(tmp_path / "again", b_files), MergeSettings(heading_tolerance_deg=180.0)
    )
    assert _outcome(no_check, "40") == ["merged"]


# ---------------------------------------------------------------------------
# Route names shared by several agencies
# ---------------------------------------------------------------------------


def _name_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    agency = pd.DataFrame(
        {"agency_id": ["A", "B", "C"], "agency_name": ["Connector", "WMATA", "Ride On"]}
    )
    routes = pd.DataFrame(
        {
            "route_id": ["a10", "b10", "b10x", "c1", "a20", "b_silver", "c_silver"],
            "agency_id": ["A", "B", "B", "C", "A", "B", "C"],
            "route_short_name": ["10", "10", "10", "10_WMATA", "20", "", ""],
            "route_long_name": ["", "", "", "", "", "Silver Line", "Silver Line"],
        }
    )
    return routes, agency


def test_rename_duplicate_route_names_lower_priority() -> None:
    routes, agency = _name_tables()
    renamed, renames = gfm.rename_duplicate_route_names(routes, agency, "lower_priority")
    names = dict(zip(renamed["route_id"], renamed["route_short_name"]))
    # A keeps "10"; both of B's "10" variants get one new name, which skips the
    # "10_WMATA" that Ride On already uses.
    assert (names["a10"], names["b10"], names["b10x"]) == ("10", "10_WMATA_2", "10_WMATA_2")
    assert names["c1"] == "10_WMATA" and names["a20"] == "20"
    # Routes without a short name compare (and are renamed) by long name.
    long_names = dict(zip(renamed["route_id"], renamed["route_long_name"]))
    assert (long_names["b_silver"], long_names["c_silver"]) == (
        "Silver Line",
        "Silver Line_Ride On",
    )
    assert {(r["agency_id"], r["new_name"]) for r in renames} == {
        ("B", "10_WMATA_2"),
        ("C", "Silver Line_Ride On"),
    }


def test_rename_duplicate_route_names_all_and_keep() -> None:
    routes, agency = _name_tables()
    renamed, _ = gfm.rename_duplicate_route_names(routes, agency, "all")
    names = dict(zip(renamed["route_id"], renamed["route_short_name"]))
    assert (names["a10"], names["b10"]) == ("10_Connector", "10_WMATA_2")

    kept, renames = gfm.rename_duplicate_route_names(routes, agency, "keep")
    assert renames == []
    assert kept["route_short_name"].tolist() == routes["route_short_name"].tolist()


def test_merge_renames_lower_priority_route_names(tmp_path: Path) -> None:
    result = merge_feeds(_feeds(tmp_path), MergeSettings())
    routes = result.tables["routes"].set_index("route_id")
    assert routes.loc["R1", "route_short_name"] == "10"  # dominant keeps its name
    assert routes.loc["B_R1", "route_short_name"] == "10_Agency B"
    assert routes.loc["R5", "route_short_name"] == "50"
    crosswalk = result.crosswalk
    rows = crosswalk[crosswalk["id_type"] == "route_name"]
    assert rows[["feed", "original_id", "merged_id"]].to_numpy().tolist() == [
        ["B", "10", "10_Agency B"]
    ]
    summary = result.feed_summary.set_index("feed")
    assert summary["route_names_renamed"].to_dict() == {"A": 0, "B": 1}
