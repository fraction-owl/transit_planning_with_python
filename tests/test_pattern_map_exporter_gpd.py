from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for savefig

import pandas as pd
import pytest
from shapely.geometry import LineString

import scripts.gtfs_exports.pattern_map_exporter_gpd as target

# Stops along 38.90 N from west (A) to east (E); X sits north of the line.
STOPS = {
    "A": (38.90, -77.05),
    "B": (38.90, -77.04),
    "C": (38.90, -77.03),
    "D": (38.90, -77.02),
    "E": (38.90, -77.01),
    "X": (38.91, -77.025),
}


def _stop_rows(trip_id: str, stops: list[str], start_hour: int, **overrides: str) -> list[str]:
    rows = []
    for seq, stop_id in enumerate(stops, start=1):
        time = f"{start_hour:02d}:{seq * 5:02d}:00"
        pickup, drop_off = overrides.get(stop_id, ",").split(",")
        rows.append(f"{trip_id},{time},{time},{stop_id},{seq},{pickup},{drop_off}")
    return rows


def _write_feed(parent: Path, name: str = "feed", zipped: bool = False) -> Path:
    """Write a two-route feed with one example of every kind of pattern variation.

    Route 10, direction 0 (west to east):
      * main A-B-C-D-E: three weekday trips on SH_MAIN, one weekday trip on
        SH_COPY (identical coordinates), one Saturday trip -> P1, 5 trips;
      * short-turn A-B-C on the full SH_MAIN shape, two trips -> P2;
      * detour A-B-X-D-E on SH_DEV (adds X, drops C);
      * same stops as main, but D is pickup-only (drop_off_type 1);
      * express A-C-E on SH_MAIN.
    Route 10, direction 1 runs E-D-C-B-A on SH_MAIN, which is drawn west to east.
    Route 20 runs A-C-E with no shape_id.
    """
    feed_dir = parent / name
    feed_dir.mkdir(parents=True)
    (feed_dir / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name,route_type\n"
        "R10,10,Main Street,3\n"
        "R20,20,Crosstown,3\n",
        encoding="utf-8",
    )
    (feed_dir / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "WKD,1,1,1,1,1,0,0,20260101,20261231\n"
        "SAT,0,0,0,0,0,1,0,20260101,20261231\n",
        encoding="utf-8",
    )
    (feed_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id,direction_id,shape_id,trip_headsign\n"
        "R10,WKD,T1,0,SH_MAIN,East\n"
        "R10,WKD,T2,0,SH_MAIN,East\n"
        "R10,WKD,T3,0,SH_MAIN,East\n"
        "R10,WKD,T4,0,SH_COPY,East\n"
        "R10,SAT,T10,0,SH_MAIN,East\n"
        "R10,WKD,T5,0,SH_MAIN,Charlie Rd\n"
        "R10,WKD,T6,0,SH_MAIN,Charlie Rd\n"
        "R10,WKD,T7,0,SH_DEV,East via X\n"
        "R10,WKD,T8,0,SH_MAIN,East\n"
        "R10,WKD,T9,0,SH_MAIN,East Express\n"
        "R10,WKD,T11,1,SH_MAIN,West\n"
        "R20,WKD,T12,0,,East\n",
        encoding="utf-8",
    )
    main = ["A", "B", "C", "D", "E"]
    rows = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type"]
    for trip_id, hour in (("T1", 6), ("T2", 7), ("T3", 8), ("T4", 9), ("T10", 10)):
        rows += _stop_rows(trip_id, main, hour)
    rows += _stop_rows("T5", ["A", "B", "C"], 11)
    rows += _stop_rows("T6", ["A", "B", "C"], 12)
    rows += _stop_rows("T7", ["A", "B", "X", "D", "E"], 13)
    rows += _stop_rows("T8", main, 14, D="0,1")
    rows += _stop_rows("T9", ["A", "C", "E"], 15)
    rows += _stop_rows("T11", main[::-1], 16)
    rows += _stop_rows("T12", ["A", "C", "E"], 17)
    (feed_dir / "stop_times.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (feed_dir / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        + "".join(f"{sid},{sid} St,{lat},{lon}\n" for sid, (lat, lon) in STOPS.items()),
        encoding="utf-8",
    )
    shape_rows = ["shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence"]
    for shape_id, stops in (
        ("SH_MAIN", main),
        ("SH_COPY", main),
        ("SH_DEV", ["A", "B", "X", "D", "E"]),
    ):
        for seq, stop_id in enumerate(stops, start=1):
            lat, lon = STOPS[stop_id]
            shape_rows.append(f"{shape_id},{lat},{lon},{seq}")
    (feed_dir / "shapes.txt").write_text("\n".join(shape_rows) + "\n", encoding="utf-8")

    if zipped:
        zip_path = parent / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for member in sorted(feed_dir.iterdir()):
                archive.write(member, arcname=member.name)
        shutil.rmtree(feed_dir)
        return zip_path
    return feed_dir


def _pattern(
    summary: pd.DataFrame, route: str, direction: str, n_stops: int, **eq: object
) -> pd.Series:
    rows = summary[
        (summary["route"] == route)
        & (summary["direction_id"] == direction)
        & (summary["n_stops"] == n_stops)
    ]
    for col, value in eq.items():
        rows = rows[rows[col] == value]
    assert len(rows) == 1, rows
    return rows.iloc[0]


@pytest.fixture(scope="module")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, Path]:
    base = tmp_path_factory.mktemp("patterns")
    feed = _write_feed(base)
    out = base / "out"
    summary = target.run(gtfs_path=feed, output_dir=out, day_types=["weekday", "saturday"])
    return summary, out


def test_every_variation_becomes_its_own_pattern(default_run) -> None:
    summary, _ = default_run
    outbound = summary[(summary["route"] == "10") & (summary["direction_id"] == "0")]
    assert len(outbound) == 5
    assert list(outbound["pattern_id"]) == ["P1", "P2", "P3", "P4", "P5"]
    assert list(outbound["n_trips"]) == [5, 2, 1, 1, 1]

    main = outbound.iloc[0]
    # SH_COPY duplicates SH_MAIN's coordinates, so its trip joins the main pattern.
    assert main["shape_ids"] == "SH_COPY, SH_MAIN"
    assert main["differs_from_main"] == "main pattern (most trips)"
    assert (main["weekday_trips"], main["saturday_trips"]) == (4, 1)
    assert (main["weekday_first_departure"], main["weekday_last_departure"]) == ("06:05", "09:05")
    assert main["first_stop_name"] == "A St" and main["last_stop_name"] == "E St"

    short = outbound.iloc[1]
    assert (short["n_stops"], short["headsign"]) == (3, "Charlie Rd")
    assert short["differs_from_main"] == "vs P1: −2 stops"
    assert short["dropped_stop_ids"] == "D; E"
    # The short-turn reuses the full shape; its line is cut back to stop C.
    assert bool(short["shape_trimmed_to_stops"])
    assert short["length_mi"] == pytest.approx(main["length_mi"] / 2, rel=0.02)
    assert short["off_main_path_mi"] == 0


def test_detour_restriction_and_express_differences(default_run) -> None:
    summary, _ = default_run
    detour = _pattern(summary, "10", "0", 5, shape_ids="SH_DEV")
    assert (detour["stops_added_vs_main"], detour["stops_dropped_vs_main"]) == (1, 1)
    assert (detour["added_stop_ids"], detour["dropped_stop_ids"]) == ("X", "C")
    assert detour["off_main_path_mi"] > 0.5
    assert "mi off P1's path" in detour["differs_from_main"]

    restricted = _pattern(summary, "10", "0", 5, shape_ids="SH_MAIN")
    assert restricted["differs_from_main"] == "vs P1: different pickup/drop-off rules"
    assert restricted["restricted_stops"] == "D (pickup_only)"

    express = _pattern(summary, "10", "0", 3, headsign="East Express")
    assert express["differs_from_main"] == "vs P1: −2 stops"
    assert not bool(express["shape_trimmed_to_stops"])


def test_shared_shape_reversed_and_missing_shape_drawn_stop_to_stop(default_run) -> None:
    summary, _ = default_run
    inbound = _pattern(summary, "10", "1", 5)
    assert bool(inbound["shape_reversed"])
    assert inbound["first_stop_id"] == "E"

    no_shape = _pattern(summary, "20", "0", 3)
    assert no_shape["geometry_source"] == "stop-to-stop"
    assert no_shape["shape_ids"] == ""


def test_outputs_and_run_log_written(default_run) -> None:
    summary, out = default_run
    maps = sorted(p.name for p in (out / target.MAPS_SUBDIR).glob("*.png"))
    assert maps == ["route_10_dir0.png", "route_10_dir1.png", "route_20_dir0.png"]
    for map_file in summary["map_file"]:
        assert (out / map_file).exists()

    on_disk = pd.read_csv(out / target.SUMMARY_FILENAME)
    assert len(on_disk) == len(summary) == 7
    stops = pd.read_csv(out / target.STOPS_FILENAME, dtype=str)
    detour_stops = stops[
        (stops["route"] == "10") & (stops["direction_id"] == "0") & (stops["stop_id"] == "X")
    ]
    assert list(detour_stops["stop_name"]) == ["X St"]

    text = (out / target.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "# === BEGIN CONFIG ===" in text
    assert "weekday: 2026-" in text
    assert "Patterns found:       7" in text


def test_boarding_restrictions_merged_when_split_is_off(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path, zipped=True)
    summary = target.run(
        gtfs_path=feed,
        output_dir=tmp_path / "out",
        route_names=["10"],
        split_on_boarding=False,
        day_types=[],
    )
    outbound = summary[summary["direction_id"] == "0"]
    assert list(outbound["n_trips"]) == [6, 2, 1, 1]
    assert not any(c.startswith("weekday_") for c in summary.columns)
    assert set(summary["route"]) == {"10"}


def test_service_filter_limits_trips(tmp_path: Path) -> None:
    feed = _write_feed(tmp_path)
    summary = target.run(
        gtfs_path=feed, output_dir=tmp_path / "out", route_names=["10"], service_ids=["SAT"]
    )
    assert list(summary["n_trips"]) == [1]
    assert summary["saturday_trips"].iloc[0] == 1


def test_orient_and_trim_helpers() -> None:
    line = LineString([(0, 0), (100, 0)])
    reversed_line, flipped = target.orient_line_to_stops(line, [(90, 1), (50, 1), (10, 1)])
    assert flipped and reversed_line.coords[0] == (100, 0)

    trimmed, cut = target.trim_line_to_stops(line, [(20, 0), (60, 0)], slack=1.0)
    assert cut and trimmed.length == pytest.approx(40)

    loop = LineString([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
    same, cut = target.trim_line_to_stops(loop, [(0, 0), (100, 50), (0, 0)], slack=1.0)
    assert not cut and same.length == loop.length


def test_arrow_segments_point_along_travel() -> None:
    line = LineString([(0, 0), (100, 0)])
    arrows = target.arrow_segments(line, spacing=30)
    assert [round((t[0] + h[0]) / 2) for t, h in arrows] == [15, 45, 75]
    assert all(h[0] > t[0] for t, h in arrows)
    back = target.arrow_segments(LineString([(100, 0), (0, 0)]), spacing=30)
    assert all(h[0] < t[0] for t, h in back)
    assert len(target.arrow_segments(LineString([(0, 0), (10, 0)]), spacing=30)) == 1


def test_describe_difference_wording() -> None:
    main = {
        "pattern_id": "P1",
        "stop_ids": ("A", "B", "C"),
        "boarding": ("regular",) * 3,
        "path_key": "SH1",
        "geometry_source": "shapes.txt",
    }
    # A short-turn on its own shape: the stop change explains the shape change.
    short = {**main, "stop_ids": ("A", "B"), "boarding": ("regular",) * 2, "path_key": "SH2"}
    assert target.describe_difference(short, main, 0, 1, 0.0) == "vs P1: −1 stop"
    # Same stops on a separately digitized shape.
    redrawn = {**main, "path_key": "SH3"}
    assert (
        target.describe_difference(redrawn, main, 0, 0, 0.0)
        == "vs P1: different shape, same streets"
    )


def test_main_exit_codes(tmp_path: Path) -> None:
    assert target.main([]) == 2  # placeholder paths
    feed = _write_feed(tmp_path)
    out = tmp_path / "out"
    args = ["--gtfs-path", str(feed), "--output-dir", str(out)]
    assert target.main([*args, "--service-ids", "NOPE"]) == 2
    assert target.main([*args, "--routes", "99"]) == 1
    assert target.main([*args, "--day-types", "weekend"]) == 1
    assert target.main([*args, "--routes", "20"]) == 0
    assert (out / "pattern_maps" / "route_20_dir0.png").exists()
