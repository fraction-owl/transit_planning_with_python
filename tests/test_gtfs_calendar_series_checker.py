from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import scripts.gtfs_data_quality.gtfs_calendar_series_checker as target

CALENDAR_HEADER = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date"
)


def _weekday(service_id: str, start: str, end: str) -> str:
    return f"{service_id},1,1,1,1,1,0,0,{start},{end}"


def _feed(
    folder: Path,
    calendar: list[str],
    *,
    calendar_dates: list[str] | None = None,
    trip_services: list[str] | None = None,
    agency: str = "Metro Transit",
    feed_info: tuple[str, str] | None = None,
) -> Path:
    """Write a minimal feed; by default one trip uses each calendar service."""
    folder.mkdir(parents=True)
    (folder / "calendar.txt").write_text(
        "\n".join([CALENDAR_HEADER, *calendar]) + "\n", encoding="utf-8"
    )
    if calendar_dates is not None:
        (folder / "calendar_dates.txt").write_text(
            "\n".join(["service_id,date,exception_type", *calendar_dates]) + "\n",
            encoding="utf-8",
        )
    services = trip_services or [row.split(",")[0] for row in calendar]
    (folder / "trips.txt").write_text(
        "route_id,service_id,trip_id\n"
        + "".join(f"R1,{sid},T{i}\n" for i, sid in enumerate(services)),
        encoding="utf-8",
    )
    (folder / "agency.txt").write_text(
        f"agency_name,agency_url,agency_timezone\n{agency},https://example.com,America/New_York\n",
        encoding="utf-8",
    )
    if feed_info is not None:
        (folder / "feed_info.txt").write_text(
            "feed_publisher_name,feed_publisher_url,feed_lang,feed_start_date,feed_end_date\n"
            f"Metro,https://example.com,en,{feed_info[0]},{feed_info[1]}\n",
            encoding="utf-8",
        )
    return folder


def _inventory_row(tables: dict[str, pd.DataFrame], name: str) -> pd.Series:
    inventory = tables["feed_inventory"]
    return inventory[inventory["feed_name"] == name].iloc[0]


def test_overlapping_versions_flagged_with_files_to_review(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    new = _feed(
        tmp_path / "new",
        [_weekday("WKD2", "20260601", "20261231")],
        feed_info=("20260601", "20261231"),
    )
    old = _feed(
        tmp_path / "old",
        [_weekday("WKD", "20260105", "20260630")],
        feed_info=("20260105", "20260630"),
    )
    out = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        tables = target.run([new, old], out)

    overlaps = tables["feed_overlaps"]
    assert len(overlaps) == 1
    row = overlaps.iloc[0]
    assert (row["file_a"], row["file_b"]) == ("old", "new")  # earlier version first
    assert row["shared_service_days"] == 22  # the weekdays of June 2026
    assert str(row["shared_first_date"]) == "2026-06-01"
    assert str(row["shared_last_date"]) == "2026-06-30"
    assert row["files_to_review_a"] == (
        "calendar.txt (service_id WKD); "
        "feed_info.txt (feed_start_date 20260105, feed_end_date 20260630)"
    )
    assert row["files_to_review_b"].startswith("calendar.txt (service_id WKD2); feed_info.txt")

    dates = tables["feed_overlap_dates"]
    assert len(dates) == 22
    assert set(dates["service_ids_a"]) == {"WKD"}
    assert set(dates["service_ids_b"]) == {"WKD2"}

    assert _inventory_row(tables, "old")["status"] == "REVIEW"
    assert "Overlaps new" in _inventory_row(tables, "old")["warnings"]
    assert any(
        "Series overlap between old and new" in record.getMessage()
        and "calendar.txt (service_id WKD)" in record.getMessage()
        for record in caplog.records
    )

    written = pd.read_csv(out / "feed_inventory.csv")
    assert written.columns[0] == "feed_id"
    assert (out / "feed_overlaps.csv").exists()
    assert (out / "feed_overlap_dates.csv").exists()
    runlog = (out / "gtfs_calendar_series_checker_runlog.txt").read_text(encoding="utf-8")
    assert "# === BEGIN CONFIG ===" in runlog
    assert "Overlapping pairs: 1" in runlog


def test_back_to_back_versions_are_not_flagged(tmp_path: Path) -> None:
    old = _feed(
        tmp_path / "old",
        [_weekday("WKD", "20260105", "20260531")],
        feed_info=("20260105", "20260531"),
    )
    new = _feed(
        tmp_path / "new",
        [_weekday("WKD", "20260601", "20261231")],
        feed_info=("20260601", "20261231"),
    )
    tables = target.run([old, new], tmp_path / "out")
    assert tables["feed_overlaps"].empty
    assert tables["feed_overlap_dates"].empty
    assert set(tables["feed_inventory"]["status"]) == {"OK"}


def test_added_date_points_to_calendar_dates(tmp_path: Path) -> None:
    old = _feed(tmp_path / "old", [_weekday("WKD", "20260105", "20260531")])
    new = _feed(
        tmp_path / "new",
        [_weekday("WKD2", "20260601", "20261231")],
        calendar_dates=["WKD2,20260529,1"],  # a Friday old.zip already serves
    )
    row = target.run([old, new], tmp_path / "out")["feed_overlaps"].iloc[0]
    assert row["shared_service_days"] == 1
    assert row["files_to_review_a"] == "calendar.txt (service_id WKD)"
    assert row["files_to_review_b"] == "calendar_dates.txt (dates added for service_id WKD2)"


def test_overlapping_declared_periods_flagged_without_shared_dates(tmp_path: Path) -> None:
    old = _feed(
        tmp_path / "old",
        [_weekday("WKD", "20260105", "20260531")],
        feed_info=("20260105", "20260630"),
    )
    new = _feed(
        tmp_path / "new",
        [_weekday("WKD", "20260601", "20261231")],
        feed_info=("20260601", "20261231"),
    )
    row = target.run([old, new], tmp_path / "out")["feed_overlaps"].iloc[0]
    assert row["shared_service_days"] == 0
    assert str(row["declared_overlap_start"]) == "2026-06-01"
    assert str(row["declared_overlap_end"]) == "2026-06-30"
    assert row["files_to_review_a"].startswith("feed_info.txt")
    assert row["files_to_review_b"].startswith("feed_info.txt")


def test_differing_agency_name_flagged(tmp_path: Path) -> None:
    a = _feed(tmp_path / "a", [_weekday("S", "20240101", "20241231")], agency="Metro Transit")
    b = _feed(tmp_path / "b", [_weekday("S", "20250101", "20251231")], agency="metro  transit")
    c = _feed(tmp_path / "c", [_weekday("S", "20260101", "20261231")], agency="Other Bus")
    tables = target.run([a, b, c], tmp_path / "out")
    assert _inventory_row(tables, "a")["status"] == "OK"  # case and spacing ignored
    assert _inventory_row(tables, "b")["status"] == "OK"
    assert "differs from other feeds" in _inventory_row(tables, "c")["warnings"]


def test_two_way_agency_tie_flags_both(tmp_path: Path) -> None:
    a = _feed(tmp_path / "a", [_weekday("S", "20250101", "20251231")], agency="Metro Transit")
    b = _feed(tmp_path / "b", [_weekday("S", "20260101", "20261231")], agency="Other Bus")
    tables = target.run([a, b], tmp_path / "out")
    assert set(tables["feed_inventory"]["status"]) == {"REVIEW"}


def test_failed_feed_reported_and_rest_still_compared(tmp_path: Path) -> None:
    old = _feed(tmp_path / "old", [_weekday("WKD", "20260105", "20260630")])
    new = _feed(tmp_path / "new", [_weekday("WKD", "20260601", "20261231")])
    bad = _feed(tmp_path / "bad", [_weekday("WKD", "2026XX01", "20261231")])
    out = tmp_path / "out"
    rc = target.main(["--gtfs", str(old), str(new), str(bad), "--output-dir", str(out)])
    assert rc == 1
    inventory = pd.read_csv(out / "feed_inventory.csv")
    bad_row = inventory[inventory["feed_name"] == "bad"].iloc[0]
    assert bad_row["status"] == "ERROR"
    assert "expected YYYYMMDD" in bad_row["error"]
    overlaps = pd.read_csv(out / "feed_overlaps.csv")
    assert list(zip(overlaps["file_a"], overlaps["file_b"])) == [("old", "new")]


def test_unused_service_excluded_and_literal_na_id_kept(tmp_path: Path) -> None:
    feed = _feed(
        tmp_path / "feed",
        [_weekday("NA", "20260105", "20260109"), "UNUSED,0,0,0,0,0,1,1,20260105,20260111"],
        trip_services=["NA"],
    )
    row = _inventory_row(target.run([feed], tmp_path / "out"), "feed")
    assert row["error"] == ""
    assert row["service_ids"] == 2
    assert row["active_days"] == 5  # weekend-only UNUSED service is not counted
    assert row["unused_service_ids"] == "UNUSED"
    assert "unused service_id" in row["warnings"]


def test_collection_folder_with_wrapped_zip(tmp_path: Path) -> None:
    series = tmp_path / "series"
    old = _feed(tmp_path / "staging" / "old", [_weekday("WKD", "20260105", "20260630")])
    series.mkdir()
    with zipfile.ZipFile(series / "old.zip", "w") as archive:
        for path in old.iterdir():
            archive.write(path, f"old/{path.name}")  # files inside a wrapper folder
    _feed(series / "new", [_weekday("WKD", "20260601", "20261231")])
    tables = target.run(series, tmp_path / "out")
    assert sorted(tables["feed_inventory"]["feed_name"]) == ["new", "old.zip"]
    assert len(tables["feed_overlaps"]) == 1


def test_folder_without_any_feed_raises(tmp_path: Path) -> None:
    feed = tmp_path / "feed"
    feed.mkdir()
    (feed / "trips.txt").write_text("route_id,service_id,trip_id\nR1,A,T1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No GTFS ZIPs"):
        target.run(feed, tmp_path / "out")


def test_main_placeholder_paths_return_2() -> None:
    assert target.main([]) == 2
