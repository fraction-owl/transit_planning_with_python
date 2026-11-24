from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

import scripts.gtfs_data_quality.gtfs_calendar_inspector as target

FIXTURES = Path(__file__).parent / "fixtures"


def _summary_row(df: pd.DataFrame, service_id: str) -> pd.Series:
    return df[df["service_id"] == service_id].iloc[0]


def _day_type_row(df: pd.DataFrame, day_type: str) -> pd.Series:
    return df[df["day_type"] == day_type].iloc[0]


def test_run_holiday_negation_sees_through_negated_pattern(tmp_path: Path) -> None:
    service, day_types = target.run(
        gtfs_path=FIXTURES / "gtfs_holiday_negation", output_dir=tmp_path
    )

    hol = _summary_row(service, "HOL")
    # calendar.txt claims Mon/Wed/Fri for HOL; the expansion must reveal a
    # 5-days-per-span holiday service instead.
    assert hol["labels"] == "Holiday"
    assert hol["active_days"] == 5
    # 4 federal holidays + the day after Thanksgiving (not federal).
    assert hol["holiday_days"] == 4
    assert hol["trip_count"] == 3

    wkd = _summary_row(service, "WKD")
    assert wkd["labels"] == "Weekday"
    assert wkd["trip_count"] == 6
    assert _summary_row(service, "SUN")["labels"] == "Sunday"

    weekday = _day_type_row(day_types, "weekday")
    assert weekday["service_ids"] == "WKD"
    assert weekday["trip_count"] == 6

    assert (tmp_path / "calendar_service_summary.csv").exists()
    assert (tmp_path / "calendar_day_type_summary.csv").exists()
    runlog = tmp_path / "gtfs_calendar_inspector_runlog.txt"
    assert runlog.exists()
    assert "# === BEGIN CONFIG ===" in runlog.read_text(encoding="utf-8")


def test_run_split_weekday_day_type_summary(tmp_path: Path) -> None:
    service, day_types = target.run(gtfs_path=FIXTURES / "gtfs_split_weekday", output_dir=tmp_path)
    assert _summary_row(service, "MON")["labels"] == "Weekday"
    assert _summary_row(service, "TWR")["labels"] == "Weekday"
    assert _summary_row(service, "FRI")["labels"] == "Weekday"

    weekday = _day_type_row(day_types, "weekday")
    assert weekday["service_ids"] == "TWR"  # modal midweek pattern, no M/F union
    assert weekday["trip_count"] == 6
    assert _day_type_row(day_types, "saturday")["service_ids"] == "SAT"
    assert _day_type_row(day_types, "sunday")["service_ids"] == "SUN"


def test_run_labels_rare_non_holiday_service_special(tmp_path: Path) -> None:
    # WMATA service_id 3 runs on five Fridays, none a federal holiday: the feed shows
    # they are exception dates but not why, so the label is Special, not Holiday.
    feed = tmp_path / "feed"
    feed.mkdir()
    (feed / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "WKD,1,1,1,1,1,0,0,20260105,20260630\n",
        encoding="utf-8",
    )
    fridays = ["20260109", "20260206", "20260306", "20260403", "20260501"]
    (feed / "calendar_dates.txt").write_text(
        "service_id,date,exception_type\n" + "".join(f"3,{d},1\n" for d in fridays),
        encoding="utf-8",
    )
    service, _ = target.run(gtfs_path=feed, output_dir=tmp_path / "out")
    special = _summary_row(service, "3")
    assert special["labels"] == "Special"
    assert special["active_days"] == 5
    assert special["holiday_days"] == 0
    assert _summary_row(service, "WKD")["labels"] == "Weekday"


def test_run_rejects_feed_without_any_calendar(tmp_path: Path) -> None:
    feed = tmp_path / "feed"
    feed.mkdir()
    (feed / "trips.txt").write_text("route_id,service_id,trip_id\nR1,A,T1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="neither"):
        target.run(gtfs_path=feed, output_dir=tmp_path / "out")


def test_run_works_without_trips(tmp_path: Path) -> None:
    feed = tmp_path / "feed"
    feed.mkdir()
    src = FIXTURES / "gtfs_split_weekday"
    for name in ("calendar.txt", "calendar_dates.txt"):
        (feed / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    service, day_types = target.run(gtfs_path=feed, output_dir=tmp_path / "out")
    assert _summary_row(service, "TWR")["trip_count"] == 0
    assert _day_type_row(day_types, "weekday")["service_ids"] == "TWR"


_WEEK_CALENDAR = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "WK,1,1,1,1,1,0,0,20260105,20260109\n"
)
_ONE_CANCELLATION = "service_id,date,exception_type\nWK,20260107,2\n"


def test_run_stops_when_calendar_dates_is_damaged(tmp_path: Path) -> None:
    feed = tmp_path / "feed"
    feed.mkdir()
    (feed / "calendar.txt").write_text(_WEEK_CALENDAR, encoding="utf-8")
    (feed / "calendar_dates.txt").write_text(_ONE_CANCELLATION, encoding="utf-8")
    service, _ = target.run(gtfs_path=feed, output_dir=tmp_path / "ok")
    assert _summary_row(service, "WK")["active_days"] == 4

    # One malformed record must not discard the valid cancellation above it.
    with (feed / "calendar_dates.txt").open("a", encoding="utf-8") as handle:
        handle.write("WK,20260108,2,extra,fields\n")
    with pytest.raises(ValueError, match="Could not read calendar_dates.txt"):
        target.run(gtfs_path=feed, output_dir=tmp_path / "damaged")
    assert not (tmp_path / "damaged" / "calendar_service_summary.csv").exists()
    rc = target.main(["--gtfs", str(feed), "--output-dir", str(tmp_path / "damaged")])
    assert rc == 1


def test_run_accepts_zipped_feed_without_calendar_dates(tmp_path: Path) -> None:
    zip_path = tmp_path / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("wrapper/calendar.txt", _WEEK_CALENDAR)
    service, _ = target.run(gtfs_path=zip_path, output_dir=tmp_path / "out")
    assert _summary_row(service, "WK")["active_days"] == 5


def test_main_placeholder_paths_return_2() -> None:
    assert target.main([]) == 2


def test_main_runs_end_to_end(tmp_path: Path) -> None:
    rc = target.main(
        ["--gtfs", str(FIXTURES / "gtfs_holiday_negation"), "--output-dir", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "calendar_service_summary.csv").exists()
