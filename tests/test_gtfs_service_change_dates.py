from __future__ import annotations

import ast
import datetime as dt
import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

import scripts.field_tools.gtfs_service_change_dates as target

_CAL_HEADER = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date"
)


def _write_feed(
    parent: Path,
    name: str,
    calendar_rows=(),
    calendar_dates_rows=(),
    agency_name: str = "Metro Transit",
    timezone: str = "America/New_York",
    feed_info=None,
    extra_files=None,
    zipped: bool = False,
) -> Path:
    feed_dir = parent / name
    feed_dir.mkdir(parents=True)
    if calendar_rows:
        lines = [_CAL_HEADER]
        for sid, days, start, end in calendar_rows:
            lines.append(f"{sid},{','.join(days)},{start},{end}")
        (feed_dir / "calendar.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if calendar_dates_rows:
        lines = ["service_id,date,exception_type"]
        for sid, date, etype in calendar_dates_rows:
            lines.append(f"{sid},{date},{etype}")
        (feed_dir / "calendar_dates.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if agency_name:
        (feed_dir / "agency.txt").write_text(
            "agency_id,agency_name,agency_url,agency_timezone\n"
            f"A1,{agency_name},https://example.com,{timezone}\n",
            encoding="utf-8",
        )
    if feed_info:
        header = ",".join(feed_info)
        values = ",".join(str(v) for v in feed_info.values())
        (feed_dir / "feed_info.txt").write_text(f"{header}\n{values}\n", encoding="utf-8")
    for file_name, content in (extra_files or {}).items():
        (feed_dir / file_name).write_text(content, encoding="utf-8")
    if zipped:
        zip_path = parent / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for member in sorted(feed_dir.iterdir()):
                archive.write(member, arcname=f"{name}/{member.name}")
        shutil.rmtree(feed_dir)
        return zip_path
    return feed_dir


def test_within_feed_change_detected(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "single",
        calendar_rows=[
            ("WKD1", "1111100", "20250106", "20250613"),
            ("WKD2", "1111100", "20250616", "20251226"),
            ("SAT", "0000010", "20250111", "20251227"),
        ],
    )
    out = tmp_path / "out"
    changes, feeds_df = target.run(feeds_dir=feeds, output_dir=out)

    assert list(changes["change_date"]) == ["2025-06-16"]
    row = changes.iloc[0]
    assert row["day_of_week"] == "Monday"
    assert row["change_type"] == "Service change"
    assert row["services_added"] == "WKD2"
    assert row["services_removed"] == "WKD1"

    assert len(feeds_df) == 1
    assert feeds_df.iloc[0]["first_active_date"] == "2025-01-06"
    assert feeds_df.iloc[0]["last_active_date"] == "2025-12-27"
    assert feeds_df.iloc[0]["service_ids"] == 3

    assert (out / "service_change_quick_reference.xlsx").exists()
    runlog = out / "gtfs_service_change_dates_runlog.txt"
    assert runlog.exists()
    assert "# === BEGIN CONFIG ===" in runlog.read_text(encoding="utf-8")


def test_holiday_week_is_transient(tmp_path: Path, caplog) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "holiday",
        calendar_rows=[("WKD", "1111100", "20250106", "20251226")],
        calendar_dates_rows=[("WKD", "20250704", "2"), ("HOL", "20250704", "1")],
    )
    with caplog.at_level(logging.INFO):
        changes, _ = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert changes.empty
    assert "transient" in caplog.text.lower()


def test_cross_feed_succession_between_zips(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "2025_spring",
        calendar_rows=[("WKD_A", "1111100", "20250106", "20250613")],
        zipped=True,
    )
    _write_feed(
        feeds,
        "2025_fall",
        calendar_rows=[("WKD_B", "1111100", "20250616", "20251226")],
        zipped=True,
    )
    changes, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")

    assert list(changes["change_date"]) == ["2025-06-16"]
    row = changes.iloc[0]
    assert row["change_type"] == "New service period"
    assert row["services_added"] == "WKD_B"
    assert row["services_removed"] == "WKD_A"
    assert "2025_spring" in row["source_feeds"]
    assert "2025_fall" in row["source_feeds"]
    assert len(feeds_df) == 2


def test_overlapping_feeds_disagree(tmp_path: Path, caplog) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "archive_flat",
        calendar_rows=[("WKD_ALL", "1111100", "20250106", "20251226")],
    )
    _write_feed(
        feeds,
        "archive_split",
        calendar_rows=[
            ("WKD1", "1111100", "20250106", "20250613"),
            ("WKD2", "1111100", "20250616", "20251226"),
        ],
    )
    with caplog.at_level(logging.WARNING):
        changes, _ = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")

    assert list(changes["change_date"]) == ["2025-06-16"]
    assert changes.iloc[0]["disputed_by"] == "archive_flat"
    assert "disagree" in caplog.text.lower()


def test_mixed_agencies_warn_about_same_system(tmp_path: Path, caplog) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "metro",
        calendar_rows=[("WKD", "1111100", "20250106", "20250613")],
        agency_name="Metro Transit",
    )
    _write_feed(
        feeds,
        "sunshine",
        calendar_rows=[("WKD", "1111100", "20250616", "20251226")],
        agency_name="Sunshine Shuttles",
        timezone="America/Chicago",
    )
    with caplog.at_level(logging.WARNING):
        _, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert "same system" in caplog.text.lower()
    assert "agency_timezones" in caplog.text
    assert feeds_df["issues"].str.contains("same system").any()


def test_cutoffs_limit_changes(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "quarterly",
        calendar_rows=[
            ("W1", "1111100", "20250106", "20250328"),
            ("W2", "1111100", "20250331", "20250627"),
            ("W3", "1111100", "20250630", "20250926"),
            ("W4", "1111100", "20250929", "20251226"),
        ],
    )
    out = tmp_path / "out"

    all_changes, _ = target.run(feeds_dir=feeds, output_dir=out)
    assert list(all_changes["change_date"]) == ["2025-03-31", "2025-06-30", "2025-09-29"]

    recent, _ = target.run(feeds_dir=feeds, output_dir=out, max_changes=1)
    assert list(recent["change_date"]) == ["2025-09-29"]

    # Half a year back from the newest active date (2025-12-26) is 2025-06-26.
    within, _ = target.run(feeds_dir=feeds, output_dir=out, max_years=0.5)
    assert list(within["change_date"]) == ["2025-06-30", "2025-09-29"]


def test_folder_that_is_itself_a_feed(tmp_path: Path) -> None:
    feed = _write_feed(
        tmp_path, "rootfeed", calendar_rows=[("WKD", "1111100", "20250106", "20251226")]
    )
    changes, feeds_df = target.run(feeds_dir=feed, output_dir=tmp_path / "out")
    assert changes.empty
    assert len(feeds_df) == 1
    assert feeds_df.iloc[0]["feed"] == "rootfeed"


def test_unusable_feed_is_listed_with_issue(tmp_path: Path, caplog) -> None:
    feeds = tmp_path / "archive"
    _write_feed(feeds, "good", calendar_rows=[("WKD", "1111100", "20250106", "20251226")])
    _write_feed(
        feeds,
        "no_dates",
        extra_files={"trips.txt": "route_id,service_id,trip_id\nR1,A,T1\n"},
    )
    with caplog.at_level(logging.WARNING):
        _, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert len(feeds_df) == 2
    bad = feeds_df[feeds_df["feed"] == "no_dates"].iloc[0]
    assert "calendar" in bad["issues"]
    assert bad["first_active_date"] == ""
    assert "no_dates" in caplog.text


def test_feed_info_disagreement_warns(tmp_path: Path, caplog) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "short_calendar",
        calendar_rows=[("WKD", "1111100", "20250106", "20250613")],
        feed_info={
            "feed_publisher_name": "Metro Transit",
            "feed_publisher_url": "https://example.com",
            "feed_lang": "en",
            "feed_start_date": "20250101",
            "feed_end_date": "20251231",
            "feed_version": "2025.1",
        },
    )
    with caplog.at_level(logging.WARNING):
        _, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    row = feeds_df.iloc[0]
    assert row["feed_version"] == "2025.1"
    assert row["declared_end"] == "2025-12-31"
    assert "feed_info declares service through" in row["issues"]
    assert "feed_info declares service through" in caplog.text


def test_csv_outputs_are_machine_readable(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    calendar_rows = [("WKD0", "1111100", "20250106", "20250613")] + [
        (f"W_{letter}", "1111100", "20250616", "20251226") for letter in "ABCDEFG"
    ]
    _write_feed(feeds, "many_services", calendar_rows=calendar_rows)
    out = tmp_path / "out"
    changes, feeds_df = target.run(feeds_dir=feeds, output_dir=out)

    # The CSV carries the full, untruncated service_id list.
    csv_changes = pd.read_csv(out / "service_changes.csv")
    assert list(csv_changes["change_date"]) == ["2025-06-16"]
    assert csv_changes.iloc[0]["services_added"] == "W_A; W_B; W_C; W_D; W_E; W_F; W_G"
    assert csv_changes.iloc[0]["services_removed"] == "WKD0"
    assert list(csv_changes.columns) == list(changes.columns)

    # The printable XLSX shortens the same list for display.
    workbook = load_workbook(out / "service_change_quick_reference.xlsx")
    assert "+1 more" in str(workbook["Service Changes"]["D4"].value)

    csv_feeds = pd.read_csv(out / "service_change_feeds.csv")
    assert list(csv_feeds["feed"]) == list(feeds_df["feed"])
    assert csv_feeds.iloc[0]["service_ids"] == 8


def test_empty_folder_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No GTFS feeds"):
        target.run(feeds_dir=empty, output_dir=tmp_path / "out")


def test_main_placeholder_paths_return_2() -> None:
    assert target.main([]) == 2


def test_main_missing_folder_returns_1(tmp_path: Path) -> None:
    rc = target.main(["--feeds-dir", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")])
    assert rc == 1


def test_main_runs_end_to_end(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "single",
        calendar_rows=[
            ("WKD1", "1111100", "20250106", "20250613"),
            ("WKD2", "1111100", "20250616", "20251226"),
        ],
    )
    out = tmp_path / "out"
    rc = target.main(["--feeds-dir", str(feeds), "--output-dir", str(out), "--max-changes", "5"])
    assert rc == 0

    workbook = load_workbook(out / "service_change_quick_reference.xlsx")
    assert workbook.sheetnames == ["Service Changes", "Feeds"]
    sheet = workbook["Service Changes"]
    assert str(sheet["A1"].value).startswith("Service Change Quick Reference")
    assert sheet["A4"].value == "2025-06-16"


def test_boundary_dated_where_new_feeds_stable_pattern_begins(tmp_path: Path) -> None:
    # The newer feed opens with one leftover week of A before B takes over.
    feeds = tmp_path / "archive"
    _write_feed(feeds, "old", calendar_rows=[("A", "1111100", "20251201", "20260116")])
    _write_feed(
        feeds,
        "new",
        calendar_rows=[
            ("A", "1111100", "20260119", "20260123"),
            ("B", "1111100", "20260126", "20260327"),
        ],
    )
    changes, _ = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert list(changes["change_date"]) == ["2026-01-26"]
    assert changes.iloc[0]["services_added"] == "B"
    assert changes.iloc[0]["services_removed"] == "A"


def test_isolated_holiday_does_not_pull_change_date_backward(tmp_path: Path) -> None:
    # B runs on the Jan 19 holiday only, A resumes, then B takes over Feb 2.
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "single",
        calendar_rows=[
            ("A", "1111100", "20251201", "20260130"),
            ("B", "1111100", "20260202", "20260327"),
        ],
        calendar_dates_rows=[("A", "20260119", "2"), ("B", "20260119", "1")],
    )
    changes, _ = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert list(changes["change_date"]) == ["2026-02-02"]


def test_mid_week_change_with_holiday_in_transition(tmp_path: Path) -> None:
    # B starts Wednesday Jan 28; Thursday Jan 29 is a holiday running neither pattern.
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "single",
        calendar_rows=[
            ("A", "1111100", "20251201", "20260127"),
            ("B", "1111100", "20260128", "20260327"),
        ],
        calendar_dates_rows=[("B", "20260129", "2"), ("HOL", "20260129", "1")],
    )
    changes, _ = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert list(changes["change_date"]) == ["2026-01-28"]


def test_coverage_gap_uses_whole_archive(tmp_path: Path, caplog) -> None:
    # B ends in February, but A covers March, so C's April start is no gap.
    feeds = tmp_path / "archive"
    _write_feed(feeds, "a_year", calendar_rows=[("W", "1111100", "20260105", "20261231")])
    _write_feed(feeds, "b_feb", calendar_rows=[("W", "1111100", "20260202", "20260227")])
    _write_feed(feeds, "c_q2", calendar_rows=[("W", "1111100", "20260406", "20260626")])
    with caplog.at_level(logging.WARNING):
        changes, _ = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert changes.empty
    assert "coverage gap" not in caplog.text.lower()


def test_real_coverage_gap_still_reported(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(feeds, "spring", calendar_rows=[("W", "1111100", "20260105", "20260227")])
    _write_feed(feeds, "summer", calendar_rows=[("W", "1111100", "20260406", "20260626")])
    changes, _ = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert list(changes["change_date"]) == ["2026-04-06"]
    assert changes.iloc[0]["change_type"] == "New service period"
    assert "37-day gap" in changes.iloc[0]["notes"]


def test_folder_and_zip_with_same_name_get_distinct_labels(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds, "2026", calendar_rows=[("B", "1111100", "20260202", "20260327")], zipped=True
    )
    _write_feed(feeds, "2026", calendar_rows=[("A", "1111100", "20260105", "20260130")])
    changes, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert sorted(feeds_df["feed"]) == ["2026", "2026.zip"]
    assert list(changes["change_date"]) == ["2026-02-02"]
    assert changes.iloc[0]["source_feeds"] == "2026; 2026.zip"


def test_duplicate_labels_rejected_by_merge(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    first = _write_feed(feeds, "x", calendar_rows=[("A", "1111100", "20260105", "20260130")])
    second = _write_feed(feeds, "y", calendar_rows=[("B", "1111100", "20260202", "20260327")])
    summaries = [target.inspect_feed(first, "same"), target.inspect_feed(second, "same")]
    with pytest.raises(ValueError, match="unique label"):
        target.merge_service_changes(summaries)


def test_literal_na_service_ids_survive_and_blank_ids_are_flagged(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "nas",
        calendar_rows=[
            ("NA", "1111100", "20260105", "20260130"),
            ("NULL", "1111100", "20260202", "20260327"),
            ("", "0000010", "20260105", "20260327"),
        ],
    )
    changes, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert list(changes["change_date"]) == ["2026-02-02"]
    assert changes.iloc[0]["services_added"] == "NULL"
    assert changes.iloc[0]["services_removed"] == "NA"
    assert feeds_df.iloc[0]["service_ids"] == 2
    assert "blank service_id" in feeds_df.iloc[0]["issues"]


def _event(day: int, feed: str, added: str, removed: str) -> target.ServiceChange:
    return target.ServiceChange(
        dt.date(2026, 1, day), (feed,), "Service change", frozenset({added}), frozenset({removed})
    )


def test_cluster_does_not_chain_successive_changes() -> None:
    events = [
        _event(1, "f1", "B", "A"),
        _event(4, "f2", "C", "B"),
        _event(7, "f3", "D", "C"),
        _event(10, "f4", "E", "D"),
    ]
    first_active = {f"f{i}": dt.date(2025, i, 1) for i in range(1, 5)}
    merged = target._cluster_events(events, first_active)
    assert [(c.date.day, set(c.added), set(c.removed)) for c in merged] == [
        (1, {"B"}, {"A"}),
        (4, {"C"}, {"B"}),
        (7, {"D"}, {"C"}),
        (10, {"E"}, {"D"}),
    ]


def test_cluster_span_is_bounded_from_its_first_event() -> None:
    # Same change reported 1, 4 and 7 Jan: 7 Jan is 6 days from the cluster start.
    events = [_event(1, "f1", "B", "A"), _event(4, "f2", "B", "A"), _event(7, "f3", "B", "A")]
    first_active = {f"f{i}": dt.date(2025, i, 1) for i in range(1, 4)}
    merged = target._cluster_events(events, first_active)
    assert [c.feeds for c in merged] == [("f1", "f2"), ("f3",)]
    assert merged[0].date == dt.date(2026, 1, 4)


def test_run_log_written_without_source_file(tmp_path: Path, monkeypatch) -> None:
    # A notebook cell has no __file__; the run log falls back to runtime values.
    monkeypatch.delitem(target.__dict__, "__file__")
    feeds = tmp_path / "archive"
    _write_feed(feeds, "single", calendar_rows=[("A", "1111100", "20260105", "20260327")])
    out = tmp_path / "out"
    target.run(feeds_dir=feeds, output_dir=out)
    text = (out / "gtfs_service_change_dates_runlog.txt").read_text(encoding="utf-8")
    assert "runtime values" in text
    assert "MIN_STABLE_WEEKS = 2" in text


def test_config_names_match_config_block() -> None:
    block = target.extract_config_block(Path(target.__file__))
    names = {
        node.target.id
        for node in ast.parse(block).body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert names == set(target._CONFIG_NAMES)


def test_historical_long_calendar_is_not_anchored_to_today(tmp_path: Path) -> None:
    feeds = tmp_path / "archive"
    _write_feed(feeds, "hist", calendar_rows=[("H", "1111100", "20100104", "20161230")])
    _, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    row = feeds_df.iloc[0]
    assert row["last_active_date"] == "2016-12-30"
    assert row["active_days"] > 1000


def test_long_calendar_anchored_on_feeds_ordinary_rows(tmp_path: Path) -> None:
    # A 2000-2099 placeholder row is clamped around the feed's 2012 rows, not today.
    feeds = tmp_path / "archive"
    _write_feed(
        feeds,
        "old_feed",
        calendar_rows=[
            ("WKD", "1111100", "20120102", "20121228"),
            ("SUN", "0000001", "20000102", "20991227"),
        ],
    )
    _, feeds_df = target.run(feeds_dir=feeds, output_dir=tmp_path / "out")
    assert feeds_df.iloc[0]["last_active_date"] < "2016-01-01"


def test_range_outside_clamp_window_warns_explicitly(caplog) -> None:
    calendar = pd.DataFrame(
        [
            {
                "service_id": "H",
                **{day: "1" for day in ("monday", "tuesday", "wednesday", "thursday")},
                "friday": "1",
                "saturday": "0",
                "sunday": "0",
                "start_date": "20100104",
                "end_date": "20161230",
            }
        ]
    )
    with caplog.at_level(logging.WARNING):
        active = target.expand_service_active_dates(calendar, today=dt.date(2026, 10, 6))
    assert active == {"H": set()}
    assert "entirely outside the expansion window" in caplog.text
