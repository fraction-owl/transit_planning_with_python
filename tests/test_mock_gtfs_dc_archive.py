"""Tests for tests/fixtures/mock_gtfs_dc_archive.zip, three dated copies of the mock DC feed.

The archive is written by dev_tools/generate_mock_gtfs.py (ARCHIVE_EDITIONS):

- dc_2024 -> dc_2025: service change only. Sunday service starts on routes
  10, 30 and 40, and route 30 goes from every 20 to every 15 minutes on
  weekdays. Fares are unchanged.
- dc_2025 -> dc_2026: fare change only. The LOCAL fare goes from $1.75 to
  $2.00, a FREE fare is added and route 50H moves to it; EXPRESS is unchanged.
  Service is unchanged.
- dc_2026 is the same feed as tests/fixtures/mock_gtfs_dc.zip.

These tests pin that contract and run the multi-feed scripts over it.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import scripts.field_tools.gtfs_service_change_dates as service_change_dates
from scripts.stop_analysis.gtfs_route_diff import compare_routes, load_feed
from utils.gtfs_helpers import load_gtfs_data

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE_ZIP = FIXTURES / "mock_gtfs_dc_archive.zip"
MOCK_ZIP = FIXTURES / "mock_gtfs_dc.zip"
FEEDS = ("dc_2024", "dc_2025", "dc_2026")

# Threshold knobs matching gtfs_route_diff.py's CONFIG defaults.
KNOBS = {
    "rekey_jaccard": 0.50,
    "align_minor_jaccard": 0.80,
    "trips_major": 0.30,
    "span_major": 60.0,
    "headway_major": 0.50,
    "fare_price_major": 0.25,
    "split_merge_enable": 1.0,
    "split_merge_min_coverage": 0.60,
    "split_merge_min_share": 0.30,
}


@pytest.fixture(scope="module")
def archive_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract the archive once; each feed is a sub-folder (dc_2024/, ...)."""
    dest = tmp_path_factory.mktemp("mock_gtfs_dc_archive")
    with zipfile.ZipFile(ARCHIVE_ZIP) as zf:
        zf.extractall(dest)
    return dest


def _diff(archive_dir: Path, before: str, after: str) -> dict:
    return compare_routes(
        load_feed(archive_dir / before, label=before),
        load_feed(archive_dir / after, label=after),
        weekday="tuesday",
        knobs=KNOBS,
    )


def test_archive_holds_three_contiguous_yearly_feeds(archive_dir: Path) -> None:
    assert sorted(p.name for p in archive_dir.iterdir()) == list(FEEDS)

    windows = []
    for feed in FEEDS:
        info = load_gtfs_data(str(archive_dir / feed), files=("feed_info.txt",))["feed_info"]
        start = dt.datetime.strptime(info.loc[0, "feed_start_date"], "%Y%m%d").date()
        end = dt.datetime.strptime(info.loc[0, "feed_end_date"], "%Y%m%d").date()
        assert (start, end) == (dt.date(start.year, 1, 1), dt.date(start.year, 12, 31))
        assert info.loc[0, "feed_version"].startswith(str(start.year))
        windows.append((start, end))
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == prev_end + dt.timedelta(days=1)


def test_latest_archive_feed_matches_mock_gtfs_dc(archive_dir: Path) -> None:
    with zipfile.ZipFile(MOCK_ZIP) as zf:
        mock = {Path(name).name: zf.read(name) for name in zf.namelist() if name.endswith(".txt")}
    latest = {p.name: p.read_bytes() for p in (archive_dir / "dc_2026").glob("*.txt")}
    assert latest == mock


def test_2024_to_2025_is_a_service_change_only(archive_dir: Path) -> None:
    results = _diff(archive_dir, "dc_2024", "dc_2025")

    overview = results["overview"].set_index("route_id_after")
    assert set(overview.loc[["DC_R10", "DC_R30", "DC_R40"], "daytype_change"]) == {"+sunday"}
    assert overview.loc["DC_R30", "schedule_change"] == "major"
    assert "headway 20->15 min" in overview.loc["DC_R30", "change_summary"]
    assert set(overview.loc[["DC_R20", "DC_R50H"], "headline_change"]) == {"unchanged"}

    calendar = results["calendar_changes"].set_index("service_id")
    assert calendar.loc["sunday", "status"] == "added"
    assert results["fare_attribute_changes"].empty
    assert results["route_fare_changes"].empty


def test_2025_to_2026_is_a_fare_change_only(archive_dir: Path) -> None:
    results = _diff(archive_dir, "dc_2025", "dc_2026")

    summary = results["summary"]
    assert summary.schedule_major_count == summary.schedule_minor_count == 0
    assert summary.alignment_major_count == summary.alignment_minor_count == 0
    assert summary.daytype_changed_count == 0

    fares = results["fare_attribute_changes"].set_index("fare_id")
    assert set(fares.index) == {"LOCAL", "FREE"}  # EXPRESS is unchanged
    assert fares.loc["LOCAL", "status"] == "changed"
    assert fares.loc["LOCAL", "price_delta"] == pytest.approx(0.25)
    assert fares.loc["LOCAL", "severity"] == "major"
    assert fares.loc["FREE", "status"] == "added"

    assert list(results["route_fare_changes"]["route_id_after"]) == ["DC_R50H"]


def test_service_change_dates_finds_the_service_change_but_not_the_fare_change(
    archive_dir: Path, tmp_path: Path
) -> None:
    """The calendar-based finder sees the Sunday service start, not the fare-only feed."""
    changes, feeds = service_change_dates.run(feeds_dir=archive_dir, output_dir=tmp_path)

    assert list(feeds["feed"]) == list(FEEDS)
    assert list(changes["change_date"]) == ["2025-01-05"]  # first Sunday with service
    row = changes.iloc[0]
    assert row["services_added"] == "sunday"
    assert pd.isna(row["services_removed"]) or row["services_removed"] == ""
