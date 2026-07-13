"""Tests for the GTFS service-expansion helpers in utils/gtfs_helpers.py."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.gtfs_helpers import classify_day_type, expand_service_dates, find_exception_dates


def _calendar() -> pd.DataFrame:
    """A weekday service and a Saturday service, both valid through January 2025."""
    return pd.DataFrame(
        {
            "service_id": ["WKDY", "SAT"],
            "monday": ["1", "0"],
            "tuesday": ["1", "0"],
            "wednesday": ["1", "0"],
            "thursday": ["1", "0"],
            "friday": ["1", "0"],
            "saturday": ["0", "1"],
            "sunday": ["0", "0"],
            "start_date": ["20250101", "20250101"],
            "end_date": ["20250131", "20250131"],
        }
    )


def _calendar_dates() -> pd.DataFrame:
    """MLK Monday 2025-01-20: weekday service removed, Saturday service added."""
    return pd.DataFrame(
        {
            "service_id": ["WKDY", "SAT"],
            "date": ["20250120", "20250120"],
            "exception_type": ["2", "1"],
        }
    )


def test_expand_weekly_pattern_counts_days() -> None:
    """A Mon-Fri service expands to exactly the weekdays inside the window."""
    out = expand_service_dates(
        _calendar(), None, pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-12")
    )
    wkdy = out.loc[out["service_id"] == "WKDY", "service_date"]
    assert len(wkdy) == 5
    assert wkdy.min() == pd.Timestamp("2025-01-06")
    assert wkdy.max() == pd.Timestamp("2025-01-10")
    sat = out.loc[out["service_id"] == "SAT", "service_date"]
    assert list(sat) == [pd.Timestamp("2025-01-11")]


def test_expand_respects_calendar_validity_range() -> None:
    """Dates outside a service's start/end range are not activated."""
    cal = _calendar()
    cal["end_date"] = ["20250107", "20250131"]
    out = expand_service_dates(cal, None, pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-12"))
    wkdy = out.loc[out["service_id"] == "WKDY", "service_date"]
    assert list(wkdy) == [pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-07")]


def test_expand_applies_exceptions_both_ways() -> None:
    """Type 2 removes a weekly activation; type 1 adds one the pattern lacks."""
    out = expand_service_dates(
        _calendar(),
        _calendar_dates(),
        pd.Timestamp("2025-01-20"),
        pd.Timestamp("2025-01-20"),
    )
    assert list(out["service_id"]) == ["SAT"]


def test_expand_calendar_dates_only_feed() -> None:
    """Feeds with no calendar.txt expand purely from type-1 exceptions."""
    cd = pd.DataFrame(
        {
            "service_id": ["S1", "S1", "S1"],
            "date": ["20250106", "20250107", "20250301"],
            "exception_type": ["1", "1", "1"],
        }
    )
    out = expand_service_dates(None, cd, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31"))
    assert len(out) == 2  # the March date falls outside the window
    assert set(out["service_id"]) == {"S1"}


def test_expand_no_calendars_raises() -> None:
    """Both inputs missing/empty is an actionable error."""
    with pytest.raises(ValueError, match="cannot expand"):
        expand_service_dates(
            None, pd.DataFrame(), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")
        )


def test_expand_inverted_window_raises() -> None:
    """start_date after end_date is an actionable error."""
    with pytest.raises(ValueError, match="after end_date"):
        expand_service_dates(
            _calendar(), None, pd.Timestamp("2025-02-01"), pd.Timestamp("2025-01-01")
        )


def test_expand_missing_calendar_column_raises() -> None:
    """A calendar.txt without required columns is an actionable error."""
    with pytest.raises(ValueError, match="missing required column"):
        expand_service_dates(
            _calendar().drop(columns=["monday"]),
            None,
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-31"),
        )


def test_classify_day_type_labels() -> None:
    """Friday/Saturday/Sunday map to Weekday/Saturday/Sunday."""
    dates = pd.Series(pd.to_datetime(["2025-01-17", "2025-01-18", "2025-01-19"]))
    assert list(classify_day_type(dates)) == ["Weekday", "Saturday", "Sunday"]


def test_find_exception_dates_windowed() -> None:
    """Both exception types count, and window bounds filter the result."""
    dates = find_exception_dates(_calendar_dates())
    assert dates == {pd.Timestamp("2025-01-20")}
    assert find_exception_dates(_calendar_dates(), end_date=pd.Timestamp("2025-01-10")) == set()
    assert find_exception_dates(None) == set()
