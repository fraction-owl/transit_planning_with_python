from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import pytest

from utils.calendar_helpers import (
    classify_service_ids,
    expand_service_active_dates,
    representative_service_date,
    service_ids_active_on,
)

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = FIXTURES / name
    calendar = pd.read_csv(folder / "calendar.txt", dtype=str)
    calendar_dates = pd.read_csv(folder / "calendar_dates.txt", dtype=str)
    return calendar, calendar_dates


def _simple_calendar() -> pd.DataFrame:
    """One weekday service and one Saturday service over two weeks."""
    return pd.DataFrame(
        {
            "service_id": ["WK", "SAT"],
            "monday": ["1", "0"],
            "tuesday": ["1", "0"],
            "wednesday": ["1", "0"],
            "thursday": ["1", "0"],
            "friday": ["1", "0"],
            "saturday": ["0", "1"],
            "sunday": ["0", "0"],
            "start_date": ["20260601", "20260601"],
            "end_date": ["20260614", "20260614"],
        }
    )


# ---------------------------------------------------------------------------
# expand_service_active_dates
# ---------------------------------------------------------------------------


def test_expand_basic_pattern() -> None:
    active = expand_service_active_dates(_simple_calendar())
    assert len(active["WK"]) == 10  # two full Mon-Fri weeks
    assert len(active["SAT"]) == 2
    assert dt.date(2026, 6, 1) in active["WK"]  # a Monday
    assert dt.date(2026, 6, 6) not in active["WK"]  # a Saturday
    assert dt.date(2026, 6, 6) in active["SAT"]


def test_expand_applies_exceptions_and_redundant_add() -> None:
    calendar_dates = pd.DataFrame(
        {
            "service_id": ["WK", "WK", "SAT"],
            "date": ["20260601", "20260607", "20260606"],
            "exception_type": ["2", "1", "1"],  # SAT add is redundant
        }
    )
    active = expand_service_active_dates(_simple_calendar(), calendar_dates)
    assert dt.date(2026, 6, 1) not in active["WK"]  # removed
    assert dt.date(2026, 6, 7) in active["WK"]  # added (a Sunday)
    assert len(active["WK"]) == 10  # net unchanged: one out, one in
    assert len(active["SAT"]) == 2  # redundant addition is a no-op


def test_expand_calendar_dates_only_feed() -> None:
    calendar_dates = pd.DataFrame(
        {
            "service_id": ["A", "A", "B"],
            "date": ["20260601", "20260602", "20260606"],
            "exception_type": ["1", "1", "1"],
        }
    )
    active = expand_service_active_dates(None, calendar_dates)
    assert active["A"] == {dt.date(2026, 6, 1), dt.date(2026, 6, 2)}
    assert active["B"] == {dt.date(2026, 6, 6)}


def test_expand_skips_unparseable_and_reversed_dates(caplog: pytest.LogCaptureFixture) -> None:
    calendar = _simple_calendar()
    calendar.loc[0, "start_date"] = "not-a-date"
    calendar.loc[1, ["start_date", "end_date"]] = ["20261231", "20260101"]
    with caplog.at_level(logging.WARNING):
        active = expand_service_active_dates(calendar)
    assert active["WK"] == set()
    assert active["SAT"] == set()
    assert "unparseable" in caplog.text
    assert "precedes" in caplog.text


def test_expand_clamps_placeholder_ranges(caplog: pytest.LogCaptureFixture) -> None:
    calendar = _simple_calendar().iloc[[0]].copy()
    calendar.loc[0, ["start_date", "end_date"]] = ["20000101", "20991231"]
    anchor = dt.date(2026, 7, 15)
    with caplog.at_level(logging.WARNING):
        active = expand_service_active_dates(calendar, max_days_per_service=30, today=anchor)
    assert "placeholder" in caplog.text
    assert active["WK"]
    assert all(abs((d - anchor).days) <= 15 for d in active["WK"])


def test_expand_missing_columns_raises() -> None:
    calendar = _simple_calendar().drop(columns=["start_date"])
    with pytest.raises(ValueError, match="start_date"):
        expand_service_active_dates(calendar)


# ---------------------------------------------------------------------------
# service_ids_active_on / classify_service_ids
# ---------------------------------------------------------------------------


def test_service_ids_active_on() -> None:
    active = expand_service_active_dates(_simple_calendar())
    assert service_ids_active_on(active, dt.date(2026, 6, 2)) == {"WK"}
    assert service_ids_active_on(active, dt.date(2026, 6, 6)) == {"SAT"}
    assert service_ids_active_on(active, dt.date(2026, 6, 7)) == set()


def test_classify_holiday_negation_fixture() -> None:
    calendar, calendar_dates = _load_fixture("gtfs_holiday_negation")
    active = expand_service_active_dates(calendar, calendar_dates)
    labels = classify_service_ids(active)
    # HOL's calendar.txt row claims Mon/Wed/Fri, but every one of those dates
    # is negated except five holidays — the classifier must see through it.
    assert labels["HOL"] == {"Holiday"}
    assert len(active["HOL"]) == 5
    assert labels["WKD"] == {"Weekday"}
    assert labels["SAT"] == {"Saturday"}
    assert labels["SUN"] == {"Sunday"}  # despite 4 weekday-holiday additions


def test_classify_split_weekday_fixture() -> None:
    calendar, calendar_dates = _load_fixture("gtfs_split_weekday")
    active = expand_service_active_dates(calendar, calendar_dates)
    labels = classify_service_ids(active)
    assert labels["MON"] == {"Weekday"}
    assert labels["TWR"] == {"Weekday"}
    assert labels["FRI"] == {"Weekday"}
    assert labels["SAT"] == {"Saturday"}


def test_classify_empty_service() -> None:
    assert classify_service_ids({"GHOST": set()}) == {"GHOST": set()}


# ---------------------------------------------------------------------------
# representative_service_date
# ---------------------------------------------------------------------------


def test_representative_weekday_on_split_fixture_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calendar, calendar_dates = _load_fixture("gtfs_split_weekday")
    active = expand_service_active_dates(calendar, calendar_dates)
    with caplog.at_level(logging.WARNING):
        chosen, ids = representative_service_date(active, "weekday")
    # Tue-Thu is the modal weekday pattern; Monday and Friday differ.
    assert ids == {"TWR"}
    assert chosen.weekday() in (1, 2, 3)
    assert "varies by day of week" in caplog.text


def test_representative_single_days_on_split_fixture() -> None:
    calendar, calendar_dates = _load_fixture("gtfs_split_weekday")
    active = expand_service_active_dates(calendar, calendar_dates)
    assert representative_service_date(active, "monday")[1] == {"MON"}
    assert representative_service_date(active, "friday")[1] == {"FRI"}
    assert representative_service_date(active, "saturday")[1] == {"SAT"}


def test_representative_weekday_on_holiday_fixture_excludes_holiday_services(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calendar, calendar_dates = _load_fixture("gtfs_holiday_negation")
    active = expand_service_active_dates(calendar, calendar_dates)
    with caplog.at_level(logging.WARNING):
        chosen, ids = representative_service_date(active, "weekday")
    assert ids == {"WKD"}  # HOL and holiday-extended SUN never leak in
    assert chosen in active["WKD"]
    assert "varies by day of week" not in caplog.text


def test_representative_override_date() -> None:
    calendar, calendar_dates = _load_fixture("gtfs_split_weekday")
    active = expand_service_active_dates(calendar, calendar_dates)
    labor_day = dt.date(2026, 9, 7)
    chosen, ids = representative_service_date(active, "weekday", override_date=labor_day)
    assert chosen == labor_day
    assert ids == {"SUN"}  # Sunday schedule runs on the holiday


def test_representative_override_with_no_service_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    active = expand_service_active_dates(_simple_calendar())
    with caplog.at_level(logging.WARNING):
        chosen, ids = representative_service_date(
            active, "weekday", override_date=dt.date(2030, 1, 2)
        )
    assert ids == set()
    assert "No service is active" in caplog.text


def test_representative_exclude_dates() -> None:
    calendar, calendar_dates = _load_fixture("gtfs_split_weekday")
    active = expand_service_active_dates(calendar, calendar_dates)
    all_but_one_friday = {d for d in active["FRI"] if d != dt.date(2026, 6, 5)}
    chosen, ids = representative_service_date(active, "friday", exclude_dates=all_but_one_friday)
    assert chosen == dt.date(2026, 6, 5)
    assert ids == {"FRI"}


def test_representative_rejects_bad_day_and_empty_candidates() -> None:
    active = expand_service_active_dates(_simple_calendar())
    with pytest.raises(ValueError, match="service_day"):
        representative_service_date(active, "weekend")
    with pytest.raises(ValueError, match="No active dates"):
        representative_service_date(active, "sunday")
