from __future__ import annotations

import datetime as dt

import numpy as np

from utils.time_helpers import federal_holidays_observed, minutes_to_hhmm, parse_time_to_minutes

# ---------------------------------------------------------------------------
# parse_time_to_minutes
# ---------------------------------------------------------------------------


def test_parse_time_to_minutes_hhmmss() -> None:
    assert parse_time_to_minutes("07:05:00") == 425


def test_parse_time_to_minutes_hhmm() -> None:
    assert parse_time_to_minutes("7:05") == 425


def test_parse_time_to_minutes_midnight() -> None:
    assert parse_time_to_minutes("00:00:00") == 0


def test_parse_time_to_minutes_past_midnight() -> None:
    assert parse_time_to_minutes("26:30:00") == 1590


def test_parse_time_to_minutes_rounds_seconds() -> None:
    assert parse_time_to_minutes("06:00:31") == 361
    assert parse_time_to_minutes("06:00:29") == 360
    # Python round() is round-half-even, so :30 rounds down here.
    assert parse_time_to_minutes("06:00:30") == 360


def test_parse_time_to_minutes_strips_whitespace() -> None:
    assert parse_time_to_minutes(" 07:05:00 ") == 425


def test_parse_time_to_minutes_none_returns_none() -> None:
    assert parse_time_to_minutes(None) is None


def test_parse_time_to_minutes_non_string_returns_none() -> None:
    assert parse_time_to_minutes(425) is None  # type: ignore[arg-type]


def test_parse_time_to_minutes_malformed_returns_none() -> None:
    assert parse_time_to_minutes("not-a-time") is None
    assert parse_time_to_minutes("07") is None
    assert parse_time_to_minutes("07:05:00:00") is None


def test_parse_time_to_minutes_out_of_range_fields_return_none() -> None:
    assert parse_time_to_minutes("07:65") is None
    assert parse_time_to_minutes("07:05:99") is None
    assert parse_time_to_minutes("-1:05") is None


# ---------------------------------------------------------------------------
# minutes_to_hhmm
# ---------------------------------------------------------------------------


def test_minutes_to_hhmm_basic() -> None:
    assert minutes_to_hhmm(425) == "07:05"


def test_minutes_to_hhmm_midnight() -> None:
    assert minutes_to_hhmm(0) == "00:00"


def test_minutes_to_hhmm_past_midnight() -> None:
    assert minutes_to_hhmm(1590) == "26:30"


def test_minutes_to_hhmm_rounds_fractional_minutes() -> None:
    assert minutes_to_hhmm(425.6) == "07:06"


def test_minutes_to_hhmm_none_returns_default_missing() -> None:
    assert minutes_to_hhmm(None) == ""


def test_minutes_to_hhmm_nan_returns_default_missing() -> None:
    assert minutes_to_hhmm(float("nan")) == ""
    assert minutes_to_hhmm(np.nan) == ""


def test_minutes_to_hhmm_custom_missing_sentinel() -> None:
    assert minutes_to_hhmm(None, "–") == "–"


def test_minutes_to_hhmm_round_trips_with_parse() -> None:
    assert minutes_to_hhmm(parse_time_to_minutes("26:30:00")) == "26:30"


# ---------------------------------------------------------------------------
# federal_holidays_observed
# ---------------------------------------------------------------------------


def test_federal_holidays_2025_exact_set() -> None:
    # 2025 has no weekend-shifted holidays; every observed date is the holiday.
    assert federal_holidays_observed(2025) == {
        dt.date(2025, 1, 1),  # New Year's Day (Wed)
        dt.date(2025, 1, 20),  # MLK Day (3rd Mon Jan)
        dt.date(2025, 2, 17),  # Washington's Birthday (3rd Mon Feb)
        dt.date(2025, 5, 26),  # Memorial Day (last Mon May)
        dt.date(2025, 6, 19),  # Juneteenth (Thu)
        dt.date(2025, 7, 4),  # Independence Day (Fri)
        dt.date(2025, 9, 1),  # Labor Day (1st Mon Sep)
        dt.date(2025, 10, 13),  # Columbus Day (2nd Mon Oct)
        dt.date(2025, 11, 11),  # Veterans Day (Tue)
        dt.date(2025, 11, 27),  # Thanksgiving (4th Thu Nov)
        dt.date(2025, 12, 25),  # Christmas (Thu)
    }


def test_federal_holidays_saturday_observed_preceding_friday() -> None:
    # July 4, 2026 is a Saturday -> observed Friday July 3.
    holidays = federal_holidays_observed(2026)
    assert dt.date(2026, 7, 3) in holidays
    assert dt.date(2026, 7, 4) not in holidays


def test_federal_holidays_weekend_observance_can_cross_year() -> None:
    # Jan 1, 2022 was a Saturday -> observed Friday Dec 31, 2021 (previous year).
    holidays = federal_holidays_observed(2022)
    assert dt.date(2021, 12, 31) in holidays
    assert dt.date(2022, 1, 1) not in holidays
    # Christmas 2022 was a Sunday -> observed Monday Dec 26.
    assert dt.date(2022, 12, 26) in holidays
    assert dt.date(2022, 12, 25) not in holidays


def test_federal_holidays_juneteenth_absent_before_2021() -> None:
    assert dt.date(2021, 6, 18) in federal_holidays_observed(2021)  # Sat -> Fri
    assert not any(
        day.month == 6 and day.day in (18, 19, 21) for day in federal_holidays_observed(2020)
    )
