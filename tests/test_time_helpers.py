from __future__ import annotations

import numpy as np

from utils.time_helpers import minutes_to_hhmm, parse_time_to_minutes

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
