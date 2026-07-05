from __future__ import annotations

import pandas as pd
import pytest

import scripts.operations_tools.convert_to_tides_stop_visits as target

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _input_df() -> pd.DataFrame:
    """Two-stop trip mirroring a CLEVER Stop Visit Events export."""
    return pd.DataFrame(
        {
            "Date": ["1/15/2025", "1/15/2025"],
            "Trip": ["04:02 1550064", "04:02 1550064"],
            "Route": ["101 - Fort Hunt - Mount Vernon", "101 - Fort Hunt - Mount Vernon"],
            "Direction": ["NORTH", "NORTH"],
            "Vehicle": ["7906.0", "7906.0"],
            "Variation": ["1", "1"],
            "Timepoint Order": ["1", "2"],
            "Timepoint ID": ["TP1", "TP2"],
            "Actual Time": ["1/15/2025 4:05:00 AM", "1/15/2025 4:20:00 AM"],
            "Arrival Time": ["1/15/2025 4:04:30 AM", "1/15/2025 4:19:00 AM"],
            "Departure Time": ["1/15/2025 4:05:10 AM", "1/15/2025 4:20:30 AM"],
            "Scheduled Passing Time": ["1/15/2025 4:02:00 AM", "1/15/2025 4:18:00 AM"],
        }
    )


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_strips_and_collapses_whitespace() -> None:
    s = pd.Series(["  a   b  ", "c"])
    assert list(target.normalize_text(s)) == ["a b", "c"]


def test_normalize_text_coerces_blank_tokens_to_na() -> None:
    s = pd.Series(["", "nan", "N/A", "ok"])
    out = target.normalize_text(s)
    assert out.isna().tolist() == [True, True, True, False]


# ---------------------------------------------------------------------------
# parse_service_date
# ---------------------------------------------------------------------------


def test_parse_service_date_converts_to_iso() -> None:
    out = target.parse_service_date(pd.Series(["1/15/2025", "12/31/2024"]))
    assert list(out) == ["2025-01-15", "2024-12-31"]


def test_parse_service_date_bad_values_become_na(caplog) -> None:
    with caplog.at_level("WARNING"):
        out = target.parse_service_date(pd.Series(["not a date"]))
    assert out.isna().all()
    assert "Failed to parse" in caplog.text


# ---------------------------------------------------------------------------
# split_trip_token
# ---------------------------------------------------------------------------


def test_split_trip_token_extracts_second_token() -> None:
    out = target.split_trip_token(pd.Series(["04:02 1550064"]))
    assert out.iloc[0] == "1550064"


def test_split_trip_token_missing_token_is_na() -> None:
    out = target.split_trip_token(pd.Series(["04:02"]))
    assert out.isna().iloc[0]


# ---------------------------------------------------------------------------
# stable_id
# ---------------------------------------------------------------------------


def test_stable_id_is_deterministic() -> None:
    assert target.stable_id("a", "b") == target.stable_id("a", "b")
    assert len(target.stable_id("a", "b")) == 16


def test_stable_id_differs_for_different_inputs() -> None:
    assert target.stable_id("a", "b") != target.stable_id("a", "c")


# ---------------------------------------------------------------------------
# parse_route_short / normalize_vehicle_id
# ---------------------------------------------------------------------------


def test_parse_route_short_takes_token_left_of_dash() -> None:
    out = target.parse_route_short(pd.Series(["101 - Fort Hunt - Mount Vernon"]))
    assert out.iloc[0] == "101"


def test_parse_route_short_blank_is_na() -> None:
    assert target.parse_route_short(pd.Series([""])).isna().iloc[0]


def test_normalize_vehicle_id_strips_float_suffix() -> None:
    out = target.normalize_vehicle_id(pd.Series(["7906.0", " 123 "]))
    assert list(out) == ["7906", "123"]


# ---------------------------------------------------------------------------
# normalize_timepoint_order
# ---------------------------------------------------------------------------


def test_normalize_timepoint_order_parses_integers() -> None:
    out = target.normalize_timepoint_order(pd.Series(["1", "2", "3"]))
    assert list(out) == [1, 2, 3]


def test_normalize_timepoint_order_zero_shifts_to_one() -> None:
    out = target.normalize_timepoint_order(pd.Series(["0", "1"]))
    assert list(out) == [1, 1]


def test_normalize_timepoint_order_negative_becomes_na() -> None:
    out = target.normalize_timepoint_order(pd.Series(["-1", "2"]))
    assert out.isna().iloc[0]
    assert out.iloc[1] == 2


def test_normalize_timepoint_order_unparseable_becomes_na() -> None:
    out = target.normalize_timepoint_order(pd.Series(["abc"]))
    assert out.isna().all()


# ---------------------------------------------------------------------------
# dt_to_iso
# ---------------------------------------------------------------------------


def test_dt_to_iso_formats_and_preserves_na() -> None:
    s = pd.to_datetime(pd.Series(["2025-01-15 04:05:00", None]))
    out = target.dt_to_iso(s)
    assert out.iloc[0] == "2025-01-15T04:05:00"
    assert pd.isna(out.iloc[1])


# ---------------------------------------------------------------------------
# warn_missing_columns
# ---------------------------------------------------------------------------


def test_warn_missing_columns_raises_on_missing_required() -> None:
    df = _input_df().drop(columns=["Date"])
    with pytest.raises(ValueError, match="Date"):
        target.warn_missing_columns(df)


def test_warn_missing_columns_optional_only_warns(caplog) -> None:
    df = _input_df().drop(columns=["Vehicle"])
    with caplog.at_level("WARNING"):
        target.warn_missing_columns(df)
    assert "Vehicle" in caplog.text


# ---------------------------------------------------------------------------
# convert_to_tides (end-to-end)
# ---------------------------------------------------------------------------


def test_convert_to_tides_output_columns_match_schema_order() -> None:
    out = target.convert_to_tides(_input_df())
    assert list(out.columns) == target.TIDES_COLS


def test_convert_to_tides_core_fields() -> None:
    out = target.convert_to_tides(_input_df())
    row = out.iloc[0]
    assert row["service_date"] == "2025-01-15"
    assert row["trip_id_performed"] == "1550064"
    assert row["vehicle_id"] == "7906"
    assert row["stop_id"] == "TP1"
    assert row["trip_stop_sequence"] == 1
    assert row["scheduled_stop_sequence"] == 1
    assert bool(row["timepoint"]) is True
    assert row["schedule_relationship"] == "Scheduled"


def test_convert_to_tides_pattern_id_combines_route_direction_variation() -> None:
    out = target.convert_to_tides(_input_df())
    assert out["pattern_id"].iloc[0] == "101|NORTH|1"


def test_convert_to_tides_dwell_from_arrival_and_departure() -> None:
    out = target.convert_to_tides(_input_df())
    # 4:04:30 -> 4:05:10 is 40 seconds.
    assert out["dwell"].iloc[0] == 40


def test_convert_to_tides_negative_dwell_left_blank(caplog) -> None:
    df = _input_df()
    df.loc[0, "Departure Time"] = "1/15/2025 4:00:00 AM"  # before arrival
    with caplog.at_level("WARNING"):
        out = target.convert_to_tides(df)
    assert pd.isna(out["dwell"].iloc[0])
    assert "negative dwell" in caplog.text


def test_convert_to_tides_actual_time_fallback() -> None:
    df = _input_df().drop(columns=["Arrival Time", "Departure Time"])
    out = target.convert_to_tides(df)
    assert out["actual_arrival_time"].iloc[0] == "2025-01-15T04:05:00"
    assert out["actual_departure_time"].iloc[0] == "2025-01-15T04:05:00"


def test_convert_to_tides_scheduled_passing_fills_both_schedule_times() -> None:
    out = target.convert_to_tides(_input_df())
    assert out["schedule_arrival_time"].iloc[0] == "2025-01-15T04:02:00"
    assert out["schedule_departure_time"].iloc[0] == "2025-01-15T04:02:00"


def test_convert_to_tides_unsupported_fields_left_blank() -> None:
    out = target.convert_to_tides(_input_df())
    assert out["boarding_1"].isna().all()
    assert out["revenue"].isna().all()
