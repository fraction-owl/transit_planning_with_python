from __future__ import annotations

import pandas as pd
import pytest

import scripts.operations_tools.convert_to_tides_trips_performed as target

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _input_df() -> pd.DataFrame:
    """Two revenue trips plus one deadhead, CLEVER Event Runtime Analysis style."""
    return pd.DataFrame(
        {
            "Vehicle": ["7785.0", "7786", "7787"],
            "Route": ["301 - Telegraph Rd", "302 - Beacon Hill", "301 - Telegraph Rd"],
            "Direction": ["Northbound", "Southbound", "Northbound"],
            "Block": ["B1", "B2", "B3"],
            "TripID": ["1550064", "1550065", "1550066"],
            "Trip Type": ["Revenue", "Revenue", "Deadhead"],
            "Scheduled Start Time": [
                "1/15/2025 4:02:00 AM",
                "1/15/2025 5:00:00 AM",
                "1/15/2025 3:30:00 AM",
            ],
            "Scheduled Finish Time": [
                "1/15/2025 4:45:00 AM",
                "1/15/2025 5:45:00 AM",
                "1/15/2025 4:00:00 AM",
            ],
            "Actual Start Time": [
                "1/15/2025 4:03:10 AM",
                "1/15/2025 5:01:00 AM",
                "1/15/2025 3:31:00 AM",
            ],
            "Actual Finish Time": [
                "1/15/2025 4:46:00 AM",
                "1/15/2025 5:44:00 AM",
                "1/15/2025 3:59:00 AM",
            ],
            "Operator": ["OP1", "OP2", "OP3"],
            "Last Stop": ["S100", "S200", "S300"],
        }
    )


# ---------------------------------------------------------------------------
# parse_route_id_from_route_text
# ---------------------------------------------------------------------------


def test_parse_route_id_takes_left_of_dash() -> None:
    out = target.parse_route_id_from_route_text(pd.Series(["301 - Telegraph Rd"]))
    assert out.iloc[0] == "301"


def test_parse_route_id_keeps_non_numeric_token() -> None:
    out = target.parse_route_id_from_route_text(pd.Series(["REX - Richmond Hwy Express"]))
    assert out.iloc[0] == "REX"


def test_parse_route_id_blank_is_na() -> None:
    assert target.parse_route_id_from_route_text(pd.Series([""])).isna().iloc[0]


# ---------------------------------------------------------------------------
# normalize_vehicle_id
# ---------------------------------------------------------------------------


def test_normalize_vehicle_id_strips_float_suffix_and_whitespace() -> None:
    out = target.normalize_vehicle_id(pd.Series(["7785.0", " 123 ", ""]))
    assert out.iloc[0] == "7785"
    assert out.iloc[1] == "123"
    assert pd.isna(out.iloc[2])


# ---------------------------------------------------------------------------
# stable_id / direction_id_from_text
# ---------------------------------------------------------------------------


def test_stable_id_deterministic_and_short() -> None:
    assert target.stable_id("x", "y") == target.stable_id("x", "y")
    assert len(target.stable_id("x", "y")) == 16


def test_direction_id_from_text_maps_cardinal_directions() -> None:
    s = pd.Series(["Northbound", "SOUTHBOUND", "eastbound", "Loop"])
    out = target.direction_id_from_text(s)
    assert out.iloc[0] == 0
    assert out.iloc[1] == 1
    assert out.iloc[2] == 1
    assert pd.isna(out.iloc[3])


# ---------------------------------------------------------------------------
# summarize_trip_type_drops
# ---------------------------------------------------------------------------


def test_summarize_trip_type_drops_filters_and_counts() -> None:
    df = _input_df()
    kept, dropped = target.summarize_trip_type_drops(df, "Trip Type", keep_trip_type="Revenue")
    assert len(kept) == 2
    assert dropped == {"Deadhead": 1}


def test_summarize_trip_type_drops_disabled_when_none() -> None:
    df = _input_df()
    kept, dropped = target.summarize_trip_type_drops(df, "Trip Type", keep_trip_type=None)
    assert len(kept) == 3
    assert dropped == {}


# ---------------------------------------------------------------------------
# map_clever_trip_type_to_tides
# ---------------------------------------------------------------------------


def test_map_trip_type_known_labels() -> None:
    s = pd.Series(["Revenue", "PULL-IN", "pull out", "Layover"])
    out = target.map_clever_trip_type_to_tides(s)
    assert list(out) == ["In service", "Pullin", "Pullout", "Layover"]


def test_map_trip_type_unknown_label_defaults() -> None:
    out = target.map_clever_trip_type_to_tides(pd.Series(["Charter"]))
    assert out.iloc[0] == "Other not in service"


def test_map_trip_type_missing_stays_na() -> None:
    out = target.map_clever_trip_type_to_tides(pd.Series([pd.NA], dtype="string"))
    assert out.isna().iloc[0]


# ---------------------------------------------------------------------------
# choose_trip_id_performed
# ---------------------------------------------------------------------------


def test_choose_trip_id_performed_unique_ids_pass_through() -> None:
    service_date = pd.Series(["2025-01-15", "2025-01-15"], dtype="string")
    scheduled = pd.Series(["A", "B"], dtype="string")
    vehicle = pd.Series(["V1", "V2"], dtype="string")
    start = pd.to_datetime(pd.Series(["2025-01-15 04:00", "2025-01-15 05:00"]))
    perf, n_dupes = target.choose_trip_id_performed(service_date, scheduled, vehicle, start)
    assert list(perf) == ["A", "B"]
    assert n_dupes == 0


def test_choose_trip_id_performed_hashes_duplicates() -> None:
    service_date = pd.Series(["2025-01-15", "2025-01-15"], dtype="string")
    scheduled = pd.Series(["A", "A"], dtype="string")
    vehicle = pd.Series(["V1", "V2"], dtype="string")
    start = pd.to_datetime(pd.Series(["2025-01-15 04:00", "2025-01-15 05:00"]))
    perf, n_dupes = target.choose_trip_id_performed(service_date, scheduled, vehicle, start)
    assert n_dupes == 2
    assert perf.iloc[0].startswith("perf_")
    assert perf.iloc[1].startswith("perf_")
    assert perf.iloc[0] != perf.iloc[1]


# ---------------------------------------------------------------------------
# convert_to_tides (end-to-end)
# ---------------------------------------------------------------------------


def test_convert_to_tides_output_columns_match_schema_order() -> None:
    out = target.convert_to_tides(_input_df())
    assert list(out.columns) == target.TIDES_COLS


def test_convert_to_tides_keeps_only_revenue_trips() -> None:
    out = target.convert_to_tides(_input_df())
    assert len(out) == 2
    assert set(out["trip_id_scheduled"]) == {"1550064", "1550065"}


def test_convert_to_tides_core_fields() -> None:
    out = target.convert_to_tides(_input_df())
    row = out.iloc[0]
    assert row["service_date"] == "2025-01-15"
    assert row["vehicle_id"] == "7785"
    assert row["route_id"] == "301"
    assert row["direction_id"] == 0
    assert row["block_id"] == "B1"
    assert row["operator_id"] == "OP1"
    assert row["trip_end_stop_id"] == "S100"
    assert row["trip_type"] == "In service"
    assert row["schedule_relationship"] == "Scheduled"
    assert row["schedule_trip_start"] == "2025-01-15T04:02:00"
    assert row["actual_trip_end"] == "2025-01-15T04:46:00"


def test_convert_to_tides_missing_required_column_raises() -> None:
    df = _input_df().drop(columns=["Vehicle"])
    with pytest.raises(ValueError, match="Vehicle"):
        target.convert_to_tides(df)


def test_convert_to_tides_drops_rows_without_vehicle(caplog) -> None:
    df = _input_df()
    df.loc[0, "Vehicle"] = ""
    with caplog.at_level("WARNING"):
        out = target.convert_to_tides(df)
    assert len(out) == 1
    assert "missing Vehicle" in caplog.text


def test_convert_to_tides_drops_undatable_rows(caplog) -> None:
    df = _input_df()
    df.loc[0, "Scheduled Start Time"] = "garbage"
    df.loc[0, "Actual Start Time"] = "garbage"
    with caplog.at_level("WARNING"):
        out = target.convert_to_tides(df)
    assert len(out) == 1
    assert "Dropping 1 rows" in caplog.text


def test_convert_to_tides_falls_back_to_actual_start_for_dating() -> None:
    df = _input_df()
    df.loc[0, "Scheduled Start Time"] = ""
    out = target.convert_to_tides(df)
    assert out.iloc[0]["service_date"] == "2025-01-15"
