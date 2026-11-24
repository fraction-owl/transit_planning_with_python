import openpyxl
import pandas as pd
import pytest

from scripts.ridership_tools import load_factor_monitor


# Define a fixture for the CSV data
@pytest.fixture
def input_df() -> pd.DataFrame:
    # Load the CSV fixture
    # We assume the test is run from the root of the repo
    csv_path = "tests/fixtures/statistics_by_route_and_trip.csv"
    df = pd.read_csv(csv_path)

    # The script expects 'TRIP_START_TIME' to be time-like objects (Timestamp or time)
    # The CSV has strings "HH:MM". We need to convert them.
    # We use errors='coerce' to handle the empty string in row 9, which becomes NaT.
    # We keep them as Timestamps because assign_service_period handles Timestamp.hour.
    df["TRIP_START_TIME"] = pd.to_datetime(df["TRIP_START_TIME"], format="%H:%M", errors="coerce")

    return df


def test_process_data_structure(input_df) -> None:
    """Test that process_data adds the required columns and handles basic logic."""
    processed = load_factor_monitor.process_data(
        input_df,
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=4,
    )

    expected_cols = [
        "SERVICE_PERIOD",
        "LOAD_FACTOR",
        "LOAD_FACTOR_VIOLATION",
        "ROUTE_LIMIT_TYPE",
    ]
    for col in expected_cols:
        assert col in processed.columns


def test_process_data_calculations(input_df) -> None:
    """Test specific calculations for load factor and violations."""
    # Row 5 in CSV: 20B, MAX_LOAD=50. Capacity=39.
    # Load Factor = 50/39 = 1.2821
    # Route 20B is not in LOWER_LIMIT_ROUTES (default), so limit is 1.25.
    # Should be a violation.

    processed = load_factor_monitor.process_data(
        input_df,
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=4,
    )

    # Find the row for 20B, Inbound, 17:45 (Serial 5)
    row_5 = processed[processed["SERIAL_NUMBER"] == 5].iloc[0]

    assert row_5["LOAD_FACTOR_VIOLATION"] == "TRUE"
    assert row_5["ROUTE_LIMIT_TYPE"] == "HIGH"
    assert row_5["SERVICE_PERIOD"] == "PM Peak"  # 17:45 -> 17 is in [15, 18)
    assert round(row_5["LOAD_FACTOR"], 2) == 1.28

    # Row 8: 40D, Inbound, 03:30, MAX_LOAD=10.
    # 03:30 -> hour 3. 4 <= 3 < 6 is False. -> "Other"
    row_8 = processed[processed["SERIAL_NUMBER"] == 8].iloc[0]
    assert row_8["SERVICE_PERIOD"] == "Other"

    # Row 9: Missing time. NaT. NaT.hour is nan. "Other".
    row_9 = processed[processed["SERIAL_NUMBER"] == 9].iloc[0]
    assert row_9["SERVICE_PERIOD"] == "Other"


def test_process_data_filtering(input_df) -> None:
    """Test route filtering logic."""
    # Filter IN only 10A
    processed_in = load_factor_monitor.process_data(
        input_df.copy(),
        bus_capacity=39,
        filter_in_routes=["10A"],
        filter_out_routes=[],
        decimals=4,
    )
    assert all(processed_in["ROUTE_NAME"] == "10A")

    # Filter OUT 10A
    processed_out = load_factor_monitor.process_data(
        input_df.copy(),
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=["10A"],
        decimals=4,
    )
    assert "10A" not in processed_out["ROUTE_NAME"].to_numpy()


def test_integration_exports(input_df, tmp_path, monkeypatch) -> None:
    """Integration test for the export functions, including the real openpyxl paths."""
    # create_route_workbooks writes to os.path.dirname(OUTPUT_FILE), and
    # export_to_excel uses OUTPUT_FILE itself, so point it into tmp_path.
    fake_output_xlsx = tmp_path / "processed_stats.xlsx"
    monkeypatch.setattr(load_factor_monitor, "OUTPUT_FILE", str(fake_output_xlsx))

    processed = load_factor_monitor.process_data(
        input_df,
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=4,
    )

    # 1. Export CSV
    csv_out = tmp_path / "out.csv"
    load_factor_monitor.export_to_csv(processed, str(csv_out))
    assert csv_out.exists()

    # 2. Export combined Excel, then read the workbook back to verify it
    load_factor_monitor.export_to_excel(processed, str(fake_output_xlsx))
    assert fake_output_xlsx.exists()

    combined_wb = openpyxl.load_workbook(fake_output_xlsx)
    combined_ws = combined_wb["Sheet1"]
    assert [cell.value for cell in combined_ws[1]] == list(processed.columns)
    assert combined_ws.max_row == len(processed) + 1  # header + one row per trip

    # 3. Create route workbooks: one file per route, one sheet per direction
    load_factor_monitor.create_route_workbooks(processed)

    expected_sheets_by_route = {
        "10A": {"Outbound", "Inbound"},
        "20B": {"Outbound", "Inbound"},
        "30C": {"Outbound"},
        "40D": {"Inbound"},
    }
    for route, directions in expected_sheets_by_route.items():
        route_wb_path = tmp_path / f"{route}.xlsx"
        assert route_wb_path.exists()
        route_wb = openpyxl.load_workbook(route_wb_path)
        assert set(route_wb.sheetnames) == directions

    # Spot-check one sheet: 20B Inbound holds the single 17:45 violation trip
    sheet = openpyxl.load_workbook(tmp_path / "20B.xlsx")["Inbound"]
    headers = [cell.value for cell in sheet[1]]
    assert headers == list(processed.columns)
    assert all(cell.font.bold for cell in sheet[1])
    assert sheet.max_row == 2
    trip = dict(zip(headers, [cell.value for cell in sheet[2]]))
    assert trip["SERIAL_NUMBER"] == 5
    assert trip["MAX_LOAD"] == 50
    assert trip["LOAD_FACTOR_VIOLATION"] == "TRUE"
    assert sheet.cell(row=2, column=headers.index("TRIP_START_TIME") + 1).number_format == "hh:mm"

    # 4. Violation Log
    log_out = tmp_path / "violations.txt"
    load_factor_monitor.write_violation_log(processed, str(log_out))
    assert log_out.exists()

    # Verify content of violation log
    content = log_out.read_text(encoding="utf-8")
    assert "20B" in content  # Should be listed as violation


_COLUMNS = [
    "SERIAL_NUMBER",
    "ROUTE_NAME",
    "DIRECTION_NAME",
    "TRIP_START_TIME",
    "BLOCK",
    "MAX_LOAD",
    "MAX_LOAD_P",
    "ALL_RECORDS_MAX_LOAD",
]


def _write_workbook(path, rows) -> str:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(_COLUMNS)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return str(path)


def test_numeric_excel_route_gets_its_configured_limit(tmp_path, monkeypatch) -> None:
    """A numeric ROUTE_NAME cell (101) matches the configured text route "101"."""
    monkeypatch.setattr(load_factor_monitor, "LOWER_LIMIT_ROUTES", ["101"])
    path = _write_workbook(
        tmp_path / "stats.xlsx",
        [
            [1, 101, "Inbound", "08:00", 1, 42, 42, 42],
            [2, "007", "Inbound", "08:30", 2, 10, 10, 10],  # A text ID with a leading zero.
        ],
    )
    data = load_factor_monitor.load_data(path)
    data["TRIP_START_TIME"] = pd.to_datetime(data["TRIP_START_TIME"], format="%H:%M")
    processed = load_factor_monitor.process_data(
        data, bus_capacity=39, filter_in_routes=[], filter_out_routes=[], decimals=4
    ).set_index("SERIAL_NUMBER")

    assert processed.loc[1, "ROUTE_NAME"] == "101"
    assert processed.loc[1, "LOAD_FACTOR"] == pytest.approx(1.0769)
    assert processed.loc[1, "ROUTE_LIMIT_TYPE"] == "LOW"
    assert processed.loc[1, "LOAD_FACTOR_VIOLATION"] == "TRUE"
    assert processed.loc[2, "ROUTE_NAME"] == "007"


def test_route_filters_and_limits_normalize_route_names(monkeypatch) -> None:
    """Numeric route values, including floats from a column with blanks, match text config."""
    monkeypatch.setattr(load_factor_monitor, "LOWER_LIMIT_ROUTES", [101])
    data = pd.DataFrame(
        {
            "SERIAL_NUMBER": [1, 2, 3],
            "ROUTE_NAME": [101.0, " 202 ", "007"],
            "TRIP_START_TIME": pd.to_datetime(["08:00"] * 3, format="%H:%M"),
            "MAX_LOAD": [42, 42, 42],
        }
    )
    kept = load_factor_monitor.process_data(
        data, bus_capacity=39, filter_in_routes=["101", 202], filter_out_routes=[], decimals=4
    ).set_index("SERIAL_NUMBER")
    assert sorted(kept.index) == [1, 2]
    assert kept["ROUTE_NAME"].to_dict() == {1: "101", 2: "202"}
    assert kept["ROUTE_LIMIT_TYPE"].to_dict() == {1: "LOW", 2: "HIGH"}
    # "7" does not match the text ID "007": leading zeros are kept, not stripped.
    dropped = load_factor_monitor.process_data(
        data, bus_capacity=39, filter_in_routes=[], filter_out_routes=["7"], decimals=4
    )
    assert "007" in set(dropped["ROUTE_NAME"])


def _one_trip(max_load: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SERIAL_NUMBER": [1],
            "ROUTE_NAME": ["101"],
            "DIRECTION_NAME": ["Inbound"],
            "TRIP_START_TIME": pd.to_datetime(["08:00"], format="%H:%M"),
            "MAX_LOAD": [max_load],
        }
    )


def test_blank_max_load_is_unknown_not_compliant(tmp_path, monkeypatch) -> None:
    """A trip with no observed load is flagged UNKNOWN, and the log does not call it compliant."""
    processed = load_factor_monitor.process_data(
        _one_trip(float("nan")),
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=4,
    )
    assert pd.isna(processed.iloc[0]["LOAD_FACTOR"])
    assert processed.iloc[0]["LOAD_FACTOR_VIOLATION"] == "UNKNOWN"

    log_out = tmp_path / "violations.txt"
    load_factor_monitor.write_violation_log(processed, str(log_out))
    content = log_out.read_text(encoding="utf-8")
    assert "all trips within permissible limits" not in content
    assert "1 trip(s) have no MAX_LOAD" in content


@pytest.mark.parametrize("decimals", [0, 1, 4])
def test_display_precision_does_not_change_compliance(monkeypatch, decimals: int) -> None:
    """40 riders on a 39-seat bus (1.026) exceed a 1.0 limit at any display precision."""
    monkeypatch.setattr(load_factor_monitor, "LOWER_LIMIT_ROUTES", ["101"])
    processed = load_factor_monitor.process_data(
        _one_trip(40),
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=decimals,
    )
    row = processed.iloc[0]
    assert row["LOAD_FACTOR"] == round(40 / 39, decimals)
    assert row["LOAD_FACTOR_VIOLATION"] == "TRUE"
