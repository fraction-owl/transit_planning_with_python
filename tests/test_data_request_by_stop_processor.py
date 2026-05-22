import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add the script directory to sys.path to allow importing the module
script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import data_request_by_stop_processor as target  # noqa: E402

# Fixture path
FIXTURE_PATH = Path("tests/fixtures/ridership_by_route_and_stop.csv")


def test_extract_config_block(tmp_path: Path) -> None:
    """extract_config_block returns only the lines between the two markers."""
    source = tmp_path / "script.py"
    source.write_text(
        "# preamble\n"
        "# === BEGIN CONFIG ===\n"
        "KEY = 1\n"
        "OTHER = 2\n"
        "# === END CONFIG ===\n"
        "# epilogue\n",
        encoding="utf-8",
    )
    block = target.extract_config_block(source)
    assert "KEY = 1" in block
    assert "OTHER = 2" in block
    assert "preamble" not in block
    assert "epilogue" not in block
    assert "BEGIN CONFIG" not in block
    assert "END CONFIG" not in block


def test_extract_config_block_missing_markers(tmp_path: Path) -> None:
    """extract_config_block raises ValueError when markers are absent."""
    source = tmp_path / "script.py"
    source.write_text("KEY = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Config markers not found"):
        target.extract_config_block(source)


def test_write_run_log_creates_file(tmp_path: Path) -> None:
    """write_run_log writes a _runlog.txt next to the output file."""
    output_file = tmp_path / "output.xlsx"
    log_path = tmp_path / "output_runlog.txt"

    fake_source = "# === BEGIN CONFIG ===\nKEY = 1\n# === END CONFIG ===\n"
    with patch(
        "data_request_by_stop_processor._resolve_script_source",
        return_value=(fake_source, "<test>"),
    ):
        result = target.write_run_log(output_file)

    assert result is True
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "RIDERSHIP PROCESSING RUN LOG" in content
    assert "KEY = 1" in content
    assert str(output_file) in content


def test_write_run_log_returns_false_on_io_error(tmp_path: Path) -> None:
    """write_run_log returns False (and logs) when the file cannot be written."""
    output_file = tmp_path / "output.xlsx"

    fake_source = "# === BEGIN CONFIG ===\nK=1\n# === END CONFIG ===\n"
    with (
        patch(
            "data_request_by_stop_processor._resolve_script_source",
            return_value=(fake_source, "<test>"),
        ),
        patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
    ):
        result = target.write_run_log(output_file)

    assert result is False


def test_full_processing_integration() -> None:
    """Verify the full processing pipeline using the CSV fixture."""
    # 1. Load the fixture data
    if not FIXTURE_PATH.exists():
        pytest.fail(f"Fixture file not found at {FIXTURE_PATH}")

    fixture_df = pd.read_csv(FIXTURE_PATH)

    # 2. Patch dependencies and configuration
    # We patch the module-level variables and functions
    with (
        patch("data_request_by_stop_processor.read_excel_file") as mock_read,
        patch("data_request_by_stop_processor.write_to_excel") as mock_write,
        patch("data_request_by_stop_processor.write_run_log", return_value=True),
        patch("data_request_by_stop_processor.INPUT_FILE_PATH", Path("dummy_input.xlsx")),
        patch("data_request_by_stop_processor.OUTPUT_DIR", Path("dummy_output")),
        patch("data_request_by_stop_processor.ROUTES", []),
        patch("data_request_by_stop_processor.ROUTES_EXCLUDE", []),
        patch("data_request_by_stop_processor.STOP_IDS", []),
        patch("data_request_by_stop_processor.TIME_PERIODS", ["AM PEAK", "PM PEAK"]),
        patch("data_request_by_stop_processor.AGGREGATE_ROUTES_TOGETHER", True),
        patch("data_request_by_stop_processor.APPLY_ROUNDING", True),
        patch("data_request_by_stop_processor.AGGREGATE_BIN_RANGES", False),
    ):
        # Setup mock return value for read_excel_file
        mock_read.return_value = fixture_df.copy()

        # 3. Run the main function
        target.main()

        # 4. Assertions
        mock_read.assert_called_once()
        mock_write.assert_called_once()

        # Extract arguments passed to write_to_excel
        args, _ = mock_write.call_args
        # signature: (output_file, filtered_data, aggregated_peaks, all_time_aggregated)
        # args[0] is output_file
        filtered_data = args[1]
        aggregated_peaks = args[2]
        all_time_aggregated = args[3]

        # Check filtered_data (Original sheet)
        # Should contain all rows since we didn't filter by route/stop
        # And it should contain MIDDAY even though it's not in TIME_PERIODS config
        assert "MIDDAY" in filtered_data["TIME_PERIOD"].to_numpy(), (
            "MIDDAY rows should be preserved in Original sheet"
        )

        # Check aggregated_peaks (AM PEAK, PM PEAK)
        assert "AM PEAK" in aggregated_peaks
        assert "PM PEAK" in aggregated_peaks
        assert "MIDDAY" not in aggregated_peaks, (
            "MIDDAY should not be in aggregated_peaks as it was not in TIME_PERIODS config"
        )

        # Check specific aggregation logic
        # Focus on Stop 1001 (Main St & 1st Ave)
        # Fixture Data for Stop 1001:
        # AM PEAK, 10A: Board 12.4, Alight 3.2
        # AM PEAK, 10B: Board 18.1, Alight 6.7
        # PM PEAK, 10A: Board 22,   Alight 11.9
        # MIDDAY,  10A: Board 30,   Alight 28

        # --- Test AM PEAK Aggregation ---
        # Expected AM PEAK Board: 12.4 + 18.1 = 30.5
        # Expected AM PEAK Alight: 3.2 + 6.7 = 9.9
        am_peak_df = aggregated_peaks["AM PEAK"]
        stop_1001_am = am_peak_df[am_peak_df["STOP_ID"] == 1001]

        assert not stop_1001_am.empty, "Stop 1001 should be present in AM PEAK aggregation"
        # Access using iloc[0]
        assert stop_1001_am.iloc[0]["BOARD_ALL_TOTAL"] == 30.5, (
            f"Expected 30.5, got {stop_1001_am.iloc[0]['BOARD_ALL_TOTAL']}"
        )
        assert stop_1001_am.iloc[0]["ALIGHT_ALL_TOTAL"] == 9.9, (
            f"Expected 9.9, got {stop_1001_am.iloc[0]['ALIGHT_ALL_TOTAL']}"
        )

        # --- Test All Time Aggregation ---
        # Expected All Time Board: 12.4 (AM) + 18.1 (AM) + 30 (MIDDAY) + 22 (PM) = 82.5
        # Expected All Time Alight: 3.2 + 6.7 + 28 + 11.9 = 49.8
        stop_1001_all = all_time_aggregated[all_time_aggregated["STOP_ID"] == 1001]

        assert not stop_1001_all.empty, "Stop 1001 should be present in All Time aggregation"
        assert stop_1001_all.iloc[0]["BOARD_ALL_TOTAL"] == 82.5, (
            f"Expected 82.5, got {stop_1001_all.iloc[0]['BOARD_ALL_TOTAL']}"
        )
        assert stop_1001_all.iloc[0]["ALIGHT_ALL_TOTAL"] == 49.8, (
            f"Expected 49.8, got {stop_1001_all.iloc[0]['ALIGHT_ALL_TOTAL']}"
        )

        # Verify Routes column aggregation
        # For Stop 1001, routes are 10A and 10B in AM PEAK.
        # The script sorts and joins unique routes.
        assert stop_1001_am.iloc[0]["ROUTES"] == "10A, 10B", (
            f"Expected '10A, 10B', got {stop_1001_am.iloc[0]['ROUTES']}"
        )


def test_build_route_level_analysis_percentages() -> None:
    """Route-level percentages divide by full-route totals, not the filtered slice."""
    df = pd.read_csv(FIXTURE_PATH)
    df["ROUTE_NAME"] = df["ROUTE_NAME"].astype(str).str.strip()

    # Filter to one stop on route 30; the other stop (4005) must still appear in
    # the denominator so the % columns reflect the full route.
    result = target.build_route_level_analysis(df, stop_ids_filter=[4004])

    route_30 = result[result["ROUTE_NAME"] == "30"].set_index("STOP_ID")
    assert set(route_30.index) == {4004, 4005}, (
        "Route Analysis must include every stop on the route, not just filtered ones"
    )

    # Route 30 totals:
    #   4004: board = 42 + 55.2 = 97.2,  alight = 38.1 + 47   = 85.1
    #   4005: board = 15.6 + 18.9 = 34.5, alight = 12.3 + 16.2 = 28.5
    # Route board total = 131.7; route alight total = 113.6; route total = 245.3
    assert route_30.loc[4004, "Boardings"] == pytest.approx(97.2)
    assert route_30.loc[4004, "Route Boardings"] == pytest.approx(131.7)
    assert route_30.loc[4004, "% of Route Boardings"] == pytest.approx(73.8, abs=0.01)
    assert route_30.loc[4004, "% of Route Total"] == pytest.approx(74.32, abs=0.01)

    # The two stops' percentages must sum to 100 on each metric.
    assert route_30["% of Route Boardings"].sum() == pytest.approx(100.0, abs=0.01)
    assert route_30["% of Route Alightings"].sum() == pytest.approx(100.0, abs=0.01)
    assert route_30["% of Route Total"].sum() == pytest.approx(100.0, abs=0.01)

    # In Filter flag tracks STOP_IDS membership.
    assert bool(route_30.loc[4004, "In Filter"]) is True
    assert bool(route_30.loc[4005, "In Filter"]) is False


def test_build_route_level_analysis_no_stop_filter() -> None:
    """With no STOP_IDS filter, every row is marked In Filter = True."""
    df = pd.read_csv(FIXTURE_PATH)
    df["ROUTE_NAME"] = df["ROUTE_NAME"].astype(str).str.strip()
    result = target.build_route_level_analysis(df, stop_ids_filter=[])
    assert result["In Filter"].all()


def test_main_passes_route_analysis_when_enabled() -> None:
    """main() builds route analysis from route-only-filtered data when the flag is on."""
    fixture_df = pd.read_csv(FIXTURE_PATH)

    with (
        patch("data_request_by_stop_processor.read_excel_file") as mock_read,
        patch("data_request_by_stop_processor.write_to_excel") as mock_write,
        patch("data_request_by_stop_processor.write_run_log", return_value=True),
        patch("data_request_by_stop_processor.INPUT_FILE_PATH", Path("dummy_input.xlsx")),
        patch("data_request_by_stop_processor.OUTPUT_DIR", Path("dummy_output")),
        patch("data_request_by_stop_processor.ROUTES", ["30"]),
        patch("data_request_by_stop_processor.ROUTES_EXCLUDE", []),
        patch("data_request_by_stop_processor.STOP_IDS", [4004]),
        patch("data_request_by_stop_processor.TIME_PERIODS", ["AM PEAK", "PM PEAK"]),
        patch("data_request_by_stop_processor.AGGREGATE_ROUTES_TOGETHER", True),
        patch("data_request_by_stop_processor.APPLY_ROUNDING", True),
        patch("data_request_by_stop_processor.AGGREGATE_BIN_RANGES", False),
        patch("data_request_by_stop_processor.EXPORT_ROUTE_LEVEL_ANALYSIS", True),
    ):
        mock_read.return_value = fixture_df.copy()
        target.main()

        _, kwargs = mock_write.call_args
        route_analysis = kwargs["route_analysis"]

    assert route_analysis is not None
    # Both stops on route 30 must be present even though STOP_IDS filtered to 4004.
    assert set(route_analysis["STOP_ID"].tolist()) == {4004, 4005}
    assert bool(route_analysis.loc[route_analysis["STOP_ID"] == 4004, "In Filter"].iloc[0]) is True
    assert bool(route_analysis.loc[route_analysis["STOP_ID"] == 4005, "In Filter"].iloc[0]) is False


def test_main_omits_route_analysis_when_disabled() -> None:
    """main() leaves route_analysis as None when the flag is off (default)."""
    fixture_df = pd.read_csv(FIXTURE_PATH)

    with (
        patch("data_request_by_stop_processor.read_excel_file") as mock_read,
        patch("data_request_by_stop_processor.write_to_excel") as mock_write,
        patch("data_request_by_stop_processor.write_run_log", return_value=True),
        patch("data_request_by_stop_processor.INPUT_FILE_PATH", Path("dummy_input.xlsx")),
        patch("data_request_by_stop_processor.OUTPUT_DIR", Path("dummy_output")),
        patch("data_request_by_stop_processor.ROUTES", []),
        patch("data_request_by_stop_processor.ROUTES_EXCLUDE", []),
        patch("data_request_by_stop_processor.STOP_IDS", []),
        patch("data_request_by_stop_processor.TIME_PERIODS", ["AM PEAK"]),
        patch("data_request_by_stop_processor.AGGREGATE_ROUTES_TOGETHER", True),
        patch("data_request_by_stop_processor.APPLY_ROUNDING", True),
        patch("data_request_by_stop_processor.AGGREGATE_BIN_RANGES", False),
        patch("data_request_by_stop_processor.EXPORT_ROUTE_LEVEL_ANALYSIS", False),
    ):
        mock_read.return_value = fixture_df.copy()
        target.main()
        _, kwargs = mock_write.call_args

    assert kwargs["route_analysis"] is None
