import hashlib
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


def test_write_run_log_omits_snapshot_when_no_stop_ids_file(tmp_path: Path) -> None:
    """No STOP_IDS_FILE SNAPSHOT section when stop_ids_file is not given."""
    output_file = tmp_path / "output.xlsx"
    fake_source = "# === BEGIN CONFIG ===\nKEY = 1\n# === END CONFIG ===\n"
    with patch(
        "data_request_by_stop_processor._resolve_script_source",
        return_value=(fake_source, "<test>"),
    ):
        target.write_run_log(output_file)

    content = (tmp_path / "output_runlog.txt").read_text(encoding="utf-8")
    assert "STOP_IDS_FILE SNAPSHOT" not in content


def test_write_run_log_includes_stop_ids_file_snapshot(tmp_path: Path) -> None:
    """The snapshot section pins the file's hash and the resolved ID list."""
    output_file = tmp_path / "output.xlsx"
    stop_ids_file = tmp_path / "stop_ids.txt"
    stop_ids_file.write_text("1001\n3003\n", encoding="utf-8")
    expected_hash = hashlib.sha256(stop_ids_file.read_bytes()).hexdigest()

    fake_source = "# === BEGIN CONFIG ===\nKEY = 1\n# === END CONFIG ===\n"
    with patch(
        "data_request_by_stop_processor._resolve_script_source",
        return_value=(fake_source, "<test>"),
    ):
        result = target.write_run_log(
            output_file, stop_ids_file=stop_ids_file, effective_stop_ids=[1001, 3003]
        )

    assert result is True
    content = (tmp_path / "output_runlog.txt").read_text(encoding="utf-8")
    assert "STOP_IDS_FILE SNAPSHOT" in content
    assert str(stop_ids_file) in content
    assert expected_hash in content
    assert "Resolved STOP_IDS (n=2): 1001, 3003" in content


def test_write_run_log_snapshot_handles_unreadable_file(tmp_path: Path) -> None:
    """An unreadable stop_ids_file logs a placeholder instead of raising."""
    output_file = tmp_path / "output.xlsx"
    missing_file = tmp_path / "does_not_exist.txt"

    fake_source = "# === BEGIN CONFIG ===\nKEY = 1\n# === END CONFIG ===\n"
    with patch(
        "data_request_by_stop_processor._resolve_script_source",
        return_value=(fake_source, "<test>"),
    ):
        result = target.write_run_log(
            output_file, stop_ids_file=missing_file, effective_stop_ids=[1001]
        )

    assert result is True
    content = (tmp_path / "output_runlog.txt").read_text(encoding="utf-8")
    assert "<unreadable at log time>" in content


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
        assert target.main() == 0

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
        assert target.main() == 0

        _, kwargs = mock_write.call_args
        route_analysis = kwargs["route_analysis"]

    assert route_analysis is not None
    # Both stops on route 30 must be present even though STOP_IDS filtered to 4004.
    assert set(route_analysis["STOP_ID"].tolist()) == {4004, 4005}
    assert bool(route_analysis.loc[route_analysis["STOP_ID"] == 4004, "In Filter"].iloc[0]) is True
    assert bool(route_analysis.loc[route_analysis["STOP_ID"] == 4005, "In Filter"].iloc[0]) is False


def test_main_scopes_route_analysis_to_routes_serving_filtered_stops() -> None:
    """STOP_IDS without ROUTES should still limit Route Analysis to relevant routes."""
    fixture_df = pd.read_csv(FIXTURE_PATH)

    with (
        patch("data_request_by_stop_processor.read_excel_file") as mock_read,
        patch("data_request_by_stop_processor.write_to_excel") as mock_write,
        patch("data_request_by_stop_processor.write_run_log", return_value=True),
        patch("data_request_by_stop_processor.INPUT_FILE_PATH", Path("dummy_input.xlsx")),
        patch("data_request_by_stop_processor.OUTPUT_DIR", Path("dummy_output")),
        patch("data_request_by_stop_processor.ROUTES", []),
        patch("data_request_by_stop_processor.ROUTES_EXCLUDE", []),
        # Stop 4004 belongs to route 30 only. The fixture contains many other
        # routes that don't serve this stop; those must not appear in the sheet.
        patch("data_request_by_stop_processor.STOP_IDS", [4004]),
        patch("data_request_by_stop_processor.TIME_PERIODS", ["AM PEAK", "PM PEAK"]),
        patch("data_request_by_stop_processor.AGGREGATE_ROUTES_TOGETHER", True),
        patch("data_request_by_stop_processor.APPLY_ROUNDING", True),
        patch("data_request_by_stop_processor.AGGREGATE_BIN_RANGES", False),
        patch("data_request_by_stop_processor.EXPORT_ROUTE_LEVEL_ANALYSIS", True),
    ):
        mock_read.return_value = fixture_df.copy()
        assert target.main() == 0
        _, kwargs = mock_write.call_args
        route_analysis = kwargs["route_analysis"]

    assert route_analysis is not None
    # Only route 30 serves stop 4004, so no other route should appear.
    assert set(route_analysis["ROUTE_NAME"].unique()) == {"30"}
    # Both stops on route 30 are still included for comparison.
    assert set(route_analysis["STOP_ID"].tolist()) == {4004, 4005}


# =============================================================================
# bin_ridership_value
# =============================================================================


def test_bin_ridership_value_first_bucket() -> None:
    """Values strictly below 5 land in '0-4.9'."""
    assert target.bin_ridership_value(0) == "0-4.9"
    assert target.bin_ridership_value(4.9) == "0-4.9"
    assert target.bin_ridership_value(0.001) == "0-4.9"


def test_bin_ridership_value_second_bucket() -> None:
    """Values in [5, 25) land in '5-24.9'."""
    assert target.bin_ridership_value(5.0) == "5-24.9"
    assert target.bin_ridership_value(24.9) == "5-24.9"
    assert target.bin_ridership_value(10) == "5-24.9"


def test_bin_ridership_value_third_bucket() -> None:
    """Values in [25, 50) land in '25-49.9'."""
    assert target.bin_ridership_value(25.0) == "25-49.9"
    assert target.bin_ridership_value(49.9) == "25-49.9"


def test_bin_ridership_value_top_bucket() -> None:
    """Values >= 50 land in '50 or more'."""
    assert target.bin_ridership_value(50.0) == "50 or more"
    assert target.bin_ridership_value(100) == "50 or more"
    assert target.bin_ridership_value(9999) == "50 or more"


# =============================================================================
# aggregate_by_stop
# =============================================================================


def test_aggregate_by_stop_routes_together_sums_and_labels() -> None:
    """With aggregate_routes_together=True, multi-route stops are summed and ROUTES is set."""
    df = pd.read_csv(FIXTURE_PATH)
    # Stop 1001 has routes 10A (12.4/3.2) and 10B (18.1/6.7) in AM PEAK
    am_df = df[df["TIME_PERIOD"] == "AM PEAK"]
    result = target.aggregate_by_stop(am_df, aggregate_routes_together=True)

    stop_1001 = result[result["STOP_ID"] == 1001].iloc[0]
    assert stop_1001["BOARD_ALL_TOTAL"] == pytest.approx(30.5)
    assert stop_1001["ALIGHT_ALL_TOTAL"] == pytest.approx(9.9)
    assert stop_1001["ROUTES"] == "10A, 10B"
    assert "ROUTE_NAME" not in result.columns


def test_aggregate_by_stop_routes_together_single_route() -> None:
    """A stop served by exactly one route still gets a ROUTES string, not a list."""
    df = pd.read_csv(FIXTURE_PATH)
    am_df = df[df["TIME_PERIOD"] == "AM PEAK"]
    result = target.aggregate_by_stop(am_df, aggregate_routes_together=True)

    # Stop 2002 is only served by route 10A in AM PEAK
    stop_2002 = result[result["STOP_ID"] == 2002].iloc[0]
    assert stop_2002["ROUTES"] == "10A"


def test_aggregate_by_stop_routes_separate() -> None:
    """With aggregate_routes_together=False, routes produce separate rows."""
    df = pd.read_csv(FIXTURE_PATH)
    am_df = df[df["TIME_PERIOD"] == "AM PEAK"]
    result = target.aggregate_by_stop(am_df, aggregate_routes_together=False)

    assert "ROUTE_NAME" in result.columns
    assert "ROUTES" not in result.columns

    # Stop 1001 has two routes in AM PEAK → two rows
    stop_1001_rows = result[result["STOP_ID"] == 1001]
    assert len(stop_1001_rows) == 2

    row_10a = stop_1001_rows[stop_1001_rows["ROUTE_NAME"] == "10A"].iloc[0]
    assert row_10a["BOARD_ALL_TOTAL"] == pytest.approx(12.4)
    assert row_10a["ALIGHT_ALL_TOTAL"] == pytest.approx(3.2)


# =============================================================================
# log_missing_stop_ids
# =============================================================================


def test_log_missing_stop_ids_empty_request() -> None:
    """Empty requested list is a no-op — warning is never issued."""
    with patch("logging.warning") as mock_warn:
        target.log_missing_stop_ids([], [1001, 2002])
    mock_warn.assert_not_called()


def test_log_missing_stop_ids_all_present(caplog: pytest.LogCaptureFixture) -> None:
    """All requested IDs present → info-level confirmation, no warning."""
    import logging as _logging

    with caplog.at_level(_logging.INFO):
        target.log_missing_stop_ids([1001, 2002], [1001, 2002, 3003])
    assert "All requested STOP_IDs are present" in caplog.text


def test_log_missing_stop_ids_some_missing() -> None:
    """Missing IDs trigger a warning that names them."""
    with patch("logging.warning") as mock_warn:
        target.log_missing_stop_ids([1001, 9999], [1001])
    mock_warn.assert_called_once()
    # The sorted missing list is passed as the last positional arg
    assert 9999 in mock_warn.call_args[0][-1]


def test_log_missing_stop_ids_invalid_present_ids() -> None:
    """Non-coercible present_ids raise TypeError."""
    with pytest.raises(TypeError, match="Unable to evaluate present_ids"):
        target.log_missing_stop_ids([1001], ["not-a-number"])


# =============================================================================
# build_selection_summary
# =============================================================================


def test_build_selection_summary_unfiltered() -> None:
    """When filtered equals full, all four percentage rows are 100."""
    df = pd.read_csv(FIXTURE_PATH)
    result = target.build_selection_summary(df, df)
    pcts = result.set_index("Metric")["Percent"]
    assert pcts["Stops (unique STOP_ID)"] == pytest.approx(100.0)
    assert pcts["Boardings (BOARD_ALL sum)"] == pytest.approx(100.0)
    assert pcts["Alightings (ALIGHT_ALL sum)"] == pytest.approx(100.0)
    assert pcts["Total Ridership (Board + Alight)"] == pytest.approx(100.0)


def test_build_selection_summary_subset() -> None:
    """Filtering to a single route produces correct stop count and boardings."""
    df = pd.read_csv(FIXTURE_PATH)
    # Route 10A: stops 1001 and 2002; boards = 12.4+22+30+4.6+7.3+9.8 = 86.1
    filtered = df[df["ROUTE_NAME"] == "10A"]
    result = target.build_selection_summary(df, filtered)
    row = result.set_index("Metric")

    assert row.loc["Stops (unique STOP_ID)", "Selected"] == 2
    assert row.loc["Stops (unique STOP_ID)", "Total"] == 10
    assert row.loc["Stops (unique STOP_ID)", "Percent"] == pytest.approx(20.0)
    assert row.loc["Boardings (BOARD_ALL sum)", "Selected"] == pytest.approx(86.1)
    assert row.loc["Stops (unique STOP_ID)", "Percent"] < 100.0


def test_build_selection_summary_empty_selection() -> None:
    """An empty filtered DataFrame yields zeros and 0% for all metrics."""
    df = pd.read_csv(FIXTURE_PATH)
    empty = df[df["ROUTE_NAME"] == "NONEXISTENT"]
    result = target.build_selection_summary(df, empty)
    row = result.set_index("Metric")

    assert row.loc["Stops (unique STOP_ID)", "Selected"] == 0
    assert row.loc["Stops (unique STOP_ID)", "Percent"] == pytest.approx(0.0)
    assert row.loc["Boardings (BOARD_ALL sum)", "Percent"] == pytest.approx(0.0)
    assert row.loc["Total Ridership (Board + Alight)", "Percent"] == pytest.approx(0.0)


# =============================================================================
# filter_data
# =============================================================================


def test_filter_data_no_filters_returns_all() -> None:
    """No filters → identical row count, no mutation."""
    df = pd.read_csv(FIXTURE_PATH)
    result = target.filter_data(df)
    assert len(result) == len(df)


def test_filter_data_routes_keep() -> None:
    """Routes keep-list retains only rows with a matching ROUTE_NAME."""
    df = pd.read_csv(FIXTURE_PATH)
    result = target.filter_data(df, routes=["30"])
    assert set(result["ROUTE_NAME"].unique()) == {"30"}
    assert set(result["STOP_ID"].unique()) == {4004, 4005}


def test_filter_data_routes_exclude() -> None:
    """routes_exclude removes matching routes and leaves others intact."""
    df = pd.read_csv(FIXTURE_PATH)
    result = target.filter_data(df, routes_exclude=["10A", "10B"])
    assert "10A" not in result["ROUTE_NAME"].to_numpy()
    assert "10B" not in result["ROUTE_NAME"].to_numpy()
    assert "20X" in result["ROUTE_NAME"].to_numpy()


def test_filter_data_stop_ids() -> None:
    """stop_ids keep-list retains only rows with a matching STOP_ID."""
    df = pd.read_csv(FIXTURE_PATH)
    result = target.filter_data(df, stop_ids=[1001, 3003])
    assert set(result["STOP_ID"].unique()) == {1001, 3003}


def test_filter_data_routes_then_stop_ids_intersection() -> None:
    """Routes filter applies before stop_ids; result is their intersection."""
    df = pd.read_csv(FIXTURE_PATH)
    # Route 10A only serves stops 1001 and 2002; stop 4004 is on route 30 only
    result = target.filter_data(df, routes=["10A"], stop_ids=[4004])
    assert result.empty


def test_filter_data_routes_exclude_before_stop_ids() -> None:
    """routes_exclude removes a route before stop_ids can match its stops."""
    df = pd.read_csv(FIXTURE_PATH)
    # Route 30 is excluded; stop 4004 belongs only to route 30 → no rows survive
    result = target.filter_data(df, routes_exclude=["30"], stop_ids=[4004])
    assert result.empty


# =============================================================================
# load_stop_ids_from_file / resolve_stop_ids
# =============================================================================

STOP_IDS_FIXTURE_PATH = Path("tests/fixtures/stop_ids.txt")


def test_load_stop_ids_from_file_parses_fixture() -> None:
    """The bundled stop_ids.txt fixture parses to the expected unique IDs in order."""
    result = target.load_stop_ids_from_file(STOP_IDS_FIXTURE_PATH)
    assert result == [1001, 3003, 4004, 5006, 7008]


def test_load_stop_ids_from_file_missing_file_exits() -> None:
    """A nonexistent STOP_IDS_FILE path raises SystemExit."""
    with pytest.raises(SystemExit):
        target.load_stop_ids_from_file(Path("tests/fixtures/does_not_exist.txt"))


def test_load_stop_ids_from_file_skips_bad_tokens(tmp_path: Path) -> None:
    """Non-integer tokens are skipped (with a warning) rather than aborting."""
    stop_ids_file = tmp_path / "stop_ids.txt"
    stop_ids_file.write_text("1001\nSTOP_ID\n2002 abc\n", encoding="utf-8")

    with patch("logging.warning") as mock_warn:
        result = target.load_stop_ids_from_file(stop_ids_file)

    assert result == [1001, 2002]
    mock_warn.assert_called_once()
    assert "STOP_ID" in mock_warn.call_args[0][-1]
    assert "abc" in mock_warn.call_args[0][-1]


def test_load_stop_ids_from_file_no_valid_ids_exits(tmp_path: Path) -> None:
    """A file with no parseable integers raises SystemExit."""
    stop_ids_file = tmp_path / "stop_ids.txt"
    stop_ids_file.write_text("# just a comment\nnot_a_number\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        target.load_stop_ids_from_file(stop_ids_file)


def test_resolve_stop_ids_file_takes_precedence() -> None:
    """STOP_IDS_FILE wins over the inline STOP_IDS list when both are set."""
    with patch("logging.warning") as mock_warn:
        result = target.resolve_stop_ids([9999], STOP_IDS_FIXTURE_PATH)
    assert result == [1001, 3003, 4004, 5006, 7008]
    mock_warn.assert_called_once()
    assert "Both STOP_IDS and STOP_IDS_FILE" in mock_warn.call_args[0][0]


def test_resolve_stop_ids_no_file_uses_inline() -> None:
    """With no STOP_IDS_FILE, the inline STOP_IDS list passes through unchanged."""
    result = target.resolve_stop_ids([1001, 2002], None)
    assert result == [1001, 2002]


def test_main_filters_using_stop_ids_file() -> None:
    """main() filters by STOP_IDS_FILE contents when STOP_IDS_FILE is set."""
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
        patch("data_request_by_stop_processor.STOP_IDS_FILE", STOP_IDS_FIXTURE_PATH),
        patch("data_request_by_stop_processor.TIME_PERIODS", ["AM PEAK", "PM PEAK"]),
        patch("data_request_by_stop_processor.AGGREGATE_ROUTES_TOGETHER", True),
        patch("data_request_by_stop_processor.APPLY_ROUNDING", True),
        patch("data_request_by_stop_processor.AGGREGATE_BIN_RANGES", False),
        patch("data_request_by_stop_processor.EXPORT_ROUTE_LEVEL_ANALYSIS", False),
    ):
        mock_read.return_value = fixture_df.copy()
        assert target.main() == 0

        args, _ = mock_write.call_args
        filtered_data = args[1]

    # Fixture lists stop IDs 1001, 3003, 4004, 5006, 7008 — no others should remain.
    assert set(filtered_data["STOP_ID"].unique()) == {1001, 3003, 4004, 5006, 7008}


def test_main_passes_stop_ids_file_snapshot_info_to_run_log() -> None:
    """main() forwards STOP_IDS_FILE and the resolved IDs to write_run_log."""
    fixture_df = pd.read_csv(FIXTURE_PATH)

    with (
        patch("data_request_by_stop_processor.read_excel_file") as mock_read,
        patch("data_request_by_stop_processor.write_to_excel"),
        patch("data_request_by_stop_processor.write_run_log", return_value=True) as mock_log,
        patch("data_request_by_stop_processor.INPUT_FILE_PATH", Path("dummy_input.xlsx")),
        patch("data_request_by_stop_processor.OUTPUT_DIR", Path("dummy_output")),
        patch("data_request_by_stop_processor.ROUTES", []),
        patch("data_request_by_stop_processor.ROUTES_EXCLUDE", []),
        patch("data_request_by_stop_processor.STOP_IDS", []),
        patch("data_request_by_stop_processor.STOP_IDS_FILE", STOP_IDS_FIXTURE_PATH),
        patch("data_request_by_stop_processor.TIME_PERIODS", ["AM PEAK", "PM PEAK"]),
        patch("data_request_by_stop_processor.AGGREGATE_ROUTES_TOGETHER", True),
        patch("data_request_by_stop_processor.APPLY_ROUNDING", True),
        patch("data_request_by_stop_processor.AGGREGATE_BIN_RANGES", False),
        patch("data_request_by_stop_processor.EXPORT_ROUTE_LEVEL_ANALYSIS", False),
    ):
        mock_read.return_value = fixture_df.copy()
        assert target.main() == 0

        _, kwargs = mock_log.call_args

    assert kwargs["stop_ids_file"] == STOP_IDS_FIXTURE_PATH
    assert kwargs["effective_stop_ids"] == [1001, 3003, 4004, 5006, 7008]


# =============================================================================
# verify_required_columns
# =============================================================================


def test_verify_required_columns_all_present() -> None:
    """No exception when every required column exists."""
    df = pd.DataFrame(columns=["A", "B", "C"])
    target.verify_required_columns(df, ["A", "B"])  # should not raise


def test_verify_required_columns_missing_raises_system_exit() -> None:
    """SystemExit when a required column is absent."""
    df = pd.DataFrame(columns=["A"])
    with pytest.raises(SystemExit):
        target.verify_required_columns(df, ["A", "MISSING"])


# =============================================================================
# enrich_with_gtfs
# =============================================================================


def _make_gtfs_stops() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["1001", "2002"],
            "stop_code": ["S1", "S2"],
            "stop_name": ["Stop One", "Stop Two"],
            "stop_lat": [38.9, 38.8],
            "stop_lon": [-77.1, -77.2],
        }
    )


def test_enrich_with_gtfs_stop_id_join() -> None:
    """Correct lat/lon is appended when joining on stop_id."""
    ridership = pd.DataFrame({"STOP_ID": [1001, 2002], "BOARD_ALL": [10, 20]})
    gtfs = _make_gtfs_stops()
    result = target.enrich_with_gtfs(ridership, gtfs, "stop_id")

    assert result.loc[result["STOP_ID"] == 1001, "LATITUDE"].iloc[0] == pytest.approx(38.9)
    assert result.loc[result["STOP_ID"] == 2002, "LONGITUDE"].iloc[0] == pytest.approx(-77.2)
    assert "_join_key" not in result.columns


def test_enrich_with_gtfs_no_match_produces_nan() -> None:
    """A stop_id absent from GTFS gets NaN coordinates."""
    ridership = pd.DataFrame({"STOP_ID": [9999], "BOARD_ALL": [5]})
    gtfs = _make_gtfs_stops()
    result = target.enrich_with_gtfs(ridership, gtfs, "stop_id")
    assert pd.isna(result.loc[0, "LATITUDE"])
    assert pd.isna(result.loc[0, "LONGITUDE"])


def test_enrich_with_gtfs_stop_code_join() -> None:
    """Joining on stop_code uses the stop_code column, not stop_id."""
    ridership = pd.DataFrame({"STOP_ID": ["S1", "S2"], "BOARD_ALL": [10, 20]})
    gtfs = _make_gtfs_stops()
    result = target.enrich_with_gtfs(ridership, gtfs, "stop_code")
    assert result.loc[result["STOP_ID"] == "S1", "LATITUDE"].iloc[0] == pytest.approx(38.9)


def test_enrich_with_gtfs_invalid_join_key() -> None:
    """ValueError for a join_key that is neither 'stop_id' nor 'stop_code'."""
    ridership = pd.DataFrame({"STOP_ID": [1001]})
    gtfs = _make_gtfs_stops()
    with pytest.raises(ValueError, match="join_key must be"):
        target.enrich_with_gtfs(ridership, gtfs, "bad_key")


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
        assert target.main() == 0
        _, kwargs = mock_write.call_args

    assert kwargs["route_analysis"] is None
