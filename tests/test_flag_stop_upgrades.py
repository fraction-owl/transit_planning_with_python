import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

script_dir = Path("scripts/facilities_tools").resolve()
sys.path.append(str(script_dir))

import flag_stop_upgrades as target  # noqa: E402

FIXTURE_PATH = Path("tests/fixtures/stop_usage_by_stop_id_sample.csv")


# =============================================================================
# Helpers
# =============================================================================


def _make_ridership_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STOP_ID": ["1257", "2169", "9839"],
            "XBOARDINGS": ["4", "5", "30"],
        }
    )


def _make_flaggable_df() -> pd.DataFrame:
    """Covers all threshold boundary cases for compute_flags."""
    return pd.DataFrame(
        {
            "STOP_ID": ["A", "B", "C", "D", "E"],
            "XBOARDINGS": [0, 1, 10, 25, 100],
            "SHELTER": ["N", "N", "N", "N", "Y"],
            "BENCH": ["N", "N", "N", "N", "Y"],
            "TRASHCAN": ["N", "N", "N", "N", "Y"],
            "PAD": ["N", "N", "N", "N", "Y"],
        }
    )


def _make_duplicate_stop_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STOP_ID": ["1257", "1257", "2169"],
            "XBOARDINGS": [5, 8, 12],
            "SHELTER": ["N", "Y", "N"],
            "BENCH": ["N", "N", "Y"],
            "TRASHCAN": ["N", "N", "N"],
            "PAD": ["Y", "N", "Y"],
        }
    )


def _make_ridership_merge_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STOP_ID": ["1257", "2169", "9999"],
            "XBOARDINGS": ["4", "12", "5"],
        }
    )


def _make_amenity_merge_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_code": ["1257", "2169"],
            "SHELTER": ["Y", "N"],
            "BENCH": ["N", "Y"],
            "TRASHCAN": ["N", "N"],
            "PAD": ["Y", "N"],
        }
    )


# =============================================================================
# _standardise_yn
# =============================================================================


def test_standardise_yn_uppercases_lowercase_values() -> None:
    """Lowercase 'y' and 'n' are converted to uppercase."""
    s = pd.Series(["y", "Y", "n", "N"])
    result = target._standardise_yn(s)
    assert list(result) == ["Y", "Y", "N", "N"]


def test_standardise_yn_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped before uppercasing."""
    s = pd.Series([" y ", " N ", "  Y  "])
    result = target._standardise_yn(s)
    assert list(result) == ["Y", "N", "Y"]


def test_standardise_yn_fills_na_with_n() -> None:
    """NaN and None values are treated as 'N'."""
    s = pd.Series([None, float("nan"), "Y"])
    result = target._standardise_yn(s)
    assert result.iloc[0] == "N"
    assert result.iloc[1] == "N"
    assert result.iloc[2] == "Y"


def test_standardise_yn_empty_series() -> None:
    """An empty series passes through without error."""
    s = pd.Series([], dtype=str)
    result = target._standardise_yn(s)
    assert len(result) == 0


# =============================================================================
# _prepare_amenity_columns
# =============================================================================


def test_prepare_amenity_columns_adds_missing_columns() -> None:
    """Missing amenity columns are created and defaulted to 'N'."""
    df = _make_ridership_df()
    result = target._prepare_amenity_columns(df)
    for col in ["SHELTER", "BENCH", "TRASHCAN", "PAD"]:
        assert col in result.columns
        assert (result[col] == "N").all()


def test_prepare_amenity_columns_standardises_existing() -> None:
    """An existing amenity column with mixed case/whitespace is normalised."""
    df = _make_ridership_df()
    df["SHELTER"] = [" y ", "n", "Y"]
    result = target._prepare_amenity_columns(df)
    assert list(result["SHELTER"]) == ["Y", "N", "Y"]


def test_prepare_amenity_columns_preserves_other_columns() -> None:
    """Non-amenity columns are not dropped."""
    df = _make_ridership_df()
    result = target._prepare_amenity_columns(df)
    assert "STOP_ID" in result.columns
    assert "XBOARDINGS" in result.columns


# =============================================================================
# _convert_ridership
# =============================================================================


def test_convert_ridership_float_strings_truncated_to_int() -> None:
    """Float strings are coerced to int (truncated, not rounded)."""
    df = pd.DataFrame({"XBOARDINGS": ["4.034", "12.354", "3.712"]})
    result = target._convert_ridership(df)
    assert result["XBOARDINGS"].tolist() == [4, 12, 3]
    assert result["XBOARDINGS"].dtype == int


def test_convert_ridership_non_numeric_becomes_zero() -> None:
    """Non-numeric and null values are coerced to 0."""
    df = pd.DataFrame({"XBOARDINGS": ["abc", None, "10"]})
    result = target._convert_ridership(df)
    assert result["XBOARDINGS"].tolist() == [0, 0, 10]


def test_convert_ridership_fixture_values_are_non_negative_ints() -> None:
    """All fixture ridership values convert to non-negative integers."""
    df = pd.read_csv(FIXTURE_PATH, dtype=str)
    result = target._convert_ridership(df)
    assert (result["XBOARDINGS"] >= 0).all()
    assert result["XBOARDINGS"].dtype == int


# =============================================================================
# _needs_aggregation
# =============================================================================


def test_needs_aggregation_forced_true_regardless_of_data() -> None:
    """Returns True when AGGREGATE_BY_STOP is True, even without duplicates."""
    df = pd.DataFrame({"STOP_ID": ["1", "2", "3"]})
    with patch.object(target, "AGGREGATE_BY_STOP", True):
        assert target._needs_aggregation(df) is True


def test_needs_aggregation_forced_false_ignores_duplicates() -> None:
    """Returns False when AGGREGATE_BY_STOP is False, even with duplicates."""
    df = pd.DataFrame({"STOP_ID": ["1", "1"]})
    with patch.object(target, "AGGREGATE_BY_STOP", False):
        assert target._needs_aggregation(df) is False


def test_needs_aggregation_auto_no_duplicates_returns_false() -> None:
    """'auto' returns False when all STOP_IDs are unique."""
    df = pd.DataFrame({"STOP_ID": ["1257", "2169", "9839"]})
    with patch.object(target, "AGGREGATE_BY_STOP", "auto"):
        assert not target._needs_aggregation(df)


def test_needs_aggregation_auto_with_duplicates_returns_true() -> None:
    """'auto' returns True when duplicate STOP_IDs are present."""
    df = pd.DataFrame({"STOP_ID": ["1257", "1257", "9839"]})
    with patch.object(target, "AGGREGATE_BY_STOP", "auto"):
        assert target._needs_aggregation(df)


def test_needs_aggregation_fixture_has_no_duplicates() -> None:
    """The sample fixture has unique STOP_IDs, so 'auto' returns False."""
    df = pd.read_csv(FIXTURE_PATH, dtype=str)
    with patch.object(target, "AGGREGATE_BY_STOP", "auto"):
        assert not target._needs_aggregation(df)


# =============================================================================
# _aggregate_by_stop
# =============================================================================


def test_aggregate_by_stop_sums_ridership_for_duplicate_stops() -> None:
    """Ridership is summed across rows sharing the same STOP_ID."""
    df = _make_duplicate_stop_df()
    result = target._aggregate_by_stop(df)
    stop_1257 = result[result["STOP_ID"] == "1257"].iloc[0]
    assert stop_1257["XBOARDINGS"] == 13


def test_aggregate_by_stop_ors_amenities() -> None:
    """Amenity value is 'Y' if any row for that stop has 'Y'."""
    df = _make_duplicate_stop_df()
    result = target._aggregate_by_stop(df)
    stop_1257 = result[result["STOP_ID"] == "1257"].iloc[0]
    assert stop_1257["SHELTER"] == "Y"  # N + Y → Y
    assert stop_1257["BENCH"] == "N"  # N + N → N
    assert stop_1257["PAD"] == "Y"  # Y + N → Y


def test_aggregate_by_stop_unique_stop_passes_through() -> None:
    """Stops without duplicates are unchanged after aggregation."""
    df = _make_duplicate_stop_df()
    result = target._aggregate_by_stop(df)
    stop_2169 = result[result["STOP_ID"] == "2169"].iloc[0]
    assert stop_2169["XBOARDINGS"] == 12
    assert stop_2169["BENCH"] == "Y"


def test_aggregate_by_stop_one_row_per_stop_id() -> None:
    """Output contains exactly one row per unique STOP_ID."""
    df = _make_duplicate_stop_df()
    result = target._aggregate_by_stop(df)
    assert len(result) == 2


# =============================================================================
# _compute_flags
# =============================================================================


def test_compute_flags_adds_all_flag_and_summary_columns() -> None:
    """FLAG_* columns and NEEDS_IMPROVEMENT are present after compute_flags."""
    df = _make_flaggable_df()
    result, flag_cols = target._compute_flags(df)
    assert set(flag_cols) == {"FLAG_SHELTER", "FLAG_BENCH", "FLAG_TRASHCAN", "FLAG_PAD"}
    for col in flag_cols:
        assert col in result.columns
    assert "NEEDS_IMPROVEMENT" in result.columns


def test_compute_flags_pad_threshold_is_one() -> None:
    """FLAG_PAD is False at 0 boardings and True at 1+ boardings (when no pad)."""
    df = _make_flaggable_df()
    result, _ = target._compute_flags(df)
    assert not result.loc[result["STOP_ID"] == "A", "FLAG_PAD"].iloc[0]  # 0 boardings
    assert result.loc[result["STOP_ID"] == "B", "FLAG_PAD"].iloc[0]  # 1 boarding


def test_compute_flags_bench_threshold_is_ten() -> None:
    """FLAG_BENCH is True at exactly 10 boardings and False below."""
    df = _make_flaggable_df()
    result, _ = target._compute_flags(df)
    assert not result.loc[result["STOP_ID"] == "B", "FLAG_BENCH"].iloc[0]  # 1 boarding
    assert result.loc[result["STOP_ID"] == "C", "FLAG_BENCH"].iloc[0]  # 10 boardings


def test_compute_flags_shelter_threshold_is_25() -> None:
    """FLAG_SHELTER is True at exactly 25 boardings and False below."""
    df = _make_flaggable_df()
    result, _ = target._compute_flags(df)
    assert not result.loc[result["STOP_ID"] == "C", "FLAG_SHELTER"].iloc[0]  # 10 boardings
    assert result.loc[result["STOP_ID"] == "D", "FLAG_SHELTER"].iloc[0]  # 25 boardings


def test_compute_flags_amenity_present_suppresses_flag() -> None:
    """A stop with all amenities marked 'Y' is never flagged, even at high ridership."""
    df = pd.DataFrame(
        {
            "STOP_ID": ["X"],
            "XBOARDINGS": [100],
            "SHELTER": ["Y"],
            "BENCH": ["Y"],
            "TRASHCAN": ["Y"],
            "PAD": ["Y"],
        }
    )
    result, _ = target._compute_flags(df)
    assert not result["NEEDS_IMPROVEMENT"].iloc[0]


def test_compute_flags_needs_improvement_is_any_flag() -> None:
    """NEEDS_IMPROVEMENT is True iff at least one FLAG_* is True."""
    df = _make_flaggable_df()
    result, _ = target._compute_flags(df)
    assert not result.loc[result["STOP_ID"] == "A", "NEEDS_IMPROVEMENT"].iloc[0]
    assert result.loc[result["STOP_ID"] == "B", "NEEDS_IMPROVEMENT"].iloc[0]


# =============================================================================
# _is_placeholder_path
# =============================================================================


def test_is_placeholder_path_default_ridership_path() -> None:
    """The default RIDERSHIP_XLSX placeholder is detected."""
    p = Path(r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).xlsx")
    assert target._is_placeholder_path(p) is True


def test_is_placeholder_path_default_output_folder() -> None:
    """The default OUTPUT_FOLDER placeholder is detected."""
    p = Path(r"Your\Folder\Path\To\Output")
    assert target._is_placeholder_path(p) is True


def test_is_placeholder_path_real_path_returns_false() -> None:
    """A real, non-placeholder path is not flagged."""
    p = Path("/home/user/data/stops.xlsx")
    assert target._is_placeholder_path(p) is False


def test_is_placeholder_path_detection_is_case_insensitive() -> None:
    """Placeholder detection ignores case."""
    p = Path("YOUR/FILE/PATH/stops.xlsx")
    assert target._is_placeholder_path(p) is True


# =============================================================================
# _merge_ridership_and_amenities
# =============================================================================


def test_merge_joins_amenity_columns_onto_matching_stop_ids() -> None:
    """Amenity values from the amenity df are present on matched STOP_IDs."""
    rider = _make_ridership_merge_df()
    amen = _make_amenity_merge_df()
    result = target._merge_ridership_and_amenities(rider, amen)
    row = result[result["STOP_ID"] == "1257"].iloc[0]
    assert row["SHELTER"] == "Y"
    assert row["BENCH"] == "N"
    assert row["PAD"] == "Y"


def test_merge_unmatched_stop_gets_nan_amenities() -> None:
    """A STOP_ID absent from the amenity file receives NaN for amenity columns."""
    rider = _make_ridership_merge_df()
    amen = _make_amenity_merge_df()
    result = target._merge_ridership_and_amenities(rider, amen)
    row = result[result["STOP_ID"] == "9999"].iloc[0]
    assert pd.isna(row["SHELTER"])


def test_merge_drops_join_key_column() -> None:
    """The amenity join key (stop_code) is not retained in the output."""
    rider = _make_ridership_merge_df()
    amen = _make_amenity_merge_df()
    result = target._merge_ridership_and_amenities(rider, amen)
    assert "stop_code" not in result.columns


def test_merge_all_ridership_rows_retained() -> None:
    """All ridership rows are kept even without an amenity match (left join)."""
    rider = _make_ridership_merge_df()
    amen = _make_amenity_merge_df()
    result = target._merge_ridership_and_amenities(rider, amen)
    assert len(result) == 3


# =============================================================================
# _write_txt_log
# =============================================================================


def test_write_txt_log_creates_file(tmp_path: Path) -> None:
    """The text log file is created at the specified path."""
    df = _make_flaggable_df()
    df, flag_cols = target._compute_flags(df)
    out = tmp_path / "log.txt"
    target._write_txt_log(df, flag_cols, out)
    assert out.exists()


def test_write_txt_log_contains_amenity_category_names(tmp_path: Path) -> None:
    """The log lists each amenity category."""
    df = _make_flaggable_df()
    df, flag_cols = target._compute_flags(df)
    out = tmp_path / "log.txt"
    target._write_txt_log(df, flag_cols, out)
    content = out.read_text(encoding="utf-8")
    for name in ["Shelter", "Bench", "TrashCan", "Pad"]:
        assert name in content


def test_write_txt_log_total_count_is_accurate(tmp_path: Path) -> None:
    """The reported total flagged stop count matches the DataFrame."""
    df = _make_flaggable_df()
    df, flag_cols = target._compute_flags(df)
    expected = int(df["NEEDS_IMPROVEMENT"].sum())
    out = tmp_path / "log.txt"
    target._write_txt_log(df, flag_cols, out)
    content = out.read_text(encoding="utf-8")
    assert f"Total flagged stops: {expected}" in content


def test_write_txt_log_zero_flagged_when_all_amenities_present(tmp_path: Path) -> None:
    """When all stops have all amenities, the log reports zero flagged stops."""
    df = pd.DataFrame(
        {
            "STOP_ID": ["1", "2"],
            "XBOARDINGS": [100, 50],
            "SHELTER": ["Y", "Y"],
            "BENCH": ["Y", "Y"],
            "TRASHCAN": ["Y", "Y"],
            "PAD": ["Y", "Y"],
        }
    )
    df, flag_cols = target._compute_flags(df)
    out = tmp_path / "log.txt"
    target._write_txt_log(df, flag_cols, out)
    content = out.read_text(encoding="utf-8")
    assert "Total flagged stops: 0" in content
