"""Tests for scripts/national_data_tools/uscensus_tiger_join_gpd.py."""

from __future__ import annotations

import gzip
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from scripts.national_data_tools import uscensus_tiger_join_gpd as mod

FIXTURE_DIR = Path("tests/fixtures")
FIXTURE_ZIPS = (
    "tl_2025_11_tabblock20_sample.zip",
    "tl_2025_24_tabblock20_sample.zip",
    "tl_2025_51_tabblock20_sample.zip",
)
EXPECTED_COUNTS = (14, 40, 59)
EXPECTED_TOTAL = sum(EXPECTED_COUNTS)

# Block GEO_ID: positions [9:20] → "11001000100" (tract), [9:24] → "110010001001001" (block)
_BLOCK_GEO_ID = "1000000US110010001001001"
_TRACT_GEO_ID = "1400000US11001000100"
_COUNTY_FIPS = "11001"


# =============================================================================
# Helpers
# =============================================================================


def _census_csv(header: str, label: str, *data_rows: str) -> str:
    """Return a Census-style CSV string with a human-readable label row at index 1."""
    return "\n".join([header, label, *data_rows]) + "\n"


def _write_plain_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_gz_csv(path: Path, content: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(content)


def _write_zip_csv(zip_path: Path, csv_name: str, content: str) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(csv_name, content)


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture()
def tiger_dir(tmp_path: Path) -> Path:
    """Stage all three TIGER zip fixtures in an isolated directory."""
    staged = tmp_path / "tiger"
    staged.mkdir()
    for name in FIXTURE_ZIPS:
        shutil.copy(FIXTURE_DIR / name, staged / name)
    return staged


@pytest.fixture()
def minimal_block_files(tmp_path: Path) -> dict[str, list[str]]:
    """Minimal block-level CSV fixtures for build_joined_table."""
    pop_path = tmp_path / "P1-Data.csv"
    hh_path = tmp_path / "H9-Data.csv"
    wac_path = tmp_path / "wac_S000_JT00.csv.gz"

    _write_plain_csv(
        pop_path,
        _census_csv(
            "GEO_ID,NAME,P1_001N",
            "Geography,Geographic Area Name,!!Total:",
            f"{_BLOCK_GEO_ID},Test Block,100",
        ),
    )
    _write_plain_csv(
        hh_path,
        _census_csv("GEO_ID,H9_001N", "Geography,!!Total:", f"{_BLOCK_GEO_ID},40"),
    )
    _write_gz_csv(wac_path, "w_geocode,C000,CE01,CE02,CE03\n110010001001001,50,10,15,25\n")
    return {
        "pop_files": [str(pop_path)],
        "hh_files": [str(hh_path)],
        "jobs_files": [str(wac_path)],
    }


# =============================================================================
# _token_match
# =============================================================================


def test_token_match_single_token_present() -> None:
    assert mod._token_match("DECENNIALPL2020.P1-Data.csv", ("P1",)) is True


def test_token_match_single_token_absent() -> None:
    assert mod._token_match("DECENNIALPL2020.P1-Data.csv", ("H9",)) is False


def test_token_match_is_case_insensitive() -> None:
    assert mod._token_match("dc_wac_S000_JT00_2023.csv.gz", ("_s000_jt00_",)) is True


def test_token_match_all_tokens_must_match() -> None:
    assert mod._token_match("ACSDT5Y2024.B19001-Data.csv", ("B19001", "ACSDT5Y2024")) is True


def test_token_match_partial_tokens_fails() -> None:
    assert mod._token_match("ACSDT5Y2024.B19001-Data.csv", ("B19001", "DECENNIAL")) is False


def test_token_match_string_token_treated_as_single() -> None:
    assert mod._token_match("my_P1_file-Data.csv", "P1") is True


# =============================================================================
# discover_census_files
# =============================================================================


def test_discover_census_files_finds_data_csv(tmp_path: Path) -> None:
    (tmp_path / "DECENNIALPL2020.P1-Data.csv").write_text("x")
    result = mod.discover_census_files(tmp_path, {"POP": ("P1",)})
    assert len(result["POP"]) == 1
    assert result["POP"][0].endswith("P1-Data.csv")


def test_discover_census_files_finds_csv_gz(tmp_path: Path) -> None:
    (tmp_path / "dc_wac_S000_JT00_2023.csv.gz").write_bytes(b"")
    result = mod.discover_census_files(tmp_path, {"JOBS": ("_S000_JT00_",)})
    assert len(result["JOBS"]) == 1


def test_discover_census_files_finds_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "ACSDT5Y2024.B19001-Data.zip"
    _write_zip_csv(zip_path, "ACSDT5Y2024.B19001-Data.csv", "GEO_ID\n")
    result = mod.discover_census_files(tmp_path, {"INCOME": ("B19001",)})
    assert len(result["INCOME"]) == 1


def test_discover_census_files_ignores_unmatched_files(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("hello")
    (tmp_path / "some_other-Data.csv").write_text("x")
    result = mod.discover_census_files(tmp_path, {"POP": ("P1",)})
    assert result["POP"] == []


def test_discover_census_files_results_are_sorted(tmp_path: Path) -> None:
    for name in ["z_P1-Data.csv", "a_P1-Data.csv", "m_P1-Data.csv"]:
        (tmp_path / name).write_text("x")
    result = mod.discover_census_files(tmp_path, {"POP": ("P1",)})
    assert result["POP"] == sorted(result["POP"])


def test_discover_census_files_recurses_into_subdirs(tmp_path: Path) -> None:
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    (sub / "P1-Data.csv").write_text("x")
    result = mod.discover_census_files(tmp_path, {"POP": ("P1",)})
    assert len(result["POP"]) == 1


# =============================================================================
# _read_csv_any
# =============================================================================


def test_read_csv_any_plain_csv(tmp_path: Path) -> None:
    p = tmp_path / "test-Data.csv"
    _write_plain_csv(p, "a,b\n1,2\n3,4\n")
    df = mod._read_csv_any(p)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_csv_any_csv_gz(tmp_path: Path) -> None:
    p = tmp_path / "test.csv.gz"
    _write_gz_csv(p, "a,b\n1,2\n")
    df = mod._read_csv_any(p)
    assert len(df) == 1
    assert df["a"].iloc[0] == 1


def test_read_csv_any_zip_with_data_csv(tmp_path: Path) -> None:
    zip_path = tmp_path / "bundle.zip"
    _write_zip_csv(zip_path, "bundle-Data.csv", "x,y\n10,20\n")
    df = mod._read_csv_any(zip_path)
    assert list(df.columns) == ["x", "y"]
    assert df["x"].iloc[0] == 10


def test_read_csv_any_zip_missing_data_csv_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    _write_zip_csv(zip_path, "notes.txt", "no csv here")
    with pytest.raises(FileNotFoundError):
        mod._read_csv_any(zip_path)


# =============================================================================
# _fill_numeric_only
# =============================================================================


def test_fill_numeric_only_fills_numeric_nans() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": [float("nan"), 3.0]})
    result = mod._fill_numeric_only(df)
    assert result["a"].iloc[1] == 0
    assert result["b"].iloc[0] == 0


def test_fill_numeric_only_leaves_object_columns_untouched() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")], "name": ["Alice", None]})
    result = mod._fill_numeric_only(df)
    assert pd.isna(result["name"].iloc[1])


# =============================================================================
# _clean_name_cols
# =============================================================================


def test_clean_name_cols_strips_control_chars() -> None:
    df = pd.DataFrame({"NAME": ["foo\r\nbar", "baz\ttab"]})
    mod._clean_name_cols(df)
    assert df["NAME"].iloc[0] == "foo bar"
    assert df["NAME"].iloc[1] == "baz tab"


def test_clean_name_cols_only_touches_name_prefixed_columns() -> None:
    df = pd.DataFrame({"NAME": ["a\nb"], "OTHER": ["c\nd"]})
    mod._clean_name_cols(df)
    assert df["OTHER"].iloc[0] == "c\nd"


# =============================================================================
# _merge_on_geo_id
# =============================================================================


def test_merge_on_geo_id_outer_merges_on_geo_id() -> None:
    left = pd.DataFrame({"GEO_ID": ["A", "B"], "pop": [10, 20]})
    right = pd.DataFrame({"GEO_ID": ["B", "C"], "jobs": [5, 15]})
    result = mod._merge_on_geo_id(left, right)
    assert set(result["GEO_ID"]) == {"A", "B", "C"}
    assert result.loc[result["GEO_ID"] == "B", "pop"].iloc[0] == 20
    assert result.loc[result["GEO_ID"] == "B", "jobs"].iloc[0] == 5


def test_merge_on_geo_id_left_empty_returns_right() -> None:
    right = pd.DataFrame({"GEO_ID": ["A"], "pop": [5]})
    result = mod._merge_on_geo_id(pd.DataFrame(), right)
    assert len(result) == 1
    assert "pop" in result.columns


def test_merge_on_geo_id_right_empty_returns_left() -> None:
    left = pd.DataFrame({"GEO_ID": ["A"], "pop": [5]})
    result = mod._merge_on_geo_id(left, pd.DataFrame())
    assert len(result) == 1


def test_merge_on_geo_id_drops_duplicate_columns() -> None:
    left = pd.DataFrame({"GEO_ID": ["A"], "NAME": ["Left"], "pop": [5]})
    right = pd.DataFrame({"GEO_ID": ["A"], "NAME": ["Right"], "jobs": [10]})
    result = mod._merge_on_geo_id(left, right)
    assert result.columns.tolist().count("NAME") == 1


# =============================================================================
# _drop_unfriendly_cols
# =============================================================================


def test_drop_unfriendly_cols_removes_raw_census_codes() -> None:
    # NP001E matches ^[A-Z]{2,}\d{3,}.*
    df = pd.DataFrame({"GEO_ID": ["A"], "NP001E": [100], "total_pop": [200]})
    result = mod._drop_unfriendly_cols(df)
    assert "NP001E" not in result.columns
    assert "total_pop" in result.columns
    assert "GEO_ID" in result.columns


def test_drop_unfriendly_cols_keeps_friendly_columns() -> None:
    df = pd.DataFrame({"GEO_ID": ["A"], "low_income": [5], "perc_lep": [0.1]})
    result = mod._drop_unfriendly_cols(df)
    assert list(result.columns) == ["GEO_ID", "low_income", "perc_lep"]


# =============================================================================
# _load_and_concat
# =============================================================================


def test_load_and_concat_empty_list_returns_empty_df() -> None:
    result = mod._load_and_concat([])
    assert result.empty


def test_load_and_concat_reads_and_skips_label_row(tmp_path: Path) -> None:
    p = tmp_path / "p1-Data.csv"
    _write_plain_csv(p, "GEO_ID,NAME,P1_001N\nlabel,label,label\nA,Area A,50\n")
    result = mod._load_and_concat([str(p)], skiprows=[1])
    assert len(result) == 1
    assert result["P1_001N"].iloc[0] == 50


def test_load_and_concat_concatenates_multiple_files(tmp_path: Path) -> None:
    for i, name in enumerate(["a-Data.csv", "b-Data.csv"]):
        (tmp_path / name).write_text(f"GEO_ID,val\nGEO_{i},{i * 10}\n", encoding="utf-8")
    result = mod._load_and_concat([str(tmp_path / "a-Data.csv"), str(tmp_path / "b-Data.csv")])
    assert len(result) == 2


def test_load_and_concat_applies_column_rename(tmp_path: Path) -> None:
    p = tmp_path / "pop-Data.csv"
    _write_plain_csv(p, "GEO_ID,NAME,P1_001N\nlabel,label,label\nA,Area A,42\n")
    result = mod._load_and_concat([str(p)], skiprows=[1], rename={"P1_001N": "total_pop"})
    assert "total_pop" in result.columns
    assert result["total_pop"].iloc[0] == 42


def test_load_and_concat_dedupes_repeated_geo_id_keep_first(tmp_path: Path) -> None:
    # Two "vintages" of one table repeat a GEO_ID; only the first file's row survives.
    a = tmp_path / "ACSDT5Y2023.B19001-Data.csv"
    b = tmp_path / "ACSDT5Y2024.B19001-Data.csv"
    _write_plain_csv(a, "GEO_ID,val\n1400000US11001000100,10\n")
    _write_plain_csv(b, "GEO_ID,val\n1400000US11001000100,99\n")
    result = mod._load_and_concat([str(a), str(b)])
    assert len(result) == 1
    assert result["val"].iloc[0] == 10


def test_load_and_concat_keeps_distinct_geo_ids(tmp_path: Path) -> None:
    a = tmp_path / "a-Data.csv"
    b = tmp_path / "b-Data.csv"
    _write_plain_csv(a, "GEO_ID,val\nX,1\n")
    _write_plain_csv(b, "GEO_ID,val\nY,2\n")
    assert len(mod._load_and_concat([str(a), str(b)])) == 2


def test_load_and_concat_dedupe_key_none_disables(tmp_path: Path) -> None:
    a = tmp_path / "a-Data.csv"
    _write_plain_csv(a, "GEO_ID,val\nX,1\nX,2\n")
    assert len(mod._load_and_concat([str(a)], dedupe_key=None)) == 2


def test_load_and_concat_dedupes_on_alternate_key(tmp_path: Path) -> None:
    # LODES files key on w_geocode, not GEO_ID, so several WAC vintages must dedupe there.
    a = tmp_path / "wac_2021.csv"
    b = tmp_path / "wac_2022.csv"
    _write_plain_csv(a, "w_geocode,C000\n110010001001001,50\n")
    _write_plain_csv(b, "w_geocode,C000\n110010001001001,77\n")
    result = mod._load_and_concat([str(a), str(b)], dedupe_key="w_geocode")
    assert len(result) == 1
    assert result["C000"].iloc[0] == 50


# =============================================================================
# _ensure_fips_column_df  (DataFrame version)
# =============================================================================


def test_ensure_fips_column_df_creates_five_digit_fips() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID]})
    mod._ensure_fips_column_df(df)
    assert "FIPS" in df.columns
    assert df["FIPS"].iloc[0] == _COUNTY_FIPS


def test_ensure_fips_column_df_is_idempotent() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "FIPS": ["99999"]})
    mod._ensure_fips_column_df(df)
    assert df["FIPS"].iloc[0] == "99999"


def test_ensure_fips_column_df_uses_fallback_geo_id_column() -> None:
    df = pd.DataFrame({"GEO_ID_blk": [_BLOCK_GEO_ID]})
    mod._ensure_fips_column_df(df)
    assert df["FIPS"].iloc[0] == _COUNTY_FIPS


def test_ensure_fips_column_df_raises_when_no_geo_col() -> None:
    df = pd.DataFrame({"other": [1]})
    with pytest.raises(KeyError):
        mod._ensure_fips_column_df(df)


# =============================================================================
# _apply_fips_filter_df
# =============================================================================


def test_apply_fips_filter_df_keeps_only_matching_rows() -> None:
    df = pd.DataFrame(
        {
            "GEO_ID": [_BLOCK_GEO_ID, "1000000US240310001001001"],
            "pop": [10, 20],
        }
    )
    result = mod._apply_fips_filter_df(df, fips=[_COUNTY_FIPS])
    assert len(result) == 1
    assert result["pop"].iloc[0] == 10


def test_apply_fips_filter_df_empty_fips_returns_unchanged() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "pop": [10]})
    assert len(mod._apply_fips_filter_df(df, fips=[])) == 1


def test_apply_fips_filter_df_none_returns_unchanged() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "pop": [10]})
    assert len(mod._apply_fips_filter_df(df, fips=None)) == 1


def test_apply_fips_filter_df_zero_pads_short_fips() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "pop": [10]})
    result = mod._apply_fips_filter_df(df, fips=["11001"])
    assert len(result) == 1


# =============================================================================
# Derivation functions
# =============================================================================


def test_derive_income_computes_low_income_sum_and_percentage() -> None:
    bands = [
        "sub_10k",
        "10k_15k",
        "15k_20k",
        "20k_25k",
        "25k_30k",
        "30k_35k",
        "35k_40k",
        "40k_45k",
        "45k_50k",
        "50k_60k",
    ]
    data: dict[str, Any] = {b: [10] for b in bands}
    data["total_hh"] = [100]
    data["GEO_ID"] = [_TRACT_GEO_ID]
    result = mod._derive_income(pd.DataFrame(data))
    assert result["low_income"].iloc[0] == 100
    assert result["perc_low_income"].iloc[0] == pytest.approx(1.0)
    assert "total_hh" not in result.columns


def test_derive_ethnicity_computes_minority_sum_and_percentage() -> None:
    df = pd.DataFrame(
        {
            "GEO_ID": [_TRACT_GEO_ID],
            "total_pop": [200],
            "black": [40],
            "native": [10],
            "asian": [20],
            "pac_isl": [5],
            "other": [5],
            "multi": [10],
        }
    )
    result = mod._derive_ethnicity(df)
    assert result["minority"].iloc[0] == 90
    assert result["perc_minority"].iloc[0] == pytest.approx(0.45)
    assert "total_pop" not in result.columns


def test_derive_language_computes_lep_percentage() -> None:
    df = pd.DataFrame(
        {
            "GEO_ID": [_TRACT_GEO_ID],
            "total_lang_pop": [200],
            "spanish_engnwell": [20],
            "korean_engnwell": [10],
        }
    )
    result = mod._derive_language(df)
    assert result["all_nwell"].iloc[0] == 30
    assert result["perc_lep"].iloc[0] == pytest.approx(0.15)


def test_derive_language_zero_lang_pop_yields_zero_lep() -> None:
    df = pd.DataFrame({"GEO_ID": [_TRACT_GEO_ID], "total_lang_pop": [0], "spanish_engnwell": [0]})
    result = mod._derive_language(df)
    assert result["perc_lep"].iloc[0] == pytest.approx(0.0)


def test_derive_vehicle_computes_low_vehicle_metrics() -> None:
    df = pd.DataFrame(
        {
            "GEO_ID": [_TRACT_GEO_ID],
            "all_hhs": [100],
            "veh_0_all_hh": [20],
            "veh_1_all_hh": [40],
            "veh_1_hh_1": [15],
        }
    )
    result = mod._derive_vehicle(df)
    assert result["all_lo_veh_hh"].iloc[0] == 60
    assert result["perc_lo_veh"].iloc[0] == pytest.approx(0.6)
    assert result["perc_0_veh"].iloc[0] == pytest.approx(0.2)
    assert result["perc_1_veh"].iloc[0] == pytest.approx(0.4)
    assert result["perc_lo_veh_mod"].iloc[0] == pytest.approx(round(0.6 - 0.15, 3))


def test_derive_age_computes_youth_and_elderly() -> None:
    df = pd.DataFrame(
        {
            "GEO_ID": [_TRACT_GEO_ID],
            "total_pop": [1000],
            "m_15_17": [20],
            "f_15_17": [18],
            "m_18_19": [10],
            "f_18_19": [10],
            "m_20": [5],
            "f_20": [5],
            "m_21": [4],
            "f_21": [4],
            "m_65_66": [30],
            "f_65_66": [32],
            "m_67_69": [15],
            "f_67_69": [17],
            "m_70_74": [10],
            "f_70_74": [10],
            "m_75_79": [8],
            "f_75_79": [9],
            "m_80_84": [5],
            "f_80_84": [6],
            "m_a_85": [4],
            "f_a_85": [5],
        }
    )
    result = mod._derive_age(df)
    expected_youth = 20 + 18 + 10 + 10 + 5 + 5 + 4 + 4
    expected_elderly = 30 + 32 + 15 + 17 + 10 + 10 + 8 + 9 + 5 + 6 + 4 + 5
    assert result["all_youth"].iloc[0] == expected_youth
    assert result["all_elderly"].iloc[0] == expected_elderly
    assert result["perc_youth"].iloc[0] == pytest.approx(round(expected_youth / 1000, 3))
    assert "total_pop" not in result.columns


def test_derive_age_without_total_pop_skips_percentages() -> None:
    df = pd.DataFrame({"GEO_ID": [_TRACT_GEO_ID], "m_15_17": [10], "f_15_17": [10]})
    result = mod._derive_age(df)
    assert "perc_youth" not in result.columns
    assert result["all_youth"].iloc[0] == 20


# =============================================================================
# build_joined_table  (Stage 1 integration)
# =============================================================================


def test_build_joined_table_block_only_produces_rows(
    minimal_block_files: dict[str, list[str]],
) -> None:
    df = mod.build_joined_table(**minimal_block_files)
    assert len(df) >= 1
    assert "total_pop" in df.columns
    assert df["total_pop"].iloc[0] == 100


def test_build_joined_table_no_unfriendly_cols(
    minimal_block_files: dict[str, list[str]],
) -> None:
    df = mod.build_joined_table(**minimal_block_files)
    unfriendly = [c for c in df.columns if len(c) >= 5 and c[:2].isupper() and c[2:5].isdigit()]
    assert unfriendly == [], f"Raw Census codes leaked into output: {unfriendly}"


def test_build_joined_table_fips_filter_removes_other_counties(
    tmp_path: Path,
    minimal_block_files: dict[str, list[str]],
) -> None:
    pop2 = _census_csv(
        "GEO_ID,NAME,P1_001N",
        "Geography,Geographic Area Name,!!Total:",
        "1000000US240310001001001,Block 2,999",
    )
    pop_path2 = tmp_path / "P1b-Data.csv"
    _write_plain_csv(pop_path2, pop2)
    files = dict(minimal_block_files)
    files["pop_files"] = files["pop_files"] + [str(pop_path2)]

    df = mod.build_joined_table(**files, county_fips_filter=[_COUNTY_FIPS])
    assert all(df["FIPS"] == _COUNTY_FIPS)


def test_build_joined_table_with_income_files(tmp_path: Path) -> None:
    """Adding income files produces low_income and perc_low_income columns."""
    pop_path = tmp_path / "P1-Data.csv"
    hh_path = tmp_path / "H9-Data.csv"
    income_path = tmp_path / "B19001-Data.csv"

    _write_plain_csv(
        pop_path,
        _census_csv(
            "GEO_ID,NAME,P1_001N",
            "Geography,Area,!!Total:",
            f"{_BLOCK_GEO_ID},Block,100",
        ),
    )
    _write_plain_csv(
        hh_path,
        _census_csv(
            "GEO_ID,H9_001N",
            "Geography,!!Total:",
            f"{_BLOCK_GEO_ID},40",
        ),
    )
    bands = ",".join(
        [
            "B19001_001E",
            "B19001_002E",
            "B19001_003E",
            "B19001_004E",
            "B19001_005E",
            "B19001_006E",
            "B19001_007E",
            "B19001_008E",
            "B19001_009E",
            "B19001_010E",
            "B19001_011E",
        ]
    )
    _write_plain_csv(
        income_path,
        _census_csv(
            f"GEO_ID,NAME,{bands}",
            "Geography,Area Name," + ",".join(["label"] * 11),
            f"{_TRACT_GEO_ID},Test Tract,100,10,10,10,10,10,10,10,10,10,10",
        ),
    )
    df = mod.build_joined_table(
        pop_files=[str(pop_path)],
        hh_files=[str(hh_path)],
        jobs_files=[],
        income_files=[str(income_path)],
    )
    assert "low_income" in df.columns
    assert "perc_low_income" in df.columns


def test_build_joined_table_duplicate_topic_files_do_not_explode(tmp_path: Path) -> None:
    """Several files for one topic must not fan rows out into duplicate block keys.

    Regression: when a topic bucket gathered more than one file for the same
    geography (e.g. two ACS vintages, or race-iteration tables matched by the same
    signature), the repeated rows multiplied through the GEO_ID and block<->tract
    merges. That produced duplicate block keys, which aborted the Stage 3 ``1:1``
    join and left demographics absent from the bundle. Each block must stay unique.
    """
    blocks = ["1000000US110010001001001", "1000000US110010001001002"]
    pop_path = tmp_path / "P1-Data.csv"
    _write_plain_csv(
        pop_path,
        _census_csv(
            "GEO_ID,NAME,P1_001N",
            "Geo,Name,!!Total:",
            f"{blocks[0]},Block 1,100",
            f"{blocks[1]},Block 2,200",
        ),
    )
    hh_path = tmp_path / "H9-Data.csv"
    _write_plain_csv(
        hh_path,
        _census_csv("GEO_ID,H9_001N", "Geo,!!Total:", f"{blocks[0]},40", f"{blocks[1]},80"),
    )
    bands = ",".join(f"B19001_{n:03d}E" for n in range(1, 12))
    income_row = f"{_TRACT_GEO_ID},Tract,{','.join(['100'] * 11)}"
    income_csv = _census_csv(
        f"GEO_ID,NAME,{bands}", "Geo,Name," + ",".join(["lab"] * 11), income_row
    )
    # Two vintages of the same income table covering the same tract.
    inc1 = tmp_path / "ACSDT5Y2023.B19001-Data.csv"
    inc2 = tmp_path / "ACSDT5Y2024.B19001-Data.csv"
    _write_plain_csv(inc1, income_csv)
    _write_plain_csv(inc2, income_csv)

    df = mod.build_joined_table(
        pop_files=[str(pop_path)],
        hh_files=[str(hh_path)],
        jobs_files=[],
        income_files=[str(inc1), str(inc2)],
    )
    normalized = mod.normalize_attribute_keys(df)
    assert not normalized[mod.RIGHT_KEY].duplicated().any()
    assert normalized[mod.RIGHT_KEY].nunique() == len(blocks)
    # The kept income row still attaches to every block in the tract.
    assert "low_income" in df.columns


def test_build_joined_table_disaggregates_tract_count_to_blocks(tmp_path: Path) -> None:
    """A tract-level count is split across its blocks by population and sums back.

    The two blocks share one tract carrying 40 minority residents. Disaggregation must
    apportion that by block population (100 vs 300) into 10 and 30 -- not copy 40 onto
    each block -- so the parts still total the tract figure.
    """
    blocks = [("1000000US110010001001001", 100), ("1000000US110010001001002", 300)]
    pop_path = tmp_path / "P1-Data.csv"
    _write_plain_csv(
        pop_path,
        _census_csv(
            "GEO_ID,NAME,P1_001N",
            "Geo,Name,!!Total:",
            *[f"{g},Block,{p}" for g, p in blocks],
        ),
    )
    hh_path = tmp_path / "H9-Data.csv"
    _write_plain_csv(
        hh_path,
        _census_csv("GEO_ID,H9_001N", "Geo,!!Total:", *[f"{g},{p // 2}" for g, p in blocks]),
    )
    eth_cols = "P9_001N,P9_002N,P9_005N,P9_006N,P9_007N,P9_008N,P9_009N,P9_010N,P9_011N"
    eth_path = tmp_path / "P9-Data.csv"
    _write_plain_csv(
        eth_path,
        _census_csv(
            f"GEO_ID,NAME,{eth_cols}",
            "Geo,Name," + ",".join(["lab"] * 9),
            # total_pop=200, all_hisp=0, white=160, black=40, rest 0 -> minority=40.
            f"{_TRACT_GEO_ID},Tract,200,0,160,40,0,0,0,0,0",
        ),
    )
    df = mod.build_joined_table(
        pop_files=[str(pop_path)],
        hh_files=[str(hh_path)],
        jobs_files=[],
        ethnicity_files=[str(eth_path)],
    ).sort_values("total_pop")
    assert df["minority"].tolist() == pytest.approx([10.0, 30.0])
    assert df["minority"].sum() == pytest.approx(40.0)


# =============================================================================
# disaggregate_tract_counts_to_blocks
# =============================================================================


def test_disaggregate_splits_count_by_weight_and_sums_back() -> None:
    df = pd.DataFrame(
        {
            "tract_id_synth": ["T", "T", "T"],
            "total_pop": [100, 300, 0],
            "minority": [40, 40, 40],  # tract total copied onto each block
        }
    )
    out = mod.disaggregate_tract_counts_to_blocks(df)
    assert out["minority"].tolist() == pytest.approx([10.0, 30.0, 0.0])
    assert out["minority"].sum() == pytest.approx(40.0)


def test_disaggregate_renames_source_to_output_column() -> None:
    df = pd.DataFrame({"tract_id_synth": ["T", "T"], "total_pop": [1, 1], "all_youth": [10, 10]})
    out = mod.disaggregate_tract_counts_to_blocks(df)
    assert "youth" in out.columns
    assert "all_youth" not in out.columns
    assert out["youth"].tolist() == pytest.approx([5.0, 5.0])


def test_disaggregate_zero_weight_tract_yields_zero() -> None:
    df = pd.DataFrame({"tract_id_synth": ["Z", "Z"], "total_pop": [0, 0], "minority": [10, 10]})
    out = mod.disaggregate_tract_counts_to_blocks(df)
    assert out["minority"].tolist() == [0.0, 0.0]


def test_disaggregate_skips_field_when_weight_column_missing() -> None:
    df = pd.DataFrame({"tract_id_synth": ["T"], "minority": [40]})  # no total_pop weight
    out = mod.disaggregate_tract_counts_to_blocks(df)
    assert out["minority"].tolist() == [40]


def test_disaggregate_noop_without_tract_key() -> None:
    df = pd.DataFrame({"total_pop": [100], "minority": [40]})
    out = mod.disaggregate_tract_counts_to_blocks(df)
    assert out["minority"].tolist() == [40]


# =============================================================================
# discover_tiger_datasets  (Stage 2)
# =============================================================================


def test_discover_tiger_datasets_finds_all_zips(tiger_dir: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    assert len(paths) == len(FIXTURE_ZIPS)
    assert all(p.startswith("zip://") for p in paths)


def test_discover_tiger_datasets_stems_match_fixtures(tiger_dir: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    stems = {Path(p.removeprefix("zip://")).stem for p in paths}
    assert stems == {Path(name).stem for name in FIXTURE_ZIPS}


def test_discover_tiger_datasets_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.discover_tiger_datasets(tmp_path, "tl_*_*_*.shp")


def test_discover_tiger_datasets_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        mod.discover_tiger_datasets(tmp_path / "nonexistent", "tl_*_*_*.shp")


# =============================================================================
# merge_shapefiles  (Stage 2)
# =============================================================================


def test_merge_shapefiles_concatenates_features_and_preserves_crs(tiger_dir: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = mod.merge_shapefiles(paths)
    assert len(merged) == EXPECTED_TOTAL
    assert str(merged.crs) == "EPSG:4269"


def test_merge_shapefiles_state_codes_present(tiger_dir: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = mod.merge_shapefiles(paths)
    assert set(merged["STATEFP20"].unique()) == {"11", "24", "51"}


def test_merge_shapefiles_raises_on_crs_mismatch(tiger_dir: Path, tmp_path: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    gdf1 = gpd.read_file(paths[0])
    gdf2 = gpd.read_file(paths[1]).to_crs("EPSG:4326")

    p1 = tmp_path / "a.gpkg"
    p2 = tmp_path / "b.gpkg"
    gdf1.to_file(p1)
    gdf2.to_file(p2)

    with pytest.raises(RuntimeError, match="CRS mismatch"):
        mod.merge_shapefiles([str(p1), str(p2)])


# =============================================================================
# ensure_fips_column  (Stage 2, GeoDataFrame version)
# =============================================================================


def test_ensure_fips_column_builds_five_digit_code(tiger_dir: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = mod.merge_shapefiles(paths)
    result = mod.ensure_fips_column(merged)
    assert "FIPS" in result.columns
    assert (result["FIPS"].str.len() == 5).all()


def test_ensure_fips_column_is_idempotent(
    tiger_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gdf = mod._read_shapefile(f"zip://{tiger_dir / FIXTURE_ZIPS[0]}")
    mod.ensure_fips_column(gdf)
    original = gdf["FIPS"].copy()
    with caplog.at_level("INFO"):
        mod.ensure_fips_column(gdf)
    assert (gdf["FIPS"] == original).all()
    assert any("already present" in r.getMessage() for r in caplog.records)


def test_ensure_fips_column_raises_when_state_county_missing() -> None:
    gdf = gpd.GeoDataFrame({"val": [1], "geometry": [Point(0, 0)]})
    with pytest.raises(KeyError):
        mod.ensure_fips_column(gdf)


# =============================================================================
# filter_by_fips  (Stage 2)
# =============================================================================


def test_filter_by_fips_keeps_only_requested_counties(tiger_dir: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = mod.ensure_fips_column(mod.merge_shapefiles(paths))
    wanted = ["11001", "24031"]
    selected = mod.filter_by_fips(merged, wanted)
    assert set(selected["FIPS"].unique()) == set(wanted)
    assert len(selected) < len(merged)


def test_filter_by_fips_empty_list_returns_everything(tiger_dir: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    merged = mod.ensure_fips_column(mod.merge_shapefiles(paths))
    assert len(mod.filter_by_fips(merged, [])) == len(merged)


# =============================================================================
# normalize_attribute_keys  (Stage 3)
# =============================================================================


def test_normalize_attribute_keys_trims_24char_geo_id_to_15() -> None:
    df = pd.DataFrame({"GEO_ID": ["1000000US110010001001000", "1000000US110010001002000"]})
    result = mod.normalize_attribute_keys(df)
    assert (result["GEO_ID"].str.len() == 15).all()
    assert result["GEO_ID"].iloc[0] == "110010001001000"


def test_normalize_attribute_keys_leaves_15char_id_unchanged() -> None:
    df = pd.DataFrame({"GEO_ID": ["110010001001000"]})
    result = mod.normalize_attribute_keys(df)
    assert result["GEO_ID"].iloc[0] == "110010001001000"


def test_normalize_attribute_keys_derives_from_fallback_column() -> None:
    df = pd.DataFrame({"GEO_ID_blk": ["1000000US110010001001000"], "pop": [10]})
    result = mod.normalize_attribute_keys(df)
    assert "GEO_ID" in result.columns
    assert result["GEO_ID"].iloc[0] == "110010001001000"


def test_normalize_attribute_keys_raises_when_no_key_present() -> None:
    df = pd.DataFrame({"unrelated": [1]})
    with pytest.raises(KeyError, match="Neither"):
        mod.normalize_attribute_keys(df)


# =============================================================================
# join_blocks_to_attributes  (Stage 3)
# =============================================================================


@pytest.fixture()
def blocks_and_attrs(tiger_dir: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """DC block geometry + a 5-row attribute DataFrame keyed on GEOID20."""
    gdf = mod._read_shapefile(f"zip://{tiger_dir / FIXTURE_ZIPS[0]}")
    geoids = gdf["GEOID20"].tolist()[:5]
    attrs = pd.DataFrame({"GEO_ID": geoids, "total_pop": list(range(5))})
    return gdf, attrs


def test_join_left_preserves_all_block_rows(
    blocks_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    blocks, attrs = blocks_and_attrs
    result = mod.join_blocks_to_attributes(blocks, attrs)
    assert len(result) == len(blocks)


def test_join_returns_geodataframe_with_geometry(
    blocks_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    blocks, attrs = blocks_and_attrs
    result = mod.join_blocks_to_attributes(blocks, attrs)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.geometry.name in result.columns


def test_join_inner_keeps_only_matched_rows(
    blocks_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    blocks, attrs = blocks_and_attrs
    result = mod.join_blocks_to_attributes(blocks, attrs, how="inner")
    assert len(result) == len(attrs)


def test_join_attribute_columns_present_in_result(
    blocks_and_attrs: tuple[gpd.GeoDataFrame, pd.DataFrame],
) -> None:
    blocks, attrs = blocks_and_attrs
    assert "total_pop" in mod.join_blocks_to_attributes(blocks, attrs).columns


def test_join_raises_on_duplicate_attribute_keys(tiger_dir: Path) -> None:
    gdf = mod._read_shapefile(f"zip://{tiger_dir / FIXTURE_ZIPS[0]}")
    geoid = gdf["GEOID20"].iloc[0]
    dup_attrs = pd.DataFrame({"GEO_ID": [geoid, geoid], "total_pop": [1, 2]})
    with pytest.raises(ValueError):
        mod.join_blocks_to_attributes(gdf, dup_attrs)


def test_join_casts_int64_attribute_column_to_float(tiger_dir: Path) -> None:
    block = mod._read_shapefile(f"zip://{tiger_dir / FIXTURE_ZIPS[0]}").iloc[:1].copy()
    geoid = block["GEOID20"].iloc[0]
    attrs = pd.DataFrame({"GEO_ID": [geoid], "count": pd.array([5], dtype="Int64")})
    result = mod.join_blocks_to_attributes(block, attrs, how="inner")
    assert result["count"].dtype == "float64"


# =============================================================================
# _cast_int64_to_float
# =============================================================================


def test_cast_int64_to_float_converts_integer_column() -> None:
    gdf = gpd.GeoDataFrame(
        {"val": pd.array([1, 2, 3], dtype="Int64"), "geometry": [Point(0, 0)] * 3}
    )
    mod._cast_int64_to_float(gdf)
    assert gdf["val"].dtype == "float64"


def test_cast_int64_to_float_leaves_string_column_unchanged() -> None:
    gdf = gpd.GeoDataFrame({"label": ["a", "b", "c"], "geometry": [Point(0, 0)] * 3})
    mod._cast_int64_to_float(gdf)
    assert not pd.api.types.is_float_dtype(gdf["label"])


def test_cast_int64_to_float_noop_when_no_numeric_columns() -> None:
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
    mod._cast_int64_to_float(gdf)  # must not raise


# =============================================================================
# _truncate_field_names
# =============================================================================


def test_truncate_field_names_shortens_long_names() -> None:
    gdf = gpd.GeoDataFrame(
        {"a_very_long_column_name": [1], "another_verbose_col": [2], "geometry": [Point(0, 0)]}
    )
    result = mod._truncate_field_names(gdf, max_len=10)
    non_geom = [c for c in result.columns if c != "geometry"]
    assert all(len(c) <= 10 for c in non_geom)


def test_truncate_field_names_makes_collisions_unique() -> None:
    gdf = gpd.GeoDataFrame(
        {"total_population": [1], "total_pop_data": [2], "geometry": [Point(0, 0)]}
    )
    result = mod._truncate_field_names(gdf, max_len=10)
    non_geom = [c for c in result.columns if c != "geometry"]
    assert len(non_geom) == len(set(non_geom)), "Truncated names must be unique"


def test_truncate_field_names_preserves_short_names() -> None:
    gdf = gpd.GeoDataFrame({"fips": [1], "pop": [2], "geometry": [Point(0, 0)]})
    result = mod._truncate_field_names(gdf, max_len=10)
    assert "fips" in result.columns
    assert "pop" in result.columns


def test_truncate_field_names_never_renames_geometry_column() -> None:
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
    result = mod._truncate_field_names(gdf, max_len=5)
    assert "geometry" in result.columns


# =============================================================================
# write_geo
# =============================================================================


def test_write_geo_geopackage_roundtrip(tiger_dir: Path, tmp_path: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    gdf = mod.merge_shapefiles(paths)
    out = tmp_path / "out" / "blocks.gpkg"
    mod.write_geo(gdf, str(out))
    assert out.exists()
    assert len(gpd.read_file(out)) == len(gdf)


def test_write_geo_shapefile_roundtrip(tiger_dir: Path, tmp_path: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    gdf = mod.merge_shapefiles(paths)
    out = tmp_path / "out" / "blocks.shp"
    mod.write_geo(gdf, str(out))
    assert out.exists()
    assert len(gpd.read_file(out)) == len(gdf)


def test_write_geo_creates_missing_parent_dirs(tiger_dir: Path, tmp_path: Path) -> None:
    paths = mod.discover_tiger_datasets(tiger_dir, "tl_*_*_*.shp")
    gdf = mod.merge_shapefiles(paths)
    out = tmp_path / "a" / "b" / "c" / "blocks.gpkg"
    mod.write_geo(gdf, str(out))
    assert out.exists()


def test_write_geo_shapefile_truncates_long_column_names(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "a_very_long_column_name_here": [1],
            "geometry": gpd.GeoSeries([Point(0, 0)], crs="EPSG:4326"),
        }
    )
    out = tmp_path / "out.shp"
    mod.write_geo(gdf, str(out))
    roundtripped = gpd.read_file(out)
    non_geom = [c for c in roundtripped.columns if c != "geometry"]
    assert all(len(c) <= 10 for c in non_geom)


def test_shp_schema_handles_all_nan_object_column() -> None:
    # A tract attribute that matched no block in a left join is an all-NaN object
    # column; astype(str).str.len().max() is NaN (pandas 3.x keeps NaN), which used
    # to crash int(NaN). It must size to a string field instead of raising.
    gdf = gpd.GeoDataFrame(
        {
            "name_blk": ["block a", "block b"],
            "name_trt": [None, None],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    props = mod._shp_schema(gdf)["properties"]
    assert props["name_trt"].startswith("str:")
    assert props["name_blk"] == "str:7"


def test_write_geo_shapefile_with_all_nan_object_column_roundtrips(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "minority": [10.0, 30.0],
            "empty_str": pd.Series([None, None], dtype="object"),
            "geometry": gpd.GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:4326"),
        }
    )
    out = tmp_path / "out.shp"
    mod.write_geo(gdf, str(out))  # must not raise on the all-NaN object column
    roundtripped = gpd.read_file(out)
    assert len(roundtripped) == 2
    assert roundtripped["minority"].tolist() == [10.0, 30.0]


# =============================================================================
# write_csv
# =============================================================================


def test_write_csv_creates_file_with_correct_content(tmp_path: Path) -> None:
    df = pd.DataFrame({"GEO_ID": ["A", "B"], "pop": [10, 20]})
    out = tmp_path / "out.csv"
    mod.write_csv(df, str(out))
    assert out.exists()
    roundtripped = pd.read_csv(out)
    assert len(roundtripped) == 2
    assert list(roundtripped.columns) == ["GEO_ID", "pop"]


def test_write_csv_creates_missing_parent_dirs(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1]})
    out = tmp_path / "nested" / "dir" / "out.csv"
    mod.write_csv(df, str(out))
    assert out.exists()


# =============================================================================
# _is_blank
# =============================================================================


def test_is_blank_none() -> None:
    assert mod._is_blank(None) is True


def test_is_blank_empty_string() -> None:
    assert mod._is_blank("") is True


def test_is_blank_whitespace_only() -> None:
    assert mod._is_blank("   ") is True


def test_is_blank_real_path_is_not_blank() -> None:
    assert mod._is_blank("/some/path/file.shp") is False


def test_is_blank_single_char_is_not_blank() -> None:
    assert mod._is_blank("x") is False


# =============================================================================
# _check_placeholders
# =============================================================================


def test_check_placeholders_returns_true_when_defaults_unchanged() -> None:
    # All module-level paths are still at their placeholder defaults out of the box.
    assert mod._check_placeholders() is True


def test_check_placeholders_returns_false_when_paths_overridden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mod, "INPUT_CSV_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(mod, "INPUT_SHP_DIR", str(tmp_path / "shp"))
    monkeypatch.setattr(mod, "FINAL_JOINED_FEATURES", str(tmp_path / "out.gpkg"))
    monkeypatch.setattr(mod, "INTERMEDIATE_COMBINED_CSV", "")
    monkeypatch.setattr(mod, "INTERMEDIATE_MERGED_SHP", "")
    assert mod._check_placeholders() is False


def test_skip_placeholder_output_blanks_placeholder() -> None:
    assert (
        mod._skip_placeholder_output(
            mod._DEFAULT_INTERMEDIATE_COMBINED_CSV,
            mod._DEFAULT_INTERMEDIATE_COMBINED_CSV,
            "--intermediate-csv",
        )
        == ""
    )


def test_skip_placeholder_output_passes_through_real_path() -> None:
    assert (
        mod._skip_placeholder_output(
            "out/joined.csv", mod._DEFAULT_INTERMEDIATE_COMBINED_CSV, "--intermediate-csv"
        )
        == "out/joined.csv"
    )


def test_run_proceeds_when_only_intermediates_are_placeholders(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The orchestrator wires --input-csv-dir/--input-shp-dir/--output but not the
    # optional intermediates, so they fall back to their placeholder defaults.
    # That must no longer abort the run with "No processing performed".
    csv_dir = tmp_path / "census"
    csv_dir.mkdir()
    shp_dir = tmp_path / "tiger"
    shp_dir.mkdir()
    # The empty input dirs make the real pipeline fail downstream (run() then
    # sys.exit(1)); that's incidental. What matters is that it reached Stage 1
    # instead of short-circuiting at the placeholder guard.
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit):
        mod.run(
            input_csv_dir=str(csv_dir),
            input_shp_dir=str(shp_dir),
            final_joined_features=str(tmp_path / "out.shp"),
        )
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "No processing performed" not in messages
    assert "Stage 1/3" in messages  # got past the guard and began real work
