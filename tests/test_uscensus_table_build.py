"""Tests for scripts/census_tools/uscensus_table_build.py."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.census_tools.uscensus_table_build import (
    GEO_ID_COL,
    _apply_fips_filter,
    _clean_name_cols,
    _derive_age,
    _derive_ethnicity,
    _derive_income,
    _derive_language,
    _derive_vehicle,
    _drop_unfriendly_cols,
    _ensure_fips_column,
    _fill_numeric_only,
    _load_and_concat,
    _merge_on_geo_id,
    _read_csv_any,
    _token_match,
    build_joined_table,
    discover_census_files,
)

FIXTURE_DIR = Path("tests/fixtures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_GEO_PREFIX = "1000000US"
_TRACT_GEO_PREFIX = "1400000US"

# A block GEOID whose numeric portion is exactly 15 digits.
# Positions [9:20] → "11001000100" (tract), [9:24] → "110010001001001" (block).
_BLOCK_GEO_ID = f"{_BLOCK_GEO_PREFIX}110010001001001"
_TRACT_GEO_ID = f"{_TRACT_GEO_PREFIX}11001000100"

# County FIPS embedded in both IDs above.
_COUNTY_FIPS = "11001"


def _census_csv(header: str, label: str, *data_rows: str) -> str:
    """Return a Census-style CSV string with a human-readable label row at index 1."""
    lines = [header, label, *data_rows]
    return "\n".join(lines) + "\n"


def _write_plain_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_gz_csv(path: Path, content: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(content)


def _write_zip_csv(zip_path: Path, csv_name: str, content: str) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(csv_name, content)


# ---------------------------------------------------------------------------
# _token_match
# ---------------------------------------------------------------------------


def test_token_match_single_token_present() -> None:
    assert _token_match("DECENNIALPL2020.P1-Data.csv", ("P1",)) is True


def test_token_match_single_token_absent() -> None:
    assert _token_match("DECENNIALPL2020.P1-Data.csv", ("H9",)) is False


def test_token_match_case_insensitive() -> None:
    assert _token_match("dc_wac_S000_JT00_2023.csv.gz", ("_s000_jt00_",)) is True


def test_token_match_all_tokens_must_match() -> None:
    assert _token_match("ACSDT5Y2024.B19001-Data.csv", ("B19001", "ACSDT5Y2024")) is True


def test_token_match_partial_tokens_fails() -> None:
    assert _token_match("ACSDT5Y2024.B19001-Data.csv", ("B19001", "DECENNIAL")) is False


def test_token_match_string_token_treated_as_single() -> None:
    assert _token_match("my_P1_file-Data.csv", "P1") is True


# ---------------------------------------------------------------------------
# discover_census_files
# ---------------------------------------------------------------------------


def test_discover_census_files_finds_data_csv(tmp_path: Path) -> None:
    (tmp_path / "DECENNIALPL2020.P1-Data.csv").write_text("x")
    result = discover_census_files(tmp_path, {"POP": ("P1",)})
    assert len(result["POP"]) == 1
    assert result["POP"][0].endswith("P1-Data.csv")


def test_discover_census_files_finds_csv_gz(tmp_path: Path) -> None:
    (tmp_path / "dc_wac_S000_JT00_2023.csv.gz").write_bytes(b"")
    result = discover_census_files(tmp_path, {"JOBS": ("_S000_JT00_",)})
    assert len(result["JOBS"]) == 1


def test_discover_census_files_finds_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "ACSDT5Y2024.B19001-Data.zip"
    _write_zip_csv(zip_path, "ACSDT5Y2024.B19001-Data.csv", "GEO_ID\n")
    result = discover_census_files(tmp_path, {"INCOME": ("B19001",)})
    assert len(result["INCOME"]) == 1


def test_discover_census_files_ignores_unmatched_files(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("hello")
    (tmp_path / "some_other-Data.csv").write_text("x")
    result = discover_census_files(tmp_path, {"POP": ("P1",)})
    assert result["POP"] == []


def test_discover_census_files_results_are_sorted(tmp_path: Path) -> None:
    for name in ["z_P1-Data.csv", "a_P1-Data.csv", "m_P1-Data.csv"]:
        (tmp_path / name).write_text("x")
    result = discover_census_files(tmp_path, {"POP": ("P1",)})
    assert result["POP"] == sorted(result["POP"])


def test_discover_census_files_recurses_into_subdirs(tmp_path: Path) -> None:
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    (sub / "P1-Data.csv").write_text("x")
    result = discover_census_files(tmp_path, {"POP": ("P1",)})
    assert len(result["POP"]) == 1


# ---------------------------------------------------------------------------
# _read_csv_any
# ---------------------------------------------------------------------------


def test_read_csv_any_plain_csv(tmp_path: Path) -> None:
    p = tmp_path / "test-Data.csv"
    _write_plain_csv(p, "a,b\n1,2\n3,4\n")
    df = _read_csv_any(p)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_csv_any_csv_gz(tmp_path: Path) -> None:
    p = tmp_path / "test.csv.gz"
    _write_gz_csv(p, "a,b\n1,2\n")
    df = _read_csv_any(p)
    assert len(df) == 1
    assert df["a"].iloc[0] == 1


def test_read_csv_any_zip_with_data_csv(tmp_path: Path) -> None:
    zip_path = tmp_path / "bundle.zip"
    _write_zip_csv(zip_path, "bundle-Data.csv", "x,y\n10,20\n")
    df = _read_csv_any(zip_path)
    assert list(df.columns) == ["x", "y"]
    assert df["x"].iloc[0] == 10


def test_read_csv_any_zip_missing_data_csv_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    _write_zip_csv(zip_path, "notes.txt", "no csv here")
    with pytest.raises(FileNotFoundError):
        _read_csv_any(zip_path)


# ---------------------------------------------------------------------------
# _fill_numeric_only
# ---------------------------------------------------------------------------


def test_fill_numeric_only_fills_numeric_nans() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": [float("nan"), 3.0]})
    result = _fill_numeric_only(df)
    assert result["a"].iloc[1] == 0
    assert result["b"].iloc[0] == 0


def test_fill_numeric_only_leaves_object_columns_untouched() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")], "name": ["Alice", None]})
    result = _fill_numeric_only(df)
    assert pd.isna(result["name"].iloc[1])


# ---------------------------------------------------------------------------
# _clean_name_cols
# ---------------------------------------------------------------------------


def test_clean_name_cols_strips_control_chars() -> None:
    df = pd.DataFrame({"NAME": ["foo\r\nbar", "baz\ttab"]})
    _clean_name_cols(df)
    assert df["NAME"].iloc[0] == "foo bar"
    assert df["NAME"].iloc[1] == "baz tab"


def test_clean_name_cols_only_touches_name_columns() -> None:
    df = pd.DataFrame({"NAME": ["a\nb"], "OTHER": ["c\nd"]})
    _clean_name_cols(df)
    assert "c\nd" == df["OTHER"].iloc[0]


# ---------------------------------------------------------------------------
# _merge_on_geo_id
# ---------------------------------------------------------------------------


def test_merge_on_geo_id_outer_merges_on_geo_id() -> None:
    left = pd.DataFrame({"GEO_ID": ["A", "B"], "pop": [10, 20]})
    right = pd.DataFrame({"GEO_ID": ["B", "C"], "jobs": [5, 15]})
    result = _merge_on_geo_id(left, right)
    assert set(result["GEO_ID"]) == {"A", "B", "C"}
    assert result.loc[result["GEO_ID"] == "B", "pop"].iloc[0] == 20
    assert result.loc[result["GEO_ID"] == "B", "jobs"].iloc[0] == 5


def test_merge_on_geo_id_left_empty_returns_right() -> None:
    right = pd.DataFrame({"GEO_ID": ["A"], "pop": [5]})
    result = _merge_on_geo_id(pd.DataFrame(), right)
    assert len(result) == 1
    assert "pop" in result.columns


def test_merge_on_geo_id_right_empty_returns_left() -> None:
    left = pd.DataFrame({"GEO_ID": ["A"], "pop": [5]})
    result = _merge_on_geo_id(left, pd.DataFrame())
    assert len(result) == 1
    assert "pop" in result.columns


def test_merge_on_geo_id_drops_duplicate_columns() -> None:
    left = pd.DataFrame({"GEO_ID": ["A"], "NAME": ["Left"], "pop": [5]})
    right = pd.DataFrame({"GEO_ID": ["A"], "NAME": ["Right"], "jobs": [10]})
    result = _merge_on_geo_id(left, right)
    # NAME is a duplicate (not GEO_ID) — only one copy should remain.
    assert result.columns.tolist().count("NAME") == 1


# ---------------------------------------------------------------------------
# _drop_unfriendly_cols
# ---------------------------------------------------------------------------


def test_drop_unfriendly_cols_removes_raw_census_codes() -> None:
    # The regex ^[A-Z]{2,}\d{3,}.* requires 2+ uppercase letters then 3+ digits.
    # NP001E and GEO123 are synthetic names that match this pattern.
    df = pd.DataFrame({"GEO_ID": ["A"], "NAME": ["x"], "NP001E": [100], "total_pop": [200]})
    result = _drop_unfriendly_cols(df)
    assert "NP001E" not in result.columns
    assert "total_pop" in result.columns
    assert "GEO_ID" in result.columns


def test_drop_unfriendly_cols_keeps_friendly_columns() -> None:
    df = pd.DataFrame({"GEO_ID": ["A"], "low_income": [5], "perc_lep": [0.1]})
    result = _drop_unfriendly_cols(df)
    assert list(result.columns) == ["GEO_ID", "low_income", "perc_lep"]


# ---------------------------------------------------------------------------
# _ensure_fips_column
# ---------------------------------------------------------------------------


def test_ensure_fips_column_creates_five_digit_fips() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID]})
    _ensure_fips_column(df)
    assert "FIPS" in df.columns
    assert df["FIPS"].iloc[0] == _COUNTY_FIPS


def test_ensure_fips_column_is_idempotent() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "FIPS": ["99999"]})
    _ensure_fips_column(df)
    assert df["FIPS"].iloc[0] == "99999"


def test_ensure_fips_column_raises_when_no_geo_col() -> None:
    df = pd.DataFrame({"other": [1]})
    with pytest.raises(KeyError):
        _ensure_fips_column(df)


# ---------------------------------------------------------------------------
# _apply_fips_filter
# ---------------------------------------------------------------------------


def test_apply_fips_filter_keeps_only_matching_rows() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID, "1000000US240310001001001"], "pop": [10, 20]})
    result = _apply_fips_filter(df, fips=[_COUNTY_FIPS])
    assert len(result) == 1
    assert result["pop"].iloc[0] == 10


def test_apply_fips_filter_empty_fips_returns_unchanged() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "pop": [10]})
    result = _apply_fips_filter(df, fips=[])
    assert len(result) == 1


def test_apply_fips_filter_none_returns_unchanged() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "pop": [10]})
    result = _apply_fips_filter(df, fips=None)
    assert len(result) == 1


def test_apply_fips_filter_zero_pads_short_fips() -> None:
    df = pd.DataFrame({"GEO_ID": [_BLOCK_GEO_ID], "pop": [10]})
    # Pass county FIPS without leading zero to ensure zero-padding works.
    result = _apply_fips_filter(df, fips=["11001"])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Derivation functions
# ---------------------------------------------------------------------------


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
    data: dict[str, list[int | str]] = {b: [10] for b in bands}
    data["total_hh"] = [100]
    data[GEO_ID_COL] = [_TRACT_GEO_ID]
    df = pd.DataFrame(data)

    result = _derive_income(df)

    assert result["low_income"].iloc[0] == 100
    assert result["perc_low_income"].iloc[0] == pytest.approx(1.0)
    assert "total_hh" not in result.columns


def test_derive_ethnicity_computes_minority_sum_and_percentage() -> None:
    df = pd.DataFrame(
        {
            GEO_ID_COL: [_TRACT_GEO_ID],
            "total_pop": [200],
            "black": [40],
            "native": [10],
            "asian": [20],
            "pac_isl": [5],
            "other": [5],
            "multi": [10],
        }
    )
    result = _derive_ethnicity(df)
    assert result["minority"].iloc[0] == 90
    assert result["perc_minority"].iloc[0] == pytest.approx(0.45)
    assert "total_pop" not in result.columns


def test_derive_language_computes_lep_percentage() -> None:
    df = pd.DataFrame(
        {
            GEO_ID_COL: [_TRACT_GEO_ID],
            "total_lang_pop": [200],
            "spanish_engnwell": [20],
            "korean_engnwell": [10],
        }
    )
    result = _derive_language(df)
    assert result["all_nwell"].iloc[0] == 30
    assert result["perc_lep"].iloc[0] == pytest.approx(0.15)


def test_derive_language_zero_lang_pop_yields_zero_lep() -> None:
    df = pd.DataFrame(
        {
            GEO_ID_COL: [_TRACT_GEO_ID],
            "total_lang_pop": [0],
            "spanish_engnwell": [0],
        }
    )
    result = _derive_language(df)
    assert result["perc_lep"].iloc[0] == pytest.approx(0.0)


def test_derive_vehicle_computes_low_vehicle_metrics() -> None:
    df = pd.DataFrame(
        {
            GEO_ID_COL: [_TRACT_GEO_ID],
            "all_hhs": [100],
            "veh_0_all_hh": [20],
            "veh_1_all_hh": [40],
            "veh_1_hh_1": [15],
        }
    )
    result = _derive_vehicle(df)
    assert result["all_lo_veh_hh"].iloc[0] == 60
    assert result["perc_lo_veh"].iloc[0] == pytest.approx(0.6)
    assert result["perc_0_veh"].iloc[0] == pytest.approx(0.2)
    assert result["perc_1_veh"].iloc[0] == pytest.approx(0.4)
    # perc_lo_veh_mod = perc_lo_veh - perc_veh_1_hh_1
    assert result["perc_lo_veh_mod"].iloc[0] == pytest.approx(round(0.6 - 0.15, 3))


def test_derive_age_computes_youth_and_elderly() -> None:
    df = pd.DataFrame(
        {
            GEO_ID_COL: [_TRACT_GEO_ID],
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
    result = _derive_age(df)
    expected_youth = 20 + 18 + 10 + 10 + 5 + 5 + 4 + 4
    expected_elderly = 30 + 32 + 15 + 17 + 10 + 10 + 8 + 9 + 5 + 6 + 4 + 5
    assert result["all_youth"].iloc[0] == expected_youth
    assert result["all_elderly"].iloc[0] == expected_elderly
    assert result["perc_youth"].iloc[0] == pytest.approx(round(expected_youth / 1000, 3))
    assert "total_pop" not in result.columns


def test_derive_age_without_total_pop_skips_percentages() -> None:
    df = pd.DataFrame({GEO_ID_COL: [_TRACT_GEO_ID], "m_15_17": [10], "f_15_17": [10]})
    result = _derive_age(df)
    assert "perc_youth" not in result.columns
    assert result["all_youth"].iloc[0] == 20


# ---------------------------------------------------------------------------
# _load_and_concat
# ---------------------------------------------------------------------------


def test_load_and_concat_empty_list_returns_empty_df(tmp_path: Path) -> None:
    result = _load_and_concat([])
    assert result.empty


def test_load_and_concat_reads_and_returns_single_file(tmp_path: Path) -> None:
    p = tmp_path / "p1-Data.csv"
    _write_plain_csv(p, "GEO_ID,NAME,P1_001N\nlabel,label,label\nA,Area A,50\n")
    result = _load_and_concat([str(p)], skiprows=[1])
    assert len(result) == 1
    assert result["P1_001N"].iloc[0] == 50


def test_load_and_concat_concatenates_multiple_files(tmp_path: Path) -> None:
    for i, name in enumerate(["a-Data.csv", "b-Data.csv"]):
        (tmp_path / name).write_text(f"GEO_ID,val\nGEO_{i},{i * 10}\n", encoding="utf-8")
    result = _load_and_concat([str(tmp_path / "a-Data.csv"), str(tmp_path / "b-Data.csv")])
    assert len(result) == 2


def test_load_and_concat_applies_column_rename(tmp_path: Path) -> None:
    p = tmp_path / "pop-Data.csv"
    _write_plain_csv(p, "GEO_ID,NAME,P1_001N\nlabel,label,label\nA,Area A,42\n")
    result = _load_and_concat(
        [str(p)],
        skiprows=[1],
        rename={"P1_001N": "total_pop"},
    )
    assert "total_pop" in result.columns
    assert result["total_pop"].iloc[0] == 42


def test_load_and_concat_reads_zip_transparently(tmp_path: Path) -> None:
    zip_path = tmp_path / "P1-Data.zip"
    csv_content = "GEO_ID,P1_001N\nA,99\n"
    _write_zip_csv(zip_path, "P1-Data.csv", csv_content)
    result = _load_and_concat([str(zip_path)])
    assert result["P1_001N"].iloc[0] == 99


# ---------------------------------------------------------------------------
# build_joined_table  (integration — generates minimal in-memory fixtures)
# ---------------------------------------------------------------------------


def _make_block_pop_csv(geo_id: str = _BLOCK_GEO_ID, pop: int = 100) -> str:
    return _census_csv(
        "GEO_ID,NAME,P1_001N",
        "Geography,Geographic Area Name,!!Total:",
        f"{geo_id},Test Block,{pop}",
    )


def _make_block_hh_csv(geo_id: str = _BLOCK_GEO_ID, hh: int = 40) -> str:
    return _census_csv(
        "GEO_ID,H9_001N",
        "Geography,!!Total:",
        f"{geo_id},{hh}",
    )


def _make_wac_csv(geocode: str = "110010001001001", jobs: int = 50) -> str:
    return f"w_geocode,C000,CE01,CE02,CE03\n{geocode},{jobs},10,15,25\n"


@pytest.fixture()
def minimal_block_files(tmp_path: Path) -> dict[str, list[str]]:
    """Write minimal block-level CSV fixtures and return paths by role."""
    pop_path = tmp_path / "P1-Data.csv"
    hh_path = tmp_path / "H9-Data.csv"
    wac_path = tmp_path / "wac_S000_JT00.csv.gz"

    _write_plain_csv(pop_path, _make_block_pop_csv())
    _write_plain_csv(hh_path, _make_block_hh_csv())
    _write_gz_csv(wac_path, _make_wac_csv())

    return {
        "pop_files": [str(pop_path)],
        "hh_files": [str(hh_path)],
        "jobs_files": [str(wac_path)],
    }


def test_build_joined_table_block_only_produces_rows(
    minimal_block_files: dict[str, list[str]],
) -> None:
    df = build_joined_table(**minimal_block_files)
    assert len(df) >= 1
    assert "total_pop" in df.columns
    assert df["total_pop"].iloc[0] == 100


def test_build_joined_table_no_unfriendly_cols(
    minimal_block_files: dict[str, list[str]],
) -> None:
    df = build_joined_table(**minimal_block_files)
    unfriendly = [c for c in df.columns if len(c) >= 5 and c[:2].isupper() and c[2:5].isdigit()]
    assert unfriendly == [], f"Raw Census codes leaked into output: {unfriendly}"


def test_build_joined_table_fips_filter_removes_other_counties(
    tmp_path: Path,
    minimal_block_files: dict[str, list[str]],
) -> None:
    # Add a second block in a different county (24031).
    pop2 = _make_block_pop_csv(geo_id="1000000US240310001001001", pop=999)
    pop_path2 = tmp_path / "P1b-Data.csv"
    _write_plain_csv(pop_path2, pop2)
    files = dict(minimal_block_files)
    files["pop_files"] = files["pop_files"] + [str(pop_path2)]

    df = build_joined_table(**files, county_fips_filter=[_COUNTY_FIPS])
    assert all(df["FIPS"] == _COUNTY_FIPS)


def test_build_joined_table_with_tract_income(tmp_path: Path) -> None:
    """Adding income files produces low_income and perc_low_income columns."""
    pop_path = tmp_path / "P1-Data.csv"
    hh_path = tmp_path / "H9-Data.csv"
    income_path = tmp_path / "B19001-Data.csv"

    _write_plain_csv(pop_path, _make_block_pop_csv())
    _write_plain_csv(hh_path, _make_block_hh_csv())

    # Tract-level income CSV — GEO_ID[9:] must match block's GEO_ID[9:20].
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
            "Geography,Geographic Area Name," + ",".join(["label"] * 11),
            f"{_TRACT_GEO_ID},Test Tract,100,10,10,10,10,10,10,10,10,10,10",
        ),
    )

    df = build_joined_table(
        pop_files=[str(pop_path)],
        hh_files=[str(hh_path)],
        jobs_files=[],
        income_files=[str(income_path)],
    )
    assert "low_income" in df.columns
    assert "perc_low_income" in df.columns


# ---------------------------------------------------------------------------
# discover_census_files  (integration using existing test fixtures)
# ---------------------------------------------------------------------------


def test_discover_census_files_with_real_fixtures() -> None:
    """The real fixture directory contains files for all expected topics."""
    result = discover_census_files(FIXTURE_DIR)
    # At minimum the P1 (population) and B19001 (income) fixtures must be found.
    assert len(result["POP_FILES"]) >= 1
    assert len(result["INCOME_FILES"]) >= 1
