"""Tests for the pandas (CSV) stage of scripts/national_data_tools/uscensus_tiger_join_arcpy.py.

ArcPy is unavailable outside ArcGIS Pro, so a minimal stand-in module is installed
for the import. Only the pandas-level CSV stage is exercised; no geoprocessing
function is called.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

MODULE = "scripts.national_data_tools.uscensus_tiger_join_arcpy"

_BLOCKS = ("1000000US110010001001001", "1000000US110010001001002", "1000000US240310001001001")
_TRACTS = ("1400000US11001000100", "1400000US24031000100")


@pytest.fixture()
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the script against a stand-in ``arcpy`` without leaking either module."""
    stub = types.ModuleType("arcpy")
    stub.env = types.SimpleNamespace(overwriteOutput=False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arcpy", stub)
    sys.modules.pop(MODULE, None)
    yield importlib.import_module(MODULE)
    sys.modules.pop(MODULE, None)


def _write_csv(path: Path, header: str, *rows: str) -> None:
    labels = ",".join(["label"] * len(header.split(",")))
    path.write_text("\n".join([header, labels, *rows]) + "\n", encoding="utf-8")


def _write_pop(folder: Path) -> None:
    _write_csv(
        folder / "DECENNIALPL2020.P1-Data.csv",
        "GEO_ID,NAME,P1_001N",
        *[f"{geo},Block,{100 * (i + 1)}" for i, geo in enumerate(_BLOCKS)],
    )


def test_county_filter_uses_block_id_after_tract_merge(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    # A tract table makes the block<->tract merge suffix GEO_ID into GEO_ID_blk /
    # GEO_ID_trt; the county filter used to read the vanished GEO_ID and raise KeyError.
    _write_pop(tmp_path)
    bands = ",".join(f"B19001_{n:03d}E" for n in range(1, 12))
    _write_csv(
        tmp_path / "ACSDT5Y2024.B19001-Data.csv",
        f"GEO_ID,NAME,{bands}",
        *[f"{tract},Tract," + ",".join(["10"] * 11) for tract in _TRACTS],
    )
    df = mod.build_joined_table_from_folder(tmp_path, county_fips_filter=["11001"])
    assert sorted(df["GEO_ID_blk"]) == sorted(_BLOCKS[:2])


def test_county_filter_without_tract_tables(mod: types.ModuleType, tmp_path: Path) -> None:
    _write_pop(tmp_path)
    df = mod.build_joined_table_from_folder(tmp_path, county_fips_filter=["24031"])
    assert df["GEO_ID"].tolist() == [_BLOCKS[2]]


def test_duplicated_inputs_do_not_multiply_rows(mod: types.ModuleType, tmp_path: Path) -> None:
    # The same block and tract rows supplied twice used to fan out to 2 x 2 rows per
    # block; identical repeats now collapse before any merge.
    bands = ",".join(f"B19001_{n:03d}E" for n in range(1, 12))
    for sub in ("a", "b"):
        folder = tmp_path / sub
        folder.mkdir()
        _write_pop(folder)
        _write_csv(
            folder / "ACSDT5Y2024.B19001-Data.csv",
            f"GEO_ID,NAME,{bands}",
            *[f"{tract},Tract," + ",".join(["10"] * 11) for tract in _TRACTS],
        )
    df = mod.build_joined_table_from_folder(tmp_path)
    assert sorted(df["GEO_ID_blk"].dropna()) == sorted(_BLOCKS)


def test_conflicting_vintages_are_rejected(mod: types.ModuleType, tmp_path: Path) -> None:
    _write_pop(tmp_path)
    bands = ",".join(f"B19001_{n:03d}E" for n in range(1, 12))
    for vintage, value in (("ACSDT5Y2023", "10"), ("ACSDT5Y2024", "12")):
        _write_csv(
            tmp_path / f"{vintage}.B19001-Data.csv",
            f"GEO_ID,NAME,{bands}",
            f"{_TRACTS[0]},Tract," + ",".join([value] * 11),
        )
    with pytest.raises(ValueError, match="conflicting"):
        mod.build_joined_table_from_folder(tmp_path)


def test_lep_counts_every_less_than_very_well_row(mod: types.ModuleType, tmp_path: Path) -> None:
    # Each C16001 estimate holds a distinct value, so a missing LEP row (_029E,
    # Tagalog) or a stray "very well" row (_037E) changes the total.
    _write_pop(tmp_path)
    codes = [f"C16001_{n:03d}E" for n in range(1, 39)]
    _write_csv(
        tmp_path / "ACSDT5Y2024.C16001-Data.csv",
        "GEO_ID,NAME," + ",".join(codes),
        f"{_TRACTS[0]},Tract," + ",".join(str(n) for n in range(1, 39)),
    )
    df = mod.build_joined_table_from_folder(tmp_path, county_fips_filter=["11001"])
    lep_rows = (5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38)
    # The tract's LEP total, split across its two blocks by population (100 vs 200).
    assert df["LEP_CNT"].sum() == pytest.approx(sum(lep_rows))
    assert sorted(df["LEP_CNT"]) == pytest.approx([sum(lep_rows) / 3, sum(lep_rows) * 2 / 3])
    assert "TGL_NWELL" in df.columns


def test_tract_counts_are_split_across_blocks(mod: types.ModuleType, tmp_path: Path) -> None:
    # Count allocation: a tract count is split by block households (income) or block
    # population (everything else), summing back to the tract total, and the table
    # is marked so the demographics script reads the counts as block counts.
    _write_pop(tmp_path)
    _write_csv(
        tmp_path / "DECENNIALDHC2020.H9-Data.csv",
        "GEO_ID,H9_001N",
        f"{_BLOCKS[0]},10",
        f"{_BLOCKS[1]},30",
        f"{_BLOCKS[2]},50",
    )
    bands = ",".join(f"B19001_{n:03d}E" for n in range(1, 12))
    _write_csv(
        tmp_path / "ACSDT5Y2024.B19001-Data.csv",
        f"GEO_ID,NAME,{bands}",
        f"{_TRACTS[0]},Tract,100," + ",".join(["8"] * 10),
    )
    df = mod.build_joined_table_from_folder(tmp_path, county_fips_filter=["11001"])
    df = df.sort_values("GEO_ID_blk")
    assert df["HH_LOWINC"].tolist() == pytest.approx([20.0, 60.0])  # 80 split 10:30
    assert df["PCT_LOWINC"].tolist() == pytest.approx([0.8, 0.8])  # tract rate kept
    assert set(df[mod.COUNT_ALLOCATION_FIELD]) == {1}
