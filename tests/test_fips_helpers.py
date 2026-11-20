from __future__ import annotations

from pathlib import Path

import pytest

from utils.fips_helpers import load_fips_codes

# ---------------------------------------------------------------------------
# load_fips_codes
# ---------------------------------------------------------------------------


def test_load_fips_codes_empty_inputs_mean_no_filter() -> None:
    assert load_fips_codes(None, None) == []
    assert load_fips_codes([], "") == []


def test_load_fips_codes_inline_only_sorted_and_deduplicated() -> None:
    assert load_fips_codes(["51059", "11001", "51059", " 24031 "]) == ["11001", "24031", "51059"]


def test_load_fips_codes_unions_inline_and_file(tmp_path: Path) -> None:
    fips_file = tmp_path / "fips.txt"
    fips_file.write_text(
        "# Planning area\n\n51059  # Fairfax County, VA\n24031, 24033\n11001 51013\n",
        encoding="utf-8",
    )
    assert load_fips_codes(["51059", "51510"], fips_file) == [
        "11001",
        "24031",
        "24033",
        "51013",
        "51059",
        "51510",
    ]


def test_load_fips_codes_zero_pads_four_digit_codes(tmp_path: Path) -> None:
    fips_file = tmp_path / "fips.txt"
    fips_file.write_text("1001\n", encoding="utf-8")
    assert load_fips_codes([6037], str(fips_file)) == ["01001", "06037"]  # type: ignore[list-item]


def test_load_fips_codes_tolerates_byte_order_mark(tmp_path: Path) -> None:
    fips_file = tmp_path / "fips.txt"
    fips_file.write_bytes("﻿51059\n".encode("utf-8"))
    assert load_fips_codes(None, fips_file) == ["51059"]


def test_load_fips_codes_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FIPS file not found"):
        load_fips_codes(["51059"], tmp_path / "missing.txt")


def test_load_fips_codes_bad_entries_raise_with_source(tmp_path: Path) -> None:
    fips_file = tmp_path / "fips.txt"
    fips_file.write_text("51059\nFairfax\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_fips_codes(["510590"], fips_file)
    message = str(excinfo.value)
    assert "'510590' (inline list)" in message
    assert "'Fairfax' (fips.txt line 2)" in message
