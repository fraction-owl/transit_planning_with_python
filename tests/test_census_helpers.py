"""Tests for utils/census_helpers.py: reading county FIPS codes from a text file."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from utils import census_helpers as ch

FIPS_FIXTURE = Path("tests/fixtures/county_fips.txt")
FIPS_EXAMPLE = Path("sample_data/county_fips_example.txt")

# The Washington, DC area counties in the uscensus_* scripts' inline FIPS lists.
DC_AREA_FIPS = {
    "11001",
    "24031",
    "24033",
    "51013",
    "51059",
    "51107",
    "51153",
    "51510",
    "51600",
    "51610",
    "51683",
    "51685",
}


# --- load_fips_file ---------------------------------------------------------------


def test_fixture_loads_codes_in_order_without_repeats() -> None:
    assert ch.load_fips_file(FIPS_FIXTURE) == ["11001", "24031", "51059"]


def test_example_file_lists_no_codes_as_shipped() -> None:
    # Every code in the sample is commented out, so pointing a script at it unedited
    # stops the run rather than filtering to the DC area by mistake.
    with pytest.raises(ValueError, match="lists no codes"):
        ch.load_fips_file(FIPS_EXAMPLE)


def test_example_file_uncommented_lists_the_dc_area_counties(tmp_path: Path) -> None:
    text = FIPS_EXAMPLE.read_text(encoding="utf-8")
    uncommented = re.sub(r"(?m)^# (\d{5}\b)", r"\1", text)
    path = tmp_path / "county_fips.txt"
    path.write_text(uncommented, encoding="utf-8")
    assert set(ch.load_fips_file(path)) == DC_AREA_FIPS


def test_reads_fcc_style_lines_and_a_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "fips.txt"
    path.write_text(
        "      11001        District of Columbia\n\t51510\tAlexandria city\n",
        encoding="utf-8-sig",
    )
    assert ch.load_fips_file(path) == ["11001", "51510"]


@pytest.mark.parametrize("bad", ["01", "1001", "110010", "11001,", "1100A", "١١٠٠١"])
def test_rejects_a_line_that_does_not_start_with_a_five_digit_code(
    tmp_path: Path, bad: str
) -> None:
    path = tmp_path / "fips.txt"
    path.write_text(f"11001\n{bad}  Somewhere\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2: expected a 5-digit county FIPS code"):
        ch.load_fips_file(path)


def test_reports_every_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "fips.txt"
    path.write_text("state-level\n11001\nFIPS code  name\n", encoding="utf-8")
    with pytest.raises(ValueError) as err:
        ch.load_fips_file(path)
    assert "line 1:" in str(err.value)
    assert "line 3:" in str(err.value)


def test_missing_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read the FIPS file"):
        ch.load_fips_file(tmp_path / "missing.txt")


# --- resolve_fips_filter ----------------------------------------------------------


@pytest.mark.parametrize("unset", [None, "", "   "])
def test_resolve_uses_inline_list_when_no_file(unset: str | None) -> None:
    assert ch.resolve_fips_filter(["11001", "51059"], unset) == ["11001", "51059"]
    assert ch.resolve_fips_filter(None, unset) == []


def test_resolve_prefers_the_file_and_warns_about_the_inline_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        codes = ch.resolve_fips_filter(["51013"], FIPS_FIXTURE)
    assert codes == ["11001", "24031", "51059"]
    assert "ignoring the inline list" in caplog.text


def test_resolve_with_only_a_file_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert ch.resolve_fips_filter([], str(FIPS_FIXTURE)) == ["11001", "24031", "51059"]
    assert caplog.text == ""


def test_resolve_raises_when_the_file_is_unusable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read the FIPS file"):
        ch.resolve_fips_filter(["11001"], tmp_path / "missing.txt")
