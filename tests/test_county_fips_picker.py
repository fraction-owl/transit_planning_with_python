"""Tests for scripts/national_data_tools/county_fips_picker.py.

The reference files are small synthetic stand-ins laid out like the Census
originals: a delineation workbook with title rows above the header and notes
below the data, and a pipe-delimited adjacency file with self rows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from scripts.national_data_tools import county_fips_picker as mod
from utils.fips_helpers import load_fips_codes

WASHINGTON = ("47900", "Washington-Arlington-Alexandria, DC-VA-MD-WV", "Metropolitan")
BALTIMORE = ("12580", "Baltimore-Columbia-Towson, MD", "Metropolitan")
BLACKSBURG = ("13980", "Blacksburg-Christiansburg-Radford, VA", "Metropolitan")
LEXINGTON = ("30500", "California-Lexington Park, MD", "Micropolitan")

# (CBSA, county name, state name, state FIPS, county FIPS)
METRO_ROWS = (
    (WASHINGTON, "District of Columbia", "District of Columbia", "11", "001"),
    (WASHINGTON, "Montgomery County", "Maryland", "24", "031"),
    (WASHINGTON, "Prince George's County", "Maryland", "24", "033"),
    (WASHINGTON, "Arlington County", "Virginia", "51", "013"),
    (WASHINGTON, "Fairfax County", "Virginia", "51", "059"),
    (WASHINGTON, "Fairfax city", "Virginia", "51", "600"),
    (WASHINGTON, "Jefferson County", "West Virginia", "54", "037"),
    (BALTIMORE, "Baltimore County", "Maryland", "24", "005"),
    (BALTIMORE, "Howard County", "Maryland", "24", "027"),
    (BALTIMORE, "Baltimore city", "Maryland", "24", "510"),
    (BLACKSBURG, "Montgomery County", "Virginia", "51", "121"),
    (LEXINGTON, "St. Mary's County", "Maryland", "24", "037"),
)

# Shared borders; Page County, VA (51139) is in no metro or micropolitan area.
BORDERS = (
    ("51059", "51600"),
    ("51059", "51013"),
    ("51059", "24031"),
    ("51013", "11001"),
    ("11001", "24031"),
    ("11001", "24033"),
    ("24031", "24027"),
    ("24027", "24005"),
    ("24005", "24510"),
    ("24033", "24037"),
    ("54037", "51139"),
)
ADJACENCY_NAMES = {
    "11001": "District of Columbia, DC",
    "24005": "Baltimore County, MD",
    "24027": "Howard County, MD",
    "24031": "Montgomery County, MD",
    "24033": "Prince George's County, MD",
    "24037": "St. Mary's County, MD",
    "24510": "Baltimore city, MD",
    "51013": "Arlington County, VA",
    "51059": "Fairfax County, VA",
    "51121": "Montgomery County, VA",
    "51139": "Page County, VA",
    "51600": "Fairfax city, VA",
    "54037": "Jefferson County, WV",
}


def _write_metro_xlsx(path: Path) -> Path:
    header = list(mod.METRO_COLUMNS.values()) + ["State Name"]
    rows: list[list[object]] = [
        ["Table with row headers in column A and column headers in row 3"],
        ["List 1. CORE BASED STATISTICAL AREAS (CBSAs), JULY 2023"],
        header,
    ]
    for (code, title, kind), name, state_name, state, county in METRO_ROWS:
        rows.append([code, title, f"{kind} Statistical Area", name, state, county, state_name])
    rows += [[None], ["Note: synthetic test data."], ["Source: tests."]]
    width = max(len(r) for r in rows)
    frame = pd.DataFrame([r + [None] * (width - len(r)) for r in rows])
    frame.to_excel(path, header=False, index=False)
    return path


def _write_adjacency(path: Path) -> Path:
    lines = ["County Name|County GEOID|Neighbor Name|Neighbor GEOID"]
    pairs = {(a, a) for a in ADJACENCY_NAMES}
    pairs |= set(BORDERS) | {(b, a) for a, b in BORDERS}
    for a, b in sorted(pairs):
        lines.append(f"{ADJACENCY_NAMES[a]}|{a}|{ADJACENCY_NAMES[b]}|{b}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def metro_file(tmp_path: Path) -> Path:
    return _write_metro_xlsx(tmp_path / "list1_2023.xlsx")


@pytest.fixture()
def adjacency_file(tmp_path: Path) -> Path:
    return _write_adjacency(tmp_path / "county_adjacency2024.txt")


@pytest.fixture()
def ref(metro_file: Path, adjacency_file: Path) -> mod.Reference:
    names, neighbors = mod.load_county_adjacency(adjacency_file)
    return mod.build_reference(mod.load_metro_delineation(metro_file), names, neighbors)


@pytest.fixture()
def metro_only_ref(metro_file: Path) -> mod.Reference:
    return mod.build_reference(mod.load_metro_delineation(metro_file))


def scripted(*answers: str) -> mod.Ask:
    """Return an ask() that replays *answers* and fails if more are requested."""
    queue: Iterator[str] = iter(answers)

    def ask(prompt: str) -> str:
        try:
            return next(queue)
        except StopIteration:
            pytest.fail(f"Unexpected question: {prompt!r}")

    return ask


# ---------------------------------------------------------------------------
# Reference files
# ---------------------------------------------------------------------------


def test_load_metro_delineation_skips_title_and_note_rows(metro_file: Path) -> None:
    metro = mod.load_metro_delineation(metro_file)
    assert len(metro) == len(METRO_ROWS)
    row = metro.set_index("fips").loc["51600"]
    assert row["name"] == "Fairfax city"
    assert row["cbsa_code"] == "47900"
    assert row["cbsa_kind"] == "metro area"
    assert metro.set_index("fips").loc["24037", "cbsa_kind"] == "micropolitan area"


def test_load_metro_delineation_zero_pads_numeric_codes(tmp_path: Path) -> None:
    path = tmp_path / "list1.csv"
    path.write_text(
        "title row\n"
        + ",".join(mod.METRO_COLUMNS.values())
        + "\n10100,Aberdeen SD,Micropolitan Statistical Area,Brown County,46,13\n"
        + "11500,Anniston AL,Metropolitan Statistical Area,Calhoun County,1,15\n",
        encoding="utf-8",
    )
    assert mod.load_metro_delineation(path)["fips"].tolist() == ["46013", "01015"]


def test_load_metro_delineation_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CBSA Code"):
        mod.load_metro_delineation(path)


def test_load_county_adjacency_reads_codes_names_and_drops_self_rows(
    adjacency_file: Path,
) -> None:
    names, neighbors = mod.load_county_adjacency(adjacency_file)
    assert names["51059"] == "Fairfax County"
    assert neighbors["51059"] == {"51600", "51013", "24031"}
    assert "51059" not in neighbors["51059"]
    assert neighbors["51121"] == set()


def test_load_county_adjacency_rejects_2010_layout(tmp_path: Path) -> None:
    path = tmp_path / "county_adjacency2010.txt"
    path.write_text('"Autauga County, AL"\t01001\t"Autauga County, AL"\t01001\n')
    with pytest.raises(ValueError, match="pipe-delimited"):
        mod.load_county_adjacency(path)


def test_build_reference_adds_counties_outside_metro_areas(ref: mod.Reference) -> None:
    assert ref.counties["51139"].name == "Page County"
    assert ref.counties["51139"].cbsa_code == ""
    assert ref.cbsa_members["47900"] == ["11001", "24031", "24033", "51013", "51059"] + [
        "51600",
        "54037",
    ]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_parse_states_accepts_names_abbreviations_and_codes() -> None:
    assert mod.parse_states(["va, Maryland", "11", "VA"]) == (["51", "24", "11"], [])
    assert mod.parse_states(["VA, Narnia"]) == (["51"], ["Narnia"])


@pytest.mark.parametrize(
    ("entry", "states", "expected"),
    [
        ("Fairfax", ["51"], ["51600", "51059"]),
        ("Fairfax County", ["51"], ["51059"]),
        ("fairfax city", [], ["51600"]),
        ("Montgomery County", ["24", "51"], ["24031", "51121"]),
        ("Montgomery County, VA", ["24"], ["51121"]),
        ("Montgomery, Maryland", [], ["24031"]),
        ("Prince Georges", ["24"], ["24033"]),
        ("Saint Mary's County", ["24"], ["24037"]),
        ("Fairfax County", ["24"], []),
    ],
)
def test_find_matches(
    ref: mod.Reference, entry: str, states: list[str], expected: list[str]
) -> None:
    assert mod.find_matches(entry, ref, states)[0] == expected


def test_resolve_entry_ambiguous_name_is_skipped_with_warning(
    ref: mod.Reference, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert mod.resolve_entry("Baltimore", ref, ["24"], None) == []
    assert "matches 2 jurisdictions" in caplog.text
    assert "'Baltimore city, MD'" in caplog.text


def test_resolve_entry_ambiguous_name_asks_which(ref: mod.Reference) -> None:
    assert mod.resolve_entry("Baltimore", ref, ["24"], scripted("2")) == ["24005"]


def test_resolve_entry_typo_suggests_close_matches(
    ref: mod.Reference, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert mod.resolve_entry("Arlingten County", ref, ["51"], None) == []
    assert "No jurisdiction matches 'Arlingten County'" in caplog.text
    assert "Arlington County, VA (51013)" in caplog.text


def test_resolve_entry_out_of_state_match_is_reported(
    ref: mod.Reference, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert mod.resolve_entry("Jefferson County", ref, ["51"], None) == []
    assert "Jefferson County, WV (54037)" in caplog.text


def test_resolve_entry_rural_county_needs_adjacency_file(
    metro_only_ref: mod.Reference, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert mod.resolve_entry("Page County", metro_only_ref, ["51"], None) == []
    assert "COUNTY_ADJACENCY_FILE" in caplog.text


def test_resolve_entry_keeps_unknown_fips_with_warning(
    ref: mod.Reference, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert mod.resolve_entry("1001", ref, [], None) == ["01001"]
    assert "01001 is not in the reference files" in caplog.text


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("", []), ("all", [0, 1, 2, 3]), ("1,3", [0, 2]), ("2-4", [1, 2, 3]), ("1 1", [0])],
)
def test_parse_picks(answer: str, expected: list[int]) -> None:
    assert mod.parse_picks(answer, 4) == expected


@pytest.mark.parametrize("answer", ["0", "5", "x", "3-1", "1-2-3"])
def test_parse_picks_rejects_bad_answers(answer: str) -> None:
    assert mod.parse_picks(answer, 4) is None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _select(ref: mod.Reference, **overrides: object) -> list[str]:
    settings: dict[str, object] = {
        "states": ["VA"],
        "core": ["Fairfax County"],
        "metro_scope": "none",
        "adjacent_scope": "none",
        "extras": [],
        "ask": None,
    }
    settings.update(overrides)
    selected, _answers = mod.select_jurisdictions(ref, **settings)  # type: ignore[arg-type]
    return sorted(selected)


def test_metro_scope_all_adds_every_other_jurisdiction(ref: mod.Reference) -> None:
    assert _select(ref, metro_scope="all") == [
        "11001",
        "24031",
        "24033",
        "51013",
        "51059",
        "51600",
        "54037",
    ]


def test_metro_scope_state_keeps_only_chosen_states(ref: mod.Reference) -> None:
    assert _select(ref, metro_scope="state") == ["51013", "51059", "51600"]
    assert _select(ref, states=["VA", "DC"], metro_scope="state") == [
        "11001",
        "51013",
        "51059",
        "51600",
    ]


def test_metro_scope_state_uses_core_state_when_states_empty(ref: mod.Reference) -> None:
    assert _select(ref, states=[], core=["51059"], metro_scope="state") == [
        "51013",
        "51059",
        "51600",
    ]


def test_adjacent_scope_all_adds_bordering_jurisdictions(ref: mod.Reference) -> None:
    assert _select(ref, adjacent_scope="all") == ["24031", "51013", "51059", "51600"]


def test_ask_without_answerer_adds_nothing(ref: mod.Reference) -> None:
    assert _select(ref, metro_scope="ask", adjacent_scope="ask") == ["51059"]


def test_unrecognized_states_or_core_raise(ref: mod.Reference) -> None:
    with pytest.raises(mod.SelectionError, match="STATES"):
        _select(ref, states=["Narnia"])
    with pytest.raises(mod.SelectionError, match="No core jurisdiction"):
        _select(ref, core=["Fairfax"])  # ambiguous, nobody to ask


def test_interactive_flow_declines_all_then_takes_state_subset(ref: mod.Reference) -> None:
    ask = scripted(
        "VA, DC",  # states
        "Fairfax",  # core: ambiguous ...
        "2",  # ... Fairfax County
        "",  # core done
        "n",  # all 6 others in the metro? no
        "y",  # only the 3 in VA, DC? yes
        "y",  # proximity suggestions? yes
        "1",  # Montgomery County, MD
        "Falls Church",  # explicit: no match, warned
        "24510",  # explicit FIPS
        "",  # done
    )
    selected, answers = mod.select_jurisdictions(
        ref,
        states=[],
        core=[],
        metro_scope="ask",
        adjacent_scope="ask",
        extras=[],
        ask=ask,
    )
    assert selected == {
        "51059": mod.REASON_CORE,
        "11001": mod.REASON_METRO,
        "51013": mod.REASON_METRO,
        "51600": mod.REASON_METRO,
        "24031": mod.REASON_ADJACENT,
        "24510": mod.REASON_EXTRA,
    }
    assert answers["states"] == ["VA", "DC"]
    assert answers["core"] == ["51059"]


def test_state_question_skipped_when_metro_is_all_in_chosen_states(ref: mod.Reference) -> None:
    # Every other Baltimore-area jurisdiction is in MD, so "only those in MD?" is
    # the same question and is not asked again.
    _select(ref, states=["MD"], core=["24005"], metro_scope="ask", ask=scripted("n", ""))


def test_micropolitan_reason(ref: mod.Reference) -> None:
    selected, _ = mod.select_jurisdictions(
        ref,
        states=["MD"],
        core=["24033"],
        metro_scope="none",
        adjacent_scope="all",
        extras=[],
    )
    assert selected["24037"] == mod.REASON_ADJACENT
    assert mod.REASON_MICRO == "same micropolitan area"


def test_ct_vintage_warnings() -> None:
    assert mod.ct_vintage_warnings(["51059"]) == []
    assert "planning regions" in mod.ct_vintage_warnings(["09110"])[0]
    assert "former Connecticut counties" in mod.ct_vintage_warnings(["09003"])[0]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_writes_fips_file_the_other_scripts_can_read(
    metro_file: Path, adjacency_file: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    argv = [
        "--metro-file",
        str(metro_file),
        "--adjacency-file",
        str(adjacency_file),
        "--output-dir",
        str(out_dir),
        "--states",
        "VA",
        "DC",
        "--core",
        "Fairfax County",
        "--metro-scope",
        "state",
        "--adjacent-scope",
        "none",
        "--extra",
        "Baltimore County, MD",
        "--no-prompt",
    ]
    assert mod.main(argv) == 0
    fips_file = out_dir / mod.OUTPUT_FILENAME
    assert load_fips_codes(None, fips_file) == ["11001", "24005", "51013", "51059", "51600"]
    text = fips_file.read_text(encoding="utf-8")
    assert "51059  # Fairfax County, VA - core jurisdiction" in text
    assert "24005  # Baltimore County, MD - added by name or FIPS" in text

    run_log = (out_dir / "planning_area_fips_runlog.txt").read_text(encoding="utf-8")
    assert "CORE_JURISDICTIONS: List[str] = ['51059']" in run_log
    assert "EXTRA_JURISDICTIONS: List[str] = ['11001', '51013', '51600', '24005']" in run_log
    assert mod.CONFIG_BEGIN_MARKER in run_log


def test_main_uses_answers_from_ask(metro_file: Path, tmp_path: Path) -> None:
    argv = ["--metro-file", str(metro_file), "--output-dir", str(tmp_path)]
    ask = scripted("MD", "Baltimore city", "", "y", "")
    assert mod.main(argv, ask=ask) == 0
    assert load_fips_codes(None, tmp_path / mod.OUTPUT_FILENAME) == ["24005", "24027", "24510"]


def test_main_config_errors_return_2(metro_file: Path, tmp_path: Path) -> None:
    assert mod.main([]) == 2  # placeholders
    missing = ["--metro-file", str(tmp_path / "nope.xlsx"), "--output-dir", str(tmp_path)]
    assert mod.main(missing) == 2
    nothing = ["--metro-file", str(metro_file), "--output-dir", str(tmp_path), "--no-prompt"]
    assert mod.main(nothing) == 2
    assert not (tmp_path / mod.OUTPUT_FILENAME).exists()
