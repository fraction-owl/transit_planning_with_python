"""Build a county FIPS list for a planning area, with metro-area and nearby suggestions.

Asks which state(s) the planning area is in and which county or county
equivalent (independent city, parish, borough, Connecticut planning region, ...)
is at its core. It then offers the other jurisdictions in the same metropolitan
or micropolitan area: first all of them, and if not, only those in the chosen
states. It can also suggest jurisdictions that share a border with the
selection, and takes any others by name or FIPS code. A name that matches
nothing, or more than one jurisdiction, is reported.

Questions are asked only for settings left empty (or "ask"), and only when the
script runs in a terminal or notebook. Fill the settings in, or pass the flags,
to run without questions.

Inputs
------
- Census metro-area delineation workbook (``list1_2023.xlsx`` or later): every
  metropolitan and micropolitan area and the counties in it.
- Optional Census county adjacency file (``county_adjacency2024.txt`` or later),
  needed for suggestions by proximity and to find, by name, counties outside
  every metropolitan or micropolitan area.

Outputs
-------
- ``planning_area_fips.txt`` (``OUTPUT_FILENAME``): one 5-digit FIPS code per
  line, with the jurisdiction's name and why it was added as a comment.
- ``planning_area_fips_runlog.txt``: run-log sidecar with the verbatim
  CONFIGURATION block and the answers given, ready to paste back in to repeat
  the run without questions.

Notes:
    Use reference files from the same period as the Census data being filtered.
    In 2022 Connecticut's planning regions (09110-09190) replaced its counties
    (09001-09015) as county equivalents: the 2023 delineation and ACS 2022+ use
    the regions, while 2020 Census blocks and tables use the counties. The
    script warns when the list holds either.

See Also:
    - ``uscensus_table_build.py``
    - ``uscensus_tiger_join_gpd.py``
    - ``uscensus_tiger_join_arcpy.py``
    - ``gtfs_service_demographics_gpd.py``

Helpful links
-------------
    https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
    https://www.census.gov/geographies/reference-files/time-series/geo/county-adjacency.html

Typical usage
-------------
Set METRO_DELINEATION_FILE and OUTPUT_DIR in the CONFIGURATION section (or pass
``--metro-file`` / ``--output-dir``), run from a shell, ArcGIS Pro's Python
window, or a Jupyter notebook, and answer the questions. Point the other
scripts' FIPS file settings at the output.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import logging
import re
import sys
import unicodedata
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# === BEGIN CONFIG ===

# Census metro-area delineation workbook (.xlsx, .xls, or .csv), e.g. list1_2023.xlsx.
METRO_DELINEATION_FILE: str = r"Path\To\Your\list1_2023.xlsx"  # <<< EDIT ME

# Optional Census county adjacency file, e.g. county_adjacency2024.txt. Needed for
# suggestions by proximity and to find, by name, counties outside every metro or
# micropolitan area. None to skip.
COUNTY_ADJACENCY_FILE: Optional[str] = None

OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"  # <<< EDIT ME
OUTPUT_FILENAME: str = r"planning_area_fips.txt"

# ---- Answers: leave empty (or "ask") to be asked when the script runs ------
# State(s) the planning area is in: names, abbreviations, or 2-digit FIPS codes,
# e.g. ["VA", "MD", "DC"]. Names below are looked up in these states; empty =
# every state.
STATES: List[str] = []

# Core jurisdiction(s): names such as "Fairfax County" or "Fairfax County, VA",
# or 5-digit FIPS codes.
CORE_JURISDICTIONS: List[str] = []

# Other jurisdictions in the core's metropolitan or micropolitan area:
#   "all"   - add all of them
#   "state" - add those in STATES (the core's own state when STATES is empty)
#   "none"  - add none
#   "ask"   - ask; treated as "none" when there is no one to ask
METRO_SCOPE: str = "ask"

# Jurisdictions that share a border with the selection (needs
# COUNTY_ADJACENCY_FILE): "all", "none", or "ask" to pick from a numbered list.
ADJACENT_SCOPE: str = "ask"

# Other jurisdictions to add, by name or 5-digit FIPS code. Left empty, the
# script asks for them.
EXTRA_JURISDICTIONS: List[str] = []

# When True, a failed run-log write is treated as a failure (exit code 1).
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL: int = logging.INFO

# === END CONFIG ===

_DEFAULT_METRO_DELINEATION_FILE = r"Path\To\Your\list1_2023.xlsx"
_DEFAULT_OUTPUT_DIR = r"Path\To\Your\Output_Folder"

SCOPE_CHOICES: Tuple[str, ...] = ("all", "state", "none", "ask")
ADJACENT_CHOICES: Tuple[str, ...] = ("all", "none", "ask")

# Delineation-file headers this script reads, keyed by the internal name.
METRO_COLUMNS: Dict[str, str] = {
    "cbsa_code": "CBSA Code",
    "cbsa_title": "CBSA Title",
    "cbsa_type": "Metropolitan/Micropolitan Statistical Area",
    "name": "County/County Equivalent",
    "state_fips": "FIPS State Code",
    "county_fips": "FIPS County Code",
}
ADJACENCY_COLUMNS: Tuple[str, ...] = (
    "County Name",
    "County GEOID",
    "Neighbor Name",
    "Neighbor GEOID",
)

# (FIPS, USPS abbreviation, name) for the states, DC, and the territories.
STATES_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("01", "AL", "Alabama"),
    ("02", "AK", "Alaska"),
    ("04", "AZ", "Arizona"),
    ("05", "AR", "Arkansas"),
    ("06", "CA", "California"),
    ("08", "CO", "Colorado"),
    ("09", "CT", "Connecticut"),
    ("10", "DE", "Delaware"),
    ("11", "DC", "District of Columbia"),
    ("12", "FL", "Florida"),
    ("13", "GA", "Georgia"),
    ("15", "HI", "Hawaii"),
    ("16", "ID", "Idaho"),
    ("17", "IL", "Illinois"),
    ("18", "IN", "Indiana"),
    ("19", "IA", "Iowa"),
    ("20", "KS", "Kansas"),
    ("21", "KY", "Kentucky"),
    ("22", "LA", "Louisiana"),
    ("23", "ME", "Maine"),
    ("24", "MD", "Maryland"),
    ("25", "MA", "Massachusetts"),
    ("26", "MI", "Michigan"),
    ("27", "MN", "Minnesota"),
    ("28", "MS", "Mississippi"),
    ("29", "MO", "Missouri"),
    ("30", "MT", "Montana"),
    ("31", "NE", "Nebraska"),
    ("32", "NV", "Nevada"),
    ("33", "NH", "New Hampshire"),
    ("34", "NJ", "New Jersey"),
    ("35", "NM", "New Mexico"),
    ("36", "NY", "New York"),
    ("37", "NC", "North Carolina"),
    ("38", "ND", "North Dakota"),
    ("39", "OH", "Ohio"),
    ("40", "OK", "Oklahoma"),
    ("41", "OR", "Oregon"),
    ("42", "PA", "Pennsylvania"),
    ("44", "RI", "Rhode Island"),
    ("45", "SC", "South Carolina"),
    ("46", "SD", "South Dakota"),
    ("47", "TN", "Tennessee"),
    ("48", "TX", "Texas"),
    ("49", "UT", "Utah"),
    ("50", "VT", "Vermont"),
    ("51", "VA", "Virginia"),
    ("53", "WA", "Washington"),
    ("54", "WV", "West Virginia"),
    ("55", "WI", "Wisconsin"),
    ("56", "WY", "Wyoming"),
    ("60", "AS", "American Samoa"),
    ("66", "GU", "Guam"),
    ("69", "MP", "Northern Mariana Islands"),
    ("72", "PR", "Puerto Rico"),
    ("78", "VI", "U.S. Virgin Islands"),
)
STATE_ABBR: Dict[str, str] = {fips: abbr for fips, abbr, _ in STATES_TABLE}

# Name endings stripped so "Fairfax" finds both Fairfax County and Fairfax city.
# Longer endings come first ("city and borough" before "borough").
NAME_SUFFIXES: Tuple[str, ...] = (
    " city and borough",
    " census area",
    " planning region",
    " municipality",
    " municipio",
    " borough",
    " county",
    " parish",
    " city",
)

# Connecticut county equivalents before and after the 2022 change.
CT_PLANNING_REGIONS: Set[str] = {f"091{n}0" for n in range(1, 10)}
CT_FORMER_COUNTIES: Set[str] = {f"090{n:02d}" for n in range(1, 16, 2)}

# Reasons recorded in the output file's comments.
REASON_CORE = "core jurisdiction"
REASON_METRO = "same metro area"
REASON_MICRO = "same micropolitan area"
REASON_ADJACENT = "shares a border"
REASON_EXTRA = "added by name or FIPS"

Ask = Callable[[str], str]

# =============================================================================
# HELPERS (canonical copies — see utils/)
# =============================================================================


def notebook_safe_argv(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Return the argv to parse, shielding notebook kernels from stray flags.

    When a script's ``main()`` runs with no explicit ``argv`` inside a
    Jupyter/IPython kernel, ``sys.argv`` holds kernel plumbing (for example
    ``-f /path/kernel.json``) rather than flags meant for the script, and
    strict ``argparse.parse_args`` would reject it and abort.  This helper
    detects the notebook case and substitutes an empty argument list so the
    CONFIGURATION constants stay in charge, while shell runs keep strict
    parsing (a typo in a flag fails loudly instead of being silently ignored).

    Canonical implementation: ``utils/cli_helpers.py``.

    Args:
        argv: Explicit argument list passed to ``main()``, or ``None`` to
            fall back to ``sys.argv``.

    Returns:
        ``list(argv)`` when *argv* was provided; ``[]`` when running inside a
        notebook kernel; otherwise ``None`` so argparse reads ``sys.argv[1:]``.
    """
    if argv is not None:
        return list(argv)
    if "ipykernel" in sys.modules:
        return []
    return None


def stdin_is_interactive() -> bool:
    """Return True when ``input()`` can reach a live user.

    True inside a Jupyter/IPython kernel (ipykernel routes ``input()`` to a
    notebook prompt widget) or when stdin is a real terminal. False under
    captured or redirected stdin — CI runners, orchestrator pipelines, cron —
    where an ``input()`` call would hang or crash rather than guide anyone.
    Scripts use this to decide between prompting for missing configuration
    (guided setup) and failing fast with an exit code.

    Canonical implementation: ``utils/cli_helpers.py``.

    Returns:
        True when prompting a user is possible, False otherwise.
    """
    if "ipykernel" in sys.modules:
        return True
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def extract_config_block(source_file: Path) -> str:
    r"""Return the text between the CONFIG markers in *source_file*.

    Reads ``source_file`` as UTF-8 text and slices out the lines strictly
    *between* the first occurrence of ``# === BEGIN CONFIG ===`` and the first
    subsequent occurrence of ``# === END CONFIG ===``.  The marker lines
    themselves are excluded; whitespace and inline comments inside the block
    are preserved verbatim.

    Args:
        source_file: Path to the Python source file to scan (typically
            ``Path(__file__)`` from the calling script).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    _BEGIN = "# === BEGIN CONFIG ==="
    _END = "# === END CONFIG ==="

    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == _BEGIN:
            begin_idx = i
        elif begin_idx is not None and stripped == _END:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. Expected '{_BEGIN}' and '{_END}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


# =============================================================================
# REFERENCE DATA
# =============================================================================


class SelectionError(ValueError):
    """A setting or answer that leaves nothing to select (exit code 2)."""


class County(NamedTuple):
    """One county or county equivalent from the reference files."""

    fips: str
    name: str
    state_fips: str
    cbsa_code: str = ""
    cbsa_title: str = ""
    cbsa_kind: str = ""  # "metro area", "micropolitan area", or ""


class Reference:
    """Counties, their metro areas, and (optionally) which counties share a border."""

    def __init__(self, counties: Dict[str, County], neighbors: Dict[str, Set[str]]) -> None:
        """Store the lookups and index metro-area membership.

        Args:
            counties: County records keyed by 5-digit FIPS code.
            neighbors: FIPS code -> FIPS codes sharing a border. Empty when no
                adjacency file was loaded.
        """
        self.counties = counties
        self.neighbors = neighbors
        self.has_adjacency = bool(neighbors)
        self.cbsa_members: Dict[str, List[str]] = {}
        for fips in sorted(counties):
            code = counties[fips].cbsa_code
            if code:
                self.cbsa_members.setdefault(code, []).append(fips)

    def label(self, fips: str) -> str:
        """Return e.g. ``Fairfax County, VA (51059)``, or the bare code if unknown."""
        county = self.counties.get(fips)
        if county is None:
            return f"{fips} (not in the reference files)"
        return f"{county.name}, {state_label(county.state_fips)} ({fips})"

    def sort_key(self, fips: str) -> Tuple[str, str]:
        """Sort by state abbreviation, then name."""
        county = self.counties.get(fips)
        if county is None:
            return ("~", fips)
        return (state_label(county.state_fips), county.name.casefold())


def state_label(state_fips: str) -> str:
    """Return the USPS abbreviation for a 2-digit state code, or the code itself."""
    return STATE_ABBR.get(state_fips, state_fips)


def _clean_codes(series: pd.Series, width: int) -> pd.Series:
    """Strip, drop a trailing ``.0`` (codes read as numbers), and zero-pad digit codes."""
    text = series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    is_code = text.str.fullmatch(r"\d+")
    return text.where(~is_code, text.str.zfill(width))


def load_metro_delineation(path: Path) -> pd.DataFrame:
    """Read the Census metro-area delineation file into one row per county.

    The Census workbook has title rows above its header row, and notes below
    the data, so the header is found by searching for ``CBSA Code`` and rows
    without a numeric state and county code are dropped.

    Args:
        path: ``.xlsx``/``.xls`` workbook or a ``.csv`` saved from it.

    Returns:
        Columns ``fips``, ``name``, ``state_fips``, ``cbsa_code``, ``cbsa_title``
        and ``cbsa_kind``.

    Raises:
        ValueError: If no header row is found or required columns are missing.
    """
    if path.suffix.lower() == ".csv":
        # The title and note rows have fewer fields than the table, so read with
        # the csv module and pad, rather than with pandas' fixed-width parser.
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        rows = list(csv.reader(text.splitlines()))
        width = max((len(row) for row in rows), default=0)
        raw = pd.DataFrame([row + [""] * (width - len(row)) for row in rows], dtype=str)
    else:
        raw = pd.read_excel(path, header=None, dtype=str)

    header_rows = [
        i for i in range(len(raw)) if raw.iloc[i].astype(str).str.strip().eq("CBSA Code").any()
    ]
    if not header_rows:
        raise ValueError(
            f"{path.name} has no 'CBSA Code' header row. Is it the Census metro-area "
            "delineation file (list1_2023.xlsx or similar)?"
        )
    header = header_rows[0]
    table = raw.iloc[header + 1 :].copy()
    table.columns = [str(c).strip() for c in raw.iloc[header]]

    missing = [col for col in METRO_COLUMNS.values() if col not in table.columns]
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")

    table = table.rename(columns={v: k for k, v in METRO_COLUMNS.items()})
    table["state_fips"] = _clean_codes(table["state_fips"], 2)
    table["county_fips"] = _clean_codes(table["county_fips"], 3)
    table = table[
        table["state_fips"].str.fullmatch(r"\d{2}") & table["county_fips"].str.fullmatch(r"\d{3}")
    ].copy()

    kind = table["cbsa_type"].fillna("").str.strip()
    table["cbsa_kind"] = kind.str.lower()
    table.loc[kind.str.startswith("Metropolitan"), "cbsa_kind"] = "metro area"
    table.loc[kind.str.startswith("Micropolitan"), "cbsa_kind"] = "micropolitan area"
    table["fips"] = table["state_fips"] + table["county_fips"]
    for col in ("name", "cbsa_code", "cbsa_title"):
        table[col] = table[col].fillna("").astype(str).str.strip()

    out = table[["fips", "name", "state_fips", "cbsa_code", "cbsa_title", "cbsa_kind"]]
    logging.info(
        "Read %d counties in %d metro and micropolitan areas from %s.",
        len(out),
        out["cbsa_code"].nunique(),
        path.name,
    )
    return out.drop_duplicates("fips").reset_index(drop=True)


def load_county_adjacency(path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Read the Census county adjacency file (2023+ pipe-delimited layout).

    Args:
        path: ``county_adjacency2024.txt`` or a later edition.

    Returns:
        ``(names, neighbors)``: county name (without the ``, ST`` suffix) by FIPS
        code for every county in the file, and the FIPS codes sharing a border
        with each county (a county is never its own neighbor).

    Raises:
        ValueError: If the file does not have the expected columns.
    """
    try:
        table = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        table = pd.read_csv(path, sep="|", dtype=str, encoding="latin-1")
    table.columns = [str(c).strip() for c in table.columns]
    missing = [col for col in ADJACENCY_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing column(s): {', '.join(missing)}. The script reads the "
            "pipe-delimited layout of the 2023 and later adjacency files."
        )

    names: Dict[str, str] = {}
    neighbors: Dict[str, Set[str]] = {}
    rows = table[list(ADJACENCY_COLUMNS)].itertuples(index=False, name=None)
    for county_name, county, neighbor_name, neighbor in rows:
        if not isinstance(county, str) or not isinstance(neighbor, str):
            continue
        county, neighbor = county.strip().zfill(5), neighbor.strip().zfill(5)
        for code, label in ((county, county_name), (neighbor, neighbor_name)):
            if code not in names and isinstance(label, str):
                names[code] = label.rsplit(",", 1)[0].strip()
        neighbors.setdefault(county, set())
        if neighbor != county:
            neighbors[county].add(neighbor)
    logging.info("Read borders for %d counties from %s.", len(neighbors), path.name)
    return names, neighbors


def build_reference(
    metro: pd.DataFrame,
    adjacency_names: Optional[Dict[str, str]] = None,
    neighbors: Optional[Dict[str, Set[str]]] = None,
) -> Reference:
    """Combine the delineation rows and the adjacency file into one lookup.

    Counties come from the adjacency file (which lists every county) when it is
    loaded, plus every county in the delineation file; metro-area details and
    names come from the delineation file where it has them.
    """
    counties: Dict[str, County] = {
        fips: County(fips=fips, name=name, state_fips=fips[:2])
        for fips, name in (adjacency_names or {}).items()
    }
    for record in metro.to_dict("records"):
        values = {field: str(record[field]) for field in County._fields}
        counties[values["fips"]] = County(**values)
    return Reference(counties, neighbors or {})


# =============================================================================
# MATCHING
# =============================================================================


def normalize_name(text: str) -> str:
    """Lower-case, drop accents, periods and apostrophes, and spell "saint" as "st"."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "").replace("'", "").replace(".", " ")
    text = re.sub(r"\bsaint\b", "st", text)
    return " ".join(text.split())


def base_name(normalized: str) -> str:
    """Strip one county-type ending ("county", "city", "parish", ...)."""
    for suffix in NAME_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            return normalized[: -len(suffix)]
    return normalized


def resolve_state(text: str) -> Optional[str]:
    """Return the 2-digit state code for a name, abbreviation, or code, else None."""
    token = text.strip()
    if token.isdigit() and len(token) <= 2:
        code = token.zfill(2)
        return code if code in STATE_ABBR else None
    wanted = normalize_name(token)
    for fips, abbr, name in STATES_TABLE:
        if wanted in (abbr.lower(), normalize_name(name)):
            return fips
    return None


def parse_states(entries: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split comma-separated state entries into known codes and unknown text.

    Returns:
        ``(state_codes, unknown)`` with codes de-duplicated in the order given.
    """
    codes: List[str] = []
    unknown: List[str] = []
    for entry in entries:
        for token in str(entry).split(","):
            if not token.strip():
                continue
            code = resolve_state(token)
            if code is None:
                unknown.append(token.strip())
            elif code not in codes:
                codes.append(code)
    return codes, unknown


def find_matches(entry: str, ref: Reference, states: Sequence[str]) -> Tuple[List[str], str]:
    """Return the FIPS codes a jurisdiction name matches, and the name searched for.

    A trailing ``, VA`` or ``, Virginia`` limits the search to that state;
    otherwise *states* does (empty = every state). The name matches a county's
    full name ("Fairfax County") or its name without the ending ("Fairfax").
    """
    text = entry.strip()
    scope = set(states)
    if "," in text:
        head, tail = text.rsplit(",", 1)
        tail_state = resolve_state(tail)
        if tail_state is not None:
            text, scope = head.strip(), {tail_state}
    wanted = normalize_name(text)
    matches = []
    for fips, county in ref.counties.items():
        if scope and county.state_fips not in scope:
            continue
        full = normalize_name(county.name)
        if wanted in (full, base_name(full)):
            matches.append(fips)
    return sorted(matches, key=ref.sort_key), text


def close_names(text: str, ref: Reference, states: Sequence[str], limit: int = 3) -> List[str]:
    """Return labels of the jurisdictions whose names are closest to *text*."""
    pool: Dict[str, str] = {}
    for fips, county in ref.counties.items():
        if not states or county.state_fips in states:
            pool.setdefault(normalize_name(county.name), fips)
    hits = difflib.get_close_matches(normalize_name(text), list(pool), n=limit, cutoff=0.6)
    return [ref.label(pool[hit]) for hit in hits]


# =============================================================================
# QUESTIONS
# =============================================================================


def ask_yes_no(ask: Ask, question: str) -> bool:
    """Ask a yes/no question; a blank answer means no."""
    while True:
        answer = ask(f"{question} [y/N]: ").strip().lower()
        if answer in ("", "n", "no"):
            return False
        if answer in ("y", "yes"):
            return True
        print("  Please answer y or n.")


def parse_picks(answer: str, count: int) -> Optional[List[int]]:
    """Parse ``1,3,5-7`` / ``all`` / blank into 0-based indexes, or None if invalid."""
    text = answer.strip().lower()
    if not text:
        return []
    if text == "all":
        return list(range(count))
    picks: List[int] = []
    for part in text.replace(" ", ",").split(","):
        if not part:
            continue
        bounds = part.split("-")
        if not all(b.isdigit() for b in bounds) or len(bounds) > 2:
            return None
        low, high = int(bounds[0]), int(bounds[-1])
        if not 1 <= low <= high <= count:
            return None
        picks.extend(i - 1 for i in range(low, high + 1) if i - 1 not in picks)
    return picks


def ask_picks(ask: Ask, labels: Sequence[str], question: str) -> List[int]:
    """Show a numbered list and return the 0-based indexes the user picks."""
    for number, label in enumerate(labels, start=1):
        print(f"  {number:>3}. {label}")
    while True:
        answer = ask(f"{question} (e.g. 1,3,5-7, 'all', or blank for none): ")
        picks = parse_picks(answer, len(labels))
        if picks is not None:
            return picks
        print(f"  Enter numbers between 1 and {len(labels)}, 'all', or leave blank.")


def resolve_entry(
    entry: str, ref: Reference, states: Sequence[str], ask: Optional[Ask]
) -> List[str]:
    """Turn one name or FIPS entry into FIPS codes, warning when nothing matches.

    A 4- or 5-digit number is taken as a FIPS code even when the reference files
    don't list it (with a warning). A name matching several jurisdictions is
    offered as a numbered list when someone can answer, and skipped with a
    warning otherwise.
    """
    text = str(entry).strip()
    if not text:
        return []
    if text.isdigit() and len(text) in (4, 5):
        code = text.zfill(5)
        if code not in ref.counties:
            logging.warning(
                "FIPS code %s is not in the reference files; it is kept as entered. "
                "Check it against the Census data you will filter.",
                code,
            )
        return [code]

    matches, searched = find_matches(text, ref, states)
    if len(matches) == 1:
        return matches
    if len(matches) > 1:
        labels = [ref.label(code) for code in matches]
        if ask is None:
            first = ref.counties[matches[0]]
            logging.warning(
                "'%s' matches %d jurisdictions: %s. None was added; enter the full name "
                "with its state (e.g. '%s, %s') or the FIPS code.",
                text,
                len(matches),
                "; ".join(labels),
                first.name,
                state_label(first.state_fips),
            )
            return []
        print(f"'{text}' matches more than one jurisdiction:")
        picks = ask_picks(ask, labels, "Which to add?")
        return [matches[i] for i in picks]

    elsewhere, _ = find_matches(searched, ref, [])
    if elsewhere and states:
        logging.warning(
            "No match for '%s' in %s, but it matches %s. Add that state, or enter the FIPS code.",
            text,
            ", ".join(state_label(s) for s in states),
            "; ".join(ref.label(code) for code in elsewhere[:5]),
        )
        return []
    suggestions = close_names(searched, ref, states)
    hint = f" Close matches: {'; '.join(suggestions)}." if suggestions else ""
    if not ref.has_adjacency:
        hint += (
            " Counties outside every metro or micropolitan area can be found by name only "
            "with COUNTY_ADJACENCY_FILE set; or enter the FIPS code."
        )
    logging.warning("No jurisdiction matches '%s'.%s", text, hint)
    return []


# =============================================================================
# SELECTION STEPS
# =============================================================================


def add_entries(
    entries: Sequence[str],
    selected: "OrderedDict[str, str]",
    reason: str,
    ref: Reference,
    states: Sequence[str],
    ask: Optional[Ask],
) -> List[str]:
    """Resolve each entry and add codes not yet selected; return the new codes."""
    new: List[str] = []
    for entry in entries:
        for code in resolve_entry(entry, ref, states, ask):
            if code not in selected:
                selected[code] = reason
                new.append(code)
    return new


def ask_and_add(
    ask: Ask,
    question: str,
    selected: "OrderedDict[str, str]",
    reason: str,
    ref: Reference,
    states: Sequence[str],
) -> List[str]:
    """Ask for entries one at a time, adding each as soon as it is typed.

    Returns:
        The entries given, in order.
    """
    entries: List[str] = []
    while True:
        answer = ask(f"{question} (blank when done): ").strip()
        if not answer:
            return entries
        entries.append(answer)
        for code in add_entries([answer], selected, reason, ref, states, ask):
            print(f"  Added {ref.label(code)}")


def ask_states(ask: Ask) -> List[str]:
    """Ask for the state(s) until every entry is recognized."""
    while True:
        answer = ask(
            "Which state(s) is the planning area in? Names, abbreviations, or 2-digit "
            "codes, separated by commas (blank = search every state): "
        )
        codes, unknown = parse_states([answer])
        if not unknown:
            return codes
        print(f"  Not recognized as a state: {', '.join(unknown)}. Try again.")


def metro_step(
    selected: "OrderedDict[str, str]",
    core: Sequence[str],
    ref: Reference,
    states: Sequence[str],
    scope: str,
    ask: Optional[Ask],
    decisions: List[str],
) -> None:
    """Offer the other jurisdictions in each core jurisdiction's metro area.

    With ``scope == "ask"`` the user is asked whether to add all of them, and if
    not, whether to add only those in the chosen states (asked only when that
    is a different, non-empty set). Additions go into *selected*.
    """
    by_cbsa: "OrderedDict[str, List[str]]" = OrderedDict()
    for code in core:
        county = ref.counties.get(code)
        if county is None or not county.cbsa_code:
            logging.info("%s is not in a metropolitan or micropolitan area.", ref.label(code))
            continue
        by_cbsa.setdefault(county.cbsa_code, []).append(code)

    for cbsa_code, cores in by_cbsa.items():
        first = ref.counties[cores[0]]
        title = f"{first.cbsa_title} {first.cbsa_kind}".strip()
        others = [c for c in ref.cbsa_members.get(cbsa_code, []) if c not in selected]
        others.sort(key=ref.sort_key)
        if not others:
            logging.info("Every jurisdiction in the %s is already selected.", title)
            continue
        state_scope = list(states) or sorted({ref.counties[c].state_fips for c in cores})
        in_states = [c for c in others if ref.counties[c].state_fips in state_scope]
        state_text = ", ".join(state_label(s) for s in state_scope)

        decision = scope
        if scope == "ask":
            if ask is None:
                decision = "none"
                logging.info("Not adding the rest of the %s (METRO_SCOPE is 'ask').", title)
            else:
                print(f"\nThe {title} also includes:")
                for code in others:
                    print(f"  {ref.label(code)}")
                if ask_yes_no(ask, f"Add all {len(others)} of them?"):
                    decision = "all"
                elif 0 < len(in_states) < len(others):
                    print(f"\nOf those, {len(in_states)} are in {state_text}:")
                    for code in in_states:
                        print(f"  {ref.label(code)}")
                    decision = "state" if ask_yes_no(ask, "Add just these?") else "none"
                else:
                    decision = "none"

        chosen = others if decision == "all" else in_states if decision == "state" else []
        reason = REASON_MICRO if first.cbsa_kind == "micropolitan area" else REASON_METRO
        for code in chosen:
            selected.setdefault(code, reason)
        decisions.append(
            f"{title}: added {len(chosen)} of {len(others)} other jurisdictions ({decision})."
        )
        logging.info("Added %d jurisdiction(s) from the %s.", len(chosen), title)


def adjacent_step(
    selected: "OrderedDict[str, str]",
    ref: Reference,
    scope: str,
    ask: Optional[Ask],
    decisions: List[str],
) -> None:
    """Suggest jurisdictions that share a border with the current selection."""
    if scope == "none":
        return
    if not ref.has_adjacency:
        if scope == "all" or ask is not None:
            logging.info("Suggestions by proximity need COUNTY_ADJACENCY_FILE; skipped.")
        return
    candidates = sorted(
        {n for code in selected for n in ref.neighbors.get(code, ())} - set(selected),
        key=ref.sort_key,
    )
    if not candidates:
        logging.info("No other jurisdictions share a border with the selection.")
        return

    if scope == "all":
        chosen = candidates
    elif ask is None:
        logging.info("Not adding bordering jurisdictions (ADJACENT_SCOPE is 'ask').")
        return
    else:
        if not ask_yes_no(
            ask,
            f"\nWould you like suggestions based on proximity ({len(candidates)} "
            "jurisdictions share a border with your selection)?",
        ):
            return
        labels = []
        for code in candidates:
            county = ref.counties.get(code)
            area = (
                f" - {county.cbsa_title} {county.cbsa_kind}" if county and county.cbsa_code else ""
            )
            labels.append(ref.label(code) + area)
        chosen = [candidates[i] for i in ask_picks(ask, labels, "Which to add?")]

    for code in chosen:
        selected.setdefault(code, REASON_ADJACENT)
    decisions.append(f"Bordering jurisdictions: added {len(chosen)} of {len(candidates)}.")


def ct_vintage_warnings(codes: Sequence[str]) -> List[str]:
    """Return warnings for Connecticut codes whose vintage may not match the data."""
    found = set(codes)
    messages = []
    if found & CT_PLANNING_REGIONS:
        messages.append(
            "The list holds Connecticut planning regions (09110-09190), used from 2022 on "
            "(ACS 2022+, 2023 metro delineation). 2020 Census blocks and tables use the former "
            "counties (09001-09015), so these codes will match nothing in them."
        )
    if found & CT_FORMER_COUNTIES:
        messages.append(
            "The list holds former Connecticut counties (09001-09015). ACS 2022+ and later "
            "TIGER files use planning regions (09110-09190), so these codes will match "
            "nothing in them."
        )
    return messages


def select_jurisdictions(
    ref: Reference,
    *,
    states: Sequence[str],
    core: Sequence[str],
    metro_scope: str,
    adjacent_scope: str,
    extras: Sequence[str],
    ask: Optional[Ask] = None,
) -> Tuple["OrderedDict[str, str]", Dict[str, Any]]:
    """Run the selection steps and return the chosen codes with their reasons.

    Args:
        ref: Reference lookups from :func:`build_reference`.
        states: State entries (names, abbreviations, codes); empty = ask, or
            every state when *ask* is None.
        core: Core jurisdiction entries; empty = ask.
        metro_scope: One of :data:`SCOPE_CHOICES`.
        adjacent_scope: One of :data:`ADJACENT_CHOICES`.
        extras: Extra entries by name or FIPS; empty = ask.
        ask: Function that asks a question and returns the answer (``input``
            in a terminal or notebook), or None to never ask.

    Returns:
        ``(selected, answers)``: FIPS code -> reason, in selection order, and
        the answers used (for the run log).

    Raises:
        SelectionError: No state entry is recognized, or no core jurisdiction
            is resolved.
    """
    if states:
        state_codes, unknown = parse_states(states)
        if unknown:
            logging.warning("Not recognized as a state: %s", ", ".join(unknown))
        if not state_codes:
            raise SelectionError("None of the STATES entries is a recognized state.")
    elif ask is not None:
        state_codes = ask_states(ask)
    else:
        state_codes = []

    selected: "OrderedDict[str, str]" = OrderedDict()
    if core:
        add_entries(core, selected, REASON_CORE, ref, state_codes, ask)
    elif ask is not None:
        while not selected:
            ask_and_add(
                ask,
                "Core county or other jurisdiction (name or 5-digit FIPS)",
                selected,
                REASON_CORE,
                ref,
                state_codes,
            )
            if not selected:
                print("  At least one core jurisdiction is needed.")
    if not selected:
        raise SelectionError(
            "No core jurisdiction was found. Set CORE_JURISDICTIONS (or --core), or run "
            "the script in a terminal or notebook to be asked."
        )
    core_codes = list(selected)
    logging.info("Core: %s", "; ".join(ref.label(code) for code in core_codes))

    decisions: List[str] = []
    metro_step(selected, core_codes, ref, state_codes, metro_scope, ask, decisions)
    adjacent_step(selected, ref, adjacent_scope, ask, decisions)

    before = len(selected)
    if extras:
        extra_entries = list(extras)
        add_entries(extra_entries, selected, REASON_EXTRA, ref, state_codes, ask)
    elif ask is not None:
        print()
        extra_entries = ask_and_add(
            ask,
            "Add other jurisdictions explicitly, by name or 5-digit FIPS",
            selected,
            REASON_EXTRA,
            ref,
            state_codes,
        )
    else:
        extra_entries = []
    added = len(selected) - before
    if extra_entries:
        decisions.append(
            f"Explicit entries: {len(extra_entries)} given, {added} not already selected."
        )

    answers: Dict[str, Any] = {
        "states": [state_label(s) for s in state_codes],
        "core": core_codes,
        "added": [code for code in selected if code not in core_codes],
        "decisions": decisions,
    }
    return selected, answers


# =============================================================================
# OUTPUT
# =============================================================================


def write_fips_file(
    path: Path, selected: "OrderedDict[str, str]", ref: Reference, source_names: Sequence[str]
) -> None:
    """Write one FIPS code per line, sorted, with name and reason as a comment."""
    lines = [
        "# County FIPS codes for the planning area, written by county_fips_picker.py",
        f"# {datetime.now().isoformat(timespec='seconds')} from {', '.join(source_names)}",
        "# One code per line; text after '#' is a comment. Use this file as a FIPS file",
        "# setting (e.g. FIPS_TO_FILTER_FILE) in the Census and coverage scripts.",
        "",
    ]
    for code in sorted(selected):
        county = ref.counties.get(code)
        name = (
            f"{county.name}, {state_label(county.state_fips)}"
            if county
            else "not in the reference files"
        )
        lines.append(f"{code}  # {name} - {selected[code]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_config_from_text(source_text: str, source_label: str) -> str:
    """Return the text between the CONFIG markers in *source_text*.

    Same slicing as :func:`extract_config_block`, for source that is already
    in memory (a notebook cell) rather than on disk.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
    """
    lines = source_text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_label}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )
    return "\n".join(lines[begin_idx + 1 : end_idx])


def resolve_config_source() -> Tuple[str, str]:
    """Return ``(config_text, source_label)`` for the running configuration.

    Inside a Jupyter/IPython kernel (including ArcGIS Pro notebooks) the most
    recent cell holding both CONFIG markers is used, so pasting the whole script
    into a cell still yields a faithful run log; otherwise ``__file__`` is read.

    Raises:
        RuntimeError: If neither source can be located.
        ValueError: If the located source has no CONFIG markers.
        OSError: If the source file cannot be read.
    """
    _ipython = sys.modules.get("IPython")
    ip = _ipython.get_ipython() if _ipython is not None else None  # type: ignore[attr-defined]
    if ip is not None:
        for cell in reversed(ip.user_ns.get("In", [])):
            if CONFIG_BEGIN_MARKER in cell and CONFIG_END_MARKER in cell:
                return _extract_config_from_text(cell, "<Jupyter cell>"), "<Jupyter cell>"

    file_attr: Optional[str] = globals().get("__file__")
    if file_attr is not None:
        source_path = Path(file_attr).resolve()
        return extract_config_block(source_path), str(source_path)

    raise RuntimeError(
        "Cannot locate the script source for the run log: __file__ is not defined and no "
        "notebook cell containing the CONFIG markers was found."
    )


def write_run_log(output_file: Path, answers: Dict[str, Any], ref: Reference) -> bool:
    """Write ``<output stem>_runlog.txt`` with the config block and the answers given.

    The answers are written as CONFIGURATION lines that reproduce this run's
    list without questions.

    Returns:
        ``True`` if the log was written, ``False`` otherwise.
    """
    log_path = output_file.with_name(f"{output_file.stem}_runlog.txt")
    try:
        config_text, source_label = resolve_config_source()
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Could not read the CONFIGURATION block for the run log: %s", exc)
        return False

    lines: List[str] = [
        "=" * 72,
        "COUNTY FIPS PICKER RUN LOG",
        "=" * 72,
        f"Run timestamp:  {datetime.now().isoformat(timespec='seconds')}",
        f"Output file:    {output_file}",
        f"Source script:  {source_label}",
        "",
        "-" * 72,
        "WHAT WAS ADDED",
        "-" * 72,
        *(f"Core: {ref.label(code)}" for code in answers["core"]),
        *answers["decisions"],
        "",
        "-" * 72,
        "ANSWERS (paste into CONFIGURATION to repeat this run without questions)",
        "-" * 72,
        f"STATES: List[str] = {answers['states']!r}",
        f"CORE_JURISDICTIONS: List[str] = {answers['core']!r}",
        'METRO_SCOPE: str = "none"',
        'ADJACENT_SCOPE: str = "none"',
        f"EXTRA_JURISDICTIONS: List[str] = {answers['added']!r}",
        "",
        "-" * 72,
        "CONFIGURATION (verbatim)",
        "-" * 72,
        CONFIG_BEGIN_MARKER,
        config_text,
        CONFIG_END_MARKER,
        "",
    ]
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logging.error("Could not write run log '%s': %s", log_path, exc)
        return False
    logging.info("Run log written → %s", log_path)
    return True


# =============================================================================
# ENTRY POINT
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser; every default is the matching CONFIGURATION value."""
    parser = argparse.ArgumentParser(
        description="Build a county FIPS list for a planning area.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metro-file", default=METRO_DELINEATION_FILE, help="Delineation file.")
    parser.add_argument(
        "--adjacency-file", default=COUNTY_ADJACENCY_FILE, help="County adjacency file."
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--output-filename", default=OUTPUT_FILENAME, help="Output file name.")
    parser.add_argument("--states", nargs="*", default=STATES, help="State(s) of the area.")
    parser.add_argument("--core", nargs="*", default=CORE_JURISDICTIONS, help="Core names/FIPS.")
    parser.add_argument("--metro-scope", choices=SCOPE_CHOICES, default=METRO_SCOPE)
    parser.add_argument("--adjacent-scope", choices=ADJACENT_CHOICES, default=ADJACENT_SCOPE)
    parser.add_argument("--extra", nargs="*", default=EXTRA_JURISDICTIONS, help="Extra names/FIPS.")
    parser.add_argument(
        "--no-prompt", action="store_true", help="Never ask questions, even in a terminal."
    )
    parser.add_argument(
        "--log-level", default=logging.getLevelName(LOG_LEVEL), help="DEBUG/INFO/WARNING/ERROR."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, ask: Optional[Ask] = None) -> int:
    """Load the reference files, build the selection, and write the FIPS file.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv`` (or nothing in
            a notebook).
        ask: Question function to use instead of ``input`` (for tests). When
            None, ``input`` is used if the script runs in a terminal or notebook
            and ``--no-prompt`` is not set.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 for a configuration
        problem (placeholder paths, missing files, or nothing selected).
    """
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.metro_file == _DEFAULT_METRO_DELINEATION_FILE or args.output_dir == _DEFAULT_OUTPUT_DIR:
        logging.warning(
            "METRO_DELINEATION_FILE and/or OUTPUT_DIR are still placeholders. Update the "
            "CONFIGURATION section (or pass --metro-file / --output-dir) and re-run."
        )
        return 2
    metro_path = Path(args.metro_file)
    adjacency_path = Path(args.adjacency_file) if args.adjacency_file else None
    for label, path in (("Metro delineation", metro_path), ("Adjacency", adjacency_path)):
        if path is not None and not path.is_file():
            logging.error("%s file not found: %s", label, path)
            return 2

    if ask is None and not args.no_prompt and stdin_is_interactive():
        ask = input

    try:
        metro = load_metro_delineation(metro_path)
        names, neighbors = load_county_adjacency(adjacency_path) if adjacency_path else ({}, {})
        ref = build_reference(metro, names, neighbors)
        selected, answers = select_jurisdictions(
            ref,
            states=args.states,
            core=args.core,
            metro_scope=args.metro_scope,
            adjacent_scope=args.adjacent_scope,
            extras=args.extra,
            ask=ask,
        )
    except (EOFError, KeyboardInterrupt):
        logging.warning("Cancelled; nothing was written.")
        return 2
    except SelectionError as exc:
        logging.error("%s", exc)
        return 2
    except ValueError as exc:
        logging.error("Could not read a reference file: %s", exc)
        return 2
    except Exception:  # noqa: BLE001
        logging.exception("Selection failed")
        return 1

    for message in ct_vintage_warnings(list(selected)):
        logging.warning(message)

    output_path = Path(args.output_dir) / args.output_filename
    sources = [p.name for p in (metro_path, adjacency_path) if p is not None]
    try:
        write_fips_file(output_path, selected, ref, sources)
    except OSError as exc:
        logging.error("Could not write %s: %s", output_path, exc)
        return 1
    logging.info("Wrote %d FIPS code(s) → %s", len(selected), output_path)

    if not write_run_log(output_path, answers, ref) and REQUIRE_RUN_LOG:
        return 1
    logging.info("Done. Point the FIPS file settings of the other scripts at %s.", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
