"""Regression tests for the config_section banner test.

CONTRIBUTING.md fixes the banner's *spelling* — ``CONFIGURATION`` in full caps —
and leaves its *shape* to the file, so rule characters and a trailing note head
a section just as the bare word does. These tests pin that split, and pin the
two things the test must keep rejecting: the abbreviated ``# CONFIG``, and prose
that merely mentions configuration.
"""

from __future__ import annotations

import pytest

from dev_tools.check_style import Violation, check_config_section

MARKERS = "# === BEGIN CONFIG ===\nOUTPUT_DIR = r'out'\n# === END CONFIG ===\n"
BODY = "OUTPUT_DIR = r'out'\n"

NO_SECTION = "No CONFIGURATION section found"
LACKS_MARKERS = "lacks BEGIN/END markers"


def _only(violations: list[Violation]) -> str:
    assert len(violations) == 1, f"expected one finding, got {violations}"
    assert violations[0].check == "config_section"
    return violations[0].message


# =============================================================================
# Spelling is fixed; shape is the file's own business
# =============================================================================


@pytest.mark.parametrize(
    "banner",
    [
        "# CONFIGURATION",
        "# ==== CONFIGURATION ====",
        "# --- CONFIGURATION ---",
        "# CONFIGURATION -- edit here; IDs must remain strings",
        "# CONFIGURATION  --  notebook users edit these; CLI flags override them",
        "# CONFIGURATION CHECKS",
    ],
)
def test_banner_shapes_count_as_a_section(banner: str) -> None:
    assert check_config_section(f"{banner}\n{MARKERS}") == []


def test_banner_without_markers_reports_the_markers_not_a_missing_section() -> None:
    # The section exists; the accurate complaint is about its markers.
    message = _only(check_config_section(f"# CONFIGURATION\n{BODY}"))
    assert LACKS_MARKERS in message
    assert NO_SECTION not in message


# =============================================================================
# The abbreviation is reserved for the markers, so it is not a banner
# =============================================================================


@pytest.mark.parametrize("banner", ["# CONFIG", "# Config", "# config", "# Configuration"])
def test_abbreviated_or_lowercase_banners_are_not_a_section(banner: str) -> None:
    assert NO_SECTION in _only(check_config_section(f"{banner}\n{BODY}"))


def test_markers_alone_are_not_a_section() -> None:
    # A config block still needs its banner; the markers are not a substitute.
    assert NO_SECTION in _only(check_config_section(MARKERS))


# =============================================================================
# Prose that mentions configuration is not a banner
# =============================================================================


@pytest.mark.parametrize(
    "comment",
    [
        # A wrapped comment whose continuation line opens with the word. Four
        # scripts carry almost exactly this text above their marker constants.
        "# Sentinel markers used by extract_config_block to identify the\n"
        "# configuration block within this file. Edit with care.",
        "# Full config OR previous run-log location. Join filenames with /.",
        "# Anything still unset after arg + config block falls to a prompt.",
        "# Update the paths in the CONFIGURATION block before running.",
        "# --- <<< EDIT ME ------------------------------------------------",
        "    # CONFIGURATION value when checking it against --feed.",
    ],
)
def test_prose_mentioning_configuration_is_not_a_banner(comment: str) -> None:
    assert NO_SECTION in _only(check_config_section(f"{comment}\n{BODY}"))


def test_no_configuration_section_at_all() -> None:
    assert NO_SECTION in _only(check_config_section(BODY))


# =============================================================================
# Mismatched markers still get their own specific message
# =============================================================================


def test_begin_without_end() -> None:
    src = "# CONFIGURATION\n# === BEGIN CONFIG ===\n" + BODY
    assert "no matching `# === END CONFIG ===`" in _only(check_config_section(src))


def test_end_without_begin() -> None:
    src = "# CONFIGURATION\n" + BODY + "# === END CONFIG ===\n"
    assert "no matching `# === BEGIN CONFIG ===`" in _only(check_config_section(src))
