"""Regression tests for the config_section banner test.

CONTRIBUTING.md asks for "a clear configuration section at the top" without
mandating the banner's spelling, so `# CONFIG` and `# Config` head a section
just as `# CONFIGURATION` does. These tests pin that, and pin the two things
the looser test must not do: treat prose that merely mentions config as a
banner, or stop reporting a section whose BEGIN/END markers are missing.
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
# A section is recognised however its banner is spelled
# =============================================================================


@pytest.mark.parametrize(
    "banner",
    [
        "# CONFIGURATION",
        "# CONFIG",
        "# Config",
        "# config",
        "# ==== CONFIGURATION ====",
        "# --- CONFIG ---",
        "# CONFIG  --  notebook users edit these; CLI flags override them",
    ],
)
def test_banner_spellings_count_as_a_section(banner: str) -> None:
    # With markers present too, a recognised banner leaves nothing to report.
    assert check_config_section(f"{banner}\n{MARKERS}") == []


def test_banner_without_markers_reports_the_markers_not_a_missing_section() -> None:
    # The section exists; the accurate complaint is about its markers.
    message = _only(check_config_section(f"# CONFIG\n{BODY}"))
    assert LACKS_MARKERS in message
    assert NO_SECTION not in message


def test_markers_alone_count_as_a_section() -> None:
    # A block delimited by both markers is a configuration section, banner or not.
    assert check_config_section(MARKERS) == []


# =============================================================================
# The looser test must not match prose that merely mentions config
# =============================================================================


@pytest.mark.parametrize(
    "comment",
    [
        "# Full config OR previous run-log location. Join filenames with /.",
        "# Anything still unset after arg + config block falls to a prompt.",
        "# --- <<< EDIT ME ------------------------------------------------",
        "# Update the paths in the Config block before running.",
    ],
)
def test_prose_mentioning_config_is_not_a_banner(comment: str) -> None:
    assert NO_SECTION in _only(check_config_section(f"{comment}\n{BODY}"))


def test_no_configuration_section_at_all() -> None:
    assert NO_SECTION in _only(check_config_section(BODY))


# =============================================================================
# Mismatched markers still get their own specific message
# =============================================================================


def test_begin_without_end() -> None:
    src = "# CONFIGURATION\n# === BEGIN CONFIG ===\n" + BODY
    message = _only(check_config_section(src))
    assert "no matching `# === END CONFIG ===`" in message


def test_end_without_begin() -> None:
    src = "# CONFIGURATION\n" + BODY + "# === END CONFIG ===\n"
    message = _only(check_config_section(src))
    assert "no matching `# === BEGIN CONFIG ===`" in message
