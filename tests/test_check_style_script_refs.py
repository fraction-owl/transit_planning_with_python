"""Tests for the docstring_script_refs check.

A module docstring that names another script by filename (in ``Inputs``,
``See Also:``, or prose) goes stale silently when that script is renamed. The
check flags every ``name.py`` in the docstring that matches no file in the
repository, while leaving glob and template forms (``*_arcpy.py``,
``<stem>.py``) alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dev_tools.check_style import Violation, check_docstring_script_refs

DUMMY = Path("example.py")
KNOWN = frozenset({"otp_monthly_panel.py", "prep_features_public.py", "example.py"})


def _module(doc: str) -> str:
    return f'"""{doc}"""\n\nX = 1\n'


def _check(doc: str) -> list[Violation]:
    return check_docstring_script_refs(_module(doc), DUMMY, known_names=KNOWN)


@pytest.mark.parametrize(
    "doc",
    [
        "Summarize OTP.\n\nInputs:\n    - Panel CSV from ``otp_monthly_panel.py``.\n",
        (
            "Summarize OTP.\n\nSee Also:\n"
            "    - ``otp_monthly_panel.py``\n"
            "    - ``prep_features_public.py``\n"
        ),
        "Summarize OTP.\n\nSee ``scripts/modeling/prep_features_public.py`` for the join.\n",
        "Summarize OTP.\n\nRun as ``python example.py --input ...``.\n",
        "Summarize OTP.\n\nThe twin (otp_monthly_panel.py) owns the definition.\n",
    ],
)
def test_known_names_pass(doc: str) -> None:
    assert _check(doc) == []


@pytest.mark.parametrize(
    "doc",
    [
        "Summarize OTP.\n\nApplies to every ``*_arcpy.py`` variant.\n",
        "Summarize OTP.\n\nOutputs go under ``{work}/out/<stem>.py``.\n",
        "Summarize OTP.\n\nByte-compiled ``otp_by_route.pyc`` files are ignored.\n",
        "Summarize OTP.\n\nNo filenames here at all.\n",
    ],
)
def test_globs_templates_and_other_extensions_are_ignored(doc: str) -> None:
    assert _check(doc) == []


def test_unknown_name_is_flagged_with_its_line() -> None:
    doc = "Clean a FRED export.\n\nRun as ``python clean_fred_series.py`` (CLI).\n"
    violations = _check(doc)
    assert len(violations) == 1
    assert violations[0].check == "docstring_script_refs"
    assert "`clean_fred_series.py`" in violations[0].message
    assert violations[0].line == 3


def test_each_unknown_name_is_reported_once() -> None:
    doc = (
        "Summarize OTP.\n\n"
        "Reads ``old_panel.py`` output.\n\n"
        "See Also:\n    - ``old_panel.py``\n    - ``gone_script.py``\n"
    )
    violations = _check(doc)
    names = sorted(v.message.split("`")[1] for v in violations)
    assert names == ["gone_script.py", "old_panel.py"]


def test_names_outside_the_docstring_are_not_checked() -> None:
    src = _module("Summarize OTP.\n") + "# see missing_script.py\n"
    assert check_docstring_script_refs(src, DUMMY, known_names=KNOWN) == []


def test_missing_docstring_and_unparseable_source_are_skipped() -> None:
    assert check_docstring_script_refs("X = 1\n", DUMMY, known_names=KNOWN) == []
    assert check_docstring_script_refs("def (:\n", DUMMY, known_names=KNOWN) == []


def test_default_names_come_from_the_repository() -> None:
    # This file and the checker itself both exist, so naming them passes.
    doc = "Test.\n\nSee ``test_check_style_script_refs.py`` and ``check_style.py``.\n"
    assert check_docstring_script_refs(_module(doc), DUMMY) == []
    assert len(check_docstring_script_refs(_module("Test.\n\nSee ``nope_xyz.py``.\n"), DUMMY)) == 1
