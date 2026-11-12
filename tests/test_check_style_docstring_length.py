"""Tests for the docstring_length check.

CONTRIBUTING.md asks for module docstrings of about 50 lines. The check warns
when one runs past DOCSTRING_MAX_LINES, counting from the opening quotes to the
closing quotes, so a script's README-style docstring does not drift long again.
"""

from __future__ import annotations

from pathlib import Path

from dev_tools.check_style import DOCSTRING_MAX_LINES, check_docstring_length

DUMMY = Path("example.py")


def _module(n_lines: int) -> str:
    """Return module source whose docstring spans exactly *n_lines* lines (n >= 3)."""
    body = "\n".join(f"Line {i}." for i in range(n_lines - 2))
    return f'"""Summary.\n{body}\n"""\n\nX = 1\n'


def test_helper_builds_the_requested_span() -> None:
    src = _module(7)
    assert src.splitlines()[6] == '"""'
    assert src.splitlines()[7] == ""


def test_docstring_at_the_limit_passes() -> None:
    assert check_docstring_length(_module(DOCSTRING_MAX_LINES), DUMMY) == []


def test_docstring_over_the_limit_is_flagged() -> None:
    found = check_docstring_length(_module(DOCSTRING_MAX_LINES + 1), DUMMY)
    assert len(found) == 1
    assert found[0].check == "docstring_length"
    assert found[0].line == 1
    assert f"{DOCSTRING_MAX_LINES + 1} lines" in found[0].message


def test_custom_limit() -> None:
    assert check_docstring_length(_module(10), DUMMY, max_lines=10) == []
    assert len(check_docstring_length(_module(11), DUMMY, max_lines=10)) == 1


def test_count_starts_at_the_docstring_not_the_file() -> None:
    src = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n" + _module(10)
    found = check_docstring_length(src, DUMMY, max_lines=9)
    assert len(found) == 1
    assert found[0].line == 3
    assert "10 lines" in found[0].message


def test_one_line_docstring_passes() -> None:
    assert check_docstring_length('"""Summary."""\n\nX = 1\n', DUMMY, max_lines=1) == []


def test_missing_docstring_is_left_to_docstring_sections() -> None:
    assert check_docstring_length("X = 1\n", DUMMY, max_lines=1) == []


def test_syntax_error_is_skipped() -> None:
    assert check_docstring_length('"""Summary."""\n\ndef broken(:\n', DUMMY, max_lines=1) == []
