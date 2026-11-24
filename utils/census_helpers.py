"""Canonical county-FIPS helpers for the Census scripts in scripts/national_data_tools/.

The uscensus_* scripts carry verbatim copies of these functions so each stays
self-contained. When updating a function here, mirror the change in every script
that carries a copy — the docstring of each copy names this file as the source.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence


def load_fips_file(path: str | Path) -> list[str]:
    """Read 5-digit county FIPS codes from a text file, one code per line.

    The code is the first word on its line. Anything after it on the same line,
    such as the county name in a line copied from the FCC list at
    https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt, is ignored,
    as are blank lines and anything after ``#``. A code listed twice is kept
    once, in first-seen order.

    Canonical implementation: ``utils/census_helpers.py``.

    Args:
        path: The text file.

    Returns:
        The codes as 5-character strings, in file order.

    Raises:
        ValueError: If the file cannot be read, a line does not start with a
            5-digit code, or the file lists no codes.
    """
    file_path = Path(path).expanduser()
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read the FIPS file '{file_path}': {exc}") from exc
    codes: list[str] = []
    seen: set[str] = set()
    problems: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        words = raw.split("#", 1)[0].split()
        if not words:
            continue
        code = words[0]
        if len(code) != 5 or not (code.isascii() and code.isdigit()):
            problems.append(f"line {number}: expected a 5-digit county FIPS code, got {code!r}")
        elif code not in seen:
            seen.add(code)
            codes.append(code)
    if problems:
        raise ValueError(f"Could not read the FIPS file '{file_path}':\n  " + "\n  ".join(problems))
    if not codes:
        raise ValueError(
            f"The FIPS file '{file_path}' lists no codes. Add one 5-digit county code "
            "per line (or uncomment the ones you want), or unset the FIPS file setting "
            "to use the inline FIPS list instead."
        )
    return codes


def resolve_fips_filter(
    inline_codes: Sequence[str] | None, fips_file: str | Path | None
) -> list[str]:
    """Pick the county FIPS codes to filter on: the FIPS file if set, else the inline list.

    Canonical implementation: ``utils/census_helpers.py``.

    Args:
        inline_codes: The inline FIPS list from the CONFIGURATION block. Empty or
            ``None`` keeps every county.
        fips_file: Path to a text file read by ``load_fips_file``, or ``None`` /
            ``""`` to use *inline_codes*.

    Returns:
        The codes to keep; an empty list keeps every county.

    Raises:
        ValueError: If *fips_file* is set and cannot be used (see ``load_fips_file``).
    """
    if fips_file is None or not str(fips_file).strip():
        return list(inline_codes or [])
    codes = load_fips_file(fips_file)
    if inline_codes:
        logging.warning(
            "Both the inline FIPS list and a FIPS file are set; using the FIPS file '%s' "
            "and ignoring the inline list.",
            fips_file,
        )
    logging.info("Loaded %d county FIPS code(s) from '%s'.", len(codes), fips_file)
    return codes
