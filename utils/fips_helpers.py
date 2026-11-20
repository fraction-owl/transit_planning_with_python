"""Canonical county FIPS-list helpers for Census and coverage scripts.

Holds the canonical version of the loader that the Census and service-coverage
scripts use to combine an inline FIPS list with an optional ``.txt`` file (for
example one written by ``county_fips_picker.py``). Per CONTRIBUTING.md, scripts
do not import this at runtime — they carry verbatim copies, and CI's
helper-function audit flags any copy that drifts from this file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_fips_codes(
    inline_codes: Sequence[str] | None = None,
    txt_path: str | Path | None = None,
) -> list[str]:
    """Combine an inline list and an optional text file of county FIPS codes.

    The text file holds one or more codes per line, separated by commas or
    spaces. Blank lines are skipped and ``#`` starts a comment, so a line such as
    ``51059  # Fairfax County, VA`` reads as ``51059``. Four-digit codes are
    zero-padded to five, which restores the leading ``0`` a spreadsheet drops
    from states 01-09 (``1001`` becomes ``01001``).

    Unlike a general id-list loader, a file path that is set but missing is an
    error rather than a warning: an empty FIPS filter means "keep every county",
    so a mistyped path would otherwise process the whole country silently.

    Canonical implementation: ``utils/fips_helpers.py``.

    Args:
        inline_codes: Codes supplied directly (e.g. a CONFIGURATION list).
            ``None`` is treated as empty.
        txt_path: Path to a text file of codes, or ``None``/``""`` to skip.

    Returns:
        The combined codes as sorted, de-duplicated 5-digit strings. An empty
        list means no filter.

    Raises:
        FileNotFoundError: *txt_path* is set but no file exists there.
        ValueError: An entry is not a 4- or 5-digit number. The message lists
            every bad entry and where it came from.
    """
    entries: list[tuple[str, str]] = [(str(code), "inline list") for code in inline_codes or ()]

    if txt_path:
        path = Path(txt_path)
        if not path.is_file():
            raise FileNotFoundError(f"FIPS file not found: {path}")
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                for token in line.split("#", 1)[0].replace(",", " ").split():
                    entries.append((token, f"{path.name} line {line_no}"))

    codes: set[str] = set()
    bad: list[str] = []
    for raw, source in entries:
        text = raw.strip()
        if not text:
            continue
        if len(text) in (4, 5) and text.isascii() and text.isdigit():
            codes.add(text.zfill(5))
        else:
            bad.append(f"'{text}' ({source})")

    if bad:
        raise ValueError(
            "These entries are not 5-digit county FIPS codes: "
            + ", ".join(bad)
            + ". Each code is a 2-digit state code followed by a 3-digit county code, "
            "e.g. 51059 for Fairfax County, VA."
        )

    if txt_path:
        logging.info("Read county FIPS codes from %s.", txt_path)
    logging.info("Resolved %d county FIPS code(s).", len(codes))
    return sorted(codes)
