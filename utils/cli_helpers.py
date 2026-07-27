"""Utility helpers for command-line parsing in transit analysis scripts.

This module contains the canonical implementation of the CLI helpers that are
intentionally reproduced (verbatim) inside each argparse-enabled analysis
script so that scripts remain self-contained and runnable without the project
on ``sys.path``.  When updating any function here, mirror the change in every
script that carries a copy — the docstring comment in those copies names this
file as the source.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence


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


def prompt_for_path(
    prompt: str,
    *,
    must_exist: bool = True,
    default: Optional[Path] = None,
    allow_skip: bool = False,
) -> Optional[Path]:
    """Ask for one path on stdin, re-asking until the answer is usable.

    Surrounding quotes are stripped so values pasted from Windows Explorer's
    "Copy as path" (which wraps the path in double quotes) work as-is. Blank
    input returns *default* when one is set, returns None when *allow_skip*
    is True, and otherwise re-asks. Only call this after
    :func:`stdin_is_interactive` has confirmed a user is present.

    Canonical implementation: ``utils/cli_helpers.py``.

    Args:
        prompt: Text shown to the user; include any default/skip hint.
        must_exist: Re-ask until the entered path exists on disk (applies to
            typed answers only, never to *default*).
        default: Returned on blank input.
        allow_skip: Blank input returns None instead of re-asking (ignored
            when *default* is set).

    Returns:
        The entered path, or *default* / None per the blank-input rules.

    Raises:
        KeyboardInterrupt: The user cancelled with Ctrl+C.
        EOFError: Stdin closed mid-prompt. Callers should catch both and
            treat them as "user aborted the guided setup".
    """
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            if default is not None:
                return default
            if allow_skip:
                return None
            logging.warning("A path is required here — enter one, or press Ctrl+C to abort.")
            continue
        path = Path(raw)
        if must_exist and not path.exists():
            logging.warning("Path not found: %s — check it and try again.", path)
            continue
        return path
