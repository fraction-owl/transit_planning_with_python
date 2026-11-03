"""Regression tests for the three style checks that used to misfire.

Each check is covered in both directions: the shapes that were wrongly flagged
now pass, and a genuine violation of the same rule is still reported, so a fix
that merely disabled the check would fail here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dev_tools.check_style import (
    check_notebook_guard,
    check_raw_string_paths,
    check_success_message,
)

DUMMY = Path("example.py")


def _config(body: str) -> str:
    return f"# === BEGIN CONFIG ===\n{body}\n# === END CONFIG ===\n"


# =============================================================================
# raw_string_paths
# =============================================================================
# CONTRIBUTING.md gives r"Path\\To\\Output_Folder" and "input_data.csv" as two
# correct placeholders, so only a backslash-bearing plain string is a problem.


@pytest.mark.parametrize(
    "assignment",
    [
        'OUTPUT_FILENAME = "routes_summary.xlsx"',  # bare filename
        'SERVICE_TYPES_PATH = ""',  # empty placeholder
        'SHAPEFILE_INPUT: str = "output/stations.shp"',  # forward slashes
        'GTFS_DIR = Path("/path/to/gtfs_directory")',  # Path(), forward slashes
        r'OUTPUT_DIR = r"Path\To\Output_Folder"',  # already raw
        r'GTFS_DIR = Path(r"C:\GTFS\data")',  # already raw, inside Path()
    ],
)
def test_raw_string_paths_allows_safe_literals(assignment: str) -> None:
    assert check_raw_string_paths(_config(assignment)) == []


@pytest.mark.parametrize(
    "assignment",
    [
        r'OUTPUT_DIR = "Path\To\Output_Folder"',
        r'GTFS_DIR = Path("C:\GTFS\data")',
    ],
)
def test_raw_string_paths_flags_plain_backslash_literals(assignment: str) -> None:
    violations = check_raw_string_paths(_config(assignment))
    assert len(violations) == 1
    assert violations[0].check == "raw_string_paths"


# =============================================================================
# notebook_guard
# =============================================================================
# The guard is matched structurally, so formatting around it is irrelevant.

_MAIN = "def main():\n    return 0\n"

_PLAIN = 'if __name__ == "__main__":\n    raise SystemExit(main())\n'
_COMMENTED = (
    'if __name__ == "__main__":\n'
    "    # keep the kernel's argv away from argparse\n"
    "    raise SystemExit(main())\n"
)
_WRAPPED = (
    'if __name__ == "__main__":\n'
    "    try:\n"
    "        raise SystemExit(main())\n"
    "    except Exception:\n"
    "        raise\n"
)
_PRAGMA = 'if __name__ == "__main__":  # pragma: no cover\n    raise SystemExit(main())\n'
_SYS_EXIT = 'if __name__ == "__main__":\n    sys.exit(main())\n'
_BARE = 'if __name__ == "__main__":\n    main()\n'
_REVERSED = 'if "__main__" == __name__:\n    raise SystemExit(main())\n'

_NO_MAIN_CALL = 'if __name__ == "__main__":\n    print("hi")\n'
_NESTED = 'def f():\n    if __name__ == "__main__":\n        main()\n'


@pytest.mark.parametrize(
    "guard",
    [_PLAIN, _COMMENTED, _WRAPPED, _PRAGMA, _SYS_EXIT, _BARE, _REVERSED],
)
def test_notebook_guard_accepts_valid_guards(guard: str) -> None:
    assert check_notebook_guard(_MAIN + guard, DUMMY) == []


@pytest.mark.parametrize(
    "source",
    [
        _MAIN,  # no guard at all
        _MAIN + _NO_MAIN_CALL,  # guard never calls main()
        _MAIN + _NESTED,  # guard is nested, not at module level
    ],
)
def test_notebook_guard_flags_missing_guards(source: str) -> None:
    violations = check_notebook_guard(source, DUMMY)
    assert len(violations) == 1
    assert violations[0].check == "notebook_guard"


def test_notebook_guard_ignores_unparseable_source() -> None:
    assert check_notebook_guard("def main(:\n", DUMMY) == []


# =============================================================================
# success_message
# =============================================================================
# A module-level `logger = logging.getLogger(__name__)` alias is still the
# logging module, which is what CONTRIBUTING.md asks for.


@pytest.mark.parametrize(
    "call",
    [
        'logging.info("Script completed successfully.")',
        'logger.info("Script completed successfully.")',
        'LOGGER.info("done")',
        '_logger.info("finished")',
    ],
)
def test_success_message_accepts_logger_aliases(call: str) -> None:
    assert check_success_message(call + "\n") == []


@pytest.mark.parametrize(
    "call",
    [
        'logger.info("Wrote %s", path)',  # logs, but not a completion message
        'print("complete")',  # not the logging module
        'catalog.info("complete")',  # unrelated object that merely ends in "log"-ish text
    ],
)
def test_success_message_flags_missing_completion(call: str) -> None:
    violations = check_success_message(call + "\n")
    assert len(violations) == 1
    assert violations[0].check == "success_message"
