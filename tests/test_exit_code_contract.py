"""Pin the 0/1/2 exit-code convention by actually running every script's main().

CONTRIBUTING.md gives every script's ``main()`` an integer exit-code contract:
``0`` success, ``1`` runtime failure, ``2`` configuration error.  A script ships
with placeholder CONFIGURATION values, so calling ``main()`` on a fresh checkout
must take the configuration-error path and return ``2`` without touching disk.

``dev_tools/check_style.py``'s ``exit_code_contract`` check covers the static
half of the contract (``-> int``, no bare ``return``, a guard that propagates
the value).  It cannot see behaviour: a script that returns ``1`` where ``2``
belongs, or raises instead of returning, is only visible by running it.  That is
what this module does.

Scripts that do not yet honour the contract are listed in
:data:`KNOWN_VIOLATIONS` with the reason, and are marked ``xfail(strict=True)``
rather than skipped — the test body still runs, so fixing one turns its entry
into an unexpected pass and fails the suite until the entry is removed.  The
list can only shrink.
"""

from __future__ import annotations

import ast
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

SCRIPTS = sorted(p for p in SCRIPTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)

# Scripts whose main() does not return 2 on its shipped placeholder config.
# Each entry is a bug or a deliberate exemption, not a blank cheque — see
# https://github.com/fraction-owl/transit_planning_with_python/issues/129.
KNOWN_VIOLATIONS: dict[str, str] = {
    # No placeholder guard: mkdir()s the output folder, then lets the GTFS
    # loader raise OSError on the placeholder feed path.
    "facilities_tools/stop_upgrades_survey.py": "raises OSError; no placeholder guard",
    # No placeholder guard: the missing-input error is caught and mapped to 1.
    "gbfs_tools/bikeshare_ridership_trends.py": "returns 1; no placeholder guard",
    "gbfs_tools/gbfs_ridership_join.py": "returns 1; no placeholder guard",
    # Config defaults are None and the local prompt loop calls input()
    # unconditionally, so a non-interactive run dies instead of returning 2.
    # utils/cli_helpers.py:stdin_is_interactive() exists for exactly this.
    "national_data_tools/clean_eia_gas_prices.py": "raises EOFError; prompts unconditionally",
    "national_data_tools/clean_fred_unrate.py": "raises EOFError; prompts unconditionally",
    "national_data_tools/clean_noaa_weather.py": "raises EOFError; prompts unconditionally",
    "national_data_tools/schools_prep_join_gpd.py": "raises EOFError; prompts unconditionally",
    # Creates a timestamped run directory before validating configuration.
    "operations_tools/runtime_workflow.py": "returns 1; validates after creating run dir",
    "stop_analysis/gtfs_stop_diff.py": "returns 1 for a placeholder feed path",
}

# arcpy ships only inside ArcGIS Pro's bundled Python, so the *_arcpy scripts
# cannot be imported in CI without a stand-in.  A MagicMock is enough to reach
# every placeholder guard: arcpy.Exists() returns a truthy mock and validation
# falls through to the plain-Path checks that actually fail.
_ARCPY_MODULES = (
    "arcpy",
    "arcpy.analysis",
    "arcpy.cartography",
    "arcpy.conversion",
    "arcpy.da",
    "arcpy.management",
    "arcpy.mp",
    "arcpy.na",
    "arcpy.sa",
)


def _script_id(path: Path) -> str:
    return path.relative_to(SCRIPTS_ROOT).as_posix()


def _params(*, mark_known: bool) -> list[pytest.param]:
    """Build the parametrize list, optionally xfailing the known violations."""
    params = []
    for script in SCRIPTS:
        script_id = _script_id(script)
        reason = KNOWN_VIOLATIONS.get(script_id) if mark_known else None
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(script, marks=marks, id=script_id))
    return params


def _takes_argv(script: Path) -> bool:
    """Return True if this script's main() accepts an ``argv`` parameter.

    Two scripts take something else first (``main(stages, argv)`` and
    ``main(config_file)``), so argv goes in by keyword, never by position.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return any(arg.arg == "argv" for arg in node.args.posonlyargs + node.args.args)
    return False


@pytest.fixture
def arcpy_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Import scripts against a stand-in ``arcpy`` without leaking either module."""
    for name in _ARCPY_MODULES:
        monkeypatch.setitem(sys.modules, name, MagicMock(name=name))
    already_cached = set(sys.modules)
    yield
    # Drop the script modules that bound the stand-in, so a later test that
    # installs its own arcpy stub re-imports cleanly (house pattern — see
    # tests/test_district_ridership_share_arcpy.py).
    for name in set(sys.modules) - already_cached:
        if name.startswith("scripts.") and isinstance(
            getattr(sys.modules[name], "arcpy", None), MagicMock
        ):
            del sys.modules[name]


@pytest.mark.parametrize("script", _params(mark_known=True))
def test_main_returns_2_on_shipped_placeholders(
    script: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arcpy_stub: None,
) -> None:
    """main() must report a configuration error, not crash and not do work."""
    # A couple of scripts mkdir() before validating; chdir keeps any such stray
    # output (e.g. a literal "Path\To\Your\Output_Folder") out of the working
    # tree rather than letting it land wherever pytest was started.
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module(".".join(script.relative_to(REPO_ROOT).with_suffix("").parts))
    exit_code = module.main(argv=[]) if _takes_argv(script) else module.main()

    assert exit_code == 2, (
        f"{_script_id(script)}: main() returned {exit_code!r} on its shipped "
        "placeholder CONFIGURATION; the contract requires 2 (see CONTRIBUTING.md)"
    )


@pytest.mark.parametrize("script", _params(mark_known=False))
def test_main_requires_no_arguments(script: Path) -> None:
    """main() must be callable with no arguments, so a notebook cell can run it."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert main is not None, f"{_script_id(script)}: no top-level main()"
    positional = main.args.posonlyargs + main.args.args
    assert len(positional) == len(main.args.defaults), (
        f"{_script_id(script)}: main() has a parameter with no default; every "
        "CONFIGURATION value must be editable in the script itself"
    )


def test_known_violations_are_all_real_scripts() -> None:
    """A renamed or deleted script must not leave a stale exemption behind."""
    missing = sorted(name for name in KNOWN_VIOLATIONS if not (SCRIPTS_ROOT / name).is_file())
    assert not missing, f"KNOWN_VIOLATIONS names files that no longer exist: {missing}"
