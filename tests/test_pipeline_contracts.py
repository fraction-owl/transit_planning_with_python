"""Drift tests for helper functions that are deliberately copied between scripts.

Several contracts in this repo are maintained as verbatim copies so each script
stays self-contained (runnable in a bare ArcGIS Pro / notebook environment with
no project on sys.path):

    - ``_canonical_key``: join-key normalization shared by the split modeling
      pipeline. The anchor and the feature bundles are produced on different
      machines, so a one-character divergence here produces silent partial joins
      that neither box can debug alone.
    - ``normalise_route``: the NTD-side route-token folding that
      ``_canonical_key`` must agree with, or anchor route ids never match
      GTFS-derived bundle keys.
    - ``extract_config_block``: the run-log helper whose canonical version lives
      in ``utils/run_log.py``.

The key-normalization copies are compared by AST (no imports, so the
arcpy-dependent scripts are checked too); the run-log copies are checked
behaviorally against the canonical, since they legitimately differ in
structure. Cross-family behavioral checks keep the two normalization
families aligned.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pandas as pd
import pytest

import scripts.modeling.route_performance_model as rpm
import scripts.ridership_tools.ntd_anchor_builder as nab
import scripts.ridership_tools.ntd_monthly_trends_export as trends
from utils import run_log

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_KEY_FILES = [
    "scripts/modeling/route_performance_model.py",
    "scripts/modeling/monthly_ridership_model.py",
    "scripts/modeling/prep_features_public.py",
    "scripts/modeling/prep_features_private.py",
    "scripts/gtfs_exports/gtfs_route_features.py",
]

NORMALISE_ROUTE_FILES = [
    "scripts/ridership_tools/ntd_anchor_builder.py",
    "scripts/ridership_tools/ntd_monthly_trends_export.py",
]


def _function_fingerprint(path: Path, name: str) -> str:
    """AST dump of *name*'s signature and body (docstring excluded) in *path*.

    Comparing dumps instead of source text tolerates docstring and formatting
    differences while catching any change to actual behavior.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]  # drop the docstring
            args_dump = ast.dump(node.args)
            body_dump = "\n".join(ast.dump(stmt) for stmt in body)
            return f"{args_dump}\n{body_dump}"
    raise AssertionError(f"Function '{name}' not found in {path}")


# ---------------------------------------------------------------------------
# _canonical_key: the five copies must be byte-for-byte the same logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relpath", CANONICAL_KEY_FILES[1:])
def test_canonical_key_copies_are_identical(relpath: str) -> None:
    reference = _function_fingerprint(REPO_ROOT / CANONICAL_KEY_FILES[0], "_canonical_key")
    assert _function_fingerprint(REPO_ROOT / relpath, "_canonical_key") == reference, (
        f"_canonical_key in {relpath} has drifted from {CANONICAL_KEY_FILES[0]}. "
        "These copies define the join contract across the secured/unsecured "
        "boundary and must stay identical."
    )


# ---------------------------------------------------------------------------
# normalise_route: the NTD-side copies must behave identically...
# ---------------------------------------------------------------------------

ROUTE_TOKEN_SAMPLES = ["  rt 5 ", "Rt 5", "RT5", "101", "101.0", 610.0, " A-12 ", "sun 30", 42]


@pytest.mark.parametrize("value", ROUTE_TOKEN_SAMPLES)
def test_normalise_route_copies_behave_identically(value: object) -> None:
    assert nab.normalise_route(value) == trends.normalise_route(value)


# ---------------------------------------------------------------------------
# ...and _canonical_key must agree with normalise_route, or anchor route ids
# ("RT5") silently never match GTFS-derived bundle keys ("Rt 5").
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ROUTE_TOKEN_SAMPLES)
def test_canonical_key_agrees_with_normalise_route(value: object) -> None:
    canonical = rpm._canonical_key(pd.Series([value])).iloc[0]
    assert canonical == nab.normalise_route(value), (
        f"_canonical_key({value!r}) = {canonical!r} but normalise_route gives "
        f"{nab.normalise_route(value)!r}; anchor and bundle keys will not join."
    )


# ---------------------------------------------------------------------------
# extract_config_block: every script copy identical; behavior matches canonical
# ---------------------------------------------------------------------------


def _extract_config_block_copies() -> list[Path]:
    """Every script carrying a copy of extract_config_block (excludes the canonical)."""
    copies = [
        p
        for p in (REPO_ROOT / "scripts").rglob("*.py")
        if "\ndef extract_config_block" in p.read_text(encoding="utf-8")
    ]
    assert len(copies) >= 10, "extract_config_block copies vanished — did a rename break discovery?"
    return sorted(copies)


# Unlike _canonical_key, the run-log copies legitimately differ in structure
# (some bind the markers locally, one delegates to a text-based variant), so the
# contract checked here is behavioral: every copy must slice a config block and
# reject missing markers exactly like the canonical utils/run_log.py.
@pytest.mark.parametrize(
    "relpath",
    [str(p.relative_to(REPO_ROOT)) for p in _extract_config_block_copies()],
)
def test_extract_config_block_copies_match_canonical_behavior(relpath: str, tmp_path: Path) -> None:
    module_name = ".".join(Path(relpath).with_suffix("").parts)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, SystemExit) as exc:
        # e.g. arcpy scripts outside ArcGIS Pro, or ridership_ml_model without
        # scikit-learn — those environments simply can't host this copy.
        pytest.skip(f"cannot import {relpath}: {exc}")

    good = tmp_path / "good.py"
    good.write_text(
        "x = 1\n# === BEGIN CONFIG ===\nA = 1\nB = 2  # inline\n# === END CONFIG ===\ny = 2\n",
        encoding="utf-8",
    )
    expected = run_log.extract_config_block(good)
    assert expected == "A = 1\nB = 2  # inline"
    assert module.extract_config_block(good) == expected

    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        module.extract_config_block(bad)
