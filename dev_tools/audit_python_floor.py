#!/usr/bin/env python3
"""Audit the two-tier Python version floor across scripts/ and utils/.

The repository supports two runtimes:

* **ArcPro tier** — scripts meant to run inside ArcGIS Pro's bundled Python
  (3.9 on ArcGIS Pro 3.0/3.1). This covers every ``*_arcpy`` script *and*
  every script whose imports are satisfied by the Pro-bundled stack
  (see requirements-arcpro.txt). These files must stay Python 3.9 compatible.
* **Open-source tier** — scripts importing the open-source geospatial stack
  (geopandas, shapely, rapidfuzz, ...), which is not available inside
  ArcGIS Pro and whose pinned versions require newer Python. These files
  follow the repo-wide 3.12 baseline and are not checked here.

Tier membership is derived from each file's imports, not its filename,
because the ``_gpd`` suffix is not applied consistently.

For ArcPro-tier files the audit flags:

1. Syntax not accepted by Python 3.9 (via ``ast.parse(feature_version=(3, 9))``,
   e.g. ``match`` statements). This is best-effort: ``feature_version`` does not
   reject every post-3.9 construct, and 3.10+ *library* APIs (e.g.
   ``itertools.pairwise``) are out of scope.
2. ``X | Y`` union syntax in annotations that CPython 3.9 evaluates at runtime:
   function signatures and module/class-level annotated assignments. Local
   variable annotations are never evaluated, so they are allowed. Files with
   ``from __future__ import annotations`` are exempt from this check.

Exit status:
    0 - all ArcPro-tier files are Python 3.9 compatible
    1 - violations detected (each is logged with file and line)
    2 - unrecoverable error (e.g. unreadable source file)
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Directories whose .py files ship to end users and therefore carry a tier.
AUDITED_DIRS: tuple[str, ...] = ("scripts", "utils")

# Importing any of these marks a file as open-source tier: the packages are
# absent from ArcGIS Pro's bundled environment, and the versions pinned in
# requirements.txt require Python >= 3.10 anyway.
OPEN_SOURCE_ONLY_LIBS: frozenset[str] = frozenset(
    {
        "contextily",
        "fiona",
        "geopandas",
        "momepy",
        "osmnx",
        "pyproj",
        "rapidfuzz",
        "rtree",
        "shapely",
    }
)

ARCPRO_FLOOR: tuple[int, int] = (3, 9)

LOGGER = logging.getLogger("audit_python_floor")

# =============================================================================
# TIER CLASSIFICATION
# =============================================================================


def imported_top_level_modules(tree: ast.AST) -> set[str]:
    """Return the top-level module names imported anywhere in *tree*."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def is_arcpro_tier(tree: ast.AST) -> bool:
    """Return True when a file must run in ArcGIS Pro's bundled Python.

    ``arcpy`` importers obviously run there; everything else defaults to the
    ArcPro tier too (README Option A pastes any script into a Pro notebook)
    unless it imports a package that only exists in the open-source stack.
    """
    modules = imported_top_level_modules(tree)
    if "arcpy" in modules:
        return True
    return not (modules & OPEN_SOURCE_ONLY_LIBS)


# =============================================================================
# COMPATIBILITY CHECKS
# =============================================================================


def syntax_violation(source: str, path: Path) -> str | None:
    """Return a message if *source* fails to parse as Python 3.9 syntax."""
    try:
        ast.parse(source, filename=str(path), feature_version=ARCPRO_FLOOR)
    except SyntaxError as exc:
        return f"{path}:{exc.lineno}: syntax not valid on Python 3.9 ({exc.msg})"
    return None


def _annotations_evaluated_at_runtime(tree: ast.Module) -> list[ast.expr]:
    """Collect annotation expressions that CPython 3.9 evaluates at runtime.

    Function-signature annotations are evaluated at ``def`` time; annotated
    assignments are evaluated at module and class scope (but not inside
    function bodies, so those are skipped).
    """
    collected: list[ast.expr] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in (
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                args.vararg,
                args.kwarg,
            ):
                if arg is not None and arg.annotation is not None:
                    collected.append(arg.annotation)
            if node.returns is not None:
                collected.append(node.returns)

    def collect_annassigns(body: list[ast.stmt]) -> None:
        """Recurse through statement blocks, skipping function bodies."""
        for stmt in body:
            if isinstance(stmt, ast.AnnAssign):
                collected.append(stmt.annotation)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # local variable annotations are never evaluated
            else:
                for _, value in ast.iter_fields(stmt):
                    if isinstance(value, list) and all(isinstance(s, ast.stmt) for s in value):
                        collect_annassigns(value)

    collect_annassigns(tree.body)

    return collected


def union_annotation_violations(tree: ast.Module, path: Path) -> list[str]:
    """Return messages for ``X | Y`` annotations that break at runtime on 3.9."""
    has_future_annotations = any(
        isinstance(stmt, ast.ImportFrom)
        and stmt.module == "__future__"
        and any(alias.name == "annotations" for alias in stmt.names)
        for stmt in tree.body
    )
    if has_future_annotations:
        return []

    violations: list[str] = []
    for annotation in _annotations_evaluated_at_runtime(tree):
        for sub in ast.walk(annotation):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                violations.append(
                    f"{path}:{sub.lineno}: `X | Y` union annotation is evaluated at "
                    "runtime and fails on Python 3.9 — add "
                    "`from __future__ import annotations` or use typing.Optional/Union"
                )
                break  # one report per annotation expression is enough
    return violations


# =============================================================================
# DRIVER
# =============================================================================


def audit_file(path: Path) -> list[str] | None:
    """Audit one file; return violation messages, or None if open-source tier."""
    source = path.read_text(encoding="utf-8")
    path = path.relative_to(REPO_ROOT)
    tree = ast.parse(source, filename=str(path))

    if not is_arcpro_tier(tree):
        return None

    violations: list[str] = []
    syntax_msg = syntax_violation(source, path)
    if syntax_msg is not None:
        violations.append(syntax_msg)
    violations.extend(union_annotation_violations(tree, path))
    return violations


def main() -> int:
    """Run the audit over every .py file in the audited directories."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    files = sorted(
        candidate
        for directory in AUDITED_DIRS
        for candidate in (REPO_ROOT / directory).rglob("*.py")
    )
    if not files:
        LOGGER.error("no Python files found under %s — wrong working directory?", AUDITED_DIRS)
        return 2

    arcpro_count = 0
    open_source_count = 0
    all_violations: list[str] = []

    for path in files:
        try:
            violations = audit_file(path)
        except (OSError, SyntaxError) as exc:
            LOGGER.error("cannot audit %s: %s", path, exc)
            return 2
        if violations is None:
            open_source_count += 1
            continue
        arcpro_count += 1
        all_violations.extend(violations)

    LOGGER.info(
        "audited %d files: %d ArcPro-tier (Python %s floor), %d open-source tier (skipped)",
        len(files),
        arcpro_count,
        ".".join(map(str, ARCPRO_FLOOR)),
        open_source_count,
    )

    if all_violations:
        for message in all_violations:
            LOGGER.error(message)
        LOGGER.error(
            "%d violation(s): the files above run in ArcGIS Pro's bundled Python "
            "and must stay Python 3.9 compatible (see README Requirements).",
            len(all_violations),
        )
        return 1

    LOGGER.info("all ArcPro-tier files are Python 3.9 compatible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
