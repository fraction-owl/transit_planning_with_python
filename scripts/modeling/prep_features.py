"""Drop-folder feature-prep orchestrator (PART A of a two-stage split).

This is the unsecured-box half of a split pipeline that keeps proprietary NTD
data on one machine while feature prep happens elsewhere:

    PART A (this script, runs anywhere, NO NTD)
        Runs the non-NTD feature-generation scripts as subprocesses, collects
        the tabular outputs they write, describes each output's join keys and
        shippable columns (registry-or-inference), then groups everything by
        join-key signature and writes one CSV bundle per signature plus a
        manifest (row counts + per-bundle SHA-256 + provenance).

    PART B (``fit_model.py``, runs only where the NTD anchor lives)
        Loads the NTD anchor, verifies and joins each prepped bundle onto it,
        then fits the regression and exports results.

Pipeline (front half is new; the "combine" back half is unchanged):

    scan -> extract -> run -> collect -> describe -> combine

    scan      Discover feature scripts (``SCRIPTS_DIR`` and an optional inbox)
              and an optional ``jobs.json`` registry.
    extract   Optionally unzip ``*.zip`` archives found under the input root
              (zip-slip guarded), so a human can drop either loose files or
              zips.
    run       Invoke each script as a subprocess using a per-script command
              template (from the registry, else ``DEFAULT_CMD_TEMPLATE``),
              each into its own output subdir, capturing logs, applying a
              per-script timeout, and recording the exit code.
    collect   Gather the ``*.csv`` / ``*.xlsx`` tables each successful script
              produced.
    describe  Resolve each table's join keys + kept columns from the registry
              when present, else by inference, building a ``list[FeatureTable]``
              dynamically instead of hardcoding it.
    combine   Run the existing back half: ``validate_config`` (fail-closed
              governance), then ``write_bundles`` -> ``write_manifest`` ->
              ``write_run_log``.

Design notes:
    - One bundle per join-key signature. Tables that share a join key (e.g. all
      the ``route_id`` coverage tables) are outer-merged into a single bundle; a
      differently keyed table (e.g. ``period``-keyed exogenous series) goes to
      its own bundle. Part B then joins each bundle only if the anchor carries
      its keys, which is what makes ``period`` optional downstream.
    - Run-what-you-can, fail-closed only on leakage. A script that exits
      non-zero or times out is logged and skipped; the run continues. The only
      hard failures are (a) a shipped column matching ``FORBIDDEN_COLUMN_PATTERNS``
      (caught by ``validate_config`` *before* any bundle is written) and (b) zero
      usable bundles.
    - CSV only. Bundles cross to a stock ``arcgispro-py3`` env that has no
      ``pyarrow``, so Parquet is deliberately avoided.
    - This script must never read, require, or reference the NTD anchor; that is
      ``fit_model.py``'s job on the secured box.

Inputs:
    - Feature-generation scripts (``*.py``) plus their input data, and an
      optional ``jobs.json`` registry describing each script's command and the
      join keys / kept columns of its outputs.

Outputs:
    - One CSV bundle per join-key signature in ``OUTPUT_DIR``.
    - ``manifest.json`` describing every bundle (keys, shape, SHA-256, and the
      ``produced_by`` provenance of the script(s) that fed it).
    - A run log capturing the verbatim configuration block plus, per script, its
      exit code, duration, and collected output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shlex
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Final, NamedTuple, Sequence

import pandas as pd

# Sentinel markers used by extract_config_block / write_run_log to identify
# the configuration block within this file's source. Each string must appear
# exactly once in this file as a stand-alone comment line (other than these
# constant definitions themselves). Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# Path to this file, used to extract the config block for the run log. ``__file__``
# is undefined when the code is pasted into a notebook cell, so a configured
# fallback keeps the run log working there too.
SELF_PATH: Final[Path] = Path(__file__) if "__file__" in globals() else Path("prep_features.py")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class FeatureTable(NamedTuple):
    """A non-NTD predictor source to be bundled for transfer to the secured box.

    Built dynamically by the orchestrator (one per collected output table)
    rather than hardcoded. The combine half consumes ``path`` / ``join_keys`` /
    ``keep_cols`` / ``sheet``; the ``source_*`` fields carry provenance through
    to the manifest and run log.

    Attributes:
        label: Human-readable name used in logs (e.g. ``"headway/headway.csv"``).
        path: Path to the CSV or XLSX file.
        join_keys: Columns the table is keyed on; tables sharing a key signature
            are merged into one bundle.
        keep_cols: Predictor columns to ship. REQUIRED and non-empty — the
            boundary is enumerated explicitly, so there is no "keep everything"
            shortcut on this side.
        sheet: Worksheet name for XLSX inputs (ignored for CSV).
        source_script: Filename of the script that produced this table, if known.
        source_exit_code: Exit code of that script's subprocess run.
        source_log: Path to that script's captured log, if known.
    """

    label: str
    path: Path
    join_keys: tuple[str, ...]
    keep_cols: tuple[str, ...]
    sheet: str | int = 0
    source_script: str | None = None
    source_exit_code: int | None = None
    source_log: str | None = None


class BundleResult(NamedTuple):
    """A written bundle and the metadata recorded for it in the manifest."""

    filename: str
    join_keys: tuple[str, ...]
    n_rows: int
    n_cols: int
    columns: list[str]
    sha256: str
    produced_by: tuple[dict[str, object], ...] = ()


class ScriptRun(NamedTuple):
    """The outcome of running one feature script as a subprocess."""

    script: str
    cmd: list[str]
    exit_code: int
    duration_s: float
    log_path: str
    timed_out: bool
    output_files: list[str]


# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# -----------------------------------------------------------------------------
#  Locations
# -----------------------------------------------------------------------------

# Folder scanned for feature-generation scripts (``*.py``).
SCRIPTS_DIR: Path = Path(r"Path\To\Your\scripts")

# Input-data root. Topic subfolders (e.g. ``gtfs``, ``shapefiles``, ``census``)
# are referenced by each script's command template via the ``{input}`` placeholder.
INPUT_DIR: Path = Path(r"Path\To\Your\input")

# Where the bundle CSVs and manifest.json are written for transfer to Part B.
OUTPUT_DIR: Path = Path(r"Path\To\Your\prepped_features")

# Scratch space for per-script output subdirs, staged data, and captured logs.
WORK_DIR: Path = Path(r"Path\To\Your\work")

# Optional drop/inbox folder. When set, ``*.py`` found here are added to the run
# set and ``*.zip`` found here are extracted into INPUT_DIR. Leave None to use
# only SCRIPTS_DIR + INPUT_DIR.
INBOX_DIR: Path | None = None

# Optional registry (jobs.json) mapping each script to its command template and
# each output file to its join keys / kept columns. Leave None to auto-discover
# scripts and infer keys/columns.
REGISTRY_PATH: Path | None = None

# -----------------------------------------------------------------------------
#  Execution behaviour
# -----------------------------------------------------------------------------

# When True, every ``*.zip`` found under INPUT_DIR (and INBOX_DIR) is extracted
# before scripts run, so humans can drop loose files OR zips ("both" model).
EXTRACT_ZIPS: bool = True

# Per-script wall-clock timeout in seconds (overridable per script in the registry).
PER_SCRIPT_TIMEOUT_SEC: int = 1800

# Candidate join keys, in priority order, used by inference when a collected
# output is not described in the registry.
CANDIDATE_JOIN_KEYS: tuple[str, ...] = ("route_id", "period")

# Command template used for any script lacking an explicit registry ``cmd``.
# Tokens are substituted with {python}, {script}, {input}, {output}, {work},
# {scripts}. Provided as a token list so paths with spaces/backslashes are not
# re-parsed by a shell.
DEFAULT_CMD_TEMPLATE: tuple[str, ...] = (
    "{python}",
    "{script}",
    "--input-dir",
    "{input}",
    "--output-dir",
    "{output}",
)

# Scripts never run when auto-discovering (no registry). The orchestrator and
# the secured-box model are always excluded; the latter must never run here.
EXCLUDE_SCRIPT_NAMES: tuple[str, ...] = (
    "prep_features.py",
    "fit_model.py",
    "ridership_regression_model.py",
    "ridership_ml_model.py",
    "__init__.py",
)
EXCLUDE_SCRIPT_GLOBS: tuple[str, ...] = ("test_*", "conftest*", "_*")

# -----------------------------------------------------------------------------
#  Governance guards (fail closed)
# -----------------------------------------------------------------------------

# Optional path to the NTD anchor. If set, the run aborts should any described
# feature table resolve to it — a defense against accidentally bundling the
# proprietary anchor on the unsecured box. Leave None if the anchor path is not
# known here (the forbidden-pattern denylist below is the always-on check).
ANCHOR_PATH_GUARD: Path | None = None

# Case-insensitive regex patterns matched against every shipped (non-key)
# column name. Any match aborts the run BEFORE any bundle is written. These
# target NTD-sourced quantities (boardings, unlinked passenger trips, revenue
# hours/miles) that must never cross the boundary in a feature bundle — they
# belong only in the anchor.
FORBIDDEN_COLUMN_PATTERNS: tuple[str, ...] = (
    r"boarding",
    r"ridership",
    r"\bntd\b",
    r"\bupt\b",
    r"unlinked",
    r"passenger_trip",
    r"revenue_mile",
    r"revenue_hour",
    r"scheduled_hour",
    r"\bvrm\b",
    r"\bvrh\b",
)

# -----------------------------------------------------------------------------
#  Output behaviour
# -----------------------------------------------------------------------------

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so a bundle directory is
# never left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# SHARED HELPERS (copied into fit_model.py — keep both copies in sync)
# =============================================================================


def load_table(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame.

    Args:
        path: Path to a ``.csv``, ``.xlsx``, or ``.xls`` file.
        sheet: Worksheet name or index for Excel inputs (ignored for CSV).

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported file type '{suffix}' for {path}. Use .csv, .xlsx, or .xls.")


def _canonical_key(series: pd.Series) -> pd.Series:
    """Normalize a join-key column so the anchor and a bundle match reliably.

    The anchor and bundle are produced on different machines from different
    files, so a key can arrive as ``101`` on one side and ``"101"`` (or
    ``"101.0"`` from a float round-trip) on the other. This collapses every
    key to a trimmed string and strips a single trailing ``.0`` so an integer
    that survived a float cast still matches its string form. The same helper
    is copied into fit_model.py and MUST stay byte-identical between the two.
    """
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.fillna("")


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# VALIDATION
# =============================================================================


def validate_config(
    tables: list[FeatureTable],
    *,
    forbidden_patterns: Sequence[str] = FORBIDDEN_COLUMN_PATTERNS,
    anchor_guard: Path | None = ANCHOR_PATH_GUARD,
) -> None:
    """Run the fail-closed governance checks before any bundle is written.

    This is the single fail-closed point (brief §6). It is intentionally run on
    the inferred keep_cols too, so a script that emits an NTD-like column aborts
    the run rather than having that column silently dropped.

    Args:
        tables: The dynamically built feature tables to ship.
        forbidden_patterns: Case-insensitive regexes; any match in a shipped
            column name aborts the run.
        anchor_guard: If set, any table resolving to this path aborts the run.

    Raises:
        ValueError: If a table declares no keep_cols, a shipped column matches a
            forbidden pattern, or a table resolves to the guarded anchor path.
    """
    forbidden = [re.compile(p, re.IGNORECASE) for p in forbidden_patterns]
    guard = anchor_guard.resolve() if anchor_guard is not None else None

    problems: list[str] = []
    for ft in tables:
        if not ft.keep_cols:
            problems.append(
                f"'{ft.label}' declares no keep_cols (the boundary must be enumerated)."
            )

        if guard is not None:
            try:
                if ft.path.resolve() == guard:
                    problems.append(f"'{ft.label}' resolves to the guarded NTD anchor path.")
            except OSError:
                pass  # resolve() can fail on non-existent paths; the loader reports those later.

        overlap = set(ft.keep_cols) & set(ft.join_keys)
        if overlap:
            problems.append(f"'{ft.label}' lists join key(s) {sorted(overlap)} in keep_cols.")

        for col in ft.keep_cols:
            for pat in forbidden:
                if pat.search(col):
                    problems.append(
                        f"'{ft.label}' ships column '{col}', which matches forbidden "
                        f"pattern /{pat.pattern}/ (looks NTD-derived)."
                    )

    if problems:
        raise ValueError(
            "Configuration failed governance checks (fail closed):\n  - " + "\n  - ".join(problems)
        )


# =============================================================================
# FRONT HALF: SCAN -> EXTRACT -> RUN -> COLLECT -> DESCRIBE
# =============================================================================


def load_registry(path: Path | None) -> tuple["OrderedDict[str, dict]", dict[str, dict]]:
    """Parse the optional jobs.json registry.

    Returns:
        ``(scripts, outputs)`` where ``scripts`` maps a script filename to
        ``{"cmd", "timeout_sec", "outputs": {filename: spec}}`` (order
        preserved), and ``outputs`` is a global ``{filename: spec}`` map. A
        ``spec`` is ``{"join_keys", "keep_cols", "sheet"}``. Both are empty when
        ``path`` is None or missing.

    Raises:
        ValueError: If the registry JSON is malformed.
    """
    scripts: "OrderedDict[str, dict]" = OrderedDict()
    outputs: dict[str, dict] = {}
    if path is None or not Path(path).exists():
        return scripts, outputs

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Registry '{path}' is not valid JSON: {exc}") from exc

    def _spec(entry: dict) -> dict:
        return {
            "join_keys": tuple(entry["join_keys"]),
            "keep_cols": tuple(entry["keep_cols"]),
            "sheet": entry.get("sheet", 0),
        }

    for entry in data.get("scripts", []):
        name = str(entry["script"])
        per_file = {str(o["file"]): _spec(o) for o in entry.get("outputs", [])}
        scripts[name] = {
            "cmd": entry.get("cmd"),
            "timeout_sec": entry.get("timeout_sec"),
            "outputs": per_file,
        }
    for o in data.get("outputs", []):
        outputs[str(o["file"])] = _spec(o)

    logging.info(
        "Loaded registry '%s': %d script entr(ies), %d global output spec(s).",
        path,
        len(scripts),
        len(outputs),
    )
    return scripts, outputs


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract ``zip_path`` into ``dest_dir``, rejecting zip-slip members.

    Raises:
        ValueError: If any member would resolve outside ``dest_dir``.
    """
    dest = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if target != dest and dest not in target.parents:
                raise ValueError(
                    f"Refusing zip-slip member '{member}' in '{zip_path}' "
                    f"(escapes target '{dest}')."
                )
        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)


def extract_zips(input_dir: Path, inbox_dir: Path | None) -> list[Path]:
    """Extract archives so loose files and zips both end up under the input root.

    Every ``*.zip`` under ``input_dir`` is extracted into a sibling folder named
    after the archive stem; any ``*.zip`` in ``inbox_dir`` is extracted into
    ``input_dir/<stem>``. Targets that already exist and are non-empty are left
    alone, so the step is safe to re-run and a human may pre-extract.

    Returns:
        The list of destination directories that were written.
    """
    written: list[Path] = []
    candidates: list[tuple[Path, Path]] = []
    if input_dir.exists():
        for zp in sorted(input_dir.rglob("*.zip")):
            candidates.append((zp, zp.parent / zp.stem))
    if inbox_dir is not None and inbox_dir.exists():
        for zp in sorted(inbox_dir.glob("*.zip")):
            candidates.append((zp, input_dir / zp.stem))

    for zip_path, dest in candidates:
        if dest.exists() and any(dest.iterdir()):
            logging.info("Skipping '%s'; target '%s' already populated.", zip_path.name, dest)
            continue
        _safe_extract_zip(zip_path, dest)
        logging.info("Extracted '%s' -> '%s'.", zip_path.name, dest)
        written.append(dest)
    return written


def discover_scripts(
    scripts_dir: Path,
    inbox_dir: Path | None,
    registry_scripts: "OrderedDict[str, dict]",
    *,
    exclude_names: Sequence[str] = EXCLUDE_SCRIPT_NAMES,
    exclude_globs: Sequence[str] = EXCLUDE_SCRIPT_GLOBS,
) -> list[tuple[str, Path, list[str] | str | None]]:
    """Resolve the ordered set of scripts to run.

    When the registry lists scripts it is an allowlist: only those are run, in
    registry order, each with its registry ``cmd``. Otherwise every ``*.py`` in
    ``scripts_dir`` (and ``inbox_dir``) is run except excluded names/globs.

    Returns:
        A list of ``(name, path, cmd)`` where ``cmd`` is the registry command
        (list/str) or None to use ``DEFAULT_CMD_TEMPLATE``.
    """
    found: dict[str, Path] = {}
    search_dirs = [scripts_dir] + ([inbox_dir] if inbox_dir is not None else [])
    for directory in search_dirs:
        if not directory.exists():
            continue
        for py in sorted(directory.glob("*.py")):
            found.setdefault(py.name, py)

    if registry_scripts:
        resolved: list[tuple[str, Path, list[str] | str | None]] = []
        for name, entry in registry_scripts.items():
            if name in found:
                resolved.append((name, found[name], entry.get("cmd")))
            else:
                logging.warning("Registry lists script '%s' but it was not found; skipping.", name)
        return resolved

    resolved = []
    for name, path in found.items():
        if name in exclude_names or any(fnmatch(name, g) for g in exclude_globs):
            continue
        resolved.append((name, path, None))
    return resolved


def resolve_cmd(
    script_path: Path,
    registry_cmd: list[str] | str | None,
    default_template: Sequence[str],
    placeholders: dict[str, str],
) -> list[str]:
    """Build the subprocess argv for a script from its command template.

    Args:
        script_path: Path to the script to run (already in ``placeholders``).
        registry_cmd: The registry ``cmd`` (token list or string) or None.
        default_template: Token list used when ``registry_cmd`` is None.
        placeholders: Substitution values for ``{python}`` / ``{script}`` / etc.

    Returns:
        A fully substituted argv token list.
    """
    cmd = registry_cmd if registry_cmd is not None else default_template
    tokens = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    return [str(token).format(**placeholders) for token in tokens]


def run_script(cmd_tokens: list[str], log_path: Path, timeout_sec: int) -> tuple[int, float, bool]:
    """Run one script subprocess, capturing stdout+stderr to ``log_path``.

    Returns:
        ``(exit_code, duration_s, timed_out)``. A timeout yields exit code -1.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[orchestrator] command: {cmd_tokens}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd_tokens,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -1
            log.write(f"\n[orchestrator] TIMEOUT after {timeout_sec}s\n")
    return exit_code, time.monotonic() - start, timed_out


def _log_tail(log_path: Path, max_lines: int = 12) -> str:
    """Return the last ``max_lines`` of a captured script log, indented for display."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "    | (log unavailable)"
    tail = [ln for ln in lines if ln.strip()][-max_lines:]
    return "\n".join(f"    | {ln}" for ln in tail) if tail else "    | (empty)"


def collect_outputs(out_dir: Path) -> list[Path]:
    """Return the tabular files (``*.csv`` / ``*.xlsx`` / ``*.xls``) under ``out_dir``."""
    files: list[Path] = []
    for pattern in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(out_dir.rglob(pattern))
    return sorted(files)


def _is_shippable_dtype(series: pd.Series) -> bool:
    """Return True for numeric/boolean columns (the only inference-shippable ones)."""
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def describe_output(
    file: Path,
    run: ScriptRun,
    script_outputs: dict[str, dict],
    global_outputs: dict[str, dict],
    *,
    candidate_join_keys: Sequence[str] = CANDIDATE_JOIN_KEYS,
) -> FeatureTable | None:
    """Describe one collected output as a FeatureTable, or skip it.

    Resolution is registry-first: a per-script output spec wins, then a global
    output spec, then inference. Inference picks join keys from
    ``candidate_join_keys`` and ships every other numeric/boolean column.
    Forbidden columns are NOT filtered here — ``validate_config`` is the single
    fail-closed gate, so leakage aborts the run rather than being hidden.

    Returns:
        A FeatureTable, or None when the table cannot be joined / has nothing to
        contribute (skipped with a warning).
    """
    spec = script_outputs.get(file.name) or global_outputs.get(file.name)
    sheet = spec["sheet"] if spec else 0
    try:
        df = load_table(file, sheet)
    except (FileNotFoundError, ValueError) as exc:
        logging.warning("Could not read collected output '%s': %s — skipping.", file.name, exc)
        return None
    columns = list(df.columns)

    if spec is not None:
        join_keys = tuple(spec["join_keys"])
        keep_cols = tuple(spec["keep_cols"])
        missing = [c for c in (*join_keys, *keep_cols) if c not in columns]
        if missing:
            logging.warning(
                "Registry-described '%s' is missing column(s) %s — skipping.", file.name, missing
            )
            return None
    else:
        join_keys = tuple(k for k in candidate_join_keys if k in columns)
        if not join_keys:
            logging.warning(
                "Inferred no join key for '%s' (none of %s present) — skipping.",
                file.name,
                list(candidate_join_keys),
            )
            return None
        keep_cols = tuple(c for c in columns if c not in join_keys and _is_shippable_dtype(df[c]))
        if not keep_cols:
            logging.warning(
                "Inferred no shippable column for '%s' (nothing numeric/boolean) — skipping.",
                file.name,
            )
            return None

    return FeatureTable(
        label=f"{Path(run.script).stem}/{file.name}",
        path=file,
        join_keys=join_keys,
        keep_cols=keep_cols,
        sheet=sheet,
        source_script=run.script,
        source_exit_code=run.exit_code,
        source_log=run.log_path,
    )


# =============================================================================
# BUNDLING
# =============================================================================


def _bundle_filename(join_keys: tuple[str, ...]) -> str:
    """Derive a deterministic bundle filename from a join-key signature."""
    safe = "__".join(re.sub(r"[^0-9A-Za-z]+", "_", k) for k in join_keys)
    return f"features__{safe}.csv"


def group_tables_by_keys(
    tables: list[FeatureTable],
) -> "OrderedDict[tuple[str, ...], list[FeatureTable]]":
    """Group feature tables by their join-key signature, preserving order."""
    groups: OrderedDict[tuple[str, ...], list[FeatureTable]] = OrderedDict()
    for ft in tables:
        groups.setdefault(ft.join_keys, []).append(ft)
    return groups


def build_bundle(join_keys: tuple[str, ...], tables: list[FeatureTable]) -> pd.DataFrame:
    """Merge every table in one key-group into a single deduplicated frame.

    Args:
        join_keys: The shared join-key signature for this group.
        tables: Feature tables that all share ``join_keys``.

    Returns:
        The outer-merged bundle, keyed on ``join_keys`` with canonicalized keys.

    Raises:
        KeyError: If a table is missing one of its declared columns.
        ValueError: If two tables in the group ship the same value column.
    """
    keys = list(join_keys)
    bundle: pd.DataFrame | None = None
    seen_value_cols: dict[str, str] = {}  # value column -> originating label

    for ft in tables:
        df = load_table(ft.path, ft.sheet)

        missing_keys = [k for k in keys if k not in df.columns]
        if missing_keys:
            raise KeyError(f"'{ft.label}' is missing join key(s): {missing_keys}")
        missing_vals = [c for c in ft.keep_cols if c not in df.columns]
        if missing_vals:
            raise KeyError(f"'{ft.label}' is missing keep_cols: {missing_vals}")

        for col in ft.keep_cols:
            if col in seen_value_cols:
                raise ValueError(
                    f"Column '{col}' shipped by both '{seen_value_cols[col]}' and '{ft.label}' "
                    f"(same key group {join_keys}). Rename one before bundling."
                )
            seen_value_cols[col] = ft.label

        subset = df[keys + list(ft.keep_cols)].copy()
        for key in keys:
            subset[key] = _canonical_key(subset[key])

        before = len(subset)
        subset = subset.drop_duplicates(subset=keys)
        if len(subset) < before:
            logging.info(
                "'%s': dropped %d duplicate key row(s) before bundling.",
                ft.label,
                before - len(subset),
            )

        if bundle is None:
            bundle = subset
        else:
            bundle = bundle.merge(subset, on=keys, how="outer")
        logging.info(
            "Added '%s' to bundle %s: %d cols, %d unique keys.",
            ft.label,
            join_keys,
            len(ft.keep_cols),
            subset[keys].drop_duplicates().shape[0],
        )

    assert bundle is not None  # group is non-empty by construction
    return bundle


def write_bundles(tables: list[FeatureTable], output_dir: Path) -> list[BundleResult]:
    """Build, write, and hash one CSV bundle per join-key signature."""
    results: list[BundleResult] = []
    for join_keys, group in group_tables_by_keys(tables).items():
        frame = build_bundle(join_keys, group)
        filename = _bundle_filename(join_keys)
        out_path = output_dir / filename
        frame.to_csv(out_path, index=False, encoding="utf-8")

        digest = _sha256_file(out_path)
        produced_by: list[dict[str, object]] = []
        for ft in group:
            if ft.source_script is None:
                continue
            record: dict[str, object] = {
                "script": ft.source_script,
                "exit_code": ft.source_exit_code,
                "log": ft.source_log,
            }
            if record not in produced_by:
                produced_by.append(record)

        results.append(
            BundleResult(
                filename=filename,
                join_keys=join_keys,
                n_rows=int(frame.shape[0]),
                n_cols=int(frame.shape[1]),
                columns=list(frame.columns),
                sha256=digest,
                produced_by=tuple(produced_by),
            )
        )
        logging.info(
            "Wrote bundle '%s': %d rows x %d cols (sha256=%s…).",
            filename,
            frame.shape[0],
            frame.shape[1],
            digest[:12],
        )
    return results


def write_manifest(output_dir: Path, bundles: list[BundleResult]) -> Path:
    """Write manifest.json describing every bundle for Part B to verify."""
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_script": str(SELF_PATH.resolve() if SELF_PATH.exists() else SELF_PATH),
        "bundles": [
            {
                "filename": b.filename,
                "join_keys": list(b.join_keys),
                "n_rows": b.n_rows,
                "n_cols": b.n_cols,
                "columns": b.columns,
                "sha256": b.sha256,
                "produced_by": list(b.produced_by),
            }
            for b in bundles
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logging.info("Manifest written to '%s'.", manifest_path)
    return manifest_path


# =============================================================================
# RUN-LOG HELPERS (copied into fit_model.py — keep both copies in sync)
# =============================================================================


def extract_config_block(source_file: Path) -> str:
    """Return the text between the CONFIG markers in *source_file*.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def write_run_log(
    output_dir: Path,
    bundles: list[BundleResult],
    runs: list[ScriptRun],
    source_path: Path = SELF_PATH,
) -> bool:
    """Write a run log of the configuration block plus the script + bundle summary."""
    log_path = output_dir / "prep_features_runlog.txt"

    try:
        config_text: str = extract_config_block(source_path)
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    if runs:
        script_lines: list[str] = []
        for r in runs:
            status = "TIMEOUT" if r.timed_out else f"exit={r.exit_code}"
            files = ", ".join(r.output_files) if r.output_files else "(none)"
            script_lines.append(f"  {r.script}  {status}  {r.duration_s:.1f}s  outputs=[{files}]")
    else:
        script_lines = ["  (none)"]

    if bundles:
        bundle_lines = [
            f"  {b.filename}  keys={list(b.join_keys)}  rows={b.n_rows}  cols={b.n_cols}  "
            f"sha256={b.sha256}"
            for b in bundles
        ]
    else:
        bundle_lines = ["  (none)"]

    lines: list[str] = [
        "=" * 72,
        "FEATURE PREP RUN LOG (PART A — unsecured box, orchestrator)",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {source_path.resolve() if source_path.exists() else source_path}",
        "",
        "-" * 72,
        "SCRIPTS RUN",
        "-" * 72,
        *script_lines,
        "",
        "-" * 72,
        "BUNDLES WRITTEN",
        "-" * 72,
        *bundle_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# ORCHESTRATION
# =============================================================================


def orchestrate(
    scripts_dir: Path,
    input_dir: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    inbox_dir: Path | None = None,
    registry_path: Path | None = None,
    candidate_join_keys: Sequence[str] = CANDIDATE_JOIN_KEYS,
    per_script_timeout_sec: int = PER_SCRIPT_TIMEOUT_SEC,
    extract_zips_flag: bool = EXTRACT_ZIPS,
    default_cmd_template: Sequence[str] = DEFAULT_CMD_TEMPLATE,
    forbidden_patterns: Sequence[str] = FORBIDDEN_COLUMN_PATTERNS,
    anchor_guard: Path | None = ANCHOR_PATH_GUARD,
    exclude_names: Sequence[str] = EXCLUDE_SCRIPT_NAMES,
    exclude_globs: Sequence[str] = EXCLUDE_SCRIPT_GLOBS,
    require_run_log: bool = REQUIRE_RUN_LOG,
    source_path: Path = SELF_PATH,
) -> list[BundleResult]:
    """Run the full scan -> extract -> run -> collect -> describe -> combine pipeline.

    Returns:
        The written bundles.

    Raises:
        ValueError: On a governance failure (leakage / guarded anchor / bad
            keep_cols) — raised BEFORE any bundle is written.
        RuntimeError: If no usable bundles result.
    """
    registry_scripts, global_outputs = load_registry(registry_path)

    # --- EXTRACT ---------------------------------------------------------
    if extract_zips_flag:
        extract_zips(input_dir, inbox_dir)

    # --- SCAN ------------------------------------------------------------
    scripts = discover_scripts(
        scripts_dir,
        inbox_dir,
        registry_scripts,
        exclude_names=exclude_names,
        exclude_globs=exclude_globs,
    )
    if not scripts:
        logging.warning("No feature scripts discovered under %s.", scripts_dir)

    out_root = work_dir / "out"
    log_root = work_dir / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    # --- RUN -> COLLECT -> DESCRIBE -------------------------------------
    tables: list[FeatureTable] = []
    runs: list[ScriptRun] = []
    for name, path, registry_cmd in scripts:
        per_script_out = out_root / Path(name).stem
        per_script_out.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{Path(name).stem}.log"
        placeholders = {
            "python": sys.executable,
            "script": str(path),
            "scripts": str(scripts_dir),
            "input": str(input_dir),
            "output": str(per_script_out),
            "work": str(work_dir),
        }
        cmd_tokens = resolve_cmd(path, registry_cmd, default_cmd_template, placeholders)
        timeout = per_script_timeout_sec
        if name in registry_scripts and registry_scripts[name].get("timeout_sec"):
            timeout = int(registry_scripts[name]["timeout_sec"])

        logging.info("Running '%s' (timeout %ds)...", name, timeout)
        exit_code, duration, timed_out = run_script(cmd_tokens, log_path, timeout)

        run = ScriptRun(
            script=name,
            cmd=cmd_tokens,
            exit_code=exit_code,
            duration_s=duration,
            log_path=str(log_path),
            timed_out=timed_out,
            output_files=[],
        )

        if timed_out or exit_code != 0:
            status = "timed out" if timed_out else f"exited {exit_code}"
            hint = ""
            if exit_code == 2:
                hint = (
                    " — exit 2 is an argument error: the command flags do not match this script's "
                    "CLI. Point REGISTRY_PATH (or --registry) at a jobs.json giving this script's "
                    "real cmd, or it falls back to DEFAULT_CMD_TEMPLATE (--input-dir/--output-dir)."
                )
            logging.warning(
                "Script '%s' %s — skipping its outputs (continuing).%s\nLast log lines (%s):\n%s",
                name,
                status,
                hint,
                log_path,
                _log_tail(log_path),
            )
            runs.append(run)
            continue

        collected = collect_outputs(per_script_out)
        run = run._replace(output_files=[f.name for f in collected])
        runs.append(run)
        if not collected:
            logging.warning("Script '%s' produced no tables.", name)
            continue

        script_outputs = registry_scripts.get(name, {}).get("outputs", {})
        for file in collected:
            ft = describe_output(
                file,
                run,
                script_outputs,
                global_outputs,
                candidate_join_keys=candidate_join_keys,
            )
            if ft is not None:
                tables.append(ft)
                logging.info(
                    "Described '%s': keys=%s, keep_cols=%s.",
                    ft.label,
                    list(ft.join_keys),
                    list(ft.keep_cols),
                )

    # --- COMBINE ---------------------------------------------------------
    # Fail-closed governance runs BEFORE any bundle is written (brief §6).
    validate_config(tables, forbidden_patterns=forbidden_patterns, anchor_guard=anchor_guard)
    logging.info("Governance checks passed: %d feature table(s) cleared.", len(tables))

    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = write_bundles(tables, output_dir)
    if not bundles:
        raise RuntimeError(
            "No usable bundles were produced. Every feature script failed or emitted no "
            f"joinable table. See the per-script logs in {work_dir / 'logs'} for the exact "
            "errors. If scripts exited with code 2, their command flags do not match — point "
            "REGISTRY_PATH (or --registry) at a jobs.json describing each script's real cmd "
            "and input subfolders."
        )

    write_manifest(output_dir, bundles)
    if not write_run_log(output_dir, bundles, runs, source_path) and require_run_log:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    logging.info(
        "All processing complete. %d bundle(s) ready for transfer to the secured box.",
        len(bundles),
    )
    return bundles


# =============================================================================
# MAIN
# =============================================================================


def _source_path() -> Path:
    """Path to this file for run-log extraction, with a notebook-safe fallback."""
    return Path(__file__) if "__file__" in globals() else SELF_PATH


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIGURATION block.

    ``parse_known_args`` is used so a notebook kernel's injected argv does not
    raise ``SystemExit: 2``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Drop-folder feature-prep orchestrator (Part A). Runs feature scripts, "
            "collects their tables, and writes bundles + manifest for fit_model.py. "
            "Defaults come from the CONFIGURATION block at the top of this file."
        )
    )
    parser.add_argument("--scripts-dir", default=str(SCRIPTS_DIR), help="Folder of *.py scripts.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Input-data root.")
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR), help="Where bundles + manifest are written."
    )
    parser.add_argument("--work-dir", default=str(WORK_DIR), help="Scratch dir for outputs/logs.")
    parser.add_argument(
        "--inbox-dir",
        default=str(INBOX_DIR) if INBOX_DIR is not None else None,
        help="Optional drop folder for extra *.py and *.zip.",
    )
    parser.add_argument(
        "--registry",
        default=str(REGISTRY_PATH) if REGISTRY_PATH is not None else None,
        help="Optional jobs.json registry path.",
    )
    parser.add_argument(
        "--timeout", type=int, default=PER_SCRIPT_TIMEOUT_SEC, help="Per-script timeout (seconds)."
    )
    parser.add_argument(
        "--extract-zips",
        action=argparse.BooleanOptionalAction,
        default=EXTRACT_ZIPS,
        help="Extract *.zip under the input root before running.",
    )
    parser.add_argument(
        "--log-level",
        default=logging.getLevelName(LOG_LEVEL),
        help="DEBUG / INFO / WARNING / ERROR.",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. Defaults fall back to the CONFIGURATION block."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    placeholder = "Path\\To\\Your"
    if any(placeholder in str(p) for p in (args.scripts_dir, args.input_dir, args.output_dir)):
        logging.warning(
            "SCRIPTS_DIR / INPUT_DIR / OUTPUT_DIR are still placeholders. Update the "
            "CONFIGURATION block or pass --scripts-dir/--input-dir/--output-dir before running."
        )
        return

    try:
        orchestrate(
            scripts_dir=Path(args.scripts_dir),
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            work_dir=Path(args.work_dir),
            inbox_dir=Path(args.inbox_dir) if args.inbox_dir else None,
            registry_path=Path(args.registry) if args.registry else None,
            per_script_timeout_sec=args.timeout,
            extract_zips_flag=args.extract_zips,
            source_path=_source_path(),
        )
    except ValueError as exc:  # governance failure (fail closed)
        logging.error("Governance check failed (fail closed): %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logging.error("%s", exc)
        sys.exit(1)


def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":
    # In a notebook (pasted cell or %run), argparse would choke on the kernel's
    # own argv; parse_known_args already guards that, and main() with no argv is
    # safe in both contexts.
    main()
