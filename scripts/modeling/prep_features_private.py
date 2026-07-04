"""Secured-box feature-prep orchestrator (the PRIVATE half of the split).

This is the secured-box counterpart to ``prep_features_public.py``. The two are
symmetric at the *interface* -- each runs a set of feature scripts, collects
their tabular outputs, groups them by join-key signature, and writes one CSV
bundle per signature plus a manifest -- but they have opposite trust postures:

    prep_features_public.py (runs anywhere, NO NTD)
        A guarded *exporter*. Its whole reason to exist is a fail-closed
        boundary: it must never let an NTD-derived quantity cross to the
        unsecured box, so it runs a forbidden-column denylist before writing.

    prep_features_private.py (this script, runs only on the secured box)
        An internal *assembler*. The proprietary data (NTD ridership, plus the
        TIDES-derived OTP and runtime operational measures) all live here, so
        there is no boundary to police. The dependent variable (NTD boardings)
        and the service-supply predictors are *expected* in these bundles, not
        forbidden. Nothing produced here is meant to leave the secured box.

Pipeline (identical shape to the public half):

    scan -> extract -> run -> collect -> describe -> combine

    scan      Discover the private feature scripts under ``SCRIPTS_DIR``
              (searched recursively, since the private set spans
              ``ridership_tools`` and ``operations_tools``) using the
              ``jobs.private.json`` registry as an ordered allowlist.
    extract   Optionally unzip ``*.zip`` archives found under the input root
              (zip-slip guarded), so NTD workbooks / TIDES exports can be dropped
              either loose or zipped.
    run       Invoke each script as a subprocess using its registry command
              template, each into its own output subdir, capturing logs and
              applying a per-script timeout. Downstream scripts read an upstream
              script's output via ``{work}/out/<script_stem>/`` (e.g.
              ``otp_by_route`` consumes ``otp_monthly_tides``'s panel).
    collect   Gather the ``*.csv`` / ``*.xlsx`` tables each successful script
              produced.
    describe  Resolve each table's join keys + kept columns from the registry
              when present, else by inference, building a ``list[FeatureTable]``.
    combine   Validate (hygiene only -- no denylist), then ``write_bundles`` ->
              ``write_manifest`` -> ``write_run_log``.

Design notes:
    - One bundle per join-key signature, exactly like the public half. The
      route-level rollups (NTD cross-section anchor + OTP + runtime) merge into a
      single ``route_id`` bundle -- the "one route-level table" the OLS and ML
      models consume -- while any route x month panel collected (e.g. the runtime
      monthly panel) becomes its own bundle for the future monthly-change model.
    - Governance is inverted relative to the public half: there is NO forbidden
      column denylist and NO anchor guard. The only checks are hygiene (every
      shipped table must enumerate its keep_cols; a join key may not also be a
      shipped value; two tables in one key group may not ship the same column),
      plus an advisory check that the configured dependent variable is present.
    - CSV only, for parity with the public bundles and the stock
      ``arcgispro-py3`` env (no ``pyarrow``).

Inputs:
    - The private feature scripts (NTD anchor builder; OTP and runtime rollups)
      plus their input data, and a ``jobs.private.json`` registry describing each
      script's command and its outputs' join keys / kept columns.

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

# Sentinel markers used by extract_config_block / write_run_log to identify the
# configuration block within this file's source. Each string must appear exactly
# once in this file as a stand-alone comment line (other than these constant
# definitions themselves). Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# Path to this file, used to extract the config block for the run log. ``__file__``
# is undefined when the code is pasted into a notebook cell, so a configured
# fallback keeps the run log working there too.
SELF_PATH: Final[Path] = (
    Path(__file__) if "__file__" in globals() else Path("prep_features_private.py")
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class FeatureTable(NamedTuple):
    """A private predictor / anchor source to be bundled for the models.

    Built dynamically by the orchestrator (one per collected output table). The
    combine half consumes ``path`` / ``join_keys`` / ``keep_cols`` / ``sheet``;
    the ``source_*`` fields carry provenance through to the manifest and run log.

    Attributes:
        label: Human-readable name used in logs (e.g. ``"otp_by_route/otp_by_route.csv"``).
        path: Path to the CSV or XLSX file.
        join_keys: Columns the table is keyed on; tables sharing a key signature
            are merged into one bundle.
        keep_cols: Predictor / dependent columns to ship. REQUIRED and non-empty.
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

# Folder scanned (recursively) for the private feature scripts. The private set
# spans ridership_tools + operations_tools, so this points at the scripts root.
SCRIPTS_DIR: Path = Path(r"Path\To\Your\scripts")

# Input-data root. Topic subfolders (e.g. ``ntd``, ``tides``) are referenced by
# each script's command template via the ``{input}`` placeholder.
INPUT_DIR: Path = Path(r"Path\To\Your\input")

# Where the bundle CSVs and manifest.json are written for the models to ingest.
OUTPUT_DIR: Path = Path(r"Path\To\Your\private_features")

# Scratch space for per-script output subdirs, staged data, and captured logs.
WORK_DIR: Path = Path(r"Path\To\Your\work")

# Optional drop/inbox folder. When set, ``*.py`` found here are added to the run
# set and ``*.zip`` found here are extracted into INPUT_DIR. Leave None to use
# only SCRIPTS_DIR + INPUT_DIR.
INBOX_DIR: Path | None = None

# Registry (jobs.private.json) mapping each script to its command template and
# each output file to its join keys / kept columns. Unlike the public half this
# is effectively required: the private scripts have heterogeneous CLIs and the
# dependent / supply columns must be enumerated deliberately. Leave None to fall
# back to auto-discovery + inference.
REGISTRY_PATH: Path | None = None

# -----------------------------------------------------------------------------
#  Execution behaviour
# -----------------------------------------------------------------------------

# When True, every ``*.zip`` found under INPUT_DIR (and INBOX_DIR) is extracted
# before scripts run, so NTD workbooks / TIDES exports can be dropped loose OR
# zipped ("both" model).
EXTRACT_ZIPS: bool = True

# Per-script wall-clock timeout in seconds (overridable per script in the registry).
PER_SCRIPT_TIMEOUT_SEC: int = 1800

# Candidate join keys, in priority order, used by inference when a collected
# output is not described in the registry. Covers the route, panel, and stop
# grains the model family spans.
CANDIDATE_JOIN_KEYS: tuple[str, ...] = ("route_id", "period", "month", "stop_id")

# Command template used for any script lacking an explicit registry ``cmd``.
DEFAULT_CMD_TEMPLATE: tuple[str, ...] = (
    "{python}",
    "{script}",
    "--input-dir",
    "{input}",
    "--output-dir",
    "{output}",
)

# Scripts never run when auto-discovering (no registry). Both orchestrators and
# the secured-box models are always excluded.
EXCLUDE_SCRIPT_NAMES: tuple[str, ...] = (
    "prep_features_private.py",
    "prep_features_public.py",
    "prep_features.py",
    "monthly_ridership_model.py",
    "route_performance_model.py",
    "ridership_ml_model.py",
    "__init__.py",
)
EXCLUDE_SCRIPT_GLOBS: tuple[str, ...] = ("test_*", "conftest*", "_*")

# -----------------------------------------------------------------------------
#  Governance (hygiene only -- no denylist on this side)
# -----------------------------------------------------------------------------

# The dependent variable expected to appear in (at least) one shipped table on
# the secured box. This is NOT fail-closed: a private run with no DV (a
# features-only refresh) is allowed, but the absence is logged so a misconfigured
# run is easy to spot. Set "" to disable the advisory check entirely.
EXPECTED_DV_COLUMN: str = "ntd_boardings"

# -----------------------------------------------------------------------------
#  Output behaviour
# -----------------------------------------------------------------------------

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so a bundle directory is
# never left without a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# SHARED HELPERS (self-contained copy; mirrors prep_features_public.py)
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
    """Normalize a join-key column so two independently produced tables match.

    Collapses every key to a trimmed, upper-cased, space-free string and strips a
    single trailing ``.0`` — the same folding as ntd_anchor_builder's
    normalise_route, so alphanumeric route names join reliably.
    """
    out = series.astype("string").str.strip().str.upper()
    out = out.str.replace(" ", "", regex=False)
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
# VALIDATION (hygiene only)
# =============================================================================


def validate_config(
    tables: list[FeatureTable],
    *,
    expected_dv: str = EXPECTED_DV_COLUMN,
) -> None:
    """Run hygiene checks before any bundle is written.

    Unlike the public half this performs NO leakage denylist and NO anchor
    guard: the proprietary data legitimately lives here. The checks are purely
    structural, plus an advisory (non-fatal) check that the dependent variable is
    present somewhere so a misconfigured private run is easy to notice.

    Args:
        tables: The dynamically built feature tables to ship.
        expected_dv: Dependent-variable column expected somewhere; "" disables
            the advisory check.

    Raises:
        ValueError: If a table declares no keep_cols or lists a join key in its
            keep_cols.
    """
    problems: list[str] = []
    for ft in tables:
        if not ft.keep_cols:
            problems.append(
                f"'{ft.label}' declares no keep_cols (the boundary must be enumerated)."
            )
        overlap = set(ft.keep_cols) & set(ft.join_keys)
        if overlap:
            problems.append(f"'{ft.label}' lists join key(s) {sorted(overlap)} in keep_cols.")

    if problems:
        raise ValueError("Configuration failed hygiene checks:\n  - " + "\n  - ".join(problems))

    if expected_dv:
        carriers = [ft.label for ft in tables if expected_dv in ft.keep_cols]
        if not carriers:
            logging.warning(
                "Dependent variable '%s' is not shipped by any table. This is allowed "
                "(features-only refresh), but a model run will need the DV from elsewhere.",
                expected_dv,
            )
        else:
            logging.info(
                "Dependent variable '%s' present in: %s.", expected_dv, ", ".join(carriers)
            )


# =============================================================================
# FRONT HALF: SCAN -> EXTRACT -> RUN -> COLLECT -> DESCRIBE
# =============================================================================


def load_registry(path: Path | None) -> tuple["OrderedDict[str, dict]", dict[str, dict]]:
    """Parse the optional jobs.private.json registry.

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

    ``scripts_dir`` is searched RECURSIVELY (unlike the public half), because the
    private feature scripts live in different topic folders (``ridership_tools``,
    ``operations_tools``). When the registry lists scripts it is an allowlist:
    only those are run, in registry order, each with its registry ``cmd``.

    Returns:
        A list of ``(name, path, cmd)`` where ``cmd`` is the registry command
        (list/str) or None to use ``DEFAULT_CMD_TEMPLATE``.
    """
    found: dict[str, Path] = {}
    search_dirs = [scripts_dir] + ([inbox_dir] if inbox_dir is not None else [])
    for directory in search_dirs:
        if not directory.exists():
            continue
        for py in sorted(directory.rglob("*.py")):
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

    Returns:
        A FeatureTable, or None when the table cannot be joined / has nothing to
        contribute (skipped with a warning).
    """
    spec = script_outputs.get(file.name) or global_outputs.get(file.name)
    # Convention: a registry spec with empty keep_cols deliberately drops an
    # intermediate from bundling (e.g. otp_monthly_tides' panel, which
    # otp_by_route consumes but which must not become a bundle of its own).
    if spec is not None and not spec["keep_cols"]:
        logging.info(
            "Output '%s' is registry-marked ignore (empty keep_cols) — skipping.", file.name
        )
        return None
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

    # A value column shipped by two differently keyed bundles will collide at
    # model-join time if both bundles join the same anchor. The model aborts on
    # that; warn here too, where the producing tables can still be named.
    col_owners: dict[str, list[str]] = {}
    for res in results:
        for col in res.columns:
            if col not in res.join_keys:
                col_owners.setdefault(col, []).append(res.filename)
    cross_bundle = {col: owners for col, owners in col_owners.items() if len(owners) > 1}
    if cross_bundle:
        logging.warning(
            "Column name(s) shipped by multiple bundles (the model will refuse to "
            "join the second if both bundles match the same anchor): %s",
            cross_bundle,
        )
    return results


def write_manifest(output_dir: Path, bundles: list[BundleResult]) -> Path:
    """Write manifest.json describing every bundle for the models to verify."""
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
# RUN-LOG HELPERS (self-contained copy; mirrors prep_features_public.py)
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
    log_path = output_dir / "prep_features_private_runlog.txt"

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
        "FEATURE PREP RUN LOG (PRIVATE — secured box, orchestrator)",
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
    expected_dv: str = EXPECTED_DV_COLUMN,
    exclude_names: Sequence[str] = EXCLUDE_SCRIPT_NAMES,
    exclude_globs: Sequence[str] = EXCLUDE_SCRIPT_GLOBS,
    require_run_log: bool = REQUIRE_RUN_LOG,
    source_path: Path = SELF_PATH,
) -> list[BundleResult]:
    """Run the full scan -> extract -> run -> collect -> describe -> combine pipeline.

    Returns:
        The written bundles.

    Raises:
        ValueError: On a hygiene failure (bad keep_cols) — raised BEFORE any
            bundle is written.
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
                    "CLI. Fix this script's cmd in the jobs.private.json registry."
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
    # Hygiene checks run BEFORE any bundle is written (no denylist on this side).
    validate_config(tables, expected_dv=expected_dv)
    logging.info("Hygiene checks passed: %d feature table(s) cleared.", len(tables))

    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = write_bundles(tables, output_dir)
    if not bundles:
        raise RuntimeError(
            "No usable bundles were produced. Every feature script failed or emitted no "
            f"joinable table. See the per-script logs in {work_dir / 'logs'} for the exact "
            "errors. If scripts exited with code 2, their command flags do not match — fix "
            "their cmd entries in the jobs.private.json registry."
        )

    write_manifest(output_dir, bundles)
    if not write_run_log(output_dir, bundles, runs, source_path) and require_run_log:
        raise RuntimeError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )

    logging.info(
        "All processing complete. %d bundle(s) ready for the models.",
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
            "Secured-box feature-prep orchestrator (PRIVATE). Runs the NTD / OTP / runtime "
            "feature scripts, collects their tables, and writes bundles + manifest for the "
            "ridership models. Defaults come from the CONFIGURATION block at the top of this file."
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
        help="Optional jobs.private.json registry path.",
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
    except ValueError as exc:  # hygiene failure
        logging.error("Hygiene check failed: %s", exc)
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
