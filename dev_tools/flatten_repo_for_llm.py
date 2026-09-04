#!/usr/bin/env python3
"""Copy the repository into a flat, upload-ready tree for an LLM project.

Claude Projects and ChatGPT Projects store uploaded files as a flat knowledge
base — the folder tree is not preserved, and most upload widgets cannot select
files spread across eleven nested folders in one gesture. This tool writes a
throwaway copy of the repo in which every top-level folder keeps its name but
its subfolders are dissolved, so each folder can be opened once and select-all
dragged into the uploader.

Two output shapes are available:

* ``flat`` — one folder per top-level directory (``scripts/``, ``utils/``,
  ``dev_tools/``, ...) holding every file found beneath it. Root-level files
  land in the output root.
* ``bundle`` — one Markdown file per top-level directory, concatenating its
  files under ``## original/path`` headers inside fenced code blocks. Use this
  when the target project caps the number of uploadable files: it turns ~290
  files into a handful.

Flattening throws away the folder taxonomy, which is itself context (a reader
learns something from ``scripts/gtfs_data_quality/``). Both modes preserve it:
names collide only when two files share a basename, and the MANIFEST maps every
output file back to its original path.

Nothing in the source repository is modified, moved, or deleted; the tool only
reads the repo and writes into OUTPUT_DIR.

Inputs
------
- REPO_ROOT: The repository to ingest. Files are listed with
  ``git ls-files`` (tracked files only, so ignored build junk never appears)
  and fall back to a filtered directory walk when git is unavailable.

Outputs
-------
- OUTPUT_DIR: The flattened copy (``flat``) or the per-folder Markdown
  bundles (``bundle``).
- MANIFEST.txt: The verbatim CONFIGURATION block, a timestamp, the source
  repo path, and one ``output name <- original/path`` line per file.
- .flatten_repo_for_llm: A marker file identifying the directory as tool
  output, so a later run with CLEAN_OUTPUT can safely replace it.

Typical usage
-------------
Update the paths in the CONFIGURATION section (or pass the matching CLI
flags) and run from a shell or a Jupyter notebook::

    python dev_tools/flatten_repo_for_llm.py
    python dev_tools/flatten_repo_for_llm.py --mode bundle --clean
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
# === BEGIN CONFIG ===

# Repository to ingest (defaults to the repo containing this script).
REPO_ROOT: str = str(Path(__file__).resolve().parent.parent)

# Where the upload-ready copy is written. Keep this OUTSIDE the repository so
# the copy is never committed and never ingested by the next run.
OUTPUT_DIR: str = r"../transit_planning_flat"

# "flat"   - one folder per top-level directory, subfolders dissolved.
# "bundle" - one Markdown file per top-level directory (fewer files to upload).
MODE: str = "flat"

# How flattened files are named:
# "auto"     - keep the bare filename; prefix with the subfolder path only when
#              two files in the same destination folder would collide.
# "prefixed" - always prefix with the subfolder path (gtfs_exports__foo.py).
# "basename" - never prefix; a colliding name gets a numeric suffix instead.
NAME_STYLE: str = "auto"

# Top-level folders to skip entirely (matched against the first path segment).
EXCLUDE_TOP_LEVEL: Tuple[str, ...] = (".github",)

# Only these extensions are copied. Empty tuple means "every tracked file",
# which pulls in the .zip/.gz test fixtures — rarely what an LLM project wants.
INCLUDE_EXTENSIONS: Tuple[str, ...] = (
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".csv",
    ".json",
    ".yml",
    ".yaml",
    ".cfg",
    ".ini",
)

# Files larger than this are skipped and reported (0 disables the limit).
MAX_FILE_BYTES: int = 1_000_000

# Prefer `git ls-files` over a directory walk when the repo is a git checkout.
USE_GIT: bool = True

# Replace a previous run's output. Only a directory carrying this tool's
# marker file is ever removed; anything else aborts the run.
CLEAN_OUTPUT: bool = False

LOG_LEVEL: str = "INFO"

# === END CONFIG ===

MARKER_FILENAME: str = ".flatten_repo_for_llm"
MANIFEST_FILENAME: str = "MANIFEST.txt"

WALK_SKIP_DIRS: Tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "node_modules",
)

FENCE_LANGUAGES: Dict[str, str] = {
    ".py": "python",
    ".toml": "toml",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".csv": "csv",
    ".md": "markdown",
}

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================


def notebook_safe_argv(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Return the argv to parse, shielding notebook kernels from stray flags.

    Canonical implementation: ``utils/cli_helpers.py``.

    Args:
        argv: Explicit argument list passed to ``main()``, or ``None`` to fall
            back to ``sys.argv``.

    Returns:
        ``list(argv)`` when *argv* was provided; ``[]`` when running inside a
        notebook kernel; otherwise ``None`` so argparse reads ``sys.argv[1:]``.
    """
    if argv is not None:
        return list(argv)
    if "ipykernel" in sys.modules:
        return []
    return None


def list_git_files(repo_root: Path) -> Optional[List[Path]]:
    """List repository-relative paths of git-tracked files.

    Args:
        repo_root: Repository checkout to query.

    Returns:
        Sorted relative paths, or ``None`` when the directory is not a git
        checkout or the ``git`` executable is unavailable.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    return sorted(Path(name) for name in names)


def walk_files(repo_root: Path) -> List[Path]:
    """List repository-relative paths by walking the directory tree.

    Args:
        repo_root: Directory to walk.

    Returns:
        Sorted relative paths, excluding well-known cache and virtualenv dirs.
    """
    found: List[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root)
        if any(part in WALK_SKIP_DIRS for part in relative.parts):
            continue
        found.append(relative)
    return sorted(found)


def select_files(
    repo_root: Path,
    candidates: Sequence[Path],
    exclude_top_level: Sequence[str],
    include_extensions: Sequence[str],
    max_file_bytes: int,
    output_dir: Path,
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Filter candidate paths down to the files worth copying.

    Args:
        repo_root: Repository the candidates are relative to.
        candidates: Repository-relative paths to consider.
        exclude_top_level: First path segments to drop entirely.
        include_extensions: Extensions to keep; empty keeps every extension.
        max_file_bytes: Size ceiling in bytes; 0 disables the check.
        output_dir: Resolved output directory, skipped if nested in the repo.

    Returns:
        A ``(kept, skipped)`` pair, where *skipped* holds ``(path, reason)``.
    """
    excluded = set(exclude_top_level)
    extensions = {ext.lower() for ext in include_extensions}
    kept: List[Path] = []
    skipped: List[Tuple[Path, str]] = []

    for relative in candidates:
        absolute = repo_root / relative
        if relative.parts[0] in excluded:
            skipped.append((relative, "excluded top-level folder"))
            continue
        if extensions and absolute.suffix.lower() not in extensions:
            skipped.append((relative, "extension not in INCLUDE_EXTENSIONS"))
            continue
        if not absolute.is_file():
            skipped.append((relative, "missing on disk"))
            continue
        if output_dir == absolute or output_dir in absolute.parents:
            skipped.append((relative, "inside OUTPUT_DIR"))
            continue
        if max_file_bytes and absolute.stat().st_size > max_file_bytes:
            skipped.append((relative, f"larger than {max_file_bytes} bytes"))
            continue
        kept.append(relative)

    return kept, skipped


def group_by_top_level(files: Sequence[Path]) -> Dict[str, List[Path]]:
    """Group repository-relative paths by their first path segment.

    Args:
        files: Repository-relative paths.

    Returns:
        A mapping of top-level folder name to its files. Files sitting at the
        repository root are grouped under ``""``.
    """
    groups: Dict[str, List[Path]] = {}
    for relative in files:
        top = relative.parts[0] if len(relative.parts) > 1 else ""
        groups.setdefault(top, []).append(relative)
    return dict(sorted(groups.items()))


def prefixed_name(relative: Path) -> str:
    """Build a collision-resistant filename from a path's subfolders.

    Args:
        relative: Repository-relative path, e.g. ``scripts/gtfs_exports/a.py``.

    Returns:
        The subfolder segments below the top level joined to the filename with
        double underscores, e.g. ``gtfs_exports__a.py``.
    """
    middle = relative.parts[1:-1]
    return "__".join([*middle, relative.name])


def plan_names(files: Sequence[Path], name_style: str) -> Dict[Path, str]:
    """Assign an output filename to each file in one destination folder.

    Args:
        files: Repository-relative paths sharing a top-level folder.
        name_style: ``"auto"``, ``"prefixed"``, or ``"basename"``.

    Returns:
        A mapping of source path to its unique output filename.
    """
    counts = Counter(relative.name for relative in files)
    names: Dict[Path, str] = {}
    taken: set[str] = set()
    for relative in files:
        if name_style == "prefixed" or (name_style == "auto" and counts[relative.name] > 1):
            candidate = prefixed_name(relative)
        else:
            candidate = relative.name
        # "basename" style, and the rare prefixed collision, fall back to a suffix.
        stem, suffix = Path(candidate).stem, Path(candidate).suffix
        attempt = 2
        while candidate in taken:
            candidate = f"{stem}_{attempt}{suffix}"
            attempt += 1
        taken.add(candidate)
        names[relative] = candidate
    return names


def check_output_target(output_dir: Path, repo_root: Path) -> None:
    """Reject an output target that would overwrite the repository itself.

    Args:
        output_dir: Proposed destination root.
        repo_root: Repository being ingested.

    Raises:
        ValueError: The target is the repo, one of its parents, or a file.
    """
    if output_dir == repo_root or output_dir in repo_root.parents:
        raise ValueError(
            f"OUTPUT_DIR ({output_dir}) is the repository or one of its parents; "
            "choose an empty directory outside the repo."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"OUTPUT_DIR ({output_dir}) exists and is not a directory.")


def prepare_output_dir(output_dir: Path, repo_root: Path, clean_output: bool) -> None:
    """Create the output directory, refusing to touch unrelated content.

    Args:
        output_dir: Directory to (re)create.
        repo_root: Repository being ingested, which must never be the target.
        clean_output: Whether a previous run's output may be replaced.

    Raises:
        ValueError: The target is an existing non-empty directory this tool
            did not create.
    """
    check_output_target(output_dir, repo_root)

    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not clean_output:
                raise ValueError(
                    f"OUTPUT_DIR ({output_dir}) is not empty. Delete it, point OUTPUT_DIR "
                    "elsewhere, or re-run with --clean / CLEAN_OUTPUT = True."
                )
            if not (output_dir / MARKER_FILENAME).exists():
                raise ValueError(
                    f"OUTPUT_DIR ({output_dir}) holds files this tool did not write "
                    f"(no {MARKER_FILENAME} marker); refusing to delete it."
                )
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MARKER_FILENAME).write_text(
        "Written by dev_tools/flatten_repo_for_llm.py. Safe to delete.\n", encoding="utf-8"
    )


def write_flat(
    repo_root: Path, output_dir: Path, groups: Dict[str, List[Path]], name_style: str
) -> List[Tuple[str, Path]]:
    """Copy every file into a single folder per top-level directory.

    Args:
        repo_root: Repository being ingested.
        output_dir: Destination root.
        groups: Files grouped by top-level folder.
        name_style: Naming strategy passed to :func:`plan_names`.

    Returns:
        ``(output path relative to output_dir, source path)`` pairs.
    """
    written: List[Tuple[str, Path]] = []
    for top, files in groups.items():
        destination = output_dir / top if top else output_dir
        destination.mkdir(parents=True, exist_ok=True)
        for relative, name in plan_names(files, name_style).items():
            shutil.copy2(repo_root / relative, destination / name)
            written.append((str(Path(top) / name) if top else name, relative))
    return written


def read_text(path: Path) -> Optional[str]:
    """Read a file as UTF-8 text.

    Args:
        path: File to read.

    Returns:
        The decoded text, or ``None`` when the file is not valid UTF-8.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def write_bundles(
    repo_root: Path, output_dir: Path, groups: Dict[str, List[Path]]
) -> List[Tuple[str, Path]]:
    """Concatenate each top-level folder into one Markdown file.

    Args:
        repo_root: Repository being ingested.
        output_dir: Destination root.
        groups: Files grouped by top-level folder.

    Returns:
        ``(bundle filename, source path)`` pairs, one per bundled file.
    """
    written: List[Tuple[str, Path]] = []
    for top, files in groups.items():
        bundle_name = f"{top or 'root'}.md"
        lines = [f"# {top or repo_root.name} (flattened bundle)", ""]
        for relative in files:
            text = read_text(repo_root / relative)
            if text is None:
                logging.warning("Skipping non-UTF-8 file in bundle: %s", relative)
                continue
            language = FENCE_LANGUAGES.get(relative.suffix.lower(), "")
            lines.extend(
                [f"## {relative.as_posix()}", "", f"```{language}", text.rstrip(), "```", ""]
            )
            written.append((bundle_name, relative))
        (output_dir / bundle_name).write_text("\n".join(lines), encoding="utf-8")
    return written


def extract_config_block(script_path: Path) -> str:
    """Return this script's CONFIGURATION block verbatim.

    Args:
        script_path: Path to the running script.

    Returns:
        The text between the BEGIN/END CONFIG markers, or an explanatory note
        when the markers cannot be found.
    """
    source = read_text(script_path)
    if source is None:
        return "(configuration block unavailable: source file unreadable)"
    match = re.search(r"# === BEGIN CONFIG ===\n(.*?)# === END CONFIG ===", source, re.DOTALL)
    if match is None:
        return "(configuration block unavailable: markers not found)"
    return match.group(1).strip("\n")


def write_manifest(
    output_dir: Path,
    repo_root: Path,
    written: Sequence[Tuple[str, Path]],
    skipped: Sequence[Tuple[Path, str]],
    mode: str,
) -> Path:
    """Write the mapping from output files back to their original paths.

    Args:
        output_dir: Destination root.
        repo_root: Repository that was ingested.
        written: ``(output name, source path)`` pairs.
        skipped: ``(source path, reason)`` pairs.
        mode: The output shape that produced *written*.

    Returns:
        The manifest path.
    """
    lines = [
        f"# Flattened copy of {repo_root}",
        f"# Generated {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"# Source script: {Path(__file__).resolve()}",
        f"# Mode: {mode}",
        "",
        "# === BEGIN CONFIG ===",
        extract_config_block(Path(__file__).resolve()),
        "# === END CONFIG ===",
        "",
        f"# {len(written)} files. Each line maps an output file to its path in the repository.",
        "",
    ]
    lines.extend(f"{name} <- {relative.as_posix()}" for name, relative in written)
    if skipped:
        lines.extend(["", f"# {len(skipped)} files skipped:", ""])
        lines.extend(f"# {relative.as_posix()} — {reason}" for relative, reason in skipped)

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def flatten_repo(
    repo_root: Path,
    output_dir: Path,
    mode: str,
    name_style: str,
    exclude_top_level: Sequence[str],
    include_extensions: Sequence[str],
    max_file_bytes: int,
    use_git: bool,
    clean_output: bool,
) -> Path:
    """Build the upload-ready copy of a repository.

    Args:
        repo_root: Repository to ingest.
        output_dir: Destination root.
        mode: ``"flat"`` or ``"bundle"``.
        name_style: ``"auto"``, ``"prefixed"``, or ``"basename"``.
        exclude_top_level: Top-level folders to drop.
        include_extensions: Extensions to keep; empty keeps everything.
        max_file_bytes: Size ceiling in bytes; 0 disables the check.
        use_git: Whether to list files with ``git ls-files``.
        clean_output: Whether a previous run's output may be replaced.

    Returns:
        The manifest path.

    Raises:
        ValueError: The configuration is unusable or no files were selected.
    """
    if mode not in {"flat", "bundle"}:
        raise ValueError(f"MODE must be 'flat' or 'bundle', got {mode!r}.")
    if name_style not in {"auto", "prefixed", "basename"}:
        raise ValueError(
            f"NAME_STYLE must be 'auto', 'prefixed', or 'basename', got {name_style!r}."
        )
    if not repo_root.is_dir():
        raise ValueError(f"REPO_ROOT does not exist or is not a directory: {repo_root}")
    check_output_target(output_dir, repo_root)

    candidates = list_git_files(repo_root) if use_git else None
    if candidates is None:
        logging.info("Falling back to a directory walk (no git listing available).")
        candidates = walk_files(repo_root)

    kept, skipped = select_files(
        repo_root, candidates, exclude_top_level, include_extensions, max_file_bytes, output_dir
    )
    if not kept:
        raise ValueError("No files selected. Check EXCLUDE_TOP_LEVEL and INCLUDE_EXTENSIONS.")

    groups = group_by_top_level(kept)
    prepare_output_dir(output_dir, repo_root, clean_output)
    if mode == "flat":
        written = write_flat(repo_root, output_dir, groups, name_style)
    else:
        written = write_bundles(repo_root, output_dir, groups)

    for top, files in groups.items():
        size_kb = sum((repo_root / f).stat().st_size for f in files) / 1024
        logging.info("%-12s %3d files  %7.1f KB", top or "(root)", len(files), size_kb)

    manifest_path = write_manifest(output_dir, repo_root, written, skipped, mode)
    renamed = sum(1 for name, relative in written if Path(name).name != relative.name)
    if mode == "flat" and renamed:
        logging.info("%d file(s) renamed to avoid collisions — see the manifest.", renamed)
    logging.info(
        "Wrote %d source files as %d upload file(s) to %s",
        len(written),
        len({name for name, _ in written}),
        output_dir,
    )
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, defaulting every flag to its CONFIG constant."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy a repository into a flat, upload-ready tree for a Claude or ChatGPT "
            "project. Defaults come from the CONFIGURATION block in this file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=REPO_ROOT, help="Repository to ingest.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Where the copy is written.")
    parser.add_argument("--mode", choices=("flat", "bundle"), default=MODE, help="Output shape.")
    parser.add_argument(
        "--name-style",
        choices=("auto", "prefixed", "basename"),
        default=NAME_STYLE,
        help="How flattened filenames are built.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(EXCLUDE_TOP_LEVEL),
        metavar="FOLDER",
        help="Top-level folders to skip.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(INCLUDE_EXTENSIONS),
        metavar="EXT",
        help="Extensions to include; pass none to include every file.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=MAX_FILE_BYTES,
        help="Per-file size ceiling (0 = no limit).",
    )
    parser.add_argument(
        "--no-git",
        dest="use_git",
        action="store_false",
        default=USE_GIT,
        help="Walk the directory tree instead of using `git ls-files`.",
    )
    parser.add_argument(
        "--clean",
        dest="clean_output",
        action="store_true",
        default=CLEAN_OUTPUT,
        help="Replace a previous run's output directory.",
    )
    parser.add_argument("--log-level", default=LOG_LEVEL, help="Logging level.")
    return parser


# ==================================================================================================
# MAIN
# ==================================================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        manifest_path = flatten_repo(
            repo_root=repo_root,
            output_dir=output_dir,
            mode=args.mode,
            name_style=args.name_style,
            exclude_top_level=args.exclude,
            include_extensions=args.extensions,
            max_file_bytes=args.max_bytes,
            use_git=args.use_git,
            clean_output=args.clean_output,
        )
    except ValueError as exc:
        logging.error("%s", exc)
        return 2
    except OSError as exc:
        logging.error("File system error: %s", exc)
        return 1

    logging.info("Manifest: %s", manifest_path)
    logging.info("Done. Upload the folders in %s to your project.", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
