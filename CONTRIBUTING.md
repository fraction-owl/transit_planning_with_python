# Contributing to This Repository

This project is built for transit planners, transit analysts, and civic technologists who want readable,
self-contained Python scripts for transportation planning. We value clarity, consistency, and user-friendliness
in our scripts to make them usable by a wider audience. Please follow these principles when contributing:

## 👥 How to Contribute

Participation is welcome from anyone, whether you’re new to Python, an experienced GitHub user, or a seasoned developer:
- **Beginners:**  
  - Feel free to copy, modify, and use scripts without any expectation of interaction.
- **Intermediate Users:**  
  - Create a GitHub Issue to report bugs, request new features, or ask questions.
  - Clearly describe your issue, including error messages, expected vs. actual results, and steps to reproduce the issue.
- **Advanced Users:**  
  - Submit Pull Requests (PRs) with proposed improvements or fixes.
  - Follow the instructions below to ensure your PRs meet project standards.

## 🧱 Code Structure

- Scripts **must be modular**, with a clearly defined `main()` function.
- Include a clear **configuration section at the top** of each script.
  - Prefer inline variable configuration over `argparse`.
- Use intuitive success messages (`logging`) at the end of script execution.
- Scripts that write an output file **must** also write a `_runlog.txt` sidecar next to that file. The sidecar captures the CONFIGURATION block verbatim (bounded by `# === BEGIN CONFIG ===` / `# === END CONFIG ===` markers) plus a timestamp and source-script path. This creates a drift-proof record that lets anyone reconstruct exactly what settings produced a given output. See `scripts/ridership_tools/data_request_by_stop_processor.py` for the reference implementation (`extract_config_block` / `write_run_log`).
- Do **not import** functions from the shared `utils/` directory at runtime.
  - Instead, **copy the relevant helper functions** into your script.
  - This keeps each script self-contained and easier for beginners to understand, run, and modify.
- The `utils/` directory holds the **canonical version** of shared functions. Any differences between a script’s local copy and the canonical version will be flagged in CI.

## ⚙️ Runtime Behavior

- Prefer the `logging` module over `print()` for all success messages, diagnostics, or warnings.
- Implement **graceful, actionable error handling** — no cryptic tracebacks.
- Use placeholder filenames that are clean, minimal, and safe to run (e.g., r"Path\\To\\Output_Folder", "input_data.csv").
- **Output paths are a directory constant plus separate filename constant(s).** Define `OUTPUT_DIR` (a folder) in the CONFIGURATION block, then join it with a separate `*_FILENAME`/`*_NAME` constant (e.g., `OUTPUT_DIR / OUTPUT_FILENAME`) to build each output path. This confines the only hand-edited, backslash-prone value to the directory, and lets users redirect every output by changing one line. If a script writes a dynamically-named or per-loop set of outputs (one file per route, service_id, etc.), define `OUTPUT_DIR` only — see `stop_pattern_exporter.py`. For a fixed set of named outputs, see `route_transfer_calculator.py` (`OUTPUT_DIR` + `SUMMARY_FILENAME`/`DETAIL_FILENAME`).
- **Run logs are required.** Any script that produces an output file must write a matching `_runlog.txt` sidecar (same directory, same stem). A `REQUIRE_RUN_LOG: bool = True` config variable controls enforcement: when `True` (the default), a failed log write aborts the script so analysts are never left with an untraced output. Set it to `False` only when writing to a genuinely read-only location.
- Default to:
  - **Washington, DC CRS** unless otherwise noted (chosen because DC is the U.S. capital).
  - **Imperial units** (feet/miles), with metric option available and clearly noted.

## 🧪 Testing & Review

- All commits and **Pull Request titles** must use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for clear project history.

**PR title format:** `<type>(<optional-scope>): <description>`
- Allowed types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`
- Scope (optional): lowercase letters, digits, and hyphens only
- Description: starts with a lowercase letter

Example: `feat(gtfs): add stop spacing validator`
- All pull requests are automatically tested for:
  - Style and formatting using `ruff`.
  - Static typing using [`ty`](https://github.com/astral-sh/ty).
  - Unit tests (where present) using `pytest`.
- You do not need to run linters or type checkers manually, but you **must fix** any issues flagged by the CI pipeline before requesting a review.

**Manual testing policy (scripts):**  
New or modified scripts under `scripts/` **must be manually tested** before opening a PR. See the checklist below.

**Manual test checklist (scripts):**
- [ ] Script runs end-to-end with appropriate logging output.
- [ ] Configuration section at the top is clear and minimal; defaults produce a safe no-op or sample run.
- [ ] Input/output paths are valid; exported files are created with expected names and sizes.
- [ ] A `_runlog.txt` sidecar is created alongside the output file and contains the correct configuration block, timestamp, and source path.
- [ ] Error messages are actionable (no cryptic tracebacks for expected user mistakes).
- [ ] Runtime is reasonable on a small sample dataset; no hidden network or large temporary files.

---

### 🧩 Unit Tests for Helper Functions

If you add or significantly modify a function in the `utils/` directory:

- Write a **unit test** that exercises its normal behavior and at least one error condition.
- Save new tests under `tests/` following the naming pattern `test_<module>.py`.
- Use small, synthetic input data—do **not** rely on external files or network access.
- Tests should run quickly (<1 s each) and be deterministic.
- These tests protect against silent failures caused by future changes to dependencies (e.g., pandas, geopandas).
- A pull request adding or modifying utils **without** a corresponding test may be asked to add one before review.

## 🧼 Code Style

This project uses `ruff` to enforce formatting, linting, and docstring style, and `ty` for static type checking.

Many common issues are auto-corrected by Ruff on PRs.

- The following are enforced in CI:
  - PEP 8 layout, formatting, and common bug detection (via `pycodestyle`, `pyflakes`, and `flake8-bugbear`)
  - The enforced line length is **100 characters**
  - Google-style docstrings (Note: `tests/` are exempt from docstring requirements)
  - Consistent import ordering (`isort`-compatible)
  - Type annotations and type-checking imports, with some leniency for `Any` (`ANN401` is ignored)
  - Pandas best practices (`pandas-vet`)
  - **Python version floors** (via `dev_tools/audit_python_floor.py`): scripts that can run inside
    ArcGIS Pro's bundled Python — the `*_arcpy` variants plus any script that doesn't import the
    open-source geospatial stack (geopandas, shapely, rapidfuzz, …) — must stay **Python 3.9**
    compatible: no 3.10+ syntax (e.g. `match`), and no `X | Y` union annotations in function
    signatures or module/class-level variables unless the file has
    `from __future__ import annotations`. Scripts importing the open-source stack follow the
    repo-wide **3.12** baseline. The tier is determined by a script's imports, not its filename.
        
**Note:** Ruff auto-fixes for unused imports (`F401`), docstring styling (`D`), and import sorting (`I`) are pushed
back to your PR branch automatically by the GitHub Actions workflow. While `ty` is configured to simply warn for
many type mismatches, it *will* block/error on unresolved imports, unresolved references, and redundant casts.

**Running ruff locally (optional but recommended):** install dev dependencies and enable the pre-commit hook to
catch ruff issues before pushing, instead of waiting for a CI round-trip:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

After this, `ruff check` and `ruff format` run automatically on every `git commit`. The hook reads its config
from `pyproject.toml` and uses the same ruff version that CI does (pinned in `requirements-dev.txt`).

## 📁 File Organization

- Add new scripts to the appropriate subfolder within `scripts/`, based on function (e.g., `ridership_tools/`, `gtfs_exports/`).
- If you create a helper that’s reused across multiple scripts:
  - Add the canonical version to the appropriate file under `utils/`.
  - Then copy that helper into any script that uses it.
- Do **not** import functions from one script into another or from `utils/` at runtime.

### Requirements files

The repository uses three requirements files:

| File | Purpose |
|---|---|
| `requirements.txt` | Open-source / non-ArcGIS runtime stack (geopandas, rapidfuzz, etc.). **This is the pip install target for running the scripts.** |
| `requirements-dev.txt` | Dev and CI tooling (pytest, ruff, ty, pre-commit, typing stubs). Install alongside `requirements.txt` when contributing: `pip install -r requirements.txt -r requirements-dev.txt`. CI installs both. |
| `requirements-arcpro.txt` | Reference only — documents packages already present in the ArcGIS Pro Python environment. Nothing here needs to be pip-installed. |

If you add a new runtime dependency, add it to `requirements.txt`; if it's a test/lint/type-checking tool, add it to `requirements-dev.txt`. If you're noting that something is available in the ArcGIS Pro environment, add it to `requirements-arcpro.txt` as a comment or entry with a note.

## 🌳 GitHub Contribution Workflow

Follow these instructions when contributing code via GitHub:

1. **Fork the repository.**
   - Click the "Fork" button on GitHub to create your own copy.
2. **Clone the repository locally.**
   ```bash
   git clone https://github.com/<YOUR-USERNAME>/<REPO-NAME>.git
   cd <REPO-NAME>
   ```
3. **Create a feature branch.**
   ```bash
   git checkout -b feature/your-descriptive-feature-name
   ```
4. **Make your changes.**
   - Keep commits small and clearly described with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. **Push your branch to your fork.**
   ```bash
   git push -u origin feature/your-descriptive-feature-name
   ```
6. **Open a Pull Request.**
   - Title your Pull Request using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
   - Clearly describe your changes, referencing any related issues.
7. **Respond to feedback.**
   - Update your PR with suggested changes until your contribution is approved.

---

For further details, contact the repository maintainer.
