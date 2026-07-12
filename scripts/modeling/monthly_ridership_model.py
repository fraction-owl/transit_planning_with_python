"""Fit an OLS ridership model on the secured box (PART B of a two-stage split).

This is the secured-box half of a split pipeline that keeps proprietary NTD
data on one machine while feature prep happens elsewhere:

    PART A (``prep_features_public.py``, runs anywhere, no NTD)
        Loads, validates, dedups, and bundles the *non-NTD* feature tables,
        grouping them by join-key signature and writing one CSV bundle per
        signature plus a manifest (row counts + per-bundle SHA-256).

    PART B (this script, runs only where the NTD anchor lives)
        Loads the NTD anchor, verifies and joins each prepped bundle onto it,
        then fits the regression and exports results.

Because the NTD anchor is this script's *input* (it holds the dependent
variable plus the service-supplied predictors), the script must execute on the
machine that has NTD access. Nothing NTD-derived crosses the boundary: only the
bundles produced by Part A are transferred in.

Period is optional. Each bundle declares its own join keys; a bundle is joined
only if every one of its keys is present in the anchor. A cross-sectional anchor
(``route_id`` only) therefore auto-skips a ``period`` bundle, while a panel
anchor (``route_id`` + ``period``) joins both. The analysis mode is determined
by the anchor's grain, not by a flag:
    - Cross-sectional rollup -> residuals rank routes over/under fundamentals.
    - Route x period panel -> month/period effects surface "weird" months.

Everything is implemented with the ArcGIS Pro Python stack (``numpy`` /
``scipy`` / ``pandas`` / ``matplotlib``); no ``scikit-learn`` / ``statsmodels``
and no ``pyarrow`` (bundles are CSV so a stock ``arcgispro-py3`` env suffices).

Inputs:
    - One anchor table (CSV or XLSX) holding the dependent variable (NTD).
    - A bundle directory + manifest produced by ``prep_features_public.py``.

Outputs:
    - An Excel workbook with model summary, coefficient, correlation, and
      per-observation (fitted / residual) sheets.
    - Optional diagnostic plots (residuals-vs-fitted, Q-Q, residual histogram,
      predicted-vs-actual).
    - A run log capturing the verbatim configuration block plus the verified
      provenance (filename + SHA-256) of every bundle that fed the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Sentinel markers used by extract_config_block / write_run_log to identify
# the configuration block within this file's source. Each string must appear
# exactly once in this file as a stand-alone comment line (other than these
# constant definitions themselves). Edit with care.
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# Path to this file, used to extract the config block for the run log. ``__file__``
# is undefined when the code is pasted into a notebook cell, so a configured
# fallback keeps the run log working there too.
SELF_PATH: Final[Path] = (
    Path(__file__) if "__file__" in globals() else Path("monthly_ridership_model.py")
)


# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===


# -----------------------------------------------------------------------------
#  Input / output paths
# -----------------------------------------------------------------------------

# Anchor table: one row per observation, holding the dependent variable plus the
# "service supplied" predictors (scheduled/revenue hours and revenue miles).
# This is the proprietary NTD payload and is read only on the secured box.
#   - Cross-sectional analysis -> a route-level roll-up (one row per route_id).
#   - Panel analysis           -> a route x period table (route_id + period).
ANCHOR_PATH: Final[Path] = Path(r"Path\To\Your\ntd_route_panel.csv")
ANCHOR_SHEET: Final[str | int] = 0  # worksheet name/index if ANCHOR_PATH is XLSX

OUTPUT_DIR: Final[Path] = Path(r"Path\To\Your\Output\Folder")

# Feature bundles produced by prep_features_public.py (PART A) and transferred in.
# BUNDLE_DIR holds the bundle CSVs; MANIFEST_PATH is the JSON sidecar that lists
# each bundle's join keys, row/column counts, and SHA-256. Every bundle is
# joined onto the anchor only if all of its join keys are present in the anchor,
# so a "period" bundle is silently skipped for a cross-sectional anchor.
BUNDLE_DIR: Final[Path] = Path(r"Path\To\Your\prepped_features")
MANIFEST_PATH: Final[Path] = Path(r"Path\To\Your\prepped_features\manifest.json")

# When True, every bundle's on-disk SHA-256 must match the manifest before it is
# joined; a mismatch aborts the run. This ties the secured-box output to the
# exact prep that produced the bundles and catches truncated/edited transfers.
VERIFY_BUNDLE_HASHES: Final[bool] = True

# Minimum share of anchor rows each joined bundle must match (0..1). A join
# below this floor almost always means a key-normalization or grain problem
# upstream, so the run aborts rather than silently modeling a subset of the
# system. Lower it (or set 0.0) only for a bundle that is legitimately sparse.
MIN_BUNDLE_MATCH_RATE: Final[float] = 0.9

# Join key(s) used for predictor *exclusion* (keys are never modeled) and to
# derive a month dummy. The actual joins are driven by each bundle's own keys
# from the manifest, not by this constant. Use ("route_id",) for a purely
# cross-sectional model, or ("route_id", "period") for a route x month panel.
JOIN_KEYS: Final[tuple[str, ...]] = ("route_id",)

# -----------------------------------------------------------------------------
#  Model specification
# -----------------------------------------------------------------------------

# Dependent variable (must exist in the anchor table).
DEPENDENT_VAR: Final[str] = "ntd_boardings"

# Predictors to include. Leave empty to auto-select every numeric column that is
# neither a join key nor the dependent variable.
PREDICTORS: Final[tuple[str, ...]] = (
    "scheduled_hours",
    "revenue_miles",
    "avg_headway_min",  # mean gap between trips (from headway_span_exporter.py)
    "span_hrs",  # service span first to last departure
    "gas_price",
    "unemployment_rate",
    # Daily-derived monthly weather aggregates.
    "avg_temp_f",
    "max_daily_precip_in",
    "days_with_precip",
    "total_snow_in",
    "max_daily_snow_in",
    "pop_served",
    "low_income_served",
    "empl_served",  # Census LODES employment within service catchment
    # Generic POI coverage is presence-only: just how many sites a route
    # reaches, with no sense of their size.
    "sites_served",
    # Schools and CABI stations add a magnitude alongside the count: enrollment
    # reached and weekday ridership reached, respectively.
    "schools_served",
    "enrollment_served",
    "cabi_stations_served",
    "cabi_weekday_riders_served",
    # Saturday/Sunday bikeshare averages ship alongside the weekday figure.
    # The three day-type columns are strongly correlated across stations, so
    # analysts trimming a collinear fit will usually keep weekday only.
    "cabi_saturday_riders_served",
    "cabi_sunday_riders_served",
)

# Categorical columns to one-hot encode (first level dropped as the reference).
# "month" is derived automatically from the "period" column after table assembly.
CATEGORICAL_PREDICTORS: Final[tuple[str, ...]] = ("month",)

# -----------------------------------------------------------------------------
#  Transforms & estimator options
# -----------------------------------------------------------------------------

# Log-transform the dependent variable. Combined with logged predictors this
# yields constant-elasticity ("% change in Y per % change in X") coefficients.
LOG_DEPENDENT: Final[bool] = True

# Predictors to log-transform. log1p is used so zeros are handled gracefully;
# any column containing negative values is left untransformed with a warning.
LOG_PREDICTORS: Final[tuple[str, ...]] = (
    "scheduled_hours",
    "revenue_miles",
    # Headway is right-skewed (a few infrequent express routes pull the tail);
    # log gives a proportional interpretation. span_hrs is roughly symmetric
    # but log is harmless and consistent with the other supply variables.
    "avg_headway_min",
    "span_hrs",
    # Precipitation/snow are right-skewed counts and magnitudes; log1p handles
    # the many zero-snow months gracefully. avg_temp_f is left linear because
    # temperature has a natural zero and can be negative.
    "max_daily_precip_in",
    "days_with_precip",
    "total_snow_in",
    "max_daily_snow_in",
    "pop_served",
    "low_income_served",
    "empl_served",
    "sites_served",
    "schools_served",
    "enrollment_served",
    "cabi_stations_served",
    "cabi_weekday_riders_served",
    # Saturday/Sunday bikeshare averages ship alongside the weekday figure.
    # The three day-type columns are strongly correlated across stations, so
    # analysts trimming a collinear fit will usually keep weekday only.
    "cabi_saturday_riders_served",
    "cabi_sunday_riders_served",
)

# Standardize predictors to mean 0 / unit variance before fitting. Useful for
# comparing effect sizes across predictors on different scales. Standardized
# (beta) coefficients are always reported regardless of this setting.
STANDARDIZE_PREDICTORS: Final[bool] = False

# Standard-error estimator: "classical" (homoskedastic), "HC1"
# (heteroskedasticity-robust, recommended for cross-sectional agency data), or
# "HC3" (MacKinnon-White; better coverage when observations number in the dozens).
SE_TYPE: Final[str] = "HC1"

# Predictors whose VIF exceeds this threshold are reported (and optionally
# dropped) as multicollinear. Set DROP_HIGH_VIF to True to prune them.
VIF_THRESHOLD: Final[float] = 10.0
DROP_HIGH_VIF: Final[bool] = False

# -----------------------------------------------------------------------------
#  Output behaviour
# -----------------------------------------------------------------------------

MAKE_PLOTS: Final[bool] = True
PLOT_STYLE: Final[dict[str, Any]] = {"figsize": (8, 6), "dpi": 120}

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so the analyst is never
# left with an output directory that lacks a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class OLSResult(NamedTuple):
    """Container for a fitted OLS model and its diagnostics.

    Attributes:
        term_names: Ordered design-matrix column names (intercept first).
        params: Estimated coefficients, aligned with ``term_names``.
        std_errors: Standard errors of ``params``.
        t_values: t-statistics (``params / std_errors``).
        p_values: Two-sided p-values for each t-statistic.
        conf_int: 95% confidence intervals, shape ``(k, 2)``.
        std_coef: Standardized ("beta") coefficients; ``nan`` for the intercept.
        vif: Variance inflation factors keyed by predictor (excludes intercept).
        fitted: Fitted values aligned with the modeling rows.
        residuals: Raw residuals (``y - fitted``).
        leverage: Hat-matrix diagonal per observation (influence).
        loo_residuals: Exact leave-one-out residuals (``e / (1 - h)``).
        n_obs: Number of observations used.
        n_params: Number of estimated parameters (including the intercept).
        r_squared: Coefficient of determination.
        adj_r_squared: Adjusted R^2.
        loo_r_squared: Leave-one-out (PRESS) R^2 — out-of-sample fit.
        f_stat: Overall F-statistic.
        f_pvalue: p-value of the F-statistic.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        sigma: Residual standard error.
        durbin_watson: Durbin-Watson autocorrelation statistic.
        condition_number: Condition number of the scaled design matrix.
        se_type: Standard-error estimator used ("classical" or "HC1").
    """

    term_names: list[str]
    params: np.ndarray
    std_errors: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    conf_int: np.ndarray
    std_coef: np.ndarray
    vif: dict[str, float]
    fitted: np.ndarray
    residuals: np.ndarray
    leverage: np.ndarray
    loo_residuals: np.ndarray
    n_obs: int
    n_params: int
    r_squared: float
    adj_r_squared: float
    loo_r_squared: float
    f_stat: float
    f_pvalue: float
    aic: float
    bic: float
    sigma: float
    durbin_watson: float
    condition_number: float
    se_type: str
    dropped_terms: tuple[str, ...] = ()


# =============================================================================
# DATA ASSEMBLY
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


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_key(series: pd.Series) -> pd.Series:
    """Normalize a join-key column so the anchor and a bundle match reliably.

    The anchor and bundle are produced on different machines from different
    files, so a key can arrive as ``101`` on one side and ``"101"`` (or
    ``"101.0"`` from a float round-trip) on the other. This collapses every
    key to a trimmed, upper-cased, space-free string and strips a single trailing
    ``.0`` — the same folding as ntd_anchor_builder's normalise_route, so an
    anchor keyed ``RT5`` matches a bundle keyed ``"Rt 5"``. The same helper
    is copied into prep_features_public.py and MUST stay byte-identical between the two.
    """
    out = series.astype("string").str.strip().str.upper()
    out = out.str.replace(" ", "", regex=False)
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.fillna("")


class BundleSpec(NamedTuple):
    """A prepped feature bundle described by the Part A manifest.

    Attributes:
        filename: Bundle CSV filename (resolved against ``BUNDLE_DIR``).
        join_keys: Columns the bundle is keyed on (joined onto the anchor).
        sha256: Expected SHA-256 of the bundle file (verified before joining).
        n_rows: Row count recorded by Part A (logged for cross-check).
    """

    filename: str
    join_keys: tuple[str, ...]
    sha256: str
    n_rows: int


def load_manifest(manifest_path: Path) -> list[BundleSpec]:
    """Parse the Part A manifest into an ordered list of bundle specs.

    Raises:
        FileNotFoundError: If ``manifest_path`` does not exist.
        ValueError: If the manifest is malformed or lists no bundles.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_bundles = data.get("bundles", [])
    if not raw_bundles:
        raise ValueError(f"Manifest '{manifest_path}' lists no bundles.")

    specs: list[BundleSpec] = []
    for entry in raw_bundles:
        try:
            specs.append(
                BundleSpec(
                    filename=str(entry["filename"]),
                    join_keys=tuple(entry["join_keys"]),
                    sha256=str(entry["sha256"]),
                    n_rows=int(entry["n_rows"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed bundle entry in '{manifest_path}': {entry}") from exc
    return specs


def assemble_model_table(
    anchor_path: Path,
    anchor_sheet: str | int,
    bundle_dir: Path,
    manifest_path: Path,
    verify_hashes: bool,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load the NTD anchor and left-join every prepped bundle onto it.

    A bundle is joined only if every one of its declared join keys exists in the
    anchor; otherwise it is skipped with a warning. This is what makes ``period``
    optional: a cross-sectional anchor (``route_id`` only) silently skips a
    ``period`` bundle, while a panel anchor joins it.

    Args:
        anchor_path: Path to the NTD anchor (the proprietary dependent variable).
        anchor_sheet: Worksheet for an Excel anchor (ignored for CSV).
        bundle_dir: Directory holding the bundle CSVs named in the manifest.
        manifest_path: Path to the Part A manifest JSON.
        verify_hashes: When True, each bundle's on-disk SHA-256 must match the
            manifest before it is joined.

    Returns:
        A tuple ``(merged, provenance)`` where ``merged`` is the modeling table
        (one row per anchor observation) and ``provenance`` is a list of
        ``(filename, sha256)`` pairs for every bundle that was actually joined,
        for recording in the run log.

    Raises:
        FileNotFoundError: If the anchor or a manifest-listed bundle is missing.
        ValueError: If ``verify_hashes`` is True and a bundle's hash mismatches.
    """
    merged = load_table(anchor_path, anchor_sheet)
    logging.info("Anchor table '%s' loaded: %d rows, %d cols.", anchor_path.name, *merged.shape)

    specs = load_manifest(manifest_path)
    provenance: list[tuple[str, str]] = []

    for spec in specs:
        bundle_path = bundle_dir / spec.filename
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Manifest lists '{spec.filename}' but it is not in {bundle_dir}."
            )

        actual_hash = _sha256_file(bundle_path)
        if verify_hashes and actual_hash != spec.sha256:
            raise ValueError(
                f"SHA-256 mismatch for bundle '{spec.filename}'. The transferred file does not "
                f"match the manifest (expected {spec.sha256[:12]}…, got {actual_hash[:12]}…). "
                "Re-transfer the bundle or set VERIFY_BUNDLE_HASHES = False to override."
            )

        keys = list(spec.join_keys)
        missing_anchor = [k for k in keys if k not in merged.columns]
        if missing_anchor:
            logging.warning(
                "Skipping bundle '%s': anchor lacks its join key(s) %s "
                "(expected when the anchor grain excludes them, e.g. a cross-sectional "
                "anchor and a 'period' bundle).",
                spec.filename,
                missing_anchor,
            )
            continue

        df = load_table(bundle_path)
        missing_bundle = [k for k in keys if k not in df.columns]
        if missing_bundle:
            raise KeyError(
                f"Bundle '{spec.filename}' is missing its own join key(s): {missing_bundle}."
            )

        # Canonicalize join keys on both sides so a string/int/float mismatch
        # across the machine boundary does not silently produce zero matches.
        for key in keys:
            merged[key] = _canonical_key(merged[key])
            df[key] = _canonical_key(df[key])

        value_cols = [c for c in df.columns if c not in keys]

        # A value column already present on the anchor (or an earlier bundle) would
        # make pandas rename BOTH sides with _x/_y suffixes, silently breaking every
        # downstream lookup of the original name — fail loudly instead.
        collisions = sorted(set(value_cols) & set(merged.columns))
        if collisions:
            raise ValueError(
                f"Bundle '{spec.filename}' carries column(s) {collisions} that already "
                "exist on the anchor or an earlier bundle. Drop or rename them on one "
                "side before joining."
            )

        n_dup = int(df.duplicated(subset=keys).sum())
        if n_dup:
            logging.warning(
                "Bundle '%s' has %d duplicate row(s) on %s; keeping the first occurrence "
                "of each key. Differing values in the dropped rows point to an upstream "
                "prep problem worth checking.",
                spec.filename,
                n_dup,
                keys,
            )
        subset = df[keys + value_cols].drop_duplicates(subset=keys)

        before = len(merged)
        # The merge indicator counts matches exactly, unlike inferring them from a
        # value column that may itself contain legitimate NaNs.
        merged = merged.merge(subset, on=keys, how="left", indicator="_bundle_matched")
        matched = int((merged["_bundle_matched"] == "both").sum())
        merged = merged.drop(columns="_bundle_matched")
        logging.info(
            "Joined bundle '%s' on %s: %d/%d anchor rows matched (%d feature col(s)).",
            spec.filename,
            keys,
            matched,
            before,
            len(value_cols),
        )
        match_rate = matched / before if before else 1.0
        if match_rate < MIN_BUNDLE_MATCH_RATE:
            raise ValueError(
                f"Bundle '{spec.filename}' matched only {matched}/{before} anchor rows "
                f"({match_rate:.0%}), below MIN_BUNDLE_MATCH_RATE ({MIN_BUNDLE_MATCH_RATE:.0%}). "
                "This usually means a join-key normalization or grain mismatch between the "
                "anchor and the bundle; lower the floor only if the bundle is legitimately "
                "sparse."
            )
        provenance.append((spec.filename, actual_hash))

    if not provenance:
        logging.warning(
            "No bundles were joined onto the anchor. Check that the manifest join keys "
            "match the anchor grain."
        )
    return merged, provenance


# =============================================================================
# MODEL-FRAME PREPARATION
# =============================================================================


def select_predictors(
    df: pd.DataFrame,
    explicit: tuple[str, ...],
    dependent: str,
    join_keys: tuple[str, ...],
    categoricals: tuple[str, ...],
) -> list[str]:
    """Resolve the predictor list, auto-selecting numerics when none is given.

    Args:
        df: The merged modeling table.
        explicit: Analyst-specified predictors; empty means auto-select.
        dependent: Dependent variable name (always excluded).
        join_keys: Join key columns (always excluded from auto-selection).
        categoricals: Categorical predictors (kept even though non-numeric).

    Returns:
        The ordered list of predictor column names.

    Raises:
        KeyError: If any explicitly named predictor is absent from ``df``.
    """
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise KeyError(f"Configured predictors not found in data: {missing}")
        return list(explicit)

    auto = [
        c
        for c in df.columns
        if c not in {dependent, *join_keys, *categoricals} and pd.api.types.is_numeric_dtype(df[c])
    ]
    logging.info("Auto-selected %d numeric predictors: %s", len(auto), auto)
    return auto + list(categoricals)


def build_design_matrix(
    df: pd.DataFrame,
    dependent: str,
    predictors: list[str],
    categoricals: tuple[str, ...],
    log_dependent: bool,
    log_predictors: tuple[str, ...],
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, np.ndarray]:
    """Construct the response vector and design matrix for OLS.

    Applies optional log transforms, one-hot encodes categoricals, drops rows
    with missing values, and prepends an intercept column.

    Args:
        df: The merged modeling table.
        dependent: Dependent variable name.
        predictors: Predictor column names (numeric and categorical).
        categoricals: Subset of ``predictors`` to one-hot encode.
        log_dependent: Whether to ``log`` the dependent variable.
        log_predictors: Numeric predictors to ``log1p``.
        standardize: Whether to z-score numeric predictors.

    Returns:
        A tuple ``(y, X, term_names, model_frame, keep_mask)`` where ``y`` is the
        response vector, ``X`` the design matrix (intercept first), ``term_names``
        the column labels for ``X``, ``model_frame`` the cleaned predictor frame
        prior to intercept/encoding, and ``keep_mask`` the boolean row mask
        (aligned with ``df``) of observations retained after dropping NaNs.

    Raises:
        ValueError: If no complete rows remain or the response is non-positive
            under a log transform.
    """
    numeric_predictors = [c for c in predictors if c not in categoricals]
    used_cols = [dependent, *predictors]
    frame = df[used_cols].copy()

    # Coerce numeric predictors and the response; categoricals stay as labels.
    for col in [dependent, *numeric_predictors]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    before = len(frame)
    missing_by_col = frame[used_cols].isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    frame = frame.dropna(subset=used_cols)
    keep_mask = df.index.isin(frame.index)
    dropped = before - len(frame)
    if dropped:
        # Name the offending columns: one sparse predictor can quietly cost a
        # big chunk of the sample.
        logging.warning(
            "Dropped %d row(s) with missing values across model columns. Missing "
            "counts by column: %s.",
            dropped,
            {col: int(n) for col, n in missing_by_col.items()},
        )
    if frame.empty:
        raise ValueError("No complete rows remain after dropping missing values.")

    # --- Response ---------------------------------------------------------
    y = frame[dependent].to_numpy(dtype=float)
    if log_dependent:
        if np.any(y <= 0):
            raise ValueError(
                f"LOG_DEPENDENT is True but '{dependent}' contains non-positive values; "
                "filter those rows or disable the log transform."
            )
        y = np.log(y)

    # --- Numeric predictors ----------------------------------------------
    feature_cols: dict[str, np.ndarray] = {}
    for col in numeric_predictors:
        values = frame[col].to_numpy(dtype=float)
        name = col
        if col in log_predictors:
            if np.any(values < 0):
                logging.warning("'%s' has negative values; skipping its log transform.", col)
            else:
                values = np.log1p(values)
                name = f"log_{col}"
        if standardize:
            std = values.std(ddof=0)
            values = (values - values.mean()) / std if std > 0 else values - values.mean()
            name = f"z_{name}"
        feature_cols[name] = values

    design = pd.DataFrame(feature_cols, index=frame.index)

    # --- Categorical predictors (one-hot, drop first level) --------------
    for col in categoricals:
        dummies = pd.get_dummies(frame[col].astype("string"), prefix=col, drop_first=True)
        design = pd.concat([design, dummies.astype(float)], axis=1)

    term_names = ["intercept", *design.columns.tolist()]
    x_matrix = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])

    return y, x_matrix, term_names, frame, keep_mask


# =============================================================================
# OLS ENGINE (numpy / scipy only)
# =============================================================================


def _variance_inflation_factors(x_matrix: np.ndarray, term_names: list[str]) -> dict[str, float]:
    """Compute VIFs for every non-intercept column of the design matrix.

    Each predictor is regressed on the remaining predictors (intercept
    included); ``VIF = 1 / (1 - R^2)``.

    Args:
        x_matrix: Design matrix with the intercept in column 0.
        term_names: Column labels aligned with ``x_matrix``.

    Returns:
        Mapping of predictor name to its VIF (``inf`` when perfectly collinear).
    """
    vif: dict[str, float] = {}
    n_terms = x_matrix.shape[1]
    for j in range(1, n_terms):  # skip intercept
        target = x_matrix[:, j]
        others = np.delete(x_matrix, j, axis=1)  # keeps the intercept column
        beta, _, _, _ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif[term_names[j]] = float("inf") if r_sq >= 1.0 else 1.0 / (1.0 - r_sq)
    return vif


def _durbin_watson(residuals: np.ndarray) -> float:
    """Return the Durbin-Watson statistic for first-order autocorrelation."""
    diff = np.diff(residuals)
    denom = float(residuals @ residuals)
    return float(diff @ diff) / denom if denom > 0 else float("nan")


def fit_ols(
    y: np.ndarray,
    x_matrix: np.ndarray,
    term_names: list[str],
    se_type: str,
) -> OLSResult:
    """Fit OLS and compute a full suite of diagnostics with numpy / scipy.

    Args:
        y: Response vector, length ``n``.
        x_matrix: Design matrix of shape ``(n, k)`` with an intercept column.
        term_names: Column labels aligned with ``x_matrix``.
        se_type: ``"classical"`` for homoskedastic SEs, ``"HC1"`` or ``"HC3"``
            for heteroskedasticity-robust SEs (HC3 has better small-sample
            coverage).

    Returns:
        A populated :class:`OLSResult`.

    Raises:
        ValueError: If there are fewer observations than parameters, or
            ``se_type`` is not recognized.
    """
    n_obs, n_params = x_matrix.shape
    if n_obs <= n_params:
        raise ValueError(
            f"Need more observations ({n_obs}) than parameters ({n_params}); "
            "reduce predictors or add data."
        )
    if se_type not in {"classical", "HC1", "HC3"}:
        raise ValueError(f"SE_TYPE must be 'classical', 'HC1', or 'HC3', got '{se_type}'.")

    beta, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)
    fitted = x_matrix @ beta
    residuals = y - fitted
    dof = n_obs - n_params

    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / dof

    sigma2 = ss_res / dof
    xtx_inv = np.linalg.inv(x_matrix.T @ x_matrix)
    # Hat-matrix diagonal, shared by HC3, the exact LOO residuals, and the
    # studentized per-observation diagnostics downstream.
    leverage = np.sum((x_matrix @ xtx_inv) * x_matrix, axis=1)

    if se_type == "classical":
        cov = sigma2 * xtx_inv
    elif se_type == "HC1":
        meat = x_matrix.T @ (x_matrix * (residuals**2)[:, None])
        cov = (n_obs / dof) * (xtx_inv @ meat @ xtx_inv)
    else:  # HC3: weight by e²/(1-h)² for better small-sample coverage.
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = (residuals / (1.0 - leverage)) ** 2
        meat = x_matrix.T @ (x_matrix * weights[:, None])
        cov = xtx_inv @ meat @ xtx_inv

    std_errors = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(std_errors > 0, beta / std_errors, np.nan)
    p_values = 2.0 * stats.t.sf(np.abs(t_values), dof)
    t_crit = stats.t.ppf(0.975, dof)
    conf_int = np.column_stack([beta - t_crit * std_errors, beta + t_crit * std_errors])

    # Standardized (beta) coefficients: b_j * sd(x_j) / sd(y). Intercept -> nan.
    y_sd = y.std(ddof=0)
    x_sd = x_matrix.std(axis=0, ddof=0)
    if y_sd > 0:
        std_coef = np.where(x_sd > 0, beta * x_sd / y_sd, np.nan)
    else:
        std_coef = np.full(n_params, np.nan)
    std_coef[0] = np.nan

    # Overall F-test (model vs. intercept-only).
    n_restrictions = n_params - 1
    if n_restrictions > 0 and (1.0 - r_squared) > 0:
        f_stat = (r_squared / n_restrictions) / ((1.0 - r_squared) / dof)
        f_pvalue = float(stats.f.sf(f_stat, n_restrictions, dof))
    else:
        f_stat = float("nan")
        f_pvalue = float("nan")

    # Information criteria (Gaussian log-likelihood).
    log_like = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(ss_res / n_obs) + 1)
    aic = 2 * n_params - 2 * log_like
    bic = n_params * np.log(n_obs) - 2 * log_like

    # Exact leave-one-out residuals via the hat matrix: e_loo = e / (1 - h_ii),
    # and the PRESS-based out-of-sample R².
    with np.errstate(divide="ignore", invalid="ignore"):
        loo_residuals = residuals / (1.0 - leverage)
    press = float(np.nansum(loo_residuals**2))
    loo_r_squared = 1.0 - press / ss_tot if ss_tot > 0 else float("nan")

    # Condition number of the scaled design matrix (multicollinearity signal).
    col_norms = np.linalg.norm(x_matrix, axis=0)
    scaled = x_matrix / np.where(col_norms > 0, col_norms, 1.0)
    condition_number = float(np.linalg.cond(scaled))

    return OLSResult(
        term_names=term_names,
        params=beta,
        std_errors=std_errors,
        t_values=t_values,
        p_values=p_values,
        conf_int=conf_int,
        std_coef=std_coef,
        vif=_variance_inflation_factors(x_matrix, term_names),
        fitted=fitted,
        residuals=residuals,
        leverage=leverage,
        loo_residuals=loo_residuals,
        n_obs=n_obs,
        n_params=n_params,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        loo_r_squared=loo_r_squared,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        aic=aic,
        bic=bic,
        sigma=float(np.sqrt(sigma2)),
        durbin_watson=_durbin_watson(residuals),
        condition_number=condition_number,
        se_type=se_type,
    )


def prune_collinear(
    y: np.ndarray,
    x_matrix: np.ndarray,
    term_names: list[str],
    threshold: float,
    se_type: str,
) -> OLSResult:
    """Iteratively drop the highest-VIF predictor until all VIFs are acceptable.

    Args:
        y: Response vector.
        x_matrix: Design matrix with an intercept column.
        term_names: Column labels aligned with ``x_matrix``.
        threshold: Maximum tolerated VIF.
        se_type: Standard-error estimator passed through to :func:`fit_ols`.

    Returns:
        The refit :class:`OLSResult` with ``dropped_terms`` recording removals.
    """
    dropped: list[str] = []
    names = list(term_names)
    matrix = x_matrix
    while True:
        vif = _variance_inflation_factors(matrix, names)
        if not vif:
            break
        worst_term = max(vif, key=lambda k: vif[k])
        if vif[worst_term] <= threshold or matrix.shape[1] <= 2:
            break
        worst_idx = names.index(worst_term)
        logging.warning("Dropping '%s' (VIF=%.1f > %.1f).", worst_term, vif[worst_term], threshold)
        matrix = np.delete(matrix, worst_idx, axis=1)
        names.pop(worst_idx)
        dropped.append(worst_term)

    result = fit_ols(y, matrix, names, se_type)
    return result._replace(dropped_terms=tuple(dropped))


# =============================================================================
# REPORTING
# =============================================================================


def build_coefficient_frame(result: OLSResult) -> pd.DataFrame:
    """Assemble the coefficient table (estimates, SEs, tests, CIs, VIFs)."""
    return pd.DataFrame(
        {
            "term": result.term_names,
            "coefficient": result.params,
            "std_error": result.std_errors,
            "t_value": result.t_values,
            "p_value": result.p_values,
            "ci_low_95": result.conf_int[:, 0],
            "ci_high_95": result.conf_int[:, 1],
            "std_coefficient": result.std_coef,
            "vif": [result.vif.get(name, float("nan")) for name in result.term_names],
        }
    )


def build_summary_frame(result: OLSResult, dependent: str, log_dependent: bool) -> pd.DataFrame:
    """Assemble the one-row-per-metric model summary table."""
    metrics: list[tuple[str, object]] = [
        ("dependent_variable", f"log({dependent})" if log_dependent else dependent),
        ("observations", result.n_obs),
        ("parameters", result.n_params),
        ("r_squared", round(result.r_squared, 4)),
        ("adj_r_squared", round(result.adj_r_squared, 4)),
        ("loo_r_squared", round(result.loo_r_squared, 4)),
        ("f_statistic", round(result.f_stat, 4)),
        ("f_pvalue", result.f_pvalue),
        ("residual_std_error", round(result.sigma, 4)),
        ("aic", round(result.aic, 2)),
        ("bic", round(result.bic, 2)),
        ("durbin_watson", round(result.durbin_watson, 4)),
        ("condition_number", round(result.condition_number, 1)),
        ("se_type", result.se_type),
        ("dropped_for_collinearity", ", ".join(result.dropped_terms) or "none"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def export_results(
    result: OLSResult,
    model_frame: pd.DataFrame,
    dependent: str,
    log_dependent: bool,
    output_dir: Path,
) -> Path:
    """Write the model summary, coefficients, correlations, and observations.

    Args:
        result: The fitted model.
        model_frame: Cleaned predictor frame (pre-encoding) for the kept rows.
        dependent: Dependent variable name.
        log_dependent: Whether the response was log-transformed.
        output_dir: Destination directory.

    Returns:
        The path to the written workbook.
    """
    workbook = output_dir / "monthly_ridership_results.xlsx"

    coef = build_coefficient_frame(result)
    summary = build_summary_frame(result, dependent, log_dependent)
    numeric = model_frame.select_dtypes(include="number")
    correlations = numeric.corr().reset_index(names="variable")

    observations = model_frame.copy()
    observations["fitted"] = result.fitted
    observations["residual"] = result.residuals
    resid_sd = result.residuals.std(ddof=0)
    observations["std_residual"] = result.residuals / resid_sd if resid_sd > 0 else result.residuals
    # Leverage-adjusted diagnostics: a high-leverage observation drags the fit
    # toward itself, so its raw residual understates how anomalous it is.
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = result.residuals / (result.sigma * np.sqrt(1.0 - result.leverage))
        cooks_d = (studentized**2 / result.n_params) * (result.leverage / (1.0 - result.leverage))
    observations["studentized_residual"] = studentized
    observations["loo_residual"] = result.loo_residuals
    observations["leverage"] = result.leverage
    observations["cooks_d"] = cooks_d

    with pd.ExcelWriter(workbook) as writer:
        summary.to_excel(writer, sheet_name="ModelSummary", index=False)
        coef.to_excel(writer, sheet_name="Coefficients", index=False)
        correlations.to_excel(writer, sheet_name="Correlations", index=False)
        observations.to_excel(writer, sheet_name="Observations", index=False)

    logging.info("Results workbook written to '%s'.", workbook)
    return workbook


def make_diagnostic_plots(result: OLSResult, output_dir: Path) -> None:
    """Save residual-vs-fitted, Q-Q, residual-histogram, and pred-vs-actual plots."""
    figsize = result_figsize()
    dpi = int(PLOT_STYLE.get("dpi", 120))

    # 1) Residuals vs. fitted.
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(result.fitted, result.residuals, s=18, alpha=0.7)
    ax.axhline(0.0, color="red", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_residuals_vs_fitted.png", dpi=dpi)
    plt.close(fig)

    # 2) Normal Q-Q plot of standardized residuals.
    resid_sd = result.residuals.std(ddof=0)
    std_resid = result.residuals / resid_sd if resid_sd > 0 else result.residuals
    osm, osr = stats.probplot(std_resid, dist="norm", fit=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(osm, osr, s=18, alpha=0.7)
    lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Normal Q-Q")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_qq.png", dpi=dpi)
    plt.close(fig)

    # 3) Residual histogram.
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(result.residuals, bins="auto", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_residual_hist.png", dpi=dpi)
    plt.close(fig)

    # 4) Predicted vs. actual.
    actual = result.fitted + result.residuals
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(actual, result.fitted, s=18, alpha=0.7)
    lims = [min(actual.min(), result.fitted.min()), max(actual.max(), result.fitted.max())]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs. Actual")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_predicted_vs_actual.png", dpi=dpi)
    plt.close(fig)

    logging.info("Diagnostic plots written to '%s'.", output_dir)


def result_figsize() -> tuple[float, float]:
    """Return the configured figure size as a ``(width, height)`` tuple."""
    size = PLOT_STYLE.get("figsize", (8, 6))
    width, height = size
    return float(width), float(height)


def log_model_report(result: OLSResult) -> None:
    """Emit a compact, human-readable summary of the fit to the logger."""
    logging.info("=== MODEL FIT ===")
    logging.info(
        "Observations: %d | Parameters: %d | SE: %s",
        result.n_obs,
        result.n_params,
        result.se_type,
    )
    logging.info(
        "R^2 = %.4f | Adj R^2 = %.4f | LOO R^2 = %.4f",
        result.r_squared,
        result.adj_r_squared,
        result.loo_r_squared,
    )
    logging.info(
        "F = %.3f (p = %.3g) | AIC = %.1f | BIC = %.1f",
        result.f_stat,
        result.f_pvalue,
        result.aic,
        result.bic,
    )
    logging.info(
        "Durbin-Watson = %.3f | Condition number = %.1f",
        result.durbin_watson,
        result.condition_number,
    )
    for name, coef, p_val in zip(result.term_names, result.params, result.p_values):
        flag = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        logging.info("  %-22s %+.5f  (p=%.3g) %s", name, coef, p_val, flag)


# =============================================================================
# RUN-LOG HELPERS
# =============================================================================


# Canonical version lives in utils/run_log.py — keep this copy in sync.
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
    output_dir: Path, provenance: list[tuple[str, str]], anchor_path: Path | None = None
) -> bool:
    """Write a run log of the configuration block into *output_dir*.

    Args:
        output_dir: Destination directory for the sidecar log.
        provenance: ``(filename, sha256)`` pairs for every bundle joined into
            the model, recorded so the secured-box output traces back to the
            exact Part A prep that fed it.
        anchor_path: The NTD anchor actually modeled; its SHA-256 is recorded so
            the run log is a complete provenance record (bundles are already
            hashed via the manifest).

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "monthly_ridership_model_runlog.txt"

    try:
        config_text: str = extract_config_block(SELF_PATH)
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    if provenance:
        provenance_lines = [f"  {name}  sha256={digest}" for name, digest in provenance]
    else:
        provenance_lines = ["  (none — no bundles were joined)"]

    anchor_sha = (
        _sha256_file(anchor_path)
        if anchor_path is not None and anchor_path.exists()
        else "(not recorded)"
    )

    lines: list[str] = [
        "=" * 72,
        "RIDERSHIP REGRESSION MODEL RUN LOG (PART B — secured box)",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Anchor:           {anchor_path if anchor_path is not None else '(not recorded)'}",
        f"Anchor SHA-256:   {anchor_sha}",
        f"Source script:    {SELF_PATH.resolve() if SELF_PATH.exists() else SELF_PATH}",
        "",
        "-" * 72,
        "FEATURE BUNDLE PROVENANCE (verified SHA-256 of joined bundles)",
        "-" * 72,
        *provenance_lines,
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
# MAIN
# =============================================================================


def main() -> None:
    """Assemble the modeling table, fit the OLS model, and export results."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if (
        "Path\\To\\Your" in str(ANCHOR_PATH)
        or "Path\\To\\Your" in str(OUTPUT_DIR)
        or ("Path\\To\\Your" in str(BUNDLE_DIR))
    ):
        logging.warning(
            "File paths are still set to their defaults. Update ANCHOR_PATH, OUTPUT_DIR, "
            "BUNDLE_DIR, and MANIFEST_PATH in the CONFIGURATION section before running."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === STEP 1: ASSEMBLE MODELING TABLE =====================================
    logging.info("=== STEP 1: ASSEMBLE MODELING TABLE ===")
    table, provenance = assemble_model_table(
        ANCHOR_PATH, ANCHOR_SHEET, BUNDLE_DIR, MANIFEST_PATH, VERIFY_BUNDLE_HASHES
    )
    if DEPENDENT_VAR not in table.columns:
        logging.error("Dependent variable '%s' not found in the assembled table.", DEPENDENT_VAR)
        sys.exit(1)

    if "month" in CATEGORICAL_PREDICTORS:
        if "period" in table.columns:
            table["month"] = pd.to_datetime(table["period"], format="%Y-%m").dt.strftime("%b")
            logging.info(
                "Derived 'month' column from 'period' (%d unique values).",
                table["month"].nunique(),
            )
        else:
            logging.warning(
                "'month' is in CATEGORICAL_PREDICTORS but 'period' is not in the table — skipping."
            )

    # === STEP 2: BUILD DESIGN MATRIX =========================================
    logging.info("=== STEP 2: BUILD DESIGN MATRIX ===")
    predictors = select_predictors(
        table, PREDICTORS, DEPENDENT_VAR, JOIN_KEYS, CATEGORICAL_PREDICTORS
    )
    y, x_matrix, term_names, model_frame, _ = build_design_matrix(
        table,
        DEPENDENT_VAR,
        predictors,
        CATEGORICAL_PREDICTORS,
        LOG_DEPENDENT,
        LOG_PREDICTORS,
        STANDARDIZE_PREDICTORS,
    )

    # === STEP 3: FIT OLS =====================================================
    logging.info("=== STEP 3: FIT OLS (%s standard errors) ===", SE_TYPE)
    if DROP_HIGH_VIF:
        result = prune_collinear(y, x_matrix, term_names, VIF_THRESHOLD, SE_TYPE)
    else:
        result = fit_ols(y, x_matrix, term_names, SE_TYPE)
    log_model_report(result)

    high_vif = {k: v for k, v in result.vif.items() if v > VIF_THRESHOLD}
    if high_vif:
        logging.warning(
            "Predictors with VIF > %.1f (possible multicollinearity): %s", VIF_THRESHOLD, high_vif
        )

    # === STEP 4: EXPORT ======================================================
    logging.info("=== STEP 4: EXPORT RESULTS ===")
    export_results(result, model_frame, DEPENDENT_VAR, LOG_DEPENDENT, OUTPUT_DIR)
    if MAKE_PLOTS:
        make_diagnostic_plots(result, OUTPUT_DIR)

    if not write_run_log(OUTPUT_DIR, provenance, ANCHOR_PATH) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        sys.exit(1)

    logging.info("All processing complete. Script completed successfully.")


if __name__ == "__main__":
    main()
