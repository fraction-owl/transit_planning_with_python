"""Fit an ordinary-least-squares ridership model from agency planning data.

This script assembles a single modeling table from several common transit
planning sources and fits a multiple linear regression that explains NTD
ridership as a function of the service supplied and the context it operates in.
Everything is implemented with the ArcGIS Pro Python stack (``numpy`` /
``scipy`` / ``pandas`` / ``matplotlib``) so no machine-learning or statistics
package (``scikit-learn``, ``statsmodels``) is required.

Typical predictors:
    - Service supplied: scheduled / revenue hours and revenue miles (from NTD).
    - Exogenous context: gas prices, unemployment, weather (``exogenous_tools``).
    - Demographic service coverage: population, low-income, minority, zero-car
      households reached by each route (``service_coverage`` outputs).
    - Points-of-interest coverage: strategic sites / jobs reached by each route
      (``points_of_interest_coverage`` outputs).

Features:
    - Joins an "anchor" ridership table to any number of feature tables on a
      configurable key (cross-sectional by ``route_id`` or panel by
      ``route_id`` + ``period``).
    - Optional log transforms (for constant-elasticity interpretation),
      predictor standardization, and one-hot encoding of categoricals.
    - Hand-rolled OLS with classical or heteroskedasticity-robust (HC1)
      standard errors, t-tests, confidence intervals, R^2 / adjusted R^2,
      F-test, AIC / BIC, Durbin-Watson, design-matrix condition number, and
      per-predictor variance inflation factors (VIF).

Inputs:
    - One anchor table (CSV or XLSX) holding the dependent variable.
    - Zero or more feature tables (CSV or XLSX), each joined on shared keys.

Outputs:
    - An Excel workbook with model summary, coefficient, correlation, VIF, and
      per-observation (fitted / residual) sheets.
    - Optional diagnostic plots (residuals-vs-fitted, Q-Q, residual histogram,
      predicted-vs-actual).
    - A run log capturing the verbatim configuration block.
"""

from __future__ import annotations

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


# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===


class FeatureTable(NamedTuple):
    """A predictor source to be joined onto the anchor ridership table.

    Attributes:
        label: Human-readable name used in logs (e.g. ``"exogenous"``).
        path: Path to the CSV or XLSX file.
        join_keys: Columns shared with the anchor table used for the merge.
        keep_cols: Predictor columns to bring in. Leave empty to keep every
            non-key column found in the file.
        sheet: Worksheet name for XLSX inputs (ignored for CSV).
    """

    label: str
    path: Path
    join_keys: tuple[str, ...]
    keep_cols: tuple[str, ...] = ()
    sheet: str | int = 0


# -----------------------------------------------------------------------------
#  Input / output paths
# -----------------------------------------------------------------------------

# Anchor table: one row per observation, holding the dependent variable plus the
# "service supplied" predictors (scheduled/revenue hours and revenue miles).
# A route-level roll-up of the ntd_monthly_summary.py output is a natural fit.
ANCHOR_PATH: Final[Path] = Path(r"Path\To\Your\ntd_route_panel.csv")
ANCHOR_SHEET: Final[str | int] = 0  # worksheet name/index if ANCHOR_PATH is XLSX

OUTPUT_DIR: Final[Path] = Path(r"Path\To\Your\Output\Folder")

# Join key(s) tying every table together. Use ("route_id",) for a purely
# cross-sectional model, or ("route_id", "period") for a route x month panel.
JOIN_KEYS: Final[tuple[str, ...]] = ("route_id",)

# Feature tables to merge onto the anchor. Comment out any you do not have.
FEATURE_TABLES: Final[list[FeatureTable]] = [
    FeatureTable(
        label="exogenous",
        path=Path(r"Path\To\Your\exogenous_monthly.csv"),
        join_keys=("period",),  # system-wide series; varies by month only
        keep_cols=("gas_price", "unemployment_rate", "avg_temp_f", "total_precip_in"),
    ),
    FeatureTable(
        label="demographic_coverage",
        path=Path(r"Path\To\Your\service_demographics_by_route.csv"),
        join_keys=("route_id",),
        keep_cols=("pop_served", "low_income_served", "minority_served", "zero_car_hh_served"),
    ),
    FeatureTable(
        label="poi_coverage",
        path=Path(r"Path\To\Your\points_of_interest_coverage.csv"),
        join_keys=("route_id",),
        keep_cols=("sites_served", "jobs_served"),
    ),
]

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
    "gas_price",
    "unemployment_rate",
    "pop_served",
    "low_income_served",
    "sites_served",
)

# Categorical columns to one-hot encode (first level dropped as the reference).
CATEGORICAL_PREDICTORS: Final[tuple[str, ...]] = ()

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
    "pop_served",
    "low_income_served",
    "sites_served",
)

# Standardize predictors to mean 0 / unit variance before fitting. Useful for
# comparing effect sizes across predictors on different scales. Standardized
# (beta) coefficients are always reported regardless of this setting.
STANDARDIZE_PREDICTORS: Final[bool] = False

# Standard-error estimator: "classical" (homoskedastic) or "HC1"
# (heteroskedasticity-robust, recommended for cross-sectional agency data).
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
        n_obs: Number of observations used.
        n_params: Number of estimated parameters (including the intercept).
        r_squared: Coefficient of determination.
        adj_r_squared: Adjusted R^2.
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
    n_obs: int
    n_params: int
    r_squared: float
    adj_r_squared: float
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


def assemble_model_table(
    anchor_path: Path,
    anchor_sheet: str | int,
    feature_tables: list[FeatureTable],
) -> pd.DataFrame:
    """Left-join every feature table onto the anchor table.

    Args:
        anchor_path: Path to the anchor table holding the dependent variable.
        anchor_sheet: Worksheet for an Excel anchor (ignored for CSV).
        feature_tables: Feature sources to merge in, each on its own join keys.

    Returns:
        The merged modeling table, one row per anchor observation.

    Raises:
        KeyError: If a feature table is missing one of its declared join keys.
    """
    merged = load_table(anchor_path, anchor_sheet)
    logging.info("Anchor table '%s' loaded: %d rows, %d cols.", anchor_path.name, *merged.shape)

    for ft in feature_tables:
        df = load_table(ft.path, ft.sheet)
        missing = [k for k in ft.join_keys if k not in df.columns]
        if missing:
            raise KeyError(f"Feature table '{ft.label}' is missing join key(s): {missing}")
        missing_anchor = [k for k in ft.join_keys if k not in merged.columns]
        if missing_anchor:
            raise KeyError(
                f"Anchor table is missing join key(s) {missing_anchor} required by '{ft.label}'."
            )

        cols = list(ft.join_keys) + [
            c for c in (ft.keep_cols or _non_key_columns(df, ft.join_keys)) if c in df.columns
        ]
        subset = df[cols].drop_duplicates(subset=list(ft.join_keys))

        before = len(merged)
        merged = merged.merge(subset, on=list(ft.join_keys), how="left")
        matched = merged[cols[-1]].notna().sum() if len(cols) > len(ft.join_keys) else before
        logging.info(
            "Merged '%s' on %s: %d/%d rows matched.",
            ft.label,
            ft.join_keys,
            int(matched),
            before,
        )

    return merged


def _non_key_columns(df: pd.DataFrame, join_keys: tuple[str, ...]) -> list[str]:
    """Return DataFrame columns that are not join keys."""
    return [c for c in df.columns if c not in join_keys]


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
    frame = frame.dropna(subset=used_cols)
    keep_mask = df.index.isin(frame.index)
    dropped = before - len(frame)
    if dropped:
        logging.warning("Dropped %d row(s) with missing values across model columns.", dropped)
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
        se_type: ``"classical"`` for homoskedastic SEs or ``"HC1"`` for
            heteroskedasticity-robust SEs.

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
    if se_type not in {"classical", "HC1"}:
        raise ValueError(f"SE_TYPE must be 'classical' or 'HC1', got '{se_type}'.")

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

    if se_type == "classical":
        cov = sigma2 * xtx_inv
    else:  # HC1 sandwich estimator
        meat = x_matrix.T @ (x_matrix * (residuals**2)[:, None])
        cov = (n_obs / dof) * (xtx_inv @ meat @ xtx_inv)

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
        n_obs=n_obs,
        n_params=n_params,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
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
    workbook = output_dir / "ridership_regression_results.xlsx"

    coef = build_coefficient_frame(result)
    summary = build_summary_frame(result, dependent, log_dependent)
    numeric = model_frame.select_dtypes(include="number")
    correlations = numeric.corr().reset_index(names="variable")

    observations = model_frame.copy()
    observations["fitted"] = result.fitted
    observations["residual"] = result.residuals
    resid_sd = result.residuals.std(ddof=0)
    observations["std_residual"] = result.residuals / resid_sd if resid_sd > 0 else result.residuals

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
    logging.info("R^2 = %.4f | Adj R^2 = %.4f", result.r_squared, result.adj_r_squared)
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


def write_run_log(output_dir: Path) -> bool:
    """Write a run log of the configuration block into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "ridership_regression_model_runlog.txt"

    try:
        config_text: str = extract_config_block(Path(__file__))
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: list[str] = [
        "=" * 72,
        "RIDERSHIP REGRESSION MODEL RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {output_dir}",
        f"Source script:    {Path(__file__).resolve()}",
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

    if "Path\\To\\Your" in str(ANCHOR_PATH) or "Path\\To\\Your" in str(OUTPUT_DIR):
        logging.warning(
            "File paths are still set to their defaults. Update ANCHOR_PATH, "
            "OUTPUT_DIR, and the FEATURE_TABLES paths in the CONFIGURATION section "
            "before running."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === STEP 1: ASSEMBLE MODELING TABLE =====================================
    logging.info("=== STEP 1: ASSEMBLE MODELING TABLE ===")
    table = assemble_model_table(ANCHOR_PATH, ANCHOR_SHEET, FEATURE_TABLES)
    if DEPENDENT_VAR not in table.columns:
        logging.error("Dependent variable '%s' not found in the assembled table.", DEPENDENT_VAR)
        sys.exit(1)

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

    if not write_run_log(OUTPUT_DIR) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        sys.exit(1)

    logging.info("All processing complete. Script completed successfully.")


if __name__ == "__main__":
    main()
