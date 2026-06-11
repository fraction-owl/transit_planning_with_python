"""Fit machine-learning ridership models from the same agency planning data.

This is the machine-learning counterpart to ``ridership_regression_model.py``.
It ingests the *exact same* modeling table — the dependent NTD ridership
variable plus the service-supplied, exogenous, demographic, and
points-of-interest predictors — but replaces the single hand-rolled OLS fit
with tree-ensemble learners that capture the nonlinearities and feature
interactions a linear model cannot.

Why this can "do a better job" than the regression:
    - Ridership responds to service and context in ways that are rarely linear
      or additive: doubling frequency on an already-frequent route buys less
      than the first doubling, and the payoff from population coverage depends
      on whether the route also reaches jobs. Trees model those thresholds and
      interactions directly, without the analyst pre-specifying them.
    - Two complementary ensembles are fit and compared head-to-head:
        * Random forest — bagged trees, robust and low-maintenance.
        * Gradient boosting (histogram-based) — sequentially corrected trees,
          usually the strongest performer on tabular data.
    - The original OLS is kept as a baseline in the same comparison so the
      accuracy gain (or lack of one) is explicit and honest.

Honesty about accuracy:
    - Every headline metric is *out-of-sample*: predictions are pooled from
      K-fold cross-validation, so each row is scored by a model that never saw
      it during training. R^2 here is genuine generalization, not the in-sample
      R^2 the regression reports.
    - Metrics are reported in both the modeling space (log ridership, if
      ``LOG_DEPENDENT``) and back-transformed level space (boardings), because
      a great log-space fit can still mis-rank routes by absolute ridership.

Interpretability (the thing OLS coefficients give you for free):
    - Permutation importance ranks predictors by how much out-of-sample
      accuracy degrades when each is shuffled — a model-agnostic analog to
      standardized coefficients.
    - Partial-dependence plots show the marginal shape the model learned for
      the most important predictors, recovering the "effect of X" reading that
      a regression coefficient provides.

Unlike the regression, this script depends on scikit-learn. It targets the
"Demand Modeling (advanced)" audience that is comfortable installing it; it is
*not* meant to run inside a bare ArcGIS Pro environment.

Inputs:
    - One anchor table (CSV or XLSX) holding the dependent variable.
    - Zero or more feature tables (CSV or XLSX), each joined on shared keys.

Outputs:
    - An Excel workbook with a model-comparison sheet (cross-validated metrics
      for the baseline and both ensembles), a permutation-importance sheet for
      the best model, and a per-observation out-of-fold prediction sheet.
    - Optional diagnostic plots (predicted-vs-actual, residuals, importance bar
      chart, and partial-dependence curves for the top predictors).
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

try:
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import PartialDependenceDisplay, permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised only without sklearn
    raise SystemExit(
        "ridership_ml_model.py requires scikit-learn, which is not installed.\n"
        "Install it into your environment with:\n"
        "    pip install scikit-learn\n"
        "(This advanced ML script intentionally depends on it; the linear "
        "ridership_regression_model.py runs without it if you cannot install "
        "scikit-learn.)"
    ) from exc

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
        keep_cols=(
            "gas_price",
            "unemployment_rate",
            # Daily-derived weather aggregates (from clean_noaa_weather.aggregate_monthly).
            # These capture event intensity and frequency rather than bare monthly totals.
            "avg_temp_f",  # mean daily temperature
            "max_daily_precip_in",  # peak single-day rain (severity, not monthly total)
            "days_with_precip",  # count of rainy days
            "total_snow_in",  # total monthly snowfall
            "max_daily_snow_in",  # peak single-day snowfall
        ),
    ),
    FeatureTable(
        label="demographic_coverage",
        path=Path(r"Path\To\Your\service_demographics_by_route.csv"),
        join_keys=("route_id",),
        keep_cols=(
            "pop_served",
            "low_income_served",
            "minority_served",
            "zero_car_hh_served",
            "empl_served",  # Census LODES total employment within service catchment
        ),
    ),
    FeatureTable(
        label="poi_coverage",
        path=Path(r"Path\To\Your\points_of_interest_coverage.csv"),
        join_keys=("route_id",),
        keep_cols=("sites_served", "jobs_served"),
    ),
    # School points reached by each route. Unlike a generic POI, a school
    # carries a size we can measure, so we bring in both the count served and
    # the total enrollment those schools represent.
    FeatureTable(
        label="school_coverage",
        path=Path(r"Path\To\Your\school_coverage_by_route.csv"),
        join_keys=("route_id",),
        keep_cols=("schools_served", "enrollment_served"),
    ),
    # Capital Bikeshare (CABI) stations reached by each route. Like schools,
    # these come with a measurable level of activity, so we bring in both the
    # count served and the weekday average ridership those stations carry.
    FeatureTable(
        label="cabi_coverage",
        path=Path(r"Path\To\Your\cabi_coverage_by_route.csv"),
        join_keys=("route_id",),
        keep_cols=("cabi_stations_served", "cabi_weekday_riders_served"),
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
)

# Categorical columns to one-hot encode (first level dropped as the reference).
# "month" is derived automatically from the "period" column after table assembly.
CATEGORICAL_PREDICTORS: Final[tuple[str, ...]] = ("month",)

# -----------------------------------------------------------------------------
#  Transforms
# -----------------------------------------------------------------------------

# Log-transform the dependent variable. The tree ensembles do not need it, but
# it (a) keeps the target on the same scale as the regression for an apples-to-
# apples comparison and (b) stabilizes the long right tail typical of route
# ridership. Level-space metrics are always reported regardless.
LOG_DEPENDENT: Final[bool] = True

# Predictors to log-transform (log1p, so zeros are handled). Trees are invariant
# to monotone transforms of a single feature, so this mainly matters for the
# linear baseline; it is kept aligned with the regression for comparability.
LOG_PREDICTORS: Final[tuple[str, ...]] = (
    "scheduled_hours",
    "revenue_miles",
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
)

# -----------------------------------------------------------------------------
#  Estimator & evaluation options
# -----------------------------------------------------------------------------

# Number of cross-validation folds used for every out-of-sample metric. With
# small route-level tables, 5 is a sensible default; raise it if you have many
# observations and want lower-variance estimates.
CV_FOLDS: Final[int] = 5

# Fraction of rows held out (once) to compute permutation importance and
# partial-dependence curves on data the model did not train on.
HOLDOUT_FRACTION: Final[float] = 0.25

# Number of shuffles per predictor when estimating permutation importance.
PERMUTATION_REPEATS: Final[int] = 30

# Master seed for every randomized step (CV shuffling, the holdout split, the
# ensembles themselves) so a given config reproduces exactly.
RANDOM_STATE: Final[int] = 42

# Random-forest hyperparameters. Sensible, lightly-regularized defaults; tune
# n_estimators / max_depth / min_samples_leaf to your data size.
RF_PARAMS: Final[dict[str, Any]] = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
}

# Histogram gradient-boosting hyperparameters. early_stopping guards against
# over-fitting by watching an internal validation split.
GBT_PARAMS: Final[dict[str, Any]] = {
    "learning_rate": 0.05,
    "max_iter": 600,
    "max_depth": None,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": True,
}

# Number of top predictors to draw partial-dependence curves for.
N_PARTIAL_DEPENDENCE: Final[int] = 6

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


class ModelScore(NamedTuple):
    """Out-of-sample performance of one model, in log and level space.

    Attributes:
        name: Human-readable model label.
        r2_model: Cross-validated R^2 in the modeling space (logged target if
            ``LOG_DEPENDENT``, otherwise the raw target).
        rmse_model: Cross-validated RMSE in the modeling space.
        mae_model: Cross-validated MAE in the modeling space.
        r2_level: Cross-validated R^2 in back-transformed level space.
        rmse_level: Cross-validated RMSE in level space (original units).
        mae_level: Cross-validated MAE in level space.
        mape_level: Cross-validated mean absolute percentage error (level
            space), expressed as a fraction (0.10 == 10%).
        oof_pred_model: Pooled out-of-fold predictions in modeling space,
            aligned with the modeling rows.
    """

    name: str
    r2_model: float
    rmse_model: float
    mae_model: float
    r2_level: float
    rmse_level: float
    mae_level: float
    mape_level: float
    oof_pred_model: np.ndarray


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


def build_feature_frame(
    df: pd.DataFrame,
    dependent: str,
    predictors: list[str],
    categoricals: tuple[str, ...],
    log_dependent: bool,
    log_predictors: tuple[str, ...],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Construct the target vector and feature matrix for the ML models.

    Applies optional log transforms, one-hot encodes categoricals, and drops
    rows with missing values. No intercept column is added (the estimators
    handle that themselves) and predictors are *not* standardized here — the
    linear baseline scales internally via a pipeline and the trees do not need
    it.

    Args:
        df: The merged modeling table.
        dependent: Dependent variable name.
        predictors: Predictor column names (numeric and categorical).
        categoricals: Subset of ``predictors`` to one-hot encode.
        log_dependent: Whether to ``log`` the dependent variable.
        log_predictors: Numeric predictors to ``log1p``.

    Returns:
        A tuple ``(y, x_frame, model_frame)`` where ``y`` is the (possibly
        logged) target, ``x_frame`` the model-ready feature matrix with encoded
        column names, and ``model_frame`` the cleaned, pre-transform predictor
        frame (used for reporting per-observation results).

    Raises:
        ValueError: If no complete rows remain or the target is non-positive
            under a log transform.
    """
    numeric_predictors = [c for c in predictors if c not in categoricals]
    used_cols = [dependent, *predictors]
    frame = df[used_cols].copy()

    for col in [dependent, *numeric_predictors]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=used_cols).reset_index(drop=True)
    dropped = before - len(frame)
    if dropped:
        logging.warning("Dropped %d row(s) with missing values across model columns.", dropped)
    if frame.empty:
        raise ValueError("No complete rows remain after dropping missing values.")

    # --- Target -----------------------------------------------------------
    y = frame[dependent].to_numpy(dtype=float)
    if log_dependent:
        if np.any(y <= 0):
            raise ValueError(
                f"LOG_DEPENDENT is True but '{dependent}' contains non-positive values; "
                "filter those rows or disable the log transform."
            )
        y = np.log(y)

    # --- Numeric features -------------------------------------------------
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
        feature_cols[name] = values

    x_frame = pd.DataFrame(feature_cols, index=frame.index)

    # --- Categorical features (one-hot, drop first level) -----------------
    for col in categoricals:
        dummies = pd.get_dummies(frame[col].astype("string"), prefix=col, drop_first=True)
        x_frame = pd.concat([x_frame, dummies.astype(float)], axis=1)

    return y, x_frame, frame


# =============================================================================
# MODEL ZOO
# =============================================================================


def build_models(
    random_state: int,
    rf_params: dict[str, Any],
    gbt_params: dict[str, Any],
) -> dict[str, Any]:
    """Construct the estimators to compare, keyed by display name.

    The OLS baseline is wrapped in a standardizing pipeline so it matches the
    linear ``ridership_regression_model.py`` fit (scaling does not change its
    R^2 but keeps coefficients comparable); the ensembles are seeded for
    reproducibility.

    Args:
        random_state: Seed shared by every randomized estimator.
        rf_params: Keyword arguments for :class:`RandomForestRegressor`.
        gbt_params: Keyword arguments for :class:`HistGradientBoostingRegressor`.

    Returns:
        Ordered mapping ``{display_name: unfitted_estimator}``.
    """
    return {
        "OLS (baseline)": Pipeline([("scale", StandardScaler()), ("ols", LinearRegression())]),
        "Random Forest": RandomForestRegressor(random_state=random_state, **rf_params),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state, **gbt_params),
    }


# =============================================================================
# EVALUATION
# =============================================================================


def out_of_fold_predictions(
    model: Any,
    x_frame: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return pooled out-of-fold predictions for an unfitted estimator.

    Each row is predicted by a clone of ``model`` trained on the other folds,
    so the resulting vector is a fully out-of-sample picture of the model's
    behaviour across the whole table.

    Args:
        model: An unfitted scikit-learn estimator.
        x_frame: Feature matrix.
        y: Target vector aligned with ``x_frame``.
        n_splits: Number of cross-validation folds.
        random_state: Seed for the shuffled fold assignment.

    Returns:
        Predictions in the modeling space, aligned with the input rows.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in kfold.split(x_frame):
        fitted = clone(model)
        fitted.fit(x_frame.iloc[train_idx], y[train_idx])
        oof[test_idx] = fitted.predict(x_frame.iloc[test_idx])
    return oof


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination of ``predicted`` against ``actual``."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(actual - predicted)))


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error (fraction); ignores non-positive actuals."""
    mask = actual > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def score_model(
    name: str,
    model: Any,
    x_frame: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    log_dependent: bool,
) -> ModelScore:
    """Cross-validate one model and compute metrics in log and level space.

    Args:
        name: Display label for the model.
        model: An unfitted scikit-learn estimator.
        x_frame: Feature matrix.
        y: Target vector in the modeling space.
        n_splits: Number of cross-validation folds.
        random_state: Seed for the fold assignment.
        log_dependent: Whether ``y`` is log-transformed (controls the
            back-transform used for level-space metrics).

    Returns:
        A populated :class:`ModelScore`.
    """
    oof = out_of_fold_predictions(model, x_frame, y, n_splits, random_state)

    if log_dependent:
        actual_level = np.exp(y)
        pred_level = np.exp(oof)
    else:
        actual_level = y
        pred_level = oof

    return ModelScore(
        name=name,
        r2_model=_r2(y, oof),
        rmse_model=_rmse(y, oof),
        mae_model=_mae(y, oof),
        r2_level=_r2(actual_level, pred_level),
        rmse_level=_rmse(actual_level, pred_level),
        mae_level=_mae(actual_level, pred_level),
        mape_level=_mape(actual_level, pred_level),
        oof_pred_model=oof,
    )


def compute_permutation_importance(
    model: Any,
    x_frame: pd.DataFrame,
    y: np.ndarray,
    holdout_fraction: float,
    n_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, Any, pd.DataFrame, np.ndarray]:
    """Fit on a train split and rank predictors by held-out permutation importance.

    Args:
        model: An unfitted scikit-learn estimator.
        x_frame: Feature matrix.
        y: Target vector aligned with ``x_frame``.
        holdout_fraction: Fraction of rows reserved for the importance estimate.
        n_repeats: Number of shuffles per predictor.
        random_state: Seed for the split and the shuffles.

    Returns:
        A tuple ``(importance_frame, fitted_model, x_train, y_train)`` where
        ``importance_frame`` is sorted by mean importance (descending) and the
        fitted model plus its training data are returned for partial-dependence
        plotting.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x_frame, y, test_size=holdout_fraction, random_state=random_state
    )
    fitted = clone(model)
    fitted.fit(x_train, y_train)

    result = permutation_importance(
        fitted,
        x_test,
        y_test,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    importance = pd.DataFrame(
        {
            "feature": x_frame.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    return importance, fitted, x_train, y_train


# =============================================================================
# REPORTING
# =============================================================================


def build_comparison_frame(scores: list[ModelScore], log_dependent: bool) -> pd.DataFrame:
    """Assemble the model-comparison table from a list of scores."""
    space = "log" if log_dependent else "level"
    rows = [
        {
            "model": s.name,
            f"cv_r2_{space}": round(s.r2_model, 4),
            f"cv_rmse_{space}": round(s.rmse_model, 4),
            f"cv_mae_{space}": round(s.mae_model, 4),
            "cv_r2_level": round(s.r2_level, 4),
            "cv_rmse_level": round(s.rmse_level, 2),
            "cv_mae_level": round(s.mae_level, 2),
            "cv_mape_level": round(s.mape_level, 4),
        }
        for s in scores
    ]
    return pd.DataFrame(rows)


def export_results(
    scores: list[ModelScore],
    best: ModelScore,
    importance: pd.DataFrame,
    model_frame: pd.DataFrame,
    dependent: str,
    log_dependent: bool,
    output_dir: Path,
) -> Path:
    """Write the model comparison, feature importance, and OOF predictions.

    Args:
        scores: Cross-validated scores for every model.
        best: The winning model's score (per level-space R^2).
        importance: Permutation-importance frame for the winning model.
        model_frame: Cleaned predictor frame (pre-transform) for the kept rows.
        dependent: Dependent variable name.
        log_dependent: Whether the target was log-transformed.
        output_dir: Destination directory.

    Returns:
        The path to the written workbook.
    """
    workbook = output_dir / "ridership_ml_results.xlsx"

    comparison = build_comparison_frame(scores, log_dependent)

    observations = model_frame.copy()
    actual_level = observations[dependent].to_numpy(dtype=float)
    pred_level = np.exp(best.oof_pred_model) if log_dependent else best.oof_pred_model
    observations["predicted"] = pred_level
    observations["residual"] = actual_level - pred_level
    with np.errstate(divide="ignore", invalid="ignore"):
        observations["abs_pct_error"] = np.where(
            actual_level > 0, np.abs(observations["residual"]) / actual_level, np.nan
        )

    with pd.ExcelWriter(workbook) as writer:
        comparison.to_excel(writer, sheet_name="ModelComparison", index=False)
        importance.to_excel(writer, sheet_name="FeatureImportance", index=False)
        observations.to_excel(writer, sheet_name="Predictions", index=False)

    logging.info("Results workbook written to '%s'.", workbook)
    return workbook


def make_diagnostic_plots(
    best: ModelScore,
    importance: pd.DataFrame,
    fitted_model: Any,
    x_train: pd.DataFrame,
    model_frame: pd.DataFrame,
    dependent: str,
    log_dependent: bool,
    n_partial_dependence: int,
    output_dir: Path,
) -> None:
    """Save predicted-vs-actual, residual, importance, and partial-dependence plots."""
    figsize = result_figsize()
    dpi = int(PLOT_STYLE.get("dpi", 120))

    actual_level = model_frame[dependent].to_numpy(dtype=float)
    pred_level = np.exp(best.oof_pred_model) if log_dependent else best.oof_pred_model
    residual = actual_level - pred_level

    # 1) Predicted vs. actual (out-of-fold), level space.
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(actual_level, pred_level, s=18, alpha=0.7)
    lims = [
        min(actual_level.min(), pred_level.min()),
        max(actual_level.max(), pred_level.max()),
    ]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set_xlabel("Actual boardings")
    ax.set_ylabel("Predicted boardings (out-of-fold)")
    ax.set_title(f"Predicted vs. Actual — {best.name}")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_predicted_vs_actual.png", dpi=dpi)
    plt.close(fig)

    # 2) Residuals vs. predicted, level space.
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pred_level, residual, s=18, alpha=0.7)
    ax.axhline(0.0, color="red", linewidth=1)
    ax.set_xlabel("Predicted boardings (out-of-fold)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residuals vs. Predicted — {best.name}")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_residuals_vs_predicted.png", dpi=dpi)
    plt.close(fig)

    # 3) Permutation-importance bar chart (top features first).
    top = importance.head(max(n_partial_dependence, 10)).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], alpha=0.85)
    ax.set_xlabel("Permutation importance (drop in R^2)")
    ax.set_title(f"Feature Importance — {best.name}")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_feature_importance.png", dpi=dpi)
    plt.close(fig)

    # 4) Partial-dependence curves for the top predictors of the fitted model.
    is_linear = isinstance(fitted_model, Pipeline)
    if not is_linear and n_partial_dependence > 0:
        top_features = importance.head(n_partial_dependence)["feature"].tolist()
        top_features = [f for f in top_features if f in x_train.columns]
        if top_features:
            n_cols = min(3, len(top_features))
            n_rows = int(np.ceil(len(top_features) / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(figsize[0] * 1.4, figsize[1] * 0.6 * n_rows)
            )
            PartialDependenceDisplay.from_estimator(fitted_model, x_train, top_features, ax=axes)
            fig.suptitle(f"Partial Dependence — {best.name}")
            fig.tight_layout()
            fig.savefig(output_dir / "diag_partial_dependence.png", dpi=dpi)
            plt.close(fig)

    logging.info("Diagnostic plots written to '%s'.", output_dir)


def result_figsize() -> tuple[float, float]:
    """Return the configured figure size as a ``(width, height)`` tuple."""
    size = PLOT_STYLE.get("figsize", (8, 6))
    width, height = size
    return float(width), float(height)


def log_model_report(scores: list[ModelScore], best: ModelScore) -> None:
    """Emit a compact, human-readable comparison of the models to the logger."""
    logging.info("=== MODEL COMPARISON (cross-validated, out-of-sample) ===")
    logging.info("  %-20s %10s %14s %12s", "model", "R2(level)", "RMSE(level)", "MAPE")
    for s in scores:
        flag = "  <-- best" if s.name == best.name else ""
        logging.info(
            "  %-20s %10.4f %14.1f %11.1f%%%s",
            s.name,
            s.r2_level,
            s.rmse_level,
            100.0 * s.mape_level,
            flag,
        )


# =============================================================================
# RUN-LOG HELPERS
# =============================================================================


# Canonical version lives in utils/run_log.py — keep this copy in sync.
def extract_config_block(source_file: Path) -> str:
    r"""Return the text between the CONFIG markers in *source_file*.

    Reads ``source_file`` as UTF-8 text and slices out the lines strictly
    *between* the first occurrence of ``# === BEGIN CONFIG ===`` and the first
    subsequent occurrence of ``# === END CONFIG ===``.  The marker lines
    themselves are excluded; whitespace and inline comments inside the block
    are preserved verbatim.

    Args:
        source_file: Path to the Python source file to scan (typically
            ``Path(__file__)`` from the calling script).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    _BEGIN = "# === BEGIN CONFIG ==="
    _END = "# === END CONFIG ==="

    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == _BEGIN:
            begin_idx = i
        elif begin_idx is not None and stripped == _END:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. Expected '{_BEGIN}' and '{_END}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def write_run_log(output_dir: Path) -> bool:
    """Write a run log of the configuration block into *output_dir*.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_dir / "ridership_ml_model_runlog.txt"

    try:
        config_text: str = extract_config_block(Path(__file__))
    except (OSError, ValueError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: list[str] = [
        "=" * 72,
        "RIDERSHIP ML MODEL RUN LOG",
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
    """Assemble the modeling table, compare ML models, and export results."""
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

    # === STEP 2: BUILD FEATURE MATRIX ========================================
    logging.info("=== STEP 2: BUILD FEATURE MATRIX ===")
    predictors = select_predictors(
        table, PREDICTORS, DEPENDENT_VAR, JOIN_KEYS, CATEGORICAL_PREDICTORS
    )
    y, x_frame, model_frame = build_feature_frame(
        table,
        DEPENDENT_VAR,
        predictors,
        CATEGORICAL_PREDICTORS,
        LOG_DEPENDENT,
        LOG_PREDICTORS,
    )

    # === STEP 3: CROSS-VALIDATE & COMPARE MODELS =============================
    logging.info("=== STEP 3: CROSS-VALIDATE & COMPARE MODELS (%d folds) ===", CV_FOLDS)
    models = build_models(RANDOM_STATE, RF_PARAMS, GBT_PARAMS)
    scores = [
        score_model(name, model, x_frame, y, CV_FOLDS, RANDOM_STATE, LOG_DEPENDENT)
        for name, model in models.items()
    ]
    best = max(scores, key=lambda s: s.r2_level)
    log_model_report(scores, best)

    # === STEP 4: INTERPRET THE WINNING MODEL =================================
    logging.info("=== STEP 4: PERMUTATION IMPORTANCE (%s) ===", best.name)
    importance, fitted_model, x_train, _ = compute_permutation_importance(
        models[best.name],
        x_frame,
        y,
        HOLDOUT_FRACTION,
        PERMUTATION_REPEATS,
        RANDOM_STATE,
    )

    # === STEP 5: EXPORT ======================================================
    logging.info("=== STEP 5: EXPORT RESULTS ===")
    export_results(scores, best, importance, model_frame, DEPENDENT_VAR, LOG_DEPENDENT, OUTPUT_DIR)
    if MAKE_PLOTS:
        make_diagnostic_plots(
            best,
            importance,
            fitted_model,
            x_train,
            model_frame,
            DEPENDENT_VAR,
            LOG_DEPENDENT,
            N_PARTIAL_DEPENDENCE,
            OUTPUT_DIR,
        )

    if not write_run_log(OUTPUT_DIR) and REQUIRE_RUN_LOG:
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        sys.exit(1)

    logging.info("All processing complete. Script completed successfully.")


if __name__ == "__main__":
    main()
