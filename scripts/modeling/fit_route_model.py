"""Cross-sectional route ridership model — ENGINE 1 (secured box).

Fits a single cross-sectional OLS of NTD boardings on route-level fundamentals and
emits the two reads the strategic deliverable needs:

    fitted (potential)  = what a route in this context + service "should" carry
                          -> sketch-level route planning.
    residual (over/under) = carries more / fewer riders than its fundamentals predict
                          -> strategic prioritization ("which routes are bad").

Because log(revenue_hours) is in the predictors, service *quantity* is already
controlled, so the residual is a productivity-adjusted over/under read rather than a
"this route is just small" artifact. The service *levers* (headway, span, speed) are
deliberately NOT regressors: they ride alongside the residual as diagnostic overlays
so an underperformer can be read as thin-service vs car-oriented-market vs cannibalized.

This is the secured-box fit: the NTD anchor (the dependent variable) is read here and
never leaves; the non-NTD feature tables (GTFS supply/competition + demographics) are
prepped elsewhere and joined in.

Inputs (all keyed on route_id = public route number):
    ANCHOR_PATH          NTD anchor: route_id + ntd_boardings (the proprietary DV).
    DEMOGRAPHICS_PATH    route_id + population / employment / enrollment / equity counts.
    GTFS_FEATURES_PATH   output of gtfs_route_features.py (supply + competition).

Outputs:
    ridership_route_model_results.xlsx
        ModelSummary | Coefficients | RoutePerformance | Correlations
    diagnostic plots + a run-log sidecar.

ArcGIS Pro Python stack only (numpy / scipy / pandas / matplotlib); no statsmodels,
scikit-learn, or pyarrow. Runs in a notebook via %run or as a script.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Final, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CONFIG_BEGIN_MARKER: Final[str] = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: Final[str] = "# === END CONFIG ==="


# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# --- Inputs (left-joined on route_id) ---------------------------------------
ANCHOR_PATH: Final[Path] = Path(r"Path\To\Your\ntd_route_boardings.csv")  # <<< EDIT ME
ANCHOR_SHEET: Final[str | int] = 0
# A long/panel anchor (route x period) is collapsed to one row per route, since
# engine 1 is cross-sectional. "mean" = typical period, "sum" = total across
# periods, "median" = robust typical. Applied to every numeric anchor column.
ANCHOR_AGG: Final[str] = "mean"  # <<< EDIT ME
# A zero in a monthly anchor almost always means the route did not operate / report
# that month, not that it ran empty. When True, zero-boarding months are dropped
# before the collapse so the average reflects operating months only (and the
# revenue_hours/_miles averages use the same months, keeping PPH consistent).
ANCHOR_EXCLUDE_ZERO_MONTHS: Final[bool] = True
DEMOGRAPHICS_PATH: Final[Path] = Path(r"Path\To\Your\input\route_demographics.csv")  # <<< EDIT ME
DEMOGRAPHICS_SHEET: Final[str | int] = 0
GTFS_FEATURES_PATH: Final[Path] = Path(
    r"Path\To\Your\prepped_features\gtfs_route_features.csv"
)  # <<< EDIT ME

# revenue_hours / revenue_miles are reported by BOTH the NTD anchor and the GTFS
# extractor. "ntd" keeps boardings and revenue_hours on the same reporting basis
# (so the residual is the official boardings-per-revenue-hour); "gtfs" uses the
# schedule-derived snapshot. The unused source's copies are dropped before the join.
SUPPLY_SOURCE: Final[str] = "ntd"  # "ntd" (anchor) or "gtfs" (extractor)
SHARED_SUPPLY_COLS: Final[tuple[str, ...]] = ("revenue_hours", "revenue_miles")

OUTPUT_DIR: Final[Path] = Path(r"Path\To\Your\output")  # <<< EDIT ME

ROUTE_KEY: Final[str] = "route_id"
DEPENDENT_VAR: Final[str] = "ntd_boardings"

# Rename messy source columns to the names used below (applied after load). The
# demographics export ships shapefile-derived counts like "Metrorail_Stations.shp".
COLUMN_ALIASES: Final[dict[str, str]] = {
    "Metrorail_Stations.shp": "Metrorail_Stations",
    "Hospitals_and_Urgent_Care_Facilities.shp": "Hospitals",
}

# --- Core regressors (predict potential) ------------------------------------
PREDICTORS: Final[tuple[str, ...]] = (
    "revenue_hours",  # supply quantity / productivity offset  [GTFS]
    "total_pop",  # population scale                       [demographics]
    "tot_empl",  # employment                            [demographics]
    "enrollment_9_12_served",  # high-school enrollment reached        [demographics]
    "enrollment_postsec_served",  # postsecondary enrollment reached    [demographics] (sparse)
    "Metrorail_Stations",  # Metro connections (count)             [demographics]
    "shared_stop_share",  # network redundancy                    [GTFS]
    "competition_intensity",  # cannibalization at shared stops       [GTFS]
    "is_express",  # service-type flag (derived below)
)

# Predictors to log1p (zeros handled). Counts/quantities are right-skewed; shares,
# the Metro count, and the binary flag are left linear.
LOG_PREDICTORS: Final[tuple[str, ...]] = (
    "revenue_hours",
    "total_pop",
    "tot_empl",
    "enrollment_9_12_served",
    "enrollment_postsec_served",
)
LOG_DEPENDENT: Final[bool] = True

# --- Service-type flag -------------------------------------------------------
# Routes flagged is_express = 1; everything else 0.
EXPRESS_ROUTES: Final[tuple[str, ...]] = (
    "159",
    "393",
    "394",
    "395",
    "396",
    "494",
    "495",
    "598",
    "599",
    "660",
    "663",
    "670",
    "671",
    "697",
    "698",
    "699",
    "722",
    "798",
    "835",
)

# --- Diagnostic overlays (attached to RoutePerformance, NOT regressors) ------
# Service levers + equity context to explain why a route under/over-performs.
OVERLAY_COLS: Final[tuple[str, ...]] = (
    "median_headway_min",
    "span_hours",
    "avg_speed_mph",
    "n_competitor_routes",
    "trips_per_day",
)
# Equity percentages derived as count / denominator (both from demographics).
EQUITY_PCT_SPEC: Final[tuple[tuple[str, str, str], ...]] = (
    ("pct_low_income", "low_income", "total_pop"),
    ("pct_minority", "minority", "total_pop"),
    ("pct_lep", "lep", "total_pop"),
    ("pct_lo_veh_hh", "lo_veh_hh", "total_hh"),
    ("pct_youth", "youth", "total_pop"),
    ("pct_elderly", "elderly", "total_pop"),
)

# --- Manual employment patches ----------------------------------------------
# LEHD LODES omits some federal/military worksites (the Pentagon is excluded from
# Census employment data). Each (site, jobs, routes) ADDS `jobs` to tot_empl for
# every listed route. Adding the full count to each serving route is correct, not
# double-counting: tot_empl is per-route catchment employment and catchments
# legitimately overlap at a shared destination. Clear the tuple to disable.
EMPL_COLUMN: Final[str] = "tot_empl"
MANUAL_EMPL_SITES: Final[tuple[tuple[str, float, tuple[str, ...]], ...]] = (
    ("Pentagon", 26000, ("306", "393", "394", "495", "396", "698", "834", "835", "598")),
)

# --- Estimator options -------------------------------------------------------
SE_TYPE: Final[str] = "HC1"  # "classical" or "HC1"
VIF_THRESHOLD: Final[float] = 10.0
# Std-residual magnitude beyond which a route is flagged strongly over/under.
PERF_FLAG_SD: Final[float] = 1.0
# Add an is_express 0/1 column to the RoutePerformance sheet (express routes
# dominate both residual tails, so it's handy for filtering/sorting them out).
SHOW_EXPRESS_COLUMN: Final[bool] = True

# How to treat express vs non-express routes. Express run on peak commuter demand
# and don't share a slope with locals, so pooling can distort the fundamentals:
#   "pooled"       one model, all routes, is_express as a dummy (default).
#   "separate"     two independent models (local + express), each residualized
#                  within its own group; outputs gain a "group" column.
#   "local_only"   fit only non-express routes.
#   "express_only" fit only express routes (small n — interpret with caution).
# In any non-pooled mode is_express is dropped as a regressor (it's constant
# within a group) but still shown in RoutePerformance.
ROUTE_GROUP_MODE: Final[str] = "pooled"

MAKE_PLOTS: Final[bool] = True
LOG_LEVEL: int = logging.INFO

# === END CONFIG ===


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class OLSResult(NamedTuple):
    """Fitted OLS model plus diagnostics."""

    term_names: list[str]
    params: np.ndarray
    std_errors: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    std_coef: np.ndarray
    vif: dict[str, float]
    fitted: np.ndarray
    residuals: np.ndarray
    loo_residuals: np.ndarray
    n_obs: int
    n_params: int
    r_squared: float
    adj_r_squared: float
    loo_r_squared: float
    f_stat: float
    f_pvalue: float
    sigma: float
    durbin_watson: float
    condition_number: float
    se_type: str


# =============================================================================
# DATA ASSEMBLY
# =============================================================================


def _canonical_key(series: pd.Series) -> pd.Series:
    """Normalize a join-key column (byte-identical to the other pipeline scripts)."""
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.fillna("")


def load_table(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported file type '{suffix}' for {path}.")


def assemble_model_table() -> pd.DataFrame:
    """Left-join demographics and GTFS features onto the NTD anchor on route_id."""
    anchor = load_table(ANCHOR_PATH, ANCHOR_SHEET)
    demo = load_table(DEMOGRAPHICS_PATH, DEMOGRAPHICS_SHEET)
    gtfs = load_table(GTFS_FEATURES_PATH)
    logging.info(
        "Loaded anchor (%d), demographics (%d), gtfs features (%d).",
        len(anchor),
        len(demo),
        len(gtfs),
    )

    anchor = anchor.rename(columns=COLUMN_ALIASES)
    demo = demo.rename(columns=COLUMN_ALIASES)
    gtfs = gtfs.rename(columns=COLUMN_ALIASES)
    for frame in (anchor, demo, gtfs):
        if ROUTE_KEY not in frame.columns:
            raise KeyError(f"Input is missing the join key '{ROUTE_KEY}'.")
        frame[ROUTE_KEY] = _canonical_key(frame[ROUTE_KEY])

    for label, frame in (("anchor", anchor), ("demographics", demo), ("gtfs", gtfs)):
        logging.info("%s columns: %s", label, list(frame.columns))

    # Engine 1 is cross-sectional: collapse a long/panel anchor to one row per route.
    # Every numeric anchor column is aggregated with ANCHOR_AGG, so a panel
    # ntd_boardings becomes (by default) a typical-period figure and any NTD service
    # columns survive the collapse.
    if anchor[ROUTE_KEY].duplicated().any():
        rows_per = len(anchor) / anchor[ROUTE_KEY].nunique()
        if DEPENDENT_VAR in anchor.columns:
            anchor[DEPENDENT_VAR] = pd.to_numeric(anchor[DEPENDENT_VAR], errors="coerce")

        if ANCHOR_EXCLUDE_ZERO_MONTHS and DEPENDENT_VAR in anchor.columns:
            routes_before = anchor[ROUTE_KEY].nunique()
            zero_rows = int((anchor[DEPENDENT_VAR] == 0).sum())
            anchor = anchor[anchor[DEPENDENT_VAR] != 0]
            dropped_routes = routes_before - anchor[ROUTE_KEY].nunique()
            logging.info(
                "Excluded %d zero-boarding month(s) as non-operating%s.",
                zero_rows,
                f"; {dropped_routes} route(s) had no operating months and were dropped"
                if dropped_routes
                else "",
            )

        num_cols = list(anchor.select_dtypes(include="number").columns)
        logging.warning(
            "Anchor is long-format (~%.1f rows/route); collapsing %d numeric column(s) "
            "to one row per route with ANCHOR_AGG='%s'.",
            rows_per,
            len(num_cols),
            ANCHOR_AGG,
        )
        anchor = anchor.groupby(ROUTE_KEY, as_index=False)[num_cols].agg(ANCHOR_AGG)
        logging.info("Anchor collapsed to %d routes.", len(anchor))

    # Feature tables are deduped (keep first) so they can't fan out the anchor.
    for label, frame in (("demographics", demo), ("gtfs features", gtfs)):
        dups = int(frame[ROUTE_KEY].duplicated().sum())
        if dups:
            logging.warning("%s has %d duplicate route_id(s); keeping first of each.", label, dups)
    demo = demo.drop_duplicates(subset=ROUTE_KEY)
    gtfs = gtfs.drop_duplicates(subset=ROUTE_KEY)

    # Resolve columns reported by more than one source so the join can't produce
    # _x/_y suffixes. revenue_hours/_miles come from NTD or GTFS per SUPPLY_SOURCE;
    # other GTFS metrics win over any stale demographics copy; NTD wins shared supply.
    if SUPPLY_SOURCE == "ntd":
        gtfs = gtfs.drop(
            columns=[c for c in SHARED_SUPPLY_COLS if c in gtfs.columns], errors="ignore"
        )
    elif SUPPLY_SOURCE == "gtfs":
        anchor = anchor.drop(
            columns=[c for c in SHARED_SUPPLY_COLS if c in anchor.columns], errors="ignore"
        )
    else:
        raise ValueError(f"SUPPLY_SOURCE must be 'ntd' or 'gtfs', got '{SUPPLY_SOURCE}'.")
    logging.info("Supply columns (revenue_hours/_miles) sourced from %s.", SUPPLY_SOURCE.upper())

    demo = demo.drop(
        columns=[
            c for c in demo.columns if c != ROUTE_KEY and (c in gtfs.columns or c in anchor.columns)
        ],
        errors="ignore",
    )
    merged = anchor.merge(demo, on=ROUTE_KEY, how="left").merge(gtfs, on=ROUTE_KEY, how="left")
    logging.info("Assembled table: %d routes x %d columns.", *merged.shape)
    return merged


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the express flag and the equity percentage overlays."""
    out = df.copy()

    express = {
        _canonical_key(pd.Series(EXPRESS_ROUTES)).iloc[i] for i in range(len(EXPRESS_ROUTES))
    }
    out["is_express"] = out[ROUTE_KEY].isin(express).astype(float)
    logging.info("Flagged %d express route(s).", int(out["is_express"].sum()))

    for pct_name, count_col, denom_col in EQUITY_PCT_SPEC:
        if count_col in out.columns and denom_col in out.columns:
            denom = pd.to_numeric(out[denom_col], errors="coerce").replace(0, np.nan)
            out[pct_name] = pd.to_numeric(out[count_col], errors="coerce") / denom
        else:
            logging.warning(
                "Skipping %s: missing %s or %s in the assembled table.",
                pct_name,
                count_col,
                denom_col,
            )

    # Manual employment patches for LODES-suppressed worksites (e.g. Pentagon).
    if MANUAL_EMPL_SITES and EMPL_COLUMN in out.columns:
        out[EMPL_COLUMN] = pd.to_numeric(out[EMPL_COLUMN], errors="coerce")
        for site, jobs, routes in MANUAL_EMPL_SITES:
            canon = set(_canonical_key(pd.Series(routes, dtype="object")))
            mask = out[ROUTE_KEY].isin(canon)
            out.loc[mask, EMPL_COLUMN] = out.loc[mask, EMPL_COLUMN].fillna(0) + jobs
            logging.info(
                "Added %s employment (+%.0f) to %s for %d/%d route(s).",
                site,
                jobs,
                EMPL_COLUMN,
                int(mask.sum()),
                len(routes),
            )
    elif MANUAL_EMPL_SITES:
        logging.warning(
            "MANUAL_EMPL_SITES set but '%s' not in the table; skipping patch.", EMPL_COLUMN
        )

    return out


# =============================================================================
# MODEL-FRAME PREPARATION
# =============================================================================


def build_design_matrix(
    df: pd.DataFrame,
    predictors: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, np.ndarray]:
    """Construct the response vector and design matrix; drop incomplete rows.

    Returns (y, X, term_names, model_frame, keep_mask). ``model_frame`` is the
    cleaned, kept-row frame (untransformed) so per-route outputs and overlays can
    be aligned back to it. ``predictors`` is the group's regressor list (may differ
    from the global PREDICTORS, e.g. is_express dropped for a single-group fit).
    """
    missing = [c for c in predictors if c not in df.columns]
    if missing:
        raise KeyError(
            f"Configured predictors not in the assembled table: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )

    used = [DEPENDENT_VAR, *predictors]
    # Carry is_express + overlays + equity %s even when not regressors, so the
    # per-route output can still label/diagnose them.
    carry = [
        c
        for c in ("is_express", *OVERLAY_COLS, *[s[0] for s in EQUITY_PCT_SPEC])
        if c in df.columns and c not in used
    ]
    frame = df[[ROUTE_KEY, *used, *carry]].copy()
    for col in used:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=used).reset_index(drop=True)
    dropped = before - len(frame)
    if dropped:
        logging.warning("Dropped %d route(s) with missing model columns.", dropped)
    if frame.empty:
        raise ValueError("No complete rows remain after dropping missing values.")

    y = frame[DEPENDENT_VAR].to_numpy(dtype=float)
    if LOG_DEPENDENT:
        if np.any(y <= 0):
            raise ValueError(
                f"LOG_DEPENDENT is True but '{DEPENDENT_VAR}' has non-positive values."
            )
        y = np.log(y)

    columns: dict[str, np.ndarray] = {}
    for col in predictors:
        values = frame[col].to_numpy(dtype=float)
        name = col
        if col in LOG_PREDICTORS:
            if np.any(values < 0):
                logging.warning("'%s' has negative values; skipping log transform.", col)
            else:
                values = np.log1p(values)
                name = f"log_{col}"
        columns[name] = values

    design = pd.DataFrame(columns, index=frame.index)
    term_names = ["intercept", *design.columns.tolist()]
    x_matrix = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    keep_mask = np.ones(len(frame), dtype=bool)
    return y, x_matrix, term_names, frame, keep_mask


# =============================================================================
# OLS ENGINE (numpy / scipy)
# =============================================================================


def _vif(x_matrix: np.ndarray, term_names: list[str]) -> dict[str, float]:
    """Variance inflation factor for each non-intercept column."""
    vif: dict[str, float] = {}
    for j in range(1, x_matrix.shape[1]):
        target = x_matrix[:, j]
        others = np.delete(x_matrix, j, axis=1)
        beta, _, _, _ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif[term_names[j]] = float("inf") if r_sq >= 1.0 else 1.0 / (1.0 - r_sq)
    return vif


def _durbin_watson(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    denom = float(residuals @ residuals)
    return float(diff @ diff) / denom if denom > 0 else float("nan")


def fit_ols(y: np.ndarray, x_matrix: np.ndarray, term_names: list[str], se_type: str) -> OLSResult:
    """Fit OLS with HC1/classical SEs plus VIF, Durbin-Watson, and exact LOO-R²."""
    n_obs, n_params = x_matrix.shape
    if n_obs <= n_params:
        raise ValueError(f"Need more routes ({n_obs}) than parameters ({n_params}).")
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
    else:  # HC1 sandwich
        meat = x_matrix.T @ (x_matrix * (residuals**2)[:, None])
        cov = (n_obs / dof) * (xtx_inv @ meat @ xtx_inv)
    std_errors = np.sqrt(np.diag(cov))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(std_errors > 0, beta / std_errors, np.nan)
    p_values = 2.0 * stats.t.sf(np.abs(t_values), dof)

    y_sd = y.std(ddof=0)
    x_sd = x_matrix.std(axis=0, ddof=0)
    std_coef = (
        np.where((x_sd > 0) & (y_sd > 0), beta * x_sd / y_sd, np.nan)
        if y_sd > 0
        else np.full(n_params, np.nan)
    )
    std_coef[0] = np.nan

    n_restrictions = n_params - 1
    if n_restrictions > 0 and (1.0 - r_squared) > 0:
        f_stat = (r_squared / n_restrictions) / ((1.0 - r_squared) / dof)
        f_pvalue = float(stats.f.sf(f_stat, n_restrictions, dof))
    else:
        f_stat = f_pvalue = float("nan")

    # Exact leave-one-out residuals via the hat matrix: e_loo = e / (1 - h_ii).
    leverage = np.sum((x_matrix @ xtx_inv) * x_matrix, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        loo_residuals = residuals / (1.0 - leverage)
    press = float(np.nansum(loo_residuals**2))
    loo_r_squared = 1.0 - press / ss_tot if ss_tot > 0 else float("nan")

    col_norms = np.linalg.norm(x_matrix, axis=0)
    scaled = x_matrix / np.where(col_norms > 0, col_norms, 1.0)

    return OLSResult(
        term_names=term_names,
        params=beta,
        std_errors=std_errors,
        t_values=t_values,
        p_values=p_values,
        std_coef=std_coef,
        vif=_vif(x_matrix, term_names),
        fitted=fitted,
        residuals=residuals,
        loo_residuals=loo_residuals,
        n_obs=n_obs,
        n_params=n_params,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        loo_r_squared=loo_r_squared,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        sigma=float(np.sqrt(sigma2)),
        durbin_watson=_durbin_watson(residuals),
        condition_number=float(np.linalg.cond(scaled)),
        se_type=se_type,
    )


# =============================================================================
# REPORTING
# =============================================================================


def build_summary_frame(result: OLSResult, group: Optional[str] = None) -> pd.DataFrame:
    """Assemble the one-row-per-metric model summary table."""
    metrics = [
        ("dependent_variable", f"log({DEPENDENT_VAR})" if LOG_DEPENDENT else DEPENDENT_VAR),
        ("observations", result.n_obs),
        ("parameters", result.n_params),
        ("r_squared", round(result.r_squared, 4)),
        ("adj_r_squared", round(result.adj_r_squared, 4)),
        ("loo_r_squared", round(result.loo_r_squared, 4)),
        ("f_statistic", round(result.f_stat, 4)),
        ("f_pvalue", result.f_pvalue),
        ("residual_std_error", round(result.sigma, 4)),
        ("durbin_watson", round(result.durbin_watson, 4)),
        ("condition_number", round(result.condition_number, 1)),
        ("se_type", result.se_type),
    ]
    df = pd.DataFrame(metrics, columns=["metric", "value"])
    if group is not None:
        df.insert(0, "group", group)
    return df


def build_coefficient_frame(result: OLSResult, group: Optional[str] = None) -> pd.DataFrame:
    """Assemble the coefficient table (estimates, SEs, tests, std betas, VIFs)."""
    df = pd.DataFrame(
        {
            "term": result.term_names,
            "coefficient": result.params,
            "std_error": result.std_errors,
            "t_value": result.t_values,
            "p_value": result.p_values,
            "std_coefficient": result.std_coef,
            "vif": [result.vif.get(name, float("nan")) for name in result.term_names],
        }
    )
    if group is not None:
        df.insert(0, "group", group)
    return df


def build_route_performance(
    result: OLSResult, model_frame: pd.DataFrame, predictors: list[str], group: Optional[str] = None
) -> pd.DataFrame:
    """Per-route potential (fitted), over/under (residual), and diagnostic overlays."""
    out = pd.DataFrame({ROUTE_KEY: model_frame[ROUTE_KEY].to_numpy()})
    if group is not None:
        out["group"] = group
    out[DEPENDENT_VAR] = model_frame[DEPENDENT_VAR].to_numpy()
    # Back-transform the fitted value to a boardings-scale "potential" (median, no
    # smearing correction — fine for ranking; note it under log retransform).
    out["fitted_potential"] = np.exp(result.fitted) if LOG_DEPENDENT else result.fitted
    out["residual_log"] = result.residuals
    resid_sd = result.residuals.std(ddof=0)
    out["std_residual"] = result.residuals / resid_sd if resid_sd > 0 else result.residuals
    out["loo_residual_log"] = result.loo_residuals

    def _flag(z: float) -> str:
        if z >= PERF_FLAG_SD:
            return "over"
        if z <= -PERF_FLAG_SD:
            return "under"
        return "expected"

    out["performance"] = [_flag(z) for z in out["std_residual"]]

    if SHOW_EXPRESS_COLUMN and "is_express" in model_frame.columns:
        out["is_express"] = (
            pd.to_numeric(model_frame["is_express"], errors="coerce").astype("Int64").to_numpy()
        )

    # Model fundamentals: the raw regressor values actually fit for this group, so
    # each route's drivers sit next to its residual. Shown untransformed.
    fundamentals = [c for c in predictors if c != "is_express" and c in model_frame.columns]
    for col in fundamentals:
        out[col] = pd.to_numeric(model_frame[col], errors="coerce").to_numpy()

    # Diagnostic overlays (service levers + equity %s), not in the model.
    overlay_cols = [
        c for c in (*OVERLAY_COLS, *[s[0] for s in EQUITY_PCT_SPEC]) if c in model_frame.columns
    ]
    for col in overlay_cols:
        out[col] = pd.to_numeric(model_frame[col], errors="coerce").to_numpy()

    return out.sort_values("std_residual").reset_index(drop=True)


def export_results(fits: list[tuple[str, list[str], OLSResult, pd.DataFrame]], multi: bool) -> Path:
    """Write the workbook for one or more group fits.

    ``fits`` is a list of (label, predictors, result, model_frame). With more than
    one group, ModelSummary / Coefficients / RoutePerformance gain a ``group``
    column; residuals are standardized within each group.
    """
    workbook = OUTPUT_DIR / "ridership_route_model_results.xlsx"

    summaries, coefs, perfs, mframes = [], [], [], []
    for label, predictors, result, mf in fits:
        g = label if multi else None
        summaries.append(build_summary_frame(result, g))
        coefs.append(build_coefficient_frame(result, g))
        perfs.append(build_route_performance(result, mf, predictors, g))
        mframes.append(mf)

    summary = pd.concat(summaries, ignore_index=True)
    coef = pd.concat(coefs, ignore_index=True)
    performance = pd.concat(perfs, ignore_index=True)

    # Correlations computed on the union of the modeled frames.
    union = pd.concat(mframes, ignore_index=True)
    numeric = union.select_dtypes(include="number")
    with np.errstate(invalid="ignore", divide="ignore"):  # constant subgroup cols -> NaN corr
        if DEPENDENT_VAR in numeric.columns:
            corr_col = numeric.corrwith(numeric[DEPENDENT_VAR]).rename(f"corr_with_{DEPENDENT_VAR}")
            corr_with_target = corr_col.reset_index().rename(columns={"index": "variable"})
            order = (
                corr_with_target[f"corr_with_{DEPENDENT_VAR}"]
                .abs()
                .sort_values(ascending=False)
                .index
            )
            corr_with_target = corr_with_target.loc[order].reset_index(drop=True)
        else:
            corr_with_target = pd.DataFrame(columns=["variable", f"corr_with_{DEPENDENT_VAR}"])

        modeled = [c for c in [DEPENDENT_VAR, *PREDICTORS] if c in numeric.columns]
        collinearity = numeric[modeled].corr().reset_index().rename(columns={"index": "variable"})

    with pd.ExcelWriter(workbook) as writer:
        summary.to_excel(writer, sheet_name="ModelSummary", index=False)
        coef.to_excel(writer, sheet_name="Coefficients", index=False)
        performance.to_excel(writer, sheet_name="RoutePerformance", index=False)
        corr_with_target.to_excel(writer, sheet_name="Correlations", index=False)
        collinearity.to_excel(writer, sheet_name="CollinearityMatrix", index=False)

    logging.info("Results workbook written to '%s'.", workbook)
    return workbook


def make_diagnostic_plots(result: OLSResult, suffix: str = "") -> None:
    """Save residuals-vs-fitted, normal Q-Q, and predicted-vs-actual plots."""
    figsize, dpi = (8, 6), 120

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(result.fitted, result.residuals, s=18, alpha=0.7)
    ax.axhline(0.0, color="red", linewidth=1)
    ax.set(
        xlabel="Fitted (log boardings)", ylabel="Residual", title=f"Residuals vs. Fitted{suffix}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"diag_residuals_vs_fitted{suffix}.png", dpi=dpi)
    plt.close(fig)

    resid_sd = result.residuals.std(ddof=0)
    std_resid = result.residuals / resid_sd if resid_sd > 0 else result.residuals
    osm, osr = stats.probplot(std_resid, dist="norm", fit=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(osm, osr, s=18, alpha=0.7)
    lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set(xlabel="Theoretical quantiles", ylabel="Std residuals", title=f"Normal Q-Q{suffix}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"diag_qq{suffix}.png", dpi=dpi)
    plt.close(fig)

    actual = result.fitted + result.residuals
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(actual, result.fitted, s=18, alpha=0.7)
    lims = [min(actual.min(), result.fitted.min()), max(actual.max(), result.fitted.max())]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set(xlabel="Actual (log)", ylabel="Predicted (log)", title=f"Predicted vs. Actual{suffix}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"diag_predicted_vs_actual{suffix}.png", dpi=dpi)
    plt.close(fig)

    logging.info("Diagnostic plots written to '%s'.", OUTPUT_DIR)


def _extract_config_block() -> str:
    try:
        source = Path(__file__)
    except NameError:
        return "(config block unavailable: not run from a source file)"
    lines = source.read_text(encoding="utf-8").splitlines()
    begin = end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if begin is None and s == CONFIG_BEGIN_MARKER:
            begin = i
        elif begin is not None and s == CONFIG_END_MARKER:
            end = i
            break
    if begin is None or end is None:
        return "(config block markers not found)"
    return "\n".join(lines[begin + 1 : end])


def write_run_log(fits: list[tuple[str, list[str], OLSResult, pd.DataFrame]]) -> None:
    """Write the run-log sidecar (per-group fit stats + verbatim config block)."""
    log_path = OUTPUT_DIR / "ridership_route_model_runlog.txt"
    fit_lines = [
        f"  {label:<10} n={result.n_obs:<4} k={result.n_params:<3} "
        f"R2={result.r_squared:.4f}  adjR2={result.adj_r_squared:.4f}  "
        f"LOO={result.loo_r_squared:.4f}"
        for label, _, result, _ in fits
    ]
    lines = [
        "=" * 72,
        "CROSS-SECTIONAL ROUTE RIDERSHIP MODEL RUN LOG (ENGINE 1, secured box)",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Anchor:           {ANCHOR_PATH}",
        f"Demographics:     {DEMOGRAPHICS_PATH}",
        f"GTFS features:    {GTFS_FEATURES_PATH}",
        f"Route group mode: {ROUTE_GROUP_MODE}",
        "",
        "-" * 72,
        "FITS",
        "-" * 72,
        *fit_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        _extract_config_block(),
        "=" * 72,
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Run log written to '%s'.", log_path)


def log_report(result: OLSResult, label: str = "") -> None:
    """Emit a compact, human-readable summary of a group fit to the logger."""
    tag = f" [{label}]" if label else ""
    logging.info("=== MODEL FIT%s ===", tag)
    logging.info("Routes: %d | Params: %d | SE: %s", result.n_obs, result.n_params, result.se_type)
    logging.info(
        "R2=%.4f | adjR2=%.4f | LOO-R2=%.4f | DW=%.3f | cond=%.1f",
        result.r_squared,
        result.adj_r_squared,
        result.loo_r_squared,
        result.durbin_watson,
        result.condition_number,
    )
    for name, coef, p in zip(result.term_names, result.params, result.p_values):
        flag = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        logging.info("  %-26s %+.4f  (p=%.3g) %s", name, coef, p, flag)
    high_vif = {k: round(v, 1) for k, v in result.vif.items() if v > VIF_THRESHOLD}
    if high_vif:
        logging.warning("VIF > %.1f: %s", VIF_THRESHOLD, high_vif)


# =============================================================================
# ENTRY POINT
# =============================================================================


def resolve_groups(table: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Resolve (label, predictors) groups from ROUTE_GROUP_MODE.

    In any non-pooled mode is_express is dropped as a regressor (it is constant
    within a single group). Returns the label + predictor list per group; the row
    subset is taken in run() from the is_express flag.
    """
    base = list(PREDICTORS)
    non_express = [p for p in base if p != "is_express"]
    mode = ROUTE_GROUP_MODE
    if mode == "pooled":
        return [("all", base)]
    if mode == "local_only":
        return [("local", non_express)]
    if mode == "express_only":
        return [("express", non_express)]
    if mode == "separate":
        return [("local", non_express), ("express", non_express)]
    raise ValueError(
        f"ROUTE_GROUP_MODE must be 'pooled', 'separate', 'local_only', or "
        f"'express_only', got '{mode}'."
    )


def _subset_for(table: pd.DataFrame, label: str) -> pd.DataFrame:
    """Row subset for a group label (full table for 'all')."""
    if label == "all":
        return table
    if "is_express" not in table.columns:
        logging.warning("No is_express column; cannot split — using all routes for '%s'.", label)
        return table
    flag = pd.to_numeric(table["is_express"], errors="coerce").fillna(0)
    return table[flag == (1 if label == "express" else 0)]


def run() -> Optional[OLSResult]:
    """Assemble the route table, fit the cross-sectional model(s), and export results."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if any(
        "Path\\To\\Your" in str(p)
        for p in (ANCHOR_PATH, DEMOGRAPHICS_PATH, GTFS_FEATURES_PATH, OUTPUT_DIR)
    ):
        logging.warning("Set the input/output paths (marked '# <<< EDIT ME') before running.")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table = assemble_model_table()
    table = derive_features(table)
    if DEPENDENT_VAR not in table.columns:
        raise KeyError(f"Dependent variable '{DEPENDENT_VAR}' not in the assembled table.")

    group_specs = resolve_groups(table)
    multi = len(group_specs) > 1
    logging.info(
        "Route group mode: %s (%d model%s).",
        ROUTE_GROUP_MODE,
        len(group_specs),
        "s" if multi else "",
    )

    fits: list[tuple[str, list[str], OLSResult, pd.DataFrame]] = []
    for label, predictors in group_specs:
        subtbl = _subset_for(table, label)
        n_params = len(predictors) + 1
        if len(subtbl) <= n_params:
            logging.warning(
                "Group '%s': %d routes <= %d params; skipping.", label, len(subtbl), n_params
            )
            continue
        if len(subtbl) < 10 * n_params:
            logging.warning(
                "Group '%s': only %d routes for %d parameters — coefficients are "
                "unstable/overfit-prone; lean on LOO-R² and read with caution.",
                label,
                len(subtbl),
                n_params,
            )
        y, x_matrix, term_names, model_frame, _ = build_design_matrix(subtbl, predictors)
        result = fit_ols(y, x_matrix, term_names, SE_TYPE)
        log_report(result, label if multi else "")
        fits.append((label, predictors, result, model_frame))

    if not fits:
        logging.error("No group produced a model; check ROUTE_GROUP_MODE and the data.")
        return None

    export_results(fits, multi)
    if MAKE_PLOTS:
        for label, _, result, _ in fits:
            make_diagnostic_plots(result, f"_{label}" if multi else "")
    write_run_log(fits)

    logging.info("Done.")
    return fits[0][2]


if __name__ == "__main__":
    run()
