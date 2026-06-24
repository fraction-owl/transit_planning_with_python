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

This is the secured-box fit (PART B): the NTD anchor (the dependent variable plus the
service-supplied predictors revenue_hours / revenue_miles) is read here and never
leaves; the non-NTD feature tables (GTFS competition + demographics) are prepped on the
unsecured box by prep_features_public.py (PART A) and transferred in as one governance-checked
CSV bundle per join-key signature, each verified against a manifest before joining.

Inputs:
    ANCHOR_PATH    NTD anchor: route_id + ntd_boardings (the proprietary DV) plus the
                   service-supplied predictors revenue_hours / revenue_miles.
    BUNDLE_DIR     Feature bundle CSVs produced by prep_features_public.py (Part A).
    MANIFEST_PATH  prep_features_public manifest (each bundle's join keys + SHA-256). A bundle
                   is joined only if every one of its join keys is present in the anchor,
                   so a cross-sectional (route_id) anchor auto-skips a period bundle.

Outputs:
    route_performance_results.xlsx
        ModelSummary | Coefficients | RoutePerformance | Correlations
    diagnostic plots + a run-log sidecar.

ArcGIS Pro Python stack only (numpy / scipy / pandas / matplotlib); no statsmodels,
scikit-learn, or pyarrow. Runs in a notebook via %run or as a script.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
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

# --- Inputs ------------------------------------------------------------------
# NTD anchor: the proprietary dependent variable (ntd_boardings) plus the
# service-supplied predictors (revenue_hours / revenue_miles). Read only here on
# the secured box; nothing NTD-derived ever crosses to the feature-prep side.
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

# Feature bundles produced by prep_features_public.py (PART A) and transferred in.
# BUNDLE_DIR holds the bundle CSVs; MANIFEST_PATH is the JSON sidecar listing each
# bundle's join keys, row/column counts, and SHA-256. A bundle is joined onto the
# anchor only if every one of its join keys is present in the anchor, so a
# cross-sectional (route_id) anchor silently skips a period-keyed bundle. Supply
# columns (revenue_hours / revenue_miles) are NTD-side and live on the anchor;
# prep_features_public governance forbids them from ever entering a feature bundle.
BUNDLE_DIR: Final[Path] = Path(r"Path\To\Your\prepped_features")  # <<< EDIT ME
MANIFEST_PATH: Final[Path] = Path(r"Path\To\Your\prepped_features\manifest.json")  # <<< EDIT ME
# When True, every bundle's on-disk SHA-256 must match the manifest before it is
# joined; a mismatch aborts the run (catches truncated/edited transfers).
VERIFY_BUNDLE_HASHES: Final[bool] = True

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
    "revenue_hours",  # supply quantity / productivity offset  [anchor/NTD]
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
# Routes flagged is_express = 1; everything else 0. Replace these placeholders
# with your agency's express route numbers.
EXPRESS_ROUTES: Final[tuple[str, ...]] = ("101", "202", "303")  # <<< EDIT ME

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

# --- Estimator options -------------------------------------------------------
SE_TYPE: Final[str] = "HC1"  # "classical" or "HC1"
VIF_THRESHOLD: Final[float] = 10.0
# Std-residual magnitude beyond which a route is flagged strongly over/under.
PERF_FLAG_SD: Final[float] = 1.0
# Add an is_express 0/1 column to the RoutePerformance sheet (express routes
# dominate both residual tails, so it's handy for filtering/sorting them out).
SHOW_EXPRESS_COLUMN: Final[bool] = True

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


class BundleSpec(NamedTuple):
    """A prepped feature bundle described by the prep_features_public (Part A) manifest.

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


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> list[BundleSpec]:
    """Parse the prep_features_public manifest into an ordered list of bundle specs.

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


def _collapse_panel_anchor(anchor: pd.DataFrame) -> pd.DataFrame:
    """Collapse a long/panel anchor (route x period) to one row per route.

    Engine 1 is cross-sectional. Every numeric anchor column is aggregated with
    ANCHOR_AGG, so a panel ntd_boardings becomes (by default) a typical-period
    figure and the NTD service columns (revenue_hours/_miles) survive the collapse
    on the same basis. A non-duplicated (already cross-sectional) anchor is
    returned unchanged.
    """
    if not anchor[ROUTE_KEY].duplicated().any():
        return anchor

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
    return anchor


def assemble_model_table() -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load the NTD anchor and left-join every prepped feature bundle onto it.

    The anchor holds the dependent variable plus the service-supplied predictors
    (revenue_hours / revenue_miles); a long/panel anchor is first collapsed to one
    row per route. Each bundle named in the manifest is then verified (SHA-256) and
    joined only if every one of its join keys is present in the anchor, so a
    route_id-only anchor silently skips a period bundle. Bundles are deduplicated
    on their keys so they can never fan out the anchor.

    Returns:
        ``(merged, provenance)`` where ``merged`` is the assembled route table and
        ``provenance`` is the ``(filename, sha256)`` of every bundle actually
        joined, recorded in the run log.

    Raises:
        KeyError: If the anchor or a joined bundle lacks the join key(s).
        FileNotFoundError: If a manifest-listed bundle is missing.
        ValueError: If VERIFY_BUNDLE_HASHES is True and a bundle's hash mismatches.
    """
    anchor = load_table(ANCHOR_PATH, ANCHOR_SHEET).rename(columns=COLUMN_ALIASES)
    if ROUTE_KEY not in anchor.columns:
        raise KeyError(f"Anchor is missing the join key '{ROUTE_KEY}'.")
    anchor[ROUTE_KEY] = _canonical_key(anchor[ROUTE_KEY])
    logging.info("Anchor '%s' loaded: %d rows, %d cols.", ANCHOR_PATH.name, *anchor.shape)

    merged = _collapse_panel_anchor(anchor)

    specs = load_manifest(MANIFEST_PATH)
    provenance: list[tuple[str, str]] = []
    for spec in specs:
        bundle_path = BUNDLE_DIR / spec.filename
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Manifest lists '{spec.filename}' but it is not in {BUNDLE_DIR}."
            )

        actual_hash = _sha256_file(bundle_path)
        if VERIFY_BUNDLE_HASHES and actual_hash != spec.sha256:
            raise ValueError(
                f"SHA-256 mismatch for bundle '{spec.filename}' (expected {spec.sha256[:12]}…, "
                f"got {actual_hash[:12]}…). Re-transfer the bundle or set "
                "VERIFY_BUNDLE_HASHES = False to override."
            )

        keys = list(spec.join_keys)
        missing_anchor = [k for k in keys if k not in merged.columns]
        if missing_anchor:
            logging.warning(
                "Skipping bundle '%s': anchor lacks its join key(s) %s "
                "(expected for a cross-sectional anchor and a 'period' bundle).",
                spec.filename,
                missing_anchor,
            )
            continue

        df = load_table(bundle_path).rename(columns=COLUMN_ALIASES)
        missing_bundle = [k for k in keys if k not in df.columns]
        if missing_bundle:
            raise KeyError(
                f"Bundle '{spec.filename}' is missing its own join key(s): {missing_bundle}."
            )

        # Canonicalize join keys on both sides so a string/int/float mismatch across
        # the machine boundary cannot silently produce zero matches.
        for key in keys:
            df[key] = _canonical_key(df[key])

        value_cols = [c for c in df.columns if c not in keys]
        subset = df[keys + value_cols].drop_duplicates(subset=keys)

        before = len(merged)
        merged = merged.merge(subset, on=keys, how="left")
        matched = int(merged[value_cols[0]].notna().sum()) if value_cols else before
        logging.info(
            "Joined bundle '%s' on %s: %d/%d routes matched (%d feature col(s)).",
            spec.filename,
            keys,
            matched,
            before,
            len(value_cols),
        )
        provenance.append((spec.filename, actual_hash))

    if not provenance:
        logging.warning(
            "No bundles were joined onto the anchor. Check that the manifest join keys "
            "match the anchor grain."
        )
    logging.info("Assembled table: %d routes x %d columns.", *merged.shape)
    return merged, provenance


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
    return out


# =============================================================================
# MODEL-FRAME PREPARATION
# =============================================================================


def build_design_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, np.ndarray]:
    """Construct the response vector and design matrix; drop incomplete rows.

    Returns (y, X, term_names, model_frame, keep_mask). ``model_frame`` is the
    cleaned, kept-row frame (untransformed) so per-route outputs and overlays can
    be aligned back to it.
    """
    missing = [c for c in PREDICTORS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Configured predictors not in the assembled table: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )

    used = [DEPENDENT_VAR, *PREDICTORS]
    frame = df[
        [
            ROUTE_KEY,
            *used,
            *[c for c in OVERLAY_COLS if c in df.columns],
            *[s[0] for s in EQUITY_PCT_SPEC if s[0] in df.columns],
        ]
    ].copy()
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
    for col in PREDICTORS:
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
    """Durbin-Watson statistic for residual autocorrelation."""
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


def build_summary_frame(result: OLSResult) -> pd.DataFrame:
    """Assemble the model-fit summary metrics into a tidy frame."""
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
    return pd.DataFrame(metrics, columns=["metric", "value"])


def build_coefficient_frame(result: OLSResult) -> pd.DataFrame:
    """Assemble the per-term coefficient table (estimate, SE, t, p, std coef, VIF)."""
    return pd.DataFrame(
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


def build_route_performance(result: OLSResult, model_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-route potential (fitted), over/under (residual), and diagnostic overlays."""
    out = pd.DataFrame({ROUTE_KEY: model_frame[ROUTE_KEY].to_numpy()})
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

    # Model fundamentals: the raw regressor values, so each route's drivers sit next
    # to its residual. Shown untransformed even where the fit logs them.
    fundamentals = [c for c in PREDICTORS if c != "is_express" and c in model_frame.columns]
    for col in fundamentals:
        out[col] = pd.to_numeric(model_frame[col], errors="coerce").to_numpy()

    # Diagnostic overlays (service levers + equity %s), not in the model.
    overlay_cols = [
        c for c in (*OVERLAY_COLS, *[s[0] for s in EQUITY_PCT_SPEC]) if c in model_frame.columns
    ]
    for col in overlay_cols:
        out[col] = pd.to_numeric(model_frame[col], errors="coerce").to_numpy()

    return out.sort_values("std_residual").reset_index(drop=True)


def export_results(result: OLSResult, model_frame: pd.DataFrame) -> Path:
    """Write the five-sheet results workbook and return its path."""
    workbook = OUTPUT_DIR / "route_performance_results.xlsx"
    summary = build_summary_frame(result)
    coef = build_coefficient_frame(result)
    performance = build_route_performance(result, model_frame)
    numeric = model_frame.select_dtypes(include="number")

    # (1) Bivariate correlation of every numeric variable (regressors + overlays +
    # equity %s) with the dependent variable — the "what tracks ridership" read.
    if DEPENDENT_VAR in numeric.columns:
        corr_col = numeric.corrwith(numeric[DEPENDENT_VAR]).rename(f"corr_with_{DEPENDENT_VAR}")
        corr_with_target = corr_col.reset_index().rename(columns={"index": "variable"})
        order = (
            corr_with_target[f"corr_with_{DEPENDENT_VAR}"].abs().sort_values(ascending=False).index
        )
        corr_with_target = corr_with_target.loc[order].reset_index(drop=True)
    else:
        corr_with_target = pd.DataFrame(columns=["variable", f"corr_with_{DEPENDENT_VAR}"])

    # (2) Full correlation matrix restricted to the MODELED variables (DV + the nine
    # regressors) — a focused collinearity companion to the VIF column. Raw, untransformed
    # variables; VIF in Coefficients reflects the logged design matrix.
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


def make_diagnostic_plots(result: OLSResult) -> None:
    """Write the residuals-vs-fitted, normal Q-Q, and predicted-vs-actual plots."""
    figsize, dpi = (8, 6), 120

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(result.fitted, result.residuals, s=18, alpha=0.7)
    ax.axhline(0.0, color="red", linewidth=1)
    ax.set(xlabel="Fitted (log boardings)", ylabel="Residual", title="Residuals vs. Fitted")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diag_residuals_vs_fitted.png", dpi=dpi)
    plt.close(fig)

    resid_sd = result.residuals.std(ddof=0)
    std_resid = result.residuals / resid_sd if resid_sd > 0 else result.residuals
    osm, osr = stats.probplot(std_resid, dist="norm", fit=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(osm, osr, s=18, alpha=0.7)
    lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set(xlabel="Theoretical quantiles", ylabel="Std residuals", title="Normal Q-Q")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diag_qq.png", dpi=dpi)
    plt.close(fig)

    actual = result.fitted + result.residuals
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(actual, result.fitted, s=18, alpha=0.7)
    lims = [min(actual.min(), result.fitted.min()), max(actual.max(), result.fitted.max())]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set(xlabel="Actual (log)", ylabel="Predicted (log)", title="Predicted vs. Actual")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diag_predicted_vs_actual.png", dpi=dpi)
    plt.close(fig)

    logging.info("Diagnostic plots written to '%s'.", OUTPUT_DIR)


def _extract_config_block() -> str:
    """Return the verbatim text between the CONFIG markers in this source file."""
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


def write_run_log(result: OLSResult, provenance: list[tuple[str, str]]) -> None:
    """Write the run-log sidecar (timestamp, fit headline, bundle provenance, config)."""
    log_path = OUTPUT_DIR / "route_performance_model_runlog.txt"
    bundle_lines = (
        [f"  {name}  sha256={sha}" for name, sha in provenance] if provenance else ["  (none)"]
    )
    lines = [
        "=" * 72,
        "CROSS-SECTIONAL ROUTE RIDERSHIP MODEL RUN LOG (ENGINE 1, secured box)",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Anchor:           {ANCHOR_PATH}",
        f"Bundle dir:       {BUNDLE_DIR}",
        f"Manifest:         {MANIFEST_PATH}",
        f"Routes modeled:   {result.n_obs}",
        f"R2 / adjR2 / LOO: {result.r_squared:.4f} / {result.adj_r_squared:.4f} / "
        f"{result.loo_r_squared:.4f}",
        "",
        "-" * 72,
        "FEATURE BUNDLES JOINED (verified provenance)",
        "-" * 72,
        *bundle_lines,
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        _extract_config_block(),
        "=" * 72,
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Run log written to '%s'.", log_path)


def log_report(result: OLSResult) -> None:
    """Log the fit headline and per-term coefficient significance to the logger."""
    logging.info("=== MODEL FIT ===")
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


def run() -> Optional[OLSResult]:
    """Assemble the route table, fit the cross-sectional model, and export results."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if any(
        "Path\\To\\Your" in str(p) for p in (ANCHOR_PATH, BUNDLE_DIR, MANIFEST_PATH, OUTPUT_DIR)
    ):
        logging.warning("Set the input/output paths (marked '# <<< EDIT ME') before running.")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table, provenance = assemble_model_table()
    table = derive_features(table)
    if DEPENDENT_VAR not in table.columns:
        raise KeyError(f"Dependent variable '{DEPENDENT_VAR}' not in the assembled table.")

    y, x_matrix, term_names, model_frame, _ = build_design_matrix(table)
    result = fit_ols(y, x_matrix, term_names, SE_TYPE)
    log_report(result)

    export_results(result, model_frame)
    if MAKE_PLOTS:
        make_diagnostic_plots(result)
    write_run_log(result, provenance)

    logging.info("Done.")
    return result


if __name__ == "__main__":
    run()
