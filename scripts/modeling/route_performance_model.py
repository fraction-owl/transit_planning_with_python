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

Express routes are held out of the fit (EXCLUDE_EXPRESS_FROM_FIT) rather than modeled
with a dummy: their ridership comes from park-and-ride catchment and destination
employment cores that the buffer-based demand features (population, enrollment, schools)
do not measure, so pooling them both mis-scores every express route and pulls the
land-use coefficients toward zero for the local routes the features actually describe.
They are reported descriptively (boardings per revenue hour) on the ExpressBench sheet.

This is the secured-box fit (PART B): the NTD anchor (the dependent variable plus the
service-supplied predictors revenue_hours / revenue_miles) is read here and never
leaves; the non-NTD feature tables (GTFS competition + demographics) are prepped on the
unsecured box by prep_features_public.py (PART A) and transferred in as one governance-checked
CSV bundle per join-key signature, each verified against a manifest before joining.

Inputs:
    ANCHOR_PATH    NTD anchor: route_id + ntd_boardings (the proprietary DV) plus the
                   service-supplied predictors revenue_hours / revenue_miles. Built by
                   ntd_anchor_builder.py on a single service-day basis (weekday /
                   saturday / sunday / combined); its service_day stamp is verified
                   against EXPECTED_SERVICE_DAY so the DV and the supply predictors
                   are guaranteed to sit on the same day-type basis.
    BUNDLE_DIR     Feature bundle CSVs produced by prep_features_public.py (Part A).
    MANIFEST_PATH  prep_features_public manifest (each bundle's join keys + SHA-256). A bundle
                   is joined only if every one of its join keys is present in the anchor,
                   so a cross-sectional (route_id) anchor auto-skips a period bundle.

Outputs:
    route_performance_results.xlsx
        ModelSummary | Coefficients | RoutePerformance | Correlations | CollinearityMatrix
        | ExpressBench (descriptive; only when express routes are held out of the fit)
    diagnostic plots + a run-log sidecar.

ArcGIS Pro Python stack only (numpy / scipy / pandas / matplotlib); no statsmodels,
scikit-learn, or pyarrow. Runs in a notebook via %run or as a script.

Typical usage:
    Transfer in the Part A bundle folder, update the paths in the CONFIGURATION section,
    and run on the secured box from a shell, ArcGIS Pro's Python window, or a Jupyter
    notebook (via ``%run``).
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
# that month, not that it ran empty. When True, zero- AND missing-boarding months
# are dropped before the collapse so the average reflects operating months only
# (and the revenue_hours/_miles averages use the same months, keeping PPH
# consistent — a NaN-boardings month would otherwise still count toward the
# revenue averages because mean() skips NaN only in the boardings column).
ANCHOR_EXCLUDE_ZERO_MONTHS: Final[bool] = True

# --- Service-day contract ------------------------------------------------------
# The wide anchor from ntd_anchor_builder.py encodes the service day in the column
# names (weekday_/saturday_/sunday_avg_*) rather than a per-row stamp, and this
# model already pins the day by targeting weekday_avg_ntd_boardings with the
# weekday supply averages — so DV and predictors are guaranteed on the same
# (weekday) basis. EXPECTED_SERVICE_DAY is therefore left blank: a wide anchor has
# no single service_day stamp to verify. It is retained only to keep verifying a
# legacy single-day anchor (one stamped service_day column); set it non-empty
# ("weekday" / "saturday" / "sunday" / "combined") only when pointing ANCHOR_PATH
# at such a legacy stamped anchor.
SERVICE_DAY_COLUMN: Final[str] = "service_day"
EXPECTED_SERVICE_DAY: Final[str] = ""  # <<< EDIT ME ("" = accept any / wide anchor)

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
# Minimum share of anchor routes each joined bundle must match (0..1). A join
# below this floor almost always means a key-normalization or grain problem
# upstream, so the run aborts rather than silently modeling a subset of the
# system. Lower it (or set 0.0) only for a bundle that is legitimately sparse.
MIN_BUNDLE_MATCH_RATE: Final[float] = 0.9

OUTPUT_DIR: Final[Path] = Path(r"Path\To\Your\output")  # <<< EDIT ME

ROUTE_KEY: Final[str] = "route_id"
# The anchor breaks out all three service days on a daily-average basis
# (weekday_/saturday_/sunday_avg_ntd_boardings, ..._avg_revenue_hours, ...). This
# model targets the weekday boardings average, paired with the weekday supply
# averages below; the saturday/sunday columns ride along unused here.
DEPENDENT_VAR: Final[str] = "weekday_avg_ntd_boardings"

# Rename messy source columns to the names used below (applied after load). The
# demographics export ships shapefile-derived counts like "Metrorail_Stations.shp".
COLUMN_ALIASES: Final[dict[str, str]] = {
    "Metrorail_Stations.shp": "Metrorail_Stations",
    "Hospitals_and_Urgent_Care_Facilities.shp": "Hospitals",
}

# --- Core regressors (predict potential) ------------------------------------
PREDICTORS: Final[tuple[str, ...]] = (
    "weekday_avg_revenue_hours",  # supply quantity / productivity offset  [anchor/NTD]
    "total_pop",  # population scale                       [demographics]
    "tot_empl",  # employment                            [demographics]
    "enrollment_9_12_served",  # high-school enrollment reached        [demographics]
    "enrollment_postsec_served",  # postsecondary enrollment reached    [demographics] (sparse)
    "Metrorail_Stations",  # Metro connections (count)             [demographics]
    "shared_stop_share",  # network redundancy                    [GTFS]
    "competition_intensity",  # cannibalization at shared stops       [GTFS]
    "transfer_route_count",  # network connectivity / feeder pull     [GTFS]
    "stops_per_mile",  # stop access density (local vs limited) [GTFS]
    # NOTE: is_express is intentionally NOT a regressor. Express routes are held out
    # of the fit (EXCLUDE_EXPRESS_FROM_FIT) and benchmarked descriptively instead,
    # because the buffer-based demand features misrepresent their catchment. To
    # restore the old pooled model with an express intercept dummy, re-add
    # "is_express" here and set EXCLUDE_EXPRESS_FROM_FIT = False.
)

# Predictors to log1p (zeros handled). Counts/quantities are right-skewed; shares,
# rates (stops_per_mile), the small Metro/transfer counts, and the binary flag are
# left linear (transfer_route_count is a low-count integer, so its raw scale reads
# as "each extra connection", not a log-elasticity).
LOG_PREDICTORS: Final[tuple[str, ...]] = (
    "weekday_avg_revenue_hours",
    "total_pop",
    "tot_empl",
    "enrollment_9_12_served",
    "enrollment_postsec_served",
)
LOG_DEPENDENT: Final[bool] = True
# What to do when the dependent variable is <= 0 under LOG_DEPENDENT. True drops
# those routes with a logged count — right for Saturday/Sunday runs, where a zero
# usually means "this route runs no weekend service", a non-operator rather than
# an underperformer. False keeps the hard abort — right when a zero in a weekday
# anchor is a data error worth stopping on.
DROP_NONPOSITIVE_DEPENDENT: Final[bool] = True

# --- Service-type flag -------------------------------------------------------
# Routes flagged is_express = 1; everything else 0. List your agency's express
# route numbers here. Leave it empty (the default) if your system has no express
# routes — nothing is flagged, nothing is held out, and no ExpressBench sheet is
# written, so the express machinery below is a no-op you can ignore entirely.
EXPRESS_ROUTES: Final[tuple[str, ...]] = ()  # <<< EDIT ME (empty = no express routes)

# When True, express routes are dropped from the regression entirely (not modeled
# with a dummy) and reported descriptively on the ExpressBench sheet instead. Their
# ridership is driven by park-and-ride catchment and destination employment cores
# that the buffer-based demand features (total_pop, enrollment_*, schools) do not
# capture, so pooling them (a) mis-scores every express route against a local-demand
# model and (b) drags the land-use coefficients toward zero for the local routes the
# features actually describe. Set False (and re-add "is_express" to PREDICTORS) only
# to reproduce the old pooled fit with an express intercept shift. Harmless to leave
# True when EXPRESS_ROUTES is empty: no routes are flagged, so nothing is held out.
EXCLUDE_EXPRESS_FROM_FIT: Final[bool] = True

# --- Demand diagnostic (supply-free companion fit) ---------------------------
# The primary fit includes log(revenue_hours), which dominates and is collinear with
# the demand features (service is allocated where the riders already are), so a
# land-use coefficient can read non-significant simply because revenue_hours has
# already absorbed its signal. When True, a second model is fit on the SAME local
# routes with the supply term removed — a pure demand model — and its coefficients are
# dropped beside the primary ones (SupplyVsDemand sheet). A feature that jumps to
# significant here is a demand driver currently MASKED by supply, not an absent effect.
# NOTE: this demand model's residual is NOT productivity-adjusted (the "this route is
# just small" artifact returns), so it is a diagnostic companion only, never the
# prioritization engine — read its coefficients, not its residuals.
SUPPLY_PREDICTOR: Final[str] = "weekday_avg_revenue_hours"
FIT_DEMAND_DIAGNOSTIC: Final[bool] = True

# --- Diagnostic overlays (attached to RoutePerformance + Correlations, NOT regressors) ------
# Two roles, neither in the fit: (a) service levers + equity context that explain why a
# route under/over-performs, and (b) a screening bench of candidate features not (yet)
# promoted to PREDICTORS. Everything here flows into the Correlations sheet (bivariate
# corr with the DV) and RoutePerformance without touching the coefficients — and, unlike
# a predictor, an overlay is never in the dropna subset, so a sparse candidate (e.g.
# bikeshare) can be assessed here without silently costing modeled routes. A candidate
# that correlates cleanly here is a promotion candidate for PREDICTORS. Columns absent
# from the assembled table are silently skipped, so the bench can list more than any one
# agency's bundles will supply.
OVERLAY_COLS: Final[tuple[str, ...]] = (
    # Service levers + demand context (diagnostic).
    "median_headway_min",
    "span_hours",
    "avg_speed_mph",
    "n_competitor_routes",
    "trips_per_day",
    # Candidate features under evaluation — screen via Correlations before promoting.
    "jobs_served",  # employment reached at POI job sites      [POI coverage]
    "sites_served",  # POI destinations reached                 [POI coverage]
    "schools_served",  # schools reached                          [school coverage]
    "enrollment_served",  # total enrollment reached (all grades)    [school coverage]
    "enrollment_1_8_served",  # K-8 enrollment reached                   [school coverage]
    "cabi_stations_served",  # bikeshare docks reached (first/last mi)  [bikeshare]
    "cabi_weekday_riders_served",  # bikeshare demand reached (non-holiday weekdays) [bikeshare]
    "cabi_saturday_riders_served",  # bikeshare demand reached (Saturdays)     [bikeshare]
    "cabi_sunday_riders_served",  # bikeshare demand reached (Sun + holidays) [bikeshare]
    "n_stops",  # stop count (access points)               [GTFS]
    "route_length_mi",  # route length (scale)                     [GTFS]
    "n_directions",  # 1 = loop, 2 = bidirectional              [GTFS]
    "pct_day_with_service",  # span coverage completeness               [GTFS]
    "competitor_trips_at_shared_stops",  # raw competing supply at shared stops [GTFS]
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
# "classical", "HC1", or "HC3". HC3 has better small-sample coverage
# (MacKinnon-White) and is worth preferring when routes number in the dozens.
SE_TYPE: Final[str] = "HC1"
VIF_THRESHOLD: Final[float] = 10.0
# Studentized-residual magnitude beyond which a route is flagged strongly
# over/under. Studentized (leverage-adjusted) rather than raw residuals, so a
# high-leverage route that drags the fit toward itself is still flagged.
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
    leverage: np.ndarray
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
    """Normalize a join-key column (byte-identical to the other pipeline scripts).

    Trims, upper-cases, removes internal spaces, and strips a single trailing
    ``.0`` — the same folding as ntd_anchor_builder's normalise_route, so an
    anchor keyed ``RT5`` matches a GTFS-derived bundle keyed ``"Rt 5"``.
    """
    out = series.astype("string").str.strip().str.upper()
    out = out.str.replace(" ", "", regex=False)
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


def _verify_service_day(anchor: pd.DataFrame) -> Optional[str]:
    """Verify the anchor's service-day stamp against EXPECTED_SERVICE_DAY.

    Returns the anchor's (single) service day, or ``None`` when the anchor has no
    stamp column (pre-stamp anchors warn but run). The stamp is written by
    ntd_anchor_builder.py; the residual read is only meaningful when every row —
    DV and supply predictors alike — sits on one day-type basis.

    Raises:
        ValueError: If the anchor mixes service days, or its stamp disagrees with
            a non-empty EXPECTED_SERVICE_DAY.
    """
    if SERVICE_DAY_COLUMN not in anchor.columns:
        if EXPECTED_SERVICE_DAY:
            logging.warning(
                "Anchor has no '%s' column, so it cannot be verified as a '%s' anchor. "
                "Rebuild it with ntd_anchor_builder.py to get the stamp.",
                SERVICE_DAY_COLUMN,
                EXPECTED_SERVICE_DAY,
            )
        return None

    days = sorted(set(anchor[SERVICE_DAY_COLUMN].astype(str).str.strip().str.lower()))
    if len(days) != 1:
        raise ValueError(
            f"Anchor mixes service days {days}; this model analyzes one day type per "
            "run. Rebuild single-day anchors with ntd_anchor_builder.py."
        )
    day = days[0]
    expected = EXPECTED_SERVICE_DAY.strip().lower()
    if expected and day != expected:
        raise ValueError(
            f"Anchor is stamped {SERVICE_DAY_COLUMN}='{day}' but EXPECTED_SERVICE_DAY "
            f"is '{EXPECTED_SERVICE_DAY}'. Point ANCHOR_PATH at the {EXPECTED_SERVICE_DAY} "
            "anchor, or change EXPECTED_SERVICE_DAY to match."
        )
    logging.info("Anchor service day verified: '%s'.", day)
    return day


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
        boardings = anchor[DEPENDENT_VAR]
        # Drop NaN-boardings months along with zeros: mean() would skip them for
        # boardings but still count them toward the revenue_hours/_miles averages,
        # breaking the "same months on both sides" consistency promised above.
        zero_rows = int((boardings == 0).sum())
        nan_rows = int(boardings.isna().sum())
        anchor = anchor[boardings.notna() & (boardings != 0)]
        dropped_routes = routes_before - anchor[ROUTE_KEY].nunique()
        logging.info(
            "Excluded %d zero- and %d missing-boarding month(s) as non-operating%s.",
            zero_rows,
            nan_rows,
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


def assemble_model_table() -> tuple[pd.DataFrame, list[tuple[str, str]], Optional[str]]:
    """Load the NTD anchor and left-join every prepped feature bundle onto it.

    The anchor holds the dependent variable plus the service-supplied predictors
    (revenue_hours / revenue_miles); a long/panel anchor is first collapsed to one
    row per route. Each bundle named in the manifest is then verified (SHA-256) and
    joined only if every one of its join keys is present in the anchor, so a
    route_id-only anchor silently skips a period bundle. Bundles are deduplicated
    on their keys so they can never fan out the anchor.

    Returns:
        ``(merged, provenance, service_day)`` where ``merged`` is the assembled
        route table, ``provenance`` is the ``(filename, sha256)`` of every bundle
        actually joined (recorded in the run log), and ``service_day`` is the
        anchor's verified day-type stamp (``None`` for an unstamped anchor).

    Raises:
        KeyError: If the anchor or a joined bundle lacks the join key(s).
        FileNotFoundError: If a manifest-listed bundle is missing.
        ValueError: If VERIFY_BUNDLE_HASHES is True and a bundle's hash mismatches,
            or a bundle's value columns collide with columns already on the table.
    """
    anchor = load_table(ANCHOR_PATH, ANCHOR_SHEET).rename(columns=COLUMN_ALIASES)
    if ROUTE_KEY not in anchor.columns:
        raise KeyError(f"Anchor is missing the join key '{ROUTE_KEY}'.")
    anchor[ROUTE_KEY] = _canonical_key(anchor[ROUTE_KEY])
    logging.info("Anchor '%s' loaded: %d rows, %d cols.", ANCHOR_PATH.name, *anchor.shape)

    service_day = _verify_service_day(anchor)
    # Once verified, the stamp column has done its job; drop it so the assembled
    # table stays purely numeric alongside the join key.
    anchor = anchor.drop(columns=[SERVICE_DAY_COLUMN], errors="ignore")

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
            "Joined bundle '%s' on %s: %d/%d routes matched (%d feature col(s)).",
            spec.filename,
            keys,
            matched,
            before,
            len(value_cols),
        )
        match_rate = matched / before if before else 1.0
        if match_rate < MIN_BUNDLE_MATCH_RATE:
            raise ValueError(
                f"Bundle '{spec.filename}' matched only {matched}/{before} anchor routes "
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
    logging.info("Assembled table: %d routes x %d columns.", *merged.shape)
    return merged, provenance, service_day


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
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Construct the response vector and design matrix; drop incomplete rows.

    Returns (y, X, term_names, model_frame). ``model_frame`` is the cleaned,
    kept-row frame (untransformed) so per-route outputs and overlays can be
    aligned back to it.
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
    missing_by_col = frame[used].isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    frame = frame.dropna(subset=used).reset_index(drop=True)
    dropped = before - len(frame)
    if dropped:
        # Name the offending columns: one sparse predictor (e.g. postsec enrollment)
        # can quietly cost a big chunk of the sample.
        logging.warning(
            "Dropped %d route(s) with missing model columns. Missing counts by column: %s.",
            dropped,
            {col: int(n) for col, n in missing_by_col.items()},
        )
    if frame.empty:
        raise ValueError("No complete rows remain after dropping missing values.")

    if LOG_DEPENDENT:
        nonpos = frame[DEPENDENT_VAR] <= 0
        n_nonpos = int(nonpos.sum())
        if n_nonpos and DROP_NONPOSITIVE_DEPENDENT:
            # On a Saturday/Sunday anchor these are routes that run no service on
            # that day type — non-operators, not underperformers — so they cannot
            # (and should not) be in the fit.
            dropped_ids = frame.loc[nonpos, ROUTE_KEY].astype(str).tolist()
            shown = ", ".join(dropped_ids[:10]) + (", …" if n_nonpos > 10 else "")
            logging.warning(
                "Dropping %d route(s) with non-positive '%s' before the log transform "
                "(no service on this day type, or a data gap): %s",
                n_nonpos,
                DEPENDENT_VAR,
                shown,
            )
            frame = frame[~nonpos].reset_index(drop=True)
            if frame.empty:
                raise ValueError("No routes remain after dropping non-positive dependents.")
        elif n_nonpos:
            raise ValueError(
                f"LOG_DEPENDENT is True but '{DEPENDENT_VAR}' has {n_nonpos} non-positive "
                "value(s). Set DROP_NONPOSITIVE_DEPENDENT = True to drop them instead."
            )

    y = frame[DEPENDENT_VAR].to_numpy(dtype=float)
    if LOG_DEPENDENT:
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
    return y, x_matrix, term_names, frame


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
    """Fit OLS with HC1/HC3/classical SEs plus VIF, Durbin-Watson, and exact LOO-R²."""
    n_obs, n_params = x_matrix.shape
    if n_obs <= n_params:
        raise ValueError(f"Need more routes ({n_obs}) than parameters ({n_params}).")
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
    # studentized per-route flags downstream.
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
        leverage=leverage,
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


def build_summary_frame(result: OLSResult, service_day: Optional[str]) -> pd.DataFrame:
    """Assemble the model-fit summary metrics into a tidy frame."""
    metrics = [
        ("dependent_variable", f"log({DEPENDENT_VAR})" if LOG_DEPENDENT else DEPENDENT_VAR),
        ("service_day", service_day or "(not stamped)"),
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
    if LOG_DEPENDENT:
        # Duan smearing: exp(fitted) alone estimates the conditional *median* and
        # systematically undershoots the boardings the sheet sits next to; the
        # mean(exp(residual)) factor rescales to a mean without changing the ranking.
        smear = float(np.mean(np.exp(result.residuals)))
        out["fitted_potential"] = np.exp(result.fitted) * smear
        logging.info("Duan smearing factor applied to fitted_potential: %.4f", smear)
    else:
        out["fitted_potential"] = result.fitted

    resid_suffix = "_log" if LOG_DEPENDENT else ""
    out[f"residual{resid_suffix}"] = result.residuals
    resid_sd = result.residuals.std(ddof=0)
    out["std_residual"] = result.residuals / resid_sd if resid_sd > 0 else result.residuals
    out[f"loo_residual{resid_suffix}"] = result.loo_residuals

    # Leverage-adjusted (studentized) residuals drive the over/under flag: a
    # high-leverage route drags the fit toward itself, so its raw residual
    # understates how far off expectation it really is.
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = result.residuals / (result.sigma * np.sqrt(1.0 - result.leverage))
        cooks_d = (studentized**2 / result.n_params) * (result.leverage / (1.0 - result.leverage))
    out["studentized_residual"] = studentized
    out["leverage"] = result.leverage
    out["cooks_d"] = cooks_d

    def _flag(z: float) -> str:
        if not np.isfinite(z):
            # Leverage ≈ 1: the model fits this route exactly (e.g. it is the only
            # member of a dummy level), so its residual carries no information.
            return "undetermined"
        if z >= PERF_FLAG_SD:
            return "over"
        if z <= -PERF_FLAG_SD:
            return "under"
        return "expected"

    out["performance"] = [_flag(z) for z in out["studentized_residual"]]

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

    return out.sort_values("studentized_residual").reset_index(drop=True)


def build_express_bench(express_frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Descriptive productivity bench for the express routes held out of the fit.

    Express routes are not scored against the local-demand model (their catchment is
    not what the buffer features measure), so instead of a residual they get a plain
    productivity read — weekday boardings per revenue hour — next to their raw
    fundamentals and service levers, sorted most- to least-productive. This is a
    ranking aid, not a model output: there is deliberately no "expected" boardings or
    residual column here, because a local-demand fit has no standing to call an express
    route over- or under-performing.
    """
    if express_frame is None or express_frame.empty:
        return None

    boardings_col = DEPENDENT_VAR
    rev_hours_col = "weekday_avg_revenue_hours"
    out = pd.DataFrame({ROUTE_KEY: express_frame[ROUTE_KEY].to_numpy()})
    boardings = pd.to_numeric(express_frame.get(boardings_col), errors="coerce")
    out[boardings_col] = boardings.to_numpy()

    if rev_hours_col in express_frame.columns:
        rev_hours = pd.to_numeric(express_frame[rev_hours_col], errors="coerce")
        out[rev_hours_col] = rev_hours.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["boardings_per_rev_hour"] = (boardings / rev_hours.replace(0, np.nan)).to_numpy()

    # Raw fundamentals + a few service levers, so the bench reads like RoutePerformance
    # without implying a fitted expectation. is_express is dropped (constant here).
    context_cols = [
        c
        for c in (*PREDICTORS, "span_hours", "median_headway_min", "avg_speed_mph", "trips_per_day")
        if c not in {boardings_col, rev_hours_col, "is_express"} and c in express_frame.columns
    ]
    for col in context_cols:
        out[col] = pd.to_numeric(express_frame[col], errors="coerce").to_numpy()

    sort_key = (
        "boardings_per_rev_hour" if "boardings_per_rev_hour" in out.columns else boardings_col
    )
    return out.sort_values(sort_key, ascending=False).reset_index(drop=True)


def build_supply_vs_demand(primary: OLSResult, demand: OLSResult) -> pd.DataFrame:
    """Side-by-side coefficient comparison of the primary fit vs. the supply-free fit.

    One row per non-intercept term, showing the coefficient / p-value / standardized
    coefficient with the supply term in (primary) and out (demand). The ``note`` column
    calls out the read leadership cares about: a term that was non-significant with
    revenue_hours in but crosses p<0.05 once it is removed was MASKED by supply, i.e. a
    real demand driver the primary model could not surface. The supply term itself shows
    only its primary column (it is absent from the demand model by construction).
    """

    def as_map(res: OLSResult, attr: str) -> dict[str, float]:
        return dict(zip(res.term_names, getattr(res, attr)))

    p_coef, p_p, p_sc = (
        as_map(primary, "params"),
        as_map(primary, "p_values"),
        as_map(primary, "std_coef"),
    )
    d_coef, d_p, d_sc = (
        as_map(demand, "params"),
        as_map(demand, "p_values"),
        as_map(demand, "std_coef"),
    )
    removed = set(primary.term_names) - set(demand.term_names)

    rows: list[dict[str, object]] = []
    for term in primary.term_names:
        if term == "intercept":
            continue
        wp, np_p = p_p.get(term), d_p.get(term)
        if term in removed:
            note = "supply term — removed in demand model"
        elif wp is not None and np_p is not None and wp >= 0.05 and np_p < 0.05:
            note = "SURFACED (<0.05 without supply)"
        elif wp is not None and np_p is not None and np_p < wp:
            note = "stronger without supply"
        else:
            note = ""
        rows.append(
            {
                "term": term,
                "coef_with_supply": p_coef.get(term),
                "p_with_supply": wp,
                "std_coef_with_supply": p_sc.get(term),
                "coef_no_supply": d_coef.get(term, np.nan),
                "p_no_supply": d_p.get(term, np.nan),
                "std_coef_no_supply": d_sc.get(term, np.nan),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def export_results(
    result: OLSResult,
    model_frame: pd.DataFrame,
    service_day: Optional[str] = None,
    express_frame: Optional[pd.DataFrame] = None,
    demand_result: Optional[OLSResult] = None,
) -> Path:
    """Write the results workbook, with ExpressBench / demand-diagnostic sheets when applicable."""
    workbook = OUTPUT_DIR / "route_performance_results.xlsx"
    summary = build_summary_frame(result, service_day)
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

    # (2) Full correlation matrix restricted to the MODELED variables (DV + the
    # regressors) — a focused collinearity companion to the VIF column. Raw, untransformed
    # variables; VIF in Coefficients reflects the logged design matrix.
    modeled = [c for c in [DEPENDENT_VAR, *PREDICTORS] if c in numeric.columns]
    collinearity = numeric[modeled].corr().reset_index().rename(columns={"index": "variable"})

    express_bench = build_express_bench(express_frame)

    with pd.ExcelWriter(workbook) as writer:
        summary.to_excel(writer, sheet_name="ModelSummary", index=False)
        coef.to_excel(writer, sheet_name="Coefficients", index=False)
        performance.to_excel(writer, sheet_name="RoutePerformance", index=False)
        corr_with_target.to_excel(writer, sheet_name="Correlations", index=False)
        collinearity.to_excel(writer, sheet_name="CollinearityMatrix", index=False)
        if express_bench is not None and not express_bench.empty:
            express_bench.to_excel(writer, sheet_name="ExpressBench", index=False)
            logging.info(
                "ExpressBench sheet written for %d held-out express route(s).", len(express_bench)
            )
        if demand_result is not None:
            build_coefficient_frame(demand_result).to_excel(
                writer, sheet_name="DemandModelCoef", index=False
            )
            build_supply_vs_demand(result, demand_result).to_excel(
                writer, sheet_name="SupplyVsDemand", index=False
            )
            logging.info("Demand-diagnostic sheets written (DemandModelCoef, SupplyVsDemand).")

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


def write_run_log(
    result: OLSResult, provenance: list[tuple[str, str]], service_day: Optional[str] = None
) -> None:
    """Write the run-log sidecar (timestamp, fit headline, bundle provenance, config)."""
    log_path = OUTPUT_DIR / "route_performance_model_runlog.txt"
    bundle_lines = (
        [f"  {name}  sha256={sha}" for name, sha in provenance] if provenance else ["  (none)"]
    )
    # Hash the anchor too, so the run log is a complete provenance record — the
    # bundles are already hashed via the manifest, but the anchor is the input
    # that matters most.
    anchor_sha = _sha256_file(ANCHOR_PATH) if ANCHOR_PATH.exists() else "(file not found)"
    lines = [
        "=" * 72,
        "CROSS-SECTIONAL ROUTE RIDERSHIP MODEL RUN LOG (ENGINE 1, secured box)",
        "=" * 72,
        f"Run timestamp:    {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Anchor:           {ANCHOR_PATH}",
        f"Anchor SHA-256:   {anchor_sha}",
        f"Service day:      {service_day or '(not stamped)'}",
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

    if EXCLUDE_EXPRESS_FROM_FIT and "is_express" in PREDICTORS:
        raise ValueError(
            "EXCLUDE_EXPRESS_FROM_FIT is True but 'is_express' is still in PREDICTORS. "
            "With express routes held out of the fit, is_express is constant (all 0) and "
            "would make the design matrix singular. Remove 'is_express' from PREDICTORS, "
            "or set EXCLUDE_EXPRESS_FROM_FIT = False to fit the pooled model with a dummy."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table, provenance, service_day = assemble_model_table()
    table = derive_features(table)
    if DEPENDENT_VAR not in table.columns:
        raise KeyError(f"Dependent variable '{DEPENDENT_VAR}' not in the assembled table.")

    # Express routes run on a different demand construct than the buffer-based features
    # measure, so they are held out of the fit and benchmarked descriptively instead
    # (see EXCLUDE_EXPRESS_FROM_FIT). Splitting here keeps the design matrix, the
    # residuals, and every correlation on the modeled (local) routes only.
    express_frame: Optional[pd.DataFrame] = None
    if EXCLUDE_EXPRESS_FROM_FIT:
        is_exp = table["is_express"] == 1
        express_frame = table[is_exp].copy()
        table = table[~is_exp].copy()
        logging.info(
            "Held %d express route(s) out of the fit; %d local route(s) remain to model.",
            len(express_frame),
            len(table),
        )

    y, x_matrix, term_names, model_frame = build_design_matrix(table)
    result = fit_ols(y, x_matrix, term_names, SE_TYPE)
    log_report(result)

    # Supply-free companion fit: same response, same local rows, same transforms, with
    # the supply column dropped from the design so a demand feature masked by
    # revenue_hours can show itself. Slicing the already-built matrix (rather than
    # rebuilding) guarantees the two models sit on an identical route set.
    demand_result: Optional[OLSResult] = None
    if FIT_DEMAND_DIAGNOSTIC:
        supply_term = (
            f"log_{SUPPLY_PREDICTOR}" if SUPPLY_PREDICTOR in LOG_PREDICTORS else SUPPLY_PREDICTOR
        )
        if supply_term in term_names:
            keep = [i for i, name in enumerate(term_names) if name != supply_term]
            demand_result = fit_ols(y, x_matrix[:, keep], [term_names[i] for i in keep], SE_TYPE)
            logging.info(
                "Demand diagnostic (supply term '%s' removed): adjR2=%.4f LOO=%.4f — expected "
                "LOWER than the primary fit; this is a demand model, read its coefficients not "
                "its residuals.",
                supply_term,
                demand_result.adj_r_squared,
                demand_result.loo_r_squared,
            )
        else:
            logging.warning(
                "FIT_DEMAND_DIAGNOSTIC is set but supply term '%s' is not in the model; "
                "skipping the demand companion.",
                supply_term,
            )

    export_results(result, model_frame, service_day, express_frame, demand_result)
    if MAKE_PLOTS:
        make_diagnostic_plots(result)
    write_run_log(result, provenance, service_day)

    logging.info("Done.")
    return result


def main() -> int:
    """Run the model as a script and translate the outcome into an exit code.

    Thin shell wrapper around :func:`run` — notebook users should keep calling
    ``run()`` directly to get the fitted :class:`OLSResult` back.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 if required
        CONFIGURATION values are still placeholders.
    """
    try:
        result = run()
    except (OSError, KeyError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    # run() returns None only when the '# <<< EDIT ME' placeholder guard fired.
    return 2 if result is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
