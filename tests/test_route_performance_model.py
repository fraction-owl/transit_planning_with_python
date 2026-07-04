import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.modeling import route_performance_model as rpm


@pytest.fixture
def design() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A small, well-conditioned design matrix with a known linear signal."""
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(10.0, 2.0, n)
    x2 = rng.normal(5.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    # True model: y = 2 + 3*x1 - 1.5*x2 + noise
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + noise
    x_matrix = np.column_stack([np.ones(n), x1, x2])
    return y, x_matrix, ["intercept", "x1", "x2"]


def test_fit_ols_recovers_coefficients(design) -> None:
    """OLS should recover the known coefficients and report a high R^2."""
    y, x_matrix, names = design
    result = rpm.fit_ols(y, x_matrix, names, se_type="classical")

    assert result.n_obs == 80
    assert result.n_params == 3
    np.testing.assert_allclose(result.params[1:], [3.0, -1.5], atol=0.2)
    assert result.r_squared > 0.98
    assert result.p_values[1] < 0.01
    assert result.p_values[2] < 0.01


def test_fit_ols_requires_enough_observations() -> None:
    """Fewer observations than parameters must raise."""
    x_matrix = np.ones((2, 3))
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        rpm.fit_ols(y, x_matrix, ["intercept", "a", "b"], se_type="classical")


def test_fit_ols_rejects_unknown_se_type(design) -> None:
    """An unrecognised SE type must raise rather than silently fall through."""
    y, x_matrix, names = design
    with pytest.raises(ValueError):
        rpm.fit_ols(y, x_matrix, names, se_type="bootstrap")


def test_hc1_matches_classical_under_homoskedasticity(design) -> None:
    """HC1 and classical SEs should be close when errors are homoskedastic."""
    y, x_matrix, names = design
    classical = rpm.fit_ols(y, x_matrix, names, se_type="classical")
    robust = rpm.fit_ols(y, x_matrix, names, se_type="HC1")

    np.testing.assert_allclose(classical.params, robust.params)
    np.testing.assert_allclose(classical.std_errors, robust.std_errors, rtol=0.5)


def test_loo_residuals_match_bruteforce_refits(design) -> None:
    """The hat-matrix LOO residuals must equal explicit leave-one-out refits."""
    y, x_matrix, names = design
    result = rpm.fit_ols(y, x_matrix, names, se_type="classical")

    n = len(y)
    brute = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        beta, _, _, _ = np.linalg.lstsq(x_matrix[keep], y[keep], rcond=None)
        brute[i] = y[i] - x_matrix[i] @ beta

    np.testing.assert_allclose(result.loo_residuals, brute, rtol=1e-7, atol=1e-9)


def test_hc3_close_to_hc1_under_homoskedasticity(design) -> None:
    """HC3 must be accepted, leave the fit unchanged, and stay near HC1's SEs."""
    y, x_matrix, names = design
    hc1 = rpm.fit_ols(y, x_matrix, names, se_type="HC1")
    hc3 = rpm.fit_ols(y, x_matrix, names, se_type="HC3")

    np.testing.assert_allclose(hc1.params, hc3.params)
    assert np.all(np.isfinite(hc3.std_errors))
    np.testing.assert_allclose(hc3.std_errors, hc1.std_errors, rtol=0.5)


def test_leverage_matches_hat_matrix_diagonal(design) -> None:
    """The stored leverage must equal the hat-matrix diagonal."""
    y, x_matrix, names = design
    result = rpm.fit_ols(y, x_matrix, names, se_type="classical")
    hat = x_matrix @ np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T
    np.testing.assert_allclose(result.leverage, np.diag(hat), atol=1e-10)


def test_vif_flags_collinearity() -> None:
    """A duplicated predictor should yield a very large VIF."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=50)
    x2 = x1 + rng.normal(scale=1e-6, size=50)  # nearly identical to x1
    x_matrix = np.column_stack([np.ones(50), x1, x2])
    vif = rpm._vif(x_matrix, ["intercept", "x1", "x2"])
    assert vif["x1"] > 100
    assert vif["x2"] > 100


def test_canonical_key_normalizes_join_values() -> None:
    """Keys are trimmed, case/space-folded, de-floated, and NaN-filled."""
    out = rpm._canonical_key(pd.Series([" 159 ", "159.0", 42, None, "rt 5", "Rt 5"]))
    assert out.tolist() == ["159", "159", "42", "", "RT5", "RT5"]


def test_derive_features_flags_express_and_equity_pct() -> None:
    """Express routes get a 1/0 flag and equity %s divide count by denominator."""
    table = pd.DataFrame(
        {
            "route_id": ["101", "100"],  # 101 is in EXPRESS_ROUTES, 100 is not
            "low_income": [250.0, 100.0],
            "total_pop": [1000.0, 500.0],
        }
    )
    out = rpm.derive_features(table)

    assert out["is_express"].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(out["pct_low_income"], [0.25, 0.20])


def test_end_to_end_export(tmp_path, monkeypatch) -> None:
    """A full design-matrix build, fit, and export writes the expected sheets."""
    monkeypatch.setattr(rpm, "PREDICTORS", ("revenue_hours", "total_pop"))
    monkeypatch.setattr(rpm, "LOG_PREDICTORS", ("revenue_hours", "total_pop"))
    monkeypatch.setattr(rpm, "OUTPUT_DIR", tmp_path)

    rng = np.random.default_rng(7)
    n = 60
    hours = rng.uniform(50, 500, n)
    pop = rng.uniform(1000, 50000, n)
    boardings = np.exp(1.0 + 0.6 * np.log(hours) + 0.3 * np.log(pop) + rng.normal(0, 0.1, n))
    table = pd.DataFrame(
        {
            "route_id": [str(i) for i in range(n)],
            "ntd_boardings": boardings,
            "revenue_hours": hours,
            "total_pop": pop,
        }
    )

    y, x_matrix, names, frame = rpm.build_design_matrix(table)
    assert names == ["intercept", "log_revenue_hours", "log_total_pop"]

    result = rpm.fit_ols(y, x_matrix, names, se_type="HC1")
    workbook = rpm.export_results(result, frame)

    assert workbook.exists()
    sheets = pd.read_excel(workbook, sheet_name=None)
    assert set(sheets) == {
        "ModelSummary",
        "Coefficients",
        "RoutePerformance",
        "Correlations",
        "CollinearityMatrix",
    }
    # Recover the data-generating elasticities; the signal is strong and clean.
    assert result.r_squared > 0.8

    # The over/under flag is driven by the leverage-adjusted studentized residual,
    # and the influence diagnostics ride along in the sheet.
    perf = sheets["RoutePerformance"]
    assert {"studentized_residual", "leverage", "cooks_d"} <= set(perf.columns)
    finite = perf["studentized_residual"].notna()
    over = perf.loc[finite, "studentized_residual"] >= rpm.PERF_FLAG_SD
    under = perf.loc[finite, "studentized_residual"] <= -rpm.PERF_FLAG_SD
    assert (over == (perf.loc[finite, "performance"] == "over")).all()
    assert (under == (perf.loc[finite, "performance"] == "under")).all()

    # Duan smearing puts fitted_potential on the mean (not median) boardings
    # scale, so its mean tracks the actual mean closely on clean data.
    ratio = perf["fitted_potential"].mean() / perf["ntd_boardings"].mean()
    assert 0.9 < ratio < 1.1


def _write_bundle_manifest(bundle_dir, entries) -> None:
    """Write a prep_features_public-style manifest, hashing each bundle from disk.

    ``entries`` is a list of ``(filename, join_keys)``; SHA-256 is computed with
    the same helper the model uses so VERIFY_BUNDLE_HASHES passes.
    """
    bundles = []
    for filename, join_keys in entries:
        path = bundle_dir / filename
        n_rows = max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
        bundles.append(
            {
                "filename": filename,
                "join_keys": list(join_keys),
                "n_rows": n_rows,
                "n_cols": 0,
                "sha256": rpm._sha256_file(path),
            }
        )
    (bundle_dir / "manifest.json").write_text(json.dumps({"bundles": bundles}), encoding="utf-8")


def _make_anchor_and_bundles(tmp_path) -> tuple[Path, Path, Path]:
    """Write a route-level anchor, a route_id bundle, and a period bundle to disk."""
    anchor_path = tmp_path / "anchor.csv"
    pd.DataFrame(
        {
            "route_id": ["101", "102", "103"],
            "ntd_boardings": [1000.0, 2000.0, 3000.0],
            # Supply lives on the NTD anchor (more accurate than the GTFS copy).
            "revenue_hours": [50.0, 80.0, 120.0],
            "revenue_miles": [500.0, 800.0, 1200.0],
        }
    ).to_csv(anchor_path, index=False)

    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    pd.DataFrame(
        {
            "route_id": ["101", "102", "103"],
            "total_pop": [10000, 20000, 30000],
            "shared_stop_share": [0.2, 0.3, 0.4],
            "competition_intensity": [0.5, 0.6, 0.7],
            # A shapefile-derived name the model aliases back to a clean column.
            "Metrorail_Stations.shp": [1, 0, 2],
        }
    ).to_csv(bundle_dir / "features__route_id.csv", index=False)
    pd.DataFrame({"period": ["2024-01", "2024-02"], "gas_price": [3.1, 3.2]}).to_csv(
        bundle_dir / "features__period.csv", index=False
    )

    _write_bundle_manifest(
        bundle_dir,
        [("features__route_id.csv", ["route_id"]), ("features__period.csv", ["period"])],
    )
    return anchor_path, bundle_dir, bundle_dir / "manifest.json"


def test_assemble_model_table_joins_route_bundle_and_skips_period(tmp_path, monkeypatch) -> None:
    """The route_id bundle joins; the period bundle is skipped (anchor has no period)."""
    anchor_path, bundle_dir, manifest_path = _make_anchor_and_bundles(tmp_path)
    monkeypatch.setattr(rpm, "ANCHOR_PATH", anchor_path)
    monkeypatch.setattr(rpm, "BUNDLE_DIR", bundle_dir)
    monkeypatch.setattr(rpm, "MANIFEST_PATH", manifest_path)

    merged, provenance, service_day = rpm.assemble_model_table()

    # The fixture anchor carries no service_day stamp: run proceeds, day unknown.
    assert service_day is None
    assert len(merged) == 3
    # Feature columns from the route_id bundle are joined on.
    assert {"total_pop", "shared_stop_share", "competition_intensity"} <= set(merged.columns)
    # Supply columns stay on the anchor (sourced from NTD, never from a bundle).
    assert {"revenue_hours", "revenue_miles"} <= set(merged.columns)
    # The COLUMN_ALIASES rename is applied to bundle columns too.
    assert "Metrorail_Stations" in merged.columns and "Metrorail_Stations.shp" not in merged.columns
    # The period bundle was skipped, so none of its columns leak in.
    assert "gas_price" not in merged.columns
    # Only the joined bundle is recorded in provenance.
    assert [name for name, _ in provenance] == ["features__route_id.csv"]


def test_assemble_model_table_rejects_column_collision(tmp_path, monkeypatch) -> None:
    """A bundle re-shipping a column that already exists on the table must fail loudly."""
    anchor_path, bundle_dir, manifest_path = _make_anchor_and_bundles(tmp_path)
    # A second route_id bundle that re-ships total_pop (already joined by the first);
    # a silent merge would _x/_y-rename both copies and break downstream lookups.
    pd.DataFrame({"route_id": ["101", "102", "103"], "total_pop": [1, 2, 3]}).to_csv(
        bundle_dir / "clash__route_id.csv", index=False
    )
    _write_bundle_manifest(
        bundle_dir,
        [
            ("features__route_id.csv", ["route_id"]),
            ("clash__route_id.csv", ["route_id"]),
            ("features__period.csv", ["period"]),
        ],
    )
    monkeypatch.setattr(rpm, "ANCHOR_PATH", anchor_path)
    monkeypatch.setattr(rpm, "BUNDLE_DIR", bundle_dir)
    monkeypatch.setattr(rpm, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="total_pop"):
        rpm.assemble_model_table()


def test_assemble_model_table_enforces_match_rate_floor(tmp_path, monkeypatch) -> None:
    """A bundle matching too few anchor routes aborts instead of modeling a subset."""
    anchor_path, bundle_dir, manifest_path = _make_anchor_and_bundles(tmp_path)
    # Break one of the three route keys so the bundle matches only 2/3 (67%),
    # under the 90% default floor.
    bundle_file = bundle_dir / "features__route_id.csv"
    df = pd.read_csv(bundle_file)
    df["route_id"] = df["route_id"].astype(str)
    df.loc[df["route_id"] == "103", "route_id"] = "999"
    df.to_csv(bundle_file, index=False)
    _write_bundle_manifest(
        bundle_dir,
        [("features__route_id.csv", ["route_id"]), ("features__period.csv", ["period"])],
    )
    monkeypatch.setattr(rpm, "ANCHOR_PATH", anchor_path)
    monkeypatch.setattr(rpm, "BUNDLE_DIR", bundle_dir)
    monkeypatch.setattr(rpm, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="MIN_BUNDLE_MATCH_RATE"):
        rpm.assemble_model_table()

    # Disabling the floor lets a legitimately sparse bundle through.
    monkeypatch.setattr(rpm, "MIN_BUNDLE_MATCH_RATE", 0.0)
    merged, provenance, _ = rpm.assemble_model_table()
    assert len(merged) == 3
    assert [name for name, _ in provenance] == ["features__route_id.csv"]


def test_assemble_model_table_joins_case_and_space_variant_keys(tmp_path, monkeypatch) -> None:
    """An anchor keyed 'RT 5' joins a bundle keyed 'rt5' after canonicalization."""
    anchor_path, bundle_dir, manifest_path = _make_anchor_and_bundles(tmp_path)
    anchor = pd.read_csv(anchor_path)
    anchor["route_id"] = ["RT 5", "Rt-6", "route7"]
    anchor.to_csv(anchor_path, index=False)
    bundle_file = bundle_dir / "features__route_id.csv"
    bundle = pd.read_csv(bundle_file)
    bundle["route_id"] = ["rt5", "RT-6", "Route7"]
    bundle.to_csv(bundle_file, index=False)
    _write_bundle_manifest(
        bundle_dir,
        [("features__route_id.csv", ["route_id"]), ("features__period.csv", ["period"])],
    )
    monkeypatch.setattr(rpm, "ANCHOR_PATH", anchor_path)
    monkeypatch.setattr(rpm, "BUNDLE_DIR", bundle_dir)
    monkeypatch.setattr(rpm, "MANIFEST_PATH", manifest_path)

    merged, _, _ = rpm.assemble_model_table()
    assert merged["total_pop"].notna().all(), "case/space variants failed to join"


def test_assemble_model_table_rejects_hash_mismatch(tmp_path, monkeypatch) -> None:
    """A bundle edited after the manifest was written aborts the run when verifying."""
    anchor_path, bundle_dir, manifest_path = _make_anchor_and_bundles(tmp_path)
    # Corrupt the route bundle so its on-disk hash no longer matches the manifest.
    (bundle_dir / "features__route_id.csv").write_text(
        "route_id,total_pop\n101,999999\n", encoding="utf-8"
    )
    monkeypatch.setattr(rpm, "ANCHOR_PATH", anchor_path)
    monkeypatch.setattr(rpm, "BUNDLE_DIR", bundle_dir)
    monkeypatch.setattr(rpm, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rpm, "VERIFY_BUNDLE_HASHES", True)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rpm.assemble_model_table()


def test_collapse_panel_anchor_rolls_up_to_one_row_per_route(monkeypatch) -> None:
    """A route x period anchor collapses to one row per route via ANCHOR_AGG."""
    monkeypatch.setattr(rpm, "ANCHOR_AGG", "mean")
    monkeypatch.setattr(rpm, "ANCHOR_EXCLUDE_ZERO_MONTHS", True)
    panel = pd.DataFrame(
        {
            "route_id": rpm._canonical_key(pd.Series(["101", "101", "102", "102"])),
            "ntd_boardings": [100.0, 0.0, 200.0, 400.0],  # the zero month is dropped
            "revenue_hours": [10.0, 99.0, 20.0, 40.0],
        }
    )
    out = rpm._collapse_panel_anchor(panel)

    assert sorted(out["route_id"]) == ["101", "102"]
    # Route 101's zero-boarding month is excluded, so its mean is 100, not 50.
    r101 = out.loc[out["route_id"] == "101"].iloc[0]
    assert r101["ntd_boardings"] == 100.0
    assert r101["revenue_hours"] == 10.0


def test_collapse_panel_anchor_drops_nan_months_with_zeros(monkeypatch) -> None:
    """NaN-boardings months are excluded so revenue averages use the same months."""
    monkeypatch.setattr(rpm, "ANCHOR_AGG", "mean")
    monkeypatch.setattr(rpm, "ANCHOR_EXCLUDE_ZERO_MONTHS", True)
    panel = pd.DataFrame(
        {
            "route_id": rpm._canonical_key(pd.Series(["101", "101", "101", "102", "102"])),
            "ntd_boardings": [100.0, np.nan, 0.0, 200.0, 400.0],
            "revenue_hours": [10.0, 77.0, 99.0, 20.0, 40.0],
        }
    )
    out = rpm._collapse_panel_anchor(panel)

    r101 = out.loc[out["route_id"] == "101"].iloc[0]
    assert r101["ntd_boardings"] == 100.0
    # Neither the NaN month's 77 hours nor the zero month's 99 leak into the mean:
    # mean() alone would skip the NaN for boardings but still count its hours.
    assert r101["revenue_hours"] == 10.0


def test_log_dependent_rejects_nonpositive(monkeypatch) -> None:
    """A non-positive dependent under the log transform must raise when not dropping."""
    monkeypatch.setattr(rpm, "PREDICTORS", ("revenue_hours",))
    monkeypatch.setattr(rpm, "LOG_PREDICTORS", ("revenue_hours",))
    monkeypatch.setattr(rpm, "DROP_NONPOSITIVE_DEPENDENT", False)
    table = pd.DataFrame(
        {
            "route_id": ["1", "2", "3"],
            "ntd_boardings": [100.0, 0.0, 300.0],  # a zero is invalid under log
            "revenue_hours": [10.0, 20.0, 30.0],
        }
    )
    with pytest.raises(ValueError):
        rpm.build_design_matrix(table)


def test_log_dependent_drops_nonpositive_when_enabled(monkeypatch) -> None:
    """Non-operating routes (zero boardings on this day type) are dropped, not fatal."""
    monkeypatch.setattr(rpm, "PREDICTORS", ("revenue_hours",))
    monkeypatch.setattr(rpm, "LOG_PREDICTORS", ("revenue_hours",))
    monkeypatch.setattr(rpm, "DROP_NONPOSITIVE_DEPENDENT", True)
    table = pd.DataFrame(
        {
            "route_id": ["1", "2", "3"],
            "ntd_boardings": [100.0, 0.0, 300.0],  # e.g. route 2 runs no Saturday service
            "revenue_hours": [10.0, 20.0, 30.0],
        }
    )
    y, x_matrix, _, frame = rpm.build_design_matrix(table)

    assert list(frame["route_id"]) == ["1", "3"]
    assert len(y) == 2 and x_matrix.shape[0] == 2
    np.testing.assert_allclose(y, np.log([100.0, 300.0]))


def _stamped_anchor(service_days: list[str]) -> pd.DataFrame:
    """A minimal anchor frame carrying a service_day stamp column."""
    n = len(service_days)
    return pd.DataFrame(
        {
            "route_id": [str(100 + i) for i in range(n)],
            "service_day": service_days,
            "ntd_boardings": [1000.0] * n,
            "revenue_hours": [50.0] * n,
        }
    )


def test_verify_service_day_accepts_matching_stamp(monkeypatch) -> None:
    """A uniformly stamped anchor matching EXPECTED_SERVICE_DAY verifies cleanly."""
    monkeypatch.setattr(rpm, "EXPECTED_SERVICE_DAY", "saturday")
    assert rpm._verify_service_day(_stamped_anchor(["saturday", "Saturday "])) == "saturday"


def test_verify_service_day_rejects_mismatched_stamp(monkeypatch) -> None:
    """A combined anchor fed to a weekday-only run must abort."""
    monkeypatch.setattr(rpm, "EXPECTED_SERVICE_DAY", "weekday")
    with pytest.raises(ValueError, match="combined"):
        rpm._verify_service_day(_stamped_anchor(["combined", "combined"]))


def test_verify_service_day_rejects_mixed_stamps(monkeypatch) -> None:
    """An anchor mixing day types can never be modeled in one run."""
    monkeypatch.setattr(rpm, "EXPECTED_SERVICE_DAY", "")
    with pytest.raises(ValueError, match="mixes"):
        rpm._verify_service_day(_stamped_anchor(["weekday", "saturday"]))


def test_verify_service_day_unstamped_anchor_warns_and_passes(monkeypatch) -> None:
    """A pre-stamp anchor (no service_day column) runs with a warning."""
    monkeypatch.setattr(rpm, "EXPECTED_SERVICE_DAY", "weekday")
    anchor = _stamped_anchor(["weekday"]).drop(columns=["service_day"])
    assert rpm._verify_service_day(anchor) is None
