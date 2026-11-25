"""Tests for scripts/modeling/monthly_ridership_model.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.modeling import monthly_ridership_model as mrm

# ---------------------------------------------------------------------------
# OLS engine
# ---------------------------------------------------------------------------


@pytest.fixture
def design() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A small, well-conditioned design matrix with a known linear signal."""
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(10.0, 2.0, n)
    x2 = rng.normal(5.0, 1.0, n)
    # True model: y = 2 + 3*x1 - 1.5*x2 + noise
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + rng.normal(0.0, 0.5, n)
    x_matrix = np.column_stack([np.ones(n), x1, x2])
    return y, x_matrix, ["intercept", "x1", "x2"]


def test_fit_ols_recovers_coefficients(design) -> None:
    y, x_matrix, names = design
    result = mrm.fit_ols(y, x_matrix, names, se_type="classical")
    np.testing.assert_allclose(result.params[1:], [3.0, -1.5], atol=0.2)
    assert result.r_squared > 0.98
    assert result.loo_r_squared > 0.95  # strong signal survives leave-one-out


def test_fit_ols_rejects_unknown_se_type(design) -> None:
    y, x_matrix, names = design
    with pytest.raises(ValueError):
        mrm.fit_ols(y, x_matrix, names, se_type="bootstrap")


def test_hc3_close_to_hc1_under_homoskedasticity(design) -> None:
    """HC3 must be accepted, leave the fit unchanged, and stay near HC1's SEs."""
    y, x_matrix, names = design
    hc1 = mrm.fit_ols(y, x_matrix, names, se_type="HC1")
    hc3 = mrm.fit_ols(y, x_matrix, names, se_type="HC3")

    np.testing.assert_allclose(hc1.params, hc3.params)
    assert np.all(np.isfinite(hc3.std_errors))
    np.testing.assert_allclose(hc3.std_errors, hc1.std_errors, rtol=0.5)


def test_leverage_matches_hat_matrix_diagonal(design) -> None:
    y, x_matrix, names = design
    result = mrm.fit_ols(y, x_matrix, names, se_type="classical")
    hat = x_matrix @ np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T
    np.testing.assert_allclose(result.leverage, np.diag(hat), atol=1e-10)


def test_loo_residuals_match_bruteforce_refits(design) -> None:
    """The hat-matrix LOO residuals must equal explicit leave-one-out refits."""
    y, x_matrix, names = design
    result = mrm.fit_ols(y, x_matrix, names, se_type="classical")

    n = len(y)
    brute = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        beta, _, _, _ = np.linalg.lstsq(x_matrix[keep], y[keep], rcond=None)
        brute[i] = y[i] - x_matrix[i] @ beta

    np.testing.assert_allclose(result.loo_residuals, brute, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# Assembly fixtures
# ---------------------------------------------------------------------------


def _write_manifest(bundle_dir: Path, entries: list[tuple[str, list[str]]]) -> Path:
    """Write a Part A-style manifest, hashing each bundle from disk."""
    bundles = []
    for filename, join_keys in entries:
        path = bundle_dir / filename
        n_rows = max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
        bundles.append(
            {
                "filename": filename,
                "join_keys": join_keys,
                "n_rows": n_rows,
                "n_cols": 0,
                "sha256": mrm._sha256_file(path),
            }
        )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"bundles": bundles}), encoding="utf-8")
    return manifest_path


def _make_panel_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A route x period anchor plus a route_id bundle and a period bundle."""
    anchor_path = tmp_path / "anchor.csv"
    pd.DataFrame(
        {
            "route_id": ["101", "101", "202", "202"],
            "period": ["2024-01", "2024-02", "2024-01", "2024-02"],
            "ntd_boardings": [1000.0, 1100.0, 2000.0, 2100.0],
            "scheduled_hours": [50.0, 52.0, 80.0, 81.0],
        }
    ).to_csv(anchor_path, index=False)

    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    pd.DataFrame({"route_id": ["101", "202"], "pop_served": [10000, 20000]}).to_csv(
        bundle_dir / "features__route_id.csv", index=False
    )
    pd.DataFrame({"period": ["2024-01", "2024-02"], "gas_price": [3.1, 3.2]}).to_csv(
        bundle_dir / "features__period.csv", index=False
    )
    manifest_path = _write_manifest(
        bundle_dir,
        [("features__route_id.csv", ["route_id"]), ("features__period.csv", ["period"])],
    )
    return anchor_path, bundle_dir, manifest_path


# ---------------------------------------------------------------------------
# Assembly behavior
# ---------------------------------------------------------------------------


def test_assemble_panel_anchor_joins_route_and_period_bundles(tmp_path: Path) -> None:
    """A panel anchor (route_id + period) joins both bundle keyings."""
    anchor_path, bundle_dir, manifest_path = _make_panel_inputs(tmp_path)
    merged, provenance = mrm.assemble_model_table(
        anchor_path, 0, bundle_dir, manifest_path, verify_hashes=True
    )

    assert len(merged) == 4
    assert merged["pop_served"].notna().all()
    assert merged["gas_price"].notna().all()
    assert [name for name, _ in provenance] == [
        "features__route_id.csv",
        "features__period.csv",
    ]


def test_assemble_cross_sectional_anchor_skips_period_bundle(tmp_path: Path) -> None:
    """An anchor without 'period' silently skips the period-keyed bundle."""
    anchor_path, bundle_dir, manifest_path = _make_panel_inputs(tmp_path)
    pd.read_csv(anchor_path).drop(columns="period").drop_duplicates("route_id").to_csv(
        anchor_path, index=False
    )
    _write_manifest(
        bundle_dir,
        [("features__route_id.csv", ["route_id"]), ("features__period.csv", ["period"])],
    )
    merged, provenance = mrm.assemble_model_table(
        anchor_path, 0, bundle_dir, manifest_path, verify_hashes=True
    )
    assert "gas_price" not in merged.columns
    assert [name for name, _ in provenance] == ["features__route_id.csv"]


def test_assemble_rejects_column_collision(tmp_path: Path) -> None:
    """A bundle re-shipping an existing column must fail loudly, not _x/_y-rename."""
    anchor_path, bundle_dir, manifest_path = _make_panel_inputs(tmp_path)
    pd.DataFrame({"route_id": ["101", "202"], "pop_served": [1, 2]}).to_csv(
        bundle_dir / "clash__route_id.csv", index=False
    )
    _write_manifest(
        bundle_dir,
        [
            ("features__route_id.csv", ["route_id"]),
            ("clash__route_id.csv", ["route_id"]),
            ("features__period.csv", ["period"]),
        ],
    )
    with pytest.raises(ValueError, match="pop_served"):
        mrm.assemble_model_table(anchor_path, 0, bundle_dir, manifest_path, verify_hashes=True)


def test_assemble_warns_on_duplicate_bundle_keys(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Duplicate keys inside a bundle are logged (upstream prep signal), first kept."""
    anchor_path, bundle_dir, manifest_path = _make_panel_inputs(tmp_path)
    dup = pd.DataFrame({"route_id": ["101", "202", "202"], "pop_served": [1, 2, 3]})
    dup.to_csv(bundle_dir / "features__route_id.csv", index=False)
    _write_manifest(
        bundle_dir,
        [("features__route_id.csv", ["route_id"]), ("features__period.csv", ["period"])],
    )
    with caplog.at_level(logging.WARNING):
        merged, _ = mrm.assemble_model_table(
            anchor_path, 0, bundle_dir, manifest_path, verify_hashes=True
        )
    assert "duplicate row(s)" in caplog.text
    # First occurrence wins: route 202 keeps pop_served == 2.
    assert set(merged.loc[merged["route_id"] == "202", "pop_served"]) == {2}


def test_assemble_rejects_hash_mismatch(tmp_path: Path) -> None:
    anchor_path, bundle_dir, manifest_path = _make_panel_inputs(tmp_path)
    (bundle_dir / "features__route_id.csv").write_text(
        "route_id,pop_served\n101,999\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        mrm.assemble_model_table(anchor_path, 0, bundle_dir, manifest_path, verify_hashes=True)


def test_assemble_enforces_match_rate_floor(tmp_path: Path, monkeypatch) -> None:
    """A bundle matching too few anchor rows aborts instead of modeling a subset."""
    anchor_path, bundle_dir, manifest_path = _make_panel_inputs(tmp_path)
    # Only route 101 matches -> 2 of 4 anchor rows (50%) < the 90% floor.
    pd.DataFrame({"route_id": ["101", "999"], "pop_served": [10000, 1]}).to_csv(
        bundle_dir / "features__route_id.csv", index=False
    )
    _write_manifest(
        bundle_dir,
        [("features__route_id.csv", ["route_id"]), ("features__period.csv", ["period"])],
    )
    with pytest.raises(ValueError, match="MIN_BUNDLE_MATCH_RATE"):
        mrm.assemble_model_table(anchor_path, 0, bundle_dir, manifest_path, verify_hashes=True)

    monkeypatch.setattr(mrm, "MIN_BUNDLE_MATCH_RATE", 0.0)
    merged, _ = mrm.assemble_model_table(
        anchor_path, 0, bundle_dir, manifest_path, verify_hashes=True
    )
    assert len(merged) == 4


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def _model_table(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    hours = rng.uniform(50, 500, n)
    pop = rng.uniform(1000, 50000, n)
    boardings = np.exp(1.0 + 0.6 * np.log(hours) + 0.3 * np.log(pop) + rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "route_id": [str(i) for i in range(n)],
            "ntd_boardings": boardings,
            "scheduled_hours": hours,
            "pop_served": pop,
        }
    )


def test_build_design_matrix_logs_missing_counts_by_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = _model_table()
    table.loc[table.index[:3], "pop_served"] = np.nan
    with caplog.at_level(logging.WARNING):
        y, x_matrix, names, frame, keep = mrm.build_design_matrix(
            table,
            "ntd_boardings",
            ["scheduled_hours", "pop_served"],
            (),
            log_dependent=True,
            log_predictors=("scheduled_hours", "pop_served"),
            standardize=False,
        )
    assert "pop_served" in caplog.text
    assert len(y) == len(table) - 3
    assert names == ["intercept", "log_scheduled_hours", "log_pop_served"]


def test_select_predictors_raises_on_missing_column() -> None:
    with pytest.raises(KeyError):
        mrm.select_predictors(_model_table(), ("nope",), "ntd_boardings", ("route_id",), ())


# ---------------------------------------------------------------------------
# Export + run log
# ---------------------------------------------------------------------------


def test_end_to_end_export_includes_influence_diagnostics(tmp_path: Path) -> None:
    table = _model_table()
    y, x_matrix, names, frame, _ = mrm.build_design_matrix(
        table,
        "ntd_boardings",
        ["scheduled_hours", "pop_served"],
        (),
        log_dependent=True,
        log_predictors=("scheduled_hours", "pop_served"),
        standardize=False,
    )
    result = mrm.fit_ols(y, x_matrix, names, se_type="HC1")
    workbook = mrm.export_results(result, frame, "ntd_boardings", True, tmp_path)

    sheets = pd.read_excel(workbook, sheet_name=None)
    assert set(sheets) == {"ModelSummary", "Coefficients", "Correlations", "Observations"}

    obs = sheets["Observations"]
    for col in ("studentized_residual", "loo_residual", "leverage", "cooks_d"):
        assert col in obs.columns, f"missing column {col}"
    assert obs["leverage"].between(0, 1 + 1e-9).all()

    summary = sheets["ModelSummary"].set_index("metric")["value"]
    assert "loo_r_squared" in summary.index


def test_write_run_log_records_anchor_sha(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.csv"
    anchor.write_text("route_id,ntd_boardings\n101,1000\n", encoding="utf-8")
    assert mrm.write_run_log(tmp_path, [("b.csv", "abc123")], anchor) is True
    content = (tmp_path / "monthly_ridership_model_runlog.txt").read_text(encoding="utf-8")
    assert f"Anchor SHA-256:   {mrm._sha256_file(anchor)}" in content
    assert "b.csv  sha256=abc123" in content
