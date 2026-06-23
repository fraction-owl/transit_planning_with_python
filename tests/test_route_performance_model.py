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
    """Keys are trimmed, de-floated, and NaN-filled to a stable string form."""
    out = rpm._canonical_key(pd.Series([" 159 ", "159.0", 42, None]))
    assert out.tolist() == ["159", "159", "42", ""]


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

    y, x_matrix, names, frame, _ = rpm.build_design_matrix(table)
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


def test_log_dependent_rejects_nonpositive(monkeypatch) -> None:
    """A non-positive dependent under the log transform must raise."""
    monkeypatch.setattr(rpm, "PREDICTORS", ("revenue_hours",))
    monkeypatch.setattr(rpm, "LOG_PREDICTORS", ("revenue_hours",))
    table = pd.DataFrame(
        {
            "route_id": ["1", "2", "3"],
            "ntd_boardings": [100.0, 0.0, 300.0],  # a zero is invalid under log
            "revenue_hours": [10.0, 20.0, 30.0],
        }
    )
    with pytest.raises(ValueError):
        rpm.build_design_matrix(table)
