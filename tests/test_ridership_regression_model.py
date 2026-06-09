import numpy as np
import pandas as pd
import pytest

from scripts.modeling import ridership_regression_model as rrm


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
    result = rrm.fit_ols(y, x_matrix, names, se_type="classical")

    assert result.n_obs == 80
    assert result.n_params == 3
    # Slopes recover tightly; the intercept carries more sampling noise at n=80.
    np.testing.assert_allclose(result.params[1:], [3.0, -1.5], atol=0.2)
    np.testing.assert_allclose(result.params[0], 2.0, atol=0.8)
    assert result.r_squared > 0.98
    # Slopes are strongly significant; the intercept may or may not be.
    assert result.p_values[1] < 0.01
    assert result.p_values[2] < 0.01


def test_hc1_matches_classical_under_homoskedasticity(design) -> None:
    """HC1 and classical SEs should be close when errors are homoskedastic."""
    y, x_matrix, names = design
    classical = rrm.fit_ols(y, x_matrix, names, se_type="classical")
    robust = rrm.fit_ols(y, x_matrix, names, se_type="HC1")

    np.testing.assert_allclose(classical.params, robust.params)
    np.testing.assert_allclose(classical.std_errors, robust.std_errors, rtol=0.5)


def test_fit_ols_requires_enough_observations() -> None:
    """Fewer observations than parameters must raise."""
    x_matrix = np.ones((2, 3))
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        rrm.fit_ols(y, x_matrix, ["intercept", "a", "b"], se_type="classical")


def test_vif_flags_collinearity() -> None:
    """A duplicated predictor should yield a very large VIF."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=50)
    x2 = x1 + rng.normal(scale=1e-6, size=50)  # nearly identical to x1
    x_matrix = np.column_stack([np.ones(50), x1, x2])
    vif = rrm._variance_inflation_factors(x_matrix, ["intercept", "x1", "x2"])
    assert vif["x1"] > 100
    assert vif["x2"] > 100


def test_build_design_matrix_log_and_dropna() -> None:
    """Log transforms apply and rows with NaNs are dropped."""
    df = pd.DataFrame(
        {
            "ntd_boardings": [100.0, 200.0, 400.0, np.nan],
            "scheduled_hours": [10.0, 20.0, 40.0, 50.0],
            "gas_price": [3.0, 3.5, 4.0, 4.5],
        }
    )
    y, x_matrix, term_names, frame, keep_mask = rrm.build_design_matrix(
        df,
        dependent="ntd_boardings",
        predictors=["scheduled_hours", "gas_price"],
        categoricals=(),
        log_dependent=True,
        log_predictors=("scheduled_hours",),
        standardize=False,
    )

    assert len(y) == 3  # NaN row dropped
    assert keep_mask.tolist() == [True, True, True, False]
    assert term_names == ["intercept", "log_scheduled_hours", "gas_price"]
    # Response was log-transformed.
    np.testing.assert_allclose(y, np.log([100.0, 200.0, 400.0]))
    # Intercept column is all ones.
    np.testing.assert_allclose(x_matrix[:, 0], 1.0)


def test_log_dependent_rejects_nonpositive() -> None:
    """Non-positive response under a log transform must raise."""
    df = pd.DataFrame({"y": [1.0, 0.0, 3.0], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        rrm.build_design_matrix(
            df,
            dependent="y",
            predictors=["x"],
            categoricals=(),
            log_dependent=True,
            log_predictors=(),
            standardize=False,
        )


def test_assemble_model_table_left_join(tmp_path) -> None:
    """Feature tables left-join onto the anchor on their declared keys."""
    anchor = tmp_path / "anchor.csv"
    feature = tmp_path / "feature.csv"
    pd.DataFrame({"route_id": [1, 2, 3], "ntd_boardings": [100, 200, 300]}).to_csv(
        anchor, index=False
    )
    pd.DataFrame({"route_id": [1, 2], "pop_served": [500, 600]}).to_csv(feature, index=False)

    table = rrm.assemble_model_table(
        anchor,
        0,
        [
            rrm.FeatureTable(
                label="demo",
                path=feature,
                join_keys=("route_id",),
                keep_cols=("pop_served",),
            )
        ],
    )

    assert list(table.columns) == ["route_id", "ntd_boardings", "pop_served"]
    assert table.loc[table["route_id"] == 3, "pop_served"].isna().all()  # unmatched -> NaN
    assert table.loc[table["route_id"] == 1, "pop_served"].iloc[0] == 500


def test_end_to_end_export(tmp_path) -> None:
    """Fit then export produces a workbook with the expected sheets."""
    rng = np.random.default_rng(7)
    n = 60
    hours = rng.uniform(50, 500, n)
    pop = rng.uniform(1000, 50000, n)
    boardings = np.exp(1.0 + 0.6 * np.log(hours) + 0.3 * np.log(pop) + rng.normal(0, 0.1, n))
    df = pd.DataFrame({"ntd_boardings": boardings, "scheduled_hours": hours, "pop_served": pop})

    y, x_matrix, names, frame, _ = rrm.build_design_matrix(
        df,
        dependent="ntd_boardings",
        predictors=["scheduled_hours", "pop_served"],
        categoricals=(),
        log_dependent=True,
        log_predictors=("scheduled_hours", "pop_served"),
        standardize=False,
    )
    result = rrm.fit_ols(y, x_matrix, names, se_type="HC1")
    workbook = rrm.export_results(result, frame, "ntd_boardings", True, tmp_path)

    assert workbook.exists()
    sheets = pd.read_excel(workbook, sheet_name=None)
    assert set(sheets) == {"ModelSummary", "Coefficients", "Correlations", "Observations"}
    assert result.r_squared > 0.8
