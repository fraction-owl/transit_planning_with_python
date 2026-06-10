import numpy as np
import pandas as pd
import pytest

from scripts.modeling import ridership_ml_model as rml


@pytest.fixture
def panel() -> pd.DataFrame:
    """A synthetic route table with a non-log-linear, interacting signal.

    The relationship between (log) ridership and the predictors is oscillatory
    in hours, convex in population, and carries a threshold interaction — none
    of which a linear-in-logs OLS can represent, but tree ensembles can. This
    is exactly the regime where the ML variant should out-predict the baseline.
    """
    rng = np.random.default_rng(7)
    n = 300
    hours = rng.uniform(50, 500, n)
    pop = rng.uniform(1000, 50000, n)
    gas = rng.uniform(2.5, 4.5, n)
    signal = (
        3.0
        + 1.5 * np.sin(hours / 80.0)  # oscillatory in hours
        + 1.2 * (pop / 50000.0) ** 2  # convex in population
        + 1.0 * ((hours > 250) & (pop > 25000))  # threshold interaction
    )
    boardings = np.exp(signal + rng.normal(0, 0.05, n))
    return pd.DataFrame(
        {"ntd_boardings": boardings, "scheduled_hours": hours, "pop_served": pop, "gas_price": gas}
    )


@pytest.fixture
def feature_matrix(panel) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build the model-ready target and feature matrix from the panel fixture."""
    return rml.build_feature_frame(
        panel,
        dependent="ntd_boardings",
        predictors=["scheduled_hours", "pop_served", "gas_price"],
        categoricals=(),
        log_dependent=True,
        log_predictors=("scheduled_hours", "pop_served"),
    )


def test_build_feature_frame_log_and_dropna() -> None:
    """Log transforms apply, rows with NaNs drop, and no intercept is added."""
    df = pd.DataFrame(
        {
            "ntd_boardings": [100.0, 200.0, 400.0, np.nan],
            "scheduled_hours": [10.0, 20.0, 40.0, 50.0],
            "gas_price": [3.0, 3.5, 4.0, 4.5],
        }
    )
    y, x_frame, frame = rml.build_feature_frame(
        df,
        dependent="ntd_boardings",
        predictors=["scheduled_hours", "gas_price"],
        categoricals=(),
        log_dependent=True,
        log_predictors=("scheduled_hours",),
    )

    assert len(y) == 3  # NaN row dropped
    assert list(x_frame.columns) == ["log_scheduled_hours", "gas_price"]
    np.testing.assert_allclose(y, np.log([100.0, 200.0, 400.0]))
    np.testing.assert_allclose(x_frame["log_scheduled_hours"], np.log1p([10.0, 20.0, 40.0]))
    assert len(frame) == 3


def test_log_dependent_rejects_nonpositive() -> None:
    """Non-positive target under a log transform must raise."""
    df = pd.DataFrame({"y": [1.0, 0.0, 3.0], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        rml.build_feature_frame(
            df,
            dependent="y",
            predictors=["x"],
            categoricals=(),
            log_dependent=True,
            log_predictors=(),
        )


def test_out_of_fold_predictions_cover_every_row(feature_matrix) -> None:
    """Every observation receives exactly one out-of-fold prediction."""
    y, x_frame, _ = feature_matrix
    model = rml.build_models(0, rml.RF_PARAMS, rml.GBT_PARAMS)["Random Forest"]
    oof = rml.out_of_fold_predictions(model, x_frame, y, n_splits=5, random_state=0)

    assert oof.shape == y.shape
    assert not np.isnan(oof).any()


def test_metrics_match_known_values() -> None:
    """The metric helpers reproduce hand-computed values."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0, 5.0])  # one unit off on the last point
    assert rml._rmse(actual, predicted) == pytest.approx(0.5)
    assert rml._mae(actual, predicted) == pytest.approx(0.25)
    assert rml._r2(actual, predicted) == pytest.approx(0.8)
    assert rml._mape(actual, predicted) == pytest.approx(0.25 / 4)


def test_gradient_boosting_beats_ols_on_nonlinear_signal(feature_matrix) -> None:
    """On an interacting, nonlinear signal the ensembles out-predict OLS."""
    y, x_frame, _ = feature_matrix
    models = rml.build_models(0, rml.RF_PARAMS, rml.GBT_PARAMS)
    scored = {
        name: rml.score_model(name, model, x_frame, y, 5, 0, log_dependent=True)
        for name, model in models.items()
    }

    ols_r2 = scored["OLS (baseline)"].r2_level
    gbt_r2 = scored["Gradient Boosting"].r2_level
    assert gbt_r2 > ols_r2
    # Out-of-fold metrics are honest but should still be a strong fit here.
    assert gbt_r2 > 0.8


def test_permutation_importance_ranks_real_predictors(feature_matrix) -> None:
    """Permutation importance returns one row per feature, sorted descending."""
    y, x_frame, _ = feature_matrix
    model = rml.build_models(0, rml.RF_PARAMS, rml.GBT_PARAMS)["Gradient Boosting"]
    importance, fitted, x_train, y_train = rml.compute_permutation_importance(
        model, x_frame, y, holdout_fraction=0.25, n_repeats=10, random_state=0
    )

    assert list(importance.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(importance["feature"]) == set(x_frame.columns)
    # Sorted by mean importance, descending.
    means = importance["importance_mean"].to_numpy()
    assert np.all(np.diff(means) <= 1e-9)
    # The fitted model and its training rows come back for downstream plotting.
    assert len(x_train) == len(y_train)


def test_assemble_model_table_left_join(tmp_path) -> None:
    """Feature tables left-join onto the anchor on their declared keys."""
    anchor = tmp_path / "anchor.csv"
    feature = tmp_path / "feature.csv"
    pd.DataFrame({"route_id": [1, 2, 3], "ntd_boardings": [100, 200, 300]}).to_csv(
        anchor, index=False
    )
    pd.DataFrame({"route_id": [1, 2], "pop_served": [500, 600]}).to_csv(feature, index=False)

    table = rml.assemble_model_table(
        anchor,
        0,
        [
            rml.FeatureTable(
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


def test_end_to_end_export(tmp_path, feature_matrix) -> None:
    """Score, interpret, then export produces a workbook with the expected sheets."""
    y, x_frame, model_frame = feature_matrix
    models = rml.build_models(0, rml.RF_PARAMS, rml.GBT_PARAMS)
    scores = [
        rml.score_model(name, model, x_frame, y, 5, 0, log_dependent=True)
        for name, model in models.items()
    ]
    best = max(scores, key=lambda s: s.r2_level)
    importance, *_ = rml.compute_permutation_importance(models[best.name], x_frame, y, 0.25, 10, 0)

    workbook = rml.export_results(
        scores, best, importance, model_frame, "ntd_boardings", True, tmp_path
    )

    assert workbook.exists()
    sheets = pd.read_excel(workbook, sheet_name=None)
    assert set(sheets) == {"ModelComparison", "FeatureImportance", "Predictions"}
    assert len(sheets["ModelComparison"]) == 3  # one row per model
