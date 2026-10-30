"""Fold-local preprocessing: imputation and scaling never see the outer test fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from smalldata import Candidate, nested_cv
from smalldata.modeling import _final_fit_predict

from conftest import SMALL


def test_pipeline_has_imputer_and_scaler_inside(linear_candidate):
    pipe = linear_candidate.pipeline()
    steps = [name for name, _ in pipe.regressor.steps]
    assert steps == ["impute", "scale", "model"]


def test_imputation_uses_training_fold_mean_only():
    X_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    y_tr = pd.Series([1.0, 2.0, 3.0, 4.0])
    X_te = pd.DataFrame({"a": [np.nan, 100.0], "b": [10.0, 200.0]})
    cand = Candidate("lin", LinearRegression(), feature_selection=False)
    ctx = {"target": "t", "model": "lin", "seed": 0}
    pred = _final_fit_predict(cand.pipeline(), X_tr, y_tr, X_te, ["a", "b"], ctx, 0)
    # the NaN in column a must be replaced by the TRAINING mean (2.5), not by anything from X_te
    X_manual = X_te.copy()
    X_manual.loc[0, "a"] = X_tr["a"].mean()
    pred_manual = _final_fit_predict(cand.pipeline(), X_tr, y_tr, X_manual, ["a", "b"], ctx, 0)
    np.testing.assert_allclose(pred, pred_manual)


def test_outer_test_rows_do_not_influence_each_other(grouped_data, linear_candidate):
    """Perturbing other rows of the same outer test fold (and all test targets) leaves a
    row's prediction unchanged, so imputation, scaling and fitting used training rows only."""
    X, y, g = grouped_data
    res = nested_cv(linear_candidate, X, y, g, seed=0, **SMALL)
    nan_row = X.index[X["f4"].isna()][0]
    fold = res.fold_assignments[nan_row]
    others = res.fold_assignments.index[(res.fold_assignments == fold) & (res.fold_assignments.index != nan_row)]
    X2, y2 = X.copy(), y.copy()
    X2.loc[others] = X2.loc[others].fillna(0) * 1000 + 50  # wildly different test-fold statistics
    y2.loc[res.fold_assignments == fold] += 1e3  # test targets are never used for fitting
    res2 = nested_cv(linear_candidate, X2, y2, g, seed=0, **SMALL)
    pd.testing.assert_series_equal(res.fold_assignments, res2.fold_assignments)
    assert np.isclose(res.predictions[nan_row], res2.predictions[nan_row])
