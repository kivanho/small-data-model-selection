"""Warnings are captured (not silenced), failed search candidates are excluded, invalid
final fits raise, and the complete procedure selects inside the inner loop."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from smalldata import COMPLETE_PROCEDURE, Candidate, InvalidFitError, flat_cv, nested_cv, nested_cv_complete
from smalldata.modeling import _M, jaccard_stability

from conftest import SMALL


class Fragile(BaseEstimator, RegressorMixin):
    """Fails to fit when ``threshold`` is large, warns when it is tiny, else fits a mean."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        if self.threshold > 5:
            raise ValueError("deliberate failure")
        if self.threshold < 0:
            warnings.warn("deliberate convergence warning", UserWarning)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class NaNPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


def test_warnings_are_recorded_with_context(grouped_data):
    X, y, g = grouped_data
    cand = Candidate("fragile", Fragile(), {_M + "threshold": [-1.0]}, feature_selection=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing may escape to the caller
        res = nested_cv(cand, X, y, g, seed=0, **SMALL)
    assert res.warnings, "warnings raised inside fits must be captured"
    w = res.warnings[0]
    assert (w.target, w.model, w.seed) == ("target", "fragile", 0)
    assert w.fold >= 0 and w.stage in {"random search", "final fit"}
    assert w.category == "UserWarning"
    assert any("convergence" in w.message for w in res.warnings)
    assert {w.stage for w in res.warnings} == {"random search", "final fit"}
    assert sum(w.count for w in res.warnings) > 0


def test_failed_search_candidates_are_excluded_and_recorded(grouped_data):
    X, y, g = grouped_data
    cand = Candidate("fragile", Fragile(), {_M + "threshold": [0.0, 10.0]}, feature_selection=False)
    res = nested_cv(cand, X, y, g, seed=0, outer_splits=3, inner_splits=3, n_iter=2)
    assert res.failures and all(f.params[_M + "threshold"] == 10.0 for f in res.failures)
    assert all(f.n_failed_inner_folds == 3 for f in res.failures)
    assert all(p[_M + "threshold"] == 0.0 for p in res.params)
    assert all(d.n_failed_candidates == 1 and d.n_search_candidates == 2 for d in res.fold_details)


def test_all_candidates_failing_raises(grouped_data):
    X, y, g = grouped_data
    cand = Candidate("fragile", Fragile(), {_M + "threshold": [10.0, 20.0]}, feature_selection=False)
    with pytest.raises(InvalidFitError, match="all 2 search candidates failed"):
        nested_cv(cand, X, y, g, seed=0, outer_splits=3, inner_splits=3, n_iter=2)


def test_untuned_candidate_fit_failure_raises(grouped_data):
    X, y, g = grouped_data
    cand = Candidate("fragile", Fragile(threshold=10.0), feature_selection=False)
    with pytest.raises(ValueError, match="deliberate failure"):
        nested_cv(cand, X, y, g, seed=0, **SMALL)


def test_nonfinite_final_predictions_raise(grouped_data):
    X, y, g = grouped_data
    cand = Candidate("nan", NaNPredictor(), feature_selection=False)
    with pytest.raises((InvalidFitError, ValueError)):
        nested_cv(cand, X, y, g, seed=0, **SMALL)


def test_complete_procedure_picks_family_inside_inner_loop(grouped_data):
    X, y, g = grouped_data
    mean = Candidate("Mean", DummyRegressor(strategy="mean"), feature_selection=False)
    lin = Candidate("Linear regression", LinearRegression(), feature_selection=False)
    res = nested_cv_complete([mean, lin], X, y, g, seed=0, **SMALL)
    assert res.model == COMPLETE_PROCEDURE
    assert set(res.chosen_families) <= {"Mean", "Linear regression"}
    assert res.chosen_families == ["Linear regression"] * SMALL["outer_splits"], "linear data must favour the linear model"
    assert all(p["family"] == "Linear regression" for p in res.params)
    assert res.metrics["R2"] > 0.9
    assert res.config["families"] == ["Mean", "Linear regression"]


def test_complete_procedure_with_sfs_reports_features(grouped_data, sfs_candidate, linear_candidate):
    X, y, g = grouped_data
    res = nested_cv_complete([linear_candidate, sfs_candidate], X, y, g, seed=1, **SMALL)
    assert len(res.selected_features) == SMALL["outer_splits"]
    assert all(set(f) <= set(X.columns) and f for f in res.selected_features)
    assert np.isfinite(res.predictions).all()


def test_sfs_candidate_records_selection_time_and_subset(grouped_data, sfs_candidate):
    X, y, g = grouped_data
    res = nested_cv(sfs_candidate, X, y, g, seed=0, **SMALL)
    assert all(d.select_s > 0 for d in res.fold_details)
    assert all(1 <= len(f) <= 4 for f in res.selected_features)
    assert res.config["feature_selection"] is True
    # f4 carries no signal and is partly missing; the informative f1 should always survive
    assert all("f1" in f for f in res.selected_features)


def test_flat_cv_is_labelled_optimistic(grouped_data, ridge_candidate):
    X, y, g = grouped_data
    res = flat_cv(ridge_candidate, X, y, g, n_splits=3, n_iter=3, seed=0)
    assert res.protocol.startswith("flat")
    assert res.config["optimistic"] is True
    assert res.predictions.notna().all()
    assert len(res.selected_features) == 1  # one global subset, by construction of the thesis protocol


def test_jaccard_stability():
    assert jaccard_stability([["a", "b"], ["a", "b"], ["a", "b"]]) == 1.0
    assert jaccard_stability([["a"], ["b"]]) == 0.0
    assert np.isnan(jaccard_stability([["a"]]))
    assert np.isclose(jaccard_stability([["a", "b"], ["a"]]), 0.5)
