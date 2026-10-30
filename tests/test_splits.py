"""Group separation in outer and inner folds, deterministic splits, prediction coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smalldata import modeling, nested_cv, nested_cv_complete
from smalldata.modeling import _splitter

from conftest import SMALL, make_grouped_data


def _groups_never_split(fold_ids: pd.Series, groups: pd.Series) -> bool:
    return (pd.concat([fold_ids.rename("fold"), groups.rename("g")], axis=1).groupby("g")["fold"].nunique() == 1).all()


def test_outer_folds_keep_groups_together(grouped_data, linear_candidate):
    X, y, g = grouped_data
    res = nested_cv(linear_candidate, X, y, g, seed=3, **SMALL)
    assert _groups_never_split(res.fold_assignments, g)
    assert res.protocol == "nested, GroupKFold"
    assert sorted(res.fold_assignments.unique()) == [0, 1, 2]


def test_inner_folds_keep_groups_together_and_exclude_outer_test(grouped_data, linear_candidate, monkeypatch):
    """Spy on the inner splitter actually handed to the tuning step inside nested_cv."""
    X, y, g = grouped_data
    seen = []
    real = modeling._fit_candidate_in_fold

    def spy(candidate, X_tr, y_tr, inner, g_tr, *args, **kwargs):
        assert g_tr is not None and g_tr.index.equals(X_tr.index)
        for tr, te in inner.split(X_tr, y_tr, g_tr):
            assert not set(g_tr.iloc[tr]) & set(g_tr.iloc[te]), "a group spans inner train and inner test"
            assert len(tr) + len(te) == len(X_tr)
        seen.append(set(X_tr.index))
        return real(candidate, X_tr, y_tr, inner, g_tr, *args, **kwargs)

    monkeypatch.setattr(modeling, "_fit_candidate_in_fold", spy)
    res = nested_cv(linear_candidate, X, y, g, seed=1, **SMALL)
    assert len(seen) == SMALL["outer_splits"]
    for fold, train_rows in enumerate(seen):
        test_rows = set(res.fold_assignments.index[res.fold_assignments == fold])
        assert not train_rows & test_rows, "outer test rows leaked into the inner loop"
        assert train_rows | test_rows == set(X.index)


def test_random_kfold_without_groups_is_labelled(grouped_data, linear_candidate):
    X, y, _ = grouped_data
    res = nested_cv(linear_candidate, X, y, None, seed=0, **SMALL)
    assert res.protocol == "nested, random KFold"
    assert res.predictions.notna().all()


def test_every_sample_gets_exactly_one_prediction(grouped_data, linear_candidate):
    X, y, g = grouped_data
    res = nested_cv(linear_candidate, X, y, g, seed=0, **SMALL)
    assert res.predictions.index.equals(y.index)
    assert np.isfinite(res.predictions.to_numpy()).all()
    assert (res.fold_assignments >= 0).all()
    covered = np.concatenate([d.test_index for d in res.fold_details])
    assert sorted(covered) == sorted(X.index)


def test_same_seed_is_deterministic(grouped_data, ridge_candidate):
    X, y, g = grouped_data
    a = nested_cv(ridge_candidate, X, y, g, seed=5, **SMALL)
    b = nested_cv(ridge_candidate, X, y, g, seed=5, **SMALL)
    pd.testing.assert_series_equal(a.predictions, b.predictions)
    pd.testing.assert_series_equal(a.fold_assignments, b.fold_assignments)
    assert a.params == b.params


def test_fold_seed_and_search_seed_are_separable(grouped_data, ridge_candidate):
    X, y, g = grouped_data
    base = nested_cv(ridge_candidate, X, y, g, seed=0, **SMALL)
    same_folds = nested_cv(ridge_candidate, X, y, g, seed=7, fold_seed=0, **SMALL)
    other_folds = nested_cv(ridge_candidate, X, y, g, seed=7, **SMALL)
    pd.testing.assert_series_equal(base.fold_assignments, same_folds.fold_assignments)
    assert same_folds.search_seed == 7 and same_folds.fold_seed == 0
    assert not base.fold_assignments.equals(other_folds.fold_assignments)


def test_baselines_and_complete_procedure_share_outer_folds(grouped_data, linear_candidate, ridge_candidate):
    X, y, g = grouped_data
    a = nested_cv(linear_candidate, X, y, g, seed=2, **SMALL)
    b = nested_cv(ridge_candidate, X, y, g, seed=2, **SMALL)
    c = nested_cv_complete([linear_candidate, ridge_candidate], X, y, g, seed=2, **SMALL)
    pd.testing.assert_series_equal(a.fold_assignments, b.fold_assignments)
    pd.testing.assert_series_equal(a.fold_assignments, c.fold_assignments)


def test_splitter_refuses_too_few_groups():
    _, _, g = make_grouped_data(n_groups=3)
    with pytest.raises(ValueError, match="too few"):
        _splitter(5, g, 0)
