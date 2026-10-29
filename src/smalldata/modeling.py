"""Model candidates and (nested) cross-validation for small tabular datasets.

Protocol per (model, target, seed) in ``nested_cv``:

* The outer loop produces the reported scores. With ``groups`` it is a ``GroupKFold``,
  so vehicles that share a platform (badge-engineered twins) are always in the same
  fold. Each outer test fold is never seen by imputation, scaling, hyperparameter
  tuning or feature selection.
* Inside each outer training fold (inner loop grouped the same way):
  1. ``RandomizedSearchCV`` tunes the hyperparameters on all features.
  2. Sequential floating forward selection (SFFS) picks a feature subset for the tuned model.
  3. The tuned model is refit on the selected features and predicts the outer test fold.

``flat_cv`` is a re-implementation of the thesis protocol, where tuning and selection
used the same random folds as scoring. It is kept to measure how optimistic that is.

All models predict a standardised target (``TransformedTargetRegressor``), so the
hyperparameter ranges (Lasso alpha, SVR C/epsilon) do not depend on the target's units.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import loguniform, randint
from sklearn import config_context
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SCORING = "neg_mean_squared_error"
N_ITER = 15
"""Randomised-search draws per tuning run, used by every protocol (the thesis used 50)."""
SEEDS = (0, 1, 2, 3, 4)
"""Seeds for repeated CV; each seed reshuffles the folds and the search draws."""

_M = "regressor__model__"  # parameter prefix inside TransformedTargetRegressor(Pipeline)


@dataclass
class Candidate:
    name: str
    estimator: object
    param_distributions: dict | list[dict] = field(default_factory=dict)

    def pipeline(self) -> TransformedTargetRegressor:
        features = Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                ("model", clone(self.estimator)),
            ]
        )
        return TransformedTargetRegressor(regressor=features, transformer=StandardScaler())

    def param_names(self) -> set[str]:
        spaces = self.param_distributions
        spaces = spaces if isinstance(spaces, list) else [spaces]
        return {name for space in spaces for name in space}


def default_candidates(n_samples: int = 31) -> list[Candidate]:
    """The four model families compared in the thesis.

    Search spaces are on the standardised target, with log-uniform priors for scale
    parameters. SVR samples ``gamma`` only for kernels that use it.
    """
    # k is drawn from 1..k_max (scipy's randint excludes the upper bound, hence +1).
    # 0.4 * n keeps k below the smallest inner training fold (about 0.6 * n rows);
    # the cap of 8 keeps k small relative to the 20-25 training rows.
    k_max = max(1, min(8, int(0.4 * n_samples)))
    C = loguniform(1e-2, 1e2)
    epsilon = loguniform(1e-3, 1e0)
    gamma = loguniform(1e-3, 1e0)
    return [
        Candidate("Linear regression", LinearRegression()),
        Candidate("Lasso", Lasso(max_iter=50_000), {_M + "alpha": loguniform(1e-3, 1e0)}),
        Candidate(
            "kNN",
            KNeighborsRegressor(),
            {_M + "n_neighbors": randint(1, k_max + 1), _M + "weights": ["uniform", "distance"]},
        ),
        Candidate(
            "SVR",
            SVR(),
            [
                {_M + "kernel": ["linear"], _M + "C": C, _M + "epsilon": epsilon},
                {_M + "kernel": ["rbf", "sigmoid", "poly"], _M + "C": C, _M + "epsilon": epsilon, _M + "gamma": gamma},
            ],
        ),
    ]


def _splitter(n_splits: int, groups: pd.Series | None, seed: int):
    if groups is None:
        return KFold(n_splits, shuffle=True, random_state=seed)
    return GroupKFold(min(n_splits, groups.nunique()), shuffle=True, random_state=seed)


def _tune(candidate: Candidate, X, y, cv, groups, n_iter: int, seed: int):
    est = candidate.pipeline()
    if not candidate.param_distributions:
        return est
    search = RandomizedSearchCV(
        est, candidate.param_distributions, n_iter=n_iter, cv=cv, scoring=SCORING, random_state=seed
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_


def _select(est, X, y, cv, groups) -> list[str]:
    sffs = SFS(clone(est), k_features="best", forward=True, floating=True, scoring=SCORING, cv=cv)
    sffs.fit(X, y, groups=groups)
    return list(sffs.k_feature_names_)


def _metrics(y_true: pd.Series, y_pred: pd.Series, folds: list[np.ndarray]) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),  # on pooled out-of-fold predictions
        "R2_fold_mean": float(np.mean([r2_score(y_true.iloc[f], y_pred.iloc[f]) for f in folds])),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE_%": 100 * mean_absolute_percentage_error(y_true, y_pred),
    }


@dataclass
class CVResult:
    model: str
    target: str
    protocol: str
    seed: int
    metrics: dict[str, float]
    predictions: pd.Series
    selected_features: list[list[str]]
    params: list[dict]


def nested_cv(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_iter: int = N_ITER,
    seed: int = 0,
) -> CVResult:
    """Nested CV: tuning and feature selection happen only inside each outer training fold.

    With ``groups``, outer and inner loops are ``GroupKFold`` (no group is split across
    folds); without, both are shuffled ``KFold``.
    """
    outer = _splitter(outer_splits, groups, seed)
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    features, params, folds = [], [], []

    for train_idx, test_idx in outer.split(X, y, groups):
        X_tr, X_te, y_tr = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx]
        g_tr = None if groups is None else groups.iloc[train_idx]
        inner = _splitter(inner_splits, g_tr, seed + 1000)
        tuned = _tune(candidate, X_tr, y_tr, inner, g_tr, n_iter, seed)
        chosen = _select(tuned, X_tr, y_tr, inner, g_tr)
        final = clone(tuned).fit(X_tr[chosen], y_tr)
        oof.iloc[test_idx] = final.predict(X_te[chosen])
        features.append(chosen)
        params.append({k: v for k, v in tuned.get_params().items() if k in candidate.param_names()})
        folds.append(test_idx)

    protocol = "nested, random KFold" if groups is None else "nested, GroupKFold"
    return CVResult(candidate.name, str(y.name), protocol, seed, _metrics(y, oof, folds), oof, features, params)


def flat_cv(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    n_iter: int = N_ITER,
    seed: int = 0,
) -> CVResult:
    """Re-implementation of the thesis protocol, kept to measure its optimism.

    Tune and select features with 5-fold random ``KFold`` on all rows, then score with
    5-fold CV on the same folds. ``groups`` is accepted for a uniform call signature
    and ignored: the thesis did not group platform twins.
    """
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    tuned = _tune(candidate, X, y, cv, None, n_iter, seed)
    chosen = _select(tuned, X, y, cv, None)
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    folds = []
    for train_idx, test_idx in cv.split(X):
        model = clone(tuned).fit(X.iloc[train_idx][chosen], y.iloc[train_idx])
        oof.iloc[test_idx] = model.predict(X.iloc[test_idx][chosen])
        folds.append(test_idx)
    params = {k: v for k, v in tuned.get_params().items() if k in candidate.param_names()}
    return CVResult(candidate.name, str(y.name), "flat (thesis re-implementation)", seed,
                    _metrics(y, oof, folds), oof, [chosen], [params])


def _run_one(protocol, candidate, X, y, groups, seed, kwargs) -> CVResult:
    # Skipping sklearn's input re-validation saves about a quarter of the runtime here;
    # the data are validated once by load_dataset. Convergence and failed-fit warnings
    # on tiny folds are silenced (a failed fit scores NaN and is never selected).
    with warnings.catch_warnings(), config_context(assume_finite=True, skip_parameter_validation=True):
        warnings.simplefilter("ignore")
        return protocol(candidate, X, y, groups, seed=seed, **kwargs)


def run_repeated(protocol, candidates, dataset, targets, seeds=SEEDS, group=True, n_jobs=-1, **kwargs):
    """Run ``protocol`` (``nested_cv`` or ``flat_cv``) for every target, model and seed in parallel."""
    jobs = []
    for target in targets:
        X, y, groups = dataset.target(target)
        for candidate in candidates:
            for seed in seeds:
                jobs.append(delayed(_run_one)(protocol, candidate, X, y, groups if group else None, seed, kwargs))
    return Parallel(n_jobs=n_jobs)(jobs)


def results_table(results: list[CVResult], metrics=("R2", "RMSE", "MAE", "MAPE_%")) -> pd.DataFrame:
    """Mean and standard deviation over seeds for each (target, model)."""
    rows = [{"target": r.target, "model": r.model, "seed": r.seed, **r.metrics} for r in results]
    df = pd.DataFrame(rows)
    table = df.groupby(["target", "model"], sort=False)[list(metrics)].agg(["mean", "std"])
    return table


def selection_frequency(results: CVResult | list[CVResult], features: list[str]) -> pd.Series:
    """Share of outer folds (over all given runs) in which SFFS kept each feature."""
    results = results if isinstance(results, list) else [results]
    counts = pd.Series(0, index=features, dtype=float)
    n = 0
    for result in results:
        for chosen in result.selected_features:
            counts[chosen] += 1
            n += 1
    return counts / n
