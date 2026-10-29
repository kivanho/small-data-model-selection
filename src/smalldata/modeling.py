"""Model candidates and nested cross-validation for small tabular datasets.

Protocol per (model, target):

* Outer loop (``KFold``) produces the reported scores. Each outer test fold is
  never seen by imputation, scaling, hyperparameter tuning or feature selection.
* Inside each outer training fold:
  1. ``RandomizedSearchCV`` (inner ``KFold``) tunes the hyperparameters on all features.
  2. Sequential floating forward selection (SFFS, inner ``KFold``) picks a
     feature subset for the tuned model.
  3. The tuned model is refit on the selected features and scored on the outer test fold.

The ``flat_cv`` function reproduces the original thesis protocol, where tuning
and selection used the same folds as scoring, for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SCORING = "neg_mean_squared_error"


@dataclass
class Candidate:
    name: str
    estimator: object
    param_distributions: dict = field(default_factory=dict)

    def pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                ("model", clone(self.estimator)),
            ]
        )


def default_candidates(n_samples: int = 31) -> list[Candidate]:
    """The four model families compared in the thesis, with its search spaces."""
    # k must stay below the size of the smallest inner training fold
    max_k = max(2, min(8, int(n_samples * 0.6)))
    return [
        Candidate("Linear regression", LinearRegression()),
        Candidate("Lasso", Lasso(max_iter=50_000), {"model__alpha": uniform(0.01, 1)}),
        Candidate(
            "kNN",
            KNeighborsRegressor(),
            {"model__n_neighbors": randint(1, max_k), "model__weights": ["uniform", "distance"]},
        ),
        Candidate(
            "SVR",
            SVR(),
            {
                "model__C": uniform(0.1, 100),
                "model__gamma": uniform(0.001, 1),
                "model__epsilon": uniform(0.001, 1),
                "model__kernel": ["linear", "rbf", "sigmoid", "poly"],
            },
        ),
    ]


def _tune(candidate: Candidate, X: pd.DataFrame, y: pd.Series, cv, n_iter: int, seed: int) -> Pipeline:
    pipe = candidate.pipeline()
    if not candidate.param_distributions:
        return pipe
    search = RandomizedSearchCV(
        pipe, candidate.param_distributions, n_iter=n_iter, cv=cv, scoring=SCORING, random_state=seed
    )
    search.fit(X, y)
    return search.best_estimator_


def _select(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> list[str]:
    sffs = SFS(clone(pipe), k_features="best", forward=True, floating=True, scoring=SCORING, cv=cv)
    sffs.fit(X, y)
    return list(sffs.k_feature_names_)


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE_%": 100 * mean_absolute_percentage_error(y_true, y_pred),
    }


@dataclass
class CVResult:
    model: str
    target: str
    metrics: dict[str, float]
    predictions: pd.Series
    selected_features: list[list[str]]
    params: list[dict]


def nested_cv(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_iter: int = 30,
    seed: int = 42,
) -> CVResult:
    """Nested CV: tuning and feature selection happen only inside each outer training fold."""
    outer = KFold(outer_splits, shuffle=True, random_state=seed)
    inner = KFold(inner_splits, shuffle=True, random_state=seed + 1)
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    features, params = [], []

    for train_idx, test_idx in outer.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        tuned = _tune(candidate, X_tr, y_tr, inner, n_iter, seed)
        chosen = _select(tuned, X_tr, y_tr, inner)
        final = clone(tuned).fit(X_tr[chosen], y_tr)
        oof.iloc[test_idx] = final.predict(X_te[chosen])
        features.append(chosen)
        params.append({k: v for k, v in tuned.get_params().items() if k in candidate.param_distributions})

    return CVResult(candidate.name, str(y.name), _metrics(y, oof), oof, features, params)


def flat_cv(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_iter: int = 30,
    seed: int = 42,
) -> CVResult:
    """Original thesis protocol: tune and select on the full data, then score on the same folds.

    Kept only to show how optimistic this is compared with ``nested_cv``.
    """
    folds = KFold(n_splits, shuffle=True, random_state=seed)
    tuned = _tune(candidate, X, y, folds, n_iter, seed)
    chosen = _select(tuned, X, y, folds)
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    for train_idx, test_idx in folds.split(X):
        model = clone(tuned).fit(X.iloc[train_idx][chosen], y.iloc[train_idx])
        oof.iloc[test_idx] = model.predict(X.iloc[test_idx][chosen])
    params = {k: v for k, v in tuned.get_params().items() if k in candidate.param_distributions}
    return CVResult(candidate.name, str(y.name), _metrics(y, oof), oof, [chosen], [params])


def results_table(results: list[CVResult]) -> pd.DataFrame:
    rows = [{"target": r.target, "model": r.model, **r.metrics} for r in results]
    return pd.DataFrame(rows).set_index(["target", "model"])


def selection_frequency(result: CVResult, features: list[str]) -> pd.Series:
    """Share of outer folds in which each feature was selected (stability of SFFS)."""
    counts = pd.Series(0, index=features, dtype=float)
    for chosen in result.selected_features:
        counts[chosen] += 1
    return counts / len(result.selected_features)
