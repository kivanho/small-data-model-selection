"""Model candidates and (nested) cross-validation for small grouped tabular datasets.

Protocol per (model, target, seed) in ``nested_cv``:

* The outer loop produces the reported scores. With ``groups`` it is a ``GroupKFold``,
  so vehicles that share a platform (badge-engineered twins) are always in the same
  fold. Each outer test fold is never seen by imputation, scaling, hyperparameter
  tuning or feature selection.
* Inside each outer training fold (inner loop grouped the same way):
  1. ``RandomizedSearchCV`` tunes the hyperparameters on all features. A drawn setting
     that fails on any inner fold is recorded and excluded; if every draw fails the run
     raises ``InvalidFitError``.
  2. If the candidate uses feature selection, sequential floating forward selection (SFFS)
     picks a feature subset for the tuned model. Baselines skip this step.
  3. The tuned model is refit on the selected features and predicts the outer test fold.
     Non-finite predictions raise ``InvalidFitError``.

``nested_cv_complete`` runs the *complete* selection procedure: inside each outer training
fold it tunes and selects features for every candidate family, picks the family with the
best inner-CV score, and scores only that choice on the outer fold. Its outer score is the
honest estimate for "pick the best family automatically"; the per-family scores from
``nested_cv`` answer the separate question "how did each prespecified family perform?".

``flat_cv`` re-implements the thesis protocol, where tuning and selection used the same
random folds as scoring. It is intentionally optimistic and kept only to measure that optimism.

Randomness: ``seed`` sets both the fold assignment (``fold_seed``) and the random-search
draws (``search_seed``). Pass them separately to vary one source while fixing the other.
Seed-to-seed spread is therefore a resampling-plus-search variability, not a confidence
interval for the generalisation error.

All models predict a standardised target (``TransformedTargetRegressor``), so the
hyperparameter ranges (Lasso alpha, SVR C/epsilon) do not depend on the target's units.
Warnings raised while fitting are captured per fold (never silenced) and returned on the
result as ``WarningRecord`` entries.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SCORING = "neg_mean_squared_error"
N_ITER = 15
"""Randomised-search draws per tuning run, used by every protocol (the thesis used 50)."""
SEEDS = (0, 1, 2, 3, 4)
"""Seeds for repeated CV; each seed reshuffles the folds AND the search draws."""
COMPLETE_PROCEDURE = "Complete procedure"
"""Model name used for ``nested_cv_complete`` results."""

_M = "regressor__model__"  # parameter prefix inside TransformedTargetRegressor(Pipeline)


class InvalidFitError(RuntimeError):
    """A verification or final fit produced invalid output, or no search candidate was valid."""


@dataclass
class Candidate:
    name: str
    estimator: object
    param_distributions: dict | list[dict] = field(default_factory=dict)
    feature_selection: bool = True
    """Run SFFS inside each outer training fold. Baselines set this to False."""

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

    def max_n_neighbors(self) -> int | None:
        """Largest ``n_neighbors`` the search can draw, or None if the model has no such parameter."""
        key = _M + "n_neighbors"
        spaces = self.param_distributions
        spaces = spaces if isinstance(spaces, list) else [spaces]
        values = []
        for space in spaces:
            if key in space:
                dist = space[key]
                values.append(int(dist.support()[1]) if hasattr(dist, "support") else int(max(dist)))
        if values:
            return max(values)
        params = self.estimator.get_params()
        return int(params["n_neighbors"]) if "n_neighbors" in params else None


def default_candidates(n_samples: int = 31) -> list[Candidate]:
    """The four model families compared in the thesis, each with SFFS feature selection.

    Search spaces are on the standardised target, with log-uniform priors for scale
    parameters. SVR samples ``gamma`` only for kernels that use it.
    """
    # k is drawn from 1..k_max (scipy's randint excludes the upper bound, hence +1).
    # 0.4 * n keeps k below the smallest inner training fold (about 0.6 * n rows);
    # the cap of 8 keeps k small relative to the 20-25 training rows. ``validate_inputs``
    # checks the bound against the actual folds before any fit.
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


def baseline_candidates() -> list[Candidate]:
    """Reference models without sequential feature selection.

    * ``Mean`` predicts the training-fold mean of the target (the R² = 0 reference).
    * ``Ridge (all features)`` is L2-regularised linear regression on all nine features,
      alpha tuned in the inner loop.
    * ``Lasso (no SFS)`` is the Lasso family with its alpha tuned in the inner loop but
      without SFFS, so the Lasso vs Lasso (no SFS) gap isolates what SFFS adds.
    """
    return [
        Candidate("Mean", DummyRegressor(strategy="mean"), feature_selection=False),
        Candidate("Ridge (all features)", Ridge(), {_M + "alpha": loguniform(1e-3, 1e2)}, feature_selection=False),
        Candidate("Lasso (no SFS)", Lasso(max_iter=50_000), {_M + "alpha": loguniform(1e-3, 1e0)}, feature_selection=False),
    ]


# ----------------------------------------------------------------------------- records


@dataclass
class WarningRecord:
    target: str
    model: str
    seed: int
    fold: int
    stage: str
    category: str
    message: str
    count: int = 1


@dataclass
class FailureRecord:
    """A search candidate (hyperparameter draw) that failed on at least one inner fold."""

    target: str
    model: str
    seed: int
    fold: int
    stage: str
    params: dict
    n_failed_inner_folds: int
    error: str


@dataclass
class FoldDetail:
    fold: int
    test_index: list
    family: str
    features: list[str]
    params: dict
    inner_score: float
    """Inner-CV score (``SCORING``) of the final configuration, as used for selection."""
    n_search_candidates: int
    n_failed_candidates: int
    tune_s: float
    select_s: float
    fit_s: float


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
    fold_assignments: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    groups: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    fold_details: list[FoldDetail] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)
    fold_seed: int = 0
    search_seed: int = 0
    runtime_s: float = float("nan")
    config: dict = field(default_factory=dict)

    @property
    def chosen_families(self) -> list[str]:
        return [d.family for d in self.fold_details]


# --------------------------------------------------------------------------- validation


def validate_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    outer_splits: int,
    inner_splits: int,
    candidates: list[Candidate] | Candidate | None = None,
) -> None:
    """Check that a grouped nested CV on (X, y, groups) is well posed; raise ``ValueError`` otherwise.

    Checks: pandas inputs with aligned, unique indices; finite targets; features finite or
    missing (NaN is allowed, inf is not); at least one column; enough groups for the outer
    loop and enough groups in every outer training fold for the inner loop; the largest
    kNN ``n_neighbors`` a candidate can draw fits in the smallest inner training fold.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X must be a DataFrame and y a Series")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have the same index")
    if X.shape[1] == 0:
        raise ValueError("X has no feature columns")
    if not X.index.is_unique:
        raise ValueError("X has duplicate index labels")
    if y.isna().any():
        raise ValueError(f"y has {int(y.isna().sum())} missing values; drop them first (Dataset.target does)")
    y_values = pd.to_numeric(y, errors="coerce")
    if y_values.isna().any() or not np.isfinite(y_values.to_numpy(dtype=float)).all():
        raise ValueError("y contains non-numeric or non-finite values")
    if y_values.nunique() < 2:
        raise ValueError("y is constant")
    X_values = X.apply(pd.to_numeric, errors="coerce")
    if (X_values.isna() & X.notna()).any().any():
        raise ValueError("X contains non-numeric values")
    if np.isinf(X_values.to_numpy(dtype=float)).any():
        raise ValueError("X contains infinite values")
    if X_values.isna().all(axis=0).any():
        cols = X_values.columns[X_values.isna().all(axis=0)].tolist()
        raise ValueError(f"feature columns are entirely missing: {cols}")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must be at least 2")
    if groups is not None:
        if len(groups) != len(y) or not groups.index.equals(y.index):
            raise ValueError("groups must share the index of X and y")
        if groups.isna().any():
            raise ValueError("groups has missing values")
        n_groups = groups.nunique()
        if n_groups < outer_splits:
            raise ValueError(f"{n_groups} groups are too few for {outer_splits} outer folds")
    elif len(y) < outer_splits:
        raise ValueError(f"{len(y)} samples are too few for {outer_splits} outer folds")

    outer = _splitter(outer_splits, groups, 0)
    smallest_inner_train = np.inf
    for train_idx, _ in outer.split(X, y, groups):
        g_tr = None if groups is None else groups.iloc[train_idx]
        n_inner_units = len(train_idx) if g_tr is None else g_tr.nunique()
        if n_inner_units < inner_splits:
            raise ValueError(
                f"an outer training fold has only {n_inner_units} groups/samples, fewer than {inner_splits} inner folds"
            )
        inner = _splitter(inner_splits, g_tr, 0)
        for inner_train, _ in inner.split(X.iloc[train_idx], y.iloc[train_idx], g_tr):
            smallest_inner_train = min(smallest_inner_train, len(inner_train))
    if candidates is not None:
        for candidate in candidates if isinstance(candidates, list) else [candidates]:
            k = candidate.max_n_neighbors()
            if k is not None and k > smallest_inner_train:
                raise ValueError(
                    f"{candidate.name}: n_neighbors can reach {k} but the smallest inner training fold has "
                    f"{int(smallest_inner_train)} rows; shrink the kNN search space"
                )


def _check_coverage(oof: pd.Series, folds: list[np.ndarray]) -> None:
    """Every sample must have exactly one finite out-of-fold prediction."""
    covered = np.concatenate(folds) if folds else np.array([], dtype=int)
    if len(covered) != len(oof) or not np.array_equal(np.sort(covered), np.arange(len(oof))):
        raise InvalidFitError("outer folds do not partition the samples exactly once")
    if oof.isna().any() or not np.isfinite(oof.to_numpy(dtype=float)).all():
        bad = oof.index[~np.isfinite(oof.to_numpy(dtype=float))].tolist()
        raise InvalidFitError(f"non-finite out-of-fold predictions for samples {bad}")


# ------------------------------------------------------------------------------ fitting


def _splitter(n_splits: int, groups: pd.Series | None, seed: int):
    if groups is None:
        return KFold(n_splits, shuffle=True, random_state=seed)
    if groups.nunique() < n_splits:
        raise ValueError(f"{groups.nunique()} groups are too few for {n_splits} folds")
    return GroupKFold(n_splits, shuffle=True, random_state=seed)


class _WarningCollector:
    """Record (instead of silence) every warning raised in a block, aggregated by message."""

    def __init__(self, result_ctx: dict, stage: str, fold: int):
        self.ctx, self.stage, self.fold = result_ctx, stage, fold
        self.records: list[WarningRecord] = []

    def __enter__(self):
        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        seen: dict[tuple[str, str], WarningRecord] = {}
        for w in self._log:
            key = (w.category.__name__, str(w.message).strip())
            if key in seen:
                seen[key].count += 1
            else:
                seen[key] = WarningRecord(
                    self.ctx["target"], self.ctx["model"], self.ctx["seed"], self.fold, self.stage, key[0], key[1]
                )
        self.records = list(seen.values())
        return False


def _clean_params(params: dict) -> dict:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def _tune(candidate: Candidate, X, y, cv, groups, n_iter: int, seed: int, ctx: dict, fold: int):
    """Randomised search with a valid-candidate policy.

    A drawn setting is valid only if it scored a finite value on every inner fold. Invalid
    settings are returned as ``FailureRecord`` entries. If no setting is valid the run
    raises ``InvalidFitError``. Returns (unfitted tuned pipeline, inner score, n_candidates, failures).
    """
    est = candidate.pipeline()
    if not candidate.param_distributions:
        scores = cross_val_score(est, X, y, cv=cv, groups=groups, scoring=SCORING, error_score="raise")
        return est, float(np.mean(scores)), 1, []

    search = RandomizedSearchCV(
        est, candidate.param_distributions, n_iter=n_iter, cv=cv, scoring=SCORING,
        random_state=seed, refit=False, error_score=np.nan,
    )
    search.fit(X, y, groups=groups)
    cvres = search.cv_results_
    n_splits = search.n_splits_
    split_scores = np.vstack([cvres[f"split{i}_test_score"] for i in range(n_splits)])  # (n_splits, n_cand)
    valid = np.isfinite(split_scores).all(axis=0)
    failures = []
    for i in np.flatnonzero(~valid):
        failures.append(FailureRecord(
            ctx["target"], ctx["model"], ctx["seed"], fold, "random search",
            _clean_params(dict(cvres["params"][i])), int((~np.isfinite(split_scores[:, i])).sum()),
            "non-finite inner-fold score (fit failed or produced non-finite predictions)",
        ))
    if not valid.any():
        raise InvalidFitError(
            f"{ctx['model']} / {ctx['target']} / seed {ctx['seed']} / fold {fold}: all {len(valid)} search "
            "candidates failed on at least one inner fold"
        )
    means = np.where(valid, cvres["mean_test_score"], -np.inf)
    best = int(np.argmax(means))
    tuned = clone(est).set_params(**cvres["params"][best])
    return tuned, float(means[best]), len(valid), failures


def _select(est, X, y, cv, groups, ctx: dict, fold: int) -> tuple[list[str], float]:
    """SFFS with the tuned (unfitted) pipeline; returns (features, inner score of that subset)."""
    sffs = SFS(clone(est), k_features="best", forward=True, floating=True, scoring=SCORING, cv=cv)
    sffs.fit(X, y, groups=groups)
    score = float(sffs.k_score_)
    chosen_scores = np.asarray(sffs.subsets_[len(sffs.k_feature_idx_)]["cv_scores"], dtype=float)
    if not np.isfinite(score) or not np.isfinite(chosen_scores).all():
        raise InvalidFitError(
            f"{ctx['model']} / {ctx['target']} / seed {ctx['seed']} / fold {fold}: SFFS selected a subset "
            "with a non-finite inner-fold score"
        )
    return list(sffs.k_feature_names_), score


def _final_fit_predict(est, X_tr, y_tr, X_te, chosen: list[str], ctx: dict, fold: int) -> np.ndarray:
    final = clone(est).fit(X_tr[chosen], y_tr)
    pred = np.asarray(final.predict(X_te[chosen]), dtype=float)
    if pred.shape != (len(X_te),) or not np.isfinite(pred).all():
        raise InvalidFitError(
            f"{ctx['model']} / {ctx['target']} / seed {ctx['seed']} / fold {fold}: final fit produced "
            "non-finite predictions"
        )
    return pred


def _fit_candidate_in_fold(candidate, X_tr, y_tr, inner, g_tr, n_iter, search_seed, ctx, fold, warn_out):
    """Tune, optionally select features, for one candidate inside one outer training fold."""
    ctx = {**ctx, "model": candidate.name}
    with _WarningCollector(ctx, "random search", fold) as wc:
        t0 = time.perf_counter()
        tuned, inner_score, n_cand, failures = _tune(candidate, X_tr, y_tr, inner, g_tr, n_iter, search_seed, ctx, fold)
        tune_s = time.perf_counter() - t0
    warn_out.extend(wc.records)
    chosen, select_s = list(X_tr.columns), 0.0
    if candidate.feature_selection:
        with _WarningCollector(ctx, "SFFS", fold) as wc:
            t0 = time.perf_counter()
            chosen, inner_score = _select(tuned, X_tr, y_tr, inner, g_tr, ctx, fold)
            select_s = time.perf_counter() - t0
        warn_out.extend(wc.records)
    params = _clean_params({k: v for k, v in tuned.get_params().items() if k in candidate.param_names()})
    return tuned, chosen, params, inner_score, n_cand, failures, tune_s, select_s


def _metrics(y_true: pd.Series, y_pred: pd.Series, folds: list[np.ndarray]) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),  # on pooled out-of-fold predictions
        "R2_fold_mean": float(np.mean([r2_score(y_true.iloc[f], y_pred.iloc[f]) for f in folds])),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": float(100 * mean_absolute_percentage_error(y_true, y_pred)),
    }


def _resolve_seeds(seed: int, fold_seed: int | None, search_seed: int | None) -> tuple[int, int]:
    return (seed if fold_seed is None else fold_seed), (seed if search_seed is None else search_seed)


def _finish(ctx, protocol, y, oof, folds, features, params, details, warns, fails, fold_seed, search_seed, t0, config):
    _check_coverage(oof, folds)
    fold_ids = pd.Series(-1, index=y.index, dtype=int)
    for i, f in enumerate(folds):
        fold_ids.iloc[f] = i
    groups = ctx.get("groups")
    groups = pd.Series(y.index.astype(str), index=y.index, name="group") if groups is None else groups.rename("group")
    return CVResult(
        ctx["model"], ctx["target"], protocol, ctx["seed"], _metrics(y, oof, folds), oof, features, params,
        fold_assignments=fold_ids, groups=groups, fold_details=details, warnings=warns, failures=fails,
        fold_seed=fold_seed, search_seed=search_seed, runtime_s=time.perf_counter() - t0, config=config,
    )


# ---------------------------------------------------------------------------- protocols


def nested_cv(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_iter: int = N_ITER,
    seed: int = 0,
    fold_seed: int | None = None,
    search_seed: int | None = None,
) -> CVResult:
    """Nested CV for one prespecified candidate: tuning and feature selection happen only
    inside each outer training fold.

    With ``groups``, outer and inner loops are ``GroupKFold`` (no group is split across
    folds); without, both are shuffled ``KFold``. ``fold_seed`` / ``search_seed`` default
    to ``seed`` and can be set separately to vary one randomness source at a time.
    """
    t0 = time.perf_counter()
    fold_seed, search_seed = _resolve_seeds(seed, fold_seed, search_seed)
    validate_inputs(X, y, groups, outer_splits, inner_splits, candidate)
    ctx = {"target": str(y.name), "model": candidate.name, "seed": seed, "groups": groups}
    outer = _splitter(outer_splits, groups, fold_seed)
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    features, params, folds, details, warns, fails = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
        X_tr, X_te, y_tr = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx]
        g_tr = None if groups is None else groups.iloc[train_idx]
        inner = _splitter(inner_splits, g_tr, fold_seed + 1000)
        tuned, chosen, p, inner_score, n_cand, f, tune_s, select_s = _fit_candidate_in_fold(
            candidate, X_tr, y_tr, inner, g_tr, n_iter, search_seed, ctx, fold, warns
        )
        with _WarningCollector(ctx, "final fit", fold) as wc:
            t1 = time.perf_counter()
            oof.iloc[test_idx] = _final_fit_predict(tuned, X_tr, y_tr, X_te, chosen, ctx, fold)
            fit_s = time.perf_counter() - t1
        warns.extend(wc.records)
        fails.extend(f)
        features.append(chosen)
        params.append(p)
        folds.append(test_idx)
        details.append(FoldDetail(fold, X.index[test_idx].tolist(), candidate.name, chosen, p, inner_score,
                                  n_cand, len(f), tune_s, select_s, fit_s))

    protocol = "nested, random KFold" if groups is None else "nested, GroupKFold"
    config = {"outer_splits": outer_splits, "inner_splits": inner_splits, "n_iter": n_iter,
              "feature_selection": candidate.feature_selection}
    return _finish(ctx, protocol, y, oof, folds, features, params, details, warns, fails, fold_seed, search_seed, t0, config)


def nested_cv_complete(
    candidates: list[Candidate],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_iter: int = N_ITER,
    seed: int = 0,
    fold_seed: int | None = None,
    search_seed: int | None = None,
) -> CVResult:
    """The complete selection procedure, evaluated honestly.

    Inside each outer training fold every candidate is tuned (and, if enabled, feature-
    selected) on the inner folds; the candidate whose final configuration has the best
    inner-CV score is refit and predicts the outer test fold. The outer score estimates how
    well "choose family + hyperparameters + features automatically" generalises; it is NOT
    the same as the best per-family outer score, which is selected after seeing the outer
    scores and is optimistic for that reason.
    """
    t0 = time.perf_counter()
    if not candidates:
        raise ValueError("candidates is empty")
    fold_seed, search_seed = _resolve_seeds(seed, fold_seed, search_seed)
    validate_inputs(X, y, groups, outer_splits, inner_splits, candidates)
    ctx = {"target": str(y.name), "model": COMPLETE_PROCEDURE, "seed": seed, "groups": groups}
    outer = _splitter(outer_splits, groups, fold_seed)
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    features, params, folds, details, warns, fails = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
        X_tr, X_te, y_tr = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx]
        g_tr = None if groups is None else groups.iloc[train_idx]
        inner = _splitter(inner_splits, g_tr, fold_seed + 1000)
        best, tune_s, select_s, n_cand_total = None, 0.0, 0.0, 0
        for candidate in candidates:
            tuned, chosen, p, inner_score, n_cand, f, ts, ss = _fit_candidate_in_fold(
                candidate, X_tr, y_tr, inner, g_tr, n_iter, search_seed, ctx, fold, warns
            )
            fails.extend(f)
            tune_s, select_s, n_cand_total = tune_s + ts, select_s + ss, n_cand_total + n_cand
            if best is None or inner_score > best[3]:
                best = (candidate, tuned, chosen, inner_score, p)
        candidate, tuned, chosen, inner_score, p = best
        fold_ctx = {**ctx, "model": f"{COMPLETE_PROCEDURE} -> {candidate.name}"}
        with _WarningCollector(fold_ctx, "final fit", fold) as wc:
            t1 = time.perf_counter()
            oof.iloc[test_idx] = _final_fit_predict(tuned, X_tr, y_tr, X_te, chosen, fold_ctx, fold)
            fit_s = time.perf_counter() - t1
        warns.extend(wc.records)
        features.append(chosen)
        params.append({"family": candidate.name, **p})
        folds.append(test_idx)
        n_failed = sum(1 for r in fails if r.fold == fold)
        details.append(FoldDetail(fold, X.index[test_idx].tolist(), candidate.name, chosen, p, inner_score,
                                  n_cand_total, n_failed, tune_s, select_s, fit_s))

    protocol = "nested, random KFold" if groups is None else "nested, GroupKFold"
    config = {"outer_splits": outer_splits, "inner_splits": inner_splits, "n_iter": n_iter,
              "families": [c.name for c in candidates]}
    return _finish(ctx, protocol, y, oof, folds, features, params, details, warns, fails, fold_seed, search_seed, t0, config)


def flat_cv(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    n_iter: int = N_ITER,
    seed: int = 0,
    fold_seed: int | None = None,
    search_seed: int | None = None,
) -> CVResult:
    """OPTIMISTIC re-implementation of the thesis protocol, kept only to measure its optimism.

    Tune and select features with ``n_splits``-fold random ``KFold`` on all rows, then score
    with CV on the same folds. ``groups`` is accepted for a uniform call signature and
    ignored: the thesis did not group platform twins. Not a valid generalisation estimate.
    """
    t0 = time.perf_counter()
    fold_seed, search_seed = _resolve_seeds(seed, fold_seed, search_seed)
    validate_inputs(X, y, None, n_splits, n_splits, candidate)
    ctx = {"target": str(y.name), "model": candidate.name, "seed": seed, "groups": groups}
    cv = KFold(n_splits, shuffle=True, random_state=fold_seed)
    warns, fails = [], []
    tuned, chosen, params, inner_score, n_cand, f, tune_s, select_s = _fit_candidate_in_fold(
        candidate, X, y, cv, None, n_iter, search_seed, ctx, -1, warns
    )
    fails.extend(f)
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    folds, details = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        with _WarningCollector(ctx, "final fit", fold) as wc:
            t1 = time.perf_counter()
            oof.iloc[test_idx] = _final_fit_predict(tuned, X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx],
                                                    chosen, ctx, fold)
            fit_s = time.perf_counter() - t1
        warns.extend(wc.records)
        folds.append(test_idx)
        details.append(FoldDetail(fold, X.index[test_idx].tolist(), candidate.name, chosen, params, inner_score,
                                  n_cand, len(f), tune_s if fold == 0 else 0.0, select_s if fold == 0 else 0.0, fit_s))
    config = {"n_splits": n_splits, "n_iter": n_iter, "feature_selection": candidate.feature_selection,
              "optimistic": True}
    return _finish(ctx, "flat (thesis re-implementation)", y, oof, folds, [chosen], [params], details, warns, fails,
                   fold_seed, search_seed, t0, config)


# -------------------------------------------------------------------------- repetition


def _run_one(protocol, candidate, X, y, groups, seed, kwargs) -> CVResult:
    # Nothing is silenced or skipped here: warnings are captured per fold inside the
    # protocol and returned on the result, and sklearn validates every input as usual.
    return protocol(candidate, X, y, groups, seed=seed, **kwargs)


def run_repeated(protocol, candidates, dataset, targets, seeds=SEEDS, group=True, n_jobs=-1, **kwargs):
    """Run ``protocol`` for every target, model and seed in parallel.

    ``protocol`` is ``nested_cv`` or ``flat_cv`` (``candidates`` is a list of ``Candidate``),
    or ``nested_cv_complete`` (``candidates`` is the list of families to choose among, run
    once per target and seed). Extra keyword arguments (e.g. ``fold_seed=0``) are passed
    through, so ``run_repeated(nested_cv, ..., fold_seed=0)`` varies only the search seed.
    """
    jobs = []
    for target in targets:
        X, y, groups = dataset.target(target)
        g = groups if group else None
        if protocol is nested_cv_complete:
            for seed in seeds:
                jobs.append(delayed(_run_one)(protocol, list(candidates), X, y, g, seed, kwargs))
        else:
            for candidate in candidates:
                for seed in seeds:
                    jobs.append(delayed(_run_one)(protocol, candidate, X, y, g, seed, kwargs))
    return Parallel(n_jobs=n_jobs)(jobs)


# ---------------------------------------------------------------------------- summaries


def results_table(results: list[CVResult], metrics=("R2", "RMSE", "MAE", "MAPE_%")) -> pd.DataFrame:
    """Mean and standard deviation over seeds for each (target, model).

    The SD is the spread over seeds, each of which changes both the fold assignment and the
    random-search draws. It is not a confidence interval for the generalisation error.
    """
    rows = [{"target": r.target, "model": r.model, "seed": r.seed, **r.metrics} for r in results]
    df = pd.DataFrame(rows)
    return df.groupby(["target", "model"], sort=False)[list(metrics)].agg(["mean", "std"])


def selection_frequency(results: CVResult | list[CVResult], features: list[str]) -> pd.Series:
    """Share of outer folds (over all given runs) in which the feature was kept."""
    results = results if isinstance(results, list) else [results]
    counts = pd.Series(0, index=features, dtype=float)
    n = 0
    for result in results:
        for chosen in result.selected_features:
            counts[chosen] += 1
            n += 1
    return counts / n


def jaccard_stability(feature_sets: list[list[str]]) -> float:
    """Mean pairwise Jaccard similarity of selected-feature sets (1 = identical in every fold)."""
    sets = [set(s) for s in feature_sets]
    pairs = list(combinations(sets, 2))
    if not pairs:
        return float("nan")
    return float(np.mean([len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in pairs]))


def selection_stability(results: list[CVResult]) -> pd.DataFrame:
    """Per (target, model): feature-subset stability across all outer folds and runtime.

    Columns: ``n_folds``, ``mean_n_features``, ``jaccard`` (mean pairwise Jaccard similarity
    of the selected subsets), ``runtime_s_mean`` per run, ``select_share`` (fraction of fit
    time spent in SFFS), ``n_warnings`` and ``n_failed_candidates`` summed over runs.
    """
    rows = []
    keys = sorted({(r.target, r.model) for r in results}, key=lambda k: (k[0], k[1]))
    for target, model in keys:
        rs = [r for r in results if r.target == target and r.model == model]
        sets = [s for r in rs for s in r.selected_features]
        tune = sum(d.tune_s for r in rs for d in r.fold_details)
        select = sum(d.select_s for r in rs for d in r.fold_details)
        fit = sum(d.fit_s for r in rs for d in r.fold_details)
        rows.append({
            "target": target, "model": model, "n_runs": len(rs), "n_folds": len(sets),
            "mean_n_features": float(np.mean([len(s) for s in sets])), "jaccard": jaccard_stability(sets),
            "runtime_s_mean": float(np.mean([r.runtime_s for r in rs])),
            "select_share": select / (tune + select + fit) if (tune + select + fit) > 0 else float("nan"),
            "n_warnings": int(sum(w.count for r in rs for w in r.warnings)),
            "n_failed_candidates": int(sum(len(r.failures) for r in rs)),
        })
    return pd.DataFrame(rows).set_index(["target", "model"])
