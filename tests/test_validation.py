"""Malformed inputs, missing targets, non-finite data, insufficient groups, infeasible k."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor

from smalldata import Candidate, Dataset, load_dataset, nested_cv, validate_inputs
from smalldata.data import FEATURES, TARGETS
from smalldata.modeling import _M

from conftest import SMALL, make_grouped_data


def _csv(tmp_path, df: pd.DataFrame) -> str:
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return str(p)


def _frame(n=6):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES) + len(TARGETS))), columns=FEATURES + TARGETS)
    df.insert(0, "vehicle_id", [f"v{i}" for i in range(n)])
    return df


def test_load_dataset_requires_columns(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(_csv(tmp_path, _frame().drop(columns=["height_in"])))


def test_load_dataset_rejects_non_numeric(tmp_path):
    df = _frame()
    df["length_in"] = df["length_in"].astype(object)
    df.loc[2, "length_in"] = "tall"
    with pytest.raises(ValueError, match="Non-numeric"):
        load_dataset(_csv(tmp_path, df))


def test_load_dataset_rejects_infinite(tmp_path):
    df = _frame()
    df.loc[1, "wheelbase_in"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        load_dataset(_csv(tmp_path, df))


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    df = _frame()
    df.loc[1, "vehicle_id"] = df.loc[0, "vehicle_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(_csv(tmp_path, df))


def test_missing_target_rows_are_dropped_per_target(tmp_path):
    df = _frame()
    df.loc[[0, 3], "towing_capacity_lb"] = np.nan
    ds = load_dataset(_csv(tmp_path, df))
    X, y, g = ds.target("towing_capacity_lb")
    assert len(y) == 4 and y.notna().all() and X.index.equals(y.index) and g.index.equals(y.index)
    assert len(ds.target("height_in" if False else "cargo_volume_cuft")[1]) == 6


def test_missing_group_falls_back_to_row_id(tmp_path):
    df = _frame()
    df["platform_group"] = ["twins", "twins", None, None, "x", None]
    ds = load_dataset(_csv(tmp_path, df))
    assert ds.groups.nunique() == 5 and ds.groups.notna().all()


def test_validate_rejects_missing_target():
    X, y, g = make_grouped_data()
    y = y.copy()
    y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_inputs(X, y, g, 3, 3)


def test_validate_rejects_nonfinite_features():
    X, y, g = make_grouped_data()
    X = X.copy()
    X.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_inputs(X, y, g, 3, 3)


def test_validate_rejects_misaligned_index():
    X, y, g = make_grouped_data()
    with pytest.raises(ValueError, match="same index"):
        validate_inputs(X.iloc[:-1], y, g, 3, 3)
    with pytest.raises(ValueError, match="groups must share"):
        validate_inputs(X, y, g.reset_index(drop=True), 3, 3)


def test_validate_rejects_insufficient_groups_for_outer_loop():
    X, y, g = make_grouped_data(n_groups=4)
    with pytest.raises(ValueError, match="too few for 5 outer folds"):
        validate_inputs(X, y, g, 5, 2)


def test_validate_rejects_insufficient_groups_for_inner_loop():
    X, y, g = make_grouped_data(n_groups=5)
    # 5 groups, 3 outer folds -> an outer training fold holds 3 groups, fewer than 4 inner folds
    with pytest.raises(ValueError, match="inner folds"):
        validate_inputs(X, y, g, 3, 4)


def test_validate_rejects_constant_target():
    X, y, g = make_grouped_data()
    with pytest.raises(ValueError, match="constant"):
        validate_inputs(X, pd.Series(1.0, index=y.index, name="t"), g, 3, 3)


def test_infeasible_knn_k_is_refused_before_fitting():
    X, y, g = make_grouped_data()  # 24 rows, smallest inner training fold ~ 10 rows
    knn = Candidate("kNN", KNeighborsRegressor(), {_M + "n_neighbors": randint(1, 51)}, feature_selection=False)
    assert knn.max_n_neighbors() == 50
    with pytest.raises(ValueError, match="n_neighbors can reach 50"):
        nested_cv(knn, X, y, g, seed=0, **SMALL)


def test_feasible_knn_runs():
    X, y, g = make_grouped_data()
    knn = Candidate("kNN", KNeighborsRegressor(), {_M + "n_neighbors": randint(1, 4)}, feature_selection=False)
    res = nested_cv(knn, X, y, g, seed=0, **SMALL)
    assert res.predictions.notna().all()
    assert all(p[_M + "n_neighbors"] <= 3 for p in res.params)


def test_dataset_target_shape():
    X, y, g = make_grouped_data()
    Y = pd.DataFrame({"t": y})
    ds = Dataset(X=X, Y=Y, groups=g)
    assert ds.n_samples == 24
    Xt, yt, gt = ds.target("t")
    assert Xt.shape == (24, 4) and len(yt) == 24 and len(gt) == 24
