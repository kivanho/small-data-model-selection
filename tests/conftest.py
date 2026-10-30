"""Small synthetic fixtures: 24 samples in 12 twin groups, 4 features, a linear target."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from smalldata import Candidate
from smalldata.modeling import _M

N_GROUPS = 12
FEATURES = ["f1", "f2", "f3", "f4"]


def make_grouped_data(seed: int = 0, n_groups: int = N_GROUPS, missing: bool = True):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_groups, len(FEATURES)))
    # each group is a pair of near-identical "twins"
    X = np.vstack([base + rng.normal(scale=0.02, size=base.shape), base + rng.normal(scale=0.02, size=base.shape)])
    groups = np.array([f"g{i}" for i in range(n_groups)] * 2)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.1, size=len(X))
    index = pd.Index([f"s{i:02d}" for i in range(len(X))], name="sample")
    Xdf = pd.DataFrame(X, columns=FEATURES, index=index)
    if missing:
        Xdf.iloc[[i for i in (1, 7, 15) if i < len(Xdf)], 3] = np.nan
    return Xdf, pd.Series(y, index=index, name="target"), pd.Series(groups, index=index, name="group")


@pytest.fixture
def grouped_data():
    return make_grouped_data()


@pytest.fixture
def linear_candidate():
    """Linear regression without SFS: the fastest exercised candidate."""
    return Candidate("Linear regression", LinearRegression(), feature_selection=False)


@pytest.fixture
def ridge_candidate():
    return Candidate("Ridge", Ridge(), {_M + "alpha": [0.01, 0.1, 1.0, 10.0]}, feature_selection=False)


@pytest.fixture
def sfs_candidate():
    return Candidate("Linear regression + SFS", LinearRegression(), feature_selection=True)


SMALL = dict(outer_splits=3, inner_splits=3, n_iter=3)
