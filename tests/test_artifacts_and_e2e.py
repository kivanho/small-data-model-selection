"""Metric calculations, artifact export round-trip, saved-run verification, and a reduced
end-to-end grouped nested-CV run on the bundled dataset through the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smalldata import (
    TARGETS,
    baseline_candidates,
    check_run,
    default_candidates,
    file_sha256,
    load_dataset,
    load_run,
    nested_cv,
    results_table,
    save_run,
    selection_frequency,
    selection_stability,
)
from smalldata.modeling import _metrics
from smalldata.run import main

from conftest import SMALL

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "midsize_sedans.csv"


def test_metrics_match_hand_calculation():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    p = pd.Series([1.0, 2.0, 2.0, 5.0])
    folds = [np.array([0, 1]), np.array([2, 3])]
    m = _metrics(y, p, folds)
    assert np.isclose(m["RMSE"], np.sqrt(2 / 4))
    assert np.isclose(m["MAE"], 0.5)
    assert np.isclose(m["R2"], 1 - 2.0 / 5.0)
    assert np.isclose(m["MAPE_%"], 100 * np.mean([0, 0, 1 / 3, 1 / 4]))
    # fold 0 is perfect; fold 1 has SS_res = 2, SS_tot = 0.5 -> R2 = -3
    assert np.isclose(m["R2_fold_mean"], (1.0 + (1 - 2 / 0.5)) / 2)


def test_results_table_and_selection_frequency(grouped_data, linear_candidate, sfs_candidate):
    X, y, g = grouped_data
    runs = [nested_cv(linear_candidate, X, y, g, seed=s, **SMALL) for s in (0, 1)]
    table = results_table(runs, metrics=("R2",))
    assert table.loc[("target", "Linear regression"), ("R2", "mean")] == pytest.approx(np.mean([r.metrics["R2"] for r in runs]))
    freq = selection_frequency(runs, list(X.columns))
    assert (freq == 1.0).all()  # no SFS: every feature kept in every fold
    sfs = nested_cv(sfs_candidate, X, y, g, seed=0, **SMALL)
    stab = selection_stability([sfs])
    assert stab.loc[("target", sfs_candidate.name), "n_folds"] == SMALL["outer_splits"]
    assert 0 <= stab.loc[("target", sfs_candidate.name), "jaccard"] <= 1


def test_save_and_load_run_round_trip(tmp_path, grouped_data, linear_candidate, ridge_candidate):
    X, y, g = grouped_data
    runs = [nested_cv(c, X, y, g, seed=0, **SMALL) for c in (linear_candidate, ridge_candidate)]
    csv = tmp_path / "data.csv"
    pd.concat([X, y], axis=1).to_csv(csv)
    out = save_run(runs, tmp_path / "run", {"note": "test"}, data_path=csv, y_by_target={"target": y}, repo_root=ROOT)
    tables, meta = load_run(out)
    assert set(tables) == {"predictions", "folds", "selections", "warnings", "failures", "metrics", "summary", "stability"}
    pred = tables["predictions"]
    assert len(pred) == 2 * len(y)
    assert set(pred.columns) >= {"protocol", "target", "model", "seed", "fold_seed", "search_seed", "sample_id", "group", "fold", "y_true", "y_pred"}
    lin = pred[pred["model"] == "Linear regression"].set_index("sample_id")
    np.testing.assert_allclose(lin.loc[y.index, "y_pred"], runs[0].predictions.to_numpy(), rtol=1e-6)  # 7 significant digits
    np.testing.assert_allclose(lin.loc[y.index, "y_true"], y.to_numpy(), rtol=1e-6)
    assert tables["selections"]["params"].map(json.loads).notna().all()
    assert meta["data"]["sha256"] == file_sha256(csv) and len(meta["data"]["sha256"]) == 64
    assert meta["versions"]["scikit-learn"] and meta["config"] == {"note": "test"}
    assert "git_commit" in meta
    assert check_run(out, expected_n={"target": len(y)})["n_runs"] == 2
    # a published copy may omit the redundant folds table; predictions and metrics are required
    (out / "folds.csv").unlink()
    tables, _ = load_run(out)
    assert tables["folds"].empty and check_run(out, expected_n={"target": len(y)})["n_runs"] == 2
    (out / "metrics.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_run(out)


def test_check_run_detects_nonfinite_and_incomplete(tmp_path, grouped_data, linear_candidate):
    X, y, g = grouped_data
    run = nested_cv(linear_candidate, X, y, g, seed=0, **SMALL)
    out = save_run([run], tmp_path / "run", {}, y_by_target={"target": y}, repo_root=ROOT)
    pred = pd.read_csv(out / "predictions.csv")
    pred.loc[0, "y_pred"] = np.nan
    pred.to_csv(out / "predictions.csv", index=False)
    with pytest.raises(AssertionError, match="non-finite predictions"):
        check_run(out)
    pred = pd.read_csv(out / "predictions.csv").dropna()
    pred.to_csv(out / "predictions.csv", index=False)
    with pytest.raises(AssertionError, match="covers 23 of 24"):
        check_run(out, expected_n={"target": len(y)})


def test_default_and_baseline_candidates():
    names = [c.name for c in default_candidates(31)]
    assert names == ["Linear regression", "Lasso", "kNN", "SVR"]
    assert all(c.feature_selection for c in default_candidates(31))
    base = baseline_candidates()
    assert [c.name for c in base] == ["Mean", "Ridge (all features)", "Lasso (no SFS)"]
    assert not any(c.feature_selection for c in base)
    assert default_candidates(31)[2].max_n_neighbors() == 8
    assert default_candidates(10)[2].max_n_neighbors() == 4


@pytest.mark.slow
def test_reduced_end_to_end_grouped_nested_cv_cli(tmp_path):
    """Smoke-sized run on the real CSV: all families, baselines and the complete procedure
    on identical outer folds, every sample covered, artifacts verified."""
    out = tmp_path / "smoke"
    rc = main(["--preset", "smoke", "--out", str(out), "--n-jobs", "2"])
    assert rc == 0
    tables, meta = load_run(out)
    ds = load_dataset(DATA)
    n = len(ds.target("cargo_volume_cuft")[1])
    report = check_run(out, expected_n={"cargo_volume_cuft": n})
    models = set(tables["metrics"]["model"])
    assert models == {"Linear regression", "Lasso", "kNN", "SVR", "Mean", "Ridge (all features)", "Lasso (no SFS)",
                      "Complete procedure"}
    assert report["n_runs"] == 8 * 2
    assert len(tables["predictions"]) == 8 * 2 * n
    # identical outer folds across every model for a given seed
    folds = tables["folds"]
    for seed, sub in folds.groupby("seed"):
        pivot = sub.pivot(index="sample_id", columns="model", values="fold")
        assert (pivot.nunique(axis=1) == 1).all(), "models were not scored on identical outer folds"
        # platform twins never split
        gf = sub[sub["model"] == "Lasso"].groupby("group")["fold"].nunique()
        assert (gf == 1).all()
    assert meta["data"]["sha256"] == file_sha256(DATA)
    assert meta["config"]["preset"] == "smoke" and meta["config"]["targets"] == ["cargo_volume_cuft"]
    sel = tables["selections"]
    complete = sel[sel["model"] == "Complete procedure"]
    assert set(complete["family"]) <= {"Linear regression", "Lasso", "kNN", "SVR"}
    assert (sel[sel["model"] == "Mean"]["n_features"] == 9).all()  # baselines keep all features
    assert tables["summary"]["R2_mean"].notna().all()
    assert set(TARGETS) >= set(tables["metrics"]["target"])
