"""Persist complete experiment artifacts for a list of ``CVResult``.

``save_run(results, out_dir, ...)`` writes:

* ``predictions.csv``   one row per (run, sample): out-of-fold prediction, truth, fold id, group
* ``folds.csv``         fold assignment and group id per (run, sample), i.e. the exact splits
* ``selections.csv``    per (run, outer fold): chosen family, features, hyperparameters, inner
                        score, number of search candidates and failed candidates, stage timings
* ``warnings.csv``      every warning captured during fitting, with target/model/seed/fold/stage
* ``failures.csv``      search candidates excluded because an inner fold failed
* ``metrics.csv``       per-run aggregate metrics (R2, RMSE, MAE, MAPE_%, R2_fold_mean, runtime)
* ``summary.csv``       mean and SD over seeds per (protocol, target, model)
* ``stability.csv``     selected-feature stability (Jaccard, subset size) and runtime per (target, model)
* ``run_meta.json``     configuration, data fingerprint (sha256), git commit, dependency versions

``load_run(out_dir)`` reads the CSV/JSON tables back as DataFrames and a dict.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import file_sha256
from .modeling import CVResult, results_table, selection_stability

TABLES = ("predictions", "folds", "selections", "warnings", "failures", "metrics", "summary", "stability")


def _run_key(r: CVResult) -> dict:
    return {"protocol": r.protocol, "target": r.target, "model": r.model, "seed": r.seed,
            "fold_seed": r.fold_seed, "search_seed": r.search_seed}


def predictions_frame(results: list[CVResult], y_by_target: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    rows = []
    for r in results:
        y = None if y_by_target is None else y_by_target.get(r.target)
        for sample, pred in r.predictions.items():
            rows.append({**_run_key(r), "sample_id": sample, "group": r.groups.get(sample),
                         "fold": int(r.fold_assignments.get(sample)),
                         "y_true": None if y is None else float(y.loc[sample]), "y_pred": float(pred)})
    return pd.DataFrame(rows)


def folds_frame(results: list[CVResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for sample, fold in r.fold_assignments.items():
            rows.append({**_run_key(r), "sample_id": sample, "group": r.groups.get(sample), "fold": int(fold)})
    return pd.DataFrame(rows)


def selections_frame(results: list[CVResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for d in r.fold_details:
            rows.append({**_run_key(r), "fold": d.fold, "family": d.family, "n_features": len(d.features),
                         "features": "|".join(d.features), "params": json.dumps(d.params, default=str),
                         "inner_score": d.inner_score, "n_search_candidates": d.n_search_candidates,
                         "n_failed_candidates": d.n_failed_candidates, "tune_s": d.tune_s,
                         "select_s": d.select_s, "fit_s": d.fit_s})
    return pd.DataFrame(rows)


def warnings_frame(results: list[CVResult]) -> pd.DataFrame:
    cols = ["protocol", "target", "model", "seed", "fold", "stage", "category", "message", "count"]
    rows = [{"protocol": r.protocol, "target": w.target, "model": w.model, "seed": w.seed, "fold": w.fold,
             "stage": w.stage, "category": w.category, "message": w.message, "count": w.count}
            for r in results for w in r.warnings]
    return pd.DataFrame(rows, columns=cols)


def failures_frame(results: list[CVResult]) -> pd.DataFrame:
    cols = ["protocol", "target", "model", "seed", "fold", "stage", "params", "n_failed_inner_folds", "error"]
    rows = [{"protocol": r.protocol, "target": f.target, "model": f.model, "seed": f.seed, "fold": f.fold,
             "stage": f.stage, "params": json.dumps(f.params, default=str),
             "n_failed_inner_folds": f.n_failed_inner_folds, "error": f.error}
            for r in results for f in r.failures]
    return pd.DataFrame(rows, columns=cols)


def metrics_frame(results: list[CVResult]) -> pd.DataFrame:
    return pd.DataFrame([{**_run_key(r), **r.metrics, "runtime_s": r.runtime_s,
                          "n_warnings": sum(w.count for w in r.warnings), "n_failed_candidates": len(r.failures)}
                         for r in results])


def summary_frame(results: list[CVResult]) -> pd.DataFrame:
    frames = []
    for protocol in sorted({r.protocol for r in results}):
        sub = [r for r in results if r.protocol == protocol]
        t = results_table(sub, metrics=("R2", "R2_fold_mean", "RMSE", "MAE", "MAPE_%"))
        t.columns = [f"{m}_{s}" for m, s in t.columns]
        t = t.reset_index()
        t.insert(0, "protocol", protocol)
        t["n_seeds"] = [sum(1 for r in sub if r.target == a and r.model == b) for a, b in zip(t["target"], t["model"])]
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def stability_frame(results: list[CVResult]) -> pd.DataFrame:
    frames = []
    for protocol in sorted({r.protocol for r in results}):
        t = selection_stability([r for r in results if r.protocol == protocol]).reset_index()
        t.insert(0, "protocol", protocol)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def dependency_versions() -> dict[str, str]:
    import joblib
    import mlxtend
    import numpy
    import scipy
    import sklearn

    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "mlxtend": mlxtend.__version__,
            "joblib": joblib.__version__}


def save_run(results: list[CVResult], out_dir: str | Path, config: dict, data_path: str | Path | None = None,
             y_by_target: dict[str, pd.Series] | None = None, repo_root: str | Path | None = None) -> Path:
    """Write all artifact tables and ``run_meta.json`` to ``out_dir`` (created if needed)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError("no results to save")
    predictions_frame(results, y_by_target).to_csv(out / "predictions.csv", index=False, float_format="%.7g")
    folds_frame(results).to_csv(out / "folds.csv", index=False, float_format="%.7g")
    selections_frame(results).to_csv(out / "selections.csv", index=False, float_format="%.7g")
    warnings_frame(results).to_csv(out / "warnings.csv", index=False, float_format="%.7g")
    failures_frame(results).to_csv(out / "failures.csv", index=False, float_format="%.7g")
    metrics_frame(results).to_csv(out / "metrics.csv", index=False, float_format="%.7g")
    summary_frame(results).to_csv(out / "summary.csv", index=False, float_format="%.7g")
    stability_frame(results).to_csv(out / "stability.csv", index=False, float_format="%.7g")

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "data": None if data_path is None else {"path": str(data_path), "sha256": file_sha256(data_path)},
        "git_commit": git_commit(root),
        "versions": dependency_versions(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "n_runs": len(results),
        "n_warnings": int(sum(w.count for r in results for w in r.warnings)),
        "n_failed_candidates": int(sum(len(r.failures) for r in results)),
        "total_runtime_s": float(sum(r.runtime_s for r in results)),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return out


def load_run(out_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read the artifact tables and ``run_meta.json``.

    ``predictions.csv``, ``metrics.csv`` and ``run_meta.json`` are required. The other tables
    are optional so that a published copy can omit the redundant ``folds.csv`` (its fold and
    group columns are also in ``predictions.csv``); missing optional tables are returned empty.
    """
    out = Path(out_dir)
    tables = {}
    for name in TABLES:
        path = out / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
        elif name in ("predictions", "metrics"):
            raise FileNotFoundError(path)
        else:
            tables[name] = pd.DataFrame()
    meta = json.loads((out / "run_meta.json").read_text())
    return tables, meta


def check_run(out_dir: str | Path, expected_n: dict[str, int] | None = None) -> dict:
    """Verify a saved run: finite metrics and predictions, complete out-of-fold coverage per run.

    Raises ``AssertionError`` with a precise message on the first violation. Returns a small
    dict of counts for logging. ``expected_n`` maps target -> number of evaluated samples.
    """
    tables, meta = load_run(out_dir)
    pred, metrics = tables["predictions"], tables["metrics"]
    assert len(metrics) > 0, "metrics.csv is empty"
    for col in ("R2", "RMSE", "MAE"):
        bad = metrics[~metrics[col].apply(lambda v: pd.notna(v) and abs(v) != float("inf"))]
        assert bad.empty, f"non-finite {col} in metrics.csv:\n{bad[['target', 'model', 'seed', col]]}"
    assert pred["y_pred"].notna().all() and (pred["y_pred"].abs() != float("inf")).all(), "non-finite predictions"
    keys = ["protocol", "target", "model", "seed", "fold_seed", "search_seed"]
    for key, sub in pred.groupby(keys, sort=False):
        assert sub["sample_id"].is_unique, f"duplicate out-of-fold predictions in run {key}"
        if expected_n is not None:
            target = key[1]
            assert len(sub) == expected_n[target], f"run {key} covers {len(sub)} of {expected_n[target]} samples"
        assert (sub["fold"] >= 0).all(), f"run {key} has samples without a fold"
    return {"n_runs": len(metrics), "n_prediction_rows": len(pred), "n_warnings": meta["n_warnings"],
            "n_failed_candidates": meta["n_failed_candidates"]}
