"""Command-line entry point for reproducible experiments.

Examples (from the repository root, with ``src`` on ``PYTHONPATH`` or the package installed)::

    # smoke run used by CI: one target, 2 seeds, 3 outer / 3 inner folds, 4 search draws
    python -m smalldata.run --preset smoke --out results/runs/smoke

    # the published benchmark: all targets, families + baselines + complete procedure,
    # nested GroupKFold, seeds 0-4, plus the optimistic flat and random-KFold comparisons
    python -m smalldata.run --preset benchmark --out results/benchmark

    # one target, main protocol only
    python -m smalldata.run --target cargo_volume_cuft --protocol nested-group --out results/runs/cargo

    # separate the two randomness sources: fixed folds, varying search seed (and vice versa)
    python -m smalldata.run --vary search --fold-seed 0 --seeds 0 1 2 3 4 --out results/runs/vary_search
    python -m smalldata.run --vary folds --search-seed 0 --seeds 0 1 2 3 4 --out results/runs/vary_folds

Every run writes the artifact tables listed in ``smalldata.artifacts`` and then verifies them
(finite values, complete out-of-fold coverage), exiting non-zero on any violation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .artifacts import check_run, save_run
from .data import TARGETS, load_dataset
from .modeling import (
    COMPLETE_PROCEDURE,
    N_ITER,
    SEEDS,
    baseline_candidates,
    default_candidates,
    flat_cv,
    nested_cv,
    nested_cv_complete,
    run_repeated,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "midsize_sedans.csv"
PROTOCOLS = {
    "nested-group": (nested_cv, True),
    "nested-kfold": (nested_cv, False),
    "flat": (flat_cv, False),  # optimistic thesis re-implementation, comparison only
}
MODEL_GROUPS = {"families": default_candidates, "baselines": lambda n: baseline_candidates()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m smalldata.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with the 14 required columns")
    p.add_argument("--out", required=True, help="output directory for the artifact tables")
    p.add_argument("--target", nargs="+", default=["all"], help="target column(s) or 'all'")
    p.add_argument("--models", nargs="+", default=["families", "baselines"],
                   help="'families', 'baselines' and/or individual candidate names")
    p.add_argument("--protocol", nargs="+", default=["nested-group"], choices=list(PROTOCOLS),
                   help="nested-group is the main protocol; nested-kfold and flat are comparisons")
    p.add_argument("--complete", action=argparse.BooleanOptionalAction, default=True,
                   help="also run the complete selection procedure over the families (nested-group only)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--vary", choices=["both", "folds", "search"], default="both",
                   help="which randomness source the seeds change; the other is fixed by --fold-seed/--search-seed")
    p.add_argument("--fold-seed", type=int, default=0, help="fixed fold seed when --vary search")
    p.add_argument("--search-seed", type=int, default=0, help="fixed search seed when --vary folds")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=N_ITER)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--preset", choices=["smoke", "benchmark"], help="override options with a named configuration")
    return p


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "smoke":
        args.target = ["cargo_volume_cuft"]
        args.models, args.protocol = ["families", "baselines"], ["nested-group"]
        args.seeds, args.outer_splits, args.inner_splits, args.n_iter = [0, 1], 3, 3, 4
        args.complete = True
    elif args.preset == "benchmark":
        args.target, args.models = ["all"], ["families", "baselines"]
        args.protocol, args.complete, args.seeds = ["nested-group", "nested-kfold", "flat"], True, list(SEEDS)
        args.outer_splits, args.inner_splits, args.n_iter = 5, 5, N_ITER
    return args


def select_candidates(names: list[str], n_samples: int):
    pool = default_candidates(n_samples) + baseline_candidates()
    chosen, seen = [], set()
    for name in names:
        picks = MODEL_GROUPS[name](n_samples) if name in MODEL_GROUPS else [c for c in pool if c.name == name]
        if not picks:
            raise SystemExit(f"unknown model '{name}'; choose from {[c.name for c in pool]} or {list(MODEL_GROUPS)}")
        for c in picks:
            if c.name not in seen:
                chosen.append(c)
                seen.add(c.name)
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = apply_preset(build_parser().parse_args(argv))
    ds = load_dataset(args.data)
    targets = list(TARGETS) if args.target == ["all"] else args.target
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; choose from {TARGETS}")
    candidates = select_candidates(args.models, ds.n_samples)
    families = [c for c in candidates if c.feature_selection]

    seed_kwargs = {}
    if args.vary == "search":
        seed_kwargs["fold_seed"] = args.fold_seed
    elif args.vary == "folds":
        seed_kwargs["search_seed"] = args.search_seed
    common = dict(seeds=args.seeds, n_jobs=args.n_jobs, n_iter=args.n_iter, **seed_kwargs)

    results, timings = [], {}
    for name in args.protocol:
        protocol, group = PROTOCOLS[name]
        t0 = time.perf_counter()
        if protocol is flat_cv:
            results += run_repeated(flat_cv, candidates, ds, targets, group=False, n_splits=args.outer_splits, **common)
        else:
            results += run_repeated(nested_cv, candidates, ds, targets, group=group, outer_splits=args.outer_splits,
                                    inner_splits=args.inner_splits, **common)
            if args.complete and name == "nested-group" and families:
                results += run_repeated(nested_cv_complete, families, ds, targets, group=True,
                                        outer_splits=args.outer_splits, inner_splits=args.inner_splits, **common)
        timings[name] = round(time.perf_counter() - t0, 1)
        print(f"{name}: {len(results)} runs so far, {timings[name]} s", file=sys.stderr)

    config = {k: v for k, v in vars(args).items()}
    config["targets"], config["candidates"], config["timings_s"] = targets, [c.name for c in candidates], timings
    config["complete_procedure_families"] = [c.name for c in families] if args.complete else []
    y_by_target = {t: ds.target(t)[1] for t in targets}
    out = save_run(results, args.out, config, data_path=args.data, y_by_target=y_by_target, repo_root=ROOT)
    report = check_run(out, expected_n={t: len(y) for t, y in y_by_target.items()})
    print(f"wrote {out} : {report}", file=sys.stderr)
    if report["n_failed_candidates"]:
        print(f"note: {report['n_failed_candidates']} search candidates were excluded (see failures.csv)", file=sys.stderr)
    _ = COMPLETE_PROCEDURE
    return 0


if __name__ == "__main__":
    sys.exit(main())
