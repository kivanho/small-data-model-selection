# Small-data model selection: how reliable is the evaluation?

[![CI](https://github.com/kivanho/small-data-model-selection/actions/workflows/ci.yml/badge.svg)](https://github.com/kivanho/small-data-model-selection/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-142E46)](https://www.python.org/)
[![scikit-learn 1.9](https://img.shields.io/badge/scikit--learn-1.9-087E85)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-4F6F94)](LICENSE)

## 1. The engineering question

With about 30 vehicles, 9 design variables and badge-engineered twins in the data, which
regression results can be trusted, and how much of a reported score is produced by the
evaluation protocol rather than by the model? Concretely:

* How much does reusing the scoring folds for tuning and feature selection inflate R²?
* How much do platform twins on both sides of a split inflate it further?
* Does sequential feature selection (SFFS) add accuracy, or mainly runtime and instability,
  compared with ridge or Lasso on all features?
* What does the *complete* procedure (choose family, hyperparameters and feature subset
  automatically) achieve, as opposed to the best family picked after seeing the scores?

The answer on this dataset is a mostly negative result: the honest scores are low, the
families are indistinguishable within the seed spread, SFFS does not beat the no-selection
baselines, and the complete procedure scores below the post-hoc best family on every target.

## 2. What the repository implements

`src/smalldata/` is a small library plus a CLI, tested with pytest:

| Piece | What it does |
|---|---|
| `data.py` | CSV loader with schema, numeric, finite and duplicate-id checks; rows with a missing target are dropped per target; `platform_group` defines CV groups; sha256 fingerprint |
| `modeling.py` | `nested_cv` (repeated nested `GroupKFold`, tuning + SFFS inside the outer training fold), `nested_cv_complete` (family + hyperparameters + features chosen in the inner loop), `flat_cv` (optimistic thesis re-implementation, comparison only), baselines without SFFS, input validation, warning capture, valid-candidate policy, stability and runtime summaries |
| `artifacts.py` | `save_run` / `load_run` / `check_run`: per-sample out-of-fold predictions, fold assignments and group ids, selected features and hyperparameters per fold, captured warnings, excluded search candidates, metrics, configuration, data sha256, git commit, dependency versions |
| `run.py` | `python -m smalldata.run`: reproducible experiments with `smoke` and `benchmark` presets, and a mode that fixes the folds while varying the search seed (or vice versa) |
| `tests/` | 42 tests: group separation in both loops, fold-local imputation and scaling, malformed and non-finite inputs, insufficient groups, infeasible kNN k, deterministic splits and complete coverage, failed-fit policy, metric arithmetic, artifact round trip, and a reduced end-to-end run on the bundled CSV |

Pipelines: mean imputation, feature standardisation and target standardisation
(`TransformedTargetRegressor`) are fitted inside every training fold. Families: linear
regression, Lasso, kNN, SVR, each with randomised search (`N_ITER = 15` draws, log-uniform
priors) and SFFS. Baselines on identical outer folds: training-fold mean, ridge on all nine
features (alpha tuned), Lasso without SFFS (alpha tuned). All loops are `GroupKFold`, so a
platform group never spans a split. Nothing is silenced: every warning raised during fitting
is stored with target, model, seed, fold and stage; a search draw that fails on any inner
fold is excluded and recorded; if all draws fail, or a final fit produces a non-finite
prediction, the run raises `InvalidFitError`.

## 3. Data provenance and the public/private boundary

[`data/midsize_sedans.csv`](data/midsize_sedans.csv): 31 midsize sedans sold in the USA in
2002, compiled for my bachelor's thesis (TU Berlin, 2024) from the vehicle table published in
de Weck (2006), a Springer chapter that is not open access. The CSV in this repository is the
only data used anywhere; no private or unpublished data are involved, and CI runs on it.

Changes relative to the thesis spreadsheet: column names, four vehicle-name typos, the
`platform_group` column (added), and the towing placeholders (removed):

* **Towing capacity: n = 20.** Eleven vehicles carried exactly 1,225 lb, the mean of the other
  twenty, i.e. imputed placeholders. They are stored as missing.
* **Ground clearance** is missing for five vehicles and is mean-imputed inside the pipeline.
* **Platform twins.** Seven groups (Chevrolet Impala / Monte Carlo, Buick Century / Regal,
  Pontiac Grand Am / Oldsmobile Alero, Dodge Intrepid / Chrysler Concorde, Ford Taurus /
  Mercury Sable, Hyundai Sonata / Kia Optima, Chrysler Sebring / Dodge Stratus / Mitsubishi
  Galant) give 23 groups for 31 vehicles. Wider families such as all GM W-body cars are not grouped.
* **Front track width** is design variable x3, following the thesis code and spreadsheet (the
  thesis text says weight). Curb weight and horsepower are in the CSV for reference only.
* `tire_diameter_in` is the rim diameter (15 or 16 in).

Targets: passenger volume, cargo volume, towing capacity, fuel efficiency, 0–60 mph time.
Features: fuel tank capacity, engine displacement, front track width, wheelbase, ground
clearance, height, length, tyre width, tyre diameter. Another CSV with these 14 columns can be
used directly (`vehicle_id` and `platform_group` optional; `column_map=` renames headers).

## 4. Quick start

Tested with Python 3.12 and the pinned versions in `requirements.txt` (macOS and Ubuntu CI).

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.12 .venv
pip install -r requirements.txt
python -m pytest -q                                          # 42 tests, about 25 s
PYTHONPATH=src python -m smalldata.run --preset smoke --out results/runs/smoke   # about 20 s on 8 cores
```

The smoke run is one target, 2 seeds, 3 × 3 folds, 4 search draws, all families, baselines
and the complete procedure; it writes the artifact tables and fails if any prediction is
non-finite or any sample lacks an out-of-fold prediction. Its numbers are a smoke test, not
the benchmark. The full benchmark that produced the tables below:

```bash
PYTHONPATH=src python -m smalldata.run --preset benchmark --out results/benchmark   # 11.5 min on 8 cores (M1)
python scripts/make_figures.py
```

Other useful invocations: `--target cargo_volume_cuft`, `--protocol nested-group`
(default; `nested-kfold` and `flat` are the optimistic comparisons), `--models families`
or `--models Lasso "Lasso (no SFS)"`, `--vary search --fold-seed 0` (fixed folds, varying
search draws), `--vary folds --search-seed 0`. `python -m smalldata.run --help` lists all.
`jupyter lab notebooks/` opens the exploration notebooks; `02_model_selection.ipynb`
recomputes everything in 20–30 minutes and saves its artifacts to the gitignored
`results/runs/notebook/`.

## 5. Main evidence

All numbers: out-of-fold R² on pooled predictions, mean ± SD over seeds 0–4, nested
`GroupKFold` with 5 outer and 5 inner folds, `N_ITER = 15`, identical outer folds for every
row of a target. Source: [`results/benchmark/`](results/benchmark/) (git commit, data sha256
and package versions in `run_meta.json`; 550 runs, 15,840 out-of-fold predictions, 0 captured
warnings, 0 excluded search candidates). **The SD over seeds is not a confidence interval**:
each seed reshuffles the folds *and* redraws the search, and the sampling uncertainty of a
31-vehicle dataset is not included.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/r2_by_protocol_dark.png">
  <img src="docs/r2_by_protocol_light.png" alt="Out-of-fold R² per target for the four SFFS families, the three baselines and the complete procedure: the flat thesis protocol scores 0.3 to 0.75, nested GroupKFold scores are much lower, ridge on all features matches or beats the SFFS families on most targets, and towing capacity is below zero for every model" width="100%">
</picture>

### 5a. Families vs baselines vs the complete procedure (nested GroupKFold)

"How did each prespecified family perform?" (first four columns, exploratory) and "how well
does the complete selection procedure generalise?" (last column, honest estimate).

| Target (n) | Linear regr. + SFFS | Lasso + SFFS | kNN + SFFS | SVR + SFFS | Mean | Ridge (all feat.) | Lasso (no SFS) | **Complete procedure** |
|---|---|---|---|---|---|---|---|---|
| Passenger volume (31) | 0.35 ± 0.11 | 0.42 ± 0.05 | 0.28 ± 0.06 | 0.40 ± 0.12 | −0.15 ± 0.12 | **0.46 ± 0.08** | 0.44 ± 0.06 | 0.36 ± 0.10 |
| Cargo volume (31) | **0.54 ± 0.07** | 0.52 ± 0.07 | 0.15 ± 0.19 | 0.42 ± 0.09 | −0.11 ± 0.07 | 0.42 ± 0.05 | 0.52 ± 0.07 | 0.47 ± 0.14 |
| Towing capacity (20) | −0.38 ± 0.63 | −0.40 ± 0.46 | −0.86 ± 0.56 | −0.62 ± 0.28 | **−0.13 ± 0.14** | −0.56 ± 0.56 | −0.59 ± 0.80 | −0.41 ± 0.31 |
| Fuel efficiency (31) | −0.02 ± 0.15 | 0.11 ± 0.18 | **0.20 ± 0.14** | −0.13 ± 0.21 | −0.12 ± 0.06 | 0.05 ± 0.09 | 0.08 ± 0.17 | −0.08 ± 0.24 |
| 0–60 mph time (31) | 0.15 ± 0.15 | 0.27 ± 0.13 | 0.23 ± 0.20 | 0.27 ± 0.11 | −0.06 ± 0.02 | **0.40 ± 0.12** | 0.36 ± 0.12 | 0.23 ± 0.12 |
| Median over targets | 0.15 | 0.27 | 0.20 | 0.27 | −0.12 | **0.40** | 0.36 | 0.23 |

What this says:

* **A simpler baseline is as good or better.** Ridge on all nine features has the best median
  R² (0.40) and is the best model on passenger volume and acceleration. Lasso without SFFS
  matches Lasso with SFFS on cargo and passenger volume, beats it on acceleration (0.36 vs
  0.27) and is worse on towing (−0.59 vs −0.40) and fuel efficiency (0.08 vs 0.11), both
  within the seed SD. Only cargo volume (linear regression + SFFS, 0.54 vs 0.42 ridge) and
  fuel efficiency (kNN, 0.20 vs 0.08) favour an SFFS family, by less than the seed SD in the
  second case.
* **The complete procedure is below the post-hoc best family on every target** (0.36 vs 0.42,
  0.47 vs 0.54, −0.41 vs −0.38, −0.08 vs 0.20, 0.23 vs 0.27) and below ridge on four of five.
  Choosing among near-equal families from noisy inner-CV scores adds variance, and the
  post-hoc "best family" number is not an estimate of what any automatic procedure achieves.
  Its family picks over the 25 outer folds show why: passenger volume LR 11 / SVR 13 / kNN 1;
  cargo LR 18 / SVR 4 / kNN 2 / Lasso 1; towing LR 15 / SVR 5 / kNN 5; fuel efficiency kNN 14 /
  LR 8 / SVR 3; acceleration LR 11 / SVR 11 / kNN 2 / Lasso 1.
* **Towing capacity is not predictable** at n = 20 (four distinct ratings). The best "model"
  is the training-fold mean at −0.13; every fitted model is worse.
* **Fuel efficiency and acceleration are weak** (best honest R² 0.20 and 0.40); cargo and
  passenger volume are partly predictable (0.4–0.5).

### 5b. Does SFFS help? Stability and runtime

Mean pairwise Jaccard similarity of the selected feature subsets across the 25 outer folds
(1 = the same subset every fold), mean subset size, and wall time per nested run
(`results/benchmark/stability.csv`).

| Target | LR + SFFS | Lasso + SFFS | kNN + SFFS | SVR + SFFS | Ridge / Lasso (no SFS) |
|---|---|---|---|---|---|
| Passenger volume | J 0.45, 3.8 feat., 11.8 s | 0.40, 3.4, 12.0 s | 0.39, 4.5, 13.9 s | 0.56, 5.8, 12.7 s | 9 feat., 2.1 s |
| Cargo volume | 0.60, 4.5, 9.8 s | 0.48, 4.0, 11.2 s | 0.41, 2.5, 15.5 s | 0.50, 5.0, 17.4 s | 9, 2.6–3.7 s |
| Towing capacity | 0.56, 5.6, 15.1 s | 0.29, 3.0, 14.9 s | 0.20, 2.5, 16.0 s | 0.44, 5.3, 14.9 s | 9, 2.3 s |
| Fuel efficiency | 0.40, 2.5, 11.3 s | 0.53, 2.1, 12.7 s | 0.40, 2.9, 16.2 s | 0.37, 3.9, 15.3 s | 9, 3.2 s |
| 0–60 mph time | 0.39, 3.7, 15.7 s | 0.39, 3.1, 16.7 s | 0.44, 3.1, 18.2 s | 0.46, 5.1, 14.3 s | 9, 2.5–2.9 s |

SFFS takes 80–99 % of each run's fit time (5–6× the runtime of the tuned no-selection
baselines), and the chosen subsets agree only moderately between folds (Jaccard 0.2–0.6).
Given the table in 5a, on this dataset SFFS buys instability and runtime, not accuracy.
The stable choices are the physically plausible ones (length, fuel-tank capacity and height
for cargo volume; engine displacement for fuel efficiency and acceleration); the rest of the
subset depends on which handful of vehicles is in the training fold.

### 5c. How much the protocol alone inflates the score

Median R² over the five targets for the four families
(`results/benchmark/summary.csv`). The flat protocol is the thesis re-implementation
(tune and select on all rows with random 5-fold CV, score on the same folds) and is
intentionally optimistic; it is kept only to measure that optimism.

| Model | Flat (thesis re-impl.) | Nested, random KFold | Nested, GroupKFold | Flat − nested KFold | Nested KFold − GroupKFold |
|---|---|---|---|---|---|
| Linear regression + SFFS | 0.63 | 0.32 | 0.15 | 0.31 | 0.17 |
| Lasso + SFFS | 0.55 | 0.36 | 0.27 | 0.19 | 0.09 |
| kNN + SFFS | 0.56 | 0.26 | 0.20 | 0.30 | 0.06 |
| SVR + SFFS | 0.69 | 0.23 | 0.27 | 0.46 | −0.04 |
| Ridge (all features) | 0.46 | 0.38 | 0.40 | 0.08 | −0.02 |
| Lasso (no SFS) | 0.44 | 0.34 | 0.36 | 0.10 | −0.02 |

Reusing the scoring folds for tuning and SFFS inflates every family on every target; for the
flat protocol towing capacity scores 0.5–0.7, which no model achieves under nested CV. The
inflation is much smaller for the baselines without SFFS (0.08–0.10), which points at SFFS as
the main overfitting channel. Grouping twins lowers the SFFS families further; it changes the
baselines by at most 0.02 in the median.

### 5d. Separating the two randomness sources

A seed changes both the fold assignment and the random-search draws, so the SD in 5a mixes
the two. Two extra runs separate them (`results/benchmark/variance/`; nested GroupKFold,
seeds 0–4, families and tuned baselines, 175 runs each, 0 warnings): `--vary search
--fold-seed 0` keeps the folds of seed 0 and redraws only the search; `--vary folds
--search-seed 0` reshuffles only the folds. SD of R² over the 5 seeds:

| Target | Source | LR + SFFS | Lasso + SFFS | kNN + SFFS | SVR + SFFS | Ridge | Lasso (no SFS) |
|---|---|---|---|---|---|---|---|
| Passenger volume | both / search only / folds only | 0.11 / 0.00 / 0.11 | 0.05 / 0.01 / 0.05 | 0.06 / 0.06 / 0.07 | 0.12 / 0.06 / 0.05 | 0.08 / 0.02 / 0.07 | 0.06 / 0.02 / 0.06 |
| Cargo volume | | 0.07 / 0.00 / 0.07 | 0.07 / 0.02 / 0.06 | 0.19 / 0.13 / 0.17 | 0.09 / 0.13 / 0.10 | 0.05 / 0.02 / 0.08 | 0.07 / 0.01 / 0.07 |
| Towing capacity | | 0.63 / 0.00 / 0.63 | 0.46 / 0.17 / 0.40 | 0.56 / 0.25 / 0.56 | 0.28 / 0.42 / 0.39 | 0.56 / 0.02 / 0.19 | 0.80 / 0.31 / 0.36 |
| Fuel efficiency | | 0.15 / 0.00 / 0.15 | 0.18 / 0.04 / 0.08 | 0.14 / 0.06 / 0.10 | 0.21 / 0.19 / 0.17 | 0.09 / 0.11 / 0.10 | 0.17 / 0.05 / 0.12 |
| 0–60 mph time | | 0.15 / 0.00 / 0.15 | 0.13 / 0.01 / 0.15 | 0.20 / 0.07 / 0.25 | 0.11 / 0.03 / 0.07 | 0.12 / 0.02 / 0.12 | 0.12 / 0.01 / 0.13 |

Fold assignment dominates for the linear models (linear regression has no search, so its
search-only SD is exactly 0 and SFFS is deterministic given the folds). For kNN and SVR the
search draws alone move R² by 0.06–0.42 at 15 draws, i.e. a comparable or larger share, and
with only five values each SD is itself imprecise. Neither component is a confidence interval.

### Relation to the thesis numbers

`flat_cv` re-implements the thesis protocol but does not reproduce the thesis defence values
(e.g. passenger volume LR 0.58, kNN 0.75, SVR 0.78; towing LR −2.91) because the thesis held
out the Honda Accord (n = 30), used the 1,225 lb towing placeholders as data, averaged R² over
five folds of about six cars, drew 50 uniform settings on the raw target with one seed, and
mean-imputed ground clearance before splitting. The thesis response-surface model with
interaction terms is not part of this repository.

## 6. Limitations and unsupported uses

* 31 vehicles (20 for towing) from one market segment and one model year. The numbers describe
  this dataset and this protocol; they do not describe vehicle design in general and must not be
  used to size or rate a vehicle.
* The seed SD is a spread over fold assignment and search draws, not a confidence interval; a
  single unusual vehicle can move R² by 0.1 or more. No interval in this repository is a
  confidence interval.
* Twin groups are defined by hand from badge engineering and identical chassis dimensions.
  Grouping whole platform families would be stricter and would likely lower the scores further.
* The complete procedure chooses among the four SFFS families only, not among the baselines;
  including ridge would probably raise it, but that is a different procedure and was not run.
* Towing capacity takes four distinct values and is better treated as a rating than a
  continuous quantity; its negative R² values are reported as such.
* Search budget 15 draws per tuning run, chosen for runtime; SFFS with floating steps on about
  20 training rows is itself a source of variance (5b).
* The optimistic `flat_cv` protocol exists only for comparison. Do not use its numbers.

## 7. Tests and reproducibility

* `python -m pytest -q`: 42 tests in about 25 s (one marked `slow` runs the smoke preset
  on the bundled CSV, about 20 s). They check that groups are never split in the outer or inner
  loop, that imputation and scaling are fitted on training rows only (perturbing the other test
  rows leaves a prediction unchanged), that missing targets, non-numeric, infinite and
  misaligned inputs, too few groups and infeasible kNN k are rejected before fitting, that the
  same seed gives identical splits and predictions while `fold_seed` and `search_seed` can be
  varied separately, that failed search draws are excluded and recorded and all-failed raises,
  that warnings are captured with context and nothing escapes, that metrics match hand
  calculations, that artifacts round-trip and `check_run` catches non-finite or missing
  predictions, and that families, baselines and the complete procedure are scored on identical
  outer folds.
* CI (`.github/workflows/ci.yml`) installs the pinned Python 3.12 environment, runs the tests
  and the smoke experiment; the full benchmark is a `workflow_dispatch` option that uploads
  its artifacts.
* Every run writes `run_meta.json` with the configuration, the CSV's sha256, the git commit and
  the dependency versions; `predictions.csv` lets every claimed score be recomputed from the
  per-sample out-of-fold predictions. The published benchmark was run at commit `d35e5da`
  (the modelling code is unchanged since; later commits only round artifact CSV floats to 7
  significant digits and make secondary tables optional when loading) on an 8-core Apple M1,
  11.5 minutes wall time.

## Repository layout

| Path | Content |
|---|---|
| [`data/midsize_sedans.csv`](data/midsize_sedans.csv) | 31 vehicles, 9 design variables, 5 functional attributes, platform groups |
| [`src/smalldata/`](src/smalldata/) | `data.py`, `modeling.py`, `artifacts.py`, `run.py` (CLI) |
| [`tests/`](tests/) | pytest suite with small synthetic fixtures |
| [`notebooks/01_explore_data.ipynb`](notebooks/01_explore_data.ipynb) | summary statistics, twins, correlations, in-sample single-variable fits (exploratory, not validated scores), outliers |
| [`notebooks/02_model_selection.ipynb`](notebooks/02_model_selection.ipynb) | families, baselines, complete procedure, SFFS stability, protocol comparison |
| [`results/benchmark/`](results/benchmark/) | artifacts of the published benchmark |
| [`scripts/make_figures.py`](scripts/make_figures.py) | draws `docs/*.png` from `results/benchmark/metrics.csv` |

## Citation and context

* de Weck, O. L. (2006). Determining product platform extent. In T. W. Simpson, Z. Siddique &
  J. R. Jiao (Eds.), *Product Platform and Product Family Design: Methods and Applications*
  (pp. 241–301). Springer. https://doi.org/10.1007/0-387-29197-0_12 (source of the vehicle data)
* Rahn, M., Göhlich, D., Fay, T.-A., & van Ho, K. (2025). Data-driven design: predicting
  functional attributes in early-stage automotive engineering. *Proceedings of the Design
  Society*, 5, 1091–1099. https://doi.org/10.1017/pds.2025.10123 (follow-up paper on the same
  dataset; it held out the Mercury Sable as its evaluation car)
* Bachelor's thesis: Kien van Ho, TU Berlin, 2024.

## License

MIT, see [LICENSE](LICENSE).
