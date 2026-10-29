# Small-data model selection for early-stage vehicle design

Which regression model should you trust when you only have about 30 data points?
This repository is a cleaned-up, public version of the code from my bachelor's thesis
(TU Berlin, 2024). It predicts five functional attributes of midsize sedans from nine
design variables and compares four model families under a leakage-free nested
cross-validation protocol.

| Functional attributes (targets) | Design variables (features) |
|---|---|
| passenger volume, cargo volume, towing capacity, fuel efficiency, 0–60 mph acceleration | fuel tank capacity, engine displacement, front track width, wheelbase, ground clearance, height, length, tyre width, tyre diameter |

Models: linear regression, Lasso, k-nearest neighbours, support vector regression.
Each one runs in a pipeline with mean imputation, standardisation, randomised
hyperparameter search and sequential floating forward selection (SFFS).

## Data

[`data/midsize_sedans.csv`](data/midsize_sedans.csv) holds the 31 midsize sedans
(model years around 2000) used in the thesis. The specifications come from de Weck's
product-platform study (see References) and are public. Notes on the file:

* `ground_clearance_in` is missing for 5 vehicles; it is mean-imputed inside the pipeline.
* `tire_diameter_in` is the wheel (rim) diameter, 15 or 16 inches.
* `horsepower_hp` and `curb_weight_lb` are included for reference but are not model
  inputs. In de Weck's model they are intermediate variables, not design variables.
* Compared with the thesis spreadsheet, only column names and four typos in vehicle
  names were changed.

To run on another dataset, provide a CSV with these columns (an optional `vehicle_id`
column becomes the index; extra columns are ignored; missing feature values are fine):

```
fuel_tank_capacity_gal, engine_displacement_cc, front_track_width_in, wheelbase_in,
ground_clearance_in, height_in, length_in, tire_width_mm, tire_diameter_in,
passenger_volume_cuft, cargo_volume_cuft, towing_capacity_lb, fuel_efficiency_mpg,
acceleration_0_60_s
```

If your headers differ, pass `column_map={"your_name": "expected_name"}` to `load_dataset`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Run on another file:

```bash
DATA_PATH=/path/to/your.csv jupyter nbconvert --to notebook --execute notebooks/02_model_selection.ipynb --output-dir /tmp
```

| Notebook | Content |
|---|---|
| [`01_explore_data.ipynb`](notebooks/01_explore_data.ipynb) | summary statistics, correlation of design vs functional variables, single-variable baselines, outliers |
| [`02_model_selection.ipynb`](notebooks/02_model_selection.ipynb) | nested CV for all models and targets, predicted vs actual, feature-selection stability, comparison with the original protocol |

The reusable code is in [`src/smalldata/`](src/smalldata/): `data.py` (loader and
schema) and `modeling.py` (model candidates, `nested_cv`, `flat_cv`).

## What changed compared with the thesis

The thesis tuned hyperparameters and selected features with 5-fold CV on all
vehicles, then reported 5-fold CV scores **on the same folds**. Model selection had
therefore already seen every test fold, so the reported R² values were optimistic.
Missing values were also mean-imputed on the full dataset before splitting.

This version fixes both:

* **Nested CV.** An outer 5-fold loop only scores. Hyperparameter search and SFFS run
  in an inner 5-fold loop on the outer training data.
* **Imputation and scaling inside the pipeline**, fitted on training folds only.
* Scores are computed on pooled out-of-fold predictions (R², RMSE, MAE, MAPE), and the
  SFFS feature choice is reported as a frequency over outer folds.

`flat_cv` keeps the original protocol so the gap can be measured on the same data.


| Model | R² flat (thesis protocol) | R² nested | Optimism |
|---|---|---|---|
| Linear regression | 0.62 | 0.26 | 0.36 |
| Lasso | 0.59 | 0.37 | 0.07 |
| kNN | 0.61 | 0.38 | 0.21 |
| SVR | 0.59 | 0.31 | 0.27 |

*Median over the five targets, out-of-fold R², seed 42.*

Under the flat protocol all four models look alike (R² ≈ 0.6). Nested CV separates
them and roughly halves the scores. The volumes stay predictable (R² ≈ 0.65), fuel
efficiency and acceleration drop to about 0.35–0.4, and towing capacity is not
predictable from these design variables. The notebook has the per-target tables.

Two smaller differences: the thesis held out one vehicle as an extra evaluation case,
and it also built a separate de Weck-style regression with interaction terms. Neither
is part of this repository.

## References

* de Weck, O. L. (2006). Determining product platform extent. In T. W. Simpson,
  Z. Siddique & J. R. Jiao (Eds.), *Product Platform and Product Family Design:
  Methods and Applications* (pp. 241–301). Springer.
  https://doi.org/10.1007/0-387-29197-0_12 (source of the vehicle data)
* Rahn, M., Göhlich, D., Fay, T.-A., van Ho, K. (2025). Data-driven design: predicting
  functional attributes in early-stage automotive engineering. *Proceedings of the
  Design Society*, 5. https://doi.org/10.1017/pds.2025.10123

## License

MIT, see [LICENSE](LICENSE).
