"""Model selection for small engineering datasets with nested cross-validation."""

from .data import FEATURES, GROUP_COLUMN, TARGETS, Dataset, load_dataset
from .modeling import (
    N_ITER,
    SEEDS,
    Candidate,
    CVResult,
    default_candidates,
    flat_cv,
    nested_cv,
    results_table,
    run_repeated,
    selection_frequency,
)

__all__ = [
    "FEATURES",
    "GROUP_COLUMN",
    "TARGETS",
    "Dataset",
    "load_dataset",
    "N_ITER",
    "SEEDS",
    "Candidate",
    "CVResult",
    "default_candidates",
    "flat_cv",
    "nested_cv",
    "results_table",
    "run_repeated",
    "selection_frequency",
]
