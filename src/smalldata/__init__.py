"""Model selection for small engineering datasets with nested cross-validation."""

from .data import FEATURES, TARGETS, Dataset, load_dataset
from .modeling import (
    Candidate,
    CVResult,
    default_candidates,
    flat_cv,
    nested_cv,
    results_table,
    selection_frequency,
)

__all__ = [
    "FEATURES",
    "TARGETS",
    "Dataset",
    "load_dataset",
    "Candidate",
    "CVResult",
    "default_candidates",
    "flat_cv",
    "nested_cv",
    "results_table",
    "selection_frequency",
]
