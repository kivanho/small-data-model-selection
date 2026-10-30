"""Loading and validating a small vehicle design dataset from CSV.

The expected schema has 9 design variables (features) and 5 functional
attributes (targets). An optional ``vehicle_id`` column is used as the index, and an
optional ``platform_group`` column marks vehicles that share a platform (badge-engineered
twins), so cross-validation can keep each group in a single fold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "fuel_tank_capacity_gal",
    "engine_displacement_cc",
    "front_track_width_in",
    "wheelbase_in",
    "ground_clearance_in",
    "height_in",
    "length_in",
    "tire_width_mm",
    "tire_diameter_in",
]

TARGETS: list[str] = [
    "passenger_volume_cuft",
    "cargo_volume_cuft",
    "towing_capacity_lb",
    "fuel_efficiency_mpg",
    "acceleration_0_60_s",
]

ID_COLUMN = "vehicle_id"
GROUP_COLUMN = "platform_group"


@dataclass
class Dataset:
    X: pd.DataFrame
    Y: pd.DataFrame
    groups: pd.Series
    """Platform group per row. Rows without a group are their own group."""

    @property
    def n_samples(self) -> int:
        return len(self.X)

    def target(self, name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return (X, y, groups) for one target, dropping rows where that target is missing."""
        y = self.Y[name]
        mask = y.notna()
        return self.X.loc[mask], y.loc[mask], self.groups.loc[mask]


def load_dataset(path: str | Path, column_map: dict[str, str] | None = None) -> Dataset:
    """Load a CSV with the 9 feature and 5 target columns.

    Parameters
    ----------
    path:
        CSV file. Extra columns are ignored.
    column_map:
        Optional ``{your_column: expected_column}`` mapping, applied before validation,
        so a file with different headers can be used without editing it.

    Missing feature values are allowed: they are mean-imputed inside the model
    pipeline, i.e. fitted on training folds only. Missing target values are allowed
    and those rows are skipped for that target. Without a ``platform_group`` column,
    every row is its own group (plain shuffled K-fold).
    """
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=column_map)

    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if ID_COLUMN in df.columns:
        df = df.set_index(ID_COLUMN)

    numeric = df[FEATURES + TARGETS].apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() & df[FEATURES + TARGETS].notna()
    if bad.any().any():
        cols = bad.columns[bad.any()].tolist()
        raise ValueError(f"Non-numeric values found in columns: {cols}")

    if np.isinf(numeric.to_numpy(dtype=float)).any():
        cols = numeric.columns[np.isinf(numeric).any()].tolist()
        raise ValueError(f"Infinite values found in columns: {cols}")
    if len(numeric) == 0:
        raise ValueError("CSV has no data rows")
    if not numeric.index.is_unique:
        raise ValueError(f"Duplicate {ID_COLUMN} values: {numeric.index[numeric.index.duplicated()].tolist()}")

    if GROUP_COLUMN in df.columns:
        groups = df[GROUP_COLUMN].astype("string").fillna(pd.Series(df.index.astype(str), index=df.index))
    else:
        groups = pd.Series(df.index.astype(str), index=df.index)

    return Dataset(X=numeric[FEATURES], Y=numeric[TARGETS], groups=groups.rename(GROUP_COLUMN))


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the file bytes, recorded in run artifacts as the data fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
