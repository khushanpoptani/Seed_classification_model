"""Preprocessing utilities for tabular classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class StandardScaler:
    """Small NumPy-only feature scaler."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before calling transform.")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class LabelEncoder:
    """Simple label encoder that preserves readable class names."""

    classes_: List[str] | None = None
    class_to_index_: Dict[str, int] | None = None

    def fit(self, labels: np.ndarray) -> "LabelEncoder":
        classes = sorted({str(label) for label in labels})
        self.classes_ = classes
        self.class_to_index_ = {label: index for index, label in enumerate(classes)}
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        if self.class_to_index_ is None:
            raise ValueError("Encoder must be fitted before calling transform.")
        return np.array([self.class_to_index_[str(label)] for label in labels], dtype=int)

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Encoder must be fitted before calling inverse_transform.")
        return np.array([self.classes_[int(index)] for index in indices], dtype=object)

    def fit_transform(self, labels: np.ndarray) -> np.ndarray:
        return self.fit(labels).transform(labels)


def train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split without external dependencies."""

    rng = np.random.default_rng(random_state)
    unique_labels = np.unique(labels)
    train_indices = []
    test_indices = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        cutoff = max(1, int(round(len(shuffled) * test_size)))
        test_indices.extend(shuffled[:cutoff])
        train_indices.extend(shuffled[cutoff:])

    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        features[train_indices],
        features[test_indices],
        labels[train_indices],
        labels[test_indices],
    )
