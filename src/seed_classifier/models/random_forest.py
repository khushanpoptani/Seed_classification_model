"""Random forest built from the local decision tree implementation."""

from __future__ import annotations

import math

import numpy as np

from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Bootstrap aggregation of numeric decision trees."""

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 8,
        min_samples_split: int = 4,
        max_features: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        num_features = x.shape[1]
        max_features = self.max_features or max(1, int(math.sqrt(num_features)))
        self.trees_ = []

        for _ in range(self.n_estimators):
            sample_indices = rng.integers(0, x.shape[0], size=x.shape[0])
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(x[sample_indices], y[sample_indices])
            self.trees_.append(tree)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("Model must be fitted before prediction.")
        predictions = np.vstack([tree.predict(x) for tree in self.trees_])
        votes = []
        for column in predictions.T:
            values, counts = np.unique(column, return_counts=True)
            votes.append(int(values[np.argmax(counts)]))
        return np.array(votes, dtype=int)
