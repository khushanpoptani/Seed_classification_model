"""A small CART-style decision tree for numeric classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[int] = None


class DecisionTreeClassifier:
    """Binary-split decision tree using Gini impurity."""

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 4,
        max_features: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root_: TreeNode | None = None
        self.classes_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.classes_ = np.unique(y)
        self.root_ = self._build_tree(x, y, depth=0)
        return self

    def _gini(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1.0 - np.sum(probabilities**2)

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        num_features = x.shape[1]
        feature_indices = np.arange(num_features)

        if self.max_features is not None and self.max_features < num_features:
            feature_indices = self._rng.choice(feature_indices, size=self.max_features, replace=False)

        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature_index in feature_indices:
            values = np.unique(x[:, feature_index])
            if values.size < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = x[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gini = (
                    left_mask.sum() / len(y) * self._gini(y[left_mask])
                    + right_mask.sum() / len(y) * self._gini(y[right_mask])
                )
                if gini < best_gini:
                    best_gini = gini
                    best_feature = int(feature_index)
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _leaf_value(self, y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or x.shape[0] < self.min_samples_split
        ):
            return TreeNode(value=self._leaf_value(y))

        feature_index, threshold = self._best_split(x, y)
        if feature_index is None or threshold is None:
            return TreeNode(value=self._leaf_value(y))

        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return TreeNode(value=self._leaf_value(y))

        left = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(x[right_mask], y[right_mask], depth + 1)
        return TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> int:
        if node.value is not None:
            return int(node.value)
        if row[node.feature_index] <= node.threshold:
            return self._predict_one(row, node.left)
        return self._predict_one(row, node.right)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model must be fitted before prediction.")
        return np.array([self._predict_one(row, self.root_) for row in x], dtype=int)
