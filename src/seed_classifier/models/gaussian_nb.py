"""Gaussian Naive Bayes classifier."""

from __future__ import annotations

import numpy as np


class GaussianNBClassifier:
    """Minimal Gaussian NB implementation for numeric features."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.class_priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNBClassifier":
        classes = np.unique(y)
        means = []
        vars_ = []
        priors = []

        for class_id in classes:
            class_x = x[y == class_id]
            means.append(class_x.mean(axis=0))
            vars_.append(class_x.var(axis=0) + 1e-9)
            priors.append(class_x.shape[0] / x.shape[0])

        self.classes_ = classes
        self.means_ = np.vstack(means)
        self.vars_ = np.vstack(vars_)
        self.class_priors_ = np.array(priors)
        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in [self.classes_, self.means_, self.vars_, self.class_priors_]):
            raise ValueError("Model must be fitted before prediction.")

        joint = []
        for index, class_id in enumerate(self.classes_):
            mean = self.means_[index]
            var = self.vars_[index]
            log_prior = np.log(self.class_priors_[index])
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var, axis=1)
            joint.append(log_prior + log_likelihood)
        return np.vstack(joint).T

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self._joint_log_likelihood(x)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]
