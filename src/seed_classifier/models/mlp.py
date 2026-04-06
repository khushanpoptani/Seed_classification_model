"""A small fully connected neural network implemented with NumPy."""

from __future__ import annotations

import numpy as np


class MLPClassifier:
    """Single-hidden-layer MLP for multiclass classification."""

    def __init__(
        self,
        hidden_units: int = 16,
        learning_rate: float = 0.03,
        epochs: int = 400,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights_1: np.ndarray | None = None
        self.bias_1: np.ndarray | None = None
        self.weights_2: np.ndarray | None = None
        self.bias_2: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        y_idx = np.array([class_to_index[int(label)] for label in y], dtype=int)

        input_units = x.shape[1]
        output_units = len(self.classes_)
        self.weights_1 = rng.normal(0.0, 0.2, size=(input_units, self.hidden_units))
        self.bias_1 = np.zeros((1, self.hidden_units))
        self.weights_2 = rng.normal(0.0, 0.2, size=(self.hidden_units, output_units))
        self.bias_2 = np.zeros((1, output_units))

        y_one_hot = np.eye(output_units)[y_idx]

        for _ in range(self.epochs):
            permutation = rng.permutation(x.shape[0])
            x_shuffled = x[permutation]
            y_shuffled = y_one_hot[permutation]

            for start in range(0, x.shape[0], self.batch_size):
                end = start + self.batch_size
                xb = x_shuffled[start:end]
                yb = y_shuffled[start:end]

                z1 = xb @ self.weights_1 + self.bias_1
                a1 = self._relu(z1)
                z2 = a1 @ self.weights_2 + self.bias_2
                probs = self._softmax(z2)

                error_2 = (probs - yb) / xb.shape[0]
                grad_w2 = a1.T @ error_2
                grad_b2 = error_2.sum(axis=0, keepdims=True)

                error_1 = (error_2 @ self.weights_2.T) * self._relu_grad(z1)
                grad_w1 = xb.T @ error_1
                grad_b1 = error_1.sum(axis=0, keepdims=True)

                self.weights_2 -= self.learning_rate * grad_w2
                self.bias_2 -= self.learning_rate * grad_b2
                self.weights_1 -= self.learning_rate * grad_w1
                self.bias_1 -= self.learning_rate * grad_b1

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in [self.weights_1, self.bias_1, self.weights_2, self.bias_2, self.classes_]):
            raise ValueError("Model must be fitted before prediction.")

        z1 = x @ self.weights_1 + self.bias_1
        a1 = self._relu(z1)
        z2 = a1 @ self.weights_2 + self.bias_2
        probs = self._softmax(z2)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
