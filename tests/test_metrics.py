"""Metric tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from seed_classifier.metrics import accuracy_score, classification_report, confusion_matrix


class MetricsTestCase(unittest.TestCase):
    def test_accuracy_score(self) -> None:
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 1, 1, 1])
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix[1, 1], 2)

    def test_classification_report_contains_macro_scores(self) -> None:
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = classification_report(y_true, y_pred, labels=[0, 1, 2])
        self.assertIn("macro_f1", report)
        self.assertIn("accuracy", report)


if __name__ == "__main__":
    unittest.main()
