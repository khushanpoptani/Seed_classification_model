"""Evaluation metrics for classification tasks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[label_to_index[int(truth)], label_to_index[int(pred)]] += 1
    return matrix


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels)
    precision = []
    recall = []
    f1 = []
    support = []

    for index in range(len(labels)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        class_support = matrix[index, :].sum()

        class_precision = tp / (tp + fp) if (tp + fp) else 0.0
        class_recall = tp / (tp + fn) if (tp + fn) else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall)
            else 0.0
        )
        precision.append(float(class_precision))
        recall.append(float(class_recall))
        f1.append(float(class_f1))
        support.append(int(class_support))

    return {
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "support": support,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": matrix.tolist(),
    }


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, object]:
    report = precision_recall_f1(y_true, y_pred, labels)
    report["accuracy"] = accuracy_score(y_true, y_pred)
    return report
