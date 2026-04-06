"""Dataset loading and synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "kernel_groove_length",
]

LABEL_NAMES = ["Kama", "Rosa", "Canadian"]

INTEGER_TO_LABEL = {1: "Kama", 2: "Rosa", 3: "Canadian"}


@dataclass
class DatasetBundle:
    """Container for model-ready data."""

    frame: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip().lower().replace(" ", "_") for column in frame.columns}
    frame = frame.rename(columns=renamed)

    expected_variants = {
        "kernel.length": "kernel_length",
        "kernel.width": "kernel_width",
        "kernel_groove_length": "kernel_groove_length",
        "groove_length": "kernel_groove_length",
        "compactness": "compactness",
        "area": "area",
        "perimeter": "perimeter",
        "asymmetry": "asymmetry_coefficient",
        "asymmetry_coefficient": "asymmetry_coefficient",
        "kernel_length": "kernel_length",
        "kernel_width": "kernel_width",
        "label": "label",
        "class": "label",
        "target": "label",
    }
    frame = frame.rename(columns={column: expected_variants.get(column, column) for column in frame.columns})
    return frame


def _coerce_labels(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        numeric = series.astype(int)
        return numeric.map(INTEGER_TO_LABEL).fillna(numeric.astype(str))

    normalized = series.astype(str).str.strip()
    replacements = {
        "1": "Kama",
        "2": "Rosa",
        "3": "Canadian",
        "kama": "Kama",
        "rosa": "Rosa",
        "canadian": "Canadian",
    }
    return normalized.str.lower().map(replacements).fillna(normalized)


def load_seed_dataset(path: str | Path) -> DatasetBundle:
    """Load a CSV, TSV, or whitespace-delimited UCI Seeds style dataset."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        frame = pd.read_csv(path)
        if frame.shape[1] == 1:
            raise ValueError("single column fallback")
    except Exception:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        if frame.shape[1] != 8:
            raise ValueError(
                "Expected 8 columns for whitespace-delimited UCI Seeds data "
                f"but found {frame.shape[1]} columns."
            )
        frame.columns = FEATURE_NAMES + ["label"]

    frame = _normalize_columns(frame)

    missing = [column for column in FEATURE_NAMES + ["label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = frame[FEATURE_NAMES + ["label"]].copy()
    frame["label"] = _coerce_labels(frame["label"])
    frame = frame.dropna().reset_index(drop=True)

    features = frame[FEATURE_NAMES].astype(float).to_numpy()
    labels = frame["label"].astype(str).to_numpy()
    return DatasetBundle(frame=frame, features=features, labels=labels)


def generate_demo_dataset(
    samples_per_class: int = 80,
    random_state: int = 42,
    output_path: Optional[str | Path] = None,
) -> DatasetBundle:
    """Create a synthetic UCI Seeds style dataset for local demos and tests."""

    rng = np.random.default_rng(random_state)
    centers = {
        "Kama": np.array([14.7, 14.5, 0.88, 5.55, 3.25, 2.15, 5.10]),
        "Rosa": np.array([13.4, 13.4, 0.84, 5.10, 3.00, 3.10, 4.85]),
        "Canadian": np.array([12.3, 13.1, 0.81, 4.75, 2.85, 4.20, 5.00]),
    }
    spread = np.array([0.45, 0.35, 0.02, 0.22, 0.10, 0.35, 0.18])

    rows = []
    for label, center in centers.items():
        values = rng.normal(loc=center, scale=spread, size=(samples_per_class, len(FEATURE_NAMES)))
        values[:, 2] = np.clip(values[:, 2], 0.70, 0.95)
        for sample in values:
            rows.append(dict(zip(FEATURE_NAMES, sample), label=label))

    frame = pd.DataFrame(rows)
    frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    return DatasetBundle(
        frame=frame,
        features=frame[FEATURE_NAMES].to_numpy(dtype=float),
        labels=frame["label"].to_numpy(dtype=str),
    )


def dataset_summary(frame: pd.DataFrame) -> dict:
    """Return a small descriptive summary suitable for logs or reports."""

    label_counts = frame["label"].value_counts().sort_index().to_dict()
    feature_ranges = {
        feature: {
            "min": float(frame[feature].min()),
            "max": float(frame[feature].max()),
            "mean": float(frame[feature].mean()),
        }
        for feature in FEATURE_NAMES
    }
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "label_counts": label_counts,
        "feature_ranges": feature_ranges,
    }
