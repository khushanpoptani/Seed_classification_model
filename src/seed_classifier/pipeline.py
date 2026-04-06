"""Training and evaluation pipeline for seed classifiers."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .datasets import FEATURE_NAMES, DatasetBundle, dataset_summary, generate_demo_dataset, load_seed_dataset
from .metrics import classification_report
from .models import DecisionTreeClassifier, GaussianNBClassifier, MLPClassifier, RandomForestClassifier
from .preprocessing import LabelEncoder, StandardScaler, train_test_split


@dataclass
class TabularTrainingConfig:
    dataset_path: str | None = None
    use_demo_data: bool = False
    model_name: str = "all"
    test_size: float = 0.2
    random_state: int = 42
    output_dir: str = "artifacts/tabular"


@dataclass
class ModelArtifact:
    model_name: str
    scaler: StandardScaler
    label_encoder: LabelEncoder
    model: object
    feature_names: List[str]
    metrics: Dict[str, object]
    config: Dict[str, object]


def _make_model(name: str, random_state: int) -> object:
    if name == "gaussian_nb":
        return GaussianNBClassifier()
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_split=4, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=31,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_units=24,
            learning_rate=0.03,
            epochs=500,
            batch_size=24,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model name: {name}")


def _resolve_model_names(model_name: str) -> List[str]:
    if model_name == "all":
        return ["gaussian_nb", "decision_tree", "random_forest", "mlp"]
    return [model_name]


def _load_dataset(config: TabularTrainingConfig) -> DatasetBundle:
    if config.use_demo_data:
        return generate_demo_dataset(random_state=config.random_state)
    if config.dataset_path is None:
        raise ValueError("Provide --dataset or pass --demo-data.")
    return load_seed_dataset(config.dataset_path)


def train_and_evaluate_tabular_models(config: TabularTrainingConfig) -> Dict[str, Dict[str, object]]:
    dataset = _load_dataset(config)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    label_ids = list(range(len(label_encoder.classes_ or [])))

    summaries: Dict[str, Dict[str, object]] = {}

    for model_name in _resolve_model_names(config.model_name):
        model = _make_model(model_name, config.random_state)
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)
        report = classification_report(y_test, predictions, label_ids)
        report["dataset_summary"] = dataset_summary(dataset.frame)
        report["labels"] = list(label_encoder.classes_ or [])

        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact = ModelArtifact(
            model_name=model_name,
            scaler=scaler,
            label_encoder=label_encoder,
            model=model,
            feature_names=list(FEATURE_NAMES),
            metrics=report,
            config=asdict(config),
        )

        with (model_dir / "model.pkl").open("wb") as handle:
            pickle.dump(artifact, handle)

        with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        prediction_frame = pd.DataFrame(
            {
                "true_label": label_encoder.inverse_transform(y_test),
                "predicted_label": label_encoder.inverse_transform(predictions),
            }
        )
        prediction_frame.to_csv(model_dir / "predictions.csv", index=False)

        summaries[model_name] = report

    return summaries


def load_artifact(artifact_dir: str | Path) -> ModelArtifact:
    artifact_path = Path(artifact_dir) / "model.pkl"
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    return artifact


def predict_from_features(artifact: ModelArtifact, features: Iterable[float]) -> str:
    row = np.array(list(features), dtype=float).reshape(1, -1)
    if row.shape[1] != len(artifact.feature_names):
        raise ValueError(
            f"Expected {len(artifact.feature_names)} features but received {row.shape[1]}."
        )
    transformed = artifact.scaler.transform(row)
    prediction = artifact.model.predict(transformed)
    return str(artifact.label_encoder.inverse_transform(prediction)[0])
