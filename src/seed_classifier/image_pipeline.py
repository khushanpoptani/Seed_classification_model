"""Optional image-classification scaffold aligned with the thesis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class ImageTrainingConfig:
    dataset_dir: str
    num_classes: int = 14
    image_height: int = 224
    image_width: int = 224
    channels: int = 3
    dropout: float = 0.5
    dense_units: int = 256
    train_dir_name: str = "train"
    val_dir_name: str = "val"
    test_dir_name: str = "test"


def expected_image_classes() -> List[str]:
    return [f"class_{index + 1:02d}" for index in range(14)]


def validate_image_dataset_layout(config: ImageTrainingConfig) -> Dict[str, object]:
    root = Path(config.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image dataset directory not found: {root}")

    split_dirs = [config.train_dir_name, config.val_dir_name, config.test_dir_name]
    summary = {"dataset_dir": str(root), "splits": {}, "expected_classes": expected_image_classes()}

    for split_name in split_dirs:
        split_path = root / split_name
        if not split_path.exists():
            summary["splits"][split_name] = {"exists": False, "class_dirs": []}
            continue
        class_dirs = sorted([path.name for path in split_path.iterdir() if path.is_dir()])
        summary["splits"][split_name] = {"exists": True, "class_dirs": class_dirs}

    return summary


def build_vgg16_transfer_learning_plan(config: ImageTrainingConfig) -> Dict[str, object]:
    return {
        "backbone": "VGG16",
        "input_shape": [config.image_height, config.image_width, config.channels],
        "head_layers": [
            {"type": "AveragePooling2D", "pool_size": [7, 7]},
            {"type": "Flatten"},
            {"type": "Dense", "units": config.dense_units, "activation": "relu"},
            {"type": "Dropout", "rate": config.dropout},
            {"type": "Dense", "units": config.num_classes, "activation": "softmax"},
        ],
        "notes": [
            "Matches the thesis description of a modified VGG16 classifier.",
            "Requires TensorFlow/Keras and an image dataset arranged by split/class directories.",
            "Recommended image augmentation: rotation, horizontal flip, zoom, brightness shift, and rescaling.",
        ],
    }


def train_image_model(config: ImageTrainingConfig) -> Dict[str, object]:
    layout = validate_image_dataset_layout(config)
    plan = build_vgg16_transfer_learning_plan(config)

    try:
        import tensorflow  # type: ignore  # noqa: F401
    except ModuleNotFoundError as error:
        return {
            "status": "tensorflow_missing",
            "config": asdict(config),
            "dataset_layout": layout,
            "model_plan": plan,
            "next_steps": [
                "Install requirements-image.txt",
                "Prepare train/val/test folders with one subdirectory per seed class",
                "Implement or plug in the actual Keras training loop",
            ],
            "message": (
                "TensorFlow is not installed in the current environment, so the thesis-aligned "
                "VGG16 pipeline is prepared but not executed."
            ),
        }

    return {
        "status": "ready_for_training",
        "config": asdict(config),
        "dataset_layout": layout,
        "model_plan": plan,
    }
