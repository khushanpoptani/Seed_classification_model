#!/usr/bin/env python3
"""Validate the thesis-aligned image pipeline setup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seed_classifier.image_pipeline import ImageTrainingConfig, train_image_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the image classification pipeline.")
    parser.add_argument("--dataset-dir", required=True, help="Root directory for the image dataset.")
    parser.add_argument("--num-classes", type=int, default=14, help="Number of image classes.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ImageTrainingConfig(dataset_dir=args.dataset_dir, num_classes=args.num_classes)
    result = train_image_model(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
