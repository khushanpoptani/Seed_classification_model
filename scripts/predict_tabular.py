#!/usr/bin/env python3
"""Run tabular inference from a saved seed-classification artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seed_classifier.datasets import FEATURE_NAMES
from seed_classifier.pipeline import load_artifact, predict_from_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a seed class from numeric features.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing model.pkl")
    parser.add_argument("--area", type=float, required=True)
    parser.add_argument("--perimeter", type=float, required=True)
    parser.add_argument("--compactness", type=float, required=True)
    parser.add_argument("--kernel-length", type=float, required=True)
    parser.add_argument("--kernel-width", type=float, required=True)
    parser.add_argument("--asymmetry", type=float, required=True)
    parser.add_argument("--groove-length", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = load_artifact(args.artifact_dir)
    features = [
        args.area,
        args.perimeter,
        args.compactness,
        args.kernel_length,
        args.kernel_width,
        args.asymmetry,
        args.groove_length,
    ]
    prediction = predict_from_features(artifact, features)
    print(json.dumps({"prediction": prediction, "feature_order": FEATURE_NAMES}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
