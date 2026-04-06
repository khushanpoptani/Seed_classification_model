#!/usr/bin/env python3
"""Train thesis-aligned tabular seed classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seed_classifier.pipeline import TabularTrainingConfig, train_and_evaluate_tabular_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train seed-classification models.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a CSV or whitespace-delimited UCI Seeds style dataset.",
    )
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Use a built-in synthetic thesis-style dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "gaussian_nb", "decision_tree", "random_forest", "mlp"],
        help="Model to train.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/tabular",
        help="Directory where models and metrics are stored.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TabularTrainingConfig(
        dataset_path=args.dataset,
        use_demo_data=args.demo_data,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        output_dir=args.output_dir,
    )
    reports = train_and_evaluate_tabular_models(config)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
