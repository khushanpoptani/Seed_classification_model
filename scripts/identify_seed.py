#!/usr/bin/env python3
"""Identify a seed from a single query image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seed_classifier.image_pipeline import IdentificationConfig, identify_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identify a seed from a single image.")
    parser.add_argument("--image", required=True, help="Path to the query seed image.")
    parser.add_argument(
        "--registry-dir",
        default="artifacts/image_registry",
        help="Directory containing registered seed models.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many ranked matches to return.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = IdentificationConfig(
        image_path=args.image,
        registry_dir=args.registry_dir,
        top_k=args.top_k,
    )
    result = identify_seed(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
