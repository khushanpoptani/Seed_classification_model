#!/usr/bin/env python3
"""Register a seed from six directional images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seed_classifier.image_pipeline import SeedRegistrationConfig, register_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register a seed from six directional images.")
    parser.add_argument("--name", required=True, help="Seed name supplied by the user.")
    parser.add_argument("--front", required=True, help="Front-view image path.")
    parser.add_argument("--back", required=True, help="Back-view image path.")
    parser.add_argument("--left", required=True, help="Left-view image path.")
    parser.add_argument("--right", required=True, help="Right-view image path.")
    parser.add_argument("--top", required=True, help="Top-view image path.")
    parser.add_argument("--bottom", required=True, help="Bottom-view image path.")
    parser.add_argument(
        "--registry-dir",
        default="artifacts/image_registry",
        help="Directory where registered seeds are stored.",
    )
    parser.add_argument("--species", default="", help="Optional biological or commercial species name.")
    parser.add_argument("--source", default="", help="Optional source or supplier detail.")
    parser.add_argument("--notes", default="", help="Optional free-form notes for the registered seed.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = SeedRegistrationConfig(
        name=args.name,
        front_image=args.front,
        back_image=args.back,
        left_image=args.left,
        right_image=args.right,
        top_image=args.top,
        bottom_image=args.bottom,
        registry_dir=args.registry_dir,
        species=args.species,
        source=args.source,
        notes=args.notes,
    )
    result = register_seed(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
