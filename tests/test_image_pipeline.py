"""Tests for the simplified multi-view image pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from PIL import Image, ImageDraw

from seed_classifier.image_pipeline import IdentificationConfig, SeedRegistrationConfig, identify_seed, register_seed


def _save_seed_shape(path: Path, kind: str) -> None:
    image = Image.new("L", (160, 160), color=255)
    draw = ImageDraw.Draw(image)

    if kind == "round":
        draw.ellipse((35, 45, 125, 115), fill=20)
    elif kind == "long":
        draw.ellipse((20, 55, 140, 105), fill=20)
    elif kind == "top":
        draw.ellipse((45, 30, 115, 130), fill=20)
    elif kind == "side":
        draw.ellipse((25, 50, 135, 110), fill=20)
    else:
        draw.ellipse((30, 40, 130, 120), fill=20)

    image.save(path)


class ImagePipelineTestCase(unittest.TestCase):
    def test_register_and_identify_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            seed_one_dir = tmp_path / "seed_one"
            seed_two_dir = tmp_path / "seed_two"
            seed_one_dir.mkdir()
            seed_two_dir.mkdir()

            for view in ("front", "back", "top", "bottom"):
                _save_seed_shape(seed_one_dir / f"{view}.png", "round")
                _save_seed_shape(seed_two_dir / f"{view}.png", "long")

            for view in ("left", "right"):
                _save_seed_shape(seed_one_dir / f"{view}.png", "side")
                _save_seed_shape(seed_two_dir / f"{view}.png", "long")

            registry_dir = tmp_path / "registry"
            register_seed(
                SeedRegistrationConfig(
                    name="Round Seed",
                    front_image=str(seed_one_dir / "front.png"),
                    back_image=str(seed_one_dir / "back.png"),
                    left_image=str(seed_one_dir / "left.png"),
                    right_image=str(seed_one_dir / "right.png"),
                    top_image=str(seed_one_dir / "top.png"),
                    bottom_image=str(seed_one_dir / "bottom.png"),
                    registry_dir=str(registry_dir),
                    species="Test",
                )
            )
            register_seed(
                SeedRegistrationConfig(
                    name="Long Seed",
                    front_image=str(seed_two_dir / "front.png"),
                    back_image=str(seed_two_dir / "back.png"),
                    left_image=str(seed_two_dir / "left.png"),
                    right_image=str(seed_two_dir / "right.png"),
                    top_image=str(seed_two_dir / "top.png"),
                    bottom_image=str(seed_two_dir / "bottom.png"),
                    registry_dir=str(registry_dir),
                    species="Test",
                )
            )

            query_path = tmp_path / "query.png"
            _save_seed_shape(query_path, "long")
            result = identify_seed(
                IdentificationConfig(
                    image_path=str(query_path),
                    registry_dir=str(registry_dir),
                    top_k=2,
                )
            )

            self.assertEqual(result["matches"][0]["name"], "Long Seed")
            self.assertGreater(result["matches"][0]["score"], result["matches"][1]["score"])
            self.assertTrue((registry_dir / "index.json").exists())


if __name__ == "__main__":
    unittest.main()
