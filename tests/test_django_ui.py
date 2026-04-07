"""Smoke tests for the Django UI workflow."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "seedweb.settings")

import django

django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.test.utils import override_settings


def _make_image(box) -> bytes:
    image = Image.new("L", (160, 160), color=255)
    draw = ImageDraw.Draw(image)
    draw.ellipse(box, fill=20)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class DjangoUiSmokeTestCase(unittest.TestCase):
    def test_registration_and_identification_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_root = Path(tmp) / "media"
            registry_root = Path(tmp) / "registry"
            with override_settings(MEDIA_ROOT=media_root, SEED_REGISTRY_DIR=registry_root):
                client = Client()

                register_payload = {
                    "name": "Django Seed",
                    "species": "Wheat",
                    "source": "Automated test",
                    "notes": "Generated during smoke verification",
                    "front": SimpleUploadedFile("front.png", _make_image((28, 45, 132, 115)), content_type="image/png"),
                    "back": SimpleUploadedFile("back.png", _make_image((28, 45, 132, 115)), content_type="image/png"),
                    "left": SimpleUploadedFile("left.png", _make_image((18, 56, 142, 104)), content_type="image/png"),
                    "right": SimpleUploadedFile("right.png", _make_image((18, 56, 142, 104)), content_type="image/png"),
                    "top": SimpleUploadedFile("top.png", _make_image((46, 26, 114, 134)), content_type="image/png"),
                    "bottom": SimpleUploadedFile("bottom.png", _make_image((46, 26, 114, 134)), content_type="image/png"),
                }

                register_response = client.post("/register/", register_payload)
                register_body = register_response.content.decode("utf-8")

                self.assertEqual(register_response.status_code, 200)
                self.assertIn("Seed Registered", register_body)
                self.assertIn("Django Seed", register_body)

                identify_payload = {
                    "top_k": 3,
                    "image": SimpleUploadedFile(
                        "query.png",
                        _make_image((26, 47, 134, 113)),
                        content_type="image/png",
                    ),
                }
                identify_response = client.post("/identify/", identify_payload)
                identify_body = identify_response.content.decode("utf-8")

                self.assertEqual(identify_response.status_code, 200)
                self.assertIn("Best Matches", identify_body)
                self.assertIn("Django Seed", identify_body)

                home_response = client.get("/")
                self.assertEqual(home_response.status_code, 200)
                self.assertIn("Registered Profiles", home_response.content.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
