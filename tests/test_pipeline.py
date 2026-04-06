"""Smoke tests for the tabular pipeline."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seed_classifier.pipeline import TabularTrainingConfig, train_and_evaluate_tabular_models


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = ROOT / "artifacts" / "test-run"
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def tearDown(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_demo_pipeline_trains_random_forest(self) -> None:
        config = TabularTrainingConfig(
            use_demo_data=True,
            model_name="random_forest",
            output_dir=str(self.output_dir),
            random_state=7,
        )
        results = train_and_evaluate_tabular_models(config)
        self.assertIn("random_forest", results)
        self.assertGreaterEqual(results["random_forest"]["accuracy"], 0.70)
        self.assertTrue((self.output_dir / "random_forest" / "model.pkl").exists())


if __name__ == "__main__":
    unittest.main()
