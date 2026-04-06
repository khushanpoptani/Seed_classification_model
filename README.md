# Seed Classification Model Project

This repository turns the attached thesis into a practical seed-classification project.

The thesis mixes two different problem statements:

- A tabular UCI Seeds dataset with 3 classes: `Kama`, `Rosa`, and `Canadian`
- A 14-class image-classification pipeline built with transfer learning on `VGG16`

To keep the project usable, this repository implements both in a structured way:

- A working tabular pipeline using only the Python packages already available here (`numpy` and `pandas`)
- An optional image-classification scaffold that matches the thesis design and can be activated once deep-learning dependencies and the 14-class image dataset are available

## Project Structure

```text
.
├── Seed_Thesis.docx
├── README.md
├── requirements.txt
├── requirements-image.txt
├── docs/
│   ├── system_design.md
│   └── thesis_mapping.md
├── scripts/
│   ├── predict_tabular.py
│   ├── train_image.py
│   └── train_tabular.py
├── src/
│   └── seed_classifier/
│       ├── __init__.py
│       ├── datasets.py
│       ├── image_pipeline.py
│       ├── metrics.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       └── models/
│           ├── __init__.py
│           ├── decision_tree.py
│           ├── gaussian_nb.py
│           ├── mlp.py
│           └── random_forest.py
└── tests/
    ├── test_metrics.py
    └── test_pipeline.py
```

## What The Project Covers From The Thesis

- Problem statement and motivation for automated seed classification
- UCI Seeds-style feature-based classification workflow
- Preprocessing, training, testing, prediction, and evaluation
- Classical comparison models:
  - Gaussian Naive Bayes
  - Decision Tree
  - Random Forest
- ANN-style prediction through a NumPy multi-layer perceptron
- Confusion matrix and macro precision/recall/F1 reporting
- VGG16 transfer-learning design notes for the 14-class image pipeline
- System design documentation, including DFD and UML-style diagrams

## Quick Start

Use the built-in demo dataset first:

```bash
python3 scripts/train_tabular.py --demo-data --model all
```

Train with your own UCI Seeds-style dataset:

```bash
python3 scripts/train_tabular.py --dataset data/raw/seeds_dataset.csv --model all
```

Run inference from a saved artifact:

```bash
python3 scripts/predict_tabular.py \
  --artifact-dir artifacts/tabular/random_forest \
  --area 15.2 \
  --perimeter 14.7 \
  --compactness 0.87 \
  --kernel-length 5.6 \
  --kernel-width 3.2 \
  --asymmetry 2.1 \
  --groove-length 5.1
```

Run tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Dataset Format

The tabular pipeline expects the following columns:

- `area`
- `perimeter`
- `compactness`
- `kernel_length`
- `kernel_width`
- `asymmetry_coefficient`
- `kernel_groove_length`
- `label`

Labels may be:

- Text labels: `Kama`, `Rosa`, `Canadian`
- Integer labels: `1`, `2`, `3`

If you do not have the original dataset yet, `--demo-data` creates a synthetic thesis-style dataset with the same feature names and classes so the pipeline can be exercised end to end.

## Output Artifacts

Each training run writes files under `artifacts/tabular/<model_name>/`:

- `model.pkl`: trained model + scaler + label mapping
- `metrics.json`: evaluation metrics
- `predictions.csv`: predictions for the held-out split

## Image Pipeline

The image pipeline mirrors the thesis architecture:

- Input size `224 x 224 x 3`
- Transfer learning with `VGG16`
- Added layers:
  - average pooling
  - flatten
  - dense + ReLU
  - dropout `0.5`
  - softmax output

Because this environment does not currently include `tensorflow`, `keras`, or image-processing libraries, the repository includes a ready-to-extend scaffold and dataset-layout validation instead of a fake training script.

## Notes On Thesis Consistency

The thesis includes several conflicting details, including:

- 14 image classes in the CNN section
- 3 tabular classes in the UCI Seeds section
- references to unrelated medical-image preprocessing language

This repository keeps those details visible but separates them into:

- a working tabular implementation
- an optional image extension

See [docs/thesis_mapping.md](/Volumes/Data/Projects/Major/Seed_classification_model/docs/thesis_mapping.md) for the full mapping.
