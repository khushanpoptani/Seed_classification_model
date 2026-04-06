# Thesis Mapping

This file maps the thesis content to the repository implementation.

## Extracted Thesis Requirements

The document contains these major implementation requirements:

- Automated seed classification
- Python-based implementation
- Preprocessing, feature extraction, training, testing, and prediction
- Comparison models and graphical evaluation
- Accuracy, precision, recall, F1 score, and confusion matrix
- Transfer learning with VGG16 for a 14-class image dataset
- Support for future website/mobile deployment

## Conflicts Found In The Thesis

The document is not internally consistent. The main conflicts are:

1. The CNN chapters describe a 14-class image dataset and VGG16 transfer learning.
2. The proposed work chapter cites the UCI Seeds dataset, which is a tabular dataset with 3 classes.
3. Some preprocessing text refers to unrelated dermoscopic-image and hair-removal steps.
4. The result section mentions Random Forest, Naive Bayes, and Decision Tree, which fit the tabular dataset better than the VGG16 image story.

## How This Repository Resolves Those Conflicts

- The tabular pipeline is implemented completely because the thesis provides enough concrete feature definitions for it.
- The image pipeline is scaffolded and documented because the thesis describes the architecture but does not provide the dataset needed to train it here.
- The project documentation keeps the thesis structure visible so it remains useful for presentations, demos, and viva preparation.

## Chapter To Code Mapping

- Chapter 1 Introduction
  - Reflected in [README.md](/Volumes/Data/Projects/Major/Seed_classification_model/README.md)
- Chapter 3 Methodology
  - Reflected in [src/seed_classifier/pipeline.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/pipeline.py)
  - Reflected in [src/seed_classifier/preprocessing.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/preprocessing.py)
- Chapter 4 Proposed Work
  - Reflected in [src/seed_classifier/datasets.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/datasets.py)
  - Reflected in [src/seed_classifier/models/gaussian_nb.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/models/gaussian_nb.py)
  - Reflected in [src/seed_classifier/models/decision_tree.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/models/decision_tree.py)
  - Reflected in [src/seed_classifier/models/random_forest.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/models/random_forest.py)
  - Reflected in [src/seed_classifier/models/mlp.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/models/mlp.py)
- Chapter 5 Design and Implementation
  - Reflected in [docs/system_design.md](/Volumes/Data/Projects/Major/Seed_classification_model/docs/system_design.md)
  - Reflected in [scripts/train_tabular.py](/Volumes/Data/Projects/Major/Seed_classification_model/scripts/train_tabular.py)
  - Reflected in [scripts/predict_tabular.py](/Volumes/Data/Projects/Major/Seed_classification_model/scripts/predict_tabular.py)
- Chapter 6 Result and Analysis
  - Reflected in [src/seed_classifier/metrics.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/metrics.py)
- Chapter 7 Conclusion and Future Scope
  - Reflected in [src/seed_classifier/image_pipeline.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/image_pipeline.py)
