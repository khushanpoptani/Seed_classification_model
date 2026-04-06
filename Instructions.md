# Instructions

This file explains the prediction command arguments, the meaning of each seed factor, and how the project saves and uses data.

## Important Clarification

This command:

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

does **not** train or save a new model.

It does this:

1. Loads an already trained model from `artifacts/tabular/random_forest`
2. Takes the 7 input factors you provide
3. Scales them using the same scaler used during training
4. Predicts the seed class
5. Prints the result

The model is actually trained and saved during:

```bash
python3 scripts/train_tabular.py --demo-data --model all
```

or:

```bash
python3 scripts/train_tabular.py --dataset data/raw/seeds_demo.csv --model all
```

## Command Arguments Explained

### `--artifact-dir`

Example:

```bash
--artifact-dir artifacts/tabular/random_forest
```

Meaning:

- This is the folder that contains the trained model files.
- It tells the prediction script which saved model to use.

Inside that folder, the project stores:

- `model.pkl`: saved trained model, scaler, label encoder, and config
- `metrics.json`: accuracy, precision, recall, F1 score, and confusion matrix
- `predictions.csv`: true and predicted labels on the test set

This is **not** a seed feature. It is just the path to the saved model.

## Seed Factors Explained

These 7 values are the input features used by the model to classify the seed.

### `--area`

Example:

```bash
--area 15.2
```

Meaning:

- Total surface area of the seed kernel.
- It represents how much 2D space the seed occupies in the image/measurement.

Simple idea:

- Bigger seed -> larger area
- Smaller seed -> smaller area

Why it matters:

- Different seed classes often have different average sizes.

### `--perimeter`

Example:

```bash
--perimeter 14.7
```

Meaning:

- Total length around the outer boundary of the seed.
- It measures the outline of the kernel.

Simple idea:

- If you trace the edge of the seed, the full traced length is the perimeter.

Why it matters:

- Seeds with different shapes and sizes can have different boundary lengths.

### `--compactness`

Example:

```bash
--compactness 0.87
```

Meaning:

- A shape descriptor that tells how compact or round the seed is.
- In the original UCI Seeds dataset, compactness is calculated from area and perimeter.

Common formula:

```text
compactness = 4 * pi * area / perimeter^2
```

Simple idea:

- More regular and rounded shape -> higher compactness
- More stretched or irregular shape -> lower compactness

Why it matters:

- Two seeds may have similar size but different shape regularity.

### `--kernel-length`

Example:

```bash
--kernel-length 5.6
```

Meaning:

- The length of the seed from one end to the other.
- This is the longest main dimension of the kernel.

Simple idea:

- It tells how long the seed is.

Why it matters:

- Some seed varieties are longer than others.

### `--kernel-width`

Example:

```bash
--kernel-width 3.2
```

Meaning:

- The width of the seed measured across the shorter side.
- It tells how broad or thick-looking the seed is in 2D measurement.

Simple idea:

- It tells how wide the seed is.

Why it matters:

- Length and width together help the model understand the seed proportions.

### `--asymmetry`

Example:

```bash
--asymmetry 2.1
```

Meaning:

- A numerical measure of how symmetric or asymmetric the seed shape is.
- It describes whether the two sides of the seed are balanced or uneven.

Simple idea:

- Low asymmetry -> more balanced shape
- High asymmetry -> more uneven shape

Why it matters:

- Some seed classes are more regular, while others have more shape imbalance.

### `--groove-length`

Example:

```bash
--groove-length 5.1
```

Meaning:

- Length of the groove or crease running along the seed kernel.
- In wheat-style seed datasets, this groove is an important identifying feature.

Simple idea:

- It measures the central line or channel visible on the seed.

Why it matters:

- Different seed varieties can have different groove characteristics.

## How The Model Uses These Factors

During training, each row looks like this:

```text
area, perimeter, compactness, kernel_length, kernel_width, asymmetry_coefficient, kernel_groove_length, label
```

Example:

```text
14.6, 14.4, 0.86, 5.64, 3.20, 1.72, 4.86, Kama
```

The first 7 values are inputs.

The last value, `label`, is the correct class:

- `Kama`
- `Rosa`
- `Canadian`

The model learns patterns such as:

- seeds with one combination of size and shape tend to be `Kama`
- another combination tends to be `Rosa`
- another tends to be `Canadian`

## How Data Is Saved

### 1. Training Data

The dataset itself is either:

- generated with `--demo-data`, or
- loaded from a CSV file such as [data/raw/seeds_demo.csv](/Volumes/Data/Projects/Major/Seed_classification_model/data/raw/seeds_demo.csv)

### 2. Saved Model Artifacts

After training, data is saved in folders like:

- [artifacts/tabular/gaussian_nb](/Volumes/Data/Projects/Major/Seed_classification_model/artifacts/tabular/gaussian_nb)
- [artifacts/tabular/decision_tree](/Volumes/Data/Projects/Major/Seed_classification_model/artifacts/tabular/decision_tree)
- [artifacts/tabular/random_forest](/Volumes/Data/Projects/Major/Seed_classification_model/artifacts/tabular/random_forest)
- [artifacts/tabular/mlp](/Volumes/Data/Projects/Major/Seed_classification_model/artifacts/tabular/mlp)

Each folder contains:

### `model.pkl`

This saves:

- trained model parameters
- scaler values
- label encoder
- feature names
- training config

### `metrics.json`

This saves evaluation results such as:

- accuracy
- macro precision
- macro recall
- macro F1 score
- confusion matrix

### `predictions.csv`

This saves test-set comparison results:

- `true_label`
- `predicted_label`

This is how the project stores model performance results.

## How Data Is Compared

The project compares data in two main ways.

### 1. Comparing input patterns to learned patterns

When you enter:

```bash
--area 15.2
--perimeter 14.7
--compactness 0.87
--kernel-length 5.6
--kernel-width 3.2
--asymmetry 2.1
--groove-length 5.1
```

the model checks how close this feature combination is to patterns it learned during training.

It does not compare only one field.

It looks at the full feature combination together.

### 2. Comparing predictions with actual labels

During evaluation, the model predicts labels for the test set and compares:

- actual label
- predicted label

From this comparison, it calculates:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

This tells us how good the model is.

## Example Prediction Flow

Command:

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

Flow:

1. Load `artifacts/tabular/random_forest/model.pkl`
2. Read the 7 factor values
3. Put them into the expected feature order
4. Apply the saved scaler
5. Send the scaled values to the trained random forest model
6. Get predicted class
7. Print the result, such as `Kama`

## Simple Summary

- The **features** describe the seed's size and shape.
- The **training script** learns from many rows of those features plus the correct label.
- The **saved model** stores learned patterns.
- The **prediction script** uses a new seed's 7 factors and predicts which class it matches best.

## Image Workflow

This section explains how to work with seed images instead of CSV rows.

Important:

- The current repository already contains an image pipeline scaffold in [scripts/train_image.py](/Volumes/Data/Projects/Major/Seed_classification_model/scripts/train_image.py) and [image_pipeline.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/image_pipeline.py).
- At the moment, it validates the dataset layout and prints the VGG16 training plan.
- It does **not yet run full TensorFlow image training** in this repository.

Even so, these are the correct steps to prepare image-based training.

### 1. Add Image

Save your seed images inside the image dataset folder:

[data/image](/Volumes/Data/Projects/Major/Seed_classification_model/data/image)

Use this structure:

```text
data/image/
  train/
    class_01/
    class_02/
    class_03/
    ...
    class_14/
  val/
    class_01/
    class_02/
    class_03/
    ...
    class_14/
  test/
    class_01/
    class_02/
    class_03/
    ...
    class_14/
```

What to do:

1. Put training images in `train/<class_name>/`
2. Put validation images in `val/<class_name>/`
3. Put test images in `test/<class_name>/`
4. Keep the class folder names consistent across all three splits

Example:

```text
data/image/
  train/
    class_01/
      seed_001.jpg
      seed_002.jpg
    class_02/
      seed_101.jpg
  val/
    class_01/
      seed_201.jpg
    class_02/
      seed_202.jpg
  test/
    class_01/
      seed_301.jpg
    class_02/
      seed_302.jpg
```

Recommended split:

- `70%` images in `train`
- `15%` images in `val`
- `15%` images in `test`

What each folder means:

- `train/`: images used to teach the model
- `val/`: images used during training to monitor performance
- `test/`: unseen images used for final checking

### 2. Train Data Over Image

First install the image dependencies:

```bash
cd /Volumes/Data/Projects/Major/Seed_classification_model
source .venv/bin/activate
pip install -r requirements-image.txt
```

Then run the image pipeline command:

```bash
python3 scripts/train_image.py --dataset-dir data/image --num-classes 14
```

What this command currently does:

1. Checks whether `train`, `val`, and `test` folders exist
2. Checks the class folder names inside each split
3. Prints the thesis-aligned VGG16 model plan
4. Confirms the expected input shape is `224 x 224 x 3`
5. Shows the added head layers:
   - average pooling
   - flatten
   - dense + ReLU
   - dropout
   - softmax

What full image training is supposed to do conceptually:

1. Read every image from the `train/` folder
2. Resize it to `224 x 224`
3. Convert it into pixel arrays
4. Normalize pixel values
5. Feed the images into a CNN or VGG16 model
6. Learn patterns such as shape, texture, groove, and color
7. Validate using the `val/` images
8. Save the best trained image model

### 3. Test It

In an image project, testing means using the `test/` folder that contains unseen images.

Current repository status:

- The repo can already verify that the `test/` folder exists and is structured correctly.
- The repo does **not yet contain the final TensorFlow evaluation loop** that outputs image accuracy from the test split.

What you should do now:

1. Make sure the `test/` folder is present inside `data/image/`
2. Make sure every class has test images
3. Run:

```bash
python3 scripts/train_image.py --dataset-dir data/image --num-classes 14
```

How to read the output:

- if `dataset_layout.splits.test.exists` is `true`, the test split exists
- if `class_dirs` are listed, the class folders were detected
- if TensorFlow is missing, the script will still print the expected model plan

What full testing will do once image training is implemented:

1. Load the saved trained image model
2. Read images from `data/image/test/`
3. Predict a class for each image
4. Compare predicted class with actual folder label
5. Calculate:
   - accuracy
   - precision
   - recall
   - F1 score
   - confusion matrix

## Image Workflow Summary

```text
Add seed images
  ->
Organize into train / val / test class folders
  ->
Run image pipeline command
  ->
Validate image dataset structure
  ->
Train VGG16 model when TensorFlow training loop is enabled
  ->
Evaluate on test images
  ->
Save trained image model and metrics
```
