# Instructions

This file explains exactly how to run, use, and understand the current Django-based seed registration and identification system.

## 1. Purpose Of The System

This project is designed for two main tasks:

### A. Seed Registration

Register one seed into the system by uploading 6 images of the same seed:
- front
- back
- left
- right
- top
- bottom

The system then calculates the seed parameters automatically and stores a simplified 3D-style seed profile.

### B. Seed Identification

Upload 1 image of an unknown seed and compare it with all registered seed profiles. The system returns the top matching seeds with similarity scores.

## 2. Before You Start

Make sure you are in the project folder:

```bash
cd /Volumes/Data/Projects/Major/Seed_classification_model
```

## 3. Install And Start The Project

### Step 1. Create Virtual Environment

```bash
python3 -m venv .venv
```

### Step 2. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Step 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 4. Start Django Server

```bash
python manage.py runserver
```

### Step 5. Open In Browser

```text
http://127.0.0.1:8000/
```

## 4. Main Pages

### Home Page

URL:

```text
http://127.0.0.1:8000/
```

Use:
- view project overview
- access register page
- access identify page
- view recent seeds

### Register Seed Page

URL:

```text
http://127.0.0.1:8000/register/
```

Use:
- upload 6 images for one seed
- enter seed metadata
- create a saved seed profile

### Identify Seed Page

URL:

```text
http://127.0.0.1:8000/identify/
```

Use:
- upload 1 query image
- compare with all stored seeds
- see ranked matches

### Seed Detail Page

URL format:

```text
http://127.0.0.1:8000/seed/<seed_id>/
```

Use:
- inspect saved parameters for each original view
- inspect generated projection parameters
- review the stored seed profile

## 5. How To Register Or Train A Seed

Go to:

```text
http://127.0.0.1:8000/register/
```

Fill in these fields:
- `Seed Name`
- `Species` optional
- `Source` optional
- `Notes` optional
- `Front View`
- `Back View`
- `Left View`
- `Right View`
- `Top View`
- `Bottom View`

Then click the register button.

### What Happens Internally

When you submit the registration form:

1. Django receives the form and uploaded images.
2. The uploaded files are saved in:
   - `media/uploads/registration/`
3. The backend sends the image paths to the image pipeline.
4. Each image is converted into a binary seed silhouette.
5. The system calculates shape parameters from each silhouette.
6. The system builds a simplified voxel-style 3D model from all 6 views.
7. The system creates generated projection views from that voxel model.
8. The system saves the complete seed profile in:
   - `artifacts/image_registry/<seed_id>/`
9. The system updates:
   - `artifacts/image_registry/index.json`

### What Is Meant By "Training"

In this project, training does not mean deep-learning model training with epochs and neural network weights.

Here, "training" means:
- capturing a seed from 6 directions
- extracting its shape information
- storing that information as a reusable seed profile

That stored profile is later used for comparison.

## 6. Parameters Calculated Automatically

The user does not enter these values manually. The system calculates them from the seed image.

### `area_pixels`

Total number of foreground pixels belonging to the seed.

### `area_ratio`

How much of the normalized image area is covered by the seed.

### `perimeter_pixels`

Estimated boundary length of the seed silhouette.

### `compactness`

A shape measure showing how closely the seed shape resembles a compact rounded form.

### `bbox_width`

Width of the seed’s bounding box.

### `bbox_height`

Height of the seed’s bounding box.

### `aspect_ratio`

Width divided by height. This helps describe whether the seed looks more long or more wide.

### `major_axis`

The main longest axis of the seed shape.

### `minor_axis`

The shorter axis of the seed shape.

### `vertical_asymmetry`

Difference between left half and right half of the seed silhouette.

### `horizontal_asymmetry`

Difference between top half and bottom half of the seed silhouette.

## 7. How The 3D-Style Seed Model Is Created

The system creates a simplified voxel representation using the 6 silhouettes.

### Inputs Used

- front + back
- left + right
- top + bottom

### Logic

1. Each image is normalized to the same size.
2. Opposite views are combined to make stable directional projections.
3. These projections are expanded across a 3D voxel grid.
4. The overlapping occupied regions form the final voxel volume.
5. New 2D projections are generated back from this voxel volume.

### Important Note

This is a simplified 3D approximation, not a photorealistic 3D scan.

## 8. How Identification Works

Go to:

```text
http://127.0.0.1:8000/identify/
```

Upload:
- one image of the seed from any visible side

Choose:
- number of top matches to display

Then submit the form.

### What Happens Internally

1. The query image is saved in:
   - `media/uploads/queries/`
2. The query image is converted into a binary silhouette.
3. The query parameters are calculated.
4. The query descriptor is created.
5. The system loads every registered seed from the registry.
6. The query is compared against:
   - all stored training views
   - all generated projection views
7. A similarity score is calculated for each candidate seed.
8. The system keeps the best score per seed.
9. The final results are sorted from highest score to lowest score.

## 9. How Comparison Is Done

The project compares seed shapes, not just filenames or manually typed parameters.

The comparison uses:
- overlap between silhouettes
- descriptor similarity
- row profile similarity
- column profile similarity

### In Simple Language

The system checks:
- how much the query shape overlaps with a stored seed shape
- how similar the overall structure looks
- how the seed is distributed vertically
- how the seed is distributed horizontally

The highest score means the strongest visual match.

## 10. Where Data Is Saved

### A. Uploaded Images

Saved in:

```text
media/uploads/
```

Subfolders:
- `media/uploads/registration/`
- `media/uploads/queries/`

### B. Registered Seed Profiles

Saved in:

```text
artifacts/image_registry/
```

Each registered seed gets its own folder:

```text
artifacts/image_registry/<seed_id>/
```

Typical saved files:
- `model.json`
- `model_bundle.npz`
- `voxel.npy`
- `front_mask.png`
- `back_mask.png`
- `left_mask.png`
- `right_mask.png`
- `top_mask.png`
- `bottom_mask.png`
- `front_projection.png`
- `back_projection.png`
- `left_projection.png`
- `right_projection.png`
- `top_projection.png`
- `bottom_projection.png`

### C. Registry Index

Saved in:

```text
artifacts/image_registry/index.json
```

This file stores the master list of registered seeds used by the UI.

## 11. What Each Saved File Means

### `model.json`

Contains:
- seed metadata
- per-view parameters
- projection parameters
- storage paths

### `model_bundle.npz`

Compressed NumPy bundle containing:
- original training masks
- generated projection masks
- voxel data

### `voxel.npy`

The simplified 3D voxel representation of the seed.

### `*_mask.png`

Binary masks created from the original uploaded images.

### `*_projection.png`

Projection images generated from the voxel model.

## 12. Command-Line Alternatives

The web UI is the main workflow, but CLI scripts are also available.

### Register Through CLI

```bash
python3 scripts/train_image.py \
  --name "Sample Seed" \
  --front path/to/front.jpg \
  --back path/to/back.jpg \
  --left path/to/left.jpg \
  --right path/to/right.jpg \
  --top path/to/top.jpg \
  --bottom path/to/bottom.jpg
```

### Identify Through CLI

```bash
python3 scripts/identify_seed.py \
  --image path/to/query.jpg \
  --top-k 5
```

## 13. Validation And Testing

Run these commands to verify the project:

```bash
python manage.py check
python -m unittest discover -s tests -p "test_*.py"
```

## 14. Good Image Capture Practices

For better results:
- use a plain and contrasting background
- keep lighting even
- avoid strong shadows
- avoid blurry photos
- keep the seed centered
- keep scale reasonably consistent
- ensure each of the 6 views is clear

## 15. Current Limitation

This system is based on image-shape comparison and a simplified voxel model. It is not currently using a CNN or deep neural network for seed classification.

## 16. Quick Summary

### Registration Flow

```text
Upload 6 images
-> save files
-> extract seed silhouettes
-> calculate parameters
-> build voxel model
-> generate projections
-> save seed profile
```

### Identification Flow

```text
Upload 1 image
-> save file
-> extract query silhouette
-> compare with all stored views and projections
-> score matches
-> rank results
-> show top seeds
```

For a shorter project overview, see [README.md](/Volumes/Data/Projects/Major/Seed_classification_model/README.md).
