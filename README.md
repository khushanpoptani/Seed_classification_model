# Seed Scope Django UI

This repository now exposes the simplified seed-image workflow through a Django web interface.

## Main User Flow

1. A user opens the Django site.
2. On the **Register Seed** page, the user uploads 6 directional images:
   - front
   - back
   - left
   - right
   - top
   - bottom
3. The system extracts shape parameters automatically and builds a lightweight voxel-style 3D profile.
4. The seed profile is stored in the local registry.
5. On the **Identify Seed** page, the user uploads 1 seed image.
6. The system compares that query image with all saved seed profiles and returns the best matches.

## Run The UI

Create and activate the virtual environment:

```bash
cd /Volumes/Data/Projects/Major/Seed_classification_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the Django server:

```bash
python manage.py runserver
```

Open the UI in your browser:

```text
http://127.0.0.1:8000/
```

## Pages

- `/`
  - dashboard
  - recent registered seeds
  - quick overview of the flow
- `/register/`
  - 6-image registration form
  - optional metadata fields
  - calculated parameters shown after registration
- `/identify/`
  - single-image query form
  - ranked matching seeds with scores
- `/seed/<seed_id>/`
  - stored profile detail page
  - original view parameters
  - generated projection parameters

## How Registration Works

The Django registration form calls the image pipeline in [image_pipeline.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/image_pipeline.py).

For each uploaded image, the pipeline:

1. segments the seed from the background
2. normalizes the silhouette
3. computes parameters like:
   - area
   - perimeter
   - compactness
   - width and height
   - aspect ratio
   - major and minor axis
   - asymmetry
4. combines the 6 silhouettes into a voxel-style 3D seed profile
5. saves the profile under `artifacts/image_registry/<seed_id>/`

## How Identification Works

When the user uploads one query image, the pipeline:

1. extracts the query silhouette
2. computes its descriptor
3. compares it against:
   - all stored training views
   - all generated voxel projections
4. computes similarity scores
5. returns ranked matches

## Saved Data

Generated seed profiles are stored under:

```text
artifacts/image_registry/
```

Uploaded web images are stored under:

```text
media/uploads/
```

Each registered seed stores:

- `model.json`
- `model_bundle.npz`
- `voxel.npy`
- mask images for each original view
- projection images for each generated view

## Legacy Scripts

The command-line scripts still exist:

- [scripts/train_image.py](/Volumes/Data/Projects/Major/Seed_classification_model/scripts/train_image.py)
- [scripts/identify_seed.py](/Volumes/Data/Projects/Major/Seed_classification_model/scripts/identify_seed.py)

But the main user-facing experience is now the Django UI.

See [Instructions.md](/Volumes/Data/Projects/Major/Seed_classification_model/Instructions.md) for step-by-step usage.
