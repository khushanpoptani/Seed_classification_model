# Seed Scope Django UI

This project is a Django-based seed registration and identification system. Users register a seed by uploading 6 images of the same seed from different directions, and the system builds a simplified 3D-style seed profile from those views. Later, a user can upload 1 query image and the system compares it against all registered seed profiles to return the closest matches.

## What The Project Does

- Registers a seed from 6 directional images:
  - front
  - back
  - left
  - right
  - top
  - bottom
- Extracts the seed silhouette automatically from each image
- Calculates shape-based parameters from each view
- Builds a lightweight voxel-style 3D representation from the 6 silhouettes
- Saves the registered seed in a local image registry
- Identifies an unknown seed from a single query image
- Returns ranked matches with similarity scores

## Main Workflow

### 1. Register A Seed

The user opens the Django UI and goes to `/register/`.

The form asks for:
- seed name
- species (optional)
- source (optional)
- notes (optional)
- 6 seed images

After submission, the system:
1. saves the uploaded images under `media/uploads/registration/`
2. preprocesses each image into a normalized binary silhouette
3. computes shape parameters for each view
4. builds a simplified voxel-style 3D seed model
5. generates 6 projection views from that model
6. stores the complete seed profile under `artifacts/image_registry/<seed_id>/`
7. updates the seed index in `artifacts/image_registry/index.json`

### 2. Identify A Seed

The user opens `/identify/` and uploads 1 seed image.

After submission, the system:
1. saves the query image under `media/uploads/queries/`
2. preprocesses the image into a normalized silhouette
3. computes the query descriptor and shape parameters
4. compares the query against every registered seed
5. checks similarity against:
   - original training views
   - voxel-generated projection views
6. sorts all seeds by similarity score
7. shows the top matches in the browser

## Tech Stack

- Python 3
- Django 4.2
- NumPy
- Pillow

## Project Structure

```text
Seed_classification_model/
├── manage.py
├── README.md
├── Instructions.md
├── requirements.txt
├── seedweb/                  # Django project settings
├── seedui/                   # Django app for forms and pages
├── templates/seedui/         # HTML templates
├── src/seed_classifier/      # Image and legacy tabular pipelines
├── scripts/                  # CLI entry points
├── tests/                    # Unit and Django UI tests
├── media/                    # Uploaded files at runtime
└── artifacts/image_registry/ # Saved seed profiles
```

## Important Files

- [manage.py](/Volumes/Data/Projects/Major/Seed_classification_model/manage.py)
  Django entry point
- [seedweb/settings.py](/Volumes/Data/Projects/Major/Seed_classification_model/seedweb/settings.py)
  Django configuration
- [seedweb/urls.py](/Volumes/Data/Projects/Major/Seed_classification_model/seedweb/urls.py)
  Root routing
- [seedui/views.py](/Volumes/Data/Projects/Major/Seed_classification_model/seedui/views.py)
  Registration, identification, and detail-page logic
- [seedui/forms.py](/Volumes/Data/Projects/Major/Seed_classification_model/seedui/forms.py)
  Django forms for image upload
- [src/seed_classifier/image_pipeline.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/image_pipeline.py)
  Core registration and identification pipeline
- [templates/seedui/register.html](/Volumes/Data/Projects/Major/Seed_classification_model/templates/seedui/register.html)
  Seed registration UI
- [templates/seedui/identify.html](/Volumes/Data/Projects/Major/Seed_classification_model/templates/seedui/identify.html)
  Seed identification UI

## Setup

### 1. Create A Virtual Environment

```bash
cd Seed_classification_model
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start The Django Server

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Available Pages

- `/`
  Dashboard with quick actions and recent seeds
- `/register/`
  Upload 6 views and create a seed profile
- `/identify/`
  Upload 1 image and get ranked seed matches
- `/seed/<seed_id>/`
  View the stored details of a registered seed

## How The Image Pipeline Works

The pipeline is implemented in [image_pipeline.py](/Volumes/Data/Projects/Major/Seed_classification_model/src/seed_classifier/image_pipeline.py).

### A. Image Preprocessing

Each uploaded image is:
1. converted to grayscale
2. thresholded to separate seed from background
3. cropped to the largest seed region
4. centered on a square canvas
5. resized to a normalized mask size

Output:
- a clean binary mask of the seed silhouette

### B. Automatic Parameters

For each view, the system calculates:
- `area_pixels`
- `area_ratio`
- `perimeter_pixels`
- `compactness`
- `bbox_width`
- `bbox_height`
- `aspect_ratio`
- `major_axis`
- `minor_axis`
- `vertical_asymmetry`
- `horizontal_asymmetry`

These values are calculated from the silhouette automatically. The user does not enter them manually.

### C. Seed Descriptor

The system also builds a descriptor from:
- key shape parameters
- row density profile
- column density profile

This descriptor helps compare one seed silhouette to another.

### D. Simplified 3D Model

The system combines the 6 orthogonal silhouettes into a voxel grid:
- front + back
- left + right
- top + bottom

This produces a simplified 3D-style volume, not a photorealistic 3D reconstruction.

### E. Matching Logic

When a query image is uploaded:
1. the query mask and descriptor are generated
2. the query is compared to every registered seed
3. each registered seed is checked against:
   - saved training views
   - generated projection views
4. similarity is scored using:
   - IoU overlap
   - descriptor similarity
   - row profile similarity
   - column profile similarity
5. the highest-scoring seeds are returned

## Saved Data

### Uploaded Files

Temporary web uploads are stored under:

```text
media/uploads/
```

Examples:
- `media/uploads/registration/`
- `media/uploads/queries/`

### Registered Seed Profiles

Each seed is stored under:

```text
artifacts/image_registry/<seed_id>/
```

Typical files:
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

### Registry Index

All registered seeds are listed in:

```text
artifacts/image_registry/index.json
```

## CLI Scripts

The Django UI is the main interface, but CLI scripts are also available:

- [scripts/train_image.py](/Volumes/Data/Projects/Major/Seed_classification_model/scripts/train_image.py)
  Register a seed from 6 image paths
- [scripts/identify_seed.py](/Volumes/Data/Projects/Major/Seed_classification_model/scripts/identify_seed.py)
  Identify a seed from 1 query image

Legacy tabular scripts also remain in the repo, but the current primary workflow is image-based and UI-driven.

## Testing

Run the main checks with:

```bash
python manage.py check
python -m unittest discover -s tests -p "test_*.py"
```

## Limitations

- This is a simplified image-comparison system, not a deep-learning classifier
- Accuracy depends heavily on image quality and background contrast
- The voxel model is an approximation built from silhouettes
- Similar seeds with nearly identical silhouettes may produce close scores

## Recommended Image Capture Tips

- Use a plain background
- Keep lighting consistent
- Keep the seed centered in the frame
- Avoid shadows crossing the seed boundary
- Capture the same seed clearly from all 6 directions
- Keep scale roughly similar across views

## More Help

For step-by-step operating instructions, see [Instructions.md](/Volumes/Data/Projects/Major/Seed_classification_model/Instructions.md).
