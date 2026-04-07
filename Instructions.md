# Instructions

This project now runs as a Django web application.

## 1. Install And Start The UI

```bash
cd /Volumes/Data/Projects/Major/Seed_classification_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## 2. Register Or Train A Seed

Go to:

```text
http://127.0.0.1:8000/register/
```

Fill in:

- Seed name
- Species (optional)
- Source (optional)
- Notes (optional)
- Front image
- Back image
- Left image
- Right image
- Top image
- Bottom image

Then click:

```text
Register Seed
```

What the system does:

1. saves the uploaded images temporarily in `media/uploads/registration/`
2. extracts the seed silhouette from each image
3. calculates parameters automatically
4. builds a lightweight voxel-style 3D seed profile
5. saves the final registered seed model into `artifacts/image_registry/`

## 3. Parameters Calculated Automatically

The user does not manually enter these values.

The system calculates:

- area
- perimeter
- compactness
- bounding box width
- bounding box height
- aspect ratio
- major axis
- minor axis
- vertical asymmetry
- horizontal asymmetry

## 4. Identify A Seed

Go to:

```text
http://127.0.0.1:8000/identify/
```

Upload:

- one seed image from any visible side

Choose:

- how many top matches to return

Then click:

```text
Identify Seed
```

What the system does:

1. saves the query image in `media/uploads/queries/`
2. extracts the query silhouette
3. compares it with every registered seed
4. checks both:
   - stored original training views
   - generated 3D-style projections
5. ranks the best matches
6. shows the results in the browser

## 5. Where Data Is Saved

### Uploaded web images

Saved in:

```text
media/uploads/
```

### Registered seed profiles

Saved in:

```text
artifacts/image_registry/
```

Each seed gets its own folder containing:

- `model.json`
- `model_bundle.npz`
- `voxel.npy`
- `*_mask.png`
- `*_projection.png`

### Registry index

The registry also stores:

```text
artifacts/image_registry/index.json
```

This keeps the list of all registered seeds for the UI.

## 6. How The 3D Model Works

This is a simplified 3D architecture, not a photorealistic reconstruction.

The system:

1. uses front/back silhouettes as one constraint
2. uses left/right silhouettes as another constraint
3. uses top/bottom silhouettes as a third constraint
4. combines them in a voxel grid
5. stores that voxel grid as the seed's 3D-style profile

## 7. Useful Pages

- `/`
  - dashboard
- `/register/`
  - seed registration form
- `/identify/`
  - one-image identification form
- `/seed/<seed_id>/`
  - saved profile details

## 8. Optional CLI Mode

The old image scripts still work if needed:

```bash
python3 scripts/train_image.py ...
python3 scripts/identify_seed.py ...
```

But the main user workflow is now Django-based.
