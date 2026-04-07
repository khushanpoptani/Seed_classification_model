"""Simplified multi-view image registration and identification pipeline."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    Image = None  # type: ignore

VIEW_NAMES = ("front", "back", "left", "right", "top", "bottom")
NORMALIZED_MASK_SIZE = 96
VOXEL_GRID_SIZE = 48


@dataclass
class SeedRegistrationConfig:
    """Inputs required to register a new seed into the image registry."""

    name: str
    front_image: str
    back_image: str
    left_image: str
    right_image: str
    top_image: str
    bottom_image: str
    registry_dir: str = "artifacts/image_registry"
    species: str = ""
    source: str = ""
    notes: str = ""


@dataclass
class IdentificationConfig:
    """Inputs required to identify a seed from a single image."""

    image_path: str
    registry_dir: str = "artifacts/image_registry"
    top_k: int = 5


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError(
            "Pillow is required for image-based training. Install requirements.txt or "
            "requirements-image.txt before using the image pipeline."
        )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.strip().lower()).strip("-")
    return slug or "seed"


def _unique_seed_id(registry_dir: Path, seed_name: str) -> str:
    base = slugify(seed_name)
    candidate = base
    counter = 2
    while (registry_dir / candidate).exists():
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


def _load_grayscale(path: str | Path) -> np.ndarray:
    _require_pillow()
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_mask_image(mask: np.ndarray, path: Path) -> None:
    _require_pillow()
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image.save(path)


def _otsu_threshold(gray: np.ndarray) -> float:
    values = np.clip((gray * 255).astype(np.uint8), 0, 255)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    cumulative = np.cumsum(hist)
    weighted = np.cumsum(hist * np.arange(256))
    global_mean = weighted[-1] / total

    denominator = cumulative * (total - cumulative)
    denominator[denominator == 0] = 1.0
    between_class = ((global_mean * cumulative - weighted) ** 2) / denominator
    threshold = np.argmax(between_class)
    return float(threshold / 255.0)


def _largest_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("No seed silhouette could be extracted from the image.")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(y_min), int(y_max), int(x_min), int(x_max)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    _require_pillow()
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((size, size), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 127


def preprocess_seed_image(path: str | Path, normalized_size: int = NORMALIZED_MASK_SIZE) -> np.ndarray:
    """Load an image and convert it into a normalized binary seed mask."""

    gray = _load_grayscale(path)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    border_mean = float(border.mean())
    threshold = _otsu_threshold(gray)

    if border_mean >= 0.5:
        mask = gray < threshold
    else:
        mask = gray > threshold

    fill_ratio = float(mask.mean())
    if fill_ratio < 0.01 or fill_ratio > 0.95:
        adaptive = border_mean - 0.08 if border_mean >= 0.5 else border_mean + 0.08
        if border_mean >= 0.5:
            mask = gray < adaptive
        else:
            mask = gray > adaptive

    y_min, y_max, x_min, x_max = _largest_bbox(mask)
    cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

    canvas_size = max(cropped.shape) + 12
    canvas = np.zeros((canvas_size, canvas_size), dtype=bool)
    y_offset = (canvas_size - cropped.shape[0]) // 2
    x_offset = (canvas_size - cropped.shape[1]) // 2
    canvas[y_offset : y_offset + cropped.shape[0], x_offset : x_offset + cropped.shape[1]] = cropped
    return _resize_mask(canvas, normalized_size)


def _downsample_vector(vector: np.ndarray, target_length: int = 16) -> np.ndarray:
    indices = np.linspace(0, len(vector), target_length + 1, dtype=int)
    values = []
    for start, end in zip(indices[:-1], indices[1:]):
        segment = vector[start:end]
        values.append(float(segment.mean()) if segment.size else 0.0)
    return np.array(values, dtype=np.float32)


def compute_shape_parameters(mask: np.ndarray) -> Dict[str, float]:
    """Compute simple image-based seed parameters from the binary silhouette."""

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Seed mask is empty.")

    area_pixels = float(mask.sum())
    area_ratio = float(mask.mean())

    padded = np.pad(mask.astype(np.uint8), 1)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    boundary = mask & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    perimeter_pixels = float(boundary.sum())
    compactness = float((4.0 * math.pi * area_pixels) / ((perimeter_pixels**2) + 1e-9))

    ys = coords[:, 0].astype(np.float32)
    xs = coords[:, 1].astype(np.float32)
    bbox_height = float(ys.max() - ys.min() + 1.0)
    bbox_width = float(xs.max() - xs.min() + 1.0)
    aspect_ratio = float(bbox_width / max(bbox_height, 1.0))

    if coords.shape[0] > 1:
        cov = np.cov(np.stack([xs, ys]), bias=True)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    else:
        eigenvalues = np.array([0.0, 0.0], dtype=np.float32)
    major_axis = float((4.0 * math.sqrt(max(float(eigenvalues[0]), 1e-9))) / mask.shape[1])
    minor_axis = float((4.0 * math.sqrt(max(float(eigenvalues[1]), 1e-9))) / mask.shape[0])

    left_half = mask[:, : mask.shape[1] // 2]
    right_half = mask[:, mask.shape[1] - left_half.shape[1] :]
    vertical_asymmetry = float(np.mean(np.abs(left_half.astype(np.float32) - np.fliplr(right_half).astype(np.float32))))

    top_half = mask[: mask.shape[0] // 2, :]
    bottom_half = mask[mask.shape[0] - top_half.shape[0] :, :]
    horizontal_asymmetry = float(
        np.mean(np.abs(top_half.astype(np.float32) - np.flipud(bottom_half).astype(np.float32)))
    )

    return {
        "area_pixels": area_pixels,
        "area_ratio": area_ratio,
        "perimeter_pixels": perimeter_pixels,
        "compactness": compactness,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "aspect_ratio": aspect_ratio,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "vertical_asymmetry": vertical_asymmetry,
        "horizontal_asymmetry": horizontal_asymmetry,
    }


def build_descriptor(mask: np.ndarray) -> np.ndarray:
    """Build a compact descriptor from the normalized silhouette."""

    params = compute_shape_parameters(mask)
    row_profile = _downsample_vector(mask.mean(axis=1), 16)
    col_profile = _downsample_vector(mask.mean(axis=0), 16)

    descriptor = np.concatenate(
        [
            np.array(
                [
                    params["area_ratio"],
                    params["compactness"],
                    params["aspect_ratio"],
                    params["major_axis"],
                    params["minor_axis"],
                    params["vertical_asymmetry"],
                    params["horizontal_asymmetry"],
                ],
                dtype=np.float32,
            ),
            row_profile,
            col_profile,
        ]
    )
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm else descriptor


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def score_view_similarity(query_mask: np.ndarray, query_descriptor: np.ndarray, candidate_mask: np.ndarray) -> float:
    """Compare a query image to one stored seed view or projection."""

    if candidate_mask.shape != query_mask.shape:
        candidate_mask = _resize_mask(candidate_mask, query_mask.shape[0])

    candidate_descriptor = build_descriptor(candidate_mask)
    intersection = float(np.logical_and(query_mask, candidate_mask).sum())
    union = float(np.logical_or(query_mask, candidate_mask).sum()) or 1.0
    iou = intersection / union

    query_rows = _downsample_vector(query_mask.mean(axis=1), 16)
    query_cols = _downsample_vector(query_mask.mean(axis=0), 16)
    candidate_rows = _downsample_vector(candidate_mask.mean(axis=1), 16)
    candidate_cols = _downsample_vector(candidate_mask.mean(axis=0), 16)

    row_similarity = _cosine_similarity(query_rows, candidate_rows)
    col_similarity = _cosine_similarity(query_cols, candidate_cols)
    descriptor_similarity = _cosine_similarity(query_descriptor, candidate_descriptor)

    return float(0.45 * iou + 0.35 * descriptor_similarity + 0.10 * row_similarity + 0.10 * col_similarity)


def _pair_union(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.logical_or(first, second)


def build_voxel_model(view_masks: Dict[str, np.ndarray], grid_size: int = VOXEL_GRID_SIZE) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build a lightweight voxel-style model from six orthogonal seed silhouettes."""

    resized = {view: _resize_mask(mask, grid_size) for view, mask in view_masks.items()}

    front_projection = _pair_union(resized["front"], resized["back"])
    side_projection = _pair_union(resized["left"], resized["right"])
    top_projection = _pair_union(resized["top"], resized["bottom"])

    front_expand = front_projection[:, :, None]  # y, x, 1
    side_expand = side_projection[:, None, :]  # y, 1, z
    top_expand = top_projection.T[None, :, :]  # 1, x, z

    voxel = front_expand & side_expand & top_expand
    if not voxel.any():
        voxel = front_expand & side_expand

    front = voxel.any(axis=2)
    left = voxel.any(axis=1)
    top = voxel.any(axis=0).T

    projections = {
        "front": front,
        "back": np.fliplr(front),
        "left": left,
        "right": np.fliplr(left),
        "top": top,
        "bottom": np.flipud(top),
    }
    return voxel.astype(np.uint8), projections


def _update_registry_index(registry_dir: Path, record: Dict[str, object]) -> None:
    index_path = registry_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"seeds": []}

    seeds = [entry for entry in index.get("seeds", []) if entry.get("seed_id") != record["seed_id"]]
    seeds.append(record)
    index["seeds"] = sorted(seeds, key=lambda entry: str(entry["name"]).lower())
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def register_seed(config: SeedRegistrationConfig) -> Dict[str, object]:
    """Register a seed from six directional images."""

    registry_dir = Path(config.registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)

    seed_id = _unique_seed_id(registry_dir, config.name)
    seed_dir = registry_dir / seed_id
    seed_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {
        "front": config.front_image,
        "back": config.back_image,
        "left": config.left_image,
        "right": config.right_image,
        "top": config.top_image,
        "bottom": config.bottom_image,
    }

    view_masks: Dict[str, np.ndarray] = {}
    view_summary: Dict[str, Dict[str, object]] = {}

    for view_name in VIEW_NAMES:
        image_path = image_paths[view_name]
        mask = preprocess_seed_image(image_path)
        view_masks[view_name] = mask
        params = compute_shape_parameters(mask)
        descriptor = build_descriptor(mask)
        view_summary[view_name] = {
            "source_image": str(Path(image_path).resolve()),
            "parameters": params,
            "descriptor": descriptor.tolist(),
        }
        _save_mask_image(mask, seed_dir / f"{view_name}_mask.png")

    voxel, projections = build_voxel_model(view_masks)
    np.save(seed_dir / "voxel.npy", voxel)

    projection_summary: Dict[str, Dict[str, object]] = {}
    for view_name, projection in projections.items():
        projection_summary[view_name] = {
            "parameters": compute_shape_parameters(projection),
            "descriptor": build_descriptor(projection).tolist(),
        }
        _save_mask_image(projection, seed_dir / f"{view_name}_projection.png")

    arrays_to_save = {f"view_{view}": mask.astype(np.uint8) for view, mask in view_masks.items()}
    arrays_to_save.update({f"projection_{view}": mask.astype(np.uint8) for view, mask in projections.items()})
    arrays_to_save["voxel"] = voxel.astype(np.uint8)
    np.savez_compressed(seed_dir / "model_bundle.npz", **arrays_to_save)

    metadata = {
        "seed_id": seed_id,
        "name": config.name,
        "species": config.species,
        "source": config.source,
        "notes": config.notes,
        "created_at": _timestamp(),
        "views": view_summary,
        "projections": projection_summary,
        "storage": {
            "seed_dir": str(seed_dir.resolve()),
            "bundle": str((seed_dir / "model_bundle.npz").resolve()),
            "voxel": str((seed_dir / "voxel.npy").resolve()),
        },
    }
    (seed_dir / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    index_record = {
        "seed_id": seed_id,
        "name": config.name,
        "species": config.species,
        "source": config.source,
        "seed_dir": str(seed_dir.resolve()),
        "model_json": str((seed_dir / "model.json").resolve()),
        "created_at": metadata["created_at"],
    }
    _update_registry_index(registry_dir, index_record)

    return {
        "status": "registered",
        "seed_id": seed_id,
        "name": config.name,
        "registry_dir": str(registry_dir.resolve()),
        "seed_dir": str(seed_dir.resolve()),
        "views": {view: {"parameters": data["parameters"]} for view, data in view_summary.items()},
    }


def _load_registry_index(registry_dir: Path) -> Dict[str, object]:
    index_path = registry_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Registry index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _load_seed_bundle(seed_dir: Path) -> Dict[str, np.ndarray]:
    with np.load(seed_dir / "model_bundle.npz") as bundle:
        return {key: bundle[key].astype(bool) for key in bundle.files}


def identify_seed(config: IdentificationConfig) -> Dict[str, object]:
    """Identify a seed from a single image against all registered seeds."""

    registry_dir = Path(config.registry_dir)
    index = _load_registry_index(registry_dir)

    query_mask = preprocess_seed_image(config.image_path)
    query_descriptor = build_descriptor(query_mask)
    query_parameters = compute_shape_parameters(query_mask)

    matches = []
    for entry in index.get("seeds", []):
        seed_dir = Path(entry["seed_dir"])
        bundle = _load_seed_bundle(seed_dir)
        metadata = json.loads((seed_dir / "model.json").read_text(encoding="utf-8"))

        best_score = -1.0
        best_view = ""
        best_source = ""

        for view_name in VIEW_NAMES:
            for prefix, source_label in [("view", "training_view"), ("projection", "voxel_projection")]:
                candidate_mask = bundle[f"{prefix}_{view_name}"]
                score = score_view_similarity(query_mask, query_descriptor, candidate_mask)
                if score > best_score:
                    best_score = score
                    best_view = view_name
                    best_source = source_label

        matches.append(
            {
                "seed_id": metadata["seed_id"],
                "name": metadata["name"],
                "species": metadata.get("species", ""),
                "source": metadata.get("source", ""),
                "score": round(best_score, 4),
                "best_view": best_view,
                "matched_against": best_source,
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query_image": str(Path(config.image_path).resolve()),
        "query_parameters": query_parameters,
        "matches": matches[: config.top_k],
        "registry_dir": str(registry_dir.resolve()),
    }
