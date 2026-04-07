"""Django views for the simplified seed image workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from django.conf import settings
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils.text import slugify

from seed_classifier.image_pipeline import IdentificationConfig, SeedRegistrationConfig, identify_seed, register_seed

from .forms import SeedIdentificationForm, SeedRegistrationForm


def _registry_dir() -> Path:
    return Path(settings.SEED_REGISTRY_DIR)


def _ensure_upload_dir(bucket: str) -> Path:
    upload_dir = Path(settings.MEDIA_ROOT) / "uploads" / bucket
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _save_uploaded_file(upload, bucket: str) -> tuple[Path, str]:
    upload_dir = _ensure_upload_dir(bucket)
    stem = slugify(Path(upload.name).stem) or "seed-image"
    suffix = Path(upload.name).suffix.lower() or ".png"
    filename = f"{stem}-{uuid4().hex[:10]}{suffix}"
    path = upload_dir / filename
    with path.open("wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)
    relative_url = f"{settings.MEDIA_URL}uploads/{bucket}/{filename}"
    return path, relative_url


def _load_registry_entries() -> List[Dict[str, object]]:
    index_path = _registry_dir() / "index.json"
    if not index_path.exists():
        return []
    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = []
    for entry in data.get("seeds", []):
        entry = dict(entry)
        created_at = entry.get("created_at")
        if created_at:
            try:
                parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                entry["created_display"] = parsed.strftime("%d %b %Y, %I:%M %p")
            except ValueError:
                entry["created_display"] = created_at
        entry["detail_url"] = reverse("seed-detail", kwargs={"seed_id": entry["seed_id"]})
        entries.append(entry)
    return entries


def _load_seed_metadata(seed_id: str) -> Dict[str, object]:
    seed_dir = _registry_dir() / seed_id
    metadata_path = seed_dir / "model.json"
    if not metadata_path.exists():
        raise Http404("Seed profile not found.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata


def home(request: HttpRequest) -> HttpResponse:
    entries = _load_registry_entries()
    context = {
        "seed_count": len(entries),
        "seeds": entries[:8],
        "register_form": SeedRegistrationForm(),
        "identify_form": SeedIdentificationForm(),
    }
    return render(request, "seedui/home.html", context)


def register_seed_view(request: HttpRequest) -> HttpResponse:
    result = None
    error = ""
    form = SeedRegistrationForm(request.POST or None, request.FILES or None)

    if request.method == "POST" and form.is_valid():
        try:
            saved_paths = {}
            upload_urls = {}
            for view_name in ("front", "back", "left", "right", "top", "bottom"):
                saved_paths[view_name], upload_urls[view_name] = _save_uploaded_file(
                    form.cleaned_data[view_name],
                    "registration",
                )

            config = SeedRegistrationConfig(
                name=form.cleaned_data["name"],
                species=form.cleaned_data["species"],
                source=form.cleaned_data["source"],
                notes=form.cleaned_data["notes"],
                front_image=str(saved_paths["front"]),
                back_image=str(saved_paths["back"]),
                left_image=str(saved_paths["left"]),
                right_image=str(saved_paths["right"]),
                top_image=str(saved_paths["top"]),
                bottom_image=str(saved_paths["bottom"]),
                registry_dir=str(_registry_dir()),
            )
            result = register_seed(config)
            result["upload_urls"] = upload_urls
        except Exception as exc:  # pragma: no cover - exercised through view smoke, not unit
            error = str(exc)

    context = {
        "form": form,
        "result": result,
        "error": error,
        "seeds": _load_registry_entries(),
    }
    return render(request, "seedui/register.html", context)


def identify_seed_view(request: HttpRequest) -> HttpResponse:
    form = SeedIdentificationForm(request.POST or None, request.FILES or None)
    result = None
    error = ""
    query_image_url = ""

    if request.method == "POST" and form.is_valid():
        try:
            query_path, query_image_url = _save_uploaded_file(form.cleaned_data["image"], "queries")
            config = IdentificationConfig(
                image_path=str(query_path),
                registry_dir=str(_registry_dir()),
                top_k=form.cleaned_data["top_k"],
            )
            result = identify_seed(config)
        except Exception as exc:  # pragma: no cover
            error = str(exc)

    context = {
        "form": form,
        "result": result,
        "error": error,
        "query_image_url": query_image_url,
        "seeds": _load_registry_entries(),
    }
    return render(request, "seedui/identify.html", context)


def seed_detail_view(request: HttpRequest, seed_id: str) -> HttpResponse:
    metadata = _load_seed_metadata(seed_id)

    def _as_cards(section: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
        cards = []
        for view_name, payload in section.items():
            parameters = payload.get("parameters", {})
            cards.append(
                {
                    "view_name": view_name,
                    "parameters": sorted(parameters.items()),
                }
            )
        return cards

    context = {
        "metadata": metadata,
        "view_cards": _as_cards(metadata.get("views", {})),
        "projection_cards": _as_cards(metadata.get("projections", {})),
    }
    return render(request, "seedui/seed_detail.html", context)
