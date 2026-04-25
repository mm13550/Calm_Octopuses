"""
pipelines/generate_embeddings_images.py
========================================
Generates 512-D CLIP image embeddings for all food and interior images in
``data/social_images.csv``.

Each row in the images CSV that has a valid image path is passed through
``openai/clip-vit-base-patch32`` and the resulting L2-normalised vector is
appended to the appropriate JSONL file
(``data/embeddings/image_embeddings_food.jsonl`` or ``image_embeddings_interior.jsonl``).

Usage::

    python pipelines/generate_embeddings_images.py [--batch-size 64]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "pipelines" else Path.cwd()
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "social_images_cleaned.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "embeddings"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
UTC = timezone.utc

OUTPUT_FILES = {
    "food": "image_embeddings_food_latest.jsonl",
    "interior": "image_embeddings_interior_latest.jsonl",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        raise ValueError("Vector must not be all zeros.")
    return [x / norm for x in vector]


_CLIP_CACHE: dict[str, Any] = {}


def load_clip_backend(model_id: str = DEFAULT_MODEL_ID):
    cached = _CLIP_CACHE.get(model_id)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    _CLIP_CACHE[model_id] = (processor, model, device)
    return processor, model, device


def resolve_image_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(str(raw_path).strip())
    if p.is_absolute():
        return p
    if p.exists():
        return p
    candidate = base_dir / p
    if candidate.exists():
        return candidate
    return candidate


@torch.no_grad()
def embed_image_clip(image: Image.Image, *, model_id: str = DEFAULT_MODEL_ID) -> list[float]:
    processor, model, device = load_clip_backend(model_id=model_id)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pixel_values = inputs["pixel_values"]

    vision_outputs = model.vision_model(pixel_values=pixel_values)

    if not hasattr(vision_outputs, "pooler_output") or vision_outputs.pooler_output is None:
        raise TypeError("vision_model output has no pooler_output")

    features = vision_outputs.pooler_output

    if hasattr(model, "visual_projection") and model.visual_projection is not None:
        features = model.visual_projection(features)

    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Failed to extract tensor from CLIP image output: {type(features)!r}")

    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalize(features[0].detach().cpu().tolist())
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image embeddings from social_images_cleaned.csv")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", default="food", choices=["food", "interior"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick testing")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILES[args.mode]

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    required = {
        "image_uid",
        "rest_id",
        "restaurant_name",
        "source",
        "image_path",
        "image_category",
        "keep_for_food_embedding",
        "keep_for_ambiance_embedding",
        "quality_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"social_images_cleaned.csv missing required columns: {sorted(missing)}")

    keep_col = "keep_for_food_embedding" if args.mode == "food" else "keep_for_ambiance_embedding"
    category_value = "food" if args.mode == "food" else "interior"
    content_type_value = f"image_{args.mode}"

    filtered = df[df[keep_col].astype(bool)].copy()

    filtered = filtered[filtered["image_category"].astype(str).str.strip().str.lower() == category_value].copy()

    if args.limit is not None:
        filtered = filtered.head(args.limit).copy()

    if filtered.empty:
        raise ValueError(f"No rows matched mode={args.mode}. Check keep flags and image_category.")

    rows_written = 0
    missing_files = 0
    failed_images = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in filtered.to_dict(orient="records"):
            raw_path = str(row.get("image_path") or "").strip()
            full_path = resolve_image_path(raw_path, BASE_DIR)

            if not full_path.exists():
                missing_files += 1
                continue

            try:
                with Image.open(full_path) as img:
                    image = img.convert("RGB")
                    vector = embed_image_clip(image, model_id=args.model_id)
            except Exception as exc:
                failed_images += 1
                if failed_images <= 5:
                    print(f"[embed_error] {raw_path} -> {type(exc).__name__}: {exc}")
                continue

            out = {
                "doc_id": str(row.get("image_uid") or "").strip(),
                "restaurant_id": str(row.get("rest_id") or "").strip(),
                "restaurant_name": str(row.get("restaurant_name") or "").strip(),
                "content_type": content_type_value,
                "source": str(row.get("source") or "").strip(),
                "image_path": raw_path,
                "image_category": str(row.get("image_category") or "").strip(),
                "quality_score": float(row.get("quality_score") or 0.0),
                "created_at": utc_now_iso(),
                "text": f"Image for restaurant {row.get('restaurant_name')} categorized as {row.get('image_category')}",
                "vector": vector,
                "metadata": {
                    "mode": args.mode,
                    "keep_column": keep_col,
                    "notes": str(row.get("notes") or "").strip(),
                },
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"Loaded {len(df)} cleaned image rows from: {input_path}")
    print(f"Mode: {args.mode}")
    print(f"Eligible rows after filtering: {len(filtered)}")
    print(f"Wrote {rows_written} image embeddings to: {output_path}")
    print(f"Missing image files during embedding: {missing_files}")
    print(f"Failed image opens/embeddings: {failed_images}")


if __name__ == "__main__":
    main()
