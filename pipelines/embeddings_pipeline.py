"""
pipelines/embeddings_pipeline.py
===============================
Unified embedding generation pipeline for the Calm Octopuses project.

This script consolidates legacy embedding scripts into a single tool for:
1. Menus: 512-D CLIP text embeddings for Michelin dishes.
2. Reviews: 512-D CLIP text embeddings for social reviews.
3. Images: 512-D CLIP image embeddings for food and interior photos.

Usage::

    python pipelines/embeddings_pipeline.py menus
    python pipelines/embeddings_pipeline.py reviews --backend {clip,hash}
    python pipelines/embeddings_pipeline.py images --mode {food,interior}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "pipelines" else Path.cwd()
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
UTC = timezone.utc

DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# --- Common Helpers ---

def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        raise ValueError("Vector must not be all zeros.")
    return [x / norm for x in vector]


_CLIP_CACHE: dict[str, Any] = {}

def get_clip_backend(model_id: str):
    if model_id in _CLIP_CACHE:
        return _CLIP_CACHE[model_id]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    _CLIP_CACHE[model_id] = (model, tokenizer, processor, device)
    return model, tokenizer, processor, device


@torch.no_grad()
def embed_text_clip(text: str, model_id: str) -> list[float]:
    model, tokenizer, _, device = get_clip_backend(model_id)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    features = model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalize(features[0].detach().cpu().tolist())


def embed_text_hash(text: str, dim: int = 512) -> list[float]:
    vec = [0.0] * dim
    tokens = str(text).lower().split()
    if not tokens: tokens = ["empty"]
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign
    return normalize(vec)


@torch.no_grad()
def embed_image_clip(image: Image.Image, model_id: str) -> list[float]:
    model, _, processor, device = get_clip_backend(model_id)
    inputs = processor(images=image, return_tensors="pt").to(device)
    vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    features = vision_outputs.pooler_output
    if hasattr(model, "visual_projection") and model.visual_projection is not None:
        features = model.visual_projection(features)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalize(features[0].detach().cpu().tolist())


# --- Menu Embedding Logic ---

def run_menus(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        raw_records = json.load(f)

    dish_rows, summary_rows = [], []
    vectors_by_restaurant = defaultdict(list)
    dedup_seen = set()

    for record in raw_records:
        rid = str(record.get("rest_id") or record.get("restaurant_id") or "").strip()
        rname = str(record.get("restaurant_name") or record.get("name") or "").strip()
        dname = str(record.get("dish_name") or record.get("dish") or "").strip()
        if not rid or not rname or not dname: continue

        key = f"{rid}|{rname}|{dname}|{record.get('ingredients')}".lower()
        if key in dedup_seen: continue
        dedup_seen.add(key)

        text = f"Restaurant: {rname}. Dish: {dname}. Ingredients: {record.get('ingredients', '')}"
        vector = embed_text_clip(text, args.model_id) if args.backend == "clip" else embed_text_hash(text)
        
        out = {
            "doc_id": f"dish_{hashlib.md5(key.encode()).hexdigest()[:12]}",
            "restaurant_id": rid, "restaurant_name": rname, "content_type": "menu",
            "dish_name": dname, "created_at": utc_now_iso(), "text": text, "vector": vector
        }
        dish_rows.append(out)
        vectors_by_restaurant[(rid, rname)].append(vector)

    # Summaries
    for (rid, rname), vecs in vectors_by_restaurant.items():
        dim = len(vecs[0])
        acc = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        summary_rows.append({
            "doc_id": f"rest_sum_{hashlib.md5(rid.encode()).hexdigest()[:12]}",
            "restaurant_id": rid, "restaurant_name": rname, "content_type": "restaurant_summary",
            "text": f"Summary for {rname}. Items: {len(vecs)}", "vector": normalize(acc),
            "metadata": {"menu_item_count": len(vecs)}
        })

    with (output_dir / "menu_embeddings_latest.jsonl").open("w", encoding="utf-8") as f:
        for r in dish_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (output_dir / "restaurant_summary_latest.jsonl").open("w", encoding="utf-8") as f:
        for r in summary_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Menus complete. Wrote {len(dish_rows)} dishes and {len(summary_rows)} summaries.")


# --- Review Embedding Logic ---

def run_reviews(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    lookup_df = pd.read_csv(Path(args.lookup_csv))
    lookup = dict(zip(lookup_df["rest_id"], lookup_df["name"]))

    output_path = Path(args.output_dir) / "review_embeddings_latest.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            rid = str(row.get("rest_id") or row.get("restaurant_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not rid or not text: continue
            
            rname = lookup.get(rid, rid)
            full_text = f"Restaurant: {rname}. Review: {text}"
            vector = embed_text_clip(full_text, args.model_id) if args.backend == "clip" else embed_text_hash(full_text)
            
            out = {
                "doc_id": str(row.get("uid") or f"rev_{i}"),
                "restaurant_id": rid, "restaurant_name": rname, "content_type": "review",
                "text": text, "rating": row.get("rating"), "created_at": utc_now_iso(), "vector": vector
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Reviews complete. Wrote embeddings to {output_path.name}")


# --- Image Embedding Logic ---

def run_images(args: argparse.Namespace) -> None:
    df = pd.read_csv(Path(args.input))
    keep_col = "keep_for_food_embedding" if args.mode == "food" else "keep_for_ambiance_embedding"
    filtered = df[df[keep_col].astype(bool)].copy()
    
    output_path = Path(args.output_dir) / f"image_embeddings_{args.mode}_latest.jsonl"
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in filtered.to_dict(orient="records"):
            img_path = Path(str(row.get("image_path") or "").strip())
            if not img_path.is_absolute(): img_path = BASE_DIR / img_path
            if not img_path.exists(): continue
            
            try:
                with Image.open(img_path) as img:
                    vector = embed_image_clip(img.convert("RGB"), args.model_id)
                out = {
                    "doc_id": str(row.get("image_uid") or ""),
                    "restaurant_id": str(row.get("rest_id") or ""),
                    "restaurant_name": str(row.get("restaurant_name") or ""),
                    "content_type": f"image_{args.mode}", "image_path": str(row.get("image_path")),
                    "created_at": utc_now_iso(), "vector": vector
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
            except Exception: continue
    print(f"Images ({args.mode}) complete. Wrote {written} embeddings.")


# --- Main CLI ---

def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Embedding Generation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Menus
    m_parser = subparsers.add_parser("menus", help="Embed Michelin menus.")
    m_parser.add_argument("--input", default=str(DATA_DIR / "extracted_menus" / "final_parsed_menus.json"))
    m_parser.add_argument("--output-dir", default=str(EMBEDDINGS_DIR))
    m_parser.add_argument("--backend", default="clip", choices=["clip", "hash"])
    m_parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL_ID)

    # Reviews
    r_parser = subparsers.add_parser("reviews", help="Embed social reviews.")
    r_parser.add_argument("--input", default=str(DATA_DIR / "social_reviews.csv"))
    r_parser.add_argument("--lookup-csv", default=str(DATA_DIR / "csv" / "restaurant_lookup.csv"))
    r_parser.add_argument("--output-dir", default=str(EMBEDDINGS_DIR))
    r_parser.add_argument("--backend", default="clip", choices=["clip", "hash"])
    r_parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL_ID)

    # Images
    i_parser = subparsers.add_parser("images", help="Embed restaurant images.")
    i_parser.add_argument("--input", default=str(DATA_DIR / "social_images_cleaned.csv"))
    i_parser.add_argument("--output-dir", default=str(EMBEDDINGS_DIR))
    i_parser.add_argument("--mode", choices=["food", "interior"], required=True)
    i_parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL_ID)

    args = parser.parse_args()

    if args.command == "menus": run_menus(args)
    elif args.command == "reviews": run_reviews(args)
    elif args.command == "images": run_images(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
