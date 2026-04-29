"""
pipelines/generate_embeddings_reviews.py
=========================================
Generates 512-D CLIP text embeddings for social reviews.

This pipeline:
1. Reads review text from ``data/social_reviews.csv``.
2. Resolves restaurant names via ``data/csv/restaurant_lookup.csv``.
3. Encodes reviews using ``openai/clip-vit-base-patch32``.
4. Writes results to ``data/embeddings/review_embeddings_latest.jsonl``.

Usage::

    python pipelines/generate_embeddings_reviews.py [--backend {clip,hash}]
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "social_reviews.csv"
DEFAULT_LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "embeddings"
DEFAULT_OUTPUT_FILE = "review_embeddings_latest.jsonl"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
UTC = timezone.utc


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        raise ValueError("Vector must not be all zeros.")
    return [x / norm for x in vector]


def _find_column(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None

def load_restaurant_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        print(f"Lookup CSV not found, will fallback to rest_id as restaurant_name: {path}")
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}

    rest_id_col = _find_column(df, ["rest_id", "restaurant_id", "place_id"], required=True)
    name_col = _find_column(df, ["name", "restaurant_name"], required=True)

    lookup: dict[str, str] = {}
    for row in df.to_dict(orient="records"):
        rest_id = str(row.get(rest_id_col) or "").strip()
        name = str(row.get(name_col) or "").strip()
        if rest_id and name:
            lookup[rest_id] = name

    return lookup

_CLIP_CACHE: dict[str, Any] = {}


def _load_clip_backend(model_id: str = DEFAULT_CLIP_MODEL_ID):
    cached = _CLIP_CACHE.get(model_id)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    _CLIP_CACHE[model_id] = (tokenizer, model, device)
    return tokenizer, model, device


def embed_text_clip(text: str, *, model_id: str = DEFAULT_CLIP_MODEL_ID) -> list[float]:
    tokenizer, model, device = _load_clip_backend(model_id)

    with torch.no_grad():
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.get_text_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            features = outputs
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
            if hasattr(model, "text_projection") and model.text_projection is not None:
                features = model.text_projection(features)
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            features = outputs[0]
        else:
            raise TypeError(f"Unsupported CLIP text output type: {type(outputs)!r}")

        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Failed to extract tensor from CLIP output: {type(features)!r}")

        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return normalize(features[0].detach().cpu().tolist())


def embed_text_hash(text: str, dim: int = 512) -> list[float]:
    vec = [0.0] * dim
    tokens = str(text).lower().split()
    if not tokens:
        tokens = ["empty"]

    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign

    return normalize(vec)


def build_review_text(row: dict[str, Any]) -> str:
    restaurant_name = str(row.get("restaurant_name") or "").strip()
    rating = str(row.get("rating") or "").strip()
    text = str(row.get("text") or "").strip()

    parts = []
    if restaurant_name:
        parts.append(f"Restaurant: {restaurant_name}")
    if rating:
        parts.append(f"Rating: {rating}")
    if text:
        parts.append(f"Review: {text}")

    return " | ".join(parts).strip()


def parse_created_at(raw_value: Any) -> str:
    if raw_value is None or str(raw_value).strip() == "":
        return utc_now_iso()

    value = str(raw_value).strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    except Exception:
        return utc_now_iso()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate review embeddings from social_reviews.csv")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--lookup-csv", default=str(DEFAULT_LOOKUP_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--backend", default="clip", choices=["clip", "hash"])
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_OUTPUT_FILE
    lookup_path = Path(args.lookup_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    
    restaurant_lookup = load_restaurant_lookup(lookup_path)
    print(f"Loaded {len(restaurant_lookup)} restaurant name mappings from: {lookup_path}")

    rest_id_col = _find_column(df, ["rest_id", "restaurant_id", "place_id"], required=True)
    text_col = _find_column(df, ["text", "review_text", "review", "content"], required=True)
    rating_col = _find_column(df, ["rating", "stars", "score"])
    source_col = _find_column(df, ["source", "platform"])
    created_at_col = _find_column(df, ["created_at", "timestamp", "time", "date", "published_at"])
    uid_col = _find_column(df, ["uid", "review_id", "id"])

    rows_written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(df.to_dict(orient="records")):
            review_text = str(record.get(text_col) or "").strip()
            restaurant_id = str(record.get(rest_id_col) or "").strip()

            if not review_text or not restaurant_id:
                skipped += 1
                continue

            restaurant_name = restaurant_lookup.get(restaurant_id, restaurant_id)
            rating_value = record.get(rating_col) if rating_col else None

            payload = {
                "restaurant_name": restaurant_name,
                "rating": rating_value,
                "text": review_text,
            }
            full_text = build_review_text(payload)

            if args.backend == "clip":
                vector = embed_text_clip(full_text)
            else:
                vector = embed_text_hash(full_text)

            raw_uid = str(record.get(uid_col)).strip() if uid_col else ""
            doc_id = raw_uid if raw_uid else f"review_{restaurant_id}_{i}"

            out = {
                "doc_id": doc_id,
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "content_type": "review",
                "source": str(record.get(source_col) or "social_reviews"),
                "text": review_text,
                "rating": rating_value,
                "created_at": parse_created_at(record.get(created_at_col) if created_at_col else None),
                "vector": vector,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"Loaded {len(df)} raw review rows from: {input_path}")
    print(f"Skipped {skipped} incomplete rows")
    print(f"Wrote {rows_written} review embeddings to: {output_path}")
    print(f"Backend: {args.backend}")


if __name__ == "__main__":
    main()
