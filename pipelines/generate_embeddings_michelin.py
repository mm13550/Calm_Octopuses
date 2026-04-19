from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import CLIPModel, CLIPTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "extracted_menus" / "final_parsed_menus.json"
DEFAULT_OUTPUT_DIR = DATA_DIR / "embeddings"

# canonical names
MENU_OUTPUT_FILE = "menu_embeddings_latest.jsonl"
SUMMARY_OUTPUT_FILE = "restaurant_summary_latest.jsonl"

DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
UTC = timezone.utc

BEVERAGE_TERMS = {
    "wine", "pairing", "pairings", "cocktail", "cocktails", "beer", "beverage",
    "drink", "drinks", "sake", "tea pairing", "juice", "mocktail", "spirits"
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        raise ValueError("Vector must not be all zeros.")
    return [x / norm for x in vector]


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


def load_menu_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("records", "items", "menus", "data", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    raise ValueError(f"Unsupported menu JSON structure in: {path}")


def pick(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def normalize_record(record: dict[str, Any]) -> dict[str, str]:
    restaurant_id = as_text(pick(record, "rest_id", "restaurant_id", "place_id", "id"))
    restaurant_name = as_text(pick(record, "restaurant_name", "name", "restaurant"))
    dish_name = as_text(pick(record, "dish_name", "dish", "menu_item", "item_name", "title"))
    ingredients = as_text(pick(record, "ingredients", "description", "details", "content"))
    price = as_text(pick(record, "price", "price_display", "cost"))
    source = as_text(pick(record, "source", "source_type")) or "official_menu_parsed"

    return {
        "restaurant_id": restaurant_id,
        "restaurant_name": restaurant_name,
        "dish_name": dish_name,
        "ingredients": ingredients,
        "price": price,
        "source": source,
    }


def is_beverage_like(dish_name: str, ingredients: str) -> bool:
    blob = f"{dish_name} {ingredients}".lower()
    return any(term in blob for term in BEVERAGE_TERMS)


def infer_record_kind(dish_name: str, ingredients: str) -> str:
    blob = f"{dish_name} {ingredients}".lower()

    if "omakase" in blob:
        return "omakase"
    if any(term in blob for term in ["tasting menu", "chef tasting", "prix fixe", "course menu"]):
        return "tasting_menu"
    if "dessert" in blob:
        return "dessert"
    if "yakitori" in blob:
        return "yakitori"
    if "sushi" in blob:
        return "sushi"
    return "menu_item"


def infer_style_tags(dish_name: str, ingredients: str, record_kind: str) -> list[str]:
    blob = f"{dish_name} {ingredients}".lower()
    tags: list[str] = []

    if record_kind not in tags:
        tags.append(record_kind)

    for term in ["dessert", "vegetarian", "seafood", "yakitori", "sushi", "omakase"]:
        if term in blob and term not in tags:
            tags.append(term)

    return tags


def build_menu_text(restaurant_name: str, dish_name: str, ingredients: str, price: str, record_kind: str, style_tags: list[str]) -> str:
    parts = [f"Restaurant: {restaurant_name}"]
    if dish_name:
        parts.append(f"Dish: {dish_name}")
    if ingredients:
        parts.append(f"Ingredients: {ingredients}")
    if price:
        parts.append(f"Price: {price}")
    parts.append(f"Record kind: {record_kind}")
    if style_tags:
        parts.append(f"Style tags: {', '.join(style_tags)}")
    return ". ".join(parts)


def mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot pool empty vector list.")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for i, x in enumerate(vec):
            acc[i] += float(x)
    acc = [x / len(vectors) for x in acc]
    return normalize(acc)


def main():
    parser = argparse.ArgumentParser(description="Generate menu embeddings from parsed menu JSON")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--backend", default="clip", choices=["clip", "hash"])
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    menu_output_path = output_dir / MENU_OUTPUT_FILE
    summary_output_path = output_dir / SUMMARY_OUTPUT_FILE

    raw_records = load_menu_records(input_path)
    print(f"Loaded {len(raw_records)} raw menu records from: {input_path}")

    dedup_seen: set[str] = set()
    dish_rows: list[dict[str, Any]] = []
    vectors_by_restaurant: dict[tuple[str, str], list[list[float]]] = defaultdict(list)

    skipped_incomplete = 0
    skipped_beverage = 0

    for record in raw_records:
        row = normalize_record(record)

        restaurant_id = row["restaurant_id"]
        restaurant_name = row["restaurant_name"]
        dish_name = row["dish_name"]
        ingredients = row["ingredients"]
        price = row["price"]
        source = row["source"]

        if not restaurant_id or not restaurant_name or not dish_name:
            skipped_incomplete += 1
            continue

        if is_beverage_like(dish_name, ingredients):
            skipped_beverage += 1
            continue

        dedup_key = " | ".join([restaurant_id, restaurant_name, dish_name, ingredients, price]).strip().lower()
        if dedup_key in dedup_seen:
            continue
        dedup_seen.add(dedup_key)

        record_kind = infer_record_kind(dish_name, ingredients)
        style_tags = infer_style_tags(dish_name, ingredients, record_kind)
        text = build_menu_text(restaurant_name, dish_name, ingredients, price, record_kind, style_tags)

        if args.backend == "clip":
            vector = embed_text_clip(text)
        else:
            vector = embed_text_hash(text)

        doc_hash = hashlib.md5(dedup_key.encode("utf-8")).hexdigest()[:12]
        doc_id = f"dish_{doc_hash}"

        out = {
            "doc_id": doc_id,
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant_name,
            "content_type": "menu",
            "source": source,
            "dish_name": dish_name,
            "price": price,
            "created_at": utc_now_iso(),
            "text": text,
            "vector": vector,
            "metadata": {
                "record_kind": record_kind,
                "style_tags": style_tags,
            },
        }

        dish_rows.append(out)
        vectors_by_restaurant[(restaurant_id, restaurant_name)].append(vector)

    summary_rows: list[dict[str, Any]] = []
    for (restaurant_id, restaurant_name), vectors in vectors_by_restaurant.items():
        pooled = mean_pool(vectors)
        summary_rows.append(
            {
                "doc_id": f"rest_{hashlib.md5(restaurant_id.encode('utf-8')).hexdigest()[:12]}",
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "content_type": "restaurant_summary",
                "source": "menu_summary",
                "dish_name": None,
                "price": None,
                "created_at": utc_now_iso(),
                "text": f"Restaurant summary for {restaurant_name}. Menu item count: {len(vectors)}.",
                "vector": pooled,
                "metadata": {
                    "menu_item_count": len(vectors),
                },
            }
        )

    for path, rows in [
        (menu_output_path, dish_rows),
        (summary_output_path, summary_rows),
    ]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Skipped {skipped_incomplete} incomplete rows")
    print(f"Skipped {skipped_beverage} beverage-like rows")
    print(f"Wrote {len(dish_rows)} dish-level embeddings to: {menu_output_path}")
    print(f"Wrote {len(summary_rows)} restaurant summary embeddings to: {summary_output_path}")
    print(f"Backend: {args.backend}")


if __name__ == "__main__":
    main()