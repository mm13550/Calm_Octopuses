"""
pipelines/build_restaurant_profiles.py
=======================================
Aggregates per-modality CLIP embeddings into unified restaurant profiles.

This pipeline:
1. Loads menu, review, and image (food/interior) embeddings.
2. Performs weighted fusion of modality-specific vectors.
3. Generates a descriptive "bio" text for each restaurant.
4. Writes profiles to ``data/embeddings/restaurant_profiles.jsonl``.

Usage::

    python pipelines/build_restaurant_profiles.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "pipelines" else Path.cwd()
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

DEFAULT_MENU = EMBEDDINGS_DIR / "menu_embeddings.jsonl"
DEFAULT_REVIEW = EMBEDDINGS_DIR / "review_embeddings.jsonl"
DEFAULT_IMAGE_FOOD = EMBEDDINGS_DIR / "image_embeddings_food.jsonl"
DEFAULT_IMAGE_INTERIOR = EMBEDDINGS_DIR / "image_embeddings_interior.jsonl"
DEFAULT_OUTPUT = EMBEDDINGS_DIR / "restaurant_profiles.jsonl"

UTC = timezone.utc


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        raise ValueError("Vector must not be all zeros.")
    return [x / norm for x in vector]


def mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot mean-pool an empty vector list.")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("Vector dimensions are inconsistent.")
        for i, x in enumerate(vec):
            acc[i] += float(x)
    acc = [x / len(vectors) for x in acc]
    return normalize(acc)


def weighted_fuse(components: list[tuple[list[float], float]]) -> list[float]:
    valid = [(vec, weight) for vec, weight in components if vec and weight > 0]
    if not valid:
        raise ValueError("No valid vectors to fuse.")

    dim = len(valid[0][0])
    acc = [0.0] * dim
    total_weight = 0.0

    for vec, weight in valid:
        if len(vec) != dim:
            raise ValueError("Vector dimensions are inconsistent.")
        for i, x in enumerate(vec):
            acc[i] += float(x) * weight
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Total weight must be positive.")

    acc = [x / total_weight for x in acc]
    return normalize(acc)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return rows


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def build_bio_text(
    restaurant_name: str,
    menu_count: int,
    review_count: int,
    food_image_count: int,
    interior_image_count: int,
    top_menu_items: list[str],
    top_review_snippets: list[str],
) -> str:
    parts = [f"Restaurant: {restaurant_name}"]
    parts.append(f"Menu items indexed: {menu_count}")
    parts.append(f"Reviews indexed: {review_count}")
    parts.append(f"Food images indexed: {food_image_count}")
    parts.append(f"Interior images indexed: {interior_image_count}")

    if top_menu_items:
        parts.append("Sample menu matches: " + " | ".join(top_menu_items[:3]))
    if top_review_snippets:
        parts.append("Sample review snippets: " + " | ".join(top_review_snippets[:2]))

    return ". ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build restaurant-level profiles from menu/review/image embeddings.")
    parser.add_argument("--menu", default=str(DEFAULT_MENU))
    parser.add_argument("--review", default=str(DEFAULT_REVIEW))
    parser.add_argument("--image-food", default=str(DEFAULT_IMAGE_FOOD))
    parser.add_argument("--image-interior", default=str(DEFAULT_IMAGE_INTERIOR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    menu_rows = load_jsonl(Path(args.menu))
    review_rows = load_jsonl(Path(args.review))
    food_rows = load_jsonl(Path(args.image_food))
    interior_rows = load_jsonl(Path(args.image_interior))

    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "restaurant_id": "",
        "restaurant_name": "",
        "menu_vectors": [],
        "review_vectors": [],
        "food_vectors": [],
        "interior_vectors": [],
        "menu_items": [],
        "review_texts": [],
        "food_paths": [],
        "interior_paths": [],
    })

    def ensure_bucket(row: dict[str, Any]) -> dict[str, Any]:
        restaurant_id = safe_text(row.get("restaurant_id"))
        restaurant_name = safe_text(row.get("restaurant_name"))
        key = restaurant_id or restaurant_name
        bucket = grouped[key]
        if restaurant_id and not bucket["restaurant_id"]:
            bucket["restaurant_id"] = restaurant_id
        if restaurant_name and not bucket["restaurant_name"]:
            bucket["restaurant_name"] = restaurant_name
        return bucket

    for row in menu_rows:
        if not row.get("vector"):
            continue
        bucket = ensure_bucket(row)
        bucket["menu_vectors"].append(row["vector"])
        dish = safe_text(row.get("dish_name"))
        price = safe_text(row.get("price"))
        if dish:
            bucket["menu_items"].append(f"{dish} ({price})" if price else dish)

    for row in review_rows:
        if not row.get("vector"):
            continue
        bucket = ensure_bucket(row)
        bucket["review_vectors"].append(row["vector"])
        text = safe_text(row.get("text"))
        if text:
            bucket["review_texts"].append(text[:240])

    for row in food_rows:
        if not row.get("vector"):
            continue
        bucket = ensure_bucket(row)
        bucket["food_vectors"].append(row["vector"])
        path = safe_text(row.get("image_path"))
        if path:
            bucket["food_paths"].append(path)

    for row in interior_rows:
        if not row.get("vector"):
            continue
        bucket = ensure_bucket(row)
        bucket["interior_vectors"].append(row["vector"])
        path = safe_text(row.get("image_path"))
        if path:
            bucket["interior_paths"].append(path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for bucket in grouped.values():
            restaurant_id = bucket["restaurant_id"]
            restaurant_name = bucket["restaurant_name"] or restaurant_id
            if not restaurant_id and not restaurant_name:
                continue

            mean_menu_vector = mean_pool(bucket["menu_vectors"]) if bucket["menu_vectors"] else None
            mean_review_vector = mean_pool(bucket["review_vectors"]) if bucket["review_vectors"] else None
            mean_food_image_vector = mean_pool(bucket["food_vectors"]) if bucket["food_vectors"] else None
            mean_interior_image_vector = mean_pool(bucket["interior_vectors"]) if bucket["interior_vectors"] else None

            fusion_inputs: list[tuple[list[float], float]] = []
            if mean_menu_vector:
                fusion_inputs.append((mean_menu_vector, 0.35))
            if mean_review_vector:
                fusion_inputs.append((mean_review_vector, 0.35))
            if mean_food_image_vector:
                fusion_inputs.append((mean_food_image_vector, 0.20))
            if mean_interior_image_vector:
                fusion_inputs.append((mean_interior_image_vector, 0.10))

            fused_vector = weighted_fuse(fusion_inputs) if fusion_inputs else None
            if fused_vector is None:
                continue

            bio_text = build_bio_text(
                restaurant_name=restaurant_name,
                menu_count=len(bucket["menu_vectors"]),
                review_count=len(bucket["review_vectors"]),
                food_image_count=len(bucket["food_vectors"]),
                interior_image_count=len(bucket["interior_vectors"]),
                top_menu_items=bucket["menu_items"],
                top_review_snippets=bucket["review_texts"],
            )

            out = {
                "doc_id": f"rest_profile_{restaurant_id or restaurant_name}",
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "content_type": "restaurant_profile",
                "source": "aggregated_profile",
                "created_at": utc_now_iso(),
                "text": bio_text,
                "vector": fused_vector,
                "metadata": {
                    "menu_item_count": len(bucket["menu_vectors"]),
                    "review_count": len(bucket["review_vectors"]),
                    "food_image_count": len(bucket["food_vectors"]),
                    "interior_image_count": len(bucket["interior_vectors"]),
                    "top_menu_items": bucket["menu_items"][:5],
                    "top_review_snippets": bucket["review_texts"][:3],
                    "food_image_paths": bucket["food_paths"][:5],
                    "interior_image_paths": bucket["interior_paths"][:5],
                    "mean_menu_vector": mean_menu_vector,
                    "mean_review_vector": mean_review_vector,
                    "mean_food_image_vector": mean_food_image_vector,
                    "mean_interior_image_vector": mean_interior_image_vector,
                },
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"Loaded menu rows: {len(menu_rows)}")
    print(f"Loaded review rows: {len(review_rows)}")
    print(f"Loaded food image rows: {len(food_rows)}")
    print(f"Loaded interior image rows: {len(interior_rows)}")
    print(f"Wrote {rows_written} restaurant profiles to: {output_path}")


if __name__ == "__main__":
    main()
