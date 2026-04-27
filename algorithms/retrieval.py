from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from core.data_loader import (
    _clean_text,
    _resolve_path,
    _join_snippets,
    load_restaurant_embeddings,
    load_finegrained_embeddings,
    DEFAULT_CLIP_MODEL_ID,
    MENU_EMBEDDINGS_JSONL,
    REVIEW_EMBEDDINGS_JSONL,
    FOOD_EMBEDDINGS_JSONL,
    INTERIOR_EMBEDDINGS_JSONL
)


def _format_menu_item_display(dish_name: Any, ingredients: Any = "", price: Any = "") -> str:
    dish = _clean_text(dish_name)
    ingredients_text = _clean_text(ingredients)
    price_text = _clean_text(price)
    parts = [part for part in [dish, ingredients_text, f"${price_text}" if price_text else ""] if part]
    return " — ".join(parts)


@st.cache_resource(show_spinner=False)
def load_menu_embedding_records() -> Dict[str, List[Dict[str, Any]]]:
    if not MENU_EMBEDDINGS_JSONL.exists():
        return {}

    records_by_restaurant: Dict[str, List[Dict[str, Any]]] = {}
    with MENU_EMBEDDINGS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            rest_id = _clean_text(data.get("restaurant_id", data.get("rest_id")))
            vector = data.get("vector")
            if not rest_id or not vector:
                continue

            dish_name = _clean_text(data.get("dish_name"))
            price = _clean_text(data.get("price"))
            display_text = _format_menu_item_display(dish_name, price=price) or _clean_text(data.get("text"))
            search_text = " ".join([dish_name, _clean_text(data.get("text"))]).strip()

            records_by_restaurant.setdefault(rest_id, []).append(
                {
                    "display_text": display_text,
                    "search_text": search_text,
                    "vector": np.array(vector, dtype=np.float32),
                }
            )

    return records_by_restaurant

@st.cache_resource(show_spinner=False)
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(DEFAULT_CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(DEFAULT_CLIP_MODEL_ID).to(device)
    model.eval()
    return processor, model, device

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

@st.cache_data(show_spinner=False)
def embed_text(text: str) -> tuple:
    text = _clean_text(text)
    if not text:
        return tuple()

    processor, model, device = load_clip_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        
        if isinstance(outputs, torch.Tensor):
            features = outputs
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
            if hasattr(model, "text_projection") and model.text_projection is not None:
                features = model.text_projection(features)
        elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            features = outputs.text_embeds
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            features = outputs[0]
        else:
            raise TypeError(f"Unexpected CLIP text output type: {type(outputs)}")
            
        outputs = features / features.norm(p=2, dim=-1, keepdim=True)

    return tuple(float(value) for value in outputs[0].detach().cpu().tolist())

@st.cache_data(show_spinner=False)
def embed_image(image_path: str) -> tuple:
    path = _resolve_path(image_path)
    if not path.exists():
        return tuple()

    processor, model, device = load_clip_model()
    with Image.open(path) as image_file:
        image = image_file.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        if not hasattr(vision_outputs, "pooler_output") or vision_outputs.pooler_output is None:
            raise TypeError("CLIP vision_model output missing pooler_output.")

        features = vision_outputs.pooler_output
        if hasattr(model, "visual_projection") and model.visual_projection is not None:
            features = model.visual_projection(features)

        if not isinstance(features, torch.Tensor):
            raise TypeError("Unexpected CLIP image output type.")
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return tuple(float(value) for value in features[0].detach().cpu().tolist())

def _keyword_overlap(text: str, query_terms: set) -> float:
    if not text or not query_terms:
        return 0.0
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not tokens:
        return 0.0
    matches = len(tokens & query_terms)
    return float(matches) / float(max(len(query_terms), 1))

def score_text_results(catalog: pd.DataFrame, query: str, scope: str) -> pd.DataFrame:
    if catalog.empty or not _clean_text(query):
        return pd.DataFrame()

    query_vector = np.array(embed_text(query), dtype=np.float32)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

    if scope == "Menu items":
        menu_records_map = load_menu_embedding_records()
        embeddings_map = None
        generic_map = None
    elif scope == "Reviews":
        menu_records_map = None
        embeddings_map = load_finegrained_embeddings(str(REVIEW_EMBEDDINGS_JSONL))
        generic_map = None
    else:
        menu_records_map = None
        embeddings_map = None
        generic_map = load_restaurant_embeddings()

    rows: List[Dict[str, Any]] = []
    for row in catalog.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        
        if scope == "Menu items":
            candidate_text = _join_snippets(row.get("menu_items", []))
        elif scope == "Reviews":
            candidate_text = _join_snippets(row.get("review_snippets", []))
        elif scope == "Bios":
            candidate_text = _clean_text(row.get("bio_text"))
        else:
            candidate_text = _clean_text(row.get("search_text"))

        if not candidate_text:
            continue

        semantic_score = 0.0
        lexical_score = 0.0
        matched_menu_items: List[str] = []

        if scope == "Menu items":
            menu_records = menu_records_map.get(rest_id, []) if menu_records_map is not None else []
            if not menu_records:
                continue

            scored_records = []
            for record in menu_records:
                record_semantic_score = _cosine_similarity(query_vector, record["vector"])
                record_lexical_score = _keyword_overlap(record["search_text"], query_terms)
                record_score = (0.85 * record_semantic_score) + (0.15 * record_lexical_score)
                scored_records.append(
                    (
                        record_score,
                        record_semantic_score,
                        record_lexical_score,
                        record["display_text"],
                    )
                )

            scored_records.sort(key=lambda item: item[0], reverse=True)
            if not scored_records or scored_records[0][0] <= 0.0:
                continue

            final_score, semantic_score, lexical_score, _ = scored_records[0]
            seen_matches = set()
            for _, _, _, display_text in scored_records:
                cleaned_display_text = _clean_text(display_text)
                if not cleaned_display_text or cleaned_display_text.lower() in seen_matches:
                    continue
                matched_menu_items.append(cleaned_display_text)
                seen_matches.add(cleaned_display_text.lower())
                if len(matched_menu_items) == 3:
                    break
        elif embeddings_map is not None:
            vectors = embeddings_map.get(rest_id, [])
            if vectors:
                scores = [_cosine_similarity(query_vector, v) for v in vectors]
                semantic_score = max(scores)
        elif generic_map is not None:
            candidate_vector = generic_map.get(rest_id)
            if candidate_vector is not None:
                semantic_score = _cosine_similarity(query_vector, candidate_vector)

        if scope != "Menu items" and semantic_score == 0.0:
            continue

        if scope != "Menu items":
            lexical_score = _keyword_overlap(candidate_text, query_terms)
            final_score = (0.85 * semantic_score) + (0.15 * lexical_score)

        rows.append(
            {
                **row,
                "score": final_score,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "matched_menu_items": matched_menu_items,
            }
        )

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(by="score", ascending=False)

def score_image_results(catalog: pd.DataFrame, image_path: str) -> pd.DataFrame:
    if catalog.empty or not _clean_text(image_path):
        return pd.DataFrame()

    query_vector = np.array(embed_image(image_path), dtype=np.float32)
    if query_vector.size == 0:
        return pd.DataFrame()

    food_map = load_finegrained_embeddings(str(FOOD_EMBEDDINGS_JSONL))
    interior_map = load_finegrained_embeddings(str(INTERIOR_EMBEDDINGS_JSONL))

    rows: List[Dict[str, Any]] = []
    for row in catalog.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        
        food_vectors = food_map.get(rest_id, [])
        interior_vectors = interior_map.get(rest_id, [])
        all_vectors = food_vectors + interior_vectors
        
        if not all_vectors:
            continue

        scores = [_cosine_similarity(query_vector, v) for v in all_vectors]
        best_score = max(scores)
        
        rows.append({**row, "score": best_score})

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(by="score", ascending=False)

def score_exact_dish_search(catalog: pd.DataFrame, menus_df: pd.DataFrame, query: str) -> pd.DataFrame:
    if catalog.empty or menus_df.empty or not _clean_text(query):
        return pd.DataFrame()

    query_lower = query.lower().strip()
    
    mask = (
        menus_df['dish_name'].str.contains(query_lower, case=False, regex=False, na=False) |
        menus_df['ingredients'].str.contains(query_lower, case=False, regex=False, na=False)
    )
    
    matched_menus = menus_df[mask]
    if matched_menus.empty:
        return pd.DataFrame()

    matched_items_by_restaurant: Dict[str, List[str]] = {}
    match_counts = matched_menus.groupby("rest_id").size().to_dict()
    for rest_id, group in matched_menus.groupby("rest_id"):
        matched_items: List[str] = []
        seen_items = set()
        for row in group.to_dict(orient="records"):
            display_text = _format_menu_item_display(
                row.get("dish_name"),
                row.get("ingredients"),
                row.get("price"),
            )
            cleaned_display_text = _clean_text(display_text)
            if not cleaned_display_text or cleaned_display_text.lower() in seen_items:
                continue
            matched_items.append(cleaned_display_text)
            seen_items.add(cleaned_display_text.lower())
            if len(matched_items) == 3:
                break
        matched_items_by_restaurant[rest_id] = matched_items

    matched_rest_ids = set(matched_menus['rest_id'].unique())
    
    results = catalog[catalog['rest_id'].isin(matched_rest_ids)].copy()
    if results.empty:
        return pd.DataFrame()

    results["matched_menu_items"] = results["rest_id"].map(lambda rest_id: matched_items_by_restaurant.get(rest_id, []))
    results["match_count"] = results["rest_id"].map(lambda rest_id: int(match_counts.get(rest_id, 0)))
    results['score'] = 1.0
    return results.sort_values(by=["match_count", "restaurant_name"], ascending=[False, True])
