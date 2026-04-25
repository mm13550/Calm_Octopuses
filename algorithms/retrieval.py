from __future__ import annotations

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
        embeddings_map = load_finegrained_embeddings(str(MENU_EMBEDDINGS_JSONL))
        generic_map = None
    elif scope == "Reviews":
        embeddings_map = load_finegrained_embeddings(str(REVIEW_EMBEDDINGS_JSONL))
        generic_map = None
    else:
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
        
        if embeddings_map is not None:
            vectors = embeddings_map.get(rest_id, [])
            if vectors:
                scores = [_cosine_similarity(query_vector, v) for v in vectors]
                semantic_score = max(scores)
        elif generic_map is not None:
            candidate_vector = generic_map.get(rest_id)
            if candidate_vector is not None:
                semantic_score = _cosine_similarity(query_vector, candidate_vector)

        if semantic_score == 0.0:
            continue

        lexical_score = _keyword_overlap(candidate_text, query_terms)
        final_score = (0.85 * semantic_score) + (0.15 * lexical_score)

        rows.append(
            {
                **row,
                "score": final_score,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
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
        
    matched_rest_ids = set(matched_menus['rest_id'].unique())
    
    results = catalog[catalog['rest_id'].isin(matched_rest_ids)].copy()
    if results.empty:
        return pd.DataFrame()
        
    results['score'] = 1.0
    return results
