"""Streamlit frontend for the Calm Octopuses raw data MVP.

This entrypoint stays lightweight: it reads the raw data files already present
in the repo, lets users search by text or image, and renders restaurant-level
cards keyed by rest_id.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


import sys
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from algorithms.mdn_regression import MDNScorer
from algorithms.retrieval import score_text_results, score_image_results, score_exact_dish_search

DATA_DIR = PROJECT_ROOT / "data"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
REVIEWS_CSV = DATA_DIR / "social_reviews.csv"
IMAGES_CSV = DATA_DIR / "social_images.csv"
MENUS_JSON = DATA_DIR / "extracted_menus" / "final_parsed_menus.json"
BIOS_JSON = DATA_DIR / "extracted_bios" / "restaurant_bios_joinable.json"
EMBEDDINGS_JSONL = DATA_DIR / "embeddings" / "restaurant_profiles.jsonl"
MENU_EMBEDDINGS_JSONL = DATA_DIR / "embeddings" / "menu_embeddings.jsonl"
REVIEW_EMBEDDINGS_JSONL = DATA_DIR / "embeddings" / "review_embeddings.jsonl"
FOOD_EMBEDDINGS_JSONL = DATA_DIR / "embeddings" / "image_embeddings_food.jsonl"
INTERIOR_EMBEDDINGS_JSONL = DATA_DIR / "embeddings" / "image_embeddings_interior.jsonl"
MDN_CHECKPOINT = DATA_DIR / "yelp_sandbox" / "mdn_models" / "lightning_logs" / "version_0" / "checkpoints" / "epoch=9-step=270.ckpt"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def _truncate(text: str, limit: int = 220) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        for key in ("records", "items", "data", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

    return []


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


@st.cache_data(show_spinner=False)
def load_lookup_df() -> pd.DataFrame:
    return _load_csv(LOOKUP_CSV)


@st.cache_data(show_spinner=False)
def load_reviews_df() -> pd.DataFrame:
    return _load_csv(REVIEWS_CSV)


@st.cache_data(show_spinner=False)
def load_images_df() -> pd.DataFrame:
    return _load_csv(IMAGES_CSV)


@st.cache_data(show_spinner=False)
def load_menus_df() -> pd.DataFrame:
    rows = _load_json_records(MENUS_JSON)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_bios_df() -> pd.DataFrame:
    rows = _load_json_records(BIOS_JSON)
    if not rows:
        return pd.DataFrame()

    flattened: List[Dict[str, Any]] = []
    for row in rows:
        bio = row.get("bio", {})
        if isinstance(bio, dict):
            bio_text = " ".join(
                _clean_text(bio.get(field)) for field in ("description", "culinary_style", "history")
            ).strip()
        else:
            bio_text = _clean_text(bio)

        flattened.append(
            {
                "rest_id": _clean_text(row.get("rest_id")),
                "restaurant_name": _clean_text(row.get("name")),
                "bio_text": bio_text,
                "match_source": _clean_text(row.get("match_source")),
            }
        )

    return pd.DataFrame(flattened)


@st.cache_resource(show_spinner=False)
def load_restaurant_embeddings() -> Dict[str, np.ndarray]:
    """
    Loads pre-computed 512-D CLIP text embeddings from the JSONL profiles.
    Returns a dictionary mapping 'restaurant_id' -> np.ndarray.
    """
    if not EMBEDDINGS_JSONL.exists():
        return {}

    embeddings: Dict[str, np.ndarray] = {}
    with EMBEDDINGS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rest_id = _clean_text(data.get("restaurant_id", data.get("rest_id")))
                vector = data.get("vector")
                if rest_id and vector:
                    embeddings[rest_id] = np.array(vector, dtype=np.float32)
            except json.JSONDecodeError:
                continue

    return embeddings


@st.cache_resource(show_spinner=False)
def load_finegrained_embeddings(file_path_str: str) -> Dict[str, List[np.ndarray]]:
    """
    Loads fine-grained vectors from a specific JSONL file.
    Returns a dictionary mapping 'restaurant_id' -> List[np.ndarray].
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        return {}
        
    embeddings = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                rest_id = _clean_text(data.get("restaurant_id", data.get("rest_id")))
                vector = data.get("vector")
                if rest_id and vector:
                    if rest_id not in embeddings:
                        embeddings[rest_id] = []
                    embeddings[rest_id].append(np.array(vector, dtype=np.float32))
            except json.JSONDecodeError:
                continue
    return embeddings


def _resolve_path(path_value: str) -> Path:
    path = Path(_clean_text(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _first_existing_path(paths: List[str]) -> Optional[str]:
    for raw_path in paths:
        candidate = _resolve_path(raw_path)
        if candidate.exists():
            return str(candidate)
    return str(_resolve_path(paths[0])) if paths else None


def _join_snippets(values: List[str], limit: int = 6) -> str:
    snippets = [_clean_text(value) for value in values if _clean_text(value)]
    return " ".join(snippets[:limit]).strip()


def _extract_menu_items(menu_rows: List[Dict[str, Any]], limit: int = 6) -> List[str]:
    items: List[str] = []
    for row in menu_rows[:limit]:
        dish = _clean_text(row.get("dish_name"))
        ingredients = _clean_text(row.get("ingredients"))
        price = _clean_text(row.get("price"))
        parts = [part for part in [dish, ingredients, f"${price}" if price else ""] if part]
        if parts:
            items.append(" — ".join(parts))
    return items


def _extract_review_snippets(review_rows: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    snippets: List[str] = []
    for row in review_rows[:limit]:
        text = _clean_text(row.get("text"))
        rating = _clean_text(row.get("rating"))
        if text:
            snippets.append(f"{text} {'(⭐ ' + rating + ')' if rating else ''}".strip())
    return snippets


@st.cache_data(show_spinner=False)
def build_restaurant_catalog() -> pd.DataFrame:
    # Cache buster: 1
    lookup_df = load_lookup_df()
    reviews_df = load_reviews_df()
    images_df = load_images_df()
    menus_df = load_menus_df()
    bios_df = load_bios_df()

    if lookup_df.empty:
        return pd.DataFrame()

    bio_map: Dict[str, str] = {}
    if not bios_df.empty:
        for row in bios_df.to_dict(orient="records"):
            rest_id = _clean_text(row.get("rest_id"))
            if not rest_id:
                continue
            bio_map[rest_id] = _clean_text(row.get("bio_text"))

    records: List[Dict[str, Any]] = []
    menu_groups = menus_df.groupby("rest_id") if not menus_df.empty and "rest_id" in menus_df.columns else None
    review_groups = reviews_df.groupby("rest_id") if not reviews_df.empty and "rest_id" in reviews_df.columns else None
    image_groups = images_df.groupby("rest_id") if not images_df.empty and "rest_id" in images_df.columns else None

    lookup_records = lookup_df.to_dict(orient="records")
    lookup_records.sort(key=lambda row: _clean_text(row.get("name")).lower())

    for row in lookup_records:
        rest_id = _clean_text(row.get("rest_id"))
        if not rest_id:
            continue

        menu_rows = menu_groups.get_group(rest_id).to_dict(orient="records") if menu_groups is not None and rest_id in menu_groups.groups else []
        review_rows = review_groups.get_group(rest_id).to_dict(orient="records") if review_groups is not None and rest_id in review_groups.groups else []
        image_rows = image_groups.get_group(rest_id).to_dict(orient="records") if image_groups is not None and rest_id in image_groups.groups else []

        image_paths = [_clean_text(image_row.get("image_path")) for image_row in image_rows if _clean_text(image_row.get("image_path"))]
        representative_image_path = _first_existing_path(image_paths)

        menu_items = _extract_menu_items(menu_rows)
        review_snippets = _extract_review_snippets(review_rows)
        bio_text = bio_map.get(rest_id, "")

        search_text = " ".join(
            [
                _clean_text(row.get("name")),
                _clean_text(row.get("borough")),
                _clean_text(row.get("michelin_category")),
                bio_text,
                _join_snippets(menu_items),
                _join_snippets(review_snippets),
            ]
        ).strip()

        records.append(
            {
                "rest_id": rest_id,
                "restaurant_name": _clean_text(row.get("name")),
                "homepage": _clean_text(row.get("homepage")),
                "borough": _clean_text(row.get("borough")),
                "michelin_category": _clean_text(row.get("michelin_category")),
                "bio_text": bio_text,
                "menu_items": menu_items,
                "review_snippets": review_snippets,
                "image_paths": image_paths,
                "representative_image_path": representative_image_path,
                "menu_count": len(menu_rows),
                "review_count": len(review_rows),
                "image_count": len(image_rows),
                "has_menu": bool(menu_rows),
                "has_reviews": bool(review_rows),
                "has_food_images": bool(image_rows),
                "search_text": search_text,
            }
        )

    return pd.DataFrame(records)


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
    """Scores restaurant text results by max-pooling semantic search over fine-grained vectors and adding a lexical bonus."""
    if catalog.empty or not _clean_text(query):
        return pd.DataFrame()

    query_vector = np.array(embed_text(query), dtype=np.float32)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

    # Determine which embedding file to use for semantic score
    if scope == "Menu items":
        embeddings_map = load_finegrained_embeddings(str(MENU_EMBEDDINGS_JSONL))
        generic_map = None
    elif scope == "Reviews":
        embeddings_map = load_finegrained_embeddings(str(REVIEW_EMBEDDINGS_JSONL))
        generic_map = None
    else:
        # For "Bios" or "All", fallback to the generic restaurant profiles, 
        # or we could combine. But to keep it performant, we use generic map.
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
                # Max-pooling: compute similarity against all raw vectors and take the max
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
    """Scores visual search results using cross-modal alignment between query image and raw food/interior image vectors."""
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

        # Max-pooling over all specific image vectors
        scores = [_cosine_similarity(query_vector, v) for v in all_vectors]
        best_score = max(scores)
        
        rows.append({**row, "score": best_score})

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(by="score", ascending=False)


def _render_result_card(row: Dict[str, Any], score_label: str) -> None:
    with st.container(border=True):
        left, right = st.columns([1, 2])

        with left:
            image_path = _clean_text(row.get("representative_image_path"))
            resolved = _resolve_path(image_path) if image_path else None
            if resolved and resolved.exists():
                st.image(str(resolved), use_container_width=True)
            else:
                st.info("No representative image available.")

        with right:
            st.markdown(f"### {_clean_text(row.get('restaurant_name'))}")
            st.caption(
                f"{_clean_text(row.get('rest_id'))} • {_clean_text(row.get('borough')) or 'N/A'} • "
                f"{_clean_text(row.get('michelin_category')) or 'N/A'}"
            )
            st.metric(score_label, f"{float(row.get('score', 0.0)):.3f}")

            homepage = _clean_text(row.get("homepage"))
            if homepage:
                st.markdown(f"[Homepage]({homepage})")

            counts = st.columns(4)
            counts[0].metric("Menus", int(row.get("menu_count", 0)))
            counts[1].metric("Reviews", int(row.get("review_count", 0)))
            counts[2].metric("Images", int(row.get("image_count", 0)))
            counts[3].metric("Bio", "Yes" if _clean_text(row.get("bio_text")) else "No")

            bio_text = _clean_text(row.get("bio_text"))
            if bio_text:
                title = _truncate(bio_text, 80)
                with st.expander(title):
                    st.write(bio_text)

            menu_items = row.get("menu_items", []) or []
            if menu_items:
                st.markdown("**Menu highlights**")
                for item in menu_items[:3]:
                    st.write(f"- {_truncate(_clean_text(item), 160)}")

            review_snippets = row.get("review_snippets", []) or []
            if review_snippets:
                st.markdown("**Review snippets**")
                for snippet in review_snippets[:2]:
                    title = _truncate(snippet, 75)
                    with st.expander(title):
                        st.write(snippet)


def _render_data_overview(catalog: pd.DataFrame) -> None:
    lookup_df = load_lookup_df()
    reviews_df = load_reviews_df()
    images_df = load_images_df()
    menus_df = load_menus_df()
    bios_df = load_bios_df()

    st.subheader("Current Data Configuration")
    cols = st.columns(5)
    cols[0].metric("Restaurants", len(lookup_df))
    cols[1].metric("Reviews", len(reviews_df))
    cols[2].metric("Images", len(images_df))
    cols[3].metric("Menu rows", len(menus_df))
    cols[4].metric("Bios", len(bios_df))

    st.markdown("### Source files")
    file_rows = [
        (str(LOOKUP_CSV), len(lookup_df)),
        (str(REVIEWS_CSV), len(reviews_df)),
        (str(IMAGES_CSV), len(images_df)),
        (str(MENUS_JSON), len(menus_df)),
        (str(BIOS_JSON), len(bios_df)),
    ]
    st.dataframe(pd.DataFrame(file_rows, columns=["file", "rows"]), use_container_width=True, hide_index=True)

    if not catalog.empty:
        st.markdown("### Coverage summary")
        coverage_df = pd.DataFrame(
            [
                {"field": "has_menu", "restaurants": int(catalog["has_menu"].sum())},
                {"field": "has_reviews", "restaurants": int(catalog["has_reviews"].sum())},
                {"field": "has_food_images", "restaurants": int(catalog["has_food_images"].sum())},
                {"field": "has_bio", "restaurants": int(catalog["bio_text"].astype(bool).sum())},
            ]
        )
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)


@st.cache_resource(show_spinner=False)
def load_mdn_model():
    if not MDN_CHECKPOINT.exists():
        return None
    model = MDNScorer.load_from_checkpoint(str(MDN_CHECKPOINT), map_location="cpu")
    model.eval()
    return model


def _score_mdn_recommendations(catalog: pd.DataFrame, user_ratings: Dict[str, float]) -> pd.DataFrame:
    if catalog.empty or not user_ratings:
        return pd.DataFrame()
        
    model = load_mdn_model()
    if model is None:
        st.error("MDN checkpoint not found.")
        return pd.DataFrame()
        
    embeddings_map = load_restaurant_embeddings()
    
    hist_vecs = []
    hist_weights = []
    for rest_id, rating in user_ratings.items():
        vec = embeddings_map.get(rest_id)
        if vec is not None:
            weight = float(rating) / 5.0
            hist_vecs.append(torch.from_numpy(vec).float() * weight)
            hist_weights.append(weight)
            
    if not hist_vecs:
        return pd.DataFrame()
        
    user_vec = torch.stack(hist_vecs).sum(dim=0) / sum(hist_weights)
    mean_hist_rating = sum(user_ratings.values()) / len(user_ratings)
    scalar_feature = torch.tensor([mean_hist_rating], dtype=torch.float32)
    
    rows = []
    for row in catalog.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        if rest_id in user_ratings:
            continue
            
        target_vec_np = embeddings_map.get(rest_id)
        if target_vec_np is None:
            continue
            
        target_vec = torch.from_numpy(target_vec_np).float()
        feature_vec = torch.cat([user_vec, target_vec, scalar_feature]).unsqueeze(0)
        
        with torch.no_grad():
            mus, log_sigmas, pi_logits = model(feature_vec)
            pis = torch.softmax(pi_logits, dim=1)
            expected_mu = (mus * pis).sum(dim=1).item()
            
            # Calculate PDF for HDR Visualization
            grid_y = torch.linspace(1.0, 5.0, 101)
            grid_y_expanded = grid_y.view(1, -1, 1)
            mus_exp = mus.unsqueeze(1)
            sigmas_exp = torch.exp(log_sigmas).unsqueeze(1)
            pis_exp = pis.unsqueeze(1)
            component_pdfs = (1.0 / (2.0 * sigmas_exp)) * torch.exp(-torch.abs(grid_y_expanded - mus_exp) / sigmas_exp)
            total_pdfs = (pis_exp * component_pdfs).sum(dim=2)
            pdf_array = total_pdfs[0].cpu().numpy()
            
        rows.append({**row, "score": expected_mu, "pdf_grid": pdf_array})
        
    if not rows:
        return pd.DataFrame()
        
    result_df = pd.DataFrame(rows)
    return result_df.sort_values(by="score", ascending=False)


def main() -> None:
    st.set_page_config(page_title="Calm Octopuses Frontend", layout="wide")
    st.title("Calm Octopuses")
    st.caption("Raw-data Streamlit frontend for the current vector DB handoff.")

    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    catalog = build_restaurant_catalog()

    search_tab, dish_search_tab, browse_tab, rec_tab, overview_tab = st.tabs(["Search", "Dish Search", "Browse Restaurants", "Recommended", "Data Overview"])

    with search_tab:
        st.subheader("Multimodal Search")
        search_mode = st.radio("Search by", ["Text", "Image"], horizontal=True)
        top_k = st.slider("Number of results", min_value=3, max_value=12, value=6)

        if search_mode == "Text":
            scope = st.selectbox("Search scope", ["All", "Menu items", "Reviews", "Bios"])
            query = st.text_input("Enter a dish, description, or restaurant concept")

            if query:
                with st.spinner("Searching restaurant cards..."):
                    results_df = score_text_results(catalog, query, scope)

                if results_df.empty:
                    st.info("No matches found. Try a broader query.")
                else:
                    st.success(f"Found {len(results_df)} matching restaurants.")
                    for _, result_row in results_df.head(top_k).iterrows():
                        _render_result_card(result_row.to_dict(), "Match score")

        else:
            uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Query image", use_container_width=True)
                with st.spinner("Finding visually similar restaurant images..."):
                    image_bytes = uploaded_image.getvalue()
                    image_digest = hashlib.sha1(image_bytes).hexdigest()
                    temp_path = PROJECT_ROOT / f".streamlit_uploaded_query_image_{image_digest}.png"
                    temp_path.write_bytes(uploaded_image.getbuffer())
                    try:
                        results_df = score_image_results(catalog, str(temp_path))
                    finally:
                        if temp_path.exists():
                            temp_path.unlink()

                if results_df.empty:
                    st.info("No visually similar restaurants found.")
                else:
                    st.success(f"Found {len(results_df)} visually similar restaurants.")
                    for _, result_row in results_df.head(top_k).iterrows():
                        _render_result_card(result_row.to_dict(), "Image similarity")

    with dish_search_tab:
        st.subheader("Exact Dish Search")
        st.write("Search specifically for an exact dish or ingredient across all menus.")
        dish_query = st.text_input("Enter exact dish name or ingredient")
        
        if dish_query:
            with st.spinner("Searching menus..."):
                menus_df = load_menus_df()
                exact_results_df = score_exact_dish_search(catalog, menus_df, dish_query)
                
            if exact_results_df.empty:
                st.info("No restaurants found.")
            else:
                user_ratings = st.session_state.get("user_ratings", {})
                if user_ratings:
                    with st.spinner("Ranking by predicted rating..."):
                        scored_df = _score_mdn_recommendations(exact_results_df, user_ratings)
                    if not scored_df.empty:
                        exact_results_df = scored_df
                        sort_label = "Predicted Rating"
                    else:
                        sort_label = "Exact Match"
                else:
                    sort_label = "Exact Match"

                st.success(f"Found {len(exact_results_df)} restaurants with exact matches.")
                for _, result_row in exact_results_df.iterrows():
                    _render_result_card(result_row.to_dict(), sort_label)

    with browse_tab:
        st.subheader("Restaurant Browser")
        if catalog.empty:
            st.warning("No restaurant data available.")
        else:
            restaurant_names = catalog["restaurant_name"].tolist()
            selected_name = st.selectbox("Choose a restaurant", restaurant_names)
            selected_row = catalog[catalog["restaurant_name"] == selected_name].iloc[0].to_dict()
            
            st.subheader("Rate this Restaurant")
            current_rating = st.session_state.user_ratings.get(selected_row["rest_id"])
            
            if current_rating:
                st.write(f"Your rating: **{int(current_rating)} ⭐**")
            
            feedback_val = st.feedback("stars", key=f"stars_{selected_row['rest_id']}")
            
            if feedback_val is not None:
                st.session_state.user_ratings[selected_row["rest_id"]] = float(feedback_val + 1)
                
            if current_rating:
                if st.button("Clear Rating", key=f"clear_{selected_row['rest_id']}"):
                    del st.session_state.user_ratings[selected_row["rest_id"]]
                    st.rerun()
            st.divider()

            _render_result_card(selected_row, "Catalog score")

            st.markdown("### Raw records")
            raw_sections = st.tabs(["Lookup", "Menus", "Reviews", "Images", "Bio"])

            with raw_sections[0]:
                lookup_df = load_lookup_df()
                st.dataframe(
                    lookup_df[lookup_df["rest_id"] == selected_row["rest_id"]],
                    use_container_width=True,
                    hide_index=True,
                )

            with raw_sections[1]:
                menus_df = load_menus_df()
                if menus_df.empty:
                    st.info("No menu rows available.")
                else:
                    st.dataframe(
                        menus_df[menus_df["rest_id"] == selected_row["rest_id"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            with raw_sections[2]:
                reviews_df = load_reviews_df()
                if reviews_df.empty:
                    st.info("No review rows available.")
                else:
                    st.dataframe(
                        reviews_df[reviews_df["rest_id"] == selected_row["rest_id"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            with raw_sections[3]:
                images_df = load_images_df()
                subset = images_df[images_df["rest_id"] == selected_row["rest_id"]]
                if subset.empty:
                    st.info("No image rows available.")
                else:
                    image_paths = subset["image_path"].tolist()
                    for image_path in image_paths[:12]:
                        candidate = _resolve_path(image_path)
                        if candidate.exists():
                            st.image(str(candidate), caption=candidate.name, use_container_width=True)
                        else:
                            st.write(image_path)

            with raw_sections[4]:
                bio_text = _clean_text(selected_row.get("bio_text"))
                if bio_text:
                    st.write(bio_text)
                else:
                    st.info("No bio available.")

    with rec_tab:
        st.subheader("Personalized Recommendations")
        if not st.session_state.user_ratings:
            st.info("Please rate some restaurants in the 'Browse Restaurants' tab to get personalized recommendations!")
        else:
            st.write(f"You have rated **{len(st.session_state.user_ratings)}** restaurants.")
            with st.spinner("Generating MDN recommendations..."):
                rec_df = _score_mdn_recommendations(catalog, st.session_state.user_ratings)
            
            if rec_df.empty:
                st.warning("Could not generate recommendations.")
            else:
                for _, result_row in rec_df.head(6).iterrows():
                    _render_result_card(result_row.to_dict(), "Predicted Rating")
                    
                    st.caption("Rating Probability Distribution (HDR)")
                    chart_df = pd.DataFrame({"Probability Density": result_row["pdf_grid"]}, index=np.linspace(1.0, 5.0, 101))
                    st.area_chart(chart_df, height=150, color="#FF4B4B")

    with overview_tab:
        _render_data_overview(catalog)


if __name__ == "__main__":
    main()



