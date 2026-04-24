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


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
REVIEWS_CSV = DATA_DIR / "social_reviews.csv"
IMAGES_CSV = DATA_DIR / "social_images.csv"
MENUS_JSON = DATA_DIR / "extracted_menus" / "final_parsed_menus.json"
BIOS_JSON = DATA_DIR / "extracted_bios" / "restaurant_bios_joinable.json"
EMBEDDINGS_JSONL = DATA_DIR / "embeddings" / "restaurant_profiles.jsonl"
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
        text = _truncate(_clean_text(row.get("text")), 180)
        rating = _clean_text(row.get("rating"))
        if text:
            snippets.append(f"{text} {'(' + rating + ')' if rating else ''}".strip())
    return snippets


@st.cache_data(show_spinner=False)
def build_restaurant_catalog() -> pd.DataFrame:
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


def _score_text_results(catalog: pd.DataFrame, query: str, scope: str) -> pd.DataFrame:
    """Scores restaurant text results by combining semantic search over pre-computed profiles and a lexical bonus."""
    if catalog.empty or not _clean_text(query):
        return pd.DataFrame()

    query_vector = np.array(embed_text(query), dtype=np.float32)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

    embeddings_map = load_restaurant_embeddings()

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

        candidate_vector = embeddings_map.get(rest_id)
        if candidate_vector is None:
            continue

        semantic_score = _cosine_similarity(query_vector, candidate_vector)
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


def _score_image_results(catalog: pd.DataFrame, image_path: str) -> pd.DataFrame:
    """Scores visual search results using cross-modal alignment between the query image and pre-computed text profiles."""
    if catalog.empty or not _clean_text(image_path):
        return pd.DataFrame()

    query_vector = np.array(embed_image(image_path), dtype=np.float32)
    if query_vector.size == 0:
        return pd.DataFrame()

    embeddings_map = load_restaurant_embeddings()

    rows: List[Dict[str, Any]] = []
    for row in catalog.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        candidate_vector = embeddings_map.get(rest_id)
        if candidate_vector is None:
            continue

        score = _cosine_similarity(query_vector, candidate_vector)
        rows.append({**row, "score": score})

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
                st.write(_truncate(bio_text, 280))

            menu_items = row.get("menu_items", []) or []
            if menu_items:
                st.markdown("**Menu highlights**")
                for item in menu_items[:3]:
                    st.write(f"- {_truncate(_clean_text(item), 160)}")

            review_snippets = row.get("review_snippets", []) or []
            if review_snippets:
                st.markdown("**Review snippets**")
                for snippet in review_snippets[:2]:
                    st.write(f"- {_truncate(_clean_text(snippet), 180)}")


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


def main() -> None:
    st.set_page_config(page_title="Calm Octopuses Frontend", layout="wide")
    st.title("Calm Octopuses")
    st.caption("Raw-data Streamlit frontend for the current vector DB handoff.")

    catalog = build_restaurant_catalog()

    search_tab, browse_tab, overview_tab = st.tabs(["Search", "Browse Restaurants", "Data Overview"])

    with search_tab:
        st.subheader("Multimodal Search")
        search_mode = st.radio("Search by", ["Text", "Image"], horizontal=True)
        top_k = st.slider("Number of results", min_value=3, max_value=12, value=6)

        if search_mode == "Text":
            scope = st.selectbox("Search scope", ["All", "Menu items", "Reviews", "Bios"])
            query = st.text_input("Enter a dish, description, or restaurant concept")

            if query:
                with st.spinner("Searching restaurant cards..."):
                    results_df = _score_text_results(catalog, query, scope)

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
                        results_df = _score_image_results(catalog, str(temp_path))
                    finally:
                        if temp_path.exists():
                            temp_path.unlink()

                if results_df.empty:
                    st.info("No visually similar restaurants found.")
                else:
                    st.success(f"Found {len(results_df)} visually similar restaurants.")
                    for _, result_row in results_df.head(top_k).iterrows():
                        _render_result_card(result_row.to_dict(), "Image similarity")

    with browse_tab:
        st.subheader("Restaurant Browser")
        if catalog.empty:
            st.warning("No restaurant data available.")
        else:
            restaurant_names = catalog["restaurant_name"].tolist()
            selected_name = st.selectbox("Choose a restaurant", restaurant_names)
            selected_row = catalog[catalog["restaurant_name"] == selected_name].iloc[0].to_dict()
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

    with overview_tab:
        _render_data_overview(catalog)


if __name__ == "__main__":
    main()
