"""
frontend.py
===========
Main Streamlit entry point for the Calm Octopuses restaurant recommendation UI.

This entrypoint stays lightweight: it reads the raw data files already present
in the repo, lets users search by text or image, and renders restaurant-level
cards keyed by rest_id.

Tabs
----
Search          CLIP semantic search over the full restaurant catalog.
Dish Search     Exact-match search over individual menu items.
Explore         Browse and rate restaurants; get MDN-personalised recommendations.
Data Overview   Coverage metrics and source file health-check.

Run with::

    streamlit run frontend.py
"""

from __future__ import annotations

from html import escape
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.data_loader import (
    build_restaurant_catalog,
    load_menus_df,
    load_lookup_df,
    load_reviews_df,
    load_images_df,
    _clean_text,
    _resolve_path,
)
from algorithms.retrieval import score_text_results, score_image_results, score_exact_dish_search
from algorithms.mdn_regression import _score_mdn_recommendations, add_mdn_predictions
from ui_components.cards import _render_result_card
from ui_components.overview import _render_data_overview
from ui_components.theme import apply_global_theme, render_app_hero, render_section_intro


def _render_menu_browser(menu_rows: pd.DataFrame) -> None:
    """Render parsed menu items in a reader-friendly format."""
    if menu_rows.empty:
        st.info("No parsed menu items are available for this restaurant yet.")
        return

    st.markdown("<p class='co-note'>Parsed dishes from the current menu data.</p>", unsafe_allow_html=True)
    for _, menu_row in menu_rows.head(18).fillna("").iterrows():
        dish_name = _clean_text(menu_row.get("dish_name")) or "Unnamed dish"
        ingredients = _clean_text(menu_row.get("ingredients")) or "No ingredient notes available."
        price = _clean_text(menu_row.get("price"))
        price_html = (
            f"<span class='co-inline-badge co-inline-badge--accent'>${escape(price)}</span>"
            if price
            else ""
        )
        st.markdown(
            f"""
            <div class="co-content-card">
                <div class="co-content-head">
                    <p class="co-content-title">{escape(dish_name)}</p>
                    {price_html}
                </div>
                <p class="co-content-copy">{escape(ingredients)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(menu_rows) > 18:
        st.caption(f"Showing 18 of {len(menu_rows)} parsed menu rows.")


def _render_review_browser(review_rows: pd.DataFrame) -> None:
    """Render reviews as readable quote cards instead of raw rows."""
    if review_rows.empty:
        st.info("No review text is available for this restaurant yet.")
        return

    st.markdown("<p class='co-note'>Actual review text pulled into the catalog.</p>", unsafe_allow_html=True)
    for _, review_row in review_rows.head(8).fillna("").iterrows():
        review_text = _clean_text(review_row.get("text"))
        rating = _clean_text(review_row.get("rating"))
        source = _clean_text(review_row.get("source"))
        if not review_text:
            continue

        meta_parts = []
        if source:
            meta_parts.append(_clean_text(source))
        if rating:
            meta_parts.append(f"{rating} stars")
        meta_html = "".join(
            f"<span class='co-inline-badge co-inline-badge--soft'>{escape(part)}</span>"
            for part in meta_parts
        )

        st.markdown(
            f"""
            <div class="co-content-card">
                <div class="co-review-meta">{meta_html}</div>
                <p class="co-review-body">{escape(review_text)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(review_rows) > 8:
        st.caption(f"Showing 8 of {len(review_rows)} review rows.")


def _render_photo_browser(image_rows: pd.DataFrame) -> None:
    """Render a simple image gallery for the selected restaurant."""
    if image_rows.empty:
        st.info("No restaurant images are available yet.")
        return

    image_paths = []
    seen_paths = set()
    for image_path in image_rows["image_path"].tolist():
        cleaned_path = _clean_text(image_path)
        if cleaned_path and cleaned_path not in seen_paths:
            image_paths.append(cleaned_path)
            seen_paths.add(cleaned_path)

    if not image_paths:
        st.info("No restaurant images are available yet.")
        return

    cols = st.columns(3)
    rendered = 0
    for idx, image_path in enumerate(image_paths[:9]):
        candidate = _resolve_path(image_path)
        target_col = cols[idx % 3]
        with target_col:
            if candidate.exists():
                st.image(str(candidate), width="stretch")
            else:
                st.markdown(
                    f"""
                    <div class="co-content-card">
                        <p class="co-content-copy">{escape(image_path)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        rendered += 1

    if len(image_paths) > rendered:
        st.caption(f"Showing {rendered} of {len(image_paths)} image rows.")


def _render_bio_browser(bio_text: str) -> None:
    """Render the restaurant bio in a simple bordered card."""
    cleaned_bio = _clean_text(bio_text)
    if not cleaned_bio:
        st.info("No bio is available for this restaurant yet.")
        return

    st.markdown(
        f"""
        <div class="co-content-card">
            <p class="co-review-body">{escape(cleaned_bio)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_debug_table(df: pd.DataFrame, max_rows: int = 12) -> None:
    """Render debug records in a light-weight table instead of the dark grid widget."""
    if df.empty:
        st.info("No rows available.")
        return

    preview_df = df.head(max_rows).copy().fillna("")
    st.table(preview_df)
    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} rows.")

def main() -> None:
    """
    Build and render the full Calm Octopuses Streamlit frontend.

    Called once per page load by Streamlit's script runner.  Sets up
    session state, builds the tab layout, and delegates to the relevant
    retrieval and UI component functions for each tab.
    """
    st.set_page_config(page_title="Calm Octopuses Frontend", layout="wide")
    apply_global_theme()

    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    catalog = build_restaurant_catalog()
    render_app_hero(len(catalog))

    search_tab, dish_search_tab, browse_tab, rec_tab, my_rest_tab, overview_tab = st.tabs(["Search", "Dish Search", "Browse Restaurants", "Recommended", "My Restaurants", "Data Overview"])

    with search_tab:
        render_section_intro(
            "Discover",
            "Multimodal Search",
            "Search across dishes, reviews, and restaurant profiles, or start with an image to explore the catalog visually.",
        )
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
                    if st.session_state.user_ratings:
                        with st.spinner("Adding personalized predicted ratings..."):
                            results_df = add_mdn_predictions(results_df, st.session_state.user_ratings)
                    
                    st.success(f"Found {len(results_df)} matching restaurants.")
                    for idx, (_, result_row) in enumerate(results_df.head(top_k).iterrows()):
                        _render_result_card(result_row.to_dict(), "Match score", render_key=f"search_text_{idx}")

        else:
            uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_image is not None:
                st.markdown("<p class='co-note'>Reference image</p>", unsafe_allow_html=True)
                st.image(uploaded_image, caption="Query image", width="stretch")
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
                    if st.session_state.user_ratings:
                        with st.spinner("Adding personalized predicted ratings..."):
                            results_df = add_mdn_predictions(results_df, st.session_state.user_ratings)
                            
                    st.success(f"Found {len(results_df)} visually similar restaurants.")
                    for idx, (_, result_row) in enumerate(results_df.head(top_k).iterrows()):
                        _render_result_card(result_row.to_dict(), "Image similarity", render_key=f"search_image_{idx}")

    with dish_search_tab:
        render_section_intro(
            "Precision",
            "Exact Dish Search",
            "Use a direct dish or ingredient lookup when you want to see which restaurants explicitly list a specific preparation.",
        )
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
                        scored_df = add_mdn_predictions(exact_results_df, user_ratings)
                        if "predicted_rating" in scored_df.columns:
                            scored_df = scored_df.sort_values(by="predicted_rating", ascending=False)
                    if not scored_df.empty:
                        exact_results_df = scored_df
                
                sort_label = "Exact Match"

                st.success(f"Found {len(exact_results_df)} restaurants with exact matches.")
                for idx, (_, result_row) in enumerate(exact_results_df.iterrows()):
                    _render_result_card(result_row.to_dict(), sort_label, render_key=f"exact_dish_{idx}")

    with browse_tab:
        render_section_intro(
            "Browse",
            "Restaurant Browser",
            "Move through the full catalog, look at the actual dishes and reviews, and leave ratings to shape your recommendations.",
        )
        if catalog.empty:
            st.warning("No restaurant data available.")
        else:
            restaurant_names = catalog["restaurant_name"].tolist()
            selected_name = st.selectbox("Choose a restaurant", restaurant_names)
            selected_row = catalog[catalog["restaurant_name"] == selected_name].iloc[0].to_dict()
            selected_rest_id = selected_row["rest_id"]

            menus_df = load_menus_df()
            reviews_df = load_reviews_df()
            images_df = load_images_df()
            lookup_df = load_lookup_df()
            selected_menu_rows = (
                menus_df[menus_df["rest_id"] == selected_rest_id].copy()
                if not menus_df.empty and "rest_id" in menus_df.columns
                else pd.DataFrame()
            )
            selected_review_rows = (
                reviews_df[reviews_df["rest_id"] == selected_rest_id].copy()
                if not reviews_df.empty and "rest_id" in reviews_df.columns
                else pd.DataFrame()
            )
            selected_image_rows = (
                images_df[images_df["rest_id"] == selected_rest_id].copy()
                if not images_df.empty and "rest_id" in images_df.columns
                else pd.DataFrame()
            )
            
            st.markdown("<p class='co-eyebrow'>Preference Signal</p>", unsafe_allow_html=True)
            st.markdown("### Rate This Restaurant")
            current_rating = st.session_state.user_ratings.get(selected_rest_id)

            rating_options = ["Not rated", "1", "2", "3", "4", "5"]
            rating_index = 0 if current_rating is None else rating_options.index(str(int(current_rating)))
            selected_rating = st.radio(
                "Choose a rating",
                rating_options,
                index=rating_index,
                horizontal=True,
                key=f"rating_{selected_rest_id}",
                label_visibility="collapsed",
                format_func=lambda option: "Not rated" if option == "Not rated" else f"{option} star",
            )

            if selected_rating == "Not rated":
                if current_rating is not None:
                    del st.session_state.user_ratings[selected_rest_id]
            else:
                st.session_state.user_ratings[selected_rest_id] = float(selected_rating)

            active_rating = st.session_state.user_ratings.get(selected_rest_id)
            if active_rating is not None:
                st.caption(f"Current rating: {int(active_rating)} / 5")
            else:
                st.caption("Rate a few restaurants here so the recommendation tab has a preference signal.")
            st.divider()

            _render_result_card(selected_row, "Catalog score", render_key="browse_selected")

            browse_sections = st.tabs(["Menu", "Reviews", "Photos", "Bio"])

            with browse_sections[0]:
                _render_menu_browser(selected_menu_rows)

            with browse_sections[1]:
                _render_review_browser(selected_review_rows)

            with browse_sections[2]:
                _render_photo_browser(selected_image_rows)

            with browse_sections[3]:
                _render_bio_browser(selected_row.get("bio_text", ""))

            st.markdown("<div class='co-raw-shell'></div>", unsafe_allow_html=True)
            with st.expander("Source records (debug)"):
                raw_sections = st.tabs(["Lookup", "Menus", "Reviews", "Images", "Bio"])

                with raw_sections[0]:
                    _render_debug_table(
                        lookup_df[lookup_df["rest_id"] == selected_rest_id].reset_index(drop=True),
                        max_rows=5,
                    )

                with raw_sections[1]:
                    _render_debug_table(selected_menu_rows.reset_index(drop=True), max_rows=12)

                with raw_sections[2]:
                    _render_debug_table(selected_review_rows.reset_index(drop=True), max_rows=10)

                with raw_sections[3]:
                    _render_debug_table(selected_image_rows.reset_index(drop=True), max_rows=10)

                with raw_sections[4]:
                    bio_text = _clean_text(selected_row.get("bio_text"))
                    if bio_text:
                        st.write(bio_text)
                    else:
                        st.info("No bio available.")

    with rec_tab:
        render_section_intro(
            "Taste Profile",
            "Personalized Recommendations",
            "Turn your ratings into a restaurant shortlist, with predicted affinity and uncertainty pulled from the MDN model.",
        )
        if not st.session_state.user_ratings:
            st.info("Please rate some restaurants to get personalized recommendations!")
        else:
            st.write(f"You have rated **{len(st.session_state.user_ratings)}** restaurants.")
            with st.spinner("Generating MDN recommendations..."):
                rec_df = _score_mdn_recommendations(catalog, st.session_state.user_ratings)
            
            if rec_df.empty:
                st.warning("Could not generate recommendations.")
            else:
                for idx, (_, result_row) in enumerate(rec_df.head(6).iterrows()):
                    score_label = "Predicted Rating"
                    if result_row.get("recommendation_mode") == "embedding_fallback":
                        score_label = "Recommendation Score"

                    _render_result_card(result_row.to_dict(), score_label, render_key=f"recommendation_{idx}")

                    if "pdf_grid" in result_row and isinstance(result_row["pdf_grid"], (list, np.ndarray)):
                        st.caption("Rating Probability Distribution (HDR)")
                        chart_df = pd.DataFrame({"Probability Density": result_row["pdf_grid"]}, index=np.linspace(1.0, 5.0, 101))
                        st.area_chart(chart_df, height=150, color="#FF4B4B")

    with my_rest_tab:
        st.subheader("My Rated Restaurants")
        if not st.session_state.user_ratings:
            st.info("You haven't rated any restaurants yet. Browse or search to add ratings!")
        else:
            rated_ids = list(st.session_state.user_ratings.keys())
            my_df = catalog[catalog["rest_id"].isin(rated_ids)].copy()
            my_df["actual_rating"] = my_df["rest_id"].map(st.session_state.user_ratings)
            my_df = my_df.sort_values(by="actual_rating", ascending=False)
            
            st.write(f"You have rated **{len(my_df)}** restaurants.")
            for idx, (_, result_row) in enumerate(my_df.iterrows()):
                _render_result_card(result_row.to_dict(), "Your Rating", render_key=f"my_rated_{idx}")

    with overview_tab:
        render_section_intro(
            "Coverage",
            "Data Overview",
            "Inspect how complete the current catalog is, including which source files are feeding the frontend and where the gaps still are.",
        )
        _render_data_overview(catalog)

if __name__ == "__main__":
    main()
