"""Streamlit frontend for the Calm Octopuses raw data MVP.

This entrypoint stays lightweight: it reads the raw data files already present
in the repo, lets users search by text or image, and renders restaurant-level
cards keyed by rest_id.
"""

from __future__ import annotations

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
                    if st.session_state.user_ratings:
                        with st.spinner("Adding personalized predicted ratings..."):
                            results_df = add_mdn_predictions(results_df, st.session_state.user_ratings)
                    
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
                    if st.session_state.user_ratings:
                        with st.spinner("Adding personalized predicted ratings..."):
                            results_df = add_mdn_predictions(results_df, st.session_state.user_ratings)
                            
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
                        scored_df = add_mdn_predictions(exact_results_df, user_ratings)
                        if "predicted_rating" in scored_df.columns:
                            scored_df = scored_df.sort_values(by="predicted_rating", ascending=False)
                    if not scored_df.empty:
                        exact_results_df = scored_df
                
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
