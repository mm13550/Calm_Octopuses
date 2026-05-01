"""
frontend.py
===========
Main Streamlit entry point for the Calm Octopuses restaurant recommendation UI.

This entrypoint stays lightweight: it reads the raw data files already present
in the repo, lets users search by text or image, and renders restaurant-level
cards keyed by rest_id.

Tabs
----
Browse Restaurants  Default landing page; lightweight catalog view (near-instant).
Search              CLIP semantic search (text/image) over the restaurant catalog.
Dish Search         Vector-based search over individual menu items.
Recommended         Personalised MDN-regression based rating predictions.
My Restaurants      Filtered list of restaurants the user has rated.
Data Overview       Coverage metrics and source file health-check.

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
    _format_menu_price,
    _resolve_path,
)
from core.logic import get_restaurant_details, filter_my_restaurants
from ui_components.cards import _render_result_card
from ui_components.overview import _render_data_overview
from ui_components.theme import apply_global_theme, render_app_hero, render_section_intro


def _load_mdn_functions() -> tuple[object | None, object | None, str | None]:
    """
    Import optional MDN helpers without taking down the whole frontend.

    Some local environments have the core Streamlit stack installed but are
    missing heavier recommendation dependencies such as PyTorch Lightning.
    The search and browse UX should still load in that case.

    Returns
    -------
    tuple[object | None, object | None, str | None]
        A 3-tuple of (add_mdn_predictions, score_mdn_recommendations, error_message).
        If import succeeds, error_message is None. If heavy dependencies are missing,
        the functions are None and error_message contains user-facing text.
    """
    try:
        from algorithms.mdn_regression import add_mdn_predictions, _score_mdn_recommendations
        return add_mdn_predictions, _score_mdn_recommendations, None
    except ModuleNotFoundError as exc:
        # Gracefully handle missing PyTorch Lightning or related heavy dependencies
        if exc.name in {"pytorch_lightning", "lightning", "lightning_fabric", "torchmetrics"}:
            return None, None, (
                "Recommendation features are temporarily unavailable because "
                f"`{exc.name}` is not installed in the current Python environment."
            )
        raise


def _render_menu_browser(menu_rows: pd.DataFrame) -> None:
    """
    Render parsed menu items in a reader-friendly format.
    
    Displays up to 18 menu items with dish name, ingredients, and price in
    styled cards. Items beyond 18 are indicated with a count caption.

    Parameters
    ----------
    menu_rows : pd.DataFrame
        DataFrame with columns 'dish_name', 'ingredients', and 'price'.
    """
    if menu_rows.empty:
        st.info("No parsed menu items are available for this restaurant yet.")
        return

    st.html("<p class='co-note'>Parsed dishes from the current menu data.</p>")
    # Iterate and render up to 18 menu items with sanitized HTML
    for _, menu_row in menu_rows.head(18).fillna("").iterrows():
        dish_name = _clean_text(menu_row.get("dish_name")) or "Unnamed dish"
        ingredients = _clean_text(menu_row.get("ingredients")) or "No ingredient notes available."
        price = _format_menu_price(menu_row.get("price"))
        price_html = (
            f"<span class='co-inline-badge co-inline-badge--accent'>{escape(price)}</span>"
            if price
            else ""
        )
        st.html(
            f"""
            <div class="co-content-card">
                <div class="co-content-head">
                    <p class="co-content-title">{escape(dish_name)}</p>
                    {price_html}
                </div>
                <p class="co-content-copy">{escape(ingredients)}</p>
            </div>
            """
        )

    # Show count of remaining items if there are more than 18
    if len(menu_rows) > 18:
        st.caption(f"Showing 18 of {len(menu_rows)} parsed menu rows.")


def _render_review_browser(review_rows: pd.DataFrame) -> None:
    """
    Render reviews as readable quote cards instead of raw rows.
    
    Displays up to 8 reviews with metadata (source, rating) and the review text.
    Uses styled cards for a narrative presentation rather than a table view.

    Parameters
    ----------
    review_rows : pd.DataFrame
        DataFrame with columns 'text', 'rating', and 'source'.
    """
    if review_rows.empty:
        st.info("No review text is available for this restaurant yet.")
        return

    st.markdown("<p class='co-note'>Actual review text pulled into the catalog.</p>", unsafe_allow_html=True)
    # Iterate through up to 8 reviews and build styled cards
    for _, review_row in review_rows.head(8).fillna("").iterrows():
        review_text = _clean_text(review_row.get("text"))
        rating = _clean_text(review_row.get("rating"))
        source = _clean_text(review_row.get("source"))
        if not review_text:
            continue

        # Build metadata badges (source and/or rating)
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

    # Show count of remaining reviews if there are more than 8
    if len(review_rows) > 8:
        st.caption(f"Showing 8 of {len(review_rows)} review rows.")


def _render_photo_browser(image_rows: pd.DataFrame) -> None:
    """
    Render a simple image gallery for the selected restaurant.
    
    Displays up to 9 images in a 3-column grid. Deduplicates paths and shows
    actual images if the path exists, otherwise renders the path as text.

    Parameters
    ----------
    image_rows : pd.DataFrame
        DataFrame with column 'image_path' containing file paths.
    """
    if image_rows.empty:
        st.info("No restaurant images are available yet.")
        return

    # Deduplicate image paths to avoid rendering duplicates
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

    # Create 3-column layout and render up to 9 images
    cols = st.columns(3)
    rendered = 0
    for idx, image_path in enumerate(image_paths[:9]):
        candidate = _resolve_path(image_path)
        target_col = cols[idx % 3]
        with target_col:
            if candidate.exists():
                st.image(str(candidate), width="stretch")
            else:
                # Show the path as a fallback if image file is missing
                st.markdown(
                    f"""
                    <div class="co-content-card">
                        <p class="co-content-copy">{escape(image_path)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        rendered += 1

    # Show count of remaining images if there are more than 9
    if len(image_paths) > rendered:
        st.caption(f"Showing {rendered} of {len(image_paths)} image rows.")


def _render_bio_browser(bio_text: str) -> None:
    """
    Render the restaurant bio in a simple bordered card.
    
    Parameters
    ----------
    bio_text : str
        Plain text bio of the restaurant.
    """
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
    """
    Render debug records in a light-weight table instead of the dark grid widget.
    
    Used in the "Source records (debug)" expander to show raw underlying data
    from the data loaders (lookups, menus, reviews, images, bios).

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to render.
    max_rows : int, optional
        Maximum number of rows to display (default: 12).
    """
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

    Called once per page load by Streamlit's script runner. Sets up
    session state, builds the tab layout, and delegates to the relevant
    retrieval and UI component functions for each tab.

    Tab Structure
    ------
    - Search: Multimodal (text/image) semantic search over restaurants.
    - Dish Search: Exact dish/ingredient lookup across menu items.
    - Browse Restaurants: Full catalog browser with details and ratings.
    - Recommended: MDN-based personalized recommendations (requires rating history).
    - My Restaurants: Filtered view of user-rated restaurants.
    - Data Overview: Coverage metrics and source file health check.
    """
    # Configure Streamlit page settings and apply custom theme
    st.set_page_config(page_title="Calm Octopuses Frontend", layout="wide")
    apply_global_theme()

    # Lazy load heavy algorithm modules inside the blocks where they are used
    # to ensure the initial app load is near-instant.

    # Initialize session state for user ratings (persists across reruns)
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    # Load and cache the restaurant catalog
    catalog = build_restaurant_catalog()
    render_app_hero(len(catalog))

    # Create the six main tabs for different app sections
    browse_tab, search_tab, dish_search_tab, rec_tab, my_rest_tab, overview_tab = st.tabs(["Browse Restaurants", "Search", "Dish Search", "Recommended", "My Restaurants", "Data Overview"])

    # ============================================================================
    # SEARCH TAB: Multimodal semantic search (text or image)
    # ============================================================================
    with search_tab:
        from algorithms.retrieval import score_text_results, score_image_results
        render_section_intro(
            "Discover",
            "Multimodal Search",
            "Search across dishes, reviews, and restaurant profiles, or start with an image to explore the catalog visually.",
        )
        search_mode = st.radio("Search by", ["Text", "Image"], horizontal=True)
        top_k = st.slider("Number of results", min_value=3, max_value=12, value=6)

        # TEXT SEARCH BRANCH
        if search_mode == "Text":
            scope = st.selectbox("Search scope", ["All", "Menu items", "Reviews", "Bios"])
            query = st.text_input("Enter a dish, description, or restaurant concept")

            if query:
                with st.spinner("Searching restaurant cards..."):
                    results_df = score_text_results(catalog, query, scope)

                if results_df.empty:
                    st.info("No matches found. Try a broader query.")
                else:
                    # Optionally augment results with MDN predictions if user has ratings
                    if st.session_state.user_ratings:
                        add_mdn_predictions, _, mdn_error = _load_mdn_functions()
                        if add_mdn_predictions is not None:
                            with st.spinner("Adding personalized predicted ratings..."):
                                results_df = add_mdn_predictions(results_df, st.session_state.user_ratings)
                        elif mdn_error:
                            st.info(mdn_error)
                    
                    visible_results = results_df.head(top_k)
                    st.success(f"Showing top {len(visible_results)} of {len(results_df)} matching restaurants.")
                    for idx, (_, result_row) in enumerate(visible_results.iterrows()):
                        _render_result_card(result_row.to_dict(), "Match score", render_key=f"search_text_{idx}")

        # IMAGE SEARCH BRANCH
        else:
            uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_image is not None:
                st.markdown("<p class='co-note'>Reference image</p>", unsafe_allow_html=True)
                st.image(uploaded_image, caption="Query image", width="stretch")
                with st.spinner("Finding visually similar restaurant images..."):
                    image_bytes = uploaded_image.getvalue()
                    results_df = score_image_results(catalog, image_bytes)

                if results_df.empty:
                    st.info("No visually similar restaurants found.")
                else:
                    # Optionally augment results with MDN predictions if user has ratings
                    if st.session_state.user_ratings:
                        add_mdn_predictions, _, mdn_error = _load_mdn_functions()
                        if add_mdn_predictions is not None:
                            with st.spinner("Adding personalized predicted ratings..."):
                                results_df = add_mdn_predictions(results_df, st.session_state.user_ratings)
                        elif mdn_error:
                            st.info(mdn_error)
                            
                    visible_results = results_df.head(top_k)
                    st.success(f"Showing top {len(visible_results)} of {len(results_df)} visually similar restaurants.")
                    for idx, (_, result_row) in enumerate(visible_results.iterrows()):
                        _render_result_card(result_row.to_dict(), "Image similarity", render_key=f"search_image_{idx}")

    # ============================================================================
    # DISH SEARCH TAB: Exact dish/ingredient lookup
    # ============================================================================
    with dish_search_tab:
        from algorithms.retrieval import score_exact_dish_search
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
                # Score results with MDN predictions if user has ratings
                user_ratings = st.session_state.get("user_ratings", {})
                if user_ratings:
                    add_mdn_predictions, _, mdn_error = _load_mdn_functions()
                    if add_mdn_predictions is not None:
                        with st.spinner("Ranking by predicted rating..."):
                            scored_df = add_mdn_predictions(exact_results_df, user_ratings)
                            if "predicted_rating" in scored_df.columns:
                                scored_df = scored_df.sort_values(by="predicted_rating", ascending=False)
                        if not scored_df.empty:
                            exact_results_df = scored_df
                    elif mdn_error:
                        st.info(mdn_error)
                
                # Dynamic slider for number of results (up to 12)
                sort_label = "Menu match"
                max_visible_results = min(12, len(exact_results_df))
                default_visible_results = min(6, max_visible_results)
                query_key = hashlib.md5(dish_query.strip().lower().encode("utf-8")).hexdigest()[:8]
                dish_top_k = st.slider(
                    "Number of dish search results",
                    min_value=1,
                    max_value=max_visible_results,
                    value=default_visible_results,
                    key=f"dish_search_top_k_{query_key}_{max_visible_results}",
                )

                visible_results = exact_results_df.head(dish_top_k)
                st.success(f"Showing top {len(visible_results)} of {len(exact_results_df)} restaurants with menu matches.")
                for idx, (_, result_row) in enumerate(visible_results.iterrows()):
                    _render_result_card(result_row.to_dict(), sort_label, render_key=f"exact_dish_{idx}")

    # ============================================================================
    # BROWSE TAB: Full restaurant catalog with ratings
    # ============================================================================
    with browse_tab:
        render_section_intro(
            "Browse",
            "Restaurant Browser",
            "Move through the full catalog, look at the actual dishes and reviews, and leave ratings to shape your recommendations.",
        )
        if catalog.empty:
            st.warning("No restaurant data available.")
        else:
            # Allow user to select a restaurant from the full catalog
            restaurant_names = catalog["restaurant_name"].tolist()
            selected_name = st.selectbox("Choose a restaurant", restaurant_names)
            selected_row = catalog[catalog["restaurant_name"] == selected_name].iloc[0].to_dict()
            selected_rest_id = selected_row["rest_id"]

            # Use logic helper to get all restaurant-specific data
            details = get_restaurant_details(selected_rest_id)
            selected_menu_rows = details["menus"]
            selected_review_rows = details["reviews"]
            selected_image_rows = details["images"]
            lookup_df = details["lookup"]
            
            # Display main restaurant card with score
            _render_result_card(selected_row, "Catalog score", render_key="browse_selected")

            # Create sub-tabs for different content types
            browse_sections = st.tabs(["Menu", "Reviews", "Photos", "Bio"])

            with browse_sections[0]:
                _render_menu_browser(selected_menu_rows)

            with browse_sections[1]:
                _render_review_browser(selected_review_rows)

            with browse_sections[2]:
                _render_photo_browser(selected_image_rows)

            with browse_sections[3]:
                _render_bio_browser(selected_row.get("bio_text", ""))

            # Debug section: show raw source data behind the rendered content
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

    # ============================================================================
    # RECOMMENDATIONS TAB: MDN-based personalized ratings
    # ============================================================================
    with rec_tab:
        render_section_intro(
            "Taste Profile",
            "Personalized Recommendations",
            "Turn your ratings into a restaurant shortlist, with predicted affinity and uncertainty pulled from the MDN model.",
        )
        if not st.session_state.user_ratings:
            st.info("Please rate some restaurants to get personalized recommendations!")
        else:
            _, score_mdn_recommendations, mdn_error = _load_mdn_functions()
            st.write(f"You have rated **{len(st.session_state.user_ratings)}** restaurants.")
            recommendations_unavailable = score_mdn_recommendations is None
            
            # Generate recommendations using MDN model
            if score_mdn_recommendations is None:
                st.warning(mdn_error or "Recommendation features are temporarily unavailable.")
                rec_df = pd.DataFrame()
            else:
                with st.spinner("Generating MDN recommendations..."):
                    rec_df = score_mdn_recommendations(catalog, st.session_state.user_ratings)
            
            if recommendations_unavailable:
                pass
            elif rec_df.empty:
                st.warning("Could not generate recommendations.")
            else:
                # Display top 6 recommended restaurants with uncertainty visualization
                for idx, (_, result_row) in enumerate(rec_df.head(6).iterrows()):
                    score_label = "Predicted Rating"
                    if result_row.get("recommendation_mode") == "embedding_fallback":
                        score_label = "Recommendation Score"

                    _render_result_card(result_row.to_dict(), score_label, render_key=f"recommendation_{idx}")

                    # Show probability distribution from MDN if available
                    if "pdf_grid" in result_row and isinstance(result_row["pdf_grid"], (list, np.ndarray)):
                        st.caption("Rating Probability Distribution (HDR)")
                        chart_df = pd.DataFrame({"Probability Density": result_row["pdf_grid"]}, index=np.linspace(1.0, 5.0, 101))
                        st.area_chart(chart_df, height=150, color="#FF4B4B")

    # ============================================================================
    # MY RESTAURANTS TAB: User's rated restaurants
    # ============================================================================
    with my_rest_tab:
        st.subheader("My Rated Restaurants")
        if not st.session_state.user_ratings:
            st.info("You haven't rated any restaurants yet. Browse or search to add ratings!")
        else:
            # Filter catalog to show only restaurants user has rated
            my_df = filter_my_restaurants(catalog, st.session_state.user_ratings)
            
            st.write(f"You have rated **{len(my_df)}** restaurants.")
            for idx, (_, result_row) in enumerate(my_df.iterrows()):
                _render_result_card(result_row.to_dict(), "Your Rating", render_key=f"my_rated_{idx}")

    # ============================================================================
    # DATA OVERVIEW TAB: Catalog coverage and file health
    # ============================================================================
    with overview_tab:
        render_section_intro(
            "Coverage",
            "Data Overview",
            "Inspect how complete the current catalog is, including which source files are feeding the frontend and where the gaps still are.",
        )
        _render_data_overview(catalog)

if __name__ == "__main__":
    main()
