"""
ui_components/overview.py
=========================
Streamlit UI component for the data coverage overview panel.

Renders row counts and per-field coverage statistics for the current
restaurant catalog, giving operators a quick health-check of how much
of the dataset has menus, reviews, images, and bios.
"""
import pandas as pd
import streamlit as st
from core.data_loader import (
    load_lookup_df, 
    load_reviews_df, 
    load_images_df, 
    load_menus_df, 
    load_bios_df,
    LOOKUP_CSV,
    REVIEWS_CSV,
    IMAGES_CSV,
    MENUS_JSON,
    BIOS_JSON
)

def _render_data_overview(catalog: pd.DataFrame) -> None:
    """
    Render the data coverage overview panel into the current Streamlit context.

    Displays five top-level row-count metrics (restaurants, reviews, images,
    menu rows, bios) followed by a table of source file paths and a coverage
    breakdown showing how many restaurants have each data type.

    Parameters
    ----------
    catalog : pd.DataFrame
        The fully joined restaurant catalog from ``build_restaurant_catalog()``.
        Only used for the coverage breakdown; if empty the breakdown is skipped.
    """
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
