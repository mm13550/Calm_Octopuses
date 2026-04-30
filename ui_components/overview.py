"""
ui_components/overview.py
=========================
Data health and coverage dashboard for the Calm Octopuses catalog.

Provides `_render_data_overview`, which calculates and displays:
1. Total row counts across all major data tables.
2. Per-field coverage percentages (how many restaurants have menus, images, etc.).
3. Status indicators for the underlying CSV and JSON source files.
"""
import pandas as pd
import streamlit as st
from core.data_loader import (
    load_lookup_df, 
    load_reviews_df, 
    load_images_df, 
    load_menus_df, 
    load_bios_df,
    load_michelin_awards_df,
    LOOKUP_CSV,
    MICHELIN_AWARDS_CSV,
    MICHELIN_AWARDS_XLSX,
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
    awards_df = load_michelin_awards_df()

    st.markdown("<p class='co-note'>Current data footprint feeding the frontend.</p>", unsafe_allow_html=True)
    cols = st.columns(6)
    cols[0].metric("Restaurants", len(lookup_df))
    cols[1].metric("Reviews", len(reviews_df))
    cols[2].metric("Images", len(images_df))
    cols[3].metric("Menu rows", len(menus_df))
    cols[4].metric("Bios", len(bios_df))
    cols[5].metric("Michelin awards", len(awards_df))

    st.markdown("### Source files")
    awards_path = MICHELIN_AWARDS_CSV if MICHELIN_AWARDS_CSV.exists() else MICHELIN_AWARDS_XLSX
    file_rows = [
        (str(LOOKUP_CSV), len(lookup_df)),
        (str(awards_path), len(awards_df)),
        (str(REVIEWS_CSV), len(reviews_df)),
        (str(IMAGES_CSV), len(images_df)),
        (str(MENUS_JSON), len(menus_df)),
        (str(BIOS_JSON), len(bios_df)),
    ]
    st.dataframe(pd.DataFrame(file_rows, columns=["file", "rows"]), width="stretch", hide_index=True)

    if not catalog.empty:
        st.markdown("### Coverage summary")
        coverage_df = pd.DataFrame(
            [
                {"field": "Restaurants with menus", "restaurants": int(catalog["has_menu"].sum())},
                {"field": "Restaurants with reviews", "restaurants": int(catalog["has_reviews"].sum())},
                {"field": "Restaurants with food images", "restaurants": int(catalog["has_food_images"].sum())},
                {"field": "Restaurants with bios", "restaurants": int(catalog["bio_text"].astype(bool).sum())},
            ]
        )
        st.dataframe(coverage_df, width="stretch", hide_index=True)
