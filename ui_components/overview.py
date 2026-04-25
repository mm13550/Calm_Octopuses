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
