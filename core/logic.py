"""
core/logic.py
=============
Business logic and data orchestration layer.

This module decouples the presentation layer (frontend.py) from the 
data access layer (core.data_loader.py). It handles:
1. Cross-table joining and filtering for specific restaurant details.
2. User state management for rated restaurants.
3. High-level result set manipulation for the UI.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from core.data_loader import (
    load_menus_df,
    load_reviews_df,
    load_images_df,
    load_lookup_df,
    _clean_text
)

def get_restaurant_details(rest_id: str) -> Dict[str, pd.DataFrame]:
    """
    Retrieve all data slices for a specific restaurant ID.
    
    Returns a dictionary containing DataFrames for:
    - menus
    - reviews
    - images
    - lookup
    """
    menus_df = load_menus_df()
    reviews_df = load_reviews_df()
    images_df = load_images_df()
    lookup_df = load_lookup_df()
    
    details = {
        "menus": (
            menus_df[menus_df["rest_id"] == rest_id].copy()
            if not menus_df.empty and "rest_id" in menus_df.columns
            else pd.DataFrame()
        ),
        "reviews": (
            reviews_df[reviews_df["rest_id"] == rest_id].copy()
            if not reviews_df.empty and "rest_id" in reviews_df.columns
            else pd.DataFrame()
        ),
        "images": (
            images_df[images_df["rest_id"] == rest_id].copy()
            if not images_df.empty and "rest_id" in images_df.columns
            else pd.DataFrame()
        ),
        "lookup": (
            lookup_df[lookup_df["rest_id"] == rest_id].copy()
            if not lookup_df.empty and "rest_id" in lookup_df.columns
            else pd.DataFrame()
        )
    }
    
    return details

def filter_my_restaurants(catalog: pd.DataFrame, user_ratings: Dict[str, float]) -> pd.DataFrame:
    """Filter and sort the catalog based on user ratings."""
    if not user_ratings:
        return pd.DataFrame()
        
    rated_ids = list(user_ratings.keys())
    my_df = catalog[catalog["rest_id"].isin(rated_ids)].copy()
    my_df["actual_rating"] = my_df["rest_id"].map(user_ratings)
    return my_df.sort_values(by="actual_rating", ascending=False)
