# Module B Handoff (Leo) — Backend Embeddings, Retrieval, and Profiles

## 1. Scope of Module B

This handoff covers the backend work completed for Module B.

My scope focused on:
- social image cleaning and categorization
- menu / review / image embeddings
- retrieval backend interfaces
- restaurant-level aggregation
- profile-level retrieval outputs for downstream use

This module does **not** cover the final frontend UI.  
Frontend / Streamlit integration should be handled later together with Merry.

---

## 2. Canonical Lookup and Join Rules

### Canonical metadata
Use:

- `data/csv/seeds_resolved.csv`

as the canonical restaurant metadata table.

### Stable join table
Use:

- `data/csv/restaurant_lookup.csv`

as the stable join table for:
- `name`
- `rest_id`

### Join rule
When joining menus / reviews / images / profiles, use:

- `rest_id`

as the primary key.

---

## 3. Main Input Files Used

The current backend pipeline uses these source files:

- `data/social_reviews.csv`
- `data/social_images.csv`
- `data/images/`
- `data/extracted_menus/final_parsed_menus.json`
- `data/csv/seeds_resolved.csv`
- `data/csv/restaurant_lookup.csv`

---

## 4. Files Implemented / Updated

### Pipelines
- `pipelines/generate_embeddings_michelin.py`
- `pipelines/generate_embeddings_reviews.py`
- `pipelines/clean_social_images.py`
- `pipelines/generate_embeddings_images.py`
- `pipelines/build_restaurant_profiles.py`

### Retrieval backends
- `algorithms/retrieval_engine.py`
- `algorithms/retrieval_engine_reviews.py`
- `algorithms/retrieval_engine_hybrid.py`
- `algorithms/retrieval_engine_profiles.py`

---

## 5. Outputs Produced

### Menu outputs
- `data/embeddings/menu_embeddings_latest.jsonl`
- `data/embeddings/restaurant_summary_latest.jsonl`

### Review outputs
- `data/embeddings/review_embeddings_latest.jsonl`

### Image outputs
- `data/social_images_cleaned.csv`
- `data/embeddings/image_embeddings_food_latest.jsonl`
- `data/embeddings/image_embeddings_interior_latest.jsonl`

### Restaurant-level outputs
- `data/embeddings/restaurant_profiles_latest.jsonl`

---

## 6. What Each Pipeline Does

### `clean_social_images.py`
Reads `data/social_images.csv` and produces:

- `data/social_images_cleaned.csv`

Main image categories used:
- `food`
- `interior`
- `people`
- `text_or_menu`
- `other_noise`

Important flags:
- `keep_for_food_embedding`
- `keep_for_ambiance_embedding`

### `generate_embeddings_images.py`
Uses `social_images_cleaned.csv` and creates:
- `image_embeddings_food_latest.jsonl`
- `image_embeddings_interior_latest.jsonl`

### `generate_embeddings_reviews.py`
Reads `data/social_reviews.csv` and creates:
- `review_embeddings_latest.jsonl`

Each review is embedded at **single-review level**.

### `generate_embeddings_michelin.py`
Reads:
- `data/extracted_menus/final_parsed_menus.json`

Creates:
- `menu_embeddings_latest.jsonl`
- `restaurant_summary_latest.jsonl`

Each menu item is embedded at **dish / menu-item level**.

### `build_restaurant_profiles.py`
Aggregates:
- menu embeddings
- review embeddings
- food image embeddings
- interior image embeddings

Creates:
- `restaurant_profiles_latest.jsonl`

Each row is **one restaurant profile**.

---

## 7. Current Retrieval Interfaces

### Menu retrieval
- `algorithms/retrieval_engine.py`

Useful for:
- dishes
- menu items
- omakase
- dessert
- price-related menu results

### Review retrieval
- `algorithms/retrieval_engine_reviews.py`

Useful for:
- service
- staff
- quiet atmosphere
- romantic / cozy
- user experience wording

### Hybrid retrieval
- `algorithms/retrieval_engine_hybrid.py`

Returns:
- `menu_results`
- `review_results`
- `restaurant_cards`

Useful for:
- combined menu + review matching

### Restaurant profile retrieval
- `algorithms/retrieval_engine_profiles.py`

Returns restaurant-level results with:
- `restaurant_name`
- score
- menu/review/image counts
- profile text
- top menu items
- top review snippets

This is the best current backend entry point for frontend card-style search.

---

## 8. Recommended Production Vector DB Layout

The backend currently works with JSONL outputs and retrieval scripts.  
For the formal next step, the vector DB should be split into separate logical tables instead of one mixed table.

Recommended split-table layout:

### `image_vectors_food`
Fields:
- `image_uid`
- `rest_id`
- `restaurant_name`
- `source`
- `image_path`
- `vector`
- `quality_score`

### `image_vectors_interior`
Fields:
- `image_uid`
- `rest_id`
- `restaurant_name`
- `source`
- `image_path`
- `vector`
- `quality_score`

### `menu_item_vectors`
Fields:
- `doc_id`
- `rest_id`
- `restaurant_name`
- `dish_name`
- `ingredients`
- `price`
- `text`
- `vector`

### `review_vectors`
Fields:
- `uid` or `doc_id`
- `rest_id`
- `restaurant_name`
- `source`
- `text`
- `rating`
- `vector`

### `restaurant_profiles`
Fields:
- `rest_id`
- `restaurant_name`
- `text`
- `vector`
- `menu_item_count`
- `review_count`
- `food_image_count`
- `interior_image_count`
- `top_menu_items`
- `top_review_snippets`

This split-table design is strongly recommended for the next formal storage step.

---

## 9. Current Status of the Frontend

`app.py` has **not** yet been upgraded to use the new restaurant-level retrieval stack.

The current frontend file is still the older image similarity explorer.

So for now:
- backend retrieval is complete enough for handoff
- frontend integration is still a later step

The best backend endpoint for frontend integration is:

- `search_profiles_api()` in `algorithms/retrieval_engine_profiles.py`

---

## 10. Known Gaps / Next Recommended Work

### Highest priority next steps
1. Connect frontend to `search_profiles_api()`
2. Use hybrid / menu / review retrieval as detail-level drill-down
3. Formalize split-table LanceDB storage
4. Refresh `README.md` to reflect the current backend state

### Lower-priority follow-up
- add ABSA-enriched profile metadata
- clustering over restaurant profiles
- quantile regression using restaurant-level features

---

## 11. Dependencies Note

If retrieval or storage is being reinstalled in a fresh environment, confirm that vector DB dependencies are included in `requirements.txt` (especially `lancedb`).

---

## 12. Git / Push Note

Do **not** push local data artifacts.

These should remain local:
- `data/images/`
- `data/social_images.csv`
- `data/social_reviews.csv`
- `data/embeddings/`
- local vector DB artifacts

Only push:
- code
- documentation
- dependency files

---

## 13. Summary

Module B backend work is complete enough to hand off.

Completed backend layers:
- image cleaning
- menu embeddings
- review embeddings
- food/interior image embeddings
- menu retrieval
- review retrieval
- hybrid retrieval
- restaurant-level profile aggregation
- restaurant-level profile retrieval

Recommended handoff target for the frontend:
- `algorithms/retrieval_engine_profiles.py`