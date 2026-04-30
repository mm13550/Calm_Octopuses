# Calm Octopuses Project Plan

This document reflects the current repository state and the remaining work needed to keep the Streamlit Michelin NYC restaurant explorer reproducible.

## 1. Current Objective

Calm Octopuses is a local Streamlit application for exploring Michelin-listed New York City restaurants. The current product path combines tracked restaurant metadata with local generated artifacts:

- Restaurant lookup and Michelin award metadata
- Parsed restaurant biographies
- Local menu, review, image, and embedding artifacts
- CLIP-based text and image retrieval
- MDN-based personalized recommendation scoring with an embedding fallback
- Streamlit cards, browsing, search, recommendation, and data-overview UI

The current main entry point is:

```powershell
streamlit run frontend.py
```

## 2. Current Architecture

### Application and UI

- `frontend.py` is the primary Streamlit application.
- `ui_components/cards.py` renders restaurant result cards, Michelin badges, menu/review snippets, and review sentiment panels.
- `ui_components/theme.py` owns the app styling and reusable header/section rendering helpers.
- `ui_components/overview.py` renders data coverage checks.
- `ui_components/image_grid.py` is an optional helper for visual-search grid experiments.

### Data Loading

- `core/data_loader.py` is the canonical path and schema layer.
- It loads tracked CSV/JSON data, local generated menu/review/image artifacts, and embedding JSONL files.
- It assembles the wide restaurant catalog consumed by `frontend.py`, `algorithms/retrieval.py`, and `algorithms/mdn_regression.py`.

### Retrieval and Recommendation

- `algorithms/retrieval.py` is the active frontend retrieval path.
- It supports CLIP text query embedding, image query embedding, cosine similarity scoring, lexical overlap blending, menu-item scoped search, review scoped search, and exact dish matching.
- `algorithms/mdn_regression.py` provides personalized rating predictions when an MDN checkpoint is available.
- When the MDN checkpoint is missing, the recommendation flow falls back to embedding-based scoring.
- `quarry/` contains LanceDB-style retrieval prototypes. These are useful for future vector database work but are not the active frontend path.

### Data and Embedding Pipelines

- `pipelines/social_scraper.py`, `pipelines/menu_crawler.py`, `pipelines/bio_crawler.py`, `pipelines/resolve_homepages.py`, `pipelines/fetch_and_embed_reviews.py`, and `pipelines/clean_social_images.py` collect and normalize source records.
- Grace's NLP and ABSA work supplies the aspect sentiment fields consumed by the frontend, with food quality, service, ambiance, value, and wait time data synchronized into `data/csv/restaurant_profiles.csv`.
- `pipelines/menu_pipeline.py` consolidates menu merge and zero-dish retry maintenance tasks.
- `pipelines/embeddings_pipeline.py` is the unified entry point for menu, review, and image embedding generation.
- `pipelines/build_restaurant_profiles.py` generates local restaurant-profile embedding artifacts.
- `pipelines/check_embedding_integrity.py` and `pipelines/maintenance.py` maintain, audit, deduplicate, refresh, and realign generated artifacts after source changes.
- `pipelines/yelp/` contains sandbox scripts for Yelp Open Dataset experiments and regression export support.

### Tests and Diagnostics

- `tests/test_algorithms.py` covers GMM clustering behavior and retrieval helper determinism.
- `tests/test_api.py` covers shared data-loader utilities.
- Diagnostics currently live in the main app's Data Overview tab and targeted pipeline checks such as `pipelines/check_embedding_integrity.py`.

## 3. Data Policy

Tracked source data is intentionally small:

- `data/csv/restaurant_lookup.csv`
- `data/csv/nyc_michelin_awards.xlsx`
- `data/csv/nyc_michelin_names_cleaned.csv`
- `data/csv/seeds_resolved.csv`
- `data/csv/restaurant_profiles.csv`
- `data/csv/social_reviews.csv`
- `data/csv/social_images.csv`
- `data/extracted_bios/restaurant_bios_joinable.json`

Generated or large artifacts stay local and are ignored by Git:

- `data/embeddings/*.jsonl`
- `data/images/*`
- `data/extracted_menus/final_parsed_menus.json`
- `data/vector_db/`
- `data/yelp_sandbox/`

## 4. Remaining Work

### Short-Term Cleanup

- Keep README, PLAN, CHANGELOG, and `data/README.md` synchronized when files move.
- Keep generated artifacts out of Git and update `.gitignore` when new local output paths are introduced.
- Add small deterministic tests when changing shared utilities in `core/`, `algorithms/`, or `ui_components/`.

### Retrieval and Embeddings

- Validate that every generated embedding JSONL file has matching restaurant IDs against `data/csv/restaurant_lookup.csv`.
- Keep `core/data_loader.py` fallback behavior compatible with both canonical and `*_latest.jsonl` embedding exports.
- Decide whether the LanceDB prototypes in `quarry/` should become the active retrieval backend or remain experimental.

### Recommendation and Clustering

- Keep `algorithms/mdn_regression.py` robust when the MDN checkpoint is unavailable.
- Expand clustering tests when adding new clustering or dimensionality-reduction utilities.
- Document any trained model checkpoints or Yelp sandbox outputs in `data/README.md` before sharing them outside a local environment.

### Frontend

- Keep `frontend.py` focused on app flow and state.
- Move reusable display logic into `ui_components/`.
- Preserve data coverage and diagnostics surfaces so missing local artifacts are easy to understand.

## 5. Validation Commands

Run automated checks:

```powershell
python -m pytest
```

Run a quick syntax check:

```powershell
python -m py_compile frontend.py core\data_loader.py ui_components\cards.py ui_components\theme.py
```

Run the app:

```powershell
streamlit run frontend.py
```
