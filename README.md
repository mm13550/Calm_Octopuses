# Calm Octopuses

Calm Octopuses is a Streamlit application for exploring Michelin-listed New York City restaurants. It combines restaurant profiles, menus, review snippets, local image assets, CLIP embeddings, and Michelin award metadata into one searchable catalog.

The current app supports:

- Text search across restaurant profiles, reviews, and menu items
- Image-based visual similarity search
- Exact dish/menu search
- Restaurant browsing with representative images and Michelin badges
- Lightweight personalized ranking from user ratings
- Data coverage checks in the UI

## Team & Division of Labor

- **Neil (Module A):** Data operations and scraping, including Google Places/Apify collection, Yelp dataset utilities, menu crawling, image collection, and source-data cleanup.
- **Leo (Module B):** Embedding and retrieval work, including CLIP-based text/image search, JSONL embedding artifacts, similarity scoring, and experimental LanceDB retrieval prototypes under `quarry/`.
- **Craig (Module C):** Advanced ML algorithms, including Gaussian Mixture clustering, dimensionality-reduction hooks, MDN-based personalized rating recommendations, rating uncertainty outputs, and embedding-based fallback recommendation logic.
- **Merry (Module D):** Streamlit frontend development, including app navigation, multimodal search inputs, exact dish search, restaurant cards, browse/rating/recommendation flows, data overview panels, and dynamic Michelin/sentiment UI badges.
- **Grace (Module E):** NLP and system integration, including ABSA-derived review sentiment fields, aspect score schema alignment, and synchronization of sentiment/profile data consumed by the frontend.

## Core Architecture & Directory Structure

The project is organized around a Streamlit frontend, a shared data-loading layer, local source/embedding artifacts, repeatable data pipelines, active retrieval/recommendation algorithms, and experimental LanceDB-style retrieval prototypes. The current app path is local-file based: `frontend.py` calls `core/data_loader.py`, `algorithms/retrieval.py`, and `algorithms/mdn_regression.py`; those modules read tracked source data plus local generated artifacts under `data/`.

### 1. Application Layer (`frontend.py`, `ui_components/`)

- **`frontend.py` (Merry):** Main Streamlit entry point. It builds the Search, Dish Search, Browse Restaurants, Recommended, and Data Overview tabs, and wires retrieval, rating input, recommendation scoring, and catalog inspection into one app flow.
- **`ui_components/cards.py`, `theme.py`, and `overview.py` (Merry):** Reusable UI pieces for restaurant cards, Michelin badges, review sentiment charts, global styling, section headers, and data coverage reporting.
- **`ui_components/image_grid.py` (Merry):** Optional helper for image similarity grids and visual-search experiments.

### 2. Data Layer (`core/`, `data/`)

- **`core/data_loader.py` and `core/logic.py` (Merry + Leo):** The central path, schema, and app-logic layer. These files load tracked CSV/JSON data, read local embedding JSONL files, join menus/reviews/images/bios/Michelin metadata, and expose catalog/detail helpers used by the app.
- **Tracked source data (Neil + Grace + Leo):** `data/csv/restaurant_lookup.csv`, `data/csv/michelin_awards.csv`, `data/csv/nyc_michelin_names_cleaned.csv`, `data/csv/seeds_resolved.csv`, `data/csv/restaurant_profiles.csv`, `data/csv/social_reviews.csv`, `data/csv/social_images.csv`, and `data/extracted_bios/restaurant_bios_joinable.json`.
- **Local generated artifacts (Neil + Leo):** `data/embeddings/*.jsonl`, `data/images/`, `data/extracted_menus/final_parsed_menus.json`, and `data/vector_db/` are generated locally and intentionally kept out of Git.

### 3. Data and Embedding Pipelines (`pipelines/`)

- **Collection and cleaning (Neil):** `social_scraper.py`, `menu_crawler.py`, `merge_menus.py`, `clean_social_images.py`, `remap_problematic_rest_ids.py`, and related scripts collect reviews/images/menus and clean source records before embedding generation.
- **Text cleaning and ABSA support (Grace):** `text_cleaning.py` documents the text-cleaning and aspect-based sentiment analysis path, with ABSA-derived fields such as food quality, service, ambiance, value, and wait time synchronized into `data/csv/restaurant_profiles.csv` for frontend display.
- **Embedding artifact layer (Leo):** `generate_embeddings_images.py`, `generate_embeddings_reviews.py`, `generate_embeddings_michelin.py`, and `build_restaurant_profiles.py` generate the image, review, menu, summary, and fused restaurant-profile vectors consumed by retrieval and recommendation code.
- **Embedding maintenance (Leo):** `check_embedding_integrity.py`, `clean_restaurant_profiles.py`, `refresh_social_derived_artifacts.py`, and `refresh_review_profile_artifacts.py` validate, deduplicate, refresh, and realign embedding-derived artifacts when source records change.
- **Yelp sandbox (`pipelines/yelp/`) (Neil + Craig):** `download_yelp_dataset.py`, `preprocess_yelp.py`, `generate_embeddings_yelp.py`, `aggregate_restaurant_embeddings.py`, `export_regression_train.py`, and `evaluate_generalization.py` support Yelp Open Dataset experiments, restaurant-level embedding aggregation, training exports, and generalization evaluation outside the main app path.

### 4. Algorithms and Retrieval (`algorithms/`, `quarry/`)

- **`algorithms/retrieval.py` (Leo):** Active frontend retrieval path. It embeds text/image queries with CLIP, loads local JSONL embedding artifacts, computes cosine similarity, blends text search with lexical overlap, and supports exact dish matching.
- **`algorithms/mdn_regression.py` (Craig):** Personalized recommendation module. It uses trained MDN artifacts when available, produces rating predictions and uncertainty-style PDF outputs, and falls back to embedding-based scoring when model checkpoints are missing.
- **`algorithms/clustering.py` (Craig):** Latent-style clustering utility. The active clustering utility fits Gaussian Mixture Models and returns labels, soft cluster probabilities, BIC, and AIC diagnostics for restaurant embedding analysis.
- **`quarry/retrieval_engine*.py` (Leo):** Experimental LanceDB-style retrieval prototypes for menu, review, profile, and hybrid search. These are not the current Streamlit app path, but they document the vector database direction and metadata-rich retrieval API experiments.

### 5. Tests and Diagnostics (`tests/`, `tools/`)

- **`tests/test_algorithms.py` and `tests/test_api.py` (Leo + Craig + Merry):** Automated checks for clustering behavior, retrieval helper determinism, and shared data-loader utilities.
- **`tools/diagnostics_app.py` (Merry + Leo):** Optional local Streamlit diagnostics app for inspecting data and model artifacts outside the main product UI.

### Directory Overview

This tree reflects the current project files in the repository, excluding `.git/` internals and ignored local generated artifacts.

```text
.
|-- .cursorrules
|-- .env.example
|-- .gitignore
|-- CHANGELOG.md
|-- PLAN.md
|-- README.md
|-- REGRESSION_TUNING.md
|-- algorithms/                 # Retrieval, recommendation, clustering, and similarity logic
|   |-- __init__.py
|   |-- clustering.py
|   |-- mdn_regression.py
|   `-- retrieval.py
|-- core/                       # Shared data loading, catalog assembly, and app logic
|   |-- data_loader.py
|   `-- logic.py
|-- data/                       # Tracked source data plus ignored local artifacts
|   |-- README.md
|   |-- csv/
|   |   |-- michelin_awards.csv
|   |   |-- nyc_michelin_names_cleaned.csv
|   |   |-- restaurant_profiles.csv
|   |   |-- restaurant_lookup.csv
|   |   |-- seeds_resolved.csv
|   |   |-- social_images.csv
|   |   `-- social_reviews.csv
|   `-- extracted_bios/
|       `-- restaurant_bios_joinable.json
|-- frontend.py                 # Main Streamlit app entry point
|-- pipelines/                  # Data collection, cleaning, embedding, and refresh scripts
|   |-- __init__.py
|   |-- apply_social_rest_id_mapping.py
|   |-- audit_restaurant_lookup.py
|   |-- bio_crawler.py
|   |-- generate_embeddings_images.py
|   |-- generate_embeddings_reviews.py
|   |-- generate_embeddings_michelin.py
|   |-- build_restaurant_profiles.py
|   |-- check_embedding_integrity.py
|   |-- clean_restaurant_profiles.py
|   |-- clean_social_images.py
|   |-- dedupe_image_records.py
|   |-- dedupe_restaurant_summaries.py
|   |-- dedupe_review_records.py
|   |-- fetch_and_embed_reviews.py
|   |-- menu_crawler.py
|   |-- merge_menus.py
|   |-- refetch_missing_yelp_images.py
|   |-- refresh_review_profile_artifacts.py
|   |-- refresh_social_derived_artifacts.py
|   |-- remap_problematic_rest_ids.py
|   |-- rerun_likely_problematic_restaurants.py
|   |-- resolve_homepages.py
|   |-- retry_zero_dishes.py
|   |-- social_scraper.py
|   |-- text_cleaning.py
|   `-- yelp/                   # Yelp sandbox preprocessing and embedding experiments
|       |-- __init__.py
|       |-- aggregate_restaurant_embeddings.py
|       |-- download_yelp_dataset.py
|       |-- evaluate_generalization.py
|       |-- export_regression_train.py
|       |-- generate_embeddings_yelp.py
|       `-- preprocess_yelp.py
|-- pyproject.toml              # Project metadata and pytest configuration
|-- quarry/                     # LanceDB-style retrieval prototypes
|   |-- retrieval_engine.py
|   |-- retrieval_engine_hybrid.py
|   |-- retrieval_engine_profiles.py
|   |-- retrieval_engine_reviews.py
|   `-- retrieval_metadata.py
|-- requirements.txt            # Runtime dependencies
|-- tests/                      # Automated tests
|   |-- __init__.py
|   |-- test_algorithms.py
|   `-- test_api.py
|-- tools/                      # Optional diagnostics utilities
|   |-- README.md
|   `-- diagnostics_app.py
`-- ui_components/              # Streamlit UI components and theme
    |-- __init__.py
    |-- cards.py
    |-- image_grid.py
    |-- overview.py
    `-- theme.py
```

Large generated assets are intentionally ignored by Git. Keep local embeddings, image downloads, vector databases, logs, and cache files out of commits.

The main embedding artifact flow is:

```text
raw images/reviews/menus
  -> image/review/menu embedding JSONL files
  -> restaurant profile vectors
  -> core/data_loader.py
  -> algorithms/retrieval.py and algorithms/mdn_regression.py
  -> frontend.py
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local `.env` file from the example when running collection pipelines:

```powershell
Copy-Item .env.example .env
```

The main app can run without API keys if the required local data files already exist.

## Required Local Data

The app expects a small set of tracked CSV/JSON files and several larger ignored artifacts:

Tracked or intentionally small:

- `data/csv/restaurant_lookup.csv`
- `data/csv/michelin_awards.csv`
- `data/csv/nyc_michelin_names_cleaned.csv`
- `data/csv/seeds_resolved.csv`
- `data/csv/restaurant_profiles.csv`
- `data/csv/social_reviews.csv`
- `data/csv/social_images.csv`
- `data/extracted_bios/restaurant_bios_joinable.json`

Generated or local-only:

- `data/embeddings/*.jsonl`
- `data/images/*`
- `data/extracted_menus/final_parsed_menus.json`
- `data/vector_db/`

See `data/README.md` for the data policy.

## Run

Start the application:

```powershell
streamlit run frontend.py
```

Then open:

```text
http://localhost:8501
```

Optional legacy diagnostics live in:

```powershell
streamlit run tools/diagnostics_app.py
```

## Test

Run the automated checks:

```powershell
python -m pytest
```

For a quick syntax check:

```powershell
python -m py_compile frontend.py core\data_loader.py ui_components\cards.py ui_components\theme.py
```

## Development Notes

- Keep `frontend.py` focused on app flow and interactions.
- Put reusable UI rendering in `ui_components/`.
- Put disk access and schema normalization in `core/data_loader.py`.
- Put scoring, clustering, and model logic in `algorithms/`.
- Do not commit generated images, embeddings, logs, cache folders, local databases, or secret files.
- Prefer adding small deterministic tests when changing shared logic.
