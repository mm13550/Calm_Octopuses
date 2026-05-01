# Calm Octopuses

Calm Octopuses is a Streamlit application for exploring Michelin-listed New
York City restaurants. It combines restaurant profiles, menus, review snippets,
local image assets, CLIP embeddings, and Michelin award metadata into one
searchable catalog.

The current app supports:

- Text search across restaurant profiles, reviews, and menu items
- Image-based visual similarity search
- Exact dish/menu search
- Restaurant browsing with representative images and Michelin badges
- Lightweight personalized ranking from user ratings
- Data coverage checks in the UI

## 🚀 Quick Start (New Users)

This repository does not include the large generated assets needed for the full
demo. A fresh clone will not run end-to-end until those external assets are
downloaded or copied into `data/`.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Download the external asset bundle
python download_assets.py

# 4. Start the application
streamlit run frontend.py
```

## External Assets (Required)

Large menus, embeddings, images, and model checkpoints are intentionally kept
out of GitHub. Use one of these setup paths before grading or presenting:

1. Run `python download_assets.py` to fetch the standard asset bundle from the
   shared Hugging Face dataset mirror.
2. Copy the same files from the team's shared asset drive into the matching
   paths under `data/`.

At minimum, the app expects these external files or directories to exist
locally:

- `data/extracted_menus/final_parsed_menus.json`
- `data/embeddings/restaurant_profiles.jsonl`
- `data/embeddings/menu_embeddings.jsonl`
- `data/embeddings/review_embeddings.jsonl`
- `data/embeddings/image_embeddings_food.jsonl`
- `data/embeddings/image_embeddings_interior.jsonl`
- `data/embeddings/restaurant_metadata.json`
- `data/images/`

The recommendation tab can also use:

- `data/yelp_sandbox/mdn_models/clip_v2/clip_v2_full.ckpt`

If the MDN checkpoint is missing, the app falls back to embedding-based
recommendations instead of crashing.

## Team & Division of Labor

- **Neil (Module A):** Data operations and scraping, including Google Places/Apify collection, Yelp dataset utilities, menu crawling, image collection, and source-data cleanup.
- **Leo (Module B):** Embedding and retrieval work, including CLIP-based text/image search, JSONL embedding artifacts, similarity scoring, and embedding maintenance utilities.
- **Craig (Module C):** Advanced ML algorithms, including MDN-based personalized rating recommendations, rating uncertainty outputs, Yelp sandbox evaluation, and embedding-based fallback recommendation logic.
- **Merry (Module D):** Streamlit frontend development, including app navigation, multimodal search inputs, exact dish search, restaurant cards, browse/rating/recommendation flows, data overview panels, and dynamic Michelin/sentiment UI badges.
- **Grace (Module E):** NLP and system integration, including ABSA-derived review sentiment fields, aspect score schema alignment, and synchronization of sentiment/profile data consumed by the frontend.

## Core Architecture & Directory Structure

The project is organized around a Streamlit frontend, a shared data-loading layer, local source/embedding artifacts, repeatable data pipelines, and active retrieval/recommendation algorithms. The current app path is local-file based: `frontend.py` calls `core/data_loader.py`, `algorithms/retrieval.py`, and `algorithms/mdn_regression.py`; those modules read tracked source data plus local generated artifacts under `data/`.

### 1. Application Layer (`frontend.py`, `ui_components/`)

- **`frontend.py` (Merry):** Main Streamlit entry point. It builds the Search, Dish Search, Browse Restaurants, Recommended, and Data Overview tabs, and wires retrieval, rating input, recommendation scoring, and catalog inspection into one app flow.
- **`ui_components/cards.py`, `theme.py`, and `overview.py` (Merry):** Reusable UI pieces for restaurant cards, Michelin badges, review sentiment charts, global styling, section headers, and data coverage reporting.
- **`ui_components/image_grid.py` (Merry):** Optional helper for image similarity grids and visual-search experiments.

### 2. Data Layer (`core/`, `data/`)

- **`core/data_loader.py` and `core/logic.py` (Merry + Leo):** The central path, schema, and app-logic layer. These files load tracked CSV/JSON data, read local embedding JSONL files, join menus/reviews/images/bios/Michelin metadata, and expose catalog/detail helpers used by the app.
- **Tracked source data (Neil + Grace + Leo):** `data/csv/restaurant_lookup.csv`, `data/csv/nyc_michelin_awards.xlsx`, `data/csv/nyc_michelin_names_cleaned.csv`, `data/csv/seeds_resolved.csv`, `data/csv/restaurant_profiles.csv`, `data/csv/social_reviews.csv`, `data/csv/social_images.csv`, and `data/extracted_bios/restaurant_bios_joinable.json`.
- **Local generated artifacts (Neil + Leo):** `data/embeddings/*.jsonl`, `data/images/`, `data/extracted_menus/final_parsed_menus.json`, and `data/vector_db/` are generated locally, downloaded from the shared asset bundle, and intentionally kept out of Git.

### 3. Data and Embedding Pipelines (`pipelines/`)

- **Collection and crawling (Neil):** `social_scraper.py`, `menu_crawler.py`, `bio_crawler.py`, `resolve_homepages.py`, `fetch_and_embed_reviews.py`, and `clean_social_images.py` collect and normalize reviews, images, biographies, homepages, and menu source records.
- **Text cleaning and ABSA support (Grace):** Grace's NLP work provides the aspect-based sentiment fields consumed by the frontend, including food quality, service, ambiance, value, and wait time values synchronized into `data/csv/restaurant_profiles.csv`.
- **Menu maintenance (Neil):** `menu_pipeline.py` consolidates legacy menu tasks. It can merge parsed/retried menu exports into `data/extracted_menus/final_parsed_menus.json` and retry restaurants with zero dishes.
- **Embedding artifact layer (Leo):** `embeddings_pipeline.py` is the unified embedding entry point for menu, review, and image JSONL artifacts. `build_restaurant_profiles.py` generates fused restaurant-profile vectors consumed by retrieval and recommendation code.
- **Embedding maintenance (Leo):** `check_embedding_integrity.py` and `maintenance.py` validate, deduplicate, audit, refresh, and realign generated artifacts when source records change.
- **Yelp sandbox (`pipelines/yelp/`) (Neil + Craig):** `download_yelp_dataset.py`, `preprocess_yelp.py`, `generate_embeddings_yelp.py`, `aggregate_restaurant_embeddings.py`, `export_regression_train.py`, and `evaluate_generalization.py` support Yelp Open Dataset experiments, restaurant-level embedding aggregation, training exports, and generalization evaluation outside the main app path.

### 4. Algorithms and Retrieval (`algorithms/`)

- **`algorithms/retrieval.py` (Leo):** Active frontend retrieval path. It embeds text/image queries with CLIP, loads local JSONL embedding artifacts, computes cosine similarity, blends text search with lexical overlap, and supports exact dish matching.
- **`algorithms/mdn_regression.py` (Craig):** Personalized recommendation module. It uses trained MDN artifacts when available, produces rating predictions and uncertainty-style PDF outputs, and falls back to embedding-based scoring when model checkpoints are missing.

### 5. Tests and Diagnostics (`tests/`)

- **`tests/test_algorithms.py` and `tests/test_api.py` (Leo + Craig + Merry):** Automated checks for retrieval helper determinism and shared data-loader utilities.
- Diagnostics currently live in the main app's Data Overview tab and in targeted pipeline checks such as `pipelines/check_embedding_integrity.py`; there is no tracked `tools/` directory in the current repository.

### Directory Overview

This tree reflects the current project files in the repository, excluding `.git/` internals and ignored local generated artifacts.

```text
.
|-- .cursorrules
|-- .env.example
|-- .gitignore
|-- activate.bat
|-- activate.ps1
|-- CHANGELOG.md
|-- PLAN.md
|-- README.md
|-- REGRESSION_TUNING.md
|-- algorithms/                 # Retrieval, recommendation, and similarity logic
|   |-- __init__.py
|   |-- mdn_regression.py
|   `-- retrieval.py
|-- core/                       # Shared data loading, catalog assembly, and app logic
|   |-- data_loader.py
|   `-- logic.py
|-- data/                       # Tracked source data plus ignored local artifacts
|   |-- README.md
|   |-- csv/
|   |   |-- nyc_michelin_awards.xlsx
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
|   |-- bio_crawler.py
|   |-- build_restaurant_profiles.py
|   |-- check_embedding_integrity.py
|   |-- clean_social_images.py
|   |-- embeddings_pipeline.py
|   |-- fetch_and_embed_reviews.py
|   |-- menu_crawler.py
|   |-- maintenance.py
|   |-- menu_pipeline.py
|   |-- resolve_homepages.py
|   |-- social_scraper.py
|   `-- yelp/                   # Yelp sandbox preprocessing and embedding experiments
|       |-- __init__.py
|       |-- aggregate_restaurant_embeddings.py
|       |-- download_yelp_dataset.py
|       |-- evaluate_generalization.py
|       |-- export_regression_train.py
|       |-- generate_embeddings_yelp.py
|       `-- preprocess_yelp.py
|-- pyproject.toml              # Project metadata and pytest configuration
|-- requirements.txt            # Runtime dependencies
|-- tests/                      # Automated tests
|   |-- __init__.py
|   |-- test_algorithms.py
|   `-- test_api.py
`-- ui_components/              # Streamlit UI components and theme
    |-- __init__.py
    |-- cards.py
    |-- image_grid.py
    |-- overview.py
    `-- theme.py
```

Large generated assets are intentionally ignored by Git. Keep local
embeddings, image downloads, vector databases, logs, and cache files out of
commits.

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

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the standard external asset bundle:

```bash
python download_assets.py
```

Create a local `.env` file from the example when running collection pipelines:

```bash
cp .env.example .env  # macOS / Linux
# or: Copy-Item .env.example .env  # Windows PowerShell
```

The main app can run without API keys if the required local data files already exist.

## Required Local Data

The app expects a small set of tracked CSV/JSON files and several larger local
artifacts:

Tracked or intentionally small:

- `data/csv/restaurant_lookup.csv`
- `data/csv/nyc_michelin_awards.xlsx`
- `data/csv/nyc_michelin_names_cleaned.csv`
- `data/csv/seeds_resolved.csv`
- `data/csv/restaurant_profiles.csv`
- `data/csv/social_reviews.csv`
- `data/csv/social_images.csv`
- `data/extracted_bios/restaurant_bios_joinable.json`

Generated, downloaded, or local-only:

- `data/extracted_menus/final_parsed_menus.json`
- `data/embeddings/restaurant_profiles.jsonl`
- `data/embeddings/menu_embeddings.jsonl`
- `data/embeddings/review_embeddings.jsonl`
- `data/embeddings/image_embeddings_food.jsonl`
- `data/embeddings/image_embeddings_interior.jsonl`
- `data/embeddings/restaurant_metadata.json`
- `data/images/`
- `data/vector_db/`
- `data/yelp_sandbox/mdn_models/clip_v2/clip_v2_full.ckpt` (optional; used by the MDN path)

See `data/README.md` for the data policy.

## Run

Start the application:

```bash
streamlit run frontend.py
```

Then open:

```text
http://localhost:8501
```

## Test

Run the automated checks:

```bash
python -m pytest
```

For a quick syntax check:

```bash
python -m py_compile frontend.py core/data_loader.py ui_components/cards.py ui_components/theme.py
```

## Development Notes

- Keep `frontend.py` focused on app flow and interactions.
- Put reusable UI rendering in `ui_components/`.
- Put disk access and schema normalization in `core/data_loader.py`.
- Put retrieval, scoring, and model logic in `algorithms/`.
- Do not commit generated images, embeddings, logs, cache folders, local databases, or secret files.
- Prefer adding small deterministic tests when changing shared logic.
