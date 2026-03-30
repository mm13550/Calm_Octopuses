# Calm Octopuses: Michelin NYC Data Project

This project is a comprehensive toolkit for collecting, analyzing, and exploring data and images related to Michelin-listed restaurants in New York City.

## Core Features & Scripts (v2.0 Architecture)

The project pipeline focuses on building a multimodal recommendation web application for Michelin-listed restaurants using official menus and low-friction User-Generated Content (UGC).

### 1. Data Pipelines (`pipelines/`)
- **`pipelines/resolve_homepages.py`**
  Automatically resolves official restaurant homepages from names using SerpAPI. Includes robust relevance scoring and `--resume` flags.
- **`pipelines/menu_crawler.py`**
  Crawls websites for HTML/PDF menus, leveraging OCR and LLM-assisted (VLM) extraction to handle complex menu formatting.
- **`pipelines/image_scrapper.py`**
  Fetches photos via Google Maps APIs, prioritizing food/dish shots. (Saves to `.gitignore`'d `data/images/`).
- **`pipelines/social_scraper.py`** [NEW v2.0]
  Lightweight crawler wrapping Reddit PRAW & Yelp APIs to fetch UGC images and reviews with strict timestamp limits.
- **`pipelines/text_cleaning.py`** [NEW v2.0]
  Utilizes LLMs to translate, run Aspect-Based Sentiment Analysis (ABSA) on UGC (food/service/ambiance), and clean unstructured reviews.
- **`pipelines/generate_embeddings.py`**
  Maps images and text into a shared Euclidean vector space using OpenAI CLIP (or SigLIP). Saves vectors into Parquet files or LanceDB.

### 2. Core Algorithms (`algorithms/`)
Decoupled mathematical logic utilizing Multimodal Style Feature Fusion:
  - `image_comparison.py`: Multi-modal vector math combining Cosine Similarity with a Time-Decay weight ($1 + \lambda e^{-\alpha \Delta t}$) to boost trending dishes.
  - `text_comparison.py`: Semantic text similarity logic natively mapping descriptions.
  - `dimensionality_reduction.py`: UMAP integration to preserve local topological structures of fused text+image vectors.
  - `clustering.py`: Applies Gaussian Mixture Models (GMM) on UMAP-reduced vectors to discover latent restaurant styles.
  - `quantile_regression.py`: Predicts "Risk/Confidence Intervals" for ratings by incorporating ABSA sentiment inputs (Service vs Food).

### 3. Applications & UI (`ui_components/`)
- **`app.py`**
  Streamlit GUI supporting multimodal cross-domain search (Text-to-Image / Image-to-Image) and latent style visualization.
- **`ui_components/image_grid.py`**
  Renders matrix maps with visual badges (e.g., "Trending") based on recent metadata and time-decay scores.

### 4. Testing Structure (`tests/`)
- A modular scaffolding directory containing foundational test suites:
  - `test_algorithms.py`: Initial test hooks validating the native mathematical abstractions.
  - `test_api.py`: Initial test hooks validating external Google Maps and SerpAPI data pipelines.

## Installation & Setup

1. **Virtual Environment**: 
   Ensure you use a virtual environment (`venv`).
   ```bash
   python -m venv venv
   # Windows Activation
   venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Keys**:
   Copy the included `.env.example` file to securely define your custom configuration variables, or expose the keys natively in your terminal:
   ```bash
   set SERPAPI_API_KEY=your_key_here
   set GOOGLE_MAPS_API_KEY=your_key_here
   ```

## Usage Examples

**Homepage Resolver:**
```bash
python pipelines/resolve_homepages.py --input data/nyc_michelin_names_cleaned.csv --output data/seeds_resolved.csv --delay 1.0
```

**Image Scraper:**
```bash
python pipelines/image_scrapper.py --limit 400
```

**Generate Content Embeddings:**
```bash
python pipelines/generate_embeddings.py
```

**Launch the Similarity App:**
```bash
streamlit run app.py
```

**Run Test Suites:**
```bash
pytest tests/
```

## Workflows & Agent Guidelines
The project enforces strict guidelines (see `.cursorrules`):
- All code changes must be tracked in version control and pushed with descriptive summaries.
- The virtual environment (`venv`) must always be respected.
- All code must be logically documented with docstrings and internal comments.
- Agent activities and architectural revisions are logged historically within `CHANGELOG.md`.
- **This `README.md` must be kept fully up to date with the structure of the project.**
