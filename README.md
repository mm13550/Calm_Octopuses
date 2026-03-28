# Calm Octopuses: Michelin NYC Data Project

This project is a comprehensive toolkit for collecting, analyzing, and exploring data and images related to Michelin-listed restaurants in New York City.

## Core Features & Scripts

The project pipeline covers homepage resolution, menu crawling, image scraping from Google Maps, and visual similarity exploration.

### 1. Data Pipelines (`pipelines/`)
- **`pipelines/resolve_homepages.py`**
  Automatically resolves official restaurant homepages from names using SerpAPI. Includes robust relevance scoring, rate-limit protections (HTTP 429), and a smooth `--resume` flag to safely pause and continue execution without losing data.
- **`pipelines/menu_crawler.py`**
  Crawls restaurant websites to fetch HTML pages and PDF files, extracting and structuring menu content.
- **`pipelines/image_scrapper.py`**
  Fetches and downloads restaurant photos using the Google Maps API, intelligently prioritizing food/dish photos while avoiding standard exterior shots. Note: Images are saved to the `data/images/` directory which is automatically `.gitignore`'d.
- **`pipelines/generate_embeddings.py`**
  Utilizes the OpenAI CLIP (`clip-vit-base-patch32`) model to parse the downloaded images and generate normalized semantic feature vectors. Results are securely saved as `data/embeddings/image_embeddings.parquet`.

### 2. Core Algorithms (`algorithms/`)
A dedicated package containing mathematical and analytical logic decoupled from the UI:
  - `image_comparison.py`: Handles vector math like dot products for cosine similarity.
  - `text_comparison.py`: Structural stubs for semantic text similarity.
  - `dimensionality_reduction.py`: Structural stubs for mapping high-dimensional spaces (e.g., Autoencoder).
  - `clustering.py`: Structural stubs for unsupervised grouping (e.g., Gaussian Mixture Models).
  - `quantile_regression.py`: Structural stubs for analyzing conditional subsets and variance.

### 3. Applications & UI (`ui_components/`)
- **`app.py`**
  The central Streamlit-based graphical user interface (GUI) designed to explore image embeddings. Select any scraped image and visually discover the top `N` most similar images across your dataset using cosine similarities.
- **`ui_components/`**
  A dedicated module for rendering standalone layout abstractions (e.g., `image_grid.py`). This cleanly decouples complex view-rendering logic from the primary `app.py` controller.

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
