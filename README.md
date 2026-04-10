# Calm Octopuses: Michelin NYC Dining Recommender (v3.0)

This project is a comprehensive, multimodal recommendation web application for Michelin-listed restaurants in New York City. Moving beyond traditional coarse-grained tags (e.g., cuisine, price), our engine enables precise dish-level searches using **images or natural language**. It also dynamically clusters latent restaurant styles and predicts personalized rating expectations, including risk intervals.

## 👥 Team & Division of Labor

- **Neil (Module A):** Data Ops & Scraping (Google Places API & Reddit PRAW pipelines, raw image downloading).
- **Leo (Module B):** Vector Extraction, Retrieval & DB (CLIP embeddings, LanceDB management, Time-Decay logic).
- **Craig (Module C):** Advanced ML Algorithms (UMAP & GMM for style clustering, Quantile Regression for risk/volatility).
- **Merry (Module D):** Frontend Developer (Streamlit GUI, multimodal search inputs, dynamic UI badges).
- **Grace (Module E):** NLP & System Integration (ABSA pipeline for review sentiment, data schema synchronization).

## 🏗️ Core Architecture & Directory Structure

The project pipeline is built on a highly decoupled architecture utilizing Vision-Language Models (CLIP/SigLIP), Vector Databases (LanceDB), and advanced unsupervised/supervised ML algorithms.

### 1. Data Pipelines (`pipelines/`)

- **`social_scraper.py`** *(Neil)*: Lightweight hybrid crawler utilizing **Google Places API** and **Apify Yelp Scraper**. Automatically circumvents API barriers by proxying queries to Yelp for unlimited high-quality images and full timestamped reviews.
- **`fetch_and_embed_reviews.py`** *(Neil)*: Scalable script to scrape Google Places text reviews and directly embed them using the local `distilbert-base-uncased` language model into standard Parquet format for downstream algorithms.
- **`menu_crawler.py`** *(Neil)*: Crawls websites and PDFs, leveraging lightweight LLMs (e.g., GPT-4o-mini) to extract and structure complex fine-dining menus into clean JSON.
- **`absa_processor.py`** *(Grace)*: Runs Aspect-Based Sentiment Analysis (ABSA) on UGC text to extract specific sentiment scores for Food, Service, and Ambiance.

### 2. Core Algorithms (`algorithms/`)

- **`generate_embeddings.py`** *(Leo)*: Maps local images and text into a shared Euclidean vector space using OpenAI CLIP (or SigLIP).
- **`retrieval_engine.py`** *(Leo)*: Vector DB querying logic natively applying an exponential **Time-Decay weight** ($1 + \lambda e^{-\alpha \Delta t}$) to boost trending dishes.
- **`dimensionality_reduction.py` & `clustering.py`** *(Craig)*: Integrates UMAP to preserve local topological structures of fused text+image vectors, feeding into Gaussian Mixture Models (GMM) to discover latent restaurant styles.
- **`quantile_regression.py`** *(Craig)*: Predicts "Risk/Confidence Intervals" for ratings by incorporating Grace's ABSA sentiment inputs (Service vs. Food).

### 3. Vector Storage (`vector_db/`)

- A localized, lightweight Vector Database (e.g., **LanceDB** or **ChromaDB**) replacing static `.parquet` files to efficiently manage multimodal queries, incremental delta updates, and timestamp metadata filtering. *(Managed by Leo)*

### 4. Applications & UI (`ui_components/`)

- **`app.py`** *(Merry)*: Native Streamlit GUI supporting multimodal cross-domain search (Text-to-Image / Image-to-Image) and latent style visualization.
- **`ui_components/image_grid.py`** *(Merry)*: Renders matrix maps with visual badges (e.g., "🔥 Trending") based on recent metadata and Leo's time-decay scores.

### 5. Testing Structure (`tests/`)

- Modular scaffolding containing foundational test suites (`test_algorithms.py`, `test_api.py`) validating mathematical abstractions and data flows.

## ⚙️ Installation & Setup

1. **Virtual Environment**:
Ensure you use a virtual environment (`.venv`) managed via `uv`.
    
    ```bash
    uv venv
    # Mac/Linux Activation
    source ./.venv/bin/activate
    # Windows Activation
    .venv\Scripts\activate
    ```
    
2. **Install Dependencies**:
    
    ```
    pip install -r requirements.txt
    ```
    
3. **Environment Keys**:
Copy the included `.env.example` file to `.env` to securely define your custom configuration variables (DO NOT commit `.env` to Git):
    
    ```
    GOOGLE_PLACES_API_KEY=your_google_key_here
    REDDIT_CLIENT_ID=your_reddit_id_here
    REDDIT_CLIENT_SECRET=your_reddit_secret_here
    OPENAI_API_KEY=your_openai_key_here
    APIFY_API_TOKEN=your_apify_token_here
    KAGGLE_USERNAME=your_kaggle_username_here
    KAGGLE_KEY=your_kaggle_key_here
    ```
    

## 🚀 Usage Examples

**Run the UGC Scraper (Google Places/Apify):**

```
python pipelines/social_scraper.py
```

**Generate Embeddings & Populate Vector DB:**

```
python algorithms/generate_embeddings.py
```

**Launch the Streamlit Recommendation App:**

```
streamlit run app.py
```

## 📜 Workflows & Agent Guidelines

The project enforces strict guidelines (see `.cursorrules`):

- All code changes must be tracked in version control and pushed with descriptive summaries.
- The virtual environment (`venv`) must always be respected.
- All code must be logically documented with docstrings and internal comments.
- Agent activities and architectural revisions are logged historically within `CHANGELOG.md`.