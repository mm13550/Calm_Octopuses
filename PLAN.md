# Project Proposal & Implementation Plan: Michelin NYC Dining Recommender (v3.0)

## 1. Problem & Objective

Current dining search tools (Google Maps, Yelp, Michelin Guide) rely heavily on coarse tags. For food enthusiasts seeking specific dish presentations (e.g., "creamy uni pasta") or precise ambiance, search is inefficient and lacks personalization.

This v3.0 project builds a **multimodal recommendation web application** focused on Michelin-starred restaurants. By integrating official menus with low-friction User-Generated Content (UGC), it enables precise dish-level searches using **images or natural language**. It also clusters latent restaurant styles and predicts personalized rating expectations, including risk intervals.

## 2. Core Methodology & Algorithmic Enhancements

1. **Multimodal Cross-Domain Search:**
    - Embeds restaurant menus (text) and images into a shared vector space via OpenAI CLIP (or SigLIP) for text-to-image and image-to-image similarity retrieval.
    - **Enhancement:** Introduces a lightweight **Vector Database** (e.g., LanceDB or ChromaDB) to efficiently manage query retrieval and incremental delta updates.
2. **Dynamic UGC Integration:**
    - Uses **Google Places API** (for reviews and photos) and **Apify Yelp Scraper** (for high-quality food images and additional reviews).
    - Retrieval scoring is based purely on **Cosine Similarity** between query embeddings and stored vectors. No temporal or popularity weighting is applied, as neither Google Places API nor Apify Yelp Scraper provide reliable metadata for such scoring.
3. **LLM-Assisted Menu Parsing:**
    - **Enhancement:** Leverages Vision-Language Models (e.g., GPT-4o-mini / Claude 3) to extract structured JSON (Dish, Description, Price) from unstructured PDF and HTML menus, drastically improving text processing over brittle regular expressions.
4. **Multimodal Style Feature Fusion & Clustering:**
    - Concatenates official text descriptions (Text Embeddings) with the mean vector of all social images per restaurant (Image Embeddings).
    - Utilizes **UMAP** for dimensionality reduction, followed by **Gaussian Mixture Models (GMM)** to cluster restaurants and generate "Style Tags" (e.g., "Classic French" vs. "Modern Creative").
5. **Quantile Regression for Score Interval Prediction:**
    - Moves beyond a static average predicted score by calculating a Confidence Interval (representing the risk of a bad experience).
    - **Enhancement:** Integrates **Aspect-Based Sentiment Analysis (ABSA)** derived from the UGC text (scoring food vs. service vs. ambiance separately) directly into the Quantile Regression framework.

## 3. Phased Implementation Plan (4-Week Agile Sprint)

### Phase 1: Infrastructure & Data Foundation

- Register Developer APIs for Apify and Google Cloud (Places API).
- Build the `social_scraper.py` pipeline: Use Google to resolve unique Place IDs (`rest_id`) and extract the 5 most recent Google maps reviews. Proxy search terms to Apify Yelp Scraper to autonomously fetch 30+ URLs of user-generated food imagery per location.
- Enhance `menu_crawler.py` utilizing LLMs to standardize PDF extractions, alongside a robust three-stage extraction pipeline (Crawl, Retry, Merge) that automatically handles API Rate Limits, Next.js/React Server Components hidden menus, and strict firewalls (406 Bypass).
- **[NEW]** Build the `bio_crawler.py` pipeline: Crawl homepage and 'About Us' URLs to extract official restaurant descriptions, culinary styles, and chef history to properly anchor styling representations during the UMAP/GMM clustering phase.
- Ensure data lands in standard schemas: reviews as `[uid, rest_id, source, text, rating]`, images as `[image_uid, rest_id, source, image_path]`.

### Phase 2: Core Vectors & Recommendation Logic

- Switch core storage backend to a Vector Database (`LanceDB`) handling `image_embeddings` and `social_embeddings` seamlessly.
- Implement cosine similarity retrieval in `retrieval_engine.py`.

### Phase 3: Advanced Clustering & Prediction

- Structure the Multi-modal fusion mappings in `clustering.py`.
- Apply UMAP dimensionality reduction on the text+image vectors, feeding output into the GMM clusters.
- Update `quantile_regression.py` to ingest the new multidimensional ABSA vectors and train the variance prediction models.

### Phase 4: Streamlit Frontend UX Integration

- Upgrade GUI in `app.py` allowing interactive text and image query upload modes.
- Implement visual overlays in `ui_components/image_grid.py` for search results.
- Expose the visual Style Tags dynamically on the UI.

## 4. Existing Scripts That Require Modification

The implementation of v3.0 requires immediate cascading updates to the following logic files:

1. **`requirements.txt`**: Needs addition of vector DB, API clients, and ML libs (`lancedb`, `apify-client`, `requests`, `openai`, `umap-learn`).
2. **`pipelines/menu_crawler.py`**: Update to call LLM prompts instead of relying solely on sequential text parsing.
3. **`algorithms/retrieval_engine.py`**: Implement cosine similarity retrieval against the vector database.
4. **`algorithms/clustering.py`**: Refactor internal logic blocks to map concatenated Text+Image embeddings through UMAP into GMM.
5. **`algorithms/quantile_regression.py`**: Modify the input tensor shape to accept decoupled ABSA sentiment arrays (food, service, ambiance) instead of just naive representations.
6. **`app.py`**: Redesign the Streamlit UI Sidebar to include file uploads for image searching and a text bar for natural language query execution.