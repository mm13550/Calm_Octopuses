# Project Proposal & Implementation Plan: Michelin NYC Dining Recommender (v3.0)

## 1. Problem & Objective

Current dining search tools (Google Maps, Yelp, Michelin Guide) rely heavily on coarse tags. For food enthusiasts seeking specific dish presentations (e.g., "creamy uni pasta") or precise ambiance, search is inefficient and lacks personalization.

This v3.0 project builds a **multimodal recommendation web application** focused on Michelin-starred restaurants. By integrating official menus with low-friction User-Generated Content (UGC), it enables precise dish-level searches using **images or natural language**. It also clusters latent restaurant styles and predicts personalized rating expectations, including risk intervals.

## 2. Core Methodology & Algorithmic Enhancements

1. **Multimodal Cross-Domain Search:**
    - Embeds restaurant menus (text) and images into a shared vector space via OpenAI CLIP (or SigLIP) for text-to-image and image-to-image similarity retrieval.
    - **Enhancement:** Introduces a lightweight **Vector Database** (e.g., LanceDB or ChromaDB) to efficiently manage query retrieval and incremental delta updates.
2. **Dynamic UGC Integration & Time-Decay Matching:**
    - Uses **Google Places API** and **Reddit API** to fetch image-bound reviews accompanied by strict timestamps (specifically querying for the `NEWEST` reviews to ensure data relevance).
    - Applies an exponential time-decay factor to Cosine Similarity ($Score = \text{Cosine}(Q, V) \times (1 + \lambda \cdot e^{-\alpha \cdot \Delta t})$), organically boosting trending dishes and diminishing discontinued ones.
3. **LLM-Assisted Menu Parsing:**
    - **Enhancement:** Leverages Vision-Language Models (e.g., GPT-4o-mini / Claude 3) to extract structured JSON (Dish, Description, Price) from unstructured PDF and HTML menus, drastically improving text processing over brittle regular expressions.
4. **Multimodal Style Feature Fusion & Clustering:**
    - Concatenates official text descriptions (Text Embeddings) with the active mean vector of recent social images (Image Embeddings).
    - Utilizes **UMAP** for dimensionality reduction, followed by **Gaussian Mixture Models (GMM)** to cluster restaurants and generate "Style Tags" (e.g., "Classic French" vs. "Modern Creative").
5. **Quantile Regression for Score Interval Prediction:**
    - Moves beyond a static average predicted score by calculating a Confidence Interval (representing the risk of a bad experience).
    - **Enhancement:** Integrates **Aspect-Based Sentiment Analysis (ABSA)** derived from the UGC text (scoring food vs. service vs. ambiance separately) directly into the Quantile Regression framework.

## 3. Phased Implementation Plan (4-Week Agile Sprint)

### Phase 1: Infrastructure & Data Foundation

- Register Developer APIs for Reddit (PRAW) and Google Cloud (Places API).
- Build the `social_scraper.py` pipeline pulling recent (3-6 months) structured JSON. Implement strict Field Masking for Google to optimize billing.
- Enhance `menu_crawler.py` utilizing LLMs to standardize PDF extractions.
- Ensure data lands in standard schemas (e.g., `[uid, rest_id, source, image_path, text, timestamp]`).

### Phase 2: Core Vectors & Recommendation Logic

- Switch core storage backend to a Vector Database (`LanceDB`) handling `image_embeddings` and `recent_social_embeddings` seamlessly.
- Integrate mathematical formula for exponential Time-Decay weight into `retrieval_engine.py`.

### Phase 3: Advanced Clustering & Prediction

- Structure the Multi-modal fusion mappings in `clustering.py`.
- Apply UMAP dimensionality reduction on the text+image vectors, feeding output into the GMM clusters.
- Update `quantile_regression.py` to ingest the new multidimensional ABSA vectors and train the variance prediction models.

### Phase 4: Streamlit Frontend UX Integration

- Upgrade GUI in `app.py` allowing interactive text and image query upload modes.
- Implement "Trending" and "Recently Spotted" graphic overlays in `ui_components/image_grid.py` fueled by decay rankings.
- Expose the visual Style Tags dynamically on the UI.

## 4. Existing Scripts That Require Modification

The implementation of v3.0 requires immediate cascading updates to the following logic files:

1. **`requirements.txt`**: Needs addition of vector DB, API clients, and ML libs (`lancedb`, `praw`, `requests`, `openai`, `umap-learn`). *Removed `yelpapi`.*
2. **`pipelines/menu_crawler.py`**: Update to call LLM prompts instead of relying solely on sequential text parsing.
3. **`algorithms/retrieval_engine.py`**: Inject the $1 + \lambda e^{-\alpha \Delta t}$ time decay modifier into the vector query scoring function.
4. **`algorithms/clustering.py`**: Refactor internal logic blocks to map concatenated Text+Image embeddings through UMAP into GMM.
5. **`algorithms/quantile_regression.py`**: Modify the input tensor shape to accept decoupled ABSA sentiment arrays (food, service, ambiance) instead of just naive representations.
6. **`app.py`**: Redesign the Streamlit UI Sidebar to include file uploads for image searching and a text bar for natural language query execution.