# Conversation Logs

## Session: Live Training & Review Embedding (April 9, 2026)
- **Objective:** The user requested reviewing `ReviewsPlan.md` and setting up the environment for a live Minibatch training system based on their architecture.
- **Actions Taken:** 
  1. We planned the live training algorithms out, shifting their GMM to `MiniBatchKMeans`.
  2. We built a script `pipelines/fetch_and_embed_reviews.py` using `DistilBERT` to embed Google Places API reviews into `data/embeddings/reviews_embeddings.parquet`.
  3. We set up an API safety net explicitly limiting to 2 restaurants, tested successfully, and then stripped the throttle.
  4. The background tool executed through over 350+ restaurants globally fetching 1,750 text reviews mapping them directly to a Parquet output.
  5. Finally, we reviewed `.cursorrules` adjusting the code documentation `README.md`, logging this conversation directly, and committing the new changes to Git.

## Session: Push modified scripts and docs to GitHub (April 12, 2026)
- **Objective:** Publish local changes to [mm13550/Calm_Octopuses](https://github.com/mm13550/Calm_Octopuses).
- **Actions Taken:**
  1. Confirmed `origin` points at the GitHub repo; local `main` was one commit ahead with a clean working tree.
  2. Initial `git push` failed because `origin/main` had newer commits (housekeeping moves under `data/embeddings`, CSV routing, PyTorch toy isolation).
  3. Ran `git pull --rebase origin main`; resolved a rename/delete conflict (`debug_output.txt` moved upstream to `tests/tests_output/debug_output.txt` while the local commit removed it) by removing `tests/tests_output/debug_output.txt` to match the cleanup intent.
  4. Pushed rebased `main` successfully (`9854efa..9f9951b`).

## Session: Pipeline Reorganization & Generalization Analysis Tab (April 16, 2026)
- **Objective:** Organize Yelp pipelines into a subfolder and add a cross-modal generalization analysis tab to the Streamlit app.
- **Actions Taken:**
  1. Moved all Yelp-related pipelines (`cross_modal_embeddings.py`, `download_yelp_dataset.py`, `generate_embeddings_yelp.py`, `preprocess_yelp.py`) into a new `pipelines/yelp/` subfolder and fixed all broken `DATA_DIR` relative paths caused by the move.
  2. Fixed the `cross_modal_embeddings.py` subprocess path reference in `app.py` Tab 2.
  3. Created `pipelines/yelp/evaluate_generalization.py`: loads the best model checkpoint, runs inference on both the Yelp general (train) and Philadelphia high-end (val) cohorts, and computes alignment MSE, reconstruction MSE, cosine similarity distributions, t-SNE projections, and restaurant discriminability metrics.
  4. Added a new **"Generalization Analysis"** tab (Tab 3) to `app.py` with KPI cards, epoch curves, reconstruction MSE bar chart, cosine similarity histogram, t-SNE scatter, inter-restaurant pairwise distance histogram, and an auto-classified interpretation table (🟢/🟡/🔴).
  5. Added `compute_discriminability()` to the evaluator: groups latents by `business_id`, computes per-restaurant centroids, and measures inter-restaurant pairwise cosine distance and intra-restaurant cosine similarity to produce a discriminability score.
  6. Fixed a stale `@st.cache_data` schema issue via an auto-invalidation guard.
  7. Updated `README.md` to document the new `pipelines/yelp/` structure and `evaluate_generalization.py`.

## Session: Aggregating Yelp Embeddings & Regression Architecture Plan (April 16, 2026, Part 2)
- **Objective:** Design the regression network architecture and build Phase 1: a pipeline to aggregate per-restaurant generic embeddings.
- **Actions Taken:** 
  1. Drafted an Implementation Plan for predicting rating intervals via PyTorch Quantile Loss (`IntervalScorer`).
  2. Created `pipelines/yelp/aggregate_restaurant_embeddings.py` to extract latents and group them via mathematical mean-pooling into singular unified 256-D restaurant vectors.
  3. Ran the aggregation offline: successfully reduced 9,861 raw pairs into 6,975 discrete Yelp restaurants, and 139 high-end pairs into 54 discrete validation restaurants.

