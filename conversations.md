# Conversation Logs

## Session: Live Training & Review Embedding (April 9, 2026)
- **Objective:** The user requested reviewing `ReviewsPlan.md` and setting up the environment for a live Minibatch training system based on their architecture.
- **Actions Taken:** 
  1. We planned the live training algorithms out, shifting their GMM to `MiniBatchKMeans`.
  2. We built a script `pipelines/fetch_and_embed_reviews.py` using `DistilBERT` to embed Google Places API reviews into `data/embeddings/reviews_embeddings.parquet`.
  3. We set up an API safety net explicitly limiting to 2 restaurants, tested successfully, and then stripped the throttle.
  4. The background tool executed through over 350+ restaurants globally fetching 1,750 text reviews mapping them directly to a Parquet output.
  5. Finally, we reviewed `.cursorrules` adjusting the code documentation `README.md`, logging this conversation directly, and committing the new changes to Git.
