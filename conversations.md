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
