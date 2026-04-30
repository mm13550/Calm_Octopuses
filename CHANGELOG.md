# Changelog

This file summarizes meaningful project changes. It is not a full conversation log.
For the current architecture, setup, and active entry points, see `README.md` and `PLAN.md`.

## Unreleased

- Reworked the app around `frontend.py` as the primary Streamlit entry point.
- Added `core/data_loader.py` as the shared data-loading and catalog assembly layer.
- Added reusable Streamlit UI components under `ui_components/`.
- Added active retrieval helpers in `algorithms/retrieval.py` for CLIP text search, image search, scoped menu/review search, lexical blending, and exact dish matching.
- Added Gaussian Mixture clustering utilities in `algorithms/clustering.py`.
- Added MDN-based recommendation support in `algorithms/mdn_regression.py`, including an embedding-based fallback when the trained checkpoint is unavailable.
- Added tests for clustering behavior, retrieval helper determinism, and shared data-loader utilities.
- Added `pyproject.toml` pytest configuration.
- Added `data/README.md` to clarify local data and generated-artifact policy.
- Cleaned repository documentation to match the current file structure, consolidated pipeline scripts, and local-data policy.
- Updated Michelin award loading to support the current `data/csv/nyc_michelin_awards.xlsx` workbook, including Bib Gourmand distinctions.

## Current Data Policy

- Small source data can be tracked in Git, including lookup CSVs, Michelin metadata, bios, and `data/csv/restaurant_profiles.csv`.
- Large generated artifacts stay local and are ignored by Git, including downloaded images, embedding JSONL files, parsed menu exports, vector databases, and Yelp sandbox outputs.

## Legacy History

Earlier project iterations included prototype files and names such as `app.py`, `generate_embeddings.py`, `image_scrapper.py`, and `quantile_regression.py`. Those names appear in older notes but are not the current application path. The current application path is:

```text
frontend.py
  -> core/data_loader.py
  -> algorithms/retrieval.py
  -> algorithms/mdn_regression.py
  -> ui_components/
```

Older one-off notes about local virtual environments, API-key testing, scratch scripts, and temporary repository trees were intentionally removed from this changelog because they are not useful for public project history.
