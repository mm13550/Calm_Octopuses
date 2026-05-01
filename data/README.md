# Data Directory

This folder contains the small source files needed to assemble the restaurant catalog, plus local generated artifacts that are intentionally ignored by Git.

## Tracked Data

- `csv/restaurant_lookup.csv`
- `csv/nyc_michelin_awards.xlsx`
- `csv/nyc_michelin_names_cleaned.csv`
- `csv/seeds_resolved.csv`
- `csv/restaurant_profiles.csv`
- `csv/social_reviews.csv`
- `csv/social_images.csv`
- `extracted_bios/restaurant_bios_joinable.json`

## Local-Only Data

These files are generated or too large for the repository and must be copied in
from the shared asset bundle before the app will run end-to-end:

- `embeddings/*.jsonl`
- `images/*`
- `extracted_menus/final_parsed_menus.json`
- `vector_db/`
- Yelp sandbox outputs

The standard setup path is `python download_assets.py`, which downloads the
default asset bundle into `data/`. If your team uses a shared Google Drive
mirror instead, copy the same files into the matching paths under `data/`.

Keep local-only data out of commits. If a new generated artifact is required to
reproduce the app locally, document where it should be placed and add it to
`.gitignore`.
