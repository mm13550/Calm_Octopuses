# Data Directory

This folder contains the small source files needed to assemble the restaurant catalog, plus local generated artifacts that are intentionally ignored by Git.

## Tracked Data

- `csv/restaurant_lookup.csv`
- `csv/michelin_awards.csv`
- `csv/nyc_michelin_names_cleaned.csv`
- `csv/seeds_resolved.csv`
- `csv/restaurant_profiles.csv`
- `csv/social_reviews.csv`
- `csv/social_images.csv`
- `extracted_bios/restaurant_bios_joinable.json`

## Local-Only Data

These files are generated or too large for the repository:

- `embeddings/*.jsonl`
- `images/*`
- `extracted_menus/final_parsed_menus.json`
- `vector_db/`
- Yelp sandbox outputs

Keep local-only data out of commits. If a new generated artifact is required to reproduce the app locally, document where it should be placed and add it to `.gitignore`.
