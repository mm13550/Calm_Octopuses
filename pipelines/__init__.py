"""
pipelines/
==========
One-off data preparation and embedding generation scripts.

These scripts are run manually (or via CI) to (re-)generate the data artefacts
that the frontend and ML pipeline consume.  They are **not** imported by the
Streamlit apps; they are standalone executables invoked from the command line.

Key scripts
-----------
embeddings_pipeline.py
    Generate menu, review, and image CLIP embedding JSONL files under
    ``data/embeddings/``.
build_restaurant_profiles.py
    Merge per-modality embeddings into a single restaurant profile vector.
social_scraper.py
    Scrape Yelp and Google Places data for each Michelin restaurant.
menu_crawler.py
    Crawl restaurant homepages and extract structured menu data.
menu_pipeline.py
    Merge parsed menu exports and retry restaurants with zero dishes.
maintenance.py
    Deduplicate, audit, refresh, and realign generated artifacts.

Subpackage: yelp/
    Scripts for the Yelp Open Dataset training pipeline.
"""
