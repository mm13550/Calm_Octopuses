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
    Generate menu, review, and image embedding JSONL files under
    ``data/embeddings/``.
michelin_absa_pipeline.py
    Run aspect-based sentiment analysis over review embeddings and aggregate
    restaurant-level scores.
build_restaurant_profiles.py
    Merge per-modality embeddings into a single restaurant profile vector.
social_scraper.py
    Scrape Yelp and Google Places data for each Michelin restaurant.
menu_crawler.py
    Crawl restaurant homepages and extract structured menu data.
maintenance.py
    Deduplicate, audit, refresh, and realign generated artifacts.

Subpackage: yelp/
    Scripts for the Yelp Open Dataset training pipeline.
"""
