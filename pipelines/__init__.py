"""
pipelines/
==========
One-off data preparation and embedding generation scripts.

These scripts are run manually (or via CI) to (re-)generate the data artefacts
that the frontend and ML pipeline consume.  They are **not** imported by the
Streamlit apps; they are standalone executables invoked from the command line.

Key scripts
-----------
generate_embeddings_michelin.py
    Embed NYC Michelin restaurant menus and reviews into 512-D CLIP vectors
    stored in ``data/embeddings/``.
generate_embeddings_images.py
    Embed social food/interior images into per-image CLIP vectors.
generate_embeddings_reviews.py
    Embed social review text into per-review CLIP vectors.
build_restaurant_profiles.py
    Merge per-modality embeddings into a single restaurant profile vector.
social_scraper.py
    Scrape Yelp and Google Places data for each Michelin restaurant.
menu_crawler.py
    Crawl restaurant homepages and extract structured menu data.

Subpackage: yelp/
    Scripts for the Yelp Open Dataset training pipeline.
"""
