# Agent Conversations Log

*This file contains the historical logs of all conversations with the AI agent for this repository.*

## March 25, 2026 - Google Maps Image Scraper
- Designed and implemented `image_scrapper.py` to fetch images from the Google Maps API for restaurants in `nyc_michelin_names_cleaned.csv`.
- Built-in safety limits to ensure free tier is respected.
- Resolved CSV BOM encoding issue to correctly parse restaurant names.
- Confirmed that the user's Google Maps API key works by successfully testing the script.

## March 25, 2026 - Virtual Environment Setup
- Created a project-specific Python virtual environment (`venv`).
- Generated `requirements.txt` containing dependencies (`requests`) and installed them into the venv.
- Created `activate.bat` and `activate.ps1` scripts for easy activation on Windows.
- Updated `.gitignore` to prevent tracking of the virtual environment.

## March 25, 2026 - Enhancing Scraper for Food Photos
- Modified `image_scrapper.py` to heavily prioritize food photos over building exteriors.
- Adjusted query strings to include "food dishes" and implemented a two-step API flow (Text Search -> Place Details) to retrieve all available photos.
- Configured the script to systematically skip the first photo (typically storefronts) and download up to 5 additional photos per restaurant.

## March 25, 2026 - Background Execution
- Launched the final `image_scrapper.py` script to continuously fetch all 1,750 photos in the background.
- Added the `images/` directory to `.gitignore` to protect the GitHub repository from massive file uploads.

## March 28, 2026 - Fixing Embeddings Generation
- Diagnosed an issue in `generate_embeddings.py` where `CLIPModel.get_image_features()` returned a `BaseModelOutputWithPooling` object instead of a tensor.
- Modified the script to correctly extract `pooler_output` or the first tuple element, ensuring backward compatibility.
- Successfully generated and saved embeddings for 468 images to `image_embeddings.parquet`.

## March 28, 2026 - Updating Agent Rules
- Updated `.cursorrules` to instruct the agent to always use the project's virtual environment (`venv`).
- Added a rule to ensure code is well-documented with clear comments explaining logic across files and functions.

## March 28, 2026 - Embeddings Directory Refactor
- Created an `embeddings` folder to store all future embeddings (images, text, etc.), making the structure scalable.
- Moved `image_embeddings.parquet` into `embeddings/` and updated `generate_embeddings.py` to save to this new location.
- Appended `embeddings/` to `.gitignore` to prevent committing massive ML output files to GitHub.

## March 28, 2026 - Streamlit Debug App
- Designed and implemented `app.py`, a Streamlit-based interface to debug and visualize image similarities using the pre-computed CLIP embeddings.
- Installed `streamlit` in the virtual environment.
- Verified that the application correctly computes cosine similarities via dot products and renders the Top `N` visually similar images alongside their embeddings.

## March 28, 2026 - Keeping the README Updated
- Appended a strict rule to `.cursorrules` enforcing that the agent maintains an up-to-date `README.md` reflecting the live state of the codebase and project structure.
- Completely rewrote the `README.md` to properly document the entire data pipeline (from homepage resolution in SerpAPI, into Google Maps image scraping, through CLIP embeddings generation, and finally rendering on the Streamlit App).

## March 28, 2026 - Data Directory Reorganization
- Reorganized the project structure by creating a `data/` directory and moved all CSV files into it.
- Automatically updated internal file paths and documentation inside `image_scrapper.py`, the SerpAPI resolvers, the menu crawler, and the README to reference the new paths.

## March 28, 2026 - Merged Dependencies
- Merged the contents of `requirements_menu_crawler.txt` into the main `requirements.txt` file setup to unify backend dependencies.
- Deleted `requirements_menu_crawler.txt` to avoid project clutter.

## March 28, 2026 - SerpAPI Resolvers Merged
- Migrated all rate-limit protection (`RateLimitError` handling), exponential backoffs, and `--resume` functionalities from `resolve_homepages_with_serpapi_resume.py` directly into the standard `resolve_homepages.py`.
- Cleanly deleted the redundant `_resume.py` file to simplify the project's ecosystem.
- Reflected these smart capabilities in the `README.md`.

## March 28, 2026 - Script Renaming
- Renamed `nyc_michelin_menu_crawler.py` to `menu_crawler.py` and `resolve_homepages_with_serpapi.py` to `resolve_homepages.py` for brevity.
- Updated all internal string references across `README.md`, Python docstrings, and old conversation logs.

## March 28, 2026 - Documentation Updates
- Removed all "AI" buzzword terminology from `README.md`, replacing it with standard descriptive phrasing for similarity metrics.

## March 28, 2026 - Gitignore Update
- Appended `PLAN.md` to `.gitignore` and removed it from Git tracking to avoid cluttering the remote commit history with temporary plan objectives.
