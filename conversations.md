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
