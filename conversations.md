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
