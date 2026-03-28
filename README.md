# Calm Octopuses: Michelin NYC Data Project

This project is a comprehensive toolkit for collecting, analyzing, and exploring data and images related to Michelin-listed restaurants in New York City.

## Core Features & Scripts

The project pipeline covers homepage resolution, menu crawling, image scraping from Google Maps, and AI-powered visual similarity exploration.

### 1. Data Collection & Scraping
- **`resolve_homepages_with_serpapi.py`**
  Automatically resolves official restaurant homepages from names using SerpAPI. Includes robust relevance scoring.
- **`resolve_homepages_with_serpapi_resume.py`**
  A resume-supported version of the homepage resolver for handling rate limits and network interruptions.
- **`nyc_michelin_menu_crawler.py`**
  Crawls restaurant websites to fetch HTML pages and PDF files, extracting and structuring menu content.
- **`image_scrapper.py`**
  Fetches and downloads restaurant photos using the Google Maps API, intelligently prioritizing food/dish photos while avoiding standard exterior shots. Note: Images are saved to the `images/` directory which is automatically `.gitignore`'d.

### 2. AI & Data Analytics
- **`generate_embeddings.py`**
  Utilizes the OpenAI CLIP (`clip-vit-base-patch32`) model to parse the downloaded images and generate normalized semantic feature vectors. Results are securely saved as `embeddings/image_embeddings.parquet`.

### 3. Applications
- **`app.py`**
  A Streamlit-based graphical user interface (GUI) designed to explore image embeddings. Select any scraped image and visually discover the top `N` most similar images across your dataset using cosine similarities.

## Installation & Setup

1. **Virtual Environment**: 
   Ensure you use a virtual environment (`venv`).
   ```bash
   python -m venv venv
   # Windows Activation
   venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # (or requirements_menu_crawler.txt if only crawling menus)
   ```
3. **Environment Keys**:
   Make sure you expose any necessary API keys (like Google Maps or SerpAPI).
   ```bash
   set SERPAPI_API_KEY=your_key_here
   ```

## Usage Examples

**Homepage Resolver:**
```bash
python resolve_homepages_with_serpapi.py --input nyc_michelin_names_cleaned.csv --output seeds_resolved.csv --delay 1.0
```

**Image Scraper:**
```bash
python image_scrapper.py --limit 400
```

**Generate Content Embeddings:**
```bash
python generate_embeddings.py
```

**Launch the Similarity App:**
```bash
streamlit run app.py
```

## Workflows & Agent Guidelines
The project enforces strict guidelines (see `.cursorrules`):
- All code changes must be tracked in version control and pushed with descriptive summaries.
- The virtual environment (`venv`) must always be respected.
- All code must be logically documented with docstrings and internal comments.
- Agent activities are logged within `conversations.md`.
- **This `README.md` must be kept fully up to date with the structure of the project.**
