# Michelin NYC Menu Crawler

This project helps collect menu information for Michelin-listed restaurants in New York City.

## Files

- `nyc_michelin_menu_crawler.py`  
  Crawls restaurant websites and extracts menu content from HTML pages and PDF menus.

- `resolve_homepages_with_serpapi.py`  
  Resolves restaurant official homepages automatically from restaurant names using SerpAPI.

- `resolve_homepages_with_serpapi_resume.py`  
  Resume-friendly version of the homepage resolver. Useful when API rate limits interrupt the run.

- `requirements_menu_crawler.txt`  
  Python dependencies.

- `nyc_michelin_names_cleaned.csv`  
  Cleaned list of Michelin-listed NYC restaurants.

- `seeds_resolved.csv`  
  Restaurant list with resolved homepage URLs.

## Workflow

### 1. Install dependencies

```bash
python -m pip install -r requirements_menu_crawler.txt
```

### 2. Set SerpAPI key

Windows CMD:

```bash
set SERPAPI_API_KEY=your_key_here
```

### 3. Resolve restaurant homepages

```bash
python resolve_homepages_with_serpapi.py --input nyc_michelin_names_cleaned.csv --output seeds_resolved.csv --delay 1.0
```

If rate limits interrupt the run:

```bash
python resolve_homepages_with_serpapi_resume.py --input nyc_michelin_names_cleaned.csv --output seeds_resolved.csv --delay 20 --resume
```

### 4. Crawl menus

```bash
python nyc_michelin_menu_crawler.py --input seeds_resolved.csv --output-dir out --per-domain-delay 2.0 --max-pages-per-site 25
```

## Notes

- The crawler targets restaurant official websites, not Michelin Guide pages.
- Some restaurants may not publish menus publicly.
- Some menus are embedded in images or JavaScript-heavy pages and may not be extracted successfully.
- API rate limits may require resume mode and slower delays.

## Output

Typical output files:

- `out/menus.csv`
- `out/menus.jsonl`
