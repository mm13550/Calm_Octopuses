import sys
import os

# Add project root to python path so we can import pipelines
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pipelines.menu_crawler import scrape_menu_text

# Target URL for Le Pavillon
url = "https://www.lepavillonnyc.com/"

print(f"Scraping HTML/PDF content from: {url}")
print("=" * 80)
raw_text = scrape_menu_text(url)

print("\n--- EXTRACTED RAW TEXT OUTPUT ---")
if raw_text:
    print(raw_text)
else:
    print("[NO TEXT EXTRACTED]")
print("=" * 80)
