"""
pipelines/refetch_missing_yelp_images.py
=========================================
Re-fetches Yelp photo URLs for restaurants whose images were not downloaded
during the initial social scrape.

Reads the expected image index from ``data/social_images.csv``, identifies
missing files on disk, fetches the photos from Yelp, and saves them to the
correct paths.

Usage::

    python pipelines/refetch_missing_yelp_images.py
"""
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from apify_client import ApifyClient
except Exception:  # pragma: no cover - optional local dependency
    ApifyClient = None


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
SOCIAL_IMAGES_CSV = DATA_DIR / "social_images.csv"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")


def parse_expected_index(image_uid: str) -> int:
    return int(str(image_uid).rsplit("_", 1)[-1])


def get_restaurant_name(rest_id: str) -> str:
    lookup_df = pd.read_csv(LOOKUP_CSV)
    name_col = "restaurant_name" if "restaurant_name" in lookup_df.columns else "name"
    match = lookup_df.loc[lookup_df["rest_id"] == rest_id, name_col]
    if match.empty:
        raise ValueError(f"rest_id '{rest_id}' not found in {LOOKUP_CSV}")
    return str(match.iloc[0])


def get_missing_yelp_rows(rest_id: str) -> pd.DataFrame:
    df = pd.read_csv(SOCIAL_IMAGES_CSV)
    mask = (df["rest_id"] == rest_id) & (df["source"] == "yelp")
    yelp_rows = df.loc[mask].copy()
    if yelp_rows.empty:
        raise ValueError(f"No Yelp image rows found for rest_id '{rest_id}'")

    yelp_rows["file_exists"] = yelp_rows["image_path"].apply(
        lambda rel_path: (PROJECT_ROOT / rel_path).exists()
    )
    missing = yelp_rows.loc[~yelp_rows["file_exists"]].copy()
    if not missing.empty:
        missing["expected_index"] = missing["image_uid"].apply(parse_expected_index)
        missing = missing.sort_values("expected_index")
    return missing


def fetch_yelp_photo_urls(restaurant_name: str) -> list[str]:
    if not APIFY_API_TOKEN:
        raise ValueError("Missing APIFY_API_TOKEN in environment.")
    if ApifyClient is None:
        raise ValueError("apify_client is not installed in this environment.")

    client = ApifyClient(APIFY_API_TOKEN)
    run_input = {
        "searchTerms": [restaurant_name],
        "locations": ["New York"],
        "searchLimit": 1,
        "reviewsCount": 40,
        "scrapeReview": False,
        "scrapeImages": True,
    }

    run = client.actor("tri_angle/yelp-scraper").call(run_input=run_input)

    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        photos = item.get("photos") or item.get("images") or item.get("imageUrls") or []
        urls: list[str] = []
        for photo in photos:
            url = photo if isinstance(photo, str) else (photo.get("url") or photo.get("link"))
            if not url:
                continue
            if url.startswith("//"):
                url = "https:" + url
            urls.append(url)
        return urls

    return []


def download_to_expected_paths(missing_rows: pd.DataFrame, photo_urls: list[str]) -> tuple[int, list[str]]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    downloaded = 0

    expected = list(missing_rows.itertuples(index=False))
    if len(photo_urls) < len(expected):
        raise ValueError(
            f"Only fetched {len(photo_urls)} Yelp photo URLs but need {len(expected)} files."
        )

    for row, photo_url in zip(expected, photo_urls):
        target_path = PROJECT_ROOT / row.image_path
        try:
            response = requests.get(photo_url, timeout=15)
            content_type = response.headers.get("content-type", "")
            if response.status_code != 200 or "image" not in content_type:
                failures.append(f"{row.image_uid} -> bad response ({response.status_code}, {content_type})")
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as f:
                f.write(response.content)
            downloaded += 1
        except Exception as exc:
            failures.append(f"{row.image_uid} -> {exc}")

    return downloaded, failures


def main(rest_id: str) -> None:
    restaurant_name = get_restaurant_name(rest_id)
    missing_rows = get_missing_yelp_rows(rest_id)

    if missing_rows.empty:
        print(f"No missing Yelp images for {restaurant_name} ({rest_id}).")
        return

    print(f"Target restaurant: {restaurant_name} ({rest_id})")
    print(f"Missing Yelp image files: {len(missing_rows)}")

    photo_urls = fetch_yelp_photo_urls(restaurant_name)
    print(f"Fetched {len(photo_urls)} Yelp photo URLs from Apify.")

    downloaded, failures = download_to_expected_paths(missing_rows, photo_urls)
    print(f"Downloaded {downloaded} files.")

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python pipelines/refetch_missing_yelp_images.py <rest_id>")

    main(sys.argv[1])

