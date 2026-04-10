import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from apify_client import ApifyClient

# Load environment variables
load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "DataOpsBot/1.0")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def download_google_photo(photo_reference, filename_prefix):
    """Download actual image file from Google Places Photo API"""
    photo_url = f"https://places.googleapis.com/v1/{photo_reference}/media?maxHeightPx=800&maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
    response = requests.get(photo_url)
    if response.status_code == 200:
        file_path = os.path.join(IMAGES_DIR, f"{filename_prefix}.jpg")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return file_path
    return None

def fetch_google_places_data(restaurant_name):
    """Fetch recent reviews and photos using Google Places API"""
    review_results = []
    image_results = []
    place_id_val = None
    
    if not GOOGLE_PLACES_API_KEY:
        print("Warning: GOOGLE_PLACES_API_KEY is missing.")
        return review_results, image_results, None

    # 1. Text Search to get place.id
    search_url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.id"
    }
    payload = {"textQuery": f"{restaurant_name} restaurant NYC"}
    
    response = requests.post(search_url, headers=headers, json=payload)
    if response.status_code != 200 or not response.json().get('places'):
        return review_results, image_results, None
        
    place_id = response.json()['places'][0]['id']
    place_id_val = place_id

    # 2. Place Details to get reviews and photos
    details_url = f"https://places.googleapis.com/v1/places/{place_id}"
    details_headers = {
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "reviews,photos" 
    }
    
    details_response = requests.get(details_url, headers=details_headers)
    if details_response.status_code != 200:
        return review_results, image_results, place_id_val
        
    details_data = details_response.json()
    
    reviews = details_data.get('reviews', [])
    photos = details_data.get('photos', [])
    
    # Time filtering: 1 year (365 days) ago
    one_year_ago = datetime.now() - timedelta(days=365)
    
    for i, review in enumerate(reviews):
        # Time filter
        publish_time_str = review.get('publishTime')
        if publish_time_str:
            try:
                # e.g. "2023-01-01T12:00:00Z"
                publish_time = pd.to_datetime(publish_time_str).replace(tzinfo=None)
                if publish_time < one_year_ago:
                    continue # Skip older reviews
                timestamp = publish_time.timestamp()
            except Exception:
                timestamp = datetime.now().timestamp()
        else:
            timestamp = datetime.now().timestamp()
            
        review_results.append({
            "uid": f"g_{place_id}_{i}",
            "rest_id": place_id,
            "source": "google_places",
            "text": review.get('text', {}).get('text', ''),
            "timestamp": timestamp,
            "rating": review.get('rating', None)
        })
        
    # Process Google Photos (No reliable timestamp, using current time)
    current_timestamp = datetime.now().timestamp()
    for i, photo in enumerate(photos):
        photo_ref = photo.get('name')
        if photo_ref:
            image_path = download_google_photo(photo_ref, f"{place_id}_google_{i}")
            if image_path:
                image_results.append({
                    "image_uid": f"img_g_{place_id}_{i}",
                    "rest_id": place_id,
                    "source": "google_places",
                    "image_path": image_path,
                    "timestamp": current_timestamp
                })
        
    return review_results, image_results, place_id_val

def fetch_yelp_data_apify(restaurant_name, rest_id):
    """Fetch reviews and images from Yelp using Apify scraper"""
    reviews_out = []
    images_out = []
    
    if not APIFY_API_TOKEN:
        print("Missing APIFY_API_TOKEN. Skipping Yelp Apify Scrape.")
        return reviews_out, images_out

    client = ApifyClient(APIFY_API_TOKEN)
    
    run_input = {
        "searchTerms": [restaurant_name],
        "locations": ["New York"],
        "searchLimit": 1,
        "reviewsCount": 40,           # Set parameter to 40 images/reviews
        "scrapeReview": True,
        "scrapeImages": True
    }

    try:
        print(f"  > Contacting Apify Yelp Scraper for {restaurant_name}...")
        run = client.actor("tri_angle/yelp-scraper").call(run_input=run_input)
        
        # Time filtering: 1 year (365 days) ago
        one_year_ago = datetime.now() - timedelta(days=365)
        current_timestamp = datetime.now().timestamp()
        
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            print("  [DEBUG] Received Yelp Business Item:", item.get("name"))
            # 1. Process Reviews
            reviews = item.get("reviews", [])
            print(f"  [DEBUG] Found {len(reviews)} reviews inside 'reviews' key.")
            for i, rev in enumerate(reviews):
                timestamp_str = rev.get("date") # e.g. "2024-11-01T10:23:00.000Z"
                if timestamp_str:
                    try:
                        publish_time = pd.to_datetime(timestamp_str).replace(tzinfo=None)
                        if publish_time < one_year_ago:
                            continue # older than 1 year
                        ts = publish_time.timestamp()
                    except Exception:
                        ts = current_timestamp
                else:
                    ts = current_timestamp
                
                content = rev.get("text") or rev.get("body") or ""
                
                reviews_out.append({
                    "uid": f"y_{rest_id}_{i}",
                    "rest_id": rest_id,
                    "source": "yelp",
                    "text": content,
                    "timestamp": ts,
                    "rating": rev.get("rating", None)
                })
            
            # 2. Process Photos
            # photos might be under 'photos', 'images', 'imageUrls'
            photos = item.get("photos") or item.get("images") or item.get("imageUrls") or []
            print(f"  [DEBUG] Found {len(photos)} photos keys.")
            
            for i, photo in enumerate(photos):
                image_url = photo if isinstance(photo, str) else (photo.get("url") or photo.get("link"))
                if not image_url: 
                    print(f"  [DEBUG] Skipping photo {i}: No URL found. {photo}")
                    continue
                
                if image_url.startswith("//"):
                    image_url = "https:" + image_url
                
                try:
                    img_res = requests.get(image_url, timeout=5)
                    content_type = img_res.headers.get('content-type', '')
                    if img_res.status_code == 200 and 'image' in content_type:
                        filename = f"{rest_id}_yelp_{i}.jpg"
                        file_path = os.path.join(IMAGES_DIR, filename)
                        with open(file_path, 'wb') as f:
                            f.write(img_res.content)
                        images_out.append({
                            "image_uid": f"img_y_{rest_id}_{i}",
                            "rest_id": rest_id,
                            "source": "yelp",
                            "image_path": file_path,
                            "timestamp": current_timestamp
                        })
                    else:
                        print(f"  [DEBUG] Skipping photo {i}: Bad status or content-type ({img_res.status_code}, {content_type})")
                except Exception as e:
                    print(f"  [DEBUG] Skipping photo {i}: Exception: {e}")
                    pass # ignore broken links or timeouts
                    
            # We only expect 1 business result
            break
            
    except Exception as e:
        print(f"  > Error running Apify: {e}")
            
    return reviews_out, images_out

def main():
    # === BATCH CONFIGURATION ===
    # For free Custom Search limits (100 req/day), process 33 restaurants per batch.
    # Day 1: start_row=0, end_row=33
    # Day 2: start_row=33, end_row=66
    # Day 3: start_row=66, end_row=99
    start_row = 0
    end_row = 33
    # ===========================
    
    input_file = os.path.join(DATA_DIR, 'nyc_michelin_names_cleaned.csv')
    reviews_output = os.path.join(DATA_DIR, 'social_reviews.csv')
    images_output = os.path.join(DATA_DIR, 'social_images.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return
        
    df = pd.read_csv(input_file).iloc[start_row:end_row]
    print(f"Starting batch process for restaurants {start_row} to {end_row}...")
    
    all_reviews = []
    all_images = []
    
    for index, row in df.iterrows():
        rest_name = row['name']
        print(f"Processing: {rest_name}")
        
        # 1. Google Places Data
        g_reviews, g_images, place_id = fetch_google_places_data(rest_name)
        all_reviews.extend(g_reviews)
        all_images.extend(g_images)
        
        rest_id = place_id if place_id else f"dummy_{index}"
        
        # 2. Yelp Apify Data (Reviews & Images)
        y_reviews, y_images = fetch_yelp_data_apify(rest_name, rest_id)
        all_reviews.extend(y_reviews)
        all_images.extend(y_images)

        
        time.sleep(1) # Safety delay
        
    # Standardize relational schema and export to CSV
    reviews_df = pd.DataFrame(all_reviews, columns=['uid', 'rest_id', 'source', 'text', 'timestamp', 'rating'])
    images_df = pd.DataFrame(all_images, columns=['image_uid', 'rest_id', 'source', 'image_path', 'timestamp'])
    
    # Use append mode ('a') so we don't overwrite previous runs!
    reviews_df.to_csv(reviews_output, mode='a', header=not os.path.exists(reviews_output), index=False)
    images_df.to_csv(images_output, mode='a', header=not os.path.exists(images_output), index=False)
    
    print(f"Success! Data batch mapped and added.")
    print(f" > {reviews_output} appended with {len(reviews_df)} stored reviews.")
    print(f" > {images_output} appended with {len(images_df)} stored images.")

if __name__ == "__main__":
    main()