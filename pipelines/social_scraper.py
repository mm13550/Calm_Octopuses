import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "DataOpsBot/1.0")

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

def fetch_custom_search_images(restaurant_name, rest_id):
    """Fetch images from UGC sites using Custom Search API with date restrict"""
    images = []
    if not GOOGLE_CSE_ID or not GOOGLE_PLACES_API_KEY:
        print("Missing CSE ID. Skipping Custom Search.")
        return images
    
    def execute_search(date_restrict, start_index=1):
        url = "https://customsearch.googleapis.com/customsearch/v1"
        query_str = f"{restaurant_name} restaurant NYC food OR menu"
        params = {
            "key": GOOGLE_PLACES_API_KEY, 
            "cx": GOOGLE_CSE_ID,
            "q": query_str,
            "searchType": "image",
            "dateRestrict": date_restrict,
            "num": 10,
            "start": start_index
        }
        res = requests.get(url, params=params)
        if res.status_code == 200:
            return res.json().get("items", [])
        return []
            
    items = []
    # Fetch up to 30 images (3 pages) from the last year
    for start_idx in [1, 11, 21]:
        page_items = execute_search("y1", start_idx)
        if not page_items:
            break
        items.extend(page_items)
        time.sleep(1) # rate limiting
        
    current_timestamp = datetime.now().timestamp()
    
    for i, item in enumerate(items):
        image_url = item.get("link")
        if not image_url: continue
        
        try:
            # Download image and verify content type
            img_res = requests.get(image_url, timeout=5)
            content_type = img_res.headers.get('content-type', '')
            if img_res.status_code == 200 and 'image' in content_type:
                filename = f"{rest_id}_cse_{i}.jpg"
                file_path = os.path.join(IMAGES_DIR, filename)
                with open(file_path, 'wb') as f:
                    f.write(img_res.content)
                images.append({
                    "image_uid": f"img_cse_{rest_id}_{i}",
                    "rest_id": rest_id,
                    "source": "google_custom_search",
                    "image_path": file_path,
                    "timestamp": current_timestamp
                })
        except Exception:
            pass # ignore broken links or timeouts
            
    return images

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
        
        # 2. Google Custom Search Images
        cse_images = fetch_custom_search_images(rest_name, rest_id)
        all_images.extend(cse_images)

        
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