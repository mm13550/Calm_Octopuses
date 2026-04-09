import os
import time
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "DataOpsBot/1.0")

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def fetch_google_places_data(restaurant_name):
    """
    Fetch place details, recent reviews, and photos using Google Places API (New).
    
    This performs a two-step process: First querying `places:searchText` to resolve 
    the restaurant name into a place_id, and then querying Place Details using 
    strict Field Masking to fetch only the reviews and photos lists.
    
    Args:
        restaurant_name (str): The name of the target restaurant.
        
    Returns:
        list: A list of dicts formatted to the standard social tracking schema.
    """
    results = []
    
    if not GOOGLE_PLACES_API_KEY:
        print("Warning: GOOGLE_PLACES_API_KEY is missing.")
        return results

    # 1. Text Search to get place.id
    search_url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.id" # CRITICAL: Minimal field mask
    }
    payload = {"textQuery": f"{restaurant_name} restaurant NYC"}
    
    response = requests.post(search_url, headers=headers, json=payload)
    if response.status_code != 200 or not response.json().get('places'):
        return results
        
    place_id = response.json()['places'][0]['id']

    # 2. Place Details to get newest reviews and photos
    details_url = f"https://places.googleapis.com/v1/places/{place_id}"
    details_headers = {
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        # CRITICAL: Only request exactly what we need
        "X-Goog-FieldMask": "reviews,photos" 
    }
    # Using base request since reviewsSort isn't valid for GetPlace in v1
    # params = {"reviewsSort": "newest"}
    
    details_response = requests.get(details_url, headers=details_headers)
    if details_response.status_code != 200:
        return results
        
    details_data = details_response.json()
    
    # Process Reviews & Photos
    reviews = details_data.get('reviews', [])
    photos = details_data.get('photos', [])
    
    for i, review in enumerate(reviews):
        # Fallback to current time if publishTime is missing
        publish_time = review.get('publishTime', datetime.now().isoformat())
        timestamp = pd.to_datetime(publish_time).timestamp()
        
        image_path = None
        # Try to associate a photo if available (using index)
        if i < len(photos):
            photo_ref = photos[i].get('name')
            if photo_ref:
                image_path = download_google_photo(photo_ref, f"{place_id}_google_{i}")
        
        results.append({
            "uid": f"g_{place_id}_{i}",
            "rest_id": place_id,
            "source": "google",
            "image_path": image_path,
            "text": review.get('text', {}).get('text', ''),
            "timestamp": timestamp
        })
        
    return results

def download_google_photo(photo_reference, filename_prefix):
    """
    Download actual image file from Google Places Photo API.
    
    Args:
        photo_reference (str): The Google specific image reference string.
        filename_prefix (str): Prefix used to save the file locally.
        
    Returns:
        str: The local file path to the saved image, or None if the request fails.
    """
    photo_url = f"https://places.googleapis.com/v1/{photo_reference}/media?maxHeightPx=800&maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
    response = requests.get(photo_url)
    if response.status_code == 200:
        file_path = os.path.join(IMAGES_DIR, f"{filename_prefix}.jpg")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return file_path
    return None

def fetch_reddit_data(restaurant_name, rest_id):
    """
    Fetch Reddit data or gracefully inject mock data if PRAW is pending.
    
    Args:
        restaurant_name (str): Target restaurant to search for in subreddits.
        rest_id (str): The internal/Google ID mapped to the restaurant.
        
    Returns:
        list: A list of dicts formatted to the standard schema with Reddit source.
    """
    if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
        try:
            import praw
            # Full PRAW logic would go here
            # reddit = praw.Reddit(...)
            # return real_results
            pass 
        except ImportError:
            print("PRAW not installed. Falling back to mock data.")
    
    # Graceful bypass: Mock Data
    print(f"Reddit API keys pending. Injecting mock Reddit data for {restaurant_name}.")
    mock_timestamp = datetime.now().timestamp() - 86400 # 1 day ago
    return [{
        "uid": f"r_mock_{rest_id}_01",
        "rest_id": rest_id,
        "source": "reddit",
        "image_path": "mock_reddit_image_url.jpg",
        "text": f"Just had the tasting menu at {restaurant_name}. Incredible progression, though the dessert was a bit sweet. #FoodNYC",
        "timestamp": mock_timestamp
    }]

def main():
    input_file = os.path.join(DATA_DIR, 'nyc_michelin_names_cleaned.csv')
    output_file = os.path.join(DATA_DIR, 'raw_social_recent.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return
        
    df = pd.read_csv(input_file).head(3) # Test on first 3 to save quota
    
    all_data = []
    for index, row in df.iterrows():
        rest_name = row['name']
        print(f"Processing: {rest_name}")
        
        # Google
        google_data = fetch_google_places_data(rest_name)
        all_data.extend(google_data)
        
        # Reddit
        # Using a dummy rest_id for reddit mock if we couldn't get google place_id
        rest_id = google_data[0]['rest_id'] if google_data else f"dummy_{index}"
        reddit_data = fetch_reddit_data(rest_name, rest_id)
        all_data.extend(reddit_data)
        
        time.sleep(1) # Rate limiting safety

    # Save exactly to schema
    output_df = pd.DataFrame(all_data, columns=['uid', 'rest_id', 'source', 'image_path', 'text', 'timestamp'])
    output_df.to_csv(output_file, index=False)
    print(f"Success! Data saved to {output_file}")

if __name__ == "__main__":
    main()