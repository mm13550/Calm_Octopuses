import argparse
import csv
import os
import requests
import sys
import time

CSV_FILE = "data/nyc_michelin_names_cleaned.csv"
OUTPUT_DIR = "data/images"

# Maximum number of restaurants to process by default
MAX_PROCESSED = 400

def get_place_photo_references(restaurant_name, borough, api_key):
    """
    Retrieves photo references for a specific restaurant using the Google Maps API.
    
    This function performs a 2-step process:
    1. Text Search API: Finds the exact Place ID for the restaurant using its name and borough.
    2. Place Details API: Fetches the 'photos' array for that specific Place ID.
    
    Args:
        restaurant_name (str): The name of the restaurant.
        borough (str): The NYC borough where the restaurant is located.
        api_key (str): The Google Maps API key.
        
    Returns:
        list: A list of photo reference strings that can be used to download the actual images.
    """
    # Step 1: Text Search to get the specific Place ID
    # We append 'food dishes' to prioritize culinary photos over storefronts
    query = f"{restaurant_name} restaurant food dishes"
    if borough:
        query += f" {borough}"
    query += " New York City"
    
    url_search = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params_search = {"query": query, "key": api_key}
    
    resp_search = requests.get(url_search, params=params_search)
    if resp_search.status_code != 200:
        return []
        
    data_search = resp_search.json()
    
    # Handle API rate limits or denial errors
    if data_search.get("status") in ["OVER_QUERY_LIMIT", "REQUEST_DENIED"]:
        print(f"\n[!] API Error during Text Search: {data_search.get('status')} - {data_search.get('error_message', '')}")
        sys.exit(1)
        
    results = data_search.get("results", [])
    if not results:
        return []
        
    # Extract the unique identifier for the place
    place_id = results[0].get("place_id")
    if not place_id:
        return []
        
    # Step 2: Place Details API to get the full array of up to 10 photos
    url_details = "https://maps.googleapis.com/maps/api/place/details/json"
    params_details = {
        "place_id": place_id,
        "fields": "photos",
        "key": api_key
    }
    
    resp_details = requests.get(url_details, params=params_details)
    if resp_details.status_code != 200:
        return []
        
    data_details = resp_details.json()
    if data_details.get("status") in ["OVER_QUERY_LIMIT", "REQUEST_DENIED"]:
        print(f"\n[!] API Error during Place Details: {data_details.get('status')} - {data_details.get('error_message', '')}")
        sys.exit(1)
        
    photos = data_details.get("result", {}).get("photos", [])
    if not photos:
        return []
        
    # Skip the first image (index 0), as it's usually the building exterior cover photo.
    # We collect the next 5 pictures (index 1 to 5) which are typically food items.
    if len(photos) == 1:
        selected_photos = photos
    else:
        selected_photos = photos[1:6]
        
    return [p.get("photo_reference") for p in selected_photos]

def download_photo(photo_reference, save_path, api_key):
    """
    Downloads the actual image from Google Maps using its photo reference.
    
    Args:
        photo_reference (str): The photo reference token from Google Maps API.
        save_path (str): The local file path where the image should be saved.
        api_key (str): The Google Maps API key.
        
    Returns:
        bool: True if the image was downloaded successfully, False otherwise.
    """
    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {
        "maxwidth": 800,  # Restrict max width to save bandwidth/storage while maintaining quality
        "photoreference": photo_reference,
        "key": api_key
    }
    response = requests.get(url, params=params, stream=True)
    if response.status_code == 200:
        # Save image data in chunks to handle large files properly and use minimal RAM
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    elif response.status_code == 403:
        # A 403 usually indicates project quota limits have been reached
        print("\n[!] 403 Forbidden when downloading photo. You may have exceeded your quota.")
        sys.exit(1)
    return False

def main():
    """
    Main execution loop for scraping restaurant photos.
    Reads restaurant names from a CSV file, fetches their photos via Google Maps API,
    and downloads them to a local directory.
    """
    parser = argparse.ArgumentParser(description="Scrape restaurant images from Google Maps")
    parser.add_argument("--limit", type=int, help="Limit the number of restaurants to process for testing.")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_MAPS_API_KEY", ""), help="Google Maps API Key")
    args = parser.parse_args()

    if not args.api_key:
        print("Missing Google Maps API key. Pass --api-key or set GOOGLE_MAPS_API_KEY.", file=sys.stderr)
        return 2

    # Create the output directory if it doesn't already exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    processed_count = 0
    limit = args.limit if args.limit is not None else MAX_PROCESSED
    
    # Read the dataset containing Michelin-starred restaurants
    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if processed_count >= limit:
                print(f"\nReached the processing limit of {limit}. Stopping.")
                break
                
            name = row.get("name")
            borough = row.get("borough")
            
            if not name:
                continue
                
            # Create a filesystem-safe version of the restaurant name for image filenames
            safe_name = "".join([c if c.isalnum() else "_" for c in name]).lower().strip("_")
            if not safe_name:
                continue
                
            print(f"\nProcessing {name}...")
            
            # Retrieve the list of photo reference tokens mapping to images
            photo_refs = get_place_photo_references(name, borough, args.api_key)
            if photo_refs:
                for idx, photo_ref in enumerate(photo_refs):
                    save_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{idx+1}.jpg")
                    
                    # Prevent redundant downloads if the image already exists locally
                    if os.path.exists(save_path):
                        print(f" -> Skipping {save_path}, image already exists.")
                        continue
                        
                    success = download_photo(photo_ref, save_path, args.api_key)
                    if success:
                        print(f" -> Successfully saved {save_path}")
                    else:
                        print(f" -> Failed to download photo {idx+1}.")
            else:
                print(f" -> No suitable photos found for {name}.")
            
            processed_count += 1
            # Slight delay to ensure we do not violently hit the Google API rate limits
            time.sleep(0.1)

    print("\nFinished scraping.")

if __name__ == "__main__":
    main()