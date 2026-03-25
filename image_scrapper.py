import argparse
import csv
import os
import requests
import sys
import time

GOOGLE_MAPS_API_KEY = "AIzaSyBh4rL1nvaFmZXoZH7VtjE1mV7aH8H8Ymc"
CSV_FILE = "nyc_michelin_names_cleaned.csv"
OUTPUT_DIR = "images"

MAX_PROCESSED = 400

def get_place_photo_references(restaurant_name, borough):
    # Step 1: Text Search to get the specific Place ID
    query = f"{restaurant_name} restaurant food dishes"
    if borough:
        query += f" {borough}"
    query += " New York City"
    
    url_search = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params_search = {"query": query, "key": GOOGLE_MAPS_API_KEY}
    
    resp_search = requests.get(url_search, params=params_search)
    if resp_search.status_code != 200:
        return []
        
    data_search = resp_search.json()
    
    if data_search.get("status") in ["OVER_QUERY_LIMIT", "REQUEST_DENIED"]:
        print(f"\n[!] API Error during Text Search: {data_search.get('status')} - {data_search.get('error_message', '')}")
        sys.exit(1)
        
    results = data_search.get("results", [])
    if not results:
        return []
        
    place_id = results[0].get("place_id")
    if not place_id:
        return []
        
    # Step 2: Place Details API to get the full array of up to 10 photos
    url_details = "https://maps.googleapis.com/maps/api/place/details/json"
    params_details = {
        "place_id": place_id,
        "fields": "photos",
        "key": GOOGLE_MAPS_API_KEY
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
        
    # Skip the first image (index 0, usually the building exterior cover photo)
    # Collect the next 5 pictures (index 1 to 5)
    if len(photos) == 1:
        selected_photos = photos
    else:
        selected_photos = photos[1:6]
        
    return [p.get("photo_reference") for p in selected_photos]

def download_photo(photo_reference, save_path):
    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {
        "maxwidth": 800,
        "photoreference": photo_reference,
        "key": GOOGLE_MAPS_API_KEY
    }
    response = requests.get(url, params=params, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    elif response.status_code == 403:
        print("\n[!] 403 Forbidden when downloading photo. You may have exceeded your quota.")
        sys.exit(1)
    return False

def main():
    parser = argparse.ArgumentParser(description="Scrape restaurant images from Google Maps")
    parser.add_argument("--limit", type=int, help="Limit the number of restaurants to process for testing.")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    processed_count = 0
    limit = args.limit if args.limit is not None else MAX_PROCESSED
    
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
                
            safe_name = "".join([c if c.isalnum() else "_" for c in name]).lower().strip("_")
            if not safe_name:
                continue
                
            print(f"\nProcessing {name}...")
            
            photo_refs = get_place_photo_references(name, borough)
            if photo_refs:
                for idx, photo_ref in enumerate(photo_refs):
                    save_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{idx+1}.jpg")
                    
                    if os.path.exists(save_path):
                        print(f" -> Skipping {save_path}, image already exists.")
                        continue
                        
                    success = download_photo(photo_ref, save_path)
                    if success:
                        print(f" -> Successfully saved {save_path}")
                    else:
                        print(f" -> Failed to download photo {idx+1}.")
            else:
                print(f" -> No suitable photos found for {name}.")
            
            processed_count += 1
            time.sleep(0.1)

    print("\nFinished scraping.")

if __name__ == "__main__":
    main()