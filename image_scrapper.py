import argparse
import csv
import os
import requests
import sys
import time

GOOGLE_MAPS_API_KEY = "AIzaSyBh4rL1nvaFmZXoZH7VtjE1mV7aH8H8Ymc"
CSV_FILE = "nyc_michelin_names_cleaned.csv"
OUTPUT_DIR = "images"

# To ensure you're not charged beyond the free limit, 
# you should set up Quota Limits in the Google Cloud Console. 
# However, this script is designed to safely exit if the quota is exceeded 
# and has a hardcoded MAX limit to prevent infinite loops or huge bills.
MAX_PROCESSED = 400

def get_place_photo_reference(restaurant_name, borough):
    query = f"{restaurant_name} restaurant"
    if borough:
        query += f" {borough}"
    query += " New York"
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": GOOGLE_MAPS_API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error checking API for {restaurant_name}: {response.status_code}")
        return None
        
    data = response.json()
    
    status = data.get("status")
    if status == "OVER_QUERY_LIMIT":
        print("\n[!] OVER QUERY LIMIT. Your quota has been exceeded or billing is not enabled.")
        print("[!] Stopping script to prevent any unexpected charges.")
        sys.exit(1)
    elif status == "REQUEST_DENIED":
        print(f"\n[!] REQUEST DENIED: {data.get('error_message', 'Invalid API key or restricted access.')}")
        sys.exit(1)
        
    results = data.get("results", [])
    if not results:
        return None
        
    place = results[0]
    photos = place.get("photos", [])
    if not photos:
        return None
        
    return photos[0].get("photo_reference")

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
    elif response.status_code == 403: # Over quota for photos sometimes gives 403
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
                
            save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.jpg")
            
            if os.path.exists(save_path):
                print(f"Skipping {name}, image already exists.")
                continue
                
            print(f"Processing {name}...")
            
            photo_ref = get_place_photo_reference(name, borough)
            if photo_ref:
                success = download_photo(photo_ref, save_path)
                if success:
                    print(f" -> Successfully saved {save_path}")
                else:
                    print(f" -> Failed to download photo.")
            else:
                print(f" -> No photo found for {name}.")
            
            processed_count += 1
            # Respect Google API rate limits slightly
            time.sleep(0.1)

    print("\nFinished scraping.")

if __name__ == "__main__":
    main()