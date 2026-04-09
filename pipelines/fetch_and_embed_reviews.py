import os
import time
import requests
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# Load environment variables
load_dotenv()

# We will check both GOOGLE_PLACES_API_KEY and GOOGLE_MAPS_API_KEY as fallbacks
API_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Using a lightweight local text embedding model (DistilBERT)
MODEL_NAME = "distilbert-base-uncased"
print(f"Loading '{MODEL_NAME}' for text embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def embed_text(text):
    """Generate a vector embedding for a given string of text."""
    if not text.strip():
        return None
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = model(**inputs)
        # Use mean pooling over the token sequence
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().flatten()

def get_place_id(restaurant_name):
    """Retrieve the Google Place ID using a text search."""
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id"
    }
    payload = {"textQuery": f"{restaurant_name} restaurant NYC"}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        places = response.json().get('places', [])
        if places:
            return places[0]['id']
    else:
        print(f"Search API Error for '{restaurant_name}': {response.status_code} - {response.text}")
    return None

def get_reviews(place_id, max_reviews=10):
    """Retrieve up to 'max_reviews' reviews for a given Google Place ID."""
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "reviews"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        reviews = data.get('reviews', [])
        # Strict limit to max_reviews (default 10)
        return reviews[:max_reviews]
    else:
        print(f"Details API Error for place '{place_id}': {response.status_code} - {response.text}")
    return []

def main():
    if not API_KEY:
        print("Error: API Key is missing. Ensure GOOGLE_MAPS_API_KEY or GOOGLE_PLACES_API_KEY is in .env")
        return

    input_file = os.path.join(DATA_DIR, 'nyc_michelin_names_cleaned.csv')
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    # SAFETY NET REMOVED: Running on all restaurants
    print("SAFETY SETTING LIFTED: Generating embeddings for all restaurants.")
    df_restaurants = pd.read_csv(input_file)

    all_review_records = []

    for idx, row_data in df_restaurants.iterrows():
        name = row_data['name']
        print(f"\nProcessing restaurant: {name.encode('ascii', 'ignore').decode('ascii')}")

        place_id = get_place_id(name)
        if not place_id:
            print(f"Could not find Place ID for {name.encode('ascii', 'ignore').decode('ascii')}. Skipping...")
            continue
            
        print(f"Found Place ID: {place_id}")
        
        # Get at most 10 reviews
        reviews = get_reviews(place_id, max_reviews=10)
        print(f"Found {len(reviews)} reviews. Generating embeddings...")

        for rev in reviews:
            # Safely extract text field. The Google response usually puts the raw text inside `text.text`
            text_dict = rev.get('text', {})
            actual_text = text_dict.get('text', '') if isinstance(text_dict, dict) else rev.get('text', '')
            
            if actual_text:
                embedding = embed_text(actual_text)
                if embedding is not None:
                    # Storing metadata and vector together
                    record = {
                        "restaurant_name": name,
                        "place_id": place_id,
                        "review_id": rev.get('name', ''),
                        "rating": rev.get('rating', None),
                        "publish_time": rev.get('publishTime', ''),
                        "text": actual_text,
                        "embedding": embedding
                    }
                    all_review_records.append(record)

        # Rate limiting to play explicitly safe with API quota rules
        time.sleep(1)

    print(f"\nTotal embedded reviews generated: {len(all_review_records)}")

    if all_review_records:
        out_df = pd.DataFrame(all_review_records)
        output_path = os.path.join(EMBEDDINGS_DIR, 'reviews_embeddings.parquet')
        out_df.to_parquet(output_path, index=False)
        print(f"Successfully saved test embeddings to: {output_path}")
    else:
        print("No valid reviews were completely mapped and embedded.")

if __name__ == "__main__":
    main()
