"""
Yelp Preprocessing Pipeline

This script prepares data for the Phase 1 Cross-Modal Autoencoder.
It streams multi-gigabyte Yelp Open Dataset files, extracting a strict
selection of high-quality dining reviews safely without overflowing memory.
It sets apart Philadelphia as a proxy for the NYC Michelin Validation Set.
"""

import os
import json
import random
import pandas as pd
from collections import defaultdict

# Path Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'yelp_sandbox'))
BUSINESS_JSON = os.path.join(DATA_DIR, 'yelp_academic_dataset_business.json')
USER_JSON = os.path.join(DATA_DIR, 'yelp_academic_dataset_user.json')
REVIEW_JSON = os.path.join(DATA_DIR, 'yelp_academic_dataset_review.json')
PHOTOS_JSON = os.path.join(DATA_DIR, 'train.json')
PHOTOS_DIR = os.path.join(DATA_DIR, 'train')

TRAIN_OUT = os.path.join(DATA_DIR, 'train_pairs.parquet')
VAL_OUT = os.path.join(DATA_DIR, 'val_pairs.parquet')

def main():
    print("--- Phase 1: Restaurant Isolation & High-End City Split ---")
    train_businesses = set()
    val_businesses = set()
    
    with open(BUSINESS_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                b = json.loads(line)
            except:
                continue
                
            categories = b.get('categories')
            if not categories or ('Restaurants' not in categories and 'Food' not in categories):
                continue
            
            b_id = b['business_id']
            city = b.get('city', '')
            stars = b.get('stars', 0.0)
            reviews = b.get('review_count', 0)
            attrs = b.get('attributes') or {}
            price = attrs.get('RestaurantsPriceRange2')
            
            # Identify the proxy city logic
            if city == 'Philadelphia':
                # Strictly mimic high-end NYC metrics for our Val Set
                if price in ['3', '4'] and stars >= 4.0 and reviews >= 50:
                    val_businesses.add(b_id)
            else:
                train_businesses.add(b_id)
                
    print(f" > Extracted {len(train_businesses)} target Training Restaurants.")
    print(f" > Extracted {len(val_businesses)} High-End Validation Restaurants located in Philadelphia.")

    print("\n--- Phase 2: User Filtering ---")
    valid_users = set()
    with open(USER_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                u = json.loads(line)
                if u.get('review_count', 0) >= 10:
                    valid_users.add(u['user_id'])
            except:
                pass
    print(f" > Isolated {len(valid_users)} seasoned food reviewers.")

    print("\n--- Phase 3: Review Extraction (Streaming ~5GB JSON) ---")
    # Store aggregated text reviews for each restaurant
    train_reviews_map = defaultdict(list)
    val_reviews_map = defaultdict(list)
    
    # We use chunks to keep RAM from completely crashing
    chunk_size = 50_000
    chunk_iter = pd.read_json(REVIEW_JSON, lines=True, chunksize=chunk_size)
    
    chunks_processed = 0
    for chunk in chunk_iter:
        # Fast filter on the Pandas Dataframe
        valid_chunk = chunk[chunk['user_id'].isin(valid_users)]
        
        # Training Map Additions
        train_chunk = valid_chunk[valid_chunk['business_id'].isin(train_businesses)]
        for _, row in train_chunk.iterrows():
            train_reviews_map[row['business_id']].append(row['text'])
            
        # Validation Map Additions
        val_chunk = valid_chunk[valid_chunk['business_id'].isin(val_businesses)]
        for _, row in val_chunk.iterrows():
            val_reviews_map[row['business_id']].append(row['text'])
            
        chunks_processed += 1
        print(f"   ... Processed {(chunks_processed * chunk_size):,}-row mark ...", end='\r')
        
    print(f"\n > Finalized Review Mapping! (Train places: {len(train_reviews_map)}, Val places: {len(val_reviews_map)})")

    print("\n--- Phase 4: Cross-Modal Weak Supervision Pairing ---")
    train_pairs = []
    val_pairs = []
    
    with open(PHOTOS_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                photo_meta = json.loads(line)
            except:
                continue
                
            photo_id = photo_meta.get('photo_id')
            b_id = photo_meta.get('business_id')
            label = photo_meta.get('label')
            
            # We strictly only want images of food/plates, not outside buildings or menus
            if label != 'food':
                continue
                
            photo_path = os.path.join(PHOTOS_DIR, f"{photo_id}.jpg")
            if not os.path.exists(photo_path):
                continue
                
            # Randomly pair exactly ONE text review to this incoming photo
            # to prevent dataset explosion
            if b_id in train_reviews_map and len(train_reviews_map[b_id]) > 0:
                selected_text = random.choice(train_reviews_map[b_id])
                train_pairs.append({"image_path": photo_path, "text": selected_text, "business_id": b_id})
                
            elif b_id in val_reviews_map and len(val_reviews_map[b_id]) > 0:
                selected_text = random.choice(val_reviews_map[b_id])
                val_pairs.append({"image_path": photo_path, "text": selected_text, "business_id": b_id})
                
    print(f" > Assembled Train Pairs [{len(train_pairs)}]  |  Assembled Val Pairs [{len(val_pairs)}]")
    
    print("\n--- Phase 5: Exporting Final Target Datasets ---")
    if train_pairs:
        pd.DataFrame(train_pairs).to_parquet(TRAIN_OUT, index=False)
        print(f" [SUCCESS] Wrote Training Set to {TRAIN_OUT}")
    if val_pairs:
        pd.DataFrame(val_pairs).to_parquet(VAL_OUT, index=False)
        print(f" [SUCCESS] Wrote Validation Set to {VAL_OUT}")
        
    if not train_pairs and not val_pairs:
        print(" [WARNING] Zero pairs were generated.")

if __name__ == "__main__":
    main()
