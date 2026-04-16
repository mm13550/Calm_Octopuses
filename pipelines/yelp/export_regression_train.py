"""
Export Regression Training Set
==============================
Generates a `regression_train_set.json` natively from the massive `yelp_relations.db`.
Targets 20,000 profiles where BOTH the target and the historical lists are 
generic casual restaurants (`is_val_target = 0`).

Crucially, it uses an inner join optimization to prioritize exactly the 
users mapped in `regression_val_set.json` first, allowing natural Domain Adaptation.
"""

import os
import json
import sqlite3
import random
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
DATA_DIR = os.path.join(_ROOT, 'data', 'yelp_sandbox')
DB_PATH = os.path.join(DATA_DIR, 'yelp_relations.db')

REGRESSION_TRAIN_OUT = os.path.join(DATA_DIR, 'regression_train_set.json')
REGRESSION_VAL_IN    = os.path.join(DATA_DIR, 'regression_val_set.json')

def main():
    if not os.path.exists(DB_PATH):
        print("SQLite DB not found! Please run preprocess_yelp.py first.")
        return
        
    print("--- Extracting Safe Generic Quantile Regression Profiles ---")
    
    # 1. Get priority user_ids from the Michelin proxy validation set
    priority_users = set()
    if os.path.exists(REGRESSION_VAL_IN):
        with open(REGRESSION_VAL_IN, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            priority_users = {row['user_id'] for row in val_data}
        print(f"Loaded {len(priority_users)} priority users for Domain Adaptation.")
    
    # 2. Bulk load all generic reviews directly into memory (blazing fast in Python, avoids SQLite locking on complex JOINs)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("\nBulk extracting safe reviews from DB to Memory...")
    # Get all reviews for generic targets explicitly without heavy JOIN GROUP BY logic
    cur.execute("""
        SELECT r.u_id, r.b_id, r.stars 
        FROM reviews r
        JOIN businesses b ON r.b_id = b.b_id
        WHERE b.is_val_target = 0
    """)
    records = cur.fetchall()
    conn.close()
    
    print(f"Loaded {len(records)} raw review records. Grouping temporally...")
    from collections import defaultdict
    user_groups = defaultdict(list)
    for u, b, s in records:
        user_groups[u].append((b, s))
        
    dataset = []
    
    # 3. Process Domain Adaptation prioritization!
    # First, handle the users that exist in our validation tier to force exact-taste learning safely
    count = 0
    for u_id in priority_users:
        if u_id in user_groups and len(user_groups[u_id]) >= 6:
            hist = user_groups[u_id]
            target = random.choice(hist)
            history = [r for r in hist if r[0] != target[0]]
            if len(history) >= 5:
                dataset.append({
                    "user_id": u_id,
                    "target_michelin_business": target[0],
                    "target_actual_rating": target[1],
                    "historical_business_list": [h[0] for h in history],
                    "historical_ratings": [h[1] for h in history]
                })
                count += 1
                
    print(f"Injected {count} priority overlapping Domain Adaptation users.")
    
    # 4. Fill the rest with massive baseline generic profiles until we hit 8000 for toy scope
    total_needed = 8000
    current_count = len(dataset)
    
    for u_id, hist in user_groups.items():
        if current_count >= total_needed:
            break
        if u_id in priority_users: 
            continue # already processed
            
        if len(hist) >= 6:
            target = random.choice(hist)
            history = [r for r in hist if r[0] != target[0]]
            if len(history) >= 5:
                dataset.append({
                    "user_id": u_id,
                    "target_michelin_business": target[0],
                    "target_actual_rating": target[1],
                    "historical_business_list": [h[0] for h in history],
                    "historical_ratings": [h[1] for h in history]
                })
                current_count += 1
                
    with open(REGRESSION_TRAIN_OUT, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n[SUCCESS] Extracted {len(dataset)} perfectly mapped generic profiles.")
    print(f"[SUCCESS] Exported into: {os.path.basename(REGRESSION_TRAIN_OUT)}")

if __name__ == "__main__":
    main()
