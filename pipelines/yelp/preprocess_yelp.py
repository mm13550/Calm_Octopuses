"""
Yelp Out-Of-Core Database Preprocessing Pipeline

This pipeline ingests the multi-gigabyte Yelp dataset and streams it directly
into a local SQLite database to prevent RAM starvation. It extracts two natively
distinct datasets:
1. Weakly supervised Image-To-Text subsets for Autoencoder training.
2. Temporal User-History profiles for Quantile Regression testing.
"""

import os
import json
import sqlite3
import pandas as pd
import random

# Core Configurations
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'yelp_sandbox'))
DB_PATH = os.path.join(DATA_DIR, 'yelp_relations.db')

BUSINESS_JSON = os.path.join(DATA_DIR, 'yelp_academic_dataset_business.json')
REVIEW_JSON = os.path.join(DATA_DIR, 'yelp_academic_dataset_review.json')
PHOTOS_JSON = os.path.join(DATA_DIR, 'train.json')
PHOTOS_DIR = os.path.join(DATA_DIR, 'train')

TRAIN_AUTOENCODER = os.path.join(DATA_DIR, 'autoencoder_train.parquet')
VAL_AUTOENCODER = os.path.join(DATA_DIR, 'autoencoder_val.parquet')
REGRESSION_VAL_OUT = os.path.join(DATA_DIR, 'regression_val_set.json')

def init_db():
    """Initializes the SQLite Schema for streaming ingest"""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH) # Start fresh for pipeline runs
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Extreme DB Optimization for massive bulk streaming
    cur.execute('PRAGMA synchronous = OFF')
    cur.execute('PRAGMA journal_mode = MEMORY')
    
    # Pruned schema for speed
    cur.execute('''
        CREATE TABLE businesses (
            b_id TEXT PRIMARY KEY,
            is_val_target INTEGER DEFAULT 0
        )
    ''')
    cur.execute('''
        CREATE TABLE reviews (
            r_id TEXT PRIMARY KEY,
            b_id TEXT,
            u_id TEXT,
            stars REAL,
            text TEXT,
            FOREIGN KEY (b_id) REFERENCES businesses (b_id)
        )
    ''')
    # Indices for blazing fast aggregation lookup natively on disk
    cur.execute('CREATE INDEX idx_review_bid ON reviews (b_id)')
    cur.execute('CREATE INDEX idx_review_uid ON reviews (u_id)')
    conn.commit()
    return conn

def stream_businesses(conn):
    """Phase 1: Filter restaurants and designate the NYC-proxy val set"""
    print("\n--- Phase 1: Business Extraction & Validation Tiering ---")
    cur = conn.cursor()
    
    train_count = 0
    val_count = 0
    
    with open(BUSINESS_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                b = json.loads(line)
            except: continue
                
            categories = b.get('categories')
            if not categories or ('Restaurants' not in categories and 'Food' not in categories):
                continue
                
            b_id = b['business_id']
            city = b.get('city', '')
            stars = b.get('stars', 0.0)
            rev_cnt = b.get('review_count', 0)
            attrs = b.get('attributes') or {}
            price = attrs.get('RestaurantsPriceRange2')
            
            is_val = 0
            if city == 'Philadelphia' and price in ['3', '4'] and stars >= 4.0 and rev_cnt >= 50:
                is_val = 1
                val_count += 1
            else:
                train_count += 1
                
            cur.execute("INSERT OR IGNORE INTO businesses (b_id, is_val_target) VALUES (?, ?)", (b_id, is_val))
            
    conn.commit()
    print(f" [DB] Populated {train_count} standard Training Restaurants.")
    print(f" [DB] Populated {val_count} elite Philadelphia Validation proxies.")

def stream_reviews(conn):
    """Phase 2: Use Pandas chunking to stream 5GB of JSON lines directly to SQL"""
    print("\n--- Phase 2: Streaming Review Ledgers (Minibatch Ingestion) ---")
    
    chunk_size = 100_000
    chunk_iter = pd.read_json(REVIEW_JSON, lines=True, chunksize=chunk_size)
    
    chunks_processed = 0
    for chunk in chunk_iter:
        # We only want reviews for our pre-filtered restaurants (this drops mechanics/plumbers natively)
        # However, SQL foreign key logic handles this if we do a quick inner join or python filter.
        # To be safe and fast, we just insert and ignore conflicts, or we can filter via pandas.
        
        # Keep only required columns
        subset = chunk[['review_id', 'business_id', 'user_id', 'stars', 'text']]
        subset.columns = ['r_id', 'b_id', 'u_id', 'stars', 'text']
        
        # Write directly to SQLite on disk!
        subset.to_sql('reviews', conn, if_exists='append', index=False, chunksize=50_000)
        
        chunks_processed += 1
        print(f"   ... Flushed {(chunks_processed * chunk_size):,}-row mark to Disk ...", end='\r', flush=True)
        
    print(f"\n [DB] Out-Of-Core review extraction perfectly completed.")

def export_autoencoder_pairs(conn):
    """Phase 3: Creates Random (Weak-Supervision) Photo:Review matches"""
    print("\n--- Phase 3: Exporting Autoencoder Cross-Modal Vectors ---")
    train_pairs = []
    val_pairs = []
    
    cur = conn.cursor()
    
    if os.path.exists(PHOTOS_JSON):
        with open(PHOTOS_JSON, 'r', encoding='utf-8') as f:
            for line in f:
                try: p_meta = json.loads(line)
                except: continue
                
                if p_meta.get('label') != 'food':
                    continue
                    
                b_id = p_meta.get('business_id')
                photo_id = p_meta.get('photo_id')
                photo_path = os.path.join(PHOTOS_DIR, f"{photo_id}.jpg")
                
                if not os.path.exists(photo_path):
                    continue
                
                # Fetch exactly 1 random review targeting this business natively from disk
                cur.execute("""
                    SELECT r.text, b.is_val_target 
                    FROM reviews r
                    JOIN businesses b ON r.b_id = b.b_id
                    WHERE r.b_id = ? 
                    ORDER BY RANDOM() LIMIT 1
                """, (b_id,))
                
                res = cur.fetchone()
                if res:
                    review_text, is_val = res
                    pair = {"image_path": photo_path, "text": review_text, "business_id": b_id}
                    if is_val == 1:
                        val_pairs.append(pair)
                    else:
                        train_pairs.append(pair)
                        
        if train_pairs:
            pd.DataFrame(train_pairs).to_parquet(TRAIN_AUTOENCODER, index=False)
            print(f" [SUCCESS] Exported {len(train_pairs)} Autoencoder Training Pairs.")
        if val_pairs:
            pd.DataFrame(val_pairs).to_parquet(VAL_AUTOENCODER, index=False)
            print(f" [SUCCESS] Exported {len(val_pairs)} Autoencoder Validation Pairs.")
    else:
        print(" [WARNING] train.json (Photos) not found. Skipping autoencoder matched exports.")

def export_regression_histories(conn):
    """Phase 4: Computes the temporal history vectors for explicit Quantile Regression"""
    print("\n--- Phase 4: Exporting Quantile Regression Histories ---")
    cur = conn.cursor()
    
    # Find active users mapping to at least 1 high-end val place and >= 5 basic train places
    query = """
        SELECT u_id
        FROM reviews r
        JOIN businesses b ON r.b_id = b.b_id
        GROUP BY u_id
        HAVING SUM(CASE WHEN b.is_val_target = 1 THEN 1 ELSE 0 END) >= 1
           AND SUM(CASE WHEN b.is_val_target = 0 THEN 1 ELSE 0 END) >= 5
    """
    cur.execute(query)
    valid_users = [row[0] for row in cur.fetchall()]
    
    regression_dataset = []
    
    for u_id in valid_users:
        # Get baseline history
        cur.execute("""
            SELECT r.b_id, r.stars FROM reviews r
            JOIN businesses b ON r.b_id = b.b_id
            WHERE r.u_id = ? AND b.is_val_target = 0
        """, (u_id,))
        hist_records = cur.fetchall()
        
        # Get target validation targets
        cur.execute("""
            SELECT r.b_id, r.stars FROM reviews r
            JOIN businesses b ON r.b_id = b.b_id
            WHERE r.u_id = ? AND b.is_val_target = 1
        """, (u_id,))
        val_records = cur.fetchall()
        
        # Ensure we just pick one regression target profile properly
        if val_records and len(hist_records) >= 5:
            target_business, target_rating = random.choice(val_records)
            
            regression_dataset.append({
                "user_id": u_id,
                "target_michelin_business": target_business,
                "target_actual_rating": target_rating,
                "historical_business_list": [h[0] for h in hist_records],
                "historical_ratings": [h[1] for h in hist_records]
            })
            
    if regression_dataset:
        with open(REGRESSION_VAL_OUT, 'w', encoding='utf-8') as f:
            json.dump(regression_dataset, f, indent=4)
        print(f" [SUCCESS] Aggregated {len(regression_dataset)} valid App-User historical profiles.")
        print(f" [SUCCESS] Wrote to {REGRESSION_VAL_OUT}")
    else:
        print(" [WARNING] Found no users strictly matching history specifications.")

def main():
    conn = init_db()
    stream_businesses(conn)
    stream_reviews(conn)
    export_autoencoder_pairs(conn)
    export_regression_histories(conn)
    conn.close()
    print("\n--- Complete Pipeline Finalized ---")

if __name__ == "__main__":
    main()
