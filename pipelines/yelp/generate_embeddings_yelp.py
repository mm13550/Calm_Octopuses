"""
PyTorch Raw Offline Embedding Generator (Disjoint Mode)

Processes the Yelp dataset, pairing Images and Texts utilizing
local SQLite data. Natively streams through HuggingFace
models (DistilBERT + ResNet50) computing massive matrix layers
into compressed PyTorch embeddings to feed a future
Auto-Encoder Dual-Network loop.
"""
import os
import json
import sqlite3
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Path Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'yelp_sandbox'))
DB_PATH = os.path.join(DATA_DIR, 'yelp_relations.db')
PHOTOS_JSON = os.path.join(DATA_DIR, 'train.json')
PHOTOS_DIR = os.path.join(DATA_DIR, 'train', 'train') # Nested Kaggle Extractor dir

TOY_DIR = os.path.join(DATA_DIR, 'toy_embeddings')
os.makedirs(TOY_DIR, exist_ok=True)
TRAIN_EMBEDDINGS_OUT = os.path.join(TOY_DIR, 'toy_train_embeddings.pt')
VAL_EMBEDDINGS_OUT = os.path.join(TOY_DIR, 'toy_val_embeddings.pt')

# Performance Restrictions
BATCH_LIMIT = 10000

def get_device():
    """Identifies the optimal local Hardware processor natively"""
    if torch.cuda.is_available(): 
        return torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        return torch.device('mps')
    return torch.device('cpu')

def main():
    device = get_device()
    print(f"--- Booting PyTorch Embedding Extractor (Device: {device}) ---")

    print("> Loading CLIP Model [512-D]...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    print("> Connecting to Database for Relational Pairings...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    train_data = [] # Lists of dictionary tensors
    val_data = []
    
    if not os.path.exists(PHOTOS_JSON):
        print("ERROR: train.json photos mapping file missing! Ensure Kaggle Sandbox extracted completely.")
        return

    print(f"\n--- Crunching Offline PyTorch Arrays (Cap: {BATCH_LIMIT}) ---")
    processed_count = 0
    with open(PHOTOS_JSON, 'r', encoding='utf-8') as f:
        # Wrap native file output in tqdm for visual load progression mapping
        for line in tqdm(f, desc="Synthesizing Vectors..."):
            if processed_count >= BATCH_LIMIT:
                break
                
            try: p_meta = json.loads(line)
            except: continue
            
            if p_meta.get('label') != 'food': continue
            
            b_id = p_meta.get('business_id')
            photo_id = p_meta.get('photo_id')
            photo_path = os.path.join(PHOTOS_DIR, f"{photo_id}.jpg")
            
            if not os.path.exists(photo_path): continue
            
            # Fetch relational text safely bypassing gigabytes of python RAM
            cur.execute("""
                SELECT r.text, b.is_val_target 
                FROM reviews r JOIN businesses b ON r.b_id = b.b_id
                WHERE r.b_id = ? ORDER BY RANDOM() LIMIT 1
            """, (b_id,))
            res = cur.fetchone()
            
            if res:
                review_text, is_val = res
                
                try:
                    # 1. Open and Validate Image
                    img = Image.open(photo_path).convert('RGB')
                    
                    # 2. Process both Text and Image using CLIP (Truncating text to max 77 tokens)
                    inputs = processor(
                        text=[review_text[:512]], 
                        images=img, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(device)
                    
                    # Mathematical forward-pass without updating weights (Inferencing mode)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        img_emb = outputs.image_embeds.flatten()
                        text_emb = outputs.text_embeds.flatten()
                        
                        # L2-normalize to align with LanceDB expectation
                        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                        
                    # Detach from VRAM natively and store exclusively on standard CPU for array saves
                    item = {
                        "image_embedding": img_emb.cpu(),
                        "text_embedding": text_emb.cpu(),
                        "business_id": b_id
                    }
                    if is_val == 1: val_data.append(item)
                    else: train_data.append(item)
                    
                    processed_count += 1
                except Exception as e:
                    # Catch broken images naturally instead of crashing
                    pass
                    
    conn.close()
    print(f"\n--- Execution Finalized ---")
    print(f"Generated {len(train_data)} Train Arrays.")
    print(f"Generated {len(val_data)} Val Arrays.")
    
    # Export the Tensors offline (PyTorch loading these binaries takes <0.1 seconds later!)
    if train_data:
        torch.save(train_data, TRAIN_EMBEDDINGS_OUT)
        print(f"Saved -> {TRAIN_EMBEDDINGS_OUT}")
    if val_data:
        torch.save(val_data, VAL_EMBEDDINGS_OUT)
        print(f"Saved -> {VAL_EMBEDDINGS_OUT}")

if __name__ == "__main__":
    main()
