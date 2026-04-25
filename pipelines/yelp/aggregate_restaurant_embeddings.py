"""
Restaurant Embedding Aggregator
================================

Takes the raw 512-D native CLIP image / 512-D text paired embeddings
and merges them directly. It then groups these vectors by `business_id` 
and performs mean-pooling to generate a single, stable Taste Vector per restaurant.

These finalized vectors will be used downstream by the regression head.
"""

import os
import sys
import glob
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# Replicate paths locally since we removed the larger import block above
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, '..', '..'))
DATA_DIR = os.path.join(_ROOT, 'data', 'yelp_sandbox')
TOY_DIR  = os.path.join(DATA_DIR, 'toy_embeddings')

TRAIN_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_train_embeddings.pt')
VAL_EMBEDDINGS   = os.path.join(TOY_DIR, 'toy_val_embeddings.pt')

def aggregate_embeddings(data: list) -> dict:
    """
    Directly averages native CLIP modalities instead of relying on a trained encoder,
    then mean-pools multiple photos/reviews for the same business_id into a 
    single 512-D centroid.
    
    Returns: Dict[str, np.ndarray] mapping business_id -> 512-D vector.
    """
    groups = defaultdict(list)
    
    for item in tqdm(data, desc="Aggregating CLIP latents"):
        img_x = item['image_embedding'].flatten().float()
        txt_x = item['text_embedding'].flatten().float()
        
        # Using the average of the naturally aligned CLIP modalities 
        # to form a unified taste vector robustly
        unified_latent = (img_x + txt_x) / 2.0
        
        # Re-normalize to unit length for consistency across all taste vectors
        unified_latent = unified_latent / (unified_latent.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        
        b_id = str(item.get('business_id', f'unknown_{len(groups)}'))
        groups[b_id].append(unified_latent.unsqueeze(0).numpy())
            
    final_dict = {}
    for b_id, latents in groups.items():
        arr = np.concatenate(latents, axis=0) # Shape: (N, 512)
        final_dict[b_id] = np.mean(arr, axis=0) # Mean-pooling over N inputs to (512,)
        
    return final_dict

def main():
    """
        Run the restaurant embedding aggregation pipeline.\n\n    Loads raw CLIP paired embeddings from `toy_train_embeddings.pt` and\n    `toy_val_embeddings.pt`, aggregates each business's vectors into a single\n    512-D mean-pooled unit vector, and saves the result as\n    `toy_restaurant_embeddings_{train|val}.pt`.
    """
    print("--- Aggregating Restaurant Embeddings ---")
    

    
    for in_path, out_name in [
        (TRAIN_EMBEDDINGS, 'toy_restaurant_embeddings_train.pt'),
        (VAL_EMBEDDINGS,   'toy_restaurant_embeddings_val.pt')
    ]:
        if not os.path.exists(in_path):
            print(f"File not found, skipping: {in_path}")
            continue
            
        print(f"\nProcessing {os.path.basename(in_path)}...")
        data = torch.load(in_path, weights_only=False)
        
        aggregated_dict = aggregate_embeddings(data)
        
        out_path = os.path.join(TOY_DIR, out_name)
        print(f"Saving {len(aggregated_dict)} unique restaurant embeddings to {out_path}...")
        torch.save(aggregated_dict, out_path)
        
    print("\nAggregation complete!")

if __name__ == "__main__":
    main()
