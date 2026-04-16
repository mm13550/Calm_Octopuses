"""
Restaurant Embedding Aggregator
================================

Passes the raw 2048-D image / 768-D text paired embeddings through the 
trained CrossModalAutoencoder to extract the aligned 256-D latent vectors.
It then groups these vectors by `business_id` and performs mean-pooling
to generate a single, stable Taste Vector per restaurant.

These finalized vectors will be used downstream by the regression head.
"""

import os
import sys
import glob
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Allow sibling import of CrossModalAutoencoder
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cross_modal_embeddings import CrossModalAutoencoder

_ROOT    = os.path.abspath(os.path.join(_HERE, '..', '..'))
DATA_DIR = os.path.join(_ROOT, 'data', 'yelp_sandbox')
TOY_DIR  = os.path.join(DATA_DIR, 'toy_embeddings')

TRAIN_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_train_embeddings.pt')
VAL_EMBEDDINGS   = os.path.join(TOY_DIR, 'toy_val_embeddings.pt')
MODEL_DIR        = os.path.join(DATA_DIR, 'models')

def find_best_checkpoint() -> str | None:
    """Return the checkpoint path with the lowest val_loss, or None."""
    ckpts = glob.glob(os.path.join(MODEL_DIR, 'best_model-*.ckpt'))
    if not ckpts:
        return None

    def _val_loss(path: str) -> float:
        try:
            for part in os.path.basename(path).replace('.ckpt', '').split('-'):
                if part.startswith('val_loss='):
                    return float(part.split('=')[1])
        except Exception:
            pass
        return float('inf')

    return min(ckpts, key=_val_loss)

def aggregate_embeddings(model: CrossModalAutoencoder, data: list, device: torch.device) -> dict:
    """
    Forward-passes through the encoder, then mean-pools multiple photos/reviews 
    for the same business_id into a single 256-D centroid.
    
    Returns: Dict[str, np.ndarray] mapping business_id -> 256-D vector.
    """
    model.eval()
    groups = defaultdict(list)
    
    with torch.no_grad():
        for item in tqdm(data, desc="Inferencing latents"):
            img_x = item['image_embedding'].flatten().float().unsqueeze(0).to(device)
            txt_x = item['text_embedding'].flatten().float().unsqueeze(0).to(device)
            
            img_latent, _, txt_latent, _ = model(img_x, txt_x)
            
            # Using the average of the aligned modalities to form a unified taste vector
            # (they are trained to align, so averaging improves robustness)
            unified_latent = (img_latent + txt_latent) / 2.0
            
            b_id = str(item.get('business_id', f'unknown_{len(groups)}'))
            groups[b_id].append(unified_latent.cpu().numpy())
            
    final_dict = {}
    for b_id, latents in groups.items():
        arr = np.concatenate(latents, axis=0) # Shape: (N, 256)
        final_dict[b_id] = np.mean(arr, axis=0) # Mean-pooling over N inputs to (256,)
        
    return final_dict

def main():
    print("--- Aggregating Restaurant Embeddings ---")
    
    # Check what device to run on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt_path = find_best_checkpoint()
    if not ckpt_path:
        print("No checkpoint found. Please train the Autoencoder first.")
        return
        
    print(f"Loading checkpoint: {os.path.basename(ckpt_path)}")
    model = CrossModalAutoencoder.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    
    for in_path, out_name in [
        (TRAIN_EMBEDDINGS, 'toy_restaurant_embeddings_train.pt'),
        (VAL_EMBEDDINGS,   'toy_restaurant_embeddings_val.pt')
    ]:
        if not os.path.exists(in_path):
            print(f"File not found, skipping: {in_path}")
            continue
            
        print(f"\nProcessing {os.path.basename(in_path)}...")
        data = torch.load(in_path, weights_only=False)
        
        aggregated_dict = aggregate_embeddings(model, data, device)
        
        out_path = os.path.join(TOY_DIR, out_name)
        print(f"Saving {len(aggregated_dict)} unique restaurant embeddings to {out_path}...")
        torch.save(aggregated_dict, out_path)
        
    print("\nAggregation complete!")

if __name__ == "__main__":
    main()
