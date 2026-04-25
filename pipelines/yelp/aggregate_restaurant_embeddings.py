"""
pipelines/yelp/aggregate_restaurant_embeddings.py
==================================================
Aggregates raw per-photo CLIP embeddings into a single 512-D restaurant
taste vector, using the **same fusion strategy** as the Michelin pipeline
(``pipelines/build_restaurant_profiles.py``) so that training and inference
embeddings are in the same geometric space.

Fusion strategy (mirrors Michelin ``weighted_fuse``):
  - Review text CLIP  →  weight 0.70   (matches menu 0.35 + review 0.35)
  - Food image CLIP   →  weight 0.30   (matches food 0.20 + interior 0.10)

Each modality is mean-pooled across all photos/reviews **independently**,
L2-normalised, then fused.  The fused vector is L2-normalised again.
The final norm is exactly 1.0, matching the Michelin profile norm.
"""

import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, '..', '..'))
DATA_DIR = os.path.join(_ROOT, 'data', 'yelp_sandbox')
TOY_DIR  = os.path.join(DATA_DIR, 'toy_embeddings')

TRAIN_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_train_embeddings.pt')
VAL_EMBEDDINGS   = os.path.join(TOY_DIR, 'toy_val_embeddings.pt')

# Fusion weights — must match build_restaurant_profiles.py proportions
_TEXT_WEIGHT  = 0.70   # review text  (= menu 0.35 + review 0.35)
_IMAGE_WEIGHT = 0.30   # food images  (= food  0.20 + interior 0.10)


# ── Helpers ────────────────────────────────────────────────────────────────

def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    """Return *arr* L2-normalised to exactly unit length."""
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        raise ValueError("Zero vector cannot be normalised.")
    return arr / norm


def _mean_pool(vecs: list[np.ndarray]) -> np.ndarray:
    """Mean-pool a list of vectors and L2-normalise the result."""
    return _l2_normalise(np.stack(vecs, axis=0).mean(axis=0))


def _weighted_fuse(components: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Weighted sum of modality vectors, renormalised by available weight, then L2-normalised.

    Mirrors ``weighted_fuse`` in ``build_restaurant_profiles.py`` so that
    restaurants with only a subset of modalities are still handled gracefully.
    """
    valid = [(vec, w) for vec, w in components if vec is not None and w > 0]
    if not valid:
        raise ValueError("No valid modality vectors to fuse.")
    total_w = sum(w for _, w in valid)
    acc = np.zeros(valid[0][0].shape, dtype=np.float64)
    for vec, w in valid:
        acc += vec.astype(np.float64) * (w / total_w)
    return _l2_normalise(acc.astype(np.float32))


# ── Core aggregation ───────────────────────────────────────────────────────

def aggregate_embeddings(data: list) -> dict:
    """
    Aggregate per-photo CLIP embeddings into one 512-D vector per restaurant.

    Strategy (matches Michelin pipeline):
    1. Group image and text embeddings separately by ``business_id``.
    2. Mean-pool + L2-normalise each modality independently.
    3. Weighted-fuse (text=0.70, image=0.30) and L2-normalise the result.

    Parameters
    ----------
    data : list[dict]
        Each dict has keys ``image_embedding`` (Tensor, 512-D),
        ``text_embedding`` (Tensor, 512-D), and ``business_id`` (str).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of ``business_id`` to a float32 unit vector of shape (512,).
    """
    text_groups  = defaultdict(list)
    image_groups = defaultdict(list)

    for item in tqdm(data, desc="Grouping CLIP latents by modality"):
        b_id = str(item.get('business_id', f'unknown_{id(item)}'))

        img_x = item['image_embedding'].flatten().float().numpy()
        txt_x = item['text_embedding'].flatten().float().numpy()

        # L2-normalise each raw embedding first (CLIP outputs are already normed,
        # but guard against any numerical drift from earlier pipeline steps)
        image_groups[b_id].append(_l2_normalise(img_x))
        text_groups[b_id].append(_l2_normalise(txt_x))

    all_ids = set(text_groups.keys()) | set(image_groups.keys())
    final_dict = {}

    for b_id in tqdm(all_ids, desc="Fusing modalities (text 0.70 / image 0.30)"):
        text_vec  = _mean_pool(text_groups[b_id])  if text_groups[b_id]  else None
        image_vec = _mean_pool(image_groups[b_id]) if image_groups[b_id] else None

        fused = _weighted_fuse([
            (text_vec,  _TEXT_WEIGHT),
            (image_vec, _IMAGE_WEIGHT),
        ])
        final_dict[b_id] = fused

    return final_dict


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    """
    Run the restaurant embedding aggregation pipeline.

    Loads raw CLIP paired embeddings from ``toy_train_embeddings.pt`` and
    ``toy_val_embeddings.pt``, aggregates each business's image and text vectors
    into a single 512-D fused unit vector using the Michelin-aligned 0.70/0.30
    text/image weighting, and saves the result as
    ``toy_restaurant_embeddings_{train|val}.pt``.
    """
    print("--- Aggregating Restaurant Embeddings (Michelin-aligned fusion) ---")
    print(f"  Text weight : {_TEXT_WEIGHT:.2f}  (review text CLIP)")
    print(f"  Image weight: {_IMAGE_WEIGHT:.2f}  (food image CLIP)")

    for in_path, out_name in [
        (TRAIN_EMBEDDINGS, 'toy_restaurant_embeddings_train.pt'),
        (VAL_EMBEDDINGS,   'toy_restaurant_embeddings_val.pt'),
    ]:
        if not os.path.exists(in_path):
            print(f"File not found, skipping: {in_path}")
            continue

        print(f"\nLoading {os.path.basename(in_path)}...")
        data = torch.load(in_path, weights_only=False)
        print(f"  {len(data)} photo-level records loaded.")

        aggregated_dict = aggregate_embeddings(data)

        # Sanity-check norms
        norms = [np.linalg.norm(v) for v in aggregated_dict.values()]
        print(f"  {len(aggregated_dict)} unique restaurants aggregated.")
        print(f"  Vector norm  mean={np.mean(norms):.6f}  std={np.std(norms):.6f}  "
              f"min={np.min(norms):.6f}  max={np.max(norms):.6f}")

        out_path = os.path.join(TOY_DIR, out_name)
        torch.save(aggregated_dict, out_path)
        print(f"  Saved -> {out_path}")

    print("\nAggregation complete!")


if __name__ == "__main__":
    main()
