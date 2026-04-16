"""
Cross-Modal Generalization Evaluator

Evaluates how well the cross-modal autoencoder — trained on general Yelp casual
dining — generalizes to the held-out Philadelphia high-end restaurant cohort
(price tier 3-4, stars >= 4.0). This cohort was tagged is_val_target=1 during
preprocessing and lives in toy_val_embeddings.pt.

The key question: does the 256-D compressed latent space that learned to align
image+text pairs from casual dining still produce well-aligned pairs when applied
to OOD fine-dining data?
"""

import os
import sys
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F

# Allow sibling import of CrossModalAutoencoder from the same package directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cross_modal_embeddings import CrossModalAutoencoder  # noqa: E402

# --- Paths ---
_ROOT    = os.path.abspath(os.path.join(_HERE, '..', '..'))
DATA_DIR = os.path.join(_ROOT, 'data', 'yelp_sandbox')
TOY_DIR  = os.path.join(DATA_DIR, 'toy_embeddings')

TRAIN_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_train_embeddings.pt')
VAL_EMBEDDINGS   = os.path.join(TOY_DIR, 'toy_val_embeddings.pt')
MODEL_DIR        = os.path.join(DATA_DIR, 'models')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _run_inference(
    model: CrossModalAutoencoder,
    data: list,
    device: torch.device,
    max_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Forward-pass up to max_samples items through the full autoencoder.

    Returns a dict with per-cohort metrics:
        img_latents   : (N, 256) ndarray  — compressed image latent vectors
        txt_latents   : (N, 256) ndarray  — compressed text latent vectors
        cos_sims      : (N,)     ndarray  — per-pair cosine similarity
        img_recon_mse : float             — mean image reconstruction error
        txt_recon_mse : float             — mean text reconstruction error
        alignment_mse : float             — mean latent alignment loss (MSE)
        n_samples     : int
    """
    model.eval()

    # Reproducibly subsample
    rng = random.Random(seed)
    sample = data[:]
    rng.shuffle(sample)
    sample = sample[:max_samples]

    img_latents_list, txt_latents_list = [], []
    img_recon_errs, txt_recon_errs, align_errs = [], [], []
    b_ids: list[str] = []

    with torch.no_grad():
        for item in sample:
            img_x = item['image_embedding'].flatten().float().unsqueeze(0).to(device)
            txt_x = item['text_embedding'].flatten().float().unsqueeze(0).to(device)

            img_latent, img_recon, txt_latent, txt_recon = model(img_x, txt_x)

            img_latents_list.append(img_latent.cpu())
            txt_latents_list.append(txt_latent.cpu())
            img_recon_errs.append(F.mse_loss(img_recon, img_x).item())
            txt_recon_errs.append(F.mse_loss(txt_recon, txt_x).item())
            align_errs.append(F.mse_loss(img_latent, txt_latent).item())
            b_ids.append(str(item.get('business_id', f'unk_{len(b_ids)}')))

    img_latents = torch.cat(img_latents_list).numpy()  # (N, 256)
    txt_latents = torch.cat(txt_latents_list).numpy()

    cos_sims = F.cosine_similarity(
        torch.from_numpy(img_latents),
        torch.from_numpy(txt_latents),
        dim=1,
    ).numpy()

    return {
        'img_latents':   img_latents,
        'txt_latents':   txt_latents,
        'cos_sims':      cos_sims,
        'img_recon_mse': float(np.mean(img_recon_errs)),
        'txt_recon_mse': float(np.mean(txt_recon_errs)),
        'alignment_mse': float(np.mean(align_errs)),
        'business_ids':  b_ids,
        'n_samples':     len(sample),
    }


# ---------------------------------------------------------------------------
# Discriminability
# ---------------------------------------------------------------------------

def compute_discriminability(img_latents: np.ndarray, business_ids: list[str]) -> dict:
    """
    Measures how well the 256-D image latent space separates different restaurants.

    Strategy:
      1. Group latents by business_id and compute a per-restaurant centroid
         (mean-pool all photo embeddings for that restaurant).
      2. Compute all pairwise cosine distances between restaurant centroids.
         High distance = well-separated = good discriminability.
      3. For restaurants with multiple samples, compute intra-restaurant cosine
         similarity. High similarity = stable, consistent representation.

    Returns:
        n_restaurants         : int   — unique restaurants in this cohort
        inter_restaurant_dist : float — mean pairwise cosine distance between centroids
        inter_dist_std        : float — std dev of pairwise cosine distances
        intra_restaurant_sim  : float | None — mean within-restaurant cosine sim
        discriminability_score: float — inter_dist / (1 - intra_sim); higher = better
        pairwise_distances    : ndarray — raw pairwise distance values (for histogram)
    """
    from collections import defaultdict
    from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

    # ── Group latents by restaurant ──
    groups: dict[str, list] = defaultdict(list)
    for b_id, latent in zip(business_ids, img_latents):
        groups[b_id].append(latent)

    centroids   = []
    intra_sims  = []

    for b_id, latents_list in groups.items():
        arr = np.array(latents_list)       # (n_photos, 256)
        centroids.append(arr.mean(axis=0))

        # Intra-restaurant similarity only when there are ≥ 2 photos
        if len(latents_list) >= 2:
            sim_mat = cosine_similarity(arr)   # (n, n)
            n = len(latents_list)
            upper = sim_mat[np.triu_indices(n, k=1)]
            intra_sims.extend(upper.tolist())

    centroid_array = np.array(centroids)   # (n_restaurants, 256)
    n_rest = len(centroid_array)

    # ── Pairwise cosine distances between restaurant centroids ──
    if n_rest >= 2:
        dist_mat       = cosine_distances(centroid_array)   # (n, n)
        upper_idx      = np.triu_indices(n_rest, k=1)
        pairwise_dists = dist_mat[upper_idx]
        inter_dist     = float(pairwise_dists.mean())
        inter_std      = float(pairwise_dists.std())
    else:
        pairwise_dists = np.array([0.0])
        inter_dist     = 0.0
        inter_std      = 0.0

    intra_sim = float(np.mean(intra_sims)) if intra_sims else None

    # Discriminability score: how far apart inter-restaurant centroids are
    # relative to how spread out embeddings within a single restaurant are.
    # Range: 0 (no discrimination) → ∞ (perfect). Typical good range: > 1.0
    if intra_sim is not None:
        disc_score = inter_dist / max(1e-8, 1.0 - intra_sim)
    else:
        disc_score = inter_dist   # fallback when every restaurant has 1 photo

    return {
        'n_restaurants':          n_rest,
        'inter_restaurant_dist':  inter_dist,
        'inter_dist_std':         inter_std,
        'intra_restaurant_sim':   intra_sim,
        'discriminability_score': disc_score,
        'pairwise_distances':     pairwise_dists,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_generalization_metrics(
    max_train_samples: int = 1000,
    max_val_samples: int = 1000,
) -> dict:
    """
    Load the best checkpoint and run inference on both cohorts.

    Returns a metrics dict:
        checkpoint     : str  — filename of the loaded checkpoint
        train          : dict — metrics for the Yelp general cohort
        val            : dict — metrics for the Philadelphia high-end cohort
        alignment_gap  : float — val_alignment_mse - train_alignment_mse
        cos_sim_gap    : float — val_mean_cos_sim - train_mean_cos_sim (negative = degraded)

    Raises FileNotFoundError if any required file is missing.
    """
    ckpt_path = find_best_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {MODEL_DIR}. Run 'Start Training' in the Dual Encoder tab first."
        )
    for path, label in [(TRAIN_EMBEDDINGS, 'train'), (VAL_EMBEDDINGS, 'val')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label.capitalize()} embeddings not found at:\n  {path}\n"
                "Run 'generate_embeddings_yelp.py' to create them."
            )

    device = torch.device('cpu')  # CPU for reliable Streamlit inference

    model = CrossModalAutoencoder.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()

    print("Loading training embeddings from disk (may take ~30s for the first run)...")
    train_raw = torch.load(TRAIN_EMBEDDINGS, weights_only=False)
    print("Loading validation embeddings...")
    val_raw   = torch.load(VAL_EMBEDDINGS,   weights_only=False)

    print(f"Running inference: {min(max_train_samples, len(train_raw))} train samples, "
          f"{min(max_val_samples, len(val_raw))} val samples...")

    train_m = _run_inference(model, train_raw, device, max_samples=max_train_samples)
    val_m   = _run_inference(model, val_raw,   device, max_samples=max_val_samples)

    print("Computing discriminability metrics...")
    train_disc = compute_discriminability(train_m['img_latents'], train_m['business_ids'])
    val_disc   = compute_discriminability(val_m['img_latents'],   val_m['business_ids'])

    return {
        'checkpoint':    os.path.basename(ckpt_path),
        'train':         train_m,
        'val':           val_m,
        'train_disc':    train_disc,
        'val_disc':      val_disc,
        'alignment_gap': val_m['alignment_mse'] - train_m['alignment_mse'],
        'cos_sim_gap':   val_m['cos_sims'].mean() - train_m['cos_sims'].mean(),
    }


def compute_tsne(
    train_latents: np.ndarray,
    val_latents: np.ndarray,
    n_per_cohort: int = 400,
) -> tuple[np.ndarray, list[str], int]:
    """
    Run t-SNE on a combined sample of image latents from both cohorts.

    Returns:
        coords     : (N, 2) ndarray — 2-D t-SNE coordinates
        labels     : list[str]     — cohort label per point
        n_train    : int           — number of train points (for split indexing)
    """
    from sklearn.manifold import TSNE

    n_train = min(n_per_cohort, len(train_latents))
    n_val   = min(n_per_cohort, len(val_latents))

    combined = np.vstack([train_latents[:n_train], val_latents[:n_val]])
    labels   = (
        ['Yelp General (Train)'] * n_train
        + ['Philadelphia High-End (Val)'] * n_val
    )

    perplexity = min(30, max(5, len(combined) // 4))
    tsne   = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=500)
    coords = tsne.fit_transform(combined)

    return coords, labels, n_train


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("--- Cross-Modal Generalization Evaluation ---")
    m = compute_generalization_metrics()

    train_m = m['train']
    val_m   = m['val']

    print(f"\nCheckpoint : {m['checkpoint']}")
    print(f"Train samples : {train_m['n_samples']}  |  Val samples : {val_m['n_samples']}")
    print(f"\n{'Metric':<30} {'Train':>10} {'Val':>10}  {'Gap':>10}")
    print("-" * 64)

    rows = [
        ("Alignment MSE",        train_m['alignment_mse'], val_m['alignment_mse']),
        ("Image Recon MSE",      train_m['img_recon_mse'], val_m['img_recon_mse']),
        ("Text Recon MSE",       train_m['txt_recon_mse'], val_m['txt_recon_mse']),
        ("Mean Cosine Sim",      train_m['cos_sims'].mean(), val_m['cos_sims'].mean()),
    ]
    for name, t, v in rows:
        print(f"{name:<30} {t:>10.4f} {v:>10.4f}  {v - t:>+10.4f}")

    print(f"\n{'Alignment gap (val - train)':>42} : {m['alignment_gap']:+.4f}")
    print(f"{'Cosine sim gap (val - train)':>42} : {m['cos_sim_gap']:+.4f}")

    td, vd = m['train_disc'], m['val_disc']
    print(f"\n{'Discriminability':^64}")
    print("-" * 64)
    print(f"{'Metric':<30} {'Train':>10} {'Val':>10}")
    print("-" * 64)
    print(f"{'Unique restaurants':<30} {td['n_restaurants']:>10} {vd['n_restaurants']:>10}")
    print(f"{'Inter-restaurant dist':<30} {td['inter_restaurant_dist']:>10.4f} {vd['inter_restaurant_dist']:>10.4f}")
    print(f"{'Intra-restaurant sim':<30} {str(round(td['intra_restaurant_sim'], 4)) if td['intra_restaurant_sim'] else 'N/A':>10} "
          f"{str(round(vd['intra_restaurant_sim'], 4)) if vd['intra_restaurant_sim'] else 'N/A':>10}")
    print(f"{'Discriminability score':<30} {td['discriminability_score']:>10.4f} {vd['discriminability_score']:>10.4f}")
