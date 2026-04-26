"""
Mixture Density Network (Laplace MoG) for Rating Intervals
Predicts Mixture of Laplace components for user-restaurant pairs.
"""

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
_EMBEDDINGS_CACHE = {}

def _load_cached_embeddings(paths):
    global _EMBEDDINGS_CACHE
    merged = {}
    for p in paths:
        if p not in _EMBEDDINGS_CACHE:
            print(f"Loading embeddings from {os.path.basename(p)}...")
            _EMBEDDINGS_CACHE[p] = torch.load(p, map_location='cpu', weights_only=False)
        merged.update(_EMBEDDINGS_CACHE[p])
    return merged

from pytorch_lightning.callbacks import EarlyStopping

def get_michelin_centroid(embeddings_dict):
    """Computes the average embedding vector across all restaurant embeddings."""
    embs = np.array(list(embeddings_dict.values()))
    return np.mean(embs, axis=0)

class UserRestaurantDataset(Dataset):
    """
    PyTorch Dataset that pairs a user's rating history with an unseen target restaurant.

    Each item is a feature vector built from:
    - A 512-D *user taste vector*: rating-weighted mean of the user's historical
      restaurant CLIP embeddings.
    - A 512-D *target vector*: the CLIP embedding of the restaurant to be predicted.
    - A 1-D *scalar*: the user's mean historical rating (absolute preference signal).

    Records whose target restaurant or whose entire history lack embeddings are
    silently dropped during ``__init__``.
    """
    def __init__(self, json_path, embeddings_paths, metadata_paths=None):
        """
        Initialise the dataset.

        Parameters
        ----------
        json_path : str
            Path to the regression JSON file (list of user-rating dicts).
        embeddings_paths : list[str]
            One or more ``.pt`` files containing ``{business_id: np.ndarray}``
            restaurant embedding dicts.  Multiple files are merged so that train
            and val restaurant embeddings are both available at once.
        """
        super().__init__()
        print(f"Loading dataset from {os.path.basename(json_path)}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.restaurant_embeddings = _load_cached_embeddings(embeddings_paths)
        print(f"Total merged embeddings available: {len(self.restaurant_embeddings)}")
        
        self.metadata = {}
        if metadata_paths:
            for mpath in metadata_paths:
                if os.path.exists(mpath):
                    print(f"Loading metadata from {os.path.basename(mpath)}")
                    with open(mpath, 'r') as f:
                        self.metadata.update(json.load(f))
        
        # Filter out records where target or any historical business isn't in embeddings
        self.valid_data = []
        # Compute local training centroid for alignment
        if self.restaurant_embeddings:
            all_vecs = np.stack(list(self.restaurant_embeddings.values()))
            self.centroid = torch.from_numpy(all_vecs.mean(axis=0)).float()
        else:
            self.centroid = None
        for row in self.data:
            target_id = row['target_michelin_business']
            if target_id not in self.restaurant_embeddings:
                continue
                
            hist_ids = row['historical_business_list']
            # Only keep the historical businesses that we actually have embeddings for
            valid_hist = [i for i in range(len(hist_ids)) if hist_ids[i] in self.restaurant_embeddings]
            
            if len(valid_hist) == 0:
                continue
                
            self.valid_data.append({
                'target_id': target_id,
                'target_rating': row['target_actual_rating'],
                'hist_ids': [hist_ids[i] for i in valid_hist],
                'hist_ratings': [row['historical_ratings'][i] for i in valid_hist]
            })
            
        print(f"Filtered {len(self.data)} -> {len(self.valid_data)} valid items with embeddings.")

    def __len__(self):
        """Return the number of valid (user, target-restaurant) pairs in the dataset."""
        return len(self.valid_data)
        
    def __getitem__(self, idx):
        """
        Return a single training sample as a tuple ``(feature_vec, rating, sample_weight)``.
        """
        row = self.valid_data[idx]
        
        # 1. Target Embedding (512-D) — centroid subtraction + L2-norm
        target_raw = self.restaurant_embeddings[row['target_id']]
        target_vec = torch.from_numpy(target_raw).float()
        if self.centroid is not None:
            target_vec = target_vec - self.centroid
        target_vec = target_vec / (torch.linalg.norm(target_vec) + 1e-9)
        
        # 2. User Taste Embeddings (Dual: Like & Dislike)
        like_vecs, like_weights = [], []
        dislike_vecs, dislike_weights = [], []
        
        for b_id, r in zip(row['hist_ids'], row['hist_ratings']):
            raw_vec = torch.from_numpy(self.restaurant_embeddings[b_id]).float()
            vec = raw_vec - self.centroid if self.centroid is not None else raw_vec
            vec = vec / (torch.linalg.norm(vec) + 1e-9)
            if r >= 3.5:
                w = r / 5.0
                like_vecs.append(vec * w)
                like_weights.append(w)
            else:
                w = (6.0 - r) / 5.0 
                dislike_vecs.append(vec * w)
                dislike_weights.append(w)

        if like_vecs:
            user_like_vec = torch.stack(like_vecs).sum(dim=0) / sum(like_weights)
            user_like_vec = user_like_vec / (torch.linalg.norm(user_like_vec) + 1e-9)
        else:
            user_like_vec = torch.zeros(512)
            
        if dislike_vecs:
            user_dislike_vec = torch.stack(dislike_vecs).sum(dim=0) / sum(dislike_weights)
            user_dislike_vec = user_dislike_vec / (torch.linalg.norm(user_dislike_vec) + 1e-9)
        else:
            user_dislike_vec = torch.zeros(512)
        
        mean_hist_rating = sum(row['hist_ratings']) / len(row['hist_ratings'])
        scalar_feature = torch.tensor([mean_hist_rating], dtype=torch.float32)

        # 3. Target Metadata (3-D)
        meta = self.metadata.get(row['target_id'], {'stars': 4.0, 'review_count': 1, 'price': 2})
        t_stars = torch.tensor([meta.get('stars', 4.0) / 5.0], dtype=torch.float32)
        t_reviews = torch.tensor([torch.log1p(torch.tensor(meta.get('review_count', 1), dtype=torch.float32)) / 10.0], dtype=torch.float32)
        t_price = torch.tensor([meta.get('price', 2) / 4.0], dtype=torch.float32)
        scalar_features = torch.cat([scalar_feature, t_stars, t_reviews, t_price])

        feature_vec = torch.cat([user_like_vec, user_dislike_vec, target_vec, scalar_features])
        
        raw_count = len(row['hist_ids'])
        sample_weight = torch.log1p(torch.tensor(raw_count, dtype=torch.float32))
        
        return feature_vec, torch.tensor(row['target_rating'], dtype=torch.float32), sample_weight

def mixture_laplace_nll_loss(mus, log_sigmas, pi_logits, y_true, sample_weights=None, sharpness_alpha=0.8, entropy_beta=0.05, coverage_gamma=0.5):
    """
    Negative Log Likelihood of a Mixture of Laplaces.
    """
    y_true = y_true.unsqueeze(1) # (batch, 1)
    
    # Stability Fix 1: Use log_softmax directly
    log_pi = torch.log_softmax(pi_logits, dim=1) # (batch, K)
    pis = torch.exp(log_pi)
    
    # Stabilized Bounded scale
    log_sigmas = torch.clamp(log_sigmas, min=-3.5, max=-0.7)
    sigmas = torch.exp(log_sigmas)
    
    # Laplace PDF components: 1/(2*b) * exp(-|y-mu| / b)
    # Computation in log space: -log(2) - log(b) - |y-mu| / b
    # ln(2) approx 0.6931
    log_component_pdfs = -0.6931 - log_sigmas - torch.abs(y_true - mus) / (sigmas + 1e-7)
    
    # Combined log likelihood using logsumexp trick
    total_log_likelihood = torch.logsumexp(log_pi + log_component_pdfs, dim=1)
    nll_loss = -total_log_likelihood.mean()
    
    # --- Regularization 1: Sharpness (Balanced Width Penalty) ---
    width_penalty = (pis * sigmas).sum(dim=1).mean()
    
    # --- Regularization 2: Entropy (Connectivity Penalty) ---
    entropy = -(pis * log_pi).sum(dim=1).mean()
    
    # --- Regularization 3: Coverage (Soft 95% HDR Penalty) ---
    # Closed-form Laplace CDF: F(y|mu,b) = 0.5 + 0.5*sign(y-mu)*(1 - exp(-|y-mu|/b))
    # Mixture CDF at y_true: sum_k pi_k * F_k(y_true) -> (batch,)
    diff = y_true - mus  # (batch, K)
    laplace_cdf = 0.5 + 0.5 * torch.sign(diff) * (1.0 - torch.exp(-torch.abs(diff) / (sigmas + 1e-7)))
    mix_cdf = (pis * laplace_cdf).sum(dim=1)  # (batch,)
    # Hinge penalty: fire only when y_true is outside the central 95% mass
    coverage_penalty = (F.relu(0.025 - mix_cdf) + F.relu(mix_cdf - 0.975)).mean()
    
    # Apply sample weights
    if sample_weights is not None:
        total_loss = torch.mean((-total_log_likelihood) * sample_weights) + sharpness_alpha * width_penalty + entropy_beta * entropy + coverage_gamma * coverage_penalty
    else:
        total_loss = nll_loss + sharpness_alpha * width_penalty + entropy_beta * entropy + coverage_gamma * coverage_penalty
        
    return total_loss


class MDNScorer(pl.LightningModule):
    """
    Mixture Density Network that outputs a Laplace mixture over the [1, 5] rating range.

    Architecture: MLP backbone → (mu, log_sigma, pi_logit) heads for K components.
    Mus are bounded to [0.9, 5.1] via a sigmoid gate to prevent extreme extrapolation.
    Log-sigmas are clamped to [-3.5, 0.5] for numerical stability.
    """
    def __init__(self, input_dim=1540, hidden_dims=[512, 1024, 1024, 512], k=3, lr=8e-4, sharpness_alpha=0.8, entropy_beta=0.02, coverage_gamma=0.5):
        """
        Initialise MDNScorer.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the concatenated input feature vector
            (user_vec + target_vec + scalar = 512 + 512 + 1 = 1025 by default).
        hidden_dims : list[int]
            Sizes of successive hidden layers.  Each layer is followed by
            BatchNorm1d, ReLU, and Dropout(0.2).
        k : int
            Number of Laplace mixture components.
        lr : float
            Adam learning rate.
        sharpness_alpha : float
            Weight of the sharpness (width) penalty in the composite loss.
        entropy_beta : float
            Weight of the entropy penalty; discourages the model from placing
            all mass in a single mixture component (connectivity regulariser).
        """
        super().__init__()
        self.save_hyperparameters()
        self.k = k
        self.sharpness_alpha = sharpness_alpha
        self.entropy_beta = entropy_beta
        self.coverage_gamma = coverage_gamma
        
        layers = []
        prev_dim = input_dim + 2
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
            
        # We predict K sets of (mu, log_b, pi_logit)
        self.final_layer = nn.Linear(prev_dim, 3 * k)
        
        # Custom initialization for diverse centers
        with torch.no_grad():
            self.final_layer.bias.fill_(0)
            for i in range(k):
                # Spread mus: k=0 -> 2, k=1 -> 3.5, k=2 -> 4.8 (logits)
                self.final_layer.bias[i] = -2.0 + (i * 2.0) 
                # Nuclear Sharp Initialization: log(0.08) approx -2.5
                self.final_layer.bias[k + i] = -2.5
                # Mixing weights: starts uniform
                self.final_layer.bias[2*k + i] = 0.0
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: [batch, 1540]
        # Structure: [512 Like, 512 Dislike, 512 Target, 4 Scalars]
        #
        # Dropout policy (REGRESSION_TUNING.md):
        #   - Restaurant features (target): NO dropout — always present
        #   - Interaction features (sim_like, sim_dis): NO dropout — always present
        #   - User features (u_like, u_dis): 25% dropout together
        #   - Metadata scalars: 25% dropout
        
        u_like = x[:, 0:512]
        u_dis  = x[:, 512:1024]
        target = x[:, 1024:1536]   # Never dropped
        meta   = x[:, 1536:1540]
        
        # 1. Interaction features — computed before dropout so sims always reflect
        #    the true cosine similarity between user history and the restaurant
        sim_like = (u_like * target).sum(dim=1, keepdim=True)  # Never dropped
        sim_dis  = (u_dis  * target).sum(dim=1, keepdim=True)  # Never dropped
        
        if self.training:
            # 2a. User feature dropout: zero out BOTH towers together 25% of the time
            user_mask = (torch.rand(x.shape[0], 1, device=x.device) > 0.25).float()
            u_like = u_like * user_mask
            u_dis  = u_dis  * user_mask
            
            # 2b. Metadata dropout: zero out all 4 scalars 25% of the time
            meta_mask = (torch.rand(x.shape[0], 1, device=x.device) > 0.25).float()
            meta = meta * meta_mask
            
        # 3. Concatenate all signals (Total: 1542 dims)
        x_new = torch.cat([u_like, u_dis, target, meta, sim_like, sim_dis], dim=1)
            
        features = self.mlp(x_new)
        out = self.final_layer(features)
        
        # Split into (batch, K) components
        mus_raw        = out[:, :self.k]
        log_sigmas_raw = out[:, self.k : 2*self.k]
        pi_logits      = out[:, 2*self.k:]
        
        # Bounded means with breathing room
        mus = 0.9 + 4.2 * torch.sigmoid(mus_raw)
        
        # Low-temperature: tighter clamp on log_sigma forces sharper distributions
        log_sigmas = torch.clamp(log_sigmas_raw, min=-3.5, max=-0.7)
        
        return mus, log_sigmas, pi_logits
        
    def training_step(self, batch, batch_idx):
        """Compute the weighted MDN loss for one training mini-batch and log it."""
        x, y, w = batch
        mus, log_sigmas, pi_logits = self(x)
        loss = mixture_laplace_nll_loss(
            mus, log_sigmas, pi_logits, y,
            sample_weights=w,
            sharpness_alpha=self.sharpness_alpha,
            entropy_beta=self.entropy_beta,
            coverage_gamma=self.coverage_gamma
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Compute validation MDN loss and MAE; log both for early-stopping and monitoring."""
        x, y, w = batch
        mus, log_sigmas, pi_logits = self(x)
        loss = mixture_laplace_nll_loss(
            mus, log_sigmas, pi_logits, y,
            sample_weights=w,
            sharpness_alpha=self.sharpness_alpha,
            entropy_beta=self.entropy_beta,
            coverage_gamma=self.coverage_gamma
        )
        self.log("val_loss", loss, prog_bar=True)
        
        # MAE vs the weighted mean of locations
        pis = torch.softmax(pi_logits, dim=1)
        expected_mu = (mus * pis).sum(dim=1)
        mae = F.l1_loss(expected_mu, y)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Return the Adam optimiser with the learning rate defined at construction."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


import pandas as pd

def evaluate_regression(train_json, test_json, train_emb, test_emb, max_epochs=20):
    """
    Called by app.py: 
    1. Trains MDNScorer locally on the generic training payload.
    2. Zero-shot inference against the Michelin testing payload.
    3. Returns DataFrames mapped for Streamlit visualizations.
    """
    print("--- Localizing Regression Context ---")
    train_ds = UserRestaurantDataset(train_json, [train_emb, test_emb])
    if len(train_ds) == 0:
        return None, None
        
    test_ds = UserRestaurantDataset(test_json, [train_emb, test_emb])
    
    # 80/20 train/val structural split
    t_size = int(0.8 * len(train_ds))
    v_size = len(train_ds) - t_size
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [t_size, v_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_sub, batch_size=128, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_sub, batch_size=128, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, num_workers=0)
    
    model = MDNScorer()
    trainer = pl.Trainer(accelerator="cpu", devices=1, enable_progress_bar=True, num_sanity_val_steps=0, 
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        enable_checkpointing=False,
        logger=False
    )
    
    print("--- Training MDNScorer on Baseline Users ---")
    trainer.fit(model, train_loader, val_loader)
    
    print("--- Zero-Shot Inference on Testing Set ---")
    model.eval()
    
    def _extract_results(dl):
        """
        Run inference on a DataLoader and return a DataFrame of per-sample results.

        Each row contains the actual rating, the MDN expected mean prediction,
        the 50% and 95% Highest Density Region segments, and a boolean coverage flag.
        """
        import numpy as np
        rows = []
        
        # Grid for numerical HDR finding
        # [1.0, 1.04, 1.08, ..., 5.0]
        grid_y = torch.linspace(1.0, 5.0, 101)
        dy = (5.0 - 1.0) / 100
        
        with torch.no_grad():
            for x, y, _ in dl:
                mus, log_sigmas, pi_logits = model(x)
                sigmas = torch.exp(log_sigmas)
                pis = torch.softmax(pi_logits, dim=1)
                
                # Compute PDF on grid for each sample in batch
                # (batch, grid_size, k)
                grid_y_expanded = grid_y.view(1, -1, 1) # (1, 101, 1)
                mus_exp = mus.unsqueeze(1) # (batch, 1, k)
                sigmas_exp = sigmas.unsqueeze(1) # (batch, 1, k)
                pis_exp = pis.unsqueeze(1) # (batch, 1, k)
                
                # Laplace PDF Formula: 1/(2*b) * exp(-|y-mu| / b)
                # (batch, 101, k)
                component_pdfs = (1.0 / (2.0 * sigmas_exp)) * torch.exp(-torch.abs(grid_y_expanded - mus_exp) / sigmas_exp)
                # (batch, 101)
                total_pdfs = (pis_exp * component_pdfs).sum(dim=2)
                
                for i in range(len(y)):
                    truth = y[i].item()
                    sample_pdf_grid = total_pdfs[i] # (101,)
                    
                    # Expected Median (weighted mean of locations)
                    m = (mus[i] * pis[i]).sum().item()
                    
                    # --- Find HDR segments ---
                    def get_segments(target_mass):
                        """
                        Find contiguous HDR segments on the 101-point rating grid.

                        Parameters
                        ----------
                        target_mass : float
                            Desired probability mass (e.g. 0.50 for 50% HDR).

                        Returns
                        -------
                        list[tuple[float, float]]
                            List of (low, high) rating boundary pairs.
                        """
                        # Force 100% confidence in [1, 5] range by normalizing the truncated PDF
                        grid_mass = (sample_pdf_grid * dy).sum()
                        if grid_mass > 0:
                            norm_pdf = sample_pdf_grid / grid_mass
                        else:
                            norm_pdf = sample_pdf_grid

                        # Sort PDFs
                        sorted_pdfs, _ = torch.sort(norm_pdf, descending=True)
                        # Find threshold where mass is captured
                        mass = torch.cumsum(sorted_pdfs * dy, dim=0)
                        threshold_idx = (mass >= target_mass).nonzero()
                        if len(threshold_idx) == 0: 
                            threshold = sorted_pdfs[-1].item()
                        else:
                            threshold = sorted_pdfs[threshold_idx[0]].item()
                        
                        # Find contiguous segments where pdf >= threshold
                        active = (norm_pdf >= threshold).cpu().numpy()
                        segs = []
                        start = None
                        for j in range(len(active)):
                            if active[j] and start is None:
                                start = grid_y[j].item()
                            elif not active[j] and start is not None:
                                segs.append((start, grid_y[j-1].item()))
                                start = None
                        if start is not None:
                            segs.append((start, grid_y[-1].item()))
                        return segs

                    h50_segs = get_segments(0.50)
                    h95_segs = get_segments(0.95)
                    
                    # Coverage check
                    in_bounds = 0
                    for slo, shi in h95_segs:
                        if (slo - 0.02 <= truth <= shi + 0.02):
                            in_bounds = 1
                            break
                    
                    rows.append({
                        "Actual_Rating": truth,
                        "Predicted_Median": m,
                        "HDR_50_Segments": h50_segs,
                        "HDR_95_Segments": h95_segs,
                        "HDR_95_Width": sum([shi - slo for slo, shi in h95_segs]),
                        "HDR_95_Segments_Count": len(h95_segs),
                        "In_Bounds": in_bounds
                    })
        return pd.DataFrame(rows)

    train_results = _extract_results(val_loader)
    test_results = _extract_results(test_loader)
    
    return train_results, test_results


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, '..'))
    
    TRAIN_JSON = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'regression_train_set.json')
    TEST_JSON  = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'regression_val_set.json')
    TRAIN_EMB  = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'toy_embeddings', 'toy_restaurant_embeddings_train.pt')
    VAL_EMB    = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'toy_embeddings', 'toy_restaurant_embeddings_val.pt')
    
    if not os.path.exists(TRAIN_JSON):
        print("Missing train JSON. Execute pipelines/yelp/export_regression_train.py first!")
        exit(1)
        
    evaluate_regression(TRAIN_JSON, TEST_JSON, TRAIN_EMB, VAL_EMB)

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MDN_CHECKPOINT = DATA_DIR / "yelp_sandbox" / "mdn_models" / "clip_v2" / "clip_v2_full.ckpt"

@st.cache_resource(show_spinner=False)
def load_mdn_model():
    """
    Load the trained MDNScorer from disk and return it in eval mode.

    The checkpoint path is resolved relative to the project ``data/`` directory.
    Returns ``None`` if the checkpoint file does not exist, allowing callers
    to degrade gracefully (no crash, just no predictions).

    The result is cached by Streamlit so the model is only loaded once per session.
    """
    if not MDN_CHECKPOINT.exists():
        return None
    model = MDNScorer.load_from_checkpoint(str(MDN_CHECKPOINT), map_location="cpu")
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def get_michelin_centroid():
    """
    Compute and return the mean vector (centroid) of all Michelin restaurant embeddings.
    
    This is used for Centroid Subtraction to amplify the discriminative signal 
    within the tightly-clustered Michelin embedding space.
    """
    from core.data_loader import load_restaurant_embeddings
    embeddings_map = load_restaurant_embeddings()
    if not embeddings_map:
        return None
    all_vecs = np.stack(list(embeddings_map.values()))
    centroid = all_vecs.mean(axis=0)
    return torch.from_numpy(centroid).float()

def _center_vec(vec: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Subtract the centroid and re-normalise the vector."""
    if centroid is None:
        return vec
    centered = vec - centroid
    return centered / (centered.norm() + 1e-8)

def _score_mdn_recommendations(catalog: pd.DataFrame, user_ratings: Dict[str, float]) -> pd.DataFrame:
    if catalog.empty or not user_ratings:
        return pd.DataFrame()
        
    model = load_mdn_model()
    if model is None:
        st.error("MDN checkpoint not found.")
        return pd.DataFrame()
        
    embeddings_map = load_restaurant_embeddings()
    centroid = get_michelin_centroid()
    
    # Load metadata
    meta_path = os.path.join('data', 'embeddings', 'restaurant_metadata.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    like_vecs, like_weights = [], []
    dislike_vecs, dislike_weights = [], []
    for rid, rating in user_ratings.items():
        vec = embeddings_map.get(rid)
        if vec is not None:
            w = float(rating) / 5.0
            centered_vec = _center_vec(torch.from_numpy(vec).float(), centroid)
            if rating >= 3.5:
                like_vecs.append(centered_vec * w)
                like_weights.append(w)
            else:
                aw = (6.0 - rating) / 5.0
                dislike_vecs.append(centered_vec * aw)
                dislike_weights.append(aw)

    if not like_vecs and not dislike_vecs:
        return pd.DataFrame()

    user_like_vec = torch.stack(like_vecs).sum(dim=0) / sum(like_weights) if like_vecs else torch.zeros(512)
    if like_vecs: user_like_vec = user_like_vec / (torch.linalg.norm(user_like_vec) + 1e-9)
    
    user_dislike_vec = torch.stack(dislike_vecs).sum(dim=0) / sum(dislike_weights) if dislike_vecs else torch.zeros(512)
    if dislike_vecs: user_dislike_vec = user_dislike_vec / (torch.linalg.norm(user_dislike_vec) + 1e-9)
    
    mean_hist_rating = sum(user_ratings.values()) / len(user_ratings)
    scalar_feats = torch.tensor([mean_hist_rating], dtype=torch.float32)

    rows = []
    for row in catalog.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        if rest_id in user_ratings:
            continue
        target_vec_np = embeddings_map.get(rest_id)
        if target_vec_np is None:
            continue
        
        target_vec = _center_vec(torch.from_numpy(target_vec_np).float(), centroid)
        m = metadata.get(rest_id, {'stars': 4.0, 'review_count': 1, 'price': 2})
        t_meta = torch.tensor([m.get('stars', 4.0)/5.0, torch.log1p(torch.tensor(m.get('review_count', 1)))/10.0, m.get('price', 2)/4.0], dtype=torch.float32)
        all_scalars = torch.cat([scalar_feats, t_meta])
        
        feature_vec = torch.cat([user_like_vec, user_dislike_vec, target_vec, all_scalars]).unsqueeze(0)
        with torch.no_grad():
            mus, log_sigmas, pi_logits = model(feature_vec)
            pis = torch.softmax(pi_logits, dim=1)
            expected_mu = (mus * pis).sum(dim=1).item()
            
            grid_y = torch.linspace(1.0, 5.0, 101)
            grid_y_expanded = grid_y.view(1, -1, 1)
            mus_exp = mus.unsqueeze(1)
            sigmas_exp = torch.exp(log_sigmas).unsqueeze(1)
            pis_exp = pis.unsqueeze(1)
            component_pdfs = (1.0 / (2.0 * sigmas_exp)) * torch.exp(-torch.abs(grid_y_expanded - mus_exp) / sigmas_exp)
            total_pdfs = (pis_exp * component_pdfs).sum(dim=2)
            pdf_array = total_pdfs[0].cpu().numpy()
        rows.append({**row, "score": expected_mu, "pdf_grid": pdf_array})
    return pd.DataFrame(rows).sort_values(by="score", ascending=False)


def add_mdn_predictions(catalog: pd.DataFrame, user_ratings: Dict[str, float]) -> pd.DataFrame:
    if catalog.empty or not user_ratings:
        return catalog
    
    embeddings_map = load_restaurant_embeddings()
    centroid = get_michelin_centroid()
    model = load_mdn_model()
    if model is None:
        return catalog

    meta_path = os.path.join('data', 'embeddings', 'restaurant_metadata.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    like_vecs, like_weights = [], []
    dislike_vecs, dislike_weights = [], []
    for rid, rating in user_ratings.items():
        vec = embeddings_map.get(rid)
        if vec is not None:
            w = float(rating) / 5.0
            centered_vec = _center_vec(torch.from_numpy(vec).float(), centroid)
            if rating >= 3.5:
                like_vecs.append(centered_vec * w)
                like_weights.append(w)
            else:
                aw = (6.0 - rating) / 5.0
                dislike_vecs.append(centered_vec * aw)
                dislike_weights.append(aw)

    if not like_vecs and not dislike_vecs:
        return catalog

    user_like_vec = torch.stack(like_vecs).sum(dim=0) / sum(like_weights) if like_vecs else torch.zeros(512)
    if like_vecs: user_like_vec = user_like_vec / (torch.linalg.norm(user_like_vec) + 1e-9)
    user_dislike_vec = torch.stack(dislike_vecs).sum(dim=0) / sum(dislike_weights) if dislike_vecs else torch.zeros(512)
    if dislike_vecs: user_dislike_vec = user_dislike_vec / (torch.linalg.norm(user_dislike_vec) + 1e-9)
    
    mean_hist_rating = sum(user_ratings.values()) / len(user_ratings)
    scalar_feats = torch.tensor([mean_hist_rating], dtype=torch.float32)

    rows = []
    for row in catalog.to_dict(orient="records"):
        rid = _clean_text(row.get("rest_id"))
        if rid in user_ratings:
            rows.append({**row, "actual_rating": user_ratings[rid]})
            continue
        
        target_vec_np = embeddings_map.get(rid)
        if target_vec_np is None:
            rows.append(row)
            continue
        
        target_vec = _center_vec(torch.from_numpy(target_vec_np).float(), centroid)
        m = metadata.get(rid, {'stars': 4.0, 'review_count': 1, 'price': 2})
        t_meta = torch.tensor([m.get('stars', 4.0)/5.0, torch.log1p(torch.tensor(m.get('review_count', 1)))/10.0, m.get('price', 2)/4.0], dtype=torch.float32)
        all_scalars = torch.cat([scalar_feats, t_meta])
        
        feature_vec = torch.cat([user_like_vec, user_dislike_vec, target_vec, all_scalars]).unsqueeze(0)
        with torch.no_grad():
            mus, log_sigmas, pi_logits = model(feature_vec)
            pis = torch.softmax(pi_logits, dim=1)
            expected_mu = (mus * pis).sum(dim=1).item()
            
            grid_y = torch.linspace(1.0, 5.0, 101)
            grid_y_expanded = grid_y.view(1, -1, 1)
            mus_exp = mus.unsqueeze(1)
            sigmas_exp = torch.exp(log_sigmas).unsqueeze(1)
            pis_exp = pis.unsqueeze(1)
            component_pdfs = (1.0 / (2.0 * sigmas_exp)) * torch.exp(-torch.abs(grid_y_expanded - mus_exp) / sigmas_exp)
            total_pdfs = (pis_exp * component_pdfs).sum(dim=2)
            pdf_grid = total_pdfs[0].cpu().numpy()
            
        rows.append({**row, "predicted_rating": expected_mu, "pdf_grid": pdf_grid})
    return pd.DataFrame(rows)
