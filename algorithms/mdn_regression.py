"""
Mixture Density Network (Laplace MoG) for Rating Intervals
Predicts Mixture of Laplace components for user-restaurant pairs.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class UserRestaurantDataset(Dataset):
    def __init__(self, json_path, embeddings_paths):
        super().__init__()
        print(f"Loading dataset from {os.path.basename(json_path)}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.restaurant_embeddings = {}
        for path in embeddings_paths:
            print(f"Loading embeddings from {os.path.basename(path)}")
            emb = torch.load(path, map_location='cpu', weights_only=False)
            self.restaurant_embeddings.update(emb)
        print(f"Total merged embeddings available: {len(self.restaurant_embeddings)}")
        
        # Filter out records where target or any historical business isn't in embeddings
        self.valid_data = []
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
        return len(self.valid_data)
        
    def __getitem__(self, idx):
        row = self.valid_data[idx]
        
        # 1. Target Embedding (512-D)
        target_vec = self.restaurant_embeddings[row['target_id']]
        target_vec = torch.from_numpy(target_vec).float()
        
        # 2. User Taste Embedding (512-D)
        # Weighted mean of historical restaurant embeddings
        hist_vecs = []
        hist_weights = []
        for b_id, r in zip(row['hist_ids'], row['hist_ratings']):
            vec = torch.from_numpy(self.restaurant_embeddings[b_id]).float()
            weight = r / 5.0 # normalize 1-5 scale
            hist_vecs.append(vec * weight)
            hist_weights.append(weight)
            
        # Sum and normalize by total weight
        user_vec = torch.stack(hist_vecs).sum(dim=0) / sum(hist_weights)
        
        # Add scalar for absolute rating average
        mean_hist_rating = sum(row['hist_ratings']) / len(row['hist_ratings'])
        scalar_feature = torch.tensor([mean_hist_rating], dtype=torch.float32)
        
        # 3. Concatenate (1025-D)
        feature_vec = torch.cat([user_vec, target_vec, scalar_feature])
        
        # 4. Compute Sample Confidence Weight (logarithmic scaling of history size)
        # Users with large histories produce a higher weight multiplier on the loss
        raw_count = len(row['hist_ids'])
        sample_weight = torch.log1p(torch.tensor(raw_count, dtype=torch.float32))
        
        return feature_vec, torch.tensor(row['target_rating'], dtype=torch.float32), sample_weight


def mixture_laplace_nll_loss(mus, log_sigmas, pi_logits, y_true, sample_weights=None, sharpness_alpha=0.5, entropy_beta=0.2):
    """
    Negative Log Likelihood of a Mixture of Laplaces.
    """
    y_true = y_true.unsqueeze(1) # (batch, 1)
    
    # Stability Fix 1: Use log_softmax directly
    log_pi = torch.log_softmax(pi_logits, dim=1) # (batch, K)
    pis = torch.exp(log_pi)
    
    # Stabilized Bounded scale
    log_sigmas = torch.clamp(log_sigmas, min=-3.5, max=-1.0)
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
    
    # Apply sample weights
    if sample_weights is not None:
        total_loss = torch.mean((-total_log_likelihood) * sample_weights) + sharpness_alpha * width_penalty + entropy_beta * entropy
    else:
        total_loss = nll_loss + sharpness_alpha * width_penalty + entropy_beta * entropy
        
    return total_loss


class MDNScorer(pl.LightningModule):
    def __init__(self, input_dim=1025, hidden_dims=[1024, 1024, 512], k=3, lr=8e-4, sharpness_alpha=0.8, entropy_beta=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.k = k
        self.sharpness_alpha = sharpness_alpha
        self.entropy_beta = entropy_beta
        
        layers = []
        prev_dim = input_dim
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
                self.final_layer.bias[i] = -1.0 + (i * 1.5) 
                # Nuclear Sharp Initialization: log(0.08) approx -2.5
                self.final_layer.bias[k + i] = -2.5
                # Mixing weights: starts uniform
                self.final_layer.bias[2*k + i] = 0.0
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        features = self.mlp(x)
        out = self.final_layer(features)
        
        # Split into (batch, K) components
        mus_raw     = out[:, :self.k]
        log_sigmas_raw = out[:, self.k : 2*self.k]
        pi_logits   = out[:, 2*self.k:]
        
        # Bounded with breathing room to allow robust edge optimization while preventing unconstrained drift
        mus = 0.9 + 4.2 * torch.sigmoid(mus_raw)
        
        # Stabilized Bounded Scale
        log_sigmas = torch.clamp(log_sigmas_raw, min=-3.5, max=0.5)
        
        return mus, log_sigmas, pi_logits
        
    def training_step(self, batch, batch_idx):
        x, y, w = batch
        mus, log_sigmas, pi_logits = self(x)
        loss = mixture_laplace_nll_loss(
            mus, log_sigmas, pi_logits, y, 
            sample_weights=w,
            sharpness_alpha=self.sharpness_alpha,
            entropy_beta=self.entropy_beta
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        mus, log_sigmas, pi_logits = self(x)
        loss = mixture_laplace_nll_loss(
            mus, log_sigmas, pi_logits, y, 
            sample_weights=w,
            sharpness_alpha=self.sharpness_alpha,
            entropy_beta=self.entropy_beta
        )
        self.log("val_loss", loss, prog_bar=True)
        
        # MAE vs the weighted mean of locations
        pis = torch.softmax(pi_logits, dim=1)
        expected_mu = (mus * pis).sum(dim=1)
        mae = F.l1_loss(expected_mu, y)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
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
    
    train_loader = DataLoader(train_sub, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_sub, batch_size=128)
    test_loader  = DataLoader(test_ds, batch_size=256)
    
    model = MDNScorer()
    trainer = pl.Trainer(
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
MDN_CHECKPOINT = DATA_DIR / "yelp_sandbox" / "mdn_models" / "lightning_logs" / "version_0" / "checkpoints" / "epoch=9-step=270.ckpt"

@st.cache_resource(show_spinner=False)
def load_mdn_model():
    if not MDN_CHECKPOINT.exists():
        return None
    model = MDNScorer.load_from_checkpoint(str(MDN_CHECKPOINT), map_location="cpu")
    model.eval()
    return model

def _score_mdn_recommendations(catalog: pd.DataFrame, user_ratings: Dict[str, float]) -> pd.DataFrame:
    if catalog.empty or not user_ratings:
        return pd.DataFrame()
        
    model = load_mdn_model()
    if model is None:
        st.error("MDN checkpoint not found.")
        return pd.DataFrame()
        
    from core.data_loader import load_restaurant_embeddings, _clean_text
    embeddings_map = load_restaurant_embeddings()
    
    hist_vecs = []
    hist_weights = []
    for rest_id, rating in user_ratings.items():
        vec = embeddings_map.get(rest_id)
        if vec is not None:
            weight = float(rating) / 5.0
            hist_vecs.append(torch.from_numpy(vec).float() * weight)
            hist_weights.append(weight)
            
    if not hist_vecs:
        return pd.DataFrame()
        
    user_vec = torch.stack(hist_vecs).sum(dim=0) / sum(hist_weights)
    mean_hist_rating = sum(user_ratings.values()) / len(user_ratings)
    scalar_feature = torch.tensor([mean_hist_rating], dtype=torch.float32)
    
    rows = []
    for row in catalog.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        if rest_id in user_ratings:
            continue
            
        target_vec_np = embeddings_map.get(rest_id)
        if target_vec_np is None:
            continue
            
        target_vec = torch.from_numpy(target_vec_np).float()
        feature_vec = torch.cat([user_vec, target_vec, scalar_feature]).unsqueeze(0)
        
        with torch.no_grad():
            mus, log_sigmas, pi_logits = model(feature_vec)
            pis = torch.softmax(pi_logits, dim=1)
            expected_mu = (mus * pis).sum(dim=1).item()
            
            # Calculate PDF for HDR Visualization
            grid_y = torch.linspace(1.0, 5.0, 101)
            grid_y_expanded = grid_y.view(1, -1, 1)
            mus_exp = mus.unsqueeze(1)
            sigmas_exp = torch.exp(log_sigmas).unsqueeze(1)
            pis_exp = pis.unsqueeze(1)
            component_pdfs = (1.0 / (2.0 * sigmas_exp)) * torch.exp(-torch.abs(grid_y_expanded - mus_exp) / sigmas_exp)
            total_pdfs = (pis_exp * component_pdfs).sum(dim=2)
            pdf_array = total_pdfs[0].cpu().numpy()
            
        rows.append({**row, "score": expected_mu, "pdf_grid": pdf_array})
        
    if not rows:
        return pd.DataFrame()
        
    result_df = pd.DataFrame(rows)
    return result_df.sort_values(by="score", ascending=False)
def add_mdn_predictions(catalog: pd.DataFrame, user_ratings: Dict[str, float]) -> pd.DataFrame:
    if catalog.empty or not user_ratings:
        return catalog
        
    model = load_mdn_model()
    if model is None:
        return catalog
        
    from core.data_loader import load_restaurant_embeddings, _clean_text
    embeddings_map = load_restaurant_embeddings()
    
    hist_vecs = []
    hist_weights = []
    for rest_id, rating in user_ratings.items():
        vec = embeddings_map.get(rest_id)
        if vec is not None:
            weight = float(rating) / 5.0
            hist_vecs.append(torch.from_numpy(vec).float() * weight)
            hist_weights.append(weight)
            
    if not hist_vecs:
        return catalog
        
    user_vec = torch.stack(hist_vecs).sum(dim=0) / sum(hist_weights)
    mean_hist_rating = sum(user_ratings.values()) / len(user_ratings)
    scalar_feature = torch.tensor([mean_hist_rating], dtype=torch.float32)
    
    rows = []
    for row in catalog.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        
        if rest_id in user_ratings:
            rows.append({**row, "actual_rating": user_ratings[rest_id]})
            continue
            
        target_vec_np = embeddings_map.get(rest_id)
        if target_vec_np is None:
            rows.append(row)
            continue
            
        target_vec = torch.from_numpy(target_vec_np).float()
        feature_vec = torch.cat([user_vec, target_vec, scalar_feature]).unsqueeze(0)
        
        with torch.no_grad():
            mus, log_sigmas, pi_logits = model(feature_vec)
            pis = torch.softmax(pi_logits, dim=1)
            expected_mu = (mus * pis).sum(dim=1).item()
            
            # Calculate PDF for HDR Visualization
            grid_y = torch.linspace(1.0, 5.0, 101)
            grid_y_expanded = grid_y.view(1, -1, 1)
            mus_exp = mus.unsqueeze(1)
            sigmas_exp = torch.exp(log_sigmas).unsqueeze(1)
            pis_exp = pis.unsqueeze(1)
            component_pdfs = (1.0 / (2.0 * sigmas_exp)) * torch.exp(-torch.abs(grid_y_expanded - mus_exp) / sigmas_exp)
            total_pdfs = (pis_exp * component_pdfs).sum(dim=2)
            pdf_array = total_pdfs[0].cpu().numpy()
            
        rows.append({**row, "predicted_rating": expected_mu, "pdf_grid": pdf_array})
        
    return pd.DataFrame(rows)
