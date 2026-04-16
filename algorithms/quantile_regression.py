"""
Quantile Regression Neural Network for Rating Intervals
Predicts the 2.5th, 50th, and 97.5th percentile expected ratings for a user-restaurant pair.
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
        
        # 1. Target Embedding (256-D)
        target_vec = self.restaurant_embeddings[row['target_id']]
        target_vec = torch.from_numpy(target_vec).float()
        
        # 2. User Taste Embedding (256-D)
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
        
        # 3. Concatenate (512-D)
        feature_vec = torch.cat([user_vec, target_vec])
        
        # 4. Compute Sample Confidence Weight (logarithmic scaling of history size)
        # Users with large histories produce a higher weight multiplier on the loss
        raw_count = len(row['hist_ids'])
        sample_weight = torch.log1p(torch.tensor(raw_count, dtype=torch.float32))
        
        return feature_vec, torch.tensor(row['target_rating'], dtype=torch.float32), sample_weight


def pinball_loss(y_pred, y_true, quantiles, sample_weights=None):
    """
    Computes the quantile (pinball) loss, heavily weighted by user statistical certainty.
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, i]
        loss_q = torch.max((q - 1) * errors, q * errors)
        losses.append(loss_q)
        
    stacked_losses = torch.stack(losses, dim=1) # (batch_size, num_quantiles)
    sample_losses = torch.sum(stacked_losses, dim=1) # (batch_size,)
    
    if sample_weights is not None:
        sample_losses = sample_losses * sample_weights
        
    return torch.mean(sample_losses)


class IntervalScorer(pl.LightningModule):
    def __init__(self, input_dim=512, hidden_dims=[256, 128], quantiles=[0.025, 0.50, 0.975], lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.quantiles = quantiles
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2)) # Prevent overfitting easily
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, len(quantiles)))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)
        
    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        loss = pinball_loss(y_pred, y, self.quantiles, sample_weights=w)
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)
        loss = pinball_loss(y_pred, y, self.quantiles, sample_weights=w)
        self.log("val_loss", loss, prog_bar=True)
        
        # Calculate MAE on the median prediction (50th percentile)
        mae = F.l1_loss(y_pred[:, 1], y)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


import pandas as pd

def evaluate_regression(train_json, test_json, train_emb, test_emb, max_epochs=20):
    """
    Called by app.py: 
    1. Trains IntervalScorer locally on the generic training payload.
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
    
    model = IntervalScorer()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        enable_checkpointing=False,
        logger=False
    )
    
    print("--- Training IntervalScorer on Baseline Users ---")
    trainer.fit(model, train_loader, val_loader)
    
    print("--- Zero-Shot Inference on Testing Set ---")
    model.eval()
    
    def _extract_results(dl):
        rows = []
        with torch.no_grad():
            for x, y, _ in dl:
                preds = model(x)
                for i in range(len(y)):
                    lower = preds[i, 0].item()
                    median = preds[i, 1].item()
                    upper = preds[i, 2].item()
                    truth = y[i].item()
                    coverage = 1 if (lower <= truth <= upper) else 0
                    rows.append({
                        "Actual_Rating": truth,
                        "Predicted_Median": median,
                        "Lower_CI": lower,
                        "Upper_CI": upper,
                        "In_Bounds": coverage
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
