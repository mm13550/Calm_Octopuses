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
        
        return feature_vec, torch.tensor(row['target_rating'], dtype=torch.float32)


def pinball_loss(y_pred, y_true, quantiles):
    """
    Computes the quantile (pinball) loss.
    y_pred: (batch_size, num_quantiles)
    y_true: (batch_size,)
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, i]
        loss_q = torch.max((q - 1) * errors, q * errors)
        losses.append(loss_q)
    # Sum over quantiles, mean over batch
    return torch.mean(torch.sum(torch.stack(losses, dim=1), dim=1))


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
        x, y = batch
        y_pred = self(x)
        loss = pinball_loss(y_pred, y, self.quantiles)
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = pinball_loss(y_pred, y, self.quantiles)
        self.log("val_loss", loss, prog_bar=True)
        
        # Calculate MAE on the median prediction (50th percentile)
        mae = F.l1_loss(y_pred[:, 1], y)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, '..'))
    
    # We dynamically train and evaluate the performance of our embeddings using the sandbox regressions
    JSON_PATH  = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'regression_val_set.json')
    TRAIN_EMB  = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'toy_embeddings', 'toy_restaurant_embeddings_train.pt')
    VAL_EMB    = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'toy_embeddings', 'toy_restaurant_embeddings_val.pt')
    
    if not os.path.exists(JSON_PATH) or not os.path.exists(TRAIN_EMB) or not os.path.exists(VAL_EMB):
        print("Missing required data files. Please ensure you have run Phase 1 aggregation.")
        exit(1)
        
    dataset = UserRestaurantDataset(JSON_PATH, [TRAIN_EMB, VAL_EMB])
    
    if len(dataset) == 0:
        print("Error: Emtpy dataset after filtering. Adjust logic if no elements matched.")
        exit(1)
    
    # Force 80/20 train/val split so it actually trains locally
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64)
    
    model = IntervalScorer()
    
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        enable_checkpointing=False, # We don't need to bloat the artifact dir for toy runs
        logger=False
    )
    
    print("--- Starting Quantile Regression Neural Network Training ---")
    trainer.fit(model, train_loader, val_loader)
    
    # Run a quick check on the validation set to prove interval sanity
    print("\n--- 95% Confidence Interval Sanity Check (Validation Set) ---")
    model.eval()
    x, y = next(iter(val_loader))
    with torch.no_grad():
        preds = model(x)
        
    for i in range(5):
        print(f"Target Rating (Truth): {y[i]:.1f}")
        print(f"  Lower Bound 95% CI (Risk Level) : {preds[i, 0]:.2f}")
        print(f"  50th %ile Median (Expected)     : {preds[i, 1]:.2f}")
        print(f"  Upper Bound 95% CI (Best Case)  : {preds[i, 2]:.2f}\n")
