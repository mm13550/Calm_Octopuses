import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'yelp_sandbox'))
TOY_DIR = os.path.join(DATA_DIR, 'toy_embeddings')
TRAIN_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_train_embeddings.pt')
VAL_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_val_embeddings.pt')
MODEL_OUT_DIR = os.path.join(DATA_DIR, 'models')

# Ensure output directory exists
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# --- Data Loading ---
class YelpEmbeddingDataset(Dataset):
    def __init__(self, data):
        """Accepts a pre-sliced list of embedding dicts."""
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure flat float32 vectors regardless of original shape
        return item['image_embedding'].flatten().float(), item['text_embedding'].flatten().float()

def get_dataloaders(batch_size=128, val_fraction=0.15):
    """
    Loads all embeddings from both .pt files, merges them into a single pool,
    then performs a random 85/15 split. This ensures train and val share the
    same business distribution, preventing the val loss from being inflated by
    a completely disjoint set of restaurants.
    """
    print("Loading embedding tensors from disk...")
    train_raw = torch.load(TRAIN_EMBEDDINGS, weights_only=False)
    
    # Gracefully handle missing val file 
    if os.path.exists(VAL_EMBEDDINGS):
        val_raw = torch.load(VAL_EMBEDDINGS, weights_only=False)
        all_data = train_raw + val_raw
    else:
        all_data = train_raw
    
    print(f"Total samples available: {len(all_data)}")
    
    # Random shuffle + split
    import random
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - val_fraction))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")
    
    train_dataset = YelpEmbeddingDataset(train_data)
    val_dataset = YelpEmbeddingDataset(val_data)
    
    # Num_workers=0 to prevent multiprocessing lockups on Windows during basic training.
    # drop_last=True on BOTH loaders keeps InfoNCE batch sizes identical.
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_dl, val_dl

# --- Model Architecture ---
class CrossModalAutoencoder(pl.LightningModule):
    def __init__(self, recon_weight=2.0, alignment_weight=1.0, dropout=0.3):
        """
        Dual-tower autoencoder with MSE latent alignment.

        Loss = (recon_weight * reconstruction_loss) + (alignment_weight * alignment_loss)

        - reconstruction_loss: MSE between original and reconstructed embeddings.
          Prevents dimensional collapse by forcing the decoders to recover inputs.
        - alignment_loss: MSE between img_latent and txt_latent for the same item.
          Pulls matching pairs together in the shared 256-D space. Simpler and
          more stable than InfoNCE on small (10k) datasets because it has no
          batch-size sensitivity and presents a smoother optimization surface.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.recon_weight = recon_weight
        self.alignment_weight = alignment_weight

        # Image Tower (2048 -> 256 -> 2048)
        # Reduced bottleneck to 256-D (from 512-D) to limit memorization capacity.
        # BatchNorm1d stabilizes activations and acts as an additional regularizer.
        self.image_encoder = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        self.image_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2048)
        )
        
        # Text Tower (768 -> 256 -> 768)
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256)
        )
        self.text_decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 768)
        )

    def forward_image(self, x):
        latent = self.image_encoder(x)
        recon = self.image_decoder(latent)
        return latent, recon
        
    def forward_text(self, x):
        latent = self.text_encoder(x)
        recon = self.text_decoder(latent)
        return latent, recon

    def forward(self, img_x, txt_x):
        img_latent, img_recon = self.forward_image(img_x)
        txt_latent, txt_recon = self.forward_text(txt_x)
        return img_latent, img_recon, txt_latent, txt_recon
        
    def _alignment_loss(self, img_latent, txt_latent):
        """MSE between matched image and text latent vectors.
        Pulls same-item pairs together without penalizing negatives.
        Batch-size independent and presents a smooth optimization surface."""
        return F.mse_loss(img_latent, txt_latent)

    def step(self, batch, batch_idx, phase):
        img_x, txt_x = batch
        
        img_latent, img_recon, txt_latent, txt_recon = self(img_x, txt_x)
        
        # 1. Reconstruction loss — keeps both towers from collapsing to zero
        img_recon_loss = F.mse_loss(img_recon, img_x)
        txt_recon_loss = F.mse_loss(txt_recon, txt_x)
        recon_loss = img_recon_loss + txt_recon_loss
        
        # 2. Alignment loss — pulls matched (image, text) latent pairs together
        alignment_loss = self._alignment_loss(img_latent, txt_latent)
        
        total_loss = (self.recon_weight * recon_loss) + (self.alignment_weight * alignment_loss)
        
        self.log(f'{phase}_loss', total_loss, prog_bar=True)
        self.log(f'{phase}_recon_loss', recon_loss, prog_bar=False)
        self.log(f'{phase}_alignment_loss', alignment_loss, prog_bar=False)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        # Reduced lr to 3e-5 to slow the training trajectory and allow val loss
        # to keep pace before the model overfits the 10k training pairs.
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, weight_decay=1e-4)
        # Cosine annealing gradually decays lr, helping the model settle into better minima.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

# --- Execution ---
def main():
    print("--- Initializing Cross-Modal Dual Encoder Training ---")
    
    # 1. Prepare Data
    train_dl, val_dl = get_dataloaders(batch_size=128)
    
    # 2. Initialize Model
    model = CrossModalAutoencoder()
    
    # 3. Setup CSV Logger for Streamlit Visualization
    logger = CSVLogger(save_dir=MODEL_OUT_DIR, name="cross_modal_logs")
    
    # 4. Setup EarlyStopping — halt training if val_loss stops improving
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,   # Stop if no improvement for 3 consecutive epochs
        mode='min',
        verbose=True
    )
    
    # 5. Setup Checkpointing — only keep best val_loss checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_OUT_DIR,
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=30,  # Allow more epochs; EarlyStopping will halt automatically
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices='auto'
    )
    
    # 6. Train the model
    print("> Commencing PyTorch Lightning Execution...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print("--- Training Finalized ---")
    print(f"Checkout the logs in: {logger.log_dir}")

if __name__ == "__main__":
    main()
