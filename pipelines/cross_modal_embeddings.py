import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'yelp_sandbox'))
TOY_DIR = os.path.join(DATA_DIR, 'toy_embeddings')
TRAIN_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_train_embeddings.pt')
VAL_EMBEDDINGS = os.path.join(TOY_DIR, 'toy_val_embeddings.pt')
MODEL_OUT_DIR = os.path.join(DATA_DIR, 'models')

# Ensure output directory exists
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# --- Data Loading ---
class YelpEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        print(f"Loading local tensor dataset from {data_path}...")
        self.data = torch.load(data_path, weights_only=False)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Depending on how text/image embeddings were shaped, ensure flat vectors:
        return item['image_embedding'].flatten(), item['text_embedding'].flatten()

def get_dataloaders(batch_size=32):
    train_dataset = YelpEmbeddingDataset(TRAIN_EMBEDDINGS)
    val_dataset = YelpEmbeddingDataset(VAL_EMBEDDINGS)
    
    # Num_workers=0 to prevent multiprocessing lockups on windows during basic training
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_dl, val_dl

# --- Model Architecture ---
class CrossModalAutoencoder(pl.LightningModule):
    def __init__(self, recon_weight=1.0, contrastive_weight=1.0, temperature=0.07):
        super().__init__()
        self.save_hyperparameters()
        
        # Hyperparams
        self.recon_weight = recon_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        # Image Tower (2048 -> 512 -> 2048)
        self.image_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.image_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048)
        )
        
        # Text Tower (768 -> 512 -> 768)
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.text_decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
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
        
    def _contrastive_loss(self, img_latent, txt_latent):
        # L2 Normalize the embeddings
        img_norm = F.normalize(img_latent, p=2, dim=1)
        txt_norm = F.normalize(txt_latent, p=2, dim=1)
        
        # Cosine similarity matrix (Batch x Batch)
        logits = torch.matmul(img_norm, txt_norm.T) / self.temperature
        
        # Labels are the diagonal (each image matches with its corresponding text)
        labels = torch.arange(logits.size(0), device=self.device)
        
        # Cross entropy loss on both axes (InfoNCE)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2.0

    def step(self, batch, batch_idx, phase):
        img_x, txt_x = batch
        
        # Forward pass
        img_latent, img_recon, txt_latent, txt_recon = self(img_x, txt_x)
        
        # 1. Reconstruction Losses (MSE)
        img_recon_loss = F.mse_loss(img_recon, img_x)
        txt_recon_loss = F.mse_loss(txt_recon, txt_x)
        recon_loss = img_recon_loss + txt_recon_loss
        
        # 2. Contrastive Loss pulling latent vectors of matches together, pushing mismatches apart
        contrastive_loss = self._contrastive_loss(img_latent, txt_latent)
        
        # Total Loss
        total_loss = (self.recon_weight * recon_loss) + (self.contrastive_weight * contrastive_loss)
        
        self.log(f'{phase}_loss', total_loss, prog_bar=True)
        self.log(f'{phase}_recon_loss', recon_loss, prog_bar=False)
        self.log(f'{phase}_contrastive_loss', contrastive_loss, prog_bar=False)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# --- Execution ---
def main():
    print("--- Initializing Cross-Modal Dual Encoder Training ---")
    
    # 1. Prepare Data
    train_dl, val_dl = get_dataloaders(batch_size=128)
    
    # 2. Initialize Model
    model = CrossModalAutoencoder()
    
    # 3. Setup CSV Logger for Streamlit Visualization
    logger = CSVLogger(save_dir=MODEL_OUT_DIR, name="cross_modal_logs")
    
    # 4. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_OUT_DIR,
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=10, # Keep epochs low for sandbox verification
        logger=logger,
        callbacks=[checkpoint_callback],
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
