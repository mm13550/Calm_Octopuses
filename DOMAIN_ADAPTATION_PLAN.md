# Domain Adaptation Plan: Yelp → Michelin Fine-Tuning

## Problem

The cross-modal dual encoder (`pipelines/cross_modal_embeddings.py`) was trained exclusively
on **general Yelp Open Dataset** data (~10k casual dining food photos + reviews). It will be
applied at inference time to two out-of-distribution populations:

1. **Regression validation set** — high-end restaurants in a held-out city
2. **App inference target** — NYC Michelin star restaurants

### Why This Is A Risk

The ResNet50 and DistilBERT **backbones** generalize well (pretrained on massive corpora).
However, the **compression layers** (2048→256 image, 768→256 text) learned *which dimensions
to preserve* based on casual Yelp signal. They may compress away exactly the fine-dining-specific
semantic information needed for Michelin restaurant discrimination:

| Dimension         | Yelp Training Distribution              | Michelin Target Distribution              |
|-------------------|-----------------------------------------|-------------------------------------------|
| Food photography  | Casual smartphone shots, variable lighting | Professional plating, styled compositions |
| Review vocabulary | "great value, came back 3x"             | "the dashi consommé had remarkable umami depth" |
| Price / quality   | Broad spectrum, mostly mid-range        | Exclusively fine dining                   |

The downstream consequence is a **lower accuracy ceiling on the regression head** — the
centroid vectors for Michelin restaurants will be less discriminative than they could be.

---

## Current State of Michelin Pipeline

`pipelines/generate_embeddings_michelin.py` currently uses **CLIP** (not ResNet50/DistilBERT)
to embed images and saves to a `.parquet` file. It does **not** produce the dual-tower `.pt`
format expected by the encoder, and it has **no text review** component.

> [!IMPORTANT]
> Before the fine-tuning step below can proceed, `generate_embeddings_michelin.py` must be
> rewritten to use the same ResNet50 + DistilBERT backbone as `generate_embeddings_yelp.py`,
> and must output `.pt` files in the same `{image_embedding, text_embedding, business_id}` dict
> format.

---

## Proposed Fix: Two-Stage Domain Adaptation

### Stage 1 — Rewrite `generate_embeddings_michelin.py`

Refactor the script to mirror `generate_embeddings_yelp.py`:

- **Image tower**: ResNet50 (strip final FC layer → 2048-D vector)
- **Text tower**: DistilBERT (CLS token → 768-D vector)
- **Source data**:
  - Images from `data/images/` (the scraped Michelin restaurant photos)
  - Text from scraped Google Places reviews (already fetched by `fetch_and_embed_reviews.py`)
- **Output**: `data/michelin_sandbox/michelin_embeddings.pt` in the same dict-list format

### Stage 2 — Fine-Tune the Encoder on Michelin Data

Add a `fine_tune()` function to `cross_modal_embeddings.py` that:

1. Loads the **best checkpoint** saved by the Yelp training run (from `data/yelp_sandbox/models/`)
2. Loads the Michelin `.pt` embeddings through the same `YelpEmbeddingDataset` / `get_dataloaders()`
3. Continues training for a **small number of epochs** (5–10 max) with a **much lower learning rate**
   (`lr=1e-5` vs. the original `3e-5`) to nudge the compression layers toward fine-dining semantics
   without catastrophically forgetting the Yelp-learned alignment
4. Saves a separate fine-tuned checkpoint: `cross_modal_michelin_finetuned.pt`

```python
# Pseudocode for fine_tune() entry point
def fine_tune():
    model = CrossModalAutoencoder.load_from_checkpoint(YELP_BEST_CHECKPOINT)
    train_dl, val_dl = get_dataloaders(
        train_path=MICHELIN_EMBEDDINGS,
        batch_size=32,        # smaller — less Michelin data available
        val_fraction=0.20
    )
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
        ...
    )
    trainer.fit(model, train_dl, val_dl)
    trainer.save_checkpoint(MICHELIN_FINETUNED_OUT)
```

> [!TIP]
> If the Michelin dataset is very small (< 1000 pairs), consider **freezing the decoder
> weights** during fine-tuning and only updating the encoder layers. This allows the latent
> space geometry to shift toward fine-dining without risking reconstruction collapse.

---

## Acceptance Criteria

- [ ] `generate_embeddings_michelin.py` rewritten to produce ResNet50 + DistilBERT `.pt` output
- [ ] Fine-tuning run completes with `val_loss` declining consistently (no overfitting)
- [ ] Fine-tuned checkpoint used as the encoder in Phase 3 (regression head training)
- [ ] Regression head val MAE meaningfully lower with fine-tuned encoder vs. Yelp-only encoder

---

## Files Affected

| File | Change |
|---|---|
| `pipelines/generate_embeddings_michelin.py` | Full rewrite (ResNet50 + DistilBERT, `.pt` output) |
| `pipelines/cross_modal_embeddings.py` | Add `fine_tune()` entry point and `__main__` mode flag |
| `data/michelin_sandbox/` | New directory for Michelin `.pt` embeddings and fine-tuned checkpoint |
