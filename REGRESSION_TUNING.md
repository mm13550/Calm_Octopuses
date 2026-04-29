# Regression Tuning Notes

These notes describe the current design intent for the MDN recommendation model in `algorithms/mdn_regression.py`.

## Goals

- Keep the model restaurant-focused rather than overly dependent on user history.
- Preserve target restaurant features during training.
- Preserve interaction features, because they directly express the relationship between the user profile and the target restaurant.
- Use centroid subtraction and normalization because restaurant embeddings are compact in high-dimensional space.
- Keep predictions opinionated enough to produce useful ranking differences.

## Dropout Policy

- Restaurant target features: no dropout.
- Interaction features: no dropout.
- User preference features: drop together 25% of the time during training.
- Metadata scalar features: drop 25% of the time during training.

## Implementation Notes

- The current model uses centered and normalized restaurant embeddings.
- The active frontend path loads the MDN checkpoint if available.
- If the checkpoint is missing, `frontend.py` uses the embedding-based fallback recommendation path in `algorithms/mdn_regression.py`.
