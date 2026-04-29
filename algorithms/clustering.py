"""Gaussian Mixture clustering utilities for restaurant embeddings.

The original file was a stub. This version is safe to import from notebooks,
pipelines, or the Streamlit frontend. It validates the embedding matrix, fits a
GMM, and returns labels/probabilities plus model diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GMMClusteringResult:
    """Structured output from a Gaussian Mixture clustering run."""

    model: GaussianMixture
    labels: np.ndarray
    probabilities: np.ndarray
    transformed_matrix: np.ndarray
    bic: float
    aic: float
    scaler: StandardScaler | None = None


def _validate_matrix(data_matrix: Any) -> np.ndarray:
    matrix = np.asarray(data_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"data_matrix must be 2D, got shape={matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError("data_matrix must contain at least two rows")
    if not np.isfinite(matrix).all():
        raise ValueError("data_matrix contains NaN or infinite values")
    return matrix


def fit_gaussian_mixture(
    data_matrix: Any,
    n_components: int,
    *,
    covariance_type: str = "full",
    random_state: int = 42,
    scale: bool = True,
    max_iter: int = 500,
    reg_covar: float = 1e-6,
) -> GMMClusteringResult:
    """Fit a Gaussian Mixture Model to an embedding matrix.

    Args:
        data_matrix: 2D array-like object of shape (n_samples, n_features).
        n_components: Number of latent clusters/styles to discover.
        covariance_type: sklearn GMM covariance type: full, tied, diag, or spherical.
        random_state: Reproducibility seed.
        scale: Whether to standardize features before fitting.
        max_iter: Maximum EM iterations.
        reg_covar: Non-negative regularization added to covariance diagonal.

    Returns:
        GMMClusteringResult containing the fitted model, labels, soft cluster
        probabilities, transformed input matrix, BIC, and AIC.
    """
    matrix = _validate_matrix(data_matrix)
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if n_components > matrix.shape[0]:
        raise ValueError("n_components cannot exceed the number of samples")

    scaler: StandardScaler | None = None
    transformed = matrix
    if scale:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(matrix)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
        reg_covar=reg_covar,
    )
    labels = model.fit_predict(transformed)
    probabilities = model.predict_proba(transformed)

    return GMMClusteringResult(
        model=model,
        labels=labels,
        probabilities=probabilities,
        transformed_matrix=transformed,
        bic=float(model.bic(transformed)),
        aic=float(model.aic(transformed)),
        scaler=scaler,
    )
