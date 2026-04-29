import numpy as np
import pytest

from algorithms.clustering import fit_gaussian_mixture
from algorithms.retrieval import _cosine_similarity, _keyword_overlap


def test_fit_gaussian_mixture_returns_labels_and_probabilities():
    matrix = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [0.1, 0.2],
            [8.0, 8.1],
            [8.2, 8.0],
            [8.1, 8.2],
        ],
        dtype=float,
    )

    result = fit_gaussian_mixture(matrix, n_components=2, random_state=7)

    assert result.labels.shape == (6,)
    assert result.probabilities.shape == (6, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.isfinite(result.bic)
    assert np.isfinite(result.aic)


def test_fit_gaussian_mixture_rejects_too_many_components():
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_gaussian_mixture([[1.0, 2.0], [2.0, 3.0]], n_components=3)


def test_retrieval_helpers_are_deterministic():
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert _cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert _keyword_overlap("crispy duck and tasting menu", {"duck", "menu"}) == 1.0
