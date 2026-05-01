"""
tests/test_algorithms.py
========================
Unit tests for the project's machine learning and retrieval algorithms.

This suite verifies the mathematical and logical correctness of:
1. `algorithms/retrieval.py`: Cosine similarity metrics and keyword overlap heuristics.
"""
import numpy as np

from algorithms.retrieval import _cosine_similarity, _keyword_overlap


def test_retrieval_helpers_are_deterministic():
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert _cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert _keyword_overlap("crispy duck and tasting menu", {"duck", "menu"}) == 1.0
