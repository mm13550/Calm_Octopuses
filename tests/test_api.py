"""
tests/test_api.py
==================
Unit tests for the project's core data loading and path resolution utilities.

This suite verifies that the `core/data_loader.py` module correctly handles:
1. Directory constants and project root resolution.
2. Text cleaning (whitespace collapse, None handling).
3. String truncation for UI display.
4. Relative vs. Absolute path resolution for images.
"""
from pathlib import Path

from core import data_loader


def test_clean_text_collapses_whitespace_and_handles_none():
    assert data_loader._clean_text(None) == ""
    assert data_loader._clean_text("  Katsu\n  curry\t set  ") == "Katsu curry set"


def test_truncate_adds_ellipsis_only_when_needed():
    assert data_loader._truncate("short text", limit=20) == "short text"
    assert data_loader._truncate("abcdefghij", limit=8) == "abcde..."


def test_resolve_path_keeps_absolute_paths():
    absolute = data_loader.PROJECT_ROOT / "data" / "images" / "example.jpg"

    assert data_loader._resolve_path(str(absolute)) == absolute


def test_data_paths_are_project_relative():
    assert data_loader.DATA_DIR == Path(data_loader.PROJECT_ROOT) / "data"
    assert data_loader.LOOKUP_CSV.name == "restaurant_lookup.csv"
