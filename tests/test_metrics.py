"""Tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.encoders.fusion import fuse, fuse_batch, l2_normalize
from src.eval.metrics import (
    catalogue_coverage,
    gini_exposure,
    intra_list_diversity,
    mean_intra_list_diversity,
    precision_at_k,
)


# ------------------------------------------------------------- precision@k

def test_precision_all_relevant():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_none_relevant():
    assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_precision_partial():
    assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5


def test_precision_denominator_is_k_not_list_length():
    """A short list is penalised. Avoiding short lists is one of our stated
    constraints, so the metric has to reflect it."""
    assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(0.4)


def test_precision_empty_list_is_zero():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_precision_k_zero_is_zero():
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0


# --------------------------------------------------------------- coverage

def test_coverage_counts_distinct_items_across_queries():
    recs = [["a", "b"], ["b", "c"], ["a", "c"]]
    assert catalogue_coverage(recs, catalogue_size=10) == pytest.approx(0.3)


def test_coverage_full():
    assert catalogue_coverage([["a", "b", "c"]], catalogue_size=3) == 1.0


def test_coverage_no_recommendations():
    assert catalogue_coverage([], catalogue_size=100) == 0.0


def test_coverage_empty_catalogue():
    assert catalogue_coverage([["a"]], catalogue_size=0) == 0.0


def test_coverage_rewards_variety_over_repetition():
    """This is the behaviour the headline metric exists to capture: a system
    that keeps showing the same popular items scores badly even when it is
    busy."""
    narrow = [["a", "b"]] * 20
    broad = [[f"i{n}", f"i{n+1}"] for n in range(0, 40, 2)]
    assert catalogue_coverage(broad, 40) > catalogue_coverage(narrow, 40)


# -------------------------------------------------------------- diversity

def test_diversity_of_identical_items_is_zero():
    v = np.tile(l2_normalize(np.array([1.0, 0.0, 0.0])), (3, 1))
    assert intra_list_diversity(v) == pytest.approx(0.0, abs=1e-6)


def test_diversity_of_orthogonal_items_is_one():
    v = np.eye(3, dtype=np.float32)
    assert intra_list_diversity(v) == pytest.approx(1.0, abs=1e-6)


def test_diversity_single_item_is_zero():
    assert intra_list_diversity(np.array([[1.0, 0.0]], dtype=np.float32)) == 0.0


def test_diversity_empty_is_zero():
    assert intra_list_diversity(np.empty((0, 4), dtype=np.float32)) == 0.0


def test_diversity_increases_with_spread():
    tight = np.stack([
        l2_normalize(np.array([1.0, 0.05, 0.0])),
        l2_normalize(np.array([1.0, 0.10, 0.0])),
    ])
    loose = np.stack([
        l2_normalize(np.array([1.0, 0.0, 0.0])),
        l2_normalize(np.array([0.0, 1.0, 0.0])),
    ])
    assert intra_list_diversity(loose) > intra_list_diversity(tight)


def test_mean_diversity_skips_degenerate_lists():
    lists = [
        np.eye(2, dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    ]
    assert mean_intra_list_diversity(lists) == pytest.approx(1.0, abs=1e-6)


def test_diversity_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        intra_list_diversity(np.array([1.0, 0.0]))


# ------------------------------------------------------------------- gini

def test_gini_uniform_exposure_is_near_zero():
    recs = [[f"i{n}"] for n in range(10)]
    assert gini_exposure(recs, catalogue_size=10) == pytest.approx(0.0, abs=1e-6)


def test_gini_concentrated_exposure_is_high():
    recs = [["a"]] * 50
    assert gini_exposure(recs, catalogue_size=100) > 0.9


def test_gini_empty():
    assert gini_exposure([], catalogue_size=10) == 0.0


# ----------------------------------------------------------------- fusion

def test_fuse_alpha_zero_is_image_only():
    img = l2_normalize(np.array([1.0, 0.0, 0.0]))
    txt = l2_normalize(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(fuse(img, txt, alpha=0.0), img, atol=1e-6)


def test_fuse_alpha_one_is_text_only():
    img = l2_normalize(np.array([1.0, 0.0, 0.0]))
    txt = l2_normalize(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(fuse(img, txt, alpha=1.0), txt, atol=1e-6)


def test_fuse_output_is_unit_length():
    img = l2_normalize(np.array([1.0, 2.0, 3.0]))
    txt = l2_normalize(np.array([3.0, 1.0, 0.0]))
    assert np.linalg.norm(fuse(img, txt, alpha=0.5)) == pytest.approx(1.0, abs=1e-6)


def test_fuse_falls_back_when_text_missing():
    """Listings without descriptions are common in thrift catalogues and must
    still be recommendable."""
    img = l2_normalize(np.array([1.0, 0.0, 0.0]))
    zero = np.zeros(3, dtype=np.float32)
    np.testing.assert_allclose(fuse(img, zero, alpha=0.5), img, atol=1e-6)
    np.testing.assert_allclose(fuse(img, None, alpha=0.5), img, atol=1e-6)


def test_fuse_both_missing_raises():
    with pytest.raises(ValueError, match="at least one"):
        fuse(np.zeros(3), np.zeros(3), alpha=0.5)


def test_fuse_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        fuse(np.array([1.0, 0.0]), np.array([0.0, 1.0]), alpha=1.5)


def test_fuse_batch_matches_fuse_row_by_row():
    rng = np.random.default_rng(0)
    img = l2_normalize(rng.normal(size=(5, 4)).astype(np.float32), axis=1)
    txt = l2_normalize(rng.normal(size=(5, 4)).astype(np.float32), axis=1)
    batched = fuse_batch(img, txt, alpha=0.3)
    for i in range(5):
        np.testing.assert_allclose(batched[i], fuse(img[i], txt[i], 0.3), atol=1e-6)


def test_fuse_batch_handles_missing_text_rows():
    img = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), axis=1)
    txt = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = fuse_batch(img, txt, alpha=0.5)
    np.testing.assert_allclose(out[0], img[0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)
