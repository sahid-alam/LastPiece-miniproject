"""Shared test fixtures.

Everything here is synthetic. Tests must never download model weights or the
dataset: four people run this on four laptops, often on college wifi, and a
test suite that needs 600 MB of downloads is a test suite nobody runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.index.stock import Stock

DIM = 8  # small enough to reason about by hand in a failing test


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n).astype(np.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def query_vec() -> np.ndarray:
    """A query pointing straight down axis 0."""
    return unit([1, 0, 0, 0, 0, 0, 0, 0])


@pytest.fixture
def clustered_candidates() -> tuple[np.ndarray, list[str]]:
    """Three near-duplicates plus three genuinely different items.

    This is the exact situation MMR exists to fix. Plain similarity ranking
    returns dup_a, dup_b, dup_c: three versions of the same jacket. MMR should
    break the cluster up.

    Returned in descending similarity to the axis-0 query, matching what a
    retrieval step would hand us.
    """
    vecs = np.stack([
        unit([1.00, 0.02, 0, 0, 0, 0, 0, 0]),   # dup_a  ~1.000
        unit([0.99, 0.05, 0, 0, 0, 0, 0, 0]),   # dup_b  ~0.999
        unit([0.98, 0.08, 0, 0, 0, 0, 0, 0]),   # dup_c  ~0.997
        unit([0.80, 0.60, 0, 0, 0, 0, 0, 0]),   # near   ~0.800
        unit([0.60, 0.00, 0.80, 0, 0, 0, 0, 0]),  # mid  ~0.600
        unit([0.30, 0.00, 0, 0.95, 0, 0, 0, 0]),  # far  ~0.300
    ]).astype(np.float32)
    ids = ["dup_a", "dup_b", "dup_c", "near", "mid", "far"]
    return vecs, ids


@pytest.fixture
def full_stock(clustered_candidates) -> Stock:
    _, ids = clustered_candidates
    return Stock(ids)


@pytest.fixture
def retrieve_fn(clustered_candidates):
    """A fake retrieval function matching VectorIndex.search's signature.

    Returns items in descending similarity, truncated to the requested depth,
    exactly as FAISS would.
    """
    vecs, ids = clustered_candidates

    def _retrieve(q: np.ndarray, depth: int):
        scores = vecs @ q
        order = np.argsort(-scores)[:depth]
        return ([ids[i] for i in order], vecs[order], scores[order])

    return _retrieve
