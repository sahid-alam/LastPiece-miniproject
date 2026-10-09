"""Tests for MMR re-ranking."""

from __future__ import annotations

import numpy as np
import pytest

from src.rerank.mmr import mmr_select


def test_lambda_one_is_plain_similarity_ranking(query_vec, clustered_candidates):
    """lambda=1.0 switches diversity off entirely.

    This is the ablation arm of the experiment, so it has to be exact: with
    lambda=1.0 the output must match pure descending-similarity order.
    """
    vecs, _ = clustered_candidates
    got = mmr_select(query_vec, vecs, k=3, lambda_=1.0)
    expected = list(np.argsort(-(vecs @ query_vec))[:3])
    assert got == expected


def test_diversity_breaks_up_the_duplicate_cluster(query_vec, clustered_candidates):
    """The whole point: three near-identical items must not fill the list."""
    vecs, ids = clustered_candidates

    plain = [ids[i] for i in mmr_select(query_vec, vecs, k=3, lambda_=1.0)]
    diverse = [ids[i] for i in mmr_select(query_vec, vecs, k=3, lambda_=0.5)]

    assert plain == ["dup_a", "dup_b", "dup_c"], "fixture should be duplicate-heavy"

    n_dups = sum(1 for i in diverse if i.startswith("dup_"))
    assert n_dups < 3, f"MMR failed to break the cluster: {diverse}"


def test_first_pick_is_always_most_relevant(query_vec, clustered_candidates):
    """With an empty selected set there is nothing to be diverse from, so the
    first pick must be the nearest item at every lambda."""
    vecs, _ = clustered_candidates
    best = int(np.argmax(vecs @ query_vec))
    for lam in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert mmr_select(query_vec, vecs, k=1, lambda_=lam)[0] == best


def test_returns_k_distinct_indices(query_vec, clustered_candidates):
    vecs, _ = clustered_candidates
    out = mmr_select(query_vec, vecs, k=4, lambda_=0.7)
    assert len(out) == 4
    assert len(set(out)) == 4


def test_k_larger_than_candidates_is_clipped(query_vec, clustered_candidates):
    vecs, _ = clustered_candidates
    out = mmr_select(query_vec, vecs, k=99, lambda_=0.7)
    assert len(out) == vecs.shape[0]
    assert sorted(out) == list(range(vecs.shape[0]))


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty(query_vec, clustered_candidates, k):
    vecs, _ = clustered_candidates
    assert mmr_select(query_vec, vecs, k=k) == []


def test_empty_candidate_set(query_vec):
    assert mmr_select(query_vec, np.empty((0, 8), dtype=np.float32), k=5) == []


def test_single_candidate(query_vec, clustered_candidates):
    vecs, _ = clustered_candidates
    assert mmr_select(query_vec, vecs[:1], k=5, lambda_=0.5) == [0]


def test_is_deterministic(query_vec, clustered_candidates):
    """The examiner may re-run this. Repeated calls must agree."""
    vecs, _ = clustered_candidates
    runs = [tuple(mmr_select(query_vec, vecs, k=4, lambda_=0.6)) for _ in range(5)]
    assert len(set(runs)) == 1


def test_unnormalised_input_gives_same_result(query_vec, clustered_candidates):
    """We normalise defensively, so scaled inputs must not change the order."""
    vecs, _ = clustered_candidates
    a = mmr_select(query_vec, vecs, k=4, lambda_=0.6)
    b = mmr_select(query_vec * 7.0, vecs * 3.0, k=4, lambda_=0.6)
    assert a == b


@pytest.mark.parametrize("bad_lambda", [-0.1, 1.1, 2.0])
def test_lambda_out_of_range_raises(query_vec, clustered_candidates, bad_lambda):
    vecs, _ = clustered_candidates
    with pytest.raises(ValueError, match="lambda_"):
        mmr_select(query_vec, vecs, k=3, lambda_=bad_lambda)


def test_dimension_mismatch_raises(query_vec):
    with pytest.raises(ValueError, match="dimension mismatch"):
        mmr_select(query_vec, np.ones((3, 5), dtype=np.float32), k=2)


def test_lower_lambda_never_increases_duplicate_count(query_vec, clustered_candidates):
    """Monotonicity sanity check across the sweep range we actually report."""
    vecs, ids = clustered_candidates
    counts = []
    for lam in (1.0, 0.8, 0.6, 0.4):
        picked = [ids[i] for i in mmr_select(query_vec, vecs, k=3, lambda_=lam)]
        counts.append(sum(1 for i in picked if i.startswith("dup_")))
    assert counts == sorted(counts, reverse=True)
