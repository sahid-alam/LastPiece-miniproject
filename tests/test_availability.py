"""Tests for Stage C: availability filtering, substitution, and the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.index.stock import Stock
from src.rerank.availability import filter_available, rerank_with_availability


# ---------------------------------------------------------------- C1 filter

def test_filter_keeps_only_available():
    stock = Stock(["a", "b", "c"])
    stock.mark_sold("b")
    kept, dropped = filter_available(["a", "b", "c"], stock)
    assert kept == [0, 2]
    assert dropped == 1


def test_filter_unknown_id_treated_as_unavailable():
    """Fail closed. An unknown ID is an upstream bug; surfacing it as a missing
    result is better than showing a dead listing."""
    kept, dropped = filter_available(["ghost"], Stock(["a"]))
    assert kept == []
    assert dropped == 1


def test_filter_everything_sold():
    stock = Stock(["a", "b"])
    stock.mark_sold("a")
    stock.mark_sold("b")
    kept, dropped = filter_available(["a", "b"], stock)
    assert kept == []
    assert dropped == 2


# ------------------------------------------------------------ full pipeline

def test_sold_items_never_appear(query_vec, retrieve_fn, clustered_candidates):
    _, ids = clustered_candidates
    stock = Stock(ids)
    stock.mark_sold("dup_a")
    stock.mark_sold("dup_b")

    res = rerank_with_availability(
        query_vec, retrieve_fn, stock, k=3, n_candidates=6, lambda_=1.0
    )
    assert "dup_a" not in res.item_ids
    assert "dup_b" not in res.item_ids
    assert res.n_dropped_sold == 2


def test_list_stays_full_when_top_matches_are_sold(
    query_vec, retrieve_fn, clustered_candidates
):
    """The headline claim. Over-fetching means filtering does not shorten the
    returned list, so the user still sees k results."""
    _, ids = clustered_candidates
    stock = Stock(ids)
    stock.mark_sold("dup_a")
    stock.mark_sold("dup_b")
    stock.mark_sold("dup_c")

    res = rerank_with_availability(
        query_vec, retrieve_fn, stock, k=3, n_candidates=6, lambda_=1.0
    )
    assert len(res) == 3
    assert not res.short_result
    assert res.item_ids == ["near", "mid", "far"]


def test_substitution_widens_depth_when_needed(query_vec, clustered_candidates):
    """Start too shallow on purpose. The pipeline must widen rather than
    return a short list."""
    vecs, ids = clustered_candidates
    stock = Stock(ids)
    for sold in ("dup_a", "dup_b", "dup_c"):
        stock.mark_sold(sold)

    calls: list[int] = []

    def counting_retrieve(q, depth):
        calls.append(depth)
        scores = vecs @ q
        order = np.argsort(-scores)[:depth]
        return ([ids[i] for i in order], vecs[order], scores[order])

    res = rerank_with_availability(
        query_vec, counting_retrieve, stock,
        k=3, n_candidates=3, max_candidates=24, lambda_=1.0,
    )

    assert res.n_widening_rounds >= 1
    assert calls[0] == 3 and calls[-1] > 3
    assert len(res) == 3


def test_accepts_short_list_when_catalogue_is_exhausted(
    query_vec, retrieve_fn, clustered_candidates
):
    """Late in a session almost everything is sold. We must degrade to a short
    list rather than loop forever."""
    _, ids = clustered_candidates
    stock = Stock(ids)
    for iid in ids[1:]:
        stock.mark_sold(iid)

    res = rerank_with_availability(
        query_vec, retrieve_fn, stock, k=5, n_candidates=6, max_candidates=12
    )
    assert len(res) == 1
    assert res.short_result


def test_empty_result_when_all_sold(query_vec, retrieve_fn, clustered_candidates):
    _, ids = clustered_candidates
    stock = Stock(ids)
    for iid in ids:
        stock.mark_sold(iid)

    res = rerank_with_availability(query_vec, retrieve_fn, stock, k=3, n_candidates=6)
    assert res.item_ids == []
    assert res.short_result


def test_exclude_ids_suppresses_the_query_item(
    query_vec, retrieve_fn, clustered_candidates
):
    """Item-to-item is the main use case. Without exclusion the top result is
    the item the user is already looking at."""
    _, ids = clustered_candidates
    stock = Stock(ids)

    res = rerank_with_availability(
        query_vec, retrieve_fn, stock, k=3, n_candidates=6,
        lambda_=1.0, exclude_ids=["dup_a"],
    )
    assert "dup_a" not in res.item_ids


def test_lambda_one_is_the_mmr_off_arm(query_vec, retrieve_fn, clustered_candidates):
    """lambda_=1.0 is the "MMR off" baseline arm (D14); lower lambda must
    actually change the list."""
    _, ids = clustered_candidates
    stock = Stock(ids)

    plain = rerank_with_availability(
        query_vec, retrieve_fn, stock, k=3, n_candidates=6, lambda_=1.0
    )
    diverse = rerank_with_availability(
        query_vec, retrieve_fn, stock, k=3, n_candidates=6, lambda_=0.4
    )
    assert plain.item_ids == ["dup_a", "dup_b", "dup_c"]
    assert diverse.item_ids != plain.item_ids


def test_diagnostics_are_populated(query_vec, retrieve_fn, clustered_candidates):
    """The report needs these numbers; a silently-zero field is a lost result."""
    _, ids = clustered_candidates
    stock = Stock(ids)
    stock.mark_sold("mid")

    res = rerank_with_availability(
        query_vec, retrieve_fn, stock, k=3, n_candidates=6
    )
    assert res.n_retrieved == 6
    assert res.n_dropped_sold == 1
    assert res.final_candidate_depth >= 6
    assert len(res.scores) == len(res.item_ids)


def test_k_zero_returns_empty(query_vec, retrieve_fn, clustered_candidates):
    _, ids = clustered_candidates
    res = rerank_with_availability(
        query_vec, retrieve_fn, Stock(ids), k=0
    )
    assert len(res) == 0


# ------------------------------------------------------------- stock table

def test_mark_sold_is_irreversible():
    """Single-unit means one sale exhausts the item. Selling twice is a no-op,
    not an error."""
    stock = Stock(["a"])
    assert stock.is_available("a")
    stock.mark_sold("a")
    stock.mark_sold("a")
    assert not stock.is_available("a")
    assert stock.n_available() == 0


def test_available_ids_shrinks_monotonically():
    stock = Stock(["a", "b", "c"])
    assert stock.available_ids() == {"a", "b", "c"}
    stock.mark_sold("b")
    assert stock.available_ids() == {"a", "c"}
