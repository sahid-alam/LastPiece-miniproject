"""Availability-aware re-ranking: Stage C of the pipeline.

This is the project's contribution. Everything upstream (encoders, FAISS) is
standard published machinery; this module is what makes the system work for a
catalogue where stock is permanently one.

Three steps:

  C1  availability filter  - drop anything already sold
  C2  substitution         - widen retrieval until k survivors exist
  C3  diversity re-rank    - MMR over what survived

The ordering matters and is a deliberate decision (see docs/DECISIONS.md D5).
Filtering a list of k produces a short list; filtering an over-fetched list of
n >> k produces a full one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from src.index.stock import Stock

from .mmr import mmr_select

__all__ = [
    "RerankResult",
    "filter_available",
    "rerank_with_availability",
]


@dataclass
class RerankResult:
    """Outcome of Stage C, including the diagnostics the report needs."""

    item_ids: list[str]
    scores: list[float]
    # Diagnostics. These are not decoration: the substitution story is a
    # headline claim, so we need the numbers to back it up in the report.
    n_retrieved: int = 0
    n_dropped_sold: int = 0
    n_widening_rounds: int = 0
    final_candidate_depth: int = 0
    short_result: bool = False  # true if we could not fill k
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.item_ids)


def filter_available(
    item_ids: Sequence[str],
    stock: Stock,
) -> tuple[list[int], int]:
    """C1. Keep positions whose items are still in stock.

    Returns:
        (kept_positions, n_dropped). Positions index into item_ids so the
        caller can slice its own parallel arrays (vectors, scores) the same way.
    """
    kept = [i for i, iid in enumerate(item_ids) if stock.is_available(iid)]
    return kept, len(item_ids) - len(kept)


def rerank_with_availability(
    query_vec: np.ndarray,
    retrieve_fn: Callable[[np.ndarray, int], tuple[list[str], np.ndarray, np.ndarray]],
    stock: Stock,
    k: int = 5,
    n_candidates: int = 50,
    max_candidates: int = 500,
    lambda_: float = 0.7,
    exclude_ids: Sequence[str] | None = None,
) -> RerankResult:
    """Run the full Stage C pipeline.

    Args:
        query_vec: query embedding, shape (d,), L2-normalised.
        retrieve_fn: callable (query_vec, depth) -> (item_ids, vectors, scores),
            where vectors has shape (depth, d). Injected rather than imported so
            this module stays testable without FAISS, and so the index
            implementation can change without touching the contribution.
        stock: availability oracle.
        k: how many results to return.
        n_candidates: starting over-fetch depth. Should be well above k.
        max_candidates: ceiling for substitution widening. Past this we accept
            a short list rather than scanning the whole catalogue on every
            query.
        lambda_: MMR relevance/diversity dial. 1.0 switches diversity off and
            reproduces plain similarity order exactly, which is the "MMR off"
            ablation arm (D14). There is deliberately no separate on/off flag.
        exclude_ids: items to suppress regardless of stock, e.g. the item the
            user is currently viewing in the item-to-item case. Without this the
            top result is always the query item itself.

    Returns:
        RerankResult with up to k item IDs plus diagnostics.

    Notes:
        Widening doubles the depth each round. In a late-session state where
        most stock has sold, the first pass of 50 may yield fewer than k
        survivors; doubling converges quickly without a linear crawl.
    """
    if k <= 0:
        return RerankResult(item_ids=[], scores=[])

    excluded = set(exclude_ids or ())

    depth = max(n_candidates, k)
    rounds = 0
    total_dropped = 0
    n_retrieved = 0

    kept_ids: list[str] = []
    kept_vecs = np.empty((0, query_vec.shape[0]), dtype=np.float32)
    kept_scores = np.empty((0,), dtype=np.float32)

    while True:
        item_ids, vecs, scores = retrieve_fn(query_vec, depth)
        n_retrieved = len(item_ids)

        # Suppress the query item and anything else the caller rules out.
        if excluded:
            keep = [i for i, iid in enumerate(item_ids) if iid not in excluded]
            item_ids = [item_ids[i] for i in keep]
            vecs = vecs[keep] if len(keep) else vecs[:0]
            scores = scores[keep] if len(keep) else scores[:0]

        # C1 availability filter
        kept_pos, dropped = filter_available(item_ids, stock)
        total_dropped = dropped

        kept_ids = [item_ids[i] for i in kept_pos]
        kept_vecs = vecs[kept_pos] if kept_pos else vecs[:0]
        kept_scores = scores[kept_pos] if kept_pos else scores[:0]

        # C2 substitution: enough survivors, or nothing more to be had?
        if len(kept_ids) >= k:
            break
        if depth >= max_candidates or n_retrieved < depth:
            # Either we hit the ceiling, or the index returned fewer rows than
            # we asked for, which means we have already seen the whole
            # catalogue. Widening again would be pointless.
            break

        depth = min(depth * 2, max_candidates)
        rounds += 1

    # C3 diversity re-rank over the survivors
    chosen = mmr_select(query_vec, kept_vecs, k=k, lambda_=lambda_)

    return RerankResult(
        item_ids=[kept_ids[i] for i in chosen],
        scores=[float(kept_scores[i]) for i in chosen],
        n_retrieved=n_retrieved,
        n_dropped_sold=total_dropped,
        n_widening_rounds=rounds,
        final_candidate_depth=depth,
        short_result=len(chosen) < k,
    )
