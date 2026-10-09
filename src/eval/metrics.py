"""Evaluation metrics. Pure functions, no I/O.

Four metrics, matching objective 2:

  precision_at_k        - of what we showed, how much was relevant
  catalogue_coverage    - how much of the catalogue ever got shown  [headline]
  intra_list_diversity  - how different the items in one list are
  latency               - measured in the runner, not here

Coverage is the headline metric for this project, not precision. In a catalogue
of unique pieces, an item that never appears in any list can never be sold, so
breadth of exposure is the business outcome. See docs/DECISIONS.md D2.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "precision_at_k",
    "catalogue_coverage",
    "intra_list_diversity",
    "mean_intra_list_diversity",
    "gini_exposure",
]


def precision_at_k(
    recommended: Sequence[str],
    relevant: set[str],
    k: int | None = None,
) -> float:
    """Fraction of the top-k recommendations that are relevant.

    Args:
        recommended: item IDs in presentation order.
        relevant: ground-truth relevant item IDs for this query. Built by the
            rule in docs/DECISIONS.md D10, which is a stated proxy, not real
            click data.
        k: cutoff. Defaults to len(recommended).

    Returns:
        Value in [0, 1]. An empty recommendation list scores 0.0, which is the
        honest reading: showing nothing is not a success.

    Note:
        The denominator is k, not len(recommended). A system that returns 3
        items when 5 were asked for is penalised for the short list. That is
        deliberate, since avoiding short lists is one of our constraints.
    """
    if k is None:
        k = len(recommended)
    if k <= 0:
        return 0.0

    topk = list(recommended)[:k]
    hits = sum(1 for iid in topk if iid in relevant)
    return hits / k


def catalogue_coverage(
    all_recommendations: Iterable[Sequence[str]],
    catalogue_size: int,
) -> float:
    """Fraction of the catalogue that appeared in at least one list.

    Args:
        all_recommendations: one sequence of item IDs per query, across a whole
            evaluation session.
        catalogue_size: total distinct items available at session start.

    Returns:
        Value in [0, 1].

    Why this is the headline: collaborative filtering concentrates exposure on
    a popular minority. Acceptable when stock is replenishable; fatal when every
    item is a single unique piece that needs its own buyer.
    """
    if catalogue_size <= 0:
        return 0.0

    shown: set[str] = set()
    for rec in all_recommendations:
        shown.update(rec)

    return len(shown) / catalogue_size


def intra_list_diversity(vectors: np.ndarray) -> float:
    """Mean pairwise cosine distance within one recommendation list.

    Args:
        vectors: embeddings of the recommended items, shape (n, d).

    Returns:
        Mean of (1 - cosine_similarity) over all n*(n-1)/2 unordered pairs.
        Higher means a more varied list. Lists of 0 or 1 items return 0.0,
        since a single item has no internal variety to measure.

    This is the number that should rise when MMR is switched on. Report it
    alongside precision so the trade-off is visible rather than buried.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")

    n = vectors.shape[0]
    if n < 2:
        return 0.0

    V = vectors.astype(np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    V = V / norms

    sims = V @ V.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sims[iu]))


def mean_intra_list_diversity(list_vectors: Iterable[np.ndarray]) -> float:
    """Average intra-list diversity across many queries.

    Empty lists are skipped rather than counted as zero, so the figure reflects
    the variety of lists actually shown.
    """
    values = [
        intra_list_diversity(v)
        for v in list_vectors
        if v is not None and getattr(v, "shape", (0,))[0] >= 2
    ]
    return float(np.mean(values)) if values else 0.0


def gini_exposure(
    all_recommendations: Iterable[Sequence[str]],
    catalogue_size: int,
) -> float:
    """Gini coefficient of how unevenly exposure is spread over the catalogue.

    0.0 means every item was shown equally often; values near 1.0 mean a tiny
    minority absorbed all the exposure.

    Complements coverage. Coverage asks "was it ever shown?"; Gini asks "was
    exposure fair?". A system can reach decent coverage while still showing a
    handful of items hundreds of times, and in a single-unit catalogue repeat
    exposure of an already-popular piece is wasted.
    """
    if catalogue_size <= 0:
        return 0.0

    counts: dict[str, int] = {}
    for rec in all_recommendations:
        for iid in rec:
            counts[iid] = counts.get(iid, 0) + 1

    # Items never shown are genuine zeros and must be in the distribution.
    values = np.zeros(catalogue_size, dtype=np.float64)
    observed = np.array(sorted(counts.values()), dtype=np.float64)
    if observed.size:
        values[-observed.size:] = observed

    total = values.sum()
    if total == 0.0:
        return 0.0

    values.sort()
    n = values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)
