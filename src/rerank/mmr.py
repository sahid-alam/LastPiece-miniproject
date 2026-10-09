"""Maximal Marginal Relevance re-ranking.

Pure functions, no I/O. This module is the diversity half of the project's
contribution, so it is deliberately small and fully testable.

Reference: Carbonell & Goldstein, "The use of MMR, diversity-based reranking
for reordering documents and producing summaries", SIGIR 1998.

Why we need it: pure similarity search over a visually homogeneous catalogue
returns near-duplicates. Five almost-identical black denim jackets scores well
on accuracy and is useless to a buyer. Worse, in a single-unit catalogue it
means most of the stock is never shown to anyone, and an item never shown is
an item never sold.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mmr_select"]


def _validate(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    lambda_: float,
) -> None:
    if query_vec.ndim != 1:
        raise ValueError(f"query_vec must be 1-D, got shape {query_vec.shape}")
    if candidate_vecs.ndim != 2:
        raise ValueError(
            f"candidate_vecs must be 2-D, got shape {candidate_vecs.shape}"
        )
    if candidate_vecs.shape[1] != query_vec.shape[0]:
        raise ValueError(
            f"dimension mismatch: query is {query_vec.shape[0]}-d, "
            f"candidates are {candidate_vecs.shape[1]}-d"
        )
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")


def mmr_select(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Select k candidate indices balancing relevance against variety.

    Vectors are assumed L2-normalised, so an inner product is a cosine
    similarity in [-1, 1]. We normalise defensively anyway because a caller
    passing un-normalised vectors would silently get wrong scores rather than
    an error.

    At each step the score of a remaining candidate i is

        lambda_ * sim(query, i) - (1 - lambda_) * max_{j in selected} sim(i, j)

    The first pick has an empty selected set, so it reduces to the single most
    similar candidate. From the second pick onward, anything resembling what is
    already chosen is penalised.

    Args:
        query_vec: the query embedding, shape (d,).
        candidate_vecs: candidate embeddings, shape (n, d), aligned with the
            caller's own candidate ordering.
        k: how many to return. Clipped to n if larger.
        lambda_: relevance/diversity dial. 1.0 disables diversity entirely and
            reproduces plain similarity ranking, which is exactly what we want
            for the ablation baseline.

    Returns:
        Indices into candidate_vecs, in presentation order. Length min(k, n).

    Raises:
        ValueError: on shape mismatch or lambda_ outside [0, 1].
    """
    _validate(query_vec, candidate_vecs, lambda_)

    n = candidate_vecs.shape[0]
    if n == 0 or k <= 0:
        return []

    k = min(k, n)

    q = _unit(query_vec.astype(np.float32))
    C = _unit_rows(candidate_vecs.astype(np.float32))

    # Relevance to the query, computed once.
    sim_to_query = C @ q                      # (n,)

    # Pairwise candidate similarity, computed once. n is the over-fetch depth
    # (50 by default), so this is a trivially small matrix.
    sim_between = C @ C.T                     # (n, n)

    selected: list[int] = []
    remaining = list(range(n))

    # Running max similarity from each remaining candidate to anything already
    # selected. Starts at -inf so the first pick is pure relevance.
    max_sim_to_selected = np.full(n, -np.inf, dtype=np.float32)

    for _ in range(k):
        if not selected:
            scores = sim_to_query[remaining]
        else:
            scores = (
                lambda_ * sim_to_query[remaining]
                - (1.0 - lambda_) * max_sim_to_selected[remaining]
            )

        # argmax over `remaining` positions, mapped back to candidate indices.
        # np.argmax breaks ties by lowest position, which keeps the original
        # retrieval order as a deterministic tie-break. Determinism matters:
        # the examiner may re-run this.
        best_local = int(np.argmax(scores))
        best = remaining[best_local]

        selected.append(best)
        remaining.pop(best_local)

        if remaining:
            max_sim_to_selected[remaining] = np.maximum(
                max_sim_to_selected[remaining], sim_between[best, remaining]
            )

    return selected


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector, leaving a zero vector untouched."""
    norm = float(np.linalg.norm(v))
    return v if norm == 0.0 else (v / norm)


def _unit_rows(M: np.ndarray) -> np.ndarray:
    """L2-normalise each row, leaving zero rows untouched."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return M / norms
