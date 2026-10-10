"""Replay one simulated session against one system and score it.

Pure apart from the wall clock: numpy only, no files, no FAISS, no pandas, so
the whole scoring path is unit-testable on hand-built fixtures.

A system is any callable (query_id, stock) -> list of item IDs in display
order. Every system gets the same treatment:

  - a fresh, full Stock per replay, so running A then B cannot leak A's state
  - the identical list of steps (generated once by simulate_session)
  - the identical scoring rule for sold items (docs/DECISIONS.md D15)

That is what makes the rows of the results table comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from src.index.stock import Stock
from src.rerank.availability import rerank_with_availability

from .metrics import (
    catalogue_coverage,
    gini_exposure,
    mean_intra_list_diversity,
    precision_at_k,
)
from .simulation import Step

__all__ = [
    "System",
    "StepRecord",
    "RunResult",
    "replay",
    "ours",
    "content_only",
]

System = Callable[[str, Stock], list[str]]

_P95 = 95.0


class SearchIndex(Protocol):
    """What the builders need from an index: VectorIndex's public surface."""

    def search(
        self, query_vec: np.ndarray, depth: int
    ) -> tuple[list[str], np.ndarray, np.ndarray]: ...

    def get_vector(self, item_id: str) -> np.ndarray: ...


@dataclass(frozen=True)
class StepRecord:
    """One served query. Enough to bucket the session by stock remaining for
    the substitution diagnostics table without re-running anything."""

    query_id: str
    shown: tuple[str, ...]
    n_sold_shown: int
    stock_remaining: float  # fraction of the catalogue for sale when served
    latency_ms: float


@dataclass(frozen=True)
class RunResult:
    precision_at_k: float
    coverage: float
    gini: float
    ild: float
    sold_shown_pct: float  # share of shown slots that were already sold
    latency_mean_ms: float
    latency_p95_ms: float
    short_list_rate: float  # share of lists shorter than k
    steps: tuple[StepRecord, ...]


def replay(
    steps: Sequence[Step],
    item_ids: Sequence[str],
    system: System,
    relevant_fn: Callable[[str], set[str]],
    vector_of: Callable[[str], np.ndarray],
    k: int,
) -> RunResult:
    """Run the replay protocol from src/eval/simulation.py and score it.

    Scoring is D15, identical for every system. Relevance is intersected with
    what is still for sale at that step, so a sold item shown is a miss; and
    coverage/Gini only credit exposure of items that could still be bought.
    The content-only baselines skip the stock filter on purpose, and without
    this rule they would earn credit for recommending things nobody can buy.

    ILD is measured on the list as displayed, sold items included: it is a
    property of what the user saw. vector_of should come from one shared
    vector space for every system, or the ILD column compares different spaces.

    A system that learns online (item-CF, #14) can expose an after_step(step)
    attribute; it is called after the step's sales, so the model only ever
    sees the past.
    """
    stock = Stock(item_ids)  # fresh and full, every replay
    n_items = len(item_ids)
    after_step = getattr(system, "after_step", None)

    records: list[StepRecord] = []
    precisions: list[float] = []
    available_shown: list[list[str]] = []
    list_vectors: list[np.ndarray] = []

    for step in steps:
        stock_remaining = stock.n_available() / n_items if n_items else 0.0

        t0 = time.perf_counter()
        recs = list(system(step.query_id, stock))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        relevant_now = {i for i in relevant_fn(step.query_id) if stock.is_available(i)}
        precisions.append(precision_at_k(recs, relevant_now, k))
        still_for_sale = [i for i in recs if stock.is_available(i)]
        available_shown.append(still_for_sale)
        if recs:
            list_vectors.append(np.stack([vector_of(i) for i in recs]))

        records.append(StepRecord(
            query_id=step.query_id,
            shown=tuple(recs),
            n_sold_shown=len(recs) - len(still_for_sale),
            stock_remaining=stock_remaining,
            latency_ms=latency_ms,
        ))

        for iid in step.sold_after:
            stock.mark_sold(iid)
        if after_step is not None:
            after_step(step)

    n_slots = sum(len(r.shown) for r in records)
    latencies = np.array([r.latency_ms for r in records], dtype=np.float64)
    return RunResult(
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        coverage=catalogue_coverage(available_shown, n_items),
        gini=gini_exposure(available_shown, n_items),
        ild=mean_intra_list_diversity(list_vectors),
        sold_shown_pct=(
            100.0 * sum(r.n_sold_shown for r in records) / n_slots if n_slots else 0.0
        ),
        latency_mean_ms=float(latencies.mean()) if latencies.size else 0.0,
        latency_p95_ms=float(np.percentile(latencies, _P95)) if latencies.size else 0.0,
        short_list_rate=(
            float(np.mean([len(r.shown) < k for r in records])) if records else 0.0
        ),
        steps=tuple(records),
    )


def ours(
    index: SearchIndex,
    lambda_: float,
    k: int,
    n_candidates: int,
    max_candidates: int,
) -> System:
    """The full pipeline: over-fetch, stock filter, widen, MMR. lambda_=1.0 is
    the MMR-off arm (D14), so both "ours" rows come from this one builder."""

    def system(query_id: str, stock: Stock) -> list[str]:
        return rerank_with_availability(
            query_vec=index.get_vector(query_id),
            retrieve_fn=index.search,
            stock=stock,
            k=k,
            n_candidates=n_candidates,
            max_candidates=max_candidates,
            lambda_=lambda_,
            exclude_ids=[query_id],
        ).item_ids

    return system


def content_only(index: SearchIndex, k: int) -> System:
    """Plain nearest neighbours. No stock filter and no MMR: that is the point
    of the baseline, it shows what Stage C adds. Fetches k+1 because the query
    item is its own nearest neighbour and must be dropped."""

    def system(query_id: str, stock: Stock) -> list[str]:
        ids, _, _ = index.search(index.get_vector(query_id), k + 1)
        return [i for i in ids if i != query_id][:k]

    return system
