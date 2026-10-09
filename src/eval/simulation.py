"""Simulated shopping session over a static catalogue. Pure, no I/O.

The dataset is static, so we fake single-unit dynamics (docs/DECISIONS.md D16):

  - n_users shoppers arrive one after another; each issues queries_per_user
    item-to-item queries ("show me things like this listing").
  - Each query item is drawn from what is still for sale.
  - After each query, a fixed share of the remaining stock sells and is gone
    for good. Sales are attributed to the user who made that query, which
    gives the item-CF baseline the (user, item) log it needs.
  - Over the whole session, deplete_fraction of the catalogue sells, evenly
    spread, so the late-session state (most stock gone) is actually reached.

Depletion is exogenous: what sells does not depend on what any system showed.
That is the price of fairness. Every system under test replays the identical
list of steps against the identical stock state at every step, so the numbers
are comparable. The cost: we cannot claim "our recommendations caused more
sales", only "our lists stayed full, valid and varied as stock vanished".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["Step", "simulate_session"]


@dataclass(frozen=True)
class Step:
    user_id: str
    query_id: str  # the listing being viewed; available when the query runs
    sold_after: tuple[str, ...]  # items that sell once this query is served


def simulate_session(
    item_ids: Sequence[str],
    n_users: int,
    queries_per_user: int,
    deplete_fraction: float,
    seed: int,
) -> list[Step]:
    """Generate one reproducible session.

    Replay protocol for the runner, identical for every system:
        stock = Stock(item_ids)
        for step in steps:
            recs = system(step.query_id, stock)   # measured
            for iid in step.sold_after: stock.mark_sold(iid)
    """
    if not 0.0 <= deplete_fraction < 1.0:
        raise ValueError(f"deplete_fraction must be in [0, 1), got {deplete_fraction}")

    rng = np.random.default_rng(seed)
    available = list(dict.fromkeys(item_ids))  # dedupe, keep order
    n_steps = n_users * queries_per_user
    total_to_sell = int(deplete_fraction * len(available))

    steps: list[Step] = []
    for t in range(n_steps):
        if not available:
            break
        user_id = f"u{t // queries_per_user}"
        query_id = available[int(rng.integers(len(available)))]

        # Spread sales evenly: cumulative target at step t+1 minus at step t.
        n_sell = (t + 1) * total_to_sell // n_steps - t * total_to_sell // n_steps
        sold_pos = set(rng.choice(len(available), size=n_sell, replace=False).tolist())
        sold = tuple(available[i] for i in sorted(sold_pos))
        # ponytail: O(n) list rebuild per step; fine at 44k x 1000 steps.
        available = [iid for i, iid in enumerate(available) if i not in sold_pos]

        steps.append(Step(user_id, query_id, sold))
    return steps
