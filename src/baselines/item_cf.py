"""Item-based collaborative filtering baseline.

STATUS: STUB. Owner: Track A (after the data work). Implement the TODOs below.

Build it honestly and fairly. We expect it to fail, and the reason is
structural, so there is no need to weaken it:

  - Interactions are views (each step's query item) and purchases (each
    step's sold_after), attributed to that step's user (D16).
  - CF may only learn from steps BEFORE the current one. Fitting on the whole
    session first would leak the future.
  - Co-occurrence needs the same item to appear in two users' histories. A
    single-unit item can be bought once, so purchase co-occurrence can only
    ever point at items that are already sold, and sold items count as misses
    (D15). Whatever CF manages to recommend comes from the view signal alone.

Report with the numbers: matrix density, the share of queries that got an
empty list, and the share of recommended slots that were already sold.

Do not fall back to popularity or to content similarity when CF has nothing
to say. The empty list is the finding.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["ItemCFBaseline"]


class ItemCFBaseline:
    """Item-item co-occurrence CF, updated online as the session replays."""

    def __init__(self) -> None:
        # ponytail: plain dict counts; ~1000 users x a few hundred items each is
        # tiny, so no scipy.sparse dependency.
        self._user_items: dict[str, set[str]] = {}
        self._co: dict[str, dict[str, int]] = {}

    def observe(self, user_id: str, item_ids: Iterable[str]) -> None:
        """Add one step's interactions (views + purchases) to the model.

        TODO(owner): for each new item, bump co-occurrence counts against every
        item already in that user's history, both directions.
        """
        raise NotImplementedError("ItemCFBaseline.observe")

    def recommend(self, item_id: str, k: int = 5) -> list[str]:
        """Top-k items by co-occurrence count with item_id; ties by item_id.

        Returns [] when item_id has no co-occurrences, which will be common.

        TODO(owner): implement.
        """
        raise NotImplementedError("ItemCFBaseline.recommend")

    def density(self, n_items: int) -> float:
        """Non-zero share of the n_items x n_items co-occurrence matrix: the
        headline diagnostic for this row of the results table.

        TODO(owner): implement.
        """
        raise NotImplementedError("ItemCFBaseline.density")
