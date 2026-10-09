"""Stock table: the single source of truth for item availability.

Held deliberately outside the vector index (docs/DECISIONS.md D4). Marking an
item sold is one set removal: no re-encoding, no index rebuild, no downtime.
Say this out loud in the viva, it is the practical payoff of the design.

A set, not a quantity per item (D13). Every listing is exactly one unit, so
"in stock" is a yes/no fact. Modelling counts would quietly admit the
multi-unit inventory this project is built against.

Resetting between experiment runs means constructing a new Stock, never
"restocking": real single-unit inventory does not come back.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["Stock"]


class Stock:
    """The set of item IDs that are still for sale.

    Items not in the set are unavailable, including IDs it has never seen. An
    unknown ID is a bug somewhere upstream, and failing closed means it shows
    up as a missing recommendation rather than a dead link in the UI.
    """

    def __init__(self, item_ids: Iterable[str] = ()):
        self._available: set[str] = set(item_ids)

    def is_available(self, item_id: str) -> bool:
        return item_id in self._available

    def mark_sold(self, item_id: str) -> None:
        """Remove an item for good. Selling twice, or selling an unknown ID,
        is a no-op."""
        self._available.discard(item_id)

    def available_ids(self) -> set[str]:
        return set(self._available)

    def n_available(self) -> int:
        return len(self._available)
