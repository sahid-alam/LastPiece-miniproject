"""Synthetic product index for the Phase 1 demo.

Why this exists: the Streamlit demo must work on a machine that has no
real embeddings, no manifest, and no downloaded FashionCLIP weights.
This module generates ~300 plausible synthetic products with clustered
embeddings so the UI, the availability-aware reranker, and the MMR
diversity control can all be exercised without any external data.

Interface contract: every public method matches VectorIndex exactly.
Switching from FakeIndex to VectorIndex.from_embeddings() in app/main.py
requires changing one line, nothing else.

Random seed is fixed so every demo run and every test run produces the
same products, the same cluster structure, and the same search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["FakeProduct", "FakeIndex", "build_fake_index"]

# --------------------------------------------------------------------------- #
# Cluster definitions                                                          #
# --------------------------------------------------------------------------- #

# Each entry: (label, colour, n_items)
# Clusters are deliberately unequal in size so coverage metrics are
# non-trivial. The three near-duplicate clusters (jackets, sneakers,
# tote bags) are large to make MMR's benefit visible: lambda=1.0 returns
# many nearly identical items; lambda=0.5 breaks them apart.
_CLUSTERS = [
    ("Denim Jacket",         "Blue",    30),
    ("Leather Jacket",       "Black",   20),
    ("Running Sneakers",     "White",   25),
    ("Canvas Sneakers",      "Red",     20),
    ("Tote Bag",             "Beige",   25),
    ("Crossbody Bag",        "Brown",   18),
    ("Floral Dress",         "Pink",    22),
    ("Linen Shirt",          "White",   18),
    ("Slim Trousers",        "Navy",    20),
    ("Cargo Pants",          "Olive",   18),
    ("Wool Coat",            "Grey",    20),
    ("Graphic Tee",          "Black",   20),
    ("Striped Polo",         "Blue",    18),
    ("Corduroy Blazer",      "Brown",   15),
    ("Silk Scarf",           "Multi",   13),
]

DIM = 512
SEED = 42


@dataclass
class FakeProduct:
    """Metadata for one synthetic catalogue item."""

    item_id: str
    title: str
    colour: str
    # Placeholder image as a small RGB array (H, W, 3) uint8.
    # Generated lazily: the Streamlit app converts this to a PIL Image.
    thumbnail: np.ndarray = field(repr=False)


def _random_colour_array(colour_name: str, rng: np.random.Generator) -> np.ndarray:
    """Return a 64×64 solid-colour thumbnail with a small noise texture.

    Using a colour derived from the cluster label makes the grid immediately
    readable: all denim jackets are blue, all tote bags are beige, etc.
    This lets a reviewer see cluster structure without any real images.
    """
    palette = {
        "Blue":   (70,  130, 180),
        "Black":  (30,   30,  30),
        "White":  (240, 240, 240),
        "Red":    (200,  50,  50),
        "Beige":  (210, 190, 160),
        "Brown":  (130,  80,  40),
        "Pink":   (220, 150, 170),
        "Navy":   (20,   40, 100),
        "Olive":  (100, 120,  50),
        "Grey":   (140, 140, 140),
        "Multi":  (180, 100, 220),
    }
    base = np.array(palette.get(colour_name, (160, 160, 160)), dtype=np.uint8)
    noise = rng.integers(-15, 16, size=(64, 64, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _generate_synthetic_data(
    seed: int = SEED,
) -> tuple[list[FakeProduct], np.ndarray, dict[str, int]]:
    """Build products, embeddings, and ID→row mapping.

    Cluster structure: each cluster has a random unit-sphere centre; items
    in the cluster are that centre plus small Gaussian noise, renormalised.
    A fraction of items are inter-cluster outliers (larger noise). This gives
    the demo the near-duplicate behaviour that makes MMR's benefit visible.

    Returns:
        products   – ordered list of FakeProduct (row i == products[i])
        vectors    – (n, 512) float32, L2-normalised
        id_to_row  – dict[item_id, row_index]
    """
    rng = np.random.default_rng(seed)

    products: list[FakeProduct] = []
    vectors: list[np.ndarray] = []

    cluster_centres = []
    for label, colour, n in _CLUSTERS:
        # Random unit vector in 512-d as the cluster centre.
        centre = rng.standard_normal(DIM).astype(np.float32)
        centre /= np.linalg.norm(centre)
        cluster_centres.append(centre)

        for i in range(n):
            item_id = f"{label.replace(' ', '_').lower()}_{i:03d}"
            title = f"{colour} {label}" + (f" #{i}" if i > 0 else "")

            # Tight cluster: small noise → near-duplicate behaviour
            noise_scale = 0.05 if i < int(n * 0.8) else 0.30
            vec = centre + rng.standard_normal(DIM).astype(np.float32) * noise_scale
            vec = (vec / np.linalg.norm(vec)).astype(np.float32)

            thumbnail = _random_colour_array(colour, rng)
            products.append(FakeProduct(item_id=item_id, title=title,
                                        colour=colour, thumbnail=thumbnail))
            vectors.append(vec)

    vecs = np.stack(vectors, axis=0).astype(np.float32)  # (n, 512)

    # Sanity check: all rows normalised.
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "embeddings must be L2-normalised"

    id_to_row: dict[str, int] = {p.item_id: i for i, p in enumerate(products)}
    return products, vecs, id_to_row


# --------------------------------------------------------------------------- #
# FakeIndex                                                                    #
# --------------------------------------------------------------------------- #

class FakeIndex:
    """Drop-in replacement for VectorIndex used during Phase 1.

    Every method name, argument name, return type, and edge-case matches the
    real VectorIndex so the UI does not need to know which backend it uses.
    The only difference is that this index is built from synthetic embeddings
    instead of from precomputed FashionCLIP embeddings.
    """

    def __init__(self, seed: int = SEED) -> None:
        self._products, self._vectors, self._id_to_row = (
            _generate_synthetic_data(seed)
        )

    # ------------------------------------------------------------------ #
    # VectorIndex interface                                                 #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_vec: np.ndarray,
        depth: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return the `depth` most similar items by inner-product similarity.

        Signature matches VectorIndex.search exactly so rerank_with_availability
        can receive either this or the real index as its retrieve_fn without
        modification.

        Args:
            query_vec: shape (512,), float32, L2-normalised.
            depth: number of results requested.

        Returns:
            (item_ids, vectors, scores) ordered descending by score.
            item_ids has length min(depth, n_items).
            vectors has shape (m, 512), scores has shape (m,).

        Raises:
            ValueError: if query_vec has wrong shape or wrong dtype,
                        matching the error convention the tests expect.
        """
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != DIM:
            raise ValueError(
                f"query_vec must have shape ({DIM},), got {q.shape}"
            )
        if depth <= 0:
            return [], np.empty((0, DIM), dtype=np.float32), np.empty(0, dtype=np.float32)

        depth = min(depth, len(self._products))
        scores = self._vectors @ q           # (n,) inner products
        top_idx = np.argsort(-scores)[:depth]

        item_ids = [self._products[i].item_id for i in top_idx]
        vecs = self._vectors[top_idx]
        scs = scores[top_idx].astype(np.float32)
        return item_ids, vecs, scs

    def get_vector(self, item_id: str) -> np.ndarray:
        """Return the stored 512-d vector for one item, shape (512,).

        Raises:
            KeyError: if item_id is unknown, consistent with how a dict
                      lookup fails so the caller gets a clear error.
        """
        try:
            row = self._id_to_row[item_id]
        except KeyError:
            raise KeyError(f"Unknown item_id: {item_id!r}") from None
        return self._vectors[row]

    def __len__(self) -> int:
        return len(self._products)

    # ------------------------------------------------------------------ #
    # Extra helpers used only by the Streamlit UI                          #
    # ------------------------------------------------------------------ #

    def get_product(self, item_id: str) -> FakeProduct:
        """Return FakeProduct metadata for one item.

        The real-data path has a manifest DataFrame for this role; the demo
        path has this method. The UI calls a helper function that abstracts
        the difference (see app/main.py).
        """
        try:
            row = self._id_to_row[item_id]
        except KeyError:
            raise KeyError(f"Unknown item_id: {item_id!r}") from None
        return self._products[row]

    def all_item_ids(self) -> list[str]:
        """All item IDs in insertion order (cluster order).

        Used to initialise Stock with the full catalogue and to populate
        the product grid in the UI.
        """
        return [p.item_id for p in self._products]

    def all_products(self) -> list[FakeProduct]:
        """All FakeProduct objects in insertion order."""
        return list(self._products)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_fake_index(seed: int = SEED) -> FakeIndex:
    """Construct and return the synthetic index.

    Kept as a module-level function so st.cache_resource can wrap it
    without caching the class constructor itself, which Streamlit does not
    support cleanly.
    """
    return FakeIndex(seed=seed)
