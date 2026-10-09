"""FAISS vector index.

STATUS: STUB. Owner should implement the TODOs below.

Design notes for the implementer:

  - IndexFlatIP, not IVF or HNSW. 44k vectors is tiny; exact search returns in
    milliseconds and avoids approximation error we would otherwise have to
    defend in the viva. Revisit only past ~1M items.
  - Inner product, not L2. Because every vector is L2-normalised, inner product
    IS cosine similarity, and FAISS computes it faster.
  - FAISS speaks row positions; the rest of this codebase speaks string item
    IDs. Translation happens here and nowhere else. Never let a row index
    escape this module.
  - The index is immutable between catalogue builds. Availability changes go to
    the stock table (src/index/stock.py), never to the index. See
    docs/DECISIONS.md D4.
  - Nothing FAISS-specific is saved to disk. scripts/encode_catalogue.py saves
    raw image and text embeddings; from_embeddings() fuses at the requested
    alpha and builds the flat index in well under a second. One code path for
    the app, the runner and every point of the alpha sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["VectorIndex"]


class VectorIndex:
    """Wraps a FAISS flat inner-product index with string item IDs."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._index = None
        self._item_ids: list[str] = []
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray, item_ids: Sequence[str]) -> None:
        """Build from fused vectors, shape (n, dim), aligned with item_ids.

        TODO(owner): implement.
            import faiss
            assert vectors.shape[0] == len(item_ids)
            assert vectors.shape[1] == self.dim
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self._item_ids = list(item_ids)
            self._vectors = vectors  # kept for MMR, which needs the vectors

        Keep self._vectors: Stage C re-ranks over candidate vectors, and
        fetching them back out of FAISS row by row is slower than holding the
        array. At 44k x 512 float32 it is about 90 MB, which fits our budget.
        """
        raise NotImplementedError("VectorIndex.build")

    def search(
        self,
        query_vec: np.ndarray,
        depth: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return the `depth` nearest items.

        This signature is exactly what rerank_with_availability expects as its
        retrieve_fn, so the two compose without an adapter.

        Returns:
            (item_ids, vectors, scores) where vectors is (m, dim) and scores is
            (m,) cosine similarities, m = min(depth, index size), ordered
            descending by score.

        TODO(owner): implement.
            q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
            scores, idx = self._index.search(q, min(depth, self._index.ntotal))
            idx, scores = idx[0], scores[0]
            keep = idx >= 0          # FAISS pads with -1 when short
            idx, scores = idx[keep], scores[keep]
            return ([self._item_ids[i] for i in idx], self._vectors[idx], scores)
        """
        raise NotImplementedError("VectorIndex.search")

    def get_vector(self, item_id: str) -> np.ndarray:
        """Fetch one item's stored vector, shape (dim,).

        Needed for the item-to-item path, which is the main use case: the user
        is viewing an item, so its vector is already indexed and no encoding is
        required.

        TODO(owner): implement with an id -> position dict built in build().
        """
        raise NotImplementedError("VectorIndex.get_vector")

    @classmethod
    def from_embeddings(
        cls,
        image_emb_path: str | Path,
        text_emb_path: str | Path,
        ids_path: str | Path,
        alpha: float,
    ) -> "VectorIndex":
        """Load cached embeddings, fuse at alpha, build. Used by the app and
        the experiment runner.

        TODO(owner): implement.
            img = np.load(image_emb_path); txt = np.load(text_emb_path)
            ids = json.load(open(ids_path))
            idx = cls(dim=img.shape[1])
            idx.build(fuse_batch(img, txt, alpha), ids)
            return idx
        For the content-only baselines, alpha=0.0 is image-only and alpha=1.0
        is text-only; no separate baseline class needed.
        """
        raise NotImplementedError("VectorIndex.from_embeddings")

    def __len__(self) -> int:
        return len(self._item_ids)
