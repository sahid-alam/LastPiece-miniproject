"""FAISS vector index.

Design notes:

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

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from src.encoders.fusion import fuse_batch

__all__ = ["VectorIndex"]


def _import_faiss():
    """Import faiss so it can live in the same process as torch on macOS.

    torch and faiss each ship their own copy of the OpenMP runtime. When both
    load into one process (the demo does: text search needs the encoder, every
    search needs the index), macOS builds crash with a segfault unless
    (1) torch initialises OpenMP first, (2) the duplicate-runtime check is
    relaxed, and (3) faiss runs single-threaded. Tested on Apple Silicon: same
    results as the faiss-only process, ~3 ms per search at 44k items.

    "Single-threaded" must hold in every thread: OpenMP's thread count is a
    per-thread setting, and Streamlit runs each rerun on its own thread. So
    OMP_NUM_THREADS sets the process-wide default before either runtime starts,
    and search() re-applies the limit on whichever thread calls it.

    ponytail: KMP_DUPLICATE_LIB_OK is LLVM's documented "unsafe" escape hatch.
    Ceiling: it relies on both runtimes behaving when one is idle. Upgrade
    path if it ever misbehaves: replace the flat index with a numpy matmul
    (exact same results at our scale) or install both libraries from
    conda-forge so they share one OpenMP.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch  # noqa: F401  only for its side effect: OpenMP init order
    except ImportError:
        pass  # no torch (e.g. CI, the experiment runner on a lean install): no clash
    import faiss

    faiss.omp_set_num_threads(1)
    return faiss


class VectorIndex:
    """Wraps a FAISS flat inner-product index with string item IDs."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._index = None
        self._item_ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray, item_ids: Sequence[str]) -> None:
        """Build from fused vectors, shape (n, dim), aligned with item_ids.

        The vectors are kept alongside FAISS because Stage C re-ranks over
        candidate vectors, and fetching them back out of FAISS row by row is
        slower than holding the array. At 44k x 512 float32 that is ~90 MB.
        """
        faiss = _import_faiss()

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {vectors.shape}")
        if vectors.shape[0] != len(item_ids):
            raise ValueError(
                f"{vectors.shape[0]} vectors but {len(item_ids)} item IDs"
            )
        pos = {iid: i for i, iid in enumerate(item_ids)}
        if len(pos) != len(item_ids):
            raise ValueError("duplicate item IDs; every listing must be unique")

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors)
        self._item_ids = list(item_ids)
        self._pos = pos
        self._vectors = vectors

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
        """
        if self._index is None:
            raise RuntimeError("index is empty; call build() or from_embeddings()")
        depth = min(int(depth), self._index.ntotal)
        if depth <= 0:
            return [], self._vectors[:0], np.empty(0, dtype=np.float32)

        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        self._faiss.omp_set_num_threads(1)  # per-thread setting; see _import_faiss
        scores, idx = self._index.search(q, depth)
        idx, scores = idx[0], scores[0]
        keep = idx >= 0  # FAISS pads with -1 when it has fewer results
        idx, scores = idx[keep], scores[keep]
        return [self._item_ids[i] for i in idx], self._vectors[idx], scores

    def get_vector(self, item_id: str) -> np.ndarray:
        """Fetch one item's stored vector, shape (dim,).

        The item-to-item path is the main use case: the user is viewing an
        item, so its vector is already indexed and no encoding is required.
        """
        return self._vectors[self._pos[item_id]]

    def all_item_ids(self) -> list[str]:
        """Every indexed ID, e.g. to seed a full Stock. Same name as the demo's
        FakeIndex so the app swaps one for the other without a branch."""
        return list(self._item_ids)

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

        For the content-only baselines, alpha=0.0 is image-only and alpha=1.0
        is text-only; no separate baseline class needed.
        """
        img = np.load(image_emb_path)
        txt = np.load(text_emb_path)
        ids = json.loads(Path(ids_path).read_text())
        if img.shape != txt.shape or img.shape[0] != len(ids):
            raise ValueError(
                f"embeddings out of step: image {img.shape}, text {txt.shape}, "
                f"{len(ids)} IDs. Were they produced by the same encode run?"
            )
        index = cls(dim=img.shape[1])
        index.build(fuse_batch(img, txt, alpha), ids)
        return index

    def __len__(self) -> int:
        return len(self._item_ids)
