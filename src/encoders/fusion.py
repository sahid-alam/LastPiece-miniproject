"""Image/text embedding fusion.

Implemented, not a stub: this is small, pure, and needed by both the encoder
and the tests.

    v = normalize((1 - alpha) * v_img + alpha * v_txt)

alpha is the single dial controlling how much the description matters relative
to the photograph. 0.0 is image-only, 1.0 is text-only. It is swept in the
experiments (config.yaml fusion.alpha_sweep) and exposed as a slider in the
demo, because watching results change as you drag it is the clearest way to
show what the fused space actually is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fuse", "fuse_batch", "l2_normalize"]


def l2_normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise along an axis, leaving zero vectors as zeros.

    Zero rows are legitimate here: an item with no description gets a zero text
    vector. Dividing by zero would produce NaNs that silently poison the index,
    so we leave them alone and let fuse() fall back to the other modality.
    """
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (v / norm).astype(np.float32)


def fuse(
    v_img: np.ndarray | None,
    v_txt: np.ndarray | None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Fuse one image and one text vector into a single unit vector.

    Args:
        v_img: image embedding, shape (d,), or None.
        v_txt: text embedding, shape (d,), or None.
        alpha: text weight in [0, 1].

    Returns:
        L2-normalised float32 vector, shape (d,).

    Graceful degradation is the point here. A listing with no description still
    has to be recommendable, so a missing or zero text vector collapses to the
    image vector rather than producing garbage. In a thrift catalogue where
    sellers write inconsistently, this case is common, not exotic.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    img_ok = v_img is not None and np.any(v_img)
    txt_ok = v_txt is not None and np.any(v_txt)

    if not img_ok and not txt_ok:
        raise ValueError("need at least one non-zero modality to fuse")
    if not txt_ok:
        return l2_normalize(np.asarray(v_img, dtype=np.float32))
    if not img_ok:
        return l2_normalize(np.asarray(v_txt, dtype=np.float32))

    a = np.asarray(v_img, dtype=np.float32)
    b = np.asarray(v_txt, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: image {a.shape} vs text {b.shape}")

    return l2_normalize((1.0 - alpha) * a + alpha * b)


def fuse_batch(
    img_vecs: np.ndarray,
    txt_vecs: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Fuse aligned batches, shape (n, d) each, into (n, d).

    Rows where one modality is all zeros fall back to the other, matching
    fuse(). A row with both modalities zero stays zero rather than raising:
    in a 44k batch one unreadable image should not abort the build, and a zero
    row simply never matches anything.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if img_vecs.shape != txt_vecs.shape:
        raise ValueError(
            f"shape mismatch: image {img_vecs.shape} vs text {txt_vecs.shape}"
        )

    A = np.asarray(img_vecs, dtype=np.float32)
    B = np.asarray(txt_vecs, dtype=np.float32)

    img_ok = np.any(A, axis=1)
    txt_ok = np.any(B, axis=1)

    fused = (1.0 - alpha) * A + alpha * B
    # Single-modality rows: use whichever vector is real.
    fused[img_ok & ~txt_ok] = A[img_ok & ~txt_ok]
    fused[~img_ok & txt_ok] = B[~img_ok & txt_ok]

    return l2_normalize(fused, axis=1)
