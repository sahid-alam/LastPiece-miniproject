"""CLIP / FashionCLIP encoder wrapper.

Why a wrapper rather than calling transformers directly: the rest of the
codebase must not care which model is loaded. We swap FashionCLIP for base CLIP
in the ablation by changing one config line, and tests must never download
weights.

Key facts:
  - FashionCLIP (patrickjohncyh/fashion-clip) is CLIP fine-tuned on apparel.
    It understands "oversized", "washed denim", "boxy fit"; base CLIP does not.
  - Output is 512-d for both towers. Always L2-normalised on the way out so an
    inner product is a cosine similarity everywhere downstream.
  - CLIP truncates text at 77 tokens. Long seller descriptions get cut, which
    is a real limitation worth a sentence in the report.
  - Weights download once from Hugging Face and cache locally. After that the
    pipeline is fully offline, which is a load-bearing claim for this project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .fusion import fuse, l2_normalize

__all__ = ["ClipEncoder"]

log = logging.getLogger(__name__)

# CLIP's positional embedding has exactly 77 slots; this is the model's limit,
# not a tunable.
CLIP_MAX_TOKENS = 77


class ClipEncoder:
    """Encodes images and text into one shared 512-d space."""

    def __init__(
        self,
        model_name: str = "patrickjohncyh/fashion-clip",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load weights. Separate from __init__ so constructing the object is
        cheap and tests can inject a fake without touching the network."""
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    @property
    def dim(self) -> int:
        return int(self._model.config.projection_dim)

    def encode_images(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """Encode images to L2-normalised float32 vectors, shape (n, 512).

        A corrupt or missing file becomes a zero row plus a log line rather
        than an exception: one bad JPEG must not kill a 44k batch, and a zero
        row simply falls back to the text vector at fusion time.
        """
        import torch
        from PIL import Image

        out = np.zeros((len(image_paths), self.dim), dtype=np.float32)
        for start in range(0, len(image_paths), self.batch_size):
            images, rows = [], []
            for row, path in enumerate(image_paths[start:start + self.batch_size], start):
                try:
                    with Image.open(path) as im:
                        images.append(im.convert("RGB"))
                    rows.append(row)
                except (OSError, ValueError) as e:
                    log.warning("unreadable image %s (%s); using a zero row", path, e)
            if not images:
                continue
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = _embeds(self._model.get_image_features(**inputs))
            out[rows] = feats.float().cpu().numpy()
        return l2_normalize(out, axis=1)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts to L2-normalised float32 vectors, shape (n, 512).

        An empty string becomes a zero row, not a crash: listings legitimately
        lack descriptions and fusion falls back to the image vector.
        """
        import torch

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = [(row, t) for row, t in enumerate(texts[start:start + self.batch_size], start)
                     if t and t.strip()]
            if not batch:
                continue
            rows, strings = zip(*batch)
            inputs = self._processor(
                text=list(strings), padding=True, truncation=True,
                max_length=CLIP_MAX_TOKENS, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                feats = _embeds(self._model.get_text_features(**inputs))
            out[list(rows)] = feats.float().cpu().numpy()
        return l2_normalize(out, axis=1)

    def encode_query(
        self,
        text: str | None = None,
        image_path: str | Path | None = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Encode a user query into the same space as the index, shape (512,).

        Handles all three entry points:
          - text only        -> text vector
          - image only       -> image vector
          - both             -> fused with the same alpha used at index time

        The item-to-item case does not come through here at all: that item's
        vector is already in the index, so it costs zero inference.
        """
        v_txt = self.encode_texts([text])[0] if text else None
        v_img = self.encode_images([image_path])[0] if image_path else None
        return fuse(v_img, v_txt, alpha)


def _embeds(output):
    """Projected embeddings from get_*_features.

    Older transformers return the tensor directly; newer releases may wrap it
    in a model-output object. Accept both so a teammate's newer install does
    not silently produce the wrong vectors.
    """
    if hasattr(output, "detach"):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    raise TypeError(f"unexpected output type from CLIP: {type(output).__name__}")
