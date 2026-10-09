"""CLIP / FashionCLIP encoder wrapper.

STATUS: STUB. Owner should implement the TODOs below.

Why a wrapper rather than calling transformers directly: the rest of the
codebase must not care which model is loaded. We swap FashionCLIP for base CLIP
in the ablation by changing one config line, and tests must never download
weights.

Key facts for whoever implements this:
  - FashionCLIP (patrickjohncyh/fashion-clip) is CLIP fine-tuned on apparel.
    It understands "oversized", "washed denim", "boxy fit"; base CLIP does not.
  - Output is 512-d for both towers. Always L2-normalise on the way out so an
    inner product is a cosine similarity everywhere downstream.
  - CLIP truncates text at 77 tokens. Long seller descriptions get cut, which
    is a real limitation worth a sentence in the report.
  - Weights download once from Hugging Face and cache locally. After that the
    pipeline is fully offline, which is a load-bearing claim for this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["ClipEncoder"]


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
        cheap and tests can inject a fake without touching the network.

        TODO(owner): implement.
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        """
        raise NotImplementedError("ClipEncoder.load")

    def encode_images(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """Encode images to L2-normalised float32 vectors, shape (n, 512).

        TODO(owner): implement.
            - batch by self.batch_size
            - PIL open + .convert("RGB"); a corrupt file must not kill a 44k
              batch, so log and emit a zero row instead of raising
            - torch.no_grad()
            - model.get_image_features(...)
            - normalise rows, return float32
        """
        raise NotImplementedError("ClipEncoder.encode_images")

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts to L2-normalised float32 vectors, shape (n, 512).

        TODO(owner): implement.
            - processor(text=..., truncation=True, max_length=77, padding=True)
            - model.get_text_features(...)
            - empty string -> zero row, not a crash; items legitimately lack
              descriptions and the fusion step handles a zero text vector
        """
        raise NotImplementedError("ClipEncoder.encode_texts")

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

        Note the item-to-item case does not come through here at all: that
        item's vector is already in the index, so there is nothing to encode.
        That is the main use case and it costs zero inference.

        TODO(owner): implement using fuse() from .fusion.
        """
        raise NotImplementedError("ClipEncoder.encode_query")
