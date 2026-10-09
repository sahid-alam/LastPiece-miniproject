"""Encode the whole catalogue once and cache the raw embeddings.

STATUS: STUB. Owner: Track B.

Usage:
    python scripts/encode_catalogue.py --config config.yaml
    python scripts/encode_catalogue.py --limit 200      # smoke test first

Steps:
    1. load manifest (src/data/dataset.py)
    2. encode images in batches  -> paths.image_embeddings  (slowest step)
    3. encode descriptions        -> paths.text_embeddings
    4. write the item ID list     -> paths.item_ids, aligned row-for-row

No fusing and no FAISS here. VectorIndex.from_embeddings() does both at load
time for whatever alpha is asked for, so the alpha sweep never re-encodes.

Notes for the implementer:
    - Print progress. A silent twenty-minute script looks like a hang and
      someone will kill it.
    - Cache per batch (or per 5k items) so a crash at item 40,000 does not
      cost the whole run.
    - Record items/second and the machine in docs/EXPERIMENTS.md (latency).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="encode only the first N items, for a smoke test")
    parser.parse_args()

    raise NotImplementedError("Track B: implement encode_catalogue")


if __name__ == "__main__":
    main()
