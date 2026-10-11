"""Encode the whole catalogue once and cache the raw embeddings.

Usage:
    python scripts/encode_catalogue.py --limit 200      # smoke test first
    python scripts/encode_catalogue.py                  # full run
    python scripts/encode_catalogue.py --device mps     # Apple Silicon GPU

Steps:
    1. load manifest (src/data/dataset.py)
    2. encode images in chunks  -> paths.image_embeddings  (slowest step)
    3. encode descriptions       -> paths.text_embeddings
    4. write the item ID list    -> paths.item_ids, aligned row-for-row

No fusing and no FAISS here. VectorIndex.from_embeddings() does both at load
time for whatever alpha is asked for, so the alpha sweep never re-encodes.

Resumable: each chunk is cached under data/processed/encode_cache/<run>/, where
<run> names the model and a fingerprint of the manifest's item IDs. A crash at
item 40,000 costs one chunk, and a changed manifest or model can never reuse
stale chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_manifest  # noqa: E402
from src.encoders.clip_encoder import ClipEncoder  # noqa: E402


def _run_dir(processed_dir: Path, model_name: str, item_ids: list[str]) -> Path:
    fingerprint = hashlib.sha1("\n".join(item_ids).encode()).hexdigest()[:10]
    return processed_dir / "encode_cache" / f"{model_name.replace('/', '__')}_{fingerprint}"


def _limited(path: str, limit: int | None) -> Path:
    """Smoke-test outputs get their own names so they never clobber a full run."""
    p = Path(path)
    return p if limit is None else p.with_name(f"{p.stem}.limit{limit}{p.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="encode only the first N items, for a smoke test")
    parser.add_argument("--device", default=None, help="override encoder.device")
    parser.add_argument("--chunk", type=int, default=5000,
                        help="items per cached chunk (resume granularity)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths, enc_cfg = cfg["paths"], cfg["encoder"]

    manifest = load_manifest(paths["manifest"])
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    item_ids = manifest["item_id"].tolist()
    raw_dir = Path(paths["raw_dir"])
    image_paths = [raw_dir / p for p in manifest["image_path"]]
    texts = manifest["description"].tolist()

    encoder = ClipEncoder(
        model_name=enc_cfg["model_name"],
        device=args.device or enc_cfg["device"],
        batch_size=enc_cfg["batch_size"],
    )
    print(f"Loading {encoder.model_name} on {encoder.device} ...", flush=True)
    encoder.load()

    run_dir = _run_dir(Path(paths["image_embeddings"]).parent, encoder.model_name, item_ids)
    run_dir.mkdir(parents=True, exist_ok=True)
    n = len(item_ids)
    print(f"Encoding {n} items in chunks of {args.chunk}; cache: {run_dir}", flush=True)

    t0 = time.perf_counter()
    done_fresh = 0
    for start in range(0, n, args.chunk):
        end = min(start + args.chunk, n)
        img_f = run_dir / f"img_{start:06d}.npy"
        txt_f = run_dir / f"txt_{start:06d}.npy"
        if img_f.exists() and txt_f.exists():
            print(f"  {end:>6}/{n}  cached", flush=True)
            continue
        np.save(txt_f, encoder.encode_texts(texts[start:end]))
        np.save(img_f, encoder.encode_images(image_paths[start:end]))
        done_fresh += end - start
        rate = done_fresh / (time.perf_counter() - t0)
        eta_min = (n - end) / rate / 60 if rate else float("nan")
        print(f"  {end:>6}/{n}  {rate:6.1f} items/s  ETA {eta_min:5.1f} min", flush=True)

    starts = range(0, n, args.chunk)
    img = np.concatenate([np.load(run_dir / f"img_{s:06d}.npy") for s in starts])
    txt = np.concatenate([np.load(run_dir / f"txt_{s:06d}.npy") for s in starts])
    assert img.shape == txt.shape == (n, encoder.dim)

    out_img = _limited(paths["image_embeddings"], args.limit)
    out_txt = _limited(paths["text_embeddings"], args.limit)
    out_ids = _limited(paths["item_ids"], args.limit)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_img, img)
    np.save(out_txt, txt)
    out_ids.write_text(json.dumps(item_ids))

    elapsed = time.perf_counter() - t0
    zero_img = int((~img.any(axis=1)).sum())
    zero_txt = int((~txt.any(axis=1)).sum())
    print(f"Done in {elapsed / 60:.1f} min "
          f"({done_fresh / elapsed if elapsed and done_fresh else 0:.1f} items/s fresh).")
    print(f"Unreadable images (zero rows): {zero_img}; empty descriptions: {zero_txt}")
    print(f"Wrote {out_img}, {out_txt}, {out_ids}")


if __name__ == "__main__":
    main()
