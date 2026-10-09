# LastPiece

**Availability-aware multimodal recommendation for single-unit inventory.**

Mini project, BMSIT&M, Dept of AI & ML, BAI506, AY 2026-27.

## What this is

A recommender for online shops where **every listing is one physical unit**:
thrift, vintage, antiques, handmade. Once an item sells it is gone forever.

That breaks the standard approach completely. Collaborative filtering needs the
same item bought by many users to find a pattern. Here no item is ever bought
twice, so the user-item matrix can never fill. The cold start is not slow, it is
permanent.

We rank from image and text content alone (evaluated on Fashion Product
Images, see `docs/DATA.md`), treat availability as a ranking
stage rather than a filter at the end, and measure success primarily by how much
of the catalogue gets exposure.

## How it works

```
STAGE A — offline, once per catalogue update
  image -> encoder -> v_img              (512-d)
  text  -> encoder -> v_txt              (512-d)
  fuse  -> v = normalize((1-a)*v_img + a*v_txt)
  -> FAISS index   +   separate stock table

STAGE B — per query
  query (viewed item | text | image) -> same encoders -> q
  -> FAISS top-50  (deliberately 50, not 5)

STAGE C — our contribution
  C1 drop sold items
  C2 widen retrieval until 5 survivors exist
  C3 MMR re-rank for variety
  -> 5 available, mutually distinct results
```

Nothing is generated. The system retrieves from existing stock, which is the
only correct behaviour when every item is a real physical object.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                      # 67 tests, no downloads needed
```

The logic modules (`src/rerank/`, `src/eval/`, `src/encoders/fusion.py`,
`src/index/stock.py`) are implemented and tested. The model-dependent modules
are stubs with TODOs. See `docs/WORKPLAN.md` for who owns what.

Once the encoder and index are implemented:

```bash
# dataset: see docs/DATA.md (manual download into data/raw)
python scripts/encode_catalogue.py   # image + text embeddings, cached
python scripts/run_experiments.py    # metrics table into docs/EXPERIMENTS.md
streamlit run app/main.py            # demo
```

## Constraints we hold ourselves to

- **No paid APIs, no network at inference.** Open weights, downloaded once.
- **CPU-only works.** 8 GB laptop, no GPU required.
- **Under one second** per query on a 50k catalogue.
- **No training from scratch.** Pre-trained encoders, inference only.

The first one is a design claim we defend in the viva, not a convenience.

## Metrics

| Metric | What it asks |
|---|---|
| Precision@k | Of what we showed, how much was relevant |
| **Catalogue coverage** | How much of the catalogue ever got shown (**headline**) |
| Intra-list diversity | How different the items within one list are |
| Latency | End-to-end response time on CPU |
| Gini of exposure | How evenly exposure is spread (supports coverage, D17) |

Coverage leads because an item never shown is an item never sold.

## Documentation

- `CLAUDE.md` — working agreement for Claude Code
- `docs/DECISIONS.md` — what we chose and why, with rejected alternatives
- `docs/WORKPLAN.md` — module ownership and the build order
- `docs/EXPERIMENTS.md` — results log, written as we go
- `docs/FUTURE_WORK.md` — deliberately out of scope
