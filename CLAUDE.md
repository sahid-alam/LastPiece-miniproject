# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

An **availability-aware multimodal recommender for single-unit inventory**.

Academic mini project, BMSIT&M, Dept of AI & ML, course BAI506, AY 2026-27.

The domain is thrift / vintage / antique resale, where **every listing is one
physical unit**. Once sold, it never returns. This means:

- Collaborative filtering is **structurally impossible**, not merely weak.
  No item is ever purchased twice, so the user-item matrix can never fill.
- The cold-start condition is **permanent**, not transient. Every published
  cold-start mitigation assumes items eventually warm up. Ours never do.
- **Catalogue coverage** is the headline metric, not accuracy. Every item needs
  its own single buyer, so an item never shown is an item never sold.

Keep this framing in mind. If a proposed change quietly reintroduces an
assumption that items accumulate interaction history, it is wrong for this
project even if it would improve a benchmark number.

## The two objectives (fixed, do not expand)

1. **Retrieval pipeline.** Encode each listing's image and text into one shared
   512-d space with a pre-trained vision-language model; retrieve in real time
   via approximate nearest-neighbour search; use zero interaction history.
2. **Availability-aware re-ranking + evaluation.** Stock filtering, candidate
   substitution, diversity control. Benchmark against item-based CF and
   content-only baselines on Precision@k, catalogue coverage, intra-list
   diversity, latency.

Scope creep is the main risk on a semester project. Anything outside these two
objectives goes in `docs/FUTURE_WORK.md`, not into `src/`.

## Hard constraints

- **No paid APIs. No network calls at inference time.** No OpenAI, no Anthropic,
  no hosted inference. Everything runs from locally downloaded open weights.
  This is a stated design claim in the synopsis and the viva; breaking it breaks
  the argument.
- **CPU-only must work.** Target: 8 GB RAM laptop, no GPU. GPU may be used to
  speed up batch encoding, never required for serving.
- **Sub-second query latency** end to end for a 50k-item catalogue.
- **No model training from scratch.** Pre-trained weights, inference only.
- Python 3.10+.

## Architecture

Three stages. Stage C is the project's actual contribution.

```
STAGE A — offline indexing (once per catalogue update)
  listing (image + text + stock=1)
    -> image encoder (FashionCLIP / CLIP)   -> v_img (512)
    -> text  encoder (same model)           -> v_txt (512)
    -> fuse: v = normalize((1-a)*v_img + a*v_txt)
    -> FAISS IndexFlatIP  +  separate stock table

STAGE B — online query (per request)
  query = viewed item | text | image
    -> same encoders, same fusion -> q (512)
    -> FAISS search, top-N (N=50, deliberately >> k)
    -> candidate set

STAGE C — availability-aware re-ranking  [OUR CONTRIBUTION]
  C1 availability filter : drop stock == 0
  C2 substitution        : widen N until >= k survivors
  C3 diversity re-rank   : MMR, lambda tunable
  C4 final top-k         : k available, mutually distinct
```

Design decisions worth preserving:

- **One fused vector per item**, not separate image/text indexes. A text query
  compared against a fused item vector leaves a modality gap; we handle that by
  tuning `alpha`, and a dual-index fallback is documented in
  `docs/FUTURE_WORK.md` if measurements demand it.
- **Stock lives in its own table, never in the vector.** Marking an item sold is
  a single boolean flip. No re-encoding, no index rebuild. Say this in the viva.
- **Over-fetch then filter**, never filter then fetch. Filtering 50 candidates
  down to 5 keeps the list full; filtering a list of 5 shortens it.

## Repo layout

```
src/data/        manifest build, relevance ground truth
src/encoders/    CLIP / FashionCLIP wrappers, fusion
src/index/       FAISS build + search, stock set
src/rerank/      MMR and the availability pipeline   <- the contribution
src/baselines/   item-based CF (content-only = index at alpha 0/1)
src/eval/        metrics, session simulation
app/             Streamlit demo
scripts/         CLI entry points (encode, evaluate)
tests/           pytest
docs/            decisions, future work, experiment log
```

## Conventions

- Type hints on every public function. Docstrings explain *why*, not *what*.
- Pure functions in `src/rerank/mmr.py`, `src/eval/metrics.py` and
  `src/eval/simulation.py`: no I/O, no
  globals, fully unit-testable. These are the parts the examiner will probe.
- All tunables (`alpha`, `lambda`, `k`, `n_candidates`) are parameters with
  defaults in `config.yaml`. **Never hardcode a magic number in a function
  body.** We have to sweep these and plot the results.
- Vectors are `np.float32`, L2-normalised, shape `(n, 512)`. Normalise once at
  creation so inner product == cosine similarity everywhere.
- Item IDs are strings. Never positional indices across module boundaries;
  FAISS row position is an internal detail of `src/index/`.
- `random_seed` from config everywhere something is sampled.

## Testing

`pytest tests/ -v` must pass before any commit.

Logic that must stay covered: MMR selection order, availability filtering,
substitution widening, each metric, the empty/short-catalogue edge cases.

Do not write tests that download models or the dataset. Use the synthetic
fixtures in `tests/conftest.py`.

## Working style for Claude

- Read `docs/DECISIONS.md` before proposing an architectural change. If a
  decision there is being revisited, say so explicitly and update the file.
- Prefer small diffs. One module at a time. Four people are working in parallel
  across module boundaries.
- When you finish something measurable, append a row to
  `docs/EXPERIMENTS.md`. Unlogged results are lost results.
- If a design choice has a trade-off, state it in the response. Do not silently
  pick. The team has to defend these choices in a viva.
- Do not add dependencies without saying why. Every added package is something
  a teammate has to install on a laptop that may be fragile.

## Deliverables calendar

- Phase 0 (done): synopsis + proposal presentation.
- Phase 1: working retrieval pipeline + baselines + first metrics table.
- Phase 2: full Stage C + parameter sweeps + demo + report.

Code is graded alongside the report. Keep `docs/EXPERIMENTS.md` current so the
report can be written from it rather than reconstructed from memory.
