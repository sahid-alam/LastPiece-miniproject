# Work plan and module ownership

Plan and due dates live in the team doc; task status lives in GitHub issues
(https://github.com/sahid-alam/LastPiece-miniproject/issues). Team doc:
**[LastPiece — Team Plan (v2)](https://docs.google.com/document/d/19wkNwTWie4CMqVADh_WS9QJAU1NDvy3reeXPg1vkyRM/edit)** (includes the GitHub how-to and a starter AI prompt per teammate)

Key dates: Review 1 on **13 Oct 2026** (target 50%), Final on **28 Oct 2026**.

## Ownership

| Track | Owner | Files |
|---|---|---|
| **A. Data + CF baseline** | Swati Agarwal | `src/data/dataset.py`, `src/baselines/item_cf.py` |
| **B. Encoders + index** | Madiha Iram | `src/encoders/clip_encoder.py`, `src/index/faiss_index.py`, `scripts/encode_catalogue.py` |
| **C. Stage C + evaluation** | Sahid Alam | `src/rerank/*`, `src/eval/*`, `scripts/run_experiments.py` |
| **D. Demo + slides + plots** | Lucky Dhawan | `app/main.py` |

## Contracts between tracks

Change these only after telling the group.

- **Manifest columns** (A → everyone): `MANIFEST_COLUMNS` in `src/data/dataset.py`.
- **`retrieve_fn`** (B → C): `(query_vec, depth) -> (item_ids, vectors, scores)`,
  which is exactly `VectorIndex.search`. Stage C never imports FAISS, so C is
  tested against the fake retriever in `tests/conftest.py`.
- **Embedding files** (B → C, D): `paths.image_embeddings`,
  `paths.text_embeddings`, `paths.item_ids` in `config.yaml`, loaded with
  `VectorIndex.from_embeddings(..., alpha)`.
- **Session steps** (C → A): `Step(user_id, query_id, sold_after)` from
  `src/eval/simulation.py`; the CF baseline learns from them.

## Rules

1. **Branch per task, PR into main, one review, CI green.** Branch name = task
   ID, e.g. `a2-manifest`. Nobody pushes straight to main.
2. **Never commit `data/` or `.npy` files.** Share big files through the team
   Drive folder.
3. **Log every measurement** in `docs/EXPERIMENTS.md` the moment you get it.
4. **New dependency means saying why** in the PR.
5. **Design changes go in `docs/DECISIONS.md`** as a new entry.

## Using Claude Code on this repo

- It reads `CLAUDE.md` automatically. Keep that file accurate.
- Point it at your track's files and keep it there.
- Ask it to run `pytest` after changes rather than trusting the diff.
- If it proposes something that reintroduces interaction history, push back.
