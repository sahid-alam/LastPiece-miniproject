# Design decisions

Append-only. Each entry: what we chose, what we rejected, and why. When a
decision is revisited, add a new entry rather than editing the old one, and
mark the old one superseded.

---

## D1 — Problem framing: permanent cold start

**Decision.** Treat single-unit inventory as a setting where the cold-start
condition never ends, rather than as a sparse-data variant of standard
recommendation.

**Rejected.** Framing this as "cold start with very sparse data". That framing
invites the examiner to ask why we do not simply apply an existing cold-start
method, and we would have no good answer.

**Why.** An item with stock 1 can be purchased at most once, so its column in
the user-item matrix has at most one non-zero entry, ever. This is an upper
bound imposed by the inventory model, not a data-collection problem. Published
cold-start work (bridging, graph-based) is designed to carry an item until
interactions arrive; here they never arrive.

**Consequence.** We never implement a "warm-up" path. Content is not a bridge,
it is the whole mechanism.

---

## D2 — Catalogue coverage is the primary metric

**Decision.** Report Precision@k, but treat catalogue coverage as the headline.

**Why.** In a catalogue of unique pieces, an item that never appears in any
recommendation list can never be sold. Concentrating recommendations on a
popular subset is not a mild inefficiency here, it is dead stock.

**Consequence.** We expect and accept a small Precision@k drop in exchange for
a large coverage gain. The experiments table must show both so the trade-off is
visible rather than hidden.

---

## D3 — Single fused index, not dual index

**Decision.** One FAISS index holding one fused vector per item:
`v = normalize((1 - alpha) * v_img + alpha * v_txt)`.

**Rejected (for now).** Separate image and text indexes, routed by query type.

**Why.** Simpler, half the memory, one code path. CLIP already places image and
text in a shared space, so a text query against a fused item vector is
approximately valid.

**Known weakness.** The modality gap. A pure-text query vector is not drawn
from the same distribution as a fused item vector, so similarity scores are
slightly miscalibrated. We mitigate by tuning `alpha` empirically.

**Trigger to revisit.** If text-query Precision@5 is more than ~0.08 below
item-to-item Precision@5 after the alpha sweep, build the dual index. See
`docs/FUTURE_WORK.md`.

---

## D4 — Availability in a separate mutable table

> Still holds. The "dict / SQLite" implementation detail is superseded by D13.

**Decision.** Stock is a dict / SQLite table keyed by item ID, held outside the
vector index.

**Rejected.** Encoding availability into the vector, or removing vectors from
the index on sale.

**Why.** Marking an item sold becomes one boolean write. No re-encoding, no
index rebuild, no downtime. FAISS `IndexFlatIP` does not support cheap deletion
anyway. This also keeps the index immutable between catalogue updates, which
makes results reproducible.

---

## D5 — Over-fetch then filter

**Decision.** Retrieve `n_candidates` (default 50) for a display size `k`
(default 5), then filter and re-rank down.

**Rejected.** Retrieving k and filtering afterwards.

**Why.** Filtering a list of 5 produces a list of 2 or 3. Filtering a list of 50
produces a full list of 5. Over-fetching costs microseconds because the index is
flat and small.

**Consequence.** Substitution (C2) progressively widens `n_candidates` when even
50 is not enough, which happens late in a simulated session when most stock has
sold.

---

## D6 — MMR for diversity, not a learned re-ranker

**Decision.** Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

**Rejected.** A learned diversity model; an LLM re-ranker.

**Why.** MMR is deterministic, explainable in one slide, has a single
interpretable parameter, needs no training data, and costs nothing at runtime.
We can defend every line of it in a viva. A learned re-ranker would need
interaction data, which is exactly what this domain does not have.

---

## D7 — No paid APIs anywhere in the pipeline

**Decision.** Local open weights only. No hosted inference at any stage.

**Why.** Two reasons, one practical and one academic. Practically, the target
user is a small independent seller for whom a per-query bill is not viable.
Academically, the published literature on LLM re-rankers in cold-start settings
flags inference cost as a scalability limit, so avoiding it is a defensible
position rather than a shortcut.

**Consequence.** Any proposal that introduces an API key is rejected by default.

---

## D8 — Pre-trained encoders, no training

**Decision.** FashionCLIP as the primary encoder, base CLIP as the ablation.
(ResNet-50 image-only ablation moved to FUTURE_WORK on 2026-10-09: no config,
no experiment row, and its 2048-d output breaks the 512-d convention.)

**Why.** Fine-tuning needs labelled data, GPU time and a validation protocol we
do not have the semester budget for. The contribution is Stage C, not the
encoder. Using off-the-shelf encoders also makes the result easier to reproduce.

**Consequence.** If encoder quality is a bottleneck, that is a finding to report,
not a problem to fix by training.

---

## D9 — DeepFashion-MultiModal as the evaluation dataset

> **Superseded by D11.**

**Decision.** 44,096 images, each with a paired text description. Single-unit
stock is simulated by setting quantity 1 on every item and depleting stock over
a simulated session.

**Why.** Free, public, image-and-text paired, apparel domain, large enough to be
credible and small enough to encode on a laptop.

**Known limitation.** It is a fashion dataset, not a thrift catalogue. Real
thrift listings have messier, seller-written descriptions. Acknowledge this as a
threat to validity in the report rather than overclaiming.

---

## D10 — Relevance ground truth for Precision@k

> **Superseded by D12** (attributes changed with the dataset; the idea holds).

**Decision.** An item is relevant to a query if it matches on category plus at
least one secondary attribute (colour / material / fit) derived from dataset
annotations.

**Why.** There are no real click logs. A rule over the dataset's own annotations
is transparent, reproducible, and stated openly as a proxy.

**Known limitation.** This proxy mildly favours content-based methods over CF by
construction. State this explicitly in the report; do not let it look accidental.

---

*Entries D11–D18 added 2026-10-09, before the team split the work.*

---

## D11 — Fashion Product Images replaces DeepFashion-MultiModal

**Decision.** Use Fashion Product Images (small): about 44k single products
on plain backgrounds, labelled with `articleType`, `baseColour`, `usage`,
`gender`, `season`, and a short product title.

**Rejected.** DeepFashion-MultiModal (D9).

**Why.** On inspection, DeepFashion-MultiModal is photos of *people wearing
whole outfits*, with shape / fabric / pattern labels and no per-garment
category. One image is not one listing, and the D10 relevance rule (category +
colour / material / fit) cannot be computed from its labels. Fashion Product
Images is one product per image, which is what "one listing = one unit" means,
and its labels map directly onto a relevance rule.

**Known weaknesses.** The text is a 6–8 word product title, not a seller
description, so the text modality is weaker than in a real thrift catalogue.
The laptop-sized "small" release has low-resolution images. Both go in the
report as threats to validity, alongside "clean catalogue photos, not thrift
photos".

**Tell the reviewers.** This changes the dataset named in the synopsis. The
reason is an evaluation-validity finding, not convenience.

---

## D12 — Relevance proxy on the new labels

**Decision.** Item j is relevant to query item q if `category` (articleType)
matches and at least 1 of {`colour`, `usage`} matches. Empty values never
match. q itself is never relevant.

**Why.** Same reasoning as D10: transparent, reproducible, stated openly as a
proxy. `gender` is left out because it would make relevance nearly equal to
"same category".

---

## D13 — Stock is a set of available IDs, not a quantity per item

**Decision.** `Stock` holds the set of item IDs still for sale. `mark_sold` =
remove. Reset between runs = construct a new `Stock`. One implementation.

**Rejected.** `dict[item, quantity]` with `initial_stock`, `restock()`, a
manifest `stock` column, and a second SQLite implementation for the demo.

**Why.** The premise is that quantity is always 1, so availability is a yes/no
fact. Modelling counts quietly admits multi-unit inventory. The SQLite copy
existed only so the demo would remember sales across a page reload, which is
the opposite of what you want when resetting between demos; the app keeps a
`Stock` in `st.session_state` instead.

---

## D14 — "MMR off" means lambda = 1.0, nothing else

**Decision.** No `use_mmr` flag and no `rerank.enabled` config key. The MMR-off
ablation is `lambda_=1.0`.

**Why.** With lambda = 1.0 the MMR score reduces to similarity and ties break by
retrieval order, so the output equals plain similarity ranking exactly (tested
in `test_lambda_one_is_plain_similarity_ranking`). Three switches for one
behaviour invite the experiment runner and the demo to disagree.

---

## D15 — How sold items are scored

**Decision.** Applied identically to every system:

- A sold item in a list is a miss for P@k (relevant set ∩ items available at
  that step).
- Catalogue coverage and Gini count only items that were available when shown.
- Log the share of shown slots that were already sold.

**Why.** The content-only baselines deliberately skip the stock filter. If a
sold item could count as a hit, or as exposure, the baselines would score
credit for recommending things nobody can buy. Exposure only sells an item
while it is for sale.

---

## D16 — Simulation: exogenous depletion with simulated users

**Decision.** `simulate_session` produces a fixed list of steps (user, query
item, items that sell afterwards). `n_users × queries_per_user` queries.
`deplete_fraction` of the catalogue sells, spread evenly over the session.
Sales are attributed to the querying user. What sells does **not** depend on
what any system showed.

**Rejected.** "Each shown item is bought with probability p" (endogenous).

**Why.** Endogenous sales make the stock state depend on the system, so two
systems would see different sessions even with the same seed, and the
comparison stops being like-for-like. Separately, the old setting (1000
queries × 5 shown × 0.3) sold about 1,500 of 44k items (~3%), so the late-
session state that substitution exists for was never reached.

**Cost.** We cannot claim our lists *caused* more sales. We can claim they
stayed full, valid and varied as stock ran out, which is what Stage C is for.

**Users.** Users exist so the item-CF baseline has a (user, item) log to learn
from; CF sees only steps before the current one.

---

## D17 — Gini of exposure as coverage's companion metric

**Decision.** Report Gini of exposure next to coverage.

**Why.** Coverage counts an item shown once the same as one shown 500 times.
Gini catches a system that reaches decent coverage while still piling
exposure on a few items. This supports the headline metric rather than
adding a new objective.

---

## D18 — Open question: is progressive widening (C2) worth a loop?

**Status.** Open. Decide with data by Oct 20.

**The issue.** `IndexFlatIP` scans every vector whatever the requested depth,
so asking for 500 costs about the same as asking for 50 (D5 already says
over-fetching "costs microseconds"). If so, fetching `max_candidates` once,
filtering, and passing the first `n_candidates` survivors to MMR does the same
substitution without a loop, and gives MMR the same pool size on every query.

**How to decide.** Measure FAISS search time at depth 50 vs 500 on the full
index (Track B), and Stage C time at both pool sizes. If the difference is
negligible, simplify and say why. If it is not, keep the loop and cite the
numbers. Either way the answer to "why not just fetch 500?" is ready.
