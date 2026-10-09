# Future work

Deliberately out of scope for this semester. Listed here so that good ideas
are recorded rather than quietly smuggled into `src/`.

Scope creep is the main failure mode on a project this size. When something
seems worth adding, write it here first and decide next week.

## Dual index for the modality gap

A text query vector and a fused item vector are not drawn from the same
distribution, so similarity scores are slightly miscalibrated (DECISIONS D3).
We mitigate by tuning alpha.

A cleaner fix is three indexes: pure image, pure text, fused. Route by query
type. Doubles memory, adds a code path.

**Trigger to build it:** text-query P@5 more than about 0.08 below item-to-item
P@5 after the alpha sweep.

## Session-based personalisation

Keep a running mean of the vectors of items the user clicked this session, and
bias ranking toward it. Gives personalisation with no account and no user-item
matrix, which fits the domain, since thrift shopping is mostly browsing.

Out of scope because it needs a session store and a second evaluation protocol.
Mention it in the viva as the natural extension; do not build it.

## Auto-tagging listings with no description

CLIP cannot generate text, but it can score an image against a fixed tag
vocabulary (colours, garments, fits, materials) zero-shot, which gives tags
without any new model. BLIP would give a real caption but needs another 250M
parameter model and produces generic output that misses exactly the details
that matter in resale: fade, wear, provenance.

Out of scope: the fusion step already degrades gracefully to image-only when
text is missing, so this is an improvement, not a requirement.

## Hard category diversity floor

A rule on top of MMR: at most 2 of 5 results from the same category. Guarantees
a floor that vector distance alone does not.

Build only if the lambda sweep shows MMR alone is insufficient. Do not promise
it in a review before measuring.

## Richer relevance ground truth

Our relevance proxy (DECISIONS D10) is a rule over dataset annotations. A small
human-annotated set, even 100 queries judged by the four of us, would be
stronger evidence. Costs a weekend.

## Scaling beyond flat search

IndexFlatIP is exact and fast at 44k. Past roughly a million items, switch to
IVF or HNSW and report the recall/latency trade-off.

## Real thrift catalogue

Fashion Product Images is a retail catalogue, not a thrift one. Real listings have
messier seller-written text and inconsistent photography. Validating on a real
catalogue is the strongest possible follow-up and the honest limitation to
state in the report.

## ResNet-50 image-only ablation

Moved out of D8. A non-CLIP image encoder would show how much the shared
image-text space matters. Needs torchvision, a 2048-d index (or a projection),
and its own experiment row. Only worth it if there is time after the main
sweeps.
