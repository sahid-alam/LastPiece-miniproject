# Experiment log

Append a row every time you measure something. Unlogged results are lost
results, and the Phase 2 report gets written from this file.

Always record the config: seed, alpha, lambda, k, n_candidates, dataset size,
and the simulation settings (n_users, queries_per_user, deplete_fraction).
Sold items are scored per docs/DECISIONS.md D15 in every row.
A number without its settings is not reproducible and cannot go in the report.

---

## Main comparison

Fill this in once the pipeline runs end to end. Same simulated session, same
seed, for every row, or the comparison is meaningless.

| Date | System | alpha | lambda | P@5 | Coverage | Gini | ILD | Sold shown % | Latency (ms) | Notes |
|------|--------|-------|--------|-----|----------|------|-----|--------------|--------------|-------|
| | Item-based CF | n/a | n/a | | | | | | | record matrix density and empty-result rate |
| | Content-only (image) | 0.0 | n/a | | | | | | | no stock filter, no MMR |
| | Content-only (text) | 1.0 | n/a | | | | | | | |
| | Ours, MMR off | | 1.0 | | | | | | | isolates the availability layer |
| | Ours, full | | 0.7 | | | | | | | |

### 2026-10-12 17:06 (real, main)

real data, 44419 items, 1000 steps; seed=42, k=5, n_candidates=50, max_candidates=500, n_users=200, queries_per_user=5, deplete_fraction=0.9, encoder=patrickjohncyh/fashion-clip; sold items scored per D15; ILD in the alpha=0.5 fused space

| System | alpha | lambda | P@5 | Coverage | Gini | ILD | Sold shown % | Latency mean (ms) | Latency p95 (ms) | Short lists % |
|---|---|---|---|---|---|---|---|---|---|---|
| Content-only (image) | 0 | n/a | 0.448 | 0.058 | 0.946 | 0.179 | 45.5 | 2.35 | 3.52 | 0.0 |
| Content-only (text) | 1 | n/a | 0.469 | 0.058 | 0.946 | 0.123 | 45.3 | 3.01 | 7.12 | 0.0 |
| Ours, MMR off | 0.5 | 1 | 0.866 | 0.099 | 0.911 | 0.131 | 0.0 | 5.43 | 13.28 | 0.0 |
| Ours, full | 0.5 | 0.7 | 0.856 | 0.101 | 0.908 | 0.158 | 0.0 | 2.39 | 3.04 | 0.0 |

## Alpha sweep (fusion weight)

lambda fixed at 0.7. Looking for where text starts helping and where it starts
hurting.

| alpha | P@5 | Coverage | ILD | Notes |
|-------|-----|----------|-----|-------|
| 0.0 | | | | image only |
| 0.2 | | | | |
| 0.3 | | | | |
| 0.4 | | | | |
| 0.5 | | | | |
| 0.6 | | | | |
| 0.7 | | | | |
| 0.8 | | | | |
| 1.0 | | | | text only |

## Lambda sweep (diversity dial)

alpha fixed at the best value from above. This is the precision/diversity
trade-off curve, and it is the most important plot in the report.

| lambda | P@5 | Coverage | ILD | Notes |
|--------|-----|----------|-----|-------|
| 1.0 | | | | MMR off |
| 0.9 | | | | |
| 0.8 | | | | |
| 0.7 | | | | |
| 0.6 | | | | |
| 0.5 | | | | |
| 0.4 | | | | |
| 0.3 | | | | |

## Substitution diagnostics

Evidence that Stage C keeps lists full as stock depletes. Bucket the simulated
session by how much stock remains.

| Stock remaining | Mean items returned (ours) | Mean items returned (no substitution) | Mean widening rounds |
|-----------------|---------------------------|---------------------------------------|----------------------|
| 90-100% | | | |
| 50-70% | | | |
| 20-40% | | | |
| under 20% | | | |

## Latency

Measured on an actual laptop, CPU only. Name the machine.

| Machine | Catalogue size | Encode query (ms) | FAISS search (ms) | Stage C (ms) | Total (ms) |
|---------|----------------|-------------------|-------------------|--------------|------------|
| | | | | | |

FAISS search at depth 50 vs 500 (decides D18):

| Machine | depth 50 (ms) | depth 500 (ms) | Stage C, pool 50 (ms) | Stage C, pool 500 (ms) |
|---------|---------------|----------------|-----------------------|------------------------|
| | | | | |

## Observations

Free text. Anything surprising, anything that looks wrong, anything you want to
explain in the viva. Date each entry.

**2026-10-12: first real run (Sahid).**
- **Encoding:** all 44,419 items, FashionCLIP on Apple M1 (16 GB), `--device mps`: 10.9 min at 67.8 items/s. 0 unreadable images, 7 empty descriptions.
- **Coverage has a ceiling:** 1000 queries x k=5 = 5000 slots for 44,419 items, so the maximum possible coverage is 0.113. Ours full reaches 0.101 (about 89% of the ceiling) vs 0.058 for content-only (about 51%). Read coverage against that ceiling, not against 1.0. The same structural limit keeps Gini high for every system.
- **The P@5 gap is mostly the availability layer.** Content-only shows about 45% sold items, which D15 scores as misses. Our 0.86 vs their 0.45 is the cost of ignoring stock, which is the claim this project makes.
- **MMR at lambda=0.7 is a small effect here:** ILD 0.131 -> 0.158, P@5 0.866 -> 0.856. The lambda sweep (#17) will show the full trade-off curve.
- **Latency:** every system is far under the 1 s budget (p95 at most 13 ms for search + Stage C; a text query encode adds about 35 ms after warm-up). "Ours, MMR off" mean 5.4 ms vs "full" 2.4 ms is almost certainly warm-up/order noise, not MMR being faster. #15 measures this properly with repeats.
- **Short lists:** 0% for ours even at 90% depletion. Substitution kept every list full, but the widening-round counts aren't recorded yet (see #17).
