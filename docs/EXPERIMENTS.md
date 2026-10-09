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
