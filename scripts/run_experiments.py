"""Run the evaluation and write results to docs/EXPERIMENTS.md.

STATUS: STUB. Owner: Track C.

Usage:
    python scripts/run_experiments.py --config config.yaml
    python scripts/run_experiments.py --sweep alpha
    python scripts/run_experiments.py --sweep lambda

What it must do:
    1. load manifest; VectorIndex.from_embeddings(..., alpha)
    2. steps = simulate_session(...) ONCE, from config + project.random_seed
    3. for each system under test:
         - stock = Stock(all item ids)          fresh, full
         - replay the same steps (protocol in src/eval/simulation.py)
         - per step record the list, P@5, ILD, latency
    4. coverage + Gini over the whole session; append a table to
       docs/EXPERIMENTS.md with the full config

Systems under test:
    - item-based CF                 ItemCFBaseline, observe() after each step
    - content-only, image           from_embeddings(alpha=0.0).search(q, k)
    - content-only, text            from_embeddings(alpha=1.0).search(q, k)
                                    (no stock filter, no MMR: that is the point)
    - ours, MMR off                 rerank_with_availability(lambda_=1.0)
    - ours, full                    rerank_with_availability(lambda_=config)

Scoring rule for sold items (docs/DECISIONS.md D15), applied to every system:
    - P@k: relevant = relevant_items(...) & stock.available_ids() at that step,
      so a sold item shown is a miss.
    - Coverage and Gini count only items that were available when shown.
    - Also log the share of shown slots that were already sold.

Item-to-item queries exclude the query item itself (exclude_ids=[query_id]).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sweep", choices=["alpha", "lambda"], default=None)
    parser.add_argument("--systems", nargs="*", default=None)
    parser.parse_args()

    raise NotImplementedError("Track C: implement run_experiments")


if __name__ == "__main__":
    main()
