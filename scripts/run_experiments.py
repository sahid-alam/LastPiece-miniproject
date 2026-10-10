"""Run the evaluation and print / save the results table.

Owner: Track C. Replay and scoring live in src/eval/runner.py; this file is
only wiring (config, data, which systems, where the table goes).

Usage:
    python scripts/run_experiments.py --synthetic          # runs today, no data
    python scripts/run_experiments.py --config config.yaml # real data
    python scripts/run_experiments.py --sweep lambda [--synthetic]
    python scripts/run_experiments.py --sweep alpha        # real data only
    python scripts/run_experiments.py --synthetic --systems ours

What it does:
    1. build the catalogue: real (manifest + VectorIndex.from_embeddings at
       alpha) or synthetic (seeded clustered vectors, exact numpy search)
    2. steps = simulate_session(...) ONCE, from config + project.random_seed
    3. for each system under test: replay the same steps on a fresh, full
       Stock (protocol in src/eval/simulation.py), score per D15
    4. print a markdown table, save it under paths.results_dir; real mode also
       appends it, dated, with the full config, to docs/EXPERIMENTS.md.
       Synthetic mode never touches docs/EXPERIMENTS.md: those numbers say the
       plumbing works, nothing about FashionCLIP on real listings.

Systems under test (main comparison):
    - content-only, image           from_embeddings(alpha=0.0), search(q, k)
    - content-only, text            from_embeddings(alpha=1.0), search(q, k)
                                    (no stock filter, no MMR: that is the point;
                                    synthetic mode has a single content-only row)
    - ours, MMR off                 rerank_with_availability(lambda_=1.0)  (D14)
    - ours, full                    rerank_with_availability(lambda_=config)
    - item-based CF                 lands with #14 (see main_rows)

Item-to-item queries exclude the query item itself (exclude_ids=[query_id]).
ILD for every row is measured in the config-alpha fused space, so the column
compares lists, not vector spaces.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `python scripts/...` puts scripts/, not the root, on the path

from src.eval.runner import RunResult, System, content_only, ours, replay  # noqa: E402
from src.eval.simulation import simulate_session  # noqa: E402

EXPERIMENTS_MD = ROOT / "docs" / "EXPERIMENTS.md"

# Synthetic catalogue shape. Small enough to run in seconds, big enough that
# 90% depletion leaves the late-session state substitution exists for.
SYNTH_N_ITEMS = 2000
SYNTH_DIM = 512
SYNTH_CATEGORIES = [f"cat{i}" for i in range(8)]
SYNTH_COLOURS = ["black", "white", "blue", "red", "green", "brown"]
SYNTH_USAGES = ["Casual", "Formal", "Sports"]
# Vector = category + weaker colour + weaker usage + noise, so D12 relevance is
# learnable from the vectors but not trivially so.
SYNTH_W_CATEGORY, SYNTH_W_COLOUR, SYNTH_W_USAGE, SYNTH_W_NOISE = 1.0, 0.5, 0.3, 2.0
# Share of items that are near-copies of another listing: the "five identical
# black jackets" case MMR exists to break up.
SYNTH_DUP_FRACTION = 0.3
SYNTH_DUP_NOISE = 0.05

SECTIONS = {
    "main": "## Main comparison",
    "alpha": "## Alpha sweep (fusion weight)",
    "lambda": "## Lambda sweep (diversity dial)",
}


class Row(NamedTuple):
    name: str
    alpha: float | None
    lambda_: float | None
    build: Callable[[], System]  # built lazily, so sweeps hold one index at a time


class NumpyIndex:
    """Exact inner-product search with VectorIndex's search/get_vector shape."""

    # ponytail: stand-in for VectorIndex (#6) in --synthetic mode only. Full
    # argsort per query is O(n log n); fine at 2k items, real mode uses FAISS.

    def __init__(self, vectors: np.ndarray, item_ids: list[str]):
        self._vectors = vectors
        self._item_ids = item_ids
        self._pos = {iid: i for i, iid in enumerate(item_ids)}

    def search(self, query_vec: np.ndarray, depth: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        scores = self._vectors @ query_vec
        order = np.argsort(-scores, kind="stable")[:depth]
        return [self._item_ids[i] for i in order], self._vectors[order], scores[order]

    def get_vector(self, item_id: str) -> np.ndarray:
        return self._vectors[self._pos[item_id]]


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return (m / np.linalg.norm(m, axis=-1, keepdims=True)).astype(np.float32)


def synthetic_catalogue(seed: int) -> tuple[list[str], np.ndarray, list[tuple[str, str, str]]]:
    """Seeded fake catalogue: ids, float32 unit vectors (n, 512), and
    (category, colour, usage) per item, so the D12 rule can be applied."""
    rng = np.random.default_rng(seed)
    cat_dir = _unit_rows(rng.normal(size=(len(SYNTH_CATEGORIES), SYNTH_DIM)))
    col_dir = _unit_rows(rng.normal(size=(len(SYNTH_COLOURS), SYNTH_DIM)))
    use_dir = _unit_rows(rng.normal(size=(len(SYNTH_USAGES), SYNTH_DIM)))

    attrs: list[tuple[int, int, int]] = []
    vecs = np.empty((SYNTH_N_ITEMS, SYNTH_DIM), dtype=np.float32)
    for i in range(SYNTH_N_ITEMS):
        if i > 0 and rng.random() < SYNTH_DUP_FRACTION:
            src = int(rng.integers(i))
            attrs.append(attrs[src])
            vecs[i] = _unit_rows(vecs[src] + SYNTH_DUP_NOISE * _unit_rows(rng.normal(size=SYNTH_DIM)))
            continue
        c, col, u = (int(rng.integers(len(x))) for x in (SYNTH_CATEGORIES, SYNTH_COLOURS, SYNTH_USAGES))
        attrs.append((c, col, u))
        vecs[i] = _unit_rows(
            SYNTH_W_CATEGORY * cat_dir[c]
            + SYNTH_W_COLOUR * col_dir[col]
            + SYNTH_W_USAGE * use_dir[u]
            + SYNTH_W_NOISE * _unit_rows(rng.normal(size=SYNTH_DIM))
        )

    ids = [f"s{i}" for i in range(SYNTH_N_ITEMS)]
    named = [(SYNTH_CATEGORIES[c], SYNTH_COLOURS[col], SYNTH_USAGES[u]) for c, col, u in attrs]
    return ids, vecs, named


def synthetic_relevance(
    ids: list[str], attrs: list[tuple[str, str, str]], min_secondary_matches: int
) -> Callable[[str], set[str]]:
    """D12 on the synthetic labels: same category, and at least
    min_secondary_matches of (colour, usage) equal; never the query itself."""
    by_cat: dict[str, list[int]] = {}
    for i, (cat, _, _) in enumerate(attrs):
        by_cat.setdefault(cat, []).append(i)
    pos = {iid: i for i, iid in enumerate(ids)}

    @functools.cache
    def relevant(query_id: str) -> set[str]:
        q = pos[query_id]
        cat, col, use = attrs[q]
        return {
            ids[j] for j in by_cat[cat]
            if j != q and (attrs[j][1] == col) + (attrs[j][2] == use) >= min_secondary_matches
        }

    return relevant


def real_setup(cfg: dict) -> tuple[list[str], Callable[[float], object], Callable[[str], set[str]]]:
    """Wire the teammates' modules. Imported here, not at the top, so
    --synthetic never needs pandas or faiss installed."""
    from src.data.dataset import load_manifest, relevant_items
    from src.index.faiss_index import VectorIndex

    paths, rel = cfg["paths"], cfg["evaluation"]["relevance"]
    manifest = load_manifest(paths["manifest"])

    @functools.lru_cache(maxsize=2)  # main-alpha index + one other: alpha sweep stays at ~2x90 MB
    def index_at(alpha: float) -> VectorIndex:
        return VectorIndex.from_embeddings(
            paths["image_embeddings"], paths["text_embeddings"], paths["item_ids"], alpha
        )

    index_at(cfg["fusion"]["alpha"])  # fail fast before simulating
    item_ids = [str(i) for i in json.loads(Path(paths["item_ids"]).read_text())]

    @functools.cache  # every system asks about the same query items
    def relevant(query_id: str) -> set[str]:
        return relevant_items(
            manifest, query_id, rel["secondary_attributes"], rel["min_secondary_matches"]
        )

    return item_ids, index_at, relevant


def main_rows(cfg: dict, index_at: Callable[[float], object], synthetic: bool, sweep: str | None) -> list[Row]:
    k = cfg["retrieval"]["k"]
    n_cand, max_cand = cfg["retrieval"]["n_candidates"], cfg["retrieval"]["max_candidates"]
    alpha, lam = cfg["fusion"]["alpha"], cfg["rerank"]["lambda_"]

    def ours_at(a: float, l: float) -> Callable[[], System]:
        return lambda: ours(index_at(a), l, k, n_cand, max_cand)

    if sweep == "lambda":
        return [Row(f"Ours, lambda={l}", alpha, l, ours_at(alpha, l)) for l in cfg["rerank"]["lambda_sweep"]]
    if sweep == "alpha":
        return [Row(f"Ours, alpha={a}", a, lam, ours_at(a, lam)) for a in cfg["fusion"]["alpha_sweep"]]

    if synthetic:
        rows = [Row("Content-only", alpha, None, lambda: content_only(index_at(alpha), k))]
    else:
        rows = [
            Row("Content-only (image)", 0.0, None, lambda: content_only(index_at(0.0), k)),
            Row("Content-only (text)", 1.0, None, lambda: content_only(index_at(1.0), k)),
        ]
    # Item-based CF (#14): a Row whose build() wraps a fresh ItemCFBaseline and
    # sets system.after_step = lambda st: cf.observe(st.user_id, (st.query_id, *st.sold_after)).
    return rows + [
        Row("Ours, MMR off", alpha, 1.0, ours_at(alpha, 1.0)),
        Row("Ours, full", alpha, lam, ours_at(alpha, lam)),
    ]


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:g}"


def results_table(rows: list[Row], results: list[RunResult], k: int) -> str:
    lines = [
        f"| System | alpha | lambda | P@{k} | Coverage | Gini | ILD | Sold shown % "
        "| Latency mean (ms) | Latency p95 (ms) | Short lists % |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row, r in zip(rows, results):
        lines.append(
            f"| {row.name} | {_fmt(row.alpha)} | {_fmt(row.lambda_)} | {r.precision_at_k:.3f} "
            f"| {r.coverage:.3f} | {r.gini:.3f} | {r.ild:.3f} | {r.sold_shown_pct:.1f} "
            f"| {r.latency_mean_ms:.2f} | {r.latency_p95_ms:.2f} | {100 * r.short_list_rate:.1f} |"
        )
    return "\n".join(lines)


def append_to_experiments(section: str, block: str) -> None:
    """Insert block at the end of `section`, i.e. just before the next ## heading."""
    text = EXPERIMENTS_MD.read_text()
    start = text.index(section) + len(section)
    nxt = text.find("\n## ", start)
    nxt = len(text) if nxt == -1 else nxt
    EXPERIMENTS_MD.write_text(text[:nxt].rstrip("\n") + "\n\n" + block.rstrip("\n") + "\n" + text[nxt:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sweep", choices=["alpha", "lambda"], default=None)
    parser.add_argument("--systems", nargs="*", default=None,
                        help="only run rows whose name contains one of these (case-insensitive)")
    parser.add_argument("--synthetic", action="store_true",
                        help="seeded fake catalogue + numpy search; never writes docs/EXPERIMENTS.md")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed, k = cfg["project"]["random_seed"], cfg["retrieval"]["k"]
    sim = cfg["simulation"]

    if args.synthetic:
        if args.sweep == "alpha":
            raise SystemExit("--sweep alpha needs real image and text embeddings; drop --synthetic.")
        item_ids, vecs, attrs = synthetic_catalogue(seed)
        index = NumpyIndex(vecs, item_ids)
        index_at = lambda alpha: index  # noqa: E731  one vector set, alpha is not modelled
        relevant = synthetic_relevance(item_ids, attrs, cfg["evaluation"]["relevance"]["min_secondary_matches"])
    else:
        try:
            item_ids, index_at, relevant = real_setup(cfg)
        except (NotImplementedError, FileNotFoundError) as e:
            raise SystemExit(
                f"Real mode cannot run yet: {type(e).__name__}: {e}\n"
                "  load_manifest  -> #3 (Track A, Swati)\n"
                "  relevant_items -> #4 (Track A, Swati)\n"
                "  VectorIndex    -> #6 (Track B, Madiha)\n"
                "  embeddings     -> #7 (Track B, Madiha)\n"
                "Use --synthetic until those land."
            ) from e

    ild_index = index_at(cfg["fusion"]["alpha"])  # one shared space for every row's ILD
    rows = main_rows(cfg, index_at, args.synthetic, args.sweep)
    if args.systems:
        rows = [r for r in rows if any(s.lower() in r.name.lower() for s in args.systems)]

    steps = simulate_session(
        item_ids, sim["n_users"], sim["queries_per_user"], sim["deplete_fraction"], seed
    )
    results: list[RunResult] = []
    for row in rows:
        print(f"replaying {len(steps)} steps: {row.name}", file=sys.stderr)
        try:
            results.append(replay(steps, item_ids, row.build(), relevant, ild_index.get_vector, k))
        except NotImplementedError as e:
            raise SystemExit(f"{e} is still a stub; see #3/#4/#6/#7. Use --synthetic meanwhile.") from e

    mode = "synthetic" if args.synthetic else "real"
    kind = args.sweep or "main"
    now = dt.datetime.now()
    config_line = (
        f"{mode} data, {len(item_ids)} items, {len(steps)} steps; seed={seed}, k={k}, "
        f"n_candidates={cfg['retrieval']['n_candidates']}, max_candidates={cfg['retrieval']['max_candidates']}, "
        f"n_users={sim['n_users']}, queries_per_user={sim['queries_per_user']}, "
        f"deplete_fraction={sim['deplete_fraction']}, encoder={cfg['encoder']['model_name'] if not args.synthetic else 'n/a'}; "
        f"sold items scored per D15; ILD in the alpha={cfg['fusion']['alpha']} fused space"
    )
    block = f"### {now:%Y-%m-%d %H:%M} ({mode}, {kind})\n\n{config_line}\n\n{results_table(rows, results, k)}\n"
    print(block)

    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{mode}_{kind}_{now:%Y%m%d-%H%M%S}.md"
    out.write_text(block)
    print(f"saved {out}", file=sys.stderr)

    if not args.synthetic:
        append_to_experiments(SECTIONS[kind], block)
        print(f"appended to {EXPERIMENTS_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
