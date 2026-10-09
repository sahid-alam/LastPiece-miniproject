"""Streamlit demo.

STATUS: STUB. Owner: Track D.

Usage:
    streamlit run app/main.py

What it must show, in order of how much it matters in a review:

1.  Item-to-item. Click a product, see five similar available pieces. This is
    the main use case and needs no input from the user at all. Lead with it.
2.  Text search. Type "brown corduroy shirt medium".
3.  Image upload. Optional, rare in practice, but it demos in three seconds
    and reinforces that one model handles both modalities.
4.  A "mark as sold" button on every result. Click it, search again, watch the
    item vanish and a substitute appear in its place. This is the single most
    convincing thing in the whole demo, because it makes Stage C visible.
    Keep one Stock in st.session_state; a "reset stock" button makes a new one.
    Load the index with st.cache_resource(VectorIndex.from_embeddings).
5.  Sliders for alpha and lambda, live (alpha rebuilds the index from cached
    embeddings, well under a second). Dragging lambda from 1.0 down to 0.4
    and watching near-duplicates break apart explains MMR faster than any
    slide can.
6.  A small panel showing the diagnostics from RerankResult: how many
    candidates were retrieved, how many were dropped as sold, how many
    widening rounds were needed.

Keep it plain. The content is the point, not the styling.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Track D: implement the Streamlit demo")


if __name__ == "__main__":
    main()
