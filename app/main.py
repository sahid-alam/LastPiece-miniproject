"""LastPiece — Streamlit demo.

Owner: Track D (Lucky Dhawan).

Usage — synthetic demo (no real data required):
    streamlit run app/main.py

Usage — real-data mode (requires embeddings + manifest from Team Drive):
    DATA_MODE=real streamlit run app/main.py

What the demo shows, in priority order (from the original stub):

1.  Item-to-item: click a product, see up to k similar available pieces.
    The query item is automatically excluded from its own results.
2.  Text search: type "brown leather bag" to get semantically similar items.
    Requires real embeddings + model weights; shown as unavailable otherwise.
3.  Lambda slider (0.3–1.0): drag left for diversity, right for pure
    relevance. At 1.0 (MMR off per D14) near-duplicates dominate. At 0.5
    they break apart. This is the single clearest way to explain MMR.
4.  Mark Sold: click on a result to sell it. Watch the item vanish and a
    substitute appear (Stage C substitution). Resets with a button.
5.  Diagnostics panel: live n_retrieved, n_dropped_sold, n_widening_rounds,
    n_available from the actual RerankResult — never fabricated.

Design decisions preserved:
  - Stock lives in st.session_state, never recreated unless Reset clicked.
  - Index is st.cache_resource — built once per server process.
  - rerank_with_availability() is the sole authority for filtering/widening.
  - Synthetic mode and real-data mode are explicit, never silently mixed.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
from typing import Callable

# Ensure repository root is on sys.path so 'app.*' and 'src.*' resolve cleanly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st
import yaml

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

_CONFIG_PATH = _REPO_ROOT / "config.yaml"


@st.cache_resource
def _load_config() -> dict:
    """Load config.yaml once.  Fail loudly if it is absent or malformed."""
    if not _CONFIG_PATH.is_file():
        st.error(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Clone the full repository and run from its root."
        )
        st.stop()
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _cfg_get(cfg: dict, *keys, default=None):
    """Traverse nested config keys safely, returning default on missing."""
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# --------------------------------------------------------------------------- #
# Mode detection                                                               #
# --------------------------------------------------------------------------- #

_MODE = os.environ.get("DATA_MODE", "synthetic").strip().lower()
_REAL_MODE = _MODE == "real"

# --------------------------------------------------------------------------- #
# Index factory                                                                #
# --------------------------------------------------------------------------- #

@st.cache_resource
def _build_synthetic_index():
    """Build the FakeIndex once per process lifecycle.

    Cached separately from mutable Stock so Reset Stock never recreates
    the embeddings, just re-initialises the stock set.
    """
    from app.fake_index import build_fake_index
    return build_fake_index()


@st.cache_resource
def _build_real_index(alpha: float):
    """Build VectorIndex from precomputed embeddings at the given alpha.

    Loaded once; the alpha slider in real mode triggers a cache miss only
    when alpha changes (acceptable: rebuild is well under 1 second for 44k
    vectors).

    Raises st.error + st.stop() if any required file is missing, so the
    app degrades gracefully rather than crashing with a raw exception.
    """
    cfg = _load_config()
    img_path = Path(_cfg_get(cfg, "paths", "image_embeddings", default=""))
    txt_path = Path(_cfg_get(cfg, "paths", "text_embeddings", default=""))
    ids_path = Path(_cfg_get(cfg, "paths", "item_ids", default=""))

    missing = [p for p in [img_path, txt_path, ids_path] if not p.is_file()]
    if missing:
        st.error(
            "Real-data mode requires precomputed embeddings. "
            "Missing files:\n" + "\n".join(f"  • {p}" for p in missing) + "\n\n"
            "Obtain them from the team Drive folder and place them under "
            "data/processed/ as described in config.yaml."
        )
        st.stop()

    from src.index.faiss_index import VectorIndex
    try:
        return VectorIndex.from_embeddings(img_path, txt_path, ids_path, alpha)
    except NotImplementedError:
        st.error(
            "VectorIndex.from_embeddings is not yet implemented (Track B stub). "
            "Run in synthetic mode until Track B is complete:\n"
            "    streamlit run app/main.py"
        )
        st.stop()


# --------------------------------------------------------------------------- #
# Manifest loading (real mode)                                                 #
# --------------------------------------------------------------------------- #

@st.cache_resource
def _load_manifest():
    """Load the real manifest CSV once, validate its columns."""
    cfg = _load_config()
    mpath = Path(_cfg_get(cfg, "paths", "manifest", default=""))
    if not mpath.is_file():
        st.error(
            f"Manifest not found: {mpath}\n"
            "Generate it with: python scripts/encode_catalogue.py\n"
            "(or obtain the pre-built manifest from the team Drive folder)"
        )
        st.stop()
    from src.data.dataset import load_manifest
    return load_manifest(mpath)


# --------------------------------------------------------------------------- #
# Encoder (real text search)                                                   #
# --------------------------------------------------------------------------- #

@st.cache_resource
def _load_encoder():
    """Load FashionCLIP once for text queries.

    Fails gracefully when weights are missing so the rest of the app still
    works (text search section shows an error; item-to-item still runs).
    """
    cfg = _load_config()
    model_name = _cfg_get(cfg, "encoder", "model_name",
                           default="patrickjohncyh/fashion-clip")
    device = _cfg_get(cfg, "encoder", "device", default="cpu")
    batch_size = _cfg_get(cfg, "encoder", "batch_size", default=32)

    from src.encoders.clip_encoder import ClipEncoder
    enc = ClipEncoder(model_name=model_name, device=device, batch_size=batch_size)
    try:
        enc.load()
        return enc
    except NotImplementedError:
        return None           # Track B stub; caller shows an informative error
    except Exception as exc:
        st.warning(
            f"Could not load FashionCLIP ({exc}). "
            "Text search will be unavailable. "
            "Make sure you have run: pip install transformers torch "
            "and that model weights are cached locally."
        )
        return None


# --------------------------------------------------------------------------- #
# Image rendering helpers                                                       #
# --------------------------------------------------------------------------- #

def _thumbnail_bytes_fake(product) -> bytes:
    """Convert a FakeProduct's thumbnail ndarray to PNG bytes for st.image."""
    from PIL import Image
    img = Image.fromarray(product.thumbnail, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _thumbnail_bytes_real(image_path: str, raw_dir: Path) -> bytes | None:
    """Resolve a relative image path and return PNG bytes.

    Returns None if the file does not exist or cannot be opened, so the
    caller can fall back to a placeholder without crashing.
    """
    from PIL import Image
    full = raw_dir / image_path
    if not full.is_file():
        return None
    try:
        img = Image.open(full).convert("RGB").resize((64, 64))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _placeholder_bytes(colour: tuple[int, int, int] = (160, 160, 160)) -> bytes:
    """64×64 solid-colour placeholder for missing real images."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), colour)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Session state initialisation                                                  #
# --------------------------------------------------------------------------- #

def _init_state(index) -> None:
    """Initialise mutable session state that must survive Streamlit reruns.

    Stock is only created here when it does not already exist; recreating
    it on every rerun would undo the user's sold-item changes.
    """
    if "stock" not in st.session_state:
        from src.index.stock import Stock
        if _REAL_MODE:
            all_ids = _get_all_ids_real(index)
        else:
            all_ids = index.all_item_ids()
        st.session_state.stock = Stock(all_ids)

    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    if "query_mode" not in st.session_state:
        st.session_state.query_mode = "item"   # "item" | "text"

    if "text_query" not in st.session_state:
        st.session_state.text_query = ""


def _get_all_ids_real(index) -> list[str]:
    """Extract all item IDs from the real VectorIndex."""
    return index._item_ids  # VectorIndex stores _item_ids as a list


# --------------------------------------------------------------------------- #
# Product card rendering                                                        #
# --------------------------------------------------------------------------- #

def _render_product_card_fake(product, col, show_select: bool = True) -> None:
    """Render a FakeProduct in a Streamlit column with a Select button."""
    img_bytes = _thumbnail_bytes_fake(product)
    col.image(img_bytes, width=80)
    col.caption(f"**{product.title}**")
    col.caption(f"`{product.item_id}`")
    if show_select:
        if col.button("Select", key=f"sel_{product.item_id}"):
            st.session_state.selected_id = product.item_id
            st.session_state.query_mode = "item"
            st.rerun()


def _render_result_card(
    item_id: str,
    score: float,
    title: str,
    img_bytes: bytes,
    col,
) -> None:
    """Render one recommendation card with image, title, score, Mark Sold."""
    col.image(img_bytes, width=80)
    col.caption(f"**{title}**")
    col.caption(f"`{item_id}`  score: `{score:.3f}`")
    if col.button("🛒 Mark sold", key=f"sold_{item_id}"):
        st.session_state.stock.mark_sold(item_id)
        st.rerun()


# --------------------------------------------------------------------------- #
# Main application                                                              #
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(
        page_title="LastPiece — Availability-Aware Recommender",
        page_icon="🏷️",
        layout="wide",
    )

    cfg = _load_config()
    k: int = int(_cfg_get(cfg, "retrieval", "k", default=5))
    n_candidates: int = int(_cfg_get(cfg, "retrieval", "n_candidates", default=50))
    max_candidates: int = int(_cfg_get(cfg, "retrieval", "max_candidates", default=500))
    default_lambda: float = float(_cfg_get(cfg, "rerank", "lambda_", default=0.7))
    alpha: float = float(_cfg_get(cfg, "fusion", "alpha", default=0.5))
    raw_dir = Path(_cfg_get(cfg, "paths", "raw_dir", default="data/raw"))

    # ------------------------------------------------------------------ #
    # Build/load index                                                      #
    # ------------------------------------------------------------------ #
    if _REAL_MODE:
        index = _build_real_index(alpha)
        manifest = _load_manifest()
        manifest_by_id = {row["item_id"]: row
                          for row in manifest.to_dict("records")}
        mode_label = "🟢 Real-data mode"
    else:
        index = _build_synthetic_index()
        manifest = None
        manifest_by_id = {}
        mode_label = "🟡 Synthetic mode (no real data required)"

    _init_state(index)
    stock = st.session_state.stock

    # ------------------------------------------------------------------ #
    # Header                                                                #
    # ------------------------------------------------------------------ #
    st.title("🏷️ LastPiece")
    st.caption(
        "Availability-aware multimodal recommender for single-unit inventory. "
        f"**{mode_label}**"
    )

    # ------------------------------------------------------------------ #
    # Sidebar controls                                                      #
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.header("Controls")

        lambda_ = st.slider(
            "λ (MMR relevance/diversity)",
            min_value=0.3,
            max_value=1.0,
            value=default_lambda,
            step=0.05,
            help=(
                "1.0 = pure relevance (MMR off, near-duplicates dominate). "
                "0.3 = maximum diversity. "
                "Drag left to watch near-duplicates break apart."
            ),
        )

        st.markdown("---")
        st.metric("Items in stock", stock.n_available())

        if st.button("🔄 Reset stock"):
            from src.index.stock import Stock
            if _REAL_MODE:
                all_ids = _get_all_ids_real(index)
            else:
                all_ids = index.all_item_ids()
            st.session_state.stock = Stock(all_ids)
            st.rerun()

        st.markdown("---")
        st.caption(
            f"k={k} | n_candidates={n_candidates} | "
            f"max_candidates={max_candidates}"
        )

    # ------------------------------------------------------------------ #
    # Query mode selector                                                   #
    # ------------------------------------------------------------------ #
    mode_col1, mode_col2 = st.columns([1, 1])
    with mode_col1:
        if st.button(
            "🖼️ Item-to-item",
            type="primary" if st.session_state.query_mode == "item" else "secondary",
        ):
            st.session_state.query_mode = "item"
            st.rerun()
    with mode_col2:
        if st.button(
            "🔤 Text search",
            type="primary" if st.session_state.query_mode == "text" else "secondary",
        ):
            st.session_state.query_mode = "text"
            st.rerun()

    # ------------------------------------------------------------------ #
    # Text search input (shown only in text mode)                          #
    # ------------------------------------------------------------------ #
    query_vec: np.ndarray | None = None
    exclude_ids: list[str] = []

    if st.session_state.query_mode == "text":
        text_input = st.text_input(
            "Enter a product description",
            value=st.session_state.text_query,
            placeholder="e.g. blue vintage denim jacket",
        )
        st.session_state.text_query = text_input

        if text_input.strip():
            if _REAL_MODE:
                encoder = _load_encoder()
                if encoder is None:
                    st.error(
                        "Text encoder not available. "
                        "ClipEncoder.load() is not yet implemented (Track B stub), "
                        "or model weights are missing. "
                        "Switch to item-to-item mode, or wait for Track B."
                    )
                else:
                    with st.spinner("Encoding query…"):
                        try:
                            vec = encoder.encode_query(
                                text=text_input.strip(), alpha=alpha
                            )
                            if vec.shape != (512,):
                                st.error(
                                    f"Encoder returned vector with shape {vec.shape}; "
                                    "expected (512,). Check ClipEncoder.encode_query()."
                                )
                            else:
                                query_vec = vec.astype(np.float32)
                        except Exception as exc:
                            st.error(f"Encoding failed: {exc}")
            else:
                # Synthetic mode: generate a plausible query vector by
                # hashing the text into a reproducible direction.  This is
                # not semantically meaningful but lets the UI exercise the
                # full pipeline without a real encoder.
                h = abs(hash(text_input.strip())) % (2**31)
                rng = np.random.default_rng(h)
                v = rng.standard_normal(512).astype(np.float32)
                query_vec = v / np.linalg.norm(v)
                st.caption("ℹ️ Synthetic mode: text query mapped to a random vector.")
        else:
            st.info("Enter a description above to search.")

    # ------------------------------------------------------------------ #
    # Item-to-item: product grid                                            #
    # ------------------------------------------------------------------ #
    if st.session_state.query_mode == "item":
        st.subheader("Select a product to find similar available pieces")

        if _REAL_MODE:
            # Show a small random sample from the manifest for selection.
            sample_ids = list(manifest_by_id.keys())[:60]
            _render_real_grid(
                sample_ids, manifest_by_id, raw_dir,
                st.session_state.selected_id
            )
        else:
            _render_fake_grid(index)

        selected_id = st.session_state.selected_id
        if selected_id is not None:
            try:
                query_vec = index.get_vector(selected_id)
                exclude_ids = [selected_id]
            except KeyError:
                st.error(f"Selected item {selected_id!r} not found in index.")
                st.session_state.selected_id = None

    # ------------------------------------------------------------------ #
    # Recommendation                                                        #
    # ------------------------------------------------------------------ #
    if query_vec is not None:
        st.divider()
        _render_recommendations(
            query_vec=query_vec,
            index=index,
            stock=stock,
            k=k,
            n_candidates=n_candidates,
            max_candidates=max_candidates,
            lambda_=lambda_,
            exclude_ids=exclude_ids,
            manifest_by_id=manifest_by_id,
            raw_dir=raw_dir,
            is_real=_REAL_MODE,
        )
    elif st.session_state.query_mode == "item" and st.session_state.selected_id is None:
        st.info("👆 Click **Select** on any product above to see recommendations.")


# --------------------------------------------------------------------------- #
# Grid renderers                                                                #
# --------------------------------------------------------------------------- #

def _render_fake_grid(index) -> None:
    """Render the synthetic product catalogue as a scrollable grid."""
    products = index.all_products()
    # Group by cluster (consecutive products share cluster label)
    # Show first 60 products to keep the grid manageable.
    display = products[:60]

    cols_per_row = 6
    for row_start in range(0, len(display), cols_per_row):
        row_products = display[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for product, col in zip(row_products, cols):
            _render_product_card_fake(product, col)


def _render_real_grid(
    item_ids: list[str],
    manifest_by_id: dict,
    raw_dir: Path,
    selected_id: str | None,
) -> None:
    """Render a subset of the real catalogue for query selection."""
    cols_per_row = 6
    for row_start in range(0, len(item_ids), cols_per_row):
        row_ids = item_ids[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for item_id, col in zip(row_ids, cols):
            row = manifest_by_id.get(item_id, {})
            img_path = row.get("image_path", "")
            title = row.get("description", item_id)

            img_bytes = _thumbnail_bytes_real(img_path, raw_dir)
            if img_bytes is None:
                img_bytes = _placeholder_bytes()

            col.image(img_bytes, width=80)
            col.caption(f"**{title[:30]}**")
            col.caption(f"`{item_id}`")
            is_selected = item_id == selected_id
            label = "✓ Selected" if is_selected else "Select"
            if col.button(label, key=f"sel_{item_id}"):
                st.session_state.selected_id = item_id
                st.session_state.query_mode = "item"
                st.rerun()


# --------------------------------------------------------------------------- #
# Recommendation renderer                                                       #
# --------------------------------------------------------------------------- #

def _render_recommendations(
    query_vec: np.ndarray,
    index,
    stock,
    k: int,
    n_candidates: int,
    max_candidates: int,
    lambda_: float,
    exclude_ids: list[str],
    manifest_by_id: dict,
    raw_dir: Path,
    is_real: bool,
) -> None:
    """Run the full Stage C pipeline and display results with diagnostics."""
    from src.rerank.availability import rerank_with_availability

    result = rerank_with_availability(
        query_vec=query_vec,
        retrieve_fn=index.search,
        stock=stock,
        k=k,
        n_candidates=n_candidates,
        max_candidates=max_candidates,
        lambda_=lambda_,
        exclude_ids=exclude_ids if exclude_ids else None,
    )

    # ------------------------------------------------------------------ #
    # Diagnostics panel                                                     #
    # ------------------------------------------------------------------ #
    st.subheader("Diagnostics")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Retrieved", result.n_retrieved)
    d2.metric("Dropped (sold)", result.n_dropped_sold)
    d3.metric("Widening rounds", result.n_widening_rounds)
    d4.metric("Final depth", result.final_candidate_depth)
    d5.metric("In stock", stock.n_available())

    if result.short_result:
        st.warning(
            f"⚠️ Only {len(result)} of {k} requested results available "
            "(catalogue nearly exhausted)."
        )

    # ------------------------------------------------------------------ #
    # Result cards                                                          #
    # ------------------------------------------------------------------ #
    st.subheader(f"Recommendations ({len(result)} of {k})")

    if not result.item_ids:
        st.error(
            "No available products found. "
            "Try clicking **Reset stock** in the sidebar."
        )
        return

    cols_per_row = min(k, 5)
    cols = st.columns(cols_per_row)
    for i, (item_id, score) in enumerate(zip(result.item_ids, result.scores)):
        col = cols[i % cols_per_row]

        if is_real:
            row = manifest_by_id.get(item_id, {})
            title = row.get("description", item_id)
            img_path = row.get("image_path", "")
            img_bytes = _thumbnail_bytes_real(img_path, raw_dir)
            if img_bytes is None:
                img_bytes = _placeholder_bytes()
        else:
            try:
                product = index.get_product(item_id)
                title = product.title
                img_bytes = _thumbnail_bytes_fake(product)
            except KeyError:
                title = item_id
                img_bytes = _placeholder_bytes()

        _render_result_card(item_id, score, title, img_bytes, col)


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
else:
    # Streamlit imports the module; the app body must run at module level
    # (not inside a function) for st.cache_resource and st.session_state to
    # work correctly.  Calling main() here makes both `streamlit run` and
    # direct import work.
    main()
