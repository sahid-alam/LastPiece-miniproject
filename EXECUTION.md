# EXECUTION.md: Real-Data Integration, macOS OpenMP Crash Fix & Demo Polish

> **Project**: LastPiece — Availability-Aware Multimodal Recommender for Single-Unit Resale Inventory  
> **Track**: Track D (Demo + Slides + Interactive Evaluation)  
> **Owner**: Lucky Dhawan  
> **Milestone**: Review 1 (13 Oct 2026, 50% Target)  
> **Related Issues & PRs**: Closes [#11](https://github.com/sahid-alam/LastPiece-miniproject/issues/11), links [#10](https://github.com/sahid-alam/LastPiece-miniproject/issues/10), [#12](https://github.com/sahid-alam/LastPiece-miniproject/issues/12), PR [#26](https://github.com/sahid-alam/LastPiece-miniproject/pull/26), PR [#28](https://github.com/sahid-alam/LastPiece-miniproject/pull/28)

---

## 1. Executive Summary & Problem Analysis

### 1.1 Objective
Transition the Streamlit demo from synthetic mode to real-data mode (`DATA_MODE=real`) indexing the full 44,419 items from the Fashion Product Images dataset. The demo demonstrates item-to-item retrieval, multimodal text search, MMR diversity re-ranking ($\lambda \in [0.3, 1.0]$), and Stage C availability-aware substitution when single-unit items sell out.

### 1.2 The macOS OpenMP Segfault Bug (Exit Code 139)
* **Symptom**: On macOS, running a search query over the 44k index crashes the entire Python runtime with a segmentation fault (`exit 139`).
* **Root Cause**: `torch` (used by `ClipEncoder`) and `faiss` (used by `VectorIndex`) each bundle their own dynamic copy of the OpenMP runtime (`libomp.dylib`). When both libraries are loaded into the same process, the two OpenMP runtimes clash during multi-threaded index execution.
* **Two-Layer Resolution**:
  1. **Core Index Guard**: Inside `src/index/faiss_index.py`, encapsulate FAISS loading in a helper `_import_faiss()`. This helper pre-loads `torch` before `faiss`, sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`, and sets FAISS to single-threaded mode via `faiss.omp_set_num_threads(1)` (~3 ms query latency over 44k items, zero thread collision).
  2. **Top-Level App Guard**: In `app/main.py`, right after detecting `_REAL_MODE`, set `KMP_DUPLICATE_LIB_OK=TRUE` and import `torch` before any index factory is called.

### 1.3 Safety Net & Fallback Strategy (#10)
If dataset or embedding files are delayed past the review deadline, the demo gracefully falls back to synthetic mode (`streamlit run app/main.py` without `DATA_MODE=real`). Synthetic mode requires zero external files, no GPU, and exercises the exact same Streamlit UI and Stage C re-ranking logic.

---

## 2. Phase-by-Phase Execution Plan

```
┌────────────────────────────────────────────────────────┐
│ Phase 0: Pre-requisites & Branch Management            │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Phase 1: Dataset & Embedding Placement (Drive Setup)   │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Phase 2: macOS OpenMP Collision Fix                    │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Phase 3: VectorIndex Implementation & Embedding Fusion  │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Phase 4: Streamlit Product Grid Upgrade (app/main.py)  │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Phase 5: Verification & Quality Assurance Checklist    │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Phase 6: Slide Asset Collection (#12)                  │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Phase 7: Pull Request & Delivery Workflow              │
└────────────────────────────────────────────────────────┘
```

---

### Phase 0: Pre-requisites & Branch Management

1. **Verify Base PR #28**:
   Ensure PR #28 (`DATA_MODE=real` wiring and synthetic demo fixes) is committed and tests pass.
2. **Check Track B Status (PR #26)**:
   Verify whether PR #26 (`ClipEncoder`, `VectorIndex`, `encode_catalogue.py`) is merged or implemented as a backup so real mode does not fail with `NotImplementedError`.
3. **Branch Creation**:
   Create and check out the delivery branch:
   ```bash
   git checkout -b d2-real-data
   ```

---

### Phase 1: Dataset & Embeddings Placement

Download the assets from Google Drive (`LastPiece — Shared Files`) and place them in the local repository structure:

| Drive Source | Local Destination | Content & Format |
| :--- | :--- | :--- |
| `1-dataset/raw.zip` | Extract into `data/raw/` | Produces `data/raw/styles.csv` and `data/raw/images/<id>.jpg` |
| `1-dataset/manifest.csv` | `data/processed/manifest.csv` | Processed catalogue manifest (44,419 rows) |
| `2-embeddings/emb_image.npy` | `data/processed/emb_image.npy` | FashionCLIP image vectors `(44419, 512)` float32 |
| `2-embeddings/emb_text.npy` | `data/processed/emb_text.npy` | FashionCLIP text vectors `(44419, 512)` float32 |
| `2-embeddings/item_ids.json` | `data/processed/item_ids.json` | 44,419 item IDs aligned with embedding rows |

**Security Check**: Confirm `.gitignore` ignores `data/raw/images/`, `data/processed/*.npy`, and `data/processed/*.json`. Never commit raw data or binary embeddings.

**Validation Command**:
```python
import json, numpy as np
img = np.load("data/processed/emb_image.npy")
txt = np.load("data/processed/emb_text.npy")
with open("data/processed/item_ids.json") as f:
    ids = json.load(f)
assert img.shape == txt.shape == (44419, 512), f"Bad shape: {img.shape}"
assert len(ids) == 44419, f"Bad ID count: {len(ids)}"
print("Dataset and embeddings verified successfully.")
```

---

### Phase 2: macOS OpenMP Collision Fix

1. **In `src/index/faiss_index.py`**:
   Add the private importer `_import_faiss()`:
   ```python
   def _import_faiss():
       """Safely import faiss avoiding the OpenMP runtime clash on macOS (#11).

       torch and faiss each bundle their own copy of OpenMP. On macOS, importing
       faiss before torch causes a segfault (exit 139) on search. This helper:
         1. Pre-loads torch if installed.
         2. Sets KMP_DUPLICATE_LIB_OK=TRUE.
         3. Sets faiss to single-threaded mode (takes ~3 ms for 44k items).
       """
       import os
       os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
       try:
           import torch  # noqa: F401
       except ImportError:
           pass
       import faiss
       if hasattr(faiss, "omp_set_num_threads"):
           faiss.omp_set_num_threads(1)
       return faiss
   ```

2. **In `app/main.py`**:
   Immediately after defining `_REAL_MODE`:
   ```python
   _MODE = os.environ.get("DATA_MODE", "synthetic").strip().lower()
   _REAL_MODE = _MODE == "real"

   if _REAL_MODE:
       os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must be set before torch loads
       try:
           import torch  # noqa: F401 load before faiss, or text search crashes on macOS (#11)
       except ImportError:
           pass
   ```

---

### Phase 3: VectorIndex Implementation & Fusion

Implement `VectorIndex` in `src/index/faiss_index.py`:

```python
class VectorIndex:
    """Wraps a FAISS flat inner-product index with string item IDs."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._index = None
        self._item_ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._id_to_pos: dict[str, int] = {}

    def build(self, vectors: np.ndarray, item_ids: Sequence[str]) -> None:
        """Build from fused vectors, shape (n, dim), aligned with item_ids."""
        faiss = _import_faiss()
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.shape[0] == len(item_ids)
        assert vectors.shape[1] == self.dim
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._item_ids = list(item_ids)
        self._vectors = vectors
        self._id_to_pos = {iid: idx for idx, iid in enumerate(self._item_ids)}

    def search(
        self,
        query_vec: np.ndarray,
        depth: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return the `depth` nearest items."""
        if self._index is None or self._vectors is None:
            raise RuntimeError("VectorIndex must be built before search.")
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, idx = self._index.search(q, min(depth, self._index.ntotal))
        idx, scores = idx[0], scores[0]
        keep = idx >= 0
        idx, scores = idx[keep], scores[keep]
        return ([self._item_ids[i] for i in idx], self._vectors[idx], scores)

    def get_vector(self, item_id: str) -> np.ndarray:
        """Fetch one item's stored vector, shape (dim,)."""
        if self._vectors is None:
            raise RuntimeError("VectorIndex must be built before fetching vectors.")
        pos = self._id_to_pos.get(item_id)
        if pos is None:
            raise KeyError(f"Item {item_id!r} not found in index.")
        return self._vectors[pos]

    @classmethod
    def from_embeddings(
        cls,
        image_emb_path: str | Path,
        text_emb_path: str | Path,
        ids_path: str | Path,
        alpha: float,
    ) -> "VectorIndex":
        """Load cached embeddings, fuse at alpha, build."""
        import json
        from src.encoders.fusion import fuse_batch

        img = np.load(image_emb_path)
        txt = np.load(text_emb_path)
        with open(ids_path, "r", encoding="utf-8") as fh:
            ids = json.load(fh)

        idx = cls(dim=img.shape[1])
        fused = fuse_batch(img, txt, alpha=alpha)
        idx.build(fused, ids)
        return idx

    def __len__(self) -> int:
        return len(self._item_ids)
```

---

### Phase 4: Streamlit Product Grid Upgrade (`app/main.py`)

Replace the static 60-item slice with a category-filtered, deterministic 24-item card grid:

```python
        if _REAL_MODE:
            # Category selection dropdown & deterministic shuffle for 24 items (#11)
            categories = sorted([c for c in manifest["category"].dropna().unique() if str(c).strip()])
            col_cat, col_shuf = st.columns([3, 1])
            with col_cat:
                selected_cat = st.selectbox(
                    "Filter by category",
                    options=categories,
                    key="real_category_select",
                )
            with col_shuf:
                st.write("")
                st.write("")
                if st.button("🔀 Shuffle", key="real_shuffle_btn"):
                    st.session_state.real_grid_seed = int(np.random.randint(0, 1_000_000))
                    st.rerun()

            cfg = _load_config()
            default_seed = _cfg_get(cfg, "project", "random_seed", default=42)
            if "real_grid_seed" not in st.session_state:
                st.session_state.real_grid_seed = default_seed

            cat_df = manifest[manifest["category"] == selected_cat]
            if cat_df.empty:
                cat_df = manifest

            # Sample 24 items deterministically using session seed
            n_sample = min(24, len(cat_df))
            sample_df = cat_df.sample(n=n_sample, random_state=st.session_state.real_grid_seed)
            sample_ids = sample_df["item_id"].tolist()

            _render_real_grid(
                sample_ids, manifest_by_id, raw_dir,
                st.session_state.selected_id
            )
        else:
            _render_fake_grid(index)
```

---

### Phase 5: Verification & Quality Assurance Checklist

#### 1. Execution Commands
* **Windows (PowerShell)**:
  ```powershell
  $env:DATA_MODE="real"; streamlit run app/main.py
  ```
* **macOS / Linux**:
  ```bash
  DATA_MODE=real streamlit run app/main.py
  ```

#### 2. Acceptance Checklist ("Done" Criteria)
- [x] **Automated Tests**: Run `pytest` and verify all tests pass without errors (114+ passed).
- [x] **Real Catalog Display**: Real product images, titles, and IDs render in the 24-item card grid.
- [x] **Item Query & Self-Exclusion**: Clicking **Select** on any product returns 5 recommendations; the clicked item is never in the recommendation set.
- [x] **Stage C Availability Substitution**: Clicking **Mark Sold** immediately removes the item, replaces it with a substitute, and increments `n_dropped_sold` in the Diagnostics panel.
- [x] **MMR Diversity Control**:
  - At $\lambda = 1.0$, returns near-identical visual duplicates.
  - Dragging to $\lambda = 0.4$ spreads out results across colors and cuts.
- [x] **Text Search**: Handled safely in UI with clear status/guidance.
- [x] **Stock Reset**: Clicking **🔄 Reset stock** restores all items back to available inventory.

---

### Phase 6: Slide Asset Collection (#12)

Capture 5 screenshots for the review presentation deck:
1. **Item-to-Item Grid**: 24 items with category selector and an item selected.
2. **Availability Substitution (Before & After)**: Side-by-side shots showing item sold, replacement appearing, and `n_dropped_sold` increasing.
3. **MMR Diversity Comparison**:
   - $\lambda = 1.0$ (pure relevance, near-duplicates).
   - $\lambda = 0.4$ (diverse results).
4. **Multimodal Text Search**: Query `"blue denim jacket"` showing semantic retrieval.
5. **Diagnostics Panel**: Close-up displaying live pipeline telemetry (`n_retrieved`, `n_dropped_sold`, `n_widening_rounds`, `n_available`).

---

### Phase 7: Pull Request & Delivery Workflow

1. **Commit Changes**:
   ```bash
   git add app/main.py src/index/faiss_index.py EXECUTION.md
   git commit -m "feat(demo): real-data mode, macOS OpenMP segfault fix, and seeded category grid"
   ```
2. **Push & Create PR**:
   ```bash
   git push -u origin d2-real-data
   ```
3. **PR Description**:
   ```markdown
   Closes #11

   ### Summary
   - Switches Streamlit demo to real-data mode (`DATA_MODE=real`) over 44,419 catalogue items.
   - Fixes macOS OpenMP collision segfault (exit 139) via `_import_faiss()` and early torch pre-loading.
   - Upgrades product grid with category dropdown, deterministic config seed (`project.random_seed: 42`), and 24-item shuffle button.
   - Implements VectorIndex with flat inner-product search and batch fusion.
   - Verified item-to-item, Mark Sold substitution, diversity slider, and text search.
   ```

---

## 3. Copy-Paste AI Execution Prompts

### Master All-in-One Execution Prompt
Use this prompt to execute or audit the complete migration in one shot:

```text
You are an expert AI software engineer working on Track D of the LastPiece mini-project.
Your task is to implement Issue #11 ("Make demo run on real data") on branch d2-real-data.

Requirements:
1. Fix macOS OpenMP crash (Exit 139):
   - In src/index/faiss_index.py, implement _import_faiss() to load torch first if installed, set KMP_DUPLICATE_LIB_OK=TRUE, and run faiss.omp_set_num_threads(1).
   - In app/main.py, right after _REAL_MODE is evaluated, set os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE") and import torch inside a try/except.
2. Implement VectorIndex in src/index/faiss_index.py:
   - build(vectors, item_ids): index with IndexFlatIP, store _vectors and _id_to_pos mapping.
   - search(query_vec, depth): search index, handle short results, return (ids, vectors, scores).
   - get_vector(item_id): retrieve vector by ID via _id_to_pos.
   - from_embeddings(img_path, txt_path, ids_path, alpha): load arrays, call fuse_batch(), build index.
3. Enhance product grid in app/main.py:
   - Add category dropdown filtering manifest['category'].
   - Add a Shuffle button sampling 24 items.
   - Initialize session seed from project.random_seed (42) in config.yaml so the first view is deterministic.
4. Verify all tests pass with pytest.
Deliver clean code conforming to the project architecture.
```

### Prompt for Product Grid Refinement Only
```text
In app/main.py of the LastPiece repository, upgrade the real-data product grid in main() under st.session_state.query_mode == "item":
1. Extract unique sorted categories from manifest["category"].
2. Render a category selectbox side-by-side with a "🔀 Shuffle" button.
3. Use project.random_seed (default 42) from config.yaml as st.session_state.real_grid_seed so the first view is deterministic.
4. When "Shuffle" is clicked, randomize st.session_state.real_grid_seed and rerun.
5. Sample 24 items from the chosen category using cat_df.sample(n=min(24, len(cat_df)), random_state=st.session_state.real_grid_seed) and pass to _render_real_grid.
Ensure synthetic mode grid remains untouched.
```
