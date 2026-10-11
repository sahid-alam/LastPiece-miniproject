"""Fashion Product Images loading, manifest and relevance ground truth.

STATUS: STUB. Owner: Track A. Implement the TODOs below.

Dataset: Fashion Product Images (small), about 44k single products on plain
backgrounds, each with a short product title and category labels. See
docs/DATA.md for where to get it and docs/DECISIONS.md D11 for why it replaced
DeepFashion-MultiModal.

The manifest is the contract every other module reads. Build it once, store
as CSV, and never let another module parse raw dataset files.

Manifest columns (frozen; changing one means telling the whole group):
    item_id      str   styles.csv `id`, as a string
    image_path   str   relative to data/raw, e.g. "images/15970.jpg"
    description  str   styles.csv `productDisplayName`; may be empty
    category     str   styles.csv `articleType`, e.g. "Tshirts"
    colour       str   styles.csv `baseColour`; may be empty
    usage        str   styles.csv `usage`, e.g. "Casual"; may be empty
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["MANIFEST_COLUMNS", "build_manifest", "load_manifest", "relevant_items"]

MANIFEST_COLUMNS = ["item_id", "image_path", "description", "category", "colour", "usage"]


def build_manifest(raw_dir, out_path):
    """Parse styles.csv + images/ into the manifest CSV."""
    import pandas as pd
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)

    # Rows with stray commas break the CSV; skip them but count them.
    skipped = []

    def _skip(bad_line):
        skipped.append(bad_line)
        return None  # None tells pandas to drop the line

    df = pd.read_csv(
        raw_dir / "styles.csv",
        dtype=str,
        engine="python",
        on_bad_lines=_skip,
    )

    rename = {
        "id": "item_id",
        "productDisplayName": "description",
        "articleType": "category",
        "baseColour": "colour",
        "usage": "usage",
    }
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise ValueError(f"styles.csv is missing columns: {missing}")
    df = df[list(rename)].rename(columns=rename)

    # Empty strings, not NaN, for missing text.
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].str.strip()

    df["image_path"] = "images/" + df["item_id"] + ".jpg"

    # Drop rows whose image file does not exist.
    has_image = df["image_path"].map(lambda p: (raw_dir / p).is_file())
    n_dropped = int((~has_image).sum())
    df = df[has_image].reset_index(drop=True)

    assert df["item_id"].is_unique, "item_id must be unique"

    df = df[MANIFEST_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Rows skipped (malformed): {len(skipped)}")
    print(f"Rows dropped (no image):  {n_dropped}")
    print(f"Final manifest rows:      {len(df)}")
    return df


def load_manifest(path):
    """Read the manifest CSV and validate its columns."""
    import pandas as pd
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    return df


_INDEX = {"manifest": None, "by_id": None, "by_category": None}


def _index(manifest):
    # Build the lookup once per manifest, not once per query.
    if _INDEX["manifest"] is not manifest:
        records = manifest.to_dict("records")
        by_category = {}
        for r in records:
            by_category.setdefault(r["category"], []).append(r)
        _INDEX["manifest"] = manifest
        _INDEX["by_id"] = {r["item_id"]: r for r in records}
        _INDEX["by_category"] = by_category
    return _INDEX["by_id"], _INDEX["by_category"]


def relevant_items(manifest, query_id, secondary_attributes, min_secondary_matches):
    """Ground-truth relevant items for one query (docs/DECISIONS.md D12).

    Relevant = same category AND at least min_secondary_matches of the
    secondary attributes equal (empty values never match). The query item
    itself is excluded.
    """
    by_id, by_category = _index(manifest)
    query = by_id[query_id]
    if query["category"] == "":
        return set()

    relevant = set()
    for row in by_category[query["category"]]:
        if row["item_id"] == query_id:
            continue
        matches = sum(
            1
            for attr in secondary_attributes
            if query[attr] != "" and row[attr] == query[attr]
        )
        if matches >= min_secondary_matches:
            relevant.add(row["item_id"])
    return relevant