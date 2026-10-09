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


def build_manifest(raw_dir: str | Path, out_path: str | Path):
    """Parse styles.csv + images/ into the manifest CSV.

    TODO(owner): implement.
        - pandas.read_csv(raw_dir / "styles.csv", on_bad_lines="skip"):
          a few rows have stray commas in productDisplayName. Log how many
          were skipped; the report should state it.
        - drop rows whose image file does not exist, and log the count
        - item_id as str, empty strings (not NaN) for missing text fields
        - assert item_id is unique
        - write CSV with exactly MANIFEST_COLUMNS, return the DataFrame
    """
    raise NotImplementedError("build_manifest")


def load_manifest(path: str | Path):
    """Read the manifest CSV and validate its columns.

    TODO(owner): implement. dtype=str for every column, fillna(""). Raise a
    clear error naming any missing column; a cryptic KeyError three modules
    downstream wastes a teammate's afternoon.
    """
    raise NotImplementedError("load_manifest")


def relevant_items(
    manifest,
    query_id: str,
    secondary_attributes: list[str],
    min_secondary_matches: int,
) -> set[str]:
    """Ground-truth relevant items for one query (docs/DECISIONS.md D12).

    Relevant = same category AND at least min_secondary_matches of the
    secondary attributes equal (empty values never match). The query item
    itself is excluded.

    A stated proxy, not click data. It mildly favours content-based methods by
    construction; the report says so openly.

    TODO(owner): implement. Called once per query in a 1000-query session, so
    pre-group the manifest by category rather than scanning 44k rows each time.
    """
    raise NotImplementedError("relevant_items")
