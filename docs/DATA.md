# Dataset

## Fashion Product Images (small)

About 44k single fashion products on plain backgrounds, each with a short
product title and category labels (`styles.csv`). Chosen over
DeepFashion-MultiModal for the reasons in `docs/DECISIONS.md` D11.

Fill this table in when you download it. The report needs the citation and
the reviewers may ask where the data came from.

| Field | Value |
|---|---|
| Source URL | https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small |
| Accessed | 2026-10-11 |
| Licence | MIT |
| Rows in styles.csv | 44446 |
| Rows skipped (malformed) | 22 |
| Rows dropped (no image) | 5 |
| Final manifest rows | 44419 |
| Size on disk | 632 MB |

## Getting it

Download the zip from the Kaggle page in a browser (free account needed),
unzip into `data/raw/` so you have:

```
data/raw/styles.csv
data/raw/images/<id>.jpg
```

Never commit it: `data/` is gitignored. Every teammate downloads their own
copy. Then:

```bash
python -c "from src.data.dataset import build_manifest; build_manifest('data/raw', 'data/processed/manifest.csv')"
```

(Track A may wrap that in a script once it works.)

## Manifest schema

The contract between Track A and everyone else. Frozen; changing a column
means telling the whole group. Source of truth is `MANIFEST_COLUMNS` in
`src/data/dataset.py`.

| Column | From styles.csv | Notes |
|---|---|---|
| item_id | `id` | string, unique |
| image_path | — | `images/<id>.jpg`, relative to `data/raw` |
| description | `productDisplayName` | may be empty; fusion falls back to image-only |
| category | `articleType` | used by the relevance proxy (D12) |
| colour | `baseColour` | secondary attribute, may be empty |
| usage | `usage` | secondary attribute, may be empty |

No `stock` column: every item is one unit, and availability lives in the
`Stock` set at run time (D13).

## Simulating single-unit inventory

The dataset is static, so `src/eval/simulation.py` fakes a session in which
users browse and items sell for good (D16). Seeded from `project.random_seed`
so every system under test replays the identical session.

## Known limitations

- Clean catalogue photography, not thrift photos taken on a bedroom floor.
- Product titles, not seller-written descriptions; the text side is weaker
  than it would be on a real resale catalogue.
- Low-resolution images in the small release.

State these as threats to validity in the report rather than letting a
reviewer find them.
