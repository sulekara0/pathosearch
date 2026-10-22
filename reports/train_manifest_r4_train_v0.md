# Training manifest: r4_train_v0

- command: `uv run python scripts/build_train_manifest.py --golden eval_data/golden_v0 --out eval_data/r4_train_v0`
- git: `12f747a` · seed 42 · tau 0.99 · val target 2000

## Exclusion

| step | removed | remaining |
|---|---|---|
| candidates (on disk, train-only) | | 64369 |
| no_embedding | 26285 | 38084 |
| golden_ids | 0 | 38084 |
| golden_nearest | 7 | 38077 |
| near_dup_golden | 35 | 38042 |
| source_group | 10 | 38032 |
| caption_equal | 164 | 37868 |
| no_caption | 0 | 37868 |

## Validation and scales

- val: 1929 images from 713 whole source groups; 71 dropped for a near copy in train
- train: 35868 images; scales: 10k=10000, 30k=30000, full=35868
