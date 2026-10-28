# Eval: golden_v0 / dev

- command: `uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split dev --systems clip,quilt,lora-10k-s0,lora-30k-s0 --k 10 --adapter lora-10k-s0=artifacts/r4/runs/lora-10k-s0/adapter --adapter lora-30k-s0=artifacts/r4/runs/lora-30k-s0/adapter --compare lora-30k-s0:clip --compare lora-10k-s0:clip --compare lora-30k-s0:lora-10k-s0 --compare quilt:clip --compare lora-30k-s0:quilt`
- git: `29234fb` · k=10 · queries=10 · unsure pairs=23
- config: {"alpha": 0.75, "beta": 0.25, "rerank_pool_factor": 5, "domain_gate": "off", "thresholds": {"lenient": 1, "strict": 2}, "bootstrap": {"n": 1000, "seed": 42, "ci": 0.95, "paired": true}, "adapters": {"lora-10k-s0": {"dir": "artifacts/r4/runs/lora-10k-s0/adapter", "base_model": "openai/clip-vit-base-patch32", "epoch": 7, "config_hash": "a5cadd39d30863f3", "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "targets": ["q_proj", "v_proj"]}, "train_git_sha": null, "train_dirty": null, "best_val_loss": null, "train_status": null}, "lora-30k-s0": {"dir": "artifacts/r4/runs/lora-30k-s0/adapter", "base_model": "openai/clip-vit-base-patch32", "epoch": 6, "config_hash": "80f8cdb07a536818", "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "targets": ["q_proj", "v_proj"]}, "train_git_sha": null, "train_dirty": null, "best_val_loss": null, "train_status": null}}}

## Pool coverage (judged@10)

Share of each top-k list judged by a human (graded or unsure); unjudged images count as non-relevant.

| system | judged@k |
|---|---|
| clip | 1.000 |
| quilt | 1.000 |
| lora-10k-s0 | 1.000 |
| lora-30k-s0 | 1.000 |

## Metrics — lenient (grade >= 1, primary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.212 [0.113, 0.312] | 0.875 [0.625, 1.000] | 0.217 [0.136, 0.303] | 0.362 [0.183, 0.583] | 0.191 [0.097, 0.302] |
| quilt | 0.613 [0.475, 0.713] | 1.000 [1.000, 1.000] | 0.673 [0.565, 0.788] | 0.708 [0.500, 0.917] | 0.640 [0.490, 0.769] |
| lora-10k-s0 | 0.200 [0.087, 0.338] | 0.750 [0.500, 1.000] | 0.199 [0.076, 0.339] | 0.399 [0.149, 0.681] | 0.188 [0.069, 0.347] |
| lora-30k-s0 | 0.188 [0.100, 0.300] | 0.875 [0.625, 1.000] | 0.201 [0.096, 0.319] | 0.328 [0.162, 0.552] | 0.194 [0.086, 0.332] |

8 sorgu üzerinden, 2 sorgu dışlandı (q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | -0.025 [-0.138, +0.087] | +0.000 [-0.375, +0.375] | -0.017 [-0.159, +0.137] | -0.034 [-0.118, +0.048] | +0.003 [-0.096, +0.101] |
| lora-10k-s0 − clip | -0.013 [-0.125, +0.087] | -0.125 [-0.500, +0.250] | -0.018 [-0.151, +0.128] | +0.037 [-0.117, +0.201] | -0.003 [-0.085, +0.077] |
| lora-30k-s0 − lora-10k-s0 | -0.013 [-0.075, +0.050] | +0.125 [+0.000, +0.375] | +0.002 [-0.063, +0.059] | -0.071 [-0.226, +0.068] | +0.006 [-0.053, +0.071] |
| quilt − clip | +0.400 [+0.287, +0.500]* | +0.125 [+0.000, +0.375] | +0.456 [+0.308, +0.626]* | +0.346 [+0.142, +0.600]* | +0.450 [+0.302, +0.598]* |
| lora-30k-s0 − quilt | -0.425 [-0.525, -0.325]* | -0.125 [-0.375, +0.000] | -0.473 [-0.550, -0.379]* | -0.380 [-0.609, -0.177]* | -0.447 [-0.561, -0.338]* |

## Metrics — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.050 [0.017, 0.083] | 0.500 [0.167, 0.833] | 0.143 [0.018, 0.292] | 0.074 [0.017, 0.138] | 0.071 [0.011, 0.139] |
| quilt | 0.217 [0.083, 0.383] | 0.833 [0.500, 1.000] | 0.653 [0.361, 0.903] | 0.639 [0.306, 0.917] | 0.572 [0.313, 0.792] |
| lora-10k-s0 | 0.033 [0.000, 0.067] | 0.333 [0.000, 0.667] | 0.102 [0.000, 0.269] | 0.076 [0.000, 0.188] | 0.052 [0.000, 0.123] |
| lora-30k-s0 | 0.067 [0.033, 0.100] | 0.667 [0.333, 1.000] | 0.310 [0.060, 0.602] | 0.162 [0.037, 0.310] | 0.134 [0.053, 0.215] |

6 sorgu üzerinden, 4 sorgu dışlandı (q004, q008, q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | +0.017 [-0.033, +0.067] | +0.167 [-0.333, +0.667] | +0.167 [-0.167, +0.583] | +0.088 [-0.031, +0.227] | +0.063 [-0.066, +0.180] |
| lora-10k-s0 − clip | -0.017 [-0.067, +0.033] | -0.167 [-0.667, +0.333] | -0.042 [-0.292, +0.208] | +0.003 [-0.114, +0.117] | -0.019 [-0.118, +0.081] |
| lora-30k-s0 − lora-10k-s0 | +0.033 [+0.000, +0.067] | +0.333 [+0.000, +0.667] | +0.208 [+0.000, +0.542] | +0.086 [+0.014, +0.167]* | +0.082 [-0.002, +0.184] |
| quilt − clip | +0.167 [+0.050, +0.317]* | +0.333 [+0.000, +0.667] | +0.509 [+0.176, +0.843]* | +0.565 [+0.248, +0.860]* | +0.501 [+0.260, +0.735]* |
| lora-30k-s0 − quilt | -0.150 [-0.300, -0.033]* | -0.167 [-0.500, +0.000] | -0.343 [-0.518, -0.167]* | -0.477 [-0.778, -0.190]* | -0.438 [-0.594, -0.236]* |

## Metrics, caption-unseen pairs only — lenient (grade >= 1, primary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.212 [0.113, 0.312] | 0.875 [0.625, 1.000] | 0.217 [0.136, 0.303] | 0.362 [0.183, 0.583] | 0.191 [0.097, 0.302] |
| quilt | 0.613 [0.475, 0.713] | 1.000 [1.000, 1.000] | 0.673 [0.565, 0.788] | 0.708 [0.500, 0.917] | 0.640 [0.490, 0.769] |
| lora-10k-s0 | 0.200 [0.087, 0.338] | 0.750 [0.500, 1.000] | 0.199 [0.076, 0.339] | 0.399 [0.149, 0.681] | 0.188 [0.069, 0.347] |
| lora-30k-s0 | 0.188 [0.100, 0.300] | 0.875 [0.625, 1.000] | 0.201 [0.096, 0.319] | 0.328 [0.162, 0.552] | 0.194 [0.086, 0.332] |

8 sorgu üzerinden, 2 sorgu dışlandı (q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | -0.025 [-0.138, +0.087] | +0.000 [-0.375, +0.375] | -0.017 [-0.159, +0.137] | -0.034 [-0.118, +0.048] | +0.003 [-0.096, +0.101] |
| lora-10k-s0 − clip | -0.013 [-0.125, +0.087] | -0.125 [-0.500, +0.250] | -0.018 [-0.151, +0.128] | +0.037 [-0.117, +0.201] | -0.003 [-0.085, +0.077] |
| lora-30k-s0 − lora-10k-s0 | -0.013 [-0.075, +0.050] | +0.125 [+0.000, +0.375] | +0.002 [-0.063, +0.059] | -0.071 [-0.226, +0.068] | +0.006 [-0.053, +0.071] |
| quilt − clip | +0.400 [+0.287, +0.500]* | +0.125 [+0.000, +0.375] | +0.456 [+0.308, +0.626]* | +0.346 [+0.142, +0.600]* | +0.450 [+0.302, +0.598]* |
| lora-30k-s0 − quilt | -0.425 [-0.525, -0.325]* | -0.125 [-0.375, +0.000] | -0.473 [-0.550, -0.379]* | -0.380 [-0.609, -0.177]* | -0.447 [-0.561, -0.338]* |

## Metrics, caption-unseen pairs only — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.050 [0.017, 0.083] | 0.500 [0.167, 0.833] | 0.143 [0.018, 0.292] | 0.074 [0.017, 0.138] | 0.071 [0.011, 0.139] |
| quilt | 0.217 [0.083, 0.383] | 0.833 [0.500, 1.000] | 0.653 [0.361, 0.903] | 0.639 [0.306, 0.917] | 0.572 [0.313, 0.792] |
| lora-10k-s0 | 0.033 [0.000, 0.067] | 0.333 [0.000, 0.667] | 0.102 [0.000, 0.269] | 0.076 [0.000, 0.188] | 0.052 [0.000, 0.123] |
| lora-30k-s0 | 0.067 [0.033, 0.100] | 0.667 [0.333, 1.000] | 0.310 [0.060, 0.602] | 0.162 [0.037, 0.310] | 0.134 [0.053, 0.215] |

6 sorgu üzerinden, 4 sorgu dışlandı (q004, q008, q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | +0.017 [-0.033, +0.067] | +0.167 [-0.333, +0.667] | +0.167 [-0.167, +0.583] | +0.088 [-0.031, +0.227] | +0.063 [-0.066, +0.180] |
| lora-10k-s0 − clip | -0.017 [-0.067, +0.033] | -0.167 [-0.667, +0.333] | -0.042 [-0.292, +0.208] | +0.003 [-0.114, +0.117] | -0.019 [-0.118, +0.081] |
| lora-30k-s0 − lora-10k-s0 | +0.033 [+0.000, +0.067] | +0.333 [+0.000, +0.667] | +0.208 [+0.000, +0.542] | +0.086 [+0.014, +0.167]* | +0.082 [-0.002, +0.184] |
| quilt − clip | +0.167 [+0.050, +0.317]* | +0.333 [+0.000, +0.667] | +0.509 [+0.176, +0.843]* | +0.565 [+0.248, +0.860]* | +0.501 [+0.260, +0.735]* |
| lora-30k-s0 − quilt | -0.150 [-0.300, -0.033]* | -0.167 [-0.500, +0.000] | -0.343 [-0.518, -0.167]* | -0.477 [-0.778, -0.190]* | -0.438 [-0.594, -0.236]* |

Caption-unseen: 0 of 243 judged pairs removed (0.0%); queries with a relevant image remaining: {'lenient': 8, 'strict': 6} of 10.

## Circularity check

Spearman(grade, caption overlap) = 0.212 over 243 pairs.

## Caveats

- QuiltNet held-out status of CSV val split: unverified (STATUS §7.8.3).
- Judgments are pooled: unjudged images count as non-relevant (see the random control sample of the pool for how often this is wrong).
- Annotators are not pathologists; grades are noisy (see the agreement report).
- The golden set passed the 0.99 near-duplicate exclusion; it is not clean at the 0.95 criterion (ADR-0003).
