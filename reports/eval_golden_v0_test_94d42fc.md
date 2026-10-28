# Eval: golden_v0 / test — **TEST SPLIT**

- command: `uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split test --i-know-this-is-test --systems clip,quilt,lora-10k-s0,lora-30k-s0 --k 10 --adapter lora-10k-s0=artifacts/r4/runs/lora-10k-s0/adapter --adapter lora-30k-s0=artifacts/r4/runs/lora-30k-s0/adapter --compare lora-30k-s0:clip --compare lora-10k-s0:clip --compare lora-30k-s0:lora-10k-s0 --compare quilt:clip --compare lora-30k-s0:quilt`
- git: `94d42fc` · k=10 · queries=10 · unsure pairs=18
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
| clip | 0.330 [0.200, 0.490] | 1.000 [1.000, 1.000] | 0.344 [0.247, 0.449] | 0.608 [0.399, 0.814] | 0.309 [0.196, 0.434] |
| quilt | 0.530 [0.330, 0.720] | 0.900 [0.700, 1.000] | 0.449 [0.320, 0.552] | 0.594 [0.355, 0.811] | 0.523 [0.335, 0.688] |
| lora-10k-s0 | 0.110 [0.060, 0.160] | 0.700 [0.400, 1.000] | 0.090 [0.043, 0.138] | 0.312 [0.142, 0.500] | 0.100 [0.054, 0.144] |
| lora-30k-s0 | 0.130 [0.060, 0.200] | 0.700 [0.400, 1.000] | 0.142 [0.059, 0.228] | 0.467 [0.233, 0.717] | 0.206 [0.094, 0.311] |

10 sorgu üzerinden, 0 sorgu dışlandı (-). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | -0.200 [-0.360, -0.070]* | -0.300 [-0.600, +0.000] | -0.202 [-0.309, -0.099]* | -0.141 [-0.360, +0.133] | -0.102 [-0.253, +0.045] |
| lora-10k-s0 − clip | -0.220 [-0.370, -0.090]* | -0.300 [-0.600, +0.000] | -0.254 [-0.386, -0.133]* | -0.296 [-0.501, -0.112]* | -0.209 [-0.337, -0.103]* |
| lora-30k-s0 − lora-10k-s0 | +0.020 [-0.030, +0.070] | +0.000 [-0.300, +0.300] | +0.051 [-0.011, +0.132] | +0.155 [-0.008, +0.327] | +0.107 [+0.012, +0.203]* |
| quilt − clip | +0.200 [+0.020, +0.380]* | -0.100 [-0.300, +0.000] | +0.105 [-0.097, +0.300] | -0.013 [-0.319, +0.273] | +0.214 [+0.009, +0.413]* |
| lora-30k-s0 − quilt | -0.400 [-0.590, -0.220]* | -0.200 [-0.500, +0.000] | -0.307 [-0.447, -0.165]* | -0.128 [-0.467, +0.228] | -0.316 [-0.524, -0.108]* |

## Metrics — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.171 [0.014, 0.386] | 0.429 [0.143, 0.857] | 0.149 [0.036, 0.293] | 0.333 [0.048, 0.667] | 0.204 [0.028, 0.407] |
| quilt | 0.286 [0.114, 0.486] | 0.857 [0.571, 1.000] | 0.597 [0.352, 0.817] | 0.488 [0.238, 0.774] | 0.503 [0.290, 0.711] |
| lora-10k-s0 | 0.043 [0.000, 0.100] | 0.286 [0.000, 0.571] | 0.055 [0.000, 0.129] | 0.119 [0.000, 0.286] | 0.058 [0.000, 0.119] |
| lora-30k-s0 | 0.071 [0.014, 0.157] | 0.429 [0.143, 0.857] | 0.243 [0.029, 0.529] | 0.262 [0.048, 0.571] | 0.189 [0.055, 0.367] |

7 sorgu üzerinden, 3 sorgu dışlandı (q009, q015, q017). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | -0.100 [-0.300, +0.043] | +0.000 [-0.571, +0.571] | +0.094 [-0.211, +0.445] | -0.071 [-0.405, +0.215] | -0.015 [-0.266, +0.237] |
| lora-10k-s0 − clip | -0.129 [-0.343, +0.000] | -0.143 [-0.429, +0.000] | -0.095 [-0.226, +0.000] | -0.214 [-0.500, +0.000] | -0.146 [-0.337, +0.000] |
| lora-30k-s0 − lora-10k-s0 | +0.029 [-0.029, +0.071] | +0.143 [-0.286, +0.714] | +0.188 [-0.062, +0.484] | +0.143 [-0.071, +0.357] | +0.131 [-0.028, +0.302] |
| quilt − clip | +0.114 [+0.057, +0.171]* | +0.429 [+0.143, +0.857]* | +0.448 [+0.148, +0.733]* | +0.155 [-0.203, +0.536] | +0.300 [+0.084, +0.553]* |
| lora-30k-s0 − quilt | -0.214 [-0.386, -0.057]* | -0.429 [-0.857, +0.143] | -0.354 [-0.783, +0.141] | -0.226 [-0.643, +0.214] | -0.315 [-0.649, +0.021] |

## Metrics, caption-unseen pairs only — lenient (grade >= 1, primary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.330 [0.200, 0.490] | 1.000 [1.000, 1.000] | 0.344 [0.247, 0.449] | 0.608 [0.399, 0.814] | 0.309 [0.196, 0.434] |
| quilt | 0.530 [0.330, 0.720] | 0.900 [0.700, 1.000] | 0.449 [0.320, 0.552] | 0.594 [0.355, 0.811] | 0.523 [0.335, 0.688] |
| lora-10k-s0 | 0.110 [0.060, 0.160] | 0.700 [0.400, 1.000] | 0.090 [0.043, 0.138] | 0.312 [0.142, 0.500] | 0.100 [0.054, 0.144] |
| lora-30k-s0 | 0.130 [0.060, 0.200] | 0.700 [0.400, 1.000] | 0.142 [0.059, 0.228] | 0.467 [0.233, 0.717] | 0.206 [0.094, 0.311] |

10 sorgu üzerinden, 0 sorgu dışlandı (-). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | -0.200 [-0.360, -0.070]* | -0.300 [-0.600, +0.000] | -0.202 [-0.309, -0.099]* | -0.141 [-0.360, +0.133] | -0.102 [-0.253, +0.045] |
| lora-10k-s0 − clip | -0.220 [-0.370, -0.090]* | -0.300 [-0.600, +0.000] | -0.254 [-0.386, -0.133]* | -0.296 [-0.501, -0.112]* | -0.209 [-0.337, -0.103]* |
| lora-30k-s0 − lora-10k-s0 | +0.020 [-0.030, +0.070] | +0.000 [-0.300, +0.300] | +0.051 [-0.011, +0.132] | +0.155 [-0.008, +0.327] | +0.107 [+0.012, +0.203]* |
| quilt − clip | +0.200 [+0.020, +0.380]* | -0.100 [-0.300, +0.000] | +0.105 [-0.097, +0.300] | -0.013 [-0.319, +0.273] | +0.214 [+0.009, +0.413]* |
| lora-30k-s0 − quilt | -0.400 [-0.590, -0.220]* | -0.200 [-0.500, +0.000] | -0.307 [-0.447, -0.165]* | -0.128 [-0.467, +0.228] | -0.316 [-0.524, -0.108]* |

## Metrics, caption-unseen pairs only — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.171 [0.014, 0.386] | 0.429 [0.143, 0.857] | 0.149 [0.036, 0.293] | 0.333 [0.048, 0.667] | 0.204 [0.028, 0.407] |
| quilt | 0.286 [0.114, 0.486] | 0.857 [0.571, 1.000] | 0.597 [0.352, 0.817] | 0.488 [0.238, 0.774] | 0.503 [0.290, 0.711] |
| lora-10k-s0 | 0.043 [0.000, 0.100] | 0.286 [0.000, 0.571] | 0.055 [0.000, 0.129] | 0.119 [0.000, 0.286] | 0.058 [0.000, 0.119] |
| lora-30k-s0 | 0.071 [0.014, 0.157] | 0.429 [0.143, 0.857] | 0.243 [0.029, 0.529] | 0.262 [0.048, 0.571] | 0.189 [0.055, 0.367] |

7 sorgu üzerinden, 3 sorgu dışlandı (q009, q015, q017). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| lora-30k-s0 − clip | -0.100 [-0.300, +0.043] | +0.000 [-0.571, +0.571] | +0.094 [-0.211, +0.445] | -0.071 [-0.405, +0.215] | -0.015 [-0.266, +0.237] |
| lora-10k-s0 − clip | -0.129 [-0.343, +0.000] | -0.143 [-0.429, +0.000] | -0.095 [-0.226, +0.000] | -0.214 [-0.500, +0.000] | -0.146 [-0.337, +0.000] |
| lora-30k-s0 − lora-10k-s0 | +0.029 [-0.029, +0.071] | +0.143 [-0.286, +0.714] | +0.188 [-0.062, +0.484] | +0.143 [-0.071, +0.357] | +0.131 [-0.028, +0.302] |
| quilt − clip | +0.114 [+0.057, +0.171]* | +0.429 [+0.143, +0.857]* | +0.448 [+0.148, +0.733]* | +0.155 [-0.203, +0.536] | +0.300 [+0.084, +0.553]* |
| lora-30k-s0 − quilt | -0.214 [-0.386, -0.057]* | -0.429 [-0.857, +0.143] | -0.354 [-0.783, +0.141] | -0.226 [-0.643, +0.214] | -0.315 [-0.649, +0.021] |

Caption-unseen: 0 of 260 judged pairs removed (0.0%); queries with a relevant image remaining: {'lenient': 10, 'strict': 7} of 10.

## Circularity check

Spearman(grade, caption overlap) = 0.2387 over 260 pairs.

## Caveats

- QuiltNet held-out status of CSV val split: unverified (STATUS §7.8.3).
- Judgments are pooled: unjudged images count as non-relevant (see the random control sample of the pool for how often this is wrong).
- Annotators are not pathologists; grades are noisy (see the agreement report).
- The golden set passed the 0.99 near-duplicate exclusion; it is not clean at the 0.95 criterion (ADR-0003).
