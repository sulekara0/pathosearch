# Eval: golden_v0 / test — **TEST SPLIT**

- command: `uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split test --i-know-this-is-test --systems clip,quilt,clip+rerank,quilt+rerank --k 10`
- git: `8640cc1` · k=10 · queries=10 · unsure pairs=18
- config: {"alpha": 0.75, "beta": 0.25, "rerank_pool_factor": 5, "domain_gate": "off", "thresholds": {"lenient": 1, "strict": 2}, "bootstrap": {"n": 1000, "seed": 42, "ci": 0.95, "paired": true}}

## Metrics — lenient (grade >= 1, primary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.330 [0.200, 0.490] | 1.000 [1.000, 1.000] | 0.344 [0.247, 0.449] | 0.608 [0.399, 0.814] | 0.309 [0.196, 0.434] |
| quilt | 0.530 [0.330, 0.720] | 0.900 [0.700, 1.000] | 0.449 [0.320, 0.552] | 0.594 [0.355, 0.811] | 0.523 [0.335, 0.688] |
| clip+rerank | 0.410 [0.280, 0.560] | 1.000 [1.000, 1.000] | 0.437 [0.319, 0.591] | 0.631 [0.414, 0.848] | 0.439 [0.293, 0.607] |
| quilt+rerank | 0.550 [0.330, 0.760] | 0.800 [0.500, 1.000] | 0.454 [0.292, 0.614] | 0.733 [0.433, 1.000] | 0.531 [0.321, 0.722] |

10 sorgu üzerinden, 0 sorgu dışlandı (-). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| quilt − clip | +0.200 [+0.020, +0.380]* | -0.100 [-0.300, +0.000] | +0.105 [-0.097, +0.300] | -0.013 [-0.319, +0.273] | +0.214 [+0.009, +0.413]* |
| clip+rerank − clip | +0.080 [-0.040, +0.220] | +0.000 [+0.000, +0.000] | +0.093 [-0.006, +0.199] | +0.023 [-0.267, +0.313] | +0.131 [-0.012, +0.304] |
| quilt+rerank − quilt | +0.020 [-0.050, +0.080] | -0.100 [-0.300, +0.000] | +0.005 [-0.083, +0.086] | +0.139 [+0.000, +0.300] | +0.009 [-0.081, +0.097] |

## Metrics — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.171 [0.014, 0.386] | 0.429 [0.143, 0.857] | 0.149 [0.036, 0.293] | 0.333 [0.048, 0.667] | 0.204 [0.028, 0.407] |
| quilt | 0.286 [0.114, 0.486] | 0.857 [0.571, 1.000] | 0.597 [0.352, 0.817] | 0.488 [0.238, 0.774] | 0.503 [0.290, 0.711] |
| clip+rerank | 0.171 [0.029, 0.357] | 0.571 [0.143, 0.857] | 0.281 [0.071, 0.538] | 0.286 [0.095, 0.571] | 0.245 [0.060, 0.427] |
| quilt+rerank | 0.229 [0.114, 0.357] | 0.857 [0.571, 1.000] | 0.621 [0.302, 0.887] | 0.536 [0.286, 0.786] | 0.513 [0.280, 0.727] |

7 sorgu üzerinden, 3 sorgu dışlandı (q009, q015, q017). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| quilt − clip | +0.114 [+0.057, +0.171]* | +0.429 [+0.143, +0.857]* | +0.448 [+0.148, +0.733]* | +0.155 [-0.203, +0.536] | +0.300 [+0.084, +0.553]* |
| clip+rerank − clip | +0.000 [-0.043, +0.043] | +0.143 [+0.000, +0.429] | +0.132 [-0.033, +0.429] | -0.048 [-0.239, +0.143] | +0.041 [-0.071, +0.220] |
| quilt+rerank − quilt | -0.057 [-0.214, +0.043] | +0.000 [+0.000, +0.000] | +0.024 [-0.143, +0.214] | +0.048 [-0.286, +0.357] | +0.010 [-0.226, +0.266] |

## Metrics, caption-unseen pairs only — lenient (grade >= 1, primary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.330 [0.200, 0.490] | 1.000 [1.000, 1.000] | 0.344 [0.247, 0.449] | 0.608 [0.399, 0.814] | 0.309 [0.196, 0.434] |
| quilt | 0.530 [0.330, 0.720] | 0.900 [0.700, 1.000] | 0.449 [0.320, 0.552] | 0.594 [0.355, 0.811] | 0.523 [0.335, 0.688] |
| clip+rerank | 0.410 [0.280, 0.560] | 1.000 [1.000, 1.000] | 0.437 [0.319, 0.591] | 0.631 [0.414, 0.848] | 0.439 [0.293, 0.607] |
| quilt+rerank | 0.550 [0.330, 0.760] | 0.800 [0.500, 1.000] | 0.454 [0.292, 0.614] | 0.733 [0.433, 1.000] | 0.531 [0.321, 0.722] |

10 sorgu üzerinden, 0 sorgu dışlandı (-). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| quilt − clip | +0.200 [+0.020, +0.380]* | -0.100 [-0.300, +0.000] | +0.105 [-0.097, +0.300] | -0.013 [-0.319, +0.273] | +0.214 [+0.009, +0.413]* |
| clip+rerank − clip | +0.080 [-0.040, +0.220] | +0.000 [+0.000, +0.000] | +0.093 [-0.006, +0.199] | +0.023 [-0.267, +0.313] | +0.131 [-0.012, +0.304] |
| quilt+rerank − quilt | +0.020 [-0.050, +0.080] | -0.100 [-0.300, +0.000] | +0.005 [-0.083, +0.086] | +0.139 [+0.000, +0.300] | +0.009 [-0.081, +0.097] |

## Metrics, caption-unseen pairs only — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.171 [0.014, 0.386] | 0.429 [0.143, 0.857] | 0.149 [0.036, 0.293] | 0.333 [0.048, 0.667] | 0.204 [0.028, 0.407] |
| quilt | 0.286 [0.114, 0.486] | 0.857 [0.571, 1.000] | 0.597 [0.352, 0.817] | 0.488 [0.238, 0.774] | 0.503 [0.290, 0.711] |
| clip+rerank | 0.171 [0.029, 0.357] | 0.571 [0.143, 0.857] | 0.281 [0.071, 0.538] | 0.286 [0.095, 0.571] | 0.245 [0.060, 0.427] |
| quilt+rerank | 0.229 [0.114, 0.357] | 0.857 [0.571, 1.000] | 0.621 [0.302, 0.887] | 0.536 [0.286, 0.786] | 0.513 [0.280, 0.727] |

7 sorgu üzerinden, 3 sorgu dışlandı (q009, q015, q017). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Paired differences (mean [95% CI]; * = CI excludes 0):

| comparison | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| quilt − clip | +0.114 [+0.057, +0.171]* | +0.429 [+0.143, +0.857]* | +0.448 [+0.148, +0.733]* | +0.155 [-0.203, +0.536] | +0.300 [+0.084, +0.553]* |
| clip+rerank − clip | +0.000 [-0.043, +0.043] | +0.143 [+0.000, +0.429] | +0.132 [-0.033, +0.429] | -0.048 [-0.239, +0.143] | +0.041 [-0.071, +0.220] |
| quilt+rerank − quilt | -0.057 [-0.214, +0.043] | +0.000 [+0.000, +0.000] | +0.024 [-0.143, +0.214] | +0.048 [-0.286, +0.357] | +0.010 [-0.226, +0.266] |

Caption-unseen: 0 of 260 judged pairs removed (0.0%); queries with a relevant image remaining: {'lenient': 10, 'strict': 7} of 10.

## Circularity check

Spearman(grade, caption overlap) = 0.2387 over 260 pairs.

## Caveats

- QuiltNet held-out status of CSV val split: unverified (STATUS §7.8.3).
- Judgments are pooled: unjudged images count as non-relevant (see the random control sample of the pool for how often this is wrong).
- Annotators are not pathologists; grades are noisy (see the agreement report).
- The golden set passed the 0.99 near-duplicate exclusion; it is not clean at the 0.95 criterion (ADR-0003).
