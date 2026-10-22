# Eval: golden_v0 / dev

- command: `uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split dev --systems clip,quilt,clip+rerank,quilt+rerank --k 10`
- git: `4c6435c` · k=10 · queries=10 · unsure pairs=23
- config: {"alpha": 0.75, "beta": 0.25, "rerank_pool_factor": 5, "domain_gate": "off", "thresholds": {"lenient": 1, "strict": 2}, "bootstrap": {"n": 1000, "seed": 42, "ci": 0.95, "paired": true}}

## Metrics — lenient (grade >= 1, primary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.212 [0.113, 0.312] | 0.875 [0.625, 1.000] | 0.217 [0.136, 0.303] | 0.362 [0.183, 0.583] | 0.191 [0.097, 0.302] |
| quilt | 0.613 [0.475, 0.713] | 1.000 [1.000, 1.000] | 0.673 [0.565, 0.788] | 0.708 [0.500, 0.917] | 0.640 [0.490, 0.769] |
| clip+rerank | 0.250 [0.150, 0.350] | 0.875 [0.625, 1.000] | 0.262 [0.175, 0.340] | 0.512 [0.250, 0.775] | 0.279 [0.160, 0.396] |
| quilt+rerank | 0.650 [0.550, 0.762] | 1.000 [1.000, 1.000] | 0.738 [0.628, 0.858] | 0.938 [0.812, 1.000] | 0.772 [0.679, 0.859] |

8 sorgu üzerinden, 2 sorgu dışlandı (q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

## Metrics — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.050 [0.017, 0.083] | 0.500 [0.167, 0.833] | 0.143 [0.018, 0.292] | 0.074 [0.017, 0.138] | 0.071 [0.011, 0.139] |
| quilt | 0.217 [0.083, 0.383] | 0.833 [0.500, 1.000] | 0.653 [0.361, 0.903] | 0.639 [0.306, 0.917] | 0.572 [0.313, 0.792] |
| clip+rerank | 0.067 [0.033, 0.100] | 0.667 [0.333, 1.000] | 0.227 [0.056, 0.394] | 0.373 [0.039, 0.706] | 0.193 [0.051, 0.387] |
| quilt+rerank | 0.233 [0.117, 0.367] | 1.000 [1.000, 1.000] | 0.843 [0.676, 1.000] | 0.889 [0.667, 1.000] | 0.774 [0.543, 0.944] |

6 sorgu üzerinden, 4 sorgu dışlandı (q004, q008, q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

## Metrics, caption-unseen pairs only — lenient (grade >= 1, primary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.212 [0.113, 0.312] | 0.875 [0.625, 1.000] | 0.217 [0.136, 0.303] | 0.362 [0.183, 0.583] | 0.191 [0.097, 0.302] |
| quilt | 0.613 [0.475, 0.713] | 1.000 [1.000, 1.000] | 0.673 [0.565, 0.788] | 0.708 [0.500, 0.917] | 0.640 [0.490, 0.769] |
| clip+rerank | 0.250 [0.150, 0.350] | 0.875 [0.625, 1.000] | 0.262 [0.175, 0.340] | 0.512 [0.250, 0.775] | 0.279 [0.160, 0.396] |
| quilt+rerank | 0.650 [0.550, 0.762] | 1.000 [1.000, 1.000] | 0.738 [0.628, 0.858] | 0.938 [0.812, 1.000] | 0.772 [0.679, 0.859] |

8 sorgu üzerinden, 2 sorgu dışlandı (q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

## Metrics, caption-unseen pairs only — strict (grade >= 2, secondary)

| system | p | hit | recall | mrr | ndcg |
|---|---|---|---|---|---|
| clip | 0.050 [0.017, 0.083] | 0.500 [0.167, 0.833] | 0.143 [0.018, 0.292] | 0.074 [0.017, 0.138] | 0.071 [0.011, 0.139] |
| quilt | 0.217 [0.083, 0.383] | 0.833 [0.500, 1.000] | 0.653 [0.361, 0.903] | 0.639 [0.306, 0.917] | 0.572 [0.313, 0.792] |
| clip+rerank | 0.067 [0.033, 0.100] | 0.667 [0.333, 1.000] | 0.227 [0.056, 0.394] | 0.373 [0.039, 0.706] | 0.193 [0.051, 0.387] |
| quilt+rerank | 0.233 [0.117, 0.367] | 1.000 [1.000, 1.000] | 0.843 [0.676, 1.000] | 0.889 [0.667, 1.000] | 0.774 [0.543, 0.944] |

6 sorgu üzerinden, 4 sorgu dışlandı (q004, q008, q019, q020). Dışlanan: havuzda ilgili bulunmadı (ADR-0001).

Caption-unseen: 0 of 243 judged pairs removed (0.0%); queries with a relevant image remaining: {'lenient': 8, 'strict': 6} of 10.

## Circularity check

Spearman(grade, caption overlap) = 0.212 over 243 pairs.

## Caveats

- QuiltNet held-out status of CSV val split: unverified (STATUS §7.8.3).
- Judgments are pooled: unjudged images count as non-relevant (see the random control sample of the pool for how often this is wrong).
- Annotators are not pathologists; grades are noisy (see the agreement report).
- The golden set passed the 0.99 near-duplicate exclusion; it is not clean at the 0.95 criterion (ADR-0003).
