# Agreement: golden_v0 overlap, A vs claude

- command: `uv run python scripts/agreement_report.py --golden eval_data/golden_v0 --b claude`
- git: `c9e5296` · bootstrap n=1000, seed=42, 95% percentile CI; units: pair (independent pairs) and query (whole queries resampled)
- labels as recorded, before adjudication; no qrels written
- adjudication: claude notes were optional (hidden by default, shown on request); opened for 0 of 27 adjudicated pairs

## 1. Pairs

| | count |
|---|---|
| overlap pairs | 111 |
| unsure by A | 6 |
| unsure by claude | 15 |
| unsure by both | 4 |
| excluded (unsure on either side) | 17 |
| comparable pairs | 94 (20 queries) |

## 2. Ordinal agreement (grades 0/1/2)

| stat | value | 95% CI (pair) | 95% CI (query) |
|---|---|---|---|
| raw agreement | 0.713 | [0.617, 0.809] | [0.562, 0.839] |
| Cohen's kappa | 0.447 | [0.303, 0.595] | [0.263, 0.637] |
| linear weighted kappa | 0.506 | [0.369, 0.637] | [0.336, 0.677] |

## 3. Binary thresholds

| threshold | stat | value | 95% CI (pair) | 95% CI (query) |
|---|---|---|---|---|
| lenient (grade >= 1) | raw agreement | 0.819 | [0.745, 0.894] | [0.707, 0.920] |
| lenient (grade >= 1) | kappa | 0.616 | [0.459, 0.768] | [0.395, 0.817] |
| strict (grade >= 2) | raw agreement | 0.883 | [0.809, 0.947] | [0.765, 0.979] |
| strict (grade >= 2) | kappa | 0.110 | [-0.065, 0.420] | [-0.073, 0.558] |

Binary confusion (rows = A, cols = claude; 0 / 1):

- lenient: `[51, 14]` / `[3, 26]`
- strict: `[82, 2]` / `[9, 1]`

## 4. Confusion matrix and marginals

Comparable pairs, rows = A, columns = claude.

| A \ claude | 0 | 1 | 2 | total |
|---|---|---|---|---|
| **0** | 51 | 14 | 0 | 65 |
| **1** | 2 | 15 | 2 | 19 |
| **2** | 1 | 8 | 1 | 10 |
| **total** | 54 | 37 | 3 | 94 |

| annotator | set | 0 | 1 | 2 | unsure |
|---|---|---|---|---|---|
| A | comparable | 65 | 19 | 10 | 0 |
| A | all overlap | 72 | 22 | 11 | 6 |
| claude | comparable | 54 | 37 | 3 | 0 |
| claude | all overlap | 55 | 38 | 3 | 15 |

## 5. Disagreements

27 disagreements: |A-claude|=2: 1, |A-claude|=1: 26. Sorted by |diff|, then qid.

By query category (all overlap pairs):

| category | pairs | unsure excl. | comparable | agree | diff 1 | diff 2 |
|---|---|---|---|---|---|---|
| dermatopathology | 22 | 4 | 18 | 7 | 10 | 1 |
| gastrointestinal | 22 | 2 | 20 | 11 | 9 | 0 |
| genitourinary | 24 | 6 | 18 | 16 | 2 | 0 |
| pulmonary | 22 | 4 | 18 | 18 | 0 | 0 |
| soft tissue | 21 | 1 | 20 | 15 | 5 | 0 |

| abs diff | qid | query | category | image | A | claude |
|---|---|---|---|---|---|---|
| 2 | q001 | basal cell carcinoma | dermatopathology | `QDb68_G1HR4_image_71fc4c72-1681-462d-bf6b-3fdef87d70ed.jpg` | 2 | 0 |
| 1 | q001 | basal cell carcinoma | dermatopathology | `3n3qWxn3Q4M_image_44e48028-2e91-431b-bc62-945dc6941964.jpg` | 2 | 1 |
| 1 | q001 | basal cell carcinoma | dermatopathology | `b283ddea-898e-4942-bb77-d116027c1884_0.jpg` | 2 | 1 |
| 1 | q001 | basal cell carcinoma | dermatopathology | `iklRyY1nBIE_image_927af67d-722b-4641-8bd6-630e17a89c1a.jpg` | 1 | 2 |
| 1 | q002 | malignant melanoma | dermatopathology | `8S4LeiO6Bbk_image_a7e10d09-4695-4ab5-94d5-0729c138ff60.jpg` | 2 | 1 |
| 1 | q002 | malignant melanoma | dermatopathology | `jCw_NtnS6XU_image_84cfabcd-cf9e-431b-8216-e9fb221612ce.jpg` | 0 | 1 |
| 1 | q002 | malignant melanoma | dermatopathology | `jCw_NtnS6XU_image_b2128b39-f5d4-4e06-9b03-c7551e28c457.jpg` | 0 | 1 |
| 1 | q002 | malignant melanoma | dermatopathology | `udoW6VSqsm4_image_4acaf616-1a3a-4ed9-9fe3-b2e04775e31a.jpg` | 0 | 1 |
| 1 | q003 | seborrheic keratosis | dermatopathology | `89e649d7-d86a-48bc-84f1-58f5a72d0268_0.jpg` | 0 | 1 |
| 1 | q003 | seborrheic keratosis | dermatopathology | `udoW6VSqsm4_image_291dd1e4-5e70-4e69-ba16-3182c42e2a94.jpg` | 0 | 1 |
| 1 | q004 | actinic keratosis | dermatopathology | `8S4LeiO6Bbk_image_96db8195-9fd3-493b-af2d-cae6a387df27.jpg` | 0 | 1 |
| 1 | q005 | colonic adenocarcinoma | gastrointestinal | `1608581674287370240_0.jpg` | 2 | 1 |
| 1 | q005 | colonic adenocarcinoma | gastrointestinal | `85609ad7-6ae2-467c-8eeb-d6806f581966.jpg` | 1 | 0 |
| 1 | q007 | normal colonic mucosa | gastrointestinal | `00010000103021.jpg` | 2 | 1 |
| 1 | q007 | normal colonic mucosa | gastrointestinal | `00035000359719.jpg` | 2 | 1 |
| 1 | q007 | normal colonic mucosa | gastrointestinal | `34d0d094-6cd5-4957-889f-b746826fd054_0.jpg` | 2 | 1 |
| 1 | q007 | normal colonic mucosa | gastrointestinal | `sDFjOtMAYrk_image_b6fff9b0-fcc6-4f64-9d86-d564469daca6.jpg` | 1 | 2 |
| 1 | q008 | helicobacter pylori gastritis | gastrointestinal | `00004000048078.jpg` | 0 | 1 |
| 1 | q008 | helicobacter pylori gastritis | gastrointestinal | `1604521878965452800_1.jpg` | 0 | 1 |
| 1 | q008 | helicobacter pylori gastritis | gastrointestinal | `e921f606-f04b-4064-998d-ce9ef5bd427e_1.jpg` | 1 | 0 |
| 1 | q009 | lipoma | soft tissue | `1613592513448935425_0.jpg` | 0 | 1 |
| 1 | q009 | lipoma | soft tissue | `QDb68_G1HR4_image_7228117a-7566-4287-aa7d-eefef7b88dd8.jpg` | 0 | 1 |
| 1 | q009 | lipoma | soft tissue | `j_rG5XPImFQ_image_4f92c831-4b76-430f-ad70-77301c2c8b4b.jpg` | 0 | 1 |
| 1 | q010 | liposarcoma | soft tissue | `QDb68_G1HR4_image_58733c53-2917-48ff-9e89-5e60a990067c.jpg` | 0 | 1 |
| 1 | q010 | liposarcoma | soft tissue | `QDb68_G1HR4_image_7228117a-7566-4287-aa7d-eefef7b88dd8.jpg` | 0 | 1 |
| 1 | q014 | clear cell renal cell carcinoma | genitourinary | `thWl623HaHI_image_c785c9e4-54c9-49f1-ae7b-d174b1126efc.jpg` | 2 | 1 |
| 1 | q015 | normal kidney glomerulus | genitourinary | `85609ad7-6ae2-467c-8eeb-d6806f581966.jpg` | 0 | 1 |

## 6. Limitations

- Small sample: 94 comparable pairs from 111 overlap pairs over 20 queries. Kappa CI width is 0.29 (pair) and 0.37 (query); the point estimate alone should not be quoted.
- Pairs are not independent: each query contributes several images judged with the same understanding of that query. The query-clustered CI accounts for this and is the more honest interval.
- Marginals differ (grade shares on comparable pairs, A: 0: 69% / 1: 20% / 2: 11%; claude: 0: 57% / 1: 39% / 2: 3%). Kappa depends on prevalence and on each annotator's marginals, so the same raw agreement can give quite different kappa values; read it together with the raw agreement and the confusion matrix.
- Unsure exclusion is not random: 6 unsure by A, 15 by claude (4 by both), 17 excluded in total. If unsure marks the harder pairs (plausible, not verified), kappa is biased upwards.
- Computed before adjudication on raw labels; it measures reproducibility of the grading guideline, not correctness of the labels.
- Neither annotator is a pathologist. High agreement means the two read the grading guideline the same way, not that the grades are clinically correct.
- Annotator 'claude' is an LLM judging blind (caption_seen=false). Agreement with it is human-model agreement, not inter-human agreement: a model can share systematic errors with the human or have its own consistent bias, so it is not a substitute for a second human annotator.
