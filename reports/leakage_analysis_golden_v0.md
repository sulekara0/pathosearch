# golden_v0 sızıntı analizi: 820 görsel (0.99 eleme SONRASI)

Komut: `uv run python scripts/leakage_analysis.py --golden eval_data/golden_v0` (seed 42, ham sayılar `leakage_analysis_golden_v0.json`).
Manifest: `build_golden_corpus.py --rule val_only_on_disk --out eval_data/golden_v0 --exclude-near-dup 0.99 --dedup-within 0.99` (928 aday → 37 ana korpusa yakın, 71 golden içi kopya çıkarıldı → 820). Gerekçe: [ADR-0003](../docs/adr/0003-near-duplicate-threshold.md). Eleme öncesi analiz: [leakage_analysis_golden_v0_candidates928.md](leakage_analysis_golden_v0_candidates928.md).
τ=0.95 sızıntı raporu (`leakage_golden_v0.json`) bilgi amaçlıdır; `passed=False` beklenen durumdur.

## 1. Golden → ana korpus (820 görsel, 114,222 ana görsele karşı)

| | CLIP | QuiltNet |
|---|---|---|
| min | 0.5498 | 0.7777 |
| p50 | 0.9600 | 0.9178 |
| p90 | 0.9768 | 0.9515 |
| p99 | 0.9837 | 0.9832 |
| maks | **0.9886** | **0.9877** |
| ≥ 0.95 | 575 | 94 |
| ≥ 0.99 / 0.999 / 0.9999 | 0 / 0 / 0 | 0 / 0 / 0 |

0.99 üstü kalıntı yok. ≥ 0.9999 çift olmadığı için yan yana HTML üretilmedi (928'lik aday kümenin çift listesi: `leakage_pairs_golden_v0_candidates928.html`).
Ana korpusa ≥ 0.95 olan ayrı görsel: CLIP 575, QuiltNet 94, birleşim 592, kesişim 77.

## 2. Referans: rastgele 820 ana korpus görseli → ana korpusun geri kalanı (kendi hariç)

Seed 42. Örnek boyutu 820 olduğu için 928'lik analizdeki referans örnekle aynı değildir.

| | CLIP | QuiltNet |
|---|---|---|
| p50 | 0.9802 | 0.9774 |
| p90 | 0.9995 | 0.9998 |
| ≥ 0.95 | 694 (%85) | 525 (%64) |
| ≥ 0.99 | 319 | 330 |
| ≥ 0.9999 | 52 | 72 |

CLIP'te ana korpusun kendisi bile 0.95'in üstünde yoğun; golden'daki 575 bu alandaki CLIP geometrisinin yansıması (yorum).

## 3. Golden içi yakın kopyalar

| | CLIP τ=0.95 | QuiltNet τ=0.95 | τ=0.99 (iki encoder) |
|---|---|---|---|
| çift | 1,294 | 81 | 0 |
| kapsanan görsel | 406 | 114 | 0 |
| küme | 37 | 45 | 0 |
| en büyük küme | 289 | 5 | – |

CLIP τ=0.95'teki 289'luk küme zincirleme bağ; QuiltNet'te en büyük küme 5.

## 4. Sınırlar

- Cosine yakınlığı gösterge; ana korpus görsellerinin çoğu diskte yok, eşleşmeler gözle doğrulanamadı.
- Referans örneklem golden→ana ile birebir eşlenik değil.
- 0.99 ile 0.95 arasındaki görseller korpusta kalır (kabul edilen karar, ADR-0003 eki).
