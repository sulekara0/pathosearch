# golden_v0 sızıntı analizi (betimsel)

Komut: `uv run python scripts/leakage_analysis.py --golden eval_data/golden_v0` (seed 42, ham sayılar `leakage_analysis_golden_v0.json`, çiftler `leakage_pairs_golden_v0.html`).
Ön koşul: `reports/leakage_golden_v0.json` (`scripts/leakage_report.py`, τ=0.95). Bu analiz τ'yu ve sızıntı kodunu **değiştirmez**, yalnızca ölçer.

## 0. Önceki raporun düzeltmesi

`leakage_golden_v0.json` içindeki `violation_counts` (799 ve 2194) **iki encoder'ın toplamıdır**, ayrı görsel/çift sayısı değildir.

| | CLIP | QuiltNet | Birleşim | Kesişim |
|---|---|---|---|---|
| Ana korpusa cos > 0.95 olan golden görseli | 659 | 140 | 677 | 122 |

## 1. Golden → ana korpus: en yakın komşu cosine dağılımı (928 görsel, 114,222 ana görsele karşı)

| | CLIP | QuiltNet |
|---|---|---|
| min | 0.5498 | 0.7777 |
| p50 | 0.9606 | 0.9189 |
| p90 | 0.9789 | 0.9606 |
| p99 | 0.9988 | 0.9987 |
| maks | 1.0000 | 1.0000 |
| ≥ 0.95 | 659 | 140 |
| ≥ 0.99 | 28 | 33 |
| ≥ 0.999 | 9 | 8 |
| ≥ 0.9999 | 7 | 7 |

Histogram (20 kutu, [min, 1]) JSON'da `golden_vs_main.histogram`.

## 2. Referans: rastgele 928 ana korpus görseli → ana korpusun geri kalanı (kendi hariç)

Aynı korpus boyutu, seed 42.

| | CLIP | QuiltNet |
|---|---|---|
| min | 0.6741 | 0.8162 |
| p50 | 0.9774 | 0.9741 |
| p90 | 0.9997 | 0.9995 |
| p99 | 1.0000 | 1.0000 |
| ≥ 0.95 | 804 | 597 |
| ≥ 0.99 | 345 | 355 |
| ≥ 0.999 | 142 | 145 |
| ≥ 0.9999 | 66 | 66 |

**Okuma:**
- CLIP'te 0.95 bu alanda "normal": ana korpus görsellerinin bile %86.6'sının (804/928) eşi ≥ 0.95. Golden'ın medyanı (0.9606) referansın (0.9774) altında. Yani CLIP için 0.95 yakın kopyayı ayırt etmiyor, ihlal sayısı (659) büyük ölçüde CLIP'in histoloji görselleri birbirine yaklaştırmasını yansıtıyor (yorum).
- QuiltNet'te 0.95 daha seçici (referansta %64 ≥ 0.95, golden'da %15).
- Her iki encoder'da golden, ana korpusun kendi iç kopya yoğunluğundan **daha az** kopyaya sahip (≥ 0.9999: golden 7, referans 66). Ana korpusun kendi içinde çok kopya olması bu veri kümesinin (çok-kaynaklı, tekrar eden görseller) özelliği.
- Yüksek eşiklerde (≥ 0.99) iki encoder uyumlu: CLIP 28, QuiltNet 33. Bu, ≥ 0.99 bölgesinde encoder'dan bağımsız gerçek kopyaların bulunduğunu gösteriyor (yorum).

## 3. cos ≥ 0.9999 çiftler

Her iki encoder'da **aynı 7 golden görsel** çıkıyor (en yakın komşu 2 görselde encoder'a göre değişiyor). Yan yana HTML: `reports/leakage_pairs_golden_v0.html`. Not: ana korpus görsellerinin çoğu diskte yok (STATUS §7.7.3), bu yüzden 14 satırın çoğunda yalnızca golden tarafı görüntülenebildi, ana taraf "image not on disk" gösteriyor. Görsellere bakılarak değil, ID ve caption'lara dayanarak yorumlandı.

| golden görsel | en yakın ana görsel (CLIP) | caption |
|---|---|---|
| 1592807763981524992_2.jpg | 1071718073856446464_2.jpg | birebir aynı ("Signet-ring cell carcinoma. HE and MUC5AC+MUC6…") |
| 1604766599784718336_2.jpg | 1114049447330177024_2.jpg | birebir aynı ("Walthard cell nest. Ovary…") |
| 1604947727770533888_3.jpg | 1139872568561639424_3.jpg | bir harf farkı (sebaceus / sebaceous) |
| 1610021142663950338_1.jpg | 1141530066620260352_1.jpg | birebir aynı ("Mucoepidermoid carcinoma, clear cell variant…") |
| iklRyY1nBIE_image_1ca11c38….jpg | 0HcMuZN-pYo_image_ae57649b….jpg | farklı, genel cümleler |
| OqkkM3fudXk_image_687a7d08….jpg | ZesFec_tGt0_image_fc401e2d….jpg | farklı, genel cümleler |
| OqkkM3fudXk_image_8d3b8680….jpg | ZesFec_tGt0_image_d9a4c97b….jpg | farklı, genel cümleler |

Tam caption'lar JSON'da (`pairs_ge_threshold`). Yorum (doğrulanmadı): sayısal ID'li 4 çift aynı tweet görselinin farklı kimlikle tekrarı, 3 YouTube çifti farklı videoların kare kopyaları olabilir. Bu, golden `val` görsellerinin ana korpustaki `train` görselleriyle örtüştüğünü gösterir; Quilt-1M'in kendi içinde val/train kopyası var.

## 4. Golden içi yakın kopyalar

| | CLIP τ=0.95 | QuiltNet τ=0.95 | CLIP τ=0.99 | QuiltNet τ=0.99 |
|---|---|---|---|---|
| çift sayısı | 1,899 | 295 | 77 | 73 |
| kapsanan ayrı görsel | 529 (%57) | 258 (%28) | 115 | 107 |
| küme sayısı (bağlı bileşen) | 52 | 85 | 51 | 47 |
| en büyük küme | 357 | 12 | 4 | 6 |

(2194 = 1899 + 295 toplamıydı, ayrı çift sayısı değil.)

**Okuma:** CLIP τ=0.95'teki tek küme 357 görseli bağlıyor (zincirleme bağ); bu bir CLIP geometri etkisi gibi görünüyor, çünkü QuiltNet'te en büyük küme 12. τ=0.99'da iki encoder aynı büyüklükte sonuç veriyor (~110 görsel, ~50 küme, en büyük 4-6): golden içinde yaklaşık 110 görsellik, ~50 küme oluşturan gerçek kopya grubu var.

## 5. Sınırlar

- Referans (§2) ana korpusun kendi içindeki komşuluğudur; golden→ana ile birebir aynı örnekleme değildir (boyut aynı, seed 42).
- Cosine yakınlığı kanıt değil göstergedir; §3 yalnızca caption ve ID'lere dayanıyor.
- Hiçbir eşik değiştirilmedi; karar (eşik, eleme, `golden_v1`) proje sahibine ait.
