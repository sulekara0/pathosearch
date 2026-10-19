# ADR-0003: Near-duplicate threshold and golden corpus exclusion

Status: accepted (2026-10-19) · Context: R2, golden_v0 · Evidence: [reports/leakage_analysis_golden_v0.md](../../reports/leakage_analysis_golden_v0.md)

## Bağlam

Golden set held-out görsellerden (diskteki yalnız-`val`, 928 görsel) kurulur ve ana korpusla (114,222 görsel) örtüşmemelidir. A8'de yakın kopya eşiği **τ=0.95** (CLIP ve QuiltNet için ayrı), veriye bakılmadan önce sabitlendi (`eval/heldout.py`, `NEAR_DUP_TAU`).

B3 sızıntı raporu `passed=False` verdi: ana korpusa cos > 0.95 olan golden görseli CLIP'te 659, QuiltNet'te 140 (birleşim 677). Bu sayılar iki encoder'ın toplamı olarak raporlanmıştı, ayrı görsel sayısı değildi.

## Bulgular (`leakage_analysis_golden_v0`, seed 42)

- **0.95 bu alanda ayırt edici değil.** Rastgele 928 ana korpus görseli, kendi hariç ana korpusun geri kalanına karşı: CLIP'te %87'sinin (804/928), QuiltNet'te %64'ünün (597/928) en yakın komşusu ≥ 0.95. Golden'ın medyan en yakın komşu cosine'i referansın altında (CLIP 0.961 vs 0.977; QuiltNet 0.919 vs 0.974).
- **Golden, ana korpustan daha temiz.** cos ≥ 0.9999: golden 7, referans 66 görsel. ≥ 0.99: golden CLIP 28 / QuiltNet 33; referans 345 / 355.
- **≥ 0.99 iki encoder'ın bağımsız uyuştuğu bölge.** İki encoder eşiğin üstünde benzer sayıda görsel veriyor (28 ve 33), ≥ 0.9999'da aynı 7 golden görseli çıkıyor. Bu 7 çiftte 3'ünün caption'ı birebir aynı, 1'inde tek harf farkı var.
- Golden içinde τ=0.99'da CLIP'te 115 görsel / 51 küme, QuiltNet'te 107 görsel / 47 küme (en büyük küme 4-6). τ=0.95'te CLIP'te 357 görsellik tek küme var, QuiltNet'te en büyüğü 12: 0.95'teki zincirleme CLIP geometri etkisi gibi görünüyor (yorum).

## Karar

1. **Raporlamada τ=0.95 kalır** (`leakage_report.py`, `NEAR_DUP_TAU`). Geçmişle karşılaştırılabilirlik için kod ve eşik değişmez; `leakage_golden_v0` raporu aynı tanımla yeniden üretilebilir.
2. **Ayrıca bir eleme adımı uygulanır** (`scripts/build_golden_corpus.py`):
   - `--exclude-near-dup <tau>`: iki encoder'dan **herhangi birinde** ana korpusa cos ≥ tau olan görseller manifestten çıkarılır. v0 için tau = **0.99**.
   - `--dedup-within <tau>`: kalan görseller arasında, iki encoder'dan herhangi birinde cos ≥ tau olan çiftlerin bağlı bileşenlerinden **tek temsilci** (dosya adı sırasıyla ilki, seed'siz, deterministik) bırakılır. v0 için tau = **0.99**.
   - İkisi de varsayılan **kapalı**; çıkarılanlar `manifest_meta.json`'a ve ayrı listelere yazılır.
3. **0.99 değeri bu analizden seçildi**, yani veriye bakıldıktan sonra. Bu bir kalibrasyon değil gerekçeli bir karardır; golden set etiketleri ve sorgular henüz yazılmadı, dolayısıyla eşik sonuç metriklerine bakılarak seçilmedi. Eşik değişirse yeni ADR gerekir.

## Sınırlar

- Cosine yakınlığı **gösterge**, kopya kanıtı değil. ≥ 0.9999 çiftlerinin yorumu ID ve caption'a dayanıyor, görsellere bakılmadı.
- Ana korpus görsellerinin çoğu diskte yok (76,126/114,222, STATUS §7.7.3), bu yüzden eşleşen çiftler gözle doğrulanamadı.
- Referans örneklemi golden→ana ile **birebir eşlenik değil** (ana korpus kendi içinde kıyaslanıyor, boyut ve seed aynı).
- Eleme yalnızca embedding benzerliğine bakar. 0.99 altında kalan yakın görseller (kırpma, renk farkı) korpusta kalabilir; bu kalıntı risk raporlanır, giderilmiş sayılmaz.
- Ana korpus eğitim verisi olduğu için, elemeden sonra da QuiltNet'in golden görselleri görmediği doğrulanmış olmaz (STATUS §7.8.3).

## Sonuçlar

- Golden korpus küçülür; kalan sayı ve ≥ 20 görselli kategori sayısı `manifest_meta.json`'da raporlanır.
- İki encoder'dan herhangi birine göre eleme, keskin tarafa yatar (daha çok görsel çıkar).
