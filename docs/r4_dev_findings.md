# R4: Dev bulguları (dondurulmuş yorum)

Tarih: 2026-10-28 · Kaynak: [reports/eval_golden_v0_dev_29234fb.md](../reports/eval_golden_v0_dev_29234fb.md) (komut, config ve seed raporda) · Ön kayıt: [ADR-0004](adr/0004-r4-lora-preregistration.md) (Ek 1 dahil) · Birincil eşik: lenient (ADR-0001) · Sistemler: clip, quilt, lora-10k-s0, lora-30k-s0 · k=10

Bu belge test split koşmadan **önce** yazıldı. Bu noktadan sonra dev sonuçlarına bakılarak parametre, sorgu ya da kural değiştirilmeyecek.

## Yöntem notu: çift yargılar

Aynı (sorgu, görsel) için A.jsonl'da birden fazla kayıt olabiliyor (4 çift). `run_eval` yargıları `latest_judgments` ile okuyor (`golden.py`): log sırasında son kayıt kazanır, `unsure` (`grade: null`) bir sonraki kayıtla ezilebilir. `r4_extension` içindeki iki çiftten q009/`pheenISyPWk_…` için kayıtlar sırayla 0 ve 1; **kullanılan değer 1**. Diğeri (q013) iki kayıtta da 0. Bu bir kural değişikliği değildir, var olan davranışın kaydıdır.

## Bulgular (dev, lenient, 8 sorgu üzerinden; q019 ve q020 dışlandı)

Aralıklar seed'li (42) paired bootstrap %95 GA'dır (n=1000). Sayılar raporun fark tablosundan alındı. judged@10 dört sistemde de 1.000.

- **(a) Birincil karşılaştırma: fark gösterilemedi.** `lora-30k-s0 − clip`, lenient nDCG@10 = +0.003 [−0.096, +0.101]. Aralık sıfırı içeriyor.
- **(b) Keşifsel LoRA karşılaştırmaları da sıfırı içeriyor.** `lora-10k-s0 − clip` nDCG@10 −0.003 [−0.085, +0.077]; `lora-30k-s0 − lora-10k-s0` nDCG@10 +0.006 [−0.053, +0.071]. 10K → 30K arasında da fark gösterilemedi. (Strict'te `lora-30k-s0 − lora-10k-s0` MRR farkı +0.086 [+0.014, +0.167] sıfırı dışlıyor; 6 sorgu, çoklu karşılaştırma düzeltmesi yok, kanıt sayılmaz.)
- **(c) QuiltNet hâlâ referans olarak çok önde.** `quilt − clip` nDCG@10 +0.450 [+0.302, +0.598]; `lora-30k-s0 − quilt` nDCG@10 −0.447 [−0.561, −0.338]. İkisi de sıfırı dışlıyor. Bu, R3 dev sonucunun (quilt − clip) aynısıdır. QuiltNet'in eğitim verisi belirsizliği geçerli (ADR-0004 sınırları, STATUS §7.8.3, §7.10).
- **(d) İç doğrulama kazancı golden'a yansımadı.** İç val kaybı temel CLIP için 8.105 (`reports/r4_val_baseline.json`), `lora-30k-s0` için en iyi epoch'ta 6.233 (`reports/r4_train_lora-30k-s0.json`; `lora-10k-s0`: 6.569). Kayıpta büyük düşüş var, golden nDCG@10'unda fark yok. Nedeni için iki **hipotez** var, ikisi de **test edilmedi, kanıtlanmadı**:
  1. Eğitim caption'ları uzun ve gürültülü, golden sorguları kısa ve kavramsal; model eğitimde gördüğü dağılıma uyum sağlamış olabilir.
  2. İç val eğitimle aynı dağılımdan (aynı kaynaktan, aynı caption biçiminden) olduğu için kayıp düşüşü golden'a taşınmayabilir.
- **(e) Kanıt düzeyi: eğilim yok, kanıt zayıf.** 8 sorgu (strict'te 6), tek eğitim tohumu (s0), ölçek olarak yalnızca iki nokta (10K, 30K). İki nokta bir eğri değildir; ölçekleme iddiası yapılmaz. `lora-full-s0` koşulmadı (ADR-0004 Ek 1). Sıfırı içeren aralık "etki yok" demek değildir, bu ölçümle etki ayırt edilemedi demektir.

## Sınırlar

- Tek tohum: seed varyansı ölçülmedi.
- Hiperparametreler golden'a bakılarak ayarlanmadı (ADR-0004 §4); sonuç "LoRA'nın en iyi hâli" değildir.
- Adapter checkpoint'leri iç val kaybına göre seçildi: 10k epoch 7, 30k epoch 6.
- Dört sistemde judged@10 = 1.000 (240 çiftlik `r4_extension` genişletmesi sonrası); havuz sınırı yine de geçerli: havuza girmeyen görsel ilgisiz sayılır.
- Yalnızca metin→görsel ölçüldü. Klinik geçerlilik iddiası yoktur.
- Çoklu karşılaştırma düzeltmesi yok; tek bir `*` kesin kanıt sayılmaz.

## Sonraki adım

ADR-0004 §10: test split, bu belge commit edildikten sonra bir kez, `--i-know-this-is-test` ile, aynı sistem listesiyle koşar. Bu belgedeki yorum test sonucuna göre değiştirilmez.
