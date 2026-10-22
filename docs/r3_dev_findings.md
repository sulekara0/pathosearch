# R3: Dev bulguları (dondurulmuş yorum)

Tarih: 2026-10-22 · Kaynak: [reports/eval_golden_v0_dev_0102190.md](../reports/eval_golden_v0_dev_0102190.md) (komut, config ve seed raporda; `4c6435c` raporuyla sayılar birebir aynı) · Birincil metrik: lenient (ADR-0001)

Bu belge test split koşmadan **önce** yazıldı. Bu noktadan sonra dev sonuçlarına bakılarak parametre (α, β, havuz), sorgu ya da kural değiştirilmeyecek.

## Bulgular (dev, lenient, 8 sorgu üzerinden; q019 ve q020 dışlandı)

Aralıklar seed'li paired bootstrap %95 GA'dır; sayılar yukarıdaki raporun fark tablosundan alındı.

- **(a) QuiltNet, CLIP'ten belirgin iyi.** quilt − clip farkı P@10 +0.400 [+0.287, +0.500], Recall@10 +0.456 [+0.308, +0.626], MRR +0.346 [+0.142, +0.600], nDCG@10 +0.450 [+0.302, +0.598]. Dört aralık da sıfırı dışlıyor. Hit@10 aralığı sıfırı içeriyor (+0.125 [+0.000, +0.375]).
- **(b) QuiltNet üzerinde rerank yalnızca MRR'de sıfırı dışlıyor.** quilt+rerank − quilt: MRR +0.229 [+0.062, +0.458]. P@10 +0.037 [−0.013, +0.087], Recall@10 +0.064 [−0.029, +0.207], nDCG@10 +0.132 [−0.020, +0.325] aralıkları sıfırı içeriyor.
- **(c) CLIP üzerinde rerank P@10, Recall ve nDCG'de sıfırı dışlıyor, mutlak fark küçük.** clip+rerank − clip: P@10 +0.037 [+0.013, +0.075], Recall@10 +0.044 [+0.008, +0.088], nDCG@10 +0.088 [+0.006, +0.201]. MRR aralığı sıfırı içeriyor (+0.150 [−0.008, +0.313]). P@10 ve Recall farkı yaklaşık +0.04, nDCG farkı daha büyük (+0.088) ama alt sınırı sıfıra yakın.
- **(d) Kanıt düzeyi.** 8 sorgu (strict'te 6). Bu bir eğilim düzeyinde kanıttır, kesin ayrım değildir. Kategori başına en çok 2 sorgu var, pulmonary'de hiç sorgu kalmadı. Sorgu başına yeniden örnekleme güven aralıklarını geniş tutuyor ve sonuçlar tek tek sorgulara duyarlı.
- **(e) Tezdeki V4→V5b kazancı.** Bitirme raporundaki P@10 kazancı +0.070 (0.539 → 0.609, STATUS §7.9.2) bu ölçümde bu büyüklükte görünmüyor: QuiltNet'te rerank'in P@10 farkı +0.037 ve aralığı sıfırı içeriyor. Ölçümler farklı olduğu için (korpus, sorgu seti, relevance tanımı: kategori düzeyi ikili etiket ve golden set derecelendirmesi) iki sayı doğrudan karşılaştırılamaz. Bu belge eski iddiayı ne doğruluyor ne çürütüyor.

## Sınırlar

- Strict (ikincil) sonuçlar 6 sorguya dayanıyor ve aralıkları daha geniş; strict etiket uyumu düşük (kappa 0.11, ADR-0001).
- Rerank'in caption'a dayanması nedeniyle döngüsellik riski var. Dev'de Spearman(grade, caption overlap) = 0.212 (eşik 0.5, uyarı yok); caption-unseen alt kümesi bu koşuda tüm çiftleri koruduğu için ayrıca bilgi vermiyor.
- Yargılar havuzlanmış; havuza girmeyen görsel ilgisiz sayılıyor. Caveat'ler raporun `caveats` alanında.

## Sonraki adım

Test split tek kez, `--i-know-this-is-test` ile, aynı dört sistem ve k=10 ile koşar. Sonuç ne olursa olsun bu belgedeki yorum değişmez; test sonucu ayrıca raporlanır.
