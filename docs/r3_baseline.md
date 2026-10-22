# R3: Baseline raporu (golden_v0, dev ve test)

Tarih: 2026-10-22 · Kaynaklar: [dev 0102190](../reports/eval_golden_v0_dev_0102190.md) (sayılar `4c6435c` ile birebir aynı), [test 8640cc1](../reports/eval_golden_v0_test_8640cc1.md) · k=10 · α=0.75, β=0.25, havuz çarpanı 5, domain gate kapalı, bootstrap n=1000 seed=42 · Birincil metrik: lenient (ADR-0001) · Test split bir kez çalıştırıldı.

Bölüm 1-2'deki tablolar raporlardan otomatik aktarıldı; Bölüm 3-5'teki metin içi sayılar elle yazıldı ve raporlarla betikle karşılaştırıldı (2026-10-22). Aralıklar %95 paired bootstrap GA'dır; `*` = GA sıfırı dışlıyor. Lenient'te dev 8 sorgu üzerinden (q019, q020 dışlandı), test 10 sorgu üzerinden (dışlanan yok).

## 1. Sistem tablosu (lenient, mean [GA])

| sistem | metrik | dev (8 sorgu) | test (10 sorgu) |
|---|---|---|---|
| clip | P@10 | 0.212 [0.113, 0.312] | 0.330 [0.200, 0.490] |
| clip | Recall@10 | 0.217 [0.136, 0.303] | 0.344 [0.247, 0.449] |
| clip | MRR | 0.362 [0.183, 0.583] | 0.608 [0.399, 0.814] |
| clip | nDCG@10 | 0.191 [0.097, 0.302] | 0.309 [0.196, 0.434] |
| clip | Hit@10 (ayrıca) | 0.875 [0.625, 1.000] | 1.000 [1.000, 1.000] |
| quilt | P@10 | 0.613 [0.475, 0.713] | 0.530 [0.330, 0.720] |
| quilt | Recall@10 | 0.673 [0.565, 0.788] | 0.449 [0.320, 0.552] |
| quilt | MRR | 0.708 [0.500, 0.917] | 0.594 [0.355, 0.811] |
| quilt | nDCG@10 | 0.640 [0.490, 0.769] | 0.523 [0.335, 0.688] |
| quilt | Hit@10 (ayrıca) | 1.000 [1.000, 1.000] | 0.900 [0.700, 1.000] |
| clip+rerank | P@10 | 0.250 [0.150, 0.350] | 0.410 [0.280, 0.560] |
| clip+rerank | Recall@10 | 0.262 [0.175, 0.340] | 0.437 [0.319, 0.591] |
| clip+rerank | MRR | 0.512 [0.250, 0.775] | 0.631 [0.414, 0.848] |
| clip+rerank | nDCG@10 | 0.279 [0.160, 0.396] | 0.439 [0.293, 0.607] |
| clip+rerank | Hit@10 (ayrıca) | 0.875 [0.625, 1.000] | 1.000 [1.000, 1.000] |
| quilt+rerank | P@10 | 0.650 [0.550, 0.762] | 0.550 [0.330, 0.760] |
| quilt+rerank | Recall@10 | 0.738 [0.628, 0.858] | 0.454 [0.292, 0.614] |
| quilt+rerank | MRR | 0.938 [0.812, 1.000] | 0.733 [0.433, 1.000] |
| quilt+rerank | nDCG@10 | 0.772 [0.679, 0.859] | 0.531 [0.321, 0.722] |
| quilt+rerank | Hit@10 (ayrıca) | 1.000 [1.000, 1.000] | 0.800 [0.500, 1.000] |

## 2. Fark tabloları

### Lenient (birincil)

| karşılaştırma | metrik | dev | test |
|---|---|---|---|
| quilt − clip | P@10 | +0.400 [+0.287, +0.500]* | +0.200 [+0.020, +0.380]* |
| quilt − clip | Recall@10 | +0.456 [+0.308, +0.626]* | +0.105 [-0.097, +0.300] |
| quilt − clip | MRR | +0.346 [+0.142, +0.600]* | -0.013 [-0.319, +0.273] |
| quilt − clip | nDCG@10 | +0.450 [+0.302, +0.598]* | +0.214 [+0.009, +0.413]* |
| quilt − clip | Hit@10 | +0.125 [+0.000, +0.375] | -0.100 [-0.300, +0.000] |
| clip+rerank − clip | P@10 | +0.037 [+0.013, +0.075]* | +0.080 [-0.040, +0.220] |
| clip+rerank − clip | Recall@10 | +0.044 [+0.008, +0.088]* | +0.093 [-0.006, +0.199] |
| clip+rerank − clip | MRR | +0.150 [-0.008, +0.313] | +0.023 [-0.267, +0.313] |
| clip+rerank − clip | nDCG@10 | +0.088 [+0.006, +0.201]* | +0.131 [-0.012, +0.304] |
| clip+rerank − clip | Hit@10 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| quilt+rerank − quilt | P@10 | +0.037 [-0.013, +0.087] | +0.020 [-0.050, +0.080] |
| quilt+rerank − quilt | Recall@10 | +0.064 [-0.029, +0.207] | +0.005 [-0.083, +0.086] |
| quilt+rerank − quilt | MRR | +0.229 [+0.062, +0.458]* | +0.139 [+0.000, +0.300] |
| quilt+rerank − quilt | nDCG@10 | +0.132 [-0.020, +0.325] | +0.009 [-0.081, +0.097] |
| quilt+rerank − quilt | Hit@10 | +0.000 [+0.000, +0.000] | -0.100 [-0.300, +0.000] |

### Strict (ikincil)

Strict'te dev 6 sorgu üzerinden (q004, q008, q019, q020 dışlandı), test 7 sorgu üzerinden (q009, q015, q017 dışlandı).

| karşılaştırma | metrik | dev | test |
|---|---|---|---|
| quilt − clip | P@10 | +0.167 [+0.050, +0.317]* | +0.114 [+0.057, +0.171]* |
| quilt − clip | Recall@10 | +0.509 [+0.176, +0.843]* | +0.448 [+0.148, +0.733]* |
| quilt − clip | MRR | +0.565 [+0.248, +0.860]* | +0.155 [-0.203, +0.536] |
| quilt − clip | nDCG@10 | +0.501 [+0.260, +0.735]* | +0.300 [+0.084, +0.553]* |
| quilt − clip | Hit@10 | +0.333 [+0.000, +0.667] | +0.429 [+0.143, +0.857]* |
| clip+rerank − clip | P@10 | +0.017 [+0.000, +0.050] | +0.000 [-0.043, +0.043] |
| clip+rerank − clip | Recall@10 | +0.083 [+0.000, +0.250] | +0.132 [-0.033, +0.429] |
| clip+rerank − clip | MRR | +0.299 [-0.036, +0.645] | -0.048 [-0.239, +0.143] |
| clip+rerank − clip | nDCG@10 | +0.122 [-0.015, +0.334] | +0.041 [-0.071, +0.220] |
| clip+rerank − clip | Hit@10 | +0.167 [+0.000, +0.500] | +0.143 [+0.000, +0.429] |
| quilt+rerank − quilt | P@10 | +0.017 [-0.033, +0.067] | -0.057 [-0.214, +0.043] |
| quilt+rerank − quilt | Recall@10 | +0.190 [-0.037, +0.523] | +0.024 [-0.143, +0.214] |
| quilt+rerank − quilt | MRR | +0.250 [-0.167, +0.667] | +0.048 [-0.286, +0.357] |
| quilt+rerank − quilt | nDCG@10 | +0.202 [-0.097, +0.556] | +0.010 [-0.226, +0.266] |
| quilt+rerank − quilt | Hit@10 | +0.167 [+0.000, +0.500] | +0.000 [+0.000, +0.000] |

## 3. Dev bulgularının testte durumu

Karşılaştırılan metin: [r3_dev_findings.md](r3_dev_findings.md) (test koşmadan önce donduruldu). Ölçüt: yönün ve aralığın sıfıra göre durumu, lenient.

- **(a) QuiltNet CLIP'ten belirgin iyi: kısmen doğrulandı, zayıfladı.** Testte quilt − clip P@10 (+0.200 [+0.020, +0.380]) ve nDCG@10 (+0.214 [+0.009, +0.413]) için sıfırı dışlıyor, ama alt sınırlar sıfıra yakın. Recall@10 (+0.105 [−0.097, +0.300]) ve MRR (−0.013 [−0.319, +0.273]) için aralıklar sıfırı içeriyor; MRR farkının işareti bile dev'den farklı. Dev'de dört metriğin dördü sıfırı dışlıyordu. Strict'te testte P@10, Hit@10, Recall@10 ve nDCG@10 sıfırı dışlıyor (MRR dışlamıyor).
- **(b) Rerank QuiltNet'te yalnızca MRR'de sıfırı dışlıyor: kısmen doğrulandı, MRR kısmı zayıfladı.** Testte P@10 (+0.020), Recall@10 (+0.005) ve nDCG@10 (+0.009) aralıkları sıfırı içeriyor; bu kısım doğrulandı. MRR farkı testte +0.139 [+0.000, +0.300]: nokta tahmin dev'den (+0.229) küçük ve alt sınır 0.000 (raporda `excludes_zero=false`), yani MRR'deki dev bulgusu testte sıfırı dışlamıyor.
- **(c) CLIP'te rerank P@10, Recall, nDCG'de sıfırı dışlıyor, fark küçük: zayıfladı.** Testte üç metrikte de aralık sıfırı içeriyor (P@10 +0.080 [−0.040, +0.220], Recall@10 +0.093 [−0.006, +0.199], nDCG@10 +0.131 [−0.012, +0.304]). Nokta tahminler pozitif ve dev'deki değerlerden büyük, ama dev'deki "sıfırı dışlıyor" bulgusu tekrarlanmadı. Strict'te testte de sıfırı dışlayan yok.
- **(d) Kanıt düzeyi: eğilim, kesin ayrım değil: doğrulandı.** Testte de 10 sorgu; aralıklar geniş ve birçok farkta sıfıra yakın.
- **(e) V4→V5b kazancı (+0.070) bu ölçümde görünmüyor: doğrulandı (yönü).** QuiltNet'te rerank'in P@10 farkı testte +0.020 [−0.050, +0.080], dev'de +0.037 [−0.013, +0.087]; ikisi de sıfırı içeriyor. Ölçümler farklı olduğu için eski iddia bu sonuçla ne doğrulanmış ne çürütülmüş olur.

Çürütülen madde yok. (a), (b) ve (c) dev'deki kadar kesin tekrarlanmadı.

## 4. Hit@10 örüntüsü

- CLIP'in Hit@10'u testte 1.000 [1.000, 1.000], devde 0.875 [0.625, 1.000]. clip+rerank testte de 1.000. QuiltNet testte 0.900, quilt+rerank 0.800, yani Hit@10 sıralaması testte P@10 ve nDCG@10 sıralamasından farklı.
- Hit@10 "ilk 10'da en az bir ilgili görsel var mı" sorusunu yanıtlar ve lenient eşikte (grade ≥ 1) kolay sağlanır. 10 sorguda bir sorgu 0.1'e denk gelir; testte quilt − clip Hit@10 farkı −0.100 [−0.300, +0.000] bu tek sorguya dayanıyor.
- Bu örüntü, CLIP'in tüm sorgularda en az bir ilgili görseli ilk 10'a getirdiğini gösterir; kaç tane getirdiğini göstermez (P@10 ve Recall@10 CLIP'te daha düşük). Hit@10 bu veride sistemleri ayıran bir metrik olarak zayıf; ADR-0001 birincil metrik seçimini değiştirmez.

## 5. Kanıt düzeyi ve sınırlar

- **Sorgu sayısı:** dev 8 + test 10 lenient sorgu (strict'te 6 + 7). Kategori başına en çok 2 sorgu; pulmonary dev'de hiç sorgu bırakmadı. İki split birleştirilmemiştir; farklı sorgular, farklı kategori dağılımı.
- **Etiketleyiciler:** birincil etiketleyici bir kişi (A), ikinci etiketleyici bir yapay zekâ (claude). İkisi de patolog değil; strict ikili kappa 0.11, lenient 0.62 (`agreement_golden_v0`). Nihai qrels'te uyuşmazlıklar adjudication ile çözüldü; bu, doğruluğu değil tutarlılığı artırır.
- **QuiltNet eğitim verisi belirsizliği:** golden görseller CSV `val` bölümünden alındı, ama QuiltNet eğitiminde `val`'in kullanılıp kullanılmadığı hiçbir kaynakta doğrulanmadı (STATUS §7.8.3). QuiltNet − CLIP farkı bu nedenle "yalnızca alan ön-eğitimi" diye yorumlanamaz.
- **Havuz sınırı:** yargılar havuzlanmıştır; havuza girmeyen görsel ilgisiz sayılır. Dışlanan sorgular "havuzda ilgili bulunmadı" demektir, "korpusta ilgili yok" demek değildir (ADR-0001). Metrikler havuz kapsamına bağlıdır.
- **Döngüsellik:** Spearman(grade, caption örtüşmesi) dev'de 0.212 (243 çift), testte 0.239 (260 çift); eşik 0.5, uyarı yok. Caption-unseen alt kümesi iki split'te de hiçbir çifti çıkarmadığı için ek bilgi vermiyor. Rerank caption'a dayandığı için bu kontrol rerank sonuçlarında ayrıca önemlidir.
- **Yakın kopya:** golden set 0.99 eşiğinden geçti, 0.95 ölçütüne göre temiz değil (ADR-0003).
- **Çoklu karşılaştırma:** her tabloda 4 metrik × 3 karşılaştırma × 2 eşik GA'sı var ve düzeltme uygulanmadı; tek bir `*` kesin kanıt sayılmamalıdır.
