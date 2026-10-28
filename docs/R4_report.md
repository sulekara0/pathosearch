# R4: LoRA fine-tune raporu (golden_v0, dev ve test)

Tarih: 2026-10-28 · Ön kayıt: [ADR-0004](adr/0004-r4-lora-preregistration.md) (Ek 1 dahil) · Dev yorumu (test öncesi donduruldu): [r4_dev_findings.md](r4_dev_findings.md) · Kaynak raporlar: [dev 29234fb](../reports/eval_golden_v0_dev_29234fb.md), [test 94d42fc](../reports/eval_golden_v0_test_94d42fc.md), [eğitim kümesi](../reports/train_manifest_r4_train_v0.md), [lora-10k-s0 eğitimi](../reports/r4_train_lora-10k-s0.json), [lora-30k-s0 eğitimi](../reports/r4_train_lora-30k-s0.json), [temel CLIP iç val](../reports/r4_val_baseline.json), [judged@10](../reports/r4_judged_at_10.json)

Bu belgedeki tüm sayılar yukarıdaki raporlardan, `eval_data/golden_v0/pool.jsonl` ve `judgments/A.jsonl`'dan betikle aktarıldı; elle yazılmadı. Test split ADR-0004 §10 uyarınca bir kez, dev yorumu dondurulduktan sonra koşuldu: test raporunun git sha'sı (`94d42fc`) dev bulgularını donduran commit'tir. [kayıt]

Tablolarda "birincil" yalnızca önceden kayıtlı tek hücreyi (`lora-30k-s0 − clip`, lenient nDCG@10) işaretler; aynı çiftin diğer metrikleri ve strict eşik keşifseldir.

**Kanıt düzeyi etiketleri** (her iddiada köşeli parantez içinde):
- **[ölçüldü, sıfırı dışlıyor]**: paired bootstrap %95 GA (n=1000, seed 42) sıfırı dışlıyor.
- **[ölçüldü, ayırt edilemedi]**: GA sıfırı içeriyor; "etki yok" değil, bu ölçümle ayırt edilemedi.
- **[tek ölçüm]**: GA'sız tek koşu sayısı (tek tohum).
- **[kayıt]**: git geçmişi, kod veya belgeden doğrudan.
- **[hipotez, test edilmedi]**: kanıt değil.

## 1. Amaç ve önceden kayıtlı hipotez

Soru (ADR-0004): genel CLIP (`openai/clip-vit-base-patch32`), golden set'ten bağımsız olduğu garanti edilen train-only Quilt görselleriyle LoRA üzerinden uyarlandığında golden_v0'da ölçülebilir kazanç sağlıyor mu?

- **Birincil karşılaştırma (tek):** `lora-30k-s0 − clip`, lenient nDCG@10, **test split**. "Kazanç var" iddiası yalnızca bu farkın %95 GA'sı sıfırı dışlarsa yapılır. Ek 1 (2026-10-26, hiçbir LoRA metriği görülmeden) birincili `lora-full-s0 − clip`'ten buna çevirdi ve `lora-full-s0`'ı kapsamdan çıkardı. [kayıt]
- **Keşifsel:** `lora-10k-s0 − clip`, `lora-30k-s0 − lora-10k-s0`, `quilt − clip`, `lora-30k-s0 − quilt`, diğer metrikler ve strict eşik. Çoklu karşılaştırma düzeltmesi yok. [kayıt]
- Domain gate kapalı, rerank yok, k=10. QuiltNet yalnızca referans. [kayıt]

## 2. Eğitim kümesi, sızıntı kontrolleri, ölçekler

Komut: `uv run python scripts/build_train_manifest.py --golden eval_data/golden_v0 --out eval_data/r4_train_v0` (git `12f747a`, seed 42, τ=0.99). Adaylar diskteki CSV-train-only görseller. [kayıt]

| adım | çıkarılan | kalan |
|---|---|---|
| `candidates (on disk, train-only)` |  | 64,369 |
| `no_embedding` | 26,285 | 38,084 |
| `golden_ids` | 0 | 38,084 |
| `golden_nearest` | 7 | 38,077 |
| `near_dup_golden` | 35 | 38,042 |
| `source_group` | 10 | 38,032 |
| `caption_equal` | 164 | 37,868 |
| `no_caption` | 0 | 37,868 |

- **Kaynak grubu:** golden ile aynı video / tweet / PubMed figürünü paylaşan 10 görsel çıkarıldı. PubMed UUID = figür varsayımı tahmindir (STATUS §7.10). [kayıt]
- **Yakın kopya (cos ≥ 0.99):** herhangi bir encoder'da bir golden adayına cos ≥ 0.99 olan 35 görsel, ayrıca ADR-0003'te golden adayının elenmesine yol açan 7 ana korpus görseli çıkarıldı. [kayıt]
- **Caption eşitliği:** normalize caption'ı bir golden caption'ına eşit 164 görsel çıkarıldı (çoğu kalıp caption `no medical information provided.`, ADR-0004 §1). [kayıt]
- **Embedding'i olmayan 26,285 görsel** yakın kopya kontrolü yapılamadığı için dışarıda; eğitim kümesi adayların 38,084/64,369 ≈ %59'lık kısmından geliyor. [kayıt]
- **İç doğrulama:** 713 bütün kaynak grubundan 1,929 görsel; train'de yakın kopyası olan 71 görsel düşürüldü. Yalnızca checkpoint seçimi içindir, golden'a dokunmaz. [kayıt]
- **Ölçekler (iç içe):** 10k = 10,000 ⊂ 30k = 30,000 ⊂ full = 35,868 (`full` koşulmadı, Ek 1). [kayıt]
- `assert_disjoint(train ∪ val, golden 928)` geçti (ADR-0004 §1). [kayıt]

## 3. Eğitim sonuçları

Ayarlar (iki koşuda aynı, golden'a bakılarak ayarlanmadı): r=16, α=32, dropout 0.1, hedef q_proj, v_proj; lr 0.0001, weight decay 0.1, warmup 0.05, batch 256, 10 epoch, grad clip 1.0, caption_seed 42. GPU Tesla T4, AMP bf16, eğitim git `8560d26` (`dirty=false`). Eğitilebilir parametre 983,041 / 152,260,354. Checkpoint: iç val kaybı en düşük epoch. [kayıt]

İç doğrulama (1,929 görsel, aynı val manifest ve caption dosyası, sha256 eşleşiyor):

| model | eğitim görseli | iç val kaybı | t2i R@1 | t2i R@5 | i2t R@1 | i2t R@5 |
|---|---|---|---|---|---|---|
| temel CLIP (eğitimsiz) | – | 8.105 | 0.019 | 0.036 | 0.017 | 0.035 |
| lora-10k-s0 (epoch 7) | 10,000 | 6.569 | 0.027 | 0.066 | 0.024 | 0.064 |
| lora-30k-s0 (epoch 6) | 30,000 | 6.233 | 0.030 | 0.090 | 0.032 | 0.091 |

- İç val kaybı 8.105 → 6.569 (10k) / 6.233 (30k); 30k'da göreli düşüş %23.1. t2i R@5 0.036 → 0.066 / 0.090. [tek ölçüm: tek tohum, GA yok]
- Süre: 10k 30 dk (390 adım), 30k 96 dk (1170 adım). [kayıt]
- İç val kaybı 30k'da epoch 6'dan, 10k'da epoch 7'den sonra düşmeyi bıraktı (son epoch: 6.247 / 6.575). [tek ölçüm]

**Sıcaklık hatasının yakalanması.** İlk sürümde sıcaklık `forward` içinde `clamp(max=100)` ile sınırlanıyordu. CLIP'in kayıtlı `logit_scale` değeri ln(100)'ün bir tık üstünde (exp = 100.0000076), yani başlangıç tam sınırdaydı ve orada clamp'in gradyanı sıfır: sıcaklık fiilen dondurulmuş olacaktı. Düzeltme (`10b7ab6`, 2026-10-22): sınır parametreye her optimizer adımından sonra uygulanıyor (OpenCLIP'teki gibi), `forward` clamp'siz. Regresyon testi `tests/test_train_loss.py::test_temperature_learns_at_the_cap` başlangıç noktasında gradyanın sıfır olmadığını doğruluyor. Düzeltme her iki eğitim koşusundan (`8560d26`) ve temel val ölçümünden (`d54df93`) önce geliyor (git ataları). [kayıt] Koşularda `logit_scale` 99.66–99.77 aralığında değişti, yani sıcaklık öğrendi; değerin 100'e yakın kalması ayrı bir gözlemdir, yorumlanmadı. [tek ölçüm]

## 4. Golden sonuçları (dev ve test yan yana)

Havuz genişletmesi (ADR-0004 §8, metriklerden önce): yalnızca `lora-*-s0` sistemlerinin yargılanmamış çiftleri `r4_extension` olarak havuza eklendi ve annotator A tarafından kör etiketlendi. [kayıt]
- Eklenen çift: 240 (dev 123, test 117); yargılanmayan 0; unsure 0; grade dağılımı 0 → 118, 1 → 93, 2 → 29. 2 çift iki kez yargılandı, `run_eval` son kaydı kullanır (`latest_judgments`). [kayıt]
- Etiketleme süresi: ardışık kayıtlar arasındaki ≤5 dk boşlukların toplamı ≈ 91 dk (eşik 2 dk: 70 dk, 10 dk: 98 dk); kayıt başına medyan aralık 11 sn. Zaman damgasından türetilmiş tahmin, ölçülmüş çalışma süresi değil. [kayıt, türetilmiş]

judged@10 (genişletme öncesi değerler `r4_judged_at_10.json`'dan; clip/quilt orada ölçülmedi):

| sistem | dev: genişletme öncesi | dev: sonrası | test: genişletme öncesi | test: sonrası |
|---|---|---|---|---|
| clip | – | 1.000 | – | 1.000 |
| quilt | – | 1.000 | – | 1.000 |
| lora-10k-s0 | 0.260 | 1.000 | 0.260 | 1.000 |
| lora-30k-s0 | 0.250 | 1.000 | 0.290 | 1.000 |

### 4.1 Sistem tablosu (lenient, mean [GA])

Lenient'te dev 8 sorgu üzerinden (dışlanan: q019, q020), test 10 sorgu üzerinden (dışlanan: yok). Dışlanan = havuzda ilgili bulunmadı (ADR-0001). clip ve quilt değerleri R3 raporlarıyla birebir aynı (betikte kontrol edildi).

| sistem | metrik | dev | test |
|---|---|---|---|
| clip | P@10 | 0.212 [0.113, 0.312] | 0.330 [0.200, 0.490] |
| clip | Recall@10 | 0.217 [0.136, 0.303] | 0.344 [0.247, 0.449] |
| clip | MRR | 0.362 [0.183, 0.583] | 0.608 [0.399, 0.814] |
| clip | nDCG@10 | 0.191 [0.097, 0.302] | 0.309 [0.196, 0.434] |
| quilt | P@10 | 0.613 [0.475, 0.713] | 0.530 [0.330, 0.720] |
| quilt | Recall@10 | 0.673 [0.565, 0.788] | 0.449 [0.320, 0.552] |
| quilt | MRR | 0.708 [0.500, 0.917] | 0.594 [0.355, 0.811] |
| quilt | nDCG@10 | 0.640 [0.490, 0.769] | 0.523 [0.335, 0.688] |
| lora-10k-s0 | P@10 | 0.200 [0.087, 0.338] | 0.110 [0.060, 0.160] |
| lora-10k-s0 | Recall@10 | 0.199 [0.076, 0.339] | 0.090 [0.043, 0.138] |
| lora-10k-s0 | MRR | 0.399 [0.149, 0.681] | 0.312 [0.142, 0.500] |
| lora-10k-s0 | nDCG@10 | 0.188 [0.069, 0.347] | 0.100 [0.054, 0.144] |
| lora-30k-s0 | P@10 | 0.188 [0.100, 0.300] | 0.130 [0.060, 0.200] |
| lora-30k-s0 | Recall@10 | 0.201 [0.096, 0.319] | 0.142 [0.059, 0.228] |
| lora-30k-s0 | MRR | 0.328 [0.162, 0.552] | 0.467 [0.233, 0.717] |
| lora-30k-s0 | nDCG@10 | 0.194 [0.086, 0.332] | 0.206 [0.094, 0.311] |

### 4.2 Fark tablosu (lenient, birincil eşik; `*` = GA sıfırı dışlıyor)

| karşılaştırma | rol | metrik | dev | test |
|---|---|---|---|---|
| lora-30k-s0 − clip | keşifsel | P@10 | -0.025 [-0.138, +0.087] | -0.200 [-0.360, -0.070]* |
| lora-30k-s0 − clip | keşifsel | Recall@10 | -0.017 [-0.159, +0.137] | -0.202 [-0.309, -0.099]* |
| lora-30k-s0 − clip | keşifsel | MRR | -0.034 [-0.118, +0.048] | -0.141 [-0.360, +0.133] |
| lora-30k-s0 − clip | birincil | nDCG@10 | +0.003 [-0.096, +0.101] | -0.102 [-0.253, +0.045] |
| lora-10k-s0 − clip | keşifsel | P@10 | -0.013 [-0.125, +0.087] | -0.220 [-0.370, -0.090]* |
| lora-10k-s0 − clip | keşifsel | Recall@10 | -0.018 [-0.151, +0.128] | -0.254 [-0.386, -0.133]* |
| lora-10k-s0 − clip | keşifsel | MRR | +0.037 [-0.117, +0.201] | -0.296 [-0.501, -0.112]* |
| lora-10k-s0 − clip | keşifsel | nDCG@10 | -0.003 [-0.085, +0.077] | -0.209 [-0.337, -0.103]* |
| lora-30k-s0 − lora-10k-s0 | keşifsel | P@10 | -0.013 [-0.075, +0.050] | +0.020 [-0.030, +0.070] |
| lora-30k-s0 − lora-10k-s0 | keşifsel | Recall@10 | +0.002 [-0.063, +0.059] | +0.051 [-0.011, +0.132] |
| lora-30k-s0 − lora-10k-s0 | keşifsel | MRR | -0.071 [-0.226, +0.068] | +0.155 [-0.008, +0.327] |
| lora-30k-s0 − lora-10k-s0 | keşifsel | nDCG@10 | +0.006 [-0.053, +0.071] | +0.107 [+0.012, +0.203]* |
| quilt − clip | keşifsel | P@10 | +0.400 [+0.287, +0.500]* | +0.200 [+0.020, +0.380]* |
| quilt − clip | keşifsel | Recall@10 | +0.456 [+0.308, +0.626]* | +0.105 [-0.097, +0.300] |
| quilt − clip | keşifsel | MRR | +0.346 [+0.142, +0.600]* | -0.013 [-0.319, +0.273] |
| quilt − clip | keşifsel | nDCG@10 | +0.450 [+0.302, +0.598]* | +0.214 [+0.009, +0.413]* |
| lora-30k-s0 − quilt | keşifsel | P@10 | -0.425 [-0.525, -0.325]* | -0.400 [-0.590, -0.220]* |
| lora-30k-s0 − quilt | keşifsel | Recall@10 | -0.473 [-0.550, -0.379]* | -0.307 [-0.447, -0.165]* |
| lora-30k-s0 − quilt | keşifsel | MRR | -0.380 [-0.609, -0.177]* | -0.128 [-0.467, +0.228] |
| lora-30k-s0 − quilt | keşifsel | nDCG@10 | -0.447 [-0.561, -0.338]* | -0.316 [-0.524, -0.108]* |

### 4.3 Birincil karşılaştırma sonucu

**`lora-30k-s0 − clip`, lenient nDCG@10, test: -0.102 [-0.253, +0.045].** GA sıfırı içeriyor. **Önceden kayıtlı hipotez desteklenmedi: LoRA'nın CLIP'e göre kazancı gösterilemedi.** [ölçüldü, ayırt edilemedi] Nokta tahmin negatif; bu, "LoRA daha kötü" iddiası için de yeterli değildir (GA sıfırı içeriyor).

### 4.4 Strict (ikincil)

Strict'te dev 6 sorgu (dışlanan: q004, q008, q019, q020), test 7 sorgu (dışlanan: q009, q015, q017). Strict etiket uyumu düşük (kappa 0.11, `agreement_golden_v0`).

| karşılaştırma | rol | metrik | dev | test |
|---|---|---|---|---|
| lora-30k-s0 − clip | keşifsel | P@10 | +0.017 [-0.033, +0.067] | -0.100 [-0.300, +0.043] |
| lora-30k-s0 − clip | keşifsel | Recall@10 | +0.167 [-0.167, +0.583] | +0.094 [-0.211, +0.445] |
| lora-30k-s0 − clip | keşifsel | MRR | +0.088 [-0.031, +0.227] | -0.071 [-0.405, +0.215] |
| lora-30k-s0 − clip | keşifsel | nDCG@10 | +0.063 [-0.066, +0.180] | -0.015 [-0.266, +0.237] |
| lora-10k-s0 − clip | keşifsel | P@10 | -0.017 [-0.067, +0.033] | -0.129 [-0.343, +0.000] |
| lora-10k-s0 − clip | keşifsel | Recall@10 | -0.042 [-0.292, +0.208] | -0.095 [-0.226, +0.000] |
| lora-10k-s0 − clip | keşifsel | MRR | +0.003 [-0.114, +0.117] | -0.214 [-0.500, +0.000] |
| lora-10k-s0 − clip | keşifsel | nDCG@10 | -0.019 [-0.118, +0.081] | -0.146 [-0.337, +0.000] |
| lora-30k-s0 − lora-10k-s0 | keşifsel | P@10 | +0.033 [+0.000, +0.067] | +0.029 [-0.029, +0.071] |
| lora-30k-s0 − lora-10k-s0 | keşifsel | Recall@10 | +0.208 [+0.000, +0.542] | +0.188 [-0.062, +0.484] |
| lora-30k-s0 − lora-10k-s0 | keşifsel | MRR | +0.086 [+0.014, +0.167]* | +0.143 [-0.071, +0.357] |
| lora-30k-s0 − lora-10k-s0 | keşifsel | nDCG@10 | +0.082 [-0.002, +0.184] | +0.131 [-0.028, +0.302] |
| quilt − clip | keşifsel | P@10 | +0.167 [+0.050, +0.317]* | +0.114 [+0.057, +0.171]* |
| quilt − clip | keşifsel | Recall@10 | +0.509 [+0.176, +0.843]* | +0.448 [+0.148, +0.733]* |
| quilt − clip | keşifsel | MRR | +0.565 [+0.248, +0.860]* | +0.155 [-0.203, +0.536] |
| quilt − clip | keşifsel | nDCG@10 | +0.501 [+0.260, +0.735]* | +0.300 [+0.084, +0.553]* |
| lora-30k-s0 − quilt | keşifsel | P@10 | -0.150 [-0.300, -0.033]* | -0.214 [-0.386, -0.057]* |
| lora-30k-s0 − quilt | keşifsel | Recall@10 | -0.343 [-0.518, -0.167]* | -0.354 [-0.783, +0.141] |
| lora-30k-s0 − quilt | keşifsel | MRR | -0.477 [-0.778, -0.190]* | -0.226 [-0.643, +0.214] |
| lora-30k-s0 − quilt | keşifsel | nDCG@10 | -0.438 [-0.594, -0.236]* | -0.315 [-0.649, +0.021] |

### 4.5 Diğer kontroller

- Döngüsellik: Spearman(grade, caption örtüşmesi) dev 0.212 (243 çift), test 0.239 (260 çift); eşik 0.5, uyarı yok. [ölçüldü]
- Caption-unseen alt kümesi iki split'te de 0 / 0 çift çıkardı; ek bilgi vermiyor. [kayıt]
- ADR-0004 §9'daki keşifsel duyarlılık analizi (eğitim kümesine en yakın komşu cos < 0.95 alt kümesi) **bu raporda koşulmadı**. [kayıt]

## 5. Dev ve test tutarlılığı (lenient)

- **Hiçbir split'te LoRA, CLIP'ten iyi değil.** Dört metrik × iki LoRA sistemi × iki split'te, `lora − clip` farkının pozitif yönde sıfırı dışladığı hücre sayısı: 0. Birincil fark dev'de +0.003 [-0.096, +0.101], test'te -0.102 [-0.253, +0.045]. [ölçüldü, ayırt edilemedi]
- **Negatif yönde sıfırı dışlayan LoRA − CLIP hücreleri:** dev'de yok. Test'te `lora-10k-s0 − clip`: P@10, Recall@10, MRR, nDCG@10; `lora-30k-s0 − clip`: P@10, Recall@10. [ölçüldü, sıfırı dışlıyor; keşifsel, düzeltmesiz]
- **Test'te 10K belirgin kötü:** `lora-10k-s0 − clip` nDCG@10 test -0.209 [-0.337, -0.103]*, dev -0.003 [-0.085, +0.077]. Dev'deki "ayırt edilemedi" test'te negatif yönde sıfırı dışlayan bir farka döndü. [ölçüldü, sıfırı dışlıyor; keşifsel]
- **30K − 10K:** nDCG@10 test +0.107 [+0.012, +0.203]* sıfırı dışlıyor, dev +0.006 [-0.053, +0.071] dışlamıyor. İki nokta bir eğri değildir; ölçekleme iddiası yapılmaz (Ek 1). [ölçüldü, sıfırı dışlıyor (yalnız test); keşifsel]
- **QuiltNet referansı:** `quilt − clip` nDCG@10 dev +0.450 [+0.302, +0.598]*, test +0.214 [+0.009, +0.413]*; `lora-30k-s0 − quilt` dev -0.447 [-0.561, -0.338]*, test -0.316 [-0.524, -0.108]*. İki split'te de sıfırı dışlıyor. [ölçüldü, sıfırı dışlıyor; keşifsel; QuiltNet eğitim verisi belirsiz, §7]
- **Dev yorumu ile karşılaştırma ([r4_dev_findings.md](r4_dev_findings.md)):** (a) birincil fark "gösterilemedi" test'te de geçerli. (b) dev'de sıfırı içeren `lora-10k-s0 − clip` ve `lora-30k-s0 − lora-10k-s0` test'te sıfırı dışlıyor; yani (b) test'te tekrarlanmadı. (c) QuiltNet'in önde olması test'te de geçerli. [kayıt]

## 6. Neden? Test edilmemiş hipotezler

İç val kaybı belirgin düştü (§3), golden'da kazanç görülmedi (§4). Aşağıdakilerin **hiçbiri test edilmedi ve hiçbiri kanıtlanmadı**; R4 kapsamında bunları ayırt edecek bir deney koşulmadı.

1. **Dağılım farkı (caption ↔ sorgu):** eğitim caption'ları uzun ve gürültülü (video anlatımı, tweet metni), golden sorguları kısa ve kavramsal. Model eğitim caption dağılımına uyum sağlayıp kısa sorgularda zayıflamış olabilir. İlgili ama test olmayan gözlem: eğitim boyunca metin kodlayıcının uzunluk sınırında kesilen caption sayısı 10k'da 4,336 / 99,840 çekim, 30k'da 12,839 / 299,520 çekim. [hipotez, test edilmedi]
2. **İç doğrulama aynı dağılımdan:** iç val, eğitimle aynı kaynak türünden ve aynı caption biçiminden geliyor. Kayıp düşüşü bu dağılıma uyumu ölçer; golden'daki görev (kısa sorgu → görsel, insan derecelendirmesi) farklı olduğundan iç val iyileşmesi golden'a taşınmayabilir. [hipotez, test edilmedi]
3. **Unutma:** LoRA uyarlaması, temel CLIP'in genel görsel-dil hizalamasının golden sorgularına yarayan kısmını bozmuş olabilir (10k'nın test'te CLIP'in altına düşmesi bununla tutarlı olurdu, ama tek başına kanıt değildir). [hipotez, test edilmedi]

## 7. Sınırlar

- **Sorgu sayısı:** lenient'te dev 8 + test 10 sorgu (strict 6 + 7). Aralıklar geniş.
- **Tek tohum:** yalnızca s0; seed varyansı ölçülmedi. 10k ile 30k arasındaki fark tohum gürültüsünden ayrılamaz.
- **`lora-full-s0` koşulmadı** (ADR-0004 Ek 1, süre ve kaynak). Full ölçeği (35,868) ölçülmedi; ölçek karşılaştırması yalnızca iki noktadır.
- **Hiperparametre ayarlanmadı:** sonuç "LoRA'nın en iyi hâli" değil, ön kayıtlı ayarın sonucudur. Epoch sabit olduğu için ölçek farkı adım sayısı farkını da içerir.
- **Çoklu karşılaştırma düzeltmesi yok:** her tabloda 5 karşılaştırma × 4 metrik × 2 eşik; tek bir `*` kesin kanıt sayılmaz. Yalnızca birincil karşılaştırma önceden kayıtlıdır.
- **QuiltNet referansı belirsiz:** golden görsellerinin QuiltNet eğitiminde görülüp görülmediği bilinmiyor (STATUS §7.8.3, §7.10). "LoRA QuiltNet'in gerisinde" genellenebilir bir model karşılaştırması değildir.
- **`r4_extension` için ikinci etiketleyici yok:** 240 çiftin tamamını yalnız A etiketledi; bu çiftlerde uyum (kappa) ölçülmedi. Etiketleyiciler patolog değil (orijinal havuzda lenient kappa 0.62, strict 0.11, `agreement_golden_v0`).
- **Eğitim kümesi adayların %59'ı**, rastgele seçilmedi; diskteki görseller yalnızca `images_part_1`. Eleme cos 0.99'da yapıldı, benzer ama eşik altı görseller eğitimde kalabilir; §9 duyarlılık analizi koşulmadı.
- **GPU'da bit düzeyinde tekrar üretilebilirlik garanti edilmez** (ADR-0004).
- Yalnızca metin→görsel ölçüldü. Sistem eğitim/araştırma amaçlıdır; klinik geçerlilik iddiası yoktur.

## 8. CV'ye aktarılabilir ifade

Türkçe:
> Histopatoloji metin→görsel aramada CLIP ViT-B/32'nin LoRA ile uyarlanmasını önceden kaydettim (ADR); golden set'ten kaynak grubu, yakın kopya (cos ≥ 0.99) ve caption eşitliğiyle ayrıştırılmış 30,000 Quilt-1M görseliyle eğittim. İç doğrulama kaybı %23 düştü, ancak insan etiketli golden test setinde önceden kayıtlı birincil karşılaştırmada kazanç gösterilemedi (nDCG@10 farkı -0.102, %95 GA [-0.253, +0.045], 10 sorgu); sonucu sınırlarıyla birlikte negatif sonuç olarak raporladım.

English:
> Pre-registered a LoRA fine-tuning study of CLIP ViT-B/32 for histopathology text-to-image retrieval, trained on 30,000 Quilt-1M images separated from the golden set by source group, near-duplicate (cos ≥ 0.99) and caption-equality filters. In-domain validation loss dropped by 23%, but the pre-registered primary comparison on a human-judged golden test set showed no detectable gain (ΔnDCG@10 -0.102, 95% CI [-0.253, +0.045], 10 queries); reported as a negative result with its limitations.
