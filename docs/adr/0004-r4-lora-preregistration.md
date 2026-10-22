# ADR-0004: R4 LoRA fine-tune ön kaydı (eğitim kümesi, ölçekler, birincil karşılaştırma)

Status: accepted (2026-10-22) · Context: R4 · Evidence: [reports/train_manifest_r4_train_v0.md](../../reports/train_manifest_r4_train_v0.md), [reports/train_captions_r4_train_v0.json](../../reports/train_captions_r4_train_v0.json), [reports/source_groups.json](../../reports/source_groups.json)

## Zamanlama

Bu karar **hiçbir LoRA modeli eğitilmeden ve golden set üzerinde hiçbir LoRA metriği görülmeden önce** yazıldı. O ana kadar golden_v0 üzerinde yalnızca R3 sistemleri (clip, quilt, ±rerank) ölçülmüştü (`docs/r3_baseline.md`). Aşağıdaki herhangi bir madde sonuçlar görüldükten sonra değişirse yeni bir ADR gerekir. Değişikliğin gerekçesi ve sonuçları gördükten sonra yapıldığı o ADR'de açıkça yazılır.

## Bağlam

R4'ün sorusu: genel CLIP (`openai/clip-vit-base-patch32`, HF), **golden set'ten bağımsız olduğunu kendimiz garanti ettiğimiz** train-only Quilt görselleriyle LoRA üzerinden uyarlandığında golden_v0'da ölçülebilir bir kazanç sağlıyor mu, ve bu kazanç veri miktarıyla nasıl değişiyor? QuiltNet yalnızca referanstır: golden görsellerini eğitimde görüp görmediği bilinmiyor (STATUS §7.8.3, §7.10).

## Karar

### 1. Eğitim kümesi: `eval_data/r4_train_v0`

Komut: `uv run python scripts/build_train_manifest.py --golden eval_data/golden_v0 --out eval_data/r4_train_v0` (seed 42, τ=0.99). Adaylar: diskteki CSV-train-only görseller. Eleme adımları sırayla:

| adım | çıkarılan | kalan |
|---|---|---|
| aday (diskte, train-only) | | 64,369 |
| `no_embedding`: ana korpus CLIP/QuiltNet embedding'i yok | 26,285 | 38,084 |
| `golden_ids`: golden adayının kendisi | 0 | 38,084 |
| `golden_nearest`: ADR-0003'te bir golden adayının elenmesine yol açan ana korpus görseli | 7 | 38,077 |
| `near_dup_golden`: herhangi bir encoder'da 928 golden adayından birine cos ≥ 0.99 | 35 | 38,042 |
| `source_group`: golden ile aynı video / tweet / PubMed figürü | 10 | 38,032 |
| `caption_equal`: normalize caption'ı bir golden caption'ına eşit | 164 | 37,868 |
| `no_caption`: kullanılabilir caption yok | 0 | 37,868 |

- **İç doğrulama:** kaynak grubu bazında 713 bütün grup alındı (hedef 2,000 görsel). Train'de cos ≥ 0.99 yakın kopyası olan 71 görsel düşürüldü, **1,929 val** görseli kaldı. Val yalnızca checkpoint seçimi ve sağlık kontrolü içindir, golden'a dokunmaz.
- **Train: 35,868 görsel.** Son kontrol olarak `assert_disjoint(train ∪ val, golden 928)` geçti.
- **Sınır (bilinçli):** eğitim kümesi, diskteki train-only görsellerin **ana korpus embedding'i olan alt kümesidir**, 38,084 / 64,369 ≈ **%59**. Kalan 26,285 görselin embedding'i yok, bu yüzden golden'a yakın kopya kontrolü bunlar için yapılamaz ve hepsi dışarıda bırakıldı. Bu görsellerin Colab'da embed edilip eklenmesi [ideas.md](../ideas.md)'de; R4 kapsamında değildir.
- `caption_equal` ile çıkan 164 görselin büyük çoğunluğu (eleme öncesi sayımda 177 görselden 158'i) kalıp caption `no medical information provided.` yüzünden düştü; bu adım kaynak kanıtından çok kalıp metin elemesi işlevi gördü. Elemeyi temkinli tarafta bıraktık.

### 2. Düşük bilgili caption'lar: eleme yok

Komut: `uv run python scripts/analyze_train_captions.py --manifest eval_data/r4_train_v0`. "Kısa" = normalize caption < 40 karakter.

| ölçüt | train | val |
|---|---|---|
| görsel / caption | 35,868 / 63,180 | 1,929 / 3,280 |
| kısa caption | 3,332 (%5.3) | 160 (%4.9) |
| en az bir caption'ı kısa görsel | 2,899 (%8.1) | 132 (%6.8) |
| tüm caption'ları kısa görsel | 547 (%1.5) | 35 (%1.8) |
| epoch 0 caption'ı kısa görsel | 1,445 (%4.0) | 82 (%4.3) |
| en sık 20 caption'dan birini taşıyan görsel | 77 | 37 |
| en sık caption'ın görsel sayısı | 10 | |

**Karar:** ek eleme yapılmaz. Gerekçe: oran küçük (%5.3 caption, %1.5 görsel). Aynı caption'ı taşıyan farklı görseller, kayıptaki **aynı-çift maskesi** sayesinde birbirine negatif sayılmaz (`train.loss.same_pair_mask`: aynı görsel kimliği veya aynı normalize caption). Bu yüzden tekrar eden kısa caption'lar yanlış negatif üretmez.

### 3. Ölçekler (iç içe)

`scale_rank` = görsel adının `sha1(f"42:{ad}")` sırası. Ölçekler bu sıranın önekleridir: **10k = 10,000 · 30k = 30,000 · full = 35,868**. 10k ⊂ 30k ⊂ full.

### 4. Eğitim ayarları (golden'a bakılarak ayarlanmadı)

`src/pathosearch/train/config.py` varsayılanları:
- LoRA r=16, α=32, dropout 0.1, hedef `q_proj` ve `v_proj`, iki kulede tüm katmanlar. Temel model tamamen dondurulmuş.
- Sıcaklık öğrenilebilir, temel `logit_scale`'den başlar, en çok 100.
- AdamW lr 1e-4, weight decay 0.1 (sıcaklıkta 0), %5 warmup + cosine, grad clip 1.0, batch 256.
- **10 epoch, tüm ölçeklerde aynı.**
- AMP: bf16 destekleniyorsa bf16, değilse fp16 + GradScaler.
- Caption seçimi: hash ile rotasyon (`caption_seed` 42, tüm koşularda aynı).
- **Checkpoint seçimi:** iç val kaybı en düşük epoch.
- Golden dev'de hiperparametre araması yapılmaz. Eğitim ıraksarsa düzeltme yalnızca iç val'e bakarak yapılır ve `run.json`'a gerekçesiyle yazılır.

### 5. Koşular ve seed'ler

- **Planlanan:** `lora-10k-s0`, `lora-30k-s0`, `lora-full-s0` (eğitim seed'i 0).
- **Zaman kalırsa:** `s1` ve `s2` (aynı üç ölçek, seed 1 ve 2). Koşulursa aynı protokolle raporlanır. Koşulmazsa R4 raporunda "seed varyansı ölçülmedi" sınırı olarak yazılır. s1/s2 sonuçları birincil karşılaştırmayı değiştirmez.

### 6. Değerlendirme

- golden_v0, k=10. Birincil eşik lenient (ADR-0001), strict ikincil. Paired bootstrap n=1000, seed 42, %95 GA.
- Systems: `clip, quilt, lora-10k-s0, lora-30k-s0, lora-full-s0` (koşulduysa s1/s2 de). Domain gate kapalı, rerank yok (rerank R5'in konusu).

### 7. Birincil karşılaştırma (tek)

**`lora-full-s0 − clip`, lenient nDCG@10, test split.** "Kazanç var" iddiası yalnızca bu farkın %95 GA'sı sıfırı dışlarsa yapılır. Diğer her şey **keşifsel**tir; çoklu karşılaştırma düzeltmesi uygulanmaz ve tek bir `*` kanıt sayılmaz:
- `lora-10k-s0 − clip`, `lora-30k-s0 − clip` (ölçek eğrisi, yalnızca nokta tahmin olarak anlatılır)
- `lora-full-s0 − quilt`, `quilt − clip`
- diğer metrikler ve strict eşik

### 8. Havuz genişletmesi (metriklerden önce)

Qrels clip/quilt (±rerank) sistemlerinden havuzlandı; LoRA'nın getirdiği yargılanmamış görseller ilgisiz sayılırsa sonuç LoRA aleyhine kayar.
1. Önce yalnızca sıralı listeler ve **judged@10** üretilir, metrik hesaplanmaz.
2. **Yalnızca `lora-*-s0` sistemlerinin** yargılanmamış (sorgu, görsel) çiftleri havuza eklenir. Annotator A bunları kör etiketler (`label_app.py`).
3. Bu işlem dev ve test için ayrı ayrı, ilgili eval koşmadan önce yapılır.
4. s1/s2 için genişletme yapılmaz, yalnızca judged@10 raporlanır.
5. Eklenen çift sayısı ve etiketleme süresi raporlanır.

### 9. Keşifsel duyarlılık

Aynı metrikler, golden görsellerinden eğitim kümesine (full) en yakın komşu cosine'i her iki encoder'da < 0.95 olan alt kümede de raporlanır. Bu bir kontrol, kanıt değildir.

### 10. Test split

`docs/r4_dev_findings.md` dev sonuçlarıyla yazılıp dondurulduktan sonra, **bir kez**, `--i-know-this-is-test` ile, 6. maddedeki systems listesiyle koşar.

## Sınırlar ve iddia edilemeyecekler

- **Eğitim kümesi %59:** diskteki train-only görsellerin yalnızca embedding'i olan kısmı. Bu kısım rastgele seçilmedi, ana korpusa girmiş görseller. Diskteki görseller de yalnızca `images_part_1`.
- **Ölçek ile adım sayısı karışık:** epoch sabit olduğu için ölçek farkı adım sayısı farkını da içerir. 30k ile full arasındaki fark küçük (30,000 vs 35,868, %84); bu iki nokta arasındaki ayrım zayıf olacaktır.
- **QuiltNet referansı belirsiz:** golden'daki 175 PubMed görselinin 123'ünün CSV'de `train`'de aynı figürden kardeşi var (STATUS §7.10). "LoRA, QuiltNet'e yaklaştı/geçti" genellenebilir bir iddia değildir.
- **8 + 10 sorgu:** aralıklar geniş. Ölçek eğrisinde monotonluk ya da "ölçekleme yasası" iddia edilmez.
- **Hiperparametreler ayarlanmadı:** sonuç "LoRA'nın en iyi hâli" değil, bu ayarla elde edilen sonuçtur.
- **0.99 altı benzerlik:** eleme 0.99'da yapıldı. Aynı kaynaktan olmayan ama benzer görseller eğitimde kalabilir (bkz. 9. madde).
- **Kaynak anahtarları dosya adından çıkarıldı:** pubmed UUID = figür varsayımı tahmindir; laion görsellerinin anahtarı yok.
- **Etiketler:** annotator'lar patolog değil, lenient kappa 0.62. Havuz genişletmesini yalnız A etiketler.
- **GPU'da bit düzeyinde tekrar üretilebilirlik garanti edilmez.** CPU'da resume'un bit düzeyinde aynı olduğu testle doğrulandı (`tests/test_train_loop.py`).
- Yalnızca metin→görsel ölçülür. Klinik geçerlilik iddiası yoktur.
