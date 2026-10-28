# PathoSearch: Durum Raporu (R0 keşfi)

Tarih: 2026-10-14 · Kapsam: `v0-prototype` etiketli kod (commit `d123df5` + `5e1c645`) · Yöntem: tüm `.py` dosyaları ve `reports/*.html` baştan okundu, kod çalıştırılmadı.

> Kural: Bu belgede "bulunamadı" yazan şey repoda yoktur. Tahmin ayrıca **(tahmin)** diye işaretlendi.

## Güncel özet

Aşağıdaki tablo belgenin sonraki bölümlerindeki bulguları tek yerde toplar; ayrıntı ve uyarılar ilgili bölümdedir. Durum değerleri: **doğrulandı** (repodaki kod/veriden doğrudan görüldü), **rapordan aktarıldı** (bitirme raporunda yazıyor, ham çıktı repoda yok, tekrar üretilemiyor), **tahmin** (çıkarım, doğrulanmadı), **bulunamadı** (repoda yok). Kaynak değerleri: `analyze_corpus` komutu (§7.8; tekrar üretilebilir), kod okuması, artifact okuması (komutsuz), rapor (§3 HTML raporları veya §7.9 bitirme raporu), makale, R4 komutları (`build_train_manifest`, `train_lora`, `eval_val_baseline`, `run_eval`; tekrar üretilebilir, ayrıntı ve komutlar [R4_report.md](R4_report.md)). R4 satırlarındaki sayılar raporlardan betikle aktarıldı; `*` = %95 GA sıfırı dışlıyor.

| Bulgu | Durum | Kaynak | Bölüm |
|---|---|---|---|
| Korpus 114,222 görsel; CLIP ve QuiltNet `ids.json` kümesi ve sırası aynı | doğrulandı | artifact okuması (komutsuz) | §7.1 |
| 114,222 görselin CSV'ye göre dağılımı: 114,186 saf `train` + 36 `train`/`val` karışık (eski "%100 train" JSONL'daki yazılan değere dayanıyordu) | doğrulandı | `analyze_corpus` komutu | §7.2, §7.7.2 |
| JSONL 168,739 satır / 114,222 görsel; 35,605 görselde birden fazla farklı caption | doğrulandı | `analyze_corpus` komutu | §7.3 |
| `unknown` oranı %21.89 (görsel bazında); eski "~%33" korpusla örtüşmüyor | doğrulandı | `analyze_corpus` komutu | §7.4, §4 risk 6 |
| Mevcut JSONL `balanced` modla üretilmiş (komut kaydı yok, yüksek güven) | tahmin | `analyze_corpus` karşılaştırma tablosu + kod okuması | §7.7.5 |
| `data/images` altında 65,322 görsel; `ids.json`'dakilerden 38,096 var, 76,126 eksik | doğrulandı | `analyze_corpus` komutu | §7.7.3 |
| Eksik görsellerin nedeni: part-2/3 dosyalarının başka ortamda kalması | tahmin | rapor | §7.9.4 |
| Çok-caption'lı görselde kullanılan caption: JSONL'daki son satır (seed'e bağlı) | doğrulandı | kod okuması | §7.6 (b) |
| P@10 relevance tanımı: `pathology ∈ expected` (kategori düzeyi, ikili) | doğrulandı | kod okuması | §4 |
| "0.539 → 0.609" = V4 → V5b; 0.539 rerank'siz QuiltNet, 0.609 V5b. **Güncelleme (2026-10-28):** golden_v0'da QuiltNet üzerinde rerank'in P@10 farkı dev'de de test'te de sıfırı içeriyor; ölçümler farklı olduğu için eski iddia ne doğrulandı ne çürütüldü ([R3 §3 (e)](r3_baseline.md#3-dev-bulgularının-testte-durumu)) | rapordan aktarıldı | bitirme raporu (ham çıktı yok) · güncel: `run_eval` komutu | §7.9.2, §4, [R3 §3](r3_baseline.md#3-dev-bulgularının-testte-durumu) |
| V1–V7 sürüm tablosu (P@10) | rapordan aktarıldı | bitirme raporu (ham çıktı yok) | §7.9.1 |
| `comparison_*.html` raporlarındaki P@10/mAP/nDCG değerleri | rapordan aktarıldı | `reports/*.html` | §3 |
| "64,052 görsel" ifadesinin kaynağı: V1-V3'ün 64K alt kümeyle koşulması olabilir | tahmin | rapor | §7.9.3, §3 |
| Dolaylı sızıntı: reranker caption sözcüklerine bakıyor, etiket aynı satırın kategorisi; γ=0 yalnız doğrudan sızıntıyı kapatır. **Güncelleme (2026-10-28):** risk eski kategori etiketli ölçüm için geçerli; golden_v0'da relevance insan derecesi, döngüsellik kontrolü (Spearman grade–caption örtüşmesi) R3 ve R4 koşularında uyarı eşiğinin altında ([R3 §5](r3_baseline.md#5-kanıt-düzeyi-ve-sınırlar), [R4 §4.5](R4_report.md#45-diğer-kontroller)) | tahmin (risk) · güncel: golden_v0'da ölçüldü, uyarı yok | kod okuması + rapor · `run_eval` komutu | §4 risk 1, 10 |
| QuiltNet bu görselleri eğitimde görmüş olabilir (korpus train); `val`'in eğitimde kullanılmadığı hiçbir kaynakta doğrulanmadı | tahmin (risk) | artifact okuması + makale | §7.2, §7.8.3, §4 risk 5, 9 |
| CSV `val` = QuiltNet retrieval holdout'u (13,559 sayısal örtüşmesi) | tahmin | makale | §7.8.3 |
| CSV split'i `quilt` videolarında ve openpath tweet'lerinde kaynak bazında (karışık grup 0); pubmed figürlerinde değil (1,343 karışık figür) | doğrulandı (anahtarlar dosya adından) | `analyze_source_groups` komutu | §7.10 |
| Golden'ın 37 videosu / 185 tweet'i hiçbir train görseliyle kaynak paylaşmıyor | doğrulandı | `analyze_source_groups` komutu | §7.10 |
| Golden'ın 175 pubmed görselinden 123'ünün CSV'de aynı figürden `train` kardeşi var; diskte 32 train adayı golden ile figür paylaşıyor | doğrulandı (figür = UUID varsayımı tahmin) | `analyze_source_groups` komutu | §7.10 |
| Yalnız-`val` 928 görsel, %50.75 `unknown`, ≥20 geçen 5 kategori | doğrulandı | `analyze_corpus` komutu | §7.8.1 |
| `stratified_sample` top-up hatası (tekrar eden satır) | doğrulandı | `tests/test_prepare_quilt.py` (`xfail(strict=True)`) | §7.7.6 |
| Domain gate: 12 çıpa, top-3 ortalama, eşik 0.80, 10 sorguluk öz-test; ROC/PR yok | doğrulandı | kod okuması | §2, §5 |
| Domain gate 10/10 ve 0.8015 vs 0.8026 kıl payı vakalar | rapordan aktarıldı | bitirme raporu | §7.9.5 |
| V6 cross-encoder CPU'da sorgu başına ~26 sn; demo V5b ~200 ms | rapordan aktarıldı | bitirme raporu | §7.9.5 |
| Görsel→Görsel modu nicel ölçülmemiş | rapordan aktarıldı | bitirme raporu | §7.9.5 |
| FAISS indeksi Flat (dosya boyutu düz depolamayla uyumlu) | tahmin | artifact okuması (indeks açılmadı) | §7.5 |
| `test_thresholds.py` ve `visualize_search.py` çalışmaz durumda. **Güncelleme (2026-10-28):** giderildi; `17541c5` ile `visualize_search` onarıldı (`tests/test_visualize_search.py`), `test_thresholds.py` silindi | doğrulandı (R0) · güncel: giderildi | kod okuması (çalıştırılmadı) · git geçmişi | §2 |
| Golden set, qrels, kaynak/video bazlı split, metrik testi, Recall@k/Hit@k/MRR, HNSW. **Güncelleme (2026-10-28):** HNSW dışında hepsi var. Golden set `eval_data/golden_v0` (protokol [golden_set.md](golden_set.md)); qrels `qrels.tsv` (adjudication sonrası); Recall@k/Hit@k/MRR/graded nDCG `eval/metrics.py`, test `tests/test_metrics.py`; ölçümler [R3](r3_baseline.md), [R4](R4_report.md). Golden'ın dev/test bölünmesi **sorgu** düzeyinde (`eval/splits.py`); golden ile R4 eğitim kümesinin ayrımı **kaynak grubu** (video/tweet/figür) bazında ([R4 §2](R4_report.md#2-eğitim-kümesi-sızıntı-kontrolleri-ölçekler), §7.10). HNSW hâlâ bulunamadı | bulunamadı (R0) · güncel: HNSW dışında doğrulandı | kod okuması · dosya listesi | §5, [R3](r3_baseline.md), [R4](R4_report.md) |
| "Literatürde eşdeğeri yok", "0.45-0.65 tipik aralık" iddiaları | bulunamadı (repoda doğrulama yok) | bitirme raporu | §7.9.6 (d), §4 risk 12 |
| R4 eğitim kümesi: 35,868 train + 1,929 iç val, golden'dan kaynak grubu / cos ≥ 0.99 / caption eşitliğiyle ayrık; adayların %59'ı (embedding'i olanlar) | doğrulandı | `build_train_manifest` komutu | [R4 §2](R4_report.md#2-eğitim-kümesi-sızıntı-kontrolleri-ölçekler) |
| R4 iç val kaybı 8.105 → 6.569 (10k) / 6.233 (30k); t2i R@5 0.036 → 0.066 / 0.090 (tek tohum) | doğrulandı | `train_lora` + `eval_val_baseline` komutları | [R4 §3](R4_report.md#3-eğitim-sonuçları) |
| Sıcaklık `forward` clamp hatası (başlangıçta gradyan 0) eğitimden önce düzeltildi (`10b7ab6`), regresyon testi var | doğrulandı | kod okuması + test | [R4 §3](R4_report.md#3-eğitim-sonuçları) |
| R4 birincil (önceden kayıtlı): `lora-30k-s0 − clip` lenient nDCG@10 test -0.102 [-0.253, +0.045], sıfırı içeriyor; hipotez desteklenmedi | doğrulandı | `run_eval` komutu (test, bir kez) | [R4 §4.3](R4_report.md#43-birincil-karşılaştırma-sonucu) |
| R4 keşifsel (test): `lora-10k-s0 − clip` nDCG@10 -0.209 [-0.337, -0.103]\*, `lora-30k-s0 − lora-10k-s0` +0.107 [+0.012, +0.203]\*; dev'de ikisi de sıfırı içeriyordu; düzeltmesiz | doğrulandı | `run_eval` komutu | [R4 §5](R4_report.md#5-dev-ve-test-tutarlılığı-lenient) |
| İç val kazancının golden'a yansımama nedeni (caption/sorgu dağılım farkı, aynı-dağılım iç val, unutma) | tahmin (test edilmedi) | – | [R4 §6](R4_report.md#6-neden-test-edilmemiş-hipotezler) |
| `lora-full-s0` koşulmadı; seed varyansı ölçülmedi; `r4_extension` (240 çift) yalnız A etiketledi | doğrulandı | ADR-0004 Ek 1, `pool.jsonl`, `A.jsonl` | [R4 §7](R4_report.md#7-sınırlar) |

## 0. Önce bilinmesi gerekenler

1. **İlk keşif oturumunda `artifacts/` ve `data/` erişilebilir değildi** (`.gitignore`'da; `E:\Multisearch_data` de erişilebilir değildi). Embedding, FAISS indeksi, `captions_quilt.jsonl`, `eval_queries*.json` ve `benchmark_*.json` dosyalarının hiçbiri incelenemedi. Hiçbir metrik o oturumda yeniden üretilemedi; aşağıdaki sayılar sadece `reports/*.html` içinde yazanlardır. **Güncelleme:** `artifacts/` ve `data/` sonradan okundu (§7); bu madde yalnızca ilk keşif oturumunun durumunu anlatır.
2. **"P@10 0.539 → 0.609" iddiası doğrulanamıyor.** Ayrıntı §4. **Güncelleme:** bitirme raporuna göre bu V4 → V5b geçişidir (§7.9.2); kaynak rapor, ham çıktı repoda yok, bu yüzden iddia hâlâ repodan tekrar üretilemiyor.
3. `scripts/` içinde iki dosya **çalışmaz durumda** (§2, "Kırık").

## 1. Repo haritası

| Yol | Rol |
|---|---|
| `app.py` (408 satır) | Streamlit UI + `search_pipeline` + artifact kontrolü (mantık UI'ya gömülü) |
| `scripts/` | 18 dosya, ayrıntı §2 |
| `inspect_quilt.py` | Tek seferlik keşif; sabit yollar `C:\Users\user\Downloads\...` (kullanıcı dizinine bağlı mutlak yol) |
| `reports/` | 6 statik HTML (üçü karşılaştırma, üçü tek sorgu görseli). **Not:** `comparison_v1_v2_v3.html` dışındakiler `data:image` ile gömülü Quilt-1M görselleri içerdiği için yerelde üretilir, repoda yok (bkz. §3 notu) |
| `requirements.txt` | Sürümler `>=` ile açık, lock yok |
| `.githooks/commit-msg`, `.claude/settings.json` | Atıf satırlarını temizleme |
| `tests/`, `pyproject.toml`, CI, `uv.lock`, `LICENSE` | **Bulunamadı** |

## 2. Script'ler: görev, girdi/çıktı, metrik

| Script | Görev | Girdi → Çıktı | Metrik? |
|---|---|---|---|
| `00_prepare_quilt.py` | Quilt-1M CSV'sinden alt küme seç, görselleri çıkar/hard-link yap | `quilt_1M_lookup.csv` + zip'ler veya klasörler → `E:\Multisearch_data\images`, `captions_quilt.jsonl` (`path, caption, pathology, split`), `subset_stats.json` | Yok. Filtre: `not_histology==0`, caption dolu, `--split` (**varsayılan `train`**). `pathology` = listenin **ilk** elemanı (`parse_pathology`), boşsa `unknown`. İki örnekleme: `stratified` (varsayılan), `balanced` (`BALANCED_TARGETS`). Seed 42. Sabit `E:\` varsayılan yolları var |
| `01_build_embeddings.py` | Görselleri encoder ile embed et | `--img_dir data/images` (`--captions_jsonl` ile filtre) → `<out_dir>/embeddings.npy`, `ids.json`, `model_info.json` | Yok. `ids` = dosya adı (basename). `sys.path` hack'i var |
| `encoders.py` | `BaseEncoder`, `CLIPEncoder` (HF, `openai/clip-vit-base-patch32`), `OpenCLIPEncoder` (`wisdomik/QuiltNet-B-32`), registry/factory | – | Yok. Çıktılar L2-normalize |
| `faiss_search.py` | `build` (Flat IP/L2), `search`, `bench` | `embeddings.npy` → `faiss_index.bin`, `faiss_meta.json`, `bench_report.json` | **Gecikme** (`bench`: p50/p95/p99, rastgele *indeks vektörleri* sorgu olarak, sadece FAISS araması, encode yok). Yalnız Flat; HNSW **bulunamadı** |
| `compare_models.py` | CLIP vs QuiltNet benchmark + HTML rapor | `eval_queries*.json`, iki indeks, captions → `artifacts/benchmark_*.json`, `reports/*.html` | **Precision@K, mAP@K, nDCG@K, gecikme burada.** Ayrıca RRF çoklu-sorgu, `unknown` filtresi, Jaccard rerank, cross-encoder rerank seçenekleri |
| `make_queries.py` / `_v2.py` / `_ensemble.py` | Değerlendirme sorgusu üret (V1: 63 etiketli+3 genel, V2: 90+3, V3: 56 kavram×3 varyant) | → `artifacts/eval_queries*.json` | Yok. Etiket = elle yazılmış `expected` kategori listesi |
| `make_comparison_report.py` | V1/V2/V3 `benchmark_*.json`'larını tek HTML'de birleştir | `artifacts/benchmark_*.json` → `reports/comparison_v1_v2_v3.html` | Sadece sunum |
| `reranker.py` | Ağırlıklı Jaccard: `α·cosine + β·caption_overlap + γ·pathology_bonus` | aday listesi → yeniden sıralı liste | Yok. Varsayılan `α=.75, β=.25, γ=0`. "V5b" adı sadece `app.py` ve `test_pipeline.py` içinde; sürüm geçmişi/kayıt yok |
| `cross_encoder_reranker.py` | BiomedCLIP **text–text** cosine (gerçek cross-encoder değil, bi-encoder) veya ms-marco | aday listesi → `ce_score` | Yok. Caption ile sorguyu karşılaştırıyor |
| `domain_gate.py` | CLIP text kulesiyle 12 çıpa cümleye cosine, en yakın 3'ün ortalaması, eşik 0.80 | sorgu → skor | Küçük öz-test (10 sorgu), ROC yok |
| `thresholds.py` | Gate kararını `ok / domain_out / empty` durumuna paketler | – | Yok |
| `test_domain_anchors.py` | Çıpa fikrinin 10 sorguda ayrışıp ayrışmadığını dener | – | Konsol çıktısı (kayıt yok) |
| `test_pipeline.py` | `app.search_pipeline`'ın başsız kopyası, 5 vaka, early-exit assert | indeksler gerekir | pytest değil, elle çalışan script |
| `test_thresholds.py` | Eski cosine eşiği kalibrasyonu | – | **Kırık**: `thresholds.apply_thresholds` yok (silinmiş) |
| `visualize_search.py` | Tek sorgu için CLIP vs QuiltNet HTML | → `reports/search_*.html` | **Kırık**: `faiss_search._load_index/_load_embeddings` yok (adlar `load_index/load_embeddings`). Raporlar eski kodla üretilmiş |
| `shot.py` | Playwright ile ekran görüntüsü | çalışan :8501 → `artifacts/shots` | Yok. `playwright` requirements'ta yok |

Hardcode/hack özeti: `os.chdir` (app, `domain_gate`, `test_*`), `sys.path.insert` (app, 6+ script), mutlak `E:\` ve `C:\Users\user\...` yolları, `data/images` → `data/images_test` ve `captions_quilt` → `captions_quilt_test` **sessiz fallback**'i (`app.py:20-32`).

## 3. Raporların iddiaları

> **Not (2026-10-26):** `comparison_64k.html`, `comparison_114k.html` ve `search_*.html` ×3 yerelde üretilir, repoda yok: `data:image` ile gömülü Quilt-1M görselleri içerdikleri için git takibinden çıkarıldı (`.gitignore`: `reports/*.html`). Repoda yalnız görsel içermeyen `comparison_v1_v2_v3.html` kalır. Aşağıdaki sayılar bu dosyaların takipten çıkarılmadan önceki içeriğinden okunmuştur; dosyalar git geçmişinde hâlâ duruyor.

| Rapor | Ne iddia ediyor | Notlar |
|---|---|---|
| `comparison_64k.html` (2026-05-22) | 63 etiketli sorgu, K=10: P@10 CLIP **0.175** vs QuiltNet **0.395**; mAP 0.272/0.520; nDCG 0.172/0.410; gecikme 32.7/62.5 ms (CPU) | = V1 (rerank yok, unknown filtresi yok) |
| `comparison_114k.html` (2026-06-04) | 56 etiketli sorgu, K=10: P@10 CLIP **0.323** vs QuiltNet **0.539**; mAP 0.440/0.633; nDCG 0.326/0.547; gecikme 106.3/181.9 ms | **Dosya adı "114k" ama içindeki korpus "64,052 görsel"**; `compare_models.py` ise sabit olarak "114K image corpus" yazıyor. Rapor ile kod senkron değil. **Açık soru:** "64,052" sayısının kaynağı bulunamadı (§7.1: gerçek korpus 114,222). Sorgu adları (`skin squamous cancer`, `q003`) V3 ensemble'ın genel varyantı. Başlıkta rerank etiketi yok → rerank **kapalı** koşmuş görünüyor |
| `comparison_v1_v2_v3.html` (2026-05-22) | QuiltNet P@10: V1 0.395 → V2 0.457 → V3 0.480; CLIP: 0.175 → 0.216 → 0.230 | Raporun kendisi, sorgu setleri farklı olduğu için değerlerin doğrudan karşılaştırılamayacağını yazıyor |
| `search_*.html` ×3 (2026-04-29) | Tek sorgu CLIP vs QuiltNet görsel çıktı; iddia yok | `lymph node` için QuiltNet'in 5 sonucundan 3'ü `unknown`/alakasız görünüyor |

Tutarsızlık: V3 JSON'unda QuiltNet 0.480 (56 kavram), `comparison_114k`'de 0.539 (56 sorgu). Aynı sorgu sayısı, farklı sayı; hangi konfigürasyonla (filtre/ensemble/pool) çalıştığı raporda **kayıtlı değil**.

## 4. "Precision@10 0.539 → 0.609" iddiası

**Kaynak**
- Tek geçtiği yer: `app.py:235` (sidebar `help` metni). Kodda, JSON'da, raporda **0.609 hiç geçmiyor**. 0.609'un nasıl hesaplandığı **bulunamadı**. (Güncelleme: bitirme raporuna göre 0.609 V5b'dir, bkz. §7.9.2; ham çıktı ve komut repoda yok.)
- **0.539**, `reports/comparison_114k.html`'de (yerelde üretilir, repoda yok; §3 notu) **QuiltNet'in rerank'siz P@10'u** olarak geçiyor. Yani "0.539" muhtemelen ham QuiltNet taban çizgisi, "0.609" ise rerank'li sonuç **(tahmin)**; rerank'li koşunun raporu/çıktısı repoda yok.
- `compare_models.py` rerank varsayılanı `α=.6, β=.4, pool=50`; uygulama `α=.75, β=.25, pool=topk×5`. Rapordaki/ iddiadaki sayının hangi ağırlıklarla çıktığı **bulunamadı**; uygulamadaki ayar bu sayıyla bağlanamıyor.
- Komut, seed, config yok → CLAUDE.md'nin "tekrar üretilebilir komut + config + seed" kuralını karşılamıyor.

**Relevance etiketi nasıl tanımlı** (`compare_models.py:100-131`)
- Bir sonuç, `result["pathology"] ∈ query["expected"]` ise ilgili sayılır (ikili).
- `pathology` = Quilt-1M CSV'sindeki `pathology` listesinin ilk elemanı (`00_prepare_quilt.py:83`), 17 geniş uzmanlık kategorisinden biri (Dermatopathology, Pulmonary, ...). Yani "ilgili" = **aynı uzmanlık dalı**, aynı hastalık değil.
- `expected` = sorguyu yazan kişinin elle verdiği kategori. Bağımsız insan etiketi yok.

**Döngüsellik / geçerlilik riskleri**

| # | Risk | Kanıt | Şiddet |
|---|---|---|---|
| 1 | **Rerank ve etiket aynı kaynaktan.** Rerank, sorgu ile `caption` token örtüşmesine bakıyor; etiket de aynı Quilt satırının metadatası. Caption'daki "skin", "lung" gibi sözcükler doğrudan kategoriyle korele; rerank etiketi görmese de **etiket vekilini** (caption sözcükleri) kullanıyor. `γ=0` sadece doğrudan sızıntıyı kapatır | `reranker.py:166-170`, `compare_models.py:100` | Yüksek |
| 2 | **Rerank fiilen sözcüksel yeniden sıralama.** Havuzda cosine ~0.32 ve neredeyse düz (kodun kendi notu), `α·cos≈0.24`; Jaccard ∈[0,1] ile `β·J` en fazla 0.25. Sıralamayı Jaccard belirliyor | `thresholds.py` docstring, `reranker.py` | Orta (yorum) |
| 3 | **Tek set üzerinde ayar, ayrı split yok.** V1→V2'de zayıf kategorilerin sorguları sonuçlara bakılarak yeniden yazıldı ("Weak categories fully rewritten"); α/β de aynı sorgularda. Dev/test ayrımı yok | `make_queries_v2.py:1-8` | Yüksek |
| 4 | **Değerlendirmede `unknown` sonuçları etiketle atılıyor** (`--filter_unknown`): retrieval aşamasında etiket kullanılıyor. Uygulama bunu yapmıyor → rapor ≠ uygulama davranışı | `compare_models.py:42-57` | Orta |
| 5 | **Eğitim/test sızıntısı (QuiltNet adil mi?).** `00_prepare` varsayılan split `train`; QuiltNet Quilt-1M üzerinde eğitildi. Hangi split'in gerçekten kullanıldığı **bulunamadı** (artifact yok). Eğer `train` ise QuiltNet kendi eğitim verisinde ölçülmüş olur (güncelleme: korpusun `train` olduğu §7.2 ve §7.7.2'de doğrulandı) | `00_prepare_quilt.py:48` | Yüksek (doğrulanmalı) |
| 6 | **Etiket gürültüsü ve kaba ilgi.** Etiket otomatik/metadata kaynaklı; ilk eleman alınıyor; `unknown` korpusta **%21.89** (görsel bazında, §7.4; eski "~%33" rapor notu korpusla örtüşmüyordu). Kategori düzeyi P@10 hastalık düzeyi doğruluğu değil | `make_comparison_report.py` metodoloji | Orta |
| 7 | **nDCG'nin IDCG'si her kategoride ≥K ilgili belge olduğunu varsayıyor**; küçük kategorilerde (Ophthalmic 1 sorgu) şişirir/saptırır | `compare_models.py:130`, rapor dipnotu | Düşük-Orta |
| 8 | **İstatistiksel güç düşük**: kategori başına 1-10 sorgu, güven aralığı/seed yok, kategori ağırlığı eşit değil (makro yerine sorgu ortalaması) | rapor tabloları | Orta |
| 9 | **Korpus tamamen `train`; CLIP-QuiltNet farkı "yalnızca alan ön-eğitimi" diye yorumlanamaz.** Bitirme raporu %50.6'lık farkı alan ön-eğitimine yüklüyor, ama QuiltNet bu görselleri eğitimde görmüş olabilir | §7.2, §7.7.2, §7.9 | Yüksek |
| 10 | **Dolaylı sızıntı: rapor γ=0'ı yeterli sayıyor.** Reranker caption sözcüklerine bakıyor, relevance etiketi aynı satırın kategorisinden geliyor. γ=0 yalnızca doğrudan sızıntıyı kapatır; caption sözcükleri ile kategori arasındaki korelasyon durur. V5b'nin +0.070 kazancı (0.539 → 0.609) bu yüzden şişkin olabilir (risk 1'in raporla da desteklenen hâli) | §7.9, `reranker.py` | Yüksek |
| 11 | **Sorgular sonuçlara bakılarak yeniden yazılmış (V2), ayar-test ayrımı yok.** Sürümler arası artış (V1 → V2 → V3) kısmen sorgu ayarından gelebilir (risk 3 ile aynı kök) | §7.9, `make_queries_v2.py:1-8` | Yüksek |
| 12 | **Doğrulanmamış dış iddialar.** Raporda "literatürde eşdeğeri yok" gibi özgünlük iddiaları ve "0.45-0.65 tipik aralık" karşılaştırması var; bu repoda doğrulanmadı | §7.9 | Düşük-Orta |

**Sonuç:** İddia, şu haliyle bağımsız ölçüm sayılamaz. 0.539 rerank'siz QuiltNet ölçümü olarak raporda var, 0.609 repodaki hiçbir dosyada yok (bitirme raporuna göre V5b, bkz. §7.9.2; ham çıktı yok). Rerank kazancı büyük olasılıkla (1)'in etkisiyle şişkin. Arayüzdeki metin R2'ye kadar kaldırılmalı ya da "doğrulanmadı" denmeli (karar açık, §8-3).

## 5. PLAN.md §1 tablosundaki ❓ satırları

| Satır | Sonuç |
|---|---|
| `scripts/`, `reports/` içeriği | Okundu (§2, §3) |
| Golden set / qrels / split | **Bulunamadı.** Sadece elle yazılmış kategori-sorgu çiftleri var; bağımsız golden set, qrels ve kaynak/video bazlı split yok |
| Metrik modülü + test | Metrikler (`precision_at_k`, `average_precision_at_k`, `ndcg_at_k`) `compare_models.py` içinde, **test yok**. Recall@k, Hit@k, MRR **bulunamadı** |
| p50/p95 raporu | Sadece `faiss_search.py bench` (yalnız FAISS, encode hariç). Rapor çıktısı repoda yok |
| HNSW/ANN | **Bulunamadı** (yalnız `IndexFlatIP/L2`) |
| Gate kalibrasyonu | 10 elle seçilmiş sorgu; ROC/PR yok. Eşik 0.80 `domain_gate.py` ve app'te sabit |

## 6. Diğer bulgular (R1 için girdi)

- `app.py`, `search_pipeline` ve `_raw_faiss` ile `test_pipeline.py`'daki kopya ayrışabilir (kopya kod).
- `faiss_search.search` sorguyu `ip` için normalize ediyor, `app._raw_faiss` normalize etmiyor (encoder zaten normalize ettiği için sonuç aynı olmalı **(tahmin)**).
- `compare_models.py` ve `make_comparison_report.py` çıktıları `artifacts/benchmark_*.json`'a yazıyor; bunlar `.gitignore`'da, yani sonuçların ham kaynağı repoya girmiyor.
- `README` "scripts/ ... reports/" diyor ama `requirements.txt`'te `playwright`, `sentence-transformers` (ms-marco fallback) yok.
- Türkçe/İngilizce karışık yorum ve UI metni; klinik teşhis ibaresi yok, "eğitim/araştırma" uyarısı app'te yok (README'de var).

## 7. Artifact incelemesi (2026-10-14)

Yöntem: `data/captions_quilt.jsonl` ve `artifacts/{clip,quilt}/*/` dosyaları **sadece okundu**, hiçbiri değiştirilmedi. Geçici analiz betikleri repo dışında tutuldu. Görsel sayısı kontrolü bu bölümde yapılmadı (zip'ler henüz çıkarılmamıştı; diskteki görseller sonradan §7.7.3'te sayıldı). Bu bölüm §0 madde 1'i (artifact yok) ve §4'teki 2. sorusunu yanıtlar.

### 7.1 Korpus boyutu: CLIP ve QuiltNet aynı mı?

**Evet, birebir aynı.**

| Kontrol | CLIP | QuiltNet |
|---|---|---|
| `ids.json` uzunluğu | 114,222 | 114,222 |
| Tekil id | 114,222 | 114,222 |
| `embeddings.npy` şekli (başlıktan) | (114222, 512) float32 | (114222, 512) float32 |
| `faiss_meta.json` `n` | 114,222 | 114,222 |

İki `ids.json` kümesi aynı ve **sıraları da aynı** (fark 0). Sonuç: korpus **114k**; `comparison_114k.html` dosya adı doğru, ama içindeki "64,052 görsel" ifadesi bu korpusla uyuşmuyor, nedeni **bulunamadı** (§3'teki tutarsızlık sürüyor). Bitirme raporuna göre olası açıklama: V1-V3 64K alt kümeyle, V4 ve sonrası 114,222 görselle koşulmuş (§7.9.3, tahmin).

### 7.2 Split dağılımı (`ids.json`'da olan satırlar)

`ids.json`'daki 114,222 id'nin **tamamı** captions'ta bulundu (eksik 0). Eşleştirme anahtarı `path`'in dosya adı (basename).

| split | satır | oran |
|---|---|---|
| `train` | 168,739 | 100% |
| `test` / `val` / diğer | 0 | 0% |

> **Düzeltme (§7.7.2):** Aşağıdaki "%100 `train`" JSONL'daki yazılan `split` değerine dayanır. CSV'ye göre 114,222 görselin 114,186'sı saf `train`, 36'sı `train`/`val` karışıktır. Bulgu silinmedi, bu not ile birlikte okunmalıdır.

Korpus **tamamen `train`** (JSONL değerine göre; yukarıdaki düzeltmeye bakın). Dosyanın tamamı da `train`; yani dışarıda bırakılmış test satırı yok. §4 risk 5 ("QuiltNet kendi eğitim verisinde ölçülmüş olabilir") artık **sadece varsayım değil, veriyle destekli**: korpus `train` etiketli. Tek belirsizlik: `split` alanının Quilt-1M CSV'sinden mi geldiği kodda doğrulanmalı, ve QuiltNet'in eğitim kümesiyle tam örtüşme derecesi **bulunamadı** (kontrol edilmedi).

### 7.3 Caption satırı ≠ görsel sayısı

`captions_quilt.jsonl` 168,739 satır, ama tekil `path` (ve basename) sayısı **114,222**. Fark, aynı görselin birden fazla caption'a sahip olması (Quilt-1M'de bir görsel için birden çok metin var):

| Satır/görsel | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Görsel sayısı | 78,617 | 22,327 | 9,131 | 3,044 | 828 | 199 | 45 | 29 | 2 |

- 35,605 görselin birden fazla **farklı** caption'ı var. Aynı görselin satırları arasında `pathology` farkı **0** (etiket görsel başına tutarlı).
- Tam yinelenen (path+caption+pathology) satır yok.
- **Etki (doğrulanmadı):** Reranker ve CE caption kullanıyor; çok-caption'lı görselde hangisinin seçildiği kodda net değil (`src/` içinde captions yükleyen bir yer bulunamadı, grep sonucu `config.py` dışında yok). Birleştirme/ilk satır/son satır hangisi ise Jaccard skoru buna bağlı; **R2 öncesi incelenmeli** (güncelleme: kod okunarak çözüldü, bkz. §7.6 (b)).
- `config.py` captions varsayılanı `<artifacts>/captions_quilt.jsonl` idi, dosya `data/` altında. **Düzeltildi** (`f14bb9b`): varsayılan artık `<data>/captions_quilt.jsonl`.
- Hangi caption'ın kullanıldığı kod okunarak çözüldü: §7.6 (b).

### 7.4 Pathology dağılımı ve `unknown` oranı

Doğru birim **görsel** (satır değil). Satır bazlı sayım çok-caption'lı görselleri fazla sayar ve `unknown`'u yanlış (%14.82) gösterir.

| pathology | görsel | oran |
|---|---|---|
| unknown | 25,000 | 21.89% |
| Dermatopathology | 17,248 | 15.10% |
| Gastrointestinal | 11,669 | 10.22% |
| Pulmonary | 8,312 | 7.28% |
| Gynecologic | 7,578 | 6.63% |
| Soft tissue | 7,454 | 6.53% |
| Hematopathology | 5,370 | 4.70% |
| Genitourinary | 5,093 | 4.46% |
| Renal | 4,371 | 3.83% |
| Bone | 4,015 | 3.52% |
| Breast | 3,502 | 3.07% |
| Neuropathology | 3,492 | 3.06% |
| Endocrine | 3,046 | 2.67% |
| Head and Neck | 2,874 | 2.52% |
| Cardiac | 1,759 | 1.54% |
| Ophthalmic | 1,152 | 1.01% |
| Hepatopathology | 998 | 0.87% |
| Cytopathology | 975 | 0.85% |
| Pediatric | 311 | 0.27% |
| `Unknown` (büyük U) | 3 | 0.00% |

- 20 ayrı değer: 18 uzmanlık + `unknown` + 3 görsellik `Unknown` (büyük/küçük harf tutarsızlığı; `unknown` filtresi büyük U'yu kaçırır).
- `unknown` oranı **%21.89** görsel bazında. §4 risk 6'da "~%33 (rapor notu)" yazıyordu (düzeltildi); bu değer korpusla örtüşmüyor, kaynağı (muhtemelen sonuç listesindeki oran, korpus oranı değil) **doğrulanmadı**.
- Satır bazında sayılar yuvarlak (unknown 25,000, Dermatopathology 20,000, Gastrointestinal 18,000...). Bu, `00_prepare_quilt.py` örneklemesinde kategori başına **satır** kotası uygulandığını düşündürüyor **(tahmin, kod okunarak doğrulanmalı)**. Bu doğruysa dağılım Quilt-1M'in doğal dağılımı değil, örneklemenin ürünü.
- Çok sınıflı dengesizlik: Ophthalmic, Hepatopathology, Cytopathology, Pediatric < %1.1. Bu kategorilerde sorgu başına anlamlı P@10 için yeterli ilgili belge olmayabilir (§4 risk 7 ile ilişkili).

### 7.5 `model_info.json` ve `faiss_meta.json`

| Alan | CLIP | QuiltNet |
|---|---|---|
| `name` | `clip:openai/clip-vit-base-patch32` | `openclip:hf-hub:wisdomik/QuiltNet-B-32` |
| `embed_dim` | 512 | 512 |
| `device` | cuda | cuda |
| `n_images` | 114,222 | 114,222 |
| `batch_size` | 64 | 64 |
| `duration_seconds` | 3017.03 | 3122.96 |
| `creation_time` | 2026-06-03T08:52:00 | 2026-06-03T13:01:17 |

`faiss_meta.json` iki modelde de: `{"metric": "ip", "n": 114222, "d": 512}`. İndeks tipi (Flat/HNSW) bu dosyada **yazmıyor**; `faiss_index.bin` boyutu (233,926,701 B ≈ 114,222×512×4 + başlık) düz depolamayla uyumlu, yani **Flat** olması muhtemel **(tahmin, indeks açılıp doğrulanmadı)**.

`model_info.json`'da model sürümü/commit hash'i, `open_clip`/`transformers` sürümü, seed ve komut **yok**; embedding üretimi tekrar üretilebilirlik kuralını karşılamıyor.

### 7.6 Kod okuyarak yanıtlanan sorular (2026-10-14)

Yöntem: sadece kod okundu, artifact/CSV açılmadı, hiçbir şey çalıştırılmadı. Satır numaraları bu tarihteki `src/pathosearch/` hâli.

**(a) `split` nereden geliyor?** CSV sütunundan.
- `prepare_quilt.py` `NEEDED_COLS` içinde `split` var ve `pd.read_csv(usecols=NEEDED_COLS)` ile okunuyor. Filtre: `chunk["split"] == split` (`collect_candidates`). `--split` varsayılanı `train`, seçenekler `train/val/test`.
- JSONL'daki `split` alanı satırın kendi değeri **değil**; `extract_and_write` her satıra `args.split` değerini yazıyor. Filtre doğru çalıştıysa ikisi aynıdır.
- **Bulunamadı:** CSV'deki gerçek değerlerin (`train/val/test` yazımı, dağılımı) doğrulanması; CSV bu incelemede açılmadı. §7.2'deki "%100 train" bulgusu bu yüzden JSONL'daki yazılan değere dayanıyor, CSV'ye değil.

**(b) Çok-caption'lı görselde hangi caption kullanılıyor?** Her yerde tek caption: **JSONL'daki o görselin son satırı.**
- `faiss_search.load_captions` `{basename: {caption, pathology}}` sözlüğü kuruyor; aynı basename tekrar gelince atama üzerine yazıyor (`cap_map[fname] = ...`). Birleştirme yok, ilk satır korunmuyor.
- Aynı sözlük `raw_search` ile her sonuca `caption`/`pathology` olarak ekleniyor. Reranker (`rerank_results`), cross-encoder (`c.get("caption")`), uygulama (`pipeline.SearchIndex.captions`) ve `compare_models` hep bu sonuç alanını okuyor. Başka bir caption yükleyen yer **bulunamadı**.
- "Son satır" anlamsal bir seçim değil: `prepare_quilt` örneklemeden sonra satırları `sample(frac=1, random_state=seed)` ile karıştırıyor, yani hangi caption'ın kazandığı JSONL sırasına (seed'e) bağlı. Jaccard skoru bu rastgele seçime bağlı.
- JSONL'da bir görselin tüm Quilt caption'ları da olmayabilir: örnekleme **satır** bazlı (bkz. c), bir görselin yalnız örneklenen satırları dosyaya girer.
- Not: klasör modunda ikinci kez gelen satır `dst.exists()` olduğu için görsel kopyalamayı atlıyor (`skip`) ama yine de JSONL'a yazılıyor; çok-caption'lı görsellerin JSONL'da birden fazla satırı olmasının kod tarafındaki nedeni bu.

**(c) Kategori başına satır kotası var mı?** Evet, iki modda da, birim **satır** (görsel değil).
- `stratified` (varsayılan): kategori başına `quota = max(1, round(n * count / total))`, `count/total` satır sayısı. Orantılı; `--n` toplam satır hedefi.
- `balanced`: `BALANCED_TARGETS` kategori başına en fazla `min(hedef, havuz)` satır alır; `--n` yok sayılır. Kategori başına yuvarlak tavanlar (`unknown` 25000, `dermatopathology` 20000, ...) buradan geliyor.
- Mevcut `captions_quilt.jsonl` hangi modla üretildi: **bulunamadı** (komut ve `subset_stats.json` incelenmedi; güncelleme: satır sayıları `BALANCED_TARGETS` ile karşılaştırılarak `balanced` sonucuna varıldı, bkz. §7.7.5). §7.4'teki yuvarlak satır sayıları `balanced` ile tutarlı **(tahmin)**; `artifacts/subset_stats.json` (`category_distribution`) bunu doğrular.
- Sonuç: `balanced` ise dağılım Quilt-1M'in doğal dağılımı değil, hedef tablosunun ürünü; kategori oranlarından korpus temsiliyeti çıkarılamaz.
- Okurken görülen, doğrulanmamış şüphe: `stratified_sample` içinde `remaining = df[~df.index.isin(result.index)]` satırında `result` `ignore_index=True` ile yeniden indekslenmiş; indeksler `df` ile eşleşmediği için doldurma (`len(result) < n`) yanlış satırları dışlayıp tekrar eden satır ekleyebilir. Çalıştırılıp test edilmedi. (Güncelleme: test ile doğrulandı, bkz. §7.7.6.)

**Bu bölümle birlikte yapılan kod değişiklikleri**
- `f14bb9b`: captions varsayılan yolu `data/captions_quilt.jsonl`.
- Pathology normalizasyonu (`labels.normalize_pathology`: kırp + küçük harf): `load_captions`, `parse_pathology`, `is_relevant` ve reranker bonusunda uygulanıyor, böylece `Unknown`/`unknown` tek kategori. `BALANCED_TARGETS` anahtarları küçük harfe çevrildi ve ayrı `"Unknown": 4000` girdisi kaldırıldı (normalizasyondan sonra `unknown` ile çakışırdı). Etkisi: arayüzde ve raporlarda kategori adları küçük harfle görünür (ör. `dermatopathology`); bellekte, mevcut JSONL'u yeniden üretmeden de uygulanır çünkü normalizasyon yükleme sırasında yapılıyor.

### 7.7 CSV, disk ve JSONL çapraz kontrolü (2026-10-14)

Yöntem: `data/quilt_1M_lookup.csv` (chunk'lı okuma), `data/captions_quilt.jsonl`, `artifacts/clip/ids.json` ve `data/images` (yalnız dosya adı sayımı) **salt okunur** incelendi; hiçbir veri dosyası değiştirilmedi. Eşleştirme anahtarı dosya adı (CSV `image_path` = basename). Analiz betiği repo dışında tutuldu, bu yüzden bu sayılar henüz repoda tekrar üretilebilir bir komuta bağlı değil. Not: 7.7.5 ve 7.7.6 sonuçları aşağıda.

#### 7.7.1 CSV `split` sütunu

CSV: 1,017,708 satır, 653,209 tekil `image_path`. `split` yalnızca iki değer alıyor (NaN yok):

| split | satır | tekil görsel |
|---|---|---|
| `train` | 1,004,149 | 643,522 |
| `val` | 13,559 | 9,906 |

- Toplam tekil (653,209) iki satırın toplamından (653,428) küçük: **219 görsel hem `train` hem `val` satırında** geçiyor.
- `prepare_quilt` filtresi (`not_histology==0`, caption dolu) sonrası: `train` 999,198 satır / 640,098 görsel, `val` 13,559 satır / 9,906 görsel.

#### 7.7.2 `ids.json`'daki 114,222 görselin CSV split'i

114,222 id'nin tamamı CSV'de bulundu (eksik 0, tekrar 0).

| CSV'deki split kümesi | görsel |
|---|---|
| yalnız `train` | 114,186 |
| `train` + `val` (ikisinde de satır var) | **36** |
| yalnız `val` | 0 |

**Hepsi "gerçekten train" değil:** 36 görselin CSV'de `val` satırı da var. Yani 7.2'deki "%100 train" ifadesi (JSONL'daki yazılan değere dayanıyordu) CSV'ye göre **114,186 saf train + 36 train/val karışık** olarak düzeltilmeli. Etki küçük (%0.03) ama held-out değerlendirmede bu 36 görsel dışlanmalı.

#### 7.7.3 `data/images` diskte ne var?

Görseller tek alt klasörde (`data/images/images_part_1`); tekil dosya adı sayısı **65,322**.

| Kontrol | Sayı |
|---|---|
| Diskte, `ids.json`'da **var** | 38,096 |
| Diskte, `ids.json`'da **yok** | 27,226 |
| `ids.json`'da olup diskte **eksik** | **76,126** (114,222'nin %66.6'sı) |
| Diskte olup CSV'de olmayan | 0 |

Diskteki 65,322 görselin CSV split'i: yalnız `train` 64,369 · yalnız `val` 928 · `train`+`val` 25.
Diskte olup `ids.json`'da olmayan 27,226 görselin split'i: yalnız `train` 26,285 · yalnız `val` 928 · `train`+`val` 13.

**Sonuç:** `ids.json`/embedding/indeks 114,222 görseli kapsıyor ama diskte bunların yalnızca 38,096'sı var. Arayüzde sonuç görseli gösterimi ve görsel→görsel mod diskte olmayan görseller için çalışamaz. Bunun nedeni (görsellerin sonradan taşınması/silinmesi, yalnız `images_part_1`'in çıkarılmış olması) **bulunamadı**; diskte kaç `images_part_*.zip` olduğu bu incelemede kontrol edilmedi.

#### 7.7.4 CSV'de `train` dışı görseller

- `val` satırı: 13,559; tekil görsel: 9,906. Bunlardan 219'u aynı zamanda `train`'de de var, yani **yalnız-`val` görsel: 9,687**.
- Diskte `val` görseli: 953 (yalnız-`val` 928 + `train`+`val` 25). Bu 928 yalnız-`val` görselin **hiçbiri `ids.json`'da yok**, yani korpusa girmemiş.
- `ids.json`'daki `val` bağlantılı görsel: 36 (hepsi `train` ile de eşleşen, bkz. 7.7.2).
- Gözlem (yorum, doğrulanmadı): diskteki 928 yalnız-`val` görsel, indeksin dışında kaldıkları için R2'de held-out değerlendirme adayı olabilir; ama bunların QuiltNet eğitiminde kullanılıp kullanılmadığı ve sayının golden set için yeterliliği **bulunamadı/kontrol edilmedi**.

#### 7.7.5 Mevcut JSONL `balanced` mı `stratified` mı?

**Sonuç: `balanced` ile üretilmiş (yüksek güven, komut kaydı yok).** JSONL: 168,739 satır, 114,222 tekil görsel. Satır sayıları `BALANCED_TARGETS` ile karşılaştırıldı (kategori adları normalize edilmiş, küçük harf; `BALANCED_TARGETS` şu anki hâli):

| kategori | hedef | JSONL satır | JSONL tekil görsel | durum |
|---|---|---|---|---|
| unknown | 25,000 | 25,003 | 25,003 | hedefe ulaştı (+3, aşağıya bak) |
| dermatopathology | 20,000 | 20,000 | 17,248 | birebir |
| gastrointestinal | 18,000 | 18,000 | 11,669 | birebir |
| pulmonary | 14,000 | 14,000 | 8,312 | birebir |
| gynecologic | 14,000 | 14,000 | 7,578 | birebir |
| soft tissue | 12,000 | 12,000 | 7,454 | birebir |
| hematopathology | 10,000 | 10,000 | 5,370 | birebir |
| genitourinary | 8,000 | 8,000 | 5,093 | birebir |
| bone | 7,000 | 7,000 | 4,015 | birebir |
| neuropathology | 6,000 | 6,000 | 3,492 | birebir |
| endocrine | 5,000 | 5,000 | 3,046 | birebir |
| head and neck | 5,000 | 5,000 | 2,874 | birebir |
| breast | 12,000 | 6,500 | 3,502 | hedefin altında |
| renal | 10,000 | 8,504 | 4,371 | hedefin altında |
| cardiac | 5,000 | 3,194 | 1,759 | hedefin altında |
| hepatopathology | 5,000 | 1,934 | 998 | hedefin altında |
| cytopathology | 4,000 | 1,687 | 975 | hedefin altında |
| ophthalmic | 3,000 | 2,286 | 1,152 | hedefin altında |
| pediatric | 3,000 | 631 | 311 | hedefin altında |
| **toplam** | **186,000** | **168,739** | 114,222 | |

Gerekçe:
- 11 kategori hedef değerle **tam yuvarlak** satır sayısına sahip. `stratified` kotaları `round(n·count/total)` olduğu için böyle yuvarlak sayılar üretmez, ayrıca `--n` varsayılanı 10,000 iken burada 168,739 satır var.
- Hedefin altındaki 7 kategori `take = min(hedef, havuz)` davranışıyla (`balanced_sample`) uyumlu: havuz yetmemiş. Tam CSV'de bu kategorilerin havuzu hedefleri aşıyor (örn. breast 21,495 satır; cardiac 11,088), yani yetersizlik CSV'den değil **o günkü görsel kaynağından** (zip/klasör filtresi) geliyor. O günkü kaynak kümesi çalışma ortamında yok, bu yüzden havuz boyutları doğrulanamadı.
- `unknown` 25,003 = 25,000 + 3: eski kodda ayrı `"Unknown": 4000` hedefi vardı ve büyük-U etiketli 3 satır oraya düşmüş; normalizasyonla `unknown`'a birleşti (7.4'teki 3 görsellik `Unknown` ile tutarlı).
- Stratified olsaydı kategori payları havuzla orantılı olurdu; burada `dermatopathology` JSONL'da %11.9, CSV `train` havuzunda ~%31, yani dağılım orantılı değil.

Sınırlar: bunu doğrudan kanıtlayan komut/`subset_stats.json` bulunamadı (çıkarım). Tekil görsel sayısı satırdan düşük, çünkü örnekleme **satır** bazlı ve aynı görselin birden çok satırı aynı kategoriye gidiyor (görsel başına en çok 9 satır). Korpus oranları Quilt-1M'in doğal dağılımı değil, hedef tablosunun ürünüdür (7.6 c ile aynı sonuç, artık veriyle destekli).

#### 7.7.6 `stratified_sample` top-up şüphesi: doğrulandı

`tests/test_prepare_quilt.py` sahte 30 satırlık DataFrame (3 kategori × 10) ile `n=10` çağırıyor: kotalar `round(3.33)=3` ×3 = 9 satır, yani `len(result) < n` dalı çalışıyor. 30 seed'in içinde **tekrar eden satır çıkıyor** (ilk hata seed 4 civarında), test başarısız. Neden: `result` `ignore_index=True` ile 0..8'e yeniden indekslendi, `df.index.isin(result.index)` ise orijinal etiketlerle karşılaştırıyor; bu yüzden (a) zaten seçilmiş satırlar `remaining`'e geri girip yeniden çekilebiliyor, (b) indeksi 0..8 olan seçilmemiş satırlar hiç çekilemiyor. Kod **düzeltilmedi**; test `xfail(strict=True)` olarak duruyor (düzeltilince XPASS ile CI'ı kırıp işaretler). Gerçek veride etkisi: `stratified` modu ancak `n` kotaların toplamından büyükse tetikleniyor; mevcut JSONL `balanced` üretildiği için (7.7.5) bu hatadan etkilenmemiş.

### 7.8 Yalnız-`val` görseller, caption durumu ve QuiltNet split kaynakları (2026-10-15)

Yöntem: yalnız okuma. Zip dosyalarına bakılmadı, hiçbir veri dosyası değiştirilmedi. Sayılar tek komutla yeniden üretilir:

```
uv run python scripts/analyze_corpus.py --encoder clip --min_per_category 20
```

Çıktı: `reports/corpus_analysis.json` (yollar `.env`'den: `PATHOSEARCH_CSV_PATH`, `_CAPTIONS_PATH`, `_IMAGES_DIR`, `_ARTIFACTS_DIR`). Seed yok, rastgelelik yok. Mantık `src/pathosearch/eval/corpus_analysis.py`, test `tests/test_corpus_analysis.py`.

**Tekrar üretilebilirlik kontrolü:** komut 7.7.1, 7.7.2, 7.7.3, 7.7.4 (CSV split, `ids.json`/disk sayıları), 7.3 (satır/görsel histogramı), 7.4 (görsel bazlı kategori dağılımı) ve 7.7.5 (`BALANCED_TARGETS` karşılaştırma tablosu) sayılarını **birebir** üretti. Bu komutun kapsamadığı bölümler (7.1 CLIP/QuiltNet id eşitliği, 7.5 `model_info.json`/`faiss_meta.json`, 7.6 kod okuması, 7.7.6 test) hâlâ elle/ayrı kaynaktan.

#### 7.8.1 Diskteki 928 yalnız-`val` görselin kategori dağılımı

Tanım: `data/images` altında dosya adı var **ve** CSV'de yalnız `val` satırları var (hiç `train` satırı yok). 928 görsel, 1,283 CSV satırı. Hiçbiri `ids.json`'da yok (0). Etiket: `pathology` listesinin ilk elemanı, küçük harfe normalize (`prepare_quilt.parse_pathology`, korpusla aynı kural). Görsel içinde etiket çelişkisi: 0.

| kategori | görsel | oran | ≥20 |
|---|---|---|---|
| unknown | 471 | 50.75% | (kategori sayılmadı) |
| dermatopathology | 161 | 17.35% | evet |
| gastrointestinal | 107 | 11.53% | evet |
| soft tissue | 72 | 7.76% | evet |
| genitourinary | 31 | 3.34% | evet |
| pulmonary | 22 | 2.37% | evet |
| renal | 18 | 1.94% | hayır |
| cardiac | 11 | 1.19% | hayır |
| hematopathology | 10 | 1.08% | hayır |
| bone | 6 | 0.65% | hayır |
| neuropathology | 5 | 0.54% | hayır |
| endocrine | 4 | 0.43% | hayır |
| ophthalmic | 4 | 0.43% | hayır |
| gynecologic | 3 | 0.32% | hayır |
| head and neck | 2 | 0.22% | hayır |
| pediatric | 1 | 0.11% | hayır |

- **`unknown` oranı %50.75** (471/928). Korpustaki %21.89'dan (7.4) belirgin yüksek.
- **Eşik ≥20 (unknown hariç) geçen yalnız 5 kategori:** dermatopathology, gastrointestinal, soft tissue, genitourinary, pulmonary. Kalan 10 kategori eşiğin altında; bunlar için bu 928'lik küme tek başına golden set kaynağı olamaz.
- Hazırlık filtresi (`not_histology==0` ve caption dolu) 928 görselin **hiçbirini** elemiyor (`not_histology` hepsinde `0`); filtreli dağılım yukarıdakiyle aynı.
- Eşik 20 isteğe bağlı örnek değerdir (`--min_per_category` ile değişir). 20 görsel, sorgu başına P@10 için bile sınırda (§4 risk 7 ile ilişkili); yeterlilik kararı açık (§8).

#### 7.8.2 Caption durumu ve CSV alanları

| CSV satırı/görsel | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| görsel sayısı | 698 | 143 | 58 | 22 | 5 | 2 |

- Her görselde CSV satır sayısı = farklı caption sayısı (görsel başına yinelenen caption yok). **230 görselin (%24.8) birden fazla caption'ı var**; korpustaki oran %31.2 (35,605/114,222) idi.
- `caption`, `subset`, `split`, `not_histology`, `single_wsi`: 1,283 satırın hepsinde dolu.
- **Yalnız bir alt kümede dolu alanlar** (satır: 812; görsel: 457): `pathology`, `noisy_text`, `corrected_text`, `med_umls_ids`, `magnification`, `height`, `width`. `roi_text`: 630 satır / 354 görsel (boş liste `[]` eksik sayıldı).
- Kalan 471 görselde (= `unknown` sayısı) bu alanların hepsi boş. `subset` değerleri (satır): `quilt` 812, `openpath` 210, `pubmed` 175, `laion` 86. 812 sayısı dolu-alan satırlarıyla örtüşüyor; ama "alanlar yalnız `quilt` alt kümesinde dolu" sonucu satır düzeyinde ayrıca çaprazlanmadı, **doğrulanmadı**.
- `magnification` değerleri `0.0` (480), `1.0` (129), `2.0` (203), eksik (471). Bu sayısal kodların neye karşılık geldiği bu çalışmada **bulunamadı**; yorumlanmadı.
- `single_wsi`: `1` → 612 satır, `0` → 671 satır.

#### 7.8.3 QuiltNet eğitiminde `val` kullanıldı mı, retrieval için hangi split holdout?

Kaynaklar WebFetch ile çekildi (sayfa metni küçük bir model tarafından özetlendi; alıntılar o çıktıdan, PDF'in tam metni elle okunmadı). Tablo kaynak düzeyini gösterir.

| Soru | Bulgu | Kaynak / doğrulama düzeyi |
|---|---|---|
| Retrieval hangi veride ölçüldü? | "Our experiments are conducted on two datasets: our holdout dataset from Quilt-1M and the ARCH dataset." Tablo 2 başlığı: "Cross-modal retrieval results on the Quilt-1M holdout set and ARCH dataset." | Makale §4 (arXiv 2306.11207, HTML). **Birincil kaynak** |
| Holdout nasıl oluşturuldu / boyutu? | **Bulunamadı.** Makalede holdout'un nasıl ayrıldığı ve boyutu yazmıyor. | Makale tam metni tarandı |
| Önerilen split var mı? | "There are no recommended data splits, as this data was curated mainly for pretraining rather than evaluation." | Makale Ek E (Datasheet), §2 Data composition. Birincil. §4'teki "holdout" ifadesiyle gerilim içeriyor; makale çözmüyor |
| Tablo 7 (Ek C.1) | Retrieval satırı: `histopathology image-text retrieval \| - \| Quilt-1M \| 1.02M \| - \| 13,559 \| -`. Başlık "Size (Train/Val/Test)" ama düzleştirilmiş metinden hangi sayının hangi sütuna ait olduğu **belirlenemedi** | Makale Ek C.1. Birincil, ama okuma belirsiz |
| QuiltNet nasıl eğitildi? | "We train QuiltNet by finetuning an OpenAI pre-trained CLIP model on Quilt-1M". Hangi satırların (train/val) eğitime girdiği **yazmıyor** | Makale §4. Birincil |
| Resmi repo README (kök, `eval/README.md`, `data/README.md`) | Split/holdout/`split` sütunu hakkında **hiçbir ifade yok**. `eval/README.md` retrieval'ı yalnız "For evaluation (zero-shot, retrieval, linear probing)" diye anıyor | GitHub `wisdomikezogwo/quilt1m`. Birincil |
| HF model kartı `wisdomik/QuiltNet-B-32` | Split bilgisi yok; "Evaluation done with code in the CLIP Benchmark suite." | Birincil (model kartı) |
| Resmi repo **kodu** (holdout'u seçen / eğitimde `val`'i dışlayan kod) | **Bulunamadı / incelenmedi.** Yalnız README'ler ve kök dizin listesi okundu; eğitim ve eval kodu okunmadı | – |
| Deepwiki (`deepwiki.com/wisdomikezogwo/quilt1m/...`) | Sayfaya erişilemedi (HTTP 429), içeriği okunmadı. Bir arama özetinde "değerlendirme sayfası val split'ten söz etmiyor" gibi bir ifade vardı | **Doğrulanmadı (ikincil kaynak).** Hiçbir sonuç buna dayandırılmadı |

**Sayısal örtüşme (çıkarım, makalede yazmıyor):** CSV'de `val` satır sayısı **13,559**, Tablo 7'deki sayı da **13,559**; ayrıca "1.02M" CSV'nin toplam satırına (1,017,708) uyuyor. Bu, CSV'deki `val`'in makaledeki retrieval holdout'u olduğuna işaret **ediyor**, ama makale bunu açıkça söylemiyor ve Tablo 7'nin sütun ataması belirsiz. Sonuç: "CSV `val` = QuiltNet'in retrieval holdout'u" **olası ama doğrulanmadı**; "val eğitimde kullanılmadı" ise **hiçbir kaynakta doğrulanmadı**.

**Projeye etkisi (yorum):**
- 928 yalnız-`val` görselin QuiltNet'in görmediği veri olduğunu **iddia edilemez**; mevcut en iyi kanıt yukarıdaki sayısal örtüşme.
- Korpus (114,222) 114,186 saf `train` + 36 `train`/`val` karışık görsel (7.7.2); QuiltNet'in bunları gördüğü varsayımı en makul varsayım, ama split metadatasından değil "genel eğitim kümesi" varsayımından geliyor.
- Karar gereken nokta §8-2 (held-out korpus mu, dürüstlük notu mu). 928 görsel yalnız 5 kategoride ≥20 ve %51'i `unknown`; tek başına golden set için yetersiz görünüyor. Alternatif (CSV'de 9,687 yalnız-`val` görsel var, diskte 928) bu çalışmada **denenmedi**, zip'lere bakılmadı.

### 7.9 Bitirme raporundan aktarılan bilgiler (2026-10-19)

Kaynak: ortak bitirme raporu. **Rapor repoda yok; ham çıktıları ve komutları da repoda bulunmuyor.** Bu bölümdeki her sayı ve ifade "raporda yazıyor, repodan tekrar üretilemiyor" statüsündedir; doğrulanmış sayılmaz. Bölüm yalnız raporun anlattığını aktarır, kod veya artifact okunarak teyit edilmedi (aksi belirtilen yerler hariç). CLAUDE.md kuralı gereği komut + config + seed olmadan metrik iddiası sayılmaz.

#### 7.9.1 Sürüm tablosu (rapora göre)

P@10, K=10. **Sorgu setleri ve korpus sürümler arasında değiştiği için satırlar doğrudan karşılaştırılamaz** (bkz. §3, `comparison_v1_v2_v3.html` notu).

| Sürüm | Açıklama | QuiltNet P@10 | CLIP P@10 |
|---|---|---|---|
| V1 | Baseline, 64K alt küme, basit sorgular | 0.395 | 0.175 |
| V2 | Yeniden yazılmış sorgular + `unknown` filtresi | 0.457 | 0.216 |
| V3 | RRF ensemble (56 kavram × 3 varyant) | 0.480 | 0.230 |
| V4 | Inverse-frequency dengeli 114K alt küme | 0.539 | 0.323 |
| V5 | Naïve Jaccard (α=.6, β=.4, min-max) | 0.548 | 0.273 (rapor başarısız sayıyor) |
| V5b | Ağırlıklı Jaccard (α=.75, β=.25, γ=0, min-max yok) | 0.609 | 0.336 |
| V6 | BiomedCLIP cross-encoder (top-100 → top-10) | 0.654 | 0.434 |
| V7 chain | Jaccard → cross-encoder | 0.634 (rapor başarısız sayıyor) | – |
| V7 final | V6 + domain gate | 0.654 | 0.434 |

#### 7.9.2 "0.539 → 0.609" iddiasının kaynağı (rapora göre)

- Rapora göre `app.py`'deki "0.539 → 0.609" ifadesi **V4 → V5b** geçişidir: 0.539 rerank'siz QuiltNet (V4), 0.609 V5b.
- Bu, §4'teki "0.609 hiçbir yerde yok" bulgusunu **kaynak rapor düzeyinde** çözer. Ham çıktı yok, bu yüzden 0.609 repodan **tekrar üretilemiyor** ve §4'teki tekrar üretilebilirlik eksikliği sürüyor. §4'teki "0.609 muhtemelen rerank'li sonuç" tahmini rapor tarafından desteklenmiş oldu.
- `compare_models.py`'nin varsayılan α=.6/β=.4 ayarı rapordaki **başarısız V5** ayarıdır (V5b değil). Uygulamadaki α=.75/β=.25 V5b ile örtüşüyor.

#### 7.9.3 "64,052" tutarsızlığı (tahmin)

Rapora göre V1-V3 64K alt kümeyle, V4 ve sonrası 114,222 görselle koşulmuş. `comparison_114k.html` içindeki "64,052" ifadesi bu karışıklıktan kaynaklanıyor **olabilir (tahmin, doğrulanmadı)**. §8-5 açık kalıyor.

#### 7.9.4 Diskte eksik görseller (tahmin)

Rapora göre part-2 ve part-3 zip'leri bir noktada indirilip açılmış (195,966 aday; dengeli örnekleme sonrası 114,222). Çalışma ortamındaki `data/images` tek klasör (`images_part_1`, 65,322 görsel, §7.7.3). Bu, `ids.json`'daki 76,126 görselin diskte olmamasını açıklıyor **olabilir**: part-2/3 dosyaları yalnız raporun üretildiği ortamda kalmış olabilir. **Tahmin, doğrulanmadı.**

#### 7.9.5 Rapordaki ölçüm yöntemi

- Ground truth: elle yazılan `expected` kategori listesi; relevance: `pathology ∈ expected` (§4 ile aynı tanım).
- γ=0 ile doğrudan sızıntı kapatıldığı belirtiliyor; metrikler `unknown` filtreli, K=10.
- Domain gate: 12 çapa cümlesi, CLIP text kulesi, en yakın 3'ün ortalaması, eşik 0.80. 10 sorguluk kalibrasyonda 10/10; rapor 0.8015 vs 0.8026 gibi kıl payı vakalar olduğunu da yazıyor (eşik hassas, ROC yok, bkz. §5).
- V6 cross-encoder CPU'da sorgu başına ~26 sn; demo V5b kullanıyor (~200 ms). Gecikme değerleri rapordan, repoda ölçüm çıktısı yok.
- Rapor Görsel→Görsel modunu **nicel olarak ölçmemiş**.

#### 7.9.6 Raporun tartışmadığı noktalar

Bunlar §4 risk tablosuna 9-12 olarak eklendi:

- **(a)** Korpus tamamen `train` split (§7.7.2). Rapor %50.6'lık CLIP-QuiltNet farkını "yalnızca alan ön-eğitiminin etkisi" olarak yorumluyor, ama QuiltNet bu görselleri eğitimde görmüş olabilir (§7.8.3: tersi hiçbir kaynakta doğrulanmadı).
- **(b)** Reranker caption sözcüklerine bakıyor, relevance etiketi de aynı satırın kategorisinden geliyor. γ=0 yalnızca doğrudan sızıntıyı kapatır; dolaylı korelasyon durur. V5b'nin +0.070 kazancı bu yüzden şişkin olabilir.
- **(c)** Sorgu setleri sürümler arasında sonuçlara bakılarak yeniden yazılmış (V2) ve ayar-test ayrımı yok.
- **(d)** Rapordaki "literatürde eşdeğeri yok" gibi özgünlük iddiaları ve "0.45-0.65 tipik aralık" karşılaştırması bu repoda doğrulanmadı.

### 7.10 Kaynak grupları: video, tweet, pubmed figürü (2026-10-22)

Yöntem: salt okuma. CSV'de kaynak sütunu yok. Kaynak, dosya adından çıkarılır (`pathosearch.data.source_groups.source_key`):

| alt küme | dosya adı | anahtar |
|---|---|---|
| quilt | `<11 karakter video id>_image_<uuid>.jpg` | `video:<id>` |
| openpath | `<tweet id>_<k>.jpg/png` | `tweet:<id>` |
| pubmed | `<uuid>.jpg` veya `<uuid>_<k>.jpg` | `figure:<uuid>` |
| laion | `<sayı>.jpg` | yok |

Diskte bu eşleme alt kümeyle birebir örtüşüyor (quilt 43,693 → video, openpath 13,414 → tweet, pubmed 5,941 → figür, laion 2,274 → anahtar yok). "Aynı UUID = aynı figürün paneli" ve "aynı tweet id = aynı tweet" yorumları **tahmin**dir; dosya adı biçiminden çıkarıldı, görsellere bakılmadı.

Komut (tek komutla yeniden üretilir, seed yok):

```
uv run python scripts/analyze_source_groups.py --golden eval_data/golden_v0
```

Çıktı: [reports/source_groups.json](../reports/source_groups.json).

**CSV split'i kaynak bazında mı?**

| anahtar | grup | yalnız-train | yalnız-val | train+val karışık |
|---|---|---|---|---|
| video | 4,475 | 4,435 | 40 | **0** |
| tweet | 53,727 | 52,844 | 883 | **0** |
| figür | 29,983 | 28,040 | 600 | **1,343** |

- `quilt` ve openpath'te split kaynak (video/tweet) bazında yapılmış. pubmed'de figür bazında yapılmamış.
- Video bazında ayrık bir `val`, bilinçli olarak ayrılmış bir holdout ile **uyumlu** (§7.8.3'teki 13,559 örtüşmesiyle birlikte). Bu yine de QuiltNet'in `val`'i eğitimde görmediğini **kanıtlamaz** (tahmin).

**Golden ile train arasında kaynak paylaşımı**

| kontrol | sonuç |
|---|---|
| Golden 928 aday: video / tweet / figür / anahtarsız | 457 görsel (37 video) / 210 (191 tweet) / 175 (175 figür) / 86 |
| Golden 820: video / tweet / figür / anahtarsız | 356 (37 video) / 203 (185 tweet) / 175 / 86 |
| Diskteki train-only adaylar (64,369) | video 43,236 (4,067 video; video başına medyan 6, en çok 177), tweet 13,204, figür 5,766, anahtarsız 2,163 |
| Golden ile kaynak paylaşan diskteki train adayı | **32** (hepsi figür, 31 farklı figür; 10'u ana 114K korpusta). Video 0, tweet 0 |
| Golden ile kaynak paylaşan ana korpus (114,222) görseli | 46 |
| **CSV'nin tamamında `train` kardeşi olan golden görseli** | **123** (hepsi figür; golden'daki 175 pubmed görselinin %70'i, 820'nin %15'i) |

**Etkisi (yorum):**
- **QuiltNet karşılaştırması (R3):** QuiltNet CSV `train` ile eğitildiyse, golden'daki 123 pubmed görselinin aynı figürden kardeş panellerini görmüş olabilir. Bu, §4 risk 9'u ve R3 raporundaki "QuiltNet eğitim verisi belirsizliği"ni golden düzeyinde somutlaştırıyor. Video ve tweet kaynaklı golden görselleri (559/820) için kaynak düzeyinde sızıntı yok. Kardeş panelin golden görseline ne kadar benzediği ölçülmedi.
- **R4 eğitim kümesi:** golden ile kaynak paylaşan 32 train adayı eğitimden çıkarılır. İç doğrulama kümesi kaynak grubu bazında bölünür, çünkü video başına medyan 6 kare var ve görsel bazında bölünürse aynı videonun kareleri train ve val'e dağılır.
- İlk taramada (yalnız `UUID_k.jpg` biçimi) figür eşleşmesi 19 bulunmuştu. Eksiz `UUID.jpg` adları da anahtara dahil edilince 32 oldu. Geçerli değer komutun ürettiği 32'dir.

## 8. Açık sorular ve kararlar

1. §4'teki yorum (0.539 = rerank'siz QuiltNet, 0.609 kaynağı yok) proje sahibinin bildiği bilgiyle uyuşuyor mu? Varsa 0.609'un çıktığı koşunun komutu/dosyası nedir? (Güncelleme: bitirme raporuna göre 0.539 = V4, 0.609 = V5b, bkz. §7.9.2; açık kalan kısım ham çıktı ve komutun repoda bulunmamasıdır.)
2. ~~Gerçek korpus train mi test mi, 64k mı 114k mı?~~ **Yanıtlandı (§7.1-7.2):** 114,222 görsel, %100 `train`. Karar: "QuiltNet'in train verisi üzerinde ölçüm" sonucu R2'de nasıl ele alınacak (ayrı held-out korpus mu, mevcut korpusta dürüstlük notu mu)?
3. Uygulamadaki iddia metni R1'de nötrlenip R2'de yeniden hesaplansın mı? (Not: git geçmişine göre iddia metni zaten değiştirilmiş görünüyor: `76b5bbc`; uygulama durumu elle doğrulanmadı.)
4. §7.3: Çok-caption'lı görseller için reranker hangi caption'ı kullanmalı? Karar R2'den önce gerekli. (Mevcut davranış §7.6 (b): JSONL'daki son satır, seed'e bağlı.)
5. Açık soru: `comparison_114k.html` içindeki "64,052 görsel" ifadesi nereden geldi? (§3, §7.1; olası açıklama §7.9.3, tahmin)
