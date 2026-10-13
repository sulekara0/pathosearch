# PathoSearch: Durum Raporu (R0 keşfi)

Tarih: 2026-10-14 · Kapsam: `v0-prototype` etiketli kod (commit `d123df5` + `5e1c645`) · Yöntem: tüm `.py` dosyaları ve `reports/*.html` baştan okundu, kod çalıştırılmadı.

> Kural: Bu belgede "bulunamadı" yazan şey repoda yoktur. Tahmin ayrıca **(tahmin)** diye işaretlendi.

## 0. Önce bilinmesi gerekenler

1. **Yerelde `artifacts/` ve `data/` yok** (`.gitignore`'da, `E:\Multisearch_data` de bu makinede yok). Embedding, FAISS indeksi, `captions_quilt.jsonl`, `eval_queries*.json` ve `benchmark_*.json` dosyalarının hiçbiri incelenemedi. Hiçbir metrik bu oturumda yeniden üretilemedi; aşağıdaki sayılar sadece `reports/*.html` içinde yazanlardır.
2. **"P@10 0.539 → 0.609" iddiası doğrulanamıyor.** Ayrıntı §4.
3. `scripts/` içinde iki dosya **çalışmaz durumda** (§2, "Kırık").

## 1. Repo haritası

| Yol | Rol |
|---|---|
| `app.py` (408 satır) | Streamlit UI + `search_pipeline` + artifact kontrolü (mantık UI'ya gömülü) |
| `scripts/` | 18 dosya, ayrıntı §2 |
| `inspect_quilt.py` | Tek seferlik keşif; sabit yollar `C:\Users\Çağrı\Downloads\...` (kullanıcı adı içeriyor) |
| `reports/` | 6 statik HTML (üçü karşılaştırma, üçü tek sorgu görseli) |
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

Hardcode/hack özeti: `os.chdir` (app, `domain_gate`, `test_*`), `sys.path.insert` (app, 6+ script), mutlak `E:\` ve `C:\Users\Çağrı\...` yolları, `data/images` → `data/images_test` ve `captions_quilt` → `captions_quilt_test` **sessiz fallback**'i (`app.py:20-32`).

## 3. Raporların iddiaları

| Rapor | Ne iddia ediyor | Notlar |
|---|---|---|
| `comparison_64k.html` (2026-05-22) | 63 etiketli sorgu, K=10: P@10 CLIP **0.175** vs QuiltNet **0.395**; mAP 0.272/0.520; nDCG 0.172/0.410; gecikme 32.7/62.5 ms (CPU) | = V1 (rerank yok, unknown filtresi yok) |
| `comparison_114k.html` (2026-06-04) | 56 etiketli sorgu, K=10: P@10 CLIP **0.323** vs QuiltNet **0.539**; mAP 0.440/0.633; nDCG 0.326/0.547; gecikme 106.3/181.9 ms | **Dosya adı "114k" ama içindeki korpus "64,052 görsel"**; `compare_models.py` ise sabit olarak "114K image corpus" yazıyor. Rapor ile kod senkron değil. Sorgu adları (`skin squamous cancer`, `q003`) V3 ensemble'ın genel varyantı. Başlıkta rerank etiketi yok → rerank **kapalı** koşmuş görünüyor |
| `comparison_v1_v2_v3.html` (2026-05-22) | QuiltNet P@10: V1 0.395 → V2 0.457 → V3 0.480; CLIP: 0.175 → 0.216 → 0.230 | Raporun kendisi, sorgu setleri farklı olduğu için değerlerin doğrudan karşılaştırılamayacağını yazıyor |
| `search_*.html` ×3 (2026-04-29) | Tek sorgu CLIP vs QuiltNet görsel çıktı; iddia yok | `lymph node` için QuiltNet'in 5 sonucundan 3'ü `unknown`/alakasız görünüyor |

Tutarsızlık: V3 JSON'unda QuiltNet 0.480 (56 kavram), `comparison_114k`'de 0.539 (56 sorgu). Aynı sorgu sayısı, farklı sayı; hangi konfigürasyonla (filtre/ensemble/pool) çalıştığı raporda **kayıtlı değil**.

## 4. "Precision@10 0.539 → 0.609" iddiası

**Kaynak**
- Tek geçtiği yer: `app.py:235` (sidebar `help` metni). Kodda, JSON'da, raporda **0.609 hiç geçmiyor**. 0.609'un nasıl hesaplandığı **bulunamadı**.
- **0.539**, `reports/comparison_114k.html`'de **QuiltNet'in rerank'siz P@10'u** olarak geçiyor. Yani "0.539" muhtemelen ham QuiltNet taban çizgisi, "0.609" ise rerank'li sonuç **(tahmin)**; rerank'li koşunun raporu/çıktısı repoda yok.
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
| 5 | **Eğitim/test sızıntısı (QuiltNet adil mi?).** `00_prepare` varsayılan split `train`; QuiltNet Quilt-1M üzerinde eğitildi. Hangi split'in gerçekten kullanıldığı **bulunamadı** (artifact yok). Eğer `train` ise QuiltNet kendi eğitim verisinde ölçülmüş olur | `00_prepare_quilt.py:48` | Yüksek (doğrulanmalı) |
| 6 | **Etiket gürültüsü ve kaba ilgi.** Etiket otomatik/metadata kaynaklı; ilk eleman alınıyor; `unknown` ~%33 (rapor notu). Kategori düzeyi P@10 hastalık düzeyi doğruluğu değil | `make_comparison_report.py` metodoloji | Orta |
| 7 | **nDCG'nin IDCG'si her kategoride ≥K ilgili belge olduğunu varsayıyor**; küçük kategorilerde (Ophthalmic 1 sorgu) şişirir/saptırır | `compare_models.py:130`, rapor dipnotu | Düşük-Orta |
| 8 | **İstatistiksel güç düşük**: kategori başına 1-10 sorgu, güven aralığı/seed yok, kategori ağırlığı eşit değil (makro yerine sorgu ortalaması) | rapor tabloları | Orta |

**Sonuç:** İddia, şu haliyle bağımsız ölçüm sayılamaz. 0.539 rerank'siz QuiltNet ölçümü olarak raporda var, 0.609 hiçbir yerde yok. Rerank kazancı büyük olasılıkla (1)'in etkisiyle şişkin. Arayüzdeki metin R2'ye kadar kaldırılmalı ya da "doğrulanmadı" denmeli (karar sende).

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

## 7. Senin onayın gereken noktalar

1. §4'teki yorum (0.539 = rerank'siz QuiltNet, 0.609 kaynağı yok) sende bildiğin bilgiyle uyuşuyor mu? Varsa 0.609'un çıktığı koşunun komutu/dosyası?
2. Gerçek korpus `train` mi `test` split'i, 64k mı 114k mı? (Yerelde `subset_stats.json` / `ids.json` bakılırsa netleşir.)
3. Uygulamadaki iddia metni R1'de nötrlenip R2'de yeniden hesaplansın mı?
