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
| `inspect_quilt.py` | Tek seferlik keşif; sabit yollar `C:\Users\user\Downloads\...` (kullanıcı adı içeriyor) |
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

Hardcode/hack özeti: `os.chdir` (app, `domain_gate`, `test_*`), `sys.path.insert` (app, 6+ script), mutlak `E:\` ve `C:\Users\user\...` yolları, `data/images` → `data/images_test` ve `captions_quilt` → `captions_quilt_test` **sessiz fallback**'i (`app.py:20-32`).

## 3. Raporların iddiaları

| Rapor | Ne iddia ediyor | Notlar |
|---|---|---|
| `comparison_64k.html` (2026-05-22) | 63 etiketli sorgu, K=10: P@10 CLIP **0.175** vs QuiltNet **0.395**; mAP 0.272/0.520; nDCG 0.172/0.410; gecikme 32.7/62.5 ms (CPU) | = V1 (rerank yok, unknown filtresi yok) |
| `comparison_114k.html` (2026-06-04) | 56 etiketli sorgu, K=10: P@10 CLIP **0.323** vs QuiltNet **0.539**; mAP 0.440/0.633; nDCG 0.326/0.547; gecikme 106.3/181.9 ms | **Dosya adı "114k" ama içindeki korpus "64,052 görsel"**; `compare_models.py` ise sabit olarak "114K image corpus" yazıyor. Rapor ile kod senkron değil. **Açık soru:** "64,052" sayısının kaynağı bulunamadı (§7.1: gerçek korpus 114,222). Sorgu adları (`skin squamous cancer`, `q003`) V3 ensemble'ın genel varyantı. Başlıkta rerank etiketi yok → rerank **kapalı** koşmuş görünüyor |
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
| 6 | **Etiket gürültüsü ve kaba ilgi.** Etiket otomatik/metadata kaynaklı; ilk eleman alınıyor; `unknown` korpusta **%21.89** (görsel bazında, §7.4; eski "~%33" rapor notu korpusla örtüşmüyordu). Kategori düzeyi P@10 hastalık düzeyi doğruluğu değil | `make_comparison_report.py` metodoloji | Orta |
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

## 7. Artifact incelemesi (2026-10-14)

Yöntem: `data/captions_quilt.jsonl` ve `artifacts/{clip,quilt}/*/` dosyaları **sadece okundu**, hiçbiri değiştirilmedi. Geçici analiz betikleri repo dışında tutuldu. Görsel sayısı kontrolü yapılmadı (zip'ler henüz çıkarılmadı). Bu bölüm §0 madde 1'i (artifact yok) ve §4'teki 2. sorusunu yanıtlar.

### 7.1 Korpus boyutu: CLIP ve QuiltNet aynı mı?

**Evet, birebir aynı.**

| Kontrol | CLIP | QuiltNet |
|---|---|---|
| `ids.json` uzunluğu | 114,222 | 114,222 |
| Tekil id | 114,222 | 114,222 |
| `embeddings.npy` şekli (başlıktan) | (114222, 512) float32 | (114222, 512) float32 |
| `faiss_meta.json` `n` | 114,222 | 114,222 |

İki `ids.json` kümesi aynı ve **sıraları da aynı** (fark 0). Sonuç: korpus **114k**; `comparison_114k.html` dosya adı doğru, ama içindeki "64,052 görsel" ifadesi bu korpusla uyuşmuyor, nedeni **bulunamadı** (§3'teki tutarsızlık sürüyor).

### 7.2 Split dağılımı (`ids.json`'da olan satırlar)

`ids.json`'daki 114,222 id'nin **tamamı** captions'ta bulundu (eksik 0). Eşleştirme anahtarı `path`'in dosya adı (basename).

| split | satır | oran |
|---|---|---|
| `train` | 168,739 | 100% |
| `test` / `val` / diğer | 0 | 0% |

Korpus **tamamen `train`**. Dosyanın tamamı da `train`; yani dışarıda bırakılmış test satırı yok. §4 risk 5 ("QuiltNet kendi eğitim verisinde ölçülmüş olabilir") artık **sadece varsayım değil, veriyle destekli**: korpus `train` etiketli. Tek belirsizlik: `split` alanının Quilt-1M CSV'sinden mi geldiği kodda doğrulanmalı, ve QuiltNet'in eğitim kümesiyle tam örtüşme derecesi **bulunamadı** (kontrol edilmedi).

### 7.3 Caption satırı ≠ görsel sayısı

`captions_quilt.jsonl` 168,739 satır, ama tekil `path` (ve basename) sayısı **114,222**. Fark, aynı görselin birden fazla caption'a sahip olması (Quilt-1M'de bir görsel için birden çok metin var):

| Satır/görsel | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Görsel sayısı | 78,617 | 22,327 | 9,131 | 3,044 | 828 | 199 | 45 | 29 | 2 |

- 35,605 görselin birden fazla **farklı** caption'ı var. Aynı görselin satırları arasında `pathology` farkı **0** (etiket görsel başına tutarlı).
- Tam yinelenen (path+caption+pathology) satır yok.
- **Etki (doğrulanmadı):** Reranker ve CE caption kullanıyor; çok-caption'lı görselde hangisinin seçildiği kodda net değil (`src/` içinde captions yükleyen bir yer bulamadım, grep sonucu `config.py` dışında yok). Birleştirme/ilk satır/son satır hangisi ise Jaccard skoru buna bağlı; **R2 öncesi incelenmeli**.
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
- **Bulunamadı:** CSV'deki gerçek değerlerin (`train/val/test` yazımı, dağılımı) doğrulanması; CSV bu makinede incelenmedi. §7.2'deki "%100 train" bulgusu bu yüzden JSONL'daki yazılan değere dayanıyor, CSV'ye değil.

**(b) Çok-caption'lı görselde hangi caption kullanılıyor?** Her yerde tek caption: **JSONL'daki o görselin son satırı.**
- `faiss_search.load_captions` `{basename: {caption, pathology}}` sözlüğü kuruyor; aynı basename tekrar gelince atama üzerine yazıyor (`cap_map[fname] = ...`). Birleştirme yok, ilk satır korunmuyor.
- Aynı sözlük `raw_search` ile her sonuca `caption`/`pathology` olarak ekleniyor. Reranker (`rerank_results`), cross-encoder (`c.get("caption")`), uygulama (`pipeline.SearchIndex.captions`) ve `compare_models` hep bu sonuç alanını okuyor. Başka bir caption yükleyen yer **bulunamadı**.
- "Son satır" anlamsal bir seçim değil: `prepare_quilt` örneklemeden sonra satırları `sample(frac=1, random_state=seed)` ile karıştırıyor, yani hangi caption'ın kazandığı JSONL sırasına (seed'e) bağlı. Jaccard skoru bu rastgele seçime bağlı.
- JSONL'da bir görselin tüm Quilt caption'ları da olmayabilir: örnekleme **satır** bazlı (bkz. c), bir görselin yalnız örneklenen satırları dosyaya girer.
- Not: klasör modunda ikinci kez gelen satır `dst.exists()` olduğu için görsel kopyalamayı atlıyor (`skip`) ama yine de JSONL'a yazılıyor; çok-caption'lı görsellerin JSONL'da birden fazla satırı olmasının kod tarafındaki nedeni bu.

**(c) Kategori başına satır kotası var mı?** Evet, iki modda da, birim **satır** (görsel değil).
- `stratified` (varsayılan): kategori başına `quota = max(1, round(n * count / total))`, `count/total` satır sayısı. Orantılı; `--n` toplam satır hedefi.
- `balanced`: `BALANCED_TARGETS` kategori başına en fazla `min(hedef, havuz)` satır alır; `--n` yok sayılır. Kategori başına yuvarlak tavanlar (`unknown` 25000, `dermatopathology` 20000, ...) buradan geliyor.
- Mevcut `captions_quilt.jsonl` hangi modla üretildi: **bulunamadı** (komut ve `subset_stats.json` incelenmedi). §7.4'teki yuvarlak satır sayıları `balanced` ile tutarlı **(tahmin)**; `artifacts/subset_stats.json` (`category_distribution`) bunu doğrular.
- Sonuç: `balanced` ise dağılım Quilt-1M'in doğal dağılımı değil, hedef tablosunun ürünü; kategori oranlarından korpus temsiliyeti çıkarılamaz.
- Okurken görülen, doğrulanmamış şüphe: `stratified_sample` içinde `remaining = df[~df.index.isin(result.index)]` satırında `result` `ignore_index=True` ile yeniden indekslenmiş; indeksler `df` ile eşleşmediği için doldurma (`len(result) < n`) yanlış satırları dışlayıp tekrar eden satır ekleyebilir. Çalıştırılıp test edilmedi.

**Bu bölümle birlikte yapılan kod değişiklikleri**
- `f14bb9b`: captions varsayılan yolu `data/captions_quilt.jsonl`.
- Pathology normalizasyonu (`labels.normalize_pathology`: kırp + küçük harf): `load_captions`, `parse_pathology`, `is_relevant` ve reranker bonusunda uygulanıyor, böylece `Unknown`/`unknown` tek kategori. `BALANCED_TARGETS` anahtarları küçük harfe çevrildi ve ayrı `"Unknown": 4000` girdisi kaldırıldı (normalizasyondan sonra `unknown` ile çakışırdı). Etkisi: arayüzde ve raporlarda kategori adları küçük harfle görünür (ör. `dermatopathology`); bellekte, mevcut JSONL'u yeniden üretmeden de uygulanır çünkü normalizasyon yükleme sırasında yapılıyor.

## 8. Senin onayın gereken noktalar

1. §4'teki yorum (0.539 = rerank'siz QuiltNet, 0.609 kaynağı yok) sende bildiğin bilgiyle uyuşuyor mu? Varsa 0.609'un çıktığı koşunun komutu/dosyası?
2. ~~Gerçek korpus train mi test mi, 64k mı 114k mı?~~ **Yanıtlandı (§7.1-7.2):** 114,222 görsel, %100 `train`. Sende: "QuiltNet'in train verisi üzerinde ölçüm" sonucunu R2'de nasıl ele alalım (ayrı held-out korpus mu, mevcut korpusta dürüstlük notu mu)?
3. Uygulamadaki iddia metni R1'de nötrlenip R2'de yeniden hesaplansın mı? (Not: git geçmişine göre iddia metni zaten değiştirilmiş görünüyor: `76b5bbc`; uygulama durumu elle doğrulanmadı.)
4. §7.3: Çok-caption'lı görseller için reranker hangi caption'ı kullanmalı? Karar R2'den önce gerekli. (Mevcut davranış §7.6 (b): JSONL'daki son satır, seed'e bağlı.)
5. Açık soru: `comparison_114k.html` içindeki "64,052 görsel" ifadesi nereden geldi? (§3, §7.1)
