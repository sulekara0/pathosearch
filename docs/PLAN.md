# PathoSearch: Yol Haritası v2 (repo durumuna göre)

Repo: `sulekara0/pathosearch` · `main` · v0 prototipi eski `Multisearch` reposundan alındı (etiket `v0-prototype`). Bu belge `docs/PLAN.md` olarak repoya konur. İlk rehberdeki (v1) Faz 3-7 ayrıntıları geçerlidir, burada sadece repo durumuna göre **farklar ve sıra** var.

## 1. Şu an neredeyiz?

| Bileşen | Durum | Not |
|---|---|---|
| Streamlit demo (`app.py`, 408 satır) | ✅ Var | 3 mod: Tek Model, Karşılaştırma (CLIP vs QuiltNet), Görsel→Görsel |
| Encoder'lar | ✅ Var (hazır) | Generic CLIP ViT-B/32 + QuiltNet ViT-B/32. Kendi eğitimin yok |
| FAISS indeks + embedding üretimi | ✅ Var | `scripts/01_build_embeddings.py`, `scripts/faiss_search.py` (ip/l2) |
| Domain gate | ✅ Var | CLIP text-anchor benzerliği, eşik elle seçilmiş (0.80) |
| Reranker V5b | 🟡 Kısmi | Caption tabanlı weighted Jaccard (α=.75, β=.25). Arayüzde "P@10 0.539 → 0.609" iddiası var, etiket tanımı doğrulanmalı |
| Gecikme ölçümü | 🟡 Kısmi | encode/search ms arayüzde var, p50/p95 raporu yok |
| Veri keşfi | 🟡 Kısmi | `inspect_quilt.py` sabit Windows yolları içeriyor |
| `scripts/`, `reports/` içeriği | ❓ Okunamadı | GitHub bu klasörlerin otomatik okunmasına izin vermedi. **Claude Code yerelde okuyacak** (R0) |
| Golden set / qrels / split | ❌ Görünmüyor | scripts veya reports'ta olabilir, R0'da doğrulanacak |
| Metrik modülü + test | ❌ Yok | Ana dizinde test yok |
| Paketleme (pyproject), ruff, mypy, CI | ❌ Yok | `requirements.txt` sürümleri sabitlenmemiş (`>=`) |
| README, açıklama, lisans, topic | ❌ Yok | Profil README'si başka repoya (`Multimodal_search`) link veriyor, `pathosearch` linklenmemiş |
| Fine-tuning | ❌ Yok | |
| Hibrit arama, cross-encoder, Qdrant | ❌ Yok | Sadece FAISS + sözcüksel rerank |
| FastAPI, PostgreSQL | ❌ Yok | Mantık Streamlit içinde |
| Ajan, MCP, tracing | ❌ Yok | Domain gate ajanın router/guardrail'ine temel olabilir |
| Docker, deploy, canlı demo | ❌ Yok | |

**Özet:** Çalışan bir retrieval prototipin var (güçlü yanı: domain gate, dürüst karşılaştırma modu, artifact kontrolü). Eksik olan mühendislik iskeleti ve **ölçüm güvenilirliği**.

## 2. Tespit edilen teknik borçlar (öncelik sırasıyla)

1. **Ölçüm güvenilirliği (en kritik).** P@10 iddiasında "ilgili" nasıl tanımlandı? Etiket caption/pathology alanından türetiliyorsa ve reranker de caption benzerliğini kullanıyorsa sonuç döngüsel (yapay yüksek) olabilir. Bağımsız golden set şart. Ayrıca Karşılaştırma modundaki "ortak görsel sayısı" kalite metriği değildir.
2. **Mantık UI'ya gömülü.** `search_pipeline` `app.py` içinde, `scripts/` `sys.path` hack'i ve `os.chdir` ile içe aktarılıyor. API/MCP/test aynı çekirdeği kullanamaz. Çözüm: `src/pathosearch/` paketi.
3. **Sabit yollar.** `inspect_quilt.py` mutlak Windows yolları içeriyor (kullanıcı adı dahil). Config/env değişkenine taşı.
4. **Sessiz veri fallback'i.** `_find_img_dir` / `_find_captions` `images_test` ve `captions_quilt_test` dosyalarına sessizce düşebiliyor. Yanlış veriyle çalıştığını fark etmeyebilirsin. Açık config kullan, bulunamazsa hata ver.
5. **Gate eşiği kalibre değil.** 0.80 sezgisel. In-domain ve out-of-domain sorgu setiyle precision/recall (ROC) çıkar.
6. **Bağımlılıklar sabit değil.** `uv` ile lock dosyası üret.
7. **Büyük artifact'lar** (`embeddings.npy`, `faiss_index.bin`) için barınak planı yok. Hugging Face Hub veya release asset kullan.
8. **Repo hijyeni:** README, açıklama, topic'ler, lisans, profil README'sine doğru link.

## 3. Revize sıra

Her aşama küçük PR/commit'lerle ilerler. "Bitti sayılır" kriteri Claude Code'a prompt'ta verilir.

**R0: Keşif ve sabitleme (1-2 gün)**
- [ ] Mevcut hâli `v0-prototype` olarak etiketle
- [ ] Claude Code `scripts/` ve `reports/` içeriğini okuyup `docs/STATUS.md` yazar (her script ne yapar, hangi artifact üretir, hangi metrik nerede hesaplanır, raporlar neyi iddia eder, P@10 etiketi nasıl tanımlı)
- **Bitti:** STATUS.md senin onayınla doğrulanmış, bu tablodaki ❓ satırları netleşmiş

**R1: Paketleme ve kalite (Hafta 1)**
- [ ] `pyproject.toml` + `uv`, `src/pathosearch/` düzeni, `scripts/` → modüle taşı (`git mv` ile geçmişi koru)
- [ ] `config.py` (env tabanlı yollar), `inspect_quilt.py` sabit yollardan arındırılır
- [ ] ruff, mypy, pytest, pre-commit, GitHub Actions
- [ ] `app.py` ince istemci olur (sadece UI), `sys.path`/`os.chdir` hack'leri kalkar
- **Bitti:** CI yeşil, `uv sync && uv run pytest` temiz kurulumda çalışıyor, uygulama eskisi gibi açılıyor

**R2: Ölçüm temeli (Hafta 1-2)**
- [ ] `eval/metrics.py`: Recall@k, Hit@k, MRR, nDCG@k + testler
- [ ] Split stratejisi (kaynak/video bazlı), sızıntı testi
- [ ] Golden set v0 (100 sorgu, pooling, kappa), v1 rehberindeki yöntemle
- [ ] Mevcut P@10 iddiasını golden set üzerinde **yeniden hesapla**
- [ ] Domain gate kalibrasyonu (ROC, eşik seçimi)
- **Bitti:** `run_eval.py` tek komutla rapor üretiyor, eski iddia ya doğrulanıyor ya da düzeltilmiş sayıyla README'ye yazılıyor

**R3: Baseline raporu (Hafta 2)**
- [ ] Karşılaştırma tablosu: CLIP / QuiltNet / +Jaccard rerank / FAISS Flat vs HNSW (recall kaybı, p50/p95)
- [ ] ADR-0001 (metrikler), ADR-0002 (split)
- **Bitti:** README'de ilk dürüst sonuç tablosu

**R4: Fine-tuning (Hafta 3-5).** v1 Faz 3 aynen. Fark: baseline artık **QuiltNet** (hazır domain modeli). Hedef onu veya genel CLIP'i LoRA ile geçmek ya da neden geçemediğini analiz etmek. İkisi de değerli sonuçtur.

**R5: Modern retrieval (Hafta 6-7).** v1 Faz 4 aynen. Fark: Jaccard reranker silinmez, ablation'da "sözcüksel rerank" satırı olur. Cross-encoder ve hibrit (BM25 + RRF) ile karşılaştır, Qdrant benchmark'ı ekle.

**R6: Backend (Hafta 8-9).** v1 Faz 5 aynen. Çekirdek paket zaten R1'de ayrıldığı için FastAPI ince katman olur. Streamlit, API'yi çağıran istemciye döner.

**R7: Ajan + MCP (Hafta 10-11).** v1 Faz 6 aynen. Fark: mevcut domain gate ajanın ilk düğümü (kapsam dışı sorguyu reddet) olur.

**R8: Generation eval + deploy + yük testi (Hafta 11-12).** v1 Faz 7 aynen.

**R9: Dokümantasyon (Hafta 12).** README, teknik rapor, demo videosu, profil README'sini bu repoya yönlendir, CV maddesi.

## 4. Claude Code ile çalışma yöntemi

1. Repoyu yerelde klonla, kökte `claude` çalıştır. `CLAUDE.md` kökte durur, her oturumda otomatik yüklenir. `docs/PLAN.md` onun içinden `@docs/PLAN.md` ile çağrılır.
2. **Her aşama:** Plan Mode'da başla (Shift+Tab), planı oku, onayla, sonra uygulat. Aşamayı küçük issue'lara böl, her issue ayrı commit.
3. **Her prompt'a** hedef + "bitti sayılır" kriteri + dokunulmayacak yerleri yaz.
4. Context uzayınca yeni oturum aç. Kalıcı bilgi `docs/STATUS.md` ve ADR'lerde yaşar.
5. Gerçek veri yollarını, anahtarları `CLAUDE.local.md` ve `.env`'e koy (commit etme).
6. Her değişiklikte ne yaptığını ve **neden** yaptığını anlatmasını iste. Amaç sadece kodun bitmesi değil, senin mülakatta savunabilmen.

**İlk prompt (R0):**
> Henüz kod değiştirme. Repoyu baştan oku, özellikle `scripts/` ve `reports/`. `docs/STATUS.md` yaz: her script'in görevi, ürettiği artifact'lar, metriklerin hesaplandığı yer, raporların iddiaları. `app.py`'deki "Precision@10 0.539 → 0.609" iddiasının relevance etiketinin nasıl tanımlandığını bul ve döngüsellik riski var mı değerlendir. Bulamadığın şeyi uydurma, "bulunamadı" yaz.

**İkinci prompt (R1, Plan Mode):**
> `docs/PLAN.md` içindeki R1'i uygula. Önce plan çıkar: hangi dosya nereye taşınacak, hangi import'lar değişecek. Davranışı değiştirme (refactor), her adımda `streamlit run app.py` çalışmaya devam etmeli.