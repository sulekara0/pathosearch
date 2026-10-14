# Plan: Golden set v0 (R2 ölçüm temeli)

Bu belge onaylanmış R2 planıdır. Yol haritası için [PLAN.md](PLAN.md), güncel durum için [STATUS.md](STATUS.md).

## İlerleme

| İş | Durum | Commit |
|---|---|---|
| A1 qrels tabanlı Recall/Hit/MRR/nDCG | ✅ | `0ec8f3e` |
| A2 golden set şeması ve IO | ✅ | `fd53e69` |
| A3 sorgu düzeyinde dev/test split + sızıntı kontrolü | ✅ | `585d7d6` |
| A4 pooling + rastgele kontrol örneklemi | ✅ | `335e415` |
| A5 Cohen kappa + uzlaştırma | ✅ | `c3abc0d` |
| A6–A9 | ⏳ sonraki oturum | – |
| B1–B9 (veri gerektiren) | ⏳ | – |

> **Onay ekleri (2026-10-15):**
> 1. İlk oturumda yalnızca **A1–A5** yapıldı (yukarıdaki tablo). A6–A9 ayrı oturumda.
> 2. B4'te önce **20 sorguluk pilot** yapılır, sonra büyütülür. Sorgu sayısı ve havuz derinliği CLI parametresi olarak kalır. Dev oranı parametredir (`dev_frac`); verilmezse sorgu sayısı ≤50 ise 0.5 (50/50), değilse 0.3.
> 3. B2 için GPU gerekmez; 928 görsel CPU'da işlenir.
> 4. `docs/golden_set.md` dosyasına etiketleme süresi tahmini (saat) eklenir.
> 5. Push yok.

## Context

Mevcut "ilgili" tanımı metadata kategorisi (`pathology ∈ expected`, `eval/metrics.py`). Bu kaba (uzmanlık dalı düzeyi), `unknown` görselleri hiç ilgili saymıyor ve caption tabanlı reranker ile döngüsel (STATUS §4 risk 1). Korpus da tamamen `train` (§7.7.2), yani QuiltNet kendi eğitim verisinde ölçülüyor olabilir.

R2'nin hedefi bağımsız, insan etiketli bir golden set. v0 için aday korpus: diskteki **928 yalnız-`val` görsel** (§7.8.1). Hiçbiri `ids.json`'da yok. 5 kategoride ≥20 görsel var, %51'i `unknown`. Relevance insan etiketi olduğu için `unknown` görseller de havuzda kalıyor ve ilgili sayılabiliyor.

**Varsayım sınırı:** "val QuiltNet eğitiminde görülmedi" iddiası doğrulanmadı (§7.8.3). Golden setin geçerliliği buna bağlı değil, çünkü etiketler insandan geliyor. Bu varsayım yalnızca CLIP ve QuiltNet karşılaştırmasının adil olup olmadığını etkiliyor. O yüzden her raporda sabit bir `caveats` alanı olacak: *"QuiltNet held-out status of CSV val split: unverified (STATUS §7.8.3)."*

Alınan kararlar:
- Annotator caption'ı başta görmüyor; isterse açıyor ve açtığı `caption_seen=true` olarak kaydediliyor.
- Kappa'yı ikinci bir kişi hesaplıyor (inter-annotator).
- Sorgular yeni yazılıyor ve havuz kurulmadan önce donduruluyor.

## Tasarım ilkeleri

- **Korpus boyutundan bağımsız:** Her şey sürümlü bir golden dizini okur: `eval_data/golden_v0/`. Korpus, bir *seçim kuralıyla* üretilen bir manifest dosyasıdır (`corpus_manifest.jsonl`). Kod 928 sayısını hiçbir yerde bilmez. Genişletmek için yeni kuralla `golden_v1/` üretilir (zip'ten çıkarılan 9,687 val görseli, distractor eklemek vb.). Kod aynı kalır.
- **Repoya girenler:** görsel adları, kategori etiketi, sorgular, split, havuz, etiketler, qrels. **Girmeyenler:** görsel, caption metni, embedding, indeks. Caption runtime'da CSV'den okunur. Embedding'ler `artifacts/golden_v0/<encoder>/` altında durur (gitignore'da).
- Yeni bağımlılık yok. Kappa, bootstrap ve rank korelasyonu numpy ile yazılır. Arayüz için Streamlit zaten bağımlılıklarda var.
- Çekirdek mantık `src/pathosearch/eval/` altında. `scripts/` ve `label_app.py` yalnızca ince sarmalayıcı.

## Etiket kuralı (0/1/2), `docs/golden_set.md`

| Etiket | Kural |
|---|---|
| **2** | Sorgunun doku/organını **ve** sorgulanan bulguyu/antiteyi gösteriyor (annotator'ın görebildiği kadarıyla) |
| **1** | Doğru organ/uzmanlık veya yakın bir bulgu, ama sorgulanan antite değil ya da kısmen eşleşiyor |
| **0** | İlgisiz, histoloji değil veya sorguyla çelişiyor |
| **unsure** | Karar verilemedi. Qrels'e girmez, sayısı raporlanır |

- Binary metrikler iki eşikte raporlanır: **strict** (rel≥2) ve **lenient** (rel≥1). nDCG kademeli kazanç kullanır (`2^rel − 1`). Hangi eşiğin birincil metrik olacağına R3'te ADR-0001'de karar verilir.
- Belgeye örnekler eklenir. Bu bir eğitim/araştırma aracıdır, teşhis değil; bu not belgede de yer alır.

---

## A. Veri gerektirmeyen işler (sahte veriyle test edilir, CI'da koşar)

Her madde ayrı bir commit. `ruff`, `mypy` ve `pytest` geçmeden commit yok.

**A1. `feat(eval): qrels-based Recall/Hit/MRR/nDCG`.** Dosyalar: `eval/metrics.py`, `tests/test_metrics.py`.
- Yeni fonksiyonlar `ranked_ids: list[str]` ve `qrels: dict[str, int]` (image→grade) alır: `hit_at_k`, `recall_at_k`, `reciprocal_rank` (MRR = ortalaması), `graded_ndcg_at_k`. Hepsinde `min_rel` parametresi var.
- `recall_at_k` paydası: qrels'teki ilgili sayısı. Etiketlenmemiş görsel 0 sayılır (pooling varsayımı, bkz. A4 kontrol örneklemi).
- `graded_ndcg_at_k` için IDCG gerçek qrels'ten hesaplanır. Böylece eski nDCG'deki "≥k ilgili var" varsayımı (risk 7) yeni metrikte yok.
- İlgili belgesi olmayan sorgu NaN döner ve raporda sayılır.
- Eski kategori fonksiyonlarına dokunulmaz. Eski P@10 ile karşılaştırma için `precision_at_k` için qrels'li bir sürüm eklenir.
- Testler: elle hesaplanmış küçük vakalar, k > sonuç sayısı, boş qrels, `min_rel` 1 ve 2.

**A2. `feat(eval): golden set schema and IO`.** Dosyalar: `eval/golden.py`, `tests/test_golden.py`.
- Tipli dataclass'lar: `GoldenQuery(qid, text, category, provenance)`, `Judgment(qid, image, grade|None, annotator, caption_seen, ts)`, `CorpusEntry(image, pathology, split_class)`.
- Okuma/yazma: JSONL; qrels TREC formatında (`qid 0 image grade`).
- `golden_dir` doğrulaması: eksik dosya → `DataNotFoundError`. Sessiz fallback yok.

**A3. `feat(eval): query-level dev/test split + query leakage check`.** Dosyalar: `eval/splits.py`, `tests/test_splits.py`.
- Kategoriye göre tabakalı bölme, seed 42. `dev_frac` parametresi verilmezse sorgu sayısı ≤50 için 0.5, aksi halde 0.3 kullanılır. Pilot sorgular dev'de kalır. `splits.json` sorgu yazılır yazılmaz commit edilir ve sonra değişmez.
- Neden sorgu düzeyinde: CSV'de video/kaynak kimliği yok (sütunlar §7.8.2'de; `subset` yalnız kaynak türünü veriyor). PLAN'daki "video bazlı split" bu veriyle yapılamıyor; bu durum ADR-0002'de yazılacak.
- Sızıntı kontrolü: dev ve test arasında normalize edilmiş metni aynı olan sorgu hata verir. Token Jaccard ≥0.8 olan sorgu çiftleri uyarı olarak raporlanır.

**A4. `feat(eval): pooling`.** Dosyalar: `eval/pooling.py`, `tests/test_pooling.py`.
- `build_pool(runs: dict[system, dict[qid, ranked_ids]], depth, seed)` her sistemin top-`depth` sonucunun birleşimini alır. Sıra seed'li karıştırılır ve sistem kimliği annotator'dan gizlenir (kör etiketleme).
- **Kontrol örneklemi:** her sorgu için havuz dışından seed'li `n_random` (varsayılan 5) görsel eklenir. Bunlardan çıkan ilgili oranı, "etiketlenmemiş = 0" varsayımının ne kadar ihlal edildiğini ölçer.
- `overlap_subset(pool, frac=0.2, seed)`: ikinci annotator'ın etiketleyeceği alt küme. Sorgu bazında tabakalı.
- Havuz büyüklüğü tahmini: 100 sorgu × derinlik 10 × 4 sistem, örtüşmeyle birlikte en fazla ~2,500 çift. Derinlik CLI parametresi.

**A5. `feat(eval): Cohen's kappa (unweighted + linear weighted)`.** Dosyalar: `eval/agreement.py`, `tests/test_agreement.py`.
- Testlerde bilinen ders kitabı değerleri ve kenar durumlar kullanılır: tam uyum, tek sınıf (beklenen uyum 1 olunca payda 0), `unsure` çiftlerinin dışlanması.
- Ek çıktılar: uyuşmazlık listesi ve karışıklık matrisi.
- Uzlaştırma: uyuşmazlıklar `adjudication.jsonl` dosyasına çözülerek yazılır. Final qrels = annotator A'nın etiketleri, üzerine uzlaştırma kararları. Kappa uzlaştırmadan **önce** hesaplanır.

**A6. `feat: Streamlit labeling tool`.** Dosyalar: `eval/labeling.py` (durum ve kayıt), `label_app.py` (yalnız UI), `tests/test_labeling.py` + AppTest smoke testi.
- Çalıştırma: `uv run streamlit run label_app.py -- --golden eval_data/golden_v0 --annotator A`.
- Ekranda sorgu metni, görsel ve 0/1/2/unsure butonları var. "Caption göster" butonu caption'ı CSV'den çeker ve `caption_seen=true` kaydeder.
- Sistem adı ve sıra gösterilmez. Annotator B yalnızca overlap alt kümesini görür.
- Kayıt append-only JSONL'a yapılır (`judgments/<annotator>.jsonl`), aynı çift için son yazılan geçerli. Kaldığı yerden devam eder, ilerleme göstergesi var.
- Görsel yolu adla çözülür (`data/images` recursive). Bulunamazsa açık hata verir, başka görsele geçmez.

**A7. `feat(eval): run_eval`.** Dosyalar: `eval/run_eval.py`, `scripts/run_eval.py`, `tests/test_run_eval.py`.
- Komut: `uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split dev --systems clip,quilt,clip+rerank,quilt+rerank --k 10`.
- Sistemler `pipeline.run_text_search` ve `reranker.rerank_results` üzerinden koşar; app ile aynı α/β/pool değerleri kullanılır. Domain gate değerlendirmede **kapalı**. Encoder ve indeks enjekte edilir, testler sahte indeksle çalışır.
- Çıktı: `reports/eval_<golden>_<split>_<sha>.json` + `.md`. İçerik:
  - komut, config, seed, git SHA ve paket sürümleri;
  - strict/lenient Hit/Recall/MRR/nDCG@k ve P@k;
  - kategori kırılımı;
  - paired bootstrap %95 CI (seed'li, risk 8 için);
  - `caveats` alanı (QuiltNet notu dahil).
- **Test split koruması:** `--split test` ancak `--i-know-this-is-test` bayrağıyla koşar ve rapora "test" damgası basılır. Ayar (α/β, havuz) yalnızca dev'de yapılır.
- **Döngüsellik kontrolü:** reranker satırları ayrıca yalnız `caption_seen=false` etiketler üzerinden raporlanır. Ek olarak etiket ile Jaccard(sorgu, caption) arasındaki Spearman korelasyonu raporlanır; yüksekse "etiket caption sözcüklerine bağımlı" uyarısı verilir.

**A8. `feat(eval): held-out corpus selection + leakage checks`.** Dosyalar: `eval/heldout.py`, `tests/test_heldout.py`.
- `select_heldout(scan, disk, ids, rule)` → `CorpusEntry` listesi. v0 kuralı: diskte var, yalnız `val`, `ids.json`'da yok. Mevcut `corpus_analysis.scan_csv`, `list_image_names` ve `prepare_quilt.parse_pathology` yeniden kullanılır.
- Sızıntı fonksiyonları:
  - (a) golden ∩ ana korpus `ids` = ∅;
  - (b) golden görsellerin hiçbirinin CSV'de `train` satırı yok;
  - (c) yakın kopya: golden embedding'leri ile ana korpus embedding'leri arasındaki max cosine τ'yu aşarsa işaretlenir. τ veriye bakılmadan önce sabitlenir: 0.95, CLIP ve QuiltNet ayrı ayrı;
  - (d) golden içi yinelenen görseller;
  - (e) `assert_disjoint(train_ids, golden_ids)`: R4'te eğitim manifesti için tekrar kullanılacak.
- Testler sahte vektör ve adlarla.

**A9. `docs: golden set protocol`.** `docs/golden_set.md`: etiket kuralı, iş akışı, dosya şeması, varsayım sınırları. Etiketleme süresi tahmini de burada.
- İlk tahmin 10–15 sn/etiket. Pilot (20 sorgu × ~25 çift ≈ 500 etiket) ≈ 1.5–2 saat.
- 100 sorgu (~2,500 etiket) A için ≈ 7–10 saat, B için (%20) ≈ 1.5–2 saat.
- Pilotta zaman damgalarından gerçek hız ölçülür ve tahmin güncellenir.

## B. Veri gerektiren işler (yerelde koşar, sırası önemli)

| # | İş | Kim | Çıktı (repo / gitignore) |
|---|---|---|---|
| B1 | `scripts/build_golden_corpus.py --rule val_only_on_disk --out eval_data/golden_v0`. Sayılar §7.8.1 ile birebir tutmalı (928; kategori tablosu) | ben | `corpus_manifest.jsonl` + `manifest_meta.json` (kural, kaynak dosya boyutu/mtime, sayılar) |
| B2 | 928 görsel için CLIP ve QuiltNet embedding'leri: mevcut `build_embeddings --captions_jsonl <manifest> --out_dir artifacts/golden_v0/<enc>` + `faiss_search build` | ben (CPU) | `artifacts/golden_v0/*` (gitignore) |
| B3 | Sızıntı raporu (A8 a–d) | ben | `reports/leakage_golden_v0.json` |
| B4 | Önce **20 sorguluk pilot** (5 kategori × 4), pilot B5–B7'den geçer, sonra **100'e** büyütülür (`--n_queries`, `--depth` parametre). Sorgular, kavram listesinden, golden sonuçlarına ve caption'lara bakılmadan yazılır. İstersen kategori bazında aday listesi taslağını ben çıkarırım, final seçim senin. Ardından `splits.json` | sen (+ben taslak) | `queries.jsonl`, `splits.json`: **havuzdan önce commit** |
| B5 | Pooling (4 sistem, derinlik 10, +5 rastgele, seed 42) + overlap alt kümesi | ben | `pool.jsonl`, `overlap.jsonl` |
| B6 | Etiketleme: A tüm havuz, B overlap (%20) | sen + 2. kişi | `judgments/A.jsonl`, `judgments/B.jsonl` |
| B7 | Kappa raporu → uzlaştırma → final qrels | ben + sen | `reports/agreement_golden_v0.json`, `adjudication.jsonl`, `qrels.tsv` |
| B8 | `run_eval` dev. Test yalnızca R2 sonunda **bir kez** koşar. Eski "0.539 → 0.609" iddiası golden set P@10 ile yeniden hesaplanır; korpus farklı olduğu için sayılar doğrudan karşılaştırılamaz, raporda bu yazılır | ben | `reports/eval_golden_v0_dev_*.{json,md}` |
| B9 | `docs/STATUS.md` güncellemesi (yeni §9) | ben | – |

## Kapsam dışı (R2'nin ayrı issue'ları)

Domain gate ROC kalibrasyonu, HNSW, ADR yazımı (R3) ve golden setin zip'lerle genişletilmesi. Bu planın kodu bunlara hazır ama bu işleri yapmıyor.

## Riskler / açık notlar

- Annotator patolog değil. 0/1/2 etiketleri gürültülü olabilir; bunu kappa ve unsure oranı ölçer, raporda açıkça yazılır.
- 928 görsellik küçük korpusta kolay sorgular tavana vurabilir. Kategori başına 20–161 görsel var; dağılım raporlanır.
- Çok-caption'lı görseller (§8-4): golden manifestte reranker için CSV'deki **ilk** caption deterministik olarak seçilir (seed'e bağlı değil), tüm caption'lar listede saklanır. Ana korpustaki "son satır" davranışına dokunulmaz.

## Doğrulama

- A işleri: her commit'te `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`. CI yeşil kalmalı (veri olmadan çalışır).
- `label_app.py` için AppTest smoke testi (sahte golden dizin, sahte görsel) ve bir kez elle `streamlit run`.
- `run_eval` uçtan uca sahte indeksle test edilir. B8'de rapor JSON'unda komut, seed, SHA ve `caveats` alanlarının bulunduğu kontrol edilir.
- B1 sayıları `reports/corpus_analysis.json` → `sec_7_8_val_only_on_disk` ile karşılaştırılır.
- Push yok; yalnızca yerel commit.
