# Golden set protokolü (R2)

Bu belge insan etiketli golden set'in kuralını, iş akışını, dosya şemasını ve bilinen sınırlarını anlatır. Plan için [R2_PLAN.md](R2_PLAN.md), durum için [STATUS.md](STATUS.md).

> Sistem eğitim/araştırma amaçlıdır; klinik teşhis aracı değildir. Etiketler bir görselin **sorguyla ne kadar eşleştiğini** söyler, bir tanıyı doğrulamaz.

## 1. Neden var

Eski ölçüm "ilgili" sayısını metadata kategorisinden türetiyordu (`pathology ∈ expected`). Bu uzmanlık dalı düzeyinde kaba, `unknown` görselleri hiç ilgili saymıyor ve caption kullanan reranker ile döngüsel (STATUS §4 risk 1). Golden set bunun yerine her (sorgu, görsel) çifti için bir insanın verdiği derece (0/1/2) kullanır.

## 2. Etiket kuralı

| Etiket | Kural |
|---|---|
| **2** | Sorgunun doku/organını **ve** sorgulanan bulguyu/antiteyi gösteriyor (annotator'ın görebildiği kadarıyla) |
| **1** | Doğru organ/uzmanlık veya yakın bir bulgu, ama sorgulanan antite değil ya da kısmen eşleşiyor |
| **0** | İlgisiz, histoloji değil veya sorguyla çelişiyor |
| **unsure** | Karar verilemedi. Qrels'e girmez, sayısı raporlanır |

Örnekler (kuralı göstermek için kurgusal; gerçek golden set görseli değil):

| Sorgu | Görselde ne var | Etiket | Gerekçe |
|---|---|---|---|
| "squamous cell carcinoma of skin" | Deri dokusunda keratin incileri olan skuamöz tümör adaları | 2 | Doku ve antite eşleşiyor |
| "squamous cell carcinoma of skin" | Deride iltihabi hücre infiltrasyonu, tümör yok | 1 | Doğru organ, sorgulanan antite yok |
| "squamous cell carcinoma of skin" | Akciğerde skuamöz karsinom | 1 | Yakın bulgu, farklı organ |
| "squamous cell carcinoma of skin" | Bağırsak mukozası, normal | 0 | İlgisiz |
| "alveolar spaces" | Şema, grafik veya histoloji dışı fotoğraf | 0 | Histoloji değil |
| "glandular structures" | Doku çok bulanık, ne olduğu seçilemiyor | unsure | Karar verilemedi |

Kurallar:
- Annotator caption'ı başta görmez. Gerekirse "Caption göster" ile açar ve etiket `caption_seen=true` olarak kaydedilir.
- Etiketleme **kör**: hangi sistemin hangi görseli getirdiği ve sırası gösterilmez.
- Binary metrikler iki eşikte raporlanır: **strict** (derece ≥ 2) ve **lenient** (derece ≥ 1). nDCG kademeli kazanç kullanır (`2^derece − 1`). Birincil eşik R3'te ADR-0001 ile seçilir.
- Annotator patolog değildir. Etiketler gürültülü olabilir; bunu kappa ve unsure oranı ölçer.

## 3. İş akışı

Her adımın hangi kodla yapıldığı ve **şu an CLI'nin var olup olmadığı**:

| Adım | Ne yapılır | Kod | Durum |
|---|---|---|---|
| B1 | Held-out korpus manifesti (`val_only_on_disk` kuralı) | `eval/heldout.py: select_heldout` | Modül var, `scripts/build_golden_corpus.py` henüz yok |
| B2 | 928 görsel için CLIP ve QuiltNet embedding + FAISS indeksi (`artifacts/golden_v0/<enc>/`) | `build_embeddings`, `faiss_search build` | Mevcut araçlar; golden manifest ile çalıştırılacak |
| B3 | Sızıntı raporu | `eval/heldout.py` (a–d) | Fonksiyonlar var, rapor betiği yok |
| B4 | Sorguların yazılması, dev/test bölünmesi | `eval/splits.py` | Sorgular insan işi; `splits.json` havuzdan **önce** commit edilir |
| B5 | Pooling (4 sistem, derinlik 10, +5 rastgele) ve overlap alt kümesi | `eval/pooling.py` | Fonksiyonlar var, CLI yok |
| B6 | Etiketleme | `label_app.py` | Hazır |
| B7 | Kappa, uzlaştırma, final qrels | `eval/agreement.py` | Fonksiyonlar var, CLI yok |
| B8 | Değerlendirme | `scripts/run_eval.py` | Hazır |

### Etiketleme aracı

```
uv run streamlit run label_app.py -- --golden eval_data/golden_v0 --annotator A
```

- Annotator **A** bütün havuzu, diğer annotator'lar (örn. **B**) yalnızca `overlap.jsonl` alt kümesini görür.
- Kısayollar: `0`, `1`, `2` etiket, `u` geri al. "Emin değilim" ve "Caption göster" butonla.
- Kayıt `judgments/<annotator>.jsonl` dosyasına eklenir (append-only, aynı çift için son satır geçerli). Kapatıp açınca kaldığın yerden devam eder.
- Geri al, log'u silmez; önceki çifti eski etiketiyle tekrar gösterir. Yeniden etiketlersen yeni satır geçerli olur.
- Görsel adla bulunur (`--images-dir`, varsayılan `PATHOSEARCH_IMAGES_DIR`, alt klasörler dahil). Bulunamazsa ya da aynı ad iki dosyada varsa açık hata verir.

Gerçek veri olmadan denemek için:

```
uv run python scripts/make_fake_golden.py --out artifacts/fake_golden
```

Komut, sahte görseller ve caption CSV'si üretir ve çalıştırılacak `streamlit` komutunu yazdırır.

### Değerlendirme

```
uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split dev --systems clip,quilt,clip+rerank,quilt+rerank --k 10
```

- Çıktı: `reports/eval_<golden>_<split>_<sha>.json` ve `.md` (komut, config, seed, git SHA, paket sürümleri, strict/lenient Hit/Recall/MRR/nDCG/P, kategori kırılımı, seed'li paired bootstrap %95 GA, `caveats`).
- Domain gate değerlendirmede kapalıdır. Rerank, uygulamayla aynı α/β/havuz değerlerini kullanır.
- **Test split koruması:** `--split test` yalnızca `--i-know-this-is-test` ile koşar ve rapora TEST damgası basılır. α/β ve havuz ayarı yalnızca dev'de yapılır; test R2 sonunda bir kez koşar.
- **Döngüsellik kontrolü:** tüm sistemler ayrıca yalnızca `caption_seen=false` çiftler üzerinden skorlanır. Caption'ı açılan (veya açılıp açılmadığı bilinmeyen) çiftler qrels'ten **ve** sistemlerin sıralı listelerinden çıkarılır; ilgisiz sayılmazlar. Liste kısalırsa kalan uzunlukta skorlanır. Rapor çıkarılan çift sayısını/oranını ve kalan sorgu sayısını yazar; oran %30'u aşarsa uyarı verir. Etiket ile sorgu–caption sözcük örtüşmesi arasındaki Spearman korelasyonu raporlanır; |ρ| ≥ 0.5 ise uyarı çıkar (eşik veriye bakmadan sabitlenmiştir).

## 4. Dosya şeması

`eval_data/golden_v0/` (repoya girer) ve gitignore'da kalanlar:

| Dosya | İçerik |
|---|---|
| `queries.jsonl` | `qid, text, category, provenance` |
| `splits.json` | `seed, dev_frac, dev: [qid], test: [qid]` |
| `corpus_manifest.jsonl` | `image, pathology, split_class` (caption, görsel, embedding yok) |
| `pool.jsonl` | `qid, image, order, source (pooled/random), systems` |
| `overlap.jsonl` | `pool.jsonl` ile aynı şema; B için alt küme |
| `judgments/<id>.jsonl` | `qid, image, grade (0/1/2/null), annotator, caption_seen, ts` |
| `adjudication.jsonl` | Çözülmüş uyuşmazlıklar (`annotator="adjudicated"`) |
| `qrels.tsv` | TREC biçimi: `qid 0 image grade` |
| `artifacts/golden_v0/<encoder>/` | Embedding ve indeks (gitignore) |

Final qrels = annotator A'nın etiketleri, üzerine uzlaştırma kararları. Kappa uzlaştırmadan **önce** hesaplanır; `unsure` çiftleri kappa'dan dışlanır ve sayılır.

## 5. Varsayımlar ve sınırlar

- **Held-out varsayımı doğrulanmadı.** CSV `val` bölümünün QuiltNet eğitiminde görülmediği hiçbir kaynakta doğrulanmadı (STATUS §7.8.3). Golden setin geçerliliği buna bağlı değil (etiketler insandan), ama CLIP–QuiltNet karşılaştırmasının adil olup olmadığı buna bağlı. Bu yüzden her raporda sabit bir `caveats` notu var.
- **Pooling varsayımı:** etiketlenmemiş görsel ilgisiz sayılır. Havuz dışından eklenen rastgele kontrol örneği, bu varsayımın ne kadar yanlış olduğunu ölçer.
- **Split sorgu düzeyinde.** CSV'de video/kaynak kimliği yok, bu yüzden PLAN'daki "video bazlı split" yapılamıyor (ADR-0002'de yazılacak). Dev ve test arasında aynı normalize metinli sorgu hata verir, token Jaccard ≥ 0.8 olan çiftler uyarı olur.
- **Küçük korpus.** v0 adayı diskteki 928 yalnız-`val` görsel (STATUS §7.8.1): %51'i `unknown`, yalnızca 5 kategoride ≥ 20 görsel var. Kolay sorgular tavana vurabilir; kategori dağılımı raporlanır.
- **Çok-caption'lı görsel:** reranker için CSV'deki **ilk** caption kullanılır (seed'e bağlı değil). Ana korpustaki "son satır" davranışına dokunulmaz.
- **Sızıntı kontrolleri** (`eval/heldout.py`): golden ∩ ana korpus `ids` boş olmalı; golden görsellerin CSV'de `train` satırı olmamalı; golden ile ana korpus arasında cosine > 0.95 olan yakın kopya işaretlenir (eşik CLIP ve QuiltNet için ayrı uygulanır, veriye bakmadan sabit); golden içinde yinelenen görsel olmamalı. `assert_disjoint`, R4'te eğitim manifesti için de kullanılacak.

## 6. Etiketleme süresi tahmini

Bunlar **ölçülmemiş ilk tahminlerdir**: etiket başına 10–15 sn varsayımı.

| Kapsam | Etiket sayısı | Tahmini süre |
|---|---|---|
| Pilot: 20 sorgu × ~25 çift | ~500 | 1.5–2 saat |
| Tam set, annotator A: 100 sorgu | ~2,500 | 7–10 saat |
| Annotator B (overlap %20) | ~500 | 1.5–2 saat |

Havuz büyüklüğü sorgu × derinlik × sistem sayısına ve örtüşmeye bağlıdır (en fazla ~2,500 çift). Pilotta zaman damgalarından gerçek hız ölçülür; etiketleme aracı oturum içi ortalama sn/etiket gösterir. Ölçülen değer bu tabloyu güncellemek için kullanılır.
