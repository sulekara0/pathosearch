# ADR-0005: Ürün modeli (R6 arama servisi)

Status: accepted (2026-10-28) · Context: R6, FastAPI servisi · Evidence: [docs/r3_baseline.md](../r3_baseline.md), [docs/R4_report.md](../R4_report.md)

## Bağlam

R6, mevcut arama çekirdeğini (`pipeline`, `encoders`, `faiss_search`, `domain_gate`, `reranker`) bir HTTP servisi olarak dışarı açar. Serviste tek bir arama modeli, bir rerank varsayılanı ve bir domain gate eşiği sabitlenmelidir. Bu belge yeni bir ölçüm yapmaz; yalnız R3 ve R4'te ölçülmüş sonuçlara dayanarak ürün seçimini kaydeder.

## Karar

1. **Arama modeli: QuiltNet B-32** (`hf-hub:wisdomik/QuiltNet-B-32`). Genel CLIP ve LoRA sürümleri serviste yoktur.
2. **Rerank varsayılan kapalı.** İstekte `rerank: true` ile açılabilir (V5b ağırlıkları, `pipeline.RERANK_*`).
3. **Domain gate: CLIP metin kulesi, eşik 0.80** (`domain_gate.DEFAULT_THRESHOLD`). Eşik `PATHOSEARCH_GATE_THRESHOLD` ile değiştirilebilir, varsayılan değişmez. Görsel sorgulara gate uygulanmaz.
4. **Geri bildirim tablosu (`feedback`) değerlendirme etiketi değildir.** Golden set'e, qrels'e veya ayar kararlarına girmez.

## Gerekçe

- **QuiltNet:** golden_v0 lenient nDCG@10, test split: `quilt − clip` +0.214 [+0.009, +0.413]; `lora-30k-s0 − quilt` −0.316 [−0.524, −0.108] (paired bootstrap %95 GA, n=1000, seed 42; [R4 §4.2](../R4_report.md#42-fark-tablosu-lenient-birincil-eşik--ga-sıfırı-dışlıyor)). Dev split'te de aynı yön. Önceden kayıtlı LoRA hipotezi desteklenmedi ([R4 §4.3](../R4_report.md#43-birincil-karşılaştırma-sonucu)).
- **Rerank kapalı:** QuiltNet'te rerank'in P@10 ve nDCG@10 farkı dev'de de test'te de sıfırı içeriyor ([R3 §3 (b), (e)](../r3_baseline.md#3-dev-bulgularının-testte-durumu)). Ölçülmüş bir kazanç olmadan varsayılan açık tutulmaz.
- **Gate:** mevcut tek gate uygulaması budur. Alternatifi (kalibre gate) R2'den açık kalan iştir ve R6 kapsamında değildir.
- **Geri bildirim:** kullanıcılar uzman değil, sorgu seçimi kontrolsüz ve kayıt sorgu metnini saklamaz; bu veriden ilgi etiketi türetmek ölçüm kurallarını (CLAUDE.md) ihlal eder.

## Sınırlar

- **QuiltNet eğitim verisi belirsiz.** Golden görsellerinin QuiltNet eğitiminde görülüp görülmediği bilinmiyor (STATUS §7.8.3, §7.10). Bu karar bir *ürün seçimidir*; "QuiltNet genel olarak daha iyi" iddiası değildir.
- **Domain gate eşiği kalibre edilmedi.** 0.80 sezgisel; 10 elle seçilmiş sorguyla denendi, ROC/PR yok, 0.8015 vs 0.8026 gibi kıl payı vakalar var (STATUS §5, §7.9.5). Sınır servis tarafından `/health` yanıtında da bildirilir.
- **Görsel→görsel arama nicel olarak ölçülmedi** (STATUS §7.9.5). Servis bu modu sunar ama kalitesi hakkında iddia yoktur.
- Serviste iki model yüklenir (arama: QuiltNet, gate: CLIP). Bellek ve cold start bu yüzden tek modelden yüksektir.
- **Docker ve PostgreSQL yerelde doğrulanmadı.** Geliştirme ortamında Docker yok. `Dockerfile` yalnızca GitHub Actions'taki `docker-build` job'ında build edilir (çalıştırılmaz); imaj boyutu oradan okunur. Testler SQLite üzerinde koşar; PostgreSQL ile `docker compose` akışı elle doğrulanmadı.
- Sistem eğitim/araştırma amaçlıdır; klinik teşhis aracı değildir.
