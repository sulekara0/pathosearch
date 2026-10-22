# ADR-0001: Birincil metrik ve ilgili çifti olmayan sorgular

Status: accepted (2026-10-22) · Context: R2, golden_v0 · Evidence: [reports/agreement_golden_v0.md](../../reports/agreement_golden_v0.md)

## Zamanlama

Bu karar **hiçbir sistemin golden_v0 üzerindeki metrikleri görülmeden önce** verildi. `reports/` altında henüz `eval_golden_v0_*` raporu yok; `run_eval` bu karardan önce hiç gerçek veriyle koşturulmadı. Karar yalnızca etiket uyum raporuna (`agreement_golden_v0`) ve qrels dağılımına dayanır. Sonuçlar göründükten sonra birincil metrik ya da dışlama kuralı değiştirilirse yeni ADR gerekir.

## Bağlam

Derecelendirme 0/1/2 ölçeğindedir. İki eşik raporlanır: **lenient** (grade ≥ 1) ve **strict** (grade ≥ 2).

- Strict ikili kappa **0.11** (CI pair [-0.06, 0.42], CI query [-0.07, 0.56]); lenient ikili kappa 0.62. İki annotator "tam ilgili" ayrımında uyuşmuyor (marjinaller: A 10 adet grade 2, B 3 adet).
- Strict ilgili çiftlerin **28/57'si iki sorgudan** geliyor; strict ortalama birkaç sorgunun davranışına bağlı.
- **7/20 sorguda hiç strict ilgili çift yok.**

## Karar

1. **Birincil metrik lenient (rel ≥ 1), strict ikincil.** Tablolarda lenient önce gelir. Strict raporlanır ama sistem karşılaştırma sonucu lenient üzerinden okunur. Gerekçe: strict etiketin güvenilirliği düşük (kappa) ve az sorguya yaslanıyor.
2. **İlgili çifti olmayan sorgular metrik ortalamasından dışlanır.** Bu sorgular NaN olur, **0 sayılmaz** (0 saymak, etiketlenebilir ilgili görsel olmamasını sistemin hatası gibi gösterir). Bootstrap da yalnızca kalan sorgular üzerinden çalışır.
3. **Her rapor tablosunun altında** şu satır bulunur: `N sorgu üzerinden, M sorgu dışlandı (qid listesi)`. Dışlanan qid listesi JSON raporunda eşik başına tutulur.
4. **Havuz sınırı.** "İlgili yok", *havuzda ilgili bulunamadı* demektir; görüntü korpusunda ilgili görsel olmadığı anlamına gelmez. Yargılar havuzlanmıştır (`pool.jsonl`), havuza girmeyen görsel ilgisiz sayılır. Bu yüzden dışlama kuralı mutlak bir "ilgili yok" iddiası değildir ve metrikler havuz kapsamına bağlıdır.

## Sonuçlar

- Dışlama ortalamayı, ilgili çifti bulunan sorgulara koşullar. Strict'te bu 20 sorgunun yalnızca 13'ü olabilir; strict güven aralıkları geniş çıkması beklenir.
- Kategori kırılımında ilgili çifti olmayan kategoriler `n/a` görünür.
- Sorgu dışlaması sistemden bağımsızdır (yalnız qrels'e bağlı), tüm sistemler aynı sorgu kümesinde karşılaştırılır.

## Sınırlar

- Lenient de gürültülüdür (A–B gözlenen uyum 0.82, kappa CI geniş). Birincil olması "doğru" olduğu değil, daha güvenilir olduğu anlamına gelir.
- Annotator'lardan biri LLM'dir (`claude`), ikisi de patolog değildir (bkz. agreement raporu).
- Dışlama kuralı küçük kategorileri daha çok etkileyebilir (ADR-0003 ek: eleme sonrası küçük kategoriler).
