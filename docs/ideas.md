# Fikirler (plan dışı)

Plan dışı ama değerlendirilmeye değer işler. Buradaki bir madde, plana alınmadan uygulanmaz.

## R4 sonrası: eğitim kümesini diskteki tüm train-only görsellere genişletmek

- **Durum:** R4'te eğitim kümesi yalnızca ana korpus embedding'i olan 38,084 train-only görselden seçiliyor (diskteki 64,369'un ~%59'u). Kalan 26,285 görselin CLIP/QuiltNet embedding'i yok. Bu embedding'ler golden'a yakın kopya elemesi için gerekli, ve yerel CPU'da üretmek ~2 saat 10 dakika sürüyor (500 görsellik ölçüm, `artifacts/r4_train_candidates/*/timing.json`).
- **Koşul:** R4 ölçek eğrisi (10k → 30k → full; full **çalıştırılmadı**, ADR-0004 Ek 1) dev'de umut verici görünürse.
- **Yöntem:** `scripts/embed_sharded.py` (parçalı, devam edilebilir, hazır) Colab GPU'da çalıştırılır. Ardından aynı eleme kuralları uygulanarak `r4_train_v1` manifesti kurulur. Yeni eğitim kümesi yeni bir ölçek noktası (~60K) olarak raporlanır. Mevcut ölçekler değişmez.
- **Dikkat:** eklenecek 26K görsel rastgele bir alt küme değil, ana korpusa girmemiş görseller. Bu yüzden dağılımı (kategori, alt küme) mevcut kümeyle karşılaştırılmadan ölçek etkisi diye yorumlanmamalı.

## R6 sonrası: Streamlit'i API istemcisine çevirmek

- **Durum:** PLAN R6 "Streamlit, API'yi çağıran istemciye döner" diyor. Minimal R6'da (ADR-0005) yapılmadı: `app.py` hâlâ `pathosearch.pipeline`'ı doğrudan çağırıyor.
- **Engel:** API yalnızca QuiltNet sunuyor ve görsel döndürmüyor. Uygulamanın Karşılaştırma modu (CLIP vs QuiltNet) ve görsel kartları API'ye taşınamaz; ya bu modlar uygulamada yerel kalır ya da API kapsamı genişler.
- **Yöntem:** Tek Model ve Görsel→Görsel modları `httpx` ile `/search/text` ve `/search/image`'ı çağırır; görseller yine yerel diskten gösterilir.

## R6 sonrası: artifact'ları Hugging Face Hub'dan çekmek

- **Durum:** Servis `faiss_index.bin`, `ids.json` ve caption JSONL'ını salt okunur volume'dan okur; dosyalar yerelde olmalı.
- **Yöntem:** açılışta sabit bir revision'dan (`huggingface_hub`) indirip sha256 doğrulamak. Yeni bağımlılık ve lisans kontrolü gerekir.
