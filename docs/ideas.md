# Fikirler (plan dışı)

Plan dışı ama değerlendirilmeye değer işler. Buradaki bir madde, plana alınmadan uygulanmaz.

## R4 sonrası: eğitim kümesini diskteki tüm train-only görsellere genişletmek

- **Durum:** R4'te eğitim kümesi yalnızca ana korpus embedding'i olan 38,084 train-only görselden seçiliyor (diskteki 64,369'un ~%59'u). Kalan 26,285 görselin CLIP/QuiltNet embedding'i yok. Bu embedding'ler golden'a yakın kopya elemesi için gerekli, ve yerel CPU'da üretmek ~2 saat 10 dakika sürüyor (500 görsellik ölçüm, `artifacts/r4_train_candidates/*/timing.json`).
- **Koşul:** R4 ölçek eğrisi (10k → 30k → full; full **çalıştırılmadı**, ADR-0004 Ek 1) dev'de umut verici görünürse.
- **Yöntem:** `scripts/embed_sharded.py` (parçalı, devam edilebilir, hazır) Colab GPU'da çalıştırılır. Ardından aynı eleme kuralları uygulanarak `r4_train_v1` manifesti kurulur. Yeni eğitim kümesi yeni bir ölçek noktası (~60K) olarak raporlanır. Mevcut ölçekler değişmez.
- **Dikkat:** eklenecek 26K görsel rastgele bir alt küme değil, ana korpusa girmemiş görseller. Bu yüzden dağılımı (kategori, alt küme) mevcut kümeyle karşılaştırılmadan ölçek etkisi diye yorumlanmamalı.
