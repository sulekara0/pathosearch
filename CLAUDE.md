# PathoSearch (repo: pathosearch)

Histopatoloji görsel/metin arama sistemi (Quilt-1M, CLIP/QuiltNet, FAISS). Hedef: ölçülmüş, test edilmiş, canlı yayında bir multimodal arama + ajan platformu.

## Çalışma şekli
- Yol haritası: @docs/PLAN.md · Güncel durum: @docs/STATUS.md
- Kullanıcı yeni mezun bir bilgisayar mühendisi ve bu projeyi mülakatta savunacak. Her değişiklikte **ne yaptığını ve neden** kısaca açıkla.
- Yanıtlar Türkçe. Kod, değişken adları, commit mesajları ve docstring'ler İngilizce.
- Büyük değişiklikten önce Plan Mode'da plan sun. Belirsizlik varsa tahmin etme, sor.
- Bir seferde tek issue/aşama. Küçük, geri alınabilir commit'ler.

## Kod kuralları
- Çekirdek mantık `src/pathosearch/` altında. `app.py` sadece UI, `scripts/` sadece ince CLI sarmalayıcı.
- Mutlak yol, `sys.path` hack'i, `os.chdir` yok. Yollar `pathlib` + config/env üzerinden.
- Geliştirme ortamı Windows. FAISS'e non-ASCII yol vermek sorun çıkardı, göreli/ASCII yol kullan.
- Tüm yeni fonksiyonlarda tip ipucu ve pytest testi. `ruff` ve `mypy` geçmeden commit yok.
- Yeni bağımlılık eklemeden önce sor. Sürümler `uv.lock` ile sabit.
- Veri dizini yoksa **sessizce başka dizine düşme**, açık hata ver.

## Git kuralları
- Commit mesajlarına **asla** `Co-Authored-By`, `Claude-Session`, "Generated with Claude Code" veya benzeri atıf satırı ekleme. PR açıklamalarına da ekleme. Tek yazar, repo sahibidir.
- Commit mesajı: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`), İngilizce, kısa başlık + gerekirse "neden" gövdesi.
- `git config user.name/email` değerlerini değiştirme, `--author` kullanma.
- Kullanıcı istemedikçe `push`, `--force`, `reset --hard`, geçmiş yeniden yazma yapma.

## Ölçüm kuralları (en önemli)
- Metrik iddiası, tekrar üretilebilir komut + config + seed olmadan yazılmaz.
- Test split'e parametre ayarı için bakılmaz. Ayar sadece dev'de yapılır.
- Golden set görselleri ve yakın kopyaları eğitim verisine girmez.
- Reranker gibi caption kullanan bileşenlerin değerlendirmesinde etiket sızıntısı/döngüsellik kontrol edilir.
- Sonuç tabloları `reports/` altına, komutuyla birlikte kaydedilir.

## Yapma
- Veri, embedding, FAISS indeksi, `.env`, API anahtarı commit etme (büyük dosyalar Hugging Face Hub'a).
- Plan dışı özellik ekleme. Fikirler `docs/ideas.md`'ye.
- Çıktılarda klinik teşhis iması yok: sistem eğitim/araştırma amaçlıdır.

## Komutlar (R1 sonrası doldurulacak)
- Kurulum: `uv sync`
- Test/lint/tip: `uv run pytest` · `uv run ruff check .` · `uv run mypy src`
- Uygulama: `uv run streamlit run app.py`