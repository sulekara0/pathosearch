from pathosearch.train.caption_stats import caption_stats
from pathosearch.train.manifest import caption_for


def test_caption_stats_counts() -> None:
    caps = {
        "a.jpg": ["short", "a much longer caption that clearly exceeds forty characters"],
        "b.jpg": ["tiny", "Tiny "],  # normalises to the same caption as c.jpg's
        "c.jpg": ["tiny"],
        "d.jpg": ["another caption that is definitely longer than forty chars"],
    }
    s = caption_stats(caps, short_chars=40, top_n=1, caption_seed=42)
    assert s["images"] == 4 and s["captions"] == 6
    assert s["short_captions"] == 4  # short, tiny, Tiny, tiny
    assert s["images_any_short"] == 3 and s["images_all_short"] == 2
    expected_epoch0 = sum(len(caption_for(n, c, 0, seed=42).strip()) < 40 for n, c in caps.items())
    assert s["epoch0_short"] == expected_epoch0
    assert s["top_images"] == [2] and "top_preview" not in s  # no caption text by default
    assert s["images_any_top"] == 2 and s["images_all_top"] == 2


def test_caption_stats_truncates_top_caption_text() -> None:
    caps = {"a.jpg": ["x" * 100], "b.jpg": ["x" * 100]}
    s = caption_stats(caps, top_n=1, preview_chars=60)
    assert s["top_preview"] == ["x" * 60] and s["top_images"] == [2]
