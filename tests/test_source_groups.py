import pytest

from pathosearch.data.source_groups import (
    csv_split_by_source,
    group_key,
    sharing_source,
    source_key,
    source_overlap_report,
    with_train_source_in_csv,
)

TRAIN, VAL = 1, 2
UUID = "27df52b1-6eb6-4e69-aea0-fd7129a73f34"


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("0hJpUJTsZAk_image_a94bd2ba-f365-4bae-91b6-8eb5dddcf5aa.jpg", "video:0hJpUJTsZAk"),
        ("a-b_c-d_e-f_image_x.jpg", "video:a-b_c-d_e-f"),  # ids may contain '_' and '-'
        ("1592807763981524992_2.jpg", "tweet:1592807763981524992"),
        ("1592807763981524992_0.PNG", "tweet:1592807763981524992"),
        (f"{UUID}_3.jpg", f"figure:{UUID}"),
        (f"{UUID}.jpg", f"figure:{UUID}"),
        (f"{UUID.upper()}_0.jpg", f"figure:{UUID}"),
        ("00000000001534.jpg", None),  # laion: no source
        ("short_image_x.jpg", None),  # 5-char prefix is not a video id
        ("1592807763981524992.jpg", None),  # tweet id without image index
    ],
)
def test_source_key(name: str, key: str | None) -> None:
    assert source_key(name) == key


def test_group_key_falls_back_to_image() -> None:
    assert group_key("00000000001534.jpg") == "image:00000000001534.jpg"
    assert group_key("1_2.jpg") == "tweet:1"


def test_sharing_source() -> None:
    train = ["vidAAAAAAAA_image_1.jpg", "vidBBBBBBBB_image_1.jpg", "9_0.jpg", "1.jpg"]
    golden = ["vidAAAAAAAA_image_2.jpg", "9_5.jpg", "1.jpg"]
    # laion-style "1.jpg" has no key, so an identical name does not count as shared source
    assert sharing_source(train, golden) == ["9_0.jpg", "vidAAAAAAAA_image_1.jpg"]


def test_csv_split_by_source_counts_mixed_groups() -> None:
    mask = {
        "vidAAAAAAAA_image_1.jpg": TRAIN,
        "vidAAAAAAAA_image_2.jpg": TRAIN,
        "vidBBBBBBBB_image_1.jpg": VAL,
        f"{UUID}_0.jpg": TRAIN,
        f"{UUID}_1.jpg": VAL,
        "5.jpg": VAL,
    }
    out = csv_split_by_source(mask)
    assert out["video"] == {
        "groups": 2,
        "split_class": {"train_only": 1, "val_only": 1},
        "mixed_examples": [],
    }
    assert out["figure"]["split_class"] == {"train_and_val": 1}
    assert out["figure"]["mixed_examples"] == [f"figure:{UUID}"]
    assert out["tweet"]["groups"] == 0


def test_source_overlap_report() -> None:
    mask = {
        "vidAAAAAAAA_image_t.jpg": TRAIN,
        "vidAAAAAAAA_image_g.jpg": VAL,  # same video split across train/val
        f"{UUID}_0.jpg": TRAIN,
        f"{UUID}_1.jpg": VAL,
        "7_0.jpg": TRAIN,
        "8_0.jpg": TRAIN | VAL,  # not train-only: never a train candidate
        "8_1.jpg": VAL,
        "00001.jpg": TRAIN,
    }
    disk = list(mask)
    subset = {n: "quilt" if "_image_" in n else "other" for n in disk}
    cand = ["vidAAAAAAAA_image_g.jpg", f"{UUID}_1.jpg", "8_1.jpg"]
    golden = ["vidAAAAAAAA_image_g.jpg", "8_1.jpg"]
    main = ["vidAAAAAAAA_image_t.jpg"]

    r = source_overlap_report(mask, disk, subset, cand, golden, main)

    assert r["train_candidates"]["images"] == 4
    assert r["train_candidates"]["images_by_kind"] == {
        "figure": 1,
        "none": 1,
        "tweet": 1,
        "video": 1,
    }
    s = r["train_sharing_golden_candidates"]
    assert s["train_images"] == 2
    assert s["train_images_by_kind"] == {"figure": 1, "video": 1}
    assert s["train_images_in_main_corpus"] == 1
    assert r["train_sharing_golden"]["train_images"] == 1  # figure candidate not in golden
    assert r["main_corpus_sharing_golden"] == 1
    assert r["csv"]["split_by_source"]["video"]["split_class"] == {"train_and_val": 1}
    assert r["disk"]["kind_by_subset"]["quilt"] == {"video": 2}
    # golden video has a train sibling; golden tweet 8_1's group has a train row via 8_0
    assert r["golden_with_train_source_in_csv"] == {"tweet": 1, "video": 1}


def test_with_train_source_in_csv_ignores_val_only_groups() -> None:
    mask = {"1_0.jpg": VAL, "1_1.jpg": VAL, "2_0.jpg": TRAIN, "2_1.jpg": VAL}
    assert with_train_source_in_csv(["1_1.jpg", "2_1.jpg", "x.jpg"], mask) == {"tweet": 1}
