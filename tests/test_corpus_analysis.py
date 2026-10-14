import json
from pathlib import Path

import pandas as pd

from pathosearch.eval.corpus_analysis import is_missing, run_analysis, split_class


def test_is_missing_and_split_class() -> None:
    assert is_missing(None) and is_missing(float("nan")) and is_missing("[]") and is_missing(" ")
    assert not is_missing("40x")
    assert split_class(2) == "val_only" and split_class(3) == "train_and_val"


def test_run_analysis_on_synthetic_corpus(tmp_path: Path) -> None:
    rows = [
        # name, split, pathology, caption, magnification, not_histology
        ("a.jpg", "train", "['Renal']", "kidney", None, "0"),
        ("b.jpg", "val", "['Bone', 'Renal']", "bone one", "20", "0"),
        ("b.jpg", "val", "['Bone']", "bone two", "20", "0"),
        ("c.jpg", "val", None, "no label", None, "0"),
        ("d.jpg", "val", "['Bone']", "not histology", None, "1"),
        ("e.jpg", "train", "['Renal']", "x", None, "0"),
        ("e.jpg", "val", "['Renal']", "x", None, "0"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["image_path", "split", "pathology", "caption", "magnification", "not_histology"],
    )
    df["subset"] = "quilt"
    df["single_wsi"] = "1"
    csv = tmp_path / "q.csv"
    df.to_csv(csv)
    img = tmp_path / "images" / "part"
    img.mkdir(parents=True)
    for n in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"):
        (img / n).write_bytes(b"x")
    cap = tmp_path / "c.jsonl"
    cap.write_text(
        "\n".join(
            json.dumps({"path": f"p/{n}", "caption": "t", "pathology": p, "split": "train"})
            for n, p in [("a.jpg", "Renal"), ("a.jpg", "renal"), ("e.jpg", "Unknown")]
        ),
        encoding="utf-8",
    )
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["a.jpg", "e.jpg", "zzz.jpg"]), encoding="utf-8")

    out = run_analysis(csv, cap, tmp_path / "images", ids, min_per_category=2)

    assert out["sec_7_7_1_csv_split"]["image_split_classes"] == {
        "train_and_val": 1,
        "train_only": 1,
        "val_only": 3,
    }
    disk = out["sec_7_7_2_to_7_7_4_disk_and_ids"]
    assert disk["ids_missing_on_disk"] == 1 and disk["disk_in_ids"] == 2
    assert disk["ids_split_classes"]["not_in_csv"] == 1
    j = out["sec_7_x_jsonl"]
    assert j["rows"] == 3 and j["unique_images"] == 2
    assert j["rows_per_image_histogram"] == {"1": 1, "2": 1}
    assert j["image_label_distribution"]["renal"]["images"] == 1

    v = out["sec_7_8_val_only_on_disk"]
    assert v["images"] == 3 and v["also_in_ids"] == 0
    assert v["csv_rows_per_image_histogram"] == {"1": 2, "2": 1}
    allv = v["all_val_only_images"]
    assert allv["per_category"]["bone"]["images"] == 2  # b (first label) + d
    assert allv["unknown_images"] == 1
    assert allv["categories_with_at_least_2_excluding_unknown"] == ["bone"]
    # d.jpg has not_histology=1 -> excluded from the eligible view
    assert v["eligible_images_not_histology0_and_caption_present"]["images"] == 2
    assert v["column_availability"]["magnification"]["images_with_any_non_missing"] == 1
