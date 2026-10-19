import numpy as np
import pandas as pd
import pytest

from pathosearch.eval.corpus_analysis import CsvScan
from pathosearch.eval.heldout import (
    NEAR_DUP_TAU,
    LeakageError,
    assert_disjoint,
    build_leakage_report,
    compare_with_analysis,
    duplicate_names,
    manifest_meta,
    near_duplicates,
    overlap_with_main_ids,
    select_heldout,
    train_row_violations,
    within_near_duplicates,
)

TRAIN, VAL = 1, 2


def make_scan() -> CsvScan:
    masks = {
        "a_val.jpg": VAL,  # chosen
        "b_val.jpg": VAL,  # chosen (unknown pathology)
        "c_both.jpg": TRAIN | VAL,  # has a train row
        "d_train.jpg": TRAIN,
        "e_val_in_ids.jpg": VAL,  # in the main corpus
        "f_val_off_disk.jpg": VAL,  # not on disk
        "g_other.jpg": VAL | 4,  # other split value mixed in
    }
    rows = pd.DataFrame(
        {
            "image_path": ["a_val.jpg", "a_val.jpg", "b_val.jpg", "c_both.jpg", "e_val_in_ids.jpg"],
            "pathology": [
                "['Dermatopathology', 'Bone']",
                "['Bone']",
                None,
                "['Renal']",
                "['Bone']",
            ],
        }
    )
    return CsvScan(n_rows=5, image_mask=masks, disk_rows=rows)


DISK = {"a_val.jpg", "b_val.jpg", "c_both.jpg", "d_train.jpg", "e_val_in_ids.jpg", "g_other.jpg"}


def test_select_heldout_v0_rule() -> None:
    entries = select_heldout(make_scan(), DISK, ["e_val_in_ids.jpg"], "val_only_on_disk")
    assert [e.image for e in entries] == ["a_val.jpg", "b_val.jpg"]
    assert [e.pathology for e in entries] == [
        "dermatopathology",
        "unknown",
    ]  # first row, first label
    assert {e.split_class for e in entries} == {"val_only"}


def test_select_heldout_excludes_main_ids_and_handles_empty() -> None:
    assert (
        select_heldout(
            make_scan(), DISK, ["a_val.jpg", "b_val.jpg", "e_val_in_ids.jpg"], "val_only_on_disk"
        )
        == []
    )
    assert select_heldout(make_scan(), set(), [], "val_only_on_disk") == []


def test_select_heldout_unknown_rule() -> None:
    with pytest.raises(ValueError, match="unknown rule"):
        select_heldout(make_scan(), DISK, [], "nope")


def test_select_heldout_requires_rows_for_chosen_images() -> None:
    scan = make_scan()
    scan.disk_rows = scan.disk_rows[scan.disk_rows["image_path"] != "b_val.jpg"]
    with pytest.raises(ValueError, match="no CSV rows"):
        select_heldout(scan, DISK, [], "val_only_on_disk")


def test_overlap_and_assert_disjoint() -> None:
    assert overlap_with_main_ids(["x", "y"], ["y", "z"]) == ["y"]
    assert_disjoint(["a"], ["b"])
    with pytest.raises(LeakageError, match="1 golden images"):
        assert_disjoint(["a", "b"], ["b", "c"])


def test_train_row_violations() -> None:
    out = train_row_violations(
        ["a_val.jpg", "c_both.jpg", "d_train.jpg", "zzz.jpg"], make_scan().image_mask
    )
    assert out == {"has_train_row": ["c_both.jpg", "d_train.jpg"], "not_in_csv": ["zzz.jpg"]}


def test_near_duplicates_respects_threshold() -> None:
    main = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    ids_m = ["m0", "m1"]
    c = NEAR_DUP_TAU - 0.01
    below = [c, np.sqrt(1 - c**2), 0]  # cosine to m0 just under tau
    golden = np.array([[2, 0, 0], below, [0.99, 0.1, 0], [0, 0, 1]], dtype=np.float32)
    found = near_duplicates(golden, ["g0", "g1", "g2", "g3"], main, ids_m)
    assert [(d.image, d.nearest) for d in found] == [("g0", "m0"), ("g2", "m0")]
    assert found[0].max_cosine == pytest.approx(1.0)  # unnormalised input is normalised
    loose = near_duplicates(golden, ["g0", "g1", "g2", "g3"], main, ids_m, tau=0.5)
    assert [d.image for d in loose] == ["g0", "g2", "g1"]  # sorted by descending cosine


def test_near_duplicates_input_validation() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        near_duplicates(np.ones((2, 3)), ["a"], np.ones((1, 3)), ["m"])
    with pytest.raises(ValueError, match="dimension"):
        near_duplicates(np.ones((1, 3)), ["a"], np.ones((1, 4)), ["m"])


def test_near_duplicates_crosses_chunk_boundary() -> None:
    rng = np.random.default_rng(0)
    main = rng.normal(size=(5, 8)).astype(np.float32)
    golden = rng.normal(size=(600, 8)).astype(np.float32)  # > one 256-row chunk
    golden[599] = main[3]
    found = near_duplicates(
        golden, [f"g{i}" for i in range(600)], main, [f"m{i}" for i in range(5)], tau=0.99
    )
    assert [(d.image, d.nearest) for d in found] == [("g599", "m3")]


def test_duplicates_inside_golden() -> None:
    assert duplicate_names(["a", "b", "a", "c", "c", "c"]) == ["a", "c"]
    emb = np.array([[1, 0], [1, 0.001], [0, 1]], dtype=np.float32)
    assert [(a, b) for a, b, _ in within_near_duplicates(emb, ["y", "x", "z"])] == [("x", "y")]
    assert within_near_duplicates(np.eye(3, dtype=np.float32), ["a", "b", "c"]) == []


# --- manifest metadata, analysis comparison, aggregate report --------------------------


def _entries():
    return select_heldout(make_scan(), DISK, ["e_val_in_ids.jpg"], "val_only_on_disk")


def test_manifest_meta_records_rule_counts_and_sources(tmp_path) -> None:
    src = tmp_path / "ids.json"
    src.write_text("[]", encoding="utf-8")
    meta = manifest_meta("val_only_on_disk", _entries(), {"ids": src}, disk_images=6)
    assert meta["rule"] == "val_only_on_disk" and meta["images"] == 2
    assert meta["per_category"] == {"dermatopathology": 1, "unknown": 1}
    assert meta["split_classes"] == {"val_only": 2} and meta["disk_images_scanned"] == 6
    assert meta["sources"]["ids"]["size_bytes"] == 2 and "mtime_utc" in meta["sources"]["ids"]


def _analysis(images, per_category):
    return {
        "sec_7_8_val_only_on_disk": {
            "images": images,
            "all_val_only_images": {
                "per_category": {k: {"images": v} for k, v in per_category.items()}
            },
        }
    }


def test_compare_with_analysis() -> None:
    ok = _analysis(2, {"dermatopathology": 1, "unknown": 1})
    assert compare_with_analysis(_entries(), ok) == []
    bad = _analysis(3, {"dermatopathology": 1, "unknown": 1, "bone": 1})
    problems = compare_with_analysis(_entries(), bad)
    assert (
        "images: manifest 2 vs analysis 3" in problems
        and "bone: manifest 0 vs analysis 1" in problems
    )


def _emb(rows):
    return np.array(rows, dtype=np.float32)


def test_leakage_report_clean_and_dirty() -> None:
    golden_ids = ["g1", "g2"]
    mask = {"g1": VAL, "g2": VAL}
    g = {"clip": (_emb([[1, 0], [0, 1]]), golden_ids)}
    m = {"clip": (_emb([[0.5, 0.5], [-1, 0.2]]), ["m1", "m2"])}
    clean = build_leakage_report(golden_ids, ["m1", "m2"], mask, g, m)
    assert clean["passed"] and sum(clean["violation_counts"].values()) == 0

    dirty_m = {"clip": (_emb([[1, 0.001], [0.5, 0.5]]), ["g1", "m2"])}  # g1 is also a main image
    dirty = build_leakage_report(golden_ids, ["g1", "m2"], {"g1": VAL | TRAIN}, g, dirty_m)
    assert not dirty["passed"]
    assert dirty["a_overlap_with_main_ids"] == ["g1"]
    assert dirty["b_has_train_row"] == ["g1"] and dirty["b_not_in_csv"] == ["g2"]
    assert dirty["c_near_duplicates_of_main_corpus"]["clip"][0]["image"] == "g1"
    assert dirty["violation_counts"]["a_overlap_with_main_ids"] == 1


def test_leakage_report_validates_inputs() -> None:
    g = {"clip": (_emb([[1, 0]]), ["g1"])}
    with pytest.raises(ValueError, match="same encoders"):
        build_leakage_report(["g1"], ["m"], {}, g, {})
    with pytest.raises(ValueError, match="differ from the manifest"):
        build_leakage_report(["g1", "g2"], ["m"], {}, g, {"clip": (_emb([[1, 0]]), ["m"])})
