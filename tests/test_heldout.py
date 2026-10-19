import numpy as np
import pandas as pd
import pytest

from pathosearch.eval.corpus_analysis import CsvScan
from pathosearch.eval.heldout import (
    NEAR_DUP_TAU,
    LeakageError,
    assert_disjoint,
    duplicate_names,
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
