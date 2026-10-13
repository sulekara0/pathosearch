from pathlib import Path

import pytest

from pathosearch.config import load_settings, read_dotenv, require_dir, require_file
from pathosearch.errors import DataNotFoundError


def test_defaults_are_relative_and_derived() -> None:
    s = load_settings({})
    assert s.data_dir == Path("data")
    assert s.artifacts_dir == Path("artifacts")
    assert s.images_dir == Path("data") / "images"
    assert s.captions_path == Path("artifacts") / "captions_quilt.jsonl"
    assert s.encoder_dir("quilt") == Path("artifacts") / "quilt"


def test_env_overrides() -> None:
    s = load_settings(
        {
            "PATHOSEARCH_DATA_DIR": "d",
            "PATHOSEARCH_ARTIFACTS_DIR": "a",
            "PATHOSEARCH_IMAGES_DIR": "imgs",
            "PATHOSEARCH_CAPTIONS_PATH": "c.jsonl",
        }
    )
    assert (s.data_dir, s.artifacts_dir) == (Path("d"), Path("a"))
    assert s.images_dir == Path("imgs")
    assert s.captions_path == Path("c.jsonl")


def test_images_default_follows_data_dir() -> None:
    assert load_settings({"PATHOSEARCH_DATA_DIR": "x"}).images_dir == Path("x") / "images"


def test_read_dotenv(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text('# c\nA=1\nB = "two"\n\nbad line\n', encoding="utf-8")
    assert read_dotenv(f) == {"A": "1", "B": "two"}
    assert read_dotenv(tmp_path / "missing") == {}


def test_require_dir_missing_empty_and_ok(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError, match="hint text"):
        require_dir(tmp_path / "nope", "hint text")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataNotFoundError):
        require_dir(empty)
    (empty / "a.jpg").write_bytes(b"x")
    assert require_dir(empty) == empty


def test_require_file(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError):
        require_file(tmp_path / "x.jsonl")
    f = tmp_path / "x.jsonl"
    f.write_text("{}", encoding="utf-8")
    assert require_file(f) == f


def test_no_silent_fallback_to_test_dirs(tmp_path: Path) -> None:
    # A populated *_test dir must NOT be picked up when the configured one is missing.
    (tmp_path / "images_test").mkdir()
    (tmp_path / "images_test" / "a.jpg").write_bytes(b"x")
    with pytest.raises(DataNotFoundError):
        require_dir(tmp_path / "images")
