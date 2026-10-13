from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_stops_with_clear_error_when_data_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "PATHOSEARCH_DATA_DIR",
        "PATHOSEARCH_ARTIFACTS_DIR",
        "PATHOSEARCH_IMAGES_DIR",
        "PATHOSEARCH_CAPTIONS_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: no data/, no artifacts/

    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception  # no traceback
    assert len(at.error) == 1
    assert "Veri bulunamadı" in at.error[0].value
    assert "artifacts" in at.error[0].value
    assert len(at.text_input) == 0  # script stopped before the search UI
    assert any("teşhis aracı değildir" in w.value for w in at.warning)  # disclaimer is always shown
