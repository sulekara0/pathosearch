import sys

from pathosearch.keep_awake import keep_awake


def test_keep_awake_is_a_context_manager() -> None:
    with keep_awake() as active:
        assert active is (sys.platform == "win32") or active is False
