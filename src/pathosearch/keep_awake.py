"""Keep Windows from sleeping while a long job runs (no system setting is changed).

Uses ``SetThreadExecutionState``: the request lasts until the context exits or the process
ends. It does not stop sleep on lid close or a manual sleep. No-op on other platforms.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake() -> Iterator[bool]:
    """Yield True if the sleep block is active, False if unsupported on this platform."""
    if sys.platform != "win32":
        yield False
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    active = bool(kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    try:
        yield active
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
