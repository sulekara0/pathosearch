"""Project-specific exceptions."""

from __future__ import annotations


class DataNotFoundError(FileNotFoundError):
    """A required data directory/file is missing.

    Raised instead of silently falling back to another location, so a run can
    never quietly use the wrong data (e.g. a test subset instead of the corpus).
    """
