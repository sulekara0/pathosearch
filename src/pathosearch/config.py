"""Environment-based configuration for data and artifact locations.

Nothing here is hard-coded to a machine. Paths come from environment variables
(optionally loaded from a ``.env`` file in the working directory) and default
to ``data/`` and ``artifacts/`` relative to the current working directory,
i.e. the repository root when commands are run from there.

Variables:
    PATHOSEARCH_DATA_DIR       root for raw data           (default: data)
    PATHOSEARCH_ARTIFACTS_DIR  embeddings, indexes, etc.   (default: artifacts)
    PATHOSEARCH_IMAGES_DIR     image corpus                (default: <data>/images)
    PATHOSEARCH_CAPTIONS_PATH  captions JSONL              (default: <data>/captions_quilt.jsonl)
    PATHOSEARCH_CSV_PATH       Quilt-1M lookup CSV         (default: <data>/quilt_1M_lookup.csv)

API service (``load_api_settings``):
    PATHOSEARCH_GATE_THRESHOLD domain gate threshold       (default: 0.80, uncalibrated)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pathosearch.errors import DataNotFoundError

ENV_DATA_DIR = "PATHOSEARCH_DATA_DIR"
ENV_ARTIFACTS_DIR = "PATHOSEARCH_ARTIFACTS_DIR"
ENV_IMAGES_DIR = "PATHOSEARCH_IMAGES_DIR"
ENV_CAPTIONS_PATH = "PATHOSEARCH_CAPTIONS_PATH"
ENV_CSV_PATH = "PATHOSEARCH_CSV_PATH"
ENV_GATE_THRESHOLD = "PATHOSEARCH_GATE_THRESHOLD"

# Same value as domain_gate.DEFAULT_THRESHOLD; chosen by hand, not calibrated (ADR-0005).
DEFAULT_GATE_THRESHOLD = 0.80


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifacts_dir: Path
    images_dir: Path
    captions_path: Path
    csv_path: Path

    def encoder_dir(self, encoder_name: str) -> Path:
        """Directory holding embeddings/ids/index for one encoder."""
        return self.artifacts_dir / encoder_name


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` file. Missing file -> empty dict."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from ``env`` (default: ``.env`` file overlaid by os.environ)."""
    if env is None:
        env = {**read_dotenv(Path(".env")), **os.environ}
    data_dir = Path(env.get(ENV_DATA_DIR, "data"))
    artifacts_dir = Path(env.get(ENV_ARTIFACTS_DIR, "artifacts"))
    images_dir = Path(env[ENV_IMAGES_DIR]) if ENV_IMAGES_DIR in env else data_dir / "images"
    captions_path = (
        Path(env[ENV_CAPTIONS_PATH])
        if ENV_CAPTIONS_PATH in env
        else data_dir / "captions_quilt.jsonl"
    )
    csv_path = Path(env[ENV_CSV_PATH]) if ENV_CSV_PATH in env else data_dir / "quilt_1M_lookup.csv"
    return Settings(data_dir, artifacts_dir, images_dir, captions_path, csv_path)


@dataclass(frozen=True)
class ApiSettings:
    gate_threshold: float


def load_api_settings(env: Mapping[str, str] | None = None) -> ApiSettings:
    """Build API service settings from ``env`` (default: ``.env`` overlaid by os.environ)."""
    if env is None:
        env = {**read_dotenv(Path(".env")), **os.environ}
    raw = env.get(ENV_GATE_THRESHOLD)
    threshold = DEFAULT_GATE_THRESHOLD if raw is None else float(raw)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"{ENV_GATE_THRESHOLD} must be in (0, 1], got {threshold}")
    return ApiSettings(gate_threshold=threshold)


def require_dir(path: Path, hint: str = "") -> Path:
    """Return ``path`` if it is a non-empty directory, else raise DataNotFoundError."""
    if not path.is_dir() or not any(path.iterdir()):
        raise DataNotFoundError(_message("Directory not found or empty", path, hint))
    return path


def require_file(path: Path, hint: str = "") -> Path:
    """Return ``path`` if it is an existing file, else raise DataNotFoundError."""
    if not path.is_file():
        raise DataNotFoundError(_message("File not found", path, hint))
    return path


def _message(what: str, path: Path, hint: str) -> str:
    msg = f"{what}: {path}"
    return f"{msg}\n{hint}" if hint else msg
