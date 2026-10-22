"""Training bundle for Colab: resized images + captions in one zip (never committed).

Images are resized so the short side is ``size`` (bicubic, as ``CLIPImageProcessor``), which
keeps the processor's own resize a no-op and shrinks the upload; aspect ratio is kept and
nothing is upscaled. File names are unchanged so manifest names resolve inside the zip.
Everything is validated before writing; the zip is written to a temp name, then renamed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from pathosearch.errors import DataNotFoundError

BUNDLE_SIZE = 224
JPEG_QUALITY = 95


def resize_short_side(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    if short <= size:
        return img
    scale = size / short
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)


def _encode(img: Image.Image, name: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if name.lower().endswith(".png"):
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_bundle(
    names: Sequence[str],
    paths: Mapping[str, Path],
    captions: Mapping[str, Sequence[str]],
    out_zip: Path,
    size: int = BUNDLE_SIZE,
    quality: int = JPEG_QUALITY,
) -> dict[str, Any]:
    """Write ``images/<name>`` and ``captions.jsonl`` for ``names``; return bundle meta."""
    if out_zip.exists():
        raise FileExistsError(f"{out_zip} exists; a bundle is never overwritten")
    ordered = sorted(set(names))
    no_caps = [n for n in ordered if not captions.get(n)]
    if no_caps:
        raise ValueError(f"{len(no_caps)} images have no captions, e.g. {no_caps[:3]}")
    missing = [n for n in ordered if n not in paths or not Path(paths[n]).is_file()]
    if missing:
        raise DataNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    lines = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for n in ordered:
            with Image.open(paths[n]) as im:
                small = resize_short_side(im.convert("RGB"), size)
            z.writestr(f"images/{n}", _encode(small, n, quality))
            lines.append(_caption_line(n, captions[n]))
        z.writestr("captions.jsonl", "".join(lines))
    os.replace(tmp, out_zip)
    return {
        "images": len(ordered),
        "size": size,
        "resample": "bicubic",
        "jpeg_quality": quality,
        "zip": out_zip.name,
        "zip_bytes": out_zip.stat().st_size,
        "zip_sha256": _sha256(out_zip),
    }


def _caption_line(name: str, caps: Sequence[str]) -> str:
    # same format as data.write_captions_jsonl
    return json.dumps({"image": name, "captions": list(caps)}) + "\n"
