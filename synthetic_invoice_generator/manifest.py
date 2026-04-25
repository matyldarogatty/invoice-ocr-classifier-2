"""Append-only manifest.jsonl."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def append_manifest(manifest_path: Path, row: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
