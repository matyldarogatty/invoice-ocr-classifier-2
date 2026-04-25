"""Filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_run_dirs(out_dir: Path) -> None:
    (out_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
