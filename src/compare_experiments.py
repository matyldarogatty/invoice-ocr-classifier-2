#!/usr/bin/env python3
"""Aggregate metrics.json from multiple experiment directories into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pick(m: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur: Any = m
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _load_metrics(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"metrics.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_metrics(exp_dir: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    name = data.get("model_name", exp_dir.name)
    inp = data.get("input_type", "")
    return {
        "experiment_dir": str(exp_dir.resolve()),
        "model_name": name,
        "input_type": inp,
        "val_accuracy": _pick(data, ["val", "accuracy"]),
        "val_macro_f1": _pick(data, ["val", "macro_f1"]),
        "val_weighted_f1": _pick(data, ["val", "weighted_f1"]),
        "test_accuracy": _pick(data, ["test", "accuracy"]),
        "test_macro_f1": _pick(data, ["test", "macro_f1"]),
        "test_weighted_f1": _pick(data, ["test", "weighted_f1"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare metrics.json from CNN, text baseline, or future hybrid runs."
    )
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Parent directory: each subdirectory with a metrics.json is included.",
    )
    p.add_argument(
        "--experiment",
        type=Path,
        action="append",
        default=[],
        help="Path to an experiment directory containing metrics.json (repeatable).",
    )
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    exp_paths: List[Path] = []
    for e in args.experiment:
        exp_paths.append(e)
    if args.experiments_dir is not None:
        if not args.experiments_dir.is_dir():
            print(f"Not a directory: {args.experiments_dir}", file=sys.stderr)
            return 1
        for child in sorted(args.experiments_dir.iterdir()):
            if child.is_dir() and (child / "metrics.json").is_file():
                exp_paths.append(child)
    if not exp_paths:
        print("No experiment directories found. Use --experiment or --experiments-dir.", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for exp_dir in exp_paths:
        mpath = exp_dir / "metrics.json"
        try:
            data = _load_metrics(mpath)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            return 1
        rows.append(_row_from_metrics(exp_dir, data))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print table
    def fmt(x: Any) -> str:
        if x is None:
            return "n/a"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    print("=== Experiment comparison ===")
    for r in rows:
        print(
            f"{r['model_name'][:32]:32}  {r['input_type'][:12]:12}  "
            f"val_acc={fmt(r['val_accuracy'])}  val_mF1={fmt(r['val_macro_f1'])}  "
            f"test_acc={fmt(r['test_accuracy'])}  test_mF1={fmt(r['test_macro_f1'])}"
        )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
