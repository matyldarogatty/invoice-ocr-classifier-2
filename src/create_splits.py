#!/usr/bin/env python3
"""Create document-level train/val/test CSV splits from a labels file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from splitting import (  # noqa: E402
    build_split_metadata,
    document_level_split,
    resolve_group_column,
)


def _assert_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, col: str) -> None:
    a = set(train[col].fillna("").astype(str))
    b = set(val[col].fillna("").astype(str))
    c = set(test[col].fillna("").astype(str))
    if a & b or a & c or b & c:
        raise RuntimeError("Overlap between splits detected (internal error).")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Split labels CSV at document (invoice) level into train/val/test."
    )
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--group-column",
        type=str,
        default=None,
        help="Group column: invoice_id or source_pdf. If omitted, auto-detect.",
    )
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.labels_csv.is_file():
        print(f"File not found: {args.labels_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.labels_csv)
    group_col = resolve_group_column(df, args.group_column)
    tr, va, te, meta = document_level_split(
        df, group_col, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    _assert_no_overlap(tr, va, te, group_col)
    out = build_split_metadata(tr, va, te, meta)
    out["source_labels_csv"] = str(args.labels_csv.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tr.to_csv(args.output_dir / "train.csv", index=False)
    va.to_csv(args.output_dir / "val.csv", index=False)
    te.to_csv(args.output_dir / "test.csv", index=False)
    with open(args.output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=== Document-level splits ===")
    print(f"Group column: {group_col}  (seed={args.seed})")
    print(f"Documents — train: {out['n_documents_train']}, val: {out['n_documents_val']}, test: {out['n_documents_test']}")
    print(f"Rows      — train: {out['n_rows_train']}, val: {out['n_rows_val']}, test: {out['n_rows_test']}")
    print("Class distribution (train):", out["class_distribution_train"])
    print("Class distribution (val):", out["class_distribution_val"])
    print("Class distribution (test):", out["class_distribution_test"])
    print(f"Wrote: {args.output_dir / 'train.csv'}, val.csv, test.csv, split_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
