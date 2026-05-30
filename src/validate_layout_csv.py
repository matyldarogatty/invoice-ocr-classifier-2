#!/usr/bin/env python3
"""Validate layout feature columns in an exported labels CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from layout_features import LAYOUT_FEATURE_COLUMNS, validate_layout_features  # noqa: E402


def validate_layout_csv(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    result: Dict[str, Any] = {
        "path": str(path.resolve()),
        "row_count": len(df),
        "layout_columns_present": 0,
        "missing_columns": [],
        "nan_count": 0,
        "out_of_range_count": 0,
        "invalid_bbox_order_count": 0,
        "ok": False,
    }

    missing = [c for c in LAYOUT_FEATURE_COLUMNS if c not in df.columns]
    result["missing_columns"] = missing
    result["layout_columns_present"] = len(LAYOUT_FEATURE_COLUMNS) - len(missing)
    if missing:
        return result

    for col in LAYOUT_FEATURE_COLUMNS:
        result["nan_count"] += int(df[col].isna().sum())
        numeric = pd.to_numeric(df[col], errors="coerce")
        result["nan_count"] += int(numeric.isna().sum() - df[col].isna().sum())
        result["out_of_range_count"] += int(((numeric < 0) | (numeric > 1)).sum())

    result["invalid_bbox_order_count"] = int(
        (df["bbox_x_min_norm"] > df["bbox_x_max_norm"]).sum()
        + (df["bbox_y_min_norm"] > df["bbox_y_max_norm"]).sum()
    )

    sample_errors = 0
    for _, row in df.head(min(100, len(df))).iterrows():
        try:
            validate_layout_features({col: float(row[col]) for col in LAYOUT_FEATURE_COLUMNS})
        except ValueError:
            sample_errors += 1

    result["sample_validation_errors"] = sample_errors
    result["ok"] = (
        result["nan_count"] == 0
        and result["out_of_range_count"] == 0
        and result["invalid_bbox_order_count"] == 0
        and sample_errors == 0
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Validate layout columns in labels CSV.")
    p.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "data" / "labels_synthetic_with_layout.csv",
    )
    args = p.parse_args()
    if not args.csv_path.is_file():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        return 1

    result = validate_layout_csv(args.csv_path)
    print(f"CSV: {result['path']}")
    print(f"Rows: {result['row_count']}")
    print(f"Layout columns present: {result['layout_columns_present']}/10")
    if result["missing_columns"]:
        print(f"Missing columns: {result['missing_columns']}")
    print(f"NaN count: {result['nan_count']}")
    print(f"Out of range [0,1]: {result['out_of_range_count']}")
    print(f"Invalid bbox order: {result['invalid_bbox_order_count']}")
    print(f"Sample validation errors: {result.get('sample_validation_errors', 0)}")
    print(f"OK: {result['ok']}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
