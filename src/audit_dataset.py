#!/usr/bin/env python3
"""Audit a labels CSV + image directory: counts, class balance, missing files (read-only)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# Run from repo root: python src/audit_dataset.py
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import LABELS, NUM_CLASSES  # noqa: E402

OPTIONAL_COLUMNS = ("filename", "text", "label", "invoice_id", "semantic_name", "source_pdf")


def _empty_text(s: Any) -> bool:
    if pd.isna(s):
        return True
    return str(s).strip() == ""


def _missing_label(s: Any) -> bool:
    if pd.isna(s):
        return True
    try:
        float(s)
        return False
    except (TypeError, ValueError):
        return True


def run_audit(
    labels_csv: Path,
    images_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> int:
    summary_path = output_dir / "dataset_audit_summary.json"
    if output_dir.exists() and not force:
        for p in (summary_path, output_dir / "class_distribution.csv"):
            if p.exists():
                print(
                    f"Refusing to overwrite {p}; use --force or choose an empty output directory.",
                    file=sys.stderr,
                )
                return 2

    if not labels_csv.is_file():
        print(f"Labels CSV not found: {labels_csv}", file=sys.stderr)
        return 1
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)
    df.columns = [str(c).strip() for c in df.columns]
    columns_present: Set[str] = set(df.columns)
    detected = {c: (c in columns_present) for c in OPTIONAL_COLUMNS}

    if "filename" not in columns_present:
        print("Required column 'filename' is missing.", file=sys.stderr)
        return 1
    if "label" not in columns_present:
        print("Required column 'label' is missing (needed for class statistics).", file=sys.stderr)
        return 1

    n_total = len(df)
    n_missing_label = int(df["label"].apply(_missing_label).sum()) if "label" in df.columns else n_total
    n_missing_text = (
        int(df["text"].apply(_empty_text).sum()) if "text" in df.columns else 0
    )

    n_missing_image = 0
    missing_rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        fn = row["filename"]
        if pd.isna(fn) or str(fn).strip() == "":
            n_missing_image += 1
            missing_rows.append(
                {
                    "row_index": int(i) if i == i else 0,
                    "filename": fn,
                    "reason": "empty_filename",
                }
            )
            continue
        p = images_dir / str(fn)
        if not p.is_file():
            n_missing_image += 1
            missing_rows.append(
                {
                    "row_index": int(i),
                    "filename": str(fn),
                    "reason": "file_not_found",
                }
            )

    # Class distribution (numeric label)
    by_label: Counter = Counter()
    for _, row in df.iterrows():
        if "label" not in df.columns or _missing_label(row["label"]):
            continue
        try:
            lid = int(float(row["label"]))
            by_label[str(lid)] += 1
        except (TypeError, ValueError):
            continue

    by_semantic: Optional[Counter] = None
    if "semantic_name" in df.columns:
        by_semantic = Counter()
        for v in df["semantic_name"]:
            if pd.isna(v) or str(v).strip() == "":
                by_semantic[""] += 1
            else:
                by_semantic[str(v).strip()] += 1

    doc_dist: Optional[Counter] = None
    if "invoice_id" in df.columns:
        doc_dist = Counter()
        for v in df["invoice_id"].fillna("__MISSING__").astype(str):
            doc_dist[v] += 1
    elif "source_pdf" in df.columns:
        doc_dist = Counter()
        for v in df["source_pdf"].fillna("__MISSING__").astype(str):
            doc_dist[v] += 1

    # Config classes with zero examples (by integer id 0..NUM_CLASSES-1)
    label_ids_in_data: Set[int] = set()
    for l in by_label:
        try:
            label_ids_in_data.add(int(l))
        except ValueError:
            continue

    class_names = {str(i): LABELS.get(i, str(i)) for i in range(NUM_CLASSES)}
    zero_example_classes: List[Dict[str, Any]] = []
    for i in range(NUM_CLASSES):
        if i not in label_ids_in_data:
            zero_example_classes.append({"label_id": i, "name": LABELS.get(i, str(i))})

    summary: Dict[str, Any] = {
        "labels_csv": str(labels_csv.resolve()),
        "images_dir": str(images_dir.resolve()),
        "columns_detected": {k: v for k, v in detected.items()},
        "n_rows": n_total,
        "n_rows_missing_label": n_missing_label,
        "n_rows_missing_text": n_missing_text,
        "n_rows_missing_image_file": n_missing_image,
        "class_distribution_by_label": dict(by_label),
        "class_id_to_name": class_names,
        "classes_in_config_with_zero_examples": zero_example_classes,
    }
    if by_semantic is not None:
        summary["class_distribution_by_semantic_name"] = dict(by_semantic)
    if doc_dist is not None:
        summary["document_count_by_id"] = dict(doc_dist)
        # Per-document row counts already in document_count? Actually that's count of rows - Counter sums rows per doc key. Good.
        with open(output_dir / "document_distribution.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["document_id", "row_count"])
            for k, v in sorted(doc_dist.items(), key=lambda x: -x[1]):
                w.writerow([k, v])
    else:
        summary["document_distribution_note"] = "No invoice_id or source_pdf column; document_distribution.csv not written"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label_id", "class_name", "count"])
        for lid in sorted(by_label, key=lambda x: int(x)):
            name = LABELS.get(int(lid), "")
            w.writerow([lid, name, by_label[lid]])

    if missing_rows:
        with open(output_dir / "missing_images.csv", "w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(
                f,
                fieldnames=["row_index", "filename", "reason"],
            )
            dw.writeheader()
            for r in missing_rows:
                dw.writerow(r)

    # Console summary
    print("=== Dataset audit ===")
    print(f"CSV: {labels_csv}")
    print(f"Images: {images_dir}")
    print(f"Rows: {n_total}")
    print(f"Missing labels: {n_missing_label}  |  missing text: {n_missing_text}  |  missing image file: {n_missing_image}")
    print("Columns found:", {k: v for k, v in detected.items() if v})
    print("Class distribution (label_id -> count):")
    for lid in sorted(by_label, key=lambda x: int(x)):
        nm = LABELS.get(int(lid), "?")
        print(f"  {lid} ({nm}): {by_label[lid]}")
    if zero_example_classes:
        print("Config classes with ZERO examples in this dataset:")
        for z in zero_example_classes:
            print(f"  id={z['label_id']}: {z['name']}")
    if doc_dist is not None:
        print(f"Documents (by {'invoice_id' if 'invoice_id' in columns_present else 'source_pdf'}): {len(doc_dist)}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {output_dir / 'class_distribution.csv'}")
    if missing_rows:
        print(f"Wrote: {output_dir / 'missing_images.csv'}")
    if doc_dist is not None:
        print(f"Wrote: {output_dir / 'document_distribution.csv'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Audit labels CSV and image files (read-only).")
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting files in --output-dir if they already exist.",
    )
    args = p.parse_args()
    return run_audit(args.labels_csv, args.images_dir, args.output_dir, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
