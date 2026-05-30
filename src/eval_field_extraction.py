#!/usr/bin/env python3
"""
Field-level extraction metrics (extension for thesis).

Given labeled lines (ground truth or model predictions mapped to semantics),
compare extracted values against canonical values from synthetic invoice JSON.

Uses the same parsing helpers as match_utils (NIP digits, PL amounts, dates).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import LABELS  # noqa: E402
from match_utils import (  # noqa: E402
    amount_key,
    digits_only,
    try_parse_date_iso,
)

PROJECT_DIR = _SRC.parent

# Semantics evaluated at field (document) level — first matching line wins.
FIELD_SEMANTICS = [
    "SELLER_NIP",
    "BUYER_NIP",
    "INVOICE_NUMBER",
    "INVOICE_DATE",
    "SALE_DATE",
    "NET_AMOUNT",
    "VAT_AMOUNT",
    "TOTAL_PRICE",
    "CURRENCY",
    "SELLER_NAME",
    "BUYER_NAME",
]


def _canonical_from_hints(hints: List[Dict[str, Any]], semantic: str) -> Optional[str]:
    for h in hints:
        if h.get("semantic") == semantic:
            return str(h.get("canonical_value") or "")
    return None


def _extract_value(text: str, semantic: str) -> Optional[str]:
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    if "NIP" in semantic:
        d = digits_only(t)
        return d if len(d) >= 10 else None
    if semantic in ("NET_AMOUNT", "VAT_AMOUNT", "TOTAL_PRICE"):
        return amount_key(t)
    if semantic in ("INVOICE_DATE", "SALE_DATE"):
        return try_parse_date_iso(t)
    if semantic == "CURRENCY":
        from match_utils import normalize_basic, _ocr_line_to_iso_currencies

        iso = _ocr_line_to_iso_currencies(normalize_basic(t))
        return sorted(iso)[0] if iso else None
    # Names / invoice number: normalized string comparison
    from match_utils import normalize_basic

    return normalize_basic(t) or None


def _normalize_gt(canonical: str, semantic: str) -> Optional[str]:
    if not canonical:
        return None
    if "NIP" in semantic:
        d = digits_only(canonical)
        return d if len(d) >= 10 else None
    if semantic in ("NET_AMOUNT", "VAT_AMOUNT", "TOTAL_PRICE"):
        return amount_key(canonical)
    if semantic in ("INVOICE_DATE", "SALE_DATE"):
        return try_parse_date_iso(canonical) or canonical.strip()
    if semantic == "CURRENCY":
        return canonical.strip().upper()
    from match_utils import normalize_basic

    return normalize_basic(canonical)


def _lines_by_invoice(df) -> Dict[str, List[Tuple[str, str]]]:
    """invoice_id -> list of (semantic_name, text)."""
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for _, row in df.iterrows():
        inv = str(row.get("invoice_id", "") or "")
        if not inv:
            continue
        sem = str(row.get("semantic_name", "") or "")
        if sem == "OTHER" or not sem:
            # Map numeric label if semantic_name missing
            lid = row.get("label")
            if lid is not None and int(lid) in LABELS:
                sem = LABELS[int(lid)]
        text = str(row.get("text", "") or "")
        if sem and sem != "OTHER":
            out[inv].append((sem, text))
    return dict(out)


def evaluate_fields(
    labels_csv: Path,
    json_dir: Path,
    semantics: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import pandas as pd

    df = pd.read_csv(labels_csv)
    by_inv = _lines_by_invoice(df)

    per_field = {s: {"correct": 0, "total": 0, "missing_pred": 0} for s in semantics}
    per_doc_rows: List[Dict[str, Any]] = []

    for invoice_id, lines in sorted(by_inv.items()):
        json_path = json_dir / f"{invoice_id}.json"
        if not json_path.is_file():
            continue
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        hints = doc.get("classification_hints") or []

        doc_row: Dict[str, Any] = {"invoice_id": invoice_id, "fields": {}}
        for sem in semantics:
            gt_raw = _canonical_from_hints(hints, sem)
            if gt_raw is None:
                continue
            gt_norm = _normalize_gt(gt_raw, sem)
            per_field[sem]["total"] += 1

            pred_text = None
            for line_sem, text in lines:
                if line_sem == sem:
                    pred_text = text
                    break

            if pred_text is None:
                per_field[sem]["missing_pred"] += 1
                doc_row["fields"][sem] = {"gt": gt_norm, "pred": None, "correct": False}
                continue

            pred_norm = _extract_value(pred_text, sem)
            ok = pred_norm is not None and gt_norm is not None and pred_norm == gt_norm
            if ok:
                per_field[sem]["correct"] += 1
            doc_row["fields"][sem] = {
                "gt": gt_norm,
                "pred": pred_norm,
                "text": pred_text,
                "correct": ok,
            }
        if doc_row["fields"]:
            per_doc_rows.append(doc_row)

    summary: Dict[str, Any] = {
        "labels_csv": str(labels_csv.resolve()),
        "json_dir": str(json_dir.resolve()),
        "n_invoices_evaluated": len(per_doc_rows),
        "per_field": {},
        "document_level_all_fields_correct": 0,
    }

    all_correct_docs = 0
    for doc_row in per_doc_rows:
        if doc_row["fields"] and all(v["correct"] for v in doc_row["fields"].values()):
            all_correct_docs += 1
    summary["document_level_all_fields_correct"] = all_correct_docs
    if per_doc_rows:
        summary["document_level_accuracy"] = round(all_correct_docs / len(per_doc_rows), 4)

    for sem, stats in per_field.items():
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        summary["per_field"][sem] = {
            **stats,
            "accuracy": round(acc, 4),
        }

    return summary, per_doc_rows


def main() -> int:
    p = argparse.ArgumentParser(description="Field-level exact-match metrics vs JSON ground truth.")
    p.add_argument(
        "--labels-csv",
        type=Path,
        default=PROJECT_DIR / "data" / "labels_synthetic.csv",
    )
    p.add_argument(
        "--json-dir",
        type=Path,
        default=PROJECT_DIR / "synthetic_output" / "batch_100_v2" / "json",
    )
    p.add_argument("--output-json", type=Path, default=PROJECT_DIR / "output" / "field_extraction_metrics.json")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_DIR / "output" / "field_extraction_per_field.csv",
    )
    args = p.parse_args()

    summary, _ = evaluate_fields(args.labels_csv, args.json_dir, FIELD_SEMANTICS)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["semantic", "total", "correct", "missing_pred", "accuracy"])
        for sem, stats in sorted(summary.get("per_field", {}).items()):
            w.writerow([sem, stats["total"], stats["correct"], stats["missing_pred"], stats["accuracy"]])

    print("=== Field extraction metrics (ground-truth labels) ===")
    print(f"Invoices evaluated: {summary['n_invoices_evaluated']}")
    print(f"Document-level all-fields-correct: {summary.get('document_level_all_fields_correct')}")
    for sem, stats in sorted(summary.get("per_field", {}).items()):
        print(f"  {sem:20}  acc={stats['accuracy']:.4f}  ({stats['correct']}/{stats['total']})")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
