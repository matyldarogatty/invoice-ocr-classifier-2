#!/usr/bin/env python3
"""
Export synthetic invoice PDFs + JSON ground truth to OCR training crops and CSVs.

Safe defaults: never writes to data/images/ or data/labels.csv.
See README or run with --help for paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow `python src/export_synthetic_to_labels.py` from repo root
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import LABELS  # noqa: E402

from doctr.io import DocumentFile  # noqa: E402
from doctr.models import ocr_predictor  # noqa: E402
from PIL import Image  # noqa: E402

from match_utils import MatchDecision, build_caption_sets, match_ocr_line  # noqa: E402

PROJECT_ROOT = _SRC.parent

SEMANTIC_TO_LABEL_ID: Dict[str, int] = {name: i for i, name in LABELS.items()}
OTHER_ID = SEMANTIC_TO_LABEL_ID["OTHER"]


def get_line_bbox_pixels(line, page) -> Tuple[int, int, int, int]:
    xs, ys = [], []
    for word in line.words:
        (x1, y1), (x2, y2) = word.geometry
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    h, w = page.dimensions
    x1 = int(min(xs) * w)
    x2 = int(max(xs) * w)
    y1 = int(min(ys) * h)
    y2 = int(max(ys) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _parse_manifest_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def _load_hints(data: Dict[str, Any], log: logging.Logger) -> List[Dict[str, Any]]:
    raw = data.get("classification_hints") or []
    out: List[Dict[str, Any]] = []
    for h in raw:
        if not isinstance(h, dict):
            continue
        sem = h.get("semantic")
        if sem not in SEMANTIC_TO_LABEL_ID:
            log.warning("Skipping unknown semantic %r (not in config.LABELS)", sem)
            continue
        if sem == "OTHER":
            continue
        out.append(h)
    return out


def _semantic_to_label(semantic: Optional[str]) -> Tuple[int, str]:
    if not semantic:
        return OTHER_ID, "OTHER"
    lid = SEMANTIC_TO_LABEL_ID.get(semantic)
    if lid is None:
        return OTHER_ID, "OTHER"
    return lid, semantic


def run_export(args: argparse.Namespace) -> int:
    log = logging.getLogger("export_synthetic")

    manifest_path = Path(args.manifest).expanduser()
    pdf_dir = Path(args.pdf_dir).expanduser()
    json_dir = Path(args.json_dir).expanduser()
    images_dir = Path(args.images_dir).expanduser()
    csv_path = Path(args.csv_path).expanduser()
    review_csv_path = Path(args.review_csv_path).expanduser()
    summary_json_path: Optional[Path] = None
    if args.summary_json_path:
        summary_json_path = Path(args.summary_json_path).expanduser()

    for p in (csv_path, review_csv_path):
        if p.resolve() == (PROJECT_ROOT / "data" / "labels.csv").resolve():
            log.error("Refusing to write to data/labels.csv (use labels_synthetic.csv).")
            return 2

    if images_dir.resolve() == (PROJECT_ROOT / "data" / "images").resolve():
        log.error("Refusing to write to data/images/ by default. Use data/images_synthetic/ or another folder.")
        return 2

    if not manifest_path.is_file():
        log.error("Manifest not found: %s", manifest_path)
        return 1

    images_dir.mkdir(parents=True, exist_ok=True)

    lines_manifest: List[str] = manifest_path.read_text(encoding="utf-8").splitlines()
    rows_out: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []

    stats = {
        "invoices_processed": 0,
        "invoices_skipped_no_hints": 0,
        "ocr_lines_total": 0,
        "labeled_non_other": 0,
        "labeled_other": 0,
        "ambiguous_matches": 0,
        "per_class_counts": Counter(),
        "missing_pdf": 0,
        "missing_json": 0,
        "ocr_errors": 0,
    }

    ocr_model = None

    def get_ocr():
        nonlocal ocr_model
        if ocr_model is None:
            log.info("Loading docTR OCR model (first use)...")
            ocr_model = ocr_predictor(pretrained=True)
        return ocr_model

    limit = args.limit or None
    processed = 0

    for raw_line in lines_manifest:
        if limit is not None and processed >= limit:
            break
        rec = _parse_manifest_line(raw_line)
        if rec is None:
            continue
        invoice_id = rec.get("invoice_id")
        if not invoice_id:
            log.warning("Manifest row without invoice_id, skipping")
            continue

        pdf_rel = rec.get("pdf_rel") or ""
        json_rel = rec.get("json_rel") or ""
        pdf_name = Path(pdf_rel).name if pdf_rel else f"{invoice_id}.pdf"
        json_name = Path(json_rel).name if json_rel else f"{invoice_id}.json"
        pdf_path = pdf_dir / pdf_name
        json_path = json_dir / json_name

        if not pdf_path.is_file():
            log.warning("Missing PDF: %s", pdf_path)
            stats["missing_pdf"] += 1
            if args.fail_fast:
                return 1
            continue
        if not json_path.is_file():
            log.warning("Missing JSON: %s", json_path)
            stats["missing_json"] += 1
            if args.fail_fast:
                return 1
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.error("Bad JSON %s: %s", json_path, e)
            if args.fail_fast:
                return 1
            continue

        hints = _load_hints(data, log)
        render_block = data.get("render") if isinstance(data.get("render"), dict) else {}
        caption_norms, caption_prefixes = build_caption_sets(render_block)

        if not hints:
            log.warning("No classification_hints for %s, skipping invoice", invoice_id)
            stats["invoices_skipped_no_hints"] += 1
            if args.fail_fast:
                return 1
            continue

        try:
            model = get_ocr()
            doc = DocumentFile.from_pdf(str(pdf_path))
            result = model(doc)
            page = result.pages[0]
            page_image = Image.fromarray(doc[0])
        except Exception as e:
            log.error("OCR failed for %s: %s", pdf_path, e)
            stats["ocr_errors"] += 1
            if args.fail_fast:
                return 1
            continue

        consumed: Set[str] = set()
        line_idx = 0
        for block in page.blocks:
            for line in block.lines:
                text = " ".join(w.value for w in line.words)
                x1, y1, x2, y2 = get_line_bbox_pixels(line, page)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                crop = page_image.crop((x1, y1, x2, y2))
                filename = f"{invoice_id}_{line_idx:04d}.png"
                crop.save(images_dir / filename)

                decision = match_ocr_line(text, hints, consumed, caption_norms, caption_prefixes)
                label_id, sem_name = _semantic_to_label(decision.semantic)
                if decision.semantic:
                    consumed.add(decision.semantic)
                    stats["labeled_non_other"] += 1
                else:
                    stats["labeled_other"] += 1
                    if decision.reason.startswith("ambiguous"):
                        stats["ambiguous_matches"] += 1

                stats["per_class_counts"][LABELS[label_id]] += 1
                stats["ocr_lines_total"] += 1

                rows_out.append(
                    {
                        "filename": filename,
                        "text": text,
                        "label": label_id,
                        "invoice_id": invoice_id,
                        "semantic_name": sem_name,
                    }
                )
                review_rows.append(
                    {
                        "invoice_id": invoice_id,
                        "filename": filename,
                        "text": text,
                        "label": label_id,
                        "semantic_name": sem_name,
                        "matched_value": decision.matched_value,
                        "match_type": decision.match_type,
                        "reason": decision.reason,
                    }
                )
                line_idx += 1

        stats["invoices_processed"] += 1
        processed += 1
        log.info("Exported invoice %s (%s OCR lines)", invoice_id, line_idx)

    # Write main CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filename", "text", "label", "invoice_id", "semantic_name"],
        )
        w.writeheader()
        w.writerows(rows_out)

    review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    review_write = review_rows
    if args.review_sample_size and args.review_sample_size > 0:
        rng = random.Random(args.seed)
        k = min(args.review_sample_size, len(review_rows))
        review_write = rng.sample(review_rows, k) if k < len(review_rows) else list(review_rows)
        sample_path = review_csv_path.with_name(
            review_csv_path.stem + "_sample" + review_csv_path.suffix
        )
        with sample_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "invoice_id",
                    "filename",
                    "text",
                    "label",
                    "semantic_name",
                    "matched_value",
                    "match_type",
                    "reason",
                ],
            )
            w.writeheader()
            w.writerows(review_write)
        log.info("Wrote review sample (%s rows) to %s", len(review_write), sample_path)
        review_write = review_rows

    with review_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "invoice_id",
                "filename",
                "text",
                "label",
                "semantic_name",
                "matched_value",
                "match_type",
                "reason",
            ],
        )
        w.writeheader()
        w.writerows(review_write)

    total_lines = max(1, stats["ocr_lines_total"])
    match_rate = stats["labeled_non_other"] / total_lines
    summary = {
        **{k: v for k, v in stats.items() if k != "per_class_counts"},
        "per_class_counts": dict(stats["per_class_counts"]),
        "match_rate_non_other": round(match_rate, 4),
    }

    log.info("=== Export summary ===")
    log.info("Invoices processed: %s", stats["invoices_processed"])
    log.info("OCR lines total: %s", stats["ocr_lines_total"])
    log.info("Non-OTHER labels: %s", stats["labeled_non_other"])
    log.info("OTHER labels: %s", stats["labeled_other"])
    log.info("Ambiguous (tie) -> OTHER: %s", stats["ambiguous_matches"])
    log.info("Missing PDFs: %s  Missing JSON: %s", stats["missing_pdf"], stats["missing_json"])
    log.info("OCR errors: %s", stats["ocr_errors"])
    log.info("Match rate (non-OTHER / all lines): %.2f%%", 100.0 * match_rate)
    log.info("Per-class counts: %s", dict(stats["per_class_counts"]))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary_json_path:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Wrote summary JSON to %s", summary_json_path)

    log.info("Main CSV: %s", csv_path)
    log.info("Review CSV: %s", review_csv_path)
    log.info("Images: %s", images_dir)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Export synthetic PDFs + JSON to labels + crops.")
    p.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "synthetic_invoice_generator" / "out" / "manifest.jsonl"),
    )
    p.add_argument(
        "--pdf-dir",
        type=str,
        default=str(PROJECT_ROOT / "synthetic_invoice_generator" / "out" / "pdfs"),
    )
    p.add_argument(
        "--json-dir",
        type=str,
        default=str(PROJECT_ROOT / "synthetic_invoice_generator" / "out" / "json"),
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "images_synthetic"),
    )
    p.add_argument(
        "--csv-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "labels_synthetic.csv"),
    )
    p.add_argument(
        "--review-csv-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "labels_synthetic_review.csv"),
    )
    p.add_argument(
        "--summary-json-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "labels_synthetic_summary.json"),
        help="Write machine-readable summary JSON to this path",
    )
    p.add_argument(
        "--no-summary-json",
        action="store_true",
        help="Do not write summary JSON (stdout summary still printed)",
    )
    p.add_argument("--limit", type=int, default=0, help="Max invoices to process (0 = all)")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument(
        "--review-sample-size",
        type=int,
        default=0,
        help="If > 0, also write labels_synthetic_review_sample.csv with N random rows",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for review sample")
    args = p.parse_args()
    if args.limit == 0:
        args.limit = None
    if args.no_summary_json:
        args.summary_json_path = None

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    return run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
