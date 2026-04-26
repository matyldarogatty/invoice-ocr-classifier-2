"""CLI for batch synthetic invoice generation."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import traceback
from pathlib import Path

from .data_generator import build_render_helpers, generate_invoice
from .hints import build_display_values, build_invoice_document
from .io_utils import ensure_run_dirs, write_json
from .label_captions import pick_captions
from .manifest import append_manifest, sha256_file
from .paths import (
    DEFAULT_OUT_DIR,
    HTML_SUBDIR,
    JSON_SUBDIR,
    MANIFEST_FILENAME,
    PDF_SUBDIR,
    STATIC_DIR,
)
from .renderer import build_html_context, html_to_pdf, render_invoice_html

SCHEMA_VERSION = 1
TEMPLATES = ("layout_a", "layout_b", "layout_c")


def _parse_template(arg: str) -> str | None:
    if arg == "any":
        return None
    if arg in TEMPLATES:
        return arg
    raise argparse.ArgumentTypeError(
        f"template must be one of: any, {', '.join(TEMPLATES)}"
    )


def _parse_locale(arg: str) -> str:
    if arg in ("pl", "en", "mixed"):
        return arg
    raise argparse.ArgumentTypeError("label-locale must be pl, en, or mixed (en/mixed are deprecated)")


def _parse_currency_mode(arg: str) -> str:
    if arg in ("mixed", "PLN", "EUR", "USD"):
        return arg
    raise argparse.ArgumentTypeError("currency-mode must be mixed, PLN, EUR, or USD")


def _output_dir_has_generated_files(out_dir: Path) -> bool:
    """True if a previous run likely wrote PDFs and/or a non-empty manifest."""
    manifest = out_dir / MANIFEST_FILENAME
    if manifest.is_file() and manifest.stat().st_size > 0:
        return True
    pdfs = out_dir / PDF_SUBDIR
    if pdfs.is_dir() and any(p.is_file() for p in pdfs.iterdir()):
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic invoice PDFs and ground truth.")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--template", type=_parse_template, default="any")
    p.add_argument("--label-locale", type=_parse_locale, default="pl")
    p.add_argument("--items-min", type=int, default=1)
    p.add_argument("--items-max", type=int, default=8)
    p.add_argument("--currency-mode", type=_parse_currency_mode, default="mixed")
    p.add_argument("--batch-id", type=str, default="")
    p.add_argument("--keep-html", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow using an --out-dir that already contains a manifest and/or PDFs (otherwise use a new directory).",
    )

    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    log = logging.getLogger("synthetic_invoice_generator")

    if args.label_locale in ("en", "mixed"):
        log.warning(
            "--label-locale=%s is deprecated; captions and metadata are Polish-only (pl).",
            args.label_locale,
        )

    out_dir: Path = args.out_dir.resolve()
    if not args.overwrite and _output_dir_has_generated_files(out_dir):
        log.error(
            "Output directory %s already contains generated files (manifest and/or PDFs). "
            "Use a new --out-dir for a clean batch, or pass --overwrite to continue here "
            "(manifest lines append; PDFs/JSON with the same invoice_id may be replaced).",
            out_dir,
        )
        return 2
    ensure_run_dirs(out_dir)
    if args.keep_html:
        (out_dir / HTML_SUBDIR).mkdir(parents=True, exist_ok=True)

    batch_id = args.batch_id.strip() or f"batch_{args.seed}"
    rng = random.Random(args.seed)

    manifest_path = out_dir / MANIFEST_FILENAME
    log.info("Writing outputs under %s", out_dir)
    log.info("batch_id=%s seed=%s count=%s", batch_id, args.seed, args.count)

    ok = 0
    for i in range(args.count):
        template_id = args.template or rng.choice(TEMPLATES)
        try:
            inv = generate_invoice(
                rng,
                invoice_index=i,
                batch_id=batch_id,
                seed=args.seed,
                template_id=template_id,
                label_locale=args.label_locale,
                items_min=args.items_min,
                items_max=args.items_max,
                currency_mode=args.currency_mode,
            )
            captions = pick_captions(rng)
            display = build_display_values(inv, rng)
            helpers = build_render_helpers(rng)
            ctx = build_html_context(inv, captions, helpers, display)
            html = render_invoice_html(template_id, ctx)

            pdf_rel = f"{PDF_SUBDIR}/{inv.meta.invoice_id}.pdf"
            json_rel = f"{JSON_SUBDIR}/{inv.meta.invoice_id}.json"
            pdf_path = out_dir / pdf_rel
            json_path = out_dir / json_rel

            if args.keep_html:
                html_path = out_dir / HTML_SUBDIR / f"{inv.meta.invoice_id}.html"
                html_path.write_text(html, encoding="utf-8")

            if not args.dry_run:
                html_to_pdf(html, pdf_path, base_url=STATIC_DIR)
                pdf_hash = sha256_file(pdf_path)
            else:
                pdf_hash = ""

            doc = build_invoice_document(
                schema_version=SCHEMA_VERSION,
                invoice=inv,
                captions=captions,
                display=display,
                pdf_rel=pdf_rel,
            )
            write_json(json_path, doc)

            append_manifest(
                manifest_path,
                {
                    "invoice_id": inv.meta.invoice_id,
                    "template_id": template_id,
                    "pdf_rel": pdf_rel,
                    "json_rel": json_rel,
                    "seed": args.seed,
                    "batch_id": batch_id,
                    "label_locale": "pl",
                    "page_count": 1,
                    "pdf_sha256": pdf_hash,
                },
            )
            ok += 1
            if (i + 1) % max(1, args.count // 10) == 0 or i == args.count - 1:
                log.info("Progress: %s / %s", i + 1, args.count)
        except Exception as e:
            log.error("Failed invoice index %s: %s", i, e)
            if log.isEnabledFor(logging.DEBUG):
                traceback.print_exc()
            if args.fail_fast:
                return 1

    log.info("Done. Successful: %s / %s", ok, args.count)
    return 0 if ok == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
