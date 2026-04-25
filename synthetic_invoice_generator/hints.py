"""Classification hints and display strings for ground truth."""

from __future__ import annotations

import random
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List

from .data_generator import _format_pl_amount, _nip_canonical
from .models import InvoiceRecord, invoice_record_to_dict
from .semantic_labels import GENERATOR_VERSION, SEMANTIC_TO_LABEL_ID


def _canon_amount(d: Decimal) -> str:
    return f"{d:.2f}"


def build_display_values(invoice: InvoiceRecord, rng: random.Random) -> Dict[str, str]:
    """Rendered strings as they should appear on the PDF (best effort)."""
    pln_suffix = rng.random() < 0.25 and invoice.totals.currency == "PLN"
    cur_display = "zł" if pln_suffix else invoice.totals.currency

    def rd(dt: date) -> str:
        if rng.random() < 0.65:
            return dt.strftime("%d.%m.%Y")
        return dt.isoformat()

    return {
        "invoice_number": invoice.invoice_number,
        "invoice_date": rd(invoice.invoice_date),
        "sale_date": rd(invoice.sale_date),
        "payment_due": rd(invoice.payment_due_date),
        "seller_name": invoice.seller.name,
        "buyer_name": invoice.buyer.name,
        "seller_nip": invoice.seller.nip,
        "buyer_nip": invoice.buyer.nip,
        "net": _format_pl_amount(invoice.totals.net),
        "vat": _format_pl_amount(invoice.totals.vat),
        "gross": _format_pl_amount(invoice.totals.gross),
        "currency": cur_display,
    }


def build_classification_hints(
    invoice: InvoiceRecord,
    display: Dict[str, str],
) -> List[Dict[str, Any]]:
    """One hint per main semantic field (no line-item class labels)."""
    hints: List[Dict[str, Any]] = []

    def add(semantic: str, canonical: str, rendered: str) -> None:
        hints.append(
            {
                "semantic": semantic,
                "label_id": SEMANTIC_TO_LABEL_ID[semantic],
                "canonical_value": canonical,
                "rendered_value": rendered,
            }
        )

    add("INVOICE_NUMBER", invoice.invoice_number.strip(), display["invoice_number"])
    add("SELLER_NAME", invoice.seller.name.strip(), display["seller_name"])
    add("SELLER_NIP", _nip_canonical(invoice.seller.nip), display["seller_nip"])
    add("BUYER_NAME", invoice.buyer.name.strip(), display["buyer_name"])
    add("BUYER_NIP", _nip_canonical(invoice.buyer.nip), display["buyer_nip"])
    add("INVOICE_DATE", invoice.invoice_date.isoformat(), display["invoice_date"])
    add("SALE_DATE", invoice.sale_date.isoformat(), display["sale_date"])
    add("NET_AMOUNT", _canon_amount(invoice.totals.net), display["net"])
    add("VAT_AMOUNT", _canon_amount(invoice.totals.vat), display["vat"])
    add("TOTAL_PRICE", _canon_amount(invoice.totals.gross), display["gross"])
    add("CURRENCY", invoice.totals.currency, display["currency"])

    return hints


def build_line_items_export(invoice: InvoiceRecord) -> List[Dict[str, Any]]:
    """Structured line rows with rendered cell strings for JSON only."""
    rows = []
    for it in invoice.items:
        rows.append(
            {
                "ordinal": it.ordinal,
                "name": it.name,
                "quantity": str(it.quantity),
                "unit": it.unit,
                "vat_rate_percent": it.vat_rate_percent,
                "unit_net_canonical": _canon_amount(it.unit_net),
                "unit_net_rendered": _format_pl_amount(it.unit_net),
                "line_net_canonical": _canon_amount(it.line_net),
                "line_net_rendered": _format_pl_amount(it.line_net),
                "line_vat_canonical": _canon_amount(it.line_vat),
                "line_vat_rendered": _format_pl_amount(it.line_vat),
                "line_gross_canonical": _canon_amount(it.line_gross),
                "line_gross_rendered": _format_pl_amount(it.line_gross),
            }
        )
    return rows


def build_invoice_document(
    *,
    schema_version: int,
    invoice: InvoiceRecord,
    captions: Dict[str, str],
    display: Dict[str, str],
    pdf_rel: str,
) -> Dict[str, Any]:
    render_block = {
        "invoice_number": captions.get("invoice_number", ""),
        "invoice_date": captions.get("invoice_date", ""),
        "sale_date": captions.get("sale_date", ""),
        "seller_nip": captions.get("seller_nip", ""),
        "buyer_nip": captions.get("buyer_nip", ""),
        "net": captions.get("net", ""),
        "vat": captions.get("vat", ""),
        "gross": captions.get("gross", ""),
        "currency": captions.get("currency", ""),
    }

    return {
        "schema_version": schema_version,
        "invoice_id": invoice.meta.invoice_id,
        "pdf_rel": pdf_rel,
        "meta": {
            "template_id": invoice.meta.template_id,
            "seed": invoice.meta.seed,
            "batch_id": invoice.meta.batch_id,
            "label_locale": invoice.meta.label_locale,
            "generator_version": GENERATOR_VERSION,
        },
        "invoice": invoice_record_to_dict(invoice),
        "render": render_block,
        "classification_hints": build_classification_hints(invoice, display),
        "line_items_detail": build_line_items_export(invoice),
    }
